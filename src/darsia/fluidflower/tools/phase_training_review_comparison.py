"""Compare immutable saved phase maps on newly reviewed regions and late frames.

This evaluates training evidence, not independent generalization. No inference,
training, publication, model changes or live review edits are performed.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


from darsia.fluidflower.tools.phase_training_geometry_revision import read, sha, _native
from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
from darsia.fluidflower.phase_training.review_regions import annotation_mask


FORMAT = "saved-phase-training-evidence-comparison-v1"
LABELS = {"water": 0, "aqueous": 1, "gas": 2, "mixed": 3}
GAS_ABSENCE = {"gas_absent_region", "aqueous_present_gas_absent_region", "water", "aqueous", "co2_absent_region", "current_experiment_co2_artifact"}
AQUEOUS_PRESENCE = {"aqueous_present_region", "aqueous_present_gas_absent_region", "aqueous", "mixed_gas_aqueous", "mixed_aqueous_water", "mixed_gas_aqueous_water"}
AQUEOUS_ABSENCE = {"aqueous_absent_region", "water", "gas", "co2_absent_region", "current_experiment_co2_artifact"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def decode_phase(path, content_box, hashes=None):
    """Decode exact palette values; transparent means no visible CO2 label."""
    path = Path(path)
    raw = path.read_bytes()
    if hashes is not None:
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    x, y, width, height = map(int, content_box)
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode != "RGBA" or image.width < x+width or image.height < y+height:
            raise ValueError(f"Not an exact native categorical RGBA phase map: {path}")
        rgba = np.asarray(image.crop((x, y, x+width, y+height)))
    labels = np.zeros((height, width), dtype=np.uint8)
    known = rgba[..., 3] == 0
    for name in ("aqueous", "gas", "mixed"):
        selected = np.all(rgba == PHASE_FILL_COLORS[name], axis=-1)
        labels[selected] = LABELS[name]
        known |= selected
    if not known.all():
        raise ValueError(f"Unknown phase palette values in {path}")
    return labels


def fractions(labels, mask):
    size = int(np.count_nonzero(mask))
    histogram = np.bincount(labels[mask], minlength=4)
    counts = {name: int(histogram[value]) for name, value in LABELS.items()}
    return dict(pixels=size, counts=counts,
        fractions={name: count/size if size else None for name, count in counts.items()},
        gas_evidence_pixels=counts["gas"]+counts["mixed"],
        gas_evidence_fraction=(counts["gas"]+counts["mixed"])/size if size else None,
        aqueous_evidence_pixels=counts["aqueous"]+counts["mixed"],
        aqueous_evidence_fraction=(counts["aqueous"]+counts["mixed"])/size if size else None,
        co2_pixels=size-counts["water"], co2_fraction=(size-counts["water"])/size if size else None)


def assess_region(old, new, phase_class, positive_fraction=.10):
    if old["pixels"] != new["pixels"] or not new["pixels"]:
        raise ValueError("Comparisons require a nonempty identical observed region")
    result = dict(gas_evidence_change_pp=100*(new["gas_evidence_fraction"]-old["gas_evidence_fraction"]),
        aqueous_evidence_change_pp=100*(new["aqueous_evidence_fraction"]-old["aqueous_evidence_fraction"]),
        flags=[], interpretations=[])
    if phase_class in GAS_ABSENCE:
        result.update(gas_absence_old_satisfied=old["gas_evidence_pixels"] == 0,
                      gas_absence_new_satisfied=new["gas_evidence_pixels"] == 0)
        if new["gas_evidence_pixels"]:
            result["flags"].append("residual_gas_in_explicit_gas_absence")
        if new["gas_evidence_pixels"] > old["gas_evidence_pixels"]:
            result["flags"].append("gas_absence_regression")
        result["interpretations"].append("Gas absence does not choose between aqueous CO2 and water.")
    if phase_class in AQUEOUS_PRESENCE:
        result.update(aqueous_old_any=old["aqueous_evidence_pixels"] > 0,
            aqueous_new_any=new["aqueous_evidence_pixels"] > 0,
            aqueous_coverage_reference_fraction=float(positive_fraction),
            aqueous_old_at_reference=old["aqueous_evidence_fraction"] >= positive_fraction,
            aqueous_new_at_reference=new["aqueous_evidence_fraction"] >= positive_fraction,
            mil_probability_loss_evaluated=False)
        if not new["aqueous_evidence_pixels"]:
            result["flags"].append("no_visible_aqueous_in_positive_region")
        if old["aqueous_evidence_pixels"] and not new["aqueous_evidence_pixels"]:
            result["flags"].append("aqueous_presence_regression")
        if new["aqueous_evidence_fraction"] < positive_fraction:
            result["flags"].append("aqueous_coverage_below_top_fraction_reference")
        result["interpretations"].append("Regional presence requires aqueous CO2 somewhere, not that the whole box be aqueous. Categorical coverage at the MIL top-fraction reference is descriptive; it does not reconstruct the probability-based training loss.")
    if phase_class in AQUEOUS_ABSENCE:
        result.update(aqueous_absence_old_satisfied=old["aqueous_evidence_pixels"] == 0,
                      aqueous_absence_new_satisfied=new["aqueous_evidence_pixels"] == 0)
        if new["aqueous_evidence_pixels"]:
            result["flags"].append("residual_aqueous_in_explicit_aqueous_absence")
        if new["aqueous_evidence_pixels"] > old["aqueous_evidence_pixels"]:
            result["flags"].append("aqueous_absence_regression")
        result["interpretations"].append("Aqueous absence alone does not choose between free gas and water.")
    def diagnostic_error(metrics):
        return ((metrics["gas_evidence_fraction"] if phase_class in GAS_ABSENCE else 0.)
            + (metrics["aqueous_evidence_fraction"] if phase_class in AQUEOUS_ABSENCE else 0.)
            + (max(0., positive_fraction-metrics["aqueous_evidence_fraction"])/positive_fraction
               if phase_class in AQUEOUS_PRESENCE else 0.))
    result["label_consistent_change_score"] = diagnostic_error(old)-diagnostic_error(new)
    result["change_score_scope"] = "Only for deterministic figure selection: reduction in excluded-phase fraction or increase toward aqueous coverage reference; not an independent accuracy score."
    return result


def fragmentation(labels, domain):
    co2 = (labels != 0) & domain
    area = int(co2.sum())
    count, _, stats, _ = cv2.connectedComponentsWithStats(co2.astype(np.uint8), connectivity=8)
    small = stats[1:, cv2.CC_STAT_AREA] < 16
    perimeter = int(np.count_nonzero(co2 & ~cv2.erode(co2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)))
    return dict(co2_components=max(0, count-1), components_under_16_native_pixels=int(small.sum()),
        pixels_in_components_under_16=int(stats[1:, cv2.CC_STAT_AREA][small].sum()),
        co2_boundary_pixels=perimeter, co2_boundary_fraction=perimeter/area if area else 0.,
        interpretation="Descriptive topology at native resolution. Real fingers and separate plumes also change perimeter/components; this is not an accuracy score.")


def phase_path(job, index):
    return Path(job["phase_map_output_dir"]) / "native_phase_maps" / job["run"] / ("A_"+job["params_sha256"][:16]) / f"frame_{index:03d}.png"


def _regions(root):
    from darsia.fluidflower.phase_training import trainer as t
    request = read(root / "training_request.json")
    bundles = []
    for bundle in request["bundles"]:
        pair = []
        for name in ("decisions", "combined_manifest"):
            entry = bundle[name]
            path = Path(entry["snapshot_path"])
            if sha(path) != entry["sha256"]:
                raise ValueError("Frozen comparison training evidence changed")
            pair.append(path)
        bundles.append(t.ReviewBundle(*pair))
    _, regions, _ = t._load_sources_and_regions(bundles)
    return regions, request


def _completed(root):
    manifest, binding = root / "validation/postflight_manifest.json", root / "validation/validation_binding.json"
    if not manifest.is_file() or not binding.is_file():
        return False
    return bool(read(binding).get("completed_utc"))


def mil_fraction_reference(model):
    """Read the fitted model's recorded top-k fraction, never infer from maps."""
    path = Path(model) / "models/generic/co2_presence_signal.json"
    found = set()
    def visit(value):
        if isinstance(value, dict):
            if "positive_top_fraction" in value:
                found.add(float(value["positive_top_fraction"]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(read(path))
    if len(found) != 1 or not 0 < next(iter(found)) <= 1:
        raise ValueError("Fitted MIL model does not specify one unambiguous top-fraction reference")
    return next(iter(found)), dict(path=str(path), sha256=sha(path))


def _figure(path, photo, old, new, *, title, bbox=None, outline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        photo, old, new = photo[y0:y1, x0:x1], old[y0:y1, x0:x1], new[y0:y1, x0:x1]
        if outline is not None:
            outline = outline[y0:y1, x0:x1]
    scale = min(1., 1250/photo.shape[1])
    size = (max(1, round(photo.shape[1]*scale)), max(1, round(photo.shape[0]*scale)))
    rgb = cv2.resize(photo, size, interpolation=cv2.INTER_AREA)
    old, new = [cv2.resize(value, size, interpolation=cv2.INTER_NEAREST) for value in (old, new)]
    if outline is not None:
        outline = cv2.resize(outline.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3.2, min(9., 5*size[1]/size[0]))), layout="constrained")
    for ax, value, label in zip(axes, (None, old, new), ("Fotografi", "Forrige modell", "Ny modell")):
        ax.imshow(rgb)
        if value is not None:
            rgba = np.zeros((*value.shape, 4), dtype=np.float32)
            for name in ("aqueous", "gas", "mixed"):
                selected = value == LABELS[name]
                rgba[selected, :3] = np.asarray(PHASE_FILL_COLORS[name][:3])/255
                rgba[selected, 3] = .50
            ax.imshow(rgba)
        if outline is not None and outline.any() and not outline.all():
            ax.contour(outline, levels=[.5], colors=["white"], linewidths=.65)
        ax.set_title(label, fontsize=10); ax.set_axis_off()
    fig.suptitle(title+"\nGrønn: innløst CO₂ · rosa: fri gass · gul: blandet", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def _photo(path, box, hashes):
    raw = Path(path).read_bytes(); hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    x, y, width, height = map(int, box)
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB").crop((x, y, x+width, y+height)))


def compare_series(campaign_root, series, *, model=None):
    from darsia.fluidflower import phase_runtime
    from darsia.fluidflower.tools.train_phase_model_further import _continuation_geometry_binding
    from darsia_extensions.registration import warp_labels
    campaign_root = Path(campaign_root)
    output = campaign_root / series / "saved_map_comparison.json"
    state = read(campaign_root / series / "state.json")
    parent = Path(state["parent"]).resolve()
    model = Path(model or state.get("selected_model_path") or state["output_root"]).resolve()
    result = dict(format=FORMAT, series=series, status="pending", parent=str(parent), model=str(model),
        figures=[], regions=[], late_frames=[], summary=[], independent_validation=False,
        scope="Training-evidence diagnostics and descriptive late-frame comparison; all new boxes were used for fitting.")
    if not _completed(model):
        result["summary"] = ["Venter på ferdige fasekart fra den nye modellen."]
        write(output, result); return result
    if not _completed(parent):
        raise ValueError("Parent validation is not complete")
    old_manifest, new_manifest = [read(root / "validation/postflight_manifest.json") for root in (parent, model)]
    for root, manifest in ((parent, old_manifest), (model, new_manifest)):
        binding = read(root / "validation/validation_binding.json")
        if any(binding.get(key) != manifest.get(key) for key in ("model_id", "model_sha256", "physics_runtime")):
            raise ValueError("Saved prediction identity differs between manifest and binding")
    result.update(parent_model_id=old_manifest["model_id"], model_id=new_manifest["model_id"])
    old_regions, _ = _regions(parent)
    all_regions, request = _regions(model)
    old_by_id = {region.region_id: region for region in old_regions}
    selected = [region for region in all_regions if region.region_id not in old_by_id
                or region.phase_class != old_by_id[region.region_id].phase_class]
    positive_fraction, fraction_source = mil_fraction_reference(model)
    result["mil_coverage_reference"] = dict(fraction=positive_fraction, source=fraction_source,
        interpretation="Coverage reference from fitted MIL top-k setting; saved categorical maps cannot evaluate the probability-based MIL loss.")
    selected_by_frame = defaultdict(list)
    for region in selected:
        selected_by_frame[(region.run, region.frame_index)].append(region)
    result["selected_evidence"] = dict(total=len(selected), classes=dict(Counter(region.phase_class for region in selected)),
        selection="New effective region IDs or changed phase class relative to parent; geometry migration alone is not selected.")
    old_jobs, new_jobs = [{job["run"]: job for job in manifest["jobs"]} for manifest in (old_manifest, new_manifest)]
    hashes, frame_records = {}, {}
    profile = phase_runtime.load_physics_profile(model)
    metadata = {}
    for run, new_job in new_jobs.items():
        old_job = old_jobs[run]
        old_root, old_path = _native(old_job); new_root, new_path = _native(new_job)
        old_native, new_native = read(old_path), read(new_path)
        hashes.update({str(path): sha(path) for path in (old_path, new_path)})
        geometry = _continuation_geometry_binding(old_path, old_native, new_path, new_native)
        context, _ = phase_runtime.load_run_context(model, profile, series, run, native_manifest_sha256=sha(new_path))
        domain = np.asarray(context.domain, dtype=bool)
        late = set(range(max(0, int(new_native["frame_count"])-3), int(new_native["frame_count"])))
        wanted = sorted(late | {index for r, index in selected_by_frame if r == run})
        metadata[run] = (old_job, new_job, old_native, new_native, new_root, geometry, domain)
        for index in wanted:
            old_frame, new_frame = old_native["frame_sources"][index], new_native["frame_sources"][index]
            if old_frame["source_name"] != new_frame["source_name"] or abs(float(old_frame["elapsed_h"])-float(new_frame["elapsed_h"])) > 1e-7:
                raise ValueError("Compared saved frames are not the same source image/time")
            old_labels = decode_phase(phase_path(old_job, index), old_native["content_box_xywh"], hashes)
            new_labels = decode_phase(phase_path(new_job, index), new_native["content_box_xywh"], hashes)
            valid = np.ones(new_labels.shape, dtype=bool)
            transport = None
            if geometry is not None:
                entry = geometry["bindings"][(run, index)]
                receipt = entry["binding"]["receipt"]
                old_labels = warp_labels(old_labels, receipt)
                left, top, right, bottom = receipt["valid_bounds_xyxy"]
                valid[:] = False; valid[top:bottom, left:right] = True
                common = new_native["correction_contract"]["rig_registration"]["valid_bounds_xyxy"]
                common_mask = np.zeros_like(valid)
                left, top, right, bottom = common; common_mask[top:bottom, left:right] = True
                valid &= common_mask
                transport = dict(geometry_revision_id=geometry["geometry_revision_id"],
                    shift_xy_native_px=receipt["shift_xy_native_px"], interpolation="nearest categorical transport",
                    comparison="Old predictions transported; new predictions freshly inferred from registered photographs.")
            if old_labels.shape != new_labels.shape or domain.shape != new_labels.shape:
                raise ValueError("Compared native labels and physical domain have different shapes")
            frame_records[(run, index)] = dict(run=run, frame_index=index, elapsed_h=float(new_frame["elapsed_h"]),
                old_map=str(phase_path(old_job, index)), new_map=str(phase_path(new_job, index)), old_geometry_transport=transport)
            for region in selected_by_frame[(run, index)]:
                mask = annotation_mask(region.payload, canvas_wh=new_native["canvas_wh"], content_box=new_native["content_box_xywh"])
                all_pixels = int(mask.sum()); mask &= valid
                if not mask.any():
                    raise ValueError(f"New region has no observed pixels: {region.region_id}")
                old_metrics, new_metrics = fractions(old_labels, mask), fractions(new_labels, mask)
                row = dict(**frame_records[(run, index)], region_id=region.region_id, phase_class=region.phase_class,
                    old=old_metrics, new=new_metrics, excluded_unobserved_pixels=all_pixels-int(mask.sum()),
                    old_in_physical_domain=fractions(old_labels, mask & domain),
                    new_in_physical_domain=fractions(new_labels, mask & domain),
                    assessment=assess_region(old_metrics, new_metrics, region.phase_class, positive_fraction))
                result["regions"].append(row)
            if index in late:
                row = dict(**frame_records[(run, index)], old=fractions(old_labels, domain & valid),
                    new=fractions(new_labels, domain & valid), old_fragmentation=fragmentation(old_labels, domain & valid),
                    new_fragmentation=fragmentation(new_labels, domain & valid))
                photo = _photo(new_root / "native_frames" / run / f"frame_{index:03d}.jpg", new_native["content_box_xywh"], hashes)
                path = campaign_root / series / "comparison_figures" / f"{run}_late_{index:03d}.png"
                _figure(path, photo, old_labels, new_labels, title=f"{series.upper()} · {run.upper()} · {row['elapsed_h']:.2f} t")
                figure = dict(path=str(path), caption="Sammenligning av lagrede fasekart i samme koordinatsystem.",
                    run=run, frame_index=index, elapsed_h=row["elapsed_h"], kind="late_full_frame")
                result["figures"].append(figure); row["figure"] = str(path); result["late_frames"].append(row)
    # Deterministic selection makes both improvements and remaining errors visible.
    chosen = []
    for run in new_jobs:
        rows = [row for row in result["regions"] if row["run"] == run]
        ordering = [sorted(rows, key=lambda row: (-len(row["assessment"]["flags"]), -row["new"]["gas_evidence_fraction"], row["region_id"])),
            sorted(rows, key=lambda row: (-row["assessment"]["label_consistent_change_score"], row["region_id"])),
            sorted(rows, key=lambda row: (row["assessment"]["label_consistent_change_score"], row["region_id"]))]
        seen = set()
        for candidates in ordering:
            for row in candidates[:2]:
                if row["region_id"] not in seen:
                    chosen.append(row); seen.add(row["region_id"])
    selected_regions = {region.region_id: region for region in selected}
    for row in chosen:
        run, index = row["run"], row["frame_index"]
        old_job, new_job, old_native, new_native, new_root, geometry, domain = metadata[run]
        old_labels = decode_phase(phase_path(old_job, index), old_native["content_box_xywh"], hashes)
        new_labels = decode_phase(phase_path(new_job, index), new_native["content_box_xywh"], hashes)
        if geometry is not None:
            old_labels = warp_labels(old_labels, geometry["bindings"][(run, index)]["binding"]["receipt"])
        region = selected_regions[row["region_id"]]
        mask = annotation_mask(region.payload, canvas_wh=new_native["canvas_wh"], content_box=new_native["content_box_xywh"])
        ys, xs = np.nonzero(mask)
        margin = max(16, round(max(xs.max()-xs.min(), ys.max()-ys.min())*.15))
        bbox = (max(0, int(xs.min())-margin), max(0, int(ys.min())-margin),
                min(mask.shape[1], int(xs.max())+margin+1), min(mask.shape[0], int(ys.max())+margin+1))
        photo = _photo(new_root / "native_frames" / run / f"frame_{index:03d}.jpg", new_native["content_box_xywh"], hashes)
        short = region.region_id.split(":")[-1][:12]
        path = campaign_root / series / "comparison_figures" / f"{run}_region_{short}.png"
        _figure(path, photo, old_labels, new_labels, bbox=bbox, outline=mask,
            title=f"{run.upper()} · {row['elapsed_h']:.2f} t · {region.phase_class}\nGass: {100*row['old']['gas_evidence_fraction']:.2f}% → {100*row['new']['gas_evidence_fraction']:.2f}% · innløst: {100*row['old']['aqueous_evidence_fraction']:.2f}% → {100*row['new']['aqueous_evidence_fraction']:.2f}%")
        row["figure"] = str(path)
        result["figures"].append(dict(path=str(path), caption=f"Ny treningsmarkering {short}; hvit linje viser boksen.",
            run=run, frame_index=index, elapsed_h=row["elapsed_h"], region_id=region.region_id, kind="selected_training_region"))
    flags = Counter(flag for row in result["regions"] for flag in row["assessment"]["flags"])
    negative = [row for row in result["regions"] if row["phase_class"] in GAS_ABSENCE]
    positive = [row for row in result["regions"] if row["phase_class"] in AQUEOUS_PRESENCE]
    aqueous_negative = [row for row in result["regions"] if row["phase_class"] in AQUEOUS_ABSENCE]
    result["summary"] = [f"{len(selected)} nye/endrede treningsmarkeringer sammenlignet; dette er kontroll mot treningsdata.",
        f"{sum(row['new']['gas_evidence_pixels'] == 0 for row in negative)}/{len(negative)} gasseksklusjoner er helt uten gass i det nye kartet (før: {sum(row['old']['gas_evidence_pixels'] == 0 for row in negative)}).",
        f"{sum(row['new']['aqueous_evidence_pixels'] > 0 for row in positive)}/{len(positive)} positive markeringer inneholder synlig innløst CO₂ (før: {sum(row['old']['aqueous_evidence_pixels'] > 0 for row in positive)}).",
        f"{sum(row['new']['aqueous_evidence_pixels'] == 0 for row in aqueous_negative)}/{len(aqueous_negative)} eksklusjoner av innløst CO₂ er uten denne fasen i det nye kartet (før: {sum(row['old']['aqueous_evidence_pixels'] == 0 for row in aqueous_negative)}).",
        "Kantmålene beskriver fragmentering; de skiller ikke alene mellom reelle fingre og feilklassifisert støy."]
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), flags=dict(flags),
        source_hashes=hashes, source_files_modified=False, image_selection="Per run: up to two most flagged/residual, two largest improvements, two largest regressions; all last three frames.")
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=local_cache_root() / 'phase_diagnostics/all_series_review_update_20260912')
    parser.add_argument("--series", choices=("ffac", "ffum", "ffbm"), nargs="+", default=["ffac", "ffum", "ffbm"])
    parser.add_argument("--model", type=Path, help="Explicit final candidate; requires exactly one --series. Otherwise use state.selected_model_path when available.")
    args = parser.parse_args()
    if args.model is not None and len(args.series) != 1:
        parser.error("--model requires exactly one --series")
    cv2.setNumThreads(2)
    for series in args.series:
        result = compare_series(args.root, series, model=args.model)
        print(json.dumps(dict(series=series, status=result["status"], summary=result["summary"])), flush=True)


if __name__ == "__main__":
    main()
