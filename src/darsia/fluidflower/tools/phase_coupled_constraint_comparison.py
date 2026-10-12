"""Audit saved coupled-revision phase maps against the frozen pre-coupled seed.

This reads training evidence and saved predictions only. It does not train,
infer, publish, or modify a model. Categorical maps do not reconstruct joint
probability losses or provide an independent generalization evaluation.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np


from darsia.fluidflower.tools.phase_training_geometry_revision import read, sha, _native
from darsia.fluidflower.tools.phase_training_review_comparison import (
    AQUEOUS_ABSENCE, AQUEOUS_PRESENCE, GAS_ABSENCE,
    _completed, _figure, _photo, _regions, assess_region, decode_phase,
    fragmentation, fractions, mil_fraction_reference, phase_path, write,
)

FORMAT = "coupled-constraint-saved-map-comparison-v1"
GAS_PRESENCE = {"gas", "gas_present_region", "mixed_gas_aqueous", "mixed_gas_water", "mixed_gas_aqueous_water"}
SELECTED_CLASSES = AQUEOUS_ABSENCE | AQUEOUS_PRESENCE | GAS_ABSENCE | GAS_PRESENCE


def annotation_window(annotation, *, canvas_wh, content_box):
    """Exactly rasterize trainer geometry in its smallest native pixel window.

    Float32 conversion, vertex clipping and ties-to-even rounding deliberately
    match review_regions.annotation_mask, including historical border boxes.
    The returned xyxy bounds are half open and refer to cropped native content.
    """
    cw, ch = map(float, canvas_wh)
    cx, cy, width, height = map(int, content_box)
    points = annotation.get("points") or []
    if not points:
        x, y = float(annotation["x"]), float(annotation["y"])
        w, h = float(annotation["width"]), float(annotation["height"])
        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    polygon = np.asarray([(float(x)*cw-cx, float(y)*ch-cy) for x, y in points], dtype=np.float32)
    if polygon.shape[0] < 3 or not np.isfinite(polygon).all() or min(width, height) < 1:
        raise ValueError("Invalid training polygon or native content dimensions")
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width-1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height-1)
    polygon = np.rint(polygon).astype(np.int32)
    x0, y0 = polygon.min(axis=0).tolist()
    x1, y1 = (polygon.max(axis=0)+1).tolist()
    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon-np.asarray((x0, y0), dtype=np.int32)], color=1)
    return (x0, y0, x1, y1), mask.astype(bool)


def _evidence_key(region):
    payload = region.payload
    geometry = {key: payload[key] for key in ("x", "y", "width", "height", "points") if key in payload}
    return (region.dataset, region.run, region.frame_index, region.elapsed_h,
            region.phase_class, region.confidence, region.source, region.reviewer,
            json.dumps(geometry, sort_keys=True, separators=(",", ":")))


def require_same_evidence(old, new):
    old_ids = {r.region_id: _evidence_key(r) for r in old}
    new_ids = {r.region_id: _evidence_key(r) for r in new}
    if len(old_ids) != len(old) or len(new_ids) != len(new):
        raise ValueError("Duplicate effective training-region IDs")
    if old_ids != new_ids:
        raise ValueError("Coupled comparison requires identical frozen region IDs, classes, geometry and human supervision")


def raw_phase_path(job, index):
    return (Path(job["raw_phase_map_output_dir"]) / "native_phase_maps" / job["run"] /
            ("A_"+job["optical_model_sha256"][:16]) / f"frame_{index:03d}.png")


def assessment(old, new, phase_class, positive_fraction):
    result = assess_region(old, new, phase_class, positive_fraction)
    if phase_class in GAS_PRESENCE:
        result.update(gas_old_any=old["gas_evidence_pixels"] > 0,
            gas_new_any=new["gas_evidence_pixels"] > 0,
            gas_old_at_reference=old["gas_evidence_fraction"] >= positive_fraction,
            gas_new_at_reference=new["gas_evidence_fraction"] >= positive_fraction,
            gas_coverage_reference_fraction=float(positive_fraction), mil_probability_loss_evaluated=False)
        if not new["gas_evidence_pixels"]:
            result["flags"].append("no_visible_gas_in_positive_region")
        if old["gas_evidence_pixels"] and not new["gas_evidence_pixels"]:
            result["flags"].append("gas_presence_regression")
        if old["gas_evidence_fraction"] >= positive_fraction > new["gas_evidence_fraction"]:
            result["flags"].append("gas_coverage_crossed_below_reference")
    if phase_class in AQUEOUS_PRESENCE:
        if old["aqueous_evidence_fraction"] >= positive_fraction > new["aqueous_evidence_fraction"]:
            result["flags"].append("aqueous_coverage_crossed_below_reference")
    if phase_class in {"gas", "aqueous"}:
        key = phase_class+"_evidence_fraction"
        result["pure_anchor_fraction_change_pp"] = 100*(new[key]-old[key])
        if new[key] < old[key]:
            result["flags"].append("pure_phase_anchor_coverage_decreased")
        result["interpretations"].append("Pure-phase anchors supervise the full marked area; their coverage is reported separately from regional presence.")
    return result


def aggregate_group(rows, phase_class, positive_fraction):
    """Region-level summaries preserve explicit absence and partial presence."""
    result = dict(phase_class=phase_class, regions=len(rows), stages={})
    for stage, keys in (("raw", ("raw_old", "raw_new")), ("final", ("old", "new"))):
        values = {}
        for side, key in zip(("old", "new"), keys):
            records = [r[key] for r in rows]
            values[side] = dict(region_pixels_sum=sum(r["pixels"] for r in records))
            for phase in ("gas", "aqueous"):
                vals = np.asarray([r[phase+"_evidence_fraction"] for r in records])
                values[side][phase] = dict(mean_fraction=float(vals.mean()), median_fraction=float(np.median(vals)),
                    max_fraction=float(vals.max()), any_regions=int((vals > 0).sum()), zero_regions=int((vals == 0).sum()),
                    at_coverage_reference_regions=int((vals >= positive_fraction).sum()),
                    full_coverage_regions=int((vals == 1).sum()),
                    pixels_sum=sum(r[phase+"_evidence_pixels"] for r in records))
        values["changes"] = {}
        for phase in ("gas", "aqueous"):
            before = np.asarray([r[keys[0]][phase+"_evidence_fraction"] for r in rows])
            after = np.asarray([r[keys[1]][phase+"_evidence_fraction"] for r in rows])
            values["changes"][phase] = dict(increased_regions=int((after > before).sum()),
                decreased_regions=int((after < before).sum()), unchanged_regions=int((after == before).sum()),
                lost_all_presence_regions=int(((before > 0) & (after == 0)).sum()),
                gained_presence_regions=int(((before == 0) & (after > 0)).sum()),
                crossed_below_coverage_reference_regions=int(((before >= positive_fraction) & (after < positive_fraction)).sum()),
                mean_change_pp=float(100*(after-before).mean()))
        result["stages"][stage] = values
    return result


def _validation(root, hashes):
    manifest_path = root / "validation/postflight_manifest.json"
    binding_path = root / "validation/validation_binding.json"
    manifest, binding = read(manifest_path), read(binding_path)
    for key in ("model_id", "model_sha256", "physics_runtime"):
        if not manifest.get(key) or binding.get(key) != manifest[key]:
            raise ValueError("Saved prediction identity differs between manifest and binding")
    hashes.update({str(path): sha(path) for path in (manifest_path, binding_path, root/"training_request.json")})
    return manifest


def compare_coupled(campaign_root, series, *, model=None, base=None, figures=True):
    from darsia.fluidflower import phase_runtime
    campaign_root = Path(campaign_root).resolve()
    state = read(campaign_root / series / "state.json")
    base = Path(base or state["output_root"]).resolve()
    selected = model or state.get("selected_model_path")
    model = Path(selected).resolve() if selected else None
    output = campaign_root / series / "coupled_constraint_comparison.json"
    result = dict(format=FORMAT, series=series, status="pending", base=str(base), model=str(model) if model else None,
        summary=[], figures=[], regions=[], late_frames=[], groups={}, unavailable_regions=[],
        independent_validation=False, mil_probability_loss_evaluated=False,
        scope="Frozen training-evidence comparison of the pre-coupled seed against the coupled revision. No held-out/OOF or scientific-acceptance claims.",
        aggregation="Each region has equal weight in means/medians. Summed region pixels can overlap; these are not unique physical area or mass.")
    if model is None or model == base or not _completed(base) or not _completed(model):
        result["summary"] = ["Venter på ferdige lagrede fasekart for både grunnmodellen og en separat modell med koblet trening."]
        write(output, result); return result
    hashes = {}
    old_manifest, new_manifest = [_validation(root, hashes) for root in (base, model)]
    result.update(base_model_id=old_manifest["model_id"], model_id=new_manifest["model_id"],
        base_physics_runtime=old_manifest["physics_runtime"], model_physics_runtime=new_manifest["physics_runtime"])
    old_regions, _ = _regions(base)
    new_regions, _ = _regions(model)
    require_same_evidence(old_regions, new_regions)
    selected_regions = [r for r in new_regions if r.phase_class in SELECTED_CLASSES]
    positive_fraction, fraction_source = mil_fraction_reference(model)
    result["mil_coverage_reference"] = dict(fraction=positive_fraction, source=fraction_source,
        interpretation="Descriptive categorical coverage reference. It does not evaluate the probability-based regional or joint loss.")
    result["evidence"] = dict(all_effective_regions=len(new_regions), selected_regions=len(selected_regions),
        classes=dict(Counter(r.phase_class for r in selected_regions)),
        identical_frozen_evidence=True, selection="All effective exclusions and positive-phase anchors, including inherited evidence.")
    by_frame = defaultdict(list)
    for region in selected_regions:
        by_frame[(region.run, region.frame_index)].append(region)
    old_jobs, new_jobs = [{j["run"]: j for j in m["jobs"]} for m in (old_manifest, new_manifest)]
    metadata = {}
    old_profile, new_profile = [phase_runtime.load_physics_profile(root) for root in (base, model)]
    fig_dir = campaign_root / series / "coupled_comparison_figures" / new_manifest["model_id"]
    for run in sorted(set(old_jobs) & set(new_jobs)):
        old_job, new_job = old_jobs[run], new_jobs[run]
        old_native_root, old_path = _native(old_job)
        new_native_root, new_path = _native(new_job)
        native = read(new_path)
        if sha(old_path) != sha(new_path):
            raise ValueError(f"Coupled comparison requires exactly the same native image revision: {run}")
        hashes.update({str(path): sha(path) for path in (old_path, new_path)})
        contexts = [phase_runtime.load_run_context(root, profile, series, run,
            native_manifest_sha256=sha(new_path))[0] for root, profile in ((base, old_profile), (model, new_profile))]
        domain = np.asarray(contexts[1].domain, dtype=bool)
        if not np.array_equal(contexts[0].domain, domain):
            raise ValueError(f"Physical domain changed between base and coupled model: {run}")
        late = set(range(max(0, int(native["frame_count"])-3), int(native["frame_count"])))
        wanted = sorted(late | {i for r, i in by_frame if r == run})
        metadata[run] = (old_job, new_job, native, new_native_root)
        for index in wanted:
            if not 0 <= index < int(native["frame_count"]):
                continue
            old_final, new_final = [decode_phase(phase_path(job, index), native["content_box_xywh"], hashes) for job in (old_job, new_job)]
            if old_final.shape != new_final.shape or new_final.shape != domain.shape:
                raise ValueError("Native phase maps and physical domain have different shapes")
            record = dict(run=run, frame_index=index, elapsed_h=float(native["frame_sources"][index]["elapsed_h"]),
                old_map=str(phase_path(old_job, index)), new_map=str(phase_path(new_job, index)))
            if by_frame[(run, index)]:
                old_raw, new_raw = [decode_phase(raw_phase_path(job, index), native["content_box_xywh"], hashes) for job in (old_job, new_job)]
                for region in by_frame[(run, index)]:
                    bbox, mask = annotation_window(region.payload, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
                    x0, y0, x1, y1 = bbox
                    records = {name: fractions(labels[y0:y1, x0:x1], mask) for name, labels in
                        (("old", old_final), ("new", new_final), ("raw_old", old_raw), ("raw_new", new_raw))}
                    row = dict(**record, region_id=region.region_id, phase_class=region.phase_class, bbox_xyxy=bbox,
                        **records, native_pixels_outside_physical_domain=int((mask & ~domain[y0:y1, x0:x1]).sum()),
                        assessment=assessment(records["old"], records["new"], region.phase_class, positive_fraction),
                        raw_assessment=assessment(records["raw_old"], records["raw_new"], region.phase_class, positive_fraction))
                    result["regions"].append(row)
                del old_raw, new_raw
            if index in late:
                row = dict(**record, old=fractions(old_final, domain), new=fractions(new_final, domain),
                    old_fragmentation=fragmentation(old_final, domain), new_fragmentation=fragmentation(new_final, domain))
                if figures:
                    photo = _photo(new_native_root/"native_frames"/run/f"frame_{index:03d}.jpg", native["content_box_xywh"], hashes)
                    path = fig_dir/f"{run}_late_{index:03d}.png"
                    _figure(path, photo, old_final, new_final, title=f"{series.upper()} · {run.upper()} · {record['elapsed_h']:.2f} t · før og etter koblet trening")
                    result["figures"].append(dict(path=str(path), caption="Samme fotografi og fysikkdomene; venstre modell før og høyre modell etter koblet trening.",
                        run=run, frame_index=index, elapsed_h=record["elapsed_h"], kind="late_full_frame"))
                    row["figure"] = str(path)
                result["late_frames"].append(row)
    measured = {row["region_id"] for row in result["regions"]}
    result["unavailable_regions"] = [dict(region_id=r.region_id, run=r.run, frame_index=r.frame_index, phase_class=r.phase_class,
        reason="No corresponding run/frame in both saved validation manifests") for r in selected_regions if r.region_id not in measured]
    for phase_class in sorted({row["phase_class"] for row in result["regions"]}):
        rows = [r for r in result["regions"] if r["phase_class"] == phase_class]
        result["groups"][phase_class] = aggregate_group(rows, phase_class, positive_fraction)
    if figures:
        # Include both failure/exclusion behavior and positive anchor retention.
        chosen = []
        for phase_class in ("aqueous_absent_region", "gas_absent_region", "aqueous_present_region", "gas", "aqueous"):
            rows = [r for r in result["regions"] if r["phase_class"] == phase_class]
            if not rows:
                continue
            phase = "gas" if phase_class in {"gas_absent_region", "gas"} else "aqueous"
            direction = 1 if phase_class.endswith("absent_region") else -1
            chosen.extend(sorted(rows, key=lambda r: (-direction*r["new"][phase+"_evidence_fraction"], r["region_id"]))[:1])
            chosen.extend(sorted(rows, key=lambda r: (direction*(r["new"][phase+"_evidence_fraction"]-r["old"][phase+"_evidence_fraction"]), r["region_id"]))[:1])
        lookup = {r.region_id: r for r in selected_regions}
        for row in {r["region_id"]: r for r in chosen}.values():
            run, index = row["run"], row["frame_index"]
            old_job, new_job, native, native_root = metadata[run]
            old_labels, new_labels = [decode_phase(phase_path(job, index), native["content_box_xywh"], hashes) for job in (old_job, new_job)]
            bbox, mask = annotation_window(lookup[row["region_id"]].payload, canvas_wh=native["canvas_wh"], content_box=native["content_box_xywh"])
            x0, y0, x1, y1 = bbox
            margin = max(16, round(max(x1-x0, y1-y0)*.15))
            crop = (max(0, x0-margin), max(0, y0-margin), min(new_labels.shape[1], x1+margin), min(new_labels.shape[0], y1+margin))
            cx0, cy0, cx1, cy1 = crop
            outline = np.zeros((cy1-cy0, cx1-cx0), dtype=bool)
            outline[y0-cy0:y1-cy0, x0-cx0:x1-cx0] = mask
            photo = _photo(native_root/"native_frames"/run/f"frame_{index:03d}.jpg", native["content_box_xywh"], hashes)
            path = fig_dir/f"{run}_region_{row['region_id'].split(':')[-1][:12]}.png"
            _figure(path, photo[cy0:cy1, cx0:cx1], old_labels[cy0:cy1, cx0:cx1], new_labels[cy0:cy1, cx0:cx1], outline=outline,
                title=f"{run.upper()} · {row['elapsed_h']:.2f} t · {row['phase_class']}\nGass: {100*row['old']['gas_evidence_fraction']:.2f}% → {100*row['new']['gas_evidence_fraction']:.2f}% · innløst: {100*row['old']['aqueous_evidence_fraction']:.2f}% → {100*row['new']['aqueous_evidence_fraction']:.2f}%")
            row["figure"] = str(path)
            result["figures"].append(dict(path=str(path), caption="Frossen treningsmarkering; hvit linje viser området.",
                run=run, frame_index=index, elapsed_h=row["elapsed_h"], region_id=row["region_id"], kind="training_constraint_region"))
    result["summary"] = [f"{len(measured)}/{len(selected_regions)} relevante treningsmarkeringer sammenlignet før og etter koblet trening."]
    for phase_class, group in result["groups"].items():
        stage = group["stages"]["final"]
        if phase_class == "aqueous_absent_region":
            result["summary"].append(f"Eksplisitt ingen innløst CO₂: {stage['old']['aqueous']['zero_regions']} → {stage['new']['aqueous']['zero_regions']} helt korrekte områder av {group['regions']}; {stage['changes']['aqueous']['decreased_regions']} forbedret og {stage['changes']['aqueous']['increased_regions']} med mer restsignal.")
        if phase_class in {"aqueous_present_region", "gas", "aqueous"}:
            phase = "gas" if phase_class == "gas" else "aqueous"
            result["summary"].append(f"{phase_class}: synlig forventet fase i {stage['old'][phase]['any_regions']} → {stage['new'][phase]['any_regions']} av {group['regions']} områder; gjennomsnittlig dekning {100*stage['old'][phase]['mean_fraction']:.2f}% → {100*stage['new'][phase]['mean_fraction']:.2f}%.")
    result["summary"].append("Dette er treningsdata. Rene faseankre, svak tilstedeværelse og kantfragmentering vurderes hver for seg; kategoriske kart bekrefter ikke alene kalibrerte sannsynligheter.")
    result.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
        coverage_complete=not result["unavailable_regions"],
        flags=dict(Counter(flag for row in result["regions"] for flag in row["assessment"]["flags"])),
        raw_flags=dict(Counter(flag for row in result["regions"] for flag in row["raw_assessment"]["flags"])),
        source_hashes=hashes, source_files_modified=False)
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=local_cache_root() / 'phase_diagnostics/all_series_review_update_20260912')
    parser.add_argument("--series", choices=("ffac", "ffum", "ffbm"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    result = compare_coupled(args.root, args.series, model=args.model, base=args.base, figures=not args.no_figures)
    print(json.dumps(dict(series=args.series, status=result["status"], summary=result["summary"])), flush=True)


if __name__ == "__main__":
    main()
