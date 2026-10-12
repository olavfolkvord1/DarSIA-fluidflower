"""Two-frame paired native diagnostic over the frozen FF_UM parent output.

The raw stage caches the exact parent's float32 CPU optical scores. The paired
stage isolates the continuity option over saved final labels and causal history;
it is not a new-model full chronological replay or independent validation.
"""
from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

REPO = series_root("ffac")
pass  # Installed package imports do not modify sys.path.
from darsia.fluidflower.tools.phase_mass_overnight_data import read, write, decode_labels, sha
from darsia.fluidflower.tools.phase_inference_benchmark_backends import CpuFeaturePredictor
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower import phase_aqueous_continuity as continuity
from darsia.fluidflower.phase_reference_support import ReferenceSettings

ROOT = local_cache_root() / 'phase_diagnostics/ffum_review_update_20260914'
OUTPUT = ROOT / "continuity_diagnostic"
PARENT = configured_tool_path('ffac', 'calibration_logs') / 'phase_models/studio_retraining/ffum/phase-model-20260912T154819Z-25d9ebee'
TARGETS = (("run20", 20), ("run18", 20))


def crop_uint8(path, box):
    x, y, w, h = box
    with Image.open(path) as im:
        if im.size == (w, h):
            return np.asarray(im.convert("RGB"))
        if im.width < x+w or im.height < y+h:
            raise ValueError(f"Image does not cover the registered crop: {path}")
        return np.asarray(im.crop((x, y, x+w, y+h)).convert("RGB"))


def inputs(run, frame):
    validation = PARENT / "validation"
    manifest = read(validation / "postflight_manifest.json")
    job = next(j for j in manifest["jobs"] if j["run"] == run)
    source = Path(job["analysis_source_review_root"])
    native_path = source / "native_frames" / run / "manifest.json"
    native = read(native_path)
    profile = manifest["physics_profile"]
    context_path = PARENT / profile["runs"][f"ffum/{run}"]["path"]
    phase_path = (Path(job["phase_map_output_dir"]) / "native_phase_maps" / run /
                  ("A_" + manifest["pipeline_sha256"][:16]) / f"frame_{frame:03d}.png")
    baseline_path = source / "native_references" / run / "pre_injection.jpg"
    frame_path = source / "native_frames" / run / f"frame_{frame:03d}.jpg"
    previous_path = source / "native_frames" / run / f"frame_{frame-1:03d}.jpg"
    support_path = validation / "presence_diagnostics" / run / f"frame_{frame:03d}.npz"
    physics_report = validation / "physics_reports" / f"ffum_{run}.json"
    report = next(f for f in read(physics_report)["frames"] if f["frame_index"] == frame)
    return dict(run=run, frame=frame, native=native, profile=profile, report=report,
        paths=dict(context=context_path, native=native_path, phase=phase_path, baseline=baseline_path,
                   current=frame_path, previous=previous_path, support=support_path, physics_report=physics_report),
        output=OUTPUT / f"{run}_frame{frame:03d}")


def raw_stage(info):
    output, paths = info["output"], info["paths"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw_optical_receipt.json").exists():
        receipt = read(output / "raw_optical_receipt.json")
        if sha(output / "raw_co2.npy") != receipt["raw_co2_sha256"]:
            raise ValueError("Cached raw optical evidence changed")
        print(f"Retained cached parent CPU prediction: {info['run']}", flush=True)
        return
    started = time.perf_counter()
    box = info["native"]["content_box_xywh"]
    baseline, current, previous = [crop_uint8(paths[k], box) for k in ("baseline", "current", "previous")]
    model = HierarchicalPhaseModel.load(PARENT, adapter="generic")
    print(f"CPU parent prediction {info['run']} frame {info['frame']}, shape {current.shape[:2]}", flush=True)
    predictor = CpuFeaturePredictor(model, baseline)
    prediction = predictor.predict(current, previous, current_key=info["frame"], previous_key=info["frame"]-1)
    prediction_seconds = time.perf_counter() - started
    with (output / "raw_co2.npy").open("xb") as stream:
        np.save(stream, prediction["co2_presence_probability"], allow_pickle=False)
    sources = {str(path): sha(path) for path in paths.values()}
    sources.update({str(path): sha(path) for path in (PARENT / "models/generic").glob("*.json")})
    write(output / "raw_optical_receipt.json", dict(run=info["run"], frame=info["frame"],
        elapsed_h=info["report"]["elapsed_h"], model=str(PARENT), backend="CpuFeaturePredictor float32",
        feature_contract=model.feature_contract.to_dict(), prediction_seconds=prediction_seconds,
        source_hashes=sources, raw_co2_sha256=sha(output / "raw_co2.npy"), gpu_used=False,
        source_code_sha256=sha(Path(sys.modules[CpuFeaturePredictor.__module__].__file__))))
    del predictor, prediction, baseline, current, previous
    gc.collect()
    print(f"Cached {info['run']} raw CO2: {prediction_seconds:.1f} seconds", flush=True)


def metrics(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return dict(area_pixels=int(mask.sum()), external_components=len(contours), external_perimeter_px=perimeter,
                short_external_components=sum(cv2.arcLength(c, True) < 25 for c in contours))


def contours_on_photo(photo, labels, *, changes=None):
    image = photo.copy()
    for label, colour in ((1, (49, 217, 160)), (2, (242, 76, 157))):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour, 2)
    if changes is not None:
        added, removed = changes
        image[added] = (.3 * image[added] + .7 * np.array([70, 180, 255])).astype(np.uint8)
        image[removed] = (.3 * image[removed] + .7 * np.array([255, 164, 50])).astype(np.uint8)
    return image


def comparison_image(path, current, before, after, *, region=None, title="", boxes=()):
    offset_y, offset_x = (region[0].start, region[1].start) if region is not None else (0, 0)
    if region is not None:
        current, before, after = current[region], before[region], after[region]
    added, removed = (before == 0) & (after == 1), (before == 1) & (after == 0)
    panels = [current, contours_on_photo(current, before), contours_on_photo(current, after),
              contours_on_photo(current, after, changes=(added, removed))]
    labels = ("Photograph", "Frozen parent", "Continuity candidate", "Blue: additions / orange: removals")
    width = 1100 if region is None else 800
    resized = []
    for panel, label in zip(panels, labels):
        if boxes:
            panel = panel.copy()
            for x, y, w, h in boxes:
                cv2.rectangle(panel, (x-offset_x, y-offset_y), (x+w-offset_x, y+h-offset_y), (255, 215, 65), 2)
        height = round(panel.shape[0] * width / panel.shape[1])
        resized_panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height + 58, width, 3), 245, np.uint8)
        canvas[58:] = resized_panel
        cv2.putText(canvas, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 1, cv2.LINE_AA)
        resized.append(canvas)
    combined = np.concatenate(resized, axis=1)
    header = np.full((52, combined.shape[1], 3), 255, np.uint8)
    cv2.putText(header, title, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, .85, (20, 20, 20), 2, cv2.LINE_AA)
    Image.fromarray(np.concatenate((header, combined), axis=0)).save(path)


def marked_region_metrics(info, before, after):
    before_aq, after_aq = before == 1, after == 1
    rows = []
    for region in read(ROOT / "effective_region_snapshot.json"):
        if region["run"] != info["run"] or region["frame_index"] != info["frame"]:
            continue
        phase = region["phase_class"]
        if phase not in {"aqueous_present_region", "aqueous_absent_region", "gas_absent_region"}:
            continue
        mask = t._annotation_mask(region["payload"], canvas_wh=info["native"]["canvas_wh"],
                                   content_box=info["native"]["content_box_xywh"])
        size = int(mask.sum())
        if not size:
            continue
        y, x = np.where(mask)
        old, new = int((before_aq & mask).sum()), int((after_aq & mask).sum())
        rows.append(dict(region_id=region["region_id"], phase_class=phase, pixels=size,
            before_aqueous_pixels=old, after_aqueous_pixels=new, before_present=old > 0,
            after_present=new > 0, aqueous_fraction_delta=(new-old)/size,
            gas_pixels_before=int(((before == 2) & mask).sum()), gas_pixels_after=int(((after == 2) & mask).sum()),
            bounding_box_xywh=[int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)]))
    return rows


def paired_stage(info):
    output, paths = info["output"], info["paths"]
    receipt = read(output / "raw_optical_receipt.json")
    if sha(output / "raw_co2.npy") != receipt["raw_co2_sha256"]:
        raise ValueError("Cached raw optical evidence changed")
    for path, digest in receipt["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    box = info["native"]["content_box_xywh"]
    with np.load(paths["context"]) as archive:
        context = SimpleNamespace(**{k: archive[k] for k in ("domain", "facies", "x_m", "y_m")})
    with np.load(paths["support"]) as archive:
        history = archive["trusted_history"]
    baseline, current = [crop_uint8(paths[k], box) for k in ("baseline", "current")]
    before = decode_labels(paths["phase"], box)
    raw = np.load(output / "raw_co2.npy", allow_pickle=False)
    settings = continuity.ContinuitySettings()
    reference_settings = ReferenceSettings(**info["profile"]["reference_support"].get("reference_settings", {}))
    old = dict(co2_presence_probability=np.where(before > 0, .5, .1).astype(np.float32),
               gas_probability_conditional_on_co2=np.where(before == 2, .9, .2).astype(np.float32))
    controls = info["report"]["reference_support"]["facies_controls"]
    started = time.perf_counter()
    print(f"Continuity reference signal {info['run']} frame {info['frame']}", flush=True)
    control_mask = continuity.unchanged_control_mask(context, old, raw, history, reference_settings)
    signal_diagnostics = {}
    score, native_score, available = continuity.reference_signal(baseline, current, context, before == 2,
        controls, reference_settings, settings, control_mask=control_mask, diagnostics=signal_diagnostics)
    signal_seconds = time.perf_counter() - started
    refine_started = time.perf_counter()
    result, detail = continuity.refine(old, raw, context, history, score, native_score, available, settings)
    refine_seconds = time.perf_counter() - refine_started
    after = np.zeros_like(before)
    after[result["co2_presence_probability"] >= .5] = 1
    after[(result["co2_presence_probability"] >= .5) & (result["gas_probability_conditional_on_co2"] >= .6)] = 2
    if not np.array_equal(before == 2, after == 2):
        raise AssertionError("Continuity changed gas")
    # Exercise the real runtime's explicit pre-injection bypass at native size.
    # This is a bypass identity control, not an extra baseline optical fit.
    proxy = SimpleNamespace(context=context, settings=settings,
        inherited=SimpleNamespace(calibration_baseline=baseline, last_diagnostics={}))
    negative_output, negative_report = continuity.AqueousBoundarySequence._finish_continuity(
        proxy, old, old, {"pre_injection_negative_control": True}, baseline)
    baseline_unchanged = all(np.array_equal(negative_output[k], old[k]) for k in old)
    if not baseline_unchanged:
        raise AssertionError("Pre-injection bypass changed inherited output")
    rows = marked_region_metrics(info, before, after)
    old_metrics, new_metrics = metrics(before > 0), metrics(after > 0)
    exclusions = [row for row in rows if row["phase_class"] == "aqueous_absent_region"]
    positive = [row for row in rows if row["phase_class"] == "aqueous_present_region"]
    added, removed = (before == 0) & (after == 1), (before == 1) & (after == 0)
    report = dict(run=info["run"], frame=info["frame"], elapsed_h=info["report"]["elapsed_h"],
        scope="Isolated continuity applied to frozen parent labels and exact parent raw optical scores; reviewed training frame, not independent validation or a chronological replay",
        before=old_metrics, after=new_metrics, continuity=detail, signal_seconds=signal_seconds,
        reference_signal_diagnostics=signal_diagnostics,
        refine_seconds=refine_seconds, cpu_raw_prediction_seconds=receipt["prediction_seconds"],
        perimeter_change_percent=100*(new_metrics["external_perimeter_px"]/old_metrics["external_perimeter_px"]-1),
        area_change_percent=100*(new_metrics["area_pixels"]/old_metrics["area_pixels"]-1),
        marked_positive_boxes=len(positive), positive_boxes_before=sum(row["before_present"] for row in positive),
        positive_boxes_after=sum(row["after_present"] for row in positive),
        aqueous_exclusion_boxes=len(exclusions), exclusions_worse=sum(row["after_aqueous_pixels"] > row["before_aqueous_pixels"] for row in exclusions),
        added_pixels_in_aqueous_exclusion_boxes=sum(max(0, row["after_aqueous_pixels"]-row["before_aqueous_pixels"]) for row in exclusions),
        gas_labels_unchanged=True, baseline_bypass_unchanged=baseline_unchanged,
        baseline_control_scope="Real pre-injection bypass on identical native inherited output; no additional baseline optical prediction",
        parent_model_unchanged=True, history_unchanged=True, settings=asdict(settings),
        continuity_source_sha256=sha(Path(continuity.__file__)), source_hashes=receipt["source_hashes"],
        region_results=rows)
    write(output / "result.json", report)
    np.savez_compressed(output / "paired_masks.npz", before=before, after=after, added=added, removed=removed,
                        trusted_history=history, available=available)
    title = f"{info['run'].upper()} / {info['report']['elapsed_h']:.2f} h / frozen parent + isolated continuity"
    comparison_image(output / "native_comparison.png", current, before, after, title=title)
    if (added | removed).any():
        coarse = cv2.resize((added | removed).astype(np.float32), (16, 9), interpolation=cv2.INTER_AREA)
        row, col = np.unravel_index(coarse.argmax(), coarse.shape)
        h, w = before.shape
        region = np.s_[max(0, int(row*h/9)-100):min(h, int((row+1)*h/9)+100),
                       max(0, int(col*w/16)-100):min(w, int((col+1)*w/16)+100)]
        comparison_image(output / "edge_detail.png", current, before, after, region=region, title=title + " / most changed tile")
    if positive:
        selected = max(positive, key=lambda row: (not row["before_present"], row["after_present"],
                       row["after_aqueous_pixels"]-row["before_aqueous_pixels"]))
        x, y, w, h = selected["bounding_box_xywh"]
        region = np.s_[max(0, y-180):min(before.shape[0], y+h+180), max(0, x-180):min(before.shape[1], x+w+180)]
        comparison_image(output / "marked_positive_detail.png", current, before, after, region=region,
            boxes=[selected["bounding_box_xywh"]],
            title=title + f" / marked presence: {selected['before_aqueous_pixels']} to {selected['after_aqueous_pixels']} aqueous pixels")
    print({k: v for k, v in report.items() if k not in {"region_results", "source_hashes"}}, flush=True)
    del raw, before, after, baseline, current, score, native_score, result
    gc.collect()


def cache_fields_stage(info):
    """Persist the measured fields for bounded parameter diagnostics, without refitting."""
    from darsia.fluidflower import phase_native_reference as native, phase_reviewed_reference as reviewed
    output, paths = info["output"], info["paths"]
    report = read(output / "result.json")
    code_hash = sha(Path(continuity.__file__))
    if code_hash != report["continuity_source_sha256"]:
        raise ValueError("Reference implementation changed since the paired diagnostic")
    box = info["native"]["content_box_xywh"]
    with np.load(paths["context"]) as archive:
        context = SimpleNamespace(**{k: archive[k] for k in ("domain", "facies", "x_m", "y_m")})
    with np.load(output / "paired_masks.npz") as archive:
        history, before, added = archive["trusted_history"], archive["before"], archive["added"]
    baseline, current = [crop_uint8(paths[k], box) for k in ("baseline", "current")]
    raw = np.load(output / "raw_co2.npy", allow_pickle=False)
    settings = continuity.ContinuitySettings()
    reference_settings = ReferenceSettings(**info["profile"]["reference_support"].get("reference_settings", {}))
    old = dict(co2_presence_probability=np.where(before > 0, .5, .1).astype(np.float32),
               gas_probability_conditional_on_co2=np.where(before == 2, .9, .2).astype(np.float32))
    control_mask = continuity.unchanged_control_mask(context, old, raw, history, reference_settings)
    diagnostics = {}
    print(f"Cache measured continuity fields: {info['run']}", flush=True)
    score, native_score, available = continuity.reference_signal(baseline, current, context, before == 2,
        info["report"]["reference_support"]["facies_controls"], reference_settings, settings,
        control_mask=control_mask, diagnostics=diagnostics)
    dx = continuity.boundary._spacing(context.x_m, context.domain.shape[1])
    dy = continuity.boundary._spacing(context.y_m, context.domain.shape[0], descending=True)
    smooth = continuity.boundary._material_average(raw, context.domain, context.facies,
                                                    settings.sigma_m/dx, settings.sigma_m/dy)
    features, gate_available = reviewed.change_fields(baseline, current)
    gate_available &= features[..., 3] > 1e-7
    calibration = info["profile"]["calibration"]
    gate_score = native._facies_linear(reviewed.scores(features, calibration), gate_available, context.facies)
    gate = gate_score >= calibration["threshold"]
    thresholds = {}
    for label, threshold in calibration.get("facies_thresholds", {}).items():
        selected = context.facies == int(label)
        gate[selected] = gate_score[selected] >= threshold
        thresholds[label] = threshold
    arrays = dict(signal=score, native_signal=native_score, available=available, history=history,
        control=control_mask, smooth_raw_co2=smooth, reference_gate_score=gate_score, reference_gate_accepted=gate)
    fields = output / "fields"
    fields.mkdir(exist_ok=True)
    for name, value in arrays.items():
        with (fields / f"{name}.npy").open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
    absent = np.zeros_like(added)
    positive = np.zeros_like(added)
    new_ids = {r["region_id"] for r in read(ROOT / "new_regions.json")}
    annotations = []
    for region in read(ROOT / "effective_region_snapshot.json"):
        if region["run"] != info["run"] or region["frame_index"] != info["frame"]:
            continue
        negative = region["phase_class"] == "aqueous_absent_region"
        new_positive = region["region_id"] in new_ids and region["phase_class"] == "aqueous_present_region"
        if not negative and not new_positive:
            continue
        mask = t._annotation_mask(region["payload"], canvas_wh=info["native"]["canvas_wh"], content_box=box)
        if negative:
            absent |= mask
            if (added & mask).any():
                annotations.append(dict(region_id=region["region_id"], added_pixels=int((added & mask).sum()),
                    **{k: region["payload"].get(k) for k in
                       ("category", "training_phase_class", "created_utc", "updated_utc", "reclassified_utc", "reclassification_rule")}))
        else:
            positive |= mask
    def statistics(mask):
        count = int(mask.sum())
        def quantiles(field):
            values = np.asarray(field[mask], np.float64)
            values = values[np.isfinite(values)]
            return dict(zip(("q05", "q25", "median", "q75", "q95"), np.quantile(values, [.05, .25, .5, .75, .95]).tolist())) if values.size else None
        return dict(pixels=count, raw_co2=quantiles(raw), smooth_raw_co2=quantiles(smooth),
            signal=quantiles(score), native_signal=quantiles(native_score), reference_gate_score=quantiles(gate_score),
            reference_gate_accepted_fraction=float(gate[mask].mean()) if count else None,
            trusted_history_fraction=float(history[mask].mean()) if count else None)
    write(output / "field_cache_receipt.json", dict(continuity_source_sha256=code_hash,
        reference_signal_diagnostics=diagnostics, calibration=calibration,
        arrays={name: str(fields / f"{name}.npy") for name in arrays},
        raw_co2_path=str(output / "raw_co2.npy"),
        populations={label: statistics(mask) for label, mask in (
            ("additions_inside_aqueous_exclusions", added & absent),
            ("additions_outside_aqueous_exclusions", added & ~absent),
            ("new_aqueous_presence_box_pixels", positive))},
        affected_exclusion_annotations=sorted(annotations, key=lambda row: -row["added_pixels"])))
    print(f"Cached signal/native/history/control/gate fields: {info['run']}", flush=True)
    del raw, score, native_score, smooth, arrays, baseline, current, features
    gc.collect()


def gated_stage(info):
    """One paired variant retaining the parent's exact reviewed reference gate."""
    from darsia.fluidflower import phase_native_reference as native, phase_reviewed_reference as reviewed
    output, paths = info["output"], info["paths"]
    destination = output / "gated"
    destination.mkdir(exist_ok=True)
    box = info["native"]["content_box_xywh"]
    with np.load(paths["context"]) as archive:
        context = SimpleNamespace(**{k: archive[k] for k in ("domain", "facies", "x_m", "y_m")})
    with np.load(output / "paired_masks.npz") as archive:
        before, original_added, removed = archive["before"], archive["added"], archive["removed"]
    baseline, current = [crop_uint8(paths[k], box) for k in ("baseline", "current")]
    features, available = reviewed.change_fields(baseline, current)
    gate = native.accepted_proposals(features, available, context.facies, context.domain,
                                     info["profile"]["calibration"])
    saved_gate = np.load(output / "fields/reference_gate_accepted.npy", allow_pickle=False)
    if not np.array_equal(gate, saved_gate & context.domain):
        raise AssertionError("Cached reference gate differs from actual parent's accepted_proposals")
    retained = (before > 0) & ~removed
    added = continuity.boundary._connected_to_retained(original_added & gate, retained, context.domain)
    after = before.copy()
    after[removed] = 0
    after[added] = 1
    if not np.array_equal(before == 2, after == 2):
        raise AssertionError("Gated continuity changed gas labels")
    rows = marked_region_metrics(info, before, after)
    positive = [row for row in rows if row["phase_class"] == "aqueous_present_region"]
    exclusions = [row for row in rows if row["phase_class"] == "aqueous_absent_region"]
    new_ids = {r["region_id"] for r in read(ROOT / "new_regions.json")}
    negative_union = np.zeros_like(added)
    for region in read(ROOT / "effective_region_snapshot.json"):
        if region["run"] == info["run"] and region["frame_index"] == info["frame"] and region["phase_class"] == "aqueous_absent_region":
            negative_union |= t._annotation_mask(region["payload"], canvas_wh=info["native"]["canvas_wh"], content_box=box)
    old_metrics, new_metrics = metrics(before > 0), metrics(after > 0)
    report = dict(run=info["run"], frame=info["frame"], elapsed_h=info["report"]["elapsed_h"],
        scope="One isolated variant retaining the exact unchanged parent reviewed reference gate for additions, followed by retained-plume connectivity",
        before=old_metrics, after=new_metrics, additions=int(added.sum()), removals=int(removed.sum()),
        rejected_by_reviewed_gate=int((original_added & ~gate).sum()),
        rejected_by_connectivity_after_gate=int((original_added & gate & ~added).sum()),
        positive_groups={name: dict(boxes=len(group), before=sum(r["before_present"] for r in group), after=sum(r["after_present"] for r in group))
            for name, group in (("all", positive), ("new", [r for r in positive if r["region_id"] in new_ids]))},
        aqueous_exclusion_boxes=len(exclusions), exclusions_worse=sum(r["after_aqueous_pixels"] > r["before_aqueous_pixels"] for r in exclusions),
        unique_added_pixels_in_aqueous_exclusions=int((added & negative_union).sum()),
        fraction_of_additions_in_aqueous_exclusions=float((added & negative_union).sum()/max(1, added.sum())),
        perimeter_change_percent=100*(new_metrics["external_perimeter_px"]/old_metrics["external_perimeter_px"]-1),
        area_change_percent=100*(new_metrics["area_pixels"]/old_metrics["area_pixels"]-1),
        gate_matches_parent_accepted_proposals=True, gas_labels_unchanged=True, history_unchanged=True,
        original_source_result=str(output / "result.json"), parent_calibration_unchanged=True, region_results=rows)
    write(destination / "result.json", report)
    np.savez_compressed(destination / "paired_masks.npz", before=before, after=after, added=added, removed=removed)
    title = f"{info['run'].upper()} / {info['report']['elapsed_h']:.2f} h / continuity plus unchanged reviewed gate"
    comparison_image(destination / "native_comparison.png", current, before, after, title=title)
    if added.any():
        coarse = cv2.resize(added.astype(np.float32), (16, 9), interpolation=cv2.INTER_AREA)
        row, col = np.unravel_index(coarse.argmax(), coarse.shape)
        h, w = before.shape
        region = np.s_[max(0, int(row*h/9)-100):min(h, int((row+1)*h/9)+100),
                       max(0, int(col*w/16)-100):min(w, int((col+1)*w/16)+100)]
        comparison_image(destination / "edge_detail.png", current, before, after, region=region, title=title + " / most changed tile")
    original_report = read(output / "result.json")
    if original_report.get("worst_exclusion_region"):
        original_worst = original_report["worst_exclusion_region"]
        selected = next(row for row in rows if row["region_id"] == original_worst["region_id"])
        x, y, w, h = selected["bounding_box_xywh"]
        region = np.s_[max(0, y-120):min(before.shape[0], y+h+120), max(0, x-120):min(before.shape[1], x+w+120)]
        comparison_image(destination / "original_worst_exclusion_detail.png", current, before, after, region=region,
            boxes=[selected["bounding_box_xywh"]], title=title + f" / aqueous exclusion {selected['before_aqueous_pixels']} to {selected['after_aqueous_pixels']} pixels")
    print({k: v for k, v in report.items() if k != "region_results"}, flush=True)
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("raw", "paired", "cache-fields", "gated"), required=True)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    with threadpool_limits(limits=4):
        for run, frame in TARGETS:
            {"raw": raw_stage, "paired": paired_stage, "cache-fields": cache_fields_stage, "gated": gated_stage}[args.stage](inputs(run, frame))
