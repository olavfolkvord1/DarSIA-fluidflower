"""Source-bound T4 QA and in-sample L-cap descriptions, without sealed scorers.

Reads only the authorized L-cap CSV and fitted training receipts. Every saved
run is reported. Regression labels are exported from production maps without
opening any round 2--4 labels, sessions, count manifests, or scoring scripts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import report_ffac_t2_floor_candidate as shared
from . import report_ffac_t2_l1zone_candidate as floorqa
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now
from .run_ffac_t4_floor_candidate import artifact, bound_path, read_json

CONTRACT = "ffac-g3-t4-floor-candidate-qa-v1"
LCAP_SHA256 = "93645a04957468390592ccec5b9d045b3c83cd4294dc6d1b4528daa9418c7d1d"
ZONE_POLICY = dict(shared.ZONE_POLICY,
    l1_cap="reference zone4 within40mm below preceding facies1/2 barrier",
    l1_column="remaining reference zone4",
    l2_caps="reference zones2/3/5 within40mm below preceding facies1/2 barrier",
    l2_columns="remaining reference zones2/3/5",
    interpretation="fixed transported geometric zone descriptions; no validation threshold or tuning")


def identities(base, result, fill):
    if (base.dtype != np.uint8 or result.dtype != np.uint8 or base.ndim != 2
            or base.shape != result.shape or fill.shape != base.shape or fill.dtype != bool):
        raise ValueError("T4 identity arrays differ in exact categorical grid")
    before = result.copy(); before[fill] = 0
    co2 = lambda x: np.isin(x, (1, 2))
    return dict(before_floor_co2_union_differing_pixels=int(np.count_nonzero(co2(before) != co2(base))),
        final_co2_union_differing_pixels=int(np.count_nonzero(co2(result) != (co2(base) | fill))),
        outside_co2_and_fill_differing_pixels=int(np.count_nonzero((result != base) & ~co2(base) & ~fill)),
        unresolved_category_differing_pixels=int(np.count_nonzero((base == 255) != (result == 255))),
        floor_flag_not_new_aqueous_pixels=int(np.count_nonzero(fill & ((base != 0) | (result != 1))))), before


def build_reference_zones(facies_path, zones_path):
    """Use the same frozen reference geometry as earlier descriptive reports."""
    with np.load(facies_path, allow_pickle=False) as data: facies = data["facies"]
    with np.load(zones_path, allow_pickle=False) as data: zones = data["zones"]
    if facies.shape != zones.shape or __import__("hashlib").sha256(zones.tobytes()).hexdigest() != shared.ZONE_ARRAY_SHA:
        raise ValueError("Frozen reference zone array changed")
    rows = np.broadcast_to(np.arange(facies.shape[0], dtype=np.int32)[:, None], facies.shape)
    last = np.maximum.accumulate(np.where(np.isin(facies, (1, 2)), rows, -1), axis=0)
    depth = (rows-last).astype(np.float32) * np.float32(ZONE_POLICY["reference_pixel_area_mm2"] ** .5)
    cap = (last >= 0) & (depth >= 0) & (depth <= 40)
    reference = np.zeros(facies.shape, np.uint8)
    reference[(zones == 4) & cap] = 1
    reference[(zones == 4) & ~cap] = 2
    reference[np.isin(zones, (2, 3, 5)) & cap] = 3
    reference[np.isin(zones, (2, 3, 5)) & ~cap] = 4
    return reference


def gas_statistics(base, result, area, zones):
    if area.shape != base.shape or zones.shape != base.shape:
        raise ValueError("Gas zones/area differ from the exact phase grid")
    selections = {"total": np.ones(base.shape, bool), **{name: zones == code
        for name, code in (("l1_cap", 1), ("l1_column", 2), ("l2_caps", 3), ("l2_columns", 4))}}
    return {name: {zone + "_cm2": float(area[(labels == 2) & mask].sum(dtype=np.float64)) / 100
                  for zone, mask in selections.items()}
            for name, labels in (("base", base), ("t4", result))}


def target_times(rows):
    groups, selected = defaultdict(list), []
    for row in rows:
        if row["elapsed_h"] > 0: groups[row["run"]].append(row)
    for run, frames in sorted(groups.items()):
        for target in (1., 2.25, 4.):
            row = min(frames, key=lambda x: (abs(x["elapsed_h"] - target), x["frame_index"]))
            selected.append(dict(run=run, light_epoch_id=row["light_epoch_id"], frame_index=row["frame_index"],
                target_elapsed_h=target, actual_elapsed_h=row["elapsed_h"],
                time_distance_seconds=abs(row["elapsed_h"] - target) * 3600, gas=row["gas"]))
    medians = []
    for epoch in sorted({r["light_epoch_id"] for r in selected}):
        for target in (1., 2.25, 4.):
            group = [r for r in selected if r["light_epoch_id"] == epoch and r["target_elapsed_h"] == target]
            medians.append(dict(light_epoch_id=epoch, target_elapsed_h=target, run_count=len(group),
                gas={model: {zone: float(np.median([r["gas"][model][zone] for r in group]))
                    for zone in group[0]["gas"][model]} for model in ("base", "t4")}))
    return selected, medians


def lcap_annotations(path):
    if shared.digest(path) != LCAP_SHA256:
        raise ValueError("Authorized L-cap label CSV changed")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream: rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    counts = defaultdict(int)
    for row in rows:
        if row["coordinate_space"] != "normalized_native_review_canvas_16x9" or row["shape"] != "rectangle":
            raise ValueError("L-cap box lacks authorized native16x9 rectangle coordinates")
        phase = {"confirmed_only_free_gas": "gas", "confirmed_only_aqueous_co2": "aqueous"}.get(row["category"])
        if phase is None: raise ValueError("Unexpected L-cap label category")
        counts[phase] += 1
        row = dict(row, phase_class=phase, frame_index=int(row["frame_index"]), run=row["run"].casefold())
        if row["run"] not in {"ac42", "ac44", "ac58", "ac60"}: raise ValueError("Unexpected L-cap run")
        groups[(row["run"], row["frame_index"])].append(row)
    if dict(counts) != {"gas": 156, "aqueous": 137}: raise ValueError("L-cap inventory changed")
    return groups


def box_metrics(annotation, base, result, rig, canvas_wh):
    from .phase_coupled_constraint_comparison import annotation_window
    bounds, mask = annotation_window(annotation, canvas_wh=canvas_wh,
        content_box=(0, 0, rig.shape[1], rig.shape[0]))
    x0, y0, x1, y1 = bounds
    mask &= rig[y0:y1, x0:x1]
    count = int(mask.sum())
    if not count: raise ValueError("An authorized L-cap box has no rig-clipped native pixels")
    row = dict(run=annotation["run"], frame_index=annotation["frame_index"], annotation_id=annotation["annotation_id"],
        phase_class=annotation["phase_class"], annotation_pixels=count, bounds_xyxy=list(bounds),
        included_in_training=annotation["phase_class"] == "gas", independent_validation=False,
        aqueous_boxes_display_only=True)
    for model, labels in (("base", base), ("t4", result)):
        row[model + "_gas_pixels"] = int(np.count_nonzero(mask & (labels[y0:y1, x0:x1] == 2)))
        row[model + "_gas_fraction"] = row[model + "_gas_pixels"] / count
    return row


def aggregate_boxes(rows, *, by_frame):
    groups, result = defaultdict(list), []
    for row in rows:
        key = (row["run"], row["phase_class"])
        if by_frame: key += (row["frame_index"],)
        groups[key].append(row)
    for key, group in sorted(groups.items()):
        item = dict(run=key[0], phase_class=key[1], boxes=len(group), independent_validation=False,
            included_in_training=key[1] == "gas", metric="gas-box recall" if key[1] == "gas" else "below-edge gas share")
        if by_frame: item["frame_index"] = key[2]
        for model in ("base", "t4"):
            item[model + "_mean_gas_fraction"] = float(np.mean([row[model + "_gas_fraction"] for row in group]))
            item[model + "_pooled_gas_fraction"] = sum(row[model + "_gas_pixels"] for row in group) / sum(row["annotation_pixels"] for row in group)
        result.append(item)
    return result


def inspect_run(task):
    from darsia.fluidflower.phase_pixel_area import load_rig_classes
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
    from .run_ffac_t4_floor_candidate import floor_contact_classes
    cv2.setNumThreads(1)
    run, entries, base_root, reference, annotations, output = task
    candidate_root = entries[0][0].parent.parent.parent
    canvas_wh = read_json(candidate_root / "source_alias/native_frames" / run / "manifest.json")["canvas_wh"]
    frames, boxes = [], []
    for path, row in entries:
        index, sha = int(row["frame_index"]), row["source_sha256"]
        parent_path = Path(base_root) / "rig_camera_frames" / run / f"frame_{index:03d}.json"
        parent = read_json(parent_path)
        if (parent["binding"]["source"] != row["binding"]["source"] or parent["rig_domain_revision"] != row["rig_domain_revision"]
                or parent["source_sha256"] != sha or float(parent["elapsed_h"]) != float(row["elapsed_h"])
                or parent["labels_map"] != row["before_labels"] or row.get("l1_rules_applied") != []):
            raise ValueError("T4 source/time/geometry/area/base or no-L1 binding changed")
        base, new = shared.labels(parent["labels_map"]), shared.labels(row["labels_map"])
        expected = path.parent.parent.parent / "production_labels" / run / f"frame_{index:03d}.png"
        if bound_path(row["labels_map"]) != expected.resolve(): raise ValueError("T4 production map escaped established layout")
        fill = floorqa.mask(row["floor_contact_fill"], new.shape, sha, index)
        area = shared.area_map(row["rig_domain_revision"]["area_map"])
        counters, before = identities(base, new, fill)
        classes, _ = load_rig_classes(row["rig_domain_revision"])
        classes = floor_contact_classes(row, classes)
        valid = classes["rig_domain"] & ~classes["hardware_excluded"]
        observed = (classes["modelled"] | classes["filled_observed"]) & valid
        filled = (classes["filled_observed"] | classes["filled_unobserved"]) & valid
        zone, geometry = prepare_bound_floor_zone(row, classes, area, observed)
        replay = apply_floor_contact(before, elapsed_hours=float(row["elapsed_h"]), zone=zone,
            observed_support=observed, filled_support=filled, unresolved=~np.isin(before, (0, 1, 2)), pixel_area_mm2=area)
        counters.update(floor_replay_label_differing_pixels=int(np.count_nonzero(replay.labels != new)),
            floor_replay_flag_differing_pixels=int(np.count_nonzero(replay.floor_contact_fill != fill)),
            rule_B_would_change_pixels=int(row["qa"]["rule_b_changed_pixels"]),
            top_clip_would_change_pixels=int(row["qa"]["top_clip_changed_pixels"]),
            preinjection_fill_pixels=int(fill.sum()) if row["elapsed_h"] <= 0 else 0)
        native_zones = shared.native_zones(reference, row, new.shape)
        for annotation in annotations.get((run, index), []):
            if abs(float(annotation["elapsed_h"]) - float(row["elapsed_h"])) > 1e-10:
                raise ValueError("L-cap annotation belongs to another source time")
            boxes.append(dict(box_metrics(annotation, base, new, classes["rig_domain"], canvas_wh),
                source_sha256=sha, light_epoch_id=row["light_epoch_id"], elapsed_h=row["elapsed_h"]))
        cap_y, cap_x = np.where(np.isin(native_zones, (1, 3)) & classes["rig_domain"])
        cap_bounds = [int(cap_x.min()), int(cap_y.min()), int(cap_x.max()+1), int(cap_y.max()+1)] if len(cap_x) else None
        value = dict(contract=CONTRACT, run=run, frame_index=index, source_sha256=sha,
            elapsed_h=float(row["elapsed_h"]), light_epoch_id=row["light_epoch_id"], identities=counters,
            gas=gas_statistics(base, new, area, native_zones), floor=read_json(path)["floor_rule"],
            floor_geometry=geometry, timing=row.get("timing", {}), technical_all_pass=not any(counters.values()),
            cap_bounds_xyxy=cap_bounds, provenance=dict(candidate_frame_receipt=artifact(path),
                base_frame_receipt=artifact(parent_path), base_labels=parent["labels_map"], candidate_labels=row["labels_map"],
                floor_contact_fill=row["floor_contact_fill"], measured_area=row["rig_domain_revision"]["area_map"]))
        atomic_json(Path(output) / "frame_qa" / run / f"frame_{index:03d}.json", value)
        frames.append(value)
    return dict(frames=frames, boxes=boxes)


def coefficient_comparison(old, new):
    result = {}
    for name, fit in (("base", old), ("epoch", new)):
        mean, scale, coef = (np.asarray(fit[key], np.float64) for key in ("mean", "scale", "coefficients"))
        if mean.shape != scale.shape or coef.shape != mean.shape or np.any(scale <= 0) or not all(np.isfinite(v).all() for v in (mean, scale, coef)):
            raise ValueError("Coefficient comparison requires finite positive declared feature scaling")
        raw = coef/scale
        result[name] = dict(stored_standardized_fit=fit, coefficients_original_feature_units=raw.tolist(),
            intercept_original_feature_units=float(fit["intercept"] - np.dot(raw, mean)))
    return result


def training_summary(plan, pixel_receipt, regional_bundle):
    from .run_ffac_t4_floor_candidate import PARENT_ID
    pixel, regional = read_json(pixel_receipt), read_json(regional_bundle)
    dataset_asset = dict(path=pixel["dataset_receipt_path"], sha256=pixel["dataset_receipt_sha256"])
    dataset = read_json(bound_path(dataset_asset))
    base = Path(plan["binding"]["parent"])
    if base.name != PARENT_ID or artifact(regional_bundle) != plan["binding"]["inputs"]["regional_bundle"]:
        raise ValueError("T4 report lost its parent or regional bundle binding")
    old_pixel, old_regional = read_json(base / "models/g3/gas_head.json"), read_json(base / "regional_model.json")
    coefficients = {}
    for epoch, heads in plan["binding"]["heads"]["epochs"].items():
        new_pixel, new_regional = read_json(bound_path(heads["g3"])), read_json(bound_path(heads["regional"]))
        expected = pixel["epochs"][epoch]["model_sha256"]
        if expected != heads["g3"]["sha256"]: raise ValueError("T4 report pixel receipt differs from fitted heads")
        coefficients[epoch] = dict(pixel=coefficient_comparison(old_pixel["fit"], new_pixel["fit"]),
            regional=coefficient_comparison(old_regional["heads"]["gas_within_co2"], new_regional["heads"]["gas_within_co2"]))
    return dict(pixel_receipt=artifact(pixel_receipt), regional_bundle=artifact(regional_bundle),
        pixel_fit=pixel, regional_fit=regional, coefficients=coefficients, independent_validation=False,
        dataset_receipt=dataset_asset, counts=dataset["training_counts"], weights=dataset["weights"],
        exclusions_and_label_source_sha_audit=dataset,
        test_labels_read=False, scorer_run=False)


def export_production(output, frames):
    exports = []
    for row in frames:
        path = Path(output) / "regression_export/production_labels" / row["run"] / f"frame_{row['frame_index']:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        source = row["provenance"]["candidate_labels"]
        shutil.copyfile(bound_path(source), path)
        if artifact(path)["sha256"] != source["sha256"]: raise ValueError("Regression copy changed label bytes")
        exports.append(dict(run=row["run"], frame_index=row["frame_index"], source_sha256=row["source_sha256"],
            elapsed_h=row["elapsed_h"], labels=artifact(path),
            production_frame_receipt=row["provenance"]["candidate_frame_receipt"],
            floor_contact_fill=row["provenance"]["floor_contact_fill"], measured_area=row["provenance"]["measured_area"]))
    value = dict(contract="ffac-g3-t4-production-labels-export-v1", complete=True, frame_count=len(exports),
        run_count=len({r["run"] for r in exports}), frames=exports, layout="production_labels/<run>/frame_NNN.png",
        annotations_read=False, scorer_run=False, scoring="pending Claude's separately sealed scorer")
    path = Path(output) / "regression_export/export_receipt.json"; atomic_json(path, value)
    return artifact(path)


def make_figures(output, candidate, rows):
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    from PySide6.QtGui import QImage
    figures = []
    for run in ("ac58", "ac42"):
        group = [row for row in rows if row["run"] == run]
        result = min(group, key=lambda r: (abs(r["elapsed_h"] - 2.25), r["frame_index"]))
        row = read_json(bound_path(result["provenance"]["candidate_frame_receipt"]))
        photo = pad_rig_photograph(load_display_image(Path(candidate) / "source_alias", run, result["frame_index"], mode="natural"), row["rig_domain_revision"])
        photo = photo.convertToFormat(QImage.Format.Format_RGB888)
        rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width()*3].reshape(photo.height(), photo.width(), 3).copy()
        base, new = shared.labels(row["before_labels"]), shared.labels(row["labels_map"])
        fill = floorqa.mask(row["floor_contact_fill"], new.shape, row["source_sha256"], row["frame_index"])
        if rgb.shape[:2] != new.shape: raise ValueError("Figure photo/maps differ in exact rig grid")
        bounds = result["cap_bounds_xyxy"]
        if bounds is None: raise ValueError("Required cap figure has no fixed cap zone")
        x0, y0, x1, y1 = bounds
        crop = [max(0, x0-80), max(0, y0-100), min(new.shape[1], x1+80), min(new.shape[0], y1+100)]
        panels = []
        for title, phase, flag in (("Photograph", None, None), ("Base overlay", base, None), ("T4 overlay", new, fill)):
            pixels = rgb.copy()
            if phase is not None:
                rgba = np.zeros((*phase.shape, 4), np.uint8)
                rgba[phase == 1] = [70, 220, 110, 128]; rgba[phase == 2] = [230, 75, 140, 128]
                if flag is not None: rgba = hatch_floor_contact_rgba(rgba, flag)
                alpha = rgba[..., 3:4].astype(np.float32)/255
                pixels = np.rint(pixels*(1-alpha)+rgba[..., :3]*alpha).astype(np.uint8)
            panel = Image.fromarray(pixels).crop(tuple(crop)); panel.thumbnail((1200, 800), Image.Resampling.LANCZOS)
            labelled = Image.new("RGB", (panel.width, panel.height+45), "#152029"); labelled.paste(panel, (0, 45))
            ImageDraw.Draw(labelled).text((10, 12), f"{run.upper()} {result['elapsed_h']:.4f} h - {title}", fill="white")
            panels.append(labelled)
        figure = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#152029")
        offset = 0
        for panel in panels: figure.paste(panel, (offset, 0)); offset += panel.width
        path = Path(output) / "figures" / f"{run.upper()}_frame_{result['frame_index']:03d}_base_t4_caps.png"
        path.parent.mkdir(parents=True, exist_ok=True); figure.save(path)
        figures.append(dict(run=run, frame_index=result["frame_index"], target_elapsed_h=2.25,
            actual_elapsed_h=result["elapsed_h"], time_distance_seconds=abs(result["elapsed_h"]-2.25)*3600,
            image=artifact(path), native_crop_xyxy=crop, labels_resampled=False,
            photograph_display="exact source-bound Natural; report thumbnail only"))
    return figures


def flatten_frames(rows):
    result = []
    for row in rows:
        item = {key: row[key] for key in ("run", "frame_index", "source_sha256", "elapsed_h", "light_epoch_id", "technical_all_pass")}
        item.update(row["identities"])
        for model, metrics in row["gas"].items(): item.update({model + "_gas_" + k: v for k, v in metrics.items()})
        result.append(item)
    return result


def run(args):
    from .run_ffac_t4_floor_candidate import candidate_state, SEAL, PLAN
    from .report_ffac_t3_floor_candidate import worker_budget
    started = time.perf_counter()
    root, plan, sources = candidate_state(args.candidate)
    if Path(args.base_candidate).resolve() != Path(plan["binding"]["parent"]): raise ValueError("QA base differs from T4 plan")
    seal = read_json(root / SEAL)
    if seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43, 864):
        raise ValueError("T4 QA requires a complete sealed43/864 candidate")
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError("T4 report requires an absent output namespace")
    annotations = lcap_annotations(args.lcap_csv)
    reference = build_reference_zones(args.facies_map, args.zone_map)
    training = training_summary(plan, args.pixel_receipt, args.regional_bundle)
    inputs = dict(candidate_plan=artifact(root / PLAN), candidate_seal=artifact(root / SEAL),
        lcap_csv=artifact(args.lcap_csv), pixel_receipt=artifact(args.pixel_receipt), regional_bundle=artifact(args.regional_bundle),
        facies_map=artifact(args.facies_map), zone_map=artifact(args.zone_map), zone_policy=ZONE_POLICY,
        report_code=artifact(__file__), all_runs_visible=True, scorer_run=False)
    available = {(s["run"], s["frame_index"]) for s in sources}
    if not set(annotations) <= available: raise ValueError("T4 misses authorized L-cap frames")
    by_run = defaultdict(list)
    for item in seal["frames"]:
        path = bound_path(item["receipt"]); row = read_json(path)
        if row.get("light_epoch_id") != plan["run_to_epoch"][row["run"]]: raise ValueError("Saved T4 epoch differs from project")
        by_run[row["run"]].append((path, row))
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(run, sorted(entries, key=lambda x: x[1]["frame_index"]), args.base_candidate, reference, annotations, output)
             for run, entries in sorted(by_run.items())]
    workers, budget = worker_budget(args.workers, len(tasks))
    frames, boxes = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(inspect_run, task) for task in tasks]):
            item = future.result(); frames.extend(item["frames"]); boxes.extend(item["boxes"])
    frames.sort(key=lambda r: (r["run"], r["frame_index"]))
    boxes.sort(key=lambda r: (r["run"], r["frame_index"], r["annotation_id"]))
    if len(frames) != 864 or len({r["run"] for r in frames}) != 43 or len(boxes) != 293:
        raise ValueError("T4 QA did not retain all43/864 frames and all293 L-cap boxes")
    selected, medians = target_times(frames)
    perframe, perrun = aggregate_boxes(boxes, by_frame=True), aggregate_boxes(boxes, by_frame=False)
    export = export_production(output, frames)
    figures = make_figures(output, root, frames)
    shared._csv(output / "gas_per_run_frame.csv", flatten_frames(frames))
    shared._csv(output / "lcap_per_box.csv", boxes)
    shared._csv(output / "lcap_per_frame.csv", perframe)
    shared._csv(output / "lcap_per_run.csv", perrun)
    totals = {key: sum(r["identities"][key] for r in frames) for key in frames[0]["identities"]}
    report = dict(contract=CONTRACT, complete=True, technical_all_pass=all(r["technical_all_pass"] for r in frames),
        created_utc=utc_now(), job_binding_sha256=canonical_sha256(inputs), inputs=inputs, run_count=43, frame_count=864,
        frames=frames, identity_totals=totals, training=training,
        in_sample=dict(independent_validation=False, gas_boxes_trained=True, aqueous_boxes_trained=False,
            below_edge_aqueous_display_only=True, per_box=boxes, per_frame=perframe, per_run=perrun,
            denominator_policy="all rig-clipped marked native pixels; equal box mean plus pooled pixel share"),
        target_times=selected, median_per_light_epoch=medians, regression_export=export, figures=figures,
        cost=dict(report_wall_s=time.perf_counter()-started, worker_budget=budget,
            inference_cpu_s=sum(r["timing"].get("cpu_s", 0) for r in frames),
            inference_wall_s_sum=sum(r["timing"].get("wall_s", 0) for r in frames)),
        blind_hold_runs=[], l1_rules_applied=[], test_labels_read=False, scorer_run=False,
        live_session_read=False, mass_eligible=False, bottom_mass_allowed=False, default_candidate_changed=False,
        regression_acceptance="pending Claude sealed scorer and overflow check")
    atomic_json(output / "qa.json", report)
    receipt = dict(contract=CONTRACT, complete=True, technical_all_pass=report["technical_all_pass"], inputs=inputs,
        report=artifact(output / "qa.json"), reports={path.name: artifact(path) for path in output.glob("*.csv")},
        regression_export=export, figures=figures, run_count=43, frame_count=864, test_labels_read=False, scorer_run=False,
        bottom_mass_allowed=False, mass_eligible=False, blind_hold_runs=[])
    atomic_json(output / "report_receipt.json", receipt)
    print(json.dumps(dict(complete=True, technical_all_pass=receipt["technical_all_pass"], receipt=str(output / "report_receipt.json"))), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "base-candidate", "lcap-csv", "pixel-receipt", "regional-bundle", "facies-map", "zone-map", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers <= 0: parser.error("CPU QA workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
