"""Source-bound T4c placement QA without reading annotations or running scorers.

The historical AC58/60 rectangles come exclusively from the already sealed
T4b QA. Candidate maps, source geometry and calibration receipts are verified
as saved; no head is fitted and no candidate is selected by this report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import report_ffac_t2_floor_candidate as shared
from . import report_ffac_t2_l1zone_candidate as floorqa
from . import report_ffac_t4b_floor_candidate as t4bqa
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now

CONTRACT = "ffac-g3-t4c-floor-candidate-qa-v1"
CANDIDATE_CONTRACT = "ffac-g3-t4c-floor-candidate-v1"
PLAN = "t4c_floor_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
SPEC_SHA256 = "53aad00f20bec56092c160477210613e7849f50467303aa0efd5243d90337daf"
CALIBRATOR_SHA256 = "6d2d36db574965362cb6d183e48e6ac1218dc01eb67e1400cdf2071453597ef4"
T4B_QA_SHA256 = "a5f7b9913c8fdf7f06c94d6b393ccc3647a5a1d4f5c28befc6d1baa986514fea"
FIXED_CONFIGURATION = dict(regional_threshold=.6, pixel_threshold=.7, minimum_mm2=2.5, connectivity=4)
ZONE_POLICY = dict(t4bqa.ZONE_POLICY)
ZONE_CODES = dict(l1_cap=1, l1_column=2, l2_caps=3, l2_columns=4)
PLACEMENTS = ("P1", "P2", "P3")
artifact, bound_path, read_json = shared.asset, shared.verified, shared.read_json
identities = t4bqa.identities
build_reference_zones = t4bqa.build_reference_zones


def parse_candidates(value):
    """Accept one comma-separated placement=root mapping or several arguments."""
    values = [value] if isinstance(value, str) else value
    result = {}
    for part in values:
        for entry in part.split(","):
            name, separator, path = entry.partition("=")
            if not separator or name not in PLACEMENTS or not path or name in result:
                raise ValueError("Candidates must be unique P1=path,P2=path,P3=path entries")
            result[name] = Path(path).resolve()
    if not result:
        raise ValueError("At least one calibrated placement candidate is required")
    return {name: result[name] for name in PLACEMENTS if name in result}


def stored_candidate(root, placement, base):
    """Verify saved receipts, without applying a current T4b code binding."""
    root, base = Path(root).resolve(), Path(base).resolve()
    plan, seal = read_json(root / PLAN), read_json(root / SEAL)
    inputs = plan["binding"]["inputs"]
    if (plan.get("contract") != CANDIDATE_CONTRACT or plan.get("placement") != placement
            or plan.get("output_root") != str(root) or Path(plan["binding"]["parent"]).resolve() != base
            or plan.get("job_sha256") != canonical_sha256(plan["binding"])
            or plan.get("pipeline_sha256") != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))
            or inputs.get("fixed_configuration") != FIXED_CONFIGURATION
            or inputs.get("l1_rules_applied") != [] or inputs.get("blind_hold_runs") != []
            or not np.isfinite(plan.get("delta", np.nan))
            or (root / "rig_bottom_review_policy.json").exists()):
        raise ValueError("T4c saved plan lost its placement, parent, fixed policy or immutable binding")
    spec = inputs.get("spec", plan["binding"].get("spec"))
    if not isinstance(spec, dict) or spec.get("sha256") != SPEC_SHA256:
        raise ValueError("T4c saved plan differs from the latest approved specification")
    sources = plan["source_inventory"]
    keys = [(s["run"], int(s["frame_index"])) for s in sources]
    if (len(keys), len(set(keys)), len({r for r, _ in keys})) != (864, 864, 43):
        raise ValueError("T4c report requires the exact unique 43/864 source inventory")
    for name in ("source_inventory", "native_sources", "source_context_plans", "parent_frame_receipts"):
        if canonical_sha256(plan[name]) != inputs[name + "_sha256"]:
            raise ValueError("T4c saved source or parent receipt inventory changed")
    epochs = plan["run_to_epoch"]
    if (set(epochs) != {r for r, _ in keys}
            or Counter(epochs.values()) != {"rig_light": 31, "new_lamps": 12}
            or {r for r, e in epochs.items() if e == "new_lamps"} != {f"ac{i}" for i in range(50, 62)}):
        raise ValueError("T4c changed the project 31/12 light epoch table")
    bound_path(inputs["light_epoch_table"])
    for name in ("parent_seal", "parent_plan", "parent_registration"):
        bound_path(inputs[name])
    if (seal.get("complete") is not True or seal.get("bottom_mass_allowed") is not False
            or (seal.get("run_count"), seal.get("frame_count")) != (43, 864)):
        raise ValueError("T4c report requires a complete sealed 43/864 bottom-blocked candidate")
    if (seal.get("source_plan_sha256") != artifact(root / PLAN)["sha256"]
            or seal.get("pipeline_sha256") != plan["pipeline_sha256"]
            or seal.get("model_identity") != plan["model_identity"]
            or seal.get("placement") != placement or seal.get("delta") != plan["delta"]):
        raise ValueError("T4c candidate seal differs from its exact sealed plan and placement")
    entries = {}
    source_by_key = dict(zip(keys, sources))
    for item in seal["frames"]:
        path = bound_path(item["receipt"]); row = read_json(path)
        key = (row["run"], int(row["frame_index"]))
        if (key in entries or key not in source_by_key or row.get("complete") is not True
                or (item["run"], item["frame_index"]) != key
                or row["binding"]["source"] != source_by_key[key]
                or row["binding"].get("pipeline_sha256") != plan["pipeline_sha256"]
                or row.get("placement") != placement or row.get("delta") != plan["delta"]
                or row.get("light_epoch_id") != epochs[key[0]] or row.get("l1_rules_applied") != []):
            raise ValueError("T4c seal/frame lost its exact source, epoch or placement binding")
        expected = root / "production_labels" / key[0] / f"frame_{key[1]:03d}.png"
        if bound_path(row["labels_map"]).resolve() != expected.resolve():
            raise ValueError("T4c production map escaped the established layout")
        entries[key] = (path, row)
    if set(entries) != set(keys):
        raise ValueError("T4c seal does not contain every source frame exactly once")
    return plan, entries


def historical_boxes(path):
    """Read only saved QA geometry; never resolve its annotation CSV descriptor."""
    path = Path(path)
    value = read_json(path)
    if path.name == "report_receipt.json":
        if (value.get("contract") != t4bqa.CONTRACT or value.get("complete") is not True
                or value.get("technical_all_pass") is not True
                or (value.get("run_count"), value.get("frame_count")) != (43, 864)):
            raise ValueError("Historical box geometry requires the successful sealed T4b QA")
        qa_asset = value["report"]
        path = bound_path(qa_asset)
        value = read_json(path)
    qa_asset = artifact(path)
    if (qa_asset["sha256"] != T4B_QA_SHA256 or value.get("contract") != t4bqa.CONTRACT
            or value.get("complete") is not True or value.get("technical_all_pass") is not True
            or (value.get("run_count"), value.get("frame_count")) != (43, 864)):
        raise ValueError("Historical T4b box receipt or exact QA SHA changed")
    groups, ids = defaultdict(list), set()
    for item in value["in_sample"]["per_box"]:
        if item["run"] not in ("ac58", "ac60"):
            continue
        item = dict(item)
        key = (item["run"], int(item["frame_index"]))
        identity = (*key, item["annotation_id"])
        bounds = item.get("bounds_xyxy")
        if (identity in ids or item.get("phase_class") not in ("gas", "aqueous")
                or not isinstance(bounds, list) or len(bounds) != 4
                or any(type(v) is not int for v in bounds) or not bounds[0] < bounds[2]
                or not bounds[1] < bounds[3] or item.get("annotation_pixels", 0) <= 0
                or len(item.get("source_sha256", "")) != 64):
            raise ValueError("Saved L-cap rectangle geometry is incomplete or ambiguous")
        ids.add(identity); groups[key].append(item)
    boxes = [box for group in groups.values() for box in group]
    if (len(boxes) != 134 or Counter(b["run"] for b in boxes) != {"ac58": 60, "ac60": 74}
            or Counter(b["phase_class"] for b in boxes) != {"gas": 74, "aqueous": 60}
            or set(groups) != {("ac58", 4), ("ac58", 8), ("ac60", 4), ("ac60", 8)}):
        raise ValueError("Historical AC58/60 L-cap box inventory changed")
    return dict(groups), qa_asset


def box_metrics(saved, maps, rig, source_sha256, elapsed_h):
    if saved["source_sha256"] != source_sha256 or abs(saved["elapsed_h"] - elapsed_h) > 1e-10:
        raise ValueError("Historical L-cap rectangle belongs to another source or time")
    x0, y0, x1, y1 = saved["bounds_xyxy"]
    if not 0 <= x0 < x1 <= rig.shape[1] or not 0 <= y0 < y1 <= rig.shape[0]:
        raise ValueError("Historical L-cap rectangle escaped its exact native canvas")
    mask = rig[y0:y1, x0:x1]
    count = int(mask.sum())
    if count != saved["annotation_pixels"]:
        raise ValueError("Historical L-cap rectangle denominator differs from the sealed geometry")
    result = {key: saved[key] for key in ("run", "frame_index", "annotation_id", "phase_class",
        "annotation_pixels", "bounds_xyxy", "source_sha256", "elapsed_h", "light_epoch_id")}
    result.update(independent_validation=False, historical_gas_boxes_in_t4b_training=saved["phase_class"] == "gas",
        aqueous_boxes_display_only=True, new_annotations_read=False)
    for name, labels in maps.items():
        if labels.shape != rig.shape:
            raise ValueError("Box metrics require the exact shared phase grid")
        pixels = int(np.count_nonzero(mask & (labels[y0:y1, x0:x1] == 2)))
        result[name + "_gas_pixels"] = pixels
        result[name + "_gas_fraction"] = pixels / count
    if result["base_gas_pixels"] != saved["base_gas_pixels"]:
        raise ValueError("Historical L-cap base count differs from the exact saved source")
    return result


def aggregate_boxes(rows, *, by_frame):
    groups, result = defaultdict(list), []
    for row in rows:
        key = (row["run"], row["phase_class"])
        if by_frame: key += (row["frame_index"],)
        groups[key].append(row)
    models = [key[:-11] for key in rows[0] if key.endswith("_gas_pixels")] if rows else []
    for key, group in sorted(groups.items()):
        value = dict(run=key[0], phase_class=key[1], boxes=len(group), independent_validation=False,
            metric="gas-box recall" if key[1] == "gas" else "below-edge gas share")
        if by_frame: value["frame_index"] = key[2]
        for name in models:
            value[name + "_mean_gas_fraction"] = float(np.mean([r[name + "_gas_fraction"] for r in group]))
            value[name + "_pooled_gas_fraction"] = sum(r[name + "_gas_pixels"] for r in group) / sum(r["annotation_pixels"] for r in group)
        result.append(value)
    return result


def selections(zones):
    return dict(total=np.ones(zones.shape, bool), **{name: zones == code for name, code in ZONE_CODES.items()})


def gas_statistics(maps, area, zones):
    if area.shape != zones.shape or any(labels.shape != zones.shape for labels in maps.values()):
        raise ValueError("Gas zones/area differ from the exact phase grid")
    return {name: {zone + "_cm2": float(area[(labels == 2) & mask].sum(dtype=np.float64)) / 100
        for zone, mask in selections(zones).items()} for name, labels in maps.items()}


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
                gas={name: {zone: float(np.median([r["gas"][name][zone] for r in group]))
                    for zone in metrics} for name, metrics in group[0]["gas"].items()}))
    return selected, medians


def p2_identities(base, result, fill, allowed, barrier, forbidden, zones):
    before = result.copy(); before[fill] = 0
    additions = (before == 2) & (base != 2)
    expected = (base == 1) & np.isin(zones, (1, 3)) & ~barrier & ~forbidden
    return dict(p2_allowed_mask_differing_pixels=int(np.count_nonzero(allowed != expected)),
        p2_additions_outside_allowed_pixels=int(np.count_nonzero(additions & ~allowed)),
        p2_additions_outside_caps_pixels=int(np.count_nonzero(additions & ~np.isin(zones, (1, 3)))),
        p2_additions_in_barrier_pixels=int(np.count_nonzero(additions & barrier)),
        p2_additions_in_frozen_rule_B_forbidden_pixels=int(np.count_nonzero(additions & forbidden)),
        p2_base_gas_removed_pixels=int(np.count_nonzero((base == 2) & (before != 2))),
        p2_changed_outside_additions_and_floor_pixels=int(np.count_nonzero((before != base) & ~additions)))


def added_statistics(maps, area, zones, distances, fraction):
    base = maps["base"]
    result = {}
    for name, labels in maps.items():
        if name == "base": continue
        additions = (labels == 2) & (base != 2)
        result[name] = {}
        for zone, mask in selections(zones).items():
            selected = additions & mask
            count = int(selected.sum())
            distance, frac = distances[selected], fraction[selected]
            finite = distance[np.isfinite(distance)]
            result[name][zone] = dict(added_gas_pixels=count,
                added_gas_area_cm2=float(area[selected].sum(dtype=np.float64)) / 100,
                distance_to_base_gas_mm_median=float(np.median(finite)) if finite.size else None,
                distance_to_base_gas_mm_p90=float(np.percentile(finite, 90)) if finite.size else None,
                distance_without_finite_base_gas_pixels=count-int(finite.size),
                base_gas_fraction_median=float(np.median(frac)) if count else None,
                base_gas_fraction_p10=float(np.percentile(frac, 10)) if count else None)
    return result


def calibration_summary(plan):
    asset = plan["binding"]["inputs"]["calibrated_shift"]
    receipt = read_json(bound_path(asset))
    if (receipt.get("contract") != "ffac-t4c-overflow-calibrated-shift-v1"
            or receipt.get("placement") != plan["placement"] or receipt.get("delta") != plan["delta"]
            or not np.isfinite(receipt.get("delta_star_unrounded", np.nan))
            or receipt["delta"] != round(receipt["delta_star_unrounded"], 2)):
        raise ValueError("T4c calibrated shift differs from the script's rounded result")
    script_output = receipt["script_output"]
    output = read_json(bound_path(script_output))
    if (output.get("contract") != "ffac-new-lamp-shift-calibration-v1"
            or output.get("script_sha256") != CALIBRATOR_SHA256):
        raise ValueError("T4c calibration lost the frozen script output contract")
    counts = artifact(bound_path(receipt["counts"]))
    returned = output["variants"][plan["placement"]]
    if (counts["sha256"] != output.get("counts_sha256")
            or returned.get("delta_star") != receipt["delta_star_unrounded"]
            or returned.get("split_half") != receipt.get("split_half")
            or returned.get("split_half_pass") != receipt.get("split_half_pass")
            or output.get("target_rig_median_seff") != receipt.get("target_rig_median_seff")):
        raise ValueError("T4c shift receipt differs from the frozen script's exact returned values or counts")
    result = dict(calibrated_shift=asset, shift_receipt=receipt, script_output=script_output,
        frozen_script_output=output, delta_star_unrounded=receipt["delta_star_unrounded"], delta=receipt["delta"],
        target_rig_median_seff=receipt.get("target_rig_median_seff"),
        split_half=receipt.get("split_half"), split_half_pass=receipt.get("split_half_pass"),
        calibration_is_validation=False, additional_tuning=False)
    if plan["placement"] in ("P1", "P2"):
        cutoff = float(np.exp(-np.logaddexp(0., -(np.log(.7 / .3) - receipt["delta"]))))
        result.update(equivalent_pixel_threshold=cutoff,
            threshold_interpretation=("combined optical plus fixed neighbourhood score; not a plain optical probability cutoff"
                if plan["placement"] == "P2" else "base optical probability cutoff before the shift"))
    return result


def floor_classes(row):
    from darsia.fluidflower.phase_pixel_area import load_rig_classes
    classes, _ = load_rig_classes(row["rig_domain_revision"])
    with np.load(bound_path(row["rig_domain_revision"]["classes"]), allow_pickle=False) as archive:
        hardware = archive["hardware_excluded"].copy()
    if hardware.dtype != bool or hardware.shape != classes["rig_domain"].shape:
        raise ValueError("Floor hardware mask differs from its exact rig grid")
    return dict(classes, hardware_excluded=hardware)


def frozen_p2_masks(plan, parent, run, index, shape):
    from darsia.fluidflower.phase_l1_neighbourhood_gas import select_frozen_base_rule_b, frozen_rule_b_forbidden_mask
    producer_asset = plan["source_context_plans"][run]["source_producer_frames"][f"{run}/frame_{index:03d}"]
    producer = read_json(bound_path(producer_asset))
    if producer["binding"]["source"] != parent["binding"]["source"]:
        raise ValueError("P2 frozen Rule B geometry belongs to another source")
    geometry = producer["rule_b_geometry"]["asset"]
    with np.load(bound_path(geometry), allow_pickle=False) as archive:
        barrier, components = archive["barrier_mask"].copy(), archive["open_components"].copy()
    if barrier.shape != shape:
        raise ValueError("P2 frozen Rule B geometry differs from the phase canvas")
    decision, selection = select_frozen_base_rule_b(parent, producer)
    forbidden, audit = frozen_rule_b_forbidden_mask(components, barrier, decision)
    return barrier, forbidden, dict(source_producer_frame=producer_asset, geometry=geometry,
        spatial_decision=audit, base_decision_selection=selection)


def inspect_run(task):
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
    from darsia.fluidflower.phase_gas_t4c_placements import base_gas_fraction, distance_to_base_gas_mm
    cv2.setNumThreads(1)
    started_cpu, started = time.process_time(), time.perf_counter()
    run, grouped, plans, base_root, reference, boxes, output, spacing = task
    names = list(grouped)
    positive = [entry[1] for entry in grouped[names[0]] if entry[1]["elapsed_h"] > 0]
    target_index = min(positive, key=lambda row: (abs(row["elapsed_h"] - 2.25), row["frame_index"]))["frame_index"]
    frames, box_rows, added_rows = [], [], []
    indexed = {name: {row["frame_index"]: (path, row) for path, row in entries} for name, entries in grouped.items()}
    for index in sorted(indexed[names[0]]):
        key = f"{run}/frame_{index:03d}"
        parent_asset = plans[names[0]]["parent_frame_receipts"][key]
        parent_path = bound_path(parent_asset); parent = read_json(parent_path)
        if parent_path.resolve() != (Path(base_root) / "rig_camera_frames" / run / f"frame_{index:03d}.json").resolve():
            raise ValueError("T4c QA parent frame escaped its exact base")
        base = shared.labels(parent["labels_map"])
        first = indexed[names[0]][index][1]
        classes = floor_classes(first)
        valid = classes["rig_domain"] & ~classes["hardware_excluded"]
        observed = (classes["modelled"] | classes["filled_observed"]) & valid
        filled = (classes["filled_observed"] | classes["filled_unobserved"]) & valid
        area = shared.area_map(first["rig_domain_revision"]["area_map"])
        native_zones = shared.native_zones(reference, first, base.shape)
        maps, counters, provenance, timings, floor_geometry = {"base": base}, {}, {}, {}, {}
        for name in names:
            path, row = indexed[name][index]
            sha, elapsed = row["source_sha256"], float(row["elapsed_h"])
            if (parent["binding"]["source"] != row["binding"]["source"]
                    or parent["rig_domain_revision"] != row["rig_domain_revision"]
                    or parent["source_sha256"] != sha or float(parent["elapsed_h"]) != elapsed
                    or parent["labels_map"] != row["before_labels"] or row.get("l1_rules_applied") != []
                    or plans[name]["parent_frame_receipts"][key] != parent_asset
                    or row["light_epoch_id"] != first["light_epoch_id"]):
                raise ValueError("T4c source/time/geometry/area/base or no-L1 binding changed")
            new = shared.labels(row["labels_map"])
            fill = floorqa.mask(row["floor_contact_fill"], new.shape, sha, index)
            count, before = identities(base, new, fill)
            zone, geometry = prepare_bound_floor_zone(row, classes, area, observed)
            replay = apply_floor_contact(before, elapsed_hours=elapsed, zone=zone, observed_support=observed,
                filled_support=filled, unresolved=~np.isin(before, (0, 1, 2)), pixel_area_mm2=area)
            count.update(floor_replay_label_differing_pixels=int(np.count_nonzero(replay.labels != new)),
                floor_replay_flag_differing_pixels=int(np.count_nonzero(replay.floor_contact_fill != fill)),
                rule_B_would_change_pixels=int(row["qa"]["rule_b_changed_pixels"]),
                top_clip_would_change_pixels=int(row["qa"]["top_clip_changed_pixels"]),
                preinjection_fill_pixels=int(fill.sum()) if elapsed <= 0 else 0,
                rig_light_gas_differing_pixels=int(np.count_nonzero((base == 2) != (new == 2))) if row["light_epoch_id"] == "rig_light" else 0,
                rig_light_labels_outside_floor_differing_pixels=int(np.count_nonzero((base != new) & ~fill)) if row["light_epoch_id"] == "rig_light" else 0)
            extra = {}
            if name == "P2" and row["light_epoch_id"] == "new_lamps":
                allowed, barrier, forbidden = (floorqa.mask(row[field], new.shape, sha, index)
                    for field in ("p2_allowed", "p2_barrier", "p2_rule_b_forbidden"))
                source_barrier, source_forbidden, audit = frozen_p2_masks(plans[name], parent, run, index, base.shape)
                count.update(p2_identities(base, new, fill, allowed, barrier, forbidden, native_zones),
                    p2_barrier_geometry_differing_pixels=int(np.count_nonzero(barrier != source_barrier)),
                    p2_rule_B_geometry_differing_pixels=int(np.count_nonzero(forbidden != source_forbidden)))
                extra = dict(p2_allowed=row["p2_allowed"], p2_barrier=row["p2_barrier"],
                    p2_rule_b_forbidden=row["p2_rule_b_forbidden"], frozen_rule_b=audit)
            maps[name], counters[name], timings[name], floor_geometry[name] = new, count, row.get("timing", {}), geometry
            provenance[name] = dict(candidate_frame_receipt=artifact(path), candidate_labels=row["labels_map"],
                floor_contact_fill=row["floor_contact_fill"], gas_inference=row.get("gas_inference", {}), **extra)
        for saved in boxes.get((run, index), []):
            box_rows.append(box_metrics(saved, maps, classes["rig_domain"], first["source_sha256"], float(first["elapsed_h"])))
        if index == target_index:
            query = np.zeros(base.shape, bool)
            for name in names: query |= (maps[name] == 2) & (base != 2)
            H = first["binding"]["source"]["rig_geometry"]["H_ac28_to_frame_native"]
            origin = first["rig_domain_revision"]["native_origin_xy"]
            distances = distance_to_base_gas_mm(base, H, origin, reference_spacing_mm=spacing, query_mask=query)
            fraction = base_gas_fraction(base)
            added_rows.append(dict(run=run, frame_index=index, source_sha256=first["source_sha256"],
                light_epoch_id=first["light_epoch_id"], target_elapsed_h=2.25, actual_elapsed_h=float(first["elapsed_h"]),
                time_distance_seconds=abs(float(first["elapsed_h"])-2.25)*3600,
                placements=added_statistics(maps, area, native_zones, distances, fraction),
                distance_geometry=dict(H_ac28_to_frame_native=H, native_origin_xy=origin,
                    reference_spacing_mm=spacing, pixel_centres="integer native canvas centres; no additional half-pixel offset"),
                fraction_policy=dict(kernel_hw=[51, 51], border="OpenCV default", computed_once_from="sealed base gas",
                    summary_weighting="native added pixels; no area weighting"), eligibility_uses_distance=False))
        yy, xx = np.where(np.isin(native_zones, (1, 3)) & classes["rig_domain"])
        cap_bounds = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if xx.size else None
        value = dict(contract=CONTRACT, run=run, frame_index=index, source_sha256=first["source_sha256"],
            elapsed_h=float(first["elapsed_h"]), light_epoch_id=first["light_epoch_id"], identities=counters,
            gas=gas_statistics(maps, area, native_zones), timing=timings, floor_geometry=floor_geometry,
            technical_all_pass=not any(value for values in counters.values() for value in values.values()),
            cap_bounds_xyxy=cap_bounds, provenance=dict(base_frame_receipt=parent_asset,
                base_labels=parent["labels_map"], measured_area=first["rig_domain_revision"]["area_map"], placements=provenance))
        atomic_json(Path(output) / "frame_qa" / run / f"frame_{index:03d}.json", value)
        frames.append(value)
    return dict(frames=frames, boxes=box_rows, additions=added_rows,
        cost=dict(run=run, report_cpu_s=time.process_time()-started_cpu, report_wall_s=time.perf_counter()-started))


def export_production(output, frames, names):
    result = {}
    for name in names:
        exports = []
        for row in frames:
            path = Path(output) / "regression_export" / name / "production_labels" / row["run"] / f"frame_{row['frame_index']:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            provenance = row["provenance"]["placements"][name]
            source = provenance["candidate_labels"]
            shutil.copyfile(bound_path(source), path)
            if artifact(path)["sha256"] != source["sha256"]:
                raise ValueError("Regression copy changed saved production label bytes")
            exports.append(dict(run=row["run"], frame_index=row["frame_index"], source_sha256=row["source_sha256"],
                elapsed_h=row["elapsed_h"], labels=artifact(path), production_frame_receipt=provenance["candidate_frame_receipt"],
                floor_contact_fill=provenance["floor_contact_fill"], measured_area=row["provenance"]["measured_area"]))
        value = dict(contract="ffac-g3-t4c-production-labels-export-v1", placement=name, complete=True,
            frame_count=len(exports), run_count=len({r["run"] for r in exports}), frames=exports,
            layout="production_labels/<run>/frame_NNN.png", annotations_read=False, scorer_run=False,
            scoring="pending Claude's separately sealed scorer")
        path = Path(output) / "regression_export" / name / "export_receipt.json"
        atomic_json(path, value); result[name] = artifact(path)
    return result


def make_figures(output, candidates, rows):
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    from PySide6.QtGui import QImage
    figures = []
    for run in ("ac58", "ac60"):
        result = min((r for r in rows if r["run"] == run and r["elapsed_h"] > 0),
            key=lambda r: (abs(r["elapsed_h"]-2.25), r["frame_index"]))
        first_name = next(iter(candidates))
        row = read_json(bound_path(result["provenance"]["placements"][first_name]["candidate_frame_receipt"]))
        photo = pad_rig_photograph(load_display_image(candidates[first_name] / "source_alias", run,
            result["frame_index"], mode="natural"), row["rig_domain_revision"])
        photo = photo.convertToFormat(QImage.Format.Format_RGB888)
        rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width()*3].reshape(photo.height(), photo.width(), 3).copy()
        if result["cap_bounds_xyxy"] is None:
            raise ValueError("Required cap figure has no fixed cap zone")
        x0, y0, x1, y1 = result["cap_bounds_xyxy"]
        crop = [max(0, x0-80), max(0, y0-100), min(rgb.shape[1], x1+80), min(rgb.shape[0], y1+100)]
        panels = []
        for title, descriptor, flag in [("Photograph", None, None), ("Base overlay", result["provenance"]["base_labels"], None)] + [
            (name+" overlay", p["candidate_labels"], p["floor_contact_fill"]) for name, p in result["provenance"]["placements"].items()]:
            pixels = rgb.copy()
            if descriptor is not None:
                phase = shared.labels(descriptor)
                if phase.shape != rgb.shape[:2]: raise ValueError("Figure photograph/maps differ in exact grid")
                rgba = np.zeros((*phase.shape, 4), np.uint8)
                rgba[phase == 1] = [70, 220, 110, 128]; rgba[phase == 2] = [230, 75, 140, 128]
                if flag is not None:
                    rgba = hatch_floor_contact_rgba(rgba, floorqa.mask(flag, phase.shape, row["source_sha256"], row["frame_index"]))
                alpha = rgba[..., 3:4].astype(np.float32)/255
                pixels = np.rint(pixels*(1-alpha)+rgba[..., :3]*alpha).astype(np.uint8)
            panel = Image.fromarray(pixels).crop(tuple(crop)); panel.thumbnail((1000, 800), Image.Resampling.LANCZOS)
            labelled = Image.new("RGB", (panel.width, panel.height+45), "#152029"); labelled.paste(panel, (0, 45))
            ImageDraw.Draw(labelled).text((10, 12), f"{run.upper()} {result['elapsed_h']:.4f} h - {title}", fill="white")
            panels.append(labelled)
        figure = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#152029")
        offset = 0
        for panel in panels: figure.paste(panel, (offset, 0)); offset += panel.width
        path = Path(output) / "figures" / f"{run.upper()}_frame_{result['frame_index']:03d}_base_t4c_caps.png"
        path.parent.mkdir(parents=True, exist_ok=True); figure.save(path)
        figures.append(dict(run=run, frame_index=result["frame_index"], target_elapsed_h=2.25,
            actual_elapsed_h=result["elapsed_h"], time_distance_seconds=abs(result["elapsed_h"]-2.25)*3600,
            image=artifact(path), native_crop_xyxy=crop, labels_resampled=False,
            photograph_display="exact source-bound Natural; report thumbnail only", panels=["photograph", "base", *candidates]))
    return figures


def flatten_frames(rows):
    result = []
    for row in rows:
        item = {key: row[key] for key in ("run", "frame_index", "source_sha256", "elapsed_h", "light_epoch_id", "technical_all_pass")}
        for name, values in row["identities"].items(): item.update({name+"_"+key: value for key, value in values.items()})
        for name, values in row["gas"].items(): item.update({name+"_gas_"+key: value for key, value in values.items()})
        result.append(item)
    return result


def run(args):
    from .report_ffac_t3_floor_candidate import worker_budget
    started, started_cpu = time.perf_counter(), time.process_time()
    candidates = parse_candidates(args.candidates)
    output, base = Path(args.output).resolve(), Path(args.base_candidate).resolve()
    if output.exists(): raise FileExistsError("T4c report requires an absent output namespace")
    boxes, boxes_asset = historical_boxes(args.t4b_qa)
    reference = build_reference_zones(args.facies_map, args.zone_map)
    maps = dict(facies_map=artifact(args.facies_map), zone_map=artifact(args.zone_map))
    plans, entries, calibrations = {}, {}, {}
    for name, root in candidates.items():
        plan, frames = stored_candidate(root, name, base)
        frozen = plan["binding"]["inputs"]["frozen_zone_policy"]
        if frozen != dict(policy=ZONE_POLICY, **maps):
            raise ValueError("T4c report differs from the candidate's frozen cap zone policy/maps")
        plans[name], entries[name], calibrations[name] = plan, frames, calibration_summary(plan)
    first = next(iter(plans))
    if any(plans[name]["source_inventory"] != plans[first]["source_inventory"]
            or plans[name]["run_to_epoch"] != plans[first]["run_to_epoch"] for name in plans):
        raise ValueError("T4c placements do not share the exact source inventory and epoch table")
    if not set(boxes) <= set(entries[first]): raise ValueError("T4c misses historical L-cap frames")
    scale_asset = artifact(base / "physical_scale_measurement.json")
    scale = read_json(bound_path(scale_asset))
    factor = scale.get("linear_factor")
    if (scale.get("physical_scale_approved") is not True or scale.get("bottom_mass_allowed") is not False
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError("Distance QA requires the approved measured global linear scale")
    spacing = ZONE_POLICY["reference_pixel_area_mm2"] ** .5 * factor
    inputs = dict(candidates={name: dict(plan=artifact(root / PLAN), seal=artifact(root / SEAL)) for name, root in candidates.items()},
        base_candidate=str(base), historical_t4b_qa=boxes_asset, **maps, zone_policy=ZONE_POLICY,
        distance_scale=scale_asset, report_code=artifact(__file__), all_runs_visible=True,
        annotation_files_read=False, historical_boxes_only=True, scorer_run=False)
    tasks = []
    for run_name in sorted(plans[first]["run_to_epoch"]):
        grouped = {name: sorted((value for key, value in frames.items() if key[0] == run_name), key=lambda x: x[1]["frame_index"])
            for name, frames in entries.items()}
        tasks.append((run_name, grouped, plans, base, reference, boxes, output, spacing))
    output.mkdir(parents=True, exist_ok=False)
    workers, budget = worker_budget(args.workers, len(tasks))
    frames, box_rows, additions, costs = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(inspect_run, task) for task in tasks]):
            result = future.result(); frames.extend(result["frames"]); box_rows.extend(result["boxes"])
            additions.extend(result["additions"]); costs.append(result["cost"])
    frames.sort(key=lambda r: (r["run"], r["frame_index"])); box_rows.sort(key=lambda r: (r["run"], r["frame_index"], r["annotation_id"]))
    additions.sort(key=lambda r: r["run"])
    if len(frames) != 864 or len({r["run"] for r in frames}) != 43 or len(box_rows) != 134 or len(additions) != 43:
        raise ValueError("T4c QA did not retain all 43/864 frames and 134 historical AC58/60 boxes")
    selected, medians = target_times(frames)
    perframe, perrun = aggregate_boxes(box_rows, by_frame=True), aggregate_boxes(box_rows, by_frame=False)
    exports, figures = export_production(output, frames, candidates), make_figures(output, candidates, frames)
    shared._csv(output / "gas_per_run_frame.csv", flatten_frames(frames))
    shared._csv(output / "lcap_per_box.csv", box_rows); shared._csv(output / "lcap_per_frame.csv", perframe)
    shared._csv(output / "lcap_per_run.csv", perrun)
    shared._csv(output / "added_gas_2h25_by_zone.csv", [dict(run=row["run"], frame_index=row["frame_index"],
        actual_elapsed_h=row["actual_elapsed_h"], placement=name, zone=zone, **metrics)
        for row in additions for name, values in row["placements"].items() for zone, metrics in values.items()])
    totals = {name: {key: sum(r["identities"][name].get(key, 0) for r in frames)
        for key in sorted({k for row in frames for k in row["identities"][name]})} for name in candidates}
    report = dict(contract=CONTRACT, complete=True, technical_all_pass=all(r["technical_all_pass"] for r in frames),
        created_utc=utc_now(), job_binding_sha256=canonical_sha256(inputs), inputs=inputs, run_count=43, frame_count=864,
        placements=list(candidates), calibration=calibrations, frames=frames, identity_totals=totals,
        target_times=selected, median_per_light_epoch=medians, added_gas_at_2h25=additions,
        in_sample=dict(independent_validation=False, historical_geometry_only=True, annotation_files_read=False,
            historical_gas_boxes_in_t4b_training=True, aqueous_boxes_display_only=True,
            per_box=box_rows, per_frame=perframe, per_run=perrun,
            denominator_policy="unchanged saved rig-clipped native rectangle pixels; equal box mean and pooled pixel share"),
        regression_exports=exports, figures=figures,
        cost=dict(report_wall_s=time.perf_counter()-started, report_parent_cpu_s=time.process_time()-started_cpu,
            report_worker_cpu_s=sum(r["report_cpu_s"] for r in costs), per_run=costs, worker_budget=budget,
            inference_cpu_s={name: sum(row["timing"][name].get("cpu_s", 0) for row in frames) for name in candidates},
            inference_wall_s_sum={name: sum(row["timing"][name].get("wall_s", 0) for row in frames) for name in candidates},
            calibration={name: value["shift_receipt"].get("cost", {}) for name, value in calibrations.items()}),
        blind_hold_runs=[], l1_rules_applied=[], training_performed=False, test_labels_read=False,
        annotation_files_read=False, live_session_read=False, scorer_run=False,
        mass_eligible=False, bottom_mass_allowed=False, default_candidate_changed=False,
        regression_acceptance="pending Claude's pre-registered regression, time-transfer, split-half and placement checks")
    atomic_json(output / "qa.json", report)
    receipt = dict(contract=CONTRACT, complete=True, technical_all_pass=report["technical_all_pass"], inputs=inputs,
        report=artifact(output / "qa.json"), reports={path.name: artifact(path) for path in output.glob("*.csv")},
        regression_exports=exports, figures=figures, placements=list(candidates), run_count=43, frame_count=864,
        annotation_files_read=False, test_labels_read=False, scorer_run=False, bottom_mass_allowed=False, mass_eligible=False)
    atomic_json(output / "report_receipt.json", receipt)
    print(json.dumps(dict(complete=True, technical_all_pass=receipt["technical_all_pass"], receipt=str(output / "report_receipt.json"))), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True)
    for name in ("base-candidate", "t4b-qa", "facies-map", "zone-map", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers <= 0: parser.error("CPU QA workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
