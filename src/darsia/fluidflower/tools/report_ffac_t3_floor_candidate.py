"""Source-bound T3 QA; training descriptions are in-sample, never test evidence.

Only the SHA-bound authorized training inventory is read. Regression frame
keys come from count manifests; their labels and the AC26/55 blind sessions
are never opened. Hidden maps are checked privately and yield pass/fail only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from darsia.fluidflower import phase_gas_t3_training as training
from darsia.fluidflower.phase_training import logistic
from darsia.fluidflower.phase_pixel_area import load_rig_classes
from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
from darsia.fluidflower.tools import report_ffac_t2_floor_candidate as oldqa
from darsia.fluidflower.tools import report_ffac_t2_l1zone_candidate as l1qa
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window

CONTRACT = "ffac-g3-t3-floor-candidate-qa-v1"
BLIND = frozenset(("ac26", "ac55"))
REGRESSION_RUNS = frozenset(("ac48", "ac53", "ac35", "ac38", "ac59", "ac22", "ac46", "ac57", "ac33"))
MAP_NAMES = ("base", "t2_floor", "t3")
GEOMETRY_POLICY = dict(l1qa.GEOMETRY_POLICY,
    application="T3 everywhere; the 60 mm disc is a report selection only",
    area="approved measured area for all cm2 and unchanged aqueous floor component minimum")


def public(rows):
    return [row for row in rows if str(row["run"]).casefold() not in BLIND]


def source_index(rows):
    result = {}
    for row in rows:
        sha = row["source_sha256"]
        if sha in result:
            raise ValueError("Ambiguous repeated training camera source")
        result[sha] = row
    return result


def identities(base, result, fill):
    if base.ndim != 2 or base.shape != result.shape or fill.shape != base.shape or fill.dtype != bool:
        raise ValueError("T3 identity arrays differ in exact categorical grid")
    before = result.copy(); before[fill] = 0
    return {
        "before_floor_co2_union_differing_pixels": int(np.count_nonzero(l1qa.co2(before) != l1qa.co2(base))),
        "final_co2_union_differing_pixels": int(np.count_nonzero(l1qa.co2(result) != (l1qa.co2(base) | fill))),
        "outside_co2_and_fill_differing_pixels": int(np.count_nonzero((result != base) & ~l1qa.co2(base) & ~fill)),
        "unresolved_category_differing_pixels": int(np.count_nonzero((result != base) & (~np.isin(base, (0, 1, 2)) | ~np.isin(result, (0, 1, 2))))),
        "floor_flag_not_new_aqueous_pixels": int(np.count_nonzero(fill & ((base != 0) | (result != 1)))),
    }, before


def gas_statistics(base, t2, result, area, reference):
    selections = {"total": np.ones(base.shape, bool), "cap_band": reference == 1,
                  "plume": reference == 2, "below_injector": reference == 3, "caps_away_l1": reference == 4}
    metrics = {name: {region + "_cm2": float(area[(labels == 2) & mask].sum(dtype=np.float64))/100
                     for region, mask in selections.items()}
               for name, labels in zip(MAP_NAMES, (base, t2, result))}
    baseline = metrics["base"]["caps_away_l1_cm2"]
    change = metrics["t3"]["caps_away_l1_cm2"] - baseline
    return metrics, {"delta_cm2": change,
        "relative_change": change/baseline if baseline else None,
        "absolute_relative_change_below_5_percent": abs(change/baseline) < .05 if baseline else change == 0,
        "acceptance": "descriptive fixed-zone proxy; Claude's sealed regression scorer remains the acceptance test"}


def target_times(rows):
    groups, result = defaultdict(list), []
    for row in public(rows):
        if row["elapsed_h"] > 0: groups[row["run"]].append(row)
    for run, frames in sorted(groups.items()):
        for kind, times, regions in (("l1_zones", (1.25, 2.25, 4., 6.), ("cap_band", "plume", "below_injector", "caps_away_l1")),
                                    ("total_gas", (1., 2.25, 4.), ("total",))):
            for target in times:
                row = min(frames, key=lambda x: (abs(x["elapsed_h"]-target), x["frame_index"]))
                result.append({"run": run, "kind": kind, "frame_index": row["frame_index"],
                    "target_elapsed_h": target, "actual_elapsed_h": row["elapsed_h"],
                    "time_distance_seconds": abs(row["elapsed_h"]-target)*3600,
                    **{name+"_gas_"+region+"_cm2": row["gas"][name][region+"_cm2"]
                       for name in MAP_NAMES for region in regions}})
    return result


def region_metrics(descriptor, maps, shared, rig, disc, near150):
    """All annotated pixels remain denominators; only promoted B-T2 is rig clipped."""
    bounds, mask = annotation_window(descriptor["payload"], canvas_wh=descriptor["canvas_wh"],
                                    content_box=(0, 0, rig.shape[1], rig.shape[0]))
    x0, y0, x1, y1 = bounds
    if descriptor["family"].startswith("bt2_"):
        mask &= rig[y0:y1, x0:x1]
    n = int(mask.sum())
    if not n:
        raise ValueError("An authorized training box has no rig-clipped native pixels")
    rows = []
    for distance, selection in (("all", np.ones(mask.shape, bool)),
            ("within60mm", disc[y0:y1, x0:x1]),
            ("60to150mm", near150[y0:y1, x0:x1] & ~disc[y0:y1, x0:x1])):
        selected = mask & selection; count = int(selected.sum())
        if not count: continue
        item = {k: descriptor[k] for k in ("region_id", "run", "family", "phase_class", "source_sha256", "frame_index", "original_frame_index")}
        item.update(distance=distance, annotation_pixels=count, bounds_xyxy=list(bounds),
                    shared_modelled_pixels=int((selected & shared[y0:y1, x0:x1]).sum()),
                    raw_box_weight=descriptor["raw_box_weight"], independent_validation=False)
        for name, labels in zip(MAP_NAMES, maps):
            part = labels[y0:y1, x0:x1]
            item[name+"_gas_pixels"] = int((selected & (part == 2)).sum())
            item[name+"_gas_fraction"] = item[name+"_gas_pixels"]/count
            item[name+"_unresolved_pixels"] = int((selected & ~np.isin(part, (0, 1, 2))).sum())
        rows.append(item)
    return rows


def aggregate_training(rows, *, by_frame=True):
    groups, result = defaultdict(list), []
    for row in public(rows):
        key = (row["family"], row["phase_class"], row["distance"], row["run"])
        if by_frame: key += (row["frame_index"],)
        groups[key].append(row)
    for key, entries in sorted(groups.items()):
        item = dict(family=key[0], phase_class=key[1], distance=key[2], run=key[3], regions=len(entries))
        if by_frame: item["frame_index"] = key[4]
        denom = sum(row["annotation_pixels"] for row in entries)
        item["metric"] = "gas recall" if key[1] == "gas" else "gas share in explicit aqueous boxes"
        for name in MAP_NAMES:
            item[name+"_equal_box_gas_fraction"] = float(np.mean([row[name+"_gas_fraction"] for row in entries]))
            item[name+"_pixel_weighted_gas_fraction"] = sum(row[name+"_gas_pixels"] for row in entries)/denom
        result.append(item)
    return result


def verified_training(dataset_root, old_extraction):
    """Read only the authorized frozen inventory; do not reopen CSVs or sessions."""
    data = oldqa.read_json(Path(dataset_root)/"dataset_receipt.json")
    if (data.get("contract") != training.CONTRACT or data.get("spec_sha256") != training.SPEC_SHA256
            or data.get("combined_regions") != 3393 or data.get("t2_regions") != 223
            or data.get("bt2_regions") != 321 or data.get("original_round1_region_count") != 871
            or data.get("old_features_and_owner_indices_bit_identical") is not True
            or data.get("frozen_co2_head_sha256") != training.CO2_SHA256
            or data.get("frozen_t2_dataset_receipt_sha256") != training.T2_DATASET_SHA256
            or any(data.get(key) is not False for key in ("rounds_2_3_4_read", "new_blind_session_labels_read", "live_session_read"))
            or data["bt2"]["annotation_csv_sha256"] != training.BT2_ANNOTATIONS_SHA256
            or data["bt2"]["session_sha256"] != training.BT2_SESSION_SHA256
            or data["t2"]["annotation_csv_sha256"] != training.t2.T2_ANNOTATIONS_SHA256):
        raise ValueError("T3 report training inventory/seals differ from authorized T3 specification")
    inventory = oldqa.verified({"path": data["region_inventory_path"], "sha256": data["region_inventory_sha256"]})
    regions = [logistic.ReviewedRegion(**row) for row in oldqa.read_json(inventory)]
    if (len(regions) != 3393 or len({r.region_id for r in regions}) != 3393
            or training.label_digest(regions) != data["combined_label_sha256"]):
        raise ValueError("T3 frozen training label digest/ordered IDs differ")
    t2ids, bt2ids = set(data["new_region_ids"]), set(data["bt2_region_ids"])
    families = [training.source_family(r, t2ids, bt2ids) for r in regions]
    counts = Counter(families)
    if (len(t2ids) != 223 or len(bt2ids) != 321 or t2ids & bt2ids
            or counts["round1"] != 871 or counts["t2_negatives"] != 223
            or counts["bt2_positives"] != 140 or counts["bt2_negatives"] != 181
            or any(r.run.casefold() in REGRESSION_RUNS for r in regions)
            or any(r.run.casefold() in BLIND for r, f in zip(regions, families) if f in ("t2_negatives", "bt2_positives", "bt2_negatives", "round1"))):
        raise ValueError("T3 exact class/ID inventory or test/new-blind exclusion failed")
    oldpath = oldqa.verified({"path": str(Path(old_extraction)/"extraction_receipt.json"),
                              "sha256": data["old_extraction_receipt_sha256"]})
    return data, regions, families, oldqa.read_json(oldpath)


def training_sources(data, regions, families, original):
    """Reconstruct original source ownership from ordered inventory and ID hashes."""
    groups = []
    older = [r for r, f in zip(regions, families) if f.startswith("g3_")]
    offset = 0
    for source in original["old_sources"]:
        members = older[offset:offset+source["regions"]]; offset += source["regions"]
        if (len(members) != source["regions"] or any(r.run != source["run"] for r in members)
                or training.digest(sorted(r.region_id for r in members)) != source["region_ids_sha256"]):
            raise ValueError("Original G3 source group ownership changed")
        groups.append((source, members, "historical"))
    if offset != len(older): raise ValueError("Original G3 source groups do not cover the frozen inventory")
    for source in original["new_labels"]["sources"]:
        groups.append((source, [r for r, f in zip(regions, families) if f == "round1" and r.run == source["run"]], "historical"))
    for family, receipt in (("t2_negatives", data["t2"]), ("bt2", data["bt2"])):
        for source in receipt["source_bindings"]:
            groups.append((source, [r for r, f in zip(regions, families) if r.run == source["run"] and (f == family or (family == "bt2" and f.startswith("bt2_")))], "rig"))
    if Counter(r.region_id for _, members, _ in groups for r in members) != Counter(r.region_id for r in regions):
        raise ValueError("Training source ownership is not exclusive and complete")
    return groups


def geometric_weight(region, native, source, mode, current=None, *, rig_domain=None):
    """Box loss mass is apportioned by literal raster overlap, not pixel count weighting."""
    if current is not None and mode == "rig":
        point, _, _ = l1qa.l1_geometry(current, tuple(current["rig_domain_revision"]["phase_shape_hw"]))
        shape = tuple(current["rig_domain_revision"]["phase_shape_hw"])
        content = (0, 0, shape[1], shape[0])
        Hbinding = {"current_source_sha256": current["source_sha256"]}
    else:
        cohort = native["v5_track1_photo_geometry"]
        declarations = [r for r in cohort["frames"] if r["frame_index"] == source["frame_index"]]
        if len(declarations) != 1: raise ValueError("Historical source has no unique bound photograph geometry")
        declaration = declarations[0]
        receipt = oldqa.read_json(oldqa.verified({"path": declaration["receipt_path"], "sha256": declaration["receipt_sha256"]}))
        if receipt["camera_jpeg_sha256"] != source["source_sha256"]:
            raise ValueError("Historical L1 geometry belongs to another camera source")
        H = np.asarray(receipt["H_ac28_to_frame_native"], float)
        point = H @ [*l1qa.L1_NATIVE_XY, 1.]
        if not np.isfinite(point).all() or abs(point[2]) < 1e-10: raise ValueError("Invalid historical homogeneous L1 point")
        point = point[:2]/point[2]
        content = tuple(native["content_box_xywh"])
        Hbinding = {"historical_photo_geometry": {"path": declaration["receipt_path"], "sha256": declaration["receipt_sha256"]}}
    bounds, mask = annotation_window(region.payload, canvas_wh=native["canvas_wh"], content_box=content)
    x0, y0, x1, y1 = bounds
    if rig_domain is not None:
        if mode != "rig" or rig_domain.dtype != bool or rig_domain.shape != shape:
            raise ValueError("B-T2 weight clipping requires the literal source rig domain")
        mask &= rig_domain[y0:y1, x0:x1]
    if not mask.any(): raise ValueError("Training weight geometry has no supported raster pixels")
    rr, cc = np.ogrid[y0:y1, x0:x1]
    near = (cc-point[0])**2+(rr-point[1])**2 <= l1qa.RADIUS_PX**2
    fraction = float((mask & near).sum()/mask.sum()) if source["elapsed_h"] >= l1qa.ACTIVATION_H else 0.
    return fraction, Hbinding


def prepare_training_tasks(data, regions, families, original, available):
    bysource = defaultdict(dict)
    for (run, index), (_path, row) in available.items():
        if row["source_sha256"] in bysource[run]: raise ValueError("Candidate source timeline is ambiguous")
        bysource[run][row["source_sha256"]] = row
    family_byid = dict(zip((r.region_id for r in regions), families))
    tasks, absent, geometry_weights = defaultdict(list), [], []
    selected_counts = Counter()
    for binding, members, mode in training_sources(data, regions, families, original):
        # Historical hidden labels are retained by fitting, but their geometry,
        # counts, distances and rates do not enter public report mathematics.
        if binding["run"].casefold() in BLIND: continue
        native = oldqa.read_json(oldqa.verified({"path": binding["native_manifest_path"], "sha256": binding["native_manifest_sha256"]}))
        last_rig_sha, rig = None, None
        for region in members:
            family = family_byid[region.region_id]
            source = native["frame_sources"][region.frame_index]
            sha = source["source_sha256"]
            if source["frame_index"] != region.frame_index: raise ValueError("Training source index is not literal")
            if mode == "rig" and region.payload["source_jpeg_sha256"] != sha: raise ValueError("Training camera SHA differs from seal")
            current = bysource[region.run].get(sha)
            if family.startswith("bt2_"):
                if current is None: raise ValueError("Promoted B-T2 source is missing from the frozen timeline")
                if sha != last_rig_sha:
                    classes, _ = load_rig_classes(current["rig_domain_revision"])
                    rig, last_rig_sha = classes["rig_domain"], sha
            else: rig = None
            frac, geom = geometric_weight(region, native, source, mode, current, rig_domain=rig)
            geometry_weights.append(dict(region_id=region.region_id, run=region.run.upper(), family=family,
                phase_class=region.phase_class, after_l1_end=source["elapsed_h"] >= l1qa.ACTIVATION_H,
                within60mm_raster_fraction=frac, raw_box_weight=region.supervision_weight,
                near60_raw_box_weight=region.supervision_weight*frac, geometry_binding=geom))
            if family.startswith("g3_"): continue
            selected_counts[family] += 1
            if current is None:
                absent.append(dict(region_id=region.region_id, run=region.run.upper(), family=family,
                    source_sha256=sha, original_frame_index=region.frame_index,
                    reason="original source outside the saved 864-frame timeline; native final-map recall unavailable; original feature-row supplement retained"))
                continue
            if list(native["canvas_wh"]) != list(current["rig_domain_revision"]["canvas_wh"]):
                raise ValueError("Training source canvas needs an explicit verified coordinate transform")
            tasks[(region.run.casefold(), int(current["frame_index"]))].append(dict(region_id=region.region_id,
                run=region.run.upper(), family=family, phase_class=region.phase_class,
                source_sha256=sha, frame_index=int(current["frame_index"]), original_frame_index=region.frame_index,
                canvas_wh=native["canvas_wh"], payload=region.payload, raw_box_weight=region.supervision_weight))
    expected_absent = {r["region_id"] for r in data["original_ac31_off_timeline"]["labels"]}
    if {r["region_id"] for r in absent} != expected_absent or len(absent) != 34:
        raise ValueError("Missing training sources differ from original 34 AC31 off-timeline sources")
    if selected_counts != {"round1": 871, "t2_negatives": 223, "bt2_positives": 140, "bt2_negatives": 181}:
        raise ValueError("Full native training report does not cover exact authorized training regions")
    return dict(tasks), absent, geometry_weights


def training_summary(data, regions, families, geometry, pixel_receipt, regional_receipt, dataset_root, candidates):
    """Public counts/loss weights only; global coefficients and fit costs are model metadata."""
    pixel, regional = oldqa.read_json(pixel_receipt), oldqa.read_json(regional_receipt)
    dataset_sha = oldqa.digest(Path(dataset_root)/"dataset_receipt.json")
    if (pixel.get("converged") is not True or pixel.get("cpu_only") is not True
            or regional.get("converged") is not True or regional.get("cpu_only") is not True
            or pixel["dataset_receipt_sha256"] != dataset_sha
            or pixel["training_label_sha256"] != data["combined_label_sha256"]
            or regional["training_label_sha256"] != data["combined_label_sha256"]
            or regional.get("g3_t3_pixel_sha256") != pixel["model_sha256"]
            or regional.get("co2_pixel_sha256") != training.CO2_SHA256
            or regional.get("all_other_regional_fields_exact") is not True):
        raise ValueError("T3 QA requires converged CPU fits on the identical frozen inventory")
    for candidate, descriptor in ((candidates[2], {"path": pixel["model_path"], "sha256": pixel["model_sha256"]}),):
        oldqa.verified(descriptor)
        if oldqa.digest(Path(candidate)/"models/g3/gas_head.json") != descriptor["sha256"]:
            raise ValueError("T3 candidate pixel head differs from fit receipt")
    if oldqa.digest(Path(candidates[2])/"regional_model.json") != regional["regional_model_sha256"]:
        raise ValueError("T3 regional model differs from its fit receipt")
    regional_models = [oldqa.read_json(Path(c)/"regional_model.json") for c in (candidates[0], candidates[2])]
    regional_gas = []
    for model in regional_models:
        regional_gas.append(model["heads"].pop("gas_within_co2"))
    if regional_models[0] != regional_models[1]:
        raise ValueError("T3 changed a frozen regional field outside the gas head")
    for candidate in candidates:
        if oldqa.digest(Path(candidate)/"models/generic/co2_presence_signal.json") != training.CO2_SHA256:
            raise ValueError("T3 report lost the frozen CO2-presence head")
    features = np.load(oldqa.verified({"path": data["feature_path"], "sha256": data["feature_sha256"]}), mmap_mode="r", allow_pickle=False)
    owners = np.load(oldqa.verified({"path": data["owner_path"], "sha256": data["owner_sha256"]}), mmap_mode="r", allow_pickle=False)
    training._check_arrays(features, owners, regions)
    _x, _y, weights, selected = logistic.balanced_pixel_view(features, owners, regions, head=training.HEAD)
    loss = np.bincount(selected, weights=weights, minlength=len(regions))
    del _x, _y, weights, selected
    sample_counts = np.bincount(owners, minlength=len(regions))
    near = {r["region_id"]: r for r in geometry}
    counts, byrun, source_weights = Counter(), defaultdict(Counter), defaultdict(lambda: defaultdict(float))
    near_weights = defaultdict(lambda: defaultdict(float))
    for i, (region, family) in enumerate(zip(regions, families)):
        if region.run.casefold() in BLIND: continue
        counts[(family, region.phase_class)] += 1; byrun[region.run.upper()][(family, region.phase_class)] += 1
        item = source_weights[family]
        item["regions"] += 1; item["sample_rows"] += int(sample_counts[i])
        item["raw_box_weight"] += region.supervision_weight
        item["balanced_direct_pixel_loss_weight"] += float(loss[i])
        target = logistic.regional_presence_target(region.phase_class, training.HEAD)
        if target is not None:
            label = "positive" if target else "negative"
            fraction = near[region.region_id]["within60mm_raster_fraction"]
            near_weights[label]["raw_box_weight"] += region.supervision_weight*fraction
            near_weights[label]["balanced_direct_pixel_loss_weight"] += float(loss[i])*fraction
    # Predictions on the frozen four-feature samples supplement the unavailable
    # AC31 sources. They are explicitly not final-map or independent evidence.
    from .train_ffac_gas_flatfield_variants import gas_probability
    heads = [oldqa.read_json(Path(c)/"models/g3/gas_head.json") for c in candidates]
    gascounts = [np.zeros(len(regions), np.float64) for _ in heads]
    for start in range(0, len(features), 131072):
        stop = min(len(features), start+131072)
        for head, totals in zip(heads, gascounts):
            scores = gas_probability(head, np.empty((stop-start, 0), np.float32), features[start:stop])
            totals += np.bincount(owners[start:stop], weights=(scores >= .7), minlength=len(regions))
    samples = []
    for i, (region, family) in enumerate(zip(regions, families)):
        if region.run.casefold() in BLIND or family.startswith("g3_"): continue
        if not sample_counts[i]: raise ValueError("Authorized training region lacks original feature rows")
        samples.append(dict(region_id=region.region_id, run=region.run.upper(), family=family,
            original_frame_index=region.frame_index, sample_rows=int(sample_counts[i]),
            **{name+"_pixel_sample_gas_fraction": float(totals[i]/sample_counts[i]) for name, totals in zip(MAP_NAMES, gascounts)}))
    return {"counts_public": [{"source": key[0], "phase_class": key[1], "regions": n} for key, n in sorted(counts.items())],
        "counts_per_run_public": [{"run": run, "source": key[0], "phase_class": key[1], "regions": n}
                                  for run, entries in sorted(byrun.items()) for key, n in sorted(entries.items())],
        "weights_by_source_public": dict(source_weights), "within60mm_after_l1_end_weights_public": dict(near_weights),
        "near_weight_policy": "Each original box weight/loss mass multiplied by its exact raster fraction in the 375px own-source disc, after 0.935h; historical native content kept, promoted B-T2 clipped to rig. Geometric attribution, not a refit or per-pixel box weighting",
        "near_weight_geometry_public": geometry,
        "old_hidden_run_inventory_present": {run.upper(): any(r.run.casefold() == run for r in regions[:3072]) for run in sorted(BLIND)},
        "pixel_coefficient_comparison": coefficient_comparison(pixel["old_fit"], pixel["new_fit"]),
        "regional_gas_fit_base": regional_gas[0], "regional_gas_fit_t3": regional_gas[1],
        "global_fit_cost": {"pixel_cpu_s": pixel["fit_cpu_seconds"], "pixel_wall_s": pixel["fit_wall_seconds"],
                            "regional_cpu_s": regional.get("fit_cpu_seconds"), "regional_wall_s": regional.get("fit_seconds")},
        "global_fit_metadata_policy": "Coefficients and fit-stage costs describe the complete intentionally unchanged historical inventory; hidden-run prediction/count/weight contributions excluded from all public QA metrics",
        "exclusions_by_session_and_label_sha": {"training_label_sources": data["training_label_sources"],
            "t2_annotation_sha256": data["t2"]["annotation_csv_sha256"], "t2_session_sha256": data["t2"]["session_sha256"],
            "promoted_bt2_annotation_sha256": data["bt2"]["annotation_csv_sha256"], "promoted_bt2_session_sha256": data["bt2"]["session_sha256"],
            "rounds_2_3_4_read": False, "new_blind_session_labels_read": False, "live_session_read": False},
        "pixel_sample_supplement_public": samples,
        "pixel_sample_scope": "in-sample original scientific feature rows at fixed pixel threshold0.7 only; T2 head supplement is not T2+floor final-map output",
        "independent_validation": False, "historical_blind_inventory_conditioning": True}


def coefficient_comparison(old, new):
    result = {"feature_names": list(training.FLATFIELD_NAMES),
              "standardized_basis": "Each fit stores its own mean and scale; coefficients below are also converted to the common original ff_log feature units"}
    for name, fit in (("base", old), ("t3", new)):
        mean, scale, coef = (np.asarray(fit[key], np.float64) for key in ("mean", "scale", "coefficients"))
        if mean.shape != scale.shape or coef.shape != mean.shape or np.any(scale <= 0) or not all(np.isfinite(v).all() for v in (mean, scale, coef)):
            raise ValueError("Coefficient comparison requires finite positive declared feature scaling")
        raw = coef/scale
        result[name] = {"stored_standardized_fit": fit, "coefficients_original_feature_units": raw.tolist(),
                        "intercept_original_feature_units": float(fit["intercept"] - np.dot(raw, mean))}
    return result


def post_floor_rechecks(row):
    """Read final-map checks, distinct from Rule B's operation report.

    The producer stores the post-floor categorical comparison in ``qa``;
    ``rule_b`` retains the unchanged barrier-rule report schema.
    """
    return {"rule_B_would_change_pixels": int(row["qa"]["rule_b_changed_pixels"]),
            "top_clip_would_change_pixels": int(row["top_clip"]["changed_pixels"])}


def inspect_run(task):
    cv2.setNumThreads(1)
    run, entries, base_root, t2_root, reference, regions, output, binding = task
    hidden = run.casefold() in BLIND
    frames, training_rows, passed, peak = [], [], True, 0
    started, cpu = time.perf_counter(), time.process_time()
    try:
        for path, row in entries:
            index, sha = int(row["frame_index"]), row["source_sha256"]
            parent_path = Path(base_root)/"rig_camera_frames"/run.upper()/f"frame_{index:03d}.json"
            t2_path = Path(t2_root)/"rig_camera_frames"/run.upper()/f"frame_{index:03d}.json"
            parent, t2row = oldqa.read_json(parent_path), oldqa.read_json(t2_path)
            if any(other["source_sha256"] != sha or float(other["elapsed_h"]) != float(row["elapsed_h"])
                    or other["binding"]["source"] != row["binding"]["source"]
                    or other["rig_domain_revision"] != row["rig_domain_revision"] for other in (parent, t2row)):
                raise ValueError("T3 source/time/geometry/area differs from parent decisions")
            if row["before_labels"] != parent["labels_map"]:
                raise ValueError("T3 before-map does not bind the exact seam base")
            maps = tuple(oldqa.labels(other["labels_map"]) for other in (parent, t2row, row))
            expected_production = path.parent.parent.parent/"production_labels"/run.upper()/f"frame_{index:03d}.png"
            if Path(row["labels_map"]["path"]).resolve() != expected_production.resolve():
                raise ValueError("T3 categorical map escaped the established production_labels layout")
            base, t2, new = maps
            shape = new.shape
            area = oldqa.area_map(row["rig_domain_revision"]["area_map"])
            fill = l1qa.mask(row["floor_contact_fill"], shape, sha, index)
            tfill = l1qa.mask(t2row["floor_contact_fill"], shape, sha, index)
            counters, before = identities(base, new, fill)
            tpre = t2.copy(); tpre[tfill] = 0
            classes, _ = load_rig_classes(row["rig_domain_revision"])
            with np.load(oldqa.verified(row["rig_domain_revision"]["classes"]), allow_pickle=False) as archive:
                classes["hardware_excluded"] = archive["hardware_excluded"].copy()
            categories, _ = l1qa.support_categories(classes, l1qa.native_photo_centre(row))
            valid = classes["rig_domain"] & ~classes["hardware_excluded"]
            observed = (classes["modelled"] | classes["filled_observed"]) & valid
            filled = (classes["filled_observed"] | classes["filled_unobserved"]) & valid
            zone, geometry = prepare_bound_floor_zone(row, classes, area, observed)
            point, disc, near = l1qa.l1_geometry(row, shape)
            replay = apply_floor_contact(before, elapsed_hours=float(row["elapsed_h"]), zone=zone,
                observed_support=observed, filled_support=filled, unresolved=~np.isin(before, (0, 1, 2)), pixel_area_mm2=area)
            counters.update(floor_replay_label_differing_pixels=int(np.count_nonzero(replay.labels != new)),
                            floor_replay_flag_differing_pixels=int(np.count_nonzero(replay.floor_contact_fill != fill)))
            parent_replay = apply_floor_contact(tpre, elapsed_hours=float(row["elapsed_h"]), zone=zone,
                observed_support=observed, filled_support=filled, unresolved=~np.isin(tpre, (0, 1, 2)), pixel_area_mm2=area)
            counters.update(t2_before_floor_union_differing_pixels=int(np.count_nonzero(l1qa.co2(tpre) != l1qa.co2(base))),
                            t2_floor_replay_flag_differing_pixels=int(np.count_nonzero(parent_replay.floor_contact_fill != tfill)),
                            t2_floor_replay_label_differing_pixels=int(np.count_nonzero(parent_replay.labels != t2)))
            floor = l1qa.floor_statistics(before, tpre, new, fill, tfill, area, categories, near, zone,
                float(row["elapsed_h"]), supported=observed | filled)
            saved = row["floor_rule"]
            counters["floor_producer_pixel_count_disagreement"] = int(floor["filled_pixels"] != saved["filled_pixels"])
            counters["floor_producer_area_disagreement"] = int(not np.isclose(floor["filled_area_cm2"], saved["filled_area_cm2"], atol=1e-8, rtol=0))
            recheck = post_floor_rechecks(row)
            passed &= l1qa.checks_pass(counters, floor, recheck)
            if hidden: continue
            for descriptor in regions.get(index, []):
                if descriptor["source_sha256"] != sha: raise ValueError("Training descriptor belongs to another source")
                training_rows.extend(region_metrics(descriptor, maps, classes["modelled"], classes["rig_domain"], disc, near))
            gas, caps = gas_statistics(*maps, area, oldqa.native_zones(reference, row, shape))
            timing, qa = row.get("timing", {}), row.get("qa", {})
            value = dict(contract=CONTRACT, run=run.upper(), frame_index=index, source_sha256=sha,
                elapsed_h=float(row["elapsed_h"]), identities=counters, floor=floor, gas=gas, caps_away_l1=caps,
                l1_point_xy=point.tolist(), floor_geometry=geometry, recheck=recheck,
                cost={"candidate_cpu_s": timing.get("cpu_s", qa.get("cpu_s")), "candidate_wall_s": timing.get("wall_s", qa.get("wall_s")),
                      "candidate_rss_bytes": timing.get("rss_bytes", qa.get("rss_bytes")),
                      "candidate_peak_working_set_bytes": timing.get("peak_working_set_bytes", qa.get("peak_working_set_bytes"))},
                provenance={"candidate_frame_receipt": oldqa.asset(path), "base_frame_receipt": oldqa.asset(parent_path),
                    "t2_frame_receipt": oldqa.asset(t2_path), "base_labels": parent["labels_map"], "t2_labels": t2row["labels_map"],
                    "candidate_labels": row["labels_map"], "floor_contact_fill": row["floor_contact_fill"],
                    "measured_area": row["rig_domain_revision"]["area_map"]}, job_binding_sha256=binding)
            oldqa._write_json(Path(output)/"frame_qa"/run/f"frame_{index:03d}.json", value)
            frames.append(value)
            try:
                import psutil
                peak = max(peak, psutil.Process().memory_info().rss)
            except ImportError: pass
    except Exception:
        if not hidden: raise
        # Never serialize hidden frame identities, errors or numeric evidence.
        return {"blind_status": {"run": run.upper(), "pass": False}}
    if hidden: return {"blind_status": {"run": run.upper(), "pass": bool(passed)}}
    return dict(frames=frames, training_rows=training_rows, all_pass=bool(passed), public_cost={"run": run.upper(),
        "report_worker_cpu_s": time.process_time()-cpu, "report_worker_wall_s": time.perf_counter()-started,
        "peak_observed_worker_rss_bytes": peak or None})


def count_inventory(validation_root, available):
    inventories, keys = [], set()
    for name, pattern, countkey in (("round2", "round2_*", "test_label_counts"),
                                   ("round3", "round3_*", "round3_counts"), ("round4", "round4b_*", "round4_counts")):
        folders = sorted((Path(validation_root)/"snapshots").glob(pattern))
        if len(folders) != 1: raise ValueError("Expected exactly one sealed regression count inventory")
        path = folders[0]/"SNAPSHOT_MANIFEST.json"; receipt = oldqa.read_json(path)
        if name == "round2":
            runs = {str(r).casefold() for r in receipt[countkey]}
            selected = {k for k in available if k[0] in runs}
            policy = "all saved frames of sealed round2 runs; count-only manifest does not name frames; explicit superset"
        else:
            selected = set()
            for key in receipt[countkey]:
                match = re.match(r"(ac\d+)\s+f(\d+)", key, re.I)
                if not match: raise ValueError("Regression count key lacks exact run/frame identity")
                selected.add((match.group(1).casefold(), int(match.group(2))))
            policy = "exact sealed run/frame count keys"
        if any(run in BLIND for run, _ in selected): raise PermissionError("Hidden T3 runs cannot enter regression exports")
        if not selected <= set(available): raise ValueError("T3 candidate misses sealed regression frames")
        inventories.append(dict(round=name, manifest=oldqa.asset(path), annotation_seal=receipt["files"]["pairwise_review_annotations.csv"],
            annotations_read=False, selection=policy, frames=[dict(run=run.upper(), frame_index=i) for run, i in sorted(selected)]))
        keys |= selected
    return inventories, keys


def export_scoring_frames(output, rows, keys, inventories):
    exports = []
    for row in public(rows):
        key = row["run"].casefold(), row["frame_index"]
        if key not in keys: continue
        directory = Path(output)/"regression_export"/key[0]/f"frame_{key[1]:03d}"
        directory.mkdir(parents=True)
        files = {}
        for name, descriptor in (("new_labels", row["provenance"]["candidate_labels"]),
                                 ("old_labels", row["provenance"]["base_labels"]),
                                 ("floor_contact_fill", row["provenance"]["floor_contact_fill"])):
            source = oldqa.verified(descriptor); target = directory/(name+source.suffix)
            shutil.copyfile(source, target)
            if oldqa.digest(target) != descriptor["sha256"]: raise IOError("Regression copy SHA differs")
            files[name] = oldqa.asset(target)
        # Keep the established T2 regression file layout; descriptive aliases
        # bind the same immutable bytes rather than creating another map.
        files["production_labels"] = files["new_labels"]
        files["base_labels"] = files["old_labels"]
        receipt = directory/"receipt.json"
        oldqa._write_json(receipt, dict(contract="ffac-g3-t3-production-labels-export-v1", run=row["run"], frame_index=key[1],
            elapsed_h=row["elapsed_h"], source_sha256=row["source_sha256"], files=files,
            production_frame_receipt=row["provenance"]["candidate_frame_receipt"], measured_area=row["provenance"]["measured_area"],
            annotations_read=False, scorer_run=False))
        exports.append(oldqa.asset(receipt))
    if len(exports) != len(keys): raise ValueError("Regression production-label export is incomplete")
    return dict(inventories=inventories, frames=exports, exported_frame_count=len(exports),
                annotations_read=False, scorer_run=False, scoring="pending Claude's separately sealed scorer",
                scope="supplementary SHA-bound copies; candidate production_labels/<run>/frame_NNN.png is the primary production layout; no new scorer compatibility claimed")


def figure_requests():
    return [(run, target) for run in ("AC28", "AC47", "AC30") for target in (2.25, 4.)]


def make_figures(output, candidate, rows):
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    from PySide6.QtGui import QImage
    figures = []
    for run, target in figure_requests():
        if run.casefold() in BLIND: raise PermissionError("Hidden figures are prohibited")
        group = [r for r in public(rows) if r["run"] == run]
        if not group: raise ValueError("Required public figure run is missing")
        result = min(group, key=lambda r: abs(r["elapsed_h"]-target))
        row = oldqa.read_json(result["provenance"]["candidate_frame_receipt"]["path"])
        photo = pad_rig_photograph(load_display_image(Path(candidate)/"source_alias", run.casefold(), result["frame_index"], mode="natural"), row["rig_domain_revision"])
        photo = photo.convertToFormat(QImage.Format.Format_RGB888)
        rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width()*3].reshape(photo.height(), photo.width(), 3).copy()
        base, new = oldqa.labels(row["before_labels"]), oldqa.labels(row["labels_map"])
        fill = l1qa.mask(row["floor_contact_fill"], new.shape, row["source_sha256"], row["frame_index"])
        if rgb.shape[:2] != new.shape: raise ValueError("Figure photo/maps differ in exact rig grid")
        x, y = result["l1_point_xy"]
        crop = [max(0, int(x-1100)), max(0, int(y-1100)), min(new.shape[1], int(x+1100)), min(new.shape[0], int(y+700))]
        panels = []
        for title, phase, flag in (("Photograph", None, None), ("Base overlay", base, None), ("T3 overlay", new, fill)):
            pixels = rgb.copy()
            if phase is not None:
                rgba = np.zeros((*phase.shape, 4), np.uint8)
                rgba[phase == 1] = [70, 220, 110, 128]; rgba[phase == 2] = [230, 75, 140, 128]
                if flag is not None: rgba = hatch_floor_contact_rgba(rgba, flag)
                alpha = rgba[..., 3:4].astype(np.float32)/255
                pixels = np.rint(pixels*(1-alpha)+rgba[..., :3]*alpha).astype(np.uint8)
            panel = Image.fromarray(pixels).crop(tuple(crop)); panel.thumbnail((1000, 700), Image.Resampling.LANCZOS)
            labelled = Image.new("RGB", (panel.width, panel.height+45), "#152029"); labelled.paste(panel, (0, 45))
            ImageDraw.Draw(labelled).text((10, 12), f"{run} {result['elapsed_h']:.4f} h — {title}", fill="white")
            panels.append(labelled)
        figure = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "#152029")
        offset = 0
        for panel in panels: figure.paste(panel, (offset, 0)); offset += panel.width
        path = Path(output)/"figures"/f"{run}_frame_{result['frame_index']:03d}_base_t3.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        figure.save(path)
        figures.append(dict(run=run, frame_index=result["frame_index"], target_elapsed_h=target,
            actual_elapsed_h=result["elapsed_h"], time_distance_seconds=abs(result["elapsed_h"]-target)*3600,
            image=oldqa.asset(path), native_crop_xyxy=crop, labels_resampled=False,
            photograph_display="exact source-bound Natural; report thumbnail only", source_frame_receipt=result["provenance"]["candidate_frame_receipt"]))
    return figures


def flatten_frames(rows):
    result = []
    for row in public(rows):
        item = {k: row[k] for k in ("run", "frame_index", "source_sha256", "elapsed_h")}
        for group in ("identities", "recheck", "cost", "caps_away_l1"):
            item.update({group+"_"+k: v for k, v in row[group].items()})
        for name, metrics in row["gas"].items(): item.update({name+"_gas_"+k: v for k, v in metrics.items()})
        item.update({"floor_"+k: v for k, v in row["floor"].items() if not isinstance(v, (dict, list))})
        for group in ("by_support", "by_L1_distance"):
            for name, metrics in row["floor"][group].items(): item.update({"floor_"+group+"_"+name+"_"+k: v for k, v in metrics.items()})
        result.append(item)
    return result


def worker_budget(requested, runs):
    workers = max(1, min(requested, os.cpu_count() or 1, runs))
    receipt = dict(requested_workers=requested, logical_cpu_bound=os.cpu_count(), worker_memory_estimate_bytes=3<<30)
    try:
        import psutil
        memory = psutil.virtual_memory(); reserve = max(2<<30, int(memory.total*.05))
        affinity = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else (os.cpu_count() or 1)
        workers = min(workers, affinity, max(1, int(max(0, memory.available-reserve)//(3<<30))))
        receipt.update(affinity_cpu_bound=affinity, available_memory_bytes=memory.available, memory_reserve_bytes=reserve)
    except ImportError: workers = min(workers, 2)
    return workers, receipt


def run(args):
    from .run_ffac_t3_floor_candidate import candidate_state
    output = Path(args.output)
    if output.exists(): raise FileExistsError("Choose a new immutable T3 QA output directory")
    _root, plan, declared_sources = candidate_state(args.candidate)
    available = oldqa.frame_receipts(args.candidate, "rig_camera_frames")
    if not args.allow_partial and (len(available) != 864 or len({key[0] for key in available}) != 43):
        raise ValueError("Full T3 QA requires exactly 43 runs and 864 committed frames")
    declared = {(row["run"].casefold(), int(row["frame_index"])): row for row in declared_sources}
    if (not set(available) <= set(declared) or (not args.allow_partial and set(available) != set(declared))
            or any(row["binding"]["source"] != declared[key] for key, (_path, row) in available.items())):
        raise ValueError("T3 report source receipts differ from the immutable source inventory")
    data, regions, families, original = verified_training(args.dataset_root, args.old_extraction)
    tasks, absent, geometry = prepare_training_tasks(data, regions, families, original, available)
    training_report = training_summary(data, regions, families, geometry, args.pixel_receipt, args.regional_receipt,
        args.dataset_root, (args.base_candidate, args.t2_candidate, args.candidate))
    inventories, keys = count_inventory(args.validation_root, available)
    inputs = {name: str(getattr(args, name)) for name in ("candidate", "base_candidate", "t2_candidate")}
    inputs.update({name: oldqa.asset(getattr(args, name)) for name in ("pixel_receipt", "regional_receipt", "facies_map", "zone_map")})
    inputs.update(dataset_receipt=oldqa.asset(Path(args.dataset_root)/"dataset_receipt.json"), report_code=oldqa.asset(__file__),
                  geometry_policy=GEOMETRY_POLICY, hidden_runs=sorted(BLIND), candidate_plan=oldqa.asset(Path(args.candidate)/"t3_floor_plan.json"))
    binding = hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest()
    reference = oldqa.build_reference_zones(args.facies_map, args.zone_map)
    byrun = defaultdict(list)
    for (run_id, _index), entry in available.items(): byrun[run_id].append(entry)
    workers, budget = worker_budget(args.workers, len(byrun))
    output.mkdir(parents=True)
    oldqa._write_json(output/"report_binding.json", dict(contract=CONTRACT, job_binding_sha256=binding, inputs=inputs))
    frames, boxrows, blind, costs, statuses = [], [], [], [], []
    jobs = [(run_id, entries, args.base_candidate, args.t2_candidate, reference,
        {index: descriptors for (r, index), descriptors in tasks.items() if r == run_id}, output, binding)
        for run_id, entries in byrun.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inspect_run, task): task[0] for task in jobs}
        for future in as_completed(pending):
            result = future.result()
            if "blind_status" in result:
                blind.append(result["blind_status"]); print(json.dumps({"withheld_run_check": "completed"}), flush=True)
            else:
                frames.extend(result["frames"]); boxrows.extend(result["training_rows"]); costs.append(result["public_cost"])
                statuses.append(result["all_pass"]); print(json.dumps({"public_run": pending[future].upper(), "public_runs_completed": len(statuses)}), flush=True)
    frames.sort(key=lambda row: (row["run"], row["frame_index"]))
    expected = {row["region_id"] for entries in tasks.values() for row in entries}
    if {row["region_id"] for row in boxrows} != expected: raise ValueError("T3 in-sample final maps missed exact authorized source IDs")
    export = export_scoring_frames(output, frames, keys, inventories)
    figures = make_figures(output, args.candidate, frames) if not args.no_figures else []
    targets = target_times(frames); perframe = aggregate_training(boxrows); perrun = aggregate_training(boxrows, by_frame=False)
    for filename, rows in (("frames_public.csv", flatten_frames(frames)), ("gas_target_times_public.csv", targets),
            ("training_regions_public.csv", boxrows), ("training_per_frame_public.csv", perframe),
            ("training_per_run_public.csv", perrun), ("training_counts_public.csv", training_report["counts_public"]),
            ("training_counts_per_run_public.csv", training_report["counts_per_run_public"]),
            ("training_pixel_sample_supplement_public.csv", training_report["pixel_sample_supplement_public"])):
        oldqa._csv(output/filename, rows)
    totals = {key: sum(row["identities"][key] for row in public(frames)) for key in frames[0]["identities"]}
    floor_totals = {key: sum(row["floor"][key] for row in public(frames)) for key in (
        "filled_pixels", "filled_area_cm2", "preinjection_fill_pixels", "filled_under_gas_pixels", "unexplained_changed_floor_columns",
        "changed_fill_without_trigger_phase_change_columns", "changed_fill_explained_by_trigger_phase_columns", "changed_fill_explained_by_component_eligibility_columns")}
    cost = {key: sum(row["cost"][key] for row in public(frames) if row["cost"][key] is not None) for key in ("candidate_cpu_s", "candidate_wall_s")}
    cost.update({key: max((row["cost"][key] for row in public(frames) if row["cost"][key] is not None), default=None)
                 for key in ("candidate_rss_bytes", "candidate_peak_working_set_bytes")})
    cost.update(report_workers=workers, worker_resource_budget=budget, runs=costs,
                report_worker_cpu_s=sum(row["report_worker_cpu_s"] for row in costs),
                peak_observed_report_worker_rss_bytes=max((row["peak_observed_worker_rss_bytes"] for row in costs if row["peak_observed_worker_rss_bytes"] is not None), default=None),
                scope="public inference/report runs only; hidden costs excluded; unavailable candidate RSS remains null")
    report = dict(contract=CONTRACT, complete=True, technical_all_pass=all(statuses) and all(row["pass"] for row in blind),
        created_utc=datetime.now(timezone.utc).isoformat(), job_binding_sha256=binding, inputs=inputs,
        public_run_count=len(statuses), public_frame_count=len(frames), public_frames=frames,
        public_identity_totals=totals, public_floor_totals=floor_totals, target_times=targets, public_cost=cost,
        training=training_report, in_sample={"independent_validation": False, "training_evidence_only": True,
            "per_region": boxrows, "per_frame": perframe, "per_run": perrun, "off_timeline_not_evaluated": absent,
            "denominator_policy": "All marked pixels; B-T2 clipped to exact rig; unknown coverage visible; radial selections use fixed 0.16 mm/px",
            "original_inventory": {"bt2_positive": 140, "bt2_negative": 181, "t2_negative": 223, "round1": 871}},
        regression_export=export, figures=figures, blind_status=sorted(blind, key=lambda row: row["run"]),
        regression_production_labels={"root": str(Path(args.candidate)/"production_labels"), "layout": "<run>/frame_NNN.png",
            "source_inventory_sha256": plan["binding"]["inputs"]["source_inventory_sha256"],
            "all_committed_maps_checked": True, "annotations_read": False, "scorer_run": False},
        blind_policy="AC26/55 pass/fail only; excluded before all public numerical aggregation, exports and figures; AC30/49/54 now authorized training/public",
        training_inventory_read=True, original_training_csv_or_session_read=False, test_labels_read=False,
        new_blind_session_labels_read=False, live_session_read=False, scorer_run=False,
        mass_eligible=False, bottom_mass_allowed=False, default_candidate_changed=False, automatic_activation=False)
    oldqa._write_json(output/"qa_public.json", report)
    receipt = {key: report[key] for key in ("contract", "complete", "technical_all_pass", "created_utc", "job_binding_sha256", "inputs",
        "public_run_count", "public_frame_count", "public_identity_totals", "public_floor_totals", "public_cost", "blind_status",
        "blind_policy", "test_labels_read", "new_blind_session_labels_read", "live_session_read", "scorer_run", "mass_eligible", "bottom_mass_allowed",
        "default_candidate_changed", "automatic_activation")}
    receipt.update(report=oldqa.asset(output/"qa_public.json"), regression_export=export,
                   regression_production_labels=report["regression_production_labels"], figures=figures,
                   reports={path.name: oldqa.asset(path) for path in sorted(output.glob("*.csv"))})
    oldqa._write_json(output/"report_receipt.json", receipt)
    print(json.dumps({"complete": True, "technical_all_pass": report["technical_all_pass"], "receipt": str(output/"report_receipt.json")}), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "base-candidate", "t2-candidate", "dataset-root", "pixel-receipt", "regional-receipt", "old-extraction",
                 "validation-root", "facies-map", "zone-map", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0: parser.error("CPU QA workers must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
