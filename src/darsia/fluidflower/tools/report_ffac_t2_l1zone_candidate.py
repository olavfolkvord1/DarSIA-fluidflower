"""Read-only §4 QA of a stored T2-at-L1 / floor-contact candidate.

No annotations, sessions, training regions, scorer or learned heads are opened.
The 223 in-sample boxes come from sealed cached raster geometry. Public numbers,
including totals and costs, exclude AC30/49/54; those return pass/fail only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from darsia.fluidflower.phase_pixel_area import load_rig_classes
from darsia.fluidflower.phase_ffac_floor_contact import prepare_bound_floor_zone
from darsia.fluidflower.tools import report_ffac_t2_floor_candidate as oldqa

CONTRACT = "ffac-t2-l1-zone-floor-candidate-qa-v1"
BLIND = oldqa.BLIND
L1_NATIVE_XY = (2786.0, 2897.0)
RADIUS_PX = 375.0
ACTIVATION_H = .935
SCORER_MM_PER_PX = .16
GEOMETRY_POLICY = {
    "L1_reference_native_xy": list(L1_NATIVE_XY),
    "zone_radius_px": RADIUS_PX, "activation_elapsed_h_inclusive": ACTIVATION_H,
    "distance150mm": "radial image distance from own-source L1 point; scorer fixed 0.16 mm/px (937.5 px), distinct from measured area and projective floor metric",
    "support": "observed_native_photo: raw-supported within bound native centre; recovered_raw_strips: raw-supported outside centre; extrapolated: filled_unobserved",
    "area": "approved measured area for cm2; original source-bound minimum_area_map only for the inherited 2.5 mm2 zone-edge gas minimum",
    "zone_regions": oldqa.ZONE_POLICY,
}


def co2(labels):
    return (labels == 1) | (labels == 2)


def public(rows):
    return [row for row in rows if row["run"].casefold() not in BLIND]


def l1_geometry(row, shape):
    H = np.asarray(row["binding"]["source"]["rig_geometry"]["H_ac28_to_frame_native"], np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("L1 QA requires the exact current-source homography")
    point = H @ [*L1_NATIVE_XY, 1.]
    if not np.isfinite(point).all() or abs(point[2]) < 1e-10:
        raise ValueError("L1 source transform has an invalid homogeneous point")
    point = point[:2] / point[2] - np.asarray(row["rig_domain_revision"]["native_origin_xy"])
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    d2 = (cc - point[0]) ** 2 + (rr - point[1]) ** 2
    return point, d2 <= RADIUS_PX ** 2, d2 <= (150 / SCORER_MM_PER_PX) ** 2


def cross_edge(zone):
    boundary = np.zeros(zone.shape, bool)
    delta = zone[1:] != zone[:-1]
    boundary[1:] |= delta; boundary[:-1] |= delta
    delta = zone[:, 1:] != zone[:, :-1]
    boundary[:, 1:] |= delta; boundary[:, :-1] |= delta
    return boundary


def identities(base, t2, result, t2_fill, fill, fragment, zone, active, minimum_area):
    """Independently reconstruct the stored-label composite and edge minimum."""
    arrays = (base, t2, result, t2_fill, fill, fragment, zone, minimum_area)
    if base.ndim != 2 or any(value.shape != base.shape for value in arrays):
        raise ValueError("L1 QA inputs differ in full phase grid")
    if any(mask.dtype != bool for mask in (t2_fill, fill, fragment, zone)):
        raise ValueError("L1 QA masks must be exact boolean grids")
    if not np.isfinite(minimum_area).all() or np.any(minimum_area <= 0):
        raise ValueError("Original inherited gas-minimum area must be positive")
    tpre, before_floor = t2.copy(), result.copy()
    tpre[t2_fill] = 0; before_floor[fill] = 0
    composed = base.copy()
    if active:
        selection = zone & co2(base)
        composed[selection] = tpre[selection]
    # Use the full four-connected gas graph, including fragments crossing the
    # disc boundary. No ROI crop is allowed to alter component membership.
    n, components = cv2.connectedComponents((composed == 2).astype(np.uint8), connectivity=4)
    area = np.bincount(components.ravel(), weights=minimum_area.ravel(), minlength=n)
    touching = np.zeros(n, bool)
    if active:
        touching[np.unique(components[cross_edge(zone)])] = True
    selected = touching & (area < 2.5)
    selected[0] = False
    expected_fragment = selected[components]
    expected = composed.copy(); expected[expected_fragment] = 1
    counters = {
        "base_t2_domain_differing_pixels": int(np.count_nonzero((base == 255) != (tpre == 255))),
        "base_t2_co2_union_differing_pixels": int(np.count_nonzero(co2(base) != co2(tpre))),
        "inactive_before_floor_differing_pixels": int(np.count_nonzero(before_floor != base)) if not active else 0,
        "active_outside_zone_differing_pixels": int(np.count_nonzero((before_floor != base) & ~zone & ~fragment)) if active else 0,
        "active_inside_zone_differing_pixels": int(np.count_nonzero((before_floor != tpre) & zone & ~fragment)) if active else 0,
        "composite_before_floor_differing_pixels": int(np.count_nonzero(before_floor != expected)),
        "zone_edge_fragment_flag_differing_pixels": int(np.count_nonzero(fragment != expected_fragment)),
        "fragment_pixels_not_aqueous": int(np.count_nonzero(fragment & (before_floor != 1))),
        "final_co2_union_differing_pixels": int(np.count_nonzero(co2(result) != (co2(base) | fill))),
        "final_outside_domain_differing_pixels": int(np.count_nonzero((base == 255) != (result == 255))),
        "floor_flag_not_new_aqueous_pixels": int(np.count_nonzero(fill & ((base != 0) | (result != 1)))),
        "t2_floor_flag_not_new_aqueous_pixels": int(np.count_nonzero(t2_fill & ((base != 0) | (t2 != 1)))),
    }
    fragments = {"pixels": int(fragment.sum()), "components": int(selected.sum()),
                 "original_minimum_area_mm2": float(minimum_area[fragment].sum(dtype=np.float64)),
                 "maximum_component_original_area_mm2": float(area[selected].max()) if selected.any() else 0.}
    return counters, fragments, before_floor, tpre


def gas_statistics(base, t2, result, area, zone, reference_zones):
    metrics = {}
    for name, value in (("base", base), ("t2_floor", t2), ("combined", result)):
        gas = value == 2
        metrics[name] = {
            "total_cm2": float(area[gas].sum(dtype=np.float64)) / 100,
            "inside_l1_disc_cm2": float(area[gas & zone].sum(dtype=np.float64)) / 100,
            "outside_l1_disc_cm2": float(area[gas & ~zone].sum(dtype=np.float64)) / 100,
        }
        for region, code in (("cap_band", 1), ("plume", 2), ("below_injector", 3)):
            metrics[name][region + "_cm2"] = float(area[gas & (reference_zones == code)].sum(dtype=np.float64)) / 100
    return metrics


def support_categories(classes, content_box):
    raw = classes["modelled"] | classes["filled_observed"]
    centre = np.zeros(raw.shape, bool)
    x, y, w, h = map(int, content_box)
    if x < 0 or y < 0 or x + w > raw.shape[1] or y + h > raw.shape[0]:
        raise ValueError("Native photo centre escaped its bound phase grid")
    centre[y:y+h, x:x+w] = True
    categories = {"observed_native_photo": raw & centre, "recovered_raw_strips": raw & ~centre,
                  "extrapolated": classes["filled_unobserved"]}
    if np.any(raw & categories["extrapolated"]):
        raise ValueError("Source support categories overlap")
    return categories, raw


def native_photo_centre(row):
    """Use the literal centre/origin of the bound scientific source descriptor."""
    descriptor = row["full_extended_scientific"]
    h, w = descriptor["native_centre"]["native_shape_hw"]
    ox, oy = descriptor["native_origin_xy"]
    if (descriptor["source_sha256"] != row["source_sha256"]
            or [ox, oy] != row["rig_domain_revision"]["native_origin_xy"]
            or any(type(x) is not int for x in (h, w, ox, oy)) or min(h, w) <= 0):
        raise ValueError("Native photo centre has another source or coordinate origin")
    return [-ox, -oy, w, h]


def floor_statistics(before, tpre, result, fill, t2_fill, area, categories, near150, floor_zone, elapsed, *, supported=None):
    q = oldqa.floor_qa(before, result, fill, area, elapsed)
    row, col = floor_zone.base_canvas_y, np.arange(result.shape[1])
    in_canvas = floor_zone.floor_in_canvas & (row >= floor_zone.y0) & (row < floor_zone.y1)
    raw = categories["observed_native_photo"] | categories["recovered_raw_strips"]
    observable = np.zeros(result.shape[1], bool)
    valid_cols = col[in_canvas]
    observable[valid_cols] = floor_zone.sand[row[valid_cols]-floor_zone.y0, valid_cols] & raw[row[valid_cols], valid_cols]
    filled_cols = fill.any(0)
    def detail(mask):
        columns = mask.any(0)
        denominator = int(observable.sum())
        return {"pixels": int(mask.sum()), "area_cm2": float(area[mask].sum(dtype=np.float64)) / 100,
                "filled_observable_floor_columns": int((columns & observable).sum()),
                "share_of_all_observable_floor_columns": float((columns & observable).sum() / denominator) if denominator else None}
    q.update(observable_floor_columns=int(observable.sum()),
             filled_observable_floor_columns=int((filled_cols & observable).sum()),
             filled_observable_floor_column_share=float((filled_cols & observable).sum()/observable.sum()) if observable.any() else None,
             by_support={name: detail(fill & mask) for name, mask in categories.items()},
             by_L1_distance={"within150mm": detail(fill & near150), "beyond150mm": detail(fill & ~near150)},
             uncategorised_fill_pixels=int(np.count_nonzero(fill & ~np.logical_or.reduce(list(categories.values())))),
             combined_only_fill_pixels=int(np.count_nonzero(fill & ~t2_fill)),
             t2_only_fill_pixels=int(np.count_nonzero(t2_fill & ~fill)),
             changed_fill_without_trigger_phase_change_columns=0,
             changed_fill_explained_by_trigger_phase_columns=0,
             changed_fill_explained_by_component_eligibility_columns=0,
             unexplained_changed_floor_columns=0,
             component_eligibility_evidence=[],
             floor_roi_y0=int(floor_zone.y0))
    changed_columns = np.flatnonzero((fill != t2_fill).any(0))
    components = None
    # The unchanged rule also requires a 4-connected supported aqueous
    # component >=2.5 mm2. A phase change elsewhere can disconnect that
    # component without changing this column's first aqueous pixel.
    if supported is None:
        supported = np.logical_or.reduce(list(categories.values()))
    if supported.dtype != bool or supported.shape != before.shape:
        raise ValueError("Floor attribution requires exact source-supported pixels")
    for c in changed_columns:
        top = min(result.shape[0]-1, int(row[c]))
        candidates = np.flatnonzero(co2(before[floor_zone.y0:top+1, c]))
        other = np.flatnonzero(co2(tpre[floor_zone.y0:top+1, c]))
        if not len(candidates) or not len(other) or candidates[-1] != other[-1]:
            q["changed_fill_without_trigger_phase_change_columns"] += 1
            q["unexplained_changed_floor_columns"] += 1
            continue
        y = int(floor_zone.y0+candidates[-1])
        if {int(before[y,c]),int(tpre[y,c])} == {1,2}:
            q["changed_fill_explained_by_trigger_phase_columns"] += 1
            continue
        q["changed_fill_without_trigger_phase_change_columns"] += 1
        if before[y,c] != 1 or tpre[y,c] != 1 or not supported[y,c]:
            q["unexplained_changed_floor_columns"] += 1
            continue
        if components is None:
            membership = ((before==1) != (tpre==1)) & supported
            components=[]
            for labels in (before,tpre):
                aqueous=(labels==1) & supported
                count, ids=cv2.connectedComponents(aqueous.astype(np.uint8),connectivity=4)
                sizes=np.bincount(ids.ravel(),weights=area.ravel(),minlength=count)
                sizes[0]=0
                affected=np.zeros(count,bool)
                affected[np.unique(ids[membership & aqueous])]=True
                components.append((ids,sizes,affected))
        a,b = (int(values[0][y,c]) for values in components)
        aa,bb=float(components[0][1][a]),float(components[1][1][b])
        crossed=(aa+1e-10 >= 2.5) != (bb+1e-10 >= 2.5)
        affected=bool(components[0][2][a] or components[1][2][b])
        if not crossed or not affected:
            q["unexplained_changed_floor_columns"] += 1
            continue
        q["changed_fill_explained_by_component_eligibility_columns"] += 1
        q["component_eligibility_evidence"].append(dict(column=int(c),trigger_row=y,
            combined_component_area_mm2=aa,t2_component_area_mm2=bb,
            minimum_mm2=2.5,changed_supported_membership_in_component=True))
    return q


def cached_t2_boxes(report, dataset, *, expected_count=223):
    """Read only existing raster geometry and the sealed ID inventory."""
    ids = dataset["new_region_ids"]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("T2 sealed inventory must contain exactly 223 unique IDs")
    selected = [r for r in report["per_region"] if r["region_id"] in set(ids)]
    if len(selected) != expected_count or {r["region_id"] for r in selected} != set(ids):
        raise ValueError("Cached T2 geometry differs from the exact sealed ID inventory")
    boxes = []
    for row in selected:
        if row["kind"] != "T2-explicit-aqueous-negative" or row["run"].casefold() in BLIND:
            raise ValueError("Cached geometry does not belong to permitted T2 training boxes")
        bounds = tuple(row["bounds_xyxy"])
        if (len(bounds) != 4 or any(type(x) is not int for x in bounds)
                or bounds[0] < 0 or bounds[1] < 0 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]
                or row["annotation_pixels"] != (bounds[2]-bounds[0])*(bounds[3]-bounds[1])):
            raise ValueError("Cached T2 raster is not a complete rectangular box; annotation fallback forbidden")
        boxes.append({k: row[k] for k in ("region_id", "run", "frame_index", "source_sha256", "bounds_xyxy", "annotation_pixels")})
    return boxes


def box_statistics(boxes, base, t2, result, modelled, source_sha):
    rows = []
    for box in boxes:
        if box["source_sha256"] != source_sha:
            raise ValueError("Cached T2 box belongs to another acquisition")
        x0, y0, x1, y1 = box["bounds_xyxy"]
        if x1 > base.shape[1] or y1 > base.shape[0]:
            raise ValueError("Cached T2 box escaped the unchanged phase grid")
        n = box["annotation_pixels"]
        row = {**box, "modelled_pixels": int(modelled[y0:y1, x0:x1].sum())}
        for name, value in (("base", base), ("t2_floor", t2), ("combined", result)):
            region = value[y0:y1, x0:x1]
            row[name + "_gas_pixels"] = int((region == 2).sum())
            row[name + "_gas_fraction"] = row[name + "_gas_pixels"] / n
            row[name + "_unresolved_pixels"] = int((~np.isin(region, (0, 1, 2))).sum())
        rows.append(row)
    return rows


def aggregate_boxes(rows, by_frame):
    groups = defaultdict(list)
    for row in public(rows):
        groups[(row["run"], row["frame_index"]) if by_frame else (row["run"],)].append(row)
    result = []
    for key, boxes in sorted(groups.items()):
        item = {"run": key[0], "boxes": len(boxes)}
        if by_frame: item["frame_index"] = key[1]
        denominator = sum(b["annotation_pixels"] for b in boxes)
        for name in ("base", "t2_floor", "combined"):
            item[name + "_equal_box_gas_fraction"] = float(np.mean([b[name + "_gas_fraction"] for b in boxes]))
            item[name + "_pixel_weighted_gas_fraction"] = sum(b[name + "_gas_pixels"] for b in boxes) / denominator
        result.append(item)
    return result


def mask(descriptor, shape, source_sha, index):
    if descriptor.get("source_sha256") != source_sha or descriptor.get("frame_index") != index or tuple(descriptor.get("shape_hw", ())) != shape:
        raise ValueError("L1 flag lost its exact source/frame/canvas binding")
    value = np.load(oldqa.verified(descriptor), allow_pickle=False)
    if descriptor.get("storage") == "packbits-npy":
        if value.dtype != np.uint8 or value.shape != ((np.prod(shape)+7)//8,):
            raise ValueError("L1 flag packbits shape or dtype differs")
        value = np.unpackbits(value, count=int(np.prod(shape)), bitorder=descriptor.get("bitorder", "big")).reshape(shape).astype(bool)
    if value.dtype != bool or value.shape != shape:
        raise ValueError("L1 flag differs from its exact label grid")
    return value


def checks_pass(counters, floor, recheck):
    bad_floor = ("filled_pixels_not_aqueous", "filled_pixels_already_co2_in_base", "preinjection_fill_pixels",
                 "filled_under_gas_pixels", "invalid_aqueous_trigger_columns", "discontinuous_fill_columns",
                 "co2_union_changes_outside_floor_contact_fill", "co2_union_removed_pixels", "uncategorised_fill_pixels",
                 "unexplained_changed_floor_columns")
    return not any(counters.values()) and not any(floor[key] for key in bad_floor) and not any(recheck.values())


def inspect_run(task):
    cv2.setNumThreads(1)
    run, entries, base_candidate, t2_candidate, reference_zones, boxes, output, binding = task
    hidden = run.casefold() in BLIND
    started, cpu = time.perf_counter(), time.process_time()
    frames, all_boxes, passed, peak_rss = [], [], True, 0
    for path, row in entries:
        index, source_sha = int(row["frame_index"]), row["source_sha256"]
        base_path = Path(base_candidate)/"rig_camera_frames"/run.upper()/f"frame_{index:03d}.json"
        t2_path = Path(t2_candidate)/"rig_camera_frames"/run.upper()/f"frame_{index:03d}.json"
        parent, t2row = oldqa.read_json(base_path), oldqa.read_json(t2_path)
        if any(other["source_sha256"] != source_sha or other["elapsed_h"] != row["elapsed_h"]
               or other["binding"]["source"] != row["binding"]["source"]
               or other["rig_domain_revision"] != row["rig_domain_revision"] for other in (parent, t2row)):
            raise ValueError("Candidate source/time/geometry/area differs from its parents")
        if (row["before_labels"] != parent["labels_map"] or row["t2_labels"] != t2row["labels_map"]
                or row["t2_floor_contact_fill"] != t2row["floor_contact_fill"]):
            raise ValueError("L1 candidate does not bind its exact parent decisions")
        base, t2, new = (oldqa.labels(x["labels_map"]) for x in (parent, t2row, row))
        shape = new.shape
        area = oldqa.area_map(row["rig_domain_revision"]["area_map"])
        minimum = oldqa.area_map(row["minimum_area_map"])
        fill = mask(row["floor_contact_fill"], shape, source_sha, index)
        tfill = mask(row["t2_floor_contact_fill"], shape, source_sha, index)
        fragment = mask(row["l1_zone_edge_fragment"], shape, source_sha, index)
        point, disc, near = l1_geometry(row, shape)
        active = row["elapsed_h"] >= ACTIVATION_H
        declared = row["l1_zone"]
        if (declared["active"] is not active or declared["radius_px"] != RADIUS_PX
                or declared["activation_h"] != ACTIVATION_H or not np.allclose(declared["point_xy"], point, atol=1e-8, rtol=0)):
            raise ValueError("Stored L1 zone differs from the source-only geometry")
        counters, edge, before, tpre = identities(base, t2, new, tfill, fill, fragment, disc, active, minimum)
        classes, _ = load_rig_classes(row["rig_domain_revision"])
        with np.load(oldqa.verified(row["rig_domain_revision"]["classes"]), allow_pickle=False) as archive:
            classes["hardware_excluded"] = archive["hardware_excluded"].copy()
        centre = native_photo_centre(row)
        categories, observed = support_categories(classes, centre)
        floor_zone, _ = prepare_bound_floor_zone(row, classes, area, observed)
        supported = (observed | classes["filled_observed"] | classes["filled_unobserved"]) & classes["rig_domain"] & ~classes["hardware_excluded"]
        floor = floor_statistics(before, tpre, new, fill, tfill, area, categories, near, floor_zone, row["elapsed_h"], supported=supported)
        recheck = {"rule_B_would_change_pixels": int(row["rule_b"]["changed_pixels"]),
                   "top_clip_would_change_pixels": int(row["top_clip"]["changed_pixels"])}
        passed &= checks_pass(counters, floor, recheck)
        if hidden:
            # No private frame numbers, areas, geometry, counters or paths are
            # persisted by this reporter, including failure messages or totals.
            continue
        reference = oldqa.native_zones(reference_zones, row, shape)
        boxrows = box_statistics([b for b in boxes if b["frame_index"] == index], base, t2, new,
                                 classes["modelled"], source_sha)
        all_boxes.extend(boxrows)
        qa, timing = row.get("qa", {}), row.get("timing", {})
        value = {"contract": CONTRACT, "job_binding_sha256": binding, "run": run.upper(),
            "frame_index": index, "source_sha256": source_sha, "elapsed_h": row["elapsed_h"],
            "identities": counters, "l1_zone": {"point_xy": point.tolist(), "active": active,
                "radius_px": RADIUS_PX, "activation_h": ACTIVATION_H},
            "zone_edge_fragments": edge, "floor": floor, "recheck": recheck,
            "gas": gas_statistics(base, t2, new, area, disc, reference),
            "cost": {"candidate_cpu_s": timing.get("cpu_s", qa.get("cpu_s")),
                "candidate_wall_s": timing.get("wall_s", qa.get("wall_s")),
                "candidate_rss_bytes": timing.get("rss_bytes"),
                "candidate_peak_working_set_bytes": timing.get("peak_working_set_bytes")},
            "provenance": {"candidate_frame_receipt": oldqa.asset(path), "base_frame_receipt": oldqa.asset(base_path),
                "t2_frame_receipt": oldqa.asset(t2_path), "base_labels": parent["labels_map"],
                "t2_labels": t2row["labels_map"], "combined_labels": row["labels_map"],
                "measured_area": row["rig_domain_revision"]["area_map"], "minimum_area": row["minimum_area_map"]}}
        oldqa._write_json(Path(output)/"frame_qa"/run/f"frame_{index:03d}.json", value)
        frames.append(value)
        try:
            import psutil
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        except ImportError:
            pass
    if hidden:
        return {"blind_status": {"run": run.upper(), "pass": bool(passed)}}
    return {"frames": frames, "boxes": all_boxes, "all_pass": bool(passed), "public_cost": {
        "run": run.upper(), "report_worker_cpu_s": time.process_time()-cpu,
        "report_worker_wall_s": time.perf_counter()-started, "peak_observed_worker_rss_bytes": peak_rss or None}}


def flatten_frames(rows):
    result = []
    for row in public(rows):
        flat = {key: row[key] for key in ("run", "frame_index", "source_sha256", "elapsed_h")}
        for group in ("identities", "zone_edge_fragments", "recheck", "cost"):
            flat.update({group + "_" + key: value for key, value in row[group].items()})
        for name, metrics in row["gas"].items():
            flat.update({name + "_gas_" + key: value for key, value in metrics.items()})
        flat.update({"floor_"+key: value for key, value in row["floor"].items() if not isinstance(value, dict)})
        for group in ("by_support", "by_L1_distance"):
            for name, metrics in row["floor"][group].items():
                flat.update({"floor_"+group+"_"+name+"_"+key: value for key, value in metrics.items()})
        result.append(flat)
    return result


def target_times(rows):
    result = []
    groups = defaultdict(list)
    for row in public(rows):
        if row["elapsed_h"] > 0: groups[row["run"]].append(row)
    for run, frames in sorted(groups.items()):
        for target in (1.25, 2.25, 4., 6.):
            row = min(frames, key=lambda x: abs(x["elapsed_h"]-target))
            result.append({"run": run, "frame_index": row["frame_index"], "target_elapsed_h": target,
                "actual_elapsed_h": row["elapsed_h"], "time_distance_seconds": abs(row["elapsed_h"]-target)*3600,
                **{name+"_gas_"+region: row["gas"][name][region+"_cm2"] for name in ("base", "combined")
                   for region in ("cap_band", "plume", "below_injector")}})
    return result


def make_figures(output, candidate, rows):
    from darsia.fluidflower.phase_v5_display_asset import load_display_image
    from darsia.fluidflower.phase_rig_display_canvas import pad_rig_photograph
    from darsia.fluidflower.phase_floor_contact_display import hatch_floor_contact_rgba
    from PySide6.QtGui import QImage
    figures = []
    requests = [(r, t, "l1") for r in ("AC28", "AC47") for t in (2.25, 4.)] + [("AC28", 2.25, "full_width_floor")]
    for run, target, kind in requests:
        if run.casefold() in BLIND: raise PermissionError("Blind figures are prohibited")
        group = [r for r in public(rows) if r["run"] == run]
        if not group: raise ValueError("Required public figure run is missing")
        result = min(group, key=lambda r: abs(r["elapsed_h"]-target))
        row = oldqa.read_json(result["provenance"]["candidate_frame_receipt"]["path"])
        photo = pad_rig_photograph(load_display_image(Path(candidate)/"source_alias", run.casefold(), result["frame_index"], mode="natural"), row["rig_domain_revision"])
        photo = photo.convertToFormat(QImage.Format.Format_RGB888)
        rgb = np.frombuffer(photo.constBits(), np.uint8).reshape(photo.height(), photo.bytesPerLine())[:, :photo.width()*3].reshape(photo.height(), photo.width(), 3).copy()
        base, new = oldqa.labels(row["before_labels"]), oldqa.labels(row["labels_map"])
        fill = mask(row["floor_contact_fill"], new.shape, row["source_sha256"], row["frame_index"])
        if rgb.shape[:2] != new.shape: raise ValueError("Figure photo and maps differ in full phase grid")
        x, y = result["l1_zone"]["point_xy"]
        crop = [0, result["floor"]["floor_roi_y0"], new.shape[1], new.shape[0]] if kind == "full_width_floor" else [max(0,int(x-1100)), max(0,int(y-1100)), min(new.shape[1],int(x+1100)), min(new.shape[0],int(y+700))]
        panels = []
        for title, phase, flag in (("Photograph", None, None), ("Base overlay", base, None), ("Combined overlay", new, fill)):
            pixels = rgb.copy()
            if phase is not None:
                rgba = np.zeros((*phase.shape,4), np.uint8)
                rgba[phase == 1] = [70,220,110,128]; rgba[phase == 2] = [230,75,140,128]
                if flag is not None: rgba = hatch_floor_contact_rgba(rgba, flag)
                alpha = rgba[...,3:4].astype(np.float32)/255
                pixels = np.rint(pixels*(1-alpha)+rgba[...,:3]*alpha).astype(np.uint8)
            panel = Image.fromarray(pixels)
            ImageDraw.Draw(panel).ellipse((x-RADIUS_PX,y-RADIUS_PX,x+RADIUS_PX,y+RADIUS_PX),outline="#ffe26a",width=5)
            panel = panel.crop(tuple(crop)); panel.thumbnail((1000,700),Image.Resampling.LANCZOS)
            titled = Image.new("RGB",(panel.width,panel.height+45),"#152029"); titled.paste(panel,(0,45))
            ImageDraw.Draw(titled).text((10,12),f"{run} {result['elapsed_h']:.4f} h — {title}",fill="white")
            panels.append(titled)
        figure = Image.new("RGB",(sum(p.width for p in panels),max(p.height for p in panels)),"#152029")
        offset = 0
        for panel in panels: figure.paste(panel,(offset,0)); offset += panel.width
        path = Path(output)/"figures"/f"{run}_frame_{result['frame_index']:03d}_{kind}.png"
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): raise FileExistsError(path)
        figure.save(path)
        figures.append({"run":run,"frame_index":result["frame_index"],"target_elapsed_h":target,
            "actual_elapsed_h":result["elapsed_h"],"image":oldqa.asset(path),"native_crop_xyxy":crop,
            "photograph_display":"source-bound Natural; circle and hatch are report display only",
            "source_frame_receipt":result["provenance"]["candidate_frame_receipt"]})
    return figures


def run(args):
    output = Path(args.output)
    if output.exists(): raise FileExistsError("Choose a new immutable L1 QA directory")
    available = oldqa.frame_receipts(args.candidate, "rig_camera_frames")
    if not args.allow_partial and (len(available) != 864 or len({k[0] for k in available}) != 43):
        raise ValueError("L1 QA requires exactly 43 runs and 864 committed frames")
    # Cached QA geometry is permitted; original CSV/session/regions payloads
    # are never opened, even for hashes or fallback rasterisation.
    box_report, dataset = oldqa.read_json(args.t2_box_report), oldqa.read_json(args.dataset_receipt)
    if (box_report["dataset_receipt_sha256"] != oldqa.digest(args.dataset_receipt)
            or box_report["training_label_sha256"] != dataset["combined_label_sha256"]
            or Path(box_report["base_candidate"]).resolve() != Path(args.base_candidate).resolve()
            or Path(box_report["new_candidate"]).resolve() != Path(args.t2_candidate).resolve()):
        raise ValueError("Cached T2 geometry lost its exact dataset/parent-candidate binding")
    boxes = cached_t2_boxes(box_report, dataset)
    inputs = {"candidate":str(args.candidate),"base_candidate":str(args.base_candidate),"t2_candidate":str(args.t2_candidate),
        "facies_map":oldqa.asset(args.facies_map),"zone_map":oldqa.asset(args.zone_map),
        "t2_cached_geometry":oldqa.asset(args.t2_box_report),"dataset_ID_inventory":oldqa.asset(args.dataset_receipt),
        "report_code":oldqa.asset(__file__),"geometry_policy":GEOMETRY_POLICY}
    binding = hashlib.sha256(json.dumps(inputs,sort_keys=True,allow_nan=False).encode()).hexdigest()
    reference = oldqa.build_reference_zones(args.facies_map,args.zone_map)
    groups = defaultdict(list)
    for (run_id,index), entry in available.items(): groups[run_id].append(entry)
    for box in boxes:
        if (box["run"].casefold(),box["frame_index"]) not in available:
            raise ValueError("A sealed T2 source frame is missing from the new candidate")
    workers = max(1,min(int(args.workers),os.cpu_count() or 1,len(groups)))
    resource_budget = {"requested_workers":int(args.workers),"logical_cpu_bound":os.cpu_count(),
                       "worker_memory_estimate_bytes":3<<30}
    try:
        import psutil
        memory = psutil.virtual_memory()
        reserve = max(2<<30,int(memory.total*.05))
        affinity = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(),"cpu_affinity") else (os.cpu_count() or 1)
        workers = min(workers,affinity,max(1,int(max(0,memory.available-reserve)//(3<<30))))
        resource_budget.update(affinity_cpu_bound=affinity,available_memory_bytes=memory.available,
                               memory_reserve_bytes=reserve)
    except ImportError:
        workers = min(workers,2)
    output.mkdir(parents=True)
    oldqa._write_json(output/"report_binding.json",{"contract":CONTRACT,"job_binding_sha256":binding,"inputs":inputs})
    frames, boxrows, blind, cost, statuses = [], [], [], [], []
    tasks = [(run,entries,args.base_candidate,args.t2_candidate,reference,
              [b for b in boxes if b["run"].casefold()==run],output,binding) for run,entries in groups.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(inspect_run,task):task[0] for task in tasks}
        for future in as_completed(pending):
            value = future.result()
            if "blind_status" in value:
                blind.append(value["blind_status"])
                print(json.dumps({"withheld_run_check":"completed"}),flush=True)
            else:
                frames.extend(value["frames"]); boxrows.extend(value["boxes"]); cost.append(value["public_cost"])
                statuses.append(value["all_pass"])
                print(json.dumps({"public_run":pending[future].upper(),"public_runs_completed":len(statuses)}),flush=True)
    frames.sort(key=lambda x:(x["run"],x["frame_index"]))
    boxrows.sort(key=lambda x:x["region_id"])
    if {b["region_id"] for b in boxrows} != {b["region_id"] for b in boxes}:
        raise ValueError("Full-map QA did not evaluate the exact 223 cached T2 boxes")
    targets = target_times(frames)
    oldqa._csv(output/"frames_public.csv",flatten_frames(frames))
    oldqa._csv(output/"gas_target_times_public.csv",targets)
    oldqa._csv(output/"t2_boxes_public.csv",boxrows)
    box_frame, box_run = aggregate_boxes(boxrows,True), aggregate_boxes(boxrows,False)
    oldqa._csv(output/"t2_boxes_per_frame_public.csv",box_frame)
    oldqa._csv(output/"t2_boxes_per_run_public.csv",box_run)
    figures = make_figures(output,args.candidate,frames) if not args.no_figures else []
    numeric_totals = {key:sum(row["identities"][key] for row in public(frames)) for key in frames[0]["identities"]}
    floor_totals = {key:sum(row["floor"][key] for row in public(frames)) for key in (
        "filled_pixels","filled_area_cm2","combined_only_fill_pixels","t2_only_fill_pixels",
        "preinjection_fill_pixels","filled_under_gas_pixels","changed_fill_without_trigger_phase_change_columns",
        "changed_fill_explained_by_trigger_phase_columns","changed_fill_explained_by_component_eligibility_columns",
        "unexplained_changed_floor_columns")}
    candidate_cost = {key:sum(row["cost"][key] for row in public(frames) if row["cost"][key] is not None)
                      for key in ("candidate_cpu_s","candidate_wall_s")}
    candidate_cost.update({key:max((row["cost"][key] or 0 for row in public(frames)),default=0)
                           for key in ("candidate_rss_bytes","candidate_peak_working_set_bytes")})
    report = {"contract":CONTRACT,"complete":True,"technical_all_pass":all(statuses) and all(r["pass"] for r in blind),
        "created_utc":datetime.now(timezone.utc).isoformat(),"job_binding_sha256":binding,"inputs":inputs,
        "public_run_count":len(statuses),"public_frame_count":len(frames),"public_frames":frames,
        "public_identity_totals":numeric_totals,"public_floor_totals":floor_totals,"target_times":targets,
        "floor_difference_policy":{"floor_algorithm_and_thresholds_unchanged":True,
            "literal_trigger_phase_only_assumption_holds":floor_totals["changed_fill_without_trigger_phase_change_columns"]==0,
            "acceptance":"No unexplained differences: changed first-CO2 phase, or supported four-connected aqueous component crossing the unchanged 2.5 mm2 minimum, with changed membership proven",
            "specification_clarification":"The unchanged floor rule depends on both trigger phase and connected component eligibility. All non-phase explanations remain visible in the report."},
        "T2_in_sample":{"cached_geometry_only":True,"annotation_payloads_read":False,"independent_validation":False,
            "box_count":len(boxrows),"ID_inventory_verified":True,"per_box":boxrows,"per_frame":box_frame,"per_run":box_run,
            "denominator":"all cached rectangular pixels; unresolved/modelled coverage retained; equal-box and pixel-weighted rates"},
        "public_cost":{**candidate_cost,"report_workers":workers,"worker_resource_budget":resource_budget,"runs":cost,
            "report_worker_cpu_s":sum(r["report_worker_cpu_s"] for r in cost),
            "peak_observed_worker_rss_bytes":max((r["peak_observed_worker_rss_bytes"] or 0 for r in cost),default=0),
            "scope":"public runs only; blind worker costs and aggregate contributions excluded; CPU software only"},
        "blind_status":sorted(blind,key=lambda x:x["run"]),"figures":figures,
        "blind_policy":"AC30/49/54 pass/fail only; no maps, figures, per-frame numbers or aggregate numerical contributions",
        "training_labels_read":False,"test_labels_read":False,"scorer_run":False,
        "mass_eligible":False,"bottom_mass_allowed":False,"automatic_activation":False,"default_candidate_changed":False}
    oldqa._write_json(output/"qa_public.json",report)
    receipt = {key:report[key] for key in ("contract","complete","technical_all_pass","created_utc","job_binding_sha256",
        "inputs","public_run_count","public_frame_count","public_identity_totals","public_floor_totals","public_cost",
        "blind_status","blind_policy","training_labels_read","test_labels_read","scorer_run","mass_eligible",
        "bottom_mass_allowed","automatic_activation","default_candidate_changed","floor_difference_policy")}
    receipt.update(report=oldqa.asset(output/"qa_public.json"),figures=figures,
        reports={path.name:oldqa.asset(path) for path in sorted(output.glob("*.csv"))})
    oldqa._write_json(output/"report_receipt.json",receipt)
    print(json.dumps({"complete":True,"technical_all_pass":report["technical_all_pass"],"receipt":str(output/"report_receipt.json")}),flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","base-candidate","t2-candidate","output","facies-map","zone-map","t2-box-report","dataset-receipt"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--allow-partial",action="store_true")
    parser.add_argument("--no-figures",action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0: parser.error("CPU QA workers must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
