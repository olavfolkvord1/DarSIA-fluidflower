"""Write new local, source-bound bottom-band reports without changing a candidate.

Default execution scores all selected sealed frames, on RAM-bounded CPU workers.
Use --runs and --frames for a method-review pilot before an all-run diagnostic.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from threading import Lock
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower import phase_ffac_bottom_band_diagnostics as diag
from darsia.fluidflower.tools.ffac_track1_display_modes import display_contract

_VERIFIED = {}
_VERIFY_LOCK = Lock()


def _hash(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _asset(path, digest=None):
    path = Path(path)
    return {"path": str(path), "sha256": digest or _hash(path)}


def _verified(asset):
    path = Path(asset["path"])
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _VERIFY_LOCK:
        digest = _VERIFIED.get(key)
    if digest is None:
        digest = _hash(path)
        with _VERIFY_LOCK:
            _VERIFIED[key] = digest
    if digest != asset["sha256"]:
        raise ValueError(f"Source binding failed: {path}")
    return path


def _load(asset):
    return np.load(_verified(asset), allow_pickle=False, mmap_mode="r")


def _bound_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")), _asset(path)


def _bundle_asset(declaration):
    for key in ("snapshot_path", "source_path"):
        path = Path(declaration[key])
        if path.is_file():
            return {"path": str(path), "sha256": declaration["sha256"]}
    return None


def annotation_inventory(candidate):
    """Inventory unchanged archived categories; deduplicate repeated snapshots."""
    request, request_asset = _bound_json(candidate / "training_request.json")
    by_run = {}
    assets = [request_asset]
    missing = []
    seen = set()
    for bundle in request.get("bundles", []):
        asset = _bundle_asset(bundle["decisions"])
        if asset is None:
            missing.append(bundle["decisions"])
            continue
        document = json.loads(_verified(asset).read_text(encoding="utf-8-sig"))
        assets.append(asset)
        for run, decision in document.get("decisions", {}).items():
            for annotation in decision.get("annotations", []):
                if annotation.get("category") != "incomplete_wash_indicator_carryover":
                    continue
                fingerprint = sha256(json.dumps(annotation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                by_run.setdefault(run.casefold(), []).append({"annotation": annotation, "asset": asset, "annotation_sha256": fingerprint})
    return by_run, {"training_request": request_asset, "verified_decision_assets": assets[1:], "missing_decision_bundle_count": len(missing)}


def proven_unfilled_native_observation(receipt, *, source_sha, producer_code_sha, valid, signal):
    """Admit only the exact direct RAW producer branch; unexplained absence is unknown."""
    audit = receipt.get("producer_audit") or {}
    if receipt.get("source_sha256") != source_sha:
        raise ValueError("Scientific observation proof belongs to another source")
    proven = (receipt.get("contract") == "ffac-camera-clock-full-native-scientific-v1"
        and receipt.get("parent_cache", "missing") is None
        and receipt.get("parent_valid_bits_preserved") is False
        and audit.get("contract") == "ffac-rig-full-native-frozen-raw-source-v1"
        and audit.get("parent_cache", "missing") is None
        and (audit.get("processor") or {}).get("code_sha256") == producer_code_sha
        and (audit.get("photometry") or {}).get("output_encoding") == "chw-float32-log-camera-linear-rgb-with-nan-invalid")
    if not proven:
        return None
    if signal.dtype != np.float32 or signal.shape != (3, *valid.shape) or valid.dtype != bool:
        raise ValueError("Direct RAW observation proof has another native encoding")
    finite = np.empty(valid.shape, bool)
    for top in range(0, valid.shape[0], 128):
        finite[top:top+128] = np.isfinite(signal[:, top:top+128]).all(0)
    if not np.array_equal(valid, finite):
        raise ValueError("Direct RAW valid mask contradicts finite scientific pixels")
    if audit.get("full_native_corrected_finite_pixels") != int(finite.sum()):
        raise ValueError("Direct RAW finite count contradicts its scientific receipt")
    return finite


def _unfilled_native_observation(centre, source_sha, valid):
    from darsia.fluidflower import phase_ffac_rig_raw_source
    receipt_asset = centre["receipt"]
    receipt = json.loads(_verified(receipt_asset).read_text(encoding="utf-8-sig"))
    if receipt.get("valid_mask_sha256") != centre["valid_mask"]["sha256"] or receipt.get("log_linear_rgb_sha256") != centre["log_linear_rgb"]["sha256"]:
        raise ValueError("Scientific observation proof has another array binding")
    producer = _asset(Path(phase_ffac_rig_raw_source.__file__))
    audit = receipt.get("producer_audit") or {}
    code_matches = (audit.get("processor") or {}).get("code_sha256") == producer["sha256"]
    signal = _load(centre["log_linear_rgb"]) if code_matches else np.empty((0,), np.float32)
    observed = proven_unfilled_native_observation(receipt, source_sha=source_sha, producer_code_sha=producer["sha256"], valid=valid, signal=signal)
    proof = {"scientific_receipt": receipt_asset, "producer_code": producer,
             "status": "direct RAW producer proven unfilled; valid mask equals strictly finite scientific RGB" if observed is not None else "unknown: missing isolated-fill mask without the exact proven direct-RAW contract/code/parent flags",
             "parent_cache": receipt.get("parent_cache"), "parent_valid_bits_preserved": receipt.get("parent_valid_bits_preserved")}
    if observed is not None:
        proof["camera_log_rgb"] = centre["log_linear_rgb"]
    return observed, proof


def _raw_observation(row, frame, manifest, provenance):
    descriptor = row["full_extended_scientific"]
    rev = row["rig_domain_revision"]
    shape = tuple(rev["phase_shape_hw"])
    origin = tuple(rev["native_origin_xy"])
    if (descriptor.get("complete") is not True
        or descriptor.get("storage") != "exact-native-centre-plus-disjoint-frozen-raw-ring-windows"
        or tuple(descriptor.get("phase_shape_hw", ())) != shape
        or tuple(descriptor.get("native_origin_xy", ())) != origin
        or descriptor.get("source_sha256") != row["source_sha256"]):
        raise ValueError("Extended RAW belongs to another source or grid")
    centre = descriptor["native_centre"]
    if centre["source_sha256"] != row["source_sha256"]:
        raise ValueError("RAW centre has another source")
    x, y, w, h = manifest["content_box_xywh"]
    if (x, y) != (-origin[0], -origin[1]) or (h, w) != tuple(centre["native_shape_hw"]):
        raise ValueError("RAW centre has another origin or native shape")
    observed = np.zeros(shape, dtype=bool)
    covered = np.zeros(shape, dtype=bool)
    valid = _load(centre["valid_mask"])
    if valid.shape != (h, w) or valid.dtype != bool:
        raise ValueError("RAW validity encoding differs from the bound native source")
    if "v5_filled_invalid_mask_path" in frame and "v5_filled_invalid_mask_sha256" in frame:
        invalid_asset = {"path": frame["v5_filled_invalid_mask_path"], "sha256": frame["v5_filled_invalid_mask_sha256"]}
        filled_invalid = _load(invalid_asset)
        if filled_invalid.shape != valid.shape or filled_invalid.dtype != bool:
            raise ValueError("Scientific fill mask has another native grid")
        observed[y:y+h, x:x+w] = valid & ~filled_invalid
        covered[y:y+h, x:x+w] = True
        provenance["native_raw_valid_mask"] = centre["valid_mask"]
        provenance["native_scientific_fill_exclusion"] = invalid_asset
    else:
        native_observed, proof = _unfilled_native_observation(centre, row["source_sha256"], valid)
        if native_observed is not None:
            observed[y:y+h, x:x+w] = native_observed
        covered[y:y+h, x:x+w] = True
        provenance["native_raw_valid_mask"] = centre["valid_mask"]
        provenance["native_scientific_fill_exclusion"] = None
        provenance["native_unfilled_raw_proof"] = proof
    provenance["raw_ring_assets"] = []
    for window in descriptor["ring_windows"]:
        a, b, c, d = window["canvas_xyxy"]
        if not (0 <= a < c <= shape[1] and 0 <= b < d <= shape[0]) or covered[b:d, a:c].any():
            raise ValueError("RAW ring window overlaps or leaves the bound canvas")
        values = _load(window["log_linear_rgb"])
        if values.dtype != np.float32 or values.shape != (3, d-b, c-a):
            raise ValueError("RAW ring has another native float encoding")
        observed[b:d, a:c] = np.isfinite(values).all(0)
        covered[b:d, a:c] = True
        provenance["raw_ring_assets"].append(window)
    provenance["raw_observation_definition"] = "bound native valid mask minus original scientific isolated-invalid fill, plus strictly finite disjoint RAW ring windows"
    return observed


def _raw_roi(row, manifest, zone, provenance):
    descriptor = row["full_extended_scientific"]
    signal = np.full((3, zone.y1-zone.y0, len(zone.floor_canvas_y)), np.nan, np.float32)
    centre_asset = descriptor["native_centre"]["log_linear_rgb"]
    centre = _load(centre_asset)
    x, y, w, h = manifest["content_box_xywh"]
    if centre.shape != (3, h, w) or centre.dtype != np.float32:
        raise ValueError("Native RAW colour has another source encoding")
    top, bottom = max(y, zone.y0), min(y+h, zone.y1)
    if top < bottom:
        signal[:, top-zone.y0:bottom-zone.y0, x:x+w] = centre[:, top-y:bottom-y]
    for window in descriptor["ring_windows"]:
        a, b, c, d = window["canvas_xyxy"]
        top, bottom = max(b, zone.y0), min(d, zone.y1)
        if top < bottom:
            values = _load(window["log_linear_rgb"])
            signal[:, top-zone.y0:bottom-zone.y0, a:c] = values[:, top-b:bottom-b]
    signal[:, ~zone.raw_observed] = np.nan
    provenance["native_raw_colour"] = centre_asset
    provenance["fixed_colour_transfer"] = display_contract()
    provenance["fixed_colour_transfer_override"] = "pointwise frozen Natural matrix and gain only; no highlight modification, invalid fill, chroma filter, or JPEG"
    return signal


def _exact_label_mask(inventory, frame, manifest, shape, source_sha):
    """Only exact current source-receipt geometry can support a labelled band."""
    mask = np.zeros(shape, bool)
    used = []
    for item in inventory:
        a = item["annotation"]
        migration = a.get("geometry_migration", {})
        if migration.get("frame_receipt_sha256") != frame.get("v5_frame_receipt_sha256"):
            continue
        if a.get("coordinate_space") != "normalized_native_review_canvas_16x9":
            continue
        canvas_w, canvas_h = manifest["canvas_wh"]
        if a.get("shape", "rectangle") == "rectangle":
            x, y, w, h = (float(a[k]) for k in ("x", "y", "width", "height"))
            vertices = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
        else:
            vertices = a.get("points", [])
        p = np.asarray(vertices, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            continue
        layer = np.zeros(shape, np.uint8)
        cv2.fillPoly(layer, [np.rint(p * [canvas_w, canvas_h] * 256).astype(np.int32)], 1, shift=8)
        mask |= layer.astype(bool)
        used.append({"decision_asset": item["asset"], "annotation_sha256": item["annotation_sha256"], "annotation_id": a.get("id"), "exact_frame_receipt_sha256": frame["v5_frame_receipt_sha256"], "source_sha256": source_sha})
    return mask if used else None, used


def measure_frame(candidate, run, frame, manifest, manifest_asset, *, baseline, inventory, mask_dir=None):
    started = time.perf_counter()
    index = int(frame["frame_index"])
    row, row_asset = _bound_json(candidate / "rig_camera_frames" / run.upper() / f"frame_{index:03d}.json")
    if not row.get("complete") or row["source_sha256"] != frame["source_sha256"] or row["frame_index"] != index:
        raise ValueError("Diagnostic frame identity is not bound to the selected source")
    if float(row["elapsed_h"]) != float(frame["elapsed_h"]):
        raise ValueError("Diagnostic timestamp differs from the candidate source")
    rev = row["rig_domain_revision"]
    parameters = rev["parameters"]
    scale = parameters["physical_scale_revision"]
    if not scale.get("physical_scale_approved") or scale["inner_width_mm"] != 900:
        raise ValueError("This diagnostic requires the bound approved 900 mm area revision")
    provenance = {"frame_receipt": row_asset, "native_manifest": manifest_asset,
                  "source_sha256": row["source_sha256"], "classes": rev["classes"],
                  "measured_area": rev["area_map"], "saved_phase_labels": row["labels_map"],
                  "geometry_source": row["binding"]["source"]["rig_geometry"],
                  "physical_scale": scale}
    classes = _load(rev["classes"])
    weights = _load(rev["area_map"])
    observed = _raw_observation(row, frame, manifest, provenance)
    geometry = row["binding"]["source"]["rig_geometry"]
    reference_polygon = np.asarray(geometry["reference_rig"]["polygon_reference_native"], dtype=np.float64)
    reference_to_frame = np.asarray(geometry["H_reference_rig_to_frame_native"], dtype=np.float64)
    transported = np.c_[reference_polygon, np.ones(len(reference_polygon))] @ reference_to_frame.T
    transported = transported[:, :2] / transported[:, 2:]
    if transported.shape != np.asarray(geometry["frame_inner_polygon_native"]).shape or not np.allclose(transported, geometry["frame_inner_polygon_native"], atol=1e-6, rtol=0):
        raise ValueError("Current polygon differs from its bound reference transport")
    reference_segments, segment_indices = diag.bottom_edge_segments(reference_polygon)
    homogeneous = np.concatenate([reference_segments, np.ones((*reference_segments.shape[:2], 1))], axis=2) @ reference_to_frame.T
    bottom_segments = homogeneous[..., :2] / homogeneous[..., 2:]
    provenance["floor_segment_binding"] = {"reference_segment_indices": segment_indices.tolist(), "maximum_abs_reference_slope": 1.0,
        "selection": "negative-x bottom run in documented right-wall, bottom, left-wall polygon order; steep sidewalls and computational top closure excluded",
        "measured_support_intervals_native": geometry["reference_rig"].get("measured_support_intervals_native"),
        "extrapolation": geometry["reference_rig"].get("extrapolation")}
    zone = diag.prepare_floor_zone(polygon_native=geometry["frame_inner_polygon_native"],
        origin_xy=rev["native_origin_xy"], H_native_to_metric_pixel=parameters["H_frame_native_to_metric_pixel"],
        metric_pixel_area_mm2=parameters["metric_pixel_area_mm2"], linear_scale=scale["linear_factor"],
        rig_domain=classes["rig_domain"], hardware_excluded=classes["hardware_excluded"], raw_observed=observed,
        pixel_area_mm2=weights[rev["area_map"].get("array_key", "pixel_area_mm2")], modelled=classes["modelled"], bottom_segments_native=bottom_segments)
    labels = np.asarray(Image.open(_verified(row["labels_map"])))
    if labels.shape != tuple(rev["phase_shape_hw"]):
        raise ValueError("Saved labels have another source canvas")
    labels_roi = labels[zone.y0:zone.y1]
    frame_result = {"run": run.upper(), "frame_index": index, "elapsed_h": row["elapsed_h"],
                    "floor_geometry_uncertainty_declared": bool(geometry.get("unobserved_wall_uncertainty")),
                    "floor_geometry_evidence": geometry.get("boundary_evidence"),
                    "roi_xyxy": [0, zone.y0, labels.shape[1], zone.y1],
                    "coverage": {}, "gap": {}, "provenance": provenance}
    reference_floor = np.c_[np.arange(labels.shape[1]) + rev["native_origin_xy"][0], zone.floor_canvas_y + rev["native_origin_xy"][1], np.ones(labels.shape[1])] @ np.linalg.inv(reference_to_frame).T
    reference_x = reference_floor[:, 0] / reference_floor[:, 2]
    interval = geometry["reference_rig"].get("measured_support_intervals_native", {}).get("bottom")
    measured = zone.geometric_columns & (reference_x >= interval[0]) & (reference_x <= interval[1]) if interval else np.zeros_like(zone.geometric_columns)
    frame_result["floor_geometry_support"] = {"measured_bottom_support_columns": int(measured.sum()), "extrapolated_or_unspecified_bottom_support_columns": int((zone.geometric_columns & ~measured).sum()),
        "observable_measured_bottom_support_columns": int((measured & zone.observable_floor_columns).sum()), "observable_extrapolated_bottom_support_columns": int((zone.geometric_columns & ~measured & zone.observable_floor_columns).sum())}
    masks = {}
    for height in (10.0, 20.0, 30.0):
        key = str(int(height))
        frame_result["coverage"][key] = zone.coverage(height)
        frame_result["gap"][key], masks[key], _ = diag.floor_gap(zone, labels_roi, height_mm=height)
    run_result = None
    if baseline:
        signal = _raw_roi(row, manifest, zone, provenance)
        rgb = diag.fixed_natural_rgb(signal, zone.raw_observed)
        positive = diag.colour_proxy_mask(rgb)
        run_result = {"run": run.upper(), "preinject_frame_index": index, "preinject_elapsed_h": row["elapsed_h"],
                      "preinject_status": "closest actual saved frame strictly before injection", "carryover_proxy": {}, "strict_colour_floor_connection": {},
                      "labelled_carryover": {}, "archived_category_annotation_count": len(inventory),
                      "provenance": provenance, "coverage": frame_result["coverage"],
                      "floor_geometry_uncertainty_declared": frame_result["floor_geometry_uncertainty_declared"]}
        label_mask, used_labels = _exact_label_mask(inventory, frame, manifest, labels.shape, row["source_sha256"])
        run_result["labelled_carryover_status"] = "exact source-bound local category regions available; no claim of exhaustive band labels" if used_labels else "no exact source-bound carryover annotation on this pre-injection frame; missing is not zero"
        run_result["labelled_carryover_provenance"] = used_labels
        for height in (10.0, 20.0, 30.0):
            key = str(int(height))
            run_result["carryover_proxy"][key], band = diag.colour_in_floor_connected_sand(zone, positive, height_mm=height)
            run_result["strict_colour_floor_connection"][key], _ = diag.floor_connected_band(zone, positive, height_mm=height)
            if used_labels:
                run_result["labelled_carryover"][key], _ = diag.colour_in_floor_connected_sand(zone, label_mask[zone.y0:zone.y1], height_mm=height)
            else:
                run_result["labelled_carryover"][key] = None
            if key == "20" and mask_dir:
                Image.fromarray((band * 255).astype(np.uint8)).save(mask_dir / f"{run.upper()}_frame_{index:03d}_proxy_20mm.png")
        if mask_dir:
            Image.fromarray(rgb).save(mask_dir / f"{run.upper()}_frame_{index:03d}_fixed_natural_raw_roi.png")
    elif mask_dir:
        signal = _raw_roi(row, manifest, zone, provenance)
        Image.fromarray(diag.fixed_natural_rgb(signal, zone.raw_observed)).save(mask_dir / f"{run.upper()}_frame_{index:03d}_fixed_natural_raw_roi.png")
    if mask_dir:
        Image.fromarray((masks["20"] * 255).astype(np.uint8)).save(mask_dir / f"{run.upper()}_frame_{index:03d}_gap_20mm.png")
        Image.fromarray((zone.raw_observed * 255).astype(np.uint8)).save(mask_dir / f"{run.upper()}_frame_{index:03d}_raw_support.png")
        np.savez_compressed(mask_dir / f"{run.upper()}_frame_{index:03d}_floor_geometry.npz", floor_canvas_y=zone.floor_canvas_y, base_canvas_y=zone.base_canvas_y, distance_mm=zone.distance_mm, roi_xyxy=frame_result["roi_xyxy"])
    classes.close()
    weights.close()
    frame_result["diagnostic_wall_seconds"] = time.perf_counter() - started
    return frame_result, run_result


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_reports(frame_rows, run_rows, output, *, source_binding):
    assets = {}
    for visibility, public in (("internal", False), ("public", True)):
        document = diag.report_document(frame_rows, run_rows, public=public)
        document["source_binding"] = source_binding
        path = output / f"bottom_band_{visibility}.json"
        path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
        frows = []
        for f in document["frames"]:
            for height in ("10", "20", "30"):
                gap = f["gap"][height]
                frows.append({"run": f["run"], "frame_index": f["frame_index"], "elapsed_h": f["elapsed_h"],
                    "search_height_mm": int(height), "observable_floor_column_share": f["coverage"][height]["observable_floor_column_share"],
                    **{k: gap[k] for k in ("observable_floor_columns", "bounded_observed_gap_area_cm2", "co2_bounded_gap_area_cm2", "positive_observed_gap_columns", "co2_immediately_above_gap_column_share", "co2_at_floor_or_immediately_above_gap_column_share", "co2_immediately_above_among_positive_observed_gap_column_share", "median_gap_thickness_contact_positive_columns_mm", "median_gap_thickness_contact_zero_inclusive_mm", "unknown_full_gap_columns", "search_censored_columns")}})
        rrows = []
        for r in document["runs"]:
            for height in ("10", "20", "30"):
                band = (r.get("carryover_proxy") or {}).get(height) or {}
                rrows.append({"run": r["run"], "preinject_frame_index": r.get("preinject_frame_index"), "preinject_elapsed_h": r.get("preinject_elapsed_h"),
                    "preinject_status": r["preinject_status"], "search_height_mm": int(height), "labelled_carryover_status": r.get("labelled_carryover_status"),
                    **{k: band.get(k) for k in ("status", "observable_photographic_floor_columns", "floor_connected_observed_sand_columns", "band_columns", "band_column_share", "band_area_cm2", "median_thickness_positive_columns_mm", "median_thickness_zero_inclusive_mm", "median_colour_span_positive_columns_mm", "median_maximum_colour_height_above_floor_positive_columns_mm", "band_columns_reaching_search_limit", "band_pixels_adjacent_to_missing_raw")}})
        fpath, rpath = output / f"bottom_gap_frames_{visibility}.csv", output / f"preinject_band_runs_{visibility}.csv"
        _write_csv(fpath, frows)
        _write_csv(rpath, rrows)
        assets[visibility] = {"json": _asset(path), "frame_csv": _asset(fpath), "run_csv": _asset(rpath), "frame_count": len(document["frames"]), "run_count": len(document["runs"])}
    receipt = {"contract": diag.CONTRACT, "reports": assets, "source_binding": source_binding,
               "prediction_maps_changed": False, "training": False, "bottom_mass_allowed": False, "mass_eligible": False}
    (output / "report_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def _available_memory():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        if os.name == "nt":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        return None


def adaptive_workers(requested, run_count):
    available = _available_memory()
    gib = 1024 ** 3
    reserve = 4 * gib
    budget = 1.5 * gib
    ram_cap = max(1, int((available - reserve) // budget)) if available is not None else 1
    actual = max(1, min(int(requested), 24, os.cpu_count() or 1, ram_cap, run_count))
    return actual, {"available_ram_bytes_at_dispatch": available, "ram_reserve_bytes": reserve,
                    "estimated_ram_budget_per_worker_bytes": int(budget), "ram_worker_cap": ram_cap,
                    "requested_worker_cap": requested, "actual_workers": actual,
                    "execution": "CPU process per run; serial frames; one OpenCV/BLAS thread per process"}


def _used_assets(frame_report):
    p = frame_report["provenance"]
    assets = [p[k] for k in ("frame_receipt", "native_manifest", "classes", "measured_area", "saved_phase_labels", "native_raw_valid_mask", "native_scientific_fill_exclusion") if p.get(k) is not None]
    assets += [w["log_linear_rgb"] for w in p["raw_ring_assets"]]
    if "native_raw_colour" in p:
        assets.append(p["native_raw_colour"])
    proof = p.get("native_unfilled_raw_proof") or {}
    assets += [proof[k] for k in ("scientific_receipt", "producer_code", "camera_log_rgb") if k in proof]
    return assets


def _function_nodes(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def _ast_bytes(nodes):
    return "\n".join(ast.dump(n, include_attributes=False) for n in nodes).encode()


def verify_conditional_import(old_output, new_output, immutable_job):
    """Prove the exercised old branch and all declared input sets are unchanged."""
    old_start, old_start_asset = _bound_json(old_output / "job_start.json")
    old_binding = old_start["binding"]
    for key in ("launch", "candidate_seal", "diagnostic_code", "method_sha256", "fixed_colour_matrix_sha256", "selected_runs", "selected_frames", "example_masks"):
        if old_binding[key] != immutable_job[key]:
            raise ValueError(f"Conditional import changes method/source/selection: {key}")
    old_archive = _asset(old_output / "executed_reporter.py")
    if old_archive["sha256"] != old_binding["reporter_code"]["sha256"]:
        raise ValueError("Archived executed old reporter does not match its original job")
    old = _function_nodes(Path(old_archive["path"]).read_text())
    new = _function_nodes(Path(__file__).read_text())
    functions = ("_hash", "_asset", "_verified", "_load", "_raw_roi", "_exact_label_mask", "measure_frame")
    for name in functions:
        if _ast_bytes([old[name]]) != _ast_bytes([new[name]]):
            raise ValueError(f"Conditional import changes executed measurement code: {name}")
    original = old["_raw_observation"].body
    revised = new["_raw_observation"].body
    start = next(i for i, node in enumerate(original) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "invalid_asset" for t in node.targets))
    stop = next(i+1 for i, node in enumerate(original) if isinstance(node, ast.Assign) and "native_scientific_fill_exclusion" in ast.unparse(node))
    branch = next(i for i, node in enumerate(revised) if isinstance(node, ast.If) and "v5_filled_invalid_mask_path" in ast.unparse(node.test))
    unchanged = (original[:start], revised[:branch]), (original[start:stop], revised[branch].body), (original[stop:], revised[branch+1:])
    if any(_ast_bytes(a) != _ast_bytes(b) for a, b in unchanged):
        raise ValueError("Present-fill-mask executed observation branch is not conditionally equivalent")
    imports = []
    for path in sorted((old_output / "frame_receipts").glob("*/frame_*.json")):
        receipt, receipt_asset = _bound_json(path)
        if receipt.get("complete") is not True or receipt.get("job_binding_sha256") != old_start["job_binding_sha256"]:
            raise ValueError("Original measurement checkpoint is incomplete or from another job")
        f = receipt["frame_report"]
        p = f["provenance"]
        manifest = json.loads(_verified(p["native_manifest"]).read_text(encoding="utf-8-sig"))
        frame = manifest["frame_sources"][f["frame_index"]]
        expected_fill = {"path": frame.get("v5_filled_invalid_mask_path"), "sha256": frame.get("v5_filled_invalid_mask_sha256")}
        if None in expected_fill.values() or p.get("native_scientific_fill_exclusion") != expected_fill:
            raise ValueError("Imported checkpoint did not exercise the unchanged fill-mask branch")
        if p["source_sha256"] != frame["source_sha256"] or f["elapsed_h"] != frame["elapsed_h"]:
            raise ValueError("Imported checkpoint has another current source/time")
        _verified(p["frame_receipt"])
        if _used_assets(f) != receipt["verified_source_assets"]:
            raise ValueError("Imported checkpoint has a changed bound input set")
        imports.append({"run": f["run"], "frame_index": f["frame_index"], "original_checkpoint": receipt_asset})
    proof = {"contract": "ffac-bottom-diagnostic-conditional-checkpoint-equivalence-v1", "old_job_start": old_start_asset,
             "old_job_binding_sha256": old_start["job_binding_sha256"], "old_executed_reporter": old_archive,
             "new_reporter": immutable_job["reporter_code"], "unchanged_diagnostic_module": immutable_job["diagnostic_code"],
             "unchanged_method_sha256": immutable_job["method_sha256"], "equivalent_measurement_functions": list(functions),
             "observation_condition": "current manifest has the same real v5_filled_invalid_mask asset; present-mask branch prefix/body/suffix AST exactly unchanged",
             "unchanged_branch_ast_sha256": sha256(_ast_bytes(original[start:stop])).hexdigest(),
             "source_identity_and_frame_receipt_hashes_verified": True, "declared_used_input_sets_unchanged": True,
             "original_input_bytes_verified_at_original_execution": True, "large_scientific_input_bytes_rehashed_at_import": False,
             "large_input_rehash_basis": "reuse of verified immutable-candidate measurements, not a new measurement; original executed checkpoint and provenance retained",
             "imported_frame_count": len(imports), "imports": imports}
    path = new_output / "conditional_equivalence_receipt.json"
    path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return {"old_output": str(old_output), "old_job_binding_sha256": old_start["job_binding_sha256"],
            "old_executed_reporter": old_archive, "conditional_equivalence_receipt": _asset(path), "imported_frame_count": len(imports)}


def _score_run(task):
    """Each process owns one run's checkpoint directory; no source writes."""
    candidate, output, run, frames, manifest, manifest_asset, inventory, job_hash, masks_enabled, resume, conditional_import = task
    cv2.setNumThreads(1)
    run_output = output / "frame_receipts" / run
    run_output.mkdir(parents=True, exist_ok=True)
    mask_dir = output / "example_masks" if masks_enabled else None
    results = []
    for frame, baseline, include in frames:
        path = run_output / f"frame_{frame['frame_index']:03d}.json"
        old_path = Path(conditional_import["old_output"]) / "frame_receipts" / run / path.name if conditional_import else None
        imported = False
        if path.exists():
            if not resume:
                raise ValueError("Existing frame checkpoint requires explicit --resume")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("complete") is not True or receipt.get("job_binding_sha256") != job_hash:
                raise ValueError("Frame checkpoint has a different method/source binding")
            f, r = receipt["frame_report"], receipt["preinject_run_report"]
            if f["provenance"]["source_sha256"] != frame["source_sha256"]:
                raise ValueError("Checkpoint belongs to another selected source")
            for asset in receipt["verified_source_assets"] + receipt.get("example_assets", []):
                _verified(asset)
            resumed = True
        elif old_path is not None and old_path.is_file():
            original, original_asset = _bound_json(old_path)
            if original.get("complete") is not True or original.get("job_binding_sha256") != conditional_import["old_job_binding_sha256"]:
                raise ValueError("Conditional imported checkpoint belongs to another executed job")
            f, r = original["frame_report"], original["preinject_run_report"]
            receipt = {"contract": diag.CONTRACT, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                "job_binding_sha256": job_hash, "frame_report": f, "preinject_run_report": r,
                "verified_source_assets": original["verified_source_assets"], "example_assets": original.get("example_assets", []),
                "imported_measurement": {"original_checkpoint": original_asset, **conditional_import}}
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
            resumed, imported = False, True
        else:
            f, r = measure_frame(candidate, run, frame, manifest, manifest_asset, baseline=baseline, inventory=inventory, mask_dir=mask_dir)
            prefix = f"{run}_frame_{frame['frame_index']:03d}_"
            example_assets = [_asset(p) for p in mask_dir.glob(prefix + "*")] if mask_dir else []
            receipt = {"contract": diag.CONTRACT, "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                       "job_binding_sha256": job_hash, "frame_report": f, "preinject_run_report": r,
                       "verified_source_assets": _used_assets(f), "example_assets": example_assets}
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, path)
            resumed = False
        print(json.dumps({"run": run, "frame_index": frame["frame_index"], "checkpoint_complete": True, "resumed": resumed, "conditionally_imported": imported}), flush=True)
        results.append((f if include else None, r, resumed, imported))
    return results


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--frames", nargs="+", help="Optional exact selections, e.g. AC50:13 AC28:0; each run baseline is measured separately")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true", help="Reuse exact source/method-bound complete per-frame checkpoints")
    parser.add_argument("--import-checkpoints", type=Path, help="New revision: import only proved conditionally equivalent checkpoints from a preserved old executed reporter")
    parser.add_argument("--superseded-pilot-report", type=Path, help="Optional preserved method-review report with known wall/floor-definition error")
    parser.add_argument("--write-example-masks", action="store_true", help="Permitted only for an explicit nonblind selection")
    args = parser.parse_args(argv)
    if str(args.output_dir).startswith("\\\\"):
        raise ValueError("The diagnostic writes new local outputs only")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("Use a fresh diagnostic output directory")
    if not 1 <= args.workers <= 24:
        raise ValueError("The CPU worker cap must be between 1 and 24")
    launch, launch_asset = _bound_json(args.launch)
    candidate = Path(launch["candidate"])
    seal, seal_asset = _bound_json(candidate / "rig_camera_candidate_receipt.json")
    if seal.get("complete") is not True:
        raise ValueError("The diagnostic requires a complete immutable candidate")
    selection = {r.upper() for r in args.runs} if args.runs else {p.name.upper() for p in (candidate / "rig_camera_frames").iterdir() if p.is_dir()}
    selected_frames = {(r.upper(), int(i)) for r, i in (s.split(":") for s in args.frames)} if args.frames else None
    if selected_frames and any(r not in selection for r, _ in selected_frames):
        raise ValueError("Every selected frame must belong to a selected run")
    if args.write_example_masks and (not args.runs or any(r.casefold() in diag.BLIND_RUNS for r in selection)):
        raise ValueError("Example prediction masks require explicit nonblind runs")
    cv2.setNumThreads(1)
    inventories, inventory_binding = annotation_inventory(candidate)
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    mask_dir = args.output_dir / "example_masks" if args.write_example_masks else None
    if mask_dir:
        mask_dir.mkdir(exist_ok=args.resume)
    method_sha = sha256(json.dumps(diag.method_document(), sort_keys=True).encode()).hexdigest()
    immutable_job = {"launch": launch_asset, "candidate_seal": seal_asset,
        "diagnostic_code": _asset(Path(diag.__file__)), "reporter_code": _asset(Path(__file__)),
        "method_sha256": method_sha, "fixed_colour_matrix_sha256": display_contract()["matrix_definition_sha256"],
        "selected_runs": sorted(selection), "selected_frames": sorted(selected_frames) if selected_frames else None,
        "example_masks": args.write_example_masks}
    if args.superseded_pilot_report:
        immutable_job["superseded_method_pilot_report"] = _asset(args.superseded_pilot_report)
        immutable_job["superseded_pilot_limitations"] = "pilot used all polygon intersections including steep sidewalls as floor and required a colour-positive exact floor pixel; retain for audit, do not use as corrected bottom-band measurement"
    conditional_import = verify_conditional_import(args.import_checkpoints, args.output_dir, immutable_job) if args.import_checkpoints else None
    if conditional_import:
        immutable_job["conditional_checkpoint_import"] = conditional_import
    job_hash = sha256(json.dumps(immutable_job, sort_keys=True).encode()).hexdigest()
    start_path = args.output_dir / "job_start.json"
    if args.resume:
        prior = json.loads(start_path.read_text())
        if prior["job_binding_sha256"] != job_hash:
            raise ValueError("Resume requires exactly the same immutable source/method/selection binding")
    else:
        start_path.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "job_binding_sha256": job_hash, "binding": immutable_job}, indent=2), encoding="utf-8")
    run_tasks, run_rows, requested_ids = [], [], set()
    for run in sorted(selection):
        manifest, manifest_asset = _bound_json(candidate / "source_alias/native_frames" / run / "manifest.json")
        frames = manifest["frame_sources"]
        pre = [f for f in frames if f.get("elapsed_h") is not None and float(f["elapsed_h"]) < 0]
        baseline = max(pre, key=lambda f: float(f["elapsed_h"])) if pre else None
        if baseline is None:
            run_rows.append({"run": run, "preinject_status": "unknown: no saved frame strictly before injection", "carryover_proxy": None, "labelled_carryover_status": "missing pre-injection observation, not zero"})
        scored_frames = []
        for frame in frames:
            is_baseline = baseline is frame
            include_frame = selected_frames is None or (run, frame["frame_index"]) in selected_frames
            if include_frame or is_baseline:
                scored_frames.append((frame, is_baseline, include_frame))
            if include_frame:
                requested_ids.add((run, frame["frame_index"]))
        run_tasks.append((candidate, args.output_dir, run, scored_frames, manifest, manifest_asset, inventories.get(run.casefold(), []), job_hash, args.write_example_masks, args.resume, conditional_import))
    if selected_frames is not None and requested_ids != selected_frames:
        raise ValueError("A requested diagnostic frame is not in the sealed selected manifest")
    frame_rows = []
    workers, resources = adaptive_workers(args.workers, len(run_tasks))
    print(json.dumps({"dispatch": resources}), flush=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    resumed_frames = 0
    imported_frames = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(_score_run, run_tasks):
            for frame, run, resumed, imported in result:
                if frame is not None:
                    frame_rows.append(frame)
                if run is not None:
                    run_rows.append(run)
                resumed_frames += resumed
                imported_frames += imported
    wall = time.perf_counter() - started
    runtime = {"created_utc": datetime.now(timezone.utc).isoformat(), "current_execution_wall_seconds": wall,
               "checkpoint_frames": sum(len(t[3]) for t in run_tasks), "resumed_checkpoint_frames": resumed_frames,
               "conditionally_imported_checkpoint_frames": imported_frames,
               "newly_measured_checkpoint_frames": sum(len(t[3]) for t in run_tasks) - resumed_frames - imported_frames,
               "frames_per_wall_second": sum(len(t[3]) for t in run_tasks) / wall,
               "summed_frame_diagnostic_wall_seconds": sum(r["diagnostic_wall_seconds"] for r in frame_rows), "resources": resources}
    (args.output_dir / "job_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    source_binding = {**immutable_job, "label_inventory_source": inventory_binding, "runtime": runtime, "cpu_only": True,
                      "frame_receipt_directory": str(args.output_dir / "frame_receipts")}
    receipt = write_reports(frame_rows, run_rows, args.output_dir, source_binding=source_binding)
    print(json.dumps({"reports": {k: {"frame_count": v["frame_count"], "run_count": v["run_count"], "json_path": v["json"]["path"]} for k, v in receipt["reports"].items()}, "runtime": runtime}))


if __name__ == "__main__":
    main()
