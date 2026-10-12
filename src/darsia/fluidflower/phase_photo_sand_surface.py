"""Source-bound, removal-only transport of an accepted photographic sand surface.

The reference curve is fitted and reviewed separately.  This module does not read
phase maps, probabilities, training annotations, or RAW pixels.  It preserves
the existing support below the curve and transports the curve with the exact
per-frame sand registration.  A pre-injection reference has its own geometry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


FORMAT = "ffac-photo-sand-top-revision-v1"
METHOD = "frozen-photo-surface-exact-homography-ceil-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _asset(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Bound surface source changed: {path}")
    return {"path": str(path), "sha256": digest}


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite an existing surface artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    temporary.replace(path)


def _atomic_json(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    _atomic_bytes(path, encoded)


def _homography(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Surface registration must be a finite 3x3 homography")
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Surface registration homography is singular")
    return matrix


def reference_homography(frame_geometry):
    """Map AC28 to the independently stored B reference, not timeline frame 0."""
    current = _homography(frame_geometry["H_ac28_to_frame_native"])
    baseline_to_current = _homography(frame_geometry["H_v5_B_to_frame_native"])
    return _homography(np.linalg.solve(baseline_to_current, current))


def transport_surface_curve(curve, homography, shape_hw):
    """Transport a photograph-derived top edge without new margins or erosion.

    Endpoint values extend to columns outside the transformed reference span,
    exactly as in the accepted photographic contact sheets.  This is explicit
    and is recorded; it is not a claim of independent edge accuracy there.
    """
    curve = np.asarray(curve)
    height, width = map(int, shape_hw)
    if curve.ndim != 1 or curve.size < 2 or not np.issubdtype(curve.dtype, np.floating):
        raise ValueError("Photographic surface curve must be a floating 1D array")
    if not np.isfinite(curve).all() or min(height, width) < 2:
        raise ValueError("Invalid surface curve or native image dimensions")
    matrix = _homography(homography)
    points = np.vstack((np.arange(curve.size, dtype=np.float64), curve, np.ones(curve.size)))
    transformed = matrix @ points
    denominator = transformed[2]
    if np.any(np.abs(denominator) < 1e-10) or np.any(denominator * denominator[0] <= 0):
        raise ValueError("Surface homography crosses a projective horizon")
    xy = transformed[:2] / denominator
    if not np.isfinite(xy).all() or np.any(np.diff(xy[0]) <= 0):
        raise ValueError("Registered surface must remain strictly increasing in x")
    result = np.interp(np.arange(width, dtype=np.float64), xy[0], xy[1])
    if np.any(result < 0) or np.any(result > height - 1):
        raise ValueError("Registered surface lies outside the native image")
    return result


def top_keep_from_curve(curve, shape_hw):
    height, width = map(int, shape_hw)
    edge = np.asarray(curve, dtype=np.float64)
    if edge.shape != (width,) or not np.isfinite(edge).all():
        raise ValueError("Surface curve differs from the native image width")
    if np.any(edge < 0) or np.any(edge > height - 1):
        raise ValueError("Surface curve lies outside the native image")
    # Native pixel centres at or below the edge are retained.  No extra margin.
    return np.arange(height, dtype=np.int32)[:, None] >= np.ceil(edge).astype(np.int32)[None, :]


def revise_photo_sand(original, top_keep):
    original, top_keep = np.asarray(original), np.asarray(top_keep)
    if original.dtype != np.bool_ or top_keep.dtype != np.bool_ or original.shape != top_keep.shape:
        raise ValueError("Original photo support and top keep must share a native boolean grid")
    if original.ndim != 2 or np.any(top_keep[:-1] & ~top_keep[1:]) or not np.all(top_keep[-1]):
        raise ValueError("Top keep must remove only a column prefix, keeping every bottom pixel")
    return original & top_keep


def _bound_original(record, shape):
    path = Path(record["v5_photo_sand_mask_path"])
    encoded = path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != record["v5_photo_sand_mask_sha256"]:
        raise ValueError(f"Original photograph-only support changed: {path}")
    original = np.load(io.BytesIO(encoded), allow_pickle=False)
    if original.dtype != np.bool_ or original.shape != tuple(shape):
        raise ValueError("Original photo-sand support differs from its native boolean grid")
    return original, {"path": str(path), "sha256": digest}


def _build_one(task, curve, curve_asset, request_sha, output):
    run, index, reference = task["run"], task["frame_index"], task["reference"]
    shape, record = task["native_shape_hw"], task["record"]
    original, original_asset = _bound_original(record, shape)
    transported = transport_surface_curve(curve, task["homography"], shape)
    keep = top_keep_from_curve(transported, shape)
    revised = revise_photo_sand(original, keep)
    stem = "pre_injection" if reference else f"frame_{index:03d}"
    folder = Path(output) / "masks" / run
    mask_path, row_path = folder / (stem + ".npz"), folder / (stem + ".json")
    binding = dict(request_sha256=request_sha, run=run, frame_index=index, reference=reference,
                   elapsed_h=float(record["elapsed_h"]), native_manifest=task["native_manifest"],
                   original_photo_mask=original_asset, registration_receipt=task["registration_receipt"],
                   curve=curve_asset, homography=task["homography"])
    binding_sha = canonical_sha256(binding)
    if row_path.exists():
        saved = _read(row_path)
        if saved.get("binding_sha256") != binding_sha:
            raise ValueError(f"Surface resume binding differs: {row_path}")
        _asset(saved["mask"]["path"], saved["mask"]["sha256"])
        with np.load(mask_path, allow_pickle=False) as archive:
            if any(not np.array_equal(archive[key], expected) for key, expected in
                   (("top_keep", keep), ("photo_sand_original", original), ("photo_sand_revised", revised))):
                raise ValueError("Surface resume arrays differ from frozen geometry")
        return saved
    encoded = io.BytesIO()
    np.savez_compressed(encoded, top_keep=keep, photo_sand_original=original, photo_sand_revised=revised)
    _atomic_bytes(mask_path, encoded.getvalue())
    row = dict(run=run, frame_index=index, reference=reference,
               elapsed_h=float(record["elapsed_h"]), native_manifest=task["native_manifest"],
               native_manifest_path=task["native_manifest"]["path"],
               native_manifest_sha256=task["native_manifest"]["sha256"],
               native_shape_hw=list(shape), content_box_xywh=task["content_box_xywh"],
               original_photo_mask=original_asset, mask=_asset(mask_path),
               registration_receipt=task["registration_receipt"],
               source_photo={"path": record["source_path"], "sha256": record["source_sha256"]},
               photographic_display=record["v5_display_modes"]["natural"],
               H_ac28_to_frame_native=task["homography"], curve=curve_asset,
               curve_min_px=float(transported.min()), curve_max_px=float(transported.max()),
               top_excluded_px=int(np.count_nonzero(~keep)),
               original_photo_sand_px=int(np.count_nonzero(original)),
               revised_photo_sand_px=int(np.count_nonzero(revised)),
               photo_sand_removed_px=int(np.count_nonzero(original & ~keep)),
               photo_support_added_px=0, pixels_below_surface_changed=0,
               binding_sha256=binding_sha, method=METHOD,
               phase_predictions_used=False, phase_labels_used=False, acceptance_margin_px=0)
    if reference:
        row["H_ac28_to_reference_native"] = task["homography"]
    _atomic_json(row_path, row)
    return row


def build_photo_sand_top_bundle(export_root, reference_receipt_path, output, *, workers=8,
                               expected_runs=43, expected_frames=864):
    """Build native mask assets for every exact timeline and its own reference."""
    started = time.perf_counter()
    export_root, output = Path(export_root), Path(output).resolve()
    job_path = export_root / "job_request.json"
    job, reference_receipt = _read(job_path), _read(reference_receipt_path)
    inventory = job["binding"]["source_inventory"]
    if canonical_sha256(inventory) != job["binding"]["source_inventory_sha256"]:
        raise ValueError("Exact source inventory binding changed")
    unique = {(row["run"], int(row["frame_index"])) for row in inventory}
    runs = sorted({row["run"] for row in inventory})
    if len(unique) != len(inventory) or len(runs) != expected_runs or len(inventory) != expected_frames:
        raise ValueError("Surface bundle requires the exact expected run/frame inventory")
    if reference_receipt.get("accepted_photo_surface") is not True:
        raise ValueError("Surface bundle requires an explicitly reviewed photographic reference")
    if reference_receipt.get("phase_predictions_used") or reference_receipt.get("phase_labels_used"):
        raise ValueError("Photographic surface must not be fitted to phase outcomes")
    curve_asset = _asset(**dict(path=reference_receipt["curve"]["path"], expected=reference_receipt["curve"]["sha256"]))
    curve = np.load(curve_asset["path"], allow_pickle=False)
    for asset in reference_receipt.get("verified_source_assets", []):
        _asset(asset["path"], asset["sha256"])
    request = dict(contract=FORMAT, method=METHOD, source_inventory_sha256=job["binding"]["source_inventory_sha256"],
                   export_job_request=_asset(job_path), reference_receipt=_asset(reference_receipt_path),
                   curve=curve_asset, code=_asset(__file__), operation="clip-only-after-approved-g3",
                   outside_phase_label=255, outside_phase_display="transparent", acceptance_margin_px=0,
                   expected_runs=expected_runs, expected_frames=expected_frames)
    request_sha = canonical_sha256(request)
    _atomic_json(output / "request.json", dict(request=request, request_sha256=request_sha))
    natives, tasks = {}, []
    for item in inventory:
        run, index = item["run"], int(item["frame_index"])
        native_asset = {"path": item["native_manifest_path"], "sha256": item["native_manifest_sha256"]}
        if run not in natives:
            _asset(native_asset["path"], native_asset["sha256"])
            native = _read(native_asset["path"])
            if native["run"] != run:
                raise ValueError("Surface source identifies another run")
            natives[run] = native
        native = natives[run]
        if index >= len(native["frame_sources"]):
            raise ValueError("Surface frame lies outside its exact native timeline")
        record = native["frame_sources"][index]
        if float(record["elapsed_h"]) != float(item["elapsed_h"]):
            raise ValueError("Surface frame time differs from frozen source inventory")
        geometry_item = next(row for row in native["v5_track1_photo_geometry"]["frames"] if int(row["frame_index"]) == index)
        geometry_asset = _asset(geometry_item["receipt_path"], geometry_item["receipt_sha256"])
        geometry = _read(geometry_asset["path"])
        probability_receipt = _read(export_root / "layers" / run / f"frame_{index:03d}" / "receipt.json")
        if probability_receipt["metadata"]["domain_binding"]["frame_geometry_receipt_sha256"] != geometry_asset["sha256"]:
            raise ValueError("Frame registration differs from the approved export source")
        if probability_receipt["binding"]["native_manifest_sha256"] != native_asset["sha256"]:
            raise ValueError("Approved export identifies another native source")
        if (geometry["run"] != run or int(geometry["frame_index"]) != index or
            geometry["source_manifest_sha256"] != native["v5_track1_photo_geometry"]["original_source_manifest_sha256"] or
            geometry["camera_jpeg_sha256"] != record["source_sha256"]):
            raise ValueError("Surface registration/source photograph binding differs")
        shape = native["analysis_image_shape_hw"]
        if list(shape) != list(item["native_shape_hw"]) or list(shape) != geometry["native_shape_hw"]:
            raise ValueError("Surface registration native shape differs")
        common = dict(run=run, native_manifest=native_asset, native_shape_hw=shape,
                      content_box_xywh=native["content_box_xywh"])
        tasks.append(dict(common, frame_index=index, reference=False, record=record,
                          registration_receipt=geometry_asset, homography=geometry["H_ac28_to_frame_native"]))
        if index == 0:
            reference = native["reference"]
            baseline_asset = _asset(reference["v5_baseline_receipt_path"], reference["v5_baseline_receipt_sha256"])
            ref_matrix = reference_homography(geometry).tolist()
            derived_path = output / "reference_registrations" / (run + ".json")
            _atomic_json(derived_path, dict(contract="ffac-photo-sand-independent-reference-geometry-v1", run=run,
                frame_index=None, reference=True, native_manifest=native_asset, baseline_receipt=baseline_asset,
                original_frame_registration=geometry_asset, H_ac28_to_reference_native=ref_matrix,
                equation="inv(H_v5_B_to_frame_native) @ H_ac28_to_frame_native",
                reference_state=reference["v5_reference_state"], native_shape_hw=shape))
            tasks.append(dict(common, frame_index=None, reference=True, record=reference,
                              registration_receipt=_asset(derived_path), homography=ref_matrix))
    if len([task for task in tasks if task["reference"]]) != expected_runs:
        raise ValueError("Every timeline requires its separately bound pre-injection reference")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_build_one, task, curve, curve_asset, request_sha, output) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 50 == 0:
                print(json.dumps(dict(complete=len(results), total=len(tasks), seconds=round(time.perf_counter()-started, 2))), flush=True)
    frames = sorted((row for row in results if not row["reference"]), key=lambda row: (row["run"], row["frame_index"]))
    references = sorted((row for row in results if row["reference"]), key=lambda row: row["run"])
    manifest = dict(contract=FORMAT, complete=True, request_sha256=request_sha, request=request,
                    frames=frames, references=references, run_count=len(runs), frame_count=len(frames),
                    reference_count=len(references), native_mask_count=len(results),
                    photo_sand_removed_px=sum(row["photo_sand_removed_px"] for row in frames),
                    reference_photo_sand_removed_px=sum(row["photo_sand_removed_px"] for row in references),
                    photo_support_added_px=0, pixels_below_surface_changed=0,
                    phase_predictions_used=False, phase_labels_used=False,
                    acceptance="photo-reviewed conservative water/sand surface; removal only",
                    uncertainty="Grain-level water/sand edge uncertainty remains; no 2 px physical-edge guarantee. "
                                "The failed independent per-run detector audit is retained and is not an acceptance certificate.",
                    reference_photo_audit=reference_receipt,
                    operation="clip-only-after-approved-g3", outside_phase_label=255, outside_phase_display="transparent")
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    result = dict(contract=FORMAT, complete=True, manifest=_asset(manifest_path), runs=len(runs),
                  frames=len(frames), references=len(references), masks=len(results),
                  seconds=round(time.perf_counter()-started, 3), workers=workers,
                  source_inventory_sha256=request["source_inventory_sha256"])
    _atomic_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--reference-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(build_photo_sand_top_bundle(args.export_root, args.reference_receipt, args.output,
                                               workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
