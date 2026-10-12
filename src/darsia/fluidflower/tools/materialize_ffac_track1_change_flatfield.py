"""Materialize the display-only FF_AC change-from-baseline photographs.

The sealed Track1 native manifests, scientific float sidecars, phase maps, and
models are inputs.  Only new ``*.change.jpg`` photographs and a separate
``change_display_manifest.json`` are created beside each run's existing
photographs.  A run's addendum is committed after *all* of its frames exist.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from .ffac_track1_change_flatfield import (
    CONTRACT,
    prepare_flatfield_reference,
    register_flatfield_reference,
    render_change_flatfield,
    smoothed_sand_luminance_cv,
)
from .ffac_track1_display_modes import CLAUDE_CONTRACT_SHA256


DEFAULT_BATCH = Path(
    r"\\UNAS-Pro-8\UiB_FluidFlower\FF_AC\DarSIA\data\prepared_review"
    r"\ffac-track1-v4r5-current43-display-modes-v2-20260930-sealed"
)
ADDENDUM = "change_display_manifest.json"
PHOTO_GEOMETRY_CONTRACT = "ffac-v5-track1-photo-geometry-frame-v1"
PHOTO_COHORT_CONTRACT = "ffac-v5-track1-photo-geometry-cohort-v1"


def _sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest() if path.stat().st_size < 1024 else _hash_stream(stream)


def _hash_stream(stream) -> str:
    digest = sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _bound(path: str | Path, expected: str, label: str) -> Path:
    target = Path(path)
    if not isinstance(expected, str) or len(expected) != 64 or _sha(target) != expected.lower():
        raise ValueError(f"{label}: SHA-256 mismatch or missing input: {target}")
    return target


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_array(path: str | Path, digest: str, *, label: str,
                dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    path = _bound(path, digest, label)
    image = np.load(path, allow_pickle=False, mmap_mode="r")
    if image.dtype != dtype or image.shape != shape:
        raise ValueError(f"{label}: expected {dtype} {shape}, got {image.dtype} {image.shape}")
    return image


def _load_photo(path: str | Path, digest: str, *, canvas_wh: tuple[int, int],
                label: str) -> np.ndarray:
    path = _bound(path, digest, label)
    with Image.open(path) as opened:
        if opened.size != canvas_wh:
            raise ValueError(f"{label}: canvas changed: {opened.size}")
        return np.asarray(opened.convert("RGB"), np.uint8).copy()


def _write_jpeg_exclusive(path: Path, rgb: np.ndarray) -> str:
    """Stage an image in the same directory, then install without replacing.

    Existing outputs are accepted only if byte-for-byte equal to a freshly
    rendered image.  This makes an interrupted run resumable without treating
    an arbitrary existing file as a successful materialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(
                stream, format="JPEG", quality=100, subsampling=0)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha(temporary)
        if path.exists():
            if _sha(path) != digest:
                raise FileExistsError(f"Existing change photograph differs: {path}")
            return digest
        try:
            # On Windows, rename is atomic and fails if the destination exists.
            temporary.rename(path)
        except FileExistsError:
            if _sha(path) != digest:
                raise FileExistsError(f"Concurrent change photograph differs: {path}")
        if _sha(path) != digest:
            raise ValueError(f"Committed change photograph changed: {path}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: dict) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha(temporary)
        if path.exists():
            if path.read_bytes() != encoded:
                raise FileExistsError(f"Existing change addendum differs: {path}")
            return digest
        try:
            temporary.rename(path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(f"Concurrent change addendum differs: {path}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _photo_geometry(native: dict, index: int, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    marker = native.get("v5_track1_photo_geometry") or {}
    rows = marker.get("frames") or []
    if (marker.get("contract") != PHOTO_COHORT_CONTRACT
            or [row.get("frame_index") for row in rows] != list(range(len(native["frame_sources"])))
            or index >= len(rows)):
        raise ValueError("Track1 source lacks its complete per-frame photo geometry")
    bound = rows[index]
    receipt_path = _bound(bound["receipt_path"], bound["receipt_sha256"],
                          f"frame {index} geometry receipt")
    receipt = _read(receipt_path)
    if (receipt.get("contract") != PHOTO_GEOMETRY_CONTRACT
            or receipt.get("run") != native["run"]
            or receipt.get("frame_index") != index
            or receipt.get("source_manifest_sha256") != marker.get("original_source_manifest_sha256")
            or receipt.get("original_source_manifest_sha256") != marker.get("original_source_manifest_sha256")
            or receipt.get("native_shape_hw") != list(shape)
            or receipt.get("camera_jpeg_sha256") != native["frame_sources"][index].get("source_sha256")
            or receipt.get("v5_B_camera_jpeg_sha256") != (
                (native.get("reference") or {}).get("v5_reference_state") or {}).get("camera_jpeg_sha256")
            or receipt.get("temporal_registration_pass") is not True):
        raise ValueError(f"Frame {index}: photo geometry receipt is not bound to this native timeline")
    h = np.asarray(receipt.get("H_v5_B_to_frame_native"), np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-9:
        raise ValueError(f"Frame {index}: invalid baseline-to-frame H")
    return h, {"receipt_path": str(receipt_path), "receipt_sha256": bound["receipt_sha256"],
               "H_v5_B_to_frame_native": h.tolist()}


def _render_canvas(current: np.ndarray, registered_reference, sand: np.ndarray,
                   lift_canvas: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, dict]:
    x, y, width, height = box
    crop = np.ascontiguousarray(lift_canvas[y:y + height, x:x + width])
    changed, metadata = render_change_flatfield(
        current, registered_reference, sand, lift1_rgb=crop)
    if changed.shape != (height, width, 3) or changed.dtype != np.uint8:
        raise ValueError("Change renderer returned another native grid or dtype")
    lift_canvas[y:y + height, x:x + width] = changed
    return lift_canvas, metadata


def _manifest_shape(native: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    shape = tuple(native.get("analysis_image_shape_hw") or ())
    canvas = tuple(native.get("canvas_wh") or ())
    box = tuple(native.get("content_box_xywh") or ())
    if (len(shape) != 2 or len(canvas) != 2 or len(box) != 4
            or shape != (box[3], box[2])
            or min(shape) <= 0 or min(canvas) <= 0
            or box[0] < 0 or box[1] < 0
            or box[0] + box[2] > canvas[0] or box[1] + box[3] > canvas[1]):
        raise ValueError("Native manifest has an inconsistent analysis/canvas grid")
    return shape, canvas, box


def _existing_complete(path: Path, native: dict, native_sha: str) -> dict | None:
    if not path.exists():
        return None
    record = _read(path)
    frames = record.get("frames") or []
    if (record.get("contract") != CONTRACT or record.get("run") != native["run"]
            or record.get("source_manifest_sha256") != native_sha
            or len(frames) != len(native["frame_sources"])
            or [row.get("frame_index") for row in frames] != list(range(len(frames)))):
        raise ValueError(f"Existing change addendum does not match current native manifest: {path}")
    for row in [record["reference"], *frames]:
        _bound(row["jpeg_path"], row["jpeg_sha256"], "existing change photograph")
    return record


def materialize_run(native_manifest_path: str | Path) -> dict:
    """Create one complete, SHA-bound run; safe to repeat after interruption."""
    cv2.setNumThreads(1)
    path = Path(native_manifest_path)
    manifest_sha = _sha(path)
    native = _read(path)
    run = str(native.get("run") or "").lower()
    frames = native.get("frame_sources") or []
    if (not run.startswith("ac") or not run[2:].isdigit() or not frames
            or [f.get("frame_index") for f in frames] != list(range(len(frames)))
            or int(native.get("frame_count") or -1) != len(frames)):
        raise ValueError(f"Native manifest has an invalid run/timeline: {path}")
    shape, canvas_wh, box = _manifest_shape(native)
    if path != path.parent / "manifest.json" or path.parent.name.lower() != run:
        raise ValueError("Native manifest is not under native_frames/<run>/manifest.json")
    if (native.get("v5_temporal_reference") or {}).get("contract") != "ffac-v5-native-crop-source-supported-v1":
        raise ValueError(f"{run}: unsupported pre-injection source geometry")
    marker = native.get("v5_track1_photo_geometry") or {}
    _bound(marker["original_source_manifest_path"],
           marker["original_source_manifest_sha256"],
           f"{run} original source manifest")
    addendum = path.parent / ADDENDUM
    complete = _existing_complete(addendum, native, manifest_sha)
    if complete is not None:
        return {"run": run, "frames": len(frames), "addendum_path": str(addendum),
                "addendum_sha256": _sha(addendum), "frame0_sand_luminance_cv":
                complete.get("frame0_sand_luminance_cv"), "resumed": True,
                "nearest_2p25": complete.get("nearest_2p25")}

    reference = native.get("reference") or {}
    _bound(reference["v5_baseline_receipt_path"],
           reference["v5_baseline_receipt_sha256"],
           f"{run} canonical pre-injection receipt")
    review_root = path.parent.parent.parent
    expected_baseline = review_root / "native_references" / run / "pre_injection.log_linear_rgb.npy"
    if Path(reference.get("v5_log_linear_rgb_path") or "") != expected_baseline:
        raise ValueError(f"{run}: reference is not the bound pre-injection float source")
    ref_float = _load_array(reference["v5_log_linear_rgb_path"],
                            reference["v5_log_linear_rgb_sha256"],
                            label=f"{run} pre-injection float", dtype=np.dtype("float32"),
                            shape=(3, *shape))
    ref_valid = _load_array(reference["v5_log_linear_valid_mask_path"],
                            reference["v5_log_linear_valid_mask_sha256"],
                            label=f"{run} pre-injection validity", dtype=np.dtype("bool"),
                            shape=shape)
    prepared = prepare_flatfield_reference(ref_float, source_valid_mask=ref_valid)
    del ref_float, ref_valid

    ref_photo = (reference.get("v5_display_modes") or {}).get("lift1") or {}
    baseline_canvas = _load_photo(ref_photo["jpeg_path"], ref_photo["jpeg_sha256"],
                                  canvas_wh=canvas_wh, label=f"{run} pre-injection lift1")
    ref_sand = _load_array(reference["v5_photo_sand_mask_path"],
                           reference["v5_photo_sand_mask_sha256"],
                           label=f"{run} pre-injection sand", dtype=np.dtype("bool"),
                           shape=shape)
    registered_ref = register_flatfield_reference(prepared, np.eye(3), target_shape_hw=shape)
    # The reference itself is a photograph for Studio's reference view, not a
    # timeline frame.  The full canvas and its content-box remain unchanged.
    ref_current = _load_array(reference["v5_log_linear_rgb_path"],
                              reference["v5_log_linear_rgb_sha256"],
                              label=f"{run} pre-injection float", dtype=np.dtype("float32"),
                              shape=(3, *shape))
    baseline_changed, baseline_meta = _render_canvas(
        ref_current, registered_ref, ref_sand, baseline_canvas, box)
    ref_target = review_root / "native_references" / run / "pre_injection.change.jpg"
    ref_digest = _write_jpeg_exclusive(ref_target, baseline_changed)
    reference_record = {
        "jpeg_path": str(ref_target), "jpeg_sha256": ref_digest,
        "source_float_sha256": reference["v5_log_linear_rgb_sha256"],
        "source_valid_mask_sha256": reference["v5_log_linear_valid_mask_sha256"],
        "source_photo_sand_sha256": reference["v5_photo_sand_mask_sha256"],
        "source_lift1_sha256": ref_photo["jpeg_sha256"],
        "render": baseline_meta,
    }
    del ref_current, ref_sand, baseline_canvas, baseline_changed, registered_ref

    rows = []
    for frame in frames:
        index = int(frame["frame_index"])
        h, geometry_record = _photo_geometry(native, index, shape)
        registered = register_flatfield_reference(prepared, h, target_shape_hw=shape)
        current = _load_array(frame["v5_log_linear_rgb_path"],
                              frame["v5_log_linear_rgb_sha256"],
                              label=f"{run} frame {index} float", dtype=np.dtype("float32"),
                              shape=(3, *shape))
        # This scientific validity is verified but never changed.  The display
        # renderer uses finite log pixels and its bounded small-hole repair.
        _load_array(frame["v5_log_linear_valid_mask_path"],
                    frame["v5_log_linear_valid_mask_sha256"],
                    label=f"{run} frame {index} validity", dtype=np.dtype("bool"),
                    shape=shape)
        sand = _load_array(frame["v5_photo_sand_mask_path"],
                           frame["v5_photo_sand_mask_sha256"],
                           label=f"{run} frame {index} sand", dtype=np.dtype("bool"),
                           shape=shape)
        lift = (frame.get("v5_display_modes") or {}).get("lift1") or {}
        canvas = _load_photo(lift["jpeg_path"], lift["jpeg_sha256"],
                             canvas_wh=canvas_wh, label=f"{run} frame {index} lift1")
        changed, metadata = _render_canvas(current, registered, sand, canvas, box)
        target = path.parent / f"frame_{index:03d}.change.jpg"
        digest = _write_jpeg_exclusive(target, changed)
        frame0_cv = None
        if index == 0:
            # The acceptance number describes the decoded artifact in Studio,
            # including its final JPEG quantisation, not a pre-save array.
            with Image.open(target) as decoded:
                x, y, width, height = box
                actual = np.asarray(decoded.convert("RGB"), np.uint8)[
                    y:y + height, x:x + width]
            frame0_cv = smoothed_sand_luminance_cv(actual, sand)
        rows.append({"frame_index": index, "elapsed_h": frame.get("elapsed_h"),
                     "jpeg_path": str(target), "jpeg_sha256": digest,
                     "source_float_sha256": frame["v5_log_linear_rgb_sha256"],
                     "source_valid_mask_sha256": frame["v5_log_linear_valid_mask_sha256"],
                     "source_photo_sand_sha256": frame["v5_photo_sand_mask_sha256"],
                     "source_lift1_sha256": lift["jpeg_sha256"],
                     "geometry": geometry_record, "render": metadata,
                     **({"sand_luminance_cv_box5": frame0_cv} if index == 0 else {})})
        del current, sand, canvas, changed, registered

    nearest = min(rows, key=lambda row: abs(float(row["elapsed_h"]) - 2.25))
    record = {
        "contract": CONTRACT, "display_only": True,
        "run": run, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": str(path), "source_manifest_sha256": manifest_sha,
        "source_display_contract_sha256": CLAUDE_CONTRACT_SHA256,
        "analysis_image_shape_hw": list(shape), "canvas_wh": list(canvas_wh),
        "content_box_xywh": list(box), "frame_count": len(rows),
        "pre_injection_reference_kind": (reference.get("v5_reference_state") or {}).get("kind"),
        "reference": reference_record, "frames": rows,
        "frame0_sand_luminance_cv": rows[0]["sand_luminance_cv_box5"],
        "frame0_cv_le_0p10": rows[0]["sand_luminance_cv_box5"] <= 0.10,
        "nearest_2p25": {"frame_index": nearest["frame_index"],
                          "elapsed_h": nearest["elapsed_h"],
                          "jpeg_path": nearest["jpeg_path"],
                          "jpeg_sha256": nearest["jpeg_sha256"]},
        "scientific_sidecars_phase_maps_and_model_modified": False,
    }
    if _sha(path) != manifest_sha:
        raise ValueError(f"{run}: sealed native manifest changed while rendering")
    digest = _write_json_exclusive(addendum, record)
    return {"run": run, "frames": len(rows), "addendum_path": str(addendum),
            "addendum_sha256": digest, "frame0_sand_luminance_cv":
            rows[0]["sand_luminance_cv_box5"], "resumed": False,
            "nearest_2p25": record["nearest_2p25"]}


def discover_manifests(batch_root: str | Path) -> list[Path]:
    root = Path(batch_root)
    found = sorted((root / "revisions").glob(
        "*/manual_pairwise_review/native_frames/ac*/manifest.json"),
        key=lambda path: path.parent.name)
    names = [path.parent.name for path in found]
    if len(names) != len(set(names)):
        raise ValueError("Sealed prepared batch contains duplicate AC run manifests")
    return found


def materialize_batch(batch_root: str | Path, *, runs: tuple[str, ...] | None = None,
                      workers: int = 8, expect_runs: int | None = None,
                      expect_frames: int | None = None) -> dict:
    """Bounded per-run CPU concurrency; a run reuses its baseline in one worker."""
    if not 1 <= workers <= 24:
        raise ValueError("Workers must be between 1 and 24")
    manifests = discover_manifests(batch_root)
    wanted = set(runs or ())
    if wanted:
        missing = wanted - {path.parent.name for path in manifests}
        if missing:
            raise ValueError(f"Requested run has no sealed native manifest: {sorted(missing)}")
        manifests = [path for path in manifests if path.parent.name in wanted]
    if expect_runs is not None and len(manifests) != expect_runs:
        raise ValueError(f"Expected {expect_runs} runs, found {len(manifests)}")
    if not manifests:
        raise ValueError("No sealed Track1 native manifests found")
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(manifests))) as pool:
        futures = {pool.submit(materialize_run, path): path.parent.name for path in manifests}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "run_complete", **result}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: row["run"])
    total = sum(row["frames"] for row in results)
    if expect_frames is not None and total != expect_frames:
        raise ValueError(f"Expected {expect_frames} frames, found {total}")
    return {"contract": CONTRACT, "batch_root": str(batch_root),
            "runs": len(results), "frames": total,
            "frame0_cv_le_0p10_all": all(
                row["frame0_sand_luminance_cv"] is not None
                and row["frame0_sand_luminance_cv"] <= 0.10 for row in results),
            "per_run": results}


def _figure(result: dict, report_dir: Path) -> Path:
    run = result["run"]
    addendum = _read(Path(result["addendum_path"]))
    selected = addendum["nearest_2p25"]
    frame = addendum["frames"][selected["frame_index"]]
    native = _read(Path(addendum["source_manifest_path"]))
    lift = native["frame_sources"][frame["frame_index"]]["v5_display_modes"]["lift1"]
    left = _load_photo(lift["jpeg_path"], lift["jpeg_sha256"],
                       canvas_wh=tuple(native["canvas_wh"]), label=f"{run} figure lift1")
    right = _load_photo(frame["jpeg_path"], frame["jpeg_sha256"],
                        canvas_wh=tuple(native["canvas_wh"]), label=f"{run} figure change")
    max_width = 1600
    with Image.fromarray(left) as a, Image.fromarray(right) as b:
        height = round(a.height * max_width / a.width)
        a = a.resize((max_width, height), Image.Resampling.LANCZOS)
        b = b.resize((max_width, height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (2 * max_width, height), (0, 0, 0))
        panel.paste(a, (0, 0))
        panel.paste(b, (max_width, 0))
        target = report_dir / f"{run}_near_2p25_lift_vs_change.jpg"
        _write_jpeg_exclusive(target, np.asarray(panel))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--runs", nargs="+", metavar="ACNN")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    wanted = (None if args.all else tuple(run.lower() for run in args.runs))
    result = materialize_batch(args.batch_root, runs=wanted, workers=args.workers,
                               expect_runs=43 if args.all else len(wanted),
                               expect_frames=864 if args.all else None)
    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        figures = {}
        for row in result["per_run"]:
            if row["run"] in {"ac28", "ac56"}:
                figures[row["run"]] = str(_figure(row, args.report_dir))
        result["figures"] = figures
        report = args.report_dir / "change_flatfield_materialization_report.json"
        _write_json_exclusive(report, result)
        print(json.dumps({"event": "report", "path": str(report),
                          "sha256": _sha(report)}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "complete", "runs": result["runs"],
                      "frames": result["frames"],
                      "frame0_cv_le_0p10_all": result["frame0_cv_le_0p10_all"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
