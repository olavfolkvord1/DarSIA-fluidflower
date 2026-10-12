"""Opt-in, read-only FF_AC sequential common-start baseline preflight.

Select the 43 canonical Studio revisions from a frozen request's parent
postflight. Each CPU worker opens one run's calibrated rig, checks the frozen
source and geometry, and fits the fixed common-start correction on the
pre-injection baseline alone. No timeline frame is read or written here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import (
    AC47_V4_SOURCE_SHA256,
    SEQUENTIAL_CHART_EXTRACTOR_REVISION,
    _chart_qc,
    _common_chart_qc,
    _frozen_registration_profile,
    _hash,
    _prepared_baseline_source,
    _read,
    _require_matching_geometry,
    _sequential_baseline_fit,
    _source_photometry_revision,
)
from darsia.fluidflower.tools.schema2_phase_transfer_campaign import (
    _array_sha256,
    _correction_diagnostics,
    _open_ffac,
)


FORMAT = "ffac-sequential-baseline-preflight-v1"
SELECTION_POLICY = "frozen-parent-postflight-43-canonical-studio-revisions-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_immutable(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Preserve an existing receipt only when its deterministic fields agree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = _read(path)
        without_time = lambda row: {key: val for key, val in row.items() if key != "created_utc"}
        if without_time(saved) != without_time(value):
            raise FileExistsError(f"Immutable preflight receipt differs: {path}")
        return saved
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def _jpeg_exif(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        metadata = image.getexif()
        capture = metadata.get_ifd(34665)
        return {
            "reader": "Pillow",
            "format": image.format,
            "shape_wh": list(image.size),
            "make": metadata.get(271),
            "model": metadata.get(272),
            "datetime": metadata.get(306),
            "datetime_original": capture.get(36867),
            "offset_original": capture.get(36881),
            "subsec_original": capture.get(37521),
        }


def _arw_exif(path: Path) -> dict[str, Any]:
    try:
        import tifffile
    except ImportError:
        return {"reader": "unavailable", "reason": "tifffile is not installed"}
    try:
        with tifffile.TiffFile(path) as image:
            page = image.pages[0]
            tags = page.tags
            capture = tags[34665].value if 34665 in tags else {}
            return {
                "reader": "tifffile",
                "shape_wh": [int(capture.get("PixelXDimension", page.imagewidth)),
                             int(capture.get("PixelYDimension", page.imagelength))],
                "preview_shape_wh": [int(page.imagewidth), int(page.imagelength)],
                "make": tags[271].value if 271 in tags else None,
                "model": tags[272].value if 272 in tags else None,
                "datetime": tags[306].value if 306 in tags else None,
                "datetime_original": capture.get("DateTimeOriginal"),
                "offset_original": capture.get("OffsetTimeOriginal"),
                "subsec_original": capture.get("SubsecTimeOriginal"),
            }
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"reader": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def _baseline_pair_inventory(jpeg: Path, expected_sha256: str) -> dict[str, Any]:
    if jpeg.suffix.casefold() not in {".jpg", ".jpeg"}:
        raise ValueError(f"Rig baseline is not a JPEG: {jpeg}")
    digest = _hash(jpeg)
    if digest != expected_sha256:
        raise ValueError(f"Frozen baseline JPEG SHA-256 changed: {jpeg}")
    jpeg_meta = _jpeg_exif(jpeg)
    arw = jpeg.with_suffix(".ARW")
    if not arw.is_file():
        alternatives = [path for path in jpeg.parent.glob(jpeg.stem + ".*")
                        if path.suffix.casefold() == ".arw"]
        arw = alternatives[0] if len(alternatives) == 1 else arw
    inventory: dict[str, Any] = {
        "scope": "rig-pre-injection-acquisition-only",
        "fit_input_format": "JPEG",
        "jpeg": {"path": str(jpeg), "sha256": digest, "exif": jpeg_meta},
        "arw": {"path": str(arw), "available": arw.is_file()},
    }
    if arw.is_file():
        arw_meta = _arw_exif(arw)
        inventory["arw"].update(sha256=_hash(arw), exif=arw_meta)
        if arw_meta.get("reader") != "unavailable":
            comparable = ("shape_wh", "make", "model", "datetime", "datetime_original",
                          "offset_original", "subsec_original")
            checked = [key for key in comparable
                       if jpeg_meta.get(key) is not None and arw_meta.get(key) is not None]
            # A camera may crop or embed a preview differently in its RAW and
            # JPEG files. Record that geometry, but pair identity uses capture
            # metadata rather than requiring equal pixel dimensions.
            mismatched = [key for key in checked
                          if key != "shape_wh" and jpeg_meta[key] != arw_meta[key]]
            inventory["exif_pair_check"] = {"checked_fields": checked,
                                             "mismatched_fields": mismatched,
                                             "status": "match" if not mismatched else "mismatch"}
            if ("shape_wh" in checked and
                    jpeg_meta["shape_wh"] != arw_meta["shape_wh"]):
                inventory["exif_pair_check"]["capture_dimensions"] = {
                    "jpeg_wh": jpeg_meta["shape_wh"], "arw_wh": arw_meta["shape_wh"],
                    "status": "different_geometry_recorded"}
        else:
            inventory["exif_pair_check"] = {"status": "unavailable"}
    else:
        inventory["exif_pair_check"] = {"status": "arw_absent"}
    return inventory


def canonical_revisions(request_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind each parent postflight run to one exact request-bundled source."""
    request = _read(request_path)
    if request["settings"].get("preserve_native_revisions") is not True:
        raise ValueError("Frozen request must preserve native revisions")
    parent_path = Path(request["settings"]["parent_model"])
    postflight_path = parent_path / "review_source" / "postflight_manifest.json"
    postflight = _read(postflight_path)
    jobs = postflight["jobs"]
    if (len(jobs) != 43 or len({str(job["run"]).casefold() for job in jobs}) != 43
            or any(str(job.get("dataset", "")).casefold() != "ffac" for job in jobs)):
        raise ValueError("Parent postflight must contain 43 unique FF_AC Studio runs")
    bundled: dict[str, set[str]] = {}
    for bundle in request["bundles"]:
        combined_spec = bundle["combined_manifest"]
        combined_path = Path(combined_spec["snapshot_path"])
        if _hash(combined_path) != combined_spec["sha256"]:
            raise ValueError(f"Frozen combined manifest SHA changed: {combined_path}")
        combined = _read(combined_path)
        for source_job in combined.get("jobs") or []:
            if str(source_job.get("dataset", "")).casefold() != "ffac":
                continue
            source_run = str(source_job["run"]).casefold()
            source_root = analysis_review_root(
                Path(source_job.get("analysis_source_review_root") or
                     source_job["review_output_dir"]), source_run)
            path = source_root / "native_frames" / source_run / "manifest.json"
            bundled.setdefault(str(path.resolve()).casefold(), set()).add(source_run)
    rows: list[dict[str, Any]] = []
    for job in jobs:
        run = str(job["run"]).casefold()
        root = analysis_review_root(
            Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
        manifest_path = root / "native_frames" / run / "manifest.json"
        selected_from_request = run in bundled.get(str(manifest_path.resolve()).casefold(), set())
        context = job.get("phase_transfer_context") or {}
        expected_sha = context.get("native_manifest_sha256")
        known_ac47_v4 = (run == "ac47" and
                         expected_sha == "4ba36305071601e11716c86609ce86138ce8b5bee2bf33f5236dcb4c1f6f252e")
        source_sha = AC47_V4_SOURCE_SHA256 if known_ac47_v4 else expected_sha
        rows.append({"run": run, "review_root": str(root),
                     "manifest_path": str(manifest_path),
                     "manifest_sha256": source_sha,
                     "selection_error": (None if selected_from_request else
                                         "canonical Studio source is absent from frozen request"),
                     "parent_context_path": context.get("path"),
                     "parent_context_sha256": context.get("sha256"),
                     "parent_context_manifest_sha256": expected_sha,
                     "certified_ac47_v4_successor": known_ac47_v4})
    selection = {
        "policy": SELECTION_POLICY,
        "parent_model": str(parent_path),
        "parent_postflight_path": str(postflight_path),
        "parent_postflight_sha256": _hash(postflight_path),
        "run_count": len(rows),
        "runs": [row["run"] for row in rows],
        "source_manifest_sha256": {row["run"]: row["manifest_sha256"] for row in rows},
    }
    return rows, selection


def _fit_one(row: dict[str, Any], target: dict[str, Any], target_sha256: str,
             request_sha256: str, postflight_sha256: str) -> dict[str, Any]:
    run = row["run"]
    result: dict[str, Any] = {
        "format": FORMAT, "run": run, "status": "rejected",
        "frozen_request_sha256": request_sha256,
        "parent_postflight_sha256": postflight_sha256,
        "source_manifest_path": row["manifest_path"],
        "source_manifest_sha256": row["manifest_sha256"],
        "parent_context_manifest_sha256": row["parent_context_manifest_sha256"],
        "certified_ac47_v4_successor": row["certified_ac47_v4_successor"],
        "target_sha256": target_sha256,
        "target_id": target["target_id"],
        "image_processing_scope": "within-run JPEG baseline plus fixed common-start fit only",
        "phase_frames_read": 0, "phase_frames_written": 0,
        "created_utc": _utc_now(),
    }
    try:
        if row["selection_error"]:
            raise ValueError(f"{run}: {row['selection_error']}")
        if not row["manifest_sha256"]:
            raise ValueError(f"{run}: parent postflight lacks a native source SHA")
        source_path = Path(row["manifest_path"])
        if _hash(source_path) != row["manifest_sha256"]:
            raise ValueError(f"{run}: frozen source manifest SHA changed")
        manifest = _read(source_path)
        if manifest.get("run") != run or str(manifest.get("dataset", "")).casefold() != "ffac":
            raise ValueError(f"{run}: source manifest identity changed")
        result["historical_photometry_revision"] = _source_photometry_revision(manifest, row)
        prepared = _prepared_baseline_source(manifest, row)
        result["prepared_baseline_source"] = prepared
        if (row.get("parent_context_sha256") != target["facies_context_sha256"]
                or _hash(Path(row["parent_context_path"])) != row["parent_context_sha256"]):
            raise ValueError(f"{run}: parent context does not match frozen common-start geometry")
        registration_profile, frozen_registration = _frozen_registration_profile(manifest, run)
        with _open_ffac(run, registration_profile=registration_profile,
                        disable_registration=registration_profile is None) as source:
            corrections = source.rig.color_corrections or []
            if (len(corrections) != 1 or
                    getattr(corrections[0], "algorithm_revision", None) != "legacy-v2"):
                raise ValueError(f"{run}: within-run ColorChecker revision differs")
            actual_registration = source.correction_contract.get("rig_registration")
            _require_matching_geometry(frozen_registration, actual_registration, run)
            expected_hw = list(manifest["analysis_image_shape_hw"])
            actual_hw = list(source.baseline.shape[:2])
            if actual_hw != expected_hw or actual_hw != target["shape_hw"]:
                raise ValueError(f"{run}: corrected baseline grid differs from source or target")
            content_box = list(manifest["content_box_xywh"])
            canvas_wh = list(manifest["canvas_wh"])
            if content_box[2:] != [actual_hw[1], actual_hw[0]]:
                raise ValueError(f"{run}: historical content box differs from calibrated grid")
            result["geometry"] = {
                "analysis_image_shape_hw": actual_hw,
                "content_box_xywh": content_box,
                "canvas_wh": canvas_wh,
                "registered": actual_registration is not None,
                "geometry_revision_sha256": (actual_registration or {}).get("geometry_revision_sha256"),
            }
            jpeg = Path(source.rig.baseline_path)
            reference = manifest.get("reference") or {}
            expected_jpeg_sha = reference.get("source_sha256") or reference.get("source_raw_sha256")
            if prepared is not None:
                member = prepared["members"][0]
                expected_jpeg_sha = member["sha256"]
                if jpeg.resolve() != Path(member["path"]).resolve():
                    raise ValueError(f"{run}: rig baseline differs from prepared acquisition")
            elif (not expected_jpeg_sha or not reference.get("source_path")
                  or jpeg.resolve() != Path(reference["source_path"]).resolve()):
                raise ValueError(f"{run}: rig baseline differs from frozen reference JPEG")
            result["baseline_pair_inventory"] = _baseline_pair_inventory(jpeg, expected_jpeg_sha)
            if result["baseline_pair_inventory"]["exif_pair_check"]["status"] == "mismatch":
                raise ValueError(f"{run}: baseline ARW/JPEG EXIF mismatch")
            result["within_run_baseline_array_sha256"] = _array_sha256(source.baseline)
            result["within_run_colorchecker"] = _chart_qc(_correction_diagnostics(source.rig))
            fit = _sequential_baseline_fit(source, target, target_sha256)
            fit["chart_extractor_revision"] = SEQUENTIAL_CHART_EXTRACTOR_REVISION
            from darsia.fluidflower.tools.phase_common_baseline_photometry import (
                apply_fixed_run_transform,
            )
            corrected = apply_fixed_run_transform(source.baseline, fit)
            result["common_start_baseline_array_sha256"] = _array_sha256(corrected)
            result["common_start_chart_qa"] = _common_chart_qc(
                corrected, corrections[0], target, fit["usable_chart_patch_mask"])
            result["fit"] = fit
            result["status"] = "accepted"
    except Exception as error:  # Deliberately continue through every canonical run.
        result["rejection"] = {"type": type(error).__name__, "reason": str(error)}
    return result


def _worker(row: dict[str, Any], target_path: Path, target_sha256: str,
            request_sha256: str, postflight_sha256: str, output_root: Path) -> dict[str, Any]:
    target = _read(target_path)
    result = _fit_one(row, target, target_sha256, request_sha256, postflight_sha256)
    receipt_path = output_root / "runs" / f"{row['run']}-{row['manifest_sha256'][:16]}.json"
    result = _write_immutable(receipt_path, result)
    return {"run": row["run"], "source_manifest_sha256": row["manifest_sha256"],
            "status": result["status"], "rejection": result.get("rejection"),
            "receipt_path": str(receipt_path), "receipt_sha256": _hash(receipt_path)}


def preflight(request_path: Path, target_path: Path, target_sha256: str,
              output_root: Path, *, workers: int = 1,
              request_sha256: str | None = None) -> dict[str, Any]:
    if not 1 <= workers <= 24:
        raise ValueError("Workers must be between 1 and 24")
    if not request_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Frozen request and common-start target must exist")
    actual_request_sha = _hash(request_path)
    if request_sha256 is not None and actual_request_sha != request_sha256:
        raise ValueError("Frozen request SHA-256 differs from required value")
    if _hash(target_path) != target_sha256:
        raise ValueError("Common-start target SHA-256 differs from required value")
    target = _read(target_path)
    from darsia.fluidflower.tools.ffac_common_start_target import validate_candidate
    validate_candidate(target, verify_source_bytes=True)
    rows, selection = canonical_revisions(request_path)
    reports: list[dict[str, Any] | None] = [None] * len(rows)
    with ProcessPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        pending = {pool.submit(_worker, row, target_path, target_sha256,
                               actual_request_sha, selection["parent_postflight_sha256"],
                               output_root): index for index, row in enumerate(rows)}
        for future in as_completed(pending):
            index = pending[future]
            try:
                reports[index] = future.result()
            except Exception as error:
                row = rows[index]
                reports[index] = {"run": row["run"],
                                  "source_manifest_sha256": row["manifest_sha256"],
                                  "status": "rejected",
                                  "rejection": {"type": type(error).__name__,
                                                "reason": str(error)},
                                  "receipt_path": None, "receipt_sha256": None}
            print(json.dumps({"completed": sum(item is not None for item in reports),
                              "total": len(rows), "run": reports[index]["run"],
                              "status": reports[index]["status"]}), flush=True)
    complete_reports = [row for row in reports if row is not None]
    result = {"format": FORMAT, "frozen_request_path": str(request_path),
              "frozen_request_sha256": actual_request_sha,
              "target_path": str(target_path), "target_sha256": target_sha256,
              "target_id": target["target_id"], "selection": selection,
              "workers": workers, "attempted": len(complete_reports),
              "accepted": sum(row["status"] == "accepted" for row in complete_reports),
              "rejected": sum(row["status"] == "rejected" for row in complete_reports),
              "all_run_receipts_written": all(row["receipt_path"] for row in complete_reports),
              "runs": complete_reports, "created_utc": _utc_now()}
    return _write_immutable(output_root / "preflight_receipt.json", result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-request", type=Path, required=True)
    parser.add_argument("--request-sha256")
    parser.add_argument("--target-profile", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    result = preflight(args.frozen_request, args.target_profile, args.target_sha256,
                       args.output_root, workers=args.workers,
                       request_sha256=args.request_sha256)
    print(json.dumps({key: result[key] for key in ("attempted", "accepted", "rejected",
                                                  "all_run_receipts_written")}), flush=True)


if __name__ == "__main__":
    main()
