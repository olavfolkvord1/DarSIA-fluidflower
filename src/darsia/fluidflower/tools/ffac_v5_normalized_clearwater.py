"""Isolated, source-bound pilot for V5-normalized FF_AC RAW clearwater.

This deliberately does not fit a concentration model or modify a phase model.
It renders one predeclared wash ARW onto the selected run's native B grid and
applies the *same frozen V4-R5 photometry* used for Track1 phase input. A
failed geometry, photometry, or support check leaves no eligible receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.clearwater_phase_reference import select_clear_reference_paths
from darsia.fluidflower.tools.ffac_raw_native_ea import registered_edge_aware_raw_linear
from darsia.fluidflower.tools.ffac_track1_v4r5_source_processor import (
    Track1V4R5SourceProcessor, _chart_xy_baseline_native, _h,
)
from darsia.fluidflower.tools.ffac_v4r5_native_pilot import correct_frame_native_v4r5
from darsia.fluidflower.tools.ffac_v5_chart_registration import register_frame_chart_xy8
from darsia.fluidflower.tools.phase_studio_joint_inputs import (
    V5_CLEARWATER_FORMAT, V5_CLEARWATER_PROCESSING_FORMAT, V5_RGB_ENCODING,
)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _save(path: Path, value: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temp.replace(path)


def _photometry_sha(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest["v5_photometry"],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_one(*, run: str, native_manifest_path: Path,
                photo_aggregate_path: Path, calibration_manifest_path: Path,
                raw_root: Path, output: Path) -> dict:
    """Render the middle of the configured five clearwater JPEG/ARW pairs.

    The middle-image choice is fixed before looking at any phase or mass data;
    all five configured names are retained as provenance. The caller must
    evaluate the endpoint independently before it is used in a mass fit.
    """
    run = str(run).casefold()
    manifest_path = Path(native_manifest_path).resolve()
    native = _read(manifest_path)
    if (manifest_path.parent.name.casefold() != run
            or (native.get("v5_photometry") or {}).get("format") !=
            "ffac-v5-native-source-materialization-v1"
            or native["v5_photometry"].get("unit_denominator") != 65535
            or native["v5_photometry"].get("display_jpeg_is_training_input") is not False):
        raise ValueError("Selected run has no exact Track1 V5 scientific source")
    processor_record = native["v5_photometry"]["processor"]
    photo_path = Path(photo_aggregate_path).resolve()
    calibration_path = Path(calibration_manifest_path).resolve()
    if (sha256_file(photo_path) != processor_record["photo_source_aggregate_sha256"]
            or sha256_file(calibration_path) !=
            processor_record["calibration_manifest_sha256"]):
        raise ValueError("V5 photo/calibration fit differs from selected source")
    source_sha = native["v5_photometry"]["source_manifest_sha256"]
    aggregate = _read(photo_path)
    matching = [row for row in aggregate["rows"] if
                row["run"] == run and row["source_manifest_sha256"] == source_sha]
    if len(matching) != 1:
        raise ValueError("Selected V5 run has no unique photo-source row")
    row = matching[0]
    if sha256_file(Path(row["source_manifest_path"])) != source_sha:
        raise ValueError("Frozen V5 source manifest changed")
    processor = Track1V4R5SourceProcessor(row,
        calibration_manifest_path=calibration_path,
        photo_aggregate_path=photo_path)
    if (processor.identity()["processor_module_sha256"] !=
            processor_record["processor_module_sha256"]):
        raise ValueError("Running V4-R5 processor differs from selected source")
    b = processor.cohort.frame(0)
    b_review = Path(b["v5_B_native_jpeg_path"])
    b_camera = Path(b["v5_B_camera_jpeg_path"])
    b_cert_path = Path(b["v5_B_registration_certificate_path"])
    for path, expected in ((b_review, b["v5_B_native_jpeg_sha256"]),
                           (b_camera, b["v5_B_camera_jpeg_sha256"]),
                           (b_cert_path, b["v5_B_registration_certificate_sha256"])):
        if sha256_file(path) != expected:
            raise ValueError("Frozen B geometry or ColorChecker source changed")
    configured = select_clear_reference_paths(raw_root, run)
    if len(configured) != 5:
        raise ValueError("Expected five configured clearwater acquisitions")
    selected = configured[2]
    arw = selected.with_suffix(".ARW")
    if not arw.is_file() or selected.stem.casefold() != arw.stem.casefold():
        raise ValueError("Configured clearwater JPEG lacks a same-capture ARW")
    sources = dict(arw_sha256=sha256_file(arw),
        camera_jpeg_sha256=sha256_file(selected),
        review_native_jpeg_sha256=sha256_file(b_review))
    linear, renderer = registered_edge_aware_raw_linear(
        arw, selected, b_review, expected_source_hashes=sources,
        content_box=native["content_box_xywh"])
    # The actual baseline canvas is the destination. The renderer's direct
    # sand fit supplies its camera warp; no historical wash JPEG is reused.
    geometry = renderer["raw_review_direct_residual"]
    checked = {}
    for region in ("center", "right_lower_sand"):
        entry = geometry.get(region) or {}
        if int(entry.get("n") or 0) < 100 or float(entry.get("p95_px") or 1e9) >= 2.0:
            raise ValueError(f"RAW wash/native B sand registration failed in {region}: {entry}")
        checked[region] = dict(n=int(entry["n"]), p95_px=float(entry["p95_px"]),
            pilot_limit_px=2.0, original_phase_limit_px=1.0,
            scope="wash-to-B-source-only-pilot; not Track1 native-geometry certification")
    h_ref_frame = _h(b["H_ac28_to_frame_native"])
    h_b_frame = _h(b["H_v5_B_to_frame_native"])
    h_ref_b = _h(np.linalg.inv(h_b_frame) @ h_ref_frame)
    bcert = _read(b_cert_path)
    xy, usable, chart = register_frame_chart_xy8(
        baseline_camera_jpeg=b_camera,
        baseline_camera_sha256=b["v5_B_camera_jpeg_sha256"],
        frame_camera_jpeg=selected,
        frame_camera_sha256=sources["camera_jpeg_sha256"],
        baseline_chart_xy8=_chart_xy_baseline_native(
            processor.fit, processor.grid, h_ref_b),
        baseline_chart_log_rgb=processor.fit.CB,
        baseline_review_to_camera_h=np.asarray(
            bcert["review_to_camera_h"], np.float64),
        frame_review_to_camera_h=np.asarray(
            renderer["review_to_camera_h"], np.float64),
        frame_native_linear_rgb=linear)
    baseline_valid_path = Path(native["reference"]["v5_log_linear_valid_mask_path"])
    if sha256_file(baseline_valid_path) != native["reference"][
            "v5_log_linear_valid_mask_sha256"]:
        raise ValueError("Selected V5 baseline validity support changed")
    baseline_valid = np.load(baseline_valid_path, allow_pickle=False)
    if baseline_valid.shape != (2979, 5472) or baseline_valid.dtype != bool:
        raise ValueError("Selected V5 baseline support is not native bool")
    corrected, correction = correct_frame_native_v4r5(
        linear, processor.fit, processor.reference, processor.grid,
        h_ref_b, baseline_valid, exact_baseline_source=False,
        frame_chart_xy8=xy, frame_chart_usable=usable,
        old_field_ref8=processor.old_field)
    valid = np.isfinite(corrected).all(axis=0) & baseline_valid
    missing = baseline_valid & ~valid
    missing_count = int(np.count_nonzero(missing))
    # Keep every missing pixel explicit. Neither correction nor output writes
    # a spatially interpolated replacement into a scientific RAW signal.
    baseline_log_path = Path(native["reference"]["v5_log_linear_rgb_path"])
    if sha256_file(baseline_log_path) != native["reference"]["v5_log_linear_rgb_sha256"]:
        raise ValueError("Selected V5 baseline scientific RGB changed")
    baseline_log = np.load(baseline_log_path, allow_pickle=False, mmap_mode="r")
    probe_valid = valid[::8, ::8]
    delta = corrected[:, ::8, ::8] - baseline_log[:, ::8, ::8]
    chromatic = .5 * (delta[0] + delta[1]) - delta[2]
    probe = chromatic[probe_valid & np.isfinite(chromatic)]
    if probe.size == 0:
        raise ValueError("Normalized wash has no baseline-supported optical sample")
    positive_fraction = float(np.mean(probe > .15))
    optical_probe = dict(sample_stride_px=8, sample_count=int(probe.size),
        chromatic_log_change_p10_p50_p90=[float(x) for x in
            np.quantile(probe, [.1, .5, .9])],
        positive_span_gt_0p15_fraction=positive_fraction,
        notes="A source-only diagnostic; positive clearwater span is not CO2 concentration")
    if missing_count:
        ys, xs = np.nonzero(missing)
        missing_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        missing_bbox = None
    rgb = np.zeros((2979, 5472, 3), np.float32)
    with np.errstate(over="raise", invalid="raise"):
        for channel in range(3):
            rgb[..., channel][valid] = (np.exp(corrected[channel][valid]) /
                                       np.float32(65535.0))
    if (not np.isfinite(rgb[valid]).all() or np.any(rgb[valid] < 0)
            or np.any(rgb[valid] > 1)):
        raise ValueError("Normalized wash is not finite unit camera-linear RGB")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rgb_path = output / "clearwater.camera_linear_rgb.npy"
    valid_path = output / "clearwater.valid_mask.npy"
    correction_path = output / "clearwater.processing.json"
    receipt_path = output / "clearwater.receipt.json"
    for path in (rgb_path, valid_path, correction_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Isolated clearwater output already exists: {path}")
    _save(rgb_path, rgb)
    _save(valid_path, valid)
    rgb_sha, valid_sha = sha256_file(rgb_path), sha256_file(valid_path)
    photometry_sha = _photometry_sha(native)
    native_sha = sha256_file(manifest_path)
    processed = dict(format=V5_CLEARWATER_PROCESSING_FORMAT,
        dataset="ffac", run=run, native_manifest_path=str(manifest_path),
        native_manifest_sha256=native_sha, v5_photometry_sha256=photometry_sha,
        source_manifest_path=str(row["source_manifest_path"]),
        source_manifest_sha256=source_sha,
        raw_source_path=str(arw), raw_source_sha256=sources["arw_sha256"],
        camera_jpeg_path=str(selected),
        camera_jpeg_sha256=sources["camera_jpeg_sha256"],
        baseline_review_jpeg_path=str(b_review),
        baseline_review_jpeg_sha256=sources["review_native_jpeg_sha256"],
        selected_rule="middle of five configured run-specific wash acquisitions, predetermined",
        configured_camera_jpegs=[str(path) for path in configured],
        calibration_manifest_path=str(calibration_path),
        calibration_manifest_sha256=sha256_file(calibration_path),
        photo_aggregate_path=str(photo_path),
        photo_aggregate_sha256=sha256_file(photo_path),
        renderer_certificate=renderer,
        registration_gate=checked, chart_registration=chart,
        v4r5_correction=correction,
        encoding=V5_RGB_ENCODING, shape_hw=[2979, 5472],
        valid_pixels=int(valid.sum()), baseline_valid_pixels=int(baseline_valid.sum()),
        missing_baseline_pixels=missing_count,
        missing_baseline_fraction=missing_count / int(baseline_valid.sum()),
        missing_baseline_bbox_xyxy=missing_bbox,
        optical_probe=optical_probe,
        rgb_sha256=rgb_sha, valid_mask_sha256=valid_sha,
        baseline_scientific_support_complete=missing_count == 0,
        physical_domain_and_facies_not_verified=True,
        scientific_mass_eligible=False, fit_performed=False)
    _write(correction_path, processed)
    receipt = dict(format=V5_CLEARWATER_FORMAT,
        dataset="ffac", run=run, native_manifest_sha256=native_sha,
        v5_photometry_sha256=photometry_sha, encoding=V5_RGB_ENCODING,
        shape_hw=[2979, 5472], raw_source_sha256=sources["arw_sha256"],
        correction_receipt_path=str(correction_path),
        correction_receipt_sha256=sha256_file(correction_path),
        rgb_path=str(rgb_path), rgb_sha256=rgb_sha,
        valid_mask_path=str(valid_path), valid_mask_sha256=valid_sha)
    _write(receipt_path, receipt)
    return dict(path=str(receipt_path), sha256=sha256_file(receipt_path),
        run=run, valid_pixels=int(valid.sum()),
        baseline_valid_pixels=int(baseline_valid.sum()),
        missing_baseline_pixels=missing_count,
        optical_probe=optical_probe,
        registration_gate=checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "native-manifest", "photo-aggregate", "calibration-manifest",
                 "raw-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = prepare_one(run=args.run,
        native_manifest_path=Path(args.native_manifest),
        photo_aggregate_path=Path(args.photo_aggregate),
        calibration_manifest_path=Path(args.calibration_manifest),
        raw_root=Path(args.raw_root), output=Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
