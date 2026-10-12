"""Read-only native categorical QA against the Sep30 photographic candidate.

Reports the requested isolated-hole, locality and phase-area checks. It never
smooths maps, updates gates, refits a model or enables mass analysis.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file
from .qa_ffac_track1_display_mask_rerun import (
    _bound, _old_manifest, _phase_folder, _phase_labels, _read,
)
from .qa_ffac_track1_isolated_invalid_rerun import write_report

CONTRACT = "ffac-track1-isolated-invalid-native-phase-map-qa-v1"
FEATURE_GUARD_RADIUS_PX = 8
EXTRA_LOCAL_RADIUS_PX = 2
SMALL_UNRESOLVED_AREA_PX = 400
AREA_CHANGE_LIMIT = .003
NONLOCAL_CHANGE_LIMIT = .001


def _mask(row, kind: str, shape) -> np.ndarray:
    path = _bound(row.get(f"v5_{kind}_mask_path"), row.get(f"v5_{kind}_mask_sha256"), kind)
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.dtype(bool) or values.shape != tuple(shape):
        raise ValueError(f"Native {kind} mask has another dtype or grid")
    return values


def _transport_affected(mask: np.ndarray, h: np.ndarray, shape) -> np.ndarray:
    matrix = np.asarray(h, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Phase comparison needs the exact nonsingular per-frame H")
    transported = cv2.warpPerspective(mask.astype(np.float32), matrix,
        (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    # All nonzero bilinear donor weights can alter source support.
    return transported > np.float32(1e-6)


def allowed_local_region(current_filled: np.ndarray, baseline_filled: np.ndarray,
        h_baseline_to_current: np.ndarray, *, previous_filled: np.ndarray | None = None,
        h_baseline_to_previous: np.ndarray | None = None):
    shape = current_filled.shape
    affected = current_filled.copy()
    affected |= _transport_affected(baseline_filled, h_baseline_to_current, shape)
    if previous_filled is not None:
        if h_baseline_to_previous is None:
            raise ValueError("Previous filled mask lacks its exact baseline homography")
        h = np.asarray(h_baseline_to_current, np.float64) @ np.linalg.inv(
            np.asarray(h_baseline_to_previous, np.float64))
        affected |= _transport_affected(previous_filled, h, shape)
    former_squares = cv2.dilate(affected.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    allowed = cv2.dilate(former_squares.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return allowed, affected, former_squares


def _relative_change(old: int, new: int):
    if old:
        return (new-old) / old, abs(new-old) / old > AREA_CHANGE_LIMIT
    return (0.0, False) if not new else (None, True)


def compare_categorical_maps(old: np.ndarray, new: np.ndarray, safe_filled: np.ndarray,
                             allowed_local: np.ndarray) -> dict:
    """Assess exact categorical arrays, never their composited photographs."""
    if (old.dtype != np.uint8 or new.dtype != np.uint8 or old.shape != new.shape
            or old.shape != safe_filled.shape or old.shape != allowed_local.shape
            or safe_filled.dtype != np.dtype(bool) or allowed_local.dtype != np.dtype(bool)
            or np.any(old > 4) or np.any(new > 4)):
        raise ValueError("Categorical QA needs native uint8 labels 0..4 and boolean masks")
    interior = cv2.erode(safe_filled.astype(np.uint8), np.ones((17, 17), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    guard = safe_filled & ~interior
    unresolved = (new == 4) & interior
    count, _, stats, _ = cv2.connectedComponentsWithStats(unresolved.astype(np.uint8), connectivity=8)
    small = [stats[index] for index in range(1, count)
             if int(stats[index, cv2.CC_STAT_AREA]) <= SMALL_UNRESOLVED_AREA_PX]
    sizes = sorted(int(row[cv2.CC_STAT_AREA]) for row in small)
    changed = (old != new) & interior
    nonlocal_changed = changed & ~allowed_local
    plume = (((old >= 1) & (old <= 3)) | ((new >= 1) & (new <= 3))) & safe_filled
    plume_count, nonlocal_count = int(plume.sum()), int(nonlocal_changed.sum())
    nonlocal_fraction = nonlocal_count / plume_count if plume_count else (0.0 if not nonlocal_count else None)
    nonlocal_flag = nonlocal_count > NONLOCAL_CHANGE_LIMIT * plume_count
    phases = {}
    for value, name in ((1, "aqueous"), (2, "gas"), (3, "mixed")):
        old_count = int(np.count_nonzero((old == value) & safe_filled))
        new_count = int(np.count_nonzero((new == value) & safe_filled))
        relative, flag = _relative_change(old_count, new_count)
        phases[name] = dict(old_pixels=old_count, new_pixels=new_count,
            signed_change_pixels=new_count-old_count, relative_change=relative,
            fraction_denominator="old_phase_area; null if old=0 and new>0",
            change_above_0p3_percent=flag)
    area_flags = [name for name, row in phases.items() if row["change_above_0p3_percent"]]
    return dict(native_shape_hw=list(old.shape), photo_sand_pixels=int(safe_filled.sum()),
        excluded_outer_8px_photo_domain_guard_pixels=int(guard.sum()),
        unresolved_pixels_inside_sand=int(np.count_nonzero((new == 4) & safe_filled)),
        unresolved_pixels_outside_outer_8px_guard=int(unresolved.sum()),
        unresolved_components_outside_outer_8px_guard=count-1,
        small_unresolved_component_area_limit_px=SMALL_UNRESOLVED_AREA_PX,
        small_unresolved_components_outside_outer_8px_guard=len(sizes),
        small_unresolved_component_areas_px=sizes,
        small_unresolved_pixel_count=sum(sizes),
        total_changed_categorical_pixels_inside_guarded_sand=int(changed.sum()),
        changed_categorical_pixels_outside_old_squares_plus_2px=nonlocal_count,
        changed_predicted_phase_pixels_outside_old_squares_plus_2px=int(np.count_nonzero(
            nonlocal_changed & plume)),
        old_or_new_plume_pixels=plume_count,
        nonlocal_categorical_change_fraction_of_plume=nonlocal_fraction,
        nonlocal_changes_above_0p1_percent_of_plume=nonlocal_flag,
        phases=phases, phase_area_change_flags=area_flags,
        passed=not sizes and not nonlocal_flag and not area_flags)


def _frozen_models(old_candidate: Path, new_candidate: Path):
    files = ("models/generic/co2_presence_signal.json", "models/generic/gas_within_co2_signal.json",
             "regional_model.json", "training_request.json")
    rows = {}
    for name in files:
        before, after = sha256_file(old_candidate / name), sha256_file(new_candidate / name)
        if before != after:
            raise ValueError(f"Comparison models do not preserve their exact frozen fit: {name}")
        rows[name] = before
    return dict(old_model_identity=list(_model_identity(old_candidate)),
        new_model_identity=list(_model_identity(new_candidate)), unchanged_trained_artifacts=rows)


def _candidate_manifest(candidate: Path, run: str):
    path = _old_manifest(candidate, run)
    manifest, digest = _read(path), sha256_file(path)
    job = _read(candidate / "review_source" / run / "original_job.json")
    pinned = job.get("v5_raw_photometry", {}).get("materialized_manifest_sha256")
    if pinned != digest:
        raise ValueError("Saved candidate does not SHA-bind this native input manifest")
    return path, manifest, digest


def _check_saved_package(candidate: Path, run: str, model_identity: list, manifest_sha: str):
    root = candidate / "validation" / "run_packages" / run
    binding_path = root / "validation_binding.json"
    binding = _read(binding_path)
    if ([binding.get("model_id"), binding.get("model_sha256")] != model_identity
            or binding.get("run_count") != 1):
        raise ValueError("Saved run package identifies another model or is incomplete")
    postflight_path = root / "postflight_manifest.json"
    postflight = _read(postflight_path)
    rows = [row for row in postflight.get("jobs") or [] if row.get("run") == run]
    if (len(rows) != 1 or rows[0].get("optical_model_sha256") != model_identity[1]
            or rows[0].get("frame_geometry_manifest_sha256") != manifest_sha):
        raise ValueError("Saved run maps identify another optical fit or native source")
    return dict(validation_binding_path=str(binding_path), validation_binding_sha256=sha256_file(binding_path),
        postflight_manifest_path=str(postflight_path), postflight_manifest_sha256=sha256_file(postflight_path))


def _geometry(manifest: dict, index: int):
    row = manifest["v5_track1_photo_geometry"]["frames"][index]
    receipt = _read(_bound(row["receipt_path"], row["receipt_sha256"], "fixed per-frame H"))
    if receipt.get("frame_index") != index:
        raise ValueError("Saved frame homography identifies another timeline position")
    return np.asarray(receipt["H_v5_B_to_frame_native"], np.float64)


def inspect_run(old_candidate: str | Path, new_candidate: str | Path, run: str, *,
                output_path: Path | None = None, frozen_models: dict | None = None) -> dict:
    old_root, new_root = Path(old_candidate), Path(new_candidate)
    run = run.casefold()
    fits = frozen_models or _frozen_models(old_root, new_root)
    old_path, old, old_sha = _candidate_manifest(old_root, run)
    new_path, new, new_sha = _candidate_manifest(new_root, run)
    for key in ("run", "frame_count", "analysis_image_shape_hw", "canvas_wh", "content_box_xywh",
                "v5_track1_photo_geometry"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Old/new categorical comparison changes fixed source/H geometry: {key}")
    old_processor = old.get("correction_contract", {}).get("processor", {})
    new_processor = new.get("correction_contract", {}).get("processor", {})
    if (old_processor.get("display_mode") != "photographic_v1"
            or old_processor.get("isolated_invalid_policy", "keep") != "keep"
            or new_processor.get("display_mode") != "natural_modes_v1"
            or new_processor.get("isolated_invalid_policy") != "fill_isolated_invalid_v1"):
        raise ValueError("Compare the unrepaired Sep30 photographic candidate to the isolated-fill v2")
    packages = dict(old=_check_saved_package(old_root, run, fits["old_model_identity"], old_sha),
        new=_check_saved_package(new_root, run, fits["new_model_identity"], new_sha))
    old_folder, new_folder = _phase_folder(old_root, run), _phase_folder(new_root, run)
    shape = tuple(new["analysis_image_shape_hw"])
    baseline = _mask(new["reference"], "filled_invalid", shape)
    rows, previous, previous_h = [], None, None
    for index, (a, b) in enumerate(zip(old["frame_sources"], new["frame_sources"])):
        for key in ("frame_index", "elapsed_h", "source_sha256", "source_raw_sha256", "v5_raw_sha256"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Phase comparison binds another acquisition/time: {index}/{key}")
        if b.get("frame_index") != index:
            raise ValueError("Categorical comparison timeline is incomplete")
        current = _mask(b, "filled_invalid", shape)
        sand = _mask(b, "photo_sand", shape)
        h = _geometry(new, index)
        allowed, affected, squares = allowed_local_region(current, baseline, h,
            previous_filled=previous, h_baseline_to_previous=previous_h)
        old_png, new_png = old_folder / f"frame_{index:03d}.png", new_folder / f"frame_{index:03d}.png"
        old_labels, new_labels = _phase_labels(old_png, old), _phase_labels(new_png, new)
        statistics = compare_categorical_maps(old_labels, new_labels, sand, allowed)
        rows.append(dict(run=run, frame_index=index, elapsed_h=b.get("elapsed_h"),
            old_phase_map_path=str(old_png), old_phase_map_sha256=sha256_file(old_png),
            new_phase_map_path=str(new_png), new_phase_map_sha256=sha256_file(new_png),
            current_filled_pixel_count=int(current.sum()), baseline_filled_pixel_count=int(baseline.sum()),
            previous_filled_pixel_count=(0 if previous is None else int(previous.sum())),
            transported_temporal_filled_pixel_count=int(affected.sum()),
            former_isolated_guard_square_pixels=int(squares.sum()),
            allowed_old_square_plus_2px_pixels=int(allowed.sum()), **statistics))
        previous, previous_h = current, h
    if len(rows) != new.get("frame_count"):
        raise ValueError("Incomplete old/new categorical phase sequence")
    report = dict(contract=CONTRACT, old_candidate=str(old_root), new_candidate=str(new_root), run=run,
        old_native_manifest_path=str(old_path), old_native_manifest_sha256=old_sha,
        new_native_manifest_path=str(new_path), new_native_manifest_sha256=new_sha,
        frozen_models=fits, saved_packages=packages, frame_count=len(rows),
        guard_radius_px=FEATURE_GUARD_RADIUS_PX, old_square_extra_dilation_radius_px=EXTRA_LOCAL_RADIUS_PX,
        old_square_affected_inputs="current plus exact-H-transported previous and pre-injection filled masks",
        nonlocal_change_limit_of_union_plume=NONLOCAL_CHANGE_LIMIT, phase_relative_area_change_limit=AREA_CHANGE_LIMIT,
        photographic_display_decoded_as_labels=False, inference_or_refitting_performed=False,
        scientific_mass_eligible=False, per_frame=rows,
        failed_frames=[row["frame_index"] for row in rows if not row["passed"]],
        passed=all(row["passed"] for row in rows))
    if output_path is not None:
        write_report(Path(output_path), report)
    return report


def inspect_maps(old_candidate: str | Path, new_candidate: str | Path, output_dir: str | Path,
                 *, runs: list[str] | None = None, workers: int = 2) -> dict:
    old_root, new_root, output = Path(old_candidate), Path(new_candidate), Path(output_dir)
    if not 1 <= workers <= 4:
        raise ValueError("Phase QA workers must be between 1 and 4")
    fits = _frozen_models(old_root, new_root)
    selected = sorted(set(run.casefold() for run in runs)) if runs else sorted(
        row["run"] for row in _read(new_root / "review_source" / "postflight_manifest.json")["jobs"])
    if not selected or (runs is None and len(set(selected)) != 43):
        raise ValueError("Full phase QA requires 43 source-bound runs; select ready runs explicitly")
    def one(run):
        return inspect_run(old_root, new_root, run, output_path=output / f"{run}_phase_qa.json",
                           frozen_models=fits)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="isolated-phase-qa") as pool:
        reports = list(pool.map(one, selected))
    frames = sum(report["frame_count"] for report in reports)
    if runs is None and frames != 864:
        raise ValueError("Full phase QA requires 43/864 exact native maps")
    summary = dict(contract=CONTRACT, old_candidate=str(old_root), new_candidate=str(new_root),
        complete_43_864_series=(runs is None), run_count=len(reports), frame_count=frames,
        failed_runs=[report["run"] for report in reports if not report["passed"]],
        passed=all(report["passed"] for report in reports), scientific_mass_eligible=False,
        per_run=[dict(run=report["run"], frame_count=report["frame_count"], passed=report["passed"],
            report_path=str(output / f"{report['run']}_phase_qa.json"),
            report_sha256=sha256_file(output / f"{report['run']}_phase_qa.json")) for report in reports])
    write_report(output / ("qa_phase_maps.json" if runs is None else "qa_ready_phase_maps.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-candidate", required=True, type=Path)
    parser.add_argument("--new-candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run", action="append")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = inspect_maps(args.old_candidate, args.new_candidate, args.output_dir,
                          runs=args.run, workers=args.workers)
    print(json.dumps(result))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
