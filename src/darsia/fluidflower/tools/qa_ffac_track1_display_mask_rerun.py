"""Read-only native QA for the frozen-model FF_AC display/mask rerun.

Numerical sidecars and categorical saved phase maps are assessed separately.
No photographic JPEG is decoded as a model feature or as a phase label.  Final
report files are created exclusively and never overwrite an earlier report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.ffac_v5_frame_context import (
    _source_photo_grid, apply_photo_safe_core_enclosed_holes,
    photo_safe_core_policy_metadata, project_source_photo_safe_core,
)


EXPECTED_RUNS = 43
EXPECTED_FRAMES = 864
SIDECAR_DELTA_LIMIT = 1e-6
PHASE_CHANGE_PLUME_FRACTION_LIMIT = 0.001
_COLORS = {
    "aqueous": (49, 217, 160, 255), "gas": (242, 76, 157, 255),
    "mixed": (255, 191, 71, 255), "unresolved": (30, 210, 255, 72),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _bound(path: Any, digest: Any, label: str) -> Path:
    candidate = Path(str(path))
    if (not isinstance(digest, str) or len(digest) != 64
            or not candidate.is_file() or sha256_file(candidate) != digest):
        raise ValueError(f"Rerun QA {label} absent or SHA changed: {candidate}")
    return candidate


def _new_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def _old_manifest(candidate: Path, run: str) -> Path:
    job_path = candidate / "review_source" / run / "original_job.json"
    if not job_path.is_file():
        raise ValueError(f"Rerun QA old candidate lacks selected source job: {run}")
    job = _read(job_path)
    root = Path(job.get("analysis_source_review_root") or job.get("review_output_dir") or "")
    path = root / "native_frames" / run / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Rerun QA selected old native manifest absent: {path}")
    return path


def _source_pairs(old_candidate: Path, new_batch_root: Path
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_path = new_batch_root / "materialization_receipt.json"
    batch = _read(receipt_path)
    mapping = batch.get("canonical_mapping") or []
    if not isinstance(mapping, list) or len(mapping) != EXPECTED_RUNS:
        raise ValueError("Rerun QA requires exactly 43 sealed current source revisions")
    pairs = []
    seen = set()
    for row in sorted(mapping, key=lambda item: item["run"]):
        run = str(row["run"]).casefold()
        if run in seen:
            raise ValueError(f"Rerun QA duplicate current source: {run}")
        seen.add(run)
        new_path = _bound(row.get("materialized_manifest_path"),
            row.get("materialized_manifest_sha256"), "new sealed native manifest")
        if row.get("receipt_path"):
            _bound(row["receipt_path"], row.get("receipt_sha256"), "source revision seal")
        old_path = _old_manifest(old_candidate, run)
        old, new = _read(old_path), _read(new_path)
        old_marker = old.get("v5_track1_photo_geometry") or {}
        new_marker = new.get("v5_track1_photo_geometry") or {}
        expected = photo_safe_core_policy_metadata("fill_enclosed_v1")
        if (old.get("run") != run or new.get("run") != run
                or old_marker.get("enclosed_hole_policy", "keep") != "keep"
                or not isinstance(old_marker.get("original_source_manifest_sha256"), str)
                or len(old_marker["original_source_manifest_sha256"]) != 64
                or any(new_marker.get(key) != value for key, value in expected.items())
                or old_marker.get("original_source_manifest_sha256") !=
                   new_marker.get("original_source_manifest_sha256")
                or old.get("analysis_image_shape_hw") != new.get("analysis_image_shape_hw")
                or old.get("canvas_wh") != new.get("canvas_wh")
                or old.get("content_box_xywh") != new.get("content_box_xywh")):
            raise ValueError(f"Rerun QA source, mask policy or native geometry changed: {run}")
        old_frames, new_frames = old.get("frame_sources") or [], new.get("frame_sources") or []
        if (len(old_frames) != len(new_frames)
                or len(new_frames) != new.get("frame_count")
                or len(old_marker.get("frames") or []) != len(old_frames)
                or len(new_marker.get("frames") or []) != len(new_frames)):
            raise ValueError(f"Rerun QA source timeline count changed: {run}")
        for index, (old_frame, new_frame) in enumerate(zip(old_frames, new_frames)):
            if (old_frame.get("frame_index") != index or new_frame.get("frame_index") != index
                    or old_frame.get("elapsed_h") != new_frame.get("elapsed_h")
                    or old_frame.get("v5_raw_sha256") != new_frame.get("v5_raw_sha256")
                    or old_marker["frames"][index].get("receipt_sha256") !=
                       new_marker["frames"][index].get("receipt_sha256")):
                raise ValueError(f"Rerun QA frame acquisition or H evidence changed: {run}/{index}")
            pairs.append(dict(run=run, frame_index=index,
                old_manifest=old, new_manifest=new,
                old_manifest_path=str(old_path), new_manifest_path=str(new_path),
                old_frame=old_frame, new_frame=new_frame))
    if len(pairs) != EXPECTED_FRAMES:
        raise ValueError("Rerun QA current source count differs from 43/864")
    return pairs, dict(path=str(receipt_path), sha256=sha256_file(receipt_path))


def _filled_holes(pair: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = pair["new_manifest"]
    row = manifest["v5_track1_photo_geometry"]["frames"][pair["frame_index"]]
    source = _read(_bound(row["receipt_path"], row["receipt_sha256"], "source-photo H receipt"))
    sand = _source_photo_grid(source["photo_sand_reference_mask_path"],
                              source["photo_sand_reference_mask_sha256"])
    keep = project_source_photo_safe_core(sand, source["H_ac28_to_frame_native"],
        shape_hw=tuple(manifest["analysis_image_shape_hw"]))
    filled, metadata, rules = apply_photo_safe_core_enclosed_holes(
        keep, enclosed_hole_policy="fill_enclosed_v1", return_rule_masks=True)
    # Keep the full-resolution masks private to frame QA; JSON reports contain
    # only counts. The guard is the fixed 8 px feature support at the filled
    # domain's outer edge, rather than every edge of the old internal holes.
    interior = cv2.erode(filled.astype(np.uint8), np.ones((17, 17), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    metadata.update(_enclosed_rule1_mask=rules["enclosed_rule1"],
        _vertical_gap_rule2_mask=rules["vertical_gap_rule2"],
        _outer_feature_guard_mask=filled & ~interior,
        old_photo_support_retained=bool(np.all(filled[keep])))
    return filled & ~keep, metadata


def _load_sidecar(frame: dict[str, Any], *, verify_payload_hashes: bool
                  ) -> tuple[np.ndarray, np.ndarray]:
    array_path = Path(frame["v5_log_linear_rgb_path"])
    mask_path = Path(frame["v5_log_linear_valid_mask_path"])
    if verify_payload_hashes:
        _bound(array_path, frame["v5_log_linear_rgb_sha256"], "float32 sidecar")
        _bound(mask_path, frame["v5_log_linear_valid_mask_sha256"], "sidecar validity")
    array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if (array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 3
            or valid.dtype != np.bool_ or valid.shape != array.shape[1:]):
        raise ValueError("Rerun QA requires native float32 CHW RAW and exact HW bool mask")
    # Materialization stores camera channels first.  This view preserves the
    # mmap and native pixels while the bounded comparison loops use HWC rows.
    return np.moveaxis(array, 0, -1), valid


def _compare_source_frame(pair: dict[str, Any], *, verify_payload_hashes: bool
                          ) -> dict[str, Any]:
    old, old_valid = _load_sidecar(pair["old_frame"], verify_payload_hashes=verify_payload_hashes)
    new, new_valid = _load_sidecar(pair["new_frame"], verify_payload_hashes=verify_payload_hashes)
    if old.shape != new.shape or old.shape[:2] != tuple(pair["new_manifest"]["analysis_image_shape_hw"]):
        raise ValueError("Rerun QA numerical sidecar dimensions changed")
    holes, support = _filled_holes(pair)
    rule1 = support.get("_enclosed_rule1_mask", holes)
    rule2 = support.get("_vertical_gap_rule2_mask", np.zeros_like(holes))
    maximum = 0.0
    finite_mismatch_old = finite_mismatch_new = lost = added = outside = 0
    old_count = new_count = invalid_comparison = 0
    rule1_added = rule2_added = 0
    # Bound scratch arrays even at full 16.3-million-pixel native resolution.
    for start in range(0, old.shape[0], 128):
        end = min(start + 128, old.shape[0])
        a, b = np.asarray(old[start:end]), np.asarray(new[start:end])
        va, vb = np.asarray(old_valid[start:end]), np.asarray(new_valid[start:end])
        fa, fb = np.isfinite(a).all(axis=-1), np.isfinite(b).all(axis=-1)
        old_count += int(np.count_nonzero(va))
        new_count += int(np.count_nonzero(vb))
        finite_mismatch_old += int(np.count_nonzero(fa != va))
        finite_mismatch_new += int(np.count_nonzero(fb != vb))
        lost += int(np.count_nonzero(va & ~vb))
        invalid_comparison += int(np.count_nonzero(va & ~fb))
        comparable = va & fb
        if comparable.any():
            delta = np.abs(a[comparable].astype(np.float64) - b[comparable].astype(np.float64))
            maximum = max(maximum, float(delta.max()))
        changed = vb & ~va
        added += int(np.count_nonzero(changed))
        rule1_added += int(np.count_nonzero(changed & rule1[start:end]))
        rule2_added += int(np.count_nonzero(changed & rule2[start:end]))
        outside += int(np.count_nonzero(changed & ~holes[start:end]))
    return dict(run=pair["run"], frame_index=pair["frame_index"],
        elapsed_h=pair["new_frame"].get("elapsed_h"), native_shape_hw=list(old.shape[:2]),
        old_valid_pixels=old_count, new_valid_pixels=new_count,
        added_valid_pixels=added, added_fraction_of_native_image=added / old_valid.size,
        expected_enclosed_hole_pixels=support["enclosed_hole_added_pixel_count"],
        enclosed_hole_components=support["enclosed_hole_added_component_count"],
        expected_vertical_gap_pixels=support.get("vertical_gap_added_pixel_count", 0),
        vertical_gap_runs=support.get("vertical_gap_added_run_count", 0),
        vertical_gap_max_length_px=64,
        rule1_added_valid_pixels=rule1_added, rule2_added_valid_pixels=rule2_added,
        photo_support_old_lower_strip_retained=support.get("old_photo_support_retained", True),
        enclosed_hole_policy="fill_enclosed_v1", enclosed_hole_max_area_px=60_000,
        old_valid_max_abs_delta=maximum if old_count else None,
        old_valid_lost_pixels=lost, added_outside_allowed_fill_pixels=outside,
        filled_hole_pixels_without_finite_sidecar=int(np.count_nonzero(rule1 & ~new_valid)),
        filled_vertical_gap_pixels_without_finite_sidecar=int(np.count_nonzero(rule2 & ~new_valid)),
        old_validity_finite_mismatch_pixels=finite_mismatch_old,
        new_validity_finite_mismatch_pixels=finite_mismatch_new,
        old_valid_nonfinite_new_pixels=invalid_comparison,
        payload_hashes_live_verified=verify_payload_hashes,
        old_sidecar_sha256=pair["old_frame"]["v5_log_linear_rgb_sha256"],
        new_sidecar_sha256=pair["new_frame"]["v5_log_linear_rgb_sha256"],
        passed=(old_count > 0 and maximum <= SIDECAR_DELTA_LIMIT
            and not any((lost, outside, finite_mismatch_old, finite_mismatch_new, invalid_comparison))))


def _parallel(pairs: list[dict[str, Any]], function: Any, workers: int) -> list[dict[str, Any]]:
    if not isinstance(workers, int) or not 1 <= workers <= 4:
        raise ValueError("Rerun QA workers must be an integer from 1 to 4")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ffac-rerun-qa") as pool:
        return list(pool.map(function, pairs))


def inspect_sources(old_candidate: str | Path, new_batch_root: str | Path,
                    output_dir: str | Path, *, workers: int = 4,
                    verify_payload_hashes: bool = True) -> dict[str, Any]:
    """Check sealed 43/864 float sidecars, writing exclusive ``qa_sidecar.json``.

    Live payload hashing is on by default.  Turning it off is explicitly
    recorded and suitable only where materialization already verified the
    immutable payloads; source receipts and manifests are still live hashed.
    """
    output = Path(output_dir) / "qa_sidecar.json"
    if output.exists():
        raise FileExistsError(output)
    pairs, binding = _source_pairs(Path(old_candidate), Path(new_batch_root))
    rows = _parallel(pairs, lambda pair: _compare_source_frame(pair,
        verify_payload_hashes=verify_payload_hashes), workers)
    return _new_report(output, dict(
        contract="ffac-track1-display-mask-rerun-sidecar-qa-v1", created_utc=utc_now(),
        old_candidate=str(old_candidate), new_batch_root=str(new_batch_root), batch=binding,
        runs=EXPECTED_RUNS, frames=len(rows), workers=workers,
        old_valid_max_abs_delta_limit=SIDECAR_DELTA_LIMIT,
        payload_hashes_live_verified=verify_payload_hashes,
        max_old_valid_abs_delta=max(row["old_valid_max_abs_delta"] or 0 for row in rows),
        total_added_valid_pixels=sum(row["added_valid_pixels"] for row in rows),
        total_rule1_added_valid_pixels=sum(row["rule1_added_valid_pixels"] for row in rows),
        total_rule2_added_valid_pixels=sum(row["rule2_added_valid_pixels"] for row in rows),
        failed_frames=[dict(run=row["run"], frame_index=row["frame_index"])
                       for row in rows if not row["passed"]],
        passed=all(row["passed"] for row in rows), per_frame=rows))


def _phase_folder(candidate: Path, run: str) -> Path:
    package = candidate / "validation" / "run_packages" / run
    postflight = package / "postflight_manifest.json"
    if postflight.is_file():
        rows = [row for row in _read(postflight).get("jobs", []) if row.get("run") == run]
        if len(rows) != 1:
            raise ValueError(f"Rerun QA ambiguous saved phase run: {run}")
        row = rows[0]
        return (Path(row["phase_map_output_dir"]) / "native_phase_maps" / run
                / f"A_{row['params_sha256'][:16]}")
    folders = list((package / "review_assets" / "native_phase_maps" / run).glob("A_*"))
    if len(folders) != 1:
        raise ValueError(f"Rerun QA missing or ambiguous final categorical maps: {run}")
    return folders[0]


def _phase_labels(path: Path, manifest: dict[str, Any]) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"Rerun QA saved categorical phase map must be RGBA: {path}")
        rgba = np.asarray(image)
    x, y, width, height = map(int, manifest["content_box_xywh"])
    canvas_width, canvas_height = map(int, manifest["canvas_wh"])
    if (rgba.shape != (canvas_height, canvas_width, 4)
            or [height, width] != manifest["analysis_image_shape_hw"]
            or x < 0 or y < 0 or x + width > canvas_width or y + height > canvas_height):
        raise ValueError("Rerun QA categorical map has another native canvas")
    rgba = rgba[y:y + height, x:x + width]
    labels = np.zeros((height, width), np.uint8)
    known = rgba[..., 3] == 0
    for value, (name, color) in enumerate(_COLORS.items(), start=1):
        mask = np.all(rgba == color, axis=-1)
        labels[mask] = value
        known |= mask
    if not known.all():
        raise ValueError(f"Rerun QA phase PNG contains noncategorical colours: {path}")
    return labels


def _compare_phase_frame(pair: dict[str, Any], old_folder: Path, new_folder: Path
                         ) -> dict[str, Any]:
    name = f"frame_{pair['frame_index']:03d}.png"
    old_path, new_path = old_folder / name, new_folder / name
    old, new = _phase_labels(old_path, pair["old_manifest"]), _phase_labels(new_path, pair["new_manifest"])
    holes, support = _filled_holes(pair)
    rule1 = support.get("_enclosed_rule1_mask", holes)
    rule2 = support.get("_vertical_gap_rule2_mask", np.zeros_like(holes))
    guard = support.get("_outer_feature_guard_mask", np.zeros_like(holes))
    near_holes = cv2.dilate(holes.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    plume = ((old >= 1) & (old <= 3)) | ((new >= 1) & (new <= 3))
    changed_phase = ((old != new) & ~near_holes
        & (((old >= 1) & (old <= 3)) | ((new >= 1) & (new <= 3))))
    change_count = int(np.count_nonzero(changed_phase))
    plume_count = int(np.count_nonzero(plume))
    fraction = change_count / plume_count if plume_count else 0.0
    unresolved = int(np.count_nonzero((new == 4) & rule1))
    unresolved_rule2 = int(np.count_nonzero((new == 4) & rule2))
    unresolved_rule1_interior = int(np.count_nonzero((new == 4) & rule1 & ~guard))
    unresolved_rule2_interior = int(np.count_nonzero((new == 4) & rule2 & ~guard))
    band = np.zeros_like(holes)
    # Identify the pre-declared bottom band by its geometry, not phase colour:
    # a wide rule-2 component in the lower 30% of the native image. Its extent
    # may slant after the frozen homography, so no horizontal-row fit is used.
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        rule2.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    candidates = [label for label in range(1, count)
        if stats[label, cv2.CC_STAT_WIDTH] >= holes.shape[1] * 0.5
        and stats[label, cv2.CC_STAT_TOP] >= holes.shape[0] * 0.7]
    if candidates:
        label = max(candidates, key=lambda value: stats[value, cv2.CC_STAT_AREA])
        band = components == label
    early_ac28 = (pair["run"] == "ac28" and pair["new_frame"].get("elapsed_h") is not None
                  and 0.25 <= float(pair["new_frame"]["elapsed_h"]) <= 1.0)
    crossing_components = None
    if early_ac28:
        crossing_components = 0
        if band.any():
            rows = np.flatnonzero(band.any(axis=1))
            y0, y1 = max(0, int(rows[0]) - 80), min(new.shape[0], int(rows[-1]) + 81)
            local_band = band[y0:y1]
            local_plume = ((new[y0:y1] >= 1) & (new[y0:y1] <= 3))
            above = np.zeros_like(local_band)
            below = np.zeros_like(local_band)
            above[:-1] = local_band[1:]
            below[1:] = local_band[:-1]
            above &= ~local_band
            below &= ~local_band
            _, plume_components = cv2.connectedComponents(local_plume.astype(np.uint8), connectivity=8)
            bridge_ids = np.intersect1d(np.unique(plume_components[above]),
                np.unique(plume_components[below]))
            bridge_ids = np.intersect1d(bridge_ids, np.unique(plume_components[local_band]))
            crossing_components = int(np.count_nonzero(bridge_ids))  # label zero is background
    return dict(run=pair["run"], frame_index=pair["frame_index"],
        elapsed_h=pair["new_frame"].get("elapsed_h"),
        old_phase_map_path=str(old_path), new_phase_map_path=str(new_path),
        old_phase_map_sha256=sha256_file(old_path), new_phase_map_sha256=sha256_file(new_path),
        enclosed_filled_hole_pixels=support["enclosed_hole_added_pixel_count"],
        vertical_gap_rule2_pixels=support.get("vertical_gap_added_pixel_count", 0),
        unresolved_pixels_in_filled_holes=unresolved,
        unresolved_pixels_in_filled_vertical_gaps=unresolved_rule2,
        unresolved_pixels_in_filled_holes_outside_outer_8px_guard=unresolved_rule1_interior,
        unresolved_pixels_in_filled_vertical_gaps_outside_outer_8px_guard=unresolved_rule2_interior,
        unresolved_filled_pixels_in_outer_8px_feature_guard=int(np.count_nonzero((new == 4) & holes & guard)),
        bottom_band_rule2_pixels=int(np.count_nonzero(band)),
        bottom_band_unresolved_pixels=int(np.count_nonzero((new == 4) & band)),
        early_ac28_bottom_band_continuity_required=early_ac28,
        early_ac28_crossing_plume_component_count=crossing_components,
        early_ac28_bottom_band_continuity_pass=(None if not early_ac28 else crossing_components > 0),
        unresolved_pixels_in_native_content=int(np.count_nonzero(new == 4)),
        old_aqueous_pixels=int(np.count_nonzero(old == 1)), new_aqueous_pixels=int(np.count_nonzero(new == 1)),
        old_gas_pixels=int(np.count_nonzero(old == 2)), new_gas_pixels=int(np.count_nonzero(new == 2)),
        old_mixed_pixels=int(np.count_nonzero(old == 3)), new_mixed_pixels=int(np.count_nonzero(new == 3)),
        changed_phase_pixels_outside_30px_hole_dilation=change_count,
        old_or_new_plume_pixels=plume_count,
        outside_change_fraction_of_plume=fraction,
        outside_changes_flagged=fraction > PHASE_CHANGE_PLUME_FRACTION_LIMIT,
        changed_all_categorical_pixels_outside_30px_hole_dilation=int(np.count_nonzero((old != new) & ~near_holes)),
        passed=(unresolved_rule1_interior == 0 and unresolved_rule2_interior == 0
            and fraction <= PHASE_CHANGE_PLUME_FRACTION_LIMIT
            and (not early_ac28 or crossing_components > 0)))


def inspect_maps(old_candidate: str | Path, new_candidate: str | Path,
                 new_batch_root: str | Path, output_dir: str | Path, *,
                 workers: int = 4) -> dict[str, Any]:
    """Compare final categorical native maps; never launch or repair inference."""
    output = Path(output_dir) / "qa_phase_maps.json"
    if output.exists():
        raise FileExistsError(output)
    pairs, binding = _source_pairs(Path(old_candidate), Path(new_batch_root))
    folders = {run: (_phase_folder(Path(old_candidate), run),
        _phase_folder(Path(new_candidate), run)) for run in {pair["run"] for pair in pairs}}
    rows = _parallel(pairs, lambda pair: _compare_phase_frame(
        pair, *folders[pair["run"]]), workers)
    return _new_report(output, dict(
        contract="ffac-track1-display-mask-rerun-native-phase-qa-v1", created_utc=utc_now(),
        old_candidate=str(old_candidate), new_candidate=str(new_candidate),
        new_batch_root=str(new_batch_root), batch=binding,
        runs=EXPECTED_RUNS, frames=len(rows), workers=workers,
        phase_encoding=_COLORS, photographic_display_used=False,
        hole_dilation_radius_px=30, plume_denominator="union_old_and_new_all_predicted_phases",
        outside_change_fraction_limit=PHASE_CHANGE_PLUME_FRACTION_LIMIT,
        total_unresolved_pixels_in_filled_holes=sum(row["unresolved_pixels_in_filled_holes"] for row in rows),
        total_unresolved_pixels_in_filled_vertical_gaps=sum(row["unresolved_pixels_in_filled_vertical_gaps"] for row in rows),
        total_unresolved_filled_pixels_outside_outer_8px_feature_guard=sum(
            row["unresolved_pixels_in_filled_holes_outside_outer_8px_guard"]
            + row["unresolved_pixels_in_filled_vertical_gaps_outside_outer_8px_guard"] for row in rows),
        early_ac28_bottom_band_continuity=[dict(frame_index=row["frame_index"], elapsed_h=row["elapsed_h"],
            crossing_plume_components=row["early_ac28_crossing_plume_component_count"],
            passed=row["early_ac28_bottom_band_continuity_pass"])
            for row in rows if row["early_ac28_bottom_band_continuity_required"]],
        max_outside_change_fraction_of_plume=max(row["outside_change_fraction_of_plume"] for row in rows),
        failed_frames=[dict(run=row["run"], frame_index=row["frame_index"])
                       for row in rows if not row["passed"]],
        passed=all(row["passed"] for row in rows), per_frame=rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("sources", "maps"))
    parser.add_argument("--old-candidate", type=Path, required=True)
    parser.add_argument("--new-batch-root", type=Path, required=True)
    parser.add_argument("--new-candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.kind == "sources":
        report = inspect_sources(args.old_candidate, args.new_batch_root, args.output_dir, workers=args.workers)
    else:
        if args.new_candidate is None:
            parser.error("maps requires --new-candidate")
        report = inspect_maps(args.old_candidate, args.new_candidate, args.new_batch_root,
            args.output_dir, workers=args.workers)
    print(json.dumps(dict(passed=report["passed"], frames=report["frames"],
        failed_frames=report["failed_frames"]), allow_nan=False))


if __name__ == "__main__":
    main()
