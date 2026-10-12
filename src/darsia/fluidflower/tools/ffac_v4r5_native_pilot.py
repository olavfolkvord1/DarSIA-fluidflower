"""Diagnostic full-native projection of the frozen V4-R5 optical correction.

This is numerical plumbing for a source-bound, ineligible pilot.  The caller
must verify original RAW/JPEG/review hashes, homographies, chart registration,
the calibration manifest and the source-eligible output mask.  The module
neither fits parameters nor changes labels or source-image coordinates.
"""

from __future__ import annotations

import numpy as np

from . import ffac_v5_observed_sand as observed
from . import ffac_v5_refgeom as refgeom
from .ffac_v5_old_field_native import apply_old_field_native


NATIVE_SHAPE_CHW = (3, 2979, 5472)
AREA_GRID_SHAPE_CHW = (3, 372, 684)


def native_area_log8(linear_native_rgb: np.ndarray) -> np.ndarray:
    """Log of the exact 8x8 arithmetic RAW mean used by the EA baseline fit."""
    linear = np.asarray(linear_native_rgb)
    if linear.dtype != np.float32 or linear.shape != NATIVE_SHAPE_CHW:
        raise TypeError("Expected full-native CHW float32 camera-linear RAW")
    _, height, width = NATIVE_SHAPE_CHW
    gh, gw = AREA_GRID_SHAPE_CHW[1:]
    if gh * 8 > height or gw * 8 != width:
        raise ValueError("Native/area-grid geometry is inconsistent")
    area = np.mean(
        linear[:, :gh * 8, :].reshape(3, gh, 8, gw, 8),
        axis=(2, 4), dtype=np.float64,
    )
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("RAW 8x8 area mean must be finite and positive")
    return np.log(area).astype(np.float32)


def frame_light_from_native_grid(
    frame_area_log8: np.ndarray,
    fit,
    grid: observed.ReferenceGrid,
    h_ac28_to_frame_native: np.ndarray,
    frame_chart_xy8: np.ndarray,
    frame_chart_usable: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit frozen frame light in AC28 reference coordinates, preserving H.

    ``frame_chart_xy8`` belongs to this frame's own 1/8 native coordinates.
    The corrected native image is never spatially warped; only the area-grid
    signal and chart locations enter the reference-light fit.
    """
    source_grid = np.asarray(frame_area_log8)
    if source_grid.dtype != np.float32 or source_grid.shape != AREA_GRID_SHAPE_CHW:
        raise TypeError("Expected source-bound 3x372x684 float32 area-log grid")
    h = refgeom.validate_homography(h_ac28_to_frame_native)
    chart_xy = np.asarray(frame_chart_xy8)
    chart_usable = np.asarray(frame_chart_usable)
    if chart_xy.shape != (24, 2) or chart_usable.shape != (24,):
        raise ValueError("Expected 24 source-bound per-frame chart centres and flags")
    if not np.issubdtype(chart_xy.dtype, np.floating) or chart_usable.dtype != bool:
        raise TypeError("Chart centres must be floats and usability must be boolean")

    centre = refgeom.remap_to_reference(source_grid, h)
    expanded = np.full((3, *grid.shape_hw), np.nan, np.float32)
    x0, y0 = -grid.origin_cell_xy[0], -grid.origin_cell_xy[1]
    h0, w0 = grid.original_shape_hw
    if centre.shape != (3, h0, w0) or x0 < 0 or y0 < 0 or \
            y0 + h0 > grid.shape_hw[0] or x0 + w0 > grid.shape_hw[1]:
        raise ValueError("Frozen area grid does not fit the expanded reference")
    expanded[:, y0:y0 + h0, x0:x0 + w0] = centre
    xy_ref = refgeom.chart_xy_to_reference(chart_xy, h).astype(np.float32)
    xy_ref -= np.asarray(grid.origin_cell_xy, np.float32)
    return observed.estimate_frame_light_expanded_diagnostic(
        fit, expanded, grid,
        frame_chart_xy_ref8=xy_ref,
        frame_chart_usable=chart_usable,
    )


def correct_frame_native_v4r5(
    linear_native_rgb: np.ndarray,
    fit,
    virtual_reference,
    grid: observed.ReferenceGrid,
    h_ac28_to_frame_native: np.ndarray,
    source_eligible_mask: np.ndarray,
    *,
    exact_baseline_source: bool,
    frame_chart_xy8: np.ndarray | None,
    frame_chart_usable: np.ndarray | None,
    old_field_ref8: np.ndarray | None,
    expected_area_log8: np.ndarray | None = None,
    tile_rows: int = 128,
) -> tuple[np.ndarray, dict]:
    """Correct one RAW frame using the immutable V4-R5 fit and source H.

    ``exact_baseline_source`` may only be true after the caller has checked
    that this RAW/JPEG/review triple is the SHA-bound B source for ``fit``.
    Otherwise source-bound chart centres and usability are mandatory.  The
    old-light residual is an explicit input, never inferred by run name.
    Output is CHW float32 *log-linear* RAW; unsupported pixels remain NaN.
    """
    linear = np.asarray(linear_native_rgb)
    if linear.dtype != np.float32 or linear.shape != NATIVE_SHAPE_CHW:
        raise TypeError("Expected full-native CHW float32 camera-linear RAW")
    eligible = np.asarray(source_eligible_mask)
    if eligible.dtype != bool or eligible.shape != NATIVE_SHAPE_CHW[1:]:
        raise TypeError("Expected exact native boolean source-eligibility mask")
    h = refgeom.validate_homography(h_ac28_to_frame_native)
    area = native_area_log8(linear)
    if expected_area_log8 is not None:
        expected = np.asarray(expected_area_log8)
        if expected.dtype != np.float32 or expected.shape != AREA_GRID_SHAPE_CHW:
            raise TypeError("Expected source area receipt must be 3x372x684 float32")
        if not np.allclose(area, expected, rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError("Full-native RAW does not match source-bound 1/8 grid")

    if exact_baseline_source:
        if frame_chart_xy8 is not None or frame_chart_usable is not None:
            raise ValueError("Exact B source must use its frozen zero light field")
        light = None
        light_info = {"mode": "exact-source-bound-B", "zero_field": True}
    else:
        if frame_chart_xy8 is None or frame_chart_usable is None:
            raise ValueError("Non-B frame requires its source-bound ColorChecker")
        light, light_info = frame_light_from_native_grid(
            area, fit, grid, h, frame_chart_xy8, frame_chart_usable,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_native = np.where(linear > 2, np.log(linear), np.nan).astype(np.float32)
    corrected = observed.apply_native(
        log_native, fit, virtual_reference, grid, h,
        source_eligible_mask=eligible, light_field_ref8=light,
        tile_rows=tile_rows,
    )
    if old_field_ref8 is not None:
        old_field = np.asarray(old_field_ref8)
        if old_field.dtype != np.float32 or old_field.shape != grid.shape_hw:
            raise TypeError("Old-light residual must match the float32 reference grid")
        corrected = apply_old_field_native(
            corrected, old_field, grid, h, tile_rows=tile_rows,
        )
    return corrected, {
        "contract": "ffac-v4r5-full-native-diagnostic-v1",
        "scientific_or_model_eligibility": False,
        "exact_baseline_source": bool(exact_baseline_source),
        "per_frame_light": light_info,
        "old_light_residual_applied": old_field_ref8 is not None,
        "output_encoding": "chw-float32-log-camera-linear-rgb-with-nan-invalid",
        "source_eligible_pixel_count": int(np.count_nonzero(eligible)),
        "finite_eligible_pixel_count": int(np.count_nonzero(
            eligible & np.all(np.isfinite(corrected), axis=0))),
    }
