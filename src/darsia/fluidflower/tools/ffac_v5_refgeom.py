"""Geometry-aware FF_AC v5 photometry on the native review coordinate system.

All installation assets and baseline fits live in one explicitly registered
reference geometry. Source frames and their frozen annotations stay in each
run's own native review geometry. Only correction fields are resampled; the
source image is never warped through the cross-run homography.

This is a separate candidate from ``ffac_v5_photometry``. The caller must
verify source, geometry, and physical-facies receipts before using it.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ffac_v5_photometry as v5


def validate_homography(h_ref_to_run: np.ndarray) -> np.ndarray:
    """Return a normalized, invertible full-native reference-to-run H."""
    h = np.asarray(h_ref_to_run, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError("Expected finite 3x3 cross-run homography")
    if abs(np.linalg.det(h)) < 1e-10 or abs(h[2, 2]) < 1e-10:
        raise ValueError("Degenerate cross-run homography")
    return h / h[2, 2]


def _map_coordinates(
    h_ref_to_run: np.ndarray,
    *,
    source_shape: tuple[int, int],
    destination_shape: tuple[int, int],
    destination_y0: int = 0,
    destination_y1: int | None = None,
    run_to_reference: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inverse-sampling maps using native pixel-centre coordinates.

    A 1/8 cell j corresponds to native pixel 8*j+3.5. ``run_to_reference``
    requests run source sampled at each reference destination point.
    """
    src = tuple(source_shape)
    dst = tuple(destination_shape)
    if src not in (v5.GRID, v5.NATIVE) or dst not in (v5.GRID, v5.NATIVE):
        raise ValueError(f"Unexpected source/destination geometry {src} -> {dst}")
    y1 = dst[0] if destination_y1 is None else destination_y1
    if not 0 <= destination_y0 < y1 <= dst[0]:
        raise ValueError("Invalid destination tile")
    scale_out = 8 if dst == v5.GRID else 1
    scale_in = 8 if src == v5.GRID else 1
    yy, xx = np.mgrid[destination_y0:y1, 0:dst[1]]
    xn = scale_out * xx.astype(np.float64) + (scale_out - 1) / 2
    yn = scale_out * yy.astype(np.float64) + (scale_out - 1) / 2
    h = validate_homography(h_ref_to_run)
    # cv2.remap needs source coordinates for each destination point.
    t = h if run_to_reference else np.linalg.inv(h)
    den = t[2, 0] * xn + t[2, 1] * yn + t[2, 2]
    if np.any(np.abs(den) < 1e-9):
        raise ValueError("Homography maps a destination pixel to infinity")
    xs = (t[0, 0] * xn + t[0, 1] * yn + t[0, 2]) / den
    ys = (t[1, 0] * xn + t[1, 1] * yn + t[1, 2]) / den
    mx = ((xs - (scale_in - 1) / 2) / scale_in).astype(np.float32)
    my = ((ys - (scale_in - 1) / 2) / scale_in).astype(np.float32)
    return mx, my


def remap_to_reference(
    run_array: np.ndarray,
    h_ref_to_run: np.ndarray,
    *,
    categorical: bool = False,
    destination_shape: tuple[int, int] = v5.GRID,
) -> np.ndarray:
    """Sample a run image/mask in the AC28 reference grid.

    Continuous arrays use bilinear interpolation; categorical arrays use
    nearest-neighbour. Out-of-footprint continuous values are NaN and class
    values are zero. For floating arrays, validity is interpolated separately
    to avoid NaN bleed near the review crop edge.
    """
    a = np.asarray(run_array)
    if a.ndim not in (2, 3):
        raise ValueError("Expected 2D or channel-first 3D array")
    src_shape = a.shape[-2:]
    mx, my = _map_coordinates(
        h_ref_to_run, source_shape=src_shape,
        destination_shape=destination_shape, run_to_reference=True,
    )
    mode = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    planes = a[None] if a.ndim == 2 else a
    out = []
    for plane in planes:
        if categorical:
            z = cv2.remap(np.asarray(plane), mx, my, mode,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            valid = np.isfinite(plane).astype(np.float32)
            num = cv2.remap(np.where(np.isfinite(plane), plane, 0).astype(np.float32),
                            mx, my, mode, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
            den = cv2.remap(valid, mx, my, mode,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            z = np.where(den > .999, num / np.maximum(den, 1e-6), np.nan)
        out.append(z)
    return out[0] if a.ndim == 2 else np.stack(out)


def chart_xy_to_reference(chart_xy_run8: np.ndarray, h_ref_to_run: np.ndarray) -> np.ndarray:
    """Map 24 source-bound run chart centres to 1/8 reference coordinates."""
    xy = np.asarray(chart_xy_run8, dtype=np.float64)
    if xy.shape != (24, 2):
        raise ValueError("Expected 24 chart centres")
    h = np.linalg.inv(validate_homography(h_ref_to_run))
    p = 8 * xy + 3.5
    q = cv2.perspectiveTransform(p.astype(np.float64)[None], h)[0]
    return (q - 3.5) / 8


def _remap_reference_tile(
    ref_array: np.ndarray, h_ref_to_run: np.ndarray,
    destination_shape: tuple[int, int], y0: int, y1: int,
    *, coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    categorical: bool = False,
) -> np.ndarray:
    a = np.asarray(ref_array)
    if coordinates is None:
        mx, my = _map_coordinates(
            h_ref_to_run, source_shape=a.shape[-2:],
            destination_shape=destination_shape, destination_y0=y0,
            destination_y1=y1,
        )
    else:
        if a.shape[-2:] != v5.GRID:
            raise ValueError("Shared reference coordinates require 1/8 asset")
        mx, my = coordinates
    mode = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    planes = a[None] if a.ndim == 2 else a
    out = [cv2.remap(np.asarray(p, np.float32), mx, my, mode,
                     borderMode=cv2.BORDER_CONSTANT if categorical else cv2.BORDER_REPLICATE,
                     borderValue=0)
           for p in planes]
    return out[0] if a.ndim == 2 else np.stack(out)


def _native_exact_sand_field(
    fit_ref: v5.Baseline, reference: v5.Baseline,
    facies_tile: np.ndarray, mx_ref8: np.ndarray, my_ref8: np.ndarray,
) -> np.ndarray:
    """Evaluate v5 polynomial/facies correction at physical reference points."""
    x = (mx_ref8 - v5.GRID[1] / 2) / (v5.GRID[1] / 2)
    y = (my_ref8 - v5.GRID[0] / 2) / (v5.GRID[0] / 2)
    coef = reference.P - fit_ref.P
    field = np.zeros(facies_tile.shape, np.float32)
    k = 0
    for d in range(1, 5):
        for ax in range(d + 1):
            ay = d - ax
            field += (coef[k] * x**ax * y**ay).astype(np.float32)
            k += 1
    for fac, level in fit_ref.a.items():
        if fac in reference.a:
            field[facies_tile == fac] += reference.a[fac] - level
    return field


def estimate_frame_light_refgeom(
    fit_ref: v5.Baseline,
    frame_log_run8: np.ndarray,
    h_ref_to_run: np.ndarray,
    *,
    frame_chart_xy_run8: np.ndarray,
    frame_chart_usable: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Estimate v5's frame-light field in the common AC28 geometry.

    Sand is transported by the certified H for this one-eighth *fit only*;
    checker locations are measured independently on this exact frame. The
    returned field is in reference coordinates, for direct resampling during
    native correction. There is no guessed baseline chart fallback.
    """
    source = v5._log_rgb(frame_log_run8, v5.GRID)
    xy = np.asarray(frame_chart_xy_run8, np.float64)
    usable = np.asarray(frame_chart_usable, bool)
    if xy.shape != (24, 2) or usable.shape != (24,):
        raise ValueError("Exact frame chart centres and usable mask required")
    frame_ref = remap_to_reference(source, h_ref_to_run)
    xy_ref = chart_xy_to_reference(xy, h_ref_to_run)
    field, info = v5.estimate_frame_light_grid(
        fit_ref, frame_ref, frame_chart_xy8=xy_ref,
        frame_chart_usable=usable,
    )
    info["coordinate_frame"] = "ac28-reference-1of8"
    info["frame_chart_source"] = "exact-run-frame-registered-then-projected"
    return field, info


def apply_refgeom_v5(
    log_rgb_run: np.ndarray,
    fit_ref: v5.Baseline,
    reference: v5.Baseline,
    h_ref_to_run: np.ndarray,
    *,
    facies_run_native: np.ndarray | None = None,
    light_field_ref8: np.ndarray | None = None,
    tile_rows: int = 256,
) -> np.ndarray:
    """Correct run-native RAW while keeping its image and labels unwarped.

    ``fit_ref`` and ``reference`` must have been fitted on the *same*
    registered AC28 reference grid. For native frames, ``facies_run_native``
    is the source-bound *optical support class* used by that fit. For v4 it is
    one synthetic safe-interior class (1) or uncertain/excluded (0), never the
    physical 8-ID facies. A caller must check this mask, H and calibration
    assets against the frozen source revision. Coverage outside the common
    reference field stays NaN; no uncorrected fallback is applied.
    """
    src = v5._log_rgb(log_rgb_run)
    shape = src.shape[1:]
    if shape not in (v5.GRID, v5.NATIVE):
        raise ValueError(f"Unexpected run source shape {shape}")
    if fit_ref.fac.shape != v5.GRID or reference.fac.shape != v5.GRID:
        raise ValueError("Fits must use the reference grid")
    if shape == v5.NATIVE:
        if facies_run_native is None or np.shape(facies_run_native) != v5.NATIVE:
            raise ValueError("Native correction requires a bound optical-support mask")
    h = validate_homography(h_ref_to_run)
    field_ref, _ = v5.baseline_field(fit_ref, reference)
    light_ref = None
    if light_field_ref8 is not None:
        lf = np.asarray(light_field_ref8, np.float32)
        if lf.shape != v5.GRID:
            raise ValueError("Per-frame light field must be in AC28 reference grid")
        light_ref = lf
    out = np.empty_like(src)
    for lo in range(0, shape[0], tile_rows):
        hi = min(lo + tile_rows, shape[0])
        mx, my = _map_coordinates(h, source_shape=v5.GRID,
                                  destination_shape=shape,
                                  destination_y0=lo, destination_y1=hi)
        coords = (mx, my)
        field = _remap_reference_tile(field_ref, h, shape, lo, hi,
                                      coordinates=coords)
        chart = _remap_reference_tile(fit_ref.chart.astype(np.float32), h, shape, lo, hi,
                                      coordinates=coords,
                                      categorical=True) > .5
        taper = _remap_reference_tile(fit_ref.taper, h, shape, lo, hi,
                                      coordinates=coords)
        if shape == v5.NATIVE:
            facies = np.asarray(facies_run_native[lo:hi])
        else:
            facies = _remap_reference_tile(fit_ref.fac, h, shape, lo, hi,
                                           coordinates=coords,
                                           categorical=True).astype(np.int16)
        sand = (facies >= 1) & ~chart
        # The source 1/8 grid uses cell centres, so the first/last native
        # pixels extend half a cell beyond its centre lattice. Allow only
        # that fixed support (and the three truncated last native rows),
        # never arbitrary cross-run out-of-view extrapolation.
        valid_ref = (np.isfinite(field) & (mx >= -.5) &
                     (mx <= v5.GRID[1] - .5) & (my >= -.5) &
                     (my <= v5.GRID[0]))
        if shape == v5.NATIVE:
            exact = _native_exact_sand_field(fit_ref, reference, facies, mx, my)
            common = np.isin(facies, list(set(fit_ref.a) & set(reference.a)))
            mask = sand & common & valid_ref
            field[mask] = exact[mask]
        if light_ref is not None:
            field -= _remap_reference_tile(light_ref, h, shape, lo, hi,
                                           coordinates=coords)
        rgb = v5._colour(src[:, lo:hi], fit_ref.Minv)
        rgb += field[None]
        if fit_ref.F is not None:
            flat = _remap_reference_tile(fit_ref.F, h, shape, lo, hi,
                                         coordinates=coords)
            gb0 = _remap_reference_tile(fit_ref.gb0, h, shape, lo, hi,
                                        coordinates=coords)
            alpha = np.clip((v5.GB_SAT - (rgb[1] - rgb[2])) /
                            np.maximum(v5.GB_SAT - gb0, .2), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            tap = np.where(sand, 1.0, taper * ~chart)
            neutral = flat[1:2]
            rgb -= (neutral + alpha[None] * (flat - neutral)) * tap[None]
        rgb[:, ~valid_ref] = np.nan
        out[:, lo:hi] = rgb
    return out
