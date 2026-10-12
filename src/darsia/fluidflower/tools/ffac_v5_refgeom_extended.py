"""Source-backed RAW ring correction around the frozen FF_AC V4 native crop.

This module leaves ``ffac_v5_refgeom`` byte-for-byte unchanged because V4
calibration receipts bind that module's SHA. The central native result is
computed by the frozen implementation; only the four non-overlapping RAW
ring rectangles are corrected here. Pixels outside its AC28 reference field
remain NaN. The caller separately binds RAW, temporal H, safe mask, and
physical observability receipts; this module does not approve phase fitting.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ffac_v5_photometry as v5
from . import ffac_v5_refgeom as rg


def _ring_coordinates(
    h_ref_to_run: np.ndarray,
    x0: int, x1: int, y0: int, y1: int,
    pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one padded destination rectangle to AC28 1/8 cell coordinates."""
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xn = xx.astype(np.float64) - pad
    yn = yy.astype(np.float64) - pad
    inverse = np.linalg.inv(rg.validate_homography(h_ref_to_run))
    den = inverse[2, 0] * xn + inverse[2, 1] * yn + inverse[2, 2]
    if np.any(np.abs(den) < 1e-9):
        raise ValueError("Cross-run homography maps RAW ring to infinity")
    xref = (inverse[0, 0] * xn + inverse[0, 1] * yn + inverse[0, 2]) / den
    yref = (inverse[1, 0] * xn + inverse[1, 1] * yn + inverse[1, 2]) / den
    return ((xref - 3.5) / 8).astype(np.float32), ((yref - 3.5) / 8).astype(np.float32)


def _correct_ring_rectangle(
    source: np.ndarray,
    safe_mask: np.ndarray,
    fit: v5.Baseline,
    reference: v5.Baseline,
    h_ref_to_run: np.ndarray,
    field_ref: np.ndarray,
    light_ref: np.ndarray | None,
    pad: int,
    x0: int, x1: int, y0: int, y1: int,
    tile_rows: int,
) -> np.ndarray:
    output = np.empty((3, y1-y0, x1-x0), np.float32)
    for lo in range(y0, y1, tile_rows):
        hi = min(lo + tile_rows, y1)
        mx, my = _ring_coordinates(h_ref_to_run, x0, x1, lo, hi, pad)
        coords = (mx, my)
        # The frozen module's sampling/equations are reused exactly; only
        # destination coordinates extend beyond the old review crop.
        field = rg._remap_reference_tile(field_ref, h_ref_to_run, v5.NATIVE,
                                         lo, hi, coordinates=coords)
        chart = rg._remap_reference_tile(
            fit.chart.astype(np.float32), h_ref_to_run, v5.NATIVE,
            lo, hi, coordinates=coords, categorical=True,
        ) > .5
        taper = rg._remap_reference_tile(fit.taper, h_ref_to_run, v5.NATIVE,
                                         lo, hi, coordinates=coords)
        facies = safe_mask[lo:hi, x0:x1]
        sand = (facies >= 1) & ~chart
        valid_ref = (np.isfinite(field) & (mx >= -.5) &
                     (mx <= v5.GRID[1] - .5) & (my >= -.5) &
                     (my <= v5.GRID[0]))
        exact = rg._native_exact_sand_field(fit, reference, facies, mx, my)
        common = np.isin(facies, list(set(fit.a) & set(reference.a)))
        mask = sand & common & valid_ref
        field[mask] = exact[mask]
        if light_ref is not None:
            field -= rg._remap_reference_tile(light_ref, h_ref_to_run,
                                              v5.NATIVE, lo, hi,
                                              coordinates=coords)
        rgb = v5._colour(source[:, lo:hi, x0:x1], fit.Minv)
        rgb += field[None]
        if fit.F is not None:
            flat = rg._remap_reference_tile(fit.F, h_ref_to_run, v5.NATIVE,
                                            lo, hi, coordinates=coords)
            gb0 = rg._remap_reference_tile(fit.gb0, h_ref_to_run, v5.NATIVE,
                                           lo, hi, coordinates=coords)
            alpha = np.clip((v5.GB_SAT - (rgb[1] - rgb[2])) /
                            np.maximum(v5.GB_SAT - gb0, .2), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            tap = np.where(sand, 1.0, taper * ~chart)
            neutral = flat[1:2]
            rgb -= (neutral + alpha[None] * (flat - neutral)) * tap[None]
        rgb[:, ~valid_ref] = np.nan
        output[:, lo-y0:hi-y0] = rgb
    return output


def apply_refgeom_v5_with_raw_ring(
    log_rgb_padded: np.ndarray,
    fit_ref: v5.Baseline,
    reference: v5.Baseline,
    h_ref_to_run: np.ndarray,
    *,
    facies_padded: np.ndarray,
    destination_origin_xy: tuple[int, int],
    light_field_ref8: np.ndarray | None = None,
    tile_rows: int = 256,
) -> np.ndarray:
    """Correct a sensor-backed symmetric RAW pad while preserving V4 centre.

    ``destination_origin_xy=(-pad,-pad)`` and shape
    ``(3,2979+2pad,5472+2pad)`` are required. The safe optical mask must be
    projected independently onto that entire destination; padding an old
    mask by edge replication is not certified. NaNs at unsupported sensor or
    AC28 reference coordinates remain NaNs.
    """
    source = v5._log_rgb(log_rgb_padded)
    origin = tuple(destination_origin_xy)
    if len(origin) != 2 or origin[0] != origin[1] or not isinstance(origin[0], (int, np.integer)):
        raise ValueError("RAW ring requires integer symmetric native origin")
    pad = -int(origin[0])
    if not 1 <= pad <= 128:
        raise ValueError("RAW ring pad must be in 1..128 native pixels")
    height, width = v5.NATIVE
    if source.shape != (3, height+2*pad, width+2*pad):
        raise ValueError("RAW ring source shape does not match origin/native shape")
    safe = np.asarray(facies_padded)
    if safe.shape != source.shape[1:]:
        raise ValueError("RAW ring needs a same-shape bound optical-support mask")
    if fit_ref.fac.shape != v5.GRID or reference.fac.shape != v5.GRID:
        raise ValueError("V4 fits must use the frozen AC28 reference grid")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    h = rg.validate_homography(h_ref_to_run)
    light = None
    if light_field_ref8 is not None:
        light = np.asarray(light_field_ref8, np.float32)
        if light.shape != v5.GRID:
            raise ValueError("Per-frame light field must use frozen AC28 grid")
    core = rg.apply_refgeom_v5(
        source[:, pad:pad+height, pad:pad+width], fit_ref, reference, h,
        facies_run_native=safe[pad:pad+height, pad:pad+width],
        light_field_ref8=light, tile_rows=tile_rows,
    )
    output = np.empty_like(source)
    output[:, pad:pad+height, pad:pad+width] = core
    field_ref, _ = v5.baseline_field(fit_ref, reference)
    rectangles = (
        (0, width+2*pad, 0, pad),
        (0, width+2*pad, pad+height, height+2*pad),
        (0, pad, pad, pad+height),
        (pad+width, width+2*pad, pad, pad+height),
    )
    for x0, x1, y0, y1 in rectangles:
        output[:, y0:y1, x0:x1] = _correct_ring_rectangle(
            source, safe, fit_ref, reference, h, field_ref, light,
            pad, x0, x1, y0, y1, tile_rows,
        )
    return output


__all__ = ["apply_refgeom_v5_with_raw_ring"]
