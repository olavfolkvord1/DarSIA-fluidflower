"""Project a frozen B-only achromatic residual to source-native RAW pixels.

The V4-R5 old-light residual was estimated on a 1/8 AC28 reference grid.
This operation only projects that immutable scalar field; it never estimates
new coefficients, changes source coordinates, or promotes unsupported pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .ffac_v5_observed_sand import ReferenceGrid


def apply_old_field_native(
    corrected_native_log_rgb: np.ndarray,
    field_ref8: np.ndarray,
    grid: ReferenceGrid,
    h_ref_to_run: np.ndarray,
    *,
    destination_origin_xy: tuple[int, int] = (0, 0),
    tile_rows: int = 128,
) -> np.ndarray:
    """Add the fixed scalar field on the same native coordinates as V4-R5.

    Input is the already corrected float32 log-linear RAW output of
    ``ffac_v5_observed_sand.apply_native``. NaNs and reference-field bounds
    remain invalid. The same addition is made to R, G and B.
    """
    rgb = np.asarray(corrected_native_log_rgb)
    field = np.asarray(field_ref8)
    if rgb.dtype != np.float32 or field.dtype != np.float32:
        raise TypeError("Old-light native field requires float32 log-linear RAW")
    if rgb.ndim != 3 or rgb.shape[0] != 3 or field.shape != grid.shape_hw:
        raise ValueError("Native RGB or reference residual field shape mismatch")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    H = np.asarray(h_ref_to_run, np.float64)
    if (H.shape != (3, 3) or not np.all(np.isfinite(H)) or
            abs(float(np.linalg.det(H))) < 1e-10):
        raise ValueError("Expected invertible reference-to-run native homography")
    inverse = np.linalg.inv(H)
    out = rgb.copy()
    height, width = rgb.shape[1:]
    ox, oy = destination_origin_xy
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        yy, xx = np.mgrid[start:stop, :width]
        xrun = xx.astype(np.float64) + ox
        yrun = yy.astype(np.float64) + oy
        den = inverse[2, 0] * xrun + inverse[2, 1] * yrun + inverse[2, 2]
        if np.any(np.abs(den) < 1e-9):
            raise ValueError("Native pixel projects to reference infinity")
        xref = (inverse[0, 0] * xrun + inverse[0, 1] * yrun + inverse[0, 2]) / den
        yref = (inverse[1, 0] * xrun + inverse[1, 1] * yrun + inverse[1, 2]) / den
        mx = ((xref - 3.5) / 8 - grid.origin_cell_xy[0]).astype(np.float32)
        my = ((yref - 3.5) / 8 - grid.origin_cell_xy[1]).astype(np.float32)
        in_field = ((mx >= -.5) & (mx <= grid.shape_hw[1] - .5) &
                    (my >= -.5) & (my <= grid.shape_hw[0] - .5))
        mapped = cv2.remap(field, mx, my, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        valid = (in_field & np.isfinite(mapped) &
                 np.all(np.isfinite(rgb[:, start:stop]), axis=0))
        tile = out[:, start:stop]
        tile += mapped[None]
        tile[:, ~valid] = np.nan
    return out
