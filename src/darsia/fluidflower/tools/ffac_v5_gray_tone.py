"""Diagnostic B-only, achromatic gray-scale response for FF_AC RAW photometry.

This is a distinct V4-R6 candidate. It is not enabled in source materialization.
Only the pre-injection ColorChecker neutral strip and source-observed B sand
are used. Saturated RAW chart patches are excluded; no yellow/post-injection
colour, phase label, model prediction or facies level enters the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ffac_v5_photometry as v5


NEUTRAL_PATCH_INDICES = (3, 7, 11, 15, 19, 23)
RAW_SATURATION_EXCLUDE = 15100.0
MIN_UNSATURATED_NEUTRALS = 4
MIN_NEUTRAL_LOG_SPAN = 1.5
MAX_ABS_SLOPE = 0.20
MAX_NEUTRAL_FIT_MAD = 0.04


@dataclass(frozen=True)
class GrayTone:
    run: str
    slope: float
    B_sand_camera_log_G: float
    max_unsaturated_gray_camera_log_G: float
    usable_neutral_indices: tuple[int, ...]
    neutral_log_span: float
    neutral_fit_mad: float


def fit_gray_tone(B_fit: v5.Baseline, reference: v5.Baseline) -> GrayTone:
    """Fit one achromatic tone slope on the exact source B chart and B sand."""
    if B_fit.CB.shape != (3, 24) or reference.CB.shape != (3, 24):
        raise ValueError("Expected 24 source-bound B chart patches")
    selected = np.asarray(NEUTRAL_PATCH_INDICES, np.intp)
    source_linear = np.exp(B_fit.CB[:, selected].astype(np.float64))
    reference_linear = np.exp(reference.CB[:, selected].astype(np.float64))
    valid = (np.all(np.isfinite(source_linear), axis=0) &
             np.all(np.isfinite(reference_linear), axis=0) &
             (np.max(source_linear, axis=0) < RAW_SATURATION_EXCLUDE) &
             (np.max(reference_linear, axis=0) < RAW_SATURATION_EXCLUDE))
    selected = selected[valid]
    if selected.size < MIN_UNSATURATED_NEUTRALS:
        raise ValueError("Insufficient unsaturated B neutral chart patches")
    camera_chart = v5._colour(B_fit.CB[:, None, :], B_fit.Minv)[:, 0, :]
    q = np.asarray(camera_chart[1, selected], np.float64)
    target = np.asarray(reference.CB[1, selected] - q, np.float64)
    if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(target)) or
            not np.all(np.diff(q) > 0)):
        raise ValueError("B gray strip not strictly monotone")
    span = float(q[-1] - q[0])
    if span < MIN_NEUTRAL_LOG_SPAN:
        raise ValueError("B gray strip lacks independent intensity span")
    pair_slopes = [(target[j] - target[i]) / (q[j] - q[i])
                   for i in range(len(q)) for j in range(i + 1, len(q))]
    slope = float(np.median(pair_slopes))
    intercept = float(np.median(target - slope * q))
    mad = float(np.median(np.abs(target - (intercept + slope * q))))
    if abs(slope) > MAX_ABS_SLOPE or mad > MAX_NEUTRAL_FIT_MAD:
        raise ValueError("B neutral response exceeds frozen plausibility gate")
    camera_B = v5._colour(B_fit.logB, B_fit.Minv)
    sand = B_fit.sand & reference.sand & np.isfinite(camera_B[1])
    if sand.sum() < 1000:
        raise ValueError("Too little source-observed B sand for tone anchor")
    sand_q = float(np.median(camera_B[1, sand]))
    if not sand_q < float(q[0]):
        raise ValueError("B sand does not precede neutral intensity range")
    return GrayTone(B_fit.run, slope, sand_q, float(q[-1]),
                    tuple(int(i) for i in selected), span, mad)


def apply_gray_tone(corrected_log_rgb: np.ndarray, camera_log_rgb: np.ndarray,
                    tone: GrayTone) -> np.ndarray:
    """Add the B-gray-derived neutral response on the observed brightness span.

    Values below the B sand anchor or above the brightest unsaturated gray
    receive the endpoint correction; no out-of-chart extrapolation is made.
    The finite mask and chroma of the existing correction are preserved.
    """
    corrected = np.asarray(corrected_log_rgb)
    camera = np.asarray(camera_log_rgb)
    if corrected.dtype != np.float32 or camera.dtype != np.float32:
        raise TypeError("Gray-tone candidate requires float32 log-linear RAW")
    if corrected.shape != camera.shape or corrected.ndim != 3 or corrected.shape[0] != 3:
        raise ValueError("Expected matching channel-first RGB arrays")
    q = np.clip(camera[1], tone.B_sand_camera_log_G,
                tone.max_unsaturated_gray_camera_log_G)
    scalar = tone.slope * (q - tone.B_sand_camera_log_G)
    result = corrected + scalar[None].astype(np.float32)
    result[:, ~np.all(np.isfinite(corrected), axis=0)] = np.nan
    return result.astype(np.float32)
