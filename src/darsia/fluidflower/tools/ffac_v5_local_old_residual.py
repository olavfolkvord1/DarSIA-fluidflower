"""Preregistered source-only V4-R4 old-light residual field.

This optical B-only scalar field never reads phase labels or predictions.  A
finite field is not a claim that a pixel is sand or eligible for training.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ffac_v5_photometry as v5


SIGMA_REFERENCE_CELLS = 8.0
MIN_SMOOTH_SUPPORT = 0.05


def residual_clean_mask(fit: v5.Baseline, consensus: np.ndarray) -> np.ndarray:
    """Reproduce the fitted B-only RESCOL=.08 mask exactly."""
    keep = fit.sand & v5._same_fac(fit.fac)
    colour = v5._colour(fit.logB, fit.Minv)
    signal = colour - (fit.F if fit.F is not None else 0)
    delta = (signal[2] - signal[1]) - (consensus[2] - consensus[1])
    finite = keep & np.isfinite(delta)
    num = cv2.GaussianBlur(np.where(finite, delta, 0).astype(np.float32),
                           (0, 0), 3)
    den = cv2.GaussianBlur(finite.astype(np.float32), (0, 0), 3)
    delta = np.where(finite, num / np.maximum(den, 1e-6), np.nan)
    centred = keep & np.isfinite(delta)
    if centred.sum() > 100:
        delta[fit.fac == 1] -= np.median(delta[centred])
    clean = keep & np.isfinite(delta) & (delta > -.08)
    fraction = float(1 - clean.sum() / max(keep.sum(), 1))
    if abs(fraction - fit.residual_exclusion_fraction) > 1e-6:
        raise ValueError(f"{fit.run}: source B RESCOL mask differs from fit")
    return clean


def estimate_old_B_field(
    fit: v5.Baseline,
    reference: v5.Baseline,
    consensus: np.ndarray,
    corrected_B: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Estimate one fixed-scale achromatic residual from pre-injection B only."""
    shape = fit.sand.shape
    if (reference.sand.shape != shape or consensus.shape != (3, *shape)
            or corrected_B.shape != (3, *shape)):
        raise ValueError("Old-light B and virtual consensus geometry differ")
    clean = residual_clean_mask(fit, consensus)
    support = (clean & reference.sand & fit.sand & ~fit.chart &
               np.isfinite(consensus[1]) & np.isfinite(corrected_B[1]))
    if int(support.sum()) < 1000:
        raise ValueError(f"{fit.run}: insufficient source-clean old-light sand")
    residual = consensus[1] - corrected_B[1]
    den = cv2.GaussianBlur(support.astype(np.float32), (0, 0),
                           SIGMA_REFERENCE_CELLS)
    num = cv2.GaussianBlur(np.where(support, residual, 0).astype(np.float32),
                           (0, 0), SIGMA_REFERENCE_CELLS)
    field = num / np.maximum(den, 1e-6)
    field[den < MIN_SMOOTH_SUPPORT] = 0
    field *= fit.taper * (~fit.chart)
    if not np.all(np.isfinite(field)):
        raise ArithmeticError("Non-finite old-light B residual field")
    return field.astype(np.float32), {
        "method": "source_B_G_residual_normalized_gaussian_scalar_all_RGB",
        "sigma_reference_cells": SIGMA_REFERENCE_CELLS,
        "minimum_smooth_support": MIN_SMOOTH_SUPPORT,
        "source_clean_sand_cells": int(support.sum()),
        "fitted_rescol_exclusion_fraction": fit.residual_exclusion_fraction,
        "chart_cells_zero": bool(np.all(field[fit.chart] == 0)),
        "same_field_all_channels": True,
    }


def apply_old_field(corrected: np.ndarray, field: np.ndarray) -> np.ndarray:
    rgb = np.asarray(corrected)
    light = np.asarray(field)
    if rgb.dtype != np.float32 or light.dtype != np.float32:
        raise TypeError("V4-R4 requires float32 log-linear RGB and scalar field")
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.shape[1:] != light.shape:
        raise ValueError("Old-light field geometry differs from RGB")
    return (rgb + light[None]).astype(np.float32)
