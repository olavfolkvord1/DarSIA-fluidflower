"""Reuse the exact frozen regional image context across conditional gas heads.

Only the two ``gas_joint_mean`` columns depend on conditional gas. The other
eighteen columns, quiet-pixel drift, thresholds and geometry are prepared once
by the authoritative regional implementation. No fitted weights or settings
are changed, and callers must keep the supplied presence/domain arrays fixed
for the lifetime of this frame-local context.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import phase_optical_regions as regional


CONTRACT = "ffac-frozen-regional-gas-shared-context-v1"
GAS_DEPENDENT_COLUMNS = (1, 11)


@dataclass(frozen=True)
class RegionalGasContext:
    prepared: dict[str, Any]
    presence: np.ndarray
    native_domain: np.ndarray
    coverage: np.ndarray
    scales: tuple[tuple[float, float, np.ndarray], ...]


def shared_context_contract() -> dict:
    return dict(contract=CONTRACT,
                code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_regional_code_sha256=sha256(Path(regional.__file__).read_bytes()).hexdigest(),
                gas_dependent_columns=list(GAS_DEPENDENT_COLUMNS),
                gas_dependent_feature_names=[regional.FEATURE_NAMES[index]
                                             for index in GAS_DEPENDENT_COLUMNS],
                numerical_policy="same-float32-pooling-and-weighted-Gaussian-operator-order",
                fitting_performed=False)


def build_shared_regional_context(current, baseline, presence, domain, spacing_m,
                                  settings=regional.Settings()) -> RegionalGasContext:
    """Run unchanged regional preparation once and cache its two blur weights."""
    p = np.asarray(presence)
    valid = np.asarray(domain, bool)
    prepared = regional.prepare(current, baseline, p, np.zeros_like(p),
                                valid, spacing_m, settings)
    shape = prepared["domain"].shape
    coverage = cv2.resize(np.asarray(valid, np.float32), shape[::-1],
                           interpolation=cv2.INTER_AREA)
    actual_spacing = np.asarray(prepared["spacing_m"], float)
    scales = []
    for sigma in (settings.fine_sigma_m, settings.coarse_sigma_m):
        sy, sx = sigma / actual_spacing
        weight = cv2.GaussianBlur(coverage, (0, 0), float(sx), sigmaY=float(sy))
        scales.append((float(sx), float(sy), np.maximum(weight, 1e-6)))
    return RegionalGasContext(prepared, p, valid, coverage, tuple(scales))


def prepare_conditional_gas(context: RegionalGasContext, conditional_gas) -> dict:
    """Recompute exactly the two gas-dependent fields on this frozen context."""
    gas = np.asarray(conditional_gas)
    if (gas.shape != context.native_domain.shape or not np.isfinite(gas).all()
            or (gas < 0).any() or (gas > 1).any()):
        raise ValueError("Conditional gas must match the fixed native probability grid")
    coverage = context.coverage
    shape = coverage.shape
    # Match regional.prepare: multiply the original p/g dtypes first, then
    # convert that joint field to float32 before native-domain masking/pooling.
    reduced_joint = np.divide(
        cv2.resize(np.asarray(context.presence * gas, np.float32) * context.native_domain,
                   shape[::-1], interpolation=cv2.INTER_AREA),
        coverage, out=np.zeros(shape, np.float32), where=coverage > 1e-6)
    weighted_joint = reduced_joint * coverage
    features = context.prepared["features"].copy()
    for column, (sigma_x, sigma_y, denominator) in zip(GAS_DEPENDENT_COLUMNS, context.scales):
        features[..., column] = cv2.GaussianBlur(
            weighted_joint, (0, 0), sigma_x, sigmaY=sigma_y) / denominator
    return dict(context.prepared, features=features)


__all__ = ["CONTRACT", "GAS_DEPENDENT_COLUMNS", "RegionalGasContext",
           "shared_context_contract", "build_shared_regional_context",
           "prepare_conditional_gas"]
