"""Bounded, opt-in native aqueous-interface refinement.

The input is continuous optical evidence, not a binary-mask blur. Physical
scales are explicit development settings, not inferred grain sizes. Additions
require independently certified support and a retained plume connection. This
stateless helper never creates gas or changes the inherited temporal history.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


VERSION = "native-aqueous-boundary-v1"


@dataclass(frozen=True)
class AqueousBoundarySettings:
    sigma_m: float = .0005
    maximum_band_m: float = .001
    threshold: float = .38

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Aqueous-boundary settings must be finite")
        if not 0 < self.sigma_m <= self.maximum_band_m:
            raise ValueError("Aqueous smoothing scale must be positive and within the boundary band")
        if self.maximum_band_m > .001:
            raise ValueError("This aqueous-boundary version permits at most a one-millimetre band")
        if not 0 < self.threshold < .5:
            raise ValueError("Aqueous boundary threshold must be between zero and the optical threshold")


def _spacing(axis, length, *, descending=False):
    axis = np.asarray(axis, np.float64)
    if axis.shape != (length,) or length < 2 or not np.isfinite(axis).all():
        raise ValueError("Aqueous boundary requires finite registered physical axes")
    steps = np.diff(axis) * (-1 if descending else 1)
    if (steps <= 0).any() or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-12):
        raise ValueError("Aqueous boundary requires an evenly spaced physical grid")
    return float(steps[0])


def _material_average(values, domain, facies, sigma_x, sigma_y):
    """Normalize optical evidence separately within each connected material.

    A disconnected piece of the same facies cannot import evidence across an
    excluded strip. Material identifiers and excluded pixels are never blurred.
    """
    result = np.zeros(values.shape, np.float32)
    for material in np.unique(facies[domain]):
        mask = domain & (facies == material)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=4)
        for component in range(1, count):
            x, y, width, height, _ = stats[component]
            region = np.s_[y:y+height, x:x+width]
            local = labels[region] == component
            weights = cv2.GaussianBlur(local.astype(np.float32), (0, 0),
                sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)
            numerator = cv2.GaussianBlur(np.where(local, values[region], 0).astype(np.float32),
                (0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)
            result[region][local] = numerator[local] / np.maximum(weights[local], 1e-7)
    return result


def _connected_to_retained(candidates, retained, domain):
    count, labels = cv2.connectedComponents(
        ((candidates | retained) & domain).astype(np.uint8), connectivity=4)
    anchored = np.zeros(count, bool)
    anchored[np.unique(labels[retained & domain])] = True
    anchored[0] = False
    return candidates & anchored[labels]


def refine(predictions, raw_co2, context, support_mask, settings=None):
    """Refine a native interface and return ``(predictions, diagnostic_report)``.

    ``support_mask`` is a native boolean mask supplied from independent causal
    presence support. It is not inferred from this frame's refined result.
    Existing weak promotions are protected: a raw-optics regularizer must not
    undo a separately supported reference-color decision with raw score < .5.
    Existing optical-positive grain outliers may be removed; additions must
    pass the continuous regional threshold and retain a four-neighbor anchor.
    """
    if settings is None:
        settings = AqueousBoundarySettings()
    elif isinstance(settings, Mapping):
        settings = AqueousBoundarySettings(**settings)
    if not isinstance(settings, AqueousBoundarySettings):
        raise TypeError("Expected aqueous-boundary settings or a settings mapping")
    domain = np.asarray(context.domain, bool)
    facies = np.asarray(context.facies)
    raw = np.asarray(raw_co2, np.float32)
    co2 = np.asarray(predictions["co2_presence_probability"], np.float32)
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"], np.float32)
    support = np.asarray(support_mask)
    if (domain.ndim != 2 or facies.shape != domain.shape or raw.shape != domain.shape
            or co2.shape != domain.shape or gas.shape != domain.shape or support.shape != domain.shape):
        raise ValueError("Aqueous boundary inputs must share the native physical grid")
    if support.dtype != bool:
        raise ValueError("Aqueous boundary support must be an explicitly certified boolean mask")
    if not np.isfinite(facies[domain]).all():
        raise ValueError("Aqueous boundary requires finite categorical facies")
    for values in (raw, co2, gas):
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Aqueous boundary expects finite optical scores in [0, 1]")
    dx = _spacing(context.x_m, domain.shape[1])
    dy = _spacing(context.y_m, domain.shape[0], descending=True)
    original = domain & (co2 >= .5)
    frozen_gas = original & (gas >= .6)
    promoted = original & (raw < .5)
    report = dict(format=VERSION, settings=asdict(settings), band_pixels=0,
        added_aqueous_pixels=0, removed_aqueous_pixels=0,
        inherited_weak_promotions_protected=int(promoted.sum()),
        gas_labels_unchanged=True, history_updates=False,
        calibrated_probability=False,
        support_basis="independent causal presence support and retained four-neighbor plume connection")
    if not original.any():
        return dict(predictions), report

    inside_distance = distance_transform_edt(original, sampling=(dy, dx))
    outside_distance = distance_transform_edt(~original, sampling=(dy, dx))
    band = domain & np.where(original, inside_distance <= settings.maximum_band_m,
                             outside_distance <= settings.maximum_band_m)
    smooth = _material_average(raw, domain, facies, settings.sigma_m/dx, settings.sigma_m/dy)
    supported_evidence = smooth >= settings.threshold
    removed = band & original & ~frozen_gas & ~promoted & ~supported_evidence
    retained = original & ~removed
    proposed = band & ~original & support & supported_evidence
    added = _connected_to_retained(proposed, retained, domain)
    changed = added | removed
    result = dict(predictions)
    adjusted = co2.copy()
    adjusted[added] = np.maximum(.5, smooth[added])
    adjusted[removed] = smooth[removed]
    conditional = gas.copy()
    conditional[added] = np.minimum(conditional[added], .59)
    result["co2_presence_probability"] = adjusted
    result["gas_probability_conditional_on_co2"] = conditional
    old_free = co2*gas
    new_free = adjusted*conditional
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
                          ("aqueous_co2_probability", co2-old_free, adjusted-new_free),
                          ("free_co2_gas_probability", old_free, new_free)):
        value = np.array(predictions.get(key, old), copy=True)
        value[changed] = new[changed]
        result[key] = value
    if not np.array_equal((adjusted >= .5) & (conditional >= .6), (co2 >= .5) & (gas >= .6)):
        raise AssertionError("Aqueous boundary refinement changed gas labels")
    report.update(band_pixels=int(band.sum()), added_aqueous_pixels=int(added.sum()),
                  removed_aqueous_pixels=int(removed.sum()))
    return result, report
