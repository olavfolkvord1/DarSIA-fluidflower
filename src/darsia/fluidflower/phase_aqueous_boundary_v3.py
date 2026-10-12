"""Opt-in evidence-constrained refinement of grain-scale aqueous contours.

This is a classification regularizer, not a claim that shape supplies chemical
evidence. It may trim a locally sparse edge only where averaged current optical
evidence remains below the classifier threshold. Long thin fingers, bridges,
gas and the interior of the plume survive. Gap completion additionally requires
current regional optical evidence and independently observed causal support.
Legacy v1 and its runtime identity are unchanged.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_aqueous_boundary as v1

VERSION = "native-aqueous-boundary-v3"


@dataclass(frozen=True)
class AqueousBoundarySettings(v1.AqueousBoundarySettings):
    fringe_sigma_m: float = .0005
    refinement_band_m: float = .001
    maximum_island_diameter_m: float = .001
    minimum_finger_length_m: float = .001
    maximum_fringe_occupancy: float = .48
    fringe_optical_threshold: float = .5
    strong_optical_threshold: float = .65
    gap_sigma_m: float = .001
    minimum_gap_occupancy: float = .6
    enable_gap_completion: bool = True

    def __post_init__(self):
        if type(self.enable_gap_completion) is not bool:
            raise ValueError("Gap completion must be selected with an explicit boolean")
        super().__post_init__()
        if not 0 < self.fringe_sigma_m <= self.maximum_island_diameter_m <= self.refinement_band_m <= .002:
            raise ValueError("V3 fringe scales must be positive and bounded to two millimetres")
        if not 0 < self.minimum_finger_length_m <= self.refinement_band_m:
            raise ValueError("Finger protection must use a positive bounded physical length")
        if not 0 < self.maximum_fringe_occupancy < .5:
            raise ValueError("Fringe occupancy must be below a local majority")
        if not 0 < self.fringe_optical_threshold <= .5:
            raise ValueError("Only optically uncertain fringe may be removed")
        if not .5 <= self.strong_optical_threshold <= 1:
            raise ValueError("Isolated evidence protection requires a strong optical score")
        if not self.fringe_sigma_m <= self.gap_sigma_m <= self.refinement_band_m:
            raise ValueError("Gap optical averaging must remain within the boundary scale")
        if not .5 < self.minimum_gap_occupancy <= 1:
            raise ValueError("Gap completion requires an existing local plume majority")


def legacy_settings(settings):
    return v1.AqueousBoundarySettings(**{key: asdict(settings)[key]
        for key in v1.AqueousBoundarySettings.__dataclass_fields__})


def trim_fringe(predictions, raw_co2, context, settings=None):
    """Remove sparse, optically uncertain pixels in an explicit v3 edge band.

    Long medial ridges protect narrow oblique fingers. A proposed removal
    connecting separate retained components is also protected. Only detached
    islands below the explicit physical diameter may be removed in entirety.
    No learned signal, calibration, or temporal history changes.
    """
    settings = settings or AqueousBoundarySettings()
    domain, facies = np.asarray(context.domain, bool), np.asarray(context.facies)
    co2 = np.asarray(predictions["co2_presence_probability"])
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"])
    raw = np.asarray(raw_co2)
    if domain.ndim != 2 or any(value.shape != domain.shape for value in (co2, gas, raw, facies)):
        raise ValueError("Fringe inputs must share the registered native grid")
    if any(not np.isfinite(value).all() or (value < 0).any() or (value > 1).any()
           for value in (co2, gas, raw)):
        raise ValueError("Fringe optical scores must be finite in [0, 1]")
    dx = v1._spacing(context.x_m, domain.shape[1])
    dy = v1._spacing(context.y_m, domain.shape[0], descending=True)
    original = domain & (co2 >= .5)
    frozen_gas = original & (gas >= .6)
    report = dict(format=VERSION, removed_aqueous_pixels=0,
        proposed_fringe_pixels=0, protected_long_fringe_pixels=0,
        protected_bridge_pixels=0, protected_isolated_pixels=0, added_aqueous_pixels=0,
        gas_labels_unchanged=True, history_updates=False,
        basis="physically bounded sparse fringe with current averaged optical score below .5")
    if not original.any():
        return dict(predictions), report
    # Both estimators stay within connected native facies. Occupancy supplies a
    # local regularity prior; current optical evidence independently vetoes it.
    occupancy = v1._material_average(original.astype(np.float32), domain, facies,
        settings.fringe_sigma_m/dx, settings.fringe_sigma_m/dy)
    optical = v1._material_average(raw, domain, facies, settings.fringe_sigma_m/dx, settings.fringe_sigma_m/dy)
    proposed = original & ~frozen_gas & (occupancy < settings.maximum_fringe_occupancy)
    proposed &= (optical < settings.fringe_optical_threshold) & (raw < settings.strong_optical_threshold)
    distance = distance_transform_edt(original, sampling=(dy, dx))
    proposed &= distance <= settings.refinement_band_m
    # A long connected fringe may run around an entire ragged outer contour.
    # Protect long *medial ridges* of narrow fingers, not that whole fringe.
    # A boundary beside a thicker plume has greater distance just inward and
    # therefore cannot masquerade as a thin-finger centreline.
    ridge = proposed & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))-1e-12)
    ridge_count, ridge_labels, ridge_stats, _ = cv2.connectedComponentsWithStats(ridge.astype(np.uint8), connectivity=8)
    protected_fingers = np.zeros_like(original)
    for component in range(1, ridge_count):
        x, y, width, height, _ = ridge_stats[component]
        if np.hypot(width*dx, height*dy) <= settings.minimum_finger_length_m:
            continue
        local_ridge = ridge_labels[y:y+height, x:x+width] == component
        radius = float(distance[y:y+height, x:x+width][local_ridge].max())
        rx, ry = int(np.ceil(radius/dx)), int(np.ceil(radius/dy))
        x0, y0 = max(0, x-rx), max(0, y-ry)
        region = np.s_[y0:min(domain.shape[0], y+height+ry), x0:min(domain.shape[1], x+width+rx)]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rx+1, 2*ry+1))
        expanded = cv2.dilate((ridge_labels[region] == component).astype(np.uint8), kernel).astype(bool)
        protected_fingers[region] |= expanded & original[region]
    report["protected_long_fringe_pixels"] = int((proposed & protected_fingers).sum())
    proposed &= ~protected_fingers
    count, labels, stats, _ = cv2.connectedComponentsWithStats(proposed.astype(np.uint8), connectivity=8)
    diameter = np.hypot(stats[:, cv2.CC_STAT_WIDTH]*dx, stats[:, cv2.CC_STAT_HEIGHT]*dy)
    removable = np.ones(count, bool)
    removable[0] = False
    removed = proposed & removable[labels]
    report["proposed_fringe_pixels"] = int(proposed.sum())
    if removed.any():
        # A compact neck may have low occupancy. Preserve it when cutting it
        # would disconnect retained parts of the same plume (including fingers).
        _, retained_labels = cv2.connectedComponents((original & ~removed).astype(np.uint8), connectivity=8)
        for component in np.flatnonzero(removable):
            x, y, width, height, _ = stats[component]
            x0, y0 = max(0, x-1), max(0, y-1)
            region = np.s_[y0:min(domain.shape[0], y+height+1), x0:min(domain.shape[1], x+width+1)]
            local = labels[region] == component
            border = cv2.dilate(local.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~local
            neighbors = np.unique(retained_labels[region][border])
            touching = np.count_nonzero(neighbors)
            protected_island = touching == 0 and (diameter[component] > settings.maximum_island_diameter_m
                or np.max(raw[region][local]) >= settings.strong_optical_threshold)
            if touching > 1 or protected_island:
                removed[region][local] = False
                report["protected_bridge_pixels" if touching > 1 else "protected_isolated_pixels"] += int(local.sum())
    adjusted = co2.copy()
    adjusted[removed] = np.minimum(optical[removed], .49)
    result = dict(predictions, co2_presence_probability=adjusted)
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
            ("aqueous_co2_probability", co2*(1-gas), adjusted*(1-gas)),
            ("free_co2_gas_probability", co2*gas, adjusted*gas)):
        result[key] = np.array(predictions.get(key, old), copy=True)
        result[key][removed] = new[removed]
    if not np.array_equal((adjusted >= .5) & (gas >= .6), (co2 >= .5) & (gas >= .6)):
        raise AssertionError("Fringe trimming changed gas labels")
    report["removed_aqueous_pixels"] = int(removed.sum())
    return result, report


def complete_gaps(predictions, raw_co2, context, support_mask, settings=None):
    """Complete locally enclosed gaps using observed support and current optics.

    A 1 mm optical estimator reduces grain fluctuations before classification.
    Additions need an existing local plume majority, independently observed
    support and a retained connection. Uniformly negative current optical
    evidence cannot supply additions, regardless of the shape or past history.
    """
    settings = settings or AqueousBoundarySettings()
    if not settings.enable_gap_completion:
        return dict(predictions), dict(enabled=False, added_aqueous_pixels=0,
            gas_labels_unchanged=True, history_updates=False,
            support_basis='disabled by explicit settings; no gap pixels proposed')
    domain = np.asarray(context.domain, bool)
    support = np.asarray(support_mask)
    if support.shape != domain.shape or support.dtype != bool:
        raise ValueError('Gap completion requires independently certified boolean support')
    co2 = np.asarray(predictions['co2_presence_probability'])
    gas = np.asarray(predictions['gas_probability_conditional_on_co2'])
    raw = np.asarray(raw_co2)
    if any(v.shape != domain.shape for v in (co2, gas, raw)):
        raise ValueError('Gap fields must share the registered native grid')
    if any(not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in (co2, gas, raw)):
        raise ValueError('Gap optical scores must be finite in [0, 1]')
    original = domain & (co2 >= .5)
    detail = dict(enabled=True, added_aqueous_pixels=0, gas_labels_unchanged=True, history_updates=False,
        support_basis='current regional optical evidence, independent support and existing local plume majority')
    if not original.any() or not support.any():
        return dict(predictions), detail
    dx = v1._spacing(context.x_m, domain.shape[1])
    dy = v1._spacing(context.y_m, domain.shape[0], descending=True)
    occupancy = v1._material_average(original.astype(np.float32), domain, context.facies,
        settings.fringe_sigma_m/dx, settings.fringe_sigma_m/dy)
    optical = v1._material_average(raw, domain, context.facies,
        settings.gap_sigma_m/dx, settings.gap_sigma_m/dy)
    candidate = domain & ~original & support & (occupancy >= settings.minimum_gap_occupancy)
    candidate &= (optical >= settings.threshold) & (distance_transform_edt(~original,
        sampling=(dy, dx)) <= settings.refinement_band_m)
    added = v1._connected_to_retained(candidate, original, domain)
    adjusted, conditional = co2.copy(), gas.copy()
    adjusted[added] = np.maximum(.5, optical[added])
    conditional[added] = np.minimum(conditional[added], .59)
    result = dict(predictions, co2_presence_probability=adjusted,
        gas_probability_conditional_on_co2=conditional)
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
            ("aqueous_co2_probability", co2*(1-gas), adjusted*(1-conditional)),
            ("free_co2_gas_probability", co2*gas, adjusted*conditional)):
        result[key] = np.array(predictions.get(key, old), copy=True)
        result[key][added] = new[added]
    if not np.array_equal((adjusted >= .5) & (conditional >= .6), (co2 >= .5) & (gas >= .6)):
        raise AssertionError('Gap completion changed gas labels')
    detail['added_aqueous_pixels'] = int(added.sum())
    return result, detail


def refine(predictions, raw_co2, context, support_mask, settings=None):
    settings = settings or AqueousBoundarySettings()
    inherited, report = v1.refine(predictions, raw_co2, context, support_mask, legacy_settings(settings))
    result, fringe = trim_fringe(inherited, raw_co2, context, settings)
    result, gaps = complete_gaps(result, raw_co2, context, support_mask, settings)
    before = np.asarray(predictions["co2_presence_probability"]) >= .5
    after = result["co2_presence_probability"] >= .5
    report.update(format=VERSION, settings=asdict(settings), fringe_cleanup=fringe, gap_completion=gaps,
        added_aqueous_pixels=int((after & ~before & context.domain).sum()),
        removed_aqueous_pixels=int((before & ~after & context.domain).sum()))
    weak = before & context.domain & (np.asarray(raw_co2) < .5)
    report['inherited_weak_promotions_before'] = int(weak.sum())
    report['inherited_weak_promotions_protected'] = int((weak & after).sum())
    return result, report
