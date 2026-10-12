"""Signal-backed completion of submillimetre aqueous boundary gaps.

This opt-in extension uses the frozen reference gate and independently learned
aqueous colour direction. It never estimates evidence from a binary phase mask.
The v1 implementation and its runtime identity remain unchanged.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_aqueous_boundary as v1
from . import phase_native_reference as native
from . import phase_reference_support as reference
from . import phase_reviewed_reference as reviewed

VERSION = "native-aqueous-boundary-v2"


@dataclass(frozen=True)
class AqueousBoundarySettings(v1.AqueousBoundarySettings):
    """Retain v1 physical scales; reference additions need a calibrated gate."""


def prune_uncertain_islands(predictions, raw_co2, context, *, protected_reference=None,
                            settings=None):
    """Reject only whole, submillimetre, weak and disconnected aqueous islands.

    A long narrow finger is retained regardless of its area. Any native optical
    positive, gas pixel, or calibrated signed-reference positive protects the
    entire component. This cannot erode the outer edge of a retained plume.
    """
    settings = settings or AqueousBoundarySettings()
    domain = np.asarray(context.domain, bool)
    co2 = np.asarray(predictions['co2_presence_probability'])
    gas = np.asarray(predictions['gas_probability_conditional_on_co2'])
    raw = np.asarray(raw_co2)
    protected = (np.zeros(domain.shape, bool) if protected_reference is None
                 else np.asarray(protected_reference))
    if any(a.shape != domain.shape for a in (co2, gas, raw, protected)) or protected.dtype != bool:
        raise ValueError('Island inputs must share a native grid and boolean reference protection')
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError('Island optical scores must be finite in [0, 1]')
    dx = v1._spacing(context.x_m, domain.shape[1])
    dy = v1._spacing(context.y_m, domain.shape[0], descending=True)
    original = domain & (co2 >= .5)
    # Eight-neighbour components conservatively retain diagonally sampled thin
    # fingers. This groups existing labels for protection; it creates no path.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(original.astype(np.uint8), connectivity=8)
    safe = original & ((gas >= .6) | (raw >= .5) | (co2 > .500001) | protected)
    anchored = np.bincount(labels[safe], minlength=count) > 0
    diameters = np.hypot(stats[:, cv2.CC_STAT_WIDTH]*dx, stats[:, cv2.CC_STAT_HEIGHT]*dy)
    rejected = (diameters <= settings.maximum_band_m) & ~anchored
    rejected[0] = False
    removed = rejected[labels] & original
    result = dict(predictions)
    adjusted = co2.copy()
    adjusted[removed] = np.minimum(raw[removed], .49)
    result['co2_presence_probability'] = adjusted
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
            ("aqueous_co2_probability", co2*(1-gas), adjusted*(1-gas)),
            ("free_co2_gas_probability", co2*gas, adjusted*gas)):
        result[key] = np.array(predictions.get(key, old), copy=True)
        result[key][removed] = new[removed]
    if not np.array_equal((adjusted >= .5) & (gas >= .6), (co2 >= .5) & (gas >= .6)):
        raise AssertionError('Island cleanup changed gas labels')
    return result, dict(removed_aqueous_pixels=int(removed.sum()),
        removed_uncertain_islands=int(rejected.sum()), maximum_component_diameter_m=settings.maximum_band_m,
        protected_native_optical_threshold=.5, existing_connected_plume_edges_changed=False,
        gas_labels_unchanged=True, history_updates=False)


def reference_evidence(baseline, current, context, gas, controls, reference_settings,
                       calibration, settings=None):
    """Return continuous signed evidence and a separately calibrated gate.

    Averaging is confined to each connected facies and explicitly excludes gas.
    Destinations within three smoothing sigmas of gas are ineligible, so nearby
    gas cannot supply the aqueous colour signal. Individual unexposed pixels
    remain ineligible. Missing calibration or directions fail closed.
    """
    settings = settings or AqueousBoundarySettings()
    shape = context.domain.shape
    domain, facies = np.asarray(context.domain, bool), np.asarray(context.facies)
    gas = np.asarray(gas, bool)
    if gas.shape != shape or facies.shape != shape:
        raise ValueError("Reference boundary fields must share the native grid")
    score = np.zeros(shape, np.float32)
    eligible = np.zeros(shape, bool)
    if not calibration or not calibration.get("enabled"):
        return score, eligible
    b, c = (reference.rgb_float(value, shape) for value in (baseline, current))
    dx = v1._spacing(context.x_m, shape[1])
    dy = v1._spacing(context.y_m, shape[0], descending=True)
    sx, sy = settings.sigma_m / dx, settings.sigma_m / dy
    brightness = b.mean(-1)
    ratio = c.mean(-1) / np.maximum(brightness, 1e-6)
    exposed = (domain & ~gas & (brightness >= reference_settings.minimum_reference_brightness)
        & (ratio >= reference_settings.minimum_brightness_ratio)
        & (ratio <= reference_settings.maximum_brightness_ratio))
    safe = exposed.copy()
    if gas.any():
        safe &= distance_transform_edt(~gas, sampling=(dy, dx)) > 3 * settings.sigma_m
    bchroma = b / np.maximum(b.sum(-1, keepdims=True), 1e-6)
    cchroma = c / np.maximum(c.sum(-1, keepdims=True), 1e-6)
    for control in controls:
        if not control.get("available"):
            continue
        area = exposed & (facies == control["facies"])
        if not area.any():
            continue
        direction = np.asarray(control["aqueous_direction"], np.float32)
        shift = np.asarray(control["common_shift"], np.float32)
        threshold = float(control["chroma_threshold"])
        if (direction.shape != (3,) or shift.shape != (3,)
                or not np.isfinite(direction).all() or not np.isfinite(shift).all()
                or not np.isfinite(threshold) or threshold <= 0):
            raise ValueError("Invalid independently learned reference direction")
        rows, columns = np.where(area)
        region = np.s_[rows.min():rows.max()+1, columns.min():columns.max()+1]
        residual = cchroma[region] - bchroma[region] - shift
        averaged = np.stack([v1._material_average(residual[..., channel], area[region],
            facies[region], sx, sy) for channel in range(3)], axis=-1)
        projection = averaged @ direction
        cosine = projection / np.maximum(np.linalg.norm(averaged, axis=-1), 1e-8)
        selected = safe[region] & area[region] & (cosine >= reference_settings.direction_cosine_minimum)
        score[region][selected] = np.maximum(projection[selected] / threshold, 0)
        eligible[region] |= selected
    # Preserve the fitted acceptance test. Interpolate its continuous score
    # within native facies, rather than expanding a coarse binary acceptance.
    features, available = reviewed.change_fields(baseline, current)
    available &= features[..., 3] > 1e-7
    gate_score = native._facies_linear(reviewed.scores(features, calibration),
                                      available, facies)
    gate = gate_score >= calibration["threshold"]
    for label, threshold in calibration.get("facies_thresholds", {}).items():
        selected = facies == int(label)
        gate[selected] = gate_score[selected] >= threshold
    return score, eligible & gate


def complete_reference_gaps(predictions, context, score, eligible, independent_support,
                            settings=None):
    """Add only evidence-backed aqueous within 1 mm of an existing component.

    Independent support must come from observed causal history, never yesterday's
    refinement. This keeps inherited narrow fingers and every gas label exact.
    It cannot recover a missing plume farther away or remove frayed positives.
    """
    settings = settings or AqueousBoundarySettings()
    domain = np.asarray(context.domain, bool)
    score, eligible = np.asarray(score), np.asarray(eligible)
    support = np.asarray(independent_support)
    if score.shape != domain.shape or eligible.shape != domain.shape or support.shape != domain.shape:
        raise ValueError("Reference completion inputs must share the native grid")
    if eligible.dtype != bool or support.dtype != bool:
        raise ValueError("Reference eligibility and support must be certified boolean masks")
    if not np.isfinite(score).all() or (score < 0).any():
        raise ValueError("Reference score must be finite and nonnegative")
    co2 = np.asarray(predictions["co2_presence_probability"])
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"])
    original = domain & (co2 >= .5)
    report = dict(format=VERSION, reference_added_aqueous_pixels=0,
        inherited_labels_removed=0, gas_labels_unchanged=True, history_updates=False,
        calibrated_probability=False, settings=asdict(settings))
    if not original.any():
        return dict(predictions), report
    dx = v1._spacing(context.x_m, domain.shape[1])
    dy = v1._spacing(context.y_m, domain.shape[0], descending=True)
    band = domain & ~original & (distance_transform_edt(~original,
        sampling=(dy, dx)) <= settings.maximum_band_m)
    proposed = band & eligible & support & (score >= 1.)
    added = v1._connected_to_retained(proposed, original, domain)
    # Distinct original objects must not be joined by this local correction.
    # Voronoi watershed ridges retain at least one untouched-water separation.
    count, labels = cv2.connectedComponents(original.astype(np.uint8), connectivity=4)
    if count > 2 and added.any():
        nearest = distance_transform_edt(~original, return_distances=False, return_indices=True)
        owner = labels[tuple(nearest)]
        ridge = np.zeros(domain.shape, bool)
        difference = owner[1:] != owner[:-1]
        ridge[1:] |= difference
        ridge[:-1] |= difference
        difference = owner[:, 1:] != owner[:, :-1]
        ridge[:, 1:] |= difference
        ridge[:, :-1] |= difference
        added &= ~ridge
        added = v1._connected_to_retained(added, original, domain)
    adjusted, conditional = co2.copy(), gas.copy()
    adjusted[added] = .5
    conditional[added] = np.minimum(conditional[added], .59)
    result = dict(predictions)
    result["co2_presence_probability"] = adjusted
    result["gas_probability_conditional_on_co2"] = conditional
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
            ("aqueous_co2_probability", co2*(1-gas), adjusted*(1-conditional)),
            ("free_co2_gas_probability", co2*gas, adjusted*conditional)):
        result[key] = np.array(predictions.get(key, old), copy=True)
        result[key][added] = new[added]
    if not np.array_equal((adjusted >= .5) & (conditional >= .6), (co2 >= .5) & (gas >= .6)):
        raise AssertionError("Reference boundary completion changed gas labels")
    report["reference_added_aqueous_pixels"] = int(added.sum())
    return result, report
