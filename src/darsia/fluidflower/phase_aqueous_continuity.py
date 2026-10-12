"""Opt-in, current-signal-backed aqueous continuity over causal observations.

History lowers the optical hurdle; it never supplies today's colour evidence.
The inherited sequence alone owns observations and gas classifications. These
development settings describe image support, not a concentration/flow model.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_aqueous_boundary as boundary
from . import phase_aqueous_boundary_runtime as legacy
from . import phase_reference_support as reference
from .project import sha256_file

VERSION = "signal-backed-aqueous-continuity-v3"


@dataclass(frozen=True)
class ContinuitySettings:
    sigma_m: float = .0012
    maximum_recovery_m: float = .02
    cleanup_band_m: float = .003
    history_optical_minimum: float = .18
    connected_optical_minimum: float = .32
    cleanup_optical_maximum: float = .28
    history_signal_minimum: float = 1.
    connected_signal_minimum: float = 1.5

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Continuity settings must be positive and finite")
        if not self.sigma_m <= self.cleanup_band_m <= self.maximum_recovery_m <= .03:
            raise ValueError("Continuity scales must be ordered and bounded to 30 mm")
        if not self.history_optical_minimum < self.cleanup_optical_maximum < self.connected_optical_minimum < .5:
            raise ValueError("Continuity optical thresholds must be ordered below .5")
        if not 1 <= self.history_signal_minimum <= self.connected_signal_minimum:
            raise ValueError("Every addition requires current colour above measured noise")


def split_profile(profile):
    specification = profile.get("boundary_refinement", {})
    if specification.get("format") != VERSION or set(specification) - {"format", "settings"}:
        raise ValueError("Unknown aqueous continuity specification")
    # Validate the inherited C1--C5 contract using the unchanged legacy parser.
    inherited, _ = legacy.split_profile({**profile, "boundary_refinement": {
        "format": legacy.VERSION, "settings": {}}})
    return inherited, ContinuitySettings(**specification.get("settings", {}))


def runtime_identity(inherited_identity, settings):
    payload = dict(format=VERSION, inherited_runtime=inherited_identity, settings=asdict(settings))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for filename in (__file__, boundary.__file__, legacy.__file__):
        digest.update(sha256_file(Path(filename)).encode())
    return digest.hexdigest()


def unchanged_control_mask(context, predictions, raw_co2, trusted_history, reference_settings):
    """Optical negatives well outside both accepted plume and earlier exposure."""
    occupied = ((np.asarray(predictions["co2_presence_probability"]) >= .5)
        | np.asarray(trusted_history, bool)) & context.domain
    dx = boundary._spacing(context.x_m, context.domain.shape[1])
    dy = boundary._spacing(context.y_m, context.domain.shape[0], descending=True)
    far = distance_transform_edt(~occupied, sampling=(dy, dx)) > reference_settings.maximum_front_distance_m
    return context.domain & far & (np.asarray(raw_co2) < .10)


def _average_channels(values, mask, sigma_x, sigma_y):
    """Share material/component weights for all RGB channels and both photos."""
    result = np.zeros_like(values, dtype=np.float32)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    for component in range(1, count):
        x, y, width, height, _ = stats[component]
        region = np.s_[y:y+height, x:x+width]
        local = labels[region] == component
        weights = cv2.GaussianBlur(local.astype(np.float32), (0, 0), sigmaX=sigma_x,
            sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)
        numerator = cv2.GaussianBlur(np.where(local[..., None], values[region], 0), (0, 0),
            sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_CONSTANT)
        result[region][local] = numerator[local]/np.maximum(weights[local, None], 1e-7)
    return result


def reference_signal(baseline, current, context, gas_mask, controls, reference_settings, settings,
                     *, control_mask=None, diagnostics=None):
    """Signed chromatic change, with per-facies drift and noise controls.

    All colour directions come from the inherited earlier strong observations.
    Gaussian averaging is applied to RGB before chromatic fractions, never labels.
    Brightness-only grains and a return to baseline have no positive signal.
    Drift/noise are remeasured on unchanged controls with exactly this estimator
    and scale. The inherited coarse-grid threshold is not reused at native scale.
    """
    domain, facies = np.asarray(context.domain, bool), np.asarray(context.facies)
    b = reference.rgb_float(baseline, domain.shape)
    c = reference.rgb_float(current, domain.shape)
    dx = boundary._spacing(context.x_m, domain.shape[1])
    dy = boundary._spacing(context.y_m, domain.shape[0], descending=True)
    score = np.zeros(domain.shape, np.float32)
    native_score = np.zeros_like(score)
    available = np.zeros_like(domain)
    if control_mask is None:
        return score, native_score, available
    control_mask = np.asarray(control_mask)
    if control_mask.shape != domain.shape or control_mask.dtype != bool:
        raise ValueError("Reference controls must be an explicit native boolean mask")
    if diagnostics is not None:
        diagnostics.update(estimator="RGB-average-then-chroma-matched-controls", facies=[])
    brightness = b.mean(-1)
    ratio = c.mean(-1) / np.maximum(brightness, 1e-6)
    exposed = (domain & ~gas_mask & (brightness >= reference_settings.minimum_reference_brightness)
        & (ratio >= reference_settings.minimum_brightness_ratio)
        & (ratio <= reference_settings.maximum_brightness_ratio))
    for control in controls:
        if not control.get("available"):
            continue
        area = exposed & (facies == control["facies"])
        if not area.any():
            continue
        direction = np.asarray(control["aqueous_direction"], np.float32)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("Invalid inherited colour direction")
        rows, cols = np.where(area)
        region = np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1]
        local_b, local_c = b[region], c[region]
        averaged = _average_channels(np.concatenate((local_b, local_c), axis=-1),
            area[region], settings.sigma_m/dx, settings.sigma_m/dy)
        smooth_b, smooth_c = averaged[..., :3], averaged[..., 3:]
        delta = (smooth_c / np.maximum(smooth_c.sum(-1, keepdims=True), 1e-6)
                 - smooth_b / np.maximum(smooth_b.sum(-1, keepdims=True), 1e-6))
        selected_controls = area[region] & control_mask[region]
        required = max(reference_settings.minimum_direction_samples,
            int(area.sum()*reference_settings.reference_control_fraction))
        if selected_controls.sum() < required:
            if diagnostics is not None:
                diagnostics["facies"].append(dict(facies=float(control["facies"]), available=False,
                    reason="insufficient unchanged controls at continuity scale"))
            continue
        shift = np.median(delta[selected_controls], axis=0)
        if np.linalg.norm(shift) > reference_settings.maximum_common_chroma_shift:
            continue
        residual = delta-shift
        noise = np.linalg.norm(residual[selected_controls], axis=-1)
        median = float(np.median(noise))
        threshold = max(reference_settings.minimum_chroma_change,
            median + reference_settings.noise_multiplier*1.4826*float(np.median(np.abs(noise-median))))
        projection = residual @ direction
        cosine = projection / np.maximum(np.linalg.norm(residual, axis=-1), 1e-8)
        aligned = area[region] & (cosine >= reference_settings.direction_cosine_minimum)
        score[region][aligned] = np.maximum(projection[aligned]/threshold, 0)
        native_delta = (local_c / np.maximum(local_c.sum(-1, keepdims=True), 1e-6)
            - local_b / np.maximum(local_b.sum(-1, keepdims=True), 1e-6) - shift)
        native_score[region][area[region]] = np.maximum((native_delta @ direction)[area[region]]/threshold, 0)
        available |= area
        if diagnostics is not None:
            diagnostics["facies"].append(dict(facies=float(control["facies"]), available=True,
                controls=int(selected_controls.sum()), chroma_threshold=threshold, common_shift=shift.tolist()))
    return score, native_score, available


def refine(predictions, raw_co2, context, trusted_history, score, native_score, available, settings=None):
    settings = settings or ContinuitySettings()
    domain, facies = np.asarray(context.domain, bool), np.asarray(context.facies)
    co2 = np.asarray(predictions["co2_presence_probability"], np.float32)
    gas = np.asarray(predictions["gas_probability_conditional_on_co2"], np.float32)
    raw, score, native_score = (np.asarray(v, np.float32) for v in (raw_co2, score, native_score))
    trusted_history, available = np.asarray(trusted_history), np.asarray(available)
    arrays = (co2, gas, raw, score, native_score, trusted_history, available, facies)
    if any(a.shape != domain.shape for a in arrays) or domain.ndim != 2:
        raise ValueError("Continuity inputs must share the registered native grid")
    if trusted_history.dtype != bool or available.dtype != bool:
        raise ValueError("History and reference availability must be boolean")
    for field in (co2, gas, raw):
        if not np.isfinite(field).all() or (field < 0).any() or (field > 1).any():
            raise ValueError("Expected finite phase scores in [0, 1]")
    if any(not np.isfinite(field).all() or (field < 0).any() for field in (score, native_score)):
        raise ValueError("Reference evidence must be finite and nonnegative")
    original = domain & (co2 >= .5)
    frozen_gas = original & (gas >= .6)
    detail = dict(format=VERSION, settings=asdict(settings), added_aqueous_pixels=0,
        removed_aqueous_pixels=0, historical_recovery_pixels=0, gas_labels_unchanged=True,
        history_updates=False, calibrated_probability=False,
        support_basis="current signed reference change above noise plus observed history or connected optical evidence")
    if not original.any() or not available.any():
        return dict(predictions), detail
    dx = boundary._spacing(context.x_m, domain.shape[1])
    dy = boundary._spacing(context.y_m, domain.shape[0], descending=True)
    smooth = boundary._material_average(raw, domain, facies, settings.sigma_m/dx, settings.sigma_m/dy)
    near = distance_transform_edt(~original, sampling=(dy, dx)) <= settings.maximum_recovery_m
    historical = (trusted_history & (smooth >= settings.history_optical_minimum)
        & (score >= settings.history_signal_minimum))
    connected = ((smooth >= settings.connected_optical_minimum)
        & (score >= settings.connected_signal_minimum))
    candidates = domain & ~original & available & near & (historical | connected)
    # The local current colour change supplies continuity across grain-sized
    # fluctuations. A path through baseline-coloured water cannot be filled.
    candidates &= score >= 1
    inside = distance_transform_edt(original, sampling=(dy, dx)) <= settings.cleanup_band_m
    removed = (original & ~frozen_gas & available & inside
        & (smooth < settings.cleanup_optical_maximum) & (score < 1)
        & (native_score < settings.connected_signal_minimum))
    retained = original & ~removed
    added = boundary._connected_to_retained(candidates, retained, domain)
    adjusted, conditional = co2.copy(), gas.copy()
    adjusted[added] = np.maximum(.5, smooth[added])
    adjusted[removed] = np.minimum(smooth[removed], .49)
    conditional[added] = np.minimum(conditional[added], .59)
    changed = added | removed
    result = dict(predictions, co2_presence_probability=adjusted,
        gas_probability_conditional_on_co2=conditional)
    for key, old, new in (("untouched_water_probability", 1-co2, 1-adjusted),
            ("aqueous_co2_probability", co2*(1-gas), adjusted*(1-conditional)),
            ("free_co2_gas_probability", co2*gas, adjusted*conditional)):
        result[key] = np.array(predictions.get(key, old), copy=True)
        result[key][changed] = new[changed]
    if not np.array_equal((adjusted >= .5) & (conditional >= .6), frozen_gas):
        raise AssertionError("Continuity changed inherited gas labels")
    detail.update(added_aqueous_pixels=int(added.sum()), removed_aqueous_pixels=int(removed.sum()),
        historical_recovery_pixels=int((added & historical).sum()))
    return result, detail


class AqueousBoundarySequence(legacy.AqueousBoundarySequence):
    def __getattr__(self, name):
        inherited = object.__getattribute__(self, "inherited")
        method = getattr(inherited, name)
        if name in ("apply_with_reference", "apply_rgb"):
            def wrapped(predictions, elapsed_h, *args, **kwargs):
                output, report = method(predictions, elapsed_h, *args, **kwargs)
                rgb = args[0] if args else kwargs.get("current_rgb")
                return self._finish_continuity(predictions, output, report, rgb)
            return wrapped
        return method

    def apply(self, predictions, elapsed_h):
        output, report = self.inherited.apply(predictions, elapsed_h)
        return self._finish_continuity(predictions, output, report, None)

    def _finish_continuity(self, original, inherited, report, rgb):
        before = self.context.domain & (inherited["co2_presence_probability"] >= .5)
        baseline = getattr(self.inherited, "calibration_baseline", None)
        controls = report.get("reference_support", {}).get("facies_controls", [])
        negative = bool(report.get("pre_injection_negative_control"))
        if not negative and rgb is not None and baseline is not None and controls:
            frozen_gas = before & (inherited["gas_probability_conditional_on_co2"] >= .6)
            history = self.inherited.last_diagnostics.get("trusted_history", np.zeros_like(before))
            control_mask = unchanged_control_mask(self.context, inherited,
                original["co2_presence_probability"], history, self.inherited.settings)
            signal_diagnostics = {}
            score, native_score, available = reference_signal(baseline, rgb, self.context,
                frozen_gas, controls, self.inherited.settings, self.settings,
                control_mask=control_mask, diagnostics=signal_diagnostics)
            result, detail = refine(inherited, original["co2_presence_probability"], self.context,
                history, score, native_score, available, self.settings)
            detail["reference_controls"] = signal_diagnostics
        else:
            result = inherited
            detail = dict(format=VERSION, settings=asdict(self.settings), added_aqueous_pixels=0,
                removed_aqueous_pixels=0, historical_recovery_pixels=0, gas_labels_unchanged=True,
                history_updates=False, calibrated_probability=False,
                unavailable_reference_evidence=True)
        after = self.context.domain & (result["co2_presence_probability"] >= .5)
        self.last_diagnostics = {**self.inherited.last_diagnostics,
            "boundary_before_co2": before,
            "boundary_before_gas": before & (inherited["gas_probability_conditional_on_co2"] >= .6),
            "boundary_added_aqueous": after & ~before,
            "boundary_removed_aqueous": before & ~after}
        return result, {**report, "boundary_refinement": {**detail, "bypassed_pre_injection": negative}}
