"""Explicit, phase-blind photometric variants for concentration experiments.

Inputs have already undergone their common geometric corrections.  This module
does not read images, choose references, inspect phase maps, or fit mass.  The
facies model is an empirical flat field in encoded RGB (the historical DarSIA
``hsl-scalar`` convention), not a claim of calibrated linear radiance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
from typing import Any, Literal

import numpy as np


VERSION = "phase-mass-photometric-variants-v1"
IlluminationMode = Literal["none", "historical", "facies"]


class PhotometricVariantError(ValueError):
    """A requested variant is unsupported; callers must not report a fallback."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = {"accepted": False, "reason": message, **(diagnostics or {})}


@dataclass(frozen=True)
class FaciesGainSettings:
    patch_size: int = 128
    minimum_purity: float = 0.90
    minimum_valid_fraction: float = 0.80
    minimum_patches: int = 40
    minimum_patches_per_facies: int = 6
    minimum_coordinate_span: float = 0.60
    maximum_condition: float = 1000.0
    minimum_lightness: float = 0.02
    maximum_channel: float = 0.98
    gain_min: float = 0.50
    gain_max: float = 2.0
    gain_policy: Literal["reject", "bounded_amplitude"] = "reject"
    minimum_gain_amplitude: float = 0.50
    huber_delta: float = 1.5
    iterations: int = 30
    convergence_tolerance: float = 1e-5


@dataclass(frozen=True)
class FaciesIlluminationFit:
    gain: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PhotometricResult:
    rgb: np.ndarray
    diagnostics: dict[str, Any]


def _rgb(image: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise PhotometricVariantError("Expected a geometry-corrected H x W x 3 RGB array")
    if shape is not None and array.shape[:2] != shape:
        raise PhotometricVariantError("Photometric input geometry differs from fitted baseline")
    if not np.isfinite(array).all():
        raise PhotometricVariantError("Photometric input contains nonfinite values")
    return array


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _basis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack((x, y, x * y, x * x, y * y)).astype(np.float32)


def _validate_settings(settings: FaciesGainSettings) -> None:
    if not all(np.isfinite(value) for name, value in asdict(settings).items() if name != "gain_policy"):
        raise PhotometricVariantError("Facies gain settings must be finite")
    if settings.patch_size < 2 or settings.minimum_patches < 6 or settings.minimum_patches_per_facies < 2:
        raise PhotometricVariantError("Patch sizes and minimum sample counts are invalid")
    if not (0 < settings.minimum_purity <= 1 and 0 < settings.minimum_valid_fraction <= 1):
        raise PhotometricVariantError("Patch purity and valid fractions must lie in (0, 1]")
    if not (0 < settings.gain_min <= 1 <= settings.gain_max):
        raise PhotometricVariantError("Gain bounds must be positive and contain one")
    if settings.gain_policy not in ("reject", "bounded_amplitude") or not (0 < settings.minimum_gain_amplitude <= 1):
        raise PhotometricVariantError("Gain policy or minimum retained amplitude is invalid")
    if not (0 < settings.minimum_coordinate_span <= 2 and settings.maximum_condition > 1):
        raise PhotometricVariantError("Spatial support settings are invalid")
    if not (0 < settings.minimum_lightness < settings.maximum_channel <= 1):
        raise PhotometricVariantError("Photometric sampling bounds are invalid")
    if settings.huber_delta <= 0 or settings.iterations < 1 or settings.convergence_tolerance <= 0:
        raise PhotometricVariantError("Robust fit settings are invalid")


def _within_facies(
    design: np.ndarray, signal: np.ndarray, groups: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.float32]]]:
    centered_design = design.copy()
    centered_signal = signal.copy()
    means = {}
    for group in np.unique(groups):
        selected = groups == group
        normalized = weights[selected] / np.sum(weights[selected], dtype=np.float32)
        mean_design = np.sum(design[selected] * normalized[:, None], axis=0, dtype=np.float32)
        mean_signal = np.sum(signal[selected] * normalized, dtype=np.float32)
        centered_design[selected] -= mean_design
        centered_signal[selected] -= mean_signal
        means[int(group)] = (mean_design, mean_signal)
    return centered_design, centered_signal, means


def fit_facies_illumination(
    baseline_rgb: np.ndarray,
    facies: np.ndarray,
    domain: np.ndarray,
    *,
    settings: FaciesGainSettings | None = None,
) -> FaciesIlluminationFit:
    """Fit one smooth common field, retaining an independent intercept per facies.

    Nonoverlapping, predominantly single-facies patches supply robust median
    HLS lightness.  A Huber fit models log(lightness) = facies intercept + common
    quadratic field.  Within-facies centering tests whether spatial variation
    is actually distinguishable from the facies intercepts.  No observations
    from the current image or a plume enter this function.

    The geometric mean scale is unidentifiable; the median porous gain is set
    to one.  The default rejects gains beyond the declared bounds.  The explicit
    ``bounded_amplitude`` policy retains the largest fraction of the centered
    log field consistent with those bounds.  This is a global amplitude
    regularization, never pointwise clipping or a mass-based choice.  Its loss
    in baseline fit is reported; retaining less than the declared minimum
    amplitude rejects the fit.  Model-identifiability checks do not establish that all
    within-facies variation is illumination; packing and chemistry can remain.
    """
    settings = settings or FaciesGainSettings()
    _validate_settings(settings)
    baseline = _rgb(baseline_rgb)
    labels = np.asarray(facies)
    mask = np.asarray(domain, dtype=bool)
    height, width = baseline.shape[:2]
    diagnostics: dict[str, Any] = {
        "version": VERSION, "method": "robust-within-facies-log-hls-quadratic",
        "settings": asdict(settings), "fit_inputs": ["baseline", "facies", "porous_mask"],
        "uses_current_image": False, "uses_phase_maps": False, "uses_mass": False,
        "baseline_sha256": _digest(baseline), "facies_sha256": _digest(labels),
        "domain_sha256": _digest(mask), "shape": [height, width],
        "interpretation": "Conditional on shared illumination and constant baseline response within each facies; does not force current phase colors or different facies to match.",
    }
    if labels.shape != mask.shape or mask.shape != (height, width) or not np.any(mask):
        raise PhotometricVariantError("Facies/domain must match baseline and contain porous pixels", diagnostics)
    if not np.issubdtype(labels.dtype, np.integer):
        raise PhotometricVariantError("Facies must contain integer labels", diagnostics)
    samples = []
    rejected = {"purity": 0, "validity": 0, "small_facies_group": 0}
    size = settings.patch_size
    for row in range(0, height - size + 1, size):
        for col in range(0, width - size + 1, size):
            local_domain = mask[row:row+size, col:col+size]
            if np.mean(local_domain) < settings.minimum_purity:
                rejected["purity"] += 1
                continue
            local_labels = labels[row:row+size, col:col+size]
            unique, counts = np.unique(local_labels[local_domain], return_counts=True)
            group = unique[np.argmax(counts)]
            selected = local_domain & (local_labels == group)
            if np.mean(selected) < settings.minimum_purity:
                rejected["purity"] += 1
                continue
            rgb = baseline[row:row+size, col:col+size]
            high = np.max(rgb, axis=-1)
            lightness = (high + np.min(rgb, axis=-1)) * np.float32(0.5)
            valid = selected & (lightness >= settings.minimum_lightness) & (high <= settings.maximum_channel)
            if np.count_nonzero(valid) < settings.minimum_valid_fraction * np.count_nonzero(selected):
                rejected["validity"] += 1
                continue
            # Patch center is fixed by geometry, not shifted toward bright pixels.
            x = np.float32(2 * (col + (size-1)/2) / max(width-1, 1) - 1)
            y = np.float32(2 * (row + (size-1)/2) / max(height-1, 1) - 1)
            samples.append((int(group), x, y, np.float32(np.log(np.median(lightness[valid])))))
    group_counts: dict[int, int] = {}
    for sample in samples:
        group_counts[sample[0]] = group_counts.get(sample[0], 0) + 1
    accepted_samples = [s for s in samples if group_counts[s[0]] >= settings.minimum_patches_per_facies]
    rejected["small_facies_group"] = len(samples) - len(accepted_samples)
    diagnostics.update(patches=len(accepted_samples), rejected_patches=rejected,
        facies_patch_counts={str(k): v for k,v in group_counts.items()},
        omitted_facies=[int(k) for k,v in group_counts.items() if v < settings.minimum_patches_per_facies])
    if len(accepted_samples) < settings.minimum_patches:
        raise PhotometricVariantError("Insufficient predominantly single-facies baseline patches", diagnostics)
    groups = np.asarray([s[0] for s in accepted_samples], dtype=np.int32)
    x = np.asarray([s[1] for s in accepted_samples], dtype=np.float32)
    y = np.asarray([s[2] for s in accepted_samples], dtype=np.float32)
    signal = np.asarray([s[3] for s in accepted_samples], dtype=np.float32)
    diagnostics["coordinate_spans"] = [float(np.ptp(x)), float(np.ptp(y))]
    if min(np.ptp(x), np.ptp(y)) < settings.minimum_coordinate_span:
        raise PhotometricVariantError("Baseline patches do not span both spatial directions", diagnostics)
    design = _basis(x, y)
    # Equal total influence per represented facies, then robust downweighting.
    base_weights = np.asarray([1 / group_counts[int(g)] for g in groups], dtype=np.float32)
    weights = base_weights.copy()
    coefficients = np.zeros(5, dtype=np.float32)
    converged = False
    for iteration in range(settings.iterations):
        centered, response, means = _within_facies(design, signal, groups, weights)
        root_weights = np.sqrt(weights).astype(np.float32)
        weighted = centered * root_weights[:, None]
        singular = np.linalg.svd(weighted, compute_uv=False)
        tolerance = np.finfo(np.float32).eps * max(weighted.shape) * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        condition = float(singular[0] / max(singular[-1], np.finfo(np.float32).tiny))
        diagnostics.update(rank=rank, condition=condition, singular_values=singular.tolist())
        if rank != 5 or condition > settings.maximum_condition:
            raise PhotometricVariantError("Spatial illumination is confounded with facies or poorly supported", diagnostics)
        updated = np.linalg.lstsq(weighted, response * root_weights, rcond=None)[0].astype(np.float32)
        residuals = response - centered @ updated
        scale = np.float32(max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 1e-4))
        robust = np.minimum(np.float32(1), np.float32(settings.huber_delta) * scale / np.maximum(np.abs(residuals), np.float32(1e-8)))
        if np.max(np.abs(updated - coefficients)) <= settings.convergence_tolerance:
            coefficients = updated
            converged = True
            break
        coefficients = updated
        weights = base_weights * robust
    diagnostics.update(iterations=iteration+1, converged=converged,
        coefficients=coefficients.tolist(), robust_residual_scale=float(scale),
        downweighted_fraction=float(np.mean(robust < .5)),
        facies_intercepts={str(group):float(value - mean @ coefficients) for group,(mean,value) in means.items()})
    if not converged:
        raise PhotometricVariantError("Robust facies illumination fit did not converge", diagnostics)
    # Evaluate in row tiles: no native-resolution five-column design matrix.
    field = np.empty((height, width), dtype=np.float32)
    xx = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    yy = np.linspace(-1, 1, height, dtype=np.float32)
    for row in range(0, height, 256):
        local_y = yy[row:row+256, None]
        field[row:row+256] = (coefficients[0]*xx + coefficients[1]*local_y + coefficients[2]*xx*local_y
            + coefficients[3]*xx*xx + coefficients[4]*local_y*local_y)
    center = np.float32(np.median(field[mask]))
    np.subtract(center, field, out=field)
    log_min, log_max = float(field.min()), float(field.max())
    unconstrained_min, unconstrained_max = float(np.exp(log_min)), float(np.exp(log_max))
    diagnostics.update(unconstrained_gain_min=unconstrained_min, unconstrained_gain_max=unconstrained_max,
        unconstrained_coefficients=coefficients.tolist(), gain_policy=settings.gain_policy)
    amplitude = np.float32(1)
    outside_bounds = unconstrained_min < settings.gain_min or unconstrained_max > settings.gain_max
    if outside_bounds and settings.gain_policy == "bounded_amplitude":
        limits = [1.0]
        if log_max > 0:
            limits.append(float(np.log(settings.gain_max)) / log_max)
        if log_min < 0:
            limits.append(float(np.log(settings.gain_min)) / log_min)
        # One conservative float32 step avoids a rounding excursion over the bound.
        amplitude = np.nextafter(np.float32(min(limits)), np.float32(0))
        diagnostics["gain_amplitude"] = float(amplitude)
        if amplitude < settings.minimum_gain_amplitude:
            raise PhotometricVariantError("Bounded illumination would retain too little of the fitted field", diagnostics)
        field *= amplitude
    diagnostics.update(gain_amplitude=float(amplitude), amplitude_regularized=bool(amplitude < 1),
        bounded_coefficients=(coefficients*amplitude).tolist(),
        facies_intercepts_after_amplitude={str(group):float(value - mean @ (coefficients*amplitude))
            for group,(mean,value) in means.items()})
    residual_after = response - centered @ (coefficients*amplitude)
    weight_sum = np.sum(weights, dtype=np.float32)
    rmse_before = float(np.sqrt(np.sum(weights*residuals*residuals,dtype=np.float32)/weight_sum))
    rmse_after = float(np.sqrt(np.sum(weights*residual_after*residual_after,dtype=np.float32)/weight_sum))
    diagnostics["baseline_fit_amplitude_effect"] = {
        "metric": "weighted RMSE of patch log lightness, refitted facies intercepts, same final robust weights",
        "before": rmse_before, "after": rmse_after, "increase": rmse_after-rmse_before,
        "mass_fit_used": False,
    }
    np.exp(field, out=field)
    diagnostics.update(gain_normalization="median porous gain = 1", field_log_center=float(center),
        gain_min=float(field.min()), gain_median_porous=float(np.median(field[mask])), gain_max=float(field.max()))
    if not np.isfinite(field).all() or field.min() < settings.gain_min or field.max() > settings.gain_max:
        raise PhotometricVariantError("Estimated illumination gain exceeds conservative declared bounds", diagnostics)
    field.setflags(write=False)
    diagnostics.update(accepted=True, gain_sha256=_digest(field))
    return FaciesIlluminationFit(field, diagnostics)


def _range_diagnostics(rgb: np.ndarray) -> dict[str, Any]:
    high = rgb > 1
    low = rgb < 0
    return {"channels": int(rgb.size), "pixels": int(rgb.shape[0]*rgb.shape[1]),
        "channels_above1": int(np.count_nonzero(high)), "pixels_any_above1": int(np.count_nonzero(high.any(axis=-1))),
        "channels_below0": int(np.count_nonzero(low)), "minimum": float(rgb.min()), "maximum": float(rgb.max()),
        "pixels_any_ge0995": int(np.count_nonzero((rgb >= np.float32(.995)).any(axis=-1)))}


class PhotometricTransformer:
    """One fitted branch; reuse it for baseline, clearwater, and all run images.

    Instances hold a private ColorChecker object and are not thread-safe.
    Existing DarSIA correction/refinement guards and algorithms are preserved.
    Range diagnostics before ColorChecker expose values its tone stage may
    clamp; they count inputs, not internal implementation-specific clamp calls.
    """

    def __init__(self, shape, illumination, historical_correction, facies_fit, colorchecker, diagnostics):
        self.shape = shape
        self.illumination = illumination
        self._historical = historical_correction
        self._facies = facies_fit
        self._colorchecker = colorchecker
        self.diagnostics = diagnostics

    def illuminate(self, geometry_rgb: np.ndarray) -> np.ndarray:
        """Apply the fixed field only, without clipping or refitting."""
        rgb = _rgb(geometry_rgb, self.shape)
        if self.illumination == "none":
            return rgb.copy()
        if self.illumination == "historical":
            return _rgb(self._historical.correct_array(rgb), self.shape)
        return np.multiply(rgb, self._facies.gain[..., None], dtype=np.float32)

    def apply(self, geometry_rgb: np.ndarray) -> PhotometricResult:
        lit = self.illuminate(geometry_rgb)
        diagnostics = {"version": VERSION, "illumination": self.illumination,
            "colorchecker_enabled": self._colorchecker is not None, "before_colorchecker": _range_diagnostics(lit)}
        if self._colorchecker is None:
            corrected = lit
        else:
            try:
                corrected = _rgb(self._colorchecker.correct_array(lit), self.shape)
            except ValueError as error:
                raise PhotometricVariantError("ColorChecker rejected variant: " + str(error), diagnostics) from error
            cc = self._colorchecker
            diagnostics["colorchecker"] = {
                "mean_residual": float(getattr(cc, "last_residual", np.nan)),
                "max_residual": float(getattr(cc, "last_max_swatch_residual", np.nan)),
                "flags": list(getattr(cc, "last_flag_reasons", []) or []),
                "saturation_fallback": copy.deepcopy(getattr(cc, "last_saturation_fallback", None)),
                "target_interpretation": "relative to branch baseline, not independent absolute chart accuracy",
            }
        diagnostics.update(accepted=True, output=(dict(diagnostics["before_colorchecker"])
            if self._colorchecker is None else _range_diagnostics(corrected)))
        return PhotometricResult(np.asarray(corrected, dtype=np.float32), diagnostics)


def build_photometric_transformer(
    baseline_rgb: np.ndarray,
    *,
    illumination: IlluminationMode,
    colorchecker_enabled: bool,
    historical_correction: Any = None,
    colorchecker_correction: Any = None,
    facies_fit: FaciesIlluminationFit | None = None,
) -> PhotometricTransformer:
    """Construct a branch using a shared, already geometry-corrected baseline.

    ColorChecker target is sampled from this branch's illuminated baseline using
    the supplied correction's *fixed, already oriented* ROI.  This reproduces
    ``auto_orient`` target sampling without re-localizing geometry or changing
    its correction method, thresholds, saturation fallback, or rejection rules.
    """
    baseline = _rgb(baseline_rgb)
    if illumination not in ("none", "historical", "facies"):
        raise PhotometricVariantError("Unknown illumination mode: " + str(illumination))
    if illumination == "historical":
        if historical_correction is None or not getattr(historical_correction, "local_scaling", None):
            raise PhotometricVariantError("Historical illumination requires an actual saved gain field")
    if illumination == "facies":
        if facies_fit is None or facies_fit.diagnostics.get("accepted") is not True:
            raise PhotometricVariantError("Facies illumination requires an accepted baseline-only fit")
        if facies_fit.gain.shape != baseline.shape[:2] or not np.isfinite(facies_fit.gain).all():
            raise PhotometricVariantError("Facies gain does not match baseline geometry")
        if facies_fit.diagnostics.get("baseline_sha256") != _digest(baseline):
            raise PhotometricVariantError("Facies gain was fitted to a different baseline")
    diagnostics = {"version": VERSION, "illumination": illumination,
        "colorchecker_enabled": bool(colorchecker_enabled), "baseline_sha256": _digest(baseline),
        "facies_fit": copy.deepcopy(facies_fit.diagnostics) if illumination == "facies" else None,
        "output_clipping": "none in this adapter; existing ColorChecker internal behavior unchanged"}
    transformer = PhotometricTransformer(baseline.shape[:2], illumination, historical_correction, facies_fit, None, diagnostics)
    if colorchecker_enabled:
        if colorchecker_correction is None:
            raise PhotometricVariantError("ColorChecker branch requires an oriented correction object")
        import darsia
        from darsia.fluidflower.tools.colorchecker_refinement import extract_swatches
        correction = copy.deepcopy(colorchecker_correction)
        lit = transformer.illuminate(baseline)
        target = extract_swatches(correction._restrict_to_roi(lit)).astype(np.float32)
        if target.shape != (4, 6, 3) or not np.isfinite(target).all():
            raise PhotometricVariantError("Invalid branch baseline ColorChecker target")
        correction.colorchecker = darsia.CustomColorChecker(reference_colors=target)
        transformer._colorchecker = correction
        diagnostics["colorchecker"] = {"roi": np.asarray(correction.roi).tolist(),
            "target_rgb": target.tolist(), "target_sha256": _digest(target),
            "target_source": "same baseline after selected illumination, fixed oriented ROI",
            "algorithm_revision": getattr(correction, "algorithm_revision", None),
            "algorithm_id": getattr(correction, "algorithm_id", None),
            "baseline_residual_is_self_match": True}
    return transformer
