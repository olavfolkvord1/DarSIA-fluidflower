"""Optional ColorChecker refinement in the BTB-sensitive colour coordinate.

The established FF-AC ColorChecker correction minimizes an RGB residual over
the 24 chart fields. A small residual can still point along the blue-to-yellow
direction used to infer acidification. This module adds a conservative final
step that targets that direction explicitly while preserving the red/green
ratio and the combined yellow-plus-blue intensity of every pixel.

The extension is opt-in and is installed by dedicated pilot campaigns. It does
not alter the currently running ColorChecker campaign or DarSIA itself.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import numpy as np

from darsia.fluidflower.tools import colorchecker_refinement as base_refinement


ALGORITHM_ID = "ffac-btb-coordinate-affine-loo-v2"


def btb_coordinate(values: Any, *, epsilon: float = 1e-6) -> np.ndarray:
    """Return the bounded blue-to-yellow coordinate used as a BTB proxy."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB values, got shape {array.shape}.")
    red = np.maximum(array[..., 0], 0.0)
    green = np.maximum(array[..., 1], 0.0)
    blue = np.maximum(array[..., 2], 0.0)
    yellow = 0.5 * (red + green)
    return (yellow - blue) / (yellow + blue + float(epsilon))


def coordinate_residual_summary(
    swatches: Any,
    reference: Any,
    *,
    swatch_weights: Any,
    maximum_weight: float,
) -> dict[str, Any]:
    """Summarize residuals in the CO2-sensitive coordinate."""

    residuals = np.abs(
        btb_coordinate(swatches) - btb_coordinate(reference)
    ).reshape(-1)
    weights = np.asarray(swatch_weights, dtype=np.float64).reshape(-1)
    if residuals.shape != weights.shape:
        raise ValueError("Need one BTB-coordinate weight per swatch.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("BTB-coordinate weights must be finite and positive.")
    max_weight = float(maximum_weight)
    if not np.isfinite(max_weight) or max_weight < 0.0:
        raise ValueError("BTB-coordinate maximum weight must be non-negative.")
    worst = int(np.argmax(residuals))
    weighted_mean = float(np.average(residuals, weights=weights))
    maximum = float(residuals[worst])
    return {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "weighted_mean": weighted_mean,
        "p95": float(np.percentile(residuals, 95.0)),
        "max": maximum,
        "score": weighted_mean + max_weight * maximum,
        "worst_swatch_index": worst,
        "worst_swatch_name": base_refinement.SWATCH_NAMES[worst],
    }


def fit_coordinate_map(
    source_swatches: Any,
    destination_swatches: Any,
    *,
    sample_weights: Any,
    ridge: float,
    huber_delta: float,
    max_iterations: int = 8,
) -> tuple[float, float, dict[str, float]]:
    """Fit a robust near-identity affine map in the BTB coordinate."""

    source = btb_coordinate(source_swatches).reshape(-1)
    destination = btb_coordinate(destination_swatches).reshape(-1)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if source.shape != destination.shape or source.shape != weights.shape:
        raise ValueError("Source, destination, and weight counts must match.")
    if source.size < 4:
        raise ValueError("At least four ColorChecker fields are required.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("BTB coordinates must be finite.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("BTB coordinate weights must be finite and positive.")
    regularization = float(ridge)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("BTB coordinate ridge must be finite and non-negative.")

    design = np.column_stack((source, np.ones_like(source)))
    prior = np.asarray([1.0, 0.0], dtype=np.float64)
    penalty = np.diag([regularization, 4.0 * regularization])
    base_weights = weights / np.mean(weights)
    robust_weights = np.ones_like(base_weights)
    coefficients = prior.copy()
    effective_weights = base_weights.copy()
    for _ in range(max(1, int(max_iterations))):
        effective_weights = base_weights * robust_weights
        weighted_design = design * effective_weights[:, None]
        coefficients = np.linalg.solve(
            design.T @ weighted_design + penalty,
            design.T @ (destination * effective_weights) + penalty @ prior,
        )
        errors = np.abs(design @ coefficients - destination)
        if huber_delta <= 0.0:
            break
        updated = np.minimum(
            1.0,
            float(huber_delta) / np.maximum(errors, 1e-12),
        )
        if np.max(np.abs(updated - robust_weights)) < 1e-6:
            robust_weights = updated
            effective_weights = base_weights * robust_weights
            break
        robust_weights = updated

    slope, offset = (float(value) for value in coefficients)
    grid = np.linspace(-0.95, 0.95, 257)
    shifts = slope * grid + offset - grid
    return slope, offset, {
        "slope": slope,
        "offset": offset,
        "maximum_coordinate_shift": float(np.max(np.abs(shifts))),
        "minimum_weight": float(np.min(effective_weights)),
        "maximum_weight": float(np.max(effective_weights)),
    }


def leave_one_out_summary(
    source_swatches: Any,
    destination_swatches: Any,
    *,
    sample_weights: Any,
    ridge: float,
    huber_delta: float,
    rgb_weights: Any | None = None,
    maximum_signal_weight: float = 0.0,
) -> dict[str, float]:
    """Cross-validate the affine BTB residual against no residual step."""

    source = np.asarray(source_swatches, dtype=np.float64).reshape(-1, 3)
    destination = np.asarray(destination_swatches, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if source.shape != destination.shape or weights.shape != (source.shape[0],):
        raise ValueError("Need matching RGB swatches and one weight per field.")
    if source.shape[0] < 5:
        raise ValueError("At least five ColorChecker fields are required for LOO.")
    rgb_sample_weights = (
        np.ones_like(weights)
        if rgb_weights is None
        else np.asarray(rgb_weights, dtype=np.float64).reshape(-1)
    )
    if rgb_sample_weights.shape != weights.shape:
        raise ValueError("Need one RGB weight per ColorChecker field.")

    predictions = np.empty_like(destination)
    for held_out in range(source.shape[0]):
        keep = np.arange(source.shape[0]) != held_out
        slope, offset, _diagnostics = fit_coordinate_map(
            source[keep],
            destination[keep],
            sample_weights=weights[keep],
            ridge=ridge,
            huber_delta=huber_delta,
        )
        predictions[held_out] = apply_coordinate_map(
            source[held_out].reshape(1, 1, 3),
            slope=slope,
            offset=offset,
        ).reshape(3)

    identity_signal = coordinate_residual_summary(
        source,
        destination,
        swatch_weights=weights,
        maximum_weight=maximum_signal_weight,
    )
    model_signal = coordinate_residual_summary(
        predictions,
        destination,
        swatch_weights=weights,
        maximum_weight=maximum_signal_weight,
    )
    identity_rgb = base_refinement.residual_summary(
        source,
        destination,
        swatch_weights=rgb_sample_weights,
        max_swatch_loss_weight=0.0,
    )
    model_rgb = base_refinement.residual_summary(
        predictions,
        destination,
        swatch_weights=rgb_sample_weights,
        max_swatch_loss_weight=0.0,
    )
    signal_before = float(identity_signal["score"])
    signal_after = float(model_signal["score"])
    return {
        "identity_signal_score": signal_before,
        "model_signal_score": signal_after,
        "signal_relative_improvement": float(
            (signal_before - signal_after) / max(signal_before, 1e-12)
        ),
        "identity_rgb_score": float(identity_rgb["score"]),
        "model_rgb_score": float(model_rgb["score"]),
    }


def apply_coordinate_map(
    image: Any,
    *,
    slope: float,
    offset: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Apply a minimal BTB-axis change while preserving R:G and Y+B."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an RGB image, got shape {array.shape}.")
    rgb = np.asarray(array[..., :3], dtype=np.float64)
    red = np.maximum(rgb[..., 0], 0.0)
    green = np.maximum(rgb[..., 1], 0.0)
    blue = np.maximum(rgb[..., 2], 0.0)
    yellow = 0.5 * (red + green)
    total = yellow + blue
    source_coordinate = (yellow - blue) / (total + float(epsilon))
    target_coordinate = np.clip(
        float(slope) * source_coordinate + float(offset),
        -0.995,
        0.995,
    )
    target_yellow = 0.5 * total * (1.0 + target_coordinate)
    target_blue = 0.5 * total * (1.0 - target_coordinate)
    yellow_gain = np.divide(
        target_yellow,
        yellow,
        out=np.ones_like(target_yellow),
        where=yellow > float(epsilon),
    )
    corrected = np.array(array, copy=True)
    corrected[..., 0] = red * yellow_gain
    corrected[..., 1] = green * yellow_gain
    corrected[..., 2] = target_blue
    return np.asarray(np.clip(corrected, 0.0, 1.0), dtype=np.float32)


def _rgb_summary(correction: Any, swatches: Any, reference: Any) -> dict[str, Any]:
    return base_refinement.residual_summary(
        swatches,
        reference,
        swatch_weights=correction.swatch_weights,
        max_swatch_loss_weight=correction.max_swatch_loss_weight,
    )


def _measure(correction: Any, image: Any, reference: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    swatches = base_refinement.extract_swatches(
        correction._restrict_to_roi(np.asarray(image))
    )
    return (
        swatches,
        _rgb_summary(correction, swatches, reference),
        coordinate_residual_summary(
            swatches,
            reference,
            swatch_weights=correction.signal_swatch_weights,
            maximum_weight=correction.signal_maximum_loss_weight,
        ),
    )


def _correct_array(correction: Any, image: Any) -> np.ndarray:
    corrected = correction._ffac_signal_original_correct_array(image)
    reference = np.asarray(correction.colorchecker.swatches_rgb)
    source_swatches, rgb_before, signal_before = _measure(
        correction, corrected, reference
    )
    slope, offset, fit = fit_coordinate_map(
        source_swatches,
        reference,
        sample_weights=correction.signal_swatch_weights,
        ridge=correction.signal_refinement_ridge,
        huber_delta=correction.signal_refinement_huber_delta,
    )
    cross_validation = leave_one_out_summary(
        source_swatches,
        reference,
        sample_weights=correction.signal_swatch_weights,
        rgb_weights=correction.swatch_weights,
        ridge=correction.signal_refinement_ridge,
        huber_delta=correction.signal_refinement_huber_delta,
        maximum_signal_weight=correction.signal_maximum_loss_weight,
    )
    bounded = bool(
        correction.signal_slope_min <= slope <= correction.signal_slope_max
        and abs(offset) <= correction.signal_offset_max_abs
        and fit["maximum_coordinate_shift"]
        <= correction.signal_maximum_coordinate_shift
    )
    accepted = False
    reason = "transform-bounds"
    rgb_after = rgb_before
    signal_after = signal_before
    loo_rgb_limit = (
        float(cross_validation["identity_rgb_score"])
        * (1.0 + correction.signal_rgb_relative_tolerance)
        + correction.signal_rgb_absolute_tolerance
    )
    loo_guard_active = correction.signal_minimum_loo_improvement >= 0.0
    loo_accepted = bool(
        not loo_guard_active
        or (
            cross_validation["signal_relative_improvement"]
            >= correction.signal_minimum_loo_improvement
            and cross_validation["model_rgb_score"] <= loo_rgb_limit
        )
    )
    if bounded and loo_accepted:
        candidate = apply_coordinate_map(
            corrected,
            slope=slope,
            offset=offset,
        )
        candidate_swatches, candidate_rgb, candidate_signal = _measure(
            correction, candidate, reference
        )
        relative_improvement = (
            float(signal_before["score"]) - float(candidate_signal["score"])
        ) / max(float(signal_before["score"]), 1e-12)
        rgb_limit = (
            float(rgb_before["score"])
            * (1.0 + correction.signal_rgb_relative_tolerance)
            + correction.signal_rgb_absolute_tolerance
        )
        if not np.isfinite(float(candidate_signal["score"])):
            reason = "non-finite-signal-score"
        elif relative_improvement < correction.signal_minimum_improvement:
            reason = "insufficient-signal-improvement"
        elif float(candidate_rgb["score"]) > rgb_limit:
            reason = "rgb-residual-regression"
        else:
            corrected = candidate
            source_swatches = candidate_swatches
            rgb_after = candidate_rgb
            signal_after = candidate_signal
            accepted = True
            reason = "accepted"
    elif bounded:
        relative_improvement = 0.0
        reason = (
            "loo-rgb-residual-regression"
            if cross_validation["model_rgb_score"] > loo_rgb_limit
            else "insufficient-loo-improvement"
        )
    else:
        relative_improvement = 0.0

    correction.last_signal_refinement_diagnostics = {
        **fit,
        "bounded": bounded,
        "accepted": accepted,
        "reason": reason,
        "leave_one_out_guard_active": loo_guard_active,
        "leave_one_out": cross_validation,
        "signal_score_before": float(signal_before["score"]),
        "signal_score_after": float(signal_after["score"]),
        "signal_relative_improvement": float(relative_improvement),
        "rgb_score_before": float(rgb_before["score"]),
        "rgb_score_after": float(rgb_after["score"]),
    }
    correction.last_signal_residual = float(signal_after["mean"])
    correction.last_signal_weighted_residual = float(
        signal_after["weighted_mean"]
    )
    correction.last_signal_max_residual = float(signal_after["max"])
    correction.last_signal_worst_swatch_name = str(
        signal_after["worst_swatch_name"]
    )

    # Downstream frame QC must describe the image that is actually returned.
    correction.last_swatch_residuals = np.asarray(
        rgb_after["residuals"], dtype=np.float64
    )
    correction.last_residual = float(rgb_after["mean"])
    correction.last_weighted_residual = float(rgb_after["weighted_mean"])
    correction.last_p95_swatch_residual = float(rgb_after["p95"])
    correction.last_max_swatch_residual = float(rgb_after["max"])
    correction.last_residual_score = float(rgb_after["score"])
    correction.last_worst_swatch_index = int(rgb_after["worst_swatch_index"])
    correction.last_worst_swatch_name = str(rgb_after["worst_swatch_name"])
    flag_reasons: list[str] = []
    if correction.last_residual > correction.residual_warn_threshold:
        flag_reasons.append("mean-residual")
    if correction.last_weighted_residual > correction.residual_warn_threshold:
        flag_reasons.append("weighted-residual")
    if correction.last_max_swatch_residual > correction.max_swatch_warn_threshold:
        flag_reasons.append("max-swatch-residual")
    correction.last_flag_reasons = flag_reasons
    correction.last_flagged = bool(flag_reasons)
    return np.asarray(corrected, dtype=np.float32)


def install(
    correction: Any,
    *,
    settings: Mapping[str, float],
) -> dict[str, Any]:
    """Install the opt-in BTB-coordinate final refinement."""

    defaults = {
        "signal_refinement_ridge": 0.05,
        "signal_refinement_huber_delta": 0.02,
        "signal_minimum_improvement": 0.01,
        "signal_slope_min": 0.75,
        "signal_slope_max": 1.25,
        "signal_offset_max_abs": 0.06,
        "signal_maximum_coordinate_shift": 0.10,
        "signal_rgb_relative_tolerance": 0.10,
        "signal_rgb_absolute_tolerance": 0.0005,
        "signal_maximum_loss_weight": 0.30,
        # -1 preserves the historical in-sample-only acceptance contract.
        # Dedicated campaigns opt into a non-negative held-out threshold.
        "signal_minimum_loo_improvement": -1.0,
        "signal_btb_swatch_weight": 1.0,
    }
    resolved = {**defaults, **{str(key): float(value) for key, value in settings.items()}}
    if not all(np.isfinite(float(value)) for value in resolved.values()):
        raise ValueError("BTB signal-refinement settings must be finite.")
    if resolved["signal_refinement_ridge"] < 0.0:
        raise ValueError("BTB signal-refinement ridge must be non-negative.")
    if not 0.0 <= resolved["signal_minimum_improvement"] < 1.0:
        raise ValueError("BTB minimum improvement must be in [0, 1).")
    if not -1.0 <= resolved["signal_minimum_loo_improvement"] < 1.0:
        raise ValueError("BTB LOO minimum improvement must be in [-1, 1).")
    if resolved["signal_btb_swatch_weight"] <= 0.0:
        raise ValueError("BTB signal swatch weight must be positive.")
    if not 0.0 < resolved["signal_slope_min"] <= resolved["signal_slope_max"]:
        raise ValueError("BTB signal slope bounds are invalid.")
    for name, value in resolved.items():
        setattr(correction, name, float(value))
    correction.signal_swatch_weights = base_refinement.build_swatch_weights(
        resolved["signal_btb_swatch_weight"]
    )
    correction.last_signal_residual = float("nan")
    correction.last_signal_weighted_residual = float("nan")
    correction.last_signal_max_residual = float("nan")
    correction.last_signal_worst_swatch_name = ""
    correction.last_signal_refinement_diagnostics = {}
    if not hasattr(correction, "_ffac_signal_original_correct_array"):
        correction._ffac_signal_original_correct_array = correction.correct_array
        correction.correct_array = MethodType(_correct_array, correction)
    return {
        "active": True,
        "implementation": "darsia.fluidflower.tools.colorchecker_signal_refinement",
        "algorithm_id": ALGORITHM_ID,
        "signal_swatch_weights": [
            float(value) for value in correction.signal_swatch_weights
        ],
        **resolved,
    }
