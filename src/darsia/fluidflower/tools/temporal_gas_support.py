"""Causal optical support and independent mass caps for free CO2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class CausalGasGateSettings:
    """Settings for current-image, source, and history-supported free gas."""

    support_level: float = 0.05
    optical_onset: float = 0.23
    optical_width: float = 0.08
    seal_optical_onset: float = 0.45
    seal_optical_width: float = 0.12
    optical_smoothing_sigma_m: float = 0.0
    minimum_component_area_m2: float = 0.0
    history_radius_m: float = 0.005
    reentry_radius_m: float = 0.015
    source_radius_x_m: float = 0.040
    source_radius_y_m: float = 0.030
    use_rate_scaled_source_prior: bool = False
    reference_source_rate_kg_s: float = 6.24e-8
    source_rate_radius_exponent: float = 0.50
    minimum_source_rate_scale: float = 0.35
    maximum_source_rate_scale: float = 2.25
    structure_scale: float = 0.20
    minimum_source_optical_fraction: float = 0.10
    minimum_history_optical_fraction: float = 0.10
    minimum_reentry_optical_fraction: float = 0.75
    seal_facies_ids: tuple[int, ...] = (1,)
    seal_component_fraction: float = 0.50
    reject_seal_dominant_components: bool = True
    allow_nonseal_reentry: bool = True
    use_plume_interior_prior: bool = False
    co2_support_level: float = 0.20
    co2_interior_scale_m: float = 0.005
    co2_boundary_floor: float = 0.15
    use_permeability_structure_prior: bool = False
    permeability_interface_scale_m: float = 0.006
    use_buoyancy_prior: bool = False
    post_injection_downward_tolerance_m: float = 0.005
    post_injection_downward_floor: float = 0.15


@dataclass(frozen=True)
class CausalGasGateFrame:
    saturation: np.ndarray
    optical_weight: np.ndarray
    retained_support_mask: np.ndarray
    removed_support_mask: np.ndarray
    component_count: int
    retained_component_count: int
    rejected_component_count: int
    source_supported_component_count: int
    history_supported_component_count: int
    reentry_supported_component_count: int
    rejected_seal_component_count: int
    rejected_small_component_count: int
    raw_support_pixels: int
    optically_supported_pixels: int
    retained_support_pixels: int
    removed_raw_gas_fraction: float
    plume_interior_gate_mean: float = 1.0
    raw_gas_plume_boundary_fraction: float = 0.0
    retained_gas_plume_boundary_fraction: float = 0.0
    permeability_barrier_affinity_mean: float = 0.0
    downward_penalty_pixel_fraction: float = 0.0
    active_source_radius_scale: float = 1.0


@dataclass(frozen=True)
class GasMassCapFrame:
    saturation: np.ndarray
    uncapped_mass_kg: float
    capped_mass_kg: float
    cap_mass_kg: float
    scale: float


def hybrid_shared_signal_weight(
    elapsed_h: float,
    *,
    injection_end_h: float,
    half_life_h: float,
) -> float:
    """Return the shared-signal contribution to the hybrid gas estimate."""

    time_h = float(elapsed_h)
    end_h = float(injection_end_h)
    half_life = float(half_life_h)
    if not np.isfinite(time_h) or not np.isfinite(end_h):
        raise ValueError("Hybrid gas times must be finite.")
    if not np.isfinite(half_life) or half_life <= 0.0:
        raise ValueError("Hybrid gas half-life must be finite and positive.")
    if time_h <= end_h:
        return 1.0
    return float(2.0 ** (-(time_h - end_h) / half_life))


def late_gas_fraction_envelope(
    elapsed_h: float,
    *,
    start_h: float,
    start_fraction: float,
    half_life_h: float,
    floor_fraction: float = 0.0,
) -> float:
    """Return a smooth upper envelope for post-injection free-gas mass."""

    time_h = float(elapsed_h)
    start = float(start_h)
    fraction = float(start_fraction)
    half_life = float(half_life_h)
    floor = float(floor_fraction)
    values = (time_h, start, fraction, half_life, floor)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Late-gas envelope settings must be finite.")
    if start < 0.0 or half_life <= 0.0:
        raise ValueError("Late-gas start must be non-negative and half-life positive.")
    if not 0.0 <= floor <= fraction <= 1.0:
        raise ValueError(
            "Late-gas fractions must satisfy 0 <= floor <= start <= 1."
        )
    if time_h <= start:
        return fraction
    return max(floor, fraction * 2.0 ** (-(time_h - start) / half_life))


def build_causal_optical_gate_weights(
    optical_scores: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    elapsed_hours: Sequence[float],
    injection_stages: Sequence[Any],
    facies_labels: np.ndarray | None,
    settings: CausalGasGateSettings,
) -> list[np.ndarray]:
    """Build parameter-independent optical weights for hybrid gas inference.

    Unit saturation is intentional: component support is then determined only
    by the current optical observation, source locations, and causal history,
    rather than by any candidate signal/Flash parameters proposed by Optuna.
    """

    scores = [np.asarray(score, dtype=np.float32) for score in optical_scores]
    if not scores:
        return []
    unit_saturation = [np.ones_like(score, dtype=np.float32) for score in scores]
    frames = apply_causal_gas_gate(
        unit_saturation,
        scores,
        valid_mask,
        x_coordinates_m=x_coordinates_m,
        y_coordinates_m=y_coordinates_m,
        elapsed_hours=elapsed_hours,
        injection_stages=injection_stages,
        facies_labels=facies_labels,
        settings=settings,
    )
    return [np.asarray(frame.saturation, dtype=np.float32) for frame in frames]


def _resize_mean(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(values, dtype=np.float32),
        target,
        interpolation=cv2.INTER_AREA,
    )


def _resample_axis(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or not len(source):
        raise ValueError("Coordinate axes must be non-empty one-dimensional arrays.")
    if len(source) == size:
        return source.copy()
    positions = np.linspace(0.0, len(source) - 1.0, int(size))
    return np.interp(positions, np.arange(len(source), dtype=float), source)


def _stage_value(stage: Any, name: str) -> float:
    if isinstance(stage, dict):
        return float(stage[name])
    return float(getattr(stage, name))


def _stage_optional(stage: Any, name: str, default: Any = None) -> Any:
    if isinstance(stage, dict):
        return stage.get(name, default)
    return getattr(stage, name, default)


def _stage_rate_kg_s(stage: Any, elapsed_h: float) -> float | None:
    """Return the instantaneous protocol rate when stage metadata provides it."""

    segments = _stage_optional(stage, "rate_segments", ()) or ()
    for segment in segments:
        start_h = _stage_value(segment, "start_h")
        end_h = _stage_value(segment, "end_h")
        if start_h <= float(elapsed_h) <= end_h:
            return max(0.0, _stage_value(segment, "rate_kg_s"))
    if segments:
        return 0.0
    fallback = _stage_optional(stage, "mean_rate_kg_s")
    if fallback is None:
        fallback = _stage_optional(stage, "peak_rate_kg_s")
    return None if fallback is None else max(0.0, float(fallback))


def _source_rate_scale(
    stage: Any,
    elapsed_h: float,
    *,
    enabled: bool,
    reference_rate_kg_s: float,
    exponent: float,
    minimum: float,
    maximum: float,
) -> float | None:
    if not enabled:
        return 1.0
    rate = _stage_rate_kg_s(stage, elapsed_h)
    if rate is None:
        return 1.0
    if rate <= 0.0:
        return None
    scale = (rate / float(reference_rate_kg_s)) ** float(exponent)
    return float(np.clip(scale, float(minimum), float(maximum)))


def _source_mask(
    shape: tuple[int, int],
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    elapsed_h: float,
    injection_stages: Sequence[Any],
    radius_x_m: float,
    radius_y_m: float,
    rate_scaled: bool,
    reference_rate_kg_s: float,
    rate_radius_exponent: float,
    minimum_rate_scale: float,
    maximum_rate_scale: float,
) -> tuple[np.ndarray, float]:
    source = np.zeros(shape, dtype=bool)
    largest_scale = 0.0
    for stage in injection_stages:
        if not (
            _stage_value(stage, "start_h")
            <= elapsed_h
            <= _stage_value(stage, "end_h")
        ):
            continue
        rate_scale = _source_rate_scale(
            stage,
            elapsed_h,
            enabled=rate_scaled,
            reference_rate_kg_s=reference_rate_kg_s,
            exponent=rate_radius_exponent,
            minimum=minimum_rate_scale,
            maximum=maximum_rate_scale,
        )
        if rate_scale is None:
            continue
        largest_scale = max(largest_scale, rate_scale)
        xx = (
            (x_m[None, :] - _stage_value(stage, "x_m"))
            / max(float(radius_x_m) * rate_scale, 1e-12)
        )
        yy = (
            (y_m[:, None] - _stage_value(stage, "y_m"))
            / max(float(radius_y_m) * rate_scale, 1e-12)
        )
        source |= xx * xx + yy * yy <= 1.0
    return source, largest_scale if largest_scale > 0.0 else 1.0


def _injection_is_active(
    elapsed_h: float,
    injection_stages: Sequence[Any],
) -> bool:
    for stage in injection_stages:
        if not (
            _stage_value(stage, "start_h")
            <= float(elapsed_h)
            <= _stage_value(stage, "end_h")
        ):
            continue
        rate = _stage_rate_kg_s(stage, elapsed_h)
        if rate is None or rate > 0.0:
            return True
    return False


def _dilate_physical(
    mask: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    radius_m: float,
) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    if not np.any(source) or radius_m <= 0.0:
        return source.copy()
    dx = float(width_m) / max(1, source.shape[1] - 1)
    dy = float(height_m) / max(1, source.shape[0] - 1)
    distance = distance_transform_edt(~source, sampling=(dy, dx))
    return distance <= float(radius_m)


def _smooth_gate(
    score: np.ndarray,
    *,
    onset: float,
    width: float,
) -> np.ndarray:
    if width <= 0.0:
        raise ValueError("Optical gate width must be positive.")
    unit = np.clip((np.asarray(score, dtype=np.float32) - onset) / width, 0.0, 1.0)
    return (unit * unit * (3.0 - 2.0 * unit)).astype(np.float32)


def _axis_spacing(axis: np.ndarray) -> float:
    differences = np.abs(np.diff(np.asarray(axis, dtype=float)))
    differences = differences[np.isfinite(differences) & (differences > 0.0)]
    if not differences.size:
        raise ValueError("Coordinate axis has no finite positive spacing.")
    return float(np.median(differences))


def _normalised_gaussian(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_x_pixels: float,
    sigma_y_pixels: float,
) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return np.zeros(np.asarray(values).shape, dtype=np.float32)
    weights = selected.astype(np.float32)
    numerator = cv2.GaussianBlur(
        np.asarray(values, dtype=np.float32) * weights,
        (0, 0),
        sigmaX=float(sigma_x_pixels),
        sigmaY=float(sigma_y_pixels),
        borderType=cv2.BORDER_REPLICATE,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=float(sigma_x_pixels),
        sigmaY=float(sigma_y_pixels),
        borderType=cv2.BORDER_REPLICATE,
    )
    result = np.zeros(np.asarray(values).shape, dtype=np.float32)
    result[selected] = numerator[selected] / np.maximum(
        denominator[selected], 1e-6
    )
    return result


def _smooth_optical_score(
    score: np.ndarray,
    *,
    valid: np.ndarray,
    seal: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    if sigma_m <= 0.0:
        return np.asarray(score, dtype=np.float32)
    sigma_x = float(sigma_m) / _axis_spacing(x_m)
    sigma_y = float(sigma_m) / _axis_spacing(y_m)
    output = np.zeros(np.asarray(score).shape, dtype=np.float32)
    masks = (valid,) if not np.any(seal) else (valid & ~seal, valid & seal)
    for mask in masks:
        smoothed = _normalised_gaussian(
            score,
            mask,
            sigma_x_pixels=sigma_x,
            sigma_y_pixels=sigma_y,
        )
        output[mask] = smoothed[mask]
    return output


def _smoothstep(values: np.ndarray) -> np.ndarray:
    unit = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.asarray(unit * unit * (3.0 - 2.0 * unit), dtype=np.float32)


def _co2_interior_affinity(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    support_level: float,
    scale_m: float,
) -> np.ndarray:
    """Return a soft distance-inside-plume score in physical units."""

    if not 0.0 < float(support_level) < 1.0:
        raise ValueError("co2_support_level must be in (0, 1).")
    if float(scale_m) <= 0.0:
        raise ValueError("co2_interior_scale_m must be positive.")
    support = np.asarray(valid, dtype=bool) & (
        np.asarray(score, dtype=np.float32) >= float(support_level)
    )
    if not np.any(support):
        return np.zeros(support.shape, dtype=np.float32)
    distance = distance_transform_edt(
        support,
        sampling=(_axis_spacing(y_m), _axis_spacing(x_m)),
    )
    return _smoothstep(distance / float(scale_m))


def _permeability_barrier_affinity(
    permeability: np.ndarray,
    valid: np.ndarray,
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Describe nearby overlying low-permeability, curved interfaces.

    The map is dimensionless and independent of facies IDs. It uses the
    direction and magnitude of the smoothed log-permeability gradient. Signed
    interface curvature is retained as a diagnostic, while its magnitude only
    broadens support; this avoids hard-coding one rig's anticline orientation.
    """

    if float(scale_m) <= 0.0:
        raise ValueError("permeability_interface_scale_m must be positive.")
    values = np.asarray(permeability, dtype=np.float64)
    selected = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values > 0.0)
    if values.shape != np.asarray(valid).shape:
        raise ValueError("permeability must match the gas field shape.")
    if not np.any(selected):
        zeros = np.zeros(values.shape, dtype=np.float32)
        return zeros, zeros

    positive = values[selected]
    floor = max(float(np.quantile(positive, 0.01)), np.finfo(float).tiny)
    log_k = np.log(np.where(selected, np.maximum(values, floor), floor))
    sigma_x = max(0.5, float(scale_m) / _axis_spacing(x_m))
    sigma_y = max(0.5, float(scale_m) / _axis_spacing(y_m))
    smooth = cv2.GaussianBlur(
        log_k.astype(np.float32),
        (0, 0),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float64)
    grad_y, grad_x = np.gradient(smooth, y_m, x_m)
    magnitude = np.hypot(grad_x, grad_y)
    finite_gradient = magnitude[selected & np.isfinite(magnitude)]
    if not finite_gradient.size or float(np.max(finite_gradient)) <= 0.0:
        zeros = np.zeros(values.shape, dtype=np.float32)
        return zeros, zeros
    reference = max(float(np.quantile(finite_gradient, 0.90)), np.finfo(float).eps)
    boundary_strength = np.clip(magnitude / reference, 0.0, 1.0)
    normal_x = grad_x / np.maximum(magnitude, np.finfo(float).eps)
    normal_y = grad_y / np.maximum(magnitude, np.finfo(float).eps)
    # The gradient points toward increasing permeability, so its opposite is
    # the direction toward the lower-permeability side of the interface.
    low_permeability_above = np.clip(-normal_y, 0.0, 1.0)
    dnx_dx = np.gradient(normal_x, x_m, axis=1)
    dny_dy = np.gradient(normal_y, y_m, axis=0)
    curvature = dnx_dx + dny_dy
    curvature_scaled = np.tanh(curvature * float(scale_m))
    seed = (
        boundary_strength
        * low_permeability_above
        * (0.75 + 0.25 * np.abs(curvature_scaled))
    )
    boundary = selected & (boundary_strength >= 0.15)
    if not np.any(boundary):
        zeros = np.zeros(values.shape, dtype=np.float32)
        return zeros, np.asarray(curvature_scaled, dtype=np.float32)
    distance, nearest = distance_transform_edt(
        ~boundary,
        sampling=(_axis_spacing(y_m), _axis_spacing(x_m)),
        return_indices=True,
    )
    affinity = seed[nearest[0], nearest[1]] * np.exp(
        -distance / float(scale_m)
    )
    affinity = np.where(selected, np.clip(affinity, 0.0, 1.0), 0.0)
    return (
        np.asarray(affinity, dtype=np.float32),
        np.asarray(curvature_scaled, dtype=np.float32),
    )


def _downward_reentry_mask(
    previous: np.ndarray,
    valid: np.ndarray,
    *,
    y_m: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    """Flag new support lying materially below its nearest prior gas support."""

    accepted = np.asarray(previous, dtype=bool)
    if not np.any(accepted):
        return np.zeros(accepted.shape, dtype=bool)
    _distance, nearest = distance_transform_edt(~accepted, return_indices=True)
    current_y = np.broadcast_to(np.asarray(y_m, dtype=float)[:, None], accepted.shape)
    nearest_y = np.asarray(y_m, dtype=float)[nearest[0]]
    return np.asarray(valid, dtype=bool) & (
        current_y < nearest_y - float(tolerance_m)
    )


def apply_causal_gas_gate(
    saturations: Sequence[np.ndarray],
    optical_scores: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    elapsed_hours: Sequence[float],
    injection_stages: Sequence[Any],
    facies_labels: np.ndarray | None,
    settings: CausalGasGateSettings,
    co2_scores: Sequence[np.ndarray] | None = None,
    permeability: np.ndarray | None = None,
) -> list[CausalGasGateFrame]:
    """Gate free gas using current optical evidence and causal plume history.

    The optical score is evaluated independently for every image. History can
    establish spatial continuity but can never preserve a component that lacks
    current-image evidence. New components are only seeded by an active source,
    with a narrow non-seal re-entry allowance for out-of-plane 3D transport.
    """

    if not saturations:
        return []
    if len(saturations) != len(optical_scores):
        raise ValueError("Saturation and optical-score sequence lengths differ.")
    if len(saturations) != len(elapsed_hours):
        raise ValueError("Saturation and elapsed-time sequence lengths differ.")
    if co2_scores is not None and len(co2_scores) != len(saturations):
        raise ValueError("CO2-score and saturation sequence lengths differ.")
    if not 0.0 < float(settings.structure_scale) <= 1.0:
        raise ValueError("structure_scale must be in (0, 1].")
    if settings.support_level <= 0.0:
        raise ValueError("support_level must be positive.")
    if settings.history_radius_m < 0.0 or settings.reentry_radius_m < 0.0:
        raise ValueError("Physical support radii cannot be negative.")
    if settings.optical_smoothing_sigma_m < 0.0:
        raise ValueError("optical_smoothing_sigma_m cannot be negative.")
    if settings.minimum_component_area_m2 < 0.0:
        raise ValueError("minimum_component_area_m2 cannot be negative.")
    if settings.reference_source_rate_kg_s <= 0.0:
        raise ValueError("reference_source_rate_kg_s must be positive.")
    if settings.source_rate_radius_exponent < 0.0:
        raise ValueError("source_rate_radius_exponent cannot be negative.")
    if not (
        0.0 < settings.minimum_source_rate_scale
        <= settings.maximum_source_rate_scale
    ):
        raise ValueError("Source-rate scales must be positive and ordered.")
    if not 0.0 <= float(settings.co2_boundary_floor) <= 1.0:
        raise ValueError("co2_boundary_floor must be in [0, 1].")
    if settings.post_injection_downward_tolerance_m < 0.0:
        raise ValueError("post-injection downward tolerance cannot be negative.")
    if not 0.0 <= float(settings.post_injection_downward_floor) <= 1.0:
        raise ValueError("post_injection_downward_floor must be in [0, 1].")
    times = np.asarray(elapsed_hours, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_hours must be finite and chronological.")

    shape = tuple(np.asarray(saturations[0]).shape)
    if len(shape) != 2:
        raise ValueError(f"Expected two-dimensional saturation fields, got {shape}.")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid_mask shape {valid.shape} != {shape}.")
    x_full = np.asarray(x_coordinates_m, dtype=float)
    y_full = np.asarray(y_coordinates_m, dtype=float)
    if len(x_full) != shape[1] or len(y_full) != shape[0]:
        raise ValueError("Coordinate axes do not match saturation field shape.")
    for index, (saturation, score) in enumerate(
        zip(saturations, optical_scores, strict=True)
    ):
        if np.asarray(saturation).shape != shape:
            raise ValueError(f"saturation[{index}] shape differs from {shape}.")
        if np.asarray(score).shape != shape:
            raise ValueError(f"optical_scores[{index}] shape differs from {shape}.")
    if facies_labels is not None and np.asarray(facies_labels).shape != shape:
        raise ValueError("facies_labels must match the saturation field shape.")
    if permeability is not None and np.asarray(permeability).shape != shape:
        raise ValueError("permeability must match the saturation field shape.")
    if settings.use_plume_interior_prior and co2_scores is None:
        raise ValueError("The plume-interior prior requires co2_scores.")
    if settings.use_permeability_structure_prior and permeability is None:
        raise ValueError("The permeability prior requires a permeability field.")

    target = (
        max(2, int(round(shape[1] * float(settings.structure_scale)))),
        max(2, int(round(shape[0] * float(settings.structure_scale)))),
    )
    small_shape = (target[1], target[0])
    valid_small = _resize_mean(valid.astype(np.float32), target) >= 0.75
    x_small = _resample_axis(x_full, target[0])
    y_small = _resample_axis(y_full, target[1])
    width_m = float(np.nanmax(x_full) - np.nanmin(x_full))
    height_m = float(np.nanmax(y_full) - np.nanmin(y_full))
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Physical coordinate spans must be positive.")

    if facies_labels is None:
        seal_full = np.zeros(shape, dtype=bool)
        seal_small = np.zeros(small_shape, dtype=bool)
    else:
        facies = np.asarray(facies_labels)
        seal_full = np.isin(facies, np.asarray(settings.seal_facies_ids))
        facies_small = cv2.resize(
            facies.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
        )
        seal_small = np.isin(
            facies_small,
            np.asarray(settings.seal_facies_ids, dtype=facies_small.dtype),
        )

    if permeability is None:
        permeability_small = None
        barrier_affinity = np.zeros(small_shape, dtype=np.float32)
    else:
        permeability_small = cv2.resize(
            np.asarray(permeability, dtype=np.float32),
            target,
            interpolation=cv2.INTER_AREA,
        )
        if settings.use_permeability_structure_prior:
            barrier_affinity, _interface_curvature = _permeability_barrier_affinity(
                permeability_small,
                valid_small,
                x_m=x_small,
                y_m=y_small,
                scale_m=float(settings.permeability_interface_scale_m),
            )
        else:
            barrier_affinity = np.zeros(small_shape, dtype=np.float32)

    accepted_previous = np.zeros(small_shape, dtype=bool)
    accepted_envelope = np.zeros(small_shape, dtype=bool)
    output: list[CausalGasGateFrame] = []
    resolved_co2_scores = (
        [None] * len(saturations)
        if co2_scores is None
        else [np.asarray(score, dtype=np.float32) for score in co2_scores]
    )
    for frame_index, (raw_saturation, raw_score, elapsed_h, raw_co2_score) in enumerate(
        zip(saturations, optical_scores, times, resolved_co2_scores, strict=True)
    ):
        saturation = np.clip(
            np.asarray(raw_saturation, dtype=np.float32), 0.0, 1.0
        )
        score = np.clip(np.asarray(raw_score, dtype=np.float32), 0.0, 1.0)
        optical_weight = _smooth_gate(
            score,
            onset=float(settings.optical_onset),
            width=float(settings.optical_width),
        )
        if np.any(seal_full):
            seal_weight = _smooth_gate(
                score,
                onset=float(settings.seal_optical_onset),
                width=float(settings.seal_optical_width),
            )
            optical_weight = np.where(seal_full, seal_weight, optical_weight)
        optical_weight = _smooth_optical_score(
            optical_weight,
            valid=valid,
            seal=seal_full,
            x_m=x_full,
            y_m=y_full,
            sigma_m=float(settings.optical_smoothing_sigma_m),
        )
        optical_weight = np.where(valid, optical_weight, 0.0).astype(np.float32)
        optically_gated = saturation * optical_weight

        saturation_small = _resize_mean(optically_gated, target)
        optical_small = _resize_mean(optical_weight, target)
        support = (
            valid_small
            & np.isfinite(saturation_small)
            & (saturation_small >= float(settings.support_level))
        )
        history_near = _dilate_physical(
            accepted_previous,
            width_m=width_m,
            height_m=height_m,
            radius_m=float(settings.history_radius_m),
        )
        envelope_near = _dilate_physical(
            accepted_envelope,
            width_m=width_m,
            height_m=height_m,
            radius_m=float(settings.reentry_radius_m),
        )
        active_source, active_source_radius_scale = _source_mask(
            small_shape,
            x_m=x_small,
            y_m=y_small,
            elapsed_h=float(elapsed_h),
            injection_stages=injection_stages,
            radius_x_m=float(settings.source_radius_x_m),
            radius_y_m=float(settings.source_radius_y_m),
            rate_scaled=bool(settings.use_rate_scaled_source_prior),
            reference_rate_kg_s=float(settings.reference_source_rate_kg_s),
            rate_radius_exponent=float(settings.source_rate_radius_exponent),
            minimum_rate_scale=float(settings.minimum_source_rate_scale),
            maximum_rate_scale=float(settings.maximum_source_rate_scale),
        )

        interior_affinity = np.ones(small_shape, dtype=np.float32)
        if raw_co2_score is not None:
            co2_small = _resize_mean(raw_co2_score, target)
            interior_affinity = _co2_interior_affinity(
                co2_small,
                valid_small,
                x_m=x_small,
                y_m=y_small,
                support_level=float(settings.co2_support_level),
                scale_m=float(settings.co2_interior_scale_m),
            )
        pre_physics_saturation_small = saturation_small.copy()
        if settings.use_plume_interior_prior:
            physical_support = np.maximum.reduce(
                (
                    interior_affinity,
                    barrier_affinity,
                    active_source.astype(np.float32),
                )
            )
            physics_gate = np.float32(settings.co2_boundary_floor) + (
                np.float32(1.0 - settings.co2_boundary_floor) * physical_support
            )
        else:
            physics_gate = np.ones(small_shape, dtype=np.float32)

        downward_penalty = np.zeros(small_shape, dtype=bool)
        if (
            settings.use_buoyancy_prior
            and np.any(accepted_previous)
            and not _injection_is_active(float(elapsed_h), injection_stages)
        ):
            downward_penalty = _downward_reentry_mask(
                accepted_previous,
                valid_small,
                y_m=y_small,
                tolerance_m=float(settings.post_injection_downward_tolerance_m),
            )
            downward_penalty &= ~active_source
            physics_gate[downward_penalty] *= np.float32(
                settings.post_injection_downward_floor
            )

        saturation_small *= physics_gate
        optical_small *= physics_gate
        support = (
            valid_small
            & np.isfinite(saturation_small)
            & (saturation_small >= float(settings.support_level))
        )

        count, component_labels, _stats, _centroids = (
            cv2.connectedComponentsWithStats(
                support.astype(np.uint8), connectivity=8
            )
        )
        retained = np.zeros_like(support)
        retained_components = 0
        rejected_components = 0
        source_components = 0
        history_components = 0
        reentry_components = 0
        rejected_seal_components = 0
        rejected_small_components = 0
        small_pixel_area_m2 = (
            width_m / max(1, support.shape[1] - 1)
        ) * (
            height_m / max(1, support.shape[0] - 1)
        )
        for component in range(1, int(count)):
            component_mask = component_labels == component
            optical_fraction = float(
                np.mean(optical_small[component_mask] >= 0.5)
            )
            seal_fraction = float(np.mean(seal_small[component_mask]))
            seal_dominant = seal_fraction >= float(settings.seal_component_fraction)
            source_supported = bool(np.any(component_mask & active_source)) and (
                optical_fraction >= float(settings.minimum_source_optical_fraction)
            )
            history_supported = bool(np.any(component_mask & history_near)) and (
                optical_fraction >= float(settings.minimum_history_optical_fraction)
            )
            reentry_supported = (
                bool(settings.allow_nonseal_reentry)
                and not seal_dominant
                and bool(np.any(component_mask & envelope_near))
                and optical_fraction
                >= float(settings.minimum_reentry_optical_fraction)
            )
            rejected_by_seal = bool(
                settings.reject_seal_dominant_components
                and seal_dominant
                and not source_supported
            )
            component_area_m2 = (
                int(np.count_nonzero(component_mask)) * small_pixel_area_m2
            )
            rejected_by_size = bool(
                component_area_m2 < float(settings.minimum_component_area_m2)
                and not source_supported
            )
            supported = (
                source_supported or history_supported or reentry_supported
            ) and not rejected_by_seal and not rejected_by_size
            if supported:
                retained[component_mask] = True
                retained_components += 1
                source_components += int(source_supported)
                history_components += int(history_supported)
                reentry_components += int(reentry_supported)
            else:
                rejected_components += 1
                rejected_seal_components += int(rejected_by_seal)
                rejected_small_components += int(rejected_by_size)

        accepted_previous = retained
        accepted_envelope |= retained
        plume_boundary = valid_small & (interior_affinity < 0.50)
        raw_small_sum = float(
            np.sum(pre_physics_saturation_small[valid_small], dtype=np.float64)
        )
        raw_boundary_sum = float(
            np.sum(
                pre_physics_saturation_small[plume_boundary],
                dtype=np.float64,
            )
        )
        retained_small = np.where(retained, saturation_small, 0.0)
        retained_small_sum = float(
            np.sum(retained_small[valid_small], dtype=np.float64)
        )
        retained_boundary_sum = float(
            np.sum(retained_small[plume_boundary], dtype=np.float64)
        )
        retained_full = cv2.resize(
            retained.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        physics_gate_full = cv2.resize(
            physics_gate,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        raw_support = valid & (saturation >= float(settings.support_level))
        optical_support = valid & (
            optically_gated >= float(settings.support_level)
        )
        filtered = np.where(
            retained_full,
            optically_gated * physics_gate_full,
            0.0,
        ).astype(np.float32)
        filtered = np.clip(filtered, 0.0, 1.0)
        retained_support = valid & (
            filtered >= float(settings.support_level)
        )
        raw_gas_sum = float(np.sum(saturation[valid], dtype=np.float64))
        retained_gas_sum = float(np.sum(filtered[valid], dtype=np.float64))
        output.append(
            CausalGasGateFrame(
                saturation=filtered,
                optical_weight=optical_weight,
                retained_support_mask=retained_support,
                removed_support_mask=raw_support & ~retained_support,
                component_count=max(0, int(count) - 1),
                retained_component_count=retained_components,
                rejected_component_count=rejected_components,
                source_supported_component_count=source_components,
                history_supported_component_count=history_components,
                reentry_supported_component_count=reentry_components,
                rejected_seal_component_count=rejected_seal_components,
                rejected_small_component_count=rejected_small_components,
                raw_support_pixels=int(np.count_nonzero(raw_support)),
                optically_supported_pixels=int(np.count_nonzero(optical_support)),
                retained_support_pixels=int(np.count_nonzero(retained_support)),
                removed_raw_gas_fraction=(
                    max(0.0, 1.0 - retained_gas_sum / raw_gas_sum)
                    if raw_gas_sum > 0.0
                    else 0.0
                ),
                plume_interior_gate_mean=float(
                    np.mean(interior_affinity[valid_small], dtype=np.float64)
                ),
                raw_gas_plume_boundary_fraction=(
                    raw_boundary_sum / raw_small_sum
                    if raw_small_sum > 0.0
                    else 0.0
                ),
                retained_gas_plume_boundary_fraction=(
                    retained_boundary_sum / retained_small_sum
                    if retained_small_sum > 0.0
                    else 0.0
                ),
                permeability_barrier_affinity_mean=float(
                    np.mean(barrier_affinity[valid_small], dtype=np.float64)
                ),
                downward_penalty_pixel_fraction=float(
                    np.mean(downward_penalty[valid_small], dtype=np.float64)
                ),
                active_source_radius_scale=float(active_source_radius_scale),
            )
        )
    return output


def apply_gas_mass_caps(
    saturations: Sequence[np.ndarray],
    cap_masses_kg: Sequence[float],
    *,
    density_g_kg_m3: np.ndarray,
    voxel_volume_m3: np.ndarray,
) -> list[GasMassCapFrame]:
    """Scale each saturation field down to an independently defined gas cap."""

    if len(saturations) != len(cap_masses_kg):
        raise ValueError("Saturation and gas-cap sequence lengths differ.")
    if not saturations:
        return []
    shape = tuple(np.asarray(saturations[0]).shape)
    density = np.broadcast_to(
        np.asarray(density_g_kg_m3, dtype=np.float64), shape
    )
    volume = np.broadcast_to(
        np.asarray(voxel_volume_m3, dtype=np.float64), shape
    )
    output: list[GasMassCapFrame] = []
    for index, (raw, raw_cap) in enumerate(
        zip(saturations, cap_masses_kg, strict=True)
    ):
        saturation = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
        if saturation.shape != shape:
            raise ValueError(f"saturation[{index}] shape differs from {shape}.")
        cap = float(raw_cap)
        if not np.isfinite(cap) or cap < 0.0:
            raise ValueError(f"Invalid gas cap at frame {index}: {raw_cap!r}.")
        mass = float(
            np.sum(density * saturation * volume, dtype=np.float64)
        )
        scale = min(1.0, cap / mass) if mass > 0.0 else 1.0
        capped = (saturation * np.float32(scale)).astype(np.float32)
        capped_mass = float(
            np.sum(density * capped * volume, dtype=np.float64)
        )
        output.append(
            GasMassCapFrame(
                saturation=capped,
                uncapped_mass_kg=mass,
                capped_mass_kg=capped_mass,
                cap_mass_kg=cap,
                scale=float(scale),
            )
        )
    return output
