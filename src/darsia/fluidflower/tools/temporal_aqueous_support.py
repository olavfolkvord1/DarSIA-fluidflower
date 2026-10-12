"""Spatiotemporal support gate for weak dissolved-CO2 components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class TemporalGateSettings:
    trace_level: float = 0.01
    core_level: float = 0.05
    confidence_level: float = 0.50
    minimum_observed_fraction: float = 0.25
    minimum_temporal_fraction: float = 0.10
    support_radius_m: float = 0.005
    structure_scale: float = 0.20


@dataclass(frozen=True)
class TemporalGateFrame:
    concentration: np.ndarray
    retained_trace_mask: np.ndarray
    removed_trace_mask: np.ndarray
    component_count: int
    retained_component_count: int
    rejected_component_count: int
    removed_trace_fraction: float


@dataclass(frozen=True)
class CausalGateSettings:
    """Settings for source- and history-anchored aqueous support."""

    trace_level: float = 0.01
    core_level: float = 0.05
    confidence_level: float = 0.50
    minimum_observed_fraction: float = 0.25
    history_radius_m: float = 0.005
    reentry_radius_m: float = 0.020
    minimum_reentry_observed_fraction: float = 0.75
    minimum_reentry_core_fraction: float = 0.10
    source_radius_x_m: float = 0.040
    source_radius_y_m: float = 0.030
    structure_scale: float = 0.20
    seal_facies_ids: tuple[int, ...] = (1,)
    seal_component_fraction: float = 0.50
    allow_nonseal_reentry: bool = True


@dataclass(frozen=True)
class CausalGateFrame:
    concentration: np.ndarray
    retained_trace_mask: np.ndarray
    removed_trace_mask: np.ndarray
    component_count: int
    retained_component_count: int
    rejected_component_count: int
    removed_trace_fraction: float
    source_supported_component_count: int
    history_supported_component_count: int
    reentry_supported_component_count: int
    rejected_seal_component_count: int
    removed_core_fraction: float
    removed_seal_support_fraction: float


@dataclass(frozen=True)
class DiffusiveGateSettings:
    """Settings for cadence-independent aqueous plume reachability.

    ``effective_diffusivity_m2_s`` is an effective image-scale spreading
    coefficient. It includes molecular diffusion, unresolved pore-scale
    dispersion and indicator response, and must not be interpreted as a direct
    measurement of the molecular CO2 diffusivity.
    """

    trace_level: float = 0.01
    core_level: float = 0.05
    confidence_level: float = 0.50
    minimum_observed_fraction: float = 0.25
    effective_diffusivity_m2_s: float = 1.0e-9
    diffusion_radius_multiplier: float = 1.0
    registration_tolerance_m: float = 0.001
    maximum_history_radius_m: float = 0.030
    minimum_reentry_radius_m: float = 0.020
    reentry_radius_multiplier: float = 2.0
    maximum_reentry_radius_m: float = 0.040
    minimum_reentry_observed_fraction: float = 0.75
    minimum_reentry_core_fraction: float = 0.10
    source_radius_x_m: float = 0.040
    source_radius_y_m: float = 0.030
    structure_scale: float = 0.20
    seal_facies_ids: tuple[int, ...] = (1,)
    seal_component_fraction: float = 0.50
    allow_nonseal_reentry: bool = True
    preserve_observed_above_level: float | None = None
    soft_transition_width_m: float = 0.0


@dataclass(frozen=True)
class DiffusiveGateFrame:
    concentration: np.ndarray
    attenuation: np.ndarray
    retained_trace_mask: np.ndarray
    removed_trace_mask: np.ndarray
    component_count: int
    retained_component_count: int
    rejected_component_count: int
    removed_trace_fraction: float
    source_supported_component_count: int
    history_supported_component_count: int
    reentry_supported_component_count: int
    rejected_seal_component_count: int
    removed_core_fraction: float
    removed_seal_support_fraction: float
    partially_clipped_component_count: int
    trusted_observed_component_count: int
    elapsed_delta_h: float
    history_radius_m: float
    reentry_radius_m: float
    mean_weak_attenuation: float


@dataclass(frozen=True)
class HardSealContactFrame:
    """Phase fields after enforcing hard seal-contact constraints."""

    concentration: np.ndarray
    saturation: np.ndarray
    retained_seal_aqueous_mask: np.ndarray
    removed_seal_aqueous_mask: np.ndarray
    removed_seal_gas_mask: np.ndarray
    seal_aqueous_component_count: int
    retained_seal_aqueous_component_count: int
    removed_seal_aqueous_component_count: int
    removed_seal_aqueous_pixel_fraction: float
    removed_seal_gas_fraction: float


def apply_hard_seal_contact_constraint(
    concentrations: Sequence[np.ndarray],
    saturations: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    facies_labels: np.ndarray,
    *,
    seal_facies_ids: Sequence[int] = (1,),
    aqueous_trace_level: float = 0.01,
    gas_support_level: float = 0.05,
) -> list[HardSealContactFrame]:
    """Reject free gas and disconnected dissolved CO2 inside sealing layers.

    Free gas is always set to zero in the specified seal facies. Dissolved CO2
    at or above ``aqueous_trace_level`` is retained in the seal only when its
    current-frame, 8-connected total-CO2 component also contains support in a
    non-seal facies. Sub-trace aqueous values in the seal are removed because
    they cannot establish an observable connection. Values outside the seal
    are left unchanged, and removed mass is not reassigned to another phase.
    """

    if not concentrations:
        return []
    if len(concentrations) != len(saturations):
        raise ValueError("Concentration and saturation sequence lengths differ.")
    if aqueous_trace_level <= 0.0 or gas_support_level <= 0.0:
        raise ValueError("Phase support levels must be positive.")

    shape = tuple(np.asarray(concentrations[0]).shape)
    if len(shape) != 2:
        raise ValueError(f"Expected two-dimensional phase fields, got {shape}.")
    valid = np.asarray(valid_mask, dtype=bool)
    facies = np.asarray(facies_labels)
    if valid.shape != shape or facies.shape != shape:
        raise ValueError("Valid mask and facies labels must match the phase fields.")
    seal = valid & np.isin(facies, np.asarray(tuple(seal_facies_ids)))
    nonseal = valid & ~seal

    output: list[HardSealContactFrame] = []
    for index, (raw_concentration, raw_saturation) in enumerate(
        zip(concentrations, saturations, strict=True)
    ):
        concentration = np.asarray(raw_concentration, dtype=np.float32)
        saturation = np.asarray(raw_saturation, dtype=np.float32)
        if concentration.shape != shape or saturation.shape != shape:
            raise ValueError(f"Phase field {index} does not match {shape}.")

        finite_aqueous = np.where(
            np.isfinite(concentration), concentration, 0.0
        ).astype(np.float32)
        finite_gas = np.clip(
            np.where(np.isfinite(saturation), saturation, 0.0), 0.0, 1.0
        ).astype(np.float32)
        gas_sum_in_seal = float(np.sum(finite_gas[seal], dtype=np.float64))
        removed_gas = seal & (finite_gas > 0.0)
        filtered_gas = np.array(finite_gas, copy=True)
        filtered_gas[seal] = 0.0

        aqueous_support = (
            valid
            & (finite_aqueous >= float(aqueous_trace_level))
        )
        gas_support = (
            valid
            & (filtered_gas >= float(gas_support_level))
        )
        total_support = aqueous_support | gas_support
        count, components = cv2.connectedComponents(
            total_support.astype(np.uint8), connectivity=8
        )
        nonseal_component_ids = np.unique(components[total_support & nonseal])
        nonseal_component_ids = nonseal_component_ids[nonseal_component_ids != 0]
        connected_to_nonseal = np.isin(components, nonseal_component_ids)

        seal_aqueous_support = seal & aqueous_support
        retained_seal = seal_aqueous_support & connected_to_nonseal
        isolated_seal = seal_aqueous_support & ~connected_to_nonseal
        subtrace_seal = (
            seal
            & (finite_aqueous > 0.0)
            & ~aqueous_support
        )
        removed_aqueous = isolated_seal | subtrace_seal
        filtered_aqueous = np.array(finite_aqueous, copy=True)
        filtered_aqueous[removed_aqueous] = 0.0

        seal_component_ids = np.unique(components[seal_aqueous_support])
        seal_component_ids = seal_component_ids[seal_component_ids != 0]
        retained_component_ids = np.intersect1d(
            seal_component_ids,
            nonseal_component_ids,
            assume_unique=True,
        )
        seal_signal_count = int(np.count_nonzero(seal & (finite_aqueous > 0.0)))
        output.append(
            HardSealContactFrame(
                concentration=filtered_aqueous,
                saturation=filtered_gas,
                retained_seal_aqueous_mask=retained_seal,
                removed_seal_aqueous_mask=removed_aqueous,
                removed_seal_gas_mask=removed_gas,
                seal_aqueous_component_count=int(len(seal_component_ids)),
                retained_seal_aqueous_component_count=int(
                    len(retained_component_ids)
                ),
                removed_seal_aqueous_component_count=int(
                    len(seal_component_ids) - len(retained_component_ids)
                ),
                removed_seal_aqueous_pixel_fraction=(
                    float(np.count_nonzero(removed_aqueous) / seal_signal_count)
                    if seal_signal_count
                    else 0.0
                ),
                removed_seal_gas_fraction=(
                    float(np.sum(finite_gas[removed_gas], dtype=np.float64))
                    / gas_sum_in_seal
                    if gas_sum_in_seal > 0.0
                    else 0.0
                ),
            )
        )
    return output


def _resize_mean(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(values, dtype=np.float32),
        target,
        interpolation=cv2.INTER_AREA,
    )


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


def diffusive_reach_radius_m(
    elapsed_delta_h: float,
    effective_diffusivity_m2_s: float,
    *,
    radius_multiplier: float = 1.0,
    registration_tolerance_m: float = 0.0,
    maximum_radius_m: float | None = None,
) -> float:
    """Return radial 2-D diffusion reach for an actual frame interval.

    The diffusion contribution is ``multiplier * sqrt(4 D dt)``. Independent
    registration uncertainty is added in quadrature. This makes the gate
    invariant to whether a time interval is represented by one or many frames.
    """

    values = (
        elapsed_delta_h,
        effective_diffusivity_m2_s,
        radius_multiplier,
        registration_tolerance_m,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Diffusive reach inputs must be finite.")
    if elapsed_delta_h < 0.0:
        raise ValueError("elapsed_delta_h cannot be negative.")
    if effective_diffusivity_m2_s < 0.0:
        raise ValueError("effective_diffusivity_m2_s cannot be negative.")
    if radius_multiplier < 0.0 or registration_tolerance_m < 0.0:
        raise ValueError("Radius multiplier and tolerance cannot be negative.")
    if maximum_radius_m is not None and maximum_radius_m < 0.0:
        raise ValueError("maximum_radius_m cannot be negative.")

    diffusion = float(radius_multiplier) * np.sqrt(
        4.0
        * float(effective_diffusivity_m2_s)
        * float(elapsed_delta_h)
        * 3600.0
    )
    radius = float(np.hypot(float(registration_tolerance_m), diffusion))
    if maximum_radius_m is not None:
        radius = min(radius, float(maximum_radius_m))
    return radius


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


def _source_mask(
    shape: tuple[int, int],
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    elapsed_h: float,
    injection_stages: Sequence[Any],
    radius_x_m: float,
    radius_y_m: float,
) -> np.ndarray:
    source = np.zeros(shape, dtype=bool)
    for stage in injection_stages:
        start_h = _stage_value(stage, "start_h")
        end_h = _stage_value(stage, "end_h")
        if not start_h <= elapsed_h <= end_h:
            continue
        stage_x = _stage_value(stage, "x_m")
        stage_y = _stage_value(stage, "y_m")
        xx = (x_m[None, :] - stage_x) / max(float(radius_x_m), 1e-12)
        yy = (y_m[:, None] - stage_y) / max(float(radius_y_m), 1e-12)
        source |= xx * xx + yy * yy <= 1.0
    return source


def apply_causal_aqueous_gate(
    concentrations: Sequence[np.ndarray],
    observation_confidences: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    elapsed_hours: Sequence[float],
    injection_stages: Sequence[Any],
    facies_labels: np.ndarray | None,
    settings: CausalGateSettings,
) -> list[CausalGateFrame]:
    """Retain aqueous components anchored to a source or accepted plume history.

    Frames are processed chronologically. During active injection, a new
    component may be seeded close to the active injector. At all later times a
    component is retained when it intersects the dilated support accepted in
    the preceding frame. A high-confidence non-seal component may also re-enter
    close to the cumulative accepted envelope; this is the explicit allowance
    for CO2 moving into and out of the camera-visible slice of the 3D volume.

    Raw neighbouring frames are deliberately not used as evidence. Persistent
    optical artefacts therefore cannot validate one another. Components mostly
    inside the low-permeability seal do not receive the re-entry exception, but
    remain allowed when they grow continuously from accepted support.
    """

    if not concentrations:
        return []
    if len(concentrations) != len(observation_confidences):
        raise ValueError("Concentration and confidence sequence lengths differ.")
    if len(concentrations) != len(elapsed_hours):
        raise ValueError("Concentration and elapsed-time sequence lengths differ.")
    if not 0.0 < float(settings.structure_scale) <= 1.0:
        raise ValueError("structure_scale must be in (0, 1].")
    if not 0.0 < float(settings.trace_level) < float(settings.core_level):
        raise ValueError("Expected 0 < trace_level < core_level.")
    if settings.history_radius_m < 0.0 or settings.reentry_radius_m < 0.0:
        raise ValueError("Physical support radii cannot be negative.")
    times = np.asarray(elapsed_hours, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_hours must be finite and chronological.")

    shape = tuple(np.asarray(concentrations[0]).shape)
    if len(shape) != 2:
        raise ValueError(f"Expected two-dimensional concentration fields, got {shape}.")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid_mask shape {valid.shape} != {shape}.")
    x_full = np.asarray(x_coordinates_m, dtype=float)
    y_full = np.asarray(y_coordinates_m, dtype=float)
    if len(x_full) != shape[1] or len(y_full) != shape[0]:
        raise ValueError("Coordinate axes do not match concentration field shape.")
    for index, (concentration, confidence) in enumerate(
        zip(concentrations, observation_confidences, strict=True)
    ):
        if np.asarray(concentration).shape != shape:
            raise ValueError(
                f"concentration[{index}] shape {np.asarray(concentration).shape} "
                f"!= {shape}."
            )
        if np.asarray(confidence).shape != shape:
            raise ValueError(
                f"confidence[{index}] shape {np.asarray(confidence).shape} "
                f"!= {shape}."
            )
    if facies_labels is not None and np.asarray(facies_labels).shape != shape:
        raise ValueError("facies_labels must match the concentration field shape.")

    target = (
        max(2, int(round(shape[1] * float(settings.structure_scale)))),
        max(2, int(round(shape[0] * float(settings.structure_scale)))),
    )
    small_shape = (target[1], target[0])
    valid_small = _resize_mean(valid.astype(np.float32), target) >= 0.75
    concentration_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in concentrations
    ]
    confidence_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in observation_confidences
    ]
    x_small = _resample_axis(x_full, target[0])
    y_small = _resample_axis(y_full, target[1])
    width_m = float(np.nanmax(x_full) - np.nanmin(x_full))
    height_m = float(np.nanmax(y_full) - np.nanmin(y_full))
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Physical coordinate spans must be positive.")
    if facies_labels is None:
        seal_small = np.zeros(small_shape, dtype=bool)
        seal_full = np.zeros(shape, dtype=bool)
    else:
        facies = np.asarray(facies_labels)
        seal_ids = np.asarray(settings.seal_facies_ids, dtype=facies.dtype)
        seal_full = np.isin(facies, seal_ids)
        facies_small = cv2.resize(
            facies.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
        )
        seal_small = np.isin(facies_small, np.asarray(settings.seal_facies_ids))

    accepted_previous = np.zeros(small_shape, dtype=bool)
    accepted_envelope = np.zeros(small_shape, dtype=bool)
    output: list[CausalGateFrame] = []
    for frame_index, (source, confidence, elapsed_h) in enumerate(
        zip(concentration_small, confidence_small, times, strict=True)
    ):
        support = valid_small & np.isfinite(source) & (
            source >= float(settings.trace_level)
        )
        core = support & (source >= float(settings.core_level))
        history_near = _dilate_physical(
            accepted_previous,
            width_m=width_m,
            height_m=height_m,
            radius_m=settings.history_radius_m,
        )
        envelope_near = _dilate_physical(
            accepted_envelope,
            width_m=width_m,
            height_m=height_m,
            radius_m=settings.reentry_radius_m,
        )
        active_source = _source_mask(
            small_shape,
            x_m=x_small,
            y_m=y_small,
            elapsed_h=float(elapsed_h),
            injection_stages=injection_stages,
            radius_x_m=settings.source_radius_x_m,
            radius_y_m=settings.source_radius_y_m,
        )

        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            support.astype(np.uint8), connectivity=8
        )
        retained = np.zeros_like(support)
        retained_components = 0
        rejected_components = 0
        source_components = 0
        history_components = 0
        reentry_components = 0
        rejected_seal_components = 0
        for component in range(1, int(count)):
            component_mask = labels == component
            observed_fraction = float(
                np.mean(
                    confidence[component_mask]
                    >= float(settings.confidence_level)
                )
            )
            core_fraction = float(np.mean(core[component_mask]))
            seal_fraction = float(np.mean(seal_small[component_mask]))
            source_supported = bool(np.any(component_mask & active_source)) and (
                observed_fraction >= float(settings.minimum_observed_fraction)
                or core_fraction >= float(settings.minimum_reentry_core_fraction)
            )
            history_supported = bool(np.any(component_mask & history_near))
            reentry_supported = (
                bool(settings.allow_nonseal_reentry)
                and seal_fraction < float(settings.seal_component_fraction)
                and bool(np.any(component_mask & envelope_near))
                and observed_fraction
                >= float(settings.minimum_reentry_observed_fraction)
                and core_fraction >= float(settings.minimum_reentry_core_fraction)
            )
            supported = source_supported or history_supported or reentry_supported
            if supported:
                retained[component_mask] = True
                retained_components += 1
                source_components += int(source_supported)
                history_components += int(history_supported)
                reentry_components += int(reentry_supported)
            else:
                rejected_components += 1
                rejected_seal_components += int(
                    seal_fraction >= float(settings.seal_component_fraction)
                )

        accepted_previous = retained
        accepted_envelope |= retained
        retained_full = cv2.resize(
            retained.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        concentration = np.asarray(concentrations[frame_index], dtype=np.float32)
        support_full = (
            valid
            & np.isfinite(concentration)
            & (concentration >= float(settings.trace_level))
        )
        core_full = support_full & (
            concentration >= float(settings.core_level)
        )
        removed = support_full & ~retained_full
        filtered = np.array(concentration, copy=True)
        filtered[removed] = 0.0
        support_count = int(np.count_nonzero(support_full))
        core_count = int(np.count_nonzero(core_full))
        seal_support_count = int(np.count_nonzero(support_full & seal_full))
        output.append(
            CausalGateFrame(
                concentration=filtered,
                retained_trace_mask=support_full & ~removed,
                removed_trace_mask=removed,
                component_count=max(0, int(count) - 1),
                retained_component_count=retained_components,
                rejected_component_count=rejected_components,
                removed_trace_fraction=(
                    float(np.count_nonzero(removed) / support_count)
                    if support_count
                    else 0.0
                ),
                source_supported_component_count=source_components,
                history_supported_component_count=history_components,
                reentry_supported_component_count=reentry_components,
                rejected_seal_component_count=rejected_seal_components,
                removed_core_fraction=(
                    float(np.count_nonzero(removed & core_full) / core_count)
                    if core_count
                    else 0.0
                ),
                removed_seal_support_fraction=(
                    float(
                        np.count_nonzero(removed & seal_full)
                        / seal_support_count
                    )
                    if seal_support_count
                    else 0.0
                ),
            )
        )
    return output


def apply_diffusive_aqueous_gate(
    concentrations: Sequence[np.ndarray],
    observation_confidences: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    x_coordinates_m: np.ndarray,
    y_coordinates_m: np.ndarray,
    elapsed_hours: Sequence[float],
    injection_stages: Sequence[Any],
    facies_labels: np.ndarray | None,
    settings: DiffusiveGateSettings,
) -> list[DiffusiveGateFrame]:
    """Apply a causal, cadence-aware reachability gate to dissolved CO2.

    Source-connected components are retained during active injection. Outside
    active injection, only the part of a component reachable from the previous
    accepted plume within ``sqrt(4 D_eff dt)`` is retained. This pixel-wise
    clipping is intentionally stricter than :func:`apply_causal_aqueous_gate`,
    where a single history contact retains an entire connected component.

    A strongly observed non-seal component may re-enter close to the cumulative
    accepted envelope to account for transport into and out of the visible
    camera slice. Removed concentration is not reassigned or mass-rescaled.
    """

    if not concentrations:
        return []
    if len(concentrations) != len(observation_confidences):
        raise ValueError("Concentration and confidence sequence lengths differ.")
    if len(concentrations) != len(elapsed_hours):
        raise ValueError("Concentration and elapsed-time sequence lengths differ.")
    if not 0.0 < float(settings.structure_scale) <= 1.0:
        raise ValueError("structure_scale must be in (0, 1].")
    if not 0.0 < float(settings.trace_level) < float(settings.core_level):
        raise ValueError("Expected 0 < trace_level < core_level.")
    if (
        settings.preserve_observed_above_level is not None
        and float(settings.preserve_observed_above_level)
        < float(settings.trace_level)
    ):
        raise ValueError("preserve_observed_above_level cannot be below trace_level.")
    if settings.maximum_history_radius_m < 0.0:
        raise ValueError("maximum_history_radius_m cannot be negative.")
    if settings.soft_transition_width_m < 0.0:
        raise ValueError("soft_transition_width_m cannot be negative.")
    if not (
        0.0 <= settings.minimum_reentry_radius_m
        <= settings.maximum_reentry_radius_m
    ):
        raise ValueError("Invalid re-entry radius bounds.")
    times = np.asarray(elapsed_hours, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("elapsed_hours must be finite and chronological.")

    shape = tuple(np.asarray(concentrations[0]).shape)
    if len(shape) != 2:
        raise ValueError(f"Expected two-dimensional concentration fields, got {shape}.")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid_mask shape {valid.shape} != {shape}.")
    x_full = np.asarray(x_coordinates_m, dtype=float)
    y_full = np.asarray(y_coordinates_m, dtype=float)
    if len(x_full) != shape[1] or len(y_full) != shape[0]:
        raise ValueError("Coordinate axes do not match concentration field shape.")
    for index, (concentration, confidence) in enumerate(
        zip(concentrations, observation_confidences, strict=True)
    ):
        if np.asarray(concentration).shape != shape:
            raise ValueError(
                f"concentration[{index}] shape {np.asarray(concentration).shape} "
                f"!= {shape}."
            )
        if np.asarray(confidence).shape != shape:
            raise ValueError(
                f"confidence[{index}] shape {np.asarray(confidence).shape} "
                f"!= {shape}."
            )
    if facies_labels is not None and np.asarray(facies_labels).shape != shape:
        raise ValueError("facies_labels must match the concentration field shape.")

    target = (
        max(2, int(round(shape[1] * float(settings.structure_scale)))),
        max(2, int(round(shape[0] * float(settings.structure_scale)))),
    )
    small_shape = (target[1], target[0])
    valid_small = _resize_mean(valid.astype(np.float32), target) >= 0.75
    concentration_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in concentrations
    ]
    confidence_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in observation_confidences
    ]
    x_small = _resample_axis(x_full, target[0])
    y_small = _resample_axis(y_full, target[1])
    width_m = float(np.nanmax(x_full) - np.nanmin(x_full))
    height_m = float(np.nanmax(y_full) - np.nanmin(y_full))
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Physical coordinate spans must be positive.")
    if facies_labels is None:
        seal_small = np.zeros(small_shape, dtype=bool)
        seal_full = np.zeros(shape, dtype=bool)
    else:
        facies = np.asarray(facies_labels)
        seal_ids = np.asarray(settings.seal_facies_ids, dtype=facies.dtype)
        seal_full = np.isin(facies, seal_ids)
        facies_small = cv2.resize(
            facies.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
        )
        seal_small = np.isin(facies_small, np.asarray(settings.seal_facies_ids))

    accepted_previous = np.zeros(small_shape, dtype=bool)
    accepted_envelope = np.zeros(small_shape, dtype=bool)
    output: list[DiffusiveGateFrame] = []
    for frame_index, (source, confidence, elapsed_h) in enumerate(
        zip(concentration_small, confidence_small, times, strict=True)
    ):
        elapsed_delta_h = (
            0.0 if frame_index == 0 else float(elapsed_h - times[frame_index - 1])
        )
        history_radius_m = diffusive_reach_radius_m(
            elapsed_delta_h,
            settings.effective_diffusivity_m2_s,
            radius_multiplier=settings.diffusion_radius_multiplier,
            registration_tolerance_m=settings.registration_tolerance_m,
            maximum_radius_m=settings.maximum_history_radius_m,
        )
        reentry_radius_m = min(
            float(settings.maximum_reentry_radius_m),
            max(
                float(settings.minimum_reentry_radius_m),
                float(settings.reentry_radius_multiplier) * history_radius_m,
            ),
        )
        support = valid_small & np.isfinite(source) & (
            source >= float(settings.trace_level)
        )
        core = support & (source >= float(settings.core_level))
        history_near = _dilate_physical(
            accepted_previous,
            width_m=width_m,
            height_m=height_m,
            radius_m=history_radius_m,
        )
        envelope_near = _dilate_physical(
            accepted_envelope,
            width_m=width_m,
            height_m=height_m,
            radius_m=reentry_radius_m,
        )
        active_source = _source_mask(
            small_shape,
            x_m=x_small,
            y_m=y_small,
            elapsed_h=float(elapsed_h),
            injection_stages=injection_stages,
            radius_x_m=settings.source_radius_x_m,
            radius_y_m=settings.source_radius_y_m,
        )

        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            support.astype(np.uint8), connectivity=8
        )
        causal_retained = np.zeros_like(support)
        output_retained = np.zeros_like(support)
        trusted_observed = (
            support
            & (source >= float(settings.preserve_observed_above_level))
            & (confidence >= float(settings.confidence_level))
            if settings.preserve_observed_above_level is not None
            else np.zeros_like(support)
        )
        retained_components = 0
        rejected_components = 0
        partially_clipped_components = 0
        trusted_components = 0
        source_components = 0
        history_components = 0
        reentry_components = 0
        rejected_seal_components = 0
        for component in range(1, int(count)):
            component_mask = labels == component
            component_size = int(np.count_nonzero(component_mask))
            observed_fraction = float(
                np.mean(
                    confidence[component_mask]
                    >= float(settings.confidence_level)
                )
            )
            core_fraction = float(np.mean(core[component_mask]))
            seal_fraction = float(np.mean(seal_small[component_mask]))
            source_supported = bool(np.any(component_mask & active_source)) and (
                observed_fraction >= float(settings.minimum_observed_fraction)
                or core_fraction >= float(settings.minimum_reentry_core_fraction)
            )
            history_pixels = component_mask & history_near
            history_supported = bool(np.any(history_pixels))
            reentry_supported = (
                bool(settings.allow_nonseal_reentry)
                and seal_fraction < float(settings.seal_component_fraction)
                and bool(np.any(component_mask & envelope_near))
                and observed_fraction
                >= float(settings.minimum_reentry_observed_fraction)
                and core_fraction >= float(settings.minimum_reentry_core_fraction)
            )

            if source_supported:
                causal_pixels = component_mask
            else:
                causal_pixels = history_pixels.copy()
                if reentry_supported:
                    causal_pixels |= component_mask & envelope_near

            trusted_pixels = component_mask & trusted_observed
            retained_pixels = causal_pixels | trusted_pixels

            retained_size = int(np.count_nonzero(retained_pixels))
            if retained_size:
                causal_retained[causal_pixels] = True
                output_retained[retained_pixels] = True
                retained_components += 1
                source_components += int(source_supported)
                history_components += int(history_supported)
                reentry_components += int(reentry_supported)
                trusted_components += int(np.any(trusted_pixels))
                partially_clipped_components += int(retained_size < component_size)
            else:
                rejected_components += 1
                rejected_seal_components += int(
                    seal_fraction >= float(settings.seal_component_fraction)
                )

        # A strong but unreachable current observation remains visible in the
        # output, but cannot by itself seed weak support in a later frame.
        accepted_previous = causal_retained
        accepted_envelope |= causal_retained
        retained_full = cv2.resize(
            output_retained.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        concentration = np.asarray(concentrations[frame_index], dtype=np.float32)
        confidence_full = np.asarray(
            observation_confidences[frame_index], dtype=np.float32
        )
        support_full = (
            valid
            & np.isfinite(concentration)
            & (concentration >= float(settings.trace_level))
        )
        core_full = support_full & (
            concentration >= float(settings.core_level)
        )
        if settings.preserve_observed_above_level is not None:
            retained_full |= (
                support_full
                & (
                    concentration
                    >= float(settings.preserve_observed_above_level)
                )
                & (confidence_full >= float(settings.confidence_level))
            )
        if settings.soft_transition_width_m > 0.0:
            dx = width_m / max(1, shape[1] - 1)
            dy = height_m / max(1, shape[0] - 1)
            distance = distance_transform_edt(
                ~retained_full,
                sampling=(dy, dx),
            )
            spatial_weight = np.clip(
                1.0 - distance / float(settings.soft_transition_width_m),
                0.0,
                1.0,
            ).astype(np.float32)
            if settings.preserve_observed_above_level is None:
                weak_blend = np.ones(shape, dtype=np.float32)
            else:
                weak_blend = (
                    concentration
                    < float(settings.preserve_observed_above_level)
                ).astype(np.float32)
            attenuation = np.ones(shape, dtype=np.float32)
            attenuation[valid] = (
                1.0
                - weak_blend[valid]
                * (1.0 - spatial_weight[valid])
            )
            filtered = concentration * attenuation
            removed = support_full & (attenuation < 1.0 - 1e-6)
        else:
            attenuation = np.ones(shape, dtype=np.float32)
            attenuation[valid & ~retained_full] = 0.0
            removed = support_full & ~retained_full
            filtered = np.array(concentration, copy=True)
            filtered[valid & ~retained_full] = 0.0
        support_count = int(np.count_nonzero(support_full))
        core_count = int(np.count_nonzero(core_full))
        seal_support_count = int(np.count_nonzero(support_full & seal_full))
        weak_limit = float(
            settings.preserve_observed_above_level
            if settings.preserve_observed_above_level is not None
            else settings.core_level
        )
        weak_signal = valid & (concentration < weak_limit)
        mean_weak_attenuation = (
            float(np.mean(attenuation[weak_signal]))
            if np.any(weak_signal)
            else 1.0
        )
        output.append(
            DiffusiveGateFrame(
                concentration=filtered,
                attenuation=attenuation,
                retained_trace_mask=support_full & ~removed,
                removed_trace_mask=removed,
                component_count=max(0, int(count) - 1),
                retained_component_count=retained_components,
                rejected_component_count=rejected_components,
                removed_trace_fraction=(
                    float(np.count_nonzero(removed) / support_count)
                    if support_count
                    else 0.0
                ),
                source_supported_component_count=source_components,
                history_supported_component_count=history_components,
                reentry_supported_component_count=reentry_components,
                rejected_seal_component_count=rejected_seal_components,
                removed_core_fraction=(
                    float(np.count_nonzero(removed & core_full) / core_count)
                    if core_count
                    else 0.0
                ),
                removed_seal_support_fraction=(
                    float(
                        np.count_nonzero(removed & seal_full)
                        / seal_support_count
                    )
                    if seal_support_count
                    else 0.0
                ),
                partially_clipped_component_count=partially_clipped_components,
                trusted_observed_component_count=trusted_components,
                elapsed_delta_h=elapsed_delta_h,
                history_radius_m=history_radius_m,
                reentry_radius_m=reentry_radius_m,
                mean_weak_attenuation=mean_weak_attenuation,
            )
        )
    return output


def apply_temporal_aqueous_gate(
    concentrations: Sequence[np.ndarray],
    observation_confidences: Sequence[np.ndarray],
    valid_mask: np.ndarray,
    *,
    width_m: float,
    height_m: float,
    settings: TemporalGateSettings,
) -> list[TemporalGateFrame]:
    """Remove only weak components lacking optical and temporal support.

    Components containing the current strong aqueous core, or lying within the
    configured physical radius of that core, are retained unconditionally.
    Other trace components require both reference-derived optical evidence and
    overlap with a physically dilated support in an adjacent frame. Values
    below the trace threshold and values at or above the core threshold are not
    modified, keeping this a narrow diagnostic of visible disconnected traces.
    """

    if not concentrations:
        return []
    if len(concentrations) != len(observation_confidences):
        raise ValueError("Concentration and confidence sequence lengths differ.")
    if not 0.0 < float(settings.structure_scale) <= 1.0:
        raise ValueError("structure_scale must be in (0, 1].")
    if not 0.0 < float(settings.trace_level) < float(settings.core_level):
        raise ValueError("Expected 0 < trace_level < core_level.")
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("Physical image dimensions must be positive.")

    shape = tuple(np.asarray(concentrations[0]).shape)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid_mask shape {valid.shape} != {shape}.")
    for index, (concentration, confidence) in enumerate(
        zip(concentrations, observation_confidences, strict=True)
    ):
        if np.asarray(concentration).shape != shape:
            raise ValueError(
                f"concentration[{index}] shape {np.asarray(concentration).shape} "
                f"!= {shape}."
            )
        if np.asarray(confidence).shape != shape:
            raise ValueError(
                f"confidence[{index}] shape {np.asarray(confidence).shape} "
                f"!= {shape}."
            )

    target = (
        max(2, int(round(shape[1] * float(settings.structure_scale)))),
        max(2, int(round(shape[0] * float(settings.structure_scale)))),
    )
    valid_small = _resize_mean(valid.astype(np.float32), target) >= 0.75
    concentration_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in concentrations
    ]
    confidence_small = [
        _resize_mean(np.asarray(value, dtype=np.float32), target)
        for value in observation_confidences
    ]
    trace_support = [
        valid_small & np.isfinite(value) & (value >= settings.trace_level)
        for value in concentration_small
    ]
    core_support = [
        valid_small & np.isfinite(value) & (value >= settings.core_level)
        for value in concentration_small
    ]

    output: list[TemporalGateFrame] = []
    for frame_index, (source, confidence) in enumerate(
        zip(concentration_small, confidence_small, strict=True)
    ):
        trace = trace_support[frame_index]
        core = core_support[frame_index]
        near_core = _dilate_physical(
            core,
            width_m=width_m,
            height_m=height_m,
            radius_m=settings.support_radius_m,
        )
        neighbour = np.zeros_like(trace)
        for adjacent_index in (frame_index - 1, frame_index + 1):
            if 0 <= adjacent_index < len(trace_support):
                neighbour |= trace_support[adjacent_index]
        temporal = _dilate_physical(
            neighbour,
            width_m=width_m,
            height_m=height_m,
            radius_m=settings.support_radius_m,
        )

        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
            trace.astype(np.uint8), connectivity=8
        )
        retained = np.zeros_like(trace)
        retained_components = 0
        rejected_components = 0
        for component in range(1, int(count)):
            component_mask = labels == component
            contains_or_near_core = bool(np.any(component_mask & near_core))
            observed_fraction = float(
                np.mean(
                    confidence[component_mask]
                    >= float(settings.confidence_level)
                )
            )
            temporal_fraction = float(np.mean(temporal[component_mask]))
            supported = contains_or_near_core or (
                observed_fraction >= float(settings.minimum_observed_fraction)
                and temporal_fraction >= float(settings.minimum_temporal_fraction)
            )
            if supported:
                retained[component_mask] = True
                retained_components += 1
            else:
                rejected_components += 1

        retained_full = cv2.resize(
            retained.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        concentration = np.asarray(concentrations[frame_index], dtype=np.float32)
        weak_trace = (
            valid
            & np.isfinite(concentration)
            & (concentration >= float(settings.trace_level))
            & (concentration < float(settings.core_level))
        )
        removed = weak_trace & ~retained_full
        filtered = np.array(concentration, copy=True)
        filtered[removed] = 0.0
        trace_count = int(np.count_nonzero(weak_trace))
        output.append(
            TemporalGateFrame(
                concentration=filtered,
                retained_trace_mask=weak_trace & ~removed,
                removed_trace_mask=removed,
                component_count=max(0, int(count) - 1),
                retained_component_count=retained_components,
                rejected_component_count=rejected_components,
                removed_trace_fraction=(
                    float(np.count_nonzero(removed) / trace_count)
                    if trace_count
                    else 0.0
                ),
            )
        )
    return output
