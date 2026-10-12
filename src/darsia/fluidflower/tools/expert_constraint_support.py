"""Physical expert-constraint variants and spatial support diagnostics.

The production configuration contains rectangular expert ROIs.  They are useful
as broad prior knowledge, but applying them as Boolean masks can create exact
horizontal and vertical phase boundaries.  This module keeps the original masks
immutable and derives controlled hard, expanded, soft, or disabled variants for
diagnostic replay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


MASK_MODES = ("hard", "expanded", "soft", "off")
DEFAULT_EXPANSION_M = 0.03


def normalise_mask_mode(value: str | None) -> str:
    mode = str(value or "hard").strip().lower().replace("_", "-")
    aliases = {
        "none": "off",
        "disabled": "off",
        "dilated": "expanded",
        "tapered": "soft",
    }
    mode = aliases.get(mode, mode)
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown expert-mask mode {value!r}; expected {MASK_MODES}.")
    return mode


def _mask_or_ones(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Expert mask shape {mask.shape} != image shape {shape}.")
    return mask


def original_expert_masks(
    calibration: Any,
    reference: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return immutable gas and aqueous masks from the configured adapter."""

    shape = tuple(np.asarray(reference.img).shape[:2])
    adapter = getattr(calibration, "expert_knowledge_adapter", None)
    if adapter is None:
        ones = np.ones(shape, dtype=bool)
        return ones, ones.copy()
    return (
        _mask_or_ones(adapter.mask_for(reference, "saturation_g"), shape),
        _mask_or_ones(adapter.mask_for(reference, "concentration_aq"), shape),
    )


@dataclass(frozen=True)
class ConstraintSettings:
    mode: str
    expansion_m: float


def constraint_settings_from_env() -> ConstraintSettings:
    """Read the shared calibration/production constraint configuration."""

    mode = normalise_mask_mode(os.environ.get("FFAC_EXPERT_CONSTRAINT_MODE"))
    raw_expansion = os.environ.get(
        "FFAC_EXPERT_CONSTRAINT_EXPANSION_M",
        str(DEFAULT_EXPANSION_M),
    )
    try:
        expansion_m = float(raw_expansion)
    except ValueError as exc:
        raise ValueError(
            "FFAC_EXPERT_CONSTRAINT_EXPANSION_M must be a number in metres."
        ) from exc
    if not np.isfinite(expansion_m) or expansion_m <= 0.0:
        raise ValueError(
            "FFAC_EXPERT_CONSTRAINT_EXPANSION_M must be finite and positive."
        )
    return ConstraintSettings(mode=mode, expansion_m=expansion_m)


def coordinate_axes_m(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-centre x/y axes in metres for a DarSIA-like image."""

    height, width = np.asarray(reference.img).shape[:2]
    coordinatesystem = getattr(reference, "coordinatesystem", None)
    if coordinatesystem is not None:
        rows = np.column_stack(
            (np.arange(height, dtype=float) + 0.5, np.zeros(height, dtype=float))
        )
        columns = np.column_stack(
            (np.zeros(width, dtype=float), np.arange(width, dtype=float) + 0.5)
        )
        y_m = np.asarray(coordinatesystem.coordinate(rows), dtype=float)[:, 1]
        x_m = np.asarray(coordinatesystem.coordinate(columns), dtype=float)[:, 0]
        return x_m, y_m

    metadata_method = getattr(reference, "metadata", None)
    metadata = metadata_method() if callable(metadata_method) else {}
    dimensions = np.asarray(metadata.get("dimensions", []), dtype=float).reshape(-1)
    if dimensions.size >= 2 and np.all(np.isfinite(dimensions[:2])):
        height_m, width_m = float(dimensions[0]), float(dimensions[1])
        x_m = (np.arange(width, dtype=float) + 0.5) * width_m / width
        y_m = (np.arange(height, dtype=float) + 0.5) * height_m / height
        return x_m, y_m
    raise ValueError(
        "Soft expert constraints require image coordinates or physical dimensions."
    )


def _axis_spacing(axis: np.ndarray, expected: int) -> float:
    values = np.asarray(axis, dtype=float).reshape(-1)
    if values.size != expected:
        raise ValueError(f"Coordinate axis length {values.size} != {expected}.")
    finite = np.abs(np.diff(values))
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if not finite.size:
        raise ValueError("Coordinate axis has no finite positive spacing.")
    return float(np.median(finite))


def constraint_weight(
    mask: np.ndarray,
    *,
    mode: str,
    x_m: np.ndarray,
    y_m: np.ndarray,
    expansion_m: float,
) -> np.ndarray:
    """Derive a float32 weight without mutating the source Boolean mask."""

    selected = normalise_mask_mode(mode)
    source = np.asarray(mask, dtype=bool)
    if selected == "off" or np.all(source):
        return np.ones(source.shape, dtype=np.float32)
    if selected == "hard":
        return source.astype(np.float32)
    radius = float(expansion_m)
    if radius <= 0.0:
        raise ValueError("Physical mask expansion must be positive.")

    dy = _axis_spacing(y_m, source.shape[0])
    dx = _axis_spacing(x_m, source.shape[1])
    outside_distance = distance_transform_edt(~source, sampling=(dy, dx))
    if selected == "expanded":
        return (outside_distance <= radius).astype(np.float32)

    unit = np.clip(1.0 - outside_distance / radius, 0.0, 1.0)
    weight = unit * unit * (3.0 - 2.0 * unit)
    weight[source] = 1.0
    return weight.astype(np.float32)


@dataclass(frozen=True)
class ConstraintWeights:
    mode: str
    expansion_m: float
    gas: np.ndarray
    aqueous: np.ndarray
    original_gas: np.ndarray
    original_aqueous: np.ndarray


def build_constraint_weights(
    *,
    gas_mask: np.ndarray,
    aqueous_mask: np.ndarray,
    mode: str,
    x_m: np.ndarray,
    y_m: np.ndarray,
    expansion_m: float,
) -> ConstraintWeights:
    selected = normalise_mask_mode(mode)
    return ConstraintWeights(
        mode=selected,
        expansion_m=float(expansion_m),
        gas=constraint_weight(
            gas_mask,
            mode=selected,
            x_m=x_m,
            y_m=y_m,
            expansion_m=expansion_m,
        ),
        aqueous=constraint_weight(
            aqueous_mask,
            mode=selected,
            x_m=x_m,
            y_m=y_m,
            expansion_m=expansion_m,
        ),
        original_gas=np.asarray(gas_mask, dtype=bool).copy(),
        original_aqueous=np.asarray(aqueous_mask, dtype=bool).copy(),
    )


def constraint_weights_for_calibration(
    calibration: Any,
    reference: Any,
    *,
    mode: str | None = None,
    expansion_m: float | None = None,
) -> ConstraintWeights:
    """Build the configured expert weights for one analysis geometry."""

    settings = constraint_settings_from_env()
    selected_mode = normalise_mask_mode(mode or settings.mode)
    selected_expansion = (
        settings.expansion_m if expansion_m is None else float(expansion_m)
    )
    gas_mask, aqueous_mask = original_expert_masks(calibration, reference)
    if selected_mode in {"soft", "expanded"}:
        x_m, y_m = coordinate_axes_m(reference)
    else:
        height, width = gas_mask.shape
        x_m = np.arange(width, dtype=float)
        y_m = np.arange(height, dtype=float)
    return build_constraint_weights(
        gas_mask=gas_mask,
        aqueous_mask=aqueous_mask,
        mode=selected_mode,
        x_m=x_m,
        y_m=y_m,
        expansion_m=selected_expansion,
    )


def call_flash_and_mass_with_constraints(
    calibration: Any,
    pH: Any,
) -> Any:
    """Run flash/mass with the configured expert weighting on CPU."""

    settings = constraint_settings_from_env()
    if settings.mode == "hard":
        return calibration.call_flash_and_mass_analysis(pH)

    concentration_aq, saturation_g = calibration.flash(pH)
    weights = constraint_weights_for_calibration(calibration, pH)
    concentration_aq = concentration_aq.copy()
    saturation_g = saturation_g.copy()
    concentration_aq.img = np.maximum(
        0.0,
        np.asarray(concentration_aq.img) * weights.aqueous,
    )
    saturation_g.img = np.clip(
        np.asarray(saturation_g.img) * weights.gas,
        0.0,
        1.0,
    )
    return calibration.co2_mass_analysis.mass_analysis(
        c_aq=concentration_aq,
        s_g=saturation_g,
    )


@dataclass(frozen=True)
class WeightedPhaseFields:
    saturation_g: np.ndarray
    concentration_aq: np.ndarray
    mass_g: np.ndarray
    mass_aq: np.ndarray
    mass: np.ndarray


def apply_constraint_weights(
    *,
    saturation_g: np.ndarray,
    concentration_aq: np.ndarray,
    density_g: np.ndarray,
    solubility_aq: np.ndarray,
    weights: ConstraintWeights,
) -> WeightedPhaseFields:
    """Apply weights and recompute the physically partitioned mass fields."""

    saturation = np.clip(
        np.asarray(saturation_g, dtype=np.float32) * weights.gas,
        0.0,
        1.0,
    )
    concentration = np.maximum(
        0.0,
        np.asarray(concentration_aq, dtype=np.float32) * weights.aqueous,
    )
    density = np.broadcast_to(np.asarray(density_g, dtype=np.float32), saturation.shape)
    solubility = np.broadcast_to(
        np.asarray(solubility_aq, dtype=np.float32), saturation.shape
    )
    mass_g = density * saturation
    mass_aq = solubility * concentration * np.clip(1.0 - saturation, 0.0, 1.0)
    return WeightedPhaseFields(
        saturation_g=saturation,
        concentration_aq=concentration,
        mass_g=mass_g,
        mass_aq=mass_aq,
        mass=mass_g + mass_aq,
    )


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(source.astype(np.uint8), kernel, iterations=1).astype(bool)
    return source & ~eroded


def contour_near_mask_boundary_fraction(
    *,
    field: np.ndarray,
    level: float,
    original_mask: np.ndarray,
    tolerance_pixels: int = 3,
) -> float:
    """Fraction of a threshold contour lying near the original ROI boundary."""

    support = np.isfinite(field) & (np.asarray(field) >= float(level))
    contour = _binary_boundary(support)
    count = int(np.count_nonzero(contour))
    if not count:
        return float("nan")
    mask_boundary = _binary_boundary(original_mask) | _binary_boundary(~original_mask)
    radius = max(0, int(tolerance_pixels))
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        mask_boundary = cv2.dilate(
            mask_boundary.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
    return float(np.count_nonzero(contour & mask_boundary) / count)


def _resize_mean(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        np.asarray(values, dtype=np.float32),
        target,
        interpolation=cv2.INTER_AREA,
    )


def _resize_extensive(values: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    resized = cv2.resize(source, target, interpolation=cv2.INTER_AREA)
    return resized * (source.shape[0] * source.shape[1]) / (target[0] * target[1])


def _dilate_physical_coarse(
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


def component_support_metrics(
    *,
    field: np.ndarray,
    extensive_mass_kg: np.ndarray,
    valid_mask: np.ndarray,
    observation_confidence: np.ndarray,
    previous_field: np.ndarray | None,
    next_field: np.ndarray | None,
    level: float,
    width_m: float,
    height_m: float,
    temporal_radius_m: float = 0.02,
    scale: float = 0.20,
    small_component_pixels: int = 16,
) -> dict[str, float]:
    """Measure how much phase mass lacks optical and neighbouring-frame support.

    Connectivity is diagnostic only.  A component is called unsupported here
    only when both its raw observation confidence and its overlap with a
    physically dilated neighbouring-frame support are low.
    """

    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must be in (0, 1].")
    source = np.asarray(field, dtype=np.float32)
    shape = source.shape
    for name, value in {
        "extensive_mass_kg": extensive_mass_kg,
        "valid_mask": valid_mask,
        "observation_confidence": observation_confidence,
    }.items():
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape {np.asarray(value).shape} != {shape}.")

    target_w = max(2, int(round(shape[1] * float(scale))))
    target_h = max(2, int(round(shape[0] * float(scale))))
    target = (target_w, target_h)
    field_small = _resize_mean(source, target)
    valid_small = _resize_mean(np.asarray(valid_mask, dtype=np.float32), target) >= 0.75
    confidence_small = _resize_mean(observation_confidence, target)
    mass_small = _resize_extensive(extensive_mass_kg, target)
    support = valid_small & np.isfinite(field_small) & (field_small >= float(level))

    neighbour = np.zeros_like(support)
    for adjacent in (previous_field, next_field):
        if adjacent is None:
            continue
        adjacent_small = _resize_mean(np.asarray(adjacent, dtype=np.float32), target)
        neighbour |= adjacent_small >= float(level)
    temporal = _dilate_physical_coarse(
        neighbour,
        width_m=width_m,
        height_m=height_m,
        radius_m=temporal_radius_m,
    )

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    total_mass = float(np.sum(np.where(support, mass_small, 0.0), dtype=np.float64))
    unsupported_mass = 0.0
    low_observation_mass = 0.0
    low_temporal_mass = 0.0
    small_mass = 0.0
    component_rows: list[tuple[float, float, float, int]] = []
    for component in range(1, int(count)):
        component_mask = labels == component
        area = int(stats[component, cv2.CC_STAT_AREA])
        mass = float(np.sum(mass_small[component_mask], dtype=np.float64))
        raw_fraction = float(np.mean(confidence_small[component_mask] >= 0.5))
        temporal_fraction = float(np.mean(temporal[component_mask]))
        if raw_fraction < 0.25:
            low_observation_mass += mass
        if temporal_fraction < 0.10:
            low_temporal_mass += mass
        if raw_fraction < 0.25 and temporal_fraction < 0.10:
            unsupported_mass += mass
        if area <= int(small_component_pixels):
            small_mass += mass
        component_rows.append((mass, raw_fraction, temporal_fraction, area))

    def fraction(value: float) -> float:
        return value / total_mass if total_mass > 0.0 else float("nan")

    largest_fraction = float("nan")
    if total_mass > 0.0 and component_rows:
        largest_fraction = max(row[0] for row in component_rows) / total_mass
    return {
        "component_count": float(max(0, count - 1)),
        "support_mass_kg": total_mass,
        "largest_component_mass_fraction": largest_fraction,
        "small_component_mass_fraction": fraction(small_mass),
        "low_observation_support_mass_fraction": fraction(low_observation_mass),
        "low_observation_support_mass_kg": low_observation_mass,
        "low_temporal_support_mass_fraction": fraction(low_temporal_mass),
        "low_temporal_support_mass_kg": low_temporal_mass,
        "unsupported_mass_fraction": fraction(unsupported_mass),
        "unsupported_mass_kg": unsupported_mass,
        "structure_scale": float(scale),
    }
