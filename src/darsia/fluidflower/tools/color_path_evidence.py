"""Full-RGB evidence for run-native DarSIA colour paths.

The production DarSIA pipeline reduces a corrected relative RGB vector to one
coordinate along a colour path.  This module retains two pieces of information
that reduction otherwise discards:

* how unlikely the RGB change is under the run's no-CO2 reference frames; and
* how closely the RGB vector agrees with the nearest point on the colour path.

The model is fitted only to no-CO2 references and fixed colour paths.  It has no
Optuna-controlled parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from darsia.fluidflower.tools.aqueous_noise_gate import smoothstep_confidence


EVIDENCE_SCHEMA = 1


def normalise_color_path_evidence_mode(value: str | None) -> str:
    """Normalize the optional full-RGB representation mode."""

    mode = str(value or "off").strip().lower().replace("_", "-")
    aliases = {
        "0": "off",
        "false": "off",
        "none": "off",
        "1": "diagnostic",
        "true": "diagnostic",
        "on": "diagnostic",
        "gate": "soft-gate",
        "gated": "soft-gate",
        "rgb-gate": "soft-gate",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "diagnostic", "soft-gate"}:
        raise ValueError(
            f"Unknown colour-path evidence mode {value!r}; expected off, "
            "diagnostic, or soft-gate."
        )
    return mode


@dataclass(frozen=True)
class LabelRgbReference:
    """Whitened full-RGB baseline distribution for one segmentation label."""

    label: int
    center: tuple[float, float, float]
    precision: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    low: float
    high: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": int(self.label),
            "center": list(self.center),
            "precision": [list(row) for row in self.precision],
            "low": float(self.low),
            "high": float(self.high),
            "sample_count": int(self.sample_count),
        }


@dataclass(frozen=True)
class ColorPathEvidenceResult:
    """Per-pixel diagnostics and the optional RGB-aware scalar coordinate."""

    coordinate: np.ndarray
    gated_coordinate: np.ndarray
    baseline_change_confidence: np.ndarray
    path_fit_confidence: np.ndarray
    path_support_confidence: np.ndarray
    observation_confidence: np.ndarray
    normalized_path_residual: np.ndarray


@dataclass(frozen=True)
class ColorPathEvidenceModel:
    """No-CO2 RGB likelihood and colour-path conformity model."""

    references: Mapping[int, LabelRgbReference]
    residual_low: float = 0.15
    residual_high: float = 0.45
    minimum_scale: float = 1e-4
    spatial_stride: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "coordinate": "whitened_relative_rgb_plus_l1_path_residual",
            "residual_low": float(self.residual_low),
            "residual_high": float(self.residual_high),
            "minimum_scale": float(self.minimum_scale),
            "spatial_stride": int(self.spatial_stride),
            "references": {
                str(label): reference.to_dict()
                for label, reference in sorted(self.references.items())
            },
        }

    def evaluate(
        self,
        relative_rgb: Any,
        labels: Any,
        interpolations: Mapping[int, Any],
        *,
        acidification_confidence: Any | None = None,
    ) -> ColorPathEvidenceResult:
        """Evaluate RGB departure, path residual, and a soft-gated coordinate."""

        colors = np.asarray(relative_rgb, dtype=np.float32)
        label_array = np.asarray(labels)
        if colors.ndim != 3 or colors.shape[-1] < 3:
            raise ValueError(f"Expected relative RGB image, got {colors.shape}.")
        colors = colors[..., :3]
        if colors.shape[:2] != label_array.shape:
            raise ValueError(
                f"Relative RGB/label shapes differ: {colors.shape[:2]} != "
                f"{label_array.shape}."
            )

        shape = label_array.shape
        coordinate = np.zeros(shape, dtype=np.float32)
        change_confidence = np.zeros(shape, dtype=np.float32)
        fit_confidence = np.zeros(shape, dtype=np.float32)
        normalized_residual = np.ones(shape, dtype=np.float32)

        for label, reference in self.references.items():
            mask = label_array == int(label)
            if not np.any(mask):
                continue
            interpolation = interpolations.get(int(label))
            if interpolation is None:
                continue
            selected = np.asarray(colors[mask], dtype=np.float64)
            center = np.asarray(reference.center, dtype=np.float64)
            precision = np.asarray(reference.precision, dtype=np.float64)
            centered = selected - center
            squared_distance = np.einsum(
                "ni,ij,nj->n", centered, precision, centered, optimize=True
            )
            distance = np.sqrt(np.clip(squared_distance, 0.0, None))
            selected_change = np.asarray(
                smoothstep_confidence(distance, reference.low, reference.high),
                dtype=np.float32,
            )
            change_confidence[mask] = selected_change

            active = selected_change > 0.0
            if not np.any(active):
                normalized_residual[mask] = 0.0
                continue

            active_colors = selected[active]
            path = interpolation.color_path
            parameters = np.asarray(
                path.fit(
                    colors=active_colors,
                    color_mode=interpolation.color_mode,
                    mode="equidistant",
                ),
                dtype=np.float64,
            )
            reconstructed = np.asarray(
                path.interpret(
                    parameters,
                    color_mode=interpolation.color_mode,
                    mode="equidistant",
                ),
                dtype=np.float64,
            )
            residual = np.linalg.norm(
                active_colors - reconstructed, ord=1, axis=-1
            )
            supports = (
                path.colors
                if str(getattr(interpolation.color_mode, "value", interpolation.color_mode)).lower()
                == "absolute"
                else path.relative_colors
            )
            origin = np.asarray(supports[0], dtype=np.float64)
            magnitude = np.linalg.norm(active_colors - origin, ord=1, axis=-1)
            ratio = residual / np.maximum(magnitude, float(self.minimum_scale))
            selected_fit = 1.0 - np.asarray(
                smoothstep_confidence(
                    ratio,
                    float(self.residual_low),
                    float(self.residual_high),
                ),
                dtype=np.float32,
            )

            values = _interpolate_values(interpolation, parameters)
            selected_coordinate = np.zeros(selected.shape[0], dtype=np.float32)
            selected_coordinate[active] = np.asarray(values, dtype=np.float32)
            selected_fit_full = np.zeros(selected.shape[0], dtype=np.float32)
            selected_fit_full[active] = selected_fit
            selected_residual = np.zeros(selected.shape[0], dtype=np.float32)
            selected_residual[active] = np.asarray(ratio, dtype=np.float32)

            coordinate[mask] = selected_coordinate
            fit_confidence[mask] = selected_fit_full
            normalized_residual[mask] = selected_residual

        path_support = np.clip(
            change_confidence * fit_confidence,
            0.0,
            1.0,
        ).astype(np.float32)
        gated_coordinate = np.asarray(
            coordinate * path_support,
            dtype=np.float32,
        )
        if acidification_confidence is None:
            observation_confidence = path_support.copy()
        else:
            acid = np.asarray(acidification_confidence, dtype=np.float32)
            if acid.shape != shape:
                raise ValueError(
                    f"Acidification/label shapes differ: {acid.shape} != {shape}."
                )
            # Acidification can rescue a real yellow response that a coarse path
            # fits poorly.  The path gate remains strict so the 2x2 experiment can
            # reveal whether that one-dimensional representation is the problem.
            observation_confidence = np.maximum(acid, path_support)

        return ColorPathEvidenceResult(
            coordinate=coordinate,
            gated_coordinate=gated_coordinate,
            baseline_change_confidence=change_confidence,
            path_fit_confidence=fit_confidence,
            path_support_confidence=path_support,
            observation_confidence=np.clip(
                observation_confidence, 0.0, 1.0
            ).astype(np.float32),
            normalized_path_residual=normalized_residual,
        )


def relative_rgb_for_color_analysis(color_analysis: Any, image: Any) -> np.ndarray:
    """Reproduce DarSIA preprocessing immediately before path projection."""

    diff = color_analysis._subtract_background(image)
    signal = color_analysis._reduce_signal(diff)
    cleaned = color_analysis._clean_signal(signal)
    balanced = color_analysis._balance_signal(cleaned)
    array = np.asarray(balanced, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            "Colour-path evidence requires a three-channel signal before "
            f"projection; got {array.shape}."
        )
    return np.asarray(array[..., :3], dtype=np.float32)


def estimate_color_path_evidence(
    relative_rgb_references: Sequence[Any],
    labels: Any,
    active_labels: Iterable[int],
    *,
    quantile_low: float = 0.99,
    quantile_high: float = 0.999,
    spatial_stride: int = 4,
    covariance_floor: float = 1e-6,
    minimum_width: float = 0.25,
    residual_low: float = 0.15,
    residual_high: float = 0.45,
) -> ColorPathEvidenceModel:
    """Estimate per-label full-RGB no-CO2 likelihoods."""

    if len(relative_rgb_references) < 2:
        raise ValueError("At least two no-CO2 RGB references are required.")
    if not 0.5 <= quantile_low < quantile_high < 1.0:
        raise ValueError("RGB evidence quantiles must satisfy 0.5 <= low < high < 1.")
    if not 0.0 <= residual_low < residual_high:
        raise ValueError("Path-residual limits must satisfy 0 <= low < high.")
    stride = max(1, int(spatial_stride))
    label_array = np.asarray(labels)
    frames = [np.asarray(value, dtype=np.float32) for value in relative_rgb_references]
    if any(frame.shape[:2] != label_array.shape or frame.shape[-1] < 3 for frame in frames):
        raise ValueError("No-CO2 RGB reference and label shapes differ.")

    references: dict[int, LabelRgbReference] = {}
    sampled_labels = label_array[::stride, ::stride]
    for label in sorted({int(value) for value in active_labels}):
        mask = sampled_labels == label
        samples = np.concatenate(
            [frame[::stride, ::stride, :3][mask] for frame in frames],
            axis=0,
        ).astype(np.float64, copy=False)
        samples = samples[np.all(np.isfinite(samples), axis=1)]
        if samples.shape[0] < 4:
            raise ValueError(f"Insufficient RGB reference samples for label {label}.")

        center = np.median(samples, axis=0)
        centered = samples - center
        covariance = np.cov(centered, rowvar=False)
        covariance = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        scale = max(
            float(covariance_floor),
            float(np.trace(covariance)) / 3.0 * 1e-3,
        )
        covariance += np.eye(3, dtype=np.float64) * scale
        precision = np.linalg.pinv(covariance, hermitian=True)
        squared_distance = np.einsum(
            "ni,ij,nj->n", centered, precision, centered, optimize=True
        )
        distance = np.sqrt(np.clip(squared_distance, 0.0, None))
        low = max(0.0, float(np.quantile(distance, quantile_low)))
        high = max(
            float(np.quantile(distance, quantile_high)),
            low + max(float(minimum_width), abs(low) * 0.05),
        )
        references[label] = LabelRgbReference(
            label=label,
            center=tuple(float(value) for value in center),
            precision=tuple(
                tuple(float(value) for value in row) for row in precision
            ),
            low=low,
            high=high,
            sample_count=int(samples.shape[0]),
        )

    return ColorPathEvidenceModel(
        references=references,
        residual_low=float(residual_low),
        residual_high=float(residual_high),
        spatial_stride=stride,
    )


def _interpolate_values(interpolation: Any, parameters: np.ndarray) -> np.ndarray:
    """Match DarSIA's segment-wise extrapolating interpolation without masking."""

    distances = np.asarray(
        interpolation.color_path.equidistant_distances, dtype=np.float64
    )
    values = np.asarray(interpolation.values, dtype=np.float64)
    if distances.shape != values.shape:
        raise ValueError(
            f"Colour-path distance/value shapes differ: {distances.shape} != "
            f"{values.shape}."
        )
    result = np.zeros_like(parameters, dtype=np.float64)
    segment_count = len(distances) - 1
    for index in range(segment_count):
        if index == 0:
            mask = parameters <= distances[index + 1]
        elif index == segment_count - 1:
            mask = parameters >= distances[index]
        else:
            mask = (parameters >= distances[index]) & (
                parameters <= distances[index + 1]
            )
        weight = (parameters[mask] - distances[index]) / (
            distances[index + 1] - distances[index]
        )
        result[mask] = values[index] + weight * (
            values[index + 1] - values[index]
        )
    return result
