"""Spatial CPU/CUDA/OpenCL evaluators for production mass analysis.

The calibration GPU evaluators intentionally return only integrated masses and
therefore include voxel-volume weights. Production analysis needs intensive
spatial fields instead. The evaluators in this module return the same
``SimpleMassAnalysisResults`` contract as DarSIA without applying geometry
weights; integration and extensive conversion remain in the normal analysis
workflow.
"""

from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import darsia
import numpy as np

try:
    from darsia.fluidflower.tools.expert_constraint_support import (
        call_flash_and_mass_with_constraints,
        constraint_settings_from_env,
        constraint_weights_for_calibration,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.expert_constraint_support import (
        call_flash_and_mass_with_constraints,
        constraint_settings_from_env,
        constraint_weights_for_calibration,
    )

try:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec
except ModuleNotFoundError:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec


logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalise_backend(value: str | None) -> str:
    backend = str(value or "cpu").strip().lower().replace("_", "-")
    aliases = {
        "numpy": "cpu",
        "prepared": "cpu",
        "gpu": "auto",
        "ocl": "opencl",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"auto", "cpu", "cuda", "opencl"}:
        raise ValueError(
            f"Unknown mass backend {value!r}; expected auto, cpu, cuda, or opencl."
        )
    return backend


def _normalise_dtype(value: str | np.dtype[Any]) -> np.dtype[Any]:
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("Spatial mass output dtype must be float32 or float64.")
    return dtype


def _active_signal_models(calibration: Any) -> dict[int, Any]:
    heterogeneous = calibration.signal_model.model[1]
    ignored = {int(label) for label in getattr(heterogeneous, "ignore_labels", [])}
    return {
        int(label): heterogeneous[label]
        for label in heterogeneous.keys()
        if int(label) not in ignored
    }


def _aqueous_noise_gate(calibration: Any) -> Any | None:
    gate = getattr(calibration, "aqueous_noise_gate", None)
    return gate if gate is not None and getattr(gate, "enabled", False) else None


def _apply_aqueous_noise_gate(calibration: Any, color: Any) -> Any:
    gate = _aqueous_noise_gate(calibration)
    if gate is None:
        return color
    return gate.apply_image(
        color,
        calibration.labels.img,
        copy=False,
    )


def _flash_lut(flash: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(flash, "_lut_y") and hasattr(flash, "_lut_caq"):
        return (
            np.asarray(flash._lut_y, dtype=np.float64),
            np.asarray(flash._lut_caq, dtype=np.float64),
        )
    identity = np.asarray([0.0, 1.0], dtype=np.float64)
    return identity, identity


def _expert_masks(calibration: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    shape = tuple(np.asarray(reference.img).shape[:2])
    gas = np.ones(shape, dtype=bool)
    aqueous = np.ones(shape, dtype=bool)
    adapter = getattr(calibration, "expert_knowledge_adapter", None)
    if adapter is not None:
        gas = np.asarray(adapter.mask_for(reference, "saturation_g"), dtype=bool)
        aqueous = np.asarray(
            adapter.mask_for(reference, "concentration_aq"), dtype=bool
        )
    return gas, aqueous


def _expert_constraint_weights(
    calibration: Any,
    reference: Any,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    weights = constraint_weights_for_calibration(calibration, reference)
    return weights.gas, weights.aqueous, weights.mode, weights.expansion_m


def _broadcast_static_mass_arrays(
    calibration: Any, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    mass_analysis = calibration.co2_mass_analysis
    try:
        density = np.broadcast_to(
            np.asarray(mass_analysis.density_gaseous_co2, dtype=np.float64),
            shape,
        )
        solubility = np.broadcast_to(
            np.asarray(mass_analysis.solubility_co2, dtype=np.float64),
            shape,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Density or solubility is incompatible with the analysis image shape "
            f"{shape}."
        ) from exc
    return density, solubility


def _mass_result(
    reference: Any,
    concentration_aq: np.ndarray,
    saturation_g: np.ndarray,
    mass_g: np.ndarray,
    mass_aq: np.ndarray,
    output_dtype: np.dtype[Any],
) -> Any:
    concentration_aq = np.asarray(concentration_aq, dtype=output_dtype)
    saturation_g = np.asarray(saturation_g, dtype=output_dtype)
    mass_g = np.asarray(mass_g, dtype=output_dtype)
    mass_aq = np.asarray(mass_aq, dtype=output_dtype)
    mass = np.asarray(mass_g + mass_aq, dtype=output_dtype)
    metadata = reference.metadata()

    def scalar(values: np.ndarray) -> darsia.ScalarImage:
        return darsia.ScalarImage(
            np.asarray(values, dtype=output_dtype),
            **metadata,
        )

    return darsia.SimpleMassAnalysisResults(
        name=reference.name,
        date=reference.date,
        time=reference.time,
        mass=scalar(mass),
        mass_g=scalar(mass_g),
        mass_aq=scalar(mass_aq),
        saturation_g=scalar(saturation_g),
        concentration_aq=scalar(concentration_aq),
        color_signal=None,
    )


def _call_color_interpretation_no_copy(
    calibration: Any,
    image: darsia.Image,
) -> darsia.Image:
    """Run DarSIA's color-analysis stages without its unconditional deepcopy."""

    analysis = getattr(calibration, "color_analysis", None)
    if (
        analysis is None
        or image.img.dtype
        not in {
            np.dtype(float),
            np.dtype(np.float32),
            np.dtype(np.float64),
        }
        or _truthy(os.environ.get("FFAC_MASS_LEGACY_COLOR_COPY"))
    ):
        return calibration.call_color_interpretation(image)

    def inspect(values: np.ndarray, title: str, legacy_name: str) -> None:
        if hasattr(analysis, "_inspect"):
            analysis._inspect(values, title=title)
        elif hasattr(analysis, legacy_name):
            getattr(analysis, legacy_name)(values)

    diff = analysis._subtract_background(image)
    inspect(diff, "Difference", "_inspect_diff")
    signal = analysis._reduce_signal(diff)
    inspect(signal, "Scalar signal", "_inspect_scalar_signal")
    clean_signal = analysis._clean_signal(signal)
    inspect(clean_signal, "Clean signal", "_inspect_clean_signal")
    balanced_signal = analysis._balance_signal(clean_signal)
    inspect(balanced_signal, "Balanced signal", "_inspect_balanced_signal")
    if analysis.first_restoration_then_model:
        smooth_signal = analysis._restore_signal(balanced_signal)
        inspect(smooth_signal, "Restored signal", "_inspect_restored_signal")
        concentration = analysis._convert_signal(smooth_signal, diff)
        inspect(concentration, "Concentration", "_inspect_concentration")
    else:
        nonsmooth = analysis._convert_signal(balanced_signal, diff)
        inspect(
            nonsmooth,
            "Nonsmooth concentration",
            "_inspect_nonsmooth_concentration",
        )
        concentration = analysis._restore_signal(nonsmooth)
        inspect(concentration, "Concentration", "_inspect_concentration")

    metadata = image.metadata()
    is_scalar = len(concentration.shape) == len(image.shape) - 1
    is_effectively_scalar = (
        concentration.ndim > 0 and concentration.shape[-1] == 1
    )
    if is_scalar:
        return darsia.ScalarImage(concentration, **metadata)
    if is_effectively_scalar:
        return darsia.ScalarImage(concentration[..., 0], **metadata)
    return type(image)(concentration, **metadata)


class CpuSpatialMassEvaluator:
    """Reference backend delegating to the unmodified DarSIA pipeline."""

    backend_name = "cpu"

    def __init__(self, calibration: Any, *, output_dtype: str = "float32") -> None:
        self.calibration = calibration
        self.output_dtype = _normalise_dtype(output_dtype)
        labels = np.asarray(calibration.labels.img)
        self.restoration_spec = signal_restoration_spec(
            calibration,
            labels.shape[:2],
        )
        self.last_timing: dict[str, float] = {}

    def evaluate(self, image: Any) -> Any:
        started = time.perf_counter()
        gate = _aqueous_noise_gate(self.calibration)
        if gate is None and constraint_settings_from_env().mode == "hard":
            result = self.calibration(image)
        else:
            color = self.calibration.call_color_interpretation(image)
            if gate is not None:
                color = _apply_aqueous_noise_gate(self.calibration, color)
            signal = self.calibration.call_pH_analysis(color)
            result = call_flash_and_mass_with_constraints(
                self.calibration,
                signal,
            )
        self.last_timing = {
            "backend_total_s": time.perf_counter() - started,
        }
        return result

    def close(self) -> None:
        return None

    def metadata(self) -> dict[str, Any]:
        settings = constraint_settings_from_env()
        return {
            "backend": self.backend_name,
            "output_dtype": "native",
            "aqueous_noise_gate": _aqueous_noise_gate(self.calibration) is not None,
            "expert_constraint_mode": settings.mode,
            "expert_constraint_expansion_m": settings.expansion_m,
        }


CUDA_COLOR_PATH_KERNEL = r"""
extern "C" __global__
void color_path_interpretation(
    const double *image,
    const double *baseline,
    const int has_baseline,
    const int *labels,
    const long long pixel_count,
    const int *label_to_model,
    const int label_map_size,
    const double *supports,
    const double *distances,
    const double *values,
    const int *point_counts,
    const unsigned char *ignore_spectrum,
    const int max_points,
    double *output
) {
    const long long pixel = (long long) (
        blockDim.x * blockIdx.x + threadIdx.x
    );
    if (pixel >= pixel_count) return;
    const int label = labels[pixel];
    if (label < 0 || label >= label_map_size) {
        output[pixel] = 0.0;
        return;
    }
    const int model = label_to_model[label];
    if (model < 0) {
        output[pixel] = 0.0;
        return;
    }

    const long long color_offset = 3 * pixel;
    const double c0 = image[color_offset]
        - (has_baseline ? baseline[color_offset] : 0.0);
    const double c1 = image[color_offset + 1]
        - (has_baseline ? baseline[color_offset + 1] : 0.0);
    const double c2 = image[color_offset + 2]
        - (has_baseline ? baseline[color_offset + 2] : 0.0);
    if (
        ignore_spectrum[model]
        && sqrt(c0 * c0 + c1 * c1 + c2 * c2) <= 0.1
    ) {
        output[pixel] = 0.0;
        return;
    }

    const int point_count = point_counts[model];
    const int model_offset = model * max_points;
    // NVRTC does not expose the C INFINITY macro consistently across CUDA
    // toolkit versions. A finite DBL_MAX literal is sufficient for this
    // minimum search and keeps the kernel self-contained.
    double minimum_distance = 1.7976931348623157e+308;
    double best_value = 0.0;
    for (int segment = 0; segment < point_count - 1; ++segment) {
        const int left = model_offset + segment;
        const int right = left + 1;
        const double s00 = supports[3 * left];
        const double s01 = supports[3 * left + 1];
        const double s02 = supports[3 * left + 2];
        const double d0 = supports[3 * right] - s00;
        const double d1 = supports[3 * right + 1] - s01;
        const double d2 = supports[3 * right + 2] - s02;
        const double denominator = d0 * d0 + d1 * d1 + d2 * d2;
        if (!(denominator > 0.0)) continue;

        const double distance_left = distances[left];
        const double distance_right = distances[right];
        double parameter = distance_left
            + (distance_right - distance_left)
            * (
                (c0 - s00) * d0
                + (c1 - s01) * d1
                + (c2 - s02) * d2
            )
            / denominator;
        if (segment == 0) {
            parameter = fmin(parameter, distance_right);
        } else if (segment == point_count - 2) {
            parameter = fmax(parameter, distance_left);
        } else {
            parameter = fmin(
                distance_right,
                fmax(distance_left, parameter)
            );
        }
        const double segment_length = distance_right - distance_left;
        if (!(segment_length != 0.0)) continue;
        const double fraction = (
            parameter - distance_left
        ) / segment_length;
        const double p0 = s00 + fraction * d0;
        const double p1 = s01 + fraction * d1;
        const double p2 = s02 + fraction * d2;
        const double color_distance = (
            fabs(c0 - p0) + fabs(c1 - p1) + fabs(c2 - p2)
        );
        if (color_distance < minimum_distance) {
            minimum_distance = color_distance;
            best_value = values[left] + fraction * (
                values[right] - values[left]
            );
        }
    }
    output[pixel] = isfinite(best_value) ? best_value : 0.0;
}
"""


@dataclass(frozen=True)
class _CudaColorPathPack:
    """Validated, padded color-path arrays independent of a CUDA runtime."""

    labels: np.ndarray
    supports: np.ndarray
    distances: np.ndarray
    values: np.ndarray
    point_counts: np.ndarray
    ignore_spectrum: np.ndarray
    label_to_model: np.ndarray
    max_points: int


def _pack_cuda_color_paths(calibration: Any) -> _CudaColorPathPack:
    """Validate and pack the exact DarSIA color-path model for the CUDA kernel."""

    analysis = getattr(calibration, "color_analysis", None)
    if analysis is None:
        raise ValueError("Calibration has no color analysis.")
    if getattr(analysis, "_diff_option", None) != "plain":
        raise ValueError("CUDA color interpretation requires plain subtraction.")
    if getattr(analysis, "first_restoration_then_model", True):
        raise ValueError(
            "CUDA color interpretation requires model-before-restoration order."
        )
    if any(
        value is not None
        for value in (
            getattr(analysis, "signal_reduction", None),
            getattr(analysis, "threshold_cleaning_filter", None),
            getattr(analysis, "balancing", None),
        )
    ):
        raise ValueError(
            "CUDA color interpretation does not support reduction, cleaning, "
            "or balancing before the color-path model."
        )

    combined = getattr(analysis, "model", None)
    models = getattr(combined, "models", None)
    if models is None or len(models) != 1:
        raise ValueError(
            "CUDA color interpretation requires one heterogeneous model."
        )
    heterogeneous = models[0]
    ignored = {
        int(label) for label in getattr(heterogeneous, "ignore_labels", [])
    }

    raw_labels = np.asarray(calibration.labels.img)
    if raw_labels.ndim != 2:
        raise ValueError("CUDA color interpretation requires 2D labels.")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        if not np.all(np.isfinite(raw_labels)) or not np.all(
            raw_labels == np.rint(raw_labels)
        ):
            raise ValueError("CUDA color labels must contain integer values.")
    labels = np.asarray(raw_labels, dtype=np.int32)

    active: list[tuple[int, Any]] = sorted(
        (
            (int(label), heterogeneous[label])
            for label in heterogeneous.keys()
            if int(label) not in ignored
        ),
        key=lambda item: item[0],
    )
    if not active:
        raise ValueError("CUDA color interpretation found no active color paths.")
    if any(label < 0 for label, _ in active):
        raise ValueError("CUDA color interpretation requires non-negative labels.")

    packed_models: list[
        tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]
    ] = []
    for label, model in active:
        if not all(
            hasattr(model, name)
            for name in ("color_path", "color_mode", "values")
        ):
            raise ValueError(
                "CUDA color interpretation only supports ColorPathInterpolation."
            )
        color_path = model.color_path
        color_mode = str(model.color_mode).strip().lower()
        if color_mode == str(darsia.ColorMode.ABSOLUTE):
            model_supports = np.asarray(color_path.colors, dtype=np.float64)
        elif color_mode == str(darsia.ColorMode.RELATIVE):
            model_supports = np.asarray(
                color_path.relative_colors,
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported CUDA color mode {model.color_mode!r}.")
        model_distances = np.asarray(
            color_path.equidistant_distances,
            dtype=np.float64,
        )
        model_values = np.asarray(model.values, dtype=np.float64)
        count = int(model_values.size)
        if (
            count < 2
            or model_supports.shape != (count, 3)
            or model_distances.shape != (count,)
        ):
            raise ValueError("Inconsistent CUDA color-path support arrays.")
        if (
            not np.all(np.isfinite(model_supports))
            or not np.all(np.isfinite(model_distances))
            or not np.all(np.isfinite(model_values))
        ):
            raise ValueError("CUDA color-path arrays contain non-finite values.")
        if np.any(np.diff(model_distances) <= 0.0):
            raise ValueError(
                "CUDA color-path distances must be strictly increasing."
            )
        packed_models.append(
            (
                label,
                model_supports,
                model_distances,
                model_values,
                getattr(model, "ignore_spectrum", None) is not None,
            )
        )

    max_points = max(model[3].size for model in packed_models)
    model_count = len(packed_models)
    supports = np.zeros((model_count, max_points, 3), dtype=np.float64)
    distances = np.zeros((model_count, max_points), dtype=np.float64)
    values = np.zeros((model_count, max_points), dtype=np.float64)
    point_counts = np.zeros(model_count, dtype=np.int32)
    ignore_spectrum = np.zeros(model_count, dtype=np.uint8)
    maximum_label = max(
        max(int(np.max(labels)), 0),
        max(label for label, *_ in packed_models),
    )
    label_to_model = np.full(maximum_label + 1, -1, dtype=np.int32)
    for model_index, (
        label,
        model_supports,
        model_distances,
        model_values,
        ignores_spectrum,
    ) in enumerate(packed_models):
        count = int(model_values.size)
        supports[model_index, :count] = model_supports
        distances[model_index, :count] = model_distances
        values[model_index, :count] = model_values
        point_counts[model_index] = count
        ignore_spectrum[model_index] = int(ignores_spectrum)
        label_to_model[label] = model_index

    return _CudaColorPathPack(
        labels=np.ascontiguousarray(labels),
        supports=np.ascontiguousarray(supports),
        distances=np.ascontiguousarray(distances),
        values=np.ascontiguousarray(values),
        point_counts=np.ascontiguousarray(point_counts),
        ignore_spectrum=np.ascontiguousarray(ignore_spectrum),
        label_to_model=np.ascontiguousarray(label_to_model),
        max_points=max_points,
    )


class CudaColorPathInterpreter:
    """Exact color-path projection with optional CUDA volume averaging."""

    def __init__(self, calibration: Any, cp: Any) -> None:
        analysis = getattr(calibration, "color_analysis", None)
        pack = _pack_cuda_color_paths(calibration)
        labels = pack.labels

        self.cp = cp
        self.analysis = analysis
        self.shape = tuple(labels.shape)
        self.pixel_count = int(labels.size)
        self.max_points = pack.max_points

        baseline = getattr(analysis, "base", None)
        self.has_baseline = baseline is not None
        baseline_values = (
            np.zeros((*self.shape, 3), dtype=np.float64)
            if baseline is None
            else np.asarray(baseline.img, dtype=np.float64)
        )
        if baseline_values.shape != (*self.shape, 3):
            raise ValueError(
                "CUDA color baseline shape differs from the active labels."
            )

        self.image = cp.empty((*self.shape, 3), dtype=cp.float64)
        self.baseline = cp.asarray(
            np.ascontiguousarray(baseline_values),
            dtype=cp.float64,
        )
        self.labels = cp.asarray(pack.labels)
        self.label_to_model = cp.asarray(pack.label_to_model)
        self.supports = cp.asarray(pack.supports.ravel())
        self.distances = cp.asarray(pack.distances.ravel())
        self.values = cp.asarray(pack.values.ravel())
        self.point_counts = cp.asarray(pack.point_counts)
        self.ignore_spectrum = cp.asarray(pack.ignore_spectrum)
        self.output = cp.empty(self.shape, dtype=cp.float64)
        self.kernel = cp.RawKernel(
            CUDA_COLOR_PATH_KERNEL,
            "color_path_interpretation",
        )

        restoration = getattr(analysis, "restoration", None)
        self.restoration = restoration
        self.restoration_supported = restoration is None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.restoration_scratch = None
        if restoration is not None:
            if not all(
                hasattr(restoration, name)
                for name in (
                    "rev_size",
                    "mask",
                    "mean_pore_volume",
                    "zero_indices",
                )
            ) or getattr(restoration, "labels", None) is not None:
                raise ValueError(
                    "CUDA color interpretation only supports homogeneous "
                    "VolumeAveraging restoration."
                )
            try:
                import cupyx.scipy.ndimage as cpx_ndimage
            except ImportError as exc:
                raise ValueError(
                    "CUDA volume averaging requires cupyx.scipy.ndimage."
                ) from exc
            self.cpx_ndimage = cpx_ndimage
            self.restoration_mask = cp.asarray(
                np.asarray(restoration.mask.img, dtype=np.float64)
            )
            self.restoration_mean = cp.asarray(
                np.asarray(restoration.mean_pore_volume, dtype=np.float64)
            )
            zero_mask = np.zeros(self.shape, dtype=bool)
            zero_mask[restoration.zero_indices] = True
            self.restoration_zero = cp.asarray(zero_mask)
            self.restoration_scratch = cp.empty(
                self.shape,
                dtype=cp.float64,
            )
            self.restoration_supported = True

    def __call__(self, image: darsia.Image) -> Any:
        values = np.asarray(image.img)
        if (
            values.shape != (*self.shape, 3)
            or not np.issubdtype(values.dtype, np.floating)
        ):
            raise ValueError(
                "CUDA color interpretation requires a floating-point RGB image "
                f"with shape {(*self.shape, 3)}, got {values.shape}/{values.dtype}."
            )
        self.image.set(np.ascontiguousarray(values, dtype=np.float64))
        threads = 256
        blocks = (self.pixel_count + threads - 1) // threads
        self.kernel(
            (blocks,),
            (threads,),
            (
                self.image,
                self.baseline,
                np.int32(self.has_baseline),
                self.labels,
                np.int64(self.pixel_count),
                self.label_to_model,
                np.int32(self.label_to_model.size),
                self.supports,
                self.distances,
                self.values,
                self.point_counts,
                self.ignore_spectrum,
                np.int32(self.max_points),
                self.output,
            ),
        )
        if self.restoration is not None:
            cp = self.cp
            cp.multiply(
                self.output,
                self.restoration_mask,
                out=self.restoration_scratch,
            )
            self.cpx_ndimage.uniform_filter(
                self.restoration_scratch,
                size=int(self.restoration.rev_size),
                output=self.output,
            )
            cp.divide(self.output, self.restoration_mean, out=self.output)
            self.output[self.restoration_zero] = 0.0
        return self.output

    def close(self) -> None:
        for name in (
            "image",
            "baseline",
            "labels",
            "label_to_model",
            "supports",
            "distances",
            "values",
            "point_counts",
            "ignore_spectrum",
            "output",
            "restoration_mask",
            "restoration_mean",
            "restoration_zero",
            "restoration_scratch",
        ):
            setattr(self, name, None)
        self.kernel = None


class CudaSpatialMassEvaluator:
    """Compute intensive spatial phase and mass fields with CuPy."""

    backend_name = "cuda"

    def __init__(
        self,
        calibration: Any,
        *,
        output_dtype: str = "float32",
        device_index: int | None = None,
        color_interpretation: str | None = None,
    ) -> None:
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CUDA mass analysis requires the optional 'cuda' dependency."
            ) from exc
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA mass analysis requested, but no device is available.")

        self.cp = cp
        self.calibration = calibration
        self.output_dtype = _normalise_dtype(output_dtype)
        self.device_index = int(
            os.environ.get("FFAC_CUDA_DEVICE", "0")
            if device_index is None
            else device_index
        )
        self.device = cp.cuda.Device(self.device_index)
        self.device.use()
        self.last_timing: dict[str, float] = {}
        labels = np.asarray(calibration.labels.img)
        self.shape = tuple(labels.shape[:2])
        self.restoration_spec = signal_restoration_spec(
            calibration,
            self.shape,
        )
        self.cuda_color_mode = str(
            color_interpretation
            if color_interpretation is not None
            else os.environ.get(
                "FFAC_MASS_CUDA_COLOR_INTERPRETATION",
                "off",
            )
        ).strip().lower()
        if self.cuda_color_mode not in {"auto", "on", "off"}:
            raise ValueError(
                "FFAC_MASS_CUDA_COLOR_INTERPRETATION must be auto, on, or off."
            )
        self.color_interpreter: CudaColorPathInterpreter | None = None
        if self.cuda_color_mode != "off":
            try:
                self.color_interpreter = CudaColorPathInterpreter(
                    calibration,
                    cp,
                )
            except Exception as exc:
                if self.cuda_color_mode == "on":
                    raise RuntimeError(
                        "Explicit CUDA color interpretation could not be prepared."
                    ) from exc
                logger.info(
                    "CUDA color interpretation unavailable; using CPU: %s",
                    exc,
                )

        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError(
                "CUDA spatial mass analysis does not support restoration inside flash."
            )
        self.flash = flash
        self.models = _active_signal_models(calibration)
        self.label_data: dict[int, dict[str, Any]] = {}
        for label, model in self.models.items():
            mask = labels == label
            if not np.any(mask):
                continue
            self.label_data[label] = {
                "mask": cp.asarray(mask),
                "supports": cp.asarray(
                    np.asarray(model.supports, dtype=np.float64)
                ),
                "values": cp.asarray(np.asarray(model.values, dtype=np.float64)),
            }

        density, solubility = _broadcast_static_mass_arrays(
            calibration, self.shape
        )
        self.density = cp.asarray(np.ascontiguousarray(density))
        self.solubility = cp.asarray(np.ascontiguousarray(solubility))
        gas_mask, aqueous_mask, constraint_mode, constraint_expansion_m = (
            _expert_constraint_weights(calibration, calibration.labels)
        )
        self.constraint_mode = constraint_mode
        self.constraint_expansion_m = constraint_expansion_m
        self.gas_constraint = cp.asarray(gas_mask, dtype=cp.float32)
        self.aqueous_constraint = cp.asarray(aqueous_mask, dtype=cp.float32)
        lut_y, lut_caq = _flash_lut(flash)
        self.lut_y = cp.asarray(lut_y)
        self.lut_caq = cp.asarray(lut_caq)
        self.signal = cp.empty(self.shape, dtype=cp.float64)
        self.transformed_signal = cp.empty(self.shape, dtype=cp.float64)
        self.device_fields = cp.empty((4, *self.shape), dtype=cp.float64)
        self.transfer_fields = (
            self.device_fields
            if self.output_dtype == np.dtype(np.float64)
            else cp.empty((4, *self.shape), dtype=cp.float32)
        )
        self.scratch = cp.empty(self.shape, dtype=cp.float64)
        self.restoration_filter = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.restoration_scratch = None
        self.restoration_output = None
        if self.restoration_spec is not None:
            try:
                import cupyx.scipy.ndimage as cpx_ndimage
            except ImportError as exc:
                raise RuntimeError(
                    "CUDA signal restoration requires cupyx.scipy.ndimage."
                ) from exc
            self.restoration_filter = cpx_ndimage.uniform_filter
            self.restoration_mask = cp.asarray(self.restoration_spec.mask)
            safe_mean = self.restoration_spec.mean_pore_volume.copy()
            safe_mean[self.restoration_spec.zero_mask] = 1.0
            self.restoration_mean = cp.asarray(safe_mean)
            self.restoration_zero = cp.asarray(self.restoration_spec.zero_mask)
            self.restoration_scratch = cp.empty(self.shape, dtype=cp.float64)
            self.restoration_output = cp.empty(self.shape, dtype=cp.float64)
        cp.cuda.get_current_stream().synchronize()

        properties = cp.cuda.runtime.getDeviceProperties(self.device_index)
        name = properties["name"]
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)
        logger.info(
            "CUDA spatial mass evaluator ready on %s; shape=%s labels=%s "
            "dtype=%s color=%s constraint=%s/%.3fm restoration=%s",
            self.device_name,
            self.shape,
            sorted(self.label_data),
            self.output_dtype,
            "cuda" if self.color_interpreter is not None else "cpu",
            self.constraint_mode,
            self.constraint_expansion_m,
            (
                f"signal/rev{self.restoration_spec.rev_size}"
                if self.restoration_spec is not None
                else "off"
            ),
        )

    def evaluate(self, image: Any) -> Any:
        total_started = time.perf_counter()
        cp = self.cp
        started = time.perf_counter()
        reference = image
        if self.color_interpreter is None:
            reference = _call_color_interpretation_no_copy(
                self.calibration,
                image,
            )
            color_seconds = time.perf_counter() - started
            started = time.perf_counter()
            host_signal = np.ascontiguousarray(
                np.asarray(reference.img, dtype=np.float64)
            )
            if tuple(host_signal.shape[:2]) != self.shape:
                raise RuntimeError(
                    f"CUDA signal shape {host_signal.shape[:2]} differs "
                    f"from {self.shape}."
                )
            self.signal.set(host_signal)
            signal = self.signal
            upload_seconds = time.perf_counter() - started
        else:
            signal = self.color_interpreter(image)
            color_seconds = time.perf_counter() - started
            upload_seconds = 0.0

        noise_gate = _aqueous_noise_gate(self.calibration)
        if noise_gate is not None:
            for label, data in self.label_data.items():
                threshold = noise_gate.threshold_for(label)
                if threshold is None:
                    continue
                mask = data["mask"]
                selected = signal[mask]
                width = max(
                    float(threshold.high) - float(threshold.low),
                    np.finfo(float).eps,
                )
                unit = cp.clip(
                    (selected - float(threshold.low)) / width,
                    0.0,
                    1.0,
                )
                confidence = unit * unit * (3.0 - 2.0 * unit)
                signal[mask] = selected * confidence

        started = time.perf_counter()
        concentration = self.device_fields[0]
        saturation = self.device_fields[1]
        mass_g = self.device_fields[2]
        mass_aq = self.device_fields[3]
        concentration.fill(0.0)
        saturation.fill(0.0)
        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        aqueous_blend = min(
            1.0,
            max(0.0, float(getattr(self.flash, "aqueous_blend", 0.0))),
        )
        aqueous_denominator = (max_aq - min_aq) or 1.0
        gas_denominator = (max_g - min_g) or 1.0

        self.transformed_signal.fill(0.0)
        for data in self.label_data.values():
            mask = data["mask"]
            self.transformed_signal[mask] = cp.interp(
                signal[mask], data["supports"], data["values"]
            )
        transformed_signal = self.transformed_signal
        if self.restoration_spec is not None:
            cp.multiply(
                self.transformed_signal,
                self.restoration_mask,
                out=self.restoration_scratch,
            )
            self.restoration_filter(
                self.restoration_scratch,
                size=self.restoration_spec.rev_size,
                output=self.restoration_output,
            )
            cp.divide(
                self.restoration_output,
                self.restoration_mean,
                out=self.restoration_output,
            )
            self.restoration_output[self.restoration_zero] = 0.0
            transformed_signal = self.restoration_output

        if self.restoration_spec is not None:
            # Signal-stage restoration deliberately crosses label boundaries.
            # DarSIA flashes the complete restored field, including values that
            # have spread into labels without their own color-path model.
            yellow = cp.clip(
                (transformed_signal - min_aq) / aqueous_denominator, 0.0, 1.0
            )
            physical_concentration = cp.interp(yellow, self.lut_y, self.lut_caq)
            concentration[...] = (
                (1.0 - aqueous_blend) * physical_concentration
                + aqueous_blend * yellow
            )
            saturation[...] = (
                cp.clip(transformed_signal, min_g, max_g) - min_g
            ) / gas_denominator
        else:
            for data in self.label_data.values():
                mask = data["mask"]
                transformed = transformed_signal[mask]
                yellow = cp.clip(
                    (transformed - min_aq) / aqueous_denominator, 0.0, 1.0
                )
                physical_concentration = cp.interp(
                    yellow, self.lut_y, self.lut_caq
                )
                concentration[mask] = (
                    (1.0 - aqueous_blend) * physical_concentration
                    + aqueous_blend * yellow
                )
                saturation[mask] = (
                    cp.clip(transformed, min_g, max_g) - min_g
                ) / gas_denominator

        concentration *= self.aqueous_constraint
        saturation *= self.gas_constraint
        cp.multiply(self.density, saturation, out=mass_g)
        cp.subtract(1.0, saturation, out=self.scratch)
        cp.maximum(self.scratch, 0.0, out=self.scratch)
        cp.multiply(self.solubility, concentration, out=mass_aq)
        cp.multiply(mass_aq, self.scratch, out=mass_aq)

        if self.transfer_fields is not self.device_fields:
            self.transfer_fields[...] = self.device_fields
        host_fields = cp.asnumpy(self.transfer_fields)
        compute_download_seconds = time.perf_counter() - started
        result = _mass_result(
            reference,
            *host_fields,
            output_dtype=self.output_dtype,
        )
        self.last_timing = {
            (
                "color_interpretation_cuda_enqueue_s"
                if self.color_interpreter is not None
                else "color_interpretation_cpu_s"
            ): color_seconds,
            "signal_upload_s": upload_seconds,
            "accelerator_compute_download_s": compute_download_seconds,
            "backend_total_s": time.perf_counter() - total_started,
        }
        return result

    def close(self) -> None:
        self.label_data.clear()
        if self.color_interpreter is not None:
            self.color_interpreter.close()
            self.color_interpreter = None
        for name in (
            "density",
            "solubility",
            "gas_constraint",
            "aqueous_constraint",
            "lut_y",
            "lut_caq",
            "signal",
            "transformed_signal",
            "device_fields",
            "transfer_fields",
            "scratch",
            "restoration_mask",
            "restoration_mean",
            "restoration_zero",
            "restoration_scratch",
            "restoration_output",
        ):
            setattr(self, name, None)
        self.restoration_filter = None
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "device_index": self.device_index,
            "device": self.device_name,
            "cupy": self.cp.__version__,
            "output_dtype": self.output_dtype.name,
            "color_interpretation": (
                "cuda" if self.color_interpreter is not None else "cpu"
            ),
            "aqueous_noise_gate": _aqueous_noise_gate(self.calibration) is not None,
            "expert_constraint_mode": self.constraint_mode,
            "expert_constraint_expansion_m": self.constraint_expansion_m,
        }


OPENCL_SPATIAL_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE double
#endif

inline double interp_linear(
    const double x,
    __global const double *xp,
    __global const double *fp,
    const int count
) {
    if (x <= xp[0]) return fp[0];
    if (x >= xp[count - 1]) return fp[count - 1];
    int high = 1;
    while (high < count && x > xp[high]) ++high;
    const int low = high - 1;
    const double denominator = xp[high] - xp[low];
    if (denominator == 0.0) return fp[high];
    const double fraction = (x - xp[low]) / denominator;
    return fp[low] + fraction * (fp[high] - fp[low]);
}

inline double clamp_unit(const double value) {
    return fmin(1.0, fmax(0.0, value));
}

inline int reflect_index(int index, const int length) {
    if (length <= 1) return 0;
    while (index < 0 || index >= length) {
        index = index < 0 ? -index - 1 : 2 * length - index - 1;
    }
    return index;
}

__kernel void map_spatial_signal(
    __global const double *signal,
    __global const uint *indices,
    const ulong pixel_count,
    __global const double *supports,
    __global const double *values,
    const int support_count,
    __global double *mapped
) {
    const ulong local_index = get_global_id(0);
    if (local_index >= pixel_count) return;
    const uint source_index = indices[local_index];
    mapped[source_index] = interp_linear(
        signal[source_index], supports, values, support_count
    );
}

__kernel void volume_average_horizontal(
    __global const double *signal,
    __global const double *mask,
    const int width,
    const int height,
    const int window,
    __global double *scratch
) {
    const ulong index = get_global_id(0);
    const ulong count = (ulong)width * (ulong)height;
    if (index >= count) return;
    const int y = (int)(index / (ulong)width);
    const int x = (int)(index - (ulong)y * (ulong)width);
    const int radius = window / 2;
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const int source_x = reflect_index(x + offset, width);
        const ulong source = (ulong)y * (ulong)width + (ulong)source_x;
        sum += signal[source] * mask[source];
    }
    scratch[index] = sum / (double)window;
}

__kernel void volume_average_vertical(
    __global const double *scratch,
    __global const double *mean_pore_volume,
    __global const uchar *zero_mask,
    const int width,
    const int height,
    const int window,
    __global double *restored
) {
    const ulong index = get_global_id(0);
    const ulong count = (ulong)width * (ulong)height;
    if (index >= count) return;
    if (zero_mask[index] != 0) {
        restored[index] = 0.0;
        return;
    }
    const int y = (int)(index / (ulong)width);
    const int x = (int)(index - (ulong)y * (ulong)width);
    const int radius = window / 2;
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const int source_y = reflect_index(y + offset, height);
        const ulong source = (ulong)source_y * (ulong)width + (ulong)x;
        sum += scratch[source];
    }
    restored[index] = (sum / (double)window) / mean_pore_volume[index];
}

__kernel void spatial_mass(
    __global const double *signal,
    __global const uint *indices,
    const ulong pixel_count,
    const int signal_is_mapped,
    __global const double *supports,
    __global const double *values,
    const int support_count,
    __global const double *lut_y,
    __global const double *lut_caq,
    const int lut_count,
    __global const double *density,
    __global const double *solubility,
    __global const float *gas_constraint,
    __global const float *aqueous_constraint,
    const double aqueous_blend,
    const double min_aq,
    const double max_aq,
    const double min_g,
    const double max_g,
    __global OUTPUT_TYPE *out_concentration,
    __global OUTPUT_TYPE *out_saturation,
    __global OUTPUT_TYPE *out_mass_g,
    __global OUTPUT_TYPE *out_mass_aq
) {
    const ulong local_index = get_global_id(0);
    if (local_index >= pixel_count) return;
    const uint source_index = indices[local_index];
    const double transformed = signal_is_mapped != 0
        ? signal[source_index]
        : interp_linear(signal[source_index], supports, values, support_count);
    const double aqueous_denominator =
        max_aq == min_aq ? 1.0 : max_aq - min_aq;
    const double gas_denominator =
        max_g == min_g ? 1.0 : max_g - min_g;
    const double yellow = clamp_unit(
        (transformed - min_aq) / aqueous_denominator
    );
    const double physical_concentration = interp_linear(
        yellow, lut_y, lut_caq, lut_count
    );
    double concentration = (
        (1.0 - aqueous_blend) * physical_concentration
        + aqueous_blend * yellow
    );
    double saturation = (
        fmin(max_g, fmax(min_g, transformed)) - min_g
    ) / gas_denominator;
    concentration *= (double)aqueous_constraint[source_index];
    saturation *= (double)gas_constraint[source_index];
    out_concentration[source_index] = (OUTPUT_TYPE) concentration;
    out_saturation[source_index] = (OUTPUT_TYPE) saturation;
    out_mass_g[source_index] = (OUTPUT_TYPE) (
        density[source_index] * saturation
    );
    out_mass_aq[source_index] = (OUTPUT_TYPE) (
        solubility[source_index]
        * concentration
        * fmax(0.0, 1.0 - saturation)
    );
}
"""


def _select_opencl_device(cl: Any) -> tuple[Any, Any]:
    platform_filter = os.environ.get("FFAC_OPENCL_PLATFORM", "").strip().lower()
    device_filter = os.environ.get("FFAC_OPENCL_DEVICE", "").strip().lower()
    candidates: list[tuple[Any, Any]] = []
    for candidate_platform in cl.get_platforms():
        if platform_filter and platform_filter not in candidate_platform.name.lower():
            continue
        for device in candidate_platform.get_devices():
            if not device.type & cl.device_type.GPU:
                continue
            identity = " ".join(
                (
                    candidate_platform.name,
                    candidate_platform.vendor,
                    device.name,
                    device.vendor,
                )
            ).lower()
            if device_filter and device_filter not in identity:
                continue
            candidates.append((candidate_platform, device))
    if not candidates:
        raise RuntimeError(
            "No matching OpenCL GPU was found "
            f"(platform={platform_filter!r}, device={device_filter!r})."
        )
    for candidate in candidates:
        identity = f"{candidate[0].vendor} {candidate[1].vendor} {candidate[1].name}"
        if "amd" in identity.lower() or "radeon" in identity.lower():
            return candidate
    return candidates[0]


class OpenCLSpatialMassEvaluator:
    """Compute intensive spatial phase and mass fields with PyOpenCL."""

    backend_name = "opencl"

    def __init__(self, calibration: Any, *, output_dtype: str = "float32") -> None:
        try:
            import pyopencl as cl
        except ImportError as exc:
            raise RuntimeError(
                "OpenCL mass analysis requires the optional 'opencl' dependency."
            ) from exc
        platform_, device = _select_opencl_device(cl)
        if "cl_khr_fp64" not in device.extensions.split():
            raise RuntimeError(
                f"OpenCL device {device.name!r} does not provide cl_khr_fp64."
            )

        self.cl = cl
        self.calibration = calibration
        self.output_dtype = _normalise_dtype(output_dtype)
        labels = np.asarray(calibration.labels.img)
        self.shape = tuple(labels.shape[:2])
        self.size = int(np.prod(self.shape))
        self.restoration_spec = signal_restoration_spec(
            calibration,
            self.shape,
        )
        if (
            self.restoration_spec is not None
            and self.restoration_spec.rev_size % 2 == 0
        ):
            raise RuntimeError(
                "OpenCL signal restoration requires an odd REV size; got "
                f"{self.restoration_spec.rev_size}."
            )
        self.platform = platform_
        self.device = device
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.last_timing: dict[str, float] = {}
        output_type = "float" if self.output_dtype == np.dtype(np.float32) else "double"
        self.program = cl.Program(self.context, OPENCL_SPATIAL_KERNEL).build(
            options=[f"-DOUTPUT_TYPE={output_type}"]
        )
        self.kernel = cl.Kernel(self.program, "spatial_mass")
        self.map_signal_kernel = cl.Kernel(self.program, "map_spatial_signal")
        self.restoration_horizontal_kernel = cl.Kernel(
            self.program,
            "volume_average_horizontal",
        )
        self.restoration_vertical_kernel = cl.Kernel(
            self.program,
            "volume_average_vertical",
        )

        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError(
                "OpenCL spatial mass analysis does not support restoration inside flash."
            )
        self.flash = flash
        flags = cl.mem_flags

        def readonly(values: Any, dtype: Any) -> Any:
            array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
            return cl.Buffer(
                self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array
            )

        density, solubility = _broadcast_static_mass_arrays(
            calibration, self.shape
        )
        gas_mask, aqueous_mask, constraint_mode, constraint_expansion_m = (
            _expert_constraint_weights(calibration, calibration.labels)
        )
        self.constraint_mode = constraint_mode
        self.constraint_expansion_m = constraint_expansion_m
        self.density = readonly(density.ravel(), np.float64)
        self.solubility = readonly(solubility.ravel(), np.float64)
        self.gas_constraint = readonly(gas_mask.ravel(), np.float32)
        self.aqueous_constraint = readonly(aqueous_mask.ravel(), np.float32)
        lut_y, lut_caq = _flash_lut(flash)
        self.lut_y = readonly(lut_y, np.float64)
        self.lut_caq = readonly(lut_caq, np.float64)
        self.lut_count = int(lut_y.size)
        self.signal = cl.Buffer(
            self.context, flags.READ_ONLY, size=self.size * np.dtype(np.float64).itemsize
        )
        self.all_indices = readonly(np.arange(self.size), np.uint32)
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        if self.restoration_spec is not None:
            restoration_bytes = self.size * np.dtype(np.float64).itemsize
            self.restoration_signal = cl.Buffer(
                self.context,
                flags.READ_WRITE,
                size=restoration_bytes,
            )
            self.restoration_scratch = cl.Buffer(
                self.context,
                flags.READ_WRITE,
                size=restoration_bytes,
            )
            self.restoration_output = cl.Buffer(
                self.context,
                flags.READ_WRITE,
                size=restoration_bytes,
            )
            self.restoration_mask = readonly(
                self.restoration_spec.mask.ravel(),
                np.float64,
            )
            self.restoration_mean = readonly(
                self.restoration_spec.mean_pore_volume.ravel(),
                np.float64,
            )
            self.restoration_zero = readonly(
                self.restoration_spec.zero_mask.ravel(),
                np.uint8,
            )
        output_bytes = int(self.size * self.output_dtype.itemsize)
        self.outputs = [
            cl.Buffer(self.context, flags.READ_WRITE, size=output_bytes)
            for _ in range(4)
        ]

        self.label_data: dict[int, dict[str, Any]] = {}
        for label, model in _active_signal_models(calibration).items():
            indices = np.flatnonzero(labels.ravel() == label).astype(np.uint32)
            if indices.size == 0:
                continue
            supports = np.asarray(model.supports, dtype=np.float64)
            values = np.asarray(model.values, dtype=np.float64)
            self.label_data[label] = {
                "indices": readonly(indices, np.uint32),
                "supports": readonly(supports, np.float64),
                "values": readonly(values, np.float64),
                "support_count": int(supports.size),
                "pixel_count": int(indices.size),
            }
        self.queue.finish()
        logger.info(
            "OpenCL spatial mass evaluator ready on %s / %s; "
            "shape=%s labels=%s dtype=%s constraint=%s/%.3fm restoration=%s",
            platform_.name,
            device.name,
            self.shape,
            sorted(self.label_data),
            self.output_dtype,
            self.constraint_mode,
            self.constraint_expansion_m,
            (
                f"signal/rev{self.restoration_spec.rev_size}"
                if self.restoration_spec is not None
                else "off"
            ),
        )

    def evaluate(self, image: Any) -> Any:
        total_started = time.perf_counter()
        cl = self.cl
        started = time.perf_counter()
        color = _call_color_interpretation_no_copy(self.calibration, image)
        color = _apply_aqueous_noise_gate(self.calibration, color)
        color_seconds = time.perf_counter() - started
        started = time.perf_counter()
        signal = np.ascontiguousarray(
            np.asarray(color.img, dtype=np.float64).ravel()
        )
        if signal.size != self.size:
            raise RuntimeError(
                f"OpenCL signal size {signal.size} differs from {self.size}."
            )
        cl.enqueue_copy(self.queue, self.signal, signal, is_blocking=False)
        upload_enqueue_seconds = time.perf_counter() - started
        started = time.perf_counter()
        zero = np.zeros(1, dtype=self.output_dtype)
        output_bytes = int(self.size * self.output_dtype.itemsize)
        for buffer in self.outputs:
            cl.enqueue_fill_buffer(
                self.queue,
                buffer,
                zero,
                0,
                output_bytes,
            )

        mapped_signal = None
        if self.restoration_spec is not None:
            restoration_bytes = self.size * np.dtype(np.float64).itemsize
            cl.enqueue_fill_buffer(
                self.queue,
                self.restoration_signal,
                np.zeros(1, dtype=np.float64),
                0,
                restoration_bytes,
            )
            for data in self.label_data.values():
                count = data["pixel_count"]
                global_size = ((count + 255) // 256 * 256,)
                self.map_signal_kernel(
                    self.queue,
                    global_size,
                    (256,),
                    self.signal,
                    data["indices"],
                    np.uint64(count),
                    data["supports"],
                    data["values"],
                    np.int32(data["support_count"]),
                    self.restoration_signal,
                )
            full_global_size = ((self.size + 255) // 256 * 256,)
            height, width = self.shape
            self.restoration_horizontal_kernel(
                self.queue,
                full_global_size,
                (256,),
                self.restoration_signal,
                self.restoration_mask,
                np.int32(width),
                np.int32(height),
                np.int32(self.restoration_spec.rev_size),
                self.restoration_scratch,
            )
            self.restoration_vertical_kernel(
                self.queue,
                full_global_size,
                (256,),
                self.restoration_scratch,
                self.restoration_mean,
                self.restoration_zero,
                np.int32(width),
                np.int32(height),
                np.int32(self.restoration_spec.rev_size),
                self.restoration_output,
            )
            mapped_signal = self.restoration_output

        min_aq = np.float64(self.flash.min_value_aq)
        max_aq = np.float64(self.flash.max_value_aq)
        min_g = np.float64(self.flash.min_value_g)
        max_g = np.float64(self.flash.max_value_g)
        aqueous_blend = np.float64(
            min(
                1.0,
                max(0.0, float(getattr(self.flash, "aqueous_blend", 0.0))),
            )
        )
        phase_rows = (
            [
                {
                    "indices": self.all_indices,
                    "pixel_count": self.size,
                    # Ignored by the kernel when signal_is_mapped is true.
                    "supports": next(iter(self.label_data.values()))["supports"],
                    "values": next(iter(self.label_data.values()))["values"],
                    "support_count": next(iter(self.label_data.values()))[
                        "support_count"
                    ],
                }
            ]
            if mapped_signal is not None
            else self.label_data.values()
        )
        for data in phase_rows:
            count = data["pixel_count"]
            global_size = ((count + 255) // 256 * 256,)
            self.kernel(
                self.queue,
                global_size,
                (256,),
                mapped_signal if mapped_signal is not None else self.signal,
                data["indices"],
                np.uint64(count),
                np.int32(1 if mapped_signal is not None else 0),
                data["supports"],
                data["values"],
                np.int32(data["support_count"]),
                self.lut_y,
                self.lut_caq,
                np.int32(self.lut_count),
                self.density,
                self.solubility,
                self.gas_constraint,
                self.aqueous_constraint,
                aqueous_blend,
                min_aq,
                max_aq,
                min_g,
                max_g,
                *self.outputs,
            )

        fields = np.empty((4, self.size), dtype=self.output_dtype)
        events = [
            cl.enqueue_copy(
                self.queue,
                fields[index],
                buffer,
                is_blocking=False,
            )
            for index, buffer in enumerate(self.outputs)
        ]
        cl.wait_for_events(events)
        reshaped = [field.reshape(self.shape) for field in fields]
        compute_download_seconds = time.perf_counter() - started
        result = _mass_result(
            color,
            *reshaped,
            output_dtype=self.output_dtype,
        )
        self.last_timing = {
            "color_interpretation_cpu_s": color_seconds,
            "signal_upload_enqueue_s": upload_enqueue_seconds,
            "accelerator_compute_download_s": compute_download_seconds,
            "backend_total_s": time.perf_counter() - total_started,
        }
        return result

    def close(self) -> None:
        try:
            self.queue.finish()
        except Exception:
            pass
        self.label_data.clear()
        self.outputs = []
        self.all_indices = None
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.map_signal_kernel = None
        self.restoration_horizontal_kernel = None
        self.restoration_vertical_kernel = None
        self.kernel = None
        self.program = None
        self.queue = None
        self.context = None

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "platform": self.platform.name,
            "device": self.device.name,
            "driver": self.device.driver_version,
            "output_dtype": self.output_dtype.name,
            "aqueous_noise_gate": _aqueous_noise_gate(self.calibration) is not None,
            "expert_constraint_mode": self.constraint_mode,
            "expert_constraint_expansion_m": self.constraint_expansion_m,
        }


class AcceleratedColorToMassAnalysis:
    """Delegate metadata to DarSIA and route only ``__call__`` to a backend."""

    def __init__(self, calibration: Any, evaluator: Any) -> None:
        self._calibration = calibration
        self._evaluator = evaluator

    def __call__(self, image: Any) -> Any:
        return self._evaluator.evaluate(image)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._calibration, name)

    @property
    def last_timing(self) -> dict[str, float]:
        return dict(getattr(self._evaluator, "last_timing", {}))


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _opencl_available() -> bool:
    try:
        import pyopencl as cl

        return any(
            device.type & cl.device_type.GPU
            for platform_ in cl.get_platforms()
            for device in platform_.get_devices()
        )
    except Exception:
        return False


def create_spatial_mass_evaluator(
    calibration: Any,
    *,
    backend: str = "cpu",
    output_dtype: str = "float32",
    cuda_color_interpretation: str | None = None,
) -> Any:
    """Create an explicit spatial evaluator; ``auto`` prefers CUDA then OpenCL."""

    selected = _normalise_backend(backend)
    if selected == "auto":
        if _cuda_available():
            selected = "cuda"
        elif _opencl_available():
            selected = "opencl"
        else:
            selected = "cpu"
    if selected == "cuda":
        return CudaSpatialMassEvaluator(
            calibration,
            output_dtype=output_dtype,
            color_interpretation=cuda_color_interpretation,
        )
    if selected == "opencl":
        return OpenCLSpatialMassEvaluator(
            calibration, output_dtype=output_dtype
        )
    return CpuSpatialMassEvaluator(calibration, output_dtype=output_dtype)


def ensure_titration_flash(context: Any, mode: str = "auto") -> dict[str, Any]:
    """Load the ff_ac titration model without relying on a modified DarSIA checkout."""

    selected = str(mode).strip().lower()
    if selected not in {"auto", "on", "off"}:
        raise ValueError("Titration flash mode must be auto, on, or off.")

    calibration = context.color_to_mass_analysis
    flash = calibration.flash
    flash_path: Path | None = None
    saved: dict[str, Any] = {}
    try:
        mass_config = context.config.analysis.mass
        embedding = context.config.color.resolve(mass_config.color)
        flash_path = Path(embedding.color_to_mass_folder) / "flash" / "flash.json"
        if flash_path.exists():
            saved = json.loads(flash_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not inspect the saved flash metadata.", exc_info=True)

    saved_titration = saved.get("model") == "titration"
    requested = (
        selected == "on"
        or (selected == "auto" and saved_titration)
        or (selected == "auto" and _truthy(os.environ.get("FFAC_TITRATION_FLASH")))
    )
    already_titration = hasattr(flash, "_lut_y") and hasattr(flash, "_lut_caq")

    if selected == "off" and already_titration:
        calibration.flash = darsia.SimpleFlash(
            min_value_aq=flash.min_value_aq,
            max_value_aq=flash.max_value_aq,
            min_value_g=flash.min_value_g,
            max_value_g=flash.max_value_g,
            restoration=getattr(flash, "restoration", None),
        )
        flash = calibration.flash
        already_titration = False
    elif requested and not already_titration:
        try:
            from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash
        except ImportError:
            from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash

        calibration.flash = TitrationFlash.from_simple(
            flash,
            aqueous_blend=float(saved.get("aqueous_blend", 0.0)),
            **dict(saved.get("titration_params") or {}),
        )
        flash = calibration.flash
        already_titration = True

    metadata = {
        "mode": selected,
        "enabled": already_titration,
        "class": type(flash).__name__,
        "saved_model": saved.get("model"),
        "saved_path": str(flash_path) if flash_path is not None else None,
        "source": (
            "saved"
            if saved_titration
            else "environment"
            if _truthy(os.environ.get("FFAC_TITRATION_FLASH"))
            else "explicit"
            if selected in {"on", "off"}
            else "simple"
        ),
    }
    logger.info(
        "Mass flash=%s enabled=%s source=%s",
        metadata["class"],
        metadata["enabled"],
        metadata["source"],
    )
    return metadata


def ensure_aqueous_noise_gate(context: Any, mode: str = "auto") -> dict[str, Any]:
    """Load a saved FF-AC observation gate for production mass analysis."""

    selected = str(mode or "auto").strip().lower().replace("_", "-")
    if selected not in {"auto", "on", "off", "baseline-soft"}:
        raise ValueError(
            "Aqueous noise-gate mode must be auto, on, off, or baseline-soft."
        )
    calibration = context.color_to_mass_analysis
    gate_path: Path | None = None
    try:
        mass_config = context.config.analysis.mass
        embedding = context.config.color.resolve(mass_config.color)
        gate_path = (
            Path(embedding.color_to_mass_folder)
            / "observation_model"
            / "aqueous_noise_gate.json"
        )
    except Exception:
        logger.debug("Could not resolve the saved observation-model path.", exc_info=True)

    if selected == "off":
        calibration.aqueous_noise_gate = None
        source = "disabled"
    elif gate_path is not None and gate_path.is_file():
        try:
            from darsia.fluidflower.tools.aqueous_noise_gate import AqueousNoiseGate
        except ImportError:
            from darsia.fluidflower.tools.aqueous_noise_gate import AqueousNoiseGate

        payload = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        calibration.aqueous_noise_gate = AqueousNoiseGate.from_dict(payload)
        source = "saved"
    elif selected in {"on", "baseline-soft"}:
        raise FileNotFoundError(
            f"Requested aqueous noise gate was not found: {gate_path}"
        )
    else:
        calibration.aqueous_noise_gate = None
        source = "none"

    gate = _aqueous_noise_gate(calibration)
    metadata = {
        "mode": selected,
        "enabled": gate is not None,
        "source": source,
        "path": str(gate_path) if gate_path is not None else None,
        "model": gate.to_dict() if gate is not None else None,
    }
    logger.info(
        "Aqueous noise gate enabled=%s source=%s",
        metadata["enabled"],
        source,
    )
    return metadata


def _git_state(path: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL,
        )
        return {
            "head": head,
            "dirty": bool(status.strip()),
            "status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {
            "head": None,
            "dirty": None,
            "status": [],
            "tracked_diff_sha256": None,
        }


def _array_sha256(array: Any) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _calibration_sha256(calibration: Any) -> str:
    payload: dict[str, Any] = {}
    for label, model in sorted(_active_signal_models(calibration).items()):
        payload[str(label)] = {
            "supports": np.asarray(model.supports, dtype=float).tolist(),
            "values": np.asarray(model.values, dtype=float).tolist(),
        }
    flash = calibration.flash
    payload["flash"] = {
        "class": type(flash).__name__,
        "min_value_aq": float(flash.min_value_aq),
        "max_value_aq": float(flash.max_value_aq),
        "min_value_g": float(flash.min_value_g),
        "max_value_g": float(flash.max_value_g),
    }
    if hasattr(flash, "_lut_y"):
        payload["flash"]["lut_y"] = np.asarray(flash._lut_y).tolist()
        payload["flash"]["lut_caq"] = np.asarray(flash._lut_caq).tolist()
    if hasattr(flash, "aqueous_blend"):
        payload["flash"]["aqueous_blend"] = float(flash.aqueous_blend)
    noise_gate = _aqueous_noise_gate(calibration)
    if noise_gate is not None:
        payload["aqueous_noise_gate"] = noise_gate.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def write_mass_backend_manifest(
    context: Any,
    evaluator: Any,
    *,
    path: Path | None = None,
    runtime: dict[str, Any] | None = None,
) -> Path:
    """Write runtime provenance for a mass-analysis run."""

    results = Path(context.config.data.results)
    destination = path or results / "provenance" / "mass_backend.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    repo = package_root()
    geometry = context.fluidflower.geometry
    depth = getattr(getattr(geometry, "depth", None), "img", None)
    source_files = [
        tool_path("analysis.py"),
        tool_path("spatial_mass_evaluator.py"),
        tool_path("ffac_titration_flash.py"),
    ]
    manifest = {
        "schema": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "darsia": getattr(darsia, "__version__", None),
        "backend": evaluator.metadata(),
        "ff_ac_git": _git_state(repo),
        "darsia_git": _git_state(repo / "external" / "darsia"),
        "source_sha256": {
            str(source.relative_to(repo)): _file_sha256(source)
            for source in source_files
        },
        "depth_sha256": _array_sha256(depth) if depth is not None else None,
        "calibration_sha256": _calibration_sha256(
            context.color_to_mass_analysis
        ),
        "image_count": len(context.image_paths),
        "runtime": dict(runtime or {}),
    }
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
