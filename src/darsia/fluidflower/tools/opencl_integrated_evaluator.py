"""OpenCL implementation of the prepared integrated-mass evaluator."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np

try:
    from darsia.fluidflower.tools.expert_constraint_support import constraint_weights_for_calibration
except ModuleNotFoundError:
    from darsia.fluidflower.tools.expert_constraint_support import constraint_weights_for_calibration

try:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
        DEFAULT_HYBRID_HALF_LIFE_H,
        IntegratedFrameResult,
        depth_model_arrays,
        resolve_depth_parameters,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
        DEFAULT_HYBRID_HALF_LIFE_H,
        IntegratedFrameResult,
        depth_model_arrays,
        resolve_depth_parameters,
    )

try:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight
except ModuleNotFoundError:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight

try:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec
except ModuleNotFoundError:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec


logger = logging.getLogger(__name__)


OPENCL_KERNEL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline double interp_linear(
    const double x,
    __global const double *xp,
    __global const double *fp,
    const int count
) {
    if (x <= xp[0]) {
        return fp[0];
    }
    if (x >= xp[count - 1]) {
        return fp[count - 1];
    }

    int high = 1;
    while (high < count && x > xp[high]) {
        ++high;
    }
    const int low = high - 1;
    const double denominator = xp[high] - xp[low];
    if (denominator == 0.0) {
        return fp[high];
    }
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

__kernel void map_signal(
    __global const double *signals,
    __global const double *supports,
    __global const double *values,
    const int support_count,
    __global const uint *indices,
    const ulong pixel_count,
    __global double *mapped
) {
    const ulong index = get_global_id(0);
    if (index >= pixel_count) return;
    mapped[indices[index]] = interp_linear(
        signals[index], supports, values, support_count
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

__kernel void phase_mass(
    __global const double *signals,
    __global const uint *indices,
    const int signal_is_mapped,
    __global const double *weight_g_per_m,
    __global const double *weight_aq_per_m,
    __global const double *pore_volume_per_m,
    __global const double *depth_shape,
    __global const float *gas_constraint,
    __global const float *hard_gas_constraint,
    __global const float *aqueous_constraint,
    __global const uchar *gas_score,
    __global const double *supports,
    __global const double *values,
    const int support_count,
    __global const double *lut_y,
    __global const double *lut_caq,
    const int lut_count,
    const double aqueous_blend,
    const double min_aq,
    const double max_aq,
    const double min_g,
    const double max_g,
    const float score_onset,
    const float score_width,
    const float arrival_scale,
    const float score_scale,
    const float hybrid_shared_weight,
    const float dual_max_saturation,
    const int phase_mode,
    const int apply_hard_gas_constraint,
    const double depth_edge_m,
    const double depth_bulge_m,
    const double depth_minimum_m,
    const double depth_maximum_m,
    const double aqueous_support_level,
    const int include_distribution,
    const int include_spatial,
    const ulong pixel_count,
    __global double *mass_g,
    __global double *mass_aq,
    __global double *aqueous_concentration,
    __global double *aqueous_c_volume,
    __global double *aqueous_c2_volume,
    __global double *aqueous_support_volume,
    __global double *aqueous_pore_volume
) {
    const ulong index = get_global_id(0);
    if (index >= pixel_count) {
        return;
    }

    const double signal = signal_is_mapped != 0
        ? signals[indices[index]]
        : interp_linear(signals[index], supports, values, support_count);
    const double aq_denominator = max_aq == min_aq ? 1.0 : max_aq - min_aq;
    const double gas_denominator = max_g == min_g ? 1.0 : max_g - min_g;
    const double y_norm = clamp_unit((signal - min_aq) / aq_denominator);
    const double physical_concentration = interp_linear(
        y_norm, lut_y, lut_caq, lut_count
    );
    double concentration_aq = (
        (1.0 - aqueous_blend) * physical_concentration
        + aqueous_blend * y_norm
    );
    concentration_aq *= (double)aqueous_constraint[index];

    const double flash_saturation = clamp_unit(
        (fmin(max_g, fmax(min_g, signal)) - min_g) / gas_denominator
    );
    double saturation_g;
    if (phase_mode != 0) {
        const float concentration_f = (float)concentration_aq;
        const float score = (float)gas_score[index] * score_scale;
        const float score_gate = clamp(
            (score - score_onset) / score_width, 0.0f, 1.0f
        );
        float saturation_f;
        if (phase_mode == 4) {
            const float hybrid_gate = (
                hybrid_shared_weight
                + (1.0f - hybrid_shared_weight) * score
            );
            saturation_f = (float)flash_saturation * hybrid_gate;
        } else if (phase_mode == 5) {
            saturation_f = score_gate * dual_max_saturation;
        } else if (phase_mode == 2) {
            saturation_f = (float)flash_saturation * score_gate;
        } else {
            saturation_f = score_gate * clamp(
                concentration_f / arrival_scale, 0.0f, 1.0f
            );
        }
        if (phase_mode >= 2) {
            saturation_f *= gas_constraint[index];
        }
        concentration_aq = (double)concentration_f;
        saturation_g = (double)saturation_f;
    } else {
        saturation_g = flash_saturation;
        saturation_g *= (double)gas_constraint[index];
    }
    if (apply_hard_gas_constraint != 0) {
        saturation_g *= (double)hard_gas_constraint[index];
    }

    const double effective_depth = fmin(
        depth_maximum_m,
        fmax(
            depth_minimum_m,
            depth_edge_m + depth_bulge_m * depth_shape[index]
        )
    );
    mass_g[index] = weight_g_per_m[index] * effective_depth * saturation_g;
    mass_aq[index] = (
        weight_aq_per_m[index]
        * effective_depth
        * concentration_aq
        * fmax(0.0, 1.0 - saturation_g)
    );
    if (include_distribution != 0 || include_spatial != 0) {
        aqueous_concentration[index] = concentration_aq;
        const double local_aqueous_pore_volume = (
            pore_volume_per_m[index]
            * effective_depth
            * fmax(0.0, 1.0 - saturation_g)
        );
        if (include_distribution != 0) {
            aqueous_c_volume[index] = (
                local_aqueous_pore_volume * concentration_aq
            );
            aqueous_c2_volume[index] = (
                local_aqueous_pore_volume * concentration_aq * concentration_aq
            );
            aqueous_support_volume[index] = (
                concentration_aq >= aqueous_support_level
                ? local_aqueous_pore_volume
                : 0.0
            );
        }
        if (include_spatial != 0) {
            aqueous_pore_volume[index] = local_aqueous_pore_volume;
        }
    }
}

__kernel void spatial_metrics(
    __global const double *mass_g,
    __global const double *mass_aq,
    __global const double *aqueous_concentration,
    __global const double *aqueous_pore_volume,
    __global const uchar *observation_confidence,
    __global const uchar *seal_mask,
    const double trace_level,
    const double core_level,
    const int metric_pass,
    const ulong pixel_count,
    __global double *output_0,
    __global double *output_1,
    __global double *output_2,
    __global double *output_3
) {
    const ulong index = get_global_id(0);
    if (index >= pixel_count) {
        return;
    }

    const float confidence_f = (float)observation_confidence[index] / 255.0f;
    const float unsupported_f = clamp(1.0f - confidence_f, 0.0f, 1.0f);
    const double confidence = (double)confidence_f;
    const double unsupported = (double)unsupported_f;
    const double aqueous_mass = mass_aq[index];
    const double total_mass = aqueous_mass + mass_g[index];
    const double concentration = aqueous_concentration[index];
    const double pore_volume = aqueous_pore_volume[index];

    if (metric_pass == 0) {
        const double observation_weight = pore_volume * confidence;
        const double recall_deficit = core_level > 0.0
            ? clamp_unit((core_level - concentration) / core_level)
            : 0.0;
        output_0[index] = aqueous_mass * unsupported;
        output_1[index] = total_mass * unsupported;
        output_2[index] = observation_weight * recall_deficit;
        output_3[index] = observation_weight;
    } else {
        const double seal = seal_mask[index] != 0 ? 1.0 : 0.0;
        const double predicted_support = concentration >= trace_level ? 1.0 : 0.0;
        output_0[index] = aqueous_mass * unsupported * seal;
        output_1[index] = total_mass * unsupported * seal;
        output_2[index] = pore_volume * predicted_support * unsupported;
        output_3[index] = pore_volume * predicted_support;
    }
}
"""


def _float32_kernel_source(source: str) -> str:
    """Build a strictly float32 OpenCL variant from the legacy kernel."""

    source = source.replace("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n", "")
    source = re.sub(r"\bdouble\b", "float", source)
    # OpenCL treats unsuffixed decimal constants as double. Avoid requesting
    # fp64 support after converting all data and scalar arguments to float.
    return re.sub(r"(?<![\w.])(\d+\.\d+)(?![\w.])", r"\1f", source)


OPENCL_FLOAT32_KERNEL_SOURCE = _float32_kernel_source(OPENCL_KERNEL_SOURCE)


def _normalise_phase_separation(value: Any) -> str:
    mode = str(value or "shared-signal").strip().lower().replace("_", "-")
    if mode in {"residual", "residual-gas"}:
        return "residual-gas"
    if mode in {"shared", "shared-signal"}:
        return "shared-signal"
    if mode in {"clear", "clearwater", "clearwater-direct", "wash-direct"}:
        return "clearwater-direct"
    if mode in {"clear-gated", "clearwater-gated", "wash-gated"}:
        return "clearwater-gated"
    if mode in {"hybrid", "causal-hybrid", "hybrid-causal"}:
        return "hybrid-causal"
    if mode in {"dual", "dual-signal", "separate-signals", "supervised-dual"}:
        return "supervised-dual"
    raise ValueError(f"Unknown phase separation mode {value!r}")


def _select_device(cl: Any) -> tuple[Any, Any]:
    platform_filter = os.environ.get("FFAC_OPENCL_PLATFORM", "").strip().lower()
    device_filter = os.environ.get("FFAC_OPENCL_DEVICE", "").strip().lower()
    candidates: list[tuple[Any, Any]] = []
    for platform in cl.get_platforms():
        if platform_filter and platform_filter not in platform.name.lower():
            continue
        for device in platform.get_devices():
            if not device.type & cl.device_type.GPU:
                continue
            identity = " ".join(
                (platform.name, platform.vendor, device.name, device.vendor)
            ).lower()
            if device_filter and device_filter not in identity:
                continue
            candidates.append((platform, device))

    if not candidates:
        filters = f"platform={platform_filter!r}, device={device_filter!r}"
        raise RuntimeError(f"No matching OpenCL GPU was found ({filters})")
    for platform, device in candidates:
        identity = f"{platform.vendor} {device.vendor} {device.name}".lower()
        if "amd" in identity or "radeon" in identity:
            return platform, device
    return candidates[0]


class OpenCLIntegratedEvaluator:
    """Evaluate integrated masses while prepared frames remain in GPU memory."""

    def __init__(
        self,
        context: Any,
        *,
        residual_score_clip: float,
        _precision: str = "legacy",
    ) -> None:
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array
            from pyopencl.tools import ImmediateAllocator, MemoryPool
        except ImportError as exc:
            raise RuntimeError(
                "OpenCL evaluation requires the optional 'opencl' dependency"
            ) from exc
        if not context._prepared_colors:
            raise RuntimeError("OpenCL evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError("OpenCL evaluation does not support flash restoration")

        precision = str(_precision).strip().lower()
        if precision not in {"legacy", "float32"}:
            raise ValueError(f"Unsupported OpenCL evaluator precision {precision!r}")
        self.precision = "mixed" if precision == "legacy" else "float32"
        self.numeric_dtype = np.dtype(
            np.float64 if precision == "legacy" else np.float32
        )
        self.scalar_dtype = self.numeric_dtype.type
        self.separate_hard_gas_constraint = precision == "float32"

        platform, device = _select_device(cl)
        if precision == "legacy" and "cl_khr_fp64" not in device.extensions.split():
            raise RuntimeError(
                f"OpenCL device {device.name!r} does not provide cl_khr_fp64"
            )

        self.cl = cl
        self.cl_array = cl_array
        self.context = context
        self.flash = flash
        self.phase_separation = _normalise_phase_separation(
            getattr(context, "phase_separation", "shared-signal")
        )
        self.arrival_scale = max(
            1e-6,
            float(
                os.environ.get(
                    "FFAC_PHASE_GAS_ARRIVAL_SCALE",
                    os.environ.get("FFAC_RESIDUAL_GAS_ARRIVAL_SCALE", "0.02"),
                )
            ),
        )
        self.score_scale = (
            1.0
            if self.phase_separation == "hybrid-causal"
            else float(getattr(context, "_gas_score_scale", residual_score_clip))
        ) / 255.0
        self.cl_context = cl.Context([device])
        self.queue = cl.CommandQueue(self.cl_context)
        self.allocator = MemoryPool(ImmediateAllocator(self.queue))
        kernel_source = (
            OPENCL_KERNEL_SOURCE
            if precision == "legacy"
            else OPENCL_FLOAT32_KERNEL_SOURCE
        )
        self.program = cl.Program(self.cl_context, kernel_source).build()
        self.map_signal_kernel = cl.Kernel(self.program, "map_signal")
        self.restoration_horizontal_kernel = cl.Kernel(
            self.program,
            "volume_average_horizontal",
        )
        self.restoration_vertical_kernel = cl.Kernel(
            self.program,
            "volume_average_vertical",
        )
        self.phase_kernel = cl.Kernel(self.program, "phase_mass")
        self.spatial_kernel = cl.Kernel(self.program, "spatial_metrics")
        self.platform = platform
        self.device = device

        shape = tuple(np.asarray(context._prepared_colors[0].img).shape[:2])
        self.shape = shape
        self.restoration_spec = signal_restoration_spec(calibration, shape)
        if (
            self.restoration_spec is not None
            and self.restoration_spec.rev_size % 2 == 0
        ):
            raise RuntimeError(
                "OpenCL signal restoration requires an odd REV size; got "
                f"{self.restoration_spec.rev_size}."
            )
        spatial_confidences = list(
            getattr(context, "_spatial_observation_confidences", []) or []
        )
        self.spatial_objective_active = bool(spatial_confidences)
        if self.spatial_objective_active and len(spatial_confidences) != len(
            context._prepared_colors
        ):
            raise RuntimeError(
                f"OpenCL spatial confidence count mismatch for {context.run}: "
                f"{len(spatial_confidences)} != {len(context._prepared_colors)}"
            )
        for frame_index, confidence in enumerate(spatial_confidences):
            if np.asarray(confidence).shape != shape:
                raise RuntimeError(
                    f"OpenCL spatial confidence shape mismatch for {context.run} "
                    f"frame {frame_index}: {np.asarray(confidence).shape} != {shape}"
                )
        context.geometry._prepare_cached_voxel_volume(list(shape))
        voxel_volume = np.asarray(context.geometry.cached_voxel_volume)
        density = np.asarray(calibration.co2_mass_analysis.density_gaseous_co2)
        solubility = np.asarray(calibration.co2_mass_analysis.solubility_co2)
        try:
            voxel_volume = np.broadcast_to(voxel_volume, shape)
            density = np.broadcast_to(density, shape)
            solubility = np.broadcast_to(solubility, shape)
        except ValueError as exc:
            raise RuntimeError(
                f"OpenCL static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, density={density.shape}, "
                f"solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"OpenCL labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        seal_mask = np.zeros(shape, dtype=np.uint8)
        if self.spatial_objective_active:
            seal_ids = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in os.environ.get(
                            "FFAC_SPATIAL_SEAL_FACIES_IDS", "1"
                        ).split(",")
                        if value.strip()
                    }
                )
            )
            facies = np.asarray(context.fluidflower.facies.img)
            if facies.shape != shape:
                raise RuntimeError(
                    f"OpenCL facies/frame shape mismatch for {context.run}: "
                    f"{facies.shape} != {shape}"
                )
            seal_mask = np.isin(facies, np.asarray(seal_ids)).astype(np.uint8)

        reference_frame = context._prepared_colors[0]
        constraint_weights = constraint_weights_for_calibration(
            calibration,
            reference_frame,
        )
        self.constraint_mode = constraint_weights.mode
        self.constraint_expansion_m = constraint_weights.expansion_m
        gas_constraint = np.asarray(constraint_weights.gas, dtype=np.float32)
        aqueous_constraint = np.asarray(
            constraint_weights.aqueous, dtype=np.float32
        )
        hard_gas_constraint = np.ones(shape, dtype=np.float32)
        hard_nonseal = getattr(context, "_hard_nonseal_gas_mask", None)
        if hard_nonseal is not None:
            hard_nonseal_array = np.asarray(hard_nonseal, dtype=np.float32)
            if hard_nonseal_array.shape != shape:
                raise RuntimeError(
                    f"OpenCL hard non-seal mask shape mismatch for {context.run}: "
                    f"{hard_nonseal_array.shape} != {shape}"
                )
            if self.separate_hard_gas_constraint:
                hard_gas_constraint = hard_nonseal_array
            else:
                # Preserve the historical mixed-precision OpenCL semantics.
                # In particular, residual-gas bypasses this folded soft mask.
                gas_constraint = gas_constraint * hard_nonseal_array
        if self.phase_separation == "hybrid-causal":
            nonseal = np.asarray(context._hybrid_nonseal_mask, dtype=np.float32)
            if nonseal.shape != shape:
                raise RuntimeError(
                    f"OpenCL hybrid non-seal mask shape mismatch for {context.run}: "
                    f"{nonseal.shape} != {shape}"
                )
            gas_constraint = np.asarray(gas_constraint, dtype=np.float32) * nonseal

        if self.phase_separation != "shared-signal":
            source_scores = (
                context._hybrid_gas_weights
                if self.phase_separation == "hybrid-causal"
                else context._gas_scores
            )
            if len(source_scores) != len(context._prepared_colors):
                raise RuntimeError(
                    f"OpenCL phase score count mismatch for {context.run}: "
                    f"{len(source_scores)} != "
                    f"{len(context._prepared_colors)}"
                )
        else:
            source_scores = []

        flags = cl.mem_flags
        resident_bytes = 0

        def readonly(values: Any, dtype: Any) -> tuple[Any, int]:
            array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
            buffer = cl.Buffer(
                self.cl_context,
                flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=array,
            )
            return buffer, int(array.nbytes)

        lut_y = (
            np.asarray(flash._lut_y, dtype=self.numeric_dtype)
            if hasattr(flash, "_lut_y")
            else np.array([0.0, 1.0], dtype=self.numeric_dtype)
        )
        lut_caq = (
            np.asarray(flash._lut_caq, dtype=self.numeric_dtype)
            if hasattr(flash, "_lut_caq")
            else np.array([0.0, 1.0], dtype=self.numeric_dtype)
        )
        self.lut_y, size = readonly(lut_y, self.numeric_dtype)
        resident_bytes += size
        self.lut_caq, size = readonly(lut_caq, self.numeric_dtype)
        resident_bytes += size
        self.lut_count = int(lut_y.size)

        weighted_g = voxel_volume * density
        weighted_aq = voxel_volume * solubility
        measured_depth, spatial_depth_shape = depth_model_arrays(context, shape)
        weighted_g_per_m = np.divide(
            weighted_g,
            measured_depth,
            out=np.zeros_like(weighted_g, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        weighted_aq_per_m = np.divide(
            weighted_aq,
            measured_depth,
            out=np.zeros_like(weighted_aq, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        pore_volume_per_m = np.divide(
            voxel_volume,
            measured_depth,
            out=np.zeros_like(voxel_volume, dtype=np.float64),
            where=measured_depth > 0.0,
        )
        signal_models = calibration.signal_model.model[1]
        self.label_data: dict[int, dict[str, Any]] = {}
        active_signal_mask = np.zeros(shape, dtype=bool)

        def allocate_data(
            mask: np.ndarray,
            model: Any,
            *,
            include_mapping_signals: bool,
        ) -> dict[str, Any]:
            nonlocal resident_bytes
            pixel_count = int(np.count_nonzero(mask))
            supports = np.asarray(model.supports, dtype=self.numeric_dtype)
            values = np.asarray(model.values, dtype=self.numeric_dtype)
            if supports.shape != values.shape:
                raise RuntimeError(
                    "OpenCL signal model shape mismatch: "
                    f"{supports.shape} != {values.shape}"
                )
            data: dict[str, Any] = {
                "model": model if include_mapping_signals else None,
                "pixel_count": pixel_count,
                "support_count": int(supports.size),
                "signals": [],
                "gas_scores": [],
                "observation_confidences": [],
            }
            flat_indices = np.flatnonzero(mask.ravel()).astype(np.uint32)
            for name, host_values, dtype in (
                ("indices", flat_indices, np.uint32),
                ("supports", supports, self.numeric_dtype),
                ("values", values, self.numeric_dtype),
                ("weight_g_per_m", weighted_g_per_m[mask], self.numeric_dtype),
                ("weight_aq_per_m", weighted_aq_per_m[mask], self.numeric_dtype),
                ("pore_volume_per_m", pore_volume_per_m[mask], self.numeric_dtype),
                ("depth_shape", spatial_depth_shape[mask], self.numeric_dtype),
                (
                    "gas_constraint",
                    gas_constraint[mask],
                    np.float32,
                ),
                (
                    "hard_gas_constraint",
                    hard_gas_constraint[mask],
                    np.float32,
                ),
                (
                    "aqueous_constraint",
                    aqueous_constraint[mask],
                    np.float32,
                ),
            ):
                data[name], size = readonly(host_values, dtype)
                resident_bytes += size

            if include_mapping_signals:
                for frame in context._prepared_colors:
                    buffer, size = readonly(
                        np.asarray(frame.img, dtype=self.numeric_dtype)[mask],
                        self.numeric_dtype,
                    )
                    data["signals"].append(buffer)
                    resident_bytes += size
            else:
                dummy_signal, size = readonly(
                    np.zeros(pixel_count, dtype=self.numeric_dtype),
                    self.numeric_dtype,
                )
                resident_bytes += size
                data["signals"] = [dummy_signal] * len(context._prepared_colors)

            if self.phase_separation != "shared-signal":
                for score in source_scores:
                    buffer, size = readonly(
                        np.asarray(score, dtype=np.uint8)[mask],
                        np.uint8,
                    )
                    data["gas_scores"].append(buffer)
                    resident_bytes += size
            else:
                data["dummy_score"], size = readonly(
                    np.zeros(pixel_count, dtype=np.uint8),
                    np.uint8,
                )
                resident_bytes += size

            if self.spatial_objective_active:
                data["seal_mask"], size = readonly(
                    seal_mask[mask],
                    np.uint8,
                )
                resident_bytes += size
                for confidence in spatial_confidences:
                    buffer, size = readonly(
                        np.asarray(confidence, dtype=np.uint8)[mask],
                        np.uint8,
                    )
                    data["observation_confidences"].append(buffer)
                    resident_bytes += size

            data["output_g"] = cl_array.empty(
                self.queue,
                pixel_count,
                dtype=self.numeric_dtype,
                allocator=self.allocator,
            )
            data["output_aq"] = cl_array.empty(
                self.queue,
                pixel_count,
                dtype=self.numeric_dtype,
                allocator=self.allocator,
            )
            for name in (
                "output_aq_concentration",
                "output_aq_c_volume",
                "output_aq_c2_volume",
                "output_aq_support_volume",
            ):
                data[name] = cl_array.empty(
                    self.queue,
                    pixel_count,
                    dtype=self.numeric_dtype,
                    allocator=self.allocator,
                )
            resident_bytes += int(
                data["output_g"].nbytes
                + data["output_aq"].nbytes
                + data["output_aq_concentration"].nbytes
                + data["output_aq_c_volume"].nbytes
                + data["output_aq_c2_volume"].nbytes
                + data["output_aq_support_volume"].nbytes
            )
            if self.spatial_objective_active:
                data["output_aq_pore_volume"] = cl_array.empty(
                    self.queue,
                    pixel_count,
                    dtype=self.numeric_dtype,
                    allocator=self.allocator,
                )
                data["spatial_scratch"] = cl_array.empty(
                    self.queue,
                    pixel_count,
                    dtype=self.numeric_dtype,
                    allocator=self.allocator,
                )
                resident_bytes += int(
                    data["output_aq_pore_volume"].nbytes
                    + data["spatial_scratch"].nbytes
                )
            else:
                data["output_aq_pore_volume"] = data["output_aq_support_volume"]
                data["spatial_scratch"] = data["output_aq_support_volume"]
            return data

        for label in context.signal_labels:
            mask = labels == int(label)
            if not np.any(mask):
                continue
            active_signal_mask |= mask
            model = signal_models[int(label)]
            self.label_data[int(label)] = allocate_data(
                mask,
                model,
                include_mapping_signals=True,
            )

        if not self.label_data:
            raise RuntimeError("OpenCL evaluator found no active signal labels")
        self.restoration_spill_data = None
        restoration_spill_mask = ~active_signal_mask
        if self.restoration_spec is not None and np.any(restoration_spill_mask):
            dummy_model = next(iter(self.label_data.values()))["model"]
            self.restoration_spill_data = allocate_data(
                restoration_spill_mask,
                dummy_model,
                include_mapping_signals=False,
            )

        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        if self.restoration_spec is not None:
            self.restoration_signal = cl_array.empty(
                self.queue,
                shape,
                dtype=self.numeric_dtype,
                allocator=self.allocator,
            )
            self.restoration_scratch = cl_array.empty(
                self.queue,
                shape,
                dtype=self.numeric_dtype,
                allocator=self.allocator,
            )
            self.restoration_output = cl_array.empty(
                self.queue,
                shape,
                dtype=self.numeric_dtype,
                allocator=self.allocator,
            )
            self.restoration_mask, size = readonly(
                self.restoration_spec.mask,
                self.numeric_dtype,
            )
            resident_bytes += size
            self.restoration_mean, size = readonly(
                self.restoration_spec.mean_pore_volume,
                self.numeric_dtype,
            )
            resident_bytes += size
            self.restoration_zero, size = readonly(
                self.restoration_spec.zero_mask,
                np.uint8,
            )
            resident_bytes += size + int(
                self.restoration_signal.nbytes
                + self.restoration_scratch.nbytes
                + self.restoration_output.nbytes
            )

        self.frame_count = len(context._prepared_colors)
        loaded_rows = list(getattr(context, "_loaded", []))
        self.frame_times_h = (
            [float(row[2]) for row in loaded_rows]
            if len(loaded_rows) == self.frame_count
            else [0.0] * self.frame_count
        )
        self.hybrid_injection_end_h = float(
            getattr(context, "hybrid_injection_end_h", 0.0)
        )
        self.resident_bytes = resident_bytes
        self.queue.finish()
        logger.info(
            "[%s] OpenCL evaluator ready on %s / %s; precision=%s "
            "frames=%d labels=%s "
            "constraint=%s/%.3fm spatial=%s restoration=%s resident_mb=%.0f "
            "global_mem_mb=%.0f",
            context.run,
            platform.name,
            device.name,
            self.precision,
            self.frame_count,
            sorted(self.label_data),
            self.constraint_mode,
            self.constraint_expansion_m,
            self.spatial_objective_active,
            (
                f"signal/rev{self.restoration_spec.rev_size}"
                if self.restoration_spec is not None
                else "off"
            ),
            resident_bytes / (1024 * 1024),
            int(device.global_mem_size) / (1024 * 1024),
        )

    def evaluate(
        self,
        params: dict[str, Any],
        *,
        include_distribution: bool = False,
        include_spatial: bool = False,
    ) -> list[Any]:
        cl = self.cl
        cl_array = self.cl_array
        if include_spatial and not self.spatial_objective_active:
            raise RuntimeError(
                "OpenCL spatial diagnostics requested without observation support"
            )
        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        aqueous_blend = min(
            1.0,
            max(0.0, float(getattr(self.flash, "aqueous_blend", 0.0))),
        )
        if self.phase_separation == "residual-gas":
            score_onset = float(params.get("gas.residual_onset", 0.20))
            score_width = max(
                1e-6,
                float(params.get("gas.residual_width", 0.40)),
            )
        elif self.phase_separation == "supervised-dual":
            score_onset = float(params.get("gas.dual_onset", 0.50))
            score_width = max(
                1e-6,
                float(params.get("gas.dual_width", 0.25)),
            )
        else:
            score_onset = float(params.get("gas.clear_onset", 0.20))
            score_width = max(
                1e-6,
                float(params.get("gas.clear_width", 0.40)),
            )
        phase_mode = {
            "shared-signal": 0,
            "residual-gas": 1,
            "clearwater-direct": 3,
            "clearwater-gated": 2,
            "hybrid-causal": 4,
            "supervised-dual": 5,
        }[self.phase_separation]
        dual_max_saturation = np.clip(
            float(params.get("gas.dual_max_saturation", 0.50)),
            0.0,
            1.0,
        )
        (
            depth_edge_m,
            depth_bulge_m,
            depth_minimum_m,
            depth_maximum_m,
            _depth_active,
        ) = resolve_depth_parameters(self.context, params)

        for data in self.label_data.values():
            values = np.ascontiguousarray(
                np.asarray(data["model"].values, dtype=self.numeric_dtype)
            )
            if values.size != data["support_count"]:
                raise RuntimeError(
                    "OpenCL model value count changed after context preparation"
                )
            cl.enqueue_copy(self.queue, data["values"], values, is_blocking=False)

        reductions: list[tuple[Any, ...]] = []
        support_level = float(
            getattr(
                self.context,
                "aqueous_diffusion_support_level",
                DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
            )
        )
        trace_level = float(getattr(self.context, "spatial_trace_level", 0.03))
        core_level = float(getattr(self.context, "spatial_core_level", 0.20))
        for frame_index in range(self.frame_count):
            restored_signal = None
            if self.restoration_spec is not None:
                self.restoration_signal.fill(
                    self.scalar_dtype(0.0), queue=self.queue
                )
                for data in self.label_data.values():
                    pixel_count = data["pixel_count"]
                    label_global_size = ((pixel_count + 255) // 256 * 256,)
                    self.map_signal_kernel(
                        self.queue,
                        label_global_size,
                        (256,),
                        data["signals"][frame_index],
                        data["supports"],
                        data["values"],
                        np.int32(data["support_count"]),
                        data["indices"],
                        np.uint64(pixel_count),
                        self.restoration_signal.data,
                    )
                full_count = int(np.prod(self.shape))
                full_global_size = ((full_count + 255) // 256 * 256,)
                height, width = self.shape
                self.restoration_horizontal_kernel(
                    self.queue,
                    full_global_size,
                    (256,),
                    self.restoration_signal.data,
                    self.restoration_mask,
                    np.int32(width),
                    np.int32(height),
                    np.int32(self.restoration_spec.rev_size),
                    self.restoration_scratch.data,
                )
                self.restoration_vertical_kernel(
                    self.queue,
                    full_global_size,
                    (256,),
                    self.restoration_scratch.data,
                    self.restoration_mean,
                    self.restoration_zero,
                    np.int32(width),
                    np.int32(height),
                    np.int32(self.restoration_spec.rev_size),
                    self.restoration_output.data,
                )
                restored_signal = self.restoration_output.data
            hybrid_weight = (
                hybrid_shared_signal_weight(
                    self.frame_times_h[frame_index],
                    injection_end_h=self.hybrid_injection_end_h,
                    half_life_h=float(
                        params.get(
                            "gas.hybrid_half_life_h",
                            DEFAULT_HYBRID_HALF_LIFE_H,
                        )
                    ),
                )
                if phase_mode == 4
                else 0.0
            )
            evaluation_rows = list(self.label_data.values())
            if self.restoration_spill_data is not None:
                evaluation_rows.append(self.restoration_spill_data)
            for data in evaluation_rows:
                pixel_count = data["pixel_count"]
                global_size = ((pixel_count + 255) // 256 * 256,)
                gas_score = (
                    data["gas_scores"][frame_index]
                    if phase_mode
                    else data["dummy_score"]
                )
                self.phase_kernel(
                    self.queue,
                    global_size,
                    (256,),
                    (
                        restored_signal
                        if restored_signal is not None
                        else data["signals"][frame_index]
                    ),
                    data["indices"],
                    np.int32(1 if restored_signal is not None else 0),
                    data["weight_g_per_m"],
                    data["weight_aq_per_m"],
                    data["pore_volume_per_m"],
                    data["depth_shape"],
                    data["gas_constraint"],
                    data["hard_gas_constraint"],
                    data["aqueous_constraint"],
                    gas_score,
                    data["supports"],
                    data["values"],
                    np.int32(data["support_count"]),
                    self.lut_y,
                    self.lut_caq,
                    np.int32(self.lut_count),
                    self.scalar_dtype(aqueous_blend),
                    self.scalar_dtype(min_aq),
                    self.scalar_dtype(max_aq),
                    self.scalar_dtype(min_g),
                    self.scalar_dtype(max_g),
                    np.float32(score_onset),
                    np.float32(score_width),
                    np.float32(self.arrival_scale),
                    np.float32(self.score_scale),
                    np.float32(hybrid_weight),
                    np.float32(dual_max_saturation),
                    np.int32(phase_mode),
                    np.int32(1 if self.separate_hard_gas_constraint else 0),
                    self.scalar_dtype(depth_edge_m),
                    self.scalar_dtype(depth_bulge_m),
                    self.scalar_dtype(depth_minimum_m),
                    self.scalar_dtype(depth_maximum_m),
                    self.scalar_dtype(support_level),
                    np.int32(1 if include_distribution else 0),
                    np.int32(1 if include_spatial else 0),
                    np.uint64(pixel_count),
                    data["output_g"].data,
                    data["output_aq"].data,
                    data["output_aq_concentration"].data,
                    data["output_aq_c_volume"].data,
                    data["output_aq_c2_volume"].data,
                    data["output_aq_support_volume"].data,
                    data["output_aq_pore_volume"].data,
                )
                gas_sum = cl_array.sum(
                    data["output_g"],
                    dtype=self.numeric_dtype,
                    queue=self.queue,
                )
                aqueous_sum = cl_array.sum(
                    data["output_aq"],
                    dtype=self.numeric_dtype,
                    queue=self.queue,
                )
                aqueous_max = None
                aqueous_c_sum = None
                aqueous_c2_sum = None
                aqueous_support_sum = None
                if include_distribution:
                    aqueous_c_sum = cl_array.sum(
                        data["output_aq_c_volume"],
                        dtype=self.numeric_dtype,
                        queue=self.queue,
                    )
                    aqueous_c2_sum = cl_array.sum(
                        data["output_aq_c2_volume"],
                        dtype=self.numeric_dtype,
                        queue=self.queue,
                    )
                    aqueous_support_sum = cl_array.sum(
                        data["output_aq_support_volume"],
                        dtype=self.numeric_dtype,
                        queue=self.queue,
                    )
                    aqueous_max = cl_array.max(
                        data["output_aq_concentration"], queue=self.queue
                    )

                spatial_values: tuple[Any, ...] = (None,) * 8
                if include_spatial:
                    spatial_outputs = (
                        data["output_aq_c_volume"],
                        data["output_aq_c2_volume"],
                        data["output_aq_support_volume"],
                        data["spatial_scratch"],
                    )
                    self.spatial_kernel(
                        self.queue,
                        global_size,
                        (256,),
                        data["output_g"].data,
                        data["output_aq"].data,
                        data["output_aq_concentration"].data,
                        data["output_aq_pore_volume"].data,
                        data["observation_confidences"][frame_index],
                        data["seal_mask"],
                        self.scalar_dtype(trace_level),
                        self.scalar_dtype(core_level),
                        np.int32(0),
                        np.uint64(pixel_count),
                        *(output.data for output in spatial_outputs),
                    )
                    first_pass = tuple(
                        cl_array.sum(
                            output,
                            dtype=self.numeric_dtype,
                            queue=self.queue,
                        )
                        for output in spatial_outputs
                    )
                    self.spatial_kernel(
                        self.queue,
                        global_size,
                        (256,),
                        data["output_g"].data,
                        data["output_aq"].data,
                        data["output_aq_concentration"].data,
                        data["output_aq_pore_volume"].data,
                        data["observation_confidences"][frame_index],
                        data["seal_mask"],
                        self.scalar_dtype(trace_level),
                        self.scalar_dtype(core_level),
                        np.int32(1),
                        np.uint64(pixel_count),
                        *(output.data for output in spatial_outputs),
                    )
                    second_pass = tuple(
                        cl_array.sum(
                            output,
                            dtype=self.numeric_dtype,
                            queue=self.queue,
                        )
                        for output in spatial_outputs
                    )
                    spatial_values = (*first_pass, *second_pass)
                reductions.append(
                    (
                        frame_index,
                        gas_sum,
                        aqueous_sum,
                        aqueous_max,
                        aqueous_c_sum,
                        aqueous_c2_sum,
                        aqueous_support_sum,
                        *spatial_values,
                    )
                )

        frame_g = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_max = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_c = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_c2 = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_support = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_low_observation = np.zeros(self.frame_count, dtype=np.float64)
        frame_co2_low_observation = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_observation_missed = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_observation_weight = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_unsupported_seal = np.zeros(self.frame_count, dtype=np.float64)
        frame_co2_unsupported_seal = np.zeros(self.frame_count, dtype=np.float64)
        frame_aq_low_observation_support = np.zeros(
            self.frame_count, dtype=np.float64
        )
        frame_aq_predicted_support = np.zeros(self.frame_count, dtype=np.float64)
        for (
            frame_index,
            gas_sum,
            aqueous_sum,
            aqueous_max,
            aqueous_c_sum,
            aqueous_c2_sum,
            aqueous_support_sum,
            aqueous_low_observation_sum,
            co2_low_observation_sum,
            aqueous_observation_missed_sum,
            aqueous_observation_weight_sum,
            aqueous_unsupported_seal_sum,
            co2_unsupported_seal_sum,
            aqueous_low_observation_support_sum,
            aqueous_predicted_support_sum,
        ) in reductions:
            frame_g[frame_index] += float(gas_sum.get())
            frame_aq[frame_index] += float(aqueous_sum.get())
            if include_distribution:
                frame_aq_max[frame_index] = max(
                    frame_aq_max[frame_index], float(aqueous_max.get())
                )
                frame_aq_c[frame_index] += float(aqueous_c_sum.get())
                frame_aq_c2[frame_index] += float(aqueous_c2_sum.get())
                frame_aq_support[frame_index] += float(aqueous_support_sum.get())
            if include_spatial:
                frame_aq_low_observation[frame_index] += float(
                    aqueous_low_observation_sum.get()
                )
                frame_co2_low_observation[frame_index] += float(
                    co2_low_observation_sum.get()
                )
                frame_aq_observation_missed[frame_index] += float(
                    aqueous_observation_missed_sum.get()
                )
                frame_aq_observation_weight[frame_index] += float(
                    aqueous_observation_weight_sum.get()
                )
                frame_aq_unsupported_seal[frame_index] += float(
                    aqueous_unsupported_seal_sum.get()
                )
                frame_co2_unsupported_seal[frame_index] += float(
                    co2_unsupported_seal_sum.get()
                )
                frame_aq_low_observation_support[frame_index] += float(
                    aqueous_low_observation_support_sum.get()
                )
                frame_aq_predicted_support[frame_index] += float(
                    aqueous_predicted_support_sum.get()
                )

        if not include_distribution and not include_spatial:
            return [
                (
                    float(frame_g[index] + frame_aq[index]),
                    float(frame_g[index]),
                    float(frame_aq[index]),
                )
                for index in range(self.frame_count)
            ]
        return [
            IntegratedFrameResult(
                total_mass=float(frame_g[index] + frame_aq[index]),
                gaseous_mass=float(frame_g[index]),
                aqueous_mass=float(frame_aq[index]),
                aqueous_concentration_max=(
                    float(frame_aq_max[index]) if include_distribution else None
                ),
                aqueous_concentration_peak_equivalent=(
                    float(frame_aq_c2[index] / frame_aq_c[index])
                    if include_distribution and frame_aq_c[index] > 0.0
                    else (0.0 if include_distribution else None)
                ),
                aqueous_affected_pore_volume_m3=(
                    float(frame_aq_support[index])
                    if include_distribution
                    else None
                ),
                aqueous_effective_pore_volume_m3=(
                    float(
                        frame_aq_c[index]
                        * frame_aq_c[index]
                        / frame_aq_c2[index]
                    )
                    if include_distribution and frame_aq_c2[index] > 0.0
                    else (0.0 if include_distribution else None)
                ),
                aqueous_low_observation_mass_kg=(
                    float(frame_aq_low_observation[index])
                    if include_spatial
                    else None
                ),
                co2_low_observation_mass_kg=(
                    float(frame_co2_low_observation[index])
                    if include_spatial
                    else None
                ),
                aqueous_observation_recall_deficit=(
                    float(
                        frame_aq_observation_missed[index]
                        / frame_aq_observation_weight[index]
                    )
                    if include_spatial and frame_aq_observation_weight[index] > 0.0
                    else (0.0 if include_spatial else None)
                ),
                aqueous_unsupported_seal_mass_kg=(
                    float(frame_aq_unsupported_seal[index])
                    if include_spatial
                    else None
                ),
                co2_unsupported_seal_mass_kg=(
                    float(frame_co2_unsupported_seal[index])
                    if include_spatial
                    else None
                ),
                aqueous_low_observation_support_volume_m3=(
                    float(frame_aq_low_observation_support[index])
                    if include_spatial
                    else None
                ),
                aqueous_predicted_support_volume_m3=(
                    float(frame_aq_predicted_support[index])
                    if include_spatial
                    else None
                ),
            )
            for index in range(self.frame_count)
        ]

    def close(self) -> None:
        try:
            self.queue.finish()
        except Exception:
            pass
        self.label_data.clear()
        self.restoration_spill_data = None
        self.lut_y = None
        self.lut_caq = None
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.map_signal_kernel = None
        self.restoration_horizontal_kernel = None
        self.restoration_vertical_kernel = None
        self.phase_kernel = None
        self.spatial_kernel = None
        self.program = None
        try:
            self.allocator.stop_holding()
        except Exception:
            pass
        self.allocator = None
        self.queue = None
        self.cl_context = None
        self.context = None


class OpenCLFloat32IntegratedEvaluator(OpenCLIntegratedEvaluator):
    """Strict float32 OpenCL evaluator with an unconditional hard gas mask."""

    def __init__(self, context: Any, *, residual_score_clip: float) -> None:
        super().__init__(
            context,
            residual_score_clip=residual_score_clip,
            _precision="float32",
        )
