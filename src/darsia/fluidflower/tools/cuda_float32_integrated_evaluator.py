"""Float32 CUDA evaluator matching the frozen Metal calibration contract.

This module deliberately lives beside, rather than replacing, the historical
float64 CuPy evaluator.  Its data layout, interpolation, phase algebra,
restoration filters, diagnostics, and two-stage reductions mirror
``mps_metal_integrated_evaluator`` so that CUDA and Metal timings exercise the
same numerical workload.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import numpy as np

try:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
        DEFAULT_HYBRID_HALF_LIFE_H,
        _normalise_phase_separation,
        _phase_score_parameters,
        depth_model_arrays,
        resolve_depth_parameters,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
        DEFAULT_HYBRID_HALF_LIFE_H,
        _normalise_phase_separation,
        _phase_score_parameters,
        depth_model_arrays,
        resolve_depth_parameters,
    )

try:
    from darsia.fluidflower.tools.expert_constraint_support import constraint_weights_for_calibration
except ModuleNotFoundError:
    from darsia.fluidflower.tools.expert_constraint_support import constraint_weights_for_calibration

try:
    from darsia.fluidflower.tools.mps_metal_integrated_evaluator import (
        PHASE_MODE_CODES,
        THREADGROUP_SIZE,
        _frame_results_from_host,
        _scalar_frame,
        _strict_axis,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.mps_metal_integrated_evaluator import (
        PHASE_MODE_CODES,
        THREADGROUP_SIZE,
        _frame_results_from_host,
        _scalar_frame,
        _strict_axis,
    )

try:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec
except ModuleNotFoundError:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec

try:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight
except ModuleNotFoundError:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight


logger = logging.getLogger(__name__)

CUDA_KERNEL_NAMES = (
    "map_signal_frames",
    "volume_average_horizontal_frames",
    "volume_average_vertical_frames",
    "phase_metric_partials",
    "reduce_metric_partials",
)


CUDA_SOURCE = r"""
#include <cuda_runtime.h>

struct MetricBlock {
    float4 v0;
    float4 v1;
    float4 v2;
    float4 v3;
};

__device__ __forceinline__ float clamp_unit(const float value)
{
    return fminf(1.0f, fmaxf(0.0f, value));
}

__device__ __forceinline__ float4 add_float4(
    const float4 lhs,
    const float4 rhs)
{
    return make_float4(
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z,
        lhs.w + rhs.w
    );
}

__device__ __forceinline__ float interp_linear(
    const float query,
    const float* xp,
    const float* fp,
    const unsigned int count)
{
    if (query <= xp[0]) {
        return fp[0];
    }
    if (query >= xp[count - 1]) {
        return fp[count - 1];
    }
    unsigned int lower = 0;
    unsigned int upper = count - 1;
    while (upper - lower > 1) {
        const unsigned int middle = lower + (upper - lower) / 2;
        if (xp[middle] <= query) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const float denominator = xp[upper] - xp[lower];
    if (denominator == 0.0f) {
        return fp[upper];
    }
    const float fraction = (query - xp[lower]) / denominator;
    return fp[lower] + fraction * (fp[upper] - fp[lower]);
}

__device__ __forceinline__ int reflect_index(int index, const int length)
{
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        index = index < 0 ? -index - 1 : 2 * length - index - 1;
    }
    return index;
}

extern "C" __global__ void map_signal_frames(
    const float* signals,
    const int* indices,
    const float* supports,
    const float* values,
    const unsigned int support_count,
    const unsigned int pixel_count,
    const unsigned int full_pixel_count,
    float* mapped)
{
    const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int frame = blockIdx.y;
    if (pixel >= pixel_count) {
        return;
    }
    const unsigned long long input_index =
        (unsigned long long)frame * pixel_count + pixel;
    const unsigned long long output_index =
        (unsigned long long)frame * full_pixel_count
        + (unsigned int)indices[pixel];
    mapped[output_index] = interp_linear(
        signals[input_index], supports, values, support_count
    );
}

extern "C" __global__ void volume_average_horizontal_frames(
    const float* signal,
    const float* mask,
    const unsigned int width,
    const unsigned int height,
    const unsigned int window,
    const unsigned int full_pixel_count,
    float* scratch)
{
    const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int frame = blockIdx.y;
    if (pixel >= full_pixel_count) {
        return;
    }
    const int y = (int)(pixel / width);
    const int x = (int)(pixel - (unsigned int)y * width);
    const int left = (int)window / 2;
    const int right = (int)window - left - 1;
    float sum = 0.0f;
    for (int offset = -left; offset <= right; ++offset) {
        const int source_x = reflect_index(x + offset, (int)width);
        const unsigned int source = (unsigned int)y * width
            + (unsigned int)source_x;
        const unsigned long long frame_source =
            (unsigned long long)frame * full_pixel_count + source;
        sum += signal[frame_source] * mask[source];
    }
    scratch[(unsigned long long)frame * full_pixel_count + pixel] =
        sum / (float)window;
}

extern "C" __global__ void volume_average_vertical_frames(
    const float* scratch,
    const float* mean_pore_volume,
    const float* zero_mask,
    const unsigned int width,
    const unsigned int height,
    const unsigned int window,
    const unsigned int full_pixel_count,
    float* restored)
{
    const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int frame = blockIdx.y;
    if (pixel >= full_pixel_count) {
        return;
    }
    const unsigned long long output_index =
        (unsigned long long)frame * full_pixel_count + pixel;
    if (zero_mask[pixel] != 0.0f) {
        restored[output_index] = 0.0f;
        return;
    }
    const int y = (int)(pixel / width);
    const int x = (int)(pixel - (unsigned int)y * width);
    const int left = (int)window / 2;
    const int right = (int)window - left - 1;
    float sum = 0.0f;
    for (int offset = -left; offset <= right; ++offset) {
        const int source_y = reflect_index(y + offset, (int)height);
        const unsigned int source = (unsigned int)source_y * width
            + (unsigned int)x;
        sum += scratch[
            (unsigned long long)frame * full_pixel_count + source
        ];
    }
    restored[output_index] = (
        sum / (float)window
    ) / mean_pore_volume[pixel];
}

extern "C" __global__ void phase_metric_partials(
    const float* signals,
    const int* indices,
    const unsigned int signal_is_mapped,
    const float* static_data,
    const unsigned char* gas_scores,
    const unsigned char* observation_confidences,
    const float* supports,
    const float* values,
    const unsigned int support_count,
    const float* lut,
    const unsigned int lut_count,
    const float* parameters,
    const unsigned int pixel_count,
    const unsigned int full_pixel_count,
    const unsigned int partial_group_count,
    MetricBlock* partials)
{
    MetricBlock metric;
    metric.v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int frame = blockIdx.y;
    if (pixel < pixel_count) {
        const unsigned long long static_offset =
            (unsigned long long)pixel * 8;
        const unsigned long long local_index =
            (unsigned long long)frame * pixel_count + pixel;
        const float weight_g_per_m = static_data[static_offset];
        const float weight_aq_per_m = static_data[static_offset + 1];
        const float pore_volume_per_m = static_data[static_offset + 2];
        const float depth_shape = static_data[static_offset + 3];
        const float gas_constraint = static_data[static_offset + 4];
        const float aqueous_constraint = static_data[static_offset + 5];
        const float seal_mask = static_data[static_offset + 6];
        const float hard_gas_constraint = static_data[static_offset + 7];
        const float aqueous_blend = parameters[0];
        const float min_aq = parameters[1];
        const float max_aq = parameters[2];
        const float min_g = parameters[3];
        const float max_g = parameters[4];
        const float score_onset = parameters[5];
        const float score_width = parameters[6];
        const float arrival_scale = parameters[7];
        const float score_scale = parameters[8];
        const float dual_max_saturation = parameters[9];
        const unsigned int phase_mode = (unsigned int)parameters[10];
        const float depth_edge_m = parameters[11];
        const float depth_bulge_m = parameters[12];
        const float depth_minimum_m = parameters[13];
        const float depth_maximum_m = parameters[14];
        const float aqueous_support_level = parameters[15];
        const float trace_level = parameters[16];
        const float core_level = parameters[17];
        const unsigned int include_distribution =
            (unsigned int)parameters[18];
        const unsigned int include_spatial = (unsigned int)parameters[19];
        const float hybrid_shared_weight = parameters[20 + frame];
        const float signal = signal_is_mapped != 0
            ? signals[
                (unsigned long long)frame * full_pixel_count
                + (unsigned int)indices[pixel]
            ]
            : interp_linear(
                signals[local_index], supports, values, support_count
            );
        const float aq_denominator = max_aq == min_aq
            ? 1.0f : max_aq - min_aq;
        const float gas_denominator = max_g == min_g
            ? 1.0f : max_g - min_g;
        const float y_norm = clamp_unit(
            (signal - min_aq) / aq_denominator
        );
        float physical_concentration;
        if (y_norm <= lut[0]) {
            physical_concentration = lut[1];
        } else if (y_norm >= lut[(lut_count - 1) * 2]) {
            physical_concentration = lut[(lut_count - 1) * 2 + 1];
        } else {
            unsigned int lut_lower = 0;
            unsigned int lut_upper = lut_count - 1;
            while (lut_upper - lut_lower > 1) {
                const unsigned int middle =
                    lut_lower + (lut_upper - lut_lower) / 2;
                if (lut[middle * 2] <= y_norm) {
                    lut_lower = middle;
                } else {
                    lut_upper = middle;
                }
            }
            const float x0 = lut[lut_lower * 2];
            const float x1 = lut[lut_upper * 2];
            const float y0 = lut[lut_lower * 2 + 1];
            const float y1 = lut[lut_upper * 2 + 1];
            physical_concentration = y0
                + (y_norm - x0) * (y1 - y0) / (x1 - x0);
        }
        float concentration_aq = (
            (1.0f - aqueous_blend) * physical_concentration
            + aqueous_blend * y_norm
        ) * aqueous_constraint;

        const float flash_saturation = clamp_unit(
            (fminf(max_g, fmaxf(min_g, signal)) - min_g)
            / gas_denominator
        );
        float saturation_g;
        if (phase_mode != 0) {
            const float score = (float)gas_scores[local_index] * score_scale;
            const float score_gate = clamp_unit(
                (score - score_onset) / score_width
            );
            if (phase_mode == 4) {
                const float hybrid_gate = hybrid_shared_weight
                    + (1.0f - hybrid_shared_weight) * score;
                saturation_g = flash_saturation * hybrid_gate;
            } else if (phase_mode == 5) {
                saturation_g = score_gate * dual_max_saturation;
            } else if (phase_mode == 2) {
                saturation_g = flash_saturation * score_gate;
            } else {
                saturation_g = score_gate * clamp_unit(
                    concentration_aq / arrival_scale
                );
            }
            if (phase_mode >= 2) {
                saturation_g *= gas_constraint;
            }
        } else {
            saturation_g = flash_saturation * gas_constraint;
        }
        // This invariant is distinct from the soft expert ROI.  Residual-gas
        // bypasses the soft gas constraint but must never bypass hard seals.
        saturation_g *= hard_gas_constraint;

        const float effective_depth = fminf(
            depth_maximum_m,
            fmaxf(
                depth_minimum_m,
                depth_edge_m + depth_bulge_m * depth_shape
            )
        );
        const float gaseous_mass = weight_g_per_m
            * effective_depth * saturation_g;
        const float aqueous_fraction = fmaxf(0.0f, 1.0f - saturation_g);
        const float aqueous_mass = weight_aq_per_m
            * effective_depth * concentration_aq * aqueous_fraction;
        metric.v0.x = gaseous_mass;
        metric.v0.y = aqueous_mass;

        if (include_distribution != 0 || include_spatial != 0) {
            const float aqueous_pore_volume = pore_volume_per_m
                * effective_depth * aqueous_fraction;
            if (include_distribution != 0) {
                metric.v0.z = aqueous_pore_volume * concentration_aq;
                metric.v0.w = aqueous_pore_volume
                    * concentration_aq * concentration_aq;
                metric.v1.x = concentration_aq >= aqueous_support_level
                    ? aqueous_pore_volume : 0.0f;
                metric.v3.y = concentration_aq;
            }
            if (include_spatial != 0) {
                const float confidence = clamp_unit(
                    (float)observation_confidences[local_index] / 255.0f
                );
                const float unsupported = clamp_unit(1.0f - confidence);
                const float total_mass = gaseous_mass + aqueous_mass;
                const float predicted_support = concentration_aq >= trace_level
                    ? 1.0f : 0.0f;
                const float observation_weight = aqueous_pore_volume
                    * confidence;
                const float recall_deficit = core_level > 0.0f
                    ? clamp_unit((core_level - concentration_aq) / core_level)
                    : 0.0f;
                metric.v1.y = aqueous_mass * unsupported;
                metric.v1.z = total_mass * unsupported;
                metric.v1.w = observation_weight * recall_deficit;
                metric.v2.x = observation_weight;
                metric.v2.y = aqueous_mass * unsupported * seal_mask;
                metric.v2.z = total_mass * unsupported * seal_mask;
                metric.v2.w = aqueous_pore_volume
                    * predicted_support * unsupported;
                metric.v3.x = aqueous_pore_volume * predicted_support;
            }
        }
    }

    __shared__ MetricBlock scratch[256];
    const unsigned int lane = threadIdx.x;
    scratch[lane] = metric;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane].v0 = add_float4(
                scratch[lane].v0, scratch[lane + stride].v0
            );
            scratch[lane].v1 = add_float4(
                scratch[lane].v1, scratch[lane + stride].v1
            );
            scratch[lane].v2 = add_float4(
                scratch[lane].v2, scratch[lane + stride].v2
            );
            scratch[lane].v3.x += scratch[lane + stride].v3.x;
            scratch[lane].v3.y = fmaxf(
                scratch[lane].v3.y,
                scratch[lane + stride].v3.y
            );
        }
        __syncthreads();
    }
    if (lane == 0) {
        partials[
            (unsigned long long)frame * partial_group_count + blockIdx.x
        ] = scratch[0];
    }
}

extern "C" __global__ void reduce_metric_partials(
    const MetricBlock* partials,
    const unsigned int partial_group_count,
    MetricBlock* frame_totals)
{
    const unsigned int frame = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    MetricBlock metric;
    metric.v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    metric.v3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (
        unsigned int group = lane;
        group < partial_group_count;
        group += 256
    ) {
        const MetricBlock value = partials[
            (unsigned long long)frame * partial_group_count + group
        ];
        metric.v0 = add_float4(metric.v0, value.v0);
        metric.v1 = add_float4(metric.v1, value.v1);
        metric.v2 = add_float4(metric.v2, value.v2);
        metric.v3.x += value.v3.x;
        metric.v3.y = fmaxf(metric.v3.y, value.v3.y);
    }

    __shared__ MetricBlock scratch[256];
    scratch[lane] = metric;
    __syncthreads();
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane].v0 = add_float4(
                scratch[lane].v0, scratch[lane + stride].v0
            );
            scratch[lane].v1 = add_float4(
                scratch[lane].v1, scratch[lane + stride].v1
            );
            scratch[lane].v2 = add_float4(
                scratch[lane].v2, scratch[lane + stride].v2
            );
            scratch[lane].v3.x += scratch[lane + stride].v3.x;
            scratch[lane].v3.y = fmaxf(
                scratch[lane].v3.y,
                scratch[lane + stride].v3.y
            );
        }
        __syncthreads();
    }
    if (lane == 0) {
        MetricBlock total = frame_totals[frame];
        total.v0 = add_float4(total.v0, scratch[0].v0);
        total.v1 = add_float4(total.v1, scratch[0].v1);
        total.v2 = add_float4(total.v2, scratch[0].v2);
        total.v3.x += scratch[0].v3.x;
        total.v3.y = fmaxf(total.v3.y, scratch[0].v3.y);
        frame_totals[frame] = total;
    }
}
"""


_MODULES: dict[int, Any] = {}


def _require_cuda() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CUDA float32 evaluation requires the optional 'cuda' dependency"
        ) from exc
    if bool(getattr(cp.cuda.runtime, "is_hip", False)):
        raise RuntimeError("CUDA float32 evaluation requires an NVIDIA CUDA runtime")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError(
            "CUDA float32 evaluation requested, but no CUDA device is available"
        )
    return cp


def _compile_module(cp: Any) -> Any:
    device_id = int(cp.cuda.Device().id)
    module = _MODULES.get(device_id)
    if module is None:
        module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++11",),
            backend="nvrtc",
            name_expressions=CUDA_KERNEL_NAMES,
        )
        for name in CUDA_KERNEL_NAMES:
            module.get_function(name)
        _MODULES[device_id] = module
    return module


class CUDAFloat32IntegratedEvaluator:
    """Evaluate prepared production frames with resident float32 CUDA data."""

    def __init__(
        self,
        context: Any,
        *,
        residual_score_clip: float,
        _array_runtime: Any = None,
        _kernel_module: Any = None,
        _backend_label: str = "CUDA float32",
    ) -> None:
        if not getattr(context, "_prepared_colors", None):
            raise RuntimeError("CUDA float32 evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError(
                "CUDA float32 evaluation does not support flash restoration"
            )

        first = np.asarray(context._prepared_colors[0].img)
        shape = (int(first.shape[0]), int(first.shape[1]))
        for frame_index, frame in enumerate(context._prepared_colors):
            _scalar_frame(frame, shape, f"CUDA float32 prepared frame {frame_index}")

        self.cp = _array_runtime if _array_runtime is not None else _require_cuda()
        self.module = (
            _kernel_module
            if _kernel_module is not None
            else _compile_module(self.cp)
        )
        self.backend_label = str(_backend_label)
        self.kernels = {
            name: self.module.get_function(name) for name in CUDA_KERNEL_NAMES
        }
        self.context = context
        self.flash = flash
        self.shape = shape
        self.frame_count = len(context._prepared_colors)
        self.full_pixel_count = int(np.prod(shape))
        self.phase_separation = _normalise_phase_separation(
            getattr(context, "phase_separation", "shared-signal")
        )
        self.phase_mode = PHASE_MODE_CODES[self.phase_separation]
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
        self._closed = False
        self.resident_bytes = 0

        def upload_float(values: Any) -> Any:
            host = np.ascontiguousarray(values, dtype=np.float32)
            array = self.cp.asarray(host)
            self.resident_bytes += int(host.nbytes)
            return array

        def upload_int(values: Any) -> Any:
            host = np.ascontiguousarray(values, dtype=np.int32)
            array = self.cp.asarray(host)
            self.resident_bytes += int(host.nbytes)
            return array

        def upload_uint8(values: Any) -> Any:
            host = np.ascontiguousarray(values, dtype=np.uint8)
            array = self.cp.asarray(host)
            self.resident_bytes += int(host.nbytes)
            return array

        self.dummy_uint8 = upload_uint8(np.zeros(1, dtype=np.uint8))

        self.restoration_spec = signal_restoration_spec(calibration, shape)
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
                f"CUDA float32 static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, "
                f"density={density.shape}, solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"CUDA float32 labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        spatial_confidences = list(
            getattr(context, "_spatial_observation_confidences", []) or []
        )
        self.spatial_objective_active = bool(spatial_confidences)
        if self.spatial_objective_active and len(spatial_confidences) != self.frame_count:
            raise RuntimeError(
                f"CUDA float32 spatial confidence count mismatch for {context.run}: "
                f"{len(spatial_confidences)} != {self.frame_count}"
            )
        for frame_index, confidence in enumerate(spatial_confidences):
            if np.asarray(confidence).shape != shape:
                raise RuntimeError(
                    f"CUDA float32 spatial confidence shape mismatch for "
                    f"{context.run} frame {frame_index}: "
                    f"{np.asarray(confidence).shape} != {shape}"
                )

        seal_mask = np.zeros(shape, dtype=np.float32)
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
                    f"CUDA float32 facies/frame shape mismatch for {context.run}: "
                    f"{facies.shape} != {shape}"
                )
            seal_mask = np.isin(facies, np.asarray(seal_ids)).astype(np.float32)

        reference_frame = context._prepared_colors[0]
        constraint_weights = constraint_weights_for_calibration(
            calibration, reference_frame
        )
        self.constraint_mode = constraint_weights.mode
        self.constraint_expansion_m = constraint_weights.expansion_m
        gas_constraint = np.asarray(constraint_weights.gas, dtype=np.float32)
        aqueous_constraint = np.asarray(
            constraint_weights.aqueous, dtype=np.float32
        )
        if gas_constraint.shape != shape or aqueous_constraint.shape != shape:
            raise RuntimeError(
                "CUDA float32 expert-constraint arrays must match the prepared "
                "frame shape"
            )
        hard_gas_constraint = np.ones(shape, dtype=np.float32)
        hard_nonseal = getattr(context, "_hard_nonseal_gas_mask", None)
        if hard_nonseal is not None:
            hard_nonseal_array = np.asarray(hard_nonseal, dtype=np.float32)
            if hard_nonseal_array.shape != shape:
                raise RuntimeError(
                    f"CUDA float32 hard non-seal mask shape mismatch for "
                    f"{context.run}: {hard_nonseal_array.shape} != {shape}"
                )
            hard_gas_constraint = hard_nonseal_array
        if self.phase_separation == "hybrid-causal":
            hybrid_nonseal = np.asarray(
                getattr(context, "_hybrid_nonseal_mask", None),
                dtype=np.float32,
            )
            if hybrid_nonseal.shape != shape:
                raise RuntimeError(
                    f"CUDA float32 hybrid non-seal mask shape mismatch for "
                    f"{context.run}: {hybrid_nonseal.shape} != {shape}"
                )
            gas_constraint = gas_constraint * hybrid_nonseal

        source_scores: list[Any]
        if self.phase_separation == "shared-signal":
            source_scores = []
        else:
            source_scores = list(
                context._hybrid_gas_weights
                if self.phase_separation == "hybrid-causal"
                else context._gas_scores
            )
            if len(source_scores) != self.frame_count:
                raise RuntimeError(
                    f"CUDA float32 phase score count mismatch for {context.run}: "
                    f"{len(source_scores)} != {self.frame_count}"
                )
            for frame_index, score in enumerate(source_scores):
                if np.asarray(score).shape != shape:
                    raise RuntimeError(
                        f"CUDA float32 phase score shape mismatch for {context.run} "
                        f"frame {frame_index}: {np.asarray(score).shape} != {shape}"
                    )

        lut_y = _strict_axis(
            getattr(flash, "_lut_y", np.array([0.0, 1.0], dtype=np.float32)),
            "CUDA float32 titration LUT coordinates",
        )
        lut_caq = np.asarray(
            getattr(flash, "_lut_caq", np.array([0.0, 1.0], dtype=np.float32)),
            dtype=np.float32,
        )
        if lut_caq.shape != lut_y.shape or not np.all(np.isfinite(lut_caq)):
            raise RuntimeError(
                "CUDA float32 titration LUT coordinate/value arrays are incompatible"
            )
        self.lut = upload_float(
            np.column_stack((lut_y, np.ascontiguousarray(lut_caq)))
        )
        self.lut_count = int(lut_y.size)

        measured_depth, spatial_depth_shape = depth_model_arrays(context, shape)
        weighted_g = voxel_volume * density
        weighted_aq = voxel_volume * solubility
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

        def frame_stack(
            mask: np.ndarray,
            rows: list[Any],
            *,
            dtype: Any = np.float32,
            name: str = "frame data",
        ) -> np.ndarray:
            output = np.empty(
                (len(rows), int(np.count_nonzero(mask))), dtype=dtype
            )
            for frame_index, row in enumerate(rows):
                selected = np.asarray(row)[mask]
                if dtype == np.uint8 and selected.dtype != np.uint8:
                    numeric = np.asarray(selected, dtype=np.float32)
                    if (
                        not np.all(np.isfinite(numeric))
                        or np.any(numeric < 0.0)
                        or np.any(numeric > 255.0)
                        or not np.allclose(numeric, np.rint(numeric), atol=0.0)
                    ):
                        raise RuntimeError(
                            f"CUDA float32 {name} frame {frame_index} is not "
                            "uint8-quantized"
                        )
                    selected = np.rint(numeric).astype(np.uint8)
                output[frame_index] = selected
            return output

        prepared_frames = [
            _scalar_frame(frame, shape, f"CUDA float32 prepared frame {index}")
            for index, frame in enumerate(context._prepared_colors)
        ]
        observation_rows = (
            spatial_confidences
            if self.spatial_objective_active
            else [np.zeros(shape, dtype=np.float32)] * self.frame_count
        )
        score_rows = (
            source_scores
            if source_scores
            else [np.zeros(shape, dtype=np.float32)] * self.frame_count
        )

        def allocate_data(
            mask: np.ndarray,
            model: Any,
            *,
            include_mapping_signals: bool,
        ) -> dict[str, Any]:
            pixel_count = int(np.count_nonzero(mask))
            if pixel_count < 1:
                raise RuntimeError(
                    "CUDA float32 evaluator cannot allocate an empty label"
                )
            if model is not None:
                supports = _strict_axis(
                    model.supports, "CUDA float32 signal supports"
                )
                model_values = np.asarray(model.values, dtype=np.float32)
                if (
                    model_values.shape != supports.shape
                    or not np.all(np.isfinite(model_values))
                ):
                    raise RuntimeError(
                        "CUDA float32 signal model support/value arrays are "
                        "incompatible"
                    )
            else:
                supports = np.array([0.0, 1.0], dtype=np.float32)
                model_values = supports.copy()
            group_count = (
                pixel_count + THREADGROUP_SIZE - 1
            ) // THREADGROUP_SIZE
            signals = (
                frame_stack(mask, prepared_frames)
                if include_mapping_signals
                else np.zeros(
                    (self.frame_count, pixel_count), dtype=np.float32
                )
            )
            static_data = np.empty((pixel_count, 8), dtype=np.float32)
            for column, values_for_column in enumerate(
                (
                    weighted_g_per_m,
                    weighted_aq_per_m,
                    pore_volume_per_m,
                    spatial_depth_shape,
                    gas_constraint,
                    aqueous_constraint,
                    seal_mask,
                    hard_gas_constraint,
                )
            ):
                static_data[:, column] = np.asarray(values_for_column)[mask]
            data = {
                "model": model,
                "pixel_count": pixel_count,
                "group_count": group_count,
                "support_count": int(supports.size),
                "indices": upload_int(np.flatnonzero(mask.ravel())),
                "supports": upload_float(supports),
                "values": upload_float(model_values),
                "signals": upload_float(signals),
                "static_data": upload_float(static_data),
                "gas_scores": (
                    upload_uint8(
                        frame_stack(
                            mask,
                            score_rows,
                            dtype=np.uint8,
                            name="phase score",
                        )
                    )
                    if source_scores
                    else self.dummy_uint8
                ),
                "observation_confidences": (
                    upload_uint8(
                        frame_stack(
                            mask,
                            observation_rows,
                            dtype=np.uint8,
                            name="observation confidence",
                        )
                    )
                    if self.spatial_objective_active
                    else self.dummy_uint8
                ),
                "partials": self.cp.empty(
                    (self.frame_count, group_count, 16), dtype=self.cp.float32
                ),
            }
            self.resident_bytes += self.frame_count * group_count * 16 * 4
            return data

        signal_models = calibration.signal_model.model[1]
        self.label_data: dict[int, dict[str, Any]] = {}
        active_signal_mask = np.zeros(shape, dtype=bool)
        for label in context.signal_labels:
            mask = labels == int(label)
            if not np.any(mask):
                continue
            active_signal_mask |= mask
            try:
                model = signal_models[int(label)]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"CUDA float32 evaluator has no signal model for label {label}"
                ) from exc
            self.label_data[int(label)] = allocate_data(
                mask, model, include_mapping_signals=True
            )
        if not self.label_data:
            raise RuntimeError("CUDA float32 evaluator found no active signal labels")

        self.restoration_spill_data = None
        restoration_spill_mask = ~active_signal_mask
        if self.restoration_spec is not None and np.any(restoration_spill_mask):
            self.restoration_spill_data = allocate_data(
                restoration_spill_mask,
                None,
                include_mapping_signals=False,
            )

        self.frame_totals = self.cp.empty(
            (self.frame_count, 16), dtype=self.cp.float32
        )
        self.resident_bytes += self.frame_count * 16 * 4
        self.kernel_parameters_host = np.zeros(
            20 + self.frame_count, dtype=np.float32
        )
        self.kernel_parameters = upload_float(self.kernel_parameters_host)

        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        if self.restoration_spec is not None:
            restoration_shape = (self.frame_count, self.full_pixel_count)
            self.restoration_signal = self.cp.empty(
                restoration_shape, dtype=self.cp.float32
            )
            self.restoration_scratch = self.cp.empty_like(self.restoration_signal)
            self.restoration_output = self.restoration_signal
            self.resident_bytes += self.frame_count * self.full_pixel_count * 8
            self.restoration_mask = upload_float(self.restoration_spec.mask)
            self.restoration_mean = upload_float(
                self.restoration_spec.mean_pore_volume
            )
            self.restoration_zero = upload_float(self.restoration_spec.zero_mask)

        loaded_rows = list(getattr(context, "_loaded", []))
        self.frame_times_h = (
            [float(row[2]) for row in loaded_rows]
            if len(loaded_rows) == self.frame_count
            else [0.0] * self.frame_count
        )
        self.hybrid_injection_end_h = float(
            getattr(context, "hybrid_injection_end_h", 0.0)
        )
        self.cp.cuda.get_current_stream().synchronize()
        properties = self.cp.cuda.runtime.getDeviceProperties(
            self.cp.cuda.Device().id
        )
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")
        logger.info(
            "[%s] %s evaluator ready on %s; dtype=float32 "
            "frames=%d labels=%s phase=%s constraint=%s/%.3fm spatial=%s "
            "restoration=%s resident_mb=%.0f",
            context.run,
            self.backend_label,
            device_name,
            self.frame_count,
            sorted(self.label_data),
            self.phase_separation,
            self.constraint_mode,
            self.constraint_expansion_m,
            self.spatial_objective_active,
            (
                f"signal/rev{self.restoration_spec.rev_size}"
                if self.restoration_spec is not None
                else "off"
            ),
            self.resident_bytes / (1024 * 1024),
        )

    @staticmethod
    def _u32(value: int) -> np.uint32:
        return np.uint32(value)

    def _copy_model_values(self) -> None:
        for label, data in self.label_data.items():
            values = np.ascontiguousarray(
                np.asarray(data["model"].values, dtype=np.float32)
            )
            if values.shape != (data["support_count"],):
                raise RuntimeError(
                    f"CUDA float32 model value count changed for label {label} "
                    "after context preparation"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"CUDA float32 model values for label {label} are non-finite"
                )
            data["values"].set(values)

    def _restore_signals(self) -> Any:
        if self.restoration_spec is None:
            return None
        self.restoration_signal.fill(0.0)
        for data in self.label_data.values():
            self.kernels["map_signal_frames"](
                (data["group_count"], self.frame_count, 1),
                (THREADGROUP_SIZE, 1, 1),
                (
                    data["signals"],
                    data["indices"],
                    data["supports"],
                    data["values"],
                    self._u32(data["support_count"]),
                    self._u32(data["pixel_count"]),
                    self._u32(self.full_pixel_count),
                    self.restoration_signal,
                ),
            )
        full_groups = (
            self.full_pixel_count + THREADGROUP_SIZE - 1
        ) // THREADGROUP_SIZE
        height, width = self.shape
        self.kernels["volume_average_horizontal_frames"](
            (full_groups, self.frame_count, 1),
            (THREADGROUP_SIZE, 1, 1),
            (
                self.restoration_signal,
                self.restoration_mask,
                self._u32(width),
                self._u32(height),
                self._u32(self.restoration_spec.rev_size),
                self._u32(self.full_pixel_count),
                self.restoration_scratch,
            ),
        )
        self.kernels["volume_average_vertical_frames"](
            (full_groups, self.frame_count, 1),
            (THREADGROUP_SIZE, 1, 1),
            (
                self.restoration_scratch,
                self.restoration_mean,
                self.restoration_zero,
                self._u32(width),
                self._u32(height),
                self._u32(self.restoration_spec.rev_size),
                self._u32(self.full_pixel_count),
                self.restoration_output,
            ),
        )
        return self.restoration_output

    def evaluate(
        self,
        params: Mapping[str, Any],
        *,
        include_distribution: bool = False,
        include_spatial: bool = False,
    ) -> list[Any]:
        if self._closed:
            raise RuntimeError("CUDA float32 evaluator is closed")
        if include_spatial and not self.spatial_objective_active:
            raise RuntimeError(
                "CUDA float32 spatial diagnostics requested without observation "
                "support"
            )

        min_aq = float(self.flash.min_value_aq)
        max_aq = float(self.flash.max_value_aq)
        min_g = float(self.flash.min_value_g)
        max_g = float(self.flash.max_value_g)
        aqueous_blend = min(
            1.0, max(0.0, float(getattr(self.flash, "aqueous_blend", 0.0)))
        )
        score_onset, score_width = _phase_score_parameters(
            params, self.phase_separation
        )
        dual_max_saturation = float(
            np.clip(params.get("gas.dual_max_saturation", 0.50), 0.0, 1.0)
        )
        (
            depth_edge_m,
            depth_bulge_m,
            depth_minimum_m,
            depth_maximum_m,
            _depth_active,
        ) = resolve_depth_parameters(self.context, params)
        support_level = float(
            getattr(
                self.context,
                "aqueous_diffusion_support_level",
                DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
            )
        )
        trace_level = float(getattr(self.context, "spatial_trace_level", 0.03))
        core_level = float(getattr(self.context, "spatial_core_level", 0.20))
        hybrid_weights = [
            hybrid_shared_signal_weight(
                frame_time,
                injection_end_h=self.hybrid_injection_end_h,
                half_life_h=float(
                    params.get(
                        "gas.hybrid_half_life_h",
                        DEFAULT_HYBRID_HALF_LIFE_H,
                    )
                ),
            )
            if self.phase_mode == 4
            else 0.0
            for frame_time in self.frame_times_h
        ]
        self.kernel_parameters_host[:] = np.asarray(
            [
                aqueous_blend,
                min_aq,
                max_aq,
                min_g,
                max_g,
                score_onset,
                score_width,
                self.arrival_scale,
                self.score_scale,
                dual_max_saturation,
                self.phase_mode,
                depth_edge_m,
                depth_bulge_m,
                depth_minimum_m,
                depth_maximum_m,
                support_level,
                trace_level,
                core_level,
                1.0 if include_distribution else 0.0,
                1.0 if include_spatial else 0.0,
                *hybrid_weights,
            ],
            dtype=np.float32,
        )

        self._copy_model_values()
        self.kernel_parameters.set(self.kernel_parameters_host)
        restored_signal = self._restore_signals()
        self.frame_totals.fill(0.0)
        evaluation_rows = list(self.label_data.values())
        if self.restoration_spill_data is not None:
            evaluation_rows.append(self.restoration_spill_data)
        for data in evaluation_rows:
            self.kernels["phase_metric_partials"](
                (data["group_count"], self.frame_count, 1),
                (THREADGROUP_SIZE, 1, 1),
                (
                    restored_signal
                    if restored_signal is not None
                    else data["signals"],
                    data["indices"],
                    self._u32(1 if restored_signal is not None else 0),
                    data["static_data"],
                    data["gas_scores"],
                    data["observation_confidences"],
                    data["supports"],
                    data["values"],
                    self._u32(data["support_count"]),
                    self.lut,
                    self._u32(self.lut_count),
                    self.kernel_parameters,
                    self._u32(data["pixel_count"]),
                    self._u32(self.full_pixel_count),
                    self._u32(data["group_count"]),
                    data["partials"],
                ),
            )
            self.kernels["reduce_metric_partials"](
                (1, self.frame_count, 1),
                (THREADGROUP_SIZE, 1, 1),
                (
                    data["partials"],
                    self._u32(data["group_count"]),
                    self.frame_totals,
                ),
            )

        self.cp.cuda.get_current_stream().synchronize()
        host = self.cp.asnumpy(self.frame_totals).reshape(
            self.frame_count, 4, 4
        )
        return _frame_results_from_host(
            host,
            include_distribution=include_distribution,
            include_spatial=include_spatial,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.cp.cuda.get_current_stream().synchronize()
        except Exception:
            pass
        self.label_data.clear()
        self.restoration_spill_data = None
        self.dummy_uint8 = None
        self.lut = None
        self.kernel_parameters = None
        self.kernel_parameters_host = None
        self.frame_totals = None
        self.restoration_signal = None
        self.restoration_scratch = None
        self.restoration_output = None
        self.restoration_mask = None
        self.restoration_mean = None
        self.restoration_zero = None
        self.context = None
        self.kernels = {}
        self.module = None
        try:
            self.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
