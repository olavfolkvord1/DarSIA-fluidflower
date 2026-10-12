"""Float32 Metal evaluator for production auto-calibration.

The evaluator mirrors the prepared CUDA/OpenCL contract, but keeps frames and
static integration arrays resident in MPS memory and performs interpolation,
phase separation, mass integration, distribution diagnostics, and spatial
diagnostics in custom Metal kernels compiled by ``torch.mps.compile_shader``.

PyTorch is imported lazily so that importing the normal CPU calibration path
does not make it an unconditional project dependency.
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
        IntegratedFrameResult,
        _normalise_phase_separation,
        _phase_score_parameters,
        depth_model_arrays,
        resolve_depth_parameters,
    )
except ModuleNotFoundError:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        DEFAULT_AQUEOUS_DIFFUSION_SUPPORT_LEVEL,
        DEFAULT_HYBRID_HALF_LIFE_H,
        IntegratedFrameResult,
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
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec
except ModuleNotFoundError:
    from darsia.fluidflower.tools.signal_restoration import signal_restoration_spec

try:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight
except ModuleNotFoundError:
    from darsia.fluidflower.tools.temporal_gas_support import hybrid_shared_signal_weight


logger = logging.getLogger(__name__)

THREADGROUP_SIZE = 256
PHASE_MODE_CODES = {
    "shared-signal": 0,
    "residual-gas": 1,
    "clearwater-gated": 2,
    "clearwater-direct": 3,
    "hybrid-causal": 4,
    "supervised-dual": 5,
}

_LIBRARY: Any = None


METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

struct MetricBlock {
    float4 v0; // gas, aqueous, integral(c dV), integral(c^2 dV)
    float4 v1; // support dV, aq low-observation, co2 low-observation, recall numerator
    float4 v2; // recall denominator, aq unsupported seal, co2 unsupported seal, low-observation support dV
    float4 v3; // predicted support dV, max concentration, reserved, reserved
};

inline float clamp_unit(float value)
{
    return min(1.0f, max(0.0f, value));
}

inline float interp_linear(
    float query,
    device const float* xp,
    device const float* fp,
    uint count)
{
    if (query <= xp[0]) {
        return fp[0];
    }
    if (query >= xp[count - 1]) {
        return fp[count - 1];
    }
    uint lower = 0;
    uint upper = count - 1;
    while (upper - lower > 1) {
        const uint middle = lower + (upper - lower) / 2;
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

inline int reflect_index(int index, int length)
{
    if (length <= 1) {
        return 0;
    }
    while (index < 0 || index >= length) {
        index = index < 0 ? -index - 1 : 2 * length - index - 1;
    }
    return index;
}

kernel void map_signal_frames(
    device const float* signals,
    device const int* indices,
    device const float* supports,
    device const float* values,
    constant uint& support_count,
    constant uint& pixel_count,
    constant uint& full_pixel_count,
    device float* mapped,
    uint3 gid [[thread_position_in_grid]])
{
    const uint pixel = gid.x;
    const uint frame = gid.y;
    if (pixel >= pixel_count) {
        return;
    }
    const ulong input_index = (ulong)frame * pixel_count + pixel;
    const ulong output_index = (ulong)frame * full_pixel_count
        + (uint)indices[pixel];
    mapped[output_index] = interp_linear(
        signals[input_index], supports, values, support_count
    );
}

kernel void volume_average_horizontal_frames(
    device const float* signal,
    device const float* mask,
    constant uint& width,
    constant uint& height,
    constant uint& window,
    constant uint& full_pixel_count,
    device float* scratch,
    uint3 gid [[thread_position_in_grid]])
{
    const uint pixel = gid.x;
    const uint frame = gid.y;
    if (pixel >= full_pixel_count) {
        return;
    }
    const int y = (int)(pixel / width);
    const int x = (int)(pixel - (uint)y * width);
    const int left = (int)window / 2;
    const int right = (int)window - left - 1;
    float sum = 0.0f;
    for (int offset = -left; offset <= right; ++offset) {
        const int source_x = reflect_index(x + offset, (int)width);
        const uint source = (uint)y * width + (uint)source_x;
        const ulong frame_source = (ulong)frame * full_pixel_count + source;
        sum += signal[frame_source] * mask[source];
    }
    scratch[(ulong)frame * full_pixel_count + pixel] = sum / (float)window;
}

kernel void volume_average_vertical_frames(
    device const float* scratch,
    device const float* mean_pore_volume,
    device const float* zero_mask,
    constant uint& width,
    constant uint& height,
    constant uint& window,
    constant uint& full_pixel_count,
    device float* restored,
    uint3 gid [[thread_position_in_grid]])
{
    const uint pixel = gid.x;
    const uint frame = gid.y;
    if (pixel >= full_pixel_count) {
        return;
    }
    const ulong output_index = (ulong)frame * full_pixel_count + pixel;
    if (zero_mask[pixel] != 0.0f) {
        restored[output_index] = 0.0f;
        return;
    }
    const int y = (int)(pixel / width);
    const int x = (int)(pixel - (uint)y * width);
    const int left = (int)window / 2;
    const int right = (int)window - left - 1;
    float sum = 0.0f;
    for (int offset = -left; offset <= right; ++offset) {
        const int source_y = reflect_index(y + offset, (int)height);
        const uint source = (uint)source_y * width + (uint)x;
        sum += scratch[(ulong)frame * full_pixel_count + source];
    }
    restored[output_index] = (
        sum / (float)window
    ) / mean_pore_volume[pixel];
}

kernel void phase_metric_partials(
    device const float* signals,
    device const int* indices,
    constant uint& signal_is_mapped,
    device const float* static_data,
    device const uchar* gas_scores,
    device const uchar* observation_confidences,
    device const float* supports,
    device const float* values,
    constant uint& support_count,
    device const float* lut,
    constant uint& lut_count,
    device const float* parameters,
    constant uint& pixel_count,
    constant uint& full_pixel_count,
    constant uint& partial_group_count,
    device MetricBlock* partials,
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]])
{
    MetricBlock metric;
    metric.v0 = float4(0.0f);
    metric.v1 = float4(0.0f);
    metric.v2 = float4(0.0f);
    metric.v3 = float4(0.0f);

    const uint pixel = gid.x;
    const uint frame = gid.y;
    if (pixel < pixel_count) {
        const ulong static_offset = (ulong)pixel * 8;
        const ulong local_index = (ulong)frame * pixel_count + pixel;
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
        const uint phase_mode = (uint)parameters[10];
        const float depth_edge_m = parameters[11];
        const float depth_bulge_m = parameters[12];
        const float depth_minimum_m = parameters[13];
        const float depth_maximum_m = parameters[14];
        const float aqueous_support_level = parameters[15];
        const float trace_level = parameters[16];
        const float core_level = parameters[17];
        const uint include_distribution = (uint)parameters[18];
        const uint include_spatial = (uint)parameters[19];
        const float hybrid_shared_weight = parameters[20 + frame];
        const float signal = signal_is_mapped != 0
            ? signals[(ulong)frame * full_pixel_count + (uint)indices[pixel]]
            : interp_linear(
                signals[local_index], supports, values, support_count
            );
        const float aq_denominator = max_aq == min_aq
            ? 1.0f : max_aq - min_aq;
        const float gas_denominator = max_g == min_g
            ? 1.0f : max_g - min_g;
        const float y_norm = clamp_unit((signal - min_aq) / aq_denominator);
        float physical_concentration;
        if (y_norm <= lut[0]) {
            physical_concentration = lut[1];
        } else if (y_norm >= lut[(lut_count - 1) * 2]) {
            physical_concentration = lut[(lut_count - 1) * 2 + 1];
        } else {
            uint lut_lower = 0;
            uint lut_upper = lut_count - 1;
            while (lut_upper - lut_lower > 1) {
                const uint middle = lut_lower + (lut_upper - lut_lower) / 2;
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
            (min(max_g, max(min_g, signal)) - min_g) / gas_denominator
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
        // The hard seal mask is an invariant in every phase mode. Residual-gas
        // intentionally bypasses the soft expert ROI, but never this mask.
        saturation_g *= hard_gas_constraint;

        const float effective_depth = min(
            depth_maximum_m,
            max(
                depth_minimum_m,
                depth_edge_m + depth_bulge_m * depth_shape
            )
        );
        const float gaseous_mass = weight_g_per_m
            * effective_depth * saturation_g;
        const float aqueous_fraction = max(0.0f, 1.0f - saturation_g);
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

    threadgroup MetricBlock scratch[256];
    scratch[lane] = metric;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane].v0 += scratch[lane + stride].v0;
            scratch[lane].v1 += scratch[lane + stride].v1;
            scratch[lane].v2 += scratch[lane + stride].v2;
            scratch[lane].v3.x += scratch[lane + stride].v3.x;
            scratch[lane].v3.y = max(
                scratch[lane].v3.y,
                scratch[lane + stride].v3.y
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partials[(ulong)frame * partial_group_count + group_id.x] = scratch[0];
    }
}

kernel void reduce_metric_partials(
    device const MetricBlock* partials,
    constant uint& partial_group_count,
    device MetricBlock* frame_totals,
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    const uint frame = gid.y;
    MetricBlock metric;
    metric.v0 = float4(0.0f);
    metric.v1 = float4(0.0f);
    metric.v2 = float4(0.0f);
    metric.v3 = float4(0.0f);
    for (uint group = lane; group < partial_group_count; group += 256) {
        const MetricBlock value = partials[
            (ulong)frame * partial_group_count + group
        ];
        metric.v0 += value.v0;
        metric.v1 += value.v1;
        metric.v2 += value.v2;
        metric.v3.x += value.v3.x;
        metric.v3.y = max(metric.v3.y, value.v3.y);
    }

    threadgroup MetricBlock scratch[256];
    scratch[lane] = metric;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[lane].v0 += scratch[lane + stride].v0;
            scratch[lane].v1 += scratch[lane + stride].v1;
            scratch[lane].v2 += scratch[lane + stride].v2;
            scratch[lane].v3.x += scratch[lane + stride].v3.x;
            scratch[lane].v3.y = max(
                scratch[lane].v3.y,
                scratch[lane + stride].v3.y
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        frame_totals[frame].v0 += scratch[0].v0;
        frame_totals[frame].v1 += scratch[0].v1;
        frame_totals[frame].v2 += scratch[0].v2;
        frame_totals[frame].v3.x += scratch[0].v3.x;
        frame_totals[frame].v3.y = max(
            frame_totals[frame].v3.y,
            scratch[0].v3.y
        );
    }
}
"""


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_mps() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Metal evaluation requires PyTorch with MPS support"
        ) from exc
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None or not backend.is_available():
        raise RuntimeError("Metal evaluation requested, but MPS is unavailable")
    if not callable(getattr(getattr(torch, "mps", None), "compile_shader", None)):
        raise RuntimeError(
            "Metal evaluation requires a PyTorch build exposing "
            "torch.mps.compile_shader"
        )
    if _truthy_environment("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError(
            "Metal evaluation requires PYTORCH_ENABLE_MPS_FALLBACK=0; "
            "CPU fallback is not permitted for calibrated timings"
        )
    if _truthy_environment("PYTORCH_MPS_FAST_MATH"):
        raise RuntimeError(
            "Metal evaluation requires PYTORCH_MPS_FAST_MATH=0 so that "
            "float32 parity is not changed by approximate Metal math"
        )
    return torch


def _compile_library(torch: Any) -> Any:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = torch.mps.compile_shader(METAL_SOURCE)
    return _LIBRARY


def _strict_axis(values: Any, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float32)
    if axis.ndim != 1 or axis.size < 2:
        raise RuntimeError(f"{name} must contain at least two coordinates")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise RuntimeError(f"{name} must be finite and strictly increasing")
    return np.ascontiguousarray(axis)


def _scalar_frame(values: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(getattr(values, "img", values))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise RuntimeError(f"{name} shape {array.shape} != {shape}")
    return np.asarray(array)


def _frame_results_from_host(
    host: np.ndarray,
    *,
    include_distribution: bool,
    include_spatial: bool,
) -> list[Any]:
    """Convert the fused kernel's 4xfloat4 layout to the shared result type."""

    values = np.asarray(host, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise RuntimeError(
            f"Metal result layout must be (frames, 4, 4), got {values.shape}"
        )
    if not include_distribution and not include_spatial:
        return [
            (
                float(row[0, 0] + row[0, 1]),
                float(row[0, 0]),
                float(row[0, 1]),
            )
            for row in values
        ]

    results: list[Any] = []
    for row in values:
        aqueous_c_volume = float(row[0, 2])
        aqueous_c2_volume = float(row[0, 3])
        observation_weight = float(row[2, 0])
        results.append(
            IntegratedFrameResult(
                total_mass=float(row[0, 0] + row[0, 1]),
                gaseous_mass=float(row[0, 0]),
                aqueous_mass=float(row[0, 1]),
                aqueous_concentration_max=(
                    float(row[3, 1]) if include_distribution else None
                ),
                aqueous_concentration_peak_equivalent=(
                    aqueous_c2_volume / aqueous_c_volume
                    if include_distribution and aqueous_c_volume > 0.0
                    else (0.0 if include_distribution else None)
                ),
                aqueous_affected_pore_volume_m3=(
                    float(row[1, 0]) if include_distribution else None
                ),
                aqueous_effective_pore_volume_m3=(
                    aqueous_c_volume * aqueous_c_volume / aqueous_c2_volume
                    if include_distribution and aqueous_c2_volume > 0.0
                    else (0.0 if include_distribution else None)
                ),
                aqueous_low_observation_mass_kg=(
                    float(row[1, 1]) if include_spatial else None
                ),
                co2_low_observation_mass_kg=(
                    float(row[1, 2]) if include_spatial else None
                ),
                aqueous_observation_recall_deficit=(
                    float(row[1, 3]) / observation_weight
                    if include_spatial and observation_weight > 0.0
                    else (0.0 if include_spatial else None)
                ),
                aqueous_unsupported_seal_mass_kg=(
                    float(row[2, 1]) if include_spatial else None
                ),
                co2_unsupported_seal_mass_kg=(
                    float(row[2, 2]) if include_spatial else None
                ),
                aqueous_low_observation_support_volume_m3=(
                    float(row[2, 3]) if include_spatial else None
                ),
                aqueous_predicted_support_volume_m3=(
                    float(row[3, 0]) if include_spatial else None
                ),
            )
        )
    return results


class MPSMetalIntegratedEvaluator:
    """Evaluate prepared production frames with resident float32 Metal data."""

    def __init__(self, context: Any, *, residual_score_clip: float) -> None:
        if not getattr(context, "_prepared_colors", None):
            raise RuntimeError("Metal evaluation requires prepared color frames")

        calibration = context.calibration
        flash = calibration.flash
        if getattr(flash, "restoration", None) is not None:
            raise RuntimeError("Metal evaluation does not support flash restoration")

        first = np.asarray(context._prepared_colors[0].img)
        shape = (int(first.shape[0]), int(first.shape[1]))
        for frame_index, frame in enumerate(context._prepared_colors):
            _scalar_frame(frame, shape, f"Metal prepared frame {frame_index}")

        self.torch = _require_mps()
        self.library = _compile_library(self.torch)
        self.device = self.torch.device("mps")
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
            if not host.flags.writeable:
                host = host.copy()
            tensor = self.torch.from_numpy(host).to(device=self.device)
            self.resident_bytes += int(host.nbytes)
            return tensor

        def upload_int(values: Any) -> Any:
            host = np.ascontiguousarray(values, dtype=np.int32)
            if not host.flags.writeable:
                host = host.copy()
            tensor = self.torch.from_numpy(host).to(device=self.device)
            self.resident_bytes += int(host.nbytes)
            return tensor

        def upload_uint8(values: Any) -> Any:
            host = np.ascontiguousarray(values, dtype=np.uint8)
            if not host.flags.writeable:
                host = host.copy()
            tensor = self.torch.from_numpy(host).to(device=self.device)
            self.resident_bytes += int(host.nbytes)
            return tensor

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
                f"Metal static-array shape mismatch for {context.run}: "
                f"shape={shape}, voxel={voxel_volume.shape}, "
                f"density={density.shape}, solubility={solubility.shape}"
            ) from exc

        labels = np.asarray(calibration.labels.img)
        if labels.shape != shape:
            raise RuntimeError(
                f"Metal labels/frame shape mismatch for {context.run}: "
                f"{labels.shape} != {shape}"
            )

        spatial_confidences = list(
            getattr(context, "_spatial_observation_confidences", []) or []
        )
        self.spatial_objective_active = bool(spatial_confidences)
        if self.spatial_objective_active and len(spatial_confidences) != self.frame_count:
            raise RuntimeError(
                f"Metal spatial confidence count mismatch for {context.run}: "
                f"{len(spatial_confidences)} != {self.frame_count}"
            )
        for frame_index, confidence in enumerate(spatial_confidences):
            if np.asarray(confidence).shape != shape:
                raise RuntimeError(
                    f"Metal spatial confidence shape mismatch for {context.run} "
                    f"frame {frame_index}: {np.asarray(confidence).shape} != {shape}"
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
                    f"Metal facies/frame shape mismatch for {context.run}: "
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
                "Metal expert-constraint arrays must match the prepared frame shape"
            )
        hard_gas_constraint = np.ones(shape, dtype=np.float32)
        hard_nonseal = getattr(context, "_hard_nonseal_gas_mask", None)
        if hard_nonseal is not None:
            hard_nonseal_array = np.asarray(hard_nonseal, dtype=np.float32)
            if hard_nonseal_array.shape != shape:
                raise RuntimeError(
                    f"Metal hard non-seal mask shape mismatch for {context.run}: "
                    f"{hard_nonseal_array.shape} != {shape}"
                )
            hard_gas_constraint = hard_nonseal_array
        if self.phase_separation == "hybrid-causal":
            hybrid_nonseal = np.asarray(
                getattr(context, "_hybrid_nonseal_mask", None),
                dtype=np.float32,
            )
            if hybrid_nonseal.shape != shape:
                raise RuntimeError(
                    f"Metal hybrid non-seal mask shape mismatch for {context.run}: "
                    f"{hybrid_nonseal.shape} != {shape}"
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
                    f"Metal phase score count mismatch for {context.run}: "
                    f"{len(source_scores)} != {self.frame_count}"
                )
            for frame_index, score in enumerate(source_scores):
                if np.asarray(score).shape != shape:
                    raise RuntimeError(
                        f"Metal phase score shape mismatch for {context.run} "
                        f"frame {frame_index}: {np.asarray(score).shape} != {shape}"
                    )

        lut_y = _strict_axis(
            getattr(flash, "_lut_y", np.array([0.0, 1.0], dtype=np.float32)),
            "Metal titration LUT coordinates",
        )
        lut_caq = np.asarray(
            getattr(flash, "_lut_caq", np.array([0.0, 1.0], dtype=np.float32)),
            dtype=np.float32,
        )
        if lut_caq.shape != lut_y.shape or not np.all(np.isfinite(lut_caq)):
            raise RuntimeError(
                "Metal titration LUT coordinate/value arrays are incompatible"
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
                            f"Metal {name} frame {frame_index} is not uint8-quantized"
                        )
                    selected = np.rint(numeric).astype(np.uint8)
                output[frame_index] = selected
            return output

        prepared_frames = [
            _scalar_frame(frame, shape, f"Metal prepared frame {index}")
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
                raise RuntimeError("Metal evaluator cannot allocate an empty label")
            if model is not None:
                supports = _strict_axis(model.supports, "Metal signal supports")
                values = np.asarray(model.values, dtype=np.float32)
                if values.shape != supports.shape or not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        "Metal signal model support/value arrays are incompatible"
                    )
            else:
                supports = np.array([0.0, 1.0], dtype=np.float32)
                values = supports.copy()
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
                "values": upload_float(values),
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
                "partials": self.torch.empty(
                    (self.frame_count, group_count, 16),
                    dtype=self.torch.float32,
                    device=self.device,
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
                    f"Metal evaluator has no signal model for label {label}"
                ) from exc
            self.label_data[int(label)] = allocate_data(
                mask, model, include_mapping_signals=True
            )
        if not self.label_data:
            raise RuntimeError("Metal evaluator found no active signal labels")

        self.restoration_spill_data = None
        restoration_spill_mask = ~active_signal_mask
        if self.restoration_spec is not None and np.any(restoration_spill_mask):
            self.restoration_spill_data = allocate_data(
                restoration_spill_mask,
                None,
                include_mapping_signals=False,
            )

        self.frame_totals = self.torch.empty(
            (self.frame_count, 16),
            dtype=self.torch.float32,
            device=self.device,
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
            self.restoration_signal = self.torch.empty(
                restoration_shape, dtype=self.torch.float32, device=self.device
            )
            self.restoration_scratch = self.torch.empty_like(
                self.restoration_signal
            )
            # Vertical filtering may overwrite the mapped input once the
            # horizontal pass has completed, saving one full frame stack.
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
        self.torch.mps.synchronize()
        logger.info(
            "[%s] Metal evaluator ready; dtype=float32 frames=%d labels=%s "
            "phase=%s constraint=%s/%.3fm spatial=%s restoration=%s "
            "resident_mb=%.0f",
            context.run,
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

    def _copy_model_values(self) -> None:
        for label, data in self.label_data.items():
            values = np.ascontiguousarray(
                np.asarray(data["model"].values, dtype=np.float32)
            )
            if values.shape != (data["support_count"],):
                raise RuntimeError(
                    f"Metal model value count changed for label {label} after "
                    "context preparation"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Metal model values for label {label} are non-finite"
                )
            data["values"].copy_(self.torch.from_numpy(values))

    def _restore_signals(self) -> Any:
        if self.restoration_spec is None:
            return None
        self.restoration_signal.zero_()
        for data in self.label_data.values():
            self.library.map_signal_frames(
                data["signals"],
                data["indices"],
                data["supports"],
                data["values"],
                data["support_count"],
                data["pixel_count"],
                self.full_pixel_count,
                self.restoration_signal,
                threads=[
                    data["group_count"] * THREADGROUP_SIZE,
                    self.frame_count,
                    1,
                ],
                group_size=[THREADGROUP_SIZE, 1, 1],
            )
        full_groups = (
            self.full_pixel_count + THREADGROUP_SIZE - 1
        ) // THREADGROUP_SIZE
        height, width = self.shape
        self.library.volume_average_horizontal_frames(
            self.restoration_signal,
            self.restoration_mask,
            width,
            height,
            self.restoration_spec.rev_size,
            self.full_pixel_count,
            self.restoration_scratch,
            threads=[full_groups * THREADGROUP_SIZE, self.frame_count, 1],
            group_size=[THREADGROUP_SIZE, 1, 1],
        )
        self.library.volume_average_vertical_frames(
            self.restoration_scratch,
            self.restoration_mean,
            self.restoration_zero,
            width,
            height,
            self.restoration_spec.rev_size,
            self.full_pixel_count,
            self.restoration_output,
            threads=[full_groups * THREADGROUP_SIZE, self.frame_count, 1],
            group_size=[THREADGROUP_SIZE, 1, 1],
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
            raise RuntimeError("Metal evaluator is closed")
        if include_spatial and not self.spatial_objective_active:
            raise RuntimeError(
                "Metal spatial diagnostics requested without observation support"
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

        with self.torch.inference_mode():
            self._copy_model_values()
            self.kernel_parameters.copy_(
                self.torch.from_numpy(self.kernel_parameters_host)
            )
            restored_signal = self._restore_signals()
            self.frame_totals.zero_()
            evaluation_rows = list(self.label_data.values())
            if self.restoration_spill_data is not None:
                evaluation_rows.append(self.restoration_spill_data)
            for data in evaluation_rows:
                self.library.phase_metric_partials(
                    restored_signal if restored_signal is not None else data["signals"],
                    data["indices"],
                    1 if restored_signal is not None else 0,
                    data["static_data"],
                    data["gas_scores"],
                    data["observation_confidences"],
                    data["supports"],
                    data["values"],
                    data["support_count"],
                    self.lut,
                    self.lut_count,
                    self.kernel_parameters,
                    data["pixel_count"],
                    self.full_pixel_count,
                    data["group_count"],
                    data["partials"],
                    threads=[
                        data["group_count"] * THREADGROUP_SIZE,
                        self.frame_count,
                        1,
                    ],
                    group_size=[THREADGROUP_SIZE, 1, 1],
                )
                self.library.reduce_metric_partials(
                    data["partials"],
                    data["group_count"],
                    self.frame_totals,
                    threads=[THREADGROUP_SIZE, self.frame_count, 1],
                    group_size=[THREADGROUP_SIZE, 1, 1],
                )

            self.torch.mps.synchronize()
            host = self.frame_totals.detach().cpu().numpy().reshape(
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
            self.torch.mps.synchronize()
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
        self.library = None
        try:
            self.torch.mps.empty_cache()
        except Exception:
            pass
