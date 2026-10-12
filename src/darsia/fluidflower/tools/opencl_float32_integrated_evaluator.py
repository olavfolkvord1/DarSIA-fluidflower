"""Packed float32 OpenCL evaluator matching CUDA and Metal semantics."""

from __future__ import annotations

import re
import threading
from typing import Any

import numpy as np

try:
    from darsia.fluidflower.tools.cuda_float32_integrated_evaluator import (
        CUDAFloat32IntegratedEvaluator,
        CUDA_KERNEL_NAMES,
        CUDA_SOURCE,
    )
    from darsia.fluidflower.tools.opencl_integrated_evaluator import _select_device
except ModuleNotFoundError:
    from darsia.fluidflower.tools.cuda_float32_integrated_evaluator import (
        CUDAFloat32IntegratedEvaluator,
        CUDA_KERNEL_NAMES,
        CUDA_SOURCE,
    )
    from darsia.fluidflower.tools.opencl_integrated_evaluator import _select_device


def _cuda_to_opencl_source(source: str) -> str:
    """Translate the small, deliberately portable CUDA kernel subset."""

    source = source.replace("#include <cuda_runtime.h>\n", "")
    source = source.replace(
        "struct MetricBlock {\n"
        "    float4 v0;\n"
        "    float4 v1;\n"
        "    float4 v2;\n"
        "    float4 v3;\n"
        "};",
        "#define make_float4(a, b, c, d) ((float4)(a, b, c, d))\n\n"
        "typedef struct MetricBlock {\n"
        "    float4 v0;\n"
        "    float4 v1;\n"
        "    float4 v2;\n"
        "    float4 v3;\n"
        "} MetricBlock;",
    )
    source = source.replace("__device__ __forceinline__", "inline")
    source = source.replace('extern "C" __global__ void', "__kernel void")
    source = source.replace("__shared__", "__local")
    source = source.replace("__syncthreads()", "barrier(CLK_LOCAL_MEM_FENCE)")
    source = source.replace("blockIdx.x", "get_group_id(0)")
    source = source.replace("blockIdx.y", "get_group_id(1)")
    source = source.replace("blockDim.x", "get_local_size(0)")
    source = source.replace("threadIdx.x", "get_local_id(0)")
    source = source.replace("unsigned long long", "ulong")
    source = source.replace("fminf", "fmin").replace("fmaxf", "fmax")

    source = re.sub(
        r"(?<!__global )\b(const\s+)?(unsigned\s+char|float|int|MetricBlock)\s*\*",
        lambda match: (
            "__global "
            + (match.group(1) or "")
            + match.group(2)
            + "*"
        ),
        source,
    )
    return source


OPENCL_PACKED_FLOAT32_SOURCE = _cuda_to_opencl_source(CUDA_SOURCE)
_OPENCL_PROGRAM_CACHE: dict[str, tuple[Any, Any]] = {}
_OPENCL_PROGRAM_CACHE_LOCK = threading.Lock()


def _opencl_program(
    cl: Any,
    platform: Any,
    device: Any,
) -> tuple[Any, Any, bool]:
    key = "|".join(
        (
            str(getattr(platform, "name", platform)),
            str(getattr(device, "name", device)),
            str(getattr(device, "driver_version", "")),
            str(hash(OPENCL_PACKED_FLOAT32_SOURCE)),
        )
    )
    with _OPENCL_PROGRAM_CACHE_LOCK:
        cached = _OPENCL_PROGRAM_CACHE.get(key)
        if cached is not None:
            return cached[0], cached[1], True
        context = cl.Context([device])
        program = cl.Program(context, OPENCL_PACKED_FLOAT32_SOURCE).build()
        _OPENCL_PROGRAM_CACHE[key] = (context, program)
        return context, program, False


class _OpenCLStream:
    def __init__(self, queue: Any) -> None:
        self.queue = queue

    def synchronize(self) -> None:
        self.queue.finish()


class _OpenCLDevice:
    id = 0


class _OpenCLRuntimeInfo:
    is_hip = False

    def __init__(self, device: Any) -> None:
        self.device = device

    @staticmethod
    def getDeviceCount() -> int:
        return 1

    def getDeviceProperties(self, _device_id: int) -> dict[str, Any]:
        return {"name": str(self.device.name)}


class _OpenCLCudaFacade:
    """Minimal CuPy-compatible synchronization facade used by the base class."""

    def __init__(self, queue: Any, device: Any) -> None:
        self.runtime = _OpenCLRuntimeInfo(device)
        self._stream = _OpenCLStream(queue)

    @staticmethod
    def Device() -> _OpenCLDevice:
        return _OpenCLDevice()

    def get_current_stream(self) -> _OpenCLStream:
        return self._stream


class _OpenCLArrayRuntime:
    float32 = np.float32

    def __init__(self, cl: Any, cl_array: Any, queue: Any, allocator: Any, device: Any):
        self.cl = cl
        self.cl_array = cl_array
        self.queue = queue
        self.allocator = allocator
        self.cuda = _OpenCLCudaFacade(queue, device)

    def asarray(self, values: Any) -> Any:
        host = np.ascontiguousarray(values)
        return self.cl_array.to_device(
            self.queue,
            host,
            allocator=self.allocator,
        )

    def empty(self, shape: Any, *, dtype: Any) -> Any:
        return self.cl_array.empty(
            self.queue,
            shape,
            dtype=dtype,
            allocator=self.allocator,
        )

    def empty_like(self, values: Any) -> Any:
        return self.cl_array.empty(
            self.queue,
            values.shape,
            dtype=values.dtype,
            allocator=self.allocator,
        )

    @staticmethod
    def asnumpy(values: Any) -> np.ndarray:
        return np.asarray(values.get())


def _unwrap_argument(value: Any) -> Any:
    return (
        value.data
        if hasattr(value, "data")
        and hasattr(value, "get")
        and hasattr(value, "shape")
        else value
    )


class _OpenCLKernel:
    def __init__(self, kernel: Any, queue: Any) -> None:
        self.kernel = kernel
        self.queue = queue

    def __call__(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: tuple[Any, ...],
    ) -> Any:
        # All packed kernels use get_group_id(1) as the frame index. Keep the
        # launch two-dimensional even for a single-frame test.
        global_size = (int(grid[0] * block[0]), int(grid[1] * block[1]))
        local_size = (int(block[0]), int(block[1]))
        return self.kernel(
            self.queue,
            global_size,
            local_size,
            *(_unwrap_argument(value) for value in args),
        )


class _OpenCLKernelModule:
    def __init__(self, cl: Any, program: Any, queue: Any) -> None:
        self.program = program
        self.kernels = {
            name: _OpenCLKernel(cl.Kernel(self.program, name), queue)
            for name in CUDA_KERNEL_NAMES
        }

    def get_function(self, name: str) -> _OpenCLKernel:
        return self.kernels[name]


class OpenCLFloat32IntegratedEvaluator(CUDAFloat32IntegratedEvaluator):
    """Use the CUDA/Metal packed workload on an OpenCL float32 device."""

    def __init__(self, context: Any, *, residual_score_clip: float) -> None:
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array
            from pyopencl.tools import ImmediateAllocator, MemoryPool
        except ImportError as exc:
            raise RuntimeError(
                "OpenCL float32 evaluation requires the optional 'opencl' dependency"
            ) from exc

        platform, device = _select_device(cl)
        cl_context, program, program_cache_hit = _opencl_program(
            cl,
            platform,
            device,
        )
        queue = cl.CommandQueue(cl_context)
        allocator = MemoryPool(ImmediateAllocator(queue))
        runtime = _OpenCLArrayRuntime(cl, cl_array, queue, allocator, device)
        module = _OpenCLKernelModule(cl, program, queue)
        self.opencl_platform = platform
        self.opencl_device = device
        self.opencl_context = cl_context
        self.opencl_queue = queue
        self.opencl_allocator = allocator
        self.opencl_program_cache_hit = program_cache_hit
        super().__init__(
            context,
            residual_score_clip=residual_score_clip,
            _array_runtime=runtime,
            _kernel_module=module,
            _backend_label="OpenCL packed float32",
        )

    def close(self) -> None:
        super().close()
        try:
            self.opencl_queue.finish()
        except Exception:
            pass
        try:
            self.opencl_allocator.stop_holding()
        except Exception:
            pass
        self.opencl_allocator = None
        self.opencl_queue = None
        self.opencl_context = None
