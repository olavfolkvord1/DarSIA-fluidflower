"""Portable execution budgets; never changes image resolution or model parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import math

GIB = 1024 ** 3
OPTICAL_EXTRACTION_RESOURCE_FORMAT = "optical-extraction-resources-v1"


def optical_extraction_resources(shape_hw, *, tasks, cpus=None, memory_gb=None,
                                 extraction_workers=None, stage_workers=None,
                                 opencv_threads=None, detected=None):
    """Plan future CPU extraction; settings never enter the feature contract.

    This is a conservative allocation estimate, not a measured throughput choice
    or an RSS limit. Each worker reserves four native working sets (96 B/pixel
    plus 512 MiB each), plus 128 B/pixel for optical feature expansion. The
    coordinator retains 4 GiB for sampled matrices and IPC. Image size is native.
    """
    if detected is None:
        from darsia.fluidflower.tools.phase_throughput_resources import available_logical_cpus
        logical = available_logical_cpus()
        available = total = available_resources()[1]
        try:
            import psutil
            memory = psutil.virtual_memory()
            available, total = int(memory.available), int(memory.total)
        except (ImportError, OSError):
            pass
    else:
        logical, available, total = map(int, detected)
    cpus = cpus if cpus is not None else os.environ.get("PHASE_CPU_LIMIT")
    memory_gb = memory_gb if memory_gb is not None else os.environ.get("PHASE_MEMORY_GB")
    requested = dict(cpus=cpus, memory_gb=memory_gb, extraction_workers=extraction_workers,
                     stage_workers=stage_workers, opencv_threads=opencv_threads)
    for key, value in requested.items():
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0
                                  or (key != "memory_gb" and int(value) != float(value))):
            raise ValueError("CPU, memory and worker budgets must be positive; counts must be integers")
    height, width = map(int, shape_hw)
    if min(height, width, int(tasks), logical, available, total) < 1:
        raise ValueError("Native dimensions, task count and available resources must be positive")
    # Keep desktop headroom even when a caller asks for the whole machine.
    cpu_ceiling = max(1, min(logical - 1, logical * 3 // 4))
    memory_ceiling = max(1, min(available * 7 // 10, total * 3 // 5))
    cpu_limit = min(int(cpus), cpu_ceiling) if cpus is not None else cpu_ceiling
    memory_bytes = min(int(float(memory_gb) * GIB), memory_ceiling) if memory_gb is not None else memory_ceiling
    pixels = height * width
    worker_bytes = 4 * (pixels * 96 + GIB // 2) + pixels * 128
    coordinator_bytes = 4 * GIB
    if memory_bytes < worker_bytes + coordinator_bytes:
        raise ValueError("Optical extraction memory budget cannot hold one native worker and coordinator")
    native_threads = min(cpu_limit, int(opencv_threads) if opencv_threads is not None else 2)
    worker_limit = min(int(tasks), cpu_limit // native_threads,
                       (memory_bytes - coordinator_bytes) // worker_bytes)
    workers = min(worker_limit, int(extraction_workers)) if extraction_workers is not None else worker_limit
    staging = min(int(tasks), cpu_limit, int(stage_workers) if stage_workers is not None else 4)
    return dict(format=OPTICAL_EXTRACTION_RESOURCE_FORMAT, cpu_limit=cpu_limit,
                memory_bytes=memory_bytes, native_shape_hw=[height, width], task_count=int(tasks),
                stage_workers=staging, extraction_workers=int(workers), opencv_threads=native_threads,
                estimated_worker_bytes=worker_bytes, coordinator_reserved_bytes=coordinator_bytes,
                estimated_working_bytes=int(workers) * worker_bytes + coordinator_bytes,
                requested_overrides={key: value for key, value in requested.items() if value is not None},
                selection="CPU and native-memory bounds; unbenchmarked conservative estimate",
                execution_only=True)


def available_resources() -> tuple[int, int]:
    """Respect affinity and leave headroom for the desktop when possible."""
    logical = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    memory = 2 * GIB
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        logical = min(logical, physical)
        memory = int(psutil.virtual_memory().available * .7)
    except (ImportError, OSError):
        pass
    return max(1, logical - 1), memory


@dataclass(frozen=True)
class ExecutionResources:
    cpu_limit: int
    memory_bytes: int
    compute_threads: int
    output_workers: int
    proposal_workers: int
    pending_outputs: int
    estimated_working_bytes: int

    def to_dict(self):
        return asdict(self)

    def apply(self):
        # Set before child processes are spawned. A helper also enforces one
        # native library thread in its initializer.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[key] = str(self.compute_threads)
        os.environ.update(PHASE_OUTPUT_WORKERS=str(self.output_workers),
                          PHASE_OUTPUT_PENDING=str(self.pending_outputs),
                          PHASE_PROPOSAL_PROCESSES=str(self.proposal_workers),
                          PHASE_PARITY_REFERENCE_THREADS=str(self.compute_threads))
        import cv2
        from threadpoolctl import threadpool_limits
        cv2.setNumThreads(self.compute_threads)
        threadpool_limits(limits=self.compute_threads)


def execution_resources(shape_hw, *, cpus=None, memory_gb=None,
                        detected=None) -> ExecutionResources:
    available_cpus, available_bytes = detected or available_resources()
    cpus = cpus if cpus is not None else os.environ.get("PHASE_CPU_LIMIT")
    memory_gb = memory_gb if memory_gb is not None else os.environ.get("PHASE_MEMORY_GB")
    limit = int(cpus) if cpus is not None else available_cpus
    memory = int(float(memory_gb) * GIB) if memory_gb is not None else available_bytes
    if limit < 1 or memory <= 0:
        raise ValueError("CPU and memory budgets must be positive")
    limit = min(limit, available_cpus if cpus is None else (getattr(os, 'process_cpu_count', os.cpu_count)() or 1))
    height, width = map(int, shape_hw)
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive")
    pixels = height * width
    # Conservative estimates, not a hard RSS cap: existing full-resolution
    # physics still owns dense arrays. Bound *additional* queued snapshots.
    working = pixels * 96 + 512 * 1024 ** 2
    snapshot = max(pixels * 24, 16 * 1024 ** 2)
    spare = max(0, memory - working)
    proposals = (min(2, max(1, limit // 6), int(spare // (4 * snapshot)))
                 if limit >= 4 and spare >= 4 * snapshot else 0)
    outputs = min(2, max(0, limit - 1 - proposals)) if spare >= 2 * snapshot else 0
    compute = max(1, limit - outputs - proposals)
    pending = max(1, min(6, spare // snapshot - 2 * proposals))
    return ExecutionResources(limit, memory, compute, outputs, proposals, int(pending), working)


def pipeline_resources(cpus, memory_gb):
    """Reserve CPU/RAM for finalization; small machines retain serial execution."""
    if cpus < 1 or memory_gb <= 0:
        raise ValueError('CPU and memory budgets must be positive')
    overlap = cpus >= 4 and memory_gb >= 12
    finish_cpus = min(2, cpus - 2) if overlap else 0
    finish_memory = min(4., memory_gb / 3) if overlap else 0.
    return dict(overlap=overlap, map_cpus=cpus-finish_cpus,
                finalizer_cpus=finish_cpus, map_memory_gb=memory_gb-finish_memory,
                finalizer_memory_gb=finish_memory)
