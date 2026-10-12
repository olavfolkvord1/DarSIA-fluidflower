"""Optional GPU reduction of native frame statistics; CPU float64 reference.

This changes neither optical responses nor their binning. A nonblocking,
machine-local lease prevents parallel frame workers from filling the same GPU.
Every process verifies its first GPU reduction against the CPU calculation.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import os
import tempfile
import time

import numpy as np

_PARITY_VERIFIED = False


@lru_cache(maxsize=1)
def gpu_status():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return {"available": False, "reason": "no_cuda_device"}
        device = cp.cuda.runtime.getDeviceProperties(0)
        cp.zeros(1, dtype=cp.float64).sum().get()
        return {"available": True, "device": device["name"].decode(),
                "cupy_version": cp.__version__,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()}
    except Exception as error:
        return {"available": False, "reason": type(error).__name__,
                "detail": str(error)[:300]}


@contextmanager
def _gpu_lease():
    path = Path(tempfile.gettempdir()) / "ffac-round2-gpu-reduction-device0.lock"
    stream = path.open("a+b")
    acquired = False
    try:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        yield acquired
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def _cpu(codes, weights):
    unique, inverse = np.unique(codes, return_inverse=True)
    sums = np.bincount(inverse, weights=weights, minlength=len(unique))
    return unique, sums


def reduce_weighted_codes(codes, weights, *, backend="auto"):
    """Return sorted int64 codes, float64 capacity sums, and backend receipt.

    Only nonnegative physical capacity weights are accepted. Signed codes are
    allowed, for an optical signal whose negative bins must not be discarded.
    No pixel is downsampled; both backends aggregate the identical input rows.
    """
    global _PARITY_VERIFIED
    codes = np.asarray(codes)
    weights = np.asarray(weights, dtype=np.float64)
    if codes.ndim != 1 or codes.dtype.kind not in "iu":
        raise ValueError("Reduction codes must be one-dimensional integers")
    if weights.shape != codes.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Finite nonnegative capacity weights must match codes")
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("Unknown reduction backend")
    codes = codes.astype(np.int64, copy=False)
    started = time.perf_counter()
    audit = {"backend": "cpu-float64", "input_rows": len(codes),
             "downsampled": False, "optical_response_changed": False,
             "reference_backend": "cpu-float64"}
    use_gpu = backend != "cpu" and len(codes) > 0
    status = gpu_status() if use_gpu else {"available": False, "reason": "cpu_requested_or_empty"}
    audit["gpu_status"] = status
    if use_gpu and status["available"]:
        with _gpu_lease() as acquired:
            if acquired:
                import cupy as cp
                free, _ = cp.cuda.runtime.memGetInfo()
                estimated = max(64 * len(codes), 16 * 1024**2)
                if free > 2 * estimated:
                    try:
                        device_codes = cp.asarray(codes)
                        device_weights = cp.asarray(weights)
                        unique, inverse = cp.unique(device_codes, return_inverse=True)
                        sums = cp.bincount(inverse, weights=device_weights, minlength=len(unique))
                        result = cp.asnumpy(unique), cp.asnumpy(sums)
                        audit.update(backend="gpu-float64", gpu_lease="acquired")
                        if not _PARITY_VERIFIED:
                            reference = _cpu(codes, weights)
                            if not np.array_equal(reference[0], result[0]) or not np.allclose(reference[1], result[1], rtol=2e-12, atol=1e-12):
                                raise ArithmeticError("GPU reduction differs from CPU float64 reference")
                            audit["parity_max_absolute_difference"] = float(np.max(np.abs(reference[1] - result[1]), initial=0.))
                            audit["parity_checked_this_call"] = True
                            _PARITY_VERIFIED = True
                        else:
                            audit["parity_checked_this_call"] = False
                        audit["session_gpu_parity_verified"] = _PARITY_VERIFIED
                        audit["wall_seconds"] = time.perf_counter() - started
                        return *result, audit
                    except ArithmeticError:
                        raise
                    except Exception as error:
                        audit["gpu_fallback"] = {"reason": type(error).__name__, "detail": str(error)[:300]}
                    finally:
                        # Native CPU arrays remain owned by the caller.
                        device_codes = device_weights = unique = inverse = sums = None
                        cp.get_default_memory_pool().free_all_blocks()
                else:
                    audit["gpu_fallback"] = {"reason": "insufficient_free_device_memory", "estimated_bytes": estimated}
            else:
                audit["gpu_fallback"] = {"reason": "another_frame_worker_has_gpu_lease"}
    result = _cpu(codes, weights)
    audit["wall_seconds"] = time.perf_counter() - started
    return *result, audit
