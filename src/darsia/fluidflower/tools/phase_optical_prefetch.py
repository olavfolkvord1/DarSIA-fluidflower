"""Bounded optical lookahead with one spawned CUDA owner.

The caller keeps chronological physics. While it finalizes frame N, this
adapter prepares and predicts N + 1. Only paths and small receipts cross the
process boundary; temporary RGB and two-head arrays live on local storage.
The existing CheckedCuda implementation supplies all numerical calculations.
Duplicate CPU parity checks require explicit audit mode. CPU-only execution
does not use this adapter.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import inspect
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np


VERSION = "bounded-optical-lookahead-v3"
_predictor = None
# Both complete class definitions were audited: the next frame only reads
# previous['rgb'], previous['gray'], previous['texture']; reference prediction
# transactionally saves/restores that dictionary. Unknown code stays untrimmed.
_TRIM_SOURCE_SHA256 = "e76692b3b49115be1898565b73bcd5a5fbd9806b02940b3e6286abec58903866"
_trim_source_verified = None


def _digest(array):
    if array is None:
        return None
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class BoundedAsyncOpticalCache:
    """Move exact cache writes off CUDA's critical path without changing cache data.

    Only the spawned CUDA owner installs this adapter. Input heads are copied
    before return so the writer owns immutable snapshots. At most two complete
    two-head snapshots are retained. Reads of an in-flight key, invalidation,
    and shutdown wait for the same underlying OpticalCache.store transaction.
    """

    def __init__(self, cache, *, max_pending=2):
        if max_pending not in (1, 2):
            raise ValueError("Optical cache write queue must hold one or two frames")
        self.cache = cache
        self.stats = cache.stats
        self.max_pending = max_pending
        self._pending = deque()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optical-cache-write")
        self._closed = False
        self.timings = dict(mode="bounded-async-exact-cache-store-v1", queue_limit=max_pending,
                            tasks_submitted=0, tasks_completed=0, max_pending=0,
                            enqueue_s=0., writer_busy_s=0., capacity_wait_s=0.,
                            load_wait_s=0., close_drain_s=0.)

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def _reap_completed(self):
        while self._pending and self._pending[0][1].done():
            _, future = self._pending.popleft()
            future.result()

    def _write(self, key, snapshot, parity):
        started = time.perf_counter()
        try:
            self.cache.store(key, snapshot, parity)
        finally:
            self.timings["writer_busy_s"] += time.perf_counter() - started
            self.timings["tasks_completed"] += 1

    def _drain(self):
        first_error = None
        while self._pending:
            _, future = self._pending.popleft()
            try:
                future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def key(self, current, previous):
        self._reap_completed()
        return self.cache.key(current, previous)

    def load(self, key):
        self._reap_completed()
        for pending_key, future in self._pending:
            if pending_key == key:
                started = time.perf_counter()
                future.result()
                self.timings["load_wait_s"] += time.perf_counter() - started
                break
        self._reap_completed()
        return self.cache.load(key)

    def store(self, key, heads, parity=None):
        if self._closed:
            raise RuntimeError("Optical cache write queue is closed")
        self._reap_completed()
        if len(self._pending) == self.max_pending:
            _, oldest = self._pending.popleft()
            started = time.perf_counter()
            oldest.result()
            self.timings["capacity_wait_s"] += time.perf_counter() - started
        # The prediction is returned to the phase layer immediately. The
        # writer gets private CPU arrays, so later frames cannot mutate them.
        started = time.perf_counter()
        snapshot = {name: np.array(heads[name], copy=True)
                    for name in ("co2_presence_probability",
                                 "gas_probability_conditional_on_co2")}
        self._pending.append((key, self._pool.submit(self._write, key, snapshot, parity)))
        self.timings["tasks_submitted"] += 1
        self.timings["max_pending"] = max(self.timings["max_pending"], len(self._pending))
        self.timings["enqueue_s"] += time.perf_counter() - started

    def invalidate(self, key):
        # A failed reference comparison must never race a pending write that
        # could recreate the invalid entry after its removal.
        pending_error = None
        try:
            self._drain()
        except BaseException as error:
            pending_error = error
        try:
            result = self.cache.invalidate(key)
        except BaseException as error:
            if pending_error is not None:
                error.add_note(f"An earlier optical cache write also failed: {pending_error}")
            raise
        if pending_error is not None:
            raise pending_error
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        started = time.perf_counter()
        try:
            self._drain()
        finally:
            self.timings["close_drain_s"] += time.perf_counter() - started
            self._pool.shutdown(wait=True)


def _initialize(model, baseline_path, parity_threads, factory, optical_cache_root=None,
                optical_cache_max_bytes=None, verify_numerics=False):
    global _predictor
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                      PHASE_PARITY_REFERENCE_THREADS=str(parity_threads))
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    threadpool_limits(limits=1)
    production_cuda = factory is None
    if production_cuda:
        from darsia.fluidflower.tools.materialize_phase_iteration import CheckedCuda
        factory = CheckedCuda
        _predictor = factory(model, np.load(baseline_path, allow_pickle=False), {"runs": []},
            optical_cache_root=optical_cache_root, optical_cache_max_bytes=optical_cache_max_bytes,
            verify_numerics=verify_numerics)
    else:
        options = {"verify_numerics": True} if verify_numerics else {}
        _predictor = factory(model, np.load(baseline_path, allow_pickle=False), {"runs": []}, **options)
    if production_cuda and getattr(_predictor, "cache", None) is not None:
        # CheckedCuda and OpticalCache source remain untouched, preserving the
        # existing content keys, archive format, hashes, and atomic rename.
        _predictor.cache = BoundedAsyncOpticalCache(_predictor.cache)
        _predictor.run_report["async_optical_cache"] = _predictor.cache.timings


def gpu_memory_snapshot(predictor):
    """Current CuPy pool/device allocation after inference, never a peak value.

    Pool figures belong to this process; device figures include other CUDA
    users. Reserved pool memory includes reusable blocks and is not equivalent
    to currently active arrays. A CPU/fake predictor has no GPU sample.
    """
    gpu = (predictor._gpu if "_gpu" in getattr(predictor, "__dict__", {}) else getattr(predictor, "gpu", None))
    cp = getattr(gpu, "cp", None)
    if cp is None:
        return None
    try:
        pool = cp.get_default_memory_pool()
        active, reserved = int(pool.used_bytes()), int(pool.total_bytes())
        free, total = cp.cuda.runtime.memGetInfo()
        return dict(sample_point="after-inference", is_peak=False,
            cupy_pool_active_bytes=active, cupy_pool_reserved_bytes=reserved,
            cupy_pool_reusable_bytes=reserved-active,
            device_free_bytes=int(free), device_total_bytes=int(total),
            device_used_bytes=int(total-free), block_rows=getattr(gpu, "block_rows", None),
            weights_dtype=str(gpu.weights.dtype) if hasattr(gpu, "weights") else None)
    except Exception as error:
        # Instrumentation must not turn a scientifically successful inference
        # into a failure or suppress an error from inference itself.
        return dict(sample_point="after-inference", is_peak=False,
                    unavailable=f"{type(error).__name__}: {error}")


def trim_previous_gpu_cache(predictor):
    """Drop only audited, unused previous-frame planes; retain exact arrays.

    This releases active allocations to CuPy's reusable pool. It neither
    changes the feature calculation nor flushes the pool back to the driver.
    """
    global _trim_source_verified
    if os.environ.get("PHASE_OPTICAL_TRIM_PREVIOUS", "1") != "1":
        return None
    # The fake/CPU paths do not import or initialize CUDA for this helper.
    gpu = (predictor._gpu if "_gpu" in getattr(predictor, "__dict__", {}) else getattr(predictor, "gpu", None))
    if gpu is None:
        return None
    from darsia.fluidflower.tools.materialize_phase_iteration import CheckedCuda
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor
    if type(predictor) is not CheckedCuda or type(gpu) is not CudaFeaturePredictor:
        return None
    if _trim_source_verified is None:
        try:
            source = "\n".join(inspect.getsource(kind).replace("\r\n", "\n")
                               for kind in (CudaFeaturePredictor, CheckedCuda))
            _trim_source_verified = hashlib.sha256(source.encode()).hexdigest() == _TRIM_SOURCE_SHA256
        except (OSError, TypeError):
            _trim_source_verified = False
    if not _trim_source_verified:
        return None
    previous = gpu.previous
    keep = {"rgb", "gray", "texture"}
    if not isinstance(previous, dict) or previous is gpu.baseline or not keep <= previous.keys():
        return None
    removed = sorted(previous.keys()-keep)
    if not removed:
        return None
    removed_bytes = sum(int(previous[name].nbytes) for name in removed)
    gpu.previous = {name: previous[name] for name in keep}
    return dict(method="audited-previous-frame-fields-v1", source_sha256=_TRIM_SOURCE_SHA256,
        removed_fields=removed, removed_array_bytes=removed_bytes,
        retained_fields=sorted(keep), driver_pool_flushed=False)


def _compute(current_path, previous_path, output_path, reference=False):
    started = time.perf_counter()
    started_unix = time.time()
    current = np.load(current_path, allow_pickle=False)
    previous = None if previous_path is None else np.load(previous_path, allow_pickle=False)
    loaded = time.perf_counter()
    if reference:
        report = {"runs": []}
        predictions = _predictor.predict_reference(current, previous, report)
    else:
        predictions = _predictor(current, previous)
        report = {"runs": [_predictor.run_report]}
    predicted = time.perf_counter()
    memory_before = gpu_memory_snapshot(_predictor)
    trim = trim_previous_gpu_cache(_predictor)
    memory = gpu_memory_snapshot(_predictor) if trim is not None else memory_before
    if memory is not None and report["runs"] and report["runs"][-1].get("frames"):
        report["runs"][-1]["frames"][-1]["gpu_memory"] = memory
        if trim is not None:
            report["runs"][-1]["frames"][-1].update(
                gpu_memory_before_trim=memory_before, gpu_previous_cache_trim=trim)
    # Avoid stacking five full-size probability arrays for interprocess IPC.
    heads = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
        shape=(*current.shape[:2], 2))
    try:
        for index, name in enumerate(("co2_presence_probability", "gas_probability_conditional_on_co2")):
            values = predictions[name]
            if values.dtype != np.float32 or values.shape != current.shape[:2]:
                raise ValueError("Optical lookahead requires unchanged float32 two-head outputs")
            heads[..., index] = values
        heads.flush()
    finally:
        del heads
    finished = time.perf_counter()
    return dict(output=str(output_path), report=report, pid=os.getpid(),
        backend="cuda-float32-checked" if getattr(_predictor, "verify_numerics", False)
                else "cuda-float32", reused=False, started_unix=started_unix,
        verify_numerics=bool(getattr(_predictor, "verify_numerics", False)),
        finished_unix=time.time(), seconds=finished-started, gpu_memory=memory,
        gpu_memory_before_trim=memory_before if trim is not None else None,
        gpu_previous_cache_trim=trim,
        timing=dict(input_load_s=loaded-started, inference_s=predicted-loaded,
                    heads_write_s=finished-predicted))


def _close_worker():
    global _predictor
    if _predictor is not None:
        predictor, _predictor = _predictor, None
        cache = getattr(predictor, "cache", None)
        try:
            if isinstance(cache, BoundedAsyncOpticalCache):
                cache.close()
                return dict(optical_cache=cache.stats, async_optical_cache=cache.timings)
        finally:
            predictor.close()


class QueuedCuda:
    """Bounded optical lookahead; reference calls do not advance history.

    In audit mode ``parity_threads`` come from the caller's CPU budget. The one
    feeder thread only decodes/corrects the next image and waits on CUDA. The
    optional factory is a picklable predictor factory for process-level tests.
    """

    def __init__(self, model, baseline, report, *, cache_root=None, parity_threads=1,
                 predictor_factory=None, optical_cache_root=None, optical_cache_max_bytes=None,
                 verify_numerics=False, lookahead_frames=2):
        if not isinstance(lookahead_frames, int) or not 1 <= lookahead_frames <= 3:
            raise ValueError("Choose one to three optical lookahead frames")
        root = Path(cache_root or os.environ.get("PHASE_OPTICAL_PREFETCH_CACHE",
            str(Path(tempfile.gettempdir()) / "darsia-optical-prefetch"))).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._scratch_root = root
        self.scratch = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        self.frame = 0
        self.lookahead_frames = lookahead_frames
        self._next_schedule = 0
        self._last_scheduled = None
        self.closed = False
        self.prepared = False
        self._pending = deque()
        self._previous_path = self._previous_digest = None
        self.last_receipt = None
        self.verify_numerics = bool(verify_numerics)
        self.run_report = dict(shape_hw=list(baseline.shape[:2]), frames=[], parity=[],
            verify_numerics=self.verify_numerics,
            execution=dict(method=VERSION, lookahead_frames=lookahead_frames,
                           parity_threads=max(1, int(parity_threads)) if self.verify_numerics else 0))
        report.setdefault("runs", []).append(self.run_report)
        baseline_path = self.scratch / "baseline.npy"
        np.save(baseline_path, baseline, allow_pickle=False)
        self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
            initializer=_initialize,
            initargs=(model, str(baseline_path), max(1, int(parity_threads)), predictor_factory,
                      optical_cache_root, optical_cache_max_bytes, self.verify_numerics))
        self._feeder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optical-lookahead")

    def prepare_frames(self, *, review_root, run_id, frame_count, content_box,
                       independent_controls, radiometry, radiometry_profile):
        self._ensure_open()
        if self.prepared or self.frame:
            raise RuntimeError("Optical timeline has already started")
        if set(independent_controls) - {0}:
            raise ValueError("Only an initial independent control is supported")
        self.review_root, self.run_id = Path(review_root), run_id
        self.frame_count = int(frame_count)
        self.content_box = tuple(content_box)
        self.independent_controls = set(independent_controls)
        self.radiometry = radiometry
        self.prepared = True
        self.run_report["run"] = run_id
        self._schedule()

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("Optical predictor is closed")

    def _schedule(self):
        while (self._next_schedule < self.frame_count
               and len(self._pending) < self.lookahead_frames):
            index = self._next_schedule
            predecessor = self._last_scheduled
            prepared_input = Future()
            prepared = self._feeder.submit(self._prepare_and_predict, index,
                predecessor, prepared_input)
            self._pending.append((index, prepared, prepared_input))
            self._last_scheduled = prepared
            self._next_schedule += 1

    def _prepare_and_predict(self, index, predecessor, prepared_input):
        started = time.perf_counter()
        try:
            if index and index-1 not in self.independent_controls:
                previous = predecessor.result()
                previous_path = previous["current_path"]
                previous_digest = previous["current_digest"]
            else:
                previous_path = previous_digest = None
            path = self.review_root / "native_frames" / self.run_id / f"frame_{index:03d}.jpg"
            if self.radiometry is not None:
                current = self.radiometry.load(path, f"frame_{index:03d}")
            else:
                from PIL import Image
                x, y, width, height = self.content_box
                with Image.open(path) as image:
                    current = np.asarray(image.convert("RGB"), dtype=np.uint8)[y:y+height, x:x+width].copy()
            current_path = self.scratch / f"frame-{index}.npy"
            np.save(current_path, current, allow_pickle=False)
            current_digest = _digest(current)
        except BaseException as error:
            prepared_input.set_exception(error)
            raise
        prepared_input.set_result(current)
        prepared = time.perf_counter()
        submitted = time.time()
        computation = self._pool.submit(_compute, str(current_path), previous_path,
            str(self.scratch / f"heads-{index}.npy"))
        return dict(current=current, current_path=str(current_path),
            current_digest=current_digest, previous_digest=previous_digest,
            computation=computation, submitted_unix=submitted, prepare_s=prepared-started)

    def prepared_frame(self, frame_index):
        self._ensure_open()
        if not self.prepared or frame_index != self.frame or not self._pending:
            return None
        # Return the same corrected values submitted to CUDA; the caller must
        # treat these as immutable. __call__ checks their content before use.
        return self._pending[0][2].result()

    @staticmethod
    def _load_heads(receipt):
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
        path = Path(receipt["output"])
        heads = np.load(path, allow_pickle=False)
        path.unlink()
        return partition(heads)

    def __call__(self, current, previous):
        self._ensure_open()
        if not self.prepared:
            return self._explicit(current, previous, reference=False)
        if not self._pending:
            raise IndexError("All prepared optical frames have already been consumed")
        started = time.perf_counter()
        try:
            index, pending, _ = self._pending[0]
            if index != self.frame:
                raise RuntimeError("Optical lookahead sequence changed")
            prepared = pending.result()
            receipt = prepared["computation"].result()
            waited = time.perf_counter()-started
            if _digest(current) != prepared["current_digest"] or _digest(previous) != prepared["previous_digest"]:
                raise ValueError("Prepared optical current/predecessor differs from chronological caller inputs")
            receipt["frame"] = index
            receipt["submitted_unix"] = prepared["submitted_unix"]
            receipt["timing"]["prepare_s"] = prepared["prepare_s"]
            loading = time.perf_counter()
            result = self._load_heads(receipt)
            receipt["consumer"] = dict(wait_s=waited, heads_load_s=time.perf_counter()-loading)
            self._accept_report(receipt)
            self._pending.popleft()
            self.frame += 1
            # Frame N-1 can be removed only after CUDA finishes frame N,
            # because N's optical features may read N-1. Queued N+1/N+2
            # have already captured their own predecessor paths.
            if index:
                (self.scratch / f"frame-{index-1}.npy").unlink(missing_ok=True)
            if self.frame == self.frame_count:
                Path(prepared["current_path"]).unlink(missing_ok=True)
            self._schedule()
            return result
        except BaseException as error:
            self._close_preserving(error)
            raise

    def _accept_report(self, receipt):
        worker_report = receipt["report"]["runs"][0]
        self.run_report.update(worker_report)
        self.last_receipt = {key: value for key, value in receipt.items() if key != "report"}

    def _explicit(self, current, previous, *, reference, report=None):
        token = uuid.uuid4().hex
        paths = [self.scratch / f"{token}-{name}.npy" for name in ("current", "previous", "heads")]
        failure = None
        try:
            np.save(paths[0], current, allow_pickle=False)
            if previous is not None:
                np.save(paths[1], previous, allow_pickle=False)
            receipt = self._pool.submit(_compute, str(paths[0]),
                None if previous is None else str(paths[1]), str(paths[2]), reference).result()
            result = self._load_heads(receipt)
            if reference:
                report.setdefault("runs", []).extend(receipt["report"]["runs"])
            else:
                self._accept_report(receipt)
                self.frame += 1
            return result
        except BaseException as error:
            failure = error
            self._close_preserving(error)
            raise
        finally:
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Temporary optical file cleanup also failed: {cleanup_error}")

    def predict_reference(self, current, previous, report):
        self._ensure_open()
        return self._explicit(current, previous, reference=True, report=report)

    def _close_preserving(self, error):
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f"Optical cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._feeder.shutdown(wait=True, cancel_futures=True)
            try:
                final_cache = self._pool.submit(_close_worker).result()
                if final_cache is not None:
                    self.run_report.update(final_cache)
            finally:
                self._pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if self._pending is not None:
                self._pending.clear()
            self._last_scheduled = None
            if (self.scratch.resolve().parent != self._scratch_root
                    or not self.scratch.name.startswith("run-")):
                raise RuntimeError("Refusing cleanup outside the optical scratch root")
            shutil.rmtree(self.scratch)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception is None:
            self.close()
        else:
            self._close_preserving(exception)
