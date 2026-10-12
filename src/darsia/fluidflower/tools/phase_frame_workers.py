"""Local shared optical-frame workers; chronological physics stays with its run.

Only paths and small receipts cross process boundaries. Reference/model caches
are bounded; completed probability fields live on disk, never in future queues.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import threading
import time
import uuid

VERSION = "shared-optical-frame-workers-v3-gpu-first"
# Measured 2026-09-17 (docs/phase_raw_to_map_critical_path_20260917.md): with twelve CPU
# optical workers beside one GPU worker, finished optical results waited a median of 565 s
# for the ordered history/physics loop, that loop ran 3x slower (113 s vs 37 s per image)
# and the GPU needed 12 s instead of about 1 s per frame. The default therefore leaves
# every frame the GPU can compute to the GPU; CPU slots then serve the ordered loops.
CPU_OPTICAL_WITH_GPU_ENV = "PHASE_FRAME_CPU_OPTICAL_WITH_GPU"
ENV = "PHASE_FRAME_BROKER"
_contexts = OrderedDict()
_cuda_validations = OrderedDict()


def _initialize_worker():
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    import cv2
    cv2.setNumThreads(1)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _read_rgb(path, box):
    import numpy as np
    from PIL import Image
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    x, y, w, h = box
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))[y:y+h, x:x+w].copy()


def compute_frame(task, backend):
    """The existing optical algorithms, supplied with explicit previous RGB."""
    import numpy as np
    import psutil
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
    from darsia.fluidflower.phase_validation import predict_image_tiled
    from darsia.fluidflower.tools.materialize_phase_iteration import CheckedCuda
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
    started = time.perf_counter()
    started_unix = time.time()
    registered_v5 = task.get("v5_geometry") is not None
    if registered_v5:
        from darsia.fluidflower.project import sha256_file
        from darsia.fluidflower.phase_validation import _model_identity
        if _model_identity(Path(task["model"]))[1] != task["model_sha"]:
            raise ValueError("V5 optical task model files changed")
        for name, digest_name in (("baseline", "baseline_sha"),
                                  ("current", "current_sha"),
                                  ("previous", "previous_sha")):
            path = task.get(name)
            expected = task.get(digest_name)
            if path is None:
                if expected is not None:
                    raise ValueError(f"V5 optical task {name} SHA without input")
            elif not expected or sha256_file(Path(path)) != expected:
                raise ValueError(f"V5 optical task {name} input changed")
        geometry = task["v5_geometry"]
        native_path = Path(geometry["native_manifest_path"])
        if sha256_file(native_path) != geometry["native_manifest_sha256"]:
            raise ValueError("V5 optical task native manifest changed")
        native = json.loads(native_path.read_text(encoding="utf-8"))
        reference_contract = (native.get("v5_temporal_reference") or {}).get("contract")
        if reference_contract != geometry.get("temporal_reference_contract") or reference_contract not in {
                "ffac-v5-native-crop-source-supported-v1",
                "ffac-v5-source-backed-extended-ring-v1"}:
            raise ValueError("V5 optical task temporal reference contract changed")
        if reference_contract == "ffac-v5-source-backed-extended-ring-v1":
            for name in ("baseline_ring_sha256", "previous_ring_sha256"):
                digest = geometry.get(name)
                if name == "baseline_ring_sha256" or task.get("previous") is not None:
                    if not isinstance(digest, str) or len(digest) != 64:
                        raise ValueError(f"V5 optical task lacks {name}")
            ring_rows = [(native.get("reference") or {}, geometry["baseline_ring_sha256"])]
            previous_index = geometry.get("previous_frame_index")
            if previous_index is not None:
                frames = native.get("frame_sources") or []
                if not isinstance(previous_index, int) or not 0 <= previous_index < len(frames):
                    raise ValueError("V5 optical task predecessor is outside native timeline")
                ring_rows.append((frames[previous_index], geometry["previous_ring_sha256"]))
            for row, digest in ring_rows:
                if (row.get("v5_extended_ring_sha256") != digest
                        or sha256_file(Path(row.get("v5_extended_ring_path") or "")) != digest):
                    raise ValueError("V5 optical task source-backed ring changed")
        elif geometry.get("baseline_ring_sha256") is not None or geometry.get("previous_ring_sha256") is not None:
            raise ValueError("V5 native-crop task unexpectedly binds an extended ring")
    output = Path(task["output"])
    receipt_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if task.get("attempt"):
        marker = output.with_suffix(".running.json")
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(attempt=task["attempt"], pid=os.getpid())), encoding="utf-8")
        os.replace(temporary, marker)
    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("key") == task["key"] and receipt.get("backend") == backend
                and (not registered_v5 or
                     (receipt.get("heads_sha256") == sha256_file(output)
                      and receipt.get("current_sha256") == task["current_sha"]
                      and receipt.get("baseline_sha256") == task["baseline_sha"]
                      and receipt.get("temporal_reference_contract") ==
                          task["v5_geometry"]["temporal_reference_contract"]
                      and receipt.get("baseline_ring_sha256") ==
                          task["v5_geometry"]["baseline_ring_sha256"]
                      and receipt.get("previous_ring_sha256") ==
                          task["v5_geometry"].get("previous_ring_sha256")))):
            try:
                with np.load(output, allow_pickle=False) as cached:
                    if cached["heads"].shape == (*task["shape"], 2):
                        return {**receipt, "reused": True, "reuse_checked_unix": time.time()}
            except (OSError, ValueError, KeyError):
                pass
    context_started = time.perf_counter()
    context_key = (task["model"], task["baseline"], backend,
                   "registered-v5" if registered_v5 else "legacy",
                   task["v5_geometry"]["native_manifest_sha256"] if registered_v5 else None)
    context_reused = False
    context = _contexts.pop(context_key, None)
    if context is None:
        # Bound BEFORE allocating a replacement, including CUDA baseline features.
        limit = 1 if backend == "cuda" else 2
        while len(_contexts) >= limit:
            _, old = _contexts.popitem(last=False)
            if old.get("gpu") is not None:
                old["gpu"].close()
            del old
        baseline = np.load(task["baseline"], allow_pickle=False, mmap_mode="r")
        model = HierarchicalPhaseModel.load(Path(task["model"]), adapter="generic")
        context = dict(model=model, baseline=baseline,
                       gpu=(CheckedCuda(model, baseline, {"runs": []})
                            if backend == "cuda" and not registered_v5 else None),
                       registered_predictor=None)
        if registered_v5:
            from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput

            native_path = Path(task["v5_geometry"]["native_manifest_path"])
            v5_input = RunV5LinearInput(
                native_manifest_path=native_path,
                run_id=native_path.parent.name,
                content_shape_hw=tuple(task["shape"]),
            )
            if task["v5_geometry"]["temporal_reference_contract"] == "ffac-v5-source-backed-extended-ring-v1":
                reference_photo = (v5_input.review_root / "native_references" /
                                   v5_input.run_id / "pre_injection.jpg")
                if (v5_input.reference.get("v5_extended_ring_sha256") !=
                        task["v5_geometry"]["baseline_ring_sha256"]):
                    raise ValueError("V5 optical task B ring differs from native manifest")
                extended, origin, valid = v5_input.load_extended(
                    reference_photo, "pre_injection")
                pad = -origin[0]
                if (origin != (-pad, -pad) or not np.array_equal(
                        extended[pad:-pad, pad:-pad], baseline)):
                    raise ValueError("V5 optical task extended B differs from prepared core")
                context.update(baseline=extended, baseline_origin=origin,
                               baseline_valid=valid, v5_input=v5_input)
            else:
                context.update(baseline_origin=(0, 0), baseline_valid=None,
                               v5_input=v5_input)
    else:
        context_reused = True
    _contexts[context_key] = context
    context_seconds = time.perf_counter() - context_started
    load_started = time.perf_counter()
    current = _read_rgb(task["current"], task["box"])
    previous = _read_rgb(task["previous"], task["box"]) if task["previous"] else None
    load_seconds = time.perf_counter() - load_started
    parity = []
    inference_started = time.perf_counter()
    if registered_v5:
        from darsia.fluidflower.project import sha256_file
        from darsia.fluidflower.tools.ffac_v5_frame_context import V5FrameGeometryCohort
        from darsia.fluidflower.tools.ffac_v5_track1_photo_context import Track1FramePhotoCohort
        from darsia.fluidflower.tools.phase_v5_registered_frame import predict_registered_v5_frame

        geometry = task["v5_geometry"]
        manifest_path = Path(geometry["native_manifest_path"])
        if sha256_file(manifest_path) != geometry["native_manifest_sha256"]:
            raise ValueError("V5 optical task native geometry manifest changed")
        native = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if native.get("v5_track1_photo_geometry") and native.get("v5_frame_geometry"):
            raise ValueError("V5 optical task cannot mix photo-only and physical geometry")
        cohort = (Track1FramePhotoCohort(manifest_path)
                  if native.get("v5_track1_photo_geometry") else
                  V5FrameGeometryCohort(manifest_path))
        binding = cohort.frame(task["frame"])
        if binding["receipt_sha256"] != geometry["frame_receipt_sha256"]:
            raise ValueError("V5 optical task frame geometry receipt changed")
        previous_index = geometry.get("previous_frame_index")
        if (previous is None) != (previous_index is None):
            raise ValueError("V5 optical task predecessor index/pixels differ")
        feature_mask, feature_receipt = cohort.feature_support(
            task["frame"], previous_index, context["v5_input"])
        if (feature_receipt["receipt_sha256"] != geometry["feature_support_receipt_sha256"]
                or feature_receipt["feature_support_mask_sha256"] !=
                    geometry["feature_support_mask_sha256"]):
            raise ValueError("V5 optical task feature source support changed")
        baseline_to_previous = None
        previous_extended = None
        previous_origin = (0, 0)
        previous_valid = None
        if previous_index is not None:
            previous_binding = cohort.frame(int(previous_index))
            if previous_binding["receipt_sha256"] != geometry["previous_receipt_sha256"]:
                raise ValueError("V5 optical task predecessor receipt changed")
            baseline_to_previous = cohort.H_v5_B_to_frame(int(previous_index))
            if geometry["temporal_reference_contract"] == "ffac-v5-source-backed-extended-ring-v1":
                v5_input = context["v5_input"]
                previous_id = f"frame_{int(previous_index):03d}"
                if (v5_input.frames[previous_id].get("v5_extended_ring_sha256") !=
                        geometry["previous_ring_sha256"]):
                    raise ValueError("V5 optical task previous ring differs from native manifest")
                previous_photo = (v5_input.review_root / "native_frames" /
                                  v5_input.run_id / f"{previous_id}.jpg")
                previous_extended, previous_origin, previous_valid = (
                    v5_input.load_extended(previous_photo, previous_id))
                pad = -previous_origin[0]
                if (previous_origin != (-pad, -pad) or not np.array_equal(
                        previous_extended[pad:-pad, pad:-pad], previous)):
                    raise ValueError("V5 optical task extended previous differs from prepared core")
            else:
                previous_extended = previous
        predictions, aligned_baseline, alignment, predictor = predict_registered_v5_frame(
            context["model"], baseline=np.asarray(context["baseline"]),
            current=np.asarray(current), previous=previous_extended,
            h_baseline_to_current=cohort.H_v5_B_to_frame(task["frame"]),
            h_baseline_to_previous=baseline_to_previous,
            required_valid_mask=feature_mask, backend=backend,
            baseline_origin_xy=context["baseline_origin"],
            previous_origin_xy=previous_origin,
            baseline_source_valid_mask=context["baseline_valid"],
            previous_source_valid_mask=previous_valid,
            predictor=context["registered_predictor"])
        context["registered_predictor"] = predictor
        alignment = {**alignment,
            "frame_geometry_receipt_sha256": binding["receipt_sha256"],
            "feature_support_receipt_sha256": feature_receipt["receipt_sha256"],
            "feature_support_mask_sha256": feature_receipt["feature_support_mask_sha256"],
            "aligned_baseline_sha256": hashlib.sha256(aligned_baseline.tobytes()).hexdigest(),
            "native_manifest_sha256": cohort.native_manifest_sha256}
        if backend == "cuda" and task["frame"] in (0, 1, 10):
            reference, _, _, _ = predict_registered_v5_frame(
                context["model"], baseline=np.asarray(context["baseline"]),
                current=np.asarray(current), previous=previous_extended,
                h_baseline_to_current=cohort.H_v5_B_to_frame(task["frame"]),
                h_baseline_to_previous=baseline_to_previous,
                required_valid_mask=feature_mask, backend="cpu",
                baseline_origin_xy=context["baseline_origin"],
                previous_origin_xy=previous_origin,
                baseline_source_valid_mask=context["baseline_valid"],
                previous_source_valid_mask=previous_valid)
            for name in ("co2_presence_probability", "gas_probability_conditional_on_co2"):
                error = np.abs(predictions[name] - reference[name])
                threshold = .5 if name.startswith("co2_") else .6
                check = dict(head=name, q99_abs_error=float(np.quantile(error, .99)),
                    disagreement=float(np.mean((predictions[name] >= threshold) !=
                                               (reference[name] >= threshold))))
                if (not np.isfinite(error).all() or check["q99_abs_error"] > 1e-3
                        or check["disagreement"] > 1e-4):
                    raise ValueError(f"V5 registered optical CPU/CUDA parity failed: {check}")
                parity.append(check)
    elif backend == "cuda":
        predictor = context["gpu"]
        # Arbitrary task order is safe: both RGBs and their actual frame keys
        # are explicit, rather than inheriting the worker's preceding task.
        proof_key = (context_key, task.get("code_sha"))
        # Only an explicitly enabled audit needs a first-task CPU proof.
        predictor.frame = (0 if predictor.verify_numerics and proof_key not in _cuda_validations
                           else task["frame"])
        predictor.gpu.previous = None
        predictor.gpu.previous_key = None
        predictions = predictor(current, previous)
        parity = list(predictor.run_report["parity"])
        if parity:
            for check in parity:
                check["frame"] = task["frame"]
            _cuda_validations[proof_key] = parity[-1]
            _cuda_validations.move_to_end(proof_key)
            while len(_cuda_validations) > 64:
                _cuda_validations.popitem(last=False)
        predictor.run_report["parity"].clear()
        predictor.run_report["frames"].clear()
        # Keep the baseline/model tensors, but release unused per-frame CUDA
        # allocations after the heads have been copied back to host memory.
        # Otherwise CuPy retains the peak frame allocation for the whole batch.
        predictor.gpu.previous = None
        predictor.gpu.previous_key = None
        predictor.gpu.cp.get_default_memory_pool().free_all_blocks()
        predictor.gpu.cp.get_default_pinned_memory_pool().free_all_blocks()
    else:
        predictions = predict_image_tiled(context["model"], current, context["baseline"],
            previous_rgb=previous, radiometry_profile=task.get("radiometry_profile"))
    inference_seconds = time.perf_counter() - inference_started
    heads = np.stack([predictions["co2_presence_probability"],
                      predictions["gas_probability_conditional_on_co2"]], axis=-1)
    # Check that storing only the independent heads loses no derived fields.
    restored = partition(heads)
    if any(not np.array_equal(predictions[name], restored[name]) for name in restored):
        raise ValueError("Optical output does not match the two-head storage contract")
    write_started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, heads=heads)
    os.replace(temporary, output)
    heads_sha = sha256_file(output) if registered_v5 else None
    write_seconds = time.perf_counter() - write_started
    memory = psutil.Process().memory_info()
    receipt = dict(key=task["key"], output=str(output), backend=backend, frame=task["frame"],
        parity=parity, seconds=time.perf_counter()-started, pid=os.getpid(),
        timing=dict(context_s=context_seconds, context_reused=context_reused,
                    raw_load_s=load_seconds, inference_s=inference_seconds,
                    write_s=write_seconds),
        cuda_validation=(_cuda_validations.get(proof_key)
                         if backend == "cuda" and not registered_v5 else None),
        v5_alignment=alignment if registered_v5 else None,
        heads_sha256=heads_sha,
        current_sha256=task["current_sha"] if registered_v5 else None,
        baseline_sha256=task["baseline_sha"] if registered_v5 else None,
        temporal_reference_contract=(task["v5_geometry"]["temporal_reference_contract"]
                                     if registered_v5 else None),
        baseline_ring_sha256=(task["v5_geometry"]["baseline_ring_sha256"]
                              if registered_v5 else None),
        previous_ring_sha256=(task["v5_geometry"].get("previous_ring_sha256")
                              if registered_v5 else None),
        rss_gib=memory.rss/2**30, peak_rss_gib=getattr(memory, "peak_wset", memory.rss)/2**30,
        # Wall clock, so that one photograph can be followed across processes
        # (scripts/phase_pipeline_timeline.py). perf_counter values are per process.
        submitted_unix=task.get("submitted_unix"), dispatched_unix=task.get("dispatched_unix"),
        started_unix=started_unix, finished_unix=time.time(),
        reused=False)
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt), encoding="utf-8")
    os.replace(temporary, receipt_path)
    return receipt


def estimated_frame_gib(task):
    """Allowance for RGBs, outputs and tiled features, not a full run history."""
    h, w = task["shape"]
    tile = min(h, 384 + max(32, int(.06 * min(h, w))))
    return 2.0 + (128 * h * w + 512 * tile * w) / 2**30


class FrameBroker:
    def __init__(self, cpu_workers, gpu_workers, *, available_memory=None, executor_factory=None,
                 cpu_optical_with_gpu=False):
        import psutil
        self.cpu_workers, self.gpu_workers = cpu_workers, gpu_workers
        # False = GPU-first. True restores the earlier mixed dispatch, e.g. for an A/B timing.
        self.cpu_optical_with_gpu = bool(cpu_optical_with_gpu)
        self.available_memory = available_memory or (lambda: psutil.virtual_memory().available / 2**30)
        self.headroom = min(24., max(8., psutil.virtual_memory().total / 2**30 * .1))
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.jobs = {}
        self.active = {}
        self.history = set()
        self.waiting_history = 0
        self.stopping = False
        self.memory_wait = ""
        self.peak_rss = 0.
        self.observed_peaks = {}
        factory = executor_factory or (lambda n: ProcessPoolExecutor(n,
            mp_context=mp.get_context("spawn"), initializer=_initialize_worker))
        # Executors spawn lazily. The CPU ceiling can change without replacing
        # running processes or discarding their model/reference caches.
        self.pools = {"cpu": factory(14)}
        self.pools["cuda"] = factory(1)
        self.dispatcher = threading.Thread(target=self._dispatch, daemon=True)
        self.dispatcher.start()

    def submit(self, owner, task):
        key = task["key"]
        with self.condition:
            if self.stopping:
                raise RuntimeError("Frame workers stopped")
            required = task.get("required_backend")
            if required == "cuda" and not self.gpu_workers:
                raise ValueError(f"This frame requires a {required} worker; none selected")
            if key not in self.jobs:
                self.jobs[key] = dict(state="pending", task=task, owner=owner, submitted_unix=time.time())
                self.pending.setdefault(owner, deque()).append(key)
            self.condition.notify_all()
        return key

    def result(self, key):
        with self.condition:
            while self.jobs[key]["state"] not in ("complete", "failed"):
                if self.stopping:
                    raise RuntimeError("Frame workers stopped")
                self.condition.wait(1)
            item = self.jobs[key]
            if item["state"] == "failed":
                raise RuntimeError(item["error"])
            return item["receipt"]

    def acquire_history(self, owner):
        # With GPU-only selection, one CPU coordinator must still run physics.
        limit = max(1, self.cpu_workers)
        with self.condition:
            self.waiting_history += 1
            try:
                while sum(v["backend"] == "cpu" for v in self.active.values()) + len(self.history) >= limit:
                    if self.stopping:
                        raise RuntimeError("Frame workers stopped")
                    self.condition.wait(1)
                if owner in self.history:
                    raise RuntimeError("Nested chronological CPU claim")
                self.history.add(owner)
            finally:
                self.waiting_history -= 1

    def release_history(self, owner):
        with self.condition:
            self.history.discard(owner)
            self.condition.notify_all()

    def cancel_owner(self, owner):
        with self.condition:
            for key in self.pending.pop(owner, ()):
                self.jobs[key].update(state="failed", error="Experiment cancelled")
            self.history.discard(owner)
            # The run's decision threads claim "<owner>#decision-<thread>" (decision_slot). A run
            # that died holding one must not keep that CPU slot for the rest of the batch.
            for claim in [name for name in self.history if str(name).startswith(f"{owner}#decision-")]:
                self.history.discard(claim)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            cpu = sum(v["backend"] == "cpu" for v in self.active.values())
            gpu = sum(v["backend"] == "cuda" for v in self.active.values())
            return dict(mode=VERSION, cpu_workers=self.cpu_workers, gpu_workers=self.gpu_workers,
                optical_policy=("cpu-only" if not self.gpu_workers else
                                "cpu-and-gpu" if self.cpu_optical_with_gpu else "gpu-first"),
                active_cpu=cpu, active_gpu=gpu, chronological_cpu=len(self.history),
                active_workers=cpu+gpu+len(self.history),
                queued_frames=sum(len(q) for q in self.pending.values()),
                memory_wait=self.memory_wait, free_headroom_gib=self.headroom,
                available_gib=round(self.available_memory(), 2), peak_worker_rss_gib=self.peak_rss)

    def set_cpu_workers(self, count):
        return self.set_workers(count, self.gpu_workers)

    def set_workers(self, cpu_workers, gpu_workers):
        if (not isinstance(cpu_workers, int) or not 0 <= cpu_workers <= 14
                or gpu_workers not in (0, 1) or cpu_workers + gpu_workers < 1):
            raise ValueError("Choose 0-14 CPU workers and 0-1 GPU workers; keep at least one")
        with self.condition:
            self.cpu_workers, self.gpu_workers = cpu_workers, gpu_workers
            self.condition.notify_all()
        return self.snapshot()

    def _take(self, backend):
        for owner, queue in list(self.pending.items()):
            for key in queue:
                task = self.jobs[key]["task"]
                if task.get("required_backend") not in (None, backend):
                    continue
                if (backend == "cpu" and self.gpu_workers and not self.cpu_optical_with_gpu
                        and task.get("required_backend") is None):
                    # GPU-first: this frame is left to the selected GPU worker. The live
                    # worker count is read on every pass, so switching the GPU off during
                    # a batch hands the remaining frames back to the CPU slots at once.
                    continue
                needed = max(estimated_frame_gib(task), 1.15 * self.observed_peaks.get(
                    (backend, tuple(task["shape"])), 0.))
                reserved = sum(self._unallocated_allowance(v) for v in self.active.values())
                available = self.available_memory()
                if available < self.headroom + reserved + needed:
                    self.memory_wait = (f"Waiting for RAM: {available:.1f} GiB available; "
                        f"{self.headroom:.1f} GiB reserve + {reserved+needed:.1f} GiB frame allowances")
                    return None
                queue.remove(key)
                if queue:
                    self.pending.move_to_end(owner)
                else:
                    del self.pending[owner]
                return key, task, needed
        return None

    @staticmethod
    def _unallocated_allowance(worker):
        # Available RAM already excludes resident arrays. Reserve only the
        # remainder; stale markers from a previous invocation cannot release it.
        try:
            import psutil
            marker = json.loads(Path(worker["marker"]).read_text(encoding="utf-8"))
            if marker["attempt"] == worker["attempt"]:
                rss = psutil.Process(marker["pid"]).memory_info().rss / 2**30
                return max(0., worker["allowance"] - rss)
        except (OSError, ValueError, KeyError, psutil.Error):
            pass
        return worker["allowance"]

    def _dispatch(self):
        while True:
            with self.condition:
                if self.stopping:
                    return
                self.memory_wait = ""
                for backend, count in (("cuda", self.gpu_workers), ("cpu", self.cpu_workers)):
                    if backend == "cpu" and self.waiting_history:
                        continue
                    busy = sum(v["backend"] == backend for v in self.active.values())
                    if backend == "cpu":
                        busy += len(self.history)
                    for _ in range(max(0, count-busy)):
                        selected = self._take(backend)
                        if selected is None:
                            break
                        key, task, needed = selected
                        task = {**task, "attempt": uuid.uuid4().hex,
                                "submitted_unix": self.jobs[key].get("submitted_unix"),
                                "dispatched_unix": time.time()}
                        try:
                            future = self.pools[backend].submit(compute_frame, task, backend)
                        except Exception as error:
                            self.jobs[key].update(state="failed", error=f"Worker pool failed: {error}")
                            self.condition.notify_all()
                            continue
                        self.active[key] = dict(backend=backend, allowance=needed,
                            attempt=task["attempt"], marker=str(Path(task.get("output", key)).with_suffix(".running.json")))
                        self.jobs[key]["state"] = "running"
                        future.add_done_callback(lambda f, k=key: self._finished(k, f))
                self.condition.wait(.1)

    def _finished(self, key, future):
        with self.condition:
            try:
                receipt = future.result()
                self.jobs[key].update(state="complete", receipt=receipt)
                self.peak_rss = max(self.peak_rss, receipt.get("peak_rss_gib", 0.))
                shape_key = (self.active[key]["backend"], tuple(self.jobs[key]["task"]["shape"]))
                self.observed_peaks[shape_key] = max(self.observed_peaks.get(shape_key, 0.),
                                                     receipt.get("peak_rss_gib", 0.))
            except BaseException as error:
                self.jobs[key].update(state="failed", error=f"{type(error).__name__}: {error}")
            self.active.pop(key, None)
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.dispatcher.join(timeout=3)
        for pool in self.pools.values():
            # Python 3.13: shutdown(cancel_futures) cannot stop running workers.
            processes = list((getattr(pool, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)


@contextmanager
def serve_frame_workers(cpu_workers, gpu_workers):
    broker = FrameBroker(cpu_workers, gpu_workers, cpu_optical_with_gpu=os.environ.get(
        CPU_OPTICAL_WITH_GPU_ENV, "").strip().lower() in {"1", "true", "yes"})
    class Server(BaseManager):
        pass
    Server.register("broker", callable=lambda: broker)
    auth = os.urandom(32)
    server = Server(address=("127.0.0.1", 0), authkey=auth).get_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    broker.connection_env = {ENV: json.dumps(dict(address=server.address, auth=auth.hex()))}
    try:
        yield broker
    finally:
        broker.close()
        server.stop_event.set()


def connect_frame_workers():
    settings = json.loads(os.environ[ENV])
    class Client(BaseManager):
        pass
    Client.register("broker")
    manager = Client(address=tuple(settings["address"]), authkey=bytes.fromhex(settings["auth"]))
    manager.connect()
    return manager.broker()


class ParallelFramePredictor:
    supports_registered_v5 = True

    def __init__(self, model_root, baseline, report, backend, cache_root):
        import numpy as np
        from darsia.fluidflower.project import sha256_file
        self.broker = connect_frame_workers()
        self.model_root, self.report, self.backend = Path(model_root), report, backend
        self.owner = str(self.model_root)
        self.frame = 0
        self.tasks = []
        self.prepared_paths = []
        self.frame_inputs = []
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(baseline.tobytes()).hexdigest()
        self.baseline_path = self.cache_root / f"baseline-{digest}.npy"
        if not self.baseline_path.exists():
            np.save(self.baseline_path, baseline, allow_pickle=False)
        from darsia.fluidflower import phase_validation, phase_features, phase_learning
        from darsia.fluidflower.tools import phase_inference_benchmark_backends, materialize_phase_iteration
        sources = [Path(__file__), *(Path(module.__file__) for module in (
            phase_validation, phase_features, phase_learning,
            phase_inference_benchmark_backends, materialize_phase_iteration))]
        self.code_sha = hashlib.sha256("".join(sha256_file(p) for p in sources).encode()).hexdigest()
        self.run_report = dict(shape_hw=list(baseline.shape[:2]), frames=[], parity=[])
        self.last_receipt = None
        self.report["runs"].append(self.run_report)

    def prepare_frames(self, *, review_root, run_id, frame_count, content_box,
                       independent_controls, radiometry, radiometry_profile,
                       frame_geometry=None):
        import numpy as np
        from darsia.fluidflower.project import sha256_file
        from darsia.fluidflower.phase_validation import _model_identity
        self.run_report["run"] = run_id
        if not hasattr(self, "frame_inputs"):
            self.frame_inputs = []
        model_sha = _model_identity(self.model_root)[1]
        previous = None
        previous_index = None
        bound_receipts = {}
        temporal_reference_contract = None
        if frame_geometry is not None:
            from darsia.fluidflower.phase_training import v5_linear_input
            from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput
            from darsia.fluidflower.tools import ffac_v5_extended_reference
            from darsia.fluidflower.tools import (
                ffac_v5_frame_context, ffac_v5_temporal_alignment,
                ffac_v5_track1_photo_context, phase_v5_registered_frame,
            )
            if not isinstance(radiometry, RunV5LinearInput) or radiometry_profile is not None:
                raise ValueError("Per-frame V5 geometry requires source-bound float input")
            native_manifest = json.loads(Path(frame_geometry.native_manifest_path).read_text(
                encoding="utf-8"))
            temporal_reference_contract = (native_manifest.get("v5_temporal_reference") or {}).get(
                "contract")
            if temporal_reference_contract not in {
                    "ffac-v5-native-crop-source-supported-v1",
                    "ffac-v5-source-backed-extended-ring-v1"}:
                raise ValueError("Per-frame V5 geometry lacks an explicit temporal reference contract")
            sources = (ffac_v5_frame_context, ffac_v5_track1_photo_context,
                       ffac_v5_temporal_alignment,
                       phase_v5_registered_frame, ffac_v5_extended_reference,
                       v5_linear_input)
            self.code_sha = hashlib.sha256((self.code_sha + "".join(
                sha256_file(Path(module.__file__)) for module in sources)).encode()).hexdigest()
            self.run_report["v5_frame_geometry_manifest_sha256"] = (
                frame_geometry.native_manifest_sha256)
        baseline_sha = sha256_file(self.baseline_path) if frame_geometry is not None else None
        for index in range(frame_count):
            path = Path(review_root) / "native_frames" / run_id / f"frame_{index:03d}.jpg"
            geometry_task = None
            required_mask = None
            if frame_geometry is not None:
                required_mask, feature_receipt = frame_geometry.feature_support(
                    index, previous_index, radiometry)
                if (required_mask.dtype != np.bool_
                        or required_mask.shape != tuple(self.run_report["shape_hw"])
                        or not required_mask.any()):
                    raise ValueError("V5 optical frame has no valid source-supported feature area")
                bound_receipts[index] = feature_receipt["frame_geometry_receipt_sha256"]
                geometry_task = dict(
                    native_manifest_path=str(frame_geometry.native_manifest_path),
                    native_manifest_sha256=frame_geometry.native_manifest_sha256,
                    frame_receipt_sha256=bound_receipts[index],
                    previous_frame_index=previous_index,
                    previous_receipt_sha256=(None if previous_index is None else
                                             bound_receipts[previous_index]),
                    feature_support_receipt_sha256=feature_receipt["receipt_sha256"],
                    feature_support_mask_sha256=feature_receipt["feature_support_mask_sha256"],
                    temporal_reference_contract=temporal_reference_contract,
                    baseline_ring_sha256=(radiometry.reference.get("v5_extended_ring_sha256")
                        if temporal_reference_contract == "ffac-v5-source-backed-extended-ring-v1"
                        else None),
                    previous_ring_sha256=(None if previous_index is None or
                        temporal_reference_contract != "ffac-v5-source-backed-extended-ring-v1"
                        else
                        radiometry.frames[f"frame_{previous_index:03d}"].get(
                            "v5_extended_ring_sha256")),
                )
            if radiometry is not None:
                corrected = (radiometry.load(path, f"frame_{index:03d}",
                                             required_valid_mask=required_mask)
                             if frame_geometry is not None else
                             radiometry.load(path, f"frame_{index:03d}"))
                key = hashlib.sha256(corrected.tobytes()).hexdigest()
                path = self.cache_root / f"corrected-{key}.npy"
                if not path.exists():
                    np.save(path, corrected, allow_pickle=False)
            task = dict(model=str(self.model_root), model_sha=model_sha,
                baseline=str(self.baseline_path), current=str(path), previous=str(previous) if previous else None,
                **({"baseline_sha": baseline_sha} if baseline_sha is not None else {}),
                current_sha=sha256_file(path), previous_sha=sha256_file(previous) if previous else None,
                frame=index, box=list(content_box), shape=self.run_report["shape_hw"],
                radiometry_profile=radiometry_profile, code_sha=self.code_sha,
                required_backend="cpu" if self.backend == "cpu" else None,
                **({"v5_geometry": geometry_task} if geometry_task is not None else {}))
            task["key"] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
            task["output"] = str(self.cache_root / f"{task['key']}.npz")
            self.tasks.append(self.broker.submit(self.owner, task))
            self.prepared_paths.append(path)
            self.frame_inputs.append(dict(current_sha=task["current_sha"],
                baseline_sha=baseline_sha, output=task["output"], key=task["key"],
                registered_v5=frame_geometry is not None,
                temporal_reference_contract=temporal_reference_contract,
                feature_support_receipt_sha256=(None if geometry_task is None else
                                                 geometry_task["feature_support_receipt_sha256"]),
                baseline_ring_sha256=(None if geometry_task is None else
                                      geometry_task["baseline_ring_sha256"]),
                previous_ring_sha256=(None if geometry_task is None else
                                      geometry_task["previous_ring_sha256"])))
            if index not in independent_controls:
                previous = path
                previous_index = index

    @contextmanager
    def cpu_slot(self):
        slot_started = time.perf_counter()
        self.broker.acquire_history(self.owner)
        # The caller's physics timer starts before this wait; report it separately.
        self.last_slot_wait_s = time.perf_counter() - slot_started
        try:
            yield
        finally:
            self.broker.release_history(self.owner)

    @contextmanager
    def decision_slot(self):
        """A further chronological CPU claim of this run: the final decision of a photograph
        runs beside the history of the next one (scripts/phase_decision_pipeline.py). One claim
        per decision thread: the broker refuses a second claim under the same name."""
        owner = f"{self.owner}#decision-{threading.get_ident()}"
        self.broker.acquire_history(owner)
        try:
            yield
        finally:
            self.broker.release_history(owner)

    def __call__(self, current, previous):
        import numpy as np
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
        wait_started = time.perf_counter()
        receipt = self.broker.result(self.tasks[self.frame])
        waited = time.perf_counter() - wait_started
        expected = self.frame_inputs[self.frame]
        if expected["registered_v5"]:
            from darsia.fluidflower.project import sha256_file
            if (receipt.get("key") != expected["key"]
                    or receipt.get("output") != expected["output"]
                    or receipt.get("current_sha256") != expected["current_sha"]
                    or receipt.get("baseline_sha256") != expected["baseline_sha"]
                    or receipt.get("temporal_reference_contract") != expected["temporal_reference_contract"]
                    or (receipt.get("v5_alignment") or {}).get("feature_support_receipt_sha256") !=
                        expected["feature_support_receipt_sha256"]
                    or receipt.get("baseline_ring_sha256") != expected["baseline_ring_sha256"]
                    or receipt.get("previous_ring_sha256") != expected["previous_ring_sha256"]
                    or receipt.get("heads_sha256") != sha256_file(Path(receipt["output"]))):
                raise ValueError("V5 optical worker result differs from its input/output SHA receipt")
        load_started = time.perf_counter()
        with np.load(receipt["output"], allow_pickle=False) as archive:
            predictions = partition(archive["heads"])
        # optical_s in the frame timing = wait_s + heads_load_s. wait_s is ~0 when the
        # result was ready before the ordered loop asked for it.
        self.last_receipt = dict(receipt, consumer=dict(wait_s=waited,
            heads_load_s=time.perf_counter() - load_started))
        self.run_report["frames"].append({k: v for k, v in receipt.items() if k not in ("parity", "key", "output")})
        self.run_report["parity"].extend(receipt["parity"])
        self.frame += 1
        return predictions

    def prepared_frame(self, frame_index: int) -> np.ndarray | None:
        """Return the already corrected frame queued for a worker, if present.

        ``prepare_frames`` materializes radiometry-corrected ``.npy`` inputs
        before inference.  The validation loop can reuse that exact array for
        physics and concentration preparation instead of correcting the same
        raw photograph a second time.
        """

        if frame_index < 0 or frame_index >= len(self.prepared_paths):
            return None
        import numpy as np
        path = Path(self.prepared_paths[frame_index])
        if path.suffix.lower() != ".npy":
            return None
        if self.frame_inputs[frame_index]["registered_v5"]:
            from darsia.fluidflower.project import sha256_file
            if sha256_file(path) != self.frame_inputs[frame_index]["current_sha"]:
                raise ValueError("V5 prepared current frame changed before native review")
        return np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"))

    def close(self):
        self.broker.cancel_owner(self.owner)
