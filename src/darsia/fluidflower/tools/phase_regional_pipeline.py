"""Bounded parallel execution of the existing frame-local regional phase method.

Only the reviewed regional sequence and its four output guards are accepted.
They have no causal state: their baseline, model and geometry are immutable;
``last_diagnostics`` is their only per-frame state. Each task receives its own
wrapper instances and calls the original method unchanged. Optical prediction,
including the actual previous RGB photograph, remains on the ordered caller.

Native-library thread limits must be set once before creating this executor.
Workers never change process-wide OpenCV/BLAS settings. The resource planner
must budget both these workers and the native thread team used by each call.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

from darsia.fluidflower import phase_free_gas_output_guard as free_gas
from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower import phase_optical_regions_runtime as runtime
from darsia.fluidflower import phase_regional_gas as gas
from darsia.fluidflower import phase_region_output_guards as guards
from darsia.fluidflower import phase_bottom_gas_guard as bottom_gas
from darsia.fluidflower import phase_region_edge_guard as region_edge


VERSION = "parallel-frame-local-regional-phase-v2"
# Source review establishes that only last_diagnostics changes. A future
# implementation with additional state keeps the ordinary serial path until
# its independence has been reviewed and tested again.
REVIEWED_SOURCE_SHA256 = {
    # The optional V5 native-frame reference setter does not change the
    # legacy sequence until explicitly called.  Once called, its extra
    # source_baseline state fails the _LEAF_FIELDS check below, so this
    # immutable-context executor cannot silently fork a moving geometry.
    runtime: "89df4b55c7f517aea086799f4c19e528267655d119cdf318b9c651113816a1eb",
    gas: "f5b7607bc9be520888332a6a776b8ed0f83c7de4c12c26ee39710976a663b750",
    guards: "d4d3a0a0cce59423e2666d3f1f363392826a389d97ea8d3b88e07b1950da9cb8",
    free_gas: "6cfeaa6fcdd9c01b18d85eda5ba5bdf9e88674fd2370e33655223bae39bf5fff",
    regional: "afa4868d5eca7a4ceed46d82d4ea86a4c081dd02ed01de644224519db935b9e9",
    bottom_gas: "a31ded11494bb92cec72562e96c6b8766f188cc6314e32c3695f5ccf2e6f673a",
    region_edge: "9a8cb989b6055ae35a6f46948f521b179db2ec97fa165289191cac1ac5033e49",
}
_LEAF_FIELDS = {"context", "model", "settings", "baseline", "last_diagnostics"}
_WRAPPER_FIELDS = {"inherited", "context", "settings", "last_diagnostics"}
_BOTTOM_FIELDS = _WRAPPER_FIELDS | {"_prepared_geometry"}
# Match the canonical outside-to-inside order in phase_runtime. Bottom gas
# uses the current time and immutable injection schedule, not previous frames;
# edge smoothing uses only the current categorical support and fixed geometry.
_WRAPPERS = (region_edge.RegionEdgeSequence, bottom_gas.BottomGasSequence,
             free_gas.FreeGasSequence, guards.GuardedRegionSequence)


def unsupported_reason(sequence):
    """Fail closed for unknown/stateful/subclassed/replaced methods."""
    node = sequence
    for kind in _WRAPPERS:
        if type(node) is kind:
            expected = _BOTTOM_FIELDS if kind is bottom_gas.BottomGasSequence else _WRAPPER_FIELDS
            if set(vars(node)) != expected:
                return "output wrapper has unreviewed instance state"
            node = node.inherited
    if type(node) not in (runtime.OpticalRegionSequence, gas.RegionalGasSequence):
        return "not an allowlisted frame-local regional sequence"
    fields = _LEAF_FIELDS | ({"gas_settings"} if type(node) is gas.RegionalGasSequence else set())
    if set(vars(node)) != fields:
        return "regional sequence has unreviewed instance state"
    if node.baseline is None:
        return "registered baseline has not been bound"
    for module, expected in REVIEWED_SOURCE_SHA256.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            return f"{Path(module.__file__).name} differs from reviewed frame-local source"
    return None


def _fork(sequence):
    """Share immutable inputs; own every object with frame-local attributes."""
    fork = copy.copy(sequence)
    fork.last_diagnostics = {}
    if type(sequence) in _WRAPPERS:
        fork.inherited = _fork(sequence.inherited)
    return fork


def _copy_diagnostics(destination, source):
    destination.last_diagnostics = source.last_diagnostics
    if type(destination) in _WRAPPERS:
        _copy_diagnostics(destination.inherited, source.inherited)


@dataclass
class Decided:
    token: object
    output: dict
    report: dict
    diagnostics: dict
    seconds: dict
    sequence: object
    result: object = None


class ParallelRegional:
    """Small ordered queue of independent, unchanged CPU phase calls."""

    def __init__(self, sequence, *, workers=2, pending=None, slot=None, apply_function=None):
        reason = unsupported_reason(sequence)
        if reason is not None:
            raise ValueError(reason)
        self.workers = int(workers)
        self.limit = int(pending if pending is not None else self.workers)
        if self.workers < 1 or self.limit < self.workers:
            raise ValueError("Positive workers and at least one pending slot per worker are required")
        self.sequence, self.slot = sequence, slot
        self.apply_function = apply_function
        self.implementation = ("original-regional-method" if apply_function is None else
                               f"{apply_function.__module__}.{apply_function.__qualname__}")
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="regional-phase")
        self._pending = deque()
        self._closed = False
        self.submitted = self.delivered = self.max_pending = 0
        self.compute_seconds = self.wait_seconds = 0.

    def _compute(self, fork, predictions, elapsed_h, current, token, queued):
        with self.slot() if self.slot is not None else nullcontext():
            started = time.perf_counter()
            if self.apply_function is None:
                output, report = fork.apply_with_reference(predictions, elapsed_h, current)
            else:
                output, report = self.apply_function(fork, predictions, elapsed_h, current)
            finished = time.perf_counter()
        return Decided(token, output, report, fork.last_diagnostics,
                       dict(decision_s=finished-started, decision_queued_s=started-queued,
                            decision_started_monotonic=started, decision_finished_monotonic=finished,
                            original_regional_method_called=self.apply_function is None,
                            execution_implementation=self.implementation), fork)

    def _wait(self, task):
        started = time.perf_counter()
        try:
            return task.result()
        finally:
            self.wait_seconds += time.perf_counter()-started

    def wait_for_capacity(self):
        # The caller drains ready() immediately after this method. A completed
        # later frame still owns memory while waiting for the head of the queue.
        if len(self._pending) >= self.limit:
            self._wait(self._pending[0])

    def submit(self, predictions, elapsed_h, current, token=None):
        if self._closed:
            raise RuntimeError("Regional executor is closed")
        for task in self._pending:
            if task.done() and task.exception() is not None:
                task.result()
        # This also enforces a hard bound for callers which neglect the
        # wait/ready protocol, rather than retaining arbitrarily many frames.
        if len(self._pending) >= self.limit:
            raise RuntimeError("Drain ready regional results before submitting another frame")
        fork = _fork(self.sequence)
        self._pending.append(self._pool.submit(self._compute, fork, predictions,
            elapsed_h, current, token, time.perf_counter()))
        self.submitted += 1
        self.max_pending = max(self.max_pending, len(self._pending))

    def _take(self):
        decided = self._pending.popleft().result()
        self.compute_seconds += decided.seconds["decision_s"]
        self.delivered += 1
        # Match the serial observable diagnostics, in frame order. References
        # do not read them as history, and workers own separate wrapper objects.
        _copy_diagnostics(self.sequence, decided.sequence)
        return decided

    def ready(self):
        results = []
        while self._pending and self._pending[0].done():
            results.append(self._take())
        return results

    def drain(self):
        results = []
        while self._pending:
            self._wait(self._pending[0])
            results.append(self._take())
        return results

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)

    def abandon(self):
        # In-flight native calls cannot be interrupted safely. Join them before
        # returning from a failed run; no worker writes or publishes any files.
        self.close()
        self._pending.clear()

    def report(self):
        return dict(execution=VERSION, active=True, phase_workers=self.workers,
                    pending_limit=self.limit, max_pending=self.max_pending,
                    photographs=self.submitted, delivered=self.delivered,
                    decision_s=self.compute_seconds, loop_waited_for_decisions_s=self.wait_seconds,
                    optical_chronology_preserved=True,
                    original_regional_method_called=self.apply_function is None,
                    execution_implementation=self.implementation,
                    source_sha256={Path(m.__file__).name: h for m, h in REVIEWED_SOURCE_SHA256.items()})


def create(sequence, *, workers=2, pending=None, slot=None, apply_function=None):
    """Return None to retain serial execution on small or unreviewed runtimes."""
    if int(workers) < 2 or unsupported_reason(sequence) is not None:
        return None
    return ParallelRegional(sequence, workers=workers, pending=pending, slot=slot,
                            apply_function=apply_function)
