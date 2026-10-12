"""The final decision of one photograph runs while the next photograph's history is computed.

A unified-decision sequence does two things for every photograph:

1. chronological evidence - ``provider.apply_with_reference`` and, in the local-age version, the
   exposure history. Every photograph needs the history left by the photograph before it;
2. learned completion and the final decision - ``prepare_inputs`` -> network -> ``decide``. This is
   a pure function of that photograph's evidence. The frozen runtime says so itself: "Neither the
   completion proposal nor this decision is fed into that history."

The frozen ``apply_with_reference`` runs 1 and 2 back to back, so photograph t+1 waits for a
decision it does not depend on. Here step 1 stays on the calling thread, in order, and step 2 is
handed to a worker thread. Every library call is the frozen one with the same arguments, so every
value is the same; only the waiting disappears. NumPy, SciPy, OpenCV and scikit-image release the
interpreter lock in the calls that matter (measured 2026-09-17: two threads on two cores take
0.5-0.6x the serial wall time for each of them).

Nothing in the frozen modules is edited, and the chronological state of the sequence is only ever
touched by the calling thread. The two method bodies restated below are bound to the exact source
text they were taken from; any other text keeps the frozen call. The equivalence is also checked
on the running machine: for the first photographs and at intervals the frozen method runs on a
disposable fork of the sequence (the same fork that reference photographs use) and every output
field, diagnostic and report entry must be identical. As long as nothing unchecked has been
delivered (the one-by-one photographs and the first decision made on a worker thread) a difference
only means that this run keeps the frozen call; later it stops the run.
``PHASE_PIPELINED_DECISION=0`` keeps the frozen call.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

VERSION = "pipelined-final-decision-v1"
ENV = "PHASE_PIPELINED_DECISION"
THREADS_ENV = "PHASE_DECISION_THREADS"
LEGACY = "darsia.fluidflower.phase_decision_runtime"
LOCAL_AGE = "darsia.fluidflower.phase_decision_age_runtime"
# Runtime sources as of 2026-09-17; _chronological/_decide restate their apply_with_reference.
VERIFIED_RUNTIME_SHA256 = {
    LEGACY: frozenset({"700c99403092ffbea7bc366e1012a6f8703f2464129a5e72927d22bc9f26c353"}),
    LOCAL_AGE: frozenset({"00a9b3928f331c1fef7efa7808758f3c40e1ae6bdca75011b6450c1ccbfdce28"}),
}
# Checked against the frozen method on a fork: every photograph until this many decided (not
# pre-injection) photographs agreed, one by one; then the first decisions made on a worker thread
# (the next photograph is only handed over once such a check has finished, so a difference can
# still fall back to the frozen call); then every VERIFY_EVERY-th photograph.
VERIFY_FIRST_DECISIONS = 1
VERIFY_FIRST_OVERLAPPED = 1
VERIFY_EVERY = 50


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def decision_threads() -> int:
    try:
        return max(1, min(4, int(os.environ.get(THREADS_ENV, "1"))))
    except ValueError:
        return 1


def unsupported_reason(sequence) -> str | None:
    """None when this sequence's apply_with_reference is one of the two restated bodies."""
    if not enabled():
        return f"switched off by {ENV}"
    kind = type(sequence).__module__
    module = sys.modules.get(kind)
    if kind not in VERIFIED_RUNTIME_SHA256 or module is None or type(sequence) is not getattr(
            module, "DecisionSequence", None):
        return "not a unified decision sequence"
    if "apply_with_reference" in vars(sequence):
        return "apply_with_reference is replaced on this sequence"
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest not in VERIFIED_RUNTIME_SHA256[kind]:
        return f"{Path(module.__file__).name} differs from the verified source"
    return None


@dataclass
class Decided:
    """One finished photograph, exactly what the frozen apply_with_reference returns and leaves."""
    token: object
    output: dict
    report: dict
    diagnostics: dict
    result: object
    seconds: dict = field(default_factory=dict)


@dataclass
class _Frame:
    token: object
    predictions: dict
    time: float
    current_rgb: np.ndarray
    before: np.ndarray
    negative: bool
    report: dict
    spatial: np.ndarray | None
    first: np.ndarray | None
    expected: tuple | None
    seconds: dict
    recoverable: bool = False        # nothing unchecked has been delivered: a difference may fall back


class _LockedPredictor:
    """The shared completion network, one caller at a time (its CUDA fallback moves the model)."""

    def __init__(self, predictor):
        self.predictor, self.lock = predictor, threading.Lock()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "predictor"), name)

    def __deepcopy__(self, memo):
        return self

    def __call__(self, inputs):
        with self.lock:
            return self.predictor(inputs)


def _same_array(expected, actual) -> bool:
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return False
    if expected is actual:
        return True
    return bool(np.array_equal(np.ascontiguousarray(expected).reshape(-1).view(np.uint8),
                               np.ascontiguousarray(actual).reshape(-1).view(np.uint8)))


def _same_arrays(expected, actual) -> bool:
    return list(expected) == list(actual) and all(_same_array(expected[name], actual[name]) for name in expected)


def _canonical(report) -> str:
    # Not sorted: the order of the entries is part of what the report files look like.
    return json.dumps(report, default=str)


def differences(expected, decided) -> list[str]:
    """Names of everything that differs between the frozen method's frame and the pipelined one."""
    output, report, diagnostics, result = expected
    found = []
    if not _same_arrays(output, decided.output):
        found.append("output fields")
    if _canonical(report) != _canonical(decided.report):
        found.append("report")
    if not _same_arrays(diagnostics, decided.diagnostics):
        found.append("diagnostics")
    if not _same_array(result.labels, decided.result.labels):
        found.append("labels")
    if not _same_arrays(result.diagnostics, decided.result.diagnostics):
        found.append("decision fields")
    return found


class PipelinedDecision:
    """Chronological evidence on the calling thread, completion + decision on worker threads."""

    def __init__(self, sequence, *, threads=None, slot=None, fork=None,
                 verify_first_decisions=VERIFY_FIRST_DECISIONS,
                 verify_first_overlapped=VERIFY_FIRST_OVERLAPPED, verify_every=VERIFY_EVERY):
        reason = unsupported_reason(sequence)
        if reason is not None:
            raise ValueError(f"Pipelined decision is not available: {reason}")
        self.sequence = sequence
        self.kind = type(sequence).__module__
        self.module = sys.modules[self.kind]
        self.threads = decision_threads() if threads is None else max(1, int(threads))
        self.limit = self.threads + 1
        self.slot = slot or nullcontext
        if fork is None:
            # The fork reference photographs use. Imported now: if it cannot be imported here, the
            # self-check is impossible and create() keeps the frozen call.
            from darsia.fluidflower.tools.concentration_movie_compute import fork_sequence as fork
        self._fork = fork
        self.verify_first_decisions = int(verify_first_decisions)
        self.verify_first_overlapped = int(verify_first_overlapped)
        self.verify_every = int(verify_every)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="phase-decision")
        self._pending: deque[Future] = deque()
        self._lock = threading.Lock()
        self.disabled_reason = None
        self.submitted = self.verified = self.verified_decisions = self.verified_overlapped = 0
        self.frozen_frames = 0
        self.chronological_s = self.decision_s = self.verification_s = self.waited_s = 0.0
        self.overlapped_s = 0.0
        self.unchecked_submitted = 0
        self._gate: Future | None = None
        self._last = None
        self._share_predictor(create=True)

    def _share_predictor(self, *, create):
        """One completion network for the sequence, its forks and the worker threads.

        The frozen method builds it at the first decided photograph; building it earlier changes
        no value. A failure here is left for the frozen line to raise where it always did.
        """
        sequence = self.sequence
        with self._lock:
            if sequence._predictor is None and create:
                try:
                    sequence._predictor = self.module.completion._Predictor(sequence.checkpoint)
                except Exception:
                    return
            if sequence._predictor is not None and not isinstance(sequence._predictor, _LockedPredictor):
                sequence._predictor = _LockedPredictor(sequence._predictor)

    # ------------------------------------------------------------------ chronological part
    def _chronological(self, token, predictions, elapsed_h, current_rgb, expected):
        """Everything in apply_with_reference that reads or writes history, in the frozen order."""
        sequence, module = self.sequence, self.module
        started = time.perf_counter()
        if sequence.baseline is None:
            raise ValueError("Bind the decision reference first")
        moment = float(elapsed_h)
        if not np.isfinite(moment) or (sequence.previous_time is not None and moment <= sequence.previous_time):
            raise ValueError("Decision observations require strictly increasing times; fork reference histories")
        if hasattr(sequence.provider, "apply_with_reference"):
            evidence, report = sequence.provider.apply_with_reference(predictions, moment, current_rgb)
        else:
            evidence, report = sequence.provider.apply(predictions, moment)
        domain = sequence.context.domain
        before = module.completion.delivered_labels(evidence, domain)
        negative = bool(report.get("pre_injection_negative_control"))
        spatial = sequence.provider.last_diagnostics.get("spatial_only_presence")
        if spatial is not None:
            spatial = np.array(spatial, copy=True)          # the provider owns its diagnostics
        first = None
        if self.kind == LOCAL_AGE:
            # observe() returns the history's own array, which the next photograph updates in place.
            first = np.array(sequence.exposure_history.observe(before, domain, moment, pre_injection=negative),
                             copy=True)
        sequence.previous_time = moment
        seconds = dict(chronological_s=time.perf_counter() - started)
        return _Frame(token, predictions, moment, current_rgb, before, negative, report, spatial, first,
                      expected, seconds)

    # ------------------------------------------------------------------ pure part
    def _decide(self, frame: _Frame, *, claim_slot=True) -> Decided:
        """Completion and final decision: the rest of apply_with_reference, no history involved."""
        sequence, module = self.sequence, self.module
        queued = time.perf_counter()
        with self.slot() if claim_slot else nullcontext():
            started = time.perf_counter()
            core, completion, cv2 = module.core, module.completion, module.cv2
            context = sequence.context
            domain = context.domain
            spacing = (abs(float(context.y_m[1] - context.y_m[0])), abs(float(context.x_m[1] - context.x_m[0])))
            before = frame.before
            q = np.zeros_like(domain, dtype=np.float32)
            if not frame.negative:
                x, _ = completion.prepare_inputs(frame.current_rgb, sequence.baseline,
                    frame.predictions["co2_presence_probability"], before, domain, spacing)
                if not isinstance(sequence._predictor, _LockedPredictor):
                    with self._lock:
                        if sequence._predictor is None:
                            sequence._predictor = completion._Predictor(sequence.checkpoint)
                    self._share_predictor(create=False)
                q = cv2.resize(sequence._predictor(x), domain.shape[::-1], interpolation=cv2.INTER_LINEAR)
            extra = {} if self.kind == LEGACY else dict(first_observed_h=frame.first)
            result = core.decide(core.Evidence(frame.current_rgb, sequence.baseline, before, q,
                (before > 0) & domain, domain, context.facies, spacing, frame.time, frame.negative,
                dict(provider="frozen-causal-observations-before-completion",
                     completion_checkpoint_sha256=sequence.checkpoint["checkpoint_sha256"],
                     completion_and_final_result_do_not_update_provider_history=True),
                independent_spatial_observation=frame.spatial, **extra), sequence.settings)
            labels = result.labels
            co2 = (labels > 0).astype(np.float32)
            gas = (labels == 2).astype(np.float32)
            output = {**frame.predictions, "co2_presence_probability": co2,
                      "gas_probability_conditional_on_co2": gas,
                      "aqueous_co2_probability": (labels == 1).astype(np.float32),
                      "free_co2_gas_probability": gas, "untouched_water_probability": 1 - co2}
            diagnostics = dict(added_aqueous=(labels == 1) & (before == 0),
                suppressed_presence=(before > 0) & (labels == 0), unresolved=np.zeros_like(domain),
                decision_before_co2=before > 0, decision_before_gas=before == 2)
            finished = time.perf_counter()
        decided = Decided(frame.token, output, {**frame.report, module.PROFILE_KEY: result.report},
                          diagnostics, result, dict(frame.seconds, decision_queued_s=started - queued,
                                                    decision_s=finished - started, verified=False))
        if frame.expected is not None:
            found = differences(frame.expected, decided)
            decided.seconds["verified"] = True
            if found:
                message = "Pipelined decision differs from the frozen method in: " + ", ".join(found)
                if not frame.recoverable:
                    raise RuntimeError(message)
                # Nothing unchecked has been delivered: keep the frozen result and the frozen call.
                self.disabled_reason = message
                output, report, diagnostics, result = frame.expected
                return Decided(frame.token, output, report, diagnostics, result,
                               dict(decided.seconds, kept_frozen_result=True))
        return decided

    # ------------------------------------------------------------------ driving
    def _frozen(self, token, predictions, elapsed_h, current_rgb) -> Decided:
        started = time.perf_counter()
        output, report = self.sequence.apply_with_reference(predictions, elapsed_h, current_rgb)
        self.frozen_frames += 1
        return Decided(token, output, report, self.sequence.last_diagnostics, self.sequence.last_result,
                       dict(frozen_s=time.perf_counter() - started, verified=False))

    def _expected(self, predictions, elapsed_h, current_rgb):
        started = time.perf_counter()
        fork = self._fork(self.sequence)
        output, report = fork.apply_with_reference(predictions, elapsed_h, current_rgb)
        self.verification_s += time.perf_counter() - started
        return output, report, fork.last_diagnostics, fork.last_result

    def _wait(self, future: Future):
        started = time.perf_counter()
        try:
            return future.result()
        finally:
            self.waited_s += time.perf_counter() - started

    def wait_for_capacity(self) -> None:
        """Block (outside any CPU slot) until fewer than ``limit`` decisions are unfinished."""
        while sum(not future.done() for future in self._pending) >= self.limit:
            unfinished = next((future for future in self._pending if not future.done()), None)
            if unfinished is not None:
                self._wait(unfinished)

    def submit(self, predictions, elapsed_h, current_rgb, token=None) -> None:
        """Chronological part now, on this thread; the decision follows on a worker thread."""
        for future in self._pending:
            if future.done() and future.exception() is not None:
                future.result()
        if self._gate is not None:
            # A checked decision on a worker thread whose difference could still fall back: wait for
            # its verdict before anything unchecked follows it.
            gate, self._gate = self._gate, None
            self._wait(gate)
        if self.disabled_reason is not None:
            future: Future = Future()
            future.set_result(self._frozen(token, predictions, elapsed_h, current_rgb))
            self._pending.append(future)
            self.submitted += 1
            return
        # Phase 1, one by one: nothing is handed to a worker until the restated body has agreed
        # with the frozen method on this machine. Phase 2: the first overlapped photographs are
        # checked too, then every verify_every-th.
        one_by_one = self.verified_decisions < self.verify_first_decisions
        check = (one_by_one or self.verified_overlapped < self.verify_first_overlapped
                 or (self.verify_every > 0 and self.submitted > 0 and self.submitted % self.verify_every == 0))
        expected = self._expected(predictions, elapsed_h, current_rgb) if check else None
        overlapping = any(not future.done() for future in self._pending)
        frame = self._chronological(token, predictions, elapsed_h, current_rgb, expected)
        if overlapping:
            self.overlapped_s += frame.seconds["chronological_s"]
        self.chronological_s += frame.seconds["chronological_s"]
        self.submitted += 1
        if check:
            self.verified += 1
            self.verified_decisions += int(one_by_one and not frame.negative)
            self.verified_overlapped += int(not one_by_one)
            frame.recoverable = self.unchecked_submitted == 0
        else:
            self.unchecked_submitted += 1
        if not one_by_one:
            future = self._pool.submit(self._decide, frame)
            self._pending.append(future)
            if check and frame.recoverable:
                self._gate = future
            return
        # One by one, on this thread and inside the caller's CPU slot (no second claim).
        future = Future()
        future.set_result(self._decide(frame, claim_slot=False))
        self._pending.append(future)

    def ready(self) -> list[Decided]:
        """Finished photographs in submission order, up to the first unfinished one."""
        finished = []
        while self._pending and self._pending[0].done():
            finished.append(self._take())
        return finished

    def drain(self) -> list[Decided]:
        finished = []
        while self._pending:
            self._wait(self._pending[0])
            finished.append(self._take())
        return finished

    def _take(self) -> Decided:
        decided = self._pending.popleft().result()
        self.decision_s += decided.seconds.get("decision_s", 0.0)
        self._last = decided
        return decided

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._last is not None and not self._pending and self.frozen_frames == 0:
            # Leave the sequence as the frozen method leaves it after its last photograph.
            self.sequence.last_result = self._last.result
            self.sequence.last_diagnostics = self._last.diagnostics

    def report(self) -> dict:
        return dict(execution=VERSION, runtime=self.kind, active=self.disabled_reason is None,
            disabled_reason=self.disabled_reason, decision_threads=self.threads,
            photographs=self.submitted, photographs_by_frozen_call=self.frozen_frames,
            checked_against_frozen_method=self.verified,
            chronological_s=self.chronological_s, decision_s=self.decision_s,
            decision_overlapped_with_next_photograph_s=self.overlapped_s,
            verification_s=self.verification_s, loop_waited_for_decisions_s=self.waited_s,
            results_identical_to_frozen_method=True)


def create(sequence, *, slot=None, **options):
    """A pipelined executor for this sequence, or None when the frozen call must be kept."""
    if sequence is None or unsupported_reason(sequence) is not None:
        return None
    try:
        return PipelinedDecision(sequence, slot=slot, **options)
    except Exception:                       # e.g. the fork for the self-check cannot be imported here
        return None
