"""Lossless, bounded shared float32 inputs for proposal helper processes.

The parent owns a mapping until the underlying worker future completes. Public
future cancellation does not release it early. Workers close their attachments
in finally; the existing proposal parent-death watcher exits orphaned helpers,
so Windows also releases their mappings after an abrupt parent exit.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import shared_memory
import os
import pickle
import sys
import threading
import time

import numpy as np


VERSION = "lossless-shared-float32-proposal-inputs-v1"
FIELDS = ("untouched_water_probability", "aqueous_co2_probability", "free_co2_gas_probability")


def _execute(function, payload, options, shared):
    attached = time.perf_counter()
    handle = views = fields = None
    try:
        if shared:
            kwargs = {"track": False} if sys.version_info >= (3, 13) else {}
            handle = shared_memory.SharedMemory(name=payload["name"], create=False, **kwargs)
            shape = tuple(payload["shape"])
            if len(shape) != 2 or min(shape) < 1 or int(np.prod(shape)) * 12 > handle.size:
                raise ValueError("Invalid shared proposal geometry")
            views = np.ndarray((3, *shape), dtype=np.float32, buffer=handle.buf)
            views.flags.writeable = False
            fields = {name: views[index] for index, name in enumerate(FIELDS)}
        else:
            fields = payload
        started = time.perf_counter()
        result = function(fields, options)
        finished = time.perf_counter()
        return result, dict(attach_s=started-attached, compute_s=finished-started)
    finally:
        fields = views = None
        if handle is not None:
            handle.close()


@dataclass
class Submitted:
    raw: Future
    result: Future


@dataclass
class _Owner:
    handle: object
    size: int
    future: Future | None = None


class ProposalTransport:
    """The ordered proposal-pool caller submits; callbacks only release storage.

    Like _ProposalPool, submit is a single-caller operation. max_slots therefore
    bounds allocations without adding another producer queue or native team.
    """
    def __init__(self, max_slots, *, enabled=None, minimum_bytes=1024**2):
        if int(max_slots) < 1:
            raise ValueError("Proposal transport requires at least one bounded slot")
        if enabled is None:
            default = "1" if os.name == "nt" else "0"
            enabled = os.environ.get("PHASE_PROPOSAL_SHARED_MEMORY", default).lower() not in ("0", "false", "no", "off")
        self.enabled = bool(enabled)
        self.max_slots, self.minimum_bytes = int(max_slots), int(minimum_bytes)
        self._lock = threading.RLock()
        self._owners = {}
        self._closed = False
        self._counts = Counter(shared_tasks=0, pickle_tasks=0, input_bytes=0,
                               shared_bytes=0, descriptor_bytes=0, pickle_input_bytes=0,
                               live_shared_bytes=0, peak_shared_bytes=0, peak_shared_slots=0,
                               worker_failures=0, cleanup_failures=0)
        self._timings = Counter(copy_s=0., attach_s=0., compute_s=0.)
        self._fallbacks = Counter()

    def _release(self, name):
        if name is None:
            return
        with self._lock:
            owner = self._owners.get(name)
            if owner is None:
                return
            try:
                owner.handle.close()
                try:
                    owner.handle.unlink()
                except FileNotFoundError:
                    pass
            except Exception:
                self._counts["cleanup_failures"] += 1
                return  # Keep ownership for a later close/reap retry.
            self._owners.pop(name)
            self._counts["live_shared_bytes"] -= owner.size

    def _reap(self):
        with self._lock:
            names = [name for name, owner in self._owners.items()
                     if owner.future is not None and owner.future.done()]
        for name in names:
            self._release(name)

    def _pack(self, fields):
        size = sum(value.nbytes for value in fields.values() if isinstance(value, np.ndarray))
        reason = None
        arrays = [fields.get(name) for name in FIELDS]
        if not self.enabled:
            reason = "disabled"
        elif set(fields) != set(FIELDS) or not all(isinstance(value, np.ndarray) for value in arrays):
            reason = "unsupported_fields"
        elif not all(value.dtype == np.dtype(np.float32) and value.ndim == 2 for value in arrays):
            reason = "unsupported_dtype_or_dimensions"
        elif min(arrays[0].shape) < 1 or any(value.shape != arrays[0].shape for value in arrays):
            reason = "unsupported_shape"
        elif size < self.minimum_bytes:
            reason = "below_minimum_size"
        self._reap()
        with self._lock:
            if reason is None and len(self._owners) >= self.max_slots:
                reason = "shared_slot_limit"
        handle = view = None
        copied = 0.
        if reason is None:
            started = time.perf_counter()
            try:
                handle = shared_memory.SharedMemory(create=True, size=size)
                view = np.ndarray((3, *arrays[0].shape), dtype=np.float32, buffer=handle.buf)
                for index, value in enumerate(arrays):
                    np.copyto(view[index], value, casting="no")
                copied = time.perf_counter()-started
                view = None
                descriptor = dict(name=handle.name, shape=arrays[0].shape)
                with self._lock:
                    self._owners[handle.name] = _Owner(handle, size)
                    self._counts.update(shared_tasks=1, input_bytes=size, shared_bytes=size,
                                        descriptor_bytes=len(pickle.dumps(descriptor, protocol=5)), live_shared_bytes=size)
                    self._counts["peak_shared_bytes"] = max(self._counts["peak_shared_bytes"], self._counts["live_shared_bytes"])
                    self._counts["peak_shared_slots"] = max(self._counts["peak_shared_slots"], len(self._owners))
                    self._timings["copy_s"] += copied
                return descriptor, handle.name
            except (OSError, MemoryError, BufferError):
                reason = "shared_allocation_failed"
                view = None
                if handle is not None:
                    try:
                        handle.close()
                    finally:
                        handle.unlink()
        with self._lock:
            self._fallbacks[reason] += 1
            self._counts.update(pickle_tasks=1, input_bytes=size, pickle_input_bytes=size)
        return dict(fields), None

    def submit(self, executor, function, fields, options):
        if self._closed:
            raise RuntimeError("Proposal transport is closed")
        payload, name = self._pack(fields)
        public = Future()
        try:
            raw = executor.submit(_execute, function, payload, dict(options), name is not None)
        except BaseException:
            self._release(name)
            raise
        if name is not None:
            with self._lock:
                self._owners[name].future = raw
        def complete(task):
            result = error = None
            cancelled = task.cancelled()
            try:
                if not cancelled:
                    result, timings = task.result()
                    with self._lock:
                        self._timings.update(timings)
            except BaseException as caught:
                error = caught
                with self._lock:
                    self._counts["worker_failures"] += 1
            finally:
                self._release(name)
            if cancelled:
                public.cancel()
            elif public.set_running_or_notify_cancel():
                if error is not None:
                    public.set_exception(error)
                else:
                    public.set_result(result)
        raw.add_done_callback(complete)
        # Capacity must follow raw, never the independently cancellable public
        # future: cancelling the latter cannot release an active worker's input.
        return Submitted(raw=raw, result=public)

    def close(self):
        """Call after the executor joins, or after its helpers are terminated."""
        self._closed = True
        with self._lock:
            names = list(self._owners)
        for name in names:
            self._release(name)

    def report(self):
        with self._lock:
            return dict(method=VERSION, enabled=self.enabled, max_shared_slots=self.max_slots,
                        minimum_bytes=self.minimum_bytes, **self._counts, **self._timings,
                        fallback_reasons=dict(self._fallbacks), float32_values_unchanged=True)
