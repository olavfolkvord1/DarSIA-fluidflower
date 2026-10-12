"""Boundary observations for native production; no numerical function changes.

``plain`` calls the original with no hooks. ``profile`` records call/return
boundaries and array metadata, never hashes array payloads. ``memory-audit`` is
a separate untimed run which may hash bounded chunks. No GPU module or device
context is created here. Existing native functions, globals and arrays remain
unchanged. Callbacks have measurable overhead; C temporaries between boundaries
and arrays outside explicitly inspected roots are not a complete heap census.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import itertools
import os
from pathlib import Path
import struct
import sys
import threading
import time
from types import CodeType, ModuleType
import weakref

ARRAY_THRESHOLD_BYTES = 10_000_000
CHUNK_BYTES = 1024**2


def _process_facts():
    """Read this process only. The OS peak is cumulative, not a stage peak."""
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        peak = getattr(memory, "peak_wset", None)
        if peak is None and os.name != "nt":
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak *= 1 if sys.platform == "darwin" else 1024
        return dict(available=True, pid=process.pid,
                    process_create_time=str(process.create_time()), rss_bytes=memory.rss,
                    os_peak_rss_bytes=peak, os_peak_scope="process_lifetime")
    except Exception as error:
        return dict(available=False, error=type(error).__name__)


def _threadpool_facts():
    try:
        from threadpoolctl import threadpool_info
        return dict(available=True, pools=threadpool_info(), settings_changed=False)
    except Exception as error:
        return dict(available=False, pools=[], error=type(error).__name__, settings_changed=False)


def _code(value):
    if isinstance(value, CodeType):
        return value
    if inspect.ismethod(value):
        value = value.__func__
    value = inspect.unwrap(value)
    code = getattr(value, "__code__", None)
    if not isinstance(code, CodeType):
        raise TypeError("Stage filters need an exact Python code object or function")
    return code


def _is_array(value):
    # Observation must not import numpy before the native operator does.
    numpy = sys.modules.get("numpy")
    return numpy is not None and isinstance(value, numpy.ndarray)


def _array_digest(array, *, chunk_bytes=CHUNK_BYTES):
    """Hash logical C-order bytes with bounded readonly iterator buffering."""
    import numpy as np
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= CHUNK_BYTES:
        raise ValueError("Digest chunks must be positive and at most 1 MiB")
    if array.dtype.hasobject or array.dtype.itemsize > chunk_bytes:
        raise ValueError("Array digest requires a fixed, nonobject bounded dtype")
    digest = hashlib.sha256()
    size = max(1, chunk_bytes // max(1, array.dtype.itemsize))
    with np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=["readonly"], order="C", buffersize=size) as iterator:
        for values in iterator:
            # Unbuffered contiguous loops can exceed buffersize; slice each one.
            for first in range(0, values.size, size):
                digest.update(values[first:first + size].tobytes(order="C"))
    return digest.hexdigest()


def _float_value_digest(array, *, chunk_bytes=CHUNK_BYTES):
    """Diagnostic lossless float32/64 promotion, never a product identity."""
    import numpy as np
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        return None
    if type(chunk_bytes) is not int or not 8 <= chunk_bytes <= CHUNK_BYTES:
        raise ValueError("Float audit chunks must be 8 bytes to 1 MiB")
    size = chunk_bytes // 8
    digest = hashlib.sha256()
    with np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=["readonly"], op_dtypes=[np.dtype("float64")],
                   casting="safe", order="C", buffersize=size) as iterator:
        for values in iterator:
            for first in range(0, values.size, size):
                digest.update(values[first:first + size].tobytes(order="C"))
    return digest.hexdigest()


def _array_metadata(array):
    pointer = int(array.__array_interface__["data"][0])
    low, high = 0, 0
    for length, stride in zip(array.shape, array.strides):
        extent = (length - 1) * stride if length else 0
        low += min(0, extent)
        high += max(0, extent)
    owner = array
    seen = set()
    while _is_array(getattr(owner, "base", None)) and id(owner) not in seen:
        seen.add(id(owner))
        owner = owner.base
    external = getattr(owner, "base", None)
    storage = owner if external is None else external
    owned = bool(owner.flags.owndata)
    return dict(shape=list(array.shape), dtype=array.dtype.str, dtype_descriptor=repr(array.dtype.descr), nbytes=int(array.nbytes),
                strides=list(array.strides), c_contiguous=bool(array.flags.c_contiguous),
                f_contiguous=bool(array.flags.f_contiguous), writeable=bool(array.flags.writeable),
                data_pointer=pointer, byte_span=[pointer + low, pointer + high + array.dtype.itemsize]
                if array.size else [pointer, pointer],
                storage_owner_id=id(storage), owned_storage_bytes=int(owner.nbytes) if owned else None,
                storage_may_be_external=external is not None,
                overlap_rule="byte_spans_are_conservative_for_strided_views")


class BoundaryObserver:
    """Call originals once and optionally observe explicit code boundaries.

    ``code_stages`` keys are exact code identities, including any call-local
    FunctionType clone's shared code object. No name/filename matching grants a
    stage. ``root_provider`` should return only the known live caches/globals.
    ``gpu_facts`` is an explicit read-only provider for an already active native
    backend; the default reports no GPU observations rather than probing one.
    """

    def __init__(self, mode="plain", *, code_stages=None, root_provider=None,
                 process_facts=None, threadpool_facts=None, gpu_facts=None,
                 requested_threads=None,
                 sample_interval=.1, array_threshold_bytes=ARRAY_THRESHOLD_BYTES,
                 max_objects=10_000, max_arrays=1_000, max_depth=8,
                 max_spans=10_000, max_hash_bytes=1024**3, chunk_bytes=CHUNK_BYTES):
        if mode not in {"plain", "profile", "memory-audit"}:
            raise ValueError("Unknown boundary observation mode")
        if not 0 < sample_interval <= 10:
            raise ValueError("Positive bounded RSS sampling interval required")
        if requested_threads is not None and (type(requested_threads) is not int or requested_threads <= 0):
            raise ValueError("Requested threads must be an explicit positive integer")
        limits = (array_threshold_bytes, max_objects, max_arrays, max_depth, max_spans, max_hash_bytes, chunk_bytes)
        if any(type(value) is not int or value <= 0 for value in limits) or not 8 <= chunk_bytes <= CHUNK_BYTES:
            raise ValueError("Inventory/hash limits must be positive and chunks at most 1 MiB")
        self.mode = mode
        self.code_stages = {}
        for value, stage in (code_stages or {}).items():
            code = _code(value)
            self.code_stages[id(code)] = (code, stage)
        if any(not isinstance(stage, str) or not stage for code, stage in self.code_stages.values()):
            raise ValueError("Stages need explicit names")
        self.root_provider = root_provider
        self.process_facts = process_facts or _process_facts
        self.threadpool_facts = threadpool_facts or _threadpool_facts
        self.gpu_facts = gpu_facts
        self.requested_threads = requested_threads
        self.sample_interval = sample_interval
        self.threshold = array_threshold_bytes
        self.max_objects, self.max_arrays, self.max_depth = max_objects, max_arrays, max_depth
        self.max_spans, self.max_hash_bytes, self.chunk_bytes = max_spans, max_hash_bytes, chunk_bytes
        self.spans, self.snapshots, self.errors = [], [], []
        self._arrays, self._active_codes, self._stack = {}, {}, []
        self._array_index = self._span_index = self._error_count = 0
        self._hash_bytes = self._callback_wall_ns = self._sample_count = 0
        self._sample_peak_rss = self._os_peak_rss = None
        self._active = False
        self._original_failed = False
        self._previous = None
        self._thread = None
        self._stop = threading.Event()
        self._wall_ns = self._cpu_ns = 0
        self.limits_hit = set()
        self.threadpools_before = self.threadpools_after = None

    def _error(self, error):
        self._error_count += 1
        if len(self.errors) < 16:
            self.errors.append(dict(type=type(error).__name__, detail=str(error)[:300]))

    def _facts(self):
        try:
            facts = self.process_facts()
            rss, peak = facts.get("rss_bytes"), facts.get("os_peak_rss_bytes")
            if type(rss) is int:
                self._sample_peak_rss = rss if self._sample_peak_rss is None else max(rss, self._sample_peak_rss)
            if type(peak) is int:
                self._os_peak_rss = peak if self._os_peak_rss is None else max(peak, self._os_peak_rss)
            return facts
        except Exception as error:
            self._error(error)
            return dict(available=False, error=type(error).__name__)

    def _sample(self):
        while not self._stop.wait(self.sample_interval):
            self._facts()
            self._sample_count += 1

    def __enter__(self):
        if self._active:
            raise RuntimeError("An observer cannot be reentered")
        self._active = True
        self._start_wall, self._start_cpu = time.perf_counter_ns(), time.process_time_ns()
        if self.mode == "plain":
            return self
        self._previous = sys.getprofile()
        try:
            self.threadpools_before = self.threadpool_facts()
            self._facts()
            if self.code_stages:
                sys.setprofile(self._profile)
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample, name="production-observation", daemon=True)
            self._thread.start()
            return self
        except BaseException:
            sys.setprofile(self._previous)
            self._stop.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join()
            self._active = False
            raise

    def __exit__(self, kind, value, traceback):
        # Measure the entire invocation, including callback costs. Report
        # bookkeeping and sampler shutdown are outside this whole-run interval.
        self._wall_ns += time.perf_counter_ns() - self._start_wall
        self._cpu_ns += time.process_time_ns() - self._start_cpu
        self._original_failed |= kind is not None
        try:
            if self.mode != "plain":
                sys.setprofile(self._previous)
                self._stop.set()
                if self._thread is not None:
                    self._thread.join()
                try:
                    self.threadpools_after = self.threadpool_facts()
                except Exception as error:
                    self._error(error)
                self._facts()
        finally:
            self._active = False
            self._active_codes.clear()
            self._stack.clear()
        return False

    def _begin(self, stage, origin):
        if len(self.spans) >= self.max_spans:
            self.limits_hit.add("spans")
            return None
        self._span_index += 1
        span = dict(span_id=self._span_index, parent_span_id=self._stack[-1]["span_id"] if self._stack else None,
                    stage=stage, origin=origin, timing="inclusive", before=self._facts(),
                    start_wall_ns=time.perf_counter_ns(), start_cpu_ns=time.process_time_ns())
        if self.gpu_facts is not None:
            try:
                span["gpu_before"] = self.gpu_facts(stage, "before")
            except Exception as error:
                self._error(error)
        self.spans.append(span)
        self._stack.append(span)
        return span

    def _end(self, span):
        if span is None:
            return
        span["wall_ns"] = time.perf_counter_ns() - span.pop("start_wall_ns")
        span["cpu_ns"] = time.process_time_ns() - span.pop("start_cpu_ns")
        span["after"] = self._facts()
        if self.gpu_facts is not None:
            try:
                span["gpu"] = self.gpu_facts(span["stage"], "after")
            except Exception as error:
                self._error(error)
        else:
            span["gpu"] = dict(observed=False, elapsed_ns=None, vram_bytes=None,
                               reason="no_explicit_already_active_backend_provider")
        if self._stack and self._stack[-1] is span:
            self._stack.pop()
        else:
            self._error(RuntimeError("Observed span stack is inconsistent"))

    def call(self, stage, original, *args, **kwargs):
        if not self._active:
            with self:
                return self.call(stage, original, *args, **kwargs)
        if self.mode == "plain":
            return original(*args, **kwargs)
        span = self._begin(stage, "operator")
        self._safe_snapshot(stage + ":before", dict(args=args, kwargs=kwargs))
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            self._original_failed = True
            if span is not None:
                span["original_exception"] = type(error).__name__
            raise
        else:
            self._safe_snapshot(stage + ":after", dict(result=result))
            return result
        finally:
            self._end(span)

    def _profile(self, frame, event, argument):
        started = time.perf_counter_ns()
        try:
            selected = self.code_stages.get(id(frame.f_code))
            if selected is not None and selected[0] is frame.f_code:
                stage = selected[1]
                key = id(frame)
                if event == "call":
                    self._active_codes[key] = self._begin(stage, "exact_code")
                    self._safe_snapshot(stage + ":call", frame.f_locals)
                elif event == "return" and key in self._active_codes:
                    self._safe_snapshot(stage + ":return", dict(locals=frame.f_locals, result=argument))
                    self._end(self._active_codes.pop(key))
        except Exception as error:
            self._error(error)
        finally:
            self._callback_wall_ns += time.perf_counter_ns() - started
        # A pre-existing profiler's exception must retain its original effect.
        if self._previous is not None:
            self._previous(frame, event, argument)

    def _safe_snapshot(self, stage, roots):
        try:
            if self.root_provider is not None:
                roots = dict(boundary=roots, known_roots=self.root_provider())
            self.snapshot(stage, roots)
        except Exception as error:
            self._error(error)

    def snapshot(self, stage, roots):
        """Bounded readonly metadata. Hashing is exclusive to memory-audit."""
        if self.mode == "plain":
            return
        visited, present = set(), set()
        pending = [("roots", roots, 0)]
        while pending:
            name, value, depth = pending.pop()
            if id(value) in visited:
                continue
            if len(visited) >= self.max_objects:
                self.limits_hit.add("objects")
                break
            visited.add(id(value))
            if _is_array(value):
                if value.nbytes > self.threshold:
                    uid = self._observe_array(value, stage, name)
                    if uid is not None:
                        present.add(uid)
                continue
            if depth >= self.max_depth:
                if isinstance(value, (Mapping, list, tuple)):
                    self.limits_hit.add("depth")
                continue
            if isinstance(value, Mapping):
                members = value.items()
            elif isinstance(value, (list, tuple)):
                members = enumerate(value)
            elif inspect.isfunction(value) or inspect.isclass(value):
                continue
            else:
                try:
                    members = object.__getattribute__(value, "__dict__").items()
                except (AttributeError, TypeError):
                    members = []
                slot_members = []
                for cls in type(value).__mro__:
                    slots = cls.__dict__.get("__slots__", ())
                    slots = (slots,) if isinstance(slots, str) else slots
                    for slot in slots:
                        if slot.startswith("__"):
                            continue
                        descriptor = cls.__dict__.get(slot)
                        if inspect.ismemberdescriptor(descriptor):
                            try:
                                if len(slot_members) >= self.max_objects:
                                    self.limits_hit.add("objects")
                                    break
                                slot_members.append((slot, descriptor.__get__(value, type(value))))
                            except AttributeError:
                                pass
                members = itertools.chain(members, slot_members)
            for key, member in members:
                # Imported modules and functions are not a whole-process walk.
                if isinstance(member, ModuleType) or callable(member):
                    continue
                if len(pending) >= self.max_objects:
                    self.limits_hit.add("objects")
                    break
                pending.append((name + "." + str(key), member, depth + 1))
        duplicates, cross_dtype, aliases = [], [], []
        owners = {}
        for uid in present:
            owners.setdefault(self._arrays[uid]["storage_owner_id"], []).append(uid)
        for owner, group in owners.items():
            if len(group) > 1:
                spans = sorted(self._arrays[uid]["byte_span"] for uid in group)
                overlap = any(right[0] < left[1] for left, right in zip(spans, spans[1:]))
                aliases.append(dict(storage_owner_id=owner, array_ids=sorted(group),
                                    shared_base_owner=True, any_byte_span_overlap=overlap,
                                    exact_strided_overlap_checked=False))
        if self.mode == "memory-audit":
            grouped = {}
            for uid in present:
                row = self._arrays[uid]
                if row.get("sha256") is not None:
                    grouped.setdefault((tuple(row["shape"]), row["dtype_descriptor"], row["sha256"]), []).append(uid)
            for group in grouped.values():
                if len(group) > 1:
                    duplicates.append(dict(array_ids=sorted(group), content="exact_dtype_shape_C_order_bytes",
                                           storage_owner_ids=sorted({self._arrays[u]["storage_owner_id"] for u in group})))
            grouped = {}
            for uid in present:
                row = self._arrays[uid]
                if row.get("float64_value_sha256") is not None:
                    grouped.setdefault((tuple(row["shape"]), row["float64_value_sha256"]), []).append(uid)
            for group in grouped.values():
                if len({self._arrays[uid]["dtype"] for uid in group}) > 1:
                    cross_dtype.append(dict(array_ids=sorted(group),
                                           content="lossless_float32_to_float64_diagnostic_values",
                                           exact_product_bit_identity=False))
        if len(self.snapshots) < self.max_spans * 2:
            self.snapshots.append(dict(stage=stage, array_ids=sorted(present), byte_duplicates=duplicates,
                                       cross_dtype_float_duplicates=cross_dtype, storage_aliases=aliases))
        else:
            self.limits_hit.add("snapshots")

    def _observe_array(self, array, stage, path):
        for uid, row in self._arrays.items():
            if row["_reference"]() is array:
                break
        else:
            if len(self._arrays) >= self.max_arrays:
                self.limits_hit.add("arrays")
                return None
            self._array_index += 1
            uid = self._array_index
            owner = weakref.ref(self)
            def collected(reference):
                observer = owner()
                if observer is not None:
                    observer._arrays[uid]["collection_observed"] = True
            row = dict(array_id=uid, first_seen_stage=stage, collection_observed=False,
                       allocation_step=None, lifetime_is_exact=False, observed_paths=[])
            row["_reference"] = weakref.ref(array, collected)
            self._arrays[uid] = row
        row.update(_array_metadata(array), last_seen_stage=stage)
        if path not in row["observed_paths"] and len(row["observed_paths"]) < 16:
            row["observed_paths"].append(path)
        if self.mode == "memory-audit":
            row.pop("sha256", None)
            row.pop("float64_value_sha256", None)
            if array.nbytes > self.max_hash_bytes - self._hash_bytes:
                row["digest_status"] = "skipped_total_byte_budget"
                self.limits_hit.add("hash_bytes")
            elif array.dtype.hasobject:
                row["digest_status"] = "object_dtype_not_hashed"
            else:
                row["sha256"] = _array_digest(array, chunk_bytes=self.chunk_bytes)
                row["digest_status"] = "readonly_bounded_chunks"
                self._hash_bytes += int(array.nbytes)
                if array.dtype.kind == "f" and array.dtype.itemsize in (4, 8):
                    canonical_bytes = int(array.size) * 8
                    if canonical_bytes <= self.max_hash_bytes - self._hash_bytes:
                        row["float64_value_sha256"] = _float_value_digest(array, chunk_bytes=self.chunk_bytes)
                        self._hash_bytes += canonical_bytes
                    else:
                        row["float_value_digest_status"] = "skipped_total_byte_budget"
                        self.limits_hit.add("hash_bytes")
        return uid

    def report(self):
        arrays = [{key: value for key, value in row.items() if not key.startswith("_")}
                  for row in self._arrays.values()]
        return dict(format="darsia-production-boundary-profile-v1", mode=self.mode,
                    complete=not self.errors and not self.limits_hit and not self._original_failed, errors=list(self.errors),
                    error_count=self._error_count, wall_ns=self._wall_ns, cpu_ns=self._cpu_ns,
                    spans=self.spans, arrays=arrays, snapshots=self.snapshots,
                    array_threshold_bytes=self.threshold, audit_hash_bytes=self._hash_bytes,
                    sampled_peak_rss_bytes=self._sample_peak_rss, os_peak_rss_bytes=self._os_peak_rss,
                    os_peak_scope="process_lifetime", rss_sample_count=self._sample_count,
                    callback_wall_ns=self._callback_wall_ns,
                    threadpools_before=self.threadpools_before, threadpools_after=self.threadpools_after,
                    requested_threads=self.requested_threads,
                    limits_hit=sorted(self.limits_hit), original_call_failed=self._original_failed,
                    stage_filter="exact_code_object_identity", scientific_eligibility=False,
                    limitations=["Python call/return callbacks and metadata scans add overhead",
                                 "Process CPU includes observer callbacks and the RSS sampler thread",
                                 "Nested spans are inclusive and must not be summed",
                                 "Only this Python thread and explicit live roots are inspected",
                                 "C temporaries and between-boundary arrays can be unseen",
                                 "First/last seen and collection are not allocation/last-use times",
                                 "RSS sampling can miss peaks; OS peak is cumulative",
                                 "Byte spans overestimate strided overlap; external storage can be unknown",
                                 "Byte duplicates require exact dtype; float32/64 value duplicates are diagnostic only"])


def numeric_product_fingerprints(products, *, chunk_bytes=CHUNK_BYTES):
    """Untimed exact products, excluding caller-owned observation receipts."""
    result = {}
    def visit(value, path):
        if _is_array(value):
            result[path] = dict(kind="array", dtype=value.dtype.str, dtype_descriptor=repr(value.dtype.descr), shape=list(value.shape),
                                sha256=_array_digest(value, chunk_bytes=chunk_bytes))
        elif isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Product keys must be strings")
            result[path] = dict(kind="mapping", keys=sorted(value))
            for key in sorted(value):
                visit(value[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, (list, tuple)):
            result[path] = dict(kind=type(value).__name__, length=len(value))
            for index, item in enumerate(value):
                visit(item, path + "/" + str(index))
        elif type(value) is float:
            result[path] = dict(kind="python_float64", bits=struct.pack("!d", value).hex())
        elif type(value) in (bool, int, str) or value is None:
            result[path] = dict(kind=type(value).__name__, value=value)
        elif hasattr(value, "dtype") and type(value).__module__.startswith("numpy"):
            if value.dtype.hasobject:
                raise ValueError("Object scalars cannot be scientific products")
            result[path] = dict(kind="numpy_scalar", dtype=value.dtype.str, bits=value.tobytes().hex())
        else:
            raise ValueError("Unsupported numeric product; no implicit conversion permitted")
    visit(products, "")
    return result


def compare_numeric_products(reference, candidate, *, plain_wall_ns=None, profile_wall_ns=None,
                             chunk_bytes=CHUNK_BYTES):
    first = numeric_product_fingerprints(reference, chunk_bytes=chunk_bytes)
    second = numeric_product_fingerprints(candidate, chunk_bytes=chunk_bytes)
    different = sorted(path for path in first.keys() | second.keys() if first.get(path) != second.get(path))
    if (plain_wall_ns is None) != (profile_wall_ns is None):
        raise ValueError("Both elapsed durations are required to report overhead")
    if plain_wall_ns is not None and (type(plain_wall_ns) is not int or type(profile_wall_ns) is not int
                                     or plain_wall_ns <= 0 or profile_wall_ns < 0):
        raise ValueError("Elapsed durations must be integer nanoseconds")
    return dict(format="darsia-production-profile-product-identity-v1", complete=True,
                exact_numeric_products=not different, different_paths=different,
                reference=first, candidate=second,
                overhead_ratio=None if plain_wall_ns is None else profile_wall_ns / plain_wall_ns,
                overhead_wall_ns=None if plain_wall_ns is None else profile_wall_ns - plain_wall_ns,
                input_setting_identity_checked=False, scientific_eligibility=False)
