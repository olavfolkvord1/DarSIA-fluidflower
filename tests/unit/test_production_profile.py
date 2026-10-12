"""Generated numeric fixtures/fake observers only; no experimental files."""
import ast
from functools import wraps
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import FunctionType, ModuleType
import weakref

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower/production_profile.py"
SPEC = importlib.util.spec_from_file_location("production_profile_unit", SOURCE)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def observer(mode="profile", **kwargs):
    return profile.BoundaryObserver(mode,
        process_facts=lambda: dict(available=True, rss_bytes=123, os_peak_rss_bytes=456),
        threadpool_facts=lambda: dict(available=True, settings_changed=False, pools=[dict(num_threads=2)]),
        **kwargs)


def test_plain_calls_exactly_once_same_args_and_no_observation(monkeypatch):
    code, global_hook = sys.getprofile(), sys.gettrace()
    def forbidden(*a, **k): pytest.fail("plain installed observations")
    item, key = object(), object()
    calls = []
    def original(arg, *, keyword):
        calls.append((arg, keyword))
        return item
    obs = profile.BoundaryObserver("plain", process_facts=forbidden,
        threadpool_facts=forbidden, root_provider=forbidden, gpu_facts=forbidden,
        code_stages={original: "native"})
    monkeypatch.setattr(profile.sys, "setprofile", forbidden)
    assert obs.call("frame", original, item, keyword=key) is item
    assert calls == [(item, key)]
    assert obs.report()["spans"] == [] and obs.report()["audit_hash_bytes"] == 0
    assert sys.getprofile() is code and sys.gettrace() is global_hook


def test_profile_calls_original_once_and_preserves_code_globals_result():
    calls = []
    arg, keyword, result = object(), object(), object()
    def original(value, *, option):
        calls.append((value, option))
        return result
    initial = (original.__code__, original.__globals__, original.__defaults__, original.__kwdefaults__)
    obs = observer(code_stages={original: "native"}, requested_threads=2)
    assert obs.call("frame", original, arg, option=keyword) is result
    assert calls == [(arg, keyword)]
    assert (original.__code__, original.__globals__, original.__defaults__, original.__kwdefaults__) == initial
    report = obs.report()
    assert report["complete"] and not report["scientific_eligibility"]
    assert report["threadpools_before"]["pools"][0]["num_threads"] == 2
    assert report["requested_threads"] == 2
    assert report["sampled_peak_rss_bytes"] == 123 and report["os_peak_rss_bytes"] == 456
    assert report["os_peak_scope"] == "process_lifetime"
    assert report["wall_ns"] >= 0 and report["cpu_ns"] >= 0


def test_exact_code_nested_spans_are_inclusive_and_no_line_events():
    def child():
        time.sleep(.002)
        return 3
    def parent(): return child() + 1
    old_trace = sys.gettrace()
    obs = observer(code_stages={parent: "parent", child: "child"})
    assert obs.call("frame", parent) == 4
    by_stage = {row["stage"]: row for row in obs.spans}
    assert by_stage["parent"]["parent_span_id"] == by_stage["frame"]["span_id"]
    assert by_stage["child"]["parent_span_id"] == by_stage["parent"]["span_id"]
    assert by_stage["parent"]["wall_ns"] >= by_stage["child"]["wall_ns"]
    assert all(row["timing"] == "inclusive" for row in obs.spans)
    assert sys.gettrace() is old_trace and not obs.errors


def test_same_code_clone_is_seen_but_equal_different_code_is_not():
    def original(value): return value + 1
    same = FunctionType(original.__code__, original.__globals__)
    different_code = original.__code__.replace()
    assert different_code == original.__code__ and different_code is not original.__code__
    different = FunctionType(different_code, original.__globals__)
    obs = observer(code_stages={original: "exact"})
    with obs:
        assert same(2) == 3
        assert different(2) == 3
    assert len([row for row in obs.spans if row["stage"] == "exact"]) == 1


def test_decorators_are_inspected_without_replacing_them():
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs): return function(*args, **kwargs)
        return wrapped
    @decorate
    def first(): return 1
    @decorate
    def second(): return 2
    assert first.__code__ is second.__code__  # Shared timing decorator wrapper.
    obs = observer(code_stages={first: "first"})
    with obs:
        assert first() == 1 and second() == 2
    assert [row["stage"] for row in obs.spans] == ["first"]


@pytest.mark.parametrize("mode", ["plain", "profile", "memory-audit"])
def test_original_exception_and_previous_hooks_restored(mode):
    prior = sys.getprofile()
    events = []
    def previous(frame, event, argument): events.append(event)
    def original(): raise ValueError("original guard failure")
    sys.setprofile(previous)
    try:
        obs = observer(mode, code_stages={original: "native"})
        with pytest.raises(ValueError, match="original guard failure"):
            obs.call("frame", original)
        assert sys.getprofile() is previous
        assert obs.report()["original_call_failed"] and not obs.report()["complete"]
        assert not obs._active and (obs._thread is None or not obs._thread.is_alive())
    finally:
        sys.setprofile(prior)


def test_failed_install_restores_prior_hook(monkeypatch):
    previous = sys.getprofile()
    def original(): return 1
    obs = observer(code_stages={original: "native"})
    monkeypatch.setattr(profile.threading.Thread, "start", lambda thread: (_ for _ in ()).throw(RuntimeError("fake start failure")))
    with pytest.raises(RuntimeError, match="fake start failure"):
        with obs: pytest.fail("entered after installation failed")
    assert sys.getprofile() is previous and not obs._active


def test_nested_observers_restore_hooks_and_all_original_calls():
    calls = []
    def original(): calls.append(1)
    before = sys.getprofile()
    outer = observer(code_stages={original: "outer"})
    inner = observer(code_stages={original: "inner"})
    with outer:
        with inner: original()
        original()
    assert calls == [1, 1] and sys.getprofile() is before
    assert [r["stage"] for r in inner.spans] == ["inner"]
    assert [r["stage"] for r in outer.spans] == ["outer", "outer"]


def test_preexisting_profiler_failure_is_not_swallowed():
    before = sys.getprofile()
    calls = []
    def original(): calls.append(1)
    state = [False]
    def previous(frame, event, argument):
        if frame.f_code is original.__code__ and event == "call" and not state[0]:
            state[0] = True
            raise LookupError("prior guard")
    sys.setprofile(previous)
    try:
        obs = observer(code_stages={original: "native"})
        with pytest.raises(LookupError, match="prior guard"): obs.call("frame", original)
        assert not calls and sys.getprofile() is previous
    finally:
        sys.setprofile(before)


def test_metadata_failure_is_reported_and_does_not_replace_original(monkeypatch):
    item = object()
    calls = []
    def original(): calls.append(1); return item
    obs = observer(root_provider=lambda: (_ for _ in ()).throw(ValueError("fake metadata failure")))
    assert obs.call("frame", original) is item
    assert calls == [1] and not obs.report()["complete"] and obs.errors


def test_gpu_provider_is_explicit_and_has_before_after_boundaries():
    events = []
    def facts(stage, boundary):
        events.append((stage, boundary))
        return dict(observed=True, already_active_backend=True, elapsed_ns=12 if boundary == "after" else None)
    obs = observer(gpu_facts=facts)
    assert obs.call("native_gpu", lambda: 5) == 5
    assert events == [("native_gpu", "before"), ("native_gpu", "after")]
    assert obs.spans[0]["gpu"]["elapsed_ns"] == 12
    unobserved = observer()
    unobserved.call("native_cpu", lambda: 7)
    assert unobserved.spans[0]["gpu"]["observed"] is False
    assert unobserved.spans[0]["gpu"]["elapsed_ns"] is None


def test_profile_array_metadata_never_hashes_payload(monkeypatch):
    array = np.arange(2_600_000, dtype=np.float32)
    before = array[:20].copy(), array.flags.writeable
    monkeypatch.setattr(profile, "_array_digest", lambda *a, **k: pytest.fail("timed payload hash"))
    monkeypatch.setattr(profile, "_float_value_digest", lambda *a, **k: pytest.fail("timed value hash"))
    obs = observer(root_provider=lambda: {"cache": array})
    result = obs.call("native", lambda: array)
    assert result is array and obs.report()["audit_hash_bytes"] == 0
    row = obs.report()["arrays"][0]
    assert row["nbytes"] == array.nbytes and row["shape"] == [array.size] and row["dtype"] == array.dtype.str
    assert row["allocation_step"] is None and not row["lifetime_is_exact"]
    assert row["first_seen_stage"] == "native:before" and row["last_seen_stage"] == "native:after"
    assert np.array_equal(array[:20], before[0]) and array.flags.writeable == before[1]


def test_audit_exact_and_cross_dtype_duplicates_are_separate():
    first = np.arange(2_600_000, dtype=np.float32)
    second = first.copy()
    other_dtype = first.astype(np.float64)
    obs = observer("memory-audit")
    obs.snapshot("boundary", dict(first=first, second=second, other_dtype=other_dtype))
    snapshot = obs.report()["snapshots"][0]
    assert len(snapshot["byte_duplicates"]) == 1
    assert len(snapshot["byte_duplicates"][0]["array_ids"]) == 2
    assert len(snapshot["cross_dtype_float_duplicates"]) == 1
    assert len(snapshot["cross_dtype_float_duplicates"][0]["array_ids"]) == 3
    assert snapshot["cross_dtype_float_duplicates"][0]["exact_product_bit_identity"] is False
    assert obs.report()["audit_hash_bytes"] > sum(a.nbytes for a in (first, second, other_dtype))
    assert not profile.compare_numeric_products(first, other_dtype)["exact_numeric_products"]


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_chunk_hash_strided_arrays_matches_logical_bytes(stride):
    root = np.arange(1003, dtype=np.float64)
    array = root[::stride]
    before = root.copy()
    assert profile._array_digest(array, chunk_bytes=64) == hashlib.sha256(array.tobytes(order="C")).hexdigest()
    assert np.array_equal(root, before) and array.flags.writeable


def test_chunk_buffer_cap_large_contiguous_and_strided(monkeypatch):
    array = np.arange(4096, dtype=np.float64).reshape(64, 64)[:, ::2]
    original = profile.hashlib.sha256
    sizes = []
    class Recorder:
        def __init__(self): self.digest = original()
        def update(self, value): sizes.append(len(value)); self.digest.update(value)
        def hexdigest(self): return self.digest.hexdigest()
    monkeypatch.setattr(profile.hashlib, "sha256", Recorder)
    profile._array_digest(array, chunk_bytes=64)
    profile._float_value_digest(array.astype(np.float32), chunk_bytes=64)
    assert sizes and max(sizes) <= 64


def test_base_negative_stride_alias_and_external_storage_metadata():
    root = np.arange(2_600_000, dtype=np.float32)
    reverse = root[::-1]
    obs = observer()
    obs.snapshot("view", dict(root=root, reverse=reverse))
    rows = obs.report()["arrays"]
    assert len(rows) == 2 and rows[0]["storage_owner_id"] == rows[1]["storage_owner_id"]
    assert rows[0]["byte_span"] == rows[1]["byte_span"]
    assert sorted(row["strides"][0] for row in rows) == [-4, 4]
    alias = obs.report()["snapshots"][0]["storage_aliases"][0]
    assert alias["shared_base_owner"] and alias["any_byte_span_overlap"]
    assert alias["exact_strided_overlap_checked"] is False
    external = np.frombuffer(bytearray(40), dtype=np.float64)
    fact = profile._array_metadata(external)
    assert fact["storage_may_be_external"] and fact["owned_storage_bytes"] is None


def test_slots_known_module_roots_and_properties_are_readonly():
    array = np.arange(32, dtype=np.float64)
    class Cache:
        __slots__ = ("array",)
        @property
        def danger(self): pytest.fail("observer invoked property")
    cache = Cache()
    cache.array = array
    known = ModuleType("explicit_known_module")
    known.cache = {"slot_cache": cache}
    obs = observer(array_threshold_bytes=1)
    obs.snapshot("known", {"module": known})  # Nested imported modules intentionally ignored.
    assert not obs.report()["arrays"]
    obs.snapshot("explicit_module", known)
    assert len(obs.report()["arrays"]) == 1


def test_observer_does_not_keep_array_alive_and_labels_collection_honestly():
    array = np.arange(32, dtype=np.float64)
    reference = weakref.ref(array)
    obs = observer(array_threshold_bytes=1)
    obs.snapshot("first_seen", {"array": array})
    del array
    gc.collect()
    assert reference() is None
    row = obs.report()["arrays"][0]
    assert row["collection_observed"] and row["last_seen_stage"] == "first_seen"
    assert row["allocation_step"] is None and row["lifetime_is_exact"] is False


@pytest.mark.parametrize("limit", ["hash", "objects", "arrays", "depth", "spans"])
def test_caps_are_reported_incomplete_without_repeating_original(limit):
    args = dict(array_threshold_bytes=1)
    if limit == "hash": args["max_hash_bytes"] = 1
    if limit == "objects": args["max_objects"] = 1
    if limit == "arrays": args["max_arrays"] = 1
    if limit == "depth": args["max_depth"] = 1
    if limit == "spans": args["max_spans"] = 1
    first, second = np.arange(16), np.arange(16)
    calls = []
    obs = observer("memory-audit", **args)
    with obs:
        obs.call("one", lambda: calls.append(1))
        obs.call("two", lambda: calls.append(2))
        obs.snapshot("bounded", {"cache": {"a": first, "b": second}})
    assert calls == [1, 2] and obs.report()["limits_hit"] and not obs.report()["complete"]


def test_exact_product_identity_including_signed_zero_nan_payload_and_dtype():
    first = dict(field=np.arange(12, dtype=np.float64).reshape(3, 4), scalar=0.0, integer=5, values=(True, None))
    second = {**first, "field": first["field"].copy()}
    assert profile.compare_numeric_products(first, second)["exact_numeric_products"]
    second["scalar"] = -0.0
    assert profile.compare_numeric_products(first, second)["different_paths"] == ["/scalar"]
    second = {**first, "field": first["field"].astype(np.float32)}
    assert profile.compare_numeric_products(first, second)["different_paths"] == ["/field"]
    nan1 = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    nan2 = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    assert not profile.compare_numeric_products(nan1, nan2)["exact_numeric_products"]


def test_container_and_structured_dtype_identity_cannot_disappear():
    assert not profile.compare_numeric_products({}, {"extra": {}})["exact_numeric_products"]
    assert not profile.compare_numeric_products([1], (1,))["exact_numeric_products"]
    first = np.array([(1,)], dtype=[("first", "i8")])
    second = np.array([(1,)], dtype=[("other", "i8")])
    assert first.dtype.str == second.dtype.str
    assert not profile.compare_numeric_products(first, second)["exact_numeric_products"]
    assert not profile.compare_numeric_products(np.float32(1), np.float64(1))["exact_numeric_products"]


def test_plain_profile_generated_products_exact_and_overhead_utility():
    calls = []
    def original():
        calls.append(1)
        signal = np.arange(48, dtype=np.float64).reshape(6, 8)
        return dict(field=signal * .125, sum=float(np.sum(signal, dtype=np.float64)))
    plain = observer("plain")
    profiled = observer(code_stages={original: "same_native"})
    reference = plain.call("frame", original)
    candidate = profiled.call("frame", original)
    assert calls == [1, 1]
    result = profile.compare_numeric_products(reference, candidate,
        plain_wall_ns=plain.report()["wall_ns"], profile_wall_ns=profiled.report()["wall_ns"])
    assert result["exact_numeric_products"] and result["overhead_ratio"] >= 0
    assert not result["scientific_eligibility"] and not result["input_setting_identity_checked"]


@pytest.mark.parametrize("durations", [(None, 2), (0, 2), (True, 2), (1, -1), (1., 2)])
def test_overhead_rejects_invalid_durations(durations):
    with pytest.raises(ValueError): profile.compare_numeric_products(1, 1,
        plain_wall_ns=durations[0], profile_wall_ns=durations[1])


@pytest.mark.parametrize("chunk", [0, True, profile.CHUNK_BYTES + 1])
def test_invalid_chunk_bounds_fail_closed(chunk):
    with pytest.raises(ValueError): profile._array_digest(np.arange(4), chunk_bytes=chunk)


def test_no_native_imports_gpu_probe_or_function_replacement_ast():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.startswith(("darsia", "cupy", "pyopencl")) for name in imports)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"settrace", "set_num_threads", "threadpool_limits"} for node in ast.walk(tree))
    assert not any(isinstance(node, (ast.Assign, ast.AugAssign)) and any(isinstance(target, ast.Attribute) and target.attr in {"__code__", "__globals__"} for target in (node.targets if isinstance(node, ast.Assign) else [node.target])) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)
