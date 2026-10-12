"""Synthetic bytes and metadata only: no scientific inputs, arrays or NAS."""

import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

# Avoid importing the numerical DarSIA package for stdlib-only IO unit tests.
MODULE_PATH = Path(__file__).resolve().parents[2] / "src/darsia/fluidflower/production_io.py"
SPEC = importlib.util.spec_from_file_location("production_io_unit", MODULE_PATH)
pio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pio)


class Clock:
    def __init__(self):
        self.value = 0.0
        self.pauses = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.pauses.append(seconds)
        self.value += seconds


def denied():
    return PermissionError(errno.EACCES, "synthetic sharing error")


@pytest.fixture
def engine(monkeypatch):
    # Simulate a NAS channel on tiny local temporary files, not a real share.
    monkeypatch.setattr(pio, "is_network_path", lambda path: True)
    clock = Clock()
    logs = []
    policy = pio.RetryPolicy(clock=clock.now, sleep=clock.sleep, on_retry=logs.append)
    return pio.NasIO(policy), clock, logs


def inventory(folder):
    return {
        path.relative_to(folder).as_posix(): {
            key: pio.content_asset(path)[key] for key in ("sha256", "size_bytes")
        }
        for path in folder.rglob("*")
        if path.is_file()
    }


def tiny_frame(root):
    root.mkdir()
    (root / "map.bytes").write_bytes(bytes(range(64)))
    (root / "stats.json").write_bytes(b'{"synthetic":true,"scalar":1.25}\n')
    return inventory(root)


def test_locator_case_separators_dots_mapped_drive_and_extended_unc():
    mapping = {"X:": r"\\NAS\Share"}
    expected = pio.normalize_locator(r"\\nas\share\job\file.json", {})
    assert pio.normalize_locator("x:/job/unused/../file.json", mapping) == expected
    assert pio.normalize_locator(r"\\?\UNC\NAS\Share\job\file.json", {}) == expected
    assert pio.normalize_locator("C:/JOB/a/../file.json", {}) == pio.normalize_locator(
        r"c:\job\file.json", {}
    )
    with pytest.raises(pio.IntegrityError):
        pio.normalize_locator("X:/file.json", {"X:": "C:/not_a_share"})


def test_content_identity_ignores_physical_locator_and_accepts_compact(tmp_path):
    first, second = tmp_path / "first.bytes", tmp_path / "second.bytes"
    first.write_bytes(b"same exact bytes")
    second.write_bytes(first.read_bytes())
    full = pio.content_asset(first)
    compact = {key: full[key] for key in ("path", "sha256")}
    assert pio.verify_asset(compact) == full
    assert pio.NasIO().verify(compact) == first
    assert pio.same_content(full, pio.content_asset(second))
    assert pio.content_identity(full) == (full["sha256"], full["size_bytes"])
    assert pio.normalize_locator(first) != pio.normalize_locator(second)


@pytest.mark.parametrize(
    "delta",
    [
        {"sha256": "0" * 64},
        {"sha256": "not SHA"},
        {"size_bytes": 99},
        {"size_bytes": True},
        {"size_bytes": -1},
        {"extra": "forbidden"},
        {"path": "relative.bytes"},
    ],
)
def test_bad_descriptor_never_retries(tmp_path, engine, delta):
    io, clock, logs = engine
    path = tmp_path / "own.bytes"
    path.write_bytes(b"own")
    bound = pio.content_asset(path)
    with pytest.raises(pio.IntegrityError):
        io.verify({**bound, **delta})
    assert not clock.pauses and not logs


def test_changed_content_and_missing_file_are_not_transient(tmp_path, engine):
    io, clock, _ = engine
    path = tmp_path / "own.bytes"
    path.write_bytes(b"own")
    bound = pio.content_asset(path)
    path.write_bytes(b"bad")
    with pytest.raises(pio.IntegrityError):
        io.verify(bound)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        io.verify(bound)
    assert not clock.pauses


@pytest.mark.parametrize("duration", [1.0, 10.0])
def test_read_faults_preserve_exact_bytes(tmp_path, engine, monkeypatch, duration):
    io, clock, logs = engine
    path = tmp_path / "own.bytes"
    expected = bytes(range(64))
    path.write_bytes(expected)
    bound = pio.content_asset(path)
    original = Path.open
    guards = []

    def opening(item, *args, **kwargs):
        if item == path and clock.now() < duration:
            raise denied()
        return original(item, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    assert io.read_bytes(bound, before_attempt=lambda: guards.append(True)) == expected
    assert len(guards) == len(logs) + 1
    assert clock.pauses[:4] == [0.1, 0.2, 0.4, 0.8]
    assert max(clock.pauses) <= 5


def test_retry_exact_budget_and_local_single_attempt(engine, monkeypatch):
    io, clock, logs = engine
    with pytest.raises(PermissionError):
        io.retry.run("read", "X:/own", lambda: (_ for _ in ()).throw(denied()))
    assert sum(clock.pauses) == pytest.approx(60)
    assert logs[-1]["exhausted"]
    monkeypatch.setattr(pio, "is_network_path", lambda path: False)
    pauses = len(clock.pauses)
    with pytest.raises(PermissionError):
        io.retry.run("read", "C:/own", lambda: (_ for _ in ()).throw(denied()))
    assert len(clock.pauses) == pauses


@pytest.mark.parametrize(
    "error",
    [FileExistsError(), FileNotFoundError(), ValueError("bad content"), OSError(errno.EIO, "bad"), PermissionError(errno.EPERM, "permanent")],
)
def test_other_errors_and_guard_failures_do_not_retry(engine, error):
    io, clock, _ = engine
    with pytest.raises(type(error)):
        io.retry.run("operation", "X:/own", lambda: (_ for _ in ()).throw(error))
    assert not clock.pauses
    with pytest.raises(PermissionError):
        io.retry.run(
            "operation", "X:/own", lambda: pytest.fail("guard prevented IO"),
            before_attempt=lambda: (_ for _ in ()).throw(denied()),
        )
    assert not clock.pauses


@pytest.mark.parametrize("winerror", [5, 32, 33, 64, 121, 258, 1460])
def test_known_windows_errors_and_timeouts(winerror):
    error = OSError(errno.EIO, "synthetic network error")
    error.winerror = winerror
    assert pio.is_transient(error)
    error.winerror = 87
    assert not pio.is_transient(error)
    assert pio.is_transient(TimeoutError())


@pytest.mark.parametrize("duration", [1.0, 10.0])
def test_write_open_faults_retry_without_false_publication(tmp_path, engine, monkeypatch, duration):
    io, clock, logs = engine
    target = tmp_path / "published.bytes"
    data = b"synthetic output bytes"
    original = Path.open

    def opening(item, *args, **kwargs):
        if item.name.endswith(".partial") and args and args[0] == "xb" and clock.now() < duration:
            raise denied()
        return original(item, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opening)
    bound = io.write_bytes(target, data)
    assert target.read_bytes() == data
    assert bound["sha256"] == hashlib.sha256(data).hexdigest()
    assert logs and clock.now() >= duration


def test_partial_write_retained_new_uuid_and_conflicts_nondestructive(tmp_path, engine, monkeypatch):
    io, _, _ = engine
    target = tmp_path / "output.bytes"
    original = Path.open
    once = [True]

    class Stream:
        def __init__(self, actual):
            self.actual = actual

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.actual.close()

        def write(self, data):
            self.actual.write(data[:3])
            raise denied()

    def opening(item, *args, **kwargs):
        actual = original(item, *args, **kwargs)
        if item.name.endswith(".partial") and args and args[0] == "xb" and once[0]:
            once[0] = False
            return Stream(actual)
        return actual

    monkeypatch.setattr(Path, "open", opening)
    data = b"complete synthetic output"
    io.write_bytes(target, data)
    partials = list(tmp_path.glob("*.partial"))
    assert len(partials) == 1 and partials[0].read_bytes() == data[:3]
    with pytest.raises(FileExistsError):
        io.write_bytes(target, b"different")
    assert target.read_bytes() == data
    io.write_bytes(target, data, allow_existing=True)
    with pytest.raises(pio.IntegrityError):
        io.write_bytes(target, b"different", allow_existing=True)
    assert target.read_bytes() == data


def test_copy_immutable_source_retained_and_hash_after_write(tmp_path, engine):
    io, _, _ = engine
    source, target = tmp_path / "local.bytes", tmp_path / "NAS_copy.bytes"
    data = b"immutable source"
    source.write_bytes(data)
    desc = pio.content_asset(source)
    copied = io.copy_file(desc, target)
    assert source.read_bytes() == target.read_bytes() == data
    assert copied["sha256"] == desc["sha256"]
    assert io.copy_file(desc, target) == copied
    target.write_bytes(b"external conflict")
    with pytest.raises(pio.IntegrityError):
        io.copy_file(desc, target)
    assert target.read_bytes() == b"external conflict" and source.read_bytes() == data


@pytest.mark.parametrize("duration", [1.0, 10.0])
def test_directory_rename_faults_exact_products_and_guard_each_attempt(tmp_path, engine, monkeypatch, duration):
    io, clock, logs = engine
    source, target = tmp_path / "frame.partial", tmp_path / "frame"
    expected = tiny_frame(source)
    original = pio._move_no_replace
    guards = []

    def move(a, b):
        if clock.now() < duration:
            raise denied()
        return original(a, b)

    monkeypatch.setattr(pio, "_move_no_replace", move)
    actual = io.publish_directory(source, target, expected, before_attempt=lambda: guards.append(True))
    assert not source.exists() and inventory(target) == expected
    assert set(actual) == set(expected) and len(guards) == len(logs) + 1


def test_ambiguous_directory_rename_exact_destination_and_source_gone(tmp_path, engine, monkeypatch):
    io, clock, _ = engine
    source, target = tmp_path / "frame.partial", tmp_path / "frame"
    expected = tiny_frame(source)
    original = pio._move_no_replace
    calls = []

    def move(a, b):
        calls.append(True)
        original(a, b)
        raise denied()

    monkeypatch.setattr(pio, "_move_no_replace", move)
    io.publish_directory(source, target, expected)
    assert not source.exists() and inventory(target) == expected
    assert calls == [True] and not clock.pauses


def test_ambiguous_directory_wrong_bytes_stops_without_retry(tmp_path, engine, monkeypatch):
    io, clock, _ = engine
    source, target = tmp_path / "frame.partial", tmp_path / "frame"
    expected = tiny_frame(source)
    original = pio._move_no_replace

    def move(a, b):
        original(a, b)
        (b / "map.bytes").write_bytes(b"wrong")
        raise denied()

    monkeypatch.setattr(pio, "_move_no_replace", move)
    with pytest.raises(pio.IntegrityError):
        io.publish_directory(source, target, expected)
    assert (target / "map.bytes").read_bytes() == b"wrong" and not clock.pauses


def test_directory_conflict_source_remaining_not_success(tmp_path, engine, monkeypatch):
    io, _, _ = engine
    source, target = tmp_path / "frame.partial", tmp_path / "frame"
    expected = tiny_frame(source)

    def move(a, b):
        b.mkdir()
        raise denied()

    monkeypatch.setattr(pio, "_move_no_replace", move)
    with pytest.raises(FileExistsError):
        io.publish_directory(source, target, expected)
    assert inventory(source) == expected and not list(target.iterdir())


def test_directory_inventory_rejects_extra_missing_escaping_and_wrong_hash(tmp_path, engine):
    io, clock, _ = engine
    source, target = tmp_path / "partial", tmp_path / "final"
    expected = tiny_frame(source)
    for bad in ({"../escape": next(iter(expected.values()))}, {"C:/escape": next(iter(expected.values()))}, {**expected, "missing": dict(sha256="a" * 64, size_bytes=1)}):
        with pytest.raises(pio.IntegrityError):
            io.publish_directory(source, target, bad)
    (source / "extra.bytes").write_bytes(b"unexpected")
    with pytest.raises(pio.IntegrityError):
        io.publish_directory(source, target, expected)
    assert source.exists() and not target.exists() and not clock.pauses


def test_copy_directory_keeps_local_products_and_idempotent_verified_destination(tmp_path, engine):
    io, _, _ = engine
    source, target = tmp_path / "local", tmp_path / "copy"
    expected = tiny_frame(source)
    result = io.copy_directory(source, target, expected)
    assert inventory(source) == inventory(target) == expected
    assert io.copy_directory(source, target, expected) == result
    (target / "map.bytes").write_bytes(b"conflict")
    with pytest.raises(pio.IntegrityError):
        io.copy_directory(source, target, expected)
    assert inventory(source) == expected and (target / "map.bytes").read_bytes() == b"conflict"


@pytest.mark.parametrize("duration", [1.0, 10.0])
def test_replace_retry_regenerates_authority_and_leaves_failed_partials(tmp_path, engine, monkeypatch, duration):
    io, clock, logs = engine
    target = tmp_path / "control.json"
    target.write_bytes(b'{"old":true}')
    original = pio.os.replace
    provided, guards = [], []

    def document():
        provided.append(True)
        return dict(sequence=len(provided), timestamp=1791773405019108500, created_utc="2026-10-12T00:00:00Z")

    def replace(a, b):
        if clock.now() < duration:
            raise denied()
        return original(a, b)

    monkeypatch.setattr(pio.os, "replace", replace)
    bound = io.replace_json(target, document, before_attempt=lambda: guards.append(True))
    assert io.read_json(bound)["sequence"] == len(provided) == len(guards)
    assert len(list(tmp_path.glob("*.partial"))) == len(logs)


def test_ambiguous_replace_success_exact_destination_source_gone(tmp_path, engine, monkeypatch):
    io, clock, _ = engine
    target = tmp_path / "control.json"
    original = pio.os.replace

    def replace(a, b):
        original(a, b)
        raise denied()

    monkeypatch.setattr(pio.os, "replace", replace)
    result = io.replace_json(target, lambda: {"timestamp": "1791773405.0191085"})
    assert io.read_json(result)["timestamp"] == "1791773405.0191085"
    assert not clock.pauses


def test_json_exact_timestamp_keys_nonfinite_duplicates_and_guard_reject(tmp_path, engine):
    io, clock, _ = engine
    target = tmp_path / "metadata.json"
    payload = {"timestamp": 1791773405019108500, "process_create_time": "1791773405.0191085", "value": 1.25}
    io.write_json(target, payload)
    assert io.read_json(target) == payload
    for bad in ({"timestamp": 1791773405.0191085}, {"value": float("nan")}, {1: "not a string key"}):
        with pytest.raises(ValueError):
            io.write_json(tmp_path / "not_created.json", bad)
    for data in (b'{"same":1,"same":2}', b'{"value":NaN}', b'{"create_time_epoch":1.25}'):
        corrupt = tmp_path / "bad.json"
        corrupt.write_bytes(data)
        with pytest.raises(ValueError):
            io.read_json(corrupt)
    with pytest.raises(PermissionError):
        io.replace_json(target, lambda: payload, before_attempt=lambda: (_ for _ in ()).throw(denied()))
    assert io.read_json(target) == payload and not clock.pauses


def test_read_limit_and_postwrite_hash_failure_not_retried(tmp_path, engine, monkeypatch):
    io, clock, _ = engine
    path = tmp_path / "own.bytes"
    path.write_bytes(b"four")
    with pytest.raises(pio.IntegrityError):
        io.read_bytes(path, max_bytes=3)
    original = pio._hash_file

    def hashing(p):
        result = original(p)
        return {**result, "sha256": "0" * 64} if Path(p).name.endswith(".partial") else result

    monkeypatch.setattr(pio, "_hash_file", hashing)
    target = tmp_path / "no_final.bytes"
    with pytest.raises(pio.IntegrityError):
        io.write_bytes(target, b"expected")
    assert not target.exists() and list(tmp_path.glob("*.partial")) and not clock.pauses


@pytest.mark.skipif(os.name != "nt", reason="Windows filename locators are case insensitive")
def test_directory_inventory_case_and_separator_normalization(tmp_path):
    folder = tmp_path / "own"
    expected = tiny_frame(folder)
    upper = {name.upper(): bound for name, bound in expected.items()}
    result = pio.NasIO().verify_directory(folder, upper)
    assert set(result) == set(upper)
    with pytest.raises(pio.IntegrityError):
        pio.NasIO().verify_directory(folder, {**expected, **upper})


def test_short_writes_complete_without_content_mutation():
    written = bytearray()

    class Short:
        def write(self, data):
            count = min(3, len(data))
            written.extend(data[:count])
            return count

    data = b"synthetic complete bytes"
    pio._write_all(Short(), data)
    assert bytes(written) == data


def test_directory_move_read_fault_after_success_does_not_replay_rename(tmp_path, engine, monkeypatch):
    io, _, _ = engine
    source, target = tmp_path / "partial", tmp_path / "final"
    expected = tiny_frame(source)
    original_move, original_snapshot = pio._move_no_replace, pio._directory_snapshot
    calls, failed = [], [False]

    def move(a, b):
        calls.append(True)
        return original_move(a, b)

    def snapshot(path, bindings):
        if Path(path) == target and not failed[0]:
            failed[0] = True
            raise denied()
        return original_snapshot(path, bindings)

    monkeypatch.setattr(pio, "_move_no_replace", move)
    monkeypatch.setattr(pio, "_directory_snapshot", snapshot)
    io.publish_directory(source, target, expected)
    assert calls == [True] and inventory(target) == expected
