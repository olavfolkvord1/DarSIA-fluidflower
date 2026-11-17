"""Unit tests for stored results: whole, never replaced, and through a share's outages."""

import contextlib
import errno
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from darsia.utils import results

DESCRIPTION = {"step": "phase-maps", "settings": {"co2_threshold": 0.8}}
OURS, THEIRS = "1" * 32, "2" * 32
SHORT = (
    "Failed to read all data for array. Expected (3,) = 3 elements, could only read "
    "0 elements. (file seems not fully written?)"
)


def _writer(values):
    """A method's own writer: one array file in the result's folder."""

    def write(folder):
        np.save(folder / "values.npy", values)

    return write


def _values(folder):
    """The method's own reader of the result's array."""
    return np.load(folder / "values.npy")


def _dropped():
    """The error of a dropped connection to the share; Windows reports a lost
    network path as a missing file."""
    if os.name == "nt":
        return OSError(None, "The network path was not found", None, 53)
    return OSError(errno.EHOSTDOWN, "Host is down")


def _token(location):
    """The token a stored result records."""
    return results.read_description(location)["token"]


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """The module's waits, recorded on a clock of their own instead of slept."""
    waits = []
    monkeypatch.setattr(results, "sleep", waits.append)
    monkeypatch.setattr(results, "monotonic", lambda: sum(waits))
    return waits


def test_an_interrupted_write_is_no_result(tmp_path):
    """A writer that fails halfway stores nothing; a second publish stores the
    result with its description."""
    location = tmp_path / "phase-maps" / "run" / "session" / "photo.ARW"

    def stops(folder):
        np.save(folder / "values.npy", np.zeros(3))
        raise RuntimeError("The writer stops halfway")

    with pytest.raises(RuntimeError):
        results.publish(location, stops, DESCRIPTION, THEIRS, root=tmp_path)
    assert not location.exists()
    assert not results.is_complete(location)

    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    assert results.is_complete(location)
    assert results.read_description(location) == dict(DESCRIPTION, token=OURS)
    assert np.array_equal(_values(location), np.arange(3.0))


def test_a_location_outside_its_results_folder_is_refused(tmp_path):
    """A location that leaves its results folder through «..» is refused before
    anything is made or written."""
    root, runs = tmp_path / "results", []
    with pytest.raises(ValueError):
        results.publish(
            root / ".." / "sibling" / "result",
            runs.append,
            DESCRIPTION,
            OURS,
            root=root,
        )
    assert not runs and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            "file",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="POSIX replaces a file; markers have own names"
            ),
        ),
        "folder",
    ],
)
def test_a_move_never_replaces_its_destination(tmp_path, kind):
    """A file or a folder moved onto an existing one is refused, and both stay."""
    paths = {"ours": tmp_path / "source", "theirs": tmp_path / "destination"}
    for text, path in paths.items():
        if kind == "folder":
            path.mkdir()
            path = path / results.RESULT
        path.write_text(text)

    with pytest.raises(FileExistsError):
        results.rename(paths["ours"], paths["theirs"])
    for text, path in paths.items():
        path = path / results.RESULT if kind == "folder" else path
        assert path.read_text() == text


def test_two_moves_of_one_marker_leave_one_owner(tmp_path):
    """Two movers take one marker to names of their own tokens at the same time:
    exactly one move succeeds, and the marker stands under its name alone."""
    for turn in range(20):
        folder = tmp_path / str(turn)
        folder.mkdir()
        (folder / "task").write_text("lease")
        start, moved, refused = threading.Barrier(2), [], []

        def take(token, folder=folder, start=start, moved=moved, refused=refused):
            start.wait()
            try:
                results.rename(folder / "task", folder / f"task.{token}")
                moved.append(token)
            except OSError:
                refused.append(token)

        movers = [threading.Thread(target=take, args=(t,)) for t in (OURS, THEIRS)]
        for mover in movers:
            mover.start()
        for mover in movers:
            mover.join()
        assert len(moved) == len(refused) == 1
        assert [path.name for path in folder.iterdir()] == [f"task.{moved[0]}"]


def test_a_share_error_after_the_writer_does_not_run_it_again(tmp_path, monkeypatch):
    """The description's write meets a dropped connection and is written again; the
    method's writer, which makes a folder of its own, has run once."""
    location = tmp_path / "phase-model"
    runs = []

    def write(folder):
        runs.append(folder)
        (folder / "models").mkdir()
        np.save(folder / "models" / "values.npy", np.arange(3.0))

    write_text, dropped = Path.write_text, []

    def drops(path, *args, **kwargs):
        if path.name == results.RESULT and not dropped:
            dropped.append(path)
            raise _dropped()
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", drops)
    results.publish(location, write, DESCRIPTION, OURS, root=tmp_path)
    assert dropped and len(runs) == 1
    assert _token(location) == OURS
    assert np.array_equal(np.load(location / "models" / "values.npy"), np.arange(3.0))


def test_a_publication_is_flushed_to_disk(tmp_path, monkeypatch):
    """A first writer fails under two new parent folders; the next publication
    flushes every file of its result to disk, and on POSIX also its folder and,
    after the commit, every folder up to the results folder."""
    location = tmp_path / "phase-maps" / "run" / "photo.ARW"
    flushed, fsync = [], os.fsync

    def records(descriptor):
        flushed.append(os.fstat(descriptor)[1:3])
        fsync(descriptor)

    def stops(folder):
        raise RuntimeError("The writer stops")

    monkeypatch.setattr(os, "fsync", records)
    with pytest.raises(RuntimeError):
        results.publish(location, stops, DESCRIPTION, THEIRS, root=tmp_path)
    flushed.clear()
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    paths = [location / "values.npy", location / results.RESULT]
    if os.name != "nt":
        paths += [location, *location.parents[:3]]
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            assert os.fstat(descriptor)[1:3] in flushed
        finally:
            os.close(descriptor)


def test_a_replayed_create_goes_on_in_its_partial(tmp_path, monkeypatch):
    """The partial's create takes effect, then the connection drops before its
    acknowledgement: the write goes on in it, and one result is stored."""
    location = tmp_path / "phase-model"
    create = os.mkdir
    dropped = []

    def drops(path, *args, **kwargs):
        create(path, *args, **kwargs)
        if Path(path).name.endswith(".partial") and not dropped:
            dropped.append(path)
            raise _dropped()

    monkeypatch.setattr(os, "mkdir", drops)
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    assert dropped
    assert [path.name for path in tmp_path.iterdir()] == ["phase-model"]
    assert np.array_equal(_values(location), np.arange(3.0))


def test_a_lasting_error_that_is_not_the_share_surfaces_soon(waits):
    """A full disk is retried once per delay and then raised; only a share that is
    out is waited for over the grace period."""
    calls = []

    def write():
        calls.append(write)
        raise OSError(errno.ENOSPC, "No space left on device")

    with pytest.raises(OSError, match="No space left"):
        results.retry_transient(write)
    assert len(calls) == len(results.RETRY_DELAYS) + 1
    assert waits == list(results.RETRY_DELAYS)


@pytest.mark.parametrize("kind", ["sharing", "lock", "busy"])
def test_a_busy_file_can_be_written_before_its_deadline(monkeypatch, waits, kind):
    """A lock longer than the ordinary retries can clear within the same lease;
    repeated operations share its absolute deadline rather than renewing it."""
    error = OSError(errno.EBUSY, "File busy")
    monkeypatch.setattr(results, "GRACE", 10.0)
    if kind != "busy":
        error.winerror = 32 if kind == "sharing" else 33
    calls = []

    def write():
        calls.append(sum(waits))
        if sum(waits) < 20.0:
            raise error
        return "written"

    assert results.retry_transient(write, deadline=30.0) == "written"
    assert len(calls) > len(results.RETRY_DELAYS) + 1
    assert max(calls) < 30.0

    def locked():
        calls.append(sum(waits))
        raise error

    with pytest.raises(OSError, match="File busy"):
        results.retry_transient(locked, deadline=30.0)
    assert max(calls) < 30.0
    assert sum(waits) < 30.0


def test_a_lease_deadline_does_not_extend_network_outage_grace(monkeypatch, waits):
    """Only a busy file may use the lease's longer remaining time; network
    outages still stop within the normal grace period."""
    monkeypatch.setattr(results, "GRACE", 10.0)

    def out():
        raise _dropped()

    with pytest.raises(OSError):
        results.retry_transient(out, deadline=30.0)
    assert 0 < sum(waits) <= 10.0


def test_an_expired_deadline_runs_no_file_operation(waits):
    """Expired permission cannot be extended by starting a new retry helper."""
    calls = []
    with pytest.raises(TimeoutError, match="deadline"):
        results.retry_transient(lambda: calls.append(True), deadline=0.0)
    assert not calls and not waits


@pytest.mark.parametrize("when", ["stored", "lost"])
@pytest.mark.parametrize("who", [OURS, THEIRS], ids=["ours", "theirs"])
def test_a_destination_is_done_with_our_token_and_refused_with_another(
    tmp_path, monkeypatch, when, who
):
    """A result stands at the location before the commit (stored) or appears in a
    commit whose acknowledgement the dropped connection lost (lost). With our token
    the publication is done, with another it is refused, and nothing is replaced."""
    location = tmp_path / "phase-model"
    standing = tmp_path / "standing"
    results.publish(standing, _writer(np.ones(2)), DESCRIPTION, who, root=tmp_path)
    if who == OURS:
        values = np.ones(2) if when == "stored" else np.arange(3.0)
        refused = contextlib.nullcontext()
    else:
        values, refused = np.ones(2), pytest.raises(FileExistsError)

    move, lost = results._move, []
    if when == "stored":
        move(standing, location)
    else:

        def loses(source, destination):
            if lost:
                return move(source, destination)
            lost.append(source)
            if who == OURS:
                move(source, destination)
            else:
                move(standing, destination)
                move(source, tmp_path / "aside")
            raise _dropped()

        monkeypatch.setattr(results, "_move", loses)
    with refused:
        results.publish(
            location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path
        )
    assert bool(lost) == (when == "lost")
    assert _token(location) == who
    assert np.array_equal(_values(location), values)


def test_a_lost_acknowledgement_with_a_stale_source_is_ours(tmp_path, monkeypatch):
    """The commit takes effect and its acknowledgement is lost; the retried move
    finds the source gone, which the share's client still shows once: the
    destination records this token, so the publication is done."""
    location = tmp_path / "phase-model"
    partial = location.with_name(f".{location.name}.{OURS}.partial")
    move, stat, moves, before = results._move, os.stat, [], []

    def loses(source, destination):
        moves.append(source)
        if len(moves) == 1:
            before.append(stat(source))
            move(source, destination)
            raise _dropped()
        move(source, destination)

    def stale(path, *args, **kwargs):
        if before and isinstance(path, (str, os.PathLike)) and Path(path) == partial:
            return before.pop()
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(results, "_move", loses)
    monkeypatch.setattr(os, "stat", stale)
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    assert len(moves) == 2
    assert _token(location) == OURS
    assert np.array_equal(_values(location), np.arange(3.0))


@pytest.mark.parametrize("case", ["short once", "short twice", "invalid"])
def test_a_short_read_is_read_once_more(tmp_path, waits, case):
    """A whole file read short once is read again after the share's cache time;
    read short twice, the result is unavailable. A reader's other errors are raised
    as they are."""
    location = tmp_path / "phase-model"
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    errors, tries = {
        "short once": ([SHORT], 2),
        "short twice": ([SHORT, SHORT], 2),
        "invalid": (["The values have another shape"], 1),
    }[case]
    calls = []

    def reader(folder):
        calls.append(folder)
        if len(calls) <= len(errors):
            raise ValueError(errors[len(calls) - 1])
        return _values(folder)

    if case == "short once":
        assert np.array_equal(results.read(location, reader), np.arange(3.0))
    else:
        failure = results.Unavailable if case == "short twice" else ValueError
        with pytest.raises(failure):
            results.read(location, reader)
    assert len(calls) == tries
    assert waits == [results.SHARE_CACHE] * (tries - 1)


@pytest.mark.parametrize("failures", [1, 2])
def test_a_file_not_found_in_a_complete_result_is_read_once_more(
    tmp_path, waits, failures
):
    """A file of a complete result that the share's client does not find is looked
    for again after the share's cache time; not found twice, the result is
    unavailable, never missing."""
    location = tmp_path / "phase-model"
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    calls = []

    def reader(folder):
        calls.append(folder)
        if len(calls) <= failures:
            raise FileNotFoundError(errno.ENOENT, "Not found", str(folder))
        return _values(folder)

    if failures == 1:
        assert np.array_equal(results.read(location, reader), np.arange(3.0))
    else:
        with pytest.raises(results.Unavailable):
            results.read(location, reader)
    assert len(calls) == 2
    assert waits == [results.SHARE_CACHE]


@pytest.mark.parametrize("committed", [True, False])
def test_a_committed_description_not_found_is_looked_for_once_more(
    tmp_path, monkeypatch, waits, committed
):
    """The share's client answers «not found» once for the description of a result
    whose commit is known: it is looked for again and the result reads. Without that
    knowledge, a description not found is a plain answer."""
    location = tmp_path / "phase-model"
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    stat, cached = os.stat, []

    def caches(path, *args, **kwargs):
        if (
            isinstance(path, (str, os.PathLike))
            and Path(path) == location / "result.json"
        ):
            if not cached:
                cached.append(path)
                raise FileNotFoundError(errno.ENOENT, "Not found", str(path))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", caches)
    if committed:
        assert np.array_equal(
            results.read(location, _values, committed=True), np.arange(3.0)
        )
        assert waits == [results.SHARE_CACHE]
    else:
        with pytest.raises(FileNotFoundError):
            results.read(location, _values)
        assert waits == []


@pytest.mark.parametrize("hidden", ["destination", "description"])
def test_a_lost_acknowledgement_hidden_by_the_cache_is_ours(
    tmp_path, monkeypatch, waits, hidden
):
    """The commit takes effect, its acknowledgement is lost, the retried move finds
    the source gone, and the share's client answers «not found» once for the
    location or its description: it is looked for again and recognised as this
    publication."""
    location = tmp_path / "phase-model"
    concealed = location if hidden == "destination" else location / "result.json"
    move, stat, moves, cached = results._move, os.stat, [], []

    def loses(source, destination):
        moves.append(source)
        move(source, destination)
        if len(moves) == 1:
            raise _dropped()

    def caches(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and len(moves) > 1 and not cached:
            if Path(path).is_relative_to(concealed):
                cached.append(path)
                raise FileNotFoundError(errno.ENOENT, "Not found", str(path))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(results, "_move", loses)
    monkeypatch.setattr(os, "stat", caches)
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    assert len(moves) == 2 and cached and results.SHARE_CACHE in waits
    assert _token(location) == OURS
    assert np.array_equal(_values(location), np.arange(3.0))


def test_an_outage_is_unavailable_not_missing(tmp_path, monkeypatch, waits):
    """The share goes out while a short read is read once more, for longer than the
    grace period: the read and the completion mark are unavailable, never missing,
    and the stored result keeps its files and token."""
    location = tmp_path / "phase-maps" / "run" / "session" / "photo.ARW"
    results.publish(location, _writer(np.arange(3.0)), DESCRIPTION, OURS, root=tmp_path)
    calls = []

    def reader(folder):
        calls.append(folder)
        if len(calls) == 1:
            raise ValueError(SHORT)
        raise _dropped()

    with pytest.raises(results.Unavailable):
        results.read(location, reader)
    assert waits[0] == results.SHARE_CACHE
    assert results.GRACE - 5.0 < sum(waits[1:]) <= results.GRACE

    stat = os.stat

    def out(path, *args, **kwargs):
        if isinstance(path, os.PathLike) and Path(path).is_relative_to(location):
            raise _dropped()
        return stat(path, *args, **kwargs)

    with monkeypatch.context() as outage:
        outage.setattr(os, "stat", out)
        with pytest.raises(results.Unavailable):
            results.is_complete(location)
    assert [path.name for path in location.parent.iterdir()] == ["photo.ARW"]
    assert _token(location) == OURS
    assert np.array_equal(_values(location), np.arange(3.0))


def test_a_result_of_another_input_is_not_reusable(tmp_path):
    """Phase maps made from one publication of the phase model serve a request
    only while that publication is the model read; a model trained again with
    unchanged settings makes them another result."""
    model = tmp_path / "phase-model"
    maps = tmp_path / "phase-maps" / "run" / "session" / "photo-2.ARW"
    request = {
        "step": "phase-maps",
        "settings": {"co2_threshold": 0.8, "gas_threshold": 0.66},
        "scope": {"run": [["session", "photo-1.ARW"], ["session", "photo-2.ARW"]]},
        "environment": {"blas_kernel": "Haswell", "threads": 1},
        "inputs": {"phase-model": OURS},
    }
    results.publish(
        model, _writer(np.ones(2)), {"step": "phase-model"}, OURS, root=tmp_path
    )
    results.publish(
        maps, _writer(np.arange(3.0)), dict(request, revision="a"), OURS, root=tmp_path
    )
    stored = results.read(maps, results.read_description)
    assert results.reusable(stored, dict(request, revision="b")) is None

    more = {"run": request["scope"]["run"] + [["session", "photo-3.ARW"]]}
    assert results.reusable(stored, dict(request, scope=more)) == (
        "Made with other scope"
    )

    shutil.rmtree(model)
    results.publish(
        model, _writer(np.ones(2)), {"step": "phase-model"}, THEIRS, root=tmp_path
    )
    request["inputs"] = {"phase-model": _token(model)}
    assert results.reusable(stored, request) == "Made with other inputs (phase-model)"
