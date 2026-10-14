"""Tiny metadata fixtures for exclusive Windows queue renames."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import threading
import time
import uuid

import pytest

from darsia.fluidflower.pipeline import job_queue as queue


def _claim_process(root, index, ready, start, answers):
    """Claim a tiny task in an independent spawned interpreter."""
    import psutil

    process = psutil.Process()
    owner = dict(pid=process.pid, process_create_time=str(process.create_time()))
    ready.put(index)
    if not start.wait(20):
        raise RuntimeError("Parent did not release metadata fixture")
    try:
        claim = queue.JobQueue(Path(root)).claim(f"child-{index}", "cpu", owner)
        answers.put(dict(index=index, claimed=None if claim is None else claim.token))
    except Exception as exc:
        answers.put(dict(index=index, error=repr(exc)))
        raise


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive-handle semantics")
def test_thread_renames_have_one_winner(tmp_path):
    source = tmp_path / "pending.json"
    source.write_bytes(b"one immutable metadata fixture")
    barrier = threading.Barrier(16)

    def attempt(index):
        destination = tmp_path / f"claimed-{index}.json"
        barrier.wait()
        try:
            queue.rename(source, destination)
        except FileNotFoundError:
            return None
        return destination

    with ThreadPoolExecutor(16) as pool:
        winners = [path for path in pool.map(attempt, range(16)) if path is not None]
    assert len(winners) == 1
    assert winners[0].read_bytes() == b"one immutable metadata fixture"
    assert not source.exists()
    assert len(list(tmp_path.glob("claimed-*.json"))) == 1


def test_spawned_claimers_have_one_owner(tmp_path):
    shared = queue.JobQueue(tmp_path)
    shared.enqueue(dict(run="sample", commit="1" * 40,
        calibration_sha256="2" * 64,
        configuration=dict(path=r"\\server\share\fixture.json", sha256="3" * 64),
        precision="float32", backend="cpu"))
    context = multiprocessing.get_context("spawn")
    ready, answers = context.Queue(), context.Queue()
    start = context.Event()
    processes = [context.Process(target=_claim_process,
        args=(str(tmp_path), index, ready, start, answers)) for index in range(4)]
    try:
        for process in processes:
            process.start()
        assert {ready.get(timeout=30) for _ in processes} == set(range(4))
        start.set()
        results = [answers.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        assert not any("error" in result for result in results)
        assert sum(result["claimed"] is not None for result in results) == 1
        assert len(list((tmp_path / "leases").glob("*/owner.json"))) == 1
        assert len(list((tmp_path / "claimed").glob("*/*.json"))) == 1
        assert not list((tmp_path / "queue").glob("*.json"))
    finally:
        start.set()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=30)
                if not process.is_alive():
                    process.close()
        for channel in (ready, answers):
            channel.close()
            channel.join_thread()


@pytest.mark.skipif(os.name != "nt", reason="Windows no-replace rename")
@pytest.mark.parametrize("directory", [False, True])
def test_existing_destination_is_preserved(tmp_path, directory):
    source, destination = tmp_path / "source", tmp_path / "destination"
    if directory:
        source.mkdir(); destination.mkdir()
        (source / "source.txt").write_bytes(b"source")
        (destination / "destination.txt").write_bytes(b"destination")
    else:
        source.write_bytes(b"source"); destination.write_bytes(b"destination")
    with pytest.raises(FileExistsError):
        queue.rename(source, destination)
    if directory:
        assert (source / "source.txt").read_bytes() == b"source"
        assert (destination / "destination.txt").read_bytes() == b"destination"
    else:
        assert source.read_bytes() == b"source"
        assert destination.read_bytes() == b"destination"


@pytest.mark.parametrize("directory", [False, True])
def test_file_and_directory_move_preserve_contents(tmp_path, directory):
    source, destination = tmp_path / "source", tmp_path / "renamed-ø-😀"
    if directory:
        source.mkdir()
        (source / "metadata.json").write_bytes(b"immutable fixture")
    else:
        source.write_bytes(b"immutable fixture")
    queue.rename(source, destination)
    assert not source.exists()
    assert (destination / "metadata.json" if directory else destination).read_bytes() == b"immutable fixture"


def test_missing_source_surfaces_without_creating_destination(tmp_path):
    with pytest.raises(FileNotFoundError):
        queue.rename(tmp_path / "missing", tmp_path / "destination")
    assert not (tmp_path / "destination").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_reader_may_close_before_bounded_retry(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"reader fixture")
    reader = source.open("rb")
    with ThreadPoolExecutor(1) as pool:
        try:
            future = pool.submit(queue.rename, source, destination)
            time.sleep(.05)
            assert not future.done()
            assert reader.read() == b"reader fixture"
        finally:
            reader.close()
        future.result(timeout=3)
    assert destination.read_bytes() == b"reader fixture"


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing policy")
def test_reader_remaining_open_stops_without_move(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"held reader fixture")
    with source.open("rb"):
        with pytest.raises(PermissionError) as failure:
            queue.rename(source, destination)
    assert failure.value.winerror in {5, 32}
    assert source.read_bytes() == b"held reader fixture"
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows SMB handle API")
def test_smb_validation_fixture_when_requested():
    """An explicit validation-only NAS root exercises the actual SMB API."""
    configured = os.environ.get("DARSIA_RENAME_VALIDATION_ROOT")
    if configured is None:
        pytest.skip("No explicit NAS validation fixture root requested")
    root = Path(configured)
    queue.require_unc(str(root))
    if "code/validation/" not in str(root).replace("\\", "/").casefold():
        raise ValueError("NAS fixture must stay under code/validation")
    owned = root / ("rename_smb_" + uuid.uuid4().hex)
    owned.mkdir()
    source, destination = owned / "pending.json", owned / "claimed.json"
    source.write_bytes(b"SMB immutable metadata fixture")
    queue.rename(source, destination)
    assert destination.read_bytes() == b"SMB immutable metadata fixture"
    source_directory, final_directory = owned / "lease", owned / "published"
    source_directory.mkdir()
    (source_directory / "metadata.json").write_bytes(b"SMB directory fixture")
    queue.rename(source_directory, final_directory)
    assert (final_directory / "metadata.json").read_bytes() == b"SMB directory fixture"
    print("Preserved NAS validation fixture:", owned)
