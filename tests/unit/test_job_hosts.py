"""Tests of the job engine on several hosts: admission by threads and memory, and
the hosts' records and requests."""

import errno
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import darsia.jobs.__main__ as jobs_main
from darsia.jobs import queue, service
from darsia.jobs.job import Sizes, create_job, read_job, task_bytes
from darsia.jobs.step import Memory, Order, Scope, Selection, Step
from darsia.utils import results
from tests.unit.test_jobs import no_share_wait  # noqa: F401
from tests.unit.test_jobs import (
    NUMBERS,
    SUMS,
    ToySettings,
    _files,
    _numbers,
    _project,
)

GIB = 2**30
SIZES = Sizes(1, 0, GIB)
"""Sizes in which a step's grid arrays are GiB."""


def _step(name, grid):
    """A step over frames whose task holds ``grid`` GiB."""
    return Step(
        name,
        {},
        Order.FRAMES,
        Memory(grid=grid),
        _numbers,
        {"toy": ToySettings},
        sections=("protocols",),
        files=_files,
    )


LIGHT, HEAVY = _step("light", 1.0), _step("heavy", 4.0)


class Process:
    """A task process that runs until the test ends it, holding no memory."""

    def __init__(self, started, command):
        self.job, self.task, self.token = command[4], command[5], command[6]
        self.code = None
        self.pid = 999_999  # no process: it holds nothing
        started.append(self)

    def poll(self):
        """Its exit code, once ended."""
        return self.code

    def end(self):
        """Mark its task done and end."""
        queue.finish(queue.Claim(Path(self.job), self.task, self.token, "here"))
        self.code = 0


@pytest.fixture
def started(monkeypatch):
    """The task processes a service starts, which run until a test ends them."""
    found = []
    processes = SimpleNamespace(
        Popen=lambda command, **_: Process(found, command),
        DEVNULL=subprocess.DEVNULL,
        STDOUT=subprocess.STDOUT,
    )
    monkeypatch.setattr(service, "subprocess", processes)
    return found


def _service(tmp_path, monkeypatch, threads=4, free=10 * GIB, kept=GIB, process=0):
    """A service of this host over the test's jobs folder, with the given free
    memory and reserve, and a task process's own bytes (none by default, so that
    the tests count the declared arrays alone)."""
    running = service.Service(tmp_path / "jobs", "here", threads, tmp_path / "logs")
    monkeypatch.setattr(service, "revision", lambda: running.revision)
    memory = {"free": free}
    monkeypatch.setattr(service, "headroom", lambda: memory["free"])
    monkeypatch.setattr(service, "reserve", lambda: kept)
    if process is not None:
        monkeypatch.setattr(service, "PROCESS", process)
    return running, memory


def _network_error(path):
    """The error of a share that is out, as the platform reports it: on Windows a
    network path not found, which Python takes for a missing file."""
    if os.name == "nt":
        return OSError(errno.ENOENT, "The network path was not found", str(path), 53)
    return OSError(errno.EHOSTDOWN, "Host is down", str(path))


def _job(tmp_path, steps, runs=("r1",), name="project"):
    """A job of steps over some runs of the test project."""
    config, selection = _project(tmp_path / name)
    chosen = Selection({r: selection.frames[r] for r in runs}, selection.sessions)
    return create_job(tmp_path / "jobs", steps, chosen, config, sizes=SIZES)


def test_a_task_declares_the_bytes_of_its_input_scope(tmp_path):
    """A task's declared bytes count its frames and those of its inputs, a run's
    result covering the run's selected frames; without the Setup's sizes, a task
    declares none."""
    config, selection = _project(tmp_path)
    held = Step(
        "held", {"sums": Scope.ALL}, Order.PARTS, Memory(1.0, 2.0, 3.0), _numbers
    )
    steps, sizes = (NUMBERS, SUMS, held, LIGHT), Sizes(4, 5, 7)
    job = create_job(tmp_path / "jobs", steps, selection, config, sizes=sizes)
    found = read_job(job)["bytes"]
    assert found["held"] == 4 * (1.0 * 5 + (2.0 + 3.0 * 9) * 7)  # 6 + 3 frames
    assert found["light.r1.0001"] == 4 * 7  # its two frames; no frames held
    task = queue.tasks(job)["held"]
    assert task_bytes(held, task, selection, sizes) == found["held"]
    again = create_job(tmp_path / "again", (LIGHT,), selection, config)
    assert set(read_job(again)["bytes"].values()) == {0}


def test_admission_by_threads_and_bytes_with_a_request_as_a_ceiling():
    """A task starts when the live tasks' threads and its own are within the
    host's, and its bytes within the free memory less the reserve and what the
    live tasks may still take; a request of a number of tasks only caps."""
    live = [service.Live(None, 1, None, bytes=4 * GIB, held=GIB) for _ in range(2)]
    # Free 10 GiB, reserve 1 GiB, 6 GiB still to come: 3 GiB are left.
    assert service.admit(3 * GIB, 1, live, "auto", 3, 10 * GIB, GIB) == (True, "")
    started, why = service.admit(3 * GIB + 1, 1, live, "auto", 3, 10 * GIB, GIB)
    assert not started
    assert why == (
        "it needs 3.00 GiB; 10.00 GiB free less the reserve 1.00 GiB and 6.00 GiB "
        "the live tasks may still take leave 3.00 GiB"
    )
    started, why = service.admit(GIB, 2, live, "auto", 3, 10 * GIB, GIB)
    assert not started and "its 2 threads and the live tasks' 2 exceed" in why
    assert service.admit(GIB, 1, live, 2, 3, 10 * GIB, GIB) == (
        False,
        "the request caps the host at 2 tasks",
    )
    assert service.admit(GIB, 1, live, 3, 3, 10 * GIB, GIB)[0]
    assert not service.admit(4 * GIB, 1, live, 9, 3, 10 * GIB, GIB)[0]  # a ceiling


CHILD = """
import subprocess, sys
held = "import time; held = bytes([1]) * (300 * 2**20); time.sleep(120)"
subprocess.run([sys.executable, "-c", held])
"""
"""A process whose child holds 300 MiB."""


def test_a_task_holds_what_its_process_and_its_children_hold():
    """What a live task holds counts the processes its process started, e.g. the
    interpreter that a launcher starts."""
    parent = subprocess.Popen([sys.executable, "-c", CHILD])
    try:
        live = service.Live(None, 1, parent)
        deadline = time.monotonic() + 60
        while live.resident() < 300 * 2**20:
            assert time.monotonic() < deadline, "the child did not hold its bytes"
            time.sleep(0.2)
    finally:
        for process in [*psutil.Process(parent.pid).children(recursive=True)]:
            process.kill()
        parent.kill()
        parent.wait(30)


@pytest.mark.skipif(os.name != "nt", reason="the commit is Windows'")
def test_the_free_memory_is_the_smaller_of_physical_and_commit(monkeypatch):
    """On Windows the free memory is the smaller of the available physical memory
    and the available commit."""

    class Kernel:
        def __init__(self, physical, commit):
            self.values = physical, commit

        def GlobalMemoryStatusEx(self, pointer):
            status = pointer._obj
            status.ullAvailPhys, status.ullAvailPageFile = self.values
            return 1

    for physical, commit in ((8 * GIB, 3 * GIB), (2 * GIB, 5 * GIB)):
        kernel = Kernel(physical, commit)
        monkeypatch.setattr(service.ctypes, "WinDLL", lambda *a, **k: kernel)
        assert service.headroom() == min(physical, commit)


def test_an_oversize_task_is_passed_over_and_says_why(
    tmp_path, monkeypatch, started, capsys
):
    """A task that would not fit even on the idle host starts nowhere, and the
    host says why once; it holds up no task of a later job."""
    heavy = _job(tmp_path, (HEAVY,), runs=("r2",), name="heavy")
    running, _ = _service(tmp_path, monkeypatch, free=4 * GIB)
    running.poll()
    running.poll()
    assert not started and not queue.markers(heavy, queue.CLAIMED)
    why = (
        f"Task heavy.r2.0000 of job {heavy.name} does not fit this host now: it "
        "needs 4.00 GiB; 4.00 GiB free less the reserve 1.00 GiB and 0.00 GiB the "
        "live tasks may still take leave 3.00 GiB."
    )
    assert running.reason == why
    assert capsys.readouterr().out.count(why) == 1
    light = _job(tmp_path, (LIGHT,), runs=("r2",), name="light")
    running.poll()
    assert [process.task for process in started] == ["light.r2.0000"]
    assert set(queue.markers(light, queue.CLAIMED)) == {"light.r2.0000"}
    assert running.reason == why


def test_mixed_heavy_and_light_tasks_never_exceed_threads_or_bytes(
    tmp_path, monkeypatch, started
):
    """Heavy and light tasks of two jobs share the host's threads and memory, the
    older job's first: at no round do the live tasks hold more threads or declare
    more bytes than the host has, and every task runs."""
    heavy = _job(tmp_path, (HEAVY,), runs=("r1", "r2"), name="heavy")
    light = _job(tmp_path, (LIGHT,), runs=("r1", "r2"), name="light")
    running, _ = _service(tmp_path, monkeypatch, threads=3, free=10 * GIB)
    rounds, most = 0, (0, 0)
    while queue.waiting(heavy) | queue.waiting(light):
        running.poll()
        threads = sum(live.threads for live in running.live.values())
        held = sum(live.bytes for live in running.live.values())
        assert threads <= 3 and held <= 9 * GIB
        most = max(most, (threads, held))
        for process in [p for p in started if p.code is None][:1]:
            process.end()  # the oldest live task ends
        rounds += 1
        assert rounds < 30
    assert most == (3, 9 * GIB)
    order = [process.task for process in started]
    assert order.index("heavy.r2.0000") < order.index("light.r1.0000")
    assert len(order) == 6 and not queue.markers(heavy, queue.FAILED)


def test_a_rise_in_outside_memory_is_seen_at_the_next_start(
    tmp_path, monkeypatch, started
):
    """The free memory is measured at each start: when other programs take memory,
    the next task waits and says why, and starts once it is free again."""
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"))
    running, memory = _service(tmp_path, monkeypatch, threads=1, free=3 * GIB)
    running.poll()
    assert len(started) == 1
    started[0].end()
    memory["free"] = int(1.5 * GIB)  # another program took 1.5 GiB
    running.poll()
    assert len(started) == 1
    assert running.reason.endswith(
        "does not fit this host now: it needs 1.00 GiB; 1.50 GiB free less the "
        "reserve 1.00 GiB and 0.00 GiB the live tasks may still take leave 0.50 GiB."
    )
    memory["free"] = 3 * GIB
    running.poll()
    assert len(started) == 2


def test_a_survivor_reserves_its_threads_and_bytes(tmp_path, monkeypatch, started):
    """A restarted service counts a task of its host that still runs with its
    threads and declared bytes, before it starts any other."""
    job = _job(tmp_path, (HEAVY,), runs=("r1",))
    claim = queue.claim(job, "heavy.r1.0000", "here")
    process, host = psutil.Process(), service.host_identity()
    queue.beat(claim, 1, process.pid, process.create_time(), host)
    monkeypatch.setattr(service.Live, "resident", lambda live: 0)
    running, memory = _service(tmp_path, monkeypatch, free=8 * GIB)
    running.poll()
    (survivor,) = running.live.values()
    assert (survivor.threads, survivor.bytes) == (1, 4 * GIB)
    assert not started  # 8 free less 1 reserved less 4 to come leave 3 GiB
    assert "heavy.r1.0001" in running.reason and "leave 3.00 GiB" in running.reason
    memory["free"] = 9 * GIB
    running.poll()
    assert [process.task for process in started] == ["heavy.r1.0001"]


def test_a_host_takes_at_most_its_cores_less_one_threads(tmp_path, capsys):
    """A service asked for more threads than the host's logical cores less one
    takes only those: a number given is a ceiling, never more."""
    cores = max(1, (os.cpu_count() or 2) - 1)
    (tmp_path / "jobs").mkdir()
    service.serve(
        tmp_path / "jobs",
        "here",
        threads=cores + 8,
        lock=tmp_path / "service.lock",
        logs=tmp_path / "logs",
        until=lambda: True,
    )
    out = capsys.readouterr().out
    assert f"here serves {tmp_path / 'jobs'} with {cores} threads" in out


@pytest.mark.parametrize("unread", ["record", "status", "claims", "lease"])
def test_a_restarted_service_admits_nothing_before_it_has_counted_its_survivors(
    tmp_path, monkeypatch, started, capsys, unread
):
    """A restarted service whose first round cannot read all about a task that
    still runs, its job's record, the record's state through the share, the job's
    claims or the task's lease, starts no task, so that the survivor's threads and
    bytes are never given away; the next round counts the survivor, and its thread
    leaves no room."""
    older = _job(tmp_path, (LIGHT,), runs=("r2",), name="older")
    claim = queue.claim(older, "light.r2.0000", "here")
    process, host = psutil.Process(), service.host_identity()
    queue.beat(claim, 1, process.pid, process.create_time(), host)
    _job(tmp_path, (LIGHT,), runs=("r2",), name="younger")
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    monkeypatch.setattr(results, "GRACE", 0.2)
    monkeypatch.setattr(results, "RETRY_DELAYS", (0.05, 0.1, 0.2))
    with pytest.MonkeyPatch.context() as patch:
        if unread == "record":
            real = service.read_job

            def short(folder):
                if Path(folder) == older:
                    raise results.Unavailable("The file job.json reads short twice")
                return real(folder)

            patch.setattr(service, "read_job", short)
        elif unread == "status":
            stat = os.stat

            def out(path, *args, **kwargs):
                if os.fspath(path) == os.fspath(older / "job.json"):
                    raise _network_error(path)
                return stat(path, *args, **kwargs)

            patch.setattr(os, "stat", out)
        else:
            listdir, read_bytes = os.listdir, Path.read_bytes

            def out(call, unreadable):  # a share's lost connection, not absence
                def blinded(path, *args):
                    if os.fspath(path) == os.fspath(unreadable):
                        raise OSError(errno.EHOSTDOWN, "Host is down", str(path))
                    return call(path, *args)

                return blinded

            if unread == "claims":
                patch.setattr(os, "listdir", out(listdir, older / "claimed"))
            else:
                patch.setattr(Path, "read_bytes", out(read_bytes, claim.path))
        running.poll()
    assert not started and not running.live
    if unread in ("record", "status"):
        assert f"The job {older.name} cannot be read now" in capsys.readouterr().out
    assert running.reason.startswith("Not every job or claim of this host can be")
    running.poll()
    assert [live.pid for live in running.live.values()] == [psutil.Process().pid]
    assert not started and "waits: its 1 threads" in running.reason


def test_a_round_looks_at_each_live_task_once_and_says_a_passed_step_once(
    tmp_path, monkeypatch, started, capsys
):
    """While memory is short, a round looks at what each live task holds once and
    passes over a step's ready tasks without looking at each; the step is said
    once, and again only after a round that did not pass it over."""
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"), name="first")
    running, memory = _service(tmp_path, monkeypatch, threads=4, free=10 * GIB)
    running.poll()
    assert len(running.live) == 3
    heavy = _job(tmp_path, (HEAVY,), runs=("r1", "r2"), name="second")
    looks, asked, admitted = [], [], service.admit
    monkeypatch.setattr(service.Live, "resident", lambda live: looks.append(0) or 0)
    monkeypatch.setattr(service, "admit", lambda *a: asked.append(0) or admitted(*a))
    memory["free"] = 2 * GIB  # every heavy task would not fit even alone
    capsys.readouterr()
    running.poll()
    running.poll()
    assert len(looks) == 2 * 3
    assert len(asked) == 2 * 2  # one task of the step, and its look alone
    said = f"here passes over step heavy. Task heavy.r1.0000 of job {heavy.name}"
    assert capsys.readouterr().out.count(said) == 1
    memory["free"] = 10 * GIB  # one heavy task starts, the next waits for a thread
    running.poll()
    assert len(started) == 4
    memory["free"] = 2 * GIB
    running.poll()
    assert capsys.readouterr().out.count("here passes over step heavy.") == 1


def test_a_host_that_refuses_every_job_says_why(tmp_path, monkeypatch, started):
    """A host that takes no task of any job, e.g. of another code revision, gives
    the first refusal as its reason."""
    job = _job(tmp_path, (LIGHT,), runs=("r2",))
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    running.revision = "another"
    monkeypatch.setattr(service, "revision", lambda: "another")
    running.poll()
    assert not started
    assert running.reason.startswith(
        f"It takes no task of job {job.name}: The job was created at revision"
    )


def test_a_task_reserves_its_process_beside_its_arrays(tmp_path, monkeypatch, started):
    """A task reserves its process's own bytes beside the arrays its step
    declares, so that a task of no declared arrays still needs them."""
    _job(tmp_path, (NUMBERS,), runs=("r2",))
    running, memory = _service(tmp_path, monkeypatch, free=GIB + 2**28, process=None)
    running.poll()
    assert not started
    assert "it needs 0.29 GiB; 1.25 GiB free less the reserve 1.00 GiB" in (
        running.reason
    )
    memory["free"] = 2 * GIB
    running.poll()
    ((_, live),) = running.live.items()
    assert live.bytes == service.PROCESS


def test_a_host_record_shows_its_live_tasks_and_why(tmp_path, monkeypatch, started):
    """A host's record holds its live tasks with the threads and bytes they
    reserve, its free memory, its reserve, its request and why it took no more."""
    job = _job(tmp_path, (HEAVY,), runs=("r1",))
    running, _ = _service(tmp_path, monkeypatch, free=6 * GIB)
    running.poll()
    record = json.loads(service.machine_files(job.parent, "here")[0].read_text())
    assert record["token"] == running.token and record["request"] == "auto"
    assert (record["headroom"], record["reserve"], record["threads"]) == (
        6 * GIB,
        GIB,
        4,
    )
    task = {"job": job.name, "task": "heavy.r1.0000", "threads": 1, "bytes": 4 * GIB}
    assert record["live"] == [{**task, "resident": 0, "backend": "cpu"}]
    assert record["reason"] == (
        f"Task heavy.r1.0001 of job {job.name} waits: it needs 4.00 GiB; 6.00 GiB "
        "free less the reserve 1.00 GiB and 4.00 GiB the live tasks may still take "
        "leave 1.00 GiB."
    )


def test_a_request_caps_the_host_and_zero_takes_no_new_tasks(
    tmp_path, monkeypatch, started, capsys
):
    """A host's request of a number of tasks caps it, 0 takes no new tasks, and a
    request file that holds no request keeps the last, said once."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"))
    running, _ = _service(tmp_path, monkeypatch)
    assert jobs_main.main(["request", str(jobs), "1", "--machine", "here"]) == 0
    running.poll()
    assert (
        len(started) == 1 and "the request caps the host at 1 tasks" in running.reason
    )
    service.write_request(jobs, "here", 0)
    started[0].end()
    running.poll()
    assert len(started) == 1
    assert running.reason == "It takes no new tasks: its request is 0."
    with pytest.raises(ValueError, match="not -1"):
        service.write_request(jobs, "here", -1)
    service.machine_files(jobs, "here")[1].write_text('"many"\n')  # edited by hand
    capsys.readouterr()
    running.poll()
    running.poll()
    assert running.request == 0 and len(started) == 1
    assert capsys.readouterr().out.count("here keeps its request 0") == 1
    service.write_request(jobs, "here", "auto")
    running.poll()
    assert len(started) == 3


def test_the_record_of_another_service_fences_the_host(
    tmp_path, monkeypatch, started, capsys
):
    """A host's record written by another service of the host's name keeps this
    service from counting the host's claims and taking tasks until the record
    stays unchanged for STALE; then it is taken over. A record taken by another
    stops this service again; an earlier service of this host that is gone is
    taken over at once, but not one of another machine of the same network
    name; an empty record is nobody's."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r2",))
    path = service.machine_files(jobs, "here")[0]
    path.parent.mkdir()
    other = {"token": "0123456789abcdef", "host": "elsewhere", "pid": 8, "birth": 1.0}
    path.write_text(json.dumps(other) + "\n")
    monkeypatch.setattr(results, "STALE", 0.5)
    running, _ = _service(tmp_path, monkeypatch)
    running.poll()
    assert not started and not running.held
    assert "written by another service (host elsewhere, process 8)" in running.reason
    path.write_text(json.dumps({**other, "time": 1.0}) + "\n")  # it writes again
    time.sleep(0.3)
    running.poll()
    time.sleep(0.3)
    running.poll()
    assert not started  # unchanged for 0.3 s only
    time.sleep(0.3)
    running.poll()
    assert len(started) == 1 and running.held
    assert json.loads(path.read_text())["token"] == running.token
    taken = {**other, "token": "fedcba9876543210", "time": 2.0}
    path.write_text(json.dumps(taken) + "\n")
    capsys.readouterr()
    running.poll()
    assert not running.held
    assert capsys.readouterr().out.count("here takes no tasks: The record") == 1
    earlier = {**other, "host": service.host_identity(), "pid": 999_999}
    path.write_text(json.dumps(earlier) + "\n")
    again = service.Service(jobs, "here", 4, tmp_path / "logs")
    again.poll()
    assert again.held
    assert json.loads(path.read_text())["token"] == again.token
    assert service.host_identity().startswith(f"{socket.gethostname()}/")
    clone = f"{socket.gethostname()}/another-machine-id"
    path.write_text(json.dumps({**earlier, "host": clone}) + "\n")
    third = service.Service(jobs, "here", 4, tmp_path / "logs")
    third.poll()
    assert not third.held and f"(host {clone}, process 999999)" in third.reason
    path.write_bytes(b"")  # a record whose creator could not write it
    fourth = service.Service(jobs, "here", 4, tmp_path / "logs")
    fourth.poll()
    assert fourth.held


def test_a_takeover_leaves_another_hosts_claims_to_their_leases(
    tmp_path, monkeypatch, started
):
    """A service that takes over the record of another machine under its name
    counts and returns only the claims of its own machine: another machine's
    claim, whose process it cannot look at, is taken back only once its lease is
    stale."""
    jobs = tmp_path / "jobs"
    job = _job(tmp_path, (LIGHT,), runs=("r2",))
    claim = queue.claim(job, "light.r2.0000", "here", "elsewhere/another-machine")
    queue.beat(claim, 1, 999_999, 1.0, "elsewhere/another-machine")
    path = service.machine_files(jobs, "here")[0]
    path.parent.mkdir()
    other = {"token": "0123456789abcdef", "host": "elsewhere/another-machine"}
    path.write_text(json.dumps({**other, "pid": 999_999, "birth": 1.0}) + "\n")
    monkeypatch.setattr(results, "STALE", 0.3)
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    running.poll()
    time.sleep(0.4)
    queue.beat(claim, 2, 999_999, 1.0, "elsewhere/another-machine")  # it beats on
    running.poll()
    assert running.held and results.exists(claim.path) and not running.live
    time.sleep(0.4)  # its lease now stays unchanged for STALE
    running.poll()
    running.poll()
    assert not results.exists(claim.path) and len(started) == 1


def test_a_service_that_lost_its_record_while_paused_does_not_write_it(
    tmp_path, monkeypatch, started
):
    """A service paused between its look at its host's record and its write finds
    the record taken over by another when it writes: it writes nothing, takes no
    tasks from then on, and the other keeps the record and its task."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"))
    first, _ = _service(tmp_path, monkeypatch, threads=1)
    first._identity["host"] = "first/machine"
    first.poll()
    second = service.Service(jobs, "here", 1, tmp_path / "logs-second")
    second._identity["host"] = "second/machine"
    monkeypatch.setattr(results, "STALE", 0.3)
    second.poll()
    first.held = False
    first._hold()  # the first service looks, and is paused
    assert first.held
    time.sleep(0.4)
    second.poll()
    assert second.held and len(second.live) == 1
    first._record()  # it goes on with the round
    assert not first.held and "Another service wrote the record" in first.reason
    found = queue.read_marker(service.machine_files(jobs, "here")[0])
    assert found["token"] == second.token
    second.poll()
    assert second.held and all(results.exists(p) for p in second.live)


def test_a_claim_whose_process_never_started_is_returned_by_the_next_service(
    tmp_path, monkeypatch, started
):
    """A service that ends between a claim and its task's process leaves a claim
    naming its machine and no process; the next service of the machine returns it
    at once."""
    jobs = tmp_path / "jobs"
    job = _job(tmp_path, (LIGHT,), runs=("r2",))
    first, _ = _service(tmp_path, monkeypatch, threads=1)
    first._identity["pid"] = 999_999  # it ended, as its record will say
    monkeypatch.setattr(first, "_start", lambda *args: None)  # it ends at the start
    first.poll()
    assert set(queue.markers(job, queue.CLAIMED)) == {"light.r2.0000"}
    again = service.Service(jobs, "here", 0, tmp_path / "logs")
    again.poll()
    assert again.held and set(queue.markers(job, queue.QUEUE)) == {"light.r2.0000"}


def test_a_record_that_cannot_be_read_takes_no_new_task(tmp_path, monkeypatch, started):
    """A service that held its host's record and cannot read it now keeps its
    live tasks but takes no new task in that round."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"))
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    running.poll()
    assert running.held and len(started) == 1
    started[0].end()
    path = service.machine_files(jobs, "here")[0]
    path.write_bytes(path.read_bytes()[:-1])  # it reads short
    running.poll()
    assert not running.held and len(started) == 1


def test_a_request_not_read_yet_takes_no_new_tasks(tmp_path, monkeypatch, started):
    """A service whose first look at its host's request cannot read it takes no
    new tasks, never taking it for «auto», until the request reads."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r2",))
    service.write_request(jobs, "here", 0)
    service.machine_files(jobs, "here")[1].write_text('"none"\n')  # edited by hand
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    running.poll()
    assert running.request is None and not started
    assert running.reason == "It takes no new tasks: its request is not read yet."
    service.write_request(jobs, "here", "auto")
    running.poll()
    assert len(started) == 1


@pytest.mark.parametrize("value", ["AUTO", "1.5", "many"])
def test_the_request_command_refuses_what_is_no_request(tmp_path, capsys, value):
    """The request command says what a request is, and sets none, when given
    something else."""
    jobs = tmp_path / "jobs"
    assert jobs_main.main(["request", str(jobs), value, "--machine", "here"]) == 2
    assert f"or a number of tasks, not {value!r}" in capsys.readouterr().out
    assert not service.machine_files(jobs, "here")[1].exists()
