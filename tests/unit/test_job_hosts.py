"""Tests of the job engine on several hosts: admission by threads and memory, the
hosts' records and requests, and outages of the share."""

import builtins
import contextlib
import errno
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import darsia
import darsia.jobs.__main__ as jobs_main
from darsia.jobs import queue, service
from darsia.jobs.job import Sizes, create_job, read_job, task_bytes
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, Term
from darsia.utils import results
from tests.unit.test_job_service import importable  # noqa: F401
from tests.unit.test_jobs import no_share_wait  # noqa: F401
from tests.unit.test_jobs import (
    NUMBERS,
    SUMS,
    ToySettings,
    _files,
    _numbers,
    _project,
    _stored,
    _write,
    read_values,
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


def _lingering(ctx):
    """A number per frame, made once the project holds no file named "linger"."""
    while (ctx.locations.project / "linger").exists():
        time.sleep(0.1)
    for frame in ctx.task.frames:
        ctx.publish(lambda folder: _write(folder, value=1), frame=frame)


LINGERING = Step("lingering", {}, Order.FRAMES, Memory(), _lingering)


def _gathered(ctx):
    """The values of a run's lingering frames."""
    values = [ctx.read("lingering", read_values, frame=f) for f in ctx.task.frames]
    ctx.publish(lambda folder: _write(folder, values=values))


GATHERED = Step("gathered", {"lingering": Scope.OWN}, Order.RUN, Memory(), _gathered)
STEPS = (LINGERING, GATHERED)
"""The test steps a task process finds by their module."""


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


def test_a_tasks_bytes_count_its_terms_fixed_arrays_and_photos(tmp_path):
    """A task's bytes count its values in the precision and its arrays of a fixed
    type per pixel, those per unit of each count its configuration sets for its
    run, and the reader's bytes of its run's photos by their format."""
    config, selection = _project(tmp_path)
    term = Term(
        lambda tables, run, project: 3 if run == "r1" else 1, grid=1.0, grid_bytes=2.0
    )
    memory = Memory(
        camera=1.0,
        grid=2.0,
        camera_bytes=5.0,
        grid_bytes=7.0,
        terms=(term,),
        photos=1,
        photo_bytes=darsia.decoded_bytes,
    )
    counted = Step("counted", {}, Order.RUN, memory, _numbers)
    job = create_job(
        tmp_path / "jobs", (counted,), selection, config, sizes=Sizes(4, 10, 100)
    )
    found = read_job(job)["bytes"]
    photo = darsia.decoded_bytes("r1-00.npz") * 10
    assert found["counted.r1"] == 4 * (10 + 5 * 100) + 50 + 13 * 100 + photo
    assert found["counted.r2"] == 4 * (10 + 3 * 100) + 50 + 9 * 100 + photo


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


TASK = r"""
import errno, os, pathlib, sys
from darsia.utils import results
results.BEAT, results.GRACE, results.RETRY_DELAYS = 0.2, 4.0, (0.05, 0.1, 0.2)
from darsia.jobs import queue
from darsia.jobs.__main__ import main
flag, real = pathlib.Path(sys.argv[1]), open
def out(kind, path):
    try:
        with real(flag, encoding="utf-8") as stream:
            what, *roots = stream.read().split("|")
    except OSError:
        return False
    return what == kind and any(os.fspath(path).startswith(r) for r in roots)
def blind(call, kind="files"):
    def blinded(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and out(kind, path):
            if os.name == "nt":
                raise OSError(errno.ENOENT, "The network path was not found", str(path), 53)
            raise OSError(errno.EHOSTDOWN, "Host is down", str(path))
        return call(path, *args, **kwargs)
    return blinded
queue.open = blind(real)
os.stat, os.listdir = blind(os.stat), blind(os.listdir)
pathlib.Path.read_bytes = blind(pathlib.Path.read_bytes)
pathlib.Path.open = blind(pathlib.Path.open)
results._move = blind(results._move, "renames")
sys.exit(main(sys.argv[2:]))
"""
"""A task process with short guards (a beat of 0.2 s, a grace of 4 s) on a share
that is out while its host's flag file exists: the file names what is out,
``files|<root>|...`` (opening, reading, listing and looking at files under the
roots, the claim's writes too) or ``renames|<root>|...`` (renaming from under
them)."""


@pytest.fixture
def hosts(tmp_path, monkeypatch, importable):  # noqa: F811
    """Task processes of the test's hosts (:data:`TASK`), the flag file of a host
    being ``outage-<host>``; this process's services retry for 0.2 s and take a
    claim back after 20 s, more than a task's grace and beat and the start of a
    task process. Task processes still running at the end are ended."""
    real, made = subprocess.Popen, []

    def popen(command, **kwargs):
        flag = tmp_path / f"outage-{command[command.index('--machine') + 1]}"
        made.append(real([command[0], "-c", TASK, str(flag), *command[3:]], **kwargs))
        return made[-1]

    processes = SimpleNamespace(
        Popen=popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT
    )
    monkeypatch.setattr(service, "subprocess", processes)
    monkeypatch.setattr(results, "GRACE", 0.2)
    monkeypatch.setattr(results, "RETRY_DELAYS", (0.05, 0.1, 0.2))
    monkeypatch.setattr(results, "STALE", 20.0)
    yield tmp_path
    for process in made:
        _end(process)


@contextlib.contextmanager
def _share_out(root):
    """The share of ``root`` is out for this process: its files and folders give
    the network's error."""
    root = os.fspath(root)

    def blind(call):
        def blinded(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)):
                if os.fspath(path).startswith(root):
                    raise _network_error(path)
            return call(path, *args, **kwargs)

        return blinded

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (os, "listdir"),
            (os, "stat"),
            (builtins, "open"),
            (pathlib.Path, "read_bytes"),
        ):
            patch.setattr(owner, name, blind(getattr(owner, name)))
        yield


def _until(condition, seconds=120.0):
    """Wait until a condition holds, at most some seconds."""
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition did not come"
        time.sleep(0.1)


def _lingering_job(root, jobs=None, steps=(LINGERING,)):
    """A job of a frame that lingers while its project holds "linger", in a jobs
    folder (by default the root's)."""
    config, selection = _project(root)
    one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
    (root / "project" / "linger").touch()
    return create_job(jobs or root / "jobs", steps, one, config)


def _beat(job, task):
    """The beat number of a task's lease, 0 when it is not claimed."""
    claimed = queue.markers(job, queue.CLAIMED)
    if task not in claimed:
        return 0
    return queue.lease(job, task, claimed[task]).get("beat", 0)


def test_an_outage_shorter_than_grace_keeps_the_claim_and_the_capacity(hosts):
    """The share is out for less than the grace, for a running task and for its
    host's service: the task keeps its claim, no host takes it back, its service
    keeps the task, its reservation and its request, and the task completes under
    its claim. While the share is out, the queue's state is not known, never
    empty."""
    jobs, task = hosts / "jobs", "lingering.r2.0000"
    job = _lingering_job(hosts)
    one = service.Service(jobs, "one", 1, hosts / "logs")
    other = service.Service(jobs, "other", 1, hosts / "logs")
    service.write_request(jobs, "one", 1)
    one.poll()
    ((path, live),) = one.live.items()
    _until(lambda: _beat(job, task) >= 3)
    (hosts / "outage-one").write_text(f"files|{jobs}")
    began = time.monotonic()
    with _share_out(jobs):
        one.poll()
        with pytest.raises(results.Unavailable):
            queue.accepted(job, task)
    assert one.live == {path: live} and one.request == 1
    while time.monotonic() - began < 2.0:  # less than the task's grace of 4 s
        other.poll()
        time.sleep(0.2)
    beaten = _beat(job, task)
    (hosts / "outage-one").unlink()
    _until(lambda: _beat(job, task) > beaten)
    other.poll()
    (hosts / "project" / "linger").unlink()
    assert live.process.wait(60) == 0
    one.poll()
    assert queue.accepted(job, task) == {live.claim.token}
    assert not one.live and not one.exits and not other.live and one.held


def test_an_outage_longer_than_grace_stops_the_task_and_it_is_taken_back_after_stale(
    hosts,
):
    """The share is out for longer than the grace for a task and its host's
    service: the task stops and says why; another host takes its claim back once
    its lease is stale, and runs it. When the share is back, the first host frees
    the task's reservation and counts no failure; the stopped owner's token is
    never accepted."""
    jobs, task = hosts / "jobs", "lingering.r2.0000"
    job = _lingering_job(hosts)
    one = service.Service(jobs, "one", 1, hosts / "logs")
    other = service.Service(jobs, "other", 1, hosts / "logs")
    service.write_request(jobs, "other", 0)
    one.poll()
    other.poll()
    ((_, live),) = one.live.items()
    _until(lambda: _beat(job, task) >= 2)
    (hosts / "outage-one").write_text(f"files|{jobs}")
    service.write_request(jobs, "other", "auto")
    while not other.live:  # the first host sees nothing, the other takes it back
        with _share_out(jobs):
            one.poll()
        other.poll()
        time.sleep(0.2)
    assert live.process.wait(60) == service.OUTAGE
    output = (one.logs / job.name / f"{task}.{live.claim.token}.log").read_text()
    assert "The share gave no answer within 4 s while" in output  # its lease or step
    assert "this task stops" in output
    with _share_out(jobs):
        one.poll()
    assert one.live  # its claim's state is not known yet
    (hosts / "outage-one").unlink()
    one.poll()
    assert not one.live and not one.exits
    ((_, taken),) = other.live.items()
    (hosts / "project" / "linger").unlink()
    assert taken.process.wait(60) == 0
    other.poll()
    accepted = queue.accepted(job, task)
    assert accepted == {taken.claim.token} and live.claim.token not in accepted


def test_a_paused_owner_returning_after_its_claim_was_taken_stays_revoked(hosts):
    """A task process paused for longer than STALE, e.g. on a sleeping host, finds
    its marker gone when it resumes: it stops, revoked, and commits nothing; the
    host that took the claim back completes the task."""
    jobs, task = hosts / "jobs", "lingering.r2.0000"
    job = _lingering_job(hosts)
    one = service.Service(jobs, "one", 1, hosts / "logs")
    other = service.Service(jobs, "other", 1, hosts / "logs")
    one.poll()
    ((_, live),) = one.live.items()
    _until(lambda: _beat(job, task) >= 2)
    paused = psutil.Process(queue.lease(job, task, live.claim.token)["pid"])
    paused.suspend()
    try:
        _until(lambda: other.poll() or other.live, 60)
    finally:
        paused.resume()
    assert live.process.wait(60) == jobs_main.REVOKED
    one.poll()
    assert not one.live and not one.exits
    ((_, taken),) = other.live.items()
    (hosts / "project" / "linger").unlink()
    assert taken.process.wait(60) == 0
    other.poll()
    assert queue.accepted(job, task) == {taken.claim.token}


@pytest.mark.parametrize("ended", ["outage", "survivor"])
def test_a_task_stopped_by_an_outage_is_returned_uncounted(tmp_path, ended):
    """A task process that the share's outage stopped is returned to the queue by
    its service once it sees the claim again, and counts as no unmarked exit; so
    is a survivor of an earlier service, whose exit cannot be known."""
    config, selection = _project(tmp_path)
    one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
    job = create_job(tmp_path / "jobs", (NUMBERS,), one, config)
    running = service.Service(tmp_path / "jobs", "here", 0, tmp_path / "logs")
    running.poll()
    claim = queue.claim(job, "numbers.r2.0000", "here")
    code = f"import os; os._exit({service.OUTAGE})"
    process = subprocess.Popen([sys.executable, "-c", code])
    assert process.wait(60) == service.OUTAGE
    live = service.Live(claim, 1, process)
    if ended == "survivor":
        live = service.Live(claim, 1, None, process.pid, 1.0)
    running.live[claim.path] = live
    running.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    assert not running.exits and not running.live


def test_a_task_stopped_by_outages_in_a_row_is_shown_with_its_reason(tmp_path):
    """A task whose process the share's outage stopped as many times in a row as
    the host's setting says is shown in the host's record with the reason its
    process gave, and is still run again, never failed; its next other end clears
    it."""
    config, selection = _project(tmp_path)
    one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
    job = create_job(tmp_path / "jobs", (NUMBERS,), one, config)
    running = service.Service(tmp_path / "jobs", "here", 0, tmp_path / "logs", 3)
    running.poll()
    why = "The share gave no answer within 300 s while reading its job (53)."
    shown = []
    for code in (service.OUTAGE,) * 3 + (0,):
        claim = queue.claim(job, "numbers.r2.0000", "here")
        running._log(claim).parent.mkdir(parents=True, exist_ok=True)
        running._log(claim).write_text(f"Starting\n2026-11-15 12:00:00 {why}\n")
        if code == 0:
            queue.finish(claim)
        process = subprocess.Popen(
            [sys.executable, "-c", f"import os; os._exit({code})"]
        )
        assert process.wait(60) == code
        running.live[claim.path] = service.Live(claim, 1, process)
        running.poll()
        record = queue.read_marker(service.machine_files(tmp_path / "jobs", "here")[0])
        shown.append(record["outages"])
    entry = {"job": job.name, "task": "numbers.r2.0000", "times": 3}
    assert shown == [[], [], [{**entry, "reason": f"2026-11-15 12:00:00 {why}"}], []]
    assert not running.exits and not queue.markers(job, queue.FAILED)
    assert queue.markers(job, queue.DONE)


def test_a_host_record_out_of_reach_keeps_the_fence(tmp_path, monkeypatch, started):
    """A service kept from its host's name by another's record stays so while
    the share is out: a record it cannot read is not taken for none."""
    jobs = tmp_path / "jobs"
    _job(tmp_path, (LIGHT,), runs=("r2",))
    path = service.machine_files(jobs, "here")[0]
    path.parent.mkdir()
    other = {"token": "0123456789abcdef", "host": "elsewhere", "pid": 8, "birth": 1.0}
    path.write_text(json.dumps(other) + "\n")
    monkeypatch.setattr(results, "GRACE", 0.2)
    monkeypatch.setattr(results, "RETRY_DELAYS", (0.05, 0.1, 0.2))
    running, _ = _service(tmp_path, monkeypatch)
    running.poll()
    assert not running.held
    with _share_out(jobs):
        running.poll()
    assert not running.held and not started


@pytest.mark.parametrize(
    "where, what",
    [
        ("start", "reading its job"),
        ("read", "running the step"),
        ("mark", "marking the task"),
    ],
)
def test_a_long_outage_at_a_tasks_start_read_or_mark_stops_it_uncounted(
    hosts, where, what
):
    """The share is out for longer than the grace while a task process reads its
    job, while its step reads an input, or while it moves its claim to done: the
    task stops, says why and leaves no mark; its service returns it uncounted
    once the share is back, and it then completes."""
    jobs, flag = hosts / "jobs", hosts / "outage-one"
    job = _lingering_job(hosts, steps=(LINGERING, GATHERED))
    (hosts / "project" / "linger").unlink()
    one = service.Service(jobs, "one", 1, hosts / "logs")
    if where == "read":  # the input is stored before its reader starts
        one.poll()
        ((_, first),) = one.live.items()
        assert first.process.wait(60) == 0
    flag.write_text(
        {
            "start": f"files|{jobs}",
            "read": f"files|{hosts / 'project' / 'results'}",
        }.get(where, f"renames|{jobs}")
    )
    one.poll()
    ((_, live),) = one.live.items()
    assert live.process.wait(120) == service.OUTAGE
    output = one.logs / job.name / f"{live.claim.task}.{live.claim.token}.log"
    assert f"The share gave no answer within 4 s while {what}" in output.read_text()
    assert not queue.markers(job, queue.DONE).get(live.claim.task)
    flag.unlink()
    _until(lambda: (one.poll(), service.settled(jobs))[1])
    assert not one.exits and not queue.markers(job, queue.FAILED)
    assert set(queue.markers(job, queue.DONE)) == {"lingering.r2.0000", "gathered.r2"}


@pytest.mark.skipif(os.name != "nt", reason="Windows' network errors")
@pytest.mark.parametrize(
    "code", [53, 64, 67, 70, 1167, 1201, 1203, 1222, 1226, 1229, 1232, 1233, 1235]
)
def test_the_networks_errors_are_retried_for_the_grace(monkeypatch, code):
    """Each error Windows gives for a share that is out is retried until the
    grace, not for the few seconds of another transient error."""
    clock = [0.0]
    monkeypatch.setattr(results, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        results, "sleep", lambda seconds: clock.append(clock.pop() + seconds)
    )

    def out():
        raise OSError(None, "The share is out", "file", code)

    with pytest.raises(OSError):
        results.retry_transient(out)
    assert results.GRACE - results.RETRY_DELAYS[-1] < clock[0] <= results.GRACE


def test_a_folder_out_of_reach_is_not_settled(tmp_path, monkeypatch):
    """While the jobs folder cannot be read, no job is settled."""
    jobs, job = tmp_path / "jobs", _job(tmp_path, (LIGHT,), runs=("r2",))
    assert not service.settled(jobs)  # its task waits
    queue.finish(queue.claim(job, "light.r2.0000", "here"))
    assert service.settled(jobs)
    monkeypatch.setattr(results, "GRACE", 0.2)
    monkeypatch.setattr(results, "RETRY_DELAYS", (0.05, 0.1, 0.2))
    with _share_out(jobs):
        assert not service.settled(jobs)


def test_service_discovery_skips_done_jobs_without_hiding_retries(
    tmp_path, monkeypatch, started
):
    """The running service remembers terminal success, while failed tasks can
    return and use its capacity after a retry."""
    job = _job(tmp_path, (LIGHT,), runs=("r2",))
    one, _ = _service(tmp_path, monkeypatch)
    one.poll()
    assert len(started) == 1
    started[0].end()
    one.poll()
    assert one._complete == {job.name}
    assert not one._jobs()
    failed = _job(tmp_path, (LIGHT,), runs=("r2",), name="failed")
    queue.finish(queue.claim(failed, "light.r2.0000", "here"), traceback="failed")
    one.poll()
    assert failed.name not in one._complete
    assert queue.retry_failed(failed, "light.r2.0000")
    one.poll()
    assert len(started) == 2
    assert Path(started[-1].job) == failed


def test_a_partial_task_listing_cannot_make_a_job_permanently_complete(tmp_path):
    """A remaining marker prevents caching even if a caller knows only an
    already completed subset of the job's tasks."""
    job = _job(tmp_path, (LIGHT,))
    known = queue.tasks(job)
    first = next(iter(known))
    queue.finish(queue.claim(job, first, "here"))
    assert not service._all_done(job, {first: known[first]})
    assert not service._all_done(job, {})


def test_simultaneous_partial_task_and_state_listings_are_not_cached(
    tmp_path, monkeypatch
):
    """The immutable record still names a runnable task hidden by both directory
    listings; neither the settled check nor discovery may forget that task."""
    job = _job(tmp_path, (LIGHT,))
    known = queue.tasks(job)
    first = next(iter(known))
    queue.finish(queue.claim(job, first, "here"))
    complete = set()
    tasks, markers = queue.tasks, queue.markers
    with monkeypatch.context() as partial:
        partial.setattr(queue, "tasks", lambda folder: {first: known[first]})
        partial.setattr(
            queue,
            "markers",
            lambda folder, state: markers(folder, state) if state == queue.DONE else {},
        )
        assert not service.settled(job.parent, complete=complete)
        one = service.Service(job.parent, "here", 1, tmp_path / "logs")
        assert [folder for folder, _ in one._jobs()] == [job]
        assert not one._complete
    assert not complete
    assert not service.settled(job.parent, complete=complete)
    # Older records without the immutable manifest retain no permanent cache.
    record = read_job(job)
    record.pop("bytes")
    queue.write_json(job / "job.json", record)
    for task in queue.ready(job, tasks(job)):
        queue.finish(queue.claim(job, task.id, "here"))
    assert service.settled(job.parent, complete=complete)
    assert not complete


def _end(process):
    """End a process this test started, with the processes it started, if it
    still runs."""
    if process.poll() is not None:
        return
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            child.kill()
    except psutil.NoSuchProcess:
        pass
    process.kill()
    process.wait(30)


def _record(jobs, machine):
    """A host's record, read again while its service rewrites it in place."""
    for _ in range(50):
        try:
            return queue.read_marker(service.machine_files(jobs, machine)[0])
        except ValueError:
            time.sleep(0.05)
    raise AssertionError(f"The record of {machine} does not read whole")


def _serving(jobs, tmp_path, machine):
    """A host's service as its own process, with its own lock file and its output
    in ``<machine>.out``, until every job is settled."""
    command = [sys.executable, "-m", "darsia.jobs", "serve", str(jobs), "--machine"]
    command += [machine, "--lock", str(tmp_path / f"{machine}.lock"), "--logs"]
    command += [str(tmp_path / "logs"), "--poll", "0.2", "--threads", "1"]
    with open(tmp_path / f"{machine}.out", "wb") as output:
        return subprocess.Popen(
            command + ["--until-settled"], stdout=output, stderr=subprocess.STDOUT
        )


def test_an_idle_host_takes_another_jobs_ready_tasks(
    tmp_path, importable  # noqa: F811
):
    """Two hosts' services, each a process with its own lock file and name: while
    the second host runs the older job's only ready task, which the rest of that
    job waits for, the idle first host runs every task of a younger job, each in
    that job's own environment. Then both jobs complete."""
    jobs = tmp_path / "jobs"
    older = _lingering_job(tmp_path / "older", jobs, (LINGERING, GATHERED))
    config, selection = _project(tmp_path / "younger", kernel="SandyBridge")
    runs = Selection({"r1": selection.frames["r1"]}, selection.sessions)
    services = [_serving(jobs, tmp_path, "two")]
    try:
        _until(lambda: _beat(older, "lingering.r2.0000") >= 1)
        younger = create_job(jobs, (NUMBERS,), runs, config)
        services.append(_serving(jobs, tmp_path, "one"))
        _until(lambda: not queue.waiting(younger), 240)
        made = queue.markers(younger, queue.DONE)
        for task, token in made.items():
            marker = queue.read_marker(younger / queue.DONE / f"{task}.{token}")
            assert marker["machine"] == "one"
        assert set(made) == {"numbers.r1.0000", "numbers.r1.0001"}
        claimed = queue.markers(older, queue.CLAIMED)
        assert set(claimed) == {"lingering.r2.0000"}
        lease = queue.lease(older, "lingering.r2.0000", claimed["lingering.r2.0000"])
        assert lease["machine"] == "two"
        record = _record(jobs, "two")
        assert [live["task"] for live in record["live"]] == ["lingering.r2.0000"]
        stored = _stored(tmp_path / "younger" / "project", "numbers/r1/day1/r1-00.npz")
        assert stored[0]["environment"]["blas_kernel"] == "SandyBridge"
        (tmp_path / "older" / "project" / "linger").unlink()
        for machine, process in zip(("two", "one"), services):
            output = (tmp_path / f"{machine}.out").read_text(errors="replace")
            assert process.wait(240) == 0, output[-3000:]
        assert set(queue.markers(older, queue.DONE)) == {
            "lingering.r2.0000",
            "gathered.r2",
        }
    finally:  # a lingering task ends by itself, also if its service died
        (tmp_path / "older" / "project" / "linger").unlink(missing_ok=True)
        for process in services:
            _end(process)


def test_a_steps_measured_peak_raises_what_its_later_tasks_reserve(
    tmp_path, monkeypatch, started
):
    """The most an ended task's processes held is kept for its step in the host's
    record, with its estimate, its job's sizes and its photos' format; a later task
    of the step reserves that peak scaled to its own estimate, being larger, and a
    later service of the host takes it up at the same revision only."""
    jobs = tmp_path / "jobs"
    job = _job(tmp_path, (LIGHT,), runs=("r1", "r2"))

    def looked(live):  # its processes hold 1 GiB now and held 3 GiB at the most
        live.peak = max(live.peak, 3 * GIB)
        return GIB

    monkeypatch.setattr(service.Live, "resident", looked)
    running, _ = _service(tmp_path, monkeypatch, threads=1)
    running._identity["pid"] = 999_999  # it ends, as its record will say
    running.poll()
    assert [live.bytes for live in running.live.values()] == [GIB]  # its estimate
    running.poll()
    started[0].end()
    running.poll()
    peak = {
        "step": "light",
        "bytes": 3 * GIB,
        "estimate": GIB,
        "base": 0,
        "job": job.name,
        "task": started[0].task,
        "itemsize": 1,
        "camera": 0,
        "grid": GIB,
        "format": ".npz",
    }
    assert _record(jobs, "here")["peaks"] == [peak]
    assert [live.bytes for live in running.live.values()] == [3 * GIB]
    assert service.need(2 * GIB, peak) == 6 * GIB and service.need(GIB, None) == GIB
    again = service.Service(jobs, "here", 0, tmp_path / "logs")
    again.poll()
    assert again.held and list(again.peaks.values()) == [peak]
    other = service.Service(jobs, "here", 0, tmp_path / "logs")
    other.revision = "another"
    other._identity["pid"] = 999_999
    again._identity["pid"] = 999_999
    again.poll()  # its record now names a process that is gone
    other.poll()
    assert other.held and other.peaks == {}


def test_learned_admission_keeps_the_process_base_and_smaller_peak(monkeypatch):
    """Array growth scales the observed array peak; the interpreter stays fixed,
    and a smaller workload retains its observed floor until measured itself."""
    monkeypatch.setattr(service, "PROCESS", 100)
    peak = {"estimate": 200, "bytes": 400, "base": 100}
    assert service.need(300, peak) == 700
    assert service.need(150, peak) == 400
    assert service.need(500, {"estimate": 100, "bytes": 450, "base": 100}) == 500


def test_peak_records_keep_distinct_workloads_and_ignore_bad_measurements(tmp_path):
    """A lower relative peak at a larger resolution must not erase a smaller
    workload's larger scaling factor; malformed records cannot break admission."""
    running = service.Service(tmp_path / "jobs", "here", 0, tmp_path / "logs")
    small = dict(
        step="maps",
        bytes=800,
        estimate=200,
        base=0,
        camera=100,
        grid=50,
        itemsize=4,
        format=".dng",
    )
    large = dict(small, bytes=1000, estimate=400, camera=200, grid=100)
    running._keep(small)
    running._keep(large)
    running._keep(dict(small, bytes=400))
    assert sorted(p["bytes"] for p in running.peaks.values()) == [800, 1000]
    for bad in (
        None,
        {},
        dict(small, bytes=-1),
        dict(small, estimate=float("inf")),
        dict(small, camera="unknown"),
        dict(small, base=-1),
    ):
        running._keep(bad)
    assert len(running.peaks) == 2


def test_a_learned_peak_raises_live_peer_reservations(tmp_path, monkeypatch, started):
    """A live peer that has not reached its peak still reserves the learned cost,
    leaving that future growth out of the space offered to a new task."""
    _job(tmp_path, (LIGHT,), runs=("r1", "r2"))
    running, _ = _service(tmp_path, monkeypatch, threads=3, free=10 * GIB)
    running.poll()
    assert len(running.live) == 3
    observed = list(running.live.values())[-1]

    def looked(live):
        live.peak = 3 * GIB if live is observed else GIB
        return GIB

    monkeypatch.setattr(service.Live, "resident", looked)
    running._reap()
    assert [live.bytes for live in running.live.values()] == [3 * GIB] * 3
