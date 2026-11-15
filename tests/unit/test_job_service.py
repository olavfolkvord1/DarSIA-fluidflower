"""Tests of the job engine's service: task processes, environments and hosts."""

import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import darsia.jobs.__main__ as jobs_main
from darsia.jobs import queue, service
from darsia.jobs.job import create_job, read_job
from darsia.jobs.step import Selection
from darsia.utils import results
from tests.unit.test_jobs import no_share_wait  # noqa: F401
from tests.unit.test_jobs import NUMBERS, SUMS, _project, _stored

ROOT = Path(__file__).resolve().parents[2]


def numpys_blas() -> list[dict]:
    """The BLAS libraries of this process that NumPy runs on (threadpoolctl's rows):
    those in NumPy's installation, inside it or in its wheel's ``numpy.libs``; every
    BLAS library of this process where none is there. SciPy's wheel brings one of the
    same builds. Another package may bring a BLAS of its own, e.g. OpenCV's Linux
    wheel an OpenBLAS for its LAPACK, which can take another kernel for this CPU."""
    rows = [row for row in threadpool_info() if row.get("user_api") == "blas"]
    package = Path(os.path.realpath(np.__file__)).parent
    places = {package, package.with_name("numpy.libs")}
    own = []
    for row in rows:
        if places & set(Path(os.path.realpath(row["filepath"])).parents):
            own.append(row)
    return own or rows


HERE = sorted(
    {r["architecture"] for r in numpys_blas() if r.get("internal_api") == "openblas"}
)[0]
"""This process's BLAS kernel, NumPy's, for a task run in this process."""


def _end_task_processes() -> list[int]:
    """End the task processes this test's service started, and wait for them."""
    ended = []
    for process in psutil.Process().children(recursive=True):
        try:
            if {"darsia.jobs", "run"} <= set(process.cmdline()):
                process.kill()
                process.wait(30)
                ended.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    return ended


def _serve(jobs, tmp_path, threads=2, seconds=300.0):
    """Serve a jobs folder in this process until every job is settled; after some
    seconds (a job the service refuses is never settled, a task may hang) the task
    processes still running are ended and the test fails."""
    deadline, overdue = time.monotonic() + seconds, []

    def until():
        if service.settled(jobs):
            return True
        if time.monotonic() > deadline:
            overdue.append(_end_task_processes())
            return True
        return False

    service.serve(
        jobs,
        "here",
        threads=threads,
        lock=tmp_path / "service.lock",
        logs=tmp_path / "logs",
        poll=0.2,
        until=until,
    )
    assert not overdue, f"The jobs did not settle within {seconds} s."


@pytest.fixture
def kept_threads(monkeypatch):
    """A task run in this process leaves this process's thread limits as they
    were. Its job's kernel is HERE. Where another OpenBLAS of this process took
    another kernel for this CPU when it loaded (:func:`numpys_blas`), no one kernel
    is this process's, so the task's check of its job's kernel, which a service's
    task process passes since its kernel is set before any OpenBLAS loads, is
    passed over here."""
    kernels = {
        r["architecture"]
        for r in threadpool_info()
        if r.get("internal_api") == "openblas"
    }
    if kernels != {HERE}:
        monkeypatch.setattr(jobs_main, "refusal", lambda environment: None)
    opencv = cv2.getNumThreads()
    with threadpool_limits(limits=None):
        yield
    cv2.setNumThreads(opencv)


@pytest.fixture
def importable(monkeypatch):
    """Task processes find the test steps and the presets' steps."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT)]))


def test_two_jobs_keep_their_own_environment(tmp_path, importable):
    """Two jobs of other kernels and threads share one service; each result
    records, and was made under, its own job's."""
    jobs = tmp_path / "jobs"
    made = []
    for kernel, threads in (("Haswell", 1), ("SandyBridge", 2)):
        config, selection = _project(tmp_path, kernel=kernel, threads=threads)
        one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
        made.append(
            create_job(jobs, (NUMBERS,), one, config, project=tmp_path / kernel)
        )
    _serve(jobs, tmp_path)
    for job, (kernel, threads) in zip(made, (("Haswell", 1), ("SandyBridge", 2))):
        assert queue.waiting(job) == set() and not queue.markers(job, queue.FAILED)
        description, values = _stored(tmp_path / kernel, "numbers/r2/day1/r2-00.npz")
        environment = {"blas_kernel": kernel, "threads": threads, "backend": "cpu"}
        assert description["environment"] == environment
        assert values["kernel"] == kernel and values["threads"] == str(threads)
        assert {k.casefold() for k in values["found"]} == {kernel.casefold()}
    # One capacity for both jobs, the oldest first: the second's two threads wait.
    first = _stored(tmp_path / "Haswell", "numbers/r2/day1/r2-02.npz")[1]
    second = _stored(tmp_path / "SandyBridge", "numbers/r2/day1/r2-00.npz")[1]
    assert second["began"] >= first["ended"]


def test_parallel_chunks_overlap_and_linear_frames_come_in_order(tmp_path, importable):
    """A run's two chunks of frames run at once; the run's sum reads its frames in
    time order."""
    jobs = tmp_path / "jobs"
    config, selection = _project(tmp_path, pause=1.5)
    one = Selection({"r1": selection.frames["r1"]}, selection.sessions)
    create_job(jobs, (NUMBERS, SUMS), one, config)
    _serve(jobs, tmp_path)
    project = tmp_path / "project"
    spans = [
        _stored(project, f"numbers/r1/day1/{frame.name}")[1]
        for frame in one.frames["r1"]
    ]
    first, second = spans[:4], spans[4:]
    assert second[0]["began"] < first[-1]["ended"]  # the chunks overlap
    order = _stored(project, "sums/r1")[1]["order"]
    assert order == [frame.name for frame in one.frames["r1"]]


def test_a_failed_console_or_log_write_stops_nothing(tmp_path, importable, monkeypatch):
    """The console raises and the job's shared log cannot be written; the tasks
    still complete."""

    class Broken:
        def write(self, text):
            raise OSError("the console is gone")

        def flush(self):
            raise OSError("the console is gone")

    jobs = tmp_path / "jobs"
    config, selection = _project(tmp_path)
    one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
    job = create_job(jobs, (NUMBERS, SUMS), one, config)
    (job / service.LOG).mkdir()  # a folder cannot be appended to
    monkeypatch.setattr(sys, "stdout", Broken())
    _serve(jobs, tmp_path)
    monkeypatch.undo()
    assert set(queue.markers(job, queue.DONE)) == {"numbers.r2.0000", "sums.r2"}


def _kill(pid):
    """End a process this test started, and wait for it."""
    try:
        process = psutil.Process(pid)
        process.kill()
        process.wait(30)
    except psutil.NoSuchProcess:
        pass


def test_one_service_per_host_and_its_survivors(tmp_path, importable):
    """A second service on a host exits. After a restart, a task the first started
    and that still runs is counted by its process and birth time, not taken back;
    once it is killed, it is returned to the queue at once."""
    lock = tmp_path / "service.lock"
    with service.HostLock(lock):
        with pytest.raises(service.ServiceRunning, match="Another service"):
            service.serve(tmp_path, "here", lock=lock, poll=0.2, until=lambda: True)
    jobs = tmp_path / "jobs"
    config, selection = _project(tmp_path, pause=600.0)
    one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
    job = create_job(jobs, (NUMBERS,), one, config)
    command = [sys.executable, "-m", "darsia.jobs", "serve", str(jobs), "--machine"]
    command += ["here", "--lock", str(lock), "--logs", str(tmp_path / "logs")]
    command += ["--poll", "0.2", "--threads", "2"]
    first = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    tree, lease = [first.pid], {}
    try:
        deadline = time.time() + 120
        while time.time() < deadline and lease.get("beat", 0) < 1:
            time.sleep(0.5)
            claimed = queue.markers(job, queue.CLAIMED)
            if claimed:
                ((task, token),) = claimed.items()
                lease = queue.lease(job, task, token)
        assert lease.get("beat", 0) >= 1, "the task did not start"
        processes = [psutil.Process(first.pid)]
        processes += processes[0].children(recursive=True)
        tree += [process.pid for process in processes]
        for process in processes:  # the first service ends; its task runs on
            try:
                serving = "serve" in process.cmdline()
            except psutil.NoSuchProcess:  # e.g. the service's look at its revision
                continue
            if serving:
                _kill(process.pid)
        with service.HostLock(lock):  # the system released the lock
            pass
        # A capacity of no threads takes no new task: a returned task stays queued.
        again = service.Service(jobs, "here", 0, tmp_path / "logs")
        again.poll()
        assert list(again.live.values())[0].pid == lease["pid"]
        assert not service.alive(lease["pid"], lease["birth"] + 1)  # a reused pid
        assert set(queue.markers(job, queue.CLAIMED)) == {task}
        assert queue.markers(job, queue.QUEUE) == {}
        _kill(lease["pid"])
        again.poll()
        assert set(queue.markers(job, queue.QUEUE)) == {task}
    finally:
        for pid in tree + ([lease["pid"]] if lease.get("pid") else []):
            _kill(pid)


def test_a_job_of_another_revision_is_not_taken(tmp_path, capsys):
    """A service whose code revision differs from the job's takes none of its
    tasks and says why."""
    config, selection = _project(tmp_path)
    job = create_job(tmp_path / "jobs", (NUMBERS,), selection, config)
    record = read_job(job)
    record["revision"] = "an-earlier-revision"
    queue.write_json(job / "job.json", record)
    service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs").poll()
    assert not queue.markers(job, queue.CLAIMED)
    assert "created at revision an-earlier-revision" in capsys.readouterr().out


@pytest.mark.parametrize(
    "kernel, backend, wanted",
    [("NoSuchKernel", "cpu", "NoSuchKernel"), ("Haswell", "opencl-gpu", "opencl-gpu")],
)
def test_a_host_without_the_job_environment_returns_the_task(
    tmp_path, importable, kernel, backend, wanted
):
    """A task process on a host that does not provide its job's BLAS kernel or
    compute backend stops; the service returns the task and takes no more of the
    job's tasks, saying why."""
    config, selection = _project(tmp_path, kernel=kernel, backend=backend)
    one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
    job = create_job(tmp_path / "jobs", (NUMBERS,), one, config)
    again = service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs")
    again.poll()
    ((_, live),) = again.live.items()
    live.process.wait(120)
    again.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    assert f"not the job's {wanted}" in again.refused[job.name]
    assert not again.live


def _one(tmp_path, **project):
    """A job of the first frame of one run, in its jobs folder."""
    config, selection = _project(tmp_path, **project)
    one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
    return create_job(tmp_path / "jobs", (NUMBERS,), one, config)


def test_a_beat_delayed_into_the_mark_lands_before_it(
    tmp_path, monkeypatch, kept_threads
):
    """A beat that the share holds until the step has ended lands before the task
    process marks its task done, never between the mark and its move: the done
    marker names the task's token."""
    job = _one(tmp_path, kernel=HERE, pause=0.3)
    claim = queue.claim(job, "numbers.r2.0000", "here")
    marked, landed = threading.Event(), threading.Event()
    beat, write, transfer = queue.beat, queue._write, queue._transfer

    def held(claim, number, pid, born):
        if number > 1:  # the beat thread's: the share holds it until the mark
            marked.wait(2.0)
        beat(claim, number, pid, born)
        if number > 1:
            landed.set()

    def mark(claim, value, **kwargs):
        write(claim, value, **kwargs)
        if "accepted" in value:
            marked.set()

    def move(source, destination, **kwargs):
        if destination.parent.name == queue.DONE:
            landed.wait(2.0)  # a held beat lands now, if it is still to come
        return transfer(source, destination, **kwargs)

    monkeypatch.setattr(results, "BEAT", 0.05)
    monkeypatch.setattr(queue, "beat", held)
    monkeypatch.setattr(queue, "_write", mark)
    monkeypatch.setattr(queue, "_transfer", move)
    assert jobs_main.run(job, claim.task, claim.token, "here") == 0
    assert landed.is_set()
    assert queue.accepted(job, claim.task) == {claim.token}


def test_a_task_process_of_another_revision_refuses_its_task(tmp_path, kept_threads):
    """A task process whose code is not of its job's revision makes nothing, says
    why in its claim and exits with the refusal's code."""
    job = _one(tmp_path, kernel=HERE)
    record = read_job(job)
    record["revision"] = "an-earlier-revision"
    queue.write_json(job / "job.json", record)
    claim = queue.claim(job, "numbers.r2.0000", "here")
    assert jobs_main.run(job, claim.task, claim.token, "here") == service.REFUSED
    why = queue.lease(job, claim.task, claim.token)["refused"]
    assert "not the job's an-earlier-revision" in why
    assert not (tmp_path / "project" / "results" / "numbers").exists()


def test_the_checkout_advanced_under_a_running_service(
    tmp_path, importable, monkeypatch, capsys
):
    """A service whose checkout has moved on since it started takes no new task; a
    task process it started just before the move refuses its task, and the
    service returns it and takes no more of the job, saying why."""
    job = _one(tmp_path)
    record = read_job(job)
    record["revision"] = "started-at"
    queue.write_json(job / "job.json", record)
    running = service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs")
    running.revision = "started-at"  # the service and the job were made at one
    monkeypatch.setattr(service, "revision", lambda: "moved-to")
    running.poll()
    assert not queue.markers(job, queue.CLAIMED) and not running.live
    assert "has moved from revision started-at to moved-to" in capsys.readouterr().out
    monkeypatch.setattr(service, "revision", lambda: "started-at")
    running.poll()  # started before the move; the process imports the moved code
    ((_, live),) = running.live.items()
    live.process.wait(120)
    running.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    assert "not the job's started-at" in running.refused[job.name]
    assert not running.live
    assert not (tmp_path / "project" / "results" / "numbers").exists()


@pytest.mark.parametrize("code", [3, service.REFUSED])
def test_an_exit_whose_claim_says_no_refusal_is_no_refusal(tmp_path, code):
    """A task process that ends with the C runtime abort's code, or even the
    refusal's, without a refusal in its claim, is an unmarked exit: its task is
    returned once, and its job is not refused."""
    job = _one(tmp_path)
    running = service.Service(tmp_path / "jobs", "here", 0, tmp_path / "logs")
    running.poll()  # no threads: it reads the job and starts nothing
    claim = queue.claim(job, "numbers.r2.0000", "here")
    ended = subprocess.Popen([sys.executable, "-c", f"import os; os._exit({code})"])
    assert ended.wait(60) == code
    running.live[claim.path] = service.Live(claim, 1, ended)
    running.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    assert job.name not in running.refused
    assert running.exits == {(job.name, "numbers.r2.0000"): 1}


def test_a_retried_task_is_returned_once_more_after_an_unmarked_exit(tmp_path):
    """A task failed after two unmarked exits and then retried starts its count
    again: its next unmarked exit returns it to the queue once more."""
    job = _one(tmp_path)
    running = service.Service(tmp_path / "jobs", "here", 0, tmp_path / "logs")
    running.poll()  # no threads: it reads the job and starts nothing
    ended = subprocess.Popen([sys.executable, "-c", "import os; os._exit(1)"])
    assert ended.wait(60) == 1
    name = "numbers.r2.0000"
    for _ in range(2):
        claim = queue.claim(job, name, "here")
        running.live[claim.path] = service.Live(claim, 1, ended)
        running.poll()
    assert set(queue.markers(job, queue.FAILED)) == {name}
    assert queue.retry_failed(job, name)
    claim = queue.claim(job, name, "here")
    running.live[claim.path] = service.Live(claim, 1, ended)
    running.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {name}


def test_a_job_read_short_stops_nothing(tmp_path, capsys):
    """A job's record or task file read short, e.g. just published on the other
    host, is said, and read again in the next round; the service goes on."""
    job = _one(tmp_path)
    record = (job / "job.json").read_bytes()
    (job / "job.json").write_bytes(record[:-20])
    running = service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs")
    running.poll()
    assert job.name not in running._records
    assert f"The job {job.name} cannot be read now" in capsys.readouterr().out
    (job / "job.json").write_bytes(record)
    (task,) = (job / "tasks").iterdir()
    task.write_bytes(task.read_bytes()[:-5])
    running.poll()
    assert job.name in running._records and not running.live
    assert "Starting ready tasks cannot be done now" in capsys.readouterr().out
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}


def test_no_threads_take_no_new_tasks(tmp_path, capsys):
    """A service of no threads takes no new tasks, and says so once."""
    job = _one(tmp_path)
    rounds = []
    service.serve(
        tmp_path / "jobs",
        "here",
        threads=0,
        lock=tmp_path / "service.lock",
        logs=tmp_path / "logs",
        poll=0.01,
        until=lambda: rounds.append(0) or len(rounds) >= 3,
    )
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    out = capsys.readouterr().out
    assert out.count("takes no new tasks: its capacity is no threads") == 1


def test_a_task_whose_process_cannot_start_is_returned(tmp_path, monkeypatch):
    """A claimed task whose process cannot be started is returned at once, and the
    host takes no more of its job's tasks, saying why."""
    job = _one(tmp_path)
    running = service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs")
    monkeypatch.setattr(service, "revision", lambda: running.revision)

    def no_process(*args, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr(service.subprocess, "Popen", no_process)
    running.poll()
    assert set(queue.markers(job, queue.QUEUE)) == {"numbers.r2.0000"}
    assert "could not be started" in running.refused[job.name]
    assert not running.live


_MUTEX = """
import ctypes, sys
from ctypes import wintypes
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.CreateMutexW.restype = wintypes.HANDLE
kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
handle = kernel32.CreateMutexW(None, False, sys.argv[1])
sys.exit(1 if not handle or ctypes.get_last_error() == 183 else 0)
"""


def test_the_host_lock_is_the_machines(tmp_path, monkeypatch):
    """The service's lock is the whole machine's, whatever an account's temporary
    folder: while one process holds it, a second take fails, also in another
    process with another temporary folder; once released, it can be taken."""
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    found = service.host_lock()
    if os.name != "nt":
        assert found.path == service.HOST_FILE
        return
    assert found.name == service.HOST_MUTEX == "Global\\darsia-jobs-service"
    name = f"Global\\darsia-jobs-test-{secrets.token_hex(8)}"
    other = {**os.environ, "TEMP": str(tmp_path / "other"), "TMP": str(tmp_path)}
    with service.HostMutex(name):
        with pytest.raises(service.ServiceRunning, match="Another service"):
            with service.HostMutex(name):
                pass
        command = [sys.executable, "-c", _MUTEX, name]
        assert subprocess.run(command, env=other, timeout=60).returncode == 1
    with service.HostMutex(name):
        pass


def test_a_delayed_renewal_keeps_its_start_time_for_terminal_marking(monkeypatch):
    """A blocked lease write cannot extend the later terminal commit deadline
    by the time its own retries consumed."""
    clock, renewed = [10.0], [0.0]
    waits = iter((False, True))
    stop = SimpleNamespace(wait=lambda seconds: next(waits))
    monkeypatch.setattr(jobs_main, "monotonic", lambda: clock[0])
    monkeypatch.setattr(service, "birth", lambda pid: 1.0)

    def delayed(*args):
        clock[0] = 20.0

    monkeypatch.setattr(queue, "beat", delayed)
    jobs_main._beats(None, stop, renewed)
    assert renewed == [10.0]


def test_completed_jobs_are_not_read_again_but_failed_jobs_can_be_retried(
    tmp_path, monkeypatch
):
    """Legacy records stay uncached; retrying failed work stays visible."""
    jobs = tmp_path / "jobs"

    def make(name):
        config, selection = _project(tmp_path / name)
        one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
        return create_job(jobs, (NUMBERS,), one, config)

    first = make("first")
    queue.finish(queue.claim(first, "numbers.r2.0000", "here"))
    complete = set()
    assert service.settled(jobs, complete=complete)
    assert not complete
    failed = make("second")
    queue.finish(queue.claim(failed, "numbers.r2.0000", "here"), traceback="failed")
    tasks = queue.tasks

    def read(job):
        return tasks(job)

    monkeypatch.setattr(queue, "tasks", read)
    assert service.settled(jobs, complete=complete)
    assert not complete
    task = "numbers.r2.0000"
    token = queue.markers(failed, queue.FAILED)[task]
    assert queue._transfer(
        failed / queue.FAILED / f"{task}.{token}",
        failed / queue.QUEUE / f"{task}.{queue.new_token()}",
    )
    assert not service.settled(jobs, complete=complete)
    queue.finish(queue.claim(failed, "numbers.r2.0000", "here"))
    assert service.settled(jobs, complete=complete)
    assert not complete
    make("third")
    assert not service.settled(jobs, complete=complete)
