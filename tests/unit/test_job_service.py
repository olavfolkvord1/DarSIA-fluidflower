"""Tests of the job engine's service: task processes, environments and hosts."""

import ctypes
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import psutil
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import darsia.jobs.__main__ as jobs_main
from darsia.jobs import queue, service
from darsia.jobs.job import Sizes, create_job, read_job
from darsia.jobs.step import Memory, Order, Selection, Step
from darsia.utils import results
from darsia.utils.gpu import backend as gpu_backend
from darsia.utils.gpu import opencl
from tests.unit.test_jobs import no_share_wait  # noqa: F401
from tests.unit.test_jobs import (
    NUMBERS,
    SUMS,
    ToySettings,
    _files,
    _numbers,
    _project,
    _stored,
)
from tests.unit.test_minimal_chain import _head

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
    processes still running are ended and the test fails. The host's free memory
    is taken as ample, so that the test does not depend on this machine's."""
    deadline, overdue = time.monotonic() + seconds, []

    def until():
        if service.settled(jobs):
            return True
        if time.monotonic() > deadline:
            overdue.append(_end_task_processes())
            return True
        return False

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service, "headroom", lambda: 2**40)
        patch.setattr(service, "reserve", lambda: 0)
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


@pytest.mark.parametrize("hidden", [psutil.NoSuchProcess, psutil.AccessDenied])
def test_a_survivor_gone_or_hidden_is_gone_and_a_live_one_keeps_its_threads(
    tmp_path, monkeypatch, capsys, hidden
):
    """A restarted service's look at its host's claims takes a process that is
    gone, or that it may not look at, for gone: it returns that task at once and
    goes on. A claim whose process still runs is counted, and its threads leave
    no room for a new task."""
    config, selection = _project(tmp_path)
    job = create_job(tmp_path / "jobs", (NUMBERS,), selection, config)
    gone = queue.claim(job, "numbers.r1.0000", "here")
    kept = queue.claim(job, "numbers.r2.0000", "here")
    host = service.host_identity()
    queue.beat(gone, 1, 999_999, 1.0e9, host)
    queue.beat(kept, 1, os.getpid(), service.birth(os.getpid()), host)
    real = psutil.Process

    class Hidden:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            raise hidden(self.pid)

    def process(pid):
        return Hidden(pid) if pid == 999_999 else real(pid)

    monkeypatch.setattr(service.psutil, "Process", process)
    again = service.Service(tmp_path / "jobs", "here", 1, tmp_path / "logs")
    again.poll()
    assert "cannot be done now" not in capsys.readouterr().out
    assert [live.pid for live in again.live.values()] == [os.getpid()]
    assert set(queue.markers(job, queue.CLAIMED)) == {"numbers.r2.0000"}
    assert "numbers.r1.0000" in queue.markers(job, queue.QUEUE)


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
    [
        ("NoSuchKernel", "cpu", "not the job's NoSuchKernel"),
        ("Haswell", "opencl-gpu", "the job's backend opencl-gpu: No compute backend"),
    ],
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
    assert wanted in again.refused[job.name]
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

    def held(claim, number, pid, born, *host):
        if number > 1:  # the beat thread's: the share holds it until the mark
            marked.wait(2.0)
        beat(claim, number, pid, born, *host)
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
    jobs_main._beats(None, stop, "here", renewed)
    assert renewed == [10.0]


def test_a_task_runs_on_its_job_backend(tmp_path, monkeypatch, kept_threads):
    """A task process takes its job's compute backend by its label, checked with
    the job's phase-model heads, runs the step on it and records its device with
    each result; a host whose device of that class fails the check refuses the
    task, saying why."""
    job = _one(tmp_path, kernel=HERE, backend="opencl:Device")
    model = tmp_path / "project" / "model" / "models" / "generic"
    model.mkdir(parents=True)
    for role, intercept in (("co2_presence", -3.0), ("gas_within_co2", -2.5)):
        (model / f"{role}_signal.json").write_text(json.dumps(_head(20, intercept)))
    policy = {"models": {"generic": {"gas_within_co2": {"presence_threshold": 0.66}}}}
    (model.parents[1] / "phase_decision_policy.json").write_text(json.dumps(policy))
    config = tmp_path / "jobs" / job.name / "config-0.toml"
    config.write_text(
        config.read_text() + '[phase_model]\nfolder = "model"\npixel_size_mm = 0.5\n'
    )
    device = SimpleNamespace(label="opencl:Device", device="Device (driver 1)")
    asked, used = [], []

    def named(label, heads=None):
        asked.append((label, heads))
        if len(asked) > 1:
            raise ValueError("none gives the CPU's numbers.")
        return device

    monkeypatch.setattr(jobs_main, "backend_named", named)
    monkeypatch.setattr(jobs_main, "use_compute_backend", used.append)
    monkeypatch.setattr(jobs_main, "compute_backend", lambda: used[-1])
    claim = queue.claim(job, "numbers.r2.0000", "here")
    assert jobs_main.run(job, claim.task, claim.token, "here") == 0
    ((label, heads),) = asked
    assert label == "opencl:Device" and used == [device]
    assert [head.intercept for head in heads] == [-3.0, -2.5]
    stored, _ = _stored(tmp_path / "project", "numbers/r2/day1/r2-00.npz")
    assert stored["device"] == "Device (driver 1)"
    assert stored["environment"]["backend"] == "opencl:Device"
    again = _one(tmp_path / "again", kernel=HERE, backend="opencl:Device")
    claim = queue.claim(again, "numbers.r2.0000", "here")
    assert jobs_main.run(again, claim.task, claim.token, "here") == service.REFUSED
    why = queue.lease(again, claim.task, claim.token)["refused"]
    assert why == (
        "This host cannot run the job's backend opencl:Device: none gives the CPU's "
        "numbers."
    )


def test_auto_backend_uses_each_steps_capability_and_records_hardware(
    tmp_path, monkeypatch, kept_threads
):
    accelerated = replace(NUMBERS, name="accelerated-numbers", accelerated=True)
    monkeypatch.setitem(jobs_main.STEPS, accelerated.name, accelerated)
    config, selection = _project(tmp_path, kernel=HERE, backend="auto")
    policy = tmp_path / "policy.toml"
    policy.write_text(
        '[jobs.qualified_backends]\nnumbers = ["opencl:Device"]\n'
        'accelerated-numbers = ["opencl:Device"]\n',
        encoding="utf-8",
    )
    one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
    job = create_job(
        tmp_path / "jobs",
        (NUMBERS, accelerated),
        one,
        [config, policy],
        sizes=Sizes(4, 6, 6, (2, 3)),
    )
    gpu = SimpleNamespace(
        label="opencl:Device",
        device="Device | driver 1",
        hardware=dict(backend="opencl", platform="Vendor", model="Device", driver="1"),
    )
    calls, used = [], []

    def choose(shape, dtype, *, heads, measured):
        calls.append((shape, dtype, measured))
        return gpu, "qualified"

    monkeypatch.setattr(jobs_main, "choose_compute_backend", choose)
    monkeypatch.setattr(jobs_main, "use_compute_backend", used.append)
    monkeypatch.setattr(jobs_main, "compute_backend", lambda: used[-1])
    monkeypatch.setattr(jobs_main, "cpu_model", lambda: "Test CPU")
    for step in (NUMBERS, accelerated):
        claim = queue.claim(job, f"{step.name}.r2.0000", "here")
        assert jobs_main.run(job, claim.task, claim.token, "here") == 0
    assert calls == [((2, 3), np.float32, ["opencl:Device"])]
    assert [device.label for device in used] == ["cpu", "opencl:Device"]
    cpu_record, _ = _stored(tmp_path / "project", "numbers/r2/day1/r2-00.npz")
    assert cpu_record["hardware"]["model"] == "Test CPU"
    assert cpu_record["hardware"]["blas"]
    gpu_record, _ = _stored(
        tmp_path / "project", "accelerated-numbers/r2/day1/r2-00.npz"
    )
    assert gpu_record["hardware"] == gpu.hardware
    assert cpu_record["environment"]["backend"] == "auto"
    assert gpu_record["environment"]["backend"] == "auto"


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


def test_a_waiting_lock_waits_for_its_holder():
    """A second taker of a host's waiting lock waits until the first releases it,
    and the first may then take it again."""
    name = f"jobs-test-{secrets.token_hex(8)}"
    first, second = service.WaitingLock(name), service.WaitingLock(name)
    held, times = threading.Event(), {}

    def other():
        held.wait(30)
        with second:
            times["taken"] = time.monotonic()

    thread = threading.Thread(target=other)
    thread.start()
    with first:
        held.set()
        time.sleep(0.5)
        times["released"] = time.monotonic()
    thread.join(30)
    assert times["taken"] >= times["released"]
    with first:
        pass


_HOLD = """
import os, sys, time
from darsia.jobs import service
lock = service.WaitingLock(sys.argv[1])
lock.__enter__()
print("held", flush=True)
time.sleep(float(sys.argv[2]))
if sys.argv[3] == "abandon":
    os._exit(0)
lock.__exit__()
"""


@pytest.mark.parametrize("ending", ["release", "abandon"])
def test_a_waiting_lock_held_by_another_process_is_taken_once_it_lets_go(
    importable, ending
):
    """A waiting lock held by another process is taken once that process releases
    it, or dies holding it: a dead holder hands it on."""
    name = f"jobs-test-{secrets.token_hex(8)}"
    command = [sys.executable, "-c", _HOLD, name, "1.0", ending]
    holder = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        began = time.monotonic()
        with service.WaitingLock(name):
            waited = time.monotonic() - began
        assert waited > 0.5
    finally:
        holder.wait(60)
        holder.stdout.close()


def test_completed_jobs_are_not_read_again_but_failed_jobs_can_be_retried(
    tmp_path, monkeypatch
):
    """Only fully done jobs are final. A retry of a failed job and a new job
    remain visible while an older done job no longer costs marker/task reads."""
    jobs = tmp_path / "jobs"

    def make(name):
        config, selection = _project(tmp_path / name)
        one = Selection({"r2": selection.frames["r2"][:1]}, selection.sessions)
        return create_job(jobs, (NUMBERS,), one, config)

    first = make("first")
    queue.finish(queue.claim(first, "numbers.r2.0000", "here"))
    complete = set()
    assert service.settled(jobs, complete=complete)
    assert complete == {first.name}
    failed = make("second")
    queue.finish(queue.claim(failed, "numbers.r2.0000", "here"), traceback="failed")
    tasks = queue.tasks

    def read(job):
        assert Path(job) != first, "A fully done job was read again"
        return tasks(job)

    monkeypatch.setattr(queue, "tasks", read)
    assert service.settled(jobs, complete=complete)
    assert complete == {first.name}
    task = "numbers.r2.0000"
    token = queue.markers(failed, queue.FAILED)[task]
    assert queue._transfer(
        failed / queue.FAILED / f"{task}.{token}",
        failed / queue.QUEUE / f"{task}.{queue.new_token()}",
    )
    assert not service.settled(jobs, complete=complete)
    queue.finish(queue.claim(failed, "numbers.r2.0000", "here"))
    assert service.settled(jobs, complete=complete)
    assert complete == {first.name, failed.name}
    make("third")
    assert not service.settled(jobs, complete=complete)


GIB = 2**30
"""A gibibyte."""


# Eligibility before memory: a host refuses a refused computing environment.

SIZES = Sizes(1, 0, GIB)
"""Sizes in which a step's grid arrays are GiB."""
ARRAYS = Step(
    "arrays",
    {},
    Order.FRAMES,
    Memory(grid=2.0),
    _numbers,
    {"toy": ToySettings},
    sections=("protocols",),
    files=_files,
)
"""A step over frames whose task declares 2 GiB."""


def _arrays(tmp_path, name, **project):
    """A job of the arrays step over both runs of a project of its own: three
    tasks of one workload."""
    config, selection = _project(tmp_path / name, **project)
    return create_job(tmp_path / "jobs", (ARRAYS,), selection, config, sizes=SIZES)


class _Process:
    """A task process of a test, which holds no memory: it runs until the test ends
    it; one of a job the test refuses says why in its claim and exits at once, as
    a task process does (:func:`darsia.jobs.__main__.run`)."""

    def __init__(self, found, command):
        job, task, token = Path(command[4]), command[5], command[6]
        self.claim = queue.Claim(job, task, token, "here")
        self.pid, self.code = 999_999, None
        found.started.append(self)
        if job.name in found.refusals:
            queue.refuse(self.claim, found.refusals[job.name])
            self.code = service.REFUSED

    def poll(self):
        """Its exit code, once it has ended."""
        return self.code

    def end(self):
        """Mark its task done and exit with code 0."""
        queue.finish(self.claim)
        self.code = 0


@pytest.fixture
def processes(monkeypatch):
    """The task processes a test's services start (:class:`_Process`), and the
    refusals of the jobs whose processes refuse them, by job name."""
    found = SimpleNamespace(started=[], refusals={})
    starting = SimpleNamespace(
        Popen=lambda command, **_: _Process(found, command),
        DEVNULL=subprocess.DEVNULL,
        STDOUT=subprocess.STDOUT,
    )
    monkeypatch.setattr(service, "subprocess", starting)
    return found


def _serving(tmp_path, monkeypatch, threads, peak=None):
    """A service of this host over the test's jobs folder, with ample memory, no
    reserve and no bytes of a task process's own, whose look at a live task finds
    its processes holding nothing now and ``peak["bytes"]`` at the most; nothing
    once its process has ended."""
    running = service.Service(tmp_path / "jobs", "here", threads, tmp_path / "logs")
    monkeypatch.setattr(service, "revision", lambda: running.revision)
    monkeypatch.setattr(service, "headroom", lambda: 100 * GIB)
    monkeypatch.setattr(service, "reserve", lambda: 0)
    monkeypatch.setattr(service, "PROCESS", 0)

    def looked(live):
        if peak is not None and (live.process is None or live.process.poll() is None):
            live.peak = max(live.peak, peak["bytes"])
        return 0

    monkeypatch.setattr(service.Live, "resident", looked)
    return running


def test_a_refused_environment_refuses_its_later_jobs_before_any_claim(
    tmp_path, monkeypatch, processes
):
    """After a task process refused its job for want of its compute backend's
    device on this host, a later job of the same BLAS kernel and backend is
    refused before any claim, look at its memory or start, and never becomes the
    reason the host waits, while a later job of the CPU starts in the same round;
    the host's record shows the environment refused."""
    monkeypatch.setattr(opencl, "opencl_devices", lambda: [])  # no such device here
    first = _arrays(tmp_path, "first", backend="opencl:Absent")
    with pytest.raises(ValueError) as refused:
        jobs_main.backend_of(first, read_job(first))
    why = str(refused.value)
    assert why == (
        "This host cannot run the job's backend opencl:Absent: opencl:Absent: no "
        "such device here."
    )
    processes.refusals[first.name] = why
    running = _serving(tmp_path, monkeypatch, threads=1)
    running.poll()  # its first task is claimed and started; its process refuses
    running.poll()  # it is returned, and its job and environment refused
    environment = service._environment(read_job(first))
    assert running.unfit == {environment: (first.name, why)}
    assert len(queue.markers(first, queue.QUEUE)) == 3
    same = _arrays(tmp_path, "same", backend="opencl:Absent")
    cpu = _arrays(tmp_path, "cpu")
    reserved, claimed = [], []
    reserve, claim = running._reserve, queue.claim

    def reserving(record, task):
        reserved.append(record["id"])
        return reserve(record, task)

    def claiming(job, *rest):
        claimed.append(Path(job).name)
        return claim(job, *rest)

    monkeypatch.setattr(running, "_reserve", reserving)
    monkeypatch.setattr(queue, "claim", claiming)
    running.poll()
    assert same.name not in reserved and same.name not in claimed
    assert not queue.markers(same, queue.CLAIMED)
    assert len(queue.markers(same, queue.QUEUE)) == 3
    assert running.refused[same.name] == (
        f"Its BLAS kernel {environment[0]} and compute backend opencl:Absent are "
        f"not this host's, as a task of job {first.name} found: {why}"
    )
    assert [process.claim.job for process in processes.started] == [first, cpu]
    assert running.reason.startswith(f"Task arrays.r2.0000 of job {cpu.name} waits")
    path = service.machine_files(tmp_path / "jobs", "here")[0]
    assert json.loads(path.read_text())["unfit"] == [
        {
            "blas_kernel": environment[0],
            "backend": "opencl:Absent",
            "job": first.name,
            "reason": why,
        }
    ]


def test_only_a_refusal_of_the_environment_refuses_its_other_jobs(
    tmp_path, monkeypatch, processes
):
    """A refusal counts for its job's environment only when the task process
    found this host without the job's BLAS kernel, or its compute backend's label
    or device: one of the code revision, of a device of this host that gave other
    numbers with the job's phase model, or of a phase model that could not be
    read stays its job's alone; and a job of more threads than the host's is
    refused as before."""
    job = _arrays(tmp_path, "gpu", backend="opencl:Device")
    record = read_job(job)
    environment = record["environment"]
    kernel = dict(record, environment=dict(environment, blas_kernel="NoSuchKernel"))
    label = dict(record, environment=dict(environment, backend="opencl-gpu"))

    def backend_refusal(record):
        with pytest.raises(ValueError) as refused:
            jobs_main.backend_of(job, record)
        return str(refused.value)

    monkeypatch.setattr(opencl, "opencl_devices", lambda: [])
    absent, unknown = backend_refusal(record), backend_refusal(label)
    device = SimpleNamespace(name="Device")
    monkeypatch.setattr(opencl, "opencl_devices", lambda: [device])
    monkeypatch.setattr(opencl, "OpenCLBackend", lambda device: device)
    monkeypatch.setattr(gpu_backend, "_agrees", lambda backend, heads: False)
    other = backend_refusal(record)
    assert other.endswith("opencl:Device: none gives the CPU's numbers.")

    def unreadable(job, record):
        raise ValueError("the phase model is not whole")

    monkeypatch.setattr(jobs_main, "heads", unreadable)
    model = backend_refusal(record)
    found = jobs_main.refusal(kernel["environment"])
    assert service._environment_refusal(kernel, found)
    assert service._environment_refusal(record, absent)
    assert service._environment_refusal(label, unknown)
    revision = jobs_main.other_revision(record, "another")
    for why in (other, model, revision):
        assert not service._environment_refusal(record, why)
    processes.refusals[job.name] = revision
    running = _serving(tmp_path, monkeypatch, threads=1)
    running.poll()  # its task process refuses the job's revision
    running.poll()
    assert running.refused[job.name] == revision and running.unfit == {}
    later = _arrays(tmp_path, "later", backend="opencl:Device")
    assert running._refusal(read_job(later)) is None
    wide = _arrays(tmp_path, "wide", threads=3)
    assert running._refusal(read_job(wide)) == (
        "Its tasks need 3 threads; this host has 1."
    )
    assert running.unfit == {}


@pytest.mark.parametrize(
    "kernel, backend", [("NoSuchKernel", "cpu"), ("Haswell", "opencl-gpu")]
)
def test_a_task_process_that_refuses_its_environment_refuses_its_later_jobs(
    tmp_path, importable, monkeypatch, kernel, backend
):
    """A task process that refuses its job's BLAS kernel or compute backend makes
    its service refuse a later job of the same kernel and backend before any claim
    or start, saying why."""
    monkeypatch.setattr(service, "headroom", lambda: 2**40)
    monkeypatch.setattr(service, "reserve", lambda: 0)

    def make(name):
        config, selection = _project(tmp_path / name, kernel=kernel, backend=backend)
        one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
        return create_job(tmp_path / "jobs", (NUMBERS,), one, config)

    first = make("first")
    running = service.Service(tmp_path / "jobs", "here", 2, tmp_path / "logs")
    running.poll()
    ((_, live),) = running.live.items()
    live.process.wait(120)
    running.poll()
    why = running.refused[first.name]
    assert running.unfit == {(kernel, backend): (first.name, why)}
    later = make("later")
    running.poll()
    assert not running.live and not queue.markers(later, queue.CLAIMED)
    assert set(queue.markers(later, queue.QUEUE)) == {"numbers.r2.0000"}
    assert running.refused[later.name] == (
        f"Its BLAS kernel {kernel} and compute backend {backend} are not this "
        f"host's, as a task of job {first.name} found: {why}"
    )


class _Family:
    """A task's family whose job object kept the given commit peaks, or none."""

    def __init__(self, job=None, one=None):
        self.job, self.one, self.asked = job, one, []

    def peaks(self, attempt):
        """Its peaks bound to an attempt (:meth:`darsia.jobs.service.Family.peaks`)."""
        self.asked.append(attempt)
        if self.job is None:
            return None
        return {
            "attempt": attempt,
            "job_commit_peak": self.job,
            "process_commit_peak": self.one,
            "processes": 2,
        }


def test_a_done_mark_keeps_its_familys_commit_peaks_bound_to_its_attempt(
    tmp_path, monkeypatch, kept_threads
):
    """A task process that ended its step marks its task done with the commit
    peaks its family's job object kept, bound to its claim's token; one whose
    family knows none, or that entered none, keeps none, and a failed task's mark
    never does."""
    marks = {}
    for name, family in (
        ("kept", _Family(3 * GIB, GIB)),
        ("unknown", _Family()),
        ("none", None),
    ):
        job = _one(tmp_path / name, kernel=HERE)
        claim = queue.claim(job, "numbers.r2.0000", "here")
        assert jobs_main.run(job, claim.task, claim.token, "here", family) == 0
        marker = queue.read_marker(job / queue.DONE / claim.path.name)
        marks[name] = (claim.token, marker.get("memory"))
        assert family is None or family.asked == [claim.token]
    token, memory = marks["kept"]
    assert memory == {
        "attempt": token,
        "job_commit_peak": 3 * GIB,
        "process_commit_peak": GIB,
        "processes": 2,
    }
    assert marks["unknown"][1] is None and marks["none"][1] is None
    job = _one(tmp_path / "failed", kernel=HERE)
    claim = queue.claim(job, "numbers.r2.0000", "here")

    def broken(job, name):
        raise ValueError("its task cannot be read")

    monkeypatch.setattr(queue, "task", broken)
    family = _Family(3 * GIB, GIB)
    assert jobs_main.run(job, claim.task, claim.token, "here", family) == 0
    marker = queue.read_marker(job / queue.FAILED / claim.path.name)
    assert "memory" not in marker and family.asked == []


def test_a_family_knows_its_peaks_only_whole_and_once_it_is_closed(monkeypatch):
    """A family's peaks are its job object's commit peaks, bound to an attempt,
    once no other process of it runs: none while one does, when the system does
    not say, when they are out of order, without a job object, and on a system
    without job objects."""

    def answering(active, job, one, known=True):
        def query(self, kind, found):
            if kind == service._ACCOUNTING:
                found.ActiveProcesses, found.TotalProcesses = active, 3
            else:
                found.PeakJobMemoryUsed, found.PeakProcessMemoryUsed = job, one
            return known

        return query

    family = service.Family(handle=1)
    monkeypatch.setattr(service.Family, "_query", answering(1, 3 * GIB, GIB))
    assert family.peaks("token") == {
        "attempt": "token",
        "job_commit_peak": 3 * GIB,
        "process_commit_peak": GIB,
        "processes": 3,
    }
    for query in (
        answering(2, 3 * GIB, GIB),  # another of its processes still runs
        answering(1, 3 * GIB, GIB, known=False),
        answering(1, GIB, 3 * GIB),
        answering(1, 0, 0),
    ):
        monkeypatch.setattr(service.Family, "_query", query)
        assert family.peaks("token") is None
    assert service.Family().peaks("token") is None
    monkeypatch.setattr(service, "os", SimpleNamespace(name="posix"))
    assert service.Family.start().peaks("token") is None
    if ctypes.sizeof(ctypes.c_void_p) == 8:  # the layouts of Windows' headers
        assert ctypes.sizeof(service._JobAccounting) == 48
        assert ctypes.sizeof(service._JobLimits) == 144


_FAMILY = """
import json, subprocess, sys
import psutil
from darsia.jobs import service
held = bytearray(int(sys.argv[1]))  # committed before it enters its job object
family = service.Family.start()
code = "import sys, time; b = bytearray(int(sys.argv[1])); time.sleep(2.0)"
child = subprocess.Popen([sys.executable, "-c", code, sys.argv[2]])
running = family.peaks("token")
child.wait()
after = family.peaks("token")
own = psutil.Process().memory_info().private
print(json.dumps({"running": running, "after": after, "own": own}))
"""


@pytest.mark.skipif(os.name != "nt", reason="job objects are Windows'")
def test_a_familys_job_object_keeps_the_commit_peak_of_all_its_processes(
    importable,
):
    """A process that entered a job object of its own knows, once its child has
    ended, the peak commit charge of both together, its own commit from before it
    entered the job included, and nothing while the child runs. A virtual
    environment's interpreter starts the child through its launcher, a process of
    the family too."""
    held, child = 64 * 2**20, 256 * 2**20
    command = [sys.executable, "-c", _FAMILY, str(held), str(child)]
    found = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert found.returncode == 0, found.stderr
    seen = json.loads(found.stdout.strip().splitlines()[-1])
    assert seen["running"] is None, seen
    after = seen["after"]
    assert after["attempt"] == "token" and after["processes"] >= 2, seen
    assert after["job_commit_peak"] >= held + child, seen
    assert after["job_commit_peak"] >= after["process_commit_peak"], seen
    assert after["process_commit_peak"] >= max(child, seen["own"]), seen
