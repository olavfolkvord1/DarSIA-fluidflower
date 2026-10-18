"""Persistent Git-pinned whole-run workers reading a shared UNC job directly."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import sys
import tempfile
import time
import threading
import traceback
from pathlib import Path, PureWindowsPath

from .job_queue import (
    JobQueue,
    LeaseLost,
    name,
    read_json,
    require_unc,
    utc,
    write_immutable,
)


def verify_startup(job_root: Path, *, stores_path=None, machine_id=None,
                   startup_context=None, expected_stores=None) -> tuple[dict, str]:
    """Check the running repository and its actual lockfile exactly once."""
    from ..job_filesystem import inspect_shared_folder

    transport = inspect_shared_folder(job_root)
    if not transport.allowed:
        raise ValueError("Shared queue transport refused: " + transport.reason + ": " + transport.evidence)
    job = read_json(job_root / "job.json")
    alias = job.get("format") == "darsia-alias-job-v1"
    if not alias:
        require_unc(str(job_root))
    elif (job.get("format_version")!=1 or job.get("legacy_paths") is not False
            or job.get("execution_ready") is not True
            or job.get("runtime_support")!="darsia-role-io-runtime-v1"):
        raise RuntimeError("Alias job is not explicitly ready for J3 runtime")
    repository = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty or head != job["commit"]:
        raise RuntimeError("Startup requires a clean Git worktree at job.json commit")
    lock = job["environment_lock"]
    relative = Path(lock["path"])
    if (
        relative.is_absolute()
        or PureWindowsPath(lock["path"]).drive
        or ".." in relative.parts
    ):
        raise ValueError("environment_lock.path must be repository-relative")
    actual = (repository / relative).resolve(strict=True)
    if not actual.is_relative_to(repository):
        raise ValueError("Environment lock escapes the repository")
    git_relative = relative.as_posix()
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "ls-tree", "--name-only",
         head, "--", git_relative], text=True).splitlines()
    if tracked != [git_relative]:
        raise RuntimeError("Environment lock is not tracked at the admitted Git commit")
    original = subprocess.check_output(
        ["git", "-C", str(repository), "show", head + ":" + git_relative])
    if hashlib.sha256(original).hexdigest() != lock["sha256"]:
        raise RuntimeError("Actual repository environment lock SHA differs")
    before = actual.stat()
    raw = actual.read_bytes()
    after = actual.stat()
    stamp = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if (stamp(before) != stamp(after)
            or raw.replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n")):
        raise RuntimeError("Actual repository environment lock differs from its Git content")
    if startup_context is not None:
        startup_context["environment_lock_identity"] = dict(
            commit=head, repo_relative_path=git_relative,
            identity_algorithm="sha256-git-blob-content",
            sha256=hashlib.sha256(original).hexdigest(), size_bytes=len(original),
            working_tree_sha256=hashlib.sha256(raw).hexdigest(),
            working_tree_size_bytes=len(raw), EOL_only_representation=(raw != original))
    if job.get("precision") not in {"float32", "float64"}:
        raise ValueError("One explicit job precision is required")
    import re

    if job.get("pipeline") == "native-phase-statistics":
        from .phase1_protocol import validate_phase1_job
        validate_phase1_job(job)
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", job.get("calibration_sha256", "")):
            raise ValueError("Explicit job calibration SHA required")
    if (
        not isinstance(job.get("expected_runs"), list)
        or not job["expected_runs"]
        or len({name(run) for run in job["expected_runs"]}) != len(job["expected_runs"])
    ):
        raise ValueError("Explicit distinct expected_runs required")
    if alias:
        from ..store_map import read_store_map
        stores=read_store_map(stores_path,forbidden_roots=(repository,))
        if name(stores.machine_id)!=name(machine_id or socket.gethostname()):
            raise ValueError("Store-map machine ID differs from --machine")
        stores.roots(job["series_id"])
        if expected_stores is not None and stores.source!=expected_stores.source:
            raise ValueError("Startup-admitted store map changed; running claims retain their original binding")
        if startup_context is not None:
            startup_context["stores"]=stores
    return job, head


class _JobControlBoundary(Exception):
    """A cooperative operational exit after a committed frame; never a task FAIL."""

    def __init__(self, state, command=None, traceback_text=None):
        self.state, self.command, self.traceback_text = state, command, traceback_text
        super().__init__(state)


def _job_control_boundary(root):
    """Use existing frame hooks; do not change the executor or scientific inputs."""
    from ..job_control import read_control

    try:
        command = read_control(root)
    except Exception:
        raise _JobControlBoundary("control_failure", traceback_text=traceback.format_exc())
    if command["action"] in {"pause", "stop"}:
        state = "paused" if command["action"] == "pause" else "control_stopped"
        raise _JobControlBoundary(state, command=command)


class ServiceLock:
    """One parent/log writer for this local machine, backend and UNC job."""

    def __init__(self, identity: str):
        """Bind a machine/backend identity to the sole shared-log ownership guard."""
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.handle = None
        self.file = None

    def __enter__(self):
        """Acquire the process ownership guard or reject a second active log writer."""
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_bool,
                ctypes.c_wchar_p,
            ]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            ctypes.set_last_error(0)
            self.handle = kernel.CreateMutexW(
                None, True, "Global\\DarSIAWorker-" + self.identity
            )
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183:
                kernel.CloseHandle(ctypes.c_void_p(self.handle))
                self.handle = None
                raise RuntimeError("A worker service already owns this machine log")
        else:
            import fcntl

            self.file = open(
                Path(tempfile.gettempdir()) / ("darsia-worker-" + self.identity), "a"
            )
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.file.close()
                raise RuntimeError(
                    "A worker service already owns this machine log"
                ) from exc
        return self

    def __exit__(self, *args):
        """Release this process ownership guard without removing preserved log files."""
        if self.handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReleaseMutex(ctypes.c_void_p(self.handle))
            kernel.CloseHandle(ctypes.c_void_p(self.handle))
        if self.file is not None:
            self.file.close()


def _identity(machine: str) -> dict:
    import psutil

    process = psutil.Process(os.getpid())
    return dict(
        machine=machine, pid=process.pid, process_create_time=str(process.create_time())
    )


def _run_tasks(
    root: str,
    job: dict,
    commit: str,
    machine: str,
    backend: str,
    messages,
    stop,
    once: bool,
    exit_when_idle: bool,
    watchdog: bool = False,
    worker_slot=None,
    worker_generation=None,
    stores=None,
):
    from darsia._opencv_runtime import scientific_thread, require_ipp_off
    ipp_start = scientific_thread("pipeline-worker:start")
    def queue_error(event):
        messages.put(event)

    alias=job.get("format")=="darsia-alias-job-v1"
    io_runtime=None
    if alias:
        from .worker_io import WorkerRunIO
        io_runtime=WorkerRunIO(root,job,stores,machine,backend,commit)
        shared=JobQueue(Path(root),on_error=queue_error,machine_id=stores.machine_id)
    else:
        shared = JobQueue(Path(root), on_error=queue_error, machine_id=machine)
    excluded_runs=set()
    owner = _identity(machine)
    from darsia_cache.adapters import bind_actual_worker_owner
    bind_actual_worker_owner(owner,slot=worker_slot,generation=worker_generation,role="slot")
    while not stop.is_set():
        try:
            _job_control_boundary(root)
        except _JobControlBoundary as requested:
            messages.put(dict(kind=requested.state, command=requested.command,
                              traceback=requested.traceback_text, actual_worker=owner))
            return
        try:
            shared.recover_publications()
            for record in shared.reclaim_stale():
                messages.put(dict(kind="reclaimed", **record))
            if io_runtime is None:
                claim = shared.claim(machine, backend, owner)
            else:
                with io_runtime.capacity_claim(owner,queue_error) as capacity:
                    claim = shared.claim(machine,backend,owner,
                        availability=capacity.availability,
                        output_factory=io_runtime.output_factory,
                        publication_factory=io_runtime.publications,exclude_runs=excluded_runs)
                    try:
                        capacity.bind(claim)
                    except Exception:
                        if claim is not None:
                            # An actual atomic claim exists even if local disk
                            # bookkeeping failed; the original parent must bind
                            # it for owned retirement before any replacement.
                            messages.put(dict(kind="claimed",run=claim.task["run"],
                                attempt_id=claim.token,actual_worker=owner))
                        raise
        except Exception:
            text = traceback.format_exc()
            messages.put(dict(kind="queue_failure", traceback=text))
            raise SystemExit(1)
        if claim is None:
            if once or exit_when_idle:
                return
            stop.wait(1)
            continue
        messages.put(
            dict(
                kind="claimed",
                run=claim.task["run"],
                attempt_id=claim.token,
                actual_worker=owner,
            )
        )
        try:
            task = claim.task
            if (task.get("pipeline") == "native-phase-statistics"
                    or job.get("pipeline") == "native-phase-statistics"):
                from .phase1_protocol import validate_phase1_claim
                validate_phase1_claim(task, job)
                if task["commit"] != commit:
                    raise ValueError("Claimed phase-one commit differs from the actual worker")
            else:
                if (
                    task["commit"] != commit
                    or task["calibration_sha256"] != job["calibration_sha256"]
                ):
                    raise ValueError("Claimed commit/calibration differs from job.json")
            precision = task.get("precision", job["precision"])
            if precision != job["precision"]:
                raise ValueError("A job cannot mix working precisions")

            io_binding=None
            if io_runtime is not None:
                from .store_runtime import InputHashMismatch,RunUnavailable
                try:
                    io_binding=io_runtime.prepare(claim)
                    from .checkpoint import resume_source_for_claim
                    if task.get("pipeline") == "native-phase-statistics":
                        from .phase1_protocol import phase1_resume_source
                        prepared_resume=phase1_resume_source(shared,claim)
                    else:
                        prepared_resume=resume_source_for_claim(shared,claim)
                except (InputHashMismatch,RunUnavailable):
                    text=traceback.format_exc()
                    record=shared.reject_before_science(claim,text,owner)
                    excluded_runs.add(name(record["run"]))
                    io_runtime.capacity_close(claim)
                    messages.put(dict(kind="input_rejected",run=task["run"],
                        attempt_id=claim.token,traceback=text,actual_worker=owner))
                    io_runtime.release(claim)
                    if once:
                        raise SystemExit(1)
                    continue

            def progress(event):
                if not isinstance(event, dict):
                    raise TypeError(
                        "Scientific progress must be an actual event dictionary"
                    )
                shared.progress(claim, event)
                if io_runtime is not None:
                    io_runtime.capacity_progress(claim,owner,event)
                messages.put(
                    dict(
                        kind="progress",
                        run=task["run"],
                        attempt_id=claim.token,
                        event=event,
                    )
                )
                # frame_completed is emitted only after the original atomic
                # publication. frame_started also permits a zero-frame pause.
                if event.get("state") == "phase1_frame_completed":
                    _job_control_boundary(root)
                elif event.get("state") in {"frame_started", "frame_completed"}:
                    _job_control_boundary(root)

            from .checkpoint import resume_source_for_claim

            science_task = dict(task)
            if "environment_lock" in job:
                science_task["environment_lock"] = job["environment_lock"]
            resume = (resume_source_for_claim(shared,claim)
                      if io_binding is None else prepared_resume)
            if resume is not None:
                science_task["resume_from"] = resume
            pipeline = task.get("pipeline", "n1")
            if pipeline == "n1":
                from darsia.fluidflower.n1_pipeline import run_n1

                execute = run_n1
            elif pipeline == "native-phase-n1":
                from .runner import run_series

                execute = run_series
            elif pipeline == "native-phase-statistics":
                from .phase_statistics import run_phase1

                execute = run_phase1
            else:
                raise ValueError("Unknown explicit task pipeline: " + str(pipeline))
            if io_binding is None:
                result = execute(science_task, claim.output, backend, precision, progress)
            else:
                with io_binding.activate():
                    result = execute(science_task, claim.output, backend, precision, progress)
            if result.get("precision") != precision or result.get("backend") != backend:
                raise ValueError("Actual scientific backend/precision differs")
            if pipeline == "native-phase-statistics":
                from .phase1_protocol import worker_result
                result = worker_result(result, task=task, job=job, commit=commit,
                                       owner=owner, token=claim.token)
            else:
                result = dict(
                    result,
                    actual_worker=owner,
                    commit=commit,
                    calibration_sha256=task["calibration_sha256"],
                    attempt_id=claim.token,
                    task=task,
                )
            result["opencv_ipp_worker"] = dict(start=ipp_start, completed=require_ipp_off("pipeline-worker:complete"))
            shared.complete(claim, result)
            if io_runtime is not None:
                io_runtime.capacity_close(claim)
            messages.put(
                dict(kind="completed", run=task["run"], attempt_id=claim.token)
            )
        except _JobControlBoundary as requested:
            # Keep this claim attached to the parent slot so the existing exact
            # child-retirement code binds its original manifest/frame receipts.
            # Never publish failed/completed for an interrupted whole run.
            messages.put(dict(kind=requested.state, run=claim.task["run"],
                              attempt_id=claim.token, command=requested.command,
                              traceback=requested.traceback_text))
            return
        except LeaseLost:
            text = traceback.format_exc()
            if io_runtime is not None:
                io_runtime.capacity_close(claim)
            messages.put(
                dict(
                    kind="lease_lost",
                    run=claim.task["run"],
                    attempt_id=claim.token,
                    traceback=text,
                )
            )
        except Exception:
            text = traceback.format_exc()
            try:
                shared.fail(claim, text, owner)
            except LeaseLost:
                pass  # Its own attempt failure stays; only the current lease publishes.
            if io_runtime is not None:
                io_runtime.capacity_close(claim)
            messages.put(
                dict(
                    kind="failed",
                    run=claim.task["run"],
                    attempt_id=claim.token,
                    traceback=text,
                )
            )
            if once:
                raise SystemExit(1)
        finally:
            if io_runtime is not None:
                io_runtime.release(claim)
        if once:
            return


class QueueStream:
    """Forward child Python stdout/stderr to the sole parent writer."""

    encoding = "utf-8"

    def __init__(self, messages, stream):
        """Bind a child console stream to the parent process message queue."""
        self.messages, self.stream = messages, stream

    def write(self, text):
        """Forward console text to the parent log writer and return the number of characters."""
        if text:
            self.messages.put(dict(kind="console", stream=self.stream, text=text))
        return len(text)

    def flush(self):
        """Perform no write because this stream retains no buffered child text."""
        pass

    def isatty(self):
        """Report a queued console stream rather than an interactive terminal."""
        return False


def _work(*args):
    slot = args[10] if len(args) > 10 else None
    generation = args[11] if len(args) > 11 else None
    forwarded = _SlotMessages(args[5], slot, generation)
    values = list(args)
    values[5] = forwarded
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = QueueStream(forwarded, "stdout"), QueueStream(
        forwarded, "stderr"
    )
    finished = threading.Event()
    heartbeat = None
    if len(args) > 9 and args[9]:
        heartbeat = threading.Thread(
            target=_heartbeat, args=(forwarded, finished, args[3]), daemon=True
        )
        heartbeat.start()
    try:
        return _run_tasks(*values)
    finally:
        finished.set()
        if heartbeat is not None:
            heartbeat.join()
        sys.stdout, sys.stderr = old_stdout, old_stderr


def emit(log, event):
    """The exact parent console bytes also go into its single log."""
    if event.get("kind") == "console":
        text = event["text"]
        stream = sys.stderr if event["stream"] == "stderr" else sys.stdout
    elif event.get("traceback"):
        text, stream = event["traceback"], sys.stderr
    else:
        import json

        text, stream = (
            utc() + " " + json.dumps(event, ensure_ascii=False) + "\n",
            sys.stdout,
        )
    print(text, file=stream, end="", flush=True)
    log.write(text)
    log.flush()


def main(argv=None):
    """Start the common CPU or CUDA queue worker with one verified code environment.

    Worker slots claim complete runs, report actual progress, and publish only
    while holding their attempt lease. Startup failures retain their traceback.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--stores",type=Path,
        help="Explicit host-local stores.toml (never written into job files)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument(
        "--watchdog",
        action="store_true",
        help="Parent restarts a silent owned worker only after exact retirement.",
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=float,
        default=600.0,
        help="Operational liveness limit; actual-progress queue lease remains 600s.",
    )
    args = parser.parse_args(argv)
    args._cache_terminal_events = []
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.watchdog_seconds > 0 or not math.isfinite(args.watchdog_seconds):
        parser.error("--watchdog-seconds must be finite and positive")
    identity = name(args.machine + "-" + args.backend)
    from ..job_filesystem import inspect_shared_folder
    transport=inspect_shared_folder(args.job)
    if not transport.allowed:
        raise ValueError("Shared queue transport refused: "+transport.reason+": "+transport.evidence)
    with ServiceLock(identity):
        log_root = args.job / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        with (log_root / (identity + ".log")).open("a", encoding="utf-8") as log:
            try:
                startup_context={}
                job, commit = verify_startup(args.job,stores_path=args.stores,
                    machine_id=args.machine,startup_context=startup_context)
                args._alias_stores=startup_context.get("stores")
                from darsia_cache.adapters import bind_actual_worker_owner, require_same_startup_stores
                if args._alias_stores is not None:
                    require_same_startup_stores(args._alias_stores)
                bind_actual_worker_owner(_identity(identity),role="service_parent")
                emit(log, dict(kind="startup_code_admission",
                    environment_lock=startup_context["environment_lock_identity"]))
            except Exception:
                literal = traceback.format_exc()
                emit(log, dict(kind="startup_failure", traceback=literal))
                from darsia_cache.adapters import seal_worker_service
                seal_worker_service(job_root=args.job, exit_code=1,
                    terminal_events=[], traceback_text=literal)
                return 1
            try:
                result = _parent(args, identity, job, commit, log)
            except BaseException:
                literal = traceback.format_exc()
                from darsia_cache.adapters import seal_worker_service
                seal_worker_service(job_root=args.job, exit_code=1,
                    terminal_events=args._cache_terminal_events, traceback_text=literal)
                raise
            from darsia_cache.adapters import seal_worker_service
            seal_worker_service(job_root=args.job, exit_code=result,
                terminal_events=args._cache_terminal_events)
            return result


class _SlotMessages:
    """Tag owned child messages; only the parent prints or writes the log."""

    def __init__(self, messages, slot, generation):
        self.messages, self.slot, self.generation = messages, slot, generation

    def put(self, event):
        self.messages.put(
            dict(event, worker_slot=self.slot, worker_generation=self.generation)
            if self.slot is not None
            else event
        )


def _heartbeat(messages, finished, machine):
    owner = _identity(machine)
    while not finished.is_set():
        messages.put(dict(kind="watchdog_heartbeat", actual_worker=owner))
        finished.wait(5.0)  # Control liveness only; never JobQueue.progress.


def _process_identity(process):
    value = dict(pid=process.pid, process_create_time=str(process.create_time()),
                 command_line=process.cmdline())
    from darsia_cache.owned_process import native_launch_identity
    try:
        native = native_launch_identity(process)
    except OSError as error:
        # Preserve missing native evidence without breaking original science.
        # Such an observation cannot satisfy automatic retirement eligibility.
        value["native_identity_unavailable"] = dict(
            exception_type=type(error).__name__, message=str(error))
    else:
        if native is not None:
            value.update(native)
    return value


def _same_process(identity):
    """PID reuse proves the old incarnation absent, never licenses a new kill."""
    import psutil

    if identity is None:
        return None
    try:
        process = psutil.Process(identity["pid"])
        return (
            process
            if process.create_time() == float(identity["process_create_time"])
            else None
        )
    except psutil.NoSuchProcess:
        return None


def retire_owned_processes(root_identity, observed=()):
    """Retire only the exact known child tree, then prove every incarnation absent."""
    import psutil

    identities = {
        (i["pid"], i["process_create_time"]): dict(i)
        for i in (*(() if root_identity is None else (root_identity,)), *observed)
    }
    root = _same_process(root_identity)
    if root is not None:
        for process in root.children(recursive=True):
            try:
                bound = _process_identity(process)
            except psutil.NoSuchProcess:
                continue
            identities[bound["pid"], bound["process_create_time"]] = bound
    # Known children were observed as descendants or arrived over this slot's
    # owned IPC. Do not include the watchdog/log parent's ancestors.
    live = []
    for bound in reversed(list(identities.values())):
        process = _same_process(bound)
        if process is not None:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                continue
            live.append(process)
    _, pending = psutil.wait_procs(live, timeout=10.0)
    for process in pending:
        bound = next(i for i in identities.values() if i["pid"] == process.pid)
        current = _same_process(bound)
        if current is not None:
            try:
                current.kill()
            except psutil.NoSuchProcess:
                continue
    psutil.wait_procs(pending, timeout=10.0)
    if any(_same_process(bound) is not None for bound in identities.values()):
        raise RuntimeError(
            "Owned child retirement is incomplete; no replacement may start"
        )
    return list(identities.values())


def _start_slot(context, args, job, commit, identity, messages, stop, number):
    import uuid

    generation = uuid.uuid4().hex
    process = context.Process(
        target=_work,
        args=(
            str(args.job),
            job,
            commit,
            identity,
            args.backend,
            messages,
            stop,
            args.once,
            args.exit_when_idle,
            args.watchdog,
            number,
            generation,
            getattr(args,"_alias_stores",None),
        ),
    )
    from darsia_cache.adapters import record_cache_observation
    record_cache_observation("worker_spawn_intent",dict(slot=number,generation=generation))
    process.start()
    import psutil

    try:
        bound = _process_identity(psutil.Process(process.pid))
    except psutil.NoSuchProcess:
        bound = None  # A completed launch has no invented psutil epoch/argv.
    # multiprocessing retains the actual original process HANDLE even when a
    # very short launch has exited; bind its native FILETIME before dropping it.
    from darsia_cache.owned_process import native_launch_identity, bind_retained_native_observation
    try:
        native = native_launch_identity(process, sentinel=getattr(process, "sentinel", None))
    except OSError as error:
        native = None
        if bound is not None:
            bound["native_identity_unavailable"] = dict(
                exception_type=type(error).__name__, message=str(error))
    if native is not None:
        bound = bind_retained_native_observation(native, bound)
    slot = dict(
        number=number,
        generation=generation,
        process=process,
        root=bound,
        observed=[],
        last_heartbeat=time.monotonic(),
        claim=None,
        retired=False,
        waiting_reclaim=False,
        finished=False,
        cache_observation_error=None,
    )
    try:
        record_cache_observation("worker_spawn_observed",dict(slot=number,generation=generation,
            original_native_identity=bound,epoch_unavailable=bound is None))
    except Exception as error:
        # This supplementary record must not discard the already-owned child.
        # Its mandatory native prelude attachment remains unchanged.
        slot["cache_observation_error"]=dict(exception_type=type(error).__name__,
            message=str(error),observation_complete=False)
        try:
            import logging
            logging.getLogger(__name__).error("Cache spawn observation incomplete: %s",
                slot["cache_observation_error"],exc_info=True)
        except BaseException:
            slot["cache_observation_error"]["logging_failed"]=True
    return slot


def _record_retirement(slot, args, identity):
    """The live parent publishes a separate genuine owned-child closure."""
    if slot["root"] is None and slot["process"].is_alive():
        raise RuntimeError(
            "Live launch has no bound epoch; preserve it and do not replace"
        )
    identities = retire_owned_processes(slot["root"], slot["observed"])
    slot["observed"] = identities
    slot["process"].join(timeout=1.0)
    claim = slot["claim"]
    if claim is not None:
        from .checkpoint import retirement_document

        output = (
            args.job
            / "results"
            / name(claim["run"])
            / ".attempts"
            / claim["attempt_id"]
        )
        if claim["task"].get("format")=="darsia-alias-task-v1":
            from .worker_io import retirement_output
            stores=getattr(args,"_alias_stores",None)
            output,actual_owner=retirement_output(args.job,claim["task"],identity,stores)
        document = retirement_document(
            claim["task"], output, identities, _identity(identity)
        )
        if claim["task"].get("format")=="darsia-alias-task-v1":
            document["machine_id"]=stores.machine_id
            document["output_reference"]=actual_owner["output"]
        document["retired_utc"] = utc()
        document["actual_launch_wrapper"] = slot["root"]
        document["owned_launch_handle_retired"] = not slot["process"].is_alive()
        path = output / "retirement.json"
        if path.exists():
            old = read_json(path)
            if (
                old.get("attempt_id") != claim["attempt_id"]
                or old.get("all_absent") is not True
                or old.get("identities") != document["identities"]
            ):
                raise ValueError("A different retirement already exists; preserve it")
        else:
            write_immutable(path, document)
        if claim["task"].get("format")=="darsia-alias-task-v1":
            from .capacity import release_retired_capacity
            release_retired_capacity(args.job,claim["task"],stores,document)
    from darsia_cache.adapters import record_cache_observation
    record_cache_observation("worker_children_retired",dict(slot=slot["number"],
        generation=slot["generation"],original_child_identities=identities,
        owned_launch_handle_retired=not slot["process"].is_alive(),
        service_parent_retirement_proven=False))
    slot["retired"] = True
    return identities


def _parent(args, identity, job, commit, log):
    """Own one log and persistent slots; optionally supervise exact child liveness."""
    import psutil

    if not hasattr(args, "_cache_terminal_events"):
        args._cache_terminal_events = []
    shared = (JobQueue(args.job,machine_id=args._alias_stores.machine_id)
              if job.get("format")=="darsia-alias-job-v1" else JobQueue(args.job,machine_id=identity))
    context = mp.get_context("spawn")
    messages, stop = context.Queue(), context.Event()
    slots = [
        _start_slot(context, args, job, commit, identity, messages, stop, i)
        for i in range(args.workers)
    ]
    emit(
        log,
        dict(kind="started", message=f"Worker service started at Git commit {commit}."),
    )
    fatal = False
    control_failed = False
    try:
        while True:
            try:
                event = messages.get(timeout=0.2)
            except queue.Empty:
                event = None
            if event is not None:
                number = event.get("worker_slot")
                slot = (
                    slots[number]
                    if isinstance(number, int) and 0 <= number < len(slots)
                    else None
                )
                if (
                    slot is not None
                    and event.get("worker_generation") != slot["generation"]
                ):
                    if event.get("kind") != "watchdog_heartbeat":
                        emit(
                            log, event
                        )  # Preserve old output, never attribute it to the replacement.
                    continue
                owner = event.get("actual_worker")
                if slot is not None and owner is not None:
                    if owner.get("machine") != identity:
                        raise ValueError("Owned worker message machine differs")
                    process = _same_process(owner)
                    if process is not None:
                        root = _same_process(slot["root"])
                        try:
                            owned = root is not None and (
                                process.pid == root.pid
                                or any(
                                    parent.pid == root.pid
                                    and parent.create_time() == root.create_time()
                                    for parent in process.parents()
                                )
                            )
                        except psutil.NoSuchProcess:
                            owned = _same_process(owner) is None
                        if not owned:
                            raise ValueError(
                                "Live worker is outside its owned launch tree"
                            )
                    if owner not in slot["observed"]:
                        slot["observed"].append(dict(owner))
                if slot is not None and event.get("kind") == "watchdog_heartbeat":
                    slot["last_heartbeat"] = time.monotonic()
                elif event.get("kind") == "watchdog_heartbeat":
                    raise ValueError("Watchdog message has no owned worker slot")
                else:
                    if slot is not None and event.get("kind") == "claimed":
                        path = (
                            args.job
                            / "claimed"
                            / identity
                            / (name(event["run"]) + ".json")
                        )
                        if job.get("format")=="darsia-alias-job-v1":
                            from .worker_io import claimed_task_after_archive
                            task=claimed_task_after_archive(args.job,identity,event)
                        else:
                            task = read_json(path)
                        if task["attempt_id"] != event["attempt_id"]:
                            raise ValueError(
                                "Claim message and actual task generation differ"
                            )
                        slot["claim"] = dict(
                            run=event["run"], attempt_id=event["attempt_id"], task=task
                        )
                    if slot is not None and event.get("kind") in {
                        "completed",
                        "failed",
                        "lease_lost",
                        "input_rejected",
                    }:
                        # Terminal/revoked attempts do not need a numerical restart.
                        slot["claim"] = None
                    emit(log, event)
                    if event.get("kind") in {"completed", "failed", "lease_lost", "input_rejected",
                            "control_failure", "paused", "control_stopped"}:
                        args._cache_terminal_events.append(dict(event))
                    if event.get("kind") == "control_failure":
                        control_failed = True
            active = False
            for number, slot in enumerate(slots):
                if slot["finished"]:
                    continue
                process = slot["process"]
                stalled = (
                    args.watchdog
                    and process.is_alive()
                    and time.monotonic() - slot["last_heartbeat"]
                    >= args.watchdog_seconds
                )
                if process.is_alive() and not stalled:
                    active = True
                    continue
                if not slot["retired"]:
                    identifiers = _record_retirement(slot, args, identity)
                    emit(
                        log,
                        dict(
                            kind=(
                                "watchdog_retired" if args.watchdog else "child_retired"
                            ),
                            slot=number,
                            identities=identifiers,
                            stalled=stalled,
                            exitcode=process.exitcode,
                        ),
                    )
                    clean_exit = process.exitcode == 0 and not stalled
                    if (
                        not args.watchdog
                        or clean_exit
                        or (args.once and slot["claim"] is None)
                    ):
                        slot["finished"] = True
                        fatal |= process.exitcode not in (0, None)
                        continue
                    slot["waiting_reclaim"] = slot["claim"] is not None
                if slot["waiting_reclaim"]:
                    # Keep the original 600-second actual-progress lease unchanged.
                    shared.recover_publications()
                    for record in shared.reclaim_stale():
                        emit(log, dict(kind="reclaimed", **record))
                    claim = slot["claim"]
                    lease = (
                        args.job
                        / "leases"
                        / (name(claim["run"]) + "." + claim["attempt_id"])
                    )
                    if lease.exists():
                        active = True
                        continue
                    if (args.job / "publishing" / name(claim["run"])).exists():
                        slot["finished"] = True
                        continue
                    slot["waiting_reclaim"] = False
                # Re-verify the same startup admission before a replacement imports
                # checkout code. No per-file code guard or generated start code.
                if job.get("format")=="darsia-alias-job-v1":
                    same_job,same_commit=verify_startup(args.job,
                        stores_path=getattr(args,"stores",None),machine_id=args.machine,
                        expected_stores=args._alias_stores)
                else:
                    same_job, same_commit = verify_startup(args.job)
                if same_job != job or same_commit != commit:
                    raise RuntimeError("Watchdog restart startup identity differs")
                slots[number] = _start_slot(
                    context, args, job, commit, identity, messages, stop, number
                )
                emit(
                    log,
                    dict(
                        kind="watchdog_restarted",
                        slot=number,
                        actual_launch_wrapper=slots[number]["root"],
                    ),
                )
                active = True
            if not active and all(slot["finished"] for slot in slots):
                break
    except KeyboardInterrupt:
        stop.set()  # Existing user interrupt behavior: finish the owned run.
    except Exception:
        fatal = True
        stop.set()
        emit(log, dict(kind="parent_failure", traceback=traceback.format_exc()))
    finally:
        for slot in slots:
            if fatal and slot["process"].is_alive():
                _record_retirement(slot, args, identity)
            else:
                slot["process"].join()
        while True:
            try:
                event = messages.get_nowait()
            except queue.Empty:
                break
            if event.get("kind") != "watchdog_heartbeat":
                emit(log, event)
            if event.get("kind") in {"completed", "failed", "lease_lost", "input_rejected",
                    "control_failure", "paused", "control_stopped"}:
                args._cache_terminal_events.append(dict(event))
            if event.get("kind") == "control_failure":
                control_failed = True
        emit(log, dict(kind="stopped", message="Worker service stopped."))
    return 1 if fatal or control_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
