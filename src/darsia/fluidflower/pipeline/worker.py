"""Persistent Git-pinned whole-run workers reading a shared UNC job directly."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path, PureWindowsPath

from .job_queue import JobQueue, LeaseLost, name, read_json, require_unc, utc


def verify_startup(job_root: Path) -> tuple[dict, str]:
    """Check the running repository and its actual lockfile exactly once."""
    require_unc(str(job_root))
    job = read_json(job_root / "job.json")
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
    if (relative.is_absolute() or PureWindowsPath(lock["path"]).drive
            or ".." in relative.parts):
        raise ValueError("environment_lock.path must be repository-relative")
    actual = (repository / relative).resolve(strict=True)
    if not actual.is_relative_to(repository):
        raise ValueError("Environment lock escapes the repository")
    if hashlib.sha256(actual.read_bytes()).hexdigest() != lock["sha256"]:
        raise RuntimeError("Actual repository environment lock SHA differs")
    if job.get("precision") not in {"float32", "float64"}:
        raise ValueError("One explicit job precision is required")
    import re

    if not re.fullmatch(r"[0-9a-f]{64}", job.get("calibration_sha256", "")):
        raise ValueError("Explicit job calibration SHA required")
    if (not isinstance(job.get("expected_runs"), list) or not job["expected_runs"]
            or len({name(run) for run in job["expected_runs"]}) != len(job["expected_runs"])):
        raise ValueError("Explicit distinct expected_runs required")
    return job, head


class ServiceLock:
    """One parent/log writer for this local machine, backend and UNC job."""

    def __init__(self, identity: str):
        self.identity = hashlib.sha256(identity.encode()).hexdigest()
        self.handle = None
        self.file = None

    def __enter__(self):
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
            kernel.CreateMutexW.restype = ctypes.c_void_p
            ctypes.set_last_error(0)
            self.handle = kernel.CreateMutexW(None, True, "Global\\DarSIAWorker-" + self.identity)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183:
                kernel.CloseHandle(ctypes.c_void_p(self.handle))
                self.handle = None
                raise RuntimeError("A worker service already owns this machine log")
        else:
            import fcntl

            self.file = open(Path(tempfile.gettempdir()) / ("darsia-worker-" + self.identity), "a")
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.file.close()
                raise RuntimeError("A worker service already owns this machine log") from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.ReleaseMutex(ctypes.c_void_p(self.handle))
            kernel.CloseHandle(ctypes.c_void_p(self.handle))
        if self.file is not None:
            self.file.close()


def _identity(machine: str) -> dict:
    import psutil

    process = psutil.Process(os.getpid())
    return dict(machine=machine, pid=process.pid, process_create_time=str(process.create_time()))


def _run_tasks(root: str, job: dict, commit: str, machine: str, backend: str,
          messages, stop, once: bool, exit_when_idle: bool):
    def queue_error(event):
        messages.put(event)

    shared = JobQueue(Path(root), on_error=queue_error)
    owner = _identity(machine)
    while not stop.is_set():
        try:
            shared.recover_publications()
            for record in shared.reclaim_stale():
                messages.put(dict(kind="reclaimed", **record))
            claim = shared.claim(machine, backend, owner)
        except Exception:
            text = traceback.format_exc()
            messages.put(dict(kind="queue_failure", traceback=text))
            raise SystemExit(1)
        if claim is None:
            if once or exit_when_idle:
                return
            stop.wait(1)
            continue
        messages.put(dict(kind="claimed", run=claim.task["run"], attempt_id=claim.token,
                          actual_worker=owner))
        try:
            task = claim.task
            if task["commit"] != commit or task["calibration_sha256"] != job["calibration_sha256"]:
                raise ValueError("Claimed commit/calibration differs from job.json")
            precision = task.get("precision", job["precision"])
            if precision != job["precision"]:
                raise ValueError("A job cannot mix working precisions")

            def progress(event):
                if not isinstance(event, dict):
                    raise TypeError("Scientific progress must be an actual event dictionary")
                shared.progress(claim, event)
                messages.put(dict(kind="progress", run=task["run"], attempt_id=claim.token, event=event))

            from darsia.fluidflower.n1_pipeline import run_n1

            result = run_n1(task, claim.output, backend, precision, progress)
            if result.get("precision") != precision or result.get("backend") != backend:
                raise ValueError("Actual scientific backend/precision differs")
            result = dict(result, actual_worker=owner, commit=commit,
                          calibration_sha256=task["calibration_sha256"],
                          attempt_id=claim.token, task=task)
            shared.complete(claim, result)
            messages.put(dict(kind="completed", run=task["run"], attempt_id=claim.token))
        except LeaseLost:
            text = traceback.format_exc()
            messages.put(dict(kind="lease_lost", run=claim.task["run"],
                              attempt_id=claim.token, traceback=text))
        except Exception:
            text = traceback.format_exc()
            try:
                shared.fail(claim, text, owner)
            except LeaseLost:
                pass  # Its own attempt failure stays; only the current lease publishes.
            messages.put(dict(kind="failed", run=claim.task["run"],
                              attempt_id=claim.token, traceback=text))
            if once:
                raise SystemExit(1)
        if once:
            return


class QueueStream:
    """Forward child Python stdout/stderr to the sole parent writer."""

    encoding = "utf-8"

    def __init__(self, messages, stream):
        self.messages, self.stream = messages, stream

    def write(self, text):
        if text:
            self.messages.put(dict(kind="console", stream=self.stream, text=text))
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def _work(*args):
    messages = args[5]
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = QueueStream(messages, "stdout"), QueueStream(messages, "stderr")
    try:
        return _run_tasks(*args)
    finally:
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

        text, stream = utc() + " " + json.dumps(event, ensure_ascii=False) + "\n", sys.stdout
    print(text, file=stream, end="", flush=True)
    log.write(text)
    log.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--machine", default=socket.gethostname())
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    identity = name(args.machine + "-" + args.backend)
    require_unc(str(args.job))
    with ServiceLock(identity):
        log_root = args.job / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        with (log_root / (identity + ".log")).open("a", encoding="utf-8") as log:
            try:
                job, commit = verify_startup(args.job)
            except Exception:
                emit(log, dict(kind="startup_failure", traceback=traceback.format_exc()))
                return 1
            return _parent(args, identity, job, commit, log)


def _parent(args, identity, job, commit, log):
        shared = JobQueue(args.job)
        context = mp.get_context("spawn")
        messages, stop = context.Queue(), context.Event()
        children = [context.Process(target=_work, args=(
            str(args.job), job, commit, identity, args.backend, messages, stop,
            args.once, args.exit_when_idle,
        )) for _ in range(args.workers)]
        # This parent is the sole writer. Children only submit log messages.
        emit(log, dict(kind="started", message=f"Worker service started at Git commit {commit}."))
        for child in children:
            child.start()
        try:
            while True:
                try:
                    event = messages.get(timeout=0.2)
                except queue.Empty:
                    if not any(child.is_alive() for child in children):
                        break
                    continue
                emit(log, event)
        except KeyboardInterrupt:
            stop.set()  # Finish the owned whole run; no scientific kill/replay.
        finally:
            for child in children:
                child.join()
            while True:
                try:
                    event = messages.get_nowait()
                except queue.Empty:
                    break
                emit(log, event)
            emit(log, dict(kind="stopped", message="Worker service stopped."))
        return 0 if all(child.exitcode == 0 for child in children) else 1


if __name__ == "__main__":
    raise SystemExit(main())
