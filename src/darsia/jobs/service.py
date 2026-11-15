"""The service of one host: it runs the ready tasks of a project's jobs.

Each host runs one service, which holds a lock of the whole machine for its life
(:func:`host_lock`), so a second start on the host, under any account, exits and
says why. Every :data:`POLL` seconds the service

- notes its task processes that exited: one that left its claim unmarked is
  returned to the queue once and failed the second time, and one that refused its
  job (its BLAS kernel, compute backend or code revision is not this host's) and
  said so in its claim is returned, and the host takes no more of that job's
  tasks;
- takes back stale claims of any host (:func:`darsia.jobs.queue.reclaim`);
- counts the claims of its host whose process outlived an earlier service, each
  known by its claim's token, its process and that process's birth time, and
  reserves their threads; it never takes them back, and returns a claim at once
  when its process is gone, which a process that ended while it looked, or that it
  may not look at, counts as;
- starts the oldest job's first ready task, while the threads of its live tasks
  and of the task are within the host's capacity, as a new process
  (``python -m darsia.jobs run``) with its job's computing environment.

Each part of a round that cannot read the jobs folder now, e.g. a job's file read
short over the share, says why and is tried again in the next round.

The machine's capacity is one number of threads, shared by every job of the
project, the oldest job first; with none, the service takes no new tasks. Each
job fixes the threads of its tasks (:class:`darsia.jobs.job.Environment`). A job
records the code's revision at its creation; a service whose code is of another
revision takes none of its tasks, a service whose checkout has moved on since it
started takes no new tasks, and a task process of another revision refuses its
task, each saying why.

What the service says goes to its console and to the job's log; a failed write of
either never stops it (:func:`say`). A task's own output goes to a file on the
host, and its traceback into its failed marker.

"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep

import psutil

from darsia.jobs import queue
from darsia.jobs.job import JOB, Environment, project_of, read_job, revision
from darsia.utils import results

try:
    import msvcrt
    from ctypes import wintypes
except ImportError:  # POSIX
    msvcrt = wintypes = None
try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

POLL = 5.0
"""Seconds between two rounds of the service."""
REFUSED = 75
"""Exit code of a task process that refused its job, having said why in its claim:
its host does not provide the job's BLAS kernel or compute backend, or its code is
of another revision. Not a code of the C runtime's ``abort`` (3) or of Python."""
LOG = "log.txt"
"""The file of a job's log, in its folder."""
HOST_MUTEX = "Global\\darsia-jobs-service"
"""The name of the service's lock of a Windows machine, seen by every account."""
HOST_FILE = Path("/tmp/darsia-jobs-service.lock")
"""The service's lock file of a POSIX machine, the same for every account."""


class ServiceRunning(RuntimeError):
    """Another service holds this host's lock."""


def say(text: str, log: Path | None = None) -> None:
    """Say something on the console and in a log; a failed write of either is
    swallowed, so that saying never stops a service or a task.

    Parameters
    ----------
    text : str
        What to say.
    log : Path, optional
        The log file to append it to, e.g. a job's on the shared folder.

    """
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    if log is not None:
        try:
            with open(log, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except Exception:
            pass


def _lock_file(path: Path):
    """A lock file opened to be locked: for writing, or only for reading where
    another account made it, since a lock needs reading only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return open(path, "a+b")
    except PermissionError:
        return open(path, "rb")


def _kernel32():
    """Windows' kernel32 with the types of its mutex calls set."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    return kernel32


class HostLock:
    """An operating-system lock on a local file, held for the life of a service;
    the system releases it when the process ends.

    Parameters
    ----------
    path : Path
        The lock file, local to the host.

    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        """The lock file."""
        self._stream = None

    def __enter__(self) -> HostLock:
        """Take the lock.

        Raises
        ------
        ServiceRunning
            If another process holds it.

        """
        stream = _lock_file(self.path)
        try:
            if msvcrt is not None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise ServiceRunning(
                f"Another service runs on this host (it holds {self.path})."
            ) from error
        self._stream = stream
        return self

    def __exit__(self, *_) -> None:
        """Release the lock."""
        self._stream.close()
        self._stream = None


class HostMutex:
    """A named mutex of a Windows machine, held for the life of a service; the
    system removes it when the last process holding it ends.

    Parameters
    ----------
    name : str
        The mutex's name; in the ``Global\\`` namespace every account and session
        sees it.

    """

    _EXISTS, _DENIED = 183, 5  # ERROR_ALREADY_EXISTS, ERROR_ACCESS_DENIED

    def __init__(self, name: str) -> None:
        self.name = name
        """The mutex's name."""
        self._handle = None

    def __enter__(self) -> HostMutex:
        """Take the mutex.

        Raises
        ------
        ServiceRunning
            If another process holds it, also under another account.

        """
        kernel32 = _kernel32()
        handle = kernel32.CreateMutexW(None, False, self.name)
        error = ctypes.get_last_error()
        if handle and error != self._EXISTS:
            self._handle, self._close = handle, kernel32.CloseHandle
            return self
        if handle:
            kernel32.CloseHandle(handle)
        if handle or error == self._DENIED:
            raise ServiceRunning(
                f"Another service runs on this host (it holds {self.name})."
            )
        raise ctypes.WinError(error)

    def __exit__(self, *_) -> None:
        """Release the mutex."""
        self._close(self._handle)
        self._handle = None


class WaitingLock:
    """A lock of the whole machine that its taker waits for, held for a short
    while, e.g. while a film's encoder opens: on Windows a named mutex of the
    ``Global\\`` namespace, which every account sees and the system hands on when
    its holder dies, and on POSIX an exclusive lock of a fixed file. One thread
    takes and releases it, and may take it again once released.

    Parameters
    ----------
    name : str
        The lock's name, e.g. ``"film-encoder"``.

    """

    _INFINITE, _OWNED, _ABANDONED = 0xFFFFFFFF, 0, 0x80

    def __init__(self, name: str) -> None:
        self.name = name
        """The lock's name."""
        self._held = None

    def __enter__(self) -> WaitingLock:
        """Wait until the lock is free, and take it."""
        if msvcrt is None:
            stream = _lock_file(Path(f"/tmp/darsia-{self.name}.lock"))
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            self._held = stream
            return self
        kernel32 = _kernel32()
        handle = kernel32.CreateMutexW(None, False, f"Global\\darsia-{self.name}")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        state = kernel32.WaitForSingleObject(handle, self._INFINITE)
        if state not in (self._OWNED, self._ABANDONED):  # abandoned: its holder died
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._held = (kernel32, handle)
        return self

    def __exit__(self, *_) -> None:
        """Release the lock."""
        if msvcrt is None:
            self._held.close()
        else:
            kernel32, handle = self._held
            kernel32.ReleaseMutex(handle)
            kernel32.CloseHandle(handle)
        self._held = None


def host_lock(lock: Path | None = None):
    """The service's lock of the host: of the whole machine, whatever the account
    or its temporary folder (:data:`HOST_MUTEX` on Windows, :data:`HOST_FILE` on
    POSIX), or a given lock file, e.g. a test's.

    Parameters
    ----------
    lock : Path, optional
        A lock file instead of the machine's lock.

    Returns
    -------
    HostLock or HostMutex
        The lock, to be held with ``with``.

    """
    if lock is not None:
        return HostLock(Path(lock))
    return HostMutex(HOST_MUTEX) if os.name == "nt" else HostLock(HOST_FILE)


def birth(pid: int) -> float | None:
    """A process's birth time (``psutil.Process.create_time``), or None if it is
    gone, also if it ended while being looked at, or may not be looked at
    (``NoSuchProcess``, ``AccessDenied``): every caller takes None for gone.

    Unlike a file the share gives no answer about, a process the system will not
    describe is taken for gone: its claim is returned, and the process, if it still
    runs, is fenced by its claim's token, so that its next beat finds its marker
    gone and its results count for nothing. psutil reads a birth time without
    opening the process where it can (on Windows it falls back to
    ``NtQuerySystemInformation``), so a task process of the service's own account
    raises no ``AccessDenied``."""
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def alive(pid: int | None, born: float | None) -> bool:
    """Whether a process still runs with the recorded birth time, so that a reused
    process number is not taken for it."""
    if pid is None or born is None:
        return False
    found = birth(int(pid))
    return found is not None and abs(found - float(born)) < 0.01


@dataclass
class Live:
    """A task running on this host: started by this service, or a survivor."""

    claim: queue.Claim
    """Its claim."""
    threads: int
    """Its threads."""
    process: subprocess.Popen | None
    """Its process, if this service started it."""
    pid: int | None = None
    """A survivor's process."""
    born: float | None = None
    """A survivor's birth time."""


class Service:
    """One host's service over a jobs folder (:func:`serve`).

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    machine : str
        This host's name in the leases.
    threads : int
        The host's capacity: threads its live tasks may use at once.
    logs : Path
        Local folder of the tasks' output.

    """

    def __init__(self, jobs: Path, machine: str, threads: int, logs: Path) -> None:
        self.jobs = Path(jobs)
        """The jobs folder."""
        self.machine = machine
        """This host's name."""
        self.threads = threads
        """The host's capacity in threads."""
        self.logs = Path(logs)
        """Local folder of the tasks' output."""
        self.revision = revision()
        """The code's revision of this service, when it started."""
        self.live: dict[Path, Live] = {}
        """The tasks running on this host, by claimed marker."""
        self.refused: dict[str, str] = {}
        """Jobs this service takes no tasks of, and why."""
        self.exits: dict[tuple[str, str], int] = {}
        """Exits without a mark, by job and task."""
        self._seen: dict[str, tuple[bytes, float]] = {}
        self._records: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}
        self._idle: str | None = None

    def _jobs(self) -> list[tuple[Path, dict]]:
        """The jobs of the folder with their records, oldest first; a job whose
        record cannot be read now is left for the next round."""
        found = []
        names = sorted(results.retry_transient(lambda: os.listdir(self.jobs)))
        for name in names:
            folder = self.jobs / name
            if name.startswith(".") or not (folder / JOB).is_file():
                continue
            if name not in self._records:
                try:
                    self._records[name] = read_job(folder)
                except (OSError, ValueError) as error:
                    say(f"The job {name} cannot be read now; trying again: {error}")
                    continue
            found.append((folder, self._records[name]))
        return found

    def _refusal(self, record: dict) -> str | None:
        """Why this service takes none of a job's tasks, or None.

        A job's code revision is fixed at its creation; a service of another
        revision takes none of its tasks. A job whose tasks need more threads than
        the host's capacity cannot start here.

        """
        if record["id"] in self.refused:
            return self.refused[record["id"]]
        if record["revision"] != self.revision:
            return (
                f"The job was created at revision {record['revision']}; this "
                f"service runs {self.revision or 'a revision not known'}."
            )
        if self.threads and record["environment"]["threads"] > self.threads:
            return (
                f"Its tasks need {record['environment']['threads']} threads; this "
                f"host's capacity is {self.threads}."
            )
        return None

    def _taking(self) -> bool:
        """Whether this service takes new tasks now, saying once when it does not:
        it has threads, and its code is of a known revision, the one it started
        with, which its checkout still holds."""
        now = revision()
        why = None
        if self.threads <= 0:
            why = "its capacity is no threads"
        elif self.revision is None:
            why = "the revision of its code is not known"
        elif now != self.revision:
            why = (
                f"its checkout has moved from revision {self.revision} to "
                f"{now or 'a revision not known'}; restart it"
            )
        if why is not None and why != self._idle:
            say(f"{self.machine} takes no new tasks: {why}.")
        self._idle = why
        return why is None

    def _refuse(self, job: Path, record: dict, why: str) -> None:
        """Take no more tasks of a job, saying why once."""
        if record["id"] not in self.refused:
            self.refused[record["id"]] = why
            say(
                f"{self.machine} takes no tasks of job {record['id']}: {why}", job / LOG
            )

    def _reap(self) -> None:
        """Note the task processes that exited, and what they left behind."""
        for path, live in list(self.live.items()):
            if live.process is not None:
                code = live.process.poll()
                if code is None:
                    continue
            elif alive(live.pid, live.born):
                continue
            else:
                code = None
            claim = live.claim
            try:
                marked = not results.exists(claim.path)  # done, failed or taken back
                lease = (
                    {} if marked else queue.lease(claim.job, claim.task, claim.token)
                )
            except (OSError, ValueError) as error:  # kept for the next round
                say(f"The claim {claim.path} cannot be read now: {error}")
                continue
            del self.live[path]
            if marked:
                continue
            record = self._records[claim.job.name]
            log = claim.job / LOG
            if code == REFUSED and lease.get("refused"):
                self._release(claim, log, "its process refused the job")
                self._refuse(claim.job, record, lease["refused"])
                continue
            key = (record["id"], claim.task)
            self.exits[key] = self.exits.get(key, 0) + 1
            if self.exits[key] == 1:
                self._release(claim, log, f"its process ended ({code}) unmarked")
                continue
            try:
                queue.finish(
                    claim,
                    traceback=f"Its process ended twice without a mark ({code}).",
                )
                del self.exits[key]  # a retry of the task is returned once again
                say(f"Task {claim.task} failed: its process ended twice unmarked", log)
            except (queue.Revoked, OSError) as error:
                say(f"Task {claim.task} could not be marked failed: {error}", log)

    def _release(self, claim: queue.Claim, log: Path, why: str) -> None:
        """Return a claim to the queue, saying why."""
        try:
            queue.release(claim)
            say(f"Task {claim.task} returned to the queue: {why}", log)
        except (queue.Revoked, OSError) as error:
            say(f"Task {claim.task} could not be returned: {error}", log)

    def _survivors(self, jobs: list[tuple[Path, dict]]) -> None:
        """Count this host's claims whose process outlived an earlier service, and
        return those whose process is gone (:func:`birth`) or never started (no
        process in the lease), going on with the next claim."""
        for job, record in jobs:
            for task, token in queue.markers(job, queue.CLAIMED).items():
                claim = queue.Claim(job, task, token, self.machine)
                if claim.path in self.live:
                    continue
                try:
                    lease = queue.lease(job, task, token)
                except (OSError, ValueError):
                    continue  # Gone, or unreadable now: never taken for dead.
                if lease.get("machine") != self.machine:
                    continue
                if alive(lease.get("pid"), lease.get("birth")):
                    threads = record["environment"]["threads"]
                    self.live[claim.path] = Live(
                        claim, threads, None, lease["pid"], lease["birth"]
                    )
                    say(f"Task {task} still runs in process {lease['pid']}", job / LOG)
                else:
                    self._release(claim, job / LOG, "its process is gone")

    def _log(self, claim: queue.Claim) -> Path:
        """The local file of a task's output."""
        return self.logs / claim.job.name / f"{claim.task}.{claim.token}.log"

    def _start(self, job: Path, record: dict, claim: queue.Claim) -> None:
        """Start a claimed task's process with its job's environment.

        The process is then the only writer of its claimed marker: its lease, and
        its mark done, failed or refused.

        """
        environment = Environment(**record["environment"])
        variables = {**os.environ, **environment.variables()}
        command = [sys.executable, "-m", "darsia.jobs", "run", str(job), claim.task]
        command += [claim.token, "--machine", self.machine]
        log = self._log(claim)
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            output = open(log, "ab")
        except OSError:
            output = subprocess.DEVNULL
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                cwd=project_of(job, record),
                env=variables,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        finally:
            if output is not subprocess.DEVNULL:
                output.close()
        self.live[claim.path] = Live(claim, environment.threads, process)
        say(f"{self.machine} started task {claim.task}", job / LOG)

    def _admit(self, jobs: list[tuple[Path, dict]]) -> None:
        """Start the oldest job's first ready task while the host's capacity
        allows; a task whose process cannot start is returned, and the host takes
        no more of its job's tasks."""
        if not self._taking():
            return
        used = sum(live.threads for live in self.live.values())
        for job, record in jobs:
            why = self._refusal(record)
            if why is not None:
                self._refuse(job, record, why)
                continue
            threads = record["environment"]["threads"]
            if record["id"] not in self._tasks:
                self._tasks[record["id"]] = queue.tasks(job)
            for task in queue.ready(job, self._tasks[record["id"]]):
                if used + threads > self.threads:
                    return
                claim = queue.claim(job, task.id, self.machine)
                if claim is None:
                    continue
                try:
                    self._start(job, record, claim)
                except Exception as error:
                    why = f"its process could not be started: {error!r}"
                    self._release(claim, job / LOG, why)
                    self._refuse(job, record, why)
                    break
                used += threads

    def _round(self, what: str, action: Callable[[], object]) -> object:
        """One part of a round; an error says why, and the part is tried again in
        the next round."""
        try:
            return action()
        except Exception as error:  # e.g. the share is out, or a file reads short
            say(f"{what} cannot be done now; trying again: {error!r}")
            return None

    def poll(self) -> None:
        """One round of the service."""
        self._round("Noting the ended tasks", self._reap)
        jobs = self._round("Reading the jobs folder", self._jobs) or []
        for job, record in jobs:

            def reclaim(job=job):
                for task in queue.reclaim(job, self._seen):
                    say(f"Task {task} taken back: its lease went stale", job / LOG)

            self._round(f"Taking back the stale claims of {record['id']}", reclaim)
        self._round("Counting this host's claims", lambda: self._survivors(jobs))
        self._round("Starting ready tasks", lambda: self._admit(jobs))


def serve(
    jobs: Path,
    machine: str | None = None,
    *,
    threads: int | None = None,
    lock: Path | None = None,
    logs: Path | None = None,
    poll: float = POLL,
    until: Callable[[], bool] | None = None,
) -> None:
    """Run this host's service over a jobs folder (``python -m darsia.jobs serve``).

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    machine : str, optional
        This host's name in the leases; by default the host's network name.
    threads : int, optional
        The host's capacity: threads its live tasks may use at once, shared by
        every job of the folder, the oldest job first; 0 takes no new tasks; by
        default the logical cores less one.
    lock : Path, optional
        A lock file instead of the machine's lock (:func:`host_lock`), e.g. a
        test's.
    logs : Path, optional
        Local folder of the tasks' output; by default in the temporary folder.
    poll : float, optional
        Seconds between two rounds (:data:`POLL`).
    until : Callable[[], bool], optional
        Asked after every round; the service stops once it is true and no task of
        the service runs; by default never.

    Raises
    ------
    ServiceRunning
        If another service runs on this host.

    """
    machine = machine or socket.gethostname()
    threads = threads if threads is not None else max(1, (os.cpu_count() or 2) - 1)
    if logs is None:
        logs = Path(tempfile.gettempdir()) / "darsia-jobs" / "tasks"
    with host_lock(lock):
        service = Service(jobs, machine, threads, Path(logs))
        say(f"{machine} serves {jobs} with {threads} threads ({service.revision})")
        while True:
            service.poll()
            if until is not None and until() and not service.live:
                return
            sleep(poll)


def settled(jobs: Path) -> bool:
    """Whether no task of any job of a folder can still run: each is done,
    failed, or waits behind a failed task (:func:`darsia.jobs.queue.waiting`)."""
    for name in sorted(os.listdir(jobs)):
        if not name.startswith(".") and (Path(jobs) / name / JOB).is_file():
            if queue.waiting(Path(jobs) / name):
                return False
    return True
