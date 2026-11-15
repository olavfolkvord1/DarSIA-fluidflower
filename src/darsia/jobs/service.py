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
  reserves their threads and bytes; it never takes them back, and returns a claim
  at once when its process is gone, which a process that ended while it looked, or
  that it may not look at, counts as;
- starts ready tasks, the oldest job's first, each as a new process
  (``python -m darsia.jobs run``) with its job's computing environment, while the
  host admits them (:func:`admit`): the live tasks' threads and the task's are at
  most the host's, and the task's declared bytes are at most the memory free now,
  less the reserve, less what the live tasks may still take.

Each part of a round that cannot read the jobs folder now, e.g. a job's file read
short over the share, says why and is tried again in the next round.

The machine's threads and memory are shared by every job of the project, the
oldest job first; a task that would not fit even on the idle host is passed over,
so that it holds up no other task. With no threads, the service takes no new
tasks. Each job fixes the threads of its tasks
(:class:`darsia.jobs.job.Environment`). A job
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
from collections.abc import Callable, Sequence
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
RESERVE = 0.10
"""Share of the host's physical memory that no task is admitted into."""
AUTO = "auto"
"""The request of a host that takes as many tasks as its threads and memory
admit."""
PROCESS = 300 * 2**20
"""Bytes a task process holds of its own beside the arrays its step declares: the
interpreter with NumPy, OpenCV and DarSIA loaded, measured at about 0.3 GiB."""


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


class _MemoryStatus(ctypes.Structure):
    """Windows' ``MEMORYSTATUSEX``."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def headroom() -> int:
    """The bytes free for a new task now, measured fresh: on Windows the smaller of
    the available physical memory and the available commit, since an allocation
    fails when either runs out; elsewhere the available physical memory."""
    if os.name != "nt":
        return int(psutil.virtual_memory().available)
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(min(status.ullAvailPhys, status.ullAvailPageFile))


def reserve() -> int:
    """The bytes no task is admitted into: :data:`RESERVE` of the physical memory."""
    return int(RESERVE * psutil.virtual_memory().total)


def _size(value: int) -> str:
    """Bytes as GiB, for what the service says."""
    return f"{value / 2**30:.2f} GiB"


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
    bytes: int = 0
    """The bytes it reserves: those it declares (:func:`darsia.jobs.job.task_bytes`)
    and its process's own (:data:`PROCESS`)."""
    held: int = 0
    """The bytes its processes held when the service last looked
    (:meth:`resident`), once per round."""

    def resident(self) -> int:
        """The bytes its process and the processes it started hold now (their
        resident sets), e.g. the interpreter a launcher starts or a film's
        encoder; none of a process that is gone or may not be looked at, so that
        its declaration stays reserved."""
        pid = self.process.pid if self.process is not None else self.pid
        try:
            process = psutil.Process(int(pid))
            tree = [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            return 0
        held = 0
        for one in tree:
            try:
                held += one.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return int(held)


def _reserved(record: dict, task: str) -> int:
    """The bytes a task of a job reserves: those it declares and its process's
    own (:data:`PROCESS`)."""
    return record.get("bytes", {}).get(task, 0) + PROCESS


def admit(
    task_bytes: int,
    task_threads: int,
    live: Sequence[Live],
    request: int | str,
    threads: int,
    free: int,
    kept: int,
) -> tuple[bool, str]:
    """Whether a task may start beside a host's live tasks, and why not.

    A task starts only when the live tasks' threads and its own are at most the
    host's threads, and its bytes are at most the free memory, less the reserve,
    less what the live tasks may still take: each one's reserved bytes beyond
    what its processes held at the round's look (:attr:`Live.held`). The live
    tasks include the survivors of an earlier service. A request of a number of
    tasks is a ceiling, never more than these rules admit; 0 takes no new tasks.

    Parameters
    ----------
    task_bytes : int
        The bytes the task declares.
    task_threads : int
        Its threads (its job's).
    live : Sequence[Live]
        The tasks running on the host.
    request : int or str
        The host's request: :data:`AUTO`, or at most this many tasks.
    threads : int
        The host's threads: its logical cores less one, or fewer.
    free : int
        The bytes free now (:func:`headroom`).
    kept : int
        The reserve (:func:`reserve`).

    Returns
    -------
    tuple of bool and str
        Whether it starts, and why not; empty when it does.

    """
    if request != AUTO and len(live) >= int(request):
        return False, f"the request caps the host at {int(request)} tasks"
    used = sum(task.threads for task in live)
    if used + task_threads > threads:
        return False, (
            f"its {task_threads} threads and the live tasks' {used} exceed the "
            f"host's {threads}"
        )
    coming = sum(max(0, task.bytes - task.held) for task in live)
    room = free - kept - coming
    if task_bytes > room:
        return False, (
            f"it needs {_size(task_bytes)}; {_size(free)} free less the reserve "
            f"{_size(kept)} and {_size(coming)} the live tasks may still take "
            f"leave {_size(room)}"
        )
    return True, ""


class Service:
    """One host's service over a jobs folder (:func:`serve`).

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    machine : str
        This host's name in the leases.
    threads : int
        The host's threads: its live tasks' threads together are at most these.
    logs : Path
        Local folder of the tasks' output.

    """

    def __init__(self, jobs: Path, machine: str, threads: int, logs: Path) -> None:
        self.jobs = Path(jobs)
        """The jobs folder."""
        self.machine = machine
        """This host's name."""
        self.threads = threads
        """The host's threads."""
        self.logs = Path(logs)
        """Local folder of the tasks' output."""
        self.revision = revision()
        """The code's revision of this service, when it started."""
        self.request: int | str = AUTO
        """The host's request: :data:`AUTO`, or at most this many tasks."""
        self.reason = ""
        """Why the host took no more tasks in its last round."""
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
        self._passed: set[tuple[str, str]] = set()
        self._unknown: set[str] = set()
        self._uncounted: str | None = None

    def _jobs(self) -> list[tuple[Path, dict]]:
        """The jobs of the folder with their records, oldest first. A job whose
        record cannot be read now, or whose record's state cannot be known
        through the share, is not known in this round (:attr:`_unknown`), never
        taken for no job, and is read again in the next."""
        found, self._unknown = [], set()
        names = sorted(results.retry_transient(lambda: os.listdir(self.jobs)))
        for name in names:
            folder = self.jobs / name
            if name.startswith("."):
                continue
            try:
                if name not in self._records:
                    if not results.exists(folder / JOB):
                        continue
                    self._records[name] = read_job(folder)
            except (OSError, ValueError) as error:
                self._unknown.add(name)
                say(f"The job {name} cannot be read now; trying again: {error}")
                continue
            found.append((folder, self._records[name]))
        return found

    def _refusal(self, record: dict) -> str | None:
        """Why this service takes none of a job's tasks, or None.

        A job's code revision is fixed at its creation; a service of another
        revision takes none of its tasks. A job whose tasks need more threads than
        the host has cannot start here.

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
                f"host has {self.threads}."
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

    def _survivors(self, jobs: list[tuple[Path, dict]]) -> bool:
        """Count this host's claims whose process outlived an earlier service,
        reserving their threads and bytes, and return those whose process is gone
        (:func:`birth`) or never started (no process in the lease), going on with
        the next claim and the next job.

        Returns
        -------
        bool
            Whether every claim of every job was looked at: a job's claims that
            cannot be listed now, or a lease that cannot be read now, leave the
            count incomplete; a marker gone since the listing does not.

        """
        complete = True
        for job, record in jobs:
            try:
                claimed = queue.markers(job, queue.CLAIMED)
            except OSError:
                complete = False
                continue
            for task, token in claimed.items():
                claim = queue.Claim(job, task, token, self.machine)
                if claim.path in self.live:
                    continue
                try:
                    lease = queue.lease(job, task, token)
                except FileNotFoundError:
                    continue  # Gone since the listing: not this host's to count.
                except (OSError, ValueError):
                    complete = False  # Unreadable now: never taken for dead.
                    continue
                if lease.get("machine") != self.machine:
                    continue
                if alive(lease.get("pid"), lease.get("birth")):
                    threads = record["environment"]["threads"]
                    self.live[claim.path] = Live(
                        claim,
                        threads,
                        None,
                        lease["pid"],
                        lease["birth"],
                        _reserved(record, task),
                    )
                    say(f"Task {task} still runs in process {lease['pid']}", job / LOG)
                else:
                    self._release(claim, job / LOG, "its process is gone")
        return complete

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
        need = _reserved(record, claim.task)
        self.live[claim.path] = Live(claim, environment.threads, process, bytes=need)
        say(f"{self.machine} started task {claim.task}", job / LOG)

    def _admit(self, jobs: list[tuple[Path, dict]]) -> None:
        """Start ready tasks, the oldest job's first, while the host admits them
        (:func:`admit`, with the free memory measured at each start).

        A task that the host would admit with no live task waits, and so does
        every task after it, so that the oldest job's tasks go first; one that it
        would not admit even then is passed over, so that it holds up no other
        task, and so is every later task of its job and step that declares at
        least as much. A step passed over is said once, and again only after a
        round that did not pass it over. What the live tasks' processes hold is
        looked at once per round (:attr:`Live.held`). A task whose process cannot
        start is returned, and the host takes no more of its job's tasks. Why the
        host took no more tasks, a task waiting, a step passed over or a job
        refused, is kept in :attr:`reason`.

        """
        self.reason = "No ready task waits."
        passing, self._passed = self._passed, set()
        if not self._taking():
            self.reason = f"It takes no new tasks: {self._idle}."
            return
        for live in self.live.values():
            live.held = live.resident()
        passed = refused = None
        for job, record in jobs:
            why = self._refusal(record)
            if why is not None:
                self._refuse(job, record, why)
                refused = refused or f"It takes no task of job {record['id']}: {why}"
                continue
            threads = record["environment"]["threads"]
            if record["id"] not in self._tasks:
                self._tasks[record["id"]] = queue.tasks(job)
            smallest: dict[str, int] = {}
            for task in queue.ready(job, self._tasks[record["id"]]):
                need = _reserved(record, task.id)
                if need >= smallest.get(task.step, need + 1):
                    continue  # its step passed over a task that declares less
                live = list(self.live.values())
                free, kept = headroom(), reserve()
                started, why = admit(
                    need, threads, live, self.request, self.threads, free, kept
                )
                if not started:
                    alone = free + sum(other.held for other in live)
                    named = f"Task {task.id} of job {record['id']}"
                    if admit(need, threads, (), AUTO, self.threads, alone, kept)[0]:
                        self.reason = f"{named} waits: {why}."
                        return
                    smallest[task.step] = need
                    unfit = f"{named} does not fit this host now: {why}."
                    passed = passed or unfit
                    step = (record["id"], task.step)
                    if step not in passing:
                        said = f"{self.machine} passes over step {step[1]}. {unfit}"
                        say(said, job / LOG)
                    self._passed.add(step)
                    continue
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
        self.reason = passed or refused or self.reason

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
        jobs = self._round("Reading the jobs folder", self._jobs)
        known, jobs = jobs is not None, jobs or []
        for job, record in jobs:

            def reclaim(job=job):
                for task in queue.reclaim(job, self._seen):
                    say(f"Task {task} taken back: its lease went stale", job / LOG)

            self._round(f"Taking back the stale claims of {record['id']}", reclaim)
        counted = self._round(
            "Counting this host's claims", lambda: self._survivors(jobs)
        )
        if not known or counted is not True or self._unknown:
            self._wait_for_count()
            return
        self._uncounted = None
        self._round("Starting ready tasks", lambda: self._admit(jobs))

    def _wait_for_count(self) -> None:
        """Take no new task in a round whose jobs or claims of this host could not
        all be read: a survivor not counted would leave its threads and bytes
        unreserved. Said once until a round counts them all."""
        why = (
            "Not every job or claim of this host can be read now; it takes no new "
            "tasks until it has counted its live tasks."
        )
        if why != self._uncounted:
            say(f"{self.machine}: {why}")
        self.reason = self._uncounted = why


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
        The host's threads: its live tasks' threads together are at most these,
        shared by every job of the folder, the oldest job first; at most the
        logical cores less one, which is the default; 0 takes no new tasks.
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
    cores = max(1, (os.cpu_count() or 2) - 1)
    threads = cores if threads is None else min(threads, cores)
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
