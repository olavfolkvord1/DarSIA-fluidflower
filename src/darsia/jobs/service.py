"""The service of one host: it runs the ready tasks of a project's jobs.

Each host runs one service, which holds a lock of the whole machine for its life
(:func:`host_lock`), so a second start on the host, under any account, exits and
says why. Every :data:`POLL` seconds the service

- notes its task processes that exited: one that left its claim unmarked is
  returned to the queue once and failed the second time; one that the share's
  outage stopped (:data:`OUTAGE`), and a survivor, whose exit cannot be known, are
  returned, uncounted; and one that refused its
  job (its BLAS kernel, compute backend or code revision is not this host's) and
  said so in its claim is returned, and the host takes no more of that job's
  tasks; when it refused the job's BLAS kernel or compute backend, the host also
  refuses every job of that kernel and backend before any claim, look at its
  memory or start (:meth:`Service._refusal`);
- takes back stale claims of any host (:func:`darsia.jobs.queue.reclaim`);
- counts the claims of its host whose process outlived an earlier service, each
  known by its claim's token, its process and that process's birth time, and
  reserves their threads and bytes; it never takes them back, and returns a claim
  at once when its process is gone, which a process that ended while it looked, or
  that it may not look at, counts as;
- starts ready tasks, the oldest job's first, each as a new process
  (``python -m darsia.jobs run``) with its job's computing environment, while the
  host admits them (:func:`admit`): the live tasks' threads and the task's are at
  most the host's, and the bytes the task reserves are at most the memory free
  now, less the reserve, less what the live tasks may still take. A task reserves
  its estimate (its step's memory in the Setup's sizes and its process's own), or
  its step's largest measured peak scaled to it, whichever is larger
  (:func:`need`): the service looks at what each live task's processes hold once
  per round, and keeps the largest observed peak of each workload for its step.

Each round it also reads its host's request (``machines/<machine>.request.json``,
:func:`write_request`): :data:`AUTO`, or at most a number of tasks, 0 taking no new
ones; a request that cannot be read now keeps the last. It writes its host's
record (``machines/<machine>.json``): the live tasks with the threads and bytes
they reserve, the free memory, the reserve, why it took no more tasks, and each
step's measured peak with the Setup's sizes and the photos' format it was
measured at, which the next service of this host and revision takes up. The
record is created once, never onto another's, and rewritten in place by the one
service whose token it names; a service that finds another's token in it counts no
claims of the host and takes no tasks until that record stays unchanged for
:data:`darsia.utils.results.STALE`, unless that service was of this host and is
gone.

Each part of a round that cannot read the jobs folder now, e.g. a job's file read
short over the share, says why and is tried again in the next round. Through an
outage of the share the service keeps its live tasks, their reservations and its
request; a task process retries its lease for up to
:data:`darsia.utils.results.GRACE`, which with a BEAT is less than STALE, so no
host takes back its claim, and the task completes.

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
import json
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep, time

import psutil

from darsia.jobs import queue
from darsia.jobs.job import (
    JOB,
    Environment,
    project_of,
    read_job,
    revision,
    task_photo,
)
from darsia.jobs.step import Task
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
try:
    import winreg
except ImportError:  # POSIX
    winreg = None

POLL = 5.0
"""Seconds between two rounds of the service."""
REFUSED = 75
"""Exit code of a task process that refused its job, having said why in its claim:
its host does not provide the job's BLAS kernel or compute backend, or its code is
of another revision. Not a code of the C runtime's ``abort`` (3) or of Python."""
OUTAGE = 74
"""Exit code of a task process stopped because the share gave no answer about its
claim within :data:`darsia.utils.results.GRACE`: its lease could not be renewed,
or its mark not written. Its service returns the task, counting no unmarked exit;
another host takes it back once its lease is stale."""
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


MACHINES = "machines"
"""The folder of the hosts' records and requests, in the jobs folder."""


class ServiceRunning(RuntimeError):
    """Another service holds this host's lock."""


def host_identity() -> str:
    """This machine's identity: its network name and its machine id (Windows'
    ``MachineGuid``, else ``/etc/machine-id``; else its boot time), so that two
    machines of one network name, e.g. cloned images, differ, while a machine is
    the same one after a restart. A lease and a host's record name it."""
    machine = None
    if winreg is not None:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                machine = str(winreg.QueryValueEx(key, "MachineGuid")[0])
        except OSError:
            pass
    for path in () if machine else ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            machine = Path(path).read_text(encoding="ascii").strip() or None
        except OSError:
            continue
        if machine:
            break
    return f"{socket.gethostname()}/{machine or f'boot {int(psutil.boot_time())}'}"


def machine_files(jobs: Path, machine: str) -> tuple[Path, Path]:
    """A host's record (``machines/<machine>.json``), which its service writes
    every round, and its request (``machines/<machine>.request.json``), which the
    Jobs page writes."""
    folder = Path(jobs) / MACHINES
    return folder / f"{machine}.json", folder / f"{machine}.request.json"


def _request(value) -> int | str:
    """A request checked: :data:`AUTO`, or a number of tasks of at least 0.

    Raises
    ------
    ValueError
        If it is neither.

    """
    if value == AUTO or (type(value) is int and value >= 0):
        return value
    raise ValueError(
        f"A host's request is {AUTO!r} or a number of tasks, not {value!r}."
    )


def _record_of(raw: bytes) -> dict:
    """A host's record from its bytes; empty when they hold no record."""
    try:
        found = json.loads(raw)
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def write_request(jobs: Path, machine: str, request: int | str) -> None:
    """Set a host's request, e.g. from the Jobs page: :data:`AUTO` takes as many
    tasks as the host's threads and memory admit, a number at most that many, and 0
    no new tasks, so that the host drains. Its service reads it every round.

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    machine : str
        The host's name.
    request : int or str
        :data:`AUTO`, or a number of tasks.

    Raises
    ------
    ValueError
        If the request is neither.

    """
    text = json.dumps(_request(request)) + "\n"
    path = machine_files(jobs, machine)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    results.retry_transient(lambda: path.write_text(text, encoding="utf-8"))


def read_request(jobs: Path, machine: str) -> int | str:
    """A host's request (:func:`write_request`); :data:`AUTO` when none is set.

    Raises
    ------
    ValueError
        If the file holds no request.
    OSError
        If it cannot be read now, e.g. the share is out, also when that is
        reported as a missing path, or the file reads short twice
        (:class:`darsia.utils.results.Unavailable`).

    """
    try:
        return _request(queue.read_json(machine_files(jobs, machine)[1]))
    except FileNotFoundError as error:
        if results.share_out(error):
            raise
        return AUTO


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


class _JobAccounting(ctypes.Structure):
    """Windows' ``JOBOBJECT_BASIC_ACCOUNTING_INFORMATION``."""

    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _BasicLimits(ctypes.Structure):
    """Windows' ``JOBOBJECT_BASIC_LIMIT_INFORMATION``."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobLimits(ctypes.Structure):
    """Windows' ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``, whose peaks the system
    keeps for every job object, also one that limits nothing."""

    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),  # IO_COUNTERS
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_ACCOUNTING, _LIMITS = 1, 9
"""The classes of :class:`_JobAccounting` and :class:`_JobLimits` in
``QueryInformationJobObject``: ``JobObjectBasicAccountingInformation`` and
``JobObjectExtendedLimitInformation``."""


def _job_kernel32():
    """Windows' kernel32 with the types of its job objects' calls set."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    return kernel32


class Family:
    """The processes of one task: on Windows a job object of its own, which the
    task's process enters as it starts, before it starts any process, so that
    every process it starts, and theirs, belongs to it, nested in any job object
    the process was in, as Windows allows since 8. The job object limits nothing
    and ends no process; the system keeps the commit charge of all its processes
    together over their whole life, which no look from outside can miss
    (:meth:`peaks`). A launcher the task's process was started through, e.g. a
    virtual environment's interpreter, which waits for it, stays outside, and its
    few megabytes are not counted. Elsewhere, and where the system refuses a step
    of it, the family knows no peaks. The peaks are a measurement only: they
    decide no admission or reservation.

    Parameters
    ----------
    handle : int, optional
        The job object's handle; None for a family that knows no peaks.

    """

    def __init__(self, handle: int | None = None) -> None:
        self._handle = handle

    @classmethod
    def start(cls) -> Family:
        """Put this process into a new job object of its own, as it starts and
        before it starts any process (``python -m darsia.jobs run``); a family
        that knows no peaks where that fails, never an error."""
        if os.name != "nt":
            return cls()
        try:
            kernel32 = _job_kernel32()
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return cls()
            if not kernel32.AssignProcessToJobObject(
                handle, kernel32.GetCurrentProcess()
            ):
                kernel32.CloseHandle(handle)
                return cls()
        except (AttributeError, OSError, TypeError, ValueError):
            return cls()
        return cls(handle)

    def _query(self, kind: int, found: ctypes.Structure) -> bool:
        """Fill one class of the job object's information
        (``QueryInformationJobObject``); whether the system did."""
        kernel32 = _job_kernel32()
        size = ctypes.sizeof(found)
        return bool(
            kernel32.QueryInformationJobObject(
                self._handle, kind, ctypes.byref(found), size, None
            )
        )

    def peaks(self, attempt: str) -> dict | None:
        """The peak commit charge of the family's processes, bound to an attempt
        of its task: of all of them together over their whole life
        (``job_commit_peak``, the job object's ``PeakJobMemoryUsed``) and of the
        largest one (``process_commit_peak``, ``PeakProcessMemoryUsed``), in bytes,
        with the processes it ever held. These are commit charge alone, neither a
        working set nor physical memory. They are known only once no other
        process of the family runs, so that the family is closed but for this
        process, which then only marks its task; None else, and where the system
        does not say.

        Parameters
        ----------
        attempt : str
            The token of the task's claim, which the peaks are bound to.

        """
        if self._handle is None:
            return None
        accounting, limits = _JobAccounting(), _JobLimits()
        try:
            known = self._query(_ACCOUNTING, accounting)
            known = known and self._query(_LIMITS, limits)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if not known:
            return None
        if accounting.ActiveProcesses != 1:  # another of its processes still runs
            return None
        job, one = int(limits.PeakJobMemoryUsed), int(limits.PeakProcessMemoryUsed)
        if not 0 < one <= job:
            return None
        return {
            "attempt": attempt,
            "job_commit_peak": job,
            "process_commit_peak": one,
            "processes": int(accounting.TotalProcesses),
        }


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
    """The bytes it reserves (:func:`need`)."""
    held: int = 0
    """The bytes its processes held when the service last looked
    (:meth:`resident`), once per round."""
    task: Task | None = None
    """Its task, whose step its measured peak is kept for."""
    estimate: int = 0
    """Its estimate (:func:`estimate`), of which its peak is a share."""
    peak: int = 0
    """The most its processes held at the service's looks (:meth:`resident`)."""

    def resident(self) -> int:
        """The bytes its process and the processes it started hold now (their
        resident sets), e.g. the interpreter a launcher starts or a film's
        encoder; none of a process that is gone or may not be looked at, so that
        its declaration stays reserved. The most they have held is noted in
        :attr:`peak`: on Windows each one's peak working set or peak commit, which
        the system keeps since the process started, elsewhere what they hold
        now."""
        pid = self.process.pid if self.process is not None else self.pid
        try:
            process = psutil.Process(int(pid))
            if self.born is not None and process.create_time() != self.born:
                return 0  # A survivor's PID was reused by another process.
            tree = [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, ValueError):
            return 0
        held = most = 0
        for one in tree:
            try:
                info = one.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            held += info.rss
            peaks = (getattr(info, "peak_wset", 0), getattr(info, "peak_pagefile", 0))
            most += max(info.rss, *peaks)
        self.peak = max(self.peak, int(most))
        return int(held)


def estimate(record: dict, task: str) -> int:
    """The bytes a task of a job holds by its step's memory
    (:func:`darsia.jobs.job.task_bytes`), and its process's own (:data:`PROCESS`)."""
    return record.get("bytes", {}).get(task, 0) + PROCESS


def need(estimate: int, peak: dict | None) -> int:
    """The bytes a task reserves: its estimate, or its step's measured peak scaled
    to it, whichever is larger. Only the array part scales: the process base is
    kept fixed. A smaller task still reserves the observed peak because one
    observation cannot distinguish unmodelled fixed costs from array costs.

    Parameters
    ----------
    estimate : int
        The task's estimate (:func:`estimate`).
    peak : dict or None
        Its step's measured peak (:attr:`Service.peaks`): its ``bytes`` and the
        ``estimate`` of its task; None when none is measured.

    Returns
    -------
    int
        The bytes.

    """
    if not peak or peak["estimate"] <= 0:
        return estimate
    base = peak.get("base", PROCESS)
    previous = peak["estimate"] - base
    if previous <= 0:
        return max(estimate, peak["bytes"])
    arrays = max(0, estimate - PROCESS)
    observed = max(0, peak["bytes"] - base)
    scaled = PROCESS + (arrays * observed + previous - 1) // previous
    return max(estimate, peak["bytes"], scaled)


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


def _environment(record: Mapping) -> tuple[str, str]:
    """A job's computing environment as a host runs it or not: its BLAS kernel and
    its compute backend."""
    found = record["environment"]
    return str(found["blas_kernel"]), str(found.get("backend", "cpu"))


def _environment_refusal(record: Mapping, why: str) -> bool:
    """Whether a task process's refusal of a job (:func:`darsia.jobs.__main__.run`)
    concerns the job's computing environment alone, so that this host runs no job
    of the same BLAS kernel and compute backend: this host's OpenBLAS runs another
    kernel than the job's, or the job's backend names no backend or no device of
    this host (:func:`darsia.backend_named`).

    A refusal of the job's code revision is none; neither is one of a device of
    this host that gave other numbers than the CPU with the job's phase model, nor
    one of a phase model that could not be read, since these concern the job. The
    texts are the task process's own; any other text is no such refusal.

    """
    kernel, backend = _environment(record)
    if why.startswith("This host runs the BLAS kernel ") and why.endswith(
        f", not the job's {kernel}."
    ):
        return True
    prefix = f"This host cannot run the job's backend {backend}: "
    return why in (
        f"{prefix}{backend}: no such device here.",
        f"{prefix}No compute backend is labelled {backend!r}.",
    )


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
    outages : int, optional
        The exits stopped by the share's outage (:data:`OUTAGE`) in a row, on
        this host, at which a task is shown in the host's record with its reason;
        it is never failed for them.

    """

    def __init__(
        self, jobs: Path, machine: str, threads: int, logs: Path, outages: int = 3
    ) -> None:
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
        self.request: int | str | None = None
        """The host's request: :data:`AUTO`, or at most this many tasks; None
        until it has been read once."""
        self.reason = ""
        """Why the host took no more tasks in its last round."""
        self.live: dict[Path, Live] = {}
        """The tasks running on this host, by claimed marker."""
        self.refused: dict[str, str] = {}
        """Jobs this service takes no tasks of, and why."""
        self.unfit: dict[tuple[str, str], tuple[str, str]] = {}
        """The computing environments this host runs no job of, by their BLAS
        kernel and compute backend, each with the job whose task process refused
        it and why (:meth:`_unfit`), for the service's life."""
        self.exits: dict[tuple[str, str], int] = {}
        """Exits without a mark, by job and task."""
        self.outages = outages
        """The exits stopped by the share's outage in a row at which a task is
        shown in the host's record."""
        self.stopped: dict[tuple[str, str], tuple[int, str]] = {}
        """Exits stopped by the share's outage in a row, by job and task, with the
        last one's reason; another end of the task's process clears them."""
        self.peaks: dict[tuple, dict] = {}
        """The largest measured peak of each distinct workload: its ``bytes``,
        the ``estimate``, fixed process ``base``, the job and task, the
        Setup's sizes (``itemsize``, ``camera`` and ``grid`` pixels) and its
        photos' ``format``. They are kept in the host's record, and a later service
        of this host and revision takes them up."""
        self.token = queue.new_token()
        """This service's token in its host's record."""
        self.held = False
        """Whether this service holds its host's record in this round
        (:meth:`_hold`): only then does it count the host's claims, take tasks and
        write the record."""
        self._seen: dict[str, tuple[bytes, float]] = {}
        self._records: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}
        self._complete: set[str] = set()
        self._idle: str | None = None
        self._passed: set[tuple[str, str]] = set()
        self._unknown: set[str] = set()
        self._uncounted: str | None = None

        self._other: tuple[bytes, float] | None = None
        self._expected: bytes | None = None
        self._fence: str | None = None
        self._unread: str | None = None
        self._identity = {
            "machine": machine,
            "token": self.token,
            "host": host_identity(),
            "pid": os.getpid(),
            "birth": birth(os.getpid()),
        }

    def _read_request(self) -> None:
        """Read the host's request; one that cannot be read now, or holds no
        request, keeps the last, said once. Before the first that is read the
        host takes no new tasks."""
        try:
            found = read_request(self.jobs, self.machine)
        except (OSError, ValueError) as error:
            why = f"{self.machine} keeps its request {self.request!r}: {error}"
            if self.request is None:
                why = f"{self.machine} cannot read its request yet: {error}"
            if why != self._unread:
                say(why)
            self._unread = why
            return
        self._unread = None
        if found != self.request:
            say(f"{self.machine}'s request is now {found!r}.")
        self.request = found

    def _hold(self) -> None:
        """Whether this service holds its host's record in this round, which
        fences the host's name: only one service counts the host's claims and
        takes tasks under it.

        The record is this service's when it names its token, or when there is
        none yet or it is empty. A record of an earlier service of this machine
        (:func:`host_identity`) whose process is gone is taken over at once, since
        this service holds the host's lock; a record of another service is taken
        over only once it stays unchanged for :data:`darsia.utils.results.STALE`
        by this service's own clock, and until then this service takes no tasks
        and says why. A record that cannot be read now holds nothing in this
        round. What was read is what :meth:`_record` checks before it writes.

        """
        path = machine_files(self.jobs, self.machine)[0]
        try:
            raw = results.retry_transient(path.read_bytes)
        except FileNotFoundError as error:
            if results.share_out(error):
                raise
            return self._take(None)
        if not raw:
            return self._take(raw)  # An empty record is nobody's.
        if not raw.endswith(b"\n"):
            raise results.Unavailable(f"The record {path} reads short")
        found = _record_of(raw)
        if found.get("token") == self.token:
            return self._take(raw)
        own = found.get("host") == self._identity["host"]
        if own and not alive(found.get("pid"), found.get("birth")):
            if found.get("revision") == self.revision:
                for peak in found.get("peaks") or ():
                    self._keep(peak)
            return self._take(raw)
        now = monotonic()
        if self._other is None or self._other[0] != raw:
            self._other = (raw, now)
        elif now - self._other[1] >= results.STALE:
            return self._take(raw)
        self._fenced(
            f"The record of {self.machine} is written by another service (host "
            f"{found.get('host')}, process {found.get('pid')}); this one takes no "
            f"tasks until it stays unchanged for {results.STALE:g} s."
        )

    def _take(self, raw: bytes | None) -> None:
        """Hold the host's record in this round, as it was read."""
        self.held, self._expected, self._other, self._fence = True, raw, None, None

    def _fenced(self, why: str) -> None:
        """Take no tasks under the host's name, saying why once."""
        if why != self._fence:
            say(f"{self.machine} takes no tasks: {why}")
        self.held, self.reason, self._fence = False, why, why

    def _record(self) -> None:
        """Write the host's record: the live tasks with the threads and bytes they
        reserve, the free memory, the reserve, the request, the reason, the
        computing environments it refuses (:attr:`unfit`) and the tasks stopped by
        the share's outage :attr:`outages` times in a row. It is
        created whole, by a partial file renamed without replacing another's, and
        then rewritten in place by its holder alone: through the one handle it
        reads the record, and writes only if it is still what :meth:`_hold` read
        or names this service's token; else this service stops holding it."""
        live = []
        for task in self.live.values():
            environment = self._records[task.claim.job.name]["environment"]
            live.append(
                {
                    "job": task.claim.job.name,
                    "task": task.claim.task,
                    "threads": task.threads,
                    "bytes": task.bytes,
                    "resident": task.held,
                    "backend": environment.get("backend", "cpu"),
                }
            )
        value = {
            **self._identity,
            "time": time(),
            "revision": self.revision,
            "request": self.request,
            "threads": self.threads,
            "headroom": headroom(),
            "reserve": reserve(),
            "live": live,
            "reason": self.reason,
            "unfit": [
                {"blas_kernel": kernel, "backend": backend, "job": job, "reason": why}
                for (kernel, backend), (job, why) in sorted(self.unfit.items())
            ],
            "peaks": [self.peaks[key] for key in sorted(self.peaks)],
            "outages": [
                {"job": job, "task": task, "times": times, "reason": why}
                for (job, task), (times, why) in self.stopped.items()
                if times >= self.outages
            ],
        }
        text = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        path = machine_files(self.jobs, self.machine)[0]

        def write() -> bool:
            try:
                with open(path, "r+b") as stream:
                    found = stream.read()
                    if found != self._expected:
                        if _record_of(found).get("token") != self.token:
                            return False
                    stream.seek(0)
                    stream.write(text)
                    stream.truncate()
                    return True
            except FileNotFoundError:
                if self._expected is not None:
                    return False  # It was there at the look: another's move.
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f".{path.name}.{self.token}.partial")
            partial.write_bytes(text)
            try:
                results.rename(partial, path)
            except FileExistsError:
                return False
            return True

        if results.retry_transient(write):
            self._expected = text
        else:
            self._fenced(
                f"Another service wrote the record of {self.machine} since this "
                "one looked at it; this one takes no tasks until it holds it again."
            )

    def _jobs(self) -> list[tuple[Path, dict]]:
        """The jobs of the folder with their records, oldest first. A job whose
        record cannot be read now, or whose record's state cannot be known
        through the share, is not known in this round (:attr:`_unknown`), never
        taken for no job, and is read again in the next. Jobs whose immutable
        task set has all reached done are remembered and skipped; a failed job
        stays discoverable because its failed tasks may be retried."""
        found, self._unknown = [], set()
        names = sorted(results.retry_transient(lambda: os.listdir(self.jobs)))
        for name in names:
            folder = self.jobs / name
            if name.startswith(".") or name in self._complete:
                continue
            try:
                if name not in self._records:
                    if not results.exists(folder / JOB):
                        continue
                    self._records[name] = read_job(folder)
                record = self._records[name]
                known = record.get("bytes")
                if known is not None and _all_done(folder, known):
                    self._complete.add(name)
                    continue
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
        the host has cannot start here. A job of a computing environment that a
        task process of an earlier job refused on this host for its BLAS kernel or
        compute backend (:attr:`unfit`) is refused too, before any claim, look at
        memory or start, so that it holds up no task.

        The first job of an environment this host has not refused yet costs one
        discovery: its task is admitted by memory, claimed and started like any
        other, and until then it may wait for memory at the head of the queue.
        The service cannot check an environment in its own process: OpenBLAS
        chooses its kernel when it loads, so only a process started with the
        job's kernel can say whether this host runs it, and a compute backend is
        checked by loading the OpenCL drivers and computing on the device with
        the job's phase model, which the service keeps out of its process.

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
        kernel, backend = _environment(record)
        unfit = self.unfit.get((kernel, backend))
        if unfit is not None:
            return (
                f"Its BLAS kernel {kernel} and compute backend {backend} are not "
                f"this host's, as a task of job {unfit[0]} found: {unfit[1]}"
            )
        return None

    def _taking(self) -> bool:
        """Whether this service takes new tasks now, saying once when it does not:
        it has threads, its request is not 0, and its code is of a known revision,
        the one it started with, which its checkout still holds."""
        now = revision()
        why = None
        if self.threads <= 0:
            why = "its capacity is no threads"
        elif self.request is None:
            why = "its request is not read yet"
        elif self.request == 0:
            why = "its request is 0"
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

    def _unfit(self, record: dict, why: str) -> None:
        """Take no task of any job of a job's computing environment when a task
        process of the job refused the job for its BLAS kernel or compute backend
        (:func:`_environment_refusal`), saying so once; any other refusal stays the
        job's alone."""
        if not _environment_refusal(record, why):
            return
        key = _environment(record)
        if key not in self.unfit:
            self.unfit[key] = (record["id"], why)
            say(
                f"{self.machine} takes no task of any job of the BLAS kernel "
                f"{key[0]} and the compute backend {key[1]}: {why}"
            )

    def _reap(self) -> None:
        """Look at what each live task's processes hold (:attr:`Live.held`), once
        per round, and note the task processes that exited, what they left behind
        and the most they held (:meth:`_note`)."""
        for path, live in list(self.live.items()):
            live.held = live.resident()
            live.bytes = max(live.bytes, live.peak)
            self._note(live)
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
            key = (claim.job.name, claim.task)
            if code == OUTAGE:
                self._stopped(claim, key)
            elif live.process is not None:
                self.stopped.pop(key, None)
            if marked:
                continue
            record = self._records[claim.job.name]
            log = claim.job / LOG
            if code == OUTAGE:
                self._release(claim, log, "its process lost the share for GRACE")
                continue
            if live.process is None:  # a survivor: its exit cannot be known
                self._release(claim, log, "its process, a survivor, ended")
                continue
            if code == REFUSED and lease.get("refused"):
                self._release(claim, log, "its process refused the job")
                self._refuse(claim.job, record, lease["refused"])
                self._unfit(record, lease["refused"])
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
        # A peak learned from one task also raises the bytes its live peers may
        # still take before any new task is admitted in this round.
        for live in self.live.values():
            record = self._records.get(live.claim.job.name)
            if live.task is not None and record is not None:
                live.bytes = max(live.bytes, self._reserve(record, live.task)[0])

    def _note(self, live: Live) -> None:
        """Keep a task's measured peak for its step (:meth:`_keep`), with
        its job's sizes and its photos' format."""
        if live.task is None or live.peak <= 0 or live.estimate <= 0:
            return
        record = self._records.get(live.claim.job.name, {})
        sizes = record.get("memory", {}).get(live.task.id) or record.get("sizes") or {}
        photo = task_photo(live.task)
        self._keep(
            {
                "step": live.task.step,
                "bytes": live.peak,
                "estimate": live.estimate,
                "base": PROCESS,
                "job": live.claim.job.name,
                "task": live.task.id,
                "itemsize": sizes.get("itemsize"),
                "camera": sizes.get("camera"),
                "grid": sizes.get("grid"),
                **(
                    {"grid_shape": sizes["grid_shape"]} if "grid_shape" in sizes else {}
                ),
                "format": sizes.get(
                    "format", None if photo is None else Path(photo).suffix.lower()
                ),
            }
        )

    def _keep(self, peak) -> None:
        """Keep the largest peak for one workload; ignore incomplete records.

        Different resolutions, precisions, formats and configured array counts
        stay separate: a large fixed peak and a large scaling factor need not
        have been observed in the same task.
        """
        if not isinstance(peak, dict) or not isinstance(peak.get("step"), str):
            return
        if not all(type(peak.get(k)) is int for k in ("bytes", "estimate")):
            return
        if peak["estimate"] <= 0 or peak["bytes"] <= 0:
            return
        if type(peak.get("base", PROCESS)) is not int or peak.get("base", PROCESS) < 0:
            return
        if not all(
            peak.get(k) is None or (type(peak[k]) is int and peak[k] >= 0)
            for k in ("itemsize", "camera", "grid")
        ):
            return
        if peak.get("format") is not None and not isinstance(peak["format"], str):
            return
        shape = peak.get("grid_shape")
        if shape is not None and (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[0] * shape[1] != peak.get("grid")
        ):
            return
        key = (
            peak["step"],
            peak.get("format") or "",
            *(peak.get(k) or 0 for k in ("itemsize", "camera", "grid")),
            peak["estimate"],
            peak.get("base", PROCESS),
        )
        best = self.peaks.get(key)
        if best is None or peak["bytes"] > best["bytes"]:
            self.peaks[key] = dict(peak)

    def _reserve(self, record: dict, task: Task) -> tuple[int, int]:
        """The bytes a task of a job reserves (:func:`need`, with its step's
        measured peak), and its estimate (:func:`estimate`)."""
        made = estimate(record, task.id)
        photo = task_photo(task)
        photo_format = (
            record.get("memory", {})
            .get(task.id, {})
            .get("format", None if photo is None else Path(photo).suffix.lower())
        )
        peaks = [
            p
            for p in self.peaks.values()
            if p["step"] == task.step
            and (photo_format is None or p.get("format") in (None, photo_format))
        ]
        return max((need(made, peak) for peak in peaks), default=made), made

    def _stopped(self, claim: queue.Claim, key: tuple[str, str]) -> None:
        """Count a task's exit stopped by the share's outage, with the reason its
        process gave in its output; at :attr:`outages` in a row it is said once in
        the job's log, and the host's record shows it."""
        try:
            lines = self._log(claim).read_text(errors="replace").splitlines()
            why = next(line for line in reversed(lines) if line.strip())
        except (OSError, StopIteration):
            why = "Its process said nothing."
        times = self.stopped.get(key, (0, ""))[0] + 1
        self.stopped[key] = (times, why)
        if times == self.outages:
            say(
                f"Task {claim.task} stopped {times} times in a row, the share giving "
                f"no answer; it is run again each time. The last: {why}",
                claim.job / LOG,
            )

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
                if lease.get("host") != self._identity["host"]:
                    continue  # Another machine's under this name: left to reclaim.
                if alive(lease.get("pid"), lease.get("birth")):
                    try:
                        one = self._tasks.get(record["id"], {}).get(task)
                        one = one or queue.task(job, task)
                    except (OSError, ValueError):
                        complete = False  # Its task unreadable now: not counted.
                        continue
                    reserved, made = self._reserve(record, one)
                    threads = record["environment"]["threads"]
                    self.live[claim.path] = Live(
                        claim,
                        threads,
                        None,
                        lease["pid"],
                        lease["birth"],
                        reserved,
                        task=one,
                        estimate=made,
                    )
                    say(f"Task {task} still runs in process {lease['pid']}", job / LOG)
                else:
                    self._release(claim, job / LOG, "its process is gone")
        return complete

    def _log(self, claim: queue.Claim) -> Path:
        """The local file of a task's output."""
        return self.logs / claim.job.name / f"{claim.task}.{claim.token}.log"

    def _start(self, job: Path, record: dict, claim: queue.Claim, task: Task) -> None:
        """Start a claimed task's process with its job's environment, reserving
        its bytes (:meth:`_reserve`).

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
        reserved, made = self._reserve(record, task)
        self.live[claim.path] = Live(
            claim,
            environment.threads,
            process,
            bytes=reserved,
            task=task,
            estimate=made,
        )
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
        looked at once per round (:meth:`_reap`). A task whose process cannot
        start is returned, and the host takes no more of its job's tasks. Why the
        host took no more tasks, a task waiting, a step passed over or a job
        refused, is kept in :attr:`reason`.

        """
        self.reason = "No ready task waits."
        passing, self._passed = self._passed, set()
        if not self._taking():
            self.reason = f"It takes no new tasks: {self._idle}."
            return
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
                reserved = self._reserve(record, task)[0]
                if reserved >= smallest.get(task.step, reserved + 1):
                    continue  # its step passed over a task that reserves less
                live = list(self.live.values())
                free, kept = headroom(), reserve()
                started, why = admit(
                    reserved, threads, live, self.request, self.threads, free, kept
                )
                if not started:
                    alone = free + sum(other.held for other in live)
                    named = f"Task {task.id} of job {record['id']}"
                    if admit(reserved, threads, (), AUTO, self.threads, alone, kept)[0]:
                        self.reason = f"{named} waits: {why}."
                        return
                    smallest[task.step] = reserved
                    unfit = f"{named} does not fit this host now: {why}."
                    passed = passed or unfit
                    step = (record["id"], task.step)
                    if step not in passing:
                        said = f"{self.machine} passes over step {step[1]}. {unfit}"
                        say(said, job / LOG)
                    self._passed.add(step)
                    continue
                claim = queue.claim(job, task.id, self.machine, self._identity["host"])
                if claim is None:
                    continue
                try:
                    self._start(job, record, claim, task)
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
        self._round("Reading this host's request", self._read_request)
        self.held = False
        self._round("Looking at this host's record", self._hold)
        if not self.held:
            return
        counted = self._round(
            "Counting this host's claims", lambda: self._survivors(jobs)
        )
        if not known or counted is not True or self._unknown:
            self._wait_for_count()
        else:
            self._uncounted = None
            self._round("Starting ready tasks", lambda: self._admit(jobs))
        self._round("Writing this host's record", self._record)

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
    outages: int = 3,
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
    outages : int, optional
        The exits of a task stopped by the share's outage in a row at which the
        host's record shows it (:class:`Service`); 3 by default.

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
        service = Service(jobs, machine, threads, Path(logs), outages)
        say(f"{machine} serves {jobs} with {threads} threads ({service.revision})")
        while True:
            service.poll()
            if until is not None and until() and not service.live:
                return
            sleep(poll)


def _all_done(job: Path, known: Collection[str]) -> bool:
    """Every task has reached done, the only state that cannot be retried.

    Known is the immutable task inventory from the job record, never a folder
    listing. No queue operation moves a done marker back. Failed jobs are not
    final: retry_failed may make them runnable again.
    """
    if not known or not set(known) <= queue.markers(job, queue.DONE).keys():
        return False
    return not any(
        queue.markers(job, state)
        for state in (queue.QUEUE, queue.CLAIMED, queue.FAILED)
    )


def settled(jobs: Path, *, complete: set[str] | None = None) -> bool:
    """Whether no task of any job of a folder can still run: each is done,
    failed, or waits behind a failed task (:func:`darsia.jobs.queue.waiting`); a
    folder that cannot be read now through the share is not settled.

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    complete : set of str, optional
        This caller's cache of fully done jobs, for this jobs folder alone.
        Updated in place. Only jobs with an authoritative task inventory in
        their record whose every task is done enter it. Earlier records without
        that inventory and failed jobs are checked again.

    Returns
    -------
    bool
        Whether no task can still run.

    """
    complete = set() if complete is None else complete
    try:
        for name in sorted(results.retry_transient(lambda: os.listdir(jobs))):
            folder = Path(jobs) / name
            if name.startswith(".") or name in complete:
                continue
            if results.exists(folder / JOB):
                inventory = read_job(folder).get("bytes")
                if inventory is not None and _all_done(folder, inventory):
                    complete.add(name)
                    continue
                state = queue.states(folder)
                known = queue.tasks(folder)
                if inventory is not None and (
                    not set(inventory) <= state.keys()
                    or not set(inventory) <= known.keys()
                ):
                    return False  # an incomplete view is not a settled job
                if queue.waiting(folder, known, state):
                    return False
    except OSError:
        return False
    return True
