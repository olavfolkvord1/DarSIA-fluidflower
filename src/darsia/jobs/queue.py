"""The queue of a job: one marker file per task, whose folder is the task's state.

A task's marker moves from ``queue/`` to ``claimed/`` when a service claims it,
and from there to ``done/`` or ``failed/`` when its process finishes, or back to
``queue/`` when it is released or taken back. Every change of state is one rename
of the one marker (:func:`darsia.utils.results.rename`), never onto another file,
so two services never take one task. Each claim mints a new token, and a marker is
named by its task and the token it moved under (``<task>.<token>``), so every
destination name is the mover's own.

The claimed marker holds the lease, which its owner rewrites every
:data:`darsia.utils.results.BEAT`: the token, the machine, the identity of the
host it runs on, the process and its birth time, a beat number and the time. A
beat is retried while the share is out, for up to
:data:`darsia.utils.results.GRACE`, so that a short outage loses no claim; GRACE
and a BEAT are less than STALE. A service takes a claim back when its lease stays
unchanged for :data:`darsia.utils.results.STALE` by its own clock; the owner's
next beat then finds its marker gone and the owner exits. Only the holder
of the current marker can move it to ``done/``, so a superseded or late owner
commits no terminal state, and a result counts only when its task's done marker
names its token (:func:`accepted`).

"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep, time

from darsia.jobs.step import Task
from darsia.utils import results

QUEUE = "queue"
"""Folder of the tasks waiting to be claimed."""
CLAIMED = "claimed"
"""Folder of the claimed tasks, whose markers hold their leases."""
DONE = "done"
"""Folder of the finished tasks, whose markers name the tokens they accept."""
FAILED = "failed"
"""Folder of the failed tasks, whose markers hold the traceback."""
STATES = (QUEUE, CLAIMED, DONE, FAILED)
"""The folders of a task's states."""


class Revoked(RuntimeError):
    """The claim is no longer its owner's: its marker was taken back or moved on."""


@dataclass(frozen=True)
class Claim:
    """A claimed task, held by the token it was claimed under."""

    job: Path
    """The job's folder."""
    task: str
    """The task's name."""
    token: str
    """The claim token."""
    machine: str
    """The machine that claimed it."""

    @property
    def path(self) -> Path:
        """The claimed marker."""
        return self.job / CLAIMED / f"{self.task}.{self.token}"


def new_token() -> str:
    """A new claim token: 16 hexadecimal digits from the system's random source."""
    return secrets.token_hex(8)


def _text(value: Mapping) -> bytes:
    """A marker's content: one line of JSON and its newline."""
    return (json.dumps(dict(value), ensure_ascii=False) + "\n").encode("utf-8")


def _content(path: Path) -> bytes:
    """A marker's content, through the share's interruptions."""
    return results.retry_transient(lambda: Path(path).read_bytes())


def read_marker(path: Path) -> dict:
    """A marker's content; a short read is read once more after the share's cache
    time, then :class:`darsia.utils.results.Unavailable`.

    Parameters
    ----------
    path : Path
        The marker.

    Returns
    -------
    dict
        Its content; empty for a marker that holds nothing yet.

    """
    for attempt in range(2):
        text = _content(path)
        if not text:
            return {}
        if text.endswith(b"\n"):
            return json.loads(text)
        if attempt == 0:
            sleep(results.SHARE_CACHE)
    raise results.Unavailable(f"The marker {path} reads short twice")


def read_json(path: Path) -> dict:
    """A job's file of JSON that ends with its newline, e.g. its record or a task:
    one read short, without its newline or whole JSON, is read once more after the
    share's cache time, then :class:`darsia.utils.results.Unavailable`.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    dict
        Its content.

    """
    for attempt in range(2):
        text = results.retry_transient(lambda: Path(path).read_bytes())
        if text.endswith(b"\n"):
            try:
                return json.loads(text)
            except ValueError:
                pass
        if attempt == 0:
            sleep(results.SHARE_CACHE)
    raise results.Unavailable(f"The file {path} reads short twice")


def write_json(path: Path, value: Mapping) -> None:
    """Write a job's file of JSON with its newline (:func:`read_json`)."""
    text = json.dumps(dict(value), indent=1, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _write(claim: Claim, value: Mapping) -> None:
    """Rewrite a claimed marker by its token's name, opened and closed at once.

    Raises
    ------
    Revoked
        If the marker is gone.
    OSError
        If the share gave no answer within :data:`darsia.utils.results.GRACE`,
        also as a missing path: an outage is not a lost claim.

    """

    def write() -> None:
        with open(claim.path, "r+b") as stream:
            stream.write(_text(value))
            stream.truncate()

    try:
        results.retry_transient(write)
    except FileNotFoundError as error:
        if results.share_out(error):
            raise
        raise Revoked(f"Task {claim.task} is no longer claimed by {claim.token}") from (
            error
        )


def _no_answer(error: OSError) -> bool:
    """Whether a file error of the share says that it gave no answer: a share that is
    out (:func:`darsia.utils.results.share_out`), as Windows reports it in place of a
    missing path (its errors 53 and 67, a :class:`FileNotFoundError`), or as POSIX
    reports it, by the error's own number (e.g. a host that is down)."""
    return results.share_out(error) and (
        isinstance(error, FileNotFoundError) or os.name != "nt"
    )


def _listing(folder: Path) -> list[str]:
    """The names in a folder of a job, through the share's interruptions; none
    when the folder is not there.

    Raises
    ------
    Unavailable
        If the share gave no answer within :data:`darsia.utils.results.GRACE`,
        also as a missing path: an outage is not an empty folder.

    """
    try:
        return results.retry_transient(lambda: os.listdir(folder))
    except OSError as error:
        if _no_answer(error):
            raise results.Unavailable(f"No answer about {folder}: {error}") from error
        if isinstance(error, FileNotFoundError):
            return []
        raise


def markers(job: Path, state: str) -> dict[str, str]:
    """The tasks in one state, with the token of each one's marker.

    Parameters
    ----------
    job : Path
        The job's folder.
    state : str
        One of :data:`STATES`.

    Returns
    -------
    dict of str to str
        The token of each task's marker, by task.

    """
    found = {}
    for name in _listing(Path(job) / state):
        task, _, token = name.rpartition(".")
        if task and not name.startswith("."):
            found[task] = token
    return found


def task(job: Path, name: str) -> Task:
    """One task of a job, by name."""
    return Task.from_json(read_json(Path(job) / "tasks" / f"{name}.json"))


def tasks(job: Path) -> dict[str, Task]:
    """The tasks of a job, by name.

    Parameters
    ----------
    job : Path
        The job's folder.

    Returns
    -------
    dict of str to Task
        Every task of the job.

    """
    names = _listing(Path(job) / "tasks")
    return {n[:-5]: task(job, n[:-5]) for n in sorted(names) if n.endswith(".json")}


def states(job: Path) -> dict[str, str]:
    """The state of each task of a job that has a marker (:data:`STATES`)."""
    return {task: state for state in STATES for task in markers(Path(job), state)}


def ready(job: Path, known: Mapping[str, Task] | None = None) -> list[Task]:
    """The queued tasks whose inputs' tasks are all done, most frames first.

    Parameters
    ----------
    job : Path
        The job's folder.
    known : Mapping[str, Task], optional
        The job's tasks, if already read (:func:`tasks`).

    Returns
    -------
    list of Task
        The tasks ready to be claimed.

    """
    known = tasks(job) if known is None else known
    done = set(markers(job, DONE))
    queued = [known[name] for name in markers(job, QUEUE) if name in known]
    found = [task for task in queued if set(task.after) <= done]
    return sorted(found, key=lambda task: (-len(task.frames), task.id))


def waiting(job: Path, known: Mapping[str, Task] | None = None) -> set[str]:
    """The tasks that can still run: queued or claimed, and not behind a failed
    task (also through the tasks they wait for).

    Parameters
    ----------
    job : Path
        The job's folder.
    known : Mapping[str, Task], optional
        The job's tasks, if already read (:func:`tasks`).

    Returns
    -------
    set of str
        The names of those tasks; empty when the job is settled.

    """
    known = tasks(job) if known is None else known
    state = states(job)
    stuck = {name for name, value in state.items() if value == FAILED}
    changed = True
    while changed:
        changed = False
        for task in known.values():
            if task.id not in stuck and stuck & set(task.after):
                stuck.add(task.id)
                changed = True
    return {
        name
        for name, value in state.items()
        if value in (QUEUE, CLAIMED) and name not in stuck
    }


def _transfer(source: Path, destination: Path) -> bool:
    """Move a marker to a name of the mover's own token; False if the source was
    gone, e.g. moved by another mover.

    Raises
    ------
    Unavailable
        If the share gave no answer within :data:`darsia.utils.results.GRACE`,
        also as a missing path.

    """
    try:
        results.rename(source, destination)
    except FileExistsError:
        pass  # A replay of the mover's own move: only it names the destination.
    except OSError as error:
        if _no_answer(error):  # not known now, never «gone»
            raise results.Unavailable(f"No answer about {source}: {error}") from error
        if not isinstance(error, FileNotFoundError):
            raise
        return results.exists(destination)
    return True


def claim(job: Path, task: str, machine: str, host: str | None = None) -> Claim | None:
    """Claim a queued task under a new token.

    The claimed marker is then written once with the token, the machine and the
    host's identity, so that what an earlier owner left in it is gone before the
    task's process starts and becomes the marker's only writer.

    Parameters
    ----------
    job : Path
        The job's folder.
    task : str
        The task's name.
    machine : str
        The claiming machine.
    host : str, optional
        The identity of the host it runs on, e.g.
        :func:`darsia.jobs.service.host_identity`.

    Returns
    -------
    Claim or None
        The claim, or None if the task was not queued, e.g. another took it.

    """
    token = markers(job, QUEUE).get(task)
    if token is None:
        return None
    held = Claim(Path(job), task, new_token(), machine)
    if not _transfer(Path(job) / QUEUE / f"{task}.{token}", held.path):
        return None
    try:
        with open(held.path, "r+b") as stream:  # once: no process writes it yet
            first = {"token": held.token, "machine": machine}
            stream.write(_text({**first, **({} if host is None else {"host": host})}))
            stream.truncate()
    except OSError:
        pass  # The task's first beat writes it.
    return held


def beat(
    claim: Claim, number: int, pid: int, birth: float, host: str | None = None
) -> None:
    """Renew a claim's lease: its token, machine, host, process and birth time, the
    beat number and the time.

    Parameters
    ----------
    claim : Claim
        The claim.
    number : int
        The beat's number.
    pid : int
        The task's process.
    birth : float
        The process's birth time (``psutil.Process.create_time``).
    host : str, optional
        The identity of the host it runs on, by which a service knows the claims
        whose process it may look at.

    Raises
    ------
    Revoked
        If the claim was taken back.

    """
    lease = {
        "token": claim.token,
        "machine": claim.machine,
        **({} if host is None else {"host": host}),
        "pid": pid,
        "birth": birth,
        "beat": number,
        "time": time(),
    }
    _write(claim, lease)


def finish(
    claim: Claim, accepted: Iterable[str] = (), traceback: str | None = None
) -> None:
    """Move a claim to ``done/``, naming its accepted tokens, or to ``failed/``.

    Parameters
    ----------
    claim : Claim
        The claim.
    accepted : Iterable[str], optional
        Tokens of earlier attempts' results the task adopted; its own is named
        always.
    traceback : str, optional
        Why the task failed; without it, the task is done.

    Raises
    ------
    Revoked
        If the claim was taken back, so that nothing is committed.

    """
    if traceback is None:
        state = DONE
        content = {"token": claim.token, "accepted": sorted({claim.token, *accepted})}
    else:
        state, content = FAILED, {"token": claim.token, "traceback": traceback}
    _write(claim, {**content, "machine": claim.machine, "time": time()})
    if not _transfer(claim.path, claim.job / state / claim.path.name):
        raise Revoked(f"Task {claim.task} is no longer claimed by {claim.token}")


def refuse(claim: Claim, why: str) -> None:
    """Note in a claimed marker that its host does not run its job, and why, e.g.
    another kernel or code revision; the task's service then returns it
    (:func:`release`) and takes no more of the job's tasks.

    Raises
    ------
    Revoked
        If the claim was taken back.

    """
    _write(claim, {"token": claim.token, "machine": claim.machine, "refused": why})


def release(claim: Claim) -> None:
    """Return a claimed task to the queue, e.g. one whose process exited or whose
    host refuses its environment.

    Raises
    ------
    Revoked
        If the claim was taken back already.

    """
    destination = claim.job / QUEUE / f"{claim.task}.{new_token()}"
    if not _transfer(claim.path, destination):
        raise Revoked(f"Task {claim.task} is no longer claimed by {claim.token}")


def retry_failed(job: Path, task: str) -> bool:
    """Return a failed task of a job to the queue under a new token, e.g. from the
    Jobs page once the cause of its failure is mended.

    Its next attempt adopts the results an earlier attempt stored that still serve
    (:meth:`darsia.jobs.step.StepContext.done`), and the tasks waiting for it can
    run once it is done. So it serves a mend of what no result of the task is
    identified by, e.g. a share that came back, a photo copied again or more
    memory. A mend of what they are identified by (a setting, a protocol, a
    configured file, an input) makes those results not serve, and the retry
    refuses them, since a result is never replaced: set them aside first
    (:func:`darsia.jobs.job.discard_failed`, which the Jobs page does only with
    the owner's word). A job over other settings is a new job
    (``Selection.parts``, ``basis``).

    Parameters
    ----------
    job : Path
        The job's folder.
    task : str
        The task's name.

    Returns
    -------
    bool
        Whether the task was failed and is queued now; False if it was not failed,
        e.g. returned already.

    """
    token = markers(job, FAILED).get(task)
    if token is None:
        return False
    failed = Path(job) / FAILED / f"{task}.{token}"
    return _transfer(failed, Path(job) / QUEUE / f"{task}.{new_token()}")


def lease(job: Path, task: str, token: str) -> dict:
    """The lease in a claimed marker; empty before its first beat.

    Raises
    ------
    FileNotFoundError
        If the marker is gone.

    """
    return read_marker(Path(job) / CLAIMED / f"{task}.{token}")


def reclaim(job: Path, seen: dict[str, tuple[bytes, float]]) -> list[str]:
    """Take back the claims whose lease stayed unchanged for STALE.

    Each claimed marker's content is compared with what this caller saw before,
    by its own clock; a changed lease restarts its clock. A marker that cannot be
    read is never stale. A claim taken back goes to the queue under a new name;
    its owner's next beat finds it gone.

    Parameters
    ----------
    job : Path
        The job's folder.
    seen : dict
        The caller's record of each claimed marker's content and when it was last
        seen to change, by marker name; updated in place.

    Returns
    -------
    list of str
        The tasks taken back.

    """
    folder = Path(job) / CLAIMED
    names = [name for name in _listing(folder) if not name.startswith(".")]
    for name in set(seen) - {f"{job}/{name}" for name in names}:
        if name.startswith(f"{job}/"):
            del seen[name]
    taken = []
    for name in names:
        key = f"{job}/{name}"
        try:
            content = _content(folder / name)
        except OSError:
            continue  # Unreadable now, or gone: never stale.
        now = monotonic()
        if key not in seen or seen[key][0] != content:
            seen[key] = (content, now)
            continue
        if now - seen[key][1] < results.STALE:
            continue
        task, _, _ = name.rpartition(".")
        del seen[key]
        if _transfer(folder / name, Path(job) / QUEUE / f"{task}.{new_token()}"):
            taken.append(task)
    return taken


def accepted(
    job: Path, task: str, done: Mapping[str, str] | None = None
) -> frozenset[str]:
    """The tokens a task's done marker names; empty while it is not done.

    Parameters
    ----------
    job : Path
        The job's folder.
    task : str
        The task's name.
    done : Mapping[str, str], optional
        The job's done markers, if already listed (:func:`markers`).

    Returns
    -------
    frozenset of str
        The accepted tokens.

    """
    token = (markers(job, DONE) if done is None else done).get(task)
    if token is None:
        return frozenset()
    content = read_marker(Path(job) / DONE / f"{task}.{token}")
    return frozenset(content.get("accepted", ()))
