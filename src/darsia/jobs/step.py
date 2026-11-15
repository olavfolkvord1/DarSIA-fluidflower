"""The contract between a step and the job engine.

A step makes one kind of stored result, the kind it is named after, from the
results it reads and the job's configuration. A step module declares each step
once (:class:`Step`): the kinds it reads and what of them one task consumes
(:class:`Scope`), how its work is split into tasks (:class:`Order`), what one of
its results covers (:class:`Each`), the arrays it holds at once (:class:`Memory`)
and its settings, the configuration tables its methods' settings classes are made
from.

A task sees the job only through its :class:`StepContext`: it reads accepted
results of its declared inputs, skips the frames an earlier attempt of the same
task stored, and publishes its own results under its claim token. Each result's
description records what a later request compares to reuse it
(:data:`darsia.utils.results.REUSED`): the step, every configuration section it
reads (each run's as the run reads it, its own tables merged), its scope (what of
the frozen selection it is made from), the job's computing environment, the token
of every input it is made from, the content of each small file its
configuration names, e.g. its protocols, and the size and time of each large one,
e.g. a depth map. Photos are not identified: a photo is never changed, by rule.

"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, TypeVar

from darsia.utils import results
from darsia.utils.locations import Frame, Locations, normal_path

T = TypeVar("T")

CHUNK_FRAMES = 4
"""Frames of a run in one task of a step whose frames are independent."""
RUNS = "runs"
"""The configuration section of each run's own tables, by run."""
FILE = "file:"
"""The prefix of a configured file's name among a result's inputs."""


class Order(Enum):
    """How a step's work is split into tasks."""

    FRAMES = "frames"
    """Parallel: each run's frames in chunks of :data:`CHUNK_FRAMES`, any order."""
    RUN = "run"
    """Linear: one task per run, its frames in time order; runs in parallel."""
    PARTS = "parts"
    """Parallel: the parts the step names (e.g. the fit and its folds, comparison
    times); one task over the whole selection if it names none."""


class Scope(Enum):
    """What of a kind it reads one task consumes."""

    OWN = "own"
    """Its own frames, or its own run's result, or its own part."""
    ALL = "all"
    """Every frame, run or part of the frozen selection, e.g. for a fit, a fold,
    an ensemble, a comparison time or a donor."""
    ONE = "one"
    """The single result, e.g. the phase model; of a step in parts, its first
    part, e.g. the global fit before the folds, or the part it names for the
    reader's run (:attr:`Step.run_part`), e.g. the fit of the run's group."""


class Each(Enum):
    """What one result of a step covers, and so where it is stored
    (:meth:`darsia.utils.locations.Locations.result`)."""

    FRAME = "frame"
    """One frame of a run."""
    RUN = "run"
    """One run."""
    PART = "part"
    """One part of the step, e.g. a fold."""
    ONE = "one"
    """The whole selection, a single result."""


@dataclass(frozen=True)
class Memory:
    """The arrays a task of a step holds at once, declared, never measured."""

    camera: float = 0.0
    """Arrays of camera pixels held at once."""
    grid: float = 0.0
    """Arrays of run-grid pixels held at once."""
    held: float = 0.0
    """Run-grid arrays kept per frame of the input scope."""

    def bytes(
        self, itemsize: int, camera_pixels: int, grid_pixels: int, frames: int
    ) -> int:
        """Bytes a task holds:
        ``itemsize * (camera * camera_pixels + (grid + held * frames) * grid_pixels)``.

        Parameters
        ----------
        itemsize : int
            Bytes per value of the Setup's precision.
        camera_pixels, grid_pixels : int
            Pixels of a camera image and of the run grid.
        frames : int
            Frames of the task's input scope.

        Returns
        -------
        int
            The bytes.

        """
        values = self.camera * camera_pixels
        values += (self.grid + self.held * frames) * grid_pixels
        return int(itemsize * values)


@dataclass(frozen=True)
class Selection:
    """The runs, frames, domains and parts a job is made for, frozen at creation."""

    frames: Mapping[str, tuple[Frame, ...]]
    """Each run's frames in time order."""
    sessions: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """Each run's session folders, by session, relative to the project."""
    domains: tuple[str, ...] = ()
    """The analysis domains a summary covers."""
    parts: tuple[str, ...] | None = None
    """Only these parts of a step in parts, e.g. the failed parts of an earlier job
    made again with other settings; None for all."""

    def to_json(self) -> dict:
        """The selection as plain JSON values."""
        return {
            "frames": {run: [list(f) for f in fs] for run, fs in self.frames.items()},
            "sessions": {run: dict(s) for run, s in self.sessions.items()},
            "domains": list(self.domains),
            "parts": None if self.parts is None else list(self.parts),
        }

    @classmethod
    def from_json(cls, value: Mapping) -> Selection:
        """A selection from its JSON values (:meth:`to_json`)."""
        return cls(
            frames={
                run: tuple(Frame(*f) for f in fs) for run, fs in value["frames"].items()
            },
            sessions={run: dict(s) for run, s in value["sessions"].items()},
            domains=tuple(value["domains"]),
            parts=None if value["parts"] is None else tuple(value["parts"]),
        )


@dataclass(frozen=True)
class Input:
    """One result a task reads, and the task of its job that makes it."""

    kind: str
    """The kind of the result."""
    run: str | None = None
    """Its run, for a result of a frame or a run."""
    frame: Frame | None = None
    """Its frame, for a result of a frame."""
    part: str | None = None
    """Its part, for a result in parts."""
    task: str | None = None
    """The task of the job that makes it; None for a result bound by its token."""

    @property
    def key(self) -> str:
        """The result's location relative to the results folder, as a name."""
        return result_key(self.kind, run=self.run, frame=self.frame, part=self.part)


@dataclass(frozen=True)
class Task:
    """One unit of a step: a chunk of a run's frames, a whole run, or one part."""

    id: str
    """The task's name in its job."""
    step: str
    """The step it runs."""
    run: str | None
    """Its run, or None for a task over the selection."""
    frames: tuple[Frame, ...]
    """Its frames: a chunk of the run's for a step over frames, all of the run's for
    a step per run, none otherwise."""
    part: str | None
    """Its part, or None."""
    inputs: tuple[Input, ...]
    """Every result it reads, enumerated from the selection by the step's scopes."""
    after: tuple[str, ...]
    """The tasks of the job that make its inputs."""

    def to_json(self) -> dict:
        """The task as plain JSON values."""
        return {
            "id": self.id,
            "step": self.step,
            "run": self.run,
            "frames": [list(f) for f in self.frames],
            "part": self.part,
            "inputs": [
                [i.kind, i.run, None if i.frame is None else list(i.frame), i.part]
                + [i.task]
                for i in self.inputs
            ],
            "after": list(self.after),
        }

    @classmethod
    def from_json(cls, value: Mapping) -> Task:
        """A task from its JSON values (:meth:`to_json`)."""
        return cls(
            id=value["id"],
            step=value["step"],
            run=value["run"],
            frames=tuple(Frame(*f) for f in value["frames"]),
            part=value["part"],
            inputs=tuple(
                Input(kind, run, None if frame is None else Frame(*frame), part, task)
                for kind, run, frame, part, task in value["inputs"]
            ),
            after=tuple(value["after"]),
        )


@dataclass(frozen=True)
class Step:
    """Code that makes one kind of stored result, declared once by its module."""

    name: str
    """The kind of result it writes, e.g. ``"phase-maps"``."""
    reads: Mapping[str, Scope]
    """The kinds it reads, and what of each one task consumes."""
    order: Order
    """How its work is split into tasks."""
    memory: Memory
    """The arrays a task holds at once."""
    run: Callable[[StepContext], None]
    """The step's code, given its task's context."""
    settings: Mapping[str, type] = field(default_factory=dict)
    """The configuration section of each settings class its methods read, e.g.
    ``{"phase_rules": darsia.PhaseRuleSettings}``; a dotted section is nested."""
    parts: Callable[[Selection, Mapping[str, Any]], tuple[str, ...]] | None = None
    """The parts of a step in parts, from the selection and the configuration's
    tables."""
    each: Each | None = None
    """What one of its results covers; by default a frame for a step over frames, a
    run for a step per run, a part for a step in parts and the whole selection for
    a step in parts that names none."""
    sections: tuple[str, ...] = ()
    """The configuration sections its code reads per run beside its settings, e.g.
    ``("data", "protocols")`` through the run's official configuration; each is
    recorded as the run reads it, the run's own tables (``[runs.<run>]``) merged
    over the job's."""
    files: Callable[[Mapping[str, Any], Path, Sequence[Path]], Iterable] | None = None
    """The small files and folders its configuration names that it reads, e.g. its
    protocols or a model folder, given a run's tables (the job's for a result of no
    run), the project folder and the run's photo folders; each one's content is
    part of its results' identity (:func:`file_identity`)."""
    stamped: Callable[[Mapping[str, Any], Path, Sequence[Path]], Iterable] | None = None
    """The large files its configuration names that it reads, given as for
    :attr:`files`, e.g. a depth map or an image mapping; each one's size and
    modification time are part of its results' identity (:func:`file_stamp`).
    Photos and stored results are not named here: a photo is never changed, by
    rule, and a result is bound by its token."""
    domains: bool = False
    """Whether its results are made from the selection's domains, which their scope
    then records."""
    selected: bool = True
    """Whether a result of no run is made from the selection's frames; False for a
    step that reads none of them, e.g. the copy of a configured model, whose scope
    is then its part alone."""
    runs: Callable[[Mapping[str, Any]], Sequence[str]] | None = None
    """The selected runs a task of the step over the selection reads, named by the
    configuration's tables, e.g. the runs a measurement is made on; by default
    every selected run (:func:`read_runs`)."""
    required: Callable[[Mapping[str, Any], Sequence[str]], Iterable[str]] | None = None
    """The runs the configuration names that a job of the step must select, given
    the configuration's tables and the selected runs, e.g. the fit runs among which
    a run's overflow ranks its donors and over which the calibration fits (of the
    calibration groups of the selected runs), or the distances' reference run; a
    job without one of them is refused (:func:`missing_runs`)."""
    run_part: Callable[[str, Mapping[str, Any]], str] | None = None
    """For a step in parts, the part that a task of a run reads by
    :attr:`Scope.ONE`, given the run and the configuration's tables, e.g. the fit
    of the run's calibration group; by default, and for a task of no run, the
    step's first part."""

    @property
    def stored(self) -> Each:
        """What one of its results covers."""
        if self.each is not None:
            return self.each
        if self.order is Order.FRAMES:
            return Each.FRAME
        if self.order is Order.RUN:
            return Each.RUN
        return Each.PART if self.parts is not None else Each.ONE


def read_runs(
    step: Step, selection: Selection, tables: Mapping[str, Any]
) -> dict[str, tuple[Frame, ...]]:
    """The frames of the runs a step reads over the whole selection (each read of
    :attr:`Scope.ALL`): every selected run's, or those of the runs its
    configuration names (:attr:`Step.runs`).

    Raises
    ------
    ValueError
        If the configuration names a run the selection lacks.

    """
    if step.runs is None:
        return dict(selection.frames)
    named = list(step.runs(tables))
    missing = [run for run in named if run not in selection.frames]
    if missing:
        raise ValueError(f"{step.name} reads run(s) {missing}, which are not selected.")
    return {run: selection.frames[run] for run in named}


def missing_runs(
    step: Step, selection: Selection, tables: Mapping[str, Any]
) -> list[str]:
    """The runs a step needs, by its configuration (:attr:`Step.required`), that
    the selection lacks."""
    selected = list(selection.frames)
    named = [] if step.required is None else list(step.required(tables, selected))
    return [run for run in dict.fromkeys(named) if run not in selection.frames]


def result_key(
    kind: str,
    *,
    run: str | None = None,
    frame: Frame | None = None,
    part: str | None = None,
) -> str:
    """A result's location relative to the results folder, as a name: the path of
    :meth:`darsia.utils.locations.Locations.result` below it, with forward slashes.

    """
    names = [kind] + ([run] if run is not None else [])
    names += list(frame) if frame is not None else []
    return "/".join(names + ([part] if part is not None else []))


def table(tables: Mapping[str, Any], section: str) -> Any:
    """A configuration table by its section; a dotted section is nested.

    Raises
    ------
    KeyError
        If the configuration has no such section.

    """
    value: Any = tables
    for name in section.split("."):
        if not isinstance(value, Mapping) or name not in value:
            raise KeyError(f"The configuration has no [{section}] section.")
        value = value[name]
    return value


def plain(value: Any) -> Any:
    """A value as plain JSON values, e.g. a configuration table with its dates."""
    return json.loads(json.dumps(value, default=str))


def merged(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Configuration tables merged as the official configuration merges its files:
    a table of the update adds to and replaces the base's keys, in depth."""
    found = dict(base)
    for key, value in update.items():
        if isinstance(found.get(key), Mapping) and isinstance(value, Mapping):
            found[key] = merged(found[key], value)
        else:
            found[key] = value
    return found


def run_tables(tables: Mapping[str, Any], run: str) -> dict[str, Any]:
    """The configuration tables of one run: its own tables (``[runs.<run>]``)
    merged over the job's."""
    return merged(tables, tables.get(RUNS, {}).get(run, {}))


def file_identity(path: Path) -> str:
    """The identity of a small file or folder that a configuration names: the
    SHA-256 of its content, of a folder its files' names and contents, through the
    share's interruptions; ``"missing"`` if there is none.

    Parameters
    ----------
    path : Path
        The file or folder.

    Returns
    -------
    str
        The identity.

    Raises
    ------
    darsia.utils.results.Unavailable
        If the share gave no answer within the retries, also as a missing path:
        an outage is not a missing file.

    """
    path = Path(path)

    def digest() -> str:
        if not path.is_dir():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        found = hashlib.sha256()
        files = [p for p in path.rglob("*") if p.is_file()]
        for file in sorted(files, key=lambda p: p.relative_to(path).as_posix()):
            found.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
            found.update(hashlib.sha256(file.read_bytes()).digest())
        return found.hexdigest()

    return _known(path, lambda: results.retry_transient(digest))


def file_stamp(path: Path) -> str:
    """The identity of a large file that a configuration names: its size and
    modification time, through the share's interruptions; ``"missing"`` if there
    is none. A file written again, even with the same content, is another one.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    str
        The identity, e.g. ``"size 1024, modified 1731628800123456789"``.

    Raises
    ------
    darsia.utils.results.Unavailable
        If the share gave no answer within the retries, also as a missing path.

    """

    def stamp() -> str:
        found = results.retry_transient(lambda: Path(path).stat())
        return f"size {found.st_size}, modified {found.st_mtime_ns}"

    return _known(path, stamp)


def _known(path: Path, look: Callable[[], str]) -> str:
    """A file's identity by a look through the share's interruptions;
    ``"missing"`` only for a file that is not there, :class:`Unavailable` when the
    share is out (:func:`darsia.utils.results.share_out`)."""
    try:
        return look()
    except OSError as error:
        if results.share_out(error):
            raise results.Unavailable(f"No answer about {path}: {error}") from error
        if isinstance(error, FileNotFoundError):
            return "missing"
        raise


def _file_name(path: Path, project: Path) -> str:
    """A configured file's name among a result's inputs: its path relative to the
    project, or absolute if it lies outside."""
    path = PurePath(normal_path(path))
    try:
        return FILE + path.relative_to(normal_path(project)).as_posix()
    except ValueError:
        return FILE + path.as_posix()


def configured(
    step: Step,
    task: Task,
    selection: Selection,
    tables: Mapping[str, Any],
    locations: Locations,
    identity: Callable[[Path], str] = file_identity,
    files: Mapping[str, str] | None = None,
) -> tuple[dict, dict[str, str]]:
    """What of the configuration one result of a task is made from: the tables of
    every section its step reads, and the identity of every file its step reads
    that the configuration names: a small file's content, a large file's size and
    time (:func:`file_stamp`).

    The sections of the step's settings are the job's; its other sections
    (:attr:`Step.sections`) and its files (:attr:`Step.files`,
    :attr:`Step.stamped`) are each run's, of the task's run or of every run it
    reads over the selection (:func:`read_runs`).

    Parameters
    ----------
    step, task : Step, Task
        The step and its task.
    selection : Selection
        The job's selection.
    tables : Mapping[str, Any]
        The job's configuration tables.
    locations : Locations
        The project's results and photos.
    identity : Callable[[Path], str], optional
        The identity of a file (:func:`file_identity`), e.g. one remembered.
    files : Mapping[str, str], optional
        The identities of the configured files taken earlier, e.g. when the task
        started; by default each file is looked at now.

    Returns
    -------
    dict
        The settings: each section's table, and each run's sections under
        ``runs``.
    dict of str to str
        The identity of each file, by its name among the inputs.

    """
    settings = {section: table(tables, section) for section in step.settings}
    if task.run is not None:
        runs: list[str | None] = [task.run]
    else:
        runs = list(read_runs(step, selection, tables)) if step.selected else [None]
    own = {run: tables if run is None else run_tables(tables, run) for run in runs}
    if step.sections:
        settings[RUNS] = {
            run: {s: own[run][s] for s in step.sections if s in own[run]}
            for run in runs
            if run is not None
        }
    if files is not None:
        return plain(settings), dict(files)
    files = {}
    for named, known in ((step.files, identity), (step.stamped, file_stamp)):
        for run in runs if named is not None else ():
            folders = list(locations.sessions.get(run, {}).values()) if run else []
            for path in named(own[run], locations.project, folders):
                full = normal_path(path, locations.project)
                files[_file_name(full, locations.project)] = known(full)
    return plain(settings), files


def outputs(step: Step, task: Task) -> tuple[Input, ...]:
    """The results a task of a step publishes, each as an :class:`Input` without a
    producer."""
    stored = step.stored
    if stored is Each.FRAME:
        return tuple(Input(step.name, task.run, frame) for frame in task.frames)
    if stored is Each.RUN:
        return (Input(step.name, task.run),)
    if stored is Each.PART:
        return (Input(step.name, part=task.part),)
    return (Input(step.name),)


def scope(
    step: Step,
    task: Task,
    selection: Selection,
    tables: Mapping[str, Any],
    frame: Frame | None = None,
) -> dict:
    """What of the frozen selection one result of a task is made from.

    A frame's result of a step over frames is made from that frame alone; a result
    of a task per run from the run's frames; any other from its part and, unless
    its step reads no selection (:attr:`Step.selected`), the frames of the runs it
    reads (:func:`read_runs`, by default the whole selection's). The selection's
    domains are recorded for a step that reads them (:attr:`Step.domains`).

    """
    if step.order is Order.FRAMES and frame is not None:
        found: dict = {"run": task.run, "frames": [list(frame)]}
    elif task.run is not None:
        found = {"run": task.run, "frames": [list(f) for f in task.frames]}
    else:
        found = {"part": task.part}
        if step.selected:
            read = read_runs(step, selection, tables)
            found["frames"] = {run: [list(f) for f in fs] for run, fs in read.items()}
    if step.domains:
        found["domains"] = list(selection.domains)
    return found


def made_from(step: Step, task: Task, frame: Frame | None = None) -> tuple[Input, ...]:
    """The inputs one result of a task is made from: for a frame's result of a
    step over frames, the inputs of that frame and every input of no frame; else
    every input of the task."""
    if step.order is Order.FRAMES and frame is not None:
        return tuple(i for i in task.inputs if i.frame is None or i.frame == frame)
    return task.inputs


def request(
    step: Step,
    task: Task,
    selection: Selection,
    tables: Mapping[str, Any],
    environment: Mapping[str, Any],
    tokens: Mapping[str, str],
    frame: Frame | None = None,
    *,
    locations: Locations,
    identity: Callable[[Path], str] = file_identity,
    files: Mapping[str, str] | None = None,
) -> dict:
    """What one result of a task would record now (:data:`REUSED`), to publish it or
    to compare a stored result with it (:func:`darsia.utils.results.reusable`).

    Parameters
    ----------
    step, task : Step, Task
        The step and its task.
    selection : Selection
        The job's selection.
    tables : Mapping[str, Any]
        The job's configuration tables.
    environment : Mapping[str, Any]
        The job's computing environment.
    tokens : Mapping[str, str]
        The token of each input, by its key (:attr:`Input.key`).
    frame : Frame, optional
        The frame of a frame's result.
    locations : Locations
        The project's results and photos, where the configured files are found.
    identity : Callable[[Path], str], optional
        The identity of a file (:func:`file_identity`), e.g. one remembered.
    files : Mapping[str, str], optional
        The identities of the configured files taken earlier (:func:`configured`).

    Returns
    -------
    dict
        The step, the tables of every configuration section it reads
        (:func:`configured`), the result's scope, the environment, and the token of
        every input it is made from with the identity of every configured file it
        reads.

    """
    used = made_from(step, task, frame)
    settings, files = configured(
        step, task, selection, tables, locations, identity, files
    )
    return {
        "step": step.name,
        "settings": settings,
        "scope": scope(step, task, selection, tables, frame),
        "environment": plain(dict(environment)),
        "inputs": {**{i.key: tokens[i.key] for i in used}, **files},
    }


class StepContext:
    """What one task of a step reads and writes: its only I/O.

    Parameters
    ----------
    step : Step
        The step.
    task : Task
        Its task.
    selection : Selection
        The job's frozen selection.
    config : Sequence[Path]
        The job's configuration files, in the order they are merged.
    tables : Mapping[str, Any]
        Their merged tables.
    settings : Mapping[str, Any]
        Each section's settings, made from its table
        (:func:`darsia.presets.workflows.config.utils.settings_from_table`).
    locations : Locations
        The project's results and photos.
    token : str
        The task's claim token.
    job : Mapping[str, Any]
        The job's record: its ``id``, ``environment``, ``revision`` and the token of
        each input made outside the job (``bound``, by key).
    accepted : Callable[[str], frozenset[str]]
        The tokens a task of the job accepts, by its done marker
        (:func:`darsia.jobs.queue.accepted`).
    revision : str
        The revision of the code this process runs, which its results record
        (:func:`darsia.jobs.job.revision`).
    device : str, optional
        The device of this process's compute backend, driver included, which its
        results record (:attr:`darsia.ComputeBackend.device`).
    locks : Callable[[str], AbstractContextManager], optional
        The host's lock of a name that its taker waits for
        (:class:`darsia.jobs.service.WaitingLock`); by default none, e.g. for a
        task run in a test's process.

    """

    def __init__(
        self,
        step: Step,
        task: Task,
        selection: Selection,
        config: Sequence[Path],
        tables: Mapping[str, Any],
        settings: Mapping[str, Any],
        locations: Locations,
        token: str,
        job: Mapping[str, Any],
        accepted: Callable[[str], frozenset[str]],
        revision: str,
        device: str = "cpu",
        locks: Callable[[str], AbstractContextManager] | None = None,
    ) -> None:
        self.step = step
        """The step."""
        self.task = task
        """Its task."""
        self.selection = selection
        """The job's selection."""
        self.config = tuple(Path(path) for path in config)
        """The job's configuration files."""
        self.tables = tables
        """Their merged tables."""
        self.settings = settings
        """Each section's settings."""
        self.locations = locations
        """The project's results and photos."""
        self.token = token
        """The task's claim token."""
        self.adopted: set[str] = set()
        """Tokens of an earlier attempt's results this task adopted."""
        self.revision = revision
        """The revision of the code that makes this task's results."""
        self.device = device
        """The device that makes this task's results."""
        self._job = job
        self._accepted = accepted
        self._locks = locks
        self._tokens: dict[str, str] = {}
        self._outputs = {i.key: i for i in outputs(step, task)}
        self._files = self._configured_files()

    def host_lock(self, name: str) -> AbstractContextManager:
        """The host's lock of a name, which a task waits for while another holds
        it, e.g. around a film encoder's open; none for a task run in a test's
        process.

        Parameters
        ----------
        name : str
            The lock's name.

        Returns
        -------
        AbstractContextManager
            The lock, to be held with ``with`` (also more than once, one after the
            other).

        """
        return nullcontext() if self._locks is None else self._locks(name)

    def _input(self, kind, run, frame, part) -> Input:
        """The declared input of a kind, frame, part and run (by default the
        task's); a single result read in parts is found without its part."""
        found = [
            i
            for i in self.task.inputs
            if i.kind == kind
            and i.frame == frame
            and (part is None or i.part == part)
            and (i.run == run if run is not None else i.run in (None, self.task.run))
        ]
        if len(found) != 1:
            key = result_key(kind, run=run, frame=frame, part=part)
            raise ValueError(f"Step {self.step.name} does not read one {key}.")
        return found[0]

    def _location(self, i: Input) -> Path:
        """The folder of an input or output."""
        return self.locations.result(i.kind, run=i.run, frame=i.frame, part=i.part)

    def token_of(self, i: Input) -> str:
        """The token of an input, checked: named by its producer's done marker, or
        bound by the job.

        Raises
        ------
        PermissionError
            If the stored result's token is not accepted.

        """
        if i.key not in self._tokens:
            location = self._location(i)
            stored = results.read(location, results.read_description, committed=True)
            token = stored["token"]
            if i.task is None:
                allowed = frozenset({self._job["bound"][i.key]})
            else:
                allowed = self._accepted(i.task)
            if token not in allowed:
                raise PermissionError(
                    f"{i.key} holds token {token}, which its producer did not accept."
                )
            self._tokens[i.key] = token
        return self._tokens[i.key]

    def read(
        self,
        kind: str,
        reader: Callable[[Path], T],
        *,
        run: str | None = None,
        frame: Frame | None = None,
        part: str | None = None,
    ) -> T:
        """Read an accepted result of a declared input with its method's reader.

        Parameters
        ----------
        kind : str
            The kind of the result.
        reader : Callable[[Path], T]
            The method's reader of the result's folder.
        run : str, optional
            Its run; by default the task's.
        frame : Frame, optional
            Its frame.
        part : str, optional
            Its part.

        Returns
        -------
        T
            What the reader returns.

        """
        i = self._input(kind, run, frame, part)
        self.token_of(i)
        return results.read(self._location(i), reader, committed=True)

    def _configured_files(self) -> dict[str, str]:
        """The identities of the configured files the task reads, looked at now."""
        return configured(
            self.step, self.task, self.selection, self.tables, self.locations
        )[1]

    def _request(self, frame: Frame | None) -> dict:
        """What one of this task's results records now, with the identities of its
        configured files as they were when the task started."""
        used = made_from(self.step, self.task, frame)
        tokens = {i.key: self.token_of(i) for i in used}
        return request(
            self.step,
            self.task,
            self.selection,
            self.tables,
            self._job["environment"],
            tokens,
            frame,
            locations=self.locations,
            files=self._files,
        )

    def _own(self, output: Input, stored: Mapping, frame: Frame | None) -> None:
        """Accept a stored result at an own location: this claim's, or an earlier
        attempt's of the same task that serves now, which is adopted.

        Raises
        ------
        FileExistsError
            If it is another's, naming its token.

        """
        token = stored.get("token")
        if token == self.token:
            return
        why = "made by another job or task"
        if stored.get("job") == self._job["id"] and stored.get("task") == self.task.id:
            why = results.reusable(stored, self._request(frame))
            if why is None:
                self.adopted.add(token)
                return
        raise FileExistsError(f"{output.key} holds token {token}: {why}.")

    def _output(self, run, frame, part) -> Input:
        """The own result of a frame or part."""
        key = result_key(
            self.step.name,
            run=self.task.run if run is None else run,
            frame=frame,
            part=part if part is not None else self.task.part,
        )
        if self.step.stored is Each.ONE:
            key = result_key(self.step.name)
        if key not in self._outputs:
            raise ValueError(f"Task {self.task.id} does not make {key}.")
        return self._outputs[key]

    def done(self, frame: Frame) -> bool:
        """Whether a frame's result of this task is stored, by this claim or by an
        earlier attempt of the same task that serves now (adopted), so that a
        resumed task skips it.

        Raises
        ------
        FileExistsError
            If another's result is stored there.

        """
        output = self._output(None, frame, None)
        location = self._location(output)
        if not results.is_complete(location):
            return False
        stored = results.read(location, results.read_description, committed=True)
        self._own(output, stored, frame)
        return True

    def own(self, reader: Callable[[Path], T], *, frame: Frame) -> T:
        """Read a frame's result of this task that :meth:`done` accepted, e.g. a
        resumed run's state at an earlier frame."""
        output = self._output(None, frame, None)
        return results.read(self._location(output), reader, committed=True)

    def publish(
        self,
        write: Callable[[Path], None],
        *,
        run: str | None = None,
        frame: Frame | None = None,
        part: str | None = None,
    ) -> None:
        """Publish one of this task's results under its claim token.

        The description records the request (:func:`request`), the job, the task,
        the revision of the code that made it and the device it ran on. A location
        that already holds an earlier attempt's result of the same task that serves
        now is adopted instead. Nothing is published once a configured file the task
        reads has changed since it started.

        Parameters
        ----------
        write : Callable[[Path], None]
            The method's writer of the result's files into a folder.
        run : str, optional
            The run; by default the task's.
        frame : Frame, optional
            The frame, for a frame's result.
        part : str, optional
            The part; by default the task's.

        Raises
        ------
        FileExistsError
            If another's result is stored there.
        RuntimeError
            If a configured file the task reads changed since it started, naming
            it.

        """
        output = self._output(run, frame, part)
        location = self._location(output)
        now = self._configured_files()
        changed = sorted(
            k
            for k in now.keys() | self._files.keys()
            if now.get(k) != self._files.get(k)
        )
        if changed:
            raise RuntimeError(
                f"{', '.join(changed)} changed while task {self.task.id} read it; "
                "it publishes nothing more."
            )
        description = {
            **self._request(frame),
            "job": self._job["id"],
            "task": self.task.id,
            "revision": self.revision,
            "device": self.device,
        }
        try:
            results.publish(
                location, write, description, self.token, root=self.locations.results
            )
        except FileExistsError:
            stored = results.read(location, results.read_description, committed=True)
            self._own(output, stored, frame)

    def photo(self, run: str, frame: Frame) -> Path:
        """The file of a frame's photo (:meth:`Locations.photo`)."""
        return self.locations.photo(run, frame)
