"""A job: one request from a page, frozen at creation.

:func:`create_job` writes a job's folder in the jobs folder: ``job.json`` with its
steps, selection, computing environment, results folder, the code's revision, the
token of each input made outside the job (``bound``), the earlier jobs it
completes (``basis``) and the bytes each task declares (``bytes``, in the Setup's
``sizes``); copies of its configuration files; one file per task in
``tasks/``; and one marker per task in ``queue/`` (:mod:`darsia.jobs.queue`).

Each task's inputs are enumerated from the frozen selection by the scope of each
read (:class:`darsia.jobs.step.Scope`): its own frames, run or part, every run or
part of the selection, or the single result. A task waits for every task of the job
that makes one of its inputs (``after``). Every other input must be stored, and
accepted by the done marker of the task that made it, and its token is bound; else
the job is refused. A result that a task would make and that is stored already is
reused when it serves the request (:func:`darsia.utils.results.reusable`), so its
task makes nothing; a stored result that does not serve refuses the job, since a
result is never replaced. What the creation does not find, a result or a done
marker's token, is looked at once more after the share's cache time.

"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import os
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any

from darsia.jobs import queue
from darsia.jobs.step import (
    CHUNK_FRAMES,
    Each,
    Input,
    Order,
    Scope,
    Selection,
    Step,
    Task,
    file_identity,
    merged,
    missing_runs,
    outputs,
    read_runs,
    request,
    table,
)
from darsia.utils import results
from darsia.utils.locations import Locations

JOB = "job.json"
"""The file of a job's record."""


@dataclass(frozen=True)
class Environment:
    """A job's computing environment, set in each task's process before any
    numerical library loads, so that its numbers depend on neither the host nor the
    order of tasks.

    The threads are fixed per job: every task of the job runs with them, since a
    reduction can change in its last bits with the thread count. So is the compute
    backend, so that a job's frames are never computed on two backends; a host that
    cannot provide the job's kernel or backend takes none of its tasks.

    """

    blas_kernel: str
    """The BLAS kernel (``OPENBLAS_CORETYPE``), e.g. ``"Haswell"``."""
    threads: int = 1
    """Threads of each task: OpenMP, BLAS and OpenCV."""
    backend: str = "cpu"
    """The compute backend: ``"cpu"``, or an OpenCL device class."""

    def to_json(self) -> dict:
        """The environment as plain JSON values."""
        return {
            "blas_kernel": self.blas_kernel,
            "threads": self.threads,
            "backend": self.backend,
        }

    def variables(self) -> dict[str, str]:
        """The process environment variables that set it."""
        threads = str(self.threads)
        return {
            "OPENBLAS_CORETYPE": self.blas_kernel,
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
        }


def _git(folder: Path, *arguments: str) -> str | None:
    """What git answers in a folder, or None when it fails or is missing."""
    try:
        found = subprocess.run(
            ["git", *arguments], cwd=folder, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return found.stdout.strip() if found.returncode == 0 else None


@dataclass(frozen=True)
class Sizes:
    """The sizes a step's declared memory is counted in
    (:class:`darsia.jobs.step.Memory`), from the Setup: the bytes of a value of its
    precision and the pixels of a camera image and of the run grid."""

    itemsize: int
    """Bytes per value of the Setup's precision, e.g. 4 for float32."""
    camera: int
    """Pixels of a camera image."""
    grid: int
    """Pixels of the run grid."""

    def to_json(self) -> dict:
        """The sizes as plain JSON values."""
        return {"itemsize": self.itemsize, "camera": self.camera, "grid": self.grid}


def task_bytes(step: Step, task: Task, selection: Selection, sizes: Sizes) -> int:
    """The bytes a task of a step holds at once by its declaration
    (:meth:`darsia.jobs.step.Memory.bytes`), with the frames of its input scope:
    its own frames and those its inputs cover, a run's result covering the run's
    selected frames.

    Parameters
    ----------
    step : Step
        The task's step.
    task : Task
        The task.
    selection : Selection
        The job's selection.
    sizes : Sizes
        The Setup's sizes.

    Returns
    -------
    int
        The bytes.

    """
    frames = {(task.run, frame) for frame in task.frames}
    for i in task.inputs:
        if i.frame is not None:
            frames.add((i.run, i.frame))
        elif i.run is not None:
            frames.update((i.run, frame) for frame in selection.frames.get(i.run, ()))
    return step.memory.bytes(sizes.itemsize, sizes.camera, sizes.grid, len(frames))


def revision(folder: Path | None = None) -> str | None:
    """The code's revision now: the commit of the checkout this package is run
    from, marked ``-dirty`` when tracked files differ from it; without a checkout
    that tracks the package, e.g. a package installed on a computer without git or
    into an ignored folder of another repository (a virtual environment inside a
    clone), its version and a hash of its source files
    (:func:`installed_revision`); None only when even those cannot be read.

    A job records the revision it was created at. A service whose code is of
    another revision takes none of its tasks, and a task process of another
    revision refuses its task (:mod:`darsia.jobs.service`); each result records
    the revision of the process that made it. Each call looks again, so that a
    service sees its checkout advanced.

    Parameters
    ----------
    folder : Path, optional
        The package's folder; by default this package's.

    """
    folder = Path(__file__).resolve().parents[1] if folder is None else Path(folder)
    if _git(folder, "ls-files", "--error-unmatch", "__init__.py") is not None:
        found = _git(
            folder, "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"
        )
        if found:
            return found
    try:
        return installed_revision(folder)
    except OSError:
        return None


def installed_revision(folder: Path) -> str:
    """The revision of a package without a checkout: its installed version and the
    SHA-256 of its Python source files (each file's path in the package and
    content, in order of their paths), e.g. ``"1.1.0+3f2a..."``, so that two
    installations of other code never share one.

    Parameters
    ----------
    folder : Path
        The package's folder.

    Returns
    -------
    str
        The revision.

    """
    try:
        version = importlib.metadata.version(Path(folder).name)
    except importlib.metadata.PackageNotFoundError:
        version = "unversioned"
    found = hashlib.sha256()
    files = sorted(
        Path(folder).rglob("*.py"), key=lambda p: p.relative_to(folder).as_posix()
    )
    for file in files:
        found.update(file.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        found.update(hashlib.sha256(file.read_bytes()).digest())
    return f"{version}+{found.hexdigest()}"


def read_tables(paths: Sequence[Path]) -> dict[str, Any]:
    """The tables of configuration files, merged in order: a later file's table
    adds to and replaces the earlier ones' keys, as the official configuration
    reads several files (:func:`darsia.jobs.step.merged`)."""
    found: dict[str, Any] = {}
    for path in paths:
        found = merged(found, tomllib.loads(Path(path).read_text(encoding="utf-8")))
    return found


def read_job(job: Path) -> dict:
    """A job's record (``job.json``); one read short is read once more after the
    share's cache time (:func:`darsia.jobs.queue.read_json`)."""
    return queue.read_json(Path(job) / JOB)


def project_of(job: Path, record: Mapping) -> Path:
    """The project folder of a job, where its task processes run."""
    return (Path(job) / record["project"]).resolve()


def locations_of(job: Path, record: Mapping) -> Locations:
    """The locations of a job's results and photos."""
    selection = Selection.from_json(record["selection"])
    return Locations(project_of(job, record), record["results"], selection.sessions)


def discard_failed(job: Path, step: Step, task: str) -> list[str]:
    """Set aside the results a failed task's attempts stored, so that its retry
    (:func:`darsia.jobs.queue.retry_failed`) makes them again after a mend of what
    they are made from, e.g. a protocol or a configured file.

    Only results that name this job and task are set aside, and none is deleted:
    each is renamed beside its location (``.<name>.<token>.discarded``), never over
    a publication. A failed task's results are accepted by no done marker, so no
    consumer reads them. The Jobs page asks the owner before it calls this.

    Parameters
    ----------
    job : Path
        The job's folder.
    step : Step
        The task's step.
    task : str
        The task's name.

    Returns
    -------
    list of str
        The keys of the results set aside.

    Raises
    ------
    ValueError
        If the task is not failed.

    """
    if task not in queue.markers(job, queue.FAILED):
        raise ValueError(f"Task {task} is not failed.")
    record = read_job(job)
    locations = locations_of(job, record)
    found = []
    for output in outputs(step, queue.tasks(job)[task]):
        location = locations.result(
            output.kind, run=output.run, frame=output.frame, part=output.part
        )
        if not results.is_complete(location):
            continue
        stored = results.read(location, results.read_description, committed=True)
        if (stored.get("job"), stored.get("task")) != (record["id"], task):
            continue
        aside = location.with_name(f".{location.name}.{stored['token']}.discarded")
        results.rename(location, aside)
        found.append(output.key)
    return found


def _ordered(steps: Sequence[Step]) -> list[Step]:
    """The steps in an order where each comes after those whose kinds it reads."""
    left, ordered = {step.name: step for step in steps}, []
    while left:
        free = [s for s in left.values() if not set(s.reads) & (set(left) - {s.name})]
        if not free:
            raise ValueError(f"The steps {sorted(left)} read each other's results.")
        for step in free:
            ordered.append(left.pop(step.name))
    return ordered


def _units(step: Step, selection: Selection, tables: Mapping):
    """The units of a step's work before chunking: (run, frames, part)."""
    if step.order is Order.PARTS:
        if step.parts is None:
            return [(None, (), None)]
        parts = step.parts(selection, tables)
        if selection.parts is not None:
            parts = tuple(part for part in parts if part in selection.parts)
        return [(None, (), part) for part in parts]
    if step.order is Order.RUN:
        return [(run, tuple(f), None) for run, f in selection.frames.items()]
    return [(run, (f,), None) for run, fs in selection.frames.items() for f in fs]


def _reads(
    step: Step, run, frames, part, kinds: Mapping[str, Step], selection, tables
) -> list[Input]:
    """The results one unit of a step reads, by the scope of each read: a read of
    every run is of the runs the step reads over the selection (:func:`read_runs`),
    of its own kind those of the other runs, e.g. a reference run's; a unit of a run
    reads the single part of a step in parts that the producer names for its run
    (:attr:`Step.run_part`), any other unit its first part."""
    found, every = [], read_runs(step, selection, tables)
    for kind, wanted in step.reads.items():
        if kind not in kinds:
            raise ValueError(f"No step makes {kind}, which {step.name} reads.")
        producer = kinds[kind]
        stored = producer.stored
        if stored is Each.ONE:
            found.append(Input(kind))
        elif wanted is Scope.ONE and stored is Each.PART:
            if producer.run_part is not None and run is not None:
                one = producer.run_part(run, tables)
            else:
                one = producer.parts(selection, tables)[0]
            found.append(Input(kind, part=one))
        elif wanted is Scope.ONE:
            raise ValueError(f"{kind} is not a single result ({step.name}).")
        elif stored is Each.FRAME:
            runs = every if wanted is Scope.ALL else {run: frames}
            found += [Input(kind, r, f) for r, fs in runs.items() for f in fs]
        elif stored is Each.RUN:
            runs = every if wanted is Scope.ALL else [run]
            found += [Input(kind, r) for r in runs if kind != step.name or r != run]
        elif wanted is Scope.ALL:
            found += [Input(kind, part=p) for p in producer.parts(selection, tables)]
        else:
            found.append(Input(kind, part=part))
    return found


class _Creation:
    """What one job's creation looked at: the stored results, the done markers of
    their tasks and the configured files, each looked at once.

    A look that finds nothing (a result not stored, a done marker that does not
    name a token) may be the share's cache of a moment ago, so after a first plan
    that met one, the creation waits for the share's cache time once and looks at
    each of them again (:meth:`again`).

    """

    def __init__(self, jobs: Path, locations: Locations) -> None:
        self.jobs = jobs
        self.locations = locations
        self.identity = functools.cache(file_identity)
        """The identity of each configured file, looked at once."""
        self.negative = False
        """Whether a look found nothing."""
        self._stored: dict[Path, dict | None] = {}
        self._done: dict[str, dict[str, str]] = {}
        self._tokens: dict[tuple[str, str], frozenset[str]] = {}
        self._missed: set[tuple[str, str]] = set()

    def location(self, i: Input) -> Path:
        """The folder of a result."""
        return self.locations.result(i.kind, run=i.run, frame=i.frame, part=i.part)

    def stored(self, i: Input) -> dict | None:
        """The description of a stored result, or None if none is stored."""
        location = self.location(i)
        if location not in self._stored:
            found = None
            if results.is_complete(location):
                found = results.read(location, results.read_description, committed=True)
            self._stored[location] = found
            self.negative |= found is None
        return self._stored[location]

    def accepted(self, stored: Mapping) -> bool:
        """Whether a stored result's token is named by its task's done marker."""
        job, task = str(stored.get("job")), str(stored.get("task"))
        if job not in self._done:
            folder = self.jobs / job
            done = {}
            if results.exists(folder / JOB):
                done = queue.markers(folder, queue.DONE)
            self._done[job] = done
        if (job, task) not in self._tokens:
            self._tokens[job, task] = queue.accepted(
                self.jobs / job, task, self._done[job]
            )
        if stored["token"] in self._tokens[job, task]:
            return True
        self.negative = True
        self._missed.add((job, task))
        return False

    def again(self) -> None:
        """Wait for the share's cache time, and forget every look that found
        nothing, so that the next plan looks at each once more."""
        sleep(results.SHARE_CACHE)
        self._stored = {k: v for k, v in self._stored.items() if v is not None}
        for job, task in self._missed:
            self._done.pop(job, None)
            self._tokens.pop((job, task), None)
        self._missed.clear()
        self.negative = False


def create_job(
    jobs: Path,
    steps: Sequence[Step],
    selection: Selection,
    config: Path | list[Path],
    environment: Environment | None = None,
    basis: Sequence[str] = (),
    *,
    known: Sequence[Step] = (),
    project: Path | None = None,
    sizes: Sizes | None = None,
) -> Path:
    """Create a job: its record, configuration, tasks and queue markers.

    The record holds the bytes each task declares (:func:`task_bytes`), which a
    host's service admits it by (:func:`darsia.jobs.service.admit`).

    Parameters
    ----------
    jobs : Path
        The jobs folder.
    steps : Sequence[Step]
        The steps of the job.
    selection : Selection
        The runs, frames, domains and parts it is made for.
    config : Path or list of Path
        The configuration file, or files merged in order; they are copied.
    environment : Environment, optional
        The computing environment; by default the configuration's ``[jobs]`` table
        (``blas_kernel``, ``threads`` and optionally ``backend``).
    basis : Sequence[str], optional
        Earlier jobs this one completes, e.g. one whose failed parts it makes again
        with other settings (``Selection.parts``).
    known : Sequence[Step], optional
        The steps of the kinds the job reads but does not make.
    project : Path, optional
        The project folder; by default the first configuration file's folder.
    sizes : Sizes, optional
        The Setup's sizes of the declared memory; without them the tasks declare
        no bytes.

    Returns
    -------
    Path
        The job's folder.

    Raises
    ------
    ValueError
        If an input is neither made by the job nor stored and accepted, if a stored
        result does not serve the job, or if the job names no environment, or
        chosen parts without the job it completes, or if the code's revision is
        not known.

    """
    jobs = Path(jobs)
    paths = [Path(config)] if isinstance(config, (str, Path)) else list(config)
    tables = read_tables(paths)
    project = Path(project) if project is not None else paths[0].parent
    if environment is None:
        try:
            given = table(tables, "jobs")
            environment = Environment(
                given["blas_kernel"], int(given["threads"]), given.get("backend", "cpu")
            )
        except KeyError:
            raise ValueError("The job names no computing environment ([jobs]).")
    if not environment.blas_kernel:
        raise ValueError("The job's computing environment names no BLAS kernel.")
    if selection.parts is not None and not basis:
        raise ValueError("A job over chosen parts names the job it completes.")
    for earlier in basis:
        if not results.exists(jobs / earlier / JOB):
            raise ValueError(f"The basis {earlier} is not a job of {jobs}.")
    code = revision()
    if code is None:
        raise ValueError(
            "The code's revision is not known (no checkout, and its source files "
            "cannot be read); a job locks the revision of its code."
        )
    results_folder = str(table(tables, "data.results"))
    locations = Locations(project, results_folder, selection.sessions)
    creation = _Creation(jobs, locations)
    kinds = {step.name: step for step in [*known, *steps]}
    env = environment.to_json()
    plan = (steps, selection, tables, env, creation, kinds)
    made_tasks, bound, refused = _plan(*plan)
    if creation.negative:  # Look once more at what was not found, after a wait.
        creation.again()
        made_tasks, bound, refused = _plan(*plan)
    if refused:
        raise ValueError("The job is refused:\n" + "\n".join(dict.fromkeys(refused)))

    created = datetime.now(timezone.utc)
    name = f"{created:%Y%m%dT%H%M%S%fZ}-{queue.new_token()[:6]}"
    partial = jobs / f".{name}.partial"
    (partial / "tasks").mkdir(parents=True)
    copies = []
    for index, path in enumerate(paths):
        copies.append(f"config-{index}.toml")
        (partial / copies[-1]).write_bytes(Path(path).read_bytes())
    try:
        relative = os.path.relpath(project.resolve(), (jobs / name).resolve())
    except ValueError:
        relative = str(project.resolve())
    record = {
        "id": name,
        "created": created.isoformat(),
        "steps": {step.name: step.run.__module__ for step in steps},
        "selection": selection.to_json(),
        "environment": env,
        "results": results_folder,
        "project": relative,
        "config": copies,
        "revision": code,
        "bound": {key: token for key, token in bound.items() if token},
        "basis": list(basis),
        "sizes": None if sizes is None else sizes.to_json(),
        "bytes": {
            task.id: (
                0
                if sizes is None
                else task_bytes(kinds[task.step], task, selection, sizes)
            )
            for task in made_tasks
        },
    }
    queue.write_json(partial / JOB, record)
    for state in queue.STATES:
        (partial / state).mkdir()
    for task in made_tasks:
        queue.write_json(partial / "tasks" / f"{task.id}.json", task.to_json())
        (partial / queue.QUEUE / f"{task.id}.{queue.new_token()}").touch()
    _durable(locations.results)
    results.rename(partial, jobs / name)
    return jobs / name


def _durable(folder: Path) -> None:
    """Make the results folder, and on POSIX flush its parent, so that its own
    name is on disk before any result is committed in it (which
    :func:`darsia.utils.results.publish` assumes)."""
    if results.exists(folder):
        return
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        descriptor = os.open(folder.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _plan(steps, selection, tables, env, creation, kinds):
    """The tasks of a job, the token of each input bound, and why it is refused:
    each step's units, with their inputs made by the job or stored and accepted,
    and those whose results are all stored and serve left out (:func:`_reuse`).
    A unit that reads results of other units of its own step, e.g. a reference
    run's, is planned after them."""
    made: dict[str, str] = {}
    bound: dict[str, str] = {}
    refused: list[str] = []
    made_tasks: list[Task] = []
    for step in _ordered(steps):
        missing = missing_runs(step, selection, tables)
        if missing:
            raise ValueError(
                f"{step.name} needs run(s) {missing}, which its configuration names "
                "and the selection lacks."
            )
        pending = [
            (unit, _reads(step, *unit, kinds, selection, tables))
            for unit in _units(step, selection, tables)
        ]
        while pending:
            waiting = {
                o.key
                for unit, _ in pending
                for o in outputs(step, Task("", step.name, *unit, (), ()))
            }
            free = [not {i.key for i in reads} & waiting for _, reads in pending]
            if not any(free):
                raise ValueError(f"The tasks of {step.name} read each other's results.")
            now = [p for p, ok in zip(pending, free) if ok]
            pending = [p for p, ok in zip(pending, free) if not ok]
            kept: list[Task] = []
            for (run, frames, part), reads in now:
                inputs = []
                for i in reads:
                    if i.key in made:
                        inputs.append(
                            Input(i.kind, i.run, i.frame, i.part, made[i.key])
                        )
                        continue
                    if i.key not in bound:
                        stored = creation.stored(i)
                        bound[i.key] = ""
                        if stored is None:
                            refused.append(
                                f"{i.key}, which {step.name} reads, is missing."
                            )
                        elif not creation.accepted(stored):
                            refused.append(
                                f"{i.key}, which {step.name} reads, is not "
                                "accepted by its task's done marker."
                            )
                        else:
                            bound[i.key] = stored["token"]
                    inputs.append(i)
                unit = Task("", step.name, run, frames, part, tuple(inputs), ())
                refusals = _reuse(step, unit, selection, tables, env, creation, bound)
                if refusals is None:
                    kept.append(unit)
                else:
                    refused += refusals
            for task in _grouped(step, kept):
                made_tasks.append(task)
                for o in outputs(step, task):
                    made[o.key] = task.id
    return made_tasks, bound, refused


def _reuse(step, unit, selection, tables, env, creation, bound) -> list | None:
    """Whether a unit of work is made: None if none of its results is stored; else
    an empty list when every one serves (they are bound, and the unit is not made),
    or why the job is refused."""
    outs = outputs(step, unit)
    found = [creation.stored(o) for o in outs]
    if all(stored is None for stored in found):
        return None
    refused = []
    for o, stored in zip(outs, found):
        why = "missing beside stored results of its task"
        if stored is not None:
            why = "not accepted by its task's done marker"
            if creation.accepted(stored):
                why = "made from inputs this job makes again"
                if all(i.task is None for i in unit.inputs):
                    tokens = {i.key: bound[i.key] for i in unit.inputs}
                    frame = o.frame if step.stored is Each.FRAME else None
                    wanted = request(
                        step,
                        unit,
                        selection,
                        tables,
                        env,
                        tokens,
                        frame,
                        locations=creation.locations,
                        identity=creation.identity,
                    )
                    why = results.reusable(stored, wanted)
        if why is not None:
            refused.append(f"{o.key} is stored and does not serve the job: {why}.")
        elif not refused:
            bound[o.key] = stored["token"]
    return refused


def _grouped(step: Step, units: Sequence[Task]) -> list[Task]:
    """The tasks of a step's units: a run's frames in chunks of CHUNK_FRAMES for a
    step over frames, else one task per unit; each named and waiting for the tasks
    that make its inputs."""
    if step.order is Order.FRAMES:
        runs: dict[str, list[Task]] = {}
        for unit in units:
            runs.setdefault(unit.run, []).append(unit)
        groups = [
            (
                f"{step.name}.{run}.{k // CHUNK_FRAMES:04d}",
                run_units[k : k + CHUNK_FRAMES],
            )
            for run, run_units in runs.items()
            for k in range(0, len(run_units), CHUNK_FRAMES)
        ]
    else:
        names = lambda u: ".".join(n for n in (step.name, u.run, u.part) if n)  # noqa
        groups = [(names(unit), [unit]) for unit in units]
    tasks = []
    for name, group in groups:
        inputs = tuple(dict.fromkeys(i for unit in group for i in unit.inputs))
        frames = tuple(f for unit in group for f in unit.frames)
        after = tuple(sorted({i.task for i in inputs if i.task}))
        first = group[0]
        tasks.append(
            Task(name, step.name, first.run, frames, first.part, inputs, after)
        )
    return tasks
