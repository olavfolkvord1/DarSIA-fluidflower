"""The job engine's commands.

- ``python -m darsia.jobs serve <jobs>`` runs this host's service over a jobs
  folder (:func:`darsia.jobs.service.serve`).
- ``python -m darsia.jobs request <jobs> <auto|n>`` sets a host's request: as
  many tasks as its threads and memory admit, at most n, or with 0 none new
  (:func:`darsia.jobs.service.write_request`).
- ``python -m darsia.jobs run <job> <task> <token>`` runs one claimed task in its
  own process, which its service started with the job's computing environment. It
  checks that its code is of the job's revision and that this host provides the
  job's BLAS kernel and compute backend, else it says why in its claim and exits
  (:data:`darsia.jobs.service.REFUSED`). The backend is the job's device class,
  checked on this host with the job's phase-model heads
  (:func:`darsia.backend_named`) and used for the step; each result records its
  device. It sets the threads, renews the lease
  every :data:`darsia.utils.results.BEAT`, runs the step and marks the task done,
  naming its accepted tokens, or failed, with the traceback; the lease is not
  renewed after the step has ended, so that nothing is written into the marker
  between its mark and its move.

:data:`STEPS` holds the steps of the presets' Setup, Calibration, Analysis,
Comparison and Media by the kind each makes. A task waits for the host's lock of a
name, e.g. around a film encoder's open, through its context
(:meth:`darsia.jobs.step.StepContext.host_lock`, a
:class:`darsia.jobs.service.WaitingLock`).

"""

from __future__ import annotations

import argparse
import functools
import importlib
import os
import socket
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
from threadpoolctl import threadpool_info, threadpool_limits

from darsia.jobs import queue, service
from darsia.jobs.job import locations_of, read_job, read_tables, revision
from darsia.jobs.step import Selection, Step, StepContext, Task, table
from darsia.multiphase.phase_model import LogisticHead, read_phase_model
from darsia.presets.workflows.analysis import steps as analysis_steps
from darsia.presets.workflows.calibration import steps as calibration_steps
from darsia.presets.workflows.comparison import steps as comparison_steps
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.media import steps as media_steps
from darsia.presets.workflows.setup import steps as setup_steps
from darsia.utils import results
from darsia.utils.gpu.backend import (
    ComputeBackend,
    backend_named,
    compute_backend,
    use_compute_backend,
)
from darsia.utils.locations import normal_path

STEPS: dict[str, Step] = {
    step.name: step
    for module in (
        setup_steps,
        calibration_steps,
        analysis_steps,
        comparison_steps,
        media_steps,
    )
    for step in module.STEPS
}
"""The presets' steps, by the kind each makes."""

REVOKED = 4
"""Exit code of a task process whose claim was taken back."""


def refusal(environment: Mapping) -> str | None:
    """Why this process does not provide a job's BLAS kernel, or None if it does.

    The kernel is set in the process's environment before NumPy loads; a name the
    library does not know leaves it on its own choice, which this check finds.

    """
    wanted = str(environment["blas_kernel"])
    found = sorted(
        {
            str(row.get("architecture"))
            for row in threadpool_info()
            if row.get("internal_api") == "openblas"
        }
    )
    if [kernel.casefold() for kernel in found] != [wanted.casefold()]:
        return f"This host runs the BLAS kernel {found}, not the job's {wanted}."
    return None


def heads(job: Path, record: Mapping) -> tuple[LogisticHead, LogisticHead] | None:
    """The heads of a job's phase model (``[phase_model]`` of its configuration),
    with which a device's phase probabilities are checked; None for a job
    without one."""
    tables = read_tables([Path(job) / name for name in record["config"]])
    source = tables.get("phase_model")
    if source is None:
        return None
    folder = normal_path(source["folder"], locations_of(job, record).project)
    model = read_phase_model(
        folder, source.get("pixel_size_mm"), canvas=None, accept_canvas=True
    )
    return model.co2_presence, model.gas_within_co2


def backend_of(job: Path, record: Mapping) -> ComputeBackend:
    """A job's compute backend on this host, checked as its task starts
    (:func:`darsia.backend_named`): the CPU, or the device of its class that gives
    the CPU's numbers, for the phase probabilities with the job's heads.

    Raises
    ------
    ValueError
        If this host has no such backend, saying why.

    """
    label = str(record["environment"].get("backend", "cpu"))
    try:
        return backend_named(
            label, heads=None if label == "cpu" else heads(job, record)
        )
    except ValueError as error:
        raise ValueError(f"This host cannot run the job's backend {label}: {error}")


def step_of(record: Mapping, name: str) -> Step:
    """A job's step by its kind: from :data:`STEPS`, else from the module the job
    recorded for it, which lists it in its ``STEPS``."""
    if name in STEPS:
        return STEPS[name]
    module = importlib.import_module(record["steps"][name])
    return {step.name: step for step in module.STEPS}[name]


def other_revision(record: Mapping, code: str | None) -> str | None:
    """Why this process's code is not of a job's revision, or None if it is.

    Parameters
    ----------
    record : Mapping
        The job's record, with its revision.
    code : str or None
        The revision of this process's code, looked at after its imports
        (:func:`darsia.jobs.job.revision`); None when it is not known.

    """
    if code is not None and code == record["revision"]:
        return None
    known = code or "not known"
    return f"This process runs revision {known}, not the job's {record['revision']}."


def context(job: Path, task: Task, token: str, code: str | None = None) -> StepContext:
    """The context of a claimed task (:class:`darsia.jobs.step.StepContext`), whose
    results record the revision of this process's code (by default looked at
    now) and the device of its compute backend, and whose steps wait for the
    host's locks (:class:`darsia.jobs.service.WaitingLock`)."""
    record = read_job(job)
    step = step_of(record, task.step)
    config = [Path(job) / name for name in record["config"]]
    tables = read_tables(config)
    settings = {
        section: settings_from_table(kind, table(tables, section))
        for section, kind in step.settings.items()
    }
    return StepContext(
        step,
        task,
        Selection.from_json(record["selection"]),
        config,
        tables,
        settings,
        locations_of(job, record),
        token,
        record,
        functools.cache(lambda name: queue.accepted(job, name)),
        revision() if code is None else code,
        compute_backend().device,
        service.WaitingLock,
    )


def _beats(claim: queue.Claim, stop: threading.Event, host: str) -> None:
    """Renew the lease every BEAT until stopped, naming this host
    (:func:`darsia.jobs.service.host_identity`); a lost claim ends the process. A
    beat that has begun ends before the thread does, and :func:`run` waits for
    the thread before it marks the task."""
    pid, born, number = os.getpid(), service.birth(os.getpid()), 1
    while not stop.wait(results.BEAT):
        number += 1
        try:
            queue.beat(claim, number, pid, born, host)
        except queue.Revoked as error:
            service.say(f"{error}; this process stops.")
            os._exit(REVOKED)
        except OSError as error:
            service.say(
                f"The lease could not be renewed ({error}); this process stops."
            )
            os._exit(REVOKED)


def run(job: Path, task_name: str, token: str, machine: str) -> int:
    """Run one claimed task in this process (``python -m darsia.jobs run``).

    Parameters
    ----------
    job : Path
        The job's folder.
    task_name : str
        The task.
    token : str
        Its claim token.
    machine : str
        The machine that claimed it.

    Returns
    -------
    int
        The exit code: 0 when the task is marked done or failed,
        :data:`darsia.jobs.service.REFUSED` when its code is not of the job's
        revision or this host does not provide the job's kernel or backend (the
        claim says why, and stays for its service to return), and :data:`REVOKED`
        when the claim was taken back.

    """
    job = Path(job)
    record = read_job(job)
    claim = queue.Claim(job, task_name, token, machine)
    code = revision()
    why = other_revision(record, code) or refusal(record["environment"])
    if why is None:
        try:
            use_compute_backend(backend_of(job, record))
        except ValueError as error:
            why = str(error)
    if why is not None:
        service.say(why)
        try:
            queue.refuse(claim, why)
        except queue.Revoked as error:
            service.say(str(error))
            return REVOKED
        return service.REFUSED
    threads, host = int(record["environment"]["threads"]), service.host_identity()
    cv2.setNumThreads(threads)
    threadpool_limits(threads)
    try:
        queue.beat(claim, 1, os.getpid(), service.birth(os.getpid()), host)
    except queue.Revoked as error:
        service.say(str(error))
        return REVOKED
    stop = threading.Event()
    beats = threading.Thread(target=_beats, args=(claim, stop, host), daemon=True)
    beats.start()
    task = queue.task(job, task_name)
    failure, adopted = None, ()
    try:
        ctx = context(job, task, token, code)
        ctx.step.run(ctx)
        adopted = tuple(ctx.adopted)
    except Exception:
        failure = traceback.format_exc()
        service.say(f"Task {task_name} failed:\n{failure}")
    stop.set()
    beats.join()  # A beat that has begun lands before the mark, never after it.
    try:
        queue.finish(claim, adopted, failure)
    except queue.Revoked as error:
        service.say(str(error))
        return REVOKED
    service.say(f"Task {task_name} {'failed' if failure else 'done'}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the job engine."""
    parser = argparse.ArgumentParser(prog="python -m darsia.jobs")
    commands = parser.add_subparsers(dest="command", required=True)
    serving = commands.add_parser("serve", help="Run this host's service.")
    serving.add_argument("jobs", type=Path)
    serving.add_argument("--machine")
    serving.add_argument("--threads", type=int)
    # A test's own lock file in place of the machine's lock; hidden from the help,
    # since the machine's lock is what keeps one service on a host.
    serving.add_argument("--lock", type=Path, help=argparse.SUPPRESS)
    serving.add_argument("--logs", type=Path)
    serving.add_argument("--poll", type=float, default=service.POLL)
    serving.add_argument("--until-settled", action="store_true")
    running = commands.add_parser("run", help="Run one claimed task.")
    running.add_argument("job", type=Path)
    running.add_argument("task")
    running.add_argument("token")
    running.add_argument("--machine", default=socket.gethostname())
    asking = commands.add_parser(
        "request", help="Set a host's request: auto, or at most n tasks (0 drains)."
    )
    asking.add_argument("jobs", type=Path)
    asking.add_argument("request")
    asking.add_argument("--machine", default=socket.gethostname())
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.job, args.task, args.token, args.machine)
    if args.command == "request":
        try:
            value = args.request if args.request == service.AUTO else int(args.request)
        except ValueError:
            service.say(
                f"A host's request is {service.AUTO!r} or a number of tasks, not "
                f"{args.request!r}."
            )
            return 2
        service.write_request(args.jobs, args.machine, value)
        return 0
    until = None
    if args.until_settled:
        until = functools.partial(service.settled, args.jobs)
    try:
        service.serve(
            args.jobs,
            args.machine,
            threads=args.threads,
            lock=args.lock,
            logs=args.logs,
            poll=args.poll,
            until=until,
        )
    except service.ServiceRunning as error:
        service.say(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
