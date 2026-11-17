"""The job engine's commands.

- ``python -m darsia.jobs serve <jobs>`` runs this host's service over a jobs
  folder (:func:`darsia.jobs.service.serve`).
- ``python -m darsia.jobs run <job> <task> <token>`` runs one claimed task in its
  own process, which its service started with the job's computing environment. It
  checks that its code is of the job's revision and that this host provides the
  job's BLAS kernel and compute backend, else it says why in its claim and exits
  (:data:`darsia.jobs.service.REFUSED`). It sets the threads and the storage of
  the results' arrays (the ``[storage]`` table of the job's configuration,
  :func:`darsia.use_storage`), and renews the lease
  every :data:`darsia.utils.results.BEAT`, runs the step and marks the task done,
  naming its accepted tokens, or failed, with the traceback; the lease is not
  renewed after the step has ended, so that nothing is written into the marker
  between its mark and its move.

:data:`STEPS` holds the steps of the presets' Setup, Calibration, Analysis and
Comparison by the kind each makes.

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
from darsia.presets.workflows.analysis import steps as analysis_steps
from darsia.presets.workflows.calibration import steps as calibration_steps
from darsia.presets.workflows.comparison import steps as comparison_steps
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.setup import steps as setup_steps
from darsia.utils import results
from darsia.utils.storage import StorageSettings, use_storage

STEPS: dict[str, Step] = {
    step.name: step
    for module in (setup_steps, calibration_steps, analysis_steps, comparison_steps)
    for step in module.STEPS
}
"""The presets' steps, by the kind each makes."""

REVOKED = 4
"""Exit code of a task process whose claim was taken back."""


BACKENDS = ("cpu",)
"""The compute backends a task process provides."""


def refusal(environment: Mapping) -> str | None:
    """Why this process does not provide a job's environment, or None if it does.

    The BLAS kernel is set in the process's environment before NumPy loads; a name
    the library does not know leaves it on its own choice, which this check finds.
    The compute backend must be one this process provides (:data:`BACKENDS`).

    """
    backend = str(environment.get("backend", "cpu"))
    if backend not in BACKENDS:
        return f"This host provides the backends {list(BACKENDS)}, not the job's {backend}."
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
    now)."""
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
    )


def _beats(claim: queue.Claim, stop: threading.Event) -> None:
    """Renew the lease every BEAT until stopped; a lost claim ends the process. A
    beat that has begun ends before the thread does, and :func:`run` waits for
    the thread before it marks the task."""
    pid, born, number = os.getpid(), service.birth(os.getpid()), 1
    while not stop.wait(results.BEAT):
        number += 1
        try:
            queue.beat(claim, number, pid, born)
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
    if why is not None:
        service.say(why)
        try:
            queue.refuse(claim, why)
        except queue.Revoked as error:
            service.say(str(error))
            return REVOKED
        return service.REFUSED
    threads = int(record["environment"]["threads"])
    cv2.setNumThreads(threads)
    threadpool_limits(threads)
    try:
        queue.beat(claim, 1, os.getpid(), service.birth(os.getpid()))
    except queue.Revoked as error:
        service.say(str(error))
        return REVOKED
    stop = threading.Event()
    beats = threading.Thread(target=_beats, args=(claim, stop), daemon=True)
    beats.start()
    task = queue.task(job, task_name)
    failure, adopted = None, ()
    try:
        ctx = context(job, task, token, code)
        storage = ctx.tables.get("storage", {})
        use_storage(settings_from_table(StorageSettings, storage))
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
    serving.add_argument("--lock", type=Path)
    serving.add_argument("--logs", type=Path)
    serving.add_argument("--poll", type=float, default=service.POLL)
    serving.add_argument("--until-settled", action="store_true")
    running = commands.add_parser("run", help="Run one claimed task.")
    running.add_argument("job", type=Path)
    running.add_argument("task")
    running.add_argument("token")
    running.add_argument("--machine", default=socket.gethostname())
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.job, args.task, args.token, args.machine)
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
