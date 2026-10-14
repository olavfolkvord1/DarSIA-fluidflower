"""Enqueue explicit whole-run task dictionaries in an existing UNC job."""
import argparse
import json
from pathlib import Path

from .job_queue import JobQueue, read_json, require_unc


def main(argv=None):
    """Enqueue explicit whole-run task metadata with the job's code and calibration pins.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True,
                        help="UNC JSON containing a task list or {'tasks': [...]}.")
    args = parser.parse_args(argv)
    require_unc(str(args.job))
    require_unc(str(args.tasks))
    job = read_json(args.job / "job.json")
    document = read_json(args.tasks)
    tasks = document if isinstance(document, list) else document["tasks"]
    shared = JobQueue(args.job)
    for original in tasks:
        task = dict(original)
        for key in ("commit", "calibration_sha256", "precision"):
            task.setdefault(key, job[key])
            if task[key] != job[key]:
                raise ValueError(f"Task {key} differs from job.json")
        path = shared.enqueue(task)
        print(json.dumps(dict(run=task["run"], queued=str(path))), flush=True)


if __name__ == "__main__":
    main()
