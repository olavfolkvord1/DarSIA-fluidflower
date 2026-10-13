"""Enqueue explicit whole-run task dictionaries in an existing UNC job."""
import argparse
import json
import sys
from pathlib import Path

# Use this checkout even when the interpreter has another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darsia.fluidflower.job_queue import JobQueue, read_json, require_unc


def main(argv=None):
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
