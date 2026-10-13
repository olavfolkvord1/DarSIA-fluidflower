"""Read genuine immutable terminal records; never infer completion from outputs."""
import argparse
import json
import sys
from pathlib import Path

# Use this checkout even when the interpreter has another editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from darsia.fluidflower.job_queue import JobQueue, require_unc, write_immutable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Fresh UNC aggregate JSON; existing files are never replaced.")
    args = parser.parse_args(argv)
    require_unc(str(args.job))
    value = JobQueue(args.job).aggregate()
    if args.output is not None:
        require_unc(str(args.output))
        write_immutable(args.output, value)
    print(json.dumps({key: value[key] for key in (
        "complete", "task_count", "completed_count", "failed_count"
    )}), flush=True)


if __name__ == "__main__":
    main()
