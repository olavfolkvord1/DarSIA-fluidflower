"""Index existing phase-model versions without moving or changing their assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from darsia.fluidflower.phase_model_library import index_series


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = index_series(args.series_root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
