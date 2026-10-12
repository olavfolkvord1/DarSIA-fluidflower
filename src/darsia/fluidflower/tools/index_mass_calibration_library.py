"""Index actual saved mass fits/frozen snapshots without moving any originals."""
import argparse
import json
from pathlib import Path

from ..mass_calibration_library import migrate_library


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", required=True, type=Path)
    parser.add_argument("--scan-root", action="append", type=Path,
                        help="Optional explicit history root; default is series/results/concentration_mass")
    parser.add_argument("--inventory-only", action="store_true", help="Audit existing library references without rescanning saved history")
    parser.add_argument("--report", type=Path, help="Write a new inventory receipt; never overwrite one")
    args = parser.parse_args(argv)
    if args.inventory_only and args.scan_root:
        parser.error("--inventory-only cannot be combined with --scan-root")
    report = migrate_library(args.series_root, roots=[] if args.inventory_only else args.scan_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
