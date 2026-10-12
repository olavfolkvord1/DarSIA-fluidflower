"""Prepare the sealed complete T2 + B-T2 dataset and fit one CPU gas head."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..phase_gas_t3_training import fit_pixel_g3_t3, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--t2-dataset", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--bt2-snapshot", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args(argv)
    if args.extraction_workers < 1 or not 1 <= args.threads <= 32:
        parser.error("Bounded CPU worker/thread count is invalid")
    if args.extract_only and args.fit_only:
        parser.error("Choose --extract-only or --fit-only, not both")
    dataset = args.output / "dataset"
    if not (dataset / "dataset_receipt.json").is_file():
        if args.fit_only:
            parser.error("The sealed dataset must be committed before --fit-only")
        prepare_dataset(args.experiment, args.t2_dataset, args.base_candidate, args.bt2_snapshot,
                        dataset, spec=args.spec, workers=args.extraction_workers)
    if not args.extract_only:
        if (args.output / "pixel_fit/fit_receipt.json").exists():
            parser.error("This isolated pixel fit is already completed; do not refit it")
        fit_pixel_g3_t3(args.experiment, dataset, args.output / "pixel_fit", threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
