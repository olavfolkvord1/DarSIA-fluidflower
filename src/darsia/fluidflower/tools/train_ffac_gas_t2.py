"""Prepare sealed G3+T2 samples and fit one detached CPU gas head."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..phase_gas_t2_training import prepare_dataset, fit_pixel_g3_t2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--old-extraction", type=Path, required=True)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--t2-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.extraction_workers <= 24 or not 1 <= args.threads <= 32:
        parser.error("Bounded CPU worker/thread count is invalid")
    dataset = args.output / "dataset"
    if not (dataset / "dataset_receipt.json").is_file():
        prepare_dataset(args.experiment, args.old_extraction, args.base_candidate,
                        args.t2_snapshot, dataset, workers=args.extraction_workers)
    if not args.extract_only:
        if (args.output / "pixel_fit/fit_receipt.json").exists():
            parser.error("This isolated pixel fit is already completed; do not refit it")
        fit_pixel_g3_t2(args.experiment, dataset, args.output / "pixel_fit", threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
