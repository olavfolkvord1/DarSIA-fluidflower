"""Preflight/prepare exact G3 plus L-cap data and fit two CPU light-epoch heads."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..phase_gas_t4_training import fit_pixel_g3_t4, preflight, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "old-extraction", "base-candidate", "lcap-snapshot", "light-epochs", "spec", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epoch", choices=("rig_light", "new_lamps"), help="Fit just one epoch; separate epoch directories allow concurrent processes")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--extract-only", action="store_true")
    modes.add_argument("--fit-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.extraction_workers <= 4 or not 1 <= args.threads <= 32:
        parser.error("Bounded CPU worker/thread count is invalid")
    dataset = args.output / "dataset"
    if args.preflight_only:
        preflight(args.experiment, args.old_extraction, args.base_candidate, args.lcap_snapshot,
            args.light_epochs, dataset, spec=args.spec)
        return 0
    if not (dataset / "dataset_receipt.json").is_file():
        if args.fit_only:
            parser.error("The exact T4 dataset must be committed before --fit-only")
        prepare_dataset(args.experiment, args.old_extraction, args.base_candidate, args.lcap_snapshot,
            args.light_epochs, dataset, spec=args.spec, workers=args.extraction_workers)
    if not args.extract_only:
        fit_pixel_g3_t4(args.experiment, dataset, args.output / "pixel_fit", threads=args.threads, epoch=args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
