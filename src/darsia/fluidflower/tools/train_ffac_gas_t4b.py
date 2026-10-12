"""Prove the unchanged T4 dataset and fit one CPU G3 gas head with lamp shift."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..phase_gas_t4b_training import fit_pixel_g3_t4b, preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "t4-dataset", "spec", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("Bounded CPU thread count is invalid")
    if args.preflight_only:
        preflight(args.experiment, args.t4_dataset, args.output, spec=args.spec)
    else:
        fit_pixel_g3_t4b(args.experiment, args.t4_dataset, args.output, spec=args.spec, threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
