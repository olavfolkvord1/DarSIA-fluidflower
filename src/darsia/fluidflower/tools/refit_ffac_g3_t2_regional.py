"""Source-exact regional G3-T2 gas fit after the sealed pixel stage completes."""
from __future__ import annotations

import argparse
from pathlib import Path

from darsia.fluidflower.phase_gas_t2_training import load_combined_training_sources, read, checked
from darsia.fluidflower.phase_gas_t2_regional_refit import fit_regional_g3_t2
from darsia.fluidflower.project import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--old-g3-candidate", type=Path, required=True)
    parser.add_argument("--round1-snapshot", type=Path, required=True)
    parser.add_argument("--t2-snapshot", type=Path, required=True)
    parser.add_argument("--pixel-fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fit-threads", type=int, default=6)
    args = parser.parse_args(argv)
    pixel = read(args.pixel_fit_receipt)
    if pixel.get("converged") is not True or pixel.get("cpu_only") is not True:
        raise ValueError("Regional stage requires the completed converged CPU pixel receipt")
    path = checked(pixel["model_path"], pixel["model_sha256"])
    bundle = load_combined_training_sources(args.old_g3_candidate, args.round1_snapshot,
                                           args.base_candidate, args.t2_snapshot)
    if bundle["receipt"]["combined_label_sha256"] != pixel["training_label_sha256"]:
        raise ValueError("Regional source bundle differs from the sealed pixel-fit inventory")
    result = fit_regional_g3_t2(args.base_candidate, path, bundle, args.output,
        pixel_model_sha256=pixel["model_sha256"], legacy_context_candidate=args.old_g3_candidate,
        workers=args.workers, threads=args.threads, fit_threads=args.fit_threads)
    print("Regional G3-T2 gas-only fit complete: " + result["regional_model_sha256"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
