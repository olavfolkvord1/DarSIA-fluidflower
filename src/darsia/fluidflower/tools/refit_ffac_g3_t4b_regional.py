"""CPU-only joint regional T4b extraction and fit at the frozen G3 settings."""
from __future__ import annotations

import argparse
from pathlib import Path

from darsia.fluidflower import phase_gas_t4_training as training
from darsia.fluidflower import phase_gas_t4b_regional_refit as regional


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-candidate", "old-g3-candidate", "round1-snapshot", "lcap-snapshot",
                 "light-epochs", "pixel-fit-receipt", "base-regional-refit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-context-root", type=Path, action="append", default=[],
        help="T4 extraction directory; reuse only an exactly matching compact source context")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fit-threads", type=int, default=2)
    parser.add_argument("--memory-per-worker-mb", type=float, default=6144)
    parser.add_argument("--memory-headroom-mb", type=float, default=2048)
    modes = parser.add_mutually_exclusive_group()
    for name in ("plan-only", "extract-only", "fit-only", "assemble-only"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args(argv)
    from darsia.fluidflower.tools.refit_ffac_g3_downstream import below_normal_priority
    below_normal_priority()
    table = training.load_light_epochs(args.light_epochs)
    pixel = training.read(args.pixel_fit_receipt)
    if pixel.get("cpu_only") is not True or pixel.get("converged") is not True:
        raise ValueError("T4b regional fit requires a converged CPU joint pixel receipt")
    bundle = training.load_combined_training_sources(args.old_g3_candidate, args.round1_snapshot,
        args.base_candidate, args.lcap_snapshot, table)
    kwargs = dict(pixel_model_sha256=pixel["model_sha256"], legacy_context_candidate=args.old_g3_candidate,
        base_regional_refit=args.base_regional_refit, source_context_roots=args.source_context_root,
        workers=args.workers, threads=args.threads, memory_per_worker_mb=args.memory_per_worker_mb,
        memory_headroom_mb=args.memory_headroom_mb)
    path = training.checked(pixel["model_path"], pixel["model_sha256"])
    if args.assemble_only:
        regional.assemble_joint_bundle(args.base_candidate, args.pixel_fit_receipt, bundle, args.output)
    elif args.plan_only or args.extract_only:
        regional.extract_regional_training(args.base_candidate, path, bundle, args.output / "extraction",
            plan_only=args.plan_only, **kwargs)
    else:
        if args.fit_only and not (args.output / "extraction/extraction_receipt.json").is_file():
            raise ValueError("T4b fit-only requires completed exact joint extraction")
        regional.fit_regional_g3_t4b(args.base_candidate, path, bundle, args.output,
            fit_threads=args.fit_threads, **kwargs)
        regional.assemble_joint_bundle(args.base_candidate, args.pixel_fit_receipt, bundle, args.output)
    print("Joint regional T4b stage complete: " + str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
