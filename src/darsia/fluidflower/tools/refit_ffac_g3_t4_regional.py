"""Source-exact CPU regional T4 extraction, independent epoch fit and assembly."""
from __future__ import annotations

import argparse
from pathlib import Path

from darsia.fluidflower import phase_gas_t4_training as training
from darsia.fluidflower import phase_gas_t4_regional_refit as regional


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--old-g3-candidate", type=Path, required=True)
    parser.add_argument("--round1-snapshot", type=Path, required=True)
    parser.add_argument("--lcap-snapshot", type=Path, required=True)
    parser.add_argument("--light-epochs", type=Path, required=True)
    parser.add_argument("--pixel-fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--light-epoch", help="Extract/fit only this epoch declared by the project table")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fit-threads", type=int, default=2)
    parser.add_argument("--memory-per-worker-mb", type=float, default=6144)
    parser.add_argument("--memory-headroom-mb", type=float, default=2048)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--extract-only", action="store_true")
    modes.add_argument("--fit-only", action="store_true")
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--epoch-result", action="append", default=[], metavar="EPOCH=DIR",
                        help="Completed or SHA-exact transported epoch result directory for assembly")
    args = parser.parse_args(argv)
    from darsia.fluidflower.tools.refit_ffac_g3_downstream import below_normal_priority
    below_normal_priority()
    epochs = training.load_light_epochs(args.light_epochs)
    if args.light_epoch is not None and args.light_epoch not in epochs["epochs"]:
        raise ValueError("Requested regional T4 epoch is absent from the series project table")
    if args.epoch_result and not args.assemble_only:
        raise ValueError("Explicit epoch result directories are used only for bundle assembly")
    if args.assemble_only and args.light_epoch is not None:
        raise ValueError("Bundle assembly requires the two independent project epochs")
    pixels = training.read(args.pixel_fit_receipt)
    selected = [args.light_epoch] if args.light_epoch else epochs["epochs"]
    if (pixels.get("cpu_only") is not True or pixels.get("converged") is not True
            or any(epoch not in pixels["epochs"] for epoch in selected)):
        raise ValueError("Regional T4 requires completed converged CPU pixel receipts")
    bundle = training.load_combined_training_sources(args.old_g3_candidate, args.round1_snapshot,
        args.base_candidate, args.lcap_snapshot, epochs)
    regional.inherited._write(args.output / "minimum_data_gate.json", regional.minimum_data_gate(bundle))
    if args.assemble_only:
        roots = {}
        for value in args.epoch_result:
            epoch, separator, raw_path = value.partition("=")
            if not separator or epoch in roots or epoch not in epochs["epochs"] or not raw_path:
                raise ValueError("Bundle assembly requires distinct declared EPOCH=DIR arguments")
            roots[epoch] = Path(raw_path)
        regional.assemble_epoch_bundle(args.base_candidate, args.pixel_fit_receipt, bundle, args.output,
                                      epoch_results=roots or None)
    elif args.extract_only or args.plan_only:
        for epoch_id in selected:
            pixel = pixels["epochs"][epoch_id]
            path = training.checked(pixel["model_path"], pixel["model_sha256"])
            regional.extract_regional_training(args.base_candidate, path, bundle,
                args.output / "epochs" / epoch_id / "extraction", pixel_model_sha256=pixel["model_sha256"],
                legacy_context_candidate=args.old_g3_candidate, light_epoch_id=epoch_id,
                workers=args.workers, threads=args.threads, plan_only=args.plan_only,
                memory_per_worker_mb=args.memory_per_worker_mb, memory_headroom_mb=args.memory_headroom_mb)
    else:
        if args.fit_only and any(not (args.output / "epochs" / epoch / "extraction/extraction_receipt.json").is_file()
                                 for epoch in selected):
            raise ValueError("Fit-only requires complete exact extraction receipts for the selected epochs")
        if args.light_epoch is None:
            regional.fit_regional_g3_t4(args.base_candidate, args.pixel_fit_receipt, bundle, args.output,
                legacy_context_candidate=args.old_g3_candidate, workers=args.workers, threads=args.threads,
                fit_threads=args.fit_threads, memory_per_worker_mb=args.memory_per_worker_mb,
                memory_headroom_mb=args.memory_headroom_mb)
        else:
            pixel = pixels["epochs"][args.light_epoch]
            path = training.checked(pixel["model_path"], pixel["model_sha256"])
            regional.fit_regional_g3_t4_epoch(args.base_candidate, path, bundle,
                args.output / "epochs" / args.light_epoch, pixel_model_sha256=pixel["model_sha256"],
                legacy_context_candidate=args.old_g3_candidate, light_epoch_id=args.light_epoch,
                workers=args.workers, threads=args.threads, fit_threads=args.fit_threads,
                memory_per_worker_mb=args.memory_per_worker_mb, memory_headroom_mb=args.memory_headroom_mb)
    print("Regional T4 stage complete: " + str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
