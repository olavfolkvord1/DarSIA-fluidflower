"""Immutable command-line stages for one common native FF_AC mass response.

This worker never discovers arbitrary experiment images.  Every input is an
explicit SHA-bound manifest and every output is a new folder.  The separate
source loader owns RAW/facies/reference registration.  No held-out frame is
read by the calibration stages, and no inference or phase training occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darsia.fluidflower import phase_ffac_common_mass as common


def _read_bound(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and common.sha256(path) != expected_sha256:
        raise ValueError(f"Source manifest SHA-256 mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_paths(request_path, output):
    request = _read_bound(request_path)
    common.validate_cohort(request["calibration_runs"], request["heldout_runs"])
    if request.get("representation") != "corrected linear current/clearwater minus registered baseline/clearwater":
        raise ValueError("Path samples do not use the frozen measured-reference representation")
    batches = request["sample_batches"]
    # Check run identities before reading any arrays.
    if any(common.normalize_run(row["run"]) not in common.CALIBRATION_RUNS for row in batches):
        raise ValueError("Held-out or excluded samples entered the common path request")
    grouped = {label: {} for label in common.SUPPORTED_FACIES}
    source_bindings = []
    for row in batches:
        run, label = common.normalize_run(row["run"]), int(row["facies"])
        if label not in grouped:
            continue
        path = Path(row["path"])
        if common.sha256(path) != row["sha256"]:
            raise ValueError(f"Pooled optical sample SHA-256 mismatch: {path}")
        values = np.load(path, allow_pickle=False)
        grouped[label].setdefault(run, []).append(values)
        source_bindings.append(dict(row))
    grouped = {label: {run: np.concatenate(parts) for run, parts in runs.items()} for label, runs in grouped.items()}
    paths, audit = common.fit_pooled_color_paths(grouped,
        min_samples=int(request.get("min_samples_per_run", 16)), min_runs=int(request.get("min_runs", 2)))
    metadata = common.save_common_path_package(output, paths, path_audit=audit,
        source_bindings=dict(request_path=str(Path(request_path).resolve()), request_sha256=common.sha256(request_path), batches=source_bindings),
        flash_bounds=request.get("flash_bounds"))
    return dict(path=str(Path(output).resolve()), supported_facies=sorted(paths), metadata=metadata)


def fit_and_freeze(request_path, path_folder, output):
    request = _read_bound(request_path)
    common.validate_cohort(request["calibration_runs"], request["heldout_runs"])
    references = request["summary_files"]
    summaries = []
    extra = {str(Path(request_path).resolve()): common.sha256(request_path)}
    for row in references:
        summary = _read_bound(row["path"], row["sha256"])
        if common.normalize_run(summary["run"]) not in common.CALIBRATION_RUNS:
            raise ValueError("A held-out frame entered the mass-fit summary inventory")
        summaries.append(summary)
        extra[str(Path(row["path"]).resolve())] = row["sha256"]
    fit = common.fit_shared_mass_scales(summaries, unknown_policy=request.get("unknown_policy", "bounded_interval"))
    freeze = common.freeze_calibration(output, path_folder=path_folder, fit_result=fit,
        spec_sha256=request["spec_sha256"], input_plan_sha256=request["input_plan_sha256"], extra_files=extra)
    return dict(frozen_calibration=str(freeze.resolve()), sha256=common.sha256(freeze), parameters=fit["parameters"],
        jacobian_rank=fit["jacobian_rank"], parameters_at_bounds=fit["parameters_at_bounds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    paths = commands.add_parser("paths")
    paths.add_argument("--request", type=Path, required=True)
    paths.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit-freeze")
    fit.add_argument("--request", type=Path, required=True)
    fit.add_argument("--path-folder", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-freeze")
    verify.add_argument("--freeze", type=Path, required=True)
    verify.add_argument("--sha256")
    args = parser.parse_args()
    if args.stage == "paths":
        report = build_paths(args.request, args.output)
    elif args.stage == "fit-freeze":
        report = fit_and_freeze(args.request, args.path_folder, args.output)
    else:
        payload, digest = common.verify_frozen_calibration(args.freeze, expected_sha256=args.sha256)
        report = dict(frozen=payload["frozen"], sha256=digest, parameters=payload["parameters"], calibration_runs=payload["calibration_runs"], heldout_runs=payload["heldout_runs"])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
