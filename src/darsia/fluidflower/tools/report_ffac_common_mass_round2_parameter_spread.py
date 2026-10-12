"""Read-only parameter and fit-coverage audit of all saved R2 LORO folds.

This companion never imports a scientific executor, evaluates predictions,
fits parameters or changes the pre-registered selection. A complete cohort is
required so a partial run cannot look like a final parameter-spread report.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid


CONTRACT = "ffac-common-mass-round2-parameter-spread-v1"
PARAMETERS = {"N1": ("gain", "beta"), "N2": ("gain", "beta"),
    "N3": ("gain", "blend", "beta"), "J": ("endpoint", "amplitude", "gamma", "beta")}
CAVEAT = ("Descriptive audit of saved calibration-only leave-one-run-out fits. "
    "No refitting, prediction evaluation or selection change. Conditional on "
    "the locked phase maps, fixed normalisation and frozen pooled colour paths; "
    "parameter spread is not an independent end-to-end validation.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha_file(path), size_bytes=path.stat().st_size)


def canonical_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def resolve_descriptor(record, base):
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path(base) / path
    if sha_file(path) != record["sha256"]:
        raise ValueError(f"Source changed: {path}")
    if "size_bytes" in record and path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Source size changed: {path}")
    return path


def quantile(values, probability):
    """Linear interpolation between sorted samples, including singletons."""
    ordered = sorted(float(value) for value in values)
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise ValueError("Finite non-empty saved parameter values required")
    position = (len(ordered) - 1) * probability
    lo = int(math.floor(position)); hi = int(math.ceil(position))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def spread(values):
    values = list(values)
    if not values:
        return dict(folds=0, min=None, q05=None, q25=None, median=None, q75=None,
            q95=None, max=None, sample_standard_deviation=None)
    return dict(folds=len(values), min=min(values), q05=quantile(values, .05),
        q25=quantile(values, .25), median=quantile(values, .5),
        q75=quantile(values, .75), q95=quantile(values, .95), max=max(values),
        sample_standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.)


def inventory(job):
    job = Path(job)
    campaign_path = job / "campaign_binding.json"
    campaign = read_json(campaign_path)
    if campaign.get("contract") != "ffac-common-mass-round2-campaign-v1":
        raise ValueError("Expected the immutable round-2 campaign")
    if campaign.get("selection_reads_heldout") is not False or campaign.get("candidate_gas_fixed") is not True:
        raise ValueError("The saved campaign does not preserve fixed gas / held-out separation")
    candidates = campaign["candidates"]
    if len(candidates) != len(set(candidates)) or set(candidates) != set(PARAMETERS):
        raise ValueError("All four pre-registered supported candidates are required")
    plan_path = resolve_descriptor(campaign["input_plan"], job)
    plan = read_json(plan_path)
    runs = sorted(plan["split"]["fit_runs"])
    if len(runs) != 26 or len(set(runs)) != 26:
        raise ValueError("The source-bound 26-run calibration cohort is required")
    held = set(plan["split"]["held_out_runs"])
    descriptive = set(plan["split"]["descriptive_runs"])
    if set(runs) & (held | descriptive):
        raise ValueError("Held-out/descriptive runs leaked into the calibration cohort")
    fingerprint = canonical_digest(campaign)
    rows = []; sources = []; missing = []
    for candidate in candidates:
        for leftout in runs:
            path = job / "work/loro" / candidate / leftout / "receipt.json"
            if not path.is_file():
                missing.append(f"{candidate}/{leftout}"); continue
            receipt = read_json(path)
            if receipt.get("complete") is not True or receipt.get("campaign_binding_sha256") != fingerprint:
                raise ValueError(f"Incomplete or foreign LORO receipt: {path}")
            if receipt.get("candidate") != candidate or receipt.get("leftout_run") != leftout:
                raise ValueError(f"LORO path and identity disagree: {path}")
            if receipt.get("stage") != "loro" or receipt.get("selection_uses_heldout") is not False or receipt.get("colour_paths_unchanged") is not True:
                raise ValueError(f"Unexpected LORO scope: {path}")
            fit = receipt["fit"]
            if set(fit["calibration_runs"]) != set(runs) - {leftout} or len(fit["calibration_runs"]) != 25:
                raise ValueError(f"The fold did not fit exactly the other 25 runs: {path}")
            if fit.get("heldout_fit_run") != leftout or fit.get("candidate") != candidate:
                raise ValueError(f"Fit identity differs from its LORO receipt: {path}")
            for flag in ("gas_branch_fitted", "overflow_in_objective", "final_heldout_used", "native_paths_refitted", "map_mass_rescaling"):
                if fit.get(flag) is not False:
                    raise ValueError(f"Forbidden fit scope ({flag}): {path}")
            params = fit["parameters"]
            if params.get("candidate") != candidate:
                raise ValueError(f"Parameter identity differs: {path}")
            values = {name: float(params[name]) for name in PARAMETERS[candidate]}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite saved parameters: {path}")
            counts = {}
            for stage in ("stage1", "stage2"):
                scope = fit["stages"][stage]
                stage_runs = scope["runs"]
                if len(stage_runs) != len(set(stage_runs)) or not set(stage_runs) <= set(runs) - {leftout}:
                    raise ValueError(f"Invalid stage run scope: {path}")
                count = int(scope["frames"])
                if count < 0 or (count == 0) != (len(stage_runs) == 0):
                    raise ValueError(f"Invalid stage frame/run counts: {path}")
                counts[stage + "_frame_count"] = count
                counts[stage + "_run_count"] = len(stage_runs)
                counts[stage + "_runs"] = sorted(stage_runs)
            beta_fitted = fit.get("beta_fitted")
            if not isinstance(beta_fitted, bool) or (counts["stage2_frame_count"] == 0 and beta_fitted):
                raise ValueError(f"Invalid beta fit coverage: {path}")
            row = dict(candidate=candidate, leftout_run=leftout, calibration_run_count=25,
                **values, **counts, beta_fitted=beta_fitted,
                stage2_optimizer_assessable=fit["optimizer_audit"].get("stage2", {}).get("assessable"),
                stage2_optimizer_reason=fit["optimizer_audit"].get("stage2", {}).get("reason"),
                optimizer_audit=fit["optimizer_audit"], wall_seconds=receipt.get("wall_seconds"),
                receipt_sha256=sha_file(path))
            rows.append(row); sources.append(descriptor(path))
    if missing:
        raise ValueError(f"All 104 saved LORO folds are required; {len(missing)} missing: " + ", ".join(missing))
    if len(rows) != 104:
        raise ValueError("Expected exactly 104 saved folds")
    return campaign, plan, rows, sources


def summarize(rows):
    parameter_rows = []; coverage = []
    for candidate, names in PARAMETERS.items():
        selected = [row for row in rows if row["candidate"] == candidate]
        late = [row for row in selected if row["beta_fitted"]]
        for name in names:
            parameter_rows.append(dict(candidate=candidate, parameter=name,
                population="all_folds", **spread(row[name] for row in selected)))
            if name == "beta":
                parameter_rows.append(dict(candidate=candidate, parameter=name,
                    population="stage2_fitted_folds_only", **spread(row[name] for row in late)))
        coverage.append(dict(candidate=candidate, folds=len(selected),
            beta_fitted_folds=len(late), beta_not_fitted_folds=len(selected) - len(late),
            zero_stage2_frames_folds=sum(row["stage2_frame_count"] == 0 for row in selected),
            stage1_frame_count=spread(row["stage1_frame_count"] for row in selected),
            stage1_run_count=spread(row["stage1_run_count"] for row in selected),
            stage2_frame_count=spread(row["stage2_frame_count"] for row in selected),
            stage2_run_count=spread(row["stage2_run_count"] for row in selected)))
    return parameter_rows, coverage


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list)) else value for key, value in row.items()})


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write("\n")


def build(job, destination):
    job = Path(job); destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Write only to a new diagnostic folder: {destination}")
    campaign, plan, rows, sources = inventory(job)
    parameter_rows, coverage = summarize(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    temporary.mkdir()
    report = dict(contract=CONTRACT, created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True, saved_fold_count=104, fit_runs=sorted(plan["split"]["fit_runs"]),
        candidates=list(campaign["candidates"]), quantile_method="linear interpolation at (n-1)*p",
        parameter_spread=parameter_rows, stage_coverage=coverage, conditional_validation_scope=CAVEAT,
        metadata_only=True, fitting_performed=False, predictions_evaluated=False,
        held_out_scientific_pixels_read=False, selection_changed=False,
        main_maps_changed=False, candidate_registered=False)
    write_json(temporary / "report.json", report)
    write_csv(temporary / "parameter_spread.csv", parameter_rows)
    write_csv(temporary / "per_fold_parameters_and_coverage.csv", rows)
    with (temporary / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("Parameter spread and fit coverage from all 104 saved round-2 LORO folds.\n\n" + CAVEAT + "\n\n")
        stream.write("The beta distribution is also reported only for folds with a fitted late stage. A beta of zero in a fold without eligible late frames is a default, not a late-time estimate.\n\n")
        for row in coverage:
            stream.write(f"- {row['candidate']}: {row['beta_fitted_folds']}/26 folds fit beta; "
                f"{row['zero_stage2_frames_folds']}/26 have no stage-2 frames.\n")
        stream.write("\nNo scientific executor was imported. Parameters and coverage were read from immutable saved fit receipts.\n")
    assets = {path.name: descriptor(path) for path in temporary.iterdir() if path.is_file()}
    for name, asset in assets.items():
        asset["path"] = name
    receipt = dict(contract=CONTRACT, complete=True, metadata_only=True,
        campaign=descriptor(job / "campaign_binding.json"),
        campaign_fingerprint=canonical_digest(campaign), input_plan=campaign["input_plan"],
        companion_source=descriptor(__file__), saved_loro_receipts=sources, assets=assets,
        saved_fold_count=104, fitting_performed=False, held_out_scientific_pixels_read=False,
        predictions_evaluated=False, selection_changed=False)
    write_json(temporary / "receipt.json", receipt)
    temporary.rename(destination)
    return dict(destination=str(destination), receipt=descriptor(destination / "receipt.json"),
        stage_coverage=coverage, parameter_spread=parameter_rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.job, args.destination), allow_nan=False))


if __name__ == "__main__":
    main()
