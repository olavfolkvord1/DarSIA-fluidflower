"""Calibration-only exact replay before a round-2 common calibration is frozen.

Every cached calibration histogram receives an analytic arithmetic bound. One
frame per fit run, plus the largest global and late bound, is replayed on the
original full-resolution float32 sources. This checks a computational shortcut;
it never selects a model or reads any final-held-out scientific pixels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from darsia.fluidflower import phase_ffac_common_mass_round2 as engine
from darsia.fluidflower import phase_ffac_common_mass_round2_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_inputs as assets
from darsia.fluidflower import phase_ffac_common_mass_round2_production as output
from darsia.fluidflower.tools import run_ffac_common_mass_round2 as campaign

CONTRACT = "ffac-common-mass-round2-exact-audit-v1"
MAX_PRECISION_FRACTION = .01
_SESSION = {}


def precision_record(summary, parameters, anchor):
    if summary.get("contract") != engine.CONTRACT or summary["run"] not in engine.FIT_RUNS:
        raise ValueError("Exact audit admits only source-bound calibration26 summaries")
    predicted = engine.evaluate_round2_summary(summary, parameters, anchor)
    bound = float(predicted["summary_quantization_bound_g"])
    dose = summary.get("injected_mass_g", summary.get("nominal_injected_mass_g"))
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("Finite nonnegative summary error bound required")
    return dict(run=summary["run"], frame_index=summary["frame_index"], elapsed_h=summary["elapsed_h"],
        injected_mass_g=dose, summary_main_mass_g=predicted["total_mass_g"],
        summary_quantization_bound_g=bound, bound_over_dose=bound/dose if dose is not None and dose > 0 else None,
        summary_gas_mass_g=predicted["gas_mass_g"], summary_gas_water_mass_g=predicted["gas_water_mass_g"])


def select_exact_rows(records):
    groups = {run: [] for run in engine.FIT_RUNS}
    seen = set()
    for row in records:
        key = row["run"], row["frame_index"]
        if row["run"] not in groups or key in seen:
            raise ValueError("Duplicated or out-of-fit exact audit source")
        seen.add(key); groups[row["run"]].append(row)
    if any(not rows for rows in groups.values()):
        raise ValueError("Every one of the 26 fit runs must be represented")
    selected = {}
    def add(row, reason):
        key = row["run"], row["frame_index"]
        selected.setdefault(key, dict(row, selection_reasons=[]))["selection_reasons"].append(reason)
    for values in groups.values():
        add(min(values, key=lambda row: (abs(row["elapsed_h"]-1), row["frame_index"])), "one_full_resolution_frame_per_fit_run_nearest_1h")
    positive = [row for row in records if row["bound_over_dose"] is not None]
    if positive:
        add(max(positive, key=lambda row: row["bound_over_dose"]), "global_largest_summary_bound_over_dose")
    late = [row for row in positive if row["elapsed_h"] >= 4]
    if late:
        add(max(late, key=lambda row: row["bound_over_dose"]), "late_largest_summary_bound_over_dose")
    return sorted(selected.values(), key=lambda row: (row["run"], row["frame_index"]))


def arithmetic_allowance(summary):
    aqueous = max(sum(summary["native_aqueous_capacity_g"]), sum(summary["joint_aqueous_capacity_g"]))
    capacity = aqueous + summary["gas_capacity_sg1_g"] + summary["gas_water_capacity_sg0_g"]
    return float(32*np.finfo(np.float32).eps*capacity + 32*np.finfo(np.float64).eps*max(1, capacity))


def compare_exact(exact, summary, parameters, anchor):
    predicted = engine.evaluate_round2_summary(summary, parameters, anchor)
    bound = predicted["summary_quantization_bound_g"]
    allowance = arithmetic_allowance(summary)
    error = abs(exact["total_mass_g"]-predicted["total_mass_g"])
    gas_error = abs(exact["gas_mass_g"]-predicted["gas_mass_g"])
    water_error = abs(exact["gas_water_mass_g"]-predicted["gas_water_mass_g"])
    return dict(exact_main_mass_g=exact["total_mass_g"], summary_main_mass_g=predicted["total_mass_g"],
        absolute_difference_g=error, analytic_error_bound_g=bound, arithmetic_allowance_g=allowance,
        within_summary_precision_bound=bool(error <= bound+allowance),
        exact_gas_mass_g=exact["gas_mass_g"], exact_gas_water_mass_g=exact["gas_water_mass_g"],
        gas_difference_g=gas_error, gas_water_difference_g=water_error,
        fixed_gas_matches_physical_summary=bool(max(gas_error, water_error) <= allowance))


def _init(context):
    campaign.init_worker(context["input_plan"]["path"], context["round1_freeze"]["path"], "cpu")
    _SESSION.update(context=context, rows={(row["run"], row["frame_index"]): row for row in context["summaries"]},
        index={(row["run"], row["frame_index"]): row for row in campaign._SESSION["index"]["frames"]})


def audit_run(task):
    run, selected = task
    context = _SESSION["context"]
    records = []
    for selection in selected:
        tick = time.perf_counter()
        key = run, selection["frame_index"]
        summary, metadata = _SESSION["rows"][key], _SESSION["index"][key]
        if (summary["elapsed_h"] != metadata["elapsed_h"] or
                summary["source_binding"]["source_receipt"] != metadata["receipt"]):
            raise ValueError("Cached summary/source/camera-time binding differs")
        frame, response = campaign.prepare(metadata)
        exact = engine.evaluate_round2_frame(response, context["parameters"], context["anchors_by_run"][run], fields=False)
        comparison = compare_exact(exact, summary, context["parameters"], context["anchors_by_run"][run])
        invariant = None
        if "global_largest_summary_bound_over_dose" in selection["selection_reasons"]:
            changed = dict(context["parameters"], beta=context["parameters"]["beta"]+1)
            if changed["candidate"] == "J":
                changed["amplitude"] = max(.001, changed["amplitude"]*.5)
                changed["endpoint"] *= 1.5; changed["gamma"] *= 1.5
            else:
                changed["gain"] *= 1.5
                if changed["candidate"] == "N3": changed["blend"] = 1-changed["blend"]
            altered = engine.evaluate_round2_frame(response, changed, context["anchors_by_run"][run], fields=False)
            invariant = {name: exact[name] == altered[name] for name in ("gas_mass_g", "gas_water_mass_g")}
            if not all(invariant.values()): raise ValueError("Fitted aqueous parameters changed fixed gas or residual-water mass")
            del altered
        records.append(dict(selection, **comparison, gas_parameter_invariance=invariant,
            scientific_input_dtype=str(frame.current_rgb.dtype), native_shape_hw=list(frame.phase_labels.shape),
            full_resolution=True, phase_labels_modified=False, bottom_mass_allowed=False,
            source_receipt=metadata["receipt"], wall_seconds=time.perf_counter()-tick))
        del frame, response, exact; gc.collect()
    return records


def run(args):
    job = args.job
    binding_path = job / "campaign_binding.json"
    binding = output.read_json(binding_path)
    fit_path = job / "calibration/shared_fit.json"
    fit = output.read_json(fit_path)
    bound_sha = output.digest(binding)
    if fit.get("campaign_binding_sha256") != bound_sha:
        raise ValueError("Fit is not bound to the R2 extraction campaign")
    if set(fit["calibration_runs"]) != set(engine.FIT_RUNS) or fit.get("heldout_fit_run") is not None:
        raise ValueError("Exact replay requires the selected all26 common fit")
    for name, expected in binding["code_sha256"].items():
        module = importlib.import_module(name)
        if assets.sha256(module.__file__) != expected: raise ValueError(f"Bound code changed: {name}")
    if assets.sha256(campaign.__file__) != binding["driver_sha256"]:
        raise ValueError("Campaign driver changed")
    if assets.sha256(args.input_plan) != binding["input_plan"]["sha256"]:
        raise ValueError("Input plan changed")
    summaries = campaign.read_extraction(job, bound_sha)
    index = inputs.metadata_index(output.read_json(args.input_plan))
    campaign.validate_full_extraction(summaries, index)
    parameters = fit["parameters"]
    records = [precision_record(row, parameters, fit["anchors_by_run"][row["run"]]) for row in summaries]
    selected = select_exact_rows(records)
    context = dict(input_plan=assets.asset(args.input_plan), round1_freeze=binding["round1_freeze"],
        fit=assets.asset(fit_path), parameters=parameters, anchors_by_run=fit["anchors_by_run"], summaries=summaries)
    output_folder = job / "calibration/exact_validation"
    output_folder.mkdir(parents=True, exist_ok=False)
    campaign.new_json(output_folder / "audit_plan.json",dict(contract=CONTRACT, selected=selected,
        precision_records=records, maximum_precision_fraction=MAX_PRECISION_FRACTION,
        heldout_pixels_read=False, context_bindings={k:v for k,v in context.items() if k not in ("summaries", "anchors_by_run")}))
    tasks = [(run, [row for row in selected if row["run"] == run]) for run in engine.FIT_RUNS]
    results, failures = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(context,)) as pool:
        futures = {pool.submit(audit_run,task):task[0] for task in tasks}
        for future in as_completed(futures):
            run_id=futures[future]
            try:
                rows=future.result(); results.extend(rows)
                campaign.new_json(output_folder / "runs" / f"{run_id}.json",dict(rows=rows,complete=True))
                print(json.dumps(dict(event="exact_replay",run=run_id,frames=len(rows))),flush=True)
            except Exception as exc:
                failure=dict(run=run_id,error=repr(exc)); failures.append(failure)
                print(json.dumps(dict(event="exact_error",**failure)),flush=True)
    dose_bounds = [row["bound_over_dose"] for row in records if row["bound_over_dose"] is not None]
    complete = not failures and len(results) == len(selected)
    verified = complete and max(dose_bounds,default=0) <= MAX_PRECISION_FRACTION and all(
        row["within_summary_precision_bound"] and row["fixed_gas_matches_physical_summary"] for row in results)
    for descriptor in (context["input_plan"],context["fit"],context["round1_freeze"]):
        if assets.sha256(descriptor["path"]) != descriptor["sha256"]:
            raise ValueError("Audit binding changed while replaying")
    report=dict(contract=CONTRACT,complete=complete,heldout_pixels_read=False,
        summary_precision_bound_verified=verified,parameters=parameters,fit=context["fit"],
        input_plan=context["input_plan"],campaign_binding=assets.asset(binding_path),
        audit_tool=assets.asset(__file__),summary_frames=len(records),exact_frames=len(results),
        maximum_bound_over_dose=max(dose_bounds,default=0),maximum_precision_fraction=MAX_PRECISION_FRACTION,
        maximum_exact_difference_g=max((row["absolute_difference_g"] for row in results),default=0),
        results=results,failures=failures,phase_labels_modified=False,bottom_mass_allowed=False,
        wall_seconds=time.perf_counter()-started,created_utc=datetime.now(timezone.utc).isoformat())
    campaign.new_json(output_folder / "validation.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True)
    parser.add_argument("--input-plan",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args(argv)
    if not 1 <= args.workers <= 24: parser.error("Use 1–24 workers, one internal thread each")
    result=run(args)
    print(json.dumps({k:v for k,v in result.items() if k not in ("results","failures")}),flush=True)
    return 0 if result["summary_precision_bound_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
