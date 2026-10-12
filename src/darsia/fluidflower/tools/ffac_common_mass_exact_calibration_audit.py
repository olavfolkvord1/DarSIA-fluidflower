"""Full-resolution numerical audit of the common native calibration summaries.

Only the predeclared calibration cohort is admitted. No calibration parameter,
phase map, scientific source, or existing job artifact is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import inspect
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from darsia.fluidflower import phase_ffac_common_mass as native
from darsia.fluidflower import phase_ffac_common_mass_inputs as inputs

CONTRACT = "ffac-common-native-exact-calibration-audit-v1"
_SESSION = {}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _stamp(value):
    return native.canonical_sha(value)


def parameters_from_fit(fit):
    pair = dict(fit["parameters"])
    if set(pair) != {"aqueous_scale", "gas_scale"}:
        raise ValueError("Exactly one shared aqueous/gas scale pair is required")
    if not np.isfinite(list(pair.values())).all() or min(pair.values()) <= 0:
        raise ValueError("Finite positive fixed scales are required")
    native.validate_cohort(fit["calibration_runs"], fit["heldout_runs"])
    return pair


def histogram_record(summary, pair):
    """Retain every analytic precision bound without reading scientific pixels."""
    if summary["run"] not in native.CALIBRATION_RUNS:
        raise ValueError("Held-out/excluded data cannot enter a calibration audit")
    if summary.get("contract") != native.CONTRACT:
        raise ValueError("Unexpected native summary contract")
    names = ("aqueous_response", "gas_response", "aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g")
    arrays = [np.asarray(summary[key], np.float64) for key in names]
    if summary.get("intervals") != native.SUMMARY_INTERVALS or any(
            array.ndim != 1 or array.shape != arrays[0].shape or
            not np.isfinite(array).all() or np.any(array < 0) for array in arrays):
        raise ValueError("Invalid native summary histogram")
    if any(np.any(array > 1) for array in arrays[:2]):
        raise ValueError("Native relative responses exceed one")
    dose = float(summary["nominal_injected_mass_g"])
    if not np.isfinite(dose) or dose <= 0 or not np.isfinite(summary["elapsed_h"]):
        raise ValueError("A sourced fitting frame needs finite time and positive dose")
    prediction = native.evaluate_common_summary(summary, pair)
    gas_positive = (arrays[1] > 0) & (arrays[4] > 0)
    return dict(run=summary["run"], frame_index=int(summary["frame_index"]),
        elapsed_h=float(summary["elapsed_h"]), injected_mass_target_g=dose,
        **prediction,
        bound_over_dose=prediction["summary_mass_error_bound_g"] / dose,
        gas_min_positive_histogram_response=float(arrays[1][gas_positive].min()) if gas_positive.any() else None,
        summary_bin_count=len(arrays[0]))


def select_audit_rows(rows):
    """Earliest and largest bound/dose, plus the global positive-gas minimum."""
    grouped = {run: [] for run in native.CALIBRATION_RUNS}
    seen = set()
    for row in rows:
        if row["run"] not in grouped:
            raise ValueError("Selection cannot admit held-out or excluded runs")
        key = row["run"], int(row["frame_index"])
        if key in seen:
            raise ValueError("A summary frame was counted twice")
        seen.add(key)
        grouped[row["run"]].append(row)
    selected = {}
    def add(row, reason):
        key = row["run"], row["frame_index"]
        if key not in selected:
            selected[key] = dict(row, selection_reasons=[])
        selected[key]["selection_reasons"].append(reason)
    for run in native.CALIBRATION_RUNS:
        if not grouped[run]:
            continue
        earliest = min(grouped[run], key=lambda row: (row["elapsed_h"], row["frame_index"]))
        largest = max(grouped[run], key=lambda row: (row["bound_over_dose"], -row["elapsed_h"], -row["frame_index"]))
        add(earliest, "earliest_committed_fit_frame")
        add(largest, "largest_analytic_summary_bound_over_dose")
    positive = [row for row in rows if row["gas_min_positive_histogram_response"] is not None]
    critical = None
    if positive:
        critical = min(positive, key=lambda row: (row["gas_min_positive_histogram_response"], row["run"], row["elapsed_h"], row["frame_index"]))
        add(critical, "global_smallest_positive_gas_histogram_response")
    return sorted(selected.values(), key=lambda row: (row["run"], row["frame_index"])), critical


def floating_allowance_g(summary):
    """Arithmetic allowance, separate from the analytic quantization bound.

    Native response and saturation fields are float32, while capacities and
    integration use float64. This fixed allowance is not a mass-fit criterion.
    """
    capacity = sum(float(np.sum(summary[name], dtype=np.float64)) for name in
                   ("aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g"))
    return float(32 * np.finfo(np.float32).eps * capacity +
                 32 * np.finfo(np.float64).eps * max(capacity, 1.))


def compare_exact_to_summary(exact_mass_g, summary, pair):
    prediction = native.evaluate_common_summary(summary, pair)
    analytic = float(prediction["summary_mass_error_bound_g"])
    arithmetic = floating_allowance_g(summary)
    exact = float(exact_mass_g)
    if not np.isfinite(exact) or exact < 0:
        raise ValueError("Exact native main mass must be finite and nonnegative")
    error = abs(exact - prediction["total_mass_g"])
    return dict(exact_main_mass_g=exact, histogram_main_mass_g=prediction["total_mass_g"],
        absolute_difference_g=error, analytic_summary_error_bound_g=analytic,
        floating_arithmetic_allowance_g=arithmetic,
        within_summary_precision_bound=bool(error <= analytic + arithmetic),
        precision_margin_g=analytic + arithmetic - error)


def _verify_source(summary, metadata, reader):
    expected = metadata["receipt"]
    for descriptor in (summary["source_binding"]["source_receipt"], summary["no_outflow_evidence"]["source"]):
        if descriptor["sha256"] != expected["sha256"] or Path(descriptor["path"]).resolve() != Path(expected["path"]).resolve():
            raise ValueError("Summary/source receipt binding differs")
        reader.path(descriptor)
    if (summary["run"] != metadata["run"] or summary["frame_index"] != metadata["frame_index"]
            or summary["elapsed_h"] != metadata["elapsed_h"]):
        raise ValueError("Summary source/run/camera-time binding differs")


def prepare_context(job, input_plan, fit_path):
    """Verify immutable metadata and component bytes before workers exist."""
    reader = inputs.VerifiedAssetReader()
    binding_path = Path(job) / "campaign_binding.json"
    binding = _json(binding_path)
    binding_stamp = _stamp(binding)
    plan_descriptor = inputs.asset(input_plan)
    if binding["input_plan"]["sha256"] != plan_descriptor["sha256"]:
        raise ValueError("Audit input plan differs from the committed campaign")
    for path, expected in binding["code_sha256"].items():
        reader.path(dict(path=path, sha256=expected))
    driver = Path(__file__).with_name("run_ffac_common_mass_campaign.py")
    reader.path(dict(path=str(driver), sha256=binding["driver_sha256"]))
    plan = reader.json(plan_descriptor)
    source = reader.json(plan["source_metadata"])
    metadata = {(row["run"], row["frame_index"]): row for row in source["frames"] if row["run"] in native.CALIBRATION_RUNS}
    fit_descriptor = inputs.asset(fit_path)
    fit = reader.json(fit_descriptor)
    pair = parameters_from_fit(fit)
    package_descriptor = inputs.asset(Path(job) / "calibration/path_package_receipt.json")
    package = reader.json(package_descriptor)
    if package.get("complete") is not True or package["campaign_binding_sha256"] != binding_stamp:
        raise ValueError("Common path package is incomplete or belongs to another campaign")
    for path, expected in package["files"].items():
        reader.path(dict(path=path, sha256=expected))
    path_folder = Path(package["path_folder"])
    file_inventory = {str(path): inputs.sha256(path) for path in path_folder.rglob("*") if path.is_file()}
    if file_inventory != package["files"]:
        raise ValueError("Common native path component inventory differs")
    summaries, receipts = [], []
    for run in native.CALIBRATION_RUNS:
        descriptor = inputs.asset(Path(job) / "work/fit_summaries" / run / "receipt.json")
        receipt = reader.json(descriptor)
        if (receipt.get("complete") is not True or receipt.get("stage") != "summaries"
                or receipt.get("run") != run or receipt["campaign_binding_sha256"] != binding_stamp):
            raise ValueError("Calibration summary is incomplete or has another binding")
        for row in receipt["summaries"]:
            if (run, row["frame_index"]) not in metadata:
                raise ValueError("Summary source is missing from the fixed calibration input")
            _verify_source(row, metadata[(run, row["frame_index"])], reader)
            histogram_record(row, pair)
            summaries.append(row)
        receipts.append(descriptor)
    used = {(row["run"], row["frame_index"]) for row in fit["frames_used"]}
    actual = {(row["run"], row["frame_index"]) for row in summaries}
    if used != actual or len(actual) != len(summaries) or len(used) != len(fit["frames_used"]):
        raise ValueError("Fixed fit frame set differs from committed summaries")
    histograms = [histogram_record(summary, pair) for summary in summaries]
    selected, critical = select_audit_rows(histograms)
    native_implementations = {str(Path(inspect.getfile(obj)).resolve()): inputs.sha256(inspect.getfile(obj)) for obj in
        (native.darsia.ColorPath, native.darsia.ColorPathInterpolation, native.darsia.ConcentrationAnalysis,
         native.darsia.PWTransformation, native.darsia.TitrationFlash, native.darsia.SimpleFlash,
         native.darsia.CO2MassAnalysis, native.darsia.ExtrudedPorousGeometry, native.NativeConcentrationAnalysis)}
    return dict(job=str(job), input_plan=plan_descriptor, fit=fit_descriptor, parameters=pair,
        code_sha256={**binding["code_sha256"], str(driver): binding["driver_sha256"]},
        campaign_binding=dict(path=str(binding_path), sha256=inputs.sha256(binding_path), canonical_sha256=binding_stamp),
        path_package=package_descriptor, path_folder=str(path_folder), path_component_sha256=file_inventory,
        summary_receipts=receipts, histogram_rows=histograms, selected_rows=selected,
        critical_min_positive_gas_histogram=critical, native_implementation_sha256=native_implementations)


def _init_worker(context):
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    reader = inputs.VerifiedAssetReader()
    plan = reader.json(context["input_plan"])
    fit = reader.json(context["fit"])
    if parameters_from_fit(fit) != context["parameters"]:
        raise ValueError("Fixed scales changed before exact evaluation")
    source = reader.json(plan["source_metadata"])
    index = {(row["run"], row["frame_index"]): row for row in source["frames"] if row["run"] in native.CALIBRATION_RUNS}
    for path, expected in {**context["code_sha256"], **context["path_component_sha256"], **context["native_implementation_sha256"]}.items():
        reader.path(dict(path=path, sha256=expected))
    _SESSION.update(context=context, plan=plan, metadata=index, reader=reader, thread_limit=threadpool_limits(limits=1))


def audit_run(task):
    run, selected = task
    if run not in native.CALIBRATION_RUNS or any(row["run"] != run for row in selected):
        raise ValueError("Exact audit is calibration-only")
    context, reader = _SESSION["context"], _SESSION["reader"]
    receipt_descriptor = next(row for row in context["summary_receipts"] if Path(row["path"]).parent.name == run)
    receipt = reader.json(receipt_descriptor)
    summaries = {row["frame_index"]: row for row in receipt["summaries"]}
    results = []
    for selection in selected:
        started = time.perf_counter(); cpu = time.process_time()
        metadata = _SESSION["metadata"][(run, selection["frame_index"])]
        summary = summaries[selection["frame_index"]]
        _verify_source(summary, metadata, reader)
        loaded = inputs.load_common_frame(metadata, _SESSION["plan"], session_reader=reader)
        frame = inputs.to_native_frame(loaded)
        response = native.prepare_native_response(frame, context["path_folder"])
        pair_before = dict(context["parameters"])
        exact = native.evaluate_native_response(response, pair_before, fields=False)
        if pair_before != context["parameters"]:
            raise ValueError("Native evaluation changed fixed scales")
        gas_mask = frame.known_physical_domain & frame.mass_eligible_mask & response["quantifiable"] & (frame.phase_labels == 2)
        raw_gas = response["prepared"].gas_saturation[gas_mask]
        scaled_gas = np.clip(raw_gas * float(pair_before["gas_scale"]), 0, 1)
        comparison = compare_exact_to_summary(exact["total_mass_g"], summary, pair_before)
        results.append(dict(run=run, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
            injected_mass_target_g=frame.nominal_injected_mass_g, selection_reasons=selection["selection_reasons"],
            **comparison, exact_aqueous_mass_g=exact["aqueous_mass_g"], exact_gas_mass_g=exact["gas_mass_g"],
            exact_locked_eligible_quantifiable_gas_pixels=int(gas_mask.sum()),
            exact_fully_saturated_gas_pixels=int(np.count_nonzero(scaled_gas == 1)),
            exact_zero_saturation_gas_pixels=int(np.count_nonzero(scaled_gas == 0)),
            exact_min_positive_raw_gas_response=float(raw_gas[raw_gas > 0].min()) if np.any(raw_gas > 0) else None,
            exact_gas_saturation_min=float(scaled_gas.min()) if len(scaled_gas) else None,
            exact_gas_saturation_max=float(scaled_gas.max()) if len(scaled_gas) else None,
            native_shape_hw=list(frame.phase_labels.shape), scientific_input_dtype=str(frame.current_rgb.dtype),
            scientific_full_resolution=True, source_receipt=metadata["receipt"],
            path_component_sha256=response["analysis"].source_hashes,
            wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
            backend="CPU", internal_threads=1, phase_labels_modified=False))
        del loaded, frame, response, exact, gas_mask, raw_gas, scaled_gas
        gc.collect()
    return dict(run=run, rows=results)


def run_audit(args):
    context = prepare_context(args.job, args.input_plan, args.fit)
    if args.output.exists():
        raise FileExistsError("Audit reports are immutable; use a new output folder")
    args.output.mkdir(parents=True)
    tasks = [(run, [row for row in context["selected_rows"] if row["run"] == run]) for run in native.CALIBRATION_RUNS]
    tasks = [task for task in tasks if task[1]]
    started = time.perf_counter()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(context,)) as pool:
        futures = {pool.submit(audit_run, task): task[0] for task in tasks}
        for future in as_completed(futures):
            run = futures[future]
            try:
                value = future.result()
                results.extend(value["rows"])
                print(json.dumps(dict(event="run_audited", run=run, exact_frame_count=len(value["rows"]))), flush=True)
            except Exception as error:
                import traceback
                failures.append(dict(run=run, error=repr(error), traceback=traceback.format_exc()))
                print(json.dumps(dict(event="audit_error", **failures[-1])), flush=True)
    reader = inputs.VerifiedAssetReader()
    reader.path(context["fit"])
    reader.path(context["input_plan"])
    reader.path(context["campaign_binding"])
    for path, expected in {**context["code_sha256"], **context["path_component_sha256"], **context["native_implementation_sha256"]}.items():
        reader.path(dict(path=path, sha256=expected))
    complete = not failures and len(results) == len(context["selected_rows"])
    verified = complete and all(row["within_summary_precision_bound"] for row in results)
    report = dict(contract=CONTRACT, complete=complete, heldout_pixels_read=False,
        summary_precision_bound_verified=verified, parameters=dict(context["parameters"]), parameters_unchanged=True,
        fit=context["fit"], input_plan=context["input_plan"], campaign_binding=context["campaign_binding"],
        path_package=context["path_package"], path_component_sha256=context["path_component_sha256"],
        code_sha256=context["code_sha256"], audit_tool=inputs.asset(__file__),
        summary_receipts=context["summary_receipts"], native_implementation_sha256=context["native_implementation_sha256"],
        calibration_runs=list(native.CALIBRATION_RUNS), heldout_runs=list(native.HELDOUT_RUNS),
        histogram_frame_count=len(context["histogram_rows"]), histogram_rows=context["histogram_rows"],
        exact_frame_count=len(results), exact_rows=sorted(results, key=lambda row: (row["run"], row["frame_index"])),
        selected_frame_count=len(context["selected_rows"]), critical_min_positive_gas_histogram=context["critical_min_positive_gas_histogram"],
        failures=failures, wall_seconds=time.perf_counter()-started,
        cpu_seconds_sum=sum(row["cpu_seconds"] for row in results), workers=args.workers,
        internal_threads=1, backend="CPU", no_fit=True, no_map_rescaling=True, maps_modified=False,
        summary=dict(max_analytic_bound_over_dose=max((row["bound_over_dose"] for row in context["histogram_rows"]), default=None),
            max_exact_difference_g=max((row["absolute_difference_g"] for row in results), default=None)),
        interpretation="Numerical precision audit only; not an additional scientific mass-fit acceptance criterion")
    destination = args.output / "validation.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(validation=str(destination), complete=complete, summary_precision_bound_verified=verified)), flush=True)
    return 0 if verified else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--input-plan", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
