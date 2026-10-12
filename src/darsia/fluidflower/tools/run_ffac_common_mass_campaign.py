"""Resumable staged common native FF_AC concentration/mass campaign.

Collection is calibration-only. Fitting also requires the existing workflow's
saved CalibrationPlanStore plan. Held-out inputs are admitted only after the
immutable freeze, and production stops if the pre-registered check fails.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from darsia.fluidflower import phase_ffac_common_mass as native
from darsia.fluidflower import phase_ffac_common_mass_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_production as output

CONTRACT = "ffac-common-native-mass-campaign-v1"
_SESSION = {}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _stamp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _new(path, value):
    return output.write_new_json(path, value)


def _bound_new_or_existing(path, value):
    path = Path(path)
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"Immutable campaign binding differs: {path}")
    else:
        _new(path, value)


def _commit_directory(folder, payload, *, arrays=None):
    """Publish once; partial producer directories never masquerade as receipts."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        raise FileExistsError(folder)
    temporary = folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.partial")
    temporary.mkdir()
    assets = {}
    if arrays:
        path = temporary / "arrays.npz"
        np.savez(path, **arrays)
        assets["arrays"] = dict(path="arrays.npz", sha256=output.sha_file(path))
    _new(temporary / "receipt.json", dict(payload, assets=assets, complete=True))
    temporary.rename(folder)


def _resume(folder, binding):
    path = Path(folder) / "receipt.json"
    if not path.is_file():
        return None
    value = _json(path)
    if value.get("complete") is not True or value.get("campaign_binding_sha256") != binding:
        raise ValueError(f"Committed receipt belongs to another campaign: {path}")
    for asset in value.get("assets", {}).values():
        if output.sha_file(path.parent / asset["path"]) != asset["sha256"]:
            raise ValueError(f"Committed stage bytes changed: {path}")
    return value


def init_worker(plan_path, clearwater_results=None, overflow_input=None):
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    plan = _json(plan_path)
    if clearwater_results:
        plan = dict(plan, clearwater_results_folder=str(clearwater_results))
    source = inputs.read_bound_json(plan["source_metadata"])
    _SESSION.update(plan=plan, index=source, reader=inputs.VerifiedAssetReader(),
                    thread_limit=threadpool_limits(limits=1))
    if overflow_input:
        descriptor = inputs.asset(overflow_input)
        comparison = inputs.read_bound_json(descriptor)
        if comparison.get("water_displacement_not_CO2_outlet") is not True:
            raise ValueError("Overflow comparison must distinguish rig inventory from CO2 outlet")
        _SESSION["overflow"] = {(row["run"], row["frame_index"]): row for row in comparison["frames"]}
        _SESSION["overflow_descriptor"] = descriptor


def stage_selection(index, run, stage):
    rows = sorted((row for row in index["frames"] if row["run"] == run), key=lambda row: row["frame_index"])
    if stage == "collect":
        choices = output.nearest_frame_indices(rows)
        keep = {row["frame_index"] for row in choices}
        return [row for row in rows if row["frame_index"] in keep], choices
    if stage == "summaries":
        return [row for row in rows if 0 < row["elapsed_h"] <= 2.25], []
    if stage == "heldout-check":
        choices = output.nearest_frame_indices(rows, (1., 2.25), include_last=False)
        keep = {row["frame_index"] for row in choices}
        return [row for row in rows if row["frame_index"] in keep], choices
    return rows, output.nearest_frame_indices(rows)


def _frame(metadata, *, freeze=None, l1_reference=None):
    prepared = inputs.load_common_frame(metadata, _SESSION["plan"], session_reader=_SESSION["reader"], frozen_calibration=freeze)
    l1 = None
    if l1_reference:
        l1, evidence = inputs.l1_protocol_proxy_neighbourhood(metadata, l1_reference, session_reader=_SESSION["reader"])
        evidence["l1_position_source_sha256"] = l1_reference["sha256"]
        prepared = replace(prepared, provenance=dict(prepared.provenance, **evidence))
    positive = np.isin(prepared.phase_labels, (1, 2))
    prepared = replace(prepared, provenance=dict(prepared.provenance,
        mass_fit_blocked_phase_counts={
            "bottom_strip": int(np.count_nonzero(positive & prepared.bottom_strip_mask)),
            "below_photo_filled": int(np.count_nonzero(positive & prepared.below_photo_filled_mask))}))
    return inputs.to_native_frame(prepared, l1_neighbourhood_mask=l1)


def no_outflow_proxy(frame, metadata):
    """Fixed early-time optical proxy, explicitly not a measured outlet flow."""
    occupied = frame.domain.any(axis=0)
    first = frame.domain.argmax(axis=0)
    top = frame.phase_labels[first, np.arange(frame.domain.shape[1])]
    contact = int(np.count_nonzero(occupied & np.isin(top, (1, 2))))
    admissible = 0 < frame.elapsed_h <= 2.25 and contact == 0
    blocked = dict(getattr(frame, "provenance", {}).get("mass_fit_blocked_phase_counts", {}))
    for name, mask in (getattr(frame, "excluded_masks", None) or {}).items():
        blocked.setdefault(name, int(np.count_nonzero(np.isin(frame.phase_labels, (1, 2)) & mask)))
    reasons = []
    if not 0 < frame.elapsed_h <= 2.25:
        reasons.append("outside_declared_early_injection_window")
    if contact:
        reasons.append("locked_CO2_contacts_open_top")
    return dict(no_co2_left_rig=admissible, source=dict(metadata["receipt"]),
        mass_fit_admissible=not reasons, mass_fit_rejection_reasons=reasons,
        blocked_positive_pixels=blocked,
        excluded_strip_fit_policy="physical-dose-interval-v1",
        blocked_positive_excluded_from_main_mass=True,
        blocked_positive_is_not_a_frame_rejection=True,
        status="inferred_early_locked_phase_open_top_proxy", camera_time_h=frame.elapsed_h,
        co2_contact_columns=contact, spatial_margin_px=0,
        policy="0 < camera time <= 2.25 h; no locked CO2 at actual first domain pixel in any occupied column",
        limitations="No CO2 outlet measurement; locked phase and top-surface rules can weaken this optical evidence",
        measured_water_displacement_is_not_co2_outlet=True)


def collect_run(task):
    run, job, binding, stride = task
    if run not in native.CALIBRATION_RUNS:
        raise ValueError("Pooled paths cannot read held-out or excluded pixels")
    target = Path(job) / "work/path_statistics" / run
    cached = _resume(target, binding)
    if cached:
        return dict(run=run, cached=True, **{k: cached[k] for k in ("frame_count", "wall_seconds")})
    started = time.perf_counter(); cpu = time.process_time()
    rows, choices = stage_selection(_SESSION["index"], run, "collect")
    samples = {}; audits = []
    for metadata in rows:
        frame = _frame(metadata)
        # Deliberately explicit: provisional geometry stays reported. Its
        # reference and uncertainty are not silently called photo certified.
        frame = replace(frame, provenance=dict(frame.provenance, physical_context_verified=True,
            physical_context_verification_scope="SHA-bound Olav-accepted provisional V3; boundary uncertainty retained", facies_interior_radius_px=20))
        contribution, audit = native.sample_frame_for_common_paths(frame, stride_px=stride)
        for facies, values in contribution.items():
            samples.setdefault(facies, []).append(values)
        audits.append(audit)
        del frame
        gc.collect()
    arrays = {f"facies_{label}": np.concatenate(values).astype(np.float32) for label, values in samples.items()}
    payload = dict(contract=CONTRACT, stage="collect", run=run, campaign_binding_sha256=binding,
        frame_count=len(rows), camera_time_selection=choices, sampling_stride_px=stride,
        sampling_only_not_image_downsampling=True, production_images_full_resolution=True,
        frames=audits, heldout_pixels_read=False, wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
    _commit_directory(target, payload, arrays=arrays)
    return dict(run=run, frame_count=len(rows), wall_seconds=payload["wall_seconds"], cpu_seconds=payload["cpu_seconds"], cached=False)


def summaries_run(task):
    run, job, binding, path_folder, policy = task
    if run not in native.CALIBRATION_RUNS or policy != "early-open-top-proxy":
        raise ValueError("Mass summaries require a calibration run and explicit predeclared no-outflow policy")
    target = Path(job) / "work/fit_summaries" / run
    cached = _resume(target, binding)
    if cached:
        return dict(run=run, cached=True, frame_count=len(cached["summaries"]))
    started = time.perf_counter(); cpu = time.process_time()
    rows, _ = stage_selection(_SESSION["index"], run, "summaries")
    summaries, rejected = [], []
    for metadata in rows:
        frame = _frame(metadata)
        frame = replace(frame, provenance=dict(frame.provenance, physical_context_verified=True,
            physical_context_verification_scope="SHA-bound Olav-accepted provisional V3; boundary uncertainty retained"))
        evidence = no_outflow_proxy(frame, metadata)
        if evidence["mass_fit_admissible"]:
            response = native.prepare_native_response(frame, path_folder)
            summaries.append(native.summarize_native_response(response, no_outflow_evidence=evidence))
            del response
        else:
            rejected.append(dict(run=run, frame_index=metadata["frame_index"], elapsed_h=metadata["elapsed_h"], evidence=evidence))
        del frame; gc.collect()
    payload = dict(contract=CONTRACT, stage="summaries", run=run, campaign_binding_sha256=binding,
        summaries=summaries, rejected_frames=rejected, path_folder=str(path_folder),
        omitted_calibration_run=not summaries,
        omission_reason="No early source-bound open-top-proxy frame" if not summaries else None,
        excluded_strip_fit_policy="physical-dose-interval-v1",
        wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu, heldout_pixels_read=False)
    _commit_directory(target, payload)
    return dict(run=run, frame_count=len(summaries), rejected_frames=len(rejected), omitted_calibration_run=not summaries,
        wall_seconds=payload["wall_seconds"], cpu_seconds=payload["cpu_seconds"], cached=False)


def _production_metadata(frame, metadata):
    result = dict(run=metadata["run"], frame_index=metadata["frame_index"], elapsed_h=metadata["elapsed_h"],
        source_sha256=metadata["source_sha256"], phase_sha256=metadata["labels"]["sha256"], pixel_area_sha256=metadata["area_map"]["sha256"],
        nominal_injected_mass_g=frame.nominal_injected_mass_g, pressure_bar=frame.pressure_bar, temperature_c=frame.temperature_c)
    evidence = no_outflow_proxy(frame, metadata)
    comparison = _SESSION.get("overflow", {}).get((metadata["run"], metadata["frame_index"]))
    if comparison is not None and (comparison["source_sha256"] != metadata["source_sha256"] or comparison["camera_time_h"] != metadata["elapsed_h"]):
        raise ValueError("Overflow observation is bound to a different camera frame")
    overflow = dict((comparison or {}).get("overflow") or metadata.get("overflow") or {})
    result["overflow"] = dict(overflow, co2_has_left_rig=False if evidence["no_co2_left_rig"] else None,
        no_loss_status=evidence["status"], no_loss_source=evidence,
        comparison_input=_SESSION.get("overflow_descriptor"),
        comparison_status=(comparison or {}).get("comparison_status", "Independent overflow input unavailable"),
        no_loss_assumption="No CO2 outlet measurement; early-time no-loss proxy is inferred from the locked top domain boundary")
    return result


def _saved_production(folder, metadata, digest):
    """Verify each committed byte once per worker session before reuse."""
    path = Path(folder) / "receipt.json"
    if not path.is_file():
        return None
    receipt = _json(path)
    expected = dict(calibration_freeze_sha256=digest, source_sha256=metadata["source_sha256"],
        elapsed_h=metadata["elapsed_h"], phase_sha256=metadata["labels"]["sha256"],
        pixel_area_sha256=metadata["area_map"]["sha256"], run=metadata["run"], frame_index=metadata["frame_index"])
    if receipt.get("complete") is not True or any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing concentration frame is not this frozen source: {path}")
    for descriptor in receipt["assets"].values():
        _SESSION["reader"].path(dict(descriptor, path=str(path.parent / descriptor["path"])))
    return receipt


def produce_run(task):
    run, job, binding, freeze_path, stage, l1_reference = task
    freeze, digest = native.verify_frozen_calibration(freeze_path)
    evidence = output.FrozenCalibrationEvidence.verify(freeze_path)
    target = Path(job) / "production" / run
    rows, choices = stage_selection(_SESSION["index"], run, stage)
    wanted_sensitivity = {item["frame_index"] for item in output.nearest_frame_indices([row for row in _SESSION["index"]["frames"] if row["run"] == run])}
    started = time.perf_counter(); cpu = time.process_time(); receipts = []
    for metadata in rows:
        folder = target / f"frame_{metadata['frame_index']:03d}"
        receipt = _saved_production(folder, metadata, digest)
        sensitivity_path = folder / "sensitivities.json"
        sensitivity_needed = metadata["frame_index"] in wanted_sensitivity
        if sensitivity_path.is_file():
            saved_sensitivity = _json(sensitivity_path)
            if (saved_sensitivity.get("calibration_freeze_sha256") != digest or
                    saved_sensitivity.get("source_sha256") != metadata["source_sha256"]):
                raise ValueError(f"Saved sensitivity has a different frozen source: {sensitivity_path}")
        if receipt and (not sensitivity_needed or sensitivity_path.is_file()):
            receipts.append(receipt)
            continue
        frame = _frame(metadata, freeze=freeze, l1_reference=l1_reference if metadata["frame_index"] in wanted_sensitivity and metadata["elapsed_h"] >= 4 else None)
        frame = replace(frame, provenance=dict(frame.provenance, physical_context_verified=True,
            physical_context_verification_scope="SHA-bound Olav-accepted provisional V3; boundary uncertainty retained"))
        response = native.prepare_native_response(frame, freeze["path_folder"], frozen_calibration_path=freeze_path)
        if receipt is None:
            result = native.evaluate_common_prepared_response(response, freeze_path, expected_freeze_sha256=digest)
            receipt = output.save_frame_result(folder, result, _production_metadata(frame, metadata), evidence)
            del result
        receipts.append(receipt)
        if sensitivity_needed and not sensitivity_path.is_file():
            callback = lambda ignored_frame, saved_freeze, **kwargs: native.evaluate_common_prepared_response(response, saved_freeze, expected_freeze_sha256=digest, **kwargs)
            scenarios = output.evaluate_sensitivities(frame, freeze_path, callback,
                heldout=run in native.HELDOUT_RUNS, injector_verified=False,
                provisional_l1_source_bound=frame.provenance.get("l1_position_source_bound") is True)
            _new(sensitivity_path, dict(run=run, frame_index=frame.frame_index, elapsed_h=frame.elapsed_h,
                calibration_freeze_sha256=digest, source_sha256=metadata["source_sha256"], scenarios=scenarios))
        del frame, response; gc.collect()
        print(json.dumps(dict(stage=stage, run=run, frame_index=metadata["frame_index"], elapsed_h=metadata["elapsed_h"], main_total_mass_g=receipt["native_result"]["total_mass_g"])), flush=True)
    return dict(run=run, frame_count=len(receipts), camera_time_selection=choices,
        wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu, receipts=[str(Path(row["frame_folder"])/"receipt.json") for row in receipts])


def _execute(tasks, function, args):
    started = time.perf_counter(); successes, failures = [], []
    started_utc = datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
            initargs=(str(args.input_plan), str(args.clearwater_results) if args.clearwater_results else None,
                      str(args.overflow_input) if args.overflow_input else None)) as pool:
        pending = {pool.submit(function, task): task[0] for task in tasks}
        for future in as_completed(pending):
            run = pending[future]
            try:
                result = future.result(); successes.append(result)
                print(json.dumps(dict(event="run_complete", stage=args.stage, **result)), flush=True)
            except Exception as error:
                failure = dict(run=run, error=repr(error)); failures.append(failure)
                print(json.dumps(dict(event="run_failed", stage=args.stage, **failure)), flush=True)
    record = dict(contract=CONTRACT, stage=args.stage, worker_count=args.workers, internal_threads=1,
        started_utc=started_utc, wall_seconds=time.perf_counter()-started,
        successes=successes, failures=failures, backend="CPU", complete=not failures,
        comparison_input=inputs.asset(args.overflow_input) if args.overflow_input else None)
    _new(args.job / "stage_receipts" / f"{args.stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.json", record)
    return record


def require_workflow_plan(args):
    if args.workflow_plan is None:
        raise ValueError("Fitting requires the existing CalibrationPlanStore saved plan; collection can run independently")
    descriptor = inputs.asset(args.workflow_plan)
    value = inputs.read_bound_json(descriptor)
    config = value.get("configuration", {})
    contract = config.get("native_fixed_phase_contract", {})
    fixed = config.get("fixed_transfer_contract", {})
    if (value.get("format") != "darsia-fluidflower-calibration-revision-v1" or
            config.get("format") != "darsia-fluidflower-calibration-plan-v1" or
            contract.get("contract") != "native-darsia-fixed-phase-mass-v1" or
            fixed.get("source") != "production_labels" or fixed.get("transfer_shape") != "locked_categorical" or
            fixed.get("c_aq_zero_labels") != [0, 255] or fixed.get("s_g_positive_labels") != [2]):
        raise ValueError("Existing workflow plan must use native locked categorical support")
    native.validate_cohort(contract.get("calibration_runs", ()), contract.get("heldout_runs", ()))
    scopes = contract.get("shared_response_parameters", ())
    if (len(scopes) != 2 or {row.get("key") for row in scopes} != {"aqueous_scale", "gas_scale"} or
            any(row.get("scope") != "shared_across_series" or row.get("domain") != "positive"
                or row.get("parameterization") != "log" or row.get("lower") != 0
                or row.get("upper") is not None for row in scopes)):
        raise ValueError("Workflow plan must fit exactly one recorded shared native scale pair")
    if (contract.get("input_plan", {}).get("sha256") != output.sha_file(args.input_plan) or
            contract.get("approved_spec_sha256") != args.spec_sha256 or
            contract.get("overflow_fit_weight") != 0 or contract.get("map_rescaling_allowed") is not False or
            contract.get("freeze_before_heldout") is not True):
        raise ValueError("Workflow plan input/spec/no-rescaling/freeze binding differs")
    policy = contract.get("no_outflow_frame_policy", {})
    if (policy.get("name") != args.no_outflow_policy or policy.get("min_time_exclusive_h") != 0 or
            policy.get("max_time_h") != 2.25 or policy.get("reject_co2_contact_with_top_domain") is not True or
            policy.get("reject_co2_in_excluded_strips") is not False or
            policy.get("excluded_strip_fit_policy") != "physical-dose-interval-v1"):
        raise ValueError("Workflow fitting-frame policy differs from the fixed campaign rule")
    objectives = {row["key"]: row for row in config.get("objectives", ())}
    if (objectives.get("injected_mass_balance", {}).get("enabled") is not True or
            objectives.get("injected_mass_balance", {}).get("weight") != 1 or
            any(row.get("enabled") and row.get("weight") != 0 for key, row in objectives.items() if key != "injected_mass_balance")):
        raise ValueError("Only injected mass may contribute to the shared objective")
    selection = value.get("selection_contract_sha256")
    if not selection or selection != value.get("objective_contract", {}).get("content_sha256"):
        raise ValueError("Workflow selection fingerprint is missing or inconsistent")
    binding = dict(contract=CONTRACT, workflow_plan=descriptor, selection_contract_sha256=selection)
    external_path = args.workflow_binding or args.job / "calibration_plan_binding_final.json"
    if external_path.is_file():
        external_descriptor = inputs.asset(external_path)
        external = inputs.read_bound_json(external_descriptor)
        if (external.get("calibration_plan", {}).get("sha256") != descriptor["sha256"] or
                external.get("selection_contract_sha256") != selection or
                external.get("input_plan", {}).get("sha256") != output.sha_file(args.input_plan)):
            raise ValueError("Saved CalibrationPlanStore descriptor differs from the actual selected plan")
        binding["saved_plan_binding"] = external_descriptor
    _bound_new_or_existing(args.job / "workflow_plan_binding.json", binding)
    return descriptor


def paths_stage(args, binding):
    require_workflow_plan(args)
    target = args.job / "calibration/native_color_paths"
    saved = args.job / "calibration/path_package_receipt.json"
    if saved.is_file():
        receipt = _json(saved)
        if receipt.get("campaign_binding_sha256") != binding:
            raise ValueError("Saved common paths belong to another input plan")
        for name, digest in receipt["files"].items():
            if output.sha_file(name) != digest:
                raise ValueError("Saved common path changed")
        return receipt
    pooled, source_bindings = {}, {}
    for run in native.CALIBRATION_RUNS:
        folder = args.job / "work/path_statistics" / run
        receipt = _resume(folder, binding)
        if not receipt:
            raise ValueError(f"Pooled collection incomplete: {run}")
        source_bindings[run] = receipt
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            for key in archive.files:
                label = int(key.removeprefix("facies_"))
                pooled.setdefault(label, {})[run] = archive[key]
    paths, audit = native.fit_pooled_color_paths(pooled)
    native.save_common_path_package(target, paths, path_audit=audit, source_bindings=source_bindings)
    receipt = dict(contract=CONTRACT, campaign_binding_sha256=binding, path_folder=str(target), audit=audit,
        files={str(path): output.sha_file(path) for path in target.rglob("*") if path.is_file()}, complete=True)
    _new(saved, receipt)
    return receipt


def freeze_stage(args, binding):
    workflow = require_workflow_plan(args)
    path_package = paths_stage(args, binding)
    freeze_path = args.job / "calibration/freeze/frozen_calibration.json"
    if freeze_path.is_file():
        native.verify_frozen_calibration(freeze_path)
        return freeze_path
    summaries = []
    for run in native.CALIBRATION_RUNS:
        receipt = _resume(args.job / "work/fit_summaries" / run, binding)
        if not receipt:
            raise ValueError(f"Mass fitting summaries incomplete: {run}")
        summaries.extend(receipt["summaries"])
    fit_path = args.job / "calibration/shared_fit.json"
    fit = _json(fit_path) if fit_path.exists() else native.fit_shared_mass_scales(summaries)
    if not fit_path.exists():
        _new(fit_path, fit)
    if fit.get("fit_success") is not True:
        raise ValueError("Shared native mass optimization did not succeed; no freeze published")
    exact_descriptor = inputs.asset(args.job / "calibration/exact_validation/validation.json")
    exact = inputs.read_bound_json(exact_descriptor)
    if (exact.get("complete") is not True or exact.get("heldout_pixels_read") is not False
            or exact.get("parameters") != fit["parameters"]
            or exact.get("fit", {}).get("sha256") != output.sha_file(fit_path)
            or exact.get("summary_precision_bound_verified") is not True):
        raise ValueError("Exact native calibration-frame validation is incomplete or belongs to another common pair")
    fit = dict(fit, exact_native_calibration_validation=exact_descriptor,
               exact_validation_summary=exact.get("summary", {}))
    extra_files = {str(args.workflow_plan): workflow["sha256"], str(args.input_plan): output.sha_file(args.input_plan),
        str(args.job/"workflow_plan_binding.json"): output.sha_file(args.job/"workflow_plan_binding.json"),
        str(args.job/"campaign_binding.json"): output.sha_file(args.job/"campaign_binding.json"),
        str(fit_path): output.sha_file(fit_path), str(exact_descriptor["path"]): exact_descriptor["sha256"]}
    saved_binding = _json(args.job/"workflow_plan_binding.json").get("saved_plan_binding")
    if saved_binding:
        extra_files[saved_binding["path"]] = saved_binding["sha256"]
    return native.freeze_calibration(args.job / "calibration/freeze", path_folder=path_package["path_folder"], fit_result=fit,
        spec_sha256=args.spec_sha256, input_plan_sha256=output.sha_file(args.input_plan),
        extra_files=extra_files)


def publish_workflow_review(args, freeze_path, *, check_path, report_path=None):
    """Attach a pending native result to the existing isolated review workflow."""
    from darsia.fluidflower.calibration import CalibrationPlanStore
    from darsia.fluidflower.calibration_review import CalibrationResultReviewStore, default_calibration_review_config
    from darsia.fluidflower.project import FluidFlowerProject
    plan_descriptor = require_workflow_plan(args)
    external_path = args.workflow_binding or args.job / "calibration_plan_binding_final.json"
    external = _json(external_path)
    project = FluidFlowerProject.load(Path(external["project_root"]))
    plan = CalibrationPlanStore(project).current()
    if inputs.asset(args.workflow_plan)["sha256"] != plan_descriptor["sha256"] or plan["plan_id"] != _json(args.workflow_plan)["plan_id"]:
        raise ValueError("Existing workflow changed its selected calibration plan")
    freeze, freeze_sha = native.verify_frozen_calibration(freeze_path)
    folder = args.job / "workflow_results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    parameter = folder / "native_parameters.json"
    _new(parameter, dict(format="darsia-fluidflower-mass-parameters-v1", native_contract="native-darsia-fixed-phase-mass-v1",
        selected_model="titration", parameter_scope="shared_across_series", phase_support_source="production_labels",
        map_mass_rescaling=False, native_frozen_calibration=dict(path=str(freeze_path), sha256=freeze_sha),
        shared_parameters=freeze["shared_scales"], run_parameters={}))
    manifest = folder / "calibration_result.json"
    _new(manifest, dict(format="darsia-fluidflower-calibration-result-v1", calibration_plan_id=plan["plan_id"],
        objective_contract_sha256=plan["selection_contract_sha256"], selected_model="titration",
        parameter_artifact=inputs.asset(parameter), native_frozen_calibration=inputs.asset(freeze_path),
        parity=dict(reference_backend="cpu-float64", execution_backend="cpu-float64", passed=True,
            basis="Same native CPU float64 implementation is both the execution and parity reference; no alternate GPU result used"),
        heldout_check=inputs.asset(check_path), report=str(report_path) if report_path else None,
        scientific_status="pending human concentration/mass review; fixed-phase conditional validation",
        diagnostic_on_failure=True, phase_parent_changed=False, default_changed=False, bottom_mass_allowed=False))
    config = default_calibration_review_config(project)
    config["calibration_result_manifest_path"] = str(manifest)
    config["run_reviews"] = {}
    revision = CalibrationResultReviewStore(project).save_review(config, actor="Codex",
        reason="Source-bound common native fit and held-out diagnostics; all run approvals remain pending")
    return dict(manifest=str(manifest), review_id=revision["review_id"], project_root=str(project.root),
        approved=False, default_changed=False)


def _production_rows(job):
    return [_json(path) for path in sorted((Path(job)/"production").glob("*/frame_*/receipt.json"))]


def _photo_path(candidate_root, run, index):
    from darsia.fluidflower import phase_rig_display_prerender as rendered
    pointer = _json(rendered.pointer_path(candidate_root))
    manifest = rendered._read_verified(pointer["manifest"])
    run_manifest = rendered._read_verified(manifest["runs"][run])
    return run_manifest["records"][f"frame_{index:03d}"]["raw_detail"]["display"]["path"]


def report_stage(args, index):
    evidence = output.FrozenCalibrationEvidence.verify(args.job / "calibration/freeze/frozen_calibration.json")
    rows = _production_rows(args.job)
    expected = {(r["run"], r["frame_index"]) for r in index["frames"] if r["run"] in native.CALIBRATION_RUNS or r["run"] in native.HELDOUT_RUNS}
    if {(r["run"], r["frame_index"]) for r in rows} != expected:
        raise ValueError("Full 39-run concentration inventory is not yet complete")
    sensitivities = []
    for path in sorted((args.job/"production").glob("*/frame_*/sensitivities.json")):
        value = _json(path)
        for choice in output.nearest_frame_indices([r for r in rows if r["run"] == value["run"]]):
            if choice["frame_index"] == value["frame_index"]:
                sensitivities.append(dict(value, **choice))
    report = args.job / ("report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    check = output.write_report_tables(report, rows, freeze=evidence, sensitivities=sensitivities)
    workflow_review = publish_workflow_review(args, evidence.path, check_path=report / "heldout_check1.json", report_path=report)
    if check["outcome"] != "pass":
        return dict(report=str(report), registered=False, check1=check["outcome"], workflow_review=workflow_review)
    candidate = Path(index["candidate_root"])
    by_key = {(r["run"], r["frame_index"]): r for r in index["frames"]}
    output.render_report_figures(report / "figures", rows,
        photo_resolver=lambda row: _photo_path(candidate, row["run"], row["frame_index"]),
        phase_resolver=lambda row: by_key[(row["run"], row["frame_index"])]["labels"]["path"])
    from darsia.fluidflower.phase_validation import _model_identity
    model_id, model_sha = _model_identity(candidate)
    series_root = next(parent.parent for parent in candidate.parents if parent.name == "results")
    pointer = output.register_concentration_review(series_root, args.job / "production", rows, freeze=evidence,
        candidate_id=model_id, candidate_sha256=model_sha)
    return dict(report=str(report), registration=str(pointer), registered=True, check1=check["outcome"], workflow_review=workflow_review)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--stage", choices=("collect", "paths", "summaries", "freeze", "heldout-check", "produce", "report"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--workflow-plan", type=Path)
    parser.add_argument("--workflow-binding", type=Path)
    parser.add_argument("--clearwater-results", type=Path)
    parser.add_argument("--l1-reference", type=Path)
    parser.add_argument("--overflow-input", type=Path)
    parser.add_argument("--no-outflow-policy", choices=("early-open-top-proxy",), default="early-open-top-proxy")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.stride < 1:
        parser.error("Workers and sampling stride must be positive")
    if len(args.spec_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in args.spec_sha256.lower()):
        parser.error("The approved specification requires its complete SHA-256")
    args.job.mkdir(parents=True, exist_ok=True)
    plan = _json(args.input_plan)
    index = inputs.read_bound_json(plan["source_metadata"])
    native.validate_cohort(plan["split"]["calibration_runs"], plan["split"]["held_out_runs"])
    binding = dict(contract=CONTRACT, spec_sha256=args.spec_sha256, input_plan=inputs.asset(args.input_plan),
        code_sha256={str(Path(module.__file__).resolve()): output.sha_file(module.__file__) for module in (native, inputs, output)},
        driver_sha256=output.sha_file(__file__), sampling_stride_px=args.stride,
        no_outflow_policy=args.no_outflow_policy,
        mass_fit_blocked_positive_policy="retain source-bound counts; excluded capacity enters physical dose interval, never main mass",
        excluded_strip_fit_policy="physical-dose-interval-v1",
        no_map_mass_rescaling=True, no_group_or_run_parameter=True)
    approved_spec = args.job / "approved_spec.md"
    if approved_spec.exists():
        if output.sha_file(approved_spec) != args.spec_sha256:
            raise ValueError("The job's approved specification differs from the requested SHA-256")
        binding["approved_spec"] = inputs.asset(approved_spec)
    _bound_new_or_existing(args.job / "campaign_binding.json", binding)
    digest = _stamp(binding)
    selected = {name.lower() for name in args.runs} if args.runs else None
    allowed = set(native.HELDOUT_RUNS if args.stage == "heldout-check" else
                  (*native.CALIBRATION_RUNS, *native.HELDOUT_RUNS) if args.stage in {"produce", "report"} else native.CALIBRATION_RUNS)
    if selected and not selected <= allowed:
        raise ValueError(f"Requested runs outside this stage's frozen cohort: {sorted(selected-allowed)}")
    if args.stage in {"collect", "summaries"}:
        runs = [run for run in native.CALIBRATION_RUNS if selected is None or run in selected]
        if args.stage == "collect":
            record = _execute([(run, str(args.job), digest, args.stride) for run in runs], collect_run, args)
        else:
            require_workflow_plan(args)
            package = paths_stage(args, digest)
            record = _execute([(run, str(args.job), digest, package["path_folder"], args.no_outflow_policy) for run in runs], summaries_run, args)
    elif args.stage == "paths":
        record = paths_stage(args, digest)
    elif args.stage == "freeze":
        record = dict(freeze=str(freeze_stage(args, digest)))
    elif args.stage in {"heldout-check", "produce"}:
        if args.overflow_input is None:
            raise ValueError("Source-bound compare-only overflow metadata is required before the first held-out computation")
        freeze_path = args.job / "calibration/freeze/frozen_calibration.json"
        native.verify_frozen_calibration(freeze_path)
        if args.stage == "produce":
            guard = _json(args.job / "heldout_check1.json")
            if guard.get("outcome") != "pass":
                raise ValueError("Held-out check 1 failed or is unassessable; stop and report, never alter a run/group")
        pool = native.HELDOUT_RUNS if args.stage == "heldout-check" else (*native.CALIBRATION_RUNS, *native.HELDOUT_RUNS)
        runs = [run for run in pool if selected is None or run in selected]
        reference = inputs.asset(args.l1_reference) if args.l1_reference else None
        record = _execute([(run, str(args.job), digest, str(freeze_path), args.stage, reference) for run in runs], produce_run, args)
        if args.stage == "heldout-check" and not record["failures"]:
            check = output.heldout_check1(_production_rows(args.job))
            _bound_new_or_existing(args.job / "heldout_check1.json", check)
            record["check1"] = check["outcome"]
            record["workflow_review"] = publish_workflow_review(args, freeze_path, check_path=args.job / "heldout_check1.json")
    else:
        record = report_stage(args, index)
    print(json.dumps(output._jsonable(record), allow_nan=False), flush=True)
    return 2 if record.get("failures") or record.get("check1") in {"fail", "not_assessable_missing_outlet_evidence"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
