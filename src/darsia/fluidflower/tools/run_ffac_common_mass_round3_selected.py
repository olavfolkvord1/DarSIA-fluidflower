"""Immutable selected-image R3 extraction, fits and literal late-balance checks.

This runner deliberately has no historical-frame fallback. A source-bound
adapter supplied by the selected-phase producer must load each selected image
on its exact phase canvas. No real plan can be sealed while that adapter, any
manifest or the 39-run Studio-time gate is merely a forward reference.

The stages are separate commands. Neither a successful check nor this module
starts 39-run mass production: that remains an explicit Olav decision.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .. import phase_ffac_common_mass_inputs as source
from .. import phase_ffac_common_mass_round2 as native
from .. import phase_ffac_common_mass_round2_production as previous
from .. import phase_ffac_common_mass_round3 as engine
from .. import phase_ffac_common_mass_round3_fit as fitter
from .. import phase_ffac_common_mass_round3_inputs as geometry_inputs
from .. import phase_ffac_round3_handoff_admission as handoff

PLAN_CONTRACT = "ffac-common-mass-round3-selected-plan-v1"
INDEX_CONTRACT = "ffac-common-mass-round3-selected-index-v1"
ADAPTER_CONTRACT = "ffac-common-mass-round3-selected-adapter-v1"
CAMPAIGN_CONTRACT = "ffac-common-mass-round3-selected-campaign-v1"
SPEC_SHA256 = "cb7b3355efb27e21d53da868b4155c56fa2fb272c1641e9ab662d56227248e34"
ALL_RUNS = set(engine.FIT_RUNS) | set(engine.HELDOUT_RUNS) | set(engine.DESCRIPTIVE_RUNS)
B0 = dict(candidate="N1", gain=engine.GAIN, beta=engine.B0_BETA)
SESSION = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_new(path, value):
    source.write_new_json(path, _jsonable(value))


def _descriptor(value, reader):
    if not isinstance(value, dict) or not value.get("path") or len(value.get("sha256", "")) != 64:
        raise ValueError("A real source file with SHA-256 is required; no unresolved plan reference")
    reader.path(value)
    return value


def validate_selected_plan(plan_descriptor, reader=None):
    """Check real source assets and all 39 manifests before any pixel loading."""
    reader = reader or source.VerifiedAssetReader()
    plan = reader.json(_descriptor(plan_descriptor, reader))
    if (plan.get("contract") != PLAN_CONTRACT or plan.get("spec_sha256") != SPEC_SHA256
            or set(plan.get("calibration_runs", ())) != set(engine.FIT_RUNS)
            or set(plan.get("heldout_runs", ())) != set(engine.HELDOUT_RUNS)
            or set(plan.get("descriptive_runs", ())) != set(engine.DESCRIPTIVE_RUNS)
            or plan.get("map_mass_rescaling") is not False
            or plan.get("overflow_in_objective") is not False):
        raise ValueError("The selected-image plan changes the approved R3 scope")
    if _descriptor(plan["spec"], reader)["sha256"] != SPEC_SHA256:
        raise ValueError("The selected-image plan uses a stale R3 specification")
    parts = reader.json(_descriptor(plan["saved_parts_plan"], reader))
    if parts["r2_freeze"]["sha256"] != engine.ROUND2_FREEZE_SHA256:
        raise ValueError("B0 must be the exact frozen round-two calibration")
    for name in ("r1_freeze", "r2_freeze", "r2_input_plan", "physical_reference",
                 "physical_coordinate_provenance", "depth_measurements", "facies_properties", "pooled_floor_curve"):
        _descriptor(parts[name], reader)
    gate_descriptor = _descriptor(plan["selected_phase_gate"], reader)
    gate = reader.json(gate_descriptor)
    handoff.require_selected_gate(gate, reader, ALL_RUNS)
    if set(gate.get("run_manifests", {})) != ALL_RUNS:
        raise ValueError("All 39 selected phase manifests and Studio-time parity must be sealed")
    for descriptor in gate["run_manifests"].values():
        _descriptor(descriptor, reader)
    index = reader.json(_descriptor(plan["selected_metadata_index"], reader))
    if index.get("contract") != INDEX_CONTRACT or index.get("complete") is not True:
        raise ValueError("A completed selected-image metadata index is required")
    entries = index.get("entries", [])
    if set(entry.get("run") for entry in entries) != ALL_RUNS:
        raise ValueError("The selected metadata index must cover exactly the 39 approved runs")
    identities, targets = set(), set()
    for entry in entries:
        key = (entry["run"], int(entry["frame_index"]))
        target = (entry["run"], float(entry["selected_target_h"]))
        if key in identities or target in targets:
            raise ValueError("A source frame or target is duplicated in the selected metadata index")
        identities.add(key); targets.add(target)
        if not np.isfinite(entry["elapsed_h"]) or not entry.get("source_binding"):
            raise ValueError("Each selected image requires exact camera time and source binding")
        _descriptor(entry["receipt"], reader)
        if entry.get("run_manifest") != gate["run_manifests"][entry["run"]]:
            raise ValueError("Selected frame and sealed run manifest differ")
    if any((run, 2.25) not in targets for run in ALL_RUNS):
        raise ValueError("Every selected run needs its explicit 2.25 h dilution anchor")
    adapter = plan.get("selected_adapter", {})
    if adapter.get("contract") != ADAPTER_CONTRACT or not adapter.get("function"):
        raise ValueError("A source-bound selected-phase adapter is required; no old-frame substitution")
    _descriptor(adapter["source"], reader)
    for descriptor in plan.get("additional_dependencies", {}).values():
        _descriptor(descriptor, reader)
    old_folds = plan.get("round2_loro_fits", {})
    if old_folds:
        if set(old_folds) != set(engine.FIT_RUNS):
            raise ValueError("B0's fit-run replay needs all 26 original N1 leave-one-run-out receipts")
        for descriptor in old_folds.values():
            _descriptor(descriptor, reader)
    bound_gate = dict(gate, source_binding=dict(receipt=gate_descriptor))
    return dict(plan=plan, parts=parts, gate=bound_gate, entries=entries, reader=reader)


def code_inventory():
    modules = (source, native, previous, engine, fitter, geometry_inputs, handoff,
        importlib.import_module("darsia.fluidflower.phase_ffac_common_mass"),
        importlib.import_module("darsia.fluidflower.phase_ffac_common_mass_round2_inputs"),
        importlib.import_module("darsia.fluidflower.phase_ffac_common_mass_round2_backend"),
        importlib.import_module("darsia.fluidflower.concentration_native"),
        importlib.import_module("darsia.fluidflower.concentration_joint"))
    return {module.__name__: source.asset(module.__file__) for module in modules} | {
        "darsia.fluidflower.tools.run_ffac_common_mass_round3_selected": source.asset(__file__)}


def initialize_campaign(plan_path, job):
    plan_descriptor = source.asset(plan_path)
    validated = validate_selected_plan(plan_descriptor)
    job = Path(job)
    if job.exists() and any(job.iterdir()):
        raise FileExistsError("Initialize in a new empty R3 selected-image job folder")
    # Freeze portable source-equivalent files in the new job, not one user's
    # repository paths. Imported worker code is still checked against these
    # exact bytes; the archive is evidence rather than an import workaround.
    inventory = code_inventory()
    code_binding = {}
    for name, descriptor in inventory.items():
        content = Path(descriptor["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
            raise ValueError("Implementation changed while publishing the R3 source snapshot")
        destination = job / "code_snapshot" / (name.replace(".", "/") + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        code_binding[name] = source.asset(destination)
    campaign = dict(contract=CAMPAIGN_CONTRACT, plan=plan_descriptor, code_binding=code_binding,
        original_code_locations=inventory, source_equivalence_exact=True,
        selected_phase_gate=validated["plan"]["selected_phase_gate"], spec_sha256=SPEC_SHA256,
        created_utc=datetime.now(timezone.utc).isoformat(), production_automatically_allowed=False)
    _write_new(job / "campaign.json", campaign)
    return source.asset(job / "campaign.json")


def init_worker(campaign_descriptor):
    import cv2
    cv2.setNumThreads(1)
    reader = source.VerifiedAssetReader()
    campaign = reader.json(campaign_descriptor)
    if campaign.get("contract") != CAMPAIGN_CONTRACT:
        raise ValueError("A sealed selected-image campaign is required")
    for name, descriptor in campaign["code_binding"].items():
        module = importlib.import_module(name)
        if source.sha256(module.__file__) != descriptor["sha256"]:
            raise ValueError("An immutable R3 implementation differs: " + name)
    validated = validate_selected_plan(campaign["plan"], reader)
    adapter_descriptor = validated["plan"]["selected_adapter"]
    adapter_path = reader.path(adapter_descriptor["source"])
    # A distinct import namespace keeps the selected producer separate from
    # any old run-index loader or previously imported implementation.
    spec = importlib.util.spec_from_file_location("darsia.fluidflower.tools._r3_selected_bound_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    loader = getattr(adapter, adapter_descriptor["function"])
    parts = validated["parts"]
    provenance = reader.json(parts["physical_coordinate_provenance"])
    with np.load(reader.path(parts["physical_reference"]), allow_pickle=False) as archive:
        matrix = geometry_inputs.physical_homography(provenance, archive["x_m"].shape)
        validation = geometry_inputs.validate_reference_coordinates(matrix, archive["x_m"], archive["y_m"])
    points, phi, swi = geometry_inputs.read_measured_properties(
        reader.path(parts["depth_measurements"]), reader.path(parts["facies_properties"]))
    SESSION.clear()
    SESSION.update(validated, campaign=campaign, campaign_descriptor=campaign_descriptor,
        loader=loader, matrix=matrix, coordinate_validation=validation,
        depth_points=points, porosity=phi, swi=swi,
        floor_curve=np.load(reader.path(parts["pooled_floor_curve"]), allow_pickle=False))


def _load_entry(entry, *, scope, frozen_calibration=None):
    if scope == "calibration":
        if entry["run"] not in engine.FIT_RUNS:
            raise ValueError("Final held-out and AC38 pixels cannot enter extraction before freeze")
    elif scope == "frozen_evaluation":
        if frozen_calibration is None:
            raise ValueError("Freeze B1 before calling the selected held-out pixel loader")
        engine.verify_round3_freeze(frozen_calibration)
    else:
        raise ValueError("Selected-image extraction needs an explicit scientific scope")
    loaded = SESSION["loader"](entry=entry, plan=SESSION["plan"], reader=SESSION["reader"],
        scope=scope, frozen_calibration=frozen_calibration)
    if not isinstance(loaded, dict) or not {"response", "receipt", "source_binding", "photo_support"} <= set(loaded):
        raise ValueError("The bound selected adapter did not return its documented exact-source payload")
    frame = loaded["response"]["frame"]
    if (frame.run_id != entry["run"] or int(frame.frame_index) != int(entry["frame_index"])
            or float(frame.elapsed_h) != float(entry["elapsed_h"])
            or loaded["source_binding"] != entry["source_binding"]):
        raise ValueError("Selected RAW/phase identity or camera time differs from the sealed manifest")
    if np.asarray(loaded["photo_support"]).shape != frame.domain.shape:
        raise ValueError("Measured photo support and exact selected phase canvas differ")
    return loaded


def _entries_for(run):
    return sorted((entry for entry in SESSION["entries"] if entry["run"] == run),
        key=lambda entry: (entry["elapsed_h"], entry["frame_index"]))


def _summary_from_loaded(loaded, entry, scope, backend):
    response, receipt = loaded["response"], loaded["receipt"]
    frame = response["frame"]
    revision = receipt["rig_domain_revision"]
    raw_geometry = receipt["full_extended_scientific"]["binding"]["geometry"]
    x, y = geometry_inputs.frame_coordinates(SESSION["matrix"], raw_geometry, revision)
    geometry = engine.build_estimate_geometry(x, y, frame.facies, SESSION["depth_points"],
        porosity_by_facies=SESSION["porosity"], swi_by_facies=SESSION["swi"],
        source_binding=dict(depth_measurements=SESSION["parts"]["depth_measurements"],
            facies_properties=SESSION["parts"]["facies_properties"],
            physical_coordinate_provenance=SESSION["parts"]["physical_coordinate_provenance"]))
    # The adapter may move the frozen pooled paths to a portable snapshot but
    # must return their source-equivalent folder as measured input, not fit one.
    path_folder = loaded.get("path_folder")
    if not path_folder:
        raise ValueError("The selected adapter must expose the unchanged frozen native path folder")
    own = engine.prepare_own_colour_response(response, path_folder, geometry,
        photo_support=loaded["photo_support"])
    row = engine.summarize_round3_response(response, geometry, own_colour=own,
        photo_support=loaded["photo_support"], no_outflow_evidence=loaded.get("no_outflow_evidence", {}),
        light_epoch=loaded.get("light_epoch"), source_binding=loaded["source_binding"], backend=backend,
        scope=scope)
    row.update(selected_target_h=float(entry["selected_target_h"]),
        frame_list_evidence=loaded.get("frame_list_evidence", {}), manifest_receipt=entry["receipt"],
        run_manifest=entry["run_manifest"], source_frame_unchanged=True,
        coordinate_validation=SESSION["coordinate_validation"])
    # Descriptive distances do not enter either estimate or the fit. The
    # source-bound per-frame adapter exposes the same pooled floor and photo
    # top as the selected-phase producer rather than a second segmentation.
    floor = loaded.get("floor_curve_canvas_y")
    domain = loaded.get("photo_sand_domain")
    if floor is not None and domain is not None:
        masks = engine._parts(frame, native._physical_partition(response)["missing"])
        for name, mask in masks.items():
            row["excluded_parts"][name]["distance_report"] = geometry_inputs.distances_for_part(mask,
                revision=revision, geometry=raw_geometry, floor_curve_canvas_y=floor, sand_domain=domain)
    else:
        row["descriptive_distance_status"] = "not available: adapter lacks bound floor/top descriptive geometry"
    return row


def _commit_summary(folder, row, campaign_sha, *, evaluation=None):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("Preserve every committed R3 selected-image extraction")
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = folder.with_name("." + folder.name + "." + uuid.uuid4().hex + ".partial")
    temporary.mkdir()
    with gzip.open(temporary / "native_statistics.json.gz", "wt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(_jsonable(row), stream, allow_nan=False, separators=(",", ":"))
    statistics_descriptor = source.asset(temporary / "native_statistics.json.gz")
    statistics_descriptor["path"] = str(folder / "native_statistics.json.gz")
    record = dict(contract=CAMPAIGN_CONTRACT, complete=True, campaign_sha256=campaign_sha,
        run=row["run"], frame_index=row["frame_index"], selected_target_h=row["selected_target_h"],
        elapsed_h=row["elapsed_h"], source_binding=row["source_binding"],
        native_statistics=statistics_descriptor, evaluation=evaluation,
        main_geometry_modified=False, main_labels_modified=False,
        committed_utc=datetime.now(timezone.utc).isoformat())
    _write_new(temporary / "receipt.json", record)
    temporary.rename(folder)
    return record


def _read_committed(folder, campaign_sha, entry=None):
    path = Path(folder) / "receipt.json"
    record = SESSION["reader"].json(source.asset(path))
    if record.get("complete") is not True or record.get("campaign_sha256") != campaign_sha:
        raise ValueError("Resume encountered a partial or differently bound extraction")
    if entry is not None and (record["source_binding"] != entry["source_binding"]
            or record["elapsed_h"] != entry["elapsed_h"]
            or record["selected_target_h"] != entry["selected_target_h"]):
        raise ValueError("Resume changed a selected source, camera time or target")
    statistics_path = SESSION["reader"].path(record["native_statistics"])
    with gzip.open(statistics_path, "rt", encoding="utf-8") as stream:
        row = json.load(stream)
    for key in ("run", "frame_index", "elapsed_h", "selected_target_h", "source_binding"):
        if row.get(key) != record[key]:
            raise ValueError("Committed statistics and source receipt have different identities")
    return row, record


def extract_calibration_run(task):
    run, job, backend = task
    if run not in engine.FIT_RUNS:
        raise ValueError("Only the 26 calibration runs can be extracted before freeze")
    count, started = 0, time.perf_counter()
    campaign_sha = SESSION["campaign_descriptor"]["sha256"]
    for entry in _entries_for(run):
        folder = Path(job) / "extraction" / run / f"frame_{entry['frame_index']:04d}"
        if (folder / "receipt.json").is_file():
            _read_committed(folder, campaign_sha, entry)
        else:
            loaded = _load_entry(entry, scope="calibration")
            row = _summary_from_loaded(loaded, entry, "calibration", backend)
            _commit_summary(folder, row, campaign_sha)
            del loaded, row
            gc.collect()
        count += 1
    return dict(run=run, frames=count, wall_seconds=time.perf_counter()-started)


def calibration_rows(job):
    cached = SESSION.get("calibration_rows_cache", {}).get(str(Path(job)))
    if cached is not None:
        # Reuse decoded statistics across folds while checking their stat/SHA
        # bindings through the once-per-asset reader. No scientific RAW reload.
        for descriptor in cached[1].values():
            SESSION["reader"].path(descriptor)
        return cached
    rows, files = [], {}
    campaign_sha = SESSION["campaign_descriptor"]["sha256"]
    expected = [entry for entry in SESSION["entries"] if entry["run"] in engine.FIT_RUNS]
    for entry in expected:
        folder = Path(job) / "extraction" / entry["run"] / f"frame_{entry['frame_index']:04d}"
        row, receipt = _read_committed(folder, campaign_sha, entry)
        rows.append(row)
        files[str(folder / "receipt.json")] = source.asset(folder / "receipt.json")
        files[receipt["native_statistics"]["path"]] = receipt["native_statistics"]
    if {row["run"] for row in rows} != set(engine.FIT_RUNS):
        raise ValueError("A complete source-bound extraction of all 26 calibration runs is required")
    SESSION.setdefault("calibration_rows_cache", {})[str(Path(job))] = (rows, files)
    return rows, files


def fit_task(task):
    job, excluded_run = task
    rows, _ = calibration_rows(job)
    return fitter.fit_round3_candidate(rows, selected_phase_gate=SESSION["gate"], excluded_run=excluded_run)


def fit_all(job, *, workers=1):
    tasks = [(str(job), None), *((str(job), run) for run in engine.FIT_RUNS)]
    files = {}
    # Stats are loaded once per fit worker, not scientific RAW. Keep this
    # concurrency independent of extraction because large pair histograms are
    # replicated in memory; a laptop can use one worker.
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
            initargs=(SESSION["campaign_descriptor"],)) as pool:
        futures = {pool.submit(fit_task, task): task[1] for task in tasks}
        for future in as_completed(futures):
            omitted = futures[future]
            result = future.result()
            path = Path(job) / "fits" / ("global.json" if omitted is None else f"leave_{omitted}_out.json")
            _write_new(path, result)
            files[str(path)] = source.asset(path)
            if result["stop_required"]:
                for pending in futures:
                    pending.cancel()
                _write_new(Path(job) / "fits" / "STOP.json", dict(stop_required=True,
                    omitted_run=omitted, reason=result["reason"], result=files[str(path)],
                    no_freeze=True, no_heldout_computation=True))
                return dict(stop_required=True, reason=result["reason"], files=files)
    _write_new(Path(job) / "fits" / "complete.json", dict(complete=True, fits=27,
        files=files, final_heldout_used=False, all_grid_audits_passed=True))
    return dict(stop_required=False, files=files)


def freeze_calibration(job, *, exact_native_audit=None):
    """Publish immutable B1 after the pre-registered global/fold grid checks.

    An additional native replay can be bound as operational evidence. It is
    neither required nor a new 26-run acceptance check before freeze.
    """
    job = Path(job)
    reader = SESSION["reader"]
    complete_descriptor = source.asset(job / "fits" / "complete.json")
    complete = reader.json(complete_descriptor)
    if (complete.get("complete") is not True or complete.get("fits") != 27
            or complete.get("all_grid_audits_passed") is not True or (job / "fits" / "STOP.json").exists()):
        raise ValueError("Every global/fold grid must pass before freezing B1")
    global_fit = reader.json(source.asset(job / "fits" / "global.json"))
    if global_fit["stop_required"] or global_fit["final_heldout_used"]:
        raise ValueError("B1 cannot be frozen from a stopped or contaminated fit")
    expected_paths = {str(job / "fits" / "global.json")} | {
        str(job / "fits" / f"leave_{run}_out.json") for run in engine.FIT_RUNS}
    if set(complete["files"]) != expected_paths:
        raise ValueError("The frozen fit inventory must contain the global fit and every omitted-run fold")
    for run in engine.FIT_RUNS:
        fold = reader.json(complete["files"][str(job / "fits" / f"leave_{run}_out.json")])
        if (fold.get("stop_required") is not False or fold.get("heldout_fit_run") != run
                or set(fold.get("calibration_runs", ())) != set(engine.FIT_RUNS)-{run}
                or fold.get("final_heldout_used") is not False):
            raise ValueError("A source-bound fold is incomplete, stopped or contains its omitted run")
    audit_descriptor = None
    if exact_native_audit is not None:
        audit_descriptor = _descriptor(exact_native_audit, reader)
        audit = reader.json(audit_descriptor)
        if (audit.get("complete") is not True or audit.get("parameters") != global_fit["parameters"]
                or audit.get("global_fit_sha256") != source.asset(job / "fits" / "global.json")["sha256"]
                or audit.get("final_heldout_used") is not False):
            raise ValueError("Optional operational replay has a different source fit or includes held-out data")
    _, files = calibration_rows(job)
    for descriptor in complete["files"].values():
        _descriptor(descriptor, reader)
        files[descriptor["path"]] = descriptor
    files[complete_descriptor["path"]] = complete_descriptor
    if audit_descriptor is not None:
        files[audit_descriptor["path"]] = audit_descriptor
    files[SESSION["campaign_descriptor"]["path"]] = SESSION["campaign_descriptor"]
    files[SESSION["campaign"]["plan"]["path"]] = SESSION["campaign"]["plan"]
    for descriptor in SESSION["campaign"]["code_binding"].values():
        files[descriptor["path"]] = descriptor
    plan = SESSION["plan"]
    for name in ("spec", "selected_phase_gate", "selected_metadata_index", "saved_parts_plan"):
        files[plan[name]["path"]] = plan[name]
    files[plan["selected_adapter"]["source"]["path"]] = plan["selected_adapter"]["source"]
    for descriptor in plan.get("additional_dependencies", {}).values():
        files[descriptor["path"]] = descriptor
    for descriptor in SESSION["gate"]["run_manifests"].values():
        files[descriptor["path"]] = descriptor
    for descriptor in plan.get("round2_loro_fits", {}).values():
        files[descriptor["path"]] = descriptor
    for descriptor in SESSION["parts"].values():
        if isinstance(descriptor, dict) and descriptor.get("path") and descriptor.get("sha256"):
            files[descriptor["path"]] = descriptor
    value = dict(contract=engine.FREEZE_CONTRACT, frozen=True,
        frozen_utc=datetime.now(timezone.utc).isoformat(), parameters=global_fit["parameters"],
        calibration_runs=list(engine.FIT_RUNS), heldout_runs=list(engine.HELDOUT_RUNS),
        descriptive_runs=list(engine.DESCRIPTIVE_RUNS), round2_freeze_sha256=engine.ROUND2_FREEZE_SHA256,
        calibration_files=files, selected_phase_gate=plan["selected_phase_gate"],
        spec_sha256=SPEC_SHA256, selected_input_plan=SESSION["campaign"]["plan"],
        global_fit=source.asset(job / "fits" / "global.json"),
        optional_operational_native_audit=audit_descriptor,
        stage2_eligibility_policy=global_fit["stage2_eligibility_policy"],
        dilution_anchor_policy="explicit selected 2.25 h target; unchanged R2 volume/time convention",
        gas_branch_fitted=False, colour_paths_refitted=False, map_mass_rescaling=False,
        overflow_in_objective=False, production_automatically_allowed=False)
    path = job / "calibration" / "freeze" / "frozen_calibration.json"
    _write_new(path, value)
    engine.verify_round3_freeze(path)
    _write_new(path.parent / "publication_receipt.json", dict(frozen_calibration=source.asset(path),
        complete=True, heldout_computation_started=False, parent_models_unchanged=True))
    return source.asset(path)


def _exact_prediction(row, response, parameters, anchor, freeze_path=None):
    result = engine.evaluate_round3_summary(row, parameters, anchor, frozen_calibration=freeze_path)
    main = native.evaluate_round2_frame(response, parameters, anchor, fields=False, allow_frozen_sensitivity=True)
    histogram_main = result["main_mass_g"]
    exact_main = main["total_mass_g"]
    result.update(histogram_main_mass_g=histogram_main, main_mass_g=exact_main, total_mass_g=exact_main,
        aqueous_mass_g=main["aqueous_mass_g"], gas_mass_g=main["gas_mass_g"],
        gas_water_mass_g=main["gas_water_mass_g"], gas_share=0. if exact_main == 0 else main["gas_mass_g"]/exact_main,
        main_aqueous_nongas_mass_g=main["aqueous_nongas_mass_g"],
        exact_native_main_replay=True, main_mass_replay_difference_g=exact_main-histogram_main,
        selected_target_h=row["selected_target_h"], scope=row["scope"])
    lower, upper = exact_main+result["E_lo_g"], None if result["U_rest_g"] is None else exact_main+result["E_hi_g"]+result["U_rest_g"]
    result.update(interval_lower_g=lower, interval_upper_g=upper)
    dose = result["injected_mass_g"]
    result["signed_interval_residual"] = None if upper is None or dose <= 0 else native.interval_residual(lower, dose, upper-lower)
    result["delta"] = None if result["signed_interval_residual"] is None else abs(result["signed_interval_residual"])
    # Full paired/native histograms remain in the compressed scientific
    # statistics asset. The human prediction receipt keeps their evaluated
    # masses and areas; duplicating the large arrays here makes NAS reporting
    # needlessly slow without adding evidence.
    def scalar_part(part):
        return {key: ({label: scalar_part(item) for label, item in value.items()} if key == "by_facies" else value)
            for key, value in part.items() if key != "own_colour_histogram"}
    result["excluded_parts"] = {name: scalar_part(part) for name, part in result.get("excluded_parts", {}).items()}
    return result


def _check_fixed_gas(predictions):
    first, second = predictions["B0"], predictions["B1"]
    for key in ("gas_mass_g", "gas_water_mass_g"):
        if np.float64(first[key]).tobytes() != np.float64(second[key]).tobytes():
            raise ValueError("B0/B1 changed the fixed gas or undiluted gas-water mass")


def evaluate_final_run(task):
    run, job, freeze_path, backend = task
    if run not in set(engine.HELDOUT_RUNS) | set(engine.DESCRIPTIVE_RUNS):
        raise ValueError("Final evaluation is the twelve held-out runs and descriptive AC38")
    freeze, freeze_sha = engine.verify_round3_freeze(freeze_path)
    entries = _entries_for(run)
    anchor_entry = next(entry for entry in entries if entry["selected_target_h"] == 2.25)
    loaded_anchor = _load_entry(anchor_entry, scope="frozen_evaluation", frozen_calibration=freeze_path)
    anchor = native.anchor_from_response(loaded_anchor["response"])
    del loaded_anchor
    gc.collect()
    campaign_sha = SESSION["campaign_descriptor"]["sha256"]
    records = []
    for entry in entries:
        folder = Path(job) / "evaluation" / run / f"frame_{entry['frame_index']:04d}"
        if (folder / "receipt.json").is_file():
            _, record = _read_committed(folder, campaign_sha, entry)
            if record["evaluation"]["freeze_sha256"] != freeze_sha:
                raise ValueError("Resume would combine two different frozen calibrations")
        else:
            loaded = _load_entry(entry, scope="frozen_evaluation", frozen_calibration=freeze_path)
            row = _summary_from_loaded(loaded, entry, "frozen_evaluation", backend)
            predictions = {"B0": _exact_prediction(row, loaded["response"], B0, anchor),
                "B1": _exact_prediction(row, loaded["response"], freeze["parameters"], anchor, freeze_path)}
            _check_fixed_gas(predictions)
            record = _commit_summary(folder, row, campaign_sha,
                evaluation=dict(freeze_sha256=freeze_sha, anchor=anchor, predictions=predictions))
            del loaded, row
            gc.collect()
        records.append(record)
    return dict(run=run, frames=len(records), freeze_sha256=freeze_sha)


def _original_fold_parameters(record, run):
    fit = record.get("fit", {})
    if (record.get("complete") is not True or record.get("candidate") != "N1"
            or record.get("leftout_run") != run or fit.get("heldout_fit_run") != run
            or set(fit.get("calibration_runs", ())) != set(engine.FIT_RUNS)-{run}
            or fit.get("final_heldout_used") is not False
            or fit.get("gas_branch_fitted") is not False):
        raise ValueError("The original N1 fold did not omit this entire fit run or changes the scientific scope")
    native._validate_parameters(fit["parameters"])
    return fit["parameters"]


def evaluate_fit_loro_run(task):
    """Replay the selected fit run with its saved R2 and new R3 omitted-run fits."""
    run, job, freeze_path = task
    if run not in engine.FIT_RUNS:
        raise ValueError("Leave-one-run-out replay requires one of the 26 calibration runs")
    _, freeze_sha = engine.verify_round3_freeze(freeze_path)
    old_descriptor = SESSION["plan"].get("round2_loro_fits", {}).get(run)
    if old_descriptor is None:
        raise ValueError("The B0 extended-time replay requires the original bound N1 fold receipt")
    old = SESSION["reader"].json(old_descriptor)
    b0 = _original_fold_parameters(old, run)
    new_descriptor = source.asset(Path(job) / "fits" / f"leave_{run}_out.json")
    new = SESSION["reader"].json(new_descriptor)
    if (new.get("stop_required") is not False or new.get("heldout_fit_run") != run
            or run in new.get("calibration_runs", ())):
        raise ValueError("R3 leave-one-run-out replay has a stopped or contaminated fit")
    b1 = new["parameters"]
    entries = _entries_for(run)
    anchor_entry = next(entry for entry in entries if entry["selected_target_h"] == 2.25)
    loaded_anchor = _load_entry(anchor_entry, scope="frozen_evaluation", frozen_calibration=freeze_path)
    anchor = native.anchor_from_response(loaded_anchor["response"])
    del loaded_anchor
    gc.collect()
    campaign_sha = SESSION["campaign_descriptor"]["sha256"]
    records = []
    for entry in entries:
        folder = Path(job) / "evaluation" / run / f"frame_{entry['frame_index']:04d}"
        if (folder / "receipt.json").is_file():
            _, record = _read_committed(folder, campaign_sha, entry)
            evaluation = record["evaluation"]
            if (evaluation["freeze_sha256"] != freeze_sha or evaluation["folds"] != dict(B0=old_descriptor, B1=new_descriptor)):
                raise ValueError("Resume changed a freeze or leave-one-run-out source fit")
        else:
            row, _ = _read_committed(Path(job) / "extraction" / run / f"frame_{entry['frame_index']:04d}", campaign_sha, entry)
            loaded = _load_entry(entry, scope="frozen_evaluation", frozen_calibration=freeze_path)
            predictions = {"B0": _exact_prediction(row, loaded["response"], b0, anchor),
                "B1": _exact_prediction(row, loaded["response"], b1, anchor)}
            _check_fixed_gas(predictions)
            for result in predictions.values():
                result.update(heldout_fit_run=run, evaluation_is_leave_one_run_out=True)
            record = _commit_summary(folder, row, campaign_sha, evaluation=dict(freeze_sha256=freeze_sha,
                anchor=anchor, folds=dict(B0=old_descriptor, B1=new_descriptor), predictions=predictions))
            del loaded, row
            gc.collect()
        records.append(record)
    return dict(run=run, frames=len(records), freeze_sha256=freeze_sha, predictions_are_leave_one_run_out=True)


def _target_row(rows, target):
    matches = [row for row in rows if float(row["selected_target_h"]) == float(target)]
    if len(matches) > 1:
        raise ValueError("A check target has duplicate predictions")
    return matches[0] if matches else None


def _late_check(rows, targets, expected_runs, required_assessable):
    per_time = []
    for target in targets:
        chosen = [_target_row([row for row in rows if row["run"] == run], target) for run in sorted(expected_runs)]
        available = [row for row in chosen if row is not None]
        eligible = [row for row in available if row.get("W_over_d") is not None and row["W_over_d"] <= .05
            and row.get("delta") is not None]
        assessable = len(eligible) >= 7
        value = statistics.median(row["delta"] for row in eligible) if assessable else None
        per_time.append(dict(target_h=target, available_run_count=len(available), eligible_run_count=len(eligible),
            eligible_runs=sorted(row["run"] for row in eligible), required_run_count=7,
            assessable=assessable, median_delta=value, pass_gate=value <= .10 if assessable else None))
    count = sum(value["assessable"] for value in per_time)
    return dict(times=per_time, assessable_time_count=count, required_assessable_times=required_assessable,
        assessable=count >= required_assessable,
        pass_gate=count >= required_assessable and all(value["pass_gate"] for value in per_time if value["assessable"]))


def score_literal_checks(heldout_rows, fit_loro_rows):
    """Literal §4 score from saved exact-main predictions; no fitting or tuning."""
    if {row["run"] for row in heldout_rows} != set(engine.HELDOUT_RUNS):
        raise ValueError("Check 1 and the 4–48 h check need exactly the twelve held-out runs")
    if {row["run"] for row in fit_loro_rows} != set(engine.FIT_RUNS):
        raise ValueError("The after-48 h pool needs all 26 source-bound leave-one-run-out runs")
    if any(row.get("heldout_fit_run") != row["run"] for row in fit_loro_rows):
        raise ValueError("A fit-run prediction in the late pool must omit its entire run")
    injection = []
    for target in (1., 2.25):
        chosen = [_target_row([row for row in heldout_rows if row["run"] == run], target) for run in engine.HELDOUT_RUNS]
        if any(row is None or not row.get("exact_native_main_replay") or row["injected_mass_g"] <= 0 for row in chosen):
            raise ValueError("Check 1 needs an exact native main-mass replay at each selected target")
        ratios = [row["main_mass_g"]/row["injected_mass_g"] for row in chosen]
        groups = {}
        for group in ("rig_light", "new_lamps", "caveat"):
            values = [ratio for row, ratio in zip(chosen, ratios) if previous.light_group(
                row["run"], final=True, supplied=row.get("light_epoch")) == group]
            if not values:
                raise ValueError("A pre-registered held-out light/chemistry group is absent")
            groups[group] = statistics.median(values)
        error = statistics.median(abs(value-1) for value in ratios)
        injection.append(dict(target_h=target, run_count=12, median_absolute_ratio_error=error,
            median_ratio_by_group=groups, pass_gate=error <= .10 and all(.9 <= value <= 1.1 for value in groups.values())))
    late = _late_check(heldout_rows, (4., 8., 12., 24., 48.), set(engine.HELDOUT_RUNS), 4)
    extended = _late_check(heldout_rows+fit_loro_rows, (56., 64., 72., 96.),
        set(engine.HELDOUT_RUNS) | set(engine.FIT_RUNS), 3)
    return dict(check1=injection, late_check=late, after48_check=extended,
        pass_gate=all(value["pass_gate"] for value in injection) and late["pass_gate"] and extended["pass_gate"],
        scope="Known R2 held-out results; fixed R3 estimate rule, no held-out fitting; fit runs predicted LORO",
        production_automatically_allowed=False)


def decision_from_checks(b0, b1):
    if b1["pass_gate"]:
        outcome = "B1_passes_requires_Olav_go_before_production"
    elif b0["pass_gate"]:
        outcome = "B0_passes_B1_fails_requires_Olav_decision"
    else:
        outcome = "stop_both_fail_or_late_checks_unassessable"
    return dict(outcome=outcome, B0=b0, B1=b1, stop_required=not b0["pass_gate"] and not b1["pass_gate"],
        production_automatically_allowed=False, no_map_mass_rescaling=True, overflow_in_objective=False)


def report_checks(job):
    """Write §4 tables and per-run interval figures from committed predictions."""
    job = Path(job)
    freeze_path = job / "calibration" / "freeze" / "frozen_calibration.json"
    _, freeze_sha = engine.verify_round3_freeze(freeze_path)
    predictions = {"B0": [], "B1": []}
    descriptors = {}
    for entry in SESSION["entries"]:
        folder = job / "evaluation" / entry["run"] / f"frame_{entry['frame_index']:04d}"
        _, record = _read_committed(folder, SESSION["campaign_descriptor"]["sha256"], entry)
        if record["evaluation"]["freeze_sha256"] != freeze_sha:
            raise ValueError("The check report combines different frozen B1 calibrations")
        descriptors[str(folder / "receipt.json")] = source.asset(folder / "receipt.json")
        for name in predictions:
            predictions[name].append(dict(record["evaluation"]["predictions"][name], variant=name))
    scores = {}
    for name, rows in predictions.items():
        scores[name] = score_literal_checks([row for row in rows if row["run"] in engine.HELDOUT_RUNS],
            [row for row in rows if row["run"] in engine.FIT_RUNS])
    decision = decision_from_checks(scores["B0"], scores["B1"])
    output = job / "report"
    if output.exists():
        raise FileExistsError("Publish the R3 report only in a new folder")
    output.mkdir()
    keys = ("variant", "run", "frame_index", "selected_target_h", "elapsed_h", "light_epoch",
        "main_mass_g", "aqueous_mass_g", "main_aqueous_nongas_mass_g", "gas_mass_g", "gas_water_mass_g",
        "gas_share", "E_lo_g", "E_hi_g", "U_rest_g", "W_g", "W_over_d", "delta",
        "interval_lower_g", "interval_upper_g", "injected_mass_g", "estimate_gas_mass_g",
        "estimate_gas_water_mass_g", "estimate_geometry_extended_area_cm2", "dilution_factor",
        "exact_native_main_replay", "histogram_main_mass_g", "main_mass_replay_difference_g",
        "summary_quantization_bound_g", "heldout_fit_run")
    with (output / "per_run_time.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(predictions["B0"]+predictions["B1"],
            key=lambda row: (row["run"], row["elapsed_h"], row["variant"])))
    # The descriptive estimate remains separate from main M in every mass
    # table. AC32/44 and AC38 get explicit flags rather than disappearing.
    part_keys = ("variant", "run", "frame_index", "selected_target_h", "elapsed_h", "part", "facies",
        "main_mass_g", "gas_mass_g", "gas_water_mass_g", "gas_share", "E_lo_g", "E_hi_g", "U_rest_g",
        "co2_area_cm2", "without_main_geometry_area_cm2", "geometry_extended_area_cm2", "capacity_g",
        "own_colour_aqueous_mass_g", "fixed_gas_mass_g", "fixed_gas_water_mass_g", "gas_rule_total_mass_g")
    with (output / "excluded_parts.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=part_keys, extrasaction="ignore")
        writer.writeheader()
        for row in predictions["B0"]+predictions["B1"]:
            base = {key: row.get(key) for key in part_keys if key in row}
            for part, values in row["excluded_parts"].items():
                writer.writerow(dict(base, **{key: value for key, value in values.items() if key != "by_facies"}, part=part, facies="all"))
                for label, values in values["by_facies"].items():
                    writer.writerow(dict(base, **values, part=part, facies=label))
    distance_keys = ("variant", "run", "frame_index", "selected_target_h", "elapsed_h", "part", "boundary",
        "pixels", "known_pixels", "min_mm", "median_mm", "p95_mm", "max_mm", "metric_source")
    with (output / "excluded_distances.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=distance_keys, extrasaction="ignore")
        writer.writeheader()
        for row in predictions["B0"]+predictions["B1"]:
            for part, values in row["excluded_parts"].items():
                for boundary, distances in values.get("distance_report", {}).items():
                    writer.writerow(dict({key: row.get(key) for key in distance_keys if key in row},
                        **distances, part=part, boundary=boundary))
    figures = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for run in sorted(ALL_RUNS):
        figure, axes = plt.subplots(figsize=(10, 4.5))
        for name, colour in (("B0", "#377eb8"), ("B1", "#d95f02")):
            rows = sorted((row for row in predictions[name] if row["run"] == run), key=lambda row: row["elapsed_h"])
            positive = [row for row in rows if row["injected_mass_g"] > 0]
            times = [row["elapsed_h"] for row in positive]
            axes.plot(times, [row["main_mass_g"]/row["injected_mass_g"] for row in positive], label=name+" main M/d", color=colour)
            lo = [row["interval_lower_g"]/row["injected_mass_g"] for row in positive]
            hi = [np.nan if row["interval_upper_g"] is None else row["interval_upper_g"]/row["injected_mass_g"] for row in positive]
            axes.fill_between(times, lo, hi, color=colour, alpha=.15, label=name+" estimate interval")
        axes.axhline(1, color="black", linewidth=.8)
        kind = "descriptive AC38" if run == "ac38" else "caveat" if run in {"ac32", "ac44"} else "LORO fit run" if run in engine.FIT_RUNS else "final held-out"
        axes.set(title=f"{run.upper()} — {kind}; estimates separate from main M", xlabel="Camera elapsed time (h)", ylabel="Mass / injected dose")
        axes.legend(fontsize=8); axes.grid(alpha=.2)
        figure.tight_layout()
        path = output / "figures" / (run + ".png")
        path.parent.mkdir(exist_ok=True)
        figure.savefig(path, dpi=150); plt.close(figure)
        figures.append(source.asset(path))
    _write_new(output / "checks.json", dict(decision, complete=True, spec_sha256=SPEC_SHA256,
        frozen_calibration=source.asset(freeze_path), prediction_receipts=descriptors, figures=figures,
        AC32_AC44_shown_separately=True, AC38_descriptive=True,
        undefined_estimate_geometry_is_unbounded_not_zero=True,
        summary_estimates_are_quantized_with_reported_error_bound=True))
    return decision


def run_pool(function, tasks, workers):
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
            initargs=(SESSION["campaign_descriptor"],)) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "extract-calibration", "fit", "freeze", "evaluate-heldout", "evaluate-loro", "checks"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--exact-native-audit")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("At least one worker is required")
    if args.stage == "init":
        if not args.plan:
            parser.error("init requires --plan with real selected adapter/manifests/gate assets")
        print(json.dumps(initialize_campaign(args.plan, args.job)))
        return
    campaign_descriptor = source.asset(Path(args.job) / "campaign.json")
    init_worker(campaign_descriptor)
    if args.stage == "extract-calibration":
        result = run_pool(extract_calibration_run,
            [(run, args.job, args.backend) for run in engine.FIT_RUNS], args.workers)
        _, files = calibration_rows(args.job)
        _write_new(Path(args.job) / "extraction" / "complete.json", dict(complete=True,
            calibration_runs=list(engine.FIT_RUNS), files=files, run_results=result,
            final_heldout_used=False, campaign_sha256=campaign_descriptor["sha256"]))
    elif args.stage == "fit":
        result = fit_all(args.job, workers=args.workers)
    elif args.stage == "freeze":
        result = freeze_calibration(args.job, exact_native_audit=
            source.asset(args.exact_native_audit) if args.exact_native_audit else None)
    elif args.stage == "evaluate-heldout":
        freeze_path = str(Path(args.job) / "calibration" / "freeze" / "frozen_calibration.json")
        engine.verify_round3_freeze(freeze_path)
        result = run_pool(evaluate_final_run, [(run, args.job, freeze_path, args.backend)
            for run in (*engine.HELDOUT_RUNS, *engine.DESCRIPTIVE_RUNS)], args.workers)
        _write_new(Path(args.job) / "evaluation" / "complete.json", dict(complete=True,
            results=result, frozen_calibration=source.asset(freeze_path),
            production_automatically_allowed=False))
    elif args.stage == "evaluate-loro":
        freeze_path = str(Path(args.job) / "calibration" / "freeze" / "frozen_calibration.json")
        engine.verify_round3_freeze(freeze_path)
        result = run_pool(evaluate_fit_loro_run, [(run, args.job, freeze_path) for run in engine.FIT_RUNS], args.workers)
        _write_new(Path(args.job) / "evaluation" / "loro_complete.json", dict(complete=True,
            results=result, frozen_calibration=source.asset(freeze_path), final_heldout_used_for_fit=False))
    else:
        result = report_checks(args.job)
    print(json.dumps(_jsonable(result), allow_nan=False))


if __name__ == "__main__":
    main()
