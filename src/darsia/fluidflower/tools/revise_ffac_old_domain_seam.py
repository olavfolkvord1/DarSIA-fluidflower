"""Isolated, resumable FF_AC old-domain seam inference and Studio packaging.

The measured parent and all scientific preparation, photographs, annotations
and review sessions remain source-owned. Only new phase maps and class masks
are produced, under a frozen worker binding in a new candidate directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import uuid

from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import (
    JobLease, atomic_json, canonical_sha256, copy_file_verified, read_json,
    sha256_file, utc_now,
)
from .run_ffac_rig_camera_candidate import artifact, admitted_workers, worker_initializer

CONTRACT = "ffac-old-domain-seam-revision-v2"
FRAME_CONTRACT = "ffac-old-domain-seam-frame-v1"
PLAN_FILE = "old_domain_seam_plan.json"
SEAL_FILE = "rig_camera_candidate_receipt.json"
QA_FILE = "production_qa/old_domain_seam_qa.json"
KIND = "approved_g3_old_domain_seam_revision"
WORKER_MODULE = "darsia.fluidflower.tools.ffac_old_domain_seam_worker"
HIDDEN = frozenset(("ac30", "ac49", "ac54"))
BOUNDARY_SCOPE = {
    "contract": "ffac-source-regional-boundary-interpolation-footprint-v1",
    "selection": "fixed source-derived exterior rejected-cell bilinear interpolation footprint",
    "inputs": "original regional coarse-grid cell geometry and coverage; unchanged source native grid",
    "labels_or_predictions_used_for_selection": False,
    "co2_outside_released_footprint_exact": True,
    "gas_scope": "components touched by the released interpolation footprint",
    "historical_one_pixel_ring_qa_unchanged": True,
}
INTERIOR_EXCEPTION = (
    "CO2 changes within the fixed source-derived regional boundary interpolation footprint; "
    "gas only in components touched by that released footprint. The mask is derived from "
    "original coarse-grid geometry and exterior rejected-cell coverage without labels or predictions."
)
WARNING = (
    "Studio review candidate: Old domain seam corrected; measured rig width; "
    "physical scale approved. Bottom strip blocked for mass; AC30/49/54 "
    "withheld pending blind check. Not approved for mass analysis or deployment."
)
INVALID = "QA FAILED: old-domain seam revision binding changed or is incomplete; do not use for mass analysis."
FILES = {
    "model_summary_sha256": "summary.json",
    "model_request_sha256": "training_request.json",
    "phase_profile_sha256": "phase_rig_camera.json",
    "native_source_manifest_sha256": "review_source/postflight_manifest.json",
    "validation_binding_sha256": "validation/validation_binding.json",
    "postflight_manifest_sha256": "validation/postflight_manifest.json",
    "production_receipt_sha256": SEAL_FILE,
    "source_plan_sha256": PLAN_FILE,
    "bottom_policy_sha256": "rig_bottom_review_policy.json",
    "studio_preparation_sha256": "studio_preparation.json",
    "qa_evidence_sha256": QA_FILE,
    "g3_model_sha256": "models/g3/gas_head.json",
    "regional_model_sha256": "regional_model.json",
    "co2_model_sha256": "models/generic/co2_presence_signal.json",
    "measurement_sha256": "physical_scale_measurement.json",
    "report_receipt_sha256": "old_domain_seam_reports/report_receipt.json",
}


def frame_key(source):
    return f"{source['run']}/frame_{source['frame_index']:03d}"


def _code_binding(worker_module=WORKER_MODULE):
    from . import run_ffac_rig_camera_inference as producer
    worker = importlib.import_module(worker_module)
    result = dict(producer._code_binding())
    result[__name__] = sha256_file(__file__)
    result[worker.__name__] = sha256_file(worker.__file__)
    for name in getattr(worker, "FROZEN_MODULES", ()):
        module = importlib.import_module(name)
        result[module.__name__] = sha256_file(module.__file__)
    for name in ("darsia.fluidflower.phase_ffac_old_domain_ring_qa",
                 "darsia.fluidflower.tools.report_ffac_old_domain_ring_qa"):
        module = importlib.import_module(name)
        result[module.__name__] = sha256_file(module.__file__)
    return result


def _extended_sources(original, producer_receipts, rig_parent):
    """Bind already-produced extended dependencies; never decode another RAW."""
    from . import run_ffac_rig_camera_inference as producer
    roots = [Path(rig_parent)]
    recovery = original.get("recovery") or {}
    old_plan_asset = recovery.get("original_inference_plan")
    if old_plan_asset:
        roots.append(bound_path(old_plan_asset).parent)
    roots = list(dict.fromkeys(roots))
    candidates, documents = {}, {}
    for root in roots:
        for path in sorted((root / "extended_scientific").glob("*/*/receipt.json")):
            task_id = path.parent.relative_to(root / "extended_scientific").as_posix()
            value = read_json(path)
            if value.get("complete") is not True:
                continue
            item = artifact(path)
            candidates.setdefault(task_id, []).append(item)
            documents[item["sha256"]] = value
    tasks = {row["task_id"]: row for row in original["upstream_plan"]["source_inventory"]["scientific_tasks"]}
    science_root = Path(original.get("scientific_source_root", original["upstream_plan"]["binding"]["candidate"]))
    prepared_assets, prepared_documents = {}, {}
    descriptors, previous = {}, {}
    for source in original["source_inventory"]:
        if source["source_changed"]:
            continue
        key = frame_key(source)
        receipt = producer_receipts[key]
        current = receipt["full_extended_scientific"]
        preferred_job = current["binding"]["inference_job_sha256"]
        for name in ("full_extended_scientific", "baseline_extended_scientific"):
            descriptor = receipt[name]
            prep_path = Path(descriptor["binding"]["prepared_source"]["path"])
            task_id = prep_path.parent.parent.name + "/" + prep_path.parent.name
            exact = [item for item in candidates.get(task_id, []) if documents[item["sha256"]] == descriptor]
            if not exact:
                raise ValueError("Producer extended scientific descriptor lacks its exact committed receipt: " + task_id)
            descriptors.setdefault(task_id, exact[0])
        task_id = source.get("previous_scientific_task_id")
        if task_id is None:
            continue
        task = tasks[task_id]
        geometry = producer.task_geometry(original, task)
        if task_id not in prepared_assets:
            prepared_assets[task_id] = artifact(science_root / "scientific_sources" / task_id / "prepared_source.json")
            prepared_documents[task_id] = read_json(bound_path(prepared_assets[task_id]))
        eligible = []
        for item in candidates.get(task_id, []):
            value = documents[item["sha256"]]
            if (value.get("source_sha256") == task["source"]["sha256"]
                    and value["binding"].get("geometry") == geometry
                    and value["binding"].get("prepared_source") == prepared_assets[task_id]
                    and value.get("native_centre") == prepared_documents[task_id]["full_native_scientific"]
                    and value["binding"].get("phase_grid") == original["binding"]["phase_grid"]):
                eligible.append(item)
        if not eligible:
            raise ValueError("Previous scientific source lacks an exact completed RAW extension: " + task_id)
        selected = next((item for item in eligible if documents[item["sha256"]]["binding"].get(
            "inference_job_sha256") == preferred_job), eligible[0])
        previous[key] = selected
        descriptors.setdefault(task_id, selected)
    return descriptors, previous


def _same_geometry(descriptor):
    """Class populations may change; measured physical geometry stays exact."""
    result = deepcopy(descriptor)
    result.pop("classes", None)
    result["area_map"] = {key: value for key, value in result["area_map"].items() if key != "path"}
    result.get("parameters", {}).pop("area_cm2_by_class", None)
    return result


def validate_plan(plan):
    binding = plan["binding"]
    if (plan.get("contract") != CONTRACT or binding.get("contract") != CONTRACT
            or plan.get("job_sha256") != canonical_sha256(binding)
            or plan.get("pipeline_sha256") != canonical_sha256(dict(binding=binding, model_identity=plan["model_identity"]))
            or plan.get("output_root") != binding.get("output_root")
            or plan.get("original_inference") != read_json(bound_path(binding["original_inference_plan"]))
            or plan.get("source_inventory") != plan["original_inference"]["source_inventory"]
            or binding.get("released_boundary_scope") != BOUNDARY_SCOPE
            or binding.get("interior_exception") != INTERIOR_EXCEPTION):
        raise ValueError("Seam immutable plan identity or original inference binding changed")
    for name in ("measured_parent_frames", "source_producer_frames", "native_sources",
                 "extended_scientific_sources", "previous_extended_scientific_sources"):
        if plan.get(name) != binding.get(name):
            raise ValueError("Seam immutable source registry changed: " + name)
    return plan


def validate_frame(root, plan, source, result):
    """Validate a committed frame before importing it into new metadata."""
    root = Path(root).resolve()
    if (result.get("run"), result.get("frame_index")) != (source["run"], source["frame_index"]):
        raise ValueError("Seam worker returned another frame")
    receipt_path = bound_path(result["receipt"])
    expected = root / "rig_camera_frames" / source["run"] / f"frame_{source['frame_index']:03d}.json"
    if receipt_path != expected:
        raise ValueError("Seam frame receipt escaped its new candidate")
    receipt = read_json(receipt_path)
    binding = dict(pipeline_sha256=plan["pipeline_sha256"], source=source,
        upstream_preparation_sha256=plan["binding"]["scientific_preparation"]["sha256"])
    key = frame_key(source)
    old = read_json(bound_path(plan["measured_parent_frames"][key]))
    if (receipt.get("contract") != FRAME_CONTRACT or receipt.get("complete") is not True
            or receipt.get("binding") != binding or receipt.get("qa", {}).get("all_pass") is not True
            or receipt.get("source_measured_receipt") != plan["measured_parent_frames"][key]
            or receipt.get("source_producer_receipt") != plan["source_producer_frames"][key]
            or receipt.get("training") is not False or receipt.get("labels_changed") is not False
            or receipt.get("mass_eligible") is not False or receipt.get("bottom_mass_allowed") is not False
            or receipt.get("physical_scale_approved") is not True
            or _same_geometry(receipt["rig_domain_revision"]) != _same_geometry(old["rig_domain_revision"])):
        raise ValueError("Seam frame changed its exact source, measured scale or declared policy")
    qa = receipt["qa"]
    if (qa.get("run"), qa.get("frame_index"), qa.get("source_sha256"), qa.get("elapsed_h")) != (
            source["run"], source["frame_index"], source["source_sha256"], source["elapsed_h"]):
        raise ValueError("Seam QA changed its source or camera time")
    if set(receipt["maps"]) != {"combined", "aqueous", "gas", "mixed"}:
        raise ValueError("Seam frame does not supply all four saved views")
    for item in [*receipt["maps"].values(), receipt["labels_map"], receipt["rig_domain_revision"]["classes"]]:
        if not bound_path(item).is_relative_to(root):
            raise ValueError("New seam map/class artifact escaped its candidate")
    bound_path(receipt["rig_domain_revision"]["area_map"])
    expected_qa = dict(map_path=receipt["labels_map"]["path"], map_sha256=receipt["labels_map"]["sha256"],
        studio_map_path=receipt["maps"]["combined"]["path"], studio_map_sha256=receipt["maps"]["combined"]["sha256"],
        area_map_sha256=receipt["rig_domain_revision"]["area_map"]["sha256"],
        classes_sha256=receipt["rig_domain_revision"]["classes"]["sha256"])
    if any(qa.get(name) != value for name, value in expected_qa.items()):
        raise ValueError("Seam QA differs from the persisted map/classes/measured-area assets")
    audit = receipt.get("seam_reassembly") or {}
    if source["source_changed"]:
        if audit.get("inference_performed") is not False or receipt.get("seam_masks") is not None:
            raise ValueError("New camera acquisition incorrectly claimed old-domain ring inference")
    elif (audit.get("inference_performed") is not True
            or any(audit.get(name) is not True for name in (
                "co2_outside_ring_exact", "categories_outside_ring_and_affected_gas_exact", "unresolved_outside_ring_exact"))
            or receipt.get("seam_masks") is None
            or not bound_path(receipt["seam_masks"]).is_relative_to(root)):
        raise ValueError("Seam frame lacks proof of its declared CO2/gas scope")
    if source["source_changed"] and (
            receipt["labels_map"]["sha256"] != old["labels_map"]["sha256"]
            or {key: value["sha256"] for key, value in receipt["maps"].items()}
            != {key: value["sha256"] for key, value in old["maps"].items()}
            or receipt["rig_domain_revision"]["classes"]["sha256"] != old["rig_domain_revision"]["classes"]["sha256"]):
        raise ValueError("Source-changed acquisition was modified by an inapplicable old-domain correction")
    return receipt


def create_revision_plan(parent, output, *, resume=False, worker_module=WORKER_MODULE, worker_function="infer_frame"):
    """Freeze the measured lineage and current numerical code before workers."""
    from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from .register_track1_studio_candidate import registration_path, registration_warning
    from . import run_ffac_rig_camera_inference as producer
    parent, root = Path(parent).resolve(), Path(output).resolve()
    if root == parent or root.is_relative_to(parent):
        raise ValueError("Old-domain revisions require a separate new candidate namespace")
    reg_path = registration_path(parent.name)
    registration = read_json(reg_path)
    if (registration.get("registration_kind") != "approved_g3_area_revision"
            or "QA FAILED" in registration_warning(parent)):
        raise ValueError("Old-domain seam revision requires its intact registered measured parent")
    parent_seal_path = parent / "physical_scale_receipt.json"
    seal = read_json(parent_seal_path)
    if (seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43, 864)
            or seal.get("physical_scale_approved") is not True or seal.get("mass_eligible") is not False
            or seal.get("bottom_mass_allowed") is not False or set(seal.get("blind_hold_runs", [])) != HIDDEN):
        raise ValueError("Measured parent is incomplete or changes review/mass exclusions")
    rig_parent = bound_path(seal["parent_seal"]).parent
    original_plan_asset = artifact(rig_parent / producer.PLAN_FILE)
    original = read_json(bound_path(original_plan_asset))
    bound_path(original["binding"]["scientific_preparation"])
    sources = original["source_inventory"]
    source_keys = {frame_key(row) for row in sources}
    measured = {frame_key(row): row["receipt"] for row in seal["frames"]}
    if len(sources) != 864 or len(source_keys) != 864 or set(measured) != source_keys:
        raise ValueError("Measured parent and frozen inference source populations differ")
    producers, producer_receipts = {}, {}
    for source in sources:
        key = frame_key(source)
        row = read_json(bound_path(measured[key]))
        if (row["source_sha256"], row["elapsed_h"]) != (source["source_sha256"], source["elapsed_h"]):
            raise ValueError("Measured frame differs from its frozen camera source/time")
        producers[key] = row["source_producer_receipt"]
        producer_receipts[key] = read_json(bound_path(producers[key]))
    extended, previous_extended = _extended_sources(original, producer_receipts, rig_parent)
    jobs = _source_jobs(parent / "validation")
    native_sources = {}
    for job in jobs:
        run = job["run"]
        native_path = Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        native = read_json(native_path)
        native_sources[run] = dict(native_manifest=artifact(native_path),
            input_manifest=native[meta.SOURCE_KEY]["input_manifest"])
        bound_path(native_sources[run]["input_manifest"])
    if len(native_sources) != 43 or set(native_sources) != {row["run"] for row in sources}:
        raise ValueError("Measured parent lacks all source-bound native runs")
    binding = dict(contract=CONTRACT, parent_root=str(parent), parent_model_identity=list(_model_identity(parent)),
        parent_registration=artifact(reg_path), parent_seal=artifact(parent_seal_path),
        original_inference_plan=original_plan_asset, scientific_preparation=original["binding"]["scientific_preparation"],
        measurement=seal["measurement"], measured_parent_frames=measured, source_producer_frames=producers,
        extended_scientific_sources=extended, previous_extended_scientific_sources=previous_extended,
        native_sources=native_sources, code=_code_binding(worker_module),
        worker=dict(module=worker_module, function=worker_function), output_root=str(root),
        cpu_only=True, threads_per_worker=1, maximum_workers=24, training=False,
        labels_changed=False, scientific_sources_changed=False, area_scale_applied=False,
        physical_scale_approved=True, mass_eligible=False, bottom_mass_allowed=False,
        blind_hold_runs=sorted(HIDDEN), automatic_activation=False,
        released_boundary_scope=deepcopy(BOUNDARY_SCOPE), interior_exception=INTERIOR_EXCEPTION)
    plan_path = root / PLAN_FILE
    if plan_path.exists():
        previous = read_json(plan_path)
        if (not resume or previous.get("binding") != binding
                or previous.get("job_sha256") != canonical_sha256(binding)
                or previous.get("model_identity") != list(_model_identity(root))):
            raise ValueError("Seam resume differs from frozen sources, scale, models or code")
        return validate_plan(previous)
    if root.exists() or resume:
        raise ValueError("Use a new absent candidate, or --resume with its immutable plan")
    root.mkdir(parents=True, exist_ok=False)
    producer._copy_heads_and_request(parent, root)
    copy_file_verified(bound_path(seal["measurement"]), root / "physical_scale_measurement.json")
    vertical = parent / "vertical_scale_check.json"
    if vertical.exists():
        copy_file_verified(vertical, root / vertical.name)
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=utc_now(),
        display_name="FF_AC · G3 gas · Rig domain · Camera clock · Measured width 900 mm · Old domain seam corrected",
        inherited_seam_parent_id=parent.name,
        old_domain_seam_revision=True, physical_scale_approved=True,
        training_performed_in_this_revision=False, optical_training_performed=False,
        regional_training_performed=False, inference_only_rerun=True,
        inference_performed_in_this_revision=True, raw_inference_performed=False,
        area_only_revision=False, area_scale_applied_in_this_revision=False,
        mass_eligible=False, scientific_acceptance=False, automatic_activation=False)
    atomic_json(root / "summary.json", summary)
    policy = read_json(parent / "rig_bottom_review_policy.json")
    policy.update(candidate_id=root.name)
    atomic_json(root / "rig_bottom_review_policy.json", policy)
    identity = list(_model_identity(root))
    plan = dict(contract=CONTRACT, created_utc=summary["created_utc"], binding=binding,
        job_sha256=canonical_sha256(binding), model_identity=identity,
        pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)), output_root=str(root),
        original_inference=original, measured_parent_frames=measured, source_producer_frames=producers,
        extended_scientific_sources=extended, previous_extended_scientific_sources=previous_extended,
        source_inventory=sources, native_sources=native_sources)
    atomic_json(plan_path, plan)
    return plan


def assemble_source_run(root, plan, run, completed):
    """Rebuild fresh input/native manifests with unchanged camera/source records."""
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    root = Path(root)
    original = plan["native_sources"][run]
    native = read_json(bound_path(original["native_manifest"]))
    inputs = read_json(bound_path(original["input_manifest"]))
    sources = [row for row in plan["source_inventory"] if row["run"] == run]
    results = sorted(completed, key=lambda row: row["frame_index"])
    if len(sources) != len(results) or len(results) != len(inputs["frames"]):
        raise ValueError("Cannot assemble an incomplete/overlapping seam source run")
    rows = []
    for source, result, record, binding in zip(sources, results, native["frame_sources"], inputs["frames"]):
        receipt = validate_frame(root, plan, source, result)
        if result["frame_index"] != record["frame_index"] or source["source_sha256"] != record["source_sha256"]:
            raise ValueError("Seam assembly changed its exact source sequence")
        record["rig_domain_revision"] = deepcopy(receipt["rig_domain_revision"])
        binding["rig_domain_revision"] = deepcopy(receipt["rig_domain_revision"])
        rows.append(deepcopy(receipt["qa"]))
    input_path = root / "source_alias/source_bindings" / (run + ".json")
    atomic_json(input_path, inputs)
    rebuilt = meta.build_native_metadata(read_json(bound_path(inputs["parent_native_manifest"])),
        read_json(bound_path(inputs["camera_timeline_run"])), native["frame_sources"], inputs["frames"], inputs["reference"],
        input_manifest=artifact(input_path))
    native_path = root / "source_alias/native_frames" / run / "manifest.json"
    atomic_json(native_path, rebuilt)
    for row in rows:
        row["native_manifest_sha256"] = sha256_file(native_path)
    result = dict(run=run, complete=True, native_manifest=artifact(native_path), input_manifest=artifact(input_path),
        frames=inputs["frames"], reference=inputs["reference"], frame_count=len(results),
        receipts=results, qa_rows=rows, pipeline_sha256=plan["pipeline_sha256"])
    atomic_json(root / "old_domain_seam_runs" / (run + ".json"), result)
    return result


def infer_run(root, plan, sources):
    worker = importlib.import_module(plan["binding"]["worker"]["module"])
    infer = getattr(worker, plan["binding"]["worker"]["function"])
    completed = []
    for source in sources:
        result = infer(root, plan, source)
        validate_frame(root, plan, source, result)
        completed.append(result)
        print(json.dumps(dict(event="old_domain_seam_frame_complete", run=source["run"],
            frame_index=source["frame_index"], completed_frames=len(completed), total_frames=len(sources))), flush=True)
    return assemble_source_run(root, plan, sources[0]["run"], completed)


def seal_candidate(root, plan, source_runs):
    """Assemble review packages and seal only the complete 43-run population."""
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from . import run_ffac_rig_camera_inference as producer
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    root = Path(root)
    validate_plan(plan)
    if plan["binding"]["code"] != _code_binding(plan["binding"]["worker"]["module"]):
        raise ValueError("Seam numerical/packaging code changed before sealing")
    meta.validate_campaign_bindings(source_runs)
    source_runs = sorted(source_runs, key=lambda row: row["run"])
    parent = Path(plan["binding"]["parent_root"])
    profile = read_json(parent / "phase_rig_camera.json")
    runtime = read_json(bound_path(profile["rig_domain_revision"]["manifest"]))
    runtime.update(sources=[{key: value for key, value in row.items() if key not in ("receipts", "qa_rows")}
                            for row in source_runs], old_domain_seam_revision=artifact(root / PLAN_FILE),
                   physical_scale_approved=True, area_scale_applied_in_this_revision=False)
    runtime_path = root / "rig_camera_revision/runtime_manifest.json"
    atomic_json(runtime_path, runtime)
    profile["rig_domain_revision"]["manifest"] = artifact(runtime_path)
    profile["physical_scale_approved"] = True
    profile["old_domain_seam_revision"] = artifact(root / PLAN_FILE)
    atomic_json(root / "phase_rig_camera.json", profile)
    packaging = dict(plan, upstream_plan=dict(binding=dict(parent=str(parent))))
    packages = producer.bind_rig_packages(root, packaging, source_runs)
    assemble_saved_candidate(root, packages, plan)
    rows = [row for run in source_runs for row in run["qa_rows"]]
    frames = [row for run in source_runs for row in run["receipts"]]
    if ({frame_key(row) for row in frames} != {frame_key(row) for row in plan["source_inventory"]}
            or len(frames) != 864):
        raise ValueError("Cannot seal partial/overlapping seam receipts")
    original_seal = read_json(bound_path(plan["binding"]["parent_seal"]))
    old_qa = read_json(bound_path(original_seal["qa"]))
    header = {key: value for key, value in old_qa.items() if key != "rows"}
    header.update(contract=CONTRACT, model_identity=plan["model_identity"], all_pass=True,
        run_count=43, frame_count=864, physical_scale_approved=True, area_scale_applied=False,
        mass_eligible=False, bottom_mass_allowed=False, labels_changed=False,
        scientific_sources_changed=False, released_boundary_scope=plan["binding"]["released_boundary_scope"],
        interior_exception=plan["binding"]["interior_exception"])
    atomic_json(root / QA_FILE, dict(header, rows=rows))
    reports = write_report_bundle(root, plan, rows)
    policy = read_json(root / "rig_bottom_review_policy.json")
    seal = dict(contract=CONTRACT, complete=True, created_utc=plan["created_utc"], model_identity=plan["model_identity"],
        pipeline_sha256=plan["pipeline_sha256"], run_count=43, frame_count=864, frames=frames,
        qa=artifact(root / QA_FILE), old_domain_seam_plan=artifact(root / PLAN_FILE),
        source_plan_sha256=policy["source_plan"]["sha256"], parent_seal=plan["binding"]["parent_seal"],
        parent_registration=plan["binding"]["parent_registration"], measurement=plan["binding"]["measurement"],
        scientific_preparation=plan["binding"]["scientific_preparation"], physical_scale_approved=True,
        area_scale_applied=False, inference_performed=True, training=False, labels_changed=False,
        scientific_sources_changed=False, bottom_mass_allowed=False, mass_eligible=False,
        blind_hold_runs=sorted(HIDDEN), automatic_activation=False,
        scientific_acceptance=False, released_boundary_scope=plan["binding"]["released_boundary_scope"],
        interior_exception=plan["binding"]["interior_exception"])
    seal["reports"] = reports
    atomic_json(root / SEAL_FILE, seal)
    return seal


def write_report_bundle(root, plan, rows):
    """Stage the read-only reporter locally, then seal its exact output assets."""
    from .report_ffac_old_domain_ring_qa import write_reports
    root = Path(root)
    values = [row["old_domain_ring_revision"] for row in rows]
    assets = {}
    with tempfile.TemporaryDirectory(prefix="ffac-old-domain-ring-report-") as temporary:
        paths = write_reports(values, Path(temporary) / "report")
        for visibility, report in paths.items():
            outputs = {}
            for format_name in ("json", "csv"):
                source = Path(report[format_name + "_path"])
                destination = root / "old_domain_seam_reports" / source.name
                copy_file_verified(source, destination)
                outputs[format_name] = artifact(destination)
            assets[visibility] = dict(outputs, frame_count=report["frame_count"])
    receipt = dict(contract=CONTRACT, pipeline_sha256=plan["pipeline_sha256"], reports=assets,
        all_frame_change_counts_reported=True, area_scale_applied=False,
        public_excluded_blind_runs=sorted(HIDDEN), mass_eligible=False, bottom_mass_allowed=False)
    path = root / "old_domain_seam_reports/report_receipt.json"
    atomic_json(path, receipt)
    return artifact(path)


def _stop_failed_pool(pool, futures, *, timeout_seconds=5.):
    """Cancel this job and bound shutdown to this executor's owned workers."""
    deadline = time.monotonic() + max(0., float(timeout_seconds))
    # Python3.13 has no terminate_workers(). Snapshot only this explicitly
    # constructed executor's multiprocessing.Process objects before shutdown
    # clears the registry; never enumerate system processes or other pools.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    for future in futures:
        future.cancel()
    terminated = []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
                terminated.append(process.pid)
        except (OSError, ValueError, AssertionError):
            pass
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.join(timeout=max(0., deadline - time.monotonic()))
            if process.is_alive():
                kill = getattr(process, "kill", process.terminate)
                kill()
                process.join(timeout=max(0., deadline - time.monotonic()))
        except (OSError, ValueError, AssertionError):
            pass
    return dict(executor_worker_pids=[process.pid for process in processes],
                termination_requested_pids=terminated, system_process_enumeration=False)


def _record_inference_failure(root, plan, error, *, run, source_runs):
    """Commit fatal telemetry before the process pool begins draining/stopping."""
    root = Path(root)
    destination = root / "inference_failure.json"
    if destination.exists():
        destination = root / ("inference_failure-" + uuid.uuid4().hex + ".json")
    value = dict(contract="ffac-old-domain-seam-inference-failure-v2", complete=False,
        failed_utc=utc_now(), run=run, error_type=type(error).__name__, error=repr(error),
        pipeline_sha256=plan["pipeline_sha256"], job_sha256=plan["job_sha256"],
        inference_plan=artifact(root / PLAN_FILE), completed_runs=len(source_runs),
        completed_frames_in_returned_runs=sum(row["frame_count"] for row in source_runs),
        committed_frame_files_observed=len(list((root / "rig_camera_frames").glob("*/frame_*.json"))),
        registration_performed=False, automatic_retry=False, cpu_only=True,
        mass_eligible=False, bottom_mass_allowed=False,
        stopping_policy="cancel pending futures and terminate only this executor's owned workers")
    atomic_json(destination, value)
    print(json.dumps(dict(event="old_domain_seam_inference_failed", run=run, error=repr(error),
                          failure_receipt=str(destination))), flush=True)
    return destination


def run_revision(plan, *, workers=24, register_candidate=True, results=None):
    validate_plan(plan)
    if plan["binding"]["code"] != _code_binding(plan["binding"]["worker"]["module"]):
        raise ValueError("Seam numerical/packaging code changed after planning")
    root = Path(plan["output_root"])
    groups = {run: [row for row in plan["source_inventory"] if row["run"] == run]
              for run in sorted(plan["native_sources"])}
    workers = admitted_workers(workers)
    source_runs = []
    with JobLease(root, plan["job_sha256"]):
        pool = ProcessPoolExecutor(max_workers=min(workers, len(groups)), mp_context=multiprocessing.get_context("spawn"),
                                   initializer=worker_initializer)
        futures, current_run = {}, None
        try:
            futures = {pool.submit(infer_run, str(root), plan, sources): run for run, sources in groups.items()}
            for future in as_completed(futures):
                current_run = futures[future]
                source_runs.append(future.result())
        except BaseException as error:
            try:
                _record_inference_failure(root, plan, error, run=current_run, source_runs=source_runs)
            except BaseException as telemetry_error:
                error.add_note("Failure receipt could not be committed: " + repr(telemetry_error))
            finally:
                try:
                    _stop_failed_pool(pool, futures)
                except BaseException as shutdown_error:
                    error.add_note("Owned-worker shutdown could not finish: " + repr(shutdown_error))
            raise
        else:
            pool.shutdown(wait=True)
        seal = seal_candidate(root, plan, source_runs)
        if register_candidate:
            register(root, results=results)
        return seal


def inspect_candidate(root):
    """Verify all declared phase assets and source lineage before registration."""
    from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
    from darsia.fluidflower.phase_evidence_audit import verified_review_binding
    from darsia.fluidflower.phase_g3_production import G3_SHA256, REFIT_SHA256, CO2_SHA256
    from .register_track1_studio_candidate import FORMAT, registration_warning
    root = Path(root).resolve()
    plan, seal = read_json(root / PLAN_FILE), read_json(root / SEAL_FILE)
    validate_plan(plan)
    identity = list(_model_identity(root))
    if (seal.get("contract") != CONTRACT or seal.get("complete") is not True
            or seal.get("model_identity") != identity or plan.get("model_identity") != identity
            or seal.get("old_domain_seam_plan") != artifact(root / PLAN_FILE)
            or (seal.get("run_count"), seal.get("frame_count")) != (43, 864)
            or any(seal.get(key) is not False for key in ("mass_eligible", "bottom_mass_allowed", "training",
                "labels_changed", "automatic_activation", "scientific_sources_changed", "area_scale_applied"))
            or seal.get("physical_scale_approved") is not True or set(seal.get("blind_hold_runs", [])) != HIDDEN):
        raise ValueError("Seam revision seal/identity/scope changed")
    bound_path(seal["parent_registration"]); bound_path(seal["parent_seal"])
    bound_path(seal["scientific_preparation"]); bound_path(seal["measurement"])
    if "QA FAILED" in registration_warning(Path(plan["binding"]["parent_root"])):
        raise ValueError("Seam parent registration changed")
    for name, expected in (("models/g3/gas_head.json", G3_SHA256), ("regional_model.json", REFIT_SHA256),
                           ("models/generic/co2_presence_signal.json", CO2_SHA256)):
        if sha256_file(root / name) != expected:
            raise ValueError("Seam revision changed a frozen head")
    verified_review_binding(root, saved_comparison=True)
    jobs = _source_jobs(root / "validation")
    if len(jobs) != 43 or len(seal["frames"]) != 864:
        raise ValueError("Seam revision coverage is incomplete")
    preparation = read_json(root / "studio_preparation.json")
    if (preparation.get("status") != "ready" or preparation.get("runs") != 43
            or preparation.get("runs_done") != 43 or preparation.get("partial") is not False
            or preparation.get("presentation_ready") is not True
            or preparation.get("default_activation_performed") is not False):
        raise ValueError("Seam Studio prepared set is incomplete or changes the default")
    from darsia.fluidflower.phase_ffac_rig_camera_runtime import load_native_source
    native_shas = {}
    for job in jobs:
        run = job["run"]
        if Path(job["review_output_dir"]).resolve() != root / "source_alias":
            raise ValueError("Seam review job escaped its source metadata namespace")
        state = load_native_source(job["review_output_dir"], run, verify_geometry=False)
        old_native = read_json(bound_path(plan["native_sources"][run]["native_manifest"]))
        old_inputs = read_json(bound_path(plan["native_sources"][run]["input_manifest"]))
        if state["native"]["reference"] != old_native["reference"] or state["inputs"]["reference"] != old_inputs["reference"]:
            raise ValueError("Seam revision changed a read-only pre-injection reference")
        for before, after in zip(old_native["frame_sources"], state["native"]["frame_sources"]):
            if ({key: value for key, value in before.items() if key != "rig_domain_revision"}
                    != {key: value for key, value in after.items() if key != "rig_domain_revision"}):
                raise ValueError("Seam revision changed source images, scientific arrays, geometry, labels or times")
        for before, after in zip(old_inputs["frames"], state["inputs"]["frames"]):
            if ({key: value for key, value in before.items() if key != "rig_domain_revision"}
                    != {key: value for key, value in after.items() if key != "rig_domain_revision"}):
                raise ValueError("Seam revision changed source-bound input provenance")
        native_shas[run] = sha256_file(state["native_path"])
        if native_shas[run] != job["native_manifest_sha256"]:
            raise ValueError("Seam source job native manifest binding changed")
    qa = read_json(bound_path(seal["qa"]))
    if (qa.get("all_pass") is not True or len(qa.get("rows", [])) != 864
            or len({frame_key(row) for row in qa["rows"]}) != 864):
        raise ValueError("Seam revision QA coverage is incomplete")
    report_receipt = read_json(bound_path(seal["reports"]))
    for visibility in ("internal", "public"):
        for format_name in ("json", "csv"):
            bound_path(report_receipt["reports"][visibility][format_name])
    source_lookup = {frame_key(row): row for row in plan["source_inventory"]}
    frame_lookup = {frame_key(row): row for row in seal["frames"]}
    if len(frame_lookup) != 864 or set(frame_lookup) != set(source_lookup):
        raise ValueError("Seam frames overlap or lose exact source coverage")
    qa_lookup = {frame_key(row): row for row in qa["rows"]}
    for key, frame in frame_lookup.items():
        receipt = validate_frame(root, plan, source_lookup[key], frame)
        row = qa_lookup[key]
        native_sha = row["native_manifest_sha256"]
        if native_sha != native_shas[row["run"]] or row != dict(receipt["qa"], native_manifest_sha256=native_sha):
            raise ValueError("Seam campaign QA differs from its committed frame QA")
    policy = read_json(root / "rig_bottom_review_policy.json")
    original_policy = read_json(Path(plan["binding"]["parent_root"]) / "rig_bottom_review_policy.json")
    expected_policy = dict(original_policy, candidate_id=root.name)
    if policy != expected_policy or sha256_file(root / "physical_scale_measurement.json") != seal["measurement"]["sha256"]:
        raise ValueError("Seam revision changed blind/mass exclusions or the measured scale")
    return dict(format=FORMAT, registration_kind=KIND, created_utc=plan["created_utc"],
        model_root=str(root), model_id=identity[0], model_sha256=identity[1], series="ffac",
        publication_scope="studio_validation_only", qa_status="passed_old_domain_seam_revision", qa_warning=WARNING,
        run_count=43, frame_count=864, physical_scale_approved=True, bottom_mass_allowed=False,
        map_sha256={f"{row['run']}/{row['frame_index']:03d}": row["studio_map_sha256"] for row in qa["rows"]},
        native_run_manifest_sha256={job["run"]: job["native_manifest_sha256"] for job in jobs},
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False, automatic_activation=False,
        labels_changed=False, optical_training_performed=False,
        **{key: sha256_file(root / name) for key, name in FILES.items()})


def registered_warning(root, value):
    try:
        from darsia.fluidflower.phase_validation import _model_identity
        from .register_track1_studio_candidate import FORMAT
        root = Path(root)
        if (value.get("format") != FORMAT or value.get("registration_kind") != KIND
                or [value.get("model_id"), value.get("model_sha256")] != list(_model_identity(root))
                or (value.get("run_count"), value.get("frame_count")) != (43, 864)
                or len(value.get("map_sha256", {})) != 864 or len(value.get("native_run_manifest_sha256", {})) != 43
                or value.get("physical_scale_approved") is not True or value.get("bottom_mass_allowed") is not False
                or any(value.get(key) is not False for key in (
                    "mass_eligible", "deployment_allowed", "scientific_acceptance", "automatic_activation"))):
            raise ValueError("Seam registration changed")
        for key, name in FILES.items():
            if sha256_file(root / name) != value.get(key):
                raise ValueError("Seam registered metadata changed")
        seal = read_json(root / SEAL_FILE)
        for key in ("parent_registration", "parent_seal", "scientific_preparation", "measurement", "qa", "old_domain_seam_plan", "reports"):
            bound_path(seal[key])
        return WARNING
    except (OSError, ValueError, KeyError, TypeError):
        return INVALID


def register(root, *, results=None):
    from .register_track1_studio_candidate import registration_path
    value = inspect_candidate(root)
    destination = registration_path(value["model_id"], results=results)
    if destination.exists() and read_json(destination) != value:
        raise FileExistsError(destination)
    atomic_json(destination, value)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args(argv)
    plan = create_revision_plan(args.parent, args.output, resume=args.resume)
    if not args.plan_only:
        run_revision(plan, workers=args.workers, register_candidate=not args.no_register)


if __name__ == "__main__":
    main()
