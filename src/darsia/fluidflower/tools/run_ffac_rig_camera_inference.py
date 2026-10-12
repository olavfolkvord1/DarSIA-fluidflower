"""Accepted-geometry inference and saved Studio packaging from frozen RAW prep.

This consumes the immutable scientific preparation as an upstream artifact.
It never changes that plan, its code snapshot, parent maps or annotations.
Every source, geometry and retained categorical cache is declared before model
work; incomplete geometry is rejected before any worker is submitted.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import io
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower import phase_ffac_rig_camera_metadata as metadata
from darsia.fluidflower.phase_ffac_rig_camera_runtime import SPEC_SHA256, bound_path
from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
from .build_ffac_g3_approved_candidate import _write_bytes, assemble_saved_candidate
from .refit_ffac_g3_downstream import (
    JobLease, ProgressLog, atomic_json, below_normal_priority, canonical_sha256,
    copy_file_verified, preflight_space, read_json, sha256_file,
    utc_now,
)
from .run_ffac_rig_camera_candidate import (
    artifact, admitted_workers, load_prepared_plan, require_inference_geometry,
    verify_prepared_source, worker_initializer, _npy,
)

CONTRACT = "ffac-rig-camera-inference-producer-v1"
FRAME_CONTRACT = "ffac-rig-camera-inference-frame-v1"
PLAN_FILE = "rig_camera_inference_plan.json"
GEOMETRY_CONTRACT = "ffac-rig-camera-certified-geometry-v1"
GIB = 1024 ** 3
_EXTENDED_CACHE=OrderedDict()
_SCIENTIFIC_CACHE=OrderedDict()
_PREPARED_RECEIPTS={}
_PARENT_PHOTO_CONTEXT={}
HIDDEN_CHART_NAMES=("left_panel","right_panel","connecting_housing")


def _code_binding():
    from darsia.fluidflower import (
        phase_rig_domain_inference, phase_ffac_rig_geometry,
        phase_retained_source_inference, phase_gas_barrier_rule,
        phase_ffac_rig_raw_extension, phase_ffac_barrier_extension,
        phase_pixel_area,phase_optical_regions,phase_gas_regional_context,
        phase_gas_flatfield,phase_features,phase_learning,phase_ffac_rig_raw_source,
        phase_ffac_rig_raw_dispatch,phase_ffac_rig_raw_dispatch_binding,
        phase_ffac_rig_edge_qa,
    )
    from . import phase_inference_benchmark_backends,ffac_v5_temporal_alignment,ffac_v5_observed_sand,ffac_v5_old_field_native,ffac_raw_geometry,ffac_raw_native_ea
    result={module.__name__: sha256_file(module.__file__) for module in (
        phase_rig_domain_inference, phase_ffac_rig_geometry,
        phase_retained_source_inference, phase_gas_barrier_rule,
        phase_ffac_rig_raw_extension, phase_ffac_barrier_extension,metadata,
        phase_pixel_area,phase_optical_regions,phase_gas_regional_context,
        phase_gas_flatfield,phase_features,phase_learning,phase_ffac_rig_raw_source,
        phase_ffac_rig_raw_dispatch,phase_ffac_rig_raw_dispatch_binding,
        phase_ffac_rig_edge_qa,
        phase_inference_benchmark_backends,ffac_v5_temporal_alignment,ffac_v5_observed_sand,
        ffac_v5_old_field_native,ffac_raw_geometry,ffac_raw_native_ea)} | {
        "driver": sha256_file(__file__)}
    tools=Path(phase_inference_benchmark_backends.__file__).parent
    for filename in ("ffac_track1_v4r5_source_processor.py","ffac_track1_v4r5_ac44_alt_processor.py","ffac_raw_native_ea_ac44_alt.py"):
        result["source_variant:"+filename]=sha256_file(tools/filename)
    return result


def validate_geometry_inputs(gate, plan):
    """Require full-canvas top, own reference, fixture and Rule B proofs."""
    geometry = gate["geometry"]
    if geometry.get("contract") != GEOMETRY_CONTRACT or geometry.get("spec_sha256") != SPEC_SHA256:
        raise ValueError("Inference requires the explicit source-bound certified rig geometry contract")
    frames = {(row["run"], row["frame_index"]): row for row in geometry["frames"]}
    grid=metadata.phase_grid(geometry.get("phase_shape_hw"))
    if geometry.get("phase_shape_hw") != grid["shape_hw"]:
        raise ValueError("Certified geometry must explicitly declare its complete phase_shape_hw")
    references = {row["run"]: row for row in geometry.get("references", [])}
    expected_refs = {row["run"]: row for row in plan["source_inventory"]["references"]}
    if len(references) != 43 or len(geometry.get("references", [])) != 43 or set(references) != set(expected_refs):
        raise ValueError("Rig geometry must certify all43 own frozen pre-injection references")
    tasks = {row["task_id"]: row for row in plan["source_inventory"]["scientific_tasks"]}
    selected_tasks={row["scientific_task_id"] for row in plan["source_inventory"]["frames"]}
    reference_tasks={row["scientific_task_id"] for row in plan["source_inventory"]["references"]}
    auxiliaries=geometry.get("auxiliary_sources") or []
    remaining={key:task for key,task in tasks.items() if key not in selected_tasks|reference_tasks}
    aux_pairs=[]
    for row in auxiliaries:
        matches=[key for key,task in remaining.items() if task["run"]==row.get("run")
                 and task["source"]["sha256"]==row.get("source_sha256")]
        if len(matches)!=1:raise ValueError("Auxiliary rig context lacks one exact departed predecessor source")
        aux_pairs.append((row,matches[0]))
    if len(aux_pairs)!=len(remaining) or {key for _,key in aux_pairs}!=set(remaining):
        raise ValueError("Rig geometry does not certify every archived temporal predecessor")
    for row, task_id in [
        *((row, expected_refs[run]["scientific_task_id"]) for run, row in references.items()),
        *((frames[(row["run"], row["frame_index"])], row["scientific_task_id"])
          for row in plan["source_inventory"]["frames"]),
        *aux_pairs,
    ]:
        task = tasks[task_id]
        if row.get("source_sha256") != task["source"]["sha256"]:
            raise ValueError("Rig frame/reference belongs to another exact source image")
        reference = row.get("reference_rig") or {}
        identity=reference.get("identity")
        if isinstance(identity,dict):bound_path(identity)
        elif isinstance(identity,str) and identity:
            bound_path(reference.get("measured_reference"))
        else:raise ValueError("Rig reference lacks an exact measured source identity")
        polygon = np.asarray(reference.get("polygon_reference_native"), float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4 or not np.isfinite(polygon).all():
            raise ValueError("Rig reference lacks photograph-measured rigid-wall geometry")
        metadata.homography(row.get("H_reference_rig_to_frame_native"))
        if row.get("phase_shape_hw") != grid["shape_hw"]:
            raise ValueError("Rig frame/reference declares a different phase raster")
        top = load_array(row["frame_photo_top_keep"])
        if top.dtype != bool or list(top.shape) != grid["shape_hw"] or not top[-1].all() or np.any(top[:-1] & ~top[1:]):
            raise ValueError("Photographic top must be a full-canvas false-prefix/true-suffix proof")
        if not isinstance(row.get("frame_fixture_polygons_native"), dict):
            raise ValueError("Source-bound hardware polygons are required before inference")
        if not row["frame_fixture_polygons_native"] and row.get("fixtures_verified_none") is not True:
            raise ValueError("Empty hardware exclusions lack photographic verification")
        if set(row["frame_fixture_polygons_native"])!=set(HIDDEN_CHART_NAMES):
            raise ValueError("Rig hardware must bind the three approved photographed ColorChecker/book polygons")
        params = row.get("area_parameters") or {}
        metadata.homography(params.get("H_ac28_to_frame_native"))
        metadata.homography(params.get("H_ac28_to_metric"))
        if params.get("fixed_area_mm2") != .027053 or type(params.get("physical_scale_approved")) is not bool:
            raise ValueError("Rig area must declare its fixed historical value and provisional scale status")
    return frames, references


def load_array(asset):
    path=bound_path(asset)
    if asset.get("array_key"):
        with np.load(path,allow_pickle=False) as archive:
            return archive[asset["array_key"]]
    return np.load(path,mmap_mode="r",allow_pickle=False)


def stage_geometry_assets(root,value):
    """Copy new geometry proofs/masks to the candidate without editing originals."""
    if isinstance(value,list):return [stage_geometry_assets(root,item) for item in value]
    if not isinstance(value,dict):return deepcopy(value)
    if isinstance(value.get("path"),str) and isinstance(value.get("sha256"),str):
        source=bound_path(value)
        # Existing RAW photographs and frozen parent calibration remain their
        # exact source contracts. New local derivative geometry is permanent
        # candidate data, copied by hash under an unambiguous content address.
        if not str(source).startswith("\\\\"):
            destination=Path(root)/"rig_camera_revision/proofs"/value["sha256"]/source.name
            copy_file_verified(source,destination)
            return dict(value,path=str(destination))
        return deepcopy(value)
    return {key:stage_geometry_assets(root,item) for key,item in value.items()}


def retained_binding(parent, source, source_task):
    """Declare the exact parent source cache before fresh inference."""
    if source["source_changed"]:
        return None
    from darsia.fluidflower.phase_retained_source_inference import CONTRACT as cache_contract
    run, index = source["run"], source["parent_source_frame_index"]
    frame_path = Path(parent) / "barrier_rule_b_frames" / run / f"frame_{index:03d}.json"
    frame = read_json(frame_path)
    # The final B proof owns the category, while the original G3 export owns
    # the scientifically supported cache mask. Both are independently bound.
    g3_binding = Path(parent) / "approved_g3_frame_bindings" / run / f"frame_{index:03d}.json"
    if not g3_binding.is_file():
        # Top/B deliberately reuse parent photographs and saved maps rather
        # than copy the probability store. Follow their exact sealed lineage,
        # never a directory search or a guessed latest experiment.
        b_seal=read_json(Path(parent)/"barrier_rule_b_candidate_receipt.json")
        top_registration=read_json(bound_path(b_seal["parent_registration"]))
        if top_registration.get("registration_kind")!="approved_g3_top_surface":
            raise ValueError("Rule B lineage lacks its exact top-surface parent")
        top=Path(top_registration["model_root"])
        top_seal=read_json(top/"top_surface_candidate_receipt.json")
        g3_registration=read_json(bound_path(top_seal["parent_registration"]))
        if g3_registration.get("registration_kind")!="approved_g3_downstream":
            raise ValueError("Top-surface lineage lacks its approved G3 probability producer")
        g3_root=Path(g3_registration["model_root"])
        loaded_parent=read_json(g3_root/"phase_optical_regions.json")
        directory=(loaded_parent.get("approved_g3_gas") or {}).get("frame_bindings_directory")
        if not directory:
            raise ValueError("Approved G3 parent lacks its exact probability receipt bindings")
        g3_binding=g3_root/directory/run/f"frame_{index:03d}.json"
    g3 = read_json(g3_binding)
    if frame["binding"]["native_manifest_sha256"] != source_task["parent_native_manifest"]["sha256"]:
        raise ValueError("Retained categorical cache belongs to another parent native source")
    probability = deepcopy(g3["probability_receipt"])
    bound_path(probability)
    return dict(contract=cache_contract, run=run, frame_index=index,
        parent_native_manifest=source_task["parent_native_manifest"],
        parent_frame_receipt=artifact(frame_path), parent_probability_receipt=probability)


def _copy_heads_and_request(parent, root):
    for directory in ("models",):
        for source in (Path(parent) / directory).rglob("*"):
            if source.is_file():
                copy_file_verified(source, Path(root) / source.relative_to(parent))
    for name in ("regional_model.json", "training_request.json", "expected_review_runs.json"):
        copy_file_verified(Path(parent) / name, Path(root) / name)


def load_scientific_preparation(path):
    """Import either the original complete stage or its explicit AC44 composition."""
    from .run_ffac_rig_camera_candidate import CONTRACT as upstream_contract,PLAN_FILE as upstream_name,SOURCE_RECEIPT_CONTRACT
    path=Path(path).resolve();plan=read_json(path);root=Path(plan["binding"]["candidate"])
    original=root/"scientific_preparation_receipt.json"
    composed=root/"scientific_preparation_composed_receipt.json"
    if original.exists():
        if composed.exists():raise ValueError("Ambiguous original/composed scientific preparation seals")
        return load_prepared_plan(path)
    if not composed.is_file():raise ValueError("A complete original or explicitly composed910-source seal is required")
    if (path!=(root/upstream_name).resolve() or plan["binding"].get("contract")!=upstream_contract
            or plan["binding"].get("spec_sha256")!=SPEC_SHA256
            or plan.get("job_sha256")!=canonical_sha256(plan["binding"])
            or (plan.get("run_count"),plan.get("frame_count"))!=(43,864)):
        raise ValueError("Composed science does not belong to the exact immutable upstream plan")
    seal=read_json(composed);composition=seal.get("composition") or {}
    recovery_asset=composition.get("recovery_plan");result_asset=composition.get("recovery_result")
    recovery=read_json(bound_path(recovery_asset));result=read_json(bound_path(result_asset))
    plan_asset=artifact(path)
    if (seal.get("contract")!="ffac-rig-camera-scientific-preparation-composed-v1"
            or seal.get("complete") is not True or seal.get("stage")!="scientific-preparation"
            or seal.get("job_sha256")!=plan["job_sha256"] or seal.get("original_plan")!=plan_asset
            or (seal.get("source_count"),seal.get("run_count"),seal.get("frame_count"))!=(910,43,864)
            or composition.get("original_failed_job_relabelled") is not False
            or (composition.get("preserved_source_count"),composition.get("recovered_source_count"))!=(889,21)
            or recovery.get("contract")!="ffac-rig-ac44-scientific-recovery-v1"
            or recovery.get("original_plan")!=plan_asset or recovery.get("original_job_sha256")!=plan["job_sha256"]
            or recovery.get("original_code")!=plan["binding"]["code"]
            or result.get("contract")!=recovery["contract"] or result.get("complete") is not True
            or result.get("recovery_plan")!=recovery_asset or result.get("preserved_sources_unchanged") is not True):
        raise ValueError("Composed preparation lost its exact original/recovery scope or provenance")
    derived=recovery.get("recovery_metadata_binding") or {}
    current=derived.get("current_code") or {};original_code=plan["binding"]["code"]
    expected_invariants=dict(source_shape=list(metadata.SOURCE_SHAPE),source_box=list(metadata.SOURCE_BOX),
        scientific_contract=metadata.SCIENTIFIC_CONTRACT,unit_denominator=65535,
        native_dtype="float32",source_receipt_contract=SOURCE_RECEIPT_CONTRACT)
    if (composition.get("recovery_metadata_binding")!=derived or derived.get("scientific_invariants")!=expected_invariants
            or derived.get("original_metadata_sha256")!=original_code["metadata_sha256"]
            or derived.get("metadata_substitution_explicit") is not True
            or derived.get("original_whole_code_binding_pass") is not False
            or derived.get("original_numerical_producer_and_raw_unchanged") is not True
            or any(current.get(key)!=original_code[key] for key in ("producer_sha256","raw_processor_sha256"))
            or current.get("metadata_sha256")!=derived.get("metadata",{}).get("sha256")):
        raise ValueError("Recovery metadata did not declare its exact nonnumerical substitution")
    bound_path(derived["metadata"])
    code=recovery.get("recovery_code") or {}
    if set(code)!={"tool","dispatcher"} or composition.get("recovery_code")!=code:
        raise ValueError("Recovery numerical dispatcher binding differs")
    for asset in [*code.values(),*recovery.get("original_failure_logs",[])]:bound_path(asset)
    if composition.get("original_failed_preparation")!=recovery.get("original_failure_logs"):
        raise ValueError("Recovery relabelled the original failed job evidence")
    recovery_hash=canonical_sha256(dict(original_plan=recovery["original_plan"],recovery_code=code,
        recovery_metadata_binding=derived,missing_tasks=recovery["missing_tasks"],preserved_sources=recovery["preserved_sources"]))
    if recovery.get("recovery_job_sha256")!=recovery_hash or composition.get("recovery_job_sha256")!=recovery_hash:
        raise ValueError("Recovery job identity changed")
    tasks={row["task_id"]:row for row in plan["source_inventory"]["scientific_tasks"]}
    preserved=recovery.get("preserved_sources") or [];recovered=result.get("recovered_sources") or []
    missing=recovery.get("missing_tasks") or []
    entries=[*preserved,*recovered];ids=[row["task_id"] for row in entries]
    if (len(tasks)!=910 or len(preserved)!=889 or len(recovered)!=21 or len(missing)!=21
            or any(task.get("run")!="ac44" or tasks.get(task.get("task_id"))!=task for task in missing)
            or {row["task_id"] for row in recovered}!={task["task_id"] for task in missing}
            or len(set(ids))!=910 or set(ids)!=set(tasks)
            or seal.get("sources")!=sorted(entries,key=lambda row:row["task_id"])):
        raise ValueError("Composed source sets differ from the889 preserved and21 exactAC44 tasks")
    frames=plan["source_inventory"].get("frames",[]);references=plan["source_inventory"].get("references",[])
    if (len(frames)!=864 or len({(row["run"],row["frame_index"]) for row in frames})!=864
            or len({row["run"] for row in frames})!=43 or len(references)!=43
            or {row["run"] for row in references}!={row["run"] for row in frames}):
        raise ValueError("Composed upstream lost the complete selected/reference camera coverage")
    for row in entries:
        task=tasks[row["task_id"]];source_path=bound_path(row["receipt"])
        receipt=read_json(source_path)
        if (source_path!=(root/"scientific_sources"/task["task_id"]/"prepared_source.json").resolve()
                or receipt.get("contract")!=SOURCE_RECEIPT_CONTRACT or receipt.get("complete") is not True
                or receipt.get("job_sha256")!=plan["job_sha256"] or receipt.get("task")!=task):
            raise ValueError("Composed source receipt differs from its exact source task")
    return dict(plan=plan,plan_asset=plan_asset,preparation_asset=artifact(composed))


def create_inference_plan(prepared_plan, geometry_manifest, geometry_acceptance, *, resume=False):
    state = load_scientific_preparation(prepared_plan)
    plan, root = state["plan"], Path(state["plan"]["binding"]["candidate"])
    gate = require_inference_geometry(geometry_manifest, geometry_acceptance, plan)
    frames, refs = validate_geometry_inputs(gate, plan)
    staged=stage_geometry_assets(root,gate["geometry"])
    staged_path=root/"rig_camera_revision/certified_geometry.json"
    atomic_json(staged_path,staged)
    original_geometry_path=root/"rig_camera_revision/original_geometry_manifest.json"
    copy_file_verified(bound_path(gate["geometry_manifest"]),original_geometry_path)
    frames={(row["run"],row["frame_index"]):row for row in staged["frames"]}
    refs={row["run"]:row for row in staged["references"]}
    acceptance=stage_geometry_assets(root,read_json(gate["geometry_acceptance"]["path"]))
    acceptance_path=root/"rig_camera_revision/geometry_acceptance.json"
    atomic_json(acceptance_path,acceptance)
    parent = Path(plan["binding"]["parent"])
    from darsia.fluidflower.phase_rig_domain_inference import require_frozen_heads
    from darsia.fluidflower.phase_runtime import load_physics_profile
    from darsia.fluidflower.phase_validation import _model_identity
    require_frozen_heads(parent)
    parent_profile = metadata.public_profile(load_physics_profile(parent))
    tasks = {row["task_id"]: row for row in plan["source_inventory"]["scientific_tasks"]}
    sources = [dict(row, rig_geometry=frames[(row["run"], row["frame_index"])],
                    retained_source_cache=retained_binding(parent, row, tasks[row["scientific_task_id"]]))
               for row in plan["source_inventory"]["frames"]]
    binding = dict(contract=CONTRACT, upstream_plan=state["plan_asset"],
        scientific_preparation=state["preparation_asset"], spec_sha256=SPEC_SHA256,
        geometry_manifest=gate["geometry_manifest"], geometry_acceptance=gate["geometry_acceptance"],
        parent_model_identity=plan["binding"]["parent_model_identity"], code=_code_binding(),
        configuration=metadata.CONFIGURATION, cpu_only=True, training=False, labels_changed=False,
        mass_eligible=False, scientific_acceptance=False, automatic_activation=False)
    grid=metadata.phase_grid(gate["geometry"]["phase_shape_hw"])
    binding["phase_grid"]=grid
    binding["staged_geometry_manifest"]=artifact(staged_path)
    binding["staged_geometry_acceptance"]=artifact(acceptance_path)
    binding["original_geometry_copy"]=artifact(original_geometry_path)
    destination = root / PLAN_FILE
    if destination.exists():
        old = read_json(destination)
        if not resume or old["binding"] != binding or old["source_inventory"] != sources:
            raise ValueError("Inference resume differs from its immutable prepared source/geometry/code plan")
        return old
    _copy_heads_and_request(parent, root)
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=plan["created_utc"],
        display_name="FF_AC · G3 gas · Barrier rule B · Rig domain · Camera clock",
        inherited_training_candidate_id=parent.name, training_performed_in_this_revision=False,
        raw_inference_performed=True, scientific_acceptance=False, mass_eligible=False)
    summary["approved_g3_production"]["rig_domain_revision"] = dict(format=metadata.REVISION_CONTRACT,
        spec_sha256=SPEC_SHA256, parent_model_identity=plan["binding"]["parent_model_identity"],
        geometry_manifest_sha256=gate["geometry_manifest"]["sha256"],
        physical_scale_approved=read_json(gate["geometry_acceptance"]["path"])["physical_scale_approved"],
        source_native_grid=metadata.SOURCE_SHAPE, phase_grid=grid)
    atomic_json(root / "summary.json", summary)
    identity = list(_model_identity(root))
    inference = dict(binding=binding, job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)),
        model_identity=identity, created_utc=plan["created_utc"], source_inventory=sources,
        references=[dict(row, rig_geometry=refs[row["run"]]) for row in plan["source_inventory"]["references"]],
        upstream_plan=plan, inherited_public_profile=parent_profile,
        auxiliary_sources=deepcopy(staged.get("auxiliary_sources",[])))
    atomic_json(destination, inference)
    return inference


def _prepared(root, inference, task_id):
    science_root=Path(inference.get("scientific_source_root",inference["upstream_plan"]["binding"]["candidate"]))
    key=(str(science_root),inference["upstream_plan"]["job_sha256"],task_id)
    if key in _PREPARED_RECEIPTS:return _PREPARED_RECEIPTS[key]
    tasks = {row["task_id"]: row for row in inference["upstream_plan"]["source_inventory"]["scientific_tasks"]}
    result=verify_prepared_source(science_root, tasks[task_id], inference["upstream_plan"]["job_sha256"])
    _PREPARED_RECEIPTS[key]=result
    return result


def load_scientific(receipt):
    full = receipt["full_native_scientific"]
    key=canonical_sha256(full)
    if key in _SCIENTIFIC_CACHE:
        _SCIENTIFIC_CACHE.move_to_end(key);return _SCIENTIFIC_CACHE[key]
    signal = np.load(bound_path(full["log_linear_rgb"]), mmap_mode="r", allow_pickle=False)
    valid = np.load(bound_path(full["valid_mask"]), mmap_mode="r", allow_pickle=False)
    certificate = read_json(bound_path(full["receipt"]))
    result=(signal, valid, certificate["producer_audit"])
    _SCIENTIFIC_CACHE[key]=result
    while len(_SCIENTIFIC_CACHE)>3:_SCIENTIFIC_CACHE.popitem(last=False)
    return result


def task_geometry(inference,task):
    if task["parent_reference"]:
        candidates=[row["rig_geometry"] for row in inference["references"] if row["run"]==task["run"]]
    else:
        candidates=[row["rig_geometry"] for row in inference.get("geometry_sources",inference["source_inventory"])
                    if row["run"]==task["run"] and row["source_sha256"]==task["source"]["sha256"]]
        candidates += [row for row in inference["auxiliary_sources"]
                       if row["run"]==task["run"] and row["source_sha256"]==task["source"]["sha256"]]
    if len(candidates)!=1 or candidates[0].get("source_sha256")!=task["source"]["sha256"]:
        raise ValueError("Scientific source lacks one own rig/top/hardware context")
    return candidates[0]


def extended_scientific(root,inference,receipt,geometry,processor):
    """Write a separate full-phase correction; never change frozen native prep."""
    from darsia.fluidflower.phase_ffac_rig_raw_extension import CONTRACT as raw_contract
    from darsia.fluidflower.phase_ffac_rig_raw_dispatch import extend_prepared_source
    from darsia.fluidflower.phase_ffac_rig_raw_dispatch_binding import bind_prepared_source_processor
    folder=Path(root)/"extended_scientific"/receipt["task"]["task_id"]
    path=folder/"receipt.json"
    binding=dict(inference_job_sha256=inference["job_sha256"],prepared_source=artifact(
        Path(inference.get("scientific_source_root",inference["upstream_plan"]["binding"]["candidate"]))/
            "scientific_sources"/receipt["task"]["task_id"]/"prepared_source.json"),
        geometry=geometry,phase_grid=inference["binding"]["phase_grid"])
    cache_key=(str(Path(root)),receipt["task"]["task_id"],canonical_sha256(binding))
    if cache_key in _EXTENDED_CACHE:
        _EXTENDED_CACHE.move_to_end(cache_key)
        return _EXTENDED_CACHE[cache_key]
    recovery=inference.get("recovery") or {}
    inherited=(recovery.get("extended_scientific") or {}).get(receipt["task"]["task_id"])
    if inherited is not None:
        old=read_json(bound_path(inherited))
        old_binding=old.get("binding") or {}
        if (old.get("complete") is not True or
                old_binding.get("inference_job_sha256")!=recovery.get("original_inference_job_sha256") or
                old_binding.get("prepared_source")!=binding["prepared_source"] or
                old_binding.get("geometry")!=geometry or old_binding.get("phase_grid")!=binding["phase_grid"]):
            raise ValueError("Imported RAW extension differs from its original source/geometry/producer")
        full,valid=compose_extended_scientific(old)
        return cache_extended(cache_key,(full,valid,old))
    if path.exists():
        old=read_json(path)
        if old.get("binding")!=binding or old.get("complete") is not True:
            raise ValueError("Full-phase scientific resume has another source/geometry/correction")
        full,valid=compose_extended_scientific(old)
        return cache_extended(cache_key,(full,valid,old))
    native,valid,audit=load_scientific(receipt)
    context,observed=geometry_context(geometry,valid,audit)
    code_aliases=bind_prepared_source_processor(processor,audit,
        scientific_receipt=receipt["full_native_scientific"]["receipt"])
    full,supported,extension=extend_prepared_source(native,audit,processor,context.rig_domain&observed)
    if code_aliases:extension["verified_prepared_code_path_aliases"]=code_aliases
    if not np.array_equal(full[:,metadata.SOURCE_BOX[1]:metadata.SOURCE_BOX[1]+metadata.SOURCE_SHAPE[0],
                                metadata.SOURCE_BOX[0]:metadata.SOURCE_BOX[0]+metadata.SOURCE_SHAPE[1]].view(np.uint32),
                          np.asarray(native).view(np.uint32)):
        raise ValueError("Full-phase RAW extension changed the prepared native centre")
    windows=[]
    for number,row in enumerate(extension["windows"]):
        a,b,c,d=row["canvas_xyxy"]
        windows.append(dict(canvas_xyxy=[a,b,c,d],log_linear_rgb=_npy(
            folder/f"ring_{number:02d}.npy",np.ascontiguousarray(full[:,b:d,a:c]))))
    descriptor=dict(contract=raw_contract,complete=True,binding=binding,
        source_sha256=receipt["task"]["source"]["sha256"],phase_shape_hw=list(supported.shape),
        native_origin_xy=metadata.ORIGIN_XY,unit_denominator=65535,
        storage="exact-native-centre-plus-disjoint-frozen-raw-ring-windows",
        native_centre=receipt["full_native_scientific"],ring_windows=windows,extension_audit=extension)
    atomic_json(path,descriptor)
    return cache_extended(cache_key,(full,supported,descriptor))


def cache_extended(key,value):
    _EXTENDED_CACHE[key]=value;_EXTENDED_CACHE.move_to_end(key)
    while len(_EXTENDED_CACHE)>3:_EXTENDED_CACHE.popitem(last=False)
    return value


def compose_extended_scientific(descriptor):
    """Strictly reconstruct a transient full raster from sparse source assets."""
    from darsia.fluidflower.phase_ffac_rig_raw_extension import CONTRACT as raw_contract
    if descriptor.get("contract")!=raw_contract or descriptor.get("complete") is not True or descriptor.get("storage")!="exact-native-centre-plus-disjoint-frozen-raw-ring-windows":
        raise ValueError("Full-phase scientific descriptor has another correction/storage contract")
    shape=metadata.phase_grid(descriptor["phase_shape_hw"])["shape_hw"]
    centre=load_array(descriptor["native_centre"]["log_linear_rgb"])
    if centre.dtype!=np.float32 or centre.shape!=(3,*metadata.SOURCE_SHAPE):
        raise ValueError("Sparse RAW ring centre has another source grid/encoding")
    full=np.full((3,*shape),np.nan,np.float32)
    x,y,width,height=metadata.SOURCE_BOX
    full[:,y:y+height,x:x+width]=centre
    covered=np.zeros(shape,bool);covered[y:y+height,x:x+width]=True
    for row in descriptor["ring_windows"]:
        a,b,c,d=row["canvas_xyxy"]
        if any(type(value) is not int for value in (a,b,c,d)) or not(0<=a<c<=shape[1] and 0<=b<d<=shape[0]) or covered[b:d,a:c].any():
            raise ValueError("Sparse RAW ring overlaps the native centre/another window or escapes its phase grid")
        values=load_array(row["log_linear_rgb"])
        if values.dtype!=np.float32 or values.shape!=(3,d-b,c-a):
            raise ValueError("Sparse RAW ring has another shape/float encoding")
        full[:,b:d,a:c]=values;covered[b:d,a:c]=True
    return full,np.isfinite(full).all(0)


def canvas_homography(native_h):
    x,y=metadata.SOURCE_BOX[:2]
    translate=np.array([[1,0,x],[0,1,y],[0,0,1]],np.float64)
    return translate@np.asarray(native_h,np.float64)@np.linalg.inv(translate)


def prepare_barrier_geometry(root,inference,source,domain):
    """Use inherited native classes, own source transport, then tagged extension."""
    from darsia.fluidflower.phase_gas_barrier_runtime import read_components
    from darsia.fluidflower.phase_ffac_barrier_extension import extend_frozen_barrier_components
    parent=Path(inference["upstream_plan"]["binding"]["parent"])
    if source["source_changed"]:
        geometry=read_json(parent/"gas_barrier_rule_b/source_geometry_manifest.json")
        original=artifact(parent/"gas_barrier_rule_b/reference_components.npz")
        if original["sha256"]!=geometry["reference_components"]["sha256"]:
            raise ValueError("Frozen Rule B reference differs from its parent derivation")
        arrays=read_components(original,metadata.SOURCE_SHAPE)
        transform=canvas_homography(np.asarray(source["rig_geometry"]["area_parameters"]["H_ac28_to_frame_native"]))
        native_ids=_canvas(arrays["open_components"],0,phase_shape_hw=domain.shape)
        native_barrier=_canvas(arrays["barrier_mask"],False,phase_shape_hw=domain.shape)
        ids=cv2.warpPerspective(native_ids,transform,(domain.shape[1],domain.shape[0]),flags=cv2.INTER_NEAREST)
        barrier=cv2.warpPerspective(native_barrier.astype(np.uint8),transform,
            (domain.shape[1],domain.shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
        source_ids=geometry["source_component_ids"]
        transport="own source H, nearest-neighbour transport of frozen AC28 reference classes"
    else:
        index=source["parent_source_frame_index"]
        receipt=read_json(parent/"barrier_rule_b_frames"/source["run"]/f"frame_{index:03d}.json")
        original=receipt["binding"]["components"]
        arrays=read_components(original,metadata.SOURCE_SHAPE)
        ids=_canvas(arrays["open_components"],0,phase_shape_hw=domain.shape)
        barrier=_canvas(arrays["barrier_mask"],False,phase_shape_hw=domain.shape)
        source_ids=receipt["qa"]["source_component_ids"]
        transport="exact frozen classes from the identical parent source image"
    values,audit=extend_frozen_barrier_components(ids,barrier,domain)
    folder=Path(root)/"rig_camera_revision/barrier"/source["run"]
    payload=io.BytesIO();np.savez_compressed(payload,**values)
    asset=_write_bytes(folder/f"frame_{source['frame_index']:03d}.npz",payload.getvalue())
    audit.update(frozen_components=original,source_component_ids=source_ids,transport=transport,asset=asset)
    return values,source_ids,audit


def align_scientific(signal, valid, transform, *, target_shape_hw=None):
    """Warp linear scientific reference, never current source pixels."""
    from darsia.fluidflower.phase_rig_domain_inference import log_to_unit_rgb
    from .ffac_v5_temporal_alignment import warp_reference_to_current
    rgb, supported = log_to_unit_rgb(signal, valid)
    aligned, supported = warp_reference_to_current(rgb, transform,
        target_shape_hw=tuple(target_shape_hw or metadata.SOURCE_SHAPE), source_valid_mask=supported)
    result = np.full((3, *aligned.shape[:2]), np.nan, np.float32)
    supported &= (aligned > 0).all(-1)
    result[:, supported] = np.log(aligned[supported].astype(np.float64) * 65535.).T.astype(np.float32)
    return result, supported


def geometry_context(row, feature_valid, science_audit):
    from darsia.fluidflower.phase_ffac_rig_geometry import independent_rig_frame_context, photographic_observation_mask
    params = row["area_parameters"]
    renderer = science_audit["renderer"]
    footprint = photographic_observation_mask(renderer["review_to_camera_h"], renderer["camera_jpeg_wh"],
        row["phase_shape_hw"][::-1], native_origin_xy=tuple(metadata.ORIGIN_XY))
    top = load_array(row["frame_photo_top_keep"])
    metric = np.asarray(params["H_ac28_to_metric"]) @ np.linalg.inv(np.asarray(params["H_ac28_to_frame_native"]))
    context = independent_rig_frame_context(row["reference_rig"],row["H_reference_rig_to_frame_native"],
        tuple(metadata.SOURCE_SHAPE),feature_valid_mask=feature_valid,
        canvas_wh=row["phase_shape_hw"][::-1],content_box_xywh=metadata.SOURCE_BOX,
        frame_photo_top_keep=top,frame_fixture_polygons_native=row["frame_fixture_polygons_native"],
        fixtures_verified_none=row.get("fixtures_verified_none",False),
        photographic_observed_mask=footprint,H_frame_native_to_metric_pixel=metric)
    return context, footprint


def _canvas(native, fill=255, *, phase_shape_hw=None):
    x,y,width,height = metadata.SOURCE_BOX
    result = np.full(metadata.phase_grid(phase_shape_hw)["shape_hw"],fill,np.asarray(native).dtype)
    result[y:y+height,x:x+width] = native
    return result


def _crop(full):
    x,y,width,height = metadata.SOURCE_BOX
    return full[y:y+height,x:x+width]


def write_maps(root, run, index, pipeline, labels, *, phase_shape_hw=None):
    from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA,GAS_RGBA
    colors=dict(aqueous=AQUEOUS_RGBA,gas=GAS_RGBA)
    if labels.dtype != np.uint8 or list(labels.shape) != metadata.phase_grid(phase_shape_hw)["shape_hw"]:
        raise ValueError("Rig production labels must occupy the declared full canvas")
    combined = np.zeros((*labels.shape,4),np.uint8)
    for name,value in (("aqueous",1),("gas",2)):
        combined[labels==value] = colors[name]
    folder = Path(root)/"validation/run_packages"/run/"review_assets/native_phase_maps"/run/("A_"+pipeline[:16])
    maps = {}
    for name in ("combined","aqueous","gas","mixed"):
        rgba = combined if name=="combined" else np.zeros_like(combined)
        if name in ("aqueous","gas"):
            value=1 if name=="aqueous" else 2
            rgba[labels==value] = colors[name]
        encoded=io.BytesIO()
        Image.fromarray(rgba,"RGBA").save(encoded,format="PNG",compress_level=1)
        path=folder/f"frame_{index:03d}.png" if name=="combined" else folder/name/f"frame_{index:03d}.png"
        maps[name]=_write_bytes(path,encoded.getvalue())
    encoded=io.BytesIO()
    Image.fromarray(labels,"L").save(encoded,format="PNG",compress_level=1)
    proof=_write_bytes(Path(root)/"production_labels"/run/f"frame_{index:03d}.png",encoded.getvalue())
    with Image.open(proof["path"]) as saved:
        if not np.array_equal(np.asarray(saved),labels):
            raise ValueError("Persisted rig phase proof differs from its computed categories")
    return maps, proof


def _context_assets(root, run, stem, context, classes):
    folder=Path(root)/"rig_camera_revision/context"/run
    encoded=io.BytesIO()
    np.savez_compressed(encoded,pixel_area_mm2=context.pixel_area_mm2)
    area=_write_bytes(folder/(stem+".area.npz"),encoded.getvalue())
    area["array_key"]="pixel_area_mm2"
    encoded=io.BytesIO()
    np.savez_compressed(encoded,rig_domain=context.rig_domain,hardware_excluded=context.hardware_excluded,**classes)
    masks=_write_bytes(folder/(stem+".classes.npz"),encoded.getvalue())
    return dict(contract=metadata.RIG_CONTRACT,canvas_wh=metadata.CANVAS_WH,
        phase_shape_hw=list(context.rig_domain.shape),
        native_origin_xy=metadata.ORIGIN_XY,fixed_area_mm2=.027053,area_map=area,classes=masks,
        parameters=context.parameters,mass_eligible=False)


def _jpeg_canvas(path,rgb):
    rgb=np.asarray(rgb)
    if np.issubdtype(rgb.dtype,np.floating):
        if not np.isfinite(rgb).all() or np.any((rgb<0)|(rgb>1)):
            raise ValueError("Floating photographic views must be finite normalized RGB")
        rgb=np.rint(rgb*255).astype(np.uint8)
    elif rgb.dtype!=np.uint8:
        raise TypeError("Photographic views must be normalized float or uint8 RGB")
    if rgb.shape!=(*metadata.SOURCE_SHAPE,3):
        raise ValueError("Photographic view differs from its declared native source shape")
    canvas=np.full((*metadata.CANVAS_WH[::-1],3),255,np.uint8)
    x,y,width,height=metadata.SOURCE_BOX
    canvas[y:y+height,x:x+width]=rgb
    encoded=io.BytesIO();Image.fromarray(canvas,"RGB").save(encoded,format="JPEG",quality=100,subsampling=0)
    return _write_bytes(Path(path),encoded.getvalue())


def parent_photographs(parent_native_path,record,index,*,reference=False):
    """Reuse declarations from the exact parent, without rehashing every JPEG."""
    from darsia.fluidflower.phase_v5_source_alias import resolve_original_native_source
    native_path=Path(parent_native_path);run=read_json(native_path)["run"]
    review_root=native_path.parent.parent.parent
    context_key=str(native_path)
    if context_key not in _PARENT_PHOTO_CONTEXT:
        original_root,original_native,_=resolve_original_native_source(review_root,run)
        change=read_json(original_root/"native_frames"/run/"change_display_manifest.json")
        if change.get("source_manifest_sha256")!=sha256_file(original_native):
            raise ValueError("Parent photographic change view binds another native source")
        _PARENT_PHOTO_CONTEXT[context_key]=change
    change=_PARENT_PHOTO_CONTEXT[context_key]
    change_row=change["reference"] if reference else change["frames"][index]
    photos={mode:dict(path=record["v5_display_modes"][mode]["jpeg_path"],
            sha256=record["v5_display_modes"][mode]["jpeg_sha256"]) for mode in ("natural","lift1","lift2")}
    photos.update(change=dict(path=change_row["jpeg_path"],sha256=change_row["jpeg_sha256"]),
        raw_detail=dict(path=record["v5_raw_detail_jpeg_path"],sha256=record["v5_raw_detail_jpeg_sha256"]),
        original=dict(path=record["source_path"],sha256=record["source_sha256"]))
    for value in photos.values():bound_path(value,verify=False)
    return photos


def materialize_source_record(root,inference,source,receipt,frame_receipt):
    """Retained photo bytes stay frozen; only the27 new sources get new views."""
    task=receipt["task"];parent_native=read_json(bound_path(task["parent_native_manifest"]))
    if not source["source_changed"]:
        record=deepcopy(parent_native["frame_sources"][source["parent_source_frame_index"]])
        photos=parent_photographs(task["parent_native_manifest"]["path"],record,source["parent_source_frame_index"])
    else:
        from .ffac_track1_display_modes import render_display_modes
        from .ffac_track1_change_flatfield import prepare_flatfield_reference,register_flatfield_reference,render_change_flatfield
        from .materialize_ffac_v5_sources import _raw_detail_display
        signal,valid,audit=load_scientific(receipt)
        baseline_row=next(row for row in inference["references"] if row["run"]==source["run"])
        baseline,b_valid,_=load_scientific(_prepared(root,inference,baseline_row["scientific_task_id"]))
        with np.load(bound_path(frame_receipt["rig_domain_revision"]["classes"]),allow_pickle=False) as archive:
            photo=_crop(archive["rig_domain"])
        modes,_=render_display_modes(signal,photo)
        prepared=prepare_flatfield_reference(baseline,source_valid_mask=b_valid)
        registered=register_flatfield_reference(prepared,np.asarray(receipt["geometry_frame"]["H_v5_B_to_frame_native"]),
            target_shape_hw=tuple(metadata.SOURCE_SHAPE))
        modes["change"],_=render_change_flatfield(signal,registered,photo,lift1_rgb=modes["lift1"])
        modes["raw_detail"]=np.rint(np.clip(_raw_detail_display(signal),0,1)*255).astype(np.uint8)
        folder=Path(root)/"source_alias/native_frames"/source["run"]
        photos={mode:_jpeg_canvas(folder/f"frame_{source['frame_index']:03d}.{mode}.jpg",rgb)
                for mode,rgb in modes.items()}
        photos["original"]=deepcopy(task["source"])
        geometry=receipt["geometry_frame"]
        record=dict(source_path=source["source_path"],source_sha256=source["source_sha256"],
            v5_raw_path=task["raw"]["path"],v5_raw_sha256=task["raw"]["sha256"],
            v5_log_linear_rgb_path=receipt["full_native_scientific"]["log_linear_rgb"]["path"],
            v5_log_linear_rgb_sha256=receipt["full_native_scientific"]["log_linear_rgb"]["sha256"],
            v5_log_linear_valid_mask_path=receipt["full_native_scientific"]["valid_mask"]["path"],
            v5_log_linear_valid_mask_sha256=receipt["full_native_scientific"]["valid_mask"]["sha256"],
            native_jpeg_sha256=photos["natural"]["sha256"],
            v5_raw_detail_jpeg_path=photos["raw_detail"]["path"],v5_raw_detail_jpeg_sha256=photos["raw_detail"]["sha256"],
            v5_display_modes={mode:dict(jpeg_path=photos[mode]["path"],jpeg_sha256=photos[mode]["sha256"])
                              for mode in ("natural","lift1","lift2")})
        record.update({key:deepcopy(value) for key,value in geometry.items() if key.startswith("v5_B_")})
        renderer=dict(audit["renderer"],run=source["run"])
        view_receipt=dict(format="ffac-v5-native-source-materialization-v1",run=source["run"],
            frame_index=source["frame_index"],source=dict(camera_jpeg=task["source"]["path"],
                camera_jpeg_sha256=task["source"]["sha256"]),certificate=dict(
                camera_jpeg_sha256=task["source"]["sha256"],native_shape_hw=metadata.SOURCE_SHAPE,
                renderer_certificate=renderer),frozen_photometry=True,scientific_source=receipt["full_native_scientific"])
        view_path=folder/f"frame_{source['frame_index']:03d}.v5.json";atomic_json(view_path,view_receipt)
        record.update(v5_frame_receipt_path=str(view_path),v5_frame_receipt_sha256=sha256_file(view_path))
    record.update(frame_index=source["frame_index"],elapsed_h=source["elapsed_h"],
        full_native_scientific=receipt["full_native_scientific"],
        full_extended_scientific=deepcopy(frame_receipt["full_extended_scientific"]),
        rig_domain_revision=frame_receipt["rig_domain_revision"])
    geometry_asset=receipt["geometry_receipt"]
    if not Path(geometry_asset["path"]).is_relative_to(Path(root)):
        target=Path(root)/"rig_camera_revision/source_geometry"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        copy_file_verified(bound_path(geometry_asset),target);geometry_asset=artifact(target)
    predecessor=deepcopy(source["temporal_predecessor"])
    if source["source_changed"] and predecessor is not None:
        previous=_prepared(root,inference,source["previous_scientific_task_id"])
        predecessor["full_native_scientific"]=previous["full_native_scientific"]
    binding=dict(run=source["run"],frame_index=source["frame_index"],elapsed_h=source["elapsed_h"],
        source_path=record["source_path"],source_sha256=record["source_sha256"],
        source_changed=source["source_changed"],own_source_geometry=source["source_changed"],
        parent_source_frame_index=source["parent_source_frame_index"],geometry_receipt=geometry_asset,
        H_ac28_to_frame_native=receipt["geometry_frame"]["H_ac28_to_frame_native"],
        H_v5_B_to_frame_native=receipt["geometry_frame"]["H_v5_B_to_frame_native"],
        phase_grid=inference["binding"]["phase_grid"],rig_domain_revision=record["rig_domain_revision"],
        full_native_scientific=record["full_native_scientific"],full_extended_scientific=record["full_extended_scientific"],photographs=photos,
        temporal_input_policy=source["temporal_input_policy"],temporal_predecessor=predecessor)
    return record,binding


def assemble_source_run(root,inference,run,sources,processor):
    destination=Path(root)/"source_alias/native_frames"/run/"manifest.json"
    run_receipt=Path(root)/"rig_camera_revision/source_runs"/(run+".json")
    if run_receipt.exists():
        old=read_json(run_receipt)
        if old.get("inference_job_sha256")!=inference["job_sha256"] or old.get("complete") is not True:
            raise ValueError("Completed source run belongs to another immutable inference")
        bound_path(old["native_manifest"]);bound_path(old["input_manifest"])
        return old
    records=[];bindings=[]
    for source in sources:
        receipt=_prepared(root,inference,source["scientific_task_id"])
        frame=read_json(Path(root)/"rig_camera_frames"/run/f"frame_{source['frame_index']:03d}.json")
        record,binding=materialize_source_record(root,inference,source,receipt,frame)
        records.append(record);bindings.append(binding)
    reference=next(row for row in inference["references"] if row["run"]==run)
    prepared=_prepared(root,inference,reference["scientific_task_id"])
    signal,valid,audit=load_scientific(prepared)
    full,fullvalid,extended=extended_scientific(root,inference,prepared,reference["rig_geometry"],processor)
    context,observed=geometry_context(reference["rig_geometry"],valid,audit)
    observed |= fullvalid
    classes=dict(modelled=context.rig_domain&fullvalid,
        filled_observed=context.rig_domain&observed&~fullvalid,
        filled_unobserved=context.rig_domain&~observed&~fullvalid)
    descriptor=_context_assets(root,run,"pre_injection",context,classes)
    parent_asset=prepared["task"]["parent_native_manifest"]
    parent=read_json(bound_path(parent_asset));ref=deepcopy(parent["reference"])
    ref.update(full_native_scientific=prepared["full_native_scientific"],full_extended_scientific=extended,
               rig_domain_revision=descriptor)
    photos=parent_photographs(parent_asset["path"],ref,None,reference=True)
    ref_binding=dict(run=run,frame_index=None,source_sha256=ref["source_sha256"],source_path=ref["source_path"],
        elapsed_h=ref["elapsed_h"],geometry_receipt=prepared["geometry_receipt"],
        H_ac28_to_frame_native=prepared["geometry_frame"]["H_ac28_to_frame_native"],
        phase_grid=inference["binding"]["phase_grid"],rig_domain_revision=descriptor,
        full_native_scientific=prepared["full_native_scientific"],full_extended_scientific=extended,photographs=photos)
    camera_asset=sources[0]["camera_run"];camera=read_json(bound_path(camera_asset))
    inputs=metadata.build_source_bindings(parent_asset,camera_asset,bindings,ref_binding,spec_sha256=SPEC_SHA256)
    inputs_path=Path(root)/"source_alias/source_bindings"/(run+".json");atomic_json(inputs_path,inputs)
    native=metadata.build_native_metadata(parent,camera,records,bindings,ref_binding,input_manifest=artifact(inputs_path))
    # A complete public declaration includes the extended centre-plus-ring
    # source, while the strict native float reader continues to use its fixed
    # source grid and never calls filled categories scientific measurements.
    native["reference"]["full_extended_scientific"]=extended
    atomic_json(destination,native)
    result=dict(complete=True,run=run,inference_job_sha256=inference["job_sha256"],
        native_manifest=artifact(destination),input_manifest=artifact(inputs_path),
        frames=bindings,reference=ref_binding,frame_count=len(records))
    atomic_json(run_receipt,result)
    return result


def infer_run(root,inference,sources):
    """One run/processor per worker, bounded source/B/previous memory caches."""
    from darsia.fluidflower.phase_ffac_rig_raw_dispatch import create_rig_raw_source_processor
    tasks={row["task_id"]:row for row in inference["upstream_plan"]["source_inventory"]["scientific_tasks"]}
    processor=create_rig_raw_source_processor(tasks[sources[0]["scientific_task_id"]]["parent_native_manifest"]["path"])
    completed=[]
    for source in sources:
        try:
            result=infer_frame(root,inference,source,processor=processor)
        except BaseException as error:
            atomic_json(Path(root)/"inference_failures"/(source["run"]+".json"),dict(
                contract="ffac-rig-inference-run-failure-v1",run=source["run"],
                frame_index=source["frame_index"],source_sha256=source["source_sha256"],
                completed_frames_in_worker=len(completed),failed_utc=utc_now(),
                error=repr(error),complete=False,inference_plan=artifact(Path(root)/PLAN_FILE),
                automatic_retry=False))
            raise
        completed.append(result)
        progress_folder=Path(root)/"worker_progress"/source["run"]
        progress_folder.mkdir(parents=True,exist_ok=True)
        log=ProgressLog(progress_folder)
        log.event("rig_frame",run=source["run"],frame_index=source["frame_index"],
                  completed_frames=len(completed),total_frames=len(sources))
    all_run=[row for row in inference["source_inventory"] if row["run"]==sources[0]["run"]]
    source_receipt=None
    if not inference["binding"].get("probe_only") and len(sources)==len(all_run):
        source_receipt=assemble_source_run(root,inference,sources[0]["run"],all_run,processor)
    return dict(run=sources[0]["run"],frames=completed,source=source_receipt)


def bind_rig_packages(root,inference,source_runs):
    from darsia.fluidflower.phase_runtime import load_physics_profile,physics_runtime_identity
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    parent=Path(inference["upstream_plan"]["binding"]["parent"])
    old_source=read_json(parent/"review_source/postflight_manifest.json")
    sources={row["run"]:row for row in source_runs}
    updated_source_jobs=[]
    for old in old_source["jobs"]:
        run=old["run"];job=deepcopy(old)
        job.update(review_output_dir=str(Path(root)/"source_alias"),analysis_source_review_root=str(Path(root)/"source_alias"),
            native_manifest_sha256=sources[run]["native_manifest"]["sha256"],
            frame_count=sources[run]["frame_count"])
        original=read_json(parent/"review_source"/run/"postflight_manifest.json")
        atomic_json(Path(root)/"review_source"/run/"postflight_manifest.json",dict(original,jobs=[job]))
        updated_source_jobs.append(job)
    atomic_json(Path(root)/"review_source/postflight_manifest.json",dict(old_source,jobs=updated_source_jobs))
    loaded=load_physics_profile(root);runtime=physics_runtime_identity(loaded);profile=metadata.public_profile(loaded)
    packages=[]
    for old in read_json(parent/"validation/postflight_manifest.json")["jobs"]:
        run=old["run"];package=Path(root)/"validation/run_packages"/run;job=deepcopy(old)
        job.update(candidate_id=Path(root).name+"__ffac__"+run+"__rigcamera_"+inference["pipeline_sha256"][:12],
            candidate_folder=str(root),review_output_dir=str(Path(root)/"source_alias"),
            analysis_source_review_root=str(Path(root)/"source_alias"),phase_map_output_dir=str(package/"review_assets"),
            params_sha256=inference["pipeline_sha256"],optical_model_sha256=inference["model_identity"][1],
            physics_runtime=runtime,output_mode="review",phase_model_only=True,
            raw_phase_map_output_dir=None,numeric_phase_output_dir=None,presence_diagnostics_dir=None,
            native_manifest_sha256=sources[run]["native_manifest"]["sha256"],presentation_status="ready",proposals_status="deferred")
        job["quality_flags"]=dict(job.get("quality_flags") or {},approved_g3_rig_camera=True,
            approved_g3_downstream=True,human_review_required=True,mass_analysis_allowed=False,scientific_acceptance=False)
        if inference.get("floor_contact_rule") is True:
            job["quality_flags"]["floor_contact_rule"] = True
        if inference.get("l1_neighbourhood_rule") is True:
            job["quality_flags"]["l1_neighbourhood_rule"] = True
        for key in ("summary","mass_plot","contour_movie","per_frame_csv"):job[key]=None
        common=dict(schema=1,format="darsia-phase-validation-campaign-v1",campaign_id="phase-validation-"+Path(root).name,
            model_id=inference["model_identity"][0],model_sha256=inference["model_identity"][1],model_root=str(root),
            source_manifest_sha256=sha256_file(Path(root)/"review_source"/run/"postflight_manifest.json"),
            pipeline_sha256=inference["pipeline_sha256"],physics_runtime=runtime,physics_profile=profile,
            output_mode="review",numeric_outputs=False,phase_display_policy=metadata.REVISION_CONTRACT)
        atomic_json(package/"postflight_manifest.json",dict(common,jobs=[job],source_campaign_root=str(Path(root)/"review_source"/run)))
        atomic_json(package/"validation_binding.json",dict(common,combined_root=str(package),ffac_root=str(package),ffum_root=str(package),
            selected_candidates={run:job["candidate_id"]},run_count=1,completed_utc=inference["created_utc"]))
        defer_validation_references(package);packages.append(package)
    return packages


def seal_rig_candidate(root,inference,completed,source_runs):
    required={(row["run"],row["frame_index"]) for row in inference["source_inventory"]}
    if len(completed)!=864 or {(row["run"],row["frame_index"]) for row in completed}!=required or len(source_runs)!=43:
        raise ValueError("Rig candidate cannot seal partial/overlapping source or phase coverage")
    metadata.validate_campaign_bindings(source_runs)
    revision=dict(contract=metadata.REVISION_CONTRACT,spec_sha256=SPEC_SHA256,
        geometry_manifest=inference["binding"]["original_geometry_copy"],
        geometry_acceptance=inference["binding"]["staged_geometry_acceptance"],
        geometry_proofs=inference["binding"]["staged_geometry_manifest"],
        phase_grid=inference["binding"]["phase_grid"],sources=source_runs)
    manifest_path=Path(root)/"rig_camera_revision/manifest.json";atomic_json(manifest_path,revision)
    profile=metadata.build_public_profile(inference["binding"]["parent_model_identity"],inference["inherited_public_profile"],
        artifact(manifest_path),spec_sha256=SPEC_SHA256,phase_shape_hw=inference["binding"]["phase_grid"]["shape_hw"],
        heads={key:artifact(Path(root)/name) for key,name in (
            ("co2","models/generic/co2_presence_signal.json"),("g3","models/g3/gas_head.json"),("regional","regional_model.json"))})
    atomic_json(Path(root)/"phase_rig_camera.json",profile)
    rows=[];frame_assets=[]
    by_run={row["run"]:row for row in source_runs}
    for item in sorted(completed,key=lambda row:(row["run"],row["frame_index"])):
        frame_path=bound_path(item["receipt"]);receipt=read_json(frame_path)
        if receipt.get("complete") is not True or receipt["binding"]["pipeline_sha256"]!=inference["pipeline_sha256"]:
            raise ValueError("Final frame changed its source or inference policy")
        for value in [*receipt["maps"].values(),receipt["labels_map"],receipt["rig_domain_revision"]["area_map"],receipt["rig_domain_revision"]["classes"]]:
            bound_path(value)
        rows.append(dict(receipt["qa"],native_manifest_sha256=by_run[item["run"]]["native_manifest"]["sha256"]))
        frame_assets.append(item)
    camera=read_json(bound_path(inference["upstream_plan"]["binding"]["camera_plan"]))
    original_label_audit=camera.get("label_audit")
    evidence=read_json(bound_path(original_label_audit))
    if evidence.get("labels_unchanged") is not True or evidence.get("labels_reindexed_or_rewritten") not in (False,0):
        raise ValueError("Camera source-label audit permits moved or rewritten annotations")
    audit_copy=Path(root)/"production_qa/archived_ac31_source_label_audit.json"
    copy_file_verified(bound_path(original_label_audit),audit_copy)
    label_audit=dict(contract="ffac-source-owned-camera-label-audit-v1",labels_unchanged=True,
        labels_reindexed_or_rewritten=0,camera_plan=inference["upstream_plan"]["binding"]["camera_plan"],
        archived_source_label_audit=artifact(audit_copy),source_sha_owned=True)
    label_path=Path(root)/"production_qa/source_label_audit.json";atomic_json(label_path,label_audit)
    header=dict(contract=metadata.QA_CONTRACT,all_pass=True,run_count=43,frame_count=864,
        configuration=metadata.CONFIGURATION,spec_sha256=SPEC_SHA256,g3_sha256=G3_SHA256,
        regional_refit_sha256=REFIT_SHA256,co2_sha256=CO2_SHA256,labels_changed=False,
        model_identity=inference["model_identity"],phase_grid=inference["binding"]["phase_grid"],
        geometry_acceptance=inference["binding"]["staged_geometry_acceptance"],
        geometry_manifest_sha256=inference["binding"]["geometry_manifest"]["sha256"],source_label_audit=artifact(label_path))
    qa_path=Path(root)/"production_qa/rig_camera_qa.json";atomic_json(qa_path,dict(header,rows=rows))
    from darsia.fluidflower.phase_ffac_rig_qa import nearest_effect_rows
    effects=[];ac31=[]
    for run in sorted(by_run):
        run_rows=[row for row in rows if row["run"]==run]
        effects.extend(nearest_effect_rows(run_rows))
        if run=="ac31":ac31=nearest_effect_rows(run_rows,(1.75,2.,2.25,2.75,3.,4.,5.))
    atomic_json(Path(root)/"production_qa/camera_target_effects.json",dict(
        contract="ffac-rig-camera-nearest-target-effects-v1",model_identity=inference["model_identity"],
        exact_actual_frames=True,interpolation=False,qa=artifact(qa_path),rows=effects,ac31_detail_rows=ac31))
    from .register_ffac_rig_camera_candidate import validate_qa
    validate_qa(dict(header,rows=rows))
    packages=bind_rig_packages(root,inference,source_runs);assemble_saved_candidate(Path(root),packages,inference)
    seal=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),model_identity=inference["model_identity"],
        pipeline_sha256=inference["pipeline_sha256"],run_count=43,frame_count=864,spec_sha256=SPEC_SHA256,
        parent_model_identity=inference["binding"]["parent_model_identity"],
        parent_registration=inference["upstream_plan"]["binding"]["parent_registration"],frames=frame_assets,
        qa=artifact(qa_path),source_plan_sha256=sha256_file(Path(root)/"rig_camera_candidate_plan.json"),
        inference_plan=artifact(Path(root)/PLAN_FILE),phase_profile_sha256=sha256_file(Path(root)/"phase_rig_camera.json"),
        studio_preparation_sha256=sha256_file(Path(root)/"studio_preparation.json"),
        cpu_only=True,retraining_performed=False,labels_changed=False,mass_eligible=False,
        scientific_acceptance=False,automatic_activation=False)
    atomic_json(Path(root)/"rig_camera_candidate_receipt.json",seal)
    return seal


def run_inference(inference,*,workers=24,pilot=False):
    root=Path(inference.get("output_root",inference["upstream_plan"]["binding"]["candidate"]))
    workers=admitted_workers(workers)
    rows=[row for row in inference["source_inventory"] if not pilot or
          (row["run"]=="ac28" and row["frame_index"] in (8,16))]
    groups={run:[row for row in rows if row["run"]==run] for run in sorted({row["run"] for row in rows})}
    # Sparse ring storage bounds extra I/O; labels+RGBA+classes remain full-grid.
    preflight_space(root,10*GIB+len(rows)*10*1024**2)
    completed=[];sources=[]
    with JobLease(root,inference["job_sha256"]):
        below_normal_priority()
        progress_folder=root/"inference_progress"
        progress_folder.mkdir(parents=True,exist_ok=True)
        log=ProgressLog(progress_folder)
        log.event("rig_inference_start",workers=workers,threads=1,frames=len(rows),cpu_only=True)
        with ProcessPoolExecutor(max_workers=min(workers,len(groups)),mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer) as pool:
            futures={pool.submit(infer_run,str(root),inference,group):run for run,group in groups.items()}
            for future in as_completed(futures):
                try:
                    result=future.result()
                except BaseException as error:
                    # Commit the failure before ProcessPoolExecutor drains the
                    # other workers, so external monitors see it immediately.
                    run=futures[future]
                    log.event("rig_run_failed",run=run,error=repr(error))
                    failure=root/"inference_failures"/(run+".json")
                    if not failure.exists():
                        atomic_json(failure,dict(contract="ffac-rig-inference-run-failure-v1",
                            run=run,failed_utc=utc_now(),error=repr(error),complete=False,
                            inference_plan=artifact(root/PLAN_FILE),automatic_retry=False))
                    raise
                if result["run"]!=futures[future]:raise ValueError("Rig worker returned another run")
                completed.extend(result["frames"])
                if result["source"] is not None:sources.append(result["source"])
                log.event("rig_run_complete",run=result["run"],completed_frames=len(completed),total_frames=len(rows))
        if pilot:
            result=dict(contract=CONTRACT,pilot=True,complete=True,model_identity=inference["model_identity"],
                inference_plan=artifact(root/PLAN_FILE),frames=completed,registration_performed=False)
            atomic_json(root/"rig_camera_pilot_receipt.json",result);return result
        result=seal_rig_candidate(root,inference,completed,sources)
        log.event("rig_candidate_complete",runs=43,frames=864)
        return result


def create_probe_plan(prepared_plan,geometry_manifest,geometry_acceptance,probe_output,*,resume=False):
    """Bind a detached two-frame diagnostic; never approve partial production."""
    plan_path=Path(prepared_plan);plan=read_json(plan_path)
    if (plan.get("job_sha256")!=canonical_sha256(plan.get("binding")) or
        plan["binding"].get("spec_sha256")!=SPEC_SHA256 or
        (plan.get("run_count"),plan.get("frame_count"))!=(43,864)):
        raise ValueError("Partial-input probe needs the exact immutable43/864 scientific plan")
    gate=require_inference_geometry(geometry_manifest,geometry_acceptance,plan)
    frames,references=validate_geometry_inputs(gate,plan)
    science_root=Path(plan["binding"]["candidate"]);root=Path(probe_output).resolve()
    parent=Path(plan["binding"]["parent"])
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_rig_domain_inference import require_frozen_heads
    require_frozen_heads(parent);bound_path(plan["binding"]["parent_registration"])
    if list(_model_identity(parent))!=plan["binding"]["parent_model_identity"]:
        raise ValueError("Partial-input probe parent identity changed")
    if root==science_root or root.is_relative_to(science_root) or root==parent or root.is_relative_to(parent):
        raise ValueError("A partial-input probe must be detached from both production and parent candidates")
    tasks={row["task_id"]:row for row in plan["source_inventory"]["scientific_tasks"]}
    selected=[row for row in plan["source_inventory"]["frames"]
              if row["run"]=="ac28" and row["frame_index"] in (8,16)]
    if len(selected)!=2:raise ValueError("Probe frame inventory must be exactlyAC28f8/f16")
    reference=next(row for row in plan["source_inventory"]["references"] if row["run"]=="ac28")
    required={reference["scientific_task_id"]}
    for row in selected:
        required.add(row["scientific_task_id"])
        if row["previous_scientific_task_id"] is not None:required.add(row["previous_scientific_task_id"])
    prepared=[]
    for task_id in sorted(required):
        verify_prepared_source(science_root,tasks[task_id],plan["job_sha256"])
        prepared.append(artifact(science_root/"scientific_sources"/task_id/"prepared_source.json"))
    binding=dict(contract=CONTRACT,probe_only=True,partial_upstream=True,publication_allowed=False,
        registration_allowed=False,upstream_plan=artifact(plan_path),scientific_sources=prepared,
        spec_sha256=SPEC_SHA256,code=_code_binding(),geometry_manifest=gate["geometry_manifest"],
        geometry_acceptance=gate["geometry_acceptance"],phase_grid=metadata.phase_grid(gate["geometry"]["phase_shape_hw"]),
        parent_model_identity=plan["binding"]["parent_model_identity"],cpu_only=True,training=False,labels_changed=False)
    destination=root/"rig_camera_probe_plan.json"
    if destination.exists():
        old=read_json(destination)
        if not resume or old.get("binding")!=binding:raise ValueError("Probe resume changed its exact source/code/geometry")
        return old
    if root.exists():raise ValueError("Refusing an existing output without the exact detached probe plan")
    root.mkdir(parents=True)
    sources=[dict(row,rig_geometry=frames[(row["run"],row["frame_index"])],
                retained_source_cache=retained_binding(parent,row,tasks[row["scientific_task_id"]])) for row in selected]
    probe=dict(binding=binding,job_sha256=canonical_sha256(binding),pipeline_sha256=canonical_sha256(binding),
        scientific_source_root=str(science_root),probe_output=str(root),upstream_plan=plan,
        model_identity=plan["binding"]["parent_model_identity"],source_inventory=sources,
        geometry_sources=[dict(row,rig_geometry=frames[(row["run"],row["frame_index"])]) for row in plan["source_inventory"]["frames"]],
        references=[dict(reference,rig_geometry=references["ac28"])],auxiliary_sources=gate["geometry"].get("auxiliary_sources",[]))
    atomic_json(destination,probe);return probe


def run_probe(probe):
    root=Path(probe["probe_output"])
    with JobLease(root,probe["job_sha256"]):
        worker_initializer();log=ProgressLog(root)
        log.event("partial_input_probe_start",frames=2,workers=1,registration_allowed=False)
        result=infer_run(str(root),probe,probe["source_inventory"])
        for item in result["frames"]:
            receipt=read_json(bound_path(item["receipt"]))
            if receipt["qa"].get("all_pass") is not True:
                raise ValueError("Partial-input probe failed its technical source/interior QA")
        receipt=dict(contract=CONTRACT,probe_only=True,complete=True,created_utc=utc_now(),
            partial_upstream=True,publication_allowed=False,registration_allowed=False,
            probe_plan=artifact(root/"rig_camera_probe_plan.json"),frames=result["frames"])
        atomic_json(root/"rig_camera_probe_receipt.json",receipt)
        log.event("partial_input_probe_complete",frames=2)
        return receipt


def edge_camera_time_summary(rows):
    """Separate true camera-clock negative controls from early post-injection.

    The selected index is not evidence of pre-injection status. In particular,
    the nominal first0.25h frame is already post-injection and may contain CO2.
    Zero flags remain literal measurements, independent of technical validity.
    """
    groups={"pre_injection":[],"post_injection":[]}
    for row in rows:
        elapsed=float(row["elapsed_h"])
        if not np.isfinite(elapsed):raise ValueError("Edge control needs its finite camera-clock elapsed_h")
        groups["pre_injection" if elapsed<0 else "post_injection"].append(row)
    def summarize(values):
        return dict(frame_count=len(values),run_count=len({row["run"] for row in values}),
            all_added_co2_zero=all(row["added_strip_by_side"]["added_co2_cm2"]==0 for row in values),
            outer_photo_extension_co2_zero=all(sum(side["outer_photo_extension_co2_cm2"] for side in
                row["added_strip_by_side"]["sides"].values())==0 for row in values))
    return dict(negative_control_rule="camera elapsed_h < 0; never selected frame index",
        post_injection_zero_required=False,**{name:summarize(values) for name,values in groups.items()})


def create_edge_preflight_plan(prepared_plan,geometry_manifest,geometry_acceptance,output,*,resume=False):
    """Declare all43 initial pairs as a detached diagnostic, never a candidate."""
    state=load_scientific_preparation(prepared_plan);plan=state["plan"]
    gate=require_inference_geometry(geometry_manifest,geometry_acceptance,plan)
    frames,references=validate_geometry_inputs(gate,plan)
    root=Path(output).resolve();science=Path(plan["binding"]["candidate"]);parent=Path(plan["binding"]["parent"])
    if root==science or root.is_relative_to(science) or root==parent or root.is_relative_to(parent):
        raise ValueError("Edge preflight must be detached from production and parent candidates")
    from darsia.fluidflower.phase_rig_domain_inference import require_frozen_heads
    from darsia.fluidflower.phase_validation import _model_identity
    require_frozen_heads(parent);bound_path(plan["binding"]["parent_registration"])
    if list(_model_identity(parent))!=plan["binding"]["parent_model_identity"]:
        raise ValueError("Edge preflight parent model identity changed")
    selected=[row for row in plan["source_inventory"]["frames"] if row["frame_index"] in (0,1)]
    if len(selected)!=86 or len({row["run"] for row in selected})!=43 or any(
            {row["frame_index"] for row in selected if row["run"]==run}!={0,1} for run in {row["run"] for row in selected}):
        raise ValueError("Edge preflight requires exactly43 source-bound initial frame pairs")
    if any(not np.isfinite(float(row["elapsed_h"])) for row in selected):
        raise ValueError("Edge preflight requires actual finite camera-clock times")
    if any(float(row["elapsed_h"])>=0 for row in selected if row["frame_index"]==0):
        raise ValueError("All43 selected frame0 controls must actually precede injection on the camera clock")
    binding=dict(contract="ffac-rig-camera-detached-edge-preflight-v1",probe_only=True,edge_preflight_only=True,
        partial_upstream=False,publication_allowed=False,registration_allowed=False,
        upstream_plan=state["plan_asset"],scientific_preparation=state["preparation_asset"],
        spec_sha256=SPEC_SHA256,code=_code_binding(),geometry_manifest=gate["geometry_manifest"],
        geometry_acceptance=gate["geometry_acceptance"],phase_grid=metadata.phase_grid(gate["geometry"]["phase_shape_hw"]),
        parent_model_identity=plan["binding"]["parent_model_identity"],cpu_only=True,training=False,labels_changed=False,
        negative_control_rule="actual camera elapsed_h < 0; frame1 is not assumed pre-injection",
        frame0_pre_injection_verified=True)
    destination=root/"rig_camera_edge_preflight_plan.json"
    if destination.exists():
        old=read_json(destination)
        if not resume or old.get("binding")!=binding:raise ValueError("Edge preflight resume changed exact sources/code/geometry")
        return old
    if root.exists():raise ValueError("Refusing an existing edge output without its immutable diagnostic plan")
    root.mkdir(parents=True)
    tasks={row["task_id"]:row for row in plan["source_inventory"]["scientific_tasks"]}
    sources=[dict(row,rig_geometry=frames[(row["run"],row["frame_index"])],
        retained_source_cache=retained_binding(parent,row,tasks[row["scientific_task_id"]])) for row in selected]
    result=dict(binding=binding,job_sha256=canonical_sha256(binding),pipeline_sha256=canonical_sha256(binding),
        scientific_source_root=str(science),preflight_output=str(root),upstream_plan=plan,
        model_identity=plan["binding"]["parent_model_identity"],source_inventory=sources,
        geometry_sources=[dict(row,rig_geometry=frames[(row["run"],row["frame_index"])]) for row in plan["source_inventory"]["frames"]],
        references=[dict(row,rig_geometry=references[row["run"]]) for row in plan["source_inventory"]["references"]],
        auxiliary_sources=gate["geometry"].get("auxiliary_sources",[]))
    atomic_json(destination,result);return result


def run_edge_preflight(preflight,*,workers=24):
    root=Path(preflight["preflight_output"]);workers=admitted_workers(workers)
    groups={run:[row for row in preflight["source_inventory"] if row["run"]==run]
        for run in sorted({row["run"] for row in preflight["source_inventory"]})}
    completed=[]
    with JobLease(root,preflight["job_sha256"]):
        below_normal_priority();log=ProgressLog(root)
        log.event("edge_preflight_start",runs=43,frames=86,workers=workers,threads=1,registration_allowed=False)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),initializer=worker_initializer) as pool:
            futures={pool.submit(infer_run,str(root),preflight,group):run for run,group in groups.items()}
            for future in as_completed(futures):
                result=future.result()
                if result["run"]!=futures[future] or result["source"] is not None:
                    raise ValueError("Detached edge worker produced a candidate source package or wrong run")
                completed.extend(result["frames"])
                log.event("edge_preflight_run",run=result["run"],completed_frames=len(completed),total_frames=86)
        completed.sort(key=lambda row:(row["run"],row["frame_index"]))
        expected={(run,index) for run in groups for index in (0,1)}
        if len(completed)!=86 or {(row["run"],row["frame_index"]) for row in completed}!=expected:
            raise ValueError("Detached edge preflight lost or duplicated an initial frame")
        rows=[read_json(bound_path(row["receipt"]))["qa"] for row in completed]
        receipt=dict(contract=preflight["binding"]["contract"],complete=True,created_utc=utc_now(),
            preflight_plan=artifact(root/"rig_camera_edge_preflight_plan.json"),frames=completed,rows=rows,
            run_count=43,frame_count=86,technical_all_pass=all(row["all_pass"] for row in rows),
            publication_allowed=False,registration_allowed=False,scientific_acceptance=False,mass_eligible=False,
            all_added_co2_zero=all(row["added_strip_by_side"]["added_co2_cm2"]==0 for row in rows),
            outer_photo_extension_co2_zero=all(sum(side["outer_photo_extension_co2_cm2"] for side in
                row["added_strip_by_side"]["sides"].values())==0 for row in rows),
            edge_co2_zero=all(sum(side["outer_photo_extension_co2_cm2"] for side in
                row["added_strip_by_side"]["sides"].values())==0 for row in rows),
            camera_time_cohorts=edge_camera_time_summary(rows),
            edge_zero_scope="extension beyond frozen parent photo domain; internal support recovery reported separately")
        atomic_json(root/"rig_camera_edge_preflight_receipt.json",receipt)
        log.event("edge_preflight_complete",frames=86,edge_co2_zero=receipt["edge_co2_zero"])
        return receipt


def infer_frame(root, inference, source, *, processor=None):
    """Frozen fresh inference, then declared parent cache, then flagged fill."""
    from darsia.fluidflower.phase_rig_domain_inference import infer_native, fill_rig_labels, interior_difference
    from darsia.fluidflower.phase_retained_source_inference import (
        prepare_retained_source_cache,merge_retained_source_inference,GAS_RULES,
        preserve_prior_categorical_fill)
    from darsia.fluidflower.phase_gas_barrier_rule import apply_barrier_rule_b
    from darsia.fluidflower.phase_ffac_rig_qa import frame_area_report
    root=Path(root); run,index=source["run"],source["frame_index"]
    path=root/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    binding=dict(pipeline_sha256=inference["pipeline_sha256"],source=source)
    if inference["binding"].get("probe_only") is True and "scientific_preparation" not in inference["binding"]:
        # A detached probe certifies only the declared completed sources. It
        # must never impersonate the complete910-source preparation seal.
        binding["partial_scientific_sources_sha256"]=canonical_sha256(inference["binding"]["scientific_sources"])
    else:
        binding["upstream_preparation_sha256"]=inference["binding"]["scientific_preparation"]["sha256"]
    if path.exists():
        old=read_json(path)
        if old.get("complete") is not True or old.get("binding")!=binding:
            raise ValueError("Frame resume differs from its immutable source/geometry/cache policy")
        for asset in [*old["maps"].values(),old["labels_map"],old["rig_domain_revision"]["area_map"],old["rig_domain_revision"]["classes"]]:
            bound_path(asset)
        return dict(run=run,frame_index=index,receipt=artifact(path))
    if inference.get("recovery"):
        from .recover_ffac_rig_camera_inference import import_completed_frame
        imported=import_completed_frame(root,inference,source)
        if imported is not None:return imported
    current_receipt=_prepared(root,inference,source["scientific_task_id"])
    current,current_valid,audit=load_scientific(current_receipt)
    from darsia.fluidflower.phase_ffac_rig_raw_dispatch import create_rig_raw_source_processor
    processor=processor or create_rig_raw_source_processor(current_receipt["task"]["parent_native_manifest"]["path"])
    full,current_support,current_extended=extended_scientific(root,inference,current_receipt,
        source["rig_geometry"],processor)
    reference=next(row for row in inference["references"] if row["run"]==run)
    baseline_receipt=_prepared(root,inference,reference["scientific_task_id"])
    baseline,baseline_valid,baseline_extended=extended_scientific(root,inference,baseline_receipt,
        reference["rig_geometry"],processor)
    h_b=np.asarray(current_receipt["geometry_frame"]["H_v5_B_to_frame_native"])
    h_b_full=canvas_homography(h_b)
    shape=inference["binding"]["phase_grid"]["shape_hw"]
    aligned_baseline,b_support=align_scientific(baseline,baseline_valid,h_b_full,target_shape_hw=shape)
    previous=None;previous_support=np.ones(shape,bool)
    if source["previous_scientific_task_id"] is not None:
        previous_receipt=_prepared(root,inference,source["previous_scientific_task_id"])
        previous_geometry=task_geometry(inference,previous_receipt["task"])
        previous_log,previous_valid,_=extended_scientific(root,inference,previous_receipt,previous_geometry,processor)
        h_previous=np.asarray(previous_receipt["geometry_frame"]["H_v5_B_to_frame_native"])
        previous,previous_support=align_scientific(previous_log,previous_valid,
            canvas_homography(h_b@np.linalg.inv(h_previous)),target_shape_hw=shape)
    initial,observed=geometry_context(source["rig_geometry"],current_valid,audit)
    allowed=_crop(initial.rig_domain)
    cache=prepare_retained_source_cache(source["retained_source_cache"],run=run,
        frame_index=source["parent_source_frame_index"] if not source["source_changed"] else index,
        current_source_sha256=source["source_sha256"],current_log=current,allowed_domain=allowed,producer_audit=audit)
    fresh,support,fresh_audit=infer_native(inference["upstream_plan"]["binding"]["parent"],
        full,aligned_baseline,initial.rig_domain,aligned_previous_log=previous,
        pixel_area_mm2=initial.pixel_area_mm2,source_baseline_log=baseline,
        h_baseline_to_current=h_b_full,source_baseline_valid=baseline_valid)
    # The context contains the photograph-only top and explicitly excluded
    # hardware. This removes categories only; no second component minimum.
    support &= initial.rig_domain&current_support&b_support&previous_support
    fresh[~support]=255
    components,source_ids,barrier_geometry_audit=prepare_barrier_geometry(root,inference,source,initial.rig_domain)
    gas_before_rule=fresh==2
    fresh,barrier_audit=apply_barrier_rule_b(fresh,components["open_components"],
        source_ids,pixel_area_mm2=.027053,
        barrier_mask=components["barrier_mask"])
    removed=gas_before_rule&(fresh!=2)
    barrier_audit["removed_gas_area_variable_cm2"]=float(initial.pixel_area_mm2[removed].sum(dtype=np.float64)/100)
    merged,masks,cache_audit=merge_retained_source_inference(cache,_crop(fresh),_crop(support),
        pixel_area_mm2=_crop(initial.pixel_area_mm2),gas_rules=GAS_RULES)
    x,y,width,height=metadata.SOURCE_BOX
    fresh[y:y+height,x:x+width]=merged
    support[y:y+height,x:x+width]=masks["modelled"]
    observed |= support
    context=initial
    fill_audit={}
    labels,classes=fill_rig_labels(fresh,context.rig_domain,observed,
        hidden=context.hardware_excluded,tie_break_reference=_canvas(cache.labels,phase_shape_hw=shape),
        tie_break_mask=_canvas(cache.tie_reference_eligible,False,phase_shape_hw=shape),audit=fill_audit)
    native_labels,populations,prior_fill_audit=preserve_prior_categorical_fill(cache,_crop(labels),
        modelled=_crop(classes["modelled"]),filled_observed=_crop(classes["filled_observed"]),
        filled_unobserved=_crop(classes["filled_unobserved"]))
    labels[y:y+height,x:x+width]=native_labels
    packed=io.BytesIO()
    np.savez_compressed(packed,shape_hw=np.asarray(metadata.SOURCE_SHAPE,np.int64),
        **{name:np.packbits(mask,bitorder="little") for name,mask in populations.items()})
    prior_fill_asset=_write_bytes(Path(root)/"rig_camera_revision/inherited_fill"/run/
        f"frame_{index:03d}.packbits.npz",packed.getvalue())
    descriptor=_context_assets(root,run,f"frame_{index:03d}",context,classes)
    maps,proof=write_maps(root,run,index,inference["pipeline_sha256"],labels,phase_shape_hw=shape)
    with Image.open(proof["path"]) as image:
        persisted=np.array(image)
    parent=Path(inference["upstream_plan"]["binding"]["parent"])
    old_index=source["parent_source_frame_index"] if not source["source_changed"] else index
    parent_frame=read_json(parent/"barrier_rule_b_frames"/run/f"frame_{old_index:03d}.json")
    with Image.open(bound_path(parent_frame["labels_map"])) as image:
        before=np.array(image)
    native=read_json(current_receipt["task"]["parent_native_manifest"]["path"])
    old_photo_row=native["frame_sources"][old_index]
    old_photo=np.load(bound_path(dict(path=old_photo_row["v5_photo_sand_mask_path"],
                                    sha256=old_photo_row["v5_photo_sand_mask_sha256"])),allow_pickle=False)
    before=before.copy();before[~old_photo]=255
    inside=interior_difference(before,_crop(persisted),old_photo,(.00016448,.00016448),
                               source_changed=source["source_changed"])
    if not source["source_changed"] and not inside["interior_exact"]:
        raise ValueError("Persisted unchanged-source interior differs more than30mm from the old map edge")
    hidden_chart=np.zeros(shape,bool)
    for name in HIDDEN_CHART_NAMES:hidden_chart |= context.excluded_by_name[name]
    area_report=frame_area_report(_canvas(before,phase_shape_hw=shape),persisted,context.pixel_area_mm2,
        dict(rig_domain=context.rig_domain,**classes),hardware_masks=context.excluded_by_name,hidden_chart_mask=hidden_chart)
    from darsia.fluidflower.phase_ffac_rig_edge_qa import added_strip_report
    added_strips=added_strip_report(_canvas(before,phase_shape_hw=shape),persisted,context.pixel_area_mm2,
        dict(rig_domain=context.rig_domain,**classes),
        inner_polygon_native=source["rig_geometry"]["frame_inner_polygon_native"],
        photo_top_keep=load_array(source["rig_geometry"]["frame_photo_top_keep"]),
        native_origin_xy=metadata.ORIGIN_XY,parent_photo_domain=_canvas(old_photo,False,phase_shape_hw=shape))
    source_window=np.zeros_like(observed);x,y,w,h=metadata.SOURCE_BOX;source_window[y:y+h,x:x+w]=True
    qa=dict(run=run,frame_index=index,elapsed_h=source["elapsed_h"],source_sha256=source["source_sha256"],
        camera_time_role="pre-injection" if float(source["elapsed_h"])<0 else "post-injection",
        source_changed=source["source_changed"],timeline_camera_error_h=0.,
        parent_valid_bits_preserved=(audit.get("parent_cache") or {}).get("parent_float_bytes_preserved") is True,
        interior_changed_gt30mm_pixels=None if source["source_changed"] else inside["changed_pixels"]["over_30_mm"],
        interior_difference=inside,phase_grid=inference["binding"]["phase_grid"],
        geometry_receipt_sha256=current_receipt["geometry_receipt"]["sha256"],
        area_map_sha256=descriptor["area_map"]["sha256"],classes_sha256=descriptor["classes"]["sha256"],
        map_path=proof["path"],map_sha256=proof["sha256"],studio_map_path=maps["combined"]["path"],
        studio_map_sha256=maps["combined"]["sha256"],area_report=area_report,
        source_window_unsupported_observed_pixels=int(np.count_nonzero(context.rig_domain&observed&~source_window&~support)),
        pixel_area_mm2_fixed_ac28=.027053,nearest_fill_audit=fill_audit,
        added_strip_by_side=dict(added_strips,source_changed=source["source_changed"]),
        all_added_co2_zero=added_strips["added_co2_cm2"]==0,
        outer_photo_extension_co2_zero=sum(side["outer_photo_extension_co2_cm2"]
            for side in added_strips["sides"].values())==0,all_pass=True)
    receipt=dict(contract=FRAME_CONTRACT,complete=True,created_utc=utc_now(),binding=binding,qa=qa,maps=maps,labels_map=proof,
        rig_domain_revision=descriptor,fresh_inference=fresh_audit,retained_source_cache=cache_audit,
        rule_b=barrier_audit,rule_b_geometry=barrier_geometry_audit,
        full_extended_scientific=current_extended,baseline_extended_scientific=baseline_extended,
        inherited_categorical_fill=dict(audit=prior_fill_audit,populations=prior_fill_asset,
            provenance="source-bound parent categorical fill; never modelled or training support"),
        cpu_only=True,training=False,labels_changed=False)
    atomic_json(path,receipt)
    return dict(run=run,frame_index=index,receipt=artifact(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-plan",required=True)
    parser.add_argument("--geometry-manifest",required=True)
    parser.add_argument("--geometry-acceptance",required=True)
    parser.add_argument("--workers",type=int,default=24)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--probe-output",help="Detached AC28f8/f16 diagnostic from exact completed source receipts; no production seal")
    parser.add_argument("--edge-preflight-output",help="Detached all43 initial-frame edge diagnostic; no candidate seal or registration")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args(argv)
    if args.edge_preflight_output:
        if args.probe_output or args.pilot:raise ValueError("Select one detached diagnostic mode")
        preflight=create_edge_preflight_plan(args.prepared_plan,args.geometry_manifest,args.geometry_acceptance,args.edge_preflight_output,resume=args.resume)
        if args.run:run_edge_preflight(preflight,workers=args.workers)
        print("Detached43-run edge preflight complete." if args.run else "Detached edge preflight validated.",flush=True)
        return 0
    if args.probe_output:
        if args.pilot:raise ValueError("Use either production pilot or detached partial-input probe")
        probe=create_probe_plan(args.prepared_plan,args.geometry_manifest,args.geometry_acceptance,args.probe_output,resume=args.resume)
        if args.run:run_probe(probe)
        print("Detached partial-input probe complete." if args.run else "Detached source-bound probe validated.",flush=True)
        return 0
    inference=create_inference_plan(args.prepared_plan,args.geometry_manifest,args.geometry_acceptance,resume=args.resume)
    if args.dry_run:
        print("Accepted source/geometry plan validated; no inference submitted.",flush=True)
        return 0
    result=run_inference(inference,workers=args.workers,pilot=args.pilot)
    print("Rig camera pilot complete." if args.pilot else "Rig camera candidate complete; not activated.",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
