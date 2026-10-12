"""Source-bound stored G3 map neighbourhood correction, floor completion and saved Studio candidate.

The sealed parent CO2 union is reused explicitly, never called fresh inference.
No labels, probabilities, RAW sidecars or model inference are read. Only flagged categorical relabelling and floor interpretation are new.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib
import io
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import (JobLease, atomic_json, canonical_sha256,
    copy_file_verified, read_json, sha256_file, utc_now, below_normal_priority)
from .run_ffac_rig_camera_candidate import artifact, worker_initializer
from .distribute_ffac_t2_l1zone_candidate import resource_admission

CONTRACT = "ffac-g3-l1-neighbourhood-floor-candidate-v1"
KIND = "ffac_l1_neighbourhood_floor_review"
PLAN = "l1_neighbourhood_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
HIDDEN = frozenset(("ac26", "ac55"))
PARENT_HIDDEN = frozenset(("ac30", "ac49", "ac54"))
RUN_DIRECTORY = "l1_neighbourhood_runs"
SPEC_SHA256 = "7135a087e9c02267336a6ad31f2bf2a04e7d2fcd66a868b5fe96cef2651ade17"
WARNING = ("Studio review candidate: base G3 gas + L1 neighbourhood rule + Floor contact. CO2 union inherited exactly except flagged "
    "floor-contact completion. Bottom blocked for mass; AC26/55 withheld pending the new candidate-bound blind check. "
    "Rounds 2-4 regression acceptance pending Claude's sealed scorer. Not approved for mass analysis.")


def code_binding():
    names = ("darsia.fluidflower.tools.run_ffac_l1_neighbourhood_candidate",
        "darsia.fluidflower.phase_l1_neighbourhood_gas",
        "darsia.fluidflower.phase_rig_bottom_review_policy", "darsia.fluidflower.phase_ffac_floor_contact",
        "darsia.fluidflower.phase_gas_barrier_rule", "darsia.fluidflower.phase_pixel_area",
        "darsia.fluidflower.phase_ffac_rig_camera_runtime", "darsia.fluidflower.phase_ffac_rig_camera_metadata",
        "darsia.fluidflower.tools.run_ffac_rig_camera_inference")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def plan_rule_settings(plan):
    """Read the approved rule variant from its immutable specification binding."""
    from darsia.fluidflower.phase_l1_neighbourhood_gas import approved_spec_settings
    binding = plan["binding"]
    settings = approved_spec_settings(binding["spec"]["sha256"])
    inputs = binding["inputs"]
    threshold = inputs["stored_map_rule"]["gas_fraction"]
    if (type(threshold) not in (int, float)
            or threshold != settings["gas_fraction_threshold"]
            or inputs["spec"]["sha256"] != settings["spec_sha256"]):
        raise ValueError("L1 neighbourhood plan lost its approved threshold/specification binding")
    return settings


def _frame_rule_binding_matches(frame, settings):
    """The first 0.3 receipts predate explicit variant fields; 0.5 requires them."""
    legacy = settings["spec_sha256"] == SPEC_SHA256
    return (frame.get("gas_fraction_threshold", .3 if legacy else None) == settings["gas_fraction_threshold"]
            and frame.get("spec_sha256", SPEC_SHA256 if legacy else None) == settings["spec_sha256"])


def native_source_bindings(plan):
    """The new plan carries only the verified native-alias inventory it needs.

    Historical plans keep their existing parent-plan loader. The SHA remains
    inside the immutable input binding and is checked before the compact map
    can replace parsing the large parent JSON in each worker.
    """
    native=plan.get("native_sources")
    expected=plan["binding"]["inputs"].get("native_sources_sha256")
    if native is None and expected is None:
        return None
    if (not isinstance(native,dict) or set(native)!={s["run"] for s in plan["source_inventory"]}
            or expected!=canonical_sha256(native)):
        raise ValueError("L1 neighbourhood native-source map lost its immutable binding")
    return native



def validate_head_revision(parent, pixel, regional):
    from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    root = Path(parent)
    if (sha256_file(root / "models/generic/co2_presence_signal.json") != CO2_SHA256
            or sha256_file(pixel) != G3_SHA256 or sha256_file(regional) != REFIT_SHA256):
        raise ValueError("L1 neighbourhood requires the original frozen G3/refit/CO2 heads, never T3")



def validate_runtime_heads(candidate, marker, heads, *, verified_plan=None):
    """Dedicated opt-in; original rig profiles keep their original exact hashes."""
    from darsia.fluidflower.phase_g3_production import CO2_SHA256
    root = Path(candidate)
    if marker.get("contract") != CONTRACT or marker.get("frozen_co2_sha256") != CO2_SHA256:
        raise ValueError("L1 neighbourhood runtime marker lost its frozen CO2 identity")
    plan_path = bound_path(marker["plan"], base=root)
    if verified_plan is None:
        plan = read_json(plan_path)
    else:
        # candidate_state has just read this same file. Keep the asset SHA
        # verification above; avoid a second parse of its large inventory.
        if plan_path != (root / PLAN).resolve():
            raise ValueError("Verified L1 neighbourhood plan is not the bound candidate plan")
        plan = verified_plan
    from darsia.fluidflower.phase_validation import _model_identity
    if (plan.get("contract") != CONTRACT or plan.get("output_root") != str(root.resolve())
            or plan.get("model_identity") != list(_model_identity(root))
            or plan.get("job_sha256") != canonical_sha256(plan["binding"])
            or plan.get("pipeline_sha256") != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))):
        raise ValueError("L1 neighbourhood runtime plan identity changed")
    settings = plan_rule_settings(plan)
    if (settings["spec_sha256"] != marker["spec_sha256"]
            or marker.get("gas_fraction_threshold", .3 if marker["spec_sha256"] == SPEC_SHA256 else None)
                != settings["gas_fraction_threshold"]):
        raise ValueError("L1 neighbourhood runtime specification changed")
    if marker["heads"] != heads or plan["binding"]["heads"] != heads:
        raise ValueError("L1 neighbourhood runtime heads differ from the immutable candidate plan")
    for key, relative in (("co2", "models/generic/co2_presence_signal.json"),
                          ("g3", "models/g3/gas_head.json"), ("regional", "regional_model.json")):
        if bound_path(heads[key], base=root) != (root / relative).resolve():
            raise ValueError("L1 neighbourhood head escaped its candidate")
    validate_head_revision(plan["binding"]["parent"], root / "models/g3/gas_head.json", root / "regional_model.json")
    return True


def validate_source_inventory(sources):
    keys=[(s["run"],s["frame_index"]) for s in sources]
    if (len(keys)!=864 or len(set(keys))!=864 or len({k[0] for k in keys})!=43):
        raise ValueError("L1 neighbourhood requires the unique exact43/864 source inventory")


def candidate_state(root):
    root=Path(root).resolve()
    plan=read_json(root/PLAN)
    if (plan.get("contract")!=CONTRACT or plan.get("output_root")!=str(root)
            or plan["binding"]["inputs"]["code"]!=code_binding()
            or plan["job_sha256"]!=canonical_sha256(plan["binding"])
            or plan["pipeline_sha256"]!=canonical_sha256(dict(binding=plan["binding"],model_identity=plan["model_identity"]))
            or canonical_sha256(plan["source_inventory"])!=plan["binding"]["inputs"]["source_inventory_sha256"]):
        raise ValueError("L1 neighbourhood candidate lost its exact code/source/pipeline binding")
    validate_source_inventory(plan["source_inventory"])
    native_source_bindings(plan)
    validate_runtime_heads(root,read_json(root/"profile_template.json")["l1_neighbourhood_floor_revision"],
        plan["binding"]["heads"],verified_plan=plan)
    from darsia.fluidflower.phase_rig_bottom_review_policy import load_policy
    policy=load_policy(root)
    if (policy is None or policy["purpose"]!="ffac-t3-gas-blind-check-v1"
            or set(policy["blind_runs"])!=HIDDEN or policy["source_plan"]!=artifact(root/PLAN)):
        raise ValueError("L1 neighbourhood candidate lost its own new blind-policy binding")
    for key in ("parent_seal","parent_plan","parent_registration","spec"):
        bound_path(plan["binding"]["inputs"][key])
    return root,plan,plan["source_inventory"]


def _write_array(root,directory,run,index,value,source,*,packed=False,contract=None):
    from . import run_ffac_rig_camera_inference as producer
    value=np.asarray(value)
    stream=io.BytesIO()
    np.save(stream,np.packbits(value) if packed else value,allow_pickle=False)
    asset=producer._write_bytes(Path(root)/directory/run/f"frame_{index:03d}.npy",stream.getvalue())
    return dict(asset,frame_index=index,source_sha256=source["source_sha256"],
                shape_hw=list(value.shape),storage="packbits-npy" if packed else "npy",contract=contract)


def prepare_candidate(parent, output, spec, *, resume=False):
    from .register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_g3_production import CO2_SHA256
    parent, root = Path(parent).resolve(), Path(output).resolve()
    if root == parent or root.is_relative_to(parent):
        raise ValueError("A candidate requires a separate absent output namespace")
    from darsia.fluidflower.phase_l1_neighbourhood_gas import approved_spec_settings
    settings = approved_spec_settings(sha256_file(spec))
    if parent.name != "phase-model-20261005T121928Z-g3-old-domain-seam-corrected-ffac":
        raise ValueError("Only the approved old-domain seam corrected base is permitted")
    if "QA FAILED" in registration_warning(parent):
        raise ValueError("Base candidate registration is not intact")
    seal = read_json(parent / SEAL)
    if (seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43,864)
            or seal.get("bottom_mass_allowed") is not False or seal.get("physical_scale_approved") is not True):
        raise ValueError("Base requires complete sealed43/864 and approved scale with bottom mass block")
    validate_head_revision(parent, parent / "models/g3/gas_head.json", parent / "regional_model.json")
    parent_plan = read_json(parent / "old_domain_seam_plan.json")
    sources = parent_plan["source_inventory"]
    native_sources = deepcopy(parent_plan["native_sources"])
    validate_source_inventory(sources)
    if set(native_sources)!={s["run"] for s in sources}:
        raise ValueError("Base native-source aliases differ from its exact run inventory")
    inputs = dict(source_inventory_sha256=canonical_sha256(sources), parent=str(parent),
        native_sources_sha256=canonical_sha256(native_sources),
        parent_seal=artifact(parent / SEAL), parent_plan=artifact(parent / "old_domain_seam_plan.json"),
        parent_registration=artifact(registration_path(parent.name)), spec=artifact(spec), code=code_binding(),
        cpu_only=True, threads_per_worker=1, bottom_mass_allowed=False, mass_eligible=False,
        blind_hold_runs=sorted(HIDDEN), output_root=str(root),
        co2_policy="exact stored base union plus separately flagged floor fill; no inference",
        fixed_configuration=dict(regional_threshold=.6,pixel_threshold=.7,minimum_mm2=2.5,connectivity=4),
        stored_map_rule=dict(activation_h=.935,radius_px=937.5,box_hw=[51,51],
            gas_fraction=settings["gas_fraction_threshold"],iterations=1),
        training=False, model_inference_performed=False, annotations_read=False)
    if (root / PLAN).exists():
        old = read_json(root / PLAN)
        if not resume or old["binding"]["inputs"] != inputs:
            raise ValueError("Resume changes source/model/code or fixed settings")
        return old
    if root.exists() or resume:
        raise ValueError("Use an absent candidate output or exact --resume")
    root.mkdir(parents=True,exist_ok=False)
    for path in (parent / "models").rglob("*"):
        if path.is_file(): copy_file_verified(path,root / path.relative_to(parent))
    for name in ("regional_model.json","training_request.json","expected_review_runs.json",
                 "physical_scale_measurement.json","vertical_scale_check.json"):
        if (parent / name).exists(): copy_file_verified(parent/name,root/name)
    copy_file_verified(spec,root / "approved_l1_neighbourhood_spec.md")
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=utc_now(),
        display_name=("FF_AC · G3 gas · L1 neighbourhood rule" +
            (" 0.5" if settings["gas_fraction_threshold"] == .5 else "") + " · Floor contact · 07.10.2026"),
        inherited_base_id=parent.name,training_performed_in_this_revision=False,optical_training_performed=False,
        regional_training_performed=False,co2_training_performed=False,mass_eligible=False,
        scientific_acceptance=False,automatic_activation=False,bottom_mass_allowed=False,
        model_inference_performed=False,regression_status="pending Claude sealed scorer",l1_neighbourhood_rule=True,
        gas_fraction_threshold=settings["gas_fraction_threshold"],l1_neighbourhood_spec_sha256=settings["spec_sha256"])
    atomic_json(root/"summary.json",summary)
    heads={key:artifact(root/relative) for key,relative in (("co2","models/generic/co2_presence_signal.json"),
        ("g3","models/g3/gas_head.json"),("regional","regional_model.json"))}
    binding=dict(inputs=inputs,parent=str(parent),spec=inputs["spec"],heads=heads)
    identity=list(_model_identity(root))
    plan=dict(contract=CONTRACT,created_utc=summary["created_utc"],binding=binding,source_inventory=sources,
        native_sources=native_sources,
        job_sha256=canonical_sha256(binding),pipeline_sha256=canonical_sha256(dict(binding=binding,model_identity=identity)),
        model_identity=identity,output_root=str(root))
    atomic_json(root/PLAN,plan)
    from darsia.fluidflower.phase_rig_bottom_review_policy import write_pending_policy
    write_pending_policy(root,source_plan=artifact(root/PLAN),purpose="ffac-t3-gas-blind-check-v1",blind_runs=("ac26","ac55"))
    profile=read_json(parent/"phase_rig_camera.json")
    profile["parent_model_identity"]=list(_model_identity(parent));profile["frozen_heads"]=heads
    profile["l1_neighbourhood_floor_revision"]=dict(contract=CONTRACT,plan=artifact(root/PLAN),
        spec_sha256=settings["spec_sha256"],gas_fraction_threshold=settings["gas_fraction_threshold"],
        frozen_co2_sha256=CO2_SHA256,heads=heads)
    atomic_json(root/"profile_template.json",profile)
    return plan



def _png(asset):
    with Image.open(bound_path(asset)) as image:
        return np.array(image)


def _classes(asset):
    with np.load(bound_path(asset), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def floor_contact_classes(frame_receipt, classes):
    """Add the separately stored hardware mask to the verified rig partition.

    load_rig_classes returns the three coverage classes and their union, not
    hardware_excluded. The floor rule needs that exact producer mask too; its
    absence must not silently become an all-valid hardware assumption.
    """
    path = bound_path(frame_receipt["rig_domain_revision"]["classes"])
    with np.load(path, allow_pickle=False) as archive:
        if "hardware_excluded" not in archive.files:
            raise ValueError("Floor contact requires the bound hardware-exclusion mask")
        hardware = archive["hardware_excluded"].copy()
    if hardware.dtype != bool or hardware.shape != classes["rig_domain"].shape:
        raise ValueError("Floor-contact hardware mask differs from the rig class canvas")
    if ("hardware_excluded" in classes
            and not np.array_equal(classes["hardware_excluded"], hardware)):
        raise ValueError("Floor-contact hardware mask differs from its bound producer asset")
    hardware.flags.writeable = False
    return dict(classes, hardware_excluded=hardware)


def infer_frame(root, plan, parent_plan, source, *, baseline_cache=None):
    from . import run_ffac_rig_camera_inference as producer
    from darsia.fluidflower.phase_l1_neighbourhood_gas import apply_l1_neighbourhood,frozen_rule_b_forbidden_mask,select_frozen_base_rule_b
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact,prepare_bound_floor_zone
    from darsia.fluidflower.phase_pixel_area import load_rig_classes,load_rig_pixel_area
    root,parent=Path(root),Path(plan["binding"]["parent"])
    settings=plan_rule_settings(plan)
    run,index=source["run"],source["frame_index"]
    path=root/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    parent_asset=artifact(parent/"rig_camera_frames"/run/f"frame_{index:03d}.json")
    binding=dict(pipeline_sha256=plan["pipeline_sha256"],source=source,parent_receipt=parent_asset)
    if path.exists():
        old=read_json(path)
        if (old.get("complete") is not True or old.get("contract")!=CONTRACT or old.get("binding")!=binding
                or not _frame_rule_binding_matches(old,settings)):
            raise ValueError("L1 neighbourhood resume binding mismatch")
        for asset in [old["labels_map"],*old["maps"].values(),old["floor_contact_fill"],
                      old["floor_trigger_canvas_y"],old["l1_neighbourhood_gas"]]: bound_path(asset)
        return dict(run=run,frame_index=index,receipt=artifact(path))
    wall,cpu=time.perf_counter(),time.process_time()
    old=read_json(bound_path(parent_asset))
    if old["binding"]["source"]!=source: raise ValueError("Source/timeline changed")
    before=_png(old["labels_map"])
    revision=old["rig_domain_revision"]
    classes,_=load_rig_classes(revision)
    classes=floor_contact_classes(old,classes)
    area=load_rig_pixel_area(revision,before.shape,origin_xy=tuple(revision["native_origin_xy"])).values_mm2
    if not np.array_equal(before==255,~classes["rig_domain"]):
        raise ValueError("Stored base labels changed the rig-domain/outside partition")
    producer_asset=old["source_producer_receipt"]
    original=read_json(bound_path(producer_asset))
    barrier=_classes(original["rule_b_geometry"]["asset"])
    frozen_b,b_source_audit=select_frozen_base_rule_b(old,original)
    forbidden,b_audit=frozen_rule_b_forbidden_mask(barrier["open_components"],barrier["barrier_mask"],frozen_b)
    result=apply_l1_neighbourhood(before,H_ac28_to_frame_native=source["rig_geometry"]["H_ac28_to_frame_native"],
        elapsed_hours=source["elapsed_h"],barrier_mask=barrier["barrier_mask"],rule_b_forbidden_mask=forbidden,
        gas_fraction_threshold=settings["gas_fraction_threshold"])
    pre_floor=result.labels
    top_keep=producer.load_array(source["rig_geometry"]["frame_photo_top_keep"])
    if top_keep.dtype!=bool or top_keep.shape!=before.shape: raise ValueError("Top clip proof has another grid")
    top_changes=int(np.count_nonzero((pre_floor==2)&~top_keep))
    if top_changes: raise ValueError("L1 neighbourhood violates the unchanged top surface")
    fill_support=(classes["filled_observed"]|classes["filled_unobserved"])&classes["rig_domain"]&~classes["hardware_excluded"]
    observed=(classes["modelled"]|classes["filled_observed"])&classes["rig_domain"]&~classes["hardware_excluded"]
    zone,floor_binding=prepare_bound_floor_zone(old,classes,area,observed)
    completed=apply_floor_contact(pre_floor,elapsed_hours=float(source["elapsed_h"]),zone=zone,
        observed_support=observed,filled_support=fill_support,unresolved=~np.isin(pre_floor,(0,1,2)),pixel_area_mm2=area)
    after,flag=completed.labels,completed.floor_contact_fill
    gasflag=result.l1_neighbourhood_gas
    expected=before.copy();expected[gasflag]=2;expected[flag]=1
    if (not np.array_equal(after,expected) or np.any(flag&(before!=0)) or
            not np.array_equal(np.isin(after,(1,2)),np.isin(before,(1,2))|flag) or
            np.any(gasflag&(~result.zone.mask|barrier["barrier_mask"]|forbidden))):
        raise ValueError("Unflagged neighbourhood/floor/CO2 union difference")
    flag_descriptor=_write_array(root,"floor_contact_masks",run,index,flag,source,packed=True,contract="ffac-t2-floor-contact-v1")
    gas_descriptor=_write_array(root,"l1_neighbourhood_masks",run,index,gasflag,source,packed=True,contract="ffac-l1-neighbourhood-gas-v1")
    trigger_descriptor=_write_array(root,"floor_contact_triggers",run,index,completed.trigger_canvas_y,source)
    maps,labels_asset=producer.write_maps(root,run,index,plan["pipeline_sha256"],after,phase_shape_hw=after.shape)
    if not np.array_equal(_png(labels_asset),after): raise ValueError("Stored labels differ from computed labels")
    timing=dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu)
    qa=dict(run=run,frame_index=index,source_sha256=source["source_sha256"],elapsed_h=source["elapsed_h"],
        gas_before_cm2=float(area[before==2].sum(dtype=np.float64)/100),gas_after_cm2=float(area[after==2].sum(dtype=np.float64)/100),
        co2_before_cm2=float(area[np.isin(before,(1,2))].sum(dtype=np.float64)/100),
        co2_after_cm2=float(area[np.isin(after,(1,2))].sum(dtype=np.float64)/100),
        l1_neighbourhood_gas_pixels=int(gasflag.sum()),l1_neighbourhood_gas_cm2=float(area[gasflag].sum(dtype=np.float64)/100),
        changed_pixels=int(np.count_nonzero(before!=after)),union_changes_outside_fill=0,rule_b_changed_pixels=0,
        top_clip_changed_pixels=top_changes,unchanged_unresolved_and_rig_domain=True,
        floor_contact_fill_pixels=int(flag.sum()),preinject_fill_pixels=int(flag.sum()) if source["elapsed_h"]<=0 else 0,
        all_pass=True,**timing)
    receipt=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),binding=binding,run=run,frame_index=index,
        gas_fraction_threshold=settings["gas_fraction_threshold"],spec_sha256=settings["spec_sha256"],
        source_sha256=source["source_sha256"],elapsed_h=source["elapsed_h"],before_labels=old["labels_map"],labels_map=labels_asset,
        maps=maps,floor_contact_fill=flag_descriptor,l1_neighbourhood_gas=gas_descriptor,
        floor_trigger_canvas_y=trigger_descriptor,floor_rule=completed.report,floor_geometry=floor_binding,
        l1_rule=result.report,l1_point_xy=list(result.zone.point_xy),l1_rule_active=result.zone.active,
        frozen_rule_b_spatial_decision=b_audit,frozen_rule_b_source_selection=b_source_audit,
        rule_b=frozen_b,rule_b_geometry=original["rule_b_geometry"],
        top_clip=dict(changed_pixels=0,check_only=True,asset=source["rig_geometry"]["frame_photo_top_keep"]),
        rig_domain_revision=revision,full_extended_scientific=old["full_extended_scientific"],
        baseline_extended_scientific=old["baseline_extended_scientific"],qa=qa,timing=timing,
        cpu_only=True,labels_changed=False,training=False,mass_eligible=False,bottom_mass_allowed=False,
        co2_cache_provenance=parent_asset,co2_inference_performed=False,new_gas_inference_performed=False,
        annotations_read=False,source_producer_receipt=producer_asset)
    atomic_json(path,receipt)
    return dict(run=run,frame_index=index,receipt=artifact(path))



def validate_run_result(root,plan,sources,result):
    settings=plan_rule_settings(plan)
    expected=[s for s in sources if s["run"]==result.get("run")]
    if (result.get("complete") is not True or result.get("frame_count")!=len(expected)
            or result.get("pipeline_sha256")!=plan["pipeline_sha256"]
            or len(result.get("receipts",[]))!=len(expected)):
        raise ValueError("L1 neighbourhood run result is incomplete or lost its pipeline binding")
    for source,item in zip(expected,result["receipts"]):
        frame=read_json(bound_path(item["receipt"]))
        if (item["run"]!=source["run"] or item["frame_index"]!=source["frame_index"]
                or frame.get("contract")!=CONTRACT or frame.get("complete") is not True
                or frame["binding"]["source"]!=source or frame["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]
                or not _frame_rule_binding_matches(frame,settings)
                or not frame["qa"]["all_pass"] or frame["qa"]["union_changes_outside_fill"]
                or frame["qa"]["preinject_fill_pixels"] or frame["qa"]["rule_b_changed_pixels"]
                or frame["qa"]["top_clip_changed_pixels"]):
            raise ValueError("L1 neighbourhood frame failed its exact source/union/floor/top/RuleB binding")
        for asset in [frame["labels_map"],frame["floor_contact_fill"],frame["floor_trigger_canvas_y"],frame["l1_neighbourhood_gas"],*frame["maps"].values()]:
            bound_path(asset)
    for key in ("native_manifest","input_manifest"):
        bound_path(result[key])
    return True


def run_one_run(root, run, indices=None, *, verified_state=None):
    worker_initializer()
    if verified_state is None:
        root, plan, inventory = candidate_state(root)
    else:
        # Internal same-process handoff from a caller that already validated
        # candidate_state. Every frame still verifies its source and assets.
        checked_root, plan, inventory = verified_state
        root=Path(root).resolve()
        if (Path(checked_root).resolve()!=root or plan.get("output_root")!=str(root)
                or inventory!=plan.get("source_inventory")):
            raise ValueError("Verified L1 neighbourhood state belongs to another candidate")
    native_sources=native_source_bindings(plan)
    parent_plan=(dict(native_sources=native_sources) if native_sources is not None
        else read_json(bound_path(plan["binding"]["inputs"]["parent_plan"])))
    sources = [s for s in inventory if s["run"] == run and (indices is None or s["frame_index"] in indices)]
    results = []
    baseline_cache = {}
    for source in sources:
        item = infer_frame(root, plan, parent_plan, source, baseline_cache=baseline_cache)
        results.append(item)
        if run not in HIDDEN:
            print(json.dumps(dict(event="l1_frame_complete", run=run, frame_index=source["frame_index"])), flush=True)
    if indices is not None:
        return results
    native_source = parent_plan["native_sources"][run]
    native = read_json(bound_path(native_source["native_manifest"]))
    inputs = read_json(bound_path(native_source["input_manifest"]))
    if len(results) != len(native["frame_sources"]) or len(results) != len(inputs["frames"]):
        raise ValueError("L1 neighbourhood source assembly is incomplete")
    for item, record, binding in zip(results, native["frame_sources"], inputs["frames"]):
        frame = read_json(bound_path(item["receipt"]))
        if (item["run"] != run or item["frame_index"] != record["frame_index"]
                or frame["source_sha256"] != record["source_sha256"]
                or binding["source_sha256"] != record["source_sha256"]):
            raise ValueError("L1 neighbourhood floor flag changed its source-owned frame order")
        for name in ("floor_contact_fill", "l1_neighbourhood_gas"):
            record[name] = frame[name]
            binding[name] = frame[name]
    input_path = root / "source_alias/source_bindings" / (run + ".json")
    atomic_json(input_path, inputs)
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    native = meta.build_native_metadata(read_json(bound_path(inputs["parent_native_manifest"])),
        read_json(bound_path(inputs["camera_timeline_run"])), native["frame_sources"], inputs["frames"], inputs["reference"],
        input_manifest=artifact(input_path))
    native_path = root / "source_alias/native_frames" / run / "manifest.json"
    atomic_json(native_path, native)
    result = dict(run=run, complete=True, native_manifest=artifact(native_path), input_manifest=artifact(input_path),
        frames=inputs["frames"], reference=inputs["reference"], frame_count=len(results), receipts=results,
        pipeline_sha256=plan["pipeline_sha256"])
    validate_run_result(root,plan,inventory,result)
    atomic_json(root / RUN_DIRECTORY / (run + ".json"), result)
    return result


def seal_candidate(root, plan, source_runs):
    from . import run_ffac_rig_camera_inference as producer
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    root, checked_plan, sources = candidate_state(root)
    if plan != checked_plan:
        raise ValueError("L1 neighbourhood seal plan differs from the exact saved pipeline")
    settings=plan_rule_settings(plan)
    parent = Path(plan["binding"]["parent"])
    if len(source_runs)!=43 or len({row["run"] for row in source_runs})!=43:
        raise ValueError("L1 neighbourhood seal requires one complete result for every run")
    for row in source_runs:
        validate_run_result(root,plan,sources,row)
    meta.validate_campaign_bindings(source_runs)
    profile = read_json(root / "profile_template.json")
    runtime = read_json(bound_path(profile["rig_domain_revision"]["manifest"]))
    runtime["sources"] = [{k: v for k, v in row.items() if k != "receipts"} for row in sorted(source_runs, key=lambda s: s["run"])]
    runtime_path = root / "rig_camera_revision/runtime_manifest.json"
    atomic_json(runtime_path, runtime)
    profile["rig_domain_revision"]["manifest"] = artifact(runtime_path)
    atomic_json(root / "phase_rig_camera.json", profile)
    packages = producer.bind_rig_packages(root, dict(plan, floor_contact_rule=True, l1_neighbourhood_rule=True,
        upstream_plan=dict(binding=dict(parent=str(parent)))), source_runs)
    assemble_saved_candidate(root, packages, plan)
    frames = [f for row in source_runs for f in row["receipts"]]
    if len(frames) != 864:
        raise ValueError("Cannot seal partial L1 neighbourhood maps")
    qa_rows = [read_json(bound_path(f["receipt"]))["qa"] for f in frames]
    if any(not r["all_pass"] or r["union_changes_outside_fill"] or r["preinject_fill_pixels"] or r["rule_b_changed_pixels"] or r["top_clip_changed_pixels"] for r in qa_rows):
        raise ValueError("L1 neighbourhood exact-union/floor QA failed")
    qa_path = root / "production_qa/l1_neighbourhood_qa.json"
    atomic_json(qa_path, dict(contract=CONTRACT, technical_all_pass=True, regression_acceptance="pending sealed scorer",
        run_count=43, frame_count=864, rows=qa_rows,**settings))
    seal = dict(contract=CONTRACT, complete=True, model_identity=plan["model_identity"], pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root / PLAN),
        run_count=43, frame_count=864, frames=frames, plan=artifact(root / PLAN), qa=artifact(qa_path),
        parent_seal=plan["binding"]["inputs"]["parent_seal"], heads=plan["binding"]["heads"],
        physical_scale_approved=True, labels_changed=False, mass_eligible=False, bottom_mass_allowed=False,
        blind_hold_runs=sorted(HIDDEN), automatic_activation=False, scientific_acceptance=False,
        **settings,
        regression_acceptance="pending Claude sealed scorer", co2_union_unchanged_except_floor_contact_fill=True)
    atomic_json(root / SEAL, seal)
    register(root)
    return seal


def registered_warning(root, registration):
    try:
        root = Path(root)
        from darsia.fluidflower.phase_validation import _model_identity
        if (registration.get("registration_kind") != KIND or registration.get("model_sha256") != _model_identity(root)[1]
                or registration.get("bottom_mass_allowed") is not False or registration.get("mass_eligible") is not False):
            raise ValueError("Changed L1 neighbourhood registration")
        seal=read_json(root/SEAL)
        if (seal.get("complete") is not True or (seal.get("run_count"),seal.get("frame_count"))!=(43,864)
                or set(seal.get("blind_hold_runs",[]))!=HIDDEN
                or seal.get("bottom_mass_allowed") is not False or seal.get("mass_eligible") is not False):
            raise ValueError("Changed L1 neighbourhood final seal/hold scope")
        for relative, digest in registration["metadata_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("Changed L1 neighbourhood candidate metadata")
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: L1 neighbourhood floor review registration changed or is incomplete; do not use for mass."


def register(root):
    from .register_track1_studio_candidate import FORMAT, registration_path
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_runtime import load_physics_profile
    root = Path(root); seal = read_json(root / SEAL)
    if seal.get("complete") is not True or (seal["run_count"], seal["frame_count"]) != (43, 864):
        raise ValueError("L1 neighbourhood registration requires complete 43/864")
    load_physics_profile(root)
    files = ("summary.json", "training_request.json", "phase_rig_camera.json", PLAN, SEAL,
        "review_source/postflight_manifest.json", "validation/postflight_manifest.json", "validation/validation_binding.json",
        "studio_preparation.json", "rig_bottom_review_policy.json", "production_qa/l1_neighbourhood_qa.json")
    identity = _model_identity(root)
    native, maps = {}, {}
    for item in seal["frames"]:
        frame = read_json(bound_path(item["receipt"]))
        for a in [frame["labels_map"], frame["floor_contact_fill"], frame["floor_trigger_canvas_y"],frame["l1_neighbourhood_gas"], *frame["maps"].values()]:
            bound_path(a)
        maps[f"{item['run']}/frame_{item['frame_index']:03d}"] = frame["maps"]["combined"]["sha256"]
    for path in (root / "source_alias/native_frames").glob("*/manifest.json"):
        native[path.parent.name] = sha256_file(path)
    value = dict(format=FORMAT, registration_kind=KIND, model_root=str(root), model_id=identity[0], model_sha256=identity[1],
        production_receipt_sha256=sha256_file(root / SEAL), source_plan_sha256=sha256_file(root / PLAN),
        run_count=43, frame_count=864, native_run_manifest_sha256=native, map_sha256=maps,
        metadata_sha256={name: sha256_file(root / name) for name in files}, publication_scope="studio_validation_only",
        warning=WARNING, qa_status="technical_pass_regression_pending", scientific_acceptance=False, mass_eligible=False,
        deployment_allowed=False, automatic_activation=False, physical_scale_approved=True, bottom_mass_allowed=False,
        created_utc=utc_now())
    dest = registration_path(root.name)
    if dest.exists():
        old = read_json(dest)
        value["created_utc"] = old["created_utc"]
        if old != value:
            raise ValueError("Refusing to overwrite a different L1 neighbourhood registration")
    atomic_json(dest, value)
    return dest


def run(plan, *, workers=24, smoke_only=False, runs=None, memory_per_worker_mb=2048, memory_headroom_mb=2048):
    root = Path(plan["output_root"])
    if plan["binding"]["inputs"]["code"] != code_binding():
        raise ValueError("Candidate code changed after the immutable plan")
    below_normal_priority()
    import socket
    lease_root=root/"operations"/(socket.gethostname()+"-"+canonical_sha256(runs or ["all"])[:12])
    with JobLease(lease_root, plan["job_sha256"]):
        if smoke_only:
            result = run_one_run(str(root), "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=result, plan=artifact(root / PLAN)))
            return result
        if not (root / "smoke_receipt.json").exists():
            result = run_one_run(str(root), "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=result, plan=artifact(root / PLAN)))
        native_sources=native_source_bindings(plan)
        if native_sources is None:
            native_sources=read_json(bound_path(plan["binding"]["inputs"]["parent_plan"]))["native_sources"]
        all_runs=sorted(native_sources)
        partition = runs is not None
        runs=all_runs if runs is None else sorted(runs)
        if not runs or len(set(runs))!=len(runs) or not set(runs).issubset(all_runs):
            raise ValueError("Owner partition has missing, repeated or unknown runs")
        rows = []
        import os, psutil
        process=psutil.Process()
        cpus=len(process.cpu_affinity()) if hasattr(process,"cpu_affinity") else (os.cpu_count() or 1)
        admission=resource_admission(workers,len(runs),memory_per_worker_mb,memory_headroom_mb,
            logical_cpus=cpus,available_bytes=psutil.virtual_memory().available)
        pool = ProcessPoolExecutor(max_workers=admission["admitted_workers"],
            mp_context=multiprocessing.get_context("spawn"), initializer=worker_initializer)
        futures = {pool.submit(run_one_run, str(root), run): run for run in runs}
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                from .run_ffac_t2_floor_pipeline import write_telemetry
                write_telemetry(lease_root / "progress.json", dict(stage="inference", pid=__import__('os').getpid(), updated_utc=utc_now(),
                    complete=False, runs_done=len(rows), frames_done=sum(row["frame_count"] for row in rows), total_runs=43, total_frames=864))
        except BaseException as error:
            from .revise_ffac_old_domain_seam import _stop_failed_pool
            _stop_failed_pool(pool, futures)
            atomic_json(root / "failure_receipt.json", dict(error=repr(error), updated_utc=utc_now(), automatic_retry=False))
            raise
        else:
            pool.shutdown(wait=True)
        if partition:
            atomic_json(lease_root/"owner_receipt.json",dict(complete=True,runs=runs,
                frame_count=sum(row["frame_count"] for row in rows),plan=artifact(root/PLAN)))
            return rows
        seal = seal_candidate(root, plan, rows)
        from .run_ffac_t2_floor_pipeline import write_telemetry
        write_telemetry(lease_root / "progress.json", dict(stage="complete", complete=True, runs_done=43, frames_done=864, updated_utc=utc_now()))
        return seal


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","output","spec"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--workers",type=int,default=24)
    p.add_argument("--memory-per-worker-mb",type=float,default=2048)
    p.add_argument("--memory-headroom-mb",type=float,default=2048)
    p.add_argument("--resume",action="store_true");p.add_argument("--smoke-only",action="store_true")
    p.add_argument("--plan-only",action="store_true");p.add_argument("--seal-only",action="store_true")
    p.add_argument("--runs",help="Comma-separated disjoint owner partition; no seal performed")
    a=p.parse_args(argv);plan=prepare_candidate(a.parent,a.output,a.spec,resume=a.resume)
    if a.seal_only:
        root=Path(plan["output_root"])
        rows=[read_json(root/RUN_DIRECTORY/(run+".json")) for run in sorted({s["run"] for s in plan["source_inventory"]})]
        seal_candidate(root,plan,rows)
    elif not a.plan_only:
        run(plan,workers=a.workers,smoke_only=a.smoke_only,runs=a.runs.split(",") if a.runs else None,
            memory_per_worker_mb=a.memory_per_worker_mb,memory_headroom_mb=a.memory_headroom_mb)



if __name__ == "__main__":
    main()
