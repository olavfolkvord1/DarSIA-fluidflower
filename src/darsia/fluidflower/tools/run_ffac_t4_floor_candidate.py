"""Source-bound CPU G3-T4 inference, floor completion and saved Studio candidate.

The sealed parent CO2 union is reused explicitly, never called fresh inference.
Only the gas partition and separately flagged floor interpretation are new.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

CONTRACT = "ffac-g3-t4-floor-candidate-v1"
KIND = "ffac_g3_t4_floor_review"
PLAN = "t4_floor_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
HIDDEN = frozenset()
RUN_DIRECTORY = "t4_floor_runs"
SPEC_SHA256 = "28040816088dfb4d08d9ce0cf53708f5bfe497adf57f36891b8da9a116e7a39d"
PARENT_ID = "phase-model-20261005T121928Z-g3-old-domain-seam-corrected-ffac"
WARNING = ("Studio review candidate: G3-T4 gas with separate light epochs + Floor contact. "
    "CO2 union inherited exactly except flagged floor-contact completion. All runs are visible; "
    "bottom blocked for mass. Claude's sealed regression and overflow checks pending. "
    "Not approved for mass analysis.")
FIXED_CONFIGURATION = dict(regional_threshold=.6, pixel_threshold=.7, minimum_mm2=2.5, connectivity=4)



def code_binding():
    names = ("darsia.fluidflower.tools.run_ffac_t4_floor_candidate", "darsia.fluidflower.phase_gas_t4_regional_inputs",
        "darsia.fluidflower.phase_gas_t4_training", "darsia.fluidflower.phase_gas_t2_regional_inputs",
        "darsia.fluidflower.phase_gas_t4_regional_refit",
        "darsia.fluidflower.phase_ffac_floor_contact", "darsia.fluidflower.phase_gas_barrier_rule",
        "darsia.fluidflower.phase_l1_neighbourhood_gas", "darsia.fluidflower.phase_gas_regional_context",
        "darsia.fluidflower.phase_ffac_rig_camera_runtime", "darsia.fluidflower.phase_ffac_rig_camera_metadata",
        "darsia.fluidflower.tools.run_ffac_rig_camera_inference")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def validate_head_revision(parent, pixel, regional, epoch):
    from darsia.fluidflower.phase_g3_production import CO2_SHA256
    from darsia.fluidflower.phase_gas_t4_regional_inputs import verify_pixel_model
    if sha256_file(Path(parent) / "models/generic/co2_presence_signal.json") != CO2_SHA256:
        raise ValueError("Frozen CO2 head changed")
    old, new = read_json(Path(parent) / "regional_model.json"), read_json(regional)
    for value in (old, new):
        value["heads"].pop("gas_within_co2")
    if old != new:
        raise ValueError("Regional refit changed more than the gas head")
    verify_pixel_model(pixel, sha256_file(pixel), light_epoch_id=epoch)


def validate_runtime_heads(candidate, marker, heads, *, verified_plan=None):
    from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_gas_t4_training import load_light_epochs
    root = Path(candidate).resolve()
    plan_path = bound_path(marker["plan"], base=root)
    plan = read_json(plan_path) if verified_plan is None else verified_plan
    if (plan_path != root / PLAN or marker.get("contract") != CONTRACT
            or marker.get("frozen_co2_sha256") != CO2_SHA256
            or plan.get("contract") != CONTRACT or plan.get("output_root") != str(root)
            or plan.get("model_identity") != list(_model_identity(root))
            or plan.get("job_sha256") != canonical_sha256(plan["binding"])
            or plan.get("pipeline_sha256") != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))):
        raise ValueError("T4 runtime plan identity changed")
    if plan["binding"]["spec"]["sha256"] != marker["spec_sha256"] or marker["spec_sha256"] != SPEC_SHA256:
        raise ValueError("T4 runtime specification changed")
    if heads != marker.get("heads") or heads != plan["binding"]["heads"] or heads["co2"]["sha256"] != CO2_SHA256:
        raise ValueError("T4 runtime heads differ from the immutable candidate plan")
    for key, relative in (("co2", "models/generic/co2_presence_signal.json"),
            ("g3", "models/g3/gas_head.json"), ("regional", "regional_model.json")):
        if bound_path(heads[key], base=root) != (root / relative).resolve():
            raise ValueError("T4 legacy head escaped its candidate")
        if heads[key]["sha256"] != {"co2": CO2_SHA256, "g3": G3_SHA256, "regional": REFIT_SHA256}[key]:
            raise ValueError("T4 changed an original frozen reference head")
    table = load_light_epochs(bound_path(plan["binding"]["inputs"]["light_epoch_table"]))
    if table["run_to_epoch"] != plan["run_to_epoch"] or set(heads["epochs"]) != set(table["epochs"]):
        raise ValueError("T4 changed project light-epoch membership")
    if marker.get("light_epoch_table") != plan["binding"]["inputs"]["light_epoch_table"]:
        raise ValueError("T4 runtime changed project light-epoch table")
    for epoch, pair in heads["epochs"].items():
        pixel, regional = bound_path(pair["g3"], base=root), bound_path(pair["regional"], base=root)
        if (pixel != (root / f"models/g3/epochs/{epoch}/gas_head.json").resolve()
                or regional != (root / f"regional_models/{epoch}.json").resolve()):
            raise ValueError("T4 epoch head escaped its candidate")
        validate_head_revision(plan["binding"]["parent"], pixel, regional, epoch)
    return True


def validate_source_inventory(sources):
    keys = [(s["run"], s["frame_index"]) for s in sources]
    if len(keys) != 864 or len(set(keys)) != 864 or len({k[0] for k in keys}) != 43:
        raise ValueError("T4 requires the unique exact43/864 source inventory")


def candidate_state(root):
    root = Path(root).resolve()
    plan = read_json(root / PLAN)
    inputs = plan["binding"]["inputs"]
    if (plan.get("contract") != CONTRACT or plan.get("output_root") != str(root)
            or inputs["code"] != code_binding()
            or plan["job_sha256"] != canonical_sha256(plan["binding"])
            or plan["pipeline_sha256"] != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))
            or canonical_sha256(plan["source_inventory"]) != inputs["source_inventory_sha256"]
            or inputs.get("blind_hold_runs") != [] or inputs.get("l1_rules_applied") != []
            or inputs.get("fixed_configuration") != FIXED_CONFIGURATION
            or (root / "rig_bottom_review_policy.json").exists()):
        raise ValueError("T4 candidate lost its exact code/source/fixed-policy binding")
    validate_source_inventory(plan["source_inventory"])
    if set(plan["run_to_epoch"]) != {row["run"] for row in plan["source_inventory"]}:
        raise ValueError("T4 source inventory differs from project epoch runs")
    if canonical_sha256(plan["native_sources"]) != inputs["native_sources_sha256"]:
        raise ValueError("T4 native source aliases changed")
    if (canonical_sha256(plan["source_context_plans"]) != inputs["source_context_plans_sha256"]
            or set(plan["source_context_plans"]) != set(plan["native_sources"])):
        raise ValueError("T4 compact exact-source context plans changed")
    if canonical_sha256(plan["parent_frame_receipts"]) != inputs["parent_frame_receipts_sha256"]:
        raise ValueError("T4 parent frame receipt inventory changed")
    validate_runtime_heads(root, read_json(root / "profile_template.json")["g3_t4_floor_revision"],
        plan["binding"]["heads"], verified_plan=plan)
    for key in ("parent_seal", "parent_plan", "parent_registration", "regional_bundle", "project", "spec", "light_epoch_table"):
        bound_path(inputs[key])
    for epoch in inputs["epochs"].values():
        for asset in epoch.values(): bound_path(asset)
    return root, plan, plan["source_inventory"]


def _write_array(root,directory,run,index,value,source,*,packed=False,contract=None):
    from . import run_ffac_rig_camera_inference as producer
    value=np.asarray(value)
    stream=io.BytesIO()
    np.save(stream,np.packbits(value) if packed else value,allow_pickle=False)
    asset=producer._write_bytes(Path(root)/directory/run/f"frame_{index:03d}.npy",stream.getvalue())
    return dict(asset,frame_index=index,source_sha256=source["source_sha256"],
                shape_hw=list(value.shape),storage="packbits-npy" if packed else "npy",contract=contract)


def prepare_candidate(parent, output, regional_bundle, project, spec, *, resume=False):
    from .register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    from darsia.fluidflower.phase_gas_t4_regional_inputs import verify_epoch_bundle
    parent, root, project = Path(parent).resolve(), Path(output).resolve(), Path(project).resolve()
    if root == parent or root.is_relative_to(parent) or parent.name != PARENT_ID:
        raise ValueError("T4 requires its own namespace and the original seam-corrected base")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("T4 candidate requires the exact approved specification")
    if "QA FAILED" in registration_warning(parent):
        raise ValueError("Base candidate registration is not intact")
    parent_seal = read_json(parent / SEAL)
    if (parent_seal.get("complete") is not True or (parent_seal.get("run_count"), parent_seal.get("frame_count")) != (43, 864)
            or parent_seal.get("bottom_mass_allowed") is not False or parent_seal.get("physical_scale_approved") is not True):
        raise ValueError("T4 requires sealed43/864, approved scale and the bottom mass block")
    for relative, expected in (("models/generic/co2_presence_signal.json", CO2_SHA256),
            ("models/g3/gas_head.json", G3_SHA256), ("regional_model.json", REFIT_SHA256)):
        if sha256_file(parent / relative) != expected:
            raise ValueError("T4 base must retain original G3/regional/common CO2 heads")
    fitted = verify_epoch_bundle(regional_bundle, project, parent)
    table_asset = artifact(fitted["light_epoch_table_path"])
    if table_asset["sha256"] != fitted["light_epoch_table_sha256"]:
        raise ValueError("Project light epoch table changed")
    table = fitted["light_epoch_table"]
    sources, native_sources, parent_frames = [], {}, {}
    from darsia.fluidflower.phase_ffac_rig_camera_metadata import SOURCE_KEY
    for item in parent_seal["frames"]:
        frame = read_json(bound_path(item["receipt"]))
        source = frame["binding"]["source"]
        if (frame.get("complete") is not True or source["run"] != item["run"]
                or source["frame_index"] != item["frame_index"]):
            raise ValueError("T4 base seal differs from its source-owned frame receipt")
        sources.append(source)
        parent_frames[f"{source['run']}/frame_{source['frame_index']:03d}"] = item["receipt"]
    sources.sort(key=lambda s: (s["run"], s["frame_index"]))
    validate_source_inventory(sources)
    for run in sorted({s["run"] for s in sources}):
        native_path = parent / "source_alias/native_frames" / run / "manifest.json"
        native = read_json(native_path)
        input_manifest = native[SOURCE_KEY]["input_manifest"]
        bound_path(input_manifest)
        if [(s["frame_index"], s["source_sha256"]) for s in sources if s["run"] == run] != [
                (s["frame_index"], s["source_sha256"]) for s in native["frame_sources"]]:
            raise ValueError("T4 base native source frame order changed")
        native_sources[run] = dict(native_manifest=artifact(native_path), input_manifest=input_manifest)
    from darsia.fluidflower.phase_gas_t4_regional_refit import compact_candidate_source_plan
    contexts = {run: compact_candidate_source_plan(parent, run,
        [s["frame_index"] for s in sources if s["run"] == run]) for run in native_sources}
    if set(table["run_to_epoch"]) != {s["run"] for s in sources}:
        raise ValueError("T4 light epoch table must cover exactly the saved43-run campaign")
    epoch_inputs = {}
    for epoch, fit in fitted["epochs"].items():
        validate_head_revision(parent, fit["pixel_model_path"], fit["regional_model_path"], epoch)
        epoch_inputs[epoch] = dict(pixel=artifact(fit["pixel_model_path"]), regional=artifact(fit["regional_model_path"]),
            regional_receipt=artifact(fit["regional_fit_receipt_path"]))
    inputs = dict(source_inventory_sha256=canonical_sha256(sources), native_sources_sha256=canonical_sha256(native_sources),
        source_context_plans_sha256=canonical_sha256(contexts), parent=str(parent),
        parent_frame_receipts_sha256=canonical_sha256(parent_frames),
        parent_seal=artifact(parent / SEAL), parent_plan=artifact(parent / "old_domain_seam_plan.json"),
        parent_registration=artifact(registration_path(parent.name)), regional_bundle=artifact(regional_bundle),
        project=table_asset, light_epoch_table=table_asset,
        epochs=epoch_inputs, spec=artifact(spec), code=code_binding(), cpu_only=True, threads_per_worker=1,
        bottom_mass_allowed=False, mass_eligible=False, blind_hold_runs=[], l1_rules_applied=[],
        output_root=str(root), co2_policy="reuse exact SHA-bound seam-parent union; no fresh CO2 inference claimed",
        fixed_configuration=FIXED_CONFIGURATION)
    if (root / PLAN).exists():
        previous = read_json(root / PLAN)
        if not resume or previous["binding"]["inputs"] != inputs:
            raise ValueError("Resume changes sources, models, epoch table, code or fixed policy")
        return previous
    if root.exists() or resume:
        raise ValueError("Use an absent output folder or exact --resume")
    root.mkdir(parents=True, exist_ok=False)
    for source in (parent / "models").rglob("*"):
        if source.is_file(): copy_file_verified(source, root / source.relative_to(parent))
    copy_file_verified(parent / "regional_model.json", root / "regional_model.json")
    for name in ("training_request.json", "expected_review_runs.json", "physical_scale_measurement.json", "vertical_scale_check.json"):
        if (parent / name).exists(): copy_file_verified(parent / name, root / name)
    for epoch, fit in epoch_inputs.items():
        copy_file_verified(bound_path(fit["pixel"]), root / f"models/g3/epochs/{epoch}/gas_head.json")
        copy_file_verified(bound_path(fit["regional"]), root / f"regional_models/{epoch}.json")
    copy_file_verified(spec, root / "approved_t4_floor_spec.md")
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=utc_now(),
        display_name="FF_AC · G3-T4 gas · Separate light epochs · Floor contact · 07.10.2026",
        inherited_base_id=parent.name, training_performed_in_this_revision=True, optical_training_performed=True,
        regional_training_performed=True, co2_training_performed=False, mass_eligible=False, scientific_acceptance=False,
        automatic_activation=False, t4_regional_bundle=inputs["regional_bundle"],
        t4_regression_status="pending Claude sealed scorer and overflow check", bottom_mass_allowed=False,
        blind_hold_runs=[], l1_rules_applied=[], light_epoch_table=table_asset)
    atomic_json(root / "summary.json", summary)
    heads = {key: artifact(root / relative) for key, relative in (("co2", "models/generic/co2_presence_signal.json"),
        ("g3", "models/g3/gas_head.json"), ("regional", "regional_model.json"))}
    heads["epochs"] = {epoch: dict(g3=artifact(root / f"models/g3/epochs/{epoch}/gas_head.json"),
        regional=artifact(root / f"regional_models/{epoch}.json")) for epoch in epoch_inputs}
    binding = dict(inputs=inputs, parent=str(parent), spec=inputs["spec"], heads=heads)
    identity = list(_model_identity(root))
    plan = dict(contract=CONTRACT, created_utc=summary["created_utc"], binding=binding, source_inventory=sources,
        run_to_epoch=table["run_to_epoch"], native_sources=native_sources, source_context_plans=contexts,
        parent_frame_receipts=parent_frames,
        job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)), model_identity=identity, output_root=str(root))
    atomic_json(root / PLAN, plan)
    profile = read_json(parent / "phase_rig_camera.json")
    profile["parent_model_identity"] = list(_model_identity(parent))
    profile["frozen_heads"] = heads
    profile["g3_t4_floor_revision"] = dict(contract=CONTRACT, plan=artifact(root / PLAN),
        spec_sha256=SPEC_SHA256, frozen_co2_sha256=CO2_SHA256, heads=heads, light_epoch_table=table_asset)
    atomic_json(root / "profile_template.json", profile)
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
    from darsia.fluidflower.phase_gas_t4_regional_inputs import evaluate_gas_frame, prepare_frame_inputs
    from darsia.fluidflower.phase_l1_neighbourhood_gas import select_frozen_base_rule_b, frozen_rule_b_forbidden_mask
    from darsia.fluidflower.phase_gas_barrier_rule import apply_barrier_rule_b
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
    root, parent = Path(root), Path(plan["binding"]["parent"])
    run, index = source["run"], source["frame_index"]
    path = root / "rig_camera_frames" / run / f"frame_{index:03d}.json"
    parent_asset = plan["parent_frame_receipts"][f"{run}/frame_{index:03d}"]
    if bound_path(parent_asset) != (parent / "rig_camera_frames" / run / f"frame_{index:03d}.json").resolve():
        raise ValueError("T4 parent frame receipt escaped its sealed base")
    binding = dict(pipeline_sha256=plan["pipeline_sha256"], source=source, parent_receipt=parent_asset)
    if path.exists():
        old = read_json(path)
        if old.get("complete") is not True or old.get("contract") != CONTRACT or old.get("binding") != binding:
            raise ValueError("T4 resume binding mismatch")
        for a in [old["labels_map"], *old["maps"].values(), old["floor_contact_fill"], old["floor_trigger_canvas_y"]]:
            bound_path(a)
        return dict(run=run, frame_index=index, receipt=artifact(path))
    wall, cpu = time.perf_counter(), time.process_time()
    old = read_json(bound_path(parent_asset))
    if old["binding"]["source"] != source:
        raise ValueError("Source sequence changed")
    before = _png(old["labels_map"])
    if before.dtype != np.uint8 or before.ndim != 2:
        raise ValueError("T4 parent labels are not an exact uint8 categorical raster")
    inputs = prepare_frame_inputs(parent, old, original_plan=parent_plan, baseline_cache=baseline_cache)
    classes = inputs.pop("classes")
    area = inputs.pop("pixel_area_mm2")
    if not np.array_equal(before == 255, ~classes["rig_domain"]):
        raise ValueError("T4 cannot change the parent unresolved/rig-domain partition")
    modelled = inputs.pop("modelled")
    inputs.pop("filled_support")
    fill = classes["filled_observed"] | classes["filled_unobserved"]
    epoch = plan["run_to_epoch"][run]
    pair = plan["binding"]["heads"]["epochs"][epoch]
    after, gas_audit, decisions = evaluate_gas_frame(
        parent,
        bound_path(pair["g3"]), bound_path(pair["regional"]), base_labels=before,
        modelled=modelled, pixel_area_mm2=area,
        expected_pixel_sha256=pair["g3"]["sha256"],
        expected_regional_sha256=pair["regional"]["sha256"], filled_support=fill, light_epoch_id=epoch, **inputs)
    producer_asset = parent_plan["source_producer_frames"][f"{run}/frame_{index:03d}"]
    original = read_json(bound_path(producer_asset))
    top_keep = producer.load_array(source["rig_geometry"]["frame_photo_top_keep"])
    if top_keep.dtype != bool or top_keep.shape != before.shape:
        raise ValueError("Top clip proof differs from the exact native source grid")
    top_application = int(np.count_nonzero((after == 2) & ~top_keep))
    after[(after == 2) & ~top_keep] = 1
    barrier = _classes(original["rule_b_geometry"]["asset"])
    frozen_b, b_source_audit = select_frozen_base_rule_b(old, original)
    if (b_source_audit.get("certified_unchanged_seam_bypass") is True
            and bound_path(b_source_audit["source_producer_receipt"]) != bound_path(producer_asset)):
        raise ValueError("T4 Rule B bypass must bind the exact original producer receipt")
    _forbidden, b_geometry_audit = frozen_rule_b_forbidden_mask(barrier["open_components"], barrier["barrier_mask"], frozen_b)
    after, rule_b_application = apply_barrier_rule_b(after, barrier["open_components"],
        original["rule_b_geometry"]["source_component_ids"], pixel_area_mm2=.027053,
        barrier_mask=barrier["barrier_mask"])
    if not np.array_equal(np.isin(before, (1, 2)), np.isin(after, (1, 2))):
        raise ValueError("Gas stage changed frozen CO2 union")
    classes = floor_contact_classes(old, classes)
    observed = (classes["modelled"] | classes["filled_observed"]) & classes["rig_domain"] & ~classes["hardware_excluded"]
    zone, floor_binding = prepare_bound_floor_zone(old, classes, area, observed)
    completed = apply_floor_contact(after, elapsed_hours=float(source["elapsed_h"]), zone=zone,
        observed_support=observed, filled_support=fill & classes["rig_domain"] & ~classes["hardware_excluded"],
        unresolved=~np.isin(after, (0, 1, 2)), pixel_area_mm2=area)
    after, flag = completed.labels, completed.floor_contact_fill
    rule_b_check, rule_b = apply_barrier_rule_b(after, barrier["open_components"],
        original["rule_b_geometry"]["source_component_ids"], pixel_area_mm2=.027053,
        barrier_mask=barrier["barrier_mask"])
    b_changes = int(np.count_nonzero(rule_b_check != after))
    top_changes = int(np.count_nonzero((after == 2) & ~top_keep))
    if b_changes or top_changes or not np.array_equal(before == 255, after == 255):
        raise ValueError("T4 final map violates the unchanged Rule B/top/domain contract")
    if (np.any(flag & np.isin(before,(1,2)))
            or not np.array_equal(np.isin(after,(1,2)),np.isin(before,(1,2)) | flag)):
        raise ValueError("Unflagged CO2 union difference")
    flag_descriptor = _write_array(root,"floor_contact_masks",run,index,flag,source,
        packed=True,contract="ffac-t2-floor-contact-v1")
    trigger_descriptor = _write_array(root,"floor_contact_triggers",run,index,completed.trigger_canvas_y,source)
    maps, labels_asset = producer.write_maps(root, run, index, plan["pipeline_sha256"], after, phase_shape_hw=after.shape)
    with Image.open(bound_path(labels_asset)) as image:
        if not np.array_equal(np.array(image), after):
            raise ValueError("Saved categorical map is not bit-identical")
    import psutil
    memory = psutil.Process().memory_info()
    timing = dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,
        rss_bytes=memory.rss,peak_working_set_bytes=getattr(memory,"peak_wset",memory.rss))
    qa = dict(run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        gas_before_cm2=float(area[before == 2].sum(dtype=np.float64) / 100),
        gas_after_cm2=float(area[after == 2].sum(dtype=np.float64) / 100),
        co2_before_cm2=float(area[np.isin(before, (1, 2))].sum(dtype=np.float64) / 100),
        co2_after_cm2=float(area[np.isin(after, (1, 2))].sum(dtype=np.float64) / 100),
        changed_pixels=int(np.count_nonzero(before != after)), union_changes_outside_fill=0,
        rule_b_changed_pixels=b_changes, top_clip_changed_pixels=top_changes,
        unchanged_unresolved_and_rig_domain=True,
        floor_contact_fill_pixels=int(flag.sum()), preinject_fill_pixels=int(flag.sum()) if source["elapsed_h"] <= 0 else 0,
        all_pass=True, wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu)
    receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
        run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        before_labels=old["labels_map"], labels_map=labels_asset, maps=maps, floor_contact_fill=flag_descriptor,
        floor_rule=completed.report, floor_geometry=floor_binding, gas_inference=gas_audit, rule_b=rule_b,
        light_epoch_id=epoch, epoch_heads=pair, l1_rules_applied=[],
        frozen_base_rule_b=frozen_b, frozen_rule_b_source_selection=b_source_audit, frozen_rule_b_geometry_audit=b_geometry_audit,
        rule_b_geometry=original["rule_b_geometry"],
        rig_domain_revision=old["rig_domain_revision"], full_extended_scientific=old["full_extended_scientific"],
        baseline_extended_scientific=old["baseline_extended_scientific"], qa=qa,
        floor_trigger_canvas_y=trigger_descriptor, timing=timing,
        rule_b_application=rule_b_application, top_clip=dict(changed_pixels=top_changes, check_only=True,
            asset=source["rig_geometry"]["frame_photo_top_keep"]),
        top_clip_application=dict(gas_to_aqueous_pixels=top_application, co2_union_unchanged=True),
        cpu_only=True, labels_changed=False, training=False, mass_eligible=False, bottom_mass_allowed=False,
        co2_cache_provenance=parent_asset, co2_inference_performed=False, new_gas_inference_performed=True)
    atomic_json(path, receipt)
    return dict(run=run, frame_index=index, receipt=artifact(path))


def validate_run_result(root,plan,sources,result):
    expected=[s for s in sources if s["run"]==result.get("run")]
    if (result.get("complete") is not True or result.get("frame_count")!=len(expected)
            or result.get("pipeline_sha256")!=plan["pipeline_sha256"]
            or len(result.get("receipts",[]))!=len(expected)):
        raise ValueError("T4 run result is incomplete or lost its pipeline binding")
    for source,item in zip(expected,result["receipts"]):
        frame=read_json(bound_path(item["receipt"]))
        if (item["run"]!=source["run"] or item["frame_index"]!=source["frame_index"]
                or frame.get("contract")!=CONTRACT or frame.get("complete") is not True
                or frame["binding"]["source"]!=source or frame["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]
                or frame.get("light_epoch_id") != plan["run_to_epoch"][source["run"]]
                or frame.get("epoch_heads") != plan["binding"]["heads"]["epochs"][plan["run_to_epoch"][source["run"]]]
                or frame.get("l1_rules_applied") != []
                or not frame["qa"]["all_pass"] or frame["qa"]["union_changes_outside_fill"]
                or frame["qa"]["preinject_fill_pixels"] or frame["qa"]["rule_b_changed_pixels"]
                or frame["qa"]["top_clip_changed_pixels"]):
            raise ValueError("T4 frame failed its exact source/union/floor/top/RuleB binding")
        for asset in [frame["labels_map"],frame["floor_contact_fill"],frame["floor_trigger_canvas_y"],*frame["maps"].values()]:
            bound_path(asset)
    for key in ("native_manifest","input_manifest"):
        bound_path(result[key])
    return True


def run_one_run(root, run, indices=None):
    worker_initializer()
    root, plan, inventory = candidate_state(root)
    sources = [s for s in inventory if s["run"] == run and (indices is None or s["frame_index"] in indices)]
    parent_plan = plan["source_context_plans"][run]
    results = []
    baseline_cache = {}
    for source in sources:
        item = infer_frame(root, plan, parent_plan, source, baseline_cache=baseline_cache)
        results.append(item)
        if run not in HIDDEN:
            print(json.dumps(dict(event="t4_frame_complete", run=run, frame_index=source["frame_index"])), flush=True)
    if indices is not None:
        return results
    native_source = plan["native_sources"][run]
    native = read_json(bound_path(native_source["native_manifest"]))
    inputs = read_json(bound_path(native_source["input_manifest"]))
    if len(results) != len(native["frame_sources"]) or len(results) != len(inputs["frames"]):
        raise ValueError("T4 source assembly is incomplete")
    for item, record, binding in zip(results, native["frame_sources"], inputs["frames"]):
        frame = read_json(bound_path(item["receipt"]))
        if (item["run"] != run or item["frame_index"] != record["frame_index"]
                or frame["source_sha256"] != record["source_sha256"]
                or binding["source_sha256"] != record["source_sha256"]):
            raise ValueError("T4 floor flag changed its source-owned frame order")
        record["floor_contact_fill"] = frame["floor_contact_fill"]
        binding["floor_contact_fill"] = frame["floor_contact_fill"]
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
        raise ValueError("T4 seal plan differs from the exact saved pipeline")
    parent = Path(plan["binding"]["parent"])
    if len(source_runs)!=43 or len({row["run"] for row in source_runs})!=43:
        raise ValueError("T4 seal requires one complete result for every run")
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
    packages = producer.bind_rig_packages(root, dict(plan, floor_contact_rule=True,
        upstream_plan=dict(binding=dict(parent=str(parent)))), source_runs)
    assemble_saved_candidate(root, packages, plan)
    frames = [f for row in source_runs for f in row["receipts"]]
    if len(frames) != 864:
        raise ValueError("Cannot seal partial T4 maps")
    qa_rows = [read_json(bound_path(f["receipt"]))["qa"] for f in frames]
    if any(not r["all_pass"] or r["union_changes_outside_fill"] or r["preinject_fill_pixels"] or r["rule_b_changed_pixels"] or r["top_clip_changed_pixels"] for r in qa_rows):
        raise ValueError("T4 exact-union/floor QA failed")
    qa_path = root / "production_qa/t4_floor_qa.json"
    atomic_json(qa_path, dict(contract=CONTRACT, technical_all_pass=True, regression_acceptance="pending sealed scorer",
        run_count=43, frame_count=864, rows=qa_rows))
    seal = dict(contract=CONTRACT, complete=True, model_identity=plan["model_identity"], pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root / PLAN),
        run_count=43, frame_count=864, frames=frames, plan=artifact(root / PLAN), qa=artifact(qa_path),
        parent_seal=plan["binding"]["inputs"]["parent_seal"], heads=plan["binding"]["heads"],
        physical_scale_approved=True, labels_changed=False, mass_eligible=False, bottom_mass_allowed=False,
        blind_hold_runs=sorted(HIDDEN), automatic_activation=False, scientific_acceptance=False,
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
            raise ValueError("Changed T4 registration")
        seal=read_json(root/SEAL)
        if (seal.get("complete") is not True or (seal.get("run_count"),seal.get("frame_count"))!=(43,864)
                or set(seal.get("blind_hold_runs",[]))!=HIDDEN
                or seal.get("bottom_mass_allowed") is not False or seal.get("mass_eligible") is not False):
            raise ValueError("Changed T4 final seal/hold scope")
        for relative, digest in registration["metadata_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("Changed T4 candidate metadata")
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: T4 floor review registration changed or is incomplete; do not use for mass."


def register(root):
    from .register_track1_studio_candidate import FORMAT, registration_path
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_runtime import load_physics_profile
    root = Path(root); seal = read_json(root / SEAL)
    if seal.get("complete") is not True or (seal["run_count"], seal["frame_count"]) != (43, 864):
        raise ValueError("T4 registration requires complete 43/864")
    load_physics_profile(root)
    files = ("summary.json", "training_request.json", "phase_rig_camera.json", PLAN, SEAL,
        "review_source/postflight_manifest.json", "validation/postflight_manifest.json", "validation/validation_binding.json",
        "studio_preparation.json", "production_qa/t4_floor_qa.json")
    identity = _model_identity(root)
    native, maps = {}, {}
    for item in seal["frames"]:
        frame = read_json(bound_path(item["receipt"]))
        for a in [frame["labels_map"], frame["floor_contact_fill"], frame["floor_trigger_canvas_y"], *frame["maps"].values()]:
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
            raise ValueError("Refusing to overwrite a different T4 registration")
    atomic_json(dest, value)
    return dest


def run(plan, *, workers=2, smoke_only=False, memory_per_worker_mb=6144, memory_headroom_mb=2048):
    root = Path(plan["output_root"])
    if plan["binding"]["inputs"]["code"] != code_binding():
        raise ValueError("Candidate code changed after the immutable plan")
    below_normal_priority()
    with JobLease(root, plan["job_sha256"]):
        if smoke_only:
            result = run_one_run(str(root), "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=result, plan=artifact(root / PLAN)))
            return result
        if not (root / "smoke_receipt.json").exists():
            result = run_one_run(str(root), "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=result, plan=artifact(root / PLAN)))
        runs = sorted(plan["native_sources"])
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
                write_telemetry(root / "progress.json", dict(stage="inference", pid=__import__('os').getpid(), updated_utc=utc_now(),
                    complete=False, runs_done=len(rows), frames_done=sum(row["frame_count"] for row in rows), total_runs=43, total_frames=864))
        except BaseException as error:
            from .revise_ffac_old_domain_seam import _stop_failed_pool
            _stop_failed_pool(pool, futures)
            atomic_json(root / "failure_receipt.json", dict(error=repr(error), updated_utc=utc_now(), automatic_retry=False))
            raise
        else:
            pool.shutdown(wait=True)
        seal = seal_candidate(root, plan, rows)
        from .run_ffac_t2_floor_pipeline import write_telemetry
        write_telemetry(root / "progress.json", dict(stage="complete", complete=True, runs_done=43, frames_done=864, updated_utc=utc_now()))
        return seal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "regional-bundle", "project", "spec"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--memory-per-worker-mb", type=float, default=6144)
    p.add_argument("--memory-headroom-mb", type=float, default=2048)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    a = p.parse_args(argv)
    plan = prepare_candidate(a.parent, a.output, a.regional_bundle, a.project, a.spec, resume=a.resume)
    if not a.plan_only:
        run(plan, workers=a.workers, smoke_only=a.smoke_only,
            memory_per_worker_mb=a.memory_per_worker_mb, memory_headroom_mb=a.memory_headroom_mb)


if __name__ == "__main__":
    main()
