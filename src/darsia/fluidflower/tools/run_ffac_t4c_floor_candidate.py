"""Frozen-head T4c calibration evaluation and three independent saved candidates.

Calibration and production call the same frame evaluator. Rig-light gas is
never inferred again; its exact base partition receives only the already
source-bound floor completion. No annotations, training or scoring are used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import JobLease, atomic_json, canonical_sha256, copy_file_verified, read_json, sha256_file, utc_now, below_normal_priority
from .run_ffac_rig_camera_candidate import artifact, worker_initializer
from .distribute_ffac_t2_l1zone_candidate import resource_admission
from .run_ffac_t4b_floor_candidate import _png, _classes, _write_array, floor_contact_classes, validate_source_inventory

CONTRACT = "ffac-g3-t4c-floor-candidate-v1"
KIND = "ffac_g3_t4c_floor_review"
PLAN = "t4c_floor_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
RUN_DIRECTORY = "t4c_floor_runs"
HIDDEN = frozenset()
SPEC_SHA256 = "53aad00f20bec56092c160477210613e7849f50467303aa0efd5243d90337daf"
CALIBRATOR_SHA256 = "6d2d36db574965362cb6d183e48e6ac1218dc01eb67e1400cdf2071453597ef4"
FIXED_CONFIGURATION = dict(regional_threshold=.6, pixel_threshold=.7, minimum_mm2=2.5, connectivity=4)
WARNING = ("Studio review candidate: G3-T4c overflow-calibrated new-lamp gas + Floor contact. "
           "Rig-light gas inherited exactly; CO2 union unchanged except flagged floor completion. "
           "All runs visible; bottom blocked for mass. Locked regression, time-transfer, split-half "
           "and placement decisions pending Claude. Not approved for mass analysis.")
PLACEMENT_NAMES = dict(P1="Optical", P2="Neighbourhood-weighted, caps only", P3="T4b boundary")
_VERIFIED_ASSETS = {}


def _verify_bound_asset(asset):
    """Verify immutable inputs once per process, invalidate on file changes."""
    path = bound_path(asset, verify=False)
    stat = path.stat()
    key = (str(path), asset["sha256"], stat.st_size, stat.st_mtime_ns,
           stat.st_ctime_ns, stat.st_ino)
    if key not in _VERIFIED_ASSETS:
        bound_path(asset)
        _VERIFIED_ASSETS[key] = True
    return path


def code_binding():
    names = ("darsia.fluidflower.tools.run_ffac_t4c_floor_candidate", "darsia.fluidflower.phase_gas_t4c_placements",
        "darsia.fluidflower.phase_gas_t2_regional_inputs", "darsia.fluidflower.phase_gas_t4b_regional_inputs",
        "darsia.fluidflower.phase_ffac_floor_contact", "darsia.fluidflower.phase_gas_barrier_rule",
        "darsia.fluidflower.phase_l1_neighbourhood_gas", "darsia.fluidflower.phase_gas_regional_context",
        "darsia.fluidflower.phase_ffac_rig_camera_runtime", "darsia.fluidflower.phase_ffac_rig_camera_metadata",
        "darsia.fluidflower.tools.report_ffac_t2_floor_candidate", "darsia.fluidflower.tools.report_ffac_t4b_floor_candidate",
        "darsia.fluidflower.tools.run_ffac_rig_camera_inference")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def prepare_calibration_config(parent, t4b, spec, facies_map, zone_map):
    from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    from .report_ffac_t4b_floor_candidate import ZONE_POLICY
    from .register_track1_studio_candidate import registration_path
    parent, t4b = Path(parent).resolve(), Path(t4b).resolve()
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("T4c requires the updated locked neighbourhood specification")
    old_seal, joint_seal = read_json(parent / SEAL), read_json(t4b / SEAL)
    joint_plan = read_json(t4b / "t4b_floor_plan.json")
    if (parent.name != "phase-model-20261005T121928Z-g3-old-domain-seam-corrected-ffac"
        or joint_plan["binding"]["parent"] != str(parent)
        or any(not x.get("complete") or (x.get("run_count"), x.get("frame_count")) != (43, 864)
               or x.get("bottom_mass_allowed") is not False for x in (old_seal, joint_seal))):
        raise ValueError("T4c requires complete original base and source-identical T4b candidates")
    if joint_seal["source_plan_sha256"] != sha256_file(t4b / "t4b_floor_plan.json"):
        raise ValueError("T4b source plan differs from its seal")
    heads = {k: artifact(parent / p) for k, p in (("co2", "models/generic/co2_presence_signal.json"),
        ("g3", "models/g3/gas_head.json"), ("regional", "regional_model.json"))}
    if any(heads[k]["sha256"] != s for k, s in (("co2", CO2_SHA256), ("g3", G3_SHA256), ("regional", REFIT_SHA256))):
        raise ValueError("T4c frozen base heads changed")
    heads["t4b"] = joint_plan["binding"]["heads"]["joint"]
    for value in heads["t4b"].values(): bound_path(value)
    inventory = joint_plan["source_inventory"]
    validate_source_inventory(inventory)
    old_frames = {f"{x['run']}/frame_{x['frame_index']:03d}": x["receipt"] for x in old_seal["frames"]}
    joint_frames = {f"{x['run']}/frame_{x['frame_index']:03d}": x["receipt"] for x in joint_seal["frames"]}
    if old_frames != joint_plan["parent_frame_receipts"] or set(joint_frames) != set(old_frames):
        raise ValueError("T4c parent frame sequence changed")
    measure = artifact(parent / "physical_scale_measurement.json")
    if read_json(bound_path(measure)).get("physical_scale_approved") is not True:
        raise ValueError("T4c physical scale is not approved")
    binding = dict(parent_seal=artifact(parent / SEAL), parent_plan=artifact(parent / "old_domain_seam_plan.json"),
        parent_registration=artifact(registration_path(parent.name)),
        t4b_seal=artifact(t4b / SEAL), t4b_plan=artifact(t4b / "t4b_floor_plan.json"),
        spec=artifact(spec), light_epoch_table=joint_plan["binding"]["inputs"]["light_epoch_table"],
        frozen_zone_policy=dict(policy=ZONE_POLICY, facies_map=artifact(facies_map), zone_map=artifact(zone_map)),
        measurement=measure, heads=heads, code=code_binding(), fixed_configuration=FIXED_CONFIGURATION,
        cpu_only=True, training=False, annotations_read=False, mass_eligible=False, bottom_mass_allowed=False)
    return dict(contract="ffac-t4c-evaluation-config-v1", parent=str(parent), t4b=str(t4b), binding=binding,
        source_inventory=inventory, source_context_plans=joint_plan["source_context_plans"],
        native_sources=joint_plan["native_sources"], run_to_epoch=joint_plan["run_to_epoch"],
        parent_frame_receipts=old_frames, t4b_frame_receipts=joint_frames)


def validate_config(config):
    if (config.get("contract") != "ffac-t4c-evaluation-config-v1"
        or config["binding"]["spec"]["sha256"] != SPEC_SHA256
        or config["binding"]["code"] != code_binding()
        or config["binding"]["fixed_configuration"] != FIXED_CONFIGURATION):
        raise ValueError("T4c evaluation configuration changed")
    validate_source_inventory(config["source_inventory"])
    for name in ("parent_seal", "parent_plan", "parent_registration", "t4b_seal", "t4b_plan", "spec", "light_epoch_table", "measurement"):
        _verify_bound_asset(config["binding"][name])
    for name in ("facies_map", "zone_map"): _verify_bound_asset(config["binding"]["frozen_zone_policy"][name])
    for key in ("co2", "g3", "regional"): _verify_bound_asset(config["binding"]["heads"][key])
    for value in config["binding"]["heads"]["t4b"].values(): _verify_bound_asset(value)
    return config


def prepare_evaluation_frame(config, run, frame_index, *, baseline_cache=None):
    from darsia.fluidflower.phase_gas_t2_regional_inputs import prepare_frame_inputs
    from darsia.fluidflower.phase_gas_t4c_placements import prepare_shift_context, prepare_p2_allowed
    from darsia.fluidflower.phase_l1_neighbourhood_gas import select_frozen_base_rule_b
    from darsia.fluidflower.phase_ffac_floor_contact import prepare_bound_floor_zone
    from . import run_ffac_rig_camera_inference as producer
    from .report_ffac_t4b_floor_candidate import build_reference_zones
    from .report_ffac_t2_floor_candidate import native_zones
    wall, cpu = time.perf_counter(), time.process_time()
    parent = Path(config["parent"])
    source = next(s for s in config["source_inventory"] if s["run"] == run and s["frame_index"] == frame_index)
    key = f"{run}/frame_{frame_index:03d}"
    old = read_json(bound_path(config["parent_frame_receipts"][key]))
    if old["binding"]["source"] != source: raise ValueError("T4c source frame changed")
    before = _png(old["labels_map"])
    context_plan = config["source_context_plans"][run]
    original = read_json(bound_path(context_plan["source_producer_frames"][key]))
    epoch = config["run_to_epoch"][run]
    if epoch != "new_lamps": raise ValueError("T4c must not prepare RAW optics for frozen rig light")
    inputs = prepare_frame_inputs(parent, old, original_plan=context_plan, baseline_cache=baseline_cache)
    classes, area = inputs.pop("classes"), inputs.pop("pixel_area_mm2")
    modelled = inputs.pop("modelled")
    inputs.pop("filled_support")
    fill = classes["filled_observed"] | classes["filled_unobserved"]
    domain = inputs.pop("context_domain", modelled)
    base_frame_binding = inputs.pop("base_frame_binding")
    heads = config["binding"]["heads"]
    context = prepare_shift_context(parent, bound_path(heads["g3"]), bound_path(heads["regional"]),
        context_domain=domain, run=run,
        light_epochs=dict(run_to_epoch=config["run_to_epoch"], receipt=config["binding"]["light_epoch_table"]),
        expected_base_pixel_sha256=heads["g3"]["sha256"], expected_base_regional_sha256=heads["regional"]["sha256"],
        t4b_pixel_path=bound_path(heads["t4b"]["g3"]), t4b_regional_path=bound_path(heads["t4b"]["regional"]),
        expected_t4b_pixel_sha256=heads["t4b"]["g3"]["sha256"],
        expected_t4b_regional_sha256=heads["t4b"]["regional"]["sha256"], base_labels=before, **inputs)
    top = producer.load_array(source["rig_geometry"]["frame_photo_top_keep"])
    barrier = _classes(original["rule_b_geometry"]["asset"])
    frozen_b, b_audit = select_frozen_base_rule_b(old, original)
    zone_assets = config["binding"]["frozen_zone_policy"]
    reference = build_reference_zones(bound_path(zone_assets["facies_map"]), bound_path(zone_assets["zone_map"]))
    zones = native_zones(reference, old, before.shape)
    allowed, p2_audit, p2_masks = prepare_p2_allowed(before, zones, barrier["barrier_mask"],
        barrier["open_components"], frozen_b)
    classes = floor_contact_classes(old, classes)
    observed = (classes["modelled"] | classes["filled_observed"]) & classes["rig_domain"] & ~classes["hardware_excluded"]
    floor_zone, floor_binding = prepare_bound_floor_zone(old, classes, area, observed)
    return dict(config=config, source=source, old=old, original=original, before=before, classes=classes, area=area,
        modelled=modelled, fill=fill, context=context, top_keep=top, barrier=barrier, frozen_b=frozen_b,
        b_audit=b_audit, p2_allowed=allowed, p2_audit=p2_audit, p2_masks=p2_masks,
        floor_zone=floor_zone, floor_binding=floor_binding, observed=observed,
        base_frame_binding=base_frame_binding,
        preparation_timing=dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu))


def evaluate_prepared_placement(state, placement, delta):
    from darsia.fluidflower.phase_gas_t4c_placements import evaluate_placement
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact
    labels, audit, masks = evaluate_placement(state["context"], placement, delta, state["before"],
        state["modelled"], state["area"], filled_support=state["fill"], top_keep=state["top_keep"],
        open_components=state["barrier"]["open_components"], barrier_mask=state["barrier"]["barrier_mask"],
        frozen_base_rule_b=state["frozen_b"], source_component_ids=state["original"]["rule_b_geometry"]["source_component_ids"],
        p2_allowed=state["p2_allowed"])
    if not np.array_equal(np.isin(labels, (1, 2)), np.isin(state["before"], (1, 2))):
        raise ValueError("T4c gas placement changed the frozen CO2 union")
    classes = state["classes"]
    completed = apply_floor_contact(labels, elapsed_hours=float(state["source"]["elapsed_h"]), zone=state["floor_zone"],
        observed_support=state["observed"], filled_support=state["fill"] & classes["rig_domain"] & ~classes["hardware_excluded"],
        unresolved=~np.isin(labels, (0, 1, 2)), pixel_area_mm2=state["area"])
    masks.update(state["p2_masks"])
    return completed.labels, dict(audit, p2_eligibility=state["p2_audit"], floor_applied=True, floor_rule_applied=True), masks, completed


def validate_runtime_heads(candidate, marker, heads, *, verified_plan=None):
    from darsia.fluidflower.phase_validation import _model_identity
    root = Path(candidate).resolve()
    plan = read_json(bound_path(marker["plan"], base=root)) if verified_plan is None else verified_plan
    if (plan.get("contract") != CONTRACT or marker.get("contract") != CONTRACT
        or plan.get("output_root") != str(root) or plan.get("placement") not in PLACEMENT_NAMES
        or marker.get("placement") != plan["placement"] or marker.get("delta") != plan["delta"]
        or marker.get("spec_sha256") != SPEC_SHA256
        or plan.get("model_identity") != list(_model_identity(root))
        or heads != plan["binding"]["heads"] or heads != marker.get("heads")
        or plan["job_sha256"] != canonical_sha256(plan["binding"])
        or plan["pipeline_sha256"] != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))):
        raise ValueError("T4c runtime identity, fixed placement or source calibration changed")
    for key in ("co2", "g3", "regional"): bound_path(heads[key], base=root)
    for value in heads["t4b"].values(): bound_path(value, base=root)
    shift = read_json(bound_path(plan["binding"]["inputs"]["calibrated_shift"]))
    if shift.get("placement") != plan["placement"] or shift.get("delta") != plan["delta"]:
        raise ValueError("T4c delta differs from its pre-registered calibration result")
    return True


def prepare_candidate(config_path, output, placement, calibrated_shift, *, resume=False):
    from darsia.fluidflower.phase_validation import _model_identity
    saved_config = read_json(config_path)
    config = validate_config(saved_config["runtime"] if saved_config.get("contract") == "ffac-t4c-overflow-calibration-sweep-v1" else saved_config)
    parent, root = Path(config["parent"]), Path(output).resolve()
    shift_path = Path(calibrated_shift)
    shift = read_json(shift_path)
    if "runtime" in saved_config and shift.get("calibration_config") != artifact(config_path):
        raise ValueError("T4c calibration result belongs to another sweep configuration")
    if placement not in PLACEMENT_NAMES or shift.get("placement") != placement or not np.isfinite(shift.get("delta", np.nan)):
        raise ValueError("T4c needs a finite calibrated shift for the chosen placement")
    raw = read_json(bound_path(shift["script_output"]))
    counts_path = bound_path(shift["counts"])
    if (raw.get("script_sha256") != CALIBRATOR_SHA256 or raw.get("counts_sha256") != sha256_file(counts_path)
        or raw["variants"][placement]["delta_star"] != shift.get("delta_star_unrounded")
        or round(float(shift["delta_star_unrounded"]), 2) != float(shift["delta"])):
        raise ValueError("T4c delta is not the frozen calibrator's returned value rounded to .01")
    if root == parent or root.is_relative_to(parent): raise ValueError("T4c requires a separate candidate namespace")
    inputs = dict(config["binding"], evaluation_config=artifact(config_path), evaluation_config_sha256=canonical_sha256(config),
        parent=str(parent), spec=config["binding"]["spec"], calibrated_shift=artifact(shift_path),
        output_root=str(root), placement=placement, delta=float(shift["delta"]),
        source_inventory_sha256=canonical_sha256(config["source_inventory"]),
        native_sources_sha256=canonical_sha256(config["native_sources"]),
        source_context_plans_sha256=canonical_sha256(config["source_context_plans"]),
        parent_frame_receipts_sha256=canonical_sha256(config["parent_frame_receipts"]),
        blind_hold_runs=[], l1_rules_applied=[], code=code_binding())
    if (root / PLAN).exists():
        previous = read_json(root / PLAN)
        if not resume or previous["binding"]["inputs"] != inputs: raise ValueError("T4c resume changes fixed inputs")
        return previous
    if root.exists() or resume: raise ValueError("Use a new output directory or exact resume")
    root.mkdir(parents=True, exist_ok=False)
    for source in (parent / "models").rglob("*"):
        if source.is_file(): copy_file_verified(source, root / source.relative_to(parent))
    copy_file_verified(parent / "regional_model.json", root / "regional_model.json")
    for name in ("training_request.json", "expected_review_runs.json", "physical_scale_measurement.json", "vertical_scale_check.json"):
        if (parent / name).exists(): copy_file_verified(parent / name, root / name)
    copy_file_verified(bound_path(config["binding"]["heads"]["t4b"]["g3"]), root / "models/g3/t4b/gas_head.json")
    copy_file_verified(bound_path(config["binding"]["heads"]["t4b"]["regional"]), root / "regional_models/t4b.json")
    copy_file_verified(bound_path(inputs["spec"]), root / "approved_t4c_floor_spec.md")
    summary = read_json(parent / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=utc_now(),
        display_name=f"FF_AC Â· G3-T4c {placement} gas Â· New lamps calibrated to overflow Â· {PLACEMENT_NAMES[placement]} Â· Floor contact Â· 08.10.2026",
        inherited_base_id=parent.name, training_performed_in_this_revision=False, optical_training_performed=False,
        regional_training_performed=False, co2_training_performed=False, mass_eligible=False, scientific_acceptance=False,
        automatic_activation=False, bottom_mass_allowed=False, blind_hold_runs=[], l1_rules_applied=[],
        t4c_placement=placement, t4c_delta=float(shift["delta"]), t4c_calibration=inputs["calibrated_shift"],
        t4c_acceptance="pending locked regression/time-transfer/split-half/placement decision")
    atomic_json(root / "summary.json", summary)
    heads = {key: artifact(root / relative) for key, relative in (("co2", "models/generic/co2_presence_signal.json"),
        ("g3", "models/g3/gas_head.json"), ("regional", "regional_model.json"))}
    heads["t4b"] = dict(g3=artifact(root / "models/g3/t4b/gas_head.json"), regional=artifact(root / "regional_models/t4b.json"))
    binding = dict(inputs=inputs, parent=str(parent), spec=inputs["spec"], heads=heads)
    identity = list(_model_identity(root))
    plan = dict(contract=CONTRACT, created_utc=summary["created_utc"], binding=binding,
        evaluation_config=config, source_inventory=config["source_inventory"], run_to_epoch=config["run_to_epoch"],
        native_sources=config["native_sources"], source_context_plans=config["source_context_plans"],
        parent_frame_receipts=config["parent_frame_receipts"], placement=placement, delta=float(shift["delta"]),
        model_identity=identity, job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)), output_root=str(root))
    atomic_json(root / PLAN, plan)
    profile = read_json(parent / "phase_rig_camera.json")
    profile["parent_model_identity"] = list(_model_identity(parent))
    for name in ("g3_t4b_floor_revision", "g3_t4_floor_revision", "l1_neighbourhood_floor_revision", "g3_t3_floor_revision", "g3_t2_l1zone_revision", "g3_t2_floor_revision"):
        profile.pop(name, None)
    profile["frozen_heads"] = heads
    profile["g3_t4c_floor_revision"] = dict(contract=CONTRACT, plan=artifact(root / PLAN),
        spec_sha256=SPEC_SHA256, heads=heads, placement=placement, delta=plan["delta"])
    atomic_json(root / "profile_template.json", profile)
    return plan


def candidate_state(root):
    root = Path(root).resolve()
    plan = read_json(root / PLAN)
    inputs = plan["binding"]["inputs"]
    if (inputs["code"] != code_binding() or inputs["blind_hold_runs"] != [] or inputs["l1_rules_applied"] != []
        or canonical_sha256(plan["evaluation_config"]) != inputs["evaluation_config_sha256"]
        or canonical_sha256(plan["source_inventory"]) != inputs["source_inventory_sha256"]):
        raise ValueError("T4c candidate lost its immutable code/source/calibration binding")
    validate_config(plan["evaluation_config"])
    bound_path(inputs["evaluation_config"])
    validate_runtime_heads(root, read_json(root / "profile_template.json")["g3_t4c_floor_revision"], plan["binding"]["heads"], verified_plan=plan)
    return root, plan, plan["source_inventory"]


def infer_frame(root, plan, source, *, frame_state=None, baseline_cache=None):
    from . import run_ffac_rig_camera_inference as producer
    root, config = Path(root), plan["evaluation_config"]
    run, index = source["run"], source["frame_index"]
    key, path = f"{run}/frame_{index:03d}", root / "rig_camera_frames" / run / f"frame_{index:03d}.json"
    parent_asset = config["parent_frame_receipts"][key]
    binding = dict(pipeline_sha256=plan["pipeline_sha256"], source=source, parent_receipt=parent_asset)
    if path.exists():
        old = read_json(path)
        if old.get("complete") is not True or old.get("contract") != CONTRACT or old.get("binding") != binding:
            raise ValueError("T4c resume frame differs from its source binding")
        for a in [old["labels_map"], old["floor_contact_fill"], old["floor_trigger_canvas_y"], *old["maps"].values()]: bound_path(a)
        return dict(run=run, frame_index=index, receipt=artifact(path))
    wall, cpu = time.perf_counter(), time.process_time()
    epoch = plan["run_to_epoch"][run]
    old = read_json(bound_path(parent_asset))
    before = _png(old["labels_map"])
    if epoch == "rig_light":
        joint_asset = config["t4b_frame_receipts"][key]
        joint = read_json(bound_path(joint_asset))
        if joint["binding"]["source"] != source or joint["before_labels"] != old["labels_map"]:
            raise ValueError("Imported floor completion does not bind the exact unchanged rig-light base")
        from darsia.fluidflower.phase_pixel_area import load_rig_classes
        from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
        from darsia.fluidflower.phase_l1_neighbourhood_gas import select_frozen_base_rule_b, frozen_rule_b_forbidden_mask
        area = producer.load_array(old["rig_domain_revision"]["area_map"])
        classes, _ = load_rig_classes(old["rig_domain_revision"])
        classes = floor_contact_classes(old, classes)
        valid = classes["rig_domain"] & ~classes["hardware_excluded"]
        observed = (classes["modelled"] | classes["filled_observed"]) & valid
        filled = (classes["filled_observed"] | classes["filled_unobserved"]) & valid
        zone, floor_binding = prepare_bound_floor_zone(old, classes, area, observed)
        completed = apply_floor_contact(before, elapsed_hours=float(source["elapsed_h"]), zone=zone,
            observed_support=observed, filled_support=filled, unresolved=~np.isin(before, (0, 1, 2)), pixel_area_mm2=area)
        after, flags, floor_rule = completed.labels, completed.floor_contact_fill, completed.report
        floor = _write_array(root, "floor_contact_masks", run, index, flags, source, packed=True, contract="ffac-t2-floor-contact-v1")
        trigger = _write_array(root, "floor_contact_triggers", run, index, completed.trigger_canvas_y, source)
        original_asset = config["source_context_plans"][run]["source_producer_frames"][key]
        original = read_json(bound_path(original_asset))
        frozen_b, b_audit = select_frozen_base_rule_b(old, original)
        barriers = _classes(original["rule_b_geometry"]["asset"])
        _, geometry_audit = frozen_rule_b_forbidden_mask(barriers["open_components"], barriers["barrier_mask"], frozen_b)
        gas_audit = dict(rig_light_bit_identical=True, labels_read=False, training=False,
            new_gas_inference_performed=False, floor_rule_applied=True, floor_computed_from="exact original base labels",
            frozen_co2_union_reused=True)
        extra = dict(frozen_base_rule_b=frozen_b, frozen_rule_b_source_selection=b_audit,
            frozen_rule_b_geometry_audit=geometry_audit, rule_b=frozen_b, rule_b_geometry=original["rule_b_geometry"])
    else:
        state = frame_state if frame_state is not None else prepare_evaluation_frame(config, run, index, baseline_cache=baseline_cache)
        if state["source"] != source: raise ValueError("Shared T4c state belongs to another source frame")
        after, gas_audit, masks, completed = evaluate_prepared_placement(state, plan["placement"], plan["delta"])
        flags, area = completed.floor_contact_fill, state["area"]
        floor = _write_array(root, "floor_contact_masks", run, index, flags, source, packed=True, contract="ffac-t2-floor-contact-v1")
        trigger = _write_array(root, "floor_contact_triggers", run, index, completed.trigger_canvas_y, source)
        floor_rule, floor_binding = completed.report, state["floor_binding"]
        extra = dict(frozen_base_rule_b=state["frozen_b"], frozen_rule_b_source_selection=state["b_audit"],
            rule_b_geometry=state["original"]["rule_b_geometry"], rule_b=gas_audit.get("rule_b_application", gas_audit.get("frozen_rule_b", {})),
            shared_frame_preparation_timing=state["preparation_timing"],
            shared_preparation_cost_scope="one prepared source frame shared by placements; do not sum once per placement")
        if plan["placement"] == "P2":
            for name in ("p2_allowed", "p2_barrier", "p2_rule_b_forbidden"):
                extra[name] = _write_array(root, name, run, index, masks[name], source, packed=True, contract="ffac-t4c-p2-spatial-proof-v1")
    from darsia.fluidflower.phase_gas_barrier_rule import apply_barrier_rule_b
    if epoch == "rig_light":
        top_keep = producer.load_array(source["rig_geometry"]["frame_photo_top_keep"])
        barrier = _classes(extra["rule_b_geometry"]["asset"])
    else:
        top_keep, barrier = state["top_keep"], state["barrier"]
    checked, _ = apply_barrier_rule_b(after, barrier["open_components"], extra["rule_b_geometry"]["source_component_ids"],
        pixel_area_mm2=.027053, barrier_mask=barrier["barrier_mask"])
    b_diff = int(np.count_nonzero(checked != after))
    top_diff = int(np.count_nonzero((after == 2) & ~top_keep))
    union_diff = int(np.count_nonzero(np.isin(after, (1, 2)) != (np.isin(before, (1, 2)) | flags)))
    rig_diff = int(np.count_nonzero((before == 2) != (after == 2))) if epoch == "rig_light" else 0
    if union_diff or rig_diff or b_diff or top_diff or not np.array_equal(before == 255, after == 255):
        raise ValueError("T4c changed frozen rig gas, unsupported categories or unflagged CO2 union")
    maps, labels_asset = producer.write_maps(root, run, index, plan["pipeline_sha256"], after, phase_shape_hw=after.shape)
    if not np.array_equal(_png(labels_asset), after): raise ValueError("T4c saved labels differ from evaluated map")
    import psutil
    memory = psutil.Process().memory_info()
    timing = dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu,
        rss_bytes=memory.rss, peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss))
    qa = dict(run=run, frame_index=index, elapsed_h=source["elapsed_h"], source_sha256=source["source_sha256"],
        gas_before_cm2=float(area[before == 2].sum(dtype=np.float64)/100),
        gas_after_cm2=float(area[after == 2].sum(dtype=np.float64)/100),
        changed_pixels=int(np.count_nonzero(before != after)), rig_light_gas_differing_pixels=rig_diff,
        union_changes_outside_fill=union_diff, rule_b_changed_pixels=b_diff, top_clip_changed_pixels=top_diff,
        floor_contact_fill_pixels=int(flags.sum()), preinject_fill_pixels=int(flags.sum()) if source["elapsed_h"] <= 0 else 0,
        unchanged_unresolved_and_rig_domain=True, all_pass=True, **timing)
    receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
        run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        placement=plan["placement"], delta=plan["delta"], light_epoch_id=epoch,
        before_labels=old["labels_map"], labels_map=labels_asset, maps=maps, floor_contact_fill=floor,
        floor_trigger_canvas_y=trigger, floor_rule=floor_rule, floor_geometry=floor_binding,
        rig_domain_revision=old["rig_domain_revision"], full_extended_scientific=old["full_extended_scientific"],
        baseline_extended_scientific=old["baseline_extended_scientific"], gas_inference=gas_audit,
        qa=qa, timing=timing, l1_rules_applied=[], cpu_only=True, labels_changed=False, training=False,
        annotations_read=False, mass_eligible=False, bottom_mass_allowed=False, co2_cache_provenance=parent_asset,
        co2_inference_performed=False, new_gas_inference_performed=epoch == "new_lamps", **extra)
    atomic_json(path, receipt)
    return dict(run=run, frame_index=index, receipt=artifact(path))


def validate_run_result(root,plan,sources,result):
    expected=[s for s in sources if s["run"]==result.get("run")]
    if (result.get("complete") is not True or result.get("frame_count")!=len(expected)
            or result.get("pipeline_sha256")!=plan["pipeline_sha256"]
            or len(result.get("receipts",[]))!=len(expected)):
        raise ValueError("T4c run result is incomplete or lost its pipeline binding")
    for source,item in zip(expected,result["receipts"]):
        frame=read_json(bound_path(item["receipt"]))
        if (item["run"]!=source["run"] or item["frame_index"]!=source["frame_index"]
                or frame.get("contract")!=CONTRACT or frame.get("complete") is not True
                or frame["binding"]["source"]!=source or frame["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]
                or frame.get("light_epoch_id") != plan["run_to_epoch"][source["run"]]
                or frame.get("placement") != plan["placement"] or frame.get("delta") != plan["delta"]
                or frame.get("l1_rules_applied") != []
                or not frame["qa"]["all_pass"] or frame["qa"]["union_changes_outside_fill"]
                or frame["qa"]["preinject_fill_pixels"] or frame["qa"]["rule_b_changed_pixels"]
                or frame["qa"]["top_clip_changed_pixels"]):
            raise ValueError("T4c frame failed its exact source/union/floor/top/RuleB binding")
        for asset in [frame["labels_map"],frame["floor_contact_fill"],frame["floor_trigger_canvas_y"],*frame["maps"].values()]:
            bound_path(asset)
    for key in ("native_manifest","input_manifest"):
        bound_path(result[key])
    return True


def run_one_run(root, run, indices=None):
    worker_initializer()
    root, plan, inventory = candidate_state(root)
    sources = [s for s in inventory if s["run"] == run and (indices is None or s["frame_index"] in indices)]
    results = []
    baseline_cache = {}
    for source in sources:
        item = infer_frame(root, plan, source, baseline_cache=baseline_cache)
        results.append(item)
        if run not in HIDDEN:
            print(json.dumps(dict(event="t4c_frame_complete", run=run, frame_index=source["frame_index"])), flush=True)
    if indices is not None:
        return results
    return assemble_run(root, plan, run)


def assemble_run(root, plan, run):
    root = Path(root)
    inventory = plan["source_inventory"]
    results = []
    for source in inventory:
        if source["run"] == run:
            path = root / "rig_camera_frames" / run / ("frame_%03d.json" % source["frame_index"])
            results.append(dict(run=run, frame_index=source["frame_index"], receipt=artifact(path)))
    native_source = plan["native_sources"][run]
    native = read_json(bound_path(native_source["native_manifest"]))
    inputs = read_json(bound_path(native_source["input_manifest"]))
    if len(results) != len(native["frame_sources"]) or len(results) != len(inputs["frames"]):
        raise ValueError("T4c source assembly is incomplete")
    for item, record, binding in zip(results, native["frame_sources"], inputs["frames"]):
        frame = read_json(bound_path(item["receipt"]))
        if (item["run"] != run or item["frame_index"] != record["frame_index"]
                or frame["source_sha256"] != record["source_sha256"]
                or binding["source_sha256"] != record["source_sha256"]):
            raise ValueError("T4c floor flag changed its source-owned frame order")
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


def seal_candidate(root, plan=None, source_runs=None):
    from . import run_ffac_rig_camera_inference as producer
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    root, checked_plan, sources = candidate_state(root)
    if plan is None: plan = checked_plan
    if source_runs is None:
        source_runs = [assemble_run(root, plan, run) for run in sorted(plan["native_sources"])]
    if plan != checked_plan:
        raise ValueError("T4c seal plan differs from the exact saved pipeline")
    parent = Path(plan["binding"]["parent"])
    if len(source_runs)!=43 or len({row["run"] for row in source_runs})!=43:
        raise ValueError("T4c seal requires one complete result for every run")
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
        raise ValueError("Cannot seal partial T4c maps")
    qa_rows = [read_json(bound_path(f["receipt"]))["qa"] for f in frames]
    if any(not r["all_pass"] or r["union_changes_outside_fill"] or r["preinject_fill_pixels"] or r["rule_b_changed_pixels"] or r["top_clip_changed_pixels"] for r in qa_rows):
        raise ValueError("T4c exact-union/floor QA failed")
    qa_path = root / "production_qa/t4c_floor_qa.json"
    atomic_json(qa_path, dict(contract=CONTRACT, technical_all_pass=True, regression_acceptance="pending sealed scorer",
        run_count=43, frame_count=864, rows=qa_rows))
    seal = dict(contract=CONTRACT, complete=True, model_identity=plan["model_identity"], pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root / PLAN), placement=plan["placement"], delta=plan["delta"],
        calibrated_shift=plan["binding"]["inputs"]["calibrated_shift"],
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
            raise ValueError("Changed T4c registration")
        seal=read_json(root/SEAL)
        if (seal.get("complete") is not True or (seal.get("run_count"),seal.get("frame_count"))!=(43,864)
                or set(seal.get("blind_hold_runs",[]))!=HIDDEN
                or seal.get("bottom_mass_allowed") is not False or seal.get("mass_eligible") is not False):
            raise ValueError("Changed T4c final seal/hold scope")
        for relative, digest in registration["metadata_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("Changed T4c candidate metadata")
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: T4c floor review registration changed or is incomplete; do not use for mass."


def register(root):
    from .register_track1_studio_candidate import FORMAT, registration_path
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_runtime import load_physics_profile
    root = Path(root); seal = read_json(root / SEAL)
    if seal.get("complete") is not True or (seal["run_count"], seal["frame_count"]) != (43, 864):
        raise ValueError("T4c registration requires complete 43/864")
    load_physics_profile(root)
    files = ("summary.json", "training_request.json", "phase_rig_camera.json", PLAN, SEAL,
        "review_source/postflight_manifest.json", "validation/postflight_manifest.json", "validation/validation_binding.json",
        "studio_preparation.json", "production_qa/t4c_floor_qa.json")
    identity = _model_identity(root)
    native, maps = {}, {}
    for item in seal["frames"]:
        frame = read_json(bound_path(item["receipt"]))
        for a in [frame["labels_map"], frame["floor_contact_fill"], frame["floor_trigger_canvas_y"], *frame["maps"].values()]:
            bound_path(a)
        maps[f"{item['run']}/frame_{item['frame_index']:03d}"] = frame["maps"]["combined"]["sha256"]
    for path in (root / "source_alias/native_frames").glob("*/manifest.json"):
        native[path.parent.name] = sha256_file(path)
    value = dict(format=FORMAT, placement=seal["placement"], delta=seal["delta"], registration_kind=KIND, model_root=str(root), model_id=identity[0], model_sha256=identity[1],
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
            raise ValueError("Refusing to overwrite a different T4c registration")
    atomic_json(dest, value)
    return dest




def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command", required=True)
    configure=sub.add_parser("configure")
    for name in ("parent", "t4b", "spec", "facies-map", "zone-map", "output"):
        configure.add_argument("--"+name, type=Path, required=True)
    prepare=sub.add_parser("prepare")
    for name in ("config", "output", "calibrated-shift"):
        prepare.add_argument("--"+name, type=Path, required=True)
    prepare.add_argument("--placement", choices=tuple(PLACEMENT_NAMES), required=True)
    prepare.add_argument("--resume", action="store_true")
    smoke=sub.add_parser("smoke")
    smoke.add_argument("--candidate", type=Path, required=True)
    smoke.add_argument("--run", required=True)
    smoke.add_argument("--frames", default="0,8")
    finalize=sub.add_parser("seal")
    finalize.add_argument("--candidate", type=Path, required=True)
    a=p.parse_args(argv)
    if a.command == "configure":
        if a.output.exists(): raise ValueError("Configuration destination already exists")
        config=prepare_calibration_config(a.parent,a.t4b,a.spec,a.facies_map,a.zone_map)
        atomic_json(a.output, config)
    elif a.command == "prepare":
        prepare_candidate(a.config,a.output,a.placement,a.calibrated_shift,resume=a.resume)
    elif a.command == "smoke":
        result=run_one_run(a.candidate,a.run,[int(x) for x in a.frames.split(",")])
        atomic_json(a.candidate/"smoke_receipt.json", dict(complete=True, frames=result))
    elif a.command == "seal":
        root,plan,_=candidate_state(a.candidate)
        rows=[assemble_run(root,plan,run) for run in sorted(plan["native_sources"])]
        seal_candidate(root,plan,rows)


if __name__ == "__main__": main()
