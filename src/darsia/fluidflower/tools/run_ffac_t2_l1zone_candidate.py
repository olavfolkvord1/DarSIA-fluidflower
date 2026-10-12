"""Compose two sealed gas decisions around L1 and reapply the fixed floor rule.

This runner reads categorical model predictions, not human annotations. It does
not evaluate or fit a model, load scientific RGB arrays, or alter either parent.
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
from .run_ffac_rig_camera_candidate import artifact, admitted_workers, worker_initializer
from .run_ffac_t2_floor_candidate import _png, _classes, floor_contact_classes

CONTRACT = "ffac-g3-t2-l1zone-candidate-v1"
KIND = "ffac_g3_t2_l1zone_review"
PLAN = "t2_l1zone_plan.json"
RUN_DIRECTORY = "t2_l1zone_runs"
SEAL = "rig_camera_candidate_receipt.json"
SPEC_SHA256 = "3ac834612c4e9d9d8eb7b19b2986eba1d0825748375c5284e0fe8fe08c378281"
HIDDEN = frozenset(("ac30", "ac49", "ac54"))
BASE_HEAD_SHA = dict(co2="8c3727110be01dbef4849fad6849e50bcc8e28bdd1ee22a1c24c0ab8027c64e3",
    g3="90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692",
    regional="d33b3eea47392268d3ddebe2d9e9c695bfc640eb6e252d83670a3eaf3e7428be")
T2_HEAD_SHA = dict(co2=BASE_HEAD_SHA["co2"],
    g3="21796599d8812af95467f948844c5beda374beeda387b60b917731729551052b",
    regional="a255b7cf6add8703dab6f353dce624e3089d067353dd6afef389abd8e7615dfe")
WARNING = ("Studio review candidate: G3 gas with T2 only in the active 60 mm L1 scorer disc; "
    "unchanged floor-contact rule. No training or fresh head inference in this revision. "
    "Bottom blocked for mass; AC30/49/54 predictions withheld until this candidate's blind check is scored. "
    "Regression and blind acceptance pending Claude. Not approved for mass analysis.")


def code_binding():
    names = (__name__, "darsia.fluidflower.phase_ffac_t2_l1zone",
        "darsia.fluidflower.phase_ffac_floor_contact", "darsia.fluidflower.phase_ffac_bottom_band_diagnostics",
        "darsia.fluidflower.phase_pixel_area", "darsia.fluidflower.phase_gas_barrier_rule",
        "darsia.fluidflower.phase_ffac_rig_camera_runtime", "darsia.fluidflower.phase_ffac_rig_camera_metadata",
        "darsia.fluidflower.tools.run_ffac_t2_floor_candidate",
        "darsia.fluidflower.tools.run_ffac_rig_camera_inference")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def _verify_heads(heads, expected, root=None, relatives=None):
    if set(heads) != set(expected):
        raise ValueError("Combined candidate changed its frozen head inventory")
    for key, digest in expected.items():
        path = bound_path(heads[key], base=root)
        if heads[key]["sha256"] != digest:
            raise ValueError("Combined candidate changed a declared frozen head")
        if relatives and path != (Path(root) / relatives[key]).resolve():
            raise ValueError("Combined candidate head escaped its isolated namespace")


def validate_runtime_heads(candidate, marker, heads):
    from darsia.fluidflower.phase_validation import _model_identity
    root = Path(candidate).resolve()
    plan = read_json(bound_path(marker["plan"], base=root))
    if (marker.get("contract") != CONTRACT or marker.get("spec_sha256") != SPEC_SHA256
            or marker.get("frozen_co2_sha256") != BASE_HEAD_SHA["co2"]
            or plan.get("contract") != CONTRACT or plan.get("output_root") != str(root)
            or plan.get("model_identity") != list(_model_identity(root))
            or plan.get("job_sha256") != canonical_sha256(plan["binding"])
            or plan.get("pipeline_sha256") != canonical_sha256(dict(binding=plan["binding"], model_identity=plan["model_identity"]))
            or marker.get("heads") != heads or heads != plan["binding"]["heads"]
            or marker.get("t2_heads") != plan["binding"]["t2_heads"]
            or plan["binding"]["spec"]["sha256"] != SPEC_SHA256):
        raise ValueError("Combined runtime marker changed its exact isolated plan")
    _verify_heads(heads, BASE_HEAD_SHA, root, dict(co2="models/generic/co2_presence_signal.json",
        g3="models/g3/gas_head.json", regional="regional_model.json"))
    _verify_heads(marker["t2_heads"], T2_HEAD_SHA, root, dict(co2="models/generic/co2_presence_signal.json",
        g3="models/t2/gas_head.json", regional="models/t2/regional_model.json"))
    return True


def _sealed_parent(root, plan_name):
    from .register_track1_studio_candidate import registration_path, registration_warning
    if "QA FAILED" in registration_warning(root):
        raise ValueError("A combined parent registration is not intact")
    seal = read_json(root / SEAL)
    if (seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43, 864)
            or seal.get("bottom_mass_allowed") is not False or set(seal.get("blind_hold_runs", [])) != HIDDEN):
        raise ValueError("Combined composition requires two complete 43/864 parents with mass/blind holds")
    return dict(root=str(root), seal=artifact(root / SEAL), plan=artifact(root / plan_name),
                registration=artifact(registration_path(root.name)))


def prepare_candidate(base, t2, output, spec, *, resume=False):
    from darsia.fluidflower.phase_validation import _model_identity
    base, t2, root = map(lambda p: Path(p).resolve(), (base, t2, output))
    if any(root == parent or root.is_relative_to(parent) for parent in (base, t2)):
        raise ValueError("Use a separate new combined-candidate namespace")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("Combined candidate requires the exact approved L1 specification")
    parents = dict(base=_sealed_parent(base, "old_domain_seam_plan.json"), t2=_sealed_parent(t2, "t2_floor_plan.json"))
    bp, tp = read_json(bound_path(parents["base"]["plan"])), read_json(bound_path(parents["t2"]["plan"]))
    if tp["binding"]["parent"] != str(base) or tp["binding"]["inputs"]["parent_plan"] != parents["base"]["plan"]:
        raise ValueError("T2 is not bound to the selected exact seam parent")
    base_profile, t2_profile = read_json(base / "phase_rig_camera.json"), read_json(t2 / "phase_rig_camera.json")
    _verify_heads(base_profile["frozen_heads"], BASE_HEAD_SHA)
    _verify_heads(t2_profile["frozen_heads"], T2_HEAD_SHA)
    sources = bp["source_inventory"]
    if len(sources) != 864 or len({s["run"] for s in sources}) != 43:
        raise ValueError("Combined composition requires the original unique 43/864 source inventory")
    inputs = dict(parents=parents, spec=artifact(spec), code=code_binding(), output_root=str(root),
        source_inventory_sha256=canonical_sha256(sources), cpu_only=True, threads_per_worker=1,
        training=False, labels_read=False, bottom_mass_allowed=False, mass_eligible=False,
        blind_hold_runs=sorted(HIDDEN), automatic_activation=False,
        method=dict(point_ac28_xy=[2786., 2897.], phase_offset_xy=[416, 496], radius_px=375,
            activation_h=.935, fragment_minimum_mm2=2.5, connectivity=4,
            floor_rule="unchanged ffac-t2-floor-contact-v1", gas_partition="stored categorical predictions only"))
    if (root / PLAN).exists():
        previous = read_json(root / PLAN)
        if not resume or previous["binding"]["inputs"] != inputs:
            raise ValueError("Resume changed frozen sources, head decisions, code or rule")
        return previous
    if root.exists() or resume:
        raise ValueError("Use an absent output folder or an exact --resume")
    root.mkdir(parents=True)
    for source in (base / "models").rglob("*"):
        if source.is_file():
            copy_file_verified(source, root / source.relative_to(base))
    for name in ("regional_model.json", "training_request.json", "expected_review_runs.json",
                 "physical_scale_measurement.json", "vertical_scale_check.json"):
        if (base / name).exists():
            copy_file_verified(base / name, root / name)
    for key, name in (("g3", "gas_head.json"), ("regional", "regional_model.json")):
        copy_file_verified(bound_path(t2_profile["frozen_heads"][key]), root / "models/t2" / name)
    copy_file_verified(spec, root / "approved_t2_l1zone_spec.md")
    summary = read_json(base / "summary.json")
    summary.update(model_id=root.name, campaign_id=root.name, created_utc=utc_now(),
        display_name="FF_AC · G3 gas · T2 at L1 · Floor contact · 06.10.2026",
        inherited_t2_parent_id=t2.name, inherited_base_id=base.name,
        training_performed_in_this_revision=False, optical_training_performed=False,
        regional_training_performed=False, co2_training_performed=False,
        mass_eligible=False, scientific_acceptance=False, automatic_activation=False, bottom_mass_allowed=False,
        t2_regression_status="pending Claude combined-candidate sealed scorer",
        t2_l1zone_spec_sha256=SPEC_SHA256, composition_only=True)
    atomic_json(root / "summary.json", summary)
    policy = read_json(base / "rig_bottom_review_policy.json")
    policy.update(candidate_id=root.name, label_seal=None, bottom_acceptance=None,
        old_blind_reveal_does_not_release=True, candidate_blind_scoring_required=True)
    atomic_json(root / "rig_bottom_review_policy.json", policy)
    heads = {key: artifact(root / name) for key, name in dict(
        co2="models/generic/co2_presence_signal.json", g3="models/g3/gas_head.json", regional="regional_model.json").items()}
    t2_heads = dict(co2=heads["co2"], g3=artifact(root / "models/t2/gas_head.json"),
        regional=artifact(root / "models/t2/regional_model.json"))
    binding = dict(inputs=inputs, parent=str(base), t2_parent=str(t2), spec=inputs["spec"], heads=heads, t2_heads=t2_heads)
    identity = list(_model_identity(root))
    plan = dict(contract=CONTRACT, created_utc=summary["created_utc"], binding=binding, source_inventory=sources,
        job_sha256=canonical_sha256(binding), pipeline_sha256=canonical_sha256(dict(binding=binding, model_identity=identity)),
        model_identity=identity, output_root=str(root))
    atomic_json(root / PLAN, plan)
    profile = deepcopy(base_profile)
    profile.pop("g3_t2_floor_revision", None)
    profile["parent_model_identity"] = list(_model_identity(base))
    profile["frozen_heads"] = heads
    profile["g3_t2_l1zone_revision"] = dict(contract=CONTRACT, plan=artifact(root / PLAN),
        spec_sha256=SPEC_SHA256, frozen_co2_sha256=BASE_HEAD_SHA["co2"], heads=heads, t2_heads=t2_heads)
    atomic_json(root / "profile_template.json", profile)
    return plan


def candidate_state(root):
    root = Path(root).resolve()
    plan = read_json(root / PLAN)
    if (plan.get("contract") != CONTRACT or plan.get("output_root") != str(root)
            or plan["binding"]["inputs"]["code"] != code_binding()
            or plan["job_sha256"] != canonical_sha256(plan["binding"])
            or canonical_sha256(plan["source_inventory"]) != plan["binding"]["inputs"]["source_inventory_sha256"]):
        raise ValueError("Combined-candidate immutable code/source binding changed")
    validate_runtime_heads(root, read_json(root / "profile_template.json")["g3_t2_l1zone_revision"], plan["binding"]["heads"])
    for parent in plan["binding"]["inputs"]["parents"].values():
        for key in ("seal", "plan", "registration"):
            bound_path(parent[key])
    return root, plan, plan["source_inventory"]


def unpack_flag(asset, source, shape):
    if (asset.get("frame_index") != source["frame_index"] or asset.get("source_sha256") != source["source_sha256"]
            or asset.get("shape_hw") != list(shape) or asset.get("storage") != "packbits-npy"):
        raise ValueError("A packed flag belongs to another camera source or raster")
    packed = np.load(bound_path(asset), allow_pickle=False)
    if packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != (int(np.prod(shape)) + 7) // 8:
        raise ValueError("Invalid packed flag storage")
    return np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)


def _write_array(root, folder, run, index, values, source, *, packed=False, contract=CONTRACT):
    from . import run_ffac_rig_camera_inference as producer
    path = root / folder / run / f"frame_{index:03d}.npy"
    stream = io.BytesIO()
    np.save(stream, np.packbits(values) if packed else values, allow_pickle=False)
    return dict(producer._write_bytes(path, stream.getvalue()), contract=contract,
        frame_index=index, source_sha256=source["source_sha256"], shape_hw=list(values.shape),
        storage="packbits-npy" if packed else "npy")


def current_source_geometry(source, frame_input):
    """Verify the current camera-clock H against its source-owned certificate."""
    geometry = source["rig_geometry"]
    H = geometry["H_ac28_to_frame_native"]
    receipt = read_json(bound_path(frame_input["geometry_receipt"]))
    if (frame_input["source_sha256"] != source["source_sha256"]
            or frame_input["frame_index"] != source["frame_index"] or frame_input["run"] != source["run"]
            or float(frame_input["elapsed_h"]) != float(source["elapsed_h"])
            or frame_input["H_ac28_to_frame_native"] != H
            or receipt.get("camera_jpeg_sha256", receipt.get("source_sha256")) != source["source_sha256"]
            or receipt.get("H_ac28_to_frame_native") != H
            or frame_input["phase_grid"]["native_origin_xy"] != [-416, -496]):
        raise ValueError("L1 geometry does not belong to the current source on its camera-clock timeline")
    return H, frame_input["geometry_receipt"]


def _floor_difference_check(composite, t2_before_floor, changed_fill, zone, *,
                            supported, pixel_area_mm2):
    """Attribute each difference to the unchanged floor rule's actual trigger.

    In addition to the first pixel's phase, that rule requires its supported
    4-connected aqueous component to have at least 2.5 mm2. Gas/aqueous changes
    elsewhere can disconnect that component while its first pixel stays aqueous.
    This audit changes no fill decision: it proves either the phase change or
    the exact component eligibility crossing, rejecting every other difference.
    """
    if (composite.shape != t2_before_floor.shape or changed_fill.dtype != bool
            or changed_fill.shape != composite.shape or supported.dtype != bool
            or supported.shape != composite.shape or pixel_area_mm2.shape != composite.shape
            or not np.isfinite(pixel_area_mm2).all() or np.any(pixel_area_mm2 < 0)
            or not np.array_equal(composite == 255, t2_before_floor == 255)
            or not np.array_equal(np.isin(composite, (1, 2)), np.isin(t2_before_floor, (1, 2)))):
        raise ValueError("Floor difference attribution changed its domain, support, area or CO2 union")
    columns = np.flatnonzero(changed_fill.any(axis=0))
    unexplained, same_aqueous = [], []
    phase_changed = 0
    for col in columns:
        base = int(zone.base_canvas_y[col])
        co2 = np.flatnonzero(np.isin(composite[:base+1, col], (1, 2)))
        if not len(co2):
            unexplained.append(int(col))
            continue
        row = int(co2[-1])
        values = (int(composite[row, col]), int(t2_before_floor[row, col]))
        if not supported[row, col]:
            unexplained.append(int(col))
        elif set(values) == {1, 2}:
            phase_changed += 1
        elif values == (1, 1):
            same_aqueous.append((row, int(col)))
        else:
            unexplained.append(int(col))
    evidence = []
    if same_aqueous:
        import cv2
        membership_change = supported & ((composite == 1) != (t2_before_floor == 1))
        component_evidence = []
        # Keep only per-trigger statistics; release each full component raster
        # before constructing the other one to bound worker memory.
        for labels in (composite, t2_before_floor):
            count, components = cv2.connectedComponents(((labels == 1) & supported).astype(np.uint8), connectivity=4)
            area = np.bincount(components.ravel(), weights=pixel_area_mm2.ravel(), minlength=count)
            changed_members = np.bincount(components[membership_change], minlength=count)
            area[0] = changed_members[0] = 0
            component_evidence.append({(row, col): (float(area[components[row, col]]),
                int(changed_members[components[row, col]])) for row, col in same_aqueous})
            del components, area, changed_members
        for row, col in same_aqueous:
            combined_area, combined_changed = component_evidence[0][(row, col)]
            t2_area, t2_changed = component_evidence[1][(row, col)]
            combined_eligible, t2_eligible = combined_area + 1e-10 >= 2.5, t2_area + 1e-10 >= 2.5
            if combined_eligible == t2_eligible or not (combined_changed or t2_changed):
                unexplained.append(col)
                continue
            evidence.append(dict(column=col, first_co2_y=row,
                combined_aqueous_component_mm2=combined_area, t2_aqueous_component_mm2=t2_area,
                combined_component_eligible=combined_eligible, t2_component_eligible=t2_eligible,
                combined_component_changed_members=combined_changed, t2_component_changed_members=t2_changed))
    if unexplained:
        raise ValueError("Floor fill differs without a changed first-CO2 phase or supported aqueous-component eligibility")
    return dict(different_columns=int(len(columns)), unexplained_columns=0,
                phase_changed_columns=phase_changed, aqueous_component_minimum_changed_columns=len(evidence),
                minimum_aqueous_component_area_mm2=2.5, connectivity=4,
                minimum_change_evidence=evidence, rule_changed=False,
                explanation="first CO2 phase changed, or its supported 4-connected aqueous component crossed the unchanged 2.5 mm2 minimum")


def infer_frame(root, plan, parent_plan, source, *, frame_input=None):
    from . import run_ffac_rig_camera_inference as producer
    from darsia.fluidflower.phase_ffac_t2_l1zone import compose_l1_decisions
    from darsia.fluidflower.phase_ffac_floor_contact import apply_floor_contact, prepare_bound_floor_zone
    from darsia.fluidflower.phase_gas_barrier_rule import apply_barrier_rule_b
    from darsia.fluidflower.phase_pixel_area import load_rig_classes, load_rig_pixel_area
    root, base, t2_root = Path(root), Path(plan["binding"]["parent"]), Path(plan["binding"]["t2_parent"])
    run, index = source["run"], source["frame_index"]
    path = root / "rig_camera_frames" / run / f"frame_{index:03d}.json"
    base_asset = artifact(base / "rig_camera_frames" / run / f"frame_{index:03d}.json")
    t2_asset = artifact(t2_root / "rig_camera_frames" / run / f"frame_{index:03d}.json")
    binding = dict(pipeline_sha256=plan["pipeline_sha256"], source=source, parent_receipt=base_asset, t2_receipt=t2_asset)
    if path.exists():
        old = read_json(path)
        if old.get("complete") is not True or old.get("contract") != CONTRACT or old.get("binding") != binding:
            raise ValueError("Combined frame resume changed its immutable binding")
        for asset in [old["labels_map"], old["floor_contact_fill"], old["l1_zone_edge_fragment"],
                      old["floor_trigger_canvas_y"], *old["maps"].values()]:
            bound_path(asset)
        return dict(run=run, frame_index=index, receipt=artifact(path))
    wall, cpu = time.perf_counter(), time.process_time()
    old, t2 = read_json(bound_path(base_asset)), read_json(bound_path(t2_asset))
    if (old["binding"]["source"] != source or t2["binding"]["source"] != source
            or old["rig_domain_revision"] != t2["rig_domain_revision"]
            or old["full_extended_scientific"] != t2["full_extended_scientific"]
            or old["baseline_extended_scientific"] != t2["baseline_extended_scientific"]):
        raise ValueError("Two frozen decisions differ in source, domain, area, geometry or radiometry")
    before, t2_labels = _png(old["labels_map"]), _png(t2["labels_map"])
    parent_fill = unpack_flag(t2["floor_contact_fill"], source, before.shape)
    revision = old["rig_domain_revision"]
    classes, _ = load_rig_classes(revision)
    classes = floor_contact_classes(old, classes)
    area = load_rig_pixel_area(revision, before.shape, origin_xy=revision["native_origin_xy"]).values_mm2
    producer_asset = parent_plan["source_producer_frames"][f"{run}/frame_{index:03d}"]
    original = read_json(bound_path(producer_asset))
    minimum_asset = old.get("seam_reassembly", {}).get("minimum_area_map") or original["rig_domain_revision"]["area_map"]
    minimum_area = producer.load_array(minimum_asset)
    if frame_input is None:
        native = parent_plan["native_sources"][run]
        frame_input = read_json(bound_path(native["input_manifest"]))["frames"][index]
    H, geometry_receipt = current_source_geometry(source, frame_input)
    composed = compose_l1_decisions(before, t2_labels, parent_fill,
        H_ac28_to_frame_native=H, elapsed_hours=float(source["elapsed_h"]), pixel_area_mm2=minimum_area)
    if not np.array_equal(before == 255, ~classes["rig_domain"]):
        raise ValueError("Sealed categorical domain differs from its source-bound rig mask")
    fill_support = (classes["filled_observed"] | classes["filled_unobserved"]) & classes["rig_domain"] & ~classes["hardware_excluded"]
    observed = (classes["modelled"] | classes["filled_observed"]) & classes["rig_domain"] & ~classes["hardware_excluded"]
    floor_zone, floor_binding = prepare_bound_floor_zone(old, classes, area, observed)
    kwargs = dict(elapsed_hours=float(source["elapsed_h"]), zone=floor_zone, observed_support=observed,
        filled_support=fill_support, pixel_area_mm2=area)
    completed = apply_floor_contact(composed.labels, unresolved=~np.isin(composed.labels, (0, 1, 2)), **kwargs)
    parent_replay = apply_floor_contact(composed.t2_without_floor,
        unresolved=~np.isin(composed.t2_without_floor, (0, 1, 2)), **kwargs)
    if not np.array_equal(parent_replay.floor_contact_fill, parent_fill):
        raise ValueError("Unchanged floor rule does not reproduce the source-bound T2 floor flags")
    after, flag = completed.labels, completed.floor_contact_fill
    floor_delta = _floor_difference_check(composed.labels, composed.t2_without_floor, flag ^ parent_fill, floor_zone,
        supported=(observed | fill_support) & np.isin(composed.labels, (0, 1, 2)), pixel_area_mm2=area)
    barrier = _classes(original["rule_b_geometry"]["asset"])
    b_check, b_report = apply_barrier_rule_b(after, barrier["open_components"],
        original["rule_b_geometry"]["source_component_ids"], pixel_area_mm2=.027053, barrier_mask=barrier["barrier_mask"])
    b_changes = int(np.count_nonzero(b_check != after))
    top_keep = producer.load_array(source["rig_geometry"]["frame_photo_top_keep"])
    if top_keep.dtype != bool or top_keep.shape != after.shape:
        raise ValueError("Top-clip proof is not on this exact phase grid")
    top_changes = int(np.count_nonzero((after == 2) & ~top_keep))
    union_changes = int(np.count_nonzero(np.isin(after, (1, 2)) != (np.isin(before, (1, 2)) | flag)))
    if b_changes or top_changes or union_changes:
        raise ValueError("Combined map failed the exact CO2 union / unchanged Rule B / top-clip check")
    flag_descriptor = _write_array(root, "floor_contact_masks", run, index, flag, source,
        packed=True, contract="ffac-t2-floor-contact-v1")
    fragment_descriptor = _write_array(root, "l1_zone_edge_fragments", run, index,
        composed.l1_zone_edge_fragment, source, packed=True)
    triggers = _write_array(root, "floor_contact_triggers", run, index, completed.trigger_canvas_y, source)
    maps, labels_asset = producer.write_maps(root, run, index, plan["pipeline_sha256"], after, phase_shape_hw=after.shape)
    if not np.array_equal(_png(labels_asset), after):
        raise ValueError("Saved combined labels are not bit-identical to the checked array")
    qa = dict(composed.report, run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        gas_before_cm2=float(area[before == 2].sum(dtype=np.float64)/100),
        gas_after_cm2=float(area[after == 2].sum(dtype=np.float64)/100),
        co2_before_cm2=float(area[np.isin(before, (1, 2))].sum(dtype=np.float64)/100),
        co2_after_cm2=float(area[np.isin(after, (1, 2))].sum(dtype=np.float64)/100),
        changed_pixels=int(np.count_nonzero(before != after)), union_changes_outside_fill=union_changes,
        floor_contact_fill_pixels=int(flag.sum()), preinject_fill_pixels=int(flag.sum()) if source["elapsed_h"] <= 0 else 0,
        parent_floor_replay_differences=0, floor_only_combined_pixels=int((flag & ~parent_fill).sum()),
        floor_only_t2_pixels=int((parent_fill & ~flag).sum()), rule_b_changed_pixels=b_changes,
        top_clip_changed_pixels=top_changes, all_pass=True, wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)
    import psutil
    memory = psutil.Process().memory_info()
    timing = dict(wall_s=qa["wall_s"], cpu_s=qa["cpu_s"], rss_bytes=memory.rss,
        peak_working_set_bytes=getattr(memory, "peak_wset", memory.rss), worker_internal_threads=1)
    receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
        run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        before_labels=old["labels_map"], t2_labels=t2["labels_map"], t2_floor_contact_fill=t2["floor_contact_fill"],
        labels_map=labels_asset, maps=maps, floor_contact_fill=flag_descriptor, l1_zone_edge_fragment=fragment_descriptor,
        floor_trigger_canvas_y=triggers, l1_zone=dict(composed.zone.metadata(), H_ac28_to_frame_native=H,
            geometry_receipt=geometry_receipt, source_sha256=source["source_sha256"]),
        composition=composed.report, minimum_area_map=minimum_asset, floor_rule=completed.report,
        floor_geometry=floor_binding, floor_difference=floor_delta, rule_b=dict(b_report, changed_pixels=b_changes, check_only=True),
        top_clip=dict(changed_pixels=top_changes, check_only=True, asset=source["rig_geometry"]["frame_photo_top_keep"]),
        rig_domain_revision=revision, full_extended_scientific=old["full_extended_scientific"],
        baseline_extended_scientific=old["baseline_extended_scientific"], qa=qa, timing=timing,
        cpu_only=True, labels_changed=False, labels_read=False, training=False,
        mass_eligible=False, bottom_mass_allowed=False, co2_cache_provenance=base_asset,
        co2_inference_performed=False, new_gas_inference_performed=False, stored_decisions_composed=True)
    atomic_json(path, receipt)
    return dict(run=run, frame_index=index, receipt=artifact(path))


def validate_run_result(root, plan, sources, result):
    expected = [s for s in sources if s["run"] == result.get("run")]
    if (result.get("complete") is not True or result.get("frame_count") != len(expected)
            or result.get("pipeline_sha256") != plan["pipeline_sha256"] or len(result.get("receipts", [])) != len(expected)):
        raise ValueError("Combined run result is incomplete or has a different pipeline")
    for source, item in zip(expected, result["receipts"]):
        frame = read_json(bound_path(item["receipt"]))
        if (item["run"] != source["run"] or item["frame_index"] != source["frame_index"]
                or frame.get("contract") != CONTRACT or frame.get("complete") is not True
                or frame["binding"]["source"] != source or frame["binding"]["pipeline_sha256"] != plan["pipeline_sha256"]
                or not frame["qa"]["all_pass"] or frame["qa"]["union_changes_outside_fill"]
                or frame["qa"]["preinject_fill_pixels"] or frame["qa"]["rule_b_changed_pixels"] or frame["qa"]["top_clip_changed_pixels"]):
            raise ValueError("Combined run frame failed its exact source/technical QA binding")
        for a in [frame["labels_map"], frame["floor_contact_fill"], frame["l1_zone_edge_fragment"],
                  frame["floor_trigger_canvas_y"], *frame["maps"].values()]:
            bound_path(a)
    for key in ("native_manifest", "input_manifest"):
        bound_path(result[key])
    return True


def run_one_run(root, run, indices=None):
    worker_initializer()
    root, plan, inventory = candidate_state(root)
    parent_plan = read_json(bound_path(plan["binding"]["inputs"]["parents"]["base"]["plan"]))
    native_source = parent_plan["native_sources"][run]
    native, inputs = read_json(bound_path(native_source["native_manifest"])), read_json(bound_path(native_source["input_manifest"]))
    sources = [s for s in inventory if s["run"] == run and (indices is None or s["frame_index"] in indices)]
    results = []
    for source in sources:
        results.append(infer_frame(root, plan, parent_plan, source, frame_input=inputs["frames"][source["frame_index"]]))
        print(json.dumps(dict(event="l1zone_frame_complete", run=run, frame_index=source["frame_index"])), flush=True)
    if indices is not None:
        return results
    if len(results) != len(native["frame_sources"]) or len(results) != len(inputs["frames"]):
        raise ValueError("Combined source assembly is incomplete")
    for item, record, row in zip(results, native["frame_sources"], inputs["frames"]):
        frame = read_json(bound_path(item["receipt"]))
        if item["frame_index"] != record["frame_index"] or frame["source_sha256"] != record["source_sha256"] or row["source_sha256"] != record["source_sha256"]:
            raise ValueError("Combined flags changed source-owned frame order")
        for key in ("floor_contact_fill", "l1_zone_edge_fragment", "l1_zone"):
            record[key] = deepcopy(frame[key])
            row[key] = deepcopy(frame[key])
    input_path = root / "source_alias/source_bindings" / (run + ".json")
    atomic_json(input_path, inputs)
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    native = meta.build_native_metadata(read_json(bound_path(inputs["parent_native_manifest"])),
        read_json(bound_path(inputs["camera_timeline_run"])), native["frame_sources"], inputs["frames"], inputs["reference"], input_manifest=artifact(input_path))
    native_path = root / "source_alias/native_frames" / run / "manifest.json"
    atomic_json(native_path, native)
    result = dict(run=run, complete=True, native_manifest=artifact(native_path), input_manifest=artifact(input_path),
        frames=inputs["frames"], reference=inputs["reference"], frame_count=len(results), receipts=results,
        pipeline_sha256=plan["pipeline_sha256"])
    validate_run_result(root, plan, inventory, result)
    atomic_json(root / RUN_DIRECTORY / (run + ".json"), result)
    return result


def seal_candidate(root, plan, source_runs):
    from . import run_ffac_rig_camera_inference as producer
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    root, checked_plan, sources = candidate_state(root)
    if checked_plan != plan:
        raise ValueError("Seal plan differs from the immutable combined plan")
    meta.validate_campaign_bindings(source_runs)
    for row in source_runs:
        validate_run_result(root, plan, sources, row)
    profile = read_json(root / "profile_template.json")
    runtime = read_json(bound_path(profile["rig_domain_revision"]["manifest"]))
    runtime["sources"] = [{k: v for k, v in row.items() if k != "receipts"} for row in sorted(source_runs, key=lambda s: s["run"])]
    runtime_path = root / "rig_camera_revision/runtime_manifest.json"
    atomic_json(runtime_path, runtime)
    profile["rig_domain_revision"]["manifest"] = artifact(runtime_path)
    atomic_json(root / "phase_rig_camera.json", profile)
    packages = producer.bind_rig_packages(root, dict(plan, floor_contact_rule=True,
        upstream_plan=dict(binding=dict(parent=plan["binding"]["parent"]))), source_runs)
    assemble_saved_candidate(root, packages, plan)
    frames = [f for row in source_runs for f in row["receipts"]]
    qa_path = root / "production_qa/t2_l1zone_qa.json"
    atomic_json(qa_path, dict(contract=CONTRACT, technical_all_pass=True, regression_acceptance="pending sealed scorer",
        run_count=43, frame_count=864, rows=[read_json(bound_path(f["receipt"]))["qa"] for f in frames]))
    seal = dict(contract=CONTRACT, complete=True, model_identity=plan["model_identity"], pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root / PLAN), run_count=43, frame_count=864, frames=frames,
        plan=artifact(root / PLAN), qa=artifact(qa_path), parent_seal=plan["binding"]["inputs"]["parents"]["base"]["seal"],
        t2_parent_seal=plan["binding"]["inputs"]["parents"]["t2"]["seal"], heads=plan["binding"]["heads"], t2_heads=plan["binding"]["t2_heads"],
        physical_scale_approved=True, labels_changed=False, labels_read=False, training=False, mass_eligible=False,
        bottom_mass_allowed=False, blind_hold_runs=sorted(HIDDEN), automatic_activation=False, scientific_acceptance=False,
        regression_acceptance="pending Claude combined scorer", co2_union_unchanged_except_floor_contact_fill=True)
    atomic_json(root / SEAL, seal)
    register(root)
    return seal


def registered_warning(root, registration):
    try:
        from darsia.fluidflower.phase_validation import _model_identity
        root = Path(root)
        if (registration.get("registration_kind") != KIND or registration.get("model_sha256") != _model_identity(root)[1]
                or registration.get("bottom_mass_allowed") is not False or registration.get("mass_eligible") is not False):
            raise ValueError("Combined registration changed")
        for relative, digest in registration["metadata_sha256"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("Combined candidate metadata changed")
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: combined L1/floor review registration changed or is incomplete; do not use for mass."


def register(root):
    from .register_track1_studio_candidate import FORMAT, registration_path
    from darsia.fluidflower.phase_validation import _model_identity
    from darsia.fluidflower.phase_runtime import load_physics_profile
    root = Path(root)
    seal = read_json(root / SEAL)
    if seal.get("complete") is not True or (seal.get("run_count"), seal.get("frame_count")) != (43, 864):
        raise ValueError("Combined registration requires complete 43/864")
    load_physics_profile(root)
    files = ("summary.json", "training_request.json", "phase_rig_camera.json", PLAN, SEAL,
        "review_source/postflight_manifest.json", "validation/postflight_manifest.json", "validation/validation_binding.json",
        "studio_preparation.json", "rig_bottom_review_policy.json", "production_qa/t2_l1zone_qa.json")
    identity = _model_identity(root)
    native, maps = {}, {}
    for item in seal["frames"]:
        frame = read_json(bound_path(item["receipt"]))
        for a in [frame["labels_map"], frame["floor_contact_fill"], frame["l1_zone_edge_fragment"], *frame["maps"].values()]:
            bound_path(a)
        maps[f"{item['run']}/frame_{item['frame_index']:03d}"] = frame["maps"]["combined"]["sha256"]
    for path in (root / "source_alias/native_frames").glob("*/manifest.json"):
        native[path.parent.name] = sha256_file(path)
    value = dict(format=FORMAT, registration_kind=KIND, model_root=str(root), model_id=identity[0], model_sha256=identity[1],
        production_receipt_sha256=sha256_file(root / SEAL), source_plan_sha256=sha256_file(root / PLAN), run_count=43, frame_count=864,
        native_run_manifest_sha256=native, map_sha256=maps, metadata_sha256={name: sha256_file(root / name) for name in files},
        publication_scope="studio_validation_only", warning=WARNING, qa_status="technical_pass_regression_pending",
        scientific_acceptance=False, mass_eligible=False, deployment_allowed=False, automatic_activation=False,
        physical_scale_approved=True, bottom_mass_allowed=False, created_utc=utc_now())
    dest = registration_path(root.name)
    if dest.exists():
        old = read_json(dest)
        value["created_utc"] = old["created_utc"]
        if old != value:
            raise ValueError("Refusing to overwrite a different combined registration")
    atomic_json(dest, value)
    return dest


def run(plan, *, workers=24, smoke_only=False):
    root, checked_plan, sources = candidate_state(plan["output_root"])
    if checked_plan != plan:
        raise ValueError("Run plan differs from the immutable composition plan")
    below_normal_priority()
    with JobLease(root, plan["job_sha256"]):
        if smoke_only:
            rows = run_one_run(root, "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=rows, plan=artifact(root / PLAN)))
            return rows
        if not (root / "smoke_receipt.json").exists():
            rows = run_one_run(root, "ac28", [0, 8])
            atomic_json(root / "smoke_receipt.json", dict(complete=True, frames=rows, plan=artifact(root / PLAN)))
        runs = sorted({s["run"] for s in sources})
        pool = ProcessPoolExecutor(max_workers=min(admitted_workers(workers), len(runs)),
            mp_context=multiprocessing.get_context("spawn"), initializer=worker_initializer)
        futures = {pool.submit(run_one_run, str(root), run): run for run in runs}
        rows = []
        try:
            for future in as_completed(futures):
                rows.append(future.result())
                from .run_ffac_t2_floor_pipeline import write_telemetry
                write_telemetry(root / "progress.json", dict(stage="compositing", complete=False,
                    updated_utc=utc_now(), runs_done=len(rows), frames_done=sum(r["frame_count"] for r in rows), total_runs=43, total_frames=864))
        except BaseException as error:
            from .revise_ffac_old_domain_seam import _stop_failed_pool
            _stop_failed_pool(pool, futures)
            atomic_json(root / "failure_receipt.json", dict(error=repr(error), automatic_retry=False, updated_utc=utc_now()))
            raise
        else:
            pool.shutdown(wait=True)
        return seal_candidate(root, plan, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "t2", "output", "spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_candidate(args.base, args.t2, args.output, args.spec, resume=args.resume)
    if not args.plan_only:
        run(plan, workers=args.workers, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()
