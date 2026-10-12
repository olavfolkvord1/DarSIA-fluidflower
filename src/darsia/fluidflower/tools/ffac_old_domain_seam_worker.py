"""Source-bound inference of the released historical domain ring.

Photographs and scientific sidecars are read from the sealed parent. Only the
fixed ring and gas components whose connectivity it changes are revised.
"""
from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import run_ffac_rig_camera_inference as producer
from .refit_ffac_g3_downstream import atomic_json, read_json, utc_now
from .run_ffac_rig_camera_candidate import artifact
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path

CONTRACT = "ffac-old-domain-seam-frame-v1"
FROZEN_MODULES = ("darsia.fluidflower.phase_ffac_old_domain_seam",)


def _png(asset):
    with Image.open(bound_path(asset)) as image:
        return np.array(image)


def _arrays(asset):
    with np.load(bound_path(asset), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _pack(path, masks):
    stream = io.BytesIO()
    first = next(iter(masks.values()))
    np.savez_compressed(stream, shape_hw=np.asarray(first.shape, np.int64),
        **{key: np.packbits(value, bitorder="little") for key, value in masks.items()})
    return producer._write_bytes(path, stream.getvalue())


def infer_frame(root, plan, source, *, processor=None):
    from darsia.fluidflower.phase_retained_source_inference import (
        prepare_retained_source_cache, load_retained_pre_minimum_gas_selection)
    from darsia.fluidflower.phase_rig_domain_inference import infer_native, fill_rig_labels
    from darsia.fluidflower.phase_ffac_old_domain_seam import revise_seam_labels
    from darsia.fluidflower.phase_ffac_old_domain_ring_qa import frame_ring_revision_report
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    timings = {}
    root = Path(root)
    run, index = source["run"], source["frame_index"]
    key = f"{run}/frame_{index:03d}"
    measured_asset = plan["measured_parent_frames"][key]
    producer_asset = plan["source_producer_frames"][key]
    measured = read_json(bound_path(measured_asset))
    original = read_json(bound_path(producer_asset))
    binding = dict(pipeline_sha256=plan["pipeline_sha256"], source=source,
        upstream_preparation_sha256=plan["binding"]["scientific_preparation"]["sha256"])
    destination = root / "rig_camera_frames" / run / f"frame_{index:03d}.json"
    if destination.exists():
        saved = read_json(destination)
        if (saved.get("contract") != CONTRACT or saved.get("complete") is not True
                or saved.get("binding") != binding or saved.get("source_measured_receipt") != measured_asset
                or saved.get("source_producer_receipt") != producer_asset):
            raise ValueError("Seam resume changed its exact source/parent/code policy")
        for value in [*saved["maps"].values(), saved["labels_map"],
                      saved["rig_domain_revision"]["area_map"], saved["rig_domain_revision"]["classes"]]:
            bound_path(value)
        return dict(run=run, frame_index=index, receipt=artifact(destination))
    if (measured["source_producer_receipt"] != producer_asset
            or original["binding"]["source"] != source
            or measured["source_sha256"] != source["source_sha256"]
            or measured["elapsed_h"] != source["elapsed_h"]):
        raise ValueError("Measured parent and original producer bind different camera sources")
    before = _png(measured["labels_map"])
    descriptor = deepcopy(measured["rig_domain_revision"])
    classes = _arrays(descriptor["classes"])
    rig = classes["rig_domain"]
    area = producer.load_array(descriptor["area_map"])
    if (before.dtype != np.uint8 or before.shape != rig.shape
            or descriptor.get("physical_scale_approved") is not True):
        raise ValueError("Seam revision needs the measured sealed native grid")
    qa = deepcopy(measured["qa"])
    audit = dict(source_changed=source["source_changed"], inference_performed=False,
                 co2_outside_ring_changed_pixels=0, changed_pixels=0)
    after = before
    ring = np.zeros(rig.shape, bool)
    domain = np.zeros(rig.shape, bool)
    pre_minimum = None
    original_area = None
    masks_asset = None
    fresh_audit = None
    cache_audit = None
    if not source["source_changed"]:
        original_plan = plan["original_inference"]
        prepared = producer._prepared(root, original_plan, source["scientific_task_id"])
        native_log, native_valid, input_audit = producer.load_scientific(prepared)
        cache = prepare_retained_source_cache(source["retained_source_cache"], run=run,
            frame_index=source["parent_source_frame_index"],
            current_source_sha256=source["source_sha256"], current_log=native_log,
            allowed_domain=producer._crop(rig), producer_audit=input_audit,
            recompute_old_domain_interpolation_footprint=True)
        cache_audit = cache.audit
        timings["bound_native_input_wall_s"] = time.perf_counter() - started_wall
        native_domain = cache.cached_parent_modelled | cache.recomputed_old_domain_ring
        # The original domain includes flatfield-invalid pixels too; do not
        # mistake those for a new exterior or omit their released edge.
        exported = read_json(bound_path(source["retained_source_cache"]["parent_probability_receipt"]))
        with np.load(bound_path(exported["files"]["extra_masks.packbits.npz"]), allow_pickle=False) as archive:
            names = exported["metadata"]["masks_order"]
            packed = archive["masks"][names.index("regional_native_domain")]
            native_domain = np.unpackbits(packed, bitorder="little", count=native_log.shape[1]*native_log.shape[2])
            native_domain = native_domain.reshape(native_log.shape[1:]).astype(bool)
        domain = producer._canvas(native_domain, False, phase_shape_hw=rig.shape)
        ring = producer._canvas(cache.recomputed_old_domain_ring, False, phase_shape_hw=rig.shape)
        full, current_support = producer.compose_extended_scientific(original["full_extended_scientific"])
        baseline, baseline_valid = producer.compose_extended_scientific(original["baseline_extended_scientific"])
        h_b = np.asarray(prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
        h_b_full = producer.canvas_homography(h_b)
        aligned, baseline_support = producer.align_scientific(baseline, baseline_valid,
            h_b_full, target_shape_hw=rig.shape)
        previous = None
        previous_support = np.ones(rig.shape, bool)
        if source["previous_scientific_task_id"] is not None:
            task_id = source["previous_scientific_task_id"]
            previous_asset = plan.get("previous_extended_scientific_sources", {}).get(key,
                plan["extended_scientific_sources"].get(task_id))
            previous_descriptor = read_json(bound_path(previous_asset))
            previous_log, previous_valid = producer.compose_extended_scientific(previous_descriptor)
            previous_prepared = producer._prepared(root, original_plan, task_id)
            h_previous = np.asarray(previous_prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
            previous, previous_support = producer.align_scientific(previous_log, previous_valid,
                producer.canvas_homography(h_b @ np.linalg.inv(h_previous)), target_shape_hw=rig.shape)
        original_area = producer.load_array(original["rig_domain_revision"]["area_map"])
        calculation_start = time.perf_counter()
        fresh, support, fresh_audit, decisions = infer_native(original_plan["upstream_plan"]["binding"]["parent"],
            full, aligned, rig, aligned_previous_log=previous, pixel_area_mm2=original_area,
            source_baseline_log=baseline, h_baseline_to_current=h_b_full,
            source_baseline_valid=baseline_valid, return_decision_layers=True)
        timings["frozen_inference_wall_s"] = time.perf_counter() - calculation_start
        support &= rig & current_support & baseline_support & previous_support
        fresh[~support] = 255
        # Keep the expensive, source-bound numerical decisions independently
        # of the categorical correction. A failed later QA does not erase them.
        fresh_decision_asset = _pack(root / "fresh_decision_layers" / run /
            f"frame_{index:03d}.packbits.npz", dict(
                fresh_co2=decisions["co2"], fresh_pre_minimum_gas=decisions["pre_minimum_gas"],
                fresh_support=support, fresh_post_minimum_gas=fresh == 2))
        observed = classes["modelled"] | classes["filled_observed"]
        filled, fresh_classes = fill_rig_labels(fresh, rig, observed,
            hidden=classes["hardware_excluded"])
        for name in ("modelled", "filled_observed", "filled_unobserved"):
            classes[name][ring] = fresh_classes[name][ring]
        inherited_selection, threshold_audit = load_retained_pre_minimum_gas_selection(cache)
        old_selection = producer._canvas(inherited_selection, False, phase_shape_hw=rig.shape)
        barrier = _arrays(original["rule_b_geometry"]["asset"])
        after, audit, masks = revise_seam_labels(before, filled, ring, domain,
            old_selection, decisions["pre_minimum_gas"], classes["modelled"], original_area,
            open_components=barrier["open_components"],
            source_component_ids=original["rule_b_geometry"]["source_component_ids"],
            barrier_mask=barrier["barrier_mask"])
        audit.update(inference_performed=True, retained_threshold_decoder=threshold_audit,
            minimum_area_map=original["rig_domain_revision"]["area_map"],
            reporting_area_map=descriptor["area_map"], measured_scale_refiltered_globally=False,
            fresh_decision_layers=fresh_decision_asset,
            correction_scope="original regional exterior-rejected-cell bilinear interpolation footprint plus native outermost ring")
        pre_minimum = masks["corrected_pre_minimum_gas"] & masks["affected_gas_graph"]
        audit["changed_pixels"] = int(np.count_nonzero(before != after))
        masks_asset = _pack(root / "old_domain_seam_masks" / run / f"frame_{index:03d}.packbits.npz",
            dict(masks, released_ring=ring, old_domain=domain))
        payload = io.BytesIO()
        np.savez_compressed(payload, **classes)
        descriptor["classes"] = producer._write_bytes(root / "rig_camera_revision/context" / run /
            f"frame_{index:03d}.classes.npz", payload.getvalue())
        descriptor["parameters"]["area_cm2_by_class"] = {
            name: float(area[classes[name]].sum(dtype=np.float64)/100)
            for name in ("modelled", "filled_observed", "filled_unobserved")}
    else:
        from .refit_ffac_g3_downstream import copy_file_verified
        target = root / "rig_camera_revision/context" / run / f"frame_{index:03d}.classes.npz"
        copy_file_verified(bound_path(descriptor["classes"]), target)
        descriptor["classes"] = artifact(target)
    report = frame_ring_revision_report(run, index, source["elapsed_h"], before, after, domain,
        rig, area, pre_gas_minimum=pre_minimum, minimum_pixel_area_mm2=original_area,
        provenance=dict(source_sha256=source["source_sha256"], measured_parent=measured_asset,
                        source_changed=source["source_changed"]))
    if np.array_equal(before, after):
        from .refit_ffac_g3_downstream import copy_file_verified
        folder = root / "validation/run_packages" / run / "review_assets/native_phase_maps" / run / ("A_"+plan["pipeline_sha256"][:16])
        maps = {}
        for mode, value in measured["maps"].items():
            target = folder / f"frame_{index:03d}.png" if mode == "combined" else folder / mode / f"frame_{index:03d}.png"
            copy_file_verified(bound_path(value), target)
            maps[mode] = artifact(target)
        target = root / "production_labels" / run / f"frame_{index:03d}.png"
        copy_file_verified(bound_path(measured["labels_map"]), target)
        proof = artifact(target)
    else:
        maps, proof = producer.write_maps(root, run, index, plan["pipeline_sha256"], after, phase_shape_hw=rig.shape)
    qa.update(map_path=proof["path"], map_sha256=proof["sha256"], studio_map_path=maps["combined"]["path"],
        studio_map_sha256=maps["combined"]["sha256"], classes_sha256=descriptor["classes"]["sha256"],
        old_domain_ring_revision=report, seam_reassembly=audit, all_pass=True,
        physical_scale_approved=True, bottom_mass_allowed=False)
    # Preserve side areas and their established side partition; update counts
    # by explicit category deltas only in a separately declared report.
    qa["area_report"]["new_cm2"].update(
        co2=float(area[np.isin(after,(1,2))].sum(dtype=np.float64)/100),
        gas=float(area[after==2].sum(dtype=np.float64)/100),
        aqueous=float(area[after==1].sum(dtype=np.float64)/100),
        resolved=float(area[np.isin(after,(0,1,2)) & rig].sum(dtype=np.float64)/100))
    if not source["source_changed"]:
        from darsia.fluidflower.phase_ffac_rig_edge_qa import added_strip_report
        from darsia.fluidflower.phase_ffac_rig_qa import frame_area_report
        old_frame = read_json(bound_path(source["retained_source_cache"]["parent_frame_receipt"]))
        historical = _png(old_frame["labels_map"])
        native = read_json(bound_path(source["retained_source_cache"]["parent_native_manifest"]))
        record = native["frame_sources"][source["parent_source_frame_index"]]
        photo = np.load(bound_path(dict(path=record["v5_photo_sand_mask_path"],
            sha256=record["v5_photo_sand_mask_sha256"])), allow_pickle=False)
        historical[~photo] = 255
        old_canvas = producer._canvas(historical, phase_shape_hw=rig.shape)
        # Side/hardware masks are independent frozen geometric photographs.
        context, _ = producer.geometry_context(source["rig_geometry"], native_valid, input_audit)
        excluded = context.excluded_by_name
        hidden_chart = np.logical_or.reduce([excluded[name] for name in producer.HIDDEN_CHART_NAMES])
        qa["area_report"] = frame_area_report(old_canvas, after, area, classes,
            hardware_masks=excluded, hidden_chart_mask=hidden_chart)
        strips = added_strip_report(old_canvas, after, area, classes,
            inner_polygon_native=source["rig_geometry"]["frame_inner_polygon_native"],
            photo_top_keep=producer.load_array(source["rig_geometry"]["frame_photo_top_keep"]),
            native_origin_xy=producer.metadata.ORIGIN_XY,
            parent_photo_domain=producer._canvas(photo, False, phase_shape_hw=rig.shape))
        qa["added_strip_by_side"] = dict(strips, source_changed=False)
        qa["all_added_co2_zero"] = strips["added_co2_cm2"] == 0
        qa["outer_photo_extension_co2_zero"] = sum(
            side["outer_photo_extension_co2_cm2"] for side in strips["sides"].values()) == 0
    receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
        run=run, frame_index=index, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
        source_measured_receipt=measured_asset, source_producer_receipt=producer_asset,
        maps=maps, labels_map=proof, rig_domain_revision=descriptor, qa=qa,
        seam_reassembly=audit, seam_masks=masks_asset, fresh_inference=fresh_audit,
        retained_source_cache=cache_audit, full_extended_scientific=original["full_extended_scientific"],
        baseline_extended_scientific=original["baseline_extended_scientific"],
        cpu_only=True, training=False, labels_changed=False, physical_scale_approved=True,
        bottom_mass_allowed=False, mass_eligible=False)
    receipt["timing"] = dict(timings, elapsed_wall_s=time.perf_counter()-started_wall,
        process_cpu_s=time.process_time()-started_cpu, worker_internal_threads=1)
    atomic_json(destination, receipt)
    return dict(run=run, frame_index=index, receipt=artifact(destination))
