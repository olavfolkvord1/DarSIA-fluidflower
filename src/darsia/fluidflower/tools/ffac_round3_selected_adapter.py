"""Exact-source adapter for the separately sealed selected-image R3 campaign.

The selected phase/source contracts remain intact. Measured RAW, geometry,
area, recipe and frozen native paths are reused; neither historical frame IDs
nor a renamed old receipt substitute for a newly selected acquisition.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import FunctionType

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower import phase_ffac_common_mass_inputs as source
from darsia.fluidflower import phase_ffac_common_mass_round2 as native
from darsia.fluidflower import phase_ffac_common_mass_round2_inputs as dose_inputs
from darsia.fluidflower import phase_ffac_common_mass_round3 as engine
from darsia.fluidflower import phase_ffac_round3_handoff_admission as handoff

CONTRACT = "ffac-common-mass-round3-selected-adapter-v1"
PHASE_CONTRACT = "ffac-selected-p1-final3-phase-v1"
SOURCE_CONTRACT = "ffac-selected-frozen-scientific-source-v1"


def _target_match(entry, manifest):
    acquisitions = [row for row in manifest["frame_sources"]
        if row["source_sha256"] == entry["source_binding"]["source_sha256"]]
    if len(acquisitions) != 1:
        raise ValueError("Selected acquisition is absent or duplicated in the accepted camera manifest")
    acquisition = acquisitions[0]
    matches = [row for row in manifest["target_matches"]
        if row.get("selected") is True and row.get("selection_index") == acquisition["selection_index"]
        and row.get("source_sha256") == acquisition["source_sha256"]
        and float(row["target_hour"]) == float(entry["selected_target_h"])]
    if len(matches) != 1 or float(matches[0]["elapsed_h"]) != float(entry["elapsed_h"]):
        raise ValueError("Selected target/source/camera time is not admitted by the run manifest")
    return matches[0]


def _admit(entry, plan, reader, scope, frozen_calibration, *, asset_reader=None):
    """Metadata-only guards run before scientific arrays or categorical maps."""
    run = entry["run"]
    if scope == "calibration":
        if run not in engine.FIT_RUNS:
            raise ValueError("Held-out/descriptive selected pixels require the R3 freeze")
    elif scope == "frozen_evaluation":
        if frozen_calibration is None:
            raise ValueError("A real R3 freeze is required before selected evaluation")
        engine.verify_round3_freeze(frozen_calibration,
            **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    else:
        raise ValueError("Selected scientific access needs an explicit R3 scope")
    allowed = set(engine.FIT_RUNS) | set(engine.HELDOUT_RUNS) | set(engine.DESCRIPTIVE_RUNS)
    if run not in allowed:
        raise ValueError("Excluded or unknown selected run")
    gate = reader.json(plan["selected_phase_gate"])
    handoff.require_selected_gate(gate, reader, allowed)
    if gate.get("run_manifests", {}).get(run) != entry["run_manifest"]:
        raise ValueError("All 39 selected phase maps and Studio time parity must be sealed")
    manifest = reader.json(entry["run_manifest"])
    if manifest.get("contract") != "ffac-selected-camera-images-v1" or manifest.get("run") != run:
        raise ValueError("Selected run manifest has a different source identity")
    acquisition = entry["source_binding"]
    matches = [row for row in manifest["frame_sources"]
        if row["source_sha256"] == acquisition["source_sha256"]]
    if len(matches) != 1 or float(matches[0]["elapsed_h"]) != float(entry["elapsed_h"]):
        raise ValueError("Selected acquisition is absent from the accepted camera manifest")
    _target_match(entry, manifest)
    receipt = reader.json(entry["receipt"])
    if (receipt.get("contract") != PHASE_CONTRACT or receipt.get("complete") is not True
            or receipt.get("binding", {}).get("source") != acquisition
            or receipt.get("run") != run or receipt.get("frame_index") != entry["frame_index"]
            or receipt.get("elapsed_h") != entry["elapsed_h"]
            or receipt.get("source_sha256") != acquisition["source_sha256"]
            or receipt.get("bottom_mass_allowed") is not False):
        raise ValueError("Selected phase source/time/blocked-bottom binding differs")
    scientific = reader.json(entry["scientific_receipt"])
    if (scientific.get("contract") != SOURCE_CONTRACT or scientific.get("complete") is not True
            or scientific.get("run") != run or scientific.get("source_sha256") != receipt["source_sha256"]
            or scientific.get("elapsed_h") != receipt["elapsed_h"]
            or scientific.get("binding", {}).get("source") != acquisition
            or scientific.get("full_extended_scientific") != receipt["full_extended_scientific"]
            or scientific.get("baseline_extended_scientific") != receipt["baseline_extended_scientific"]):
        raise ValueError("Selected phase and scientific producer bind different acquisitions")
    current = receipt["full_extended_scientific"]
    revision = receipt["rig_domain_revision"]
    if (current["source_sha256"] != receipt["source_sha256"]
            or current["phase_shape_hw"] != revision["phase_shape_hw"]
            or current["native_origin_xy"] != revision["native_origin_xy"]
            or revision.get("physical_scale_approved") is not True):
        raise ValueError("Selected RAW/phase/measured-area grids differ")
    if receipt["baseline_extended_scientific"]["source_sha256"] != manifest["baseline"]["source_sha256"]:
        raise ValueError("The pre-injection baseline is not the run's declared source")
    return receipt, scientific, manifest


def _old_domain_row(manifest, run, frame_index):
    rows = [row for row in manifest["frames"] if row["run"] == run and row["frame_index"] == frame_index]
    if len(rows) != 1:
        raise ValueError("Original bottom-block domain has no unique source registration")
    return rows[0]


def _carried_domain(row, current_matrix, shape, reader):
    descriptor = row["parent_labels_map"]
    # Only membership is retained. Aqueous/gas categories never enter the new map.
    with Image.open(reader.path(descriptor)) as image:
        membership = np.asarray(image) != 255
    if tuple(membership.shape) != tuple(row["native_shape_hw"]):
        raise ValueError("Original domain and source registration use different grids")
    original_h = np.asarray(row["H_ac28_to_frame_native"], np.float64)
    if original_h.shape != (3, 3) or not np.isfinite(original_h).all() or abs(np.linalg.det(original_h)) < 1e-12:
        raise ValueError("Original domain source homography is invalid")
    warp = np.asarray(current_matrix) @ np.linalg.inv(original_h)
    carried = cv2.warpPerspective(membership.astype(np.uint8), warp, shape[::-1],
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return carried, dict(original_labels=descriptor, original_frame_index=int(row["frame_index"]),
        original_H_ac28_to_native=original_h.tolist(), original_domain_to_current=warp.tolist(),
        prediction_categories_used=False)


def _bottom_side(domain, photo_top, polygon, origin):
    """Established nearest raster-axis boundary partition, unchanged tie order."""
    h, w = domain.shape
    if photo_top.dtype != bool or photo_top.shape != domain.shape:
        raise ValueError("Bottom strip requires the selected image's photographed top")
    polygon = np.asarray(polygon, np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        raise ValueError("Selected inner-rig polygon is invalid")
    outer = np.zeros(domain.shape, np.uint8)
    cv2.fillPoly(outer, [np.rint((polygon-np.asarray(origin))*256).astype(np.int32)], 1, shift=8)
    outer = outer.astype(bool)
    left, right = outer.argmax(1), w-1-outer[:, ::-1].argmax(1)
    floor, surface = h-1-outer[::-1].argmax(0), photo_top.argmax(0)
    x = np.arange(w, dtype=np.float32)[None]
    bottom = np.zeros(domain.shape, bool)
    for first in range(0, h, 128):
        last = min(h, first+128)
        y = np.arange(first, last, dtype=np.float32)[:, None]
        side = np.stack(np.broadcast_arrays(y-surface[None], floor[None]-y,
            x-left[first:last, None], right[first:last, None]-x)).argmin(0)
        bottom[first:last] = domain[first:last] & (side == 1)
    return bottom


def selected_bottom_block(receipt, scientific, domain, current_matrix, original_descriptor, reader):
    """Carry exact Studio membership, or the original baseline for new sources."""
    source_row = receipt["binding"]["source"]
    manifest = reader.json(original_descriptor)
    baseline_row = _old_domain_row(manifest, receipt["run"], 0)
    baseline, baseline_proof = _carried_domain(baseline_row, current_matrix, domain.shape, reader)
    geometry = scientific["rig_geometry"]
    top_descriptor = geometry["frame_photo_top_keep"]
    with np.load(reader.path(top_descriptor), allow_pickle=False) as archive:
        top = archive[top_descriptor.get("array_key", "keep")].copy()
    side = _bottom_side(domain, top, geometry["frame_inner_polygon_native"],
        receipt["rig_domain_revision"]["native_origin_xy"])
    baseline_block = side & ~baseline
    proof = dict(original_domain_manifest=original_descriptor,
        baseline_policy="source-bound original run frame-zero domain carried by this acquisition's own H",
        baseline_domain=baseline_proof, current_ac28_to_canvas=np.asarray(current_matrix).tolist(),
        current_photo_top=top_descriptor, current_modelled_class_used=False,
        prediction_categories_used=False,
        method="historical added-strip nearest raster-axis inner boundary, tie order top,bottom,left,right")
    if source_row.get("studio_anchor"):
        historical = source_row["parent_source"]
        index = historical["frame_index"] if historical["source_changed"] else historical["parent_source_frame_index"]
        exact, exact_proof = _carried_domain(_old_domain_row(manifest, receipt["run"], index),
            current_matrix, domain.shape, reader)
        result = side & ~exact
        proof.update(policy="exact original Studio acquisition domain", exact_domain=exact_proof,
            baseline_vs_exact_changed_pixels=int(np.count_nonzero(result != baseline_block)),
            baseline_block_pixels=int(baseline_block.sum()), exact_block_pixels=int(result.sum()))
    else:
        result = baseline_block
        proof.update(policy="original run baseline membership for new acquisition",
            baseline_vs_exact_changed_pixels=None, exact_domain_available=False)
    return result, proof


def _prepare_response(frame, path_folder, parts, swi, scope, freeze, *, asset_reader=None):
    """Use the unchanged R2 native function with a call-local R3 access verifier."""
    def verify_current(path, *, asset_reader=None):
        return engine.verify_round3_freeze(path,
            **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    namespace = dict(native.__dict__, verify_round2_freeze=verify_current)
    prepare = FunctionType(native.prepare_round2_response.__code__, namespace,
        native.prepare_round2_response.__name__, native.prepare_round2_response.__defaults__)
    prepare.__kwdefaults__ = native.prepare_round2_response.__kwdefaults__
    response = prepare(frame, path_folder, swi_by_facies=swi,
        r1_frozen_calibration_path=parts["r1_freeze"]["path"],
        frozen_calibration=freeze if scope == "frozen_evaluation" else None,
        **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    response.update(round3_selected_native_adapter=dict(contract=CONTRACT,
        native_prepare_source_sha256=source.sha256(native.__file__),
        original_prepare_bytecode_used=asset_reader is None, global_modules_modified=False,
        access_verifier="actual round-three freeze before pixels; original R1 source-equivalence verification retained"))
    if asset_reader is not None:
        response["round3_selected_native_adapter"].update(
            production_io_hooks_used=True, execution_commit=asset_reader.code_commit,
            current_prepare_source_sha256=source.sha256(native.__file__),
            frozen_source_provenance=asset_reader.source_bindings)
    return response


def _no_outflow(frame, receipt_descriptor):
    occupied = frame.domain.any(0)
    first = frame.domain.argmax(0)
    contact = int(np.count_nonzero(occupied & np.isin(
        frame.phase_labels[first, np.arange(frame.domain.shape[1])], (1, 2))))
    return dict(no_co2_left_rig=contact == 0, source=receipt_descriptor,
        co2_contact_columns=contact, status="inferred_locked_phase_open_top_proxy", spatial_margin_px=0,
        limitation="No measured CO2 outlet; fixed phase/top rules may weaken optical evidence",
        measured_water_displacement_is_not_CO2_outlet=True)


def load_selected(*, entry, plan, reader=None, scope, frozen_calibration=None, asset_reader=None):
    """Load exact selected float RAW and fixed phase labels without fitting."""
    if asset_reader is None:
        asset_reader = reader if callable(getattr(reader, "resolve_path", None)) else getattr(reader, "asset_reader", None)
    reader = reader or asset_reader or source.VerifiedAssetReader()
    if asset_reader is not None and reader is not asset_reader and getattr(reader, "asset_reader", None) is not asset_reader:
        raise ValueError("Selected input reader and native path resolver must use the same cache")
    receipt, scientific, manifest = _admit(entry, plan, reader, scope, frozen_calibration,
        **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    parts = reader.json(plan["saved_parts_plan"])
    input_plan = reader.json(parts["r2_input_plan"])
    dose_inputs.validate_split(input_plan["split"])
    base = reader.json(input_plan["base_input_plan"])
    run, time = entry["run"], float(entry["elapsed_h"])
    recipes = reader.json(base["recipes"])
    recipe = next(row for row in recipes["runs"] if row["run"] == run)
    atmosphere = next(row for row in reader.json(base["atmosphere"])["runs"] if row["run"] == run)
    dose = next(row for row in reader.json(input_plan["dose_inventory"])["runs"] if row["run"] == run)
    epoch = next(row["epoch"] for row in reader.json(input_plan["light_epochs"])["runs"] if row["run"] == run)
    transport_descriptor = input_plan["clearwater_transports"][run]
    transport = reader.json(transport_descriptor)
    if transport.get("contract") != "ffac-common-native-clearwater-transport-v1" or transport.get("run") != run:
        raise ValueError("Selected source has no own source-bound clearwater transport")
    clearwater_reference = dict(receipt=dict(path=transport["clearwater_receipt"],
        sha256=transport["clearwater_receipt_sha256"]),
        H_ac28_to_clearwater_grid=transport["H_ac28_to_clearwater_grid"], geometry_binding=transport_descriptor)
    revision, current_descriptor = receipt["rig_domain_revision"], receipt["full_extended_scientific"]
    shape = tuple(map(int, revision["phase_shape_hw"]))
    current, valid = source.compose_scientific_raw(current_descriptor, reader)
    baseline_descriptor = receipt["baseline_extended_scientific"]
    baseline_key = (run, baseline_descriptor["source_sha256"], tuple(baseline_descriptor["phase_shape_hw"]),
        tuple(baseline_descriptor["native_origin_xy"]))
    for descriptor in source._scientific_assets(baseline_descriptor):
        reader.path(descriptor)
    if baseline_key not in reader.raw_baselines:
        reader.raw_baselines[baseline_key] = source.compose_scientific_raw(baseline_descriptor, reader)
    baseline, baseline_valid = reader.raw_baselines[baseline_key]
    geometry = current_descriptor["binding"]["geometry"]
    current_matrix = source.matrix_ac28_to_canvas(geometry, revision["native_origin_xy"])
    baseline_matrix = source.matrix_ac28_to_canvas(baseline_descriptor["binding"]["geometry"],
        baseline_descriptor["native_origin_xy"])
    warp = current_matrix @ np.linalg.inv(baseline_matrix)
    baseline = cv2.warpPerspective(baseline, warp, shape[::-1], flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    baseline_coverage = cv2.warpPerspective(baseline_valid.astype(np.float32), warp, shape[::-1],
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    native_manifest = reader.json(dict(path=geometry["native_manifest_path"], sha256=geometry["native_manifest_sha256"]))
    photometry_sha = hashlib.sha256(json.dumps(native_manifest["v5_photometry"],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    clear, clear_supported, clear_proof = source.registered_clearwater(clearwater_reference,
        current_matrix, shape, reader, expected_run=run,
        expected_native_manifest_sha256=geometry["native_manifest_sha256"], expected_photometry_sha256=photometry_sha)
    physical = source.transport_physical_reference(source.load_physical_reference(parts["physical_reference"], reader),
        current_matrix, shape)
    with Image.open(reader.path(receipt["labels_map"])) as image:
        labels = np.asarray(image).copy()
    if labels.dtype != np.uint8 or labels.shape != shape or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("Selected fixed categorical labels are invalid")
    with np.load(reader.path(revision["classes"]), allow_pickle=False) as archive:
        classes = {key: archive[key].copy() for key in ("rig_domain", "modelled", "filled_observed", "filled_unobserved")}
    if any(value.dtype != bool or value.shape != shape for value in classes.values()):
        raise ValueError("Selected rig class masks use a different grid")
    domain = classes["rig_domain"]
    if not np.array_equal(domain, labels != 255):
        raise ValueError("Selected labels and frozen rig domain disagree")
    with np.load(reader.path(revision["area_map"]), allow_pickle=False) as archive:
        area = np.asarray(archive[revision["area_map"].get("array_key", "pixel_area_mm2")], np.float64)/1e6
    if area.shape != shape or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Measured selected pixel areas must be finite and positive")
    bottom, bottom_proof = selected_bottom_block(receipt, scientific, domain, current_matrix,
        base["original_domain_reference"], reader)
    below = classes["filled_unobserved"] & ~bottom
    unknown = domain & ~physical["physical_supported_mask"]
    optical = valid & (baseline_coverage >= 1-1e-6) & clear_supported
    photo_support = (domain & optical & np.isfinite(current).all(-1) & np.isfinite(baseline).all(-1)
        & np.isfinite(clear).all(-1) & (current > 0).all(-1) & (baseline > 0).all(-1) & (clear > 0).all(-1))
    eligible = domain & ~bottom & ~below
    provenance = dict(contract=CONTRACT, source_receipt=entry["receipt"], scientific_receipt=entry["scientific_receipt"],
        source_binding=entry["source_binding"], source_frame_unchanged=True, phase_labels_modified=False,
        selected_camera_manifest=entry["run_manifest"], selected_target_h=entry["selected_target_h"],
        physical_reference=parts["physical_reference"], physical_reference_provisional=True,
        physical_boundary_error_px="10-20 px, Olav-accepted provisional v3 reference",
        baseline_to_current=warp.tolist(), raw_encoding="float32-camera-linear-rgb-exp(log)/65535-no-clipping",
        bottom_mass_allowed=False, unobserved_mass_allowed=False, bottom_strip_source=bottom_proof,
        undefined_facies_not_borrowed=True, measured_area=revision["area_map"], clear_water_reference=clear_proof,
        physical_unknown_pixels=int(unknown.sum()), supported_pixels=int((domain & optical & ~unknown).sum()),
        mass_eligible_pixels=int(eligible.sum()), optical_unresolved_pixels=int((domain & ~optical).sum()),
        dose_source=dose["dose_kind"], dose_qa=dose.get("qa"), round2_dose_inventory=input_plan["dose_inventory"],
        dose_nominal_fallback_reasons=dose.get("fallback_reasons", []), dose_clock_source=dose.get("clock_source"),
        measured_mfc_source=dose.get("measured_source"), nominal_setpoint_mass_g=dose_inputs.dose_at(dose, time, nominal=True),
        light_epoch=epoch, physical_limits=input_plan["physical_limits"], descriptive_out_of_fit=run in engine.DESCRIPTIVE_RUNS,
        caveat=source.CAVEATS.get(run, ""))
    prepared = source.CommonMassFrameInputs(run, int(entry["frame_index"]), time, current, baseline, clear,
        clear_supported, labels, physical["facies"], domain, domain & optical & ~unknown, eligible,
        bottom, below, unknown, domain & np.isin(physical["facies"], source.UNQUANTIFIED_FACIES),
        physical["depth_m"], physical["porosity"], area, recipe, source.atmosphere_at(atmosphere, time),
        dose_inputs.dose_at(dose, time), (.49, .90), provenance)
    frame = replace(source.to_native_frame(prepared),
        excluded_masks=dict(bottom_strip=bottom, below_photo_filled=below))
    original, _ = native.verify_original_round1_freeze(parts["r1_freeze"]["path"],
        **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    path_folder = original["path_folder"]
    response = _prepare_response(frame, path_folder, parts, input_plan["physical_limits"]["swi_by_facies"],
        scope, frozen_calibration, **({} if asset_reader is None else dict(asset_reader=asset_reader)))
    from darsia.fluidflower.tools.build_ffac_p1_final2_candidate import transported_floor
    floor = transported_floor(reader.array(parts["pooled_floor_curve"]),
        dict(H_ac28_to_frame_native=geometry["H_ac28_to_frame_native"], rig_domain_revision=revision))
    return dict(response=response, receipt=receipt, source_binding=entry["source_binding"],
        photo_support=photo_support, path_folder=path_folder, light_epoch=epoch,
        no_outflow_evidence=_no_outflow(frame, entry["receipt"]),
        frame_list_evidence=dict(accepted=True, source_binding=dict(run_manifest=entry["run_manifest"],
            source_sha256=receipt["source_sha256"], elapsed_h=time,
            selected_target=_target_match(entry, manifest)),
            policy="acquisition admitted by the sealed selected-camera manifest; no historical frame substitution"),
        floor_curve_canvas_y=floor, photo_sand_domain=domain)
