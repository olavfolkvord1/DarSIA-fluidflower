"""Corrected historical native input routing for new selected-camera phases.

All frozen optical, regional, seam, barrier and P1-final3 operators are inherited
unchanged. Only the caller's native support/photo mask and temporal convention
are corrected. Full-rig scientific sidecars and edge inference remain intact.
This producer must be bound in a new immutable plan; it does not rewrite the
stopped producer or admit selected bulk inference without the strict replay.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .phase_ffac_selected_inference import (
    SelectedPhaseEvaluator as OriginalSelectedPhaseEvaluator,
    _source_path, _require_source, _png, merge_native_and_rig,
)
from .phase_ffac_selected_native_inputs_corrected import (
    original_native_input, original_photo_sand, original_temporal_input,
    new_native_input, new_temporal_input, temporal_policy,
)
from .phase_ffac_selected_phase import (
    SelectedFrameRuleInputs, apply_locked_postprocessing,
    asset, bound_json, read_parent_envelopes, replay_difference,
)
from .phase_ffac_selected_sources import load_selected_scientific
from .phase_ffac_rig_camera_runtime import bound_path
from .tools.refit_ffac_g3_downstream import canonical_sha256
from .tools import run_ffac_rig_camera_inference as rig

CORRECTION_CONTRACT = "ffac-selected-historical-native-input-correction-v2"


class SelectedPhaseEvaluator(OriginalSelectedPhaseEvaluator):
    """Same frozen inference with the exact historical native input convention."""
    def __init__(self, root, plan, run, *, backend="auto"):
        from .phase_ffac_selected_native_centre import NativeCentreEvaluator
        self.root, self.plan, self.run, self.backend = Path(root), plan, run, backend
        self.source_root = Path(plan.get("scientific_source_root",root))
        self.state = plan["runs"][run]
        self.inventory = bound_json(self.state["source_inventory"])
        self.sources = {row["source_sha256"]: row for row in self.inventory["sources"]}
        self.chain = plan["parent_chain"]
        self.native = bound_json(self.state["parent_native_manifest"])
        heads = self.chain["heads"]
        self.centre = NativeCentreEvaluator(plan["optical_candidate"], bound_path(heads["g3"]),
                                            bound_path(heads["regional"]), threads=1)
        self.envelopes = read_parent_envelopes(self.chain, run)
        floor_asset = self.chain["fixed_floor_curve"]
        self.floor_curve = np.load(bound_path(floor_asset), allow_pickle=False)
        with np.load(bound_path(self.chain["fixed_zones"]), allow_pickle=False) as archive:
            self.zones = archive["zones"].copy()
        baseline_source = self.sources[self.inventory["baseline_source_sha256"]]
        self.baseline_source = baseline_source
        self.baseline_receipt = bound_json(asset(_source_path(self.source_root, run, baseline_source["source_sha256"])))
        _require_source(self.baseline_receipt,baseline_source,run)
        self.baseline_native, self.baseline_native_valid, self.baseline_full, self.baseline_valid = load_selected_scientific(self.baseline_receipt)
        baseline_input = original_native_input(self.native["reference"],
            scientific_native_log=self.baseline_native)
        self.baseline_native, self.baseline_native_valid = baseline_input.log, baseline_input.valid
        baseline_temporal = original_temporal_input(baseline_input, self.native["reference"])
        self.baseline_temporal = baseline_temporal.log
        self.baseline_temporal_valid = baseline_temporal.valid
        self.baseline_origin = baseline_temporal.origin_xy
        self.baseline_temporal_proof = dict(baseline_temporal.provenance,
            native_input=baseline_input.provenance)
        self.native_temporal_policy = temporal_policy(self.native["reference"])
        self._previous_cache = None

    def _original_record(self, source):
        if source.get("studio_anchor") and not source["parent_source"]["source_changed"]:
            row = self.native["frame_sources"][source["parent_source"]["parent_source_frame_index"]]
            if row.get("source_sha256") != source["source_sha256"]:
                raise ValueError("Original masked native descriptor belongs to another acquisition")
            return row
        return None

    def _photo_input(self, source, receipt):
        if self._original_record(source) is not None:
            probability = bound_json(source["parent_source"]["retained_source_cache"]["parent_probability_receipt"])
            return original_photo_sand(probability)
        from .tools.ffac_v5_frame_context import _source_photo_grid, project_source_photo_safe_core
        geometry = receipt["geometry_frame"]
        descriptor = dict(path=geometry["photo_sand_reference_mask_path"],
            sha256=geometry["photo_sand_reference_mask_sha256"])
        sand = _source_photo_grid(descriptor["path"], descriptor["sha256"])
        mask = project_source_photo_safe_core(sand, geometry["H_ac28_to_frame_native"],
            shape_hw=tuple(rig.metadata.SOURCE_SHAPE), enclosed_hole_policy="fill_enclosed_v1")
        proof = dict(contract=CORRECTION_CONTRACT, reference_photo_mask=descriptor,
            H_ac28_to_frame_native=geometry["H_ac28_to_frame_native"],
            source_sha256=source["source_sha256"], phase_labels_used=False,
            premature_top_rule_applied=False, enclosed_hole_policy="fill_enclosed_v1")
        return mask, dict(proof,sha256=canonical_sha256(proof))

    def _native_input(self, source, receipt, log, valid):
        photo, proof = self._photo_input(source, receipt)
        row = self._original_record(source)
        if row is not None:
            native = original_native_input(row, scientific_native_log=log)
        else:
            native = new_native_input(log,valid,photo,
                source_binding=dict(source_sha256=source["source_sha256"],
                    elapsed_h=source["elapsed_h"],prepared_scientific=receipt["native_scientific"]),
                photo_mask_binding=proof)
        return native, photo, proof

    def evaluate(self, digest):
        from .phase_rig_domain_inference import infer_native, fill_rig_labels
        from .phase_gas_barrier_rule import apply_barrier_rule_b
        from .phase_retained_source_inference import (prepare_retained_source_cache,
            _outer_domain_ring, _old_domain_interpolation_footprint)
        from .phase_gas_t4c_placements import prepare_shift_context
        from .tools.build_ffac_p1_final2_candidate import transported_floor, native_l1

        started = time.perf_counter()
        source = self.sources[digest]
        receipt_path = _source_path(self.source_root, self.run, digest)
        scientific = bound_json(asset(receipt_path))
        _require_source(scientific,source,self.run)
        native, native_valid, full, full_valid = load_selected_scientific(scientific)
        # Rig-edge inference keeps the original unmasked source representation.
        rig_native, rig_native_valid = native, native_valid
        native_input, photo, photo_proof = self._native_input(source, scientific, native, native_valid)
        native, native_valid = native_input.log, native_input.valid
        prepared = bound_json(scientific["prepared_source"])
        audit = bound_json(scientific["native_scientific"]["receipt"])["producer_audit"]
        current_h = np.asarray(scientific["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
        aligned_base, baseline_support = rig.align_scientific(self.baseline_full, self.baseline_valid,
            rig.canvas_homography(current_h), target_shape_hw=full.shape[1:])
        previous_full = previous_support = previous_native_temporal = previous_native_valid = previous_h = previous_origin = None
        previous_digest = source["previous_source_sha256"]
        if previous_digest is not None:
            previous_source = self.sources[previous_digest]
            previous_receipt = bound_json(asset(_source_path(self.source_root, self.run, previous_digest)))
            _require_source(previous_receipt,previous_source,self.run)
            if self._previous_cache is not None and self._previous_cache[0] == previous_digest:
                _, previous_core, previous_core_valid, previous_raw, previous_raw_valid = self._previous_cache
            else:
                previous_core, previous_core_valid, previous_raw, previous_raw_valid = load_selected_scientific(previous_receipt)
            previous_h = np.asarray(previous_receipt["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
            previous_full, previous_support = rig.align_scientific(previous_raw, previous_raw_valid,
                rig.canvas_homography(current_h @ np.linalg.inv(previous_h)), target_shape_hw=full.shape[1:])
            previous_native_input, _, _ = self._native_input(previous_source, previous_receipt,
                previous_core, previous_core_valid)
            original_record = self._original_record(previous_source)
            if original_record is not None:
                previous_temporal = original_temporal_input(previous_native_input, original_record)
            else:
                previous_temporal = new_temporal_input(previous_native_input, previous_raw,
                    native_box_xywh=rig.metadata.SOURCE_BOX, frozen_policy=self.native_temporal_policy)
            previous_native_temporal, previous_native_valid = previous_temporal.log, previous_temporal.valid
            previous_origin = previous_temporal.origin_xy
        context, observed = rig.geometry_context(scientific["rig_geometry"], rig_native_valid, audit)
        original_area = context.pixel_area_mm2
        measured_revision = self.plan["physical_scale_revision"]
        if source.get("studio_anchor"):
            old_producer = bound_json(self.state["original_producer_receipts"][digest])
            original_area = rig.load_array(old_producer["rig_domain_revision"]["area_map"])
            measured_parent = bound_json(source["parent_receipt"])
            area = rig.load_array(measured_parent["rig_domain_revision"]["area_map"])
        else:
            area = (original_area * float(measured_revision["area_factor"])).astype(original_area.dtype)
        fresh, fresh_support, edge_audit, decisions = infer_native(self.chain["base_candidate"],
            full, aligned_base, context.rig_domain, aligned_previous_log=previous_full,
            pixel_area_mm2=original_area, source_baseline_log=self.baseline_full,
            h_baseline_to_current=rig.canvas_homography(current_h), source_baseline_valid=self.baseline_valid,
            return_decision_layers=True)
        fresh_support &= context.rig_domain & full_valid & baseline_support
        if previous_support is not None:
            fresh_support &= previous_support
        fresh[~fresh_support] = 255
        components, source_ids, barrier_proof = self._components(source, scientific, context.rig_domain)
        prior_cache = None
        special_masked_anchor = source.get("studio_anchor") and source["parent_source"]["source_changed"]
        if special_masked_anchor:
            base, base_b = apply_barrier_rule_b(fresh, components["open_components"],source_ids,
                pixel_area_mm2=.027053,barrier_mask=components["barrier_mask"])
            base, classes = fill_rig_labels(base, context.rig_domain, observed | fresh_support,
                                            hidden=context.hardware_excluded)
            centre_audit = dict(inference_performed=True, branch="historical changed-source masked rig path",
                cached_native_centre_used=False, original_source_changed=True)
            seam_audit = dict(rule_b=base_b, branch="historical source-change seam bypass")
            masks = {}
        else:
            probability = None
            if source.get("studio_anchor"):
                cached = source["parent_source"]["retained_source_cache"]
                probability = bound_json(cached["parent_probability_receipt"])
                prior_cache = prepare_retained_source_cache(cached, run=self.run,
                    frame_index=source["parent_source"]["parent_source_frame_index"],
                    current_source_sha256=digest, current_log=native,
                    allowed_domain=rig._crop(context.rig_domain), producer_audit=audit)
            centre = self.centre.infer(native, current_valid=native_valid,
                source_baseline_log=self.baseline_temporal, baseline_valid=self.baseline_temporal_valid,
                H_B_current=current_h, photo_sand_mask=photo,
                spacing_m=(.00016448,.00016448), source_origin_xy=self.baseline_origin,
                flat_baseline_log=self.baseline_native, flat_baseline_valid=self.baseline_native_valid,
                previous_log=previous_native_temporal, previous_valid=previous_native_valid,
                H_B_previous=previous_h, previous_origin_xy=previous_origin or (0,0),
                labelled_probability_encoding=probability is not None and probability["binding"]["labelled"] is True,
                probability_receipt=probability,
                input_provenance=dict(source=asset(receipt_path),previous_source_sha256=previous_digest,
                    history_policy=self.inventory["history_policy"], baseline_temporal=self.baseline_temporal_proof,
                    historical_native_input=native_input.provenance, photo_domain=photo_proof,
                    previous_native_temporal=None if previous_digest is None else previous_temporal.provenance))
            working = centre.provenance["probability_encoding"]["regional_working_shape_hw"]
            native_domain = centre.domain
            release_native, release_audit = _old_domain_interpolation_footprint(native_domain,working)
            release_native |= _outer_domain_ring(native_domain)
            release = rig._canvas(release_native,False,phase_shape_hw=full.shape[1:]) & context.rig_domain
            old_domain = rig._canvas(native_domain,False,phase_shape_hw=full.shape[1:])
            base, classes, composition, masks = merge_native_and_rig(centre,fresh,fresh_support,
                context.rig_domain,observed,release=release,old_domain=old_domain,
                original_area=original_area,open_components=components["open_components"],
                barrier_mask=components["barrier_mask"],source_ids=source_ids,
                fresh_pre_minimum=decisions["pre_minimum_gas"],prior_cache=prior_cache)
            centre_audit = centre.provenance
            seam_audit = dict(composition["seam"], composition=composition, released_footprint=release_audit)
        context_domain = context.rig_domain & full_valid & baseline_support
        if previous_support is not None:
            context_domain &= previous_support
        shift = prepare_shift_context(self.chain["base_candidate"],bound_path(self.chain["heads"]["g3"]),
            bound_path(self.chain["heads"]["regional"]),full,aligned_base,context_domain,
            run=self.run,light_epochs=self.plan["light_epochs"],
            expected_base_pixel_sha256=self.chain["heads"]["g3"]["sha256"],
            expected_base_regional_sha256=self.chain["heads"]["regional"]["sha256"],
            aligned_previous_log=previous_full,source_baseline_log=self.baseline_full,
            h_baseline_to_current=rig.canvas_homography(current_h), source_baseline_valid=self.baseline_valid,
            base_labels=base)
        row = dict(H_ac28_to_frame_native=scientific["rig_geometry"]["H_ac28_to_frame_native"],
            rig_domain_revision=dict(phase_shape_hw=list(base.shape),native_origin_xy=rig.metadata.ORIGIN_XY))
        floor = transported_floor(self.floor_curve,row)
        below = context.rig_domain & (np.arange(base.shape[0])[:,None] > floor[None,:])
        corrected = context.rig_domain & ~below
        inputs = SelectedFrameRuleInputs(base,classes["modelled"],
            classes["filled_observed"]|classes["filled_unobserved"],
            rig.load_array(scientific["rig_geometry"]["frame_photo_top_keep"]),
            components["open_components"],components["barrier_mask"],source_ids,
            seam_audit["rule_b"],area,corrected,below,native_l1(self.zones,row),
            float(source["elapsed_h"]),self.run,digest,centre_audit,
            dict(edge_audit,inference_performed=True))
        labels, qa, post_masks = apply_locked_postprocessing(inputs,shift,self.chain["delta"],
            self.envelopes,backend=self.backend)
        classes.update(rig_domain=corrected,hardware_excluded=context.hardware_excluded,below_sand_floor=below)
        for name in ("modelled","filled_observed","filled_unobserved"):
            classes[name] &= corrected
        replay = None
        if source.get("studio_anchor"):
            replay = replay_difference(_png(measured_parent["labels_map"]),labels,area,
                run=self.run,frame_index=source["studio_frame_index"],source_sha256=digest,
                elapsed_h=source["elapsed_h"])
        qa.update(source_history=dict(previous_source_sha256=previous_digest,
            selected=source["selected"],studio_anchor=source["studio_anchor"],
            policy=self.inventory["history_policy"]),base_seam=seam_audit,
            barrier_geometry=barrier_proof,physical_scale_revision=measured_revision,
            strict_studio_replay=replay,seconds=time.perf_counter()-started)
        qa["phase_area"]=dict(co2_pixels=int(np.isin(labels,(1,2)).sum()),
            gas_pixels=int((labels==2).sum()),aqueous_pixels=int((labels==1).sum()),
            co2_cm2=float(area[np.isin(labels,(1,2))].sum(dtype=np.float64)/100),
            gas_cm2=float(area[labels==2].sum(dtype=np.float64)/100),
            aqueous_cm2=float(area[labels==1].sum(dtype=np.float64)/100),
            physical_scale_approved=True,mass_computed=False)
        # Only selected sources enter policy-A history. Probe anchors never
        # replace the cached predecessor for the next selected acquisition.
        if source.get("selected") or source.get("baseline"):
            # Retain the unmasked source centre for exact full-rig context; the
            # source-bound native mask is applied separately at every use.
            self._previous_cache = (digest,rig_native,rig_native_valid,full,full_valid)
        return dict(labels=labels,classes=classes,pixel_area_mm2=area,
            native_inference=centre_audit,edge_inference=edge_audit,qa=qa,
            source=source,scientific=scientific,seam_masks=masks,post_masks=post_masks,
            context_parameters=context.parameters)


__all__ = ["SelectedPhaseEvaluator", "CORRECTION_CONTRACT"]
