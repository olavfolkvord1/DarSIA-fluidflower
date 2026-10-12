"""Frozen native-centre, rig-edge and P1-final3 selected-frame inference.

This producer recreates the historical split between native scientific input
and the larger masked rig context. It does not replace the native centre by a
fresh whole-rig approximation, and never treats inherited labels as a replay.
"""
from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .phase_ffac_selected_phase import (
    CONTRACT, SelectedFrameRuleInputs, apply_locked_postprocessing,
    asset, bound_json, read_parent_envelopes, replay_difference,
)
from .phase_ffac_selected_sources import load_selected_scientific
from .phase_ffac_rig_camera_runtime import bound_path
from .tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, read_json, utc_now
from .tools import run_ffac_rig_camera_inference as rig


def _png(descriptor):
    with Image.open(bound_path(descriptor)) as image:
        return np.asarray(image).copy()


def _arrays(descriptor):
    with np.load(bound_path(descriptor), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _source_path(root, run, digest):
    return Path(root) / "selected_sources" / run / digest / "receipt.json"


def _require_source(receipt,source,run):
    digest=source["source_sha256"]
    if (receipt.get("complete") is not True or receipt.get("run")!=run
        or receipt.get("source_sha256")!=digest or receipt.get("elapsed_h")!=source["elapsed_h"]
        or receipt["native_scientific"]["source_sha256"]!=digest
        or receipt["full_extended_scientific"]["source_sha256"]!=digest):
        raise ValueError("Prepared selected RAW is bound to another acquisition/camera time")


def _native_temporal(core, full, original_record=None, *, pad=64):
    """Keep the original small temporal ring; new sources use a disjoint crop."""
    if original_record is not None and original_record.get("v5_extended_ring_path"):
        from .tools.ffac_v5_extended_reference import assemble_extended_log
        ring = dict(path=original_record["v5_extended_ring_path"],
                    sha256=original_record["v5_extended_ring_sha256"])
        pad = int(original_record["v5_extended_pad_px"])
        log = assemble_extended_log(core, bound_path(ring),
            ring_sha256=ring["sha256"], expected_pad_px=pad)
        origin = tuple(original_record["v5_extended_origin_xy"])
        proof = dict(policy="exact historical temporal native centre plus small frozen ring", ring=ring)
    else:
        x, y, width, height = rig.metadata.SOURCE_BOX
        if pad < 0 or x < pad or y < pad or y+height+pad > full.shape[1] or x+width+pad > full.shape[2]:
            raise ValueError("Selected temporal ring escapes its source-supported RAW canvas")
        log = full[:, y-pad:y+height+pad, x-pad:x+width+pad]
        origin = (-pad, -pad)
        proof = dict(policy="new source same-size temporal crop from its own frozen full RAW", pad_px=pad,
                     interpolation=False, historical_pixels_changed=False)
    return log, np.isfinite(log).all(0), origin, proof


def merge_native_and_rig(centre, fresh, fresh_support, rig_domain, observed, *, release,
                         old_domain, original_area, open_components, barrier_mask,
                         source_ids, fresh_pre_minimum, prior_cache=None):
    """Literal centre reuse pattern, own categorical fill, then released seam."""
    from .phase_rig_domain_inference import fill_rig_labels
    from .phase_gas_barrier_rule import apply_barrier_rule_b
    from .phase_ffac_old_domain_seam import revise_seam_labels
    from .phase_retained_source_inference import preserve_prior_categorical_fill

    shape = rig_domain.shape
    support = fresh_support.copy()
    edge_labels, edge_b = apply_barrier_rule_b(fresh, open_components, source_ids,
        pixel_area_mm2=.027053, barrier_mask=barrier_mask)
    x, y, width, height = rig.metadata.SOURCE_BOX
    central = np.asarray(centre.labels).copy()
    central_support = (np.asarray(centre.domain) & np.asarray(centre.flatfield_valid)
        & np.isin(central, (0,1,2)) & rig._crop(rig_domain))
    central[~central_support] = 255
    central_b, native_b = apply_barrier_rule_b(central, rig._crop(open_components), source_ids,
        pixel_area_mm2=.027053, barrier_mask=rig._crop(barrier_mask))
    merged = edge_labels.copy()
    crop = merged[y:y+height, x:x+width]
    crop[central_support] = central_b[central_support]
    crop[~central_support] = edge_labels[y:y+height, x:x+width][~central_support]
    support[y:y+height, x:x+width] = central_support | fresh_support[y:y+height, x:x+width]
    observed = observed | support
    fill_audit = {}
    kwargs = {}
    if prior_cache is not None:
        kwargs = dict(tie_break_reference=rig._canvas(prior_cache.labels, phase_shape_hw=shape),
            tie_break_mask=rig._canvas(prior_cache.tie_reference_eligible, False, phase_shape_hw=shape))
    filled, classes = fill_rig_labels(merged, rig_domain, observed, audit=fill_audit, **kwargs)
    prior_audit = None
    if prior_cache is not None:
        native, _, prior_audit = preserve_prior_categorical_fill(prior_cache, rig._crop(filled),
            modelled=rig._crop(classes["modelled"]),
            filled_observed=rig._crop(classes["filled_observed"]),
            filled_unobserved=rig._crop(classes["filled_unobserved"]))
        filled[y:y+height, x:x+width] = native
    # Fresh ring fill is a separate evaluation, just as in the saved seam
    # worker. It must not inherit source-bound old fill on the released ring.
    # The saved seam worker filled the fresh numerical decision before B;
    # revise_seam_labels subsequently applies B to the assembled graph.
    fresh_filled, fresh_classes = fill_rig_labels(fresh, rig_domain, observed)
    for name in ("modelled", "filled_observed", "filled_unobserved"):
        classes[name][release] = fresh_classes[name][release]
    old_pre = rig._canvas(np.asarray(centre.preminimum_gas, bool), False, phase_shape_hw=shape)
    revised, seam_audit, seam_masks = revise_seam_labels(filled, fresh_filled, release,
        old_domain, old_pre, fresh_pre_minimum, classes["modelled"], original_area,
        open_components=open_components, source_component_ids=source_ids, barrier_mask=barrier_mask)
    return revised, classes, dict(edge_rule_b=edge_b, native_rule_b=native_b,
        fill=fill_audit, inherited_fill=prior_audit, seam=seam_audit), seam_masks


class SelectedPhaseEvaluator:
    """One worker/run, one frozen optical reference, bounded transient images."""
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
        self.baseline_temporal, self.baseline_temporal_valid, self.baseline_origin, self.baseline_temporal_proof = _native_temporal(
            self.baseline_native, self.baseline_full, self.native["reference"])
        self._previous_cache = None

    def saved(self, digest):
        """Resume only a complete frame with unchanged source/code/outputs."""
        source = self.sources[digest]
        path = self.root / "selected_phase_frames" / self.run / (digest + ".json")
        if not path.exists():
            return None
        saved = bound_json(asset(path))
        expected = dict(job_sha256=self.plan["job_sha256"],
            pipeline_sha256=self.plan["pipeline_sha256"],source=source)
        if (saved.get("contract") != CONTRACT or saved.get("complete") is not True
                or saved.get("binding") != expected
                or saved.get("source_sha256") != digest
                or saved.get("elapsed_h") != source["elapsed_h"]
                or saved.get("run") != self.run):
            raise ValueError("Selected phase resume changed its source/history/code binding")
        for descriptor in [*saved["maps"].values(),saved["labels_map"],
            saved["rig_domain_revision"]["classes"],saved["rig_domain_revision"]["area_map"],
            saved["operator_masks"]]:
            bound_path(descriptor)
        return asset(path)

    def _photo(self, source, receipt):
        if source.get("studio_anchor") and not source["parent_source"]["source_changed"]:
            index = source["parent_source"]["parent_source_frame_index"]
            row = self.native["frame_sources"][index]
            mask = np.load(bound_path(dict(path=row["v5_photo_sand_mask_path"],
                sha256=row["v5_photo_sand_mask_sha256"])), allow_pickle=False)
        else:
            from .tools.ffac_v5_frame_context import _source_photo_grid, project_source_photo_safe_core
            geometry = receipt["geometry_frame"]
            sand = _source_photo_grid(geometry["photo_sand_reference_mask_path"],
                geometry["photo_sand_reference_mask_sha256"])
            mask = project_source_photo_safe_core(sand, geometry["H_ac28_to_frame_native"],
                shape_hw=tuple(rig.metadata.SOURCE_SHAPE), enclosed_hole_policy="fill_enclosed_v1")
            mask &= rig._crop(rig.load_array(receipt["rig_geometry"]["frame_photo_top_keep"]))
        return mask

    def _components(self, source, receipt, domain):
        from .phase_gas_barrier_runtime import read_components
        from .phase_ffac_barrier_extension import extend_frozen_barrier_components
        if source.get("studio_anchor") and not source["parent_source"]["source_changed"]:
            parent = bound_json(source["parent_source"]["retained_source_cache"]["parent_frame_receipt"])
            component_asset = parent["binding"]["components"]
            arrays = read_components(component_asset, rig.metadata.SOURCE_SHAPE)
            ids = rig._canvas(arrays["open_components"], 0, phase_shape_hw=domain.shape)
            barrier = rig._canvas(arrays["barrier_mask"], False, phase_shape_hw=domain.shape)
            sources = parent["qa"]["source_component_ids"]
            proof = dict(exact_same_acquisition=True, source_components=component_asset)
        else:
            reference = self.plan["frozen_barrier_reference"]
            metadata = bound_json(reference["geometry_manifest"])
            arrays = read_components(reference["components"], rig.metadata.SOURCE_SHAPE)
            transform = rig.canvas_homography(receipt["rig_geometry"]["H_ac28_to_frame_native"])
            ids = cv2.warpPerspective(rig._canvas(arrays["open_components"],0,phase_shape_hw=domain.shape),
                transform, domain.shape[::-1], flags=cv2.INTER_NEAREST)
            barrier = cv2.warpPerspective(rig._canvas(arrays["barrier_mask"],False,phase_shape_hw=domain.shape).astype(np.uint8),
                transform, domain.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)
            sources = metadata["source_component_ids"]
            proof = dict(exact_same_acquisition=False, frozen_reference=reference,
                own_source_H=receipt["rig_geometry"]["H_ac28_to_frame_native"])
        arrays, extension = extend_frozen_barrier_components(ids, barrier, domain)
        return arrays, tuple(sources), dict(proof, extension=extension)

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
                _, previous_core, _, previous_raw, previous_raw_valid = self._previous_cache
            else:
                previous_core, _, previous_raw, previous_raw_valid = load_selected_scientific(previous_receipt)
            previous_h = np.asarray(previous_receipt["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
            previous_full, previous_support = rig.align_scientific(previous_raw, previous_raw_valid,
                rig.canvas_homography(current_h @ np.linalg.inv(previous_h)), target_shape_hw=full.shape[1:])
            original_record = None
            if previous_source.get("studio_anchor") and not previous_source["parent_source"]["source_changed"]:
                original_record = self.native["frame_sources"][previous_source["parent_source"]["parent_source_frame_index"]]
            previous_native_temporal, previous_native_valid, previous_origin, _ = _native_temporal(
                previous_core, previous_raw, original_record,
                pad=int(self.native["reference"].get("v5_extended_pad_px", 64)))
        context, observed = rig.geometry_context(scientific["rig_geometry"], native_valid, audit)
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
                H_B_current=current_h, photo_sand_mask=self._photo(source, scientific),
                spacing_m=(.00016448,.00016448), source_origin_xy=self.baseline_origin,
                flat_baseline_log=self.baseline_native, flat_baseline_valid=self.baseline_native_valid,
                previous_log=previous_native_temporal, previous_valid=previous_native_valid,
                H_B_previous=previous_h, previous_origin_xy=previous_origin or (0,0),
                labelled_probability_encoding=probability is not None and probability["binding"]["labelled"] is True,
                probability_receipt=probability,
                input_provenance=dict(source=asset(receipt_path),previous_source_sha256=previous_digest,
                    history_policy=self.inventory["history_policy"], baseline_temporal=self.baseline_temporal_proof))
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
            self._previous_cache = (digest,native,native_valid,full,full_valid)
        return dict(labels=labels,classes=classes,pixel_area_mm2=area,
            native_inference=centre_audit,edge_inference=edge_audit,qa=qa,
            source=source,scientific=scientific,seam_masks=masks,post_masks=post_masks,
            context_parameters=context.parameters)

    def save(self, result):
        source, labels = result["source"], result["labels"]
        existing = self.saved(source["source_sha256"])
        if existing is not None:
            raise FileExistsError("Computed selected phase frame already exists; resume must skip inference")
        index = int(source["work_index"])
        maps, label_asset = rig.write_maps(self.root,self.run,index,self.plan["pipeline_sha256"],labels,
                                          phase_shape_hw=labels.shape)
        payload=io.BytesIO()
        np.savez_compressed(payload,**result["classes"])
        classes=rig._write_bytes(self.root/"selected_context"/self.run/(source["source_sha256"]+".classes.npz"),payload.getvalue())
        payload=io.BytesIO();np.savez_compressed(payload,pixel_area_mm2=result["pixel_area_mm2"])
        area=rig._write_bytes(self.root/"selected_context"/self.run/(source["source_sha256"]+".area.npz"),payload.getvalue())
        operator_masks = {"seam_"+name:np.asarray(mask,bool) for name,mask in result["seam_masks"].items()}
        operator_masks.update({"post_"+name:np.asarray(mask,bool) for name,mask in result["post_masks"].items()})
        payload=io.BytesIO();np.savez_compressed(payload,shape_hw=np.asarray(labels.shape,np.int64),
            **{name:np.packbits(mask,bitorder="little") for name,mask in operator_masks.items()})
        masks=rig._write_bytes(self.root/"selected_context"/self.run/(source["source_sha256"]+".operator.packbits.npz"),payload.getvalue())
        receipt=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),
            binding=dict(job_sha256=self.plan["job_sha256"],pipeline_sha256=self.plan["pipeline_sha256"],source=source),
            run=self.run,frame_index=index,elapsed_h=source["elapsed_h"],source_sha256=source["source_sha256"],
            selected=source["selected"],studio_anchor=source["studio_anchor"],maps=maps,labels_map=label_asset,
            operator_masks=masks,operator_mask_names=sorted(operator_masks),
            full_extended_scientific=result["scientific"]["full_extended_scientific"],
            baseline_extended_scientific=result["scientific"]["baseline_extended_scientific"],
            rig_domain_revision=dict(contract="ffac-rig-domain-revision-v1",canvas_wh=rig.metadata.CANVAS_WH,
                phase_shape_hw=list(labels.shape),native_origin_xy=rig.metadata.ORIGIN_XY,fixed_area_mm2=.027053,
                area_map=dict(area,array_key="pixel_area_mm2"),classes=classes,parameters=result["context_parameters"],
                physical_scale_approved=True,physical_scale_revision=self.plan["physical_scale_revision"],mass_eligible=False),
            qa=result["qa"],training=False,annotations_read=False,bottom_mass_allowed=False,mass_eligible=False,
            inference_performed=True,parent_changed=False,automatic_activation=False)
        path=self.root/"selected_phase_frames"/self.run/(source["source_sha256"]+".json")
        atomic_json(path,receipt)
        return asset(path)


__all__=["SelectedPhaseEvaluator","merge_native_and_rig"]
