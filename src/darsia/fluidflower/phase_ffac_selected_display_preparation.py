"""Pre-render source-bound photographs before phase inference completes.

Only the existing geometric domain and photographic transfers run here. A
preparation is not a phase/display completion receipt. Final publication needs
an actual phase producer and an exactly equal domain; its JPEGs are reused.
"""
from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .phase_ffac_selected_display import CONTRACT as DISPLAY_CONTRACT, MODES, render_arrays
from .phase_ffac_selected_display_baseline_roles import BaselineRoleSelectedDisplayRenderer
from .phase_ffac_selected_inventory_scope_admission import verify_ready_source_scope
from .phase_ffac_selected_policy_a_baseline_qa import EvidenceReader, asset
from .phase_ffac_selected_sources import load_selected_scientific
from .tools import run_ffac_rig_camera_inference as rig
from .tools.build_ffac_p1_final2_candidate import transported_floor
from .tools.refit_ffac_g3_downstream import atomic_json, utc_now

CONTRACT = "ffac-selected-source-ready-display-preparation-v1"
DOMAIN_CONTRACT = "ffac-selected-source-only-display-domain-v1"


def derive_display_domain(scientific, native_valid, producer_audit, plan, *, reader=None):
    """The literal geometric part of SelectedPhaseEvaluator.evaluate()."""
    reader = EvidenceReader() if reader is None else reader
    geometry = scientific["rig_geometry"]
    context, _ = rig.geometry_context(geometry, native_valid, producer_audit)
    curve_descriptor = plan["parent_chain"]["fixed_floor_curve"]
    curve = reader.array(curve_descriptor, cache=True)
    # Identical row construction and > floor comparison to the producer.
    row = dict(H_ac28_to_frame_native=geometry["H_ac28_to_frame_native"],
        rig_domain_revision=dict(phase_shape_hw=list(context.rig_domain.shape),
            native_origin_xy=rig.metadata.ORIGIN_XY))
    floor = transported_floor(curve, row)
    below = context.rig_domain & (np.arange(context.rig_domain.shape[0])[:, None] > floor[None, :])
    domain = context.rig_domain & ~below
    if domain.dtype != bool or domain.shape != tuple(geometry["phase_shape_hw"]):
        raise ValueError("Source-only photographic domain changed the rig grid")
    return domain, dict(contract=DOMAIN_CONTRACT,
        rig_geometry=deepcopy(geometry), floor_curve=deepcopy(curve_descriptor),
        origin_xy=list(rig.metadata.ORIGIN_XY), shape_hw=list(domain.shape),
        domain_pixels=int(domain.sum()), below_floor_pixels=int(below.sum()),
        floor_comparison="row > transported pooled sand-floor line",
        geometry_context_operator=asset(rig.__file__),
        floor_operator=asset(__import__(transported_floor.__module__, fromlist=['']).__file__),
        phase_labels_read=False, inference_performed=False)


def verify_source(descriptor, source, plan, reader):
    if plan.get("execution_operation") is not None:
        from .phase_ffac_selected_execution_admission import verify_ready_source as verify_ready_source_execution
        return verify_ready_source_execution(descriptor, source, plan, reader)
    return verify_ready_source_scope(descriptor, source, plan, reader)


def _photo_source(scientific, run, reader):
    receipt = reader.record(scientific["native_scientific"]["receipt"])
    audit = receipt["producer_audit"]
    renderer = audit["renderer"]
    camera = audit["sources"]["camera_jpeg"]
    if (renderer["sources"]["camera_jpeg_sha256"] != camera["sha256"]
            or renderer.get("run", "").casefold() != run):
        raise ValueError("Camera Original renderer belongs to another source/run")
    return audit, renderer, camera


def _verify_preparation(record, binding, reader):
    if (record.get("contract") != CONTRACT or record.get("complete") is not True
            or record.get("binding") != binding
            or set(record.get("photographs", {})) != set(MODES)):
        raise ValueError("Photographic preparation changed source/reference/floor/renderer binding")
    for item in [*record["photographs"].values(), record["support_masks"], record["domain_mask"]]:
        reader.path(item)
    return record


class SourceReadyDisplayPreparer(BaselineRoleSelectedDisplayRenderer):
    """One registered reference/run, no phase/head allocations or model calls."""
    def __init__(self, root, plan, run, *, backend="cpu", reader=None):
        self.reader = EvidenceReader() if reader is None else reader
        inventory = self.reader.record(plan["runs"][run]["source_inventory"])
        baseline = next(row for row in inventory["sources"]
            if row["source_sha256"] == inventory["baseline_source_sha256"])
        source_root = Path(plan["scientific_source_root"])
        verify_source(asset(source_root/"selected_sources"/run/baseline["source_sha256"]/"receipt.json"),
            baseline, plan, self.reader)
        # The unchanged display constructor loads ONLY the separately prepared
        # historical baseline role, not the current frame-zero full image.
        super().__init__(root, plan, run, backend=backend)

    def _source(self, digest):
        source = self.sources[digest]
        if not (source.get("selected") or source.get("baseline")):
            raise ValueError("Only selected acquisitions and the exact baseline get photographs")
        descriptor = asset(self.source_root/"selected_sources"/self.run/digest/"receipt.json")
        scientific = verify_source(descriptor, source, self.plan, self.reader)
        return source, descriptor, scientific

    def preparation_binding(self, source, scientific_descriptor, scientific):
        base = self.binding(source, None, scientific_descriptor)
        base.pop("phase_receipt")
        return dict(base, preparation_code=asset(__file__),
            source_preparation=self.plan["phase_campaign_binding"]["source_preparation"],
            source_inventory=self.plan["runs"][self.run]["source_inventory"],
            domain_geometry=scientific["rig_geometry"],
            floor_curve=self.plan["parent_chain"]["fixed_floor_curve"],
            phase_receipt_required_for_finalization=True)

    def prepare(self, digest):
        started = time.perf_counter()
        source, scientific_asset, scientific = self._source(digest)
        target = self.root/"display_preparations"/self.run/digest
        receipt_path = target/"receipt.json"
        binding = self.preparation_binding(source, scientific_asset, scientific)
        if receipt_path.exists():
            _verify_preparation(self.reader.record(asset(receipt_path)), binding, self.reader)
            return asset(receipt_path)
        if target.exists():
            raise ValueError("Unreceipted photographic preparation requires a new output namespace")
        _, native_valid, full, valid = load_selected_scientific(scientific)
        audit, renderer, camera_asset = _photo_source(scientific, self.run, self.reader)
        domain, domain_audit = derive_display_domain(scientific, native_valid, audit, self.plan, reader=self.reader)
        with Image.open(self.reader.path(camera_asset)) as image:
            if list(image.size) != renderer["camera_jpeg_wh"]:
                raise ValueError("Camera photograph changed its source-bound renderer grid")
            camera = np.asarray(image.convert("RGB")).copy()
        photographs, masks, rendering = render_arrays(full, valid, domain, self.baseline,
            scientific["geometry_frame"]["H_v5_B_to_frame_native"], camera,
            renderer["review_to_camera_h"], origin_xy=self.origin, backend=self.backend)
        files = {}
        for name, pixels in photographs.items():
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="JPEG", quality=100, subsampling=0)
            files[name] = rig._write_bytes(target/(name+".jpg"), stream.getvalue())
        stream = io.BytesIO(); np.savez_compressed(stream, **masks)
        support = rig._write_bytes(target/"support_masks.npz", stream.getvalue())
        stream = io.BytesIO(); np.save(stream, domain, allow_pickle=False)
        domain_asset = rig._write_bytes(target/"rig_domain.npy", stream.getvalue())
        record = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
            run=self.run, source_sha256=digest, elapsed_h=source["elapsed_h"], frame_index=source["work_index"],
            phase_shape_hw=list(full.shape[1:]), native_origin_xy=list(self.origin),
            baseline_source_sha256=self.baseline_sha, photographs=files, support_masks=support,
            support_mask_names=sorted(masks), support_counts={name:int(mask.sum()) for name,mask in masks.items()},
            domain_mask=domain_asset, domain_audit=domain_audit, camera_source=camera_asset,
            renderer_audit=rendering, display_only=True, scientific_inputs_changed=False,
            maps_changed=False, labels_changed=False, inference_performed=False,
            phase_receipt_read=False, standard_display_receipt_finalized=False,
            seconds=time.perf_counter()-started)
        atomic_json(receipt_path, record)
        return asset(receipt_path)

    def finalize(self, digest):
        """Use the pre-rendered files only after the real phase domain agrees."""
        source, scientific_asset, scientific = self._source(digest)
        preparation_asset = asset(self.root/"display_preparations"/self.run/digest/"receipt.json")
        preparation = _verify_preparation(self.reader.record(preparation_asset),
            self.preparation_binding(source, scientific_asset, scientific), self.reader)
        phase_asset = asset(self.root/"selected_phase_frames"/self.run/(digest+".json"))
        if self.plan.get("execution_operation") is not None:
            from .phase_ffac_selected_execution_admission import verify_phase_import
            phase = verify_phase_import(phase_asset, source, self.plan, self.reader)
        else:
            phase = self.reader.record(phase_asset)
        expected = dict(job_sha256=self.plan["job_sha256"], pipeline_sha256=self.plan["pipeline_sha256"], source=source)
        if (phase.get("complete") is not True or phase.get("run") != self.run
                or phase.get("source_sha256") != digest or phase.get("elapsed_h") != source["elapsed_h"]
                or phase.get("binding") != expected
                or phase.get("full_extended_scientific") != scientific["full_extended_scientific"]
                or phase.get("baseline_extended_scientific") != scientific["baseline_extended_scientific"]):
            raise ValueError("Final photographic publication requires the actual source-bound phase producer")
        actual_domain = self.reader.array(dict(phase["rig_domain_revision"]["classes"], array_key="rig_domain"))
        prepared_domain = self.reader.array(preparation["domain_mask"])
        if (actual_domain.dtype != bool or actual_domain.shape != prepared_domain.shape
                or not np.array_equal(actual_domain, prepared_domain)):
            raise ValueError("Source-only photographic domain differs from the committed phase domain")
        binding = dict(self.binding(source, phase_asset, scientific_asset),
            display_preparation=preparation_asset, domain_bit_identical=True,
            finalization_code=asset(__file__))
        target = self.root/"display_renderings"/self.run/digest
        receipt_path = target/"receipt.json"
        if receipt_path.exists():
            saved = self.reader.record(asset(receipt_path))
            if saved.get("contract") != DISPLAY_CONTRACT or saved.get("complete") is not True or saved.get("binding") != binding:
                raise ValueError("Final display receipt already has another immutable binding")
            for item in [*saved["photographs"].values(), saved["support_masks"]]: self.reader.path(item)
            return asset(receipt_path)
        if target.exists():
            raise ValueError("Unreceipted final display requires a new output namespace")
        record = {key:deepcopy(preparation[key]) for key in (
            "run", "source_sha256", "elapsed_h", "frame_index", "phase_shape_hw", "native_origin_xy",
            "baseline_source_sha256", "photographs", "support_masks", "support_mask_names",
            "support_counts", "camera_source", "renderer_audit", "display_only",
            "scientific_inputs_changed", "maps_changed", "labels_changed")}
        record.update(contract=DISPLAY_CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
            seconds=preparation["seconds"], reused_pre_rendered_photographs=True,
            phase_domain_check=dict(source_domain=preparation["domain_mask"],
                phase_domain=phase["rig_domain_revision"]["classes"], changed_pixels=0),
            model_inference=False, original_assets_modified=False)
        atomic_json(receipt_path, record)
        return asset(receipt_path)


__all__ = ["SourceReadyDisplayPreparer", "derive_display_domain", "CONTRACT", "DOMAIN_CONTRACT"]
