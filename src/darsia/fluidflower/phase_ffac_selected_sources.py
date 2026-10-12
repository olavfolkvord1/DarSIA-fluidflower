"""Source-bound frozen RAW preparation for the selected-image phase revision.

Existing acquisitions retain their immutable native centre and disjoint RAW
ring. New acquisitions use the original persisted shape/photometry processors.
Neither source preparation nor geometry transport reads phase categories.
"""
from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path

import cv2
import numpy as np

from .phase_ffac_selected_phase import asset, bound_json
from .phase_ffac_rig_camera_runtime import bound_path
from .tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, read_json, utc_now
from .tools import run_ffac_rig_camera_inference as rig

CONTRACT = "ffac-selected-frozen-scientific-source-v1"


def previous_selected_source(source, selected, baseline):
    """Probe unselected anchors without inserting them into selected history."""
    earlier = [baseline, *selected]
    earlier = [row for row in earlier if float(row["elapsed_h"]) < float(source["elapsed_h"])]
    if not earlier:
        return None
    return max(earlier, key=lambda row: float(row["elapsed_h"]))["source_sha256"]


def selected_source_inventory(manifest, parent_rows):
    """Distinct selected production sources and all original Studio QA anchors."""
    if manifest.get("contract") != "ffac-selected-camera-images-v1":
        raise ValueError("Selected source manifest has another contract")
    run = manifest["run"]
    selected = deepcopy(manifest["frame_sources"])
    baseline = deepcopy(manifest["baseline"])
    if len({row["source_sha256"] for row in selected}) != len(selected):
        raise ValueError("Selected acquisitions must be unique")
    if any(float(a["elapsed_h"]) >= float(b["elapsed_h"]) for a, b in zip(selected, selected[1:])):
        raise ValueError("Selected acquisitions must have increasing camera times")
    by_sha = {row["source_sha256"]: row for row in selected}
    rows = {}
    for index, selected_row in enumerate(selected):
        row = deepcopy(selected_row)
        row.update(run=run, selected=True, selected_index=index, studio_anchor=False,
            previous_source_sha256=previous_selected_source(row, selected, baseline),
            temporal_input_policy="policy-A-selected-camera-chronology")
        rows[row["source_sha256"]] = row
    anchors = []
    for parent_row in parent_rows:
        parent = bound_json(parent_row)
        if parent["run"] != run or parent.get("complete") is not True:
            raise ValueError("Studio anchor belongs to another source run")
        source = parent["binding"]["source"]
        digest = source["source_sha256"]
        if digest in rows:
            row = rows[digest]
            if float(row["elapsed_h"]) != float(source["elapsed_h"]):
                raise ValueError("One acquisition has two camera times")
        else:
            geometry = source["rig_geometry"]
            row = dict(run=run, source_path=source["source_path"], source_sha256=digest,
                v5_raw_path=geometry["raw_source"]["path"],
                v5_raw_sha256=geometry["raw_source"]["sha256"],
                elapsed_h=float(source["elapsed_h"]), selected=False, selected_index=None,
                previous_source_sha256=previous_selected_source(source, selected, baseline),
                temporal_input_policy="policy-A-selected-camera-chronology")
            rows[digest] = row
        row.update(studio_anchor=True, studio_frame_index=int(parent["frame_index"]),
            parent_receipt=deepcopy(parent_row), parent_source=deepcopy(source))
        anchors.append(digest)
    if baseline["source_sha256"] not in rows:
        raise ValueError("Original frame-zero baseline is not a Studio replay anchor")
    baseline_sha = baseline["source_sha256"]
    rows[baseline_sha]["baseline"] = True
    # Keep the reference's original representation, including AC61 composites.
    rows[baseline_sha]["baseline_record"] = baseline
    return dict(run=run, sources=sorted(rows.values(), key=lambda row: float(row["elapsed_h"])),
        selected_sources=list(by_sha), studio_anchor_sources=anchors,
        baseline_source_sha256=baseline_sha,
        history_policy="baseline plus selected sources only; unselected Studio frames are probes")


def _curve_keep(curve, h_native, shape):
    from .phase_ffac_p1_final_bottom_report import transport_curve
    result = transport_curve(np.asarray(curve, np.float64), H_source_to_target=np.asarray(h_native),
        source_origin_xy=(0, 0), target_origin_xy=tuple(rig.metadata.ORIGIN_XY), target_width=shape[1])
    good = np.flatnonzero(np.isfinite(result))
    if not len(good):
        raise ValueError("Frozen photo-surface curve has no target columns")
    result[:good[0]] = result[good[0]]
    result[good[-1]+1:] = result[good[-1]]
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result >= shape[0]):
        raise ValueError("Frozen photo-surface transport escapes the phase grid")
    return np.arange(shape[0])[:, None] >= np.ceil(result).astype(np.int64)[None, :]


def new_rig_geometry(reference_row, source, own_geometry, top_curve, top_curve_asset, output):
    """Transport the run's own frozen metal/fixture geometry within that run.

    No inter-run sand-scale transform replaces the photographed rig reference.
    The same within-run photographic-stability assumption as the parent is
    explicit, while static texture is independently checked on this source.
    """
    from .phase_ffac_rig_geometry import transform_points
    original = reference_row["binding"]["geometry"]
    current_h = np.asarray(own_geometry["H_ac28_to_frame_native"], np.float64)
    reference_h = np.asarray(original["H_ac28_to_frame_native"], np.float64)
    transport = current_h @ np.linalg.inv(reference_h)
    transport /= transport[2, 2]
    row = deepcopy(original)
    row.update(reference=False, auxiliary=False, source_changed=True,
        frame_index=int(source["work_index"]), elapsed_h=float(source["elapsed_h"]),
        source_sha256=source["source_sha256"],
        scientific_task_id=source["run"] + "/source-" + source["source_sha256"],
        source_photo=dict(path=source["source_path"], sha256=source["source_sha256"]),
        raw_source=dict(path=source["v5_raw_path"], sha256=source["v5_raw_sha256"]),
        H_ac28_to_frame_native=current_h.tolist(),
        H_reference_rig_to_frame_native=(transport @ np.asarray(original["H_reference_rig_to_frame_native"])).tolist(),
        frame_inner_polygon_native=transform_points(original["frame_inner_polygon_native"], transport).tolist(),
        frame_fixture_polygons_native={name: transform_points(points, transport).tolist()
            for name, points in original["frame_fixture_polygons_native"].items()},
        source_geometry_receipt=dict(path=own_geometry["geometry_certificate_path"],
            sha256=own_geometry["geometry_certificate_sha256"]),
        photographic_renderer_receipt=None,
        hardware_geometry_method="frozen own-run reference polygons transported by independently checked source registration",
        new_source_geometry_policy=dict(within_run_photographic_stability_assumed=True,
            static_sand_registration=own_geometry["source_sand_registration"],
            own_rig_reference=original["boundary_evidence"],
            new_hardware_pose_independently_measured=False, phase_labels_read=False))
    row["area_parameters"]["H_ac28_to_frame_native"] = current_h.tolist()
    keep = _curve_keep(top_curve, current_h, tuple(row["phase_shape_hw"]))
    payload = io.BytesIO()
    np.savez_compressed(payload, top_keep=keep)
    target = Path(output) / "photo_top" / source["run"] / (source["source_sha256"] + ".npz")
    top = rig._write_bytes(target, payload.getvalue())
    row["frame_photo_top_keep"] = dict(top, array_key="top_keep", shape_hw=list(keep.shape),
        curve_source=top_curve_asset, method="photographed-static-grain-onset-v2-exact-frozen-H-ceil",
        acceptance_margin_px=0, phase_or_labels_used=False)
    return row


class SelectedRunSourcePreparer:
    """Prepare one run sequentially with bounded frozen-reference caches."""
    def __init__(self, root, plan, run):
        from .phase_ffac_rig_raw_dispatch import create_rig_raw_source_processor
        self.root, self.plan, self.run = Path(root), plan, run
        self.state = plan["runs"][run]
        self.inventory = bound_json(self.state["source_inventory"])
        self.native_path = bound_path(self.state["parent_native_manifest"])
        self.processor = create_rig_raw_source_processor(self.native_path)
        self.geometry = None
        self.reference_parent = bound_json(self.state["reference_parent_receipt"])
        self.reference_extended = self.reference_parent["baseline_extended_scientific"]
        self.top_curve = np.load(bound_path(plan["frozen_top_curve"]), allow_pickle=False)

    def prepare(self, source):
        from .phase_ffac_camera_source_geometry import CameraSourceGeometry
        from .tools.run_ffac_rig_camera_candidate import prepare_source, artifact
        destination = self.root / "selected_sources" / self.run / source["source_sha256"] / "receipt.json"
        binding = dict(contract=CONTRACT, job_sha256=self.plan["job_sha256"], source=source,
            frozen_parent=self.plan["parent_chain"]["parent_seal"],
            source_inventory=self.state["source_inventory"])
        if destination.exists():
            saved = read_json(destination)
            if saved.get("binding") == binding and saved.get("complete") is True:
                return saved
            declared = self.plan.get("source_imports", {}).get(self.run, {}).get(source["source_sha256"])
            if declared is not None:
                from .phase_ffac_selected_phase import asset
                imported = bound_json(declared)
                if (Path(declared["path"]).resolve() != destination.resolve()
                        or imported.get("complete") is not True
                        or imported.get("binding", {}).get("source") != source
                        or imported["binding"].get("frozen_parent") != binding["frozen_parent"]):
                    raise ValueError("Selected RAW import changed its exact source or frozen parent")
                equivalence = dict(contract="ffac-selected-source-import-equivalence-v1", complete=True,
                    source_sha256=source["source_sha256"], producer_receipt=declared,
                    source_inventory=self.state["source_inventory"], consumer_job_sha256=self.plan["job_sha256"],
                    native_scientific=imported["native_scientific"],
                    full_extended_scientific=imported["full_extended_scientific"],
                    scientific_arrays_changed=False, geometry_changed=False,
                    inference_performed=False, producer_receipt_preserved=True)
                target = self.root / "source_imports" / self.run / source["source_sha256"] / "receipt.json"
                if target.exists():
                    if read_json(target) != equivalence:
                        raise ValueError("Selected source import equivalence has another binding")
                else:
                    atomic_json(target, equivalence)
                return dict(imported, source_import_equivalence=asset(target))
            else:
                raise ValueError("Selected RAW resume changed its immutable source/code binding")
        if source.get("studio_anchor"):
            parent = bound_json(source["parent_receipt"])
            full = parent["full_extended_scientific"]
            prepared = bound_json(full["binding"]["prepared_source"])
            geometry = parent["binding"]["source"]["rig_geometry"]
            source_audit = dict(reused_exact_scientific_source=True,
                parent_receipt=source["parent_receipt"], native_centre_changed=False,
                rig_geometry_changed=False, inference_performed=False)
        else:
            if self.geometry is None:
                self.geometry = CameraSourceGeometry(self.native_path)
            own = self.geometry.prepare(source, self.root / "source_geometry")
            task = dict(task_id=self.run + "/source-" + source["source_sha256"], run=self.run,
                source=dict(path=source["source_path"], sha256=source["source_sha256"]),
                raw=dict(path=source["v5_raw_path"], sha256=source["v5_raw_sha256"]),
                parent_native_manifest=self.state["parent_native_manifest"], parent_frame_index=None,
                parent_reference=False, own_geometry=own)
            prepared = prepare_source(self.root, task, self.plan["job_sha256"], processor=self.processor)
            geometry = new_rig_geometry(self.reference_extended, source, own, self.top_curve,
                self.plan["frozen_top_curve"], self.root)
            inference = dict(job_sha256=self.plan["job_sha256"], scientific_source_root=str(self.root),
                upstream_plan=dict(binding=dict(candidate=str(self.root))),
                binding=dict(phase_grid=self.plan["phase_grid"]))
            _, _, full = rig.extended_scientific(self.root, inference, prepared, geometry, self.processor)
            source_audit = dict(reused_exact_scientific_source=False, new_raw_source_corrected=True,
                frozen_photometry=True, photometry_fit=False, parent_scientific_source_changed=False,
                geometry_certificate=prepared["geometry_receipt"], inference_performed=False)
        if prepared["task"]["source"]["sha256"] != source["source_sha256"]:
            raise ValueError("Prepared scientific source identifies another acquisition")
        receipt = dict(contract=CONTRACT, complete=True, created_utc=utc_now(), binding=binding,
            run=self.run, source_sha256=source["source_sha256"], elapsed_h=source["elapsed_h"],
            native_scientific=prepared["full_native_scientific"], prepared_source=full["binding"]["prepared_source"],
            full_extended_scientific=full, baseline_extended_scientific=self.reference_extended,
            geometry_frame=prepared["geometry_frame"], rig_geometry=geometry, source_audit=source_audit,
            training=False, labels_read=False, phase_inference=False, bottom_mass_allowed=False)
        atomic_json(destination, receipt)
        return receipt


def load_selected_scientific(receipt):
    """Load exact centre and transient full rig; verify each asset in session."""
    native = receipt["native_scientific"]
    log = np.load(bound_path(native["log_linear_rgb"]), allow_pickle=False, mmap_mode="r")
    valid = np.load(bound_path(native["valid_mask"]), allow_pickle=False, mmap_mode="r")
    full, full_valid = rig.compose_extended_scientific(receipt["full_extended_scientific"])
    if log.dtype != np.float32 or log.shape != (3, *rig.metadata.SOURCE_SHAPE):
        raise ValueError("Selected scientific input lost native float32 RAW")
    return log, valid, full, full_valid


__all__ = ["selected_source_inventory", "previous_selected_source", "new_rig_geometry",
           "SelectedRunSourcePreparer", "load_selected_scientific"]
