"""Pre-render source-bound full-rig photographs for selected-camera review.

The renderer writes photographs and small display receipts only. It neither
evaluates a model nor writes scientific arrays, categorical maps or annotations.
The three photographic transfers and the RAW/Change transfers are the existing
fixed display operators; missing source support is disclosed on the rig grid.
"""
from __future__ import annotations

import io
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .phase_ffac_selected_phase import asset, bound_json
from .phase_ffac_selected_sources import load_selected_scientific
from .phase_ffac_rig_camera_runtime import bound_path
from .phase_rig_raw_display import mark_missing_raw
from .tools.ffac_track1_display_modes import render_display_modes
from .tools.ffac_track1_change_flatfield import (
    prepare_flatfield_reference, register_flatfield_reference,
    render_change_flatfield, _box5_fully_valid, MIN_REFERENCE_LUMINANCE,
)
from .tools.refit_ffac_g3_downstream import atomic_json, utc_now
from .tools import run_ffac_rig_camera_inference as rig

CONTRACT = "ffac-selected-full-rig-photographs-v1"
MODES = ("natural", "lift1", "lift2", "change", "raw_detail", "original")


def render_original(camera_rgb, review_to_camera_h, *, shape_hw, native_origin_xy, rig_domain):
    """Place the real camera JPEG on the full rig grid, including observed edges."""
    camera = np.asarray(camera_rgb)
    shape = tuple(map(int, shape_hw))
    origin = tuple(map(int, native_origin_xy))
    matrix = np.asarray(review_to_camera_h, np.float64)
    if (camera.dtype != np.uint8 or camera.ndim != 3 or camera.shape[2] != 3
            or matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12
            or np.asarray(rig_domain).dtype != bool or rig_domain.shape != shape):
        raise ValueError("Original photograph requires its exact camera/grid geometry")
    native_from_canvas = np.array([[1., 0., origin[0]], [0., 1., origin[1]], [0., 0., 1.]])
    canvas_to_camera = matrix @ native_from_canvas
    photo = cv2.warpPerspective(camera, canvas_to_camera, shape[::-1],
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(19, 22, 27))
    footprint = cv2.warpPerspective(np.ones(camera.shape[:2], np.float32),
        canvas_to_camera, shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.) >= np.float32(1 - 1e-6)
    missing = mark_missing_raw(photo, rig_domain, footprint)
    return photo, footprint, dict(review_to_camera_h=matrix.tolist(),
        canvas_to_camera_h=canvas_to_camera.tolist(), native_origin_xy=list(origin),
        full_rig_shape_hw=list(shape), camera_supported_pixels=int(footprint.sum()),
        unavailable_rig_pixels=int(missing.sum()), source_pixels_modified=False)


def render_arrays(full, current_valid, domain, baseline_reference, current_h,
                  camera_rgb, review_to_camera_h, *, origin_xy, backend="auto"):
    """One pure display pass with no phase labels as a photographic input."""
    full, valid, sand = np.asarray(full), np.asarray(current_valid), np.asarray(domain)
    if (full.dtype != np.float32 or full.shape != (3, *sand.shape)
            or valid.dtype != bool or sand.dtype != bool or valid.shape != sand.shape
            or not np.array_equal(valid, np.isfinite(full).all(0))):
        raise ValueError("Selected display requires exact full RAW/support/domain grids")
    modes, natural_audit = render_display_modes(full, sand)
    photographs = {name: np.rint(rgb * 255).astype(np.uint8) for name, rgb in modes.items()}
    registered = register_flatfield_reference(baseline_reference, np.asarray(current_h),
        target_shape_hw=sand.shape, source_origin_xy=origin_xy, target_origin_xy=origin_xy)
    photographs["change"], change_audit = render_change_flatfield(
        full, registered, sand, lift1_rgb=photographs["lift1"])
    from .phase_rig_display_acceleration import raw_detail, backend_description
    photographs["raw_detail"] = raw_detail(full, valid, backend=backend)
    photographs["original"], camera_support, original_audit = render_original(camera_rgb,
        review_to_camera_h, shape_hw=sand.shape, native_origin_xy=origin_xy, rig_domain=sand)
    change_support = (_box5_fully_valid(valid) & _box5_fully_valid(registered.valid)
        & (registered.luminance_box5 >= MIN_REFERENCE_LUMINANCE))
    mode_supports = {name: valid for name in ("natural", "lift1", "lift2", "raw_detail")}
    mode_supports.update(change=change_support, original=camera_support)
    missing_counts = {}
    for name in MODES:
        missing_counts[name] = int(mark_missing_raw(photographs[name], sand, mode_supports[name]).sum())
    return photographs, dict(current_raw_support=valid,
        registered_baseline_support=registered.valid, change_support=change_support,
        original_camera_support=camera_support), dict(
        photographic_transform=natural_audit, change=change_audit, original=original_audit,
        raw_backend=backend_description() if backend != "cpu" else {"backend": "cpu"},
        missing_rig_pixels_by_mode=missing_counts, missing_display="neutral diagonal hatch",
        no_spatial_extrapolation=True, scientific_inputs_changed=False, maps_changed=False)


class SelectedDisplayRenderer:
    """A bounded per-run renderer retaining only its pre-injection light field."""
    def __init__(self, root, plan, run, *, backend="auto"):
        self.root, self.plan, self.run, self.backend = Path(root), plan, run, backend
        self.source_root = Path(plan["scientific_source_root"])
        self.inventory = bound_json(plan["runs"][run]["source_inventory"])
        self.sources = {row["source_sha256"]: row for row in self.inventory["sources"]}
        self.baseline_sha = self.inventory["baseline_source_sha256"]
        baseline_path = self.source_root/"selected_sources"/run/self.baseline_sha/"receipt.json"
        self.baseline_asset = asset(baseline_path)
        baseline = bound_json(self.baseline_asset)
        _, _, full, valid = load_selected_scientific(baseline)
        self.baseline = prepare_flatfield_reference(full, source_valid_mask=valid)
        self.origin = tuple(baseline["rig_geometry"]["native_origin_xy"])

    def binding(self, source, phase_asset, scientific_asset):
        return dict(pipeline_sha256=self.plan["pipeline_sha256"],phase_receipt=phase_asset,
            scientific_receipt=scientific_asset, baseline_scientific_receipt=self.baseline_asset,
            display_code=asset(__file__), fixed_display_code={
                "natural": asset(__import__(render_display_modes.__module__, fromlist=['']).__file__),
                "change": asset(__import__(render_change_flatfield.__module__, fromlist=['']).__file__),
                "raw": asset(__import__('darsia.fluidflower.phase_rig_display_acceleration',fromlist=['']).__file__),
                "missing": asset(__import__(mark_missing_raw.__module__,fromlist=['']).__file__)})

    def render(self, digest):
        started = time.perf_counter()
        source = self.sources[digest]
        if not (source.get("selected") or source.get("baseline")):
            raise ValueError("Only accepted selected images and the exact baseline get Studio photographs")
        phase_asset = asset(self.root/"selected_phase_frames"/self.run/(digest+".json"))
        phase = bound_json(phase_asset)
        scientific_asset = asset(self.source_root/"selected_sources"/self.run/digest/"receipt.json")
        scientific = bound_json(scientific_asset)
        expected = dict(job_sha256=self.plan["job_sha256"],pipeline_sha256=self.plan["pipeline_sha256"],source=source)
        if (phase.get("complete") is not True or phase["binding"] != expected
                or scientific.get("complete") is not True or scientific["binding"]["source"] != source):
            raise ValueError("Selected display lost its exact source/phase producer binding")
        target = self.root/"display_renderings"/self.run/digest
        receipt_path = target/"receipt.json"
        binding = self.binding(source,phase_asset,scientific_asset)
        if receipt_path.exists():
            saved = bound_json(asset(receipt_path))
            if saved.get("complete") is not True or saved.get("binding") != binding:
                raise ValueError("Selected photographic resume changed its source, pipeline or renderer")
            for item in [*saved["photographs"].values(),saved["support_masks"]]:bound_path(item)
            return asset(receipt_path)
        _, _, full, valid = load_selected_scientific(scientific)
        classes = rig.load_array(dict(phase["rig_domain_revision"]["classes"],array_key="rig_domain"))
        native_receipt = bound_json(scientific["native_scientific"]["receipt"])
        audit = native_receipt["producer_audit"]
        renderer = audit["renderer"]
        camera_descriptor = audit["sources"]["camera_jpeg"]
        if (renderer["sources"]["camera_jpeg_sha256"] != camera_descriptor["sha256"]
                or renderer.get("run","").casefold() != self.run):
            raise ValueError("Camera Original renderer belongs to another source/run")
        with Image.open(bound_path(camera_descriptor)) as image:
            if list(image.size) != renderer["camera_jpeg_wh"]:
                raise ValueError("Camera photograph no longer matches its renderer grid")
            camera = np.asarray(image.convert("RGB")).copy()
        photographs, masks, rendering = render_arrays(full,valid,classes,self.baseline,
            scientific["geometry_frame"]["H_v5_B_to_frame_native"],camera,renderer["review_to_camera_h"],
            origin_xy=tuple(phase["rig_domain_revision"]["native_origin_xy"]),backend=self.backend)
        files = {}
        for name, pixels in photographs.items():
            encoded=io.BytesIO();Image.fromarray(pixels).save(encoded,format="JPEG",quality=100,subsampling=0)
            files[name]=rig._write_bytes(target/(name+".jpg"),encoded.getvalue())
        encoded=io.BytesIO();np.savez_compressed(encoded,**masks)
        support=rig._write_bytes(target/"support_masks.npz",encoded.getvalue())
        receipt=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),binding=binding,
            run=self.run,source_sha256=digest,elapsed_h=source["elapsed_h"],frame_index=source["work_index"],
            phase_shape_hw=list(full.shape[1:]),native_origin_xy=list(self.origin),
            baseline_source_sha256=self.baseline_sha,photographs=files,support_masks=support,
            support_mask_names=sorted(masks),support_counts={name:int(mask.sum()) for name,mask in masks.items()},
            camera_source=camera_descriptor,renderer_audit=rendering,display_only=True,
            scientific_inputs_changed=False,maps_changed=False,labels_changed=False,
            seconds=time.perf_counter()-started)
        atomic_json(receipt_path,receipt)
        return asset(receipt_path)


__all__ = ['CONTRACT','MODES','render_original','render_arrays','SelectedDisplayRenderer']
