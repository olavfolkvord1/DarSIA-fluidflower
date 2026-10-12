"""Source-bound geometry for camera-clock acquisitions outside the old timeline.

Only persisted shape corrections and photographic sand texture are used. This
does not load a rig's labels/facies, train photometry or phase heads, or change
an old native photograph. The native review JPEG is an intermediate geometric
source for the existing frozen RAW renderer, not a new scientific RGB input.
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tomllib

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_camera_timeline import (
    canonical_path, file_sha256, _digest, _new_json, _read,
)

FORMAT = "ffac-camera-clock-source-geometry-v1"
SHAPE = (2979, 5472)
CONTENT_BOX = (416, 496, 5472, 2979)
STATIC_TEXTURE_POLICY = dict(method="source-sand-SIFT-and-independent-window-correlation",
    downsample=2, sift_features=22000, sift_contrast_threshold=.008,
    mutual_ratio=.72, descriptor_coordinate_gate_native_px=160.,
    ransac_native_px=1.5, heldout_cell_native_wh=[384, 320],
    heldout_cell_rule="(row * 7 + column * 11) % 5 == 0",
    minimum_train_inlier_fraction=.85, minimum_mutual_matches=80,
    minimum_independent_matches=36, minimum_independent_matches_per_spatial_cell=4,
    independent_spatial_grid=[2, 3], independent_p95_native_px_max=2.,
    independent_template_half_native_px=24, independent_search_half_native_px=8,
    independent_minimum_correlation=.6, phase_labels_used=False,
    phase_maps_used=False, colour_or_photometry_fit=False)


def normal_h(value):
    h = np.asarray(value, np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()
        or abs(np.linalg.det(h)) < 1e-12 or abs(h[2, 2]) < 1e-12):
        raise ValueError("Invalid source homography")
    return h / h[2, 2]


def compose_source_geometry(h_ac28_to_old_frame, h_b_to_old_frame, h_b_to_new):
    """Reference→B is fixed; B→new is independently measured for the source."""
    h_ref_b = normal_h(np.linalg.inv(normal_h(h_b_to_old_frame)) @ normal_h(h_ac28_to_old_frame))
    h_b_new = normal_h(h_b_to_new)
    return normal_h(h_b_new @ h_ref_b), h_ref_b


def _asset(path, expected=None):
    path = canonical_path(path)
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen source geometry input changed: {path}")
    return dict(path=str(path), sha256=actual)


def _gray(image):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] >= 3:
        # Green texture avoids a colour fit and is used by the exact RAW matcher.
        image = image[..., 1]
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Invalid source texture image")
    return np.asarray(image, np.uint8)


def _texture_image(image):
    from darsia.fluidflower.tools.ffac_raw_geometry import _bandpass
    gray = _gray(image).astype(np.float32)
    return _bandpass(np.log(gray + 1.0))


def heldout_cells(points, cell_wh=(384, 320)):
    points = np.asarray(points, np.float64)
    columns = np.floor(points[:, 0] / cell_wh[0]).astype(np.int64)
    rows = np.floor(points[:, 1] / cell_wh[1]).astype(np.int64)
    return (rows * 7 + columns * 11) % 5 == 0


def independent_texture_check(baseline, frame, mask, h_b_to_frame, *, policy=None):
    """Check correlation windows in whole cells excluded from homography fitting."""
    from darsia.fluidflower.tools.ffac_raw_geometry import _matched_rows
    settings = dict(STATIC_TEXTURE_POLICY, **(policy or {}))
    h = normal_h(h_b_to_frame)
    mask = np.asarray(mask)
    a, b = _texture_image(baseline), _texture_image(frame)
    if mask.dtype != np.bool_ or mask.shape != a.shape or a.shape != b.shape:
        raise ValueError("Source texture mask/grid differs")
    half = int(settings["independent_template_half_native_px"])
    search = int(settings["independent_search_half_native_px"])
    step = max(half * 2 + 8, min(a.shape) // 24)
    points = []
    for y in range(half + search, a.shape[0]-half-search, step):
        for x in range(half + search, a.shape[1]-half-search, step):
            if (heldout_cells([[x, y]], settings["heldout_cell_native_wh"])[0]
                and mask[y-half:y+half+1, x-half:x+half+1].all()):
                points.append((x, y))
    def guess(x, y):
        q = cv2.perspectiveTransform(np.asarray([[[x, y]]], np.float32), h)[0, 0]
        return float(q[0]), float(q[1])
    matches = _matched_rows(a, b, points, template_half=half, search_half=search,
        guess=guess, min_correlation=float(settings["independent_minimum_correlation"]))
    if not len(matches):
        raise ValueError("No independent source-sand texture matches")
    predicted = cv2.perspectiveTransform(matches[:, None, :2].astype(np.float32), h)[:, 0]
    residual = np.linalg.norm(predicted-matches[:, 2:4], axis=1)
    columns = np.clip((matches[:, 0]/(a.shape[1]/3)).astype(int), 0, 2)
    rows = np.clip((matches[:, 1]/(a.shape[0]/2)).astype(int), 0, 1)
    cells = []
    for y in range(2):
        for x in range(3):
            values = residual[(columns == x) & (rows == y)]
            cells.append(dict(row=y, column=x, matches=int(len(values)),
                p95_native_px=float(np.percentile(values, 95)) if len(values) else None))
    minimum = int(settings["minimum_independent_matches_per_spatial_cell"])
    limit = float(settings["independent_p95_native_px_max"])
    passed = len(matches) >= int(settings["minimum_independent_matches"]) and all(
        cell["matches"] >= minimum and cell["p95_native_px"] <= limit for cell in cells)
    if not passed:
        raise ValueError(f"Independent source-sand geometry gate failed: {len(matches)} matches, "
                         f"p95={float(np.percentile(residual,95)):.3f} native pixels")
    return dict(all_pass=True, matches=int(len(matches)),
        p50_native_px=float(np.median(residual)), p95_native_px=float(np.percentile(residual, 95)),
        max_native_px=float(residual.max()), spatial_cells_2x3=cells,
        whole_spatial_cells_excluded_from_fit=True)


def register_static_sand(baseline, frame, baseline_sand_mask, *, policy=None):
    """Measure own source H; no old acquisition H is used as a frame result."""
    settings = dict(STATIC_TEXTURE_POLICY, **(policy or {}))
    gray_a, gray_b = _gray(baseline), _gray(frame)
    mask = np.asarray(baseline_sand_mask)
    if (gray_a.shape != gray_b.shape or mask.dtype != np.bool_
        or mask.shape != gray_a.shape or not mask.any()):
        raise ValueError("Static source sand grid differs")
    scale = int(settings["downsample"])
    h, w = (gray_a.shape[0]//scale)*scale, (gray_a.shape[1]//scale)*scale
    shape = (w//scale, h//scale)
    a = cv2.resize(gray_a[:h,:w], shape, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray_b[:h,:w], shape, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask[:h,:w].astype(np.uint8)*255, shape, interpolation=cv2.INTER_NEAREST)
    cv2.setRNGSeed(20261004)
    sift = cv2.SIFT_create(nfeatures=int(settings["sift_features"]),
                         contrastThreshold=float(settings["sift_contrast_threshold"]))
    ka, da = sift.detectAndCompute(a, small_mask)
    kb, db = sift.detectAndCompute(b, small_mask)
    if da is None or db is None:
        raise ValueError("No static sand descriptors")
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
    ratio = float(settings["mutual_ratio"])
    def paired(first, second):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 and pair[0].distance < ratio*pair[1].distance}
    ab, ba = paired(da, db), paired(db, da)
    pairs = [(i,j) for i,j in ab.items() if ba.get(j)==i]
    pa = np.asarray([ka[i].pt for i,j in pairs], np.float32).reshape(-1,2)*scale + (scale-1)/2
    pb = np.asarray([kb[j].pt for i,j in pairs], np.float32).reshape(-1,2)*scale + (scale-1)/2
    permitted = np.linalg.norm(pa-pb, axis=1) <= float(settings["descriptor_coordinate_gate_native_px"])
    pa, pb = pa[permitted], pb[permitted]
    held = heldout_cells(pa, settings["heldout_cell_native_wh"])
    if len(pa) < int(settings["minimum_mutual_matches"]) or int((~held).sum()) < 40:
        raise ValueError("Insufficient unambiguous static source-sand matches")
    h_b_new, inliers = cv2.findHomography(pa[~held], pb[~held], cv2.RANSAC,
        float(settings["ransac_native_px"]), maxIters=10000, confidence=.999)
    if h_b_new is None or inliers is None or float(inliers.mean()) < float(settings["minimum_train_inlier_fraction"]):
        raise ValueError("Static sand homography fitting gate failed")
    h_b_new = normal_h(h_b_new)
    independent = independent_texture_check(baseline, frame, mask, h_b_new, policy=settings)
    return h_b_new, dict(policy=settings, mutual_matches=int(len(pa)),
        train_matches=int((~held).sum()), heldout_descriptor_matches=int(held.sum()),
        train_inlier_fraction=float(inliers.mean()), independent_texture=independent,
        geometry_pass=True, phase_labels_read=False, model_predictions_read=False)


def _read_native(path, box):
    with Image.open(canonical_path(path)) as image:
        rgb = np.asarray(image.convert("RGB"))
    x,y,w,h = (int(value) for value in box)
    if x < 0 or y < 0 or rgb.shape[0] < y+h or rgb.shape[1] < x+w:
        raise ValueError("Native content box escapes its photographic source")
    return rgb[y:y+h,x:x+w].copy()


def _jpeg_bytes(native_rgb, canvas_wh, box):
    x,y,w,h = (int(value) for value in box)
    if native_rgb.shape != (h,w,3):
        raise ValueError("Geometry photograph has different native dimensions")
    canvas = np.zeros((int(canvas_wh[1]),int(canvas_wh[0]),3),np.uint8)
    canvas[y:y+h,x:x+w] = native_rgb
    stream = io.BytesIO()
    Image.fromarray(canvas).save(stream, format="JPEG", quality=95, subsampling=0)
    return stream.getvalue()


class CameraSourceGeometry:
    """Cache the own frozen B and read-only persisted shape correction chain."""

    def __init__(self, parent_native_manifest):
        import darsia
        from darsia.presets.workflows.rig import Rig
        from darsia.fluidflower.tools.ffac_v5_frame_context import (
            _source_photo_grid, project_source_photo_safe_core,
        )
        self.native_path = canonical_path(parent_native_manifest)
        self.native = _read(self.native_path)
        if tuple(self.native["analysis_image_shape_hw"]) != SHAPE or tuple(self.native["content_box_xywh"]) != CONTENT_BOX:
            raise ValueError("Camera geometry requires the unchanged FF_AC native grid")
        self.native_asset = _asset(self.native_path)
        marker = self.native["v5_track1_photo_geometry"]
        self.original_asset = _asset(marker["original_source_manifest_path"],
                                     marker["original_source_manifest_sha256"])
        original = _read(self.original_asset["path"])
        if original["run"] != self.native["run"]:
            raise ValueError("Original shape source belongs to another run")
        old_transport = marker["frames"][0]
        self.transport_asset = _asset(old_transport["receipt_path"], old_transport["receipt_sha256"])
        self.transport = _read(self.transport_asset["path"])
        if (self.transport["original_source_manifest_sha256"] != self.original_asset["sha256"]
            or self.transport["run"] != self.native["run"]):
            raise ValueError("Frozen B transport belongs to another native source")
        self.frozen_b = {key:deepcopy(value) for key,value in self.transport.items() if key.startswith("v5_B_")}
        for prefix in ("v5_B_arw", "v5_B_camera_jpeg", "v5_B_native_jpeg", "v5_B_registration_certificate"):
            _asset(self.frozen_b[prefix+"_path"], self.frozen_b[prefix+"_sha256"])
        _, self.h_ref_b = compose_source_geometry(self.transport["H_ac28_to_frame_native"],
            self.transport["H_v5_B_to_frame_native"], np.eye(3))
        self.baseline = _read_native(self.frozen_b["v5_B_native_jpeg_path"], CONTENT_BOX)
        sand = _source_photo_grid(self.transport["photo_sand_reference_mask_path"],
                                 self.transport["photo_sand_reference_mask_sha256"])
        self.photo_reference_receipt = _asset(self.transport["photo_sand_reference_receipt_path"],
            self.transport["photo_sand_reference_receipt_sha256"])
        self.sand = project_source_photo_safe_core(sand, self.h_ref_b, shape_hw=SHAPE)
        configs = original["correction_contract"]["config_paths"]
        self.config_assets = [_asset(path) for path in configs]
        rig_path = None
        for path in configs:
            config = tomllib.loads(canonical_path(path).read_text(encoding="utf-8-sig"))
            if "path" in config.get("rig",{}):
                rig_path = canonical_path(config["rig"]["path"])
        if rig_path is None:
            raise ValueError("Original source has no serialized shape rig")
        if original["correction_contract"].get("rig_registration"):
            raise ValueError("Source has an extra registered shape adapter; explicit support is required")
        paths = sorted(rig_path.glob("shape_correction_*.npz"))
        if not paths:
            raise ValueError("No persisted original shape corrections")
        self.shape_assets = [_asset(path) for path in paths]
        self.shape_corrections = [darsia.read_correction(path) for path in paths]
        if any(not Rig._is_shape_correction(correction) for correction in self.shape_corrections):
            raise ValueError("A non-shape correction entered source geometry")
        self.shape_rig_path = str(rig_path)
        generated_b = self.rectify_camera(self.frozen_b["v5_B_camera_jpeg_path"])
        parity_h, parity = register_static_sand(self.baseline, generated_b, self.sand)
        samples = np.asarray([[[0,0],[5471,0],[2736,1490],[0,2978],[5471,2978]]],np.float32)
        displacements = np.linalg.norm(cv2.perspectiveTransform(samples,parity_h)-samples,axis=2)
        if float(displacements.max()) > 1.:
            raise ValueError("Persisted shape chain differs from the frozen B native geometry")
        self.baseline_geometry_parity = dict(parity, H_frozen_B_to_shape_B=parity_h.tolist(),
            maximum_fivepoint_displacement_native_px=float(displacements.max()),
            required_maximum_native_px=1., colour_pixels_compared=False)

    def rectify_camera(self, path):
        import darsia
        image = darsia.imread(canonical_path(path), transformations=self.shape_corrections,
                              dtype=np.float32)
        rgb = np.asarray(image.img)[...,:3]
        if rgb.shape != (*SHAPE,3) or not np.isfinite(rgb).all():
            raise ValueError("Original shape pipeline changed native camera dimensions/support")
        return np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)

    def prepare(self, source_row, output):
        from darsia.fluidflower.tools import ffac_raw_geometry as renderer
        camera_path = canonical_path(source_row["source_path"])
        raw_path = canonical_path(source_row["v5_raw_path"])
        inputs = dict(camera_jpeg=_asset(camera_path,source_row["source_sha256"]),
                      camera_arw=_asset(raw_path,source_row["v5_raw_sha256"]))
        binding = dict(contract=FORMAT,run=self.native["run"],input_sources=inputs,
            requested_source=dict(frame_index=source_row.get("frame_index"),
                elapsed_h=float(source_row["elapsed_h"]),camera_datetime=source_row["camera_datetime"]),
            parent_native_manifest=self.native_asset, original_native_manifest=self.original_asset,
            frozen_B_transport=self.transport_asset, frozen_B=self.frozen_b,
            original_shape_configs=self.config_assets, original_shape_corrections=self.shape_assets,
            shape_rig_path=self.shape_rig_path, policy=STATIC_TEXTURE_POLICY,
            photo_reference_receipt=self.photo_reference_receipt,
            geometry_module_sha256=file_sha256(__file__),
            raw_review_matcher_module_sha256=file_sha256(renderer.__file__),
            native_shape_hw=list(SHAPE),content_box_xywh=list(CONTENT_BOX))
        root = Path(output)/self.native["run"]/inputs["camera_jpeg"]["sha256"]
        receipt_path = root/"geometry.json"
        if receipt_path.exists():
            receipt = _read(receipt_path)
            if receipt["binding"] != binding:
                raise ValueError("Completed camera geometry has different immutable inputs")
            _asset(receipt["native_review_jpeg_path"],receipt["native_review_jpeg_sha256"])
            return dict(receipt,geometry_certificate_path=str(receipt_path),
                        geometry_certificate_sha256=file_sha256(receipt_path))
        if root.exists():
            raise ValueError("Unreceipted camera geometry requires a new output namespace")
        root.mkdir(parents=True,exist_ok=False)
        rgb = self.rectify_camera(camera_path)
        native_path = root/"native_review.jpg"
        native_path.write_bytes(_jpeg_bytes(rgb,self.native["canvas_wh"],CONTENT_BOX))
        # Measure on the actual persisted JPEG passed to the existing RAW math.
        persisted = _read_native(native_path,CONTENT_BOX)
        h_b_new, sand_qa = register_static_sand(self.baseline,persisted,self.sand)
        with Image.open(camera_path) as camera:
            camera_green = np.asarray(camera.convert("RGB"),dtype=np.float32)[...,1]
        h_review_camera, renderer_qa = renderer._review_to_camera(persisted[...,1].astype(np.float32),camera_green)
        if any(value["p95_px"] > 1.5 for value in renderer_qa["heldout"].values()):
            raise ValueError("New source review-to-camera heldout residual exceeds1.5px")
        h_ref_new = normal_h(h_b_new @ self.h_ref_b)
        receipt = dict(binding=binding,contract=FORMAT,run=self.native["run"],
            frame_index=source_row.get("frame_index"),elapsed_h=float(source_row["elapsed_h"]),
            camera_datetime=source_row["camera_datetime"],
            camera_jpeg_path=str(camera_path),camera_jpeg_sha256=inputs["camera_jpeg"]["sha256"],
            camera_arw_path=str(raw_path),camera_arw_sha256=inputs["camera_arw"]["sha256"],
            native_review_jpeg_path=str(native_path),native_review_jpeg_sha256=file_sha256(native_path),
            native_shape_hw=list(SHAPE),content_box_xywh=list(CONTENT_BOX),
            H_ac28_to_frame_native=h_ref_new.tolist(),H_v5_B_to_frame_native=h_b_new.tolist(),
            H_ac28_to_B_native=self.h_ref_b.tolist(),review_to_camera_h=normal_h(h_review_camera).tolist(),
            source_sand_registration=sand_qa,review_camera_geometry=renderer_qa,
            baseline_geometry_parity=self.baseline_geometry_parity,
            original_source_manifest_path=self.original_asset["path"],
            original_source_manifest_sha256=self.original_asset["sha256"],
            photo_sand_reference_mask_path=self.transport["photo_sand_reference_mask_path"],
            photo_sand_reference_mask_sha256=self.transport["photo_sand_reference_mask_sha256"],
            photo_sand_reference_receipt_path=self.transport["photo_sand_reference_receipt_path"],
            photo_sand_reference_receipt_sha256=self.transport["photo_sand_reference_receipt_sha256"],
            new_geometry_measured_from_own_source=True,phase_labels_read=False,
            model_inference=False,photometry_fit=False,raw_signal_materialized=False,
            **self.frozen_b)
        for name,asset in inputs.items():
            _asset(asset["path"],asset["sha256"])
        for asset in [*self.shape_assets,*self.config_assets,self.native_asset,
                      self.original_asset,self.transport_asset,self.photo_reference_receipt]:
            _asset(asset["path"],asset["sha256"])
        _new_json(receipt_path,receipt)
        return dict(receipt,geometry_certificate_path=str(receipt_path),
                    geometry_certificate_sha256=file_sha256(receipt_path))


def prepare_camera_source_geometry(parent_native_manifest, source_row, output, *, context=None):
    context = context or CameraSourceGeometry(parent_native_manifest)
    if context.native_asset["path"] != str(canonical_path(parent_native_manifest)):
        raise ValueError("Source geometry context belongs to another native manifest")
    return context.prepare(source_row,output)


_WORKER_CONTEXTS = {}
_WORKER_THREAD_LIMIT = None


def _worker_initializer(threads):
    global _WORKER_THREAD_LIMIT
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(int(threads))
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=int(threads))
    # This affects only the new source-geometry worker, never an existing GUI.
    import os
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        if not kernel.SetPriorityClass(ctypes.c_void_p(kernel.GetCurrentProcess()),0x4000):
            raise OSError("Could not set source-geometry worker BelowNormal priority")


def _prepare_chunk(run_asset, indices, output):
    _asset(run_asset["path"],run_asset["sha256"])
    run = _read(run_asset["path"])
    native_path = run["native_manifest_path"]
    _asset(native_path,run["native_manifest_sha256"])
    key = (native_path,run["native_manifest_sha256"])
    if key not in _WORKER_CONTEXTS:
        # Two contexts cap memory while keeping repeated AC31 chunks efficient.
        if len(_WORKER_CONTEXTS) >= 2:
            _WORKER_CONTEXTS.pop(next(iter(_WORKER_CONTEXTS)))
        _WORKER_CONTEXTS[key] = CameraSourceGeometry(native_path)
    context = _WORKER_CONTEXTS[key]
    return [context.prepare(run["frames"][index],output) for index in indices]


def _verify_completed_geometry(source):
    path = source["geometry_certificate_path"]
    _asset(path,source["geometry_certificate_sha256"])
    receipt = _read(path)
    if dict(receipt,geometry_certificate_path=path,
            geometry_certificate_sha256=source["geometry_certificate_sha256"]) != source:
        raise ValueError("Aggregate source geometry differs from its own receipt")
    for prefix in ("camera_jpeg","camera_arw","native_review_jpeg"):
        _asset(receipt[prefix+"_path"],receipt[prefix+"_sha256"])
    return receipt


def materialize_changed_camera_sources(camera_plan, output, *, workers=6, threads=1, resume=False):
    """Prepare only reselected sources; keep every old source artifact immutable."""
    from darsia.fluidflower.tools import ffac_raw_geometry
    if not 1 <= int(workers) <= 12 or not 1 <= int(threads) <= 2:
        raise ValueError("Source geometry requires 1–12 CPU workers and 1–2 native threads")
    plan_asset = _asset(camera_plan)
    plan = _read(plan_asset["path"])
    if plan["run_count"] != 43 or plan["frame_count"] != 864:
        raise ValueError("Camera geometry requires the complete 43/864 source plan")
    binding = dict(contract=FORMAT,camera_timeline_plan=plan_asset,
        geometry_module_sha256=file_sha256(__file__),
        raw_review_matcher_module_sha256=file_sha256(ffac_raw_geometry.__file__),
        static_texture_policy=STATIC_TEXTURE_POLICY,workers=int(workers),threads=int(threads),
        device="cpu",worker_priority="BelowNormal",phase_labels_read=False,model_inference=False)
    root = Path(output)
    binding_path = root/"geometry_batch_binding.json"
    manifest_path = root/"camera_source_geometry_manifest.json"
    if binding_path.exists():
        if not resume or _read(binding_path) != binding:
            raise ValueError("Existing source geometry requires exact --resume bindings")
    else:
        if root.exists():
            raise ValueError("Source geometry batch requires a new output namespace")
        root.mkdir(parents=True,exist_ok=False)
        _new_json(binding_path,binding)
    if manifest_path.exists():
        result = _read(manifest_path)
        if result["binding"] != binding:
            raise ValueError("Completed geometry batch has different immutable bindings")
        for source in result["sources"]:
            _verify_completed_geometry(source)
        return dict(result,manifest_path=str(manifest_path),manifest_sha256=file_sha256(manifest_path))
    grouped = {}
    sources = set()
    for run_entry in plan["runs"]:
        run_asset = dict(path=run_entry["receipt_path"],sha256=run_entry["receipt_sha256"])
        _asset(run_asset["path"],run_asset["sha256"])
        run = _read(run_asset["path"])
        indices = []
        for row in run["selection_diff"]:
            if not row["source_changed"]:
                continue
            index = int(row["frame_index"])
            source = run["frames"][index]
            key = (run["run"],source["source_sha256"])
            if key in sources:
                raise ValueError("Changed camera sources are not unique")
            sources.add(key)
            indices.append(index)
        if indices:
            grouped[run["run"]] = (run_asset,indices)
    if len(sources) != plan["changed_source_count"]:
        raise ValueError("Changed acquisition count differs from the camera timeline plan")
    tasks = [(asset,indices[start:start+3]) for asset,indices in grouped.values()
             for start in range(0,len(indices),3)]
    prepared = []
    def log(message):
        print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}",flush=True)
    log(f"Source geometry start: {len(sources)} sources, {len(grouped)} runs, CPU {workers}×{threads}")
    with ProcessPoolExecutor(max_workers=int(workers),initializer=_worker_initializer,
                             initargs=(int(threads),)) as pool:
        futures = [pool.submit(_prepare_chunk,asset,indices,str(root)) for asset,indices in tasks]
        for future in as_completed(futures):
            chunk = future.result()
            prepared.extend(chunk)
            log(f"Source geometry complete: {len(prepared)}/{len(sources)} ({chunk[0]['run']})")
    prepared.sort(key=lambda row:(row["run"],row["frame_index"]))
    if {(row["run"],row["camera_jpeg_sha256"]) for row in prepared} != sources:
        raise ValueError("Prepared geometry source coverage differs from the selected acquisitions")
    for source in prepared:
        _verify_completed_geometry(source)
    result = dict(contract=FORMAT,binding=binding,
        completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        changed_source_count=len(prepared),changed_run_count=len(grouped),sources=prepared,
        all_source_geometry_checks_pass=True,phase_labels_read=False,model_inference=False,
        old_native_sources_modified=False,raw_signal_materialized=False)
    _new_json(manifest_path,result)
    log(f"Source geometry batch ready: {len(prepared)} exact source receipts")
    return dict(result,manifest_path=str(manifest_path),manifest_sha256=file_sha256(manifest_path))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-plan",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    materialize_changed_camera_sources(args.camera_plan,args.output,
        workers=args.workers,threads=args.threads,resume=args.resume)


if __name__ == "__main__":
    main()
