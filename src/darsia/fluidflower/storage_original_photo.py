"""Publish a stored shape-only photograph ROI as an immutable RGBA display.

Only explicit saved metadata and its already-produced review JPEG are read.
No camera/RAW decode, correction chain, phase/flatfield mask or model is run.
Alpha describes stored photograph ROI coverage, never camera or domain validity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .storage_saved_products import _read_bytes, _json


_ALPHA = "known stored photograph ROI only; not per-pixel camera validity or physical domain"


def _shape(value, label):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) is not int or v < 1 for v in value)
            or value[0] * value[1] > 100_000_000):
        raise ValueError("Explicit bounded integer " + label + " required")
    return tuple(value)


def _place_known_roi(rgb, *, box, target_shape, native_origin):
    """Copy known stored RGB pixels at identical integer coordinates only."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Stored photograph must retain literal uint8 RGB")
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) is not int for v in box)):
        raise ValueError("Explicit integer stored photograph ROI required")
    x, y, width, height = box
    th, tw = _shape(target_shape, "current canvas shape")
    if (x < 0 or y < 0 or width < 1 or height < 1
            or x + width > rgb.shape[1] or y + height > rgb.shape[0]
            or x + width > tw or y + height > th
            or not isinstance(native_origin, list) or len(native_origin) != 2
            or any(type(v) is not int for v in native_origin)
            or native_origin != [-x, -y]):
        raise ValueError("Stored photograph ROI does not have the same current placement")
    rgba = np.zeros((th, tw, 4), np.uint8)
    rgba[y:y + height, x:x + width, :3] = rgb[y:y + height, x:x + width]
    rgba[y:y + height, x:x + width, 3] = 255
    return rgba


def _identity(task, spec, native, certificate, current, certificate_asset, photo_asset, producer_sha):
    if re.fullmatch(r"[0-9a-f]{64}", str(producer_sha)) is None:
        raise ValueError("Explicit original geometry producer SHA required")
    if native != dict(certificate, geometry_certificate_path=certificate_asset["path"],
                      geometry_certificate_sha256=certificate_asset["sha256"]):
        raise ValueError("Native geometry differs from its exact original producer certificate")
    camera = dict(path=native["camera_jpeg_path"], sha256=native["camera_jpeg_sha256"])
    cert = dict(path=certificate_asset["path"], sha256=certificate_asset["sha256"])
    binding = native["binding"]
    if (task["precision"] not in {"float32", "float64"}
            or not isinstance(spec["case_id"], str) or not spec["case_id"]
            or task["run"] != native["run"] or task["run"] != current["run"]
            or current["frame_index"] != spec["frame_index"]
            or current["source_sha256"] != spec["source"]["source_sha256"]
            or camera["sha256"] != spec["source"]["source_sha256"]
            or binding["geometry_module_sha256"] != producer_sha
            or binding["input_sources"]["camera_jpeg"] != camera
            or current["source_photo"] != camera
            or current["source_geometry_receipt"] != cert
            or photo_asset["path"] != native["native_review_jpeg_path"]
            or photo_asset["sha256"] != native["native_review_jpeg_sha256"]
            or current["H_ac28_to_frame_native"] != native["H_ac28_to_frame_native"]
            or current["source_content_box_xywh"] != native["content_box_xywh"]
            or binding["content_box_xywh"] != native["content_box_xywh"]
            or current["source_native_shape_hw"] != native["native_shape_hw"]
            or binding["native_shape_hw"] != native["native_shape_hw"]):
        raise ValueError("Saved original photograph source/producer/current-position identity differs")
    x, y, width, height = native["content_box_xywh"]
    if [height, width] != native["native_shape_hw"]:
        raise ValueError("Stored photograph ROI and native shape differ")
    return camera


def _asset(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def _pixels(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def publish_registered_original_png(destination, *, task, frame_spec, native_geometry,
                                    geometry_certificate, current_geometry, saved_native_review,
                                    geometry_producer_sha256, resolve_asset):
    """Producer-only saved JPEG decode and exact ROI copy; consumers use PNG only.

    The caller supplies four closed immutable asset descriptors and the original
    geometry producer's reviewed SHA. Descriptor references are never followed.
    The task/spec are the same run/case/source/precision context as its products.
    """
    destination = Path(destination)
    receipt = destination.with_name(destination.name + ".receipt.json")
    if destination.suffix.lower() != ".png":
        raise ValueError("Choose a new original photograph PNG")
    if destination.exists() or receipt.exists():
        raise FileExistsError("Original photograph products and receipts are immutable")
    loaded = []
    native = _json(_read_bytes(native_geometry, resolve_asset, loaded))
    certificate = _json(_read_bytes(geometry_certificate, resolve_asset, loaded))
    current = _json(_read_bytes(current_geometry, resolve_asset, loaded))
    camera = _identity(task, frame_spec, native, certificate, current, geometry_certificate,
                       saved_native_review, geometry_producer_sha256)
    jpeg = _read_bytes(saved_native_review, resolve_asset, loaded)
    with Image.open(io.BytesIO(jpeg)) as image:
        if (image.format != "JPEG" or image.mode != "RGB"
                or image.getexif().get(274, 1) != 1):
            raise ValueError("Saved shape-only photograph must retain its RGB JPEG encoding/orientation")
        rgb = np.asarray(image).copy()
    if tuple(rgb.shape[:2][::-1]) != tuple(current["photograph_canvas_wh"]):
        raise ValueError("Saved photograph canvas differs from its current bound metadata")
    rgba = _place_known_roi(rgb, box=native["content_box_xywh"],
        target_shape=current["phase_shape_hw"], native_origin=current["native_origin_xy"])
    x, y, width, height = native["content_box_xywh"]
    source_roi = rgb[y:y + height, x:x + width]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        Image.fromarray(rgba).save(stream, format="PNG")
    document = dict(format="darsia-stored-registered-original-photo-v1", complete=True,
        run=task["run"], case_id=frame_spec["case_id"], precision=task["precision"],
        source=deepcopy(frame_spec["source"]), camera_source=camera,
        geometry_producer_sha256=geometry_producer_sha256, loaded_assets=loaded,
        current_selected_frame_index=current["frame_index"],
        camera_timeline_index=native.get("frame_index"),
        source_review_canvas_hw=list(rgb.shape[:2]), target_canvas_hw=list(rgba.shape[:2]),
        stored_photograph_ROI_xywh=list(native["content_box_xywh"]),
        current_native_origin_xy=list(current["native_origin_xy"]),
        H_ac28_to_frame_native=deepcopy(current["H_ac28_to_frame_native"]),
        alpha_meaning=_ALPHA, alpha_inside_ROI=255, alpha_outside_ROI=0,
        full_canvas_photograph_coverage=(width * height == rgba.shape[0] * rgba.shape[1]),
        known_ROI_pixel_identity=_pixels(source_roi), copied_ROI_pixel_identity=_pixels(rgba[y:y + height, x:x + width, :3]),
        RGBA_pixel_identity=_pixels(rgba), display_dtype="uint8",
        original_shape_configs=deepcopy(native["binding"]["original_shape_configs"]),
        original_shape_corrections=deepcopy(native["binding"]["original_shape_corrections"]),
        JPEG_decode="PIL RGB; no EXIF transpose; producer only", colour_light_or_v5_changes=False,
        camera_RAW_or_correction_chain_executed=False, current_phase_products_changed=False,
        photograph=_asset(destination))
    with receipt.open("xb") as stream:
        stream.write((json.dumps(document, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, indent=2) + "\n").encode("utf8"))
    return dict(photograph=document["photograph"], receipt=_asset(receipt),
                full_canvas_photograph_coverage=document["full_canvas_photograph_coverage"])
