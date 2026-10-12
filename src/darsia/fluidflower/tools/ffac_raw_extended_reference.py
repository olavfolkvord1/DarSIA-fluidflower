"""Source-backed RAW border for temporal FF_AC V5 reference images.

The fixed Sony ARW sensor is taller than its in-camera 16:9 JPEG.  A corrected
native baseline can therefore be rendered just outside its review crop when
the sand shifts later.  This module returns the ordinary full-native core and
four disjoint camera-linear ring strips from one CPU demosaic.  It never fills
or reflects unavailable signal; nonfinite sensor pixels stay nonfinite.

Photometric correction is downstream.  The ring cannot be used as a trained
feature until its V4 correction and exact label/source-support QA pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import json

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.tools import ffac_raw_geometry as geo
from darsia.fluidflower.tools.ffac_raw_native_ea import _check_opencv_cfa_order


PAD_NATIVE = 32
CONTRACT = "ffac-source-bound-full-res-ea-camera-linear-extended-ring-v1"


def registered_edge_aware_raw_with_ring(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
    pad_native: int = PAD_NATIVE,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Return ``(core_3x2979x5472, ring, certificate)`` in RAW channel units.

    ``ring`` keys are top/bottom ``3×pad×(5472+2pad)`` and left/right
    ``3×2979×pad``.  Concatenate without overlap to reconstruct the padded
    source, whose native origin is ``(-pad,-pad)``.  The core is unchanged by
    the extension.  No GPU or colour-space conversion is used.
    """
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("FF_AC RAW extension requires rawpy/LibRaw on CPU") from error
    _check_opencv_cfa_order()
    if not isinstance(pad_native, int) or not 1 <= pad_native <= 128:
        raise ValueError("RAW reference pad must be an integer 1..128 native pixels")
    arw = Path(arw_path)
    camera = Path(camera_jpeg_path)
    review = Path(review_native_jpeg_path)
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("ARW/JPEG basename mismatch")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", review)):
        digest = expected_source_hashes.get(key)
        if not isinstance(digest, str) or geo._sha256(path) != digest.lower():
            raise ValueError(f"Missing or mismatched source SHA: {key}")
    x0, y0, width, height = map(int, content_box)
    if (width, height) != geo._CONTENT_WH:
        raise ValueError("Unexpected FF_AC native content size")
    with Image.open(camera) as image:
        if image.size != geo._CAMERA_WH:
            raise ValueError("Unexpected Sony camera JPEG size")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(review) as image:
        if x0 < 0 or y0 < 0 or x0 + width > image.width or y0 + height > image.height:
            raise ValueError("Native content box escapes review JPEG")
        review_green = np.asarray(image.convert("RGB"))[y0:y0+height, x0:x0+width, 1].astype(np.float32)
        canvas_wh = image.size
    h_review_camera, h_qa = geo._review_to_camera(review_green, camera_green)
    run = next((part.upper() for part in arw.parts
                if part.upper().startswith("AC") and part[2:].isdigit()), None)
    if run is None:
        raise ValueError("No AC run in original RAW path")
    profile = "early" if int(run[2:]) <= 44 else "late"
    distortion, ca_red, ca_blue = geo._sony_tags(arw)
    if distortion != geo._DISTORTION[profile]:
        raise ValueError("Unexpected source Sony distortion table")
    with rawpy.imread(str(arw)) as raw:
        if (raw.raw_image_visible.shape != (geo._RAW_WH[1], geo._RAW_WH[0])
                or not np.array_equal(raw.raw_pattern, [[0, 1], [3, 2]])
                or raw.color_desc != b"RGBG"
                or tuple(int(x) for x in raw.black_level_per_channel) != (800, 800, 800, 800)):
            raise ValueError("Unexpected RAW active array, CFA or black level")
        white_level = int(raw.white_level)
        if not 800 < white_level <= 65535:
            raise ValueError("Unexpected RAW white level")
        bayer = np.maximum(raw.raw_image_visible.astype(np.int32) - 800, 0).astype(np.uint16)
    demosaiced = cv2.demosaicing(bayer, cv2.COLOR_BayerRG2BGR_EA)
    if demosaiced.shape != (geo._RAW_WH[1], geo._RAW_WH[0], 3):
        raise ValueError("Edge-aware sensor demosaic has wrong shape")
    pad = pad_native
    full_h, full_w = height + 2 * pad, width + 2 * pad
    output = np.empty((3, full_h, full_w), np.float32)
    channels = [demosaiced[..., channel].astype(np.float32) for channel in range(3)]
    params = geo._PROFILE[profile]
    for y_start in range(0, full_h, 128):
        y_stop = min(y_start + 128, full_h)
        ys, xs = np.mgrid[y_start:y_stop, 0:full_w].astype(np.float32)
        xs -= np.float32(pad)
        ys -= np.float32(pad)
        for channel, ca in enumerate((ca_red, None, ca_blue)):
            sensor_x, sensor_y = geo._raw_xy(xs, ys, h_review_camera,
                                              params, distortion, ca)
            output[channel, y_start:y_stop] = cv2.remap(
                channels[channel], sensor_x.astype(np.float32),
                sensor_y.astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
    core = output[:, pad:pad+height, pad:pad+width].copy()
    ring = {
        "top": output[:, :pad, :].copy(),
        "bottom": output[:, pad+height:, :].copy(),
        "left": output[:, pad:pad+height, :pad].copy(),
        "right": output[:, pad:pad+height, pad+width:].copy(),
    }
    finite_core = np.isfinite(core)
    ring_valid = {key: int(np.isfinite(value).all(axis=0).sum()) for key, value in ring.items()}
    ring_total = {key: int(value.shape[1]*value.shape[2]) for key, value in ring.items()}
    certificate = {
        "contract": CONTRACT,
        "run": run,
        "profile": profile,
        "sources": {key: expected_source_hashes[key] for key in (
            "arw_sha256", "camera_jpeg_sha256", "review_native_jpeg_sha256")},
        "content_box_xywh": [x0, y0, width, height],
        "camera_jpeg_wh": list(geo._CAMERA_WH),
        "raw_visible_wh": list(geo._RAW_WH),
        "review_canvas_wh": list(canvas_wh),
        "pad_native_px": pad,
        "padded_native_origin_xy": [-pad, -pad],
        "padded_native_shape_chw": [3, full_h, full_w],
        "core_shape_chw": list(core.shape),
        "ring_shapes_chw": {key: list(value.shape) for key, value in ring.items()},
        "ring_all_channel_finite_pixel_count": ring_valid,
        "ring_pixel_count": ring_total,
        "core_nonfinite_fraction": float(1 - finite_core.mean()),
        "review_to_camera_h": h_review_camera.tolist(),
        "review_camera_geometry": h_qa,
        "sony_distortion_table": list(distortion),
        "sony_ca_red_table": list(ca_red),
        "sony_ca_blue_table": list(ca_blue),
        "demosaic": "OpenCV edge-aware RGGB, black-800, full visible sensor",
        "channel_basis": "camera-native black-subtracted linear RGB, RAW units",
        "white_balance_color_matrix_gamma_auto_bright": "none",
        "native_warp_interpolation": "bilinear float32; NaN outside RAW sensor",
        "white_level": white_level,
        "photometric_correction_applied": False,
        "training_eligible": False,
    }
    return core, ring, certificate


__all__ = ["PAD_NATIVE", "CONTRACT", "registered_edge_aware_raw_with_ring"]


def registered_edge_aware_raw_area_grid(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    source_certificate_path: str | Path,
    source_certificate_sha256: str,
    h_target_to_review_native: np.ndarray,
    origin_cell_xy: tuple[int, int],
    shape_cells_hw: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Direct RAW-to-target 8×8 native-area grid without an intermediate crop.

    Each target cell is the arithmetic mean of the 64 full-native camera-linear
    EA samples at target-native integer pixel centres, then natural log. The
    target-to-review H may register a run onto AC28's reference coordinates;
    it is composed with the SHA-bound original review-to-camera homography.
    Out-of-sensor values remain NaN and invalidate their whole 8×8 cell.
    This is a CPU-only *source signal*, not a fitted correction or training
    input. The ordinary grid is ``origin_cell_xy=(0,0)``, ``shape=(372,684)``.
    """
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("FF_AC RAW area rendering requires CPU rawpy/LibRaw") from error
    _check_opencv_cfa_order()
    arw, camera, review = map(Path, (arw_path, camera_jpeg_path, review_native_jpeg_path))
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("Original ARW/camera JPEG basename mismatch")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", review)):
        expected = expected_source_hashes.get(key)
        if not isinstance(expected, str) or geo._sha256(path) != expected.lower():
            raise ValueError(f"Missing or changed original source: {key}")
    cert_path = Path(source_certificate_path)
    if geo._sha256(cert_path) != source_certificate_sha256.lower():
        raise ValueError("Original RAW/review geometry certificate changed")
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    if cert["sources"] != dict(expected_source_hashes):
        raise ValueError("Original RAW/review certificate source mismatch")
    if list(cert["content_box_xywh"]) != [416, 496, 5472, 2979]:
        raise ValueError("Original review content box changed")
    h_review_camera = np.asarray(cert["review_to_camera_h"], np.float64)
    h_target_review = np.asarray(h_target_to_review_native, np.float64)
    for h in (h_review_camera, h_target_review):
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-9:
            raise ValueError("Invalid source-bound target/review homography")
    h_target_camera = h_review_camera @ h_target_review
    h_target_camera /= h_target_camera[2, 2]
    run = cert["run"]
    profile = cert["profile"]
    if profile != ("early" if int(run[2:]) <= 44 else "late"):
        raise ValueError("Original lens profile/run mismatch")
    distortion, ca_red, ca_blue = geo._sony_tags(arw)
    if (list(distortion) != cert["sony_distortion_table"] or
            list(ca_red) != cert["sony_ca_red_table"] or
            list(ca_blue) != cert["sony_ca_blue_table"]):
        raise ValueError("Original Sony lens/CA tags changed")
    with rawpy.imread(str(arw)) as raw:
        if (raw.raw_image_visible.shape != (geo._RAW_WH[1], geo._RAW_WH[0]) or
                not np.array_equal(raw.raw_pattern, [[0, 1], [3, 2]]) or
                raw.color_desc != b"RGBG" or
                tuple(int(x) for x in raw.black_level_per_channel) != (800, 800, 800, 800)):
            raise ValueError("Unexpected original RAW sensor/CFA/black level")
        bayer = np.maximum(raw.raw_image_visible.astype(np.int32)-800, 0).astype(np.uint16)
    demosaiced = cv2.demosaicing(bayer, cv2.COLOR_BayerRG2BGR_EA)
    channels = [demosaiced[..., channel].astype(np.float32) for channel in range(3)]
    x_cell, y_cell = map(int, origin_cell_xy)
    rows, cols = map(int, shape_cells_hw)
    if not 1 <= rows <= 512 or not 1 <= cols <= 800:
        raise ValueError("Unexpected target coarse-grid shape")
    out = np.empty((3, rows, cols), np.float32)
    finite_cell_count = 0
    params = geo._PROFILE[profile]
    for cell_y0 in range(0, rows, 16):
        cell_y1 = min(cell_y0 + 16, rows)
        yy, xx = np.mgrid[8*(y_cell+cell_y0):8*(y_cell+cell_y1),
                          8*x_cell:8*(x_cell+cols)].astype(np.float32)
        tile = np.empty((3, 8*(cell_y1-cell_y0), 8*cols), np.float32)
        for channel, ca in enumerate((ca_red, None, ca_blue)):
            sensor_x, sensor_y = geo._raw_xy(xx, yy, h_target_camera,
                                              params, distortion, ca)
            tile[channel] = cv2.remap(
                channels[channel], sensor_x.astype(np.float32),
                sensor_y.astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))
        area = tile.reshape(3, cell_y1-cell_y0, 8, cols, 8).mean((2, 4))
        out[:, cell_y0:cell_y1] = np.log(np.maximum(area, 1e-6)).astype(np.float32)
        finite_cell_count += int(np.isfinite(out[:, cell_y0:cell_y1]).all(axis=0).sum())
    result_cert = dict(
        contract="ffac-source-bound-ea-direct-8x8-area-grid-v1",
        original_source_certificate_path=str(cert_path),
        original_source_certificate_sha256=source_certificate_sha256.lower(),
        original_source_sha256=dict(expected_source_hashes),
        target_to_review_native_h=h_target_review.tolist(),
        target_to_camera_jpeg_h=h_target_camera.tolist(),
        target_cell_origin_xy=[x_cell, y_cell], target_grid_shape_chw=list(out.shape),
        finite_rgb_cell_count=finite_cell_count, total_cells=rows*cols,
        aggregation="8x8 arithmetic area mean of full-native EA camera-linear RGB, then natural log",
        colour_basis="RAW black-800; no WB, matrix, gamma, autoexposure or clipping",
        out_of_sensor="NaN; no nearest/reflect fill",
        model_training_eligible=False,
    )
    return out, result_cert


__all__.append("registered_edge_aware_raw_area_grid")
