"""Source-bound edge-aware camera-linear FF_AC ARW on native-review pixels.

The output is *camera-native* black-subtracted RGB in RAW channel units. No
white balance, camera-to-XYZ/sRGB matrix, gamma, brightness, or clipping is
applied: Claude v5 was calibrated in precisely this camera-native basis.
An optional display rendering must be downstream and separate from fitting.

The Bayer image is edge-aware demosaiced at full sensor resolution before the
fixed Sony distortion/CA and per-frame DarSIA native-review homography are
applied. This is CPU-only. Existing review boxes remain in native coordinates.
Its optical response differs from the existing Claude v5 1/8 Gaussian-CFA
response; using this renderer for model fitting requires a separate source-only
photometric refit and held-out validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.tools import ffac_raw_geometry as geo


_CHANNEL_ORDER_TEST = np.array([[10000, 4000, 10000, 4000, 10000, 4000],
                                [4000, 1000, 4000, 1000, 4000, 1000]] * 3,
                               dtype=np.uint16)


def _check_opencv_cfa_order() -> None:
    # OpenCV's BayerRG2BGR_EA returns RGB array for this RGGB fixture. This
    # explicit runtime control guards against a silent R/B swap across builds.
    test = cv2.demosaicing(_CHANNEL_ORDER_TEST, cv2.COLOR_BayerRG2BGR_EA)
    if not np.array_equal(test[2, 2], [10000, 4000, 1000]):
        raise ValueError("OpenCV Bayer output order differs from frozen RGGB test")


def registered_edge_aware_raw_linear(
    arw_path: str | Path, camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path, *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    """Return (camera_linear_RGB_3x2979x5472_float32, certificate).

    The certificate checks source SHAs, sensor profile, geometry and clipping.
    It does not alone make a phase label or model scientifically eligible.
    """
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("FF_AC edge-aware ARW rendering requires rawpy (CPU LibRaw)") from error
    _check_opencv_cfa_order()
    arw = Path(arw_path)
    camera = Path(camera_jpeg_path)
    native = Path(review_native_jpeg_path)
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("ARW/JPEG basename mismatch")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", native)):
        if not expected_source_hashes.get(key) or geo._sha256(path) != expected_source_hashes[key].lower():
            raise ValueError(f"Missing or mismatched SHA: {key}")
    x0, y0, width, height = map(int, content_box)
    if (width, height) != geo._CONTENT_WH:
        raise ValueError("Native review content shape mismatch")
    with Image.open(camera) as image:
        if image.size != geo._CAMERA_WH:
            raise ValueError("Unexpected Sony camera JPEG shape")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(native) as image:
        if x0 < 0 or y0 < 0 or x0+width > image.width or y0+height > image.height:
            raise ValueError("Content box outside native canvas")
        review_green = np.asarray(image.convert("RGB"))[y0:y0+height, x0:x0+width, 1].astype(np.float32)
        canvas_wh = image.size
    homography, h_qa = geo._review_to_camera(review_green, camera_green)
    run = next((part.upper() for part in arw.parts
                if part.upper().startswith("AC") and part[2:].isdigit()), None)
    if run is None:
        raise ValueError("Unknown AC run in source path")
    profile = "early" if int(run[2:]) <= 44 else "late"
    distortion, ca_red, ca_blue = geo._sony_tags(arw)
    if distortion != geo._DISTORTION[profile]:
        raise ValueError("Unexpected Sony 0x7037 lens profile")
    with rawpy.imread(str(arw)) as raw:
        if raw.raw_image_visible.shape != (geo._RAW_WH[1], geo._RAW_WH[0]):
            raise ValueError("Unexpected RAW visible shape")
        if not np.array_equal(raw.raw_pattern, [[0, 1], [3, 2]]) or raw.color_desc != b"RGBG":
            raise ValueError("Unexpected CFA pattern")
        black_levels = tuple(int(x) for x in raw.black_level_per_channel)
        if black_levels != (800, 800, 800, 800):
            raise ValueError("Unexpected black levels")
        white_level = int(raw.white_level)
        if white_level <= 800 or white_level > 65535:
            raise ValueError("Unexpected RAW white level")
        original = raw.raw_image_visible.copy()
        camera_wb = [float(x) for x in raw.camera_whitebalance]
        camera_to_xyz = np.asarray(raw.rgb_xyz_matrix, np.float64).tolist()
    sensor_saturation = original >= white_level
    # Subtract black *before* demosaic; no 0..1 scaling or automatic exposure.
    # All sampled ZV-1 inputs are unsigned <=16-bit. uint16 addition/interpolation
    # cannot overflow as sensor white is <16384 after black subtraction.
    black_subtracted = np.maximum(original.astype(np.int32)-800, 0).astype(np.uint16)
    demosaiced = cv2.demosaicing(black_subtracted, cv2.COLOR_BayerRG2BGR_EA)
    if demosaiced.shape != (geo._RAW_WH[1], geo._RAW_WH[0], 3):
        raise ValueError("Full-resolution demosaic shape mismatch")
    demosaiced_float = [demosaiced[..., channel].astype(np.float32) for channel in range(3)]
    # Independent green CFA control checks RAW↔native geometry without using
    # the demosaiced RGB itself to fit the homography.
    gg = .5*(black_subtracted[0::2, 1::2].astype(np.float32)
             + black_subtracted[1::2, 0::2].astype(np.float32))
    direct = geo._raw_review_residual(gg, review_green, homography,
                                       geo._PROFILE[profile], distortion)
    output = np.empty((3, height, width), dtype=np.float32)
    params = geo._PROFILE[profile]
    # Sony CA is a channel-specific change of radial scale, applied in the
    # camera-JPEG-centered inverse map for each full-res demosaiced channel.
    for yy0 in range(0, height, 128):
        yy1 = min(height, yy0+128)
        ys, xs = np.mgrid[yy0:yy1, 0:width].astype(np.float32)
        for channel, ca in enumerate((ca_red, None, ca_blue)):
            rx, ry = geo._raw_xy(xs, ys, homography, params, distortion, ca)
            output[channel, yy0:yy1] = cv2.remap(
                demosaiced_float[channel],
                rx.astype(np.float32), ry.astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
    finite = np.isfinite(output)
    # Interpolation should remain convex, but still report violations rather
    # than silently clipping a scientific signal.
    overshoot = output > (white_level-800)
    negative = output < 0
    certificate = {
        "contract": "ffac-source-bound-full-res-edge-aware-camera-linear-v1",
        "run": run, "profile": profile,
        "sources": {key: expected_source_hashes[key] for key in (
            "arw_sha256", "camera_jpeg_sha256", "review_native_jpeg_sha256")},
        "camera_jpeg_wh": geo._CAMERA_WH, "raw_visible_wh": geo._RAW_WH,
        "review_canvas_wh": canvas_wh, "content_box_xywh": [x0,y0,width,height],
        "review_to_camera_h": homography.tolist(),
        "review_camera_geometry": h_qa,
        "raw_review_direct_residual": direct,
        "sony_distortion_table": list(distortion),
        "sony_ca_red_table": list(ca_red),
        "sony_ca_blue_table": list(ca_blue),
        "demosaic": "OpenCV edge-aware RGGB, full sensor resolution, 16-bit integer before geometric warp",
        "native_warp_interpolation": "per-channel bilinear float32; NaN outside sensor",
        "channel_basis": "camera-native black-subtracted linear RGB, RAW units",
        "white_balance": {"applied": False, "camera_wb_metadata": camera_wb,
                          "reason": "Claude v5 fitted in unbalanced camera channel basis"},
        "camera_to_xyz": {"applied": False, "rgb_xyz_metadata": camera_to_xyz,
                          "reason": "Phase calibration is in camera channel basis"},
        "gamma_tone_auto_bright": "none",
        "black_level_per_channel": list(black_levels), "white_level": white_level,
        "sensor_saturated_fraction": float(sensor_saturation.mean()),
        "sensor_saturated_count": int(sensor_saturation.sum()),
        "native_nonfinite_fraction": float(1-finite.mean()),
        "native_negative_overshoot_fraction": float(negative.mean()),
        "native_above_white_overshoot_fraction": float(overshoot.mean()),
        "native_linear_rgb_shape": list(output.shape),
        "training_eligible": False,
        "eligibility_reason": "All-frame, exact-label and photometric parity gates are separate from this geometry/render certificate.",
    }
    return output, certificate
