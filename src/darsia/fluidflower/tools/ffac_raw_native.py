"""Full-native Sony ARW input in the camera-linear basis of FF_AC v5.

The original ARW, paired camera JPEG and exact DarSIA native review image are
SHA-bound. The 16-bit Bayer sensor samples are black-subtracted, blurred with
the existing v5 half-CFA sigma=1.8 kernel, then evaluated at *every* native
review pixel via the Sony lens/CA map and a per-frame review→camera homography.

No white balance, camera-to-sRGB conversion, gamma, auto-brightening, clipping
or phase-label-dependent operation is applied. The v5 ColorChecker/AC28 fit
was calibrated in this original camera RGB channel basis. The returned array
has native content shape (3, 2979, 5472), float32 raw units. It is CPU-only.

This renderer intentionally has the same optical low-pass as the v5 1/8 RAW
calibration. It is not claimed to preserve sub-8px phase detail; native phase
QA must assess that separately before scientific use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from darsia.fluidflower.tools import ffac_raw_geometry as geo


def registered_claude_raw_linear(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    """Return source-bound camera-linear RGB and provenance/geometry QA.

    `expected_source_hashes` must include the keys `arw_sha256`,
    `camera_jpeg_sha256`, and `review_native_jpeg_sha256`.  The last hash must
    come from the immutable native review manifest for the *same frame*.
    Training and phase-map eligibility are decided by separate source, label,
    photometry and native-map gates; this function never marks a model ready.
    """
    try:
        import rawpy
    except ImportError as error:
        raise ImportError("FF_AC native ARW rendering requires rawpy (CPU LibRaw)") from error

    arw = Path(arw_path)
    camera = Path(camera_jpeg_path)
    native = Path(review_native_jpeg_path)
    if arw.stem.casefold() != camera.stem.casefold():
        raise ValueError("ARW/camera JPEG stem mismatch")
    if len(content_box) != 4:
        raise ValueError("content_box must be (x,y,width,height)")
    bx, by, width, height = map(int, content_box)
    if (width, height) != geo._CONTENT_WH:
        raise ValueError("Native FF_AC content geometry differs from frozen v5 input")
    for key, path in (("arw_sha256", arw), ("camera_jpeg_sha256", camera),
                      ("review_native_jpeg_sha256", native)):
        expected = expected_source_hashes.get(key)
        if not expected or geo._sha256(path) != expected.lower():
            raise ValueError(f"Missing/mismatched exact source SHA: {key}")
    with Image.open(camera) as image:
        if image.size != geo._CAMERA_WH:
            raise ValueError("Unexpected Sony camera JPEG shape")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(native) as image:
        if bx < 0 or by < 0 or bx+width > image.width or by+height > image.height:
            raise ValueError("Native content box outside review canvas")
        review_green = np.asarray(image.convert("RGB"))[by:by+height,bx:bx+width,1].astype(np.float32)
        canvas_wh = image.size
    H, h_qa = geo._review_to_camera(review_green, camera_green)
    run = next((part.upper() for part in arw.parts
                if part.upper().startswith("AC") and part[2:].isdigit()), None)
    if run is None:
        raise ValueError("Cannot identify FF_AC run in original ARW path")
    profile = "early" if int(run[2:]) <= 44 else "late"
    distortion, ca_red, ca_blue = geo._sony_tags(arw)
    if distortion != geo._DISTORTION[profile]:
        raise ValueError("Sony 0x7037 lens profile changed")
    with rawpy.imread(str(arw)) as raw:
        if raw.raw_image_visible.shape != (geo._RAW_WH[1],geo._RAW_WH[0]):
            raise ValueError("RAW visible dimensions differ")
        if not np.array_equal(raw.raw_pattern,[[0,1],[3,2]]) or raw.color_desc != b"RGBG":
            raise ValueError("RAW Bayer pattern differs")
        black = tuple(int(x) for x in raw.black_level_per_channel)
        if black != (800,800,800,800):
            raise ValueError("RAW black levels differ")
        white = int(raw.white_level)
        if white <= 800 or white > 65535:
            raise ValueError("RAW white level differs")
        mosaic = raw.raw_image_visible.astype(np.float32).copy()
        wb_metadata = [float(x) for x in raw.camera_whitebalance]
        camera_xyz_metadata = np.asarray(raw.rgb_xyz_matrix,np.float64).tolist()
    saturated = mosaic >= white
    original_green = .5*(mosaic[0::2,1::2]+mosaic[1::2,0::2])-800
    direct = geo._raw_review_residual(original_green,review_green,H,
                                       geo._PROFILE[profile],distortion)
    planes = (mosaic[0::2,0::2]-800,
              original_green,
              mosaic[1::2,1::2]-800)
    planes = tuple(cv2.GaussianBlur(plane,(0,0),1.8) for plane in planes)
    output = np.empty((3,height,width),dtype=np.float32)
    params = geo._PROFILE[profile]
    for y0 in range(0,height,128):
        y1=min(height,y0+128)
        ys,xs=np.mgrid[y0:y1,0:width].astype(np.float32)
        for channel,(plane,ca,phase) in enumerate(zip(
            planes,(ca_red,None,ca_blue),((0.,0.),(.5,.5),(1.,1.)),strict=True)):
            rx,ry=geo._raw_xy(xs,ys,H,params,distortion,ca)
            mx=((rx-phase[0])/2).astype(np.float32)
            my=((ry-phase[1])/2).astype(np.float32)
            output[channel,y0:y1]=cv2.remap(
                plane,mx,my,cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
    certificate={
        "contract":"ffac-source-bound-claude-v5-full-native-camera-linear-v1",
        "run":run,"profile":profile,
        "sources":{key:expected_source_hashes[key] for key in (
            "arw_sha256","camera_jpeg_sha256","review_native_jpeg_sha256")},
        "camera_jpeg_wh":geo._CAMERA_WH,"raw_visible_wh":geo._RAW_WH,
        "review_canvas_wh":canvas_wh,"content_box_xywh":[bx,by,width,height],
        "review_to_camera_h":H.tolist(),"review_camera_geometry":h_qa,
        "raw_review_direct_residual":direct,
        "sony_distortion_table":list(distortion),
        "sony_ca_red_table":list(ca_red),"sony_ca_blue_table":list(ca_blue),
        "channel_basis":"black-subtracted camera-native RGB, original RAW units",
        "half_cfa_gaussian_sigma_px":1.8,
        "white_balance":{"applied":False,"camera_metadata":wb_metadata},
        "camera_to_xyz":{"applied":False,"matrix_metadata":camera_xyz_metadata},
        "gamma_tone_auto_bright":"none",
        "black_level_per_channel":list(black),"sensor_white_level":white,
        "sensor_saturated_fraction":float(saturated.mean()),
        "native_nonfinite_fraction":float(1-np.isfinite(output).mean()),
        "native_at_or_below_v5_raw_floor_fraction":float((output<=2).mean()),
        "native_above_v5_raw_ceiling_fraction":float((output>=15100).mean()),
        "output_shape":list(output.shape),
        "training_eligible":False,
        "eligibility_reason":"Photometric, label and native phase-map QA remain separate gates.",
    }
    return output,certificate
