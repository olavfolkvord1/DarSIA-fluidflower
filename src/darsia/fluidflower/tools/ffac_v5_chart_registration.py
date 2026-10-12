"""Source-exact per-frame ColorChecker coordinates for FF_AC v5 light fitting.

The checker can move during AC22/AC38. A fixed baseline swatch lattice then
samples the wrong pixels even when the sand and native image remain registered.
This registration uses only original camera JPEG checker texture, exact file
SHAs and the independently fitted camera↔native homographies. It does not use
phase labels, CO2 areas, or model predictions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    from darsia.fluidflower.project import sha256_file
    return sha256_file(Path(path))


def _gray_roi(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    with Image.open(path) as image:
        if image.size != (5472, 3080):
            raise ValueError("Original camera JPEG is not ZV-1 5472×3080")
        return np.asarray(image.convert("L"))[top:bottom, left:right].copy()


def register_frame_chart_xy8(
    *,
    baseline_camera_jpeg: str | Path,
    baseline_camera_sha256: str,
    frame_camera_jpeg: str | Path,
    frame_camera_sha256: str,
    baseline_chart_xy8: np.ndarray,
    baseline_chart_log_rgb: np.ndarray,
    baseline_review_to_camera_h: np.ndarray,
    frame_review_to_camera_h: np.ndarray,
    frame_native_linear_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (24×2 frame lattice, 24 usable mask, source-only certificate).

    A 3×3 lattice patch around each swatch centre must be finite, inside the
    native image and below 15100 black-subtracted RAW units. The baseline
    swatch itself must have finite calibrated RGB. At least six common patches
    are required; the downstream light fitter may enforce stronger controls.
    """
    baseline = Path(baseline_camera_jpeg)
    frame = Path(frame_camera_jpeg)
    if _sha256(baseline) != baseline_camera_sha256 or _sha256(frame) != frame_camera_sha256:
        raise ValueError("Camera JPEG changed since frozen source binding")
    xy = np.asarray(baseline_chart_xy8, np.float32)
    cb = np.asarray(baseline_chart_log_rgb, np.float32)
    bh = np.asarray(baseline_review_to_camera_h, np.float64)
    fh = np.asarray(frame_review_to_camera_h, np.float64)
    native = np.asarray(frame_native_linear_rgb)
    if xy.shape != (24, 2) or cb.shape != (3, 24):
        raise ValueError("The baseline chart must have 24 source-bound swatches")
    if bh.shape != (3, 3) or fh.shape != (3, 3):
        raise ValueError("Chart registration requires two camera homographies")
    if native.shape != (3, 2979, 5472) or native.dtype != np.dtype("float32"):
        raise ValueError("Chart saturation control requires full-native EA camera-linear RGB")
    baseline_native = xy * 8 + 3.5
    baseline_camera = cv2.perspectiveTransform(baseline_native[None], bh)[0]
    left = max(0, int(np.floor(np.min(baseline_camera[:, 0]) - 75)))
    top = max(0, int(np.floor(np.min(baseline_camera[:, 1]) - 75)))
    right = min(5472, int(np.ceil(np.max(baseline_camera[:, 0]) + 75)))
    bottom = min(3080, int(np.ceil(np.max(baseline_camera[:, 1]) + 75)))
    template_box = (left, top, right, bottom)
    if right-left < 100 or bottom-top < 100:
        raise ValueError("Baseline checker bounds are incomplete")
    search_box = (max(0, left-300), max(0, top-300),
                  min(5472, right+300), min(3080, bottom+300))
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=3000)
    template_kp, template_desc = sift.detectAndCompute(_gray_roi(baseline, template_box), None)
    frame_kp, frame_desc = sift.detectAndCompute(_gray_roi(frame, search_box), None)
    if template_desc is None or frame_desc is None:
        raise ValueError("Checker registration has no source-only descriptors")
    matches = cv2.BFMatcher().knnMatch(template_desc, frame_desc, k=2)
    good = [m for m, n in matches if m.distance < .75 * n.distance]
    if len(good) < 50:
        raise ValueError(f"Checker registration has only {len(good)} ratio matches")
    src = np.float32([template_kp[m.queryIdx].pt for m in good])
    dst = np.float32([frame_kp[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 50:
        raise ValueError("Checker registration lacks 50 geometric inliers")
    keep = inliers.ravel().astype(bool)
    residual = np.linalg.norm(cv2.transform(src[None], affine)[0]-dst, axis=1)[keep]
    p95 = float(np.percentile(residual, 95))
    if p95 > 2.0:
        raise ValueError(f"Checker registration p95 residual exceeds 2 camera px: {p95}")
    origin_b = np.asarray(template_box[:2], np.float32)
    origin_f = np.asarray(search_box[:2], np.float32)
    frame_camera = cv2.transform((baseline_camera-origin_b)[None], affine)[0] + origin_f
    frame_native = cv2.perspectiveTransform(frame_camera[None], np.linalg.inv(fh))[0]
    frame_xy = ((frame_native-3.5)/8).astype(np.float32)
    usable = np.all(np.isfinite(cb), axis=0) & np.all(np.isfinite(frame_xy), axis=1)
    for index, (x, y) in enumerate(frame_xy):
        ix, iy = int(round(float(x*8+3.5))), int(round(float(y*8+3.5)))
        # The light fitter samples 3×3 grid centres at offsets ±8 native px.
        # Include the 8×8 support around each centre for the RAW saturation gate.
        if ix-12 < 0 or iy-12 < 0 or ix+13 > 5472 or iy+13 > 2979:
            usable[index] = False
            continue
        patch = native[:, iy-12:iy+13, ix-12:ix+13]
        if not np.all(np.isfinite(patch)) or np.any(patch >= 15100):
            usable[index] = False
    if int(np.count_nonzero(usable)) < 6:
        raise ValueError("Fewer than six source-valid ColorChecker patches")
    centre_before = np.mean(baseline_camera, axis=0)
    centre_after = np.mean(frame_camera, axis=0)
    certificate = dict(
        contract="ffac-v5-source-only-per-frame-chart-registration-v1",
        baseline_camera_jpeg=str(baseline), baseline_camera_sha256=baseline_camera_sha256,
        frame_camera_jpeg=str(frame), frame_camera_sha256=frame_camera_sha256,
        baseline_review_to_camera_h=bh.tolist(), frame_review_to_camera_h=fh.tolist(),
        baseline_chart_xy8=xy.tolist(), frame_chart_xy8=frame_xy.tolist(),
        frame_chart_usable=usable.tolist(), template_box_camera_xyxy=list(template_box),
        search_box_camera_xyxy=list(search_box), ratio_matches=len(good),
        ransac_inliers=int(np.count_nonzero(keep)), inlier_residual_p95_camera_px=p95,
        centre_shift_camera_xy=(centre_after-centre_before).tolist(),
        centre_shift_camera_px=float(np.linalg.norm(centre_after-centre_before)),
        clipped_or_nonfinite_patches=int(np.count_nonzero(~usable)),
        usable_patches=int(np.count_nonzero(usable)),
        raw_saturation_threshold="black-subtracted >=15100 invalid",
        registration_affine_template_to_search=affine.tolist(),
    )
    return frame_xy, usable, certificate
