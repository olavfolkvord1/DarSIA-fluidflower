"""Source-exact Sony ARW to FF_AC native-review *content* pixels.

This is an isolated CPU reference materializer, not a model or label migration.
The returned image has the existing 2979 x 5472 native content geometry;
annotations stay in that coordinate system.  All inputs must be bound by SHA.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import tifffile
from PIL import Image


_PROFILE = {
    "early": dict(cu=2735.941390570268, cv=1540.0423206018281,
                  tx=0.008900502454740798, ty=-0.008331840772507504,
                  R=3291.1963476237365, amp=1.0033805150829112),
    "late": dict(cu=2735.971732158777, cv=1539.888967731643,
                 tx=0.00556404054693898, ty=-0.00551962331201813,
                 R=3291.066868836233, amp=1.0034606095607952),
}
_DISTORTION = {
    "early": (720, 698, 627, 507, 337, 128, -111, -365, -610, -839, -1053),
    "late": (716, 699, 630, 509, 339, 129, -112, -366, -613, -842, -1058),
}
_CA_RED = (0, 512, 384, 384, 384, 512, 512, 640, 768, 896, 896)
_CA_BLUE = (0, 512, 640, 768, 768, 640, 512, 384, 256, 128, 256)
_KNOTS = np.linspace(0.0, 1.0, 11)
_CONTENT_WH = (5472, 2979)
_CAMERA_WH = (5472, 3080)
_RAW_WH = (5496, 3672)


def _sha256(path: Path) -> str:
    from darsia.fluidflower.project import sha256_file
    return sha256_file(Path(path))


def _sony_tags(arw_path: Path) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    with tifffile.TiffFile(arw_path) as tiff:
        sub = tiff.pages[0].pages[0]
        d = tuple(int(v) for v in sub.tags[0x7037].value)
        ca = tuple(int(v) for v in sub.tags[0x7035].value)
    if d[0] != 11 or len(d) < 12 or ca[0] != 22 or len(ca) < 23:
        raise ValueError("Unexpected Sony ARW lens-table structure")
    red, blue = ca[1:12], ca[12:23]
    if red != _CA_RED or blue != _CA_BLUE:
        raise ValueError("Sony CA table differs from frozen FF_AC calibration")
    return d[1:12], red, blue


def _bandpass(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), 1.2) - cv2.GaussianBlur(image, (0, 0), 6.0)


def _matched_rows(a: np.ndarray, b: np.ndarray, points,
                  *, template_half: int, search_half: int,
                  guess, min_correlation: float) -> np.ndarray:
    rows = []
    h, s = template_half, search_half
    for x, y in points:
        x, y = int(x), int(y)
        if x-h < 0 or y-h < 0 or x+h+1 > a.shape[1] or y+h+1 > a.shape[0]:
            continue
        qx, qy = guess(x, y)
        qx, qy = int(round(qx)), int(round(qy))
        if qx-h-s < 0 or qy-h-s < 0 or qx+h+s+1 > b.shape[1] or qy+h+s+1 > b.shape[0]:
            continue
        template = a[y-h:y+h+1, x-h:x+h+1]
        search = b[qy-h-s:qy+h+s+1, qx-h-s:qx+h+s+1]
        if float(template.std()) < 1e-4:
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (px, py) = cv2.minMaxLoc(response)
        if score < min_correlation or px in (0, response.shape[1]-1) or py in (0, response.shape[0]-1):
            continue
        cx = response[py, px-1] + response[py, px+1] - 2*score
        cy = response[py-1, px] + response[py+1, px] - 2*score
        if cx >= 0 or cy >= 0:
            continue
        dx = .5*(response[py, px-1]-response[py, px+1])/cx
        dy = .5*(response[py-1, px]-response[py+1, px])/cy
        rows.append((x, y, qx-s+px+dx, qy-s+py+dy, score))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def _metrics(rows: np.ndarray, scale: float = 1.0) -> dict:
    if len(rows) == 0:
        return {"n": 0, "p50_px": None, "p95_px": None, "max_px": None}
    error = scale*np.linalg.norm(rows[:, 2:4]-rows[:, :2], axis=1)
    return {"n": len(error), "p50_px": float(np.median(error)),
            "p95_px": float(np.percentile(error, 95)),
            "max_px": float(error.max()),
            "median_match_correlation": float(np.median(rows[:, 4]))}


def _review_to_camera(review_g: np.ndarray, camera_g: np.ndarray) -> tuple[np.ndarray, dict]:
    a = _bandpass(np.log(review_g + 1.0))
    b = _bandpass(np.log(camera_g + 1.0))
    # Fitting only on the middle leaves the chart and right-lower sand independent.
    grid = ((x, y) for y in range(500, 2400, 180) for x in range(900, 4500, 180))
    rows = _matched_rows(a, b, grid, template_half=40, search_half=48,
                         guess=lambda x, y: (x, y+15), min_correlation=.5)
    if len(rows) < 100:
        raise ValueError(f"Insufficient review-to-camera matches: {len(rows)}")
    h, inliers = cv2.findHomography(rows[:, :2].astype(np.float32),
                                   rows[:, 2:4].astype(np.float32), cv2.RANSAC, 1.0)
    if h is None or inliers is None or inliers.mean() < .9:
        raise ValueError("Review-to-camera homography failed source checks")
    if not np.all(np.isfinite(h)):
        raise ValueError("Nonfinite homography")
    # Sampled local orientation must not reverse or fold the native content.
    pts = np.array([(x, y) for y in (0, 750, 1500, 2250, 2978)
                    for x in (0, 1368, 2736, 4104, 5471)], dtype=np.float32)
    q = cv2.perspectiveTransform(pts[None], h)[0]
    qx = cv2.perspectiveTransform((pts + [1, 0])[None], h)[0]
    qy = cv2.perspectiveTransform((pts + [0, 1])[None], h)[0]
    jac = (qx[:, 0]-q[:, 0])*(qy[:, 1]-q[:, 1]) - (qx[:, 1]-q[:, 1])*(qy[:, 0]-q[:, 0])
    if np.any(jac <= 0):
        raise ValueError("Review-to-camera homography folds native content")

    def predicted(x, y):
        point = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), h)[0, 0]
        return float(point[0]), float(point[1])

    zones = {}
    for name, box, step in (
        ("chart", (5000, 450, 5350, 850), 50),
        ("right_lower_sand", (3900, 2200, 5200, 2850), 80),
    ):
        x0, y0, x1, y1 = box
        checks = _matched_rows(a, b,
                               ((x, y) for y in range(y0, y1, step)
                                for x in range(x0, x1, step)),
                               template_half=25, search_half=10,
                               guess=predicted, min_correlation=.55)
        if len(checks) < 20:
            raise ValueError(f"Insufficient held-out review-camera matches in {name}")
        pred = cv2.perspectiveTransform(checks[:, None, :2].astype(np.float32), h)[:, 0]
        residual = np.linalg.norm(pred-checks[:, 2:4], axis=1)
        zones[name] = {"n": len(checks), "p50_px": float(np.median(residual)),
                       "p95_px": float(np.percentile(residual, 95)),
                       "max_px": float(residual.max()),
                       "median_match_correlation": float(np.median(checks[:, 4]))}
    return h, {"fit_n": len(rows), "fit_inlier_fraction": float(inliers.mean()),
               "sampled_jacobian_min": float(jac.min()), "heldout": zones}


def _raw_xy(u: np.ndarray, v: np.ndarray, h: np.ndarray, params: dict,
            distortion: Sequence[int], ca: Sequence[int] | None = None):
    uv = np.column_stack((u.ravel(), v.ravel())).astype(np.float32)
    xy = cv2.perspectiveTransform(uv[None], h)[0]
    du, dv = xy[:, 0]-params["cu"], xy[:, 1]-params["cv"]
    radius = np.hypot(du, dv)
    s = 1 + params["amp"]*np.interp(radius/params["R"], _KNOTS, distortion)/16384.0
    if ca is not None:
        s *= 1 + np.interp(radius/params["R"], _KNOTS, ca)/float(2**21)
    x = params["cu"]+12+params["tx"]+s*du
    y = params["cv"]+296+params["ty"]+s*dv
    return x.reshape(u.shape), y.reshape(v.shape)


def _raw_review_residual(g_plane: np.ndarray, native_green: np.ndarray,
                         h: np.ndarray, params: dict, distortion: Sequence[int]) -> dict:
    hh, ww = 1489, 2736  # exact 2x2 means inside the 2979x5472 native content
    ay, ax = np.mgrid[0:hh, 0:ww].astype(np.float32)
    u, v = 2*ax+.5, 2*ay+.5
    rx, ry = _raw_xy(u, v, h, params, distortion)
    gx = ((rx-.5)/2).astype(np.float32)
    gy = ((ry-.5)/2).astype(np.float32)
    raw = cv2.remap(np.log(np.maximum(g_plane, 1.0)), gx, gy, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    jpeg = native_green[:2978].reshape(hh, 2, ww, 2).mean((1, 3))
    a, b = _bandpass(np.log(jpeg+1.0)), _bandpass(raw)
    zones = {}
    for name, box, step in (
        ("center", (850, 550, 1750, 1050), 32),
        ("right_lower_sand", (1950, 1100, 2600, 1450), 32),
        ("chart", (2500, 225, 2670, 425), 20),
    ):
        x0, y0, x1, y1 = box
        rows = _matched_rows(a, b,
                             ((x, y) for y in range(y0, y1, step)
                              for x in range(x0, x1, step)),
                             template_half=16, search_half=5,
                             guess=lambda x, y: (x, y), min_correlation=.6)
        zones[name] = _metrics(rows, scale=2.0)
        if zones[name]["n"] < 20:
            raise ValueError(f"Insufficient direct RAW-review matches in {name}")
    return zones


def registered_raw_linear(
    arw_path: str | Path, camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path, *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    """Return CPU-remapped linear sensor RGB and a source/geometry certificate.

    `review_native_jpeg_path` must be the *same frame* as the ARW/JPEG pair.
    `expected_source_hashes` requires `arw_sha256`, `camera_jpeg_sha256`,
    `review_native_jpeg_sha256`.  The last hash must come from the immutable
    native review manifest/revision, not from an unbound file listing.
    """
    import rawpy
    arw_path = Path(arw_path)
    camera_jpeg_path = Path(camera_jpeg_path)
    review_native_jpeg_path = Path(review_native_jpeg_path)
    if arw_path.stem.lower() != camera_jpeg_path.stem.lower():
        raise ValueError("ARW and camera JPEG stem mismatch")
    if len(content_box) != 4:
        raise ValueError("content_box must be (x,y,w,h)")
    bx, by, width, height = map(int, content_box)
    if (width, height) != _CONTENT_WH:
        raise ValueError("Unrecognized native review content geometry")
    for key, path in (
        ("arw_sha256", arw_path),
        ("camera_jpeg_sha256", camera_jpeg_path),
        ("review_native_jpeg_sha256", review_native_jpeg_path),
    ):
        expected = expected_source_hashes.get(key)
        if not expected or _sha256(path) != expected.lower():
            raise ValueError(f"Missing or mismatched exact source hash: {key}")
    with Image.open(camera_jpeg_path) as image:
        if image.size != _CAMERA_WH:
            raise ValueError("Camera JPEG dimensions differ from frozen Sony contract")
        camera_green = np.asarray(image.convert("RGB"))[..., 1].astype(np.float32)
    with Image.open(review_native_jpeg_path) as image:
        if bx < 0 or by < 0 or bx+width > image.width or by+height > image.height:
            raise ValueError("Content box outside native review canvas")
        review_green = np.asarray(image.convert("RGB"))[by:by+height, bx:bx+width, 1].astype(np.float32)
        canvas_wh = image.size
    h, h_cert = _review_to_camera(review_green, camera_green)

    run = next((part.upper() for part in arw_path.parts
                if part.upper().startswith("AC") and part[2:].isdigit()), None)
    if run is None:
        raise ValueError("Cannot identify FF_AC run from original ARW path")
    profile = "early" if int(run[2:]) <= 44 else "late"
    distortion, ca_red, ca_blue = _sony_tags(arw_path)
    if distortion != _DISTORTION[profile]:
        raise ValueError(f"Sony distortion profile mismatch for {run}")
    with rawpy.imread(str(arw_path)) as raw:
        if raw.raw_image_visible.shape != (_RAW_WH[1], _RAW_WH[0]):
            raise ValueError("RAW visible dimensions differ from frozen Sony contract")
        if not np.array_equal(raw.raw_pattern, [[0, 1], [3, 2]]) or raw.color_desc != b"RGBG":
            raise ValueError("CFA Bayer pattern differs from frozen Sony contract")
        if tuple(raw.black_level_per_channel) != (800, 800, 800, 800):
            raise ValueError("RAW black level differs from frozen Sony contract")
        mosaic = raw.raw_image_visible.astype(np.float32).copy()
    planes = (
        mosaic[0::2, 0::2]-800,
        .5*(mosaic[0::2, 1::2]+mosaic[1::2, 0::2])-800,
        mosaic[1::2, 1::2]-800,
    )
    params = _PROFILE[profile]
    direct = _raw_review_residual(planes[1], review_green, h, params, distortion)
    output = np.empty((3, height, width), dtype=np.float32)
    for y0 in range(0, height, 128):
        y1 = min(height, y0+128)
        ys, xs = np.mgrid[y0:y1, 0:width].astype(np.float32)
        for channel, (plane, ca, phase) in enumerate(zip(
            planes, (ca_red, None, ca_blue), ((0., 0.), (.5, .5), (1., 1.)), strict=True
        )):
            rx, ry = _raw_xy(xs, ys, h, params, distortion, ca)
            mx = ((rx-phase[0])/2).astype(np.float32)
            my = ((ry-phase[1])/2).astype(np.float32)
            output[channel, y0:y1] = cv2.remap(
                plane, mx, my, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
            )
    certificate = {
        "contract": "ffac-source-exact-linear-raw-in-native-review-content-v1",
        "run": run, "profile": profile,
        "sources": {key: expected_source_hashes[key] for key in (
            "arw_sha256", "camera_jpeg_sha256", "review_native_jpeg_sha256")},
        "camera_jpeg_wh": _CAMERA_WH, "raw_visible_wh": _RAW_WH,
        "review_canvas_wh": canvas_wh, "content_box_xywh": [bx, by, width, height],
        "review_to_camera_h": h.tolist(), "review_camera_geometry": h_cert,
        "raw_review_direct_residual": direct,
        "sony_distortion_table": list(distortion),
        "sony_ca_red_table": list(ca_red), "sony_ca_blue_table": list(ca_blue),
        "output_shape": list(output.shape),
        "training_eligible": False,
        "eligibility_reason": "Per-annotation full-resolution boundary transfer and all-frame quality gates require separate certification; tiny boxes must fail closed.",
    }
    return output, certificate
