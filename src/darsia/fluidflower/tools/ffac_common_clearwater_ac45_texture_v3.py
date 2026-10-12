"""Bounded, source-only wash-to-baseline static-sand geometry refinement.

The reference image is rendered once from the original sensor pixels.  A small
green-CFA texture homography is composed into that inverse warp; neither a
phase image nor an already corrected scientific image is resampled.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.project import sha256_file
from . import ffac_common_clearwater_ac45_alt as exact
from . import ffac_common_clearwater_ac45_alt_profile_v2 as instrument
from . import ffac_raw_geometry as original_geo
from . import ffac_raw_native_ea as original_renderer


class _StaticSandRefinement(instrument._KnownAlternateGeo):
    """One render's private geometry state; no original module is changed."""

    def __init__(self):
        self.correction = np.eye(3, dtype=np.float64)
        self.receipt = None

    def _raw_xy(self, u, v, homography, parameters, distortion, ca=None):
        return original_geo._raw_xy(u, v, homography @ self.correction,
            parameters, distortion, ca)

    def _raw_review_residual(self, green_plane, native_green, homography, parameters, distortion):
        hh, ww = 1489, 2736
        ay, ax = np.mgrid[0:hh, 0:ww].astype(np.float32)
        rx, ry = original_geo._raw_xy(2*ax+.5, 2*ay+.5, homography, parameters, distortion)
        raw_log = cv2.remap(np.log(np.maximum(green_plane, 1.0)),
            ((rx-.5)/2).astype(np.float32), ((ry-.5)/2).astype(np.float32),
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        jpeg_green = native_green[:2978].reshape(hh, 2, ww, 2).mean((1, 3))
        a = original_geo._bandpass(np.log(jpeg_green + 1.0))
        b = original_geo._bandpass(raw_log)
        # The original center and right-lower acceptance samples are held out.
        # Points here are fixed before observing any refined residual.
        points = [(x, y) for y in range(300, 1350, 70) for x in range(300, 2500, 70)
            if not (800 <= x <= 1800 and 500 <= y <= 1100)
            and not (1900 <= x <= 2650 and 1050 <= y <= 1488)]
        rows = original_geo._matched_rows(a, b, points, template_half=20,
            search_half=5, guess=lambda x, y: (x, y), min_correlation=.6)
        if len(rows) < 100:
            raise ValueError(f"Insufficient independent static-sand matches for wash refinement: {len(rows)}")
        half_correction, inliers = cv2.findHomography(rows[:, :2].astype(np.float32),
            rows[:, 2:4].astype(np.float32), cv2.RANSAC, .75)
        if half_correction is None or inliers is None or float(inliers.mean()) < .9:
            raise ValueError("Wash texture refinement has less than 90% inliers")
        half_to_full = np.array([[2., 0., .5], [0., 2., .5], [0., 0., 1.]])
        correction = half_to_full @ half_correction @ np.linalg.inv(half_to_full)
        correction /= correction[2, 2]
        if not np.isfinite(correction).all():
            raise ValueError("Non-finite static wash geometry correction")
        points_full = np.array([(x, y) for y in (0, 750, 1500, 2250, 2978)
            for x in (0, 1368, 2736, 4104, 5471)], np.float32)
        q = cv2.perspectiveTransform(points_full[None], correction)[0]
        qx = cv2.perspectiveTransform((points_full + [1, 0])[None], correction)[0]
        qy = cv2.perspectiveTransform((points_full + [0, 1])[None], correction)[0]
        jac = (qx[:, 0]-q[:, 0])*(qy[:, 1]-q[:, 1])-(qx[:, 1]-q[:, 1])*(qy[:, 0]-q[:, 0])
        displacements = np.linalg.norm(q - points_full, axis=1)
        if float(displacements.max()) > 8. or float(jac.min()) < .95 or float(jac.max()) > 1.05:
            raise ValueError("Static wash correction exceeds the predeclared 8 px / 0.95–1.05 Jacobian bounds")
        self.correction = correction
        self.receipt = dict(contract="ffac-clearwater-static-sand-registration-v1",
            fitted_source="independent static green-CFA texture, wash ARW against own pre-injection JPEG canvas",
            image_colour_or_phase_fit=False, model_fit=False,
            training_matches=int(len(rows)), fit_inlier_fraction=float(inliers.mean()),
            half_grid_template_half=20, search_half=5, min_correlation=.6,
            holdout="original center and right_lower_sand direct RAW/JPEG acceptance samples excluded from geometry fitting",
            correction_native_to_previous_native=correction.tolist(),
            original_review_to_camera_h=homography.tolist(),
            final_review_to_camera_h=(homography @ correction).tolist(),
            maximum_displacement_px=float(displacements.max()), maximum_allowed_displacement_px=8.,
            jacobian_min=float(jac.min()), jacobian_max=float(jac.max()), jacobian_allowed=[.95, 1.05],
            interpolation="original sensor-to-native bilinear warp only; no second image resampling",
            wrapper_module_sha256=sha256_file(Path(__file__)))
        return original_geo._raw_review_residual(green_plane, native_green,
            homography @ correction, parameters, distortion)


def registered_texture_refined_wash(arw_path, camera_jpeg_path, review_native_jpeg_path,
        *, expected_source_hashes, content_box, known_alt_camera_path):
    exact._check_parents()
    if dict(expected_source_hashes) != exact.EXACT_SOURCES:
        raise ValueError("Wash refinement is restricted to the exact predeclared source triple")
    if instrument.camera_metadata(camera_jpeg_path) != instrument.KNOWN_CAMERA_METADATA:
        raise ValueError("Measured instrument metadata differs from calibrated alternate profile")
    known = Path(known_alt_camera_path)
    if sha256_file(known) != instrument.KNOWN_ALT_CAMERA_SHA256 or instrument.camera_metadata(known) != instrument.KNOWN_CAMERA_METADATA:
        raise ValueError("Known alternate instrument calibration is not source verified")
    exact._exact_tags(arw_path)
    geometry = _StaticSandRefinement()
    call = exact._local_function(original_renderer.registered_edge_aware_raw_linear, geo=geometry)
    linear, certificate = call(arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box)
    if geometry.receipt is None:
        raise ValueError("Static wash refinement did not produce a geometry receipt")
    certificate = dict(certificate)
    certificate["review_to_camera_h"] = geometry.receipt["final_review_to_camera_h"]
    certificate["source_only_static_sand_refinement"] = geometry.receipt
    certificate["source_exact_alternate_lens"] = dict(
        contract="ffac-common-ac45-clearwater-instrument-profile-v3",
        source_sha256=exact.EXACT_SOURCES, camera_metadata=instrument.KNOWN_CAMERA_METADATA,
        known_alternate_calibration_source=dict(path=str(known), sha256=instrument.KNOWN_ALT_CAMERA_SHA256),
        intrinsic_parameters=dict(original_geo._PROFILE["early"]),
        profile_selection="measured alternate D/CA table and matched instrument EXIF; existing early intrinsics",
        static_sand_geometry_refinement=True, geometry_gate_unchanged=True,
        parent_renderer_sha256=exact.PARENT_RENDERER_SHA, parent_geometry_sha256=exact.PARENT_GEOMETRY_SHA,
        wrapper_module_sha256=sha256_file(Path(__file__)), module_mutation=False)
    return linear, certificate


def prepare_texture_refined_ac45(task_row, output, *, known_alt_camera_path):
    # Only the call-local renderer differs; source selection, photometry and
    # original acceptance gates are the instrument wrapper's unchanged code.
    prepare = exact._local_function(instrument.prepare_known_alternate_ac45,
        registered_known_alternate_wash=registered_texture_refined_wash, __file__=__file__)
    result = prepare(task_row, output, known_alt_camera_path=known_alt_camera_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--known-alt-camera", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    rows = json.loads(args.tasks.read_text(encoding="utf-8-sig"))["rows"]
    selected = [row for row in rows if row["run"] == "ac45"]
    if len(selected) != 1:
        raise ValueError("One exact predeclared AC45 source task is required")
    print(json.dumps(prepare_texture_refined_ac45(selected[0], args.output,
        known_alt_camera_path=args.known_alt_camera), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
