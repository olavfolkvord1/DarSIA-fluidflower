"""Photographic book-panel pose proposals, independent of sand geometry.

Feature agreement is a pose check, not approval of a hardware silhouette.
Each actual outline still requires photographic review. No nominal injector
point, old safe-sand complement, or phase prediction enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np

CONTRACT = "ffac-photographic-hardware-pose-proposal-v1"
CHART_CONTRACT = "ffac-v5-source-only-per-frame-chart-registration-v1"
HARDWARE_CAMERA_CONTRACT = "ffac-source-only-photographed-book-camera-affine-v1"


def _polygon(value):
    points = np.asarray(value, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all() or abs(cv2.contourArea(points.astype(np.float32))) <= 1):
        raise ValueError("Hardware outline requires a finite photographed polygon")
    return points


def _homography(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Hardware transport needs a nonsingular photographic homography")
    return matrix / matrix[2, 2] if abs(matrix[2, 2]) > 1e-12 else matrix


def transport_outline(polygon, homography):
    points, matrix = _polygon(polygon), _homography(homography)
    denominator = points @ matrix[2, :2] + matrix[2, 2]
    if (np.abs(denominator) < 1e-8).any() or np.any(denominator > 0) and np.any(denominator < 0):
        raise ValueError("Hardware polygon crosses the projective horizon")
    result = cv2.perspectiveTransform(points[None], matrix)[0]
    _polygon(result)
    return result


@dataclass(frozen=True)
class HardwareFeatures:
    points_xy: np.ndarray
    descriptors: np.ndarray | None
    mask_pixels: int


def photographed_features(gray, *, polygon=None, mask=None, max_features=6000, interior_margin_px=8):
    """SIFT from the photographed part/search window, never the sand domain."""
    image = np.asarray(gray)
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Hardware pose uses photographed uint8 grayscale only")
    if (polygon is None) == (mask is None):
        raise ValueError("Declare exactly one photographed polygon or search mask")
    if polygon is not None:
        selection = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(selection, [np.rint(_polygon(polygon)).astype(np.int32)], 255)
        if interior_margin_px:
            width = 2 * int(interior_margin_px) + 1
            selection = cv2.erode(selection, np.ones((width, width), np.uint8))
    else:
        selection = np.asarray(mask)
        if selection.dtype != bool or selection.shape != image.shape:
            raise ValueError("Hardware search must have its own photographed grid")
        selection = selection.astype(np.uint8) * 255
    yy, xx = np.nonzero(selection)
    if not len(xx):
        return HardwareFeatures(np.empty((0, 2), np.float32), None, 0)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    cv2.setRNGSeed(0)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=int(max_features)).detectAndCompute(
        image[y0:y1, x0:x1], selection[y0:y1, x0:x1])
    points = np.float32([point.pt for point in keypoints]).reshape(-1, 2)
    points += (x0, y0)
    return HardwareFeatures(points, descriptors, len(xx))


def fit_photographed_panel(reference, target, reference_outline, *, minimum_inliers=12,
        ratio_threshold=.75, maximum_p95_px=3., minimum_coverage=.025):
    """Return a provisional panel pose or an explicit failure; no guessed pose."""
    polygon = _polygon(reference_outline)
    audit = dict(contract=CONTRACT, pose_pass=False, photographic_outline_reviewed=False,
        hardware_exclusion_approved=False, phase_labels_used=False, sand_texture_used=False,
        reference_features=len(reference.points_xy), target_features=len(target.points_xy),
        minimum_inliers=int(minimum_inliers), maximum_p95_px=float(maximum_p95_px),
        minimum_reference_coverage=float(minimum_coverage), model_inference=False)
    if reference.descriptors is None or target.descriptors is None or len(target.points_xy) < 2:
        return dict(audit, failure="no photographed panel descriptors")
    pairs = cv2.BFMatcher().knnMatch(reference.descriptors, target.descriptors, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance]
    audit["ratio_matches"] = len(good)
    if len(good) < minimum_inliers:
        return dict(audit, failure="too few independent printed-panel matches")
    source = reference.points_xy[[item.queryIdx for item in good]]
    destination = target.points_xy[[item.trainIdx for item in good]]
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.findHomography(source, destination, cv2.RANSAC, maximum_p95_px)
    if matrix is None or inliers is None or int(inliers.sum()) < minimum_inliers:
        return dict(audit, failure="insufficient homography inliers")
    keep = inliers.ravel().astype(bool)
    residual = np.linalg.norm(cv2.perspectiveTransform(source[keep][None], matrix)[0] - destination[keep], axis=1)
    coverage = abs(cv2.contourArea(cv2.convexHull(source[keep]))) / abs(cv2.contourArea(polygon.astype(np.float32)))
    audit.update(inliers=int(keep.sum()), residual_median_px=float(np.median(residual)),
        residual_p95_px=float(np.percentile(residual, 95)), reference_coverage=float(coverage))
    if audit["residual_p95_px"] > maximum_p95_px or coverage < minimum_coverage:
        return dict(audit, failure="panel residual or spatial coverage gate failed")
    try:
        projected = transport_outline(polygon, matrix)
    except ValueError as error:
        return dict(audit, failure=str(error))
    if cv2.contourArea(polygon.astype(np.float32), oriented=True) * cv2.contourArea(projected.astype(np.float32), oriented=True) <= 0:
        return dict(audit, failure="mirrored hardware pose")
    return dict(audit, pose_pass=True, H_reference_to_frame=np.asarray(matrix).tolist(),
        proposed_outline_xy=projected.tolist(),
        source_inlier_xy=source[keep].tolist(), frame_inlier_xy=destination[keep].tolist())


def chart_camera_transport(frame_zero_certificate, frame_certificate, *,
        frame_zero_source_sha256, frame_source_sha256):
    """Transport a reviewed run outline with chart motion, never grain motion.

Both certificates originate from the same exact baseline-camera acquisition.
The two independent frame-camera affine registrations are composed through
that baseline. New/reselected sources require their own certificate.
"""
    certificates = (frame_zero_certificate, frame_certificate)
    for certificate, digest in zip(certificates, (frame_zero_source_sha256, frame_source_sha256)):
        if certificate.get("contract") != CHART_CONTRACT or certificate.get("frame_camera_sha256") != digest:
            raise ValueError("Hardware chart transport lacks the exact source certificate")
    if (not frame_zero_certificate.get("baseline_camera_sha256")
            or frame_zero_certificate["baseline_camera_sha256"] != frame_certificate.get("baseline_camera_sha256")):
        raise ValueError("Hardware frames use different chart-camera baselines")
    affines = []
    for certificate in certificates:
        affine = np.asarray(certificate["registration_affine_template_to_search"], np.float64)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise ValueError("Invalid photographed chart-camera affine")
        template = np.asarray(certificate["template_box_camera_xyxy"], np.float64)
        search = np.asarray(certificate["search_box_camera_xyxy"], np.float64)
        if template.shape != (4,) or search.shape != (4,) or not np.isfinite(template).all() or not np.isfinite(search).all():
            raise ValueError("Chart-camera ROI origins are missing")
        t1, t2 = np.eye(3), np.eye(3)
        t1[:2, 2], t2[:2, 2] = -template[:2], search[:2]
        affines.append(t2 @ np.vstack((affine, (0, 0, 1))) @ t1)
    frame_zero_h = _homography(frame_zero_certificate["frame_review_to_camera_h"])
    frame_h = _homography(frame_certificate["frame_review_to_camera_h"])
    matrix = np.linalg.inv(frame_h) @ affines[1] @ np.linalg.inv(affines[0]) @ frame_zero_h
    return _homography(matrix), dict(contract=CONTRACT, coordinate_space="native review pixels",
        frame_zero_source_sha256=frame_zero_source_sha256, frame_source_sha256=frame_source_sha256,
        baseline_camera_sha256=frame_certificate["baseline_camera_sha256"],
        chart_camera_registration_used=True, grain_registration_used=False,
        per_frame_photographic_outline_review_required=True, hardware_exclusion_approved=False)


def panel_anchored_housing(left_outline, right_outline, *, left_anchor_indices, right_anchor_indices):
    """Propose the photographed connecting housing between two posed panels.

    The four anchors refer to explicitly traced physical inner case corners.
    This does not infer a shared plane or borrow the sand homography. Approval
    still comes from the actual full-book photograph/overlay.
    """
    left, right = _polygon(left_outline), _polygon(right_outline)
    if any(len(indices) != 2 or any(type(i) is not int for i in indices)
           for indices in (left_anchor_indices, right_anchor_indices)):
        raise ValueError("Connecting housing requires named photographed corner indices")
    try:
        lt, lb = left[list(left_anchor_indices)]
        rt, rb = right[list(right_anchor_indices)]
    except IndexError as error:
        raise ValueError("Housing anchor does not exist in its source outline") from error
    polygon = _polygon([lt, rt, rb, lb])
    if not cv2.isContourConvex(polygon.astype(np.float32)):
        raise ValueError("Connecting hardware housing has crossed/folded anchor ordering")
    return polygon, dict(contract=CONTRACT,
        method="panel-anchored-photographed-book-housing",
        left_anchor_indices=list(left_anchor_indices), right_anchor_indices=list(right_anchor_indices),
        grain_registration_used=False, shared_panel_homography_assumed=False,
        photographic_outline_review_required=True, hardware_exclusion_approved=False)


def camera_pose(*, source_sha256, H_native_to_camera, chart_certificate=None,
                exact_baseline_source=False, baseline_camera_sha256=None):
    """Normalise an already SHA-verified per-frame photographic receipt.

    Exact frozen B acquisitions legitimately have no chart-registration
    certificate; only that explicitly verified condition admits camera
    identity. For every other acquisition its own chart certificate is needed.
    """
    if not isinstance(source_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None:
        raise ValueError("Hardware camera pose needs its exact member-source SHA")
    h = _homography(H_native_to_camera)
    if exact_baseline_source:
        if chart_certificate is not None or baseline_camera_sha256 != source_sha256:
            raise ValueError("Exact B hardware identity lacks its explicit baseline source binding")
        return dict(source_sha256=source_sha256,baseline_camera_sha256=source_sha256,
            H_native_to_camera=h.tolist(),H_baseline_camera_to_frame_camera=np.eye(3).tolist(),
            method="explicit-SHA-bound-exact-B-camera-identity",grain_registration_used=False)
    chart = chart_certificate
    if not isinstance(chart, dict) or chart.get("contract") != CHART_CONTRACT or chart.get("frame_camera_sha256") != source_sha256:
        raise ValueError("New hardware source requires its own photographed chart-camera pose")
    baseline = chart.get("baseline_camera_sha256")
    if not isinstance(baseline,str) or re.fullmatch(r"[a-f0-9]{64}",baseline) is None:
        raise ValueError("Hardware chart pose lacks its baseline-source SHA")
    if baseline_camera_sha256 is not None and baseline != baseline_camera_sha256:
        raise ValueError("Hardware chart pose belongs to another baseline acquisition")
    if not np.allclose(_homography(chart["frame_review_to_camera_h"]),h,atol=1e-10,rtol=1e-12):
        raise ValueError("Hardware chart pose and photographic renderer use different geometry")
    affine = np.asarray(chart["registration_affine_template_to_search"], np.float64)
    template,search=(np.asarray(chart[name],np.float64) for name in
        ("template_box_camera_xyxy","search_box_camera_xyxy"))
    if affine.shape!=(2,3) or template.shape!=(4,) or search.shape!=(4,):
        raise ValueError("Hardware chart-camera origins/affine changed")
    t1,t2=np.eye(3),np.eye(3)
    t1[:2,2],t2[:2,2]=-template[:2],search[:2]
    a=_homography(t2 @ np.vstack((affine,(0,0,1))) @ t1)
    return dict(source_sha256=source_sha256,baseline_camera_sha256=baseline,
        H_native_to_camera=h.tolist(),H_baseline_camera_to_frame_camera=a.tolist(),
        method="source-only-per-frame-chart-camera-registration",grain_registration_used=False)


def transport_between_camera_poses(frame_zero_pose, frame_pose):
    """Native frame0 -> own frame, including a separately acquired true B."""
    if frame_zero_pose.get("baseline_camera_sha256") != frame_pose.get("baseline_camera_sha256"):
        raise ValueError("Hardware transport cannot cross different chart baselines")
    first_h,second_h=(_homography(pose["H_native_to_camera"]) for pose in (frame_zero_pose,frame_pose))
    first_a,second_a=(_homography(pose["H_baseline_camera_to_frame_camera"]) for pose in (frame_zero_pose,frame_pose))
    return _homography(np.linalg.inv(second_h) @ second_a @ np.linalg.inv(first_a) @ first_h)


def fit_photographed_camera_affine(reference, target, *, source_sha256,
        baseline_camera_sha256, H_native_to_camera, source_geometry_sha256,
        minimum_inliers=50, maximum_p95_px=2.):
    """Fit camera motion from verified photographed book texture only.

    This hardware-only certificate does not claim RAW swatch saturation QA or
    light calibration. New sources use their own measured native-camera H.
    """
    for value in (source_sha256,baseline_camera_sha256,source_geometry_sha256):
        if not isinstance(value,str) or re.fullmatch(r"[a-f0-9]{64}",value) is None:
            raise ValueError("Photographic camera affine requires exact source/geometry SHA bindings")
    h=_homography(H_native_to_camera)
    audit=dict(contract=HARDWARE_CAMERA_CONTRACT,pose_pass=False,
        source_sha256=source_sha256,baseline_camera_sha256=baseline_camera_sha256,
        source_geometry_sha256=source_geometry_sha256,H_native_to_camera=h.tolist(),
        grain_registration_used=False,phase_labels_used=False,raw_swatch_qa_claimed=False,
        radiometric_chart_certificate=False,minimum_inliers=int(minimum_inliers),
        maximum_p95_px=float(maximum_p95_px))
    if reference.descriptors is None or target.descriptors is None or len(target.points_xy)<2:
        return dict(audit,failure="No source-photographed book descriptors")
    pairs=cv2.BFMatcher().knnMatch(reference.descriptors,target.descriptors,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
    audit["ratio_matches"]=len(good)
    if len(good)<minimum_inliers:
        return dict(audit,failure="Too few photographed book matches")
    src=reference.points_xy[[match.queryIdx for match in good]]
    dst=target.points_xy[[match.trainIdx for match in good]]
    cv2.setRNGSeed(0)
    matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.)
    if matrix is None or inliers is None or int(inliers.sum())<minimum_inliers:
        return dict(audit,failure="Fewer than fixed minimum camera affine inliers")
    keep=inliers.ravel().astype(bool)
    residual=np.linalg.norm(cv2.transform(src[None],matrix)[0]-dst,axis=1)[keep]
    audit.update(inliers=int(keep.sum()),residual_median_px=float(np.median(residual)),
        residual_p95_px=float(np.percentile(residual,95)),
        source_inlier_camera_xy=src[keep].tolist(),frame_inlier_camera_xy=dst[keep].tolist())
    if audit["residual_p95_px"]>maximum_p95_px:
        return dict(audit,failure="Photographed camera affine residual gate failed")
    return dict(audit,pose_pass=True,H_baseline_camera_to_frame_camera=np.vstack((matrix,[0,0,1])).tolist(),
                method="own-JPEG-printed-book-affine-with-own-native-camera-geometry")


__all__ = ["CONTRACT", "HardwareFeatures", "photographed_features", "fit_photographed_panel",
           "transport_outline", "chart_camera_transport", "panel_anchored_housing", "camera_pose",
           "transport_between_camera_poses", "fit_photographed_camera_affine", "HARDWARE_CAMERA_CONTRACT"]
