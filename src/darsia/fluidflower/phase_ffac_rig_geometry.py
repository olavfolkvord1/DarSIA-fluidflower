"""Photograph-bound FF_AC rig geometry and physical native-pixel areas.

Coordinates are native pixel centres. A canvas is the same pixel grid with an
explicit native origin; photographs and annotations are never resampled here.
Measured physical boundaries and explicit fixture exclusions are inputs, not
inferred from phase predictions or annotations.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import BSpline, LSQUnivariateSpline

FORMAT = "ffac-photographic-rig-domain-v1"
PIXEL_AREA_MM2_FIXED_AC28 = 0.027053


def homography(value):
    matrix = np.asarray(value, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rig transform must be a finite 3x3 homography")
    if abs(np.linalg.det(matrix)) < 1e-14:
        raise ValueError("Rig transform is singular")
    return matrix


def transform_points(points, matrix):
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Rig points must be finite Nx2 native coordinates")
    projected = homography(matrix) @ np.column_stack((points, np.ones(len(points)))).T
    if np.any(np.abs(projected[2]) < 1e-12):
        raise ValueError("Rig transform crosses a projective horizon")
    return (projected[:2] / projected[2]).T


def own_reference_rig_transform(H_ac28_to_frame_native, H_ac28_to_reference_native):
    """Carry photographed rig edges from a run's own reference frame.

    The inter-run sand-scale mapping cancels algebraically. This transport is
    valid only with separately recorded within-run photographic stability;
    it does not certify an unobserved wall or an absolute metric scale.
    """
    matrix = homography(H_ac28_to_frame_native) @ np.linalg.inv(
        homography(H_ac28_to_reference_native))
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Within-run rig transform has no finite native origin")
    return matrix / matrix[2, 2]


def track_photographic_edge_family(candidate_rows, anchor_native_x, *, minimum_control_response=2.):
    """Continue a photographed seam through weak responses, before any fit.

    Rows contain y and candidate (native x, positive image-gradient response)
    pairs. Weak source responses participate in continuity, so an unavailable
    seam need not jump to another parallel bevel. They supply no fit or
    held-out control. All raw responses and omitted controls remain audited;
    no fitted geometry, phase label or acceptance residual enters this choice.
    """
    if not candidate_rows or not np.isfinite(anchor_native_x) or minimum_control_response <= 0:
        raise ValueError("A photographed anchor, positive source-response threshold and candidate rows are required")
    candidates = [np.asarray(row["candidates"], np.float64) for row in candidate_rows]
    positions = np.asarray([row["y"] for row in candidate_rows], np.float64)
    if (not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0)
            or any(c.ndim != 2 or c.shape[1] != 2 or not len(c)
                   or not np.isfinite(c).all() or np.any(c[:, 1] <= 0) for c in candidates)):
        raise ValueError("Photographic edge rows must be ordered and contain finite positive-response candidates")
    cost = (candidates[0][:, 0] - anchor_native_x) ** 2 / 2 - candidates[0][:, 1] / 20
    backs = []
    for previous, current in zip(candidates, candidates[1:]):
        all_cost = (current[:, 0, None] - previous[None, :, 0]) ** 2 + cost[None, :]
        best = np.argmin(all_cost, axis=1)
        backs.append(best)
        cost = all_cost[np.arange(len(current)), best] - current[:, 1] / 20
    chosen = [int(np.argmin(cost))]
    for back in reversed(backs):
        chosen.append(int(back[chosen[-1]]))
    chosen.reverse()
    controls, rejected, trace = [], [], []
    for row, candidates_in_row, selected in zip(candidate_rows, candidates, chosen):
        x, response = map(float, candidates_in_row[selected])
        trace.append(dict(y=float(row["y"]), x_native=x, response=response))
        if response >= minimum_control_response:
            controls.append([x, float(row["y"])])
        else:
            rejected.append(dict(position=float(row["y"]), reason="tracked_inner_family_below_source_contrast",
                                 candidate_native_x=x, response=response))
    return dict(controls_native=controls, weak_source_controls=rejected, chosen_source_trace=trace,
                all_candidate_responses_retained=True, fit_or_phase_used=False,
                minimum_control_response=float(minimum_control_response))


def fit_photographic_boundary_curve(points_native, *, dependent_axis=0,
                                   knot_spacing_px=200., gate_px=3.):
    """Fit a source-photograph edge without using its held-out positions.

    Alternating measured positions are held out. Fixed native-coordinate
    spline knots describe residual photographic curvature; every supplied
    control is retained, and no acceptance tolerance or residual is used to
    select a seam family. The seam's physical identification and trace method
    must be certified separately against the original photograph.
    """
    points = np.asarray(points_native, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or dependent_axis not in (0, 1)):
        raise ValueError("Photographic edge controls must be finite native Nx2 coordinates")
    if knot_spacing_px <= 0 or gate_px <= 0:
        raise ValueError("Photographic curve spacing and fixed acceptance gate must be positive")
    independent_axis = 1 - dependent_axis
    points = points[np.argsort(points[:, independent_axis], kind="stable")]
    independent, dependent = points[:, independent_axis], points[:, dependent_axis]
    if len(points) < 8 or np.any(np.diff(independent) <= 0):
        raise ValueError("A cubic photographic curve needs four fit positions and independent held-outs")
    fit = np.arange(len(points)) % 2 == 0
    knots = np.arange(independent[fit][0] + knot_spacing_px,
                      independent[fit][-1] - knot_spacing_px / 2., knot_spacing_px)
    spline = LSQUnivariateSpline(independent[fit], dependent[fit], knots, k=3)
    error = np.abs(spline(independent[~fit]) - dependent[~fit])
    t, c, k = spline._eval_args
    return dict(contract="ffac-photographic-native-boundary-curve-v1",
                dependent_axis=int(dependent_axis), knots_native=np.asarray(t).tolist(),
                coefficients_native=np.asarray(c).tolist(), degree=int(k),
                knot_spacing_px=float(knot_spacing_px), gate_px=float(gate_px),
                fit_points_native=points[fit].tolist(), heldout_points_native=points[~fit].tolist(),
                support_interval_native=[float(independent[0]), float(independent[-1])],
                heldout_count=int((~fit).sum()), median_abs_px=float(np.median(error)),
                p95_abs_px=float(np.percentile(error, 95)), max_abs_px=float(error.max()),
                passed=bool(np.median(error) <= gate_px and np.percentile(error, 95) <= gate_px),
                all_controls_retained=True, post_score_outliers_removed=False)


def evaluate_boundary_curve(curve, positions_native):
    """Evaluate a bound source curve; extrapolation must be reported by caller."""
    positions = np.asarray(positions_native, np.float64)
    if not np.isfinite(positions).all():
        raise ValueError("Boundary evaluation positions must be finite")
    return BSpline(np.asarray(curve["knots_native"], np.float64),
                   np.asarray(curve["coefficients_native"], np.float64),
                   int(curve["degree"]))(positions)


def inverse_homography_area_map(H_ac28_to_frame_native, shape_hw, *, origin_xy=(0, 0),
                                fixed_area_mm2=PIXEL_AREA_MM2_FIXED_AC28,
                                H_ac28_to_metric=None):
    """Area at each frame pixel from the inverse mapping, without scalar scale.

    For a projective map M, det J(x,y) = det(M)/(M[2]·[x,y,1])**3.
    A recorded optional AC28 rig-plane correction composes on the left of the
    inverse frame transform. This evaluates the mathematics, not metric
    eligibility: sand-texture registration must first be checked against the
    physical rig. A single reference correction cannot repair a rejected
    between-run rig transport.
    """
    height, width = map(int, shape_hw)
    if min(height, width) < 1 or not np.isfinite(fixed_area_mm2) or fixed_area_mm2 <= 0:
        raise ValueError("Invalid physical area or image shape")
    matrix = np.linalg.inv(homography(H_ac28_to_frame_native))
    if H_ac28_to_metric is not None:
        matrix = homography(H_ac28_to_metric) @ matrix
    x = np.arange(width, dtype=np.float64)[None, :] + float(origin_xy[0])
    y = np.arange(height, dtype=np.float64)[:, None] + float(origin_xy[1])
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if np.any(np.abs(denominator) < 1e-10) or np.min(denominator) * np.max(denominator) <= 0:
        raise ValueError("Physical area map crosses a projective horizon")
    area = fixed_area_mm2 * abs(np.linalg.det(matrix)) / np.abs(denominator) ** 3
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Physical pixel areas must be finite and positive")
    return area.astype(np.float32)


def rig_plane_area_map(H_frame_native_to_metric_pixel, shape_hw, *, origin_xy=(0, 0),
                      metric_pixel_area_mm2=PIXEL_AREA_MM2_FIXED_AC28):
    """Area from an independently certified rig-plane transform.

    The transform maps frame pixels directly into metric-reference pixels.
    It is deliberately separate from the sand-texture homography: the latter
    continues to place sand/facies objects even when rig controls reject using
    it as a physical camera-scale transform.
    """
    return inverse_homography_area_map(
        np.linalg.inv(homography(H_frame_native_to_metric_pixel)), shape_hw,
        origin_xy=origin_xy, fixed_area_mm2=metric_pixel_area_mm2)


def metric_plane_from_inner_corners(inner_corners_native, *, inner_width_mm,
                                    inner_height_mm, metric_pixel_mm=.16448):
    """Map four independently identified rigid corners into a metric rectangle.

    Corner order is top-left, top-right, bottom-right, bottom-left. Neither
    nominal crop dimensions nor sand boundaries identify these corners. This
    function only constructs the coordinate transform; the physical evidence
    and dimension source must be certified by the calling artifact builder.
    """
    corners=np.asarray(inner_corners_native,np.float64)
    if corners.shape!=(4,2) or not np.isfinite(corners).all():
        raise ValueError("Four finite independently measured inner rig corners are required")
    dims=np.asarray([inner_width_mm,inner_height_mm,metric_pixel_mm],float)
    if not np.isfinite(dims).all() or np.any(dims<=0):
        raise ValueError("Physical rig dimensions and metric pixel scale must be positive")
    edges=np.roll(corners,-1,axis=0)-corners
    turns=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if np.any(turns<=0):
        raise ValueError("Inner rig corners must form a convex clockwise image-space quadrilateral")
    target=np.array([[0,0],[inner_width_mm/metric_pixel_mm,0],
                     [inner_width_mm/metric_pixel_mm,inner_height_mm/metric_pixel_mm],
                     [0,inner_height_mm/metric_pixel_mm]],np.float64)
    matrix,_=cv2.findHomography(corners,target,method=0)
    matrix=homography(matrix)
    # Evaluating the full corner support also rejects a fold or horizon.
    projected=transform_points(corners,matrix)
    if not np.allclose(projected,target,atol=1e-4,rtol=0):
        raise ValueError("Rig-plane corner transform does not reproduce its physical anchors")
    return matrix/matrix[2,2]


def fit_independent_rig_plane(reference_rigid_points, frame_rigid_points, *,
                              heldout_reference_points, heldout_frame_points,
                              reference_inner_lines, frame_inner_edge_points,
                              gate_px=3., minimum_edge_points=1,
                              minimum_heldout_points=1):
    """Fit a rigid-photo transform and certify its independently held-out edges.

    Every supplied fit point is used. Failed held-out landmarks or edge offsets
    remain in the returned audit; they are never removed after scoring. The
    resulting geometry eligibility is separate from absolute scale approval.
    Nonempty independent photographic evidence is required. Sample counts and
    support spans belong in the audit; the specification does not impose an
    arbitrary minimum of 12 landmarks or 20 edge controls. Sources,
    photographed seam choice and dimension provenance are bound by the
    caller, rather than inferred from phase labels or nominal crop metadata.
    """
    def points(value,name):
        value=np.asarray(value,np.float64)
        if value.ndim!=2 or value.shape[1]!=2 or not np.isfinite(value).all():
            raise ValueError(name+" must contain finite native x/y coordinates")
        return value
    reference=points(reference_rigid_points,"Reference rigid anchors")
    frame=points(frame_rigid_points,"Frame rigid anchors")
    held_reference=points(heldout_reference_points,"Held-out reference anchors")
    held_frame=points(heldout_frame_points,"Held-out frame anchors")
    if reference.shape!=frame.shape or len(reference)<4:
        raise ValueError("At least four corresponding rigid-photo anchors are required")
    if held_reference.shape!=held_frame.shape:
        raise ValueError("Held-out rigid-photo anchors must correspond exactly")
    for array in (reference,frame):
        if np.linalg.matrix_rank(array-array.mean(axis=0))<2:
            raise ValueError("Rigid-photo anchors must span the rig plane, not one line")
    if not np.isfinite(gate_px) or gate_px<=0:
        raise ValueError("Physical photographic edge gate must be positive")
    matrix,_=cv2.findHomography(reference,frame,method=0)
    matrix=homography(matrix);matrix/=matrix[2,2]
    denominator=matrix[2,0]*reference[:,0]+matrix[2,1]*reference[:,1]+matrix[2,2]
    if (np.min(denominator)*np.max(denominator)<=0
            or np.any(np.linalg.det(matrix)/denominator**3<=0)):
        raise ValueError("Rigid-photo transform folds, reflects or crosses the rig plane horizon")
    predicted=transform_points(held_reference,matrix)
    error=np.linalg.norm(predicted-held_frame,axis=1)
    def audit(error,minimum):
        error=np.asarray(error,float)
        result=dict(sample_count=len(error),minimum_sample_count=int(minimum),
                    median_abs_px=None,p95_abs_px=None,max_abs_px=None,passed=False)
        if len(error):
            result.update(median_abs_px=float(np.median(error)),
                          p95_abs_px=float(np.percentile(error,95)),max_abs_px=float(np.max(error)))
            result['passed']=bool(len(error)>=minimum and result['median_abs_px']<=gate_px
                                  and result['p95_abs_px']<=gate_px)
        return result
    landmark_audit=audit(error,minimum_heldout_points)
    checks={};lines={}
    if set(reference_inner_lines)!=set(('left','right','bottom')) or set(frame_inner_edge_points)!=set(reference_inner_lines):
        raise ValueError("All three photographed inner walls/bottom need independent controls")
    inverse=np.linalg.inv(matrix)
    for side in ('left','right','bottom'):
        line=np.asarray(reference_inner_lines[side],float)
        if line.shape!=(3,) or not np.isfinite(line).all() or np.linalg.norm(line[:2])<1e-10:
            raise ValueError("Invalid independently traced reference "+side+" line")
        carried=inverse.T@line;carried/=np.linalg.norm(carried[:2])
        actual=points(frame_inner_edge_points[side],"Photographed "+side+" edge controls")
        offsets=np.abs(np.column_stack((actual,np.ones(len(actual))))@carried)
        checks[side]=audit(offsets,minimum_edge_points);lines[side]=carried.tolist()
    fit_error=np.linalg.norm(transform_points(reference,matrix)-frame,axis=1)
    fit_audit=audit(fit_error,len(reference))
    return dict(contract='ffac-independent-rig-plane-fit-v1',
                H_reference_rig_to_frame_native=matrix.tolist(),
                carried_inner_lines_native=lines,fit_controls=fit_audit,
                heldout_rigid_landmarks=landmark_audit,heldout_inner_edges=checks,
                gate_px=float(gate_px),all_fit_points_used=True,post_score_outliers_removed=False,
                photographic_geometry_eligible=bool(fit_audit['passed'] and landmark_audit['passed'] and all(v['passed'] for v in checks.values())),
                physical_scale_approved=False)


def inner_frame_step(profile, *, outer_search_px=500, interior_tail_px=150,
                     minimum_contrast=25., smoothing_px=5):
    """Find a bright wall's half-height inward edge on a full-resolution profile.

    The profile must run from the frame side toward sand. Rejected profiles
    remain missing evidence; callers must not substitute an assumed wall.
    """
    values = np.asarray(profile, np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < interior_tail_px + 10:
        raise ValueError("Frame profile is too short or nonfinite")
    smooth = ndimage.uniform_filter1d(values, int(smoothing_px), mode="nearest")
    interior = float(np.median(smooth[-interior_tail_px:]))
    stop = min(int(outer_search_px), len(values) - interior_tail_px)
    peak = int(np.argmax(smooth[:stop]))
    contrast = float(smooth[peak] - interior)
    if contrast < minimum_contrast:
        return None
    threshold = interior + contrast / 2
    crossings = np.flatnonzero(smooth[peak:stop] <= threshold)
    if not len(crossings):
        return None
    index = peak + int(crossings[0])
    if index == 0 or smooth[index - 1] <= threshold:
        return None
    # Linear interpolation avoids half-a-pixel rounding bias across camera groups.
    fraction = (smooth[index - 1] - threshold) / (smooth[index - 1] - smooth[index])
    edge = index - 1 + float(fraction)
    return dict(edge_px=edge, contrast=contrast, peak_px=peak,
                interior=interior, threshold=threshold)


def fit_boundary_line(points, *, dependent_axis=0, minimum_points=20):
    """Fit x(y) for walls or y(x) for the bottom, retaining rejected residuals."""
    points = np.asarray(points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum_points:
        raise ValueError("Too few independently observed physical-edge points")
    dependent = points[:, dependent_axis]
    independent = points[:, 1 - dependent_axis]
    design = np.column_stack((independent, np.ones(len(points))))
    keep = np.ones(len(points), bool)
    for _ in range(4):
        slope, offset = np.linalg.lstsq(design[keep], dependent[keep], rcond=None)[0]
        residual = dependent - (slope * independent + offset)
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
        next_keep = np.abs(residual) <= max(3., 3 * 1.4826 * mad)
        if next_keep.sum() < minimum_points:
            raise ValueError("Physical-edge fit has insufficient supported points")
        if np.array_equal(keep, next_keep):
            break
        keep = next_keep
    normal = np.array([1., -slope, -offset]) if dependent_axis == 0 else np.array([-slope, 1., -offset])
    normal /= np.linalg.norm(normal[:2])
    distances = np.abs(np.column_stack((points, np.ones(len(points)))) @ normal)
    return dict(line=normal.tolist(), sample_count=len(points), fit_sample_count=int(keep.sum()),
                rejected_count=int((~keep).sum()), median_abs_px=float(np.median(distances)),
                p95_abs_px=float(np.percentile(distances, 95)),
                max_abs_px=float(distances.max()))


def line_intersection(first, second):
    point = np.cross(np.asarray(first, np.float64), np.asarray(second, np.float64))
    if abs(point[2]) < 1e-12:
        raise ValueError("Physical boundaries do not intersect finitely")
    return point[:2] / point[2]


def photographic_observation_mask(H_native_to_camera, camera_wh, canvas_wh, *, native_origin_xy):
    """Whether the genuine camera actually photographed each review-canvas pixel.

    A camera-visible strip beyond the old scientific crop is observed but may
    lack features. Only pixels outside the camera's recorded footprint count
    as unobserved. Pixel-centre bounds are inclusive at 0 and camera_size-1.
    """
    camera_width,camera_height=map(int,camera_wh)
    width,height=map(int,canvas_wh)
    if min(camera_width,camera_height,width,height)<1:
        raise ValueError("Invalid camera or canvas dimensions")
    matrix=homography(H_native_to_camera)
    x=np.arange(width,dtype=float)[None,:]+float(native_origin_xy[0])
    y=np.arange(height,dtype=float)[:,None]+float(native_origin_xy[1])
    denominator=matrix[2,0]*x+matrix[2,1]*y+matrix[2,2]
    safe=np.abs(denominator)>1e-10
    with np.errstate(divide="ignore",invalid="ignore"):
        u=(matrix[0,0]*x+matrix[0,1]*y+matrix[0,2])/denominator
        v=(matrix[1,0]*x+matrix[1,1]*y+matrix[1,2])/denominator
    return safe & np.isfinite(u)&np.isfinite(v)&(u>=0)&(u<=camera_width-1)&(v>=0)&(v<=camera_height-1)


@dataclass(frozen=True)
class RigFrameContext:
    rig_domain: np.ndarray
    modelled: np.ndarray
    filled_observed: np.ndarray
    filled_unobserved: np.ndarray
    hardware_excluded: np.ndarray
    excluded_by_name: Mapping[str, np.ndarray]
    origin_xy: tuple[int, int]
    native_shape_hw: tuple[int, int]
    content_box_xywh: tuple[int, int, int, int]
    pixel_area_mm2: np.ndarray
    parameters: dict


def rig_frame_context(reference, H_ac28_to_frame_native, native_shape_hw, *,
                      feature_valid_mask, canvas_wh, content_box_xywh,
                      photographic_observed_mask=None):
    """Rasterize a photo-measured domain on its unchanged complete review canvas.

    reference contains polygon_ac28 and explicit excluded_polygons_ac28 by name.
    Absence of fixture evidence is a hard error; an empty exclusion list must
    carry fixtures_verified_none=True. No conservative sand mask is reused.
    """
    width, height = map(int, canvas_wh)
    x0, y0, native_width, native_height = map(int, content_box_xywh)
    if tuple(map(int, native_shape_hw)) != (native_height, native_width):
        raise ValueError("Native shape and canvas content box disagree")
    if x0 < 0 or y0 < 0 or x0 + native_width > width or y0 + native_height > height:
        raise ValueError("Native content is outside the unchanged review canvas")
    valid = np.asarray(feature_valid_mask)
    if valid.dtype != np.bool_ or valid.shape != tuple(native_shape_hw):
        raise ValueError("Feature validity must be a boolean mask on the exact native grid")
    exclusions = reference.get("excluded_polygons_ac28")
    if not isinstance(exclusions, Mapping) or (not exclusions and not reference.get("fixtures_verified_none")):
        raise ValueError("Rig geometry lacks explicit photograph-verified fixture exclusions")
    matrix = homography(H_ac28_to_frame_native)
    polygon = transform_points(reference["polygon_ac28"], matrix) + (x0, y0)
    if np.any(polygon.min(axis=0) < -.5) or np.any(polygon.max(axis=0) > np.array([width-.5, height-.5])):
        raise ValueError(f"Physical rig domain exceeds review canvas; required native bounds {polygon.min(axis=0)-(x0,y0)} to {polygon.max(axis=0)-(x0,y0)}")
    def raster(points):
        mask = np.zeros((height, width), np.uint8)
        # OpenCV's 1/256-pixel fixed-point coordinates preserve subpixel edges.
        cv2.fillPoly(mask, [np.rint(points * 256).astype(np.int32)], 1, shift=8)
        return mask.astype(bool)
    outer = raster(polygon)
    by_name = {str(name): raster(transform_points(points, matrix) + (x0, y0)) & outer
               for name, points in exclusions.items()}
    excluded = np.zeros_like(outer)
    for mask in by_name.values():
        excluded |= mask
    domain = outer & ~excluded
    if photographic_observed_mask is None:
        observed = np.zeros_like(domain)
        observed[y0:y0+native_height, x0:x0+native_width] = True
        observed_definition="native scientific content box only; caller has not supplied camera footprint"
    else:
        observed=np.asarray(photographic_observed_mask)
        if observed.dtype!=np.bool_ or observed.shape!=domain.shape:
            raise ValueError("Photographic observation mask must match the complete canvas")
        observed_definition="source-bound camera pixel footprint on unchanged review canvas"
    support = np.zeros_like(domain)
    support[y0:y0+native_height, x0:x0+native_width] = valid
    # Strict scientific RAW support is itself proof of observation even if
    # the simultaneously captured JPEG has a smaller footprint.
    observed = observed | support
    modelled = domain & support
    filled_observed = domain & observed & ~support
    filled_unobserved = domain & ~observed
    area = inverse_homography_area_map(matrix, (height, width), origin_xy=(-x0,-y0),
        fixed_area_mm2=reference.get("pixel_area_mm2_fixed_ac28", PIXEL_AREA_MM2_FIXED_AC28),
        H_ac28_to_metric=reference.get("H_ac28_to_metric"))
    parameters = dict(format=FORMAT, H_ac28_to_frame_native=matrix.tolist(),
        origin_xy=[-x0,-y0], canvas_wh=[width,height], content_box_xywh=[x0,y0,native_width,native_height],
        pixel_area_mm2_fixed_ac28=reference.get("pixel_area_mm2_fixed_ac28", PIXEL_AREA_MM2_FIXED_AC28),
        H_ac28_to_metric=reference.get("H_ac28_to_metric"),
        observed_definition=observed_definition,
        domain_area_cm2=float(area[domain].sum(dtype=np.float64)/100),
        area_cm2_by_class={name:float(area[mask].sum(dtype=np.float64)/100) for name,mask in
            (("modelled",modelled),("filled_observed",filled_observed),("filled_unobserved",filled_unobserved))},
        excluded_area_cm2={name:float(area[mask].sum(dtype=np.float64)/100) for name,mask in by_name.items()})
    return RigFrameContext(domain, modelled, filled_observed, filled_unobserved,
        excluded, by_name, (-x0,-y0), tuple(native_shape_hw), tuple(content_box_xywh), area, parameters)


def independent_rig_frame_context(reference_rig, H_reference_rig_to_frame_native,
                                  native_shape_hw, *, feature_valid_mask,
                                  canvas_wh, content_box_xywh, frame_photo_top_keep,
                                  frame_fixture_polygons_native,
                                  fixtures_verified_none=False,
                                  photographic_observed_mask=None,
                                  H_frame_native_to_metric_pixel=None):
    """Keep rig-plane walls, photo-sand surface and native hardware independent.

    A measured rig polygon is transported by the rigid-plane transform. The
    independently registered photographic top mask and run-specific hardware
    polygons are already on this frame's native/canvas geometry; sand-texture H
    is not reused to position the metal walls or native fixtures. Absolute
    scale eligibility remains an external source-bound certificate.
    """
    matrix=homography(H_reference_rig_to_frame_native)
    if not isinstance(frame_fixture_polygons_native,Mapping):
        raise ValueError("Native photograph-verified hardware polygons are required")
    reference=dict(polygon_ac28=reference_rig['polygon_reference_native'],
        excluded_polygons_ac28={name:transform_points(points,np.linalg.inv(matrix))
                               for name,points in frame_fixture_polygons_native.items()},
        fixtures_verified_none=bool(fixtures_verified_none),
        pixel_area_mm2_fixed_ac28=reference_rig.get('metric_pixel_area_mm2',PIXEL_AREA_MM2_FIXED_AC28))
    context=rig_frame_context(reference,matrix,native_shape_hw,
        feature_valid_mask=feature_valid_mask,canvas_wh=canvas_wh,
        content_box_xywh=content_box_xywh,photographic_observed_mask=photographic_observed_mask)
    top=np.asarray(frame_photo_top_keep)
    if top.dtype!=bool or top.shape!=context.rig_domain.shape:
        raise ValueError("Photographic sand top mask must be bool on the unchanged complete canvas")
    if not np.all(top[-1]) or np.any(top[:-1]&~top[1:]):
        raise ValueError("Photographic sand top mask must remove only a prefix above each column's surface")
    area=context.pixel_area_mm2
    if H_frame_native_to_metric_pixel is not None:
        area=rig_plane_area_map(H_frame_native_to_metric_pixel,area.shape,
            origin_xy=context.origin_xy,metric_pixel_area_mm2=reference['pixel_area_mm2_fixed_ac28'])
    domain=context.rig_domain&top
    modelled=context.modelled&top;observed=context.filled_observed&top;unobserved=context.filled_unobserved&top
    by_name={name:mask&top for name,mask in context.excluded_by_name.items()}
    excluded=context.hardware_excluded&top
    parameters=dict(context.parameters,
        format='ffac-independent-rig-frame-context-v1',
        H_reference_rig_to_frame_native=matrix.tolist(),
        H_frame_native_to_metric_pixel=None if H_frame_native_to_metric_pixel is None else homography(H_frame_native_to_metric_pixel).tolist(),
        reference_rig_identity=reference_rig.get('identity'),
        wall_transform_role='independent rigid-plane registration',
        top_geometry_role='frame-specific photograph-registered sand surface',
        fixture_geometry_role='explicit per-frame native photograph polygons',
        metric_pixel_area_mm2=reference['pixel_area_mm2_fixed_ac28'],
        pixel_area_mm2_fixed_ac28=PIXEL_AREA_MM2_FIXED_AC28,
        physical_scale_approved=False,
        domain_area_cm2=float(area[domain].sum(dtype=np.float64)/100),
        area_cm2_by_class={name:float(area[mask].sum(dtype=np.float64)/100) for name,mask in
            (('modelled',modelled),('filled_observed',observed),('filled_unobserved',unobserved))},
        excluded_area_cm2={name:float(area[mask].sum(dtype=np.float64)/100) for name,mask in by_name.items()})
    parameters.pop('H_ac28_to_frame_native',None);parameters.pop('H_ac28_to_metric',None)
    return replace(context,rig_domain=domain,modelled=modelled,
        filled_observed=observed,filled_unobserved=unobserved,hardware_excluded=excluded,
        excluded_by_name=by_name,pixel_area_mm2=area,parameters=parameters)


def fill_nearest_modelled(labels_native, context):
    """Extend labels only into declared filled classes, leaving hardware excluded.

    Distances are Euclidean on the exact frame pixel grid. The deterministic
    scipy tie rule is recorded by callers; no synthetic probabilities are made.
    """
    labels = np.asarray(labels_native)
    if labels.dtype != np.uint8 or labels.shape != context.native_shape_hw or np.any(~np.isin(labels,(0,1,2,255))):
        raise ValueError("Native labels must be uint8 water/aqueous/gas/unresolved")
    x0,y0,width,height = context.content_box_xywh
    result = np.full(context.rig_domain.shape,255,np.uint8)
    result[y0:y0+height,x0:x0+width] = labels
    modelled = context.modelled & (result != 255)
    filled = context.filled_observed | context.filled_unobserved
    if np.any(context.modelled & ~modelled):
        raise ValueError("Declared modelled pixels have unresolved labels; classify their invalid support explicitly before filling")
    if filled.any():
        if not modelled.any():
            raise ValueError("No modelled physical-domain pixel exists for nearest fill")
        indices = ndimage.distance_transform_edt(~modelled,return_distances=False,return_indices=True)
        result[filled] = result[indices[0,filled],indices[1,filled]]
    result[~context.rig_domain] = 255
    return result
