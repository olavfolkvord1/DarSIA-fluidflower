"""Read-only geometry adapters for the separately reported R3 estimates.

The R1/R2 frame loaders and their main-mass geometry are not modified. A bound
coordinate homography reproduces the saved AC28 physical grid and extends its
coordinates beyond the old crop; it does not extend a facies colour path.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def read_measured_properties(depth_path, facies_path):
    with Path(depth_path).open(encoding="utf-8-sig", newline="") as stream:
        points = np.asarray([[float(row[name]) for name in ("x", "y", "depth")]
                             for row in csv.DictReader(stream)], np.float64)
    if points.shape != (66, 3) or not np.isfinite(points).all():
        raise ValueError("R3 estimates require all 66 finite measured depth points")
    porosity, swi = {}, {}
    with Path(facies_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["id"])
            try:
                swi_name = next(name for name in row if name.replace(" ", "_").casefold() == "swi_co2")
                phi, water = float(row["porosity"]), float(row[swi_name])
            except ValueError:
                continue
            if np.isfinite(phi) and np.isfinite(water) and 0 < phi < 1 and 0 <= water <= 1:
                porosity[key], swi[key] = phi, water
    return points, porosity, swi


def scientific_raw_valid(descriptor, reader):
    """The exact decoder's measured validity, without decoding RGB a second time."""
    shape = tuple(descriptor["phase_shape_hw"])
    centre = descriptor["native_centre"]
    h, w = centre["native_shape_hw"]
    ox, oy = descriptor["native_origin_xy"]
    x, y = -int(ox), -int(oy)
    valid = np.zeros(shape, bool)
    values = reader.array(centre["log_linear_rgb"])
    support = reader.array(centre["valid_mask"])
    valid[y:y+h, x:x+w] = np.isfinite(values).all(axis=0) & support
    for window in descriptor["ring_windows"]:
        a, b, c, d = map(int, window["canvas_xyxy"])
        valid[b:d, a:c] = np.isfinite(reader.array(window["log_linear_rgb"])).all(axis=0)
    return valid


def measured_photo_support(frame, receipt, reader):
    """Reconstruct optical support independently of old physical support."""
    import cv2
    from .phase_ffac_common_mass_inputs import matrix_ac28_to_canvas
    current, baseline = receipt["full_extended_scientific"], receipt["baseline_extended_scientific"]
    current_valid = scientific_raw_valid(current, reader)
    baseline_key = (baseline["source_sha256"], tuple(baseline["phase_shape_hw"]),
                    tuple(baseline["native_origin_xy"]))
    cached = reader.raw_baselines.get(baseline_key)
    baseline_valid = cached[1] if cached is not None else scientific_raw_valid(baseline, reader)
    current_matrix = matrix_ac28_to_canvas(current["binding"]["geometry"], current["native_origin_xy"])
    baseline_matrix = matrix_ac28_to_canvas(baseline["binding"]["geometry"], baseline["native_origin_xy"])
    warp = current_matrix @ np.linalg.inv(baseline_matrix)
    coverage = cv2.warpPerspective(baseline_valid.astype(np.float32), warp,
        (frame.domain.shape[1], frame.domain.shape[0]), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (frame.domain & current_valid & (coverage >= 1-1e-6)
            & frame.clear_water_supported_mask
            & np.isfinite(frame.current_rgb).all(axis=2)
            & np.isfinite(frame.baseline_rgb).all(axis=2)
            & np.isfinite(frame.clear_water_rgb).all(axis=2)
            & (frame.current_rgb > 0).all(axis=2)
            & (frame.baseline_rgb > 0).all(axis=2)
            & (frame.clear_water_rgb > 0).all(axis=2))


def physical_homography(provenance, reference_shape_hw):
    """Exact saved legacy pixel centres -> AC28 -> ground-up metre chain."""
    if provenance.get("physical_grid_2d_exact") is not True:
        raise ValueError("Source must declare the projective 2D physical grid")
    h, w = map(int, reference_shape_hw)
    legacy_to_ac28 = np.asarray(provenance["H_legacy_to_reference"], np.float64)
    pixel_to_metre = np.asarray([[.9/w, 0, .9/(2*w)],
                                [0, -.49/h, .49-.49/(2*h)], [0, 0, 1.]])
    return pixel_to_metre @ np.linalg.inv(legacy_to_ac28)


def project_points(matrix, x, y):
    matrix = np.asarray(matrix, np.float64)
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    denominator = matrix[2, 0]*x + matrix[2, 1]*y + matrix[2, 2]
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Source-bound coordinate transform reaches infinity")
    return ((matrix[0, 0]*x+matrix[0, 1]*y+matrix[0, 2])/denominator,
            (matrix[1, 0]*x+matrix[1, 1]*y+matrix[1, 2])/denominator)


def validate_reference_coordinates(matrix, x_m, y_m, *, tolerance_m=6e-8):
    """Fixed distributed samples check the metadata chain against saved values."""
    x_m, y_m = np.asarray(x_m), np.asarray(y_m)
    if x_m.shape != y_m.shape or x_m.ndim != 2:
        raise ValueError("Physical reference needs two native coordinate arrays")
    ys, xs = np.meshgrid(np.linspace(0, x_m.shape[0]-1, 17).astype(int),
                         np.linspace(0, x_m.shape[1]-1, 23).astype(int), indexing="ij")
    x, y = project_points(matrix, xs, ys)
    if not (np.isfinite(x).all() and np.isfinite(y).all()
            and np.isfinite(x_m[ys, xs]).all() and np.isfinite(y_m[ys, xs]).all()):
        raise ValueError("The physical coordinate validation contains non-finite samples")
    maximum = float(max(np.max(np.abs(x-x_m[ys, xs])), np.max(np.abs(y-y_m[ys, xs]))))
    if maximum > tolerance_m:
        raise ValueError("The extrapolated coordinate chain does not reproduce the frozen physical grid")
    return dict(samples=int(xs.size), maximum_component_error_m=maximum,
                maximum_allowed_m=tolerance_m, source_grid_changed=False,
                coordinate_extension_role="R3 excluded estimates only; main M unchanged")


def frame_coordinates(reference_matrix, geometry, revision):
    shape = tuple(map(int, revision["phase_shape_hw"]))
    ox, oy = map(float, revision["native_origin_xy"])
    matrix = np.asarray(reference_matrix, np.float64) @ np.linalg.inv(
        np.asarray(geometry["H_ac28_to_frame_native"], np.float64))
    x = np.arange(shape[1], dtype=np.float64)[None, :] + ox
    y = np.arange(shape[0], dtype=np.float64)[:, None] + oy
    xm, ym = project_points(matrix, x, y)
    return xm.astype(np.float32), ym.astype(np.float32)


def approved_distance_scale_revision(revision):
    """Resolve the bound measured-width metadata without inventing a scale.

    Legacy receipts duplicate this document in parameters; selected receipts
    publish it at revision level. If both exist they must agree literally.
    This reader is used only for descriptive distance reports.
    """
    parameters = revision["parameters"]
    nested_present = "physical_scale_revision" in parameters
    top_present = "physical_scale_revision" in revision
    if not nested_present and not top_present:
        raise ValueError("Distances require a bound physical scale revision")
    if nested_present and top_present:
        if parameters["physical_scale_revision"] != revision["physical_scale_revision"]:
            raise ValueError("Conflicting physical scale revisions in distance metadata")
    approval = (parameters["physical_scale_revision"] if nested_present
                else revision["physical_scale_revision"])
    if not isinstance(approval, dict) or approval.get("physical_scale_approved") is not True:
        raise ValueError("Distances require the approved measured rig width")
    return approval


def metric_native_points(revision, xy):
    """Approved 900 mm metric chain, used for distances rather than M geometry."""
    parameters = revision["parameters"]
    approval = approved_distance_scale_revision(revision)
    if approval.get("physical_scale_approved") is not True:
        raise ValueError("Distances require the approved measured rig width")
    factor = float(parameters["metric_pixel_area_mm2"])**.5 * float(approval["linear_factor"])
    xy = np.asarray(xy, np.float64)
    x, y = project_points(parameters["H_frame_native_to_metric_pixel"], xy[:, 0], xy[:, 1])
    return np.column_stack((x, y))*factor


def polyline_distance(points, vertices):
    """Certified point-to-segment distance, including irregular long segments.

    A segment's distance is bounded below by its midpoint distance minus half
    its length. Adaptive nearest-midpoint queries stop only when this bound
    proves that no unexamined segment can improve the current answer. Long
    segments are subdivided collinearly for efficiency; the curve is unchanged.
    Undefined vertices do not create a new segment across the missing data.
    """
    points, vertices = np.asarray(points, np.float64), np.asarray(vertices, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("Polyline distances require two-dimensional coordinate pairs")
    if len(vertices) < 2 or len(points) == 0:
        return np.full(len(points), np.nan)
    start, end = vertices[:-1], vertices[1:]
    valid = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    valid &= np.linalg.norm(end-start, axis=1) > 1e-12
    start, end = start[valid], end[valid]
    if not len(start):
        return np.full(len(points), np.nan)
    lengths = np.linalg.norm(end-start, axis=1)
    cap = max(4*float(np.median(lengths)), float(lengths.sum())/50_000)
    counts = np.maximum(1, np.ceil(lengths/cap).astype(np.int64))
    indexes = np.repeat(np.arange(len(start)), counts)
    offsets = np.arange(int(counts.sum()))-np.repeat(np.r_[0, np.cumsum(counts)[:-1]], counts)
    fraction = offsets/counts[indexes]
    direction = end[indexes]-start[indexes]
    subdivided_start = start[indexes]+fraction[:, None]*direction
    subdivided_end = start[indexes]+(fraction+1/counts[indexes])[:, None]*direction
    start, end = subdivided_start, subdivided_end
    maximum_half_length = float(np.linalg.norm(end-start, axis=1).max()/2)
    tree = cKDTree((start+end)/2)
    answer = np.full(len(points), np.nan)
    finite_indexes = np.flatnonzero(np.isfinite(points).all(axis=1))
    for offset in range(0, len(finite_indexes), 8192):
        active = finite_indexes[offset:offset+8192]
        count = min(8, len(start))
        while len(active):
            block = points[active]
            midpoint_distance, indexes = tree.query(block, k=count, workers=1)
            if indexes.ndim == 1:
                indexes, midpoint_distance = indexes[:, None], midpoint_distance[:, None]
            a, b = start[indexes], end[indexes]
            direction = b-a
            amount = np.clip(np.sum((block[:, None]-a)*direction, axis=-1)/
                             np.sum(direction*direction, axis=-1), 0, 1)
            best = np.min(np.linalg.norm(block[:, None]-(a+amount[..., None]*direction), axis=-1), axis=1)
            if count == len(start):
                answer[active] = best
                break
            # Subtract a numerical cushion from the lower bound so rounding
            # cannot falsely certify a segment at equal distance.
            certified = midpoint_distance[:, -1]-maximum_half_length-1e-14 >= best
            answer[active[certified]] = best[certified]
            active = active[~certified]
            count = min(len(start), 2*count)
    return answer


def distances_for_part(mask, *, revision, geometry, floor_curve_canvas_y, sand_domain):
    """Return descriptive distances to the recorded floor, photo top and walls."""
    mask, domain = np.asarray(mask, bool), np.asarray(sand_domain, bool)
    if mask.shape != domain.shape:
        raise ValueError("Report masks must use the exact phase canvas")
    rows, columns = np.nonzero(mask)
    ox, oy = map(float, revision["native_origin_xy"])
    points = metric_native_points(revision, np.column_stack((columns+ox, rows+oy)))
    occupied = domain.any(axis=0)
    xs = np.flatnonzero(occupied)
    floor = metric_native_points(revision, np.column_stack((xs+ox,
        np.asarray(floor_curve_canvas_y)[xs]+oy)))
    top = metric_native_points(revision, np.column_stack((xs+ox, domain.argmax(axis=0)[xs]+oy)))
    polygon = np.asarray(geometry["frame_inner_polygon_native"], np.float64)
    steps = np.diff(polygon, axis=0)
    side = np.abs(steps[:, 1]) > np.abs(steps[:, 0])
    midpoint = (polygon[1:]+polygon[:-1])/2
    centre = float((polygon[:, 0].min()+polygon[:, 0].max())/2)
    walls = {}
    for name, select in (("left_wall", side & (midpoint[:, 0] < centre)),
                         ("right_wall", side & (midpoint[:, 0] >= centre))):
        indexes = np.flatnonzero(select)
        vertices = (polygon[np.unique(np.concatenate((indexes, indexes+1)))]
                    if len(indexes) else np.empty((0, 2)))
        walls[name] = metric_native_points(revision, vertices) if len(vertices) else vertices
    boundaries = dict(pooled_floor=floor, sand_top=top, **walls)
    output = {}
    for name, vertices in boundaries.items():
        distances = polyline_distance(points, vertices)
        finite = distances[np.isfinite(distances)]
        output[name] = dict(pixels=int(len(points)), known_pixels=int(len(finite)),
            min_mm=float(finite.min()) if len(finite) else None,
            median_mm=float(np.median(finite)) if len(finite) else None,
            p95_mm=float(np.quantile(finite, .95)) if len(finite) else None,
            max_mm=float(finite.max()) if len(finite) else None,
            metric_source="900 mm approved rigid-plane transform; recorded per-frame geometry")
    return output
