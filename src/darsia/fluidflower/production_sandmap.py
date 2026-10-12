"""Round-four sand geometry, separate from all original R3 calculations.

The caller supplies source-bound photo samples, registered coordinates, the
transported floor, measured CSV tables, and the ORIGINAL optical support.
This module reads no files, fits no calibration, and changes no phase labels.
Its loader hook is call-local and runs before native colour interpretation.
"""
from __future__ import annotations

from dataclasses import replace
from types import FunctionType

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree


UNQUANTIFIED_FACIES = (0, 3, 4, 6)


def _mask(value, shape, name):
    value = np.asarray(value)
    if value.dtype != bool or value.shape != shape:
        raise ValueError(name + " must be a boolean mask on the native grid")
    return value


def _points(value, name):
    value = np.asarray(value, np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError(name + " requires finite Nx2 pixel coordinates")
    return value


def boundary_offset_summary(map_xy, photo_xy, normals_xy):
    """Measured signed normal offsets; the fixed gate is median<=2,p95<=5px."""
    predicted, observed = _points(map_xy, "Map edges"), _points(photo_xy, "Photo edges")
    normals = _points(normals_xy, "Boundary normals")
    if predicted.shape != observed.shape or predicted.shape != normals.shape:
        raise ValueError("Every map point must have its own photo edge and normal")
    length = np.linalg.norm(normals, axis=1)
    if np.any(length <= 0):
        raise ValueError("Boundary normals must have nonzero length")
    offsets = np.sum((observed-predicted)*(normals/length[:, None]), axis=1)
    absolute = np.abs(offsets)
    median, p95 = float(np.median(absolute)), float(np.percentile(absolute, 95))
    return dict(samples=len(offsets), signed_offsets_px=offsets.tolist(),
                median_abs_offset_px=median, p95_abs_offset_px=p95,
                gate_passed=median <= 2. and p95 <= 5.)


def layer_boundary_samples(facies, sharp_region, *, stride=32, max_samples=2000):
    """Sample only positive-facies transitions in an explicitly selected region.

    The photographic sharp-region selection is supplied by the operator. The
    output columns are x,y,normal_x,normal_y, in the photo's registered grid.
    """
    labels = np.asarray(facies)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("A categorical facies map is required")
    region = _mask(sharp_region, labels.shape, "Sharp boundary region")
    if type(stride) is not int or stride < 1 or type(max_samples) is not int or max_samples < 1:
        raise ValueError("Boundary sampling must be explicitly bounded")
    positive = (labels > 0) & (labels != 255)
    horizontal = region[:, :-1] & region[:, 1:] & positive[:, :-1] & positive[:, 1:]
    horizontal &= labels[:, :-1] != labels[:, 1:]
    vertical = region[:-1] & region[1:] & positive[:-1] & positive[1:]
    vertical &= labels[:-1] != labels[1:]
    rows = []
    for selected, normal in ((horizontal, (1., 0.)), (vertical, (0., 1.))):
        yy, xx = np.nonzero(selected)
        xx, yy = xx[::stride], yy[::stride]
        if normal[0]: xx = xx + .5
        else: yy = yy + .5
        rows.extend(zip(xx, yy, np.full(len(xx), normal[0]), np.full(len(xx), normal[1])))
    result = np.asarray(rows, np.float64).reshape(-1, 4)
    if len(result) > max_samples:
        result = result[np.linspace(0, len(result)-1, max_samples).astype(int)]
    return result


def measure_layer_edges(photo_rgb, samples, *, search_radius_px=32, min_contrast):
    """Locate sharp photo edges on short normal profiles, without image mutation.

    RGB must be the registered existing photo, not a predicted phase field.
    Contrast is the norm of the difference of adjacent three-pixel means.
    All contrasts and rejected samples are exposed; no missing edge is a PASS.
    """
    image, samples = np.asarray(photo_rgb), np.asarray(samples, np.float64)
    if (image.ndim != 3 or image.shape[2] != 3 or samples.ndim != 2 or samples.shape[1] != 4
            or not np.isfinite(samples).all() or type(search_radius_px) is not int
            or search_radius_px < 1 or not np.isfinite(min_contrast) or min_contrast <= 0):
        raise ValueError("Registered RGB, finite boundary samples and explicit positive contrast are required")
    normal_length = np.linalg.norm(samples[:, 2:], axis=1)
    if np.any(normal_length <= 0):
        raise ValueError("Boundary normal is undefined")
    normals = samples[:, 2:] / normal_length[:, None]
    distances = np.arange(-search_radius_px-2, search_radius_px+3, dtype=np.float64)
    coordinates = samples[:, :2, None] + normals[:, :, None]*distances
    profiles = np.stack([map_coordinates(image[..., channel],
        (coordinates[:, 1], coordinates[:, 0]), output=np.float64, order=1,
        mode="constant", cval=np.nan, prefilter=False) for channel in range(3)], axis=-1)
    means = (profiles[:, :-2] + profiles[:, 1:-1] + profiles[:, 2:])/3.
    gradients = np.linalg.norm(means[:, 2:]-means[:, :-2], axis=-1)
    offsets = np.arange(-search_radius_px, search_radius_px+1, dtype=np.float64)
    # Select the closest peak, then centre its contiguous equal-height plateau.
    # Selecting its near end would systematically understate a sharp step offset.
    order = np.lexsort((offsets, np.abs(offsets)))
    reordered = np.where(np.isfinite(gradients[:, order]), gradients[:, order], -np.inf)
    index = order[np.argmax(reordered, axis=1)] if len(samples) else np.empty(0, int)
    contrast = gradients[np.arange(len(samples)), index]
    accepted = np.isfinite(contrast) & (contrast >= float(min_contrast))
    selected_offsets = offsets[index].copy()
    for row, centre in enumerate(index):
        if not np.isfinite(contrast[row]): continue
        left = right = int(centre)
        while left > 0 and gradients[row, left-1] == contrast[row]: left -= 1
        while right+1 < len(offsets) and gradients[row, right+1] == contrast[row]: right += 1
        selected_offsets[row] = (offsets[left]+offsets[right])/2.
    photo_xy = samples[:, :2] + normals*selected_offsets[:, None]
    return dict(map_xy=samples[accepted, :2], photo_xy=photo_xy[accepted], normals_xy=normals[accepted],
                accepted=accepted, contrast=contrast, rejected_count=int((~accepted).sum()),
                method="registered RGB three-pixel-mean gradient along selected sharp facies boundaries")


def fit_shift_scale(map_xy, photo_xy):
    """One positive isotropic scale and x/y shift; no rotation, shear or warp."""
    predicted, observed = _points(map_xy, "Map edges"), _points(photo_xy, "Photo edges")
    if predicted.shape != observed.shape or len(predicted) < 3:
        raise ValueError("At least three matched AC28 boundary points are required")
    design = np.zeros((2*len(predicted), 3), np.float64)
    design[0::2, 0], design[1::2, 0] = predicted[:, 0], predicted[:, 1]
    design[0::2, 1], design[1::2, 2] = 1., 1.
    parameters, _, rank, _ = np.linalg.lstsq(design, observed.reshape(-1), rcond=None)
    scale, tx, ty = parameters
    if rank != 3 or not np.isfinite(parameters).all() or scale <= 0:
        raise ValueError("Boundary points do not identify one positive shift/scale")
    return np.asarray([[scale, 0., tx], [0., scale, ty], [0., 0., 1.]], np.float64)


def transform_points(points_xy, matrix):
    points = _points(points_xy, "Boundary points")
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("A finite source-bound coordinate transform is required")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1.e-12):
        raise ValueError("Coordinate transform reaches infinity")
    return homogeneous[:, :2]/homogeneous[:, 2:]


def make_physical_transport(original_transport, correction, *, cv2_module):
    """Composite AC28 shift/scale with the existing categorical registration.

    Original transported domain/values remain available; correct_prepared then
    recomputes measured physical values on the phase/source-bound frame grid.
    """
    matrix = np.asarray(correction, np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0
            or matrix[0, 0] != matrix[1, 1] or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or not np.array_equal(matrix[2], (0., 0., 1.))):
        raise ValueError("Only the measured positive AC28 shift/scale is allowed")
    def transport(reference, original_matrix, shape_hw):
        result = original_transport(reference, original_matrix, shape_hw)
        if np.array_equal(matrix, np.eye(3)): return result
        result = dict(result)
        result["facies"] = cv2_module.warpPerspective(np.asarray(reference["facies"], np.uint8),
            np.asarray(original_matrix, np.float64) @ matrix, tuple(map(int, shape_hw))[::-1],
            flags=cv2_module.INTER_NEAREST, borderMode=cv2_module.BORDER_CONSTANT, borderValue=0)
        return result
    return transport


def choose_alignment(ac28_measurement, group_measurements, *, elapsed_seconds, expected_camera_groups):
    """Use an AC28 fit only when needed and all measured groups meet the gate.

    Each row has map_xy, photo_xy, normals_xy and H_ac28_to_frame. Actual photo
    measurements, including each camera group, are mandatory for acceptance.
    The time box and any missing/failed measurements retain the current map.
    """
    rows = [ac28_measurement, *group_measurements]
    identity = np.eye(3, dtype=np.float64)
    report = dict(transform_applied=False, matrix=identity.tolist(), before=[], after=[], reason=None)
    expected = tuple(expected_camera_groups)
    if not expected or len(set(expected)) != len(expected) or any(not isinstance(group, str) or not group for group in expected):
        raise ValueError("Explicit unique camera groups from the original metadata are required")
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("Actual nonnegative boundary-measurement time is required")
    if elapsed_seconds >= 3600:
        return identity, dict(report, reason="one-hour boundary time box exhausted; current map retained")
    if ac28_measurement.get("run") != "ac28" or not group_measurements:
        return identity, dict(report, reason="AC28 and camera-group photo measurements unavailable")
    try:
        for row in rows:
            report["before"].append(dict(run=row["run"], camera_group=row["camera_group"],
                **boundary_offset_summary(row["map_xy"], row["photo_xy"], row["normals_xy"])))
        groups = {row["camera_group"] for row in rows}
        if groups != set(expected):
            return identity, dict(report, reason="every declared original camera group was not measured")
        if all(row["gate_passed"] for row in report["before"]):
            return identity, dict(report, reason="current map meets the fixed 2/5px gate")
        inverse = np.linalg.inv(np.asarray(ac28_measurement["H_ac28_to_frame"], np.float64))
        correction = fit_shift_scale(transform_points(ac28_measurement["map_xy"], inverse),
                                     transform_points(ac28_measurement["photo_xy"], inverse))
        for row in rows:
            h = np.asarray(row["H_ac28_to_frame"], np.float64)
            predicted = transform_points(row["map_xy"], h @ correction @ np.linalg.inv(h))
            report["after"].append(dict(run=row["run"], camera_group=row["camera_group"],
                **boundary_offset_summary(predicted, row["photo_xy"], row["normals_xy"])))
        if not all(row["gate_passed"] for row in report["after"]):
            return identity, dict(report, reason="single AC28 shift/scale failed the fixed gate; current map retained")
    except (ValueError, np.linalg.LinAlgError) as error:
        return identity, dict(report, reason="boundary measurement/fit unavailable: " + str(error))
    report.update(transform_applied=True, matrix=correction.tolist(), reason="single AC28 shift/scale meets every measured gate")
    return correction, report


def fill_to_edges(facies, rig_domain, *, sand_labels):
    """Fill facies0 from ORIGINAL sand: nearest same row, column, then global.

    Equal row distances choose left; equal column distances choose above. The
    global Euclidean nearest-pixel tie follows SciPy's deterministic EDT. Known
    facies, including D/E/G, and all pixels outside the rig remain unchanged.
    """
    labels = np.asarray(facies)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Facies must be nonnegative native categorical values")
    rig = _mask(rig_domain, labels.shape, "Rig domain")
    allowed = tuple(map(int, sand_labels))
    if not allowed or any(label <= 0 or label == 255 for label in allowed):
        raise ValueError("Measured positive sand IDs are required")
    sand = rig & np.isin(labels, allowed)
    pending = rig & (labels == 0)
    result, method = labels.copy(), np.zeros(labels.shape, np.uint8)
    if pending.any() and not sand.any():
        raise ValueError("No measured sand pixel exists for the prescribed fill")
    for row in np.flatnonzero(pending.any(axis=1)):
        donors, targets = np.flatnonzero(sand[row]), np.flatnonzero(pending[row])
        if not len(donors): continue
        right = np.searchsorted(donors, targets)
        a, b = donors[np.clip(right-1, 0, len(donors)-1)], donors[np.clip(right, 0, len(donors)-1)]
        selected = np.where(np.abs(targets-a) <= np.abs(b-targets), a, b)
        result[row, targets], method[row, targets], pending[row, targets] = labels[row, selected], 1, False
    for column in np.flatnonzero(pending.any(axis=0)):
        donors, targets = np.flatnonzero(sand[:, column]), np.flatnonzero(pending[:, column])
        if not len(donors): continue
        below = np.searchsorted(donors, targets)
        a, b = donors[np.clip(below-1, 0, len(donors)-1)], donors[np.clip(below, 0, len(donors)-1)]
        selected = np.where(np.abs(targets-a) <= np.abs(b-targets), a, b)
        result[targets, column], method[targets, column], pending[targets, column] = labels[selected, column], 2, False
    if pending.any():
        yy, xx = distance_transform_edt(~sand, return_distances=False, return_indices=True)
        result[pending], method[pending] = labels[yy[pending], xx[pending]], 3
    return result, method


def measured_geometry(x_m, y_m, facies, depth_points, *, porosity_by_facies, swi_by_facies,
                      chunk_pixels=250000):
    """Measured phi/Swi and linear depth inside the hull, nearest outside."""
    x, y = np.asarray(x_m, np.float64), np.asarray(y_m, np.float64)
    labels, points = np.asarray(facies), np.asarray(depth_points, np.float64)
    if (x.ndim != 2 or x.shape != y.shape or labels.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or points.ndim != 2 or points.shape[1] != 3 or len(points) < 3
            or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)
            or type(chunk_pixels) is not int or chunk_pixels < 1):
        raise ValueError("Registered metre coordinates and positive measured depth triples are required")
    linear, nearest = LinearNDInterpolator(points[:, :2], points[:, 2]), cKDTree(points[:, :2])
    depth, outside = np.empty(x.size, np.float64), np.zeros(x.size, bool)
    for start in range(0, x.size, chunk_pixels):
        stop = min(x.size, start+chunk_pixels)
        query = np.column_stack((x.ravel()[start:stop], y.ravel()[start:stop]))
        values = np.asarray(linear(query), np.float64)
        bad = ~np.isfinite(values)
        if bad.any(): values[bad] = points[nearest.query(query[bad])[1], 2]
        depth[start:stop], outside[start:stop] = values, bad
    phi, swi = np.full(x.shape, np.nan, np.float64), np.full(x.shape, np.nan, np.float64)
    for table, output, name in ((porosity_by_facies, phi, "porosity"), (swi_by_facies, swi, "Swi")):
        for label, value in table.items():
            if int(label) == 0: continue
            value = float(value)
            if not np.isfinite(value) or not 0 <= value <= 1 or name == "porosity" and value == 0:
                raise ValueError("Measured " + name + " is outside its physical range")
            output[labels == int(label)] = value
    depth = depth.reshape(x.shape)
    known = np.isfinite(phi) & (phi > 0) & np.isfinite(depth) & (depth > 0)
    return dict(depth_m=depth, porosity=phi, swi=swi, known=known,
                depth_nearest_outside_hull=outside.reshape(x.shape))


def correct_prepared(prepared, *, x_m, y_m, floor_curve_canvas_y, depth_points,
                     porosity_by_facies, swi_by_facies, optical_support,
                     transported_facies=None, source_binding=None):
    """Round4 physical frame input, before ANY native optics or fit summary."""
    shape = prepared.domain.shape
    rig = _mask(prepared.domain, shape, "Original rig domain")
    optical = _mask(optical_support, shape, "Original light/colour/clearwater support")
    floor = np.asarray(floor_curve_canvas_y, np.float64)
    if floor.shape != (shape[1],) or not np.isfinite(floor).all():
        raise ValueError("The original transported sand-floor line is required")
    below = rig & (np.arange(shape[0])[:, None] > floor[None, :])
    if np.any(below & np.isin(prepared.phase_labels, (1, 2))):
        raise ValueError("Positive locked phase labels exist below the bound floor; cannot discard them")
    domain = rig & ~below
    labels = prepared.facies if transported_facies is None else transported_facies
    corrected, fill = fill_to_edges(labels, domain, sand_labels=porosity_by_facies)
    physical = measured_geometry(x_m, y_m, corrected, depth_points,
        porosity_by_facies=porosity_by_facies, swi_by_facies=swi_by_facies)
    known = domain & physical["known"]
    supported = known & optical
    eligible = supported.copy()
    # Former bottom additions above the floor are admitted only with the SAME
    # existing optical support. Remaining old excluded classes keep their names.
    bottom = prepared.bottom_strip_mask & domain & ~eligible
    below_photo = prepared.below_photo_filled_mask & domain & ~eligible & ~bottom
    old_main = prepared.mass_eligible_mask & prepared.supported_mask
    # Report the actual quantifiable main partition; the original D/E/G masks
    # and numerical eligibility fields below retain their treatment.
    quantifiable = (~np.isin(corrected, UNQUANTIFIED_FACIES)
                    | (prepared.phase_labels == 2) & np.isfinite(physical["swi"]))
    current_main = eligible & quantifiable
    previous_finite_swi = np.isin(prepared.facies,
        [int(label) for label,value in swi_by_facies.items() if np.isfinite(value)])
    previous_main = old_main & (~prepared.unquantified_facies_mask
        | (prepared.phase_labels == 2) & ~prepared.unknown_physical_mask & previous_finite_swi)
    added = current_main & ~previous_main
    filled = fill != 0
    blocked = domain & filled & ~optical
    area = np.broadcast_to(np.asarray(prepared.pixel_area_m2, np.float64), shape)
    if not np.isfinite(area[rig]).all() or np.any(area[rig] <= 0):
        raise ValueError("Original measured per-frame pixel area is required")
    co2 = np.isin(prepared.phase_labels, (1, 2))
    def areas(mask):
        return dict(area_cm2=float(np.sum(area[mask]))*1.e4,
                    co2_area_cm2=float(np.sum(area[mask & co2]))*1.e4)
    report = dict(contract="ffac-round4-corrected-sandmap-v1", source_binding=dict(source_binding or {}),
        run=prepared.run, frame_index=prepared.frame_index, elapsed_h=prepared.elapsed_h,
        added_to_M=areas(added), filled_without_optical_support=areas(blocked),
        below_sand_floor=areas(below), removed_from_M=areas(previous_main & ~current_main),
        filled_pixels_by_method={name:int(np.count_nonzero(fill == code))
                                for name,code in (("same_row",1),("same_column",2),("global_nearest",3))},
        depth_nearest_outside_hull_pixels=int(np.count_nonzero(domain & physical["depth_nearest_outside_hull"])),
        phase_labels_changed=False, RAW_changed=False, D_E_G_treatment_changed=False)
    provenance = dict(prepared.provenance, original_physical_provenance=dict(prepared.provenance),
        round4_sandmap=report, undefined_facies_not_borrowed=False,
        facies_zero_fill_rule="same row, same column, global nearest sand",
        physical_unknown_pixels=int(np.count_nonzero(domain & ~known)),
        supported_pixels=int(np.count_nonzero(supported)), mass_eligible_pixels=int(np.count_nonzero(eligible)),
        quantifiable_main_pixels=int(np.count_nonzero(current_main)),
        bottom_mass_allowed_above_sand_floor_with_existing_optical_support=True,
        optical_unresolved_pixels=int(np.count_nonzero(domain & ~optical)))
    value = replace(prepared, facies=corrected, domain=domain, supported_mask=supported,
        mass_eligible_mask=eligible, depth_m=physical["depth_m"], porosity=physical["porosity"],
        unknown_physical_mask=domain & ~known, bottom_strip_mask=bottom,
        below_photo_filled_mask=below_photo,
        unquantified_facies_mask=domain & np.isin(corrected, UNQUANTIFIED_FACIES), provenance=provenance)
    return value, report, physical["swi"]


def make_to_native_frame(original_to_native_frame, geometry_provider, *, on_corrected=None):
    """Callable selected-loader hook; also serves original calibration summaries.

    geometry_provider(prepared) returns the explicit correct_prepared kwargs.
    The original constructor validates the corrected dataclass. All raw arrays,
    phase labels, area, dose, recipe and atmospheric metadata retain their objects.
    """
    def to_native_frame(prepared, **kwargs):
        corrected, report, swi = correct_prepared(prepared, **geometry_provider(prepared))
        if on_corrected is not None: on_corrected(corrected, report, swi)
        return original_to_native_frame(corrected, **kwargs)
    return to_native_frame


class _SourceView:
    def __init__(self, original, **overrides):
        self._original, self._overrides = original, overrides

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._original, name)


def make_selected_loader(original_load_selected, source_module=None, to_native_frame=None,
                         *, source_proxy=None, physical_transport=None, prepare_response=None,
                         numpy_proxy=None, cv2_proxy=None):
    """Original load_selected code and guards with LOCAL physical IO callbacks.

    Default original metadata admission, recipe, photometry, clearwater and
    phase loading all remain in force. Optional preparation belongs to the
    round4 calibration orchestrator; this factory grants no freeze/PASS bypass.
    """
    if source_proxy is not None:
        if source_module is not None:
            raise ValueError("Supply one explicit selected source view")
        source_module = source_proxy
    if source_module is None:
        raise ValueError("The original source module or explicit source view is required")
    overrides = {} if to_native_frame is None else dict(to_native_frame=to_native_frame)
    if physical_transport is not None: overrides["transport_physical_reference"] = physical_transport
    namespace = dict(original_load_selected.__globals__, source=_SourceView(source_module, **overrides))
    original_replace = namespace["replace"]
    def replace_frame(frame, **changes):
        # The original adapter follows to_native_frame with its old exclusion
        # masks. Preserve the corrected ones for this ONE local expression.
        marker = getattr(frame, "provenance", {}).get("round4_sandmap", {})
        if marker.get("contract") == "ffac-round4-corrected-sandmap-v1" and set(changes) == {"excluded_masks"}:
            changes = dict(excluded_masks=frame.excluded_masks)
        return original_replace(frame, **changes)
    namespace["replace"] = replace_frame
    if numpy_proxy is not None: namespace["np"] = numpy_proxy
    if cv2_proxy is not None: namespace["cv2"] = cv2_proxy
    if prepare_response is not None: namespace["_prepare_response"] = prepare_response
    loader = FunctionType(original_load_selected.__code__, namespace, original_load_selected.__name__,
                          original_load_selected.__defaults__, original_load_selected.__closure__)
    loader.__kwdefaults__ = original_load_selected.__kwdefaults__
    return loader
