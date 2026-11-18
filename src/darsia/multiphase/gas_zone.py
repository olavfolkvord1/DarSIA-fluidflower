"""A causal growing gas contour and observed-signal paths to its injectors."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.graph import MCP_Geometric

import darsia
from darsia.multiphase.gas_parameters import gas_parameters
from darsia.multiphase.phase_rules import PhaseLabel, _co2, _pitch_mm, _seal


@dataclass(frozen=True)
class GasLink:
    """An ordered porous path and its smooth influence on allocation."""

    path: np.ndarray
    blend: np.ndarray
    diagnostics: dict
    path_coordinates: np.ndarray


@dataclass(frozen=True)
class GasZone:
    """One immutable frame contour shared by volume-allocation variants."""

    contour: np.ndarray
    links: dict
    diagnostics: dict


def _gaussian(values, sigma_mm, pitch_mm):
    """Gaussian mean in physical units, with zero outside the measured grid."""
    dtype = np.asarray(values).dtype
    dtype = dtype if dtype.kind == "f" else np.dtype(np.float32)
    if sigma_mm == 0:
        return np.asarray(values, dtype).copy()
    return cv2.GaussianBlur(
        np.asarray(values, dtype),
        (0, 0),
        sigmaX=sigma_mm / pitch_mm[1],
        sigmaY=sigma_mm / pitch_mm[0],
        borderType=cv2.BORDER_CONSTANT,
    )


def _rounded(mask, pitch_mm, closing_mm, dilation_mm, rounding_mm):
    """Closing and rounding on a zero-padded grid; no artificial border wall."""
    if not mask.any():
        return np.zeros_like(mask)
    reach = closing_mm + dilation_mm + 4 * rounding_mm
    padding = tuple(int(np.ceil(reach / step)) + 2 for step in pitch_mm)
    padded = np.pad(mask, tuple((p, p) for p in padding))
    if closing_mm:
        dilated = (
            ndimage.distance_transform_edt(~padded, sampling=pitch_mm) <= closing_mm
        )
        closed = ndimage.distance_transform_edt(dilated, sampling=pitch_mm) > closing_mm
        del dilated
    else:
        closed = padded
    if dilation_mm:
        closed = (
            ndimage.distance_transform_edt(~closed, sampling=pitch_mm) <= dilation_mm
        )
    if rounding_mm:
        closed = _gaussian(closed, rounding_mm, pitch_mm) >= 0.5
    rows, cols = padding
    return closed[rows:-rows, cols:-cols].copy()


def _smooth_envelope(mask, path, pitch_mm, settings):
    """Smooth outlying gas into the envelope before enforcing containment."""
    margin, rounding = settings.contour_smoothing_mm
    closed = _rounded(mask, pitch_mm, settings.closing_length_mm, margin, rounding)
    outside = mask & ~closed
    if outside.any():
        dilation, closing, rounding = settings.contour_blend_mm
        blended = closed | (
            ndimage.distance_transform_edt(~outside, sampling=pitch_mm) <= dilation
        )
        closed = _rounded(blended, pitch_mm, closing, 0, rounding)
    return ndimage.binary_fill_holes(closed | mask) & path


def _raster_path(points, shape):
    line = np.zeros(shape, np.uint8)
    vertices = np.rint(points[:, ::-1]).astype(np.int32).reshape(-1, 1, 2)
    if len(vertices) == 1:
        line[tuple(vertices[0, 0, ::-1])] = 1
        return line.astype(bool)
    cv2.polylines(line, [vertices], False, 1, 1)
    return line.astype(bool)


def _path_coordinates(points):
    """List the existing OpenCV raster in polyline order, preserving both ends."""
    vertices = np.rint(points).astype(np.int32)
    lengths = np.max(np.abs(np.diff(vertices, axis=0)), axis=1)
    coordinates = np.empty((1 + int(lengths.sum()), 2), np.int32)
    coordinates[0] = vertices[0]
    written = 1
    for start, finish, length in zip(vertices[:-1], vertices[1:], lengths, strict=True):
        if length == 0:
            continue
        origin = np.minimum(start, finish)
        extent = np.abs(finish - start) + 1
        raster = np.zeros(tuple(extent), np.uint8)
        a, b = start - origin, finish - origin
        cv2.line(raster, tuple(a[::-1]), tuple(b[::-1]), 1, 1)
        rows, cols = np.nonzero(raster)
        order = np.argsort((rows - a[0]) ** 2 + (cols - a[1]) ** 2)
        ordered = np.column_stack((rows[order], cols[order])) + origin
        coordinates[written : written + length] = ordered[1:]
        written += int(length)
    assert written == len(coordinates)
    return coordinates


def _smooth_path(points, support, pitch_mm, sigma_mm):
    """Round a path in arc length, preserving its ends and porous support.

    If rounding would leave permitted support, decrease its displacement toward
    the original path until the raster stays in support. The original
    minimum-cost path remains the limiting case.
    """
    if len(points) < 3 or sigma_mm == 0:
        return _raster_path(points, support.shape), 0.0, _path_coordinates(points)
    lengths = np.linalg.norm(np.diff(points, axis=0) * pitch_mm, axis=1)
    along = np.r_[0.0, np.cumsum(lengths)]
    sample = np.linspace(
        0, along[-1], max(2, int(np.ceil(along[-1] / min(pitch_mm))) + 1)
    )
    original = np.column_stack([np.interp(sample, along, points[:, j]) for j in (0, 1)])
    sigma = sigma_mm / (sample[1] - sample[0])
    rounded = ndimage.gaussian_filter1d(original, sigma, axis=0, mode="nearest")
    rounded[0], rounded[-1] = original[0], original[-1]
    fraction = 1.0
    previous = None
    while True:
        candidate = original + fraction * (rounded - original)
        line = _raster_path(candidate, support.shape)
        if not (line & ~support).any():
            return line, fraction, _path_coordinates(candidate)
        if previous is not None and np.array_equal(line, previous):
            return _raster_path(points, support.shape), 0.0, _path_coordinates(points)
        previous = line
        fraction *= 0.5


def _injector_link(
    port,
    contour,
    observed,
    probability,
    path,
    x,
    y,
    pitch_mm,
    settings,
    *,
    current_co2,
    depth_m,
):
    """Join the nearest current CO2 by a straight hidden segment, then seen CO2.

    The hidden segment cannot detour around excluded pixels. Beyond its current
    CO2 meeting point, routing and smoothing stay in observed prefix CO2. The
    median smoothed probability on that support sets the cost reference; a
    zero denominator remains impassable rather than receiving an offset. The
    nearest meeting cannot lie below the injector by more than its local depth.
    """
    squared = (x - port[0]) ** 2 + (y - port[1]) ** 2
    start = np.unravel_index(np.argmin(squared), squared.shape)
    note = {"port_pixel": [int(value) for value in start]}
    if contour[start] and current_co2[start]:
        return None, None, {**note, "status": "already_inside"}
    if not path[start]:
        return None, None, {**note, "status": "no_porosity_at_port"}
    targets = current_co2 & contour & path
    if not targets.any():
        return None, None, {**note, "status": "no_current_co2_target"}
    squared[~(current_co2 & path)] = np.inf
    meeting = np.unravel_index(np.argmin(squared), squared.shape)
    del squared
    note["meeting_pixel"] = [int(value) for value in meeting]
    depth = np.asarray(depth_m)
    if depth.shape != contour.shape:
        raise ValueError("The depth map must share the injector link's grid")
    local_depth_m = float(depth[start])
    if not np.isfinite(local_depth_m) or local_depth_m <= 0:
        raise ValueError(
            f"Injector {port} requires finite positive local rig depth "
            f"at pixel {start}; received {local_depth_m}"
        )
    drop_m = float(port[1]) - float(y[meeting])
    if not np.isfinite(drop_m):
        raise ValueError("The injector and meeting need finite physical heights")
    note["local_depth_mm"] = 1000 * local_depth_m
    note["downward_drop_mm"] = 1000 * drop_m
    if drop_m > local_depth_m:
        return None, None, {**note, "status": "meeting_below_local_depth"}
    hidden_coordinates = _path_coordinates(np.asarray([start, meeting]))
    if not path[tuple(hidden_coordinates.T)].all():
        return None, None, {**note, "status": "straight_hidden_segment_blocked"}
    seen = observed & path
    if not seen[meeting]:
        return None, None, {**note, "status": "no_observed_co2_path"}
    signal = _gaussian(probability, settings.link_smoothing_mm[0], pitch_mm)
    p_ref = float(np.median(signal[seen]))
    note["p_ref"] = p_ref if np.isfinite(p_ref) else None
    if not np.isfinite(p_ref) or p_ref < 0:
        return None, None, {**note, "status": "invalid_probability_reference"}
    count, components = cv2.connectedComponents(seen.astype(np.uint8), connectivity=8)
    connected = components == components[meeting]
    del components, count
    targets &= connected
    if not targets.any():
        return None, None, {**note, "status": "no_observed_co2_path"}
    rows, cols = np.nonzero(connected)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    del rows, cols
    local = connected[box]
    signal = signal[box]
    cost = np.full(local.shape, np.inf)
    denominator = signal + p_ref
    np.divide(
        1.0,
        denominator,
        out=cost,
        where=local & np.isfinite(denominator) & (denominator > 0),
    )
    origin = (meeting[0] - box[0].start, meeting[1] - box[1].start)
    if not np.isfinite(cost[origin]):
        return None, None, {**note, "status": "no_finite_path"}
    solver = MCP_Geometric(cost, fully_connected=True, sampling=pitch_mm)
    cumulative, _ = solver.find_costs(starts=[origin])
    possible = np.where(targets[box], cumulative, np.inf)
    finish = np.unravel_index(np.argmin(possible), possible.shape)
    if not np.isfinite(possible[finish]):
        return None, None, {**note, "status": "no_finite_path"}
    points = np.asarray(solver.traceback(finish), np.float64)
    line, fraction, seen_coordinates = _smooth_path(
        points, local, pitch_mm, settings.link_smoothing_mm[1]
    )
    placed = np.zeros(contour.shape, bool)
    placed[box] = line
    placed[tuple(hidden_coordinates.T)] = True
    seen_coordinates += np.array((box[0].start, box[1].start), np.int32)
    coordinates = np.concatenate((hidden_coordinates[:-1], seen_coordinates))
    pixels = tuple(coordinates.T)
    distance = float(
        1000
        * np.hypot(np.diff(x[pixels]), np.diff(y[pixels])).sum()
    )
    diagnostics = {
        **note,
        "status": "linked",
        "length_mm": distance,
        "smoothed_fraction": fraction,
        "mean_probability": float(signal[line].mean()),
        "path_pixels": int(placed.sum()),
        "hidden_segment_pixels": len(hidden_coordinates),
        "outside_observed_co2_pixels": int((placed & ~observed).sum()),
        "target_pixel": coordinates[-1].tolist(),
    }
    return placed, coordinates, diagnostics


def _envelope(mask, path, pitch_mm, settings):
    """Crop empty margins with all metric operator reaches retained."""
    if not mask.any():
        return np.zeros_like(mask)
    margin, rounding = settings.contour_smoothing_mm
    outlier, closing, blending = settings.contour_blend_mm
    reach = (
        settings.closing_length_mm
        + margin
        + 4 * rounding
        + outlier
        + closing
        + 4 * blending
    )
    padding = np.ceil(reach / np.asarray(pitch_mm)).astype(int) + 2
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    box = tuple(
        slice(max(0, a[0] - pad), min(size, a[-1] + pad + 1))
        for a, pad, size in zip((rows, cols), padding, mask.shape)
    )
    result = np.zeros_like(mask)
    result[box] = _smooth_envelope(mask[box], path[box], pitch_mm, settings)
    return result


def _half_disc_dilation(source, radius_mm, pitch_mm):
    """Exact binary upward/sideways dilation, using cropped linear convolution.

    Integer convolution counts overlapping half-discs. Thresholding at one half
    separates zero from one despite floating-point FFT roundoff. The kernel is
    reflected relative to cv2's correlation-style dilation: its upper half is
    used here. Offsets larger than the whole grid cannot affect any target.
    """
    result = np.zeros_like(source)
    if not source.any() or radius_mm <= 0:
        return source.copy()
    reach = np.minimum(
        np.ceil(radius_mm / np.asarray(pitch_mm)).astype(int),
        np.asarray(source.shape) - 1,
    )
    rows, cols = np.flatnonzero(source.any(1)), np.flatnonzero(source.any(0))
    box = (
        slice(max(0, rows[0] - reach[0]), rows[-1] + 1),
        slice(
            max(0, cols[0] - reach[1]), min(source.shape[1], cols[-1] + reach[1] + 1)
        ),
    )
    rr = np.arange(-reach[0], reach[0] + 1)[:, None]
    cc = np.arange(-reach[1], reach[1] + 1)[None, :]
    kernel = (rr <= 0) & (
        (rr * pitch_mm[0]) ** 2 + (cc * pitch_mm[1]) ** 2 <= radius_mm**2
    )
    result[box] = (
        fftconvolve(
            source[box].astype(np.float32), kernel.astype(np.float32), mode="same"
        )
        > 0.5
    )
    return result


def _depth_margin(contour, depth_m, path, pitch_mm, *, previous=None):
    """Union upward half-discs, quantised at the finer measured grid pitch.

    Rounding ties go up; the error is at most half that pixel pitch.
    The numerical resolution follows the grid; no rig depth is assumed.
    Unknown or nonpositive depth at either pixel grants no additional support.
    Only the unexpanded contour is used, preventing repeated margin growth.
    For a growing contour on the same depth/path/pitch, ``previous`` holds
    its prior bare and expanded masks. Only new bare sources need dilation;
    diagnostics still describe the complete current bare contour.
    """
    depth = np.asarray(depth_m)
    quantum = float(min(pitch_mm))
    if depth.shape != contour.shape:
        raise ValueError("The depth map must share the gas contour's grid")
    valid = np.isfinite(depth) & (depth > 0)
    sources = contour & path & valid
    values = 1000 * depth[sources]
    diagnostics = {
        "rule": "quantised_source_half_discs; sideways_and_up; no_recursive_margin",
        "depth_quantum_mm": quantum,
        "source_depth_mm": (
            None
            if not values.size
            else {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
        ),
        "added_pixels": 0,
    }
    margin = contour.copy()
    if previous is not None:
        before, expanded = previous
        margin |= expanded
        sources &= ~before
        values = 1000 * depth[sources]
    if not sources.any():
        diagnostics["added_pixels"] = int((margin & ~contour).sum())
        return margin, diagnostics
    levels = np.zeros(contour.shape, np.int64)
    # Larger radii than the grid diagonal are geometrically indistinguishable;
    # capping before conversion also avoids integer overflow on extreme depths.
    farthest = np.hypot(*((np.asarray(contour.shape) - 1) * pitch_mm))
    last_level = int(np.ceil(farthest / quantum))
    levels[sources] = np.floor(np.minimum(values / quantum, last_level) + 0.5).astype(
        np.int64
    )
    for level in np.unique(levels[sources]):
        if level > 0:
            margin |= _half_disc_dilation(
                sources & (levels == level), level * quantum, pitch_mm
            )
    margin &= path & valid
    diagnostics["added_pixels"] = int((margin & ~contour).sum())
    return contour | margin, diagnostics


def gas_model_zones(
    frames,
    dates,
    sand_map,
    domain,
    permeability,
    protocol,
    rule_settings,
    settings,
    chart=None,
    *,
    depth_m,
):
    """Yield each frame's contour using model gas and injector evidence seen so far.

    Gas means GAS after the unchanged phase rules and before volume allocation.
    The contour cannot shrink: every new smoothed envelope is unioned with the
    previous contour. A link joins the nearest current CO2 by a straight hidden
    segment, refusing excluded pixels and a downward drop exceeding the local
    measured depth at the injector cell. It then follows the active injection's
    observed prefix CO2 to current CO2 inside the model contour, with costs set
    by smoothed maximum gas probability and its median on seen CO2. No segment
    may cross UNKNOWN pixels. Once injection ends, only newly observed model gas
    can enlarge the retained contour.

    Parameters
    ----------
    frames : iterable of GasModelFrame
        Complete ordered preallocation model states; consumed exactly once.
    dates : sequence of datetime
        Selected acquisition times, one per model state.
    sand_map : darsia.Image
        Facies labels on the physical grid.
    domain : array
        Pixels whose phase is decided.
    permeability : mapping
        Permeability by porous facies label.
    protocol : darsia.InjectionProtocol
        Injection periods and measured injector positions.
    rule_settings : PhaseRuleSettings
        Phase rules defining free water and seals.
    settings : GasAllocationSettings
        Explicit contour, path and smoothing lengths in millimetres.
    chart : array, optional
        Colour checker pixels excluded from all contour evidence.
    depth_m : array
        Local rig depth in metres on this grid. Its measured value at the
        nearest contour pixel sets the upward/sideways margin; missing depth
        grants no margin.

    Yields
    ------
    GasZone
        Immutable cumulative contour and the current injection's signal paths.
    """
    from darsia.multiphase.gas_allocation import injection_ports

    settings = gas_parameters(settings, depth_m)

    columns, rows = darsia.pixel_centres(sand_map)
    x, y = np.broadcast_arrays(columns[None, :], rows[:, None])
    pitch_mm = _pitch_mm(sand_map)
    ports = injection_ports(protocol)
    porous = domain & np.isin(
        sand_map.img,
        [k for k in permeability if k not in rule_settings.free_water_facies],
    )
    path = porous & ~_seal(sand_map.img, domain, permeability, rule_settings)
    if chart is not None:
        path &= ~chart
        porous &= ~chart
    gas = np.zeros(domain.shape, bool)
    seal_gas = np.zeros_like(gas)
    previous = np.zeros_like(gas)
    previous_bare = np.zeros_like(gas)
    previous_expanded = None
    base = np.zeros_like(gas)
    observed, maximum, periods = {}, {}, {}
    for index, (date, frame) in enumerate(zip(dates, frames, strict=True)):
        observed_gas = (frame.labels == PhaseLabel.GAS) & porous
        own_seal_gas = observed_gas & ~path
        seal_gas |= own_seal_gas
        own_gas = observed_gas & path
        new_gas = bool((own_gas & ~gas).any())
        gas |= own_gas
        if new_gas:
            base = _envelope(gas, path, pitch_mm, settings)
        active = {
            port: next(
                ((begin, end) for begin, end in intervals if begin <= date < end), None
            )
            for port, intervals in ports.items()
        }
        active = {port: period for port, period in active.items() if period is not None}
        for port in list(observed):
            if port not in active:
                del observed[port], maximum[port], periods[port]
        co2 = _co2(frame.labels) & path
        for port, period in active.items():
            if periods.get(port) != period:
                observed[port] = np.zeros_like(gas)
                maximum[port] = np.zeros_like(frame.gas_probability)
                periods[port] = period
            observed[port] |= co2
            np.maximum(
                maximum[port],
                np.where(
                    co2 & np.isfinite(frame.gas_probability), frame.gas_probability, 0
                ),
                out=maximum[port],
            )
        bodies, records, lines, routes = gas.copy(), {}, {}, {}
        route_path = path & (frame.labels != PhaseLabel.UNKNOWN)
        for port in active:
            line, coordinates, note = _injector_link(
                port,
                base,
                observed[port],
                maximum[port],
                route_path,
                x,
                y,
                pitch_mm,
                settings,
                current_co2=co2,
                depth_m=depth_m,
            )
            records[port] = note
            if line is not None:
                lines[port] = line
                routes[port] = coordinates
                bodies |= (
                    ndimage.distance_transform_edt(~line, sampling=pitch_mm)
                    <= settings.link_width_mm / 2
                ) & path
        rounded = (
            _envelope(bodies, path, pitch_mm, settings) if lines else base
        )
        bare = previous_bare | rounded
        expanded, margin_record = _depth_margin(
            bare,
            depth_m,
            path,
            pitch_mm,
            previous=(
                (previous_bare, previous_expanded)
                if previous_expanded is not None
                else None
            ),
        )
        previous_expanded = expanded
        # Seal gas is preserved only after every shape operation. It cannot
        # seed smoothing, an injector route or a measured-depth margin.
        contour = previous | expanded | seal_gas
        links = {}
        for port, line in lines.items():
            distance = ndimage.distance_transform_edt(
                ~line, sampling=pitch_mm
            )
            scale = settings.link_width_mm / 2 + settings.fade_length_mm
            t = np.clip(distance / scale, 0, 1)
            blend = ((1 - t) ** 2 * (1 + 2 * t)).astype(np.float32)
            blend[~contour] = 0
            line.flags.writeable = blend.flags.writeable = False
            routes[port].flags.writeable = False
            links[port] = GasLink(line, blend, records[port], routes[port])
        assert not (gas & ~contour).any()
        assert not (seal_gas & ~contour).any()
        contour.flags.writeable = False
        previous = contour
        previous_bare = bare
        yield GasZone(
            contour,
            links,
            {
                "index": index,
                "ports": records,
                "model_gas_pixels": int(gas.sum()) + int(seal_gas.sum()),
                "contour_pixels": int(contour.sum()),
                "model_gas_on_seals_pixels": int(own_seal_gas.sum()),
                "model_gas_seen_on_seals_pixels": int(seal_gas.sum()),
                "depth_margin": margin_record,
                "geometry": settings.geometry_record(),
            },
        )
