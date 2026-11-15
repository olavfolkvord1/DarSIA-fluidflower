"""Gas allocation: the free gas of a run's frames from a free-gas volume per frame.

The phase maps (:func:`darsia.phase_maps`) say where CO2 is and where the phase
model saw free gas. A free-gas volume per frame, such as the overflow reference
(:mod:`darsia.experiment.overflow`), decides how much free gas each frame holds,
as a saturation of at most 1 - Swi per pixel:

- more than the model's gas: the model's gas stays, and the rest is added on
  dissolved CO2 joined to seed gas, by a score: the gas probability averaged over
  a window, falling off with the distance from the injecting ports' cones during
  injection and from the end-gas region after it, and fading towards the region
  below the lowest started port. Each pixel takes a share of its capacity in
  proportion to its score, up to all of it, and the score threshold that gives
  the volume is solved exactly. No gas is added on seals, on gas the phase model
  dismissed as too small, or where gravity forbids free gas;
- less than the model's gas: whole pixels are peeled from each gas piece, by an
  anisotropic cost from its core outwards (the outer and lower parts first), and
  one partial pixel makes the volume exact; a peeled pixel becomes dissolved CO2,
  recorded in the rule map;
- no volume: the model's gas stays.

After an injection end, the free gas of the last frame before the end, widened
upwards through the porous facies to the seals and along the seal undersides it
reaches, is the end-gas region. The distance from it weights the gas
probability of the later frames, in the time average of the phase rules and in
the score.

The model's gas (:func:`darsia.model_gas`) is the phase map's free gas with the
gas that the gravity limit of the phase rules or the peel made dissolved CO2; our
production kept the water of those pixels saturated. Beside each frame's maps,
the allocation gives the CO2 of its labels before the rules, which the dilution
law of the mass reads.

Every setting is a field of :class:`GasAllocationSettings`; the phase rules'
settings give the CO2 threshold, the seals, the port zones and the gravity
limit, and the phase model gives the gas threshold.

"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil, hypot, radians, tan
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import darsia
from darsia.multiphase.phase_rules import (
    PhaseLabel,
    PhaseRule,
    PhaseRuleSettings,
    _change,
    _co2,
    _gravity_limit,
    _pitch_mm,
    _port_zones,
    _seal,
    _summed_areas,
    phase_labels,
    phase_maps,
)

_RELATIVE = 1e-9
"""A peeled pixel goes whole when its room exceeds what is left by at most this."""
_FADE_REACH = 8
"""Beyond this many fade lengths the fade weight is one."""
_STEP_ROUNDING = 1e-12
"""Rounding guard of the sideways steps of the upward widening."""
PHASE_MAPS = "maps.npz"
"""The file of a frame's stored phase maps (:func:`write_phase_maps`)."""
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
"""The eight neighbours of a pixel as (row, column) steps, in one fixed order, so
that a sum over them is reproducible."""


@dataclass(frozen=True)
class GasAllocationSettings:
    """Settings of the gas allocation; one configuration table."""

    seed_mm2: float
    """Gas bodies of at least this area seed added gas."""
    persistent_seed_mm2: float
    """Smaller bodies of this area seed too where they overlap a gas body of this
    area in the nearest earlier or later frame."""
    seed_window_h: float
    """Time [h] within which the nearest frames count for persistent seeds."""
    fallback_seed_mm2: float
    """Without any seed, gas bodies of this area seed; without those, the CO2 at
    an injecting port does."""
    score_window_mm: float
    """Width of the window that averages the gas probability into the score."""
    cone_base_mm: Sequence[float]
    """Half width of each injecting port's cone at the port's height, in the order
    the ports start; the last value repeats."""
    cone_width_mm: Sequence[float]
    """Half width the cone approaches far above each port."""
    cone_rise_mm: float
    """Height over which the cone widens towards that half width."""
    seal_band_mm: float
    """Depth of the band under the first seal above a port that the cone follows."""
    cone_decay_mm: float
    """Distance outside the cones over which the score falls by a factor e."""
    end_gas_decay_mm: float
    """Distance outside the end-gas region over which the score and the gas
    probability fall by a factor e."""
    end_gas_angle_deg: float
    """Half angle at which the end-gas region widens upwards."""
    port_fade_mm: float
    """Length over which added gas fades towards the region below the ports."""
    lattice_mm: float
    """Cell size of the lattice the peeling cost runs on."""
    cost_up_per_mm: float
    """Peeling cost of an upward step per mm."""
    cost_side_per_mm: float
    """Peeling cost of a sideways step per mm."""
    cost_down_per_mm: float
    """Peeling cost of a downward step per mm."""


def _components(mask):
    """8-connected components of a mask: count (with background) and labels."""
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)


def _touching(mask, seeds, components=None):
    """The 8-connected components of ``mask`` holding a seed pixel."""
    count, labels = _components(mask) if components is None else components
    keep = np.zeros(count, bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def _box(mask):
    """The bounding box of a mask's pixels."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return np.s_[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _ports(protocol):
    """Each port's injection periods (touching ones merged), by first start."""
    periods = {}
    for row in protocol.df[protocol.df["rate_kg_s"] > 0].itertuples():
        periods.setdefault((row.location_x, row.location_y), []).append(
            (row.start, row.end)
        )
    ports = {}
    for port, rows in sorted(periods.items(), key=lambda item: min(item[1])[0]):
        merged = []
        for start, end in sorted(rows):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        ports[port] = merged
    return ports


def _under_seal(co2, seal, y, depth_m):
    """CO2 within a depth below the top of its vertical run, where a seal caps it."""
    through = co2 & ~seal
    above_open = np.zeros_like(through)
    above_open[1:] = through[:-1]
    above_seal = np.zeros_like(through)
    above_seal[1:] = seal[:-1]
    rows = np.arange(len(y))[:, None]
    top = np.maximum.accumulate(np.where(through & ~above_open, rows, -1), axis=0)
    first = np.maximum(top, 0)
    depth = y[first] - y[:, None]
    contact = np.take_along_axis(above_seal, first, axis=0)
    return through & (top >= 0) & contact & (depth >= 0) & (depth <= depth_m)


def _cones(co2, injecting, order, radius, lower, seal, x, y, settings):
    """The injecting ports' cones: where added gas is preferred during injection.

    A port's CO2 body (the CO2 reaching its radius) below the first seal above
    the port is kept within a half width that widens with the height from the
    port's base to its full width, and within the port's lower zone below it;
    the band of the body under that seal joins where the cone reaches it.
    """
    components = _components(co2)
    cones = np.zeros(co2.shape, bool)
    for port in injecting:
        base, width = (
            values[min(order.index(port), len(values) - 1)]
            for values in (settings.cone_base_mm, settings.cone_width_mm)
        )
        body = _touching(co2, radius[port], components)
        column = int(np.abs(x - port[0]).argmin())
        sealing = seal[:, column] & (y > port[1])
        top = y[sealing].min() if sealing.any() else None
        rise = y[:, None] - port[1]
        scope = body if top is None else body & (y < top)[:, None]
        height = np.maximum(rise, 0) * 1000 / settings.cone_rise_mm
        half_width = base / 1000 - (width - base) / 1000 * np.expm1(-np.square(height))
        cone = np.abs(x[None, :] - port[0]) <= half_width
        kept = body & (~scope | np.where(rise >= 0, cone, lower[port]))
        if top is not None and scope.any():
            hit = np.flatnonzero(sealing & (y == top))[0]
            seals = _components(seal)[1]
            first = seals == seals[hit, column]
            band = _under_seal(
                body & ~seal & (rise >= 0), first, y, settings.seal_band_mm / 1000
            )
            contact = band & cone
            if contact.any():
                kept |= _touching(band, contact)
        cones |= scope & kept
    return cones


def _distance(region, active, pitch_mm, decay_mm):
    """Distance of the active pixels from a region, in decay lengths."""
    box = _box(region | active)
    distance = ndimage.distance_transform_edt(~region[box], sampling=pitch_mm)
    distance /= decay_mm
    return distance[active[box]]


def _upward(path, seeds, lateral_per_row):
    """Paths upwards from the seeds through ``path``, widening sideways.

    Each row up, the reach may step sideways by as many pixels as the rise
    allows at ``lateral_per_row``; a diagonal step needs an open face pixel.
    """
    path, seeds = path[::-1], seeds[::-1] & path[::-1]
    reached = np.zeros(path.shape, bool)
    seed_rows = np.flatnonzero(seeds.any(axis=1))
    if not seed_rows.size:
        return reached
    first = int(seed_rows[0])
    reached[first] = seeds[first]
    for row in range(first + 1, path.shape[0]):
        previous, open_cells = reached[row - 1], path[row]
        advance = previous & open_cells
        rise = row - first
        moves = int(rise * lateral_per_row + _STEP_ROUNDING) - int(
            (rise - 1) * lateral_per_row + _STEP_ROUNDING
        )
        if moves:
            above = path[row - 1]
            advance[1:] |= (
                previous[:-1] & open_cells[1:] & (open_cells[:-1] | above[1:])
            )
            advance[:-1] |= (
                previous[1:] & open_cells[:-1] & (open_cells[1:] | above[:-1])
            )
            for _ in range(1, min(moves, advance.size)):
                spread = advance.copy()
                spread[1:] |= advance[:-1] & open_cells[1:]
                spread[:-1] |= advance[1:] & open_cells[:-1]
                if np.array_equal(spread, advance):
                    break
                advance = spread
        reached[row] = advance | seeds[row]
    return reached[::-1]


def _end_gas(gas, porous, seal, pitch_mm, settings):
    """The end-gas region of a frame's free gas; None without free gas."""
    if not gas.any():
        return None
    space = porous | gas
    dy, dx = pitch_mm
    slope = dy / dx * tan(radians(settings.end_gas_angle_deg))
    reached = _upward(space & ~seal, gas, slope)
    underside = np.zeros_like(space)
    underside[1:] = space[1:] & ~seal[1:] & seal[:-1]
    contact = underside & reached
    if contact.any():
        reached |= _touching(underside, contact)
    return reached | gas


def _seeds(model, candidate, allowance, neighbours, pixel_area, settings):
    """Seed pixels of added gas: large or persistent gas bodies, else fallbacks."""
    count, bodies = _components(model)
    area = _summed_areas(bodies, count, pixel_area)
    persistent = np.zeros(count, bool)
    for gas in neighbours:
        other_count, other = _components(gas)
        held = _summed_areas(other, other_count, pixel_area)
        held = held >= settings.persistent_seed_mm2
        held[0] = False
        persistent[bodies[held[other]]] = True
    large = (area >= settings.seed_mm2) | (
        persistent & (area >= settings.persistent_seed_mm2)
    )
    large[0] = False
    if not large.any():
        large = area >= settings.fallback_seed_mm2
        large[0] = False
    seeds = large[bodies]
    return seeds if seeds.any() else allowance & candidate


def _fade(below, active, pitch_mm, fade_mm):
    """Weight of the active pixels falling to zero at ``below``; None far off."""
    rows, cols = np.flatnonzero(active.any(1)), np.flatnonzero(active.any(0))
    if not rows.size:
        return None
    height, width = active.shape
    reach_rows = ceil(min(height, _FADE_REACH * (fade_mm / pitch_mm[0])))
    reach_cols = ceil(min(width, _FADE_REACH * (fade_mm / pitch_mm[1])))
    box = np.s_[
        max(0, rows[0] - reach_rows) : min(height, rows[-1] + reach_rows + 1),
        max(0, cols[0] - reach_cols) : min(width, cols[-1] + reach_cols + 1),
    ]
    if not below[box].any():
        return None
    distance = ndimage.distance_transform_edt(~below[box], sampling=pitch_mm)
    weight = -np.expm1(-np.square(distance[active[box]] / fade_mm))
    return np.maximum(weight, np.finfo(np.float32).tiny)


def _threshold(score, capacity, volume):
    """The score threshold t with sum(capacity * min(1, score / t)) = volume."""
    order = np.argsort(score, kind="stable")
    p, w = score[order], capacity[order]
    prefix_w, prefix_wp = np.cumsum(w), np.cumsum(w * p)
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    denominator = np.float64(volume) - np.maximum(0.0, prefix_w[-1] - prefix_w[ends])
    with np.errstate(divide="ignore", invalid="ignore"):
        threshold = prefix_wp[ends] / denominator
    upper = np.r_[p[ends[:-1] + 1], np.inf]
    valid = (denominator > 0) & (p[ends] <= threshold) & (threshold <= upper)
    if not valid.any():
        raise ValueError("No score threshold gives the free-gas volume")
    return float(threshold[valid.argmax()])


def _add(
    saturation,
    volume,
    gas,
    candidate,
    room,
    cap,
    scores,
    seeds,
    near,
    below,
    pitch_mm,
    settings,
):
    """Add a volume [m3] on the candidates joined to the seeds, by their score.

    ``scores`` are the scored pixels and their gas probability, ``gas`` the gas
    through which added gas joins its seeds, and ``near`` gives the distance of
    the active pixels from the preferred region in decay lengths (or None).
    """
    dy, dx = pitch_mm
    window = (
        2 * int(settings.score_window_mm / (2 * dx)) + 1,
        2 * int(settings.score_window_mm / (2 * dy)) + 1,
    )
    scored, values = scores
    box = dict(normalize=False, borderType=cv2.BORDER_CONSTANT)
    score = cv2.boxFilter(np.where(scored, values, 0), -1, window, **box)
    support = cv2.boxFilter(scored.astype(np.float32), -1, window, **box)
    np.divide(score, support, out=score, where=support > 0)
    np.clip(score, 0, 1, out=score)
    candidate = candidate & (score > 0)
    active = candidate & _touching(gas | candidate, seeds)
    probability = score[active].astype(np.float64)
    distance = near(active)
    if distance is not None:
        probability = np.exp(np.log(probability) - distance)
    capacity, peak = room[active], cap[active]
    fade = _fade(below, active, pitch_mm, settings.port_fade_mm)
    if fade is not None:
        capacity, peak = capacity * fade, peak * fade
    available = float(capacity.sum())
    placed = min(volume, available)
    if placed > 0:
        threshold = (
            float(probability.min())
            if placed == available
            else _threshold(probability, capacity, placed)
        )
        saturation[active] = peak * np.minimum(1.0, probability / threshold)


def _lattice_cost(cells, sources, step_mm, settings):
    """Least cost from the source cells through the cell lattice (inf: unreached).

    A step to one of the 8 neighbours costs its vertical part at the up or down
    cost and its horizontal part at the side cost, combined as a hypotenuse.
    Rows run downwards.
    """
    up, side, down = (
        settings.cost_up_per_mm,
        settings.cost_side_per_mm,
        settings.cost_down_per_mm,
    )
    index = np.full(cells.shape, -1, np.int64)
    index[cells] = np.arange(int(cells.sum()))
    rows, cols = cells.shape
    starts, ends, weights = [], [], []
    for dr, dc in NEIGHBOURS:
        vertical = 0.0 if dr == 0 else (up if dr == -1 else down) * step_mm[0]
        horizontal = 0.0 if dc == 0 else side * step_mm[1]
        a = np.s_[max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)]
        b = np.s_[max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)]
        both = cells[a] & cells[b]
        starts.append(index[a][both])
        ends.append(index[b][both])
        weights.append(np.full(int(both.sum()), hypot(vertical, horizontal)))
    count = int(cells.sum())
    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(starts), np.concatenate(ends))),
        shape=(count, count),
    )
    cost = np.full(cells.shape, np.inf)
    cost[cells] = dijkstra(graph, indices=index[sources & cells], min_only=True)
    return cost


def _peel_order(model, y, room, pitch_mm, settings):
    """The model's gas pixels in the order peeling takes them, with their room.

    On a lattice of cells, each 8-connected piece's core is its cells deepest
    inside it; pixels go by the highest cost from the cores first, then by
    cell, then the lowest first, then row-major.
    """
    cell = tuple(max(1, int(round(settings.lattice_mm / pitch))) for pitch in pitch_mm)
    step = (cell[0] * pitch_mm[0], cell[1] * pitch_mm[1])
    box = _box(model)
    region = model[box]
    shape = tuple(-(-n // c) for n, c in zip(region.shape, cell))
    padded = np.zeros((shape[0] * cell[0], shape[1] * cell[1]), bool)
    padded[: region.shape[0], : region.shape[1]] = region
    cells = padded.reshape(shape[0], cell[0], shape[1], cell[1]).any(axis=(1, 3))
    depth = ndimage.distance_transform_edt(np.pad(cells, 1))[1:-1, 1:-1]
    pieces, count = ndimage.label(cells, structure=np.ones((3, 3), bool))
    deepest = np.asarray(ndimage.maximum(depth, pieces, np.arange(1, count + 1)))
    core = cells & (depth == np.r_[np.inf, deepest][pieces])
    cost = _lattice_cost(cells, core, step, settings)
    rows, cols = np.nonzero(region)
    cell_rows, cell_cols = rows // cell[0], cols // cell[1]
    value = cost[cell_rows, cell_cols]
    rows, cols = rows + box[0].start, cols + box[1].start
    order = np.lexsort(
        (
            rows * model.shape[1] + cols,
            y[rows],
            cell_rows * shape[1] + cell_cols,
            -value,
        )
    )
    return rows[order], cols[order], room[rows, cols][order]


def _peel(saturation, volume, model, room, pore_volume, cap, y, pitch_mm, settings):
    """Remove a volume [m3] of the model's gas; the saturation and freed pixels.

    Whole pixels go in the peeling order while they fit, then one partial
    pixel; a volume of None removes all of it.
    """
    taken = np.zeros(model.shape)
    if volume is None:
        taken[model] = room[model]
    else:
        rows, cols, capacity = _peel_order(model, y, room, pitch_mm, settings)
        left, index = float(volume), 0
        rounding = _RELATIVE * abs(left)
        if left > rounding:
            while index < len(rows) and capacity[index] <= left * (1 + _RELATIVE):
                taken[rows[index], cols[index]] = capacity[index]
                left -= capacity[index]
                index += 1
            if left > rounding and index < len(rows):
                taken[rows[index], cols[index]] = left
    gone = taken > 0
    left = np.where(
        gone, (room - taken) / np.where(pore_volume > 0, pore_volume, 1.0), 0
    )
    saturation = np.where(gone, np.clip(left, 0, None).astype(cap.dtype), saturation)
    return saturation, gone & (saturation == 0)


def free_gas(
    probabilities: Sequence[tuple[np.ndarray, np.ndarray]],
    dates: Sequence[datetime],
    sand_map: darsia.Image,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    permeability: Mapping[int, float],
    protocol: darsia.InjectionProtocol,
    rule_settings: PhaseRuleSettings,
    pore_volume_m3: np.ndarray,
    swi: np.ndarray,
    volumes_m3: Sequence[float | None],
    settings: GasAllocationSettings,
    chart: np.ndarray | None = None,
    *,
    gas_threshold: float,
    start: int = 0,
    anchor_saturation: Callable[[int], np.ndarray] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Phase and rule maps and free-gas saturations of a run's frames, by volume.

    The phase maps are those of :func:`darsia.phase_maps`, with the end-gas
    weight described in the module; each frame's free gas then holds the
    frame's volume as described there, and the gas it peels is dissolved CO2;
    the rule map records it (:attr:`darsia.PhaseRule.PEELED`). Beside them come
    the frame's labels before the rules (:func:`darsia.phase_labels`) and their
    CO2, which neither the rules nor the allocation change: the colour of the
    frame's dissolved CO2 is read on these labels, as our production read it.

    A run resumes at its first missing frame (``start``). Across frames, the
    allocation keeps only the end-gas regions, each built from its anchor frame's
    own free-gas saturation, so the regions of the anchors before ``start`` are
    built again from their saved saturations (``anchor_saturation``); every later
    map and saturation then equals the run's made whole.

    Parameters
    ----------
    probabilities : sequence of (array, array)
        CO2-presence and gas-within-CO2 probabilities of all of the run's frames,
        in time order, on the grid of the sand map (:func:`darsia.phase_maps`).
    dates : sequence of datetime
        Acquisition time of each frame.
    sand_map : darsia.Image
        Facies ids of the run's grid, with its physical dimensions and origin.
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel, as for :func:`darsia.phase_labels`.
    permeability : mapping of int to float
        Permeability of each porous facies, in any unit.
    protocol : darsia.InjectionProtocol
        Injection timetable; each injecting location is a port.
    rule_settings : PhaseRuleSettings
        Settings of the phase rules.
    pore_volume_m3 : array
        Pore volume of each pixel of the map; pixels without one hold no gas.
    swi : array
        Irreducible water saturation of each pixel; the saturations are in its
        dtype.
    volumes_m3 : sequence of float or None
        Free-gas volume of each frame; None keeps the model's gas.
    settings : GasAllocationSettings
        Settings of the allocation.
    chart : array, optional
        The colour checker's pixels.
    gas_threshold : float
        Gas threshold of the phase model, as for :func:`darsia.phase_maps`.
    start : int, optional
        Index of the first frame whose maps are made; the earlier frames are
        read only as neighbours.
    anchor_saturation : callable, optional
        The free-gas saturation this allocation gave an earlier frame, by its
        index; needed for each injection-end anchor before ``start``.

    Yields
    ------
    tuple of arrays
        The phase map (uint8), its rule map (uint8, :class:`darsia.PhaseRule`),
        the free-gas saturation, the CO2 of the labels before the rules (bool) and
        those labels (uint8, :func:`darsia.phase_labels`), on which the colour of
        the frame's dissolved CO2 is read, of each frame from ``start`` on.

    Raises
    ------
    ValueError
        If an anchor lies before ``start`` and no saved saturation is given.

    """
    facies = sand_map.img
    x, y = darsia.pixel_centres(sand_map)
    pitch_mm = _pitch_mm(sand_map)
    ports = _ports(protocol)
    order = list(ports)
    radius, lower = _port_zones(x, y, ports, rule_settings)
    seal = _seal(facies, domain, permeability, rule_settings)
    porous = [k for k in permeability if k not in rule_settings.free_water_facies]
    porous = domain & np.isin(facies, porous)
    eligible = pore_volume_m3 > 0
    episodes = sorted({period for periods in ports.values() for period in periods})
    anchors = {}
    for _, end in episodes:
        before = [index for index, date in enumerate(dates) if date <= end]
        if before:
            anchors[end] = before[-1]
    regions, stored = {}, {}

    def end_region(saturation):
        """The end-gas region of an anchor's saturation, its distance and weight."""
        gas = _end_gas(saturation > 0, porous, seal, pitch_mm, settings)
        if gas is None:
            return None
        distance = ndimage.distance_transform_edt(~gas, sampling=pitch_mm)
        distance /= settings.end_gas_decay_mm
        return distance, np.exp(-distance)

    for index in sorted(set(anchors.values())):
        if index < start:
            if anchor_saturation is None:
                raise ValueError("A resumed allocation needs its anchors' saturation")
            found = end_region(anchor_saturation(index))
            if found is not None:
                regions[index] = found

    def end_gas(index):
        """Distance from the end-gas region that applies to a frame, and weight."""
        date = dates[index]
        if any(start <= date <= end for start, end in episodes):
            return None
        stop = max((end for _, end in episodes if end < date), default=None)
        return regions.get(anchors.get(stop))

    def weight(index):
        """The end-gas weight of a frame's gas probability, or None."""
        region = end_gas(index)
        return None if region is None else region[1]

    def stored_labels(index):
        """A frame's labels before the rules (:func:`darsia.phase_labels`)."""
        if index not in stored:
            stored[index] = phase_labels(
                *probabilities[index],
                domain,
                pixel_area_mm2,
                rule_settings,
                gas_threshold=gas_threshold,
            )
        return stored[index]

    maps = phase_maps(
        probabilities,
        dates,
        sand_map,
        domain,
        pixel_area_mm2,
        permeability,
        protocol,
        rule_settings,
        chart,
        weight,
        gas_threshold=gas_threshold,
        start=start,
    )
    for index, (labels, rules) in enumerate(maps, start=start):
        date = dates[index]
        gas_probability = probabilities[index][1]
        own = stored_labels(index)
        nearest = [
            next((j for j in side if dates[j] != date), None)
            for side in (range(index - 1, -1, -1), range(index + 1, len(dates)))
        ]
        neighbours = [
            stored_labels(j) == PhaseLabel.GAS
            for j in nearest
            if j is not None
            and abs((dates[j] - date).total_seconds()) / 3600 <= settings.seed_window_h
        ]
        for j in [j for j in stored if j < index]:
            del stored[j]

        started = [p for p, periods in ports.items() if periods[0][0] <= date]
        injecting = [
            p for p, periods in ports.items() if any(s <= date < e for s, e in periods)
        ]
        co2 = _co2(labels)
        below, lobes = _gravity_limit(
            co2, started, radius, lower, y, pitch_mm, rule_settings
        )
        dissolved = labels == PhaseLabel.DISSOLVED
        keep = co2 & eligible
        cap = np.zeros_like(swi)
        cap[keep] = swi.dtype.type(1) - swi[keep]
        dismissed = (own == PhaseLabel.DISSOLVED) & (gas_probability >= gas_threshold)
        cap[dissolved & (dismissed | below | lobes)] = 0
        pore_volume = np.where(keep, pore_volume_m3, 0).astype(np.float64)
        room = pore_volume * cap.astype(np.float64)
        allowed = keep & (pore_volume_m3 * cap > 0)
        model = allowed & (labels == PhaseLabel.GAS)
        saturation = np.where(model, cap, cap.dtype.type(0))
        target = volumes_m3[index]
        if target is not None:
            model_volume = float(np.sum(room[model]))
            if target >= model_volume:
                region = end_gas(index)
                cones = None
                if region is None and injecting:
                    cones = _cones(
                        co2, injecting, order, radius, lower, seal, x, y, settings
                    )

                def near(active):
                    """Distance of the active pixels from the preferred region."""
                    if region is not None:
                        return region[0][active]
                    if cones is None or not cones.any() or not active.any():
                        return None
                    return _distance(cones, active, pitch_mm, settings.cone_decay_mm)

                candidate = allowed & dissolved & ~seal & ~below & ~lobes
                allowance = np.zeros(co2.shape, bool)
                for port in injecting:
                    allowance |= radius[port]
                seeds = _seeds(
                    model,
                    candidate,
                    allowance & co2,
                    neighbours,
                    pixel_area_mm2,
                    settings,
                )
                scores = (
                    np.isfinite(gas_probability) & co2,
                    np.where(_co2(own), gas_probability, 0),
                )
                _add(
                    saturation,
                    target - model_volume,
                    model & ~seal,
                    candidate,
                    room,
                    cap,
                    scores,
                    seeds,
                    near,
                    below,
                    pitch_mm,
                    settings,
                )
            else:
                saturation, freed = _peel(
                    saturation,
                    model_volume - target if target > 0 else None,
                    model,
                    room,
                    pore_volume,
                    cap,
                    y,
                    pitch_mm,
                    settings,
                )
                _change(labels, rules, freed, PhaseLabel.DISSOLVED, PhaseRule.PEELED)
            saturation = np.minimum(saturation, cap)
        if index in anchors.values():
            found = end_region(saturation)
            if found is not None:
                regions[index] = found
        yield labels, rules, saturation, _co2(own), own


def _maps_file(held_out: str | None) -> str:
    """The file of a frame's stored maps, or of its maps without a held-out run's
    estimate (:func:`write_phase_maps`)."""
    return PHASE_MAPS if held_out is None else f"maps.without-{held_out}.npz"


def write_phase_maps(
    folder: Path,
    labels: np.ndarray,
    rules: np.ndarray,
    saturation: np.ndarray,
    co2: np.ndarray,
    before: np.ndarray,
    held_out: str | None = None,
) -> None:
    """Store a frame's maps as :func:`free_gas` yields them, in ``folder``
    (:data:`PHASE_MAPS`): its phase and rule maps, its free-gas saturation, the
    CO2 of its labels before the rules and those labels. With ``held_out``, the
    maps made without that run's estimated free-gas volume, which a calibration
    fold leaving it out reads, are stored beside them."""
    arrays = dict(labels=labels, rules=rules, saturation=saturation, co2=co2)
    np.savez(Path(folder) / _maps_file(held_out), **arrays, before=before)


def read_phase_maps(folder: Path, held_out: str | None = None) -> dict[str, np.ndarray]:
    """A frame's stored maps (:func:`write_phase_maps`): ``labels``, ``rules``,
    ``saturation``, ``co2`` and ``before``; with ``held_out``, those made without
    that run's estimate."""
    with np.load(Path(folder) / _maps_file(held_out)) as stored:
        return {name: stored[name] for name in stored.files}
