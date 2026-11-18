"""Allocate each frame's free-gas volume inside causal growing contours.

The contour is built from each prefix's GAS after the unchanged phase
rules, before adding or peeling gas. Observed CO2 and gas probability during
injection determine a smooth path to each injector where needed. The checker,
free water and impermeable seals are excluded. The same per-frame contours and
model maps are shared by all overflow variants; each allocation uses its own
volume and intersects its contour with that frame's CO2.

Extra gas follows a normalized Gaussian probability score, an inward fade and
the unchanged seed, gravity and capacity rules. Lower volumes peel the model
gas by the anisotropic cost, recording PEELED pixels as dissolved CO2. No volume
keeps the model gas. The pre-rule labels and CO2 remain the optical mass input.
Geometric lengths scale with the median depth of the run's measured rig map.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import darsia
from darsia.multiphase.gas_parameters import GasAllocationSettings, gas_parameters
from darsia.multiphase.gas_model_cache import GasModelCache, GasModelFrame
from darsia.multiphase.gas_bridge import bridge_scores, hidden_bridges
from darsia.multiphase.gas_zone import GasZone, gas_model_zones, injection_ports
from darsia.multiphase.gas_zone_cache import GasZoneCache
from darsia.multiphase.phase_rules import (
    PhaseLabel,
    PhaseRule,
    PhaseRuleSettings,
    _change,
    _co2,
    _gravity_limit,
    _port_zones,
    _seal,
    _pitch_mm,
    _summed_areas,
    phase_labels,
    phase_maps,
)

_RELATIVE = 1e-9
"""A peeled pixel goes whole when its room exceeds what is left by at most this."""
PHASE_MAPS = "maps.npz"
"""The file of a frame's stored phase maps (:func:`write_phase_maps`)."""
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
"""The eight neighbours of a pixel as (row, column) steps, in one fixed order, so
that a sum over them is reproducible."""


@dataclass(frozen=True)
class GasAllocationRecord:
    """Requested and placed gas volumes for one allocation frame, in m3."""

    target_m3: float | None
    """Requested volume, or None when the model gas alone decides it."""
    model_m3: float
    """Model gas after the phase rules, before adding or peeling."""
    placed_m3: float
    """Volume represented by the returned saturation."""
    unplaced_m3: float
    """Requested volume that could not be placed, at least zero."""
    limit_area_mm2: float
    """Area of the current CO2 and active bridge limit, before gravity checks."""
    disconnected_added_area_mm2: float
    """Added gas not connected to this allocation's seed body; must be zero."""
    bridge_length_mm: float = 0.0
    """Total length of active hidden stretches carrying added gas."""
    bridge_area_mm2: float = 0.0
    """Union area of hidden pixels with positive added saturation."""
    geometry: dict = field(default_factory=dict)
    """Measured depth and all derived geometric lengths for this run."""
    bridge_volume_m3: float = 0.0
    """Inferred bridge gas, already included in placed_m3, never an extra volume."""
    active_link_diagnostics: list[dict] = field(default_factory=list)
    """Raw geometry outcomes for active ports, including local-depth refusals."""


def _active_link_diagnostics(zone, ports):
    """Keep compact JSON-native link provenance after the zone cache is released."""
    rows = []
    for port in ports:
        note = zone.diagnostics.get("ports", {}).get(port, {})
        row = {
            "port_m": [float(value) for value in port],
            "status": str(note.get("status", "unrecorded_geometry")),
        }
        for name in ("port_pixel", "meeting_pixel", "target_pixel"):
            if name in note:
                row[name] = [int(value) for value in note[name]]
        for name in ("local_depth_mm", "downward_drop_mm"):
            if name in note:
                row[name] = None if note[name] is None else float(note[name])
        rows.append(row)
    return rows


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




def _inward_weight(limit, pitch_mm, decay_mm):
    """One minus exp(-d/lambda), with zero at boundary pixels and outside.

    Boundary pixel centres define zero distance. Padding makes a limit touching
    the image edge fade there too. Distances use the grid's metric pitch.
    """
    inside = ndimage.binary_erosion(limit, structure=np.ones((3, 3), bool))
    if not inside.any():
        return np.zeros(limit.shape, dtype=np.float64)
    boundary = limit & ~inside
    distance = ndimage.distance_transform_edt(~boundary, sampling=pitch_mm)
    return np.where(inside, -np.expm1(-distance / decay_mm), 0.0)


def _round_score(scored, values, pitch_mm, window_mm):
    """Normalized Gaussian probability mean, matching the box's second moment.

    Its physical sigma is window/sqrt(12), truncated at three sigma. Dividing
    by the filtered support excludes unscored pixels without a square kernel.
    """
    sigma = window_mm / np.sqrt(12)
    reach = [int(np.ceil(3 * sigma / step)) for step in pitch_mm]
    rows = np.arange(-reach[0], reach[0] + 1) * pitch_mm[0]
    cols = np.arange(-reach[1], reach[1] + 1) * pitch_mm[1]
    squared = rows[:, None] ** 2 + cols[None, :] ** 2
    disc = squared <= (3 * sigma) ** 2
    kernel = np.where(disc, np.exp(-squared / (2 * sigma**2)), 0)
    kernel /= kernel.sum()
    kernel = kernel.astype(values.dtype)
    score = fftconvolve(np.where(scored, values, 0), kernel, mode="same")
    support = fftconvolve(scored.astype(values.dtype), kernel, mode="same")
    np.divide(score, support, out=score, where=support > 0)
    np.clip(score, 0, 1, out=score)
    # FFT-backed filters can leave tiny positive roundoff outside exact support.
    positive = scored & (values > 0)
    if positive.any():
        distance = ndimage.distance_transform_edt(~positive, sampling=pitch_mm)
        score[distance > 3 * sigma] = 0
    else:
        score.fill(0)
    return score


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


def _connected_candidates(gas, candidate, seeds, meetings):
    """Seed an otherwise unseeded admissible component at its active meeting.

    ``candidate`` already includes the positive coefficient/fixed-fraction test.
    A meeting outside this graph grants nothing. Existing seed pixels are kept;
    no input mask is changed and no geometric or numerical floor is introduced.
    """
    graph = gas | candidate
    components = count, labels = _components(graph)
    seeded = np.zeros(count, bool)
    seeded[np.unique(labels[seeds])] = True
    seeded[0] = False
    selected = seeds
    for point in meetings:
        component = labels[point]
        if component and not seeded[component]:
            if selected is seeds:
                selected = seeds.copy()
            selected[point] = True
            seeded[component] = True
    return candidate & _touching(graph, selected, components), selected


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
    weight,
    pitch_mm,
    settings,
    bridges=(),
    meetings=(),
):
    """Add a volume [m3] on the candidates joined to the seeds, by their score.

    ``scores`` are the scored pixels and their gas probability, ``gas`` the gas
    through which added gas joins its seeds, and ``weight`` is the inward fade,
    zero at the limit's edge and outside it. Active link meetings can seed
    an otherwise unseeded admissible component. The effective seeds are
    returned for connectivity diagnostics.
    """
    scored, values = scores
    score = _round_score(scored, values, pitch_mm, settings.score_window_mm)
    coefficient = score.astype(np.float64) * weight
    fixed = np.zeros_like(coefficient)
    scale = None
    if bridges:
        coefficient, fixed, scale = bridge_scores(coefficient, candidate, gas, bridges)
    positive = (coefficient > 0) | (fixed > 0)
    candidate = candidate & positive
    active, seeds = _connected_candidates(gas, candidate, seeds, meetings)
    fixed_volume = float(np.sum(room[active] * fixed[active]))
    if fixed_volume > volume + 1e-12:
        raise ValueError("The measured added volume cannot fill the fixed gas bridge")
    fixed_cells = active & (fixed > 0)
    saturation[fixed_cells] = cap[fixed_cells] * fixed[fixed_cells]
    active &= ~fixed_cells
    probability = coefficient[active]
    capacity, peak = room[active], cap[active]
    if scale is not None:
        capacity = capacity * scale[active]
        peak = peak * scale[active]
    available = float(capacity.sum())
    placed = min(max(0, volume - fixed_volume), available)
    if placed > 0:
        threshold = (
            float(probability.min())
            if placed == available
            else _threshold(probability, capacity, placed)
        )
        saturation[active] = peak * np.minimum(1.0, probability / threshold)
    return seeds


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


def gas_model_frames(
    probabilities,
    dates,
    sand_map,
    domain,
    pixel_area_mm2,
    permeability,
    protocol,
    rule_settings,
    chart=None,
    *,
    gas_threshold,
):
    """Yield the model's preallocation state once for every selected frame.

    No volume or held-out overflow variant enters these maps. The probability
    sequence keeps its existing temporal window; consumers can spool each frame
    while accumulating the prefix contours.

    Parameters
    ----------
    probabilities : sequence of (array, array)
        CO2 and conditional gas probabilities in selected acquisition order.
    dates : sequence of datetime
        Acquisition times of the selected frames.
    sand_map : darsia.Image
        Facies labels and the physical grid.
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Physical pixel areas in square millimetres.
    permeability : mapping
        Permeability by porous facies label.
    protocol : darsia.InjectionProtocol
        Injection timetable and measured injector positions.
    rule_settings : PhaseRuleSettings
        Existing phase rule settings, unchanged by volume allocation.
    chart : array, optional
        Colour checker mask.
    gas_threshold : float
        Conditional gas probability threshold of the phase model.

    Yields
    ------
    GasModelFrame
        Gas probability and the pre-rule, post-rule and rule labels.
    """
    made = phase_maps(
        probabilities,
        dates,
        sand_map,
        domain,
        pixel_area_mm2,
        permeability,
        protocol,
        rule_settings,
        chart,
        None,
        gas_threshold=gas_threshold,

    )
    for index, (labels, rules) in enumerate(made):
        probability = probabilities[index]
        before = phase_labels(
            *probability,
            domain,
            pixel_area_mm2,
            rule_settings,
            gas_threshold=gas_threshold,
        )
        yield GasModelFrame(probability[1], before, labels, rules)


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
    depth_m: np.ndarray | None = None,
    start: int = 0,
    record: Callable[[int, GasAllocationRecord], None] | None = None,
    model_frames: Sequence[GasModelFrame] | None = None,
    zones: Sequence[GasZone] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Phase and rule maps and free-gas saturations of a run's frames, by volume.

    The phase maps are those of :func:`darsia.phase_maps`; each frame holds the
    frame's volume as described there, and the gas it peels is dissolved CO2;
    the rule map records it (:attr:`darsia.PhaseRule.PEELED`). Beside them come
    the frame's labels before the rules (:func:`darsia.phase_labels`) and their
    CO2, which neither the rules nor the allocation change: the colour of the
    frame's dissolved CO2 is read on these labels, as our production read it.

    A run resumes at its first missing frame (``start``). The complete selected
    run is prepared before allocation, including earlier completed frames. The
    prefix contour never depends on allocated or peeled output. Prepared maps and
    their contour may be shared across independent allocation variants.

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
    depth_m : array, optional
        Measured local rig depth in metres. Required to derive the geometric
        lengths, including when using prepared model states and contours.
    start : int, optional
        Index of the first frame to allocate. The complete run remains the
        source of the contour, neighbours and any active-injection signal.
    record : callable, optional
        Receives a frame index and its numeric :class:`GasAllocationRecord`
        before that frame is yielded. The iterator still yields five arrays.
    model_frames : sequence of GasModelFrame, optional
        Complete preallocation model states. If omitted, build them once in a
        bounded private temporary cache.
    zones : sequence of GasZone, optional
        Causal nonshrinking contours prepared from exactly those model states,
        one per frame. Required together with ``model_frames``.

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
        If prepared maps and contour disagree, or the start is out of range.

    """
    settings = gas_parameters(settings, depth_m)
    if model_frames is None:
        if zones is not None:
            raise ValueError("Supplied contours need their prepared model frames")
        if depth_m is None:
            raise ValueError("Contour preparation requires the measured rig depth map")
        with (
            GasModelCache(len(dates), kept=3) as prepared,
            GasZoneCache(len(dates)) as contours,
        ):
            originals = gas_model_frames(
                probabilities,
                dates,
                sand_map,
                domain,
                pixel_area_mm2,
                permeability,
                protocol,
                rule_settings,
                chart,
                gas_threshold=gas_threshold,

            )
            made = gas_model_zones(
                prepared.store(originals),
                dates,
                sand_map,
                domain,
                permeability,
                protocol,
                rule_settings,
                settings,

                chart=chart,
                depth_m=depth_m,
            )
            for _ in contours.store(made):
                pass
            yield from free_gas(
                probabilities,
                dates,
                sand_map,
                domain,
                pixel_area_mm2,
                permeability,
                protocol,
                rule_settings,
                pore_volume_m3,
                swi,
                volumes_m3,
                settings,
                chart,
                gas_threshold=gas_threshold,
                start=start,

                record=record,
                model_frames=prepared,
                zones=contours,
                depth_m=depth_m,
            )
        return
    if zones is None or len(model_frames) != len(dates) or len(zones) != len(dates):
        raise ValueError("Prepared maps and contours must cover the selected run")
    if not 0 <= start <= len(dates):
        raise ValueError("Allocation start lies outside the selected run")
    facies = sand_map.img
    x, y = darsia.pixel_centres(sand_map)
    pitch_mm = _pitch_mm(sand_map)
    ports = injection_ports(protocol)
    radius, lower = _port_zones(x, y, ports, rule_settings)
    seal = _seal(facies, domain, permeability, rule_settings)
    eligible = pore_volume_m3 > 0

    def stored_labels(index):
        return model_frames[index].before

    for index in range(start, len(dates)):
        frame = model_frames[index]
        zone = zones[index]
        date = dates[index]
        labels, rules = frame.labels.copy(), frame.rules.copy()
        gas_probability, own = frame.gas_probability, frame.before
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

        started = [p for p, periods in ports.items() if periods[0][0] <= date]
        injecting = [
            p for p, periods in ports.items() if any(s <= date < e for s, e in periods)
        ]
        co2 = _co2(labels)
        bridges = hidden_bridges(
            zone,
            co2,
            labels == PhaseLabel.WATER,
            eligible & ~seal,
            pitch_mm,
            settings,
        )
        hidden = np.zeros(co2.shape, bool)
        for bridge in bridges:
            hidden |= bridge.hidden
        limit = (zone.contour & co2) | hidden
        decay = settings.fade_length_mm
        if np.any((labels == PhaseLabel.GAS) & eligible & ~limit):
            raise ValueError("The contour excludes eligible preallocation model gas")
        below, lobes = _gravity_limit(
            co2, started, radius, lower, y, pitch_mm, rule_settings
        )
        dissolved = labels == PhaseLabel.DISSOLVED
        keep = (co2 | hidden) & eligible
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
        model_volume = float(np.sum(room[model]))
        seeds = None
        if target is not None:
            if target >= model_volume:
                inward = _inward_weight(limit, pitch_mm, decay)
                candidate = (
                    allowed & (dissolved | hidden) & ~seal & ~below & ~lobes & limit
                )
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
                meetings = []
                for port in injecting:
                    if port not in zone.links:
                        note = zone.diagnostics.get("ports", {}).get(port, {})
                        if note.get("status") == "already_inside":
                            point = tuple(note["port_pixel"])
                            if co2[point]:
                                meetings.append(point)
                        continue
                    points = zone.links[port].path_coordinates
                    seen = np.flatnonzero(co2[points[:, 0], points[:, 1]])
                    if seen.size:
                        meetings.append(tuple(points[int(seen[0])]))
                seeds = _add(
                    saturation,
                    target - model_volume,
                    model & ~seal,
                    candidate,
                    room,
                    cap,
                    scores,
                    seeds,
                    inward,
                    pitch_mm,
                    settings,
                    bridges=bridges,
                    meetings=meetings,
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
        inferred = hidden & (saturation > 0)
        # Keep optical WATER and before labels intact: this is inferred free
        # gas, not model evidence and not an inferred dissolved concentration.
        rules[inferred] = PhaseRule.GAS_BRIDGE
        if record is not None:
            placed = float(np.sum(pore_volume * saturation.astype(np.float64)))
            disconnected = np.zeros_like(model)
            if seeds is not None:
                added = (saturation > 0) & ~model
                disconnected = added & ~_touching(model | added, seeds)
            disconnected_area = (
                float(pixel_area_mm2) * int(disconnected.sum())
                if np.ndim(pixel_area_mm2) == 0
                else float(
                    np.sum(np.asarray(pixel_area_mm2)[disconnected], dtype=np.float64)
                )
            )
            area = (
                float(pixel_area_mm2) * int(limit.sum())
                if np.ndim(pixel_area_mm2) == 0
                else float(np.sum(np.asarray(pixel_area_mm2)[limit], dtype=np.float64))
            )
            record(
                index,
                GasAllocationRecord(
                    None if target is None else float(target),
                    model_volume,
                    placed,
                    0.0 if target is None else max(0.0, float(target) - placed),
                    area,
                    disconnected_area,
                    sum(b.length_mm for b in bridges if np.any(inferred & b.hidden)),
                    float(np.sum(pixel_area_mm2 * inferred)),
                    settings.geometry_record(),
                    float(np.sum(pore_volume[inferred] * saturation[inferred])),
                    _active_link_diagnostics(zone, injecting),
                ),
            )
        yield labels, rules, saturation, _co2(own), own
