"""Phase rules: phase maps of a run from its phase probabilities.

A phase map gives each pixel of a frame one category: water, dissolved CO2, free
gas or unknown. The labels of a frame follow from its CO2-presence and
gas-within-CO2 probabilities, the CO2 threshold of the settings and the gas
threshold of the phase model; CO2 is only where the phase model found it. The
rules then correct the labels with temporal and physical knowledge, in this
order:

1. time average: free gas is decided by the gas probability averaged with the
   run's neighbouring frames; small gas bodies of the frame itself, high enough
   above a started injection port, are kept;
2. colour checker: the chart, widened and extended downwards, is unknown;
3. free water: CO2 in free water joined to no CO2 in a porous facies is water;
4. minimum area: small CO2 components become water, small enclosed holes and
   small gas specks become dissolved CO2;
5. compartments: dissolved CO2 in a compartment (porous region bounded by seals)
   holding neither a started port nor free gas, nor joined to one by CO2, is
   water, and so is dissolved CO2 on a seal joined to no active compartment;
6. gravity: free gas below the lowest started port (outside the ports' lower
   zones), and in narrow lobes hanging below wider CO2, is dissolved CO2.

Beside each phase map, a rule map records which rule last changed each pixel's
label (:class:`PhaseRule`). A later allocation can record a gas bridge without
changing its optical WATER label. With the rule map, :func:`model_gas` gives the
free gas the phase model found before gravity (and a later peel) made it
dissolved CO2.

Distances are physical, taken from the grid of the sand map; the area of a
component is the sum of its pixels' areas, which the caller gives. Every
setting is a field of :class:`PhaseRuleSettings`.

"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from math import ceil

import cv2
import numpy as np
from scipy import ndimage

import darsia


class PhaseLabel(IntEnum):
    """Categories of a phase map (uint8)."""

    WATER = 0
    DISSOLVED = 1
    GAS = 2
    UNKNOWN = 255


class PhaseRule(IntEnum):
    """A pixel's last phase rule or allocated-gas provenance, in uint8.

    The values are those our production stored, so that its rule maps read the
    same.

    """

    UNCHANGED = 0
    FREE_WATER = 1
    """CO2 in free water joined to no CO2 in a porous facies, made water."""
    MINIMUM_AREA = 2
    """A small CO2 component, made water."""
    GRAVITY = 3
    """Free gas below the lowest started port or in a sinking lobe, made
    dissolved CO2."""
    COLOR_CHECKER = 4
    """The colour checker's region, made unknown."""
    COMPARTMENTS = 5
    """Dissolved CO2 of an inactive compartment or a lone seal, made water."""
    PEELED = 6
    """Free gas that the allocation of the measured gas peeled, made dissolved
    CO2; the rules here never set it."""
    TIME_AVERAGE_LOST = 7
    """Free gas the time average made dissolved CO2."""
    TIME_AVERAGE_GAINED = 8
    """Dissolved CO2 the time average made free gas."""
    HOLE_FILLED = 9
    """A small enclosed hole, made dissolved CO2."""
    GAS_SPECK = 10
    """A small gas speck, made dissolved CO2."""
    GAS_LIMIT = 11
    """Model gas outside a completed injection's episode limit, made
    dissolved CO2."""
    GAS_BRIDGE = 12
    """Gas allocated in an injection bridge on an optically WATER pixel. Its
    label and optical evidence stay WATER; it is not model gas or dissolved CO2."""


@dataclass(frozen=True)
class PhaseRuleSettings:
    """Settings of the phase labels and rules; one configuration table."""

    co2_threshold: float
    """Smallest CO2-presence probability of CO2."""
    smallest_gas_patch_mm2: float
    """Free gas in 4-connected patches below this area is dissolved CO2."""
    neighbour_window_h: float
    """Time [h] over which a neighbouring frame's weight falls linearly to zero."""
    neighbours_per_side: int
    """Number of nearest earlier and later frames entering the time average."""
    small_bubble_max_mm2: float
    """Averaged gas bodies below this area are dissolved; the frame's own gas
    pieces below it are its small bubbles."""
    small_bubble_min_height_mm: float
    """A small bubble is kept only if it lies wholly higher than this above the
    lowest started port."""
    chart_margin_mm: float
    """Width added around the colour checker."""
    chart_extend_below_mm: float
    """Length by which the colour checker's region extends downwards."""
    free_water_facies: Sequence[int]
    """Facies of free water; every other facies of the domain is porous."""
    min_component_mm2: float
    """CO2 in 8-connected components below this area is water."""
    max_hole_mm2: float
    """Enclosed non-CO2 holes below this area are dissolved CO2."""
    min_gas_component_mm2: float
    """Free gas in 8-connected pieces below this area is dissolved CO2."""
    seal_permeability_fraction: float
    """Porous facies below this fraction of the highest permeability are seals."""
    port_radius_mm: float
    """Radius around a started port from which CO2 hangs as an injection path."""
    port_zone_half_width_mm: float
    """Half width of the half-ellipse below a started port where gas is allowed."""
    port_zone_depth_mm: float
    """Depth of that half-ellipse."""
    finger_width_mm: float
    """CO2 narrower than this, hanging below wider CO2, is a sinking lobe."""


def _co2(labels: np.ndarray) -> np.ndarray:
    """Pixels labelled dissolved CO2 or free gas."""
    return (labels == PhaseLabel.DISSOLVED) | (labels == PhaseLabel.GAS)


def _change(labels, rules, mask, label, rule):
    """Give the pixels of a mask a label, and record the rule in the rule map."""
    labels[mask] = label
    rules[mask] = rule


def _summed_areas(components: np.ndarray, count: int, pixel_area: float | np.ndarray):
    """Area of each component (index 0: background): its pixels' areas summed."""
    if np.ndim(pixel_area) == 0:
        return np.bincount(components.ravel(), minlength=count) * pixel_area
    return np.bincount(components.ravel(), weights=pixel_area.ravel(), minlength=count)


def _areas(mask: np.ndarray, connectivity: int, pixel_area: float | np.ndarray):
    """Components of a mask, their count and their areas (index 0: background)."""
    count, components = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity
    )
    return components, _summed_areas(components, count, pixel_area)


def phase_labels(
    co2: np.ndarray,
    gas: np.ndarray,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    settings: PhaseRuleSettings,
    *,
    gas_threshold: float,
) -> np.ndarray:
    """Phase labels of one frame from its two phase probabilities.

    A domain pixel is CO2 where its CO2-presence probability reaches the CO2
    threshold, and free gas where its gas-within-CO2 probability also reaches
    the gas threshold; elsewhere it is water, including pixels without
    probabilities (NaN). Free gas in 4-connected patches smaller than
    ``smallest_gas_patch_mm2`` is dissolved CO2. Pixels outside the domain are
    unknown.

    Parameters
    ----------
    co2, gas : array
        CO2-presence and gas-within-CO2 probabilities of shape (rows, cols).
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel: one value for a uniform grid, or an array of shape
        (rows, cols), e.g. measured pixel areas. The area of a patch is the sum
        of its pixels' areas.
    settings : PhaseRuleSettings
        The CO2 threshold and the smallest gas patch.
    gas_threshold : float
        Gas threshold of the phase model that gave the probabilities
        (``PhaseModel.gas_threshold``).

    Returns
    -------
    array
        Labels (uint8) of :class:`PhaseLabel`.

    """
    present = domain & (co2 >= settings.co2_threshold)
    labels = np.where(domain, PhaseLabel.WATER, PhaseLabel.UNKNOWN).astype(np.uint8)
    labels[present] = PhaseLabel.DISSOLVED
    patches, area = _areas(present & (gas >= gas_threshold), 4, pixel_area_mm2)
    kept = area >= settings.smallest_gas_patch_mm2
    kept[0] = False
    labels[kept[patches]] = PhaseLabel.GAS
    return labels


def _time_average(
    labels, gas, neighbours, domain, chart, settings, gas_threshold, end_gas=None
):
    """Decide free gas by the time-averaged gas probability; return the raw gas.

    The gas probability of the frame (weight one) and of each neighbour (weight
    falling linearly from one to zero over the window), each taken where its own
    probabilities give CO2 and zero elsewhere, is averaged. Within the frame's
    CO2 the average decides free gas, except that gas the smallest-patch rule
    removed is not restored, nor is gas on the chart. The raw gas is the frame's
    own gas decision before the smallest-patch rule. A weight per pixel
    (``end_gas``) multiplies the average and the raw gas probability before
    their threshold.
    """
    dtype = gas.dtype.type
    scored = _co2(labels)
    total = np.where(scored, gas, 0)
    weight_sum = 1.0
    for (other_co2, other_gas), gap_h in neighbours:
        weight = max(0.0, 1.0 - gap_h / settings.neighbour_window_h)
        if weight > 0:
            other = domain & (other_co2 >= settings.co2_threshold)
            total += dtype(weight) * np.where(other, other_gas, 0)
            weight_sum += weight
    mean = total / dtype(weight_sum)
    if end_gas is not None:
        mean = (mean * end_gas).astype(mean.dtype)
    averaged = scored & (mean >= gas_threshold)
    raw = scored & (gas >= gas_threshold)
    gained = (labels == PhaseLabel.DISSOLVED) & averaged & ~raw
    if chart is not None:
        gained &= ~chart
    labels[(labels == PhaseLabel.GAS) & ~averaged] = PhaseLabel.DISSOLVED
    labels[gained] = PhaseLabel.GAS
    if end_gas is not None:
        raw &= gas * end_gas >= dtype(gas_threshold)
    return raw


def _small_bubbles(labels, raw, lowest_port_y, y, pixel_area, settings):
    """Keep the frame's own small gas bubbles beside the averaged gas bodies.

    Averaged gas bodies below ``small_bubble_max_mm2`` become dissolved CO2. An
    8-connected piece of the frame's own gas below that area, touching no large
    averaged body and lying wholly higher than ``small_bubble_min_height_mm``
    above the lowest started port, is free gas where it is CO2.
    """
    bodies, area = _areas(labels == PhaseLabel.GAS, 8, pixel_area)
    large = area >= settings.small_bubble_max_mm2
    large[0] = False
    labels[(bodies > 0) & ~large[bodies]] = PhaseLabel.DISSOLVED
    if lowest_port_y is None:
        return
    near = cv2.dilate(large[bodies].astype(np.uint8), np.ones((3, 3), np.uint8))
    pieces, area = _areas(raw, 8, pixel_area)
    small = area < settings.small_bubble_max_mm2
    small[0] = False
    small[np.unique(pieces[(near > 0) & (pieces > 0)])] = False
    bottom = [box[0].stop - 1 for box in ndimage.find_objects(pieces)]
    lowest = np.concatenate(([np.inf], y[bottom]))
    small &= lowest > lowest_port_y + settings.small_bubble_min_height_mm / 1000
    labels[small[pieces] & _co2(labels)] = PhaseLabel.GAS


def _chart_region(chart, margin_px, extend_px):
    """The chart widened by a square margin and extended downwards by its box."""
    square = np.ones((2 * margin_px + 1,) * 2, np.uint8)
    region = cv2.dilate(chart.astype(np.uint8), square) > 0
    rows, cols = np.flatnonzero(chart.any(1)), np.flatnonzero(chart.any(0))
    if extend_px > 0:
        region[
            rows[0] : rows[-1] + extend_px + 1,
            max(0, cols[0] - margin_px) : cols[-1] + margin_px + 1,
        ] = True
    return region


def _free_water(labels, rules, free):
    """Set to water the CO2 in free water joined to no CO2 in a porous facies."""
    co2 = _co2(labels)
    _, components = cv2.connectedComponents(co2.astype(np.uint8), connectivity=8)
    sand = np.unique(components[co2 & ~free])
    removed = co2 & free & ~np.isin(components, sand[sand > 0])
    _change(labels, rules, removed, PhaseLabel.WATER, PhaseRule.FREE_WATER)


def _minimum_area(labels, rules, domain, pixel_area, settings):
    """Remove small CO2 components, fill small holes, dissolve small gas specks.

    A component of a single row or column along the image border is removed
    whatever its area. A hole is a 4-connected non-CO2 component touching
    neither the image border nor an unknown pixel.
    """
    co2 = _co2(labels) & domain
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        co2.astype(np.uint8), connectivity=8
    )
    keep = _summed_areas(components, count, pixel_area) >= settings.min_component_mm2
    rows, cols = labels.shape
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    keep[(stats[:, cv2.CC_STAT_WIDTH] == 1) & np.isin(left, (0, cols - 1))] = False
    keep[(stats[:, cv2.CC_STAT_HEIGHT] == 1) & np.isin(top, (0, rows - 1))] = False
    keep[0] = False
    removed = co2 & ~keep[components]
    _change(labels, rules, removed, PhaseLabel.WATER, PhaseRule.MINIMUM_AREA)

    other = ~_co2(labels)
    holes, area = _areas(other, 4, pixel_area)
    blocked = ndimage.binary_dilation(labels == PhaseLabel.UNKNOWN)
    open_ = np.zeros(area.size, bool)
    open_[0] = True
    open_[holes[blocked & other]] = True
    open_[np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1]))] = True
    filled = (~open_ & (area < settings.max_hole_mm2))[holes]
    _change(labels, rules, filled, PhaseLabel.DISSOLVED, PhaseRule.HOLE_FILLED)

    pieces, area = _areas(labels == PhaseLabel.GAS, 8, pixel_area)
    small = area < settings.min_gas_component_mm2
    small[0] = False
    _change(labels, rules, small[pieces], PhaseLabel.DISSOLVED, PhaseRule.GAS_SPECK)


def _compartments(labels, rules, compartments, count, seal, sources):
    """Set to water the dissolved CO2 of inactive compartments and lone seals.

    A compartment is active if it holds a started port (``sources``) or free
    gas, or if an 8-connected CO2 path through compartments and seals joins it
    to an active compartment. Free gas is never removed.
    """
    co2 = _co2(labels)
    inside = co2 & (compartments > 0)
    paths, pieces = cv2.connectedComponents(
        (co2 & ((compartments > 0) | seal)).astype(np.uint8), connectivity=8
    )
    links = np.unique(np.stack([pieces[inside], compartments[inside]]), axis=1)
    active = np.zeros(count, bool)
    active[list(sources)] = True
    active[compartments[(labels == PhaseLabel.GAS) & (compartments > 0)]] = True
    while True:
        joined = links[1][np.isin(links[0], links[0][active[links[1]]])]
        if active[joined].all():
            break
        active[joined] = True
    by_area = active[compartments]
    reach = np.zeros(paths, bool)
    reach[pieces[inside & by_area]] = True
    reach[0] = False
    dissolved = labels == PhaseLabel.DISSOLVED
    removed = dissolved & (((compartments > 0) & ~by_area) | (seal & ~reach[pieces]))
    _change(labels, rules, removed, PhaseLabel.WATER, PhaseRule.COMPARTMENTS)


def _sinking_fingers(co2, anchors, width_mm, pitch_mm):
    """Narrow CO2 hanging below wider CO2 and reaching no port anchor.

    A horizontal opening of the requested width keeps the wide body. A narrow
    8-connected remnant attached to the body from above is a lobe where its
    tail below its last contact is at least as long as the remnant's width,
    capped at the requested width; only the tail below that contact is a lobe.
    """
    dy, dx = pitch_mm
    lobes = np.zeros(co2.shape, bool)
    width = ceil(width_mm / dx)
    rows, cols = np.flatnonzero(co2.any(axis=1)), np.flatnonzero(co2.any(axis=0))
    if width <= 1 or not rows.size or width > cols[-1] - cols[0] + 1:
        return lobes
    box = np.s_[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    image = np.ascontiguousarray(co2[box], dtype=np.uint8)
    kernel = np.ones((1, width), np.uint8)
    constant = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded = cv2.erode(image, kernel, anchor=(width // 2, 0), **constant)
    body = cv2.dilate(eroded, kernel, anchor=(width - 1 - width // 2, 0), **constant)
    narrow = image & (1 - body)
    if not narrow.any() or not body.any():
        return lobes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(narrow, connectivity=8)
    side = cv2.dilate(body, np.ones((1, 3), np.uint8))
    above = np.unique(labels[1:][(narrow[1:] & side[:-1]).astype(bool)])
    r, c = np.nonzero(narrow & cv2.dilate(side, np.ones((3, 1), np.uint8)))
    last_contact = np.full(count, -1, dtype=np.int32)
    np.maximum.at(last_contact, labels[r, c], r)
    bottom = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] - 1
    required = np.minimum(width_mm, stats[:, cv2.CC_STAT_WIDTH] * dx)
    hanging = np.zeros(count, bool)
    hanging[above] = (bottom[above] - last_contact[above]) * dy >= required[above]
    hanging[0] = False
    hanging[np.unique(labels[anchors[box]])] = False
    row = np.arange(narrow.shape[0])[:, None]
    lobes[box] = hanging[labels] & (row > last_contact[labels])
    return lobes


def _gravity_limit(co2, started, radius, lower, y, pitch_mm, settings):
    """Where free gas breaks gravity: below the lowest started port, and lobes.

    The started ports' lower zones are exempt from the first; CO2 reaching a
    started port's radius or lower zone is no sinking lobe. Returns both masks.
    """
    below = np.zeros(co2.shape, bool)
    anchors = np.zeros(co2.shape, bool)
    if started:
        exempt = np.logical_or.reduce([lower[port] for port in started])
        anchors = exempt | np.logical_or.reduce([radius[port] for port in started])
        below = (y[:, None] < min(port[1] for port in started)) & ~exempt
    return below, _sinking_fingers(co2, anchors, settings.finger_width_mm, pitch_mm)


def _pitch_mm(grid: darsia.Image) -> tuple[float, float]:
    """The grid's pitch [mm] down a column and along a row."""
    dy, dx = grid.voxel_size
    return dy * 1000, dx * 1000


def _port_zones(x, y, ports, settings):
    """Each port's radius disc and the lower half-ellipse below it."""
    radius, lower = {}, {}
    for port in ports:
        lateral = (x[None, :] - port[0]) * 1000
        height = (y[:, None] - port[1]) * 1000
        radius[port] = lateral**2 + height**2 <= settings.port_radius_mm**2
        lower[port] = (height <= 0) & (
            (lateral / settings.port_zone_half_width_mm) ** 2
            + (height / settings.port_zone_depth_mm) ** 2
            <= 1
        )
    return radius, lower


def _seal(facies, domain, permeability, settings):
    """Domain pixels of porous facies below the seal fraction of the highest."""
    porous = {
        k: v for k, v in permeability.items() if k not in settings.free_water_facies
    }
    limit = settings.seal_permeability_fraction * max(porous.values())
    return domain & np.isin(facies, [k for k, v in porous.items() if v < limit])


def _time_average_rules(before, labels):
    """The rule map of the time average: free gas lost and gained."""
    rules = np.zeros(labels.shape, np.uint8)
    lost = (before == PhaseLabel.GAS) & (labels == PhaseLabel.DISSOLVED)
    rules[lost] = PhaseRule.TIME_AVERAGE_LOST
    gained = (before == PhaseLabel.DISSOLVED) & (labels == PhaseLabel.GAS)
    rules[gained] = PhaseRule.TIME_AVERAGE_GAINED
    return rules


def phase_maps(
    probabilities: Sequence[tuple[np.ndarray, np.ndarray]],
    dates: Sequence[datetime],
    sand_map: darsia.Image,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    permeability: Mapping[int, float],
    protocol: darsia.InjectionProtocol,
    settings: PhaseRuleSettings,
    chart: np.ndarray | None = None,
    end_gas: Callable[[int], np.ndarray | None] | None = None,
    *,
    gas_threshold: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Phase maps of a run's frames, by its phase probabilities and the rules.

    Each frame's labels (:func:`phase_labels`) are corrected by the rules listed
    in the module description. A frame's time average uses its nearest
    ``neighbours_per_side`` earlier and later frames, read from
    ``probabilities`` by index. Each frame's rule map records which rule last
    changed each pixel's label. After an injection end, ``end_gas`` can weight
    the averaged and the frame's own gas probability by the free gas the run
    held at that end (:func:`darsia.free_gas` does); it is called with each
    frame's index when the frame is due, so it can use the earlier frames.

    Parameters
    ----------
    probabilities : sequence of (array, array)
        CO2-presence and gas-within-CO2 probabilities of each frame, in time
        order, on the grid of the sand map.
    dates : sequence of datetime
        Acquisition time of each frame.
    sand_map : darsia.Image
        Facies ids of the run's grid, with its physical dimensions and origin.
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel, as for :func:`phase_labels`.
    permeability : mapping of int to float
        Permeability of each porous facies, in any unit.
    protocol : darsia.InjectionProtocol
        Injection timetable; each injecting location is a port.
    settings : PhaseRuleSettings
        Settings of the labels and rules.
    chart : array, optional
        The colour checker's pixels.
    end_gas : callable, optional
        Weight per pixel of a frame's gas probability, by the frame's index, or
        None for no weight.
    gas_threshold : float
        Gas threshold of the phase model that gave the probabilities, as for
        :func:`phase_labels`.

    Yields
    ------
    tuple of arrays
        The phase map (uint8) and the rule map (uint8, :class:`PhaseRule`) of
        each frame, in order; the rule map is zero where no rule changed the
        label.

    """
    facies = sand_map.img
    x, y = darsia.pixel_centres(sand_map)
    pitch_mm = _pitch_mm(sand_map)

    injecting = protocol.df[protocol.df["rate_kg_s"] > 0]
    first_start = dict(injecting.groupby(["location_x", "location_y"])["start"].min())
    radius, lower = _port_zones(x, y, first_start, settings)

    free = np.isin(facies, settings.free_water_facies)
    region = None
    if chart is not None and chart.any():
        margin = int(round(settings.chart_margin_mm / pitch_mm[1]))
        extend = int(round(settings.chart_extend_below_mm / pitch_mm[0]))
        region = _chart_region(chart, margin, extend)
    seal = _seal(facies, domain, permeability, settings)
    open_ = domain & ~seal
    if region is not None:
        open_ &= ~region
    count, compartments = cv2.connectedComponents(
        open_.astype(np.uint8), connectivity=8
    )
    home = {
        port: compartments.flat[
            np.where(
                compartments > 0,
                ((y[:, None] - port[1]) * 1000) ** 2
                + ((x[None, :] - port[0]) * 1000) ** 2,
                np.inf,
            ).argmin()
        ]
        for port in first_start
    }

    for index, ((co2, gas), date) in enumerate(zip(probabilities, dates)):
        started = [port for port, start in first_start.items() if start <= date]
        earlier = [j for j in range(index - 1, -1, -1) if dates[j] < date]
        later = [j for j in range(index + 1, len(dates)) if dates[j] > date]
        neighbours = [
            (probabilities[j], abs((dates[j] - date).total_seconds()) / 3600)
            for j in earlier[: settings.neighbours_per_side]
            + later[: settings.neighbours_per_side]
        ]
        labels = phase_labels(
            co2, gas, domain, pixel_area_mm2, settings, gas_threshold=gas_threshold
        )
        before = labels.copy()
        weight = None if end_gas is None else end_gas(index)
        raw = _time_average(
            labels, gas, neighbours, domain, chart, settings, gas_threshold, weight
        )
        lowest = min((port[1] for port in started), default=None)
        _small_bubbles(labels, raw, lowest, y, pixel_area_mm2, settings)
        rules = _time_average_rules(before, labels)
        if region is not None:
            hidden = region & (labels != PhaseLabel.UNKNOWN)
            _change(labels, rules, hidden, PhaseLabel.UNKNOWN, PhaseRule.COLOR_CHECKER)
        _free_water(labels, rules, free)
        _minimum_area(labels, rules, domain, pixel_area_mm2, settings)
        sources = {home[port] for port in started} - {0}
        _compartments(labels, rules, compartments, count, seal, sources)
        below, lobes = _gravity_limit(
            _co2(labels), started, radius, lower, y, pitch_mm, settings
        )
        broken = (below | lobes) & (labels == PhaseLabel.GAS)
        _change(labels, rules, broken, PhaseLabel.DISSOLVED, PhaseRule.GRAVITY)
        yield labels, rules


def model_gas(labels: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """The free gas the phase model found in a frame, before gravity and the peel.

    The phase map's free gas and the free gas that the gravity limit of the phase
    rules (:attr:`PhaseRule.GRAVITY`) or the peel of a gas allocation
    (:attr:`PhaseRule.PEELED`) made dissolved CO2. Our production kept the
    water of those pixels saturated in its mass and left them out of its
    statistics of the dissolved CO2.

    Parameters
    ----------
    labels : array
        A frame's phase map (:class:`PhaseLabel`).
    rules : array
        Its rule map (:class:`PhaseRule`).

    Returns
    -------
    array
        The model's free gas (bool).

    """
    return (
        (labels == PhaseLabel.GAS)
        | (rules == PhaseRule.GRAVITY)
        | (rules == PhaseRule.PEELED)
    )
