"""Where the dye begins next to the static elements of a sand map.

A static element of a rig, e.g. silicone along a wall, the wall itself or a
fixture, holds no water. In a photograph of the rig whose pore water is dyed, the
porous layers turn dark, while the static element, and any light rim along its
edge, stays bright. The boundary between a static element and the porous layers of
a drawing (:class:`SandMapDrawing`) is then placed where the dye begins: at the
steepest drop of brightness from the static side into the porous side.

- :func:`static_boundaries` finds the boundaries between the polygons of static
  layers and those of porous layers, as chains of shared anchors;
- :func:`measure_dye_edge` finds where the dye begins along one such boundary, in
  an image placed in the rig by a homography;
- :func:`snap_to_dye_edge` moves the boundary onto the measured edge, so that a
  closed partition of the drawing's frame stays one, or refuses;
- :func:`dye_edge_stretches` lists where a drawing's boundary differs from the
  measured edge.

The method takes a boundary as straight over the chord that gives its direction,
over the window of a station's profiles and over the running median along it. It
needs boundaries that are smooth over those lengths, and rounds off a sharper
corner. Every distance and threshold of the method is a setting
(:class:`DyeEdgeSettings`).

"""

from dataclasses import dataclass
from typing import Collection, Literal, Sequence

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d

from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    _described,
    _fill_order,
    _project,
    _ring,
    partition_faults,
)
from darsia.presets.workflows.setup.sand_map_snapping import AnchorMove, _sides_of

_SEED = 0
"""Seed of each station's resampling, with the station's index, so that a
measurement repeats exactly and a station's spread depends on its own profiles
alone."""

Kind = Literal["reliable", "less certain", "interpolated", "not found", "unreadable"]
"""How a station's edge was found: from stable stations near it, from stations
of moderate spread near it, by interpolation along the boundary between stations
that have one, or not at all: not found where no station of the boundary has an
edge, and unreadable where too few of its profiles on either side of it are clear
of clipped and invalid pixels."""


@dataclass(frozen=True)
class DyeEdgeSettings:
    """Settings of finding where the dye begins along a static element's boundary;
    one configuration table, with distances in millimetres in the rig."""

    spacing_mm: float
    """Distance along the boundary between the stations where the edge is read;
    a snapped boundary's new anchors lie at the stations before simplification."""
    window_mm: float
    """Half the length along the boundary of the profiles whose median a station
    reads."""
    direction_mm: float
    """Half the length along the boundary of the chord whose direction is the
    boundary's at a station, and at a joint that snapping moves."""
    row_mm: float
    """Distance between the profiles of a station."""
    step_mm: float
    """Distance between the samples along a profile."""
    static_reach_mm: float
    """How far from the drawn boundary into the static side the drop is sought."""
    porous_reach_mm: float
    """How far from the drawn boundary into the porous side the drop is sought."""
    smoothing_mm: float
    """Standard deviation of the Gaussian that smooths a profile before its slope
    is taken."""
    rival_mm: float
    """Smallest distance from the steepest drop of a drop that rivals it."""
    rival_ratio: float
    """A station is stable only if no rival drop is this steep, as a fraction of
    the steepest."""
    resamples: int
    """Number of resamples of a station's profiles, drawn with replacement."""
    reliable_sd_mm: float
    """Largest spread of the drop under resampling at a stable station."""
    uncertain_sd_mm: float
    """Largest spread of a station that serves where too few stable ones are near."""
    median_mm: float
    """Half the length along the boundary of the running median of the stations."""
    fewest_profiles: int
    """Fewest profiles a station needs on each side of it, its own included, clear
    of clipped and invalid pixels; all of them on a side where fewer stay in the
    image."""
    fewest_stations: int
    """Fewest stations a running median is taken over."""
    crossing_mm: float
    """How far along the boundary from a joint that snapping moves the crossing of
    its line with the edge is sought."""
    simplify_mm: float
    """Largest distance of a snapped boundary from the measured edge, by which its
    new anchors are simplified."""
    tolerance_mm: float
    """Difference between a drawing's boundary and the edge beyond which a check
    reports it."""
    length_mm: float
    """Shortest stretch of such differences along the boundary that a check
    reports."""


@dataclass(frozen=True)
class DyeEdge:
    """Where the dye begins along a boundary of a static element, per station.

    The stations lie along the boundary at equal spacing from its first anchor.
    Positions along a station's normal are offsets in metres from the station's
    point on the drawn boundary, positive into the porous side.

    """

    chain: tuple[int, ...]
    """Anchors of the boundary, with the static polygon on their left."""
    path: np.ndarray
    """Positions (m, 2) of those anchors when measured, in metres."""
    along: np.ndarray
    """Distance of each station from the first anchor along the boundary, in
    metres."""
    base: np.ndarray
    """Each station's point (n, 2) on the drawn boundary, in metres."""
    normal: np.ndarray
    """Unit normal (n, 2) at each station, pointing into the porous side."""
    found: np.ndarray
    """Offset of the steepest drop at each station; NaN where none was read."""
    spread: np.ndarray
    """Standard deviation of that offset under resampling, in metres: the
    station's uncertainty; infinite where the median or a resample found no drop
    within the reach, NaN where unreadable."""
    rival: np.ndarray
    """Steepest rival drop as a fraction of the steepest drop; NaN where
    unreadable."""
    kind: tuple[Kind, ...]
    """How each station's edge was found."""
    offset: np.ndarray
    """Offset of the edge at each station; NaN where unreadable or not found."""

    @property
    def points(self) -> np.ndarray:
        """The edge (n, 2) in metres: each station's point moved along its normal
        by the edge's offset."""
        return self.base + self.offset[:, None] * self.normal


@dataclass(frozen=True)
class EdgeStretch:
    """Consecutive stations where a drawing's boundary differs from the measured
    edge by more than the tolerance, all to the same side."""

    first: int
    """First station of the stretch; on a closed boundary, a stretch across its
    first station has a first station after its last."""
    last: int
    """Last station of the stretch."""
    mean: float
    """Mean difference in metres; positive where the drawing's static element
    reaches beyond the edge, i.e. where porous layers are drawn as static."""
    largest: float
    """Difference of the largest magnitude, in metres."""
    layers: tuple[int, ...]
    """Porous layers at the drawing's boundary along the stretch."""


def static_boundaries(
    drawing: SandMapDrawing, static: Collection[int]
) -> list[tuple[int, ...]]:
    """The boundaries between the static and the porous layers of a drawing.

    A boundary is a chain of consecutive edges of a polygon of a static layer that
    polygons of porous layers, all others, share in the opposite direction. Its
    anchors run along the static polygon's counter-clockwise boundary, so the
    static polygon lies on their left and the porous side on their right. A static
    polygon that porous polygons surround entirely gives a closed chain, from its
    first anchor that three or more polygons use (or else its first anchor) round
    to that anchor again. Polygons of layers drawn on top lie over the partition
    and are left out.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing.
    static : collection of int
        The layers of static elements, which hold no water.

    Returns
    -------
    list of tuple of int
        The chains of two or more anchors, by static polygon in drawing order.

    """
    points = np.asarray(drawing.anchors, dtype=np.float64)
    rings = [
        [] if layer in drawing.on_top else _ring(points, indices)
        for layer, indices in drawing.polygons
    ]
    layers = [layer for layer, _indices in drawing.polygons]
    porous = {
        (start, end)
        for layer, ring in zip(layers, rings)
        if layer not in static
        for start, end in zip(ring, ring[1:] + ring[:1])
    }
    users = _users(rings)
    chains = []
    for layer, ring in zip(layers, rings):
        if layer not in static or not ring:
            continue
        n = len(ring)
        shared = [(ring[(i + 1) % n], ring[i]) in porous for i in range(n)]
        if all(shared):
            first = next((i for i in range(n) if len(users[ring[i]]) > 2), 0)
            chains.append(tuple(ring[first:] + ring[: first + 1]))
            continue
        for i in range(n):
            if shared[i] and not shared[i - 1]:
                chain = [ring[i]]
                while shared[(i + len(chain) - 1) % n]:
                    chain.append(ring[(i + len(chain)) % n])
                chains.append(tuple(chain))
    return chains


def measure_dye_edge(
    drawing: SandMapDrawing,
    chain: Sequence[int],
    image: np.ndarray,
    rig_to_image: np.ndarray,
    settings: DyeEdgeSettings,
    *,
    clip_level: float,
) -> DyeEdge:
    """Where the dye begins along a boundary of a static element, in an image.

    Stations lie along the boundary every ``spacing_mm``, the first half a spacing
    from its first anchor. At each, the boundary's direction is the chord between
    its points ``direction_mm`` before and after (within the boundary, or round a
    closed one). Profiles run along the normal to that chord, from
    ``static_reach_mm`` on the static side to ``porous_reach_mm`` on the porous
    side, sampled every ``step_mm``; they start every ``row_mm`` along the chord
    within ``window_mm`` of the station (and of the boundary's ends). The
    brightness is the mean of the image's channels, by bilinear interpolation in
    the image's floating dtype.

    A profile with a sample that leaves the image or draws on a pixel that is not
    finite or has a channel at or above ``clip_level`` is left out. A station is
    readable if a profile is left and, on each side of it along the chord (its own
    profile on both), ``fewest_profiles`` of the profiles that stay in the image
    are left, or all of them where fewer stay in it. At a readable station the
    median of its profiles is smoothed by a Gaussian of ``smoothing_mm``, and the
    drop is the steepest fall of brightness within the reach, refined by a
    parabola through its neighbours; where the steepest fall lies at either end of
    the reach, the drop lies there or beyond, and the station has none. The drop
    is sought again in ``resamples`` medians of the profiles drawn with
    replacement, by the station's own random draws, and their standard deviation
    is the station's spread, infinite if the median or a resample has no drop. A
    station is stable if its spread is at most ``reliable_sd_mm`` and no drop
    farther than ``rival_mm`` from the steepest is ``rival_ratio`` as steep.

    The edge at a readable station is the median, along its normal, of the drops
    of the stations within ``median_mm`` along the boundary (round a closed one):
    of the stable ones if at least ``fewest_stations`` are (reliable), or else of
    those with a spread of at most ``uncertain_sd_mm`` if at least
    ``fewest_stations`` are (less certain). Elsewhere the edge's offset is
    interpolated along the boundary between the stations that have one (round a
    closed one), and held beyond the last; where no station has one, the edge is
    not found.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing.
    chain : sequence of int
        The boundary's anchors, with the static polygon on their left, e.g. from
        :func:`static_boundaries`.
    image : array
        Image (rows, columns) or (rows, columns, channels) of the rig with dyed
        pore water; NaN where invalid.
    rig_to_image : array
        3 x 3 homography from rig coordinates (x, y in metres) to the image's
        pixel coordinates (column, row), with pixel centres at whole numbers.
    settings : DyeEdgeSettings
        The method's settings.
    clip_level : float
        Value at or above which a channel is clipped.

    Returns
    -------
    DyeEdge
        The edge per station.

    Raises
    ------
    ValueError
        If the boundary has fewer than two anchors or no length, or is closed and
        no longer than twice ``direction_mm`` or ``window_mm``.

    """
    path = np.asarray(drawing.anchors, dtype=np.float64)[list(chain)]
    if len(path) < 2 or not np.hypot(*np.diff(path, axis=0).T).sum() > 0:
        raise ValueError("A boundary needs two or more anchors and a length.")
    arc = _arc(path)
    closed = chain[0] == chain[-1]
    if (
        closed
        and not 2 * max(settings.direction_mm, settings.window_mm) < arc[-1] * 1000
    ):
        raise ValueError(
            "A closed boundary needs to be longer than twice the chord's and the "
            "window's half-length."
        )
    spacing = settings.spacing_mm / 1000
    along = (np.arange(int(np.floor(arc[-1] / spacing + 0.5))) + 0.5) * spacing
    base, tangent, normal = _frames(path, along, settings.direction_mm / 1000)

    step = settings.step_mm / 1000
    margin = int(np.ceil(4 * settings.smoothing_mm / settings.step_mm)) + 1
    low = int(round(settings.static_reach_mm / settings.step_mm))
    high = int(round(settings.porous_reach_mm / settings.step_mm))
    t = np.arange(-low - margin, high + margin + 1) * step
    search = np.arange(margin, margin + low + high + 1)
    rows = int(np.floor(settings.window_mm / settings.row_mm + 1e-9))
    shifts = np.arange(-rows, rows + 1) * settings.row_mm / 1000

    count = len(along)
    found, spread, rival = np.full((3, count), np.nan)
    for k in range(count):
        within = closed | ((along[k] + shifts >= 0) & (along[k] + shifts <= arc[-1]))
        shift = shifts[within]
        samples = base[k] + shift[:, None, None] * tangent[k] + t[:, None] * normal[k]
        pixels = _project(samples.reshape(-1, 2), rig_to_image)
        values, inside, valid = _sample(image, pixels, clip_level)
        present = inside.reshape(len(shift), len(t)).all(axis=1)
        usable = valid.reshape(len(shift), len(t)).all(axis=1)
        if not usable.any() or any(
            np.count_nonzero(usable & side)
            < min(settings.fewest_profiles, np.count_nonzero(present & side))
            for side in (shift <= 0, shift >= 0)
        ):
            continue
        brightness = values.mean(axis=-1).reshape(len(shift), len(t))[usable]
        (found[k],), (rival[k],) = _steepest(
            np.median(brightness, axis=0)[None], t, search, settings
        )
        picks = np.random.default_rng((_SEED, k)).integers(
            0, len(brightness), (settings.resamples, len(brightness))
        )
        drops, _rivals = _steepest(
            np.median(brightness[picks], axis=1), t, search, settings
        )
        dropped = np.isfinite(found[k]) and np.all(np.isfinite(drops))
        spread[k] = np.std(drops) if dropped else np.inf

    readable = ~np.isnan(spread)
    stable = (spread <= settings.reliable_sd_mm / 1000) & (rival < settings.rival_ratio)
    fair = spread <= settings.uncertain_sd_mm / 1000
    reach = int(np.floor(settings.median_mm / settings.spacing_mm + 1e-9))
    drop = base + found[:, None] * normal
    offset = np.full(count, np.nan)
    kind: list[Kind] = ["unreadable"] * count
    for k in np.flatnonzero(readable):
        window = np.arange(k - reach, k + reach + 1)
        window = window % count if closed else window[(window >= 0) & (window < count)]
        near = np.zeros(count, dtype=bool)
        near[window] = True
        for group, name in ((stable, "reliable"), (fair, "less certain")):
            if np.count_nonzero(near & group) >= settings.fewest_stations:
                offset[k] = np.median((drop[near & group] - base[k]) @ normal[k])
                kind[k] = name
                break
    known = np.isfinite(offset)
    missing = np.flatnonzero(readable & ~known)
    if known.any():
        offset[missing] = np.interp(
            along[missing],
            along[known],
            offset[known],
            period=arc[-1] if closed else None,
        )
    for k in missing:
        kind[k] = "interpolated" if known.any() else "not found"
    return DyeEdge(
        tuple(int(i) for i in chain),
        path,
        along,
        base,
        normal,
        found,
        spread,
        rival,
        tuple(kind),
        offset,
    )


def snap_to_dye_edge(
    drawing: SandMapDrawing, edge: DyeEdge, settings: DyeEdgeSettings
) -> tuple[SandMapDrawing, list[AnchorMove], list[int]]:
    """Move a static element's boundary onto the measured edge.

    The boundary's joints are its ends and its anchors that three or more polygons
    use, where the porous neighbour changes. The edge is readable in stretches:
    each stretch runs through its stations and on by one spacing along the
    boundary beyond its ends, and covers the boundary from half a spacing before
    its first station to half a spacing after its last (to the boundary's end
    where its first or last station is the boundary's). A joint the edge covers
    moves onto it at its own position: along the boundary's normal there, or
    along the frame's side for a joint on it, to where that line crosses the
    stretch nearest the joint, within ``crossing_mm`` of it along the boundary. A
    joint stays if it lies at a corner of the frame, if its line does not cross,
    or if a polygon other than the static one and its neighbours along the
    boundary uses it, e.g. another static element, whose boundaries the edge does
    not measure. Every other anchor the edge covers leaves the boundary. Between
    the anchors that stay, the static polygon and its neighbour share new anchors
    at the stations whose edge lies in the frame, simplified to within
    ``simplify_mm`` of the edge; where the edge is unreadable, the boundary keeps
    its anchors.

    An anchor of another boundary that leaves a moved joint, used only by polygons
    that use the joint, and that now lies in the static polygon or on its edge, is
    removed, so that boundary starts at the joint; and so on along that boundary.
    Polygons of layers drawn on top lie over the partition: they are no neighbours
    or users here, and keep their anchors. The drawing that results must be a
    closed partition of its frame, so a drawing that is one stays one, and
    snapping refuses where it would not.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing the edge was measured on, with its frame.
    edge : DyeEdge
        The edge along one of the drawing's static boundaries
        (:func:`measure_dye_edge`).
    settings : DyeEdgeSettings
        The method's settings.

    Returns
    -------
    SandMapDrawing
        The snapped drawing, its polygons counter-clockwise and without repeated
        anchors; anchor indices are kept, and added anchors follow the others.
    list of AnchorMove
        Every joint moved and every anchor added.
    list of int
        The anchors that no polygon uses any more.

    Raises
    ------
    ValueError
        If the drawing has no frame; if the boundary's anchors are not where the
        edge was measured, or a polygon of a layer drawn on top uses them; if an
        anchor that a moved joint leaves in the static polygon is also used by a
        polygon that does not use the joint, or the static polygon would cover a
        porous polygon; or if the snapped drawing is not a closed partition of
        the frame (:func:`partition_faults`), e.g. since the drawing was not one.

    """
    if drawing.frame is None:
        raise ValueError("The drawing has no frame to keep partitioned.")
    points = np.array(drawing.anchors, dtype=np.float64)
    chain = list(edge.chain)
    if not np.array_equal(points[chain], edge.path):
        raise ValueError("The boundary's anchors are not where the edge was measured.")
    layers = [layer for layer, _indices in drawing.polygons]
    rings = [_ring(points, indices) for _layer, indices in drawing.polygons]
    lying = {
        layer
        for layer, ring in zip(layers, rings)
        if layer in drawing.on_top and set(ring) & set(chain)
    }
    if lying:
        raise ValueError(
            f"Polygons of layers {sorted(lying)}, drawn on top, use anchors of the "
            "boundary, and snapping would not move them with it."
        )
    partition = [
        [] if layer in drawing.on_top else ring for layer, ring in zip(layers, rings)
    ]
    users = _users(partition)
    owner = {
        (start, end): k
        for k, ring in enumerate(partition)
        for start, end in zip(ring, ring[1:] + ring[:1])
    }
    static = owner[(chain[0], chain[1])]
    beside = {static} | {owner[(end, start)] for start, end in zip(chain, chain[1:])}
    arc = _arc(edge.path)
    spacing = settings.spacing_mm / 1000
    chord = settings.direction_mm / 1000
    window = settings.crossing_mm / 1000
    line = edge.points
    count = len(edge.along)
    known = np.isfinite(edge.offset)
    inside = np.all((line >= 0) & (line <= drawing.frame), axis=1)
    stretches = [
        run
        for run in np.split(np.arange(count), np.flatnonzero(np.diff(known)) + 1)
        if run.size and known[run[0]]
    ]
    polylines = []
    for run in stretches:
        _base, (before, after), _normal = _frames(
            edge.path, edge.along[run[[0, -1]]], chord
        )
        polylines.append(
            np.vstack(
                [
                    line[run[0]] - spacing * before,
                    line[run],
                    line[run[-1]] + spacing * after,
                ]
            )
        )

    def placed(i: int) -> tuple[np.ndarray, int, int] | None:
        """Where the boundary's i-th anchor moves on the edge, with the stations
        before and after it there; None if it stays."""
        here = arc[i]
        for run, polyline in zip(stretches, polylines):
            start = 0.0 if run[0] == 0 else edge.along[run[0]] - spacing / 2
            end = arc[-1] if run[-1] == count - 1 else edge.along[run[-1]] + spacing / 2
            if start <= here <= end:
                break
        else:
            return None
        places = np.concatenate(
            [
                edge.along[run[[0]]] - spacing,
                edge.along[run],
                edge.along[run[[-1]]] + spacing,
            ]
        )
        first = max(int(np.searchsorted(places, here - window)) - 1, 0)
        polyline = polyline[first : int(np.searchsorted(places, here + window)) + 1]
        sides = _sides_of(points[chain[i]], drawing.frame)
        if len(sides) > 1:
            return None
        if sides:
            direction = (
                np.array([0.0, 1.0]) if sides.pop() % 2 else np.array([1.0, 0.0])
            )
        else:
            direction = _frames(edge.path, np.array([here]), chord)[2][0]
        crossing = _crossing(points[chain[i]], direction, polyline)
        if crossing is None:
            return None
        segment, position = crossing
        return position, run[0] + first + segment - 1, run[0] + first + segment

    moved: dict[int, tuple[float, float]] = {}
    kept: list[tuple[int, int | None, int | None]] = []
    for i, anchor in enumerate(chain):
        stays = (i > 0 and anchor == chain[0]) or not users[anchor] <= beside
        where = None if stays else placed(i)
        if where is None:
            kept.append((i, None, None))
        elif i in (0, len(chain) - 1) or len(users[anchor]) > 2:
            moved[anchor] = tuple(points[anchor].tolist())
            points[anchor] = where[0]
            kept.append((i, where[1], where[2]))

    added: list[np.ndarray] = []
    for (i, _before, after), (j, before, _after) in zip(kept, kept[1:]):
        if after is None:
            after = int(np.searchsorted(edge.along, arc[i], side="right"))
        if before is None:
            before = int(np.searchsorted(edge.along, arc[j], side="left")) - 1
        stations = range(max(after, 0), min(before, count - 1) + 1)
        stations = [k for k in stations if known[k] and inside[k]]
        polyline = np.vstack([points[chain[i]], line[stations], points[chain[j]]])
        simple = shapely.get_coordinates(
            shapely.simplify(
                shapely.LineString(polyline),
                settings.simplify_mm / 1000,
                preserve_topology=False,
            )
        )[1:-1]
        new = list(
            range(len(points) + len(added), len(points) + len(added) + len(simple))
        )
        added.extend(simple)
        neighbour = owner[(chain[i + 1], chain[i])]
        _replace(rings[static], chain[i : j + 1], new)
        _replace(rings[neighbour], chain[i : j + 1][::-1], new[::-1])

    if added:
        points = np.vstack([points, added])
    users = _users(partition)
    inside = shapely.Polygon(points[rings[static]])
    for joint in sorted(moved):
        porous = users[joint] - {static}
        for k in sorted(porous):
            for direction in (1, -1):
                while True:
                    ring = rings[k]
                    near = ring[(ring.index(joint) + direction) % len(ring)]
                    if static in users[near] or not inside.covers(
                        shapely.Point(points[near])
                    ):
                        break
                    if not users[near] <= porous:
                        others = sorted({layers[n] for n in users[near] - porous})
                        raise ValueError(
                            f"Anchor {near} lies in the static polygon once anchor "
                            f"{joint} has moved onto the edge, and layers {others} "
                            f"use it but not anchor {joint}."
                        )
                    for n in users[near]:
                        rest = [index for index in rings[n] if index != near]
                        if len(rest) < 3 or set(rest) <= set(rings[static]):
                            raise ValueError(
                                f"The static polygon would cover the polygon of "
                                f"layer {layers[n]} once anchor {joint} has moved "
                                "onto the edge."
                            )
                    for n in users.pop(near):
                        rings[n].remove(near)

    moves = [
        AnchorMove(anchor, start, tuple(points[anchor].tolist()))
        for anchor, start in sorted(moved.items())
    ]
    moves += [
        AnchorMove(index, None, tuple(points[index].tolist()))
        for index in range(len(drawing.anchors), len(points))
    ]
    polygons = tuple(
        (layer, tuple(ring)) for (layer, _indices), ring in zip(drawing.polygons, rings)
    )
    snapped = SandMapDrawing(points, polygons, drawing.on_top, drawing.frame)
    faults = partition_faults(snapped)
    if faults:
        raise ValueError(
            "The snapped drawing is not a closed partition of its frame: "
            + _described(faults)
            + "."
        )
    used = {index for _layer, indices in drawing.polygons for index in indices}
    return snapped, moves, sorted(used - {index for ring in rings for index in ring})


def dye_edge_stretches(
    drawing: SandMapDrawing,
    edge: DyeEdge,
    static: Collection[int],
    settings: DyeEdgeSettings,
) -> list[EdgeStretch]:
    """Where a drawing's boundary of a static element differs from the measured
    edge.

    At each station with an edge, the drawing's boundary lies where, along the
    station's normal through the edge and within the frame, the layer the drawing
    shows turns porous: at the start of the porous stretch the edge lies in, or
    else of the first one beyond it. A point shows the layer of the polygon filled
    last over it, as the rasteriser fills them (larger polygons first, those on
    top last); porous layers are neither static nor on top. The difference at a
    station is the boundary's offset from the edge, positive into the porous side.
    A stretch is a run of consecutive stations whose differences exceed
    ``tolerance_mm``, all with the same sign, at least ``length_mm`` long (one
    spacing per station); round a closed boundary, a run may pass its first
    station. Stations without an edge, or along whose normal the drawing shows no
    porous layer, are left out and end a run.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, with its frame; it need not be a partition.
    edge : DyeEdge
        The measured edge (:func:`measure_dye_edge`), possibly measured along
        another drawing's boundary.
    static : collection of int
        The layers of static elements.
    settings : DyeEdgeSettings
        The method's settings.

    Returns
    -------
    list of EdgeStretch
        The stretches, in the order of their first stations, except that one
        across a closed boundary's first station comes last.

    Raises
    ------
    ValueError
        If the drawing has no frame.

    """
    if drawing.frame is None:
        raise ValueError("The drawing has no frame to look along.")
    width, height = drawing.frame
    points = np.asarray(drawing.anchors, dtype=np.float64)
    origin, normal = edge.points, edge.normal
    with np.errstate(divide="ignore", invalid="ignore"):
        ends = np.stack([-origin / normal, ([width, height] - origin) / normal])
    ends = np.where(np.isnan(ends), [[[-np.inf]], [[np.inf]]], ends)
    first, last = ends.min(axis=0).max(axis=1), ends.max(axis=0).min(axis=1)
    stations = np.flatnonzero(np.isfinite(edge.offset) & (first < 0) & (0 < last))
    if not stations.size:
        return []
    origin, normal = origin[stations], normal[stations]
    first, last = first[stations], last[stations]
    lines = shapely.linestrings(
        np.stack(
            [origin + first[:, None] * normal, origin + last[:, None] * normal], axis=1
        )
    )

    order = _fill_order(drawing)
    rings = [points[_ring(points, drawing.polygons[k][1])] for k in order]
    shapes = [shapely.make_valid(shapely.Polygon(ring)) for ring in rings]
    cuts = [[a, b] for a, b in zip(first, last)]
    for ring in rings:
        crossed = shapely.intersection(lines, shapely.LinearRing(ring))
        coordinates, which = shapely.get_coordinates(crossed, return_index=True)
        offsets = np.einsum("ij,ij->i", coordinates - origin[which], normal[which])
        for n, value in zip(which.tolist(), offsets.tolist()):
            cuts[n].append(value)
    cuts = [np.unique(np.clip(values, a, b)) for values, a, b in zip(cuts, first, last)]
    middles = np.concatenate([0.5 * (c[1:] + c[:-1]) for c in cuts])
    which = np.repeat(np.arange(len(cuts)), [len(c) - 1 for c in cuts])
    where = origin[which] + middles[:, None] * normal[which]
    shown = np.full(len(middles), -1)
    for k, shape in zip(order, shapes):
        shown[shapely.contains_xy(shape, where[:, 0], where[:, 1])] = k
    layer = np.array([layer for layer, _indices in drawing.polygons] + [0])[shown]
    porous = (shown >= 0) & ~np.isin(layer, [*static, *drawing.on_top])

    difference = np.full(len(edge.along), np.nan)
    boundary = np.zeros(len(edge.along), dtype=int)
    start = 0
    for n, values in enumerate(cuts):
        flags = porous[start : start + len(values) - 1]
        layers = layer[start : start + len(values) - 1]
        start += len(values) - 1
        here = int(np.searchsorted(values, 0.0, side="right")) - 1
        if not 0 <= here < len(flags):
            continue
        if flags[here]:
            while here > 0 and flags[here - 1]:
                here -= 1
        else:
            later = np.flatnonzero(flags[here:])
            if not later.size:
                continue
            here += int(later[0])
        difference[stations[n]] = values[here]
        boundary[stations[n]] = layers[here]

    tolerance = settings.tolerance_mm / 1000
    side = np.where(np.abs(difference) > tolerance, np.sign(difference), 0)
    runs = np.split(np.arange(len(side)), np.flatnonzero(np.diff(side)) + 1)
    if edge.chain[0] == edge.chain[-1] and len(runs) > 1 and side[0] == side[-1]:
        runs = runs[1:-1] + [np.concatenate([runs[-1], runs[0]])]
    result = []
    for run in runs:
        if side[run[0]] and len(run) * settings.spacing_mm + 1e-9 >= settings.length_mm:
            values = difference[run]
            result.append(
                EdgeStretch(
                    int(run[0]),
                    int(run[-1]),
                    float(values.mean()),
                    float(values[np.argmax(np.abs(values))]),
                    tuple(sorted(set(boundary[run].tolist()))),
                )
            )
    return result


def _users(rings: list[list[int]]) -> dict[int, set[int]]:
    """The polygons that use each anchor, by index."""
    users: dict[int, set[int]] = {}
    for k, ring in enumerate(rings):
        for anchor in ring:
            users.setdefault(anchor, set()).add(k)
    return users


def _arc(path: np.ndarray) -> np.ndarray:
    """Distance of each point of a polyline from its first point along it."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def _frames(
    path: np.ndarray, along: np.ndarray, window: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points of a polyline at distances along it, the unit direction of the chord
    between its points a window before and after each (within the polyline, or
    round it if it is closed), and the normal to that chord on its right."""
    arc = _arc(path)
    if np.array_equal(path[0], path[-1]):
        ahead, behind = (along + window) % arc[-1], (along - window) % arc[-1]
    else:
        ahead, behind = np.minimum(along + window, arc[-1]), np.maximum(
            along - window, 0
        )
    chord = np.column_stack([np.interp(ahead, arc, v) for v in path.T])
    chord -= np.column_stack([np.interp(behind, arc, v) for v in path.T])
    at = [np.interp(along, arc, value) for value in path.T]
    tangent = chord / np.hypot(*chord.T)[:, None]
    return (
        np.column_stack(at),
        tangent,
        np.column_stack([tangent[:, 1], -tangent[:, 0]]),
    )


def _sample(
    image: np.ndarray, pixels: np.ndarray, clip_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An image's channels at pixel coordinates (column, row) by bilinear
    interpolation, in its floating dtype; whether each sample lies in the image,
    drawing on four of its pixels; and whether it is valid: in the image, with
    every channel of those pixels finite and below the clip level."""
    values = image.reshape(image.shape[:2] + (-1,))
    dtype = np.result_type(image.dtype, np.float32)
    column, row = np.floor(pixels[:, 0]), np.floor(pixels[:, 1])
    across = (pixels[:, 0] - column).astype(dtype)[:, None]
    down = (pixels[:, 1] - row).astype(dtype)[:, None]
    inside = (column >= 0) & (row >= 0)
    inside &= (column <= values.shape[1] - 2) & (row <= values.shape[0] - 2)
    valid = inside.copy()
    column = np.where(inside, column, 0).astype(np.intp)
    row = np.where(inside, row, 0).astype(np.intp)
    result = np.zeros((len(pixels), values.shape[2]), dtype=dtype)
    for down_by, across_by, weight in (
        (0, 0, (1 - down) * (1 - across)),
        (0, 1, (1 - down) * across),
        (1, 0, down * (1 - across)),
        (1, 1, down * across),
    ):
        corner = values[row + down_by, column + across_by].astype(dtype, copy=False)
        valid &= np.all(corner < clip_level, axis=1)
        result += weight * corner
    return result, inside, valid


def _steepest(
    profiles: np.ndarray,
    offsets: np.ndarray,
    search: np.ndarray,
    settings: DyeEdgeSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset (m) of the steepest drop of each smoothed profile among the searched
    samples, refined by a parabola through it and its neighbours; NaN if nothing
    drops, or if the steepest searched sample is the first or last searched,
    where the drop lies at the search's end or beyond it. Also the steepest drop
    farther than the rival distance from it, as a fraction of it (1 if nothing
    drops). The samples lie at the offsets, one step apart, with samples beyond
    the search on both sides."""
    step = settings.step_mm / 1000
    smooth = gaussian_filter1d(
        profiles, settings.smoothing_mm / settings.step_mm, axis=-1
    )
    slope = -np.gradient(smooth, step, axis=-1)
    index = np.arange(len(profiles))
    best = search[np.argmax(slope[:, search], axis=1)]
    peak = slope[index, best]
    far = np.abs(search[None, :] - best[:, None]) * step > settings.rival_mm / 1000
    other = np.where(far, slope[:, search], -np.inf).max(axis=1)
    dropping = np.isfinite(other) & (peak > 0)
    rival = np.where(dropping, other / np.where(dropping, peak, 1.0), 1.0)
    drop = (peak > 0) & (best > search[0]) & (best < search[-1])
    before, after = slope[index, best - 1], slope[index, best + 1]
    bend = np.where(drop, before - 2 * peak + after, -1.0)
    shift = 0.5 * (before - after) / bend
    return np.where(drop, offsets[best] + shift * step, np.nan), rival


def _crossing(
    origin: np.ndarray, direction: np.ndarray, polyline: np.ndarray
) -> tuple[int, np.ndarray] | None:
    """Where the line through a point along a direction crosses a polyline, nearest
    the point: the segment and the position; None if it does not cross."""
    start, span = polyline[:-1], np.diff(polyline, axis=0)
    gap = start - origin
    det = direction[0] * span[:, 1] - direction[1] * span[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (gap[:, 0] * span[:, 1] - gap[:, 1] * span[:, 0]) / det
        fraction = (gap[:, 0] * direction[1] - gap[:, 1] * direction[0]) / det
    hits = np.flatnonzero((det != 0) & (fraction >= 0) & (fraction <= 1))
    if not hits.size:
        return None
    segment = int(hits[np.argmin(np.abs(distance[hits]))])
    return segment, origin + distance[segment] * direction


def _replace(ring: list[int], run: Sequence[int], new: Sequence[int]) -> None:
    """Replace the anchors between the ends of a run of consecutive anchors of a
    ring, in its direction, by new ones; a run from an anchor round to it again
    replaces all others."""
    first = ring.index(run[0])
    turned = ring[first:] + ring[:first]
    if (turned + turned[:1])[: len(run)] != list(run):
        raise ValueError(f"The polygon does not run through anchors {list(run)}.")
    ring[:] = [run[0], *new, *turned[len(run) - 1 :]]
