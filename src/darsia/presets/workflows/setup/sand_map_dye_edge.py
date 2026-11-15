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
  an image placed in the rig by a homography.

The method takes a boundary as straight over the chord that gives its direction,
over the window of a station's profiles and over the running median along it. It
needs boundaries that are smooth over those lengths, and rounds off a sharper
corner. Every distance and threshold of the method is a setting
(:class:`DyeEdgeSettings`).

"""

from dataclasses import dataclass
from typing import Collection, Literal, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

from darsia.presets.workflows.setup.sand_map import SandMapDrawing, _project, _ring

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
    """Distance along the boundary between the stations where the edge is read."""
    window_mm: float
    """Half the length along the boundary of the profiles whose median a station
    reads."""
    direction_mm: float
    """Half the length along the boundary of the chord whose direction is the
    boundary's at a station."""
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
