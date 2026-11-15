"""Tests of placing a sand map's static boundaries where the dye begins."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import erf

from darsia.presets.workflows.setup.sand_map import SandMapDrawing, partition_faults
from darsia.presets.workflows.setup.sand_map_dye_edge import (
    DyeEdgeSettings,
    measure_dye_edge,
    static_boundaries,
)

WIDTH, HEIGHT = 0.04, 0.03
"""Inner frame of the test rig in metres."""

PIXEL = 1e-4
"""Pixel size of the test photographs in metres."""

RIG_TO_IMAGE = np.array(
    [[1 / PIXEL, 0, -0.5], [0, -1 / PIXEL, HEIGHT / PIXEL - 0.5], [0, 0, 1]]
)
"""Rig metres to the photographs' pixels (column, row), the top row first."""

SETTINGS = DyeEdgeSettings(
    spacing_mm=0.5,
    window_mm=2.0,
    direction_mm=50.0,
    row_mm=0.1,
    step_mm=0.02,
    static_reach_mm=2.0,
    porous_reach_mm=2.5,
    smoothing_mm=0.15,
    rival_mm=0.5,
    rival_ratio=0.7,
    resamples=40,
    reliable_sd_mm=0.3,
    uncertain_sd_mm=0.5,
    median_mm=3.0,
    fewest_profiles=5,
    fewest_stations=5,
)
"""The settings a rig's side walls were drawn with."""

ROUND = replace(SETTINGS, window_mm=1.0, direction_mm=1.0, median_mm=1.0)
"""The settings for a round fixture 8 mm in radius."""


def _step(z: np.ndarray) -> np.ndarray:
    """A step at zero (m), blurred over 0.1 mm."""
    return 0.5 * (1 + erf(z / 1e-4))


def _shot(light, noise: float = 0.01, margin: int = 0) -> np.ndarray:
    """A photograph of the test rig in three channels from the brightness at each
    point (x, y in metres), with noise, reaching a margin of pixels beyond the
    frame."""
    x = (np.arange(round(WIDTH / PIXEL) + 2 * margin) + 0.5 - margin) * PIXEL
    y = HEIGHT - (np.arange(round(HEIGHT / PIXEL) + 2 * margin) + 0.5 - margin) * PIXEL
    x, y = np.meshgrid(x, y)
    scatter = np.random.default_rng(0).normal(0, noise, x.shape + (3,))
    return (light(x, y)[..., None] * [1.0, 0.9, 0.8] + scatter).astype(np.float32)


def _photograph(distance, rival=None, undyed=None, margin=0) -> np.ndarray:
    """A photograph of the rig with dyed pore water: the static element (0.6) with
    a light rim 0.6 mm wide (0.9), then the dyed sand (0.15). ``distance`` gives a
    point's signed distance (m) from where the dye begins, positive into the sand.
    Where ``rival`` holds, the sand darkens in two steps 1 mm apart; where
    ``undyed`` holds, the sand is as light as the rim."""

    def light(x, y):
        """Brightness at each point."""
        d = distance(x, y)
        first, second = np.full(d.shape, 0.75), np.zeros(d.shape)
        if rival is not None:
            first[rival(x, y)], second[rival(x, y)] = 0.4, 0.35
        if undyed is not None:
            first[undyed(x, y)] = 0.0
        level = 0.6 + 0.3 * _step(d + 6e-4) - first * _step(d)
        return level - second * _step(d - 1e-3)

    return _shot(light, margin=margin)


def _edge_x(y: np.ndarray) -> np.ndarray:
    """Where the dye begins along the left wall's band (m), by height (m): a
    straight line, slightly tilted."""
    return 0.007 + 0.02 * (y - 0.015)


def _wall() -> SandMapDrawing:
    """A static band along the left wall (layer 1), drawn 0.5 to 1.1 mm short of
    where the dye begins, and two sand layers meeting it at a joint at y 15 mm.
    Their boundary has an anchor 0.3 mm from the joint, inside the band as the dye
    shows it. A layer on top lies in the sand."""
    anchors = [[0, 0], [6.2, 0], [6.1, 7], [6.0, 15], [6.3, 23], [6.6, 30], [0, 30]]
    anchors += [[40, 0], [40, 15], [6.3, 15], [40, 30]]
    anchors += [[20, 20], [30, 20], [30, 25], [20, 25]]
    polygons = (
        (1, (0, 1, 2, 3, 4, 5, 6)),
        (2, (1, 7, 8, 9, 3, 2)),
        (3, (3, 9, 8, 10, 5, 4)),
        (9, (11, 12, 13, 14)),
    )
    return SandMapDrawing(
        np.array(anchors) / 1000, polygons, frozenset({9}), (WIDTH, HEIGHT)
    )


def _disc() -> SandMapDrawing:
    """A round static fixture (layer 1), 8 mm in radius as the dye shows it but
    drawn 0.4 mm smaller, between two sand layers that meet it at both ends of its
    horizontal diameter."""
    angle = 2 * np.pi * np.arange(32) / 32
    anchors = np.column_stack([20 + 7.6 * np.cos(angle), 15 + 7.6 * np.sin(angle)])
    anchors = np.vstack([anchors, [[0, 0], [40, 0], [40, 15], [40, 30], [0, 30]]])
    anchors = np.vstack([anchors, [[0, 15]]])
    polygons = (
        (1, tuple(range(32))),
        (2, (32, 33, 34, 0, *range(31, 15, -1), 37)),
        (3, (37, *range(16, -1, -1), 34, 35, 36)),
    )
    return SandMapDrawing(anchors / 1000, polygons, frame=(WIDTH, HEIGHT))


def _band(x: float) -> SandMapDrawing:
    """A static band along the left wall (layer 1) whose straight inner boundary
    lies x mm from the wall, and sand (layer 2) beyond it."""
    anchors = np.array([[0, 0], [x, 0], [x, 30], [0, 30], [40, 0], [40, 30]]) / 1000
    polygons = ((1, (0, 1, 2, 3)), (2, (1, 4, 5, 2)))
    return SandMapDrawing(anchors, polygons, frame=(WIDTH, HEIGHT))


def _measured(drawing, photograph, settings=SETTINGS, static=(1,)):
    """The edge measured along the drawing's one boundary of the static layers."""
    (chain,) = static_boundaries(drawing, set(static))
    return measure_dye_edge(
        drawing, chain, photograph, RIG_TO_IMAGE, settings, clip_level=1.0
    )


@pytest.fixture(scope="module")
def wall():
    """The wall drawing and the edge measured along its band. The sand is undyed at
    y 4 to 12 mm and darkens in two steps above y 25 mm."""
    drawing = _wall()
    photograph = _photograph(
        lambda x, y: x - _edge_x(y),
        rival=lambda x, y: y >= 0.025,
        undyed=lambda x, y: (y >= 0.004) & (y <= 0.012),
    )
    return drawing, _measured(drawing, photograph)


@pytest.fixture(scope="module")
def clipped():
    """The wall drawing, a photograph where a channel is clipped at y 22.5 to 25 mm
    and pixels are invalid at y 20 to 22.5 mm, and the edge measured on it along
    the band."""
    photograph = _photograph(lambda x, y: x - _edge_x(y))
    photograph[50:75, :120] = [1.0, 0.5, 0.5]
    photograph[75:100, :120, 1] = np.nan
    drawing = _wall()
    return drawing, photograph, _measured(drawing, photograph)


@pytest.fixture(scope="module")
def disc():
    """The fixture's drawing and the edge measured round it."""
    drawing = _disc()
    photograph = _photograph(lambda x, y: np.hypot(x - 0.02, y - 0.015) - 0.008)
    return drawing, _measured(drawing, photograph, ROUND)


def test_static_boundaries_are_chains_between_static_and_porous_polygons():
    """A chain runs along a static polygon where porous polygons share its edges,
    the static polygon on its left; edges on the frame or with another static
    polygon are not part of it; a static polygon that porous ones surround gives
    a closed chain from a joint. A polygon drawn on top is not a neighbour."""
    drawing = _wall()
    assert partition_faults(drawing) == []
    assert static_boundaries(drawing, {1}) == [(1, 2, 3, 4, 5)]
    assert static_boundaries(drawing, {1, 2}) == [(3, 4, 5), (8, 9, 3)]
    assert partition_faults(_disc()) == []
    assert static_boundaries(_disc(), {1}) == [(*range(32), 0)]
    turned = ((1, (*range(4, -1, -1), *range(31, 4, -1))),) + _disc().polygons[1:]
    assert static_boundaries(replace(_disc(), polygons=turned), {1}) == [
        (*range(16, 32), *range(17))
    ]
    over = replace(
        drawing,
        anchors=np.vstack([drawing.anchors, [[8e-3, 11e-3]]]),
        polygons=drawing.polygons + ((9, (3, 2, 15)),),
    )
    assert partition_faults(over) == []
    assert static_boundaries(over, {1, 2}) == [(3, 4, 5), (8, 9, 3)]


def test_the_edge_is_where_the_dye_begins_beyond_the_light_rim(wall):
    """The edge lies where the dyed sand begins, beyond the static element's light
    rim. Where two drops rival each other the steeper one is kept as less certain,
    and where the sand is undyed the edge is interpolated along the boundary."""
    drawing, edge = wall
    y = edge.base[:, 1]
    kind = np.array(edge.kind)
    assert set(kind) == {"reliable", "less certain", "interpolated"}
    assert np.all(kind[(y > 0.007) & (y < 0.009)] == "interpolated")
    assert np.all(kind[y > 0.0285] == "less certain")
    assert np.all(kind[(y > 0.015) & (y < 0.022)] == "reliable")
    stable = (edge.spread <= 3e-4) & (edge.rival < SETTINGS.rival_ratio)
    near = np.abs(np.subtract.outer(edge.along, edge.along)) <= 3e-3 + 1e-9
    fewest = SETTINGS.fewest_stations
    assert np.all((near.astype(int) @ stable)[kind == "reliable"] >= fewest)
    assert np.all((near.astype(int) @ stable)[kind != "reliable"] < fewest)
    assert np.abs(edge.points[:, 0] - _edge_x(edge.points[:, 1])).max() < 7e-5
    assert np.median(edge.spread[kind == "reliable"]) < 2e-5
    assert np.all(edge.rival[y > 0.0265] > SETTINGS.rival_ratio)
    assert np.allclose(np.hypot(*edge.normal.T), 1)
    assert edge.normal[:, 0].min() > 0.99
    blank = _shot(lambda x, y: 0 * x)
    for chain in ([3], [3, 3]):
        with pytest.raises(ValueError, match="two or more anchors and a length"):
            measure_dye_edge(
                drawing, chain, blank, RIG_TO_IMAGE, SETTINGS, clip_level=1
            )


def test_the_drop_is_sought_within_the_reach_and_refined_between_samples():
    """Steeper drops just beyond the reach on either side, a groove in the static
    element and a dark seam in the sand, are not taken; the dye's drop, which lies
    halfway between two samples of the profile, is found there."""

    def light(x, y):
        """Bright static element with a groove at 3.95 mm, its rim, the dye
        beginning at 7.05 mm with a small drop, and a seam at 8.95 mm."""
        level = 0.95 - 0.8 * _step(x - 3.95e-3) + 0.45 * _step(x - 4.2e-3)
        level += 0.3 * _step(x - 6.45e-3) - 0.3 * _step(x - 7.05e-3)
        return level - 0.6 * _step(x - 8.95e-3) + 0.6 * _step(x - 9.2e-3)

    edge = _measured(_band(6.2), _shot(light, noise=0.0))
    assert set(edge.kind) == {"reliable"}
    assert np.abs(edge.found - 8.5e-4).max() < 2e-6
    assert np.abs(edge.points[:, 0] - 7.05e-3).max() < 2e-6


def test_a_drop_at_or_beyond_the_ends_of_the_reach_is_no_drop():
    """Where the steepest fall lies at either end of the reach, the drop lies there
    or beyond: with the dye 0.3 mm beyond either reach, or a dark seam just beyond
    the porous reach that is steeper there than a weak drop of the dye within it,
    no station has a drop and the edge is not found; 0.2 mm within either reach,
    the dye's drop is found. Where the dye begins beyond the reach in nine rows of
    every twenty, the median profile drops but some resamples do not, so the
    spread is infinite; in eleven rows of twenty, the median does not drop, and
    the spread is infinite though the one resample may drop."""

    def seam(x, y):
        """The rim, a weak drop of the dye at 7 mm and a seam at 8.85 mm."""
        level = 0.6 + 0.3 * _step(x - 6.4e-3) - 0.15 * _step(x - 7e-3)
        return level - 0.5 * _step(x - 8.85e-3)

    for drawing, photograph in (
        (_band(6.2), _shot(seam)),
        (_band(6.2), _photograph(lambda x, y: x - 9e-3)),
        (_band(9.5), _photograph(lambda x, y: x - 7.2e-3)),
    ):
        edge = _measured(drawing, photograph)
        assert set(edge.kind) == {"not found"}
        assert np.isnan(edge.found).all() and np.isinf(edge.spread).all()
        assert np.isnan(edge.offset).all()
    for drawing, x in ((_band(6.2), 8.5e-3), (_band(9.5), 7.7e-3)):
        edge = _measured(drawing, _photograph(lambda x_, y: x_ - x))
        assert set(edge.kind) == {"reliable"}
        assert np.abs(edge.points[:, 0] - x).max() < 2e-5

    farther = (np.arange(round(HEIGHT / PIXEL)) % 20 < 9)[:, None] * 2e-3
    edge = _measured(_band(6.2), _photograph(lambda x, y: x - 0.007 - farther))
    assert np.mean(np.abs(edge.found - 8e-4) < 5e-5) > 0.9
    assert np.mean(np.isinf(edge.spread)) > 0.9
    assert "reliable" not in edge.kind and "less certain" not in edge.kind
    farther = (np.arange(round(HEIGHT / PIXEL)) % 20 < 11)[:, None] * 2e-3
    edge = _measured(
        _band(6.2),
        _photograph(lambda x, y: x - 0.007 - farther),
        replace(SETTINGS, resamples=1),
    )
    assert np.mean(np.isnan(edge.found)) > 0.9
    assert np.isinf(edge.spread[np.isnan(edge.found)]).all()


def test_no_edge_is_found_where_nothing_drops():
    """A flat profile, one that only brightens into the sand, and an undyed
    photograph with noise have no stable drop: where a station's profile does not
    drop, its spread is infinite, and as no station has an edge, none is
    interpolated and the edge is not found."""
    for light, noise in (
        (lambda x, y: 0.6 + 0 * x, 0.0),
        (lambda x, y: 0.5 + 1e7 * (x - 6.5e-3) ** 3, 0.0),
        (lambda x, y: 0.6 + 0 * x, 0.01),
    ):
        edge = _measured(_band(6.2), _shot(light, noise=noise))
        assert set(edge.kind) == {"not found"}
        assert np.isnan(edge.offset).all()
        if not noise:
            assert np.isnan(edge.found).all() and np.isinf(edge.spread).all()


def test_a_drop_that_moves_under_resampling_is_less_certain():
    """Where the dye begins 1 mm farther in nine rows of every twenty, the median
    profile drops once, at the nearer edge, but resampled profiles move the drop:
    the stations are not stable, and the edge is read from those of moderate
    spread."""
    farther = (np.arange(round(HEIGHT / PIXEL)) % 20 < 9)[:, None] * 1e-3
    edge = _measured(_band(6.2), _photograph(lambda x, y: x - 0.007 - farther))
    inner = (edge.base[:, 1] > 0.003) & (edge.base[:, 1] < 0.027)
    assert np.all(edge.rival[inner] < SETTINGS.rival_ratio)
    assert np.mean(edge.spread[inner] > 3e-4) > 0.8
    assert "reliable" not in edge.kind and "less certain" in edge.kind
    assert np.abs(edge.points[:, 0] - 0.007).max() < 7e-5


def test_a_curved_boundary_is_read_along_its_normals(disc):
    """Round a closed boundary the profiles follow its normals, so the edge lies on
    the fixture's circle all round, from the first station to the last. A closed
    boundary no longer than twice the chord's or the window's half-length is
    refused."""
    drawing, edge = disc
    assert set(edge.kind) == {"reliable"}
    radius = np.hypot(edge.points[:, 0] - 0.02, edge.points[:, 1] - 0.015)
    assert np.abs(radius - 0.008).max() < 5e-5
    outward = edge.base - [0.02, 0.015]
    outward /= np.hypot(*outward.T)[:, None]
    assert np.einsum("ij,ij->i", edge.normal, outward).min() > 0.9995
    blank = _shot(lambda x, y: 0 * x)
    for change in ({"direction_mm": 24.0}, {"window_mm": 24.0}):
        with pytest.raises(ValueError, match="closed boundary"):
            measure_dye_edge(
                drawing,
                edge.chain,
                blank,
                RIG_TO_IMAGE,
                replace(ROUND, **change),
                clip_level=1.0,
            )


def test_the_edge_is_interpolated_along_the_boundary():
    """Across an undyed stretch the edge's offset is interpolated between the
    stations on either side, also round a closed boundary across its first anchor,
    so the edge follows the dye's tilted line and its circle there."""
    undyed = _photograph(
        lambda x, y: x - _edge_x(y), undyed=lambda x, y: (y >= 0.005) & (y <= 0.025)
    )
    edge = _measured(_band(6.2), undyed)
    assert np.count_nonzero(np.array(edge.kind) == "interpolated") > 30
    assert np.abs(edge.points[:, 0] - _edge_x(edge.points[:, 1])).max() < 7e-5

    def angle(x, y):
        """Angle of a point round the fixture's drawn centre."""
        return np.arctan2(y - 0.015, x - 0.02)

    photograph = _photograph(
        lambda x, y: np.hypot(x - 0.02, y - 0.0154) - 0.008,
        undyed=lambda x, y: np.abs(angle(x, y)) < 0.4,
    )
    edge = _measured(_disc(), photograph, ROUND)
    kind = np.array(edge.kind)
    assert kind[0] == kind[-1] == "interpolated" and np.count_nonzero(
        kind == "reliable"
    )
    radius = np.hypot(edge.points[:, 0] - 0.02, edge.points[:, 1] - 0.0154)
    assert np.abs(radius - 0.008).max() < 7e-5


def test_profiles_stay_within_the_boundary():
    """A boundary that ends at a joint inside the image reads no profile beyond
    it, where the dye begins elsewhere: its stations there are stable."""
    step = np.where(np.arange(round(HEIGHT / PIXEL)) < 150, 0.007, 0.0085)[:, None]
    drawing = _wall()
    chain, _other = static_boundaries(drawing, {1, 2})
    photograph = _photograph(lambda x, y: x - step)
    edge = measure_dye_edge(
        drawing, chain, photograph, RIG_TO_IMAGE, SETTINGS, clip_level=1.0
    )
    assert edge.chain == (3, 4, 5)
    assert edge.spread[:4].max() < 5e-5
    assert np.abs(edge.points[:, 0] - 0.007).max() < 7e-5


def test_clipped_and_invalid_pixels_leave_stations_unreadable(clipped):
    """Profiles that touch a clipped channel or an invalid pixel are left out; a
    station with too few clean profiles on either side of it is unreadable, with
    no edge, and the others read the edge from their clean profiles. The profiles
    a station needs and the stations a running median needs are set apart."""
    drawing, photograph, edge = clipped
    y = edge.base[:, 1]
    gone = np.array(edge.kind) == "unreadable"
    assert np.all(gone[(y > 0.0196) & (y < 0.0254)])
    assert not gone[(y < 0.0194) | (y > 0.0256)].any()
    assert np.isnan(edge.offset[gone]).all() and np.isnan(edge.spread[gone]).all()
    assert np.isnan(edge.found[gone]).all()
    error = edge.points[~gone, 0] - _edge_x(edge.points[~gone, 1])
    assert np.abs(error).max() < 7e-5
    stricter = _measured(drawing, photograph, replace(SETTINGS, fewest_profiles=21))
    gone = np.array(stricter.kind) == "unreadable"
    assert np.all(gone[(y > 0.0181) & (y < 0.0269)])
    assert not gone[(y < 0.0179) | (y > 0.0271)].any()
    assert set(np.array(stricter.kind)[~gone]) == {"reliable"}


def test_profiles_that_leave_the_image_or_blend_a_clipped_pixel_are_left_out():
    """A profile that leaves the image on its left or top is left out, and so is
    one whose samples blend a clipped pixel with darker ones to below the clip
    level. A station is read from the profiles that stay in the image, and is
    unreadable where none does, or where too few are clean."""
    drawing = _band(6.2)
    (chain,) = static_boundaries(drawing, {1})
    photograph = _photograph(lambda x, y: x - 0.007)
    for rows, columns in ((0, 45), (100, 0)):
        edge = measure_dye_edge(
            drawing,
            chain,
            photograph[rows:, columns:],
            RIG_TO_IMAGE - [[0, 0, columns], [0, 0, rows], [0, 0, 0]],
            SETTINGS,
            clip_level=1.0,
        )
        y = edge.base[:, 1]
        gone = np.array(edge.kind) == "unreadable"
        if columns:
            assert gone.all()
        else:
            assert gone[y > 0.0221].all() and not gone[y < 0.0219].any()
            assert np.abs(edge.points[~gone, 0] - 0.007).max() < 7e-5
    photograph[100:150, 50, 0] = 1.0
    edge = _measured(drawing, photograph)
    y = edge.base[:, 1]
    gone = np.array(edge.kind) == "unreadable"
    assert gone[(y > 0.0146) & (y < 0.0204)].all()
    assert not gone[(y < 0.0144) | (y > 0.0206)].any()


def test_each_station_is_resampled_by_its_own_draws():
    """A station's spread depends on its own profiles alone: an invalid pixel that
    leaves out a profile of some stations changes none of the others."""
    drawing = _band(6.2)
    photograph = _photograph(lambda x, y: x - 0.007)
    edge = _measured(drawing, photograph)
    photograph[149, 70, 0] = np.nan
    other = _measured(drawing, photograph)
    assert not np.array_equal(edge.spread, other.spread)
    far = np.abs(edge.base[:, 1] - 0.01505) > 0.0022
    for name in ("found", "spread", "rival"):
        assert np.array_equal(getattr(edge, name)[far], getattr(other, name)[far])
