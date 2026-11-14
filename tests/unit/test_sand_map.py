"""Tests of the sand map: a drawing of sand layers placed on a run's grid."""

import numpy as np

import darsia
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    rasterize_sand_map,
    rig_grid,
)

WIDTH, HEIGHT = 0.4, 0.2
"""Inner frame of the test rig in metres."""

SHAPE = (90, 160)
"""A run's grid, viewed in perspective through PERSPECTIVE."""

PERSPECTIVE = np.array([[0.97, 0.05, 2.0], [-0.04, 1.01, 1.5], [2e-4, -1e-4, 1.0]])
"""A projective distortion of the grid's pixel coordinates."""


def _grid(shape: tuple[int, int], width: float, height: float) -> darsia.Image:
    """An image on a grid spanning the rig from its lower left corner."""
    return darsia.Image(np.zeros(shape, np.uint8), dimensions=[height, width])


def _split_drawing() -> SandMapDrawing:
    """The rig split into two layers from corner to corner along a zigzag of shared
    anchors; the lower layer (2) is the smaller."""
    anchors = [[0, 0], [0.4, 0], [0.4, 0.2], [0, 0.2]]
    anchors += [[0.13, 0.02], [0.17, 0.12], [0.27, 0.09], [0.31, 0.19]]
    return SandMapDrawing(
        np.array(anchors), ((1, (0, 4, 5, 6, 7, 2, 3)), (2, (0, 1, 2, 7, 6, 5, 4)))
    )


def _rig_to_grid() -> np.ndarray:
    """The run's rig geometry: rig coordinates to the distorted grid."""
    return PERSPECTIVE @ rig_grid(_grid(SHAPE, WIDTH, HEIGHT))


def _rectangle(anchors: list, x0: float, x1: float, y0: float, y1: float) -> tuple:
    """Append a rectangle's corners to the anchors and return their indices."""
    anchors += [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return tuple(range(len(anchors) - 4, len(anchors)))


def test_overlaps_frame_and_edge_follow_the_drawing_rules():
    """The smaller polygon wins, layers on top win, equal areas keep the drawing
    order, nothing lies outside the frame and the edge fill reaches the frame."""
    anchors = []
    polygons = [
        (2, _rectangle(anchors, 0.006, 0.45, -0.01, 0.1)),
        (5, _rectangle(anchors, 0.006, 0.45, 0.1, 0.21)),
        (3, _rectangle(anchors, 0.08, 0.14, 0.02, 0.06)),
        (4, _rectangle(anchors, 0.12, 0.30, 0.04, 0.16)),
        (6, _rectangle(anchors, 0.3125, 0.34375, 0.125, 0.15625)),
        (7, _rectangle(anchors, 0.328125, 0.359375, 0.125, 0.15625)),
    ]
    drawing = SandMapDrawing(np.array(anchors), tuple(polygons), frozenset({4}))
    # A grid of 2 mm pixels over 0.5 x 0.25 m; the frame is its lower left part.
    shape = (125, 250)
    rig_to_grid = rig_grid(_grid(shape, 0.5, 0.25))

    def at(labels, x, y):
        return labels[int((0.25 - y) / 0.002), int(x / 0.002)]

    labels, inside = rasterize_sand_map(
        drawing, shape, rig_to_grid, width=WIDTH, height=HEIGHT, edge_fill_mm=8.0
    )
    assert at(labels, 0.11, 0.04) == 3
    assert at(labels, 0.13, 0.05) == 4 and at(labels, 0.2, 0.1) == 4
    assert at(labels, 0.32, 0.14) == 6 and at(labels, 0.336, 0.14) == 7
    assert not labels[~inside].any() and not inside[:, 201:].any()
    assert not inside[:24].any() and inside[26:, :199].all()
    assert at(labels, 0.001, 0.05) == 2 and at(labels, 0.003, 0.15) == 5
    assert np.all(labels[inside] > 0)

    unfilled, _ = rasterize_sand_map(
        drawing, shape, rig_to_grid, width=WIDTH, height=HEIGHT, edge_fill_mm=0.0
    )
    assert at(unfilled, 0.001, 0.05) == 0 and at(unfilled, 0.003, 0.15) == 0


def test_a_split_at_shared_anchors_leaves_no_gap_or_overlap():
    """Two layers on shared anchors cover exactly what the whole rig covers, and
    the pixels both polygons reach belong to the smaller one."""
    drawing = _split_drawing()
    whole = SandMapDrawing(drawing.anchors, ((1, (0, 1, 2, 3)),))
    rasters = [
        rasterize_sand_map(
            d, SHAPE, _rig_to_grid(), width=WIDTH, height=HEIGHT, edge_fill_mm=0.0
        )[0]
        for d in (
            drawing,
            whole,
            SandMapDrawing(drawing.anchors, drawing.polygons[:1]),
            SandMapDrawing(drawing.anchors, drawing.polygons[1:]),
        )
    ]
    split, covered, upper, lower = rasters
    assert np.array_equal(split > 0, covered > 0)
    both = (upper > 0) & (lower > 0)
    assert both.any() and np.all(split[both] == 2)
    assert np.array_equal(split == 1, (upper > 0) & ~both)
