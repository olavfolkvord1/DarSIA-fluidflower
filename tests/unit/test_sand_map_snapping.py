"""Tests of snapping a sand-map drawing into a closed partition of its frame."""

import numpy as np
import pytest

from darsia.presets.workflows.setup.sand_map import SandMapDrawing, partition_faults
from darsia.presets.workflows.setup.sand_map_snapping import (
    SandMapSnapSettings,
    snap_to_frame,
)

WIDTH, HEIGHT = 0.4, 0.2
"""Inner frame of the test rig in metres."""

BAND = 0.0055
"""Inner edge of a band along the left wall, within the snapping distance of it."""


def _short_of_the_frame() -> SandMapDrawing:
    """A band along the left wall, sand and water, linked to each other, as drawn
    on a photograph that does not see the frame: the band starts 5 mm from the
    left wall, the water ends 2 mm under the top, the sand runs 2 mm past the
    right wall and dips 1 mm under the floor in its middle."""
    anchors = [
        [0.005, 0.001],
        [BAND, 0.001],
        [0.402, 0.001],
        [0.402, 0.1],
        [BAND, 0.1],
        [0.402, 0.198],
        [BAND, 0.198],
        [0.005, 0.198],
        [0.2, -0.001],
        [0.199, 0.0005],
    ]
    polygons = (
        (3, (0, 1, 4, 6, 7)),
        (1, (1, 8, 9, 2, 3, 4)),
        (2, (4, 3, 5, 6)),
    )
    return SandMapDrawing(np.array(anchors), polygons, frame=(WIDTH, HEIGHT))


def test_the_outer_boundary_snaps_onto_the_frame():
    """The outer anchors move onto the frame's edge, corners included; anchors
    beyond it are clipped; the band's inner anchors near the wall stay, and an
    anchor where the boundary would turn back along the floor leaves it."""
    drawing = _short_of_the_frame()
    assert partition_faults(drawing)
    snapped, moves = snap_to_frame(drawing, SandMapSnapSettings(6.0, 1.0))
    assert partition_faults(snapped) == []
    ends = {move.anchor: move.end for move in moves}
    assert ends == {
        0: (0.0, 0.0),
        1: (BAND, 0.0),
        2: (WIDTH, 0.0),
        3: (WIDTH, 0.1),
        5: (WIDTH, HEIGHT),
        6: (BAND, HEIGHT),
        7: (0.0, HEIGHT),
        8: (0.2, 0.0),
        9: (0.199, 0.0),
    }
    assert all(move.start == tuple(drawing.anchors[move.anchor]) for move in moves)
    assert np.array_equal(snapped.anchors[4], drawing.anchors[4])
    assert 8 not in snapped.polygons[1][1]

    narrow, _moves = snap_to_frame(drawing, SandMapSnapSettings(4.0, 1.0))
    assert np.array_equal(narrow.anchors[[0, 7], 0], [0.005, 0.005])
    only, moves = snap_to_frame(drawing, SandMapSnapSettings(6.0, 1.0), anchors=[7])
    assert [move.anchor for move in moves] == [7]


def test_a_boundary_across_a_corner_gets_the_corner():
    """An outer edge from the floor to the right wall that cuts the corner gets the
    corner as a new anchor; two anchors of a step in the wall's boundary become
    one; a layer on top beyond the frame is clipped onto it."""
    anchors = [[0, 0], [0.39, 0], [WIDTH, 0.01], [WIDTH, HEIGHT], [0, HEIGHT]]
    anchors += [[0.004, 0.1], [0.003, 0.1]]
    anchors += [[0.1, 0.15], [0.2, 0.15], [0.2, 0.21], [0.1, 0.21]]
    polygons = ((1, (0, 1, 2, 3, 4, 5, 6)), (9, (7, 8, 9, 10)))
    drawing = SandMapDrawing(
        np.array(anchors), polygons, frozenset({9}), (WIDTH, HEIGHT)
    )
    snapped, moves = snap_to_frame(drawing, SandMapSnapSettings(6.0, 1.0))
    assert [(move.anchor, move.end, move.joined) for move in moves] == [
        (5, (0.0, 0.1), None),
        (6, (0.0, 0.1), 5),
        (9, (0.2, HEIGHT), None),
        (10, (0.1, HEIGHT), None),
        (11, (WIDTH, 0.0), None),
    ]
    assert moves[-1].start is None
    assert snapped.polygons == ((1, (0, 1, 11, 2, 3, 4, 5)), (9, (7, 8, 9, 10)))
    assert partition_faults(snapped) == []


def test_a_boundary_near_another_does_not_reach_the_frame():
    """A layer whose boundary nearly meets a band along the floor, or reaches into
    a band thinner than the boundary distance, keeps its anchors off the floor."""
    for band, start in ((0.004, 0.0042), (0.0005, 0.0003)):
        anchors = [[0, 0], [WIDTH, 0], [WIDTH, band], [0, band]]
        anchors += [[0, start], [WIDTH, start], [WIDTH, HEIGHT], [0, HEIGHT]]
        polygons = ((1, (0, 1, 2, 3)), (2, (4, 5, 6, 7)))
        drawing = SandMapDrawing(np.array(anchors), polygons, frame=(WIDTH, HEIGHT))
        snapped, moves = snap_to_frame(drawing, SandMapSnapSettings(6.0, 1.0))
        assert moves == [] and np.array_equal(snapped.anchors, drawing.anchors)


def test_snapping_needs_the_frame():
    """Without a frame there is nothing to snap to."""
    drawing = _short_of_the_frame()
    with pytest.raises(ValueError, match="frame"):
        snap_to_frame(
            SandMapDrawing(drawing.anchors, drawing.polygons),
            SandMapSnapSettings(6.0, 0.5),
        )
