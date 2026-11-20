"""Tests of snapping a sand-map drawing into a closed partition of its frame."""

import numpy as np
import pytest

from darsia.presets.workflows.setup.sand_map import SandMapDrawing, partition_faults
from darsia.presets.workflows.setup.sand_map_snapping import (
    SandMapSnapSettings,
    link_boundaries,
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


def _split() -> SandMapDrawing:
    """The rig split into two layers along a zigzag of shared anchors."""
    anchors = [[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]
    anchors += [[0.13, 0.02], [0.17, 0.12], [0.27, 0.09], [0.31, 0.19]]
    polygons = ((1, (0, 4, 5, 6, 7, 2, 3)), (2, (0, 1, 2, 7, 6, 5, 4)))
    return SandMapDrawing(np.array(anchors), polygons, frame=(WIDTH, HEIGHT))


def test_near_boundaries_link_through_shared_anchors():
    """The lower layer's own anchors near the upper layer's boundary join it, at
    its anchors or on its edges, and the drawing becomes a closed partition; a
    boundary farther than the distance stays."""
    split = _split()
    anchors = np.vstack(
        [split.anchors, [[0.17, 0.1997], [0.2702, 0.0902], [0.29, 0.1401]]]
    )
    anchors[8] = [0.17, 0.1197]
    lower = (2, (0, 1, 2, 7, 10, 9, 8, 4))
    drawing = SandMapDrawing(anchors, (split.polygons[0], lower), frame=split.frame)
    assert {fault.kind for fault in partition_faults(drawing)} >= {"gap", "overlap"}

    linked, moves = link_boundaries(drawing, SandMapSnapSettings(6.0, 0.5))
    assert partition_faults(linked) == []
    assert {move.anchor: move.joined for move in moves} == {
        5: 8,
        6: None,
        9: None,
        10: None,
    }
    for move in moves:
        assert np.hypot(*np.subtract(move.end, move.start)) <= 5e-4
    (_layer, upper), (_layer, lower) = linked.polygons
    assert set(upper) & set(lower) == {0, 2, 4, 6, 7, 8, 9, 10}

    kept, moves = link_boundaries(drawing, SandMapSnapSettings(6.0, 0.25))
    assert 5 not in {move.anchor for move in moves}
    assert [fault.kind for fault in partition_faults(kept)] == ["gap"]


def test_a_crack_in_one_polygon_is_not_a_boundary_to_link():
    """Both sides of a crack in one polygon are its own, so nothing joins them."""
    anchors = [[0, 0], [0.2, 0], [0.2, 0.05], [0.2, 0.1], [0.2002, 0.1]]
    anchors += [[0.2002, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]]
    cracked = SandMapDrawing(
        np.array(anchors), ((1, tuple(range(9))),), frame=(WIDTH, HEIGHT)
    )
    assert [fault.kind for fault in partition_faults(cracked)] == ["gap"]
    linked, moves = link_boundaries(cracked, SandMapSnapSettings(6.0, 0.5))
    assert moves == [] and linked.polygons == cracked.polygons


def test_a_boundary_through_a_neighbours_edge_is_linked_without_moving():
    """An anchor lying on a neighbour's edge is added to that edge, unmoved; only
    the anchors given may join."""
    anchors = np.array(
        [[0, 0], [WIDTH, 0], [WIDTH, 0.1], [0, 0.1], [WIDTH, HEIGHT], [0, HEIGHT]]
        + [[0.2, 0.1]]
    )
    halves = ((1, (3, 6, 2, 4, 5)), (2, (0, 1, 2, 3)))
    drawing = SandMapDrawing(anchors, halves, frame=(WIDTH, HEIGHT))
    linked, moves = link_boundaries(drawing, SandMapSnapSettings(6.0, 0.5))
    assert [(move.anchor, move.start, move.end) for move in moves] == [
        (6, (0.2, 0.1), (0.2, 0.1))
    ]
    assert linked.polygons[1] == (2, (0, 1, 2, 6, 3))
    assert partition_faults(linked) == []
    same, moves = link_boundaries(drawing, SandMapSnapSettings(6.0, 0.5), anchors=[2])
    assert moves == [] and same.polygons == drawing.polygons


@pytest.mark.parametrize("helper", [snap_to_frame, link_boundaries])
def test_snapping_needs_the_frame(helper):
    """Without a frame there is nothing to snap to."""
    drawing = SandMapDrawing(_split().anchors, _split().polygons)
    with pytest.raises(ValueError, match="frame"):
        helper(drawing, SandMapSnapSettings(6.0, 0.5))
