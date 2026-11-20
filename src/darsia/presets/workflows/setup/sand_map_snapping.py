"""Snapping a sand-map drawing into a closed partition of its frame.

Two helpers move anchors of a drawing (:class:`SandMapDrawing`) so that its
boundaries are linked, each to a neighbouring polygon or to the frame's edge:

- :func:`snap_to_frame` moves the drawing's outer boundary onto the frame's edge;
- :func:`link_boundaries` joins near-coincident boundaries of neighbouring
  polygons through shared anchors.

They serve an editor, for an anchor being placed, and the conversion of a drawing
made without links. Both return the new drawing and every anchor they moved or
added (:class:`AnchorMove`), for a person to review; :func:`partition_faults`
tells what is left. Anchor indices are kept: an anchor that joins another is no
longer used by a polygon, and added anchors follow the others.

"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import shapely

from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    _boundary,
    _ring,
)

_NORMALS = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
"""Outward normals of the frame's sides: bottom, right, top, left."""


@dataclass(frozen=True)
class SandMapSnapSettings:
    """Distances of snapping a drawing into a partition of its frame; one
    configuration table, in millimetres in the rig."""

    frame_mm: float
    """Largest distance from the frame's edge of an anchor that moves onto it."""
    boundary_mm: float
    """Largest distance from another polygon's unlinked boundary of an anchor
    that joins it."""


@dataclass(frozen=True)
class AnchorMove:
    """An anchor moved, or added, by snapping."""

    anchor: int
    """Index of the anchor."""
    start: tuple[float, float] | None
    """Position before, in metres; None for an added anchor."""
    end: tuple[float, float]
    """Position after, in metres."""
    joined: int | None = None
    """The anchor it became, which its polygons now use; None if it stays."""


def snap_to_frame(
    drawing: SandMapDrawing,
    settings: SandMapSnapSettings,
    anchors: Sequence[int] | None = None,
) -> tuple[SandMapDrawing, list[AnchorMove]]:
    """Move the drawing's outer boundary onto the frame's edge.

    The outer edges are the unlinked edges of the polygons not on top that face
    the uncovered region reaching the frame's edge: an edge's middle lies on that
    region's border, not inside another polygon, and the point ``boundary_mm``
    outwards from it lies in no such polygon and in none of the holes they
    enclose, so a boundary near another one is not outer. Each outer edge faces
    the side of the frame nearest its outward direction; if both its anchors lie
    within ``frame_mm`` of that side, they move onto it, perpendicular to the
    side. An anchor of two such edges facing neighbouring sides moves onto their
    corner, and an anchor outside the frame moves onto it, each coordinate
    clipped. Anchors that land on one point (within a billionth of the frame's
    size) become the one of them with the lowest index. An anchor where a
    boundary then turns back along a side, enclosing nothing, leaves that
    boundary unless another polygon uses it. An unlinked edge that then runs from
    one side to another, with nothing covering the corner between, gets the
    corners it passes, so that it follows the frame's edge.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, with its frame.
    settings : SandMapSnapSettings
        The snapping distances.
    anchors : sequence of int, optional
        The anchors that may move, e.g. one being placed in an editor; by
        default all.

    Returns
    -------
    SandMapDrawing
        The snapped drawing, its polygons counter-clockwise and without repeated
        anchors.
    list of AnchorMove
        Every anchor moved or added.

    Raises
    ------
    ValueError
        If the drawing has no frame.

    """
    if drawing.frame is None:
        raise ValueError("The drawing has no frame to snap to.")
    width, height = drawing.frame
    points = np.array(drawing.anchors, dtype=np.float64)
    rings = [_ring(points, indices) for _layer, indices in drawing.polygons]
    top = [layer in drawing.on_top for layer, _indices in drawing.polygons]
    allowed = set(range(len(points)) if anchors is None else anchors)
    tolerance = settings.frame_mm / 1000
    scale = 1e-9 * max(width, height)

    filled = shapely.union_all(
        [
            shapely.Polygon(part.exterior)
            for part in shapely.get_parts(_covered(points, rings, top))
            if part.geom_type == "Polygon"
        ]
    )
    unlinked, _sides = _boundary(points, rings, top, drawing.frame)
    sides: dict[int, set[int]] = {}
    for _k, start, end in unlinked:
        a, b = points[start], points[end]
        normal = np.array([b[1] - a[1], a[0] - b[0]])
        length = float(np.hypot(*normal))
        if not length > 0:
            continue
        middle = 0.5 * (a + b)
        probe = middle + settings.boundary_mm / 1000 * normal / length
        if filled.boundary.distance(shapely.Point(middle)) > scale or filled.covers(
            shapely.Point(probe)
        ):
            continue
        side = int(np.argmax(_NORMALS @ normal))
        if (
            max(_to_sides(a, drawing.frame)[side], _to_sides(b, drawing.frame)[side])
            <= tolerance
        ):
            for index in (start, end):
                sides.setdefault(index, set()).add(side)

    moved = {}
    for index in sorted(allowed):
        x, y = points[index]
        target = [min(max(x, 0.0), width), min(max(y, 0.0), height)]
        found = sides.get(index, set())
        if len(found) == 1 or found in ({0, 1}, {1, 2}, {2, 3}, {0, 3}):
            for side in found:
                if side % 2:
                    target[0] = width if side == 1 else 0.0
                else:
                    target[1] = height if side == 2 else 0.0
        if target != [x, y]:
            moved[index] = (float(x), float(y))
            points[index] = target

    joined = {}
    for index in list(moved):
        near = np.flatnonzero(np.hypot(*(points - points[index]).T) <= scale)
        for other in near.tolist()[1:]:
            joined[other] = int(near[0])
            moved.setdefault(other, tuple(points[other].tolist()))
    rings = _without_spikes(
        points,
        [_ring(points, [joined.get(index, index) for index in ring]) for ring in rings],
        drawing.frame,
    )
    result = [
        AnchorMove(
            index,
            moved[index],
            tuple(points[joined.get(index, index)].tolist()),
            joined.get(index),
        )
        for index in sorted(moved)
    ]
    first: dict[tuple[float, float], int] = {}
    for index, point in enumerate(points.tolist()):
        if index not in joined:
            first.setdefault(tuple(point), index)

    corners = [(width, 0.0), (width, height), (0.0, height), (0.0, 0.0)]
    covered = _covered(points, rings, top)
    unlinked, _sides = _boundary(points, rings, top, drawing.frame)
    added: dict[tuple[float, float], int] = {}
    for k, start, end in unlinked:
        leave = _sides_of(points[start], drawing.frame)
        arrive = _sides_of(points[end], drawing.frame)
        a, b = points[start], points[end]
        middle = 0.5 * (a + b) + scale * np.array([b[1] - a[1], a[0] - b[0]]) / max(
            float(np.hypot(*(b - a))), scale
        )
        if (
            not leave
            or not arrive
            or leave & arrive
            or covered.covers(shapely.Point(middle))
        ):
            continue
        side = 0 if leave == {0, 3} else max(leave)
        last = 3 if arrive == {0, 3} else min(arrive)
        passed = []
        while side != last:
            corner = corners[side]
            if corner not in first and corner not in added:
                added[corner] = len(points) + len(added)
                result.append(AnchorMove(added[corner], None, corner))
            passed.append(first.get(corner, added.get(corner)))
            side = (side + 1) % 4
        position = rings[k].index(start) + 1
        rings[k][position:position] = passed
    if added:
        points = np.vstack([points, np.array(list(added), dtype=np.float64)])
    polygons = tuple(
        (layer, tuple(ring)) for (layer, _indices), ring in zip(drawing.polygons, rings)
    )
    return SandMapDrawing(points, polygons, drawing.on_top, drawing.frame), result


def link_boundaries(
    drawing: SandMapDrawing,
    settings: SandMapSnapSettings,
    anchors: Sequence[int] | None = None,
) -> tuple[SandMapDrawing, list[AnchorMove]]:
    """Join near-coincident boundaries of neighbouring polygons through shared
    anchors.

    An anchor of an unlinked edge (of a polygon not on top, neither shared with a
    neighbour nor on the frame's edge) joins the nearest unlinked edge of a
    polygon that does not use it, if that lies within ``boundary_mm``. At an end
    of that edge, the anchor becomes the end's anchor; elsewhere it moves onto the
    nearest point of the edge, which is split there to run through it. Every
    polygon that uses the anchor follows it. The anchors are taken in the order of
    their indices, in passes until one joins none; each join links an anchor to
    one more polygon or takes it out of use, so the passes end.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, with its frame.
    settings : SandMapSnapSettings
        The snapping distances.
    anchors : sequence of int, optional
        The anchors that may join, e.g. one being placed in an editor; by default
        all.

    Returns
    -------
    SandMapDrawing
        The linked drawing, its polygons counter-clockwise and without repeated
        anchors.
    list of AnchorMove
        Every anchor moved, once, from its first position to where it ends.

    Raises
    ------
    ValueError
        If the drawing has no frame.

    """
    if drawing.frame is None:
        raise ValueError("The drawing has no frame to link to.")
    points = np.array(drawing.anchors, dtype=np.float64)
    rings = [_ring(points, indices) for _layer, indices in drawing.polygons]
    top = [layer in drawing.on_top for layer, _indices in drawing.polygons]
    allowed = set(range(len(points)) if anchors is None else anchors)
    tolerance = settings.boundary_mm / 1000
    starts: dict[int, tuple[float, float]] = {}
    joined: dict[int, int] = {}

    changed = True
    while changed:
        changed = dirty = False
        unlinked, _sides = _boundary(points, rings, top, drawing.frame)
        for index in sorted({i for _k, *ends in unlinked for i in ends} & allowed):
            if dirty:
                unlinked, _sides = _boundary(points, rings, top, drawing.frame)
                dirty = False
            own = {k for k, ring in enumerate(rings) if index in ring}
            edges = [(k, u, v) for k, u, v in unlinked if k not in own]
            if not edges or not any(index in (u, v) for _k, u, v in unlinked):
                continue
            first = points[[u for _k, u, _v in edges]]
            span = points[[v for _k, _u, v in edges]] - first
            squared = np.einsum("ij,ij->i", span, span)
            along = np.einsum("ij,ij->i", points[index] - first, span)
            t = np.clip(
                np.divide(along, squared, out=0 * along, where=squared > 0), 0, 1
            )
            distance = np.hypot(*(first + t[:, None] * span - points[index]).T)
            best = int(np.argmin(distance))
            if not distance[best] <= tolerance:
                continue
            starts.setdefault(index, tuple(points[index].tolist()))
            k, u, v = edges[best]
            if t[best] in (0, 1):
                joined[index] = u if t[best] == 0 else v
                for n in own:
                    rings[n] = _ring(
                        points, [joined[index] if i == index else i for i in rings[n]]
                    )
            else:
                points[index] = first[best] + t[best] * span[best]
                rings[k].insert(rings[k].index(u) + 1, index)
            changed = dirty = True

    def final(index: int) -> int:
        """The anchor an anchor ends as, following its joins."""
        while index in joined:
            index = joined[index]
        return index

    moves = [
        AnchorMove(
            index,
            start,
            tuple(points[final(index)].tolist()),
            None if final(index) == index else final(index),
        )
        for index, start in sorted(starts.items())
    ]
    polygons = tuple(
        (layer, tuple(ring)) for (layer, _indices), ring in zip(drawing.polygons, rings)
    )
    return SandMapDrawing(points, polygons, drawing.on_top, drawing.frame), moves


def _without_spikes(
    points: np.ndarray, rings: list[list[int]], frame: tuple[float, float]
) -> list[list[int]]:
    """The rings without the anchors where one turns back along a side of the
    frame, which enclose nothing, unless another ring uses the anchor."""
    users = {}
    for ring in rings:
        for index in ring:
            users[index] = users.get(index, 0) + 1
    result = []
    for ring in rings:
        ring = list(ring)
        n = 0
        while len(ring) > 3 and n < len(ring):
            before, index, after = ring[n - 1], ring[n], ring[(n + 1) % len(ring)]
            a, b, c = points[before], points[index], points[after]
            if (
                users[index] == 1
                and _sides_of(a, frame) & _sides_of(b, frame) & _sides_of(c, frame)
                and np.dot(b - a, c - b) < 0
            ):
                ring.pop(n)
                n = max(n - 1, 0)
            else:
                n += 1
        result.append(ring)
    return result


def _covered(
    points: np.ndarray, rings: list[list[int]], top: list[bool]
) -> shapely.Geometry:
    """The region the polygons not on top cover."""
    return shapely.union_all(
        [
            shapely.make_valid(shapely.Polygon(points[ring]))
            for ring, on_top in zip(rings, top)
            if not on_top and len(ring) > 2
        ]
    )


def _to_sides(point: np.ndarray, frame: tuple[float, float]) -> np.ndarray:
    """Distances of a point inside the frame from its sides: bottom, right, top,
    left."""
    x, y = point
    width, height = frame
    return np.array([y, width - x, height - y, x])


def _sides_of(point: np.ndarray, frame: tuple[float, float]) -> set[int]:
    """The sides of the frame a point lies on exactly: 0 bottom, 1 right, 2 top,
    3 left."""
    x, y = point
    width, height = frame
    inside = 0 <= x <= width and 0 <= y <= height
    return {
        side
        for side, on in enumerate((y == 0, x == width, y == height, x == 0))
        if on and inside
    }
