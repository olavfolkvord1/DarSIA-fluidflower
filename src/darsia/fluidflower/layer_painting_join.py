"""Preview a polygon following a neighbouring outline, without changing it.

Coordinates and distances are raw photograph pixels. A neighbouring polygon
is the master: its vertices stay exactly as supplied; only the two end points
of each joined stretch may be inserted on its edges. No session setting,
classification or image is changed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import explain_validity

PointRC = tuple[float, float]
_EPS = 1.0e-7
# Two endpoints inside the existing half-pixel shared-point tolerance are
# one point, not a shared stretch. In particular, a transverse crossing
# must not collapse a long target edge onto a subpixel piece of the master.
_MIN_SHARED_ARC_PX = 1.0


class JoinEdgesError(ValueError):
    """A refused join, with its raw (row, column) when available."""

    def __init__(self, message: str, location: PointRC | None = None):
        super().__init__(message)
        self.location = location


@dataclass(frozen=True)
class JoinedStretch:
    before: tuple[PointRC, ...]
    after: tuple[PointRC, ...]
    length_px: float


@dataclass(frozen=True)
class JoinPreview:
    target: tuple[PointRC, ...]
    neighbour: tuple[PointRC, ...]
    stretches: tuple[JoinedStretch, ...]

    @property
    def length_px(self) -> float:
        return sum(row.length_px for row in self.stretches)


def _valid(points: Sequence[PointRC], description: str) -> Polygon:
    if len(points) < 3 or not np.isfinite(points).all():
        raise JoinEdgesError(f"{description} needs three finite points.")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= _EPS:
        reason = explain_validity(polygon)
        match = re.search(r"\[([^\]]+)\]", reason)
        location = None
        if match:
            try:
                row, col = map(float, match.group(1).split())
                location = (row, col)
            except ValueError:
                pass
        where = "" if location is None else f" at raw row {location[0]:.2f}, column {location[1]:.2f}"
        raise JoinEdgesError(f"{description}: {reason}{where}. Nothing was changed.", location)
    return polygon


class _Outline:
    def __init__(self, points: Sequence[PointRC]):
        self.points = tuple((float(r), float(c)) for r, c in points)
        closed = np.asarray((*self.points, self.points[0]), dtype=float)
        self.lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        self.cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.length = float(self.cumulative[-1])
        self.line = LineString(closed)

    def at(self, distance: float) -> PointRC:
        distance %= self.length
        index = min(int(np.searchsorted(self.cumulative, distance, side="right")) - 1,
                    len(self.points) - 1)
        start, end = self.points[index], self.points[(index + 1) % len(self.points)]
        fraction = (distance - self.cumulative[index]) / max(self.lengths[index], _EPS)
        if fraction <= _EPS:
            return start
        if fraction >= 1.0 - _EPS:
            return end
        return (start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]))

    def vertices_between(self, start: float, end: float) -> list[PointRC]:
        result = []
        for turn in range(math.floor(start / self.length), math.floor(end / self.length) + 1):
            for distance, point in zip(self.cumulative[:-1], self.points):
                if start + _EPS < distance + turn * self.length < end - _EPS:
                    result.append(point)
        return result

    def path(self, start: float, end: float) -> list[PointRC]:
        return [self.at(start), *self.vertices_between(start, end), self.at(end)]


def _lines(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _lines(part)


def _near_intervals(target: _Outline, neighbour: _Outline, distance: float):
    band = neighbour.line.buffer(distance, quad_segs=16)
    intervals = []
    for index, point in enumerate(target.points):
        edge = LineString((point, target.points[(index + 1) % len(target.points)]))
        if edge.length <= _EPS:
            continue
        for part in _lines(edge.intersection(band)):
            a = float(target.cumulative[index]) + edge.project(Point(part.coords[0]))
            b = float(target.cumulative[index]) + edge.project(Point(part.coords[-1]))
            a, b = sorted((a, b))
            if b - a > _EPS:
                intervals.append((a, b))
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if len(merged) > 1 and merged[0][0] <= _EPS and merged[-1][1] >= target.length - _EPS:
        merged = [(merged[-1][0], merged[0][1] + target.length), *merged[1:-1]]
    return merged


def join_to_neighbour(
    target_points: Sequence[PointRC],
    neighbour_points: Sequence[PointRC],
    distance_px: float,
) -> JoinPreview:
    """Return a complete proposed edit, or refuse a non-simple polygon.

    Each stretch of the target within ``distance_px`` is replaced by the
    master's own outline. A connecting arc must stay within that distance of
    the old stretch in both directions; a mere near corner is not an arc.
    The master never moves. This function never mutates either input.
    """
    if not math.isfinite(distance_px) or distance_px < 0:
        raise JoinEdgesError("Distance must be a finite, non-negative number.")
    _valid(target_points, "Selected polygon")
    _valid(neighbour_points, "Neighbour polygon")
    target, neighbour = _Outline(target_points), _Outline(neighbour_points)
    unchanged = JoinPreview(target.points, neighbour.points, ())
    if distance_px == 0 or target.line.distance(neighbour.line) > distance_px:
        return unchanged
    intervals = _near_intervals(target, neighbour, distance_px)
    if any(b - a >= target.length - _EPS for a, b in intervals):
        raise JoinEdgesError("The whole outline is within the distance. Choose a smaller distance.")
    accepted = []
    insertions = []
    for a, b in intervals:
        old = target.path(a, b)
        start = float(neighbour.line.project(Point(old[0])))
        end = float(neighbour.line.project(Point(old[-1])))
        forward = (end - start) % neighbour.length
        if min(forward, neighbour.length - forward) <= _EPS:
            continue
        ways = [neighbour.path(start, start + forward),
                list(reversed(neighbour.path(end, end + neighbour.length - forward)))]
        old_line = LineString(old)
        ways = [way for way in ways
                if LineString(way).hausdorff_distance(old_line) <= distance_px + _EPS]
        if not ways:
            continue
        new = min(ways, key=lambda way: LineString(way).length)
        if LineString(new).length < _MIN_SHARED_ARC_PX:
            continue
        # Already shared, including all master's vertices: leave the session alone.
        if old_line.equals(LineString(new)) and all(point in target.points for point in new):
            continue
        accepted.append((a, b, new, JoinedStretch(tuple(old), tuple(new), b - a)))
        insertions.extend((start, end))
    if not accepted:
        return unchanged
    # Start just beyond a joined interval, in a gap, so no stretch is split
    # across the beginning of the new list of vertices.
    first_end = accepted[0][1] % target.length
    rebased = sorted(((a - first_end) % target.length,
                      (a - first_end) % target.length + b - a, new, stretch)
                     for a, b, new, stretch in accepted)
    gap = rebased[0][0]
    if gap <= _EPS:
        raise JoinEdgesError("The joined stretches leave no unchanged outline. Choose a smaller distance.")
    cut = first_end + gap / 2.0
    for vertex_distance in target.cumulative[:-1]:
        if not any((vertex_distance - a) % target.length <= b - a + _EPS
                   for a, b, _new, _stretch in accepted):
            cut = float(vertex_distance)
            break
    ordered = sorted(((a - cut) % target.length + cut,
                      (a - cut) % target.length + cut + b - a, new, stretch)
                     for a, b, new, stretch in accepted)
    result = [target.at(cut)]
    cursor = cut
    for a, b, new, stretch in ordered:
        result.extend(target.vertices_between(cursor, a))
        result.extend(new)
        cursor = b
    result.extend(target.vertices_between(cursor, cut + target.length))
    result = _deduplicate(result)
    _valid(result, "Joined polygon")
    # Insert endpoints in the original master order, retaining every vertex.
    master = []
    for index, point in enumerate(neighbour.points):
        master.append(point)
        for value in sorted(set(insertions)):
            if neighbour.cumulative[index] + _EPS < value < neighbour.cumulative[index + 1] - _EPS:
                master.append(neighbour.at(value))
    _valid(master, "Neighbour polygon after endpoint insertion")
    if (Polygon(result).equals(Polygon(target.points))
            and tuple(master) == neighbour.points
            and all(point in target.points for row in accepted for point in row[2])):
        return unchanged
    return JoinPreview(tuple(result), tuple(master), tuple(row[3] for row in ordered))


def _deduplicate(points: Sequence[PointRC]) -> list[PointRC]:
    result = []
    for point in points:
        if not result or math.dist(point, result[-1]) > _EPS:
            result.append(point)
    if len(result) > 1 and math.dist(result[0], result[-1]) <= _EPS:
        result.pop()
    return result
