"""Sand maps: a drawing of a rig's sand layers placed on a run's grid.

A drawing holds the layers of a rig as polygons in rig coordinates: metres, x to
the right and y upwards from the lower left inner corner of the rig. The polygons
share their vertices (anchors) by index, so neighbouring layers meet on one
boundary: moving an anchor moves the boundary of every layer that uses it, and no
gap or overlap opens between them.

The rasteriser places a drawing on a run's grid through the run's rig geometry, a
homography from rig coordinates to the grid's pixel coordinates (column, row;
pixel centres at whole numbers), and gives the layer of each pixel:

1. Each polygon is filled by OpenCV's scanline fill at a fixed-point precision of
   1/16 pixel. Where polygons overlap, the smaller polygon (by its area in the
   rig) wins, unless the other one's layer is drawn on top; equal areas keep the
   drawing order.
2. Pixels outside the rig's inner frame, the rectangle of the rig's width and
   height, have no layer (0).
3. A pixel inside the frame that no polygon covers takes the layer of the nearest
   covered pixel (the lower layer on a tie) if both the frame's edge and that
   pixel lie within a set distance; polygons drawn up to the frame seldom reach it
   exactly. Any other uncovered pixel has no layer.

A run whose sand has settled keeps its own geometry: its adjustment moves anchors
of the common drawing for that run only (:meth:`SandMapDrawing.moved`).

A drawing is saved only as a closed partition of the rig's frame, the rectangle of
the rig's inner width and height that the rig's configuration gives: the polygons
of the layers not drawn on top cover every point of the frame exactly once, also
where the camera does not see, and each of their boundaries is shared with a
neighbouring polygon along the same anchors or lies on the frame's edge. Layers
drawn on top, e.g. a colour checker, lie inside the frame over the partition,
which continues beneath them. :func:`partition_faults` finds where a drawing is
not such a partition.

"""

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence

import cv2
import numpy as np
import shapely

import darsia

_SHIFT = 4
"""Fractional bits of the fixed-point polygon coordinates (1/16 pixel)."""


@dataclass(frozen=True)
class SandMapDrawing:
    """Sand layers as polygons on shared anchors in rig coordinates."""

    anchors: np.ndarray
    """Anchor positions (n, 2), x and y in metres."""
    polygons: tuple[tuple[int, tuple[int, ...]], ...]
    """Layer (1 to 255) and anchor indices of each polygon, in drawing order."""
    on_top: frozenset[int] = frozenset()
    """Layers whose polygons win over those of all other layers."""
    frame: tuple[float, float] | None = None
    """Width and height in metres of the rig's inner frame that the layers
    partition, from the rig's configuration; None if not given."""

    @classmethod
    def read(
        cls, path: Path, frame: tuple[float, float] | None = None
    ) -> "SandMapDrawing":
        """Read a drawing from a JSON file.

        The file holds ``anchors`` (a list of [x, y] in metres), ``polygons`` (a
        list of [layer, [anchor index, ...]]) and optionally ``on_top`` (a list of
        layers).

        Parameters
        ----------
        path : Path
            Path to the drawing.
        frame : tuple of float, optional
            Width and height in metres of the rig's inner frame, from the rig's
            configuration.

        Returns
        -------
        SandMapDrawing
            The drawing, with anchors in double precision.

        Raises
        ------
        ValueError
            If a layer lies outside 1 to 255, or a polygon has fewer than three
            anchors or names an anchor the drawing does not have.

        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        anchors = np.asarray(data["anchors"], dtype=np.float64).reshape(-1, 2)
        polygons = tuple(
            (int(layer), tuple(int(i) for i in indices))
            for layer, indices in data["polygons"]
        )
        for layer, indices in polygons:
            if (
                not 1 <= layer <= 255
                or len(indices) < 3
                or not all(0 <= i < len(anchors) for i in indices)
            ):
                raise ValueError(
                    f"A polygon needs a layer from 1 to 255 and at least three of "
                    f"the {len(anchors)} anchors; got layer {layer} with anchors "
                    f"{list(indices)}."
                )
        return cls(
            anchors=anchors,
            polygons=polygons,
            on_top=frozenset(int(layer) for layer in data.get("on_top", ())),
            frame=None if frame is None else (float(frame[0]), float(frame[1])),
        )

    def write(self, path: Path) -> None:
        """Save the drawing as a JSON file that :meth:`read` reads.

        The frame is not written; the rig's configuration gives it.

        Parameters
        ----------
        path : Path
            Path to the file.

        Raises
        ------
        ValueError
            If the drawing has no frame, or is not a closed partition of it
            (:func:`partition_faults`).

        """
        if self.frame is None:
            raise ValueError(
                "A drawing is saved as a partition of the rig's frame; give it "
                "the frame's width and height."
            )
        faults = partition_faults(self)
        if faults:
            raise ValueError(
                "The drawing is not a closed partition of the rig's frame: "
                + _described(faults)
                + "."
            )
        data = {
            "anchors": np.asarray(self.anchors, dtype=np.float64).tolist(),
            "polygons": [[layer, list(indices)] for layer, indices in self.polygons],
            "on_top": sorted(self.on_top),
        }
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    def moved(self, anchors: Mapping[int, Sequence[float]]) -> "SandMapDrawing":
        """The drawing with some anchors at new positions, e.g., a run's adjustment.

        Every polygon that uses a moved anchor follows it, so neighbouring layers
        stay joined.

        Parameters
        ----------
        anchors : mapping
            New position (x, y) in metres of each moved anchor, by index.

        Returns
        -------
        SandMapDrawing
            A new drawing; this one is unchanged.

        Raises
        ------
        ValueError
            If an index names an anchor the drawing does not have.

        """
        positions = self.anchors.copy()
        for index, position in anchors.items():
            if not 0 <= index < len(positions):
                raise ValueError(
                    f"The drawing has no anchor {index} ({len(positions)} anchors)."
                )
            positions[index] = position
        return replace(self, anchors=positions)


def read_moved_anchors(path: Path) -> dict[int, tuple[float, float]]:
    """Read a run's adjustment, the anchors it moves, from a JSON file.

    Parameters
    ----------
    path : Path
        Path to a file holding ``anchors``, a mapping of anchor index to [x, y]
        in metres.

    Returns
    -------
    dict
        New position of each moved anchor, by index.

    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(index): (float(x), float(y)) for index, (x, y) in data["anchors"].items()
    }


@dataclass(frozen=True)
class PartitionFault:
    """A place where a drawing is not a closed partition of its frame."""

    kind: Literal["gap", "overlap", "outside", "crossing", "unlinked"]
    """``gap``: part of the frame that no polygon covers; ``overlap``: part that
    two polygons not on top cover, or two on top; ``outside``: part of a polygon
    beyond the frame; ``crossing``: a point where a polygon's boundary meets
    itself; ``unlinked``: an edge that is neither shared with a neighbour nor on
    the frame's edge and bounds none of the other faults."""
    area: float
    """Area in square metres; 0 for a crossing or an unlinked edge."""
    bounds: tuple[float, float, float, float]
    """Smallest x, smallest y, largest x and largest y of the place, in metres."""
    layers: tuple[int, ...]
    """Layers of the polygons at the place: those bordering a gap along a stretch,
    those that overlap or reach outside, the polygon of a crossing or an edge."""


def partition_faults(drawing: SandMapDrawing) -> list[PartitionFault]:
    """Where a drawing is not a closed partition of its frame.

    The drawing is a closed partition if its polygons are simple, those on top lie
    inside the frame without overlapping each other, and each edge of the others
    is either shared by exactly one other of them, in the opposite direction, or
    lies on the frame's edge, which these edges then cover exactly once. The test
    is exact: it compares anchors and coordinates, without a tolerance or pixels.
    Its edges then bound the frame and nothing else, so the polygons not on top
    cover each point of the frame once. Otherwise the faults are measured by
    polygon overlays in double precision.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, with its frame.

    Returns
    -------
    list of PartitionFault
        The faults, the largest area first; empty for a closed partition.

    Raises
    ------
    ValueError
        If the drawing has no frame.

    """
    if drawing.frame is None:
        raise ValueError("The drawing has no frame to partition.")
    points = np.asarray(drawing.anchors, dtype=np.float64)
    rings = [_ring(points, indices) for _layer, indices in drawing.polygons]
    layers = [layer for layer, _indices in drawing.polygons]
    top = [layer in drawing.on_top for layer in layers]
    width, height = drawing.frame
    unlinked, sides = _boundary(points, rings, top, drawing.frame)
    crossings = [
        _crossings(points[ring]) if len(ring) > 2 else np.zeros((0, 2))
        for ring in rings
    ]
    shapes = [
        shapely.make_valid(shapely.Polygon(points[ring])) if len(ring) > 2 else None
        for ring in rings
    ]
    frame = shapely.box(0, 0, width, height)
    base = [k for k in range(len(rings)) if not top[k]]
    above = [k for k in range(len(rings)) if top[k]]

    def overlaps(group: list[int]) -> list[tuple[shapely.Geometry, tuple[int, int]]]:
        """The intersections of each pair of the polygons."""
        return [
            (shapes[i].intersection(shapes[j]), (i, j))
            for n, i in enumerate(group)
            for j in group[n + 1 :]
            if shapes[i] is not None
            and shapes[j] is not None
            and shapes[i].intersects(shapes[j])
        ]

    if (
        not unlinked
        and all(
            ends == [0.0, width if side % 2 == 0 else height]
            for side, ends in sides.items()
        )
        and all(
            len(ring) > 2 and len(set(ring)) == len(ring) and not len(found)
            for ring, found in zip(rings, crossings)
        )
        and all(frame.covers(shapes[k]) for k in above)
        and not any(geometry.area > 0 for geometry, _pair in overlaps(above))
    ):
        return []

    places = [
        ("gap", frame.difference(shapely.union_all([shapes[k] for k in base])), base),
        *(("overlap", *found) for found in overlaps(base) + overlaps(above)),
        (
            "outside",
            shapely.union_all(
                [shape for shape in shapes if shape is not None]
            ).difference(frame),
            range(len(rings)),
        ),
    ]
    faults, pieces = [], []
    for kind, geometry, near in places:
        for piece in shapely.get_parts(geometry):
            if piece.geom_type != "Polygon" or not piece.area > 0:
                continue
            shared = [
                shapes[k].intersection(piece) if shapes[k] is not None else None
                for k in near
            ]
            nearby = {
                layers[k]
                for k, part in zip(near, shared)
                if part is not None
                and (part.length if kind == "gap" else part.area) > 0
            }
            faults.append(
                PartitionFault(
                    kind,
                    float(piece.area),
                    tuple(float(value) for value in piece.bounds),
                    tuple(sorted(nearby)),
                )
            )
            pieces.append(piece)
    for k, found in enumerate(crossings):
        for x, y in found.tolist():
            faults.append(PartitionFault("crossing", 0.0, (x, y, x, y), (layers[k],)))
            pieces.append(shapely.Point(x, y))
    measured = shapely.union_all(pieces)
    for k, start, end in unlinked:
        if measured.is_empty or shapely.LineString(points[[start, end]]).distance(
            measured
        ):
            (x0, y0), (x1, y1) = points[start], points[end]
            bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            faults.append(PartitionFault("unlinked", 0.0, bounds, (layers[k],)))
    return sorted(faults, key=lambda fault: -fault.area)


def _described(faults: Sequence[PartitionFault]) -> str:
    """The faults of a drawing in words: how many of each kind, and the largest
    with its area and place in millimetres."""
    counts = Counter(fault.kind for fault in faults)
    largest = max(faults, key=lambda fault: fault.area)
    x0, y0, x1, y1 = (1000 * value for value in largest.bounds)
    return (
        ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
        + f"; the largest is a {largest.kind} of {largest.area * 1e6:.3f} mm2 "
        f"at x {x0:.1f} to {x1:.1f} mm, y {y0:.1f} to {y1:.1f} mm"
    )


def _ring(points: np.ndarray, indices: Sequence[int]) -> list[int]:
    """A polygon's anchors counter-clockwise, without repeats of the one before."""
    ring = [i for n, i in enumerate(indices) if i != indices[n - 1]]
    x, y = points[ring, 0], points[ring, 1]
    if np.dot(x, np.roll(y, -1)) < np.dot(y, np.roll(x, -1)):
        ring.reverse()
    return ring


def _boundary(
    points: np.ndarray,
    rings: list[list[int]],
    top: list[bool],
    frame: tuple[float, float],
) -> tuple[list[tuple[int, int, int]], dict[int, list[float]]]:
    """The unlinked edges (polygon, start, end) of the polygons not on top, and the
    stretches the frame's edge is covered by, per side (bottom, right, top, left).

    An edge is unlinked if no other polygon has it in the opposite direction,
    or a polygon has it twice, and it does not lie on the frame's edge in the
    frame's counter-clockwise direction. A side's stretches are given by their
    ends along the side, merged where they meet; a side covered exactly once is
    [0, its length]. Coordinates are compared exactly.

    """
    width, height = frame
    edges = Counter(
        (start, end)
        for ring, on_top in zip(rings, top)
        if not on_top
        for start, end in zip(ring, ring[1:] + ring[:1])
    )
    unlinked = []
    stretches: dict[int, list[tuple[float, float]]] = {side: [] for side in range(4)}
    for k, ring in enumerate(rings):
        if top[k]:
            continue
        for start, end in zip(ring, ring[1:] + ring[:1]):
            if edges[(end, start)] == 1 and edges[(start, end)] == 1:
                continue
            (x0, y0), (x1, y1) = points[start], points[end]
            if y0 == y1 == 0 and x0 < x1:
                stretches[0].append((x0, x1))
            elif x0 == x1 == width and y0 < y1:
                stretches[1].append((y0, y1))
            elif y0 == y1 == height and x1 < x0:
                stretches[2].append((x1, x0))
            elif x0 == x1 == 0 and y1 < y0:
                stretches[3].append((y1, y0))
            else:
                unlinked.append((k, start, end))
    sides = {}
    for side, found in stretches.items():
        ends: list[float] = []
        for low, high in sorted(found):
            if ends and ends[-1] == low:
                ends[-1] = high
            else:
                ends += [low, high]
        sides[side] = ends
    return unlinked, sides


def _crossings(points: np.ndarray) -> np.ndarray:
    """Points where a closed boundary meets itself, other than where consecutive
    edges join."""
    noded = shapely.get_parts(shapely.node(shapely.LinearRing(points)))
    ends = np.array([shapely.get_coordinates(part)[[0, -1]] for part in noded]).reshape(
        -1, 2
    )
    nodes, counts = np.unique(ends, axis=0, return_counts=True)
    return nodes[counts > 2]


def rig_grid(grid: darsia.Image) -> np.ndarray:
    """Homography from rig coordinates to the pixels of a grid in rig metres.

    The grid's pixel centres (:func:`darsia.pixel_centres`) map onto whole
    numbers. This is the rig geometry of a run whose grid is the rig's own metric
    grid, e.g. the target grid of the rig's image mapping.

    Parameters
    ----------
    grid : darsia.Image
        An image on the grid, with its physical dimensions and origin in rig
        metres.

    Returns
    -------
    array
        3 x 3 matrix mapping (x, y, 1) to (column, row, 1), with the top row first
        and pixel centres at whole numbers.

    """
    x, y = darsia.pixel_centres(grid)
    dy, dx = grid.voxel_size
    return np.array([[1 / dx, 0, -x[0] / dx], [0, -1 / dy, y[0] / dy], [0, 0, 1]])


def grid_homographies(
    grid_to_rig: Sequence[Sequence[float]] | np.ndarray | None,
    grid: darsia.Image | None,
) -> tuple[np.ndarray, np.ndarray]:
    """A run grid's homographies between its pixel coordinates (column, row, 1)
    and rig metres: its measured model and the model's inverse where the run has
    one (``[labeling.sand_map] grid_to_rig``), else those of a grid in rig metres
    (:func:`rig_grid`), in double precision.

    Parameters
    ----------
    grid_to_rig : array, optional
        The run's measured homography from its grid's pixels to rig metres.
    grid : darsia.Image, optional
        An image on the run's grid; needed without a model.

    Returns
    -------
    tuple of arrays
        The homography from the grid's pixels to rig metres, and from rig metres
        to the grid's pixels.

    """
    if grid_to_rig is None:
        to_grid = rig_grid(grid)
        return np.linalg.inv(to_grid), to_grid
    to_rig = np.asarray(grid_to_rig, dtype=float)
    return to_rig, np.linalg.inv(to_rig)


def _project(points: np.ndarray, rig_to_grid: np.ndarray) -> np.ndarray:
    """Pixel coordinates (column, row) of rig points, in the points' dtype."""
    matrix = np.asarray(rig_to_grid, dtype=points.dtype)
    homogeneous = np.column_stack([points, np.ones(len(points), points.dtype)])
    projected = homogeneous @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def _fill_order(drawing: SandMapDrawing) -> list[int]:
    """Indices of a drawing's polygons in the order they are filled, so that a later
    one wins where they overlap: the larger polygon (by its area in the rig, in
    double precision) first, those drawn on top last, and drawing order on ties."""
    anchors = np.asarray(drawing.anchors, dtype=np.float64)

    def area(indices: tuple[int, ...]) -> float:
        """Area of a polygon in the rig (shoelace formula)."""
        x, y = anchors[list(indices), 0], anchors[list(indices), 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    return sorted(
        range(len(drawing.polygons)),
        key=lambda k: (
            drawing.polygons[k][0] in drawing.on_top,
            -area(drawing.polygons[k][1]),
        ),
    )


def _fill(target: np.ndarray, points: np.ndarray, value: int) -> None:
    """Fill a polygon given in pixel coordinates at 1/16 pixel."""
    fixed = np.rint(points * (1 << _SHIFT)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(target, [fixed], int(value), lineType=cv2.LINE_8, shift=_SHIFT)


def _distance_to(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in pixels to the nearest pixel of a nonempty mask."""
    return cv2.distanceTransform(
        np.ascontiguousarray(~mask, dtype=np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def _edge_fill(labels: np.ndarray, inside: np.ndarray, edge_px: float) -> None:
    """Give uncovered pixels near the frame's edge the nearest covered layer."""
    padded = np.pad(inside, 1, constant_values=False)
    to_edge = _distance_to(~padded)[1:-1, 1:-1]
    candidate = inside & (labels == 0) & (to_edge <= edge_px)
    if not candidate.any():
        return
    nearest = np.zeros(np.count_nonzero(candidate), dtype=labels.dtype)
    distance = np.full(nearest.shape, np.inf, dtype=np.float32)
    for layer in np.unique(labels[inside]):
        if layer == 0:
            continue
        values = _distance_to((labels == layer) & inside)[candidate]
        closer = values < distance
        distance[closer] = values[closer]
        nearest[closer] = layer
    labels[candidate] = np.where(distance <= edge_px, nearest, 0)


def rasterize_sand_map(
    drawing: SandMapDrawing,
    shape: tuple[int, int],
    rig_to_grid: np.ndarray,
    *,
    width: float,
    height: float,
    edge_fill_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a drawing on a run's grid: the layer of each pixel.

    The anchors are projected in their floating dtype. Areas are compared in
    double precision. The edge fill's distance is converted to pixels with the
    frame's mean number of pixels per metre along its edges.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, possibly with a run's adjustment.
    shape : tuple of int
        Rows and columns of the run's grid.
    rig_to_grid : array
        The run's rig geometry: 3 x 3 homography from rig coordinates (x, y in
        metres) to the grid's pixel coordinates (column, row).
    width, height : float
        Width and height of the rig's inner frame in metres.
    edge_fill_mm : float
        Distance from the frame's edge within which uncovered pixels take the
        nearest covered layer; 0 for none.

    Returns
    -------
    labels : array
        Layer of each pixel (uint8); 0 where no polygon or edge fill reaches and
        outside the frame.
    inside : array
        Pixels inside the frame (bool).

    """
    dtype = np.result_type(drawing.anchors, np.float32)
    corners = [[0, height], [width, height], [width, 0], [0, 0]]
    frame = _project(np.array(corners, dtype=dtype), rig_to_grid)
    inside = np.zeros(shape, dtype=np.uint8)
    _fill(inside, frame, 1)
    inside = inside.astype(bool)

    pixels = _project(drawing.anchors.astype(dtype, copy=False), rig_to_grid)
    labels = np.zeros(shape, dtype=np.uint8)
    for k in _fill_order(drawing):
        layer, indices = drawing.polygons[k]
        _fill(labels, pixels[list(indices)], layer)
    labels[~inside] = 0

    if edge_fill_mm > 0:
        edges = frame.astype(np.float64)
        along = np.linalg.norm(edges - np.roll(edges, -1, axis=0), axis=1)
        per_metre = np.sqrt(
            0.5 * (along[0] + along[2]) / width * (0.5 * (along[3] + along[1]) / height)
        )
        _edge_fill(labels, inside, edge_fill_mm * 1e-3 * per_metre)
    return labels, inside
