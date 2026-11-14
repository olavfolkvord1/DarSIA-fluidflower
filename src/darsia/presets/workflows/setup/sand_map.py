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

"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

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

    @classmethod
    def read(cls, path: Path) -> "SandMapDrawing":
        """Read a drawing from a JSON file.

        The file holds ``anchors`` (a list of [x, y] in metres), ``polygons`` (a
        list of [layer, [anchor index, ...]]) and optionally ``on_top`` (a list of
        layers).

        Parameters
        ----------
        path : Path
            Path to the drawing.

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
        )

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
        return SandMapDrawing(positions, self.polygons, self.on_top)


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
    anchors = np.asarray(drawing.anchors, dtype=np.float64)

    def area(indices: tuple[int, ...]) -> float:
        """Area of a polygon in the rig (shoelace formula)."""
        x, y = anchors[list(indices), 0], anchors[list(indices), 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    labels = np.zeros(shape, dtype=np.uint8)
    for layer, indices in sorted(
        drawing.polygons,
        key=lambda polygon: (polygon[0] in drawing.on_top, -area(polygon[1])),
    ):
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
