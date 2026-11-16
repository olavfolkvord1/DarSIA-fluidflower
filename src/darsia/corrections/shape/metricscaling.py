"""Scaling onto whole metric cells: one grid and one set of reducers.

A canvas of given physical dimensions is covered by whole square cells of one
size, counted from the canvas' lower left corner; the canvas is extended upwards
and to the right to whole cells. Each pixel belongs to the cell that holds its
centre (half-open cells). Intensive values are averaged over the valid pixels of
each cell, weighted by pixel area, extensive values (mass, volume, area) are
summed over each cell, and a label map takes the label of the largest area.
Nothing is interpolated, so 1 mm, 5 mm or any other size is the same scaling.

"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path

import numpy as np

import darsia


@dataclass(frozen=True)
class MetricCells:
    """Whole metric cells covering a canvas, and the cell of each canvas pixel."""

    shape: tuple[int, int]
    """Rows and columns of cells, the top row first."""
    dimensions: list[float]
    """Height and width of the cells' extent, in metres (whole cells)."""
    index: np.ndarray
    """Flat index of the cell of each canvas pixel, in row-major order; -1 for a
    pixel without a cell (:func:`placed_cells`)."""
    pixel_area: float
    """Area of a canvas pixel in square metres."""


def measured_extent(
    corners: Sequence[tuple[np.ndarray, np.ndarray]],
    cell_size_mm: float,
    edge_mm: int,
) -> tuple[float, float, float, float]:
    """The whole cells of a size that cover given positions, e.g. the corner
    pixels' measured positions of the compared runs' measurement canvases
    (:func:`darsia.grid_corners`): x0, x1, y0, y1 [m]. The positions' extent is
    first taken in whole steps of ``edge_mm`` (the step at or below the smallest
    position and the one past the largest), then rounded outward to whole cells in
    exact decimal arithmetic, as our production took its comparison's cells from
    its frames' measured corners (in 10 mm steps).

    Parameters
    ----------
    corners : sequence of (array, array)
        x and y [m] of positions, e.g. one pair per compared run.
    cell_size_mm : float
        Cell size in millimetres.
    edge_mm : int
        Step [mm] of the extent's edges before its whole cells.

    Returns
    -------
    tuple of float
        The extent x0, x1, y0, y1 [m], on whole cells counted from the origin.

    """
    per_metre = 1000 // int(edge_mm)
    step = Decimal(str(float(cell_size_mm))) / Decimal(1000)
    extent = []
    for axis in (0, 1):
        values = np.concatenate([np.ravel(pair[axis]) for pair in corners])
        edges = (
            math.floor(float(np.min(values)) * per_metre) / per_metre,
            (math.floor(float(np.max(values)) * per_metre) + 1) / per_metre,
        )
        extent += [
            float(
                (Decimal(str(edge)) / step).to_integral_value(rounding=rounding) * step
            )
            for edge, rounding in zip(edges, (ROUND_FLOOR, ROUND_CEILING))
        ]
    return tuple(extent)


def metric_cells(
    native_shape: tuple[int, int],
    dimensions: list[float],
    cell_size_mm: float,
    positions: "darsia.GridPositions | None" = None,
    extent: tuple[float, float, float, float] | None = None,
) -> MetricCells:
    """Cover a canvas with whole metric cells and assign its pixels to them.

    The number of cells per axis is the canvas length divided by the cell size,
    rounded up in exact decimal arithmetic, so the cell size is exact and the
    canvas is extended to whole cells. Pixel centres are measured from the
    canvas' lower left corner; a pixel belongs to the cell whose half-open
    interval holds its centre.

    With each pixel's measured position on the rig (``positions``) and the
    cells' extent in rig metres (``extent``, e.g. :func:`measured_extent`), a pixel
    belongs to the cell that holds its position, its offset from the extent's
    lower left corner divided by the cell size in the positions' precision, as our
    production assigned its frames' pixels; a position outside the extent is
    refused.

    Parameters
    ----------
    native_shape : tuple of int
        Rows and columns of the canvas.
    dimensions : list of float
        Height and width of the canvas in metres.
    cell_size_mm : float
        Cell size in millimetres.
    positions : darsia.GridPositions, optional
        Each pixel's position on the rig.
    extent : tuple of float, optional
        x0, x1, y0, y1 [m] of the cells, whole cells; needed with positions.

    Returns
    -------
    MetricCells
        The cells and the cell of each pixel.

    Raises
    ------
    ValueError
        If a position lies outside the extent, or positions come without one.

    """
    rows, cols = (int(n) for n in native_shape)
    height, width = (float(length) for length in dimensions)
    step = Decimal(str(float(cell_size_mm))) / Decimal(1000)
    if positions is not None:
        if extent is None:
            raise ValueError("Measured positions need the cells' extent")
        return _measured_cells(rows, cols, height, width, step, positions, extent)
    counts = [
        int((Decimal(str(length)) / step).to_integral_value(rounding=ROUND_CEILING))
        for length in (height, width)
    ]
    dy, dx = np.float64(height) / rows, np.float64(width) / cols
    x = (np.arange(cols, dtype=np.float64) + 0.5) * dx
    y = np.float64(height) - (np.arange(rows, dtype=np.float64) + 0.5) * dy
    size = float(cell_size_mm) / 1000.0
    column = np.floor(x / size).astype(np.int64)
    row = counts[0] - 1 - np.floor(y / size).astype(np.int64)
    return MetricCells(
        shape=tuple(counts),
        dimensions=[float(count * step) for count in counts],
        index=(row[:, None] * counts[1] + column[None, :]).ravel(),
        pixel_area=float(dx * dy),
    )


def _measured_cells(rows, cols, height, width, step, positions, extent) -> MetricCells:
    """The cells of :func:`metric_cells` by each pixel's measured position."""
    x0, x1, y0, y1 = (float(value) for value in extent)
    counts = [
        int(((Decimal(str(high)) - Decimal(str(low))) / step).to_integral_value())
        for low, high in ((y0, y1), (x0, x1))
    ]
    size = float(step)
    column = np.floor((positions.x - x0) / size).astype(np.int64)
    row = np.floor((positions.y - y0) / size).astype(np.int64)
    if np.any((column < 0) | (column >= counts[1]) | (row < 0) | (row >= counts[0])):
        raise ValueError("A pixel's measured position lies outside the cells' extent")
    dy, dx = np.float64(height) / rows, np.float64(width) / cols
    return MetricCells(
        shape=tuple(counts),
        dimensions=[float(count * step) for count in counts],
        index=((counts[0] - 1 - row) * counts[1] + column).ravel(),
        pixel_area=float(dx * dy),
    )


def placed_cells(
    cells: MetricCells,
    cell_size_mm: float,
    extent: tuple[float, float, float, float],
    onto: tuple[float, float, float, float],
) -> MetricCells:
    """Cells of one extent placed onto the cells of another of the same size.

    Each pixel keeps its cell, which moves by the whole cells between the two
    extents' lower left corners, counted in exact decimal arithmetic. A pixel
    whose cell lies outside ``onto`` gets none (index -1), and counts in no
    cell's sum or mean. This is how our production placed its reference run's
    pixels: first on cells over the reference run's own positions
    (:func:`measured_extent`), then onto the comparison's cells, cut to them.

    Parameters
    ----------
    cells : MetricCells
        Cells over ``extent`` (:func:`metric_cells`).
    cell_size_mm : float
        Cell size in millimetres.
    extent, onto : tuple of float
        x0, x1, y0, y1 [m] of the given cells and of the cells to place them on.

    Returns
    -------
    MetricCells
        The cells over ``onto`` and the cell of each pixel, -1 for none.

    Raises
    ------
    ValueError
        If the extents do not share cell edges.

    """
    step = Decimal(str(float(cell_size_mm))) / Decimal(1000)

    def whole(length: Decimal) -> int:
        ratio = length / step
        if ratio != ratio.to_integral_value():
            raise ValueError("The cells' extents do not share cell edges")
        return int(ratio)

    x0, _, y0, _ = (Decimal(str(float(value))) for value in extent)
    u0, u1, v0, v1 = (Decimal(str(float(value))) for value in onto)
    counts = [whole(v1 - v0), whole(u1 - u0)]
    rows, cols = cells.shape
    row = rows - 1 - cells.index // cols + whole(y0 - v0)
    column = cells.index % cols + whole(x0 - u0)
    inside = (column >= 0) & (column < counts[1]) & (row >= 0) & (row < counts[0])
    inside &= cells.index >= 0
    return MetricCells(
        shape=tuple(counts),
        dimensions=[float(count * step) for count in counts],
        index=np.where(inside, (counts[0] - 1 - row) * counts[1] + column, -1),
        pixel_area=cells.pixel_area,
    )


def _cell_sums(cells: MetricCells, values: np.ndarray) -> np.ndarray:
    """Sums per cell of values (pixels, channels), in their dtype and pixel order;
    a pixel without a cell (index -1) counts in none."""
    index = cells.index
    kept = index >= 0
    if not kept.all():
        index, values = index[kept], values[kept]
    sums = np.zeros((values.shape[1], cells.shape[0] * cells.shape[1]), values.dtype)
    for channel, total in enumerate(sums):
        np.add.at(total, index, values[:, channel])
    return sums.T


def cell_sums(cells: MetricCells, values: np.ndarray) -> np.ndarray:
    """Sum of an extensive quantity (e.g., mass, volume or area) over each cell.

    The sums run in double precision whatever the dtype of the values, in pixel
    order, as our production summed its cell copies, so that the cells of a
    single-precision map add up to its total in double precision.

    Parameters
    ----------
    cells : MetricCells
        Cells of the values' canvas.
    values : array
        Values (rows, cols, ...) on the canvas: each pixel's share of the quantity.

    Returns
    -------
    array
        Sums (cell rows, cell cols, ...) in double precision; zero in an empty cell.

    """
    values = np.asarray(values)
    value = values.reshape(len(cells.index), -1).astype(np.float64, copy=False)
    return _cell_sums(cells, value).reshape(*cells.shape, *values.shape[2:])


def cell_means(
    cells: MetricCells, values: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Mean of an intensive image over the valid pixels of each cell.

    Each channel is averaged over the pixels where it is finite, weighted by
    pixel area: the cells' pixel area, or each pixel's own (e.g. its measured
    area). The sums run in the floating dtype of the image and the weights, at
    least single precision, in pixel order.

    Parameters
    ----------
    cells : MetricCells
        Cells of the image's canvas.
    values : array
        Image (rows, cols, ...) on the canvas.
    weights : array, optional
        Each pixel's area (rows, cols) on the canvas; by default the cells'
        pixel area.

    Returns
    -------
    array
        Means (cell rows, cell cols, ...); NaN in a cell without a valid pixel.

    """
    values = np.asarray(values)
    if weights is None:
        dtype = np.result_type(values, np.float32)
        weight = dtype.type(cells.pixel_area)
    else:
        dtype = np.result_type(values, weights, np.float32)
        weight = np.asarray(weights, dtype).reshape(len(cells.index), 1)
    value = values.reshape(len(cells.index), -1).astype(dtype, copy=False)
    valid = np.isfinite(value)
    numerator = _cell_sums(cells, np.where(valid, value * weight, 0))
    denominator = _cell_sums(cells, np.where(valid, weight, 0))
    means = np.full(numerator.shape, np.nan, dtype)
    np.divide(numerator, denominator, out=means, where=denominator > 0)
    return means.reshape(*cells.shape, *values.shape[2:])


def cell_fractions(cells: MetricCells, mask: np.ndarray) -> np.ndarray:
    """Fraction of the pixel area of each cell covered by a mask, in double precision.

    Parameters
    ----------
    cells : MetricCells
        Cells of the mask's canvas.
    mask : array
        Boolean mask (rows, cols) on the canvas.

    Returns
    -------
    array
        Fractions (cell rows, cell cols); NaN in a cell without pixels.

    """
    area = np.full((len(cells.index), 1), cells.pixel_area)
    covered = _cell_sums(cells, np.where(np.reshape(mask, (-1, 1)), area, 0.0))
    fractions = np.full(covered.shape, np.nan)
    total = _cell_sums(cells, area)
    np.divide(covered, total, out=fractions, where=total > 0)
    return fractions.reshape(cells.shape)


def cell_majority(cells: MetricCells, labels: np.ndarray) -> np.ndarray:
    """Label of the largest area fraction of each cell.

    Parameters
    ----------
    cells : MetricCells
        Cells of the label map's canvas.
    labels : array
        Integer labels (rows, cols) on the canvas, e.g., facies.

    Returns
    -------
    array
        Labels (cell rows, cell cols); on a tie the lowest label, and zero in a
        cell without pixels.

    """
    labels = np.asarray(labels)
    majority = np.zeros(cells.shape, labels.dtype)
    largest = np.zeros(cells.shape)
    for label in np.unique(labels):
        fraction = np.nan_to_num(cell_fractions(cells, labels == label))
        larger = fraction > largest
        majority[larger] = label
        largest[larger] = fraction[larger]
    return majority


class MetricScalingCorrection(darsia.BaseCorrection):
    """Scale images onto whole metric cells of one size (:func:`metric_cells`).

    Images are averaged over each cell (:func:`cell_means`); label maps take
    each cell's majority label (:func:`cell_majority`). The image's dimensions
    become the cells' extent, from the same lower left corner. The scaling keeps
    its canvas (shape, dimensions and origin), the extent before the cells extend
    it.

    """

    def __init__(
        self,
        cell_size_mm: float | None = None,
        native_shape: tuple[int, int] | None = None,
        dimensions: list[float] | None = None,
        origin: list[float] | None = None,
        labels: bool = False,
    ) -> None:
        """Initialize the scaling, or an empty object to be loaded from file.

        Parameters
        ----------
        cell_size_mm : float, optional
            Cell size in millimetres.
        native_shape : tuple of int, optional
            Rows and columns of the images to be scaled.
        dimensions : list of float, optional
            Height and width of their canvas in metres.
        origin : list of float, optional
            Top left corner of their canvas in metres.
        labels : bool
            Whether the images are label maps, scaled by majority.

        """
        if cell_size_mm is None:
            return
        self.cell_size_mm = float(cell_size_mm)
        """Cell size in millimetres."""
        self.native_shape = tuple(int(n) for n in native_shape)
        """Rows and columns of the images to be scaled."""
        self.native_dimensions = [float(length) for length in dimensions]
        """Height and width of the canvas of the images to be scaled."""
        self.native_origin = [float(x) for x in origin]
        """Top left corner of the canvas of the images to be scaled."""
        self.labels = bool(labels)
        """Whether the images are label maps."""
        self.cells = metric_cells(
            self.native_shape, self.native_dimensions, self.cell_size_mm
        )
        """The cells and the cell of each canvas pixel."""

    def correct_array(self, image: np.ndarray) -> np.ndarray:
        """Scale an image or a label map onto the cells.

        Parameters
        ----------
        image : array
            Image or label map on the canvas.

        Returns
        -------
        array
            Cell means, or majority labels, on the cells.

        """
        if self.labels:
            return cell_majority(self.cells, image)
        return cell_means(self.cells, image)

    def correct_metadata(self, metadata: dict = {}) -> dict:
        """Extent of the cells, from the canvas' lower left corner.

        Parameters
        ----------
        metadata : dict
            Metadata of the input image (not used).

        Returns
        -------
        dict
            Dimensions and origin of the cells.

        """
        x, y = self.native_origin
        return {
            "dimensions": list(self.cells.dimensions),
            "origin": darsia.make_coordinate(
                [x, y - self.native_dimensions[0] + self.cells.dimensions[0]]
            ),
        }

    # ! ---- I/O ----

    def save(self, path: Path) -> None:
        """Save the scaling to a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        np.savez(
            path,
            class_name=type(self).__name__,
            cell_size_mm=self.cell_size_mm,
            native_shape=np.array(self.native_shape),
            dimensions=np.array(self.native_dimensions),
            origin=np.array(self.native_origin),
            labels=self.labels,
        )

    def load(self, path: Path) -> None:
        """Load the scaling from a npz file.

        Parameters
        ----------
        path : Path
            Path to the file.

        """
        with np.load(path) as data:
            self.__init__(
                cell_size_mm=float(data["cell_size_mm"]),
                native_shape=tuple(data["native_shape"]),
                dimensions=list(data["dimensions"]),
                origin=list(data["origin"]),
                labels=bool(data["labels"]),
            )
