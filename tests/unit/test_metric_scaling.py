"""Unit tests for the scaling onto whole metric cells."""

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import numpy as np
import pytest

import darsia

ROWS = [slice(0, 2), slice(2, 6)]
COLUMNS = [slice(0, 4), slice(4, 8), slice(8, 9)]
"""Pixels of each 0.4 m cell on a 0.6 x 0.9 m canvas of 6 x 9 pixels."""


def test_metric_cells_cover_the_canvas_from_its_lower_left_corner():
    """Whole cells of the exact size; each pixel lies in the cell of its centre."""
    cells = darsia.metric_cells((6, 9), [0.6, 0.9], 400.0)

    assert cells.shape == (2, 3)
    assert cells.dimensions == [0.8, 1.2]
    expected = np.zeros((6, 9), int)
    for i, rows in enumerate(ROWS):
        for j, columns in enumerate(COLUMNS):
            expected[rows, columns] = 3 * i + j
    np.testing.assert_array_equal(cells.index.reshape(6, 9), expected)


def test_metric_cells_by_measured_positions_hold_each_pixels_position():
    """With measured positions, the cells cover every given grid's corners (the
    whole cells at or below the smallest and one past the largest position), and a
    pixel goes to the cell of its position, its offset divided by the cell size in
    the positions' precision; a position outside the extent is refused."""
    image = darsia.Image(np.zeros((30, 45)), dimensions=[0.03, 0.045])
    model = [[0.001002, 0.00003, 0.0004], [0.00002, -0.000997, 0.0294], [0, 0, 1.0]]
    positions = darsia.grid_positions(image, model, np.float32)
    corners = darsia.grid_corners(image, model, np.float32)
    for k, (r, c) in enumerate(((0, 0), (0, 44), (29, 0), (29, 44))):
        assert (corners[0][k], corners[1][k]) == (positions.x[r, c], positions.y[r, c])
    plain = darsia.grid_corners(image)
    extent = darsia.measured_extent([corners, plain], 3.0, 10)
    xs, ys = (np.concatenate([corners[a], plain[a]]) for a in (0, 1))
    step, expected = Decimal("0.003"), []
    for values in (xs, ys):  # in 10 mm steps, then out to whole 3 mm cells
        low = math.floor(float(values.min()) * 100) / 100
        high = (math.floor(float(values.max()) * 100) + 1) / 100
        for edge, rounding in ((low, ROUND_FLOOR), (high, ROUND_CEILING)):
            whole = (Decimal(str(edge)) / step).to_integral_value(rounding=rounding)
            expected.append(float(whole * step))
    assert extent == tuple(expected)
    assert darsia.measured_extent([plain], 3.0, 10) == (0.0, 0.051, 0.0, 0.03)
    assert darsia.measured_extent([plain], 3.0, 5) == (0.0, 0.045, 0.0, 0.03)
    # The corners of a measurement canvas around the grid are the model at its
    # corner pixels.
    canvas = darsia.MeasurementCanvas((-6, -4), (38, 57))
    around = darsia.grid_corners(image, model, np.float32, canvas)
    m = np.asarray(model, np.float32)
    cx = np.array([-6, 50, -6, 50], np.float32)
    cy = np.array([-4, -4, 33, 33], np.float32)
    expected_corners = darsia.project_xy(m, cx, cy)
    for axis in (0, 1):
        np.testing.assert_array_equal(around[axis], expected_corners[axis])
    assert darsia.measured_extent([around], 3.0, 10) != darsia.measured_extent(
        [corners], 3.0, 10
    )
    with pytest.raises(ValueError, match="does not hold the grid"):
        darsia.grid_corners(
            image, model, np.float32, darsia.MeasurementCanvas((1, 0), (30, 45))
        )
    cells = darsia.metric_cells((30, 45), [0.03, 0.045], 3.0, positions, extent)
    rows, cols = round((extent[3] - extent[2]) / 0.003), round(
        (extent[1] - extent[0]) / 0.003
    )
    col = np.floor((positions.x - extent[0]) / 0.003).astype(np.int64)
    row = np.floor((positions.y - extent[2]) / 0.003).astype(np.int64)
    assert cells.shape == (rows, cols)
    np.testing.assert_array_equal(cells.index, ((rows - 1 - row) * cols + col).ravel())
    nominal = darsia.metric_cells((30, 45), [0.03, 0.045], 3.0)
    assert not np.array_equal(cells.index, nominal.index)

    # A position whose cell single and double precision tell apart: single.
    rng = np.random.default_rng(4)
    near = (np.arange(1, 20000) * 0.003 + rng.uniform(-1e-9, 1e-9, 19999)) % 0.03
    near = near.astype(np.float32)
    single = np.floor((near - 0.0) / 0.003).astype(np.int64)
    double = np.floor((near.astype(np.float64) - 0.0) / 0.003).astype(np.int64)
    k = int(np.flatnonzero(single != double)[0])
    pair = darsia.GridPositions(
        np.array([[near[k]]], np.float32), np.array([[0.001]], np.float32), (1.0, 1.0)
    )
    edge = darsia.metric_cells(
        (1, 1), [0.001, 0.001], 3.0, pair, (0.0, 0.03, 0.0, 0.003)
    )
    assert edge.index[0] == single[k] != double[k]
    with pytest.raises(ValueError, match="extent"):
        darsia.metric_cells((30, 45), [0.03, 0.045], 3.0, positions)
    with pytest.raises(ValueError, match="outside the cells' extent"):
        darsia.metric_cells(
            (30, 45), [0.03, 0.045], 3.0, positions, (0.0, 0.03, 0.0, 0.03)
        )


def test_placed_cells_keep_each_pixels_cell_on_larger_cells():
    """Cells over a canvas' own positions placed onto cells over a larger extent:
    each pixel keeps its cell, moved by the whole cells between the extents' lower
    left corners, where counting on the larger cells directly can give another in
    single precision. Placed onto fewer cells, a pixel whose cell lies outside
    gets none and counts in no sum or mean. Extents without common cell edges are
    refused."""
    x = np.array([[0.0029999993, 0.006, 0.009000001, 0.0151]], np.float32)
    y = np.array([[0.001, 0.0045, 0.0061, 0.0119]], np.float32)
    positions = darsia.GridPositions(x, y, (1.0, 1.0))
    own, onto = (-0.009, 0.018, 0.0, 0.012), (-0.012, 0.021, -0.003, 0.015)
    cells = darsia.metric_cells((1, 4), [0.001, 0.004], 3.0, positions, own)
    placed = darsia.placed_cells(cells, 3.0, own, onto)
    col = np.floor((x - -0.009) / 0.003).astype(np.int64) + 1
    row = np.floor((y - 0.0) / 0.003).astype(np.int64) + 1
    assert placed.shape == (6, 11) and placed.pixel_area == cells.pixel_area
    np.testing.assert_allclose(placed.dimensions, [0.018, 0.033])
    np.testing.assert_array_equal(placed.index, ((5 - row) * 11 + col).ravel())
    direct = darsia.metric_cells((1, 4), [0.001, 0.004], 3.0, positions, onto)
    assert not np.array_equal(direct.index, placed.index)
    same = darsia.placed_cells(cells, 3.0, own, own)
    np.testing.assert_array_equal(same.index, cells.index)
    with pytest.raises(ValueError, match="share cell edges"):
        darsia.placed_cells(cells, 3.0, own, (-0.0115, 0.021, -0.003, 0.015))

    # Onto eight columns: the last pixel's cell (the tenth column) is cut.
    cut = darsia.placed_cells(cells, 3.0, own, (-0.012, 0.012, -0.003, 0.015))
    assert col.ravel().tolist() == [4, 6, 6, 9] and cut.shape == (6, 8)
    np.testing.assert_array_equal(cut.index[:3], ((5 - row) * 8 + col).ravel()[:3])
    assert cut.index[3] == -1
    values = np.array([[1.0, 2.0, 4.0, 8.0]])
    sums = darsia.cell_sums(cut, values)
    assert sums.sum() == 7.0 and sums.ravel()[-1] == 0.0
    means = darsia.cell_means(cut, values)
    assert np.isnan(means).sum() == 6 * 8 - 3
    again = darsia.placed_cells(cut, 3.0, (-0.012, 0.012, -0.003, 0.015), onto)
    np.testing.assert_array_equal(again.index[3], -1)


def test_scaling_averages_the_valid_pixels_of_each_cell():
    """Means over valid pixels per channel, NaN without one; the canvas grows up."""
    values = np.random.default_rng(1).random((6, 9, 2)).astype(np.float32)
    values[0, 0, 0] = np.nan
    values[:, 8, 1] = np.nan
    image = darsia.Image(values, dimensions=[0.6, 0.9], origin=[0.1, 0.7])

    scaling = darsia.MetricScalingCorrection(400.0, (6, 9), [0.6, 0.9], [0.1, 0.7])
    scaled = scaling(image)

    masked = np.ma.masked_invalid(values)
    expected = [
        [masked[rows, columns].mean(axis=(0, 1)).filled(np.nan) for columns in COLUMNS]
        for rows in ROWS
    ]
    assert scaled.img.dtype == np.float32
    np.testing.assert_allclose(scaled.img, expected, rtol=1e-6)
    assert np.isnan(scaled.img[:, 2, 1]).all()
    assert scaled.dimensions == [0.8, 1.2]
    np.testing.assert_allclose(scaled.origin, [0.1, 0.9])


def test_majority_takes_the_largest_area_and_the_lowest_label_on_a_tie():
    """Each cell takes the label of its largest area fraction."""
    labels = np.full((6, 9), 7)
    labels[2:6, 0:2] = 3
    labels[0:2, 4:8] = [[5, 5, 2, 2], [5, 5, 2, 2]]
    cells = darsia.metric_cells((6, 9), [0.6, 0.9], 400.0)

    np.testing.assert_allclose(darsia.cell_fractions(cells, labels == 3)[1, 0], 0.5)
    np.testing.assert_array_equal(
        darsia.cell_majority(cells, labels), [[7, 2, 7], [3, 7, 7]]
    )


def test_cell_sums_add_single_precision_values_in_double_precision():
    """Extensive values are summed per cell in float64 and keep the map's total."""
    values = np.random.default_rng(2).random((6, 9)).astype(np.float32) * 1e-3
    cells = darsia.metric_cells((6, 9), [0.6, 0.9], 400.0)

    sums = darsia.cell_sums(cells, values)

    assert sums.dtype == np.float64
    expected = [
        [values[rows, columns].astype(np.float64).sum() for columns in COLUMNS]
        for rows in ROWS
    ]
    np.testing.assert_allclose(sums, expected, rtol=1e-15)
    assert sums.sum() == pytest.approx(values.astype(np.float64).sum(), rel=1e-15)


def test_rectangular_cells_keep_measured_positions_and_mark_outside():
    positions = darsia.GridPositions(
        np.array([[0.5, 3.5], [1.0, 4.0]]),
        np.array([[2.5, 0.5], [0.0, 1.0]]),
        None,
    )
    cells = darsia.rectangular_cells(
        (2, 2), [3.0, 4.0], (3, 2), (0.0, 4.0, 0.0, 3.0), positions
    )
    # Cells are 2m wide and 1m high, lower edges included, upper edges excluded.
    np.testing.assert_array_equal(cells.index, [0, 5, 4, -1])
    summed = darsia.cell_sums(cells, np.array([[1.0, 2.0], [3.0, 99.0]]))
    np.testing.assert_array_equal(summed, [[1.0, 0.0], [0.0, 0.0], [3.0, 2.0]])
