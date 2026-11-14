"""Unit tests for the scaling onto whole metric cells."""

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
