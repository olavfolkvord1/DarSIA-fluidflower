"""Unit tests of films and figures: display means, tiles, layouts and MP4 films."""

import numpy as np
import pytest

from darsia.multiphase.phase_rules import PhaseLabel
from darsia.presets.workflows.media.display_mean import (
    block_mean,
    depth_scaled,
    display_step,
    pooled_mean,
    shown_values,
)

UNKNOWN = PhaseLabel.UNKNOWN


def layers_frame():
    """Two touching sand layers (10 left, 20 right) of grainy CO2, gas below row 6,
    an unknown phase and a missing value; with pore volumes rising to the right."""
    shape = (10, 16)
    rows, columns = np.indices(shape)
    layers = np.where(columns < 8, 10, 20).astype(np.uint8)
    saturation = np.where(rows < 6, 0.0, 0.5)
    phase = np.where(saturation > 0, PhaseLabel.GAS, PhaseLabel.DISSOLVED)
    phase = phase.astype(np.uint8)
    phase[0, 0] = UNKNOWN
    values = np.where(layers == 10, 5.0, 100.0) + np.where((rows + columns) % 2, 2, -2)
    values[saturation > 0] += 300.0
    values[0, 1] = np.nan
    volume = 1.0 + columns / 10.0
    return values, phase, saturation, layers, volume


# -- display means ---------------------------------------------------------------


def test_each_shown_cell_holds_the_mean_of_its_known_values():
    """Edge blocks hold fewer cells; unknown phases and missing values do not count;
    the means keep the map's precision."""
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    values[0, 0] = np.nan
    phase = np.zeros((5, 7), np.uint8)
    phase[0, 1] = UNKNOWN
    values[3:, 6] = np.nan
    means = block_mean(shown_values(values, phase), 3)

    assert display_step((5, 7), 3) == 3
    assert means.shape == (2, 3) and means.dtype == np.float32
    assert means[0, 0] == np.float32(np.mean([2, 7, 8, 9, 14, 15, 16]))
    assert means[0, 2] == np.float32(np.mean([6, 13, 20]))
    assert means[1, 1] == np.float32(np.mean([24, 25, 26, 31, 32, 33]))
    assert np.isnan(means[1, 2])


def test_the_depth_mean_stays_inside_each_layer_and_its_water():
    """Two touching layers stay apart, gas and unknown pixels keep their values, and
    each layer keeps its pore-volume-weighted mass."""
    values, phase, saturation, layers, volume = layers_frame()
    before = values.copy()
    depth, area = np.full(values.shape, 0.006), np.full(values.shape, 1e-6)
    shown = shown_values(values, phase)
    mean, window, depth_mm = depth_scaled(
        shown, phase, saturation, layers, [10, 20], volume, depth, area
    )

    assert window == 7 and depth_mm == pytest.approx(6.0)
    assert np.isnan(mean[0, 0]) and np.isnan(mean[0, 1])
    np.testing.assert_array_equal(mean[saturation > 0], values[saturation > 0])
    for label in (10, 20):
        water = (layers == label) & (saturation <= 0) & np.isfinite(shown)
        kept = np.sum(mean[water] * volume[water])
        assert kept == pytest.approx(np.sum(values[water] * volume[water]), rel=1e-13)
        assert np.var(mean[water]) < np.var(values[water])
        if label == 10:
            assert np.max(mean[water]) < 10
        else:
            assert np.min(mean[water]) > 90
    np.testing.assert_array_equal(values, before)


def test_the_depth_mean_keeps_each_part_of_a_layer_apart():
    """Two parts of one sand, three pixels apart across a thin layer of another
    sand: within the disk's reach of each other, yet each keeps its own mean and
    its own pore-volume-weighted total."""
    rows, columns = np.indices((5, 12))
    layers = np.where((columns >= 5) & (columns <= 6), 7, 5)
    values = np.where(columns < 5, 10.0, np.where(columns > 6, 100.0, 50.0))
    values += np.where((rows + columns) % 2, 2.0, -2.0)
    volume = 1.0 + columns / 10.0
    phase = np.full(layers.shape, PhaseLabel.WATER, np.uint8)
    mean, window, _depth_mm = depth_scaled(
        values, phase, np.zeros(layers.shape), layers, [5, 7], volume, 0.007, 1e-6
    )

    assert window == 7
    for part, low, high in ((columns < 5, 6, 14), (columns > 6, 96, 104)):
        kept = np.sum(mean[part] * volume[part])
        assert kept == pytest.approx(np.sum(values[part] * volume[part]), rel=1e-13)
        assert low < mean[part].min() and mean[part].max() < high


def test_the_depth_mean_weighs_pore_volume_in_the_sand_alone():
    """Three water pixels of one layer in a row, beside a label that is not sand: the
    window is the median depth of the sand alone, the sand's mean is weighted by
    pore volume, and the other label keeps its values."""
    layers = np.array([[5, 5, 5, 9, 9, 9, 9]])
    values = np.array([[0.0, 0.0, 1.0, 7.0, 7.5, 8.0, 8.5]])
    volume = np.array([[1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]])
    depth = np.array([[0.003, 0.003, 0.006, 0.009, 0.009, 0.009, 0.009]])
    phase = np.full(layers.shape, PhaseLabel.WATER, np.uint8)
    mean, window, _depth_mm = depth_scaled(
        values, phase, np.zeros(layers.shape), layers, [5], volume, depth, 1e-6
    )
    # The cross of a 3-pixel window: means 0, 4/6 and 4/5, scaled to keep the mass 4.
    weighted = np.array([0.0, 4 / 6, 4 / 5]) * 4 / (4 / 6 + 4 / 5 * 4)

    assert window == 3
    np.testing.assert_allclose(mean[0, :3], weighted, rtol=1e-12)
    np.testing.assert_array_equal(mean[0, 3:], values[0, 3:])


def test_shown_cells_of_the_depth_mean_are_weighted_by_pore_volume():
    """Each shown cell keeps the pore-volume-weighted mass of its known pixels."""
    values, phase, _saturation, _layers, volume = layers_frame()
    shown = shown_values(values, phase)
    pooled = pooled_mean(shown, volume, 3)
    known = np.isfinite(shown)
    weights = np.add.reduceat(
        np.add.reduceat(np.where(known, volume, 0), np.arange(0, 10, 3), axis=0),
        np.arange(0, 16, 3),
        axis=1,
    )

    assert pooled.shape == (4, 6)
    assert np.sum(pooled * weights) == pytest.approx(
        np.sum(shown[known] * volume[known]), rel=1e-13
    )
    assert pooled_mean(shown, volume, 1) is shown
