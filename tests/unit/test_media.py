"""Unit tests of films and figures: display means, tiles, layouts and MP4 films."""

import tomllib

import numpy as np
import pytest
from matplotlib import colormaps

from darsia.multiphase.phase_rules import PhaseLabel
from darsia.presets.workflows.media.display_mean import (
    block_mean,
    depth_scaled,
    display_step,
    pooled_mean,
    shown_values,
)
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    graph_tile,
    map_tile,
    photograph,
    shared_axes,
)
from darsia.presets.workflows.media.style import Curve, FigureStyle, bgr

TABLES = tomllib.loads(
    """
    # Our production's figure style: the films' colours and mass curves, and the
    # ensemble figures' palette, resolution and rule for a logarithmic time axis.
    [style]
    dpi = 150
    log_span = 50.0
    surface = "#fcfcfb"
    ink = "#0b0b0b"
    secondary = "#52514e"
    muted = "#898781"
    grid = "#e1e0d9"
    axis = "#c3c2b7"
    washes = ["#ebeae4", "#dcdcd4"]
    series = "#2a78d6"
    frame = "#1e2226"
    empty = "#3e4246"
    no_value = "#e6edf2"
    colour_map = "viridis"
    marker = "#667085"
    curves = [
        {column = "total_mass_g", name = "Total", colour = "#175aad", line = "-"},
        {column = "aqueous_mass_g", name = "Dissolved", colour = "#169879", line = "-"},
        {column = "gas_mass_g", name = "Free gas", colour = "#dc398b", line = "-"},
        {column = "injected_mass_g", name = "Injected", colour = "black", line = "--"},
    ]

    """
)
STYLE = FigureStyle(
    **{
        **TABLES["style"],
        "washes": tuple(TABLES["style"]["washes"]),
        "curves": tuple(Curve(**curve) for curve in TABLES["style"]["curves"]),
    }
)
UNKNOWN = PhaseLabel.UNKNOWN


def rgb(colour: str) -> tuple:
    """A colour as 8-bit (red, green, blue)."""
    return bgr(colour)[::-1]


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


# -- tiles -------------------------------------------------------------------------


def test_a_photograph_follows_the_colour_transfer_in_8_bits():
    """Linear light through the matrix and the sRGB transfer, rounded to 8 bits;
    pixels without a value are grey."""
    settings = PhotographSettings(
        matrix=((0.5, 0, 0), (0, 1, 0), (0, 0, 2)), invalid_grey=128
    )
    linear = np.array([[[0.002, 0.0, 0.2], [2.0, 0.5, np.nan]]])
    shown = photograph(linear, settings)

    srgb = 12.92 * 0.001, 1.055 * 0.4 ** (1 / 2.4) - 0.055
    expected = np.round(255 * np.array([srgb[0], 0.0, srgb[1]])) / 255
    np.testing.assert_allclose(shown[0, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(shown[0, 1], 128 / 255, rtol=1e-6)


def test_a_map_tile_shows_the_colour_scale_and_cells_without_a_value():
    """A value in the scale's colour map, a missing value in the no-value colour."""
    values = np.full((20, 40), 0.85)
    values[:, 20:] = np.nan
    tile = map_tile(
        values,
        (0, 0, 40, 20),
        1,
        (160, 80),
        scale=1.7,
        label="Total CO₂ mass",
        unit="g/L",
        style=STYLE,
    )
    middle = np.round(255 * np.array(colormaps["viridis"](0.5)[:3]))

    assert tile.shape == (80, 160, 3)
    np.testing.assert_allclose(tile[40, 40], middle, atol=1)
    np.testing.assert_allclose(tile[40, 120], rgb(STYLE.no_value), atol=1)


def mass_rows(count=5, hours=1.0):
    """A run's mass records, one per hour."""
    return [
        dict(
            case_id=f"r-{n}",
            elapsed_h=hours * n,
            total_mass_g=0.1 * n,
            aqueous_mass_g=0.08 * n,
            gas_mass_g=0.02 * n,
            injected_mass_g=0.1 * n,
        )
        for n in range(count)
    ]


def test_the_graph_marks_the_frame_on_shared_axes():
    """The frame's line moves with its time; the shared axes cover every run."""
    runs = [mass_rows(5), mass_rows(3, hours=2.0)]
    axes = shared_axes(runs, STYLE)
    lines = []
    for case in ("r-1", "r-3"):
        tile = graph_tile(runs[0], case, (320, 180), axes=axes, style=STYLE)
        near = np.abs(tile.astype(int) - rgb(STYLE.marker)).max(axis=2) < 30
        lines.append(int(np.argmax(near.sum(axis=0))))

    assert axes == ((0.0, 4 * 1.03), (0.0, 0.4 * 1.05))
    assert lines[1] > lines[0] + 50
