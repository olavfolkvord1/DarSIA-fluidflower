"""Unit tests for the phase labels and rules on synthetic frames."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.multiphase import phase_rules as rules_module

W, D, G, U = darsia.PhaseLabel
RULE = darsia.PhaseRule

SETTINGS = darsia.PhaseRuleSettings(
    **tomllib.loads(
        """
        # The values the rules ran with in production.
        [phase_rules]
        co2_threshold = 0.8
        smallest_gas_patch_mm2 = 25.0
        neighbour_window_h = 1.0
        neighbours_per_side = 2
        small_bubble_max_mm2 = 25.0
        small_bubble_min_height_mm = 30.0
        chart_margin_mm = 2.0
        chart_extend_below_mm = 35.0
        free_water_facies = [5]
        min_component_mm2 = 25.0
        max_hole_mm2 = 25.0
        min_gas_component_mm2 = 2.5
        seal_permeability_fraction = 0.05
        port_radius_mm = 15.0
        port_zone_half_width_mm = 38.0
        port_zone_depth_mm = 10.0
        finger_width_mm = 10.0
        pitch_stride = 64
        """
    )["phase_rules"]
)
GAS_THRESHOLD = 0.66
"""Gas threshold of our production phase model, from its folder."""

START = datetime.datetime(2000, 1, 1)
SHAPE = (60, 80)
"""Rows and columns of a 1 mm grid; row r lies at height 59.5 - r mm."""


def _frame(co2=(), gas=()):
    """Probabilities: low everywhere, 0.9 in the listed (rows, cols) boxes."""
    probabilities = np.full((2, *SHAPE), 0.1, np.float32)
    for index, boxes in enumerate((co2, gas)):
        for box in boxes:
            probabilities[index][box] = 0.9
    return tuple(probabilities)


def _maps(tmp_path, frames, hours=None, *, port=(0.0005, 0.0005), **kwargs):
    """Phase and rule maps of a run on the 1 mm grid; one port injects throughout."""
    csv = tmp_path / "protocol.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "location_x": port[0],
                "location_y": port[1],
                "start": START - datetime.timedelta(hours=1),
                "end": START + datetime.timedelta(hours=9),
                "rate_g/s": 0.01,
            }
        ]
    ).to_csv(csv, index=False)
    facies = kwargs.pop("facies", np.ones(SHAPE, int))
    hours = hours or [0.0] * len(frames)
    return tuple(
        zip(
            *darsia.phase_maps(
                frames,
                [START + datetime.timedelta(hours=h) for h in hours],
                darsia.Image(facies, dimensions=[0.06, 0.08]),
                kwargs.pop("domain", np.ones(SHAPE, bool)),
                kwargs.pop("pixel_area_mm2", 1.0),
                {1: 1.0, 2: 0.01},
                darsia.InjectionProtocol(csv),
                kwargs.pop("settings", SETTINGS),
                gas_threshold=GAS_THRESHOLD,
                **kwargs,
            )
        )
    )


def test_phase_labels_threshold_the_probabilities():
    """CO2 from both thresholds, water without probabilities, small gas dissolved."""
    co2, gas = _frame(
        co2=[np.s_[:, 10:70]], gas=[np.s_[0:5, 10:15], np.s_[20:24, 10:16]]
    )
    co2[30:40, 10:20] = np.nan
    co2[30:40, 20:30] = 0.79
    domain = np.ones(SHAPE, bool)
    domain[:, 75:] = False
    labels = darsia.phase_labels(
        co2, gas, domain, 1.0, SETTINGS, gas_threshold=GAS_THRESHOLD
    )
    assert (labels[:, 75:] == U).all() and (labels[:, :10] == W).all()
    assert (labels[30:40, 10:30] == W).all()
    assert (labels[0:5, 10:15] == G).all() and (labels[20:24, 10:16] == D).all()
    assert (labels[50:, 10:70] == D).all()


def test_time_average_decides_free_gas_with_the_neighbours(tmp_path):
    """Gas of a frame alone is lost, gas of its neighbours gained, in time weights."""
    co2 = [np.s_[10:60, :]]
    lone, shared = np.s_[10:20, 10:30], np.s_[30:40, 10:30]
    frames = [_frame(co2, [shared]), _frame(co2, [lone]), _frame(co2, [shared])]
    frames[1][1][shared] = 0.6
    maps, rules = _maps(tmp_path, frames, [0.0, 0.5, 1.0])
    assert (maps[1][lone] == D).all() and (maps[1][shared] == G).all()
    assert (rules[1][lone] == RULE.TIME_AVERAGE_LOST).all()
    assert (rules[1][shared] == RULE.TIME_AVERAGE_GAINED).all()
    maps, _ = _maps(tmp_path, frames, [0.0, 1.0, 2.0])
    assert (maps[1][lone] == G).all() and (maps[1][shared] == D).all()


def test_small_bubbles_of_the_frame_stay_high_above_the_port(tmp_path):
    """Small gas pieces of the frame are free gas only well above the port."""
    high, low = np.s_[5:9, 40:44], np.s_[40:44, 40:44]
    labels, _ = _maps(
        tmp_path, [_frame([np.s_[:, :]], [high, low, np.s_[20:30, 0:10]])]
    )
    assert (labels[0][high] == G).all() and (labels[0][low] == D).all()
    assert (labels[0][20:30, 0:10] == G).all()


def test_chart_region_is_unknown(tmp_path):
    """The chart, widened by its margin and extended downwards, is unknown."""
    chart = np.zeros(SHAPE, bool)
    chart[5:10, 30:40] = True
    (labels,), (rules,) = _maps(tmp_path, [_frame([np.s_[:, :]])], chart=chart)
    region = np.zeros(SHAPE, bool)
    region[3:45, 28:42] = True
    assert np.array_equal(labels == U, region)
    assert np.array_equal(rules == RULE.COLOR_CHECKER, region)


def test_free_water_co2_joined_to_no_sand_co2_is_water(tmp_path):
    """CO2 in free water stays only where it joins CO2 in the sand."""
    facies = np.ones(SHAPE, int)
    facies[:20] = 5
    co2 = [np.s_[5:15, 5:15], np.s_[10:40, 40:50]]
    (labels,), (rules,) = _maps(tmp_path, [_frame(co2)], facies=facies)
    assert (labels[5:15, 5:15] == W).all() and (labels[10:40, 40:50] == D).all()
    assert (rules[5:15, 5:15] == RULE.FREE_WATER).all()


def test_minimum_area_removes_specks_and_fills_small_holes(tmp_path):
    """Small components are water, small holes dissolved, small gas specks dissolved."""
    co2 = [np.s_[20:59, 20:59], np.s_[5:9, 5:10], np.s_[0:30, 0:1]]
    co2_prob, gas_prob = _frame(co2, [np.s_[40:42, 30:31], np.s_[50:58, 50:58]])
    co2_prob[30:33, 30:33] = co2_prob[45:52, 40:47] = 0.1
    settings = replace(SETTINGS, smallest_gas_patch_mm2=0.0, small_bubble_max_mm2=0.0)
    (labels,), (rules,) = _maps(tmp_path, [(co2_prob, gas_prob)], settings=settings)
    assert (labels[5:9, 5:10] == W).all() and (labels[0:30, 0] == W).all()
    assert (labels[30:33, 30:33] == D).all() and (labels[45:52, 40:47] == W).all()
    assert (labels[40:42, 30] == D).all() and (labels[50:58, 50:58] == G).all()
    assert (rules[5:9, 5:10] == RULE.MINIMUM_AREA).all()
    assert (rules[30:33, 30:33] == RULE.HOLE_FILLED).all()
    assert (rules[40:42, 30] == RULE.GAS_SPECK).all()


def test_compartments_without_port_or_gas_lose_dissolved_co2(tmp_path):
    """CO2 beyond a seal stays only where a CO2 path joins it to the port."""
    facies = np.ones(SHAPE, int)
    facies[25:30] = 2
    above, seal = np.s_[5:15, 40:70], np.s_[25:30, 0:10]
    co2 = [above, seal, np.s_[30:60, 0:10]]
    (labels,), (rules,) = _maps(
        tmp_path, [_frame(co2, [np.s_[50:60, 0:10]])], facies=facies
    )
    assert (labels[above] == W).all() and (labels[seal] == D).all()
    assert (rules[above] == RULE.COMPARTMENTS).all()
    co2.append(np.s_[0:30, 5:10])
    (labels,), _ = _maps(tmp_path, [_frame(co2, [np.s_[50:60, 0:10]])], facies=facies)
    assert (labels[above] == D).all()


def test_gravity_dissolves_gas_below_the_port_and_in_sinking_lobes(tmp_path):
    """Gas below the port outside its lower zone, and in narrow lobes, is dissolved."""
    zone, below, lobe = np.s_[32:38, 25:35], np.s_[50:60, 70:80], np.s_[10:25, 62:65]
    body = np.s_[0:10, 50:80]
    co2 = [np.s_[:, 0:45], body, lobe, np.s_[50:60, 50:80]]
    frame = _frame(co2, [zone, below, body, lobe])
    (labels,), (rules,) = _maps(tmp_path, [frame], port=(0.03, 0.0295))
    assert (labels[zone] == G).all() and (labels[below] == D).all()
    assert (labels[body] == G).all() and (labels[11:25, 62:65] == D).all()
    assert (rules[below] == RULE.GRAVITY).all()
    assert (rules[11:25, 62:65] == RULE.GRAVITY).all()
    gas = darsia.model_gas(labels, rules)
    assert gas[below].all() and gas[zone].all() and not gas[11:25, 50:62].any()


def _tilted(shape=SHAPE, dimensions=(0.06, 0.08)):
    """Measured positions in single precision, tilted and stretched against the
    pixel centres: no row and no column has one height or one x."""
    grid = darsia.Image(np.zeros(shape), dimensions=list(dimensions))
    rows, cols = shape
    dy, dx = dimensions[0] / rows, dimensions[1] / cols
    model = np.array(
        [
            [1.002 * dx, 0.03 * dy, 0.4 * dx],
            [0.02 * dx, -0.997 * dy, dimensions[0] - 2.0 * dy],
            [0, 0, 1.0],
        ]
    )
    return darsia.grid_positions(grid, model, np.float32, None, SETTINGS.pitch_stride)


def test_the_port_zones_and_compartments_take_the_measured_positions():
    """A port's radius disc and its lower half-ellipse, clipped at the floor, are
    taken at each pixel's measured position in its precision, in metres as our
    production's; a port's compartment is that of the nearest compartment pixel
    by the squared distance in metres in double precision."""
    positions = _tilted()
    x, y = positions.x, positions.y
    port = (0.0405, 0.0035)
    radius, lower = rules_module._port_zones(x, y, [port], SETTINGS)
    r = SETTINGS.port_radius_mm / 1000
    np.testing.assert_array_equal(
        radius[port], (x - port[0]) ** 2 + (y - port[1]) ** 2 <= r**2
    )
    lateral = (x - port[0]) * (1000 / SETTINGS.port_zone_half_width_mm)
    depth = (port[1] - y) * (1000 / SETTINGS.port_zone_depth_mm)
    expected = (y <= port[1]) & (y >= 0) & (lateral**2 + depth**2 <= 1)
    np.testing.assert_array_equal(lower[port], expected)
    assert (y < 0).any() and not lower[port][y < 0].any()  # the floor clip

    compartments = np.zeros(SHAPE, np.int32)
    compartments[:, :40], compartments[:, 44:] = 1, 2
    home = rules_module._home_compartments(compartments, x, y, [port])
    d2 = (x.astype(np.float64) - port[0]) ** 2 + (y.astype(np.float64) - port[1]) ** 2
    nearest = np.where(compartments > 0, d2, np.inf).argmin()
    assert home[port] == compartments.flat[nearest]

    # Pixels on the disc's rim in single precision: the disc in metres, not in
    # millimetres; two pixels nearly as near: the nearer in double precision.
    rng = np.random.default_rng(3)
    radius_m = SETTINGS.port_radius_mm / 1000
    angle, spread = rng.uniform(0, 2 * np.pi, 4096), rng.uniform(-3e-7, 3e-7, 4096)
    rim_x = (port[0] + radius_m * (1 + spread) * np.cos(angle)).astype(np.float32)
    rim_y = (port[1] + radius_m * (1 + spread) * np.sin(angle)).astype(np.float32)
    rim, _ = rules_module._port_zones(rim_x, rim_y, [port], SETTINGS)
    metres = (rim_x - port[0]) ** 2 + (rim_y - port[1]) ** 2 <= r**2
    millimetres = ((rim_x - port[0]) * 1000) ** 2 + (
        (rim_y - port[1]) * 1000
    ) ** 2 <= SETTINGS.port_radius_mm**2
    np.testing.assert_array_equal(rim[port], metres)
    assert not np.array_equal(metres, millimetres)
    d64 = (rim_x.astype(np.float64) - port[0]) ** 2 + (
        rim_y.astype(np.float64) - port[1]
    ) ** 2
    d32 = (rim_x - np.float32(port[0])) ** 2 + (rim_y - np.float32(port[1])) ** 2
    i, j = next(
        (i, j)
        for i in range(len(d64))
        for j in range(i + 1, len(d64))
        if d64[i] < d64[j] and d32[i] > d32[j]
    )
    pair = np.array([[1, 2]], np.int32)
    two = rules_module._home_compartments(
        pair, rim_x[[[i, j]]], rim_y[[[i, j]]], [port]
    )
    assert two[port] == 1


def test_gravity_and_small_bubbles_read_the_measured_heights(tmp_path):
    """Gas lies below a port by its measured height in single precision, and a
    small bubble is high enough by the lowest measured height of its pixels."""
    positions = _tilted()
    y = positions.y
    exempt = {(0.04, 0.03): np.zeros(SHAPE, bool)}
    co2 = np.zeros(SHAPE, bool)
    below, _ = rules_module._gravity_limit(
        co2, list(exempt), exempt, exempt, y, positions.pitch_mm, SETTINGS
    )
    np.testing.assert_array_equal(below, y.astype(np.float32) < np.float32(0.03))
    assert below.any() and not below.all()
    # A height just below the port in double precision is level with it in single.
    heights = np.array(y, np.float64)
    heights[5, 5] = 0.02999999966
    below, _ = rules_module._gravity_limit(
        co2, list(exempt), exempt, exempt, heights, positions.pitch_mm, SETTINGS
    )
    assert heights[5, 5] < 0.03 and not below[5, 5]

    labels = np.full(SHAPE, D, np.uint8)
    raw = np.zeros(SHAPE, bool)
    raw[10:13, 20:27] = True  # a piece of 21 mm2
    lowest = float(np.min(y.astype(np.float64)[10:13, 20:27]))
    for port, kept in ((lowest - 0.0301, True), (lowest - 0.0299, False)):
        found = labels.copy()
        rules_module._small_bubbles(found, raw, port, y, 1.0, SETTINGS)
        assert (found[raw] == G).all() == kept


def test_the_maps_read_the_measured_positions(tmp_path):
    """With measured positions the phase maps follow them: gas that the pixel
    centres put below the port lies above it by the measured heights."""
    below = np.s_[50:60, 70:80]
    frame = _frame([np.s_[:, :]], [below])
    centres = darsia.grid_positions(
        darsia.Image(np.zeros(SHAPE), dimensions=[0.06, 0.08])
    )
    lifted = darsia.GridPositions(centres.x, centres.y + 0.02, centres.pitch_mm)
    (plain,), _ = _maps(tmp_path, [frame], port=(0.075, 0.025))
    (moved,), _ = _maps(tmp_path, [frame], port=(0.075, 0.025), positions=lifted)
    (same,), _ = _maps(tmp_path, [frame], port=(0.075, 0.025), positions=centres)
    assert (plain[below] == D).all() and (moved[below] == G).all()
    np.testing.assert_array_equal(same, plain)
    # Measured positions without the grid's pitch are refused.
    unpitched = darsia.GridPositions(centres.x, centres.y + 0.02, None)
    with pytest.raises(ValueError, match="pitch"):
        _maps(tmp_path, [frame], port=(0.075, 0.025), positions=unpitched)


def test_areas_sum_the_pixel_areas(tmp_path):
    """CO2 and gas pieces are kept by their pixels' summed areas, not by counts.

    The pixel areas are 1.1 mm2 left and 0.9 mm2 right; with their mean, 1 mm2,
    each piece's outcome flips.
    """
    area = np.where(np.arange(SHAPE[1]) < 40, 1.1, 0.9) * np.ones(SHAPE)
    pieces = [np.s_[5:9, 5:11], np.s_[5:8, 50:59]]
    gas = [np.s_[40:44, 5:11], np.s_[40:43, 50:59]]
    frame = _frame([*pieces, np.s_[30:60, :]], gas)
    for pixel_area, expected in ((area, [D, W, G, D]), (1.0, [W, D, D, G])):
        (labels,), _ = _maps(tmp_path, [frame], pixel_area_mm2=pixel_area)
        assert [np.unique(labels[box]).tolist() for box in pieces + gas] == [
            [label] for label in expected
        ]
