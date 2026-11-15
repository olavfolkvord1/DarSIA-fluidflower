"""Unit tests for the gas allocation on synthetic runs."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.multiphase import gas_allocation as allocation_module

W, D, G, U = darsia.PhaseLabel
RULE = darsia.PhaseRule

TABLES = tomllib.loads(
    """
    # The values the phase rules and the gas allocation ran with in production.
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

    [gas_allocation]
    seed_mm2 = 100.0
    persistent_seed_mm2 = 25.0
    seed_window_h = 1.01
    fallback_seed_mm2 = 25.0
    score_window_mm = 5.0
    cone_base_mm = [14.74, 14.74]
    cone_width_mm = [38.1, 41.45]
    cone_rise_mm = 12.81
    seal_band_mm = 30.0
    cone_decay_mm = 10.0
    end_gas_decay_mm = 10.0
    end_gas_angle_deg = 30.0
    port_fade_mm = 10.0
    lattice_mm = 1.0
    cost_up_per_mm = 0.5
    cost_side_per_mm = 1.0
    cost_down_per_mm = 3.0
    """
)
RULES = darsia.PhaseRuleSettings(**TABLES["phase_rules"])
SETTINGS = darsia.GasAllocationSettings(**TABLES["gas_allocation"])
GAS_THRESHOLD = 0.66
"""Gas threshold of our production phase model, from its folder."""

START = datetime.datetime(2000, 1, 1)
SHAPE = (60, 80)
"""Rows and columns of a 1 mm grid; row r lies at height 59.5 - r mm."""
PORE = 1e-6 * 0.02 * 0.4
"""Pore volume of a pixel [m3]: 1 mm2, 20 mm deep, porosity 0.4."""
SWI = np.float32(0.2)


def _frame(co2=(), gas=(), level=0.9):
    """Probabilities: low everywhere, CO2 and gas at ``level`` in the listed boxes."""
    probabilities = np.full((2, *SHAPE), 0.1, np.float32)
    for index, boxes in enumerate((co2, gas)):
        for box in boxes:
            probabilities[index][box] = 0.9 if index == 0 else level
    return tuple(probabilities)


def _run(tmp_path, frames, hours, volumes, *, end_h=9.0, facies=None, positions=None):
    """Phase and rule maps and saturations of a run, one port at height 4.5 mm."""
    csv = tmp_path / "protocol.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "location_x": 0.0405,
                "location_y": 0.0045,
                "start": START - datetime.timedelta(hours=1),
                "end": START + datetime.timedelta(hours=end_h),
                "rate_g/s": 0.01,
            }
        ]
    ).to_csv(csv, index=False)
    return list(
        darsia.free_gas(
            frames,
            [START + datetime.timedelta(hours=h) for h in hours],
            darsia.Image(
                np.ones(SHAPE, int) if facies is None else facies,
                dimensions=[0.06, 0.08],
            ),
            np.ones(SHAPE, bool),
            1.0,
            {1: 1.0, 2: 0.01},
            darsia.InjectionProtocol(csv),
            RULES,
            np.full(SHAPE, PORE),
            np.full(SHAPE, SWI),
            volumes,
            SETTINGS,
            gas_threshold=GAS_THRESHOLD,
            positions=positions,
        )
    )


def test_the_seal_band_and_the_peel_read_the_measured_heights():
    """Under a seal, a pixel's depth below its run's top is taken between the
    measured heights of its own column; in a peel's cell, the pixel lowest by its
    measured height goes first."""
    rows, cols = 12, 4
    y = np.broadcast_to(0.012 - np.arange(rows)[:, None] * 0.001, (rows, cols)).copy()
    y[:, 2] = 0.012 - np.arange(rows) * 0.0015  # one column's pixels lie further apart
    co2 = np.zeros((rows, cols), bool)
    co2[3:, :] = True
    seal = np.zeros((rows, cols), bool)
    seal[2, :] = True
    band = allocation_module._under_seal(co2, seal, y, 0.0046)
    depth = y[3] - y
    np.testing.assert_array_equal(band, co2 & (depth >= 0) & (depth <= 0.0046))
    assert band[:, 2].sum() < band[:, 0].sum()

    model = np.zeros((rows, cols), bool)
    model[4:6, :] = True
    tilted = y + np.arange(cols)[None, :] * -0.0012  # the right lies lower
    room = np.ones((rows, cols))
    order_rows, order_cols, _ = allocation_module._peel_order(
        model, tilted, room, (1.0, 1.0), replace(SETTINGS, lattice_mm=50.0)
    )
    heights = tilted[order_rows, order_cols]
    assert (order_rows[0], order_cols[0]) == np.unravel_index(
        np.where(model, tilted, np.inf).argmin(), (rows, cols)
    )
    assert (np.diff(heights) >= 0).all()


def test_the_gravity_zones_are_single_and_the_cones_zones_the_positions_precision():
    """The gravity rule's port zones are taken in single precision, the cones'
    bodies and lower zones in the positions' own, as our production took each;
    positions in single precision give one set."""
    rng = np.random.default_rng(3)
    port, radius_m = (0.0405, 0.0245), RULES.port_radius_mm / 1000
    angle, spread = rng.uniform(0, 2 * np.pi, 4096), rng.uniform(-3e-7, 3e-7, 4096)
    x = (port[0] + radius_m * (1 + spread) * np.cos(angle)).reshape(64, 64)
    y = (port[1] + radius_m * (1 + spread) * np.sin(angle)).reshape(64, 64)
    positions = darsia.GridPositions(x, y, (1.0, 1.0))
    (gravity, _), (body, _) = allocation_module._zones(positions, [port], RULES)
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    single = (x32 - port[0]) ** 2 + (y32 - port[1]) ** 2 <= radius_m**2
    double = (x - port[0]) ** 2 + (y - port[1]) ** 2 <= radius_m**2
    np.testing.assert_array_equal(gravity[port], single)
    np.testing.assert_array_equal(body[port], double)
    assert not np.array_equal(single, double)
    narrow = darsia.GridPositions(x32, y32, (1.0, 1.0))
    (gravity, lower), (body, body_lower) = allocation_module._zones(
        narrow, [port], RULES
    )
    assert body is gravity and body_lower is lower


def test_the_cones_follow_each_rows_ray_to_the_first_seal():
    """A cone's first seal is the lowest seal above the port on the ray of each
    row's pixel nearest to the port across: with positions sheared across, a seal
    met only by that ray caps the cone."""
    rows, cols = 40, 40
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    x = ((c + 0.5) * 0.001 + (rows - 1 - r) * 0.00025).astype(np.float32)
    y = (0.04 - (r + 0.5) * 0.001).astype(np.float32)
    port = (0.0205, 0.0045)
    column = np.abs(x - np.float32(port[0])).argmin(axis=1)
    seal = np.zeros((rows, cols), bool)
    seal[10, column[10]] = True  # on the ray of row 10 only
    assert column[10] != column[35]
    co2 = np.zeros((rows, cols), bool)
    co2[2:39, 5:35] = True
    radius, lower = allocation_module._port_zones(x, y, [port], RULES)
    cones = allocation_module._cones(
        co2, [port], [port], radius, lower, seal, x, y, SETTINGS
    )
    top = float(y[10, column[10]])
    assert cones.any() and not cones[y >= top].any()


def test_the_allocation_reads_the_measured_positions(tmp_path):
    """At the pixel centres given as positions the allocation is the default's;
    with measured positions it places the gas by them."""
    frame = _frame([np.s_[10:60, 20:60]], [np.s_[30:40, 30:50]])
    volumes = [3 * 10 * 20 * PORE * (1 - float(SWI))]
    centres = darsia.grid_positions(
        darsia.Image(np.zeros(SHAPE), dimensions=[0.06, 0.08])
    )
    plain = _run(tmp_path, [frame], [0.0], volumes)
    same = _run(tmp_path, [frame], [0.0], volumes, positions=centres)
    for a, b in zip(plain[0], same[0]):
        np.testing.assert_array_equal(a, b)
    sheared = darsia.GridPositions(
        (centres.x + 0.004 * np.linspace(0, 1, SHAPE[0])[:, None]).astype(np.float32),
        centres.y.astype(np.float32),
        centres.pitch_mm,
    )
    moved = _run(tmp_path, [frame], [0.0], volumes, positions=sheared)
    assert not np.array_equal(moved[0][2], plain[0][2])


def test_added_gas_joins_the_model_gas_by_its_score(tmp_path):
    """The model's gas stays; the rest fills its CO2 body above the seal, exactly."""
    facies = np.ones(SHAPE, int)
    facies[40:42] = 2  # a seal across the CO2
    probabilities = _frame(co2=[np.s_[20:52, 10:70]], gas=[np.s_[22:34, 34:46]])
    probabilities[1][20:40, 10:70] += np.linspace(0, 0.3, 60, dtype=np.float32)
    model = 144 * PORE * (1 - SWI)
    labels, _, saturation, _, _ = _run(
        tmp_path, [probabilities], [0.0], [model + 6e-7], facies=facies
    )[0]
    assert np.isclose(np.sum(PORE * saturation.astype(float)), model + 6e-7, rtol=1e-6)
    assert (saturation[22:34, 34:46] == 1 - SWI).all() and (
        labels[22:34, 34:46] == G
    ).all()
    added = (saturation > 0) & (labels == D)
    assert added.sum() > 100 and not added[40:].any()
    assert saturation[30, 66] >= saturation[30, 14]


def test_removed_gas_peels_the_lower_edges_first(tmp_path):
    """Whole pixels go from the bottom and sides, one partial pixel, freed as dissolved."""
    probabilities = _frame(co2=[np.s_[10:50, 20:60]], gas=[np.s_[20:40, 30:50]])
    model = 400 * PORE * (1 - SWI)
    kept, gone = _run(tmp_path, [probabilities] * 2, [0.0, 3.0], [0.4 * model, 0.0])
    kept, kept_rules, kept_saturation, kept_co2, kept_before = kept
    gone, gone_rules, gone_saturation, gone_co2, gone_before = gone
    assert np.isclose(np.sum(PORE * kept_saturation.astype(float)), 0.4 * model)
    partial = (kept_saturation > 0) & (kept_saturation < 1 - SWI)
    assert partial.sum() <= 1
    assert ((kept == G) == (kept_saturation > 0)).all()
    assert not kept_saturation[33:40].any() and (kept_saturation[20, 35:45] > 0).all()
    assert not (gone == G).any() and not gone_saturation.any()
    # The rule map records the peel, and the model's gas is the gas before it.
    gas = np.zeros(SHAPE, bool)
    gas[20:40, 30:50] = True
    assert ((kept_rules == RULE.PEELED) == (gas & (kept == D))).all()
    assert (darsia.model_gas(kept, kept_rules) == gas).all()
    assert (darsia.model_gas(gone, gone_rules) == gas).all()
    # The CO2 before the rules, which the dilution law reads, stays the model's,
    # and so do the labels before the rules, on which the colour is read.
    co2 = np.zeros(SHAPE, bool)
    co2[10:50, 20:60] = True
    assert (kept_co2 == co2).all() and (gone_co2 == co2).all()
    model = np.where(gas, G, np.where(co2, D, W))
    np.testing.assert_array_equal(kept_before, model)
    np.testing.assert_array_equal(gone_before, model)


def test_end_gas_weights_the_gas_after_the_injection(tmp_path):
    """After the injection, gas far from the free gas at its end is dissolved."""
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    before = _frame(co2=[left], gas=[left])
    after = _frame(co2=[left, right], gas=[left, right], level=0.7)
    (_, _, saturation, _, _), (labels, _, _, _, _) = _run(
        tmp_path, [before, after], [0.5, 3.0], [None, None], end_h=1.0
    )
    assert (saturation[left] == 1 - SWI).all()
    assert (labels[left] == G).all() and (labels[right] == D).all()


def test_a_run_resumed_after_an_injection_end_equals_the_whole_run(tmp_path):
    """Resumed at its first missing frame, after an injection-end anchor, with the
    anchor's saved saturation: every later map and saturation is the whole run's."""
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    frames = [
        _frame(co2=[left], gas=[left]),
        _frame(co2=[left], gas=[np.s_[22:38, 12:28]]),
        _frame(co2=[left, right], gas=[left, right], level=0.7),
        _frame(co2=[left, right], gas=[left, np.s_[24:36, 56:68]], level=0.7),
        _frame(co2=[left, right], gas=[left, right], level=0.75),
    ]
    hours, volumes = [0.0, 0.5, 1.5, 2.0, 3.0], [None, 3e-7, 2e-7, None, 4e-7]
    whole = _run(tmp_path, frames, hours, volumes, end_h=1.0)
    csv = tmp_path / "protocol.csv"
    args = (
        frames,
        [START + datetime.timedelta(hours=h) for h in hours],
        darsia.Image(np.ones(SHAPE, int), dimensions=[0.06, 0.08]),
        np.ones(SHAPE, bool),
        1.0,
        {1: 1.0, 2: 0.01},
        darsia.InjectionProtocol(csv),
        RULES,
        np.full(SHAPE, PORE),
        np.full(SHAPE, SWI),
        volumes,
        SETTINGS,
    )
    saved = {index: maps[2] for index, maps in enumerate(whole)}
    resumed = list(
        darsia.free_gas(
            *args, gas_threshold=GAS_THRESHOLD, start=3, anchor_saturation=saved.get
        )
    )
    assert len(resumed) == 2
    for made, again in zip(whole[3:], resumed):
        for array, other in zip(made, again):
            np.testing.assert_array_equal(array, other)
    assert (whole[3][0][right] == D).any()  # the anchor's end gas weighs here
    with pytest.raises(ValueError, match="anchors' saturation"):
        next(darsia.free_gas(*args, gas_threshold=GAS_THRESHOLD, start=3))
