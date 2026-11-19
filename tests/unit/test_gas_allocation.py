"""Unit tests for the gas allocation on synthetic runs."""

import datetime
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.multiphase.gas_parameters import parameters_at_depth
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

    [gas_allocation]
    seed_mm2 = 100.0
    persistent_seed_mm2 = 25.0
    seed_window_h = 1.01
    fallback_seed_mm2 = 25.0
    lattice_mm = 1.0
    cost_up_per_mm = 0.5
    cost_side_per_mm = 1.0
    cost_down_per_mm = 3.0
    """
)
RULES = darsia.PhaseRuleSettings(**TABLES["phase_rules"])

BASE_SETTINGS = darsia.GasAllocationSettings(**TABLES["gas_allocation"])
SETTINGS = parameters_at_depth(BASE_SETTINGS, 5.0)
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


def _run(
    tmp_path,
    frames,
    hours,
    volumes,
    *,
    end_h=9.0,
    facies=None,

    record=None,
):
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
            BASE_SETTINGS,
            gas_threshold=GAS_THRESHOLD,
            depth_m=np.full(SHAPE, 0.02),

            record=record,
        )
    )






def test_added_gas_joins_the_model_gas_by_its_score(tmp_path):
    """Model gas stays; a request larger than the footprint's room is capped."""
    facies = np.ones(SHAPE, int)
    facies[40:42] = 2  # a seal across the CO2
    # The fixed smoothed contour joins the bodies. The seed-connected gas cannot
    # cross the seal to isolated CO2 below it.
    bodies = [np.s_[22:34, 34:39], np.s_[22:34, 41:46]]
    probabilities = _frame(co2=[np.s_[20:52, 10:70]], gas=bodies)
    probabilities[1][20:40, 10:70] += np.linspace(0, 0.3, 60, dtype=np.float32)
    records = []
    labels, _, saturation, _, _ = _run(
        tmp_path,
        [probabilities],
        [0.0],
        [1.0],
        facies=facies,
        record=lambda index, value: records.append(value),
    )[0]
    placed = np.sum(PORE * saturation.astype(float))
    assert placed < 1.0
    assert records[0].unplaced_m3 > 0
    assert np.isclose(placed + records[0].unplaced_m3, 1.0)
    for body in bodies:
        assert (saturation[body] == 1 - SWI).all() and (labels[body] == G).all()
    added = (saturation > 0) & (labels == D)
    assert added[25:31, 39:41].any() and not added[40:].any()
    assert not added[:20].any()
    assert records[0].disconnected_added_area_mm2 == 0


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


@pytest.mark.parametrize("end_h", [1.0, 3.0, 4.0])
def test_fixed_contour_keeps_later_model_gas_after_injection(tmp_path, end_h):
    """All selected post-rule gas contributes, irrespective of injection end."""
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    before = _frame(co2=[left], gas=[left])
    after = _frame(co2=[left, right], gas=[left, right], level=0.99)
    (_, _, saturation, _, _), (labels, rules, _, _, _) = _run(
        tmp_path, [before, after], [0.5, 3.0], [None, None], end_h=end_h
    )
    assert (saturation[left] == 1 - SWI).all()
    assert (labels[left] == G).all() and (labels[right] == G).all()
    assert not (rules == darsia.PhaseRule.GAS_LIMIT).any()


@pytest.mark.parametrize("start", [1, 3])
def test_a_resumed_run_equals_the_whole_run(tmp_path, start):
    """The full prepass reconstructs the fixed contour before resuming.

    Every later map and saturation equals the whole run's, including peeled gas.
    """
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
        BASE_SETTINGS,
    )
    resumed = list(
        darsia.free_gas(
            *args,
            gas_threshold=GAS_THRESHOLD,
            start=start,
            depth_m=np.full(SHAPE, 0.02),
        )
    )
    assert len(resumed) == len(frames) - start
    for made, again in zip(whole[start:], resumed):
        for array, other in zip(made, again):
            np.testing.assert_array_equal(array, other)
    assert (whole[3][0][24:36, 56:68] == G).all()


