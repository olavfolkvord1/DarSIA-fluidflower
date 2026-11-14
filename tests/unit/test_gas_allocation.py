"""Unit tests for the gas allocation on synthetic runs."""

import datetime
import tomllib

import numpy as np
import pandas as pd

import darsia

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


def _run(tmp_path, frames, hours, volumes, *, end_h=9.0, facies=None):
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
        )
    )


def test_added_gas_joins_the_model_gas_by_its_score(tmp_path):
    """The model's gas stays; the rest fills its CO2 body above the seal, exactly."""
    facies = np.ones(SHAPE, int)
    facies[40:42] = 2  # a seal across the CO2
    probabilities = _frame(co2=[np.s_[20:52, 10:70]], gas=[np.s_[22:34, 34:46]])
    probabilities[1][20:40, 10:70] += np.linspace(0, 0.3, 60, dtype=np.float32)
    model = 144 * PORE * (1 - SWI)
    labels, _, saturation, _ = _run(
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
    kept, kept_rules, kept_saturation, kept_co2 = kept
    gone, gone_rules, gone_saturation, gone_co2 = gone
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
    # The CO2 before the rules, which the dilution law reads, stays the model's.
    co2 = np.zeros(SHAPE, bool)
    co2[10:50, 20:60] = True
    assert (kept_co2 == co2).all() and (gone_co2 == co2).all()


def test_end_gas_weights_the_gas_after_the_injection(tmp_path):
    """After the injection, gas far from the free gas at its end is dissolved."""
    left, right = np.s_[20:40, 10:30], np.s_[20:40, 52:72]
    before = _frame(co2=[left], gas=[left])
    after = _frame(co2=[left, right], gas=[left, right], level=0.7)
    (_, _, saturation, _), (labels, _, _, _) = _run(
        tmp_path, [before, after], [0.5, 3.0], [None, None], end_h=1.0
    )
    assert (saturation[left] == 1 - SWI).all()
    assert (labels[left] == G).all() and (labels[right] == D).all()
