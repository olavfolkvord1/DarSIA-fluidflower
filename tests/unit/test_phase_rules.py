"""Unit tests for the phase labels and rules on synthetic frames."""

import datetime
import tomllib

import numpy as np
import pandas as pd

import darsia

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
    hours = hours or [0.0] * len(frames)
    return tuple(
        zip(
            *darsia.phase_maps(
                frames,
                [START + datetime.timedelta(hours=h) for h in hours],
                darsia.Image(np.ones(SHAPE, int), dimensions=[0.06, 0.08]),
                kwargs.pop("domain", np.ones(SHAPE, bool)),
                kwargs.pop("pixel_area_mm2", 1.0),
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
