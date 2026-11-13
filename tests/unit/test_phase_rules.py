"""Unit tests for the phase labels and rules on synthetic frames."""

import tomllib

import numpy as np

import darsia

W, D, G, U = darsia.PhaseLabel

SETTINGS = darsia.PhaseRuleSettings(
    **tomllib.loads(
        """
        # The values the rules ran with in production.
        [phase_rules]
        co2_threshold = 0.8
        smallest_gas_patch_mm2 = 25.0
        """
    )["phase_rules"]
)
GAS_THRESHOLD = 0.66
"""Gas threshold of our production phase model, from its folder."""

SHAPE = (60, 80)
"""Rows and columns of a 1 mm grid."""


def _frame(co2=(), gas=()):
    """Probabilities: low everywhere, 0.9 in the listed (rows, cols) boxes."""
    probabilities = np.full((2, *SHAPE), 0.1, np.float32)
    for index, boxes in enumerate((co2, gas)):
        for box in boxes:
            probabilities[index][box] = 0.9
    return tuple(probabilities)


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
