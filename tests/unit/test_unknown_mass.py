"""Unit tests for the bounds on the mass of CO2 that is not measured."""

import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

import darsia

W, D, G, U = darsia.PhaseLabel

TABLES = tomllib.loads(
    """
    # The values our production ran with.
    [unknown_mass]
    column_reach_mm = 20.0
    intervals = 32768
    """
)
SETTINGS = darsia.UnknownMassSettings(**TABLES["unknown_mass"])
CALIBRATION = darsia.MassCalibration(
    gain=1.5,
    zero_signal=0.0,
    saturation_signal=1.2,
    line_share=1.0,
    beta=0.5,
    beta_growth=0.0,
    anchor_h=2.0,
    anchor_pore_volume_m3=1e-4,
)
SOLUBILITY, DENSITY, SWI = 1.7, 1.8, 0.1


def statistics(hole=False):
    """A frame on a 10 x 3 grid of 5 mm with an excluded strip below measured CO2.

    Column 1: measured dissolved CO2 (rows 1-2) above excluded dissolved CO2 (rows
    3-7); row 3 has its own colour, rows 4-7 do not, row 7 is beyond the reach.
    Column 2: measured free gas (row 1) above excluded dissolved CO2 (row 2).
    Column 0: excluded free gas (row 5).
    """
    labels = np.zeros((10, 3), np.uint8)
    labels[1:8, 1] = D
    labels[1, 2], labels[2, 2], labels[5, 0] = G, D, G
    excluded = np.zeros((10, 3), bool)
    excluded[3:8, 1] = excluded[2, 2] = excluded[5, 0] = True
    measured = np.ones((10, 3), bool)
    measured[4:8, 1] = measured[2, 2] = False
    depth = np.full((10, 3), 0.01)
    if hole:
        depth[7, 1] = np.nan
    geometry = darsia.run_geometry(
        darsia.Image(labels, dimensions=[0.05, 0.015]),
        np.full((10, 3), 25e-6),
        depth,
        np.full((10, 3), 0.4),
        np.full((10, 3), SWI),
        np.ones((10, 3), bool),
        excluded,
    )
    co2 = SimpleNamespace(
        solubility_co2=np.full((10, 3), SOLUBILITY),
        density_gaseous_co2=np.full((10, 3), DENSITY),
    )
    relative = np.full((10, 3), 0.5)
    relative[3, 1] = 0.25
    return geometry, darsia.unknown_statistics(
        relative, labels, np.zeros_like(labels), measured, geometry, co2, SETTINGS
    )


def test_unknown_pixels_follow_the_stated_rule():
    """Own and column estimates bound dissolved CO2, free gas takes 1 - Swi with
    saturated water, the rest goes up to the capacity, undefined without geometry."""
    geometry, unknown = statistics()
    low, high, rest = darsia.unknown_bounds(unknown, CALIBRATION, 0.8)

    volume = geometry.pore_volume_m3[3, 1]
    capacity = volume * SOLUBILITY * 1000
    own = np.clip(0.25 * 1.5, 0, 1) * 0.8
    column = np.clip(0.5 * 1.5, 0, 1) * 0.8
    gas = volume * DENSITY * (1 - SWI) * 1000 + volume * SOLUBILITY * SWI * 1000
    shared = capacity * (3 * column + 1) + gas
    assert low == pytest.approx(capacity * min(own, column) + shared, rel=1e-12)
    assert high == pytest.approx(capacity * max(own, column) + shared, rel=1e-12)
    assert rest == pytest.approx(capacity, rel=1e-12)
    assert sorted(unknown.kind.tolist()) == [2, 3, 6]

    _, undefined = statistics(hole=True)
    assert undefined.rest_g is None


def test_the_calibration_fit_reads_the_column_estimates():
    """Unmeasured dissolved CO2 below measured CO2: the fit's column term, summed
    before the dilution factor, is the frame's bound on that mass."""
    labels = np.zeros((10, 3), np.uint8)
    labels[1:5, 1] = D
    measured = np.ones((10, 3), bool)
    measured[3:5, 1] = False
    geometry = darsia.run_geometry(
        darsia.Image(labels, dimensions=[0.05, 0.015]),
        np.full((10, 3), 25e-6),
        np.full((10, 3), 0.01),
        np.full((10, 3), 0.4),
        np.full((10, 3), SWI),
        np.ones((10, 3), bool),
    )
    co2 = SimpleNamespace(
        solubility_co2=np.full((10, 3), SOLUBILITY),
        density_gaseous_co2=np.full((10, 3), DENSITY),
    )
    unknown = darsia.unknown_statistics(
        np.full((10, 3), 0.5),
        labels,
        np.zeros_like(labels),
        measured,
        geometry,
        co2,
        SETTINGS,
    )
    estimate = unknown.column_estimate()
    capacity = geometry.pore_volume_m3[3, 1] * SOLUBILITY * 1000
    assert estimate.relative.tolist() == [0.5] and estimate.ratio.tolist() == [1.0]
    assert estimate.weight.tolist() == pytest.approx([2 * capacity], rel=1e-12)
    low, high, rest = darsia.unknown_bounds(unknown, CALIBRATION, 0.8)
    share = np.clip(estimate.relative * CALIBRATION.gain, 0, 1)
    term = np.sum(estimate.weight * estimate.ratio * share)
    assert low == high == pytest.approx(0.8 * term, rel=1e-12) and rest == 0
