"""Unit tests for the gas statistics of a frame."""

import numpy as np
import pytest

import darsia

W, D, G, U = darsia.PhaseLabel


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gas_statistics_hold_the_gas_its_water_and_the_co2_pore_volume(dtype):
    """Gas mass where there is gas, saturated water in the model's gas only, and
    the pore volume of all CO2, counted or not."""
    phase = np.array([[G, D, D, D], [W, G, D, U]], np.uint8)
    rules = np.zeros(phase.shape, np.uint8)
    rules[0, 1] = darsia.PhaseRule.PEELED
    saturation = np.array([[0.8, 0.0, 0.25, 0.0], [0.0, 0.8, 0.0, 0.0]], dtype)
    facies = np.array([[1, 2, 1, 2], [1, 2, 2, 2]])
    counted = np.ones(phase.shape, bool)
    counted[1, 1] = False
    pores = np.full(phase.shape, 2e-9, dtype)
    density = np.full(phase.shape, 1.8, dtype)
    solubility = np.full(phase.shape, 1.5, dtype)

    co2, gas = np.isin(phase, (D, G)), darsia.model_gas(phase, rules)
    stats = darsia.gas_statistics(
        co2, gas, saturation, facies, counted, pores, density, solubility
    )

    # Gas: the counted model gas (one pixel peeled empty) and the gas added on
    # a dissolved pixel; water at saturation in the model gas's remaining pores.
    assert stats.free_gas_g == pytest.approx(2e-9 * 1.8 * (0.8 + 0.25) * 1000)
    assert stats.gas_water_g == pytest.approx(2e-9 * 1.5 * (0.2 + 1.0) * 1000)
    assert stats.co2_pore_volume_m3 == pytest.approx(6 * 2e-9)
    assert stats.sands.keys() == {1, 2}
    assert stats.sands[2].pore_volume_m3 == pytest.approx(2e-9)
    assert stats.sands[2].gas_capacity_g == pytest.approx(2e-9 * 1.8 * 1000)
    assert stats.sands[2].water_capacity_g == pytest.approx(2e-9 * 1.5 * 1000)

    # The water's share of a pixel with added gas holds the dissolved CO2.
    capacity = darsia.dissolved_capacity(pores, solubility, saturation)
    assert capacity.dtype == dtype
    assert capacity[0, 2] == pytest.approx(2e-9 * 1.5 * 1000 * 0.75)
