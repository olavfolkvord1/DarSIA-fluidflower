"""Gas statistics: per frame, the free gas, the water in it and the CO2 pore volume.

Beside the run statistics of the dissolved CO2
(:mod:`darsia.multi_image_analysis.run_statistics`), the calibration fit reads,
for every frame:

- the free-gas mass: the sum over the counted pixels of the pore volume times
  the gas density times the free-gas saturation;
- the gas water: the CO2 dissolved at saturation in the water of the model's
  gas (:func:`darsia.model_gas`: the phase map's free gas and the free gas that
  gravity or the peel made dissolved CO2, whose water our production kept
  saturated), the pore volume times the solubility times the water's share
  1 - saturation;
- per sand, the pore volume of its pixels of the model's gas and what they
  would hold full of gas or full of water (the fit of Swi reads them);
- the CO2 pore volume: the pore volume of the CO2 pixels the dilution law
  reads, counted or not, which it compares with that of the run's anchor
  frame; our production read the CO2 of the labels before the phase rules
  (:func:`darsia.phase_labels`).

The free-gas saturation is the gas allocation's (:func:`darsia.free_gas`), which
may also hold gas on pixels labelled dissolved; the capacity of a pixel's water
for dissolved CO2 (:func:`dissolved_capacity`, the weight of the run statistics)
is then its water's share. Densities and solubilities are in kg/m3, masses in g.

The pixel arithmetic runs in the dtype of the given arrays, and so do the sums
of the masses and of the CO2 pore volume, each over the image's rows and then
its columns. The per-sand sums run in float64, as our production computed them.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SandGas:
    """The pixels of one sand labelled free gas in one frame."""

    pore_volume_m3: float
    """Their pore volume."""
    gas_capacity_g: float
    """The free-gas mass they would hold full of gas."""
    water_capacity_g: float
    """The dissolved CO2 they would hold full of water at saturation."""


@dataclass(frozen=True)
class GasStatistics:
    """The gas statistics of one frame."""

    free_gas_g: float
    """Free-gas mass."""
    gas_water_g: float
    """CO2 dissolved at saturation in the water of the pixels labelled free gas."""
    co2_pore_volume_m3: float
    """Pore volume of the CO2 pixels."""
    sands: dict[int, SandGas]
    """Each sand with pixels labelled free gas."""


def dissolved_capacity(
    pore_volume_m3: np.ndarray, solubility: np.ndarray, saturation: np.ndarray
) -> np.ndarray:
    """Capacity [g] of each pixel's water for dissolved CO2.

    The pore volume times the solubility, times 1000, times the water's share
    1 - saturation of the pore volume; in the dtype of the arrays.

    """
    capacity = pore_volume_m3 * solubility * 1000
    capacity *= capacity.dtype.type(1) - saturation
    return capacity


def _integral(pore_volume_m3, field):
    """Sum of pore volume times field over the rows, then the columns."""
    return float(np.sum(np.sum(pore_volume_m3 * field, axis=0), axis=0))


def gas_statistics(
    co2: np.ndarray,
    gas: np.ndarray,
    saturation: np.ndarray,
    facies: np.ndarray,
    counted: np.ndarray,
    pore_volume_m3: np.ndarray,
    gas_density: np.ndarray,
    solubility: np.ndarray,
) -> GasStatistics:
    """The gas statistics of one frame.

    Parameters
    ----------
    co2 : array
        CO2 pixels whose pore volume the dilution law reads, counted or not; our
        production read those of the frame's labels before the phase rules
        (:func:`darsia.phase_labels`).
    gas : array
        The model's free gas (:func:`darsia.model_gas`), whose water counts as
        saturated.
    saturation : array
        Free-gas saturation of each pixel (e.g. of :func:`darsia.free_gas`).
    facies : array
        Facies id of each pixel.
    counted : array
        Pixels whose mass counts.
    pore_volume_m3 : array
        Pore volume of each pixel; zero where there is none.
    gas_density, solubility : array
        Density of the gaseous CO2 and solubility of CO2 in water [kg/m3].

    Returns
    -------
    GasStatistics
        The frame's gas statistics.

    """
    gas = counted & gas
    zero = saturation.dtype.type(0)
    water = solubility * np.clip(saturation.dtype.type(1) - saturation, zero, None)
    pores = np.asarray(pore_volume_m3[gas], np.float64)
    density = np.asarray(gas_density[gas], np.float64)
    dissolved = np.asarray(solubility[gas], np.float64)
    sands = {}
    for sand in np.unique(facies[gas]):
        mask = facies[gas] == sand
        sands[int(sand)] = SandGas(
            pore_volume_m3=float(pores[mask].sum()),
            gas_capacity_g=float((pores[mask] * density[mask]).sum() * 1000),
            water_capacity_g=float((pores[mask] * dissolved[mask]).sum() * 1000),
        )
    return GasStatistics(
        free_gas_g=_integral(
            pore_volume_m3, np.where(counted, gas_density * saturation, zero)
        )
        * 1000,
        gas_water_g=_integral(pore_volume_m3, np.where(gas, water, zero)) * 1000,
        co2_pore_volume_m3=float(np.sum(pore_volume_m3[co2])),
        sands=sands,
    )
