"""Unknown mass: bounds on the CO2 of pixels whose CO2 is not measured.

A frame's mass maps (:func:`darsia.mass_maps`) hold the mass of the CO2 they
measure: dissolved CO2 whose colour is measured and free gas with a Swi, on the
pixels that count. CO2 of the phase map outside that, e.g. behind an excluded
part, on unmeasured colour or on free gas without a Swi, is unknown there. Its
mass is bounded, not filled, by these rules for each such pixel with a pore
volume:

- free gas with a Swi: its gas at 1 - Swi and its water saturated;
- dissolved CO2 whose colour is measured: its own colour, read like a measured
  pixel (the own estimate);
- dissolved CO2 below a measured CO2 pixel of the same column, at most a set
  distance above it: that pixel's dissolved share, or saturation below free gas,
  at this pixel's solubility (the column estimate);
- with both estimates the lower and the higher bound the pixel's dissolved mass,
  with one it is that one;
- any other pixel: from nothing to its capacity, the pore volume times the
  solubility, or the gas density where that is larger on free gas (the rest);
  without a pore volume the rest is undefined.

The frame's mass then lies between the measured mass plus the lower estimates and
the measured mass plus the higher estimates plus the rest.

The estimates are kept as statistics that do not depend on the calibration
(:func:`unknown_statistics`): the own and column shares and the solubility ratio
rounded to equal intervals, with the summed capacity of each combination, so that
any gain and dilution evaluate them without reading the frame again
(:func:`unknown_bounds`). The column estimates alone are what the calibration
fit reads (:meth:`UnknownMassStatistics.column_estimate`). Every setting is a
field of :class:`UnknownMassSettings`.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import darsia
from darsia.multi_image_analysis.calibration_fit import (
    ColumnEstimate,
    MassCalibration,
)
from darsia.multiphase.mass_maps import RunGeometry, main_pixels
from darsia.multiphase.phase_rules import PhaseLabel, model_gas

_REACH_ROUNDING = 1e-12
"""Rounding guard [m] of the column estimate's reach."""
_OWN, _COLUMN, _UNDER_GAS = 1, 2, 4
"""Bits of the kind of a pixel's estimates."""


@dataclass(frozen=True)
class UnknownMassSettings:
    """Settings of the unknown-mass bounds; one configuration table."""

    column_reach_mm: float
    """Largest distance [mm] above a pixel of the measured pixel its column
    estimate reads."""
    intervals: int
    """Number of equal intervals of the shares and ratios in the statistics."""


@dataclass(frozen=True)
class UnknownMassStatistics:
    """Calibration-free statistics of a frame's unknown mass."""

    gas_g: float
    """Mass [g] of the free gas with a Swi and its saturated water."""
    rest_g: float | None
    """Capacity [g] of the pixels without an estimate; None if one has no pore
    volume."""
    own: np.ndarray
    """Rounded relative signal of each combination's own estimate."""
    column: np.ndarray
    """Rounded relative signal of each combination's column estimate."""
    ratio: np.ndarray
    """Rounded solubility ratio (measured pixel over this pixel) of each combination."""
    kind: np.ndarray
    """Estimates of each combination: own (1), column (2), column below free gas (4)."""
    capacity_g: np.ndarray
    """Summed capacity [g] of the pixels of each combination."""

    def column_estimate(self) -> ColumnEstimate | None:
        """The statistics as the calibration fit reads them, or None without any.

        The fit reads CO2 whose estimate is its column estimate alone, not below
        free gas (a :class:`darsia.ColumnEstimate`), and sums that term before
        the dilution factor, as our production did. Statistics holding other
        unknown mass (an own estimate, a column estimate below free gas, free gas
        or a rest) are refused, since the fit would leave that mass out.

        Returns
        -------
        ColumnEstimate or None
            The column estimates.

        """
        if (self.kind != _COLUMN).any() or self.gas_g != 0 or self.rest_g != 0:
            raise ValueError("The calibration fit reads column estimates only")
        if not self.kind.size:
            return None
        return ColumnEstimate(self.column, self.ratio, self.capacity_g)


def unknown_statistics(
    relative: np.ndarray,
    labels: np.ndarray,
    rules: np.ndarray,
    measured: np.ndarray,
    geometry: RunGeometry,
    co2: darsia.CO2MassAnalysis,
    settings: UnknownMassSettings,
) -> UnknownMassStatistics:
    """Statistics of the unknown mass of one frame, for any calibration.

    Parameters
    ----------
    relative : array
        The frame's read signal relative to the concentration curve
        (:func:`darsia.relative_signal`).
    labels, rules, measured, geometry
        As for :func:`darsia.mass_maps`.
    co2 : darsia.CO2MassAnalysis
        Solubility and gas density of CO2 at the frame's pressure and temperature.
    settings : UnknownMassSettings
        Settings of the bounds.

    Returns
    -------
    UnknownMassStatistics
        The statistics, in the dtype of the relative signal and the CO2 fields.

    """
    is_gas = model_gas(labels, rules)
    dissolved, gas = main_pixels(labels, measured, geometry, is_gas)
    co2_pixels = geometry.domain & np.isin(
        labels, (PhaseLabel.DISSOLVED, PhaseLabel.GAS)
    )
    main = geometry.counted & (dissolved | gas)
    missing = co2_pixels & ~main
    valid = geometry.known
    solubility, density = co2.solubility_co2, co2.density_gaseous_co2
    volume, swi = geometry.pore_volume_m3, geometry.swi
    capacity = volume * solubility * 1000

    with_gas = missing & is_gas & valid & np.isfinite(swi)
    fixed = volume * density * (1 - swi) * 1000 + volume * solubility * swi * 1000
    gas_g = float(fixed[with_gas].sum())

    own = missing & ~is_gas & measured & valid
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])[None, :]
    above = np.maximum.accumulate(np.where(main, rows, -1), axis=0)
    donor = np.vstack((np.full((1, labels.shape[1]), -1), above[:-1]))
    height = geometry.y_m.astype(capacity.dtype)[:, None]
    distance = height[np.maximum(donor, 0), 0] - height
    reach = settings.column_reach_mm / 1000 + _REACH_ROUNDING
    column = (donor >= 0) & (distance > 0) & (distance <= reach)
    donor = np.maximum(donor, 0)
    source = relative[donor, cols]
    ratio = np.divide(
        solubility[donor, cols],
        solubility,
        out=np.zeros_like(solubility),
        where=solubility > 0,
    )
    column &= missing & ~is_gas & valid & np.isfinite(source) & np.isfinite(ratio)
    under_gas = (main & gas)[donor, cols] & column

    neither = missing & ~(with_gas | own | column)
    rest = np.where(is_gas, np.maximum(solubility, density), solubility) * volume * 1000
    rest_g = None if (neither & ~valid).any() else float(rest[neither].sum())

    pairs = missing & ~is_gas & (own | column)
    kind = (
        own[pairs] * _OWN + column[pairs] * _COLUMN + under_gas[pairs] * _UNDER_GAS
    ).astype(np.int64)
    ratio = ratio[pairs]
    scale = max(1.0, float(np.max(ratio, initial=0)))
    steps = settings.intervals
    codes = np.rint(np.clip(np.where(own, relative, 0)[pairs], 0, 1) * steps)
    codes = codes.astype(np.int64) * (steps + 1)
    codes += np.rint(np.clip(np.where(column, source, 0)[pairs], 0, 1) * steps).astype(
        np.int64
    )
    codes = codes * (steps + 1) + np.rint(ratio / scale * steps).astype(np.int64)
    codes = codes * 8 + kind
    occupied, index = np.unique(codes, return_inverse=True)
    sums = np.zeros(len(occupied), capacity.dtype)
    np.add.at(sums, index, capacity[pairs])
    kind, pair = occupied % 8, occupied // 8
    dtype = capacity.dtype
    return UnknownMassStatistics(
        gas_g=gas_g,
        rest_g=rest_g,
        own=(pair // (steps + 1) // (steps + 1)).astype(dtype) / steps,
        column=(pair // (steps + 1) % (steps + 1)).astype(dtype) / steps,
        ratio=(pair % (steps + 1)).astype(dtype) / steps * scale,
        kind=kind,
        capacity_g=sums,
    )


def unknown_bounds(
    statistics: UnknownMassStatistics, calibration: MassCalibration, factor: float
) -> tuple[float, float, float | None]:
    """The bounds on a frame's unknown mass, for a calibration.

    Parameters
    ----------
    statistics : UnknownMassStatistics
        The frame's statistics.
    calibration : MassCalibration
        The gain.
    factor : float
        The frame's dilution factor (:func:`darsia.dilution_factor`).

    Returns
    -------
    tuple
        The lower and the higher estimate [g] and the rest [g] (None if
        undefined): the frame's mass lies between its measured mass plus the lower
        estimate and its measured mass plus the higher estimate plus the rest.

    """
    dtype = statistics.capacity_g.dtype.type
    gain, factor = dtype(calibration.gain), dtype(factor)
    kind = statistics.kind
    own = np.clip(statistics.own * gain, 0, 1) * factor
    column = statistics.ratio * np.where(
        (kind & _UNDER_GAS) != 0, 1.0, np.clip(statistics.column * gain, 0, 1) * factor
    )
    has_own, has_column = (kind & _OWN) != 0, (kind & _COLUMN) != 0
    both = has_own & has_column
    single = np.where(has_own, own, column)
    weight = statistics.capacity_g
    low = dtype(np.sum(weight * np.where(both, np.minimum(own, column), single)))
    high = dtype(np.sum(weight * np.where(both, np.maximum(own, column), single)))
    return low + statistics.gas_g, high + statistics.gas_g, statistics.rest_g
