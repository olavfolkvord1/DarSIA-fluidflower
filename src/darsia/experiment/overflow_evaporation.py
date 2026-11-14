"""Evaporation correction of an overflow's displaced water.

Water evaporating from the rig or the scale lowers the weight, and with it the
displaced volume. The correction adds the evaporated water back to the weights
the volume is computed from: a rate [g/h] times the time since the injection
start, at one gram per mL (``evaporation_g_h`` of
:func:`darsia.displaced_volume`). It serves both variants of the overflow and
is off unless a rate is approved for the run.

In a circulation the weight falls by the evaporation directly, and the
corrected volume gains the rate times the time since the baseline. In a feed,
a steady evaporation lowers the weight as a slower feed would and so lies
inside the fitted feed rate: a rate added to the weights the fit reads cancels
exactly, and nothing is counted twice; the feed's stop, late window and checks
read the weights as read. The displaced volume of a feed before
the injection then has no trend either, and its proposal is near zero; a
proposal away from zero there shows readings before the injection rising at
another rate than the fitted feed, as an evaporation or a feed that changed
over time would.

The rate is proposed per run from the baseline, the readings before the
injection, where nothing is injected and nothing dissolves: the fall of the
displaced volume there, a least-squares slope with its standard error
(:func:`proposed_evaporation`). After the injection end the free gas
dissolving lowers the displaced volume too, so that period gives no rate. The
user accepts the proposal or gives a rate in its place
(:func:`approved_evaporation`).

Every setting is a field of :class:`EvaporationSettings`; the baseline window
is the overflow's ``baseline_window_h``.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from darsia.experiment.overflow import DisplacedVolume


@dataclass(frozen=True)
class EvaporationSettings:
    """Settings of the evaporation correction; the ``[overflow.evaporation]`` table."""

    minimum_readings: int
    """Fewest baseline readings that propose a rate (three at least)."""
    rates_g_h: Mapping[str, float | str]
    """Approved rate [g/h] by run: a number, or ``"measured"`` to accept the
    proposed rate. A run not listed is not corrected."""


@dataclass(frozen=True)
class EvaporationRate:
    """A run's proposed evaporation rate."""

    rate_g_h: float
    """Evaporation [g/h]: the fall of the displaced volume over the baseline."""
    sd_g_h: float
    """Standard error of the rate [g/h]."""
    readings: int
    """Number of baseline readings the rate is fitted to."""


def proposed_evaporation(
    volume: DisplacedVolume, settings: EvaporationSettings
) -> EvaporationRate | None:
    """A run's evaporation rate measured from its baseline readings.

    The rate is minus the least-squares slope of the displaced volume over the
    baseline readings, before the injection and within the baseline window;
    its standard error comes from the residuals about that line. It is
    computed in the dtype of the volume's arrays.

    Parameters
    ----------
    volume : DisplacedVolume
        The run's displaced volume, without an evaporation correction.
    settings : EvaporationSettings
        The fewest baseline readings.

    Returns
    -------
    EvaporationRate or None
        The proposed rate; None without a volume, or with fewer than
        ``minimum_readings``, or three, baseline readings, or when they share
        one time.

    """
    n = int(np.count_nonzero(volume.baseline))
    if volume.volume_ml is None or n < max(settings.minimum_readings, 3):
        return None
    t = volume.times_h[volume.baseline]
    v = volume.volume_ml[volume.baseline]
    spread = np.ptp(t)
    if not (np.isfinite(spread) and spread > 0):
        return None
    x = t - t.mean()
    y = v - v.mean()
    slope = np.dot(x, y) / np.dot(x, x)
    residual = y - slope * x
    sd = np.sqrt(np.dot(residual, residual) / (n - 2) / np.dot(x, x))
    return EvaporationRate(float(-slope), float(sd), n)


def approved_evaporation(
    run: str, proposal: EvaporationRate | None, settings: EvaporationSettings | None
) -> float | None:
    """The evaporation rate [g/h] approved for a run.

    Parameters
    ----------
    run : str
        Name of the run, as in ``rates_g_h``.
    proposal : EvaporationRate or None
        The run's proposed rate, if any.
    settings : EvaporationSettings or None
        The evaporation correction; None when none is configured.

    Returns
    -------
    float or None
        The rate given for the run, the proposed rate where that is accepted,
        or None: no correction.

    Raises
    ------
    ValueError
        If the proposed rate is accepted for a run without a proposal, or if
        the rate is not finite.

    """
    choice = None if settings is None else settings.rates_g_h.get(run)
    if choice is None:
        return None
    if choice == "measured":
        if proposal is None:
            raise ValueError(f"No evaporation rate is proposed for {run}.")
        rate = proposal.rate_g_h
    else:
        rate = float(choice)
    if not np.isfinite(rate):
        raise ValueError(f"The evaporation rate of {run} is not finite.")
    return rate
