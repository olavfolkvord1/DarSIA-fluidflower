"""Overflow reference: the free-gas volume of a run's frames from displaced water.

Injected CO2 displaces water from the porous layers, and the overflow carries
the displaced water, together with a slowly drifting feed of water, to a scale.
The overflow measures that displaced water, never CO2 leaving the rig. The
displaced volume of a run is its scale weight above the feed (one gram of water
per mL), and the free-gas volume of a frame is that volume at the frame's time,
at least zero (:func:`measured_reference`). A frame without usable readings of
its own takes the free-gas share of the injected CO2 mass from the most similar
run with readings (:func:`ranked_donors`, :func:`estimated_reference`).

The feed (:func:`displaced_volume`) stops where the readings stop rising: at
the first reading, from ``supply_check_h`` on, after which the weight rises by
less than ``supply_stop_rise_g`` within ``supply_check_span_h``. Over the late
window, from ``late_window_h[0]`` to ``late_window_h[1]`` before that stop, the
weight is the feed alone. The feed starts from the last reading before the
injection, with a rate and a constant change of rate fitted over the late window.

Every setting is a field of :class:`OverflowSettings`.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OverflowSettings:
    """Settings of the overflow reference; one configuration table."""

    statuses: Sequence[str]
    """Review statuses of the scale readings that are used."""
    minimum_readings: int
    """Fewest readings a run needs."""
    supply_check_h: float
    """Time [h] from which the feed may stop."""
    supply_check_span_h: float
    """Time [h] after a reading within which a rising weight shows the feed."""
    supply_stop_rise_g: float
    """The feed has stopped at a reading the weight then rises less than this."""
    late_window_h: Sequence[float]
    """Start and end [h] of the late window, counted back from the feed's stop."""
    minimum_supply_h: float
    """Readings whose feed stops before this time [h] are not usable."""
    maximum_late_sd_g: float
    """Largest standard deviation of the late weight about a line."""
    maximum_pre_injection_offset_ml: float
    """Largest mean distance of the readings before the injection from that line."""
    pre_injection_span_h: float
    """Readings before the injection spanning this time [h] give their rate."""
    maximum_pre_injection_rate_g_h: float
    """Readings before the injection must rise slower than this."""
    recipe_tolerance: Mapping[str, float]
    """Recipe entries a donor may differ in, by this fraction of the larger value."""


@dataclass(frozen=True)
class DisplacedVolume:
    """The displaced water volume of a run at its scale readings."""

    times_h: np.ndarray
    """Times of the readings [h since the injection start], increasing."""
    volume_ml: np.ndarray | None
    """Displaced water volume [mL] at each reading; None without a feed."""
    reasons: tuple[str, ...]
    """Why the readings are not usable; empty when they are."""


@dataclass(frozen=True)
class OverflowRun:
    """A run as the estimated reference compares it with other runs."""

    sand_map: str
    """Identity of the run's sand map; a donor shares it."""
    recipe: Mapping[str, object]
    """The run's chemistry; a donor matches it within the recipe tolerance."""
    pressure_h: np.ndarray
    """Times [h since the injection start] of the atmospheric pressure record."""
    pressure_bar: np.ndarray
    """Atmospheric pressure [bar] at those times."""
    injection_h: tuple[float, float]
    """First start and last end [h] of the run's injection."""
    volume: DisplacedVolume | None
    """The run's displaced volume, if it has readings."""


def displaced_volume(
    times_h: np.ndarray, weights_g: np.ndarray, settings: OverflowSettings
) -> DisplacedVolume:
    """Displaced water volume of a run at its scale readings.

    The volume is the weight above the feed described in the module. The
    readings are usable with at least ``minimum_readings`` of them, one before
    the injection, a feed lasting until ``minimum_supply_h``, a late weight
    close to a line (``maximum_late_sd_g``), readings before the injection close
    to that line (``maximum_pre_injection_offset_ml``) and, where they span
    ``pre_injection_span_h``, rising slower than
    ``maximum_pre_injection_rate_g_h``. The fits are in float64.

    Parameters
    ----------
    times_h : array
        Times of the readings [h since the injection start], increasing.
    weights_g : array
        Scale weights [g].
    settings : OverflowSettings
        Settings of the feed and of usable readings.

    Returns
    -------
    DisplacedVolume
        The volume at each reading and why the readings are not usable, if so.

    """
    t = np.asarray(times_h, np.float64)
    w = np.asarray(weights_g, np.float64)
    pre = t < 0
    if t.size < settings.minimum_readings or not pre.any():
        return DisplacedVolume(t, None, ("too few readings, or none before injection",))

    stop = t[-1]
    for i in np.flatnonzero(t >= settings.supply_check_h):
        later = (t > t[i]) & (t <= t[i] + settings.supply_check_span_h)
        if later.any() and w[later].max() - w[i] < settings.supply_stop_rise_g:
            stop = t[i]
            break
    late = (t >= stop - settings.late_window_h[0]) & (
        t <= stop - settings.late_window_h[1]
    )
    rate, offset = np.polyfit(t[late], w[late], 1)
    line = w - (offset + rate * t)

    first = np.flatnonzero(pre)[-1]
    dt = t - t[first]
    (rate, drift), *_ = np.linalg.lstsq(
        np.c_[dt[late], 0.5 * dt[late] ** 2], w[late] - w[first], rcond=None
    )
    volume = w - (w[first] + rate * dt + 0.5 * drift * dt**2)

    reasons = []
    if stop < settings.minimum_supply_h:
        reasons.append(f"the feed stops at {stop:g} h")
    if np.std(line[late]) > settings.maximum_late_sd_g:
        reasons.append(f"the late weight deviates by {np.std(line[late]):g} g")
    if abs(line[pre].mean()) > settings.maximum_pre_injection_offset_ml:
        reasons.append(f"the late line misses the start by {line[pre].mean():g} mL")
    if pre.sum() >= 2 and np.ptp(t[pre]) >= settings.pre_injection_span_h:
        pre_rate = np.polyfit(t[pre], w[pre], 1)[0]
        if abs(pre_rate) >= settings.maximum_pre_injection_rate_g_h:
            reasons.append(f"the weight rises {pre_rate:g} g/h before injection")
    return DisplacedVolume(t, volume, tuple(reasons))


def _volume_at(volume: DisplacedVolume, elapsed_h: float):
    """The displaced volume [mL] at a time, linear between readings; None outside.

    The interpolation runs in the dtype of the volume's arrays.
    """
    t, v = volume.times_h, volume.volume_ml
    x = t.dtype.type(elapsed_h)
    if not t[0] <= x <= t[-1]:
        return None
    high = int(np.clip(np.searchsorted(t, x, side="right"), 1, len(t) - 1))
    if x >= t[-1]:
        return v[-1]
    return v[high - 1] + (x - t[high - 1]) / (t[high] - t[high - 1]) * (
        v[high] - v[high - 1]
    )


def measured_reference(volume: DisplacedVolume, elapsed_h: float) -> float | None:
    """Free-gas volume [m3] of a frame from its run's displaced water.

    Parameters
    ----------
    volume : DisplacedVolume
        The run's usable displaced volume; the interpolation runs in the dtype
        of its arrays.
    elapsed_h : float
        The frame's time [h since the injection start].

    Returns
    -------
    float or None
        The displaced volume at the frame's time, at least zero; None outside
        the readings.

    """
    observed = _volume_at(volume, elapsed_h)
    if observed is None:
        return None
    return max(0.0, float(observed * 1e-6))


def _compatible(recipient: OverflowRun, donor: OverflowRun, settings) -> bool:
    """Same sand map and recipe, the tolerated entries within their tolerance."""
    if (
        donor.sand_map != recipient.sand_map
        or donor.recipe.keys() != recipient.recipe.keys()
    ):
        return False
    for key, value in recipient.recipe.items():
        other = donor.recipe[key]
        if key in settings.recipe_tolerance:
            if abs(value - other) / max(value, other) > settings.recipe_tolerance[key]:
                return False
        elif value != other:
            return False
    return True


def _mean_difference(recipient, donor, start, stop):
    """Mean absolute pressure difference [bar] over [start, stop].

    Both records are linear between their samples; between two knots the
    absolute difference is integrated exactly, also where its sign changes.
    """
    (rt, rp), (dt, dp) = recipient, donor
    knots = np.unique(
        np.r_[
            start, rt[(rt > start) & (rt < stop)], dt[(dt > start) & (dt < stop)], stop
        ]
    )
    difference = np.interp(knots, rt, rp) - np.interp(knots, dt, dp)
    a, b = np.abs(difference[:-1]), np.abs(difference[1:])
    area = (a + b) / 2
    crosses = difference[:-1] * difference[1:] < 0
    area[crosses] = (a[crosses] ** 2 + b[crosses] ** 2) / (
        2 * (a[crosses] + b[crosses])
    )
    return float(np.dot(area, np.diff(knots)) / (stop - start))


def ranked_donors(
    recipient: str, runs: Mapping[str, OverflowRun], settings: OverflowSettings
) -> list[str]:
    """Runs whose displaced volume can stand in for a run's, most similar first.

    A donor has usable readings, the recipient's sand map and its recipe
    (equal entries, except those in ``recipe_tolerance``, which may differ by
    that fraction of the larger value). Donors are ranked by the mean absolute
    difference of their atmospheric pressure from the recipient's over the
    comparison window: from the recipient's injection start to the later of its
    injection end and the last reading of any donor, within the recipient's
    pressure record. A donor whose pressure record misses part of the window is
    left out; ties go by name.

    Parameters
    ----------
    recipient : str
        Name of the run that needs a donor.
    runs : mapping of str to OverflowRun
        The runs by name, the recipient among them.
    settings : OverflowSettings
        The recipe tolerance.

    Returns
    -------
    list of str
        Names of the donors, most similar first.

    """
    own = runs[recipient]
    compatible = {
        name: run
        for name, run in runs.items()
        if name != recipient
        and run.volume is not None
        and not run.volume.reasons
        and _compatible(own, run, settings)
    }
    if not compatible:
        return []
    start = max(float(own.pressure_h[0]), own.injection_h[0])
    stop = min(
        float(own.pressure_h[-1]),
        max(
            own.injection_h[1], *(run.volume.times_h[-1] for run in compatible.values())
        ),
    )
    if stop <= start:
        return []
    record = (own.pressure_h, own.pressure_bar)
    ranked = sorted(
        (
            _mean_difference(record, (run.pressure_h, run.pressure_bar), start, stop),
            name,
        )
        for name, run in compatible.items()
        if run.pressure_h[0] <= start and run.pressure_h[-1] >= stop
    )
    return [name for _, name in ranked]


def estimated_reference(
    donor: DisplacedVolume,
    elapsed_h: float,
    donor_mass_g: float,
    donor_density: float,
    mass_g: float,
    density: float,
) -> float | None:
    """Free-gas volume [m3] of a frame from a donor run's displaced water.

    The donor's free gas at the frame's time, as a share of its injected CO2
    mass, is that share of the frame's injected mass, at the frame's gas
    density.

    Parameters
    ----------
    donor : DisplacedVolume
        The donor's usable displaced volume; the interpolation runs in the
        dtype of its arrays.
    elapsed_h : float
        The frame's time [h since the injection start].
    donor_mass_g, donor_density : float
        The donor's injected CO2 mass [g] and its gas density [kg/m3] at that
        time.
    mass_g, density : float
        The frame's injected CO2 mass [g] and gas density [kg/m3].

    Returns
    -------
    float or None
        The volume; None outside the donor's readings, or where its share is
        not between zero and one.

    """
    observed = _volume_at(donor, elapsed_h)
    if observed is None or mass_g < 0:
        return None
    volume = max(0.0, float(observed * 1e-6))
    if donor_mass_g > 0:
        share = volume * donor_density * 1000 / donor_mass_g
    elif volume == 0:
        share = 0.0
    else:
        return None
    if not 0 <= share <= 1:
        return None
    return share * mass_g / (density * 1000)
