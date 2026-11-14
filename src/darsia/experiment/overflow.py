"""Overflow reference: the free-gas volume of a run's frames from displaced water.

Injected CO2 displaces water from the porous layers, and the overflow carries
the displaced water, together with a slowly drifting feed of water, to a scale.
The overflow measures that displaced water, never CO2 leaving the rig. The
displaced volume of a run is its scale weight above the feed (one gram of water
per mL), and the free-gas volume of a frame is that volume at the frame's time,
at least zero (:func:`measured_reference`).

The feed (:func:`displaced_volume`) stops where the readings stop rising: at
the first reading, from ``supply_check_h`` on, after which the weight rises by
less than ``supply_stop_rise_g`` within ``supply_check_span_h``. Over the late
window, from ``late_window_h[0]`` to ``late_window_h[1]`` before that stop, the
weight is the feed alone. The feed starts from the last reading before the
injection, with a rate and a constant change of rate fitted over the late window.

Every setting is a field of :class:`OverflowSettings`.

"""

from __future__ import annotations

from collections.abc import Sequence
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


@dataclass(frozen=True)
class DisplacedVolume:
    """The displaced water volume of a run at its scale readings."""

    times_h: np.ndarray
    """Times of the readings [h since the injection start], increasing."""
    volume_ml: np.ndarray | None
    """Displaced water volume [mL] at each reading; None without a feed."""
    reasons: tuple[str, ...]
    """Why the readings are not usable; empty when they are."""


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
