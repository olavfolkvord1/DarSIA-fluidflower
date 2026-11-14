"""Overflow fed from a tank: the displaced water is the weight above the feed.

Water is pumped from a tank through the free water and over the overflow to
the scale until the tank runs empty, so the weight always grows. The water the
injected CO2 displaces from the porous layers is the weight above that feed.

The feed stops where the readings stop rising: at the first reading, from
``supply_check_h`` on, after which the weight rises by less than
``supply_stop_rise_g`` within ``supply_check_span_h``. Over the late window,
from ``late_window_h[0]`` to ``late_window_h[1]`` before that stop, the weight
is the feed alone. The feed starts from the reference reading, the last before
the injection, with a rate and a constant change of rate fitted over the late
window. A steady evaporation lowers the weight as a slower feed would, so it
lies inside the fitted rate. The stop, the late window and the checks read the
weights as read; the fit reads them with any evaporated water added back, where
a steady rate cancels exactly.

Every setting is a field of :class:`FeedSettings`.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeedSettings:
    """Settings of an overflow fed from a tank; the ``[overflow.feed]`` table."""

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

    def displaced_ml(
        self,
        times_h: np.ndarray,
        weights_g: np.ndarray,
        corrected_g: np.ndarray,
        baseline: np.ndarray,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Displaced water [mL] at each reading: the weight above the feed.

        The readings are usable with a feed lasting until ``minimum_supply_h``,
        a late weight close to a line (``maximum_late_sd_g``), readings before
        the injection close to that line (``maximum_pre_injection_offset_ml``)
        and, where they span ``pre_injection_span_h``, rising slower than
        ``maximum_pre_injection_rate_g_h``.

        Parameters
        ----------
        times_h : array
            Times of the readings [h since the injection start], increasing.
        weights_g : array
            Scale weights [g] as read; the stop, the late window and the checks
            read them.
        corrected_g : array
            The weights with any evaporated water added back; the fit reads
            them.
        baseline : array of bool
            The baseline readings. Not read: the feed counts from the last
            reading before the injection and checks all readings before it,
            whatever the baseline window.

        Returns
        -------
        tuple
            The volume at each reading, and why the readings are not usable;
            empty when they are.

        """
        t, w, c = times_h, weights_g, corrected_g
        pre = t < 0
        reference = np.flatnonzero(pre)[-1]
        stop = t[-1]
        for i in np.flatnonzero(t >= self.supply_check_h):
            later = (t > t[i]) & (t <= t[i] + self.supply_check_span_h)
            if later.any() and w[later].max() - w[i] < self.supply_stop_rise_g:
                stop = t[i]
                break
        late = (t >= stop - self.late_window_h[0]) & (t <= stop - self.late_window_h[1])
        rate, offset = np.polyfit(t[late], w[late], 1)
        line = w - (offset + rate * t)

        dt = t - t[reference]
        (rate, drift), *_ = np.linalg.lstsq(
            np.c_[dt[late], 0.5 * dt[late] ** 2], c[late] - c[reference], rcond=None
        )
        volume = c - (c[reference] + rate * dt + 0.5 * drift * dt**2)

        reasons = []
        if stop < self.minimum_supply_h:
            reasons.append(f"the feed stops at {stop:g} h")
        if np.std(line[late]) > self.maximum_late_sd_g:
            reasons.append(f"the late weight deviates by {np.std(line[late]):g} g")
        if abs(line[pre].mean()) > self.maximum_pre_injection_offset_ml:
            reasons.append(f"the late line misses the start by {line[pre].mean():g} mL")
        if pre.sum() >= 2 and np.ptp(t[pre]) >= self.pre_injection_span_h:
            pre_rate = np.polyfit(t[pre], w[pre], 1)[0]
            if abs(pre_rate) >= self.maximum_pre_injection_rate_g_h:
                reasons.append(f"the weight rises {pre_rate:g} g/h before injection")
        return volume, tuple(reasons)
