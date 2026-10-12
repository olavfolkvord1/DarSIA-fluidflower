"""Conservative, causal hypotheses for saturation within observed free gas.

This module does not recover three-dimensional saturation from a phase image.
Its parameters are declared exploratory assumptions, not learned or validated
physical constants. It leaves the observed gas support unchanged. Continuity is
only continuity at the supplied sampling times; unobserved loss and reappearance
between frames cannot be detected.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt


class GasHistoryTracker:
    """Track a local active-injection age and a physical gas-interior distance.

    Parameters
    ----------
    cell_size_m : tuple[float, float]
        Physical cell sizes in array-axis order, ``(dy, dx)``, in metres.
        Derive these from the physical domain dimensions and the analysis grid.
    low, high : float
        Baseline and limiting saturation hypotheses within observed gas.
    tau_h : float
        Active-injection exposure scale, in hours.
    core_mm : float
        Distance scale for full interior weight, in millimetres.
    gas_threshold : float
        Minimum observed gas fraction defining the binary gas core. Any positive
        fraction below this threshold still receives the baseline saturation.
    injection_intervals_h : sequence of pairs of float, optional
        Disjoint active-injection intervals ``(start_h, end_h)``. Starts must be
        nonnegative and ends must exceed starts. Intervals are sorted and
        adjacent intervals are merged; overlapping intervals are rejected.
        If omitted, injection is active from zero through ``injection_end_h``.

    Notes
    -----
    The unbounded-by-history enrichment is
    ``(1 - exp(-age / tau_h)) * min(distance / core_distance, 1)``.
    Age starts at zero on first observation or core reappearance. It advances
    only where successive observations both contain core and the local fraction
    has not decreased, using actual time overlap with the injection intervals.
    Age freezes on a decrease and resets to zero on core loss. Interior distance
    is the Euclidean distance to the nearest non-core cell centre; the exterior
    of the image is treated as non-core, including for a completely full image.

    Enrichment is multiplied by current fraction / maximum fraction since core
    arrival. A local decrease also caps it by the previous enrichment times the
    fractional retention since the previous frame. Any update interval with no
    active injection cannot increase enrichment, even if the observed core
    widens. An interval crossing an injection boundary uses only active exposure.
    There is no filling, temporal bridging, component tracking, or future input.
    """

    def __init__(
        self,
        cell_size_m: tuple[float, float],
        *,
        low: float = 0.6,
        high: float = 0.9,
        tau_h: float = 1.0,
        core_mm: float = 10.0,
        gas_threshold: float = 0.5,
        injection_intervals_h: ArrayLike | None = None,
    ) -> None:
        spacing = np.asarray(cell_size_m, dtype=float)
        if spacing.shape != (2,) or not np.all(np.isfinite(spacing)):
            raise ValueError("cell_size_m must contain two finite values (dy, dx).")
        if np.any(spacing <= 0):
            raise ValueError("cell_size_m values must be positive.")
        parameters = np.asarray([low, high, tau_h, core_mm, gas_threshold], dtype=float)
        if not np.all(np.isfinite(parameters)):
            raise ValueError("All hypothesis parameters must be finite.")
        if not 0 <= low <= high <= 1:
            raise ValueError("Saturation bounds must satisfy 0 <= low <= high <= 1.")
        if tau_h <= 0 or core_mm <= 0:
            raise ValueError("tau_h and core_mm must be positive.")
        if not 0 < gas_threshold <= 1:
            raise ValueError("gas_threshold must be in (0, 1].")

        intervals = None
        if injection_intervals_h is not None:
            supplied = np.asarray(injection_intervals_h, dtype=float)
            if (
                supplied.ndim != 2
                or supplied.shape[1] != 2
                or supplied.shape[0] == 0
                or not np.all(np.isfinite(supplied))
                or np.any(supplied[:, 0] < 0)
                or np.any(supplied[:, 1] <= supplied[:, 0])
            ):
                raise ValueError(
                    "injection_intervals_h must contain finite [start, end] pairs "
                    "with 0 <= start < end."
                )
            intervals = []
            for start, end in sorted(map(tuple, supplied.tolist())):
                if intervals and start < intervals[-1][1]:
                    raise ValueError("injection_intervals_h must not overlap.")
                if intervals and start == intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], end)
                else:
                    intervals.append((start, end))
            intervals = tuple(intervals)

        self.cell_size_m = tuple(float(value) for value in spacing)
        self.low = float(low)
        self.high = float(high)
        self.tau_h = float(tau_h)
        self.core_mm = float(core_mm)
        self.gas_threshold = float(gas_threshold)
        self.injection_intervals_h = intervals
        self._elapsed_h: float | None = None
        self._injection_end_h: float | None = None
        self._fraction: NDArray[np.float64] | None = None
        self._age_h: NDArray[np.float64] | None = None
        self._peak: NDArray[np.float64] | None = None
        self._score: NDArray[np.float64] | None = None

    def update(
        self,
        current_gas_fraction: ArrayLike,
        elapsed_h: float,
        injection_end_h: float,
    ) -> tuple[NDArray[np.float32], dict[str, NDArray | float]]:
        """Return a saturation hypothesis and matching diagnostic fields.

        Parameters
        ----------
        current_gas_fraction : array_like
            Finite two-dimensional gas-volume / pore-volume fractions in [0, 1].
            Supply zero wherever there is no pore volume or no observed gas.
        elapsed_h : float
            Finite timestamp in hours, strictly increasing across calls. Negative
            baseline timestamps are allowed; injection begins at zero hours.
        injection_end_h : float
            Finite nonnegative injection-stop time, fixed for this tracker.
            Must match the final end of ``injection_intervals_h`` when supplied.

        Returns
        -------
        saturation : numpy.ndarray
            Float32 field, zero off observed gas and bounded by ``low`` and
            ``high`` on every cell with positive observed gas fraction.
        diagnostics : dict
            Matching boolean fields ``gas_mask``, ``core_mask``, and
            ``growing_or_stable_mask``; float32 fields
            ``continuous_active_age_h``, ``interior_distance_m``,
            ``peak_fraction_since_arrival``, ``retained_fraction``, and
            ``history_score``; scalar hours ``elapsed_h``, ``interval_h``,
            ``active_interval_h``, and ``injection_end_h``. Peak and retention
            refer to the current uninterrupted core episode, and are zero off
            core. The growth/stability mask requires core in both observations.
            Returned arrays do not alias tracker state or the input.
        """
        fraction = np.asarray(current_gas_fraction, dtype=np.float64)
        if fraction.ndim != 2 or 0 in fraction.shape:
            raise ValueError("current_gas_fraction must be a nonempty 2D field.")
        if not np.all(np.isfinite(fraction)):
            raise ValueError("current_gas_fraction must be finite.")
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("current_gas_fraction must lie in [0, 1].")
        elapsed_h, injection_end_h = float(elapsed_h), float(injection_end_h)
        if not np.isfinite(elapsed_h):
            raise ValueError("elapsed_h must be finite.")
        if not np.isfinite(injection_end_h) or injection_end_h < 0:
            raise ValueError("injection_end_h must be finite and nonnegative.")
        if (
            self.injection_intervals_h is not None
            and injection_end_h != self.injection_intervals_h[-1][1]
        ):
            raise ValueError(
                "injection_end_h must match the final injection_intervals_h end."
            )
        if self._elapsed_h is not None and elapsed_h <= self._elapsed_h:
            raise ValueError("elapsed_h must be strictly increasing.")
        if self._injection_end_h is not None:
            if injection_end_h != self._injection_end_h:
                raise ValueError("injection_end_h must remain fixed for the tracker.")
        if self._fraction is not None and fraction.shape != self._fraction.shape:
            raise ValueError("current_gas_fraction shape must remain fixed.")

        gas = fraction > 0
        core = fraction >= self.gas_threshold
        age_h = np.zeros(fraction.shape, dtype=np.float64)
        peak = np.where(core, fraction, 0.0)
        growing_or_stable = np.zeros(fraction.shape, dtype=bool)
        interval_h = active_interval_h = 0.0
        persistent = np.zeros(fraction.shape, dtype=bool)

        if self._elapsed_h is not None:
            interval_h = elapsed_h - self._elapsed_h
            if not np.isfinite(interval_h):
                raise ValueError("The elapsed time interval must be finite.")
            active_interval_h = sum(
                max(
                    0.0,
                    min(elapsed_h, end, injection_end_h) - max(self._elapsed_h, start),
                )
                for start, end in (
                    self.injection_intervals_h
                    if self.injection_intervals_h is not None
                    else ((0.0, injection_end_h),)
                )
            )
            persistent = core & (self._fraction >= self.gas_threshold)
            growing_or_stable = persistent & (fraction >= self._fraction)
            age_h[persistent] = self._age_h[persistent]
            age_h[growing_or_stable] += active_interval_h
            peak[persistent] = np.maximum(self._peak[persistent], fraction[persistent])

        # Padding makes the boundary convention explicit and handles full core.
        distance_m = distance_transform_edt(
            np.pad(core, 1, constant_values=False), sampling=self.cell_size_m
        )[1:-1, 1:-1]
        interior_weight = np.clip(distance_m / (self.core_mm * 1e-3), 0.0, 1.0)
        retained = np.divide(fraction, peak, out=np.zeros_like(fraction), where=core)
        with np.errstate(over="ignore", under="ignore"):
            score = -np.expm1(-age_h / self.tau_h) * interior_weight * retained

        if self._elapsed_h is not None:
            decreasing = persistent & (fraction < self._fraction)
            score[decreasing] = np.minimum(
                score[decreasing],
                self._score[decreasing]
                * fraction[decreasing]
                / self._fraction[decreasing],
            )
            if active_interval_h == 0:
                score = np.minimum(score, self._score)

        score = np.clip(score, 0.0, 1.0)
        saturation = np.where(gas, self.low + (self.high - self.low) * score, 0.0)
        diagnostics = {
            "gas_mask": gas,
            "core_mask": core,
            "growing_or_stable_mask": growing_or_stable,
            "continuous_active_age_h": age_h.astype(np.float32),
            "interior_distance_m": distance_m.astype(np.float32),
            "peak_fraction_since_arrival": peak.astype(np.float32),
            "retained_fraction": retained.astype(np.float32),
            "history_score": score.astype(np.float32),
            "elapsed_h": elapsed_h,
            "interval_h": interval_h,
            "active_interval_h": active_interval_h,
            "injection_end_h": injection_end_h,
        }
        self._fraction = fraction.copy()
        self._age_h = age_h
        self._peak = peak
        self._score = score
        self._elapsed_h = elapsed_h
        self._injection_end_h = injection_end_h
        return saturation.astype(np.float32), diagnostics
