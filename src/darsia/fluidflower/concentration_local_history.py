"""Exploratory causal local dilution and persistent gas-core hypotheses.

These features describe fixed phase maps at observed times. They do not infer
fluid age, transport paths, advection, or three-dimensional gas saturation.
Neither injected mass nor an injection schedule is an input. The local dilution
factor replaces a global dilution factor; it must not be multiplied by one.

Neighbourhoods use separable Gaussian weights with sigma = radius / 2 and a
two-sigma truncation along each physical grid axis. Thus ``radius_m`` is an
approximate axis half-width after rounding to grid cells, not a pixel radius.
No facies barrier is imposed: nearby support across facies can contribute and
must be examined when interpreting this first exploratory model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _spacing(cell_size_m: tuple[float, float]) -> tuple[float, float]:
    spacing = np.asarray(cell_size_m, dtype=float)
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)):
        raise ValueError("cell_size_m must contain two finite values (dy, dx).")
    if np.any(spacing <= 0):
        raise ValueError("cell_size_m values must be positive.")
    return tuple(map(float, spacing))


def _field(value: ArrayLike, name: str, shape=None) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError(f"{name} must be a nonempty 2D field.")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must share the fixed grid shape.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result


def _time(elapsed_h: float, previous_h: float | None, max_gap_h: float):
    elapsed_h = float(elapsed_h)
    if not np.isfinite(elapsed_h):
        raise ValueError("elapsed_h must be finite.")
    if previous_h is None:
        return elapsed_h, 0.0, False
    interval_h = elapsed_h - previous_h
    if not np.isfinite(interval_h) or interval_h <= 0:
        raise ValueError("elapsed_h must be strictly increasing with finite intervals.")
    return elapsed_h, interval_h, interval_h > max_gap_h


def local_dilution_factor(exposure: ArrayLike, beta: float) -> NDArray[np.float32]:
    """Return ``exp(-beta * exposure)``; beta zero gives exact float32 ones.

    Exposure is nonnegative local dilution evidence, rather than elapsed time.
    Apply this factor to the optical aqueous response only. Saturated residual
    water in gas-labelled volume remains a separate hypothesis/mass term.
    """
    exposure = _field(exposure, "exposure")
    beta = float(beta)
    if not np.isfinite(beta) or beta < 0 or np.any(exposure < 0):
        raise ValueError("beta and exposure must be finite and nonnegative.")
    if beta == 0:
        return np.ones(exposure.shape, dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-beta * exposure).astype(np.float32)


class LocalDilutionTracker:
    """Track neighbourhood growth and modulate it by current optical evidence.

    ``a = Gaussian(av + gv) / Gaussian(pv)`` is local affected pore occupancy.
    On arrival of observed aqueous support, ``a0 = max(a, arrival_floor)`` and
    exposure starts at zero. On subsequent observations with continuous aqueous
    support, increasing occupancy can raise peak geometric exposure to
    ``E = max(E_previous, max(0, log(max(a, floor) / a0)))``. Exposure never grows
    merely because time passes. Current optical weakness and gas contact give
    ``L = clip(E, 0, max_exposure) * (1 - clip(signal / signal_scale, 0, 1))**p
    * (1 - contact)``. Contact is local gas pore occupancy divided by a declared
    occupancy scale, clipped to [0, 1]; a tiny gas trace offers little protection.

    The default signal scale of three is an explicit assumption about the source
    reference-normalized signal units, not an endpoint selected from outcomes.
    The arrival floor prevents tiny phase slivers from creating large ratios.
    Gap reset is strict: only intervals greater than ``max_gap_h`` reset.
    New/reappearing support and first observations have zero exposure. Negative
    baseline timestamps are allowed. Arrays are never modified in place.
    """

    def __init__(
        self,
        cell_size_m: tuple[float, float],
        *,
        radius_m: float = 0.02,
        gas_contact_radius_m: float = 0.01,
        gas_contact_scale: float = 0.1,
        signal_scale: float = 3.0,
        weakness_power: float = 2.0,
        arrival_occupancy_floor: float = 0.02,
        max_exposure: float = 6.0,
        max_gap_h: float = 48.0,
    ) -> None:
        self.cell_size_m = _spacing(cell_size_m)
        self.radius_m = _positive(radius_m, "radius_m")
        self.gas_contact_radius_m = _positive(
            gas_contact_radius_m, "gas_contact_radius_m"
        )
        self.gas_contact_scale = _positive(gas_contact_scale, "gas_contact_scale")
        self.signal_scale = _positive(signal_scale, "signal_scale")
        self.weakness_power = _positive(weakness_power, "weakness_power")
        self.arrival_occupancy_floor = _positive(
            arrival_occupancy_floor, "arrival_occupancy_floor"
        )
        if self.arrival_occupancy_floor > 1 or self.gas_contact_scale > 1:
            raise ValueError("Occupancy floor and gas contact scale cannot exceed one.")
        self.max_exposure = _positive(max_exposure, "max_exposure")
        self.max_gap_h = _positive(max_gap_h, "max_gap_h")
        self._elapsed_h = None
        self._pv = None
        self._support = None
        self._arrival = None
        self._local_occupancy = None
        self._exposure = None

    def _smooth(self, value, radius_m):
        sigma = tuple(radius_m / (2 * cell) for cell in self.cell_size_m)
        return gaussian_filter(value, sigma=sigma, truncate=2.0, mode="constant")

    def _occupancy(self, volume, pv, radius_m):
        numerator = self._smooth(volume, radius_m)
        denominator = self._smooth(pv, radius_m)
        return np.clip(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            ),
            0,
            1,
        )

    def update(
        self,
        signal: ArrayLike,
        av: ArrayLike,
        gv: ArrayLike,
        pv: ArrayLike,
        elapsed_h: float,
    ) -> tuple[NDArray[np.float32], dict]:
        """Return float32 L and diagnostic fields on the unchanged image grid.

        ``av``, ``gv`` and ``pv`` are physical cell volumes in the same units.
        Their values must be nonnegative, ``av + gv <= pv`` (apart from relative
        floating-point roundoff), and pv must remain fixed between calls.
        Signal can include finite negative baseline residuals, which give maximum
        optical weakness. Only cells with ``av > 0`` retain aqueous history.

        Diagnostic arrays are ``aqueous_support``, ``persistent_support``,
        ``local_affected_fraction``, ``arrival_occupancy``, ``growth`` (positive
        local occupancy change), ``exposure_unclipped``, ``exposure``,
        ``exposure_clipped_mask``, ``gas_contact_fraction``, ``contact``,
        ``weakness`` and ``local_exposure`` (L). Numeric arrays are float32.
        Scalars record times, gap reset, and the clipped fraction of aqueous
        support. Returned arrays do not alias the input or private tracker state.
        """
        shape = None if self._pv is None else self._pv.shape
        signal = _field(signal, "signal", shape)
        av, gv, pv = (
            _field(value, name, signal.shape)
            for value, name in ((av, "av"), (gv, "gv"), (pv, "pv"))
        )
        if any(np.any(value < 0) for value in (av, gv, pv)):
            raise ValueError("Physical volumes must be nonnegative.")
        if np.any(av + gv > pv * (1 + 1e-6)):
            raise ValueError("av + gv cannot exceed pore volume.")
        if self._pv is not None and not np.array_equal(pv, self._pv):
            raise ValueError("pv must remain fixed between observations.")
        elapsed_h, interval_h, gap_reset = _time(
            elapsed_h, self._elapsed_h, self.max_gap_h
        )

        support = av > 0
        local = self._occupancy(np.minimum(av + gv, pv), pv, self.radius_m)
        gas_fraction = self._occupancy(
            np.minimum(gv, pv), pv, self.gas_contact_radius_m
        )
        contact = np.clip(gas_fraction / self.gas_contact_scale, 0, 1)
        weakness = (
            1 - np.clip(signal / self.signal_scale, 0, 1)
        ) ** self.weakness_power
        persistent = (
            support & self._support
            if self._support is not None and not gap_reset
            else np.zeros(signal.shape, dtype=bool)
        )
        arrival = np.where(support, np.maximum(local, self.arrival_occupancy_floor), 0)
        exposure = np.zeros(signal.shape, dtype=np.float64)
        growth = np.zeros(signal.shape, dtype=np.float64)
        if np.any(persistent):
            arrival[persistent] = self._arrival[persistent]
            exposure[persistent] = self._exposure[persistent]
            growth[persistent] = np.maximum(
                0, local[persistent] - self._local_occupancy[persistent]
            )
            increasing = persistent & (growth > 0)
            candidate = np.log(
                np.maximum(local[increasing], self.arrival_occupancy_floor)
                / arrival[increasing]
            )
            exposure[increasing] = np.maximum(exposure[increasing], candidate)
        exposure = np.maximum(exposure, 0)
        clipped = exposure > self.max_exposure
        bounded_exposure = np.minimum(exposure, self.max_exposure)
        local_exposure = bounded_exposure * weakness * (1 - contact)
        diagnostics = {
            "aqueous_support": support,
            "persistent_support": persistent,
            "local_affected_fraction": local.astype(np.float32),
            "arrival_occupancy": arrival.astype(np.float32),
            "growth": growth.astype(np.float32),
            "exposure_unclipped": exposure.astype(np.float32),
            "exposure": bounded_exposure.astype(np.float32),
            "exposure_clipped_mask": clipped,
            "exposure_clipped_fraction": (
                float(clipped.sum() / support.sum()) if support.any() else 0.0
            ),
            "gas_contact_fraction": gas_fraction.astype(np.float32),
            "contact": contact.astype(np.float32),
            "weakness": weakness.astype(np.float32),
            "local_exposure": local_exposure.astype(np.float32),
            "elapsed_h": elapsed_h,
            "interval_h": interval_h,
            "history_reset": gap_reset,
        }
        self._pv = pv.copy()
        self._support = support.copy()
        self._arrival = arrival
        self._local_occupancy = local
        self._exposure = exposure
        self._elapsed_h = elapsed_h
        return local_exposure.astype(np.float32), diagnostics


class PersistentGasCoreTracker:
    """A causal persistent-core hypothesis without a global injection gate.

    Continuous age advances by actual elapsed hours only when core persists and
    neither the local gas fraction nor its physical interior distance retreats.
    First observations, core reappearance and gaps greater than ``max_gap_h``
    start at zero age. Negative timestamps are accepted; no calendar schedule is
    consulted. Local stability can therefore enrich a core after injection.

    The score is ``(1-exp(-age/tau))*clip(distance/core_distance,0,1)`` times
    retention relative to the maximum fraction and interior distance of the
    current core episode. A retreat additionally caps the score by the previous
    score times the smaller fractional/distance retention since the prior frame.
    Thus retreat drains enrichment immediately instead of being outweighed by
    ageing. During stability or growth, enrichment can only approach the current
    interior-times-retention target by ``1-exp(-dt/tau)`` of the remaining gap.
    This preserves the age formula for an unchanged core while ensuring that
    recovery after retreat takes actual time. This is an explicit saturation
    hypothesis, not a reconstruction of measured 3D gas.
    """

    def __init__(
        self,
        cell_size_m: tuple[float, float],
        *,
        low: float = 0.6,
        high: float = 0.9,
        tau_h: float = 2.0,
        core_mm: float = 10.0,
        gas_threshold: float = 0.5,
        max_gap_h: float = 48.0,
    ) -> None:
        self.cell_size_m = _spacing(cell_size_m)
        if not np.isfinite([low, high]).all() or not 0 <= low <= high <= 1:
            raise ValueError("Saturation bounds must satisfy 0 <= low <= high <= 1.")
        self.low, self.high = float(low), float(high)
        self.tau_h = _positive(tau_h, "tau_h")
        self.core_mm = _positive(core_mm, "core_mm")
        self.gas_threshold = _positive(gas_threshold, "gas_threshold")
        if self.gas_threshold > 1:
            raise ValueError("gas_threshold cannot exceed one.")
        self.max_gap_h = _positive(max_gap_h, "max_gap_h")
        self._elapsed_h = None
        self._fraction = None
        self._age_h = None
        self._distance_m = None
        self._peak_fraction = None
        self._peak_distance_m = None
        self._score = None

    def update(
        self, gas_fraction: ArrayLike, elapsed_h: float
    ) -> tuple[NDArray[np.float32], dict]:
        """Return float32 Sg plus local age, core, distance and retreat diagnostics.

        Fractions must be finite in [0, 1] on a fixed two-dimensional grid.
        The saturation is zero off gas and at least ``low`` wherever fraction>0,
        including sub-core edge cells. Padded non-core exterior makes the EDT
        convention explicit even for a completely full gas-core image.
        """
        fraction = _field(
            gas_fraction,
            "gas_fraction",
            None if self._fraction is None else self._fraction.shape,
        )
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("gas_fraction must lie in [0, 1].")
        elapsed_h, interval_h, gap_reset = _time(
            elapsed_h, self._elapsed_h, self.max_gap_h
        )
        gas = fraction > 0
        core = fraction >= self.gas_threshold
        distance = distance_transform_edt(
            np.pad(core, 1, constant_values=False), sampling=self.cell_size_m
        )[1:-1, 1:-1]
        persistent = (
            core & (self._fraction >= self.gas_threshold)
            if self._fraction is not None and not gap_reset
            else np.zeros(fraction.shape, dtype=bool)
        )
        age_h = np.zeros(fraction.shape, dtype=np.float64)
        peak_fraction = np.where(core, fraction, 0)
        peak_distance = distance.copy()
        retreat_retention = np.zeros(fraction.shape, dtype=np.float64)
        growing_or_stable = np.zeros(fraction.shape, dtype=bool)
        retreat = np.zeros(fraction.shape, dtype=bool)
        if np.any(persistent):
            age_h[persistent] = self._age_h[persistent]
            peak_fraction[persistent] = np.maximum(
                peak_fraction[persistent], self._peak_fraction[persistent]
            )
            peak_distance[persistent] = np.maximum(
                peak_distance[persistent], self._peak_distance_m[persistent]
            )
            retreat_retention[persistent] = np.minimum(
                np.minimum(fraction[persistent] / self._fraction[persistent], 1),
                np.minimum(distance[persistent] / self._distance_m[persistent], 1),
            )
            retreat = persistent & (retreat_retention < 1)
            growing_or_stable = persistent & ~retreat
            age_h[growing_or_stable] += interval_h
        retained_fraction = np.divide(
            fraction, peak_fraction, out=np.zeros_like(fraction), where=core
        )
        retained_distance = np.divide(
            distance, peak_distance, out=np.zeros_like(distance), where=core
        )
        interior = np.clip(distance / (self.core_mm * 1e-3), 0, 1)
        with np.errstate(over="ignore", under="ignore"):
            score = (
                -np.expm1(-age_h / self.tau_h)
                * interior
                * np.minimum(retained_fraction, retained_distance)
            )
        if np.any(retreat):
            score[retreat] = np.minimum(
                score[retreat], self._score[retreat] * retreat_retention[retreat]
            )
        if np.any(growing_or_stable):
            target = interior * np.minimum(retained_fraction, retained_distance)
            with np.errstate(over="ignore", under="ignore"):
                relaxation = -np.expm1(-interval_h / self.tau_h)
            recovered = self._score + np.maximum(target - self._score, 0) * relaxation
            score[growing_or_stable] = np.minimum(
                score[growing_or_stable], recovered[growing_or_stable]
            )
        score = np.clip(score, 0, 1)
        saturation = np.where(gas, self.low + (self.high - self.low) * score, 0)
        diagnostics = {
            "gas_mask": gas,
            "core_mask": core,
            "persistent_core": persistent,
            "growing_or_stable_mask": growing_or_stable,
            "retreat_mask": retreat,
            "continuous_age_h": age_h.astype(np.float32),
            "interior_distance_m": distance.astype(np.float32),
            "retained_fraction": retained_fraction.astype(np.float32),
            "retained_distance": retained_distance.astype(np.float32),
            "retreat_retention": retreat_retention.astype(np.float32),
            "history_score": score.astype(np.float32),
            "elapsed_h": elapsed_h,
            "interval_h": interval_h,
            "history_reset": gap_reset,
        }
        self._fraction = fraction.copy()
        self._age_h = age_h
        self._distance_m = distance
        self._peak_fraction = peak_fraction
        self._peak_distance_m = peak_distance
        self._score = score
        self._elapsed_h = elapsed_h
        return saturation.astype(np.float32), diagnostics
