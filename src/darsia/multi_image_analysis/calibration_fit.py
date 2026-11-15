"""Calibration fit across runs: the gain of the dissolved CO2's signal and its dilution.

The run statistics give, per frame and sand, a histogram of the relative signal of
the dissolved CO2 (its signal on the concentration curve's range, from no
dissolved CO2 to saturation) with each interval's capacity for dissolved CO2
(:mod:`darsia.multi_image_analysis.run_statistics`); the gas statistics give the
frame's free gas, the water in its gas and its CO2 pore volume
(:mod:`darsia.multi_image_analysis.gas_statistics`). At a gain g and a dilution
factor f, a frame holds the CO2 mass

    f * sum over intervals of capacity * min(1, g * relative signal)
    + free gas + gas water [+ f * column estimate],

and its residual is that mass less the CO2 injected until the frame, relative to
the injected CO2. The dilution law compares the frame's CO2 pore volume with that
of the run's anchor frame, the frame nearest a set time: with x their ratio (at
least one), f = x ** -(beta + beta_growth * ln x), and f = 1 up to the anchor's
time. The calibration is the gain, beta and beta growth that fit the frames of
the fit runs:

1. the gain, on the frames selected at target times inside a window during the
   injection, without dilution; each run weighs the same, shared equally by its
   frames;
2. beta and beta growth together, at that gain, on the frames selected from a
   later target time; each run weighs the same, shared equally by the time bins
   it occupies and within a bin by its frames.

Each stage evaluates its objective (half the sum of the squared weighted
residuals) on its search grid, the gain on a logarithmic one and every pair of
the two dilution grids, and refines the lowest value by SciPy's bounded
trust-region least squares (TRF, two-point Jacobian, with the tolerance and the
limit of evaluations of the settings). The residuals, the tables and the
parameters they read are in the dtype of the statistics; the solver works in
double precision.

Every run given, fitted or not, has its anchor frame from its own frames, and
:func:`mass_calibrations` gives each run the parameters the mass maps read;
:func:`dilution_factor` is the dilution law both read.

Leaving one run out fits the frames of all other runs and predicts the frames of
the run left out. A sand without a measured Swi may take the Swi that makes the
frames' free gas agree with their run's own overflow measurement
(:func:`fit_swi`); the fit applies it to the frames whose free gas did not
follow an overflow volume.

Every setting is a field of :class:`CalibrationFitSettings`.

"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from darsia.multi_image_analysis.gas_statistics import GasStatistics, SandGas
from darsia.multi_image_analysis.run_statistics import (
    RunStatisticsSettings,
    SandHistogram,
)

_LOG_GAIN_BOUND = 700.0
"""Bound of the logarithm of the gain in the solver, against an overflow of its
exponential."""
_SWI_CONDITION = 100.0
"""Above this condition number of the column-scaled pore volumes the frames cannot
tell the sands' gas apart, and no Swi is fitted."""
STATISTICS = "statistics.json"
"""The file of a run's frames as the fit reads them (:func:`write_frames`)."""
CALIBRATION = "calibration.json"
"""The file of a stored calibration (:func:`write_calibration`)."""


@dataclass(frozen=True)
class CalibrationFitSettings:
    """Settings of the calibration fit; one configuration table."""

    gain_window_h: Sequence[float]
    """First and last target time [h] of the frames that fit the gain."""
    dilution_start_h: float
    """Target time [h] from which the frames fit the dilution."""
    dilution_bins_h: Sequence[float]
    """Edges [h] of the time bins that share a run's weight in the dilution fit,
    increasing; the first bin starts at the dilution start, the last has no end."""
    anchor_h: float
    """Time [h] of each run's anchor frame, the frame nearest it."""
    anchor_tolerance_h: float
    """Largest distance [h] of the anchor frame from that time."""
    gain_grid: Sequence[Sequence[float]]
    """Search grid of the gain: segments [first, last, points], evenly spaced in
    the logarithm; a single value is a segment of one point."""
    beta_grid: Sequence[Sequence[float]]
    """Search grid of beta: segments [first, last, points], evenly spaced."""
    beta_growth_grid: Sequence[Sequence[float]]
    """Search grid of beta growth: segments [first, last, points], evenly spaced."""
    tolerance: float
    """Function, step and gradient tolerance of the least-squares solver."""
    evaluations: int
    """Most residual evaluations of one least-squares solve."""
    swi_limit: float
    """A fitted gas share this close to 0 or 1 leaves the sand's Swi as assumed."""


@dataclass(frozen=True)
class ColumnEstimate:
    """CO2 pixels without a measurement of their own, read from the pixels above.

    Each such pixel takes the relative signal of the nearest measured CO2 pixel
    above it in its column; pixels of equal signal and solubility ratio are summed.

    """

    relative: np.ndarray
    """The relative signal taken from above."""
    ratio: np.ndarray
    """The solubility there over the pixel's own."""
    weight: np.ndarray
    """The pixels' summed capacity for dissolved CO2 [g]."""


@dataclass(frozen=True)
class CalibrationFrame:
    """One frame of a run, as the calibration fit reads it."""

    run: str
    """The frame's run."""
    index: int
    """The frame's index in its run."""
    elapsed_h: float
    """Time [h] since the injection start."""
    target_h: float | None
    """The target time [h] the frame was selected for; None for a frame that can
    only be an anchor."""
    injected_g: float
    """CO2 injected until the frame [g]."""
    dissolved: Mapping[int, SandHistogram]
    """Run statistics of the dissolved CO2 per sand; the weights are capacities [g]."""
    gas: GasStatistics
    """Gas statistics."""
    column: ColumnEstimate | None
    """CO2 pixels without a measurement of their own, or None without any."""
    overflow_m3: float | None
    """Free-gas volume [m3] the frame's gas allocation held, or None for the
    model's gas."""
    measured: bool
    """Whether that volume is the run's own overflow measurement."""


@dataclass(frozen=True)
class DilutionAnchor:
    """A run's anchor frame of the dilution law."""

    elapsed_h: float
    """Time [h] since the injection start."""
    pore_volume_m3: float
    """CO2 pore volume."""
    index: int
    """The frame's index in its run."""


@dataclass(frozen=True)
class FrameDecision:
    """Whether a frame selected at a target time of a stage entered the fit."""

    run: str
    index: int
    target_h: float
    stage: str
    """"gain" or "dilution"."""
    used: bool
    reasons: tuple[str, ...]
    """Why the frame is left out: "leave_one_run_out" (its run is held out) or
    "no_positive_dose" (no CO2 injected yet)."""


@dataclass(frozen=True)
class Calibration:
    """The fitted calibration.

    The dissolved CO2 of a pixel is the share min(1, gain * relative signal) of
    its water's capacity, times the frame's dilution factor (with beta and beta
    growth, at its run's anchor). Sands in ``swi`` take that Swi.

    """

    gain: float
    beta: float
    beta_growth: float
    anchors: dict[str, DilutionAnchor]
    """The anchor frame of each run given, fitted or not."""
    swi: dict[int, float]
    """Swi fitted for sands without a measured value; empty without such a sand."""
    runs: tuple[str, ...]
    """The runs fitted."""
    held_out: str | None
    """The run left out, or None."""
    decisions: tuple[FrameDecision, ...]
    """The frames selected for a stage, used or left out."""


@dataclass(frozen=True)
class MassCalibration:
    """Calibration parameters the mass maps read, for one run."""

    gain: float
    """Gain on the read signal relative to the concentration curve."""
    zero_signal: float
    """Signal of no dissolved CO2: the bottom of the concentration curve."""
    saturation_signal: float
    """Signal of saturated dissolved CO2: the top of the concentration curve."""
    line_share: float
    """Share of the straight line in the concentration curve; the curve is a
    straight line (one), its titration part is not part of this method."""
    beta: float
    """Exponent of the dilution after the anchor."""
    beta_growth: float
    """Growth of the dilution exponent with the logarithm of the pore-volume ratio."""
    anchor_h: float
    """Time [h] of the run's anchor, up to which nothing is diluted."""
    anchor_pore_volume_m3: float
    """Pore volume [m3] of the run's CO2 at the anchor."""

    def __post_init__(self) -> None:
        """Refuse what would otherwise change the mass silently: a curve with a
        titration part, and an anchor without a finite time and a finite positive
        pore volume."""
        if self.line_share != 1:
            raise ValueError("Only a straight concentration curve (line_share 1)")
        anchor = (self.anchor_h, self.anchor_pore_volume_m3)
        if not np.isfinite(anchor).all() or self.anchor_pore_volume_m3 <= 0:
            raise ValueError("The dilution anchor needs a positive pore volume")


@dataclass(frozen=True)
class Prediction:
    """A frame's CO2 mass at a calibration."""

    run: str
    index: int
    elapsed_h: float
    target_h: float | None
    mass_g: float
    """CO2 mass."""
    free_gas_g: float
    """Free-gas mass."""
    injected_g: float
    """CO2 injected until the frame."""
    residual: float | None
    """Mass less the injected CO2, relative to the injected CO2; None without any."""
    factor: float
    """Dilution factor."""


def dilution_factor(
    elapsed_h: float,
    pore_volume_m3: float,
    anchor_h: float,
    anchor_pore_volume_m3: float,
    beta: float,
    beta_growth: float,
    dtype: type,
) -> float:
    """Dilution factor of the dissolved CO2 in a frame of a run.

    One up to the run's anchor. After it, with x the pore volume of the frame's
    CO2 over the anchor's, at least one, the factor is
    ``x ** -(beta + beta_growth * ln x)``, at most one. It is computed in a
    floating dtype, as our production computed it in its working precision; the
    calibration fit and the mass maps read it alike.

    Parameters
    ----------
    elapsed_h : float
        Time of the frame [h], on the anchor's clock.
    pore_volume_m3 : float
        Pore volume of the frame's CO2.
    anchor_h, anchor_pore_volume_m3 : float
        Time [h] and CO2 pore volume of the run's anchor, e.g. of a
        :class:`DilutionAnchor` or a :class:`MassCalibration`.
    beta, beta_growth : float
        Exponent of the dilution after the anchor, and its growth with ln x.
    dtype : type
        Floating dtype of the arithmetic, that of the statistics or the maps.

    Returns
    -------
    float
        The factor.

    """
    if elapsed_h <= anchor_h:
        return 1.0
    dtype = np.dtype(dtype).type
    one = dtype(1)
    ratio = max(dtype(pore_volume_m3) / dtype(anchor_pore_volume_m3), one)
    exponent = dtype(beta)
    if beta_growth:
        exponent = exponent + dtype(beta_growth) * np.log(ratio)
    return float(min(one, ratio**-exponent))


def _anchor(frames, settings):
    """A run's anchor: its frame nearest the anchor time (ties: the lower index).

    The frame needs a finite time within the tolerance and a finite, positive CO2
    pore volume, as in our production; otherwise the dilution factor would be one
    or zero without notice.

    """
    frame = min(frames, key=lambda f: (abs(f.elapsed_h - settings.anchor_h), f.index))
    volume = frame.gas.co2_pore_volume_m3
    if not (
        abs(frame.elapsed_h - settings.anchor_h) <= settings.anchor_tolerance_h
        and np.isfinite(volume)
        and volume > 0
    ):
        raise ValueError(f"No dilution anchor for run {frame.run}")
    return DilutionAnchor(frame.elapsed_h, volume, frame.index)


def fit_swi(
    frames: Sequence[CalibrationFrame],
    swi: Mapping[int, float],
    measured: Collection[int],
    settings: CalibrationFitSettings,
) -> dict[int, float]:
    """Swi of the sands without a measured value, from the overflow measurements.

    In each frame with its run's own overflow measurement, the free gas of the
    pixels labelled gas fills the share 1 - Swi of their pore volume. The shares
    of the sands without a measured Swi are fitted to the measured volumes by
    bounded linear least squares, the other sands at their Swi. A sand without gas
    in those frames, a set of sands the frames cannot tell apart (by rank, or a
    condition number above 100), and a share within ``swi_limit`` of 0 or 1 keep
    the assumed Swi.

    Parameters
    ----------
    frames : sequence of CalibrationFrame
        The frames.
    swi : mapping of int to float
        Each sand's Swi, measured or assumed.
    measured : collection of int
        The sands whose Swi is measured.
    settings : CalibrationFitSettings
        Settings of the fit, its Swi limit.

    Returns
    -------
    dict of int to float
        The fitted Swi of each sand it places.

    """
    open_sands = sorted(sand for sand in swi if sand not in measured)
    rows = [f for f in frames if f.measured and f.overflow_m3 is not None]
    rows = [f for f in rows if f.gas.sands]

    def pores(frame, sand):
        cell = frame.gas.sands.get(sand)
        return 0.0 if cell is None else cell.pore_volume_m3

    used = [s for s in open_sands if any(pores(f, s) > 0 for f in rows)]
    if not used:
        return {}
    matrix = np.array([[pores(f, s) for s in used] for f in rows], dtype=np.float64)
    scaled = matrix / matrix.max(axis=0)
    if (
        np.linalg.matrix_rank(scaled) < len(used)
        or np.linalg.cond(scaled) > _SWI_CONDITION
    ):
        return {}
    known = [
        sum(pores(f, s) * (1 - value) for s, value in swi.items() if s not in used)
        for f in rows
    ]
    wanted = np.array([f.overflow_m3 - k for f, k in zip(rows, known)], np.float64)
    shares = lsq_linear(matrix, wanted, bounds=(0.0, 1.0)).x
    return {
        sand: float(1 - share)
        for sand, share in zip(used, shares)
        if settings.swi_limit < share < 1 - settings.swi_limit
    }


def _apply_swi(frames, swi, fitted):
    """The frames with each fitted Swi in place of the assumed one.

    A frame whose free gas followed an overflow volume holds that volume and
    stays; in the others the gas fills 1 - Swi of the gas pixels' pore volume,
    and water the rest.

    """
    changed = []
    for frame in frames:
        if fitted and frame.overflow_m3 is None:
            free, water = frame.gas.free_gas_g, frame.gas.gas_water_g
            for sand, value in fitted.items():
                cell = frame.gas.sands.get(sand)
                if cell is not None:
                    shift = swi[sand] - value
                    free += shift * cell.gas_capacity_g
                    water += -shift * cell.water_capacity_g
            gas = dataclasses.replace(frame.gas, free_gas_g=free, gas_water_g=water)
            frame = dataclasses.replace(frame, gas=gas)
        changed.append(frame)
    return changed


def _stage(target_h, settings):
    """The stage of a frame selected at a target time, or None."""
    if target_h is None:
        return None
    first, last = settings.gain_window_h
    if first <= target_h <= last:
        return "gain"
    if target_h >= settings.dilution_start_h:
        return "dilution"
    return None


def _selection(frames, settings, held_out):
    """The frames of each stage and the decision on every frame of a stage."""
    stages, decisions = {"gain": [], "dilution": []}, []
    for frame in frames:
        stage = _stage(frame.target_h, settings)
        if stage is None:
            continue
        reasons = []
        if frame.run == held_out:
            reasons.append("leave_one_run_out")
        if not (np.isfinite(frame.injected_g) and frame.injected_g > 0):
            reasons.append("no_positive_dose")
        if not reasons:
            stages[stage].append(frame)
        decisions.append(
            FrameDecision(
                frame.run,
                frame.index,
                frame.target_h,
                stage,
                not reasons,
                tuple(reasons),
            )
        )
    return stages, decisions


def _weights(frames, settings, dtype, *, binned):
    """Each run the same weight, shared by its time bins (if binned) and frames."""
    scalar = np.dtype(dtype).type
    runs = Counter(frame.run for frame in frames)
    if not binned:
        return np.array(
            [scalar(1) / np.sqrt(scalar(len(runs) * runs[f.run])) for f in frames],
            dtype,
        )
    edges = settings.dilution_bins_h
    bins = [(f.run, int(np.searchsorted(edges, f.target_h))) for f in frames]
    count = Counter(bins)
    occupied = defaultdict(set)
    for run, time_bin in count:
        occupied[run].add(time_bin)
    return np.array(
        [
            scalar(1)
            / np.sqrt(scalar(len(runs) * len(occupied[run]) * count[(run, time_bin)]))
            for run, time_bin in bins
        ],
        dtype,
    )


def _tables(frame, gain, dtype):
    """A frame's dissolved and column coefficients at a gain, and its masses."""
    scalar = np.dtype(dtype).type
    dissolved = None
    for histogram in frame.dissolved.values():
        value = scalar(
            np.dot(histogram.weight, np.clip(histogram.relative * gain, 0, 1))
        )
        dissolved = value if dissolved is None else dissolved + value
    column = frame.column
    if column is not None:
        share = column.ratio * np.clip(column.relative * gain, 0, 1)
        column = scalar(np.sum(column.weight * share))
    return (
        dissolved,
        column,
        scalar(frame.gas.free_gas_g),
        scalar(frame.gas.gas_water_g),
        scalar(frame.injected_g),
    )


def _mass(tables, factor):
    """A frame's CO2 mass at a dilution factor, from its tables."""
    dissolved, column, gas, water, _ = tables
    mass = dissolved * factor + gas + water
    return mass if column is None else mass + column * factor


def _factor(frame, anchors, beta, beta_growth, dtype):
    """A frame's dilution factor at its run's anchor, in ``dtype``."""
    anchor = anchors[frame.run]
    factor = dilution_factor(
        frame.elapsed_h,
        frame.gas.co2_pore_volume_m3,
        anchor.elapsed_h,
        anchor.pore_volume_m3,
        beta,
        beta_growth,
        dtype,
    )
    return np.dtype(dtype).type(factor)


def _residuals(frames, weights, anchors, dtype, *, gain=None):
    """The weighted residuals of the frames.

    Without a gain, of the logarithm of the gain without dilution; with it, of
    beta and beta growth at that gain.

    """
    scalar = np.dtype(dtype).type
    fixed = None if gain is None else [_tables(f, gain, dtype) for f in frames]

    def residuals(vector):
        if fixed is None:
            current = scalar(np.exp(scalar(vector[0])))
            tables = [_tables(f, current, dtype) for f in frames]
            beta, growth = scalar(0), scalar(0)
        else:
            tables = fixed
            beta, growth = scalar(vector[0]), scalar(vector[1])
        values = []
        for frame, table in zip(frames, tables):
            mass = _mass(table, _factor(frame, anchors, beta, growth, dtype))
            dose = table[4]
            values.append((mass - dose) / dose)
        return weights * np.array(values, dtype)

    return residuals


def _grid(segments, logarithmic):
    """The distinct values of a search grid's segments, increasing."""
    space = np.geomspace if logarithmic else np.linspace
    return np.unique(np.concatenate([space(a, b, int(n)) for a, b, n in segments]))


def _objective(residuals, vector):
    """Half the sum of the squared weighted residuals."""
    value = residuals(vector)
    return 0.5 * float(np.dot(value, value))


def _solve(residuals, start, bounds, settings):
    """The bounded least-squares refinement from a start."""
    result = least_squares(
        residuals,
        start,
        bounds=bounds,
        max_nfev=settings.evaluations,
        ftol=settings.tolerance,
        xtol=settings.tolerance,
        gtol=settings.tolerance,
    )
    if not (result.success and np.isfinite(result.cost)):
        raise ValueError("The least-squares fit of the calibration failed")
    return result.x


def _fit_gain(residuals, settings, dtype):
    """The gain: the lowest grid value, refined."""
    scalar = np.dtype(dtype).type
    coordinates = np.log(_grid(settings.gain_grid, logarithmic=True))
    objectives = np.array([_objective(residuals, [c]) for c in coordinates])
    best = int(np.argmin(np.where(np.isfinite(objectives), objectives, np.inf)))
    x = _solve(
        residuals,
        [float(coordinates[best])],
        ([-_LOG_GAIN_BOUND], [_LOG_GAIN_BOUND]),
        settings,
    )
    return scalar(np.exp(scalar(x[0])))


def _fit_dilution(residuals, settings, dtype):
    """Beta and beta growth: the lowest pair of their grids, refined."""
    scalar = np.dtype(dtype).type
    betas = _grid(settings.beta_grid, logarithmic=False)
    growths = _grid(settings.beta_growth_grid, logarithmic=False)
    objectives = np.array(
        [[_objective(residuals, [b, g]) for g in growths] for b in betas]
    )
    finite = np.where(np.isfinite(objectives), objectives, np.inf)
    i, j = np.unravel_index(int(np.argmin(finite)), objectives.shape)
    x = _solve(
        residuals,
        [float(betas[i]), float(growths[j])],
        ([0.0, 0.0], [np.inf, np.inf]),
        settings,
    )
    return scalar(x[0]), scalar(x[1])


def _fit(frames, fit_runs, settings, swi, measured, held_out):
    """The calibration without the held-out run, and the frames it applies to."""
    identities = [(f.run, f.index) for f in frames]
    if len(set(identities)) != len(identities):
        raise ValueError("A frame of a run is given twice")
    if held_out is not None and held_out not in fit_runs:
        raise ValueError(f"The held-out run {held_out} is not a fit run")
    runs = tuple(run for run in fit_runs if run != held_out)
    fit = [f for f in frames if f.run in fit_runs]
    dtype = next(iter(fit[0].dissolved.values())).weight.dtype
    fitted = fit_swi([f for f in fit if f.run != held_out], swi, measured, settings)
    frames = _apply_swi(frames, swi, fitted)
    by_run = defaultdict(list)
    for frame in frames:
        by_run[frame.run].append(frame)
    anchors = {run: _anchor(values, settings) for run, values in by_run.items()}
    fit = [f for f in frames if f.run in fit_runs]
    stages, decisions = _selection(fit, settings, held_out)
    if not stages["gain"] or not stages["dilution"]:
        raise ValueError("A stage of the calibration fit has no frames")
    first = stages["gain"]
    gain = _fit_gain(
        _residuals(
            first, _weights(first, settings, dtype, binned=False), anchors, dtype
        ),
        settings,
        dtype,
    )
    second = stages["dilution"]
    beta, growth = _fit_dilution(
        _residuals(
            second,
            _weights(second, settings, dtype, binned=True),
            anchors,
            dtype,
            gain=gain,
        ),
        settings,
        dtype,
    )
    calibration = Calibration(
        gain=float(gain),
        beta=float(beta),
        beta_growth=float(growth),
        anchors=anchors,
        swi=fitted,
        runs=runs,
        held_out=held_out,
        decisions=tuple(decisions),
    )
    return calibration, frames


def fit_calibration(
    frames: Sequence[CalibrationFrame],
    fit_runs: Sequence[str],
    settings: CalibrationFitSettings,
    swi: Mapping[int, float],
    measured_swi: Collection[int],
) -> Calibration:
    """The calibration fitted to the frames of the fit runs.

    Parameters
    ----------
    frames : sequence of CalibrationFrame
        The frames of the fit runs and of the other runs whose mass is wanted
        (e.g. held-out runs), each run's anchor frame among them; the other runs
        give only their anchors.
    fit_runs : sequence of str
        The runs to fit.
    settings : CalibrationFitSettings
        Settings of the fit.
    swi : mapping of int to float
        Each sand's Swi, measured or assumed, as the gas statistics were computed.
    measured_swi : collection of int
        The sands whose Swi is measured; the others may be fitted (:func:`fit_swi`).

    Returns
    -------
    Calibration
        The calibration.

    """
    return _fit(frames, fit_runs, settings, swi, measured_swi, None)[0]


def _predictions(frames, calibration, run):
    """The CO2 mass of a run's frames at a calibration (its Swi in place), in order."""
    values = sorted((f for f in frames if f.run == run), key=lambda f: f.index)
    dtype = next(iter(values[0].dissolved.values())).weight.dtype
    scalar = np.dtype(dtype).type
    gain, beta, growth = (
        scalar(calibration.gain),
        scalar(calibration.beta),
        scalar(calibration.beta_growth),
    )
    predictions = []
    for frame in values:
        tables = _tables(frame, gain, dtype)
        factor = _factor(frame, calibration.anchors, beta, growth, dtype)
        mass = _mass(tables, factor)
        dose = tables[4]
        predictions.append(
            Prediction(
                run=frame.run,
                index=frame.index,
                elapsed_h=frame.elapsed_h,
                target_h=frame.target_h,
                mass_g=float(mass),
                free_gas_g=frame.gas.free_gas_g,
                injected_g=frame.injected_g,
                residual=float((mass - dose) / dose) if dose > 0 else None,
                factor=float(factor),
            )
        )
    return predictions


def calibration_fold(
    frames: Sequence[CalibrationFrame],
    fit_runs: Sequence[str],
    settings: CalibrationFitSettings,
    swi: Mapping[int, float],
    measured_swi: Collection[int],
    held_out: str,
) -> tuple[Calibration, list[Prediction]]:
    """One fold of leaving one run out: the fit without a run, and its predictions.

    Parameters
    ----------
    frames : sequence of CalibrationFrame
        The frames of the fold, as for :func:`fit_calibration`; a run whose free
        gas was estimated from the held-out run's overflow takes it from its next
        donor.
    fit_runs : sequence of str
        The runs to fit.
    settings : CalibrationFitSettings
        Settings of the fit.
    swi : mapping of int to float
        Each sand's Swi, measured or assumed, as the gas statistics were computed.
    measured_swi : collection of int
        The sands whose Swi is measured; the others may be fitted (:func:`fit_swi`).
    held_out : str
        The fit run left out of the fit.

    Returns
    -------
    tuple of Calibration and list of Prediction
        The calibration fitted without the run, and the run's predicted frames.

    """
    calibration, applied = _fit(frames, fit_runs, settings, swi, measured_swi, held_out)
    return calibration, _predictions(applied, calibration, held_out)


def leave_one_run_out(
    folds: Mapping[str, Sequence[CalibrationFrame]],
    fit_runs: Sequence[str],
    settings: CalibrationFitSettings,
    swi: Mapping[int, float],
    measured_swi: Collection[int],
) -> dict[str, tuple[Calibration, list[Prediction]]]:
    """Every fold of leaving one run out (:func:`calibration_fold`).

    ``folds`` holds the frames of each held-out run's fold. The folds are
    independent of each other.

    """
    return {
        run: calibration_fold(frames, fit_runs, settings, swi, measured_swi, run)
        for run, frames in folds.items()
    }


def mass_calibrations(
    calibration: Calibration, statistics: RunStatisticsSettings
) -> dict[str, MassCalibration]:
    """The parameters the mass maps read, for every run with an anchor.

    The concentration curve is that of the run statistics the calibration was
    fitted on (``statistics``): a straight line from ``zero_signal`` to
    ``saturation_signal``.

    """
    return {
        run: MassCalibration(
            gain=calibration.gain,
            zero_signal=statistics.zero_signal,
            saturation_signal=statistics.saturation_signal,
            line_share=1.0,
            beta=calibration.beta,
            beta_growth=calibration.beta_growth,
            anchor_h=anchor.elapsed_h,
            anchor_pore_volume_m3=anchor.pore_volume_m3,
        )
        for run, anchor in calibration.anchors.items()
    }


def _write_json(folder: Path, name: str, value) -> None:
    """Write a stored result's JSON file; NaN is refused."""
    text = json.dumps(value, indent=1, allow_nan=False)
    (Path(folder) / name).write_text(text, encoding="utf-8")


def _array(values, dtype) -> list | None:
    """An array as a list of plain numbers, or None."""
    return None if values is None else np.asarray(values, dtype).tolist()


def _rows(frames: Sequence[CalibrationFrame], name: str) -> list[dict]:
    """A run's frames as plain values, the arrays in the dtype ``name``."""
    rows = []
    for frame in frames:
        row = dataclasses.asdict(frame)
        row["dissolved"] = {
            str(sand): [_array(h.relative, name), _array(h.weight, name)]
            for sand, h in frame.dissolved.items()
        }
        if frame.column is not None:
            row["column"] = {
                key: _array(getattr(frame.column, key), name)
                for key in ("relative", "ratio", "weight")
            }
        rows.append(row)
    return rows


def write_frames(
    folder: Path,
    run: str,
    frames: Sequence[CalibrationFrame],
    dtype,
    held_out: Mapping[str, Sequence[CalibrationFrame]] | None = None,
) -> None:
    """Store a run's frames as the fit reads them, in ``folder`` (:data:`STATISTICS`).

    Parameters
    ----------
    folder : Path
        The result's folder.
    run : str
        The run.
    frames : sequence of CalibrationFrame
        Its frames.
    dtype : dtype
        The dtype of the statistics' arrays, which :func:`read_frames` restores.
    held_out : mapping of str to sequence of CalibrationFrame, optional
        For a run held out of a calibration fold whose estimated free gas the
        frames took, the frames without it, which that fold reads.

    """
    name = np.dtype(dtype).name
    record = {
        "run": run,
        "dtype": name,
        "frames": _rows(frames, name),
        "held_out": {
            other: _rows(alternative, name)
            for other, alternative in (held_out or {}).items()
        },
    }
    _write_json(folder, STATISTICS, record)


def read_frames(folder: Path, held_out: str | None = None) -> list[CalibrationFrame]:
    """A run's stored frames as the fit reads them (:func:`write_frames`); with
    ``held_out``, those without that run's estimate where they are stored, as a
    fold leaving it out reads them."""
    record = json.loads((Path(folder) / STATISTICS).read_text(encoding="utf-8"))
    dtype = np.dtype(record["dtype"]).type
    rows = record["held_out"].get(held_out, record["frames"])
    frames = []
    for row in rows:
        gas = dict(row["gas"])
        gas["sands"] = {int(k): SandGas(**v) for k, v in gas["sands"].items()}
        column = row["column"]
        if column is not None:
            column = ColumnEstimate(
                **{key: np.asarray(value, dtype) for key, value in column.items()}
            )
        frames.append(
            CalibrationFrame(
                **{
                    **row,
                    "dissolved": {
                        int(sand): SandHistogram(
                            np.asarray(relative, dtype), np.asarray(weight, dtype)
                        )
                        for sand, (relative, weight) in row["dissolved"].items()
                    },
                    "gas": GasStatistics(**gas),
                    "column": column,
                }
            )
        )
    return frames


def write_calibration(
    folder: Path,
    calibration: Calibration,
    masses: Mapping[str, MassCalibration],
    predictions: Sequence[Prediction] = (),
) -> None:
    """Store a calibration in ``folder`` (:data:`CALIBRATION`): the fit's record,
    the parameters of every run the mass maps read (:func:`mass_calibrations`) and,
    for a fold, the held-out run's predictions."""
    record = {
        "calibration": dataclasses.asdict(calibration),
        "mass": {run: dataclasses.asdict(value) for run, value in masses.items()},
        "predictions": [dataclasses.asdict(p) for p in predictions],
    }
    _write_json(folder, CALIBRATION, record)


def read_calibration(folder: Path) -> dict:
    """A stored calibration's record as written (:func:`write_calibration`)."""
    return json.loads((Path(folder) / CALIBRATION).read_text(encoding="utf-8"))


def read_mass_calibrations(folder: Path) -> dict[str, MassCalibration]:
    """The stored calibration's parameters of each run, as the mass maps read them."""
    record = read_calibration(folder)
    return {run: MassCalibration(**v) for run, v in record["mass"].items()}
