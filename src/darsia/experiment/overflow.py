"""Overflow reference: the free-gas volume of a run's frames from displaced water.

Injected CO2 displaces water from the porous layers, and the overflow carries
the displaced water to a scale. The overflow measures that displaced water,
never CO2 leaving the rig. How the weight shows it depends on the rig, and the
configuration chooses the variant: an overflow fed from a tank, where the
displaced water is the weight above the feed (:mod:`overflow_feed`), or one
circulating its water, where it is the absolute weight
(:mod:`overflow_circulation`). Either way the displaced volume of a run counts
from its baseline, the readings before the injection, at one gram of water per
mL (:func:`displaced_volume`), and the free-gas volume of a frame is that
volume at the frame's time, at least zero (:func:`measured_reference`). A
frame without usable readings of its own takes the free-gas share of the
injected CO2 mass from the most similar run with readings
(:func:`ranked_donors`, :func:`estimated_reference`).

Three optional rules keep the scale's noise out of the free gas
(:func:`displaced_volume`): a single reading after the volume's peak that
leaves both its neighbours on one side by more than ``outlier_factor`` times
the run's reading noise is left out, the volume between its neighbours taking
its place; after the peak, a volume within ``tail_noise_factor`` times that
noise of zero gives no reference, so the model's gas stands; and with
``end_at_supply_stop`` a fed overflow gives none after its feed stops, where
the weight no longer shows displaced water.

Every setting is a field of :class:`OverflowSettings` or of its variant's
settings.

"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from darsia.experiment.overflow_circulation import CirculationSettings
from darsia.experiment.overflow_evaporation import EvaporationRate, EvaporationSettings
from darsia.experiment.overflow_feed import FeedSettings
from darsia.utils.results import write_json

FRAME_VOLUMES = "overflow.json"
"""The file of a run's stored free-gas volume per frame (:func:`write_frame_volumes`)."""
OVERFLOW_RUN = "displaced.json"
"""The file of a run's stored displaced volume as a donor (:func:`write_overflow_run`)."""
MEASURED = "measured"
"""The source of a frame's volume from its run's own readings (:class:`FrameVolume`)."""


class OverflowVariant(Protocol):
    """How a rig's scale weight shows the displaced water: the settings of an
    overflow variant, e.g. :class:`darsia.FeedSettings`.

    A new variant is one module with a settings class of these two members.

    """

    minimum_readings: int
    """Fewest readings a run needs."""

    def displaced_ml(
        self,
        times_h: np.ndarray,
        weights_g: np.ndarray,
        corrected_g: np.ndarray,
        baseline: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Displaced water [mL] at each reading, and why the readings are not
        usable (empty when they are), from the readings' times [h since the
        injection start], their weights [g] as read and with any evaporated
        water added back, and the baseline readings."""


@dataclass(frozen=True)
class OverflowSettings:
    """Settings of the overflow reference; the ``[overflow]`` table.

    ``variant`` chooses the variant, and the table of that name holds its
    settings; the configuration's loader makes the nested tables their settings
    (:func:`darsia.presets.workflows.config.utils.settings_from_table`).

    """

    statuses: Sequence[str]
    """Review statuses of the scale readings that are used."""
    recipe_tolerance: Mapping[str, float]
    """Recipe entries a donor may differ in, by this fraction of the larger value."""
    variant: Literal["feed", "circulation"]
    """``"feed"`` for an overflow fed from a tank, ``"circulation"`` for one
    circulating its water."""
    feed: FeedSettings | None = None
    """The feed's settings, given with the feed variant only."""
    circulation: CirculationSettings | None = None
    """The circulation's settings, given with the circulation variant only."""
    baseline_window_h: float | None = None
    """Hours before the injection whose readings are the baseline, for the
    circulation's reference and the evaporation proposal; without it, all
    readings before the injection."""
    evaporation: EvaporationSettings | None = None
    """The evaporation correction; without it no run is corrected."""
    outlier_factor: float | None = None
    """A single reading after the volume's peak that leaves both its neighbours
    on one side by more than this many times the run's reading noise is left
    out; without it, every reading is used."""
    tail_noise_factor: float | None = None
    """After the volume's peak, a volume within this many times the run's
    reading noise of zero gives no reference; without it, every volume does."""
    end_at_supply_stop: bool = False
    """Whether a fed overflow gives no reference after its feed stops."""

    def __post_init__(self) -> None:
        """Check the variant, its table and the noise rules."""
        for name in ("outlier_factor", "tail_noise_factor"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(f"The overflow's {name} must be positive.")
        needs_noise = (
            self.outlier_factor is not None
            or self.tail_noise_factor is not None
            or self.end_at_supply_stop
        )
        if needs_noise and self.variant != "feed":
            raise ValueError(
                "The overflow's noise rules need the feed variant's reading noise"
                " and stop."
            )
        if self.variant not in ("feed", "circulation"):
            raise ValueError(f"The overflow variant {self.variant!r} is unknown.")
        for name in ("feed", "circulation"):
            given = getattr(self, name) is not None
            if name == self.variant and not given:
                raise ValueError(
                    f"The {name} variant needs its [overflow.{name}] table."
                )
            if name != self.variant and given:
                raise ValueError(
                    f"The [overflow.{name}] table does not match the {self.variant}"
                    " variant."
                )

    @property
    def variant_settings(self) -> OverflowVariant:
        """The settings of the chosen variant."""
        return getattr(self, self.variant)


@dataclass(frozen=True)
class DisplacedVolume:
    """The displaced water volume of a run at its scale readings."""

    times_h: np.ndarray
    """Times of the readings [h since the injection start], increasing."""
    volume_ml: np.ndarray | None
    """Displaced water volume [mL] at each reading; None without a reference to
    count from."""
    reasons: tuple[str, ...]
    """Why the readings are not usable; empty when they are."""
    baseline: np.ndarray
    """Which readings are the baseline: before the injection, in the window."""
    excluded: tuple[int, ...] = ()
    """Indices of the readings left out as single outliers; their volumes are
    those between their neighbours."""
    end_h: float | None = None
    """No reference after this time [h], the fed overflow's stop; None without
    an end."""
    peak_h: float | None = None
    """Time [h] of the largest volume, from which the tail rule applies."""
    tail_floor_ml: float | None = None
    """After ``peak_h``, a volume within this [mL] of zero gives no reference;
    None without the rule."""

    def astype(self, dtype) -> DisplacedVolume:
        """The displaced volume with its times and volumes in a dtype, e.g. the run
        grid's, in which :func:`measured_reference` and
        :func:`estimated_reference` then interpolate it, as our production
        interpolated its reviewed volumes in the job's precision."""
        volume = None if self.volume_ml is None else self.volume_ml.astype(dtype)
        return dataclasses.replace(
            self, times_h=self.times_h.astype(dtype), volume_ml=volume
        )


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
    times_h: np.ndarray,
    weights_g: np.ndarray,
    settings: OverflowSettings,
    evaporation_g_h: float | None = None,
) -> DisplacedVolume:
    """Displaced water volume of a run at its scale readings.

    The baseline readings are those before the injection, within
    ``baseline_window_h`` of it if that is given; the window scopes the
    circulation's reference and the evaporation proposal, while a feed counts
    from the last reading before the injection and checks all readings before
    it. The readings are usable with at least the variant's
    ``minimum_readings`` of them, one before the injection, and as the variant
    requires. With an evaporation rate, the water evaporated since the
    injection start is added back to the weights the volume is computed from:
    in a circulation the volume gains the rate times the time since the
    baseline, and in a feed, whose stop, late window and checks read the
    weights as read, a steady rate cancels exactly in the fitted feed (see
    :mod:`overflow_evaporation`). The fits are in float64.

    Parameters
    ----------
    times_h : array
        Times of the readings [h since the injection start], increasing.
    weights_g : array
        Scale weights [g].
    settings : OverflowSettings
        The overflow's variant and its settings.
    evaporation_g_h : float, optional
        The run's approved evaporation rate [g/h]; without one, no correction.
        A rate that is not finite is refused.

    Returns
    -------
    DisplacedVolume
        The volume at each reading and why the readings are not usable, if so.

    """
    if evaporation_g_h is not None and not np.isfinite(evaporation_g_h):
        raise ValueError("The evaporation rate is not finite.")
    t = np.asarray(times_h, np.float64)
    w = np.asarray(weights_g, np.float64)
    before = t < 0
    baseline = before.copy()
    if settings.baseline_window_h is not None:
        baseline &= t >= -settings.baseline_window_h
    variant = settings.variant_settings
    if t.size < variant.minimum_readings or not before.any():
        reasons = ("too few readings, or none before injection",)
        return DisplacedVolume(t, None, reasons, baseline)
    corrected = w if evaporation_g_h is None else w + evaporation_g_h * t
    volume, reasons = variant.displaced_ml(t, w, corrected, baseline)
    if volume is None or (
        settings.outlier_factor is None
        and settings.tail_noise_factor is None
        and not settings.end_at_supply_stop
    ):
        return DisplacedVolume(t, volume, reasons, baseline)
    stop = float(variant.supply_stop_h(t, w))
    noise = variant.reading_noise_ml(t, w)
    scope = (t >= 0) & (t <= stop)
    peak = int(np.argmax(np.where(scope, volume, -np.inf)))
    excluded = ()
    if settings.outlier_factor is not None:
        # The last readings before the stop follow the emptying tank, not the
        # feed: the check ends where the late window does.
        steady = stop - variant.late_window_h[1]
        volume, excluded = _without_outliers(
            t, volume, peak, steady, settings.outlier_factor * noise
        )
    return DisplacedVolume(
        t,
        volume,
        reasons,
        baseline,
        excluded=excluded,
        end_h=stop if settings.end_at_supply_stop else None,
        peak_h=float(t[peak]),
        tail_floor_ml=(
            None
            if settings.tail_noise_factor is None
            else settings.tail_noise_factor * noise
        ),
    )


def _without_outliers(
    times_h: np.ndarray,
    volume_ml: np.ndarray,
    peak: int,
    stop_h: float,
    limit_ml: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """The volumes with single outliers after the peak replaced, and their
    indices.

    A reading after the peak (index ``peak``) whose neighbours both lie within
    the feed, up to ``stop_h``, is an outlier where both neighbours lie on the
    same side of it and it leaves the line between them by more than
    ``limit_ml``. Its volume becomes that line's at its time. Before the peak
    the injection's own turns are real, and are kept.
    """
    t, v = times_h, volume_ml
    candidates = []
    for i in range(peak + 1, len(t) - 1):
        if t[i + 1] > stop_h:
            break
        up, down = v[i] - v[i - 1], v[i] - v[i + 1]
        if up * down <= 0:
            continue
        between = v[i - 1] + (t[i] - t[i - 1]) / (t[i + 1] - t[i - 1]) * (
            v[i + 1] - v[i - 1]
        )
        if abs(v[i] - between) > limit_ml:
            candidates.append((abs(v[i] - between), i, between))
    # A spike also makes its neighbours turn; the largest turn is the outlier,
    # and a reading next to one already left out is kept.
    cleaned, found = v.copy(), set()
    for _, i, between in sorted(candidates, reverse=True):
        if i - 1 not in found and i + 1 not in found:
            cleaned[i] = between
            found.add(i)
    return cleaned, tuple(sorted(found))


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


def _reference_at(volume: DisplacedVolume, elapsed_h: float):
    """Return the reading in mL and any explicit reason for withholding a reference.

    For a usable scale curve, a cutoff withholds the overflow target even when
    a donor has readings at the same time. A time outside the readings can use
    a donor until the cutoff.
    """
    if volume.end_h is not None and elapsed_h > volume.end_h:
        return None, "feed_stop"
    observed = _volume_at(volume, elapsed_h)
    if (
        observed is not None
        and volume.tail_floor_ml is not None
        and volume.peak_h is not None
        and elapsed_h > volume.peak_h
        and abs(float(observed)) <= volume.tail_floor_ml
    ):
        return None, "tail_noise"
    return observed, None


def _measured_m3(observed):
    return max(0.0, float(observed * 1e-6))


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
        the readings or where the volume's rules give none
        (:func:`displaced_volume`).

    """
    observed, _ = _reference_at(volume, elapsed_h)
    return None if observed is None else _measured_m3(observed)


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
    observed, _ = _reference_at(donor, elapsed_h)
    if observed is None or mass_g < 0:
        return None
    volume = _measured_m3(observed)
    if donor_mass_g > 0:
        share = volume * donor_density * 1000 / donor_mass_g
    elif volume == 0:
        share = 0.0
    else:
        return None
    if not 0 <= share <= 1:
        return None
    return share * mass_g / (density * 1000)


def interpolated_atmosphere(
    times_h: Sequence[float],
    pressure_bar: Sequence[float],
    temperature_c: Sequence[float],
    elapsed_h: float,
    dtype,
) -> tuple[float, float]:
    """A run's atmospheric pressure and temperature at a time, linear in the hours
    of its pressure-temperature record, with those hours in a dtype, e.g. the run
    grid's, as our production interpolated a run's atmosphere (``np.interp``);
    their upward gradients by the same rule.

    Parameters
    ----------
    times_h : sequence of float
        The record's times [h since the injection start], increasing.
    pressure_bar, temperature_c : sequence of float
        Its pressure [bar] and temperature [°C] at those times, or their upward
        gradients [bar/m] and [°C/m].
    elapsed_h : float
        The time [h since the injection start].
    dtype : dtype
        The dtype of the record's times.

    Returns
    -------
    tuple of float
        The pressure [bar] and temperature [°C], or their gradients.

    Raises
    ------
    ValueError
        If the time lies outside the record.

    """
    times = np.asarray(times_h, dtype)
    elapsed = float(elapsed_h)
    if not np.isfinite(elapsed) or not times[0] <= elapsed <= times[-1]:
        raise ValueError(f"{elapsed} h lies outside the pressure-temperature record.")
    return (
        float(np.interp(elapsed, times, pressure_bar)),
        float(np.interp(elapsed, times, temperature_c)),
    )


@dataclass(frozen=True)
class FrameVolume:
    """A frame's free-gas volume, and where it comes from."""

    volume_m3: float | None
    """The volume [m3]; None for the model's gas."""
    source: str | None
    """:data:`MEASURED` for the run's own readings, the donor run's name for an
    estimate from it, None without a volume."""


def frame_volumes(
    own: DisplacedVolume | None,
    hours: Sequence[float],
    donor: tuple[str, DisplacedVolume] | None = None,
    mass_g: Sequence[float] = (),
    density: Sequence[float] = (),
    donor_state: Callable[[float], tuple[float, float]] | None = None,
) -> list[FrameVolume]:
    """The free-gas volume of each frame of a run.

    A frame takes its run's own measured reference where the run's readings are
    usable and reach the frame's time (:func:`measured_reference`); else the
    donor's estimated reference (:func:`estimated_reference`), from the donor's
    injected CO2 mass and gas density at the frame's time; else none, the
    model's gas. A usable own-scale curve can explicitly leave the model's
    gas as the reference at its tail noise cutoff or after the feed stops.
    A run has one donor for frames that need it, which is never left for
    another outside its readings.

    Parameters
    ----------
    own : DisplacedVolume or None
        The run's own displaced volume, if it has readings.
    hours : sequence of float
        Each frame's time [h since the injection start].
    donor : tuple of str and DisplacedVolume, optional
        The donor's name and usable displaced volume.
    mass_g, density : sequence of float
        Each frame's injected CO2 mass [g] and gas density [kg/m3], for a donor.
    donor_state : callable, optional
        The donor's injected CO2 mass [g] and gas density [kg/m3] at a time [h
        since its injection start].

    Returns
    -------
    list of FrameVolume
        Each frame's volume and its source.

    """
    usable = own is not None and own.volume_ml is not None and not own.reasons
    found = []
    for k, elapsed_h in enumerate(hours):
        observed, cutoff = _reference_at(own, elapsed_h) if usable else (None, None)
        if observed is not None:
            found.append(FrameVolume(_measured_m3(observed), MEASURED))
            continue
        if cutoff is not None:
            found.append(FrameVolume(None, None))
            continue
        volume = None
        # The donor's state is taken only where its readings reach the frame:
        # elsewhere there is no estimate, and its pressure record may end before.
        if donor is not None and _reference_at(donor[1], elapsed_h)[0] is not None:
            donor_mass, donor_density = donor_state(elapsed_h)
            volume = estimated_reference(
                donor[1], elapsed_h, donor_mass, donor_density, mass_g[k], density[k]
            )
        found.append(FrameVolume(volume, None if volume is None else donor[0]))
    return found


def _volume_json(volume: DisplacedVolume | None) -> dict | None:
    """A displaced volume as plain values."""
    if volume is None:
        return None
    ml = None if volume.volume_ml is None else volume.volume_ml.tolist()
    record = {
        "times_h": volume.times_h.tolist(),
        "volume_ml": ml,
        "reasons": list(volume.reasons),
        "baseline": volume.baseline.tolist(),
        "dtype": volume.times_h.dtype.name,
    }
    rules = {
        "excluded": list(volume.excluded),
        "end_h": volume.end_h,
        "peak_h": volume.peak_h,
        "tail_floor_ml": volume.tail_floor_ml,
    }
    if volume.excluded or any(rules[k] is not None for k in rules if k != "excluded"):
        record.update(rules)
    return record


def _volume(value: Mapping | None) -> DisplacedVolume | None:
    """A displaced volume from its plain values (:func:`_volume_json`)."""
    if value is None:
        return None
    dtype = np.dtype(value["dtype"])
    ml = value["volume_ml"]
    return DisplacedVolume(
        np.asarray(value["times_h"], dtype),
        None if ml is None else np.asarray(ml, dtype),
        tuple(value["reasons"]),
        np.asarray(value["baseline"], bool),
        excluded=tuple(value.get("excluded", ())),
        end_h=value.get("end_h"),
        peak_h=value.get("peak_h"),
        tail_floor_ml=value.get("tail_floor_ml"),
    )


def write_overflow_run(
    folder: Path,
    run: OverflowRun,
    proposal: EvaporationRate | None = None,
    evaporation_g_h: float | None = None,
) -> None:
    """Store a run's displaced volume as a donor reads it, in ``folder``
    (:data:`OVERFLOW_RUN`): its sand map's identity, recipe, pressure record,
    injection period and displaced volume, with the proposed evaporation rate and
    the rate it was corrected by, if any."""
    proposed = None if proposal is None else dataclasses.asdict(proposal)
    record = {
        "sand_map": run.sand_map,
        "recipe": dict(run.recipe),
        "pressure_h": np.asarray(run.pressure_h, np.float64).tolist(),
        "pressure_bar": np.asarray(run.pressure_bar, np.float64).tolist(),
        "injection_h": list(run.injection_h),
        "volume": _volume_json(run.volume),
        "proposed_evaporation": proposed,
        "evaporation_g_h": evaporation_g_h,
    }
    write_json(Path(folder) / OVERFLOW_RUN, record)


def read_overflow_run(folder: Path) -> OverflowRun:
    """A run's stored displaced volume as a donor reads it
    (:func:`write_overflow_run`)."""
    record = json.loads((Path(folder) / OVERFLOW_RUN).read_text(encoding="utf-8"))
    return OverflowRun(
        sand_map=record["sand_map"],
        recipe=record["recipe"],
        pressure_h=np.asarray(record["pressure_h"], np.float64),
        pressure_bar=np.asarray(record["pressure_bar"], np.float64),
        injection_h=tuple(record["injection_h"]),
        volume=_volume(record["volume"]),
    )


def write_frame_volumes(
    folder: Path,
    volumes: Mapping[str, FrameVolume],
    reasons: Sequence[str] = (),
    held_out: Mapping[str, Mapping[str, FrameVolume]] | None = None,
) -> None:
    """Store a run's free-gas volume per frame in ``folder`` (:data:`FRAME_VOLUMES`):
    each frame's volume and source by its name, why the run's own readings are not
    usable, and, for a run held out of a calibration fold whose estimate the frames
    took, the frames' volumes without it."""
    record = {
        "volumes": {k: dataclasses.asdict(v) for k, v in volumes.items()},
        "reasons": list(reasons),
        "held_out": {
            run: {k: dataclasses.asdict(v) for k, v in frames.items()}
            for run, frames in (held_out or {}).items()
        },
    }
    write_json(Path(folder) / FRAME_VOLUMES, record)


def read_frame_volumes(folder: Path, held_out: str | None = None) -> dict:
    """A run's stored free-gas volume per frame (:func:`write_frame_volumes`), by
    frame: without the held-out run's estimate where the frames took it, else
    the run's volumes."""
    record = json.loads((Path(folder) / FRAME_VOLUMES).read_text(encoding="utf-8"))
    frames = record["held_out"].get(held_out, record["volumes"])
    return {k: FrameVolume(**v) for k, v in frames.items()}


def held_out_runs(folder: Path) -> tuple[str, ...]:
    """The runs held out of a calibration fold without whose estimate a run's
    frames have volumes of their own (:func:`write_frame_volumes`)."""
    record = json.loads((Path(folder) / FRAME_VOLUMES).read_text(encoding="utf-8"))
    return tuple(record["held_out"])
