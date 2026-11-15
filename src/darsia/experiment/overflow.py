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

Every setting is a field of :class:`OverflowSettings` or of its variant's
settings.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from darsia.experiment.overflow_circulation import CirculationSettings
from darsia.experiment.overflow_evaporation import EvaporationSettings
from darsia.experiment.overflow_feed import FeedSettings


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

    def __post_init__(self) -> None:
        """Check the variant and its table."""
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
    return DisplacedVolume(t, volume, reasons, baseline)


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
