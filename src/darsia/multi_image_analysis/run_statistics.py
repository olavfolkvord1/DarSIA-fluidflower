"""Run statistics: per frame and per sand, the signal of the dissolved CO2.

For every frame of a run, the colour change of each sand inside the dissolved CO2
of the frame's phase map is read as a signal by a sand reading (e.g.
:class:`darsia.SandColorPaths`: each sand on its own colour path). The signal is
then averaged over a square of a set side inside the sand's dissolved CO2, so
that sand grains leave the signal while no other sand and no pixel outside the
dissolved CO2 enters the mean. The side is an optical length, converted to
pixels with the pitch on the sand plane (:func:`darsia.optical_pitch_m`). The
free gas the phase model found, which gravity or a peel made dissolved CO2
(:func:`darsia.model_gas`), is not dissolved CO2 here, as in our production.

The statistics of a frame are, for each sand, the histogram of this signal over
the sand's dissolved pixels that count: the signal relative to the range of the
concentration curve (from no dissolved CO2 to saturation), clipped to [0, 1] and
rounded to one of a set number of equal intervals, with the summed weight of the
pixels in each interval (e.g. each pixel's capacity for dissolved CO2). A frame's
statistics depend on that frame and the run's inputs only, so a run computed in
chunks of frames equals the run computed whole.

All arithmetic runs in the floating dtype of the images; the histogram sums run
in the dtype of the weights. Every setting is a field of
:class:`RunStatisticsSettings`.

"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

import darsia


@dataclass(frozen=True)
class RunStatisticsSettings:
    """Settings of the run statistics; one configuration table."""

    mean_mm: float
    """Side [mm] of the square over which each sand's signal is averaged inside its
    dissolved CO2."""
    zero_signal: float
    """Signal of no dissolved CO2: the bottom of the concentration curve."""
    saturation_signal: float
    """Signal of saturated dissolved CO2: the top of the concentration curve."""
    intervals: int
    """Number of equal intervals of the relative signal in the histograms."""


class SandReading(Protocol):
    """A reading of the sands' dissolved CO2 as a signal, the method behind the
    statistics, e.g. :class:`darsia.SandColorPaths`.

    A new reading is one class with these two members.

    """

    sands: Sequence[int]
    """Facies ids of the sands read."""

    def __call__(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        support: np.ndarray,
        dissolved: np.ndarray,
        facies: np.ndarray,
    ) -> np.ndarray:
        """Signal of the sands' dissolved CO2 in a frame, in the images' dtype.

        Parameters
        ----------
        current, baseline : array
            RGB images of the frame and of the baseline (rows, cols, 3).
        support : array
            Pixels whose colour change is measured.
        dissolved : array
            Pixels of dissolved CO2 read.
        facies : array
            Facies id of each pixel.

        Returns
        -------
        array
            The signal (rows, cols); zero outside the dissolved pixels read.

        """
        ...


@dataclass(frozen=True)
class SandHistogram:
    """Histogram of one sand's relative signal in one frame."""

    relative: np.ndarray
    """Relative signal of each occupied interval, increasing, in [0, 1]."""
    weight: np.ndarray
    """Summed weight of the pixels in each interval."""


def sand_mean(
    signal: np.ndarray,
    dissolved: np.ndarray,
    sand_masks: dict[int, np.ndarray],
    rev: darsia.REV,
) -> np.ndarray:
    """The signal averaged over ``rev`` inside each sand's dissolved CO2.

    Only the sand's dissolved pixels enter its mean, and only they change; the
    signal elsewhere is kept. Computed in the dtype of the signal, on the process's
    compute backend (:func:`darsia.compute_backend`).

    """
    masks = [dissolved & sand_mask for sand_mask in sand_masks.values()]
    return darsia.compute_backend().sand_means(signal, masks, rev)


def _histogram(relative: np.ndarray, weight: np.ndarray, intervals: int):
    """Occupied intervals of a relative signal and the summed weight of each."""
    codes = np.rint(relative * intervals).astype(np.int64)
    occupied, index = np.unique(codes, return_inverse=True)
    sums = np.zeros(len(occupied), dtype=weight.dtype)
    np.add.at(sums, index, weight)
    return SandHistogram(occupied.astype(weight.dtype) / intervals, sums)


def run_statistics(
    frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline: np.ndarray,
    sand_map: darsia.Image,
    optical_pitch: tuple[float, float],
    counted: np.ndarray,
    reading: SandReading,
    settings: RunStatisticsSettings,
) -> Iterator[dict[int, SandHistogram]]:
    """Statistics of a run's frames, per sand.

    Parameters
    ----------
    frames : sequence of (array, array, array, array, array)
        For each frame: its RGB image (rows, cols, 3) on the grid of the sand map;
        its phase map (:class:`darsia.PhaseLabel`) and rule map
        (:class:`darsia.PhaseRule`); the pixels whose colour change is measured;
        and the weight of each pixel (e.g. its capacity for dissolved CO2). Read
        by index, so a lazy sequence keeps memory bounded.
    baseline : array
        RGB image of the run's baseline.
    sand_map : darsia.Image
        Facies ids of the run's grid.
    optical_pitch : tuple of float
        Pitch [m] of the grid on the sand plane, down a column and along a row
        (:func:`darsia.optical_pitch_m`).
    counted : array
        Pixels that count in the histograms.
    reading : SandReading
        The sands read and the signal of their dissolved CO2.
    settings : RunStatisticsSettings
        Settings of the statistics.

    Yields
    ------
    dict of int to SandHistogram
        The histogram of each sand read, for each frame in order.

    """
    # The square's side in pixels of the sand plane, as our production sized it.
    optical = darsia.optical_grid(sand_map, optical_pitch, baseline.dtype.type)
    rev = darsia.REV(size=settings.mean_mm / 1000, img=optical)
    sand_masks = {sand: sand_map.img == sand for sand in reading.sands}
    for index in range(len(frames)):
        current, phase, rules, support, weight = frames[index]
        dissolved = (
            support
            & (phase == darsia.PhaseLabel.DISSOLVED)
            & ~darsia.model_gas(phase, rules)
        )
        signal = reading(current, baseline, support, dissolved, sand_map.img)
        signal = sand_mean(signal, dissolved, sand_masks, rev)
        zero = signal.dtype.type(settings.zero_signal)
        span = signal.dtype.type(settings.saturation_signal - settings.zero_signal)
        statistics = {}
        for sand, sand_mask in sand_masks.items():
            mask = dissolved & sand_mask & counted
            relative = np.clip((signal[mask] - zero) / span, 0, 1)
            statistics[sand] = _histogram(relative, weight[mask], settings.intervals)
        yield statistics
