"""Signal of a colour change read along colour paths, without a cap at a path's end.

A sand's colour path (:class:`darsia.ColorPathInterpolation`) gives the value of
a baseline-relative colour on the path, and a signal function (the supports and
values of a :class:`darsia.PWTransformation`) maps that value to the signal. The
path's last segment goes on beyond its last key colour, also when it is the only
one, and so does the signal function's last segment beyond its last support: a
colour change beyond the end of the path keeps rising along the path's last
segment instead of being held at the path's end. Below its first support the
signal function keeps its first value. All arithmetic runs in the floating dtype
of the given colours.

A sand's factor is its path's scale over its measured colour range: the change
along its path that its dissolved CO2 reaches beside the free gas
(:func:`color_ranges`, with the settings of :class:`ColorRangeSettings`), stored
by :func:`write_color_ranges`.

"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

import darsia


class ColorPathSignal(darsia.Model):
    """Signal of colours along one colour path, continued beyond the path's end."""

    def __init__(
        self,
        interpolation: darsia.ColorPathInterpolation,
        transformation: darsia.PWTransformation,
    ) -> None:
        """Signal of one colour path.

        Parameters
        ----------
        interpolation : darsia.ColorPathInterpolation
            The colour path and its values at the key colours.
        transformation : darsia.PWTransformation
            Signal function: the signal at increasing supports of the path's value.

        """
        self.interpolation = interpolation
        """Colour path and its values."""
        self.supports = np.asarray(transformation.supports, dtype=float)
        """Supports of the signal function."""
        self.values = np.asarray(transformation.values, dtype=float)
        """Values of the signal function at its supports."""

    def __call__(self, colors: np.ndarray) -> np.ndarray:
        """Signal of colours (rows of RGB, relative to the baseline for a relative
        colour path), in their floating dtype.

        """
        path = self.interpolation.color_path
        value = (
            self.interpolation(colors)
            if path.num_segments > 1
            else self._along_segment(colors)
        )
        x = self.supports.astype(value.dtype)
        y = self.values.astype(value.dtype)
        return np.where(value < x[0], y[0], darsia.piecewise_linear(value, x, y))

    def _along_segment(self, colors: np.ndarray) -> np.ndarray:
        """Value of colours on a path of one segment, which goes on beyond its end.

        The path's interpretation holds its only segment at the end, as a first
        segment; here it is also the last one. Up to the end the value is the
        interpretation's, with the same operations.

        """
        path, interpolation = self.interpolation.color_path, self.interpolation
        dtype = colors.dtype if np.issubdtype(colors.dtype, np.floating) else float
        supports = np.asarray(
            (
                path.colors
                if interpolation.color_mode == darsia.ColorMode.ABSOLUTE
                else path.relative_colors
            ),
            dtype=dtype,
        )
        d = np.asarray(path.equidistant_distances, dtype=dtype)
        v = np.asarray(interpolation.values, dtype=dtype)
        direction = supports[1] - supports[0]
        parameter = d[0] + (d[1] - d[0]) * np.tensordot(
            colors - supports[0], direction, axes=([-1], [0])
        ) / np.dot(direction, direction)
        parameter = np.where(np.isnan(parameter), 0, parameter)
        return v[0] + (parameter - d[0]) / (d[1] - d[0]) * (v[1] - v[0])


class SandColorPaths:
    """Each sand's colour change read on its own colour path, as a share of its
    measured colour range.

    """

    def __init__(
        self, paths: Mapping[int, ColorPathSignal], factors: Mapping[int, float]
    ) -> None:
        """Colour paths of the sands and their factors.

        Parameters
        ----------
        paths : mapping of int to ColorPathSignal
            Colour path of each sand (facies id); only these facies are read.
        factors : mapping of int to float
            Factor of each sand's signal: its path's scale over its measured colour
            range, so that the signal is a share of that range.

        """
        self.paths = dict(paths)
        """Colour path of each sand."""
        self.factors = {sand: float(factors[sand]) for sand in self.paths}
        """Factor of each sand's signal."""
        self.sands = sorted(self.paths)
        """Facies ids of the sands."""

    def __call__(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        support: np.ndarray,
        dissolved: np.ndarray,
        facies: np.ndarray,
    ) -> np.ndarray:
        """Signal of the sands' dissolved CO2 in a frame; zero elsewhere.

        All pixels of a sand are read together, with no colour change where it is
        not measured, so that a pixel's signal does not depend on which pixels are
        dissolved (in single precision, a projection's rounding depends on the
        pixels projected with it).

        Parameters
        ----------
        current, baseline : array
            RGB images of the frame and of the baseline, of shape (rows, cols, 3).
        support : array
            Pixels whose colour change is measured.
        dissolved : array
            Pixels of dissolved CO2 read.
        facies : array
            Facies id of each pixel.

        Returns
        -------
        array
            The signal, in the dtype of the images.

        """
        relative = np.where(support[..., None], current - baseline, 0)
        signal = np.zeros(facies.shape, dtype=relative.dtype)
        for sand, path in self.paths.items():
            pixels = facies == sand
            mask = dissolved & pixels
            if mask.any():
                values = path(relative[pixels])[mask[pixels]]
                signal[mask] = values * values.dtype.type(self.factors[sand])
        return signal


@dataclass(frozen=True)
class ColorRangeSettings:
    """Settings of the colour-range measurement; one configuration table."""

    mean_mm: float
    """Side [mm] of the centred square, an odd number of pixels, over which the
    change along the path is averaged inside the sand's dissolved CO2."""
    inner_mm: float
    """Depth [mm] inside the sand's dissolved CO2 of the pixels measured."""
    gas_mm: float
    """Distance [mm] from the free gas of the pixels that give the range."""
    minimum_inner_mm2: float
    """Smallest area [mm²] of a frame's pixels measured."""
    minimum_gas_mm2: float
    """Smallest area [mm²] of a frame's pixels that give the range."""
    gas_percentile: float
    """Percentile of their mean change that is the frame's range."""


def color_ranges(
    runs: Sequence[tuple], settings: ColorRangeSettings
) -> dict[int, float]:
    """Each sand's colour range: the change along its colour path that its
    dissolved CO2 reaches beside the free gas.

    In each frame, a sand's colour change is projected on the direction of its
    colour path's end and averaged over the centred square inside the sand's
    dissolved CO2 (beyond the grid counts as outside it). The frame's range is a
    percentile of this mean over the pixels at least ``inner_mm`` inside the
    dissolved CO2 and within ``gas_mm`` of the free gas (erosion and dilation in
    steps of one pixel, none for a length that rounds to zero pixels), when both
    sets of pixels are large enough. The free gas the phase model found
    (:func:`darsia.model_gas`, which gravity or a peel made dissolved CO2) is free
    gas here, not dissolved CO2, as in the run statistics. A run's range is the
    largest of its frames', on the run's own colour paths, and the sand's range
    the mean of the runs'. Lengths and areas are on the sand plane, converted to
    pixels with the side of the cell of the run's optical pitch. Computed in the
    dtype of the images.

    Parameters
    ----------
    runs : sequence of (sequence, array, darsia.Image, tuple of float, mapping)
        For each run: its frames, each an RGB image (rows, cols, 3), its phase map
        (:class:`darsia.PhaseLabel`) and rule map (:class:`darsia.PhaseRule`) and
        the pixels whose colour change is measured, read by index; the RGB image
        of its baseline; its sand map; the pitch [m] of its grid on the sand
        plane, down a column and along a row (:func:`darsia.optical_pitch_m`);
        and the run's colour path of each sand measured (facies id to
        ColorPathSignal).
    settings : ColorRangeSettings
        Settings of the measurement.

    Returns
    -------
    dict of int to float
        The colour range of each sand.

    Raises
    ------
    ValueError
        If a sand has no positive range in some run.

    """
    largest, sands = [], set()
    for frames, baseline, sand_map, optical_pitch, paths in runs:
        sands |= set(paths)
        facies = sand_map.img
        pixel_mm = float(np.sqrt(optical_pitch[0] * optical_pitch[1])) * 1000
        pixel_mm2 = pixel_mm**2
        size = 2 * int(round((settings.mean_mm / pixel_mm - 1) / 2)) + 1
        inner_steps = int(round(settings.inner_mm / pixel_mm))
        gas_steps = int(round(settings.gas_mm / pixel_mm))
        run = {}
        for index in range(len(frames)):
            current, phase, rules, support = frames[index]
            dtype = current.dtype
            gas = darsia.model_gas(phase, rules)
            dissolved = support & (phase == darsia.PhaseLabel.DISSOLVED) & ~gas
            beside_gas = (
                ndimage.binary_dilation(gas, iterations=gas_steps) if gas_steps else gas
            )
            for sand, path in paths.items():
                mask = dissolved & (facies == sand)
                inner = (
                    ndimage.binary_erosion(mask, iterations=inner_steps)
                    if inner_steps
                    else mask
                )
                if inner.sum() < settings.minimum_inner_mm2 / pixel_mm2:
                    continue
                measured = beside_gas & inner
                if measured.sum() < settings.minimum_gas_mm2 / pixel_mm2:
                    continue
                end = np.asarray(
                    path.interpolation.color_path.relative_colors[-1], dtype
                )
                change = ((current - baseline) * (end / np.linalg.norm(end))).sum(-1)
                weight = ndimage.uniform_filter(
                    mask.astype(dtype), size, mode="constant"
                )
                mean = ndimage.uniform_filter(
                    np.where(mask, change, 0).astype(dtype), size, mode="constant"
                ) / np.maximum(weight, 1e-6)
                value = float(np.percentile(mean[measured], settings.gas_percentile))
                run[sand] = max(run.get(sand, value), value)
        largest.append(run)
    missing = [
        sand for sand in sorted(sands) if any(run.get(sand, 0) <= 0 for run in largest)
    ]
    if missing:
        raise ValueError(f"No positive colour range of sand(s) {missing} in every run")
    return {
        sand: float(np.mean([run[sand] for run in largest])) for sand in sorted(sands)
    }


COLOR_RANGES = "color_ranges.json"
"""The file of stored colour ranges (:func:`write_color_ranges`)."""


def write_color_ranges(folder: Path, ranges: Mapping[int, float]) -> None:
    """Store each sand's colour range (:func:`color_ranges`) in ``folder``, as
    JSON with every digit kept."""
    found = {str(int(sand)): float(value) for sand, value in ranges.items()}
    text = json.dumps(found, indent=1, allow_nan=False)
    (Path(folder) / COLOR_RANGES).write_text(text + "\n", encoding="utf-8")


def read_color_ranges(folder: Path) -> dict[int, float]:
    """Stored colour ranges by facies id (:func:`write_color_ranges`)."""
    text = (Path(folder) / COLOR_RANGES).read_text(encoding="utf-8")
    return {int(sand): float(value) for sand, value in json.loads(text).items()}
