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

"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

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
