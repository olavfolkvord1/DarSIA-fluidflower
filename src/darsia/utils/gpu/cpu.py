"""The CPU compute backend, the reference of every operation of a backend.

The operations are those that dominate a frame's time in the methods that call them
(:func:`darsia.compute_backend`). On the CPU they run with OpenCV and SciPy exactly
as the methods ran them before there were backends; every other backend is held to
these numbers.

"""

from __future__ import annotations

import cv2
import numpy as np

import darsia

_ROWS_PER_CHUNK = 64
"""Image rows per feature chunk of the phase probabilities; the full feature matrix
is never held."""


class CpuBackend:
    """The operations on the CPU; the reference and the default backend."""

    name = "cpu"
    """Name of the backend."""
    label = "cpu"
    """The backend and its device class, which a job records and locks."""
    device = "cpu"
    """The device the operations run on, recorded with each result."""

    def remap(
        self, planes: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """Bilinear samples of image planes at source coordinates.

        Each plane is sampled with :func:`cv2.remap` (``INTER_LINEAR``, a constant
        NaN border), in the planes' dtype.

        Parameters
        ----------
        planes : array
            Floating image planes (rows, cols, channels) on the source grid.
        map_x, map_y : array
            Source column and row (float32) of each target pixel, shape
            (target rows, target cols) for all planes, or with one map per plane
            (target rows, target cols, channels).

        Returns
        -------
        array
            The samples (target rows, target cols, channels).

        """
        result = np.empty((*map_x.shape[:2], planes.shape[2]), planes.dtype)
        for channel in range(planes.shape[2]):
            maps = (
                (map_x, map_y)
                if map_x.ndim == 2
                else (map_x[..., channel], map_y[..., channel])
            )
            result[..., channel] = cv2.remap(
                planes[..., channel],
                *maps,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )
        return result

    def sand_means(
        self, values: np.ndarray, masks: list[np.ndarray], rev: darsia.REV
    ) -> np.ndarray:
        """Each mask's mean of the values over a square, on the mask's pixels.

        For each mask in turn, the values are averaged over the square of ``rev``
        around each of its pixels, counting only the mask's pixels
        (:class:`darsia.VolumeAveraging` with the mask as weight, SciPy's
        reflected border), and its pixels take that mean; every other pixel keeps
        its value. The masks are disjoint, e.g. one per sand. Computed in the dtype
        of the values.

        Parameters
        ----------
        values : array
            Values (rows, cols) or (rows, cols, channels).
        masks : list of array
            Disjoint boolean masks (rows, cols).
        rev : darsia.REV
            The square of the means.

        Returns
        -------
        array
            The values with each mask's pixels replaced by their mean.

        """
        result = values.copy()
        for mask in masks:
            if mask.any():
                weight = darsia.ScalarImage(mask.astype(result.dtype))
                averaging = darsia.VolumeAveraging(rev=rev, mask=weight)
                result[mask] = averaging(result)[mask]
        return result

    def phase_probabilities(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray | None,
        *,
        texture_sigma_mm: float,
        pixel_size_mm: float,
        heads: tuple[darsia.LogisticHead, darsia.LogisticHead],
    ) -> tuple[np.ndarray, np.ndarray]:
        """The probabilities of two logistic heads on a frame's phase features.

        The features (:class:`darsia.PhaseFeatures`) are read in chunks of rows into
        one matrix, and each head gives its probability of each pixel
        (:meth:`darsia.LogisticHead.probability`); outside the features' support
        both are NaN.

        Parameters
        ----------
        current, baseline : array
            Linear RGB images of shape (rows, cols, 3) on one grid.
        previous : array or None
            Linear RGB image of the run's previous frame on the same grid.
        texture_sigma_mm : float
            Standard deviation (mm) of the Gaussian defining the image texture.
        pixel_size_mm : float
            Pixel size of the images' grid.
        heads : tuple of darsia.LogisticHead
            The CO2-presence and the gas-within-CO2 head.

        Returns
        -------
        tuple of array
            The two probabilities (rows, cols) in the images' floating dtype.

        """
        features = darsia.PhaseFeatures(
            current,
            baseline,
            previous,
            texture_sigma_mm=texture_sigma_mm,
            pixel_size_mm=pixel_size_mm,
        )
        rows, cols = features.support.shape
        probabilities = tuple(np.empty((rows, cols), features.dtype) for _ in heads)
        chunk_pixels = min(rows, _ROWS_PER_CHUNK) * cols
        columns = len(darsia.PHASE_FEATURE_NAMES)
        buffer = np.empty((chunk_pixels, columns), features.dtype)
        for start in range(0, rows, _ROWS_PER_CHUNK):
            chunk = np.s_[start : start + _ROWS_PER_CHUNK]
            size = probabilities[0][chunk].size
            matrix = features.matrix(chunk, out=buffer[:size])
            for head, probability in zip(heads, probabilities):
                probability[chunk] = head.probability(matrix).reshape(-1, cols)
        for probability in probabilities:
            probability[~features.support] = np.nan
        return probabilities
