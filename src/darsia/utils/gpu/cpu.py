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
