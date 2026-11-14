"""Lens model given in configuration, for cameras that record no lens tables.

Each colour channel is displaced along the distance from a centre: a camera point
at distance ``d`` moves outwards by ``a r + b r^3`` pixels, with ``r = d / radius``
and the channel's coefficients ``a`` and ``b``, and the camera grid's origin lies at
an offset on the sensor. The values are measured for a camera, e.g. by matching
each channel of the raw files to the camera's own images, and are given in a
configuration table. :func:`darsia.lens_coordinates` uses this model for a raw
file without recorded tables.

All arithmetic runs in the floating precision of the given coordinates (single
precision at least).

"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

_SMALLEST_DISTANCE = 1e-12
"""Distance from the centre below which the direction of displacement is not used."""


@dataclass(frozen=True)
class RadialLensSettings:
    """A camera's lens model given in configuration."""

    centre: tuple[float, float]
    """Centre (column, row) of the displacements in camera coordinates."""
    radius: float
    """Distance from the centre, in camera pixels, at which ``r`` is one."""
    coefficients: list[list[float]]
    """Per channel (red, green, blue): ``[a, b]`` of the displacement
    ``a r + b r^3`` in pixels."""
    sensor_offset: tuple[float, float]
    """Column and row of the camera grid's origin on the sensor."""

    def __post_init__(self) -> None:
        """Refuse a radius that is not a finite, positive distance.

        A negative radius would reverse the odd terms silently; zero or a
        non-finite radius would make the coordinates invalid.

        """
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("The radius of a radial lens model must be positive.")

    def coordinates(self, path: Path) -> Callable:
        """The coordinate function of this model, for any raw file.

        Parameters
        ----------
        path : Path
            Raw file (not read; the model holds for every file of the camera).

        Returns
        -------
        callable
            :func:`radial_lens_coordinates` with these settings.

        """
        return partial(radial_lens_coordinates, settings=self)


def radial_lens_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    settings: RadialLensSettings,
    channels: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor coordinates of camera coordinates, per colour channel.

    Parameters
    ----------
    x, y : array
        Camera columns and rows.
    settings : RadialLensSettings
        The lens model.
    channels : tuple of int, optional
        Colour channels (0 red, 1 green, 2 blue) to give coordinates for.

    Returns
    -------
    array
        Sensor columns, with one channel per entry of ``channels`` on a last axis.
    array
        Sensor rows, of the same shape.

    """
    dtype = np.result_type(x, y, np.float32)
    cx, cy = (dtype.type(value) for value in settings.centre)
    vx, vy = x - cx, y - cy
    distance = np.sqrt(vx * vx + vy * vy)
    relative = distance / dtype.type(settings.radius)
    columns, rows = [], []
    for channel in channels:
        a, b = (dtype.type(value) for value in settings.coefficients[channel])
        factor = (a * relative + b * relative**3) / np.maximum(
            distance, dtype.type(_SMALLEST_DISTANCE)
        )
        columns.append(x + dtype.type(settings.sensor_offset[0]) + factor * vx)
        rows.append(y + dtype.type(settings.sensor_offset[1]) + factor * vy)
    return np.stack(columns, -1), np.stack(rows, -1)
