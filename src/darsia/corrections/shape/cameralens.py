"""Lens model from the correction tables a camera records in each raw file.

Some cameras record lens-correction tables in every raw file: a radial distortion
table and, per colour, a lateral chromatic aberration table, each a function of the
distance from the optical centre. The tables of each acquisition, with the camera's
calibrated centre, radius and amplitude, map camera coordinates (a lens-corrected,
camera-sized grid) onto the sensor; red and blue get their own coordinates. Passed
to :func:`darsia.registration_mapping`, this lets one interpolation replay a frame's
registration and lens correction from the demosaiced sensor.

All arithmetic runs in the floating precision of the given coordinates (single
precision at least).

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True)
class CameraLensSettings:
    """Where a camera records its lens tables, and its calibrated lens model."""

    page: int
    """Index of the TIFF page that holds the record of the tables."""
    subpage: int
    """Index of the sub-page (SubIFD) of that page that holds the tables."""
    distortion_tag: int
    """TIFF tag of the distortion table."""
    distortion_header: int
    """First entry of a valid distortion record."""
    distortion_entries: tuple[int, int]
    """Start and stop of the distortion table in its record."""
    chromatic_tag: int
    """TIFF tag of the chromatic aberration tables."""
    chromatic_header: int
    """First entry of a valid chromatic aberration record."""
    red_entries: tuple[int, int]
    """Start and stop of the red table in the chromatic aberration record."""
    blue_entries: tuple[int, int]
    """Start and stop of the blue table in the chromatic aberration record."""
    knots: list[float]
    """Distances from the centre, relative to the radius, of the table entries."""
    distortion_scale: float
    """Table value that changes the radial scale by one (times the amplitude)."""
    chromatic_scale: float
    """Table value that changes a colour's radial scale by one."""
    centre: tuple[float, float]
    """Optical centre (column, row) in camera coordinates."""
    radius: float
    """Distance from the centre at the last knot, in camera pixels."""
    amplitude: float
    """Factor of the distortion table."""
    shift: tuple[float, float]
    """Shift (column, row) of the sensor coordinates, in sensor pixels."""
    sensor_offset: tuple[int, int]
    """Column and row of the camera grid's origin on the sensor."""


@dataclass(frozen=True)
class CameraLensTables:
    """The lens tables recorded in one raw file."""

    distortion: tuple[int, ...]
    """Radial distortion, one entry per knot."""
    red: tuple[int, ...]
    """Lateral chromatic aberration of red, one entry per knot."""
    blue: tuple[int, ...]
    """Lateral chromatic aberration of blue, one entry per knot."""


def read_camera_lens_tables(
    path: Path, settings: CameraLensSettings
) -> CameraLensTables:
    """Read the lens tables that a camera recorded in a raw file.

    Parameters
    ----------
    path : Path
        Raw file in TIFF structure (e.g. an ARW file).
    settings : CameraLensSettings
        Where the tables are recorded.

    Returns
    -------
    CameraLensTables
        The distortion table and the red and blue chromatic aberration tables.

    Raises
    ------
    ValueError
        If a record does not start with its header or is too short.

    """
    with tifffile.TiffFile(path) as tiff:
        record = tiff.pages[settings.page].pages[settings.subpage]
        distortion = tuple(int(v) for v in record.tags[settings.distortion_tag].value)
        chromatic = tuple(int(v) for v in record.tags[settings.chromatic_tag].value)
    if (
        distortion[0] != settings.distortion_header
        or len(distortion) < settings.distortion_entries[1]
        or chromatic[0] != settings.chromatic_header
        or len(chromatic) < max(settings.red_entries[1], settings.blue_entries[1])
    ):
        raise ValueError(f"{path} records lens tables of another structure.")
    return CameraLensTables(
        distortion[slice(*settings.distortion_entries)],
        chromatic[slice(*settings.red_entries)],
        chromatic[slice(*settings.blue_entries)],
    )


def camera_lens_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    tables: CameraLensTables,
    settings: CameraLensSettings,
    channels: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor coordinates of camera coordinates, per colour channel.

    A point at distance ``d`` from the optical centre moves radially by the factor
    ``1 + amplitude * D(d / radius) / distortion_scale``, where ``D`` interpolates
    the distortion table linearly between the knots (constant beyond them). Red and
    blue are scaled once more by ``1 + C(d / radius) / chromatic_scale`` with their
    own table ``C``; green is not.

    Parameters
    ----------
    x, y : array
        Camera columns and rows.
    tables : CameraLensTables
        The lens tables of the acquisition.
    settings : CameraLensSettings
        The calibrated lens model.
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
    cu, cv = (dtype.type(value) for value in settings.centre)
    tx, ty = (dtype.type(value) for value in settings.shift)
    du, dv = x - cu, y - cv
    ratio = np.hypot(du, dv) / dtype.type(settings.radius)
    scale = (
        1
        + dtype.type(settings.amplitude)
        * _interpolate(ratio, settings.knots, tables.distortion, dtype)
        / settings.distortion_scale
    )
    columns, rows = [], []
    for channel in channels:
        table = (tables.red, None, tables.blue)[channel]
        factor = (
            scale
            if table is None
            else scale
            * (
                1
                + _interpolate(ratio, settings.knots, table, dtype)
                / settings.chromatic_scale
            )
        )
        columns.append(cu + settings.sensor_offset[0] + tx + factor * du)
        rows.append(cv + settings.sensor_offset[1] + ty + factor * dv)
    return np.stack(columns, -1), np.stack(rows, -1)


def _interpolate(x: np.ndarray, points, values, dtype) -> np.ndarray:
    """Piecewise-linear interpolation, constant beyond the ends, in ``dtype``.

    It is not :func:`darsia.piecewise_linear`, which continues the end segments and
    computes ``slope * (x - x_low) + y_low``: this form, ``v_low + w * (v_high -
    v_low)``, is the arithmetic of our production's lens tables, and the other would
    move the lens coordinates by rounding.
    """
    x, points, values = (np.asarray(a, dtype=dtype) for a in (x, points, values))
    high = np.searchsorted(points, x, side="right").clip(1, len(points) - 1)
    low = high - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = (x - points[low]) / (points[high] - points[low])
    result = values[low] + weight * (values[high] - values[low])
    return np.where(
        x < points[0], values[0], np.where(x >= points[-1], values[-1], result)
    )
