"""The panels of films and figures, each drawn once per frame as a tile.

A tile is one panel of one frame at a fixed size, without labels, keys or colour
bars; a film or figure adds those once (:mod:`layouts`). The panels:

- the corrected image as a photograph (:func:`photograph`, :func:`image_tile`);
- a mass map on one fixed colour scale (:func:`map_tile`);
- the run's mass graph with the frame marked on every curve (:func:`graph_tile`).

Image and map tiles show the part of the stored grid where the corrected image
lies (its box, x, y, width and height in cells), on the display grid of every
``step``-th cell (:mod:`display_mean`), so the panels compare cell for cell.

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from matplotlib import colormaps, rc_context

from darsia.presets.workflows.media.style import FigureStyle, canvas, pixels

TEXT_PT = 10.0
"""Text size [points] around image and map tiles; colour-bar labels are a step
smaller."""
GRAPH_PT = 15.0
"""Text size [points] of a graph tile."""
COLOUR_BAR_PAD = 0.012
"""Gap between a map and its colour bar, as a share of the map's width."""
SRGB = dict(threshold=0.0031308, gain=12.92, power=2.4, scale=1.055, offset=0.055)
"""The sRGB transfer of linear light to display values."""


@dataclass(frozen=True)
class PhotographSettings:
    """How a corrected image is shown as a photograph; one table."""

    matrix: tuple[tuple[float, float, float], ...]
    """Camera RGB to linear sRGB: a pixel's row vector times this 3 x 3 matrix."""
    invalid_grey: int
    """Grey level (0 to 255) of pixels without a value."""


def tile_size(box, width: int) -> tuple[int, int]:
    """A tile's size [pixels] for the shown part ``box`` of the grid: ``width``
    wide, both sides even."""
    height = max(2, int(round(width * box[3] / box[2] / 2)) * 2)
    return int(width), height


def view_limits(box, step: int) -> tuple[float, float, float, float]:
    """Axis limits (left, right, bottom, top) of ``box`` on a display grid of every
    ``step``-th cell."""
    x, y, width, height = box
    return (
        x / step - 0.5,
        (x + width) / step - 0.5,
        (y + height) / step - 0.5,
        y / step - 0.5,
    )


def photograph(linear_rgb: np.ndarray, settings: PhotographSettings) -> np.ndarray:
    """A corrected image (rows x columns x camera RGB, linear) as display RGB in
    [0, 1], quantised to 8 bits.

    The colour transfer runs in double precision, as our production ran it.
    Pixels without a finite value in every channel are grey.

    """
    valid = np.isfinite(linear_rgb).all(axis=-1)
    linear = np.where(valid[..., None], linear_rgb, 0).astype(np.float64)
    srgb = np.clip(linear @ np.asarray(settings.matrix, dtype=np.float64), 0, 1)
    encoded = np.where(
        srgb <= SRGB["threshold"],
        SRGB["gain"] * srgb,
        SRGB["scale"] * np.power(srgb, 1 / SRGB["power"]) - SRGB["offset"],
    )
    encoded = np.rint(255 * encoded) / 255
    encoded[~valid] = settings.invalid_grey / 255
    return encoded.astype(np.float32)


def _view(axis, box, step: int) -> None:
    """Fill the figure with ``axis`` and show ``box`` of the display grid."""
    engine = axis.figure.get_layout_engine()
    engine.set(w_pad=0, h_pad=0, wspace=0, hspace=0)
    axis.set_autoscale_on(False)
    left, right, bottom, top = view_limits(box, step)
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)


def image_tile(photo: np.ndarray, box, step: int, size) -> np.ndarray:
    """A corrected image's tile: the photograph of the box's every ``step``-th
    pixel (:func:`photograph`) filling a ``size`` tile; RGB pixels."""
    figure = canvas(size)
    axis = figure.subplots()
    _view(axis, box, step)
    left, _right, _bottom, top = view_limits(box, step)
    rows, columns = photo.shape[:2]
    axis.imshow(
        photo,
        extent=(left, left + columns, top + rows, top),
        origin="upper",
        interpolation="antialiased",
    )
    axis.set_axis_off()
    return pixels(figure)[..., :3]


def map_tile(
    values: np.ndarray,
    box,
    step: int,
    size,
    *,
    scale: float,
    label: str,
    unit: str,
    style: FigureStyle,
) -> np.ndarray:
    """A mass map's tile: its display values from 0 to ``scale`` in the style's
    colour map, cells without a value in its no-value colour; RGB pixels.

    The map is drawn beside its colour bar, named by ``label`` and ``unit``, and
    its own place is cut out and scaled to ``size``, as our production drew it.

    """
    width, height = size
    with rc_context({"font.size": TEXT_PT, "axes.labelsize": "small"}):
        figure = canvas((int(width * 1.25), height + 24))
        axis = figure.subplots()
        _view(axis, box, step)
        placed = []
        figure.canvas.mpl_connect(
            "draw_event",
            lambda _event: placed.append(axis.get_position(original=False).bounds),
        )
        colours = colormaps[style.colour_map].with_extremes(bad=style.no_value)
        # Displayed in double precision, as our production displayed every map.
        shown = np.ma.masked_invalid(np.asarray(values, dtype=np.float64))
        mapped = axis.imshow(
            shown,
            cmap=colours,
            vmin=0.0,
            vmax=float(scale),
            origin="upper",
            interpolation="nearest",
        )
        figure.colorbar(
            mapped,
            ax=axis,
            fraction=0.035,
            pad=COLOUR_BAR_PAD,
            label=f"{label} ({unit}) · fixed scale 0 to {float(scale):g}",
        )
        axis.set_axis_off()
        drawn = pixels(figure)
    left, bottom, part_w, part_h = placed[-1]
    rows, columns = drawn.shape[:2]
    x0, x1 = int(round(left * columns)), int(round((left + part_w) * columns))
    y0 = int(round((1 - bottom - part_h) * rows))
    y1 = int(round((1 - bottom) * rows))
    part = np.ascontiguousarray(drawn[max(0, y0) : y1, max(0, x0) : x1, :3])
    if (part.shape[1], part.shape[0]) != (width, height):
        part = cv2.resize(part, (width, height), interpolation=cv2.INTER_AREA)
    return part


def mass_curves(
    rows: Sequence[Mapping], style: FigureStyle
) -> tuple[list[float], dict[str, list[float]]]:
    """A run's mass records as graph series: the elapsed hours and, per curve of
    the style with a finite value, its values (NaN where none)."""
    hours = [float(row["elapsed_h"]) for row in rows]
    series = {}
    for curve in style.curves:
        values = [row.get(curve.column) for row in rows]
        values = [math.nan if value is None else float(value) for value in values]
        if any(math.isfinite(value) for value in values):
            series[curve.column] = values
    return hours, series


def shared_axes(runs: Sequence[Sequence[Mapping]], style: FigureStyle) -> tuple:
    """Graph limits every run of a film shares: time from 0 to the latest frame
    and mass from 0 to the largest value, with 3 % and 5 % room."""
    latest, largest = 0.0, 0.0
    for rows in runs:
        hours, series = mass_curves(rows, style)
        latest = max([latest, *(value for value in hours if math.isfinite(value))])
        for values in series.values():
            largest = max([largest, *(v for v in values if math.isfinite(v))])
    return (0.0, latest * 1.03 or 1.0), (0.0, largest * 1.05 or 1.0)


def graph_tile(
    rows: Sequence[Mapping],
    case: str,
    size,
    *,
    axes,
    style: FigureStyle,
    top_room: float = 0.0,
) -> np.ndarray:
    """A mass graph's tile: the run's curves on the film's ``axes``, frame ``case``
    marked on each curve and by a vertical line, no key; RGB pixels.

    The tile is white and its frame markers are edged in white, as our
    production drew them. ``top_room`` is the share of the tile kept free above
    the axes, for a label a grid film puts in the tile's corner.

    """
    width, height = (int(value) for value in size)
    hours, series = mass_curves(rows, style)
    frame = next(n for n, row in enumerate(rows) if row["case_id"] == case)
    scale = GRAPH_PT / TEXT_PT
    figure = canvas((width, height), constrained=False)
    figure.patch.set_facecolor("white")
    line = GRAPH_PT * 100 / 72
    left, bottom, right, top = 6.0 * line, 3.4 * line, 1.4 * line, 0.8 * line
    axis = figure.add_axes(
        [
            left / width,
            bottom / height,
            max(0.05, (width - left - right) / width),
            max(0.05, (height - top - bottom) / height),
        ]
    )
    for curve in style.curves:
        if curve.column not in series:
            continue
        values = series[curve.column]
        axis.plot(
            hours,
            values,
            curve.line,
            color=curve.colour,
            linewidth=1.6 * scale,
            zorder=2,
        )
        if math.isfinite(values[frame]):
            axis.plot(
                [hours[frame]],
                [values[frame]],
                "o",
                color=curve.colour,
                markersize=5.5 * scale,
                markeredgecolor="white",
                markeredgewidth=0.8 * scale,
                zorder=4,
            )
    axis.axvline(hours[frame], color=style.marker, linewidth=1.0 * scale, zorder=3)
    if top_room > 0:
        x0, y0, axis_w, axis_h = axis.get_position().bounds
        top_edge = min(y0 + axis_h, 1.0 - top_room)
        axis.set_position([x0, y0, axis_w, max(0.05, top_edge - y0)])
    small = 0.9 * GRAPH_PT
    axis.set_xlabel("Elapsed time (h)", fontsize=small, labelpad=2)
    axis.set_ylabel("CO₂ mass (g)", fontsize=small, labelpad=3)
    axis.tick_params(labelsize=small, length=3 * scale, pad=2)
    axis.grid(alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_xlim(*axes[0])
    axis.set_ylim(*axes[1])
    return pixels(figure)[..., :3]
