"""Film frames composed from tiles: a pyramid of three panels, or a grid of runs.

- A pyramid shows one run in three cells of one size: one panel centred on top,
  touching the two panels of the lower row, which fill the frame's width. The
  free areas beside the top panel hold the frame's text (left) and the top
  panel's name and key and the map's colour key (right) (:func:`pyramid_layout`,
  :func:`compose_pyramid`).
- A grid shows one panel of every run, as large as the frame allows, beside a
  column or band with the title and key (:func:`grid_layout`,
  :func:`compose_grid`).

A tile is placed as it is or scaled to its cell. A held tile, the last earlier
tile of a run without one at the frame's time, is greyed and says so. Text and
keys are drawn once per size and laid over the frame by their transparency.

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from darsia.presets.workflows.media.panels import colour_norm
from darsia.presets.workflows.media.style import (
    PIXEL_DPI,
    FigureStyle,
    bgr,
    canvas,
    pixels,
)

SEPARATOR = 2
"""Pixels between neighbouring cells."""
FONT_SHARE = 0.0135
"""A pyramid frame's base text height as a share of the frame's height."""
MARGIN_PX = 12
"""The keys' and text's distance from their area's edges [pixels at 1080 high]."""
LINE_SIZES = (2.0, 1.45, 1.0, 0.9)
"""Sizes of a pyramid's text lines relative to the base text: the run, the
elapsed time, then the rest."""
GRID_SIDE = (0.11, 0.12)
"""A grid's title column (share of the width) or band (share of the height)."""
GREY = (0.45, 0.55, 205)
"""A held tile: this share of its own grey plus this share of this grey level."""


def _points(font_px: float) -> float:
    """A text size [points] that is ``font_px`` pixels high."""
    return font_px * 72 / PIXEL_DPI


def _wrapped(text: str, renderer, points: float, weight: str, width_px: float) -> str:
    """``text`` broken between words so that no line is wider than ``width_px``; a
    long word stays whole."""
    font = FontProperties(size=points, weight=weight)
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}" if line else word
            extent = renderer.get_text_width_height_descent(trial, font, ismath=False)
            if line and extent[0] > width_px:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def text_image(
    text: str,
    font_px: float,
    *,
    backing: float = 0.55,
    bold: bool = False,
    max_px: int = 0,
) -> np.ndarray:
    """White ``text`` on a black backing of opacity ``backing``, broken between
    words to at most ``max_px`` wide when given; RGBA pixels. Text sizes are drawn
    to a tenth of a pixel."""
    return _text(str(text), round(float(font_px), 1), backing, bold, max_px)


@lru_cache(maxsize=512)
def _text(text, font_px, backing, bold, max_px) -> np.ndarray:
    """:func:`text_image`, drawn once per text and size."""
    pad = max(2, round(font_px * 0.35))
    weight = "bold" if bold else "normal"
    points = _points(font_px)
    probe = canvas((10, 10), constrained=False)
    renderer = probe.canvas.get_renderer()
    if max_px:
        text = _wrapped(text, renderer, points, weight, max_px - 2 * pad)
    label = probe.text(0, 0, text, fontsize=points, fontweight=weight, linespacing=1.25)
    box = label.get_window_extent(renderer)
    width = int(math.ceil(box.width)) + 2 * pad
    height = int(math.ceil(box.height)) + 2 * pad
    figure = canvas((width, height), constrained=False)
    figure.patch.set_facecolor((0, 0, 0, backing))
    figure.text(
        pad / width,
        0.5,
        text,
        fontsize=points,
        fontweight=weight,
        color="white",
        va="center",
        ha="left",
        linespacing=1.25,
    )
    image = pixels(figure)
    image.setflags(write=False)
    return image


@lru_cache(maxsize=8)
def graph_key(font_px: float, style: FigureStyle, columns: int = 1) -> np.ndarray:
    """The mass graph's key: each curve of the style in its colour, on a light
    backing; RGBA pixels."""
    handles = [
        Line2D([], [], color=curve.colour, linestyle=curve.line, linewidth=2.0)
        for curve in style.curves
    ]
    names = [curve.name for curve in style.curves]
    options = dict(
        loc="center",
        ncols=columns,
        frameon=False,
        fontsize=_points(font_px),
        labelcolor="black",
        handlelength=2.0,
        columnspacing=1.2,
        borderpad=0.3,
        borderaxespad=0.0,
    )
    pad = max(2, round(font_px * 0.3))
    probe = canvas((2000, 600), constrained=False)
    box = probe.legend(handles, names, **options).get_window_extent(
        probe.canvas.get_renderer()
    )
    size = (int(math.ceil(box.width)) + 2 * pad, int(math.ceil(box.height)) + 2 * pad)
    figure = canvas(size, constrained=False)
    figure.patch.set_facecolor((1, 1, 1, 0.9))
    figure.legend(handles, names, **options)
    image = pixels(figure)
    image.setflags(write=False)
    return image


@lru_cache(maxsize=32)
def colour_key(
    limit: float,
    label: str,
    height: int,
    font_px: float,
    colour_map: str,
    normalization: str = "linear",
    over_range: str | None = None,
) -> np.ndarray:
    """A map's key: a thin vertical bar of ``colour_map`` from 0 to ``limit`` with
    three ticks and ``label`` above, on a dark backing; RGBA pixels."""
    width = int(round(font_px * 4.6))
    ticks = [0.0, limit / (4 if normalization == "sqrt" else 2), limit]
    figure = canvas((width, int(height)), constrained=False)
    figure.patch.set_facecolor((0, 0, 0, 0.55))
    left, bar_w = 0.5 * font_px / width, max(6, round(font_px * 0.8)) / width
    top, bottom = 1.9 * font_px / height, 0.8 * font_px / height
    axis = figure.add_axes([left, bottom, bar_w, 1 - top - bottom])
    colours = colormaps[colour_map]
    if over_range is not None:
        colours = colours.with_extremes(over=over_range)
    mappable = ScalarMappable(
        norm=colour_norm(float(limit), normalization), cmap=colours
    )
    bar = figure.colorbar(
        mappable,
        cax=axis,
        ticks=ticks,
        extend="max" if over_range is not None else "neither",
    )
    bar.ax.tick_params(labelsize=_points(font_px), colors="white", length=2)
    bar.ax.set_yticklabels([f"{value:g}" for value in ticks])
    bar.outline.set_edgecolor("white")
    figure.text(
        left,
        1 - 0.55 * top,
        label,
        color="white",
        fontsize=_points(font_px),
        va="center",
        ha="left",
    )
    image = pixels(figure)
    image.setflags(write=False)
    return image


def blend(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Lay RGBA pixels over a BGR frame at (x, y) by their alpha, clipped to the
    frame."""
    height, width = frame.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1 = min(width, int(x) + rgba.shape[1])
    y1 = min(height, int(y) + rgba.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    part = rgba[y0 - int(y) : y1 - int(y), x0 - int(x) : x1 - int(x)]
    alpha = part[..., 3:4].astype(np.float32) / 255.0
    colour = part[..., 2::-1].astype(np.float32)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.round(colour * alpha + region * (1 - alpha)).astype(
        np.uint8
    )


def held(tile: np.ndarray) -> np.ndarray:
    """A tile greyed, so that it reads as not current."""
    own, share, level = GREY
    grey = tile.astype(np.float32).mean(axis=2, keepdims=True)
    return np.round(own * grey + share * level).repeat(3, axis=2).astype(np.uint8)


def _place(frame: np.ndarray, cell, tile, is_held: bool, empty) -> None:
    """Put a BGR tile (scaled to the cell if needed, greyed when held) in its cell,
    or the empty colour without a tile."""
    x, y, width, height = cell
    if tile is None:
        frame[y : y + height, x : x + width] = empty
        return
    if (tile.shape[1], tile.shape[0]) != (width, height):
        tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)
    frame[y : y + height, x : x + width] = held(tile) if is_held else tile


@dataclass(frozen=True)
class SideBySideLayout:
    """Two equal panels, with a separate colour key and room for captions."""

    size: tuple
    cells: tuple
    key: tuple


def paired_grid_layout(size, panel_size, key_width: int) -> SideBySideLayout:
    """Two unscaled grid panels and a key, padded only at the outer even edge."""
    panel_w, panel_h = map(int, panel_size)
    logical = 2 * panel_w + key_width, panel_h
    encoded = tuple(value + value % 2 for value in logical)
    if min(panel_w, panel_h, key_width) < 1 or tuple(size) != encoded:
        raise ValueError(
            "Paired-grid size must fit two logical panels and the key with even padding"
        )
    return SideBySideLayout(
        encoded,
        ((0, 0, panel_w, panel_h), (panel_w, 0, panel_w, panel_h)),
        (2 * panel_w, 0, key_width, panel_h),
    )


def compose_paired_grid(
    tiles,
    names,
    layout,
    *,
    lines,
    style,
    graph,
    scale,
    normalization="sqrt",
    over_range=None,
):
    """Grid-sized photo and map with a separate physical colour bar; BGR pixels."""
    if graph is not None or scale is None or normalization != "sqrt":
        raise ValueError(
            "The paired-grid layout needs a photograph and a sqrt mass map"
        )
    width, height = layout.size
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    for cell, (tile, is_held, _note) in zip(layout.cells, tiles):
        _place(frame, cell, tile, is_held, bgr(style.empty))
    x, _y, bar_width, logical_h = layout.key
    bar = frame[:logical_h, x : x + bar_width]
    bar[...] = 255
    top, bottom = min(60, logical_h // 5), logical_h - min(30, logical_h // 10)
    lut = (colormaps[style.colour_map](np.linspace(0, 1, 256))[:, :3] * 255).astype(
        np.uint8
    )[:, ::-1]
    for y in range(top, bottom):
        bar[y, 14 : min(44, bar_width)] = lut[int(255 * (bottom - y) / (bottom - top))]
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = int(bottom - fraction * (bottom - top))
        cv2.putText(
            bar,
            f"{fraction ** 2 * scale[0]:.1f}",
            (48, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bar,
        "g/m2",
        (20, max(12, top - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    # Captions are overlaid without resizing or cropping either logical panel.
    font = max(10, min(18, logical_h * 0.024))
    heading = text_image(
        " · ".join(lines[:3]), font, backing=0.5, max_px=2 * layout.cells[0][2] - 16
    )
    blend(frame, heading, 8, 4)
    for name, cell in zip(names, layout.cells):
        label = text_image(name, font * 0.85, max_px=cell[2] - 16)
        blend(frame, label, cell[0] + 8, logical_h - label.shape[0] - 8)
    if len(lines) > 3:
        note = text_image(lines[3], font * 0.65, max_px=2 * layout.cells[0][2] - 16)
        blend(frame, note, 8, heading.shape[0] + 8)
    return frame


def side_by_side_layout(size, aspect: float) -> SideBySideLayout:
    """Fit two panels of the same aspect horizontally without cropping them."""
    width, height = (int(value) for value in size)
    key_width = max(60, int(width * 0.08))
    panel_w = min((width - key_width - SEPARATOR) // 2, int(height * 0.64 * aspect))
    panel_h = int(panel_w / aspect)
    if panel_w < 8 or panel_h < 8:
        raise ValueError("The side-by-side panels do not fit this frame")
    top = (height - panel_h) // 2
    left = (width - key_width - 2 * panel_w - SEPARATOR) // 2
    return SideBySideLayout(
        (width, height),
        (
            (left, top, panel_w, panel_h),
            (left + panel_w + SEPARATOR, top, panel_w, panel_h),
        ),
        (width - key_width, top, key_width, panel_h),
    )


def compose_side_by_side(
    tiles,
    names,
    layout: SideBySideLayout,
    *,
    lines,
    style: FigureStyle,
    graph,
    scale,
    normalization="linear",
    over_range=None,
) -> np.ndarray:
    """Two panels with visible source, contour and clipping captions; BGR pixels."""
    width, height = layout.size
    frame = np.empty((height, width, 3), np.uint8)
    frame[...] = bgr(style.frame)
    font = max(10.0, height * FONT_SHARE)
    for number, (cell, (tile, is_held, note)) in enumerate(zip(layout.cells, tiles)):
        x, y, cell_w, cell_h = cell
        _place(frame, cell, tile, is_held, bgr(style.empty))
        heading = text_image(names[number], font, bold=True, max_px=cell_w - 8)
        blend(frame, heading, x + 4, y - heading.shape[0] - 4)
        if note:
            message = text_image(note, font * 0.8, max_px=cell_w - 8)
            blend(frame, message, x + 4, y + cell_h - message.shape[0] - 4)
        if number == graph:
            key = graph_key(round(font, 1), style)
            blend(frame, key, x + cell_w - key.shape[1] - 4, y + 4)
    # Run and time above the panels; provenance and quantitative clipping below.
    y = 4
    for number, line in enumerate(lines):
        if number == 3:
            y = layout.cells[0][1] + layout.cells[0][3] + 8
        message = text_image(
            line,
            font if number == 0 else font * 0.8,
            bold=number == 0,
            backing=0,
            max_px=width - 16,
        )
        if y + message.shape[0] > height:
            raise ValueError("The film captions do not fit; increase size_px")
        blend(frame, message, 8, y)
        y += message.shape[0] + 3
    if scale is not None:
        x, y, _width, key_h = layout.key
        key = colour_key(
            scale[0],
            scale[1],
            key_h,
            font * 0.8,
            style.colour_map,
            normalization,
            over_range,
        )
        blend(frame, key, x, y)
    return frame


@dataclass(frozen=True)
class PyramidLayout:
    """Where a pyramid's cells (x, y, width, height) and free areas lie."""

    size: tuple
    top: tuple
    left_cell: tuple
    right_cell: tuple
    left: tuple
    right: tuple


def pyramid_layout(size, aspect: float) -> PyramidLayout:
    """The pyramid on a frame of ``size`` for tiles of ``aspect`` (width / height).

    The two lower cells fill the frame's width, as large as the frame allows;
    the top cell, of the same size, is centred on top of them and touches them.

    """
    width, height = size
    cell_h = min(
        math.floor((width - SEPARATOR) // 2 / aspect), (height - SEPARATOR) // 2
    )
    cell_w = min((width - SEPARATOR) // 2, math.floor(cell_h * aspect))
    lower_y = height - cell_h
    lower_x = (width - 2 * cell_w - SEPARATOR) // 2
    top_h = lower_y - SEPARATOR
    top_x = (width - cell_w) // 2
    right_x = top_x + cell_w + SEPARATOR
    return PyramidLayout(
        size=(width, height),
        top=(top_x, top_h - cell_h, cell_w, cell_h),
        left_cell=(lower_x, lower_y, cell_w, cell_h),
        right_cell=(lower_x + cell_w + SEPARATOR, lower_y, cell_w, cell_h),
        left=(0, 0, max(0, top_x - SEPARATOR), top_h),
        right=(right_x, 0, max(0, width - right_x), top_h),
    )


def compose_pyramid(
    tiles: Sequence[tuple],
    names: Sequence[str],
    layout: PyramidLayout,
    *,
    lines: Sequence[str],
    style: FigureStyle,
    graph: int | None,
    scale: tuple[float, str] | None,
    normalization: str = "linear",
    over_range: str | None = None,
) -> np.ndarray:
    """One pyramid frame; BGR pixels.

    Parameters
    ----------
    tiles : sequence
        The top, lower-left and lower-right panel, each as (BGR tile or None,
        held, note). A held tile is greyed and its note says why.
    names : sequence of str
        The panels' names: the top one heads the right area, the lower ones are
        written in their cells' corner.
    layout : PyramidLayout
        Where the cells lie.
    lines : sequence of str
        The text left of the top panel, largest first.
    style : FigureStyle
        Colours, and the curves of the graph's key.
    graph : int, optional
        Which panel (0 top, 1 lower left, 2 lower right) is a mass graph. Its key
        goes under the top panel's name, or in a lower graph's upper right
        corner.
    scale : tuple, optional
        The maps' colour scale (maximum, unit), whose key goes in the right
        area's lower corner, above the lower right panel.

    """
    width, height = layout.size
    frame = np.empty((height, width, 3), np.uint8)
    frame[...] = bgr(style.frame)
    font = max(11.0, height * FONT_SHARE)
    pad = max(4, round(6 * height / 1080))
    cells = (layout.top, layout.left_cell, layout.right_cell)
    for number, (cell, (tile, is_held, note)) in enumerate(zip(cells, tiles)):
        x, y, cell_w, cell_h = cell
        _place(frame, cell, tile, is_held, bgr(style.empty))
        if number > 0:
            blend(frame, text_image(names[number], font), x + pad, y + pad)
        if note:
            image = text_image(note, font * 0.9, backing=0.7, max_px=cell_w - 2 * pad)
            blend(frame, image, x + pad, y + cell_h - image.shape[0] - pad)
        if number > 0 and number == graph:
            key = graph_key(round(font, 1), style)
            blend(frame, key, x + cell_w - key.shape[1] - pad, y + pad)
    margin = round(MARGIN_PX * height / 1080)
    left_x, left_y, left_w, left_h = layout.left
    y = left_y + margin
    for number, text in enumerate(lines):
        size = round(font * LINE_SIZES[min(number, len(LINE_SIZES) - 1)], 1)
        image = text_image(
            text, size, bold=number < 2, backing=0.0, max_px=left_w - 2 * margin
        )
        if y + image.shape[0] > left_y + left_h:
            break
        blend(frame, image, left_x + margin, y)
        y += image.shape[0] + round(font * (0.5 if number < 2 else 0.25))
    right_x, right_y, right_w, right_h = layout.right
    heading = text_image(
        names[0], font, bold=True, backing=0.0, max_px=right_w - 2 * margin
    )
    blend(frame, heading, right_x + margin, right_y + margin)
    below = right_y + margin + heading.shape[0]
    if graph == 0:
        columns = 1 if right_w < 40 * font else len(style.curves)
        key = graph_key(round(font, 1), style, columns)
        below += round(font * 0.4)
        blend(frame, key, right_x + margin, below)
        below += key.shape[0]
    room = right_y + right_h - below - 2 * margin
    bar_h = int(min(0.55 * layout.right_cell[3], room))
    if scale is not None and bar_h >= 4 * font:
        bar = colour_key(
            float(scale[0]),
            scale[1],
            bar_h,
            font,
            style.colour_map,
            normalization,
            over_range,
        )
        blend(
            frame,
            bar,
            right_x + right_w - bar.shape[1] - margin,
            right_y + right_h - bar.shape[0] - margin,
        )
    return frame


@dataclass(frozen=True)
class GridLayout:
    """Where each tile of a grid lies, and the column or band of its title."""

    size: tuple
    columns: int
    rows: int
    tile: tuple
    cells: tuple
    side: tuple


def grid_layout(count: int, aspect: float, size) -> GridLayout:
    """The grid that shows ``count`` tiles of ``aspect`` (width / height) largest
    on a frame of ``size``, beside a title column on the right or a band below.

    Every column count is tried with both; the largest tile wins.

    """
    width, height = (int(value) for value in size)
    best = None
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        for side in ("right", "bottom"):
            room_w = width - (math.ceil(GRID_SIDE[0] * width) if side == "right" else 0)
            room_h = height - (
                math.ceil(GRID_SIDE[1] * height) if side == "bottom" else 0
            )
            tile_w = min(
                (room_w - SEPARATOR * (columns - 1)) / columns,
                (room_h - SEPARATOR * (rows - 1)) / rows * aspect,
            )
            tile_w = math.floor(tile_w)
            tile_h = math.floor(tile_w / aspect)
            if tile_w < 8 or tile_h < 8:
                continue
            if best is None or tile_w * tile_h > best[0]:
                best = (tile_w * tile_h, columns, rows, tile_w, tile_h, side)
    if best is None:
        raise ValueError("The tiles do not fit this frame")
    _area, columns, rows, tile_w, tile_h, side = best
    grid_w = columns * tile_w + SEPARATOR * (columns - 1)
    grid_h = rows * tile_h + SEPARATOR * (rows - 1)
    if side == "right":
        x0, y0 = 0, (height - grid_h) // 2
        side_area = (grid_w + SEPARATOR, 0, width - grid_w - SEPARATOR, height)
    else:
        x0, y0 = (width - grid_w) // 2, 0
        side_area = (0, grid_h + SEPARATOR, width, height - grid_h - SEPARATOR)
    cells = tuple(
        (
            x0 + (number % columns) * (tile_w + SEPARATOR),
            y0 + (number // columns) * (tile_h + SEPARATOR),
            tile_w,
            tile_h,
        )
        for number in range(count)
    )
    return GridLayout(
        size=(width, height),
        columns=columns,
        rows=rows,
        tile=(tile_w, tile_h),
        cells=cells,
        side=side_area,
    )


def compose_grid(
    tiles: Mapping[str, tuple],
    layout: GridLayout,
    *,
    lines: Sequence[str],
    style: FigureStyle,
    graph: bool,
    scale: tuple[float, str] | None,
    normalization: str = "linear",
    over_range: str | None = None,
) -> np.ndarray:
    """One grid frame: every run's tile in its cell, named in its corner, and the
    title column; BGR pixels.

    ``tiles`` maps each run, in the cells' order, to (BGR tile or None, held,
    note). ``lines`` are the title column's lines, title first; under them the
    key: the graph's (``graph``) or the maps' colour scale (``scale``: maximum and
    unit).

    """
    width, height = layout.size
    frame = np.empty((height, width, 3), np.uint8)
    frame[...] = bgr(style.frame)
    label_px = max(10.0, layout.tile[1] * 0.075)
    for (run, (tile, is_held, note)), cell in zip(tiles.items(), layout.cells):
        x, y, cell_w, cell_h = cell
        _place(frame, cell, tile, is_held, bgr(style.empty))
        blend(frame, text_image(str(run).upper(), label_px, bold=True), x + 4, y + 4)
        if note:
            image = text_image(note, label_px * 0.8, backing=0.7, max_px=cell_w - 8)
            blend(frame, image, x + 4, y + cell_h - image.shape[0] - 4)
    side_x, side_y, side_w, side_h = layout.side
    title_px = round(max(14.0, min(side_w * 0.07, height * 0.022)), 1)
    margin = round(title_px * 0.5)
    y = side_y + round(title_px)
    for number, text in enumerate(lines):
        size = title_px if number == 0 else round(title_px * 0.62, 1)
        image = text_image(
            text, size, bold=number == 0, backing=0.0, max_px=side_w - 2 * margin
        )
        blend(frame, image, side_x + margin, y)
        y += image.shape[0] + round(title_px * (0.6 if number == 0 else 0.3))
    key_px = round(title_px * 0.62, 1)
    key = None
    if scale is not None:
        bar_h = int(min(side_h * 0.35, height * 0.35))
        key = colour_key(
            float(scale[0]),
            scale[1],
            bar_h,
            key_px,
            style.colour_map,
            normalization,
            over_range,
        )
    elif graph:
        key = graph_key(key_px, style)
    if key is not None:
        blend(frame, key, side_x + margin, y + margin)
    return frame
