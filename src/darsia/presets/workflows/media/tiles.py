"""A run's tiles: each chosen panel of each frame drawn once from the stored results.

:func:`draw_tile` draws one panel of one frame (:class:`Frame`) as its settings
choose (:class:`Panel`, :class:`TileSettings`): the corrected image as a
photograph, a mass map with its display mean, or the mass graph. It also says how
the tile was drawn: a map's display mean and its window, a graph's room for a grid
film's label (:data:`LABEL_ROOM`). :func:`save_tile` saves a tile as a PNG with
that, and :func:`write_tile_index` lists a folder's tiles for the films
(:mod:`films`).

"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from darsia.presets.workflows.media.display_mean import (
    block_mean,
    depth_scaled,
    display_step,
    pooled_mean,
    shown_values,
)
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    graph_tile,
    image_tile,
    map_tile,
    photograph,
    shared_axes,
    tile_size,
)
from darsia.presets.workflows.media.style import FigureStyle

IMAGE, GRAPH = "image", "graph"
"""Panel contents besides mass maps: the corrected image and the mass graph."""
DISPLAY_MEANS = ("none", "depth")
"""A map's display means (:class:`Panel`)."""
LABEL_ROOM = 0.18
"""Share of a grid's graph tile kept free above the axes for the run's label."""
TILE_INDEX = "tiles.json"
"""The file that lists a folder's tiles."""


@dataclass(frozen=True)
class Panel:
    """One panel of a film: what it shows and how it is named."""

    content: str
    """"image" (the corrected image), "graph" (the mass graph) or a mass map, by
    the name of its field."""
    label: str
    """Its name in the film."""
    display_mean: Literal["none", "depth"]
    """A map's display mean: only the mean of each shown cell, or first the
    depth-scaled mean inside each sand layer (:mod:`display_mean`); "none" for
    the image and the graph."""

    def __post_init__(self):
        """Refuse a display mean that is not one of :data:`DISPLAY_MEANS`, and a
        depth-scaled image or graph."""
        if self.display_mean not in DISPLAY_MEANS:
            raise ValueError(f"The display mean is one of {DISPLAY_MEANS}")
        if self.display_mean != "none" and self.content in (IMAGE, GRAPH):
            raise ValueError(f"The {self.content} has no display mean")


@dataclass(frozen=True)
class TileSettings:
    """What every tile of a set of films shares; one table."""

    width_px: int
    """Every tile's width; its height follows the shown part of the grid."""
    display_cells: int
    """The display grid's longest side, in cells (:func:`display_step`)."""
    colour_scale: float
    """The top of every map's fixed colour scale, in the maps' unit."""
    graph_axes: tuple[tuple[float, float], tuple[float, float]] | None
    """The graphs' time [h] and mass [g] limits, or None for limits shared over
    the runs (:func:`shared_axes`)."""


@dataclass(frozen=True)
class SandGeometry:
    """A run's sand layers and pore space on its grid, for the depth-scaled mean."""

    layers: np.ndarray
    """The placed sand-map label of every pixel."""
    sand_layers: Sequence[int]
    """The labels of sand layers."""
    pore_volume: np.ndarray
    """Each pixel's pore volume."""
    depth_m: np.ndarray
    """The rig's depth at each pixel."""
    pixel_area_m2: np.ndarray
    """Each pixel's area."""


@dataclass(frozen=True)
class Frame:
    """What a frame's tiles are drawn from: its stored results as arrays."""

    case: str
    """The frame."""
    elapsed_h: float
    """Its elapsed time [h]."""
    target_h: float | None
    """Its production target time [h]; None for a frame without one."""
    shape: tuple[int, int]
    """The run's grid (rows, columns)."""
    box: tuple[int, int, int, int]
    """Where the corrected image lies in the grid: x, y, width and height."""
    image: np.ndarray | None = None
    """The corrected image of the box: rows x columns x camera RGB, linear."""
    maps: Mapping[str, np.ndarray] | None = None
    """The mass maps, by field."""
    unit: str = ""
    """The mass maps' unit."""
    phase: np.ndarray | None = None
    """The phase map."""
    saturation: np.ndarray | None = None
    """The free-gas saturation, for the depth-scaled mean."""
    sand: SandGeometry | None = None
    """The sand layers and pore space, for the depth-scaled mean."""


def graph_axes(
    settings: TileSettings, runs: Sequence[Sequence[Mapping]], style: FigureStyle
) -> tuple:
    """The graphs' limits: the settings' fixed ones, else those shared over the
    mass records of ``runs`` (:func:`shared_axes`)."""
    return settings.graph_axes or shared_axes(runs, style)


def draw_tile(
    panel: Panel,
    frame: Frame,
    settings: TileSettings,
    *,
    style: FigureStyle,
    photograph_settings: PhotographSettings | None = None,
    rows: Sequence[Mapping] = (),
    axes=None,
    label_room: bool = False,
) -> tuple[np.ndarray, dict]:
    """One panel of one frame as a tile, and how it was drawn.

    Parameters
    ----------
    panel : Panel
        What the tile shows; a map with its display mean.
    frame : Frame
        The frame's stored results.
    settings : TileSettings
        The tile's width, the display grid and the colour scale.
    style : FigureStyle
        Colours, and the curves of a graph.
    photograph_settings : PhotographSettings, optional
        How the corrected image is shown (image panels).
    rows : sequence of mapping, optional
        The run's mass records, one per frame (graph panels).
    axes : tuple, optional
        The graph's limits (:func:`graph_axes`).
    label_room : bool
        Whether a graph keeps room for a grid film's label (:data:`LABEL_ROOM`).

    Returns
    -------
    tuple
        The tile (RGB pixels) and how it was drawn: a map's ``display_mean`` (with
        a depth-scaled mean, its ``window_px`` and ``window_mm``, the median rig
        depth), or a graph's ``label_room``.

    """
    size = tile_size(frame.box, settings.width_px)
    step = display_step(frame.shape, settings.display_cells)
    if panel.content == IMAGE:
        photo = photograph(frame.image[::step, ::step], photograph_settings)
        return image_tile(photo, frame.box, step, size), {}
    if panel.content == GRAPH:
        room = LABEL_ROOM if label_room else 0.0
        tile = graph_tile(rows, frame.case, size, axes=axes, style=style, top_room=room)
        return tile, dict(label_room=label_room)
    values = shown_values(frame.maps[panel.content], frame.phase)
    drawn = dict(display_mean=panel.display_mean)
    if panel.display_mean == "depth":
        sand = frame.sand
        full, window_px, window_mm = depth_scaled(
            values,
            frame.phase,
            frame.saturation,
            sand.layers,
            sand.sand_layers,
            sand.pore_volume,
            sand.depth_m,
            sand.pixel_area_m2,
        )
        shown = pooled_mean(full, sand.pore_volume, step)
        drawn.update(window_px=window_px, window_mm=window_mm)
    else:
        shown = block_mean(values, step)
    tile = map_tile(
        shown,
        frame.box,
        step,
        size,
        scale=settings.colour_scale,
        label=panel.label,
        unit=frame.unit,
        style=style,
    )
    return tile, drawn


def tile_name(case: str, target_h: float | None) -> str:
    """A tile's file name: its frame and production target time."""
    target = "none" if target_h is None else f"{float(target_h):08.3f}h"
    return f"{case}__t{target}.png"


def save_tile(folder: Path, frame: Frame, tile: np.ndarray, drawn: Mapping) -> dict:
    """Save a frame's tile (RGB pixels) in ``folder``; its row.

    The row holds the frame, its elapsed and production target time [h], the file,
    the tile's size (width, height) and how it was drawn (``drawn``, from
    :func:`draw_tile`). Rows are listed by :func:`write_tile_index`.

    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = tile_name(frame.case, frame.target_h)
    if not cv2.imwrite(str(folder / name), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write the tile {name}")
    return dict(
        case_id=frame.case,
        elapsed_h=frame.elapsed_h,
        target_h=frame.target_h,
        file=name,
        size=[tile.shape[1], tile.shape[0]],
        **drawn,
    )


def write_tile_index(folder: Path, rows: Sequence[Mapping]) -> None:
    """List a folder's tiles, in frame order, as :data:`TILE_INDEX`."""
    text = json.dumps(dict(tiles=list(rows)), indent=1, allow_nan=False)
    (Path(folder) / TILE_INDEX).write_text(text + "\n", encoding="utf-8")


def read_tile_index(folder: Path) -> list[dict]:
    """A folder's tiles as their rows, each with its file's ``path``."""
    folder = Path(folder)
    rows = json.loads((folder / TILE_INDEX).read_text(encoding="utf-8"))["tiles"]
    return [dict(row, path=str(folder / row["file"])) for row in rows]
