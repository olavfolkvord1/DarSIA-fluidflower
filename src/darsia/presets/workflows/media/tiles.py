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
import math
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
    mass_per_area,
    pooled_mean,
    shown_values,
)
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    graph_tile,
    image_tile,
    map_tile,
    outline_masks,
    overlay_outlines,
    photograph,
    shared_axes,
    tile_size,
)
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.media.transfer import (
    area_map,
    divided_size,
    stretched_photo,
)

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
    graph_axes: tuple[tuple[float, float], tuple[float, float]] | None = None
    """The graphs' time [h] and mass [g] limits, or None (by default, and in a
    table without them) for limits shared over the runs (:func:`shared_axes`)."""
    mass_quantity: Literal["pore-volume", "area"] = "pore-volume"
    """Mass density per pore volume [g/L], or mass per measured area [g/m²]."""
    normalization: Literal["linear", "sqrt"] = "linear"
    """Fixed colour normalization; the scale and ticks keep the physical units."""
    grid_divisor: int | None = None
    """Optional integer grid reduction with area interpolation and exact panel size."""

    def __post_init__(self):
        """Refuse invalid fixed scales, quantities and normalizations."""
        if not math.isfinite(self.colour_scale) or self.colour_scale <= 0:
            raise ValueError("colour_scale must be finite and positive")
        if self.mass_quantity not in ("pore-volume", "area"):
            raise ValueError("mass_quantity must be pore-volume or area")
        if self.normalization not in ("linear", "sqrt"):
            raise ValueError("normalization must be linear or sqrt")
        if self.grid_divisor is not None:
            divided_size((self.grid_divisor, self.grid_divisor), self.grid_divisor)
            if self.mass_quantity != "area" or self.normalization != "sqrt":
                raise ValueError(
                    "Grid-divided maps require area quantity and sqrt normalization"
                )


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
    swi: np.ndarray | None = None
    """Irreducible water saturation per pixel, required for a gas contour."""


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
    rules: np.ndarray | None = None
    """Stored phase provenance, including explicitly allocated hidden gas bridges."""


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
    photograph_limits: tuple[float, float] | None = None,
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
    size = (
        tile_size(frame.box, settings.width_px)
        if settings.grid_divisor is None
        else divided_size((frame.box[3], frame.box[2]), settings.grid_divisor)
    )
    step = display_step(frame.shape, settings.display_cells)
    if panel.content == IMAGE:
        direct = photograph_settings.transfer == "first-frame"
        if direct:
            if photograph_limits is None:
                raise ValueError(
                    "First-frame transfer requires the run's fixed photograph limits"
                )
            photo = stretched_photo(
                frame.image,
                size,
                photograph_limits,
                photograph_settings.gamma,
                photograph_settings.invalid_grey,
            )
        else:
            photo = photograph(frame.image[::step, ::step], photograph_settings)
        outlines, drawn = [], {}
        if photograph_settings.co2_outline or photograph_settings.gas_contour:
            mass_area = None
            if (
                photograph_settings.co2_source == "mass"
                and photograph_settings.co2_outline
            ):
                if frame.maps is None or frame.sand is None or frame.unit != "g/L":
                    raise ValueError(
                        "Mass outlines require stored g/L mass and measured geometry"
                    )
                mass_area = mass_per_area(
                    frame.maps["mass"], frame.sand.pore_volume, frame.sand.pixel_area_m2
                )
            co2, gas = outline_masks(
                frame.phase,
                frame.saturation,
                None if frame.sand is None else frame.sand.swi,
                photograph_settings,
                rules=frame.rules,
                mass_area=mass_area,
            )
            outlines = [
                (mask, colour)
                for mask, colour in (
                    (co2, photograph_settings.co2_colour),
                    (gas, photograph_settings.gas_colour),
                )
                if mask is not None
            ]
            drawn = dict(
                co2_outline=photograph_settings.co2_outline,
                gas_contour=photograph_settings.gas_contour,
                gas_contour_fraction=photograph_settings.gas_contour_fraction,
                co2_colour=photograph_settings.co2_colour,
                gas_colour=photograph_settings.gas_colour,
            )
            if photograph_settings.co2_source != "phase":
                drawn["co2_source"] = photograph_settings.co2_source
        if direct:
            drawn.update(
                photograph_transfer="first-frame",
                photograph_limits=list(photograph_limits),
                photograph_percentiles=list(photograph_settings.percentiles),
                photograph_gamma=photograph_settings.gamma,
            )
            return (
                overlay_outlines(
                    photo,
                    frame.box,
                    outlines,
                    photograph_settings.outline_width_pt,
                    reduced_occupancy=True,
                ),
                drawn,
            )
        return (
            image_tile(
                photo,
                frame.box,
                step,
                size,
                outlines=outlines,
                outline_width_pt=photograph_settings.outline_width_pt,
            ),
            drawn,
        )
    if panel.content == GRAPH:
        room = LABEL_ROOM if label_room else 0.0
        tile = graph_tile(rows, frame.case, size, axes=axes, style=style, top_room=room)
        return tile, dict(label_room=label_room)
    values = shown_values(frame.maps[panel.content], frame.phase)
    drawn = dict(display_mean=panel.display_mean)
    if settings.grid_divisor is not None and panel.display_mean != "none":
        raise ValueError(
            "Grid-divided maps use direct area interpolation without a depth mean"
        )
    if settings.grid_divisor is not None:
        if frame.sand is None or frame.unit != "g/L":
            raise ValueError(
                "Grid-divided maps require stored g/L mass and measured geometry"
            )
        full = shown_values(
            mass_per_area(values, frame.sand.pore_volume, frame.sand.pixel_area_m2),
            frame.phase,
        )
        x, y, width, height = frame.box
        shown_box = np.s_[y : y + height, x : x + width]
        tile, shown, known = area_map(
            full[shown_box],
            frame.sand.pore_volume[shown_box],
            size,
            settings.colour_scale,
            colour_map=style.colour_map,
            no_value=style.no_value,
            over_range=style.over_range,
        )
        count = int(known.sum())
        clipped = int(np.count_nonzero(known & (shown > settings.colour_scale)))
        drawn.update(
            mass_quantity="area",
            normalization="sqrt",
            colour_scale=settings.colour_scale,
            unit="g/m²",
            over_range=style.over_range,
            known_display_cells=count,
            clipped_display_cells=clipped,
            clipped_share=None if count == 0 else clipped / count,
            grid_divisor=settings.grid_divisor,
            logical_panel_size=list(size),
        )
        return tile, drawn
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
        drawn.update(window_px=window_px, window_mm=window_mm)
        if settings.mass_quantity == "pore-volume":
            shown = pooled_mean(full, sand.pore_volume, step)
    else:
        shown = block_mean(values, step)
        full = values
    unit = frame.unit
    if settings.mass_quantity == "area":
        if frame.sand is None:
            raise ValueError(
                "Mass per area requires the measured pore and pixel geometry"
            )
        if frame.unit != "g/L":
            raise ValueError("Mass per area requires stored mass density in g/L")
        full = shown_values(
            mass_per_area(full, frame.sand.pore_volume, frame.sand.pixel_area_m2),
            frame.phase,
        )
        shown = pooled_mean(full, frame.sand.pixel_area_m2, step)
        unit = "g/m²"
    enhanced = settings.mass_quantity == "area" or settings.normalization != "linear"
    if enhanced:
        x, y, width, height = frame.box
        visible = shown[
            y // step : math.ceil((y + height) / step),
            x // step : math.ceil((x + width) / step),
        ]
        known = np.isfinite(visible)
        count = int(known.sum())
        clipped = int(np.count_nonzero(known & (visible > settings.colour_scale)))
        drawn.update(
            mass_quantity=settings.mass_quantity,
            normalization=settings.normalization,
            colour_scale=settings.colour_scale,
            unit=unit,
            over_range=style.over_range,
            known_display_cells=count,
            clipped_display_cells=clipped,
            clipped_share=None if count == 0 else clipped / count,
        )
    tile = map_tile(
        shown,
        frame.box,
        step,
        size,
        scale=settings.colour_scale,
        label=panel.label,
        unit=unit,
        style=style,
        normalization=settings.normalization,
        show_over_range=enhanced,
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
