"""Films and figures of stored results, with chosen panels, layout and runs.

Films are composed from tiles, each panel of each frame drawn once (:mod:`tiles`).
A run's image and map tiles serve both its pyramid film (:func:`write_pyramid_film`)
and its cell in a grid film over many runs (:func:`write_grid_film`); a grid's
graph tiles keep room for the run's label, so pyramids and grids have graph tiles
of their own. A film setting makes either film (:func:`write_films`), and a film
uses only tiles drawn as its panels choose.

A film has one frame per target time of its tiles. A panel without a tile at a
target shows its last earlier tile, greyed, and says which time it shows; before
its first tile it is empty and says so. A depth-scaled map's caption gives its
window, from its tiles. Every film is written as MP4, named by its setting, with
its last frame as a figure (PNG); a film never replaces another's files.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from natsort import natsort_keygen, ns

from darsia.presets.workflows.media.layouts import (
    compose_grid,
    compose_pyramid,
    grid_layout,
    pyramid_layout,
)
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.media.tiles import GRAPH, IMAGE, Panel, TileSettings
from darsia.presets.workflows.media.video import MovieWriter

LAYOUTS = {"pyramid": 3, "grid": 1}
"""The layouts and how many panels each shows."""
DEPTH_NOTE = (
    "Display only: pore-volume-weighted water mean within each sand layer over {}"
    " (the frame's median rig depth); gas unchanged"
)
"""The caption of a depth-scaled map (:mod:`display_mean`), with its window."""
RUN_ORDER = natsort_keygen(alg=ns.IGNORECASE)
"""Runs in natural order of their names (run 9 before run 10)."""


@dataclass(frozen=True)
class FilmSettings:
    """One film; one table."""

    name: str
    """The film's name: its files are ``<name>_<run>`` for a pyramid and ``<name>``
    for a grid."""
    layout: Literal["pyramid", "grid"]
    """A run's three panels as a pyramid, or one panel of many runs as a grid."""
    panels: tuple[Panel, ...]
    """A pyramid's top, lower-left and lower-right panel, or a grid's panel."""
    size_px: tuple[int, int]
    """The frame's width and height (even numbers)."""
    fps: float
    """Frames per second."""
    codec: str
    """The video codec, a four-character code such as "avc1" (H.264)."""
    note: str
    """The maps' source, written in every frame."""

    def __post_init__(self):
        """Refuse a layout that is not one of :data:`LAYOUTS`, or another number of
        panels than it shows."""
        if self.layout not in LAYOUTS:
            raise ValueError(f"A film's layout is one of {tuple(LAYOUTS)}")
        if len(self.panels) != LAYOUTS[self.layout]:
            raise ValueError(f"A {self.layout} shows {LAYOUTS[self.layout]} panels")


def film_targets(sets: Sequence[Sequence[Mapping]]) -> list[float]:
    """The target times [h] of the tiles of all ``sets``, sorted, each once."""
    return sorted(
        {
            round(float(row["target_h"]), 6)
            for rows in sets
            for row in rows
            if row["target_h"] is not None
        }
    )


def tile_at(rows: Sequence[Mapping], target: float) -> tuple[Mapping | None, bool]:
    """The tile a film shows at ``target`` and whether it is held: the tile of that
    target, else the last earlier one (held), else none (held)."""
    timed = [row for row in rows if row["target_h"] is not None]
    exact = [row for row in timed if abs(float(row["target_h"]) - target) < 1e-6]
    if exact:
        return exact[0], False
    earlier = [row for row in timed if float(row["target_h"]) < target]
    if not earlier:
        return None, True
    return max(earlier, key=lambda row: float(row["target_h"])), True


def _reader(keep: int):
    """A reader of tile PNGs as BGR pixels of a size, keeping ``keep`` of them."""

    @lru_cache(maxsize=keep)
    def read(path: str, size: tuple) -> np.ndarray:
        """The tile at ``path`` scaled to ``size`` (width, height)."""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read the tile {path}")
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    return read


def _choice(rows, target: float, size, read) -> tuple:
    """What a panel shows at ``target``: (BGR tile or None, held, note, row)."""
    row, is_held = tile_at(rows, target)
    if row is None:
        return None, True, f"no image before {target:g} h", None
    tile = read(row["path"], tuple(size))
    shown = float(row["target_h"])
    note = f"no image at {target:.2f} h · showing {shown:.2f} h" if is_held else ""
    return tile, is_held, note, row


def _aspect(sets) -> float:
    """The width / height of the tiles, by the first one."""
    width, height = next(row["size"] for rows in sets for row in rows)
    return width / height


def _drawn_as_chosen(panel: Panel, sets, film: FilmSettings) -> None:
    """Refuse tiles of ``panel`` drawn otherwise than it chooses: a map's display
    mean, a graph's room for a grid's label."""
    if panel.content == IMAGE:
        return
    key, wanted = (
        ("label_room", film.layout == "grid")
        if panel.content == GRAPH
        else ("display_mean", panel.display_mean)
    )
    if any(row.get(key) != wanted for rows in sets for row in rows):
        raise ValueError(f"The {panel.label} tiles were drawn with another {key}")


def _note(film: FilmSettings, panels: Sequence[Panel], rows) -> str:
    """The maps' source as a frame writes it, with the caption of a depth-scaled map:
    the window of the tiles shown (``rows``, per panel)."""
    depth = [panel.display_mean == "depth" for panel in panels]
    shown = [row for row, deep in zip(rows, depth) if deep and row is not None]
    windows = sorted({f"{row['window_mm']:.1f}" for row in shown}, key=float)
    caption = ""
    if windows:
        span = " to ".join(dict.fromkeys((windows[0], windows[-1])))
        caption = DEPTH_NOTE.format(span + " mm")
    return "; ".join(part for part in (film.note, caption) if part)


def _shown(choices) -> list[dict]:
    """The frame, time and hold of each panel's tile, for a film's record."""
    return [
        dict(
            case_id=None if row is None else row["case_id"],
            elapsed_h=None if row is None else row["elapsed_h"],
            held=is_held,
        )
        for _tile, is_held, _note, row in choices
    ]


def _free(output: Path, stem: str) -> None:
    """Refuse to write a film where a film's files already are."""
    for name in (f"{stem}.mp4", f"{stem}.png"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} is another film's")


def _figure(output: Path, stem: str, frame: np.ndarray) -> str:
    """Write a film's last frame as its figure; the file's name."""
    if not cv2.imwrite(str(output / f"{stem}.png"), frame):
        raise OSError(f"Could not write the figure {stem}.png")
    return f"{stem}.png"


def pyramid_frame(
    run: str,
    tiles: Sequence[Sequence[Mapping]],
    target: float,
    number: int,
    count: int,
    *,
    film: FilmSettings,
    tile_settings: TileSettings,
    unit: str,
    style: FigureStyle,
    read=None,
) -> tuple[np.ndarray, list[tuple], str]:
    """Frame ``number`` of ``count`` of a run's pyramid at ``target``.

    ``tiles`` are the rows of the film's three panels (:func:`read_tile_index`).
    The text names the run, the elapsed time of the frame shown (of the first
    lower, then the top panel not held), the target and frame, and the maps'
    source. Returns the BGR frame, each panel's choice (tile, held, note, row) and
    the maps' source as written.

    """
    read = read or _reader(8)
    layout = pyramid_layout(film.size_px, _aspect(tiles))
    choices = [_choice(rows, target, layout.left_cell[2:], read) for rows in tiles]
    shown = next(
        (row for _t, is_held, _n, row in (*choices[1:], choices[0]) if not is_held),
        None,
    )
    elapsed = (
        "Elapsed time not recorded"
        if shown is None or shown["elapsed_h"] is None
        else f"Elapsed {float(shown['elapsed_h']):.2f} h"
    )
    lines = [run.upper(), elapsed, f"Target {target:.2f} h · frame {number} of {count}"]
    note = _note(film, film.panels, [choice[3] for choice in choices])
    if note:
        lines.append("Maps: " + note)
    if any(is_held for _t, is_held, _n, _r in choices):
        lines.append(
            "Grey panel: no image at this target; its last earlier image is held."
        )
    maps = [panel for panel in film.panels if panel.content not in (IMAGE, GRAPH)]
    names = [
        panel.label + (f" ({unit})" if panel in maps else "") for panel in film.panels
    ]
    graph = next(
        (n for n, panel in enumerate(film.panels) if panel.content == GRAPH), None
    )
    frame = compose_pyramid(
        [choice[:3] for choice in choices],
        names,
        layout,
        lines=lines,
        style=style,
        graph=graph,
        scale=(tile_settings.colour_scale, unit) if maps else None,
    )
    return frame, choices, note


def write_pyramid_film(
    run: str,
    tiles: Sequence[Sequence[Mapping]],
    output: Path,
    *,
    film: FilmSettings,
    tile_settings: TileSettings,
    unit: str,
    style: FigureStyle,
    lock=None,
) -> dict:
    """A run's pyramid film from the tiles of its three panels.

    One frame per target time of any panel (:func:`pyramid_frame`). Writes
    ``<name>_<run>.mp4`` and its last frame as ``<name>_<run>.png`` in ``output``;
    ``lock`` is held while the video encoder opens (:class:`MovieWriter`).

    Returns
    -------
    dict
        The film's name, run and file names, its decoded frame count, rate and
        size, and per frame its target time, each panel's frame, elapsed time and
        hold, and the maps' source as written.

    Raises
    ------
    FileExistsError
        Where the film's files already exist in ``output``.
    ValueError
        Where a panel's tiles were drawn otherwise than it chooses.

    """
    output = Path(output)
    for panel, rows in zip(film.panels, tiles):
        _drawn_as_chosen(panel, [rows], film)
    targets = film_targets(tiles)
    if not targets:
        raise ValueError(f"Run {run} has no tiles with target times")
    stem = f"{film.name}_{run}"
    _free(output, stem)
    read = _reader(24)
    writer = MovieWriter(
        output / f"{stem}.mp4", film.size_px, film.fps, film.codec, lock=lock
    )
    frames = []
    for number, target in enumerate(targets, 1):
        frame, choices, note = pyramid_frame(
            run,
            tiles,
            target,
            number,
            len(targets),
            film=film,
            tile_settings=tile_settings,
            unit=unit,
            style=style,
            read=read,
        )
        writer.write(frame)
        frames.append(dict(target_h=target, panels=_shown(choices), note=note))
    decoded = writer.close()
    figure = _figure(output, stem, frame)
    return dict(
        name=film.name,
        run=run,
        movie=f"{stem}.mp4",
        figure=figure,
        frames=frames,
        **decoded,
    )


def grid_frame(
    tiles: Mapping[str, Sequence[Mapping]],
    target: float,
    number: int,
    count: int,
    *,
    film: FilmSettings,
    tile_settings: TileSettings,
    unit: str,
    style: FigureStyle,
    read=None,
) -> tuple[np.ndarray, dict, str]:
    """Frame ``number`` of ``count`` of a grid film at ``target``.

    ``tiles`` maps each run to the rows of the film's panel; the runs are shown
    by name. The title column names the panel (with a map's colour scale), the
    target and frame, the runs and the maps' source. Returns the BGR frame, each
    run's choice (tile, held, note, row) and the maps' source as written.

    """
    read = read or _reader(4 * len(tiles))
    runs = sorted(tiles, key=RUN_ORDER)
    layout = grid_layout(len(runs), _aspect(tiles.values()), film.size_px)
    choices = {run: _choice(tiles[run], target, layout.tile, read) for run in runs}
    (panel,) = film.panels
    is_map = panel.content not in (IMAGE, GRAPH)
    scale = tile_settings.colour_scale
    note = _note(film, [panel] * len(runs), [choice[3] for choice in choices.values()])
    lines = [
        panel.label + (f" · 0 to {scale:g} {unit}" if is_map else ""),
        f"Target {target:.2f} h · frame {number} of {count}",
        f"{len(runs)} runs, by name" + (f" · maps: {note}" if note else ""),
        "Grey: no image at this target; the run's last image is held.",
    ]
    frame = compose_grid(
        {run: choice[:3] for run, choice in choices.items()},
        layout,
        lines=lines,
        style=style,
        graph=panel.content == GRAPH,
        scale=(scale, unit) if is_map else None,
    )
    return frame, choices, note


def write_grid_film(
    tiles: Mapping[str, Sequence[Mapping]],
    output: Path,
    *,
    film: FilmSettings,
    tile_settings: TileSettings,
    unit: str,
    style: FigureStyle,
    lock=None,
) -> dict:
    """A grid film of one panel over many runs, on the runs' shared target times.

    One frame per target time of any run (:func:`grid_frame`). Writes
    ``<name>.mp4`` and its last frame as ``<name>.png`` in ``output``; ``lock`` is
    held while the video encoder opens (:class:`MovieWriter`).

    Returns
    -------
    dict
        The film's name and file names, its decoded frame count, rate and size,
        and per frame its target time, each run's frame, elapsed time and hold, and
        the maps' source as written.

    Raises
    ------
    FileExistsError
        Where the film's files already exist in ``output``.
    ValueError
        Where the panel's tiles were drawn otherwise than it chooses.

    """
    output = Path(output)
    _drawn_as_chosen(film.panels[0], tiles.values(), film)
    targets = film_targets(list(tiles.values()))
    if not targets:
        raise ValueError("A grid film needs tiles with target times")
    _free(output, film.name)
    read = _reader(4 * len(tiles))
    writer = MovieWriter(
        output / f"{film.name}.mp4", film.size_px, film.fps, film.codec, lock=lock
    )
    frames = []
    for number, target in enumerate(targets, 1):
        frame, choices, note = grid_frame(
            tiles,
            target,
            number,
            len(targets),
            film=film,
            tile_settings=tile_settings,
            unit=unit,
            style=style,
            read=read,
        )
        writer.write(frame)
        shown = dict(zip(choices, _shown(choices.values())))
        frames.append(dict(target_h=target, runs=shown, note=note))
    decoded = writer.close()
    figure = _figure(output, film.name, frame)
    return dict(
        name=film.name,
        movie=f"{film.name}.mp4",
        figure=figure,
        frames=frames,
        **decoded,
    )


def write_films(
    tiles: Mapping[str, Sequence[Sequence[Mapping]]],
    output: Path,
    *,
    film: FilmSettings,
    tile_settings: TileSettings,
    unit: str,
    style: FigureStyle,
    lock=None,
) -> list[dict]:
    """The films one film setting makes of ``tiles``: a pyramid per run, or one
    grid over the runs.

    ``tiles`` maps each run to the rows of the film's panels, in their order.
    Returns each film's record (:func:`write_pyramid_film`, :func:`write_grid_film`).

    """
    options = dict(
        film=film, tile_settings=tile_settings, unit=unit, style=style, lock=lock
    )
    if film.layout == "pyramid":
        return [
            write_pyramid_film(run, panels, output, **options)
            for run, panels in tiles.items()
        ]
    return [
        write_grid_film(
            {run: panels[0] for run, panels in tiles.items()}, output, **options
        )
    ]
