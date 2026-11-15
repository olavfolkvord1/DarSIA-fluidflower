"""The Media's steps: each run's film tiles and the configured films.

- ``film-tiles`` draws a tile of every panel the configured films choose for each
  frame of a run (:func:`draw_tile`), each drawing once however many films show
  it: the corrected image (R1) as a photograph, a mass map (R7) with its display
  mean, the depth-scaled one on the run's sand map and geometry, or the mass graph
  of the run's whole domain (R8) with the injected mass, on axes shared by the
  selected runs; a grid film's graph tiles keep room for the run's label. A
  frame's target time is its own time since the injection start.
- ``films`` makes one configured film from the selected runs' tiles
  (:func:`write_films`): a pyramid per run or one grid over the runs, each encoder
  opened under the host's waiting lock (:data:`ENCODER_LOCK`), and stores the
  films' records beside them.

"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import darsia
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, StepContext
from darsia.presets.workflows.analysis.domain_mass import read_domain_mass
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS_TABLE, DomainSettings
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
    frame_key,
    run_grid,
)
from darsia.presets.workflows.media import tiles as media
from darsia.presets.workflows.media.films import FilmSettings, write_films
from darsia.presets.workflows.media.panels import PhotographSettings
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    protocol_files,
    read_image,
    read_rig,
    run_settings,
)
from darsia.utils.results import write_json

UNIT = "g/L"
"""The unit of the mass maps a film shows: CO2 mass per pore volume (R7)."""
MAPS = ("mass", "mass_g", "mass_aq")
"""The mass maps a panel may show, by their field."""
DRAWINGS = "drawings.json"
"""The file of a run's tiles that names the drawing of each tile folder."""
RECORDS = "films.json"
"""The file of a film setting's records, beside its films."""
ENCODER_LOCK = "film-encoder"
"""The host's waiting lock held while a film's encoder opens: encoders opened at
once can crash a process on Windows."""


@dataclass(frozen=True)
class MediaSettings:
    """The films of a series; the ``[media]`` table."""

    films: tuple[FilmSettings, ...]
    """Each film, by its setting; their names differ."""


def drawings(settings: MediaSettings) -> list[dict]:
    """Every distinct drawing of a tile the films choose, in the films' order: a
    panel, and whether a graph keeps room for a grid's label.

    Raises
    ------
    ValueError
        If two films are named alike, or a panel shows another map than
        :data:`MAPS`.

    """
    names = [film.name for film in settings.films]
    if len(set(names)) != len(names):
        raise ValueError(f"Two films are named alike: {names}.")
    found = []
    for film in settings.films:
        for panel in film.panels:
            if panel.content not in (media.IMAGE, media.GRAPH, *MAPS):
                raise ValueError(f"A film shows no {panel.content}; maps: {MAPS}.")
            room = panel.content == media.GRAPH and film.layout == "grid"
            drawing = dict(panel=asdict(panel), label_room=room)
            if drawing not in found:
                found.append(drawing)
    return found


def _graph_rows(ctx: StepContext, run: str) -> list[dict]:
    """A run's mass graph: its whole domain's rows (R8), one per frame, with the
    CO2 injected up to the frame [g].

    Raises
    ------
    ValueError
        If the selection has not exactly one domain of the whole rig.

    """
    rows = ctx.read(
        "domain-mass",
        lambda f: read_domain_mass(Path(f) / DOMAIN_MASS_TABLE),
        run=run,
    )
    whole = {row["domain_id"] for row in rows if row["kind"] == "whole"}
    if len(whole) != 1:
        raise ValueError(f"A mass graph needs one whole domain selected: {whole}.")
    protocol = make_experiment(run_settings(ctx, run, EXPERIMENT)).injection_protocol
    found = []
    for row in rows:
        if row["domain_id"] not in whole:
            continue
        hours = float(row["elapsed_h"])
        masses = {
            column: None if value == "" else float(value)
            for column, value in row.items()
            if column.endswith("_mass_g")
        }
        injected = protocol.injected_mass(time=hours) * 1000
        found.append(
            dict(case_id=row["case_id"], elapsed_h=hours, injected_mass_g=injected)
            | masses
        )
    return found


def _film_tiles(ctx: StepContext) -> None:
    """Each frame's tile of every drawing the films choose."""
    run, settings = ctx.task.run, ctx.settings
    style, tile_settings = settings["style"], settings["tiles"]
    found = drawings(settings["media"])
    panels = [media.Panel(**d["panel"]) for d in found]
    graphs, axes = {run: ()}, None
    if any(panel.content == media.GRAPH for panel in panels):
        graphs = {name: _graph_rows(ctx, name) for name in ctx.selection.frames}
        axes = media.graph_axes(tile_settings, list(graphs.values()), style)
    grid = run_grid(ctx.read("rig", read_rig), settings["material"])
    rig = grid.rig
    sand = media.SandGeometry(
        rig.labels.img,
        settings["domains"].sand_labels,
        grid.geometry.pore_volume_m3,
        rig.depth.img,
        rig.pixel_area.img,
    )
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    shape = rig.facies.img.shape

    def write(folder):
        parts = [Path(folder) / f"{k:02d}" for k in range(len(found))]
        rows = [[] for _ in found]
        for frame in ctx.task.frames:
            maps = ctx.read("mass-maps", darsia.read_mass_maps, frame=frame)
            phase = ctx.read("phase-maps", darsia.read_phase_maps, frame=frame)
            date = experiment.get_datetime(ctx.photo(run, frame))
            hours = experiment.time_since_start(date)
            shown = media.Frame(
                case=frame_key(frame),
                elapsed_h=hours,
                target_h=hours,
                shape=shape,
                box=(0, 0, shape[1], shape[0]),
                image=ctx.read("corrected-image", read_image, frame=frame).img,
                maps={name: getattr(maps, name).img for name in MAPS},
                unit=UNIT,
                phase=phase["labels"],
                saturation=phase["saturation"],
                sand=sand,
            )
            for k, (panel, drawing) in enumerate(zip(panels, found)):
                tile, drawn = media.draw_tile(
                    panel,
                    shown,
                    tile_settings,
                    style=style,
                    photograph_settings=settings["photograph"],
                    rows=graphs[run],
                    axes=axes,
                    label_room=drawing["label_room"],
                )
                rows[k].append(media.save_tile(parts[k], shown, tile, drawn))
        for part, made in zip(parts, rows):
            part.mkdir(parents=True, exist_ok=True)
            media.write_tile_index(part, made)
        write_json(Path(folder) / DRAWINGS, found)

    ctx.publish(write)


def read_film_tiles(folder: Path) -> list[tuple[dict, list[dict]]]:
    """A run's stored tiles: each drawing (:func:`drawings`) with its tiles' rows
    (:func:`darsia.presets.workflows.media.tiles.read_tile_index`)."""
    folder = Path(folder)
    found = json.loads((folder / DRAWINGS).read_text(encoding="utf-8"))
    return [
        (drawing, media.read_tile_index(folder / f"{k:02d}"))
        for k, drawing in enumerate(found)
    ]


def _film_names(selection: Selection, tables: Mapping) -> tuple[str, ...]:
    """One part per configured film, by its name."""
    return tuple(film["name"] for film in tables["media"]["films"])


def _films(ctx: StepContext) -> None:
    """One film setting's films of the selected runs' tiles."""
    settings = ctx.settings
    film = next(f for f in settings["media"].films if f.name == ctx.task.part)
    tiles = {}
    for run in ctx.selection.frames:
        stored = ctx.read("film-tiles", read_film_tiles, run=run)
        tiles[run] = []
        for panel in film.panels:
            room = panel.content == media.GRAPH and film.layout == "grid"
            wanted = dict(panel=asdict(panel), label_room=room)
            tiles[run].append(next(rows for d, rows in stored if d == wanted))

    def write(folder):
        records = write_films(
            tiles,
            Path(folder),
            film=film,
            tile_settings=settings["tiles"],
            unit=UNIT,
            style=settings["style"],
            lock=ctx.host_lock(ENCODER_LOCK),
        )
        write_json(Path(folder) / RECORDS, records)

    ctx.publish(write)


FILM_TILES = Step(
    "film-tiles",
    {
        "rig": Scope.OWN,
        "corrected-image": Scope.OWN,
        "phase-maps": Scope.OWN,
        "mass-maps": Scope.OWN,
        "domain-mass": Scope.ALL,
    },
    Order.RUN,
    Memory(grid=24.0),
    _film_tiles,
    {
        "media": MediaSettings,
        "tiles": media.TileSettings,
        "style": FigureStyle,
        "photograph": PhotographSettings,
        "domains": DomainSettings,
        "material": MaterialSettings,
    },
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each run's tiles of the films' panels."""
FILMS = Step(
    "films",
    {"film-tiles": Scope.ALL},
    Order.PARTS,
    Memory(),
    _films,
    {
        "media": MediaSettings,
        "tiles": media.TileSettings,
        "style": FigureStyle,
    },
    _film_names,
)
"""Each configured film of the selected runs."""
STEPS = (FILM_TILES, FILMS)
"""The Media's steps."""
