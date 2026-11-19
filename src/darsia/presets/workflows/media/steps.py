"""The Media's steps: each run's film tiles and the configured films.

- ``film-tiles`` draws a tile of every panel the configured films choose for each
  frame of a run (:func:`draw_tile`), each drawing once however many films show
  it: the corrected image (R1) as a photograph, a mass map (R7) with its display
  mean, the depth-scaled one on the run's sand map and geometry, or the mass graph
  of the run's whole domain (R8) with the injected mass, on axes shared by the
  selected runs; a grid film's graph tiles keep room for the run's label. A
  frame carries its frozen target and its actual time since injection start.
- ``films`` makes one configured film from the selected runs' tiles
  (:func:`write_films`): a pyramid per run or one grid over the runs, each encoder
  opened under the host's waiting lock (:data:`ENCODER_LOCK`), and stores the
  films' records beside them.

"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import darsia
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, StepContext, Term
from darsia.presets.workflows.analysis.domain_mass import read_domain_mass
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS_TABLE, DomainSettings
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
    frame_key,
    run_grid,
)
from darsia.presets.workflows.media import tiles as media
from darsia.presets.workflows.media.films import FilmSettings, write_films
from darsia.presets.workflows.media.panels import (
    PhotographSettings,
    source_photograph_settings,
    tile_size,
)
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.media.transfer import divided_size, first_frame_limits
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
            room = panel.content == media.GRAPH and film.layout in ("grid", "single")
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
    shared = ctx.tables.get("ensemble", {}).get("target_h")
    targets = film_frame_targets(run, ctx.task.frames, shared, settings["media"])
    has_photo = any(panel.content == media.IMAGE for panel in panels)
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
        grid.geometry.swi,
    )
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    shape = rig.facies.img.shape

    def write(folder):
        parts = [Path(folder) / f"{k:02d}" for k in range(len(found))]
        rows = [[] for _ in found]
        photograph_limits = None
        frames = ctx.task.frames
        if settings["photograph"].transfer == "first-frame":
            frames = sorted(
                frames, key=lambda frame: experiment.get_datetime(ctx.photo(run, frame))
            )
        for frame in frames:
            maps = ctx.read("mass-maps", darsia.read_mass_maps, frame=frame)
            phase = ctx.read("phase-maps", darsia.read_phase_maps, frame=frame)
            date = experiment.get_datetime(ctx.photo(run, frame))
            hours = experiment.time_since_start(date)
            case = frame_key(frame)
            shown = media.Frame(
                case=case,
                elapsed_h=hours,
                target_h=targets[case] if targets is not None else hours,
                shape=shape,
                box=(0, 0, shape[1], shape[0]),
                image=ctx.read("corrected-image", read_image, frame=frame).img,
                maps={name: getattr(maps, name).img for name in MAPS},
                unit=UNIT,
                phase=phase["labels"],
                saturation=phase["saturation"],
                sand=sand,
                rules=phase["rules"],
                reporting_domain=(
                    rig.reporting_domain
                    if settings["photograph"].restrict_to_reporting_domain
                    else None
                ),
            )
            photo = settings["photograph"]
            if has_photo:
                photo = source_photograph_settings(photo, ctx.photo(run, frame))
            if (
                settings["photograph"].transfer == "first-frame"
                and photograph_limits is None
            ):
                size = (
                    tile_size(shown.box, tile_settings.width_px)
                    if tile_settings.grid_divisor is None
                    else divided_size(shape, tile_settings.grid_divisor)
                )
                photograph_limits = first_frame_limits(
                    shown.photo_samples(photo), size, photo.percentiles
                )
            for k, (panel, drawing) in enumerate(zip(panels, found)):
                tile, drawn = media.draw_tile(
                    panel,
                    shown,
                    tile_settings,
                    style=style,
                    photograph_settings=photo,
                    rows=graphs[run],
                    axes=axes,
                    label_room=drawing["label_room"],
                    photograph_limits=photograph_limits,
                )
                rows[k].append(media.save_tile(parts[k], shown, tile, drawn))
        for part, made in zip(parts, rows):
            part.mkdir(parents=True, exist_ok=True)
            media.write_tile_index(part, made)
        write_json(Path(folder) / DRAWINGS, found)

    ctx.publish(write)


def film_frame_targets(run, frames, shared, settings: MediaSettings):
    """Use the frozen ensemble assignment by case; never rematch by time."""
    if shared is None:
        if any(film.layout == "grid" for film in settings.films):
            raise ValueError(
                "Grid films need the frozen [ensemble.target_h] assignments."
            )
        return None
    assigned = shared.get(run, {})
    found = {}
    used = set()
    for frame in frames:
        case = frame_key(frame)
        value = assigned.get(case)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"No finite frozen film target for {run}/{case}.")
        if value in used:
            raise ValueError(
                f"Multiple selected images of {run} have target {value} h."
            )
        found[case] = float(value)
        used.add(value)
    return found


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


def _display_bytes(tables, _run, context):
    """Additional area conversion and raster-overlay arrays, in bytes.

    Area mode reserves eight float64 grid arrays for conversion, validity and
    weighted block pooling. Overlay masks/boundary reduction need twelve bytes
    per grid cell plus twelve per tile pixel; no contour objects are retained.
    The two branches run sequentially, so their maximum is reserved.
    """
    tile = tables["tiles"]
    photo = tables["photograph"]
    area = 64 * context.grid_pixels if tile.get("mass_quantity") == "area" else 0
    overlay = 0
    if photo.get("co2_outline", False) or photo.get("gas_contour", False):
        shape = context.grid_shape
        if shape is None:
            # Before measured shape metadata is available, the thinnest grid
            # gives the conservative height bound.
            shape = (context.grid_pixels, 1)
        size = media.tile_size((0, 0, shape[1], shape[0]), int(tile["width_px"]))
        if tile.get("grid_divisor") is not None and context.grid_shape is not None:
            size = divided_size(shape, tile["grid_divisor"])
        per_display_pixel = 256 if photo.get("transfer") == "first-frame" else 12
        overlay = 12 * context.grid_pixels + per_display_pixel * size[0] * size[1]
        if photo.get("co2_source") == "mass":
            overlay += 40 * context.grid_pixels
    transfer = 0
    if photo.get("transfer") == "first-frame":
        # Finite RGB handling, conversion and percentile selection coexist only
        # during first-frame measurement; all reduced RGB buffers fit this bound.
        transfer = 88 * context.grid_pixels
    masked_photo = (
        24 * context.grid_pixels if photo.get("restrict_to_reporting_domain") else 0
    )
    return max(area, overlay, transfer) + masked_photo


def _films(ctx: StepContext) -> None:
    """One film setting's films of the selected runs' tiles."""
    settings = ctx.settings
    film = next(f for f in settings["media"].films if f.name == ctx.task.part)
    tiles = {}
    for run in ctx.selection.frames:
        stored = ctx.read("film-tiles", read_film_tiles, run=run)
        tiles[run] = []
        for panel in film.panels:
            room = panel.content == media.GRAPH and film.layout in ("grid", "single")
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
    Memory(
        grid=23.0,
        camera_bytes=20.0,
        grid_bytes=82.0,
        terms=(Term(_display_bytes, fixed_bytes=1),),
    ),
    _film_tiles,
    {
        "media": MediaSettings,
        "tiles": media.TileSettings,
        "style": FigureStyle,
        "photograph": PhotographSettings,
        "domains": DomainSettings,
        "material": MaterialSettings,
    },
    sections=(*EXPERIMENT, "ensemble"),
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
