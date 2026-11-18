"""Simulator export: a series' sand geometry, protocols and mass maps as plain files.

A simulator reads the rig on a grid of metric cells: which sand lies in each cell,
its porosity and depth, and the volumes that hold the CO2. This module writes that
package for the runs of a series (:func:`write_simulator_package`).

**Geometry.** The series' sand-map drawing (:class:`SandMapDrawing`) is filled on
the export canvas, a grid of the rig's measured inner frame in rig metres (the frame
of the mass maps: from the inner corner of the left wall and the floor, x to the
right and y up), by the one rasteriser of the setup (:func:`sand_layers`); a pixel
inside the frame that no layer covers, e.g. where the camera does not see the rig,
takes the nearest layer. Every position comes from the canvas's coordinates. The
simulator's porous domain is completed from it (:func:`complete_geometry`):

1. in each column the domain runs from the sand surface down to the frame's floor:
   below the free water at the top of the frame, or, in a column whose top is not
   free water, from its highest sand or fixture; free water below the surface and
   excluded layers, e.g. silicone along the walls, stay outside the domain;
2. a fixture in front of the sand, e.g. a colour checker, hides it: the sand drawn
   beneath the fixture takes its place, a pixel beneath it that no layer covers
   taking the nearest layer drawn;
3. a pixel of the domain that is neither sand nor sand drawn beneath a fixture is
   refused.

The completed domain is scaled onto metric cells by the one scaling method
(:func:`darsia.metric_cells`; areas and volumes summed in double precision by
:func:`darsia.cell_sums`), per cell size (:func:`geometry_cells`).

**Each run's own geometry.** A run whose sand has settled has an adjustment, the
anchors it moves (:meth:`SandMapDrawing.moved`). Its geometry is computed from its
own drawing and written as its own; the other runs share the common geometry, and
the README names the runs that differ.

**Protocols.** Each run's injection, pressure and temperature, and overflow tables
are copied as their files hold them, the overflow table in the chosen columns.
Where the run's scale readings are usable, the displaced water at each reading
(:func:`darsia.displaced_volume`) is written beside them, on the time axis of the
readings' own injection start. A run without usable readings of its own borrows the
curve of a donor by the rule of the mass maps' estimate (:func:`darsia.ranked_donors`):
of the runs it may borrow from, e.g. its calibration's fit runs, those with usable
readings, its sand map and a recipe within the overflow's tolerance, the one nearest
in atmospheric pressure; without one, the run nearest in pressure alone. The package
says that the curve is borrowed, from which run, and when no run matched its sand
map and recipe.

**Properties.** The facies table and the depth measurements as given, and the
reference conditions of the mass flow controller with the gas density at them.

**Mass maps** (optional). A run's mass maps on the same cells
(:class:`darsia.MassCopy`).

Every table is written by the analyses' one CSV writer (:func:`write_rows`), so that
a number keeps all its digits. The arrays are written with their rows from the
floor up (y increasing) and their columns from the left (x increasing), the origin
at the rig's lower left inner corner, as a simulator's grid runs.

"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import openpyxl
import pandas as pd

import darsia
from darsia.presets.workflows.analysis.domains import write_rows
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    rasterize_sand_map,
    rig_grid,
)

FORMAT = "darsia-simulator-package"
"""Format of the package's manifest."""
ORIGIN = "injection_start"
"""Column of an overflow table holding the injection start of its scale readings,
from which their times are counted."""
UNMATCHED = "no run with its sand map and recipe; the run nearest in pressure alone"
"""Why a run borrows its overflow curve from a run of another sand map or recipe."""
NO_PRESSURE = (
    "the runs with its sand map and recipe lack atmospheric pressure over the "
    "comparison window"
)
"""Why a run borrows no overflow curve although runs of its sand map and recipe have
one, as the mass maps' estimate then has none."""
NO_DONOR = (
    "no run it may borrow from has usable readings and atmospheric pressure over the "
    "comparison window"
)
"""Why a run borrows no overflow curve at all."""
UNITS = {
    "facies_majority": "facies id",
    "facies_majority_area_fraction": "1 of the domain area",
    "facies_ids": "facies id",
    "facies_area_fractions": "1 of the domain area (one layer per facies id)",
    "porosity": "1 (pore over bulk volume)",
    "depth_m": "m",
    "pore_volume_m3": "m3",
    "bulk_volume_m3": "m3",
    "domain_area_m2": "m2",
    "domain_area_fraction": "1 of the cell",
    "drawn_behind_area_fraction": "1 of the cell",
    "drawn_behind": "flag",
    "drawn_behind_facies_ids": "facies id",
    "drawn_behind_facies_area_fractions": "1 of the cell (one layer per facies id)",
}
"""The arrays of a geometry on metric cells and their units."""


@dataclass(frozen=True)
class FlowReference:
    """Reference conditions of the mass flow controller of the injection."""

    gas: str
    """The injected gas."""
    flow_unit: str
    """Unit of the controller's flow rates, e.g. mL/min at the reference."""
    reference_temperature_c: float
    """Reference temperature [Celsius] of the flow rates."""
    reference_pressure_hpa: float
    """Reference pressure [hPa] of the flow rates."""


@dataclass(frozen=True)
class SimulatorExportSettings:
    """Settings of the simulator export; the ``[export.simulator]`` table.

    The flow reference, given as a mapping as read from a TOML file, becomes its
    record.

    """

    porous_facies: Sequence[int]
    """Facies of the sand, porous media and seals, which make the porous domain."""
    free_water_facies: Sequence[int]
    """Facies of free water, which lies above the sand surface."""
    fixture_facies: Sequence[int]
    """Facies of fixtures in front of the sand, e.g. a colour checker, whose sand is
    completed behind them."""
    excluded_facies: Sequence[int]
    """Facies outside the porous domain wherever they lie, e.g. silicone along the
    walls."""
    cell_sizes_mm: Sequence[float | None]
    """Sizes [mm], or None for the canvas's native rectangular cells."""
    precision: Literal["float32", "float64"]
    """Precision of the package's arrays; areas and volumes are summed in double
    precision."""
    overflow_columns: Sequence[str]
    """Columns of a run's overflow table written to the package, in this order."""
    flow_reference: FlowReference
    """Reference conditions of the mass flow controller."""
    notes: Sequence[str] = ()
    """Paragraphs the README adds about the series' package, e.g. how an earlier
    package of the series differs."""

    def __post_init__(self) -> None:
        """Turn a flow reference given as a mapping into its record."""
        if isinstance(self.flow_reference, Mapping):
            object.__setattr__(
                self, "flow_reference", FlowReference(**self.flow_reference)
            )


@dataclass(frozen=True)
class MassFrames:
    """A run's mass maps, frame by frame, with the run's geometry."""

    geometry: darsia.RunGeometry
    """The run's geometry (:func:`darsia.run_geometry`)."""
    frames: Iterable[tuple[str, darsia.SimpleMassAnalysisResults, np.ndarray]]
    """Each frame's name, its mass maps and its phase map."""
    sources: Mapping[str, "MassFrameSource"] | None = None
    """Original source identities and times by exported frame name, never inferred."""


@dataclass(frozen=True)
class MassFrameSource:
    """A mass frame's source and physical clock, independent of its output name."""

    session: str
    image: str
    capture_datetime: datetime
    injection_start: datetime

    def row(self) -> dict:
        """Source and elapsed time in seconds from injection, without rounding."""
        return dict(
            source_session=self.session,
            source_image=self.image,
            capture_datetime=self.capture_datetime.isoformat(),
            injection_start=self.injection_start.isoformat(),
            elapsed_s=(self.capture_datetime - self.injection_start).total_seconds(),
        )


@dataclass(frozen=True)
class SimulatorRun:
    """A run of the package: its protocols, its adjustment and its mass maps."""

    injection: Path | tuple[Path, str]
    """Its injection protocol: a CSV file, or a workbook and its sheet."""
    pressure_temperature: Path | tuple[Path, str]
    """Its pressure and temperature protocol, likewise."""
    recipe: Mapping[str, float]
    """Its chemistry, e.g. its alkalinity and indicator concentrations, which a run
    it borrows an overflow curve from shares within the overflow's tolerance; it
    holds every entry the tolerance compares."""
    overflow: Path | tuple[Path, str] | None = None
    """Its overflow table with the scale readings, likewise; None without one."""
    donors: Sequence[str] | None = None
    """The runs it may borrow an overflow curve from, e.g. the fit runs of its
    calibration, as the mass maps' estimate does; None for every run of the
    package. Those the package lacks are named in its manifest."""
    adjustment: Mapping[int, Sequence[float]] | None = None
    """The anchors its adjustment moves, by index, for a run whose sand has settled
    (:func:`read_moved_anchors`); None for the common drawing."""
    mass: MassFrames | None = None
    """Its mass maps, if the package holds them."""


@dataclass(frozen=True)
class SimulatorSeries:
    """What the runs of a series share in the package."""

    drawing: SandMapDrawing
    """The common sand-map drawing."""
    facies_of_layer: Mapping[int, int]
    """Facies of each layer of the drawing, e.g. the ``[facies]`` groups."""
    canvas: darsia.Image
    """Depth [m] of the rig on the export canvas, a grid of the rig's inner frame in
    rig metres, e.g. the rig's depth map."""
    facies_properties: Path
    """The facies table (CSV): ``id`` and ``porosity`` of each facies, with any
    further columns, copied as it is."""
    reference_density: float
    """Density [kg/m3] of the gas at the flow controller's reference."""
    depth_measurements: Path | None = None
    """The measurements the depth was made from, copied as they are; None for none."""


@dataclass(frozen=True)
class CompletedGeometry:
    """The simulator's porous domain on the export canvas."""

    facies: np.ndarray
    """Facies of each pixel of the domain and of the sand drawn behind fixtures; the
    sand map's facies elsewhere (-1 for none)."""
    domain: np.ndarray
    """Pixels of the porous domain."""
    drawn_behind: np.ndarray
    """Pixels where a layer is drawn beneath a fixture."""
    counts: dict
    """How the domain was found: the columns whose surface lies below free water
    and at their highest sand, and the pixels of the domain, of its sand as mapped
    and drawn behind fixtures."""


def sand_layers(
    drawing: SandMapDrawing,
    canvas: darsia.Image,
    fixture_layers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The sand map of a drawing on the export canvas, the sand drawn beneath its
    fixtures, and the pixels no polygon covers.

    The drawing is filled as the setup fills it on a run's grid
    (:func:`rasterize_sand_map`), on the canvas's own metric grid (:func:`rig_grid`).
    Every pixel inside the frame that no polygon covers takes the nearest covered
    layer, at any distance from the frame's edge: the setup's edge fill without a
    width. Such pixels lie e.g. along a wall the camera does not see, where the
    nearest layer is the wall's own, or between two layers. The drawing without the
    fixtures' polygons, filled the same way, gives the layer drawn beneath each
    fixture pixel: where nothing is drawn beneath a fixture, e.g. along the frame's
    top above the free water drawn beneath it, the nearest layer drawn.

    Parameters
    ----------
    drawing : SandMapDrawing
        The drawing, possibly with a run's adjustment.
    canvas : darsia.Image
        An image on the export canvas, spanning the rig's inner frame in metres from
        its lower left inner corner.
    fixture_layers : sequence of int
        The layers of the fixtures.

    Returns
    -------
    layers : array
        Layer of each pixel (uint8; 0 outside the frame, and inside it only for a
        drawing without layers).
    inside : array
        Pixels inside the frame (bool).
    behind : array
        Layer drawn beneath each fixture pixel (uint8; 0 elsewhere, and beneath it
        only for a drawing of fixtures alone).
    filled : array
        Pixels inside the frame that no polygon covers and that took the nearest
        layer (bool).

    Raises
    ------
    ValueError
        If the canvas's lower left corner is not the frame's, at the origin of the
        rig's coordinates.

    """
    corner = (canvas.origin[0], canvas.origin[1] - canvas.dimensions[0])
    if not np.allclose(corner, (0.0, 0.0)):
        raise ValueError(
            f"The canvas's lower left corner lies at {corner} m, not at the rig's "
            "inner corner (0, 0), so it does not span the frame."
        )
    shape = canvas.img.shape[:2]
    height, width = canvas.dimensions
    rig_to_grid = rig_grid(canvas)

    def raster(sand_map, edge_fill_mm):
        """The sand map on the canvas, with the given edge fill."""
        return rasterize_sand_map(
            sand_map,
            shape,
            rig_to_grid,
            width=width,
            height=height,
            edge_fill_mm=edge_fill_mm,
        )

    painted, inside = raster(drawing, 0.0)
    layers, _ = raster(drawing, np.inf)
    fixtures = set(fixture_layers)
    beneath = SandMapDrawing(
        drawing.anchors,
        tuple(polygon for polygon in drawing.polygons if polygon[0] not in fixtures),
        drawing.on_top - fixtures,
    )
    under, _ = raster(beneath, np.inf)
    behind = np.where(inside & np.isin(layers, list(fixtures)), under, 0)
    filled = inside & (painted == 0) & (layers != 0)
    return layers, inside, behind.astype(np.uint8), filled


def complete_geometry(
    layers: np.ndarray,
    inside: np.ndarray,
    behind: np.ndarray,
    facies_of_layer: Mapping[int, int],
    settings: SimulatorExportSettings,
) -> CompletedGeometry:
    """The simulator's porous domain, completed behind fixtures.

    In each column inside the frame the domain runs from the sand surface down to
    the frame's lowest pixel. The surface lies below the free water that starts at
    the top of the frame; in a column whose top is not free water it is the highest
    sand or fixture pixel. Free water below it and the excluded facies, wherever
    they lie, stay outside the domain, so a column of free water and excluded facies
    alone has none. Sand drawn beneath a fixture is part of the domain, whatever the
    surface, and free water or an excluded layer drawn there stays outside it.

    Parameters
    ----------
    layers, inside, behind : array
        The sand map, the frame and the sand drawn beneath the fixtures
        (:func:`sand_layers`).
    facies_of_layer : mapping
        Facies of each layer; a layer without one is neither sand nor free water.
    settings : SimulatorExportSettings
        The facies of sand, free water, fixtures and those excluded.

    Returns
    -------
    CompletedGeometry
        The domain and its facies.

    Raises
    ------
    ValueError
        If a layer drawn beneath a fixture has no facies of sand, free water or the
        excluded, a column inside the frame with pixels of neither free water nor
        excluded facies has neither free water at its top nor sand or a fixture, or a
        pixel of the domain is neither sand nor sand drawn beneath a fixture.

    """
    table = np.full(256, -1, np.int64)
    for layer, facies_id in facies_of_layer.items():
        table[int(layer)] = int(facies_id)
    facies, drawn_facies = table[layers], table[behind]
    drawn = behind != 0
    allowed = [
        *settings.porous_facies,
        *settings.free_water_facies,
        *settings.excluded_facies,
    ]
    unknown_drawn = drawn & ~np.isin(drawn_facies, allowed)
    if unknown_drawn.any():
        raise ValueError(
            f"{int(np.count_nonzero(unknown_drawn))} pixels beneath a fixture are drawn "
            f"with layers {sorted(set(behind[unknown_drawn].tolist()))}, which have no "
            "facies of sand, free water or the excluded."
        )
    water = inside & np.isin(facies, settings.free_water_facies)
    fixture = inside & np.isin(facies, settings.fixture_facies)
    porous = inside & np.isin(facies, settings.porous_facies) & ~fixture
    outside = water | (inside & np.isin(facies, settings.excluded_facies))

    rows = np.arange(layers.shape[0])[:, None]
    framed = inside.any(axis=0)
    start = np.argmax(inside, axis=0)
    floor = layers.shape[0] - 1 - np.argmax(inside[::-1], axis=0)
    open_water = ~(water & inside) & (rows >= start)
    below_water = np.where(open_water.any(axis=0), np.argmax(open_water, axis=0), -1)
    below_water = np.where(below_water < 0, layers.shape[0], below_water)
    solid = (porous | fixture) & inside
    water_top = framed & (below_water > start)
    sand_top = framed & ~water_top & solid.any(axis=0)
    unknown = (inside & ~outside).any(axis=0) & ~water_top & ~sand_top
    if unknown.any():
        raise ValueError(
            f"{int(np.count_nonzero(unknown))} columns inside the frame have neither "
            "free water at their top nor sand or a fixture, so their sand surface is "
            "unknown."
        )
    top = np.where(water_top, below_water, np.argmax(solid, axis=0))
    top = np.where(water_top | sand_top, top, layers.shape[0])
    domain = inside & ~outside & (rows >= top) & (rows <= floor)

    drawn_porous = drawn & np.isin(drawn_facies, settings.porous_facies)
    domain = (domain | drawn_porous) & ~(drawn & ~drawn_porous)
    original = porous & domain
    unfilled = domain & ~original & ~drawn_porous
    if unfilled.any():
        raise ValueError(
            f"{int(np.count_nonzero(unfilled))} pixels of the porous domain are "
            "neither sand nor sand drawn beneath a fixture."
        )
    counts = dict(
        surface_columns_below_free_water=int(np.count_nonzero(water_top)),
        surface_columns_at_highest_sand=int(np.count_nonzero(sand_top)),
        domain_pixels=int(np.count_nonzero(domain)),
        original_pixels=int(np.count_nonzero(original)),
        drawn_behind_pixels=int(np.count_nonzero(drawn)),
    )
    return CompletedGeometry(
        np.where(drawn, drawn_facies, facies), domain, drawn, counts
    )


def _export_cells(shape, dimensions, size):
    """One export grid for both geometry and masses, including native pitches."""
    if size is not None:
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Export cell size must be positive and finite.")
        return darsia.metric_cells(shape, list(dimensions), size)
    return darsia.MetricCells(
        shape=tuple(shape),
        dimensions=list(dimensions),
        index=np.arange(np.prod(shape), dtype=np.int64),
        pixel_area=float(np.prod(np.asarray(dimensions) / shape)),
    )


def _size_name(size):
    """The public resolution name, without assuming square native pixels."""
    return "full" if size is None else f"{size:g}mm"


def geometry_cells(
    completed: CompletedGeometry,
    depth_m: np.ndarray,
    porosity: Mapping[int, float],
    dimensions: Sequence[float],
    cell_size_mm: float | None,
    precision: str,
) -> dict[str, np.ndarray]:
    """The completed geometry on metric cells (:func:`darsia.metric_cells`).

    Each pixel's area, bulk volume (area times depth) and pore volume (times the
    porosity of its facies, taken in the depth's floating dtype) are formed and
    summed over each cell in double precision, as our production formed them with
    its double-precision pixel areas and summed them. Per cell (:data:`UNITS`): the
    domain's area and its fraction of the cell; the area fraction of each facies of
    the domain and the majority facies (the lowest on a tie; 0 without domain); bulk
    and pore volume, porosity (pore over bulk volume) and depth (bulk volume over
    area); and the area drawn behind fixtures,
    also outside the domain, per drawn facies. Ratios are NaN in a cell without
    domain. Rows run from the floor up.

    Parameters
    ----------
    completed : CompletedGeometry
        The domain on the export canvas (:func:`complete_geometry`).
    depth_m : array
        Depth [m] of each pixel of the canvas.
    porosity : mapping
        Porosity of each facies of the domain.
    dimensions : sequence of float
        Height and width [m] of the canvas.
    cell_size_mm : float
        Size [mm] of the cells.
    precision : str
        Precision of the arrays (``"float32"`` or ``"float64"``).

    Returns
    -------
    dict of str to array
        The arrays of :data:`UNITS`.

    """
    domain = completed.domain
    cells = _export_cells(domain.shape, dimensions, cell_size_mm)
    area = np.float64(cells.pixel_area)
    facies = np.where(domain, completed.facies, 0)
    present = [int(value) for value in np.unique(facies[domain])]
    depth_m = np.asarray(depth_m)
    phi = np.zeros(domain.shape, np.result_type(depth_m, np.float32))
    for facies_id in present:
        phi[domain & (facies == facies_id)] = porosity[facies_id]
    depth = np.where(domain, depth_m, 0).astype(np.float64)
    phi = phi.astype(np.float64)

    def sums(values):
        """Sums over each cell, the rows from the floor up."""
        return darsia.cell_sums(cells, values)[::-1]

    def stored(values):
        """The values in the package's precision."""
        return np.asarray(values).astype(precision)

    domain_area = sums(np.where(domain, area, 0.0))
    bulk = sums(area * depth)
    pore = sums(area * depth * phi)
    by_facies = [sums(np.where(domain & (facies == f), area, 0.0)) for f in present]
    majority = np.zeros(domain_area.shape, np.int32)
    largest = np.zeros(domain_area.shape)
    for facies_id, weights in zip(present, by_facies):
        larger = weights > largest
        majority[larger] = facies_id
        largest[larger] = weights[larger]
    cell_area = (
        cells.pixel_area
        if cell_size_mm is None
        else (cell_size_mm / 1000.0) * (cell_size_mm / 1000.0)
    )
    drawn = completed.drawn_behind
    drawn_ids = [int(value) for value in np.unique(completed.facies[drawn])]
    arrays = {
        "facies_majority": majority,
        "facies_majority_area_fraction": stored(
            darsia.cell_ratio(largest, domain_area)
        ),
        "facies_ids": np.asarray(present, np.int32),
        "facies_area_fractions": stored(
            np.stack([darsia.cell_ratio(w, domain_area) for w in by_facies])
            if present
            else np.zeros((0, *domain_area.shape))
        ),
        "porosity": stored(darsia.cell_ratio(pore, bulk)),
        "depth_m": stored(darsia.cell_ratio(bulk, domain_area)),
        "pore_volume_m3": stored(pore),
        "bulk_volume_m3": stored(bulk),
        "domain_area_m2": stored(domain_area),
        "domain_area_fraction": stored(domain_area / cell_area),
        "drawn_behind_area_fraction": stored(
            sums(np.where(drawn, area, 0.0)) / cell_area
        ),
        "drawn_behind_facies_ids": np.asarray(drawn_ids, np.int32),
        "drawn_behind_facies_area_fractions": stored(
            np.stack(
                [
                    sums(np.where(drawn & (completed.facies == f), area, 0.0))
                    / cell_area
                    for f in drawn_ids
                ]
            )
            if drawn_ids
            else np.zeros((0, *domain_area.shape))
        ),
    }
    arrays["drawn_behind"] = np.nan_to_num(arrays["drawn_behind_area_fraction"]) > 0
    return {name: arrays[name] for name in UNITS}


def _table(path: Path | tuple[Path, str]) -> list[list]:
    """A protocol's table as its file holds it, header first: a CSV file, or a
    workbook's sheet (its first without a name); an empty cell is ''."""
    path, sheet = (
        (path[0], path[1]) if isinstance(path, (list, tuple)) else (path, None)
    )
    if Path(path).suffix.lower() == ".csv":
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.reader(stream))
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = (book[sheet] if sheet is not None else book.worksheets[0]).iter_rows(
            values_only=True
        )
        return [["" if value is None else value for value in row] for row in rows]
    finally:
        book.close()


def _write_table(path: Path, rows: list[list], columns: Sequence | None = None) -> int:
    """Write a table's rows, in the given columns of its header (all without), by the
    one CSV writer; the number of rows below the header."""
    header = [str(name) for name in rows[0]]
    if len(set(header)) != len(header):
        raise ValueError(f"{path.name}: the table repeats a column name: {header}")
    columns = header if columns is None else list(columns)
    missing = [name for name in columns if name not in header]
    if missing:
        raise ValueError(f"{path.name}: the table has no column {missing}")
    uneven = [i for i, row in enumerate(rows[1:], start=2) if len(row) != len(header)]
    if uneven:
        raise ValueError(
            f"{path.name}: rows {uneven} have another number of cells than the "
            f"header's {len(header)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [dict(zip(header, row)) for row in rows[1:]]
    write_rows(
        path, ({name: row[name] for name in columns} for row in records), columns
    )
    return len(records)


def _reviewed_overflow(
    path: Path | tuple[Path, str], rows: list[list], overflow: darsia.OverflowSettings
) -> tuple[datetime, darsia.DisplacedVolume, list[dict]]:
    """The displaced water at each scale reading of a run, counted from the readings'
    own injection start (:data:`ORIGIN`): that start, the displaced volume, and the
    rows of its curve, none where the readings are not usable."""
    column = [str(name) for name in rows[0]].index(ORIGIN)
    origins = {row[column] for row in rows[1:]} - {""}
    if len(origins) != 1:
        raise ValueError(f"The overflow table needs one {ORIGIN}; it has {origins}")
    origin = datetime.fromisoformat(str(origins.pop()))
    times, weights = darsia.OverflowProtocol(path).readings(
        origin, list(overflow.statuses)
    )
    volume = darsia.displaced_volume(times, weights, overflow)
    if volume.reasons:
        return origin, volume, []
    curve = [
        dict(time_from_injection_h=t, weight_g=w, estimated_displaced_water_ml=v)
        for t, w, v in zip(volume.times_h, weights, volume.volume_ml)
    ]
    return origin, volume, curve


def _donor_run(
    run: SimulatorRun,
    sand_map: str,
    injection: darsia.InjectionProtocol,
    volume: darsia.DisplacedVolume | None,
) -> darsia.OverflowRun:
    """A run as the donor ranking compares it (:func:`darsia.ranked_donors`): its
    sand map and recipe, its atmospheric pressure in hours from its injection start,
    and the first start and last end of its injection at a positive rate."""
    start = injection.df["start"].min()
    pressure = darsia.PressureTemperatureProtocol(run.pressure_temperature).df
    hours = ((pressure["datetime"] - start).dt.total_seconds() / 3600).to_numpy()
    active = injection.df[injection.df["rate_kg_s"] > 0]
    window = (
        (active["start"].min() - start).total_seconds() / 3600,
        (active["end"].max() - start).total_seconds() / 3600,
    )
    return darsia.OverflowRun(
        sand_map,
        dict(run.recipe),
        hours,
        pressure["pressure_bar"].to_numpy(),
        window,
        volume,
    )


def _check_pressure(name: str, run: darsia.OverflowRun) -> None:
    """Refuse a run's atmospheric pressure record that the donor ranking cannot
    compare: fewer than two times, times not increasing, or a pressure that is not
    finite and positive."""
    times, pressure = run.pressure_h, run.pressure_bar
    if (
        times.size < 2
        or not np.isfinite(times).all()
        or not np.isfinite(pressure).all()
        or (pressure <= 0).any()
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError(
            f"The atmospheric pressure record of {name} needs two or more increasing "
            "times and finite, positive pressures to rank overflow donors."
        )


def _ranked_donors(
    name: str, pool: Mapping[str, darsia.OverflowRun], overflow: darsia.OverflowSettings
) -> tuple[list[str], str | None]:
    """A run's donors by the mass maps' rule among a pool (:func:`darsia.ranked_donors`)
    and the reason for a fallback or for none.

    The runs of its sand map and recipe come first, nearest in pressure. Where some
    match it but none has atmospheric pressure over the comparison window, it has no
    donor (:data:`NO_PRESSURE`), as the mass maps' estimate has none. Where none
    matches it, the runs nearest in pressure alone are its donors
    (:data:`UNMATCHED`), and without those it has none (:data:`NO_DONOR`).
    """
    donors = darsia.ranked_donors(name, pool, overflow)
    if donors:
        return donors, None
    own = pool[name]
    # On the run's own pressure record, every run that matches it ranks.
    alike = {
        key: replace(run, pressure_h=own.pressure_h, pressure_bar=own.pressure_bar)
        for key, run in pool.items()
    }
    if darsia.ranked_donors(name, alike, overflow):
        return [], NO_PRESSURE
    alone = {key: replace(run, sand_map="", recipe={}) for key, run in pool.items()}
    donors = darsia.ranked_donors(name, alone, overflow)
    return donors, UNMATCHED if donors else NO_DONOR


def _write_geometry(
    package: Path,
    folder: Path,
    drawing: SandMapDrawing,
    series: SimulatorSeries,
    porosity: Mapping[int, float],
    settings: SimulatorExportSettings,
) -> tuple[dict[str, str], dict[str, int]]:
    """Write one geometry, from its drawing, at each cell size into a folder of the
    package; its files by cell size, relative to the package, and the number of
    pixels no polygon covers by the layer they took."""
    fixture_layers = [
        layer
        for layer, facies_id in series.facies_of_layer.items()
        if facies_id in settings.fixture_facies
    ]
    canvas = series.canvas
    layers, inside, behind, filled = sand_layers(drawing, canvas, fixture_layers)
    filled_by_layer = {
        str(layer): int(count)
        for layer, count in zip(*np.unique(layers[filled], return_counts=True))
    }
    completed = complete_geometry(
        layers, inside, behind, series.facies_of_layer, settings
    )
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sand_map.json").write_text(
        json.dumps(
            dict(
                anchors=drawing.anchors.tolist(),
                polygons=[
                    [layer, list(indices)] for layer, indices in drawing.polygons
                ],
                on_top=sorted(drawing.on_top),
                facies_of_layer={str(k): v for k, v in series.facies_of_layer.items()},
            )
        ),
        encoding="utf-8",
    )
    files = {}
    for size in settings.cell_sizes_mm:
        arrays = geometry_cells(
            completed,
            canvas.img,
            porosity,
            canvas.dimensions,
            size,
            settings.precision,
        )
        name = _size_name(size)
        stem = folder / f"simulator_geometry_{name}"
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        shape = list(arrays["domain_area_m2"].shape)
        document = dict(
            cell_size_mm=None if size is None else float(size),
            shape=shape,
            extent_m=(
                [0.0, float(canvas.dimensions[1]), 0.0, float(canvas.dimensions[0])]
                if size is None
                else [0.0, shape[1] * size / 1000, 0.0, shape[0] * size / 1000]
            ),
            axes="rows: y from the floor up; columns: x from the left",
            canvas=dict(shape=list(layers.shape), dimensions_m=list(canvas.dimensions)),
            filled_pixels_by_layer=filled_by_layer,
            counts=completed.counts,
            arrays={
                name: dict(dtype=str(a.dtype), shape=list(a.shape), unit=UNITS[name])
                for name, a in arrays.items()
            },
        )
        if size is None:
            document["cell_pitch_m"] = dict(
                dy=float(canvas.dimensions[0] / shape[0]),
                dx=float(canvas.dimensions[1] / shape[1]),
            )
            document["resolution"] = "native rectangular cells"
        stem.with_suffix(".json").write_text(
            json.dumps(document, indent=1, allow_nan=False) + "\n", encoding="utf-8"
        )
        files[name] = stem.with_suffix(".npz").relative_to(package).as_posix()
    return files, filled_by_layer


def write_simulator_mass(
    folder: Path,
    mass: MassFrames,
    canvas: darsia.Image,
    sizes: Sequence[float | None],
) -> int:
    """Stream a run's observed maps onto the simulator geometry's exact grids.

    The shared native MassCopy reducers preserve extensive masses and known
    fractions; completed sand geometry never supplies missing observed mass.
    The uniform source grid must span the canvas and share its lower-left
    origin. Native export requires its own grid; metric export uses the common
    metric-cell reducer. Unknown mass contributes no known
    sum, and its pore-volume fraction remains explicit. Every resolution's sum
    is checked against the source finite mass before publication by the caller.

    The destination must be empty. An index gives supplied source identities,
    capture times and injection-relative seconds, as well as conservation totals.
    The output names are distinct safe stems; sessions never overwrite each other.
    """
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("The simulator mass destination must be empty.")
    sizes = tuple(sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Choose distinct simulator export resolutions.")
    lower_left = (
        float(canvas.origin[0]),
        float(canvas.origin[1] - canvas.dimensions[0]),
    )
    rows, cols = mass.geometry.domain.shape
    if not np.allclose(mass.geometry.dimensions, canvas.dimensions):
        raise ValueError("Uniform source geometry must span the export canvas.")
    copies, grids = {}, {}
    for size in sizes:
        target_cells = _export_cells(canvas.img.shape[:2], canvas.dimensions, size)
        height, width = target_cells.dimensions
        extent = (
            lower_left[0],
            lower_left[0] + width,
            lower_left[1],
            lower_left[1] + height,
        )
        if size is None and (rows, cols) != tuple(canvas.img.shape[:2]):
            raise ValueError("Uniform native export requires the canvas's own grid.")
        cells = _export_cells((rows, cols), mass.geometry.dimensions, size)
        if cells.shape != target_cells.shape:
            raise ValueError("Uniform source cells must match the export grid.")
        copy = darsia.MassCopy.from_cells(mass.geometry, cells, extent)
        name = _size_name(size)
        copies[name] = copy
        grids[name] = dict(
            shape=list(cells.shape),
            extent_m=extent,
            cell_pitch_m=dict(dy=height / cells.shape[0], dx=width / cells.shape[1]),
        )
    for name, copy in copies.items():
        target = folder / name
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "static.npz", **{k: v[::-1] for k, v in copy.static.items()}
        )
    index, seen = [], set()
    fields = {
        "mass_total_g": "mass",
        "mass_aqueous_g": "mass_aq",
        "mass_free_gas_g": "mass_g",
    }
    for name, maps, labels in mass.frames:
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name.lower() == "static"
        ):
            raise ValueError(
                "Export frame names must be safe, non-reserved file stems."
            )
        if name.casefold() in seen:
            raise ValueError(f"Duplicate simulator frame name: {name}")
        seen.add(name.casefold())
        corner = (maps.mass.origin[0], maps.mass.origin[1] - maps.mass.dimensions[0])
        if not np.allclose(corner, lower_left):
            raise ValueError(
                f"The mass maps of {name} do not start at the canvas's lower left "
                "corner, so their cells would not be the geometry's."
            )
        if labels.shape != (rows, cols) or any(
            np.shape(getattr(maps, field).img) != (rows, cols)
            for field in (*fields.values(), "concentration_aq", "saturation_g")
        ):
            raise ValueError("Every mass and phase field must match the run geometry.")
        row = dict(frame=name)
        if mass.sources is not None:
            if name not in mass.sources:
                raise ValueError(f"Missing source/time metadata for {name}")
            row.update(mass.sources[name].row())
        totals = {}
        for field, attribute in fields.items():
            values = np.asarray(getattr(maps, attribute).img, dtype=float)
            totals[field] = float(
                np.sum(
                    np.where(np.isfinite(values), values, 0)
                    * mass.geometry.pore_volume_m3
                    * 1000,
                    dtype=float,
                )
            )
            row["source_" + field] = totals[field]
        for size, copy in copies.items():
            cells = copy(maps, labels)
            for field, expected in totals.items():
                found = float(np.sum(cells[field], dtype=float))
                if not np.isclose(found, expected, rtol=1e-9, atol=1e-12):
                    raise ValueError(
                        f"The {size} export does not conserve {field} for {name}."
                    )
                row[size + "_" + field] = found
            np.savez_compressed(
                folder / size / f"{name}.npz",
                **{k: v[::-1] for k, v in cells.items()},
            )
        index.append(row)
    if mass.sources is not None and set(mass.sources) != {
        row["frame"] for row in index
    }:
        raise ValueError("Source metadata must correspond exactly to exported frames.")
    columns = list(index[0]) if index else ["frame"]
    write_rows(folder / "index.csv", index, columns)
    manifest = dict(
        frames=len(index),
        index="index.csv",
        grids=grids,
        axes="rows: y from the floor up; columns: x from the left",
        source_positions="uniform lower-left aligned",
        source_times=(
            "recorded capture and injection origin"
            if mass.sources is not None
            else "not supplied"
        ),
        static_geometry="observed source pore volume and area; not completed simulator sand",
        unknown_mass="Known finite sum only; read each mass_known_fraction. NaN coverage is not known zero.",
        units=dict(
            mass="g per cell",
            concentration_aq="1 (share of local CO2 solubility)",
            saturation_g="1",
            pore_volume_l="L",
            domain_area_m2="m2",
            fractions="1",
            elapsed_s="s from injection start",
        ),
    )
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return len(index)


README = """# Simulator package

{runs} runs; resolutions: {sizes}. Start with manifest.json.

## Frame

Every position is in the rig's measured frame, the frame of the mass maps: metres
from the inner corner of the left wall and the floor, x to the right and y up, over
the frame's {width:g} m by {height:g} m. Each cell holds the pixels whose
coordinates lie in it, on the grid on which the drawing is filled.
{notes}
## Geometry

geometry/simulator_geometry_<resolution>.npz holds the rig's porous domain on
cells (read with numpy.load, allow_pickle=False); its JSON companion lists each
array with its unit. Rows run from the floor up (y increasing) and columns from
the left (x increasing). Resolution is <size>mm for square metric cells, or full
for native rectangular cells. Cell centres lie at (index + 0.5) times each axis's
pitch from the rig's lower left inner corner; the JSON records both pitches for
full resolution. A cell with a positive domain_area_m2 holds porous medium;
outside it the intensive values are NaN. Volumes are m3 per cell, depth m, porosity
pore over bulk volume. Pixels are assigned to cells by their centres, so a cell's
domain_area_fraction may exceed 1. The facies fractions are shares of the cell's
domain area. Sand hidden behind fixtures is taken from the drawing beneath them
(drawn_behind). A pixel inside the
frame that no layer of the drawing covers, e.g. where the camera does not see the
rig, takes the nearest layer; the manifest counts these pixels by the layer they
took (filled_pixels_by_layer). A layer of an excluded facies, e.g. silicone
along the walls, lies outside the domain.

geometry/sand_map.json is the drawing the geometry is made from: anchors in metres
(x right, y up from the lower left inner corner), the polygons of each layer on
them, and the facies of each layer. geometry/facies_properties.csv is the facies
table as given, and depth_measurements.csv, when given, the depth measurements.

{own}

## Protocols

runs/<run>/ holds each run's injection, pressure_temperature and overflow tables as
their sources hold them, and overflow_reviewed.csv: the water displaced at each
usable scale reading in mL, against hours from the readings' own injection start
(in the manifest). Keep each table's own time origin when aligning them. The
manifest gives the flow controller's reference conditions and the gas density at
them.
{borrowed}{mass}"""
"""The package's README, filled in by :func:`write_simulator_package`."""
BORROWED_README = """
A run without usable scale readings of its own borrows the reviewed curve of
another run by the rule of the mass maps. Of the runs with usable readings that it
may borrow from (overflow_donors_allowed in the manifest, else every run of the
package), those with its sand map and a recipe within the overflow's tolerance
qualify, and the one nearest to it in atmospheric pressure is taken: the least mean
absolute difference of the two pressure records from its injection start on. A run
that no run qualifies for borrows from the run nearest in pressure alone, and the
manifest and the list below say so. Where runs qualify but none has atmospheric
pressure over the comparison window, it borrows none, as the mass maps' estimate
then has none. The curve is in overflow_reviewed_borrowed_from_<run>.csv, whose
column borrowed_from names that run; its times count from that run's own injection
start. Borrowed: {runs}. Without a curve: {none}.
"""
"""The README's paragraph on borrowed overflow curves, for a package with runs that
lack their own."""
MASS_README = """
## Mass maps

runs/<run>/mass/<resolution>/<frame>.npz holds the frame's masses per cell in g, each
with the share of the cell's pore volume whose mass is known, the pore-volume
weighted concentration and saturation, and the phase area fractions; static.npz
holds observed source pore volume and domain area, distinct from completed sand
behind fixtures. Resolution is full (the geometry's native rectangular cells), or
<size>mm. manifest.json gives the exact extent, shape, pitches, orientation and
units. index.csv identifies each supplied source session/image and capture time,
seconds from injection start, and source/export mass totals at every resolution.
Unknown coverage is never interpreted as zero mass. Rows run from the floor up.
concentration_aq is the stored dimensionless share of local CO2 solubility.
When the native CO2 state is supplied, concentration_aq_g_l gives g/L: each
source pixel is converted at its local pressure and temperature before averaging.
"""
"""The README's section on mass maps, for a package that holds them."""


def write_simulator_package(
    folder: Path,
    series: SimulatorSeries,
    runs: Mapping[str, SimulatorRun],
    settings: SimulatorExportSettings,
    overflow: darsia.OverflowSettings,
) -> dict:
    """Write the simulator package of a series' runs into a folder.

    The common geometry is made from the common drawing; a run with an adjustment
    gets its own, from the drawing moved by it, in its own folder. Each geometry is
    written at every cell size (:func:`geometry_cells`) with its drawing. Each run's
    protocols are copied, and its reviewed overflow and mass maps are written where
    it has them. A run without usable scale readings of its own borrows the reviewed
    curve of a donor by the mass maps' rule (:func:`darsia.ranked_donors`): of the
    runs with usable readings it may borrow from, those with its sand map and a
    recipe within the overflow's tolerance, the one nearest in atmospheric pressure;
    a run with an adjustment has a sand map of its own. Without such a run, it
    borrows from the run nearest in pressure alone, and the manifest and README say
    so; where such runs lack atmospheric pressure over the comparison window, it
    borrows none, as the mass maps' estimate has none. The curve is written with the
    donor's name in the file's name and in a column. The manifest records the frame,
    which geometry each run has and the pixels its drawing does not cover by the
    layer they took, its protocol files and their rows, the injection start, the
    time origin of the scale readings, why unusable readings are not reviewed, the
    runs that borrow, from which run and whether by pressure alone, the runs it may
    borrow from that the package lacks, why a run borrows none, and the flow
    reference.

    Parameters
    ----------
    folder : Path
        The package's folder, e.g. the partial folder of a result's publication.
    series : SimulatorSeries
        What the runs share.
    runs : mapping
        The runs, by name.
    settings : SimulatorExportSettings
        The settings of the export.
    overflow : darsia.OverflowSettings
        The settings of the overflow, which read the scale readings.

    Returns
    -------
    dict
        The manifest, also written as ``manifest.json``.

    Raises
    ------
    ValueError
        If a facies of the sand has no porosity in (0, 1] in the facies table, a
        run's recipe lacks an entry the overflow's recipe tolerance compares, a
        protocol's row has another number of cells than its header, or the
        atmospheric pressure record of a run that borrows or may lend is not
        increasing, finite and positive.

    """
    folder = Path(folder)
    table = pd.read_csv(series.facies_properties)
    porosity = dict(zip(table["id"].astype(int), table["porosity"].astype(float)))
    lacking = [f for f in settings.porous_facies if not 0 < porosity.get(f, 0) <= 1]
    if lacking:
        raise ValueError(f"The sand facies {lacking} have no porosity in (0, 1].")
    for name, run in runs.items():
        unknown = sorted(set(overflow.recipe_tolerance) - set(run.recipe))
        if unknown:
            raise ValueError(
                f"The recipe of {name} lacks {unknown}, which the overflow's recipe "
                "tolerance compares."
            )
    shared = folder / "geometry"
    common, filled = _write_geometry(
        folder, shared, series.drawing, series, porosity, settings
    )
    shutil.copyfile(series.facies_properties, shared / "facies_properties.csv")
    if series.depth_measurements is not None:
        shutil.copyfile(series.depth_measurements, shared / "depth_measurements.csv")

    entries, donor_runs, curves = {}, {}, {}
    for name, run in runs.items():
        target = folder / "runs" / name
        entry = dict(own_geometry=run.adjustment is not None, geometry=common)
        if run.adjustment is not None:
            drawing = series.drawing.moved(run.adjustment)
            entry["geometry"], entry["filled_pixels_by_layer"] = _write_geometry(
                folder, target, drawing, series, porosity, settings
            )
        injection = darsia.InjectionProtocol(run.injection, series.reference_density)
        entry["injection_start"] = injection.df["start"].min().isoformat()
        protocols = dict(
            injection=run.injection, pressure_temperature=run.pressure_temperature
        )
        if run.overflow is not None:
            protocols["overflow"] = run.overflow
        entry["protocols"], entry["rows"], tables = {}, {}, {}
        for kind, source in protocols.items():
            tables[kind] = _table(source)
            columns = settings.overflow_columns if kind == "overflow" else None
            path = target / f"{kind}.csv"
            entry["rows"][kind] = _write_table(path, tables[kind], columns)
            entry["protocols"][kind] = path.relative_to(folder).as_posix()
        volume = None
        if run.overflow is not None:
            origin, volume, curve = _reviewed_overflow(
                run.overflow, tables["overflow"], overflow
            )
            entry["overflow_time_origin"] = origin.isoformat()
            if curve:
                curves[name] = (curve, origin)
                path = target / "overflow_reviewed.csv"
                write_rows(path, curve, list(curve[0]))
                entry["protocols"]["overflow_reviewed"] = path.relative_to(
                    folder
                ).as_posix()
                entry["rows"]["overflow_reviewed"] = len(curve)
            else:
                entry["overflow_not_usable"] = list(volume.reasons)
        # A run with an adjustment has a sand map of its own.
        sand_map = "" if run.adjustment is None else name
        donor_runs[name] = _donor_run(run, sand_map, injection, volume)
        if run.mass is not None:
            entry["mass_frames"] = write_simulator_mass(
                target / "mass", run.mass, series.canvas, settings.cell_sizes_mm
            )
        entries[name] = entry

    # The mass maps' donor rule among the runs a run may borrow from; without a run
    # of its sand map and recipe among them, the one nearest in pressure alone.
    borrowed, unmatched, not_borrowed = {}, {}, {}
    for name, entry in entries.items():
        if name in curves:
            continue
        allowed = runs[name].donors
        if allowed is not None:
            entry["overflow_donors_allowed"] = sorted(set(allowed) & set(runs))
            absent = sorted(set(allowed) - set(runs))
            if absent:
                entry["overflow_donors_not_in_package"] = absent
        pool = {k: v for k, v in donor_runs.items() if allowed is None or k in allowed}
        pool[name] = donor_runs[name]
        for key, candidate in pool.items():
            usable = candidate.volume is not None and not candidate.volume.reasons
            if key == name or usable:
                _check_pressure(key, candidate)
        donors, reason = _ranked_donors(name, pool, overflow)
        if not donors:
            entry["overflow_not_borrowed"] = not_borrowed[name] = reason
            continue
        if reason is not None:
            entry["overflow_donor_unmatched"] = unmatched[name] = reason
        curve, origin = curves[donors[0]]
        rows = [dict(row, borrowed_from=donors[0]) for row in curve]
        path = (
            folder / "runs" / name / f"overflow_reviewed_borrowed_from_{donors[0]}.csv"
        )
        write_rows(path, rows, list(rows[0]))
        entry["protocols"]["overflow_reviewed"] = path.relative_to(folder).as_posix()
        entry["rows"]["overflow_reviewed"] = len(rows)
        entry["overflow_borrowed_from"] = borrowed[name] = donors[0]
        entry["overflow_donor_time_origin"] = origin.isoformat()

    own = sorted(name for name, entry in entries.items() if entry["own_geometry"])
    height, width = series.canvas.dimensions
    manifest = dict(
        format=FORMAT,
        frame=dict(
            origin="inner corner of the left wall and the floor",
            x="right",
            y="up",
            unit="m",
            width=width,
            height=height,
        ),
        cell_sizes_mm=[
            None if size is None else float(size) for size in settings.cell_sizes_mm
        ],
        geometry=common,
        filled_pixels_by_layer=filled,
        runs_with_own_geometry=own,
        missing_own_overflow=sorted(set(entries) - set(curves)),
        borrowed_overflow=borrowed,
        borrowed_overflow_unmatched=dict(sorted(unmatched.items())),
        not_borrowed_overflow=dict(sorted(not_borrowed.items())),
        flow_reference=dict(
            asdict(settings.flow_reference), gas_density_kg_m3=series.reference_density
        ),
        runs=entries,
    )
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (folder / "README.md").write_text(
        README.format(
            runs=len(entries),
            sizes=", ".join(
                "native/full" if size is None else f"{size:g} mm"
                for size in settings.cell_sizes_mm
            ),
            width=width,
            height=height,
            notes="".join(f"\n{note}\n" for note in settings.notes),
            borrowed=(
                BORROWED_README.format(
                    runs=", ".join(
                        f"{r} from {d}"
                        + (f" ({unmatched[r]})" if r in unmatched else "")
                        for r, d in sorted(borrowed.items())
                    )
                    or "none",
                    none=", ".join(
                        f"{r} ({why})" for r, why in sorted(not_borrowed.items())
                    )
                    or "none",
                )
                if borrowed or not_borrowed
                else ""
            ),
            own=(
                "Every run shares this geometry: no run has an adjustment of its own."
                if not own
                else "These runs differ from the common geometry, because their sand "
                "settled; each has its own geometry and drawing in runs/<run>/: "
                + ", ".join(own)
                + "."
            ),
            mass=(
                MASS_README
                if any(entry.get("mass_frames") for entry in entries.values())
                else ""
            ),
        ),
        encoding="utf-8",
    )
    return manifest
