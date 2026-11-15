"""Simulator export: a series' sand geometry, protocols and mass maps as plain files.

A simulator reads the rig on a grid of metric cells: which sand lies in each cell,
its porosity and depth, and the volumes that hold the CO2.

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
:func:`darsia.cell_sums`), per cell size (:func:`geometry_cells`). The arrays are
written with their rows from the floor up (y increasing) and their columns from the
left (x increasing), the origin at the rig's lower left inner corner, as a
simulator's grid runs.

"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

import darsia
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    rasterize_sand_map,
    rig_grid,
)

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
class SimulatorExportSettings:
    """Settings of the simulator export; the ``[export.simulator]`` table."""

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
    cell_sizes_mm: Sequence[float]
    """Sizes [mm] of the metric cells of the package, one geometry each."""
    precision: Literal["float32", "float64"]
    """Precision of the package's arrays; areas and volumes are summed in double
    precision."""


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


def geometry_cells(
    completed: CompletedGeometry,
    depth_m: np.ndarray,
    porosity: Mapping[int, float],
    dimensions: Sequence[float],
    cell_size_mm: float,
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
    cells = darsia.metric_cells(domain.shape, list(dimensions), cell_size_mm)
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
    cell_area = (cell_size_mm / 1000.0) * (cell_size_mm / 1000.0)
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
