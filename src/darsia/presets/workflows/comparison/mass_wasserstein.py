"""Wasserstein comparison of runs' mass maps: their mass and weight on metric cells.

At one time, each run's frame (its mass maps, e.g. of :func:`darsia.mass_maps`)
goes onto square metric cells of one size by the one scaling method, region by
region, beside the transport weight of a reference run on the same cells:

- A frame's mass is the mass of its counted pixels (:class:`darsia.RunGeometry`):
  the mass per pore volume times the pore volume, summed over each cell in double
  precision (:func:`darsia.cell_sums`) and kept per area in the maps' dtype. A
  counted pixel whose mass, concentration or saturation is not finite, or whose
  phase is unknown, holds unknown mass.
- A small hole of unknown pixels in the sand layers may borrow the mass of its
  known neighbours (:func:`fill_holes`), for the comparison only; the mass maps
  are unchanged.
- A compared region is an analysis domain
  (:class:`~darsia.presets.workflows.analysis.domains.AnalysisDomain`): the whole
  domain, a box (a pixel by its centre, half-open), a sand layer, the sand layers,
  or the sand layers in a box. Each run counts its own pixels of the region, on its
  own sand map (:func:`~darsia.presets.workflows.analysis.domains.domain_pixels`).
- The weight of a reference pixel is the largest known permeability over its own.
  Under the colour checker the sand layer beside it continues horizontally in each
  row, with its label and weight. A cell's weight is the mean over the region's
  reference pixels in it, or over all its reference pixels where the reference has
  none of the region.

Every setting is a field of :class:`WassersteinSettings`. Masses are in kg here, as
in the official comparison's table (the masses per domain are in g).

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

import darsia
from darsia.multiphase.gas_allocation import NEIGHBOURS
from darsia.multiphase.mass_maps import RunGeometry
from darsia.presets.workflows.analysis.domains import AnalysisDomain, domain_pixels

_CLIP = 1e-6
"""Share of the smallest other permeability to which permeabilities close to zero
are clipped, as in the official comparison."""


@dataclass(frozen=True)
class WassersteinSettings:
    """Settings of the Wasserstein distances; one configuration table."""

    cell_mm: float
    """Size [mm] of the square cells the mass maps are summed onto."""
    hole_pixels: int
    """Largest hole of unknown pixels that borrows its neighbours' mass; 0 fills
    none."""


@dataclass(frozen=True)
class RegionWeight:
    """The reference's transport weight of one region, per comparison cell."""

    own: np.ndarray
    """Cells holding a reference pixel of the region."""
    weight: np.ndarray
    """Weight of each cell; NaN where no reference pixel has a known weight."""
    porous: np.ndarray
    """Cells holding a reference pixel of the region of porous material without a
    known permeability."""


@dataclass(frozen=True)
class WassersteinReference:
    """The comparison cells and the reference's weights, once per comparison."""

    grid: tuple
    """Shape, dimensions and origin of the grid every compared run shares."""
    cells: darsia.MetricCells
    """The comparison cells (:func:`darsia.metric_cells`)."""
    dimensions: list[float]
    """Height and width [m] of the cells' extent."""
    origin: list[float]
    """Top left corner [m] of the cells' extent."""
    sand_labels: tuple[int, ...]
    """Labels of the sand layers on the sand maps."""
    domains: dict[str, AnalysisDomain]
    """The compared regions, by their domain identity."""
    weights: dict[str, RegionWeight]
    """The weight of each region, by its domain identity."""


@dataclass(frozen=True)
class RegionMass:
    """One frame's mass in one region, on the comparison cells."""

    mass: darsia.Image
    """Mass per area [kg/m2] of the region's own pixels, in the maps' dtype."""
    pixels: np.ndarray
    """Number of the region's pixels in each cell."""
    unknown: int
    """Pixels of the region with unknown mass."""
    filled: int
    """Pixels of the region whose mass the hole rule estimated."""


def _horizontal_sources(chart: np.ndarray, sand: np.ndarray) -> np.ndarray:
    """For each colour checker pixel, the column of the sand pixel continuing under it.

    Row by row, a run of chart pixels takes the nearest pixel left or right of the
    run that is not chart, if it is sand; with sand on both sides the nearer side,
    the left at equal distance. ``-1`` marks pixels that are not chart, and chart
    pixels with no sand beside their run.

    """
    source = np.full(chart.shape, -1, np.int32)
    rows = np.flatnonzero(chart.any(axis=1))
    if rows.size == 0:
        return source
    columns = chart.shape[1]
    row_chart, row_sand = chart[rows], sand[rows]
    column = np.arange(columns, dtype=np.int32)
    left = np.maximum.accumulate(np.where(row_chart, -1, column), axis=1)
    right = np.minimum.accumulate(
        np.where(row_chart, columns, column)[:, ::-1], axis=1
    )[:, ::-1]
    index = np.arange(rows.size)[:, None]
    left_sand = (left >= 0) & row_sand[index, np.clip(left, 0, columns - 1)]
    right_sand = (right < columns) & row_sand[index, np.clip(right, 0, columns - 1)]
    take_left = (
        row_chart & left_sand & (~right_sand | (column - left <= right - column))
    )
    take_right = row_chart & right_sand & ~take_left
    source[rows] = np.where(take_left, left, np.where(take_right, right, -1))
    return source


def wasserstein_reference(
    sand_map: darsia.Image,
    permeability: np.ndarray,
    porosity: np.ndarray,
    sand_labels: Sequence[int],
    chart_labels: Sequence[int],
    domains: Sequence[AnalysisDomain],
    settings: WassersteinSettings,
) -> WassersteinReference:
    """The comparison cells and the reference run's weight of each region.

    The weight of a pixel with a known (finite) permeability is the largest known
    permeability over its own, permeabilities close to zero clipped, in the
    permeability's dtype, as in the official comparison. A colour checker pixel
    without a known permeability takes the label and weight of the sand pixel whose
    layer continues under it in its row (the nearest side, the left at equal
    distance). Porous material without a known permeability is any other pixel
    without one whose porosity is not zero, and a chart pixel with no sand beside
    it. A cell's weight is the mean of its pixels' weights in double precision, as
    our production computed it.

    Parameters
    ----------
    sand_map : darsia.Image
        The reference run's sand map (layer labels) on the run grid.
    permeability, porosity : array
        The reference run's permeability and porosity of each pixel.
    sand_labels, chart_labels : sequence of int
        Labels of the sand layers and of the colour checker.
    domains : sequence of AnalysisDomain
        The compared regions, each once.
    settings : WassersteinSettings
        The size of the cells.

    Returns
    -------
    WassersteinReference
        The cells and the weights.

    """
    labels = np.asarray(sand_map.img)
    known = np.isfinite(permeability)
    values = permeability[known]
    smallest = np.min(values[~np.isclose(values, 0)])
    clip = _CLIP * smallest if np.isclose(np.min(values), 0) else 0
    ratio = np.max(values) / np.clip(values, clip, None)
    weight = np.full(labels.shape, np.nan, ratio.dtype)
    weight[known] = ratio

    chart = ~known & np.isin(labels, chart_labels)
    source = _horizontal_sources(chart, known & np.isin(labels, sand_labels))
    continued = source >= 0
    rows = np.nonzero(continued)[0]
    placed = labels.copy()
    placed[continued] = labels[rows, source[continued]]
    weight[continued] = weight[rows, source[continued]]
    porous = (~known & ~chart & (porosity != 0)) | (chart & ~continued)

    ids = [domain.domain_id for domain in domains]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Choose at least one region, each once")
    cells = darsia.metric_cells(labels.shape, sand_map.dimensions, settings.cell_mm)
    weight = weight.astype(np.float64)
    everywhere = darsia.cell_means(cells, weight)
    weights = {}
    for domain in domains:
        inside = domain_pixels(domain, sand_map, placed, sand_labels)
        own = darsia.cell_sums(cells, inside) > 0
        mean = darsia.cell_means(cells, np.where(inside, weight, np.nan))
        weights[domain.domain_id] = RegionWeight(
            own=own,
            weight=np.where(own, mean, everywhere),
            porous=darsia.cell_sums(cells, inside & porous) > 0,
        )
    left, top = (float(value) for value in sand_map.origin)
    bottom = top - float(sand_map.dimensions[0])
    return WassersteinReference(
        grid=_canvas(sand_map),
        cells=cells,
        dimensions=list(cells.dimensions),
        origin=[left, bottom + cells.dimensions[0]],
        sand_labels=tuple(int(label) for label in sand_labels),
        domains={domain.domain_id: domain for domain in domains},
        weights=weights,
    )


def _canvas(image: darsia.Image) -> tuple:
    """Shape, dimensions and origin of an image's grid."""
    return (
        tuple(image.img.shape[:2]),
        tuple(float(value) for value in image.dimensions),
        tuple(float(value) for value in image.origin),
    )


def fill_holes(
    density: np.ndarray,
    known: np.ndarray,
    support: np.ndarray,
    facies: np.ndarray,
    max_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass per pore volume for small holes of unknown pixels, from their neighbours.

    A hole is an 8-connected set of pixels whose mass is not known. It is filled
    whole when it has at most ``max_pixels`` pixels, does not touch the grid's
    edge, lies in ``support``, is of one facies, and its ring of neighbours lies in
    ``support`` and is of that facies. Each of its pixels then takes the mean mass
    per pore volume of its own neighbours outside the hole, summed in double
    precision in a fixed order; no filled pixel serves as a neighbour.

    Parameters
    ----------
    density : array
        Mass per pore volume of each pixel, e.g. the total mass of
        :func:`darsia.mass_maps`.
    known : array
        Pixels whose mass is known.
    support : array
        Pixels that may be filled or lend their mass, e.g. counted sand pixels.
    facies : array
        Facies of each pixel.
    max_pixels : int
        Largest hole filled.

    Returns
    -------
    tuple of arrays
        Rows, columns and mass per pore volume (double precision) of the filled
        pixels.

    """
    known = known & np.isfinite(density)
    height, width = density.shape
    holes, _ = ndimage.label(~known, structure=np.ones((3, 3), bool))
    sizes = np.bincount(holes.ravel())
    candidates = np.unique(holes[support & ~known])
    boxes = ndimage.find_objects(holes)
    rows, cols, values = [], [], []
    for hole in candidates[sizes[candidates] <= max_pixels]:
        r, c = np.nonzero(holes[boxes[hole - 1]] == hole)
        r, c = r + boxes[hole - 1][0].start, c + boxes[hole - 1][1].start
        if (
            r.min() == 0
            or c.min() == 0
            or r.max() == height - 1
            or c.max() == width - 1
        ):
            continue
        window = (slice(r.min() - 1, r.max() + 2), slice(c.min() - 1, c.max() + 2))
        inside = holes[window] == hole
        ring = ndimage.binary_dilation(inside, np.ones((3, 3), bool)) & ~inside
        near = support[window] & (facies[window] == facies[r[0], c[0]])
        if not (near[inside].all() and near[ring].all()):
            continue
        for row, col in zip(r, c):
            total, count = 0.0, 0
            for dr, dc in NEIGHBOURS:
                if known[row + dr, col + dc]:
                    total = total + np.float64(density[row + dr, col + dc])
                    count += 1
            rows.append(row)
            cols.append(col)
            values.append(total / count)
    return np.array(rows, int), np.array(cols, int), np.array(values, np.float64)


def frame_cells(
    maps: darsia.SimpleMassAnalysisResults,
    labels: np.ndarray,
    sand_map: darsia.Image,
    facies: np.ndarray,
    geometry: RunGeometry,
    reference: WassersteinReference,
    settings: WassersteinSettings,
) -> dict[str, RegionMass]:
    """One frame's mass in each region, on the comparison cells.

    A counted pixel's mass is its mass per pore volume times its pore volume where
    its mass, concentration and saturation are finite and its phase is known
    (:func:`darsia.known_mass`); elsewhere it is unknown. Holes of unknown pixels
    in the counted sand pixels borrow their neighbours' mass (:func:`fill_holes`, up
    to ``hole_pixels``).

    Parameters
    ----------
    maps : darsia.SimpleMassAnalysisResults
        The frame's mass maps, e.g. of :func:`darsia.mass_maps`.
    labels : array
        The frame's phase map (:class:`darsia.PhaseLabel`).
    sand_map : darsia.Image
        The run's sand map (layer labels) on the reference's grid.
    facies : array
        The run's facies of each pixel.
    geometry : RunGeometry
        The run's geometry: its counted pixels and their pore volume.
    reference : WassersteinReference
        The comparison cells and regions.
    settings : WassersteinSettings
        The hole rule's largest hole.

    Returns
    -------
    dict of str to RegionMass
        The frame's mass in each region, by its domain identity.

    """
    if _canvas(sand_map) != reference.grid:
        raise ValueError("A compared run is not on the reference's grid")
    mass = maps.mass.img
    known = darsia.known_mass(maps, labels)["mass_total"]
    volume = np.asarray(geometry.pore_volume_m3, np.float64) * 1000
    counted = geometry.counted
    sand = np.isin(sand_map.img, reference.sand_labels)
    grams = np.zeros(mass.shape, np.float64)
    taken = counted & known
    grams[taken] = np.asarray(mass[taken], np.float64) * volume[taken]
    filled = np.zeros(mass.shape, bool)
    if settings.hole_pixels:
        rows, cols, values = fill_holes(
            mass, known, counted & sand, facies, settings.hole_pixels
        )
        grams[rows, cols] = values * volume[rows, cols]
        filled[rows, cols] = True
    unknown = counted & ~known & ~filled

    area = (settings.cell_mm / 1000) ** 2
    result = {}
    for name, domain in reference.domains.items():
        inside = counted & domain_pixels(
            domain, sand_map, sand_map.img, reference.sand_labels
        )
        kg = darsia.cell_sums(reference.cells, np.where(inside, grams, 0.0)) / 1000
        result[name] = RegionMass(
            mass=darsia.ScalarImage(
                (kg / area).astype(mass.dtype),
                space_dim=2,
                indexing="ij",
                dimensions=reference.dimensions,
                origin=reference.origin,
            ),
            pixels=darsia.cell_sums(reference.cells, inside),
            unknown=int(np.count_nonzero(unknown & inside)),
            filled=int(np.count_nonzero(filled & inside)),
        )
    return result
