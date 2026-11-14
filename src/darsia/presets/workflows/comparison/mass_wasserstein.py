"""Wasserstein distances between runs' mass maps, on metric cells.

At one time, each run's frame (its mass maps, e.g. of :func:`darsia.mass_maps`)
goes onto square metric cells of one size by the one scaling method, and each pair
of runs is compared in each region by the official Bregman solver of
:func:`darsia.wasserstein_distance`, with the transport weight of a reference run:

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
- A region's cells are those holding a pixel of it in the reference or in either
  run. Transport runs only through those with a known weight, and no flux leaves
  them (the solver's ``region``); mass on a cell without a known weight, and
  reference pixels of porous material without a known permeability (e.g., free
  water), are refused.
- Both runs' masses are scaled to their mean before the solve. Where neither run
  holds mass the distance is zero.
- As in the official comparison, each distance is also given over the mean mass
  and over two scales of the time (:func:`distance_scales`): the injected mass
  times the radius of a disc that holds it dissolved, and times the rig's mean
  dimension.

A pair is ``"missing"`` when a run has no frame at the time, ``"unknown"`` when a
run holds unknown mass in the region, ``"one_empty"`` when only one run holds
mass there, and else ``"success"`` or ``"not_converged"``, as the solver reports
(:class:`WassersteinDistance`). Every setting is a field of
:class:`WassersteinSettings`. Masses are in kg here, as in the official comparison's
table (the masses per domain are in g).

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

import darsia
from darsia.multiphase.gas_allocation import NEIGHBOURS
from darsia.multiphase.mass_maps import RunGeometry
from darsia.presets.workflows.analysis.domains import (
    AnalysisDomain,
    domain_pixels,
    injected_mass,
)

_SOLVER = {
    "mobility_mode": darsia.MobilityMode.CELL_BASED,
    "l1_mode": darsia.L1Mode.RAVIART_THOMAS,
    "linear_solver": "direct",
    "return_info": True,
}
"""Discretisation and linear solver of the Bregman solver, those of the official
comparison."""
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
    empty_mass_kg: float
    """A region's mass [kg] at most this counts as no mass."""
    max_iterations: int
    """Largest number of iterations of the solver."""
    distance_tolerance: float
    """Relative change of the distance at which the solver stops."""
    increment_tolerance: float
    """Relative Bregman increment at which the solver stops."""
    bregman_updates: tuple[int, ...]
    """Iterations at which the solver updates its Bregman regularisation."""


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


@dataclass(frozen=True)
class WassersteinDistance:
    """The Wasserstein distance of two runs' mass in one region at one time."""

    time_h: float
    """The time [h]."""
    run_1: str
    """The first run."""
    run_2: str
    """The second run, after the first in order."""
    domain_id: str
    """The compared region, by its domain identity."""
    status: Literal["success", "not_converged", "missing", "unknown", "one_empty"]
    """``"success"``; ``"not_converged"`` when the solver stopped at its largest
    number of iterations; ``"missing"`` when a run has no frame at the time;
    ``"unknown"`` when a run holds unknown mass in the region; ``"one_empty"``
    when only one run holds mass in the region."""
    distance_kg_m: float = np.nan
    """Distance [kg m] of the two masses, each scaled to their mean; zero where
    neither run holds mass, NaN without a distance."""
    mass_1_kg: float = np.nan
    """Mass [kg] of the first run in the region; NaN when missing or unknown."""
    mass_2_kg: float = np.nan
    """Mass [kg] of the second run in the region; NaN when missing or unknown."""
    unknown_pixels: tuple[int, int] = (0, 0)
    """Pixels of unknown mass of each run in the region."""
    filled_pixels: tuple[int, int] = (0, 0)
    """Pixels of each run in the region whose mass the hole rule estimated."""
    normalized_plume: float = np.nan
    """The distance over the plume scale of the time (:class:`DistanceScales`)."""
    normalized_rig: float = np.nan
    """The distance over the rig scale of the time."""
    normalized_mass: float = np.nan
    """The distance over the mean of the two masses [m]."""
    expected_mass_kg: float = np.nan
    """Mass [kg] injected into the region by the time; NaN for a region without an
    injection of its own (a sand layer, the sand), and when missing."""


@dataclass(frozen=True)
class DistanceScales:
    """What the distances of one time are normalised by, as in the official
    comparison."""

    plume_kg_m: float
    """The injected mass [kg] times the radius [m] of a disc of the mean depth that
    holds it dissolved at the mean solubility."""
    rig_kg_m: float
    """The injected mass [kg] times the mean of the rig's two dimensions [m]."""
    injected_kg: dict[str, float]
    """Mass [kg] injected into each region, by its domain identity; NaN for a region
    without an injection of its own."""


def distance_scales(
    time_h: float,
    injection: darsia.InjectionProtocol,
    co2: darsia.CO2MassAnalysis,
    depth: np.ndarray,
    dimensions: Sequence[float],
    domains: Sequence[AnalysisDomain],
) -> DistanceScales:
    """The scales of one time's distances, as the official comparison computes them.

    Parameters
    ----------
    time_h : float
        The time [h] since the first injection start.
    injection : darsia.InjectionProtocol
        The reference run's injection protocol.
    co2 : darsia.CO2MassAnalysis
        The reference run's CO2 at the experiment's start; its mean solubility
        dissolves the injected mass.
    depth : array
        The reference rig's depth [m] of each pixel.
    dimensions : sequence of float
        The reference rig's two dimensions [m].
    domains : sequence of AnalysisDomain
        The compared regions. Each receives the domains' injected mass
        (:func:`~darsia.presets.workflows.analysis.domains.injected_mass`): all of
        it for the whole domain, that of the injection points inside a box (edges
        included, the protocol's rule); NaN for a region without an injection of
        its own (a sand layer, the sand).

    Returns
    -------
    DistanceScales
        The scales.

    """
    mass = injection.injected_mass(time=time_h)
    volume = mass / np.average(co2.solubility_co2)
    area = volume / np.average(depth)
    radius = np.sqrt(area / np.pi)
    injected = {}
    for domain in domains:
        own = injected_mass(domain, injection, time_h)
        injected[domain.domain_id] = np.nan if own is None else own
    return DistanceScales(
        plume_kg_m=radius * mass,
        rig_kg_m=mass * (0.5 * (dimensions[0] + dimensions[1])),
        injected_kg=injected,
    )


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


def _distance(
    first: RegionMass,
    second: RegionMass,
    weight: RegionWeight,
    reference: WassersteinReference,
    scales: DistanceScales,
    settings: WassersteinSettings,
) -> dict:
    """The status, distances and masses of one region of a pair of frames."""
    region = weight.own | (first.pixels > 0) | (second.pixels > 0)
    if np.any(region & weight.porous):
        raise ValueError(
            "A compared region holds porous material without a known permeability "
            "(e.g., free water)"
        )
    active = region & np.isfinite(weight.weight)
    if np.any(region & ~active & ((first.mass.img != 0) | (second.mass.img != 0))):
        raise ValueError("A compared region has mass where no permeability is known")
    if not active.any():
        raise ValueError("A compared region has no cell with a known permeability")
    rows, cols = np.nonzero(active)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    mass_1, mass_2 = first.mass.subregion(crop), second.mass.subregion(crop)
    integral_1, integral_2 = mass_1.integral(), mass_2.integral()
    empty_1, empty_2 = (
        np.isclose(integral, 0, atol=settings.empty_mass_kg)
        for integral in (integral_1, integral_2)
    )
    masses = {"mass_1_kg": float(integral_1), "mass_2_kg": float(integral_2)}
    if empty_1 and empty_2:
        return dict(
            masses,
            status="success",
            distance_kg_m=0.0,
            normalized_plume=0.0,
            normalized_rig=0.0,
            normalized_mass=0.0,
        )
    if empty_1 or empty_2:
        return dict(masses, status="one_empty")
    average = 0.5 * (integral_1 + integral_2)
    mass_1 *= average / integral_1
    mass_2 *= average / integral_2
    cell_weight = darsia.ScalarImage(
        weight.weight,
        space_dim=2,
        indexing="ij",
        dimensions=reference.dimensions,
        origin=reference.origin,
    )
    distance, info = darsia.wasserstein_distance(
        mass_1,
        mass_2,
        weight=cell_weight.subregion(crop),
        method="bregman",
        options=dict(
            _SOLVER,
            num_iter=settings.max_iterations,
            tol_distance=settings.distance_tolerance,
            tol_increment=settings.increment_tolerance,
            bregman_update=lambda iteration: iteration in settings.bregman_updates,
            region=active[crop],
        ),
    )
    return dict(
        masses,
        status="success" if info["converged"] else "not_converged",
        distance_kg_m=float(distance),
        normalized_plume=float(distance / scales.plume_kg_m),
        normalized_rig=float(distance / scales.rig_kg_m),
        normalized_mass=float(distance / average),
    )


def wasserstein_distances(
    time_h: float,
    frames: Mapping[str, Mapping[str, RegionMass] | None],
    reference: WassersteinReference,
    scales: DistanceScales,
    settings: WassersteinSettings,
) -> list[WassersteinDistance]:
    """The Wasserstein distances of every pair of runs in every region at one time.

    The pairs are each run with every later run in order (by name), and for each
    pair every region, in that order.

    Parameters
    ----------
    time_h : float
        The time [h], e.g. a production time of the series.
    frames : mapping of str to mapping or None
        Each run's frame at the time (:func:`frame_cells`), by run; None for a run
        without a frame at the time.
    reference : WassersteinReference
        The comparison cells and the weights of the regions.
    scales : DistanceScales
        The scales of the time (:func:`distance_scales`).
    settings : WassersteinSettings
        The empty mass and the solver's settings.

    Returns
    -------
    list of WassersteinDistance
        One distance per pair of runs and region.

    """
    results = []
    for run_1, first in frames.items():
        for run_2, second in frames.items():
            if run_1 >= run_2:
                continue
            for name, weight in reference.weights.items():
                if first is None or second is None:
                    results.append(
                        WassersteinDistance(time_h, run_1, run_2, name, "missing")
                    )
                    continue
                one, two = first[name], second[name]
                counts = {
                    "unknown_pixels": (one.unknown, two.unknown),
                    "filled_pixels": (one.filled, two.filled),
                    "expected_mass_kg": float(scales.injected_kg[name]),
                }
                if one.unknown or two.unknown:
                    results.append(
                        WassersteinDistance(
                            time_h, run_1, run_2, name, "unknown", **counts
                        )
                    )
                    continue
                fields = _distance(one, two, weight, reference, scales, settings)
                results.append(
                    WassersteinDistance(time_h, run_1, run_2, name, **fields, **counts)
                )
    return results
