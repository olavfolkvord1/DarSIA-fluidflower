"""Analysis domains: the parts of a run's grid whose CO2 the analyses read.

One record (:class:`AnalysisDomain`) names a domain as the window saves it, and every
analysis of the masses per domain (the masses, the events, the ensemble, the
Wasserstein distances) takes its pixels and its injected mass from here:

- ``whole``: the whole domain of the run;
- ``box``: a box in millimetres of the grid's coordinates (x right, y up), as drawn
  in the window;
- ``layer``: a sand layer by its label in the run's sand map, e.g. the free water,
  or one of its parts (:func:`layer_parts`);
- ``sand``: the sand layers, or the sand layers in a box.

A sand layer can fall into pieces that touch only through other material, e.g.
where another layer cuts it off in a corner. A layer domain then names one of the
layer's parts on the run's sand map, by our production's rule (:func:`layer_parts`):
its largest piece is part 1, every other piece of at least a set area its next
part, and its smaller pieces (``part = "pieces"``) are summed in a domain of their
own. Without a part, a layer domain holds all of the layer.

A pixel lies in a box by its position on the rig (:func:`darsia.grid_positions`: by
the run's measured model of its grid, else its centre), each interval half-open, so
that boxes sharing a side count no pixel twice: the rule of the metric cells
(:func:`darsia.metric_cells`). An injection point lies in a box by the closed
rule of :meth:`darsia.InjectionProtocol.injected_mass`, edges included. A box whose
edge passes through an injection point is refused (:func:`injected_mass`), so that
the two rules never disagree.

The tables of the analyses of domains are written by one writer (:func:`write_rows`).

"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage

import darsia

KINDS = ("whole", "box", "layer", "sand")
"""The kinds of a domain: the whole domain, a box, a sand layer, the sand layers."""
PIECES = "pieces"
"""The part of a sand layer that holds its pieces too small to be parts."""
COMPONENTS = {"total": "mass", "aqueous": "mass_aq", "gas": "mass_g"}
"""The components of the CO2 (all of it, the dissolved CO2 and the free gas), each
with its field of the mass maps (:class:`darsia.SimpleMassAnalysisResults`)."""


@dataclass(frozen=True)
class AnalysisDomain:
    """A domain of an analysis, as the window saves it."""

    kind: Literal["whole", "box", "layer", "sand"]
    """The whole domain, a box, a sand layer, or the sand layers (in a box)."""
    domain_id: str
    """Identity of the domain, unique in an analysis."""
    name: str
    """Name of the domain."""
    bounds_mm: tuple[float, float, float, float] | None = None
    """Box, and the sand in a box: x0, x1, y0, y1 [mm] in the grid's coordinates, x
    right and y up."""
    label: int | None = None
    """Sand layer: its label in the sand map (1 to 255)."""
    part: int | Literal["pieces"] | None = None
    """Sand layer: one of its parts on the run's sand map (:func:`layer_parts`), 1
    the largest, or ``"pieces"`` for its pieces that are no part; None for all of
    the layer."""

    def __post_init__(self) -> None:
        """Refuse a domain that would be read as another one or as none."""
        if self.kind not in KINDS or not self.name.strip():
            raise ValueError(f"A domain needs its kind and name: {self.domain_id}")
        boxed = self.kind == "box" or self.bounds_mm is not None
        if boxed and len(self.bounds_mm or ()) != 4:
            raise ValueError(f"A box needs its four bounds: {self.name}")
        if boxed and self.kind not in ("box", "sand"):
            raise ValueError(f"Only a box or the sand in a box has bounds: {self.name}")
        if self.kind == "layer" and (
            type(self.label) is not int or not 0 < self.label < 256
        ):
            raise ValueError(f"A sand layer needs its label in the map: {self.name}")
        if self.kind != "layer" and self.label is not None:
            raise ValueError(f"Only a sand layer has a label: {self.name}")
        if self.part is not None and (
            self.kind != "layer"
            or not (self.part == PIECES or (type(self.part) is int and self.part >= 1))
        ):
            raise ValueError(
                f"Only a sand layer has a part, a number from 1 or {PIECES!r}: "
                f"{self.name}"
            )

    @property
    def bounds_m(self) -> tuple[float, float, float, float] | None:
        """The box's bounds x0, x1, y0, y1 [m]; None without a box."""
        if self.bounds_mm is None:
            return None
        x0, x1, y0, y1 = (float(v) for v in np.asarray(self.bounds_mm, np.float64))
        return x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000


def horizontal_sources(chart: np.ndarray, sand: np.ndarray) -> np.ndarray:
    """For each colour checker pixel, the column of the sand pixel continuing under it.

    Row by row, a run of chart pixels takes the nearest pixel left or right of the
    run that is not chart, if it is sand; with sand on both sides the nearer side,
    the left at equal distance. ``-1`` marks pixels that are not chart, and chart
    pixels with no sand beside their run.

    Parameters
    ----------
    chart, sand : array
        The pixels of the colour checker and of the sand layers.

    Returns
    -------
    array
        The column of each chart pixel's source (int32), or -1.

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


def layer_parts(
    labels: np.ndarray,
    sand_labels: Sequence[int],
    area_m2: np.ndarray,
    min_area_m2: float,
    chart_labels: Sequence[int] = (),
) -> np.ndarray:
    """Each pixel's part of its sand layer, by our production's rule.

    A piece of a sand layer is a face-connected (4-neighbour) region of its label,
    found after the colour checker's pixels took the layer that continues under
    them (:func:`horizontal_sources`), so that a layer running on behind the chart
    is one piece. A layer's pieces are ranked by their pixel count, the largest
    first, equal counts by their first pixel in raster order. The largest piece
    is the layer's part 1; every other piece whose area is at least
    ``min_area_m2`` is its next part; a smaller piece is no part.

    Parameters
    ----------
    labels : array
        The label of each pixel of the run's sand map.
    sand_labels : sequence of int
        The labels of the sand layers.
    area_m2 : array
        Each pixel's area [m2]; a piece's area is their sum.
    min_area_m2 : float
        The smallest area [m2] of a part other than the largest.
    chart_labels : sequence of int, optional
        The labels of the colour checker; none by default.

    Returns
    -------
    array
        On each pixel of a sand layer, the number of its piece's part (int32,
        from 1), 0 on a smaller piece; -1 on every other pixel, also on a chart
        pixel that a layer continues under.

    """
    labels = np.asarray(labels)
    sand = np.isin(labels, sand_labels)
    placed, continued = labels, np.zeros(labels.shape, bool)
    if len(chart_labels):
        source = horizontal_sources(np.isin(labels, chart_labels), sand)
        continued = source >= 0
        placed = labels.copy()
        placed[continued] = labels[np.nonzero(continued)[0], source[continued]]
    area = np.nan_to_num(np.asarray(area_m2, np.float64)).ravel()
    parts = np.full(labels.shape, -1, np.int32)
    faces = ndimage.generate_binary_structure(2, 1)
    for label in sorted({int(label) for label in sand_labels}):
        inside = placed == label
        found, count = ndimage.label(inside, structure=faces)
        if count == 0:
            continue
        pixels = np.bincount(found.ravel(), minlength=count + 1)[1:]
        sizes = np.bincount(found.ravel(), weights=area, minlength=count + 1)[1:]
        numbers, part = np.zeros(count + 1, np.int32), 0
        # Pieces are numbered by their first pixel in raster order, so a stable
        # sort by size breaks equal counts by that first pixel.
        for rank, piece in enumerate(np.argsort(-pixels, kind="stable")):
            if rank == 0 or sizes[piece] >= min_area_m2:
                part += 1
                numbers[piece + 1] = part
        parts[inside] = numbers[found[inside]]
    parts[continued] = -1
    return parts


def domain_pixels(
    domain: AnalysisDomain,
    grid: darsia.Image,
    labels: np.ndarray,
    sand_labels: Sequence[int] = (),
    positions: darsia.GridPositions | None = None,
    parts: np.ndarray | None = None,
) -> np.ndarray:
    """The pixels of a domain on a run's grid.

    A pixel lies in a box when its position does, each interval half-open
    (``x0 <= x < x1``, ``y0 <= y < y1``), compared in double precision, as our
    production compared its measured pixel positions; a sand layer holds the
    pixels of its label, a part of it those of its part (``parts``), and the sand
    those of the sand layers' labels.

    Parameters
    ----------
    domain : AnalysisDomain
        The domain.
    grid : darsia.Image
        An image on the run's grid, e.g. its sand map, whose pixel centres the
        boxes read without positions.
    labels : array
        The label of each pixel, e.g. the sand map's.
    sand_labels : sequence of int
        The labels of the sand layers, which the sand reads.
    positions : darsia.GridPositions, optional
        Each pixel's position on the rig, by the run's measured model of its grid;
        by default the grid's pixel centres (:func:`darsia.grid_positions`).
    parts : array, optional
        Each pixel's part of its sand layer (:func:`layer_parts`), which a part of
        a layer reads.

    Returns
    -------
    array
        Whether each pixel lies in the domain (bool); the caller limits it to the
        pixels it counts, e.g. the run's domain.

    Raises
    ------
    ValueError
        If the sand is asked for without the labels of its layers, a part of a
        layer without the parts, or a part that the layer does not have.

    """
    labels = np.asarray(labels)
    if domain.kind == "layer":
        inside = labels == domain.label
        if domain.part is not None:
            if parts is None:
                raise ValueError(f"A part of a layer needs its parts: {domain.name}")
            inside &= np.asarray(parts) == (0 if domain.part == PIECES else domain.part)
            if domain.part != PIECES and not inside.any():
                raise ValueError(
                    f"The sand layer {domain.label} has no part {domain.part} on "
                    f"the run's sand map: {domain.name}"
                )
    elif domain.kind == "sand":
        if len(sand_labels) == 0:
            raise ValueError(f"The sand needs the labels of its layers: {domain.name}")
        inside = np.isin(labels, sand_labels)
    else:
        inside = np.ones(labels.shape, bool)
    if domain.bounds_mm is not None:
        if positions is None:
            positions = darsia.grid_positions(grid)
        x, y = (np.asarray(v, np.float64) for v in (positions.x, positions.y))
        x0, x1, y0, y1 = domain.bounds_m
        inside &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return inside


def injected_mass(
    domain: AnalysisDomain, protocol: darsia.InjectionProtocol, time_h: float
) -> float | None:
    """The CO2 a protocol injected into a domain up to a time.

    The whole domain receives all of it, and a box what the injection points inside
    it inject, edges included (the protocol's closed rule). A sand layer and the
    sand have no injection of their own; each caller states what it takes instead.

    Parameters
    ----------
    domain : AnalysisDomain
        The domain.
    protocol : darsia.InjectionProtocol
        The run's injection protocol.
    time_h : float
        The time [h], as :meth:`darsia.InjectionProtocol.injected_mass` takes it.

    Returns
    -------
    float or None
        The mass [kg]; None for a domain without an injection of its own.

    Raises
    ------
    ValueError
        If an edge of a box passes through an injection point: the box's pixels
        (by centre, half-open) and its injection (closed) would disagree there.

    """
    if domain.kind == "whole":
        return protocol.injected_mass(time=time_h)
    if domain.kind != "box":
        return None
    x0, x1, y0, y1 = domain.bounds_m
    x = protocol.df["location_x"].to_numpy(np.float64)
    y = protocol.df["location_y"].to_numpy(np.float64)
    inside = (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)
    edge = inside & ((x == x0) | (x == x1) | (y == y0) | (y == y1))
    if edge.any():
        raise ValueError(
            f"An edge of the box {domain.name} passes through an injection point "
            f"{(float(x[edge][0]), float(y[edge][0]))}: the box counts pixels by "
            "their centre, half-open, and injection points with its edges"
        )
    box = darsia.CoordinateArray(np.array([[x0, y0], [x1, y1]]))
    return protocol.injected_mass(time=time_h, roi=box)


def write_rows(path: Path, rows: Iterable[Mapping], columns: Sequence[str]) -> None:
    """Write rows as one CSV table in the given columns.

    A value that is None is an empty cell, and a NumPy number is written as the
    Python number it holds, in full: under the legacy printing that a dependency of
    darsia sets, its text would keep only 12 digits.

    Parameters
    ----------
    path : Path
        The CSV file.
    rows : iterable of mappings
        The rows, each with no key but the columns.
    columns : sequence of str
        The table's columns, in order.

    """
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(
            {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
            for row in rows
        )
