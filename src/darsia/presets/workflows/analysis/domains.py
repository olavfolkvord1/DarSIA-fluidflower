"""Analysis domains: the parts of a run's grid whose CO2 the analyses read.

One record (:class:`AnalysisDomain`) names a domain as the window saves it, and every
analysis of the masses per domain (the masses, the events, the ensemble, the
Wasserstein distances) takes its pixels and its injected mass from here:

- ``whole``: the whole domain of the run;
- ``box``: a box in millimetres of the grid's coordinates (x right, y up), as drawn
  in the window;
- ``layer``: a sand layer by its label in the run's sand map, e.g. the free water;
- ``sand``: the sand layers, or the sand layers in a box.

A pixel lies in a box by its centre (:func:`darsia.pixel_centres`), each interval
half-open, so that boxes sharing a side count no pixel twice: the rule of the metric
cells (:func:`darsia.metric_cells`). An injection point lies in a box by the closed
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

import darsia

KINDS = ("whole", "box", "layer", "sand")
"""The kinds of a domain: the whole domain, a box, a sand layer, the sand layers."""
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

    @property
    def bounds_m(self) -> tuple[float, float, float, float] | None:
        """The box's bounds x0, x1, y0, y1 [m]; None without a box."""
        if self.bounds_mm is None:
            return None
        x0, x1, y0, y1 = (float(v) for v in np.asarray(self.bounds_mm, np.float64))
        return x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000


def domain_pixels(
    domain: AnalysisDomain,
    grid: darsia.Image,
    labels: np.ndarray,
    sand_labels: Sequence[int] = (),
) -> np.ndarray:
    """The pixels of a domain on a run's grid.

    A pixel lies in a box when its centre does, each interval half-open
    (``x0 <= x < x1``, ``y0 <= y < y1``); a sand layer holds the pixels of its
    label, and the sand those of the sand layers' labels.

    Parameters
    ----------
    domain : AnalysisDomain
        The domain.
    grid : darsia.Image
        An image on the run's grid, e.g. its sand map, whose pixel centres
        (:func:`darsia.pixel_centres`) the boxes read.
    labels : array
        The label of each pixel, e.g. the sand map's.
    sand_labels : sequence of int
        The labels of the sand layers, which the sand reads.

    Returns
    -------
    array
        Whether each pixel lies in the domain (bool); the caller limits it to the
        pixels it counts, e.g. the run's domain.

    Raises
    ------
    ValueError
        If the sand is asked for without the labels of its layers.

    """
    labels = np.asarray(labels)
    if domain.kind == "layer":
        inside = labels == domain.label
    elif domain.kind == "sand":
        if len(sand_labels) == 0:
            raise ValueError(f"The sand needs the labels of its layers: {domain.name}")
        inside = np.isin(labels, sand_labels)
    else:
        inside = np.ones(labels.shape, bool)
    if domain.bounds_mm is not None:
        x, y = darsia.pixel_centres(grid)
        x0, x1, y0, y1 = domain.bounds_m
        inside &= ((y >= y0) & (y < y1))[:, None] & ((x >= x0) & (x < x1))[None, :]
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
