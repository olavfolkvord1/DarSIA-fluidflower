"""Mass per analysis domain: the whole domain, boxes and sand layers.

The mass maps of a run's frames (e.g., :func:`darsia.mass_maps`, or the official
flash's maps, in the same form) are summed over each domain of an analysis, frame
by frame; no photograph is read and no phase or mass is computed again. A domain
(:class:`~darsia.presets.workflows.analysis.domains.AnalysisDomain`) is

- the whole domain of the run's geometry;
- a box in millimetres of the grid's coordinates (x right, y up), as drawn in the
  window: the pixels whose centre lies in it, each interval half-open, so boxes
  sharing a side count no pixel twice (the rule of the metric cells,
  :func:`darsia.metric_cells`);
- a sand layer, e.g. the free water: the pixels of its label in the run's sand map;
- the sand layers, or the sand layers in a box.

For each frame and domain (:class:`DomainMass`):

- a pixel of the domain is unknown if it has a mass the maps do not hold (not
  finite, e.g. free gas without a Swi); a pixel without pore volume (no depth or
  porosity) whose maps hold a number holds no CO2 and is left out of the domain;
- the total, dissolved and free-gas masses are the maps times the pore volume,
  summed over the known pixels: the quantified mass. A domain with an unknown
  pixel is not complete and has no total, only its quantified mass, so a total
  never counts unknown mass as none. CO2 the maps do not measure but hold as zero
  (unmeasured colour, excluded parts) is bounded per frame by the unknown-mass
  rule (:func:`darsia.unknown_bounds`), not here;
- the free-gas saturation and the concentration of the dissolved CO2 are means
  weighted by pore volume over the known pixels where they are finite;
- each mass has its centre and its spread (one standard deviation) along each axis.

The domains are set up once per run. The sums, means and centres run in double
precision, as our production summed its domains. The rows are those of the table
of masses per domain (:func:`write_domain_mass`), and a run's rows of it are read
back as its series of masses and centres (:class:`RunSeries`).

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import darsia
from darsia.multiphase.mass_maps import RunGeometry
from darsia.presets.workflows.analysis.domains import (
    COMPONENTS,
    AnalysisDomain,
    domain_pixels,
    layer_parts,
    write_rows,
)

_CENTRE = ("_centre_x_mm", "_centre_y_mm", "_spread_x_mm", "_spread_y_mm")
CENTRES = (
    "quantified_mass_g",
    "centre_x_mm",
    "centre_y_mm",
    "spread_x_mm",
    "spread_y_mm",
)
"""Columns of a centre of :attr:`RunSeries.centres`: the quantified mass [g], and its
centre and spread (one standard deviation) along each axis [mm]."""

COLUMNS = (
    "run",
    "case_id",
    "elapsed_h",
    "kind",
    "domain_id",
    "name",
    "complete",
    "pixels",
    "unknown_pixels",
    "area_m2",
    "pore_volume_l",
    *(key + end for key in COMPONENTS for end in ("_mass_g", "_quantified_mass_g")),
    "gas_saturation_mean",
    "aqueous_concentration_mean_g_l",
    *(key + end for key in COMPONENTS for end in _CENTRE),
)
"""Columns of the table of masses per domain: a row per frame and domain."""


class DomainMass:
    """The mass of a run's frames per analysis domain.

    The domains' pixels and the pixels' areas and pore volumes are set up once
    per run; each frame's maps are then summed over the domains' pixels that hold
    CO2 in that frame (:meth:`__call__`).

    """

    def __init__(
        self,
        domains: Sequence[AnalysisDomain],
        geometry: RunGeometry,
        sand_map: darsia.Image,
        sand_labels: Sequence[int] = (),
        chart_labels: Sequence[int] = (),
        min_part_area_cm2: float = 1.0,
    ) -> None:
        """Set up the domains of a run.

        Parameters
        ----------
        domains : sequence of AnalysisDomain
            The domains, each once, in the order of the rows.
        geometry : RunGeometry
            The run's geometry (:func:`darsia.run_geometry`).
        sand_map : darsia.Image
            The run's sand layers on its grid, whose labels the layer domains
            name and whose coordinates the boxes read.
        sand_labels : sequence of int, optional
            The labels of the sand layers, which a domain of the sand reads.
        chart_labels : sequence of int, optional
            The labels of the colour checker, under which the sand layers continue
            when a domain names a part of a layer (:func:`layer_parts`).
        min_part_area_cm2 : float, optional
            The smallest area [cm2] of a layer's part other than its largest, by
            the pixels' areas of the geometry.

        Raises
        ------
        ValueError
            If a domain names a part of a layer that is not a sand layer.

        """
        ids = [domain.domain_id for domain in domains]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Choose at least one domain, each once")
        self.domains = list(domains)
        """The domains, in the order of the rows."""
        self._shape = geometry.domain.shape
        self._x, self._y = darsia.pixel_centres(sand_map)
        volume = np.asarray(geometry.pore_volume_m3, np.float64).ravel() * 1000
        self._volume_l = np.where(volume > 0, volume, 0.0)
        self._no_volume = ~(volume > 0)
        self._area = np.nan_to_num(
            np.asarray(geometry.pixel_area_m2, np.float64)
        ).ravel()
        parts = None
        named = {d.label for d in self.domains if d.part is not None}
        if named - {int(label) for label in sand_labels}:
            raise ValueError(f"A part of a layer needs a sand layer: {sorted(named)}")
        if named:
            parts = layer_parts(
                sand_map.img,
                sand_labels,
                geometry.pixel_area_m2,
                min_part_area_cm2 * 1e-4,
                chart_labels,
            )
        self._pixels = [
            np.flatnonzero(
                geometry.domain
                & domain_pixels(domain, sand_map, sand_map.img, sand_labels, parts)
            )
            for domain in self.domains
        ]

    def __call__(
        self, maps: darsia.SimpleMassAnalysisResults, co2: darsia.CO2MassAnalysis
    ) -> list[dict]:
        """The rows of one frame, one per domain.

        Parameters
        ----------
        maps : darsia.SimpleMassAnalysisResults
            The frame's mass maps on the run's grid.
        co2 : darsia.CO2MassAnalysis
            Solubility of CO2 at the frame's pressure and temperature; the
            concentration of the dissolved CO2 in g/L is the maps' dissolved share
            times the solubility, in the maps' dtype.

        Returns
        -------
        list of dict
            Per domain the table's columns (:data:`COLUMNS`) but the frame's
            ``run``, ``case_id`` and ``elapsed_h``, which the caller adds; None
            where a value is not defined (the total of a domain that is not
            complete, a mean without pore volume, a centre without mass).

        """
        if maps.mass.img.shape != self._shape:
            raise ValueError("The mass maps are not on the run's grid")
        unknown = np.zeros(self._no_volume.shape, bool)
        for name in COMPONENTS.values():
            unknown |= ~np.isfinite(getattr(maps, name).img.ravel())
        # A pixel without pore volume whose maps hold a number holds no CO2: it is
        # not part of the domain, as our production left it out of its sums.
        present = ~self._no_volume | unknown
        grams = {}
        for key, name in COMPONENTS.items():
            values = np.asarray(getattr(maps, name).img, np.float64).ravel()
            grams[key] = np.where(unknown, 0.0, np.nan_to_num(values) * self._volume_l)
        means = {
            "gas_saturation_mean": maps.saturation_g.img.ravel(),
            "aqueous_concentration_mean_g_l": (
                maps.concentration_aq.img * co2.solubility_co2
            ).ravel(),
        }
        rows = []
        for domain, pixels in zip(self.domains, self._pixels):
            pixels = pixels[present[pixels]]
            area = float(self._area[pixels].sum())
            volume = float(self._volume_l[pixels].sum())
            missing = int(np.count_nonzero(unknown[pixels]))
            complete = pixels.size > 0 and missing == 0
            row = dict.fromkeys(COLUMNS[3:])
            row.update(
                kind=domain.kind,
                domain_id=domain.domain_id,
                name=domain.name,
                complete=complete,
                pixels=int(pixels.size),
                unknown_pixels=missing,
                area_m2=area,
                pore_volume_l=volume,
            )
            x = self._x[pixels % self._shape[1]]
            y = self._y[pixels // self._shape[1]]
            for key, values in grams.items():
                amount = float(values[pixels].sum())
                row[key + "_mass_g"] = amount if complete else None
                row[key + "_quantified_mass_g"] = amount
                row.update(_centre(key, values[pixels], x, y))
            weight = self._volume_l[pixels]
            for key, values in means.items():
                values = np.asarray(values[pixels], np.float64)
                known = np.isfinite(values) & ~unknown[pixels]
                total = float(weight[known].sum())
                if total > 0:
                    row[key] = float((values[known] * weight[known]).sum() / total)
            rows.append(row)
        return rows


def _centre(key, grams, x, y):
    """Centre [mm] of a mass along each axis and its spread; none without mass."""
    total = float(grams.sum())
    if not total > 0:
        return {}
    result = {}
    for axis, values in (("x", x), ("y", y)):
        centre = float((grams * values).sum() / total)
        spread = float((grams * (values - centre) ** 2).sum() / total)
        result[f"{key}_centre_{axis}_mm"] = centre * 1000
        result[f"{key}_spread_{axis}_mm"] = spread**0.5 * 1000
    return result


def write_domain_mass(path: Path, rows: Iterable[dict]) -> None:
    """Write the table of masses per domain, a row per frame and domain.

    Parameters
    ----------
    path : Path
        The CSV file.
    rows : iterable of dict
        Rows of :class:`DomainMass` with the frame's ``run``, ``case_id`` and
        ``elapsed_h``; a value that is None is an empty cell (:func:`write_rows`).

    """
    write_rows(path, rows, COLUMNS)


@dataclass(frozen=True)
class RunSeries:
    """A run's CO2 in each analysis domain over its frames, in time order."""

    run: str
    """The run."""
    case_ids: tuple[str, ...]
    """The frames."""
    elapsed_h: np.ndarray
    """Hours since the injection start of each frame."""
    injected_g: np.ndarray
    """CO2 injected up to each frame [g]; NaN where not known."""
    masses: Mapping[str, Mapping[str, np.ndarray]]
    """Per domain identity and component, the mass of each frame [g]; NaN where
    not known."""
    centres: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    """Per domain identity and component, a row per frame in the columns
    :data:`CENTRES`; NaN where the table has no value."""

    @classmethod
    def from_domain_mass(
        cls,
        rows: Iterable[Mapping],
        injected_g: Mapping[str, float],
        masses: Literal["complete", "quantified"],
    ) -> RunSeries:
        """A run's series from its rows of the table of masses per domain.

        Parameters
        ----------
        rows : iterable of mappings
            The run's rows in any order, as :class:`DomainMass` gives them with
            the frame's ``run``, ``case_id`` and ``elapsed_h``, or as read from the
            table's file.
        injected_g : mapping
            CO2 injected up to each frame [g], by case id.
        masses : "complete" or "quantified"
            A domain's total, which a domain holding an unknown pixel lacks, or its
            quantified mass, the known part.

        Returns
        -------
        RunSeries
            Its frames in the order of their time, with the masses and centres.

        """
        if masses not in ("complete", "quantified"):
            raise ValueError(f"Unknown masses: {masses!r}")
        rows = list(rows)
        runs = {str(row["run"]) for row in rows}
        if len(runs) != 1:
            raise ValueError("The rows are not those of one run")
        times: dict[str, float] = {}
        for row in rows:
            times.setdefault(str(row["case_id"]), float(row["elapsed_h"]))
        case_ids = tuple(sorted(times, key=times.__getitem__))
        index = {case: position for position, case in enumerate(case_ids)}
        suffix = "_mass_g" if masses == "complete" else "_quantified_mass_g"
        centre = ("_quantified_mass_g", *_CENTRE)
        found: dict[str, dict[str, np.ndarray]] = {}
        centres: dict[str, dict[str, np.ndarray]] = {}
        for row in rows:
            domain, frame = str(row["domain_id"]), index[str(row["case_id"])]
            if domain not in found:
                found[domain] = {
                    key: np.full(len(case_ids), np.nan) for key in COMPONENTS
                }
                centres[domain] = {
                    key: np.full((len(case_ids), len(centre)), np.nan)
                    for key in COMPONENTS
                }
            for key in COMPONENTS:
                found[domain][key][frame] = _number(row[key + suffix])
                values = [_number(row.get(key + end)) for end in centre]
                centres[domain][key][frame] = values
        return cls(
            run=runs.pop(),
            case_ids=case_ids,
            elapsed_h=np.array([times[case] for case in case_ids]),
            injected_g=np.array([_number(injected_g.get(case)) for case in case_ids]),
            masses=found,
            centres=centres,
        )


def _number(value) -> float:
    """A table cell as a number: NaN for an empty cell or None."""
    return np.nan if value is None or value == "" else float(value)
