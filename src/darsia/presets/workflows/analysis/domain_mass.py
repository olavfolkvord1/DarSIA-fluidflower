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

- a pixel of the domain is unknown if it has no pore volume (no depth or porosity)
  or a mass the maps do not hold (not finite, e.g. free gas without a Swi);
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
precision, as our production summed its domains.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import darsia
from darsia.multiphase.mass_maps import RunGeometry
from darsia.presets.workflows.analysis.domains import (
    COMPONENTS,
    AnalysisDomain,
    domain_pixels,
)

_CENTRE = ("_centre_x_mm", "_centre_y_mm", "_spread_x_mm", "_spread_y_mm")

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

    The domains' pixels, areas and pore volumes are set up once per run; each
    frame's maps are then summed over them (:meth:`__call__`).

    """

    def __init__(
        self,
        domains: Sequence[AnalysisDomain],
        geometry: RunGeometry,
        sand_map: darsia.Image,
        sand_labels: Sequence[int] = (),
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
        area = np.nan_to_num(np.asarray(geometry.pixel_area_m2, np.float64)).ravel()
        self._pixels = [
            np.flatnonzero(
                geometry.domain
                & domain_pixels(domain, sand_map, sand_map.img, sand_labels)
            )
            for domain in self.domains
        ]
        self._sums = [
            (float(area[pixels].sum()), float(self._volume_l[pixels].sum()))
            for pixels in self._pixels
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
        unknown = self._no_volume.copy()
        for name in COMPONENTS.values():
            unknown |= ~np.isfinite(getattr(maps, name).img.ravel())
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
        for domain, pixels, (area, volume) in zip(
            self.domains, self._pixels, self._sums
        ):
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
