"""Mass maps: the CO2 mass per pore volume of each pixel of a frame.

The mass of a frame follows from its read signal, its phase map and its free gas
on the run's grid, with the calibration's parameters:

- the read signal of the dissolved CO2 (e.g., the reading of
  :func:`darsia.run_statistics`) relative to the range of the concentration curve,
  from no dissolved CO2 to saturation, clipped to [0, 1] (:func:`relative_signal`);
- times the calibrated gain, clipped to one, times the dilution factor
  (:func:`darsia.dilution_factor`): the share of the solubility that is dissolved;
- CO2 only where the phase map says CO2, a hard limit: dissolved pixels whose colour
  is measured take that share, and the water of the model's gas
  (:func:`darsia.model_gas`, from the phase and rule maps) is saturated;
- the free-gas saturation of each pixel (e.g., :func:`darsia.free_gas`).

The mass per pore volume is then the official one
(:meth:`darsia.CO2MassAnalysis.mass_analysis`): the gas density times the
saturation, plus the solubility times the share times one minus the saturation.
:class:`PrecisionCO2MassAnalysis` evaluates the official tables in the precision
of the maps. The maps are a :class:`darsia.SimpleMassAnalysisResults`, the form
the official analysis writes, so every later step reads either method's maps.

The run's geometry (:class:`RunGeometry`: pore volume from pixel area, depth and
porosity, the pixels that count, Swi) is computed once per run and serves every
frame: a frame's mass is the integral of its maps (:meth:`RunGeometry.mass_g`),
and :class:`MassCopy` copies them onto metric cells, e.g. 1 mm, by the one
scaling method.

Every setting is a field of :class:`MassCopySettings`; the calibration's
parameters are a :class:`darsia.MassCalibration`.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import darsia
from darsia.multi_image_analysis.calibration_fit import MassCalibration
from darsia.multiphase.phase_rules import PhaseLabel, model_gas
from darsia.pvt import data as pvt_data

_GRAVITY = 9.81
"""Gravitational acceleration [m/s2], as in the official mass analysis."""
_PA_TO_BAR = 1e-5
_MAPS = {"mass_total": "mass", "mass_aqueous": "mass_aq", "mass_free_gas": "mass_g"}
"""The field of the mass maps behind each mass."""


@dataclass(frozen=True)
class MassCopySettings:
    """Settings of the mass maps' copy on metric cells; one configuration table."""

    copy_mm: float
    """Size [mm] of the metric cells of the maps' copy."""


class PrecisionCO2MassAnalysis(darsia.CO2MassAnalysis):
    """The official CO2 mass analysis, its tables evaluated in the maps' precision.

    The gas density and the solubility of CO2 follow the official NIST tables
    (:mod:`darsia.pvt.data`) at each pixel's hydrostatic pressure and temperature,
    as in :class:`darsia.CO2MassAnalysis`, but every pixel operation and the linear
    interpolation of the tables run in the floating dtype of the baseline image,
    as our production computed them. The hydrostatic head of a pixel is the
    datum, the rig's declared top, where the atmospheric pressure is given, less
    the height of the pixel's centre (:func:`darsia.pixel_centres`); the head and
    the top pressure are computed in double precision and taken to the
    baseline's dtype. A pressure outside the tables is refused, as the official
    interpolation refuses it.

    """

    def __init__(
        self,
        baseline: darsia.Image,
        *,
        datum_m: float,
        atmospheric_pressure: float = 1.010,
        atmospheric_temperature: float = 23.0,
        atmospheric_pressure_gradient: float = 0.0,
        atmospheric_temperature_gradient: float = 0.0,
    ) -> None:
        """The CO2 fields of a frame on the run's grid.

        Parameters
        ----------
        baseline : darsia.Image
            An image on the run's grid, in the floating dtype of the maps.
        datum_m : float
            Height y [m] of zero hydrostatic head: the rig's declared top, the
            top of its canvas before any extension to whole metric cells
            (``MetricScalingCorrection.native_origin[1]``, the rig's
            ``resolution_correction``), in the coordinates of the baseline.
        atmospheric_pressure : float
            Atmospheric pressure [bar] at the bottom of the canvas, carried up to
            the datum by its gradient.
        atmospheric_temperature : float
            Temperature [Celsius].
        atmospheric_pressure_gradient : float
            Atmospheric pressure gradient [bar/m].
        atmospheric_temperature_gradient : float
            Temperature gradient [Celsius/m].

        """
        self.datum_m = float(datum_m)
        """Height [m] of zero hydrostatic head."""
        super().__init__(
            baseline,
            atmospheric_pressure,
            atmospheric_temperature,
            atmospheric_pressure_gradient,
            atmospheric_temperature_gradient,
        )

    @property
    def height_map(self) -> np.ndarray:
        """Hydrostatic head [m] of each pixel: the datum less the height of the
        pixel's centre, in double precision."""
        _, y = darsia.pixel_centres(self.baseline)
        head = self.datum_m - np.asarray(y, np.float64)
        return np.broadcast_to(head[:, None], self.baseline.num_voxels)

    def top_atmospheric_pressure(self) -> float:
        """Atmospheric pressure [bar] at the datum."""
        return (
            self.atmospheric_pressure
            + self.atmospheric_pressure_gradient * self.datum_m
        )

    def setup_state(self) -> None:
        """Gas density and solubility of CO2 per pixel, in the baseline's dtype.

        The official state is set first (its reference density and the key of its
        state); its gas density and solubility are then replaced by these.

        """
        super().setup_state()
        dtype = np.result_type(self.baseline.img, np.float32).type
        height = self.height_map.astype(dtype)
        temperature = np.ones_like(height) * dtype(self.atmospheric_temperature) + (
            dtype(self.atmospheric_temperature_gradient) * height
        )
        top = dtype(self.top_atmospheric_pressure())

        def between(value_20, value_23):
            """Linear in the temperature between the 20 and 23 degree values."""
            return value_20 * (dtype(23) - temperature) / dtype(3) + value_23 * (
                temperature - dtype(20)
            ) / dtype(3)

        water_20, water_23 = pvt_data.WATER_DENSITY
        water = between(dtype(water_20), dtype(water_23))
        pressure = top + water * dtype(_GRAVITY) * height * dtype(_PA_TO_BAR)
        gas, dissolved = (
            pvt_data.CO2_DENSITY_PRESSURES,
            pvt_data.CO2_SOLUBILITY_PRESSURES,
        )
        self.density_gaseous_co2 = between(
            _table(pressure, gas, pvt_data.CO2_DENSITY_20),
            _table(pressure, gas, pvt_data.CO2_DENSITY_23),
        )
        self.solubility_co2 = between(
            _table(pressure, dissolved, pvt_data.CO2_SOLUBILITY_20),
            _table(pressure, dissolved, pvt_data.CO2_SOLUBILITY_23),
        )


def _table(values, supports, table):
    """Linear interpolation of a table in the dtype of the values; a value outside
    the table is refused."""
    x = np.asarray(supports, values.dtype)
    if np.any(values < x[0]) or np.any(values > x[-1]):
        raise ValueError("Pressure outside the CO2 tables")
    return darsia.piecewise_linear(values, supports, table)


@dataclass(frozen=True)
class RunGeometry:
    """Geometry of a run's grid for its mass maps, computed once per run."""

    domain: np.ndarray
    """Pixels of the rig."""
    known: np.ndarray
    """Pixels of the domain with a finite positive depth and porosity."""
    counted: np.ndarray
    """Known pixels whose mass counts (not excluded)."""
    pore_volume_m3: np.ndarray
    """Pore volume of each pixel: its area times depth times porosity; zero where
    not known."""
    pixel_area_m2: np.ndarray
    """Measured area [m2] of each pixel."""
    swi: np.ndarray
    """Irreducible water saturation of each pixel; NaN where none is given."""
    y_m: np.ndarray
    """Height y [m] of the pixel centres of each row (:func:`darsia.pixel_centres`)."""
    dimensions: list[float]
    """Height and width of the grid [m]."""

    def mass_g(self, values: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Mass [g] of a mass map (g per litre of pore volume) over a mask.

        The map is weighted by the pore volume and summed along the columns, then
        across them, in the dtype of the map and the pore volume, as the official
        :meth:`darsia.Geometry.integrate`. Pixels of unknown mass (not finite) add
        nothing; the unknown-mass rule bounds their mass.

        Parameters
        ----------
        values : array
            Mass map, e.g. the total mass of :func:`mass_maps`.
        mask : array, optional
            Pixels to integrate over; by default the counted pixels.

        Returns
        -------
        float
            The mass in grams.

        """
        mask = self.counted if mask is None else mask
        values = np.where(mask & np.isfinite(values), values, 0)
        return float(
            np.sum(np.sum(self.pore_volume_m3 * values, axis=0), axis=0) * 1000
        )

    def pore_volume(self, mask: np.ndarray) -> float:
        """Pore volume [m3] of a mask, e.g. a frame's CO2 in the domain.

        Parameters
        ----------
        mask : array
            Pixels to sum over.

        Returns
        -------
        float
            The pore volume, summed in the dtype of the geometry.

        """
        return float(np.sum(self.pore_volume_m3[mask]))


def run_geometry(
    sand_map: darsia.Image,
    pixel_area_m2: np.ndarray,
    depth_m: np.ndarray,
    porosity: np.ndarray,
    swi: np.ndarray,
    domain: np.ndarray,
    excluded: np.ndarray | None = None,
) -> RunGeometry:
    """The geometry of a run's grid, once for all of its frames.

    Parameters
    ----------
    sand_map : darsia.Image
        An image on the run's grid, e.g. its sand map, with its physical dimensions.
    pixel_area_m2 : array
        Measured area [m2] of each pixel of the grid, the rig's
        (``Rig.pixel_area.img``); the pore volume is computed in the dtype of the
        areas, the depth and the porosity.
    depth_m : array
        Depth [m] of each pixel.
    porosity : array
        Porosity of each pixel.
    swi : array
        Irreducible water saturation of each pixel.
    domain : array
        Pixels of the rig.
    excluded : array, optional
        Pixels of the domain whose mass does not count (e.g., behind the colour
        checker).

    Returns
    -------
    RunGeometry
        The geometry.

    """
    shape = sand_map.img.shape[:2]
    known = (
        domain
        & np.isfinite(depth_m)
        & np.isfinite(porosity)
        & (depth_m > 0)
        & (porosity > 0)
    )
    counted = known if excluded is None else known & ~excluded
    _, y = darsia.pixel_centres(sand_map)
    return RunGeometry(
        domain=domain,
        known=known,
        counted=counted,
        pore_volume_m3=np.where(known, pixel_area_m2 * depth_m * porosity, 0),
        pixel_area_m2=np.broadcast_to(pixel_area_m2, shape),
        swi=np.broadcast_to(swi, shape),
        y_m=y,
        dimensions=[float(length) for length in sand_map.dimensions],
    )


def relative_signal(signal: np.ndarray, calibration: MassCalibration) -> np.ndarray:
    """The read signal relative to the concentration curve's range, in [0, 1].

    Parameters
    ----------
    signal : array
        Read signal of the dissolved CO2.
    calibration : MassCalibration
        The curve's bottom and top.

    Returns
    -------
    array
        ``(signal - zero) / (saturation - zero)`` clipped to [0, 1], in the
        signal's dtype.

    """
    dtype = signal.dtype.type
    zero = dtype(calibration.zero_signal)
    span = dtype(calibration.saturation_signal - calibration.zero_signal)
    return np.clip((signal - zero) / span, dtype(0), dtype(1))


def main_pixels(
    labels: np.ndarray,
    measured: np.ndarray,
    geometry: RunGeometry,
    gas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The pixels whose CO2 is measured: dissolved, and free gas.

    Parameters
    ----------
    labels : array
        Phase map (:class:`darsia.PhaseLabel`).
    measured : array
        Pixels whose colour is measured.
    geometry : RunGeometry
        The run's geometry.
    gas : array
        The model's free gas (:func:`darsia.model_gas`), also where the phase
        rules' gravity limit or the gas allocation took it away (its water is
        saturated).

    Returns
    -------
    tuple of arrays
        Known dissolved pixels with a measured colour, and known free-gas pixels
        with a Swi.

    """
    dissolved = geometry.known & measured & (labels == PhaseLabel.DISSOLVED)
    return dissolved, geometry.known & np.isfinite(geometry.swi) & gas


def mass_maps(
    relative: darsia.Image,
    labels: np.ndarray,
    rules: np.ndarray,
    saturation: np.ndarray,
    measured: np.ndarray,
    geometry: RunGeometry,
    co2: darsia.CO2MassAnalysis,
    calibration: MassCalibration,
    factor: float,
) -> darsia.SimpleMassAnalysisResults:
    """A frame's CO2 mass per pore volume, from its relative signal and phase map.

    The dissolved share is the relative signal times the gain, clipped to one,
    times the dilution factor, on the dissolved pixels whose colour is measured
    (:func:`main_pixels`); it is one in the water of the model's gas and zero
    elsewhere, so CO2 is only where the phase map says CO2. The saturation is the
    given one on those pixels where they count, 1 - Swi on other free gas, and
    zero elsewhere. The masses follow from ``co2.mass_analysis`` and are kept on
    the counted pixels, zero elsewhere, and unknown (NaN) on free gas without a
    Swi. The concentration and saturation are unknown (NaN) on CO2 pixels whose
    CO2 is not measured, the saturation on their free gas. All arithmetic runs in
    the dtype of the relative signal and of the CO2 fields.

    Parameters
    ----------
    relative : darsia.Image
        The read signal relative to the concentration curve
        (:func:`relative_signal`), with the frame's name, date and time.
    labels : array
        The frame's phase map (:class:`darsia.PhaseLabel`).
    rules : array
        Its rule map (:class:`darsia.PhaseRule`), e.g. of :func:`darsia.free_gas`;
        with the phase map it gives the model's gas (:func:`darsia.model_gas`).
    saturation : array
        Free-gas saturation of each pixel, e.g. of :func:`darsia.free_gas`.
    measured : array
        Pixels whose colour is measured.
    geometry : RunGeometry
        The run's geometry.
    co2 : darsia.CO2MassAnalysis
        Solubility and gas density of CO2 at the frame's pressure and temperature,
        e.g. a :class:`PrecisionCO2MassAnalysis`.
    calibration : MassCalibration
        The gain.
    factor : float
        Dilution factor of the frame (:func:`darsia.dilution_factor`).

    Returns
    -------
    darsia.SimpleMassAnalysisResults
        Mass, gas and dissolved mass per pore volume (g/L), the free-gas saturation
        and the dissolved share of the solubility.

    """
    dtype = relative.img.dtype.type
    model = model_gas(labels, rules)
    dissolved, gas = main_pixels(labels, measured, geometry, model)
    share = np.clip(relative.img * dtype(calibration.gain), dtype(0), dtype(1))
    share = np.where(dissolved, share * dtype(factor), dtype(0))
    share[gas] = dtype(1)
    held = np.zeros_like(share)
    held[gas] = dtype(1) - geometry.swi[gas]
    kept = geometry.counted & (dissolved | gas)
    held[kept] = saturation[kept]
    result = co2.mass_analysis(
        c_aq=darsia.full_like(relative, share), s_g=darsia.full_like(relative, held)
    )
    unquantified = geometry.counted & model & ~np.isfinite(geometry.swi)
    for image in (result.mass, result.mass_g, result.mass_aq):
        image.img = np.where(geometry.counted, image.img, 0)
        image.img[unquantified] = np.nan
    co2_pixels = geometry.domain & np.isin(
        labels, (PhaseLabel.DISSOLVED, PhaseLabel.GAS)
    )
    unknown = co2_pixels & ~(dissolved | gas)
    result.concentration_aq.img[unknown] = np.nan
    result.saturation_g.img[unknown & model] = np.nan
    return result


def known_mass(
    maps: darsia.SimpleMassAnalysisResults, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Where each mass of a frame's maps is known.

    A mass is known where it is finite, the pixel's phase is known and what it is
    made of is finite: the concentration and the saturation for the total mass,
    the concentration for the dissolved mass and the saturation for the free gas.
    Unknown mass is never read as no mass.

    Parameters
    ----------
    maps : darsia.SimpleMassAnalysisResults
        A frame's mass maps, of either method.
    labels : array
        The frame's phase map (:class:`darsia.PhaseLabel`).

    Returns
    -------
    dict of str to array
        ``mass_total``, ``mass_aqueous`` and ``mass_free_gas``: where each is
        known (bool).

    """
    concentration = np.isfinite(maps.concentration_aq.img)
    saturation = np.isfinite(maps.saturation_g.img)
    phase = labels != PhaseLabel.UNKNOWN
    return {
        name: np.isfinite(image.img) & phase & markers
        for name, image, markers in (
            ("mass_total", maps.mass, concentration & saturation),
            ("mass_aqueous", maps.mass_aq, concentration),
            ("mass_free_gas", maps.mass_g, saturation),
        )
    }


class MassCopy:
    """Copies of a run's mass maps on metric cells, e.g. 1 mm.

    The cells and the run's pore volume and area per cell are set up once per run
    (:attr:`static`); each frame's maps are then copied by the one scaling method
    (:func:`darsia.cell_sums`, double precision): the masses are sums over each
    cell, so every total is kept, the concentration and the saturation are
    pore-volume weighted means, and the phase fractions are shares of the cell's
    domain area.

    """

    def __init__(self, geometry: RunGeometry, settings: MassCopySettings) -> None:
        """Set up the copy of a run's maps.

        Parameters
        ----------
        geometry : RunGeometry
            The run's geometry.
        settings : MassCopySettings
            The size of the cells.

        """
        shape = geometry.domain.shape
        self.cells = darsia.metric_cells(shape, geometry.dimensions, settings.copy_mm)
        """The cells (:func:`darsia.metric_cells`)."""
        self.volume_l = np.asarray(geometry.pore_volume_m3, np.float64) * 1000
        """Pore volume [L] of each pixel."""
        self.area_m2 = np.where(geometry.domain, geometry.pixel_area_m2, 0.0)
        """Domain area [m2] of each pixel."""
        domain_area = darsia.cell_sums(self.cells, self.area_m2)
        self.static = {
            "pore_volume_l": darsia.cell_sums(self.cells, self.volume_l),
            "domain_area_m2": domain_area,
            "domain_area_fraction": domain_area / (settings.copy_mm / 1000) ** 2,
        }
        """Pore volume [L], domain area [m2] and its share of the cell's area, per
        cell."""

    def __call__(
        self, maps: darsia.SimpleMassAnalysisResults, labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        """A frame's maps on the cells.

        A pixel of unknown mass adds nothing to its cell, and each mass comes with
        the share of the cell's pore volume whose mass is known
        (:func:`known_mass`), so that unknown is never read as no mass.

        Parameters
        ----------
        maps : darsia.SimpleMassAnalysisResults
            The frame's mass maps, of either method.
        labels : array
            The frame's phase map.

        Returns
        -------
        dict of str to array
            Per cell: ``mass_total_g``, ``mass_aqueous_g``, ``mass_free_gas_g`` and
            their ``*_known_fraction``; ``concentration_aq`` and ``saturation_g``;
            ``phase_water``, ``phase_dissolved`` and ``phase_gas_area_fraction``.

        """
        cells, volume = self.cells, self.volume_l
        pore_volume = self.static["pore_volume_l"]
        concentration = np.asarray(maps.concentration_aq.img, np.float64)
        saturation = np.asarray(maps.saturation_g.img, np.float64)
        copy = {}
        for name, known in known_mass(maps, labels).items():
            values = np.asarray(getattr(maps, _MAPS[name]).img, np.float64)
            finite = np.isfinite(values)
            copy[name + "_g"] = darsia.cell_sums(
                cells, np.where(finite, values, 0.0) * volume
            )
            copy[name + "_known_fraction"] = cell_ratio(
                darsia.cell_sums(cells, np.where(known, volume, 0.0)), pore_volume
            )
        for name, values in (
            ("concentration_aq", concentration),
            ("saturation_g", saturation),
        ):
            valid = np.isfinite(values)
            copy[name] = cell_ratio(
                darsia.cell_sums(cells, np.where(valid, values * volume, 0.0)),
                darsia.cell_sums(cells, np.where(valid, volume, 0.0)),
            )
        domain_area = self.static["domain_area_m2"]
        for name, label in (
            ("water", PhaseLabel.WATER),
            ("dissolved", PhaseLabel.DISSOLVED),
            ("gas", PhaseLabel.GAS),
        ):
            area = np.where(labels == label, self.area_m2, 0.0)
            copy["phase_" + name + "_area_fraction"] = cell_ratio(
                darsia.cell_sums(cells, area), domain_area
            )
        return copy


def cell_ratio(numerator, denominator):
    """Ratio per cell in double precision; NaN where the denominator is not
    positive."""
    ratio = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    return ratio
