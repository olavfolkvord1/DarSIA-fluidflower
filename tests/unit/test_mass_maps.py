"""Unit tests for the mass maps on a small synthetic frame."""

import tomllib

import numpy as np
import pytest

import darsia

W, D, G, U = darsia.PhaseLabel

TABLES = tomllib.loads(
    """
    # A calibration record as the calibration delivers it (rounded values).
    [calibration]
    gain = 1.5
    zero_signal = 0.0
    saturation_signal = 1.2
    line_share = 1.0
    beta = 0.53
    beta_growth = 0.11
    anchor_h = 2.25
    anchor_pore_volume_m3 = 4e-4
    """
)
CALIBRATION = darsia.MassCalibration(**TABLES["calibration"])

LABELS = np.array(
    [
        [W, W, W, W, W, W],
        [U, G, G, G, W, G],
        [U, D, D, D, W, W],
        [U, D, D, D, W, W],
        [U, D, D, D, W, W],
        [U, W, W, W, W, W],
    ],
    np.uint8,
)
FREED = (2, 1)
"""Model gas the allocation peeled: dissolved in the map, its water saturated."""
UNMEASURED, EXCLUDED, NO_SWI, ADDED = (3, 3), (4, 2), (1, 5), (3, 1)


def frame(dtype=np.float32):
    """A frame on a 6 x 6 grid of 5 mm: geometry, CO2 fields, maps and labels."""
    sand = darsia.Image(np.ones((6, 6), np.uint8), dimensions=[0.03, 0.03])
    domain = np.ones((6, 6), bool)
    domain[:, 0] = False
    excluded = np.zeros((6, 6), bool)
    excluded[EXCLUDED] = True
    swi = np.full((6, 6), 0.1, dtype)
    swi[:, 5] = np.nan
    geometry = darsia.run_geometry(
        sand,
        np.full((6, 6), 25e-6, dtype),
        np.full((6, 6), 0.01, dtype),
        np.full((6, 6), 0.4, dtype),
        swi,
        domain,
        excluded,
    )
    co2 = darsia.PrecisionCO2MassAnalysis(
        darsia.ScalarImage(np.zeros((6, 6), dtype), dimensions=[0.03, 0.03]),
        datum_m=0.03,
        atmospheric_pressure=1.0,
        atmospheric_temperature=22.0,
    )
    saturation = np.where(LABELS == G, dtype(1) - swi, 0).astype(dtype)
    saturation[ADDED] = 0.3
    measured = np.ones((6, 6), bool)
    measured[UNMEASURED] = False
    rules = np.zeros((6, 6), np.uint8)
    rules[FREED] = darsia.PhaseRule.PEELED
    signal = darsia.ScalarImage(np.full((6, 6), 0.6, dtype), dimensions=[0.03, 0.03])
    relative = darsia.full_like(signal, darsia.relative_signal(signal.img, CALIBRATION))
    maps = darsia.mass_maps(
        relative,
        LABELS,
        rules,
        saturation,
        measured,
        geometry,
        co2,
        CALIBRATION,
        0.8,
    )
    return geometry, co2, saturation, maps


class _StateAtCentres(darsia.HydrostaticState):
    """The official hydrostatic state with its head at the pixel centres."""

    @property
    def height_column(self):
        return (np.arange(self.baseline.num_voxels[0]) + 0.5)[
            :, None
        ] * self.baseline.voxel_size[0]

    @property
    def height_map(self):
        return self.height_column * np.ones(self.baseline.num_voxels)


class _OfficialAtCentres(darsia.CO2MassAnalysis):
    """The official analysis with its head at the pixel centres."""

    @property
    def hydrostatic_state(self):
        return _StateAtCentres(
            self.baseline,
            self.atmospheric_pressure,
            self.atmospheric_temperature,
            self.atmospheric_pressure_gradient,
            self.atmospheric_temperature_gradient,
        )


def test_precision_tables_compute_in_the_baseline_dtype():
    """The official tables in single precision, with the head at the pixel centres
    below the declared height; out-of-table pressures are refused."""
    for dtype, tolerance in ((np.float32, 1e-6), (np.float64, 1e-12)):
        baseline = darsia.ScalarImage(np.zeros((50, 4), dtype), dimensions=[0.5, 0.1])
        ours = darsia.PrecisionCO2MassAnalysis(
            baseline, datum_m=0.5, atmospheric_pressure=1.01
        )
        official = _OfficialAtCentres(
            baseline, atmospheric_pressure=1.01, atmospheric_temperature=23.0
        )
        np.testing.assert_allclose(ours.height_map[[0, -1], 0], [0.005, 0.495])
        for field in ("density_gaseous_co2", "solubility_co2"):
            assert getattr(ours, field).dtype == dtype
            np.testing.assert_allclose(
                getattr(ours, field), getattr(official, field), rtol=tolerance
            )
    with pytest.raises(ValueError):
        darsia.PrecisionCO2MassAnalysis(baseline, datum_m=0.5, atmospheric_pressure=1.6)


def test_the_datum_is_the_declared_top_on_a_grid_extended_to_whole_cells():
    """A run grid extended to whole cells keeps the head of the rig it shows."""
    canvas = darsia.ScalarImage(np.zeros((50, 4)), dimensions=[0.5, 0.04])
    for cell_mm in (25.0, 30.0):  # 20 whole cells; 17 cells, extended to 0.51 m
        scaling = darsia.MetricScalingCorrection(
            cell_mm, (50, 4), [0.5, 0.04], canvas.origin
        )
        grid = scaling(canvas)
        co2 = darsia.PrecisionCO2MassAnalysis(
            grid, datum_m=scaling.native_origin[1], atmospheric_pressure=1.01
        )
        _, y = darsia.pixel_centres(grid)
        np.testing.assert_allclose(co2.height_map[:, 0], 0.5 - y, rtol=0, atol=1e-15)


def test_mass_lies_only_in_the_co2_the_phase_map_measures():
    """Dissolved share from the signal, saturated water under free gas, no mass
    outside CO2, and unknown CO2 left unknown."""
    geometry, co2, saturation, maps = frame()
    rho, sol = co2.density_gaseous_co2, co2.solubility_co2
    share = np.float32(np.clip(np.float32(0.5) * np.float32(1.5), 0, 1) * 0.8)

    assert maps.mass.img.dtype == np.float32
    water_or_outside = (LABELS == W) | ~geometry.domain
    assert np.all(maps.mass.img[water_or_outside] == 0)
    assert maps.concentration_aq.img[3, 2] == share
    assert maps.mass_aq.img[3, 2] == sol[3, 2] * share
    added = maps.mass_aq.img[ADDED]
    assert added == sol[ADDED] * share * (np.float32(1) - np.float32(0.3))
    assert maps.mass_g.img[1, 2] == rho[1, 2] * saturation[1, 2]
    assert (
        maps.concentration_aq.img[1, 2] == 1 and maps.concentration_aq.img[FREED] == 1
    )
    assert maps.mass.img[FREED] == sol[FREED]
    assert maps.mass.img[EXCLUDED] == 0 and maps.concentration_aq.img[EXCLUDED] == share
    assert np.isnan(maps.concentration_aq.img[UNMEASURED])
    assert maps.mass.img[UNMEASURED] == 0
    assert np.isnan(maps.mass.img[NO_SWI]) and np.isnan(maps.saturation_g.img[NO_SWI])
