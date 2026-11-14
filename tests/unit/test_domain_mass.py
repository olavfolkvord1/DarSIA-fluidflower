"""Unit tests for the mass per analysis domain on a coarse synthetic rig."""

import tomllib

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.analysis.domain_mass import DomainMass
from darsia.presets.workflows.analysis.domains import AnalysisDomain, injected_mass

DOMAINS = [
    AnalysisDomain(**row)
    for row in tomllib.loads(
        """
        # The domains of our production's analysis: the whole domain, the two
        # boxes of its saved box version, its free water and a sand layer; two
        # boxes that split the rig on a column of pixel centres, and the sand.
        [[domain]]
        kind = "whole"
        domain_id = "whole"
        name = "Whole domain"

        [[domain]]
        kind = "box"
        domain_id = "box1"
        name = "box1"
        bounds_mm = [0.0, 900.0, 0.0, 150.4]

        [[domain]]
        kind = "box"
        domain_id = "box2"
        name = "box2"
        bounds_mm = [0.0, 900.0, 150.4, 490.0]

        [[domain]]
        kind = "layer"
        domain_id = "layer:2"
        name = "Free water layer 2"
        label = 2

        [[domain]]
        kind = "layer"
        domain_id = "layer:3"
        name = "E layer 3"
        label = 3

        [[domain]]
        kind = "layer"
        domain_id = "layer:5"
        name = "C layer 5"
        label = 5

        [[domain]]
        kind = "box"
        domain_id = "left"
        name = "left"
        bounds_mm = [0.0, 455.0, 0.0, 490.0]

        [[domain]]
        kind = "box"
        domain_id = "right"
        name = "right"
        bounds_mm = [455.0, 900.0, 0.0, 490.0]

        [[domain]]
        kind = "box"
        domain_id = "outside"
        name = "outside"
        bounds_mm = [-50.0, -10.0, 0.0, 490.0]

        [[domain]]
        kind = "sand"
        domain_id = "sand"
        name = "Sand"
        """
    )["domain"]
]
NO_VOLUME, NO_SWI = (20, 10), (13, 45)
SAND = (3, 5)
"""The labels of the sand layers; 2 is the free water."""


def frame(dtype=np.float32):
    """A frame on the rig at 10 mm: the run's geometry, sand map, maps and CO2."""
    shape = (49, 90)
    labels = np.full(shape, 5, np.uint8)
    labels[:9], labels[9:30] = 2, 3
    sand = darsia.Image(labels, dimensions=[0.49, 0.9])
    domain = np.ones(shape, bool)
    domain[:, 0] = False
    rng = np.random.default_rng(1)
    porosity = rng.uniform(0.3, 0.45, shape).astype(dtype)
    porosity[NO_VOLUME] = 0
    full = np.full(shape, 0.01, dtype)
    area = np.full(shape, 1e-4, dtype)
    geometry = darsia.run_geometry(sand, area, full, porosity, full * 10, domain)
    aqueous = np.where(labels == 3, rng.uniform(0, 1.7, shape), 0).astype(dtype)
    gas = np.zeros(shape, dtype)
    gas[12:15, 40:50] = 50
    saturation = np.where(gas > 0, 0.4, 0).astype(dtype)
    share = np.where(aqueous > 0, rng.uniform(0, 1, shape), 0).astype(dtype)
    share[25, 30] = np.nan
    mass = aqueous + gas
    for values in (mass, aqueous, gas, saturation):
        values[NO_SWI] = np.nan

    def image(values):
        return darsia.ScalarImage(values, dimensions=[0.49, 0.9])

    maps = darsia.SimpleMassAnalysisResults(
        name="frame",
        date=None,
        time=None,
        mass=image(mass),
        mass_g=image(gas),
        mass_aq=image(aqueous),
        saturation_g=image(saturation),
        color_signal=None,
        concentration_aq=image(share),
    )
    co2 = darsia.PrecisionCO2MassAnalysis(
        image(np.zeros(shape, dtype)), datum_m=0.49, atmospheric_pressure=1.0
    )
    return geometry, sand, maps, co2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_whole_domain_holds_the_map_total(dtype):
    """The whole domain's mass is the map's total, and boxes sharing a side and the
    sand layers each split it with no pixel counted twice; the sand is its layers
    without the free water."""
    geometry, sand, maps, co2 = frame(dtype)
    rows = DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    row = {r["domain_id"]: r for r in rows}

    for key, image in (("total", maps.mass), ("aqueous", maps.mass_aq)):
        assert row["whole"][key + "_quantified_mass_g"] == pytest.approx(
            geometry.mass_g(image.img, geometry.domain), rel=1e-6
        )
    for parts in (
        ("box1", "box2"),
        ("left", "right"),
        ("layer:2", "layer:3", "layer:5"),
    ):
        for key in ("pixels", "pore_volume_l", "total_quantified_mass_g"):
            assert sum(row[part][key] for part in parts) == pytest.approx(
                row["whole"][key], rel=1e-12
            )
    for key in ("pixels", "pore_volume_l", "total_quantified_mass_g"):
        assert row["sand"][key] == pytest.approx(
            row["layer:3"][key] + row["layer:5"][key], rel=1e-12
        )
    assert row["whole"]["pixels"] == geometry.domain.sum()
    assert row["box1"]["pixels"] == 15 * 89


def test_an_unknown_pixel_leaves_its_domains_without_a_total():
    """A pixel without pore volume or with a mass the maps do not hold leaves the
    domain without a total, its known part kept; other domains are complete."""
    geometry, sand, maps, co2 = frame()
    rows = DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    row = {r["domain_id"]: r for r in rows}

    assert row["whole"]["unknown_pixels"] == 2 and row["whole"]["total_mass_g"] is None
    assert row["layer:3"]["unknown_pixels"] == 2 and not row["layer:3"]["complete"]
    known = geometry.domain & ~np.isnan(maps.mass.img) & (geometry.pore_volume_m3 > 0)
    assert row["whole"]["gas_quantified_mass_g"] == pytest.approx(
        geometry.mass_g(maps.mass_g.img, known), rel=1e-6
    )
    for complete in ("layer:2", "layer:5"):
        assert row[complete]["complete"] and row[complete]["unknown_pixels"] == 0
        assert row[complete]["total_mass_g"] == row[complete]["total_quantified_mass_g"]
    empty = row["outside"]
    assert empty["pixels"] == 0 and not empty["complete"]
    assert (
        empty["total_quantified_mass_g"] == 0 and empty["gas_saturation_mean"] is None
    )


def test_means_are_weighted_by_pore_volume():
    """Saturation and concentration are means over the known pixels where they are
    finite, weighted by pore volume; the concentration in g/L."""
    geometry, sand, maps, co2 = frame(np.float64)
    row = {
        r["domain_id"]: r for r in DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    }

    volume = geometry.pore_volume_m3
    for key, values in (
        (
            "aqueous_concentration_mean_g_l",
            maps.concentration_aq.img * co2.solubility_co2,
        ),
        ("gas_saturation_mean", maps.saturation_g.img),
    ):
        known = geometry.domain & (sand.img == 3) & np.isfinite(values) & (volume > 0)
        known &= np.isfinite(maps.mass.img)
        assert row["layer:3"][key] == pytest.approx(
            np.average(values[known], weights=volume[known]), rel=1e-12
        )


def test_each_mass_has_its_centre_and_spread():
    """The centre of each mass in mm of the grid and its standard deviation along
    each axis; none without mass."""
    geometry, sand, maps, co2 = frame(np.float64)
    row = {
        r["domain_id"]: r for r in DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    }

    x, y = np.meshgrid(np.arange(90) * 10 + 5.0, 485.0 - np.arange(49) * 10)
    grams = np.nan_to_num(maps.mass_g.img) * geometry.pore_volume_m3
    centre = np.average(x, weights=grams)
    assert row["whole"]["gas_centre_x_mm"] == pytest.approx(centre, rel=1e-12)
    assert row["whole"]["gas_spread_y_mm"] == pytest.approx(
        np.sqrt(np.average((y - np.average(y, weights=grams)) ** 2, weights=grams)),
        rel=1e-9,
    )
    assert row["layer:5"]["gas_centre_x_mm"] is None


def test_a_domain_receives_the_injection_of_its_own_points(tmp_path):
    """The whole domain receives all the injected CO2, a box what the points inside
    it inject and a sand layer none; a box whose edge passes through an injection
    point is refused, as its pixels and its points would disagree there."""
    path = tmp_path / "injection.csv"
    start, end = (pd.Timestamp(0, unit="s") + pd.Timedelta(hours=h) for h in (0, 2))
    path.write_text(
        "id,location_x,location_y,start,end,rate_kg/s\n"
        f"1,0.2,0.1,{start.isoformat()},{end.isoformat()},2e-9\n"
        f"2,0.6,0.3,{start.isoformat()},{end.isoformat()},3e-9\n"
    )
    protocol = darsia.InjectionProtocol(path)
    found = {d.domain_id: injected_mass(d, protocol, 1.0) for d in DOMAINS}

    assert found["whole"] == pytest.approx(3600 * 5e-9, rel=1e-12)
    assert found["box1"] == pytest.approx(3600 * 2e-9, rel=1e-12)
    assert found["box2"] == pytest.approx(3600 * 3e-9, rel=1e-12)
    assert found["layer:3"] is None and found["sand"] is None
    edge = AnalysisDomain("box", "edge", "edge", bounds_mm=(0.0, 600.0, 0.0, 490.0))
    with pytest.raises(ValueError, match="passes through an injection point"):
        injected_mass(edge, protocol, 1.0)
