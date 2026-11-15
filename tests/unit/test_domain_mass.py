"""Unit tests for the mass per analysis domain on a coarse synthetic rig."""

import csv
import tomllib

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.analysis.domain_mass import (
    COLUMNS,
    DomainMass,
    RunSeries,
    write_domain_mass,
)
from darsia.presets.workflows.analysis.domains import (
    AnalysisDomain,
    injected_mass,
    layer_parts,
)

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
def test_the_whole_domain_holds_the_map_total(dtype, tmp_path):
    """The whole domain's mass is the map's total, and boxes sharing a side and the
    sand layers each split it with no pixel counted twice; the sand is its layers
    without the free water. The rows are written in the table's columns, a NumPy
    number in full."""
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
    # The pixel without pore volume, whose maps hold a number, is no part of it.
    assert row["whole"]["pixels"] == geometry.domain.sum() - 1
    assert row["box1"]["pixels"] == 15 * 89

    path = tmp_path / "domain_mass.csv"
    with np.printoptions(legacy="1.13"):
        write_domain_mass(
            path,
            [
                dict(r, run="run", case_id="frame", elapsed_h=np.float64(1 / 3))
                for r in rows
            ],
        )
    with path.open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0]) == list(COLUMNS) and len(COLUMNS) == 31
    assert table[0]["total_mass_g"] == "" and table[3]["complete"] == "True"
    assert table[0]["elapsed_h"] == repr(1 / 3)


def test_an_unknown_pixel_leaves_its_domains_without_a_total():
    """A pixel with a mass the maps do not hold leaves the domain without a total,
    its known part kept; other domains are complete. A pixel without pore volume
    whose maps hold a number holds no CO2 and is no part of its domains, as our
    production left it out; without a number there, it is unknown."""
    geometry, sand, maps, co2 = frame()
    rows = DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    row = {r["domain_id"]: r for r in rows}

    assert row["whole"]["unknown_pixels"] == 1 and row["whole"]["total_mass_g"] is None
    assert row["layer:3"]["unknown_pixels"] == 1 and not row["layer:3"]["complete"]
    assert row["whole"]["pixels"] == geometry.domain.sum() - 1
    known = geometry.domain & ~np.isnan(maps.mass.img) & (geometry.pore_volume_m3 > 0)
    assert row["whole"]["gas_quantified_mass_g"] == pytest.approx(
        geometry.mass_g(maps.mass_g.img, known), rel=1e-6
    )
    for complete in ("layer:2", "layer:5"):
        assert row[complete]["complete"] and row[complete]["unknown_pixels"] == 0
        assert row[complete]["total_mass_g"] == row[complete]["total_quantified_mass_g"]
    maps.mass_g.img[NO_VOLUME] = np.nan
    again = {
        r["domain_id"]: r for r in DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    }
    assert again["whole"]["pixels"] == geometry.domain.sum()
    assert again["whole"]["unknown_pixels"] == 2
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


def test_boxes_and_centres_read_each_pixels_measured_position():
    """With each pixel's measured position, a box holds the pixels whose position
    lies in it, compared in double precision, and the centres of mass are taken
    at the positions."""
    geometry, sand, maps, co2 = frame(np.float64)
    centres = darsia.grid_positions(sand)
    x = (centres.x + 0.012 * np.linspace(0.0, 1.0, 49)[:, None]).astype(np.float32)
    positions = darsia.GridPositions(x, centres.y.astype(np.float32), centres.pitch_mm)
    measured = DomainMass(DOMAINS, geometry, sand, SAND, positions)(maps, co2)
    plain = DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    row = {r["domain_id"]: r for r in measured}
    before = {r["domain_id"]: r for r in plain}

    x64, y64 = (np.asarray(v, np.float64) for v in (positions.x, positions.y))
    held = geometry.domain & (geometry.pore_volume_m3 > 0)  # the maps' numbers
    left = held & (x64 >= 0) & (x64 < 0.455) & (y64 >= 0) & (y64 < 0.49)
    assert row["left"]["pixels"] == int(left.sum()) != before["left"]["pixels"]
    right = held & (x64 >= 0.455) & (x64 < 0.9) & (y64 >= 0) & (y64 < 0.49)
    assert row["right"]["pixels"] == int(right.sum())
    grams = np.nan_to_num(maps.mass_g.img) * geometry.pore_volume_m3
    assert row["whole"]["gas_centre_x_mm"] == pytest.approx(
        np.average(x64 * 1000, weights=grams), rel=1e-12
    )
    assert row["whole"]["gas_centre_x_mm"] != before["whole"]["gas_centre_x_mm"]

    # A box edge just past a pixel's position in double precision, level with it
    # in single precision: the pixel lies inside.
    edge = float(positions.x[20, 30]) + 1e-12
    box = AnalysisDomain(
        kind="box", domain_id="edge", name="edge", bounds_mm=[0.0, edge * 1000, 0, 490]
    )
    (found,) = DomainMass([box], geometry, sand, SAND, positions)(maps, co2)
    inside = held & (x64 >= 0) & (x64 < edge) & (y64 >= 0) & (y64 < 0.49)
    assert inside[20, 30] and found["pixels"] == int(inside.sum())


PARTS = [
    AnalysisDomain(**row)
    for row in tomllib.loads(
        """
        # A layer split into its parts and smaller pieces, as our production's
        # domains split two layers of its sand map.
        [[domain]]
        kind = "layer"
        domain_id = "layer:3"
        name = "3 E"
        label = 3
        part = 1

        [[domain]]
        kind = "layer"
        domain_id = "layer:3:part:2"
        name = "3 E, part 2 of 2"
        label = 3
        part = 2

        [[domain]]
        kind = "layer"
        domain_id = "layer:3:pieces"
        name = "3 E, pieces under 1 cm2"
        label = 3
        part = "pieces"

        [[domain]]
        kind = "layer"
        domain_id = "layer:5"
        name = "5 C"
        label = 5
        """
    )["domain"]
]


def test_a_layer_falls_into_its_parts_and_smaller_pieces():
    """A layer's face-connected pieces, ranked by size, are its parts while they
    reach the smallest area, the largest always; the smaller pieces are summed
    apart, and the parts and pieces hold the layer. A layer that runs on behind
    the colour checker is one piece, and a part that a layer lacks is refused."""
    geometry, sand, maps, co2 = frame(np.float64)
    labels = sand.img
    labels[9:30, 80] = 5  # layer 3 ends in a piece of 21 x 9 pixels (1 cm2 each)
    labels[40, 70:72] = 3  # and has a piece of two pixels inside layer 5
    labels[30:49, 40:43] = 17  # the chart, which layer 5 runs on behind
    area = geometry.pixel_area_m2
    parts = layer_parts(labels, SAND, area, 2.5e-4, chart_labels=(17,))
    layer = labels == 3
    assert (parts[layer] == 1).sum() == 21 * 80 and (parts[layer] == 0).sum() == 2
    np.testing.assert_array_equal(parts[9:30, 81:], 2)
    assert (parts[labels == 5] == 1).all()  # one piece behind the chart
    assert (parts[labels == 17] == -1).all() and (parts[labels == 2] == -1).all()
    alone = layer_parts(labels, SAND, area, 2.5e-4)
    assert (alone[:, 43:][labels[:, 43:] == 5] == 1).all()  # the larger side
    assert (alone[:, :40][labels[:, :40] == 5] == 2).all()

    sums = DomainMass(
        PARTS, geometry, sand, SAND, chart_labels=(17,), min_part_area_cm2=2.5
    )
    row = {r["domain_id"]: r for r in sums(maps, co2)}
    whole = DomainMass(DOMAINS[4:5], geometry, sand, SAND)(maps, co2)[0]
    assert row["layer:3:pieces"]["pixels"] == 2
    assert row["layer:3:part:2"]["pixels"] == 21 * 9
    for key in ("pixels", "pore_volume_l", "aqueous_quantified_mass_g"):
        found = sum(
            row[d][key] for d in ("layer:3", "layer:3:part:2", "layer:3:pieces")
        )
        assert found == pytest.approx(whole[key], rel=1e-12)
    assert row["layer:5"]["pixels"] == (geometry.domain & (labels == 5)).sum()
    with pytest.raises(ValueError, match="no part 2"):
        DomainMass(
            PARTS, geometry, sand, SAND, chart_labels=(17,), min_part_area_cm2=200
        )
    with pytest.raises(ValueError, match="needs a sand layer"):
        DomainMass(PARTS, geometry, sand, (5,))
    with pytest.raises(ValueError, match="Only a sand layer has a part"):
        AnalysisDomain(
            kind="box", domain_id="b", name="b", bounds_mm=(0, 1, 0, 1), part=1
        )


def test_a_run_series_is_read_from_the_table(tmp_path):
    """A run's rows, read from the table in any order, give its frames in time order,
    each domain's complete or quantified masses, an empty cell as not known, and the
    centres of its quantified masses."""
    geometry, sand, maps, co2 = frame(np.float64)
    rows = DomainMass(DOMAINS, geometry, sand, SAND)(maps, co2)
    path = tmp_path / "domain_mass.csv"
    write_domain_mass(
        path,
        [
            dict(r, run="run", case_id=case, elapsed_h=hours)
            for case, hours in (("late", 2.0), ("early", 0.5))
            for r in rows
        ],
    )
    with path.open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    injected = {"early": 0.1, "late": 0.4}
    complete = RunSeries.from_domain_mass(table, injected, "complete")
    quantified = RunSeries.from_domain_mass(table, injected, "quantified")

    row = {r["domain_id"]: r for r in rows}
    assert complete.case_ids == ("early", "late")
    assert complete.elapsed_h.tolist() == [0.5, 2.0]
    assert complete.injected_g.tolist() == [0.1, 0.4]
    assert np.isnan(complete.masses["whole"]["total"]).all()
    whole = row["whole"]["total_quantified_mass_g"]
    assert quantified.masses["whole"]["total"].tolist() == [whole, whole]
    free_water = row["layer:2"]["gas_mass_g"]
    assert complete.masses["layer:2"]["gas"].tolist() == [free_water, free_water]
    ends = ("_quantified_mass_g", "_centre_x_mm", "_centre_y_mm")
    ends += ("_spread_x_mm", "_spread_y_mm")
    centre = [row["whole"]["gas" + end] for end in ends]
    assert quantified.centres["whole"]["gas"][1].tolist() == centre
    assert np.isnan(quantified.centres["layer:5"]["gas"][:, 1:]).all()


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


def test_the_stored_maps_give_the_rows_of_the_maps_made(tmp_path):
    """The mass per domain of the stored maps, whose total mass is read as the sum
    of its parts, is that of the maps made, row for row."""
    geometry, sand, maps, co2 = frame()
    darsia.write_mass_maps(tmp_path, maps)
    stored = darsia.read_mass_maps(tmp_path)
    sums = DomainMass(DOMAINS, geometry, sand, SAND)
    assert pd.DataFrame(sums(stored, co2)).equals(pd.DataFrame(sums(maps, co2)))
