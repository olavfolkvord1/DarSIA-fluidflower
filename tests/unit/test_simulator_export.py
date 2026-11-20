"""Tests of the simulator export: the sand map filled to the frame, the completed
geometry, its cells, each run's own geometry, the mass maps, and the overflow curve a
run without readings borrows."""

import csv
import json
from dataclasses import replace

import numpy as np
import openpyxl
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.export.simulator import (
    NO_DONOR,
    NO_PRESSURE,
    CompletedGeometry,
    MassFrames,
    SimulatorExportSettings,
    SimulatorRun,
    SimulatorSeries,
    complete_geometry,
    geometry_cells,
    sand_layers,
    write_simulator_package,
)
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    rasterize_sand_map,
    rig_grid,
)

SETTINGS = dict(
    porous_facies=[2, 5],
    free_water_facies=[1],
    fixture_facies=[9],
    excluded_facies=[10],
    cell_sizes_mm=[1.0],
    precision="float32",
    overflow_columns=[
        "image_time",
        "time_from_injection_h",
        "weight_g",
        "status",
        "injection_start",
        "weight_data_valid",
        "weight_data_status",
    ],
    flow_reference=dict(
        gas="CO2",
        flow_unit="mls/min",
        reference_temperature_c=20.0,
        reference_pressure_hpa=1013.25,
    ),
)
"""Export settings of the tests: our production's cell size, precision, overflow
columns and flow reference, and the facies of the test rig."""

OVERFLOW = darsia.OverflowSettings(
    statuses=["Verified"],
    recipe_tolerance={"alkalinity_M": 0.03},
    variant="circulation",
    circulation=dict(minimum_readings=3),
)
"""Overflow settings of the tests: a circulating overflow, its weight read as is, and
our production's recipe tolerance."""


def test_sand_behind_a_fixture_is_the_sand_drawn_beneath_it():
    """Beneath a fixture the drawn sand takes its place, whatever the surface; a
    fixture pixel of the domain with nothing drawn beneath it is refused, and so is
    a layer without a facies drawn there. Silicone lies outside the domain, also a
    whole column of it."""
    rows, cols = 60, 80
    heights = (rows - np.arange(rows) - 0.5) * 0.001
    # The boundary between the upper sand (layer 2) and the lower (3) rises from
    # 25 mm at column 29 to 33 mm at column 52, between pixel centres.
    boundary = 0.025 + (np.arange(cols) - 29) * 0.008 / 23
    truth = np.where(heights[:, None] < boundary[None, :], 3, 2).astype(np.uint8)
    truth[:10] = 1  # free water
    layers = truth.copy()
    layers[15:46, 30:52] = 4  # the fixture
    layers[:, 0] = 5  # silicone along the left wall
    layers[50:56, 60:70] = 5  # and a piece of it in the sand
    hidden = layers == 4
    behind = np.where(hidden, truth, 0).astype(np.uint8)  # the sand drawn beneath it
    inside = np.ones(layers.shape, bool)
    settings = SimulatorExportSettings(**SETTINGS)
    facies = {1: 1, 2: 5, 3: 2, 4: 9, 5: 10}

    completed = complete_geometry(layers, inside, behind, facies, settings)

    domain = np.zeros(layers.shape, bool)
    domain[10:] = True
    domain &= layers != 5
    assert np.array_equal(completed.domain, domain)
    assert np.array_equal(
        completed.facies[domain], np.array([0, 1, 5, 2])[truth][domain]
    )
    assert np.array_equal(completed.drawn_behind, hidden)

    undrawn = behind.copy()
    undrawn[40, 40] = 0  # nothing drawn beneath one pixel of the fixture
    with pytest.raises(ValueError, match="neither sand nor sand drawn"):
        complete_geometry(layers, inside, undrawn, facies, settings)
    behind[30, 40] = 6  # a layer without a facies
    with pytest.raises(ValueError, match="have no facies"):
        complete_geometry(layers, inside, behind, facies, settings)


def test_every_pixel_no_layer_covers_takes_the_nearest_layer():
    """Inside the frame, a pixel that no layer covers takes the nearest layer at any
    distance from the frame's edge: the strip the drawing leaves along the left wall
    takes the wall's silicone, the top the free water below it, and a hole in the
    sand the sand around it. So does a pixel beneath a fixture that nothing is drawn
    beneath: beneath a colour checker reaching the frame's top, the top takes the
    free water drawn below it, so no sand lies above the free water. The completed
    domain then holds the hole and the sand drawn beneath the checker, but not the
    strip, and nothing is refused. A canvas that does not start at the rig's inner
    corner is refused."""
    anchors = [[0.003, 0], [0.006, 0], [0.006, 0.1], [0.003, 0.1]]  # silicone
    anchors += [[0.2, 0], [0.2, 0.02], [0.006, 0.02]]  # the lowest sand
    anchors += [[0.1, 0.02], [0.1, 0.06], [0.006, 0.06]]  # left of the hole
    anchors += [[0.104, 0.02], [0.2, 0.06], [0.104, 0.06]]  # right of it
    anchors += [[0.2, 0.098], [0.006, 0.098]]  # free water, 2 mm below the top
    anchors += [[0.15, 0.03], [0.18, 0.03], [0.18, 0.1], [0.15, 0.1]]  # the checker
    polygons = ((4, (0, 1, 2, 3)), (2, (1, 4, 5, 6)), (2, (6, 7, 8, 9)))
    polygons += ((2, (10, 5, 11, 12)), (1, (9, 11, 13, 14)), (9, (15, 16, 17, 18)))
    drawing = SandMapDrawing(np.array(anchors, float), polygons)
    canvas = darsia.Image(np.zeros((100, 200)), dimensions=[0.1, 0.2])
    shape, grid = (100, 200), rig_grid(canvas)
    painted, _ = rasterize_sand_map(
        drawing, shape, grid, width=0.2, height=0.1, edge_fill_mm=0.0
    )

    layers, inside, behind, filled = sand_layers(drawing, canvas, [9])

    assert inside.all() and (layers != 0).all()
    assert np.array_equal(filled, painted == 0)
    assert np.array_equal(layers[~filled], painted[~filled])
    strip, top = (slice(None), slice(0, 3)), (slice(0, 2), slice(9, 140))
    hole = (slice(45, 80), slice(101, 104))  # 20 to 55 mm up, 101 to 104 mm across
    assert filled[strip].all() and (layers[strip] == 4).all()
    assert filled[top].all() and (layers[top] == 1).all()
    assert filled[hole].all() and (layers[hole] == 2).all()
    checker = layers == 9
    assert (behind[checker] != 0).all() and not behind[~checker].any()
    assert (behind[:2][checker[:2]] == 1).all()  # free water beneath, at the top

    facies = {1: 1, 2: 5, 4: 10, 9: 9}
    settings = SimulatorExportSettings(**SETTINGS)
    completed = complete_geometry(layers, inside, behind, facies, settings)
    assert np.array_equal(completed.domain, (layers == 2) | (checker & (behind == 2)))

    shifted = darsia.Image(np.zeros(shape), dimensions=[0.1, 0.2], origin=[0.01, 0.11])
    with pytest.raises(ValueError, match="lower left corner"):
        sand_layers(drawing, shifted, [9])


def test_cells_sum_the_domain_s_volumes_and_take_the_lowest_majority():
    """On metric cells, bulk and pore volume are sums over the domain's pixels,
    porosity is pore over bulk volume and depth bulk volume over domain area, both
    NaN without domain; the majority goes to the lower facies on a tie, the rows run
    from the floor up, and the arrays are stored in the chosen precision."""
    facies = np.array([[2, 2, 5, 5], [2, 5, 5, 5], [2, 2, 5, 5], [5, 5, 2, 2]])
    domain = np.ones((4, 4), bool)
    domain[0, 0] = False  # a pixel outside in the upper left cell
    domain[:2, 2:] = False  # and no domain in the upper right cell
    depth = np.broadcast_to(np.float32([0.010, 0.011, 0.012, 0.013]), (4, 4))
    none = np.zeros((4, 4), bool)
    completed = CompletedGeometry(np.where(domain, facies, -1), domain, none, {})

    cells = geometry_cells(
        completed, depth, {2: 0.4, 5: 0.2}, [0.004, 0.004], 2.0, "float32"
    )

    assert cells["porosity"].dtype == np.float32
    assert np.array_equal(cells["facies_majority"], [[2, 2], [2, 0]])
    np.testing.assert_allclose(cells["domain_area_fraction"], [[1, 1], [0.75, 0]])
    d = depth[0].astype(np.float64)
    # Lower left cell: sand 2 over sand 5 in columns 0 and 1; upper left: three pixels.
    bulk = [1e-6 * 2 * (d[0] + d[1]), 1e-6 * (d[1] + d[0] + d[1])]
    pore = [1e-6 * 0.6 * (d[0] + d[1]), 1e-6 * (0.4 * d[1] + 0.4 * d[0] + 0.2 * d[1])]
    np.testing.assert_allclose(cells["bulk_volume_m3"][:, 0], bulk, rtol=1e-6)
    np.testing.assert_allclose(cells["pore_volume_m3"][:, 0], pore, rtol=1e-6)
    np.testing.assert_allclose(
        cells["porosity"][:, 0], np.divide(pore, bulk), rtol=1e-6
    )
    np.testing.assert_allclose(
        cells["depth_m"][:, 0], np.divide(bulk, [4e-6, 3e-6]), rtol=1e-6
    )
    assert np.isnan(cells["porosity"][1, 1]) and np.isnan(cells["depth_m"][1, 1])


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _series(tmp_path, drawing, facies_of_layer):
    """A series on a 0.2 x 0.1 m rig of 1 mm pixels, 10 mm deep."""
    return SimulatorSeries(
        drawing=drawing,
        facies_of_layer=facies_of_layer,
        canvas=darsia.Image(np.full((100, 200), 0.01), dimensions=[0.1, 0.2]),
        facies_properties=_write(
            tmp_path / "facies.csv",
            "id,porosity,permeability\n1,1.0,0\n2,0.44,473\n5,0.44,4295\n",
        ),
        reference_density=1.842,
    )


def _injection(tmp_path, name, steps=(("09:54:22", "10:54:22", 2.0),)):
    """A run's injection: its steps (start, end and rate [mL/min]) of the day; one
    hour at 2 mL/min from 09:54:22 by default."""
    header = "id,location_x,location_y,start,end,rate_ml/min,density kg/m3\n"
    rows = "".join(
        f"{i},0.1,0.02,2022-10-18T{start}+02:00,2022-10-18T{end}+02:00,{rate},1.842\n"
        for i, (start, end, rate) in enumerate(steps, start=1)
    )
    return _write(tmp_path / f"{name}_injection.csv", header + rows)


def _pressure(tmp_path, name, by_hour):
    """A run's pressure record: the pressure [bar] at each given hour of the day, an
    empty cell for ''."""
    header = "datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,"
    header += "temperature_gradient_celsius\n"
    rows = "".join(
        f"2022-10-18T{hour:02d}:00:00+02:00,{bar},23,0,0\n"
        for hour, bar in by_hour.items()
    )
    return _write(tmp_path / f"{name}_pt.csv", header + rows)


def _protocols(tmp_path, name, pressure_bar):
    """A run's injection and its pressure record, constant over ten hours."""
    pressure = _pressure(tmp_path, name, dict.fromkeys(range(8, 18), pressure_bar))
    return _injection(tmp_path, name), pressure


def _overflow(tmp_path, name, weights):
    """A run's overflow table: its scale readings from 09:30, all verified."""
    columns = "image_time,time_from_injection_h,weight_g,status,injection_start,"
    columns += "weight_data_valid,weight_data_status\n"
    times = ("09:30", "09:45", "10:30", "11:30", "13:00")
    return _write(
        tmp_path / f"{name}_overflow.csv",
        columns
        + "".join(
            f"2022-10-18T{t}:00,,{w},Verified,2022-10-18T09:54:22,True,ok\n"
            for t, w in zip(times, weights)
        ),
    )


def _flat_drawing():
    """A drawing of one sand layer over the whole rig."""
    return SandMapDrawing(
        np.array([[0, 0], [0.2, 0], [0.2, 0.1], [0, 0.1]], float), ((2, (0, 1, 2, 3)),)
    )


def test_adjusted_runs_carry_their_own_geometry(tmp_path):
    """A run with an adjustment gets the geometry of its own drawing; the others
    share the common geometry, and the README names the run that differs. The
    manifest counts the pixels of the strip the drawing leaves along the left wall
    by the layer they took. A protocol read from a workbook's sheet is copied as the
    same table from a CSV file."""
    anchors = [[0.003, 0], [0.2, 0], [0.2, 0.04], [0.1, 0.04], [0.003, 0.04]]
    anchors += [[0.2, 0.08], [0.003, 0.08], [0.2, 0.1], [0.003, 0.1]]
    polygons = ((1, (6, 5, 7, 8)), (2, (4, 3, 2, 5, 6)), (3, (0, 1, 2, 3, 4)))
    drawing = SandMapDrawing(np.array(anchors, float), polygons)
    series = _series(tmp_path, drawing, {1: 1, 2: 5, 3: 2})
    injection, pressure = _protocols(tmp_path, "run", 1.01)
    book = tmp_path / "injection.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "c"
    with injection.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            workbook.active.append(row)
    workbook.save(book)
    settled = {3: (0.1, 0.06)}  # the boundary's middle anchor moved up 20 mm
    recipe = {"alkalinity_M": 0.001}
    runs = {
        "a": SimulatorRun(injection, pressure, recipe),
        "b": SimulatorRun(injection, pressure, recipe, adjustment=settled),
        "c": SimulatorRun((book, "c"), pressure, recipe),
    }

    folder = tmp_path / "package"
    folder.mkdir()
    manifest = write_simulator_package(
        folder, series, runs, SimulatorExportSettings(**SETTINGS), OVERFLOW
    )

    common = manifest["geometry"]["1mm"]
    assert manifest["runs_with_own_geometry"] == ["b"]
    assert [manifest["runs"][r]["geometry"]["1mm"] for r in "ac"] == [common] * 2
    own = manifest["runs"]["b"]["geometry"]["1mm"]
    assert own != common and own.startswith("runs/b/")
    assert "b." in (folder / "README.md").read_text(encoding="utf-8")
    shared, settled_cells = (np.load(folder / path) for path in (common, own))
    # Rows run from the floor up: sand in the lowest row, free water in the highest.
    area = shared["domain_area_m2"]
    assert area[0].all() and not area[-1].any()
    # The cell 47 to 48 mm up at x 97 to 98 mm lies above the common boundary
    # (upper sand, facies 5) and below the run's raised one (lower sand, 2).
    assert shared["facies_majority"][47, 97] == 5
    assert settled_cells["facies_majority"][47, 97] == 2
    drawn = json.loads((folder / "runs" / "b" / "sand_map.json").read_text())
    assert drawn["anchors"][3] == [0.1, 0.06]
    # The 3 mm strip takes in each row the layer drawn beside it.
    for counts, sand_map in (
        (manifest["filled_pixels_by_layer"], drawing),
        (manifest["runs"]["b"]["filled_pixels_by_layer"], drawing.moved(settled)),
    ):
        painted, _ = rasterize_sand_map(
            sand_map,
            (100, 200),
            rig_grid(series.canvas),
            width=0.2,
            height=0.1,
            edge_fill_mm=0.0,
        )
        assert not painted[:, :3].any() and painted[:, 3:].all()
        layer, rows = np.unique(painted[:, 3], return_counts=True)
        assert counts == {str(k): 3 * int(n) for k, n in zip(layer, rows)}
        assert list(counts) == ["1", "2", "3"]
    copied = [(folder / "runs" / r / "injection.csv").read_bytes() for r in "ac"]
    assert copied[0] == copied[1] and manifest["runs"]["c"]["rows"]["injection"] == 1
    starts = [manifest["runs"][r]["injection_start"] for r in "ac"]
    assert starts[0] == starts[1]


def test_mass_maps_go_on_the_geometry_s_cells_from_the_floor_up(tmp_path):
    """A run's mass maps are written on the geometry's cells: the cells hold the
    map's total mass, the first row is the floor's, and maps on a grid that does not
    start at the canvas's lower left corner are refused."""
    series = _series(tmp_path, _flat_drawing(), {2: 5})
    shape, dims = (100, 200), [0.1, 0.2]
    geometry = darsia.run_geometry(
        darsia.Image(np.zeros(shape), dimensions=dims),
        np.full(shape, 1e-6),
        np.full(shape, 0.01),
        np.full(shape, 0.44),
        np.full(shape, 0.1),
        np.ones(shape, bool),
    )
    values = np.zeros(shape, np.float32)
    values[90:, :10] = 0.5  # CO2 in the lowest 10 mm at the left wall

    def frames(origin):
        """One frame of mass maps on a grid with the given origin."""
        fields = {
            key: darsia.Image(values.copy(), dimensions=dims, origin=origin)
            for key in ("mass", "mass_g", "mass_aq", "concentration_aq", "saturation_g")
        }
        maps = darsia.SimpleMassAnalysisResults(
            name="f", date=None, time=None, color_signal=None, **fields
        )
        return MassFrames(geometry, [("f", maps, np.zeros(shape, np.uint8))])

    injection, pressure = _protocols(tmp_path, "r", 1.01)
    recipe = {"alkalinity_M": 0.001}
    settings = SimulatorExportSettings(**SETTINGS)
    folder = tmp_path / "package"
    folder.mkdir()
    run = SimulatorRun(injection, pressure, recipe, mass=frames([0.0, 0.1]))
    manifest = write_simulator_package(folder, series, {"r": run}, settings, OVERFLOW)

    assert manifest["runs"]["r"]["mass_frames"] == 1
    with np.load(folder / "runs" / "r" / "mass" / "1mm" / "f.npz") as cells:
        total = cells["mass_total_g"]
    pixels = values.astype(np.float64) * geometry.pore_volume_m3 * 1000
    assert total.sum() == pytest.approx(pixels.sum(), rel=1e-12)
    assert total[:10, :10].all() and not total[10:].any() and not total[:, 10:].any()

    shifted = tmp_path / "shifted"
    shifted.mkdir()
    run = SimulatorRun(injection, pressure, recipe, mass=frames([0.0, 0.11]))
    with pytest.raises(ValueError, match="lower left"):
        write_simulator_package(shifted, series, {"r": run}, settings, OVERFLOW)


def test_a_run_without_readings_borrows_by_the_mass_rule(tmp_path):
    """A run without usable scale readings of its own takes the reviewed curve of the
    run of its recipe nearest to it in atmospheric pressure among the runs with
    usable readings it may borrow from, passing over a nearer run of another recipe
    and one whose readings are not usable; a run whose recipe no run shares takes
    the run nearest in pressure alone, and the package says so. An allowed run the
    package lacks is named. The pressures are compared from the injection's start to
    the later of its end at a positive rate and the donors' last reading."""
    series = _series(tmp_path, _flat_drawing(), {2: 5})
    usual, other, alone, late = 0.001, 0.002, 0.005, 0.004  # alkalinity [M]
    readings = (100.0, 102.0, 110.0, 120.0, 125.0)
    runs = {}
    for name, pressure_bar, alkalinity, weights, donors in (
        ("a", 1.010, usual, None, None),
        ("b", 1.011, other, (100.0, 101.0, 130.0, 160.0, 175.5), None),
        ("c", 1.014, usual, readings, None),
        ("d", 1.0112, alone, None, None),  # no run shares its recipe
        ("e", 1.014, usual, None, ["f", "z"]),  # may borrow from f (and z) only
        ("f", 1.030, usual, (100.0, 100.5, 105.0, 111.0, 113.0), None),
        ("g", 1.0101, usual, (100.0, 100.0), None),  # too few readings to use
        ("n", 1.013, late, readings, None),
    ):
        overflow = None if weights is None else _overflow(tmp_path, name, weights)
        injection, pressure = _protocols(tmp_path, name, pressure_bar)
        recipe = {"alkalinity_M": alkalinity}
        runs[name] = SimulatorRun(injection, pressure, recipe, overflow, donors)
    # k injects twice, the second time until 14:24, and then rests at rate zero
    # until 16:54. Its window ends at 14:24, after the donors' last reading at 13:00:
    # p matches its pressure until 13:00 and q from 13:00 to 15:00, while n lies
    # 3 mbar off throughout, so p is nearest until 13:00, q until 14:24 and n until
    # 16:54.
    steps = (("09:54:22", "10:54:22", 2.0), ("12:54:22", "14:24:22", 2.0))
    runs["k"] = SimulatorRun(
        _injection(tmp_path, "k", (*steps, ("14:24:22", "16:54:22", 0.0))),
        _pressure(tmp_path, "k", dict.fromkeys(range(8, 18), 1.010)),
        {"alkalinity_M": late},
    )
    hours = range(8, 18)
    for name, by_hour in (
        ("p", {h: 1.010 if h <= 13 else 1.060 for h in hours}),
        ("q", {h: 1.014 if h <= 12 else 1.010 if h <= 15 else 1.060 for h in hours}),
    ):
        runs[name] = SimulatorRun(
            _injection(tmp_path, name),
            _pressure(tmp_path, name, by_hour),
            {"alkalinity_M": late},
            _overflow(tmp_path, name, readings),
        )

    folder = tmp_path / "package"
    folder.mkdir()
    manifest = write_simulator_package(
        folder, series, runs, SimulatorExportSettings(**SETTINGS), OVERFLOW
    )

    assert manifest["missing_own_overflow"] == ["a", "d", "e", "g", "k"]
    assert manifest["borrowed_overflow"] == dict(a="c", d="b", e="f", g="c", k="q")
    assert list(manifest["borrowed_overflow_unmatched"]) == ["d"]
    assert manifest["not_borrowed_overflow"] == {}
    assert manifest["runs"]["g"]["overflow_not_usable"]
    entry = manifest["runs"]["e"]
    assert entry["overflow_donors_allowed"] == ["f"]
    assert entry["overflow_donors_not_in_package"] == ["z"]
    path = manifest["runs"]["a"]["protocols"]["overflow_reviewed"]
    assert path == "runs/a/overflow_reviewed_borrowed_from_c.csv"
    borrowed = pd.read_csv(folder / path)
    own = pd.read_csv(folder / manifest["runs"]["c"]["protocols"]["overflow_reviewed"])
    assert (borrowed.pop("borrowed_from") == "c").all()
    pd.testing.assert_frame_equal(borrowed, own)
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "a from c," in readme and "d from b (no run with its sand map" in readme


def test_a_run_left_without_a_donor_says_why_and_broken_inputs_are_refused(tmp_path):
    """A run whose runs of the same recipe lack atmospheric pressure over the
    comparison window borrows nothing, as the mass maps' estimate has nothing, and
    is not given a run of another recipe; a run whose allowed runs are all missing
    from the package borrows nothing either, and both say why. A pressure record
    with an empty cell, a recipe without an entry the tolerance compares, and a
    protocol row with a cell more than its header are refused."""
    series = _series(tmp_path, _flat_drawing(), {2: 5})
    readings = (100.0, 102.0, 110.0, 120.0, 125.0)
    hours = range(8, 18)
    runs = {
        "h": SimulatorRun(
            _injection(tmp_path, "h"),
            _pressure(tmp_path, "h", dict.fromkeys(hours, 1.010)),
            {"alkalinity_M": 0.003},
        ),
        # The same recipe, but pressure only from 11:00, after the injection began.
        "i": SimulatorRun(
            _injection(tmp_path, "i"),
            _pressure(tmp_path, "i", dict.fromkeys(range(11, 18), 1.0101)),
            {"alkalinity_M": 0.003},
            _overflow(tmp_path, "i", readings),
        ),
        "b": SimulatorRun(
            _injection(tmp_path, "b"),
            _pressure(tmp_path, "b", dict.fromkeys(hours, 1.011)),
            {"alkalinity_M": 0.005},
            _overflow(tmp_path, "b", readings),
        ),
        "x": SimulatorRun(
            _injection(tmp_path, "x"),
            _pressure(tmp_path, "x", dict.fromkeys(hours, 1.012)),
            {"alkalinity_M": 0.001},
            donors=["z"],
        ),
    }
    settings = SimulatorExportSettings(**SETTINGS)

    folder = tmp_path / "package"
    folder.mkdir()
    manifest = write_simulator_package(folder, series, runs, settings, OVERFLOW)

    assert manifest["borrowed_overflow"] == {}
    assert manifest["not_borrowed_overflow"] == dict(h=NO_PRESSURE, x=NO_DONOR)
    assert manifest["runs"]["h"]["overflow_not_borrowed"] == NO_PRESSURE
    assert manifest["runs"]["x"]["overflow_donors_allowed"] == []
    assert manifest["runs"]["x"]["overflow_donors_not_in_package"] == ["z"]
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert f"Without a curve: h ({NO_PRESSURE}), x ({NO_DONOR})." in readme

    gap = dict.fromkeys(range(11, 18), 1.0101) | {12: ""}
    longer = (
        runs["b"].pressure_temperature.read_text(encoding="utf-8") + "1,2,3,4,5,6\n"
    )
    for changed, refusal in (
        (dict(i=dict(pressure_temperature=_pressure(tmp_path, "gap", gap))), "of i"),
        (dict(h=dict(recipe={})), "recipe of h lacks"),
        (
            dict(b=dict(pressure_temperature=_write(tmp_path / "long.csv", longer))),
            "another number of cells",
        ),
    ):
        target = tmp_path / f"refused_{refusal[:5]}"
        target.mkdir()
        changed = {
            name: replace(runs[name], **fields) for name, fields in changed.items()
        }
        with pytest.raises(ValueError, match=refusal):
            write_simulator_package(
                target, series, {**runs, **changed}, settings, OVERFLOW
            )
