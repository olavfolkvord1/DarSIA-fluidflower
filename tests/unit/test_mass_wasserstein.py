"""Unit tests for the Wasserstein distances of mass maps on metric cells."""

import tomllib

import numpy as np
import pandas as pd
import pytest

import darsia
from darsia.presets.workflows.analysis.domains import AnalysisDomain, domain_pixels
from darsia.presets.workflows.comparison import mass_wasserstein as mw

SETTINGS = mw.WassersteinSettings(
    **tomllib.loads(
        """
        # The values our production ran with.
        cell_mm = 3.0
        hole_pixels = 3
        empty_mass_kg = 1e-8
        max_iterations = 60
        distance_tolerance = 1e-4
        increment_tolerance = 1e-2
        bregman_updates = [0, 10, 20]
        extent_edge_mm = 10
        """
    )
)
LOWER, UPPER, OTHER, CHART, WATER = 1, 2, 3, 7, 9
SAND = (LOWER, UPPER, OTHER)
ROWS, COLS = 30, 45
"""A grid of 1 mm pixels: two upper sands above the lower sand, free water on top,
and the colour checker between the two upper sands."""


def _sand_map():
    labels = np.full((ROWS, COLS), LOWER, np.uint8)
    labels[:15] = UPPER
    labels[:15, 39:] = OTHER
    labels[:3] = WATER
    labels[6:12, 30:39] = CHART
    return darsia.Image(labels, dimensions=[0.03, 0.045])


def _reference(domains):
    sand = _sand_map()
    permeability = np.select(
        [sand.img == LOWER, sand.img == UPPER, sand.img == OTHER], [400.0, 100.0, 200.0]
    )
    permeability[np.isin(sand.img, (CHART, WATER))] = np.nan
    porosity = np.where(sand.img == CHART, 0.0, 0.4)
    reference = mw.wasserstein_reference(
        sand, permeability, porosity, SAND, (CHART,), domains, SETTINGS
    )
    return sand, reference


def _geometry(sand):
    domain = np.ones((ROWS, COLS), bool)
    excluded = sand.img == CHART
    return darsia.run_geometry(
        sand,
        np.full((ROWS, COLS), 1e-6, np.float32),
        np.full((ROWS, COLS), 0.01, np.float32),
        np.full((ROWS, COLS), 0.4, np.float32),
        np.full((ROWS, COLS), 0.1, np.float32),
        domain,
        excluded,
    )


def _maps(mass):
    mass = mass.astype(np.float32)
    image = darsia.ScalarImage(mass, dimensions=[0.03, 0.045])
    finite = darsia.full_like(image, np.zeros_like(mass))
    return darsia.SimpleMassAnalysisResults(
        name="frame",
        date=None,
        time=None,
        mass=image,
        mass_g=finite,
        mass_aq=image,
        saturation_g=finite,
        color_signal=finite,
        concentration_aq=finite,
    )


def _cells(mass, reference, sand, labels=None):
    labels = np.ones((ROWS, COLS), np.uint8) if labels is None else labels
    return mw.frame_cells(
        _maps(mass),
        labels,
        sand,
        np.where(sand.img == LOWER, 1, 2),
        _geometry(sand),
        reference,
        SETTINGS,
    )


def _scales(regions):
    """Scales of one, with no injected mass in any region."""
    return mw.DistanceScales(1.0, 1.0, {d.domain_id: np.nan for d in regions})


def _plume(row, col, value=2.0):
    """A mass map (g/L) with a 6 x 6 mm plume whose top left pixel is given."""
    mass = np.zeros((ROWS, COLS))
    mass[row : row + 6, col : col + 6] = value
    return mass


def test_unknown_mass_gives_a_stated_status_not_a_failure():
    regions = [
        AnalysisDomain("sand", "sand", "sand"),
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.0, 45.0, 0.0, 15.0)),
    ]
    sand, reference = _reference(regions)
    labels = np.ones((ROWS, COLS), np.uint8)
    labels[4:6, 4:6] = darsia.PhaseLabel.UNKNOWN  # a hole of four pixels
    frames = {
        "a": _cells(_plume(18, 6), reference, sand),
        "b": _cells(_plume(18, 9), reference, sand),
        "c": None,
        "d": _cells(_plume(18, 12), reference, sand, labels),
    }
    results = {
        (r.run_1, r.run_2, r.domain_id): r
        for r in mw.wasserstein_distances(
            1.5, frames, reference, _scales(regions), SETTINGS
        )
    }
    assert len(results) == 12
    assert {r.status for key, r in results.items() if "c" in key} == {"missing"}
    unknown = results["a", "d", "sand"]
    assert unknown.status == "unknown" and unknown.unknown_pixels == (0, 4)
    assert np.isnan(unknown.distance_kg_m) and np.isnan(unknown.mass_2_kg)
    assert results["a", "d", "lower"].status == "success"
    assert results["a", "d", "lower"].distance_kg_m > 0
    assert results["a", "b", "sand"].status == "success"
    assert results["a", "b", "sand"].distance_kg_m > 0


def test_a_small_hole_borrows_the_mass_of_its_own_neighbours():
    density = np.arange(100, dtype=np.float32).reshape(10, 10)
    known = np.ones((10, 10), bool)
    support = np.ones((10, 10), bool)
    facies = np.zeros((10, 10), int)
    known[2, 2] = False  # one pixel
    known[2, 6:8] = False  # two pixels
    known[6:8, 1:3] = False  # four pixels: too large
    known[9, 5] = False  # on the edge
    known[6, 6], facies[5, 7] = False, 1  # a neighbour of another facies
    known[7, 8], support[7, 8] = False, False  # outside the support
    rows, cols, values = mw.fill_holes(density, known, support, facies, 3)
    assert list(zip(rows, cols)) == [(2, 2), (2, 6), (2, 7)]
    around = density[1:4, 1:4].astype(np.float64)
    assert values[0] == (around.sum() - around[1, 1]) / 8
    left = density[1:4, 5:8].astype(np.float64)
    assert values[1] == (left.sum() - left[1, 1] - left[1, 2]) / 7


def test_regions_sum_their_own_pixels_onto_the_cells():
    regions = [
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.5, 10.5, 0.0, 15.0)),
        AnalysisDomain("sand", "upper", "upper", bounds_mm=(0.0, 45.0, 15.0, 30.0)),
    ]
    sand, reference = _reference(regions)
    mass = np.full((ROWS, COLS), 3.0)
    cells = _cells(mass, reference, sand)
    pore_volume_l = 0.001**2 * 0.01 * 0.4 * 1000
    lower = cells["lower"]
    assert lower.mass.img.dtype == np.float32 and lower.mass.img.shape == (10, 15)
    assert lower.pixels.sum() == 15 * 10  # columns 0 to 9: centres 0.5 to 9.5 mm
    assert lower.mass.img.sum() * 0.003**2 == pytest.approx(
        150 * 3.0 * pore_volume_l / 1000, rel=1e-6
    )
    upper = cells["upper"]  # the sand of the upper half: no water, no chart
    assert upper.pixels.sum() == 12 * 45 - 6 * 9
    assert upper.unknown == upper.filled == 0


def test_each_run_goes_onto_the_cells_by_its_own_measured_positions():
    """With measured positions the cells cover every run's corners, the reference
    and each run go onto them by their own positions, and the boxes read them;
    without them nothing changes."""
    regions = [
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.5, 10.5, 0.0, 15.0)),
        AnalysisDomain("whole", "whole", "whole"),
    ]
    sand = _sand_map()
    centres = darsia.grid_positions(sand)
    moved = darsia.GridPositions(
        (centres.x + 0.002).astype(np.float32),
        (centres.y - 0.0004).astype(np.float32),
        centres.pitch_mm,
    )
    corners = [
        darsia.grid_corners(sand),
        (
            moved.x[[0, 0, -1, -1], [0, -1, 0, -1]],
            moved.y[[0, 0, -1, -1], [0, -1, 0, -1]],
        ),
    ]
    extent = darsia.measured_extent(corners, SETTINGS.cell_mm, 10)
    permeability = np.select(
        [sand.img == LOWER, sand.img == UPPER, sand.img == OTHER], [400.0, 100.0, 200.0]
    )
    permeability[np.isin(sand.img, (CHART, WATER))] = np.nan
    porosity = np.where(sand.img == CHART, 0.0, 0.4)
    reference = mw.wasserstein_reference(
        sand, permeability, porosity, SAND, (CHART,), regions, SETTINGS, None, extent
    )
    assert reference.extent == extent and reference.cells.shape == (10, 17)
    mass = _plume(16, 4)
    labels = np.ones((ROWS, COLS), np.uint8)
    facies = np.where(sand.img == LOWER, 1, 2)
    geometry = _geometry(sand)
    found = mw.frame_cells(
        _maps(mass), labels, sand, facies, geometry, reference, SETTINGS, moved
    )
    cells = darsia.metric_cells(sand.img.shape, sand.dimensions, 3.0, moved, extent)
    grams = np.where(geometry.counted, mass * geometry.pore_volume_m3 * 1000, 0)
    x, y = (np.asarray(v, np.float64) for v in (moved.x, moved.y))
    box = geometry.counted & (x >= 0.0005) & (x < 0.0105) & (y >= 0) & (y < 0.015)
    expected = darsia.cell_sums(cells, np.where(box, grams, 0.0)) / 1000 / 0.003**2
    np.testing.assert_array_equal(found["lower"].mass.img, expected.astype(np.float32))
    np.testing.assert_array_equal(found["lower"].pixels, darsia.cell_sums(cells, box))
    plain = _cells(mass, _reference(regions)[1], sand)
    assert not np.array_equal(plain["lower"].pixels.sum(), found["lower"].pixels.sum())


def test_the_reference_goes_onto_its_own_cells_first():
    """With measured positions the reference's pixels go onto the cells over its
    own positions, then by whole cells onto the comparison's, as in our
    production; in single precision a pixel can land in another cell than by
    counting on the comparison's cells directly."""
    sand = _sand_map()
    centres = darsia.grid_positions(sand)
    shifted = darsia.GridPositions(
        (centres.x + 0.0105).astype(np.float32),
        centres.y.astype(np.float32),
        centres.pitch_mm,
    )
    pair = [darsia.grid_corners(sand), (shifted.x, shifted.y)]
    extent = darsia.measured_extent(pair, SETTINGS.cell_mm, 10)
    own = darsia.measured_extent([(shifted.x, shifted.y)], SETTINGS.cell_mm, 10)
    assert own[0] > extent[0]
    permeability = np.select(
        [sand.img == LOWER, sand.img == UPPER, sand.img == OTHER], [400.0, 100.0, 200.0]
    )
    permeability[np.isin(sand.img, (CHART, WATER))] = np.nan
    porosity = np.where(sand.img == CHART, 0.0, 0.4)
    whole = [AnalysisDomain("whole", "whole", "whole")]
    reference = mw.wasserstein_reference(
        sand, permeability, porosity, SAND, (CHART,), whole, SETTINGS, shifted, extent
    )
    cells = darsia.metric_cells(sand.img.shape, sand.dimensions, 3.0, shifted, own)
    placed = darsia.placed_cells(cells, 3.0, own, extent)
    np.testing.assert_array_equal(reference.cells.index, placed.index)
    direct = darsia.metric_cells(sand.img.shape, sand.dimensions, 3.0, shifted, extent)
    assert not np.array_equal(direct.index, placed.index)

    # On cells over the plain grid's corners only, the reference reaches past them:
    # its cells there are cut, and every other cell keeps its weight.
    narrow = darsia.measured_extent([darsia.grid_corners(sand)], SETTINGS.cell_mm, 10)
    cut = mw.wasserstein_reference(
        sand, permeability, porosity, SAND, (CHART,), whole, SETTINGS, shifted, narrow
    )
    assert cut.cells.shape == (10, 17) and reference.cells.shape == (10, 20)
    assert np.count_nonzero(cut.cells.index < 0) == 150  # the last five columns
    for field in ("weight", "own", "porous"):
        np.testing.assert_array_equal(
            getattr(cut.weights["whole"], field),
            getattr(reference.weights["whole"], field)[:, :17],
        )


def test_the_reference_weights_its_means_by_its_pixels_areas():
    """Given each reference pixel's area, a cell's weight is the mean of its pixels'
    weights weighted by their areas, over the region's pixels in its own cells and
    over all pixels elsewhere; one area each gives the means of before."""
    labels = np.full((ROWS, COLS), LOWER, np.uint8)
    labels[:16] = UPPER  # the layers meet inside a row of cells
    sand = darsia.Image(labels, dimensions=[0.03, 0.045])
    permeability = np.where(labels == LOWER, 400.0, 100.0)
    porosity = np.full((ROWS, COLS), 0.4)
    model = [[0.001, 0, 0.0005], [0, -0.001, 0.0295], [2e-4, 1e-4, 1.0]]
    areas = darsia.grid_pixel_areas(sand, model, np.float64)
    box = AnalysisDomain("box", "left", "left", bounds_mm=(0.0, 10.5, 0.0, 30.0))
    args = (sand, permeability, porosity, SAND, (CHART,), [box], SETTINGS)
    reference = mw.wasserstein_reference(*args, None, None, areas)
    cells, pixel_weight = reference.cells, 400.0 / permeability
    inside = domain_pixels(box, sand, labels, SAND)
    own = darsia.cell_sums(cells, inside) > 0
    assert own.any() and not own.all()  # both rules
    region = darsia.cell_means(cells, np.where(inside, pixel_weight, np.nan), areas)
    everywhere = darsia.cell_means(cells, pixel_weight, areas)
    expected = np.where(own, region, everywhere)
    np.testing.assert_array_equal(reference.weights["left"].weight, expected)
    plain = mw.wasserstein_reference(*args)
    for part in (own, ~own):
        assert not np.array_equal(plain.weights["left"].weight[part], expected[part])
    uniform = np.full((ROWS, COLS), plain.cells.pixel_area)
    same = mw.wasserstein_reference(*args, None, None, uniform)
    np.testing.assert_array_equal(
        same.weights["left"].weight, plain.weights["left"].weight
    )


def test_the_sand_continues_under_the_colour_checker():
    _, reference = _reference([AnalysisDomain("sand", "sand", "sand")])
    weight = reference.weights["sand"]
    # The chart's cells take the weight of the nearer sand in each row: the upper
    # sand (4) on the left, the other sand (2) on the right, the left at equal
    # distance (pixel 34 of the chart's columns 30 to 38).
    assert np.all(weight.weight[2:4, 10] == 4.0)
    assert weight.weight[2:4, 11] == pytest.approx([10 / 3, 10 / 3], rel=1e-12)
    assert np.all(weight.weight[2:4, 12] == 2.0)
    assert np.all(weight.own[1:] & ~weight.porous[1:])
    assert np.all(weight.weight[5:] == 1.0)  # the lower sand: 400 / 400
    assert not weight.own[0].any()  # free water is no sand
    assert np.isnan(weight.weight[0]).all()  # and has no permeability

    box = AnalysisDomain("box", "box", "box", bounds_mm=(0.0, 45.0, 20.0, 30.0))
    _, reference = _reference([box])
    assert reference.weights["box"].porous[0].all()  # free water in the box


def test_both_runs_are_scaled_to_their_mean_mass():
    regions = [
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.0, 45.0, 0.0, 15.0))
    ]
    sand, reference = _reference(regions)
    frames = {
        "a": _cells(_plume(18, 6, 2.0), reference, sand),
        "b": _cells(_plume(18, 9, 4.0), reference, sand),
        "c": _cells(np.zeros((ROWS, COLS)), reference, sand),
        "d": _cells(np.zeros((ROWS, COLS)), reference, sand),
    }
    results = mw.wasserstein_distances(
        1.5, frames, reference, _scales(regions), SETTINGS
    )
    moved = results[0]
    assert (moved.run_1, moved.run_2, moved.status) == ("a", "b", "success")
    mean = 0.5 * (moved.mass_1_kg + moved.mass_2_kg)
    assert moved.mass_2_kg == pytest.approx(2 * moved.mass_1_kg, rel=1e-6)
    assert moved.distance_kg_m == pytest.approx(mean * 0.003, rel=3e-3)
    one_empty = next(r for r in results if (r.run_1, r.run_2) == ("a", "c"))
    assert one_empty.status == "one_empty" and np.isnan(one_empty.distance_kg_m)
    both_empty = next(r for r in results if (r.run_1, r.run_2) == ("c", "d"))
    assert both_empty.distance_kg_m == both_empty.normalized_mass == 0.0


def test_a_run_can_be_excluded_from_selected_domains():
    domains = [
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.0, 45.0, 0.0, 15.0)),
        AnalysisDomain("sand", "sand", "sand"),
    ]
    sand, reference = _reference(domains)
    empty = _cells(np.zeros((ROWS, COLS)), reference, sand)
    frames = {"a": empty, "b": None, "c": empty}
    result = mw.wasserstein_distances(
        1.5,
        frames,
        reference,
        _scales(domains),
        SETTINGS,
        excluded_runs={"lower": ["b"]},
    )
    assert {(row.run_1, row.run_2, row.domain_id, row.status) for row in result} == {
        ("a", "b", "sand", "missing"),
        ("a", "c", "lower", "success"),
        ("a", "c", "sand", "success"),
        ("b", "c", "sand", "missing"),
    }


def test_distances_are_normalised_by_the_scales_of_their_time(tmp_path):
    protocol = tmp_path / "injection.csv"
    times = [
        (pd.Timestamp(0, unit="s") + pd.Timedelta(hours=h)).isoformat() for h in (0, 2)
    ]
    protocol.write_text(
        "id,location_x,location_y,start,end,rate_kg/s\n"
        f"1,0.010,0.005,{times[0]},{times[1]},2e-9\n"
        f"2,0.030,0.025,{times[0]},{times[1]},3e-9\n"
    )
    injection = darsia.InjectionProtocol(protocol)
    baseline = darsia.ScalarImage(np.zeros((ROWS, COLS)), dimensions=[0.03, 0.045])
    co2 = darsia.CO2MassAnalysis(baseline, 1.0, 22.0)
    depth = np.linspace(0.01, 0.02, ROWS * COLS).reshape(ROWS, COLS)
    regions = [
        AnalysisDomain("box", "lower", "lower", bounds_mm=(0.0, 45.0, 0.0, 15.0)),
        AnalysisDomain("sand", "sand", "sand"),
        AnalysisDomain("whole", "all", "all"),
    ]
    scales = mw.distance_scales(1.5, injection, co2, depth, [0.03, 0.045], regions)
    mass = 5400 * 5e-9
    radius = np.sqrt(mass / np.average(co2.solubility_co2) / np.average(depth) / np.pi)
    assert scales.plume_kg_m == pytest.approx(radius * mass, rel=1e-12)
    assert scales.rig_kg_m == pytest.approx(mass * 0.0375, rel=1e-12)
    assert scales.injected_kg["lower"] == pytest.approx(5400 * 2e-9, rel=1e-12)
    assert np.isnan(scales.injected_kg["sand"])
    assert scales.injected_kg["all"] == pytest.approx(mass, rel=1e-12)

    sand, reference = _reference(regions[:2])
    frames = {
        "a": _cells(_plume(18, 6, 2.0), reference, sand),
        "b": _cells(_plume(18, 9, 4.0), reference, sand),
    }
    for r in mw.wasserstein_distances(1.5, frames, reference, scales, SETTINGS):
        assert r.normalized_plume == r.distance_kg_m / scales.plume_kg_m
        assert r.normalized_rig == r.distance_kg_m / scales.rig_kg_m
        mean = 0.5 * (r.mass_1_kg + r.mass_2_kg)
        assert r.normalized_mass == pytest.approx(r.distance_kg_m / mean, rel=1e-6)
        assert r.expected_mass_kg == scales.injected_kg[r.domain_id] or (
            r.domain_id == "sand" and np.isnan(r.expected_mass_kg)
        )
