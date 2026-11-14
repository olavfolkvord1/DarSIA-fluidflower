"""Unit tests for the Wasserstein comparison cells of mass maps."""

import tomllib

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.analysis.domains import AnalysisDomain
from darsia.presets.workflows.comparison import mass_wasserstein as mw

SETTINGS = mw.WassersteinSettings(
    **tomllib.loads(
        """
        # The values our production ran with.
        cell_mm = 3.0
        hole_pixels = 3
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
