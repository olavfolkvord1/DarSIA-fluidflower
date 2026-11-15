"""Tests of the simulator export: the sand map filled to the frame, the completed
geometry and its cells."""

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.export.simulator import (
    CompletedGeometry,
    SimulatorExportSettings,
    complete_geometry,
    geometry_cells,
    sand_layers,
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
)
"""Export settings of the tests: our production's cell size and precision, and the
facies of the test rig."""


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
