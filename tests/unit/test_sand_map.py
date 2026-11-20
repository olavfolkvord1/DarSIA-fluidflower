"""Tests of the sand map: a drawing of sand layers placed on a run's grid."""

import json

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.labeling import SandMapConfig
from darsia.presets.workflows.setup.sand_map import (
    SandMapDrawing,
    grid_homographies,
    rasterize_sand_map,
    rig_grid,
)
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image

WIDTH, HEIGHT = 0.4, 0.2
"""Inner frame of the test rig in metres."""

SHAPE = (90, 160)
"""A run's grid, viewed in perspective through PERSPECTIVE."""

PERSPECTIVE = np.array([[0.97, 0.05, 2.0], [-0.04, 1.01, 1.5], [2e-4, -1e-4, 1.0]])
"""A projective distortion of the grid's pixel coordinates."""


def _grid(shape: tuple[int, int], width: float, height: float) -> darsia.Image:
    """An image on a grid spanning the rig from its lower left corner."""
    return darsia.Image(np.zeros(shape, np.uint8), dimensions=[height, width])


def _split_drawing() -> SandMapDrawing:
    """The rig split into two layers from corner to corner along a zigzag of shared
    anchors; the lower layer (2) is the smaller."""
    anchors = [[0, 0], [0.4, 0], [0.4, 0.2], [0, 0.2]]
    anchors += [[0.13, 0.02], [0.17, 0.12], [0.27, 0.09], [0.31, 0.19]]
    return SandMapDrawing(
        np.array(anchors), ((1, (0, 4, 5, 6, 7, 2, 3)), (2, (0, 1, 2, 7, 6, 5, 4)))
    )


def _rig_to_grid() -> np.ndarray:
    """The run's rig geometry: rig coordinates to the distorted grid."""
    return PERSPECTIVE @ rig_grid(_grid(SHAPE, WIDTH, HEIGHT))


def _grid_to_rig() -> np.ndarray:
    """The run's measured model of its grid: the distorted grid to rig coordinates."""
    return np.linalg.inv(_rig_to_grid())


def _rectangle(anchors: list, x0: float, x1: float, y0: float, y1: float) -> tuple:
    """Append a rectangle's corners to the anchors and return their indices."""
    anchors += [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return tuple(range(len(anchors) - 4, len(anchors)))


def test_overlaps_frame_and_edge_follow_the_drawing_rules():
    """The smaller polygon wins, layers on top win, equal areas keep the drawing
    order, nothing lies outside the frame and the edge fill reaches the frame."""
    anchors = []
    polygons = [
        (2, _rectangle(anchors, 0.006, 0.45, -0.01, 0.1)),
        (5, _rectangle(anchors, 0.006, 0.45, 0.1, 0.21)),
        (3, _rectangle(anchors, 0.08, 0.14, 0.02, 0.06)),
        (4, _rectangle(anchors, 0.12, 0.30, 0.04, 0.16)),
        (6, _rectangle(anchors, 0.3125, 0.34375, 0.125, 0.15625)),
        (7, _rectangle(anchors, 0.328125, 0.359375, 0.125, 0.15625)),
    ]
    drawing = SandMapDrawing(np.array(anchors), tuple(polygons), frozenset({4}))
    # A grid of 2 mm pixels over 0.5 x 0.25 m; the frame is its lower left part.
    shape = (125, 250)
    rig_to_grid = rig_grid(_grid(shape, 0.5, 0.25))

    def at(labels, x, y):
        return labels[int((0.25 - y) / 0.002), int(x / 0.002)]

    labels, inside = rasterize_sand_map(
        drawing, shape, rig_to_grid, width=WIDTH, height=HEIGHT, edge_fill_mm=8.0
    )
    assert at(labels, 0.11, 0.04) == 3
    assert at(labels, 0.13, 0.05) == 4 and at(labels, 0.2, 0.1) == 4
    assert at(labels, 0.32, 0.14) == 6 and at(labels, 0.336, 0.14) == 7
    assert not labels[~inside].any() and not inside[:, 201:].any()
    assert not inside[:24].any() and inside[26:, :199].all()
    assert at(labels, 0.001, 0.05) == 2 and at(labels, 0.003, 0.15) == 5
    assert np.all(labels[inside] > 0)

    unfilled, _ = rasterize_sand_map(
        drawing, shape, rig_to_grid, width=WIDTH, height=HEIGHT, edge_fill_mm=0.0
    )
    assert at(unfilled, 0.001, 0.05) == 0 and at(unfilled, 0.003, 0.15) == 0


def test_a_split_at_shared_anchors_leaves_no_gap_or_overlap():
    """Two layers on shared anchors cover exactly what the whole rig covers, and
    the pixels both polygons reach belong to the smaller one."""
    drawing = _split_drawing()
    whole = SandMapDrawing(drawing.anchors, ((1, (0, 1, 2, 3)),))
    rasters = [
        rasterize_sand_map(
            d, SHAPE, _rig_to_grid(), width=WIDTH, height=HEIGHT, edge_fill_mm=0.0
        )[0]
        for d in (
            drawing,
            whole,
            SandMapDrawing(drawing.anchors, drawing.polygons[:1]),
            SandMapDrawing(drawing.anchors, drawing.polygons[1:]),
        )
    ]
    split, covered, upper, lower = rasters
    assert np.array_equal(split > 0, covered > 0)
    both = (upper > 0) & (lower > 0)
    assert both.any() and np.all(split[both] == 2)
    assert np.array_equal(split == 1, (upper > 0) & ~both)


def _setup(tmp_path, name: str, adjustment: dict | None = None):
    """Set up one run's labels by the labeling step of the setup; return them."""
    folder = tmp_path / name
    folder.mkdir()
    sand_map = {
        "drawing": (tmp_path / "drawing.json").as_posix(),
        "grid_to_rig": _grid_to_rig().tolist(),
        "edge_fill_mm": 5.0,
    }
    if adjustment is not None:
        (folder / "adjustment.json").write_text(json.dumps({"anchors": adjustment}))
        sand_map["adjustment"] = (folder / "adjustment.json").as_posix()
    table = ", ".join(f"{key} = {json.dumps(value)}" for key, value in sand_map.items())
    # The rig's mapping onto the run's grid gives the grid; its maps are not used.
    zeros = np.zeros(SHAPE, np.float32)
    mapping = darsia.ImageMappingCorrection(
        zeros, zeros, source_shape=SHAPE, dimensions=[HEIGHT, WIDTH]
    )
    mapping.save(folder / "mapping.npz")
    (folder / "config.toml").write_text(
        f'[rig]\nwidth = {WIDTH}\nheight = {HEIGHT}\ndim = 2\npath = "{folder.as_posix()}"\n'
        f'[labeling]\nlabels = "{(folder / "labels.npz").as_posix()}"\n'
        f"sand_map = {{ {table} }}\n"
        f'[corrections.mapping]\npath = "{(folder / "mapping.npz").as_posix()}"\n'
    )
    segment_colored_image(folder / "config.toml")
    return darsia.imread(folder / "labels.npz")


def _write_drawing(tmp_path) -> SandMapDrawing:
    """Write the split drawing as a file and return it as read back."""
    drawing = _split_drawing()
    (tmp_path / "drawing.json").write_text(
        json.dumps(
            {
                "anchors": drawing.anchors.tolist(),
                "polygons": [
                    [layer, list(indices)] for layer, indices in drawing.polygons
                ],
            }
        )
    )
    return SandMapDrawing.read(tmp_path / "drawing.json")


def test_the_saved_sand_map_equals_a_direct_rasterization(tmp_path):
    """The labels saved by the setup are the drawing rasterized on the run's grid,
    placed by the inverse of the run's measured model of its grid."""
    drawing = _write_drawing(tmp_path)
    saved = _setup(tmp_path, "run")
    placed = np.linalg.inv(_grid_to_rig())
    direct, _ = rasterize_sand_map(
        drawing, SHAPE, placed, width=WIDTH, height=HEIGHT, edge_fill_mm=5.0
    )
    assert np.array_equal(saved.img, direct)
    assert saved.dimensions == [HEIGHT, WIDTH]


def test_a_run_grid_is_placed_by_its_measured_model_else_in_rig_metres():
    """A run grid's homographies to and from rig metres are its measured model and
    the model's inverse, else those of a grid in rig metres; the old key, the
    inverse of the model, is refused, as a renamed matrix would place the map by
    the wrong one."""
    grid = _grid(SHAPE, WIDTH, HEIGHT)
    to_rig, to_grid = grid_homographies(None, grid)
    assert np.array_equal(to_grid, rig_grid(grid))
    assert np.array_equal(to_rig, np.linalg.inv(rig_grid(grid)))
    to_rig, to_grid = grid_homographies(_grid_to_rig().tolist(), None)
    assert np.array_equal(to_rig, _grid_to_rig())
    assert np.array_equal(to_grid, np.linalg.inv(_grid_to_rig()))
    table = {"drawing": "drawing.json", "edge_fill_mm": 5.0}
    assert SandMapConfig().load(table).grid_to_rig is None
    with pytest.raises(ValueError, match="replaced by grid_to_rig, its inverse"):
        SandMapConfig().load({**table, "rig_to_grid": _rig_to_grid().tolist()})


def test_a_run_adjustment_changes_only_that_run(tmp_path):
    """Moving a shared anchor for one run moves both layers' boundary in that run
    only; the drawing and the other run keep the common geometry."""
    drawing = _write_drawing(tmp_path)
    text = (tmp_path / "drawing.json").read_text()
    moved = _setup(tmp_path, "settled", {"5": [0.15, 0.15]}).img
    common = _setup(tmp_path, "common").img
    expected = [
        rasterize_sand_map(
            d,
            SHAPE,
            np.linalg.inv(_grid_to_rig()),
            width=WIDTH,
            height=HEIGHT,
            edge_fill_mm=5.0,
        )
        for d in (drawing, drawing.moved({5: [0.15, 0.15]}))
    ]
    assert np.array_equal(common, expected[0][0])
    assert np.array_equal(moved, expected[1][0]) and np.any(moved != common)
    assert np.array_equal(moved > 0, expected[1][1])
    assert (tmp_path / "drawing.json").read_text() == text
