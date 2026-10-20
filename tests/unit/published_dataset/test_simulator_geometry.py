"""The simulator geometry export (N18) on small rig-corrected canvases with 1 mm cells."""

import io
import json

import numpy as np
import pytest

from darsia.fluidflower.published_dataset import simulator_geometry as geometry

ROLES = {0: "free_water", 1: "porous_medium", 2: "porous_medium", 3: "porous_medium"}
WIDTH, HEIGHT = 0.06, 0.04  # 60 x 40 cells of 1 mm
FIXTURE = {"chart": [(0.020, 0.010), (0.030, 0.010), (0.030, 0.020), (0.020, 0.020)]}
TABLE = [{"facies_id": key, "name": name, "analysis_role": "porous_medium", "porosity": phi, "permeability": 1.0,
          "permeability_unit": "D", "swi": None, "grain_size": None, "source": None}
         for key, name, phi in ((1, "Upper sand", 0.4), (2, "Middle sand", 0.3), (3, "Lower sand", 0.2))]
FLAT_DEPTH = [[-1.0, -1.0, 0.01], [2.0, -1.0, 0.01], [0.5, 2.0, 0.01]]


def layered(*, dip=True):
    """Water above row 5; facies 1, then 2 (a boundary dipping to the right), then 3 from row 30."""
    facies = np.zeros((40, 60), np.int64)
    facies[5:, :] = 1
    for column in range(60):
        facies[15 + (column // 10 if dip else 0):, column] = 2
    facies[30:, :] = 3
    return facies


def test_canvas_coordinates_put_the_origin_at_the_lower_left():
    x, y, area = geometry.canvas_coordinates((40, 60), WIDTH, HEIGHT)
    assert x.shape == y.shape == (40, 60) and area == pytest.approx(1e-6)
    assert x[0, 0] == pytest.approx(0.0005) and x[0, -1] == pytest.approx(0.0595)
    assert y[0, 0] == pytest.approx(0.0395) and y[-1, 0] == pytest.approx(0.0005)
    with pytest.raises(ValueError):
        geometry.canvas_coordinates((0, 60), WIDTH, HEIGHT)


def test_fixture_polygons_in_metres_use_the_production_rasterizer():
    mask = geometry.fixture_mask((40, 60), WIDTH, HEIGHT, FIXTURE)
    rows, columns = np.nonzero(mask)
    assert mask[21:29, 21:29].all() and not mask[:19].any() and not mask[:, :19].any()
    assert 20 <= rows.min() and rows.max() <= 30 and 20 <= columns.min() and columns.max() <= 30
    assert not geometry.fixture_mask((40, 60), WIDTH, HEIGHT, {}).any()
    with pytest.raises(ValueError):
        geometry.fixture_mask((40, 60), WIDTH, HEIGHT, {"line": [(0, 0), (0.01, 0.01)]})


def test_the_domain_runs_from_the_sand_surface_to_the_floor():
    porous = np.zeros((10, 6), bool)
    porous[3:8, :4] = True
    porous[6:8, 5] = True
    water = np.zeros((10, 6), bool)
    water[:3, :3] = True
    frame = np.zeros((10, 6), bool)
    frame[8:, :] = True
    domain = geometry.porous_domain(porous, np.zeros((10, 6), bool), water, frame)
    assert domain[3:8, :3].all() and not domain[:3, :3].any()  # Below the free water, down to the floor.
    assert domain[3:8, 3].all() and not domain[:3, 3].any()  # No water: from its highest sand cell.
    assert domain[3:8, 4].all()  # No water and no sand: the nearest column's surface.
    assert domain[6:8, 5].all() and not domain[:6, 5].any()
    assert not domain[8:].any()  # The frame is outside.


def test_flat_layers_continue_unchanged_across_a_fixture():
    facies = layered(dip=False)
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, fixture_polygons_m=FIXTURE)
    hidden = geometry.fixture_mask((40, 60), WIDTH, HEIGHT, FIXTURE)
    assert np.array_equal(completed.facies, facies)
    assert np.array_equal(completed.fixture_filled, hidden) and not completed.edge_filled.any()
    assert not completed.unfilled.any() and not (completed.original & hidden).any()


def test_a_dipping_boundary_is_interpolated_between_its_two_sides():
    facies = layered()
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, fixture_polygons_m=FIXTURE)
    hidden = completed.fixture_filled
    columns = np.flatnonzero(hidden.any(axis=0))
    left, right = columns.min() - 1, columns.max() + 1
    _x, y, _area = geometry.canvas_coordinates((40, 60), WIDTH, HEIGHT)
    side = {column: 0.5 * (y[15 + column // 10 - 1, column] + y[15 + column // 10, column]) for column in (left, right)}
    for row, column in zip(*np.nonzero(hidden)):
        boundary = side[left] + (column - left) / (right - left) * (side[right] - side[left])
        expected = 3 if row >= 30 else (1 if y[row, column] > boundary else 2)  # 2|3 is flat at row 30.
        assert completed.facies[row, column] == expected, (row, column)
    assert np.array_equal(completed.facies[~hidden], facies[~hidden])  # Original cells never change.


def test_a_boundary_seen_on_one_side_only_keeps_its_height():
    facies = layered(dip=False)
    facies[25:30, 40:] = 4  # A lens on the right side only: boundaries 2|4 and 4|3 exist there alone.
    hidden = {"chart": [(0.030, 0.005), (0.040, 0.005), (0.040, 0.030), (0.030, 0.030)]}
    completed = geometry.complete_geometry(facies, {**ROLES, 4: "porous_medium"}, width_m=WIDTH, height_m=HEIGHT,
                                           fixture_polygons_m=hidden)
    filled = completed.fixture_filled
    rows = {row: set(completed.facies[row][filled[row]]) for row in range(40) if filled[row].any()}
    assert rows
    for row, values in rows.items():
        expected = 1 if row < 15 else (2 if row < 25 else (4 if row < 30 else 3))
        assert values == {expected}, row


def test_a_fixture_at_the_wall_is_continued_from_its_one_side():
    facies = layered()
    wall = {"chart": [(0.050, 0.010), (0.0605, 0.010), (0.0605, 0.025), (0.050, 0.025)]}
    completed = geometry.complete_geometry(facies, ROLES, width_m=WIDTH, height_m=HEIGHT, fixture_polygons_m=wall)
    filled = completed.fixture_filled
    columns = np.flatnonzero(filled.any(axis=0))
    side = columns.min() - 1
    for row, column in zip(*np.nonzero(filled)):
        assert completed.facies[row, column] == facies[row, side]
    assert columns.max() == 59 and not completed.unfilled.any()


def test_edges_are_replicated_to_the_walls_the_floor_and_the_sand_surface():
    truth = layered(dip=False)
    facies = truth.copy()
    unknown = 9  # A facies id without a role: nothing is known there.
    facies[5:7, :] = unknown  # A strip below the free water (the sand surface).
    facies[37:, :] = unknown  # A strip above the floor.
    facies[5:, :3] = unknown  # A strip at the left wall.
    roles = dict(ROLES)
    completed = geometry.complete_geometry(facies, roles, width_m=WIDTH, height_m=HEIGHT)
    assert not completed.fixture_filled.any() and not completed.unfilled.any()
    assert np.array_equal(completed.edge_filled, facies == unknown)
    assert np.array_equal(completed.facies[5:7, 3:], np.ones((2, 57), np.int64))  # From the sand below.
    assert np.array_equal(completed.facies[37:, 3:], np.full((3, 57), 3))  # From the cell above.
    assert np.array_equal(completed.facies[7:37, :3], truth[7:37, :3])  # From the nearest cell in the row.
    assert set(np.unique(completed.facies[37:, :3])) == {3} and set(np.unique(completed.facies[5:7, :3])) == {1}


def test_a_fixture_role_in_the_map_is_removed_and_filled():
    facies = layered(dip=False)
    facies[20:28, 20:28] = 7
    completed = geometry.complete_geometry(facies, {**ROLES, 7: "colorchecker"}, width_m=WIDTH, height_m=HEIGHT)
    assert completed.fixture_filled.sum() == 64 and np.array_equal(completed.facies, layered(dip=False))


def test_grid_products_report_fractions_porosity_depth_and_volumes():
    facies = layered(dip=False)
    npz, text, counts = geometry.export_geometry(facies, ROLES, TABLE, width_m=WIDTH, height_m=HEIGHT,
                                                 depth_points=FLAT_DEPTH, resolution_mm=2.0,
                                                 fixture_polygons_m=FIXTURE)
    arrays = np.load(io.BytesIO(npz))
    assert arrays["porosity"].shape == (20, 30) and arrays["facies_ids"].tolist() == [1, 2, 3]
    cell = 0.002 ** 2
    # Grid row 0 is the lowest (y increasing): facies 3 there, facies 1 near the top of the sand.
    assert arrays["porosity"][0, 0] == pytest.approx(0.2) and arrays["facies_majority"][0, 0] == 3
    assert arrays["depth_m"][0, 0] == pytest.approx(0.01)
    assert arrays["pore_volume_m3"][0, 0] == pytest.approx(0.2 * 0.01 * cell, rel=1e-5)
    assert arrays["bulk_volume_m3"][0, 0] == pytest.approx(0.01 * cell, rel=1e-5)
    assert arrays["facies_area_fractions"][:, 0, 0].tolist() == [0.0, 0.0, 1.0]
    sand = arrays["domain_area_fraction"] > 0
    assert np.allclose(arrays["facies_area_fractions"].sum(axis=0)[sand], 1.0)
    top = arrays["domain_area_fraction"][-1]  # The water rows lie above the sand: outside the domain.
    assert np.all(top == 0) and np.isnan(arrays["porosity"][-1]).all()
    porosity = arrays["porosity"][sand]
    assert np.all((porosity >= 0.2 - 1e-6) & (porosity <= 0.4 + 1e-6))
    assert arrays["fixture_filled"].any() and not arrays["edge_filled"].any() and not arrays["unfilled"].any()
    assert np.nanmax(arrays["fixture_filled_area_fraction"]) == pytest.approx(1.0)
    document = json.loads(text)
    assert document["format"] == "darsia-simulator-geometry-v1" and document["counts"] == counts
    assert document["grid"]["shape"] == [20, 30] and document["grid"]["resolution_mm"] == 2.0
    assert document["facies_table"] == TABLE and document["depth_points"] == FLAT_DEPTH
    assert document["method"]["fixture_polygons"] == ["chart"]
    assert set(document["arrays"]) == set(arrays.files)


def test_porosity_is_pore_over_bulk_volume_where_depth_varies():
    facies = layered(dip=False)
    sloped = [[0.0, -1.0, 0.01], [0.0, 2.0, 0.01], [0.06, 0.5, 0.03], [0.06, -1.0, 0.03], [0.06, 2.0, 0.03]]
    npz, _text, _counts = geometry.export_geometry(facies, ROLES, TABLE, width_m=WIDTH, height_m=HEIGHT,
                                                   depth_points=sloped, resolution_mm=2.0)
    arrays = np.load(io.BytesIO(npz))
    sand = arrays["domain_area_fraction"] > 0
    ratio = arrays["pore_volume_m3"][sand] / arrays["bulk_volume_m3"][sand]
    assert np.allclose(arrays["porosity"][sand], ratio)
    assert arrays["depth_m"][0, -1] > arrays["depth_m"][0, 0]


def test_the_covering_extent_lies_on_the_storage_lattice():
    assert geometry.covering_extent(0.06, 0.04, 2.0) == (0.0, 0.06, 0.0, 0.04)
    assert geometry.covering_extent(0.061, 0.0405, 2.0) == (0.0, 0.062, 0.0, 0.042)
    assert geometry.covering_extent(0.9, 0.49, 1.0) == (0.0, 0.9, 0.0, 0.49)


def test_the_facies_table_merges_the_optional_columns(tmp_path):
    path = tmp_path / "properties.csv"
    path.write_text("facies_id,name,swi,grain_size,source\n1,Upper,0.1,0.8,sieve\n2,Middle,,coarse,\n",
                    encoding="utf-8")
    extra = geometry.read_property_columns(path)
    assert extra == [{"facies_id": 1, "swi": 0.1, "grain_size": 0.8, "source": "sieve"},
                     {"facies_id": 2, "swi": None, "grain_size": "coarse", "source": None}]
    rows = geometry.facies_table([{"id": 1, "name": "Upper", "porosity": 0.4, "permeability": 10.0,
                                   "analysis_role": "porous_medium", "permeability_unit": "D"},
                                  {"id": 3, "name": "Lower", "porosity": 0.2, "permeability": None,
                                   "analysis_role": "seal", "permeability_unit": "D"}], extra)
    assert rows[0]["swi"] == 0.1 and rows[0]["source"] == "sieve" and rows[1]["swi"] is None
    assert geometry.read_property_columns(tmp_path / "missing.csv") == []


def write_revision(folder, facies, *, run_labels=None):
    """A geometry revision: labels (one label per facies here), regions with roles and a properties table."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "labels.npz", array=facies.astype(np.int32))
    (folder / "facies.csv").write_text("id,swi,grain_size,source\n1,0.1,0.5,sieve\n", encoding="utf-8")
    regions = [{"label_id": key, "facies_id": key, "name": f"Layer {key}", "analysis_role": role,
                "porosity": None if role == "free_water" else 0.4, "permeability": 5.0, "enabled": True}
               for key, role in ROLES.items()]
    configuration = {"format": "darsia-fluidflower-geometry-config-v1", "width_m": WIDTH, "height_m": HEIGHT,
                     "coordinate_convention": "origin_lower_left_x_right_y_up", "grouping_mode": "shared_facies",
                     "labels_path": str(folder / "labels.npz"), "regions": regions, "permeability_unit": "mD"}
    artifacts = {"facies_properties_path": str(folder / "facies.csv"), "run_geometry": {}}
    if run_labels is not None:
        np.savez(folder / "labels_r2.npz", array=run_labels.astype(np.int32))
        artifacts["run_geometry"]["r2"] = {"labels_path": str(folder / "labels_r2.npz")}
    revision = {"format": "darsia-fluidflower-geometry-facies-v1", "revision_id": "rev-1",
                "configuration": configuration, "artifacts": artifacts}
    (folder / "geometry.json").write_text(json.dumps(revision), encoding="utf-8")
    return revision


def test_a_run_reads_its_own_or_the_series_geometry(tmp_path):
    other = layered(dip=False)
    other[20:, :] = 3
    revision = write_revision(tmp_path / "geometry", layered(), run_labels=other)
    facies, roles, table, source = geometry.run_geometry(revision, "r1")
    assert np.array_equal(facies, layered()) and roles[0] == "free_water" and roles[2] == "porous_medium"
    assert source == {"scope": "series_reference", "labels": "labels.npz", "revision": "rev-1"}
    upper = next(row for row in table if row["facies_id"] == 1)
    assert upper["swi"] == 0.1 and upper["source"] == "sieve" and upper["permeability_unit"] == "mD"
    facies, _roles, _table, source = geometry.run_geometry(revision, "R2")
    assert np.array_equal(facies, other) and source["scope"] == "run_override" and source["labels"] == "labels_r2.npz"
    with pytest.raises(ValueError, match="rig-corrected"):
        geometry.run_geometry(dict(revision, configuration=dict(revision["configuration"],
                                                                coordinate_convention="pixels")), "r1")


def test_depth_points_come_from_the_request_or_the_run_configuration(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "depth.csv").write_text("x,y,depth\n0,0,0.01\n1,0,0.02\n0,1,0.03\n", encoding="utf-8")
    inline = geometry.depth_points_for({"depth_points": FLAT_DEPTH}, tmp_path, {})
    assert inline.tolist() == FLAT_DEPTH
    from_file = geometry.depth_points_for({"depth_points": "config/depth.csv"}, tmp_path, {})
    assert from_file.tolist() == [[0, 0, 0.01], [1, 0, 0.02], [0, 1, 0.03]]
    config = {"mass": {"template": {"shared": {"depth_points": {"path": "config/depth.csv", "sha256": "0" * 64}}}}}
    assert geometry.depth_points_for({}, tmp_path, config).shape == (3, 3)
    with pytest.raises(ValueError, match="No measured depth points"):
        geometry.depth_points_for({}, tmp_path, {})
    with pytest.raises(ValueError):
        geometry.depth_points_for({"depth_points": "../outside.csv"}, tmp_path, {})


def test_the_request_names_its_geometry_and_known_fields_only(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"format": geometry.REQUEST_FORMAT, "geometry": "geometry/geometry.json",
                                "fixtures_m": FIXTURE}), encoding="utf-8")
    assert geometry.load_request(path)["geometry"] == "geometry/geometry.json"
    path.write_text(json.dumps({"format": geometry.REQUEST_FORMAT, "geometry": "g.json", "colour": 1}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="known fields"):
        geometry.load_request(path)
    path.write_text(json.dumps({"format": "other", "geometry": "g.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="request"):
        geometry.load_request(path)


def test_continue_layers_leaves_a_region_without_sides_unfilled():
    facies = np.ones((6, 4), np.int64)
    hidden = np.zeros((6, 4), bool)
    hidden[:, :] = True
    result, filled = geometry.continue_layers(facies, ~hidden, hidden, np.linspace(0.006, 0.001, 6)[:, None]
                                              * np.ones((1, 4)))
    assert not filled.any() and np.array_equal(result, facies)


def test_replicate_edges_needs_donors_and_targets():
    facies = np.ones((4, 4), np.int64)
    x, y, _area = geometry.canvas_coordinates((4, 4), 0.004, 0.004)
    none = np.zeros((4, 4), bool)
    result, filled = geometry.replicate_edges(facies, none, ~none, x, y)
    assert not filled.any() and np.array_equal(result, facies)
