"""Simulator maps use measured geometry, preserve mass and retain source clocks."""

import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.presets.workflows.export.simulator import (
    CompletedGeometry,
    MassFrames,
    MassFrameSource,
    geometry_cells,
    write_simulator_mass,
)


def example():
    canvas = darsia.Image(np.ones((3, 4)), dimensions=[0.009, 0.008])
    source = darsia.Image(np.zeros((2, 2)), dimensions=[0.009, 0.008])
    positions = darsia.GridPositions(
        np.array([[0.001, 0.007], [0.001, 0.007]]),
        np.array([[0.0075, 0.0075], [0.0015, 0.0015]]),
        None,
    )
    geometry = darsia.run_geometry(
        source,
        np.full((2, 2), 2e-6),
        np.full((2, 2), 0.01),
        np.full((2, 2), 0.4),
        np.full((2, 2), 0.1),
        np.ones((2, 2), bool),
        positions=positions,
    )
    values = np.array([[np.nan, 2.0], [3.0, 4.0]])
    # The nominal origin is deliberately different: measured positions own placement.
    fields = {
        name: darsia.Image(
            values.copy(), dimensions=[0.009, 0.008], origin=[0.02, 0.04]
        )
        for name in ("mass", "mass_aq", "mass_g", "concentration_aq", "saturation_g")
    }
    maps = darsia.SimpleMassAnalysisResults(
        name="source", date=None, time=None, color_signal=None, **fields
    )
    labels = np.array([[255, 1], [1, 1]], np.uint8)
    return canvas, geometry, positions, maps, labels


def test_native_and_metric_maps_share_canvas_preserve_unknown_and_source_time(tmp_path):
    canvas, geometry, positions, maps, labels = example()
    start = datetime(2000, 1, 1)
    sources = {
        "frame_a": MassFrameSource(
            "session_a", "same.raw", start + timedelta(seconds=17), start
        ),
        "frame_b": MassFrameSource(
            "session_b", "same.raw", start + timedelta(seconds=31), start
        ),
    }
    mass = MassFrames(
        geometry, [(name, maps, labels) for name in sources], positions, sources
    )
    folder = tmp_path / "mass"
    assert write_simulator_mass(folder, mass, canvas, [None, 4.0]) == 2
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["grids"]["full"]["shape"] == [3, 4]
    assert manifest["grids"]["full"]["cell_pitch_m"] == pytest.approx(
        dict(dy=0.003, dx=0.002)
    )
    expected = 9.0 * 2e-6 * 0.01 * 0.4 * 1000
    for name in sources:
        for resolution in ("full", "4mm"):
            with np.load(
                folder / resolution / f"{name}.npz", allow_pickle=False
            ) as data:
                assert data["mass_total_g"].sum() == pytest.approx(expected)
        with np.load(folder / "full" / f"{name}.npz", allow_pickle=False) as data:
            # Export rows start at the floor; measured samples go to four corners.
            assert data["mass_total_g"][0, 0] == pytest.approx(expected / 3)
            assert data["mass_total_known_fraction"][2, 0] == 0.0
            assert np.isnan(data["mass_total_known_fraction"][1, 1])
    with (folder / "index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["source_session"] for row in rows] == ["session_a", "session_b"]
    assert [float(row["elapsed_s"]) for row in rows] == [17.0, 31.0]
    with pytest.raises(FileExistsError):
        write_simulator_mass(folder, mass, canvas, [None])


def test_mass_outside_canvas_is_refused_and_duplicate_names_cannot_overwrite(tmp_path):
    canvas, geometry, positions, maps, labels = example()
    outside = replace(positions, x=positions.x + 1.0)
    with pytest.raises(ValueError, match="outside"):
        write_simulator_mass(
            tmp_path / "outside", MassFrames(geometry, [], outside), canvas, [None]
        )
    mass = MassFrames(
        geometry, [("same", maps, labels), ("same", maps, labels)], positions
    )
    with pytest.raises(ValueError, match="Duplicate"):
        write_simulator_mass(tmp_path / "duplicate", mass, canvas, [None])


def test_full_geometry_retains_both_native_pitches_and_volumes():
    shape = (3, 4)
    complete = CompletedGeometry(
        np.ones(shape, int), np.ones(shape, bool), np.zeros(shape, bool), {}
    )
    arrays = geometry_cells(
        complete, np.full(shape, 0.01), {1: 0.4}, [0.009, 0.008], None, "float64"
    )
    assert arrays["pore_volume_m3"].shape == shape
    np.testing.assert_allclose(arrays["domain_area_m2"], 0.003 * 0.002)
    np.testing.assert_allclose(arrays["domain_area_fraction"], 1.0)
    assert arrays["pore_volume_m3"].sum() == pytest.approx(0.009 * 0.008 * 0.01 * 0.4)


def test_explicit_boundary_roundoff_preserves_mass_but_refuses_real_excursions(
    tmp_path,
):
    canvas, geometry, positions, maps, labels = example()
    x = positions.x.copy()
    x[0, 1] = 0.008 + 1e-10
    positions = replace(positions, x=x)
    mass = MassFrames(geometry, [("frame", maps, labels)], positions)
    with pytest.raises(ValueError, match="outside"):
        write_simulator_mass(tmp_path / "strict", mass, canvas, [None])
    folder = tmp_path / "tolerated"
    mass = replace(mass, boundary_tolerance_m=1e-9)
    write_simulator_mass(folder, mass, canvas, [None])
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["boundary_tolerance_m"] == 1e-9
    assert manifest["grids"]["full"]["boundary_assigned_source_pixels"] == 1
    with np.load(folder / "full" / "frame.npz", allow_pickle=False) as data:
        assert data["mass_total_g"].sum() == pytest.approx(9 * 2e-6 * 0.01 * 0.4 * 1000)
    assert positions.x[0, 1] == 0.008 + 1e-10
    x[0, 1] = 0.008 + 1e-6
    with pytest.raises(ValueError, match="outside"):
        write_simulator_mass(tmp_path / "too_far", mass, canvas, [None])



