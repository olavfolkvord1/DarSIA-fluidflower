"""Uniform simulator export before measured-coordinate transport is introduced."""
import csv
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import darsia
from darsia.presets.workflows.export.simulator import MassFrames, MassFrameSource, write_simulator_mass

def test_uniform_native_and_metric_export_preserves_clocks_mass_and_local_concentration(tmp_path):
    canvas = darsia.Image(np.zeros((2, 2)), dimensions=[0.006, 0.004])
    geometry = darsia.run_geometry(canvas, 6e-6, 0.01, 0.4, 0.1, np.ones((2, 2), bool))
    values = np.array([[np.nan, 2.0], [3.0, 4.0]])
    fields = {name: darsia.Image(values.copy(), dimensions=canvas.dimensions) for name in ("mass", "mass_aq", "mass_g", "concentration_aq", "saturation_g")}
    maps = darsia.SimpleMassAnalysisResults(name="source", date=None, time=None, color_signal=None, **fields)
    maps.concentration_aq.img = np.array([[np.nan, 0.2], [0.3, 0.4]])
    labels = np.array([[255, 1], [1, 1]], np.uint8)
    start = datetime(2000, 1, 1)
    sources = {"one": MassFrameSource("a", "same.raw", start + timedelta(seconds=17), start), "two": MassFrameSource("b", "same.raw", start + timedelta(seconds=31), start)}
    mass = MassFrames(geometry, [(name, maps, labels) for name in sources], sources=sources, co2=lambda _: SimpleNamespace(solubility_co2=np.array([[1., 2.], [3., 4.]])))
    assert write_simulator_mass(tmp_path, mass, canvas, [None, 10.]) == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["grids"]["full"]["cell_pitch_m"] == pytest.approx(dict(dy=0.003, dx=0.002))
    with np.load(tmp_path / "full/one.npz") as data:
        assert data["mass_total_g"].sum() == pytest.approx(9 * 6e-6 * .01 * .4 * 1000)
        assert data["mass_total_known_fraction"][1, 0] == 0
        assert data["mass_total_g"][0, 0] == pytest.approx(3 * 6e-6 * .01 * .4 * 1000)
    with np.load(tmp_path / "10mm/one.npz") as data:
        assert data["concentration_aq"][0, 0] == pytest.approx(.3)
        assert data["concentration_aq_g_l"][0, 0] == pytest.approx((.4 + .9 + 1.6) / 3)
    with (tmp_path / "index.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["source_session"] for row in rows] == ["a", "b"]
    assert [float(row["elapsed_s"]) for row in rows] == [17., 31.]
    with pytest.raises(FileExistsError):
        write_simulator_mass(tmp_path, mass, canvas, [None])
