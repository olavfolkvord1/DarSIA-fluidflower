"""Recorded observation exclusions are not unknown physical pores or barriers."""

import csv
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from darsia.presets.workflows.analysis.domains import AnalysisDomain
from darsia.presets.workflows.comparison import mass_wasserstein as mw
from darsia.presets.workflows.comparison import steps
from darsia.presets.workflows.comparison.steps import _observed_geometry
from darsia.utils.locations import Frame


def _inputs():
    sand = darsia.Image(np.ones((6, 12), np.uint8), dimensions=[0.006, 0.012])
    values = np.ones(sand.img.shape, np.float32)
    geometry = darsia.run_geometry(
        sand,
        values * 1e-6,
        values * 0.01,
        values * 0.4,
        values * 0.1,
        values.astype(bool),
    )
    labels = np.full(values.shape, PhaseLabel.DISSOLVED, np.uint8)
    rules = np.zeros(values.shape, np.uint8)
    labels[1, 2:6] = PhaseLabel.UNKNOWN
    rules[1, 2:6] = PhaseRule.COLOR_CHECKER
    return sand, geometry, values, labels, rules


def test_chart_exclusion_preserves_other_unknowns_reference_and_other_run():
    sand, geometry, values, labels, rules = _inputs()
    # Four neighbouring unknown pores elsewhere cannot use the 1-3 pixel rule.
    labels[4:6, 8:10] = PhaseLabel.UNKNOWN
    original = geometry.counted.copy()
    observed, count = _observed_geometry(geometry, labels, rules)
    assert count == 4
    assert observed.pore_volume_m3 is geometry.pore_volume_m3
    assert np.array_equal(geometry.counted, original)
    assert observed.counted[4:6, 8:10].all()
    assert not observed.counted[1, 2:6].any()
    settings = mw.WassersteinSettings(
        cell_mm=3.0,
        hole_pixels=3,
        empty_mass_kg=1e-12,
        max_iterations=60,
        distance_tolerance=1e-4,
        increment_tolerance=1e-2,
        bregman_updates=(0, 10, 20),
    )
    domains = [
        AnalysisDomain("sand", "all", "Sand"),
        AnalysisDomain("box", "upper", "Upper", bounds_mm=(0.0, 12.0, 3.0, 6.0)),
    ]
    reference = mw.wasserstein_reference(
        sand, values * 100, values * 0.4, (1,), (), domains, settings
    )
    weight = reference.weights["all"].weight.copy()
    image = darsia.ScalarImage(values, dimensions=sand.dimensions)
    zero = darsia.full_like(image, np.zeros_like(values))
    maps = darsia.SimpleMassAnalysisResults(
        name="observed",
        date=None,
        time=None,
        mass=image,
        mass_g=zero,
        mass_aq=image,
        saturation_g=zero,
        color_signal=zero,
        concentration_aq=image,
    )
    one = mw.frame_cells(
        maps, labels, sand, values.astype(int), observed, reference, settings
    )
    two = mw.frame_cells(
        maps,
        np.full(values.shape, PhaseLabel.DISSOLVED, np.uint8),
        sand,
        values.astype(int),
        geometry,
        reference,
        settings,
    )
    assert one["all"].unknown == 4 and one["upper"].unknown == 0
    assert one["upper"].filled == 0
    assert one["upper"].pixels.sum() == 32 and two["upper"].pixels.sum() == 36
    assert one["upper"].mass.integral() == pytest.approx(
        two["upper"].mass.integral() * 32 / 36
    )
    assert np.array_equal(reference.weights["all"].weight, weight)
    found = mw.wasserstein_distances(
        0.0,
        {"one": one, "two": two},
        reference,
        mw.DistanceScales(1.0, 1.0, {"all": np.nan, "upper": np.nan}),
        settings,
    )
    by_domain = {row.domain_id: row for row in found}
    assert by_domain["all"].status == "unknown"
    assert by_domain["all"].unknown_pixels == (4, 0)
    assert by_domain["upper"].status == "success"
    assert np.isfinite(by_domain["upper"].distance_kg_m)
    assert np.array_equal(labels[1, 2:6], np.full(4, PhaseLabel.UNKNOWN))
    assert np.all(maps.mass.img == 1.0)


def test_recorded_chart_mask_must_match_the_stored_phase_and_grid():
    _, geometry, _, labels, rules = _inputs()
    with pytest.raises(ValueError, match="share their grid"):
        _observed_geometry(geometry, labels, rules[:, :-1])
    with pytest.raises(ValueError, match="share their grid"):
        _observed_geometry(geometry, labels, rules.astype(float))
    labels[1, 2] = PhaseLabel.WATER
    with pytest.raises(ValueError, match="unknown phase"):
        _observed_geometry(geometry, labels, rules)


def test_unrecorded_unknown_pixels_are_not_inferred_as_chart_exclusions():
    _, geometry, _, labels, rules = _inputs()
    rules[:] = PhaseRule.UNCHANGED
    observed, count = _observed_geometry(geometry, labels, rules)
    assert count == 0
    assert np.array_equal(observed.counted, geometry.counted)


def test_distance_adapter_uses_stored_rules_and_publishes_exclusion_counts(
    tmp_path, monkeypatch
):
    sand, geometry, values, labels, rules = _inputs()
    # Empty observed masses exercise the adapter without the solver's separate
    # identical-nonempty stopping-condition case.
    image = darsia.ScalarImage(np.zeros_like(values), dimensions=sand.dimensions)
    zero = darsia.full_like(image, np.zeros_like(values))
    maps = darsia.SimpleMassAnalysisResults(
        name="frame",
        date=None,
        time=None,
        mass=image,
        mass_g=zero,
        mass_aq=image,
        saturation_g=zero,
        color_signal=zero,
        concentration_aq=image,
    )
    rig = SimpleNamespace(
        labels=sand,
        permeability=darsia.full_like(image, values * 100),
        porosity=darsia.full_like(image, values * 0.4),
        depth=image,
        facies=sand,
        baseline=image,
        grid_to_rig=None,
    )
    grid = SimpleNamespace(
        rig=rig, geometry=geometry, facies=values.astype(int), positions=None
    )
    experiment = SimpleNamespace(
        injection_protocol=None,
        get_datetime=lambda _path: datetime(2020, 1, 1),
        time_since_start=lambda _date: 0.0,
    )
    monkeypatch.setattr(steps, "run_grid", lambda *_args: grid)
    monkeypatch.setattr(steps, "run_settings", lambda *_args: None)
    monkeypatch.setattr(steps, "make_experiment", lambda *_args: experiment)
    monkeypatch.setattr(steps, "start_co2", lambda *_args: None)
    monkeypatch.setattr(
        steps,
        "distance_scales",
        lambda *_args: mw.DistanceScales(1.0, 1.0, {"sand": np.nan}),
    )
    frame = Frame("session", "frame.png")
    captured = {}

    def read(kind, reader, **_where):
        if kind == "rig":
            return (None, None) if reader is steps.read_grid_model else rig
        if kind == "mass-maps":
            return maps
        assert kind == "phase-maps"
        return dict(labels=labels, rules=rules)

    def publish(writer, **metadata):
        writer(tmp_path)
        captured.update(metadata)

    ctx = SimpleNamespace(
        settings=dict(
            wasserstein=mw.WassersteinSettings(
                3.0, 3, 1e-12, 60, 1e-4, 1e-2, (0, 10, 20)
            ),
            distances=steps.DistanceSettings((0.0,), 0.1, "a"),
            material=None,
            domains=SimpleNamespace(
                sand_labels=(1,), domains={"sand": {"kind": "sand", "name": "Sand"}}
            ),
        ),
        selection=SimpleNamespace(
            frames={"a": (frame,), "b": (frame,)}, domains=("sand",)
        ),
        task=SimpleNamespace(part="0.0h"),
        read=read,
        publish=publish,
        photo=lambda _run, _frame: tmp_path / "frame.png",
    )
    steps.DISTANCES.run(ctx)
    with (tmp_path / "distances.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "success" and float(row["distance_kg_m"]) == 0.0
    assert row["case_id_1"] == row["case_id_2"] == "session/frame.png"
    assert captured["measurements"] == {
        "colorchecker_excluded_counted_pixels": {"a": 4, "b": 4}
    }
    assert np.all(labels[1, 2:6] == PhaseLabel.UNKNOWN)
    assert geometry.counted.all()
