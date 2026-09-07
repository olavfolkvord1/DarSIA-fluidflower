"""Synthetic pipeline tests; never open or alter experimental evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.phase_training.radiometry import RunRadiometry, resolve_profile
from darsia.fluidflower.phase_training.trainer import StagedRun, _feature_cache_key
from darsia.fluidflower.phase_workflow import (
    prepare_phase_training_request,
    build_phase_training_command,
)
from darsia.fluidflower.phase_validation import materialize_phase_validation_campaign
from .test_fluidflower_phase_validation import _write_model, _write_source_campaign


def _mock_sampler(monkeypatch, reference):
    calls = []

    def sample(rgb, **kwargs):
        calls.append(kwargs)
        return np.asarray(reference), np.array([[0, 0], [10, 0], [10, 10], [0, 10]])

    monkeypatch.setattr(
        "darsia.fluidflower.image_correction._reference_colorchecker", sample
    )
    return calls


def test_profile_snapshot_in_training_request(tmp_path):
    decisions, manifest = tmp_path / "decisions.json", tmp_path / "manifest.json"
    decisions.write_text('{"decisions": {}}')
    manifest.write_text('{"jobs": []}')
    request = prepare_phase_training_request(
        output_parent=tmp_path / "models",
        cache_parent=tmp_path / "cache",
        bundles=[(decisions, manifest)],
        actor="Synthetic reviewer",
        settings={"radiometry_profile": "uniform24-after2014", "series": ["FF_AC"]},
    )
    command = build_phase_training_command(request)
    frozen = Path(command[command.index("--radiometry-profile") + 1])
    assert frozen.parent.name == "input_snapshot"
    assert json.loads(frozen.read_text())["payload"]["indicator_weighting"] == "none"
    assert command[command.index("--series") + 1] == "FF_AC"
    assert decisions.read_text() == '{"decisions": {}}'


def test_run_receipts_cache_and_corruption(tmp_path, monkeypatch):
    root = _write_source_campaign(tmp_path / "source")
    baseline = root / "manual_pairwise_review/native_references/ac52/pre_injection.jpg"
    profile = resolve_profile("uniform24-after2014")
    calls = _mock_sampler(monkeypatch, profile["payload"]["target_swatches"])
    kwargs = dict(
        series_id="ffac",
        run_id="ac52",
        baseline_path=baseline,
        content_box=(0, 0, 96, 72),
        cache_root=tmp_path / "cache",
    )
    prepared = RunRadiometry(profile, **kwargs)
    first = prepared.load(baseline, "baseline")
    before = len(calls)
    second = RunRadiometry(profile, **kwargs).load(baseline, "baseline")
    assert len(calls) == before
    np.testing.assert_array_equal(first, second)
    row = prepared.report()["inventory"][0]
    assert (row["series_id"], row["run_id"]) == ("ffac", "ac52")
    assert len(row["frames"]) == 1
    cache = next((tmp_path / "cache").glob("*.json"))
    receipt = json.loads(cache.read_text())
    receipt["payload"]["run_id"] = "run16"
    cache.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="checksum"):
        RunRadiometry(profile, **kwargs).load(baseline, "baseline")


def test_feature_cache_cannot_reuse_legacy_or_other_target(tmp_path):
    staged = StagedRun(
        "ac52",
        "ffac",
        tmp_path,
        tmp_path / "manifest.json",
        tmp_path / "base.jpg",
        (),
        "assets-hash",
    )
    profile = resolve_profile("uniform24-after2014")
    kwargs = dict(pixels_per_region=64, texture_sigma=2.0)
    assert _feature_cache_key(staged, [], **kwargs) != _feature_cache_key(
        staged, [], radiometry_profile=profile, **kwargs
    )


def test_selected_model_prepares_images_and_retains_source_mapping(
    tmp_path, monkeypatch
):
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel

    model_root = _write_model(tmp_path / "model")
    profile = resolve_profile("uniform24-after2014")
    for path in (model_root / "models/generic").glob("*.json"):
        payload = json.loads(path.read_text())
        payload["training_provenance"]["radiometry_profile"] = profile
        path.write_text(json.dumps(payload))
    source = _write_source_campaign(tmp_path / "source")
    files = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    _mock_sampler(monkeypatch, profile["payload"]["target_swatches"])
    observed_profiles = []
    original = HierarchicalPhaseModel.predict_image

    def checked(self, *args, **kwargs):
        observed_profiles.append(kwargs.get("radiometry_profile"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HierarchicalPhaseModel, "predict_image", checked)
    output = tmp_path / "validation"
    result = materialize_phase_validation_campaign(model_root, source, output)
    assert result["radiometry_profile"] == profile
    assert observed_profiles and all(p == profile for p in observed_profiles)
    assert (output / "review_assets/native_frames/ac52/frame_000.png").is_file()
    report = json.loads(
        (output / "review_assets/radiometry/ffac/ac52.json").read_text()
    )
    assert (
        len(report["receipts"]) == 3
    )  # Baseline and two frames; previous reuses the prepared frame.
    manifest = json.loads((output / "postflight_manifest.json").read_text())
    assert manifest["jobs"][0]["analysis_source_review_root"] == str(
        source / "manual_pairwise_review"
    )
    assert all(p.read_bytes() == content for p, content in files.items())
    assert materialize_phase_validation_campaign(model_root, source, output) == result
    loaded = HierarchicalPhaseModel.load(model_root)
    with pytest.raises(ValueError, match="bound radiometric"):
        loaded.predict_image(np.zeros((5, 5, 3)), np.zeros((5, 5, 3)))


def test_studio_profile_controls_and_screenshot(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    from darsia.fluidflower.ui.phase_model_studio import PhaseModelStudioWindow

    app = QApplication.instance() or QApplication([])
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")
    monkeypatch.setattr(
        "darsia.fluidflower.ui.model_training.discover_phase_review_sessions", list
    )
    monkeypatch.setattr(
        "darsia.fluidflower.ui.model_training.discover_phase_model_history",
        lambda root: [],
    )
    window = PhaseModelStudioWindow()
    page = window.training_page
    page.tabs.setCurrentIndex(1)
    assert page.radiometry_mode.currentData() == "image"
    assert page.radiometry_profile_path.isEnabled()
    page.radiometry_mode.setCurrentIndex(1)
    assert not page.radiometry_profile_path.isEnabled()
    page.radiometry_mode.setCurrentIndex(0)
    window.resize(1400, 980)
    window.show()
    app.processEvents()
    assert window.grab().save(str(tmp_path / "studio-training.png"))
    window.close()
    window.deleteLater()
    app.processEvents()
