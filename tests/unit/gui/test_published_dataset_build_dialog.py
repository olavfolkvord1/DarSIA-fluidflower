"""Build published dataset: the Results and export action and its dialog (offscreen Qt)."""

import pytest

from darsia.fluidflower.published_dataset import builder, scan
from darsia.fluidflower.published_dataset.build_dialog import BuildDatasetDialog
from darsia.fluidflower.published_dataset.manifest import load_manifest
from tests.unit.published_dataset.synthetic import make_series


@pytest.fixture(autouse=True)
def fictional_machine(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "machine_names", lambda: (
        scan.ScanName("example-build-host", "build machine host name"),))
    monkeypatch.setenv("DARSIA_PUBLICATION_SETTINGS", str(tmp_path / "user-settings" / "publication.toml"))


def _dialog(qtbot, series, output):
    dialog = BuildDatasetDialog(series.series)
    qtbot.addWidget(dialog)
    dialog.job_edit.setText(str(series.job))
    dialog.output_edit.setText(str(output))
    return dialog


def test_dialog_checks_then_builds_in_a_worker_thread(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "reference_machine", lambda request, job, report: (
        {"label": "reference machine", "backend": "cpu", "precision": "float32", "threads": 2,
         "seconds_per_frame": 6.0, "peak_memory_bytes": request.peak_memory_bytes,
         "peak_memory_basis": request.peak_memory_basis}, None))
    series = make_series(tmp_path)
    dialog = _dialog(qtbot, series, tmp_path / "record")
    dialog.memory_edit.setText("1.5")
    dialog.memory_basis_edit.setText("measured committed memory")
    assert dialog.request().peak_memory_bytes == int(1.5 * 1024**3)
    assert dialog.start(check_only=True)
    assert not dialog.start(check_only=True)  # One run at a time.
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=20000)
    assert dialog.report_view.toPlainText().startswith("Checked: nothing refuses the record.")
    assert not (tmp_path / "record").exists()
    assert dialog.start(check_only=False)
    qtbot.waitUntil(lambda: not dialog.busy(), timeout=20000)
    assert dialog.report_view.toPlainText().startswith("Record built.")
    assert dialog.last_report.built and load_manifest(tmp_path / "record")["machine"]["peak_memory_bytes"] == int(
        1.5 * 1024**3)
    assert dialog.check_button.isEnabled() and dialog.build_button.isEnabled()


def test_dialog_explains_missing_choices_and_simulator_resolution(qtbot, tmp_path):
    dialog = BuildDatasetDialog(tmp_path)
    qtbot.addWidget(dialog)
    assert not dialog.resolution_edit.isEnabled()
    assert not dialog.start(check_only=True)
    assert dialog.report_view.toPlainText() == "Choose the production job and an output folder"
    dialog.job_edit.setText(str(tmp_path / "job"))
    dialog.output_edit.setText(str(tmp_path / "out"))
    dialog.record_combo.setCurrentIndex(dialog.record_combo.findData("S"))
    assert dialog.resolution_edit.isEnabled()
    assert not dialog.start(check_only=False)
    assert "no default" in dialog.report_view.toPlainText()
    dialog.resolution_edit.setText("1")
    request = dialog.request()
    assert (request.record, request.simulator_resolution_mm, request.proof) == ("S", 1.0, None)


def test_results_and_export_offers_the_action_only_in_its_view(qtbot, tmp_path, monkeypatch):
    from darsia.fluidflower.tools import phase_studio_mass_analysis_workspace as module

    root = tmp_path / "series-a"
    root.mkdir()
    monkeypatch.setattr(module, "configured_series", lambda: (("series-a", "series-a", root),))
    page = module.MassAnalysisWorkspace(series="series-a", project_root=root)
    qtbot.addWidget(page)
    page.show()
    page.show_view("review")
    assert not page.publish_button.isVisible()
    page.show_view("results")
    assert page.publish_button.isVisible() and page.publish_button.text() == "Build published dataset…"
    page.publish_button.click()
    dialog = page.publish_dialog
    assert isinstance(dialog, BuildDatasetDialog) and dialog.series_root == root and dialog.isVisible()
    page.build_published_dataset()
    assert page.publish_dialog is dialog
    other = tmp_path / "series-b"
    other.mkdir()
    page.project_root = other
    page.build_published_dataset()
    assert page.publish_dialog is not dialog and page.publish_dialog.series_root == other
    page.publish_dialog.close()
    page._stop_lookups()
