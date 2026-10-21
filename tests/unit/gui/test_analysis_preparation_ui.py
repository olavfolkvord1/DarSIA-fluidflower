"""Offscreen plan preparation; no production data, adoption, queue or worker."""
from pathlib import Path
import threading

from darsia.fluidflower import analysis_preparation as preparation
from darsia.fluidflower.analysis_preparation_ui import AnalysisPreparationDialog
from darsia.fluidflower.jobs_panel import JobsPanel
from darsia.fluidflower.pipeline.job_queue import descriptor


def test_opening_and_changing_stage_start_nothing_and_preparation_runs_off_ui(qtbot, tmp_path, monkeypatch):
    panel = JobsPanel()
    panel.series_root = tmp_path
    qtbot.addWidget(panel)
    dialog = AnalysisPreparationDialog(panel, panel)
    qtbot.addWidget(dialog)
    calls, entered, release = [], threading.Event(), threading.Event()
    gui_thread = threading.get_ident()
    monkeypatch.setattr(preparation, "job_identity", lambda repository: dict(commit="a" * 40, precision="float32"))
    def prepare(inputs, folder, **kwargs):
        assert threading.get_ident() != gui_thread
        calls.append((inputs, folder, kwargs))
        entered.set()
        assert release.wait(30)
        target = Path(folder)
        target.mkdir()
        (target / "execution.toml").write_text("synthetic-inert-plan", encoding="utf-8")
        return dict(plan=descriptor(target / "execution.toml"), adopted_for_execution=False, worker_started=False)
    monkeypatch.setattr(preparation, "prepare_phase1_inputs", prepare)
    dialog.show()
    dialog.stage.setCurrentIndex(1)
    assert not dialog.probability.isEnabled() and dialog.calibration.isEnabled()
    assert not dialog._file_buttons[dialog.geometry].isEnabled()
    dialog.stage.setCurrentIndex(0)
    assert not calls and dialog._thread is None
    for edit, name in [(dialog.inputs, "inputs"), (dialog.geometry, "geometry"), (dialog.pressure, "pressure")]:
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        edit.setText(str(path))
    dialog.folder.setText(str(tmp_path / "prepared"))
    dialog.prepare_button.click()
    qtbot.waitUntil(entered.is_set, timeout=30000)
    dialog.reject()
    assert dialog.isVisible()
    release.set()
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=30000)
    assert len(calls) == 1 and "Prepared, not adopted" in dialog.status.text()
    assert dialog.adopt_button.isEnabled() and not (tmp_path / "job.json").exists()
    dialog.close()
