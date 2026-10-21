"""Existing Jobs registration after finite Setup metadata preparation only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from darsia.fluidflower import native_jobs_registry as registry
from darsia.fluidflower.native_jobs_workspace import NativeJobsWorkspace
from darsia.fluidflower.series_setup_jobs_ui import SeriesSetupJobsController


class _Window(QWidget):
    """Host only the existing Jobs owner and record navigation requests."""

    def __init__(self):
        super().__init__()
        self.shown = 0

    def show_jobs(self):
        self.shown += 1


class _Workspace(QWidget):
    """Expose the geometry owner's current loaded-series contract."""

    reference_changed = Signal(str)

    def __init__(self, window, root):
        super().__init__(window)
        self.project = SimpleNamespace(root=root / "geometry")
        self._loaded_root = root
        self.series_id = self._loaded_series = "campaign"


class _Status(QObject):
    """Offer the existing saved-status change signal without a painting view."""

    changed = Signal()


class _View(QWidget):
    """Expose only the public button, reason and busy contracts."""

    busy_changed = Signal(bool)

    def __init__(self, workspace):
        super().__init__(workspace)
        self.apply_build_button = QPushButton("Apply to all runs and Build run models")
        self.apply_build_button.setEnabled(False)
        self.apply_build_reason = QLabel("The current backend is unavailable")
        self.series_use_status = _Status(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.apply_build_button)
        layout.addWidget(self.apply_build_reason)

    def busy(self):
        return False


def _snapshot(row):
    """Read synthetic job display facts without resolving any scientific input."""
    return dict(root=row["root"], name="Setup metadata", series="campaign", state="Waiting",
        job_facts=dict(precision="float32"), run_count=1, completed_runs=0,
        committed=0, total_frames=None, remaining=None, workers=[], machines=[],
        registered_machines=[], logs=[], errors=[], runs=[], launch_requests=[])


@pytest.fixture
def owner(tmp_path, monkeypatch, qtbot):
    """Use the real Jobs owner/registry with temporary paths and forbidden launchers."""
    monkeypatch.setattr(registry, "require_unc", lambda value: None)
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text(
        'schema_version=1\nseries_id="campaign"\n[runs]\nids=["run-a"]\n', encoding="utf-8")
    window = _Window()
    qtbot.addWidget(window)
    workspace = _Workspace(window, root)
    view = _View(workspace)

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata preparation must not start a worker or transport")

    jobs = NativeJobsWorkspace(window, registry=registry.JobsUIRegistry(tmp_path / "registry"),
        local_machine="fixture", reader=_snapshot, local_launcher=forbidden,
        remote_launcher=forbidden, job_creator=forbidden)
    window.jobs_workspace = jobs
    jobs.set_series("campaign", root)
    qtbot.waitUntil(jobs.can_close, timeout=10000)
    context = dict(project=workspace.project, series_root=root, target_machine="fixture",
        job_root=tmp_path / "shared-job", commit="a" * 40,
        environment_lock=dict(path="requirements-worker.lock", sha256="b" * 64))
    yield SimpleNamespace(window=window, workspace=workspace, view=view, jobs=jobs, context=context)
    qtbot.waitUntil(lambda: jobs.can_close() and not view.busy(), timeout=10000)


def test_explicit_setup_action_prepares_off_thread_then_registers_and_opens_jobs(owner, qtbot, qapp):
    """The existing registry receives the prepared job; no launcher is invoked."""
    calls, busy = [], []
    owner.workspace.series_id = "display-alias"

    def prepare(project, series_root, **choices):
        calls.append((project, series_root, choices, QThread.currentThread() is not qapp.thread()))
        root = Path(choices["job_root"])
        root.mkdir()
        (root / "job.json").write_text(json.dumps(dict(expected_runs=["setup-apply"],
            precision="float32")), encoding="utf-8")
        return dict(complete=True, root=str(root))

    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=lambda workspace: dict(owner.context), preparer=prepare)
    owner.view.busy_changed.connect(busy.append)
    assert owner.view.apply_build_button.isEnabled()
    owner.view.apply_build_button.click()
    assert owner.view.busy()
    qtbot.waitUntil(lambda: controller._thread is None and owner.jobs.can_close(), timeout=10000)
    assert len(calls) == 1 and calls[0][3]
    assert calls[0][0] is owner.workspace.project
    assert calls[0][2]["target_machine"] == "fixture"
    assert "stores_path" not in calls[0][2]
    assert busy == [True, False] and owner.window.shown == 1
    registered = owner.jobs.registry.jobs(owner.context["series_root"], "campaign")
    assert [row["root"] for row in registered] == [str(owner.context["job_root"])]
    assert owner.view.apply_build_reason.text() == (
        "Setup job metadata created. Start its workers in Jobs to run Apply and Build.")


def test_setup_preparation_failure_stays_literal_and_keeps_original_traceback(owner, qtbot):
    """A failed metadata writer neither registers a job nor navigates to success."""
    def prepare(*args, **kwargs):
        raise FileExistsError("The selected Setup output already exists")

    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=lambda workspace: dict(owner.context), preparer=prepare)
    owner.view.apply_build_button.click()
    qtbot.waitUntil(lambda: controller._thread is None, timeout=10000)
    assert owner.view.apply_build_reason.text() == "FileExistsError: The selected Setup output already exists"
    assert "Traceback" in owner.jobs.panel.error_text.toPlainText()
    assert owner.window.shown == 0
    assert owner.jobs.registry.jobs(owner.context["series_root"], "campaign") == []


def test_absent_context_preserves_disabled_state_and_missing_actual_folder_is_explained(owner):
    """No folder is inferred from old jobs or from unrelated local store roots."""
    SeriesSetupJobsController(owner.workspace, owner.view, context_provider=lambda workspace: None)
    assert not owner.view.apply_build_button.isEnabled()
    assert owner.view.apply_build_reason.text() == "The current backend is unavailable"
    root = owner.workspace.project.root / "overrides" / "geometry_facies"
    root.mkdir(parents=True)
    (root / "current.json").write_text(json.dumps(dict(
        format="darsia-fluidflower-geometry-facies-v1", schema=1, revision_id="a" * 32,
        configuration=dict(frame="rig-corrected"))), encoding="utf-8")
    second = _View(owner.workspace)
    SeriesSetupJobsController(owner.workspace, second)
    assert not second.apply_build_button.isEnabled()
    assert second.apply_build_reason.text() == "Jobs has no fresh shared job folder recorded"
    assert owner.window.shown == 0


def test_existing_store_map_path_is_forwarded_only_when_captured(owner, qtbot):
    """The planner receives the actual Jobs store-map locator without a new root."""
    context = dict(owner.context, stores_path=owner.jobs.stores.path)
    captured = []

    def prepare(project, series_root, **choices):
        captured.append(choices)
        raise ValueError("Synthetic stop after the existing store-map locator was received")

    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=lambda workspace: context, preparer=prepare)
    owner.view.apply_build_button.click()
    qtbot.waitUntil(lambda: controller._thread is None, timeout=10000)
    assert len(captured) == 1 and captured[0]["stores_path"] == owner.jobs.stores.path
    assert captured[0]["job_root"] == owner.context["job_root"]
    assert owner.jobs.registry.jobs(owner.context["series_root"], "campaign") == []


def test_existing_jobs_choices_refresh_setup_readiness_without_source_identity_checks(owner):
    """Editing existing Jobs choices updates the action without running Git on typing."""
    root = owner.workspace.project.root / "overrides" / "geometry_facies"
    root.mkdir(parents=True)
    (root / "current.json").write_text(json.dumps(dict(
        format="darsia-fluidflower-geometry-facies-v1", schema=1, revision_id="a" * 32,
        configuration=dict(frame="rig-corrected"))), encoding="utf-8")
    controller = SeriesSetupJobsController(owner.workspace, owner.view)
    assert not owner.view.apply_build_button.isEnabled()
    owner.jobs.new_analysis.form.job_edit.setText(r"\\fixture\shared\fresh-setup-job")
    assert owner.view.apply_build_button.isEnabled()
    assert "commit" not in controller._context and "environment_lock" not in controller._context
    owner.jobs.panel.backend_combo.setCurrentIndex(owner.jobs.panel.backend_combo.findData("cuda"))
    assert not owner.view.apply_build_button.isEnabled()
    assert owner.view.apply_build_reason.text() == "The selected Jobs backend must be CPU for Setup"
    owner.jobs.panel.backend_combo.setCurrentIndex(owner.jobs.panel.backend_combo.findData("cpu"))
    assert owner.view.apply_build_button.isEnabled()
    owner.jobs.new_analysis.form.job_edit.clear()
    assert not owner.view.apply_build_button.isEnabled()
    assert owner.view.apply_build_reason.text() == "Jobs has no fresh shared job folder recorded"
    assert owner.window.shown == 0 and controller._thread is None


def test_new_saved_revision_clears_previous_outcomes_at_the_same_root(owner):
    """Revision metadata changes retire old success and error feedback without a worker."""
    root = owner.workspace.project.root / "overrides" / "geometry_facies"
    root.mkdir(parents=True)
    current = root / "current.json"

    def revision(identifier):
        current.write_text(json.dumps(dict(format="darsia-fluidflower-geometry-facies-v1",
            schema=1, revision_id=identifier, configuration=dict(frame="rig-corrected"))), encoding="utf-8")

    revision("a" * 32)
    owner.jobs.new_analysis.form.job_edit.setText(r"\\fixture\shared\fresh-setup-job")
    controller = SeriesSetupJobsController(owner.workspace, owner.view)
    for message, identifier in (("Previous metadata creation succeeded", "b" * 32),
                                ("Previous metadata creation failed", "c" * 32)):
        controller._outcome = message
        controller.refresh()
        assert owner.view.apply_build_reason.text() == message
        revision(identifier)
        owner.view.series_use_status.changed.emit()
        assert owner.view.apply_build_button.isEnabled()
        assert controller._context["adopted_revision_id"] == identifier
        assert message not in owner.view.apply_build_reason.text()
        assert "commit" not in controller._context and "environment_lock" not in controller._context
    assert owner.window.shown == 0 and controller._thread is None


def test_partial_adopted_scope_is_explicit_before_any_job_worker_start(owner, qtbot):
    """The planner's actual selected and remaining lists stay visible in Jobs."""
    captured = []

    def prepare(project, series_root, **choices):
        captured.append(choices)
        root = Path(choices["job_root"])
        root.mkdir()
        (root / "job.json").write_text(json.dumps(dict(expected_runs=["setup-apply", "run-a"],
            precision="float32")), encoding="utf-8")
        return dict(complete=True, root=str(root), selected_runs=["run-a"], remaining_runs=["run-b", "run-c"])

    context = dict(owner.context, adopted_revision_id="a" * 32)
    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=lambda workspace: context, preparer=prepare)
    owner.view.apply_build_button.click()
    qtbot.waitUntil(lambda: controller._thread is None and owner.jobs.can_close(), timeout=10000)
    assert len(captured) == 1 and "adopted_revision_id" not in captured[0]
    assert owner.view.apply_build_reason.text() == (
        "Setup job metadata created for 1 selected run(s): run-a. Remaining series runs (2): run-b, run-c. "
        "Start its workers in Jobs to run Apply and Build.")
    assert owner.jobs.status_label.text() == owner.view.apply_build_reason.text()
    assert owner.window.shown == 1
