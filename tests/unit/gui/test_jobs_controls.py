"""Local Qt command publication only; no worker, executor, SSH or science."""

import json
from pathlib import Path
import threading

import pytest

from darsia.fluidflower import job_control
from darsia.fluidflower import jobs_panel as ui
from .test_jobs_metadata import bytes_of, job, put
from .test_jobs_panel import app, wait


@pytest.fixture
def controls(app, job):
    root, _cases, _task = job
    launches = []

    def launcher(job_root, **kwargs):
        launches.append((job_root, kwargs))
        return dict(job_root=job_root, numerical_progress_claimed=False)

    panel = ui.JobsPanel(local_machine="Main", local_launcher=launcher)
    panel.set_context(jobs_provider=lambda: [dict(root=str(root), active=True)])
    wait(app, panel.can_close)
    yield panel, launches, root
    wait(app, panel.can_close)
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_open_refresh_only_read_bound_master_and_displayed_chain(controls, app):
    panel, launches, root = controls
    before = bytes_of(root)
    panel.refresh()
    wait(app, panel.can_close)
    assert bytes_of(root) == before and not launches
    assert panel.current["control"]["state"] == {"action": "run", "sequence": 0}
    assert panel.pause_button.isEnabled() and panel.stop_button.isEnabled()
    assert not panel.resume_button.isEnabled()
    assert "Recorded master: Main" in panel.control_state_label.text()
    assert "sequence 0" in panel.control_state_label.text()
    assert not (root / "ui" / "commands").exists()


def test_pause_and_stop_append_requested_states_without_task_or_lease_changes(controls, app):
    panel, launches, root = controls
    original = bytes_of(root)
    panel.pause_job()
    wait(app, panel.can_close)
    first = job_control.read_control(root)
    first_bytes = Path(first["command"]["path"]).read_bytes()
    assert first["action"] == "pause" and first["sequence"] == 1
    assert first["previous"] is None
    assert panel.current["state"] == "Waiting" and panel.current["committed"] == 0
    assert panel.resume_button.isEnabled() and not panel.start_button.isEnabled()
    assert first["command"]["sha256"] in panel.control_state_label.text()
    assert "exit is not inferred" in panel.control_state_label.text()
    panel.stop_job()
    wait(app, panel.can_close)
    last = job_control.read_control(root)
    assert last["action"] == "stop" and last["previous"] == first["command"]
    assert Path(first["command"]["path"]).read_bytes() == first_bytes
    assert all((root / path).read_bytes() == raw for path, raw in original.items())
    assert not launches and not (root / "failed").exists()


def test_resume_publishes_run_then_uses_normal_selected_service_and_original_lease(controls, app):
    panel, launches, root = controls
    panel.pause_job()
    wait(app, panel.can_close)
    pause = job_control.read_control(root)["command"]
    immutable_tasks = {
        path: raw for path, raw in bytes_of(root).items()
        if path.startswith(("queue/", "results/", "leases/", "claimed/"))
    }
    panel.backend_combo.setCurrentIndex(panel.backend_combo.findData("cuda"))
    panel.workers_spin.setValue(3)
    panel.resume_job()
    wait(app, panel.can_close)
    current = job_control.read_control(root)
    assert current["action"] == "run" and current["previous"] == pause
    assert len(launches) == 1
    assert launches[0][0] == str(root)
    assert {
        key: value for key, value in launches[0][1].items()
        if key != "launch_directory"
    } == dict(backend="cuda", workers=3, machine="Main")
    assert ui.CONTROL_LEASE in panel.control_state_label.text()
    assert "600 s actual-progress lease" in panel.control_state_label.text()
    assert all((root / path).read_bytes() == raw for path, raw in immutable_tasks.items())
    assert panel.current["committed"] == 0
    assert "unknown" in panel.launch_state_label.text()
    assert not panel.start_button.isEnabled()  # the original reservation still applies


def test_stale_window_supplies_its_displayed_previous_not_a_fresh_retry(controls, app):
    first, _launches, root = controls
    second = ui.JobsPanel(local_machine="Main")
    errors = []
    second.error_reported.connect(errors.append)
    second.set_context(jobs_provider=lambda: [dict(root=str(root))])
    wait(app, second.can_close)
    try:
        assert second.current["control"]["state"]["sequence"] == 0
        first.pause_job()
        wait(app, first.can_close)
        requested = job_control.read_control(root)["command"]
        second.stop_job()
        wait(app, second.can_close)
        assert errors and "refresh before applying" in errors[-1]
        assert job_control.read_control(root)["command"] == requested
        assert len(list((root / "ui" / "commands").glob("*.json"))) == 1
    finally:
        wait(app, second.can_close)
        second.close()
        second.deleteLater()
        app.processEvents()


def test_non_master_and_legacy_without_master_never_publish(controls, app):
    panel, _launches, root = controls
    before = bytes_of(root)
    panel.local_machine = "Olav"
    panel._target_changed()
    assert not any(
        button.isEnabled()
        for button in (panel.pause_button, panel.stop_button, panel.resume_button)
    )
    panel.pause_job()
    assert "recorded master machine" in panel.error_text.toPlainText()
    assert bytes_of(root) == before
    panel.local_machine = "Main"
    legacy = root.parent / "legacy"
    put(legacy / "job.json", dict(expected_runs=["ac52"]))
    panel.reader = lambda row: dict(
        root=row["root"], name="Legacy", state="Waiting", job_facts={}, errors=[]
    )
    panel.set_context(jobs_provider=lambda: [dict(root=str(legacy))])
    wait(app, panel.can_close)
    legacy_before = bytes_of(legacy)
    panel.stop_job()
    assert bytes_of(legacy) == legacy_before
    assert "recorded master" in panel.control_state_label.text()


def test_control_read_failure_blocks_direct_start_and_command_before_reservation(controls, app):
    panel, launches, root = controls
    panel.pause_job()
    wait(app, panel.can_close)
    command = Path(job_control.read_control(root)["command"]["path"])
    document = json.loads(command.read_bytes())
    document["machine"] = "foreign"
    command.write_bytes(json.dumps(document).encode())
    before = bytes_of(root)
    panel.refresh()
    wait(app, panel.can_close)
    assert panel.current["control"]["error"] is True
    panel.start_local_workers()
    panel.pause_job()
    assert bytes_of(root) == before and not launches
    assert "identity or chain" in panel.error_text.toPlainText()
    assert not (root / "ui" / "launches").exists()


def test_pending_publication_holds_owner_and_navigation_cannot_launch_for_new_context(controls, app):
    panel, launches, root = controls
    panel.pause_job()
    wait(app, panel.can_close)
    entered, release = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return job_control.publish_control(*args, **kwargs)

    panel.control_publisher = delayed
    panel.resume_job()
    wait(app, entered.is_set)
    assert not panel.can_close()
    panel.set_context(jobs_provider=lambda: [])
    release.set()
    wait(app, panel.can_close)
    assert job_control.read_control(root)["action"] == "run"
    assert panel.current is None and not launches


def test_repeated_start_cannot_bypass_pause_or_existing_reservation(controls, app):
    panel, launches, root = controls
    panel.pause_job()
    wait(app, panel.can_close)
    before = bytes_of(root)
    panel.start_selected_machine()
    assert bytes_of(root) == before and not launches
    panel.resume_job()
    wait(app, panel.can_close)
    assert len(launches) == 1
    panel.start_selected_machine()
    assert len(launches) == 1
