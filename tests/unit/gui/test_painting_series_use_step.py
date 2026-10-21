"""Step-5 controls and retained metadata, using temporary projects only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QWidget

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as editor
from darsia.fluidflower import painting_series_use_ui as status

from .test_painting_corrections import view  # noqa: F401
from .test_painting_manual_step4_route import _saved_manual


def _record(root, name, version, when, *, preparation=False):
    """Write one private existing record, with no source arrays or job start."""
    folder = root / name
    folder.mkdir(parents=True)
    value = dict(format=status._PREPARATION if preparation else sorted(status._PLACEMENT)[0],
        created_utc=when, map_version=deepcopy(version),
        runs=[dict(run_id="sample", status="prepared" if preparation else "placed")])
    raw = (json.dumps(value) + "\n").encode()
    path = folder / "index.json"
    path.write_bytes(raw)
    return value, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def test_version_actions_are_below_list_and_only_one_adopt_action_is_visible(view):
    view.step_propose.set_expanded(True)
    table = view.versions_table
    layout = table.parentWidget().layout()
    row = next(index for index in range(layout.count())
        if layout.itemAt(index).layout() is not None
        and layout.itemAt(index).layout().indexOf(view.save_version_button) >= 0)
    assert row > layout.indexOf(table)
    assert view.prepare_adoption_button is view.adopt_button
    assert view.step_use_map.content.isAncestorOf(view.use_map_progress)
    assert not view.step_propose.content.isAncestorOf(view.use_map_progress)
    assert view.step_export.isHidden()
    assert not hasattr(view, "setup_pointer")
    assert not view.apply_build_button.isEnabled()
    assert "RAW route-2 backend is not ready" in view.apply_build_reason.text()


def test_adopt_and_old_check_entry_never_open_a_picker_or_legacy_facies_dialog(view, qtbot, monkeypatch):
    assert view.adoption_controller.available
    assert view._series_adopt_action == view.adoption_controller.start
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    def forbidden(*args, **kwargs):
        raise AssertionError("Step 5 must not ask for a path or open the legacy adoption dialog")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", forbidden)
    monkeypatch.setattr(view, "open_adoption", forbidden)
    def missing_geometry():
        raise ValueError("The series has no measured frame models.")
    monkeypatch.setattr(view.run_preview, "geometry_selection", missing_geometry)
    view._adopt_clicked()
    view._prepare_adoption_clicked()
    assert view.adopt_button.isEnabled()
    assert "The series has no measured frame models" in view.use_map_progress.text()
    assert view._adoption_thread is None and view._save_thread is None


def test_actual_worker_progress_and_failure_are_displayed_at_transfer_button(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    path = tmp_path / "private-index.json"
    monkeypatch.setattr(view.run_preview, "geometry_selection", lambda: SimpleNamespace(index_path=path))
    def finite(worker):
        worker.progress.emit("Private transfer progress")
        worker.failed.emit("Private finite worker failure")
    monkeypatch.setattr(editor._PlacementWorker, "run", finite)
    assert view.place_version(record, path, runs=["sample"])
    qtbot.waitUntil(lambda: view._placement_thread is None, timeout=5000)
    assert "Private finite worker failure" in view.use_map_progress.text()
    assert "Private transfer progress" not in view.versions_note.text()
    assert view.step_use_map.content.isAncestorOf(view.use_map_progress)


def test_metadata_status_uses_literal_selected_folder_and_reads_no_arrays(tmp_path):
    root = tmp_path / "records"
    version = dict(name="Selected", sha256="a" * 64, folder=str(tmp_path / "versions" / "selected"))
    own, _ = _record(root / "map_placement", "own", version, "2026-01-01T01:00:00Z")
    other = dict(version, name="Same bytes, another saved version", folder=str(tmp_path / "versions" / "other"))
    _record(root / "map_placement", "other", other, "2026-01-01T03:00:00Z")
    _record(root / "map_adoption", "prepared", version, "2026-01-01T02:00:00Z", preparation=True)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = status.existing_progress(root, tmp_path / "no-current.json", version)
    assert result["transfer"]["map_version"] == own["map_version"]
    assert result["transfer"]["created_utc"] == "2026-01-01T01:00:00Z"
    text = status.progress_text(version, dict(approved_utc="2026-01-01T00:00:00Z"), result)
    for expected in ("Approved:", "Transferred: 1 of 1", "camera-photo grid", "Checked: preparation 1 of 1", "Adopted:", "Applied/built: not established"):
        assert expected in text
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_adopted_status_requires_current_revision_exact_preparation_identity(tmp_path):
    root = tmp_path / "records"
    version = dict(name="Selected", sha256="a" * 64, folder=str(tmp_path / "selected"))
    _, binding = _record(root / "map_adoption", "prepared", version, "2026-01-01T02:00:00Z", preparation=True)
    current = tmp_path / "current.json"
    current.write_text(json.dumps(dict(revision_id="private-revision", configuration=dict(
        provenance=dict(preparation=binding)))), encoding="utf-8")
    result = status.existing_progress(root, current, version)
    assert result["adopted"]["revision_id"] == "private-revision"
    other = dict(version, folder=str(tmp_path / "another-same-content-version"))
    assert status.existing_progress(root, current, other)["adopted"] is None
    current.write_text(json.dumps(dict(revision_id="wrong", configuration=dict(
        provenance=dict(preparation=dict(binding, sha256="b" * 64))))), encoding="utf-8")
    assert status.existing_progress(root, current, version)["adopted"] is None


def test_status_read_is_finite_off_gui_and_wait_does_not_start_stale_pending_work(qtbot, tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls, ticks = [], []
    def finite(root, current, record):
        calls.append(threading.current_thread() is threading.main_thread())
        entered.set()
        assert release.wait(5), "Private finite read must be released"
        return dict(transfer=None, preparation=None, adopted=None, errors=[])
    monkeypatch.setattr(status, "existing_progress", finite)
    parent = QWidget()
    controller = status.SeriesUseStatus(parent)
    qtbot.addWidget(parent, before_close_func=lambda _owned: controller.wait_for_workers())
    first = dict(name="First", folder=str(tmp_path / "first"), sha256="a" * 64)
    second = dict(name="Second", folder=str(tmp_path / "second"), sha256="a" * 64)
    timer = QTimer(parent)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start(5)
    try:
        controller.refresh(tmp_path, tmp_path / "current.json", first, None)
        qtbot.waitUntil(entered.is_set, timeout=2000)
        controller.refresh(tmp_path, tmp_path / "current.json", second, None)
        qtbot.waitUntil(lambda: len(ticks) >= 2, timeout=2000)
        release.set()
        controller.wait_for_workers()
        assert not controller.busy() and controller.pending is None
        assert calls == [False]
        assert controller.cached is None  # First never replaces Second.
    finally:
        release.set()
        controller.wait_for_workers()
        timer.stop()


def test_one_optional_controller_receives_series_binding_and_latest_literal_check(view, qtbot, monkeypatch, tmp_path):
    record = _saved_manual(view, qtbot)
    assert view.approve_version(record) is not None
    selection = SimpleNamespace(index_path=tmp_path / "series-bound-index.json")
    monkeypatch.setattr(view.run_preview, "geometry_selection", lambda: selection)
    def forbidden(*args, **kwargs):
        raise AssertionError("The one-button controller must not open a path picker")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", forbidden)
    calls = []
    view._series_adopt_action = lambda *args: calls.append(args)
    prepared = dict(map_version=dict(record), folder=str(tmp_path / "latest-check"))
    view.series_use_status.result = dict(preparation=prepared)
    view._adopt_clicked()
    assert len(calls) == 1
    assert calls[0][0]["folder"] == record["folder"] and calls[0][0]["sha256"] == record["sha256"]
    assert calls[0][1] is selection and calls[0][2] is prepared
    assert view._adoption_thread is None and view._save_thread is None
    assert not view.apply_build_button.isEnabled()


def test_malformed_records_are_counted_before_parse_and_stop_at_total_budget(tmp_path, monkeypatch):
    root = tmp_path / "records"
    version = dict(name="Selected", sha256="a" * 64, folder=str(tmp_path / "selected"))
    for number in range(8):
        folder = root / "map_placement" / str(number)
        folder.mkdir(parents=True)
        (folder / "index.json").write_bytes(b"{ malformed JSON" + b" " * 50)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(status, "_TOTAL", 100)
    calls = []
    read = status._read
    def counted(path, budget):
        calls.append(path)
        return read(path, budget)
    monkeypatch.setattr(status, "_read", counted)
    result = status.existing_progress(root, tmp_path / "no-current.json", version)
    assert len(calls) == 2  # The second malformed file crosses the real byte cap.
    assert any("total bounded read" in error for error in result["errors"])
    assert all(path.read_bytes() == raw for path, raw in before.items())
