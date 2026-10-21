"""One GUI action using real finite preparation, Check and Store workers.

The existing two-run fixture uses real owned 2x3 RAW-basis publication and
reader admission, with only decoder, registration and runtime identity
synthetic. No production RAW accuracy or release is claimed. All requests,
preparations, literal approvals and Store revisions are temporary.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QMessageBox, QWidget

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import raw_shape_basis_reader as basis_reader
from darsia.fluidflower import series_layer_view as editor
from darsia.fluidflower import painting_series_adoption_ui as action
from darsia.fluidflower.map_adoption_view import AdoptionDialog
from darsia.fluidflower.painting_frame_models import FrameModelSelection
from darsia.fluidflower.painting_series_adoption_ui import SeriesAdoptionController

from ..test_geometry_registered_export_apply import registered_setup  # noqa: F401
from ..test_painted_adoption_table import imported, prepared_painting, synthetic_payload_admission  # noqa: F401
from .test_painting_corrections import _Project, view  # noqa: F401


class _Host(QWidget):
    """Small GUI host borrowing the exact production preparation/retirement hooks."""

    busy_changed = Signal(bool)
    prepare_adoption_of = editor.LayerPainterView.prepare_adoption_of
    _adoption_finished = editor.LayerPainterView._adoption_finished
    _adoption_failed = editor.LayerPainterView._adoption_failed
    _adoption_thread_finished = editor.LayerPainterView._adoption_thread_finished
    _current_label_fields = editor.LayerPainterView._current_label_fields
    _approval_for_version = editor.LayerPainterView._approval_for_version

    def __init__(self, data, prepared, version, session, frame):
        super().__init__()
        self.project, self.session = data.project, session
        self.record, self.frame = deepcopy(version), deepcopy(frame)
        index = prepared["geometry_index"]
        self.selection = FrameModelSelection(index["path"], "series display binding",
            str(data.project.root), "synthetic", str(data.series), "synthetic-series",
            index["sha256"], index["size_bytes"])
        self.run_preview = SimpleNamespace(geometry_selection=lambda: self.selection)
        self.use_map_progress, self.versions_note = QLabel(self), QLabel(self)
        self.invalidations = self.refreshes = 0
        self.series_use_status = SimpleNamespace(invalidate=self._invalidate)
        self._adoption_dialog = self._adoption_thread = self._adoption_worker = None
        self.last_adoption = None
        self._map_approvals_error = ""
        self.map_approvals = []
        self.force_unapproved = False
        self.adoption_controller = SeriesAdoptionController(self)

    def _invalidate(self):
        self.invalidations += 1

    def _read_approvals(self, run):
        self.map_approvals = painting.read_map_approvals(self.project, run)
        if self.force_unapproved:
            self.map_approvals = []

    def _selected_version(self):
        return self.record

    def current_frame(self):
        return self.frame

    def _reviewer(self):
        return "synthetic"

    def _series_runs(self):
        return ["one", "two"]

    def _placeable(self, record):
        return self._approval_for_version(record) is not None

    def _refresh_versions(self):
        self.refreshes += 1

    def busy(self):
        return bool(self._adoption_thread is not None
            or self._adoption_dialog is not None and self._adoption_dialog.busy())

    def wait_for_workers(self):
        self.adoption_controller.wait_for_workers()
        if self._adoption_thread is not None:
            self._adoption_thread.wait()


@pytest.fixture
def host(qtbot, registered_setup, tmp_path):
    prepared, version, session, frame, _ = prepared_painting(registered_setup, tmp_path)
    widget = _Host(registered_setup, prepared, version, session, frame)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    widget.show()
    return widget, registered_setup, prepared


def _revision(data):
    return geometry.GeometryRevisionStore(data.project).current()["revision_id"]


@pytest.mark.parametrize("decision", ["decline", "accept", "context_changed_during_question"])
def test_one_action_prepares_checks_retires_and_confirms_once_before_real_store(
        qtbot, host, monkeypatch, decision):
    widget, data, old_preparation = host
    calls, questions = [], []
    original_prepare = adoption.prepare_adoption
    original_candidate = adoption.adoption_candidate
    original_save = geometry.GeometryRevisionStore.save_revision

    def prepare(*args, **kwargs):
        calls.append(("prepare", threading.current_thread() is threading.main_thread()))
        return original_prepare(*args, **kwargs)

    def candidate(*args, **kwargs):
        calls.append(("check", threading.current_thread() is threading.main_thread()))
        return original_candidate(*args, **kwargs)

    def save(*args, **kwargs):
        calls.append(("save", threading.current_thread() is threading.main_thread()))
        return original_save(*args, **kwargs)

    def confirm(dialog, title, text, *args):
        questions.append((title, text))
        assert threading.current_thread() is threading.main_thread()
        assert widget._adoption_thread is None and dialog._thread is None
        assert dialog.candidate is not None and "Ready to save:" in dialog.status.text()
        assert dialog.automatic and dialog.choices == {}
        assert not dialog.check_button.isVisible() and not dialog.save_button.isVisible()
        assert dialog.folder != Path(old_preparation["folder"])
        if decision == "context_changed_during_question":
            widget.selection = replace(widget.selection, series_id="changed-current-context")
        return (QMessageBox.StandardButton.No if decision == "decline"
                else QMessageBox.StandardButton.Yes)

    monkeypatch.setattr(adoption, "prepare_adoption", prepare)
    monkeypatch.setattr(adoption, "adoption_candidate", candidate)
    monkeypatch.setattr(geometry.GeometryRevisionStore, "save_revision", save)
    monkeypatch.setattr(QMessageBox, "question", confirm)
    before = _revision(data)
    controller = widget.adoption_controller
    assert controller.available
    assert controller.start(widget.record, widget.selection, old_preparation)
    qtbot.waitUntil(lambda: controller.dialog is not None and controller.context is None
                   and not widget.busy(), timeout=30000)
    assert calls == [("prepare", False), ("check", False)] + (
        [("save", False)] if decision == "accept" else [])
    assert len(questions) == 1
    assert "Current frame frozen at preparation" in questions[0][1]
    assert controller.preparation["folder"] != old_preparation["folder"]
    assert controller.preparation["map_version"]["approval"] == widget.map_approvals[-1]
    assert controller.dialog._thread is None and controller.dialog._worker is None
    if decision == "accept":
        assert _revision(data) != before
        assert controller.dialog.revision["revision_id"] == _revision(data)
        assert "Apply and Build have not run" in widget.use_map_progress.text()
    else:
        assert _revision(data) == before and controller.dialog.revision is None
        assert "nothing was adopted" in widget.use_map_progress.text().lower()


@pytest.mark.parametrize("changed", ["version", "approval", "frame", "series_geometry"])
def test_changed_context_during_real_check_never_asks_or_saves(
        qtbot, host, monkeypatch, changed):
    widget, data, _ = host
    entered, release = threading.Event(), threading.Event()
    original = adoption.adoption_candidate

    def finite(*args, **kwargs):
        entered.set()
        assert release.wait(10), "The private finite Check must be released"
        return original(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A stale checked context must not ask for adoption")

    monkeypatch.setattr(adoption, "adoption_candidate", finite)
    monkeypatch.setattr(QMessageBox, "question", forbidden)
    before = _revision(data)
    controller = widget.adoption_controller
    try:
        assert controller.start(widget.record, widget.selection)
        qtbot.waitUntil(entered.is_set, timeout=30000)
        assert widget._adoption_thread is None and controller.dialog.busy()
        if changed == "version":
            widget.record = dict(widget.record, folder=widget.record["folder"] + "-same-content")
        elif changed == "approval":
            widget.force_unapproved = True
        elif changed == "frame":
            widget.frame.width_m += .000001
        else:
            widget.selection = replace(widget.selection, series_id="another-current-series")
        release.set()
        qtbot.waitUntil(lambda: controller.context is None and not widget.busy(), timeout=30000)
        assert _revision(data) == before and controller.dialog.revision is None
        assert "changed" in widget.use_map_progress.text()
    finally:
        release.set()
        widget.wait_for_workers()


def test_real_missing_raw_basis_failure_is_plain_and_never_opens_legacy_fallback(
        qtbot, host, monkeypatch, tmp_path):
    widget, data, _ = host
    # A separate, honest missing-basis index leaves the genuine original
    # fixture and every one of its immutable certificate/model bytes intact.
    folder = tmp_path / "missing_basis"
    folder.mkdir()
    rows = {}
    for run in ("one", "two"):
        model_path = folder / (run + ".json")
        model_path.write_text(json.dumps(dict(baseline_native_grid=dict(shape_hw=[2, 3]))))
        rows[run] = dict(reference_model=dict(path=str(model_path)))
    raw = json.dumps(dict(runs=rows)).encode()
    index = folder / "index.json"
    index.write_bytes(raw)
    widget.selection = replace(widget.selection, index_path=str(index),
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Missing RAW basis must refuse before saved-map drawing")
    monkeypatch.setattr(editor.lpm, "version_layers", forbidden)
    before = _revision(data)
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: questions.append(args))
    assert widget.adoption_controller.start(widget.record, widget.selection)
    qtbot.waitUntil(lambda: widget.adoption_controller.context is None and not widget.busy(), timeout=10000)
    assert widget.adoption_controller.dialog is None and questions == []
    assert "RAW" in widget.use_map_progress.text() and "not adopted" in widget.use_map_progress.text()
    assert _revision(data) == before


def test_wait_naturally_retires_check_without_triggering_confirmation_or_save(
        qtbot, host, monkeypatch):
    widget, data, _ = host
    entered, release = threading.Event(), threading.Event()
    original = adoption.adoption_candidate
    questions = []

    def finite(*args, **kwargs):
        entered.set()
        assert release.wait(10), "The private finite Check must be released"
        return original(*args, **kwargs)

    monkeypatch.setattr(adoption, "adoption_candidate", finite)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: questions.append(args))
    before = _revision(data)
    controller = widget.adoption_controller
    try:
        assert controller.start(widget.record, widget.selection)
        qtbot.waitUntil(entered.is_set, timeout=30000)
        release.set()
        controller.wait_for_workers()
        qtbot.wait(1)  # Deliver the already queued retired signal; no new work may follow it.
        assert controller.context is None and questions == [] and not widget.busy()
        assert controller.dialog._thread is None and controller.dialog._worker is None
        assert _revision(data) == before
    finally:
        release.set()
        widget.wait_for_workers()


def test_wait_during_preparation_suppresses_queued_check_and_old_information_popup(
        qtbot, host, monkeypatch):
    widget, data, _ = host
    entered, release = threading.Event(), threading.Event()
    original = adoption.prepare_adoption
    questions = []

    def finite(*args, **kwargs):
        entered.set()
        assert release.wait(10), "The private finite preparation must be released"
        return original(*args, **kwargs)

    monkeypatch.setattr(adoption, "prepare_adoption", finite)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: questions.append(args))
    before = _revision(data)
    controller = widget.adoption_controller
    try:
        assert controller.start(widget.record, widget.selection)
        qtbot.waitUntil(entered.is_set, timeout=30000)
        release.set()
        widget.wait_for_workers()
        qtbot.waitUntil(lambda: widget._adoption_thread is None, timeout=10000)
        assert controller.context is None and controller.dialog is None and questions == []
        assert not hasattr(widget, "_adoption_message")
        assert _revision(data) == before
    finally:
        release.set()
        widget.wait_for_workers()


def test_save_worker_failure_is_plain_and_never_claims_adoption(qtbot, host, monkeypatch):
    widget, data, _ = host
    before = _revision(data)
    questions = []

    def confirm(*args):
        questions.append(args)
        return QMessageBox.StandardButton.Yes

    def failed_save(*_args, **_kwargs):
        raise OSError("private synthetic Store refusal")

    monkeypatch.setattr(QMessageBox, "question", confirm)
    monkeypatch.setattr(geometry.GeometryRevisionStore, "save_revision", failed_save)
    controller = widget.adoption_controller
    assert controller.start(widget.record, widget.selection)
    qtbot.waitUntil(lambda: controller.dialog is not None and controller.context is None
                   and not widget.busy(), timeout=30000)
    assert len(questions) == 1 and controller.dialog.revision is None
    assert "private synthetic Store refusal" in widget.use_map_progress.text()
    assert controller.dialog._thread is None and controller.dialog._worker is None
    assert _revision(data) == before


def test_automatic_action_never_substitutes_the_legacy_facies_id_dialog(host, monkeypatch):
    widget, data, prepared = host
    legacy = {key: value for key, value in prepared.items() if key != "painting_facies"}
    monkeypatch.setattr(adoption, "read_preparation", lambda *_args: legacy)
    with pytest.raises(ValueError, match="manual facies remapping is not a fallback"):
        AdoptionDialog(data.project, prepared["folder"], actor="synthetic", automatic_only=True)


def test_missing_optional_backend_keeps_the_actual_editor_action_disabled(
        view, qtbot, tmp_path, monkeypatch):
    original = action.find_spec
    checked = []
    def available(name):
        checked.append(name)
        if name == "darsia.fluidflower.raw_shape_basis_reader":
            return None
        return original(name)
    monkeypatch.setattr(action, "find_spec", available)
    widget = editor.LayerPainterView(_Project(tmp_path / "unsupported-project"), view._reference)
    qtbot.addWidget(widget, before_close_func=lambda owned: owned.wait_for_workers())
    assert checked == list(action._BACKEND_MODULES)
    assert not widget.adoption_controller.available and widget._series_adopt_action is None
    assert not widget.adopt_button.isEnabled()
    widget._adopt_clicked()
    assert "RAW basis and the automatic adoption backend" in widget.use_map_progress.text()
    assert widget._adoption_thread is None and widget._adoption_dialog is None
    assert not widget.apply_build_button.isEnabled()
