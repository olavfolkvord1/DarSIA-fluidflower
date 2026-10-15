from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.ui import review_version_controls as ui


def fake_window(tmp_path, identity="a"):
    window = QMainWindow()
    body = QWidget()
    QVBoxLayout(body).addWidget(QLabel("Existing phase-map canvas"))
    window.setCentralWidget(body)
    window.workspace = NS(
        key="current_comparison",
        review_surface="open_correction",
        single_model=True,
        approval_only=False,
        read_only=False,
        target_model_id=identity,
        output_dir=tmp_path / identity,
    )
    window.reviewer = QLineEdit("reviewer")
    window.store = object()
    window._comparison_local_error = False
    window._comparison_save_task = None
    window._comparison_pending_saves = {}
    window._comparison_unprotected_requests = {}
    window.save_dirty = False
    window.save_calls = 0

    def save():
        window.save_calls += 1

    window._save_now = save
    return window


def arguments(tmp_path):
    return NS(
        review_model_choice=[
            json.dumps({"root": str(tmp_path / key), "label": key.upper()})
            for key in ("a", "b")
        ],
        review_current_model_root=tmp_path / "a",
        evidence_audit=None,
    )


@pytest.fixture
def setup(qtbot, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        ui.QMessageBox, "warning", lambda *args: warnings.append(args[-1])
    )
    old = fake_window(tmp_path)
    qtbot.addWidget(old)
    old.show()
    args = arguments(tmp_path)
    prepared = NS(review_current_model_root=tmp_path / "b", evidence_audit=None)
    reads, builds = [], []

    def reader(choice, window, original_args, api):
        reads.append((choice, window, original_args))
        return prepared, NS(target_model_id="b")

    def builder(target, window, api, **kwargs):
        new = fake_window(tmp_path, "b")
        qtbot.addWidget(new)
        builds.append((target, window, new))
        return new

    controller = ui.ReviewVersionControls(
        old, args, None, target_reader=reader, window_builder=builder
    )
    return NS(
        old=old,
        args=args,
        prepared=prepared,
        reads=reads,
        builds=builds,
        control=controller,
        warnings=warnings,
    )


def test_arguments_are_explicit_choices_not_code_or_generated_files(tmp_path):
    choices = [
        {"root": str(tmp_path / "folder with spaces"), "display_name": "Version 1"}
    ]
    argv = ui.model_choice_arguments(choices + choices, choices[0]["root"])
    assert argv[:2] == ["--review-current-model-root", choices[0]["root"]]
    assert len(argv) == 4
    decoded = ui.parse_model_choices([argv[3]])
    assert decoded[0].root == Path(choices[0]["root"])
    assert decoded[0].label == "Version 1"
    assert list(tmp_path.iterdir()) == []


def test_dropdown_attaches_to_retained_canvas_only(setup):
    old = setup.old
    assert (
        old.review_model_choice.currentData().root
        == setup.args.review_current_model_root
    )
    assert (
        old.centralWidget().layout().itemAt(0).widget().text()
        == "Existing phase-map canvas"
    )
    assert old.save_calls == 0 and not setup.builds and not setup.reads
    old.workspace.review_surface = "blind_regions"
    with pytest.raises(ValueError, match="open-correction"):
        ui.ReviewVersionControls(old, setup.args, None)


def test_version_toolbar_preserves_real_splitter_central_widget(qtbot, tmp_path):
    from PySide6.QtWidgets import QSplitter

    window = fake_window(tmp_path)
    qtbot.addWidget(window)
    splitter = QSplitter()
    left, right = QLabel("Original photograph"), QLabel("Saved phase map")
    splitter.addWidget(left)
    splitter.addWidget(right)
    window.setCentralWidget(splitter)
    assert splitter.layout() is None
    ui.ReviewVersionControls(window, arguments(tmp_path), None)
    assert window.centralWidget() is splitter
    assert splitter.count() == 2
    assert splitter.widget(0) is left and splitter.widget(1) is right
    assert window.review_model_toolbar.parent() is window
    assert window.review_version_panel.parent() is window.review_model_toolbar
    assert window.save_calls == 0


def test_invalid_target_preserves_current_window_store_and_labels(setup):
    old_store = setup.old.store

    def fail(*_):
        raise ValueError("literal mismatched source")

    setup.control.target_reader = fail
    setup.control.choose(setup.old, 1)
    assert setup.old.isVisible() and setup.old.store is old_store
    assert setup.old.save_calls == 0 and not setup.builds
    assert setup.old.review_model_choice.currentIndex() == 0
    assert setup.warnings == ["literal mismatched source"]


def test_switch_waits_for_original_save_and_builds_a_separate_window(setup):
    old_store, old_workspace = setup.old.store, setup.old.workspace

    def pending():
        setup.old.save_calls += 1
        setup.old.save_dirty = True
        setup.old._comparison_save_task = object()

    setup.old._save_now = pending
    setup.control.choose(setup.old, 1)
    assert setup.old.save_calls == 1 and not setup.builds
    assert setup.old.isVisible() and setup.old.review_model_choice.currentIndex() == 0
    setup.old.save_dirty = False
    setup.old._comparison_save_task = None
    setup.control.finish_switch()
    new = setup.builds[0][2]
    assert new.isVisible() and not setup.old.isVisible()
    assert setup.old.store is old_store and setup.old.workspace is old_workspace
    assert new.store is not old_store and new.workspace.target_model_id == "b"
    assert (
        new.review_model_choice.currentData().root
        == setup.prepared.review_current_model_root
    )
    assert not setup.control.timer.isActive()


def test_save_failure_and_cancel_keep_original_review_open(setup):
    def failure():
        setup.old._comparison_local_error = True

    setup.old._save_now = failure
    setup.control.choose(setup.old, 1)
    assert setup.old.isVisible() and not setup.builds and setup.control.pending is None
    assert "could not be protected" in setup.warnings[0]
    setup.old._comparison_local_error = False
    setup.old._save_now = lambda: setattr(setup.old, "save_dirty", True)
    setup.control.choose(setup.old, 1)
    setup.control.cancel_switch()
    assert setup.old.isVisible() and not setup.builds and setup.old.save_dirty
    assert not setup.control.timer.isActive()


def test_builder_failure_preserves_old_review(setup):
    setup.control.window_builder = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ValueError("literal build failure")
    )
    setup.control.choose(setup.old, 1)
    assert setup.old.isVisible() and not setup.builds
    assert setup.old.review_model_choice.currentIndex() == 0
    assert setup.warnings == ["literal build failure"]


def test_real_qt_timer_waits_until_original_save_finishes(setup, qtbot):
    setup.old._save_now = lambda: setattr(setup.old, "save_dirty", True)
    setup.control.choose(setup.old, 1)
    assert not setup.builds
    setup.old.save_dirty = False
    qtbot.waitUntil(lambda: bool(setup.builds), timeout=1000)
    assert not setup.old.isVisible() and setup.builds[0][2].isVisible()


def _audit_module(monkeypatch, attach):
    module = ModuleType("darsia.fluidflower.ui.evidence_audit")
    module.attach_evidence_audit = attach
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_disagreement_toggle_reuses_existing_dock_and_rank_order(setup, monkeypatch):
    calls = []

    def attach(window, path, api):
        calls.append(path)
        dock = QDockWidget(window)
        dock.reason = QComboBox()
        dock.reason.addItem("Temporal", "temporal_rule_conflict")
        dock.reason.addItem("All flagged", "")
        dock.order = QComboBox()
        dock.order.addItem("Time", "time")
        dock.order.addItem("Highest disagreement", "score")
        dock.refresh = lambda: None
        window.evidence_audit = dock
        return dock

    _audit_module(monkeypatch, attach)
    monkeypatch.setattr(
        ui.QFileDialog, "getOpenFileName", lambda *_: ("chosen-audit.json", "")
    )
    setup.old.review_disagreements.setChecked(True)
    dock = setup.old.evidence_audit
    assert calls == [Path("chosen-audit.json")]
    assert dock.reason.currentData() == "" and dock.order.currentData() == "score"
    setup.old.review_disagreements.setChecked(False)
    assert dock.isHidden()
    setup.old.review_disagreements.setChecked(True)
    assert setup.old.evidence_audit is dock and len(calls) == 1
    assert setup.old.save_calls == 0 and not setup.builds


def test_cancelled_or_foreign_queue_preserves_review(setup, monkeypatch):
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *_: ("", ""))
    setup.old.review_disagreements.setChecked(True)
    assert not setup.old.review_disagreements.isChecked() and setup.old.isVisible()
    monkeypatch.setattr(
        ui.QFileDialog, "getOpenFileName", lambda *_: ("foreign.json", "")
    )

    def fail(*_):
        raise ValueError("Review window and audit queue belong to different models")

    _audit_module(monkeypatch, fail)
    setup.old.review_disagreements.setChecked(True)
    assert not setup.old.review_disagreements.isChecked() and setup.old.isVisible()
    assert "different models" in setup.warnings[0]


def test_target_uses_existing_binding_checks_new_session_and_same_purpose(
    qapp, tmp_path, monkeypatch
):
    root = tmp_path / "b"
    root.mkdir()
    (root / "postflight_manifest.json").write_text(
        json.dumps({"jobs": [{"dataset": "ffac"}]})
    )
    binding = dict(
        model_id="model-b",
        combined_root=str(root),
        ffac_root=str(root),
        ffum_root=str(root),
        selected_candidates={"ac52": "candidate-b"},
    )
    called = []
    audit = ModuleType("darsia.fluidflower.phase_evidence_audit")
    audit.verified_review_binding = (
        lambda model, **kw: called.append((model, kw)) or binding
    )
    training = ModuleType("darsia.fluidflower.ui.model_training")
    training._validate_saved_phase_binding = (
        lambda model, value, manifest: called.append((model, value, manifest))
    )
    paths = ModuleType("darsia.fluidflower.series_paths")
    paths.series_key = lambda value: value.casefold().replace("_", "")
    for module in (audit, training, paths):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    old = fake_window(tmp_path)
    old.app_model = NS(inventory=NS(pairs=[NS(candidate_a=NS(dataset="FF_AC"))]))
    original_args = NS(
        review_current_model_root=tmp_path / "a",
        review_session_id="old-session",
        evidence_audit=Path("old-queue.json"),
    )

    def workspaces(args):
        return {
            "current_comparison": NS(
                **{**vars(old.workspace), "target_model_id": args.target_model_id}
            )
        }

    api = NS(review_workspaces=workspaces)
    first = ui.verified_target(ui.ModelChoice(root, "B"), old, original_args, api)
    second = ui.verified_target(ui.ModelChoice(root, "B"), old, original_args, api)
    assert called[0] == (root, {"saved_comparison": True})
    assert called[1] == (root, binding, {"jobs": [{"dataset": "ffac"}]})
    assert first[0].review_session_id != second[0].review_session_id != "old-session"
    assert (
        first[0].single_candidate == ["ac52=candidate-b"]
        and first[0].evidence_audit is None
    )
    assert (
        original_args.review_session_id == "old-session"
        and original_args.evidence_audit == Path("old-queue.json")
    )
    (root / "postflight_manifest.json").write_text(
        json.dumps({"jobs": [{"dataset": "ffum"}]})
    )
    with pytest.raises(ValueError, match="another experimental series"):
        ui.verified_target(ui.ModelChoice(root, "B"), old, original_args, api)
    old.close()


def test_original_builder_refuses_previous_store_path(qapp, tmp_path):
    old = fake_window(tmp_path)
    api = NS(local_reviewer_workspace=lambda *args: old.workspace)
    args = NS(local_review_root=tmp_path, cache_root=tmp_path)
    with pytest.raises(ValueError, match="reuse the previous model"):
        ui.original_window_builder((args, old.workspace), old, api, use_opengl=False)
    old.close()
