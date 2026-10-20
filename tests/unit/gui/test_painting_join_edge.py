"""Preview, visibility, one undo and unchanged session settings for joining."""

import copy

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QDialogButtonBox, QMenu

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower.layer_painting_join import join_to_neighbour
from darsia.fluidflower.series_geometry import SeriesRunInputs


TARGET = [(5.0, 15.0), (5.0, 50.0), (28.0, 50.0), (28.0, 15.0)]
MASTER = [(30.0, 5.0), (30.0, 70.0), (55.0, 70.0), (55.0, 5.0)]


class Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}

    def update_metadata(self, *args, **kwargs):
        pass


@pytest.fixture
def view(qtbot, tmp_path, monkeypatch):
    settings = str(tmp_path / "settings.ini")
    monkeypatch.setattr(slv, "QSettings", lambda *a: QSettings(settings, QSettings.Format.IniFormat))
    monkeypatch.setattr(QMenu, "popup", lambda *a, **k: None)
    baseline = tmp_path / "raw.png"
    Image.new("RGB", (80, 60), (30, 40, 50)).save(baseline)
    config = tmp_path / "run.toml"
    config.write_text('[data]\nfolder="."\nbaseline="raw.png"\n', encoding="utf-8")
    reference = SeriesRunInputs("example", config, baseline_image=baseline,
                                width_m=0.08, height_m=0.06)
    widget = slv.LayerPainterView(Project(tmp_path / "project"), lambda: reference)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    qtbot.waitExposed(widget)
    widget.add_polygon(lp.Polygon(1, list(TARGET)))
    widget.add_polygon(lp.Polygon(2, list(MASTER)))
    widget.set_tool("edit")
    return widget


def test_apply_and_one_undo_restore_both_polygons_and_all_settings(view):
    before = copy.deepcopy(view.session.to_dict())
    undo_count = len(view._undo_stack)
    plan = join_to_neighbour(TARGET, MASTER, 3.0)
    assert view._apply_join_preview(0, 1, plan)
    assert len(view._undo_stack) == undo_count + 1
    after = copy.deepcopy(view.session.to_dict())
    assert before["polygons"] != after["polygons"]
    assert {k: v for k, v in before.items() if k != "polygons"} == {
        k: v for k, v in after.items() if k != "polygons"
    }
    # Ordinary polygons carry shared edges; no join-specific session field.
    restored = lp.PaintingSession.from_dict(after)
    assert lp.shared_edges(restored.polygons)
    view.undo()
    assert view.session.to_dict() == before
    view.redo()
    assert view.session.to_dict() == after


def test_preview_draws_stretches_and_length_but_changes_nothing(view, qtbot):
    before = copy.deepcopy(view.session.to_dict())
    dialog = slv._JoinEdgesDialog(view, view.canvas, TARGET, [(1, "Neighbour", MASTER)], 1.0)
    qtbot.addWidget(dialog)
    assert dialog.distance.value() == 5
    assert "mm" in dialog.summary.text() and "1 stretch" in dialog.summary.text()
    assert not dialog.old_item.path().isEmpty()
    assert not dialog.new_item.path().isEmpty()
    assert view.session.to_dict() == before
    dialog.distance.setValue(1)
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    assert "No stretch" in dialog.summary.text()
    dialog.reject()
    assert dialog.old_item.scene() is None
    assert dialog.new_item.scene() is None
    assert view.session.to_dict() == before


def test_hidden_neighbour_is_ignored_and_cannot_be_applied(view):
    plan = join_to_neighbour(TARGET, MASTER, 3)
    before = copy.deepcopy(view.session.to_dict())
    view._set_hidden_layers({2})
    view._join_polygon_dialog(0)
    assert "Show another polygon" in view.canvas_hint.text()
    assert not view._apply_join_preview(0, 1, plan)
    assert view.session.to_dict() == before


def test_selected_polygon_button_and_context_menu_reach_the_same_action(view, monkeypatch):
    calls = []
    monkeypatch.setattr(view, "_join_polygon_dialog", calls.append)
    view.canvas.set_selection([(0, 0), (0, 1)])
    view.join_edge_button.click()
    view._polygon_menu(0, -1, QPoint(0, 0))
    action = next(a for a in view._polygon_menu_widget.actions()
                  if a.text() == "Join to neighbour's edge…")
    action.trigger()
    assert calls == [0, 0]


def test_refused_self_intersection_shows_location_and_disables_apply(view, qtbot):
    crossed = [(0, 0), (10, 10), (0, 10), (10, 0)]
    dialog = slv._JoinEdgesDialog(view, view.canvas, crossed, [(1, "Neighbour", MASTER)], 1)
    qtbot.addWidget(dialog)
    assert "raw row 5.00, column 5.00" in dialog.summary.text()
    assert not dialog.error_item.path().isEmpty()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Apply).isEnabled()
    dialog.reject()
