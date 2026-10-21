"""One-click display filters on private synthetic paintings never change saved maps."""
from copy import deepcopy
import threading

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import series_layer_view as slv
from darsia.fluidflower import painting_run_preview_ui as preview_ui
from darsia.fluidflower.painting_run_preview import RunPreview
from .test_painting_photo_saturation import settings, make_view, view  # noqa: F401
from .test_painting_corrections import _proposal, _show
from .test_painting_facies_display import _group, _files
from .test_layer_visibility_and_add_point import _check_box_center
from .test_painting_manual_map import LEFT, RIGHT, MIDDLE, _drawn
from .test_painting_run_preview import private_settings  # noqa: F401
from .test_painting_run_preview_status import _chooser, _select


def _facies_item(view, name):
    return next(view.facies_table.item(row, 0) for row in range(view.facies_table.rowCount())
                if view.facies_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == name)


def _click_item(table, item):
    painting = table.window()
    column = item.column()
    if table is painting.facies_table:
        target = item.data(Qt.ItemDataRole.UserRole)
        painting.step_export.set_expanded(True)
    else:
        target = painting.session.layers[item.row()].layer_id
        painting.step_paint.set_expanded(True)
    painting.steps_scroll.ensureWidgetVisible(table)
    QTest.qWait(1)
    if table is painting.facies_table:
        item = _facies_item(painting, target)
    else:
        row = next(index for index, layer in enumerate(painting.session.layers) if layer.layer_id == target)
        item = table.item(row, column)
    table.scrollToItem(item)
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier, _check_box_center(table, item))


def test_legend_and_facies_table_expose_named_checkbox_and_swatch_in_layer_mode(view):
    assert view.color_mode.currentText() == "By layer"
    assert view.facies_visibility.isVisibleTo(view)
    for name in view.session.facies_order():
        box = view.facies_checks[name]
        item = _facies_item(view, name)
        members = view.session.layers_in_facies(name)
        assert box.text() == name and box.isVisibleTo(view)
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert box.isEnabled() == bool(members)
        if members:
            assert box.checkState() == item.checkState() == Qt.CheckState.Checked
        assert not box.icon().isNull()
        icon = box.icon().pixmap(14, 14).toImage()
        assert icon.pixelColor(7, 7).name() == slv._facies_color(name).name()
        assert "display only" in box.toolTip()
    assert view.layer_table.columnCount() == 5
    assert all(view.layer_table.item(row, 4).checkState() == Qt.CheckState.Checked
               for row in range(view.layer_table.rowCount()))


@pytest.mark.parametrize("mode", ["layer", "facies"])
def test_one_legend_click_hides_group_polygons_strokes_and_map_in_both_modes(view, mode):
    name, first, second = _group(view)
    other = next(layer.layer_id for layer in view.session.layers if layer.facies != name)
    view.session.polygons = [
        lp.Polygon(first, [(5., 5.), (5., 35.), (25., 35.), (25., 5.)]),
        lp.Polygon(second, [(30., 45.), (30., 75.), (55., 75.), (55., 45.)]),
        lp.Polygon(other, [(35., 5.), (35., 25.), (55., 25.), (55., 5.)]),
    ]
    view.session.strokes = [lp.Stroke(first, 3., [(20., 20.)]),
                            lp.Stroke(second, 3., [(40., 60.)]),
                            lp.Stroke(other, 3., [(45., 15.)])]
    view._render_marks()
    view.set_color_mode(mode)
    view.show_strokes.setChecked(True)
    classes = np.tile(np.repeat(np.array([first, second, other], np.uint8), 8), (12, 1))
    before = deepcopy(view.session.to_dict())
    assert [item.isVisible() for item in view.canvas._polygon_items] == [True, True, True]
    view.facies_checks[name].click()
    assert view.hidden_layers == {first, second} == view.canvas.hidden_layers
    assert [item.isVisible() for item in view.canvas._polygon_items] == [False, False, True]
    image = view.canvas._stroke_image
    assert image.pixelColor(10, 10).alpha() == 0
    assert image.pixelColor(30, 20).alpha() == 0
    assert image.pixelColor(7, 22).alpha() > 0
    rgba, shown = view._layer_overlay(classes)
    assert not shown[:, :16].any() and shown[:, 16:].all()
    assert rgba[5, 4, 3] == rgba[5, 12, 3] == 0 and rgba[5, 20, 3] > 0
    assert view.session.to_dict() == before
    view.facies_checks[name].click()
    assert not view.hidden_layers
    assert all(item.isVisible() for item in view.canvas._polygon_items)


def test_table_checkbox_is_one_click_and_facies_gates_preserve_individual_layer_boxes(view, monkeypatch):
    name, first, second = _group(view)
    before = deepcopy(view.session.to_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility click tried to rename a facies")
    monkeypatch.setattr(view, "rename_facies", forbidden)
    _click_item(view.facies_table, _facies_item(view, name))
    assert view.hidden_layers == {first, second}
    row = next(index for index, layer in enumerate(view.session.layers) if layer.layer_id == first)
    assert view.layer_table.item(row, 4).checkState() == Qt.CheckState.Checked
    _click_item(view.layer_table, view.layer_table.item(row, 4))
    assert view.hidden_layers == {first, second}
    assert view.facies_checks[name].checkState() == Qt.CheckState.Unchecked
    assert _facies_item(view, name).checkState() == Qt.CheckState.Unchecked
    view.facies_checks[name].click()
    assert view.hidden_layers == {first}
    assert view.facies_checks[name].checkState() == Qt.CheckState.Checked
    assert _facies_item(view, name).checkState() == Qt.CheckState.Checked
    view.facies_checks[name].click()
    view.facies_checks[name].click()
    assert view.hidden_layers == {first}
    _click_item(view.layer_table, view.layer_table.item(row, 4))
    assert not view.hidden_layers
    assert view.session.to_dict() == before


@pytest.mark.parametrize("owner", ["layer", "facies", "legend"])
def test_all_on_all_off_preserve_highlight_and_existing_show_only_actions(view, owner):
    name, first, second = _group(view)
    view.highlight_facies(name)
    getattr(view, f"{owner}_all_off_button").click()
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers}
    if owner != "layer":
        assert all(box.checkState() == Qt.CheckState.Unchecked for box in view.facies_checks.values())
    else:
        assert view.facies_checks[name].checkState() == Qt.CheckState.Checked
    assert view._highlight_facies == name
    getattr(view, f"{owner}_all_on_button").click()
    assert not view.hidden_layers and view._highlight_facies == name
    view.show_only_facies(name)
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.facies != name}
    view.show_only_layer(first)
    assert view.hidden_layers == {layer.layer_id for layer in view.session.layers if layer.layer_id != first}
    assert view.facies_checks[name].checkState() == Qt.CheckState.Checked
    assert not view.layer_visible(second)


def test_checkbox_clicks_preserve_session_proposal_digest_export_key_files_and_undo(view, monkeypatch):
    name, first, _second = _group(view)
    _show(view, _proposal())
    view.save_session()
    before = deepcopy(view.session.to_dict()), view.proposal.classes.copy(), view._export_state_key(), _files(view)
    digest, undo = lp.proposal_digest(view.session, view.current_frame()), deepcopy(view._undo_stack)
    changes = []
    view.changed.connect(lambda: changes.append(True))
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility checkbox tried to save a painting")
    monkeypatch.setattr(lp, "save_painting_session", forbidden)
    for mode in ("facies", "layer"):
        view.set_color_mode(mode)
        view.facies_checks[name].click()
        _click_item(view.facies_table, _facies_item(view, name))
        view.set_layer_visible(first, False)
        view.legend_all_off_button.click()
        view.legend_all_on_button.click()
    assert view.session.to_dict() == before[0]
    np.testing.assert_array_equal(view.proposal.classes, before[1])
    assert view._export_state_key() == before[2] and _files(view) == before[3]
    assert lp.proposal_digest(view.session, view.current_frame()) == digest
    assert view._undo_stack == undo and changes == []
    assert not view._dirty and not view._save_timer.isActive()


def test_preview_style_seam_uses_supplied_session_and_emits_without_saved_change(view):
    name, first, second = _group(view)
    alternate = deepcopy(view.session)
    other = next(layer for layer in alternate.layers if layer.facies != name)
    other.facies = name
    snapshot = deepcopy(alternate.to_dict())
    styles, saved = [], []
    view.display_styles_changed.connect(lambda: styles.append(True))
    view.changed.connect(lambda: saved.append(True))
    view.set_layer_visible(first, False)
    view.facies_checks[name].click()
    view.set_color_mode("facies")
    colors, hidden = view.display_layer_styles(alternate)
    assert {first, second, other.layer_id} <= hidden
    assert colors[other.layer_id] == slv._facies_color(name)
    view.facies_checks[name].click()
    assert view.hidden_layers == {first}
    assert styles and saved == [] and alternate.to_dict() == snapshot


def test_saved_manual_version_remains_identical_while_checkbox_changes_display(view, qtbot):
    name, _first, _second = _group(view)
    for polygon in (LEFT, RIGHT, MIDDLE):
        view.add_polygon(deepcopy(polygon))
    view.map_shown.setCurrentIndex(view.map_shown.findText("Manual map"))
    _drawn(view, qtbot)
    record = deepcopy(view.save_version("Drawn map"))
    assert record is not None
    displayed_record = deepcopy(next(row for row in view.versions if row["name"] == record["name"]))
    classes, key, files = view._manual.classes.copy(), view._manual_key, _files(view)
    view.facies_checks[name].click()
    view.set_color_mode("facies")
    view.legend_all_on_button.click()
    assert view._manual_thread is None and view._manual_key == key
    np.testing.assert_array_equal(view._manual.classes, classes)
    assert _files(view) == files
    assert next(row for row in view.versions if row["name"] == record["name"]) == displayed_record


def test_empty_and_renamed_facies_have_safe_controls_and_new_session_resets_visibility(view):
    name, first, _second = _group(view)
    view.set_layer_visible(first, False)
    view.facies_checks[name].click()
    view.rename_facies(name, "Fine <sand> & water")
    assert view.facies_checks["Fine <sand> & water"].text() == "Fine <sand> & water"
    assert view.facies_checks["Fine <sand> & water"].checkState() == Qt.CheckState.Unchecked
    empty = view.add_facies("Unused")
    assert not view.facies_checks[empty].isEnabled()
    assert view.facies_checks[empty].checkState() == Qt.CheckState.Unchecked
    view.session = lp.PaintingSession.from_dict(view.session.to_dict())
    view._refresh_layers()
    view._refresh_facies_table()
    assert not view.hidden_layers and view._highlight_facies is None
    assert view.facies_checks["Fine <sand> & water"].checkState() == Qt.CheckState.Checked


def _preview_packet(session, options, first, second):
    labels = np.empty((60, 80), np.uint8)
    labels[:, :40], labels[:, 40:] = first, second
    return RunPreview(options["target_run"], np.full((60, 80, 3), (120, 90, 60), np.uint8),
        labels, np.zeros(labels.shape, bool), options["target_run"] + ".png",
        session.run_id, options["source_name"])


@pytest.mark.parametrize("mode", ["layer", "facies"])
def test_other_run_controls_redraw_cached_map_without_reading_photo_or_restarting_worker(view, qtbot, monkeypatch, mode):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    loaded = []
    def prepare(session, values, **options):
        loaded.append(options["target_run"])
        return _preview_packet(session, options, first, second)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    view.set_color_mode(mode)
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    assert not controller.progress_bar.isVisible()
    assert view.facies_visibility.isVisibleTo(view) and view.facies_visibility.isEnabled()
    assert view.facies_visibility.parentWidget() is controller.widget
    assert not view.steps_scroll.isEnabled()
    result, cached = controller.last_result, controller._last_session
    before = deepcopy(view.session.to_dict()), deepcopy(cached.to_dict()), result.layers.copy(), result.photo.copy()
    arrays = id(result.layers), id(result.photo), id(result.uncovered)
    note, transform = controller.note.text(), controller.canvas.transform()
    captured = []
    original = controller.canvas.set_proposal
    def overlay(value):
        captured.append(value.copy())
        original(value)
    monkeypatch.setattr(controller.canvas, "set_proposal", overlay)
    def forbidden(*args, **kwargs):
        pytest.fail("A visibility click attempted to load or place the preview again")
    monkeypatch.setattr(controller.canvas, "set_photo", forbidden)
    monkeypatch.setattr(controller, "request", forbidden)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", forbidden)
    view.set_layer_visible(first, False)
    view.facies_checks[name].click()
    assert np.all(captured[-1][..., 3] == 0)
    view.facies_checks[name].click()
    assert captured[-1][20, 10, 3] == 0 and captured[-1][20, 60, 3] == 100
    color = view._layer_colors()[second]
    np.testing.assert_array_equal(captured[-1][20, 60, :3], [color.red(), color.green(), color.blue()])
    assert view.hidden_layers == {first}
    assert loaded == ["target_a"] and controller.last_result is result and controller._last_session is cached
    assert arrays == (id(result.layers), id(result.photo), id(result.uncovered))
    assert view.session.to_dict() == before[0] and cached.to_dict() == before[1]
    np.testing.assert_array_equal(result.layers, before[2])
    np.testing.assert_array_equal(result.photo, before[3])
    assert controller.note.text() == note and controller.canvas.transform() == transform


def test_cached_colour_redraw_keeps_partial_progress_and_later_measurement_error(view, qtbot, monkeypatch):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(session, values, **options):
        calls.append(1)
        options["on_preview"](_preview_packet(session, options, first, second))
        options["on_progress"]("Measuring boundaries on the target photograph")
        entered.set()
        assert release.wait(10)
        raise OSError("boundary measurements unavailable")
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: controller.last_result is not None and "Measuring boundaries" in controller.note.text(), timeout=3000)
        stage = controller._stage
        view.facies_checks[name].click()
        assert controller._stage == stage and controller._loading_status
        assert controller.progress_bar.isVisible() and controller.thread is not None
        assert not controller.status_overlay.isVisible()
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        error = controller.note.text()
        assert "boundary measurements unavailable" in error
        view.facies_checks[name].click()
        assert controller.note.text() == error and not controller._loading_status
        assert not controller.progress_bar.isVisible()
        assert calls == [1] and controller.last_result is not None
    finally:
        release.set()
        controller.wait()


@pytest.mark.parametrize("partial", [False, True])
def test_reference_return_hides_shared_progress_while_same_preview_worker_finishes(view, qtbot, monkeypatch, partial):
    _name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(session, values, **options):
        calls.append(options["target_run"])
        result = _preview_packet(session, options, first, second)
        if partial:
            options["on_preview"](result)
        options["on_progress"]("Reading the selected photograph" if not partial else "Checking the shown map")
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", prepare)
    try:
        _select(controller)
        qtbot.waitUntil(entered.is_set, timeout=3000)
        qtbot.waitUntil(lambda: controller.progress_bar.isVisible(), timeout=3000)
        if partial:
            qtbot.waitUntil(lambda: controller.last_result is not None, timeout=3000)
            assert not controller.status_overlay.isVisible()
        worker = controller.worker
        controller.runs.setCurrentIndex(controller.runs.findData(view.session.run_id))
        assert controller.is_reference() and not controller.progress_bar.isVisible()
        assert "Reference run" in controller.note.text()
        assert controller.worker is worker and controller.thread is not None
        release.set()
        qtbot.waitUntil(lambda: controller.thread is None, timeout=5000)
        assert not controller.progress_bar.isVisible() and not controller._elapsed_timer.isActive()
        assert calls == ["target_a"]
    finally:
        release.set()
        controller.wait()
        assert not controller.progress_bar.isVisible()


def test_visibility_signal_refuses_stale_selection_and_deleted_target_canvas(view, qtbot, monkeypatch):
    name, first, second = _group(view)
    controller = _chooser(view, qtbot, monkeypatch)
    monkeypatch.setattr(preview_ui.preview, "prepare_preview", lambda session, values, **options:
        _preview_packet(session, options, first, second))
    _select(controller)
    qtbot.waitUntil(lambda: controller.thread is None and controller.last_result is not None, timeout=5000)
    rendered = []
    monkeypatch.setattr(controller.canvas, "set_proposal", lambda values: rendered.append(values))
    controller.runs.blockSignals(True)
    controller.runs.setCurrentIndex(controller.runs.findData("target_b"))
    controller.runs.blockSignals(False)
    view.facies_checks[name].click()
    assert rendered == []
    controller.canvas.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    view.facies_checks[name].click()
    controller.redraw_display()
    assert controller._disposed and rendered == []
