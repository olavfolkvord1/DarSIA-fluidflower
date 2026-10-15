"""The Calibration header stays concise without losing row controls."""

from .test_native_row_scope import scope_window
from .test_native_series_method import method_window


def test_one_fixed_line_and_original_row_controls(scope_window, qtbot):
    w = scope_window
    selector = w.run_combo
    w._select_row(("calibration", "color"))
    qtbot.wait(20)
    header = w.calibration_header
    assert header.line.isVisible()
    assert header.line.text() == "series-a · ac28 · Method not chosen"
    assert not w.method_badge.isVisible()
    assert not w.setup_series_label.isVisible()
    assert not w.run_path.isVisible()
    assert header.scroll.isVisible()
    assert header.scroll.isAncestorOf(w.series_header)
    assert header.scroll.isAncestorOf(w.method_panel)
    assert header.scroll.isAncestorOf(w.scope_label)
    assert header.scroll.isAncestorOf(w.workflow_status.panel)
    assert w.run_combo is selector
    assert w.run_combo.isVisible()
    assert w.scope_label.isVisible()
    assert "whole series" in w.scope_label.text()


def test_switch_rows_and_restore_other_categories_without_writes(scope_window, qtbot):
    w = scope_window
    roots = w.phase_model_workspace.series_roots.values()
    before = {
        str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()
    }
    for action, row in [
        ("calibration", "color"),
        ("calibration", "training"),
        ("phase_model", "mass_from_phase"),
        ("preprocessing", "protocols"),
        ("start", "series"),
    ]:
        w._select_row((action, row))
        qtbot.wait(20)
        active = action in {"calibration", "phase_model"}
        assert w.calibration_header.line.isVisible() is active
        assert w.calibration_header.scroll.isVisible() is active
        assert w.content_stack.currentWidget() is not None
    after = {
        str(p): p.read_bytes() for root in roots for p in root.rglob("*") if p.is_file()
    }
    assert after == before


def test_phase_calibration_has_one_numbered_page_entry(method_window):
    w = method_window
    section = w.sidebar._sections["calibration"]
    assert section.get_row("phase_model:training").label == "Phase model"
    assert section.get_row("phase_model:mass_from_phase").isHidden()
    w._select_row(("calibration", "color"))
    assert not w.method_panel.isVisible()
