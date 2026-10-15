"""Native method selection: local manifests, real controls, no calculations."""

import copy
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, QSignalBlocker, Qt
from PySide6.QtWidgets import QMessageBox

from darsia.fluidflower.series_method import MethodSelection, save_series_method


@pytest.fixture
def method_window(qtbot, monkeypatch, tmp_path):
    from darsia.fluidflower import native_darsia_app as native

    old = (
        QSettings.defaultFormat(),
        QCoreApplication.organizationName(),
        QCoreApplication.applicationName(),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "settings"),
    )
    QCoreApplication.setOrganizationName("DarSIA-method-tests")
    QCoreApplication.setApplicationName(tmp_path.name)
    series = []
    for key in ("series-a", "series-b"):
        root = tmp_path / key
        root.mkdir()
        (root / "series.toml").write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            "[runs]\nids = []\n[future]\nopaque = [1, 2, 3]\n",
            encoding="utf-8",
        )
        (root / "calibration-original.json").write_text(
            '{"preserve":true}', encoding="utf-8"
        )
        series.append((key, key, root))
    monkeypatch.setattr(native, "configured_series", lambda: tuple(series))
    monkeypatch.setattr(
        native, "ProjectRegistry", lambda: SimpleNamespace(entries=lambda: [])
    )
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )
    monkeypatch.setattr(
        native.NativeFluidFlowerMainWindow,
        "ensure_mass_analysis",
        lambda self: self.mass_analysis_placeholder,
    )
    window = native.NativeFluidFlowerMainWindow(initial_series="series-a")
    qtbot.addWidget(window)
    window.show()
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *_a, **_k: pytest.fail("Unexpected numerical workflow"),
    )
    try:
        yield window
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old[1])
        QCoreApplication.setApplicationName(old[2])
        QSettings.setDefaultFormat(old[0])


def _roots(window):
    return window.phase_model_workspace.series_roots


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _phase_choice(root):
    save_series_method(
        root,
        MethodSelection(
            active="phase_model",
            model_version="model <v1>",
            mass_calibration="../fits/frozen.json",
        ),
    )


def test_method_choice_changes_only_manifest_and_survives_refresh_and_series_switch(
    method_window,
):
    window = method_window
    root = _roots(window)["series-a"]
    before = _files(root)
    config = copy.deepcopy(window.config_dict)
    window._select_row(("calibration", "color"))
    panel = window.method_panel
    panel.phase_model_button.click()
    panel.model_version.setText("model <v1>")
    panel.mass_calibration.setText("../fits/frozen.json")
    assert _files(root) == before
    panel.use_button.click()
    after = _files(root)
    assert set(after) == set(before)
    assert after["calibration-original.json"] == before["calibration-original.json"]
    manifest = tomllib.loads(after["series.toml"].decode("utf-8"))
    assert manifest["future"] == {"opaque": [1, 2, 3]}
    assert manifest["method"]["phase_model"]["model_version"] == "model <v1>"
    assert window.config_dict == config
    window._refresh_method_state(force=True)
    assert panel.selection().active == "phase_model"
    assert "model <v1>" in window.method_badge.text()
    assert window.method_badge.textFormat() == Qt.TextFormat.PlainText
    window._select_series("series-b")
    assert window._saved_method.active == ""
    assert window.method_badge.text() == "Method: not chosen"
    window._select_series("series-a")
    assert window._saved_method.active == "phase_model"
    assert _files(root) == after


def test_method_badge_is_visible_in_workflow_categories_and_analysis_repeats_it(
    method_window,
):
    window = method_window
    _phase_choice(_roots(window)["series-a"])
    window._refresh_method_state()
    assert list(window.sidebar._sections) == [
        "start",
        "setup",
        "calibration",
        "analysis",
        "comparison",
        "helper",
        "utils",
    ]
    for category in window.sidebar._sections:
        section = window.sidebar._sections[category]
        logical = window.sidebar._logical_selection(category, next(iter(section._rows)))
        window._select_row(logical)
        assert window.method_badge.isVisible() is (category not in {"start", "calibration"}), category
        assert "Method: phase model" in window.method_badge.text()
        assert window.method_panel.isVisible() is (category == "calibration")
        assert window.analysis_method_badge.isVisible() is (category == "analysis")
        if category == "start":
            assert not window.series_header.isVisible()
        if category == "analysis":
            assert window.analysis_method_badge.text() == window.method_badge.text()


def test_start_lists_each_series_saved_method_without_writing_or_opening_it(
    method_window,
):
    window = method_window
    roots = _roots(window)
    _phase_choice(roots["series-a"])
    before = {key: _files(root) for key, root in roots.items()}
    home = window.home_page
    home.refresh()
    assert home.table.horizontalHeaderItem(4).text() == "Method"
    assert not home.table.isColumnHidden(4)
    displayed = {
        home.table.item(row, 1).text(): home.table.item(row, 4).text()
        for row in range(home.table.rowCount())
    }
    assert displayed == {"series-a": "phase model", "series-b": "–"}
    assert window.content_stack.currentWidget() is home
    assert not window.method_badge.isVisible()
    assert not window.series_header.isVisible()
    assert {key: _files(root) for key, root in roots.items()} == before
    save_series_method(
        roots["series-b"],
        MethodSelection(active="colour_path", colour_calibration="colour"),
    )
    home.refresh()
    assert home.table.item(1, 4).text() == "colour path"


def test_unchanged_manifest_refresh_preserves_unsaved_panel_draft(method_window):
    window = method_window
    root = _roots(window)["series-a"]
    original = (root / "series.toml").read_bytes()
    window._select_row(("calibration", "color"))
    window.method_panel.phase_model_button.click()
    window.method_panel.model_version.setText("unsaved draft model")
    window.method_panel.mass_calibration.setText("relative unsaved fit")
    drafted = window.method_panel.selection()
    window._select_row(("setup", "depth"))
    window._select_row(("calibration", "color"))
    assert window.method_panel.selection() == drafted
    assert (root / "series.toml").read_bytes() == original
    assert window._saved_method.active == ""


@pytest.mark.parametrize("category", ["comparison", "helper", "utils"])
def test_original_category_header_hides_method_panel(method_window, qtbot, category):
    window = method_window
    window._select_row(("calibration", "color"))
    assert window.method_panel.isVisible()
    section = window.sidebar._sections[category]
    section.set_expanded(False)
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    assert not window.method_panel.isVisible()
    assert not window.analysis_method_badge.isVisible()
    assert window.method_badge.isVisible()


@pytest.mark.parametrize("method", ["colour_path", "phase_model"])
def test_mass_calibration_uses_one_readonly_display_and_cannot_launch(
    method_window, method
):
    window = method_window
    root = _roots(window)["series-a"]
    frozen = root / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "parameters": {"gain": 23.859, "beta": 0.6513},
                "metrics": {"mean_run_nmae": 0.02},
            }
        ),
        encoding="utf-8",
    )
    colour = root / "colour"
    (colour / "flash").mkdir(parents=True)
    (colour / "flash/flash.json").write_text('{"min_value_aq":0.15}', encoding="utf-8")
    for active in ("phase_model", "colour_path"):
        save_series_method(
            root,
            MethodSelection(
                active=active,
                model_version="v1",
                mass_calibration="frozen.json",
                colour_calibration="colour",
            ),
        )
    window._refresh_method_state(force=True)
    before = _files(root)
    config = copy.deepcopy(window.config_dict)
    key = (
        ("calibration", "mass")
        if method == "colour_path"
        else ("phase_model", "mass_from_phase")
    )
    window._select_row(key)
    panel = window.mass_calibration_display
    assert window.content_stack.currentWidget() is panel
    assert window.sidebar.get_selected() == key
    assert not panel.manual_button.isEnabled()
    assert not panel.automatic_button.isEnabled()
    assert "Jobs" in panel.mode_notice.text()
    assert not window.toolbar_builder.play_action.isEnabled()
    window.toolbar_builder.play_action.setEnabled(True)
    assert not window.toolbar_builder.play_action.isEnabled()
    window.run_selected_workflow()
    for hidden_action in ("mass", "mass_optuna", "default_mass"):
        window.selected_checkbox_id = hidden_action
        window.calibration_tab.on_run_clicked()
    assert window.config_dict == config
    assert _files(root) == before
    rows = {
        panel.parameters.item(i, 0).text(): panel.parameters.item(i, 1).text()
        for i in range(panel.parameters.rowCount())
    }
    if method == "phase_model":
        assert rows["Gain"] == "23.859"
        assert rows["Beta"] == "0.6513"
        assert "mean_run_nmae = 0.02" in rows["Saved mass-balance error"]
        assert "Not reported" in rows["Dilution factor"]
    else:
        assert rows["Flash min_value_aq"] == "0.15"
        assert "Not reported" in rows["Saved mass-balance error"]


def test_each_method_has_exactly_one_visible_mass_calibration_row(method_window):
    section = method_window.sidebar._sections["calibration"]
    rows = [
        row
        for row in section._rows.values()
        if row.label == "Mass calibration" and not row.isHidden()
    ]
    assert {(row.action, row.checkbox_id) for row in rows} == {
        ("calibration", "mass"),
        ("phase_model", "mass_from_phase"),
    }
    for key in ("mass_optuna", "default_mass"):
        assert section.get_row(key).isHidden()
        assert not section.get_row(key).isEnabled()


def test_mass_display_follows_series_switch_without_reusing_previous_values(
    method_window,
):
    window = method_window
    root = _roots(window)["series-a"]
    (root / "frozen.json").write_text(
        '{"parameters":{"gain":23.859,"beta":0.6513}}', encoding="utf-8"
    )
    save_series_method(
        root,
        MethodSelection(
            active="phase_model", model_version="v1", mass_calibration="frozen.json"
        ),
    )
    window._refresh_method_state(force=True)
    window._select_row(("phase_model", "mass_from_phase"))
    window._select_series("series-b")
    panel = window.mass_calibration_display
    assert window.content_stack.currentWidget() is panel
    assert "Not reported" in panel.parameters.item(0, 1).text()


def test_loaded_incomplete_choice_warns_without_fallback(method_window):
    window = method_window
    root = _roots(window)["series-a"]
    path = root / "series.toml"
    path.write_bytes(path.read_bytes() + b'\n[method]\nactive = "phase_model"\n')
    before = path.read_bytes()
    window._refresh_method_state()
    assert window._saved_method.active == "phase_model"
    assert "missing model version and mass calibration" in window.method_badge.text()
    assert window.method_panel.phase_model_button.isChecked()
    assert not window.method_panel.use_button.isEnabled()
    assert path.read_bytes() == before


def test_chosen_analysis_group_first_original_keys_and_header_default(
    method_window, qtbot
):
    window = method_window
    root = _roots(window)["series-a"]
    _phase_choice(root)
    window._refresh_method_state()
    section = window.sidebar._sections["analysis"]
    assert next(iter(section._rows)) == "mass_analysis:frozen"
    phase = section.get_row("mass_analysis:frozen")
    original = section.get_row("mass")
    assert (phase.action, phase.checkbox_id) == ("mass_analysis", "frozen")
    assert (original.action, original.checkbox_id) == ("analysis", "mass")
    assert "not the chosen method" not in phase.label
    assert "not the chosen method" in original.label
    section.set_expanded(False)
    qtbot.mouseClick(section.header_button, Qt.MouseButton.LeftButton)
    assert window.sidebar.get_selected() == ("mass_analysis", "frozen")
    save_series_method(
        root,
        MethodSelection(active="colour_path", colour_calibration="existing-colour-fit"),
    )
    window._refresh_method_state()
    assert next(iter(section._rows)) == "cropping"
    assert "not the chosen method" in phase.label
    assert "not the chosen method" not in original.label


@pytest.mark.parametrize("accept", [False, True])
def test_nonchosen_original_analysis_confirms_before_settings_sync_or_process(
    method_window, monkeypatch, accept
):
    window = method_window
    root = _roots(window)["series-a"]
    _phase_choice(root)
    window._refresh_method_state()
    before = _files(root)
    config_path = root / "unchanged-run.toml"
    window.config_path_label.setText(str(config_path))
    window.selected_action = "analysis"
    window.selected_checkbox_id = "mass"
    synced = []
    calls = []
    monkeypatch.setattr(
        window.settings_factory,
        "_sync_settings_inputs_to_config_dict",
        lambda: synced.append(True),
    )
    monkeypatch.setattr(window.settings_factory, "get_value", lambda *_: False)
    monkeypatch.setattr(
        window.process_runner,
        "start_workflow_process",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: (
            QMessageBox.StandardButton.Yes if accept else QMessageBox.StandardButton.No
        ),
    )
    window.analysis_tab.on_run_clicked()
    assert bool(synced) is accept
    assert bool(calls) is accept
    if accept:
        args, keywords = calls[0]
        assert args[0][-4:] == [
            "--config",
            str(config_path.resolve()),
            "--mass",
            "--progress",
        ]
        assert keywords["workflow"] == "analysis"
        assert keywords["actions"] == ["mass"]
    assert _files(root) == before
    assert window._saved_method.active == "phase_model"


def test_phase_export_no_prevents_original_export_and_yes_preserves_arguments(
    method_window, monkeypatch
):
    window = method_window
    calls = []
    saved = SimpleNamespace(export=lambda *a, **k: calls.append((a, k)))
    window._guard_phase_exports(SimpleNamespace(saved_results=saved))
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.No
    )
    saved.export("csv")
    assert not calls
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    saved.export("csv")
    assert calls == [(("csv",), {})]
    assert window._saved_method.active == ""


def test_method_references_are_only_explicit_manifest_choices_not_legacy_selections(
    method_window,
):
    window = method_window
    root = _roots(window)["series-a"]
    original = (root / "series.toml").read_bytes()
    # Retained viewers can hold old selectors after updating their series label.
    window.mass_analysis_page = SimpleNamespace(
        series="series-a",
        project_root=root,
        phase_catalog=None,
        model=SimpleNamespace(currentData=lambda: "old-other-series-model"),
        saved_results=SimpleNamespace(
            calibration_file=Path("old-other-series-fit.json")
        ),
    )
    window.phase_model_workspace.studio_page = SimpleNamespace(
        studio=SimpleNamespace(
            _active_model_series="series-a",
            _selected_validation_model=lambda: {"model_id": "old-model"},
        )
    )
    try:
        window._refresh_method_state(force=True)
        assert window.method_panel.selection() == MethodSelection()
        assert (root / "series.toml").read_bytes() == original
        _phase_choice(root)
        window._refresh_method_state(force=True)
        assert window.method_panel.selection().model_version == "model <v1>"
        assert window.method_panel.selection().mass_calibration == "../fits/frozen.json"
    finally:
        window.mass_analysis_page = None
        window.phase_model_workspace.studio_page = None


def test_original_compatibility_entry_opens_visible_readonly_mass_row(method_window):
    window = method_window
    window.show_original_workflow()
    assert window.sidebar.get_selected() == ("calibration", "mass")
    assert window.content_stack.currentWidget() is window.mass_calibration_display
    assert not window.toolbar_builder.play_action.isEnabled()


def test_new_series_records_explicitly_unchosen_method_without_parameters():
    from darsia.fluidflower.series_creation import _series_toml

    draft = SimpleNamespace(
        runs=[SimpleNamespace(run_id="ac52")],
        series_id="demo",
        name="Demo",
        rig_class="medium",
        created_by="test",
    )
    document = tomllib.loads(_series_toml(draft, "2026-10-15T00:00:00Z"))
    assert document["method"] == {"active": ""}
