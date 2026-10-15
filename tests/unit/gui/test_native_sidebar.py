"""Navigation regressions for the FluidFlower extension of DarSIA's sidebar.

The real window and sidebar handle every selection. Expensive page loading is
replaced by empty local widgets, so these tests need no series, NAS or workflow.
"""

import copy
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtWidgets import QLineEdit


@pytest.fixture
def native_window(qtbot, monkeypatch, tmp_path, request):
    from darsia.fluidflower import native_darsia_app as native

    # Keep startup/dock preferences local to this test, including settings read
    # by the unchanged base window and theme helpers.
    old_format = QSettings.defaultFormat()
    old_organization = QCoreApplication.organizationName()
    old_application = QCoreApplication.applicationName()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    QCoreApplication.setOrganizationName("DarSIA-sidebar-tests")
    QCoreApplication.setApplicationName(tmp_path.name)

    class EmptyProjects:
        def entries(self):
            return []

    series = []
    for key in getattr(request, "param", ()):
        root = tmp_path / key
        root.mkdir()
        (root / "series.toml").write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            "[runs]\nids = []\n",
            encoding="utf-8",
        )
        series.append((key, key, root))
    monkeypatch.setattr(native, "configured_series", lambda: tuple(series))
    monkeypatch.setattr(native, "ProjectRegistry", EmptyProjects)
    monkeypatch.setattr(
        native.PhaseModelWorkspace,
        "ensure_studio",
        lambda self: self.studio_placeholder,
    )
    monkeypatch.setattr(
        native.SeriesGeometryWorkspace, "ensure_page", lambda self: self
    )

    window = native.NativeFluidFlowerMainWindow()
    qtbot.addWidget(window)

    def unexpected_workflow(*args, **kwargs):
        pytest.fail("Sidebar navigation must not start a workflow")

    monkeypatch.setattr(
        window.process_runner, "start_workflow_process", unexpected_workflow
    )
    window.show()
    qtbot.waitUntil(window.isVisible)
    try:
        yield window
    finally:
        window.close()
        QCoreApplication.setOrganizationName(old_organization)
        QCoreApplication.setApplicationName(old_application)
        QSettings.setDefaultFormat(old_format)


def _custom_page(window, category):
    if category == "start":
        return window.home_page
    if category == "geometry_facies":
        return window.geometry_workspace
    if category == "phase_model":
        return window.phase_model_workspace
    return window.mass_analysis_placeholder


def test_sidebar_has_one_ordered_navigation_and_preserves_settings(native_window):
    window = native_window
    categories = window.sidebar.categories_data
    assert [entry[0].replace("&&", "&") for entry in categories.values()] == [
        "Start",
        "Preprocessing",
        "Setup",
        "Geometry & facies",
        "Calibration",
        "Phase model",
        "Analysis",
        "Mass analysis",
        "Comparison",
        "Helper",
        "Utils",
    ]
    assert window.content_stack.widget(0) is window.settings_scroll_area
    assert not hasattr(window, "workspace_tabs")
    assert not hasattr(window, "workspace_pages")
    assert window.settings_scroll_area.widget() is window.settings_container
    assert window.run_combo.isVisible()
    assert window.setup_series_label.isVisible()


@pytest.mark.parametrize(
    ("category", "row"),
    [
        ("start", "series"),
        ("geometry_facies", "rig_frame"),
        ("geometry_facies", "layers"),
        ("geometry_facies", "export"),
        ("phase_model", "training"),
        ("mass_analysis", "frozen"),
    ],
)
def test_custom_rows_return_to_real_preprocessing_settings(
    native_window, monkeypatch, category, row
):
    window = native_window
    displayed = []
    original_display = window.settings_factory.display_settings

    def record_display(action, selected):
        displayed.append((action, list(selected)))
        return original_display(action, selected)

    monkeypatch.setattr(window.settings_factory, "display_settings", record_display)
    if category == "phase_model":
        window.phase_model_workspace.select_calibration("mass")
    window._select_row((category, row))
    assert window.content_stack.currentWidget() is _custom_page(window, category)
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()
    assert window.run_combo.isVisible()
    assert window.setup_series_label.isVisible()
    if category == "phase_model":
        choices = window.phase_model_workspace.calibration_sidebar
        assert choices.get_selected() == ("calibration", "phase_model_training")
        window.show_mass_calibration()
        assert choices.get_selected() == ("calibration", "mass_from_phase")

    window._select_row(("preprocessing", "protocols"))
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert displayed[-1] == ("preprocessing", ["protocols"])
    assert window.sidebar.get_selected() == ("preprocessing", "protocols")
    assert window.toolbar_builder.wizard_action.isEnabled()
    assert window.settings_layout.count() > 0


@pytest.mark.parametrize(
    "category", ["start", "geometry_facies", "phase_model", "mass_analysis"]
)
def test_category_header_routes_page_and_resets_wizard_state(
    native_window, qtbot, category
):
    window = native_window
    window._select_row(("preprocessing", "protocols"))
    assert window.toolbar_builder.wizard_action.isEnabled()

    qtbot.mouseClick(
        window.sidebar._sections[category].header_button, Qt.MouseButton.LeftButton
    )
    assert window.content_stack.currentWidget() is _custom_page(window, category)
    assert window.selected_action == category
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()

    qtbot.mouseClick(
        window.sidebar._sections["preprocessing"].header_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert window.selected_action == "preprocessing"
    assert window.toolbar_builder.wizard_action.isEnabled()


def test_original_menus_and_dock_controls_survive_page_navigation(native_window):
    window = native_window
    menus = tuple(window.menuBar().actions())
    titles = {action.text().replace("&", "") for action in menus}
    assert {"File", "Settings", "Wizard", "Run", "View", "Help"} <= titles
    assert "preprocessing" in window.menu_builder.wizard_actions
    central = window.centralWidget()
    splitter = window.root_splitter
    log = window.log_dock
    stream = window.streaming_dock
    assert log.widget().isAncestorOf(window.log_text)
    assert stream.widget() is window.streaming_panel

    for selection in [
        ("geometry_facies", "layers"),
        ("phase_model", "training"),
        ("start", "series"),
        ("preprocessing", "protocols"),
    ]:
        window._select_row(selection)
        assert tuple(window.menuBar().actions()) == menus
        assert window.centralWidget() is central
        assert window.root_splitter is splitter
        for button, dock in [
            (window.log_toggle_button, log),
            (window.streaming_toggle_button, stream),
        ]:
            was_visible = dock.isVisible()
            button.click()
            assert dock.isVisible() is not was_visible
            button.click()
            assert dock.isVisible() is was_visible


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
@pytest.mark.parametrize("destination", ["geometry-row", "preprocessing-header"])
def test_start_series_selection_is_applied_before_showing_destination(
    native_window, qtbot, destination
):
    window = native_window
    assert window.phase_model_workspace.series_id == "series-a"
    table = window.home_page.table
    target = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(target)
    assert window.home_page.selected_series() == "series-b"
    shown_series = []
    window.content_stack.currentChanged.connect(
        lambda _index: shown_series.append(window.phase_model_workspace.series_id)
    )

    if destination == "geometry-row":
        window._select_row(("geometry_facies", "rig_frame"))
        assert window.content_stack.currentWidget() is window.geometry_workspace
    else:
        qtbot.mouseClick(
            window.sidebar._sections["preprocessing"].header_button,
            Qt.MouseButton.LeftButton,
        )
        assert window.content_stack.currentWidget() is window.settings_scroll_area

    assert shown_series and set(shown_series) == {"series-b"}
    assert window.phase_model_workspace.series_id == "series-b"
    assert window.geometry_workspace.series_id == "series-b"
    assert "series-b" in window.setup_series_label.text()


def test_collapsing_active_jakub_category_disables_play(native_window, qtbot):
    window = native_window
    window._select_row(("preprocessing", "protocols"))
    assert window.toolbar_builder.play_action.isEnabled()
    qtbot.mouseClick(
        window.sidebar._sections["preprocessing"].header_button,
        Qt.MouseButton.LeftButton,
    )
    assert window.selected_action is None
    assert not window.toolbar_builder.play_action.isEnabled()
    assert not window.toolbar_builder.wizard_action.isEnabled()


@pytest.mark.parametrize(
    "selection",
    [
        ("start", "series"),
        ("geometry_facies", "rig_frame"),
        ("phase_model", "training"),
        ("mass_analysis", "frozen"),
    ],
)
def test_workflow_completion_cannot_enable_play_on_custom_pages(
    native_window, selection
):
    window = native_window
    play = window.toolbar_builder.play_action
    window._select_row(selection)
    assert not play.isEnabled()

    # This is the completion handler's exact button-state operation, including
    # its abort/error paths, exercised without launching an actual process.
    play.setEnabled(True)
    assert not play.isEnabled()

    # The custom-page guard must leave Jakub's own completion behavior intact.
    window._select_row(("preprocessing", "protocols"))
    assert play.isEnabled()
    play.setEnabled(False)
    play.setEnabled(True)
    assert play.isEnabled()


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_start_jump_buttons_stay_hidden_after_refresh_and_series_selection(
    native_window,
):
    home = native_window.home_page

    def check_controls():
        assert home.open_selected_button.isHidden()
        assert home.update_run_sources_button.isHidden()
        for button in (
            home.new_series_button,
            home.add_existing_button,
            home.refresh_series_button,
        ):
            assert button.isVisible()
            assert button.isEnabled()

    check_controls()
    home.refresh()
    check_controls()
    target = next(
        row
        for row in range(home.table.rowCount())
        if home.table.item(row, 1).text() == "series-b"
    )
    home.table.selectRow(target)
    assert home.selected_series() == "series-b"
    check_controls()
    home.refresh_series_button.click()
    check_controls()


@pytest.mark.parametrize("native_window", [("series-a", "series-b")], indirect=True)
def test_continue_button_opens_training_for_selected_series(native_window, qtbot):
    window = native_window
    home = window.home_page
    target = next(
        row
        for row in range(home.table.rowCount())
        if home.table.item(row, 1).text() == "series-b"
    )
    home.table.selectRow(target)
    window.phase_model_workspace.select_calibration("mass")
    assert home.open_button.isVisible()
    assert home.open_button.isEnabled()
    qtbot.mouseClick(home.open_button, Qt.MouseButton.LeftButton)

    assert window.content_stack.currentWidget() is window.phase_model_workspace
    assert window.sidebar.get_selected() == ("phase_model", "training")
    assert window.phase_model_workspace.series_id == "series-b"
    assert window.phase_model_workspace.calibration_sidebar.get_selected() == (
        "calibration",
        "phase_model_training",
    )
    assert not window.toolbar_builder.play_action.isEnabled()


@pytest.fixture
def run_window(native_window, monkeypatch, tmp_path):
    """Two local series with distinct run-only and premerged configuration files."""
    from darsia.fluidflower import native_darsia_app as native

    roots = {}
    common = {}
    runs = {}
    merged = {}
    paths = []
    for series_index, key in enumerate(("series-a", "series-b")):
        root = tmp_path / key
        roots[key] = root
        for folder in ("config/runs", "config/merged"):
            (root / folder).mkdir(parents=True)
        manifest = root / "series.toml"
        manifest.write_text(
            f'schema_version = 1\nseries_id = "{key}"\nname = "{key}"\n'
            '[runs]\nids = ["first", "second"]\n',
            encoding="utf-8",
        )
        common[key] = root / "config/common.toml"
        common[key].write_text(
            f'common_marker = "{key}"\n[rig]\nheight = 0.5\n',
            encoding="utf-8",
        )
        paths.extend((manifest, common[key]))
        runs[key] = {}
        merged[key] = {}
        for run_index, run in enumerate(("first", "second")):
            width = 1.0 + 2.0 * series_index + run_index
            runs[key][run] = root / f"config/runs/{run}.toml"
            merged[key][run] = root / f"config/merged/{run}.toml"
            runs[key][run].write_text(
                f'run_marker = "{run}"\n[rig]\nwidth = {width}\n',
                encoding="utf-8",
            )
            merged[key][run].write_text(
                f'common_marker = "{key}"\nrun_marker = "{run}"\n'
                f"[rig]\nheight = 0.5\nwidth = {width}\n",
                encoding="utf-8",
            )
            paths.extend((runs[key][run], merged[key][run]))
    configured = tuple((key, key, root) for key, root in roots.items())
    monkeypatch.setattr(native, "configured_series", lambda: configured)
    warnings = []
    monkeypatch.setattr(
        native.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    native_window._refresh_series(preferred_series="series-a", show_errors=True)
    return SimpleNamespace(
        window=native_window,
        roots=roots,
        common=common,
        runs=runs,
        merged=merged,
        paths=paths,
        warnings=warnings,
    )


def _select_run(window, series, run):
    table = window.home_page.table
    row = next(
        row for row in range(table.rowCount()) if table.item(row, 1).text() == series
    )
    table.selectRow(row)
    assert window.phase_model_workspace.series_id == series
    index = window.run_combo.findText(run)
    assert index >= 0
    window.run_combo.setCurrentIndex(index)


def _wizard_dispatch_spy(monkeypatch):
    from darsia.gui.ui import main_window

    calls = []

    class WizardSpy:
        def __init__(self, parent):
            self.parent = parent

        def exec(self):
            calls.append(
                {
                    "path": self.parent.config_file,
                    "config": copy.deepcopy(self.parent.config_dict),
                    "selection": self.parent.sidebar.get_selected(),
                }
            )
            return 0

    def registered_dialog(category):
        assert category == "preprocessing"
        return WizardSpy

    monkeypatch.setattr(main_window, "load_wizard_dialog_class", registered_dialog)
    return calls


def _config_snapshot(window):
    return {
        "object": window.config_dict,
        "values": copy.deepcopy(window.config_dict),
        "file": window.config_file,
        "label": window.config_path_label.text(),
        "series": window._config_series,
        "stamp": window._config_stamp,
        "view": copy.deepcopy(window._last_settings_view),
    }


def _assert_config_restored(window, previous):
    assert window.config_dict is previous["object"]
    assert window.config_dict == previous["values"]
    assert window.config_file == previous["file"]
    assert window.config_path_label.text() == previous["label"]
    assert window._config_series == previous["series"]
    assert window._config_stamp == previous["stamp"]
    assert window._last_settings_view == previous["view"]


def test_setup_button_loads_selected_premerged_run_with_real_controller(
    run_window, monkeypatch, qtbot
):
    fixture = run_window
    window = fixture.window
    calls = _wizard_dispatch_spy(monkeypatch)
    _select_run(window, "series-b", "second")
    assert window.home_page.new_button.isEnabled()
    qtbot.mouseClick(window.home_page.new_button, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert Path(calls[0]["path"]) == fixture.merged["series-b"]["second"]
    assert calls[0]["config"]["common_marker"] == "series-b"
    assert calls[0]["config"]["run_marker"] == "second"
    assert calls[0]["config"]["rig"]["height"] == 0.5
    assert calls[0]["config"]["rig"]["width"] == 4.0
    assert calls[0]["selection"] == ("preprocessing", "protocols")
    assert window.content_stack.currentWidget() is window.settings_scroll_area
    assert fixture.warnings == []


@pytest.mark.parametrize("failure", ["malformed", "missing"])
def test_invalid_selected_run_never_opens_wizard_with_previous_config(
    run_window, monkeypatch, failure
):
    fixture = run_window
    window = fixture.window
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(fixture.merged["series-a"]["first"]))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    _select_run(window, "series-a", "second")
    assert window.home_page.new_button.isEnabled()
    target = fixture.merged["series-a"]["second"]
    if failure == "malformed":
        target.write_text('value = "unterminated', encoding="utf-8")
    else:
        target.unlink()
    before = {path: path.read_bytes() for path in fixture.paths if path.exists()}
    window.home_page.new_button.click()

    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    assert {path: path.read_bytes() for path in before} == before
    assert "Error loading config file:" in window.log_text.toPlainText()


def test_same_path_failed_reload_cannot_reuse_previously_parsed_values(
    run_window, monkeypatch
):
    fixture = run_window
    window = fixture.window
    target = fixture.merged["series-a"]["first"]
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(target))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    target.write_text('value = "unterminated', encoding="utf-8")
    window.home_page.new_button.click()

    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    assert target.read_text("utf-8") == 'value = "unterminated'


def test_post_assignment_load_failure_restores_old_config_and_editors(
    run_window, monkeypatch
):
    fixture = run_window
    window = fixture.window
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(fixture.merged["series-a"]["first"]))
    previous = _config_snapshot(window)
    calls = _wizard_dispatch_spy(monkeypatch)
    target = fixture.merged["series-a"]["second"]
    _select_run(window, "series-a", "second")
    original_refresh = window.refresh_sidebar_progress
    injected = []

    def fail_after_new_editors_are_built():
        if Path(window.config_file) == target and not injected:
            assert window.config_dict["rig"]["width"] == 2.0
            assert "rig.width" in window.settings_inputs
            injected.append(True)
            raise RuntimeError("injected failure after config and editor assignment")
        return original_refresh()

    monkeypatch.setattr(
        window, "refresh_sidebar_progress", fail_after_new_editors_are_built
    )
    before = {path: path.read_bytes() for path in fixture.paths}
    window.home_page.new_button.click()

    assert injected == [True]
    assert calls == []
    assert fixture.warnings
    _assert_config_restored(window, previous)
    # A later save reads these real rebuilt editors. They must not copy B's
    # width into the restored A dictionary, even after pending Qt fills run.
    window.settings_factory._sync_settings_inputs_to_config_dict()
    assert window.config_dict["rig"]["width"] == 1.0
    assert {path: path.read_bytes() for path in fixture.paths} == before


def test_real_wizard_finish_writes_only_the_selected_merged_file(
    run_window, monkeypatch
):
    from darsia.gui.ui import main_window, preprocessing_wizard
    from darsia.gui.ui.settings import unwrap_composite_widget

    fixture = run_window
    window = fixture.window
    _select_run(window, "series-b", "second")
    target = fixture.merged["series-b"]["second"]
    before = {path: path.read_bytes() for path in fixture.paths}
    dialog_class = preprocessing_wizard.PreprocessingWizardDialog
    monkeypatch.setattr(
        dialog_class,
        "_load_catalogue",
        lambda self: preprocessing_wizard.SeriesCatalogue(),
    )
    monkeypatch.setattr(main_window, "load_wizard_dialog_class", lambda _: dialog_class)
    finished = []

    def edit_and_finish(dialog):
        assert Path(dialog.main_window.config_file) == target
        rig_index = next(
            index
            for index, step in enumerate(preprocessing_wizard.STEPS)
            if step[0] == "rig"
        )
        dialog._goto(rig_index)
        width = unwrap_composite_widget(window.settings_inputs["rig.width"])
        assert isinstance(width, QLineEdit)
        width.setText("4.75")
        dialog._goto(len(preprocessing_wizard.STEPS) - 1)
        dialog._run_protocols_checkbox.setChecked(False)
        dialog._run_depth_measurements_checkbox.setChecked(False)
        dialog._finish()
        finished.append(dialog.result())
        return dialog.result()

    def unexpected_preprocessing(*args, **kwargs):
        pytest.fail("Finish with both compute checkboxes off must only save")

    monkeypatch.setattr(dialog_class, "exec", edit_and_finish)
    monkeypatch.setattr(dialog_class, "_start_preprocessing", unexpected_preprocessing)
    window.home_page.new_button.click()

    assert finished == [1]
    saved = tomllib.loads(target.read_text("utf-8"))
    assert saved["rig"]["width"] == 4.75
    assert saved["rig"]["height"] == 0.5
    assert saved["common_marker"] == "series-b"
    assert saved["run_marker"] == "second"
    assert target.read_bytes() != before[target]
    assert all(
        path.read_bytes() == raw for path, raw in before.items() if path != target
    )
    assert fixture.warnings == []


def test_unchanged_open_run_keeps_pending_widget_edit_for_real_wizard(
    run_window, monkeypatch
):
    from darsia.gui.ui import main_window, preprocessing_wizard
    from darsia.gui.ui.settings import unwrap_composite_widget

    fixture = run_window
    window = fixture.window
    target = fixture.merged["series-a"]["first"]
    _select_run(window, "series-a", "first")
    assert window._load_run_config(str(target))
    QCoreApplication.processEvents()
    width = unwrap_composite_widget(window.settings_inputs["rig.width"])
    assert isinstance(width, QLineEdit)
    width.setText("1.75")
    before = {path: path.read_bytes() for path in fixture.paths}
    dialog_class = preprocessing_wizard.PreprocessingWizardDialog
    monkeypatch.setattr(
        dialog_class,
        "_load_catalogue",
        lambda self: preprocessing_wizard.SeriesCatalogue(),
    )
    monkeypatch.setattr(main_window, "load_wizard_dialog_class", lambda _: dialog_class)
    seen = []

    def inspect_then_cancel(dialog):
        # The unchanged real constructor has already synchronized the fields.
        seen.append(
            (
                Path(dialog.main_window.config_file),
                dialog.main_window.config_dict["rig"],
            )
        )
        dialog.reject()
        return dialog.result()

    monkeypatch.setattr(dialog_class, "exec", inspect_then_cancel)
    window.home_page.new_button.click()

    assert seen and seen[0][0] == target
    assert seen[0][1]["width"] == 1.75
    assert window.config_dict["rig"]["width"] == 1.75
    assert {path: path.read_bytes() for path in fixture.paths} == before
    assert fixture.warnings == []


def test_setup_button_tracks_selected_series_run_and_missing_file(run_window):
    fixture = run_window
    window = fixture.window
    window.run_combo.setCurrentIndex(-1)
    assert not window.home_page.new_button.isEnabled()
    assert window.home_page.new_button.toolTip()

    _select_run(window, "series-a", "first")
    assert window.home_page.new_button.isEnabled()
    table = window.home_page.table
    row = next(
        row
        for row in range(table.rowCount())
        if table.item(row, 1).text() == "series-b"
    )
    table.selectRow(row)
    assert window.phase_model_workspace.series_id == "series-b"
    assert "series-b" in window.setup_series_label.text()
    assert window.run_combo.currentIndex() == -1
    assert not window.home_page.new_button.isEnabled()

    _select_run(window, "series-b", "second")
    assert window.home_page.new_button.isEnabled()
    fixture.merged["series-b"]["second"].unlink()
    window._update_setup_action()
    assert not window.home_page.new_button.isEnabled()
    assert window.home_page.new_button.toolTip()


def test_split_config_without_merged_stays_disabled_until_common_is_empty(run_window):
    fixture = run_window
    window = fixture.window
    fixture.merged["series-a"]["first"].unlink()
    window._populate_runs()
    first = window.run_combo.findText("first")
    assert not window.run_combo.model().item(first).isEnabled()
    assert not window.run_combo.itemData(first)
    window.run_combo.setCurrentIndex(first)
    assert not window.home_page.new_button.isEnabled()
    assert "no merged file" in window.run_combo.itemData(first, Qt.ToolTipRole)

    fixture.common["series-a"].write_text(
        "# No common settings yet\n", encoding="utf-8"
    )
    window._populate_runs()
    first = window.run_combo.findText("first")
    assert window.run_combo.model().item(first).isEnabled()
    assert Path(window.run_combo.itemData(first)) == fixture.runs["series-a"]["first"]
    window.run_combo.setCurrentIndex(first)
    assert window.home_page.new_button.isEnabled()
