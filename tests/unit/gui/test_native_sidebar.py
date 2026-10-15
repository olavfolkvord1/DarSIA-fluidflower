"""Navigation regressions for the FluidFlower extension of DarSIA's sidebar.

The real window and sidebar handle every selection. Expensive page loading is
replaced by empty local widgets, so these tests need no series, NAS or workflow.
"""

import pytest
from PySide6.QtCore import QCoreApplication, QSettings, Qt


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

