"""Compact Jobs controls and resource pages on local Qt fixtures.

The job list fills the left pane; the selected job's header, toolbar and detail
tabs fill the right. Sources are collapsible. No scroll area sits inside
another. The same widgets and actions as before; real Qt, no job is started.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSplitter, QWidget

from darsia.fluidflower.resource_advisor_panel import ResourceAdvisorPanel
from darsia.fluidflower.store_map_ui import StoreAvailabilityPanel, StoreSettingsDialog

from .test_native_jobs_owner import local_registry, snapshot, ui  # noqa: F401

UNCAPPED = 16777215  # QWIDGETSIZE_MAX


def _workspace(qtbot, local_registry):
    owner = ui.NativeJobsWorkspace(
        registry=local_registry,
        local_machine="local",
        reader=snapshot,
        inventory_provider=lambda: {},
        resource_factory=ResourceAdvisorPanel,
    )
    qtbot.addWidget(owner)
    return owner


def _ancestors(widget, stop):
    parent = widget.parentWidget()
    while parent is not None and parent is not stop:
        yield parent
        parent = parent.parentWidget()


def test_job_list_and_selected_job_share_a_splitter(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    assert isinstance(panel.splitter, QSplitter) and panel.splitter.count() == 2
    listing, details = panel.splitter.widget(0), panel.splitter.widget(1)
    assert listing in _ancestors(panel.job_table, panel)
    assert listing in _ancestors(panel.create_button, panel)
    for widget in (panel.master_label, panel.progress_label, panel.profile_label, panel.start_button,
                   panel.pause_button, panel.resume_button, panel.stop_button, panel.add_machine_button,
                   panel.declare_stopped_button, panel.detail_tabs):
        assert details in _ancestors(widget, panel)


def test_selected_job_has_separate_progress_and_resources_tabs(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    tabs = panel.detail_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["Progress", "Log", "Errors", "Files", "Resources"]
    assert tabs.widget(1) is panel.log_text and tabs.widget(2) is panel.error_text
    for widget in (panel.machine_table, panel.worker_table):
        assert tabs.widget(0) in _ancestors(widget, panel)
    assert tabs.widget(4) in _ancestors(panel.resource_panel, panel)
    assert not panel.resource_panel.findChildren(QScrollArea)
    for widget in (panel.capacity_button, panel.capacity_label):
        assert tabs.widget(3) in _ancestors(widget, panel)


def test_job_controls_use_compact_rows_and_fit_a_normal_window(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    owner.resize(1280, 900)
    owner.show()
    qtbot.waitUntil(owner.isVisible)
    panel = owner.panel
    assert owner.width() == 1280
    assert owner.minimumSizeHint().width() <= 1280
    assert panel.start_button.geometry().y() < panel.pause_button.geometry().y()
    assert panel.pause_button.geometry().y() == panel.stop_button.geometry().y()
    assert panel.stop_button.text() == "Stop job after current frame"
    assert panel.resume_button.text() == "Resume job"
    assert owner.new_analysis.start_button.text() == "Start workers for this job"
    for widget in (panel.start_button, panel.pause_button, panel.resume_button,
                   panel.stop_button, panel.add_machine_button, panel.declare_stopped_button):
        assert widget.geometry().right() < widget.parentWidget().width()


def test_start_controls_keep_the_borrowable_target_row(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    row = panel.start_button.parentWidget().layout()
    controls = [panel.target_combo, panel.backend_combo, panel.workers_spin, panel.start_button,
                panel.add_machine_button]
    positions = []
    for widget in controls:
        found = [index for index in range(row.count()) for item in [row.itemAt(index)]
                 if item.layout() is not None and item.layout().indexOf(widget) >= 0]
        positions.append(found)
    assert all(found == positions[0] for found in positions) and positions[0]


def test_empty_refresh_has_visible_feedback(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    qtbot.mouseClick(panel.refresh_button, Qt.MouseButton.LeftButton)
    assert "No jobs registered" in panel.progress_label.text()


def test_resource_actions_explain_refresh_copy_and_apply(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    resources = owner.resources
    rows = [dict(machine="local", inventory=dict(
        cpu=dict(physical_cores=8), memory=dict(available_bytes=120 * 1024**3)), history=[])]
    resources.set_context({"precision": "float32"}, machines_provider=lambda: rows)
    owner.resize(1280, 900)
    owner.show()
    owner.panel.detail_tabs.setCurrentWidget(owner.panel.resource_page)
    qtbot.mouseClick(resources.refresh_button, Qt.MouseButton.LeftButton)
    assert "Loaded 1 saved machine snapshot" in resources.probe_label.text()
    resources.policy_budget.setChecked(True)
    qtbot.mouseClick(resources.suggested_button, Qt.MouseButton.LeftButton)
    assert "Copied 4 worker(s)" in resources.selection_label.text()
    choices = []
    resources.workers_selected.connect(choices.append)
    qtbot.mouseClick(resources.use_button, Qt.MouseButton.LeftButton)
    assert "Applied 4 worker(s)" in resources.selection_label.text()
    assert choices[0]["workers"] == 4 and choices[0]["starts_job"] is False


def test_main_tables_log_and_error_text_have_no_height_cap(qtbot, local_registry):
    panel = _workspace(qtbot, local_registry).panel
    for widget in (panel.job_table, panel.machine_table, panel.worker_table, panel.log_text, panel.error_text):
        assert widget.maximumHeight() == UNCAPPED


def test_sources_are_collapsible_and_keep_their_actions(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    owner.show()
    assert owner.sources.title() == "Sources" and owner.sources.isCheckable()
    assert not owner.sources.isChecked() and not owner.sources_body.isVisible()
    for widget in (owner.stores, owner.open_job_button, owner.import_plan_button):
        assert owner.sources_body in _ancestors(widget, owner)
    owner.sources.setChecked(True)
    assert owner.sources_body.isVisible() and owner.open_job_button.isVisible()


def test_no_scroll_area_inside_another_on_the_jobs_page(qtbot, local_registry):
    owner = _workspace(qtbot, local_registry)
    for area in owner.findChildren(QScrollArea):
        assert not any(isinstance(parent, QScrollArea) for parent in _ancestors(area, None)), area


def test_store_details_and_preview_have_no_height_cap(qtbot, tmp_path):
    panel = StoreAvailabilityPanel(path=tmp_path / "stores.toml")
    qtbot.addWidget(panel)
    assert panel.details.maximumHeight() == UNCAPPED
    dialog = StoreSettingsDialog(path=tmp_path / "stores.toml")
    qtbot.addWidget(dialog)
    assert dialog.preview.maximumHeight() == UNCAPPED
    assert isinstance(dialog, QWidget)
