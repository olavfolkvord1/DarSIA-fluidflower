"""Native DarSIA navigation for original and phase-model FluidFlower workflows.

Run with ``python -m darsia.fluidflower.native_darsia_app``.  The ordinary
``python -m darsia.gui`` entry point and its main window remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.project import ProjectRegistry
from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace
from darsia.fluidflower.series_paths import configured_series
from darsia.fluidflower.ui.home import HomePage
from darsia.gui.ui.main_window import MainWindow
from darsia.gui.ui.sidebar import Sidebar
from darsia.gui.ui.theme import apply_theme, get_theme


class _NativeSeriesRegistry:
    """Present current series and earlier project packages on Start."""

    def __init__(
        self,
        series: Sequence[tuple[str, str, Path]],
        *,
        project_registry: ProjectRegistry | None = None,
    ) -> None:
        self.project_registry = project_registry or ProjectRegistry()
        self.replace(series)

    def replace(self, series: Sequence[tuple[str, str, Path]]) -> None:
        self.series = tuple(series)
        self.series_by_path = {str(root): key for key, _label, root in self.series}

    def forget_missing(self) -> None:
        # An unavailable folder must remain visible so users can reconnect it.
        pass

    def entries(self) -> list[dict[str, str]]:
        current = [
            {
                "name": label,
                "series_id": key,
                "run_count": self._run_count(root),
                "path": str(root),
                "template_id": "Series folder",
            }
            for key, label, root in self.series
        ]
        self.legacy_by_path: dict[str, dict] = {}
        seen_paths = {str(root).casefold() for _key, _label, root in self.series}
        for project in self.project_registry.entries():
            path = str(project.get("path") or "")
            if not path or path.casefold() in seen_paths:
                continue
            seen_paths.add(path.casefold())
            self.legacy_by_path[path] = project
            label = str(project.get("name") or project.get("series_id") or path)
            current.append(
                {
                    "name": f"{label} · earlier project",
                    "series_id": str(project.get("series_id") or ""),
                    "run_count": str(project.get("run_count") or ""),
                    "path": path,
                    "template_id": "Earlier project",
                    "last_opened_utc": str(project.get("last_opened_utc") or ""),
                }
            )
        return current

    @staticmethod
    def _run_count(root: Path) -> str:
        try:
            with (root / "series.toml").open("rb") as stream:
                runs = tomllib.load(stream).get("runs", {})
        except (OSError, ValueError):
            return ""
        ids = runs.get("ids") if isinstance(runs, dict) else None
        return str(len(ids)) if isinstance(ids, list) else ""


class _NoLegacyDrafts:
    """Configured series do not use the former series.json draft store."""

    def drafts(self) -> list:
        return []


class NativeHomePage(HomePage):
    """Start page for current series folders and earlier project packages."""

    def __init__(
        self,
        *,
        series: Sequence[tuple[str, str, Path]] | None = None,
        initial_series: str | None = None,
        project_registry: ProjectRegistry | None = None,
    ) -> None:
        self.series_registry = _NativeSeriesRegistry(
            configured_series() if series is None else series,
            project_registry=project_registry,
        )
        self.initial_series = initial_series or (
            self.series_registry.series[0][0] if self.series_registry.series else ""
        )
        super().__init__(self.series_registry, draft_store=_NoLegacyDrafts())
        self.new_button.setText("Set up an experiment")
        self.open_button.setText("Continue phase and mass work")
        self.open_selected_button.setText("Open phase model training")
        self.update_run_sources_button.setText("Open experiment setup")
        self.add_runs_button.hide()
        self.draft_heading.hide()
        self.draft_note.hide()
        self.table.setHorizontalHeaderLabels(
            ["Series", "ID", "Runs", "Type", "Last opened", "Folder"]
        )
        self.table.setColumnHidden(4, True)
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.package_note.setText(
            "Series folders contain current configuration, reviews, models and "
            "results. Earlier project packages open in their original workspace."
        )
        self.intro.setText(
            "Choose an experimental series, then set up its inputs or "
            "continue phase-model and mass calibration."
        )
        self.intro.setWordWrap(True)
        self.add_existing_button = QPushButton("Add existing series folder…")
        self.add_existing_button.setToolTip(
            "Register a series folder or its DarSIA child containing series.toml. "
            "The folder may be local or on a shared drive."
        )
        self.refresh_series_button = QPushButton("Refresh series")
        self.refresh_series_button.setToolTip(
            "Reload registered series and earlier projects from local settings."
        )
        add_row = QHBoxLayout()
        add_row.addWidget(self.add_existing_button)
        add_row.addWidget(self.refresh_series_button)
        add_row.addStretch(1)
        self.layout().insertLayout(2, add_row)
        if not self.series_registry.series:
            self.package_note.setText(
                "No experimental series is configured. Add an existing DarSIA "
                "series folder, or use Setup to begin a new experiment."
            )
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == self.initial_series:
                self.table.selectRow(row)
                break

    def selected_series(self) -> str:
        if self.selected_legacy_project():
            return ""
        return self.series_registry.series_by_path.get(
            self._selected_project_path(), self.initial_series
        )

    def selected_legacy_project(self) -> str:
        path = self._selected_project_path()
        return path if path in self.series_registry.legacy_by_path else ""

    def _project_selection_changed(self) -> None:
        super()._project_selection_changed()
        legacy = bool(self.selected_legacy_project())
        self.new_button.setText(
            "Edit earlier project setup" if legacy else "Set up an experiment"
        )
        self.open_button.setText(
            "Continue earlier project" if legacy else "Continue phase and mass work"
        )
        self.open_selected_button.setText(
            "Open earlier project" if legacy else "Open phase model training"
        )
        self.update_run_sources_button.setText(
            "Open project setup" if legacy else "Open experiment setup"
        )

    def update_series(
        self, series: Sequence[tuple[str, str, Path]], *, selected: str
    ) -> None:
        self.series_registry.replace(series)
        self.initial_series = selected
        self.refresh()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == selected:
                self.table.selectRow(row)
                break
        if series:
            self.package_note.setText(
                "Series folders contain current configuration, reviews, models and "
                "results. Earlier project packages open in their original workspace."
            )


class PhaseModelWorkspace(QWidget):
    """Training and calibration backed by one lazily loaded Studio instance."""

    def __init__(
        self,
        *,
        series: Sequence[tuple[str, str, Path]] | None = None,
        initial_series: str | None = None,
    ) -> None:
        super().__init__()
        self.configured_series = tuple(
            configured_series() if series is None else series
        )
        self.series_roots = {key: root for key, _label, root in self.configured_series}
        initial_series = initial_series or (
            self.configured_series[0][0] if self.configured_series else ""
        )
        self.setObjectName("phaseModelWorkspace")
        self.studio_page: QWidget | None = None
        self._requested_mode = "training"

        layout = QVBoxLayout(self)
        heading = QLabel("Phase-model workflow · training and calibration")
        heading_font = heading.font()
        heading_font.setBold(True)
        heading.setFont(heading_font)
        layout.addWidget(heading)

        controls = QFormLayout()
        self.series_selector = QComboBox()
        for key, label, _root in self.configured_series:
            self.series_selector.addItem(label, key)
        index = self.series_selector.findData(initial_series)
        if index >= 0:
            self.series_selector.setCurrentIndex(index)
        elif initial_series:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        controls.addRow("Experimental series", self.series_selector)

        self.series_folder = QLineEdit()
        self.series_folder.setReadOnly(True)
        self.series_folder.setToolTip("Configured DarSIA folder for this series")
        controls.addRow("Series folder", self.series_folder)
        layout.addLayout(controls)
        self.series_support_status = QLabel()
        self.series_support_status.setWordWrap(True)
        self.series_support_status.hide()
        layout.addWidget(self.series_support_status)

        workspace = QHBoxLayout()
        self.calibration_sidebar = Sidebar(
            {
                "calibration": (
                    "Phase-model calibration",
                    "fa5s.balance-scale",
                    [
                        (
                            "Phase-model methods",
                            [
                                (
                                    "Phase model training",
                                    "phase_model_training",
                                    "fa5s.circle",
                                    "Review evidence, train models and inspect validation.",
                                ),
                                (
                                    "Mass calibration from phase model",
                                    "mass_from_phase",
                                    "fa5s.circle",
                                    "Fit and inspect mass calibration from a selected phase model.",
                                ),
                            ],
                        )
                    ],
                )
            }
        )
        self.calibration_sidebar.setMinimumWidth(230)
        self.calibration_sidebar.setMaximumWidth(340)
        self.calibration_sidebar._sections["calibration"].set_expanded(True)
        mass_row = self.calibration_sidebar._sections["calibration"].get_row(
            "mass_from_phase"
        )
        mass_row._text_label.setWordWrap(True)
        mass_row.setMinimumHeight(40)
        self.calibration_sidebar.select("calibration", "phase_model_training")
        self.calibration_sidebar.selection_changed.connect(self._calibration_selected)
        workspace.addWidget(self.calibration_sidebar)

        studio_panel = QWidget()
        self.studio_host = QVBoxLayout(studio_panel)
        self.studio_host.setContentsMargins(0, 0, 0, 0)
        self.studio_placeholder = QLabel(
            "Choose phase-model training or mass calibration."
        )
        self.studio_placeholder.setWordWrap(True)
        self.studio_host.addWidget(self.studio_placeholder)
        workspace.addWidget(studio_panel, 1)
        layout.addLayout(workspace, 1)

        self._selected_series = initial_series
        self._update_series_folder()
        self.series_selector.currentIndexChanged.connect(self._series_changed)

    @property
    def series_id(self) -> str:
        return str(self.series_selector.currentData() or "")

    def _update_series_folder(self) -> None:
        root = self.series_roots.get(self.series_id)
        self.series_folder.setText(str(root) if root is not None else "")

    def update_series(
        self, series: Sequence[tuple[str, str, Path]], *, selected: str
    ) -> None:
        self.configured_series = tuple(series)
        self.series_roots = {key: root for key, _label, root in self.configured_series}
        self.series_selector.blockSignals(True)
        self.series_selector.clear()
        for key, label, _root in self.configured_series:
            self.series_selector.addItem(label, key)
        index = self.series_selector.findData(selected)
        if index >= 0:
            self.series_selector.setCurrentIndex(index)
        self.series_selector.blockSignals(False)
        self._series_changed(-1)

    def ensure_studio(self) -> QWidget:
        """Create the adapter once; it creates SharedPhaseModelTrainingPage on show."""
        if not self.series_id:
            self.studio_placeholder.setText(
                "Add an existing experimental series folder on Start before "
                "opening phase-model training or mass calibration."
            )
            return self.studio_placeholder
        if self.studio_page is None:
            from darsia.fluidflower.pages.phase_interpretation.subpages.train_model import (
                PhaseModelStudioSubpage,
            )

            self.studio_page = PhaseModelStudioSubpage(
                self.series_id, project_root=self.series_folder.text()
            )
            self.studio_page.studio_loaded.connect(self._verify_studio_series)
            self.studio_page.studio_loaded.connect(self._show_selected_studio_mode)
            if self.studio_placeholder is not None:
                self.studio_host.removeWidget(self.studio_placeholder)
                self.studio_placeholder.deleteLater()
                self.studio_placeholder = None
            self.studio_host.addWidget(self.studio_page)
        return self.studio_page

    def _bind_studio_series(self, studio: QWidget) -> None:
        """Keep an unsupported series from operating on a different model filter."""
        try:
            studio.set_model_series(self.series_id)
        except ValueError:
            self.studio_page.setEnabled(False)
            self.series_support_status.setText(
                "The phase-model Studio does not yet support this series. "
                "Its training and mass controls are disabled for this selection."
            )
            self.series_support_status.show()
        else:
            self.studio_page.setEnabled(True)
            self.series_support_status.clear()
            self.series_support_status.hide()

    def _verify_studio_series(self) -> None:
        studio = getattr(self.studio_page, "studio", None)
        if studio is not None and getattr(
            studio, "_active_model_series", self.series_id
        ) != self.series_id:
            self._bind_studio_series(studio)

    def _calibration_selected(self, _action: str, item_id: str) -> None:
        self._requested_mode = "mass" if item_id == "mass_from_phase" else "training"
        if self.isVisible():
            self.ensure_studio()
        self._show_selected_studio_mode()

    def select_calibration(self, mode: str) -> None:
        if mode not in {"training", "mass"}:
            raise ValueError(f"Unknown calibration mode: {mode}")
        item_id = {
            "training": "phase_model_training",
            "mass": "mass_from_phase",
        }[mode]
        self.calibration_sidebar.select("calibration", item_id)

    def _show_selected_studio_mode(self) -> None:
        """Navigate existing Studio tabs without duplicating its model/job state."""
        if self.studio_page is not None:
            self.studio_page.show()
        studio = getattr(self.studio_page, "studio", None)
        tabs = getattr(studio, "tabs", None)
        mass_tab = getattr(studio, "mass_workspace_tab", None)
        if not isinstance(tabs, QTabWidget) or mass_tab is None:
            return
        mass_index = tabs.indexOf(mass_tab)
        if mass_index < 0:
            return
        if self._requested_mode == "mass":
            tabs.setTabVisible(mass_index, True)
            tabs.setCurrentIndex(mass_index)
            for index in range(mass_index):
                tabs.setTabVisible(index, False)
            tabs.tabBar().hide()
        else:
            for index in range(mass_index):
                tabs.setTabVisible(index, True)
            if tabs.currentIndex() == mass_index:
                tabs.setCurrentWidget(studio.evidence_tab)
            tabs.setTabVisible(mass_index, False)
            tabs.tabBar().show()

    def _series_changed(self, _index: int) -> None:
        page = self.studio_page
        studio = getattr(page, "studio", None)
        if studio is not None:
            # A running job must retain its series context until it finishes.
            notice = page._active_job_notice(studio)
            if notice is not None:
                QMessageBox.information(self, *notice)
                self.series_selector.blockSignals(True)
                self.series_selector.setCurrentIndex(
                    self.series_selector.findData(self._selected_series)
                )
                self.series_selector.blockSignals(False)
                return

        self._selected_series = self.series_id
        self._update_series_folder()
        if page is None:
            return
        page.series_id = self.series_id
        page.project_root = Path(self.series_folder.text())
        page._mass_series_bound = False
        if studio is not None:
            if hasattr(studio, "set_project_root"):
                studio.set_project_root(page.project_root, fixed=True)
            self._bind_studio_series(studio)
            workspace = getattr(studio, "_mass_workspace", None)
            control = getattr(workspace, "series", None)
            if control is not None:
                index = control.findData(self.series_id)
                if index >= 0:
                    control.setCurrentIndex(index)

class NativeFluidFlowerMainWindow(MainWindow):
    """One native DarSIA workspace with contextual setup and distinct methods."""

    _START_TAB = 0
    _SETUP_TAB = 1
    _GEOMETRY_TAB = 2
    _ORIGINAL_TAB = 3
    _PHASE_MODEL_TAB = 4
    _MASS_ANALYSIS_TAB = 5

    def __init__(self, *, initial_series: str | None = None) -> None:
        super().__init__()
        series = tuple(configured_series())
        initial_series = initial_series or (series[0][0] if series else "")
        if initial_series and initial_series not in {
            key for key, _label, _root in series
        }:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        original_content = self.takeCentralWidget()
        if original_content is None:
            raise RuntimeError("DarSIA MainWindow has no central widget")
        self.workspace_tabs = QTabBar(self)
        self.workspace_tabs.setObjectName("nativeWorkspaceTabs")
        self.workspace_tabs.setExpanding(False)
        self.workspace_tabs.addTab("Start")
        self.workspace_tabs.addTab("Setup")
        self.workspace_tabs.addTab("Geometry and facies")
        self.workspace_tabs.setTabToolTip(
            self._GEOMETRY_TAB,
            "Sand layers, facies groups and properties of the selected series",
        )
        self.workspace_tabs.addTab("Original DarSIA workflow")
        self.workspace_tabs.addTab("Phase-model workflow")
        self.workspace_tabs.addTab("Mass analysis")
        self.workspace_tabs.setTabVisible(self._SETUP_TAB, False)
        self.workspace_pages = QStackedWidget(self)
        self.home_page = NativeHomePage(series=series, initial_series=initial_series)
        self._legacy_windows: dict[str, QWidget] = {}
        self.workspace_pages.addWidget(self.home_page)
        self.native_page = QWidget()
        native_layout = QVBoxLayout(self.native_page)
        native_layout.setContentsMargins(8, 8, 8, 8)
        setup_controls = QHBoxLayout()
        self.setup_series_label = QLabel()
        setup_controls.addWidget(self.setup_series_label)
        self.image_selection_button = QPushButton("Image selection plan…")
        self.image_selection_button.setToolTip(
            "Choose frames for review, mass calibration or production; does not start a job"
        )
        setup_controls.addWidget(self.image_selection_button)
        self.geometry_button = QPushButton("Geometry and facies…")
        self.geometry_button.setToolTip(
            "Mark sand layers, group them into facies and export labels and facies "
            "for Setup"
        )
        setup_controls.addWidget(self.geometry_button)
        setup_controls.addStretch(1)
        self.setup_controls_widget = QWidget()
        self.setup_controls_widget.setLayout(setup_controls)
        native_layout.addWidget(self.setup_controls_widget)
        self.original_heading = QLabel(
            "Original DarSIA workflow · color-path calibration and analysis"
        )
        heading_font = self.original_heading.font()
        heading_font.setBold(True)
        self.original_heading.setFont(heading_font)
        native_layout.addWidget(self.original_heading)
        native_layout.addWidget(original_content, 1)
        self.workspace_pages.addWidget(self.native_page)
        self.geometry_workspace = SeriesGeometryWorkspace(
            series=series, initial_series=initial_series
        )
        self.workspace_pages.addWidget(self.geometry_workspace)
        self.sidebar.setMinimumWidth(180)
        original_calibration = self.sidebar._sections["calibration"]
        original_calibration.category_label = "Original calibration"
        original_calibration.refresh_style()
        self._last_setup_selection = ("setup", "protocols")
        self._last_original_selection = ("calibration", "color")
        self.sidebar.selection_changed.connect(self._remember_native_selection)
        self.phase_model_workspace = PhaseModelWorkspace(
            series=series, initial_series=initial_series
        )
        self.workspace_pages.addWidget(self.phase_model_workspace)
        self.mass_analysis_page: QWidget | None = None
        self._sync_series_display()
        self.phase_model_workspace.series_selector.currentIndexChanged.connect(
            self._sync_series_display
        )
        self.image_selection_button.clicked.connect(self.open_image_selection_plan)
        self.geometry_button.clicked.connect(self.show_geometry_facies)
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.workspace_tabs)
        central_layout.addWidget(self.workspace_pages, 1)
        self.setCentralWidget(central)

        self.home_page.new_requested.connect(self.show_setup)
        self.home_page.add_existing_button.clicked.connect(self._add_existing_series)
        self.home_page.refresh_series_button.clicked.connect(
            lambda: self._refresh_series(show_errors=True)
        )
        self.home_page.update_run_sources_requested.connect(self._show_setup_for_path)
        self.home_page.open_requested.connect(self._show_home_calibration)
        self.home_page.recent_requested.connect(self._show_training_for_path)
        self._workspace_changed(self.workspace_tabs.currentIndex())

    def _series_for_path(self, path: str) -> str:
        return self.home_page.series_registry.series_by_path.get(
            path, self.home_page.selected_series()
        )

    def _refresh_series(
        self, *, preferred_series: str | None = None, show_errors: bool = False
    ) -> None:
        """Reload user settings when returning to Start or requesting a refresh."""
        try:
            series = tuple(configured_series())
        except (OSError, ValueError, KeyError) as error:
            if show_errors:
                QMessageBox.warning(self, "Could not refresh series", str(error))
            return
        workspace = self.phase_model_workspace
        if series != workspace.configured_series:
            page = workspace.studio_page
            studio = getattr(page, "studio", None)
            notice = page._active_job_notice(studio) if studio is not None else None
            if notice is not None:
                if show_errors:
                    QMessageBox.information(self, *notice)
                return
        selected_path = self.home_page._selected_project_path()
        selected = preferred_series or workspace.series_id
        if selected not in {key for key, _label, _root in series}:
            selected = series[0][0] if series else ""
        self.home_page.update_series(series, selected=selected)
        if series != workspace.configured_series:
            workspace.update_series(series, selected=selected)
        self.geometry_workspace.update_series(series, selected=workspace.series_id)
        self._sync_series_display()
        if preferred_series is None and selected_path:
            for row in range(self.home_page.table.rowCount()):
                item = self.home_page.table.item(row, 0)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == selected_path:
                    self.home_page.table.selectRow(row)
                    break

    def _open_legacy_project(self, path: str, *, setup: bool = False) -> None:
        """Open an existing series.json package in its original project UI."""
        if path not in self.home_page.series_registry.legacy_by_path:
            return
        window = self._legacy_windows.get(path)
        if window is None:
            from darsia.fluidflower.ui.shell import FluidFlowerMainWindow

            window = FluidFlowerMainWindow(
                registry=self.home_page.series_registry.project_registry,
                initial_project=Path(path),
            )
            if window.workspace is None:
                window.close()
                return
            self._legacy_windows[path] = window
        if setup and window.workspace is not None:
            window.workspace.select_module("setup")
        window.show()
        window.raise_()
        window.activateWindow()

    def _select_series(self, series_id: str) -> None:
        selector = self.phase_model_workspace.series_selector
        index = selector.findData(series_id)
        if index >= 0:
            selector.setCurrentIndex(index)

    def _sync_series_display(self, _index: int = -1) -> None:
        selector = self.phase_model_workspace.series_selector
        label = selector.currentText()
        self.setup_series_label.setText(
            f"Selected series: {label}" if label else "No series selected"
        )
        for row in range(self.home_page.table.rowCount()):
            if self.home_page.table.item(row, 1).text() == selector.currentData():
                self.home_page.table.selectRow(row)
                break
        self.geometry_workspace.set_series(self.phase_model_workspace.series_id)
        if self.mass_analysis_page is not None:
            series_id = self.phase_model_workspace.series_id
            root = self.phase_model_workspace.series_roots.get(series_id)
            if root is not None and (
                self.mass_analysis_page.series != series_id
                or self.mass_analysis_page.project_root != Path(root)
            ):
                self.mass_analysis_page.set_series(series_id, root)

    def ensure_mass_analysis(self) -> QWidget:
        """Create production mass analysis as its own top-level workspace."""
        if self.mass_analysis_page is None:
            from darsia.fluidflower.tools.phase_studio_mass_analysis_workspace import (
                MassAnalysisWorkspace,
            )

            series_id = self.phase_model_workspace.series_id
            root = self.phase_model_workspace.series_roots.get(series_id)
            if root is None:
                raise ValueError("Choose an experimental series before mass analysis")
            self.mass_analysis_page = MassAnalysisWorkspace(
                series=series_id, project_root=root
            )
            self.mass_analysis_page.series_selected.connect(
                self._mass_analysis_series_selected
            )
            self.workspace_pages.addWidget(self.mass_analysis_page)
        return self.mass_analysis_page

    def _mass_analysis_series_selected(self, series_id: str) -> None:
        self._select_series(series_id)
        # Studio can reject a series change while training. Keep both pages
        # on the same series in that case.
        actual = self.phase_model_workspace.series_id
        if actual != series_id and self.mass_analysis_page is not None:
            root = self.phase_model_workspace.series_roots.get(actual)
            if root is not None:
                self.mass_analysis_page.set_series(actual, root)

    def _add_existing_series(self) -> None:
        page = self.phase_model_workspace.studio_page
        studio = getattr(page, "studio", None)
        if studio is not None:
            notice = page._active_job_notice(studio)
            if notice is not None:
                QMessageBox.information(self, *notice)
                return
        folder = QFileDialog.getExistingDirectory(
            self, "Select a DarSIA series folder", str(Path.home())
        )
        if not folder:
            return
        from darsia.fluidflower.series_paths import register_series

        try:
            series_id, _label, _root = register_series(Path(folder))
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self,
                "Could not add series",
                f"{error}\n\nSelect an existing folder containing series.toml. "
                "Use Setup to begin a new experiment.",
            )
            return
        self._refresh_series(preferred_series=series_id, show_errors=True)

    def open_image_selection_plan(self) -> None:
        """Open the metadata-only image plan from the setup workspace."""
        from darsia.fluidflower.pages.image_correction.selection_dialog import (
            ImageSelectionPlanDialog,
        )
        from darsia.fluidflower.series_paths import series_root

        series_id = self.phase_model_workspace.series_id
        dialog = ImageSelectionPlanDialog(
            series_root(series_id), self, series_id=series_id
        )
        self.selection_dialog = dialog
        dialog.exec()

    def _show_setup_for_path(self, path: str) -> None:
        if path in self.home_page.series_registry.legacy_by_path:
            self._open_legacy_project(path, setup=True)
            return
        self._select_series(self._series_for_path(path))
        self.show_setup()

    def _show_training_for_path(self, path: str) -> None:
        if path in self.home_page.series_registry.legacy_by_path:
            self._open_legacy_project(path)
            return
        self._select_series(self._series_for_path(path))
        self.show_phase_model_workflow()

    def _show_home_calibration(self) -> None:
        legacy_path = self.home_page.selected_legacy_project()
        if legacy_path:
            self._open_legacy_project(legacy_path)
            return
        self._select_series(self.home_page.selected_series())
        self.show_phase_model_workflow()

    def show_setup(self) -> None:
        """Reveal the original setup only after an explicit start-page choice."""
        legacy_path = self.home_page.selected_legacy_project()
        if legacy_path:
            self._open_legacy_project(legacy_path, setup=True)
            return
        self._select_series(self.home_page.selected_series())
        self.workspace_tabs.setTabVisible(self._SETUP_TAB, True)
        self.workspace_tabs.setCurrentIndex(self._SETUP_TAB)

    def show_original_workflow(self) -> None:
        self.workspace_tabs.setCurrentIndex(self._ORIGINAL_TAB)

    def show_geometry_facies(self) -> None:
        """Open geometry and facies for the series selected on Start."""
        self.workspace_tabs.setCurrentIndex(self._GEOMETRY_TAB)

    def _remember_native_selection(self, action: str, item_id: str) -> None:
        if action == "setup":
            self._last_setup_selection = (action, item_id)
        else:
            self._last_original_selection = (action, item_id)

    def _show_native_mode(self, *, setup: bool) -> None:
        self.workspace_pages.setCurrentWidget(self.native_page)
        self.setup_controls_widget.setVisible(setup)
        self.original_heading.setVisible(not setup)
        for action, section in self.sidebar._sections.items():
            section.setVisible((action == "setup") == setup)
        selection = (
            self._last_setup_selection if setup else self._last_original_selection
        )
        self.sidebar._sections[selection[0]].set_expanded(True)
        self.sidebar.select(*selection)

    def _workspace_changed(self, index: int) -> None:
        if index == self._START_TAB:
            self.workspace_pages.setCurrentWidget(self.home_page)
            self._refresh_series()
        elif index == self._SETUP_TAB:
            self._show_native_mode(setup=True)
        elif index == self._GEOMETRY_TAB:
            selected = self.home_page.selected_series()
            if selected and selected != self.phase_model_workspace.series_id:
                self._select_series(selected)
            self.workspace_pages.setCurrentWidget(self.geometry_workspace)
            self.geometry_workspace.ensure_page()
        elif index == self._ORIGINAL_TAB:
            self._show_native_mode(setup=False)
        elif index == self._PHASE_MODEL_TAB:
            self.workspace_pages.setCurrentWidget(self.phase_model_workspace)
            self.phase_model_workspace.ensure_studio()
            self.phase_model_workspace._show_selected_studio_mode()
        elif index == self._MASS_ANALYSIS_TAB:
            self.workspace_pages.setCurrentWidget(self.ensure_mass_analysis())
        if index != self._SETUP_TAB:
            self.workspace_tabs.setTabVisible(self._SETUP_TAB, False)
        self.toolbar_builder.play_action.setEnabled(
            index in (self._SETUP_TAB, self._ORIGINAL_TAB)
            and not self.process_runner.active
        )

    def run_selected_workflow(self) -> None:
        """The original Run action applies only while its workflow is visible."""
        if self.workspace_tabs.currentIndex() in (
            self._SETUP_TAB,
            self._ORIGINAL_TAB,
        ):
            super().run_selected_workflow()

    def show_phase_model_workflow(self) -> None:
        self.phase_model_workspace.select_calibration("training")
        self.workspace_tabs.setCurrentIndex(self._PHASE_MODEL_TAB)

    def show_mass_calibration(self) -> None:
        self.phase_model_workspace.select_calibration("mass")
        self.workspace_tabs.setCurrentIndex(self._PHASE_MODEL_TAB)

    def show_mass_analysis(self) -> None:
        self.workspace_tabs.setCurrentIndex(self._MASS_ANALYSIS_TAB)


def create_application() -> tuple[QApplication, bool]:
    """Use Jakub's saved theme and an existing QApplication when embedded."""
    existing = QApplication.instance()
    app = existing if existing is not None else QApplication(sys.argv[:1])
    if existing is None:
        app.setOrganizationName("DarSIA")
        app.setApplicationName("DarSIA GUI")
    apply_theme(app, get_theme())
    return app, existing is None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series", choices=[key for key, _label, _root in configured_series()]
    )
    parser.add_argument(
        "--tab",
        choices=(
            "home", "setup", "geometry", "facies", "original", "phase", "mass",
            "analysis", "mass-analysis", "darsia", "calibration", "fluidflower",
        ),
        default="home",
    )
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--exit-after-ms", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app, owns_app = create_application()
    window = NativeFluidFlowerMainWindow(initial_series=args.series)
    if args.tab == "setup":
        window.show_setup()
    elif args.tab in {"geometry", "facies"}:
        window.show_geometry_facies()
    elif args.tab in {"original", "darsia"}:
        window.show_original_workflow()
    elif args.tab in {"phase", "calibration", "fluidflower"}:
        window.show_phase_model_workflow()
    elif args.tab == "mass":
        window.show_mass_calibration()
    elif args.tab in {"analysis", "mass-analysis"}:
        window.show_mass_analysis()
    window.show()
    # Keep a reference when called from an already running Qt application.
    app._native_fluidflower_window = window

    if args.screenshot is not None:
        target = args.screenshot.expanduser().resolve()

        def save_screenshot() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            app.processEvents()
            window.repaint()
            if not window.grab().save(str(target)):
                raise RuntimeError(f"Could not save screenshot to {target}")
            if not args.exit_after_ms:
                app.quit()

        # The Studio performs its first model-catalog discovery on opening.
        # Let that finish before a QA screenshot of the phase-model tab.
        QTimer.singleShot(
            5000 if args.tab in {"phase", "calibration", "fluidflower", "mass", "analysis", "mass-analysis"} else 500,
            save_screenshot,
        )
    if args.exit_after_ms:
        QTimer.singleShot(max(1, args.exit_after_ms), app.quit)
    return int(app.exec()) if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
