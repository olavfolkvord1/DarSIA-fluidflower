"""Native DarSIA navigation for original and phase-model FluidFlower workflows.

Run with ``python -m darsia.fluidflower.native_darsia_app``.  The ordinary
``python -m darsia.gui`` entry point and its main window remain unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.depth_setup_ui import NativeDepthWorkflow
from darsia.fluidflower.crop_setup_ui import NativeCropWorkflow
from darsia.fluidflower.crop_wizard_ui import prepare_crop_wizard
from darsia.fluidflower.project import ProjectRegistry
from darsia.fluidflower.series_geometry import series_rig_class
from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace
from darsia.fluidflower.series_paths import configured_series
from darsia.fluidflower.series_method import (
    MethodSelection,
    SeriesMethodPanel,
    missing_method_references,
    read_series_method,
    save_series_method,
)
from darsia.fluidflower.templates import RIG_CLASSES, rig_class_rank
from darsia.fluidflower.ui.home import HomePage
from darsia.gui.ui.main_window import MainWindow
from darsia.gui.ui.sidebar import GroupHeaderLabel, Sidebar, SidebarRow
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
        """Series folders, then earlier projects, each sorted by rig class."""

        current = []
        for key, label, root in self.series:
            rig_class = series_rig_class(root)
            current.append(
                {
                    "name": label,
                    "series_id": key,
                    "run_count": self._run_count(root),
                    "path": str(root),
                    "template_id": _rig_class_text(rig_class),
                    "rig_class": rig_class,
                }
            )
        current.sort(key=_by_rig_class)
        self.legacy_by_path: dict[str, dict] = {}
        seen_paths = {str(root).casefold() for _key, _label, root in self.series}
        legacy = []
        for project in self.project_registry.entries():
            path = str(project.get("path") or "")
            if not path or path.casefold() in seen_paths:
                continue
            seen_paths.add(path.casefold())
            self.legacy_by_path[path] = project
            label = str(project.get("name") or project.get("series_id") or path)
            rig_class = _project_rig_type(path)
            legacy.append(
                {
                    "name": f"{label} · earlier project",
                    "series_id": str(project.get("series_id") or ""),
                    "run_count": str(project.get("run_count") or ""),
                    "path": path,
                    "template_id": _rig_class_text(rig_class),
                    "rig_class": rig_class,
                    "last_opened_utc": str(project.get("last_opened_utc") or ""),
                }
            )
        legacy.sort(key=_by_rig_class)
        self.last_entries = current + legacy
        return self.last_entries

    @staticmethod
    def _run_count(root: Path) -> str:
        try:
            with (root / "series.toml").open("rb") as stream:
                runs = tomllib.load(stream).get("runs", {})
        except (OSError, ValueError):
            return ""
        ids = runs.get("ids") if isinstance(runs, dict) else None
        return str(len(ids)) if isinstance(ids, list) else ""


def _rig_class_text(key: str) -> str:
    rig_class = RIG_CLASSES.get(key)
    return rig_class.short if rig_class is not None else "Not set"


def _by_rig_class(row: dict[str, str]) -> tuple[int, str]:
    return rig_class_rank(row.get("rig_class", "")), str(row.get("name", "")).casefold()


def _project_rig_type(path: str) -> str:
    """Apparatus class recorded in an earlier project package ("" if unknown)."""

    import json

    try:
        manifest = json.loads((Path(path) / "series.json").read_text("utf-8-sig"))
    except (OSError, ValueError):
        return ""
    rig = manifest.get("rig") if isinstance(manifest, dict) else None
    value = str(rig.get("type") or "") if isinstance(rig, dict) else ""
    return value if value in RIG_CLASSES else ""


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
        self.new_button.hide()
        self.open_button.setText("Open series")
        self.open_button.hide()
        self.open_selected_button.setText("Open phase model training")
        self.open_selected_button.hide()
        self.update_run_sources_button.setText("Open experiment setup")
        self.update_run_sources_button.hide()
        self.add_runs_button.hide()
        self.draft_heading.hide()
        self.draft_note.hide()
        self.table.setHorizontalHeaderLabels(
            ["Series", "ID", "Runs", "Rig class", "Method", "Folder"]
        )
        # Open only on a user click, never on programmatic selection/refresh.
        self.table.cellClicked.connect(self._open_row)
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.package_note.setText(
            "Series folders contain current configuration, reviews, models and "
            "results. Earlier project packages open in their original workspace."
        )
        self.intro.setText(
            "Choose an existing experimental series or create a new series."
        )
        self.intro.setWordWrap(True)
        self.new_series_button = QPushButton("New series…")
        self.new_series_button.setToolTip(
            "Create a series folder for new experiments: name, rig class, image "
            "folders and baseline photographs of each run. Images are not copied."
        )
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
        add_row.addWidget(self.new_series_button)
        add_row.addWidget(self.add_existing_button)
        add_row.addWidget(self.refresh_series_button)
        add_row.addStretch(1)
        self.layout().insertLayout(2, add_row)
        if not self.series_registry.series:
            self.package_note.setText(
                "No experimental series is configured. Add an existing DarSIA "
                "series folder, or choose New series… to create one."
            )
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == self.initial_series:
                self.table.selectRow(row)
                break

    def refresh(self) -> None:
        super().refresh()
        rows = getattr(self.series_registry, "last_entries", [])
        for index, row in enumerate(rows[: self.table.rowCount()]):
            method_item = self.table.item(index, 4)
            root = row.get("path", "")
            choice = (
                read_series_method(Path(root))
                if root in self.series_registry.series_by_path
                else MethodSelection()
            )
            method_item.setText(
                {"phase_model": "phase model", "colour_path": "colour path"}.get(
                    choice.active, "–"
                )
            )
            method_item.setToolTip(
                choice.error or "Chosen method saved in series.toml."
            )
            item = self.table.item(index, 3)
            rig = RIG_CLASSES.get(row.get("rig_class", ""))
            if item is None:
                continue
            if rig is None:
                item.setToolTip(
                    "No rig class recorded. Choose one in Setup > Geometry and "
                    "facies; it is stored in series.toml."
                )
            else:
                item.setToolTip(rig.title() + (f". {rig.source}" if rig.source else ""))

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
        self.open_button.setText("Open series")
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
        if (
            studio is not None
            and getattr(studio, "_active_model_series", self.series_id)
            != self.series_id
        ):
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


def _same_executable(first: str | Path, second: str | Path) -> bool:
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return str(first) == str(second)


def use_environment_python(runner) -> None:
    """Start the original DarSIA workflows with the environment's interpreter.

    The Windows launcher runs this application with a base CPython and adds the
    environment's site-packages at run time, so ``sys.executable`` cannot import
    darsia. The original DarSIA pages start Setup, calibration and analysis as
    ``sys.executable -m darsia...``; this swaps that interpreter for the one of
    the environment (``DARSIA_FLUIDFLOWER_PYTHON`` or ``VIRTUAL_ENV``).
    """

    from darsia.fluidflower.phase_workflow import phase_python_executable

    start = runner.start_workflow_process

    def start_workflow_process(argv, *args, **kwargs):
        argv = [str(value) for value in argv]
        if argv and _same_executable(argv[0], sys.executable):
            argv[0] = phase_python_executable()
        return start(argv, *args, **kwargs)

    start_workflow_process.__wrapped__ = start
    runner.start_workflow_process = start_workflow_process


def clear_config_placeholder(window) -> None:
    """Let the original pages see that no configuration is open.

    DarSIA's main window keeps the open configuration file in a hidden label
    whose text is "No config loaded." until a file is opened, while Setup,
    calibration, analysis and the helpers treat only an empty label (or "No
    file chosen") as missing. ▶ before File > Open Config… would therefore
    start the workflow with that text as the file name and end in a
    traceback. An empty label makes them ask for a configuration instead.
    """

    if not window.config_file:
        window.config_path_label.setText("")


def _read_toml(path: Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _has_settings(path: Path) -> bool:
    """Whether a TOML file holds any key or table (comments do not count)."""
    try:
        return bool(_read_toml(path))
    except FileNotFoundError:
        return False
    except (OSError, tomllib.TOMLDecodeError):
        return True


def run_configurations(root: str | Path) -> list[tuple[str, Path | None, str]]:
    """The file the settings panel opens for each run of a series.

    DarSIA's settings panel opens one configuration file, while a series keeps
    the settings of a run in ``config/common.toml`` and
    ``config/runs/<run>.toml``. A merged single file in ``config/merged`` holds
    both (FF_AC, FF_UM and FF_BM); without one, the run file serves alone when
    the common file holds no settings (FF_RE). Returns ``(run, file, note)``
    in the order of ``series.toml``; ``file`` is None when no single file holds
    the run's settings, and ``note`` says why.
    """

    root = Path(root)
    try:
        manifest = _read_toml(root / "series.toml")
    except (OSError, tomllib.TOMLDecodeError):
        return []
    paths = manifest.get("paths") or {}
    runs = manifest.get("runs") or {}
    files = runs.get("config_files") or {}
    merged_folder = root / str(paths.get("merged_configs", "config/merged"))
    common = root / str(paths.get("common_config", "config/common.toml"))
    run_folder = str(paths.get("run_configs", "config/runs"))
    common_has_settings: bool | None = None
    choices: list[tuple[str, Path | None, str]] = []
    for run in [str(value) for value in runs.get("ids") or []]:
        merged = merged_folder / f"{run}.toml"
        if merged.is_file():
            choices.append(
                (run, merged, "Merged configuration (common and run settings)")
            )
            continue
        name = str(files.get(run) or f"{run_folder}/{run}.toml")
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file():
            choices.append((run, None, f"{name} does not exist"))
            continue
        if common_has_settings is None:
            common_has_settings = _has_settings(common)
        if common_has_settings:
            choices.append(
                (
                    run,
                    None,
                    f"The settings are split between {common.name} and {name}, the "
                    "settings panel opens one file, and there is no merged file "
                    f"{merged.relative_to(root).as_posix()}",
                )
            )
        else:
            choices.append((run, path, "Run configuration"))
    return choices


def _same_file(first: str | Path | None, second: str | Path | None) -> bool:
    if not first or not second:
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
            os.path.abspath(second)
        )


def _file_stamp(path: str | Path) -> tuple[int, int] | None:
    try:
        stat = Path(path).stat()
    except (OSError, ValueError):
        return None
    return stat.st_mtime_ns, stat.st_size


def _ask_config_reload(parent: QWidget, path: str) -> str:
    """Ask what to do with a configuration that changed on disk.

    Returns ``"reload"``, ``"overwrite"`` or ``"cancel"``.
    """

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Configuration changed on disk")
    box.setText(
        f"{Path(path).name} changed on disk after it was opened here, for example "
        "because a script regenerated it."
    )
    box.setInformativeText(
        "Reload it and run with the settings in the file, or keep the settings "
        "shown here and write them over the file."
    )
    reload_button = box.addButton("Reload and run", QMessageBox.ButtonRole.AcceptRole)
    keep_button = box.addButton(
        "Overwrite and run", QMessageBox.ButtonRole.DestructiveRole
    )
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(reload_button)
    box.exec()
    clicked = box.clickedButton()
    if clicked is reload_button:
        return "reload"
    if clicked is keep_button:
        return "overwrite"
    return "cancel"


def guard_config_file(window) -> None:
    """Keep the settings panel from writing an old copy over a changed file.

    DarSIA's settings panel writes the configuration it shows to the opened
    file before every workflow. When the file changed on disk after it was
    opened, for example because a series script regenerated it, that would put
    the older copy back. The window therefore records the file's modification
    time and size whenever it opens or saves the file, and before a workflow
    whose file has changed since, it asks whether to reload the file or to
    overwrite it.
    """

    controller = window.config_controller
    factory = window.settings_factory
    runner = window.process_runner
    load = controller.load_config
    save = factory.save_settings
    start = runner.start_workflow_process

    def record() -> None:
        path = window.config_file
        stamp = _file_stamp(path) if path else None
        window._config_stamp = (str(path), stamp) if stamp is not None else None

    def load_config(*args, **kwargs):
        result = load(*args, **kwargs)
        record()
        hook = getattr(window, "_config_loaded", None)
        if callable(hook):
            hook()
        return result

    def save_settings(*args, **kwargs):
        result = save(*args, **kwargs)
        record()
        return result

    def start_workflow_process(argv, *args, **kwargs):
        path = window.config_file
        recorded = getattr(window, "_config_stamp", None)
        if path and recorded is not None and recorded[0] == str(path):
            current = _file_stamp(path)
            if current is not None and current != recorded[1]:
                choice = _ask_config_reload(window, str(path))
                if choice == "cancel":
                    window.print_log(
                        "Run cancelled: the configuration changed on disk."
                    )
                    return None
                if choice == "reload":
                    selection = window.sidebar.get_selected()
                    controller.load_config()
                    if selection is not None:
                        window.sidebar.select(*selection)
                    window.print_log(f"Reloaded {path}, which changed on disk.")
        return start(argv, *args, **kwargs)

    load_config.__wrapped__ = load
    save_settings.__wrapped__ = save
    start_workflow_process.__wrapped__ = start
    controller.load_config = load_config
    factory.save_settings = save_settings
    runner.start_workflow_process = start_workflow_process
    # File > Save Config was connected to the unwrapped method.
    menu = getattr(window, "menu_builder", None)
    save_action = getattr(menu, "save_action", None)
    if save_action is not None:
        try:
            save_action.triggered.disconnect()
        except (RuntimeError, TypeError):
            pass
        save_action.triggered.connect(lambda _checked=False: factory.save_settings())
    window._config_stamp = None


def settle_settings_before_sync(window) -> None:
    """Let DarSIA fill its settings fields before their values are read back.

    DarSIA's settings panel fills several kinds of fields one event-loop turn
    after it builds them (``QTimer.singleShot(0, ...)``): data selections, the
    per-folder imaging and blacklist protocols and the facies label lists.
    Rebuilding the panel (selecting a sidebar row) or saving first copies every
    field back into the configuration. Read in the same turn as the build, those
    fields are still empty, so the copy drops data selections, all but the first
    protocol folder and the facies sections, and the next save writes that to
    the file. DarSIA's own flow never hits this, because a person clicks after
    the event loop has run; this window opens a run's file and selects its step
    in one call. Processing the pending events (not user input) before each copy
    makes the copy read filled fields.
    """

    factory = window.settings_factory
    sync = factory._sync_settings_inputs_to_config_dict

    def _sync_settings_inputs_to_config_dict(*args, **kwargs):
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        return sync(*args, **kwargs)

    _sync_settings_inputs_to_config_dict.__wrapped__ = sync
    factory._sync_settings_inputs_to_config_dict = _sync_settings_inputs_to_config_dict


# --------------------------------------------------------------------------- #
# Mass calibration: manual (DarSIA's sliders) or automatic (Optuna)
# --------------------------------------------------------------------------- #

#: Sidebar row of the automatic (Optuna) mass calibration, next to DarSIA's
#: interactive "mass" row in the calibration section.
OPTUNA_MASS_ROW = "mass_optuna"
_OPTUNA_HELP = (
    "Automatic mass calibration: an Optuna search for the signal values of the "
    "sand facies that makes the detected CO2 mass match the injected mass in the "
    "photographs of [calibration.mass] data_selection. It starts from the stored "
    "colour-to-mass calibration (and creates Default mass first when there is "
    "none), keeps the flash, saves the best values in the same place and keeps "
    "the previous calibration, every trial and a mass-balance figure in "
    "mass_calibration_optuna next to the colour path. Uses the same settings as "
    "Mass, manual."
)


class AutomaticMassCalibrationDialog(QDialog):
    """Options of an automatic (Optuna) mass calibration before it starts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mass, Optuna (automatic calibration)")
        layout = QVBoxLayout(self)
        text = QLabel(
            "Searches the signal values of the sand facies so that the detected CO2 "
            "mass matches the injected mass in the [calibration.mass] photographs. "
            "The best values replace the stored calibration; the previous one and a "
            "report are kept in mass_calibration_optuna."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        form = QFormLayout()
        settings = QSettings()
        self.trials = QSpinBox()
        self.trials.setRange(10, 5000)
        self.trials.setSingleStep(50)
        self.trials.setValue(settings.value("native_app/optuna/trials", 200, type=int))
        self.trials.setToolTip(
            "Optuna trials (a fraction of a second each), followed by a short local "
            "refinement of the best one"
        )
        form.addRow("Trials", self.trials)
        self.values = QComboBox()
        self.values.addItem("Shared by the sand facies", False)
        self.values.addItem("Separate per sand facies", True)
        self.values.setCurrentIndex(
            1 if settings.value("native_app/optuna/per_facies", False, type=bool) else 0
        )
        self.values.setToolTip(
            "Shared: one signal function for all sands with a colour path (robust "
            "when a sand holds little CO2 in the photographs). Separate: one per sand."
        )
        form.addRow("Signal values", self.values)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(settings.value("native_app/optuna/seed", 0, type=int))
        self.seed.setToolTip("Seed of the sampler; the same seed repeats a run")
        form.addRow("Seed", self.seed)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> dict:
        return {
            "trials": self.trials.value(),
            "per_facies": bool(self.values.currentData()),
            "seed": self.seed.value(),
        }

    def remember(self) -> None:
        settings = QSettings()
        options = self.options()
        settings.setValue("native_app/optuna/trials", options["trials"])
        settings.setValue("native_app/optuna/per_facies", options["per_facies"])
        settings.setValue("native_app/optuna/seed", options["seed"])


def _ask_optuna_options(parent: QWidget) -> dict | None:
    dialog = AutomaticMassCalibrationDialog(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    dialog.remember()
    return dialog.options()


def optuna_mass_calibration_argv(config_file: str | Path, options: dict) -> list[str]:
    """Command line of darsia.fluidflower.native_mass_calibration."""
    argv = [
        sys.executable,
        "-m",
        "darsia.fluidflower.native_mass_calibration",
        "--config",
        str(Path(config_file).resolve()),
        "--trials",
        str(int(options["trials"])),
        "--seed",
        str(int(options["seed"])),
    ]
    if options.get("per_facies"):
        argv.append("--per-facies")
    return argv


def add_automatic_mass_calibration(window) -> None:
    """Offer the mass calibration as a choice: manual (sliders) or automatic.

    DarSIA's calibration section has one interactive "Mass" row. This renames it
    "Mass, manual" and adds "Mass, Optuna" (automatic) below it, which shows the
    same settings and runs :mod:`darsia.fluidflower.native_mass_calibration`
    through DarSIA's process runner (log, stop button and done dialog as for the
    other steps).
    """

    section = window.sidebar._sections["calibration"]
    manual = section.get_row("mass")
    if manual is None or section.get_row(OPTUNA_MASS_ROW) is not None:
        return
    manual.label = "Mass, manual"
    manual._text_label.setText(manual.label)
    row = SidebarRow(
        "calibration",
        OPTUNA_MASS_ROW,
        "Mass, Optuna",
        manual.icon_name,
        _OPTUNA_HELP,
    )
    row.clicked.connect(lambda: section._on_row_clicked(OPTUNA_MASS_ROW))
    section.items_layout.insertWidget(section.items_layout.indexOf(manual) + 1, row)
    # Keep the row table in screen order (code iterates it for dots and rows).
    rows = dict(section._rows)
    rows[OPTUNA_MASS_ROW] = row
    section._rows = {
        widget.checkbox_id: widget
        for widget in (
            section.items_layout.itemAt(i).widget()
            for i in range(section.items_layout.count())
        )
        if isinstance(widget, SidebarRow) and widget.checkbox_id in rows
    }

    factory = window.settings_factory
    display = factory.display_settings

    def display_settings(action, checked_ids):
        if action == "calibration":
            checked_ids = ["mass" if i == OPTUNA_MASS_ROW else i for i in checked_ids]
        return display(action, checked_ids)

    display_settings.__wrapped__ = display
    factory.display_settings = display_settings

    tab = window.calibration_tab
    run = tab.run_calibration

    def run_calibration(*args, **kwargs):
        if window.selected_checkbox_id != OPTUNA_MASS_ROW:
            return run(*args, **kwargs)
        config_file = window.config_path_label.text()
        if not config_file or config_file == "No file chosen":
            window.print_log("Please select a config file first.")
            return None
        options = _ask_optuna_options(window)
        if options is None:
            window.print_log("Automatic mass calibration cancelled.")
            return None
        window.print_log(
            f"Starting automatic mass calibration (Optuna): {options['trials']} trials, "
            f"{'separate' if options['per_facies'] else 'shared'} signal values."
        )
        toolbar = window.toolbar_builder
        tab.process = window.process_runner.start_workflow_process(
            optuna_mass_calibration_argv(config_file, options),
            toolbar.play_action,
            toolbar.stop_action,
            cwd=Path.cwd(),
            workflow="calibration",
            actions=["mass"],
            config_path=Path(config_file),
        )
        return tab.process

    run_calibration.__wrapped__ = run
    tab.run_calibration = run_calibration

    refresh = window.refresh_sidebar_progress

    def refresh_sidebar_progress(*args, **kwargs):
        result = refresh(*args, **kwargs)
        state = getattr(section.get_row("mass"), "_completion_state", "none")
        section.set_item_state(OPTUNA_MASS_ROW, state)
        return result

    refresh_sidebar_progress.__wrapped__ = refresh
    window.refresh_sidebar_progress = refresh_sidebar_progress


def _update_group_headers(section) -> None:
    """Hide the group headers of a sidebar section whose rows are all hidden."""

    layout = section.items_layout
    header = None
    shown = False
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, GroupHeaderLabel):
            if header is not None:
                header.setVisible(shown)
            header, shown = widget, False
        elif widget is not None and not widget.isHidden():
            shown = True
    if header is not None:
        header.setVisible(shown)


class _NativeSidebar(Sidebar):
    """Display merged categories while preserving the original action keys."""

    def __init__(self, categories_data: dict, setup_rows: dict, aliases=None) -> None:
        self._setup_rows = setup_rows
        self._logical_rows = {"setup": setup_rows, **(aliases or {})}
        self._display_rows = {
            logical: (category, key)
            for category, rows in self._logical_rows.items()
            for key, logical in rows.items()
        }
        super().__init__(categories_data)
        for category, rows in self._logical_rows.items():
            for display_id, (action, item_id) in rows.items():
                row = self._sections[category].get_row(display_id)
                row.action = action
                row.checkbox_id = item_id
        for header in self._sections["setup"].findChildren(GroupHeaderLabel):
            if not header.text():
                header.hide()

    def display_selection(self, action: str, item_id: str) -> tuple[str, str]:
        if (action, item_id) == ("preprocessing", "depth_measurements"):
            action, item_id = "setup", "depth"
        return self._display_rows.get((action, item_id), (action, item_id))

    def _logical_selection(self, action: str, item_id: str) -> tuple[str, str]:
        return self._logical_rows.get(action, {}).get(item_id, (action, item_id))

    def _on_selection_changed(self, action: str, checkbox_id: str) -> None:
        # Let Sidebar maintain one visual selection, then emit the original keys.
        row = self._sections[action].get_row(checkbox_id)
        if row is None or not row.isEnabled():
            return
        with QSignalBlocker(self):
            super()._on_selection_changed(action, checkbox_id)
        if self._selected_row is not None:
            self.selection_changed.emit(*self._logical_selection(action, checkbox_id))

    def select(self, action: str, checkbox_id: str) -> None:
        super().select(*self.display_selection(action, checkbox_id))

    def get_selected(self) -> tuple[str, str] | None:
        selected = super().get_selected()
        return self._logical_selection(*selected) if selected is not None else None

    def set_item_state(self, action: str, checkbox_id: str, state: str) -> None:
        action, checkbox_id = self.display_selection(action, checkbox_id)
        super().set_item_state(action, checkbox_id, state)


class NativeFluidFlowerMainWindow(MainWindow):
    """FluidFlower pages alongside DarSIA's original sidebar categories."""

    # One ordered list owns the navigation; original rows come from their managers.
    SIDEBAR_CATEGORIES = (
        ("start", "Start", "fa5s.home", (("Series", "series"),)),
        ("setup", "Setup", "fa5s.cogs", None),
        ("calibration", "Calibration", "fa5s.balance-scale", None),
        ("analysis", "Analysis", "fa5s.chart-line", None),
        ("comparison", "Comparison", "fa5s.exchange-alt", None),
        ("helper", "Helper", "fa5s.life-ring", None),
        ("utils", "Utils", "fa5s.toolbox", None),
    )

    SETUP_WORKFLOW = (
        (
            "Protocols",
            "preprocessing",
            "protocols",
            "Read the run protocol and produce the injection and timing records.",
        ),
        (
            "Rig corners and injection points",
            "geometry_facies",
            "rig_frame",
            "Mark the inner rig corners and injection points on a photograph for the crop geometry.",
        ),
        (
            "Depth",
            "setup",
            "depth",
            "Choose constant depth or measured points, then compute the depth map using the original two steps.",
        ),
        (
            "Paint layers and facies",
            "geometry_facies",
            "layers",
            "Paint the sand layers and facies, then save the geometry revision.",
        ),
        (
            "Apply geometry to Setup",
            "geometry_facies",
            "export",
            "Open the saved native export and explicitly apply registered rig corners to the run crop settings; labels and facies remain separate reference exports.",
        ),
        (
            "Segmentation (automatic)",
            "setup",
            "segmentation",
            "Automatically segment the configured image as an alternative to painted geometry.",
        ),
        (
            "Facies (automatic)",
            "setup",
            "facies",
            "Build the facies data from automatic segmentation as an alternative to painted geometry.",
        ),
        (
            "Compute rig geometry",
            "setup",
            "rig",
            "Combine the configured labels, facies and corrections into the rig geometry.",
        ),
        (
            "Full setup",
            "setup",
            "all",
            "Run all original DarSIA Setup steps using the selected run configuration.",
        ),
    )

    def __init__(self, *, initial_series: str | None = None) -> None:
        super().__init__()
        use_environment_python(self.process_runner)
        clear_config_placeholder(self)
        guard_config_file(self)
        settle_settings_before_sync(self)
        self._config_series: str | None = None
        self._current_category = "start"
        self._last_sidebar_selection = ("start", "series")
        self._selecting_sidebar = False
        self._method_root = None
        self._method_manifest_bytes = None
        self._saved_method = MethodSelection()
        self._method_panel_seed = MethodSelection()
        self._scope_base_help = {}
        self.scope_context = None
        self.current_row_scope = None
        series = tuple(configured_series())
        initial_series = initial_series or (series[0][0] if series else "")
        if initial_series and initial_series not in {
            key for key, _label, _root in series
        }:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        self.home_page = NativeHomePage(series=series, initial_series=initial_series)
        self._legacy_windows: dict[str, QWidget] = {}

        # Retain MainWindow's splitter, menus, toolbar, log and streaming handles.
        right_layout = self.settings_scroll_area.parentWidget().layout()
        right_layout.removeWidget(self.settings_scroll_area)
        self.method_badge = QLabel("Method: not chosen")
        self.method_badge.setTextFormat(Qt.TextFormat.PlainText)
        self.method_badge.setWordWrap(True)
        right_layout.addWidget(self.method_badge)
        self.series_header = QWidget()
        top = QHBoxLayout(self.series_header)
        top.setContentsMargins(0, 0, 0, 0)
        self.setup_series_label = QLabel()
        top.addWidget(self.setup_series_label)
        top.addSpacing(16)
        self.run_label = QLabel("Run")
        top.addWidget(self.run_label)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(90)
        self.run_combo.setPlaceholderText("Choose a run")
        self.run_combo.setToolTip(
            "Run whose configuration DarSIA's steps open: config/merged/<run>.toml "
            "when the series has one, else config/runs/<run>.toml"
        )
        top.addWidget(self.run_combo)
        self.run_scope_note = QLabel("applies to the whole series")
        self.run_scope_note.setTextFormat(Qt.TextFormat.PlainText)
        self.run_scope_note.setObjectName("secondaryText")
        self.run_scope_note.hide()
        top.addWidget(self.run_scope_note)
        self.run_path = QLabel()
        self.run_path.setObjectName("secondaryText")
        self.run_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        top.addWidget(self.run_path)
        top.addStretch(1)
        right_layout.addWidget(self.series_header)
        self.method_panel = SeriesMethodPanel()
        self.method_scope_label = QLabel()
        self.method_scope_label.setTextFormat(Qt.TextFormat.PlainText)
        self.method_scope_label.setWordWrap(True)
        self.method_panel.layout().insertWidget(1, self.method_scope_label)
        self.method_panel.hide()
        self.method_panel.use_requested.connect(self._save_method_choice)
        right_layout.addWidget(self.method_panel)
        self.analysis_method_badge = QLabel("Method: not chosen")
        self.analysis_method_badge.setTextFormat(Qt.TextFormat.PlainText)
        self.analysis_method_badge.setWordWrap(True)
        self.analysis_method_badge.hide()
        right_layout.addWidget(self.analysis_method_badge)
        self.scope_label = QLabel()
        self.scope_label.setTextFormat(Qt.TextFormat.PlainText)
        self.scope_label.setWordWrap(True)
        self.scope_label.hide()
        right_layout.addWidget(self.scope_label)
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.settings_scroll_area)
        self.content_stack.addWidget(self.home_page)
        from darsia.fluidflower.mass_calibration_display import MassCalibrationDisplay

        self.mass_calibration_display = MassCalibrationDisplay()
        self._displayed_mass_method = None
        self.content_stack.addWidget(self.mass_calibration_display)
        self.geometry_workspace = SeriesGeometryWorkspace(
            series=series, initial_series=initial_series
        )
        self.geometry_workspace.series_label.hide()
        self.crop_workflow = NativeCropWorkflow(self)
        show_export = self.geometry_workspace._show_export_dialog

        def export_finished(export):
            show_export(export)
            self.crop_workflow.attach_export_dialog(
                self.geometry_workspace.export_dialog, export
            )
            self._update_painted_geometry_status()

        self.geometry_workspace._show_export_dialog = export_finished
        self.content_stack.addWidget(self.geometry_workspace)
        self.phase_model_workspace = PhaseModelWorkspace(
            series=series, initial_series=initial_series
        )
        self.phase_model_workspace.calibration_sidebar.hide()
        self.content_stack.addWidget(self.phase_model_workspace)
        self.depth_workflow = NativeDepthWorkflow(self)
        self.content_stack.addWidget(self.depth_workflow.panel)
        self.mass_analysis_page: QWidget | None = None
        self.mass_analysis_placeholder = QLabel(
            "Choose an experimental series before mass analysis."
        )
        self.content_stack.addWidget(self.mass_analysis_placeholder)
        right_layout.addWidget(self.content_stack, 1)

        sidebar_data = {}
        setup_rows = {}
        aliases = {"calibration": {}, "analysis": {}}
        for action, label, icon, rows in self.SIDEBAR_CATEGORIES:
            if action == "setup":
                display_items = []
                for text, source, item_id, help_text in self.SETUP_WORKFLOW:
                    display_id = f"{source}:{item_id}"
                    setup_rows[display_id] = (source, item_id)
                    display_items.append((text, display_id, "fa5s.circle", help_text))
                groups = [("", display_items)]
            elif action == "calibration":
                groups = self.calibration_tab.sidebar_items()
                groups[0] = ("Colour path — original method", groups[0][1])
                phase_rows = []
                for text, item_id, help_text in (
                    (
                        "Training",
                        "training",
                        "Choose, review and train a saved phase-model version.",
                    ),
                    (
                        "Mass calibration",
                        "mass_from_phase",
                        "Show the saved phase-model parameters and reported mass-balance error; calibration runs belong to Jobs.",
                    ),
                    (
                        "Image selection plan…",
                        "selection_plan",
                        "Plan the per-series image selection without calculating images.",
                    ),
                ):
                    display_id = f"phase_model:{item_id}"
                    aliases[action][display_id] = ("phase_model", item_id)
                    phase_rows.append((text, display_id, "fa5s.circle", help_text))
                groups.append(("Phase model — classification first", phase_rows))
            elif action == "analysis":
                original_rows = [
                    row
                    for _group, rows in self.analysis_tab.sidebar_items()
                    for row in rows
                ]
                phase_rows = []
                for text, item_id, help_text in (
                    (
                        "Frozen model mass analysis",
                        "frozen",
                        "Inspect and export the selected model's completed concentration and mass results.",
                    ),
                    (
                        "Phase review window",
                        "review",
                        "Open the existing prepared model-correction review window; a loaded model with bound phase maps is required.",
                    ),
                ):
                    display_id = f"mass_analysis:{item_id}"
                    aliases[action][display_id] = ("mass_analysis", item_id)
                    phase_rows.append((text, display_id, "fa5s.circle", help_text))
                groups = [
                    ("Original method", original_rows),
                    ("Phase model", phase_rows),
                ]
            else:
                groups = (
                    self.action_dispatch[action].sidebar_items()
                    if rows is None
                    else [
                        (
                            label,
                            [
                                (text, row_id, "fa5s.circle", text)
                                for text, row_id in rows
                            ],
                        )
                    ]
                )
            sidebar_data[action] = (label, icon, groups)
        old_sidebar = self.sidebar
        self.sidebar = _NativeSidebar(sidebar_data, setup_rows, aliases)
        self.sidebar.setMinimumWidth(180)
        self.sidebar.setMaximumWidth(600)
        old_sidebar.parentWidget().layout().replaceWidget(old_sidebar, self.sidebar)
        old_sidebar.hide()
        old_sidebar.deleteLater()
        self.sidebar.selection_changed.connect(self._on_sidebar_selection)
        self.sidebar.expanded_changed.connect(self._on_sidebar_expanded)
        # The helper captures its calibration section, so install it after replacement.
        add_automatic_mass_calibration(self)
        section = self.sidebar._sections["calibration"]
        mass_row = section.get_row("mass")
        mass_row.label = "Mass calibration"
        mass_row._text_label.setText(mass_row.label)
        mass_row.help_text = "Show saved colour-path parameters and reported mass-balance error; calibration runs belong to Jobs."
        # Retain the original implementations and logical IDs, but offer one
        # read-only mass row while calibration execution moves to Jobs.
        for key in (OPTUNA_MASS_ROW, "default_mass"):
            row = section.get_row(key)
            row.hide()
            row.setEnabled(False)
        original_calibration = self.calibration_tab.run_calibration

        def run_calibration(*args, **kwargs):
            if self.selected_checkbox_id in {"mass", OPTUNA_MASS_ROW, "default_mass"}:
                self.print_log(
                    "Mass calibration is read-only here. Manual and automatic calibration runs will be available on Jobs."
                )
                return None
            return original_calibration(*args, **kwargs)

        run_calibration.__wrapped__ = original_calibration
        self.calibration_tab.run_calibration = run_calibration
        self._capture_method_groups()
        self._install_method_analysis_guard()
        self.toolbar_builder.play_action.changed.connect(self._guard_custom_play)

        self.run_combo.activated.connect(self._run_chosen)
        self.run_combo.currentIndexChanged.connect(self._update_setup_action)
        self.run_combo.currentIndexChanged.connect(self._refresh_scope_display)
        self.home_page.table.itemSelectionChanged.connect(self._home_series_selected)
        self.phase_model_workspace.series_selector.currentIndexChanged.connect(
            self._sync_series_display
        )
        self.home_page.new_requested.connect(self._open_preprocessing_for_selected_run)
        self.home_page.new_series_button.clicked.connect(self._new_series)
        self.home_page.add_existing_button.clicked.connect(self._add_existing_series)
        self.home_page.refresh_series_button.clicked.connect(
            lambda: self._refresh_series(show_errors=True)
        )
        self.home_page.update_run_sources_requested.connect(self._show_setup_for_path)
        self.home_page.open_requested.connect(self._open_series_from_start)
        self.home_page.recent_requested.connect(self._open_series_from_start)
        self._sync_series_display()
        self._select_row(("start", "series"))

    def _refresh_scope_display(self, *_args) -> None:
        """Render one read-only context across row help, header and selector."""
        if not all(
            hasattr(self, name)
            for name in (
                "scope_label",
                "phase_model_workspace",
                "sidebar",
                "run_combo",
                "method_scope_label",
                "run_scope_note",
                "_scope_base_help",
            )
        ):
            return
        from darsia.fluidflower.row_scope import (
            load_scope_context,
            row_scope,
            series_scope,
        )

        workspace = self.phase_model_workspace
        series_id = workspace.series_id
        root = workspace.series_roots.get(series_id)
        run_id = (
            self.run_combo.currentText() if self.run_combo.currentIndex() >= 0 else ""
        )
        path = self.run_combo.currentData() or None
        values = None
        if self.config_file:
            # Original actions use the actual open file. An unrelated File >
            # Open Config must never be relabelled as a registered series run.
            path = self.config_file
            values = self.config_dict
            loaded_index = self._run_index(path)
            if self._config_series == series_id and loaded_index >= 0:
                run_id = self.run_combo.itemText(loaded_index)
        context = load_scope_context(
            root,
            series_id,
            run_id,
            path,
            values,
            series_label=workspace.series_selector.currentText() or series_id,
        )
        self.scope_context = context
        for category in ("calibration", "analysis"):
            section = self.sidebar._sections.get(category)
            if section is None:
                continue
            for row in section._rows.values():
                base = self._scope_base_help.setdefault(id(row), row.help_text)
                scope = row_scope(context, row.action, row.checkbox_id)
                if scope is None:
                    continue
                text = base + "\n" + scope.line
                if scope.note:
                    text += "\n" + scope.note
                row.help_text = text
                row.setToolTip(text)
                popup = getattr(row, "_help_popup", None)
                if popup is not None:
                    popup._help_text = text
                    popup._help_label.setTextFormat(Qt.TextFormat.PlainText)
                    popup._help_label.setText(text)
                    popup.adjustSize()
        whole_series = series_scope(context)
        self.method_scope_label.setText(whole_series.line)
        self.method_scope_label.setToolTip(whole_series.note)
        category = {"phase_model": "calibration", "mass_analysis": "analysis"}.get(
            self._current_category, self._current_category
        )
        selected = self.sidebar.get_selected()
        scope = (
            row_scope(context, *selected)
            if selected is not None and category in {"calibration", "analysis"}
            else None
        )
        self.current_row_scope = scope
        self.scope_label.setText(scope.line if scope is not None else "")
        self.scope_label.setToolTip(scope.note if scope is not None else "")
        self.scope_label.setVisible(
            scope is not None and category in {"calibration", "analysis"}
        )
        series_row = scope is not None and not scope.run_selector_enabled
        self.run_combo.setEnabled(not series_row)
        self.run_scope_note.setVisible(series_row)
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.keep_selection_disabled_if_running()

    def _capture_method_groups(self) -> None:
        """Retain the existing row widgets, including the original Optuna row."""
        self._method_group_widgets = {}
        self._method_row_labels = {}
        self._method_group_order = {}
        for category in ("calibration", "analysis"):
            section = self.sidebar._sections[category]
            grouped = {"colour_path": [], "phase_model": []}
            method = "colour_path"
            rows = {}
            for index in range(section.items_layout.count()):
                widget = section.items_layout.itemAt(index).widget()
                if isinstance(widget, GroupHeaderLabel) and widget.text().startswith(
                    "Phase model"
                ):
                    method = "phase_model"
                grouped[method].append(widget)
                if isinstance(widget, SidebarRow):
                    _category, key = self.sidebar.display_selection(
                        widget.action, widget.checkbox_id
                    )
                    rows[key] = widget
                    self._method_row_labels[id(widget)] = widget.label
            # The unchanged Optuna helper iterates logical row IDs. Restore the
            # custom aliases after it has inserted its row, without rebuilding UI.
            section._rows = rows
            self._method_group_widgets[category] = grouped

    def _method_series_root(self) -> Path | None:
        workspace = self.phase_model_workspace
        root = workspace.series_roots.get(workspace.series_id)
        return Path(root) if root is not None else None

    def _show_mass_calibration_values(self, method: str) -> None:
        """Read only the explicitly referenced saved calibration, never fit."""
        from darsia.fluidflower.mass_calibration_display import (
            read_mass_calibration_display,
        )

        selection = self.method_panel.selection()
        reference = (
            selection.mass_calibration
            if method == "phase_model"
            else selection.colour_calibration
        )
        data = read_mass_calibration_display(
            method, reference, series_root=self._method_series_root()
        )
        self.mass_calibration_display.set_display(data)
        self._displayed_mass_method = method
        self.content_stack.setCurrentWidget(self.mass_calibration_display)

    def _refresh_method_state(self, *, force: bool = False) -> None:
        if not hasattr(self, "method_panel") or not hasattr(
            self, "phase_model_workspace"
        ):
            return
        root = self._method_series_root()
        try:
            raw = (root / "series.toml").read_bytes() if root is not None else None
        except OSError:
            raw = None
        changed = root != self._method_root or raw != self._method_manifest_bytes
        pristine = self.method_panel.selection() == self._method_panel_seed
        if changed or force:
            self._method_root = root
            self._method_manifest_bytes = raw
            self._saved_method = (
                read_series_method(root) if root is not None else MethodSelection()
            )
        if changed or force or pristine:
            self.method_panel.set_selection(
                self._saved_method,
                available=root is not None,
            )
            self._method_panel_seed = self.method_panel.selection()
        choice = self._saved_method
        if choice.active == "phase_model":
            text = f"Method: phase model · model {choice.model_version or 'not referenced'} · mass calibration {choice.mass_calibration or 'not referenced'}"
        elif choice.active == "colour_path":
            text = f"Method: colour path · calibration {choice.colour_calibration or 'not referenced'}"
        else:
            text = "Method: not chosen"
        missing = missing_method_references(choice)
        if missing:
            text += " · missing " + " and ".join(missing)
        if choice.error:
            text += " · " + choice.error
        for badge in (self.method_badge, self.analysis_method_badge):
            badge.setText(text)
            badge.setStyleSheet(
                "color: #b77900;"
                if not choice.active or missing or choice.error
                else ""
            )
        self._order_method_groups()
        if self.content_stack.currentWidget() is self.mass_calibration_display:
            self._show_mass_calibration_values(self._displayed_mass_method)

    def _order_method_groups(self) -> None:
        if not hasattr(self, "_method_group_widgets"):
            return
        chosen = self._saved_method.active
        order = (
            ("phase_model", "colour_path")
            if chosen == "phase_model"
            else ("colour_path", "phase_model")
        )
        for category, groups in self._method_group_widgets.items():
            section = self.sidebar._sections[category]
            keys = {id(row): key for key, row in section._rows.items()}
            rows = {}
            for method in order:
                for widget in groups[method]:
                    section.items_layout.removeWidget(widget)
                    section.items_layout.addWidget(widget)
                    if isinstance(widget, SidebarRow):
                        base = self._method_row_labels[id(widget)]
                        suffix = (
                            " (not the chosen method for this series)"
                            if category == "analysis" and method != chosen
                            else ""
                        )
                        widget.label = base + suffix
                        widget._text_label.setText(widget.label)
                        if category == "analysis":
                            widget._text_label.setWordWrap(True)
                            widget.setMinimumHeight(44 if suffix else 28)
                        rows[keys[id(widget)]] = widget
            section._rows = rows
            self._method_group_order[category] = order

    def _set_method_category_visibility(self, action: str, item_id: str) -> None:
        category, _item = self.sidebar.display_selection(action, item_id)
        self.method_panel.setVisible(category == "calibration")
        self.analysis_method_badge.setVisible(category == "analysis")
        self.method_badge.setVisible(category != "start")

    def _save_method_choice(self, selection: MethodSelection) -> None:
        root = self._method_series_root()
        if root is None or root != self._method_root:
            self.print_log("Choose a series before recording its method.")
            return
        try:
            save_series_method(
                root, selection, expected_bytes=self._method_manifest_bytes
            )
        except (OSError, ValueError, RuntimeError) as error:
            self.print_log(f"Could not record the series method: {error}")
            QMessageBox.warning(self, "Could not record series method", str(error))
            return
        self._refresh_method_state(force=True)
        self.print_log(
            f"Series method recorded: {selection.active}. No calibration or run configuration changed."
        )

    def _confirm_method_action(self, method: str, description: str) -> bool:
        self._refresh_method_state()
        active = self._saved_method.active
        if (
            active == method
            and not missing_method_references(self._saved_method)
            and not self._saved_method.error
        ):
            return True
        prefix = (
            "No complete method is chosen for this series."
            if not active or missing_method_references(self._saved_method)
            else "This is not the chosen method for this series."
        )
        answer = QMessageBox.question(
            self,
            "Confirm analysis method",
            prefix
            + "\n"
            + description
            + " anyway?\nChoose the series method under Calibration. This confirmation will not change it.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _install_method_analysis_guard(self) -> None:
        original = self.analysis_tab.run_analysis

        def run_analysis(*args, **kwargs):
            if not self._confirm_method_action(
                "colour_path", "Run original DarSIA analysis"
            ):
                return None
            return original(*args, **kwargs)

        run_analysis.__wrapped__ = original
        self.analysis_tab.run_analysis = run_analysis

    def _guard_phase_exports(self, page) -> None:
        saved = getattr(page, "saved_results", page)
        if getattr(saved, "_native_method_guard", False):
            return
        original = saved.export

        def export(*args, **kwargs):
            if not self._confirm_method_action(
                "phase_model", "Export phase-model analysis results"
            ):
                return None
            return original(*args, **kwargs)

        export.__wrapped__ = original
        saved.export = export
        saved._native_method_guard = True

    def _open_existing_phase_review(self) -> None:
        studio = getattr(self.phase_model_workspace.studio_page, "studio", None)
        if (
            studio is not None
            and getattr(studio, "_active_model_series", None)
            == self.phase_model_workspace.series_id
        ):
            selected = studio._selected_validation_model() or {}
            if isinstance(selected.get("validation_binding"), dict):
                studio._launch_review("open_correction")
                return
        self.print_log(
            "Open a phase model with prepared review images under Calibration before opening its phase review window."
        )

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
                if (
                    item is not None
                    and item.data(Qt.ItemDataRole.UserRole) == selected_path
                ):
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
            f"Series: {label}" if label else "No series selected"
        )
        for row in range(self.home_page.table.rowCount()):
            if self.home_page.table.item(row, 1).text() == selector.currentData():
                self.home_page.table.selectRow(row)
                break
        self.geometry_workspace.set_series(self.phase_model_workspace.series_id)
        self._update_painted_geometry_status()
        self._populate_runs()
        if self._darsia_visible():
            self._ensure_run_loaded()
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.refresh_if_visible(reload_selected=True)
        if self.mass_analysis_page is not None:
            series_id = self.phase_model_workspace.series_id
            root = self.phase_model_workspace.series_roots.get(series_id)
            if root is not None and (
                self.mass_analysis_page.series != series_id
                or self.mass_analysis_page.project_root != Path(root)
            ):
                self.mass_analysis_page.set_series(series_id, root)
        self._refresh_method_state()
        self._refresh_scope_display()

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
            self.content_stack.addWidget(self.mass_analysis_page)
            self._guard_phase_exports(self.mass_analysis_page)
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

    def _job_notice(self) -> tuple[str, str] | None:
        page = self.phase_model_workspace.studio_page
        studio = getattr(page, "studio", None)
        return page._active_job_notice(studio) if studio is not None else None

    def _new_series(self) -> None:
        """Create a series folder with the New series wizard and register it."""

        notice = self._job_notice()
        if notice is not None:
            QMessageBox.information(self, *notice)
            return
        from darsia.fluidflower.series_wizard import NewSeriesFolderWizard

        wizard = NewSeriesFolderWizard(
            self,
            registered_ids=[key for key, _label, _root in configured_series()],
        )
        self.new_series_wizard = wizard
        if wizard.exec() != QDialog.DialogCode.Accepted or wizard.created is None:
            return
        series_id = self.register_new_series(wizard.created)
        if not series_id:
            return
        page = wizard.series_page
        rig_preset = (
            (page.physical_rig.currentText(), str(page.physical_rig.currentData()))
            if page.rig_class.currentData() == "medium"
            else None
        )
        try:
            first_run, _path = self._prepare_new_series_runs(
                wizard.created, rig_preset=rig_preset
            )
        except (OSError, ValueError, KeyError) as error:
            message = f"Series created, but Preprocessing could not be opened: {error}"
            self.print_log(message)
            QMessageBox.warning(self, "Series created; setup not opened", message)
            return
        self._populate_runs()
        self.run_combo.setCurrentIndex(self.run_combo.findText(first_run))
        self._open_preprocessing_for_selected_run(rig_preset=rig_preset)

    @staticmethod
    def _prepare_new_series_runs(
        root: Path, *, rig_preset: tuple[str, str] | None = None
    ) -> tuple[str, Path]:
        """Prepare controller files only for a just-created series, in manifest order."""
        from copy import deepcopy
        import toml
        from darsia.presets.workflows.config.utils import _deep_merge

        root = Path(root)
        manifest = _read_toml(root / "series.toml")
        paths = manifest.get("paths") or {}
        runs = manifest.get("runs") or {}
        ordered_runs = [str(run) for run in runs.get("ids") or []]
        if not ordered_runs:
            raise ValueError("The new series contains no experiments.")
        common = _read_toml(
            root / str(paths.get("common_config", "config/common.toml"))
        )
        files = runs.get("config_files") or {}
        run_folder = str(paths.get("run_configs", "config/runs"))
        merged_folder = root / str(paths.get("merged_configs", "config/merged"))
        rig = None
        if rig_preset is not None:
            from darsia.presets.workflows.config.catalogue import SeriesPreset

            pieces = SeriesPreset(rig=rig_preset[1]).pieces()
            if "rig" not in pieces:
                raise ValueError(
                    f"The existing rig profile is unavailable: {rig_preset[1]}"
                )
            rig = pieces["rig"][1]
        merged_folder.mkdir(parents=True, exist_ok=True)
        for run in ordered_runs:
            name = Path(str(files.get(run) or f"{run_folder}/{run}.toml"))
            source = name if name.is_absolute() else root / name
            merged = _deep_merge(deepcopy(common), _read_toml(source))
            if rig is not None:
                merged["rig"] = deepcopy(rig)
            target = merged_folder / f"{run}.toml"
            # Never overwrite a file if creation raced or this is not a fresh series.
            with target.open("x", encoding="utf-8", newline="\n") as stream:
                toml.dump(merged, stream)
        return ordered_runs[0], merged_folder / f"{ordered_runs[0]}.toml"

    def register_new_series(self, root: Path) -> str:
        """Register a created series folder, select it and show the next steps."""

        from darsia.fluidflower.series_paths import register_series

        try:
            series_id, label, _root = register_series(Path(root))
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self,
                "Series created but not registered",
                f"{error}\n\nAdd {root} on Start with 'Add existing series folder'.",
            )
            return ""
        self._refresh_series(preferred_series=series_id, show_errors=True)
        self._select_series(series_id)
        self.print_log(f"Series {label} created and registered in {root}.")
        return series_id

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

        self._open_series_from_start(str(_root))

    def open_image_selection_plan(self) -> None:
        """Open the metadata-only image plan from the Phase model category."""
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
        """Historical Start-opening entry point; opening now lands on Setup."""
        self._open_series_from_start(path)

    def _show_home_calibration(self) -> None:
        """Historical hidden Open-button entry point; preserve the Start route."""
        self._open_series_from_start()

    def _open_series_from_start(self, path: str | None = None) -> None:
        """Open series metadata and the first Setup row, without a wizard/model.

        The table's existing single- and double-click signals share this route.
        A repeated signal for the already-open destination is a no-op; returning
        to Start and opening the same series again remains a normal open.
        """
        path = path or self.home_page._selected_project_path()
        if path in self.home_page.series_registry.legacy_by_path:
            self._open_legacy_project(path)
            return
        series_id = self._series_for_path(path)
        if not series_id:
            return
        target = ("preprocessing", "protocols")
        if (
            self.phase_model_workspace.series_id == series_id
            and self.sidebar.get_selected() == target
            and self.content_stack.currentWidget() is self.settings_scroll_area
        ):
            return
        self._select_series(series_id)
        # Retain the existing active-job guard's rejection of a series change.
        if self.phase_model_workspace.series_id != series_id:
            return
        self._select_row(target)

    def _home_series_selected(self) -> None:
        """Keep the per-run categories on the series selected on Start."""
        selected = self.home_page.selected_series()
        if selected and selected != self.phase_model_workspace.series_id:
            self._select_series(selected)
        self._update_setup_action()

    def _update_setup_action(self, _index: int = -1) -> None:
        selected = self.home_page.selected_series()
        path = (
            self.run_combo.currentData()
            if selected and selected == self.phase_model_workspace.series_id
            else None
        )
        enabled = bool(path and Path(path).is_file())
        button = self.home_page.new_button
        button.setText("Set up an experiment")
        button.setEnabled(enabled)
        button.setToolTip(
            f"Open Preprocessing for {self.run_combo.currentText()}."
            if enabled
            else "Choose a series and an available experiment in Run above first."
        )

    def _open_preprocessing_for_selected_run(
        self, *, rig_preset: tuple[str, str] | None = None
    ) -> None:
        """Load the selected run and use DarSIA's unchanged per-run wizard."""
        self._home_series_selected()
        selected = self.home_page.selected_series()
        path = self.run_combo.currentData()
        if not selected or selected != self.phase_model_workspace.series_id or not path:
            self._update_setup_action()
            return
        recorded = getattr(self, "_config_stamp", None)
        already_loaded = (
            _same_file(self.config_file, path)
            and self._config_series == selected
            and recorded is not None
            and recorded[1] == _file_stamp(path)
        )
        # The unchanged wizard syncs pending form edits. Do not discard them by
        # reloading a file which is already open and unchanged on disk.
        if not already_loaded and not self._load_run_config(str(path)):
            message = f"Could not load the selected experiment configuration: {path}"
            self.print_log(message)
            QMessageBox.warning(self, "Could not set up experiment", message)
            return
        self._select_row(("preprocessing", "protocols"))
        if rig_preset is None:
            self.open_wizard_for("preprocessing")
            return
        from darsia.gui.ui.main_window import load_wizard_dialog_class
        from darsia.gui.ui.preprocessing_wizard import STEPS
        from darsia.presets.workflows.config.catalogue import SeriesPreset

        dialog = load_wizard_dialog_class("preprocessing")(self)
        wizard_inputs = self.settings_inputs.copy()
        # Adapt this instance only; neither Jakub's wizard nor its catalogue changes.
        label, profile = rig_preset
        preset_name = f"{label} (rig only)"
        dialog._catalogue.presets[preset_name] = SeriesPreset(
            description=f"Existing {label} rig profile ({profile}); depth and corrections are set separately.",
            rig=profile,
        )
        series_page = next(
            index for index, step in enumerate(STEPS) if step[0] == "series"
        )
        dialog._ensure_page(series_page)
        dialog._series_combo.setCurrentText(preset_name)
        # Rig-only selection leaves Data unchanged but refreshes MainWindow's
        # registry. Keep the live first-page fields owned by this wizard.
        self.settings_inputs.clear()
        self.settings_inputs.update(wizard_inputs)
        prepare_crop_wizard(dialog)
        dialog.exec()

    def open_wizard_for(self, category: str) -> None:
        if category != "preprocessing":
            return super().open_wizard_for(category)
        from darsia.gui.ui.main_window import load_wizard_dialog_class

        if not self.config_file:
            self.print_log("Wizard: create or open a config file first.")
            self.config_controller.new_config()
            if not self.config_file:
                return
        dialog = load_wizard_dialog_class(category)(self)
        prepare_crop_wizard(dialog)
        dialog.exec()

    def show_setup(self, step: str | None = None) -> None:
        """Open the requested existing setup category for the selected series."""
        legacy_path = self.home_page.selected_legacy_project()
        if legacy_path:
            self._open_legacy_project(legacy_path, setup=True)
            return
        self._select_series(self.home_page.selected_series())
        selection = {
            "protocols": ("preprocessing", "protocols"),
            "native": ("setup", "all"),
            "color": ("calibration", "color"),
        }.get(step, ("geometry_facies", "rig_frame"))
        self._select_row(selection)

    def show_original_workflow(self) -> None:
        """Compatibility entry point for the original Calibration category."""
        self._select_row(("calibration", "mass"))

    def show_geometry_facies(self) -> None:
        self._select_row(("geometry_facies", "rig_frame"))

    def show_color_path(self) -> None:
        self._select_row(("calibration", "color"))

    def _update_painted_geometry_status(self) -> None:
        """An export proves reference files exist, not that all geometry is applied."""
        from darsia.fluidflower.series_geometry import geometry_package_root
        from darsia.fluidflower.setup_display import (
            completed_current_native_setup_export,
        )

        root = self.phase_model_workspace.series_roots.get(
            self.phase_model_workspace.series_id
        )
        completed = (
            completed_current_native_setup_export(geometry_package_root(root))
            if root is not None
            else None
        )
        for item in ("segmentation", "facies"):
            row = self.sidebar._sections["setup"].get_row(f"setup:{item}")
            if row is None:
                continue
            title = next(
                (
                    text
                    for text, source, key, _help in self.SETUP_WORKFLOW
                    if source == "setup" and key == item
                ),
                row.label,
            )
            row.label = title + (" — reference export available" if completed else "")
            row._text_label.setText(row.label)
            row._text_label.setWordWrap(True)
            row.setToolTip(
                "Reference labels and facies were exported. They have not been applied to other runs."
                if completed
                else row.help_text
            )
            row.setEnabled(True)
            effect = row.graphicsEffect()
            if effect is not None:
                effect.setOpacity(1.0)
        self._update_run_action()

    def _sidebar_selection_enabled(self) -> bool:
        if not hasattr(self, "sidebar"):
            return False
        selected = self.sidebar.get_selected()
        if selected is None:
            return False
        action, item = self.sidebar.display_selection(*selected)
        row = self.sidebar._sections[action].get_row(item)
        return row is not None and row.isEnabled()

    def _darsia_visible(self) -> bool:
        return (
            getattr(self, "_current_category", None) in self.action_dispatch
            and hasattr(self, "content_stack")
            and self.content_stack.currentWidget() is self.settings_scroll_area
        )

    def _guard_custom_play(self) -> None:
        """A background workflow may finish after navigation to a custom page."""
        action = self.toolbar_builder.play_action
        if action.isEnabled() and (
            self.selected_action not in self.action_dispatch
            or not self._darsia_visible()
            or not self._sidebar_selection_enabled()
            or (
                getattr(self, "depth_workflow", None) is not None
                and self.depth_workflow.sequence.running
            )
        ):
            action.setEnabled(False)

    def _update_run_action(self) -> None:
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.sync_play_help()
        self.toolbar_builder.play_action.setEnabled(
            self._darsia_visible()
            and self.selected_action in self.action_dispatch
            and self.selected_checkbox_id is not None
            and not self.process_runner.active
            and not (
                getattr(self, "depth_workflow", None) is not None
                and self.depth_workflow.sequence.running
            )
            and self._sidebar_selection_enabled()
        )

    def _select_row(self, selection: tuple[str, str]) -> None:
        action, item_id = selection
        display_action, display_id = self.sidebar.display_selection(action, item_id)
        section = self.sidebar._sections.get(display_action)
        if section is None or section.get_row(display_id) is None:
            return
        self._selecting_sidebar = True
        try:
            section.set_expanded(True)
        finally:
            self._selecting_sidebar = False
        self.sidebar.select(action, item_id)

    def _restore_selection(self) -> None:
        """Opening a configuration clears selection in the original controller."""
        self._select_row(self._last_sidebar_selection)

    def _sync_home_series_before_navigation(self) -> None:
        if self._current_category == "start":
            selected = self.home_page.selected_series()
            if selected and selected != self.phase_model_workspace.series_id:
                self._select_series(selected)

    def _on_sidebar_expanded(self, action: str, is_expanded: bool) -> None:
        if action in {"calibration", "analysis"} and hasattr(self, "content_stack"):
            if is_expanded and not self._selecting_sidebar:
                self._sync_home_series_before_navigation()
                self._refresh_method_state()
                first = next(iter(self.sidebar._sections[action]._rows))
                logical = self.sidebar._logical_selection(action, first)
                self.sidebar.select(*logical)
                return
        if action == "setup" and hasattr(self, "content_stack"):
            self._update_painted_geometry_status()
            if is_expanded and not self._selecting_sidebar:
                self._sync_home_series_before_navigation()
                self.sidebar.select("preprocessing", "protocols")
                return
            if not is_expanded and self.selected_action in {
                "preprocessing",
                "setup",
                "geometry_facies",
            }:
                self.selected_action = None
                self.selected_checkbox_id = None
                self._sync_wizard_action_state()
            else:
                super()._on_sidebar_expanded(action, is_expanded)
        else:
            super()._on_sidebar_expanded(action, is_expanded)
        if not hasattr(self, "content_stack"):
            return
        if is_expanded:
            self._sync_home_series_before_navigation()
            self._set_method_category_visibility(action, "")
            if not self._selecting_sidebar:
                section = self.sidebar._sections[action]
                if action not in self.action_dispatch:
                    self.sidebar.select(action, next(iter(section._rows)))
                else:
                    self._current_category = action
                    self.series_header.show()
                    self.content_stack.setCurrentWidget(self.settings_scroll_area)
        self._update_run_action()
        self._refresh_scope_display()

    def _on_sidebar_selection(self, action: str, item_id: str) -> None:
        self._sync_home_series_before_navigation()
        self._current_category = action
        self._last_sidebar_selection = (action, item_id)
        self.series_header.setVisible(action != "start")
        self._set_method_category_visibility(action, item_id)
        self._refresh_method_state()
        self._refresh_scope_display()
        if (action, item_id) in {
            ("calibration", "mass"),
            ("phase_model", "mass_from_phase"),
        }:
            self.selected_action = action
            self.selected_checkbox_id = item_id
            self._show_mass_calibration_values(
                "colour_path" if action == "calibration" else "phase_model"
            )
            self._sync_wizard_action_state()
            self._update_run_action()
            return
        if (action, item_id) == ("setup", "depth"):
            self.selected_action, self.selected_checkbox_id = action, item_id
            self.depth_workflow.show()
            self.toolbar_builder.wizard_action.setEnabled(False)
            self._update_run_action()
            return
        if action in self.action_dispatch:
            self.content_stack.setCurrentWidget(self.settings_scroll_area)
            self._ensure_run_loaded()
            super()._on_sidebar_selection(action, item_id)
        else:
            self.selected_action = action
            self.selected_checkbox_id = item_id
            if action == "start":
                self.content_stack.setCurrentWidget(self.home_page)
                self._refresh_series()
            elif action == "geometry_facies":
                self.content_stack.setCurrentWidget(self.geometry_workspace)
                self.geometry_workspace.ensure_page()
                if item_id == "export":
                    self.crop_workflow.open_current_export()
                view = self.geometry_workspace.simple_view
                target = getattr(
                    view,
                    {
                        "rig_frame": "rig_panel",
                        "layers": "facies_table",
                        "export": "export_button",
                    }[item_id],
                    None,
                )
                if target is not None:
                    target.setFocus()
            elif action == "phase_model":
                workspace = self.phase_model_workspace
                if item_id == "selection_plan":
                    self.content_stack.setCurrentWidget(workspace)
                    if workspace.series_id:
                        self.open_image_selection_plan()
                    else:
                        self.print_log(
                            "Choose a series on Start before planning images."
                        )
                    self._sync_wizard_action_state()
                    self._update_run_action()
                    return
                # Select Training even without a series, without a lazy Studio load.
                mode = "mass" if item_id == "mass_from_phase" else "training"
                with QSignalBlocker(workspace.calibration_sidebar):
                    workspace.select_calibration(mode)
                workspace._requested_mode = mode
                self.content_stack.setCurrentWidget(workspace)
                if workspace.series_id:
                    workspace.ensure_studio()
                workspace._show_selected_studio_mode()
            elif action == "mass_analysis":
                if item_id == "review":
                    self._open_existing_phase_review()
                    self._sync_wizard_action_state()
                    self._update_run_action()
                    return
                page = (
                    self.ensure_mass_analysis()
                    if self.phase_model_workspace.series_id
                    else self.mass_analysis_placeholder
                )
                self.content_stack.setCurrentWidget(page)
            self._sync_wizard_action_state()
        self._update_run_action()

    def _on_open_full_config(self) -> None:
        self.series_header.hide()
        self.method_panel.hide()
        self.analysis_method_badge.hide()
        self.scope_label.hide()
        self.run_scope_note.hide()
        self.run_combo.setEnabled(True)
        self.content_stack.setCurrentWidget(self.settings_scroll_area)
        super()._on_open_full_config()
        self._update_run_action()

    # -------------------------------------------------------------- run files

    def _set_run_controls_visible(self, visible: bool) -> None:
        for widget in (self.run_label, self.run_combo, self.run_path):
            widget.setVisible(visible)

    def _populate_runs(self) -> None:
        """List the runs of the selected series and the file each one opens."""
        workspace = self.phase_model_workspace
        series_id = workspace.series_id
        root = workspace.series_roots.get(series_id)
        choices = run_configurations(root) if root is not None else []
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run, path, note in choices:
            self.run_combo.addItem(run, str(path) if path is not None else "")
            index = self.run_combo.count() - 1
            self.run_combo.setItemData(
                index,
                f"{path}\n{note}" if path is not None else note,
                Qt.ItemDataRole.ToolTipRole,
            )
            if path is None:
                self.run_combo.model().item(index).setEnabled(False)
        current = self._run_index(self.config_file)
        if current < 0 or self._config_series != series_id:
            remembered = QSettings().value(f"native_app/run/{series_id}", "", type=str)
            current = self.run_combo.findText(remembered) if remembered else -1
            if current >= 0 and not self.run_combo.itemData(current):
                current = -1
        self.run_combo.setCurrentIndex(current)
        self.run_combo.blockSignals(False)
        self._update_run_path()
        self._update_setup_action()
        self._refresh_scope_display()

    def _run_index(self, path: str | None) -> int:
        for index in range(self.run_combo.count()):
            if _same_file(self.run_combo.itemData(index), path):
                return index
        return -1

    def _update_run_path(self) -> None:
        path = self.config_file
        if not path:
            self.run_path.setText("No configuration open")
            self.run_path.setToolTip("")
            return
        root = self.phase_model_workspace.series_roots.get(
            self.phase_model_workspace.series_id
        )
        shown = str(path)
        if root is not None:
            try:
                shown = (
                    Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
                )
            except (OSError, ValueError):
                pass
        if self.run_combo.currentIndex() < 0 or not _same_file(
            self.run_combo.currentData(), path
        ):
            shown = f"Open: {shown}"
        self.run_path.setText(shown)
        self.run_path.setToolTip(str(path))

    def _run_chosen(self, index: int) -> None:
        path = self.run_combo.itemData(index)
        if not path:
            return
        series_id = self.phase_model_workspace.series_id
        QSettings().setValue(
            f"native_app/run/{series_id}", self.run_combo.itemText(index)
        )
        self._load_run_config(path)
        self._refresh_scope_display()

    def _load_run_config(self, path: str) -> bool:
        """Open a run without letting a failed load reuse the previous config."""
        previous = (
            self.config_dict,
            self.config_file,
            self.config_path_label.text(),
            self._config_series,
            getattr(self, "_config_stamp", None),
            self._last_settings_view,
        )
        errors = []

        def loading_message(text: str) -> None:
            # ConfigController catches loading/rendering errors and returns None
            # for both success and failure. Preserve that controller unchanged.
            if text.startswith("Error loading config file:"):
                errors.append(text)

        self.log_message.connect(loading_message)
        try:
            self.config_path_label.setText(str(path))
            self.config_controller.load_config()
        except Exception as error:
            errors.append(str(error))
            self.print_log(f"Error loading config file: {error}")
        finally:
            self.log_message.disconnect(loading_message)
        loaded = (
            not errors
            and self.config_dict is not previous[0]
            and _same_file(self.config_file, path)
        )
        if not loaded:
            # Rendering can fail after the controller replaced both the config
            # and its widgets. Never sync the new widgets into the old config.
            self.settings_inputs.clear()
            (
                self.config_dict,
                self.config_file,
                label,
                self._config_series,
                self._config_stamp,
                self._last_settings_view,
            ) = previous
            self.config_path_label.setText(label)
            if self._last_settings_view is None:
                self.settings_factory._render_settings_tabs({})
            else:
                self.settings_factory.refresh_current_view()
            self._update_run_path()
            self._update_run_action()
            self._refresh_scope_display()
            return False
        self._restore_selection()
        return True

    def _ensure_run_loaded(self) -> None:
        """Open the chosen run when no file of the selected series is open."""
        path = self.run_combo.currentData()
        if not path:
            return
        if (
            not self.config_file
            or self._config_series != self.phase_model_workspace.series_id
        ):
            self._load_run_config(path)

    def _config_loaded(self) -> None:
        """Called after DarSIA opened a configuration (see guard_config_file)."""
        if not self.config_file:
            return
        self._config_series = self.phase_model_workspace.series_id
        index = self._run_index(self.config_file)
        if index >= 0:
            self.run_combo.blockSignals(True)
            self.run_combo.setCurrentIndex(index)
            self.run_combo.blockSignals(False)
        self._update_run_path()
        self._refresh_method_state()
        self._refresh_scope_display()
        if getattr(self, "depth_workflow", None) is not None:
            self.depth_workflow.refresh_if_visible()

    def abort_selected_workflow(self) -> None:
        depth = getattr(self, "depth_workflow", None)
        if depth is not None and depth.sequence.running:
            depth.sequence.abort()
            return
        super().abort_selected_workflow()

    def run_selected_workflow(self) -> None:
        """Play applies only to the original DarSIA categories."""
        if (
            getattr(self, "depth_workflow", None) is not None
            and self.depth_workflow.sequence.running
        ):
            return
        if self._darsia_visible():
            super().run_selected_workflow()

    def show_phase_model_workflow(self) -> None:
        self._select_row(("phase_model", "training"))

    def show_mass_calibration(self) -> None:
        self._select_row(("phase_model", "mass_from_phase"))

    def show_mass_analysis(self) -> None:
        self._select_row(("mass_analysis", "frozen"))


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
            "home",
            "setup",
            "protocols",
            "geometry",
            "facies",
            "native-setup",
            "color",
            "color-path",
            "original",
            "phase",
            "mass",
            "analysis",
            "mass-analysis",
            "darsia",
            "calibration",
            "fluidflower",
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
    elif args.tab == "protocols":
        window.show_setup(step="protocols")
    elif args.tab == "native-setup":
        window.show_setup(step="native")
    elif args.tab in {"geometry", "facies"}:
        window.show_geometry_facies()
    elif args.tab in {"color", "color-path"}:
        window.show_color_path()
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
            (
                5000
                if args.tab
                in {
                    "phase",
                    "calibration",
                    "fluidflower",
                    "mass",
                    "analysis",
                    "mass-analysis",
                }
                else 500
            ),
            save_screenshot,
        )
    if args.exit_after_ms:
        QTimer.singleShot(max(1, args.exit_after_ms), app.quit)
    return int(app.exec()) if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
