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

from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.project import ProjectRegistry
from darsia.fluidflower.series_geometry import series_rig_class
from darsia.fluidflower.series_geometry_workspace import SeriesGeometryWorkspace
from darsia.fluidflower.series_paths import configured_series
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
        self.new_button.setText("Set up an experiment")
        self.open_button.setText("Continue phase and mass work")
        self.open_selected_button.setText("Open phase model training")
        self.update_run_sources_button.setText("Open experiment setup")
        self.add_runs_button.hide()
        self.draft_heading.hide()
        self.draft_note.hide()
        self.table.setHorizontalHeaderLabels(
            ["Series", "ID", "Runs", "Rig class", "Last opened", "Folder"]
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
                "series folder, or use Setup to begin a new experiment."
            )
        for row in range(self.table.rowCount()):
            if self.table.item(row, 1).text() == self.initial_series:
                self.table.selectRow(row)
                break

    def refresh(self) -> None:
        super().refresh()
        rows = getattr(self.series_registry, "last_entries", [])
        for index, row in enumerate(rows[: self.table.rowCount()]):
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
            choices.append((run, merged, "Merged configuration (common and run settings)"))
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
                    window.print_log("Run cancelled: the configuration changed on disk.")
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


class NativeFluidFlowerMainWindow(MainWindow):
    """One native DarSIA workspace: series, setup, two methods and analysis.

    Tabs in working order: Start (choose a series), Setup (everything that
    describes the experiments of the series and is shared by both methods),
    Original DarSIA workflow (its mass calibration and analysis), Phase-model
    workflow (training and mass calibration) and Analysis.
    """

    _START_TAB = 0
    _SETUP_TAB = 1
    _ORIGINAL_TAB = 2
    _PHASE_MODEL_TAB = 3
    _ANALYSIS_TAB = 4
    _MASS_ANALYSIS_TAB = _ANALYSIS_TAB

    # Steps of the Setup tab: key, label and tooltip.
    SETUP_STEPS = (
        (
            "protocols",
            "Protocols",
            "Imaging protocols from the photographs and templates for the "
            "injection and pressure-temperature protocols of a new series; crop "
            "correction (DarSIA Setup)",
        ),
        (
            "geometry",
            "Geometry and facies",
            "Rig frame, injection points, sand layers and facies of a reference "
            "run, exported for native Setup",
        ),
        (
            "native",
            "Native Setup",
            "Depth, labels, facies and rig of a run (DarSIA Setup)",
        ),
        (
            "color",
            "Color path",
            "Color-path calibration of a run (DarSIA Calibration > Color Path). "
            "Both mass calibrations build on it.",
        ),
    )
    # DarSIA sidebar rows shown in each Setup step (None: all rows), the
    # section titles and the row selected first.
    _STEP_SIDEBAR = {
        "protocols": (
            {"setup": {"protocols", "crop"}},
            {"setup": "Protocols"},
            ("setup", "protocols"),
        ),
        "native": (
            {"setup": {"all", "depth", "segmentation", "facies", "rig"}},
            {"setup": "Native Setup"},
            ("setup", "all"),
        ),
        "color": (
            {"calibration": {"color"}},
            {"calibration": "Color path"},
            ("calibration", "color"),
        ),
    }
    # The original workflow: its mass calibration and analysis.
    _ORIGINAL_SIDEBAR = {
        "calibration": {"mass", OPTUNA_MASS_ROW, "default_mass", "reset", "delete"},
        "analysis": None,
        "helper": None,
        "comparison": None,
        "utils": None,
    }

    def __init__(self, *, initial_series: str | None = None) -> None:
        super().__init__()
        use_environment_python(self.process_runner)
        clear_config_placeholder(self)
        guard_config_file(self)
        settle_settings_before_sync(self)
        self._config_series: str | None = None
        self._setup_step = "geometry"
        self._step_selection = {
            key: default for key, (_rows, _labels, default) in self._STEP_SIDEBAR.items()
        }
        self._last_original_selection = ("calibration", OPTUNA_MASS_ROW)
        series = tuple(configured_series())
        initial_series = initial_series or (series[0][0] if series else "")
        if initial_series and initial_series not in {
            key for key, _label, _root in series
        }:
            raise ValueError(f"Unknown experimental series: {initial_series}")
        original_content = self.takeCentralWidget()
        if original_content is None:
            raise RuntimeError("DarSIA MainWindow has no central widget")
        self.original_content = original_content
        self.workspace_tabs = QTabBar(self)
        self.workspace_tabs.setObjectName("nativeWorkspaceTabs")
        self.workspace_tabs.setExpanding(False)
        self.workspace_tabs.addTab("Start")
        self.workspace_tabs.addTab("Setup")
        self.workspace_tabs.setTabToolTip(
            self._SETUP_TAB,
            "Protocols, geometry and facies, native Setup and color path of the "
            "selected series",
        )
        self.workspace_tabs.addTab("Original DarSIA workflow")
        self.workspace_tabs.setTabToolTip(
            self._ORIGINAL_TAB, "Mass calibration and analysis of the original method"
        )
        self.workspace_tabs.addTab("Phase-model workflow")
        self.workspace_tabs.setTabToolTip(
            self._PHASE_MODEL_TAB, "Phase-model training and mass calibration"
        )
        self.workspace_tabs.addTab("Analysis")
        self.workspace_tabs.setTabToolTip(
            self._ANALYSIS_TAB, "Mass analysis with a frozen model and calibration"
        )
        self.workspace_pages = QStackedWidget(self)
        self.home_page = NativeHomePage(series=series, initial_series=initial_series)
        self._legacy_windows: dict[str, QWidget] = {}
        self.workspace_pages.addWidget(self.home_page)

        # One page shows DarSIA's own sidebar and settings, for the Setup tab
        # (with its steps) and for the original workflow.
        self.native_page = QWidget()
        native_layout = QVBoxLayout(self.native_page)
        native_layout.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
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
        self.run_path = QLabel()
        self.run_path.setObjectName("secondaryText")
        self.run_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        top.addWidget(self.run_path)
        top.addStretch(1)
        self.image_selection_button = QPushButton("Image selection plan…")
        self.image_selection_button.setToolTip(
            "Choose frames for review, mass calibration or production; does not start a job"
        )
        top.addWidget(self.image_selection_button)
        native_layout.addLayout(top)
        self.setup_steps = QTabBar()
        self.setup_steps.setObjectName("nativeSetupSteps")
        self.setup_steps.setExpanding(False)
        for index, (_key, label, tooltip) in enumerate(self.SETUP_STEPS):
            self.setup_steps.addTab(label)
            self.setup_steps.setTabToolTip(index, tooltip)
        self.setup_steps.setCurrentIndex(self._step_index(self._setup_step))
        native_layout.addWidget(self.setup_steps)
        self.original_heading = QLabel(
            "Original DarSIA workflow · mass calibration and analysis"
        )
        heading_font = self.original_heading.font()
        heading_font.setBold(True)
        self.original_heading.setFont(heading_font)
        native_layout.addWidget(self.original_heading)
        self.native_stack = QStackedWidget()
        self.native_stack.addWidget(original_content)
        self.geometry_workspace = SeriesGeometryWorkspace(
            series=series, initial_series=initial_series
        )
        # The series is named above the steps.
        self.geometry_workspace.series_label.hide()
        self.native_stack.addWidget(self.geometry_workspace)
        native_layout.addWidget(self.native_stack, 1)
        self.workspace_pages.addWidget(self.native_page)

        self.sidebar.setMinimumWidth(180)
        self._section_labels = {
            action: section.category_label
            for action, section in self.sidebar._sections.items()
        }
        self._section_labels["calibration"] = "Original calibration"
        original_calibration = self.sidebar._sections["calibration"]
        original_calibration.category_label = "Original calibration"
        original_calibration.refresh_style()
        add_automatic_mass_calibration(self)
        self.sidebar.selection_changed.connect(self._remember_native_selection)
        self.setup_steps.currentChanged.connect(self._setup_step_changed)
        self.run_combo.activated.connect(self._run_chosen)
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
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.workspace_tabs)
        central_layout.addWidget(self.workspace_pages, 1)
        self.setCentralWidget(central)

        self.home_page.new_requested.connect(self.show_setup)
        self.home_page.new_series_button.clicked.connect(self._new_series)
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
        self._populate_runs()
        if self._darsia_visible():
            self._ensure_run_loaded()
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
        self.register_new_series(wizard.created)

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
        QMessageBox.information(
            self,
            "Series created",
            f"{label} was created in {root}.\n\nNext, in Setup:\n"
            "1. Protocols: the imaging protocols and templates for the injection "
            "and pressure-temperature protocols.\n"
            "2. Geometry and facies: rig frame, injection points and sand layers.\n"
            "3. Native Setup: depth, labels, facies and rig.\n"
            "4. Color path.",
        )
        self.show_setup(step="protocols")
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

    def show_setup(self, step: str | None = None) -> None:
        """Open Setup for the series selected on Start, optionally at one step."""
        legacy_path = self.home_page.selected_legacy_project()
        if legacy_path:
            self._open_legacy_project(legacy_path, setup=True)
            return
        self._select_series(self.home_page.selected_series())
        if step is not None:
            self.setup_steps.setCurrentIndex(self._step_index(step))
        self.workspace_tabs.setCurrentIndex(self._SETUP_TAB)

    def show_original_workflow(self) -> None:
        self.workspace_tabs.setCurrentIndex(self._ORIGINAL_TAB)

    def show_geometry_facies(self) -> None:
        """Open Setup > Geometry and facies for the series selected on Start."""
        self.setup_steps.setCurrentIndex(self._step_index("geometry"))
        self.workspace_tabs.setCurrentIndex(self._SETUP_TAB)

    def show_color_path(self) -> None:
        self.setup_steps.setCurrentIndex(self._step_index("color"))
        self.workspace_tabs.setCurrentIndex(self._SETUP_TAB)

    # ------------------------------------------------------------ setup steps

    def _step_index(self, key: str) -> int:
        for index, (step, _label, _tooltip) in enumerate(self.SETUP_STEPS):
            if step == key:
                return index
        raise ValueError(f"Unknown setup step: {key}")

    @property
    def setup_step(self) -> str:
        """Key of the current Setup step (see SETUP_STEPS)."""
        return self._setup_step

    def _setup_step_changed(self, index: int) -> None:
        self._setup_step = self.SETUP_STEPS[index][0]
        if self.workspace_tabs.currentIndex() == self._SETUP_TAB:
            self._show_setup_step()

    def _darsia_visible(self) -> bool:
        """Whether DarSIA's sidebar and settings are on screen."""
        tab = self.workspace_tabs.currentIndex()
        return tab == self._ORIGINAL_TAB or (
            tab == self._SETUP_TAB and self._setup_step != "geometry"
        )

    def _update_run_action(self) -> None:
        self.toolbar_builder.play_action.setEnabled(
            self._darsia_visible() and not self.process_runner.active
        )

    def _configure_sidebar(
        self, visible: dict[str, set[str] | None], labels: dict[str, str]
    ) -> None:
        """Show the given DarSIA sidebar sections and, in each, the given rows."""
        for action, section in self.sidebar._sections.items():
            section.setVisible(action in visible)
            rows = visible.get(action)
            for row_id, row in section._rows.items():
                row.setVisible(rows is None or row_id in rows)
            _update_group_headers(section)
            label = labels.get(action, self._section_labels.get(action, ""))
            if label and section.category_label != label:
                section.category_label = label
                section.refresh_style()

    def _select_row(self, selection: tuple[str, str]) -> None:
        action, item_id = selection
        section = self.sidebar._sections.get(action)
        row = section.get_row(item_id) if section is not None else None
        if row is None or row.isHidden():
            return
        section.set_expanded(True)
        self.sidebar.select(action, item_id)

    def _show_setup_step(self) -> None:
        key = self._setup_step
        if key == "geometry":
            self._set_run_controls_visible(False)
            self.native_stack.setCurrentWidget(self.geometry_workspace)
            self.geometry_workspace.ensure_page()
        else:
            visible, labels, _default = self._STEP_SIDEBAR[key]
            self._configure_sidebar(visible, labels)
            self.native_stack.setCurrentWidget(self.original_content)
            self._set_run_controls_visible(True)
            self._ensure_run_loaded()
            self._select_row(self._step_selection[key])
        self._update_run_action()

    def _remember_native_selection(self, action: str, item_id: str) -> None:
        tab = self.workspace_tabs.currentIndex()
        if tab == self._SETUP_TAB and self._setup_step in self._STEP_SIDEBAR:
            self._step_selection[self._setup_step] = (action, item_id)
        elif tab == self._ORIGINAL_TAB:
            self._last_original_selection = (action, item_id)

    def _restore_selection(self) -> None:
        """Select the current step's row again (opening a file clears it)."""
        tab = self.workspace_tabs.currentIndex()
        if tab == self._SETUP_TAB and self._setup_step in self._STEP_SIDEBAR:
            self._select_row(self._step_selection[self._setup_step])
        elif tab == self._ORIGINAL_TAB:
            self._select_row(self._last_original_selection)

    def _show_native_mode(self, *, setup: bool) -> None:
        self.workspace_pages.setCurrentWidget(self.native_page)
        self.setup_steps.setVisible(setup)
        self.original_heading.setVisible(not setup)
        if setup:
            self._show_setup_step()
            return
        self._configure_sidebar(self._ORIGINAL_SIDEBAR, {})
        self.native_stack.setCurrentWidget(self.original_content)
        self._set_run_controls_visible(True)
        self._ensure_run_loaded()
        if not self._row_shown(self._last_original_selection):
            self._last_original_selection = ("calibration", OPTUNA_MASS_ROW)
        self._select_row(self._last_original_selection)

    def _row_shown(self, selection: tuple[str, str]) -> bool:
        section = self.sidebar._sections.get(selection[0])
        row = section.get_row(selection[1]) if section is not None else None
        return row is not None and not row.isHidden() and not section.isHidden()

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
                shown = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
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
        QSettings().setValue(f"native_app/run/{series_id}", self.run_combo.itemText(index))
        self._load_run_config(path)

    def _load_run_config(self, path: str) -> None:
        """Open a run's configuration in DarSIA's settings panel."""
        self.config_path_label.setText(str(path))
        self.config_controller.load_config()
        self._restore_selection()

    def _ensure_run_loaded(self) -> None:
        """Open the chosen run when no file of the selected series is open."""
        path = self.run_combo.currentData()
        if not path:
            return
        if not self.config_file or self._config_series != self.phase_model_workspace.series_id:
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

    def _workspace_changed(self, index: int) -> None:
        if index == self._START_TAB:
            self.workspace_pages.setCurrentWidget(self.home_page)
            self._refresh_series()
        elif index in (self._SETUP_TAB, self._ORIGINAL_TAB):
            selected = self.home_page.selected_series()
            if selected and selected != self.phase_model_workspace.series_id:
                self._select_series(selected)
            self._show_native_mode(setup=index == self._SETUP_TAB)
        elif index == self._PHASE_MODEL_TAB:
            self.workspace_pages.setCurrentWidget(self.phase_model_workspace)
            self.phase_model_workspace.ensure_studio()
            self.phase_model_workspace._show_selected_studio_mode()
        elif index == self._ANALYSIS_TAB:
            self.workspace_pages.setCurrentWidget(self.ensure_mass_analysis())
        self._update_run_action()

    def run_selected_workflow(self) -> None:
        """The original Run action applies only while DarSIA's steps are shown."""
        if self._darsia_visible():
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
