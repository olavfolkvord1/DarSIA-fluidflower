"""Native owner wiring for explicit Jobs registrations; no implicit execution."""

from copy import deepcopy
from pathlib import Path
import tomllib
import traceback

from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .jobs_metadata import create_job_from_plan
from .jobs_panel import JobsPanel
from .native_jobs_registry import JobsUIRegistry
from .pipeline.job_queue import name
from .store_map_ui import StoreAvailabilityPanel


class NativeJobsWorkspace(QWidget):
    """One passive series-scoped UI owner, created lazily by native navigation.

    Registration/selection is local UI state, not evidence that a scientific job
    was built for a series. Only the original B5 exact-plan admission can create
    a job after the explicit dialog action.
    """

    def __init__(
        self,
        parent=None,
        *,
        registry=None,
        measurement_registry=None,
        local_machine=None,
        reader=None,
        job_creator=None,
        local_launcher=None,
        remote_launcher=None,
        protocol_reader=None,
        selection_resolver=None,
        inventory_provider=None,
        resource_factory=None,
        store_panel_factory=StoreAvailabilityPanel,
    ):
        super().__init__(parent)
        default = (
            Path(
                QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation
                )
            )
            / "native_jobs"
        )
        self.registry = registry or JobsUIRegistry(default)
        self._creator = job_creator or create_job_from_plan
        self.protocol_reader, self.selection_resolver = (
            protocol_reader,
            selection_resolver,
        )
        self.series_root = self.series_id = None
        self.series_label = ""
        self._local_inventory = {}
        layout = QVBoxLayout(self)
        self.association_note = QLabel(
            "Choose a registered series. Only explicitly chosen UNC jobs/plans are listed; no NAS history is searched."
        )
        self.association_note.setWordWrap(True)
        self.association_note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.association_note)
        # Where jobs and plans come from, collapsed so the job list gets the page (N141).
        self.sources = QGroupBox("Sources")
        self.sources.setCheckable(True)
        self.sources.setChecked(False)
        sources_layout = QVBoxLayout(self.sources)
        self.sources_body = QWidget()
        body_layout = QVBoxLayout(self.sources_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        sources_layout.addWidget(self.sources_body)
        self.sources_body.setVisible(False)
        self.sources.toggled.connect(self.sources_body.setVisible)
        layout.addWidget(self.sources)
        self.stores = store_panel_factory(self.sources_body)
        body_layout.addWidget(self.stores)
        actions = QHBoxLayout()
        self.open_job_button = QPushButton("Open existing job…")
        self.import_plan_button = QPushButton("Import adopted execution plan…")
        self.open_job_button.clicked.connect(self.open_existing_job)
        self.import_plan_button.clicked.connect(self.import_execution_plan)
        actions.addWidget(self.open_job_button)
        actions.addWidget(self.import_plan_button)
        actions.addStretch()
        body_layout.addLayout(actions)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_label)
        self.panel = JobsPanel(
            local_machine=local_machine,
            reader=reader,
            job_creator=self._create_job,
            local_launcher=local_launcher,
            remote_launcher=remote_launcher,
            known_jobs_provider=self.registry.known_jobs,
            local_stores_path_provider=lambda: self.stores.path,
        )
        self.resources = None
        if resource_factory is not None:
            self.attach_resources(
                resource_factory(
                    measurement_registry=measurement_registry,
                    inventory_provider=inventory_provider,
                )
            )
        layout.addWidget(self.panel, 1)
        self.panel.job_created.connect(self._job_created)
        self.panel.job_table.cellClicked.connect(self._job_clicked)
        self.panel.job_table.itemActivated.connect(
            lambda item: self._job_clicked(item.row(), item.column())
        )
        self.panel.job_table.setToolTip(
            "Click or activate a job to remember it as active for this registered series."
        )
        self.panel.error_reported.connect(self.status_label.setText)
        self._clear_context()

    def attach_resources(self, resources):
        """Attach an explicitly supplied optional UI panel; B5 has no B6 import."""
        if self.resources is not None:
            raise ValueError("Resource panel already attached")
        self.panel.set_resource_panel(resources)
        self.resources = resources
        self.resources.hardware_inspected.connect(self._hardware_ready)

    def _clear_context(self):
        self.series_root = self.series_id = None
        self.series_label = ""
        self.stores.set_context(None, None)
        self.open_job_button.setEnabled(False)
        self.import_plan_button.setEnabled(False)
        self.panel.set_context(
            series_label="",
            series_root=None,
            series_id=None,
            jobs_provider=lambda: [],
            plans_provider=lambda: [],
            machines_provider=self._machines_provider,
        )

    def set_series(self, series_id, series_root, *, series_label=""):
        if not series_root or not series_id:
            self._clear_context()
            return
        try:
            root = Path(series_root)
            raw = (root / "series.toml").read_bytes()
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("Series manifest is too large for Jobs UI metadata")
            doc = tomllib.loads(raw.decode("utf-8-sig"))
            actual = doc.get("series_id")
            if (
                doc.get("schema_version") != 1
                or not isinstance(actual, str)
                or not actual
            ):
                raise ValueError(
                    "A registered schema-1 series with its own series_id is required"
                )
            # The registered folder's literal manifest ID is authoritative for
            # B4, not a displayed alias or a previously loaded run config.
            self.series_root, self.series_id = root, actual
            self.series_label = str(series_label or actual)
            self.stores.set_context(root, actual)
            self.open_job_button.setEnabled(True)
            self.import_plan_button.setEnabled(True)
            self.association_note.setText(
                f"Series: {self.series_label} · manifest ID: {actual}. Existing-job registration is a local UI association; scientific source bindings remain in the job itself."
            )
            self.panel.set_context(
                series_label=self.series_label,
                series_root=root,
                series_id=actual,
                jobs_provider=lambda: self.registry.jobs(root, actual),
                plans_provider=lambda: self.registry.plans(root, actual),
                machines_provider=self._machines_provider,
                selection_provider=self._selection_provider(root, actual),
            )
        except Exception:
            text = traceback.format_exc()
            self._clear_context()
            self.status_label.setText(text)

    def _selection_provider(self, root, series_id):
        def resolve(runs):
            protocol_reader, resolver = self.protocol_reader, self.selection_resolver
            if protocol_reader is None or resolver is None:
                from .series_image_selection import (
                    load_run_protocol_metadata,
                    resolve_saved_selection,
                )

                protocol_reader = protocol_reader or load_run_protocol_metadata
                resolver = resolver or resolve_saved_selection
            result = []
            for run in runs:
                protocols = protocol_reader(root, run)
                row = resolver(root, run, protocols, check_files=False)
                if row.get("series_id") != series_id:
                    raise ValueError("B4 resolver belongs to another registered series")
                result.append(row)
            return result

        return resolve

    def _machines_provider(self):
        local = self.panel.local_machine
        rows = {
            name(local): dict(
                machine=local, inventory=deepcopy(self._local_inventory), history=[]
            )
        }
        for fact in (self.panel.current or {}).get("registered_machines", []):
            machine = fact["machine"]
            key = name(machine)
            rows.setdefault(key, dict(machine=machine, inventory={}, history=[]))
            rows[key].setdefault("registrations", []).append(deepcopy(fact["source"]))
        return list(rows.values())

    def _hardware_ready(self, inventory):
        self._local_inventory = deepcopy(inventory)
        if self.resources is not None:
            self.resources.refresh_known_machines()

    def open_existing_job(self):
        if self.series_root is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open an explicit UNC job.json", "", "Job metadata (job.json)"
        )
        if not path:
            return
        if Path(path).name.casefold() != "job.json":
            self.status_label.setText("Choose the existing job.json itself.")
            return
        self.register_existing_job(Path(path).parent)

    def register_existing_job(self, root):
        if self.series_root is None:
            return
        try:
            self.registry.register_job(
                self.series_root, self.series_id, root, activate=True
            )
            self.status_label.setText(
                "Existing job registered locally. No task, result or worker was changed."
            )
            self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def import_execution_plan(self):
        if self.series_root is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import an existing adopted UNC execution plan", "", "TOML (*.toml)"
        )
        if not path:
            return
        try:
            self.registry.register_plan(self.series_root, self.series_id, path)
            self.status_label.setText(
                "Existing adopted plan registered locally. Create still requires an exact saved B4 source/time/order match; a new selection needs its authorized producer."
            )
            self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def refresh_registered(self):
        if self.series_root is None:
            return
        self.set_series(
            self.series_id, self.series_root, series_label=self.series_label
        )

    def _job_clicked(self, row, _column):
        if self.series_root is None or not 0 <= row < len(self.panel.snapshots):
            return
        try:
            self.registry.set_active(
                self.series_root, self.series_id, self.panel.snapshots[row]["root"]
            )
            for registration in self.panel.registrations:
                registration["active"] = (
                    registration["root"] == self.panel.snapshots[row]["root"]
                )
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def _create_job(self, plan, job_root, **kwargs):
        result = self._creator(plan, job_root, **kwargs)
        # UI-only result fields preserve the actual captured context even when
        # a finite background create finishes after navigation to another series.
        return dict(
            result,
            ui_series_root=str(kwargs["series_root"]),
            ui_series_id=kwargs["series_id"],
        )

    def _job_created(self, registration):
        try:
            root = registration["ui_series_root"]
            series_id = registration["ui_series_id"]
            self.registry.register_job(
                root,
                series_id,
                registration["root"],
                label=registration.get("name", ""),
                activate=True,
            )
            self.status_label.setText(
                f"Job metadata creation recorded locally for series {series_id}; no worker is started automatically."
            )
            if (
                self.series_root is not None
                and Path(root) == self.series_root
                and series_id == self.series_id
            ):
                self.refresh_registered()
        except Exception:
            self.status_label.setText(traceback.format_exc())

    def start_analysis(self):
        if self.series_root is None:
            self.status_label.setText(
                "Choose a registered series before starting analysis."
            )
            return
        self.panel.open_create_dialog()

    def install_mass_action(self, page, callback=None):
        if hasattr(page, "native_start_analysis_button"):
            return page.native_start_analysis_button
        button = QPushButton("Start analysis…", page)
        button.setToolTip(
            "Choose an already adopted execution plan that exactly matches this series' saved image selection. No job starts by opening the dialog."
        )
        button.clicked.connect(callback or self.start_analysis)
        page.layout().insertWidget(0, button)
        page.native_start_analysis_button = button
        return button

    def can_close(self):
        return self.panel.can_close() and self.stores.can_close() and (
            self.resources is None or self.resources.can_close()
        )

    def closeEvent(self, event):
        if not self.can_close():
            self.status_label.setText(
                "Job/store metadata or optional resource work is still finishing. Close the local Settings dialog first."
            )
            event.ignore()
            return
        super().closeEvent(event)
