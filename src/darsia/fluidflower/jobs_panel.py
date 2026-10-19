"""Native Jobs UI for the existing whole-run queue; no legacy execution engine."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import traceback
import uuid

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QSignalBlocker,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .job_control import publish_control, read_control
from .jobs_metadata import JobSnapshotReader, create_job_from_plan
from .storage_profile import (
    DEFAULT_STORAGE_PROFILE,
    ESTIMATE_LABEL,
    PHASE_TWO_ROUTE,
    SERIES_ROUTE,
    estimate_job_bytes,
    job_storage_profile,
)
from .pipeline.job_queue import descriptor, name, require_unc, write_immutable
from .remote_worker_launch import (
    RemoteWorkerTarget,
    build_remote_worker_launch,
    launch_remote_worker,
)
from .job_launch_state import (
    UNKNOWN,
    begin_launch_request,
    declare_requests_stopped,
    launch_key,
    read_launch_requests,
    read_registered_launch_requests,
    record_launch_error,
)

CONTROL_BOUNDARY = "Pause and stop exit after the current committed frame. Requested control is not proof of worker retirement or completed results; outputs and the original claim are preserved."
CONTROL_LEASE = "Waiting for the existing claim lease to expire (up to 10 minutes)"
REMOTE_GAP = "Remote start needs explicit existing SSH/checkout/interpreter facts and a confirmed PowerShell SSH default shell. Registration never contacts or installs a machine."


def _label(text=""):
    value = QLabel(text)
    value.setTextFormat(Qt.TextFormat.PlainText)
    value.setWordWrap(True)
    return value


def _table(headers):
    value = QTableWidget(0, len(headers))
    value.setHorizontalHeaderLabels(headers)
    value.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    value.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    value.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    value.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.ResizeToContents
    )
    value.horizontalHeader().setStretchLastSection(True)
    return value


def _fill(table, row, values):
    for column, value in enumerate(values):
        item = QTableWidgetItem(str(value))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, column, item)


def _rate(value):
    return (
        f"{value:.2f} frames/min" if isinstance(value, (int, float)) else "Not recorded"
    )


def _eta(value):
    return f"{value / 60:.1f} min" if isinstance(value, (int, float)) else "Unknown"


def _size(value):
    """Decimal units, as the documented per-file sizes (1 GB = 10^9 bytes)."""
    for unit, scale in (("TB", 10**12), ("GB", 10**9), ("MB", 10**6)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} bytes"


def choice_estimate_text(profile, free_space=None):
    """The chosen profile's shared estimate and known free space, before Create."""
    series = estimate_job_bytes(profile, 1, pipeline=SERIES_ROUTE)
    phase_two = estimate_job_bytes(profile, 1, pipeline=PHASE_TWO_ROUTE)
    text = (
        f"Storage estimate for {profile} (AC52 frame canvas; a larger canvas scales up): "
        f"{_size(series['per_frame_bytes'])} per frame in the native-phase-N1 series route, "
        f"{_size(phase_two['per_frame_bytes'])} per frame in phase two (N1)"
    )
    if profile == "lean":
        dropped = estimate_job_bytes(
            "lean", 1, pipeline=SERIES_ROUTE, colour_intermediates="drop"
        )
        text += (
            f", plus {_size(series['per_run_bytes'])} per run for its constant inputs "
            f"({_size(dropped['per_frame_bytes'])} per frame in the series route if the run "
            "configuration drops the colour intermediates)"
        )
    text += (
        f". {ESTIMATE_LABEL}. Create reports the whole-job estimate; no worker starts."
    )
    if free_space:
        text += (
            "\nFree space on this machine's results volume (last check): "
            + _size(free_space["free_bytes"])
        )
    else:
        text += "\nFree space: not checked on this machine yet; the worker checks fresh free space before every claim."
    return text


class _CallWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, call):
        super().__init__()
        self.call = call

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.call())
        except Exception:
            self.failed.emit(traceback.format_exc())


class StartAnalysisDialog(QDialog):
    """Choose an already adopted saved execution plan and a fresh UNC job root."""

    def __init__(self, plans, parent=None, *, free_space=None):
        super().__init__(parent)
        self.setWindowTitle("Start analysis — create job metadata")
        self.plans = deepcopy(list(plans))
        self.free_space = deepcopy(free_space)
        layout = QVBoxLayout(self)
        layout.addWidget(
            _label(
                "Choose an already adopted, frozen execution-series plan. Create writes job files only; no worker is started and no image is computed."
            )
        )
        form = QFormLayout()
        self.plan_combo = QComboBox()
        for row in self.plans:
            self.plan_combo.addItem(str(row.get("label") or row["path"]), row)
        self.name_edit, self.job_edit = QLineEdit(), QLineEdit()
        self.job_edit.setPlaceholderText(r"\\server\share\DarSIA\jobs\fresh-job-name")
        form.addRow("Saved execution plan", self.plan_combo)
        form.addRow("Job name", self.name_edit)
        form.addRow("Fresh UNC job folder", self.job_edit)
        # The job's storage profile is chosen here, once, before any worker.
        self.profile_combo = QComboBox()
        self.profile_combo.addItem(
            "Full — today's layout (references, qualification, exact gates)", "full"
        )
        self.profile_combo.addItem(
            "Lean — production layout: same computation, fewer stored files", "lean"
        )
        self.profile_combo.setCurrentIndex(
            self.profile_combo.findData(DEFAULT_STORAGE_PROFILE)
        )
        form.addRow("Storage profile", self.profile_combo)
        layout.addLayout(form)
        browse = QPushButton("Import saved execution plan…")
        browse.clicked.connect(self._import_plan)
        layout.addWidget(browse)
        self.binding_label = _label()
        layout.addWidget(self.binding_label)
        self.estimate_label = _label()
        layout.addWidget(self.estimate_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Create job files"
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.plan_combo.currentIndexChanged.connect(self._update_binding)
        self.job_edit.textChanged.connect(self._update_binding)
        self.profile_combo.currentIndexChanged.connect(self._update_estimate)
        self._update_binding()
        self._update_estimate()

    def _update_estimate(self, *_):
        self.estimate_label.setText(
            choice_estimate_text(self.profile_combo.currentData(), self.free_space)
        )

    def _update_binding(self, *_):
        plan = self.plan_combo.currentData() or {}
        self.binding_label.setText(
            f"Plan: {plan.get('path') or 'Not selected'}\nSHA256: {plan.get('sha256') or 'Not selected'}\nThe existing producer validates adoption, ordered runs and all scientific bindings."
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(plan) and bool(self.job_edit.text().strip())
        )

    def _import_plan(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an existing adopted execution plan", "", "TOML (*.toml)"
        )
        if not path:
            return
        try:
            require_unc(path)
            bound = descriptor(Path(path))
        except Exception:
            QMessageBox.warning(self, "Could not select plan", traceback.format_exc())
            return
        row = dict(bound, label=Path(path).name)
        self.plan_combo.addItem(row["label"], row)
        self.plan_combo.setCurrentIndex(self.plan_combo.count() - 1)

    def choices(self):
        return {
            "plan": deepcopy(self.plan_combo.currentData()),
            "job_root": self.job_edit.text().strip(),
            "display_name": self.name_edit.text().strip(),
            "storage_profile": self.profile_combo.currentData(),
        }


class RegisterMachineDialog(QDialog):
    """Record existing host/checkout/environment facts; never install or connect."""

    def __init__(self, parent=None, *, local_machine=None):
        super().__init__(parent)
        self.setWindowTitle("Register an existing machine")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(REMOTE_GAP))
        form = QFormLayout()
        self.local_machine = local_machine or socket.gethostname()
        (
            self.machine_edit,
            self.host_edit,
            self.checkout_edit,
            self.python_edit,
            self.ssh_edit,
        ) = [QLineEdit() for _ in range(5)]
        for label, edit in [
            ("Machine name", self.machine_edit),
            ("Existing host", self.host_edit),
            ("Existing Git checkout", self.checkout_edit),
            ("Existing Python interpreter", self.python_edit),
            ("Existing SSH alias", self.ssh_edit),
        ]:
            form.addRow(label, edit)
        self.ssh_executable_edit = QLineEdit()
        self.ssh_executable_edit.setReadOnly(True)
        ssh_row = QHBoxLayout()
        ssh_row.addWidget(self.ssh_executable_edit)
        browse = QPushButton("Choose existing local ssh.exe…")
        browse.clicked.connect(self._choose_ssh)
        ssh_row.addWidget(browse)
        form.addRow("Local SSH executable", ssh_row)
        self.default_shell_combo = QComboBox()
        self.default_shell_combo.addItem("Not specified — remote start unavailable", "")
        self.default_shell_combo.addItem(
            "Confirmed SSH default shell: Windows PowerShell", "windows-powershell"
        )
        form.addRow("SSH server's actual command interpreter", self.default_shell_combo)
        self.new_version = QCheckBox(
            "Save as a new immutable version; keep earlier registration"
        )
        form.addRow(self.new_version)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Register facts only")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for edit in (
            self.machine_edit,
            self.host_edit,
            self.checkout_edit,
            self.python_edit,
            self.ssh_edit,
            self.ssh_executable_edit,
        ):
            edit.textChanged.connect(self._validate)
        self.default_shell_combo.currentIndexChanged.connect(self._validate)
        self._validate()

    def _choose_ssh(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an already installed local ssh.exe", "", "ssh.exe (ssh.exe)"
        )
        if path and Path(path).is_file():
            self.ssh_executable_edit.setText(path)

    def _validate(self, *_):
        facts = self.facts()
        try:
            local = name(facts["machine"]) == name(self.local_machine)
        except ValueError:
            local = False
        basic = all(facts.get(key) for key in ("machine", "host", "checkout", "python"))
        remote = all(
            facts.get(key) for key in ("ssh_alias", "ssh_executable", "default_shell")
        )
        self.ok_button.setEnabled(bool(basic and (local or remote)))

    def facts(self):
        return {
            "machine": self.machine_edit.text().strip(),
            "host": self.host_edit.text().strip(),
            "checkout": self.checkout_edit.text().strip(),
            "python": self.python_edit.text().strip(),
            "ssh_alias": self.ssh_edit.text().strip(),
            "ssh_executable": self.ssh_executable_edit.text().strip(),
            "default_shell": self.default_shell_combo.currentData(),
            "remote_contacted": False,
            "installed": False,
        }


def launch_local_worker(
    job_root,
    *,
    backend,
    workers,
    machine,
    python=sys.executable,
    cwd=None,
    launch_directory=None,
    stores=None,
):
    """Explicit existing worker command via detached QProcess, never a generated script.

    Detaching preserves valid workers when the GUI closes. Fresh launch console
    files retain startup failures before the service's own job log is opened.
    Launch PID is only a launcher fact, never evidence of numerical progress.
    """
    _admit_gui_job_root(job_root)
    if backend not in {"cpu", "cuda"} or type(workers) is not int or workers < 1:
        raise ValueError("Explicit CPU/CUDA backend and positive workers required")
    from .store_map import read_store_map
    from .pipeline.job_queue import read_json
    from darsia_cache.adapters import independent_worker_environment
    import os
    selected_stores=read_store_map(stores)
    selected_job=read_json(Path(job_root)/"job.json")
    child_environment=independent_worker_environment(selected_stores,machine=machine,
        series_id=selected_job.get("series_id"),explicit_stores=stores is not None,
        parent_environment=os.environ)
    if cwd is None:
        from .pipeline import worker as implementation

        cwd = Path(implementation.__file__).resolve().parents[3]
    root = Path(job_root)
    job_ref = _job_reference(root, root / "job.json")
    launch_dir = (
        Path(launch_directory)
        if launch_directory is not None
        else root / "ui" / "launches" / uuid.uuid4().hex
    )
    console = launch_dir / "console.log"
    # Alias launch receipts are portable metadata, not copies of this host's argv.
    alias_console = (
        console.relative_to(root).as_posix() if "relative_path" in job_ref else None
    )
    launch_dir.mkdir(parents=True)
    worker_command = [
        str(python),
        "-m",
        "darsia.fluidflower.worker",
        "--job",
        str(root),
        "--backend",
        backend,
        "--workers",
        str(workers),
        "--machine",
        machine,
        "--watchdog",
    ]
    if stores is not None:
        if not isinstance(stores, (str, Path)) or not str(stores).strip():
            raise ValueError("Choose an existing local store map before Start")
        worker_command.extend(["--stores", str(stores)])
    # N109: the stdlib supervisor starts the unchanged worker command once more
    # when it stops before the service started; it imports no DarSIA code.
    service_log = root / "logs" / (name(machine + "-" + backend) + ".log")
    command = [str(python), "-B", "-m", "darsia_service.prestart",
               "--service-log", str(service_log), "--", *worker_command]
    process = QProcess()
    environment=QProcessEnvironment()
    for key,value in child_environment.items():
        environment.insert(key,value)
    process.setProcessEnvironment(environment)
    process.setProgram(command[0])
    process.setArguments(command[1:])
    process.setWorkingDirectory(str(cwd))
    process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    process.setStandardOutputFile(str(console))
    started, pid = process.startDetached()
    record = {
        "format": "darsia-job-local-launch-v1",
        "job_root": str(root),
        "command": command,
        "cwd": str(cwd),
        "requested_utc": datetime.now(timezone.utc).isoformat(),
        "started": bool(started),
        "launcher_pid": int(pid) if started else None,
        "console": str(console),
        "numerical_progress_claimed": False,
        "detached": True,
        "error": process.errorString() if not started else "",
    }
    if "relative_path" in job_ref:
        # Keep actual local argv/cwd/stores only in this process. The shared
        # receipt identifies the request without leaking its machine-local map.
        record = dict(
            format="darsia-job-local-launch-v2",
            job=job_ref,
            machine=machine, backend=backend, workers=workers,
            entrypoint="python -B -m darsia_service.prestart -- python -m darsia.fluidflower.worker",
            requested_utc=record["requested_utc"],
            started=record["started"], launcher_pid=record["launcher_pid"],
            console=alias_console, detached=True,
            worker_liveness_verified=False, numerical_progress_claimed=False,
            error=record["error"],
        )
    write_immutable(launch_dir / "launch.json", record)
    if not started:
        raise RuntimeError(record["error"])
    return record


def register_machine_facts(job_root, facts, *, local_machine=None, new_version=False):
    """Explicit metadata registration only. Existing registrations are not replaced."""
    _admit_gui_job_root(job_root)
    machine = name(facts["machine"])
    if not all(
        isinstance(facts.get(key), str) and facts[key].strip()
        for key in ("host", "checkout", "python")
    ):
        raise ValueError(
            "Supply the existing host, checkout and interpreter explicitly"
        )
    if machine != name(local_machine or socket.gethostname()):
        target = RemoteWorkerTarget(
            **{
                key: facts.get(key, "")
                for key in (
                    "machine",
                    "ssh_alias",
                    "python",
                    "checkout",
                    "ssh_executable",
                    "default_shell",
                )
            }
        )
        build_remote_worker_launch(
            str(job_root), target=target, backend="cpu", workers=1
        )
        if not Path(target.ssh_executable).is_file():
            raise FileNotFoundError(
                "The explicitly chosen local ssh.exe does not exist"
            )
    record = {
        **deepcopy(facts),
        "format": "darsia-job-machine-registration-v1",
        "job": _job_reference(Path(job_root), Path(job_root) / "job.json"),
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "remote_contacted": False,
        "installed": False,
    }
    path = (
        Path(job_root)
        / "ui"
        / "machines"
        / (machine + ("." + uuid.uuid4().hex if new_version else "") + ".json")
    )
    write_immutable(path, record)
    return descriptor(path)


class JobsPanel(QWidget):
    """All automatic work is bounded metadata reading from explicit registrations."""

    job_created = Signal(object)
    machine_registered = Signal(object)
    local_worker_launch_requested = Signal(object)
    remote_worker_launch_requested = Signal(object)
    operator_declaration_recorded = Signal(object)
    workers_selected = Signal(object)
    error_reported = Signal(str)

    def __init__(
        self,
        parent=None,
        *,
        jobs_provider=None,
        plans_provider=None,
        machines_provider=None,
        reader=None,
        job_creator=None,
        local_launcher=None,
        local_machine=None,
        selection_provider=None,
        remote_launcher=None,
        known_jobs_provider=None,
        control_reader=None,
        control_publisher=None,
        local_stores_path_provider=None,
    ):
        super().__init__(parent)
        self.jobs_provider, self.plans_provider, self.machines_provider = (
            jobs_provider,
            plans_provider,
            machines_provider,
        )
        self.reader = reader or JobSnapshotReader()
        self.job_creator = job_creator or create_job_from_plan
        self.local_launcher = local_launcher or launch_local_worker
        self.local_stores_path_provider = local_stores_path_provider
        self.remote_launcher = remote_launcher or launch_remote_worker
        self.local_machine = local_machine or socket.gethostname()
        self.selection_provider = selection_provider
        self.series_root = self.series_id = None
        self.known_jobs_provider = known_jobs_provider
        self.control_reader = control_reader or read_control
        self.control_publisher = control_publisher or publish_control
        self._known_launch_requests = []
        self._known_launch_errors = []
        self.series_label = ""
        self.registrations = []
        self.snapshots = []
        self.current = None
        self._read_thread = self._read_worker = self._create_thread = (
            self._create_worker
        ) = None
        self._generation = 0
        self.resource_panel = None
        self._launch_records = []
        self._refresh_pending = False
        self._session_requests = []
        self._control_thread = self._control_worker = None
        self._control_pending = False
        self._capacity_thread = self._capacity_worker = None
        self._known_free_space = None
        self._creation_estimates = {}
        self._resume_after_control = None
        layout = QVBoxLayout(self)
        self.context_label = _label("Jobs — no series selected")
        layout.addWidget(self.context_label)
        # The job list at full height on the left, the selected job on the
        # right; no table, log or error text has a height cap (N141).
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        listing = QWidget()
        list_layout = QVBoxLayout(listing)
        list_layout.setContentsMargins(0, 0, 0, 0)
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_tabs = QTabWidget()
        progress, files = QWidget(), QWidget()
        progress_layout, files_layout = QVBoxLayout(progress), QVBoxLayout(files)
        actions = QHBoxLayout()
        self.create_button, self.refresh_button = QPushButton(
            "Start analysis…"
        ), QPushButton("Refresh jobs")
        self.create_button.clicked.connect(self.open_create_dialog)
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.create_button)
        actions.addWidget(self.refresh_button)
        actions.addStretch()
        list_layout.addLayout(actions)
        self.job_table = _table(
            ["Name", "Series", "Runs", "Frames", "Created", "State"]
        )
        self.job_table.itemSelectionChanged.connect(self._selection_changed)
        list_layout.addWidget(self.job_table, 1)
        self.master_label, self.progress_label = _label("No active job"), _label("")
        detail_layout.addWidget(self.master_label)
        detail_layout.addWidget(self.progress_label)
        self.capacity_button = QPushButton("Check expected space on this machine")
        self.capacity_button.clicked.connect(self.check_capacity)
        self.capacity_label = _label("Expected space not checked. The worker checks fresh free space before claiming a run.")
        files_layout.addWidget(self.capacity_button)
        files_layout.addWidget(self.capacity_label)
        files_layout.addStretch()
        # Read-only: an existing job keeps the profile stored in its job.json.
        self.profile_label = _label("Storage profile: no job selected.")
        detail_layout.addWidget(self.profile_label)
        controls = QHBoxLayout()
        self.backend_combo, self.workers_spin = QComboBox(), QSpinBox()
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("CUDA", "cuda")
        self.workers_spin.setRange(1, 256)
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.start_button, self.add_machine_button = QPushButton(
            "Start selected machine"
        ), QPushButton("Add machine…")
        self.start_button.clicked.connect(self.start_selected_machine)
        self.add_machine_button.clicked.connect(self.open_machine_dialog)
        controls.addWidget(self.target_combo)
        controls.addWidget(self.backend_combo)
        controls.addWidget(self.workers_spin)
        controls.addWidget(self.start_button)
        self.pause_button, self.resume_button, self.stop_button = (
            QPushButton("Pause after current frame"),
            QPushButton("Resume"),
            QPushButton("Stop"),
        )
        for button in (self.pause_button, self.resume_button, self.stop_button):
            button.setEnabled(False)
            button.setToolTip(CONTROL_BOUNDARY)
            controls.addWidget(button)
        self.pause_button.clicked.connect(self.pause_job)
        self.resume_button.clicked.connect(self.resume_job)
        self.stop_button.clicked.connect(self.stop_job)
        controls.addWidget(self.add_machine_button)
        self.declare_stopped_button = QPushButton("Declare previous service stopped…")
        self.declare_stopped_button.clicked.connect(
            self.declare_previous_service_stopped
        )
        controls.addWidget(self.declare_stopped_button)
        detail_layout.addLayout(controls)
        self.launch_state_label = _label("No launch request for the selected target.")
        detail_layout.addWidget(self.launch_state_label)
        self.control_state_label = _label("Control unavailable: no job selected.")
        detail_layout.addWidget(self.control_state_label)
        self.control_note = _label(CONTROL_BOUNDARY + " " + REMOTE_GAP)
        detail_layout.addWidget(self.control_note)
        self.backend_combo.currentIndexChanged.connect(self._target_changed)
        self.machine_table = _table(
            ["Machine", "Committed frames", "Average throughput", "ETA", "Scope"]
        )
        progress_layout.addWidget(self.machine_table, 1)
        self.worker_table = _table(
            [
                "Machine",
                "Worker",
                "Run",
                "Reported state",
                "Last actual progress",
                "Verbatim error",
            ]
        )
        progress_layout.addWidget(self.worker_table, 1)
        self.resource_slot = QVBoxLayout()
        progress_layout.addLayout(self.resource_slot)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2500)
        self.error_text = QPlainTextEdit()
        self.error_text.setReadOnly(True)
        self.detail_tabs.addTab(progress, "Progress")
        self.detail_tabs.addTab(self.log_text, "Log")
        self.detail_tabs.addTab(self.error_text, "Errors")
        self.detail_tabs.addTab(files, "Files")
        detail_layout.addWidget(self.detail_tabs, 1)
        self.splitter.addWidget(listing)
        self.splitter.addWidget(details)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        layout.addWidget(self.splitter, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self.refresh)
        self._show_snapshot(None)

    def set_context(
        self,
        *,
        series_label="",
        series_root=None,
        series_id=None,
        jobs_provider=None,
        plans_provider=None,
        machines_provider=None,
        selection_provider=None,
    ):
        self._generation += 1
        self.series_label = str(series_label)
        self.series_root, self.series_id = series_root, series_id
        for key, value in (
            ("jobs_provider", jobs_provider),
            ("plans_provider", plans_provider),
            ("machines_provider", machines_provider),
        ):
            if value is not None:
                setattr(self, key, value)
        if selection_provider is not None:
            self.selection_provider = selection_provider
        self.create_button.setEnabled(
            bool(self.selection_provider and self.series_root and self.series_id)
        )
        self.create_button.setToolTip(
            ""
            if self.create_button.isEnabled()
            else "Create is blocked until the saved current-series B4 selection is bound to the existing execution plan."
        )
        self.context_label.setText(
            "Jobs — " + (self.series_label or "No series selected")
        )
        self.set_jobs([])
        if self.jobs_provider is not None:
            try:
                self.set_jobs(self.jobs_provider())
            except Exception:
                self._error(traceback.format_exc())
        self.refresh()

    def set_jobs(self, registrations):
        rows = deepcopy(list(registrations or []))
        if any(
            not isinstance(row, dict) or not isinstance(row.get("root"), str)
            for row in rows
        ):
            raise ValueError("Explicit registered job roots required")
        if len({str(Path(row["root"])) for row in rows}) != len(rows):
            raise ValueError("Duplicate registered job roots")
        if sum(bool(row.get("active")) for row in rows) > 1:
            raise ValueError("More than one explicitly active job")
        self.registrations = rows
        if not rows:
            self.snapshots = []
            self.job_table.setRowCount(0)
            self._show_snapshot(None)

    def refresh(self, *_):
        if self._read_thread is not None or self._control_pending:
            self._refresh_pending = True
            return
        if not self.registrations:
            return
        rows, generation = deepcopy(self.registrations), self._generation

        def read_all():
            result = []
            for row in rows:
                try:
                    snapshot = dict(self.reader(row))
                    snapshot["control"] = self._read_control_snapshot(row["root"])
                    result.append(snapshot)
                except Exception:
                    result.append(
                        {
                            "root": row["root"],
                            "name": row.get("name", Path(row["root"]).name),
                            "series": row.get("series_label", self.series_label),
                            "state": "Metadata error",
                            "errors": [traceback.format_exc()],
                        }
                    )
            requests, errors = self._read_known_launches()
            return {
                "generation": generation,
                "snapshots": result,
                "known_launch_requests": requests,
                "known_launch_errors": errors,
            }

        self._read_thread, self._read_worker = self._start_call(
            read_all, self._read_ready, self._read_closed
        )

    def _start_call(self, call, ready, closed):
        thread = QThread(self)
        worker = _CallWorker(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(ready)
        worker.failed.connect(self._error)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(closed)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        return thread, worker

    @Slot(object)
    def _read_ready(self, result):
        if result["generation"] != self._generation:
            return
        previous = self.current.get("root") if self.current else None
        self.snapshots = result["snapshots"]
        self._known_launch_requests = result.get("known_launch_requests", [])
        self._known_launch_errors = result.get("known_launch_errors", [])
        with QSignalBlocker(self.job_table):
            self.job_table.setRowCount(len(self.snapshots))
            for row, data in enumerate(self.snapshots):
                frames = f"{data.get('committed', '?')} / {data.get('total_frames') if data.get('total_frames') is not None else '?'}"
                _fill(
                    self.job_table,
                    row,
                    [
                        data["name"],
                        data.get("series", "Not recorded"),
                        f"{data.get('completed_runs', '?')} / {data.get('run_count', '?')}",
                        frames,
                        data.get("created_utc", "Not recorded"),
                        data["state"],
                    ],
                )
            active = next(
                (r["root"] for r in self.registrations if r.get("active")), None
            )
            desired = previous or active
            index = next(
                (i for i, s in enumerate(self.snapshots) if s["root"] == desired), 0
            )
            if self.snapshots:
                self.job_table.selectRow(index)
        self._selection_changed()

    @Slot()
    def _read_closed(self):
        # finished() precedes platform-thread teardown; retain its Python owner
        # until wait() confirms that the finite metadata task has truly exited.
        if self._read_thread is not None:
            self._read_thread.wait()
        self._read_thread = self._read_worker = None
        self._target_changed()
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _selection_changed(self):
        row = self.job_table.currentRow()
        self._show_snapshot(
            self.snapshots[row] if 0 <= row < len(self.snapshots) else None
        )

    @Slot()
    def check_capacity(self):
        if self._capacity_thread is not None or self.current is None:
            return
        root = self.current["root"]
        generation = self._generation
        stores_path = self.local_stores_path_provider() if self.local_stores_path_provider else None
        forbidden = tuple(p for p in (self.series_root, Path(__file__).resolve().parents[3]) if p)
        self.capacity_button.setEnabled(False)
        self.capacity_label.setText("Reading expected allocation and fresh free space on this machine…")
        def read_capacity():
            from .store_map import read_store_map
            from .pipeline.capacity import report_job_root
            try:
                if stores_path is None:
                    raise ValueError("Choose this machine's local store map")
                result = report_job_root(root,read_store_map(stores_path,forbidden_roots=forbidden))
            except (OSError,ValueError) as error:
                result = dict(available=False,reason=str(error))
            return dict(generation=generation,root=root,result=result)
        self._capacity_thread,self._capacity_worker=self._start_call(
            read_capacity,self._capacity_ready,self._capacity_closed)

    @Slot(object)
    def _capacity_ready(self, value):
        if value["generation"] != self._generation or value["root"] != (self.current or {}).get("root"):
            return
        result = value["result"]
        if result.get("available") is False:
            text = "Expected space unavailable: " + result["reason"]
        else:
            text = result["message"] + (" — enough by this estimate." if result["sufficient"] else " — insufficient by this estimate.")
            text += " Includes preserved native outputs, optional storage additions, estimated temporary space and reserve. Results: estimate, not coordinated across hosts. Fresh check repeats before claim; cache admission is separate."
            if type(result.get("free_bytes")) is int:
                # Free space this panel knows, shown when choosing a new job's profile.
                self._known_free_space = dict(free_bytes=result["free_bytes"],
                                              series_id=self.series_id,
                                              results_root=result.get("results_root"))
        self.capacity_label.setText(text)

    @Slot()
    def _capacity_closed(self):
        if self._capacity_thread is not None:
            self._capacity_thread.wait()
        self._capacity_thread=self._capacity_worker=None
        self.capacity_button.setEnabled(bool(self.current and self.local_stores_path_provider
            and (self.current.get("job_facts") or {}).get("format") == "darsia-alias-job-v1"))

    def _show_snapshot(self, data):
        old_root = (self.current or {}).get("root")
        self.current = data
        if (data or {}).get("root") != old_root:
            self.capacity_label.setText("Expected space not checked. The worker checks fresh free space before claiming a run.")
        self.profile_label.setText(self._profile_text(data))
        self.capacity_button.setEnabled(bool(data and self.local_stores_path_provider
            and (data.get("job_facts") or {}).get("format") == "darsia-alias-job-v1"
            and self._capacity_thread is None))
        enabled = bool(data and data.get("job_facts"))
        self.start_button.setEnabled(enabled)
        self.add_machine_button.setEnabled(enabled)
        self._refresh_targets()
        self.master_label.setText(
            f"Master: {data.get('master_machine', 'Not recorded')} · {data.get('name')} · {data.get('state')}"
            if data
            else "No active job"
        )
        if data:
            legacy = (data.get("job_facts") or {}).get("legacy_paths", True)
            self.master_label.setText(self.master_label.text() + "\nlegacy_paths=" +
                                      ("true" if legacy else "false"))
            self.progress_label.setText(
                f"Committed images: {data.get('committed', '?')} · Remaining: {data.get('remaining') if data.get('remaining') is not None else 'Unknown'} · Average: {_rate(data.get('frames_per_minute'))} · ETA: {_eta(data.get('eta_seconds'))}\n{data.get('rate_scope', '')}"
            )
        else:
            self.progress_label.setText("")
        self.machine_table.setRowCount(len((data or {}).get("machines", [])))
        for row, facts in enumerate((data or {}).get("machines", [])):
            _fill(
                self.machine_table,
                row,
                [
                    facts["machine"],
                    facts["committed"],
                    _rate(facts.get("frames_per_minute")),
                    _eta(facts.get("eta_seconds")),
                    facts.get("rate_scope", ""),
                ],
            )
        self.worker_table.setRowCount(len((data or {}).get("workers", [])))
        for row, facts in enumerate((data or {}).get("workers", [])):
            _fill(
                self.worker_table,
                row,
                [
                    facts["machine"],
                    facts["worker"],
                    facts["run"],
                    facts["state"],
                    facts["last_progress_utc"],
                    facts["error"],
                ],
            )
        logs = list((data or {}).get("logs", []))
        launches = list(self._launch_records) + self._job_requests()
        shown = set()
        for launch in launches:
            if launch.get("job_root") == (data or {}).get("root") and launch.get(
                "console"
            ):
                path = Path(launch["console"])
                if path.exists() and str(path) not in shown:
                    shown.add(str(path))
                    with path.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size - 128 * 1024))
                        logs.append(
                            {
                                "path": str(path),
                                "tail": stream.read().decode("utf-8", errors="replace"),
                            }
                        )
        self.log_text.setPlainText(
            "\n".join(f"{item['path']}\n{item['tail']}" for item in logs)
        )
        failures = [
            row.get("error", "")
            for row in (data or {}).get("runs", [])
            if row.get("error")
        ]
        failures.extend(
            row["error_traceback"]
            for row in self._job_requests()
            if row.get("error_traceback")
        )
        self.error_text.setPlainText(
            "\n".join((data or {}).get("errors", []) + failures)
        )
        if self.resource_panel is not None:
            self.resource_panel.set_context(
                (data or {}).get("job_facts"),
                series_label=self.series_label,
                machines_provider=self.machines_provider,
            )

    def _profile_text(self, data):
        """The selected job's stored storage profile; never editable here."""
        if not data:
            return "Storage profile: no job selected."
        facts = data.get("job_facts")
        if not facts:
            return "Storage profile: not shown, the job metadata could not be read."
        if facts.get("pipeline") == "native-phase-statistics" and "storage_profile" not in facts:
            return "Storage profile: not applicable; phase one writes no N1 frame products."
        try:
            profile = job_storage_profile(facts)
        except ValueError as error:
            return "Storage profile: invalid job metadata: " + str(error)
        source = (
            "stored in job.json"
            if "storage_profile" in facts
            else "legacy job.json without the field; old jobs stay full"
        )
        text = f"Storage profile: {profile} — {source}. Fixed for an existing job (read-only)."
        estimate = self._creation_estimates.get(str(data.get("root")))
        if estimate and estimate.get("complete"):
            text += (
                f"\nEstimate at creation: {_size(estimate['job_bytes'])} for "
                f"{estimate['frames']} frames, plus {_size(estimate['inflight_bytes'])} in flight. "
                f"{ESTIMATE_LABEL}."
            )
        else:
            text += "\nCheck expected space on this machine for this job's estimate against fresh free space."
        return text

    def set_resource_panel(self, panel):
        if self.resource_panel is not None:
            raise ValueError("Resource panel already connected")
        self.resource_panel = panel
        self.resource_scroll_area = QScrollArea(self)
        self.resource_scroll_area.setWidgetResizable(True)
        self.resource_scroll_area.setWidget(panel)
        self.resource_slot.addWidget(self.resource_scroll_area)
        panel.workers_selected.connect(self._resource_choice)
        self._show_snapshot(self.current)

    def _resource_choice(self, choice):
        try:
            chosen = name(choice["machine"])
        except (ValueError, KeyError):
            return
        matches = [
            i
            for i in range(self.target_combo.count())
            if name((self.target_combo.itemData(i) or {})["machine"]) == chosen
        ]
        if len(matches) == 1:
            self.target_combo.setCurrentIndex(matches[0])
            self.workers_spin.setValue(choice["workers"])
            self.backend_combo.setCurrentIndex(
                max(0, self.backend_combo.findData(choice["backend"]))
            )
        self.workers_selected.emit(deepcopy(choice))

    def open_create_dialog(self):
        if not self.selection_provider or not self.series_root or not self.series_id:
            self._error(
                "Create is blocked until the current-series B4 image selection is connected and exactly matches the saved execution plan."
            )
            return
        try:
            plans = self.plans_provider() if self.plans_provider else []
        except Exception:
            self._error(traceback.format_exc())
            return
        known = self._known_free_space
        dialog = StartAnalysisDialog(
            plans,
            self,
            free_space=known if known and known["series_id"] == self.series_id else None,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.create_job(**dialog.choices())

    def create_job(self, plan, job_root, display_name, storage_profile=None):
        if self._create_thread is not None:
            return
        self.create_button.setEnabled(False)
        kwargs = {
            "display_name": display_name,
            "series_label": self.series_label,
            "series_root": self.series_root,
            "series_id": self.series_id,
            "selection_provider": self.selection_provider,
        }
        if storage_profile is not None:
            kwargs["storage_profile"] = storage_profile
        generation = self._generation
        self._create_thread, self._create_worker = self._start_call(
            lambda: {
                "generation": generation,
                "registration": self.job_creator(plan, job_root, **kwargs),
            },
            self._created,
            self._create_closed,
        )

    @Slot(object)
    def _created(self, result):
        registration = result["registration"]
        receipt = (registration or {}).get("producer_receipt")
        if isinstance(receipt, dict) and isinstance(receipt.get("storage_estimate"), dict):
            self._creation_estimates[str(registration.get("root"))] = deepcopy(
                receipt["storage_estimate"]
            )
        self.job_created.emit(deepcopy(registration))
        if result["generation"] != self._generation:
            return
        self._generation += 1
        self.current = None
        for row in self.registrations:
            row["active"] = False
        self.registrations.append(deepcopy(registration))
        self.refresh()

    @Slot()
    def _create_closed(self):
        if self._create_thread is not None:
            self._create_thread.wait()
        self._create_thread = self._create_worker = None
        self.create_button.setEnabled(
            bool(self.selection_provider and self.series_root and self.series_id)
        )

    def start_local_workers(self):
        if not self.current or not self.current.get("job_facts"):
            return
        if self.current.get("errors"):
            self._error(
                "Worker start unavailable: selected job has metadata errors.\n"
                + "\n".join(self.current["errors"])
            )
            self._target_changed()
            return
        if not self._start_allowed(self.local_machine):
            return
        request = None
        try:
            local_options = {}
            if ((self.current.get("job_facts") or {}).get("format")
                    == "darsia-alias-job-v1"
                    and self.local_stores_path_provider is not None):
                selected_stores = self.local_stores_path_provider()
                if not isinstance(selected_stores, (str, Path)) or not str(selected_stores).strip():
                    raise ValueError("Choose an existing local store map before Start")
                local_options["stores"] = selected_stores
            request = self._begin_request(self.local_machine, "local")
            record = self.local_launcher(
                self.current["root"],
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
                machine=self.local_machine,
                launch_directory=request["launch_directory"],
                **local_options,
            )
        except Exception:
            self._launch_failed(request, traceback.format_exc())
            return
        self._launch_records.append(deepcopy(record))
        self.local_worker_launch_requested.emit(deepcopy(record))
        self.progress_label.setText(
            "Local worker launch requested. Only subsequent actual frame publications count as progress."
        )
        self.refresh()

    def _job_requests(self):
        if not self.current:
            return []
        rows = {
            row["source"]["path"]: row
            for row in self._session_requests
            if launch_key(row["job_root"], row["machine"], row["backend"])[0]
            == launch_key(self.current["root"], self.local_machine, "cpu")[0]
        }
        for row in self.current.get("launch_requests", []):
            previous = rows.get(row["source"]["path"], {})
            rows[row["source"]["path"]] = dict(
                row,
                error_traceback=row.get("error_traceback")
                or previous.get("error_traceback", ""),
            )
        return sorted(
            rows.values(), key=lambda row: (row["requested_utc"], row["source"]["path"])
        )

    def _pending_requests(self, machine):
        if not self.current:
            return []
        key = launch_key(
            self.current["root"], machine, self.backend_combo.currentData()
        )[1:]
        rows = {row["source"]["path"]: row for row in self._session_requests}
        for snapshot in self.snapshots:
            for row in snapshot.get("launch_requests", []):
                rows[row["source"]["path"]] = row
        # These are freshly bound reads from the optional all-series provider.
        for row in self._known_launch_requests:
            rows[row["source"]["path"]] = row
        for row in self._job_requests():
            if row["source"]["path"] not in rows:
                rows[row["source"]["path"]] = row
        return sorted(
            (
                row
                for row in rows.values()
                if not row["released"]
                and launch_key(row["job_root"], row["machine"], row["backend"])[1:]
                == key
            ),
            key=lambda row: (row["requested_utc"], row["source"]["path"]),
        )

    def _read_known_launches(self):
        if self.known_jobs_provider is None:
            return [], []
        try:
            return read_registered_launch_requests(list(self.known_jobs_provider()))
        except Exception:
            return [], [
                dict(
                    root="Local registered jobs", error_traceback=traceback.format_exc()
                )
            ]

    def _start_allowed(self, machine):
        facts = (self.current or {}).get("job_facts") or {}
        if (facts.get("format") == "darsia-alias-job-v1"
                and (facts.get("execution_ready") is not True
                     or facts.get("runtime_support") != "darsia-role-io-runtime-v1")):
            self._error(facts.get("execution_blocker") or "J3 alias-job runtime is not available")
            return False
        if self._control_pending:
            self._error(
                "A control command is still being published; wait for its result."
            )
            return False
        try:
            control = self._read_control_snapshot(self.current["root"])
            if control.get("error"):
                raise ValueError(control["reason"])
            if control["state"]["action"] != "run":
                raise ValueError(
                    "The job requests pause or stop. Use Resume on its recorded "
                    "master machine before starting a service."
                )
        except Exception:
            self._error(traceback.format_exc())
            return False
        # Explicit Start revalidates only known roots, never searches the NAS.
        self._known_launch_requests, self._known_launch_errors = (
            self._read_known_launches()
        )
        if self._pending_requests(machine) or self._known_launch_errors:
            self._target_changed()
            return False
        return True

    def _begin_request(self, machine, kind):
        request = begin_launch_request(
            self.current["root"],
            machine=machine,
            backend=self.backend_combo.currentData(),
            workers=self.workers_spin.value(),
            kind=kind,
        )
        self._session_requests.append(request)
        self._target_changed()
        return request

    def _launch_failed(self, request, text):
        if request is not None:
            try:
                record_launch_error(request, text)
            except Exception:
                text += (
                    "\nThe UI failure receipt could not be written:\n"
                    + traceback.format_exc()
                )
            request["error_traceback"] = text
        self._error(text)
        self._target_changed()
        self.refresh()

    def declare_previous_service_stopped(self):
        selected = self.target_combo.currentData() or {}
        pending = self._pending_requests(selected.get("machine", self.local_machine))
        if not pending:
            return
        reply = QMessageBox.question(
            self,
            "Operator declaration only",
            "Read the launch consoles first. Do you declare that the previous services for this machine and backend have stopped?\nRegistered jobs:\n"
            + "\n".join(sorted({row["job_root"] for row in pending}))
            + "\nThis records your statement only; it does not verify worker retirement, completed results or numerical progress.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            written = []
            roots = {row["job_root"] for row in pending}
            for root in sorted(roots):
                written.extend(
                    declare_requests_stopped(
                        root, [row for row in pending if row["job_root"] == root]
                    )
                )
                updated = read_launch_requests(root)
                self._session_requests = [
                    row
                    for row in self._session_requests
                    if launch_key(row["job_root"], row["machine"], row["backend"])[0]
                    != launch_key(root, self.local_machine, "cpu")[0]
                ] + updated
                for snapshot in self.snapshots:
                    if (
                        launch_key(snapshot["root"], self.local_machine, "cpu")[0]
                        == launch_key(root, self.local_machine, "cpu")[0]
                    ):
                        snapshot["launch_requests"] = updated
            self._known_launch_requests, self._known_launch_errors = (
                self._read_known_launches()
            )
        except Exception:
            self._error(traceback.format_exc())
            return
        self.operator_declaration_recorded.emit(written)
        self._target_changed()
        self.refresh()

    def _refresh_targets(self):
        previous = (self.target_combo.currentData() or {}).get("key")
        rows = [dict(key="local", machine=self.local_machine, local=True)]
        for record in (self.current or {}).get("registered_machines", []):
            if name(record["machine"]) == name(self.local_machine):
                continue
            rows.append(dict(record, key=record["source"]["path"], local=False))
        with QSignalBlocker(self.target_combo):
            self.target_combo.clear()
            for row in rows:
                label = row["machine"] + (
                    " (this machine)"
                    if row["local"]
                    else " · "
                    + str(row.get("registered_utc") or "saved facts")
                    + " · "
                    + row["source"]["sha256"][:8]
                )
                self.target_combo.addItem(label, row)
            index = next((i for i, row in enumerate(rows) if row["key"] == previous), 0)
            self.target_combo.setCurrentIndex(index)
        self._target_changed()

    def _target(self, record):
        return RemoteWorkerTarget(
            **{
                key: record.get(key, "")
                for key in (
                    "machine",
                    "ssh_alias",
                    "python",
                    "checkout",
                    "ssh_executable",
                    "default_shell",
                )
            }
        )

    def _target_changed(self, *_):
        selected = self.target_combo.currentData() or {}
        job = self.current or {}
        enabled = bool(job.get("job_facts")) and not job.get("errors")
        reason = ""
        if ((job.get("job_facts") or {}).get("format") == "darsia-alias-job-v1"
                and (job["job_facts"].get("execution_ready") is not True
                     or job["job_facts"].get("runtime_support") != "darsia-role-io-runtime-v1")):
            enabled = False
            reason = job["job_facts"].get("execution_blocker") or "J3 alias-job runtime is not available"
        if enabled and not selected.get("local"):
            try:
                build_remote_worker_launch(
                    job["root"],
                    target=self._target(selected),
                    backend=self.backend_combo.currentData(),
                    workers=self.workers_spin.value(),
                )
            except Exception:
                enabled = False
                reason = traceback.format_exc()
        pending = self._pending_requests(selected.get("machine", self.local_machine))
        self.declare_stopped_button.setEnabled(bool(pending))
        if pending:
            enabled = False
            reason = (
                UNKNOWN
                + ". Read the saved launch receipt/console; a client exit is not worker retirement."
            )
            self.launch_state_label.setText(
                reason
                + "\n"
                + "\n".join(
                    "Registered job: "
                    + row["job_root"]
                    + "\nConsole: "
                    + row["console"]
                    for row in pending
                )
            )
        elif self._known_launch_errors:
            enabled = False
            reason = "Launch state unavailable for a registered job; Start remains blocked until its own metadata can be read."
            self.launch_state_label.setText(
                reason
                + "\n"
                + "\n".join(
                    row["root"] + "\n" + row["error_traceback"]
                    for row in self._known_launch_errors
                )
            )
        else:
            rows = [
                row
                for row in self._job_requests()
                if launch_key(row["job_root"], row["machine"], row["backend"])
                == launch_key(
                    job.get("root", ""),
                    selected.get("machine", self.local_machine),
                    self.backend_combo.currentData(),
                )
            ]
            self.launch_state_label.setText(
                rows[-1]["state"]
                if rows
                else "No launch request for the selected target."
            )
        control = job.get("control") or {}
        if self._control_pending or control.get("error") or (
            control.get("state", {}).get("action", "run") != "run"
        ):
            enabled = False
            reason = reason or (
                "The job is paused/stopped or its control metadata is unavailable. "
                "Resume requires its recorded master machine."
            )
        self.start_button.setEnabled(enabled)
        self.start_button.setToolTip(
            reason
            or "Explicit start of the existing whole-run worker; frame publications alone show progress."
        )
        self.control_note.setText(CONTROL_BOUNDARY + " " + (reason or REMOTE_GAP))
        self._show_control_state()

    def _read_control_snapshot(self, root):
        """Read commands and the bound master; never initialize a UI namespace."""
        try:
            state = self.control_reader(root)
            path = Path(root) / "ui" / "master.json"
            if not path.is_file():
                return dict(
                    available=False,
                    state=state,
                    reason="No recorded master is present for this job; frame controls are unavailable.",
                )
            asset = descriptor(path)
            master = JobSnapshotReader().read(path, asset)
            from .job_control import _bindings
            reference, master_format, _control_format = _bindings(Path(root))
            if (master.get("format") != master_format
                    or master.get("job") != reference(Path(root) / "job.json")):
                raise ValueError("Job UI master is not bound to this immutable job")
            machine = master.get("machine")
            if not isinstance(machine, str) or not machine.strip():
                raise ValueError("Job UI master machine is missing")
            return dict(
                available=True, state=state, master=asset, master_machine=machine
            )
        except Exception:
            return dict(available=False, error=True, reason=traceback.format_exc())

    def _control_reason(self, action):
        job = self.current or {}
        control = job.get("control") or {}
        if not job.get("job_facts") or job.get("errors"):
            return "Control unavailable: selected job has no valid execution metadata."
        if not control.get("available"):
            return control.get("reason") or (
                "Refresh the job's control metadata before applying a command."
            )
        if control.get("master_machine") != self.local_machine:
            return "Only the job's recorded master machine writes commands: " + str(
                control.get("master_machine")
            )
        if self._control_pending or self._read_thread is not None:
            return "A metadata read or control publication is still finishing."
        if job.get("complete"):
            return "The job is complete; no remaining run needs control."
        current = control["state"]["action"]
        if action == "run" and current not in {"pause", "stop"}:
            return "Resume requires a displayed pause or stop command."
        if action == current:
            return "This requested state is already displayed."
        return ""

    def _show_control_state(self):
        control = (self.current or {}).get("control") or {}
        if not control.get("available"):
            self.control_state_label.setText(
                "Control unavailable: "
                + str(control.get("reason") or "no job selected.")
            )
        else:
            state = control["state"]
            command = state.get("command")
            text = (
                "Requested state: "
                + state["action"]
                + " · command sequence "
                + str(state["sequence"])
            )
            text += (
                "\nRecorded master: "
                + control["master_machine"]
                + " · This machine: "
                + self.local_machine
            )
            if command:
                text += (
                    "\nDisplayed command: "
                    + str(command.get("relative_path", command.get("path")))
                    + " · SHA-256 "
                    + command["sha256"]
                    + " · "
                    + str(command.get("bytes", command.get("size_bytes")))
                    + " bytes"
                )
            else:
                text += "\nNo command has been recorded (sequence 0)."
            if (
                state["action"] == "run"
                and state["sequence"]
                and not (self.current or {}).get("complete")
            ):
                text += (
                    "\nIf a previous claim remains: "
                    + CONTROL_LEASE
                    + ". Normal services retain the original 600 s actual-progress "
                    "lease; the UI does not requeue a task."
                )
            elif state["action"] in {"pause", "stop"}:
                text += "\nThe command is a request; worker exit is not inferred from it."
            self.control_state_label.setText(text)
        for action, button in (
            ("pause", self.pause_button),
            ("run", self.resume_button),
            ("stop", self.stop_button),
        ):
            reason = self._control_reason(action)
            button.setEnabled(not reason)
            button.setToolTip(
                reason or (CONTROL_LEASE if action == "run" else CONTROL_BOUNDARY)
            )

    def pause_job(self):
        self._request_control("pause")

    def stop_job(self):
        self._request_control("stop")

    def resume_job(self):
        self._request_control("run")

    def _request_control(self, action):
        reason = self._control_reason(action)
        if reason:
            self._error(reason)
            return
        root, generation = self.current["root"], self._generation
        displayed = deepcopy(self.current["control"])
        resume = (
            dict(
                root=root,
                generation=generation,
                target=deepcopy(self.target_combo.currentData()),
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
            )
            if action == "run"
            else None
        )
        self._resume_after_control = None
        self._control_pending = True
        self._target_changed()

        def publish():
            asset = self.control_publisher(
                root,
                action,
                machine=self.local_machine,
                expected_previous=displayed["state"].get("command"),
            )
            control = self._read_control_snapshot(root)
            if not control.get("available") or control["state"].get("command") != asset:
                raise ValueError(
                    "Job control changed after publication; refresh before starting a service."
                )
            return dict(root=root, generation=generation, control=control, resume=resume)

        self._control_thread, self._control_worker = self._start_call(
            publish, self._control_ready, self._control_closed
        )

    @Slot(object)
    def _control_ready(self, result):
        if (
            result["generation"] != self._generation
            or not self.current
            or self.current["root"] != result["root"]
        ):
            return
        self.current["control"] = result["control"]
        self._resume_after_control = result["resume"]
        self._target_changed()

    @Slot()
    def _control_closed(self):
        if self._control_thread is not None:
            self._control_thread.wait()
        self._control_thread = self._control_worker = None
        self._control_pending = False
        resume, self._resume_after_control = self._resume_after_control, None
        self._target_changed()
        if (
            resume is not None
            and resume["generation"] == self._generation
            and self.current
            and resume["root"] == self.current["root"]
        ):
            if (
                resume["target"] == self.target_combo.currentData()
                and resume["backend"] == self.backend_combo.currentData()
                and resume["workers"] == self.workers_spin.value()
            ):
                self.start_selected_machine()
            else:
                self._error(
                    "Run was requested, but the selected launch target changed; choose Start explicitly."
                )
        self._refresh_pending = False
        self.refresh()

    def start_selected_machine(self):
        selected = self.target_combo.currentData() or {}
        if self.current and self.current.get("errors"):
            self._error(
                "Worker start unavailable: selected job has metadata errors.\n"
                + "\n".join(self.current["errors"])
            )
            self._target_changed()
            return
        if selected.get("local"):
            return self.start_local_workers()
        if not self.current or not self.current.get("job_facts"):
            return
        if not self._start_allowed(selected.get("machine", self.local_machine)):
            return
        request = None
        try:
            bound = selected["source"]
            actual = JobSnapshotReader().read(Path(bound["path"]), bound)
            if actual.get("job") != descriptor(Path(self.current["root"]) / "job.json"):
                raise ValueError("Remote registration belongs to another job")
            target = self._target(actual)
            build_remote_worker_launch(
                self.current["root"],
                target=target,
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
            )
            request = self._begin_request(actual["machine"], "ssh")
            record = self.remote_launcher(
                self.current["root"],
                target=target,
                backend=self.backend_combo.currentData(),
                workers=self.workers_spin.value(),
                launch_directory=request["launch_directory"],
            )
        except Exception:
            self._launch_failed(request, traceback.format_exc())
            return
        self._launch_records.append(deepcopy(record))
        self.remote_worker_launch_requested.emit(deepcopy(record))
        self.progress_label.setText(
            "SSH transport launch requested. Worker startup and frame progress are not inferred from transport startup."
        )
        self.refresh()

    def open_machine_dialog(self):
        if not self.current:
            return
        dialog = RegisterMachineDialog(self, local_machine=self.local_machine)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            result = register_machine_facts(
                self.current["root"],
                dialog.facts(),
                local_machine=self.local_machine,
                new_version=dialog.new_version.isChecked(),
            )
        except Exception:
            self._error(traceback.format_exc())
            return
        self.machine_registered.emit(result)
        self.control_note.setText(REMOTE_GAP)
        self.refresh()

    @Slot(str)
    def _error(self, text):
        self.error_text.setPlainText(text)
        self.error_reported.emit(text)

    def can_close(self):
        return (
            self._read_thread is None
            and self._create_thread is None
            and self._capacity_thread is None
            and self._control_thread is None
            and not self._control_pending
            and (self.resource_panel is None or self.resource_panel.can_close())
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self.refresh()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        if not self.can_close():
            self.progress_label.setText(
                "A metadata read/create or hardware inspection is still finishing; close after it returns."
            )
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)


def _job_reference(root, path):
    from .job_control import _bindings
    return _bindings(root)[0](path)


def _admit_gui_job_root(root):
    from .pipeline.job_queue import read_json
    if read_json(Path(root) / "job.json").get("format") == "darsia-alias-job-v1":
        from .job_filesystem import inspect_shared_folder
        admission = inspect_shared_folder(root)
        if not admission.allowed:
            raise ValueError(admission.reason + ": " + admission.evidence)
    else:
        require_unc(str(root))
