"""Distributed campaign preflight, creation and host-process controls."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PureWindowsPath
import socket
import threading
import time
import uuid
from typing import Any, Callable, Mapping

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QVBoxLayout, QWidget)

from darsia.fluidflower.tools import studio_distributed_mass_queue as transport
from darsia.fluidflower.tools.studio_distributed_mass_queue import CampaignQueue


def _unc_path(value: str, label: str) -> PureWindowsPath:
    """Require an explicit UNC directory below a server/share root."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a shared UNC path")
    raw = value.strip().replace("/", "\\")
    if (not raw.startswith("\\\\") or raw.startswith(("\\\\?\\", "\\\\.\\"))
            or any(part in {"", ".", ".."} for part in raw[2:].rstrip("\\").split("\\"))):
        raise ValueError(f"{label} must be a shared UNC path without relative components")
    path = PureWindowsPath(raw)
    if len(path.parts) < 2 or not path.drive.startswith("\\\\"):
        raise ValueError(f"{label} must be below a shared UNC server/share")
    return path


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _results_available(root: PureWindowsPath) -> bool:
    return Path(root).is_dir()


def _coordinator_binding(request: Mapping[str, Any], catalog: Mapping[str, Any],
                         selection: Mapping[str, Any], results_root: PureWindowsPath) -> None:
    """Compare the current Studio choice with the immutable queue selection."""
    if request.get("dataset") != catalog.get("dataset"):
        raise ValueError("Campaign belongs to another experiment series")
    if _unc_path(str(catalog.get("results_root", "")), "Selected series results") != results_root:
        raise ValueError("Shared results root differs from the selected series results")
    bindings = request["bindings"]
    if bindings["model_id"] != catalog.get("model_id"):
        raise ValueError("Campaign phase model differs from the selected Studio model")
    model_shas = {job.get("phase_binding", {}).get("model_sha256")
                  for job in catalog.get("jobs", [])}
    if model_shas != {bindings["model_sha256"]}:
        raise ValueError("Selected phase-model content differs from the campaign")
    if bindings["method"] != selection.get("method"):
        raise ValueError("Selected mass method differs from the campaign")
    if "overflow_mode" in request.get("selection", {}):
        if selection.get("overflow_mode", "off") != request["selection"]["overflow_mode"]:
            raise ValueError("Selected overflow comparison mode differs from the campaign")
        selected_overflow = selection.get("overflow") or {}
        frozen_overflow = request.get("native_request", {}).get("overflow") or {}
        if selected_overflow != frozen_overflow:
            raise ValueError("Selected measured overflow files differ from the campaign")
        if bindings.get("overflow_sources_sha256") != transport._sha_bytes(
                transport._json_bytes({
                    run: _sha_file(Path(path)) for run, path in sorted(selected_overflow.items())
                })):
            raise ValueError("Measured overflow file contents changed from the campaign")
    frozen = request["selection"]
    profile = selection.get("physical_profile")
    if profile != frozen.get("physical_profile"):
        raise ValueError("Selected physical/TOML profile differs from the campaign")
    runs = (selection.get("cohort") or {}).get("runs")
    if runs != frozen.get("runs"):
        raise ValueError("Selected experiment cohort differs from the campaign")
    if profile == "registered":
        profile_path = Path(catalog["results_root"]).parent.parent / "concentration.toml"
        receipt_sha = None
    elif profile == "documented_ffum":
        from darsia.fluidflower.tools.phase_studio_calibration_launch import documented_ffum_profile
        profile_path, _ = documented_ffum_profile(catalog)
        receipt_sha = _sha_file(profile_path.with_suffix(".receipt.json"))
    else:
        raise ValueError("Unknown physical/TOML profile")
    if _sha_file(profile_path) != bindings["concentration_toml_sha256"]:
        raise ValueError("Selected concentration TOML changed from the campaign")
    if receipt_sha != bindings.get("physical_profile_receipt_sha256"):
        raise ValueError("Selected physical-profile receipt differs from the campaign")
    selection_path = selection.get("image_selection_receipt_path")
    selection_sha = _sha_file(Path(selection_path)) if selection_path else None
    if selection_sha != bindings.get("image_selection_receipt_sha256"):
        raise ValueError("Selected image-selection receipt differs from the campaign")
    if bindings["method"] == "native_frozen":
        path = selection.get("calibration_file")
        if not path or _sha_file(Path(path)) != bindings.get("calibration_sha256"):
            raise ValueError("Selected saved native calibration differs from the campaign")
        if not selection_path:
            indices = selection.get("frame_indices")
            for task in request.get("tasks", []):
                expected = indices if indices is not None else list(range(task.get("available_frame_count", 0)))
                if (indices is not None or "available_frame_count" in task) and task["frame_indices"] != expected:
                    raise ValueError("Selected frame indices differ from the frozen campaign")


@dataclass(frozen=True)
class AttachedCampaign:
    queue: CampaignQueue
    results_root: PureWindowsPath
    role: str
    host: str

    def snapshot(self) -> dict[str, Any]:
        self.queue._assert_campaign()
        status = self.queue.status()
        controls = self.queue.root / "control"
        limits = {path.stem: self.queue.worker_limit(path.stem)
                  for path in controls.glob("*.json")}
        from darsia.fluidflower.tools.studio_distributed_mass_queue import _read_json, _exists
        services = {path.stem: _read_json(path) for path in (self.queue.root / "services").glob("*.json")}
        tasks = []
        for task in self.queue.request.get("tasks", []):
            if "task_id" not in task:
                continue
            task_id = task["task_id"]
            state = self.queue._generation(task_id)
            generation = state["generation"]
            row = {"run": task.get("run", ""), "stage": task["stage"], "host": "",
                   "state": "failed" if state["status"] == "failed" else "pending",
                   "generation": generation, "heartbeat_age": None, "frames_per_second": None,
                   "detail": state["history"][-1]["reason"] if state["history"] else ""}
            accepted = self.queue.root / "accepted" / (task_id + ".json")
            claim_dir = self.queue.root / "claims" / task_id / f"g{generation:06d}"
            if _exists(accepted):
                receipt = _read_json(accepted)
                metadata = receipt.get("metadata", {})
                seconds = metadata.get("seconds", 0)
                row.update(state="accepted", host=receipt["host"],
                           frames_per_second=len(task["frame_indices"]) / seconds if seconds > 0 else None,
                           detail=f"generation {generation} · " + metadata.get("environment", {}).get("backend", ""))
            elif _exists(claim_dir / "claim.json"):
                claim = _read_json(claim_dir / "claim.json")
                row.update(state="in_progress", host=claim["host"], detail=f"generation {generation}")
                if _exists(claim_dir / "lease.json"):
                    row["heartbeat_age"] = max(0, time.time() - _read_json(claim_dir / "lease.json")["seen_at"])
                attempt = self.queue.root / "attempts" / task_id / f"g{generation:06d}-{claim['token']}"
                if _exists(attempt / "status.json"):
                    progress = _read_json(attempt / "status.json")
                    row["detail"] = progress.get("error") or progress.get("stage", row["detail"])
                failure = self.queue.root / "failures" / task_id / f"g{generation:06d}.json"
                if _exists(failure):
                    row.update(state="retry_pending", detail=_read_json(failure)["reason"])
            tasks.append(row)
        return {"status": status, "host_limits": limits, "services": services,
                "tasks": tasks, "published": _exists(self.queue.root / "published.json")}


@dataclass
class _BackgroundOperation:
    generation: int
    deadline: float
    callback: Callable[[Any, str], None]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str = ""
    completed_at: float | None = None
    timed_out: bool = False


def attach_existing_campaign(queue_path: str, results_path: str, *, role: str,
                             catalog: Mapping[str, Any] | None = None,
                             selection: Mapping[str, Any] | None = None,
                             opener: Callable[[str], CampaignQueue] | None = None,
                             host: str | None = None) -> AttachedCampaign:
    """Open an already frozen queue; no campaign state is changed."""
    if role not in {"coordinator", "worker"}:
        raise ValueError("Choose coordinator or worker-only role")
    queue_root = _unc_path(queue_path, "Queue root")
    results_root = _unc_path(results_path, "Results root")
    if queue_root == results_root or not queue_root.is_relative_to(results_root):
        raise ValueError("Queue root must be inside the shared results root")
    queue = (opener or CampaignQueue.open)(str(queue_root))
    if queue.request.get("dataset") not in {"ffac", "ffum", "ffbm"}:
        raise ValueError("Campaign has no frozen experiment series")
    frozen_selection = queue.request.get("selection")
    if (not isinstance(frozen_selection, Mapping)
            or frozen_selection.get("physical_profile") not in {"registered", "documented_ffum"}
            or not isinstance(frozen_selection.get("runs"), list)
            or not frozen_selection["runs"]
            or any(not isinstance(run, str) or not run for run in frozen_selection["runs"])
            or len(frozen_selection["runs"]) != len(set(frozen_selection["runs"]))):
        raise ValueError("Campaign has no frozen cohort and physical/TOML selection")
    frozen_paths = queue.request.get("shared_paths") or {}
    if not isinstance(frozen_paths, Mapping):
        raise ValueError("Campaign has no frozen shared UNC paths")
    if (_unc_path(frozen_paths.get("queue_root"), "Frozen queue root") != queue_root
            or _unc_path(frozen_paths.get("results_root"), "Frozen results root") != results_root):
        raise ValueError("Shared UNC paths differ from the immutable campaign request")
    if not _results_available(results_root):
        raise ValueError("Shared results root is unavailable on this host")
    if role == "coordinator":
        if catalog is None or selection is None:
            raise ValueError("Load a phase model and select a mass method/cohort first")
        _coordinator_binding(queue.request, catalog, selection, results_root)
    return AttachedCampaign(queue, results_root, role, host or socket.gethostname())


class DistributedMassQueuePanel(QWidget):
    """The supplied service owns numerical subprocesses and coordinator gates."""

    ATTACH_TIMEOUT_SECONDS = 8.0
    STATUS_TIMEOUT_SECONDS = 5.0
    LIMIT_TIMEOUT_SECONDS = 8.0
    MAX_PENDING_IO = 3

    def __init__(self, parent=None, *, catalog_provider=None, selection_provider=None,
                 worker_service=None, native_request_provider=None):
        super().__init__(parent)
        self.catalog_provider = catalog_provider or (lambda: None)
        self.selection_provider = selection_provider or (lambda: None)
        self.worker_service = worker_service
        self.native_request_provider = native_request_provider
        self._draft = None
        self._draft_options = None
        self.attached: AttachedCampaign | None = None
        self._status_valid = False
        self._generation = 0
        self._status_sequence = 0
        self._status_call: _BackgroundOperation | None = None
        self._applying = False
        self._pending_io: list[_BackgroundOperation] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        note = QLabel("Apply a saved native calibration to the checked experiments in Native concentration "
                      "settings. Measured overflow may be compared without refitting; select targets "
                      "there before preflight. Each worker owns one run in chronological order. Shared fitting, "
                      "photometry fitting and GPU execution are unavailable here. Production requires "
                      "passing model eligibility and measured host parity; QA-fail models are never promoted.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.role = QComboBox()
        self.role.addItem("Coordinator (optional local workers)", "coordinator")
        self.role.addItem("Worker only", "worker")
        form.addRow("This computer", self.role)
        self.purpose = QComboBox()
        self.purpose.addItem("Diagnostic · Studio validation only", "diagnostic")
        self.purpose.addItem("Production · requires model acceptance and measured host parity", "production")
        form.addRow("Campaign purpose", self.purpose)
        self.model_eligibility = QLineEdit()
        self.model_eligibility.setPlaceholderText("Production only: passing model eligibility receipt on shared UNC storage")
        self.parity_receipt = QLineEdit()
        self.parity_receipt.setPlaceholderText("Production only: measured two-host parity receipt from certify-parity")
        for label, field in (("Model eligibility receipt", self.model_eligibility),
                             ("Measured host-parity receipt", self.parity_receipt)):
            row = QHBoxLayout()
            row.addWidget(field, 1)
            browse = QPushButton("Browse…")
            def choose(_checked=False, target=field, title=label):
                path, _ = QFileDialog.getOpenFileName(self, title, "", "JSON (*.json)")
                if path:
                    target.setText(path)
            browse.clicked.connect(choose)
            row.addWidget(browse)
            form.addRow(label, row)
        self.scratch_path = QLineEdit(str(Path.home() / ".cache" / "darsia" / "distributed-mass"))
        form.addRow("Local scratch / service logs", self.scratch_path)
        self.frame_indices = QLineEdit()
        self.frame_indices.setPlaceholderText("All registered frames; or comma-separated zero-based indices")
        form.addRow("Frames in each selected run", self.frame_indices)
        self.selection_receipt = QLineEdit()
        self.selection_receipt.setPlaceholderText("Optional ready image-selection receipt; overrides manual indices")
        receipt_row = QHBoxLayout()
        receipt_row.addWidget(self.selection_receipt, 1)
        choose_receipt = QPushButton("Browse…")
        choose_receipt.clicked.connect(self._choose_selection_receipt)
        receipt_row.addWidget(choose_receipt)
        form.addRow("Image-selection receipt", receipt_row)
        self.queue_path = QLineEdit()
        self.queue_path.setPlaceholderText(r"\\server\share\series\DarSIA\results\concentration_mass\distributed_mass\campaign-id")
        self.results_path = QLineEdit()
        self.results_path.setPlaceholderText(r"\\server\share\series\DarSIA\results\concentration_mass")
        for label, field, choose in (("Shared queue (UNC)", self.queue_path, self._choose_queue),
                                     ("Shared results (UNC)", self.results_path, self._choose_results)):
            row = QHBoxLayout()
            row.addWidget(field, 1)
            browse = QPushButton("Browse…")
            browse.clicked.connect(choose)
            row.addWidget(browse)
            form.addRow(label, row)
        actions = QHBoxLayout()
        self.preflight_button = QPushButton("Preflight selected native fit")
        self.preflight_button.clicked.connect(self.preflight)
        self.create_button = QPushButton("Create reviewed campaign")
        self.create_button.clicked.connect(self.create)
        self.start_button = QPushButton("Start / resume coordinator")
        self.start_button.clicked.connect(self.start_coordinator)
        self.attach_button = QPushButton("Attach frozen campaign")
        self.attach_button.clicked.connect(self.attach)
        self.refresh_button = QPushButton("Refresh status")
        self.refresh_button.clicked.connect(self.refresh_status)
        self.refresh_button.setEnabled(False)
        actions.addWidget(self.preflight_button)
        actions.addWidget(self.create_button)
        actions.addWidget(self.start_button)
        actions.addWidget(self.attach_button)
        actions.addWidget(self.refresh_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        parity_row = QHBoxLayout()
        self.peer_queue = QLineEdit()
        self.peer_queue.setPlaceholderText("Second completed pilot campaign (same selection, run on the other host)")
        parity_row.addWidget(self.peer_queue, 1)
        choose_peer = QPushButton("Browse pilot…")
        def peer_path():
            path = QFileDialog.getExistingDirectory(self, "Second completed pilot", self.results_path.text())
            if path:
                self.peer_queue.setText(path)
        choose_peer.clicked.connect(peer_path)
        parity_row.addWidget(choose_peer)
        self.certify_button = QPushButton("Measure two-host parity")
        self.certify_button.clicked.connect(self.certify_parity)
        parity_row.addWidget(self.certify_button)
        layout.addLayout(parity_row)
        self.identity = QLabel("No campaign attached.")
        self.identity.setWordWrap(True)
        layout.addWidget(self.identity)
        self.counts = QLabel("")
        self.counts.setWordWrap(True)
        layout.addWidget(self.counts)
        self.hosts = QLabel("")
        self.hosts.setWordWrap(True)
        layout.addWidget(self.hosts)
        self.task_table = QTableWidget(0, 7)
        self.task_table.setHorizontalHeaderLabels(["Run", "Stage", "State", "Host", "Lease age", "Frames/s", "Progress / verification"])
        self.task_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_table.setMaximumHeight(170)
        layout.addWidget(self.task_table)
        worker_row = QHBoxLayout()
        self.local_limit = QSpinBox()
        self.local_limit.setRange(0, 32)
        self.local_limit.setValue(0)
        self.apply_limit = QPushButton("Set local worker limit")
        self.apply_limit.clicked.connect(self._apply_local_limit)
        self.stop_local = QPushButton("Stop local workers after current task")
        self.stop_local.clicked.connect(self._stop_local)
        worker_row.addWidget(QLabel("This host's worker limit"))
        worker_row.addWidget(self.local_limit)
        worker_row.addWidget(self.apply_limit)
        worker_row.addWidget(self.stop_local)
        worker_row.addStretch(1)
        layout.addLayout(worker_row)
        self.message = QLabel("Attach an existing campaign to inspect its frozen selection and host limits.")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self.refresh_status)
        self._completion_timer = QTimer(self)
        self._completion_timer.setInterval(50)
        self._completion_timer.timeout.connect(self._collect_background)
        self.role.currentIndexChanged.connect(self._detach)
        self.queue_path.textChanged.connect(self._detach)
        self.results_path.textChanged.connect(self._detach)
        self.frame_indices.textChanged.connect(self._clear_draft)
        self.selection_receipt.textChanged.connect(self._clear_draft)
        self.purpose.currentIndexChanged.connect(self._clear_draft)
        self.model_eligibility.textChanged.connect(self._clear_draft)
        self.parity_receipt.textChanged.connect(self._clear_draft)
        self._update_controls()

    def set_catalog(self, catalog: Mapping[str, Any] | None) -> None:
        """Suggest the same results root used by the local calibration controls."""
        self._detach()
        if catalog and catalog.get("results_root"):
            self.results_path.setText(str(catalog["results_root"]))
            self.queue_path.setText(str(Path(catalog["results_root"]) / "distributed_mass" /
                                        ("native-" + uuid.uuid4().hex[:12])))

    def _choose_selection_receipt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ready image-selection receipt", "", "JSON (*.json)")
        if path:
            self.selection_receipt.setText(path)

    def _clear_draft(self, *_):
        self._draft = None
        self._draft_options = None
        self._update_controls()

    def _native_options(self):
        if self.native_request_provider is None:
            raise ValueError("No native calculation selection is connected")
        request = dict(self.native_request_provider())
        if self.purpose.currentData() == "production":
            request["model_eligibility_receipt_path"] = self.model_eligibility.text().strip()
            request["backend_parity_receipt_path"] = self.parity_receipt.text().strip()
        if self.frame_indices.text().strip():
            request["frame_indices"] = [int(value.strip()) for value in self.frame_indices.text().split(",")]
        if self.selection_receipt.text().strip():
            if "frame_indices" in request:
                raise ValueError("Choose an image-selection receipt or manual indices, not both")
            path = Path(self.selection_receipt.text().strip())
            request["image_selection_receipt_path"] = str(path)
            # Hashing occurs in the background preflight, never on Qt's thread.
        return request

    def _current_selection(self):
        value = dict(self.selection_provider())
        if self.frame_indices.text().strip():
            value["frame_indices"] = [int(index.strip()) for index in self.frame_indices.text().split(",")]
        if self.selection_receipt.text().strip():
            value["image_selection_receipt_path"] = self.selection_receipt.text().strip()
        return value

    def preflight(self):
        try:
            from .studio_distributed_mass_service import build_request
            options = self._native_options()
            purpose = self.purpose.currentData()
            root = str(_unc_path(self.queue_path.text(), "Queue root"))
            results = _unc_path(self.results_path.text(), "Results root")
            catalog = self.catalog_provider()
            if _unc_path(catalog["results_root"], "Selected series results") != results:
                raise ValueError("Results root differs from selected series")
            self._clear_draft()
            self.message.setText("Preflight is hashing the selected photographs, phase maps, saved fit and physical inputs…")
            def work():
                request = dict(options)
                if request.get("image_selection_receipt_path"):
                    request["image_selection_receipt_sha256"] = _sha_file(Path(request["image_selection_receipt_path"]))
                return build_request(request, root, purpose=purpose)
            def ready(request, error):
                if error:
                    self.message.setText("Preflight failed: " + error)
                    return
                self._draft, self._draft_options = request, options
                runs = " · ".join(f"{task['run']}: {len(task['frame_indices'])} frames "
                                   f"({task['frame_times_h'][0]:g}–{task['frame_times_h'][-1]:g} h)"
                                   for task in request["tasks"])
                self.identity.setText(f"Preflight ready · {request['purpose']} · {request['dataset'].upper()} · model "
                    f"{request['bindings']['model_id']} · fit SHA {request['bindings']['calibration_sha256'][:12]}… · "
                    f"TOML SHA {request['bindings']['concentration_toml_sha256'][:12]}… · "
                    f"overflow {request['selection'].get('overflow_mode', 'off')} "
                    f"({len(request['selection'].get('overflow_runs', []))} runs)\n{runs}")
                self.message.setText("Review this exact coverage, then Create reviewed campaign. Creation does not start workers.")
                self._update_controls()
            self._start_background(work, ready, 3600)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self.message.setText("Preflight unavailable: " + str(error))

    def create(self):
        try:
            if self._draft is None or self._native_options() != self._draft_options:
                self._clear_draft()
                raise ValueError("Selection changed; repeat preflight before creating")
            from .studio_distributed_mass_service import create_campaign
            request = self._draft
            results = _unc_path(self.results_path.text(), "Results root")
            self._applying = True
            self._update_controls()
            self.message.setText("Creating immutable campaign and verifying preflight inputs again…")
            def created(queue, error):
                self._applying = False
                if error:
                    self.message.setText("Campaign creation failed: " + error)
                    self._update_controls()
                    return
                self._clear_draft()
                self._attached_result(AttachedCampaign(queue, results, "coordinator", socket.gethostname()), "")
            self._start_background(lambda: create_campaign(request), created, 3600)
        except (OSError, ValueError, RuntimeError) as error:
            self.message.setText("Campaign creation unavailable: " + str(error))

    def start_coordinator(self):
        if self.attached is None or self.attached.role != "coordinator" or self.worker_service is None:
            return
        try:
            attached, service = self.attached, self.worker_service
            selection, catalog = self._current_selection(), self.catalog_provider()
            scratch, count = self.scratch_path.text(), self.local_limit.value()
            self._applying = True
            self._update_controls()
            self.message.setText("Verifying this host and launching the coordinator process…")
            def launch():
                _coordinator_binding(attached.queue.request, catalog, selection, attached.results_root)
                return service.launch(attached.queue, attached.host, count, coordinator=True, scratch=scratch)
            self._start_background(launch, self._limit_result, 3600)
        except (OSError, ValueError, RuntimeError) as error:
            self._applying = False
            self._update_controls()
            self.message.setText("Coordinator launch unavailable: " + str(error))

    def certify_parity(self):
        if self.attached is None or self.attached.role != "coordinator":
            return
        try:
            from .studio_distributed_mass_parity import create_receipt
            reference = self.attached.queue.root
            peer = str(_unc_path(self.peer_queue.text(), "Second pilot queue"))
            output = Path(self.attached.results_root) / "distributed_mass" / ("parity-" + uuid.uuid4().hex + ".json")
            self._applying = True
            self._update_controls()
            self.message.setText("Comparing every accepted pilot scalar and numerical field on both hosts…")
            def measured(_receipt, error):
                self._applying = False
                if error:
                    self.message.setText("Host parity failed: " + error)
                else:
                    self.parity_receipt.setText(str(output))
                    self.message.setText("Measured host-parity receipt saved: " + str(output) +
                                         ". Production still requires passing model eligibility.")
                self._update_controls()
            self._start_background(lambda: create_receipt(reference, peer, output), measured, 3600)
        except (OSError, ValueError, RuntimeError) as error:
            self.message.setText("Host parity unavailable: " + str(error))

    def set_worker_service(self, service) -> None:
        self.worker_service = service
        self._update_controls()

    def _choose_queue(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Existing frozen campaign queue",
                                                 self.queue_path.text() or self.results_path.text())
        if value:
            self.queue_path.setText(value)

    def _choose_results(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Shared series results",
                                                 self.results_path.text())
        if value:
            self.results_path.setText(value)

    def _detach(self, *_):
        self.timer.stop()
        self._generation += 1
        self._status_sequence += 1
        self._status_call = None
        self._applying = False
        self.attached = None
        self._status_valid = False
        self._draft = None
        self._draft_options = None
        self.identity.setText("No campaign attached.")
        self.counts.clear()
        self.hosts.clear()
        self.task_table.setRowCount(0)
        self.refresh_button.setEnabled(False)
        self.message.setText("Attach an existing campaign to inspect its frozen selection and host limits.")
        self._update_controls()

    def attach(self) -> None:
        self._detach()
        try:
            role = self.role.currentData()
            queue_path = self.queue_path.text()
            results_path = self.results_path.text()
            catalog = self.catalog_provider() if role == "coordinator" else None
            selection = self._current_selection() if role == "coordinator" else None
            self.message.setText("Checking frozen campaign and shared NAS paths…")
            self._start_background(
                lambda: attach_existing_campaign(
                    queue_path, results_path, role=role,
                    catalog=catalog, selection=selection),
                self._attached_result, self.ATTACH_TIMEOUT_SECONDS)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self.message.setText("Campaign unavailable: " + str(error))

    def _attached_result(self, attached: AttachedCampaign | None, error: str) -> None:
        if error:
            self.message.setText("Campaign unavailable: " + error)
            return
        self.attached = attached
        request = attached.queue.request
        binding = request["bindings"]
        selection = request["selection"]
        self.identity.setText(
            f"{request['dataset'].upper()} · {attached.role} · {request['purpose']} campaign "
            f"{request['campaign_id']} · model {binding['model_id']} · "
            f"method {binding['method']} · profile {selection['physical_profile']} · "
            f"runs {', '.join(selection['runs'])} · "
            f"{sum(len(task.get('frame_indices', [])) for task in request.get('tasks', []))} selected frames · "
            f"overflow {selection.get('overflow_mode', 'off')} "
            f"({len(selection.get('overflow_runs', []))} runs) · "
            f"TOML SHA {binding['concentration_toml_sha256'][:12]}… · "
            f"request SHA {attached.queue.request_sha256[:12]}…")
        self.refresh_button.setEnabled(True)
        self.refresh_status()
        self.timer.start()

    def refresh_status(self) -> None:
        if self.attached is None or self._status_call is not None:
            return
        try:
            attached = self.attached
            self._status_sequence += 1
            sequence = self._status_sequence
            self._status_call = self._start_background(
                attached.snapshot,
                lambda snapshot, error: self._status_result(sequence, snapshot, error),
                self.STATUS_TIMEOUT_SECONDS)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            self._show_status_error(str(error))

    def _status_result(self, sequence: int, snapshot: Mapping[str, Any] | None,
                       error: str) -> None:
        if sequence != self._status_sequence or self.attached is None:
            return
        self._status_call = None
        if error:
            self._show_status_error(error)
            return
        try:
            counts = snapshot["status"]["counts"]
            self.counts.setText("Queue: " + " · ".join(
                f"{name} {counts[name]}" for name in ("pending", "in_progress", "accepted", "failed")))
            limits = snapshot["host_limits"]
            self.hosts.setText("Configured host limits (not active process counts): " + (
                " · ".join(f"{host} {limit}" for host, limit in sorted(limits.items()))
                if limits else "none configured"))
            self.local_limit.setValue(limits.get(self.attached.host, 0))
            services = snapshot.get("services", {})
            if services:
                self.hosts.setText(self.hosts.text() + "\nServices: " + " · ".join(
                    f"{host}: {value.get('state')} / {len(value.get('active_slots', []))} live slots"
                    + (" (heartbeat stale)" if time.time() - value.get("seen_at", 0) > 15 else "")
                    for host, value in sorted(services.items())))
            tasks = snapshot.get("tasks", [])
            self.task_table.setRowCount(len(tasks))
            for index, task in enumerate(tasks):
                age, rate = task.get("heartbeat_age"), task.get("frames_per_second")
                values = [task["run"], task["stage"], task["state"], task["host"],
                          f"{age:.0f} s" if age is not None else "", f"{rate:.4g}" if rate is not None else "",
                          task.get("detail", "")]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(str(value))
                    self.task_table.setItem(index, column, item)
            self._status_valid = True
            if self.worker_service is None:
                self.message.setText("Read-only transport status. No audited local mass-worker service is attached.")
            elif (self.attached.queue.request["purpose"] == "production"
                  and not self.attached.queue.request.get("production_gate")):
                self.message.setText("Production workers require a measured host-parity gate.")
            else:
                self.message.setText("Verified results published to Studio. Refresh saved results to open them."
                    if snapshot.get("published") else "Start/resume the coordinator once. Each worker host joins with its local limit; zero drains this host after current work.")
            self._update_controls()
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self._show_status_error(str(exc))

    def _show_status_error(self, error: str) -> None:
        self._status_call = None
        self._status_valid = False
        self._update_controls()
        self.message.setText("Shared queue status unavailable: " + error)

    def _start_background(self, work: Callable[[], Any],
                          callback: Callable[[Any, str], None],
                          timeout_seconds: float) -> _BackgroundOperation:
        if sum(not call.done.is_set() for call in self._pending_io) >= self.MAX_PENDING_IO:
            raise RuntimeError("Too many stalled shared-storage requests")
        call = _BackgroundOperation(
            self._generation, time.monotonic() + timeout_seconds, callback)
        self._pending_io.append(call)

        def run() -> None:
            try:
                call.result = work()
            except Exception as exc:
                call.error = str(exc) or type(exc).__name__
            finally:
                call.completed_at = time.monotonic()
                call.done.set()

        threading.Thread(target=run, name="studio-mass-queue-io", daemon=True).start()
        self._completion_timer.start()
        return call

    def _collect_background(self) -> None:
        now = time.monotonic()
        for call in list(self._pending_io):
            if call.done.is_set():
                self._pending_io.remove(call)
                if not call.timed_out and call.generation == self._generation:
                    error = ("timed out waiting for shared NAS" if call.completed_at > call.deadline
                             else call.error)
                    call.callback(call.result if not error else None, error)
            elif not call.timed_out and now >= call.deadline:
                call.timed_out = True
                if call.generation == self._generation:
                    call.callback(None, "timed out waiting for shared NAS")
        if not self._pending_io:
            self._completion_timer.stop()

    def _update_controls(self) -> None:
        enabled = (self.attached is not None and self._status_valid and not self._applying
                   and self.worker_service is not None
                   and (self.attached.queue.request.get("purpose") == "diagnostic"
                        or bool(self.attached.queue.request.get("production_gate"))))
        self.local_limit.setEnabled(enabled)
        self.apply_limit.setEnabled(enabled)
        self.stop_local.setEnabled(enabled)
        coordinator = self.role.currentData() == "coordinator"
        capable = self.worker_service is not None and self.native_request_provider is not None
        self.preflight_button.setEnabled(coordinator and capable and not self._applying)
        self.create_button.setEnabled(coordinator and capable and self._draft is not None and not self._applying)
        self.start_button.setEnabled(enabled and coordinator and hasattr(self.worker_service, "launch"))
        self.certify_button.setEnabled(enabled and coordinator and hasattr(self.worker_service, "launch"))
        self.frame_indices.setEnabled(coordinator)
        self.selection_receipt.setEnabled(coordinator)
        self.purpose.setEnabled(coordinator)
        self.model_eligibility.setEnabled(coordinator and self.purpose.currentData() == "production")
        self.parity_receipt.setEnabled(coordinator and self.purpose.currentData() == "production")

    def _apply_local_limit(self) -> None:
        if not self._update_allowed():
            return
        try:
            attached = self.attached
            service = self.worker_service
            requested = self.local_limit.value()
            scratch = self.scratch_path.text()
            self._status_sequence += 1  # ignore a status read from before this write
            self._status_call = None
            self._applying = True
            self._update_controls()
            self.message.setText("Updating this host's worker limit on shared storage…")

            def apply() -> int:
                attached.queue._assert_campaign()
                if hasattr(service, "launch") and requested:
                    service.launch(attached.queue, attached.host, requested,
                                   coordinator=attached.role == "coordinator", scratch=scratch)
                else:
                    service.apply_limit(attached.queue, attached.host, requested)
                actual = attached.queue.worker_limit(attached.host)
                if actual != requested:
                    raise RuntimeError("Local worker service did not apply the requested host limit")
                return actual

            self._start_background(apply, self._limit_result,
                                   3600 if hasattr(service, "launch") and requested else self.LIMIT_TIMEOUT_SECONDS)
        except (OSError, ValueError, RuntimeError) as error:
            self._applying = False
            self._update_controls()
            self.message.setText("Local worker limit update unavailable: " + str(error))

    def _limit_result(self, _result: Any, error: str) -> None:
        self._applying = False
        if error:
            self._status_valid = False
            self._update_controls()
            self.message.setText("Local worker limit update unverified: " + error)
        else:
            self.refresh_status()

    def _stop_local(self) -> None:
        if self._update_allowed():
            self.local_limit.setValue(0)
            self._apply_local_limit()

    def _update_allowed(self) -> bool:
        return bool(self.attached is not None and self._status_valid and not self._applying
                    and self.worker_service is not None
                    and (self.attached.queue.request.get("purpose") == "diagnostic"
                         or bool(self.attached.queue.request.get("production_gate"))))

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._completion_timer.stop()
        self._generation += 1
        super().closeEvent(event)
