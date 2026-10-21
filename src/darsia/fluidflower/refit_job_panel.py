"""Mass calibration's refit job: create it, follow it and read its frozen result.

The Phase-map method's Mass calibration page (N141 spec 2.4) fits the N1 gain
and beta to the injected mass of the fit runs as a refit job (pipeline
``n1-calibration-refit``) through the API of ``pipeline.refit_jobs``:

* **Create refit job** writes the refit plans under the series'
  ``calibration/refit/<job>`` folder and queues the job in the chosen jobs
  folder. It only writes job files and starts nothing.
* **Start refit job** is disabled while :data:`START_BLOCKER` names a reason:
  the job queue has no capacity estimate for ``n1-calibration-refit`` yet, so
  a launched worker would be refused at its claim. Once the queue admits the
  pipeline, Start launches one CPU worker on this machine as the Jobs page's
  Start does, after an explicit confirmation.
* The follow view reads ``refit_status`` of any refit job, and once the refit
  has finished shows the frozen calibration with its SHA-256
  (``read_frozen_calibration``).

The inputs are what the series records: each run's role as the Calibration
review run list reads it (fit, held-out, descriptive or excluded), the series'
referenced mass calibration as the baseline, a rules file the user chooses and
a finished phase-one job. Nothing here adopts a calibration, and variant F
stays off until its decisions exist. Job files are read and written in one
background call at a time; its worker is freed in the GUI thread after the
thread has finished (Q185).
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: Refit roles (the ``refit_jobs`` keys) and the Calibration review role each takes.
ROLE_KEYS = (
    ("fit", "fit"),
    ("heldout", "held-out"),
    ("descriptive", "descriptive"),
    ("excluded", "excluded"),
)
TASK_COLUMNS = ("Task", "State", "Step", "Parameters", "Stop required")
FINAL_STATES = {"completed", "failed"}
MAX_RULES_BYTES = 1024 * 1024
REFRESH_MS = 15000
#: Why Start is disabled; empty once the job queue admits the refit pipeline.
START_BLOCKER = "Waits for queue admission of the refit pipeline"
NOTICE = (
    "No refit runs before you press Start refit job here, and Start waits for queue admission "
    "of the refit pipeline: today the job queue has no capacity estimate for refit jobs, so a "
    "started worker would be refused at its claim. Create refit job only writes job files."
)
#: The note once the queue admits the refit pipeline (START_BLOCKER empty).
ADMITTED_NOTICE = "No refit runs before you press Start refit job here. Create refit job only writes job files."
F_WAITS = "Waits for decision F1-F5"
NO_JOB = "No refit job is open. Create one, or open an existing refit job."


def series_refit_roles(series_root) -> tuple[dict[str, list[str]], str]:
    """The refit's run roles from the series' records, and a summary line.

    The roles are those of the Calibration review run list: the referenced mass
    calibration's run lists and ``series.toml [runs] excluded``, in
    ``series.toml`` order. A run without a role is left out of every list and
    named in the summary.
    """
    from darsia.fluidflower.calibration_review_runs import NO_ROLE, review_run_rows

    runs = review_run_rows(series_root)
    roles = {key: [row.run for row in runs.rows if row.role == role] for key, role in ROLE_KEYS}
    summary = " · ".join(f"{len(roles[key])} {role}" for key, role in ROLE_KEYS) + "."
    unassigned = [row.run for row in runs.rows if row.role == NO_ROLE]
    if unassigned:
        summary += f" Left out of the refit, without a role: {', '.join(unassigned)}."
    if not roles["fit"]:
        summary += " " + runs.note
    return roles, summary


def baseline_calibration(series_root) -> dict:
    """The series' referenced mass calibration as a bound descriptor (path, SHA-256, size).

    Raises ValueError when the series references none, OSError when it cannot be read.
    """
    from darsia.fluidflower.pipeline.job_queue import descriptor
    from darsia.fluidflower.series_method import read_series_method

    reference = read_series_method(series_root).mass_calibration
    if not reference:
        raise ValueError("The series references no mass calibration to start the refit from.")
    path = Path(reference)
    if not path.is_absolute():
        path = Path(series_root) / path
    return descriptor(path)


def _small_json(path: Path) -> object:
    """A JSON document of at most MAX_RULES_BYTES (UTF-8, optional BOM)."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_RULES_BYTES + 1)
    if len(raw) > MAX_RULES_BYTES:
        raise ValueError(f"The file exceeds {MAX_RULES_BYTES} bytes: {path}")
    return json.loads(raw.decode("utf-8-sig"))


def _hours(values, separator: str = "-") -> str:
    """Hours as short text, such as '0.5-2.375 h' or, with ', ', '8, 48 h'."""
    return separator.join(f"{float(value):g}" for value in values) + " h"


def read_rules(path, *, roles: Mapping, baseline: Mapping) -> tuple[dict, str]:
    """Read a refit rules file and check it against the run roles and the baseline.

    The rules must be ``darsia-n1-refit-rules-v1``, admitted with these roles by
    the refit's own rule reader, and name the baseline calibration's parameters
    as their baseline. Returns the rules and a summary line; raises ValueError,
    KeyError, TypeError or OSError otherwise.
    """
    from darsia.fluidflower.mass.n1_refit import RULES_FORMAT, RefitRules

    rules = _small_json(Path(path))
    if not isinstance(rules, dict) or rules.get("format") != RULES_FORMAT:
        raise ValueError(f"Choose a {RULES_FORMAT} file.")
    checked = RefitRules.from_plan({"rules": rules, "runs": dict(roles)})
    document = _small_json(Path(baseline["path"]))
    parameters = document.get("parameters") if isinstance(document, dict) else None
    if parameters != rules.get("baseline"):
        raise ValueError("The rules' baseline differs from the parameters of the series' mass calibration.")
    parts = [
        f"Stage 1 {_hours(checked.stage1_window_h)}",
        f"stage 2 from {checked.stage2_start_h:g} h",
        f"post-injection bins split at {_hours(checked.post_bin_edges_h, ', ')}",
        f"anchor {checked.anchor_target_h:g} ± {checked.anchor_tolerance_h:g} h",
    ]
    if checked.target_times_h is not None:
        parts.append(f"{len(checked.target_times_h)} target times")
    return rules, " · ".join(parts) + f" · baseline gain {checked.baseline['gain']:g}, beta {checked.baseline['beta']:g}."


def phase_job_readiness(phase_job, fit_runs: Iterable[str]) -> tuple[list[str], list[str]]:
    """The fit runs with a published phase-one result in ``phase_job``, and the others' problems.

    It reads each run's completion record and checks its manifest's SHA-256, as
    the refit's own lookup does; it reads no statistics file.
    """
    from darsia.fluidflower.pipeline.refit_jobs import completed_attempt

    ready, problems = [], []
    for run in fit_runs:
        try:
            completed_attempt(phase_job, run)
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{run}: {error}")
        else:
            ready.append(run)
    return ready, problems


def default_job_id(now: datetime | None = None) -> str:
    """A new refit job name from the UTC time, such as ``refit-20261021T010203Z``."""
    moment = now or datetime.now(timezone.utc)
    return "refit-" + moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gui_checkout() -> Path:
    """The checkout this GUI runs from; a new job takes its commit (``job_identity``)."""
    import darsia

    return Path(darsia.__file__).resolve().parents[2]


def create_job(series_root, *, roles, rules, baseline, phase_job, jobs_root, job_id, loro=True,
               job_identity=None, repository=None) -> dict:
    """Create the queued refit job and return its prepared record; it starts nothing.

    ``job_identity`` defaults to ``refit_jobs.job_identity`` of ``repository``
    (this GUI's checkout), with the baseline's SHA-256 as the job's calibration.
    """
    from darsia.fluidflower.pipeline import refit_jobs
    from darsia.fluidflower.pipeline.job_queue import descriptor, name

    name(job_id)
    identity = job_identity or refit_jobs.job_identity
    job = identity(repository or gui_checkout(), calibration_sha256=baseline["sha256"])
    return refit_jobs.create_refit_job(
        descriptor(Path(series_root) / "series.toml"), phase_job=str(phase_job), roles=dict(roles),
        rules=dict(rules), baseline_calibration=dict(baseline), job_id=job_id, jobs_root=str(jobs_root),
        job=job, loro=bool(loro))


def read_refit_job(job_root) -> dict:
    """``refit_status`` of a refit job, plus its frozen calibration once published.

    ``frozen`` is (document, descriptor) from ``read_frozen_calibration``;
    ``frozen_error`` says why a published result could not be read.
    """
    from darsia.fluidflower.pipeline.refit_jobs import read_frozen_calibration, refit_status

    status = refit_status(job_root)
    frozen, error = None, ""
    if status.get("frozen_calibration"):
        try:
            frozen = read_frozen_calibration(job_root)
        except (OSError, ValueError) as problem:
            error = str(problem)
    return dict(status=status, frozen=frozen, frozen_error=error)


def parameters_text(parameters) -> str:
    """Fitted parameters as short text, such as 'gain 23.86 · beta 0.6513'; '—' when none."""
    if not isinstance(parameters, Mapping) or not parameters:
        return "—"
    parts = []
    for key in ("gain", "beta"):
        value = parameters.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            parts.append(f"{key} {value:.4g}")
    others = {key: value for key, value in parameters.items() if key not in {"gain", "beta"}}
    parts += [f"{key} {value}" for key, value in sorted(others.items())]
    return " · ".join(parts)


def status_rows(status: Mapping) -> list[tuple[str, str, str, str, str]]:
    """One display row per task of a refit status: task, state, step, parameters, stop."""
    rows = []
    for task, value in (status.get("tasks") or {}).items():
        stop = value.get("stop_required")
        rows.append((
            str(task),
            str(value.get("state") or "—").capitalize(),
            str(value.get("step") or "—"),
            parameters_text(value.get("parameters")),
            "—" if stop is None else ("Yes" if stop else "No"),
        ))
    return rows


def status_summary(status: Mapping) -> str:
    """One line on a refit job: tasks by state and whether the refit has finished."""
    tasks = status.get("tasks") or {}
    states = [value.get("state") for value in tasks.values()]
    counts = [f"{states.count(state)} {state}" for state in ("completed", "running", "waiting", "failed")
              if state in states]
    text = f"{len(tasks)} tasks: " + ", ".join(counts or ["none"]) + "."
    if status.get("stop_required"):
        text += " The refit stopped at its grid check: no frozen calibration."
    elif status.get("complete"):
        text += " Every task has finished."
    return text


def frozen_text(frozen, error: str = "") -> str:
    """The frozen calibration's path, SHA-256, size and parameters, or why it is not shown."""
    if error:
        return f"The published frozen calibration could not be read: {error}"
    if frozen is None:
        return ""
    document, bound = frozen
    runs = document.get("calibration_runs") if isinstance(document, Mapping) else None
    fitted = f" · fitted on {len(runs)} runs" if isinstance(runs, list) else ""
    return (f"Frozen calibration: {bound['path']}\nSHA-256 {bound['sha256']} · {bound['size_bytes']} bytes · "
            f"{parameters_text(document.get('parameters'))}{fitted}.\n"
            "It is not adopted: choosing it for the series stays a separate step.")


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _row(*widgets, below: QWidget | None = None) -> QWidget:
    """Widgets side by side in one form field, the first taking the free width.

    ``below`` goes under them in the same field, so a wrapped summary line
    takes the field's own width and leaves no gap in the form.
    """
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    line = QHBoxLayout()
    for index, widget in enumerate(widgets):
        line.addWidget(widget, 1 if index == 0 else 0)
    column.addLayout(line)
    if below is not None:
        column.addWidget(below)
    return holder


class RefitJobPanel(QGroupBox):
    """Create, start and follow the series' refit job (whole series, no run selector)."""

    def __init__(self, parent=None, *, job_identity=None, machine: str | None = None):
        """An empty panel; :meth:`set_series` gives it a series."""
        super().__init__("Refit job — fit to the injected mass (whole series)", parent)
        self._identity = job_identity
        self.machine = machine or socket.gethostname()
        self.series_root: Path | None = None
        self.roles: dict[str, list[str]] | None = None
        self.baseline: dict | None = None
        self.rules: dict | None = None
        self.phase_job = ""
        self.phase_ready = False
        self.job_root: Path | None = None
        self.snapshot: dict | None = None
        self.last_launch: dict | None = None
        self._thread = self._worker = None
        self._refresh_pending = False
        # A result read for another series than the current one is dropped.
        self._generation = self._call_generation = 0
        layout = QVBoxLayout(self)
        self.notice = _label(NOTICE if START_BLOCKER else ADMITTED_NOTICE)
        self.notice.setStyleSheet("color: #b77900;")
        layout.addWidget(self.notice)
        form = QFormLayout()
        self.roles_label = _label("—")
        form.addRow("Run roles", self.roles_label)
        self.baseline_label = _label("—")
        form.addRow("Baseline", self.baseline_label)
        self.rules_edit = QLineEdit()
        self.rules_edit.setReadOnly(True)
        self.rules_edit.setPlaceholderText("darsia-n1-refit-rules-v1 JSON")
        self.rules_button = QPushButton("Choose…")
        self.rules_button.setToolTip("Choose the refit rules file (darsia-n1-refit-rules-v1).")
        self.rules_button.clicked.connect(self.choose_rules)
        self.rules_label = _label("Choose the refit rules file.")
        form.addRow("Rules", _row(self.rules_edit, self.rules_button, below=self.rules_label))
        self.phase_edit = QLineEdit()
        self.phase_edit.setReadOnly(True)
        self.phase_edit.setPlaceholderText("Finished phase-one job folder")
        self.phase_button = QPushButton("Choose…")
        self.phase_button.setToolTip("Choose the finished phase-one job folder of the fit runs.")
        self.phase_button.clicked.connect(self.choose_phase_job)
        self.phase_label = _label("Choose the finished phase-one job of the fit runs.")
        form.addRow("Phase-one job", _row(self.phase_edit, self.phase_button, below=self.phase_label))
        self.jobs_edit = QLineEdit()
        self.jobs_edit.setPlaceholderText(r"\\server\share\DarSIA\jobs")
        self.jobs_edit.textChanged.connect(self._update_actions)
        self.jobs_button = QPushButton("Choose…")
        self.jobs_button.setToolTip("Choose the folder in which the new refit job folder is made.")
        self.jobs_button.clicked.connect(self.choose_jobs_folder)
        form.addRow("Jobs folder", _row(self.jobs_edit, self.jobs_button))
        self.name_edit = QLineEdit(default_job_id())
        self.name_edit.textChanged.connect(self._update_actions)
        form.addRow("Job name", self.name_edit)
        self.loro_check = QCheckBox("Leave-one-run-out fits")
        self.loro_check.setToolTip("One more task per fit run: the fit without that run, and its predictions.")
        self.loro_check.setChecked(True)
        form.addRow("Folds", self.loro_check)
        # Variant F (overflow-governed free gas) is shown, off and disabled.
        self.f_check = QCheckBox(F_WAITS)
        self.f_check.setChecked(False)
        self.f_check.setEnabled(False)
        self.f_check.setToolTip("Overflow-governed free gas. " + F_WAITS + "; the refit runs with it off.")
        form.addRow("Variant F", self.f_check)
        layout.addLayout(form)
        first, second = QHBoxLayout(), QHBoxLayout()
        self.create_button = QPushButton("Create refit job")
        self.create_button.setToolTip("Writes the plans and queues the job; starts nothing.")
        self.create_button.clicked.connect(self.create)
        self.open_button = QPushButton("Open refit job…")
        self.open_button.setToolTip("Follow a refit job created earlier.")
        self.open_button.clicked.connect(self.choose_job)
        self.start_button = QPushButton("Start refit job")
        self.start_button.clicked.connect(self.start)
        # Why Start is disabled, shown beside it while the queue does not admit refit jobs.
        self.start_reason = QLabel(START_BLOCKER)
        self.start_reason.setStyleSheet("color: #b77900;")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        for row, items in ((first, (self.create_button, self.start_button, self.start_reason)),
                           (second, (self.open_button, self.refresh_button))):
            for widget in items:
                row.addWidget(widget)
            row.addStretch(1)
            layout.addLayout(row)
        self.job_label = _label(NO_JOB)
        layout.addWidget(self.job_label)
        self.tasks = QTableWidget(0, len(TASK_COLUMNS))
        self.tasks.setHorizontalHeaderLabels(TASK_COLUMNS)
        self.tasks.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tasks.verticalHeader().hide()
        header = self.tasks.horizontalHeader()
        for column in range(len(TASK_COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(TASK_COLUMNS.index("Parameters"), QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tasks, 1)
        self.result_label = _label()
        layout.addWidget(self.result_label)
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._update_actions()

    # ------------------------------------------------------------ inputs
    def set_series(self, series_root) -> None:
        """Read the roles and the baseline of this series; None clears the panel.

        Another series clears the chosen rules, phase-one job and open job. The
        same series keeps them and only re-reads what the series records.
        """
        root = None if series_root is None else Path(series_root)
        if root != self.series_root:
            self._timer.stop()
            self._generation += 1
            self._refresh_pending = False
            self.series_root = root
            self.rules = None
            self.rules_edit.clear()
            self.rules_label.setText("Choose the refit rules file.")
            self.phase_job, self.phase_ready = "", False
            self.phase_edit.clear()
            self.phase_label.setText("Choose the finished phase-one job of the fit runs.")
            self.job_root = self.snapshot = self.last_launch = None
            self._show_snapshot(None)
        self.roles = self.baseline = None
        if root is None:
            self.roles_label.setText("—")
            self.baseline_label.setText("—")
            self._update_actions()
            return
        try:
            self.roles, summary = series_refit_roles(root)
        except (OSError, ValueError, TypeError) as error:
            summary = f"The run roles could not be read: {error}"
        self.roles_label.setText(summary)
        if self.roles:
            self.roles_label.setToolTip("\n".join(
                f"{key}: {', '.join(runs) or '—'}" for key, runs in self.roles.items()))
        try:
            self.baseline = baseline_calibration(root)
            self.baseline_label.setText(
                f"{Path(self.baseline['path']).name} · SHA-256 {self.baseline['sha256'][:12]}…")
            self.baseline_label.setToolTip(self.baseline["path"])
        except (OSError, ValueError) as error:
            self.baseline_label.setText(f"No baseline: {error}")
        if self.rules_edit.text():
            self.load_rules(self.rules_edit.text())
        self._update_actions()

    def choose_rules(self) -> None:
        """Choose and check the refit rules file."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose the refit rules", "", "JSON (*.json)")
        if path:
            self.load_rules(path)

    def load_rules(self, path) -> bool:
        """Check a rules file against the roles and baseline; True when it is admitted."""
        self.rules = None
        self.rules_edit.setText(str(path))
        if not self.roles or not self.baseline:
            self.rules_label.setText("The rules are checked once the run roles and the baseline are read.")
            self._update_actions()
            return False
        try:
            self.rules, summary = read_rules(path, roles=self.roles, baseline=self.baseline)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.rules_label.setText(f"These rules cannot be used: {error}")
        else:
            self.rules_label.setText(summary)
        self._update_actions()
        return self.rules is not None

    def choose_phase_job(self) -> None:
        """Choose the finished phase-one job folder and check its fit runs."""
        path = QFileDialog.getExistingDirectory(self, "Choose the finished phase-one job")
        if path:
            self.check_phase_job(path)

    def check_phase_job(self, path) -> bool:
        """Check in the background that every fit run has a published phase-one result."""
        fit = list((self.roles or {}).get("fit") or [])
        if not fit:
            self.phase_label.setText("The series records no fit run.")
            return False
        self.phase_job, self.phase_ready = str(path), False
        self.phase_edit.setText(str(path))
        self.phase_label.setText(f"Checking {len(fit)} fit runs…")
        return self._call(lambda: phase_job_readiness(str(path), fit), self._phase_checked)

    @Slot(object)
    def _phase_checked(self, result) -> None:
        if self._call_generation != self._generation:
            return
        ready, problems = result
        total = len(ready) + len(problems)
        self.phase_ready = not problems and bool(ready)
        text = f"{len(ready)} of {total} fit runs have a published phase-one result."
        if problems:
            text += " Missing: " + "; ".join(problems[:3]) + (" …" if len(problems) > 3 else "")
        self.phase_label.setText(text)

    def choose_jobs_folder(self) -> None:
        """Choose the folder in which the new refit job folder is made."""
        path = QFileDialog.getExistingDirectory(self, "Choose the jobs folder")
        if path:
            self.jobs_edit.setText(path)

    def _create_blocker(self) -> str:
        """Why Create cannot be offered yet, or '' when it can."""
        from darsia.fluidflower.pipeline.job_queue import name

        if self.series_root is None:
            return "No series."
        if not self.roles or not self.roles.get("fit"):
            return "The series records no fit run."
        if self.baseline is None:
            return "The series references no readable mass calibration."
        if self.rules is None:
            return "Choose a rules file that fits the roles and the baseline."
        if not self.phase_ready:
            return "Choose a phase-one job with a published result for every fit run."
        if not self.jobs_edit.text().strip():
            return "Choose the jobs folder."
        try:
            name(self.name_edit.text().strip())
        except ValueError as error:
            return str(error)
        return ""

    # ----------------------------------------------------------- actions
    def create(self) -> bool:
        """Ask, then create the queued refit job in the background; it starts nothing."""
        blocker = self._create_blocker()
        if blocker or self._thread is not None:
            if blocker:
                QMessageBox.information(self, "Create refit job", blocker)
            return False
        job_id = self.name_edit.text().strip()
        jobs_root = self.jobs_edit.text().strip()
        plans = self.series_root.joinpath("calibration", "refit", job_id)
        answer = QMessageBox.question(
            self, "Create refit job",
            f"Create the refit job {job_id}?\n\n"
            f"It writes the refit plans to {plans}\nand queues the job in {Path(jobs_root) / job_id}.\n\n"
            "It only writes job files and starts nothing: no refit runs before you press Start refit job"
            + (f", which {START_BLOCKER[:1].lower()}{START_BLOCKER[1:]}." if START_BLOCKER else "."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        arguments = dict(roles=dict(self.roles), rules=dict(self.rules), baseline=dict(self.baseline),
                         phase_job=self.phase_job, jobs_root=jobs_root, job_id=job_id,
                         loro=self.loro_check.isChecked(), job_identity=self._identity)
        root = self.series_root
        self.job_label.setText(f"Creating {job_id}: reading the fit runs' statistics and writing the plans…")
        return self._call(lambda: dict(created=create_job(root, **arguments),
                                       job_root=str(Path(jobs_root) / job_id)), self._created)

    @Slot(object)
    def _created(self, result) -> None:
        if self._call_generation != self._generation:
            return
        self.job_root = Path(result["job_root"])
        tasks = result["created"].get("tasks") or []
        self.job_label.setText(f"Created {self.job_root.name}: {len(tasks)} tasks queued. Nothing runs before "
                               "Start refit job.")
        self.name_edit.setText(default_job_id())
        self._refresh_pending = True

    def choose_job(self) -> None:
        """Choose an existing refit job folder to follow."""
        path = QFileDialog.getExistingDirectory(self, "Open an existing refit job")
        if path:
            self.open_job(path)

    def open_job(self, path) -> bool:
        """Follow the refit job in ``path``; another job is refused when it is read."""
        self.job_root, self.snapshot, self.last_launch = Path(path), None, None
        self._show_snapshot(None)
        self.job_label.setText(f"Reading {self.job_root.name}…")
        return self.refresh()

    def start(self) -> bool:
        """Ask, then start one CPU worker on this machine for the open refit job.

        Refused, with nothing asked or started, while :data:`START_BLOCKER`
        names a reason.
        """
        from darsia.fluidflower.pipeline.refit_jobs import start_refit_job
        from darsia.fluidflower.store_map import default_store_map_path

        if START_BLOCKER or self.job_root is None or self._thread is not None:
            return False
        stores = Path(default_store_map_path())
        if not stores.is_file():
            QMessageBox.warning(self, "Start refit job",
                                "This machine has no data-store settings yet. Set them on the Jobs page "
                                f"(Machine stores…) before Start.\nExpected: {stores}")
            return False
        answer = QMessageBox.question(
            self, "Start refit job",
            f"Start one CPU worker on this machine ({self.machine}) for the refit job "
            f"{self.job_root.name} now, as the Jobs page's Start does?\n\n"
            "Nothing is adopted: a finished refit shows its frozen calibration here, and choosing it "
            "for the series stays a separate step.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            self.last_launch = start_refit_job(self.job_root, machine=self.machine, stores=stores, workers=1)
        except Exception as error:  # The launcher's own refusal, shown as it is.
            QMessageBox.warning(self, "Start refit job", f"The worker could not be started: {error}")
            return False
        console = self.last_launch.get("console") or "the job's launch folder"
        self.job_label.setText(f"Start requested on {self.machine}; the launcher's console is {console}. "
                               "The follow view below shows what the worker records.")
        self.refresh()
        return True

    def refresh(self) -> bool:
        """Read the open job's status in the background (one read at a time)."""
        if self.job_root is None:
            return False
        if self._thread is not None:
            self._refresh_pending = True
            return False
        root = str(self.job_root)
        return self._call(lambda: read_refit_job(root), self._refreshed)

    @Slot(object)
    def _refreshed(self, result) -> None:
        if self._call_generation != self._generation:
            return
        self.snapshot = result
        self._show_snapshot(result)

    def _show_snapshot(self, snapshot) -> None:
        """Fill the follow view from a read status; None empties it."""
        if snapshot is None:
            self.tasks.setRowCount(0)
            self.result_label.clear()
            if self.job_root is None:
                self.job_label.setText(NO_JOB)
            self._timer.stop()
            self._update_actions()
            return
        status = snapshot["status"]
        rows = status_rows(status)
        self.tasks.setRowCount(len(rows))
        for index, cells in enumerate(rows):
            for column, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                self.tasks.setItem(index, column, item)
        self.job_label.setText(f"Refit job {Path(status['job_root']).name}: {status_summary(status)}")
        self.result_label.setText(frozen_text(snapshot.get("frozen"), snapshot.get("frozen_error", "")))
        if self._finished(status):
            self._timer.stop()
        elif self.isVisible():
            self._timer.start()
        self._update_actions()

    @staticmethod
    def _finished(status) -> bool:
        tasks = (status or {}).get("tasks") or {}
        return bool(tasks) and all(value.get("state") in FINAL_STATES for value in tasks.values())

    def _update_actions(self, *_args) -> None:
        idle = self._thread is None
        series = self.series_root is not None
        status = (self.snapshot or {}).get("status")
        self.create_button.setEnabled(idle and series and not self._create_blocker())
        self.create_button.setToolTip(self._create_blocker() or "Writes the plans and queues the job; starts nothing.")
        for button in (self.rules_button, self.phase_button, self.jobs_button, self.open_button):
            button.setEnabled(idle and series)
        self.start_button.setEnabled(not START_BLOCKER and idle and self.job_root is not None
                                     and not self._finished(status))
        self.start_button.setToolTip(START_BLOCKER or "Asks first, then starts one CPU worker on this machine.")
        self.start_reason.setText(START_BLOCKER)
        self.start_reason.setVisible(bool(START_BLOCKER))
        self.refresh_button.setEnabled(idle and self.job_root is not None)

    # ------------------------------------------------- background calls
    def _call(self, call, ready) -> bool:
        """Run ``call`` in a thread and hand its result to ``ready`` in the GUI thread."""
        from darsia.fluidflower.jobs_panel import _CallWorker

        if self._thread is not None:
            return False
        thread = QThread(self)
        worker = _CallWorker(call)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(ready)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # No deleteLater for the worker (Q185): _closed drops it after wait(),
        # which frees it in the GUI thread.
        thread.finished.connect(self._closed)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self._call_generation = self._generation
        self._update_actions()
        thread.start()
        return True

    @Slot(str)
    def _failed(self, text) -> None:
        last = text.strip().splitlines()[-1] if text.strip() else "unknown error"
        self.job_label.setText(f"That did not work: {last}")
        self.job_label.setToolTip(text)

    @Slot()
    def _closed(self) -> None:
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self._update_actions()
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        status = (self.snapshot or {}).get("status")
        if self.job_root is not None and status is not None and not self._finished(status):
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._timer.stop()
        super().hideEvent(event)
