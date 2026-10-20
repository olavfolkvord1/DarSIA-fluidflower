"""Build published dataset: the GUI twin of ``darsia dataset build``.

The dialog collects the same choices as the command line, checks first and
builds only on request. Both run in a worker thread; the report is shown in
the dialog. Nothing is written into the series or the job folder.
"""

from __future__ import annotations

from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from .layout import TIER_ORDER

GIB = 1024**3


class BuildThread(QThread):
    """Run one check or build off the GUI thread; emit the report or a traceback."""

    finished_report = Signal(object, str)

    def __init__(self, request, check_only: bool, parent=None):
        """Keep the build request; ``check_only`` checks without writing."""
        super().__init__(parent)
        self.request, self.check_only = request, check_only

    def run(self):
        """Check (or build) and emit ``(report, "")`` or ``(None, traceback)``."""
        from .builder import build_record, check_build

        try:
            report = check_build(self.request)[1] if self.check_only else build_record(self.request)
        except Exception:  # noqa: BLE001 - the dialog shows the traceback verbatim
            self.finished_report.emit(None, traceback.format_exc())
            return
        self.finished_report.emit(report, "")


def _path_row(edit: QLineEdit, button: QPushButton) -> QWidget:
    """One form row: a path field and its Browse button."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row


class BuildDatasetDialog(QDialog):
    """Choose the production job, record, tier and an empty output folder; check, then build."""

    def __init__(self, series_root, parent=None):
        """Lay out the choices for the series in ``series_root``; nothing runs yet."""
        super().__init__(parent)
        self.setWindowTitle("Build published dataset")
        self.series_root = Path(series_root)
        self.thread = None
        layout = QVBoxLayout(self)
        note = QLabel("Packages this series and the finished production job that made its results as "
                      "one record: R (raw images as acquired, configuration and expected results) or "
                      "S (simulator products). Check lists everything that would refuse the record; "
                      "Build writes only into a new or empty folder.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        form.addRow("Series folder", QLabel(str(self.series_root)))
        self.job_edit = QLineEdit()
        self.job_edit.setPlaceholderText("The finished production job's folder")
        job_button = QPushButton("Browse…")
        job_button.clicked.connect(lambda: self._choose_folder(self.job_edit, "Production job folder"))
        form.addRow("Production job", _path_row(self.job_edit, job_button))
        self.record_combo = QComboBox()
        self.record_combo.addItem("R: reproduction (raw images as acquired)", "R")
        self.record_combo.addItem("S: simulator products", "S")
        self.record_combo.currentIndexChanged.connect(self._record_changed)
        form.addRow("Record", self.record_combo)
        self.tier_combo = QComboBox()
        for tier in TIER_ORDER:
            self.tier_combo.addItem(tier, tier)
        form.addRow("Tier", self.tier_combo)
        self.resolution_edit = QLineEdit()
        self.resolution_edit.setPlaceholderText("Required for S; no default")
        form.addRow("Simulator resolution (mm)", self.resolution_edit)
        self.geometry_edit = QLineEdit()
        self.geometry_edit.setPlaceholderText("Optional for S: the completed porous domain")
        self.geometry_button = QPushButton("Browse…")
        self.geometry_button.clicked.connect(
            lambda: self._choose_file(self.geometry_edit, "Simulator geometry request"))
        form.addRow("Simulator geometry request", _path_row(self.geometry_edit, self.geometry_button))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("A new or empty folder outside the series")
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(lambda: self._choose_folder(self.output_edit, "Output folder"))
        form.addRow("Output folder", _path_row(self.output_edit, output_button))
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Default: the series name")
        form.addRow("Title", self.title_edit)
        self.memory_edit = QLineEdit()
        self.memory_edit.setPlaceholderText("Optional, measured on the reference machine")
        form.addRow("Peak memory per worker (GiB)", self.memory_edit)
        self.memory_basis_edit = QLineEdit()
        self.memory_basis_edit.setPlaceholderText("How it was measured")
        form.addRow("Peak memory basis", self.memory_basis_edit)
        self.proof_edit = QLineEdit()
        proof_button = QPushButton("Browse…")
        proof_button.clicked.connect(lambda: self._choose_file(self.proof_edit, "Proof record"))
        form.addRow("Revision proof (optional)", _path_row(self.proof_edit, proof_button))
        self.variation_edit = QLineEdit()
        variation_button = QPushButton("Browse…")
        variation_button.clicked.connect(lambda: self._choose_file(self.variation_edit, "Known variation"))
        form.addRow("Known variation (optional)", _path_row(self.variation_edit, variation_button))
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("Check")
        self.check_button.clicked.connect(lambda: self.start(check_only=True))
        self.build_button = QPushButton("Build")
        self.build_button.clicked.connect(lambda: self.start(check_only=False))
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        for button in (self.check_button, self.build_button, close_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMinimumHeight(220)
        layout.addWidget(self.report_view, 1)
        self.last_report = None
        self._record_changed()

    def _record_changed(self, *_):
        """Offer the simulator resolution and geometry only for an S record."""
        simulator = self.record_combo.currentData() == "S"
        for widget in (self.resolution_edit, self.geometry_edit, self.geometry_button):
            widget.setEnabled(simulator)

    def _choose_folder(self, edit: QLineEdit, title: str):
        """Let the user pick a folder for ``edit``."""
        folder = QFileDialog.getExistingDirectory(self, title, edit.text() or str(self.series_root))
        if folder:
            edit.setText(folder)

    def _choose_file(self, edit: QLineEdit, title: str):
        """Let the user pick a JSON file for ``edit``."""
        path, _filter = QFileDialog.getOpenFileName(self, title, edit.text() or str(self.series_root),
                                                    "JSON (*.json);;All files (*)")
        if path:
            edit.setText(path)

    def request(self):
        """The build request of the dialog's fields; raise ValueError for an invalid field."""
        from .builder import BuildRequest

        if not self.job_edit.text().strip() or not self.output_edit.text().strip():
            raise ValueError("Choose the production job and an output folder")
        record = self.record_combo.currentData()
        resolution = None
        if record == "S":
            if not self.resolution_edit.text().strip():
                raise ValueError("A simulator record needs an explicit resolution; there is no default")
            resolution = float(self.resolution_edit.text())
        memory = None
        if self.memory_edit.text().strip():
            memory = int(round(float(self.memory_edit.text()) * GIB))

        def optional(edit):
            """The field's path, or None when it is empty."""
            return Path(edit.text().strip()) if edit.text().strip() else None

        return BuildRequest(series_root=self.series_root, job_root=Path(self.job_edit.text().strip()),
                            record=record, tier=self.tier_combo.currentData(),
                            output=Path(self.output_edit.text().strip()), title=self.title_edit.text().strip(),
                            proof=optional(self.proof_edit), known_variation=optional(self.variation_edit),
                            peak_memory_bytes=memory, peak_memory_basis=self.memory_basis_edit.text().strip(),
                            simulator_resolution_mm=resolution,
                            simulator_geometry=optional(self.geometry_edit) if record == "S" else None)

    def start(self, *, check_only: bool) -> bool:
        """Start a check or a build in a worker thread; False when a field is invalid or one runs."""
        if self.thread is not None:
            return False
        try:
            request = self.request()
        except ValueError as error:
            self.report_view.setPlainText(str(error))
            return False
        self.report_view.setPlainText("Checking…" if check_only else "Checking, then building…")
        for button in (self.check_button, self.build_button):
            button.setEnabled(False)
        self.thread = BuildThread(request, check_only, self)
        self.thread.finished_report.connect(self._finished)
        self.thread.start()
        return True

    def _finished(self, report, error: str):
        """Show the report (or the traceback) and offer Check and Build again."""
        self.last_report = report
        self.report_view.setPlainText(error or report.text())
        for button in (self.check_button, self.build_button):
            button.setEnabled(True)
        thread, self.thread = self.thread, None
        if thread is not None:
            thread.wait()
            thread.deleteLater()

    def busy(self) -> bool:
        """True while a check or build runs."""
        return self.thread is not None

    def closeEvent(self, event):
        """Wait for a running check or build before the dialog closes."""
        if self.thread is not None:
            self.thread.wait()
        super().closeEvent(event)
