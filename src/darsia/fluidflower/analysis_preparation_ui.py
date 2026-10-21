"""Explicit staged plan preparation through the Jobs page's existing call runner."""
from __future__ import annotations

from pathlib import Path
import traceback

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from . import analysis_preparation as preparation
from .analysis_plan import adopt_prepared_plan
from .pipeline.job_queue import descriptor


class AnalysisPreparationDialog(QDialog):
    """Preparing or adopting a plan never queues or starts a calculation."""

    adopted = Signal(object)

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prepare analysis plan")
        self.series_root = panel.series_root
        self._start_call = panel._start_call
        self._thread = self._worker = None
        self.prepared_result = None
        layout = QVBoxLayout(self)
        note = QLabel("Prepare a new plan from the saved image inputs. Review and adoption are separate; no job or worker starts.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self._file_buttons = {}
        self.stage = QComboBox()
        self.stage.addItem("Phase maps and statistics", "phase1")
        self.stage.addItem("Mass from finished phase maps", "phase2")
        form.addRow("Analysis stage", self.stage)
        self.inputs = self._file_row(form, "Saved run inputs")
        self.geometry = self._file_row(form, "Measured geometry index")
        self.pressure = self._file_row(form, "Running pressure index")
        self.calibration = self._file_row(form, "Fitted mass calibration")
        self.name = QLineEdit("analysis")
        form.addRow("Plan name", self.name)
        self.folder = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.folder)
        browse = QPushButton("Choose parent folder…")
        browse.clicked.connect(self._choose_folder)
        row.addWidget(browse)
        form.addRow("Fresh plan folder", row)
        self.probability = QCheckBox("Store exact gas-probability decision values")
        form.addRow(self.probability)
        self.resolution = QLabel()
        self.resolution.setWordWrap(True)
        form.addRow("Resolution and files", self.resolution)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare plan")
        self.prepare_button.clicked.connect(self.prepare)
        self.adopt_button = QPushButton("Review and adopt prepared plan…")
        self.adopt_button.setEnabled(False)
        self.adopt_button.clicked.connect(self.adopt)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.prepare_button)
        buttons.addWidget(self.adopt_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.status = QLabel("No plan prepared. Missing per-image roles are reported for each run.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.stage.currentIndexChanged.connect(self._stage_changed)
        self._stage_changed()

    def _file_row(self, form, label):
        edit = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(edit)
        button = QPushButton("Choose…")
        button.clicked.connect(lambda: self._choose_file(edit, label))
        self._file_buttons[edit] = button
        row.addWidget(button)
        form.addRow(label, row)
        return edit

    def _choose_file(self, edit, label):
        path, _ = QFileDialog.getOpenFileName(self, label, edit.text(), "Saved data (*.json)")
        if path:
            edit.setText(path)

    def _choose_folder(self):
        parent = QFileDialog.getExistingDirectory(self, "Choose a parent for the fresh plan folder")
        if parent:
            self.folder.setText(str(Path(parent) / self.name.text().strip()))

    def _stage_changed(self):
        phase1 = self.stage.currentData() == "phase1"
        for field in (self.geometry, self.pressure, self.probability):
            field.setEnabled(phase1)
            if field in self._file_buttons:
                self._file_buttons[field].setEnabled(phase1)
        self.calibration.setEnabled(not phase1)
        self._file_buttons[self.calibration].setEnabled(not phase1)
        self.resolution.setText(
            "CPU, float32, native computation. Full native evidence; the pressure and geometry indices bind every selected image."
            if phase1 else
            "CPU, float32. The saved completion inputs bind the storage size and pixel domain; no native processing or phase inference is repeated.")

    def _run(self, call, *, adopting=False):
        if self._thread is not None:
            return
        self.prepare_button.setEnabled(False)
        self.adopt_button.setEnabled(False)
        self.status.setText("Adopting the reviewed plan…" if adopting else "Preparing and validating the plan…")
        def invoke():
            try:
                return dict(value=call(), adopting=adopting)
            except Exception:
                return dict(error=traceback.format_exc(), adopting=adopting)
        self._thread, self._worker = self._start_call(invoke, self._done, self._closed)

    @Slot(object)
    def _done(self, result):
        if "error" in result:
            self.status.setText(result["error"])
            return
        self.prepared_result = result["value"]
        bound = self.prepared_result["plan"]
        if result["adopting"]:
            self.status.setText("Plan adopted: " + bound["path"] + "\nNo job or worker started.")
            self.adopted.emit(bound)
        else:
            self.status.setText("Prepared, not adopted: " + bound["path"] + "\nSHA-256: " + bound["sha256"] + "\nNo job or worker started.")

    @Slot()
    def _closed(self):
        if self._thread is not None:
            self._thread.wait()
        self._thread = self._worker = None
        self.prepare_button.setEnabled(True)
        self.adopt_button.setEnabled(bool(self.prepared_result and not self.prepared_result.get("adopted_for_execution")))

    def prepare(self):
        phase1 = self.stage.currentData() == "phase1"
        required = [self.inputs, self.folder, self.name]
        required += [self.geometry, self.pressure] if phase1 else [self.calibration]
        if not self.series_root or any(not field.text().strip() for field in required):
            self.status.setText("Choose the registered series, saved inputs, required sources and a fresh plan folder.")
            return
        inputs, folder, job_id = self.inputs.text().strip(), self.folder.text().strip(), self.name.text().strip()
        geometry, pressure, calibration = (field.text().strip() for field in (self.geometry, self.pressure, self.calibration))
        probability, series_root = self.probability.isChecked(), self.series_root
        repository = Path(__file__).resolve().parents[3]
        self.prepared_result = None
        def call():
            job = preparation.job_identity(repository)
            if phase1:
                return preparation.prepare_phase1_inputs(descriptor(Path(inputs)), folder,
                    series_root=series_root, geometry_index=descriptor(Path(geometry)),
                    pressure_index=descriptor(Path(pressure)), gas_probability=probability,
                    job_id=job_id, job=job)
            return preparation.prepare_phase2_inputs(descriptor(Path(inputs)), folder,
                series_root=series_root, fitted_calibration=descriptor(Path(calibration)), job_id=job_id, job=job)
        self._run(call)

    def adopt(self):
        if not self.prepared_result or self._thread is not None or self.prepared_result.get("adopted_for_execution"):
            return
        bound = dict(self.prepared_result["plan"])
        actor, accepted = QInputDialog.getText(self, "Review approval", "Approved by")
        if not accepted or not actor.strip():
            return
        reason, accepted = QInputDialog.getText(self, "Review approval", "Reason for adopting these exact inputs")
        if not accepted or not reason.strip():
            return
        answer = QMessageBox.question(self, "Adopt prepared analysis plan?",
            bound["path"] + "\nSHA-256: " + bound["sha256"] +
            "\nAdopt exactly this plan? This creates no job and starts no worker.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        source = Path(bound["path"])
        self._run(lambda: adopt_prepared_plan(source, source.with_name("execution.adopted.toml"),
            expected_sha256=bound["sha256"], actor=actor.strip(), reason=reason.strip()), adopting=True)

    def closeEvent(self, event):
        if self._thread is not None:
            event.ignore()
            return
        super().closeEvent(event)

    def reject(self):
        if self._thread is None:
            super().reject()
