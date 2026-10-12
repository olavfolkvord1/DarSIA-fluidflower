"""Compact, executable calibration setup for the ordinary Phase Model Studio."""
from __future__ import annotations

import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from darsia.fluidflower.tools.phase_studio_calibration_launch import (
    calibration_cohorts, documented_ffum_profile)


class CohortLookup(QThread):
    ready = Signal(int, object, str)

    def __init__(self, catalog, generation, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.generation = generation

    def run(self):
        try:
            cohorts, blocked = calibration_cohorts(self.catalog)
            profiles = [dict(id="registered", label="Registered series physical profile",
                             cohorts=cohorts, blocked=blocked)]
            if self.catalog.get("dataset") == "ffum":
                try:
                    _, receipt = documented_ffum_profile(self.catalog)
                    cohorts, blocked = calibration_cohorts(
                        self.catalog, physical_profile="documented_ffum")
                    profiles.append(dict(id="documented_ffum",
                        label="Documented RUN9/11 assumptions · 13-run FF_UM comparison",
                        note="RUN9 uses documented pressure/temperature assumptions and "
                             "a borrowed RUN12 clear-water reference.",
                        detail=receipt.get("physical_assumptions_note", ""),
                        cohorts=cohorts, blocked=blocked))
                except (OSError, ValueError, KeyError):
                    # A different phase model must not inherit this saved
                    # model-specific physical assumption by accident.
                    pass
            self.ready.emit(self.generation, profiles, "")
        except Exception as error:
            self.ready.emit(self.generation, [], str(error))


class CalibrationControls(QWidget):
    calculate_requested = Signal()
    attach_overflow_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.catalog = None
        self.lookup = None
        self.lookups = []
        self._lookup_generation = 0
        self._cohort_summary = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        introduction = QLabel(
            "Fit the optical response on the selected phase model's reviewed images. "
            "One shared response is fitted only within confirmed chemistry; each included "
            "experiment also receives an individual comparison. Results stay on this series' NAS.")
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        form = QFormLayout()
        layout.addLayout(form)
        self.physical_profile = QComboBox()
        self.physical_profile.currentIndexChanged.connect(self._profile_changed)
        form.addRow("Physical input profile", self.physical_profile)
        self.cohort = QComboBox()
        self.cohort.currentIndexChanged.connect(self._cohort_changed)
        form.addRow("Experiments and chemistry", self.cohort)
        self.method = QComboBox()
        self.method.addItem("Registered correction · joint optical fit", "registered")
        self.method.addItem("Recalculate correction variants · same phase maps", "photometry")
        self.method.currentIndexChanged.connect(self._method_changed)
        form.addRow("Optical method", self.method)
        self.correction_set = QComboBox()
        self._set_correction_choices(None)
        form.addRow("Correction comparison", self.correction_set)
        self.overflow_mode = QComboBox()
        self.overflow_mode.addItem("No overflow target", "off")
        self.overflow_mode.addItem("Compare with measured overflow", "compare")
        self.overflow_mode.addItem("Weighted measured overflow guidance", "constrain")
        self.overflow_mode.addItem("Require agreement within measured interval; fail if infeasible", "hard")
        self.overflow_mode.currentIndexChanged.connect(self._method_changed)
        form.addRow("Overflow", self.overflow_mode)
        self.overflow_weight = QDoubleSpinBox()
        self.overflow_weight.setRange(.05, 10.)
        self.overflow_weight.setSingleStep(.25)
        self.overflow_weight.setValue(3.)
        self.overflow_weight.setToolTip(
            "Weighted guidance is a soft penalty, not a guaranteed physical match. "
            "The hard option rejects any fit that the visible gas capacity cannot support.")
        form.addRow("Guidance weight", self.overflow_weight)
        parallel = QHBoxLayout()
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, min(32, os.cpu_count() or 1)))
        self.workers.setValue(min(4, self.workers.maximum()))
        parallel.addWidget(self.workers)
        self.width = QComboBox()
        for resolution in (768, 1024, 1536):
            self.width.addItem(f"{resolution} px wide", resolution)
        self.width.setCurrentIndex(self.width.findData(1536))
        parallel.addWidget(self.width)
        parallel.addStretch(1)
        form.addRow("CPU workers / grid", parallel)
        selection = QHBoxLayout()
        self.image_selection_receipt = QLineEdit()
        self.image_selection_receipt.setPlaceholderText(
            "Optional calibrated-image selection receipt on this series' NAS"
        )
        self.image_selection_receipt.setToolTip(
            "Use a ready calibration receipt produced by the image-selection plan. "
            "The worker verifies the receipt against the selected phase model's native "
            "image timeline. Leave blank to use every prepared phase image."
        )
        selection.addWidget(self.image_selection_receipt, 1)
        choose_receipt = QPushButton("Browse…")
        choose_receipt.clicked.connect(self._choose_image_selection_receipt)
        selection.addWidget(choose_receipt)
        form.addRow("Calibration images", selection)
        attach = QHBoxLayout()
        self.target_run = QComboBox()
        attach.addWidget(self.target_run)
        self.attach = QPushButton("Attach measured overflow…")
        self.attach.clicked.connect(self._attach)
        attach.addWidget(self.attach)
        form.addRow("Overflow experiment", attach)
        self.status = QLabel("Load a phase model to list compatible experiments.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QLabel(
            "Fixed phase labels and registered image identity are retained. The optical correction "
            "comparison recalculates both each photograph and its clear-water reference. "
            "The persistent full-depth gas rule is a separate experimental scenario; "
            "this joint fit still estimates one scalar visible-gas saturation.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.calculate = QPushButton("Start calibration in background")
        self.calculate.setEnabled(False)
        self.calculate.clicked.connect(self.calculate_requested.emit)
        layout.addWidget(self.calculate)
        self._method_changed()

    def set_catalog(self, catalog):
        previous_identity = (
            (self.catalog or {}).get("dataset"),
            (self.catalog or {}).get("model_id"),
        )
        next_identity = (
            (catalog or {}).get("dataset"),
            (catalog or {}).get("model_id"),
        )
        if previous_identity != next_identity:
            self.image_selection_receipt.clear()
        self._lookup_generation += 1
        self.catalog = catalog
        self._set_correction_choices(catalog.get("dataset") if catalog else None)
        self.physical_profile.clear()
        self.cohort.clear()
        self.target_run.clear()
        self.calculate.setEnabled(False)
        if catalog is None:
            self.status.setText("Load a phase model to list compatible experiments.")
            return
        self.status.setText("Checking registered physical inputs and chemistry on NAS…")
        self.lookup = CohortLookup(catalog, self._lookup_generation, self)
        self.lookups.append(self.lookup)
        self.lookup.ready.connect(self._cohorts_ready)
        self.lookup.finished.connect(lambda lookup=self.lookup: self.lookups.remove(lookup)
                                     if lookup in self.lookups else None)
        self.lookup.start()

    def _choose_image_selection_receipt(self):
        root = (self.catalog or {}).get("results_root") or ""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select audited calibration image selection",
            str(root),
            "JSON receipt (*.json)",
        )
        if selected:
            self.image_selection_receipt.setText(selected)

    def _set_correction_choices(self, series):
        selected = self.correction_set.currentData()
        self.correction_set.clear()
        if series == "ffum":
            self.correction_set.addItem(
                "No spatial light · ColorChecker off and on", "ffum_allowed")
            self.correction_set.setToolTip(
                "Historical and facies illumination are excluded from FF_UM calibration.")
        else:
            self.correction_set.addItem(
                "Working pair: no spatial light vs facies light", "pair")
            self.correction_set.addItem(
                "All permitted light/ColorChecker combinations", "all")
            self.correction_set.setToolTip("")
        if selected is not None and self.correction_set.findData(selected) >= 0:
            self.correction_set.setCurrentIndex(self.correction_set.findData(selected))

    def _cohorts_ready(self, generation, profiles, error):
        if not self.catalog or generation != self._lookup_generation:
            return
        if error:
            self.status.setText("Calibration inputs unavailable: " + error)
            return
        self.physical_profile.blockSignals(True)
        self.physical_profile.clear()
        for profile in profiles:
            self.physical_profile.addItem(profile["label"], profile)
        documented = next((index for index, profile in enumerate(profiles)
                           if profile["id"] == "documented_ffum"), None)
        if documented is not None:
            self.physical_profile.setCurrentIndex(documented)
        self.physical_profile.blockSignals(False)
        self._profile_changed()

    def _profile_changed(self, *_):
        profile = self.physical_profile.currentData() or {}
        cohorts = profile.get("cohorts", [])
        blocked = profile.get("blocked", [])
        self.cohort.blockSignals(True)
        self.cohort.clear()
        for cohort in cohorts:
            self.cohort.addItem(cohort["label"], cohort)
        self.cohort.blockSignals(False)
        unavailable = ", ".join(row["run"].upper() for row in blocked)
        reasons = "\n".join(
            f"{row['run'].upper()}: {' '.join(row['reasons'])}" for row in blocked)
        self._cohort_summary = (str(profile.get("note", "")) + " "
                   + f"{len(cohorts)} compatible choices · "
                   f"{len(blocked)} experiments unavailable"
                   + (f" ({unavailable}). " if blocked else ". ")
                   + "Individual fits for all runs in a shared choice are included automatically.")
        self.status.setText(self._cohort_summary)
        self.status.setToolTip("\n".join(filter(None, [profile.get("detail", ""), reasons])))
        self._cohort_changed()

    def _cohort_changed(self, *_):
        cohort = self.cohort.currentData() or {}
        selected = self.target_run.currentData()
        self.target_run.clear()
        for run in cohort.get("runs", []):
            self.target_run.addItem(run.upper(), run)
        if selected in cohort.get("runs", []):
            self.target_run.setCurrentIndex(self.target_run.findData(selected))
        self.calculate.setEnabled(bool(cohort))
        self._method_changed()

    def _method_changed(self, *_):
        photometry = self.method.currentData() == "photometry"
        cohort = self.cohort.currentData() or {}
        if photometry and not cohort.get("shared"):
            self.status.setText("Correction comparison requires a shared chemistry group with at least two runs.")
        elif self._cohort_summary:
            self.status.setText(self._cohort_summary)
        self.correction_set.setEnabled(photometry)
        self.overflow_mode.setEnabled(not photometry)
        self.overflow_weight.setEnabled(not photometry and self.overflow_mode.currentData() == "constrain")
        self.attach.setEnabled(not photometry and bool(cohort))
        self.calculate.setEnabled(bool(cohort) and (not photometry or cohort.get("shared", False)))

    def _attach(self):
        run = self.target_run.currentData()
        if run:
            self.attach_overflow_requested.emit(run)

    def request_options(self):
        cohort = self.cohort.currentData()
        if not cohort:
            raise ValueError("Select compatible experiments")
        method = self.method.currentData()
        series = self.catalog["dataset"]
        if series == "ffum":
            variants = ["none_cc_off", "none_cc_on"]
        elif self.correction_set.currentData() == "all":
            variants = ([f"{illumination}_cc_on" for illumination in ("none", "historical", "facies")]
                        if series == "ffac" else
                        [f"{illumination}_cc_{setting}" for illumination in ("none", "historical", "facies")
                         for setting in ("off", "on")])
        else:
            variants = (["none_cc_on", "facies_cc_on"] if series == "ffac"
                        else ["none_cc_off", "facies_cc_off"])
        return dict(cohort=cohort, method=method, overflow_mode=(
            "off" if method == "photometry" else self.overflow_mode.currentData()),
            physical_profile=(self.physical_profile.currentData() or {}).get("id", "registered"),
            overflow_weight=self.overflow_weight.value(), workers=self.workers.value(),
            analysis_width=self.width.currentData(), photometry_variants=variants,
            image_selection_receipt_path=(
                self.image_selection_receipt.text().strip() or None
            ))
