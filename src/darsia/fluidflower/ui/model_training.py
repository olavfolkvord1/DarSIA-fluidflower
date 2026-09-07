"""Project-independent Qt work surface for shared optical phase-model training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
import os
from pathlib import Path
import uuid

from PySide6.QtCore import QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.phase_model_history import discover_phase_model_history
from darsia.fluidflower.phase_workflow import (
    build_phase_review_command,
    build_phase_review_resume_command,
    build_phase_training_command,
    default_phase_review_roots,
    discover_phase_review_sessions,
    parse_training_bundles,
    phase_python_executable,
    prepare_phase_training_request,
)
from darsia.fluidflower.project import default_project_root, sha256_file
from darsia.fluidflower.ui.common import FILE_DIALOG_OPTIONS, PathField


def _secondary(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


def _setting_field(control: QWidget, explanation: str) -> QWidget:
    field = QWidget()
    layout = QVBoxLayout(field)
    layout.setContentsMargins(0, 0, 0, 5)
    layout.setSpacing(3)
    layout.addWidget(control, 0, Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(_secondary(explanation))
    control.setToolTip(explanation)
    return field


class SharedPhaseModelTrainingPage(QWidget):
    """Collect evidence and train immutable cross-series phase models."""

    training_completed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.training_process: QProcess | None = None
        self.training_request: dict[str, object] | None = None
        self.last_output_root: Path | None = None
        self.validation_process: QProcess | None = None
        self.pending_validation_mode = ""
        self.pending_validation_session_id = ""
        self.pending_validation_output: Path | None = None
        self.review_sessions: list[dict[str, object]] = []
        self.visible_review_sessions: list[dict[str, object]] = []
        self.model_history: list[dict[str, object]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        heading = QLabel("Shared optical phase model")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        layout.addWidget(
            _secondary(
                "Develop one transferable visible-2D phase model from reviewed "
                "FF_AC, FF_UM, and future FluidFlower evidence. Region labeling, "
                "open correction, and model fitting are separate operations. "
                "Concentration, depth conversion, and mass remain specific to each "
                "experimental series."
            )
        )

        identity = QHBoxLayout()
        identity.addWidget(QLabel("Researcher / reviewer"))
        self.reviewer = QLineEdit()
        self.reviewer.setMinimumWidth(260)
        self.reviewer.setPlaceholderText("Required before reviewing or training")
        self.reviewer.setToolTip(
            "Stored on every region label, correction, training request, and model version."
        )
        identity.addWidget(self.reviewer)
        identity.addWidget(
            _secondary("Your name is provenance. It does not change model weighting."),
            1,
        )
        layout.addLayout(identity)

        self.tabs = QTabWidget()
        self.evidence_tab = self._build_evidence_tab()
        self.training_tab = self._build_training_tab()
        self.history_tab = self._build_history_tab()
        self.tabs.addTab(self.evidence_tab, "Review and label evidence")
        self.tabs.addTab(self.training_tab, "Train a new model version")
        self.tabs.addTab(self.history_tab, "Model history and validation")
        layout.addWidget(self.tabs, 1)

        self.reviewer.textChanged.connect(self._update_action_states)
        self.bundles.textChanged.connect(self._update_action_states)
        self.radiometry_mode.currentIndexChanged.connect(self._update_action_states)
        self.radiometry_profile_path.changed.connect(self._update_action_states)
        for field in (
            self.ffac_campaign,
            self.ffum_campaign,
            self.combined_campaign,
            self.review_save_root,
        ):
            field.changed.connect(self._update_action_states)
        for control in (
            self.backend,
            self.stage_workers,
            self.extraction_workers,
            self.pixels,
            self.mil_iterations,
            self.change_ablation,
        ):
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else (
                    control.stateChanged
                    if isinstance(control, QCheckBox)
                    else control.valueChanged
                )
            )
            signal.connect(self._update_training_effect_summary)

        self._refresh_review_sessions()
        self._refresh_model_history()
        self._update_training_effect_summary()
        self._update_action_states()

    def _build_evidence_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Use blind phase labeling when you want an independent human label "
                "for a model-selected region. Use open model correction when the "
                "prediction itself must be inspected and localized errors corrected. "
                "Both retain the full-resolution timeline and continuous local autosave."
            )
        )

        model = QFrame()
        model.setFrameShape(QFrame.Shape.StyledPanel)
        model_layout = QGridLayout(model)
        model_layout.addWidget(QLabel("Model to validate"), 0, 0, 1, 3)
        self.validation_model = QComboBox()
        self.validation_model.setMinimumContentsLength(58)
        self.validation_model.setToolTip(
            "Select the exact immutable model version to review. If its phase maps "
            "do not exist yet, DarSIA generates them from the prepared corrected "
            "frames before opening a new model-bound session."
        )
        self.validation_model.currentIndexChanged.connect(
            self._validation_model_changed
        )
        model_layout.addWidget(self.validation_model, 1, 0, 1, 2)
        refresh_models = QPushButton("Refresh models")
        refresh_models.clicked.connect(self._refresh_model_history)
        model_layout.addWidget(refresh_models, 1, 2)
        self.validation_model_summary = _secondary(
            "No model selected. Model choice is separate from saved review sessions."
        )
        model_layout.addWidget(self.validation_model_summary, 2, 0, 1, 3)
        layout.addWidget(model)

        recent = QFrame()
        recent.setFrameShape(QFrame.Shape.StyledPanel)
        recent_layout = QGridLayout(recent)
        recent_layout.addWidget(QLabel("Saved review sessions"), 0, 0, 1, 3)
        self.session_scope = QComboBox()
        self.session_scope.addItem("Sessions for selected model", "selected_model")
        self.session_scope.addItem("All historical sessions", "all")
        self.session_scope.setToolTip(
            "The default view only shows sessions bound to the selected model. "
            "Use all history to audit evidence for earlier models."
        )
        self.session_scope.currentIndexChanged.connect(self._rebuild_session_combo)
        recent_layout.addWidget(self.session_scope, 1, 0, 1, 3)
        self.saved_sessions = QComboBox()
        self.saved_sessions.setMinimumContentsLength(58)
        self.saved_sessions.currentIndexChanged.connect(self._session_changed)
        recent_layout.addWidget(self.saved_sessions, 2, 0, 1, 2)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_review_sessions)
        recent_layout.addWidget(refresh, 2, 2)
        self.session_summary = _secondary("No local review session found.")
        recent_layout.addWidget(self.session_summary, 3, 0, 1, 3)
        self.resume_session_button = QPushButton("Resume selected session")
        self.resume_session_button.setToolTip(
            "Reopens the original workspace and decision directory without converting "
            "or replacing existing annotations."
        )
        self.resume_session_button.clicked.connect(self._resume_selected_session)
        recent_layout.addWidget(self.resume_session_button, 4, 0, 1, 2)
        self.add_session_evidence = QPushButton("Add this session to training inputs")
        self.add_session_evidence.clicked.connect(self._add_selected_session_as_bundle)
        recent_layout.addWidget(self.add_session_evidence, 4, 2)
        layout.addWidget(recent)

        sources = QFrame()
        sources.setFrameShape(QFrame.Shape.StyledPanel)
        source_form = QFormLayout(sources)
        self.ffac_campaign = PathField(
            directory=True,
            placeholder="Prepared FF_AC campaign root",
        )
        self.ffum_campaign = PathField(
            directory=True,
            placeholder="Prepared FF_UM campaign root",
        )
        self.combined_campaign = PathField(
            directory=True,
            placeholder="Local combined review package",
        )
        self.review_save_root = PathField(
            directory=True,
            placeholder="Reviewer-owned local save root",
        )
        roots = default_phase_review_roots()
        if roots:
            self.review_save_root.setText(str(roots[0]))
        self.review_gpu = QCheckBox("Use GPU-accelerated image display")
        self.review_gpu.setChecked(True)
        self.review_gpu.setToolTip(
            "Accelerates zooming and overlays only. It does not change image pixels, "
            "phase predictions, or saved labels."
        )
        source_form.addRow(
            "FF_AC prepared campaign",
            _setting_field(
                self.ffac_campaign,
                "Folder containing FF_AC frames, phase-map assets, and its "
                "postflight manifest. This is read-only source evidence.",
            ),
        )
        source_form.addRow(
            "FF_UM prepared campaign",
            _setting_field(
                self.ffum_campaign,
                "Equivalent read-only prepared evidence for FF_UM. The two series "
                "remain identifiable even though they train one shared optical model.",
            ),
        )
        source_form.addRow(
            "Combined review package",
            _setting_field(
                self.combined_campaign,
                "The manifest that joins experiments from both series and records "
                "which immutable model candidates and frames belong together.",
            ),
        )
        source_form.addRow(
            "Local autosave location",
            _setting_field(
                self.review_save_root,
                "Your reviewer-specific JSON, recovery journal, and session record "
                "are written here continuously. Source images are never changed.",
            ),
        )
        source_form.addRow(
            "Display",
            _setting_field(
                self.review_gpu,
                "GPU display makes zooming and overlays smoother. It has no effect "
                "on source resolution, predictions, or scientific output.",
            ),
        )
        layout.addWidget(sources)

        modes = QGridLayout()
        blind = QFrame()
        blind.setFrameShape(QFrame.Shape.StyledPanel)
        blind_layout = QVBoxLayout(blind)
        blind_heading = QLabel("Blind phase labeling")
        blind_heading.setObjectName("sectionTitle")
        blind_layout.addWidget(blind_heading)
        blind_layout.addWidget(
            _secondary(
                "The selected model ranks regions by predictive uncertainty computed "
                "from its phase probabilities: boxes with overlapping class "
                "probabilities are prioritized, while phase and position are balanced "
                "for diversity. This is an active-learning score, not physical "
                "uncertainty. "
                "Its proposed phase remains hidden until you answer. You can also draw "
                "your own boxes or freehand evidence without revealing the phase map."
            )
        )
        self.open_blind_labeling = QPushButton("Open blind phase labeling")
        self.open_blind_labeling.clicked.connect(
            lambda: self._launch_review("blind_region_labeling")
        )
        blind_layout.addWidget(self.open_blind_labeling)
        modes.addWidget(blind, 0, 0)

        correction = QFrame()
        correction.setFrameShape(QFrame.Shape.StyledPanel)
        correction_layout = QVBoxLayout(correction)
        correction_heading = QLabel("Open model correction")
        correction_heading.setObjectName("sectionTitle")
        correction_layout.addWidget(correction_heading)
        correction_layout.addWidget(
            _secondary(
                "The complete phase map from the version selected under Model to "
                "validate is visible. Inspect it through time, draw boxes or freehand "
                "regions around specific errors, and assign the correct phase or "
                "artifact class. Every new session records the exact selected model."
            )
        )
        self.open_model_correction = QPushButton("Open model correction")
        self.open_model_correction.clicked.connect(
            lambda: self._launch_review("open_correction")
        )
        correction_layout.addWidget(self.open_model_correction)
        modes.addWidget(correction, 0, 1)
        modes.setColumnStretch(0, 1)
        modes.setColumnStretch(1, 1)
        layout.addLayout(modes)

        self.review_status = _secondary(
            "Select a saved session or choose the three prepared campaign locations."
        )
        layout.addWidget(self.review_status)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_training_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Training never modifies an existing model. It freezes selected "
                "review files and creates a new immutable version. Complete-run "
                "holdouts test whether the model transfers between experiments "
                "rather than only memorizing reviewed frames."
            )
        )

        evidence = QFrame()
        evidence.setFrameShape(QFrame.Shape.StyledPanel)
        evidence_layout = QFormLayout(evidence)
        self.bundles = QTextEdit()
        self.bundles.setPlaceholderText(
            "One reviewed evidence package per row:\n"
            "DECISIONS_JSON | COMBINED_MANIFEST_JSON"
        )
        self.bundles.setMaximumHeight(118)
        self.bundles.setToolTip(
            "Each row joins human decisions to the immutable image/candidate manifest "
            "that gives those decisions their exact provenance."
        )
        bundle_actions = QHBoxLayout()
        self.add_bundle_button = QPushButton("Add evidence files...")
        self.add_bundle_button.clicked.connect(self._add_bundle)
        bundle_actions.addWidget(self.add_bundle_button)
        add_selected = QPushButton("Add selected saved session")
        add_selected.clicked.connect(self._add_selected_session_as_bundle)
        bundle_actions.addWidget(add_selected)
        bundle_actions.addStretch(1)
        bundle_widget = QWidget()
        bundle_widget.setLayout(bundle_actions)
        evidence_layout.addRow("Reviewed evidence packages", self.bundles)
        evidence_layout.addRow("", bundle_widget)
        self.output_parent = PathField(directory=True)
        self.output_parent.setText(
            str(default_project_root().parent / "Shared Phase Models")
        )
        self.output_parent.setToolTip(
            "Each run creates a new timestamped model folder here; earlier versions "
            "are never overwritten."
        )
        self.cache_parent = PathField(directory=True)
        self.cache_parent.setText(
            str(default_execution_cache_root() / "phase-model-training")
        )
        self.cache_parent.setToolTip(
            "Stores reusable extracted image features locally. Deleting it only makes "
            "the next training run slower."
        )
        evidence_layout.addRow(
            "New model versions",
            _setting_field(
                self.output_parent,
                "Every training run creates a separate timestamped folder. V5 and "
                "all earlier models remain unchanged and reproducible.",
            ),
        )
        evidence_layout.addRow(
            "Reusable local feature cache",
            _setting_field(
                self.cache_parent,
                "Keeps expensive image features for reuse. It can be deleted safely, "
                "but the next fit will need to extract those features again.",
            ),
        )
        layout.addWidget(evidence)

        radiometry_form = QFormLayout()
        self.radiometry_mode = QComboBox()
        self.radiometry_mode.addItem("Uniform24 - common ColorChecker reference image", "image")
        self.radiometry_mode.addItem("Uniform24 - published ColorChecker after 2014", "uniform24-after2014")
        self.radiometry_mode.addItem("Uniform24 - measured target profile", "custom")
        self.radiometry_mode.addItem("Legacy - unchanged prepared review images", "legacy")
        self.radiometry_mode.setToolTip(
            "All 24 chart colors define one common target across series. Each frame, "
            "baseline and previous image is corrected independently before feature "
            "extraction. The published target requires a matching physical chart. "
            "Old models retain their original input basis."
        )
        self.radiometry_profile_path = PathField(directory=False)
        self.radiometry_mode.currentIndexChanged.connect(
            lambda: self.radiometry_profile_path.setEnabled(self.radiometry_mode.currentData() in {"image", "custom"})
        )
        self.radiometry_profile_path.setEnabled(True)
        self.radiometry_profile_path.setToolTip(
            "Choose a stable RGB image with the ColorChecker visible, "
            "or a frozen profile JSON when 'measured target profile' is selected. "
            "One target is shared by all selected series; each input frame is fitted independently."
        )
        self.training_series = QLineEdit()
        self.training_series.setPlaceholderText("All series in selected evidence")
        self.training_series.setToolTip(
            "Leave empty for the shared model, or enter comma-separated IDs such as "
            "FF_AC, FF_UM or FF_BM for a specialist model. Complete-run holdout "
            "requirements still apply. No labels are removed from the source sessions."
        )
        radiometry_form.addRow("New-model image normalization", self.radiometry_mode)
        radiometry_form.addRow("Common reference image / profile", self.radiometry_profile_path)
        radiometry_form.addRow("Training series subset", self.training_series)
        layout.addLayout(radiometry_form)

        parameter_frame = QFrame()
        parameter_frame.setFrameShape(QFrame.Shape.StyledPanel)
        parameter_form = QFormLayout(parameter_frame)
        self.backend = QComboBox()
        self.backend.addItem("Automatic (recommended)", "auto")
        self.backend.addItem("NVIDIA CUDA · float32", "cuda")
        self.backend.addItem("CPU reference · float64", "cpu")
        parameter_form.addRow(
            "Model-fitting device",
            _setting_field(
                self.backend,
                "Automatic uses CUDA when available and otherwise CPU. CUDA is faster; "
                "CPU float64 is the numerical reference. Backend parity is checked and "
                "the selected device is recorded in the model.",
            ),
        )
        self.stage_workers = QSpinBox()
        self.stage_workers.setRange(1, 64)
        self.stage_workers.setValue(4)
        parameter_form.addRow(
            "Evidence preparation workers",
            _setting_field(
                self.stage_workers,
                "Copies and verifies independent run assets in parallel. More workers "
                "can shorten preparation but increase storage traffic and RAM use; this "
                "setting does not change the fitted model.",
            ),
        )
        self.extraction_workers = QSpinBox()
        self.extraction_workers.setRange(1, 64)
        self.extraction_workers.setValue(4)
        parameter_form.addRow(
            "Feature extraction workers",
            _setting_field(
                self.extraction_workers,
                "Builds RGB, texture, and DarSIA change features for different runs in "
                "parallel. More workers improve speed only while CPU, RAM, and source "
                "storage have capacity.",
            ),
        )
        self.pixels = QSpinBox()
        self.pixels.setRange(64, 1_000_000)
        self.pixels.setValue(1024)
        self.pixels.setSingleStep(256)
        parameter_form.addRow(
            "Maximum pixels sampled per region",
            _setting_field(
                self.pixels,
                "Controls how densely each reviewed box or freehand area represents its "
                "internal color and texture variation. Higher values cost memory and "
                "time; they do not increase source-image resolution or replace more "
                "independent annotations.",
            ),
        )
        self.mil_iterations = QSpinBox()
        self.mil_iterations.setRange(20, 2_000)
        self.mil_iterations.setValue(160)
        self.mil_iterations.setSingleStep(20)
        parameter_form.addRow(
            "Mixed-region fitting iterations",
            _setting_field(
                self.mil_iterations,
                "Upper limit for resolving mixed labels into phase evidence. Raising it "
                "may help a difficult fit converge but cannot repair incorrect or "
                "unbalanced human labels, and increases fitting time.",
            ),
        )
        self.change_ablation = QCheckBox(
            "Train a control without DarSIA image-change evidence"
        )
        self.change_ablation.setChecked(True)
        parameter_form.addRow(
            "Scientific control",
            _setting_field(
                self.change_ablation,
                "Fits one additional non-deployable control on the same cached pixels. "
                "The complete-run holdout difference measures whether DarSIA's native "
                "change detector adds transferable information. It adds fitting time "
                "but does not alter the main model.",
            ),
        )
        layout.addWidget(parameter_frame)

        self.training_effect_summary = _secondary("")
        layout.addWidget(self.training_effect_summary)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Train new immutable model version")
        self.start_button.clicked.connect(self._start_training)
        actions.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop after current checkpoint")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_training)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.status = _secondary("Add reviewed evidence and enter a researcher name.")
        layout.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.log)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_history_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(
            _secondary(
                "Every training attempt remains an immutable model version. This "
                "view distinguishes training completion, internal complete-run "
                "holdout results, human validation, and active production use. "
                "Opening or inspecting a model never changes its files."
            )
        )

        location = QFrame()
        location.setFrameShape(QFrame.Shape.StyledPanel)
        location_layout = QFormLayout(location)
        self.model_library = PathField(
            directory=True,
            placeholder="Folder containing immutable phase-model versions",
        )
        configured = str(
            os.environ.get("DARSIA_FLUIDFLOWER_PHASE_MODEL_LIBRARY") or ""
        ).strip()
        self.model_library.setText(configured or self.output_parent.text())
        location_layout.addRow(
            "Model library",
            _setting_field(
                self.model_library,
                "The parent folder may contain model versions directly or in "
                "campaign subfolders. Only training requests, summaries, metrics, "
                "and small model files are read.",
            ),
        )
        location_actions = QHBoxLayout()
        refresh = QPushButton("Refresh model history")
        refresh.clicked.connect(self._refresh_model_history)
        location_actions.addWidget(refresh)
        use_output = QPushButton("Use new-model location")
        use_output.setToolTip(
            "Show the model versions stored under the output location selected "
            "on the training tab."
        )
        use_output.clicked.connect(self._use_training_output_as_history_root)
        location_actions.addWidget(use_output)
        location_actions.addStretch(1)
        location_layout.addRow("", location_actions)
        layout.addWidget(location)

        self.history_status = _secondary("No model library inspected yet.")
        layout.addWidget(self.history_status)
        self.history_table = QTableWidget(0, 10)
        self.history_table.setHorizontalHeaderLabels(
            [
                "Model version",
                "Created",
                "Lifecycle state",
                "Validation assets",
                "Series",
                "Runs",
                "Regions",
                "CO2 holdout",
                "Gas holdout",
                "Fit iterations",
            ]
        )
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, self.history_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.history_table.itemSelectionChanged.connect(
            self._model_history_selection_changed
        )
        layout.addWidget(self.history_table, 1)

        actions = QHBoxLayout()
        self.open_model_folder = QPushButton("Open selected model folder")
        self.open_model_folder.setEnabled(False)
        self.open_model_folder.clicked.connect(self._open_selected_model_folder)
        actions.addWidget(self.open_model_folder)
        self.use_model_for_validation = QPushButton("Use selected model for validation")
        self.use_model_for_validation.setEnabled(False)
        self.use_model_for_validation.clicked.connect(
            self._use_selected_history_model_for_validation
        )
        actions.addWidget(self.use_model_for_validation)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.model_history_details = QTextEdit()
        self.model_history_details.setReadOnly(True)
        self.model_history_details.setMinimumHeight(260)
        self.model_history_details.setPlaceholderText(
            "Select a model version to inspect training and validation provenance."
        )
        layout.addWidget(self.model_history_details)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _metric_text(value: object) -> str:
        return f"{100.0 * float(value):.2f}%" if isinstance(value, (int, float)) else "-"

    @staticmethod
    def _history_created_text(entry: Mapping[str, object]) -> str:
        raw = str(entry.get("created_utc") or "").strip()
        if not raw:
            return "Not recorded"
        try:
            return (
                datetime.fromisoformat(raw.replace("Z", "+00:00"))
                .astimezone()
                .strftime("%Y-%m-%d %H:%M")
            )
        except ValueError:
            return raw

    @staticmethod
    def _fit_iteration_text(entry: Mapping[str, object]) -> str:
        rows = [
            row
            for row in entry.get("heads") or []
            if isinstance(row, Mapping) and row.get("adapter") == "generic"
        ]
        if not rows:
            rows = [row for row in entry.get("heads") or [] if isinstance(row, Mapping)]
        values = []
        for row in rows:
            head = str(row.get("head") or "head").replace("_within_co2", "")
            iterations = int(row.get("iterations") or 0)
            suffix = "" if bool(row.get("converged")) else " max"
            values.append(f"{head}: {iterations}{suffix}")
        return "; ".join(values) if values else "-"

    @staticmethod
    def _read_manifest(root: object) -> dict[str, object]:
        path = Path(str(root or "")).expanduser() / "postflight_manifest.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, TypeError, ValueError):
            return {}
        return dict(payload) if isinstance(payload, Mapping) else {}

    def _validation_binding(
        self, entry: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Find precomputed review assets that explicitly reference one model root."""

        model_token = Path(str(entry.get("root") or "")).name.casefold()
        if not model_token:
            return None
        for session in self.review_sessions:
            selected: dict[str, str] = {}
            selected_roles: set[str] = set()
            roots_ready = True
            for key in ("ffac_root", "ffum_root"):
                manifest = self._read_manifest(session.get(key))
                if not manifest:
                    roots_ready = False
                    break
                models = manifest.get("models") or {}
                jobs = manifest.get("jobs") or []
                if not isinstance(models, Mapping) or not isinstance(jobs, list):
                    roots_ready = False
                    break
                matching_roles = []
                for role, raw_model in models.items():
                    if not isinstance(raw_model, Mapping):
                        continue
                    paths = (
                        str(raw_model.get("co2_path") or "").casefold(),
                        str(raw_model.get("gas_path") or "").casefold(),
                    )
                    if paths and all(model_token in path for path in paths):
                        matching_roles.append(str(role))
                if not matching_roles:
                    roots_ready = False
                    break
                matching_roles.sort(
                    key=lambda role: (
                        "physics" not in role.casefold(),
                        "generic" not in role.casefold(),
                        role,
                    )
                )
                selected_role = matching_roles[0]
                selected_roles.add(selected_role)
                for job in jobs:
                    if not isinstance(job, Mapping) or str(job.get("role")) != selected_role:
                        continue
                    run_id = str(job.get("run") or "").strip().casefold()
                    candidate_id = str(job.get("candidate_id") or "").strip()
                    if run_id and candidate_id:
                        selected[run_id] = candidate_id
            combined_root = Path(str(session.get("combined_root") or "")).expanduser()
            if (
                roots_ready
                and selected
                and (combined_root / "postflight_manifest.json").is_file()
            ):
                return {
                    "campaign_id": str(session.get("campaign_id") or ""),
                    "ffac_root": str(session.get("ffac_root") or ""),
                    "ffum_root": str(session.get("ffum_root") or ""),
                    "combined_root": str(combined_root.resolve()),
                    "selected_candidates": selected,
                    "roles": sorted(selected_roles),
                    "run_count": len(selected),
                }
        return None

    def _validation_source_binding(self) -> dict[str, str] | None:
        """Return the newest prepared corrected-frame campaign as source evidence."""

        for session in self.review_sessions:
            roots = {
                key: str(session.get(key) or "").strip()
                for key in ("ffac_root", "ffum_root", "combined_root")
            }
            if all(
                value
                and (Path(value).expanduser() / "postflight_manifest.json").is_file()
                for value in roots.values()
            ):
                return roots
        return None

    @staticmethod
    def _validation_model_label(entry: Mapping[str, object]) -> str:
        state = str(entry.get("state_label") or "Unknown state")
        return f"{entry.get('model_id') or 'Unnamed model'} · {state}"

    def _populate_validation_models(self) -> None:
        selected_root = str(self.validation_model.currentData() or "")
        self.validation_model.blockSignals(True)
        self.validation_model.clear()
        selected_index = 0
        for index, entry in enumerate(self.model_history):
            root = str(entry.get("root") or "")
            self.validation_model.addItem(self._validation_model_label(entry), root)
            if root == selected_root:
                selected_index = index
        self.validation_model.setCurrentIndex(
            selected_index if self.model_history else -1
        )
        self.validation_model.blockSignals(False)
        self._validation_model_changed()

    def _selected_validation_model(self) -> Mapping[str, object] | None:
        root = str(self.validation_model.currentData() or "")
        return next(
            (entry for entry in self.model_history if str(entry.get("root") or "") == root),
            None,
        )

    def _validation_model_changed(self, _index: int = -1) -> None:
        entry = self._selected_validation_model()
        if entry is None:
            self.validation_model_summary.setText("No model selected.")
            self._update_action_states()
            return
        binding = self._validation_binding(entry)
        entry["validation_binding"] = binding
        if binding:
            self.ffac_campaign.setText(str(binding["ffac_root"]))
            self.ffum_campaign.setText(str(binding["ffum_root"]))
            self.combined_campaign.setText(str(binding["combined_root"]))
            self.validation_model_summary.setText(
                f"Selected: {entry.get('model_id')}. Prepared phase maps are available "
                f"for {binding['run_count']} experiment(s) in {binding['campaign_id']}; "
                f"opening validation explicitly selects {', '.join(binding['roles'])}."
            )
        else:
            source = self._validation_source_binding()
            if source and not self.combined_campaign.text().strip():
                self.ffac_campaign.setText(source["ffac_root"])
                self.ffum_campaign.setText(source["ffum_root"])
                self.combined_campaign.setText(source["combined_root"])
            self.validation_model_summary.setText(
                f"Selected: {entry.get('model_id')}. No cached phase maps are bound to "
                "this model yet. Starting either review mode will generate the selected "
                "model's predictions, create a new session, and then open it automatically."
                + (
                    " Corrected frames from the newest prepared campaign are selected."
                    if source
                    else " Choose a prepared combined review package first."
                )
            )
        if hasattr(self, "session_scope"):
            self._rebuild_session_combo()
        self._update_action_states()

    def _use_training_output_as_history_root(self) -> None:
        self.model_library.setText(self.output_parent.text())
        self._refresh_model_history()

    def _refresh_model_history(self) -> None:
        root_text = self.model_library.text().strip()
        self.history_table.setRowCount(0)
        self.model_history_details.clear()
        self.open_model_folder.setEnabled(False)
        self.use_model_for_validation.setEnabled(False)
        if not root_text:
            self.model_history = []
            self._populate_validation_models()
            self.history_status.setText("Choose a model-library folder.")
            return
        root = Path(root_text).expanduser()
        if not root.is_dir():
            self.model_history = []
            self._populate_validation_models()
            self.history_status.setText(f"Model library is unavailable: {root}")
            return
        try:
            self.model_history = [
                dict(entry) for entry in discover_phase_model_history(root)
            ]
        except Exception as exc:  # noqa: BLE001 - visible read-only UI boundary
            self.model_history = []
            self._populate_validation_models()
            self.history_status.setText(
                f"Could not inspect model history: {type(exc).__name__}: {exc}"
            )
            return

        for entry in self.model_history:
            entry["validation_binding"] = self._validation_binding(entry)
        self.history_table.setRowCount(len(self.model_history))
        for row_index, entry in enumerate(self.model_history):
            datasets = ", ".join(str(value).upper() for value in entry.get("datasets") or [])
            values = (
                str(entry.get("model_id") or ""),
                self._history_created_text(entry),
                str(entry.get("state_label") or ""),
                (
                    f"Ready ({entry['validation_binding']['run_count']} runs)"
                    if entry.get("validation_binding")
                    else "Predictions not generated"
                ),
                datasets or "-",
                str(len(entry.get("runs") or [])),
                f"{int(entry.get('region_count') or 0):,}",
                self._metric_text(entry.get("co2_balanced_accuracy")),
                self._metric_text(entry.get("gas_balanced_accuracy")),
                self._fit_iteration_text(entry),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(entry.get("root") or ""))
                self.history_table.setItem(row_index, column, item)
        self.history_status.setText(
            f"{len(self.model_history)} model version(s) found in {root.resolve()}. "
            "Earlier campaign models are retained as read-only history."
        )
        if self.model_history:
            self.history_table.selectRow(0)
        self._populate_validation_models()

    def _selected_model_history(self) -> Mapping[str, object] | None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return self.model_history[index] if 0 <= index < len(self.model_history) else None

    def _model_history_selection_changed(self) -> None:
        entry = self._selected_model_history()
        self.open_model_folder.setEnabled(entry is not None)
        self.use_model_for_validation.setEnabled(entry is not None)
        if entry is None:
            self.model_history_details.clear()
            return
        lines = [
            f"Model version: {entry.get('model_id')}",
            f"Location: {entry.get('root')}",
            f"Created: {self._history_created_text(entry)}",
            f"Created by: {entry.get('created_by')}",
            f"Lifecycle: {entry.get('state_label')}",
            f"Internal complete-run holdout passed: {'yes' if entry.get('internal_holdout_passed') else 'no'}",
            f"Human validation: {'recorded' if entry.get('human_validation') else 'not yet linked'}",
            f"Active production deployment: {'yes' if entry.get('deployment') else 'no'}",
            (
                "Validation assets: available"
                if entry.get("validation_binding")
                else "Validation assets: phase-map predictions have not been generated"
            ),
            f"Series: {', '.join(str(value).upper() for value in entry.get('datasets') or []) or '-'}",
            f"Runs ({len(entry.get('runs') or [])}): {', '.join(str(value) for value in entry.get('runs') or []) or '-'}",
            f"Reviewed regions: {int(entry.get('region_count') or 0):,}",
            f"Sampled training pixels: {int(entry.get('pixel_rows') or 0):,}",
            f"Adapters: {', '.join(str(value) for value in entry.get('adapters') or []) or '-'}",
            f"DarSIA change evidence: {entry.get('change_detector')}",
            f"Image normalization: {entry.get('radiometry_label', 'Legacy / untracked')}",
            f"Correction index: {Path(str(entry.get('root'))) / 'radiometry_inventory.json'}",
            "",
            "Head validation and fit history:",
        ]
        for row in entry.get("heads") or []:
            if not isinstance(row, Mapping):
                continue
            accuracy = self._metric_text(row.get("balanced_accuracy"))
            convergence = "converged" if row.get("converged") else "iteration limit reached"
            lines.append(
                f"- {row.get('adapter')} / {row.get('head')}: holdout {accuracy}; "
                f"threshold {row.get('threshold')}; {int(row.get('iterations') or 0)} "
                f"iterations ({convergence}); {row.get('backend') or '-'} "
                f"{row.get('precision') or ''}; internal eligibility "
                f"{'passed' if row.get('deployment_allowed') else 'not passed/control'}"
            )
        delta = entry.get("change_ablation_delta")
        if isinstance(delta, (int, float)):
            lines.extend(
                [
                    "",
                    "DarSIA change-evidence ablation:",
                    f"- Mean balanced-accuracy difference: {float(delta):+.4f}",
                ]
            )
        phase_classes = entry.get("phase_classes") or {}
        if isinstance(phase_classes, Mapping) and phase_classes:
            lines.extend(["", "Training labels by canonical class:"])
            lines.extend(
                f"- {name}: {int(count):,}"
                for name, count in sorted(phase_classes.items())
            )
        bundles = entry.get("training_bundles") or []
        lines.extend(["", f"Frozen evidence packages: {len(bundles)}"])
        errors = entry.get("errors") or []
        if errors:
            lines.extend(["", "Inspection findings:"])
            lines.extend(f"- {value}" for value in errors)
        self.model_history_details.setPlainText("\n".join(lines))

    def _open_selected_model_folder(self) -> None:
        entry = self._selected_model_history()
        if entry is None:
            return
        path = Path(str(entry.get("root") or "")).expanduser()
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _use_selected_history_model_for_validation(self) -> None:
        entry = self._selected_model_history()
        if entry is None:
            return
        index = self.validation_model.findData(str(entry.get("root") or ""))
        if index >= 0:
            self.validation_model.setCurrentIndex(index)
            self.tabs.setCurrentWidget(self.evidence_tab)

    def _selected_session(self) -> dict[str, object] | None:
        index = self.saved_sessions.currentIndex()
        if 0 <= index < len(self.visible_review_sessions):
            return self.visible_review_sessions[index]
        return None

    def _session_matches_selected_model(
        self, session: Mapping[str, object]
    ) -> bool:
        model = self._selected_validation_model()
        if model is None:
            return False
        model_id = str(model.get("model_id") or "").strip()
        explicit = str(session.get("target_model_id") or "").strip()
        if explicit:
            return explicit == model_id
        binding = model.get("validation_binding")
        if not isinstance(binding, Mapping):
            return False
        expected = dict(binding.get("selected_candidates") or {})
        actual = dict(session.get("selected_candidates") or {})
        overlap = set(expected).intersection(actual)
        return bool(overlap) and all(expected[run] == actual[run] for run in overlap)

    @staticmethod
    def _session_label(session: Mapping[str, object]) -> str:
        modified = (
            datetime.fromtimestamp(
                float(session.get("modified_timestamp") or 0.0), tz=UTC
            )
            .astimezone()
            .strftime("%Y-%m-%d %H:%M")
        )
        workspace = str(
            session.get("workspace_title") or session.get("workspace") or ""
        )
        model_id = str(session.get("target_model_id") or "Legacy model binding")
        state = "LOCKED" if session.get("editing_locked") else "Editable"
        return (
            f"{model_id} · {state} · {workspace} · "
            f"{int(session.get('manual_regions') or 0)} corrections · "
            f"{int(session.get('blind_region_labels') or 0)} blind labels · {modified}"
        )

    def _refresh_review_sessions(self) -> None:
        selected_path = str((self._selected_session() or {}).get("session_path") or "")
        self.review_sessions = [dict(row) for row in discover_phase_review_sessions()]
        self._rebuild_session_combo(selected_path=selected_path)
        if hasattr(self, "validation_model"):
            self._validation_model_changed()

    def _rebuild_session_combo(
        self,
        _index: int = -1,
        *,
        selected_path: str = "",
    ) -> None:
        if not selected_path:
            selected_path = str(
                (self._selected_session() or {}).get("session_path") or ""
            )
        show_all = str(self.session_scope.currentData() or "") == "all"
        self.visible_review_sessions = [
            session
            for session in self.review_sessions
            if show_all or self._session_matches_selected_model(session)
        ]
        self.saved_sessions.blockSignals(True)
        self.saved_sessions.clear()
        selected_index = 0
        for index, session in enumerate(self.visible_review_sessions):
            self.saved_sessions.addItem(self._session_label(session))
            if str(session.get("session_path") or "") == selected_path:
                selected_index = index
        self.saved_sessions.setCurrentIndex(
            selected_index if self.visible_review_sessions else -1
        )
        self.saved_sessions.blockSignals(False)
        self._session_changed()

    def _session_changed(self, _index: int = -1) -> None:
        session = self._selected_session()
        if session is None:
            self.session_summary.setText(
                "No saved session is bound to the selected model. Start a new "
                "session below; older sessions remain available under All historical sessions."
            )
            self._update_action_states()
            return
        locked = bool(session.get("editing_locked"))
        model_id = str(session.get("target_model_id") or "legacy model binding")
        trained_models = ", ".join(
            str(row.get("model_id") or "")
            for row in session.get("training_locks") or []
            if isinstance(row, Mapping) and str(row.get("model_id") or "")
        )
        self.session_summary.setText(
            f"Prediction reviewed: {model_id} · Reviewer: {session.get('reviewer')} · "
            f"{int(session.get('reviewed_experiments') or 0)} experiment(s) · "
            f"{int(session.get('manual_regions') or 0)} localized corrections · "
            f"{int(session.get('blind_region_labels') or 0)} blind region labels. "
            + (
                f"Frozen into model: {trained_models or 'recorded model version'}. "
                "Open it to audit every saved decision without changing the evidence."
                if locked
                else "Resume reopens this exact editable decision store."
            )
        )
        self.resume_session_button.setText(
            "Review locked evidence (read-only)"
            if locked
            else "Resume editable session"
        )
        self._update_review_button_labels()
        self._update_action_states()

    @staticmethod
    def _session_review_mode(session: Mapping[str, object]) -> str | None:
        surface = str(session.get("review_surface") or "").strip().casefold()
        workspace = str(session.get("workspace") or "").strip().casefold()
        if surface == "blind_regions" or workspace == "training":
            return "blind_region_labeling"
        if surface == "open_correction" or workspace in {
            "current_comparison",
            "comparison",
        }:
            return "open_correction"
        return None

    def _matching_review_session(self, mode: str) -> Mapping[str, object] | None:
        selected = self._selected_session()
        campaign_id = str((selected or {}).get("campaign_id") or "").casefold()
        reviewer = self.reviewer.text().strip().casefold()
        for session in self.review_sessions:
            if self._session_review_mode(session) != mode:
                continue
            if reviewer and str(session.get("reviewer") or "").casefold() != reviewer:
                continue
            if (
                campaign_id
                and str(session.get("campaign_id") or "").casefold() != campaign_id
            ):
                continue
            return session
        return None

    def _update_review_button_labels(self) -> None:
        busy = self.validation_process is not None
        suffix = " (preparing phase maps...)" if busy else ""
        self.open_blind_labeling.setText(
            "Start new blind phase-labeling session" + suffix
        )
        self.open_model_correction.setText(
            "Start new model-validation session" + suffix
        )

    def _update_action_states(self, _value: object = None) -> None:
        reviewer_ready = bool(self.reviewer.text().strip())
        review_paths_ready = bool(
            self.combined_campaign.text().strip()
            and self.review_save_root.text().strip()
        )
        model = self._selected_validation_model()
        review_ready = bool(
            reviewer_ready
            and review_paths_ready
            and model
            and self.validation_process is None
        )
        self.open_blind_labeling.setEnabled(review_ready)
        self.open_model_correction.setEnabled(review_ready)
        session = self._selected_session()
        self.resume_session_button.setEnabled(session is not None)
        self.add_session_evidence.setEnabled(
            bool(session and Path(str(session.get("decision_path") or "")).is_file())
        )
        training_ready = reviewer_ready and bool(self.bundles.toPlainText().strip())
        if self.radiometry_mode.currentData() in {"image", "custom"}:
            training_ready = training_ready and bool(self.radiometry_profile_path.text().strip())
        self.start_button.setEnabled(training_ready and self.training_process is None)
        self._update_review_button_labels()

    def _review_command(
        self,
        mode: str,
        *,
        binding_override: Mapping[str, object] | None = None,
        session_id: str = "",
    ) -> list[str]:
        model = self._selected_validation_model()
        binding = binding_override or (
            model.get("validation_binding")
            if isinstance(model, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            raise ValueError(
                "The selected model has no generated validation phase maps. "
                "Select a model marked Ready, or generate predictions for this model first."
            )
        model_id = str(model.get("model_id") or "").strip()
        resolved_session_id = session_id or (
            f"review-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        return build_phase_review_command(
            mode=mode,
            ffac_root=Path(str(binding["ffac_root"])),
            ffum_root=Path(str(binding["ffum_root"])),
            combined_root=Path(str(binding["combined_root"])),
            reviewer=self.reviewer.text().strip(),
            save_root=Path(self.review_save_root.text()),
            cache_root=default_execution_cache_root() / "phase-review",
            use_gpu=self.review_gpu.isChecked(),
            selected_candidates=dict(binding.get("selected_candidates") or {}),
            target_model_id=model_id,
            review_session_id=resolved_session_id,
        )

    def _launch_review(self, mode: str) -> None:
        model = self._selected_validation_model()
        binding = (
            model.get("validation_binding")
            if isinstance(model, Mapping)
            else None
        )
        if not isinstance(binding, Mapping):
            self._materialize_and_launch_review(mode)
            return
        self._start_review_process(mode, binding)

    def _start_review_process(
        self,
        mode: str,
        binding: Mapping[str, object],
        *,
        session_id: str = "",
    ) -> None:
        try:
            command = self._review_command(
                mode,
                binding_override=binding,
                session_id=session_id,
            )
            Path(self.review_save_root.text()).expanduser().resolve().mkdir(
                parents=True, exist_ok=True
            )
            started, process_id = QProcess.startDetached(command[0], command[1:])
            if not started:
                raise RuntimeError("The review application did not start")
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.review_status.setText(
                f"Could not open review: {type(exc).__name__}: {exc}"
            )
            QMessageBox.critical(self, "Could not open phase review", str(exc))
            return
        label = (
            "Blind phase labeling"
            if mode == "blind_region_labeling"
            else "Open model correction"
        )
        self.review_status.setText(
            f"{label} opened as process {int(process_id)}. Every edit is autosaved "
            "to the selected local review location."
        )

    def _materialize_and_launch_review(self, mode: str) -> None:
        model = self._selected_validation_model()
        if model is None:
            QMessageBox.warning(self, "Select a model", "Select a model to validate first.")
            return
        source_root = Path(self.combined_campaign.text()).expanduser()
        if not (source_root / "postflight_manifest.json").is_file():
            QMessageBox.warning(
                self,
                "Prepared evidence is unavailable",
                "Choose the prepared combined review package that contains the corrected "
                "FF_AC and FF_UM frames.",
            )
            return
        model_root = Path(str(model.get("root") or "")).expanduser().resolve()
        model_id = str(model.get("model_id") or model_root.name).strip()
        source_manifest = source_root / "postflight_manifest.json"
        source_key = sha256_file(source_manifest)[:12]
        output_root = (
            default_execution_cache_root()
            / "phase-validation"
            / f"{model_id}-{source_key}"
        )
        session_id = (
            f"review-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        command = [
            phase_python_executable(),
            "-m",
            "darsia.fluidflower.phase_validation",
            "--model-root",
            str(model_root),
            "--source-campaign-root",
            str(source_root.resolve()),
            "--output-root",
            str(output_root),
        ]
        process = QProcess(self)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._validation_materialization_output)
        process.readyReadStandardError.connect(self._validation_materialization_error)
        process.finished.connect(self._validation_materialization_finished)
        self.validation_process = process
        self.pending_validation_mode = mode
        self.pending_validation_session_id = session_id
        self.pending_validation_output = output_root
        self.review_status.setText(
            f"Generating {model_id} phase maps from corrected full-resolution frames..."
        )
        self._update_action_states()
        process.start()
        if not process.waitForStarted(5000):
            self.validation_process = None
            self._update_action_states()
            QMessageBox.critical(
                self,
                "Could not prepare validation",
                process.errorString() or "The phase-map process did not start.",
            )

    def _validation_materialization_output(self) -> None:
        process = self.validation_process
        if process is None:
            return
        lines = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        ).strip().splitlines()
        if lines:
            self.review_status.setText(f"Preparing selected model: {lines[-1]}")

    def _validation_materialization_error(self) -> None:
        process = self.validation_process
        if process is None:
            return
        message = bytes(process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        ).strip()
        if message:
            self.review_status.setText(f"Phase-map preparation: {message.splitlines()[-1]}")

    def _validation_materialization_finished(
        self, exit_code: int, _status: QProcess.ExitStatus
    ) -> None:
        process = self.validation_process
        output_root = self.pending_validation_output
        mode = self.pending_validation_mode
        session_id = self.pending_validation_session_id
        error = (
            bytes(process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            if process is not None
            else ""
        )
        self.validation_process = None
        self.pending_validation_mode = ""
        self.pending_validation_session_id = ""
        self.pending_validation_output = None
        self._update_action_states()
        if exit_code != 0 or output_root is None:
            message = error or f"Phase-map generation exited with code {exit_code}."
            self.review_status.setText(f"Could not prepare validation: {message}")
            QMessageBox.critical(self, "Could not prepare validation", message)
            return
        try:
            binding = json.loads(
                (output_root / "validation_binding.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            if not isinstance(binding, dict):
                raise TypeError("Validation binding is not a JSON object")
            model = self._selected_validation_model()
            if isinstance(model, dict):
                model["validation_binding"] = binding
            self.ffac_campaign.setText(str(binding["ffac_root"]))
            self.ffum_campaign.setText(str(binding["ffum_root"]))
            self.combined_campaign.setText(str(binding["combined_root"]))
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not open validation", str(exc))
            return
        self._start_review_process(mode, binding, session_id=session_id)

    def _resume_selected_session(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        try:
            command = build_phase_review_resume_command(
                session,
                cache_root=default_execution_cache_root() / "phase-review",
                use_gpu=self.review_gpu.isChecked(),
                read_only=bool(session.get("editing_locked")),
            )
            started, process_id = QProcess.startDetached(command[0], command[1:])
            if not started:
                raise RuntimeError("The saved review session did not start")
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.review_status.setText(
                f"Could not resume review: {type(exc).__name__}: {exc}"
            )
            QMessageBox.critical(self, "Could not resume saved review", str(exc))
            return
        state = (
            "Opened immutable evidence for audit"
            if session.get("editing_locked")
            else "Resumed editable session"
        )
        self.review_status.setText(
            f"{state} as process {int(process_id)}. Existing decisions remain in place."
        )

    def _add_selected_session_as_bundle(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        decisions = Path(str(session.get("decision_path") or ""))
        combined = Path(str(session.get("combined_root") or ""))
        manifest = combined / "postflight_manifest.json"
        if not decisions.is_file() or not manifest.is_file():
            QMessageBox.warning(
                self,
                "Evidence package incomplete",
                "The selected session needs both saved decisions and its combined "
                "postflight manifest before it can be used for training.",
            )
            return
        self._append_bundle_row(decisions, manifest)
        self.tabs.setCurrentWidget(self.training_tab)

    def _append_bundle_row(self, decisions: Path, manifest: Path) -> None:
        row = f"{decisions.resolve()} | {manifest.resolve()}"
        existing_rows = [
            value.strip()
            for value in self.bundles.toPlainText().splitlines()
            if value.strip()
        ]
        if row not in existing_rows:
            existing_rows.append(row)
            self.bundles.setPlainText("\n".join(existing_rows))

    def _add_bundle(self) -> None:
        decisions, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose saved review decisions",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not decisions:
            return
        manifest, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose matching combined campaign manifest",
            str(Path(decisions).parent),
            "JSON files (*.json);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if manifest:
            self._append_bundle_row(Path(decisions), Path(manifest))

    def _update_training_effect_summary(self, _value: object = None) -> None:
        backend = str(self.backend.currentData() or "auto")
        device = {
            "auto": "automatic CUDA/CPU selection",
            "cuda": "CUDA float32 fitting with CPU-float64 parity checks",
            "cpu": "CPU float64 reference fitting",
        }[backend]
        control = (
            "An additional no-change-evidence control will be fitted."
            if self.change_ablation.isChecked()
            else "No change-evidence control will be fitted."
        )
        self.training_effect_summary.setText(
            f"Current effect: {device}; up to {self.stage_workers.value()} runs are "
            f"prepared and {self.extraction_workers.value()} runs have features "
            f"extracted concurrently; at most {self.pixels.value():,} pixels represent "
            f"each reviewed region; mixed-region fitting may use "
            f"{self.mil_iterations.value()} iterations. {control}"
        )

    def _start_training(self) -> None:
        if self.training_process is not None:
            return
        try:
            bundles = parse_training_bundles(self.bundles.toPlainText())
            settings = {
                "radiometry_profile": (
                    ("image:" if self.radiometry_mode.currentData() == "image" else "") + self.radiometry_profile_path.text().strip()
                    if self.radiometry_mode.currentData() in {"custom", "image"}
                    else str(self.radiometry_mode.currentData())
                ),
                "series": [v.strip() for v in self.training_series.text().split(",") if v.strip()],
                "fit_mode": "mil",
                "backend": str(self.backend.currentData() or "auto"),
                "stage_workers": self.stage_workers.value(),
                "extraction_workers": self.extraction_workers.value(),
                "opencv_threads": 2,
                "pixels_per_region": self.pixels.value(),
                "mil_max_iters": self.mil_iterations.value(),
                "change_evidence_ablation": self.change_ablation.isChecked(),
            }
            if self.radiometry_mode.currentData() in {"custom", "image"} and not self.radiometry_profile_path.text().strip():
                raise ValueError("Select a common ColorChecker reference image or target profile first")
            request = prepare_phase_training_request(
                output_parent=Path(self.output_parent.text()),
                cache_parent=Path(self.cache_parent.text()),
                bundles=bundles,
                actor=self.reviewer.text().strip(),
                settings=settings,
            )
            command = build_phase_training_command(request)
            self._refresh_review_sessions()
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(self, "Could not start phase training", str(exc))
            return

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._training_finished)
        process.errorOccurred.connect(self._training_error)
        process.setProgram(command[0])
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(Path(request["request_path"]).parent))
        self.training_request = request
        self.training_process = process
        self.last_output_root = Path(request["output_root"])
        self.log.clear()
        self.status.setText(
            f"Training {request['model_id']} from {len(bundles)} frozen evidence "
            "package(s)..."
        )
        self.stop_button.setEnabled(True)
        self._update_action_states()
        process.start()

    def _read_output(self) -> None:
        process = self.training_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def _stop_training(self) -> None:
        if self.training_process is None:
            return
        self.status.setText(
            "Stopping after the current process checkpoint. Frozen inputs remain reusable."
        )
        self.training_process.terminate()

    def _training_error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(
            f"Training process error: {error.name}. Frozen inputs were retained."
        )

    def _training_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_output()
        request = dict(self.training_request or {})
        output_root = Path(str(request.get("output_root") or ""))
        process = self.training_process
        self.training_process = None
        self.stop_button.setEnabled(False)
        self._update_action_states()
        if process is not None:
            process.deleteLater()
        if int(exit_code) != 0:
            self.status.setText(
                f"Training stopped or failed with exit code {int(exit_code)}. "
                f"Frozen evidence remains in {output_root}."
            )
            return
        try:
            summary = inspect_phase_model_bundle(output_root)
            ablation_path = output_root / "change_evidence_ablation.json"
            ablation = (
                json.loads(ablation_path.read_text(encoding="utf-8"))
                if ablation_path.is_file()
                else {}
            )
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            self.status.setText(
                f"Training completed, but model inspection failed: {exc}"
            )
            return
        delta = ablation.get("mean_balanced_accuracy_delta")
        ablation_text = (
            ""
            if not isinstance(delta, (int, float))
            else f" Mean holdout benefit from DarSIA change evidence: {delta:+.3f}."
        )
        self.status.setText(
            f"Completed {request.get('model_id')} using "
            f"{len(summary.get('runs') or [])} source runs.{ablation_text}"
        )
        self.model_library.setText(str(output_root.parent))
        self._refresh_model_history()
        self.tabs.setCurrentWidget(self.history_tab)
        self.training_completed.emit(str(output_root))


__all__ = ["SharedPhaseModelTrainingPage"]
