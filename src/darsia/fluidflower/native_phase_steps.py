"""Native-only step layout around the retained Studio widgets and callbacks.

This adapter never installs a training algorithm. Missing compatible training
routes stay unavailable instead of falling back to a different optical chain.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import uuid

from PySide6.QtCore import QSignalBlocker, Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def saved_method_text(selection):
    """Literal saved method and references; no discovery or inferred choice."""
    if selection.error:
        return "Saved method unavailable — " + selection.error
    if selection.active == "phase_model":
        return ("phase_model · " + (selection.model_version or "model not referenced")
                + " · " + (selection.mass_calibration or "calibration not referenced"))
    if selection.active == "colour_path":
        return "colour_path · " + (selection.colour_calibration or "calibration not referenced")
    return "Method: not chosen"


class SessionSettings:
    """Keep this native instance's UI preferences in memory, including binding."""

    def __init__(self, original):
        self.original, self.values = original, {}

    def value(self, key, default=None, *args, **kwargs):
        return (
            self.values[key]
            if key in self.values
            else self.original.value(key, default, *args, **kwargs)
        )

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        pass


def save_model_ui_name(request, name, actor):
    """Bind one display-only name to the original newly prepared request.

    This does not edit the scientific request, model ID, settings or summary.
    It is called only immediately before an explicitly requested job launch.
    """
    name = str(name).strip()
    if not name:
        raise ValueError("Name the new model version before training.")
    root = Path(request["output_root"])
    source = Path(request["request_path"])
    raw = source.read_bytes()
    saved = json.loads(raw)
    model_id = str(request.get("model_id") or "")
    if (
        not model_id
        or saved.get("model_id") != model_id
        or Path(saved.get("output_root", "")).resolve() != root.resolve()
        or source.resolve() != (root / "training_request.json").resolve()
    ):
        raise ValueError(
            "The display name must belong to the actual prepared training request."
        )
    payload = dict(
        format="phase-model-ui-name-v1",
        model_id=model_id,
        training_request_sha256=hashlib.sha256(raw).hexdigest(),
        display_name=name,
        recorded_by=actor,
        recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    target = root / "model_ui_name.json"
    if target.exists():
        raise ValueError("This model version already has a saved display name.")
    temporary = root / f".model-ui-name-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
        if target.exists():
            raise ValueError(
                "The display name appeared while saving; it was preserved."
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def saved_model_ui_name(entry):
    from darsia.fluidflower.model_index_names import model_ui_display_name

    return model_ui_display_name(entry)


def _move(widget, layout):
    widget.setParent(layout.parentWidget())
    layout.addWidget(widget)
    widget.show()


def _hide_form_field(widget):
    widget.hide()
    candidates = [widget]

    def hide_in(layout):
        if layout is None:
            return
        if isinstance(layout, QFormLayout):
            for candidate in candidates:
                label = layout.labelForField(candidate)
                if label is not None:
                    layout.setRowVisible(candidate, False)
                    return
        for index in range(layout.count()):
            child = layout.itemAt(index).layout()
            if child is not None:
                hide_in(child)

    parent = widget.parentWidget()
    while parent is not None:
        hide_in(parent.layout())
        candidates.append(parent)
        parent = parent.parentWidget()


def import_saved_model(series_root, series_id, folder):
    """Use existing inspection/filter/index APIs; never copy model assets."""
    from darsia.fluidflower.phase_model_history import inspect_phase_model_history_entry
    from darsia.fluidflower.ui.model_training import _models_for_series
    from darsia.fluidflower.model_index_names import index_model_with_ui_name

    entry = inspect_phase_model_history_entry(Path(folder))
    if not series_id or not _models_for_series([entry], series_id):
        raise ValueError(
            "The selected model does not belong to this experimental series."
        )
    return index_model_with_ui_name(series_root, Path(folder))


class NativePhaseSteps(QWidget):
    """The Calibration pages around the retained page's actual objects.

    One section shows at a time, chosen by its sidebar row (N141): Image
    normalisation, Phase model (tabs Basis, Review and label, Train) and Use
    for this series. ``groups`` and ``layouts`` keep the former step numbers
    as keys (1, 2-4 and 6); the saved mass values of the former step 5 are the
    Mass calibration row's own page.
    """

    SECTIONS = ("normalisation", "phase_model", "use")

    def __init__(
        self,
        studio,
        parent=None,
        *,
        normalisation=None,
        method_panel=None,
        method_use=None,
        phase_review=None,
        actor_provider=getpass.getuser,
        training_routes=None,
    ):
        super().__init__(parent)
        self.studio, self.series_root, self.series_id = studio, None, ""
        studio._native_step_adapter = self
        self.actor_provider = actor_provider
        self.training_routes = dict(training_routes or {})
        self.normalisation = normalisation
        self.method_panel, self.method_use = method_panel, method_use
        self.phase_review = phase_review
        self._ready = False
        self._native_idle_status = None
        self._legacy_start_available = studio.start_button.isEnabled()
        self._original_start = studio._start_training
        self._original_update = studio._update_action_states
        self._original_launch = studio._launch_training_job
        self._original_set_series = studio.set_model_series
        self._original_project = studio.set_project_root
        self._original_review = studio._launch_review
        self._legacy_availability = {
            name: getattr(studio, name).isEnabled()
            for name in ("open_model_correction", "resume_session_button")
        }
        self._guard_slots = {}
        self._named_model_context = {}
        self._install_instance_policy()
        self.groups, self.layouts = {}, {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        # The page's only scroll area; no section holds another one.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        content = QWidget()
        page = QVBoxLayout(content)
        self.groups[1] = QGroupBox("Image normalisation")
        self.phase_tabs = QTabWidget()
        for number, title in ((2, "Basis"), (3, "Review and label"), (4, "Train")):
            self.groups[number] = QWidget()
            self.phase_tabs.addTab(self.groups[number], title)
        self.groups[6] = QGroupBox("Use for this series")
        self.groups[6].setObjectName("nativePhaseUseStep")
        for number, group in self.groups.items():
            self.layouts[number] = QVBoxLayout(group)
            if number != 1:
                # Short pages keep their controls together at the top; the
                # normalisation table takes the height it is given.
                self.layouts[number].setAlignment(Qt.AlignmentFlag.AlignTop)
        self.sections = {
            "normalisation": self.groups[1],
            "phase_model": self.phase_tabs,
            "use": self.groups[6],
        }
        for section in self.sections.values():
            page.addWidget(section, 1)
        self.scroll.setWidget(content)
        outer.addWidget(self.scroll)
        self.section = None
        self.show_section("phase_model")
        self.normalisation_notice = QLabel(
            "Not done — the saved normalisation status step is not attached."
        )
        self.normalisation_notice.setWordWrap(True)
        self.layouts[1].addWidget(self.normalisation_notice)
        if normalisation is not None:
            self.attach_normalisation(normalisation)
        self._build_basis()
        self._build_review()
        self.review_selection_notice = QLabel(
            "Choose a review collection using the existing review controls. "
            "No collection is selected automatically."
        )
        self.review_selection_notice.setWordWrap(True)
        self.layouts[3].insertWidget(0, self.review_selection_notice)
        self._build_training()
        self.saved_method_label = QLabel("Method: not chosen")
        self.saved_method_label.setTextFormat(Qt.TextFormat.PlainText)
        self.saved_method_label.setWordWrap(True)
        self.layouts[6].addWidget(self.saved_method_label)
        if method_panel is None:
            self.layouts[6].addWidget(
                QLabel("The series method panel is not attached.")
            )
        else:
            _move(method_panel, self.layouts[6])
            original_controls = method_panel._update_controls

            def method_controls(*args):
                original_controls(*args)
                if not self._ready and method_panel.selection().active != "colour_path":
                    method_panel.use_button.setEnabled(False)

            method_panel._update_controls = method_controls
            for field in method_panel.findChildren(QLineEdit):
                field.textChanged.connect(method_controls)
            for button in method_panel.findChildren(QRadioButton):
                button.toggled.connect(method_controls)
            if method_use is not None:
                method_panel.use_requested.disconnect(method_use)
                method_panel.use_requested.connect(self._use_method)
        studio.hide()
        self.refresh()

    def _install_instance_policy(self):
        studio = self.studio
        if not isinstance(studio._model_filter_settings, SessionSettings):
            studio._model_filter_settings = SessionSettings(
                studio._model_filter_settings
            )
        # Identity is taken silently from the logged-in user, never persisted as
        # a reviewer preference merely because a native page was shown.
        studio._save_reviewer_preference = lambda: None
        studio._queue_reviewer_save = lambda: None
        studio._flush_reviewer_save = lambda: None
        studio._start_training = self.train
        studio._launch_training_job = self._launch_named_job

        def update(*args, **kwargs):
            result = self._original_update(*args, **kwargs)
            self._legacy_start_available = studio.start_button.isEnabled()
            self._legacy_availability = {
                name: getattr(studio, name).isEnabled()
                for name in ("open_model_correction", "resume_session_button")
            }
            if hasattr(self, "model_name"):
                self.refresh()
            return result

        studio._update_action_states = update
        studio._launch_review = lambda *a, **k: self._guarded(
            self._original_review, *a, **k
        )
        for name in (
            "_resume_selected_session",
            "_open_selected_experiment",
            "_start_series_phase_batch",
            "_open_transfer_comparison",
        ):
            original = getattr(studio, name, None)
            if callable(original):
                self._guard_slots[name] = original
                setattr(
                    studio,
                    name,
                    lambda *a, _original=original, **k: self._guarded(
                        _original, *a, **k
                    ),
                )
        for button_name, slot_name in (
            ("resume_session_button", "_resume_selected_session"),
            ("test_transfer_button", "_open_selected_experiment"),
            ("transfer_comparison_button", "_open_transfer_comparison"),
        ):
            button = getattr(studio, button_name, None)
            original = self._guard_slots.get(slot_name)
            if button is not None and original is not None:
                button.clicked.disconnect(original)
                button.clicked.connect(getattr(studio, slot_name))
        original_label = studio._validation_model_label

        def label(entry):
            name = saved_model_ui_name(entry)
            return (name + " · " if name else "") + original_label(entry)

        studio._validation_model_label = label
        original_version_text = getattr(studio, "_model_version_text", None)
        if callable(original_version_text):
            def version_text(entry):
                name = saved_model_ui_name(entry)
                return (name + " \u00b7 " if name else "") + original_version_text(entry)

            studio._model_version_text = version_text
        original_completed = getattr(studio, "_record_completed_model", None)
        if callable(original_completed):
            def completed(model_id, root):
                error = None
                context = self._named_model_context.get(str(model_id))
                if context is not None:
                    try:
                        from darsia.fluidflower.model_index_names import index_model_with_ui_name

                        series_root, output_root = context
                        if Path(root).resolve() != output_root:
                            raise ValueError("Completed model folder differs from its named training request")
                        index_model_with_ui_name(series_root, root, expected_model_id=str(model_id))
                    except (OSError, ValueError, TypeError, RuntimeError) as caught:
                        error = caught
                result = original_completed(model_id, root)
                if error is not None:
                    studio.status.setText("Model completed; its display name could not be indexed: " + str(error))
                return result

            studio._record_completed_model = completed

    def _build_basis(self):
        studio = self.studio
        row = QHBoxLayout()
        self.existing_basis = QRadioButton("Build on an existing model")
        self.scratch_basis = QRadioButton("Start from scratch")
        self.existing_basis.setChecked(True)
        row.addWidget(self.existing_basis)
        row.addWidget(self.scratch_basis)
        self.layouts[2].addLayout(row)
        _move(studio.validation_model, self.layouts[2])
        _move(studio.catalog_loading_status, self.layouts[2])
        self.import_button = QPushButton("Import existing model…")
        self.import_button.clicked.connect(self._import_model)
        self.layouts[2].addWidget(self.import_button)
        _move(studio.validation_model_summary, self.layouts[2])
        self.basis_note = QLabel()
        self.basis_note.setWordWrap(True)
        self.layouts[2].addWidget(self.basis_note)
        self.existing_basis.toggled.connect(self.refresh)
        self.scratch_basis.toggled.connect(self.refresh)
        studio.validation_model.currentIndexChanged.connect(self.refresh)

    def _build_review(self):
        self.studio.open_model_correction.setText("Review and label…")
        row = QHBoxLayout()
        self.layouts[3].addLayout(row)
        _move(self.studio.open_model_correction, row)
        # The former Analysis row "Phase review window" (mass_analysis:review)
        # sits beside the review it opens; its row id stays routable (N141).
        self.phase_review_button = QPushButton("Phase review window")
        self.phase_review_button.setToolTip(
            "Open the existing prepared model-correction review window; a loaded model with bound phase maps is required."
        )
        self.phase_review_button.setEnabled(self.phase_review is not None)
        if self.phase_review is not None:
            self.phase_review_button.clicked.connect(lambda: self.phase_review())
        row.addWidget(self.phase_review_button)
        row.addStretch(1)
        _move(self.studio.review_status, self.layouts[3])
        self.layouts[3].addWidget(
            QLabel(
                "Saved model maps and drawing tools open in the established review window. No map is recalculated by this page layout."
            )
        )

    def _build_training(self):
        studio = self.studio
        annotators = getattr(studio, "annotator_selection", None)
        if isinstance(annotators, QWidget):
            _move(annotators, self.layouts[4])
        _move(studio.saved_sessions, self.layouts[4])
        _move(studio.session_summary, self.layouts[4])
        _move(studio.resume_session_button, self.layouts[4])
        _move(studio.add_session_evidence, self.layouts[4])
        _move(studio.pending_training_sessions, self.layouts[4])
        self.layouts[4].addWidget(QLabel("Name the new model version"))
        self.model_name = QLineEdit()
        self.model_name.setPlaceholderText("Required before training")
        self.model_name.textChanged.connect(self.refresh)
        self.layouts[4].addWidget(self.model_name)
        self.training_note = QLabel()
        self.training_note.setWordWrap(True)
        self.layouts[4].addWidget(self.training_note)
        self.advanced = QGroupBox("Existing training settings and evidence inputs")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        advanced = QVBoxLayout(self.advanced)
        original_scroll = studio.training_tab
        original_content = (
            original_scroll.takeWidget()
            if isinstance(original_scroll, QScrollArea)
            else None
        )
        if original_content is not None:
            _move(original_content, advanced)
            original_content.hide()
            self.advanced.toggled.connect(original_content.setVisible)
        self.layouts[4].addWidget(self.advanced)
        for name in (
            "reviewer",
            "reviewer_project",
            "model_series_filter",
            "history_series_filter",
            "training_series",
            "training_scope",
            "output_parent",
            "cache_parent",
            "model_library",
            "ffac_campaign",
            "ffum_campaign",
            "combined_campaign",
            "review_save_root",
        ):
            field = getattr(studio, name, None)
            if isinstance(field, QWidget):
                _hide_form_field(field)
        for name in (
            "train_further_button",
            "compare_phase_models_button",
            "start_existing_model",
            "start_new_model",
        ):
            widget = getattr(studio, name, None)
            if isinstance(widget, QWidget):
                widget.hide()
        studio.start_button.clicked.disconnect(self._original_start)
        studio.start_button.clicked.connect(self.train)
        studio.start_button.setText("Train")
        _move(studio.start_button, self.layouts[4])
        _move(studio.stop_button, self.layouts[4])
        _move(studio.status, self.layouts[4])
        _move(studio.log, advanced)
        studio.log.hide()
        self.advanced.toggled.connect(studio.log.setVisible)
        for timer_name in ("_retraction_poll_timer",):
            timer = getattr(studio, timer_name, None)
            if timer is not None:
                timer.timeout.connect(self.refresh)

    def show_section(self, name):
        """Show the section of one sidebar row; the others stay built and hidden."""
        if name not in self.sections:
            raise ValueError(f"Unknown phase-model page section: {name}")
        self.section = name
        for key, widget in self.sections.items():
            widget.setVisible(key == name)
        self.scroll.verticalScrollBar().setValue(0)

    def attach_normalisation(self, widget):
        if self.normalisation is not None and self.normalisation is not widget:
            self.layouts[1].removeWidget(self.normalisation)
            self.normalisation.hide()
        self.normalisation = widget
        self.normalisation_notice.hide()
        _move(widget, self.layouts[1])
        widget.readiness_changed.connect(self.set_normalisation_ready)
        self.set_normalisation_ready(getattr(widget, "ready", False))

    def set_normalisation_ready(self, ready):
        self._ready = bool(ready)
        if hasattr(self, "model_name"):
            self.refresh()

    def set_context(self, series_root, series_id):
        self.series_root = Path(series_root) if series_root else None
        self.series_id = str(series_id or "")
        studio = self.studio
        for timer_name in ("_reviewer_save_timer", "_project_root_timer"):
            timer = getattr(studio, timer_name, None)
            if timer is not None:
                timer.stop()
        studio._reviewer_project_root = None
        with QSignalBlocker(studio.reviewer), QSignalBlocker(studio.reviewer_project):
            studio.reviewer.setText(str(self.actor_provider()))
            studio.reviewer_project.setText(str(self.series_root or ""))
        canonical = self.series_id.casefold().replace("_", "").replace("-", "")
        if self.series_root is not None:
            studio._configured_series_roots[canonical] = self.series_root
        self._original_set_series(self.series_id)
        with QSignalBlocker(studio.training_series):
            studio.training_series.setText(canonical)
        studio._sync_default_model_libraries()
        from .series_method import read_series_method

        choice = read_series_method(self.series_root) if self.series_root is not None else None
        model = choice.model_version if choice is not None and not choice.error else ""
        if model:
            model = Path(model)
            model = model if model.is_absolute() else self.series_root / model
        context_key = (str(self.series_root or ""), os.environ.get("DARSIA_NATIVE_CONFIGURATION_SHA256"))
        studio.set_saved_model_preference(model or None, context_key=context_key)
        if self.normalisation is not None:
            self.normalisation.set_context(self.series_root, self.series_id)
        else:
            self.set_normalisation_ready(False)
        self.model_name.clear()
        self._original_update()
        self.refresh()

    def refresh(self, *_args):
        if not hasattr(self, "groups") or not hasattr(self, "model_name"):
            return
        from darsia.fluidflower.series_method import MethodSelection, read_series_method

        saved = read_series_method(self.series_root) if self.series_root else MethodSelection()
        self.saved_method_label.setText(saved_method_text(saved))
        photometry = getattr(self, "training_photometry", None)
        if photometry is not None:
            photometry.refresh()
        self.studio.open_model_correction.setText("Review and label…")
        # Saved versions, sessions and calibration values remain inspectable
        # while only the dependent actions are gated.
        for number, group in self.groups.items():
            if number != 1:
                group.setEnabled(True)
        for name, available in self._legacy_availability.items():
            getattr(self.studio, name).setEnabled(self._ready and available)
        self.import_button.setEnabled(self._ready)
        if self.method_panel is not None:
            self.method_panel._update_controls()
        mode = "existing" if self.existing_basis.isChecked() else "scratch"
        self.studio.validation_model.setEnabled(mode == "existing")
        route = self.training_routes.get(mode)
        if mode == "existing":
            reason = "The selected model is the basis for review. A compatible existing-model training route is not available in the retained code; no fresh fit is substituted."
        else:
            reason = "The retained fresh-training route uses its recorded radiometry settings. A saved v5 status report does not configure that route; compatible v5 training is not attached."
        if mode == "scratch" and photometry is not None:
            reason = ("Fresh training uses the saved v5 input and the original fitting procedure."
                      if photometry.binding is not None else photometry.reason)
        self.basis_note.setText(reason)
        self.training_note.setText(
            reason
            if not callable(route)
            else "Training reuses the admitted original callback and requires a new version name."
        )
        training_available = self._legacy_start_available
        if mode == "scratch" and photometry is not None:
            training_available = photometry.fresh_ui_ready()
        name_ready = bool(self.model_name.text().strip())
        idle = getattr(self.studio, "training_process", None) is None and not getattr(
            self.studio, "_series_batch_active", False
        )
        self.studio.start_button.setEnabled(
            self._ready
            and name_ready
            and idle
            and callable(route)
            and training_available
        )
        self.model_name.setEnabled(idle)
        if idle and self.studio.status.text() in {
            "",
            "Add reviewed evidence and enter a researcher name.",
            self._native_idle_status,
        }:
            self._native_idle_status = (
                "Not done — complete and save Image normalisation first. "
                "Saved models and sessions remain available for inspection."
                if not self._ready
                else self.training_note.text()
            )
            self.studio.status.setText(self._native_idle_status)

    def _guarded(self, callback, *args, **kwargs):
        if not self._ready:
            self.studio.review_status.setText(
                "Complete and save the normalisation status before continuing."
            )
            return None
        return callback(*args, **kwargs)

    def train(self, *_args):
        if not self._ready:
            self.studio.status.setText(
                "Complete and save the normalisation status before training."
            )
            return
        if not self.model_name.text().strip():
            self.studio.status.setText("Name the new model version before training.")
            return
        mode = "existing" if self.existing_basis.isChecked() else "scratch"
        callback = self.training_routes.get(mode)
        if not callable(callback):
            self.studio.status.setText(self.training_note.text())
            return
        if getattr(self.studio, "training_process", None) is not None or getattr(
            self.studio, "_series_batch_active", False
        ):
            return
        try:
            return callback()
        except Exception as error:
            self.studio.status.setText(
                f"Training did not start: {type(error).__name__}: {error}"
            )
            QMessageBox.critical(self, "Training did not start", str(error))
            return None

    def _launch_named_job(self, request, command, working_directory, status):
        if not self._ready or not self.model_name.text().strip():
            raise ValueError(
                "Saved normalisation and a new model version name are required before launch."
            )
        save_model_ui_name(
            request, self.model_name.text(), self.studio.reviewer.text().strip()
        )
        if self.series_root is not None:
            self._named_model_context[str(request["model_id"])] = (
                Path(self.series_root), Path(request["output_root"]).resolve()
            )
        result = self._original_launch(request, command, working_directory, status)
        self.model_name.clear()
        self.refresh()
        return result

    def _use_method(self, selection):
        if self.method_use is not None and (
            self._ready or getattr(selection, "active", "") == "colour_path"
        ):
            return self.method_use(selection)

    def _import_model(self):
        if not self._ready or self.series_root is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Import a saved phase-model version"
        )
        if not folder:
            return
        try:
            import_saved_model(self.series_root, self.series_id, Path(folder))
            self.studio._request_async_model_refresh()
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "Model was not imported", str(error))
