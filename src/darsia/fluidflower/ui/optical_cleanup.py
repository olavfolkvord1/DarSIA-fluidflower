"""Opt-in native optical preview controls for the phase review desktop."""
from __future__ import annotations

from dataclasses import asdict
import json

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from darsia.fluidflower.phase_optical_cleanup import Settings


class _Signals(QObject):
    done = Signal(object, str)


class _Task(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.function(), "")
        except Exception as exc:
            self.signals.done.emit(None, str(exc))


class OpticalCleanup(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.token = None
        self.generation = 0
        self.tasks = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.enabled = QCheckBox("Optical cleanup preview")
        self.enabled.setToolTip("Experimental view from cached raw optical scores. Saved model maps and review scoring remain unchanged.")
        row.addWidget(self.enabled)
        self.sliders = {}
        self.values = {}
        for name, title, maximum, initial, scale in (
                ("threshold", "Threshold", 99, 50, 100),
                ("sigma_mm", "Smoothing σ", 30, 0, 10),
                ("hole_diameter_mm", "Hole diameter", 50, 0, 10)):
            row.addWidget(QLabel(title))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1 if name == "threshold" else 0, maximum)
            slider.setValue(initial)
            slider.setFixedWidth(105)
            row.addWidget(slider)
            value = QLabel()
            value.setMinimumWidth(48)
            row.addWidget(value)
            self.sliders[name] = (slider, scale)
            self.values[name] = value
            slider.valueChanged.connect(self.parameters_changed)
        self.sliders["threshold"][0].setToolTip("Lower = more aqueous support; higher = less. These are model scores, not calibrated concentration.")
        self.sliders["sigma_mm"][0].setToolTip("Gaussian standard deviation in millimetres. Zero leaves optical scores unsmoothed.")
        self.sliders["hole_diameter_mm"][0].setToolTip("Fill only enclosed holes within this physical diameter and area. Zero disables filling.")
        self.mode = QComboBox()
        for title, value in (("Processed optical", "processed"), ("Raw optical", "raw"), ("Changes only", "changes")):
            self.mode.addItem(title, value)
        row.addWidget(self.mode)
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.reset = QPushButton("Reset to raw")
        self.reset.clicked.connect(lambda: self.set_settings(Settings()))
        row.addWidget(self.reset)
        self.save = QPushButton("Save preview to project")
        self.save.clicked.connect(self.save_settings)
        row.addWidget(self.save)
        self.restore = QPushButton("Load from project")
        self.restore.clicked.connect(self.load_settings)
        row.addWidget(self.restore)
        row.addStretch(1)
        self.state_label = QLabel()
        layout.addWidget(self.state_label)
        self.status = QLabel("Settings are not applied to future models automatically.")
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        self.enabled.toggled.connect(self.changed)
        self.update_values()
        self.update_state()
        self.setVisible(bool(window.workspace.single_model))

    @property
    def regional_model(self):
        from darsia.fluidflower.phase_review_basis import is_regional
        pair = getattr(self.window, "current_pair_object", None)
        return pair is not None and is_regional(pair.candidate("A"))

    @property
    def active(self):
        return (not self.regional_model and self.enabled.isChecked() and self.window.current_pair is not None
                and not self.window.reference_mode
                and getattr(self.window, "comparison_mode", "") == "single_candidate")

    def settings(self):
        return Settings(**{name: slider.value() / scale for name, (slider, scale) in self.sliders.items()})

    def update_values(self):
        for name, value in asdict(self.settings()).items():
            self.values[name].setText(f"{value:.2f}" if name == "threshold" else f"{value:.1f} mm")

    def set_settings(self, settings):
        for name, value in asdict(settings).items():
            slider, scale = self.sliders[name]
            slider.blockSignals(True)
            slider.setValue(round(value * scale))
            slider.blockSignals(False)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(0)
        self.mode.blockSignals(False)
        self.parameters_changed()

    def parameters_changed(self, *ignored):
        # Moving an apparently usable slider must affect the displayed map.
        # Return from raw comparison to processed output when tuning a value.
        if self.mode.currentData() == "raw":
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
        self.enabled.blockSignals(True)
        self.enabled.setChecked(True)
        self.enabled.blockSignals(False)
        self.changed()

    def mode_changed(self, *ignored):
        self.enabled.blockSignals(True)
        self.enabled.setChecked(True)
        self.enabled.blockSignals(False)
        self.changed()

    def update_state(self):
        checked = self.enabled.isChecked()
        self.enabled.setText("Optical cleanup: " + ("ON" if checked else "OFF"))
        if not checked:
            text = "OFF · Showing the saved layer; cleanup settings are not applied. Move a slider to activate."
        elif getattr(self.window, "reference_mode", False):
            text = "PAUSED · Reference image; cleanup settings are not applied."
        elif self.mode.currentData() == "raw":
            text = "RAW COMPARISON · No cleanup applied. Moving a slider returns to processed optical."
        else:
            text = "ON · " + self.mode.currentText() + " · Threshold: lower = more area; higher = less."
        self.state_label.setText(text)
        self.state_label.setStyleSheet("color: #8fdfb0;" if checked else "color: #efc76e;")

    def changed(self, *ignored):
        self.update_values()
        self.update_state()
        self.generation += 1
        self.token = None
        # Invalidate old callbacks immediately, including while debounce waits.
        if self.window.current_pair is not None:
            self.window.overlay_keys = {"A": "", "B": ""}
            for viewer in self.window.viewers.values():
                viewer.set_overlay(None)
                viewer.set_small_detection_components(())
        self.timer.start()

    def refresh(self):
        if self.active and not self.window.contours.isChecked():
            self.window.contours.setChecked(True)
        self.window._request_overlays()

    def decorate(self):
        """Called after the standard legend so displayed provenance is explicit."""
        # The trained regional map is the saved review result. The earlier
        # experimental raw-score sliders must not replace it or its raw toggle.
        self.setVisible(bool(self.window.workspace.single_model) and not self.regional_model)
        self.update_state()
        self.window.layer_label.setText("Saved layer (inactive)" if self.active else "Layer")
        if self.active:
            self.window.contour_filter.setEnabled(False)
            self.window.highlight_small.setEnabled(False)
            self.window.contour_legend.setText(
                "RAW OPTICAL COMPARISON · cleanup is not applied to this view · move a slider to see processed optical"
                if self.mode.currentData() == "raw" else
                "EXPERIMENTAL optical preview · native resolution · gas fixed · cyan = added, orange = removed"
                if self.mode.currentData() == "changes" else
                "EXPERIMENTAL optical preview · native resolution · green = aqueous, pink = fixed raw gas · saved model unchanged")
        elif self.enabled.isChecked() and self.window.reference_mode:
            self.status.setText("Preview paused for reference image; the selected model's reference map is shown.")

    def request(self, index=None):
        w = self.window
        if not self.active:
            raise ValueError("Select a timeline image and enable the optical preview first")
        candidate = w.current_pair_object.candidate("A")
        return (str(w.current_review_root), w.current_run,
                json.dumps(candidate.provenance(), sort_keys=True), w.frame_index if index is None else index,
                self.settings(), str(self.mode.currentData()))

    def asset(self, deferred_type, index=None):
        from darsia.fluidflower.tools.phase_optical_cleanup_preview import resolve
        request = self.request(index)
        token = repr(request)
        if index is None or index == self.window.frame_index:
            self.token = token
        self.status.setText("Experimental preview · full native resolution · saved model and review scoring unchanged.")
        return deferred_type("optical-cleanup:" + token,
            lambda: resolve(*request, cancelled=lambda: self.token != token),
            normalize_overlay=True, prefetch_resolve=lambda: None)

    def _background(self, function, done):
        task = _Task(function)
        self.tasks.append(task)
        self.save.setEnabled(False)
        self.restore.setEnabled(False)
        def finished(result, error):
            self.tasks.remove(task)
            self.save.setEnabled(True)
            self.restore.setEnabled(True)
            if error:
                self.status.setText("Optical preview: " + error)
            else:
                done(result)
        task.signals.done.connect(finished)
        QThreadPool.globalInstance().start(task)

    def save_settings(self):
        from darsia.fluidflower.tools.phase_optical_cleanup_preview import resolve, read, save_profile
        try:
            request = self.request()
            series = str(self.window.current_pair["dataset"])
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Saving experimental settings and frame provenance to the series project folder…")
        # Save processed-result provenance even when the user is looking at raw.
        request = (*request[:-1], "processed")
        def work():
            overlay = resolve(*request)
            return save_profile(series, request[4], read(overlay.parent / "receipt.json"))
        self._background(work, lambda path: self.status.setText(f"Preview profile saved: {path}"))

    def load_settings(self):
        from darsia.fluidflower.tools.phase_optical_cleanup_preview import load_profile
        if self.window.current_pair is None:
            return
        series = str(self.window.current_pair["dataset"])
        generation = self.generation
        def done(result):
            if (self.window.current_pair is None or self.window.current_pair["dataset"] != series
                    or self.generation != generation):
                self.status.setText("Selection changed while loading; settings were not replaced.")
                return
            settings, path = result
            self.set_settings(settings)
            self.status.setText(f"Loaded experimental settings: {path}")
        self._background(lambda: load_profile(series), done)
