"""One painting run chooser; other runs are automatic, read-only previews."""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                              QMessageBox, QProgressBar, QPushButton, QStackedWidget, QTableWidget,
                              QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView)
from shiboken6 import isValid

from . import layer_painting as lp
from . import painting_run_preview as preview
from . import painting_frame_models as frame_models

_SETTINGS = ("DarSIA", "FluidFlower layer painting")
_INDEX = "placement/geometry_index"
_FIRST = "placement/first_runs"


def _plain_error(error):
    return str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"


class _PreviewWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    partial = Signal(object)

    def __init__(self, project, session, proposal, corrections, frame, options):
        super().__init__()
        self.project, self.session, self.proposal = project, session, proposal
        self.corrections, self.frame, self.options = corrections, frame, options
        self.result, self.error = None, ""

    @Slot()
    def run(self):
        try:
            frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
            self.progress.emit("Preparing the reference drawing")
            record = self.options.get("record")
            session = self.session
            if record and record.get("kind") == "drawing":
                from .layer_painting_version_archive import load_drawing_version
                session = load_drawing_version(self.project, session.run_id, record)
                record = None
                source = "drawing"
            else:
                source = self.options["source"]
            source_name = self.options["name"]
            if record is None and source == "manual":
                source_name = "Current manual map"
            elif record is None and source == "map" and self.proposal is not None:
                from . import layer_painting_polygons_only as polygons

                # Only the current-map convenience choice may use today's
                # drawing. Explicit saved versions retain their strict proof.
                if not polygons.proposal_current(self.proposal, session, self.frame):
                    source = "manual"
                    source_name = "Current manual map (the proposal is out of date)"
            session, layers = preview.preview_layers(session, self.proposal, self.corrections,
                                                     self.frame, source=source, record=record)
            def partial(value):
                frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
                self.partial.emit((value, session))

            result = preview.prepare_preview(session, layers,
                geometry_index=self.options["index"], target_run=self.options["run"],
                config=self.options["config"], measure_offsets=True,
                photo_loader=self.options["photo_loader"], source_name=source_name,
                on_progress=self.progress.emit,
                on_preview=partial)
            frame_models.verify_frame_model_selection(self.options.get("geometry_selection"))
        except Exception as error:  # reported in the preview, no write fallback
            self.error = _plain_error(error)
            self.failed.emit(self.error)
            return
        self.result = result, session
        self.finished.emit((result, session))


class PaintingRunPreview(QObject):
    display_changed = Signal()

    def __init__(self, view, editable_column, canvas_class, *, run_inputs=None, set_reference=None):
        super().__init__(view)
        self.view, self.run_inputs, self.set_reference = view, run_inputs, set_reference
        self.thread = None
        self.worker = None
        self._pending = False
        self._active_key = self._waiting_key = self._result_key = None
        self._disposed = False
        self._started = None
        self._stage = ""
        self._loading_status = False
        self._retry_attempts = 0
        self._joining = False
        self.last_result = None
        self._last_session = None
        settings = QSettings(*_SETTINGS)
        settings.sync()
        self._index = str(settings.value(_INDEX, "") or "")
        self._geometry_selection = None
        self._index_error = ""
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Run"))
        self.runs = QComboBox()
        self.runs.setToolTip("The reference run is editable; other runs show a read-only placed preview")
        self.previous = QPushButton("Previous")
        self.next = QPushButton("Next")
        self.reference_button = QPushButton("Set reference…")
        self.geometry_button = QPushButton("Frame models…")
        for widget in (self.runs, self.previous, self.next, self.reference_button, self.geometry_button):
            row.addWidget(widget)
        layout.addLayout(row)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Preview"))
        self.sources = QComboBox()
        self.sources.setToolTip("Preview an unfinished map, drawing or saved version; no approval is needed")
        source_row.addWidget(self.sources, 1)
        self.refresh_button = QPushButton("Refresh preview")
        source_row.addWidget(self.refresh_button)
        layout.addLayout(source_row)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.note)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.measurements = QTableWidget(0, 7)
        self.measurements.setHorizontalHeaderLabels(["Layers", "Median mm", "P10 mm", "P90 mm", "Δ reference mm", "Samples", "Status"])
        self.measurements.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.measurements.setMaximumHeight(180)
        self.measurements.hide()
        layout.addWidget(self.measurements)
        self.stack = QStackedWidget()
        self.stack.addWidget(editable_column)
        self.canvas = canvas_class()
        self.canvas.set_tool("pan")
        self.stack.addWidget(self.canvas)
        self._viewport = self.canvas.viewport()
        self.status_overlay = QWidget(self._viewport)
        self.status_overlay.setStyleSheet("background: #252525; color: white;")
        overlay_layout = QVBoxLayout(self.status_overlay)
        overlay_layout.addStretch(1)
        self.status_text = QLabel()
        self.status_text.setWordWrap(True)
        self.status_text.setTextFormat(Qt.TextFormat.PlainText)
        self.status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        overlay_layout.addWidget(self.status_text)
        overlay_layout.addStretch(1)
        self.status_overlay.hide()
        self._viewport.installEventFilter(self)
        self._viewport.destroyed.connect(self._invalidate)
        self.canvas.destroyed.connect(self._invalidate)
        view.destroyed.connect(self._invalidate)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._elapsed)
        self._retry_timer = QTimer(self)
        self._retry_timer.setSingleShot(True)
        self._retry_timer.setInterval(250)
        self._retry_timer.timeout.connect(self._retry_waiting)
        if hasattr(view, "busy_changed"):
            view.busy_changed.connect(self._parent_busy_changed)
        layout.addWidget(self.stack, 1)
        self.runs.currentIndexChanged.connect(self._selected)
        self.sources.currentIndexChanged.connect(self._source_changed)
        self.previous.clicked.connect(lambda: self._step(-1))
        self.next.clicked.connect(lambda: self._step(1))
        self.geometry_button.clicked.connect(self.choose_geometry)
        self.reference_button.clicked.connect(self.change_reference)
        self.refresh_button.clicked.connect(self.request)
        self.navigation_keys = []
        for key, direction in (("Alt+Left", -1), ("Alt+Right", 1)):
            shortcut = QShortcut(key, self.widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda step=direction: self._step(step))
            self.navigation_keys.append(shortcut)
        self.previous.setToolTip("Previous run (Alt+Left)")
        self.next.setToolTip("Next run (Alt+Right)")

    def busy(self):
        return self.thread is not None

    def _alive(self):
        return (not self._disposed and isValid(self) and isValid(self.view)
                and isValid(self.canvas) and isValid(self._viewport) and isValid(self.widget)
                and isValid(self.progress_bar))

    def _invalidate(self, *_args):
        self._disposed = True
        self._waiting_key = None
        self._last_session = None

    def _key(self):
        session = self.view.session
        source, record = self.sources.currentData() or ("map", None)
        version = None if record is None else (record.get("folder"), record.get("sha256"), record.get("kind"))
        return (id(self.view.project), id(session), getattr(session, "run_id", ""),
                getattr(session, "image_path", ""), tuple(getattr(session, "image_size", ()) or ()),
                getattr(session, "image_stamp", ""), getattr(session, "updated_utc", ""),
                id(getattr(self.view, "proposal", None)), str(self.runs.currentData() or "").casefold(),
                self._index, source, version, self._geometry_selection)

    def _accepts_result(self):
        return (self._alive() and not self.is_reference() and self._active_key == self._key()
                and not self._pending)

    def _status(self, message, *, loading=False):
        if not self._alive():
            return
        self._stage = str(message)
        self._loading_status = loading
        self.progress_bar.setVisible(loading and not self.is_reference())
        self._elapsed(force=True)
        shown = self.last_result is not None and self._result_key == self._key()
        self.status_overlay.setGeometry(self._viewport.rect())
        self.status_overlay.setVisible(not shown and not self.is_reference())
        self.status_overlay.raise_()
        if loading:
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.stop()

    def _elapsed(self, *, force=False):
        if not self._alive():
            if isValid(self._elapsed_timer):
                self._elapsed_timer.stop()
            return
        if not force and not self._loading_status:
            return
        elapsed = "" if self._started is None else f" · {int(time.monotonic()-self._started)} s"
        message = self._stage + elapsed
        self.note.setText(message)
        self.status_text.setText(message)

    def _cancel_waiting(self):
        self._waiting_key = None
        self._retry_timer.stop()

    def _parent_busy_changed(self, busy):
        if not busy and self._alive() and self._waiting_key is not None:
            self._retry_waiting()

    def _retry_waiting(self):
        if not self._alive():
            if isValid(self._retry_timer):
                self._retry_timer.stop()
            return
        if self.is_reference() or self._waiting_key != self._key():
            self._cancel_waiting()
            return
        if self.thread is not None:
            return
        if not self.view.busy():
            self._cancel_waiting()
            self.request()
        elif self._retry_attempts < 120:
            self._retry_attempts += 1
            self._retry_timer.start()
        else:
            self._status("The painting operation is still busy. The preview will resume when it finishes; "
                         "Refresh preview also retries.")

    def eventFilter(self, watched, event):
        if self._alive() and watched is self._viewport and event.type() == QEvent.Type.Resize:
            self.status_overlay.setGeometry(self._viewport.rect())
        return False

    def is_reference(self):
        session = self.view.session
        return session is None or str(self.runs.currentData() or "").casefold() == session.run_id.casefold()

    def ensure_saved_geometry_index(self):
        """Capture one series default or explicit USER override for all tools."""
        if not isValid(self) or self._disposed:
            return self._index
        if QThread.currentThread() != QObject.thread(self):
            raise RuntimeError("Frame-model settings must be read on the GUI thread.")
        current = self._geometry_selection
        override = self._index if self._index and (current is None
            or current.source == "user override" or current.index_path != self._index) else ""
        if not self._index or (current is not None and current.source == "series display binding"):
            settings = QSettings(*_SETTINGS)
            settings.sync()
            override = str(settings.value(_INDEX, "") or "")
        try:
            selected = frame_models.capture_frame_model_selection(self.view.project, override_path=override)
        except (AttributeError, TypeError, ValueError) as error:
            self._geometry_selection, self._index = None, ""
            self._index_error = str(error)
        else:
            self._geometry_selection, self._index = selected, selected.index_path
            self._index_error = ""
        return self._index

    def geometry_selection(self):
        self.ensure_saved_geometry_index()
        if self._geometry_selection is None:
            raise ValueError(self._index_error or "Frame models are unavailable. Retry after reopening the series.")
        return self._geometry_selection

    def refresh(self):
        self.ensure_saved_geometry_index()
        session = self.view.session
        reference = session.run_id if session else ""
        old = str(self.runs.currentData() or reference)
        try:
            runs = [row.run_id for row in self.run_inputs()] if self.run_inputs else self.view._series_runs()
        except (OSError, ValueError, KeyError):
            runs = []
        if reference and reference not in runs:
            runs.insert(0, reference)
        from .map_placement import review_order
        first = [part.strip() for part in str(QSettings(*_SETTINGS).value(_FIRST, "") or "").split(",") if part.strip()]
        runs = review_order(runs, reference, first)
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in runs:
            self.runs.addItem(run + (" — reference (editable)" if run.casefold() == reference.casefold() else " — read only"), run)
        self.runs.setCurrentIndex(max(0, self.runs.findData(old)))
        self.runs.blockSignals(False)
        self._refresh_sources()
        self._selected()

    def _refresh_sources(self):
        old = self.sources.currentData()
        self.sources.blockSignals(True)
        self.sources.clear()
        self.sources.addItem("Current map (drawing when needed)", ("map", None))
        self.sources.addItem("Current drawing", ("drawing", None))
        for record in reversed(self.view.versions):
            self.sources.addItem(str(record.get("display_name") or record.get("name") or "Saved version"),
                                 ("version", dict(record)))
        index = next((i for i in range(self.sources.count()) if self.sources.itemData(i) == old), 0)
        self.sources.setCurrentIndex(index)
        self.sources.blockSignals(False)

    def _step(self, direction):
        self.runs.setCurrentIndex(max(0, min(self.runs.count()-1, self.runs.currentIndex()+direction)))

    def _selected(self, *_args):
        if not self._alive():
            return
        reference = self.is_reference()
        self.stack.setCurrentIndex(0 if reference else 1)
        self.view.steps_scroll.setEnabled(reference)
        for shortcut in self.view.findChildren(QShortcut):
            if shortcut not in self.navigation_keys:
                shortcut.setEnabled(reference)
        self.measurements.setVisible(not reference and self.last_result is not None)
        self.reference_button.setEnabled(not reference and self.set_reference is not None and not self.view.busy())
        self.previous.setEnabled(self.runs.currentIndex() > 0)
        self.next.setEnabled(self.runs.currentIndex() < self.runs.count()-1)
        self.refresh_button.setEnabled(not reference)
        if reference:
            self._cancel_waiting()
            self._pending = False
            self._elapsed_timer.stop()
            self._loading_status = False
            self.progress_bar.hide()
            self.status_overlay.hide()
            self.note.setText("Reference run: editable. Viewing other runs keeps this drawing on its own photograph.")
        else:
            self.request()
        self.display_changed.emit()

    def _source_changed(self, *_args):
        if not self.is_reference():
            self.request()

    def choose_geometry(self):
        path, _ = QFileDialog.getOpenFileName(self.view, "Measured frame models (index.json)",
                                             self._index, "Geometry index (index.json);;JSON (*.json)")
        if path:
            self._index = path
            self._geometry_selection = frame_models.capture_frame_model_selection(self.view.project, override_path=path)
            self._index_error = ""
            settings = QSettings(*_SETTINGS)
            settings.setValue(_INDEX, path)
            settings.sync()
            self.request()
            self.display_changed.emit()

    def request(self):
        if not self._alive() or self.is_reference():
            return
        self.ensure_saved_geometry_index()
        key = self._key()
        if self.thread is not None:
            self._pending = key != self._active_key
            if self._pending:
                self._started = time.monotonic()
                self._status("Waiting for the previous preview before loading " + str(self.runs.currentData()), loading=True)
            else:
                self._status(self._stage or "This preview is already loading", loading=True)
            return
        self._started = time.monotonic()
        if self.view.busy():
            self._waiting_key = key
            self._retry_attempts = 0
            self._status("Waiting for the current painting operation before loading " + str(self.runs.currentData()),
                         loading=True)
            self._retry_timer.start()
            return
        self._cancel_waiting()
        self._pending = False
        self.canvas.clear_photo()
        self.last_result = None
        self._last_session = None
        self._result_key = None
        self.display_changed.emit()
        self.measurements.setRowCount(0)
        self.measurements.hide()
        if not self._index:
            self._status(self._index_error or "The series has no measured frame models. Use Frame models to set an override.")
            return
        session = self.view.session
        if session is None:
            self._status("Choose a reference run with a drawing first.")
            return
        try:
            source, record = self.sources.currentData() or ("map", None)
            if source == "map" and self.view._map_kind() == "manual map":
                source = "manual"
            options = {"run": str(self.runs.currentData()), "index": self._index,
                       "geometry_selection": self._geometry_selection,
                       "source": source, "record": deepcopy(record), "name": self.sources.currentText(),
                       "config": deepcopy(self.view._workspace_frame_config),
                       "photo_loader": self.view._reference_photo_loader()}
            worker = _PreviewWorker(self.view.project, deepcopy(session),
                                    self.view.proposal, deepcopy(self.view.corrections),
                                    deepcopy(self.view.current_frame()), options)
        except Exception as error:
            self._status(f"Preview unavailable: {_plain_error(error)}")
            return
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.partial.connect(self._partial)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._retired)
        self.thread, self.worker = thread, worker
        self._active_key = key
        self.sources.setEnabled(False)
        self.reference_button.setEnabled(False)
        self._status("Loading " + options["run"] + " preview… Read only; no files are written.", loading=True)
        thread.start()

    @Slot(str)
    def _progress(self, message):
        if self.worker is not None and self.sender() is self.worker and self._accepts_result():
            self._status(message, loading=True)

    @Slot(object)
    def _partial(self, value):
        if self.worker is not None and self.sender() is self.worker and self._accepts_result():
            self._finished(value, partial=True)

    @Slot(object)
    def _finished(self, value, *, partial=False):
        if not self._accepts_result():
            return
        try:
            result, session = value
            if (result.run_id != str(self.runs.currentData() or "").casefold()
                    or result.source_run != self.view.session.run_id.casefold()):
                raise ValueError("The preview result belongs to another selected photograph")
            self._render_result(result, session)
        except Exception as error:
            self._failed(f"Could not display the preview: {_plain_error(error)}")
            return
        self.last_result = result
        self._last_session = session
        self._result_key = self._active_key
        self.status_overlay.hide()
        if partial:
            self._status("Photo and placed map shown; checking boundary measurements…", loading=True)
        else:
            self._elapsed_timer.stop()
            self._loading_status = False
            self.progress_bar.hide()
        self.display_changed.emit()

    def _render_result(self, result, session):
        self.canvas.set_photo(result.photo, (result.photo.shape[1], result.photo.shape[0]),
                              token={"path": result.photo_path, "run_id": result.run_id}, reset_view=True)
        self._render_overlay(result, session)
        offsets = result.measurements.get("layer_offsets") or {}
        rows = offsets.get("boundaries") or []
        self.measurements.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (" / ".join(map(str, row["labels"])), row.get("median_mm"), row.get("p10_mm"),
                      row.get("p90_mm"), row.get("change_from_reference_mm"),
                      f"{row.get('measured_sample_count', 0)}/{row.get('sample_count', 0)}", row.get("status", ""))
            for column, value in enumerate(values):
                self.measurements.setItem(index, column, QTableWidgetItem("—" if value is None else str(value)))
        self.measurements.resizeColumnsToContents()
        self.measurements.setVisible(bool(rows))
        shift = result.measurements.get("offset_px") or {}
        checks = result.measurements.get("checks") or {}
        self.note.setText(f"{result.run_id}: {result.source_name} from {result.source_run}, placed through the saved frame models. "
                          "Read only · no approval, placement receipt or adoption is created. Magenta: unassigned or uncovered. "
                          f"Displacement: median {shift.get('median_px', '—')} px, maximum {shift.get('max_px', '—')} px. "
                          f"Uncovered: {checks.get('uncovered_pixels', '—')} px; water pockets: {checks.get('water_pocket_pixels', '—')} px. "
                          "Boundary offsets use the placement review's descriptive method.")

    def _render_overlay(self, result, session):
        """Recolour the cached categorical display without reloading its photo."""
        rgba = np.zeros((*result.layers.shape, 4), dtype=np.uint8)
        style = getattr(self.view, "display_layer_styles", None)
        if callable(style):
            colors, hidden = style(session)
        else:
            colors, hidden = {layer.layer_id: QColor(layer.color) for layer in session.layers}, frozenset()
        for layer in session.layers:
            if layer.layer_id in hidden:
                continue
            color = QColor(colors.get(layer.layer_id, layer.color))
            rgba[result.layers == layer.layer_id] = (color.red(), color.green(), color.blue(), 100)
        # A gap stays a gap, including edge-replicated pixels the source does
        # not reach. The overlay never turns those pixels into a decided layer.
        gaps = result.uncovered | (result.layers == lp.OUTSIDE)
        rgba[gaps] = (255, 0, 255, 95)
        self.canvas.set_proposal(rgba)
        self.canvas.set_overlays_visible(False, True)

    @Slot()
    def redraw_display(self):
        """Update only the current cached overlay; retain progress, errors and zoom."""
        if not self._alive() or self.is_reference() or self._pending:
            return
        result, session = self.last_result, self._last_session
        if result is None or session is None or self._result_key != self._key():
            return
        if (result.run_id != str(self.runs.currentData() or "").casefold()
                or self.view.session is None or result.source_run != self.view.session.run_id.casefold()):
            return
        try:
            self._render_overlay(result, session)
        except Exception as error:
            self._status(self.note.text() + f" Display colours unavailable: {_plain_error(error)}",
                         loading=self._loading_status)

    @Slot(str)
    def _failed(self, message):
        if self._accepts_result():
            retained = self.last_result is not None and self._result_key == self._active_key
            self._status(("Photo and placed map retained; preview update unavailable: " if retained
                          else "Preview unavailable: ") + message)
            self.display_changed.emit()

    @Slot()
    def _retired(self):
        self._retire(self.sender())

    def _retire(self, thread):
        if thread is None or thread is not self.thread:
            return
        if thread is not None:
            thread.wait()
        worker = self.worker
        if self._accepts_result():
            if worker.error:
                self._failed(worker.error)
            elif worker.result is not None:
                self._finished(worker.result)
            else:
                self._failed("The preview worker stopped without a result.")
        # Keep the worker alive until its thread has finished, then drop its
        # last owned reference here in the GUI thread (the shared worker rule).
        self.thread = self.worker = None
        del worker
        if thread is not None and isValid(thread):
            thread.deleteLater()
        if not self._alive():
            return
        self._elapsed_timer.stop()
        self._loading_status = False
        self.progress_bar.hide()
        self.sources.setEnabled(True)
        self.reference_button.setEnabled(not self.is_reference() and self.set_reference is not None)
        if not self._joining and (self._pending or (not self.is_reference() and self._active_key != self._key())):
            self._pending = False
            self.request()
        if not self._joining:
            self.display_changed.emit()

    def wait(self):
        self._cancel_waiting()
        self._pending = False
        self._active_key = None
        self._elapsed_timer.stop()
        self._loading_status = False
        if isValid(self.progress_bar):
            self.progress_bar.hide()
        self._joining = True
        try:
            if self.thread is not None:
                self._retire(self.thread)
        finally:
            self._joining = False

    def change_reference(self):
        if self.is_reference() or self.view.busy() or self.busy() or self.set_reference is None:
            return
        old = self.view.session.run_id if self.view.session else ""
        target = str(self.runs.currentData() or "")
        answer = QMessageBox.question(self.view, "Set reference run",
            f"Change the reference from {old} to {target}?\n\nThe drawing on {old} stays on that run's photograph. "
            f"The drawing belonging to {target} will open; no marks are copied between photographs.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self.set_reference(target)
