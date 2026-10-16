"""Explicit local store Settings and passive Jobs availability presentation.

Opening or changing series performs no filesystem/availability query. Load,
Check and Save are explicit finite metadata operations off the UI event thread.
No job, queue, worker, probe, input hash or result publication is started here.
"""
from __future__ import annotations

from pathlib import Path
import json
import traceback

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from .store_map import (
    LIMIT, default_store_map_path, parse_store_map, propose_manifest_roots,
    read_store_map, root_availability, save_store_map_text,
)


class _MetadataCall(QThread):
    def __init__(self, callback, parent):
        super().__init__(parent)
        self.callback, self.value, self.error = callback, None, ""

    def run(self):
        try:
            self.value = self.callback()
        except Exception:
            self.error = traceback.format_exc()


def _label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _small_bytes(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("Settings/manifest document exceeds 1 MiB")
    return raw


class StoreSettingsDialog(QDialog):
    saved = Signal(str)

    def __init__(self, series_root=None, parent=None, *, path=None,
                 reader=read_store_map, writer=save_store_map_text):
        super().__init__(parent)
        self.setWindowTitle("Machine data stores — Settings")
        self.setModal(True)
        self.resize(850, 650)
        self.path = Path(path) if path is not None else default_store_map_path()
        self.series_root = Path(series_root) if series_root else None
        self.reader, self.writer = reader, writer
        self._thread = None
        self._expected = None
        self._loaded = False
        self._displayed_text = ""
        layout = QVBoxLayout(self)
        layout.addWidget(_label("This machine's store map: " + str(self.path)))
        layout.addWidget(_label("Edit local input/output locations here. Jobs and series manifests are not changed. A local filesystem must be positively identified before loading or saving."))
        actions = QHBoxLayout()
        self.load_button = QPushButton("Load existing stores.toml")
        self.new_button = QPushButton("New local store map")
        self.save_button = QPushButton("Save local settings")
        self.roots_button = QPushButton("Preview series root migration…")
        for button in (self.load_button, self.new_button, self.save_button, self.roots_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText('Load the existing map, or choose New and enter [machine] and [stores.<series>] with actual locations.')
        layout.addWidget(self.editor, 1)
        self.status = _label("Nothing loaded. No filesystem check or write has run.")
        layout.addWidget(self.status)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(180)
        self.preview.hide()
        layout.addWidget(self.preview)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        self.load_button.clicked.connect(self.load)
        self.new_button.clicked.connect(self.new)
        self.save_button.clicked.connect(self.save)
        self.roots_button.clicked.connect(self.preview_roots)
        self.editor.textChanged.connect(self._controls)
        self._controls()

    def _controls(self):
        idle = self._thread is None
        self.load_button.setEnabled(idle)
        self.new_button.setEnabled(idle)
        self.roots_button.setEnabled(idle and self.series_root is not None)
        self.editor.setReadOnly(not idle)
        valid = False
        if self._loaded:
            try:
                parse_store_map(self.editor.toPlainText().encode("utf-8"))
                valid = True
            except (ValueError, TypeError, KeyError):
                pass
        self.save_button.setEnabled(idle and valid)

    def _start(self, callback, done):
        if self._thread is not None:
            return
        task = _MetadataCall(callback, self)
        self._thread = task
        def finished():
            self._thread = None
            if task.error:
                self.status.setText(task.error)
            else:
                done(task.value)
            task.deleteLater()
            self._controls()
        task.finished.connect(finished)
        self.status.setText("Reading local settings metadata…")
        self._controls()
        task.start()

    def load(self):
        def read():
            stores = self.reader(self.path, forbidden_roots=([self.series_root] if self.series_root else []))
            # Reuse exactly the admitted file, then recheck its descriptor.
            raw = _small_bytes(stores.source["path"])
            check = parse_store_map(raw)
            if check.source["sha256"] != stores.source["sha256"]:
                raise ValueError("Store map changed during Load; reload it")
            return stores, raw
        def done(value):
            stores, raw = value
            self.path = Path(stores.source["path"])
            self._expected, self._loaded = raw, True
            self.editor.setPlainText(raw.decode("utf-8-sig"))
            self._displayed_text = self.editor.toPlainText()
            self.status.setText("Loaded " + str(self.path) + "\nSHA-256: " + stores.source["sha256"])
        self._start(read, done)

    def new(self):
        self._expected, self._loaded = None, True
        self._displayed_text = '# Enter the real machine ID and actual paths; no value is inferred.\n[machine]\nid = ""\n\n[stores]\n'
        self.editor.setPlainText(self._displayed_text)
        self.status.setText("New map requested. Save refuses to replace an existing file that was not loaded.")
        self._controls()

    def save(self):
        if not self.save_button.isEnabled():
            return
        text = self.editor.toPlainText()
        raw = self._expected if self._expected is not None and text == self._displayed_text else text.encode("utf-8")
        expected = self._expected
        def write():
            return self.writer(self.path, raw, expected_bytes=expected,
                               forbidden_roots=([self.series_root] if self.series_root else []))
        def done(receipt):
            self.path = Path(receipt["path"])
            self._expected = raw
            self._displayed_text = self.editor.toPlainText()
            self.status.setText("Saved local settings. SHA-256: " + receipt["sha256"] + "\nPrevious content preserved: " + str(receipt["previous_copy"] or "new file"))
            self.saved.emit(str(self.path))
        self._start(write, done)

    def preview_roots(self):
        root = self.series_root
        if root is None:
            return
        def read():
            return propose_manifest_roots(root, _small_bytes(root / "series.toml"))
        def done(proposal):
            self.preview.setPlainText(json.dumps(proposal.receipt, ensure_ascii=False, indent=2))
            self.preview.show()
            self.status.setText("Proposal only. No series manifest, backup or migration receipt has been written.")
        self._start(read, done)

    def reject(self):
        if self._thread is not None:
            self.status.setText("Wait for the current local metadata operation to finish before closing.")
            return
        if self.editor.toPlainText() != self._displayed_text:
            answer = QMessageBox.question(self, "Discard unsaved local settings?", "The local store-map edits have not been saved.", QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Discard:
                return
        super().reject()

    def closeEvent(self, event):
        self.reject()
        event.setAccepted(not self.isVisible())


class StoreAvailabilityPanel(QGroupBox):
    """Compact Jobs report; explicit checks only, no passive filesystem query."""

    def __init__(self, parent=None, *, path=None, reader=read_store_map,
                 availability=root_availability, dialog_factory=StoreSettingsDialog):
        super().__init__("Machine data stores", parent)
        self.path = Path(path) if path is not None else default_store_map_path()
        self.reader, self.availability, self.dialog_factory = reader, availability, dialog_factory
        self.context = None
        self.generation = 0
        self._thread = self._dialog = None
        self.report = None
        layout = QVBoxLayout(self)
        buttons = QHBoxLayout()
        self.summary = _label("Choose a series. Availability is not checked by opening Jobs.")
        buttons.addWidget(self.summary, 1)
        self.settings_button = QPushButton("Machine stores…")
        self.check_button = QPushButton("Check availability")
        buttons.addWidget(self.settings_button)
        buttons.addWidget(self.check_button)
        layout.addLayout(buttons)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(72)
        self.details.setMaximumHeight(150)
        self.details.hide()
        layout.addWidget(self.details)
        self.settings_button.clicked.connect(self.open_settings)
        self.check_button.clicked.connect(self.check)
        self.check_button.setEnabled(False)

    def set_context(self, root, series_id):
        context = (str(root or ""), str(series_id or ""))
        if context == self.context:
            return
        self.context = context
        self.generation += 1
        self.report = None
        self.details.clear(); self.details.hide()
        self.summary.setText("Availability not checked for " + context[1] + ". Click Check availability; opening the page starts no work.")
        self.check_button.setEnabled(bool(root and series_id) and self._thread is None)

    def check(self):
        if self._thread is not None or not self.context or not all(self.context):
            return
        root, series = self.context
        generation = self.generation
        def read():
            stores = self.reader(self.path, forbidden_roots=[root])
            return stores, self.availability(stores, series)
        task = _MetadataCall(read, self)
        self._thread = task
        self.check_button.setEnabled(False)
        self.summary.setText("Checking explicitly configured directories…")
        def finished():
            self._thread = None
            if generation == self.generation:
                self.details.show()
                if task.error:
                    self.report = None
                    self.summary.setText("Machine store availability unavailable.")
                    self.details.setPlainText(task.error)
                else:
                    stores, report = task.value
                    self.report = report
                    self.summary.setText(stores.machine_id + " has " + series + ": " + " · ".join(row["alias"] + (" ✓" if row["available"] else " unavailable") for row in report["roots"]))
                    self.details.setPlainText("\n".join(row["alias"] + ": " + row["path"] + "\n" + ("Directory present" if row["available"] else row["reason"]) for row in report["roots"]) + "\n\nDirectory presence only. Input hashes, local output admission, capacity, worker eligibility and publication are not verified here. No job was started.")
            self.check_button.setEnabled(bool(self.context and all(self.context)))
            task.deleteLater()
        task.finished.connect(finished)
        task.start()

    def open_settings(self):
        if self._dialog is not None and self._dialog.isVisible():
            self._dialog.raise_()
            return
        root = self.context[0] if self.context and self.context[0] else None
        dialog = self.dialog_factory(root, self, path=self.path, reader=self.reader)
        self._dialog = dialog
        def changed(path):
            self.path = Path(path)
            self.generation += 1
            self.report = None
            self.details.clear(); self.details.hide()
            self.summary.setText("Local settings changed. Click Check availability to read the new settings.")
        dialog.saved.connect(changed)
        dialog.show()

    def can_close(self):
        return self._thread is None and (self._dialog is None or not self._dialog.isVisible())
