"""Read-only saved-calibration chooser shared by native Studio/Mass analysis."""
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout)

from darsia.fluidflower import mass_calibration_library as library

_ACTIVE_LOOKUPS = []


class LibraryLookup(QThread):
    ready = Signal(object, str)

    def __init__(self, root):
        super().__init__()
        self.root = root

    def run(self):
        try:
            self.ready.emit(library.discover_mass_calibrations(self.root), "")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.ready.emit([], str(error))


def native_application_path(selected, series_root):
    """Resolve a reference to its unchanged standalone native calibration file."""
    selected = Path(selected)
    value = library.read_json(selected)
    if value.get("format") == library.FORMAT:
        path, fit, entry = library.resolve_reference(selected, series_root)
        if entry["metadata"].get("application_adapter") != "native_frozen" or entry["target"].get("json_pointer"):
            raise ValueError("This saved calibration uses a method that the native-only Mass analysis adapter cannot apply.")
    else:
        path, fit = selected, value
    kind = library.calibration_kind(fit)
    if not kind or kind["method"] != "native":
        raise ValueError("Choose a saved native fitted/frozen calibration with its actual calibration and physical bindings.")
    return path.resolve()


class SavedMassCalibrationDialog(QDialog):
    def __init__(self, parent, series_root, *, load=True):
        super().__init__(parent)
        self.series_root = Path(series_root)
        self.selected_path = None
        self.setWindowTitle("Saved mass calibrations")
        self.resize(1000, 230)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Saved fitted or frozen calibrations · settings profiles are not model versions"))
        self.choices = QComboBox()
        self.choices.setMinimumContentsLength(80)
        layout.addWidget(self.choices)
        self.note = QLabel("Loading calibration references…")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        browse = QPushButton("Browse calibration JSON…")
        browse.clicked.connect(self._browse); buttons.addWidget(browse)
        buttons.addStretch()
        self.use = QPushButton("Use selected native calibration")
        self.use.setEnabled(False); self.use.clicked.connect(self._use); buttons.addWidget(self.use)
        close = QPushButton("Cancel"); close.clicked.connect(self.reject); buttons.addWidget(close)
        self.choices.currentIndexChanged.connect(self._selection_changed)
        if load:
            worker = LibraryLookup(self.series_root)
            _ACTIVE_LOOKUPS.append(worker)
            worker.ready.connect(self.set_rows)
            worker.finished.connect(lambda: _ACTIVE_LOOKUPS.remove(worker) if worker in _ACTIVE_LOOKUPS else None)
            worker.finished.connect(worker.deleteLater)
            worker.start()

    def set_rows(self, rows, error=""):
        self.choices.clear()
        self.choices.addItem("Choose a saved calibration", None)
        for row in rows:
            label = row.get("display_name") or Path(row["entry_path"]).stem
            if not row.get("available"):
                label += " · unavailable"
            elif row.get("application_adapter") != "native_frozen":
                label += " · separate experimental method"
            self.choices.addItem(label, row)
        if error:
            self.note.setText("Calibration library unavailable: " + error)
        elif not rows:
            self.note.setText("No saved fitted/frozen calibrations are indexed for this series. Browse an existing calibration JSON, or index the saved history first.")
        else:
            self.note.setText(f"{len(rows)} saved versions. Choose a native calibration to apply; other saved methods remain visible for reference. No calculation starts here.")

    def _selection_changed(self, *_):
        row = self.choices.currentData()
        self.use.setEnabled(bool(row and row.get("available") and row.get("application_adapter") == "native_frozen" and not row.get("json_pointer")))
        if not row:
            return
        if not row.get("available"):
            self.note.setText("Saved calibration unavailable: " + row.get("unavailable_reason", "source cannot be verified"))
        elif row.get("application_adapter") != "native_frozen":
            self.note.setText("Saved experimental calibration. Today's native-only Mass analysis adapter cannot apply this method; it remains visible in the library.")
        else:
            self.note.setText("Saved native calibration; current source, geometry and radiometry still require the existing preflight. No new fit or scientific approval is implied.")
        self.note.setToolTip("\n".join([row.get("scientific_status", ""), *row.get("warnings", []), row.get("calibration_path", "")]))

    def _accept_path(self, path):
        try:
            self.selected_path = native_application_path(path, self.series_root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Saved calibration unavailable", str(error))
            return
        self.accept()

    def _use(self):
        row = self.choices.currentData() or {}
        if row.get("available") and row.get("application_adapter") == "native_frozen":
            self._accept_path(row["entry_path"])

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Saved native calibration", str(library.library_root(self.series_root)), "JSON (*.json)")
        if path:
            self._accept_path(path)


def choose_saved_mass_calibration(parent, series_root):
    dialog = SavedMassCalibrationDialog(parent, series_root)
    return dialog.selected_path if dialog.exec() == QDialog.Accepted else None
