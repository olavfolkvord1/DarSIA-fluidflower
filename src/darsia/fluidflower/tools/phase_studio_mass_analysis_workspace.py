"""Independent DarSIA mass-analysis page for a selected frozen phase model.

This page selects frozen phase models and embeds the retained saved-result
browser. Opening it never fits a calibration, starts workers, or publishes mass output.
"""

from __future__ import annotations

from contextvars import copy_context
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from darsia.fluidflower.model_display import friendly_model_name
from darsia.fluidflower.series_paths import configured_series, model_versions_folder
from darsia.fluidflower.phase_model_history import (
    _model_directories, inspect_phase_model_history_entry, metadata_only_catalog,
    ModelHistoryEntries, registration_context_available, REGISTRATION_ROOTS_UNAVAILABLE,
    run_with_registration_context,
)
from darsia.fluidflower.tools.phase_studio_series_ui import ModelMassLookup, SeriesMassWorkspace


@metadata_only_catalog
def _series_models(series: str, project_root: Path,
                   cancelled: Callable[[], bool] | None = None) -> list[dict]:
    """Show every recorded model in this series, including QA-fail candidates."""
    native_available = registration_context_available()
    root = Path(project_root).resolve()
    found: dict[str, dict] = {}
    # Scan only this series' model library and bounded result tree. The general
    # history helper also scans registered defaults and external review roots;
    # those paths repeated NAS work and delayed this selector. Users can add an
    # external model folder explicitly below.
    paths = [*_model_directories(model_versions_folder(root), max_depth=4),
             *(path for path in _model_directories(root / "results", max_depth=3)
               if path.name.startswith("phase-model-"))]
    candidates = []
    for path in paths:
        if cancelled is not None and cancelled():
            return []
        candidates.append(inspect_phase_model_history_entry(path))
    for entry in candidates:
        model_root = Path(str(entry["root"])).resolve()
        scope = str(entry.get("series_scope") or "").casefold().replace("_", "")
        datasets = {str(value).casefold().replace("_", "")
                    for value in (entry.get("datasets") or [])}
        belongs_here = model_root.is_relative_to(root) or series.replace("_", "") in datasets
        if scope and scope not in {"all", "shared", "crossseries", series.replace("_", "")}:
            belongs_here = False
        if belongs_here:
            found[str(model_root).casefold()] = entry
    return ModelHistoryEntries(
        sorted(found.values(), key=lambda row: (float(row.get("created_timestamp") or 0),
                                              str(row.get("model_id") or "")), reverse=True),
        diagnostics=(() if native_available else (REGISTRATION_ROOTS_UNAVAILABLE,)))


class _ModelChoicesLookup(QThread):
    ready = Signal(object, str)

    def __init__(self, series: str, project_root: Path, parent=None):
        super().__init__(parent)
        self.series = series
        self.project_root = Path(project_root)
        # Match Studio catalogue refresh: capture the verified owner context,
        # including explicit absence/error, before entering a Qt worker thread.
        self._catalogue_context = copy_context()
        self._registration_error = None
        try:
            self._registration_available = registration_context_available()
        except Exception as error:
            self._registration_available, self._registration_error = None, error

    def run(self):
        try:
            if self._registration_error is not None:
                raise self._registration_error
            rows = self._catalogue_context.run(
                run_with_registration_context,
                lambda: _series_models(self.series, self.project_root, self.isInterruptionRequested),
                available=self._registration_available,
            )
            if not self.isInterruptionRequested():
                self.ready.emit(rows, "")
        except Exception as error:
            if not self.isInterruptionRequested():
                self.ready.emit(None, str(error))


class MassAnalysisWorkspace(QWidget):
    """Choose a model and saved native fit, then coordinate a frozen campaign."""

    series_selected = Signal(str)

    def __init__(self, parent=None, *, series: str, project_root: str | Path):
        super().__init__(parent)
        self.series = str(series).casefold()
        self.project_root = Path(project_root)
        self.series_roots = {
            key: Path(root) for key, _label, root in configured_series()
        }
        self.series_roots[self.series] = self.project_root
        self.phase_catalog = None
        self._generation = 0
        self._model_lookup = None
        self._catalog_lookup = None
        self._background_lookups: list[QThread] = []

        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget(scroll)
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        title = QLabel("Mass analysis · frozen model and calibration")
        font = title.font(); font.setBold(True); title.setFont(font)
        layout.addWidget(title)
        note = QLabel("Choose a phase model, saved native calibration and experiments here. "
                      "Image exclusions can come from a source-bound TOML selection receipt below. "
                      "Measured overflow can be compared with gas mass without refitting. "
                      "Opening this page never starts a calculation; validation-only models "
                      "remain ineligible for production mass analysis.")
        note.setWordWrap(True); layout.addWidget(note)
        series_row = QHBoxLayout()
        self.series_selector = QComboBox()
        for key, label, _root in configured_series():
            self.series_selector.addItem(label, key)
        if self.series_selector.findData(self.series) < 0:
            self.series_selector.addItem(self.series.upper(), self.series)
        self.series_selector.setCurrentIndex(self.series_selector.findData(self.series))
        self.series_selector.currentIndexChanged.connect(self._series_selection_changed)
        series_row.addWidget(QLabel("Experimental series"))
        series_row.addWidget(self.series_selector, 1)
        layout.addLayout(series_row)
        row = QHBoxLayout()
        self.model = QComboBox(); self.model.setMinimumWidth(440)
        self.model.currentIndexChanged.connect(self._selected_model_changed)
        row.addWidget(QLabel("Phase model")); row.addWidget(self.model, 1)
        self.refresh_button = QPushButton("Refresh models")
        self.refresh_button.clicked.connect(self.refresh_models)
        row.addWidget(self.refresh_button)
        self.browse_button = QPushButton("Add model folder…")
        self.browse_button.clicked.connect(self._browse_model)
        row.addWidget(self.browse_button)
        layout.addLayout(row)
        self.status = QLabel("Loading model versions…")
        self.status.setWordWrap(True); layout.addWidget(self.status)

        self.saved_results = SeriesMassWorkspace(self, selected_model=self._selected_model)
        self.saved_results.series.setCurrentIndex(
            self.saved_results.series.findData(self.series))
        layout.addWidget(self.saved_results)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_lookups)
        self.refresh_models()

    def set_series(self, series: str, project_root: str | Path) -> None:
        self.series = str(series).casefold()
        self.project_root = Path(project_root)
        self.series_roots[self.series] = self.project_root
        index = self.series_selector.findData(self.series)
        if index < 0:
            self.series_selector.addItem(self.series.upper(), self.series)
            index = self.series_selector.findData(self.series)
        self.series_selector.blockSignals(True)
        self.series_selector.setCurrentIndex(index)
        self.series_selector.blockSignals(False)
        self._generation += 1
        self.phase_catalog = None
        self.saved_results.phase_catalog = None
        self.saved_results.series.setCurrentIndex(
            self.saved_results.series.findData(self.series))
        self.refresh_models()

    def _series_selection_changed(self, _index: int) -> None:
        series = self.series_selector.currentData()
        root = self.series_roots.get(series)
        if not series or root is None:
            return
        self.set_series(series, root)
        self.series_selected.emit(series)

    def refresh_models(self) -> None:
        self._generation += 1
        generation = self._generation
        previous = self.model.currentData()
        self.status.setText("Reading model versions for this series…")
        self.refresh_button.setEnabled(False)
        if self._model_lookup is not None and self._model_lookup.isRunning():
            self._model_lookup.requestInterruption()
        lookup = _ModelChoicesLookup(self.series, self.project_root, self)
        self._model_lookup = lookup
        self._background_lookups.append(lookup)
        lookup.ready.connect(lambda rows, error: self._models_ready(
            generation, previous, rows, error))
        lookup.finished.connect(lambda: self.refresh_button.setEnabled(True))
        lookup.finished.connect(lambda: self._forget_lookup(lookup))
        lookup.start()

    def _forget_lookup(self, lookup: QThread) -> None:
        if lookup in self._background_lookups:
            self._background_lookups.remove(lookup)
        lookup.deleteLater()

    def _stop_lookups(self) -> None:
        # The selected NAS may still be loading when the user closes DarSIA.
        # End Qt threads before their owning widget is destroyed.
        for lookup in list(self._background_lookups):
            lookup.requestInterruption()
        for lookup in list(self._background_lookups):
            lookup.wait()

    def closeEvent(self, event) -> None:
        self._stop_lookups()
        super().closeEvent(event)

    def _models_ready(self, generation: int, previous, rows, error: str) -> None:
        if generation != self._generation:
            return
        if error:
            self.status.setText("Model discovery failed: " + error)
            return
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Choose a phase model…", None)
        for entry in rows:
            name = friendly_model_name(entry) or str(entry.get("model_id") or Path(entry["root"]).name)
            label = f"{name} · {entry.get('state_label') or 'Status unknown'} · {entry.get('model_id') or Path(entry['root']).name}"
            self.model.addItem(label, str(entry["root"]))
            self.model.setItemData(self.model.count()-1, str(entry["root"]),
                                   role=3)  # Qt.ToolTipRole
        index = self.model.findData(previous) if previous else -1
        self.model.setCurrentIndex(max(index, 0))
        self.model.blockSignals(False)
        diagnostics = tuple(getattr(rows, "diagnostics", ()))
        self.status.setText(f"{len(rows)} recorded model versions. QA warnings remain visible; choose one to load its fixed maps."
                           + ("\n" + "\n".join(diagnostics) if diagnostics else ""))
        if index > 0:
            self._selected_model_changed()

    def _browse_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Phase-model folder", str(model_versions_folder(self.project_root)))
        if path:
            index = self.model.findData(path)
            if index < 0:
                entry = inspect_phase_model_history_entry(Path(path))
                self.model.addItem(
                    f"{friendly_model_name(entry) or entry['model_id']} · {entry.get('state_label')} · {entry['model_id']}",
                    str(Path(path).resolve()))
                index = self.model.count()-1
            self.model.setCurrentIndex(index)

    def _selected_model_changed(self, _index: int = -1) -> None:
        path = self.model.currentData()
        self._generation += 1
        generation = self._generation
        self.phase_catalog = None
        self.saved_results.phase_catalog = None
        if not path:
            return
        self.status.setText("Checking fixed phase maps and registered mass inputs for " + Path(path).name + "…")
        lookup = ModelMassLookup(Path(path), self)
        self._catalog_lookup = lookup
        self._background_lookups.append(lookup)
        lookup.ready.connect(lambda catalog, error: self._catalog_ready(generation, catalog, error))
        lookup.finished.connect(lambda: self._forget_lookup(lookup))
        lookup.start()

    def _catalog_ready(self, generation: int, catalog, error: str) -> None:
        if generation != self._generation:
            return
        if error or catalog is None:
            self.status.setText("Selected model is not ready for native mass analysis: " + error)
            return
        if catalog["dataset"].casefold() != self.series:
            self.status.setText("Selected model belongs to another experimental series")
            return
        self.phase_catalog = catalog
        self.saved_results.model_results_ready(catalog, "")
        ready = sum(bool(job.get("concentration", {}).get("ready")) for job in catalog["jobs"])
        self.status.setText(f"{catalog['model_id']} · {ready}/{len(catalog['jobs'])} experiments have registered native inputs. "
                            "Inspect saved concentration and mass results below.")

    def _selected_model(self) -> dict | None:
        """Return the selected saved model for the retained result browser."""
        root = self.model.currentData()
        return {"root": root} if root else None
