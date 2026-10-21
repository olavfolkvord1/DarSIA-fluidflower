"""Adopt a prepared painted map as a geometry revision, with its facies table.

An approved-session preparation derives one row per used painting facies,
with all extra current CSV cells carried by name and conflicts refused.
Legacy preparations retain their explicit label-to-series-facies choices.
"Check" builds the revision candidate and validates it as the Store will;
"Save revision…" asks once and then saves it with the existing geometry
revision store. Both read every run's files and run in a worker thread,
never on the GUI thread. The new revision is appended; the previous one
stays.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower import map_adoption as ma

_COLUMNS = ("Label", "Layer", "Facies in the map", "Role", "Facies id in the series' table")


class _CheckWorker(QObject):
    """Build the revision candidate and validate it as the Store will."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, current: Mapping[str, Any], folder: Path, facies_ids: Mapping[int, int] | None) -> None:
        """Retain the current revision, preparation and optional explicit facies choice."""
        super().__init__()
        self.current, self.folder = dict(current), Path(folder)
        self.facies_ids = None if facies_ids is None else dict(facies_ids)

    @Slot()
    def run(self) -> None:
        """Build and validate a candidate off the GUI thread, reporting its result or error."""
        try:
            from darsia.fluidflower.geometry_facies import validate_geometry_config

            start = time.monotonic()
            candidate, summary = ma.adoption_candidate(self.current, self.folder, self.facies_ids)
            report = validate_geometry_config(candidate)
            errors = [
                str(issue.get("message")) for issue in report.get("issues") or []
                if issue.get("severity") == "error"
            ]
            result = {
                "candidate": candidate,
                "summary": summary,
                "errors": errors,
                "seconds": time.monotonic() - start,
            }
        except Exception as error:  # noqa: BLE001 - the owner must retire this thread
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(result)


class _SaveRevisionWorker(QObject):
    """Save a geometry revision candidate with the existing Store."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Any, candidate: Mapping[str, Any], actor: str, reason: str) -> None:
        super().__init__()
        self.project, self.candidate, self.actor, self.reason = project, dict(candidate), actor, reason

    @Slot()
    def run(self) -> None:
        """Append the confirmed candidate through the existing geometry revision Store."""
        try:
            from darsia.fluidflower.geometry_facies import GeometryRevisionStore

            revision = GeometryRevisionStore(self.project).save_revision(
                self.candidate, actor=self.actor, reason=self.reason
            )
        except Exception as error:  # noqa: BLE001 - shown to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(revision)


class AdoptionDialog(QDialog):
    """The facies table of a prepared map, its check and its explicit save."""

    def __init__(
        self,
        project: Any,
        folder: Path | str,
        *,
        actor: str,
        parent: QWidget | None = None,
    ) -> None:
        """Show the preparation table, validation result and explicit revision-save controls."""
        super().__init__(parent)
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        self.project = project
        self.folder = Path(folder)
        self.actor = str(actor)
        self.preparation = ma.read_preparation(self.folder)
        self.current = GeometryRevisionStore(project).current()
        self.automatic = "painting_facies" in self.preparation
        if self.automatic:
            binding = self.preparation["painting_facies"]
            snapshot = binding.get("facies_snapshot") if isinstance(binding, dict) else None
            if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("layers"), list)
                    or not isinstance(snapshot.get("facies"), list)
                    or any(not isinstance(row, dict) or not {"layer_id", "name", "facies"} <= row.keys()
                           for row in snapshot["layers"])
                    or any(not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                           or not isinstance(row[1], dict) or not {"role", "porosity", "permeability"} <= row[1].keys()
                           for row in snapshot["facies"])):
                raise ValueError("The automatic painting preparation has no readable facies snapshot; no manual fallback is offered")
        self.table_rows = [] if self.automatic else ma.facies_table(self.current.get("configuration") or {})
        self.candidate: dict[str, Any] | None = None
        self.revision: dict[str, Any] | None = None
        self._thread: QThread | None = None
        self._worker: _SaveRevisionWorker | None = None
        self.setWindowTitle("Adopt the prepared map")
        self.resize(900, 640)
        layout = QVBoxLayout(self)
        version = dict(self.preparation.get("map_version") or {})
        prepared = [row for row in self.preparation.get("runs") or [] if row.get("status") == "prepared"]
        instruction = (
            "Facies IDs come automatically from the approved saved painting, one row per used facies. "
            "Extra CSV cells come from the current table by exact name; conflicts refuse and are shown. "
            if self.automatic else
            "Give each label the facies id it has in the series' facies table; nothing is chosen for you. "
        )
        if self.automatic:
            from .painted_adoption_table import frame_basis_text
            instruction += frame_basis_text(self.preparation["painting_facies"]) + " Check verifies these bindings. "
        note = QLabel(
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…), "
            f"prepared on the label grid of {len(prepared)} of "
            f"{len(self.preparation.get('runs') or [])} runs. " + instruction + "'Check' builds the "
            "new geometry revision and validates it as the store will. 'Save revision…' then "
            f"appends it after revision {str(self.current.get('revision_id') or '')[:8]}, which "
            "stays. Apply geometry to Setup and Build run model remain separate steps. "
            f"Preparation: {self.folder}"
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        self.values = ma.label_values(self.preparation)
        regions = {int(row["layer_id"]): row for row in self.preparation.get("regions") or []}
        auto_rows = []
        if self.automatic:
            snapshot = self.preparation["painting_facies"]["facies_snapshot"]
            used = {row["facies"] for row in snapshot["layers"] if row["layer_id"] in self.values}
            for name, entry in snapshot["facies"]:
                if name in used:
                    layers = ", ".join(str(row["layer_id"]) for row in snapshot["layers"]
                                       if row["layer_id"] in self.values and row["facies"] == name)
                    auto_rows.append((name, layers, str(entry["role"]), str(entry["porosity"]),
                                      str(entry["permeability"]), "Verified during Check"))
        headings = ["Facies", "Layers", "Role", "Porosity", "Permeability", "Extra properties"] if self.automatic else list(_COLUMNS)
        self.table = QTableWidget(len(auto_rows) if self.automatic else len(self.values), len(headings))
        self.table.setHorizontalHeaderLabels(headings)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.choices: dict[int, QComboBox] = {}
        for number, row in enumerate(auto_rows):
            for column, text in enumerate(row):
                self.table.setItem(number, column, QTableWidgetItem(text))
        for number, value in enumerate(self.values if not self.automatic else ()):
            region = regions.get(value) or {}
            cells = (
                str(value),
                str(region.get("name") or ("Outside the map" if value == 0 else "")),
                str(region.get("facies") or ""),
                str(region.get("role") or ("excluded" if value == 0 else "")),
            )
            for column, text in enumerate(cells):
                self.table.setItem(number, column, QTableWidgetItem(text))
            choice = QComboBox()
            choice.addItem("Choose…", None)
            for row in self.table_rows:
                choice.addItem(
                    f"{row['facies_id']}: {row['name']} (porosity {row['porosity']}, "
                    f"permeability {row['permeability']})",
                    int(row["facies_id"]),
                )
            choice.currentIndexChanged.connect(self._choice_changed)
            self.table.setCellWidget(number, len(cells), choice)
            self.choices[value] = choice
        layout.addWidget(self.table, 1)
        self.status = QLabel("Check the automatic table and its full saved-source bindings." if self.automatic
                             else "Choose a facies id for every label, then Check.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.check_button = buttons.addButton("Check", QDialogButtonBox.ButtonRole.ActionRole)
        self.check_button.clicked.connect(lambda: self.check())
        self.save_button = buttons.addButton("Save revision…", QDialogButtonBox.ButtonRole.ActionRole)
        self.save_button.clicked.connect(self._save_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._choice_changed()

    def facies_ids(self) -> dict[int, int] | None:
        """The facies id chosen for each label so far."""

        if self.automatic:
            return None
        return {
            value: int(choice.currentData())
            for value, choice in self.choices.items()
            if choice.currentData() is not None
        }

    def _choice_changed(self, *_args: Any) -> None:
        self.candidate = None
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Check needs every facies id; Save needs a checked candidate; neither
        runs while a worker does or after the revision is saved."""

        idle = self._thread is None and self.revision is None
        self.check_button.setEnabled(idle and (self.automatic or len(self.facies_ids()) == len(self.values)))
        self.save_button.setEnabled(idle and self.candidate is not None)

    def check(self) -> bool:
        """Start building and validating the candidate in a worker thread;
        False when it cannot start. The status says the outcome."""

        if self._thread is not None or self.revision is not None:
            return False
        if not self.automatic and len(self.facies_ids()) != len(self.values):
            self.status.setText("Choose a facies id for every label first.")
            return False
        self.candidate = None
        worker = _CheckWorker(self.current, self.folder, self.facies_ids())
        worker.finished.connect(self._checked)
        worker.failed.connect(self._check_failed)
        self._start(worker, "Checking the revision; this reads every prepared run's files…")
        return True

    def _checked(self, result: object) -> None:
        """Keep a candidate that validates, or say why it does not."""

        result = dict(result)
        if result["errors"]:
            self.status.setText("The revision does not validate:\n" + "\n".join(result["errors"]))
            return
        self.candidate = result["candidate"]
        summary = result["summary"]
        if self.automatic:
            table = summary["facies_table"]
            self.table.setColumnCount(len(table["columns"]))
            self.table.setHorizontalHeaderLabels(table["columns"])
            self.table.setRowCount(len(table["rows"]))
            for number, row in enumerate(table["rows"]):
                for column, name in enumerate(table["columns"]):
                    self.table.setItem(number, column, QTableWidgetItem(str(row[name])))
        left = summary["not_prepared"]
        self.status.setText(
            f"Ready to save: {len(summary['runs'])} runs and {len(summary['regions'])} regions "
            f"(checked in {result['seconds']:.0f} s)."
            + (" " + summary["frame_basis"] if self.automatic else "")
            + (f" Not prepared, so not in the revision: {', '.join(run.upper() for run in left)}." if left else "")
        )

    def _check_failed(self, message: str) -> None:
        self.status.setText(f"The revision cannot be built: {message}")

    def _start(self, worker: QObject, message: str) -> None:
        """Run one worker in its own thread (0008 rule: freed in the GUI thread)."""

        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        thread.finished.connect(self._thread_finished)
        self._thread, self._worker = thread, worker
        self._update_buttons()
        self.status.setText(message)
        thread.start()

    def _save_clicked(self) -> None:
        """Ask before appending the checked revision with its retained frame explanation."""
        if self.candidate is None:
            return
        frame_note = ""
        if self.automatic:
            from .painted_adoption_table import frame_basis_text
            frame_note = " " + frame_basis_text(self.preparation["painting_facies"])
        answer = QMessageBox.question(
            self,
            "Save revision",
            "Append a new geometry revision that adopts this map in every prepared run? The "
            "current revision stays in the history. Run configurations do not change until "
            "Apply geometry to Setup." + frame_note,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.save()

    def save(self) -> bool:
        """Save the checked candidate in a worker thread; False when there is none."""

        if self.candidate is None or self._thread is not None:
            return False
        version = dict(self.preparation.get("map_version") or {})
        reason = (
            f"adopt the approved painted map {version.get('name') or ''} "
            f"(SHA-256 {str(version.get('sha256') or '')[:12]}) in every prepared run"
        )
        worker = _SaveRevisionWorker(self.project, self.candidate, self.actor, reason)
        worker.finished.connect(self._saved)
        worker.failed.connect(self._failed)
        self._start(worker, "Saving the revision; the store validates every run again…")
        return True

    def busy(self) -> bool:
        """Whether a check or a save is running."""

        return self._thread is not None

    def _saved(self, revision: object) -> None:
        self.revision = dict(revision)
        self.status.setText(
            f"Saved revision {self.revision['revision_id']}. Apply geometry to Setup and Build "
            "run model are the next, separate steps."
        )

    def _failed(self, message: str) -> None:
        self.status.setText(f"The revision was not saved: {message}")

    def _thread_finished(self) -> None:
        thread = self._thread
        if thread is not None:
            # After wait() the worker is freed here, in the GUI thread.
            thread.wait()
        self._thread = self._worker = None
        if thread is not None:
            thread.deleteLater()
        self._update_buttons()

    def reject(self) -> None:
        if self._thread is None:
            super().reject()


__all__ = ["AdoptionDialog"]
