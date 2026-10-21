"""Adopt a prepared painted map as a geometry revision, with its facies table.

The window lists every label of the prepared maps with its layer, facies
and role, and asks for each label's facies id in the series' full facies
table: the facies table of the map. None is filled in for the user.
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

    def __init__(self, current: Mapping[str, Any], folder: Path, facies_ids: Mapping[int, int]) -> None:
        super().__init__()
        self.current, self.folder, self.facies_ids = dict(current), Path(folder), dict(facies_ids)

    @Slot()
    def run(self) -> None:
        from darsia.fluidflower.geometry_facies import validate_geometry_config

        start = time.monotonic()
        try:
            candidate, summary = ma.adoption_candidate(self.current, self.folder, self.facies_ids)
            report = validate_geometry_config(candidate)
        except (OSError, ValueError, KeyError) as error:
            self.failed.emit(str(error))
            return
        errors = [
            str(issue.get("message")) for issue in report.get("issues") or []
            if issue.get("severity") == "error"
        ]
        self.finished.emit(
            {
                "candidate": candidate,
                "summary": summary,
                "errors": errors,
                "seconds": time.monotonic() - start,
            }
        )


class _SaveRevisionWorker(QObject):
    """Save a geometry revision candidate with the existing Store."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Any, candidate: Mapping[str, Any], actor: str, reason: str) -> None:
        super().__init__()
        self.project, self.candidate, self.actor, self.reason = project, dict(candidate), actor, reason

    @Slot()
    def run(self) -> None:
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        try:
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
        super().__init__(parent)
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        self.project = project
        self.folder = Path(folder)
        self.actor = str(actor)
        self.preparation = ma.read_preparation(self.folder)
        self.current = GeometryRevisionStore(project).current()
        self.table_rows = ma.facies_table(self.current.get("configuration") or {})
        self.candidate: dict[str, Any] | None = None
        self.revision: dict[str, Any] | None = None
        self._thread: QThread | None = None
        self._worker: _SaveRevisionWorker | None = None
        self.setWindowTitle("Adopt the prepared map")
        self.resize(900, 640)
        layout = QVBoxLayout(self)
        version = dict(self.preparation.get("map_version") or {})
        prepared = [row for row in self.preparation.get("runs") or [] if row.get("status") == "prepared"]
        note = QLabel(
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…), "
            f"prepared on the label grid of {len(prepared)} of "
            f"{len(self.preparation.get('runs') or [])} runs. Give each label the facies id it "
            "has in the series' facies table; nothing is chosen for you. 'Check' builds the "
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
        self.table = QTableWidget(len(self.values), len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.choices: dict[int, QComboBox] = {}
        for number, value in enumerate(self.values):
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
        self.status = QLabel("Choose a facies id for every label, then Check.")
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

    def facies_ids(self) -> dict[int, int]:
        """The facies id chosen for each label so far."""

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
        self.check_button.setEnabled(idle and len(self.facies_ids()) == len(self.values))
        self.save_button.setEnabled(idle and self.candidate is not None)

    def check(self) -> bool:
        """Start building and validating the candidate in a worker thread;
        False when it cannot start. The status says the outcome."""

        if self._thread is not None or self.revision is not None:
            return False
        if len(self.facies_ids()) != len(self.values):
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
        left = summary["not_prepared"]
        self.status.setText(
            f"Ready to save: {len(summary['runs'])} runs and {len(summary['regions'])} regions "
            f"(checked in {result['seconds']:.0f} s)."
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
        if self.candidate is None:
            return
        answer = QMessageBox.question(
            self,
            "Save revision",
            "Append a new geometry revision that adopts this map in every prepared run? The "
            "current revision stays in the history. Run configurations do not change until "
            "Apply geometry to Setup.",
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
