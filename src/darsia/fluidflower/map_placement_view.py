"""The review window of a map placed in all runs: one overlay per run.

The list shows the runs in review order (the reference run first) with their
offset from the reference run's grid and their checks; the viewer shows the
selected run's overlay at full resolution, zoomed with the mouse wheel and
moved by dragging. Display only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_COLUMNS = (
    "Run",
    "Status",
    "Offset median / max (px)",
    "Not covered (px)",
    "Free water (% of map)",
    "Unknown layers",
)


def _cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The table cells of one run of a placement index."""

    checks = dict(row.get("checks") or {})
    offset = dict(row.get("offset_px") or {})
    inside = int(checks.get("inside_pixels") or 0)
    water = checks.get("free_water_pixels")
    unknown = checks.get("unknown_layer_ids") or []
    return (
        str(row.get("run_id") or "").upper(),
        str(row.get("status") or ""),
        f"{offset['median_px']:.1f} / {offset['max_px']:.1f}" if offset else "",
        str(checks.get("uncovered_pixels", "")),
        f"{100.0 * int(water) / inside:.2f}" if water is not None and inside else "",
        ", ".join(str(value) for value in unknown) or ("none" if checks else ""),
    )


class OverlayView(QGraphicsView):
    """A photograph at full resolution: wheel to zoom, drag to move."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setMinimumSize(320, 240)

    def show_image(self, path: Path | str | None) -> bool:
        """Show an image file, fitted to the view; False when it cannot be read."""

        pixmap = QPixmap(str(path)) if path else QPixmap()
        self.item.setPixmap(pixmap)
        if pixmap.isNull():
            return False
        self.scene().setSceneRect(self.item.boundingRect())
        self.fit()
        return True

    def fit(self) -> None:
        """Show the whole image."""

        self.resetTransform()
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)


class MapPlacementDialog(QDialog):
    """Review a map placed in all runs: the runs, their checks, their overlays."""

    def __init__(self, placement: Mapping[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.placement = dict(placement)
        self.folder = Path(str(self.placement["folder"]))
        self.setWindowTitle("Map placed in all runs")
        self.resize(1500, 900)
        layout = QVBoxLayout(self)
        version = dict(self.placement.get("map_version") or {})
        note = QLabel(
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…) "
            f"from {str(self.placement.get('reference_run') or '').upper()}, placed in each run "
            "through its measured geometry. Review only: compare the boundaries with each "
            "run's own photograph. Nothing is adopted and no run configuration changed. "
            f"Folder: {self.folder}"
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.rows = list(self.placement.get("runs") or [])
        self.table.setRowCount(len(self.rows))
        for number, row in enumerate(self.rows):
            for column, value in enumerate(_cells(row)):
                self.table.setItem(number, column, QTableWidgetItem(value))
        self.table.itemSelectionChanged.connect(self._selected)
        splitter.addWidget(self.table)
        self.view = OverlayView()
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])
        layout.addWidget(splitter, 1)
        self.status = QLabel("Choose a run to see its overlay. Wheel: zoom · drag: move.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.open_button = buttons.addButton("Open folder", QDialogButtonBox.ButtonRole.ActionRole)
        self.open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))
        )
        self.fit_button = buttons.addButton("Fit", QDialogButtonBox.ButtonRole.ActionRole)
        self.fit_button.clicked.connect(self.view.fit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self.rows:
            self.table.selectRow(0)

    def show_run(self, number: int) -> bool:
        """Show the overlay of the run in row ``number``; False when it has none."""

        row = self.rows[number]
        overlay = row.get("overlay")
        shown = self.view.show_image(self.folder / overlay if overlay else None)
        run = str(row.get("run_id") or "").upper()
        self.status.setText(
            f"{run}: the placed map's boundaries on its reference photograph. "
            "Wheel: zoom · drag: move."
            if shown
            else f"{run}: {row.get('status') or 'no overlay'}."
        )
        return shown

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_run(rows[0].row())


__all__ = ["MapPlacementDialog", "OverlayView"]
