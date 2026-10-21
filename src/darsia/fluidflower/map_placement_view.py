"""The review window of a map placed in all runs: one overlay per run.

The list shows the runs in review order (the reference run first) with their
shift from the reference run's grid, their largest layer offset and the
checks; below it, the selected run's offset per pair of layers. The viewer
shows the selected run's overlay at full resolution, zoomed with the mouse
wheel and moved by dragging, with the legend of its two line types. Display
only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QImage, QPainter, QPixmap
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

from darsia.fluidflower.map_placement import OVERLAY_LEGEND

_COLUMNS = (
    "Run",
    "Status",
    "Shift from reference (px)",
    "Largest layer offset (mm)",
    "Gaps to fill (px)",
    "Water pockets (px)",
    "Unknown layers",
)


def _largest_offset(row: Mapping[str, Any]) -> float | None:
    """The layer-pair median offset of a run with the largest magnitude, or None."""

    offsets = dict(row.get("layer_offsets") or {})
    medians = [
        float(boundary["median_mm"])
        for boundary in offsets.get("boundaries") or []
        if boundary.get("median_mm") is not None
    ]
    return max(medians, key=abs) if medians else None


def _cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    """The table cells of one run of a placement index."""

    checks = dict(row.get("checks") or {})
    shift = dict(row.get("offset_px") or {})
    largest = _largest_offset(row)
    unknown = checks.get("unknown_layer_ids") or []
    return (
        str(row.get("run_id") or "").upper(),
        str(row.get("status") or ""),
        f"{shift['median_px']:.1f} / {shift['max_px']:.1f}" if shift else "",
        f"{largest:+.1f}" if largest is not None else ("none measured" if checks else ""),
        str(checks.get("gap_pixels", "")),
        str(checks.get("water_pocket_pixels", "")),
        ", ".join(str(value) for value in unknown) or ("none" if checks else ""),
    )


def offset_lines(row: Mapping[str, Any], layers: Mapping[str, Any]) -> list[str]:
    """One line per measured pair of layers of a run: median, P10 and P90."""

    def name(layer: int) -> str:
        entry = dict(layers.get(str(layer)) or {})
        return str(entry.get("name") or f"layer {layer}")

    def mm(value: Any) -> str:
        return "–" if value is None else f"{float(value):+.1f}"

    lines = []
    for boundary in (dict(row.get("layer_offsets") or {})).get("boundaries") or []:
        first, second = boundary["labels"]
        lines.append(
            f"{name(first)} | {name(second)}: median {mm(boundary['median_mm'])} mm "
            f"(P10 {mm(boundary['p10_mm'])}, P90 {mm(boundary['p90_mm'])}); "
            f"{boundary['measured_sample_count']} of {boundary['sample_count']} samples"
        )
    return lines


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

    def show_rgb(self, image: Any) -> bool:
        """Show a uint8 RGB array, fitted to the view; False when it is empty."""

        import numpy as np

        array = np.ascontiguousarray(image[..., :3], dtype=np.uint8)
        if array.size == 0:
            self.item.setPixmap(QPixmap())
            return False
        height, width = array.shape[:2]
        # The QImage only borrows the buffer: copy it while the array lives.
        picture = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.item.setPixmap(QPixmap.fromImage(picture))
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
        self.layers = dict(self.placement.get("layers") or {})
        self.setWindowTitle("Map placed in all runs")
        self.resize(1500, 900)
        layout = QVBoxLayout(self)
        version = dict(self.placement.get("map_version") or {})
        note = QLabel(
            f"Map {version.get('name') or ''} (SHA-256 {str(version.get('sha256') or '')[:12]}…) "
            f"from {str(self.placement.get('reference_run') or '').upper()}, placed in each run "
            "through its measured geometry. Review only: compare the boundaries with each "
            "run's own photograph. Nothing is adopted and no run configuration changed. "
            "Layer offsets: the strongest colour edge within ±25 mm of each boundary, positive "
            "from the lower towards the higher layer id. Gaps: pixels inside the map that it "
            "does not cover. Water pockets: free water apart from the main free water. "
            f"Folder: {self.folder}"
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
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
        side_layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        side_layout.addWidget(self.details)
        splitter.addWidget(side)
        self.view = OverlayView()
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 980])
        layout.addWidget(splitter, 1)
        legend = [
            (str(row.get("label") or ""), tuple(row.get("rgb") or (255, 255, 255)))
            for row in self.placement.get("overlay_legend") or []
        ] or list(OVERLAY_LEGEND)
        self.legend = QLabel(
            "&nbsp;&nbsp;&nbsp;".join(
                f"<span style='color: rgb{tuple(int(value) for value in color)}'>■</span> {label}"
                for label, color in legend
            )
        )
        layout.addWidget(self.legend)
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
        """Show the overlay and layer offsets of the run in row ``number``;
        False when it has no overlay."""

        row = self.rows[number]
        overlay = row.get("overlay")
        shown = self.view.show_image(self.folder / overlay if overlay else None)
        run = str(row.get("run_id") or "").upper()
        lines = offset_lines(row, self.layers)
        if row.get("layer_offsets") is None:
            self.details.setText(f"{run}: layer offsets not measured.")
        else:
            self.details.setText(
                f"{run}, layer offsets:\n" + ("\n".join(lines) or "no measurable boundary")
            )
        self.status.setText(
            f"{run}: the placed map on its reference photograph. Wheel: zoom · drag: move."
            if shown
            else f"{run}: {row.get('status') or 'no overlay'}."
        )
        return shown

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_run(rows[0].row())


__all__ = ["MapPlacementDialog", "OverlayView", "offset_lines"]
