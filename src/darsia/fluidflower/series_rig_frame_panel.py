"""Rig frame and injection points for a registered series.

The panel replaces the project workflow's injection-point tab when the
geometry-and-facies page is hosted for a series folder. Its coordinates follow
DarSIA's crop: four inner corners of the rig in a raw photograph map onto a
rectangle of the rig's inner width and height, x is measured from the left and
y above the bottom edge, in metres. Points can be typed in those coordinates
or placed by clicking in a photograph.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPolygonItem,
    QGraphicsSimpleTextItem,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.series_geometry import (
    FRAME_FROM_CONFIGURATION,
    FRAME_MARKED,
    RigFrame,
    SeriesRunInputs,
    default_actor,
    order_corners,
    photo_size,
    protocol_injection_points,
    remove_rig_frame,
    rig_frame_from_inputs,
    save_injection_points,
    save_rig_frame,
    saved_injection_points,
    saved_rig_frames,
)
from darsia.fluidflower.templates import RIG_CLASSES

# Clicking order for the corners (clockwise from the top left).
CLICK_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_LABELS = {
    "top_left": "Top left",
    "top_right": "Top right",
    "bottom_right": "Bottom right",
    "bottom_left": "Bottom left",
}
POINT_COLUMNS = (
    "Location ID",
    "Name",
    "x [m]",
    "y [m]",
    "Row [px]",
    "Col [px]",
    "Source",
)
_DISPLAY_LIMIT_PX = 2400


def _number(text: str) -> float | None:
    try:
        value = float(str(text).replace(",", ".").strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


class RigFramePanel(QWidget):
    """Rig frame of the reference run and the series' injection points."""

    changed = Signal()

    def __init__(
        self,
        page: Any,
        project: Any,
        reference: Callable[[], SeriesRunInputs | None],
        *,
        scrollable: bool = True,
    ) -> None:
        """``page`` provides the canvas (``page.canvas``).

        A page that shows photographs itself provides ``show_frame_photo(path)``,
        which displays the photograph in its canvas and returns whether it did.
        ``scrollable`` puts the panel in its own scroll area.
        """

        super().__init__()
        self.page = page
        self.project = project
        self._reference = reference
        self.canvas = page.canvas
        self.frame: RigFrame | None = None
        self._mode = ""
        self._corner_index = 0
        self._photo_path: Path | None = None
        self._photo_size: tuple[int, int] | None = None
        self._shown_preview: Any = None
        self._loaded_run: str | None = None
        self._overlay: list[Any] = []
        self._loading = False
        self._rig_class = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        if scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            outer.addWidget(scroll)
            scroll.setWidget(content)
        else:
            outer.addWidget(content)
        layout = QVBoxLayout(content)
        if not scrollable:
            layout.setContentsMargins(0, 0, 0, 0)

        if scrollable:
            text = (
                "Coordinates follow DarSIA's crop: the four inner corners of the rig "
                "in a raw photograph map onto a rectangle of the rig's inner width "
                "and height. x is measured from the left and y above the bottom "
                "edge, in metres. Photographs taken from the same camera position "
                "share the frame."
            )
        else:
            text = (
                "Mark the four inner corners of the rig and enter its inner size. "
                "Everything outside the frame is left out of the layers. x is "
                "measured from the left and y above the bottom edge, in metres."
            )
        note = QLabel(text)
        note.setObjectName("secondaryText")
        note.setWordWrap(True)
        layout.addWidget(note)

        photo_row = QHBoxLayout()
        self.show_reference_button = QPushButton("Show reference photo")
        self.show_reference_button.clicked.connect(self.show_reference_photo)
        photo_row.addWidget(self.show_reference_button)
        self.show_other_button = QPushButton("Show other photo…")
        self.show_other_button.setToolTip(
            "Any raw photograph from the same camera position, for example one "
            "in which the injection port is visible"
        )
        self.show_other_button.clicked.connect(self._choose_photo)
        photo_row.addWidget(self.show_other_button)
        photo_row.addStretch(1)
        layout.addLayout(photo_row)
        self.photo_label = QLabel("No photograph shown.")
        self.photo_label.setObjectName("secondaryText")
        self.photo_label.setWordWrap(True)
        layout.addWidget(self.photo_label)
        if not scrollable:
            # The host shows the reference photograph itself.
            self.show_reference_button.hide()
            self.photo_label.hide()

        frame_box = QGroupBox("Rig frame")
        frame_layout = QVBoxLayout(frame_box)
        self.frame_status = QLabel()
        self.frame_status.setWordWrap(True)
        frame_layout.addWidget(self.frame_status)
        self.corners = QTableWidget(len(CLICK_ORDER), 2)
        self.corners.setHorizontalHeaderLabels(["Row [px]", "Col [px]"])
        self.corners.setVerticalHeaderLabels(
            [CORNER_LABELS[name] for name in CLICK_ORDER]
        )
        self.corners.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.corners.setMaximumHeight(150)
        self.corners.itemChanged.connect(self._corner_edited)
        frame_layout.addWidget(self.corners)
        size_row = QHBoxLayout()
        self.width_m = QDoubleSpinBox()
        self.height_m = QDoubleSpinBox()
        for spin in (self.width_m, self.height_m):
            spin.setRange(0.0, 20.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.001)
            spin.setSuffix(" m")
            spin.editingFinished.connect(self._size_edited)
        size_row.addWidget(QLabel("Inner width"))
        size_row.addWidget(self.width_m)
        size_row.addWidget(QLabel("Inner height"))
        size_row.addWidget(self.height_m)
        size_row.addStretch(1)
        frame_layout.addLayout(size_row)
        self.class_row = QWidget()
        class_layout = QHBoxLayout(self.class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        self.class_size = QLabel()
        self.class_size.setObjectName("secondaryText")
        self.class_size.setWordWrap(True)
        class_layout.addWidget(self.class_size, 1)
        self.use_class_size = QPushButton("Use class size")
        self.use_class_size.setToolTip(
            "Enter the nominal inner size of the rig class. Measured values of "
            "this rig are better when you have them."
        )
        self.use_class_size.clicked.connect(self.apply_class_size)
        class_layout.addWidget(self.use_class_size)
        self.class_row.hide()
        frame_layout.addWidget(self.class_row)
        frame_buttons = QHBoxLayout()
        self.mark_corners = QPushButton("Mark corners in photo")
        self.mark_corners.setCheckable(True)
        self.mark_corners.setToolTip(
            "Click the four inner corners of the rig, in any order; they are named "
            "by their position. Zoom with the mouse wheel for precision."
        )
        self.mark_corners.toggled.connect(self._mark_corners_toggled)
        frame_buttons.addWidget(self.mark_corners)
        self.copy_crop = QPushButton("Copy crop section")
        self.copy_crop.clicked.connect(self._copy_crop)
        frame_buttons.addWidget(self.copy_crop)
        frame_buttons.addStretch(1)
        frame_layout.addLayout(frame_buttons)
        configuration_row = QHBoxLayout()
        self.use_configuration = QPushButton("Use series configuration")
        self.use_configuration.setToolTip(
            "Discard the frame marked here and use the run's [corrections.curvature.crop]"
        )
        self.use_configuration.clicked.connect(self._use_configuration)
        configuration_row.addWidget(self.use_configuration)
        configuration_row.addStretch(1)
        frame_layout.addLayout(configuration_row)
        layout.addWidget(frame_box)

        points_box = QGroupBox("Injection points")
        points_layout = QVBoxLayout(points_box)
        self.points = QTableWidget(0, len(POINT_COLUMNS))
        self.points.setHorizontalHeaderLabels(list(POINT_COLUMNS))
        self.points.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.points.verticalHeader().setVisible(False)
        header = self.points.horizontalHeader()
        for column in range(len(POINT_COLUMNS) - 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            len(POINT_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch
        )
        self.points.itemChanged.connect(self._point_edited)
        if not scrollable:
            self.points.setMaximumHeight(120)
        points_layout.addWidget(self.points)
        point_buttons = QHBoxLayout()
        add = QPushButton("Add")
        add.clicked.connect(self._add_point)
        point_buttons.addWidget(add)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_points)
        point_buttons.addWidget(remove)
        self.place_point = QPushButton("Click to place selected")
        self.place_point.setCheckable(True)
        self.place_point.toggled.connect(self._place_toggled)
        point_buttons.addWidget(self.place_point)
        point_buttons.addStretch(1)
        points_layout.addLayout(point_buttons)
        point_buttons = QHBoxLayout()
        self.load_protocol = QPushButton("Load from injection protocol")
        self.load_protocol.clicked.connect(self._load_protocol)
        point_buttons.addWidget(self.load_protocol)
        self.copy_points = QPushButton("Copy coordinates")
        self.copy_points.clicked.connect(self._copy_points)
        point_buttons.addWidget(self.copy_points)
        point_buttons.addStretch(1)
        points_layout.addLayout(point_buttons)
        layout.addWidget(points_box)

        self.status = QLabel()
        self.status.setObjectName("secondaryText")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.canvas.image_point_selected.connect(self._canvas_clicked)
        self.canvas.scene().sceneRectChanged.connect(lambda *_: self._draw_overlay())
        self.load()

    # ----------------------------------------------------------------- state

    def reference(self) -> SeriesRunInputs | None:
        return self._reference()

    def load(self) -> None:
        """Read the frame of the reference run and the saved injection points."""

        self._set_mode("")
        reference = self.reference()
        run_id = reference.run_id if reference is not None else ""
        if run_id != self._loaded_run:
            # Photographs shown for another run do not belong to this frame.
            self._loaded_run = run_id
            self._photo_path = None
            self._photo_size = None
            self._shown_preview = None
            self.photo_label.setText("No photograph shown.")
        frame = None
        if reference is not None:
            frame = saved_rig_frames(self.project).get(reference.run_id)
            if frame is None:
                frame = rig_frame_from_inputs(reference)
        self.frame = frame
        self._loading = True
        try:
            self._show_frame()
            self.points.setRowCount(0)
            for point in saved_injection_points(self.project):
                self._append_point(point)
            self._refresh_point_pixels()
        finally:
            self._loading = False
        if (
            self._photo_path is None
            and reference is not None
            and reference.baseline_image
        ):
            try:
                width, height = photo_size(reference.baseline_image)
                self._photo_size = (width, height)
            except (OSError, ValueError):
                self._photo_size = None
        self._update_status()
        self._draw_overlay()

    def _show_frame(self) -> None:
        reference = self.reference()
        frame = self.frame
        for row, name in enumerate(CLICK_ORDER):
            for column in range(2):
                value = None
                if frame is not None and name in frame.corners:
                    value = frame.corners[name][column]
                self.corners.setItem(row, column, QTableWidgetItem(_fmt(value, 1)))
        self.width_m.setValue(float(frame.width_m or 0.0) if frame else 0.0)
        self.height_m.setValue(float(frame.height_m or 0.0) if frame else 0.0)
        self._update_class_hint()
        if reference is None:
            text = "Choose a reference run."
        elif frame is None:
            text = (
                f"No frame for run {reference.run_id}. Mark the four inner corners in "
                "a photograph and enter the inner width and height."
            )
        elif frame.source == FRAME_FROM_CONFIGURATION:
            text = f"From the series configuration of run {reference.run_id}."
        else:
            text = f"Marked here for run {reference.run_id}."
        if frame is not None and not frame.has_size():
            text += " Enter the rig's inner width and height."
        self.frame_status.setText(text)
        configured = (
            reference is not None and rig_frame_from_inputs(reference) is not None
        )
        self.use_configuration.setEnabled(
            configured
            and frame is not None
            and frame.source != FRAME_FROM_CONFIGURATION
        )
        self.copy_crop.setEnabled(frame is not None and frame.complete())

    def _frame_from_ui(self) -> RigFrame | None:
        reference = self.reference()
        corners: dict[str, tuple[float, float]] = {}
        for row, name in enumerate(CLICK_ORDER):
            values = [
                (
                    _number(self.corners.item(row, column).text())
                    if self.corners.item(row, column)
                    else None
                )
                for column in range(2)
            ]
            if values[0] is not None and values[1] is not None:
                corners[name] = (values[0], values[1])
        if not corners:
            return None
        return RigFrame(
            # Corners clicked in another order are named by their position.
            corners=order_corners(corners),
            width_m=self.width_m.value() or None,
            height_m=self.height_m.value() or None,
            run_id=reference.run_id if reference is not None else "",
            source=FRAME_MARKED,
            image=str(
                self._photo_path or (reference.baseline_image if reference else "")
            ),
            image_size=self._photo_size,
        )

    def _store_frame(self) -> None:
        frame = self._frame_from_ui()
        if frame is None or not frame.run_id:
            return
        if self.frame is not None and self.frame.source == FRAME_FROM_CONFIGURATION:
            unchanged = (
                frame.corners == self.frame.corners
                and frame.width_m == self.frame.width_m
                and frame.height_m == self.frame.height_m
            )
            if unchanged:
                return
        try:
            save_rig_frame(self.project, frame, actor=default_actor())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not save the rig frame", str(error))
            return
        self.frame = frame
        self._loading = True
        try:
            self._show_frame()
            self._refresh_point_pixels()
        finally:
            self._loading = False
        self._store_points()
        self._draw_overlay()
        self.changed.emit()

    def _corner_edited(self, _item: QTableWidgetItem) -> None:
        if not self._loading:
            self._store_frame()

    def _size_edited(self) -> None:
        if not self._loading:
            self._store_frame()
        self._update_class_hint()

    # ------------------------------------------------------------- rig class

    def set_rig_class(self, key: str) -> None:
        """The series' rig class (a key of ``RIG_CLASSES``; "" if unset)."""

        self._rig_class = str(key or "")
        self._update_class_hint()

    def _update_class_hint(self) -> None:
        rig = RIG_CLASSES.get(self._rig_class)
        if rig is None:
            self.class_row.hide()
            return
        self.class_row.show()
        if rig.width_m is None or rig.height_m is None:
            self.class_size.setText(f"{rig.label}: no nominal size.")
            self.use_class_size.hide()
            return
        self.class_size.setText(f"{rig.label}: nominal inner size {rig.size_text()}.")
        self.class_size.setToolTip(rig.source)
        self.use_class_size.show()
        same = (
            abs(self.width_m.value() - rig.width_m) < 5.0e-5
            and abs(self.height_m.value() - rig.height_m) < 5.0e-5
        )
        self.use_class_size.setEnabled(not same)

    def apply_class_size(self) -> None:
        """Enter the class' nominal inner size as the rig size."""

        rig = RIG_CLASSES.get(self._rig_class)
        if rig is None or rig.width_m is None or rig.height_m is None:
            return
        self._loading = True
        try:
            self.width_m.setValue(float(rig.width_m))
            self.height_m.setValue(float(rig.height_m))
        finally:
            self._loading = False
        self._store_frame()
        self._update_class_hint()

    def _use_configuration(self) -> None:
        reference = self.reference()
        if reference is None:
            return
        try:
            remove_rig_frame(self.project, reference.run_id, actor=default_actor())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not remove the rig frame", str(error))
            return
        self.load()
        self.changed.emit()

    def _copy_crop(self) -> None:
        if self.frame is None or not self.frame.complete():
            return
        lines = ["[corrections.curvature.crop]"]
        for key, value in self.frame.native_crop().items():
            if value is True:
                text = "true"
            elif isinstance(value, list):
                text = f"[{value[0]}, {value[1]}]"
            else:
                text = repr(value)
            lines.append(f"{key} = {text}")
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
        self.status.setText("Crop section copied.")

    # ---------------------------------------------------------------- points

    def _append_point(self, point: dict[str, Any]) -> None:
        row = self.points.rowCount()
        self.points.insertRow(row)
        pixel = point.get("pixel") or {}
        values = (
            point.get("location_id") or "",
            point.get("name") or "",
            _fmt(point.get("x_m"), 4),
            _fmt(point.get("y_m"), 4),
            _fmt(pixel.get("row"), 1),
            _fmt(pixel.get("col"), 1),
            point.get("source") or "",
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column >= 4:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.points.setItem(row, column, item)
        self.points.item(row, 0).setData(Qt.ItemDataRole.UserRole, dict(pixel) or None)

    def _row_point(self, row: int) -> dict[str, Any]:
        def text(column: int) -> str:
            item = self.points.item(row, column)
            return item.text().strip() if item is not None else ""

        pixel = (
            self.points.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if self.points.item(row, 0)
            else None
        )
        return {
            "location_id": text(0),
            "name": text(1),
            "x_m": _number(text(2)),
            "y_m": _number(text(3)),
            "pixel": (
                pixel
                if isinstance(pixel, dict) and pixel.get("row") is not None
                else None
            ),
            "source": text(6),
        }

    def current_points(self) -> list[dict[str, Any]]:
        return [self._row_point(row) for row in range(self.points.rowCount())]

    def _store_points(self) -> None:
        try:
            save_injection_points(
                self.project, self.current_points(), actor=default_actor()
            )
        except (OSError, ValueError) as error:
            self.status.setText(f"Injection points not saved: {error}")
            return
        self._update_status()
        self.changed.emit()

    def _point_edited(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        row = item.row()
        if item.column() in (2, 3):
            self._loading = True
            try:
                self.points.item(row, 6).setText("typed")
                # Typed coordinates replace a picked pixel.
                self.points.item(row, 0).setData(Qt.ItemDataRole.UserRole, None)
                self._refresh_point_pixels()
            finally:
                self._loading = False
        self._store_points()
        self._draw_overlay()

    def _refresh_point_pixels(self) -> None:
        """Show each point's pixel in the reference photograph."""

        frame = self.frame
        reference = self.reference()
        for row in range(self.points.rowCount()):
            point = self._row_point(row)
            pixel = point.get("pixel")
            shown: tuple[float, float] | None = None
            usable_frame = frame is not None and frame.complete() and frame.has_size()
            if pixel and (
                reference is None or pixel.get("run_id") in ("", reference.run_id)
            ):
                shown = (pixel["row"], pixel["col"])
                if usable_frame and (point["x_m"] is None or point["y_m"] is None):
                    x_m, y_m = frame.pixel_to_rig(pixel["row"], pixel["col"])
                    self.points.item(row, 2).setText(_fmt(x_m, 4))
                    self.points.item(row, 3).setText(_fmt(y_m, 4))
            elif usable_frame and point["x_m"] is not None and point["y_m"] is not None:
                shown = frame.rig_to_pixel(point["x_m"], point["y_m"])
            self.points.item(row, 4).setText(_fmt(shown[0], 1) if shown else "")
            self.points.item(row, 5).setText(_fmt(shown[1], 1) if shown else "")

    def _add_point(self) -> None:
        used = {point["location_id"] for point in self.current_points()}
        number = 1
        while f"p{number}" in used:
            number += 1
        self._loading = True
        try:
            self._append_point(
                {
                    "location_id": f"p{number}",
                    "name": "",
                    "x_m": None,
                    "y_m": None,
                    "source": "",
                }
            )
        finally:
            self._loading = False
        self.points.selectRow(self.points.rowCount() - 1)
        self._store_points()

    def _remove_points(self) -> None:
        rows = sorted(
            {index.row() for index in self.points.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.points.removeRow(row)
        if rows:
            self._store_points()
            self._draw_overlay()

    def _load_protocol(self) -> None:
        reference = self.reference()
        path = reference.injection_protocol if reference is not None else None
        if path is None or not Path(path).is_file():
            QMessageBox.information(
                self,
                "No injection protocol",
                "The reference run has no readable injection protocol.",
            )
            return
        try:
            incoming = protocol_injection_points(Path(path))
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(
                self, "Could not read the injection protocol", str(error)
            )
            return
        existing = {
            (round(point["x_m"], 6), round(point["y_m"], 6))
            for point in self.current_points()
            if point["x_m"] is not None and point["y_m"] is not None
        }
        used = {point["location_id"] for point in self.current_points()}
        added = 0
        self._loading = True
        try:
            for point in incoming:
                key = (round(point["x_m"], 6), round(point["y_m"], 6))
                if key in existing:
                    continue
                identifier = point["location_id"]
                while identifier in used:
                    identifier += "'"
                used.add(identifier)
                point["location_id"] = identifier
                self._append_point(point)
                added += 1
            self._refresh_point_pixels()
        finally:
            self._loading = False
        self.status.setText(
            f"{added} injection point(s) read from {Path(path).name}."
            if added
            else "The protocol's injection points are already listed."
        )
        if added:
            self._store_points()
            self._draw_overlay()

    def _copy_points(self) -> None:
        lines = ["location_id,location_x,location_y"]
        for point in self.current_points():
            if point["x_m"] is not None and point["y_m"] is not None:
                lines.append(
                    f"{point['location_id']},{point['x_m']:.4f},{point['y_m']:.4f}"
                )
        QGuiApplication.clipboard().setText("\n".join(lines) + "\n")
        self.status.setText("Injection-point coordinates copied.")

    # ----------------------------------------------------------------- photo

    def _choose_photo(self) -> None:
        reference = self.reference()
        from darsia.fluidflower.image_formats import photo_dialog_filter

        start = ""
        baseline = None
        if reference is not None and reference.baseline_image is not None:
            baseline = Path(reference.baseline_image)
            start = str(baseline.parent)
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Show a raw photograph",
            start,
            photo_dialog_filter(baseline),
        )
        if selected:
            self.show_photo(Path(selected))

    def show_reference_photo(self) -> None:
        reference = self.reference()
        if reference is None or reference.baseline_image is None:
            QMessageBox.information(
                self, "No reference photo", "The reference run has no baseline image."
            )
            return
        self.show_photo(Path(reference.baseline_image))

    def use_host_photo(self, size: tuple[int, int], preview: Any) -> None:
        """The host shows the reference photograph (``preview``) itself."""

        self._photo_path = None
        self._photo_size = (int(size[0]), int(size[1]))
        self._shown_preview = preview
        self._draw_overlay()

    def show_photo(self, path: Path) -> bool:
        """Display a raw photograph in the canvas (reduced for display)."""

        from darsia.fluidflower.photograph_io import load_reduced

        host_show = getattr(self.page, "show_frame_photo", None)
        try:
            width, height = photo_size(path)
            if not callable(host_show):
                factor = max(1, int(math.ceil(max(width, height) / _DISPLAY_LIMIT_PX)))
                rgb = load_reduced(path, factor)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not show the photograph", str(error))
            return False
        if callable(host_show):
            if not host_show(Path(path)):
                return False
        else:
            shape = rgb.shape[:2]
            preview = {
                "baseline": rgb,
                "labels": np.zeros(shape, dtype=np.int32),
                "facies": np.zeros(shape, dtype=np.int32),
                "depth": np.full(shape, np.nan, dtype=np.float32),
            }
            self.canvas.set_model(preview, self.page._collect_config())
        self._shown_preview = self.canvas.preview
        for name in ("background", "overlay"):
            combo = getattr(self.page, name, None)
            index = combo.findData("baseline") if combo is not None else -1
            if index >= 0:
                combo.setCurrentIndex(index)
        self._photo_path = Path(path)
        self._photo_size = (int(width), int(height))
        expected = None
        if self.frame is not None and self.frame.image_size:
            expected = tuple(self.frame.image_size)
        note = ""
        if expected and expected != self._photo_size:
            note = (
                f" The frame was marked on a {expected[0]} x {expected[1]} px photograph;"
                " use photographs of the same size."
            )
        self.photo_label.setText(
            f"Showing {Path(path).name} ({width} x {height} px).{note}"
        )
        self._draw_overlay()
        return True

    # ----------------------------------------------------------- interaction

    def _host_hint(self, text: str) -> None:
        """Show what to click where the host page shows its hints ("" ends it)."""

        hint = getattr(self.page, "show_frame_hint", None)
        if callable(hint):
            hint(text)

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        if not mode:
            self._host_hint("")
        for button, name in (
            (self.mark_corners, "corners"),
            (self.place_point, "point"),
        ):
            button.blockSignals(True)
            button.setChecked(mode == name)
            button.blockSignals(False)
        self.canvas.place_mode = bool(mode)
        if mode:
            for name in ("place_injector", "select_layer", "draw_facies_example"):
                button = getattr(self.page, name, None)
                if button is not None and button.isChecked():
                    button.setChecked(False)
            self.canvas.place_mode = True

    def _ensure_photo(self) -> bool:
        """Clicks are mapped to the photograph this panel shows last."""

        if (
            self._shown_preview is not None
            and self.canvas.preview is self._shown_preview
        ):
            return True
        if self._photo_path is not None:
            return self.show_photo(self._photo_path)
        return self._show_default_photo()

    def _show_default_photo(self) -> bool:
        reference = self.reference()
        if reference is None or reference.baseline_image is None:
            QMessageBox.information(
                self, "No photograph", "Show a raw photograph of the rig first."
            )
            return False
        return self.show_photo(Path(reference.baseline_image))

    def _mark_corners_toggled(self, active: bool) -> None:
        if active and not self._ensure_photo():
            self._set_mode("")
            return
        self._corner_index = 0
        self._set_mode("corners" if active else "")
        self._update_status()

    def _place_toggled(self, active: bool) -> None:
        if active:
            if self.points.currentRow() < 0:
                self._add_point()
            if not self._ensure_photo():
                self._set_mode("")
                return
        self._set_mode("point" if active else "")
        self._update_status()

    def _canvas_clicked(self, fx: float, fy: float) -> None:
        if not self._mode or self._photo_size is None:
            return
        width, height = self._photo_size
        row = float(fy) * (height - 1)
        col = float(fx) * (width - 1)
        if self._mode == "corners":
            index = self._corner_index
            self._loading = True
            try:
                self.corners.setItem(index, 0, QTableWidgetItem(_fmt(row, 1)))
                self.corners.setItem(index, 1, QTableWidgetItem(_fmt(col, 1)))
            finally:
                self._loading = False
            self._corner_index += 1
            if self._corner_index >= len(CLICK_ORDER):
                self._set_mode("")
                self._store_frame()
            else:
                self._draw_overlay()
            self._update_status()
        elif self._mode == "point":
            selected = self.points.currentRow()
            if selected < 0:
                return
            reference = self.reference()
            self._loading = True
            try:
                self.points.item(selected, 0).setData(
                    Qt.ItemDataRole.UserRole,
                    {
                        "run_id": reference.run_id if reference is not None else "",
                        "row": row,
                        "col": col,
                    },
                )
                frame = self.frame
                if frame is not None and frame.complete() and frame.has_size():
                    x_m, y_m = frame.pixel_to_rig(row, col)
                    self.points.item(selected, 2).setText(_fmt(x_m, 4))
                    self.points.item(selected, 3).setText(_fmt(y_m, 4))
                else:
                    self.points.item(selected, 2).setText("")
                    self.points.item(selected, 3).setText("")
                self.points.item(selected, 6).setText("picked in photo")
                self._refresh_point_pixels()
            finally:
                self._loading = False
            self._set_mode("")
            self._store_points()
            self._draw_overlay()

    def _update_status(self) -> None:
        if self._mode == "corners":
            text = (
                f"Click an inner corner of the rig ({self._corner_index + 1}/4), in "
                "any order. Zoom with the mouse wheel; the right button drags the "
                "view."
            )
            self.status.setText(text)
            self._host_hint(text)
            return
        if self._mode == "point":
            text = "Click the injection point in the photograph."
            self.status.setText(text)
            self._host_hint(text)
            return
        self._host_hint("")
        missing = [
            point["location_id"] or "point"
            for point in self.current_points()
            if point["x_m"] is None or point["y_m"] is None
        ]
        if missing:
            self.status.setText(
                "No rig coordinates yet for " + ", ".join(missing) + ": define the rig "
                "frame (corners, width and height)."
            )
        else:
            self.status.setText("")

    # --------------------------------------------------------------- overlay

    def _scene_point(self, row: float, col: float) -> QPointF | None:
        if self._photo_size is None:
            return None
        bounds = self.canvas.pixmap_item.boundingRect()
        if bounds.width() <= 1 or bounds.height() <= 1:
            return None
        width, height = self._photo_size
        if (
            abs(bounds.width() / bounds.height() - width / height)
            > 0.01 * width / height
        ):
            return None
        return QPointF(
            col / max(1.0, width - 1.0) * (bounds.width() - 1.0),
            row / max(1.0, height - 1.0) * (bounds.height() - 1.0),
        )

    def _draw_overlay(self) -> None:
        scene = self.canvas.scene()
        for item in self._overlay:
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
        self._overlay = []
        if self.canvas.preview is None:
            return
        bounds = self.canvas.pixmap_item.boundingRect()
        size = max(4.0, 0.006 * max(bounds.width(), bounds.height()))
        frame = self._frame_from_ui() or self.frame
        if frame is not None:
            pen = QPen(QColor(64, 224, 208))
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            points = [
                self._scene_point(*frame.corners[name])
                for name in CLICK_ORDER
                if name in frame.corners
            ]
            points = [point for point in points if point is not None]
            if len(points) >= 2:
                polygon = QGraphicsPolygonItem(QPolygonF(points))
                polygon.setPen(pen)
                polygon.setZValue(20)
                scene.addItem(polygon)
                self._overlay.append(polygon)
            for name in CLICK_ORDER:
                if name not in frame.corners:
                    continue
                point = self._scene_point(*frame.corners[name])
                if point is None:
                    continue
                dot = QGraphicsEllipseItem(
                    point.x() - size / 2, point.y() - size / 2, size, size
                )
                dot.setPen(pen)
                dot.setZValue(21)
                scene.addItem(dot)
                self._overlay.append(dot)
        marker_pen = QPen(QColor(238, 74, 83))
        marker_pen.setCosmetic(True)
        marker_pen.setWidth(2)
        for row in range(self.points.rowCount()):
            row_text = (
                self.points.item(row, 4).text() if self.points.item(row, 4) else ""
            )
            col_text = (
                self.points.item(row, 5).text() if self.points.item(row, 5) else ""
            )
            r, c = _number(row_text), _number(col_text)
            if r is None or c is None:
                continue
            point = self._scene_point(r, c)
            if point is None:
                continue
            ring = QGraphicsEllipseItem(
                point.x() - size, point.y() - size, 2 * size, 2 * size
            )
            ring.setPen(marker_pen)
            ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ring.setZValue(22)
            scene.addItem(ring)
            label = QGraphicsSimpleTextItem(self.points.item(row, 0).text())
            label.setBrush(QBrush(QColor(248, 92, 99)))
            label.setPos(point.x() + 1.3 * size, point.y() - 2 * size)
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            label.setZValue(22)
            scene.addItem(label)
            self._overlay += [ring, label]


__all__ = ["RigFramePanel"]
