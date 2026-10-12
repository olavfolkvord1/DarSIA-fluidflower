"""Simple view of the Geometry and facies tab for registered series.

Four steps sit beside the raw baseline photograph of the reference run:

1. Rig frame and injection points (the rig frame panel).
2. Paint layers: a few brush strokes per layer on the photograph.
3. Propose boundaries: a pixel classifier trained on the strokes labels the
   rest of the photograph inside the rig frame; paint corrections and propose
   again.
4. Facies and export: group layers into facies with a role, porosity and
   permeability, then save a geometry revision and export it for native Setup.

The full geometry-and-facies page remains available as the advanced view.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import (
    RigFrame,
    SeriesRunInputs,
    default_actor,
    export_native_setup,
    photo_size,
    rig_frame_from_inputs,
    saved_rig_frames,
)

STROKE_OPACITY = 0.75
_BRUSH_STEPS = 100
_BRUSH_MIN_FRACTION = 0.0005
_BRUSH_MAX_FRACTION = 0.05
_BRUSH_DEFAULT_FRACTION = 0.006


def _muted(widget: QWidget) -> QColor:
    palette = widget.palette()
    foreground = palette.color(QPalette.ColorRole.WindowText)
    background = palette.color(QPalette.ColorRole.Window)
    return QColor(
        int(0.55 * foreground.red() + 0.45 * background.red()),
        int(0.55 * foreground.green() + 0.45 * background.green()),
        int(0.55 * foreground.blue() + 0.45 * background.blue()),
    )


def _state_color(widget: QWidget, state: str) -> QColor:
    """Status colours that stay legible in light and dark themes."""

    base = {
        "done": QColor(46, 139, 87),
        "warn": QColor(201, 138, 20),
        "error": QColor(196, 60, 50),
    }.get(state)
    if base is None:
        return _muted(widget)
    foreground = widget.palette().color(QPalette.ColorRole.WindowText)
    return QColor(
        int(0.7 * base.red() + 0.3 * foreground.red()),
        int(0.7 * base.green() + 0.3 * foreground.green()),
        int(0.7 * base.blue() + 0.3 * foreground.blue()),
    )


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("secondaryText")
    return label


def _swatch(color: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def _parse_number(text: str) -> float | None:
    cleaned = str(text).replace(",", ".").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _fit_table_height(table: QTableWidget, *, max_rows: int = 8) -> None:
    """Make a table as tall as its rows (up to ``max_rows``)."""

    rows = min(table.rowCount(), max_rows)
    height = table.horizontalHeader().height() + 2 * table.frameWidth() + 2
    for row in range(rows):
        height += table.rowHeight(row)
    if table.rowCount() == 0:
        height += 24
    table.setFixedHeight(height)


# ---------------------------------------------------------------------------
# Canvas


class LayerCanvas(QGraphicsView):
    """Photograph with painted strokes and the proposed layers on top.

    The canvas also serves the rig frame panel: ``image_point_selected`` gives
    clicks in place mode as fractions of the photograph, ``preview`` identifies
    the photograph shown and ``set_model`` shows another one.
    """

    image_point_selected = Signal(float, float)
    stroke_finished = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.proposal_item = QGraphicsPixmapItem()
        self.proposal_item.setZValue(5.0)
        self.stroke_item = QGraphicsPixmapItem()
        self.stroke_item.setZValue(10.0)
        self.stroke_item.setOpacity(STROKE_OPACITY)
        self.active_item = QGraphicsPathItem()
        self.active_item.setZValue(12.0)
        self.active_item.setOpacity(STROKE_OPACITY)
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setZValue(30.0)
        cursor_pen = QPen(QColor(255, 255, 255, 230), 1.5)
        cursor_pen.setCosmetic(True)
        self.cursor_item.setPen(cursor_pen)
        self.cursor_item.hide()
        for item in (
            self.pixmap_item,
            self.proposal_item,
            self.stroke_item,
            self.active_item,
            self.cursor_item,
        ):
            self.scene().addItem(item)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QColor("#090b0d"))
        self.setMinimumSize(480, 360)

        self.preview: dict[str, Any] | None = None
        self.place_mode = False
        self.tool = "pan"
        self.paint_enabled = False
        self.layer_id = 0
        self.layer_color = QColor("#ffffff")
        self.radius_px = 20.0
        self._raw_size: tuple[int, int] | None = None
        self._display_shape: tuple[int, int] | None = None
        self._stroke_image: QImage | None = None
        self._points: list[tuple[float, float]] = []
        self._path: QPainterPath | None = None
        self._pan_origin: QPointF | None = None
        self._user_zoomed = False

    # ----------------------------------------------------------- geometry

    def _scale(self) -> tuple[float, float]:
        """Display pixels per raw pixel along rows and columns."""

        if self._raw_size is None or self._display_shape is None:
            return 1.0, 1.0
        return (
            self._display_shape[0] / self._raw_size[1],
            self._display_shape[1] / self._raw_size[0],
        )

    def raw_to_scene(self, row: float, col: float) -> QPointF:
        sy, sx = self._scale()
        return QPointF((col + 0.5) * sx - 0.5, (row + 0.5) * sy - 0.5)

    def scene_to_raw(self, point: QPointF) -> tuple[float, float]:
        sy, sx = self._scale()
        return (point.y() + 0.5) / sy - 0.5, (point.x() + 0.5) / sx - 0.5

    def display_radius(self, radius_px: float | None = None) -> float:
        sy, sx = self._scale()
        return (
            float(radius_px if radius_px is not None else self.radius_px)
            * 0.5
            * (sx + sy)
        )

    @property
    def raw_size(self) -> tuple[int, int] | None:
        return self._raw_size

    # ------------------------------------------------------------- photo

    def set_photo(
        self,
        display: np.ndarray,
        raw_size: tuple[int, int],
        *,
        token: dict[str, Any],
        reset_view: bool,
    ) -> None:
        """Show an RGB uint8 image that represents a raw photograph."""

        image = np.ascontiguousarray(display, dtype=np.uint8)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        resized = self._display_shape != (height, width)
        self._display_shape = (height, width)
        self._raw_size = (int(raw_size[0]), int(raw_size[1]))
        self.preview = token
        if resized or self._stroke_image is None:
            self._stroke_image = QImage(
                width, height, QImage.Format.Format_ARGB32_Premultiplied
            )
            self._stroke_image.fill(Qt.GlobalColor.transparent)
            self.stroke_item.setPixmap(QPixmap())
            self.proposal_item.setPixmap(QPixmap())
        self.scene().setSceneRect(QRectF(0.0, 0.0, float(width), float(height)))
        if reset_view:
            self.fit_image()

    def set_model(self, preview: Mapping[str, Any], _config: Mapping[str, Any]) -> None:
        """Show ``preview["baseline"]`` (float RGB in [0, 1] or uint8)."""

        rgb = np.asarray(preview["baseline"])
        if rgb.dtype != np.uint8:
            rgb = np.rint(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        size = preview.get("raw_size") or (rgb.shape[1], rgb.shape[0])
        self.set_photo(rgb, size, token=dict(preview), reset_view=True)

    def clear_photo(self) -> None:
        """Show nothing (a run without a readable baseline photograph)."""

        for item in (self.pixmap_item, self.proposal_item, self.stroke_item):
            item.setPixmap(QPixmap())
        self._cancel_stroke()
        self._display_shape = None
        self._raw_size = None
        self._stroke_image = None
        self.preview = None
        self.paint_enabled = False
        self.cursor_item.hide()

    def fit_image(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False
        self._update_cursor()

    def set_overlays_visible(self, strokes: bool, proposal: bool) -> None:
        self.stroke_item.setVisible(strokes)
        self.active_item.setVisible(strokes)
        self.proposal_item.setVisible(proposal)

    def set_proposal(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.proposal_item.setPixmap(QPixmap())
            return
        image = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGBA8888,
        ).copy()
        self.proposal_item.setPixmap(QPixmap.fromImage(qimage))

    # ----------------------------------------------------------- strokes

    def _stroke_pen(self, stroke: lp.Stroke, color: QColor) -> QPen:
        width = max(1.0, 2.0 * self.display_radius(stroke.radius_px))
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    def _paint(
        self, painter: QPainter, stroke: lp.Stroke, colors: Mapping[int, QColor]
    ) -> None:
        if stroke.layer_id == lp.ERASER:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            color = QColor(0, 0, 0, 255)
        else:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            color = QColor(colors.get(stroke.layer_id, QColor("#ffffff")))
            color.setAlpha(255)
        points = [self.raw_to_scene(row, col) for row, col in stroke.points]
        if not points:
            return
        if len(points) == 1:
            radius = max(0.5, self.display_radius(stroke.radius_px))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(points[0], radius, radius)
            return
        painter.setPen(self._stroke_pen(stroke, color))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)

    def render_strokes(
        self, strokes: Sequence[lp.Stroke], colors: Mapping[int, QColor]
    ) -> None:
        if self._stroke_image is None:
            return
        self._stroke_image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for stroke in strokes:
            self._paint(painter, stroke, colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    def add_stroke_to_image(
        self, stroke: lp.Stroke, colors: Mapping[int, QColor]
    ) -> None:
        if self._stroke_image is None:
            return
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint(painter, stroke, colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    # ------------------------------------------------------------ tools

    def set_tool(self, tool: str) -> None:
        if tool not in {"pan", "paint", "erase"}:
            raise ValueError(f"Unknown tool {tool!r}")
        self.tool = tool
        self._cancel_stroke()
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.OpenHandCursor
            if tool == "pan"
            else Qt.CursorShape.CrossCursor
        )
        self._update_cursor()

    def _painting(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool in {"paint", "erase"}
            and self._raw_size is not None
        )

    def _update_cursor(self, scene_point: QPointF | None = None) -> None:
        if not self._painting():
            self.cursor_item.hide()
            return
        if scene_point is None:
            position = self.viewport().mapFromGlobal(self.cursor().pos())
            if not self.viewport().rect().contains(position):
                self.cursor_item.hide()
                return
            scene_point = self.mapToScene(position)
        radius = max(0.5, self.display_radius())
        self.cursor_item.setRect(
            QRectF(
                scene_point.x() - radius,
                scene_point.y() - radius,
                2 * radius,
                2 * radius,
            )
        )
        self.cursor_item.show()

    def _cancel_stroke(self) -> None:
        self._points = []
        self._path = None
        self.active_item.setPath(QPainterPath())

    def _extend(self, scene_point: QPointF) -> None:
        raw = self.scene_to_raw(scene_point)
        if self._points:
            previous = self.raw_to_scene(*self._points[-1])
            if (scene_point - previous).manhattanLength() < 1.0:
                return
        self._points.append(raw)
        if self._path is None:
            self._path = QPainterPath(scene_point)
        else:
            self._path.lineTo(scene_point)
        path = QPainterPath(self._path)
        if len(self._points) == 1:
            radius = max(0.5, self.display_radius())
            path.addEllipse(scene_point, radius, radius)
        self.active_item.setPath(path)

    def _start_stroke(self, scene_point: QPointF) -> None:
        self._cancel_stroke()
        color = (
            QColor(255, 255, 255) if self.tool == "erase" else QColor(self.layer_color)
        )
        pen = QPen(color, max(1.0, 2.0 * self.display_radius()))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if self.tool == "erase":
            pen.setStyle(Qt.PenStyle.SolidLine)
            color.setAlpha(120)
            pen.setColor(color)
        self.active_item.setPen(pen)
        self.active_item.setBrush(color)
        self._extend(scene_point)

    def _finish_stroke(self) -> None:
        if not self._points:
            self._cancel_stroke()
            return
        stroke = lp.Stroke(
            lp.ERASER if self.tool == "erase" else int(self.layer_id),
            float(self.radius_px),
            list(self._points),
        )
        self._cancel_stroke()
        self.stroke_finished.emit(stroke)

    # ------------------------------------------------------------ events

    def wheelEvent(self, event: Any) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)
        self._user_zoomed = True
        self._update_cursor(self.mapToScene(event.position().toPoint()))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_image()

    def mousePressEvent(self, event: Any) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        button = event.button()
        if button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_origin = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self.place_mode:
            point = self.mapToScene(event.position().toPoint())
            bounds = self.pixmap_item.boundingRect()
            if bounds.contains(point):
                self.image_point_selected.emit(
                    float(point.x() / max(1.0, bounds.width() - 1.0)),
                    float(point.y() / max(1.0, bounds.height() - 1.0)),
                )
                event.accept()
                return
        if button == Qt.MouseButton.LeftButton and self._painting():
            point = self.mapToScene(event.position().toPoint())
            if self.pixmap_item.boundingRect().contains(point):
                self._start_stroke(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(round(delta.x()))
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(round(delta.y()))
            )
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        if self._points and self._painting():
            bounds = self.pixmap_item.boundingRect()
            clamped = QPointF(
                min(max(point.x(), 0.0), bounds.width() - 1.0),
                min(max(point.y(), 0.0), bounds.height() - 1.0),
            )
            self._extend(clamped)
            self._update_cursor(point)
            event.accept()
            return
        self._update_cursor(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        button = event.button()
        if self._pan_origin is not None and button in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._pan_origin = None
            self.set_tool(self.tool)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._points:
            self._finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.cursor_item.hide()
        super().leaveEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        event.accept()


# ---------------------------------------------------------------------------
# Step sections


class StepSection(QFrame):
    """A numbered step with a header that folds its content."""

    def __init__(
        self, number: int, title: str, content: QWidget, *, expanded: bool = True
    ) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setText(f"{number}   {title}")
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)
        self.toggle.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle)
        header.addStretch(1)
        self.status = QLabel()
        header.addWidget(self.status)
        layout.addLayout(header)
        self.content = content
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.blockSignals(True)
        self.toggle.setChecked(bool(expanded))
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(bool(expanded))

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_status(self, text: str, state: str = "") -> None:
        self.status.setText(text)
        color = _state_color(self, state).name()
        self.status.setStyleSheet(f"color: {color};")


# ---------------------------------------------------------------------------
# Workers


class _ProposalWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        image: lp.PaintingImage,
        session: lp.PaintingSession,
        frame: RigFrame | None,
        destination: Path,
    ) -> None:
        super().__init__()
        self.image = image
        self.session = session
        self.frame = frame
        self.destination = destination

    @Slot()
    def run(self) -> None:
        try:
            proposal = lp.propose_layers(
                self.image, self.session, frame=self.frame, progress=self.progress.emit
            )
        except ValueError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        try:
            proposal.save(self.destination)
        except OSError as error:
            self.progress.emit(f"The proposal could not be saved: {error}")
        self.finished.emit(proposal)


class _SaveWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        project: Any,
        session: lp.PaintingSession,
        proposal: lp.LayerProposal,
        reference: SeriesRunInputs,
        frame: RigFrame | None,
        actor: str,
    ) -> None:
        super().__init__()
        self.project = project
        self.session = session
        self.proposal = proposal
        self.reference = reference
        self.frame = frame
        self.actor = actor

    @Slot()
    def run(self) -> None:
        try:
            lp.save_painted_revision(
                self.project,
                self.session,
                self.proposal,
                self.reference,
                frame=self.frame,
                actor=self.actor,
                progress=self.progress.emit,
            )
            self.progress.emit("Exporting for native Setup…")
            export = export_native_setup(
                self.project, reference=self.reference, actor=self.actor, replace=True
            )
        except ValueError as error:
            self.failed.emit(str(error))
            return
        except Exception as error:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.finished.emit(export)


# ---------------------------------------------------------------------------
# View


class LayerPainterView(QWidget):
    """Paint layers, propose boundaries, group facies and export."""

    exported = Signal(object)
    changed = Signal()
    # True while a proposal or a save runs; False when it has finished.
    busy_changed = Signal(bool)

    def __init__(
        self,
        project: Any,
        reference: Callable[[], SeriesRunInputs | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("seriesLayerView")
        self.project = project
        self._reference = reference
        self.session: lp.PaintingSession | None = None
        self.image: lp.PaintingImage | None = None
        self.proposal: lp.LayerProposal | None = None
        self.last_export: Any = None
        self._redo: list[lp.Stroke] = []
        self._dirty = False
        self._export_stale = True
        self._export_key = ""
        self._loading = False
        self._other_photo: Path | None = None
        self._reference_token: dict[str, Any] | None = None
        self._plain: np.ndarray | None = None
        self._enhanced: np.ndarray | None = None
        self._proposal_thread: QThread | None = None
        self._proposal_worker: _ProposalWorker | None = None
        self._save_thread: QThread | None = None
        self._save_worker: _SaveWorker | None = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self.save_session)

        self.canvas = LayerCanvas()
        self.canvas.stroke_finished.connect(self._stroke_finished)

        from darsia.fluidflower.series_rig_frame_panel import RigFramePanel

        self.rig_panel = RigFramePanel(self, project, reference, scrollable=False)
        self.rig_panel.changed.connect(self._frame_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_steps())
        splitter.addWidget(self._build_canvas_column())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 1000])
        self.splitter = splitter
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._install_shortcuts()
        application = QApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.wait_for_workers)
        self.reload()

    # ------------------------------------------------------------ layout

    def _build_steps(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(470)
        content = QWidget()
        scroll.setWidget(content)
        self.steps_scroll = scroll
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(
            _note(
                "Mark the layers on the raw baseline photograph of the reference run. "
                "Paint, propose, correct and propose again until the boundaries fit; "
                "then group the layers into facies and export them."
            )
        )
        self.step_frame = StepSection(
            1, "Rig frame and injection points", self.rig_panel, expanded=False
        )
        self.step_paint = StepSection(2, "Paint layers", self._build_paint_step())
        self.step_propose = StepSection(
            3, "Propose boundaries", self._build_propose_step()
        )
        self.step_export = StepSection(
            4, "Facies and export", self._build_export_step()
        )
        for section in (
            self.step_frame,
            self.step_paint,
            self.step_propose,
            self.step_export,
        ):
            layout.addWidget(section)
        layout.addStretch(1)
        return scroll

    def _build_paint_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "Choose a layer and paint a few strokes inside it: three to five per "
                "layer, spread over the layer. Start where the layers are easy to "
                "tell apart and correct the rest after the first proposal. Paint the "
                "ColorChecker, tape and fittings as Excluded."
            )
        )
        self.layer_table = QTableWidget(0, 3)
        self.layer_table.setHorizontalHeaderLabels(["", "Layer", "Strokes"])
        self.layer_table.verticalHeader().setVisible(False)
        self.layer_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.layer_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.layer_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        header = self.layer_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 28)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.layer_table.itemSelectionChanged.connect(self._layer_selected)
        self.layer_table.itemChanged.connect(self._layer_item_changed)
        self.layer_table.cellDoubleClicked.connect(self._layer_double_clicked)
        layout.addWidget(self.layer_table)
        buttons = QHBoxLayout()
        self.add_layer_button = QPushButton("Add layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.remove_layer_button = QPushButton("Remove layer")
        self.remove_layer_button.clicked.connect(lambda: self.remove_layer())
        self.clear_strokes_button = QPushButton("Clear its strokes")
        self.clear_strokes_button.clicked.connect(lambda: self.clear_layer_strokes())
        for button in (
            self.add_layer_button,
            self.remove_layer_button,
            self.clear_strokes_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return widget

    def _build_propose_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "A model learns from the painted pixels and labels the rest of the "
                "photograph inside the rig frame. Where it is wrong, paint a short "
                "stroke of the right layer there and propose again. Painted pixels "
                "keep their layer."
            )
        )
        row = QHBoxLayout()
        self.propose_button = QPushButton("Propose boundaries")
        self.propose_button.setToolTip("Shortcut: P")
        self.propose_button.clicked.connect(self.propose)
        row.addWidget(self.propose_button)
        self.propose_busy = QProgressBar()
        self.propose_busy.setRange(0, 0)
        self.propose_busy.setMaximumWidth(110)
        self.propose_busy.setTextVisible(False)
        self.propose_busy.hide()
        row.addWidget(self.propose_busy)
        self.propose_status = QLabel()
        self.propose_status.setWordWrap(True)
        row.addWidget(self.propose_status, 1)
        layout.addLayout(row)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Smoothing"))
        self.smoothing = QComboBox()
        for key, label in (("low", "Low"), ("medium", "Medium"), ("high", "High")):
            self.smoothing.addItem(label, key)
        self.smoothing.setToolTip("How strongly the proposed boundaries are smoothed")
        self.smoothing.currentIndexChanged.connect(self._settings_changed)
        settings.addWidget(self.smoothing)
        settings.addSpacing(10)
        settings.addWidget(QLabel("Merge regions under"))
        self.min_region = QDoubleSpinBox()
        self.min_region.setRange(0.0, 500.0)
        self.min_region.setDecimals(1)
        self.min_region.setSingleStep(0.5)
        self.min_region.setSuffix(" cm²")
        self.min_region.setToolTip(
            "Islands smaller than this join the surrounding layer, unless they "
            "contain painted pixels. Without the rig size, the area is taken "
            "relative to a 1 m² window."
        )
        self.min_region.editingFinished.connect(self._settings_changed)
        settings.addWidget(self.min_region)
        settings.addStretch(1)
        layout.addLayout(settings)
        self.proposal_summary = QLabel()
        self.proposal_summary.setWordWrap(True)
        layout.addWidget(self.proposal_summary)
        return widget

    def _build_export_step(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            _note(
                "Group the layers into facies: layers in the same facies share role, "
                "porosity and permeability. Pick a facies or type a new name."
            )
        )
        self.grouping_table = QTableWidget(0, 2)
        self.grouping_table.setHorizontalHeaderLabels(["Layer", "Facies"])
        self.grouping_table.verticalHeader().setVisible(False)
        self.grouping_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        header = self.grouping_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.grouping_table)
        layout.addWidget(
            _note(
                "Facies properties. Free water and excluded regions get no "
                "properties; everything outside the rig frame is added as a "
                "facies of its own."
            )
        )
        self.facies_table = QTableWidget(0, 4)
        self.facies_table.verticalHeader().setVisible(False)
        self.facies_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.facies_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.facies_table.itemChanged.connect(self._facies_item_changed)
        layout.addWidget(self.facies_table)
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Permeability unit"))
        self.unit = QComboBox()
        for value in lp.PERMEABILITY_UNITS:
            self.unit.addItem(value, value)
        self.unit.currentIndexChanged.connect(self._unit_changed)
        unit_row.addWidget(self.unit)
        unit_row.addStretch(1)
        layout.addLayout(unit_row)
        self.export_problems = QLabel()
        self.export_problems.setWordWrap(True)
        layout.addWidget(self.export_problems)
        row = QHBoxLayout()
        self.export_button = QPushButton("Save and export to native Setup…")
        self.export_button.setToolTip(
            "Save the proposed layers as a geometry revision on the raw photograph's "
            "grid and write the colored label image, facies table and [labeling] "
            "and [facies] sections"
        )
        self.export_button.clicked.connect(self.save_and_export)
        row.addWidget(self.export_button)
        self.export_busy = QProgressBar()
        self.export_busy.setRange(0, 0)
        self.export_busy.setMaximumWidth(110)
        self.export_busy.setTextVisible(False)
        self.export_busy.hide()
        row.addWidget(self.export_busy)
        row.addStretch(1)
        layout.addLayout(row)
        self.export_status = QLabel()
        self.export_status.setWordWrap(True)
        self.export_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.export_status)
        self.open_export_button = QPushButton("Open export folder")
        self.open_export_button.clicked.connect(self._open_export_folder)
        open_row = QHBoxLayout()
        open_row.addWidget(self.open_export_button)
        open_row.addStretch(1)
        layout.addLayout(open_row)
        return widget

    def _build_canvas_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons: dict[str, QToolButton] = {}
        for key, label, tip in (
            ("pan", "Pan", "Drag to move the photograph (H)"),
            ("paint", "Paint", "Paint the selected layer (B)"),
            ("erase", "Erase", "Erase strokes (E)"),
        ):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, name=key: self.set_tool(name))
            self.tool_group.addButton(button)
            self.tool_buttons[key] = button
            tools.addWidget(button)
        self.tool_group.setExclusive(True)
        tools.addSpacing(8)
        tools.addWidget(QLabel("Brush"))
        self.brush_size = QSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(0, _BRUSH_STEPS)
        self.brush_size.setFixedWidth(110)
        self.brush_size.setToolTip("Brush size ([ and ])")
        self.brush_size.valueChanged.connect(self._brush_changed)
        tools.addWidget(self.brush_size)
        self.brush_label = QLabel()
        self.brush_label.setMinimumWidth(70)
        tools.addWidget(self.brush_label)
        self.undo_button = QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("Undo the last stroke (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        tools.addWidget(self.undo_button)
        self.redo_button = QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo)
        tools.addWidget(self.redo_button)
        tools.addStretch(1)
        fit = QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Show the whole photograph (F)")
        fit.clicked.connect(self.canvas.fit_image)
        tools.addWidget(fit)
        layout.addLayout(tools)

        display = QHBoxLayout()
        display.addWidget(QLabel("Show"))
        self.show_strokes = QCheckBox("Strokes")
        self.show_strokes.setChecked(True)
        self.show_proposal = QCheckBox("Proposal")
        self.show_proposal.setChecked(True)
        self.show_uncertain = QCheckBox("Uncertain areas")
        self.show_uncertain.setToolTip(
            "Hatch pixels where the model is unsure; paint strokes there first"
        )
        for box in (self.show_strokes, self.show_proposal, self.show_uncertain):
            box.toggled.connect(self._display_changed)
            display.addWidget(box)
        display.addSpacing(8)
        display.addWidget(QLabel("Opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(90)
        self.opacity.setToolTip("Opacity of the proposal")
        self.opacity.valueChanged.connect(self._display_changed)
        display.addWidget(self.opacity)
        display.addSpacing(8)
        self.enhance = QCheckBox("Enhance contrast")
        self.enhance.setChecked(True)
        self.enhance.setToolTip(
            "Stretch the colors of the rig for display; the model always uses "
            "the original photograph"
        )
        self.enhance.toggled.connect(self._enhance_toggled)
        display.addWidget(self.enhance)
        display.addStretch(1)

        self.photo_note_row = QWidget()
        note_layout = QHBoxLayout(self.photo_note_row)
        note_layout.setContentsMargins(0, 0, 0, 0)
        self.photo_note = QLabel()
        self.photo_note.setWordWrap(True)
        note_layout.addWidget(self.photo_note, 1)
        back = QPushButton("Back to the reference photo")
        back.clicked.connect(self.show_reference_photo)
        note_layout.addWidget(back)
        self.photo_note_row.hide()
        layout.addWidget(self.photo_note_row)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(display)
        self.canvas_hint = _note(
            "Wheel: zoom · right or middle button: pan · Ctrl+Z / Ctrl+Y: undo / "
            "redo. On the photo: B paint · E erase · H pan · [ ] brush size · "
            "1-9 layer · P propose · F fit"
        )
        layout.addWidget(self.canvas_hint)
        return column

    def _install_shortcuts(self) -> None:
        """Single keys act on the photograph only, so that typing in a table
        or field is never taken as a shortcut; undo and redo work in the view."""

        def add(
            sequence: str | QKeySequence,
            slot: Callable[[], Any],
            parent: QWidget,
            context: Qt.ShortcutContext,
        ) -> None:
            shortcut = QShortcut(QKeySequence(sequence), parent)
            shortcut.setContext(context)
            shortcut.activated.connect(slot)

        canvas_only = Qt.ShortcutContext.WidgetShortcut
        for key, slot in (
            ("B", lambda: self.set_tool("paint")),
            ("E", lambda: self.set_tool("erase")),
            ("H", lambda: self.set_tool("pan")),
            ("F", self.canvas.fit_image),
            ("P", self.propose),
            ("[", lambda: self._step_brush(-6)),
            ("]", lambda: self._step_brush(6)),
        ):
            add(key, slot, self.canvas, canvas_only)
        for number in range(1, 10):
            add(
                str(number),
                lambda index=number - 1: self.select_layer_row(index),
                self.canvas,
                canvas_only,
            )
        # One shortcut per action with all platform keys, so that no key is
        # registered twice (an ambiguous shortcut never fires).
        for standard, slot, extra in (
            (QKeySequence.StandardKey.Undo, self.undo, ()),
            (QKeySequence.StandardKey.Redo, self.redo, ("Ctrl+Y",)),
        ):
            keys = list(QKeySequence.keyBindings(standard))
            for text in extra:
                if QKeySequence(text) not in keys:
                    keys.append(QKeySequence(text))
            shortcut = QShortcut(self)
            shortcut.setKeys(keys)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

    # ------------------------------------------------------------- state

    def reference(self) -> SeriesRunInputs | None:
        return self._reference()

    def current_frame(self) -> RigFrame | None:
        frame = self.rig_panel.frame
        if frame is None:
            reference = self.reference()
            if reference is not None:
                frame = saved_rig_frames(self.project).get(
                    reference.run_id
                ) or rig_frame_from_inputs(reference)
        return frame if frame is not None and frame.complete() else None

    def busy(self) -> bool:
        for thread in (self._proposal_thread, self._save_thread):
            try:
                if thread is not None and thread.isRunning():
                    return True
            except RuntimeError:
                continue
        return False

    def wait_for_workers(self) -> None:
        """Let a running proposal or save finish (before closing)."""

        for thread in (self._proposal_thread, self._save_thread):
            try:
                if thread is not None and thread.isRunning():
                    thread.wait()
            except RuntimeError:  # already deleted
                continue
        try:
            self.save_session()
        except RuntimeError:  # the view is being deleted
            pass

    def reload(self) -> None:
        """Load the session, photograph and proposal of the reference run."""

        self.save_session()
        self._loading = True
        try:
            self.session = None
            self.image = None
            self.proposal = None
            self._redo = []
            self._other_photo = None
            self._reference_token = None
            reference = self.reference()
            self.rig_panel.load()
            if reference is None or reference.baseline_image is None:
                self.canvas.clear_photo()
                self.propose_status.setText(
                    "The reference run has no baseline photograph."
                )
            else:
                try:
                    self.session = lp.load_painting_session(self.project, reference)
                except (OSError, ValueError, KeyError) as error:
                    QMessageBox.warning(
                        self,
                        "Could not read the painted layers",
                        f"{error}\n\nStarting with the default layers.",
                    )
                    self.session = lp.PaintingSession.default(
                        reference.run_id, reference.baseline_image
                    )
                self._load_reference_photo(reference)
                proposal = lp.load_proposal(self.project, reference.run_id)
                if (
                    proposal is not None
                    and self.image is not None
                    and tuple(proposal.image_size) == tuple(self.image.image_size)
                ):
                    self.proposal = proposal
            self._sync_settings()
            if self.session is not None and self.session.photo_changed:
                self.propose_status.setText(
                    "The baseline photograph of this run has changed since the "
                    "strokes were painted; check that they still fit."
                )
            self.last_export = None
            self._refresh_layers()
            self._refresh_facies_table()
            self._render_strokes()
            self._render_proposal()
            self._refresh_export_info()
            self._refresh_status()
            frame_done = (
                self.current_frame() is not None
                and self.current_frame().has_size()  # type: ignore[union-attr]
            )
            self.step_frame.set_expanded(not frame_done)
            self.set_tool("paint" if self.session is not None else "pan")
        finally:
            self._loading = False

    def _load_reference_photo(self, reference: SeriesRunInputs) -> None:
        assert reference.baseline_image is not None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image = lp.load_painting_image(Path(reference.baseline_image))
        except (OSError, ValueError) as error:
            self.image = None
            self.canvas.clear_photo()
            self.propose_status.setText(
                f"Could not read the baseline photograph: {error}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        session = self.session
        assert session is not None
        if session.image_size is None:
            session.image_size = image.image_size
        if not session.image_path:
            session.image_path = str(reference.baseline_image)
        self.image = image
        self._plain = np.rint(255.0 * np.clip(image.display, 0.0, 1.0)).astype(np.uint8)
        self._reference_token = {
            "baseline": self._plain,
            "path": str(image.path),
            "raw_size": image.image_size,
            "reference": True,
        }
        self._update_enhanced()
        self._show_reference(reset_view=True)

    def _update_enhanced(self) -> None:
        if self.image is None:
            self._enhanced = None
            return
        frame = self.current_frame()
        shape = self.image.display.shape[:2]
        scale = (
            shape[0] / self.image.image_size[1],
            shape[1] / self.image.image_size[0],
        )
        if frame is not None:
            mask = lp.frame_mask(frame, shape, scale)
        else:
            # Without a frame, leave out the border where the rig frame usually is.
            mask = np.zeros(shape, dtype=bool)
            rows, cols = int(0.12 * shape[0]), int(0.08 * shape[1])
            mask[rows : shape[0] - rows, cols : shape[1] - cols] = True
        self._enhanced = lp.enhance_contrast(self.image.display, mask)

    def _show_reference(self, *, reset_view: bool) -> None:
        if self.image is None or self._reference_token is None:
            return
        display = self._enhanced if self.enhance.isChecked() else self._plain
        assert display is not None
        self.canvas.set_photo(
            display,
            self.image.image_size,
            token=self._reference_token,
            reset_view=reset_view,
        )
        self.canvas.paint_enabled = True
        self._other_photo = None
        self.rig_panel.use_host_photo(self.image.image_size, self.canvas.preview)
        self.photo_note_row.hide()
        self._render_strokes()
        self._render_proposal()
        self._display_changed()

    def show_reference_photo(self) -> None:
        self._show_reference(reset_view=False)

    def show_frame_photo(self, path: Path) -> bool:
        """Show a raw photograph for the rig frame panel."""

        if self.image is not None and Path(path) == Path(self.image.path):
            if self.canvas.preview is not self._reference_token:
                self._show_reference(reset_view=False)
            return True
        from darsia.fluidflower.tools.injection_point_detection import load_reduced

        try:
            width, height = photo_size(Path(path))
            factor = max(1, int(math.ceil(max(width, height) / lp.DISPLAY_LIMIT_PX)))
            rgb = load_reduced(Path(path), factor)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not show the photograph", str(error))
            return False
        plain = np.rint(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        display = lp.enhance_contrast(rgb) if self.enhance.isChecked() else plain
        token = {"baseline": plain, "path": str(path), "raw_size": (width, height)}
        self.canvas.set_photo(display, (width, height), token=token, reset_view=True)
        self.canvas.paint_enabled = False
        self.canvas.set_overlays_visible(False, False)
        self._other_photo = Path(path)
        self.photo_note.setText(
            f"Showing {Path(path).name}. Painting works on the reference photo."
        )
        self.photo_note_row.show()
        return True

    def _sync_settings(self) -> None:
        session = self.session
        widgets = (self.smoothing, self.min_region, self.unit)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if session is not None:
                self.smoothing.setCurrentIndex(
                    max(0, self.smoothing.findData(session.smoothing))
                )
                self.min_region.setValue(float(session.min_region_cm2))
                self.unit.setCurrentIndex(
                    max(0, self.unit.findData(session.permeability_unit))
                )
            self._brush_from_session()
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        for widget in (
            self.propose_button,
            self.add_layer_button,
            self.remove_layer_button,
            self.clear_strokes_button,
            self.smoothing,
            self.min_region,
            self.unit,
        ):
            widget.setEnabled(session is not None)

    # ----------------------------------------------------------- saving

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._export_stale = True
        if not self._loading:
            self._save_timer.start()
        self.changed.emit()

    def save_session(self) -> None:
        self._save_timer.stop()
        if not self._dirty or self.session is None:
            return
        try:
            lp.save_painting_session(self.project, self.session, actor=default_actor())
        except OSError as error:
            self.propose_status.setText(f"The strokes could not be saved: {error}")
            return
        self._dirty = False

    def hideEvent(self, event: Any) -> None:
        self.save_session()
        super().hideEvent(event)

    # ------------------------------------------------------------- tools

    def set_tool(self, tool: str) -> None:
        if tool in {"paint", "erase"} and self._other_photo is not None:
            self.show_reference_photo()
        self.canvas.set_tool(tool)
        button = self.tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)

    def _brush_limits(self) -> tuple[float, float]:
        width = float(self.canvas.raw_size[0]) if self.canvas.raw_size else 2000.0
        return (
            max(1.0, _BRUSH_MIN_FRACTION * width),
            max(4.0, _BRUSH_MAX_FRACTION * width),
        )

    def _radius_from_slider(self, value: int) -> float:
        low, high = self._brush_limits()
        return low * (high / low) ** (value / _BRUSH_STEPS)

    def _slider_from_radius(self, radius: float) -> int:
        low, high = self._brush_limits()
        radius = min(max(radius, low), high)
        return int(round(_BRUSH_STEPS * math.log(radius / low) / math.log(high / low)))

    def _brush_from_session(self) -> None:
        if self.image is None:
            return
        radius = _BRUSH_DEFAULT_FRACTION * self.image.image_size[0]
        if self.session is not None and self.session.strokes:
            radius = self.session.strokes[-1].radius_px
        self.brush_size.blockSignals(True)
        self.brush_size.setValue(self._slider_from_radius(radius))
        self.brush_size.blockSignals(False)
        self._brush_changed(self.brush_size.value())

    def _brush_changed(self, value: int) -> None:
        radius = self._radius_from_slider(int(value))
        self.canvas.radius_px = radius
        per_metre = lp.frame_pixels_per_metre(self.current_frame())
        if per_metre:
            self.brush_label.setText(f"Ø {2000.0 * radius / per_metre:.0f} mm")
        else:
            self.brush_label.setText(f"Ø {2.0 * radius:.0f} px")
        self.canvas._update_cursor()

    def _step_brush(self, steps: int) -> None:
        self.brush_size.setValue(self.brush_size.value() + steps)

    # ------------------------------------------------------------ layers

    def _layer_colors(self) -> dict[int, QColor]:
        if self.session is None:
            return {}
        return {row.layer_id: QColor(row.color) for row in self.session.layers}

    def _refresh_layers(self) -> None:
        table = self.layer_table
        table.blockSignals(True)
        try:
            current = self.canvas.layer_id
            layers = self.session.layers if self.session is not None else []
            counts = self.session.stroke_counts() if self.session is not None else {}
            table.setRowCount(len(layers))
            selected_row = -1
            for row, layer in enumerate(layers):
                swatch = QTableWidgetItem()
                swatch.setBackground(QColor(layer.color))
                # Not selectable, so that the selection does not hide the color.
                swatch.setFlags(Qt.ItemFlag.ItemIsEnabled)
                swatch.setToolTip("Double-click to change the color")
                swatch.setData(Qt.ItemDataRole.UserRole, layer.layer_id)
                table.setItem(row, 0, swatch)
                name = QTableWidgetItem(layer.name)
                name.setToolTip(f"Key {row + 1}" if row < 9 else "")
                table.setItem(row, 1, name)
                count = QTableWidgetItem(str(counts.get(layer.layer_id, 0)))
                count.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, 2, count)
                if layer.layer_id == current:
                    selected_row = row
            if selected_row < 0 and layers:
                selected_row = 0
            if selected_row >= 0:
                table.selectRow(selected_row)
                self._use_layer(layers[selected_row].layer_id)
        finally:
            table.blockSignals(False)
        _fit_table_height(table)

    def _use_layer(self, layer_id: int) -> None:
        layer = self.session.layer(layer_id) if self.session is not None else None
        if layer is None:
            return
        self.canvas.layer_id = layer.layer_id
        self.canvas.layer_color = QColor(layer.color)

    def selected_layer_id(self) -> int | None:
        rows = {index.row() for index in self.layer_table.selectedIndexes()}
        if not rows or self.session is None:
            return None
        row = min(rows)
        if row >= len(self.session.layers):
            return None
        return self.session.layers[row].layer_id

    def select_layer_row(self, row: int) -> None:
        if self.session is None or not 0 <= row < len(self.session.layers):
            return
        self.layer_table.selectRow(row)

    def _layer_selected(self) -> None:
        layer_id = self.selected_layer_id()
        if layer_id is None:
            return
        self._use_layer(layer_id)
        if self.canvas.tool == "pan" and not self._loading:
            self.set_tool("paint")

    def _layer_item_changed(self, item: QTableWidgetItem) -> None:
        if self.session is None or item.column() != 1:
            return
        layer = (
            self.session.layers[item.row()]
            if item.row() < len(self.session.layers)
            else None
        )
        if layer is None:
            return
        try:
            self.session.rename_layer(layer.layer_id, item.text())
        except ValueError as error:
            QMessageBox.information(self, "Layer name in use", str(error))
        self._refresh_layers()
        self._refresh_facies_table()
        self._render_proposal()
        self._mark_dirty()
        self._refresh_status()

    def _layer_double_clicked(self, row: int, column: int) -> None:
        if column != 0 or self.session is None or row >= len(self.session.layers):
            return
        layer = self.session.layers[row]
        color = QColorDialog.getColor(
            QColor(layer.color), self, f"Color of {layer.name}"
        )
        if color.isValid():
            self.set_layer_color(layer.layer_id, color.name())

    def set_layer_color(self, layer_id: int, color: str) -> None:
        layer = self.session.layer(layer_id) if self.session is not None else None
        if layer is None:
            return
        layer.color = str(color)
        self._refresh_layers()
        self._refresh_facies_table()
        self._render_strokes()
        self._render_proposal()
        self._mark_dirty()

    def add_layer(self) -> None:
        if self.session is None:
            return
        try:
            layer = self.session.add_layer()
        except ValueError as error:
            QMessageBox.information(self, "Cannot add a layer", str(error))
            return
        self.canvas.layer_id = layer.layer_id
        self._refresh_layers()
        self._refresh_facies_table()
        self._mark_dirty()
        self._refresh_status()
        row = len(self.session.layers) - 1
        self.layer_table.editItem(self.layer_table.item(row, 1))

    def remove_layer(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        assert layer is not None
        strokes = self.session.stroke_counts().get(layer_id, 0)
        if confirm and strokes:
            answer = QMessageBox.question(
                self,
                "Remove layer",
                f"Remove {layer.name} and its {strokes} stroke(s)?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session.remove_layer(layer_id)
        self._redo = []
        self.canvas.layer_id = (
            self.session.layers[0].layer_id if self.session.layers else 0
        )
        self._after_stroke_change()
        self._refresh_facies_table()

    def clear_layer_strokes(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        count = self.session.stroke_counts().get(layer_id, 0)
        if not count or layer is None:
            return
        if confirm:
            answer = QMessageBox.question(
                self, "Clear strokes", f"Remove the {count} stroke(s) of {layer.name}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session.clear_layer_strokes(layer_id)
        self._redo = []
        self._after_stroke_change()

    # ----------------------------------------------------------- strokes

    def _stroke_finished(self, stroke: lp.Stroke) -> None:
        if self.session is None:
            return
        try:
            self.session.add_stroke(stroke)
        except ValueError:
            self._render_strokes()
            return
        self.canvas.add_stroke_to_image(stroke, self._layer_colors())
        self._redo = []
        self._after_stroke_change(render=False)

    def add_stroke(self, stroke: lp.Stroke) -> None:
        """Add a stroke as if it had been painted (used by scripts and tests)."""

        self._stroke_finished(stroke)

    def undo(self) -> None:
        if self.session is None or not self.session.strokes:
            return
        self._redo.append(self.session.strokes.pop())
        self._after_stroke_change()

    def redo(self) -> None:
        if self.session is None or not self._redo:
            return
        self.session.strokes.append(self._redo.pop())
        self._after_stroke_change()

    def _after_stroke_change(self, *, render: bool = True) -> None:
        if render:
            self._render_strokes()
        self._refresh_layers()
        self._mark_dirty()
        self._refresh_status()

    def _render_strokes(self) -> None:
        if self.session is None:
            self.canvas.render_strokes([], {})
            return
        self.canvas.render_strokes(self.session.strokes, self._layer_colors())

    # ---------------------------------------------------------- proposal

    def _settings_changed(self, *_args: Any) -> None:
        if self.session is None or self._loading:
            return
        self.session.smoothing = str(self.smoothing.currentData() or "medium")
        self.session.min_region_cm2 = float(self.min_region.value())
        self._mark_dirty()
        self._refresh_status()

    def proposal_current(self) -> bool:
        return (
            self.proposal is not None
            and self.session is not None
            and self.proposal.digest
            == lp.proposal_digest(self.session, self.current_frame())
        )

    def propose(self) -> None:
        if self.session is None or self.image is None or self.busy():
            return
        reference = self.reference()
        if reference is None:
            return
        if self._other_photo is not None:
            self.show_reference_photo()
        self.save_session()
        snapshot = lp.PaintingSession.from_dict(self.session.to_dict())
        thread = QThread(self)
        worker = _ProposalWorker(
            self.image,
            snapshot,
            self.current_frame(),
            lp.proposal_path(self.project, reference.run_id),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.propose_status.setText)
        worker.finished.connect(self._proposal_ready)
        worker.failed.connect(self._proposal_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._proposal_thread_finished)
        self._proposal_thread = thread
        self._proposal_worker = worker
        self.propose_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.propose_busy.show()
        self.propose_status.setText("Starting…")
        self.step_propose.set_status("Working…", "warn")
        thread.start()
        self.busy_changed.emit(True)

    @Slot(object)
    def _proposal_ready(self, proposal: lp.LayerProposal) -> None:
        worker = self._proposal_worker
        if (
            worker is not None
            and self.session is not None
            and worker.session.run_id != self.session.run_id
        ):
            return  # the reference run changed while the model worked
        self.proposal = proposal
        self.propose_status.setText(f"Proposed in {proposal.elapsed_s:.1f} s.")
        if not self.show_proposal.isChecked():
            self.show_proposal.setChecked(True)
        self._render_proposal()
        self._refresh_facies_table()
        self.step_export.set_expanded(True)

    @Slot(str)
    def _proposal_failed(self, message: str) -> None:
        worker = self._proposal_worker
        if (
            worker is not None
            and self.session is not None
            and worker.session.run_id != self.session.run_id
        ):
            return
        self.propose_status.setText(message)
        QMessageBox.information(self, "No proposal", message)

    def _proposal_thread_finished(self) -> None:
        thread = self._proposal_thread
        self._proposal_thread = None
        self._proposal_worker = None
        if thread is not None:
            thread.deleteLater()
        self.propose_busy.hide()
        self.propose_button.setEnabled(self.session is not None)
        self._refresh_status()
        self.busy_changed.emit(self.busy())

    def _render_proposal(self) -> None:
        if (
            self.proposal is None
            or self.image is None
            or self.session is None
            or self._other_photo is not None
            or tuple(self.proposal.image_size) != tuple(self.image.image_size)
        ):
            self.canvas.set_proposal(None)
            return
        import cv2

        height, width = self.image.display.shape[:2]
        classes = cv2.resize(
            self.proposal.classes, (width, height), interpolation=cv2.INTER_NEAREST
        )
        alpha = int(round(2.55 * self.opacity.value()))
        table = np.zeros((256, 4), dtype=np.uint8)
        table[lp.OUTSIDE] = (0, 0, 0, 110)
        for layer in self.session.layers:
            color = QColor(layer.color)
            table[layer.layer_id] = (color.red(), color.green(), color.blue(), alpha)
        rgba = table[classes]
        boundary = np.zeros(classes.shape, dtype=bool)
        boundary[:, 1:] |= classes[:, 1:] != classes[:, :-1]
        boundary[1:, :] |= classes[1:, :] != classes[:-1, :]
        rgba[boundary] = (255, 255, 255, 235)
        if self.show_uncertain.isChecked():
            uncertain = cv2.resize(
                self.proposal.uncertain().astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            rows, cols = np.indices(classes.shape)
            hatch = ((rows + cols) // 3) % 2 == 0
            rgba[uncertain & hatch] = (255, 0, 255, 200)
        self.canvas.set_proposal(rgba)

    def _display_changed(self, *_args: Any) -> None:
        reference_shown = self._other_photo is None
        self.canvas.set_overlays_visible(
            self.show_strokes.isChecked() and reference_shown,
            self.show_proposal.isChecked() and reference_shown,
        )
        sender = self.sender()
        if sender in (self.opacity, self.show_uncertain):
            self._render_proposal()

    def _enhance_toggled(self, _checked: bool) -> None:
        if self._other_photo is not None:
            self.show_frame_photo(self._other_photo)
        else:
            self._show_reference(reset_view=False)

    def _frame_changed(self) -> None:
        self._update_enhanced()
        if self._other_photo is None and self.enhance.isChecked():
            self._show_reference(reset_view=False)
        self._brush_changed(self.brush_size.value())
        self._refresh_status()

    # ------------------------------------------------------------ facies

    @staticmethod
    def _clear_cell_widgets(table: QTableWidget) -> None:
        """Remove cell widgets now; Qt would only delete them later."""

        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                widget = table.cellWidget(row, column)
                if widget is not None:
                    widget.hide()
                    table.removeCellWidget(row, column)

    def _refresh_facies_table(self) -> None:
        """Rebuild the layer-to-facies table and the facies properties."""

        session = self.session
        layers = session.layers if session is not None else []
        present = self._present_layers()
        names = session.facies_order() if session is not None else []

        grouping = self.grouping_table
        grouping.blockSignals(True)
        try:
            self._clear_cell_widgets(grouping)
            grouping.setRowCount(len(layers))
            for row, layer in enumerate(layers):
                label = QTableWidgetItem(_swatch(layer.color), layer.name)
                label.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if present is not None and layer.layer_id not in present:
                    label.setToolTip("Not in the current proposal")
                    label.setForeground(_muted(self))
                grouping.setItem(row, 0, label)
                box = QComboBox()
                box.setEditable(True)
                box.addItems(names)
                box.setCurrentText(layer.facies)
                box.setToolTip(
                    "Layers with the same facies share role and properties; type a "
                    "new name for a facies of its own"
                )
                box.lineEdit().editingFinished.connect(
                    lambda combo=box, lid=layer.layer_id: self._facies_edited(
                        combo, lid
                    )
                )
                box.activated.connect(
                    lambda _index, combo=box, lid=layer.layer_id: self._facies_edited(
                        combo, lid
                    )
                )
                grouping.setCellWidget(row, 1, box)
        finally:
            grouping.blockSignals(False)
        _fit_table_height(grouping)

        table = self.facies_table
        table.blockSignals(True)
        try:
            unit = str(self.unit.currentData() or "darcy")
            table.setHorizontalHeaderLabels(
                ["Facies", "Role", "Porosity [-]", f"Permeability [{unit}]"]
            )
            self._clear_cell_widgets(table)
            table.setRowCount(len(names))
            for row, name in enumerate(names):
                assert session is not None
                entry = session.facies.get(name, lp.FaciesEntry())
                members = session.layers_in_facies(name)
                item = QTableWidgetItem(
                    _swatch(members[0].color) if members else QIcon(), name
                )
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setToolTip(
                    "Layers: "
                    + ", ".join(layer.name for layer in members)
                    + ". Double-click to rename the facies."
                )
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
                table.setItem(row, 0, item)
                role = QComboBox()
                for key, text in lp.LAYER_ROLES.items():
                    role.addItem(text, key)
                role.setCurrentIndex(max(0, role.findData(entry.role)))
                role.setToolTip(
                    "Porous medium needs porosity and permeability; free water and "
                    "excluded regions (ColorChecker, tape, fittings) get none"
                )
                role.currentIndexChanged.connect(
                    lambda _index, combo=role, facies=name: self._role_edited(
                        combo, facies
                    )
                )
                table.setCellWidget(row, 1, role)
                porous = entry.role == "porous_medium"
                for column, value in ((2, entry.porosity), (3, entry.permeability)):
                    cell = QTableWidgetItem(_format_number(value) if porous else "—")
                    cell.setData(Qt.ItemDataRole.UserRole, name)
                    if porous:
                        cell.setFlags(
                            Qt.ItemFlag.ItemIsEnabled
                            | Qt.ItemFlag.ItemIsSelectable
                            | Qt.ItemFlag.ItemIsEditable
                        )
                        invalid = (
                            value is None
                            or (column == 2 and not 0.0 < value <= 1.0)
                            or (column == 3 and value <= 0.0)
                        )
                        if invalid:
                            cell.setBackground(QColor(201, 138, 20, 70))
                            cell.setToolTip(
                                "Porosity between 0 and 1"
                                if column == 2
                                else "A positive permeability"
                            )
                    else:
                        cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        cell.setForeground(_muted(self))
                    table.setItem(row, column, cell)
        finally:
            table.blockSignals(False)
        _fit_table_height(table)
        self._refresh_status()

    def _present_layers(self) -> set[int] | None:
        if self.proposal is None:
            return None
        return {
            int(row["layer_id"]) for row in self.proposal.layers if row.get("pixels")
        }

    def _facies_edited(self, box: QComboBox, layer_id: int) -> None:
        try:
            text = box.currentText()
        except RuntimeError:  # the table was rebuilt
            return
        self.set_layer_facies(layer_id, text)

    def _role_edited(self, box: QComboBox, facies: str) -> None:
        try:
            role = str(box.currentData())
        except RuntimeError:
            return
        self.set_facies_role(facies, role)

    def set_layer_facies(self, layer_id: int, facies: str) -> None:
        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if (
            layer is None
            or not str(facies).strip()
            or str(facies).strip() == layer.facies
        ):
            return
        self.session.set_layer_facies(layer_id, facies)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def rename_facies(self, old: str, new: str) -> None:
        if self.session is None or not str(new).strip() or old == str(new).strip():
            return
        self.session.rename_facies(old, new)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def set_facies_role(self, facies: str, role: str) -> None:
        if self.session is None or facies not in self.session.facies:
            return
        if self.session.facies[facies].role == role:
            return
        self.session.set_facies(facies, role=role)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def set_facies_properties(
        self,
        facies: str,
        *,
        porosity: float | None = None,
        permeability: float | None = None,
    ) -> None:
        if self.session is None:
            return
        self.session.set_facies(facies, porosity=porosity, permeability=permeability)
        self._mark_dirty()
        self._refresh_facies_table()

    def _facies_item_changed(self, item: QTableWidgetItem) -> None:
        if self.session is None:
            return
        facies = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if facies not in self.session.facies:
            return
        if item.column() == 0:
            self.rename_facies(facies, item.text())
            return
        if item.column() not in (2, 3):
            return
        value = _parse_number(item.text())
        if item.column() == 2:
            self.session.set_facies(facies, porosity=value)
        else:
            self.session.set_facies(facies, permeability=value)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def _unit_changed(self, _index: int) -> None:
        if self.session is None or self._loading:
            return
        self.session.permeability_unit = str(self.unit.currentData() or "darcy")
        self._mark_dirty()
        self._refresh_facies_table()

    # ------------------------------------------------------------ export

    def export_problems_now(self) -> list[str]:
        if self.session is None:
            return ["Choose a reference run with a baseline photograph."]
        return lp.check_ready_for_revision(
            self.session, self.proposal, self.current_frame(), self.reference()
        )

    def save_and_export(self) -> None:
        if self.busy() or self.session is None or self.proposal is None:
            if self.proposal is None:
                QMessageBox.information(
                    self, "Not ready to export", "Propose the boundaries first."
                )
            return
        problems = self.export_problems_now()
        if problems:
            QMessageBox.information(self, "Not ready to export", "\n".join(problems))
            return
        reference = self.reference()
        assert reference is not None
        self.save_session()
        thread = QThread(self)
        worker = _SaveWorker(
            self.project,
            lp.PaintingSession.from_dict(self.session.to_dict()),
            self.proposal,
            reference,
            self.current_frame(),
            default_actor(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.export_status.setText)
        worker.finished.connect(self._export_ready)
        worker.failed.connect(self._export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._save_thread_finished)
        self._save_thread = thread
        self._save_worker = worker
        self.export_button.setEnabled(False)
        self.propose_button.setEnabled(False)
        # The revision is written from a snapshot; the frame and points are
        # also written by the save, so they wait until it has finished.
        self.rig_panel.setEnabled(False)
        self._export_key = self._export_state_key()
        self.export_busy.show()
        self.export_status.setText("Starting…")
        self.step_export.set_status("Working…", "warn")
        thread.start()
        self.busy_changed.emit(True)

    @Slot(object)
    def _export_ready(self, export: Any) -> None:
        worker = self._save_worker
        if (
            worker is None
            or self.session is None
            or worker.session.run_id == self.session.run_id
        ):
            self.last_export = export
            # Edits made while the revision was written are not in it.
            self._export_stale = self._export_state_key() != self._export_key
            self._refresh_export_info()
        self.exported.emit(export)

    @Slot(str)
    def _export_failed(self, message: str) -> None:
        self.export_status.setText(message)
        QMessageBox.warning(self, "Could not save and export", message)

    def _save_thread_finished(self) -> None:
        thread = self._save_thread
        self._save_thread = None
        self._save_worker = None
        if thread is not None:
            thread.deleteLater()
        self.export_busy.hide()
        self.propose_button.setEnabled(self.session is not None)
        self.rig_panel.setEnabled(True)
        self._refresh_status()
        self.busy_changed.emit(self.busy())

    def _export_state_key(self) -> str:
        """Everything a saved revision depends on."""

        import hashlib
        import json

        if self.session is None:
            return ""
        payload = self.session.to_dict()
        payload.pop("updated_utc", None)
        payload.pop("updated_by", None)
        payload["proposal"] = self.proposal.digest if self.proposal else ""
        payload["points"] = self.rig_panel.current_points()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _refresh_export_info(self) -> None:
        export = self.last_export
        if export is not None and not self._current_revision_is(export.revision_id):
            # A later revision was saved (for example in the advanced view).
            self.last_export = export = None
            self._export_stale = True
        if export is not None:
            self.export_status.setText(
                f"Saved revision {str(export.revision_id)[:8]} and exported "
                f"{export.region_count} regions in {export.facies_count} facies to "
                f"{export.destination}."
            )
            self.open_export_button.setEnabled(True)
            return
        info = self._latest_painted_revision()
        if info:
            self.export_status.setText(info)
        else:
            self.export_status.setText("")
        self.open_export_button.setEnabled(self._export_folder() is not None)

    def _current_revision_is(self, revision_id: str) -> bool:
        from darsia.fluidflower.geometry_facies import GeometryRevisionStore

        try:
            current = GeometryRevisionStore(self.project).current()
        except (OSError, ValueError):
            return True
        return str(current.get("revision_id") or "") == str(revision_id)

    def _latest_painted_revision(self) -> str:
        from darsia.fluidflower.geometry_facies import (
            GEOMETRY_REVISION_FORMAT,
            GeometryRevisionStore,
        )

        reference = self.reference()
        try:
            current = GeometryRevisionStore(self.project).current()
        except (OSError, ValueError):
            return ""
        if current.get("format") != GEOMETRY_REVISION_FORMAT:
            return ""
        generation = dict(
            (current.get("configuration") or {}).get("facies_generation") or {}
        )
        if generation.get("method") != lp.PROPOSAL_METHOD:
            return "The current revision was saved in the advanced view."
        run = str(generation.get("reference_run") or "")
        when = str(current.get("recorded_utc") or "")[:16].replace("T", " ")
        text = f"Current revision {str(current.get('revision_id'))[:8]} ({when} UTC)"
        if reference is not None and run and run != reference.run_id:
            text += f" was painted on run {run}."
        else:
            text += " was saved from these layers."
        return text

    def _export_folder(self) -> Path | None:
        if self.last_export is not None:
            return Path(self.last_export.destination)
        root = Path(self.project.root) / "exports" / "native_setup"
        try:
            folders = [
                path
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ]
        except OSError:
            return None
        if not folders:
            return None
        return max(folders, key=lambda path: path.stat().st_mtime)

    def _open_export_folder(self) -> None:
        folder = self._export_folder()
        if folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------ status

    def _refresh_status(self) -> None:
        session = self.session
        frame = self.current_frame()
        points = self.rig_panel.current_points()
        located = sum(
            1
            for point in points
            if point["x_m"] is not None and point["y_m"] is not None
        )
        point_text = f" · {located} injection point(s)" if located else ""
        if frame is None:
            self.step_frame.set_status("No rig frame" + point_text, "warn")
        elif not frame.has_size():
            self.step_frame.set_status("Enter the inner size" + point_text, "warn")
        else:
            self.step_frame.set_status("Frame set" + point_text, "done")

        if session is None:
            self.step_paint.set_status("No photograph", "warn")
            self.step_propose.set_status("", "")
            self.step_export.set_status("", "")
            self.export_problems.setText("")
            self.export_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            return
        counts = session.stroke_counts()
        painted = [layer_id for layer_id in counts if layer_id != lp.ERASER]
        total = sum(counts.values())
        self.step_paint.set_status(
            f"{len(painted)} of {len(session.layers)} layers painted · {total} stroke(s)",
            "done" if len(painted) >= 2 else "warn",
        )
        self.undo_button.setEnabled(bool(session.strokes))
        self.redo_button.setEnabled(bool(self._redo))

        busy = self.busy()
        if self._proposal_thread is not None:
            self.step_propose.set_status("Working…", "warn")
        elif self.proposal is None:
            self.step_propose.set_status("Not proposed yet", "warn")
            self.proposal_summary.setText("")
        elif self.proposal_current():
            self.step_propose.set_status("Up to date", "done")
        else:
            self.step_propose.set_status("Out of date: propose again", "warn")
        if self.proposal is not None:
            self.proposal_summary.setText(self._proposal_text())

        problems = self.export_problems_now()
        if self._save_thread is not None:
            self.step_export.set_status("Working…", "warn")
        elif problems:
            self.step_export.set_status("Missing input", "warn")
        elif self.last_export is not None and not self._export_stale:
            self.step_export.set_status("Exported", "done")
        else:
            self.step_export.set_status("Ready to export", "done")
        self.export_problems.setText(" ".join(problems))
        self.export_problems.setStyleSheet(
            f"color: {_state_color(self, 'warn').name()};" if problems else ""
        )
        self.export_button.setEnabled(not busy and not problems)

    def _proposal_text(self) -> str:
        proposal = self.proposal
        session = self.session
        if proposal is None or session is None:
            return ""
        parts = []
        for row in proposal.layers:
            if not row.get("pixels"):
                continue
            layer = session.layer(int(row["layer_id"]))
            name = layer.name if layer is not None else str(row.get("name") or "")
            regions = int(row.get("regions") or 0)
            parts.append(f"{name}: {regions} region{'s' if regions != 1 else ''}")
        when = proposal.created_utc[11:16] if proposal.created_utc else ""
        text = (
            f"From {proposal.stroke_count} stroke(s)"
            + (f" at {when} UTC" if when else "")
            + ". "
            + " · ".join(parts)
            + "."
        )
        if proposal.merged_regions:
            text += f" {proposal.merged_regions} small island(s) merged."
        uncertain = int(np.count_nonzero(proposal.uncertain()))
        inside = int(np.count_nonzero(proposal.classes != lp.OUTSIDE))
        if inside:
            text += f" Uncertain: {100.0 * uncertain / inside:.1f} % of the rig."
        return text


__all__ = ["LayerCanvas", "LayerPainterView", "StepSection"]
