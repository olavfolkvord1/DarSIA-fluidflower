"""Simple view of the Geometry and facies tab for registered series.

Four steps sit beside the raw baseline photograph of the reference run:

1. Rig frame and injection points (the rig frame panel).
2. Mark layers: rough polygons (and brush strokes) per layer on the photograph.
3. Propose boundaries: a pixel classifier trained on the marks labels the rest
   of the photograph inside the rig frame and places the boundaries near the
   polygon edges; adjust polygons or paint corrections and propose again.
4. Facies and export: tick the layers of each facies and give the facies a
   role, porosity and permeability, then save a geometry revision and export it
   for native Setup.

The full geometry-and-facies page remains available as the advanced view.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from . import geometry_workspace_frame as workspace_frame
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSettings,
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
    QTransform,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.image_formats import image_family, photo_dialog_filter
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
# A polygon point is hit within this many screen pixels.
_VERTEX_HIT_PX = 8.0
# A right press that moves less than this (screen pixels) is a click.
_CLICK_SLOP_PX = 4.0
# Point handles of the Edit tool (screen pixels); selected points are larger.
_HANDLE_PX = 7.0
_SELECTED_HANDLE_PX = 11.0
# Screen pixels that an arrow key moves the selected points (with Shift).
_NUDGE_PX = 1.0
_NUDGE_SHIFT_PX = 10.0
# Arrow keys of the Edit tool: (rows, columns) per step.
_NUDGE_KEYS = {
    Qt.Key.Key_Left: (0, -1),
    Qt.Key.Key_Right: (0, 1),
    Qt.Key.Key_Up: (-1, 0),
    Qt.Key.Key_Down: (1, 0),
}
_TOO_FEW_POINTS = (
    "A polygon needs at least three points; use Delete polygon in its "
    "right-click menu."
)
_BRUSH_STEPS = 100
_BRUSH_MIN_FRACTION = 0.0005
_BRUSH_MAX_FRACTION = 0.05
_BRUSH_DEFAULT_FRACTION = 0.006
_UNDO_LIMIT = 200
# Column of the layer table whose box shows or hides a layer (display only).
_SHOW_COLUMN = 4
_SHOW_TIP = "Show or hide this layer in the photograph (display only)"
# Another photograph of the reference run shown under the marks (display only).
_BACKGROUND_TIP = (
    "Photograph shown under the marks. Choose another photo of the same run and "
    "camera position, for example one taken before the indicator was added. "
    "Display only: the proposal always uses the reference photo. T switches "
    "between the reference photo and the chosen photo."
)
_BACKGROUND_SETTINGS = ("DarSIA", "FluidFlower layer painting")
_BACKGROUND_CHOOSE = "choose"
# A session painted on the raw reference photo, in a project whose geometry
# is in the saved corrected rig frame, stays on the raw photo.
_RAW_SESSION_NOTE = (
    "Painting on the raw reference photo, where these marks were drawn. The "
    "project's saved corrected rig maps are not shown here."
)
_RAW_SESSION_EXPORT_NOTE = (
    "This painting is on the raw reference photo, while the project's geometry "
    "is in the saved corrected rig frame. Exporting it into the project is a "
    "separate step."
)
# Short role names for the facies table.
_ROLE_SHORT = {"porous_medium": "Porous", "free_water": "Free water"}
_CANVAS_HINT = (
    "Wheel: zoom · right or middle drag: pan · Ctrl+Z / Ctrl+Y: undo / redo. "
    "On the photo: L draw · V edit · B paint · E erase · H pan · [ ] brush "
    "size · 1-9 layer · P propose · F fit"
)
_DRAFT_HINT = (
    "Polygon with {count} point{plural}: left click adds a point · right click, "
    "Enter or a double click closes · Backspace removes the last point · Esc "
    "cancels"
)


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


def insert_point_on_nearest_edge(
    points: Sequence[tuple[float, float]], row: float, col: float
) -> tuple[list[tuple[float, float]], int]:
    """The points of a closed polygon with one point added on its nearest edge.

    Every edge counts, also the closing edge from the last point back to the
    first. The new point is where (row, col) projects at right angles onto the
    edge nearest to it, kept between the edge's two end points; it is inserted
    between those two points. Returns the new list of points and the index of
    the new point. With fewer than two points, or a position that is not a
    finite number, the points come back unchanged with index -1. ``points``
    itself is not changed.
    """

    vertices = [(float(point[0]), float(point[1])) for point in points]
    row, col = float(row), float(col)
    count = len(vertices)
    if count < 2 or not (math.isfinite(row) and math.isfinite(col)):
        return vertices, -1
    best: tuple[float, int, tuple[float, float]] | None = None
    for index in range(count):
        start = vertices[index]
        end = vertices[(index + 1) % count]
        delta_row, delta_col = end[0] - start[0], end[1] - start[1]
        length2 = delta_row * delta_row + delta_col * delta_col
        fraction = 0.0
        if length2 > 0.0:
            fraction = (
                (row - start[0]) * delta_row + (col - start[1]) * delta_col
            ) / length2
        if fraction <= 0.0:
            point = start
        elif fraction >= 1.0:
            point = end
        else:
            point = (start[0] + fraction * delta_row, start[1] + fraction * delta_col)
        distance = math.hypot(row - point[0], col - point[1])
        if best is None or distance < best[0]:
            best = (distance, index, point)
    assert best is not None
    _distance, edge, point = best
    vertices.insert(edge + 1, point)
    return vertices, edge + 1


def _segment_distance(point: QPointF, start: QPointF, end: QPointF) -> float:
    """Distance from a point to the segment between two points."""

    delta_x, delta_y = end.x() - start.x(), end.y() - start.y()
    length2 = delta_x * delta_x + delta_y * delta_y
    fraction = 0.0
    if length2 > 0.0:
        fraction = (
            (point.x() - start.x()) * delta_x + (point.y() - start.y()) * delta_y
        ) / length2
        fraction = min(max(fraction, 0.0), 1.0)
    return math.hypot(
        point.x() - (start.x() + fraction * delta_x),
        point.y() - (start.y() + fraction * delta_y),
    )


@dataclass
class _PointDrag:
    """Selected points being dragged with the Edit tool."""

    # Viewport position and raw (row, col) of the press.
    press: QPointF
    start: tuple[float, float]
    # Points of every polygon with selected points, before and while dragging.
    before: dict[int, list[tuple[float, float]]]
    after: dict[int, list[tuple[float, float]]]
    # The mouse has moved further than a click.
    moving: bool = False


# ---------------------------------------------------------------------------
# Canvas


class LayerCanvas(QGraphicsView):
    """Photograph with polygons, painted strokes and the proposed layers on top.

    The tools are ``pan``, ``polygon`` (the Draw button), ``edit``, ``paint``
    and ``erase``. The right or middle button drags the view in every tool.

    Draw (``polygon``) only draws new polygons: a left click adds a point, a
    right click (without dragging), Enter or a double click closes the
    polygon, Backspace removes the last point and Esc cancels it. Existing
    polygons show no point handles here, and a click on one of their points
    adds a point to the new polygon; it never moves the existing point.

    Edit (``edit``) changes existing polygons and never starts a polygon, a
    stroke or a draft. Every shown polygon has point handles, and selected
    points (``selected_points``) are drawn larger, white with a dark outline.
    A left press on a point selects only that point, or keeps the whole
    selection when the point is already selected; Ctrl+click adds the point
    to the selection or takes it out. Dragging moves all selected points by
    the same raw-pixel offset; they stop together at the edge of the
    photograph. A drag on empty space selects the points inside its rectangle
    (Ctrl adds them to the selection); a click there clears the selection.
    Delete or Backspace removes the selected points, but a polygon that would
    keep fewer than three points is left unchanged and ``notice`` says so.
    The arrow keys move the selected points by one screen pixel (Shift: ten)
    and Esc clears the selection. A double click on an edge adds a point there
    and selects it. Each of these changes is reported once, so that it is
    undone in one step: ``polygon_edited`` (index and new points) when one
    polygon changed, ``polygons_edited`` ({index: new points}) when several
    did.

    A right click on a polygon asks for its menu (``polygon_context``: the
    polygon, the point under the click or -1, and the screen position); with
    the Edit tool, a click near an edge counts too, also just outside the
    polygon. Just before that, ``last_context_raw`` is set to the raw
    (row, col) of the click, so that the menu can add a point to the polygon
    on its nearest edge.

    ``set_hidden_layers`` hides the polygons and strokes of some layers; this
    changes only what is shown. Polygons of hidden layers are not drawn, get
    no point handles and cannot be selected, dragged or right-clicked; hiding
    a layer drops its points from the selection. Strokes of hidden layers are
    left out of the stroke image; eraser strokes still apply to the strokes
    that are shown. ``set_overlays_visible`` (all marks, the proposal) works
    on top of this.

    The canvas also serves the rig frame panel: ``image_point_selected`` gives
    clicks in place mode as fractions of the photograph, ``preview`` identifies
    the photograph shown and ``set_model`` shows another one.

    ``set_photo`` may also get the photograph at a higher resolution
    (``detail``, normally its raw pixels). It is shown over the display copy
    whenever one of its pixels covers at least half a screen pixel, so that
    zooming in shows the photograph's own pixels; the display copy, which sets
    the scene grid for the marks, strokes and the proposal, is shown when
    zoomed out. Display only.
    """

    # The detail is shown from this many screen pixels per detail pixel.
    DETAIL_FROM = 0.5

    image_point_selected = Signal(float, float)
    stroke_finished = Signal(object)
    polygon_finished = Signal(object)
    # Index of an edited polygon and its new points (raw pixels).
    polygon_edited = Signal(int, object)
    # Several polygons changed in one step: {index: new points (raw pixels)}.
    polygons_edited = Signal(object)
    # Polygon index, point index (-1: not on a point) and global position.
    polygon_context = Signal(int, int, QPoint)
    # Points of the polygon being drawn (0: none).
    draft_changed = Signal(int)
    notice = Signal(str)

    TOOLS = ("pan", "polygon", "edit", "paint", "erase")

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        # The photograph at a higher resolution, over the display copy.
        self.detail_item = QGraphicsPixmapItem()
        self.detail_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.detail_item.setZValue(1.0)
        self.detail_item.hide()
        self.proposal_item = QGraphicsPixmapItem()
        self.proposal_item.setZValue(5.0)
        self.stroke_item = QGraphicsPixmapItem()
        self.stroke_item.setZValue(10.0)
        self.stroke_item.setOpacity(STROKE_OPACITY)
        self.active_item = QGraphicsPathItem()
        self.active_item.setZValue(12.0)
        self.active_item.setOpacity(STROKE_OPACITY)
        self.draft_item = QGraphicsPathItem()
        self.draft_item.setZValue(14.0)
        self.rubber_item = QGraphicsPathItem()
        self.rubber_item.setZValue(14.0)
        # The rectangle dragged on empty space to select points (Edit tool).
        self.band_item = QGraphicsRectItem()
        self.band_item.setZValue(25.0)
        band_pen = QPen(QColor(255, 255, 255, 230), 1.0, Qt.PenStyle.DashLine)
        band_pen.setCosmetic(True)
        self.band_item.setPen(band_pen)
        self.band_item.setBrush(QColor(255, 255, 255, 40))
        self.band_item.hide()
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setZValue(30.0)
        cursor_pen = QPen(QColor(255, 255, 255, 230), 1.5)
        cursor_pen.setCosmetic(True)
        self.cursor_item.setPen(cursor_pen)
        self.cursor_item.hide()
        for item in (
            self.pixmap_item,
            self.detail_item,
            self.proposal_item,
            self.stroke_item,
            self.active_item,
            self.draft_item,
            self.rubber_item,
            self.band_item,
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
        self._right_press: QPointF | None = None
        self._right_dragged = False
        self._user_zoomed = False
        self._marks_visible = True
        self._polygon_fill = True
        self._polygons: list[lp.Polygon] = []
        self._polygon_colors: dict[int, QColor] = {}
        self._polygon_items: list[QGraphicsPathItem] = []
        # Point handles of each polygon (Edit tool only).
        self._handle_items: list[list[QGraphicsRectItem]] = []
        self._draft: list[tuple[float, float]] = []
        self._draft_handles: list[QGraphicsRectItem] = []
        # Edit tool: the selected points as (polygon index, point index), a
        # drag of them in progress and where a rubber band started (viewport).
        self._selection: set[tuple[int, int]] = set()
        self._drag: _PointDrag | None = None
        self._band_origin: QPointF | None = None
        # Layers whose polygons and strokes are not shown (display only).
        self._hidden_layers: frozenset[int] = frozenset()
        # The strokes last rendered, to draw them again when layers are hidden.
        self._strokes: list[lp.Stroke] = []
        self._stroke_colors: dict[int, QColor] = {}
        # Raw (row, col) of the last right click that opened a polygon menu.
        self.last_context_raw: tuple[float, float] | None = None

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

    def _clamped_scene(self, point: QPointF) -> QPointF:
        bounds = self.pixmap_item.boundingRect()
        return QPointF(
            min(max(point.x(), 0.0), max(0.0, bounds.width() - 1.0)),
            min(max(point.y(), 0.0), max(0.0, bounds.height() - 1.0)),
        )

    # ------------------------------------------------------------- photo

    def set_photo(
        self,
        display: np.ndarray,
        raw_size: tuple[int, int],
        *,
        token: dict[str, Any],
        reset_view: bool,
        detail: np.ndarray | None = None,
    ) -> None:
        """Show an RGB uint8 image that represents a raw photograph.

        ``detail`` is the same photograph at a higher resolution, shown when
        zoomed in (see the class notes); ``None`` shows the display copy only.
        """

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
        self._set_detail(detail)
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
        self.refresh_detail()

    def _set_detail(self, detail: np.ndarray | None) -> None:
        """Place ``detail`` on the scene grid of the display copy."""

        if detail is None or self._display_shape is None:
            self.detail_item.setPixmap(QPixmap())
            self.detail_item.hide()
            return
        image = np.ascontiguousarray(detail, dtype=np.uint8)
        rows, cols = image.shape[:2]
        qimage = QImage(
            image.data, cols, rows, int(image.strides[0]), QImage.Format.Format_RGB888
        ).copy()
        self.detail_item.setPixmap(QPixmap.fromImage(qimage))
        # Each detail pixel centre lands where raw_to_scene puts that point of
        # the photograph, so the marks sit on the pixels they were drawn on.
        height, width = self._display_shape
        self.detail_item.setTransform(QTransform.fromScale(width / cols, height / rows))
        self.detail_item.setPos(-0.5, -0.5)

    @property
    def detail_shown(self) -> bool:
        """Whether the higher-resolution photograph is shown now."""

        return self.detail_item.isVisible()

    def refresh_detail(self) -> None:
        """Show the detail when zoomed in far enough, else the display copy."""

        pixmap = self.detail_item.pixmap()
        if pixmap.isNull() or self._display_shape is None:
            self.detail_item.hide()
            return
        scene_per_detail = self._display_shape[1] / pixmap.width()
        screen_per_scene = abs(self.transform().m11())
        self.detail_item.setVisible(
            screen_per_scene * scene_per_detail >= self.DETAIL_FROM
        )

    def set_model(self, preview: Mapping[str, Any], _config: Mapping[str, Any]) -> None:
        """Show ``preview["baseline"]`` (float RGB in [0, 1] or uint8)."""

        rgb = np.asarray(preview["baseline"])
        if rgb.dtype != np.uint8:
            rgb = np.rint(255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        size = preview.get("raw_size") or (rgb.shape[1], rgb.shape[0])
        self.set_photo(rgb, size, token=dict(preview), reset_view=True)

    def clear_photo(self) -> None:
        """Show nothing (a run without a readable baseline photograph)."""

        for item in (self.pixmap_item, self.detail_item, self.proposal_item, self.stroke_item):
            item.setPixmap(QPixmap())
        self.detail_item.hide()
        self._cancel_stroke()
        self.cancel_polygon()
        self._end_drag()
        self._cancel_band()
        self._selection = set()
        self.render_polygons([], {})
        self._strokes = []
        self._stroke_colors = {}
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
        self.refresh_detail()
        self._update_cursor()

    def set_overlays_visible(self, marks: bool, proposal: bool) -> None:
        """Show or hide the polygons and strokes, and the proposal."""

        self._marks_visible = bool(marks)
        if not marks:
            self.cancel_polygon()
            self._end_drag()
            self._cancel_band()
        self.stroke_item.setVisible(marks)
        self.active_item.setVisible(marks)
        for item, polygon in zip(self._polygon_items, self._polygons):
            item.setVisible(marks and polygon.layer_id not in self._hidden_layers)
        self._update_handles()
        self.proposal_item.setVisible(proposal)

    @property
    def hidden_layers(self) -> frozenset[int]:
        return self._hidden_layers

    def set_hidden_layers(self, layer_ids: Iterable[int]) -> None:
        """Hide the polygons and strokes of these layers; show the others.

        Display only: the marks themselves are not changed.
        """

        hidden = frozenset(int(value) for value in layer_ids)
        if hidden == self._hidden_layers:
            return
        self._hidden_layers = hidden
        if self._drag is not None and any(
            index < len(self._polygons) and self._polygons[index].layer_id in hidden
            for index in self._drag.before
        ):
            self._end_drag()
        # Points of hidden layers cannot stay selected.
        self._selection = {
            point for point in self._selection if self._selectable(*point)
        }
        for item, polygon in zip(self._polygon_items, self._polygons):
            item.setVisible(self._marks_visible and polygon.layer_id not in hidden)
        self._update_handles()
        self._draw_strokes()

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

    def _stroke_shown(self, stroke: lp.Stroke) -> bool:
        return (
            stroke.layer_id == lp.ERASER or stroke.layer_id not in self._hidden_layers
        )

    def _draw_strokes(self) -> None:
        """Draw the stored strokes, leaving out those of hidden layers."""

        if self._stroke_image is None:
            return
        self._stroke_image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for stroke in self._strokes:
            if self._stroke_shown(stroke):
                self._paint(painter, stroke, self._stroke_colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    def render_strokes(
        self, strokes: Sequence[lp.Stroke], colors: Mapping[int, QColor]
    ) -> None:
        self._strokes = list(strokes)
        self._stroke_colors = dict(colors)
        self._draw_strokes()

    def add_stroke_to_image(
        self, stroke: lp.Stroke, colors: Mapping[int, QColor]
    ) -> None:
        self._strokes.append(stroke)
        self._stroke_colors = dict(colors)
        if self._stroke_image is None or not self._stroke_shown(stroke):
            return
        painter = QPainter(self._stroke_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint(painter, stroke, colors)
        painter.end()
        self.stroke_item.setPixmap(QPixmap.fromImage(self._stroke_image))

    # ---------------------------------------------------------- polygons

    def _polygon_path(self, points: Sequence[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        if not points:
            return path
        path.moveTo(self.raw_to_scene(*points[0]))
        for row, col in points[1:]:
            path.lineTo(self.raw_to_scene(row, col))
        path.closeSubpath()
        return path

    def _style_polygon(self, item: QGraphicsPathItem, color: QColor) -> None:
        pen = QPen(QColor(color), 2.0)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        item.setPen(pen)
        if self._polygon_fill:
            fill = QColor(color)
            fill.setAlpha(55)
            item.setBrush(fill)
        else:
            item.setBrush(Qt.BrushStyle.NoBrush)

    def set_polygon_fill(self, filled: bool) -> None:
        """Fill the polygons lightly, or draw outlines only (over a proposal)."""

        if bool(filled) == self._polygon_fill:
            return
        self._polygon_fill = bool(filled)
        for item, polygon in zip(self._polygon_items, self._polygons):
            self._style_polygon(
                item, QColor(self._polygon_colors.get(polygon.layer_id, "#ffffff"))
            )

    def render_polygons(
        self, polygons: Sequence[lp.Polygon], colors: Mapping[int, QColor]
    ) -> None:
        """Draw the polygons; larger ones below smaller ones."""

        for item in self._polygon_items:
            self.scene().removeItem(item)
        self._polygon_items = []
        self._polygons = list(polygons)
        self._polygon_colors = dict(colors)
        drag = self._drag
        if drag is not None and any(
            index >= len(self._polygons) or list(self._polygons[index].points) != points
            for index, points in drag.before.items()
        ):
            # The polygons changed under the drag: it no longer applies.
            self._drag = drag = None
        # Selected points must still exist and be shown.
        self._selection = {
            point for point in self._selection if self._selectable(*point)
        }
        if self._display_shape is not None:
            areas = [polygon.area() for polygon in self._polygons]
            for index, polygon in enumerate(self._polygons):
                points = (
                    drag.after[index]
                    if drag is not None and index in drag.after
                    else polygon.points
                )
                item = QGraphicsPathItem(self._polygon_path(points))
                self._style_polygon(
                    item, QColor(colors.get(polygon.layer_id, QColor("#ffffff")))
                )
                # Smaller polygons win where layers overlap, so they lie on top.
                rank = sum(1 for other in areas if other > areas[index])
                item.setZValue(8.0 + min(rank, 999) * 1.0e-3)
                item.setVisible(
                    self._marks_visible and polygon.layer_id not in self._hidden_layers
                )
                self.scene().addItem(item)
                self._polygon_items.append(item)
        else:
            self._polygon_items = []
        self._update_handles()

    def _handle(
        self, point: QPointF, color: QColor, *, size: float = 7.0
    ) -> QGraphicsRectItem:
        item = QGraphicsRectItem(-size / 2.0, -size / 2.0, size, size)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setPos(point)
        pen = QPen(QColor(20, 20, 20, 230), 1.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QColor(color))
        item.setZValue(20.0)
        self.scene().addItem(item)
        return item

    def _style_handle(
        self, item: QGraphicsRectItem, color: QColor, selected: bool
    ) -> None:
        """A point handle in its layer's colour; larger and white when selected."""

        size = _SELECTED_HANDLE_PX if selected else _HANDLE_PX
        item.setRect(-size / 2.0, -size / 2.0, size, size)
        pen = QPen(
            QColor(20, 20, 20, 255 if selected else 230), 2.0 if selected else 1.0
        )
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QColor(255, 255, 255) if selected else QColor(color))
        item.setZValue(21.0 if selected else 20.0)

    def _update_handles(self) -> None:
        """Point handles for every shown polygon, with the Edit tool only."""

        for handles in self._handle_items:
            for item in handles:
                self.scene().removeItem(item)
        self._handle_items = []
        if (
            self.tool != "edit"
            or not self._marks_visible
            or not self.paint_enabled
            or self._display_shape is None
        ):
            return
        drag = self._drag
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                # No handles, but keep one list per polygon (indexed by polygon).
                self._handle_items.append([])
                continue
            points = (
                drag.after[index]
                if drag is not None and index in drag.after
                else polygon.points
            )
            color = QColor(self._polygon_colors.get(polygon.layer_id, "#ffffff"))
            handles = []
            for vertex, (row, col) in enumerate(points):
                item = self._handle(self.raw_to_scene(row, col), color)
                self._style_handle(item, color, (index, vertex) in self._selection)
                handles.append(item)
            self._handle_items.append(handles)

    def _restyle_handles(self, points: Iterable[tuple[int, int]]) -> None:
        """Show whether these points are selected, without making the handles
        again."""

        for index, vertex in points:
            if not (
                0 <= index < min(len(self._handle_items), len(self._polygons))
                and 0 <= vertex < len(self._handle_items[index])
            ):
                continue
            color = QColor(
                self._polygon_colors.get(self._polygons[index].layer_id, "#ffffff")
            )
            self._style_handle(
                self._handle_items[index][vertex],
                color,
                (index, vertex) in self._selection,
            )

    def _hit_vertex(self, position: QPointF) -> tuple[int, int] | None:
        """Polygon and point index of the point under a viewport position."""

        best: tuple[float, int, int] | None = None
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                continue
            for vertex, (row, col) in enumerate(polygon.points):
                spot = self.mapFromScene(self.raw_to_scene(row, col))
                distance = math.hypot(spot.x() - position.x(), spot.y() - position.y())
                if distance <= _VERTEX_HIT_PX and (best is None or distance < best[0]):
                    best = (distance, index, vertex)
        return None if best is None else (best[1], best[2])

    def _hit_polygon(self, position: QPointF) -> int | None:
        """The smallest shown polygon under a viewport position."""

        scene_point = self.mapToScene(position.toPoint())
        hits = [
            index
            for index, item in enumerate(self._polygon_items)
            if self._polygons[index].layer_id not in self._hidden_layers
            and item.path().contains(scene_point)
        ]
        if not hits:
            return None
        return min(hits, key=lambda index: self._polygons[index].area())

    def _hit_edge(self, position: QPointF) -> int | None:
        """The shown polygon with an edge within the point-hit distance of a
        viewport position: the nearest edge, and the smaller polygon where
        two edges are as near (a shared boundary)."""

        transform = self.viewportTransform()
        best: tuple[float, float, int] | None = None
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers or len(polygon.points) < 2:
                continue
            spots = [
                transform.map(self.raw_to_scene(row, col))
                for row, col in polygon.points
            ]
            distance = min(
                _segment_distance(position, start, spots[(number + 1) % len(spots)])
                for number, start in enumerate(spots)
            )
            if distance <= _VERTEX_HIT_PX:
                key = (distance, polygon.area(), index)
                if best is None or key < best:
                    best = key
        return None if best is None else best[2]

    def polygon_at(self, position: QPointF) -> tuple[int, int] | None:
        """Polygon (and point, or -1) under a viewport position.

        With the Edit tool, a position near an edge counts too, also just
        outside the polygon.
        """

        vertex = self._hit_vertex(position)
        if vertex is not None:
            return vertex
        index = self._hit_polygon(position)
        if index is None and self.tool == "edit":
            index = self._hit_edge(position)
        return None if index is None else (index, -1)

    # ---------------------------------------------------- selecting points

    @property
    def selected_points(self) -> list[tuple[int, int]]:
        """Points selected with the Edit tool: sorted (polygon, point) indices."""

        return sorted(self._selection)

    def _selectable(self, index: int, vertex: int) -> bool:
        return (
            0 <= index < len(self._polygons)
            and 0 <= vertex < len(self._polygons[index].points)
            and self._polygons[index].layer_id not in self._hidden_layers
        )

    def set_selection(self, points: Iterable[tuple[int, int]]) -> None:
        """Select these (polygon, point) indices instead of the current ones.

        Only the Edit tool has a selection; points of hidden layers and points
        that do not exist are left out.
        """

        chosen = {(int(index), int(vertex)) for index, vertex in points}
        if self.tool != "edit":
            chosen = set()
        before = self._selection
        self._selection = {point for point in chosen if self._selectable(*point)}
        self._restyle_handles(before ^ self._selection)

    def clear_selection(self) -> None:
        """Select no point; a drag or a rubber band in progress is cancelled."""

        self._cancel_drag()
        self._cancel_band()
        if self._selection:
            before, self._selection = self._selection, set()
            self._restyle_handles(before)

    def selection_without(
        self, index: int, vertices: Iterable[int]
    ) -> list[tuple[int, int]]:
        """The selection as it will be once these points of polygon ``index``
        are removed: they leave it, and later points of that polygon move up."""

        removed = {int(vertex) for vertex in vertices}
        kept = []
        for polygon, vertex in self._selection:
            if polygon != index:
                kept.append((polygon, vertex))
            elif vertex not in removed:
                earlier = sum(1 for number in removed if number < vertex)
                kept.append((polygon, vertex - earlier))
        return sorted(kept)

    def _selected_by_polygon(self) -> dict[int, set[int]]:
        grouped: dict[int, set[int]] = {}
        for index, vertex in self._selection:
            if self._selectable(index, vertex):
                grouped.setdefault(index, set()).add(vertex)
        return grouped

    def _emit_edits(self, changes: Mapping[int, list[tuple[float, float]]]) -> None:
        """Report changed polygons as one change: ``polygon_edited`` for one
        polygon, ``polygons_edited`` for several."""

        if len(changes) == 1:
            ((index, points),) = changes.items()
            self.polygon_edited.emit(int(index), list(points))
        elif changes:
            self.polygons_edited.emit(
                {int(index): list(points) for index, points in changes.items()}
            )

    def _raw_limits(self) -> tuple[float, float, float, float]:
        """First and last raw row and column that points are kept within (as
        for the points of a polygon being drawn)."""

        bounds = self.pixmap_item.boundingRect()
        top, left = self.scene_to_raw(QPointF(0.0, 0.0))
        bottom, right = self.scene_to_raw(
            QPointF(max(0.0, bounds.width() - 1.0), max(0.0, bounds.height() - 1.0))
        )
        return top, bottom, left, right

    def _clamp_offset(
        self, points: Sequence[tuple[float, float]], rows: float, cols: float
    ) -> tuple[float, float]:
        """Limit an offset so that it moves none of these points out of the
        photograph; points already outside it are not pulled in."""

        if not points:
            return 0.0, 0.0
        top, bottom, left, right = self._raw_limits()
        row_values = [point[0] for point in points]
        col_values = [point[1] for point in points]
        rows = min(
            max(rows, min(0.0, top - min(row_values))),
            max(0.0, bottom - max(row_values)),
        )
        cols = min(
            max(cols, min(0.0, left - min(col_values))),
            max(0.0, right - max(col_values)),
        )
        return rows, cols

    def _moved_points(
        self,
        polygons: Mapping[int, Sequence[tuple[float, float]]],
        rows: float,
        cols: float,
    ) -> dict[int, list[tuple[float, float]]]:
        """The points of these polygons with the selected ones moved by an
        offset (raw pixels), limited to the photograph."""

        selected = self._selected_by_polygon()
        moving = [
            polygons[index][vertex]
            for index, vertices in selected.items()
            if index in polygons
            for vertex in vertices
            if vertex < len(polygons[index])
        ]
        rows, cols = self._clamp_offset(moving, rows, cols)
        moved = {}
        for index, points in polygons.items():
            new_points = list(points)
            for vertex in selected.get(index, ()):
                if vertex < len(new_points):
                    row, col = new_points[vertex]
                    new_points[vertex] = (row + rows, col + cols)
            moved[index] = new_points
        return moved

    def nudge_selected_points(self, screen_rows: float, screen_cols: float) -> bool:
        """Move the selected points by screen pixels as one change (limited to
        the photograph); False when nothing moved."""

        selected = self._selected_by_polygon()
        if not selected:
            return False
        transform = self.transform()
        scale_rows, scale_cols = self._scale()
        zoom_rows = abs(transform.m22()) * scale_rows
        zoom_cols = abs(transform.m11()) * scale_cols
        if zoom_rows <= 0.0 or zoom_cols <= 0.0:
            return False
        before = {index: list(self._polygons[index].points) for index in selected}
        after = self._moved_points(
            before, screen_rows / zoom_rows, screen_cols / zoom_cols
        )
        changes = {
            index: points for index, points in after.items() if points != before[index]
        }
        self._emit_edits(changes)
        return bool(changes)

    def delete_selected_points(self) -> int:
        """Remove the selected points as one change; returns how many went.

        A polygon that would keep fewer than three points is left unchanged,
        its points stay selected and ``notice`` says so; other polygons still
        lose their selected points.
        """

        selection = set(self._selection)
        changes: dict[int, list[tuple[float, float]]] = {}
        refused = False
        for index, vertices in sorted(self._selected_by_polygon().items()):
            kept = [
                point
                for vertex, point in enumerate(self._polygons[index].points)
                if vertex not in vertices
            ]
            if len(kept) < 3:
                refused = True
            else:
                changes[index] = kept
        removed = 0
        if changes:
            self._emit_edits(changes)
            applied = {
                index
                for index, points in changes.items()
                if index < len(self._polygons)
                and list(self._polygons[index].points) == points
            }
            removed = sum(1 for index, _vertex in selection if index in applied)
            self.set_selection(point for point in selection if point[0] not in applied)
        if refused:
            self.notice.emit(_TOO_FEW_POINTS)
        return removed

    def _insert_point_at(self, position: QPointF) -> bool:
        """Add a point on the polygon edge under a viewport position (not on a
        point) and select it."""

        if self._hit_vertex(position) is not None:
            return False
        index = self._hit_edge(position)
        if index is None:
            return False
        row, col = self.scene_to_raw(self.mapToScene(position.toPoint()))
        points, inserted = insert_point_on_nearest_edge(
            self._polygons[index].points, row, col
        )
        if inserted < 0:
            return False
        self._emit_edits({index: points})
        if index < len(self._polygons) and list(self._polygons[index].points) == points:
            self.set_selection([(index, inserted)])
            return True
        return False

    def _edit_press(self, event: Any) -> None:
        """A left press with the Edit tool: select points, then drag them or
        a rubber band."""

        # A drag or band whose release never came (a dialog took the mouse).
        self._cancel_drag()
        self._cancel_band()
        position = event.position()
        extend = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        hit = self._hit_vertex(position)
        if hit is None:
            if not extend:
                self.set_selection(())
            self._band_origin = QPointF(position)
            return
        selection = set(self._selection)
        if extend:
            selection ^= {hit}
        elif hit not in selection:
            selection = {hit}
        self.set_selection(selection)
        if hit in self._selection:
            self._start_drag(position)

    def _band_rect(self, position: QPointF) -> QRectF:
        """Scene rectangle from where the rubber band started to a position."""

        assert self._band_origin is not None
        return QRectF(
            QPointF(self.mapToScene(self._band_origin.toPoint())),
            QPointF(self.mapToScene(position.toPoint())),
        ).normalized()

    def _move_band(self, position: QPointF) -> None:
        if self._band_origin is None:
            return
        if (
            not self.band_item.isVisible()
            and (position - self._band_origin).manhattanLength() <= _CLICK_SLOP_PX
        ):
            return
        self.band_item.setRect(self._band_rect(position))
        self.band_item.show()

    def _finish_band(self, position: QPointF) -> None:
        """Add the shown points inside the rubber band to the selection."""

        if self._band_origin is None:
            return
        dragged = (
            self.band_item.isVisible()
            or (position - self._band_origin).manhattanLength() > _CLICK_SLOP_PX
        )
        rect = self._band_rect(position)
        self._cancel_band()
        if not dragged:
            return
        chosen = set(self._selection)
        for index, polygon in enumerate(self._polygons):
            if polygon.layer_id in self._hidden_layers:
                continue
            for vertex, (row, col) in enumerate(polygon.points):
                if rect.contains(self.raw_to_scene(row, col)):
                    chosen.add((index, vertex))
        self.set_selection(chosen)

    def _cancel_band(self) -> None:
        self._band_origin = None
        self.band_item.hide()
        self.band_item.setRect(QRectF())

    def _edit_key(self, event: Any) -> bool:
        """Keys of the Edit tool; False when the key is not one of them."""

        key = event.key()
        busy = self._drag is not None or self._band_origin is not None
        if key == Qt.Key.Key_Escape:
            if not (self._selection or busy):
                return False
            self.clear_selection()
            return True
        if not self._selection or busy:
            return False
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_points()
            return True
        step = _NUDGE_KEYS.get(key)
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if step is None or modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            return False
        pixels = (
            _NUDGE_SHIFT_PX
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else _NUDGE_PX
        )
        self.nudge_selected_points(step[0] * pixels, step[1] * pixels)
        return True

    # ------------------------------------------------------ drawing polygons

    def drafting(self) -> bool:
        return bool(self._draft)

    @property
    def draft_points(self) -> list[tuple[float, float]]:
        return list(self._draft)

    def _drawing(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool == "polygon"
            and self._raw_size is not None
            and self._marks_visible
        )

    def _draft_pen(self, *, dashed: bool = False) -> QPen:
        pen = QPen(QColor(self.layer_color), 2.0)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _render_draft(self, cursor: QPointF | None = None) -> None:
        for item in self._draft_handles:
            self.scene().removeItem(item)
        self._draft_handles = []
        if not self._draft:
            self.draft_item.setPath(QPainterPath())
            self.rubber_item.setPath(QPainterPath())
            return
        scene_points = [self.raw_to_scene(row, col) for row, col in self._draft]
        path = QPainterPath(scene_points[0])
        for point in scene_points[1:]:
            path.lineTo(point)
        self.draft_item.setPen(self._draft_pen())
        self.draft_item.setPath(path)
        rubber = QPainterPath()
        if cursor is not None:
            rubber.moveTo(scene_points[-1])
            rubber.lineTo(cursor)
            if len(scene_points) >= 2:
                rubber.lineTo(scene_points[0])
        self.rubber_item.setPen(self._draft_pen(dashed=True))
        self.rubber_item.setPath(rubber)
        for index, point in enumerate(scene_points):
            # The first point is larger: clicking it closes the polygon.
            self._draft_handles.append(
                self._handle(point, self.layer_color, size=10.0 if index == 0 else 7.0)
            )

    def _add_draft_point(self, scene_point: QPointF) -> None:
        raw = self.scene_to_raw(self._clamped_scene(scene_point))
        if self._draft:
            last = self.mapFromScene(self.raw_to_scene(*self._draft[-1]))
            spot = self.mapFromScene(scene_point)
            if math.hypot(last.x() - spot.x(), last.y() - spot.y()) < 2.0:
                return
        self._draft.append(raw)
        self._render_draft(scene_point)
        self.draft_changed.emit(len(self._draft))

    def remove_last_point(self) -> None:
        if not self._draft:
            return
        self._draft.pop()
        self._render_draft()
        self.draft_changed.emit(len(self._draft))

    def cancel_polygon(self) -> None:
        had = bool(self._draft)
        self._draft = []
        self._render_draft()
        if had:
            self.draft_changed.emit(0)

    def close_polygon(self) -> bool:
        """Finish the polygon being drawn; False when it has too few points."""

        if not self._draft:
            return False
        if len(self._draft) < 3:
            self.notice.emit(
                "A polygon needs at least three points; keep clicking, or press "
                "Esc to cancel."
            )
            return False
        polygon = lp.Polygon(int(self.layer_id), list(self._draft))
        if polygon.area() < 1.0:
            self.notice.emit("The polygon encloses no area.")
            return False
        self._draft = []
        self._render_draft()
        self.draft_changed.emit(0)
        self.polygon_finished.emit(polygon)
        return True

    def _near_first_point(self, position: QPointF) -> bool:
        if len(self._draft) < 3:
            return False
        first = self.mapFromScene(self.raw_to_scene(*self._draft[0]))
        return math.hypot(first.x() - position.x(), first.y() - position.y()) <= (
            _VERTEX_HIT_PX
        )

    def _editing(self) -> bool:
        return (
            self.paint_enabled
            and not self.place_mode
            and self.tool == "edit"
            and self._raw_size is not None
            and self._marks_visible
        )

    def _start_drag(self, position: QPointF) -> None:
        """Start dragging the selected points from a viewport position."""

        before = {
            index: list(self._polygons[index].points)
            for index in sorted(self._selected_by_polygon())
        }
        if not before:
            return
        self._drag = _PointDrag(
            QPointF(position),
            self.scene_to_raw(self.mapToScene(position.toPoint())),
            before,
            {index: list(points) for index, points in before.items()},
        )

    def _move_drag(self, position: QPointF) -> None:
        """Move the dragged points with the mouse, all by the same offset."""

        drag = self._drag
        if drag is None:
            return
        if not drag.moving:
            if (position - drag.press).manhattanLength() <= _CLICK_SLOP_PX:
                return
            drag.moving = True
        row, col = self.scene_to_raw(self.mapToScene(position.toPoint()))
        drag.after = self._moved_points(
            drag.before, row - drag.start[0], col - drag.start[1]
        )
        for index, points in drag.after.items():
            if index < len(self._polygon_items):
                self._polygon_items[index].setPath(self._polygon_path(points))
            handles = (
                self._handle_items[index] if index < len(self._handle_items) else []
            )
            for vertex, item in enumerate(handles):
                if vertex < len(points):
                    item.setPos(self.raw_to_scene(*points[vertex]))

    def _end_drag(self) -> None:
        """Finish a drag: one change for every polygon whose points moved."""

        drag, self._drag = self._drag, None
        if drag is None:
            return
        changes = {
            index: points
            for index, points in drag.after.items()
            if points != drag.before[index]
        }
        # Without a change the outlines and handles are where they were.
        self._emit_edits(changes)

    def _cancel_drag(self) -> None:
        """Stop a drag and put the points back where they were."""

        drag, self._drag = self._drag, None
        if drag is None:
            return
        for index, points in drag.before.items():
            if index < len(self._polygon_items):
                self._polygon_items[index].setPath(self._polygon_path(points))
        self._update_handles()

    # ------------------------------------------------------------ tools

    def set_tool(self, tool: str) -> None:
        if tool not in self.TOOLS:
            raise ValueError(f"Unknown tool {tool!r}")
        if tool != self.tool:
            self.cancel_polygon()
            # Only the Edit tool has selected points.
            self.clear_selection()
        self.tool = tool
        self._cancel_stroke()
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.viewport().setCursor(
            {
                "pan": Qt.CursorShape.OpenHandCursor,
                "edit": Qt.CursorShape.ArrowCursor,
            }.get(tool, Qt.CursorShape.CrossCursor)
        )
        self._update_handles()
        self._update_cursor()

    def set_layer(self, layer_id: int, color: QColor) -> None:
        self.layer_id = int(layer_id)
        self.layer_color = QColor(color)
        if self._draft:
            self._render_draft()

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
        self.refresh_detail()
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
            if button == Qt.MouseButton.RightButton:
                self._right_press = event.position()
                self._right_dragged = False
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
        if button == Qt.MouseButton.LeftButton and self._editing():
            self._edit_press(event)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._drawing():
            # Draw only adds points: a click on a point of an existing polygon
            # adds a point to the new polygon there.
            point = self.mapToScene(event.position().toPoint())
            if self._near_first_point(event.position()):
                self.close_polygon()
                event.accept()
                return
            if self.pixmap_item.boundingRect().contains(point) or self._draft:
                self._add_draft_point(point)
                event.accept()
                return
        if button == Qt.MouseButton.LeftButton and self._painting():
            point = self.mapToScene(event.position().toPoint())
            if self.pixmap_item.boundingRect().contains(point):
                self._start_stroke(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drawing():
            if self._draft:
                # The first click of the double click added the last point.
                self.close_polygon()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._editing():
            self._cancel_band()
            self._insert_point_at(event.position())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self._pan_origin is not None:
            if self._right_press is not None and not self._right_dragged:
                moved = event.position() - self._right_press
                self._right_dragged = moved.manhattanLength() > _CLICK_SLOP_PX
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
        if self._drag is not None:
            self._move_drag(event.position())
            event.accept()
            return
        if self._band_origin is not None:
            self._move_band(event.position())
            event.accept()
            return
        if self._draft and self._drawing():
            self._render_draft(self._clamped_scene(point))
        if self._points and self._painting():
            self._extend(self._clamped_scene(point))
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
            clicked = (
                button == Qt.MouseButton.RightButton
                and self._right_press is not None
                and not self._right_dragged
            )
            self._right_press = None
            self.set_tool(self.tool)
            if clicked:
                self._right_click(event)
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._drag is not None:
            self._end_drag()
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._band_origin is not None:
            self._finish_band(event.position())
            event.accept()
            return
        if button == Qt.MouseButton.LeftButton and self._points:
            self._finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _right_click(self, event: Any) -> None:
        """A right click without dragging: close the polygon or open a menu."""

        if self.place_mode:
            return
        if self._draft and self._drawing():
            self.close_polygon()
            return
        if not self._marks_visible or not self.paint_enabled:
            return
        hit = self.polygon_at(event.position())
        if hit is not None:
            self.last_context_raw = self.scene_to_raw(
                self.mapToScene(event.position().toPoint())
            )
            self.polygon_context.emit(
                hit[0], hit[1], self.viewport().mapToGlobal(event.position().toPoint())
            )

    def keyPressEvent(self, event: Any) -> None:
        if self._draft:
            key = event.key()
            if key == Qt.Key.Key_Escape:
                self.cancel_polygon()
                event.accept()
                return
            if key == Qt.Key.Key_Backspace:
                self.remove_last_point()
                event.accept()
                return
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.close_polygon()
                event.accept()
                return
        if self._editing() and self._edit_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.cursor_item.hide()
        if self._draft:
            self._render_draft()
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
        self._workspace_frame_config: dict[str, Any] = {}
        # True while a session painted on the raw reference photo is shown in
        # a project whose geometry is in the saved corrected rig frame.
        self._raw_session_frame = False
        # The reference photograph on the fine proposal grid: (source, fine image).
        self._fine_image: tuple[lp.PaintingImage, lp.PaintingImage] | None = None
        self._saved_corrected_boundaries: np.ndarray | None = None
        self.session: lp.PaintingSession | None = None
        self.image: lp.PaintingImage | None = None
        self.proposal: lp.LayerProposal | None = None
        self.last_export: Any = None
        # Snapshots of (polygons, strokes) for undo and redo.
        self._undo_stack: list[tuple[list[lp.Polygon], list[lp.Stroke]]] = []
        self._redo_stack: list[tuple[list[lp.Polygon], list[lp.Stroke]]] = []
        # Layers hidden in the photograph: a display choice for the session
        # shown, never saved and never part of the proposal.
        self._hidden_layers: set[int] = set()
        self._hidden_session: lp.PaintingSession | None = None
        self._layer_table_menu_widget: QMenu | None = None
        self._dirty = False
        self._export_stale = True
        self._export_key = ""
        self._loading = False
        self._facies_menu_open = False
        self._facies_refresh_pending = False
        self._facies_buttons: dict[str, QToolButton] = {}
        self._polygon_menu_widget: QMenu | None = None
        self._other_photo: Path | None = None
        self._reference_token: dict[str, Any] | None = None
        self._plain: np.ndarray | None = None
        self._enhanced: np.ndarray | None = None
        # Another photograph of the reference run, shown under the marks
        # instead of the reference photo when chosen (display only).
        self._background_path: Path | None = None
        self._background_rgb: np.ndarray | None = None
        self._background_plain: np.ndarray | None = None
        self._background_enhanced: np.ndarray | None = None
        self._background_token: dict[str, Any] | None = None
        self._background_shown = False
        # The reference run's configured camera-U profile (N106), cached per run.
        self._camera_u_key: tuple[str, str] | None = None
        self._camera_u: Any = None
        # The reference and background photos at full resolution (uint8),
        # plain and with enhanced contrast, shown when zoomed in.
        self._detail_plain: np.ndarray | None = None
        self._detail_enhanced: np.ndarray | None = None
        self._background_detail_plain: np.ndarray | None = None
        self._background_detail_enhanced: np.ndarray | None = None
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
        self.canvas.polygon_finished.connect(self._polygon_finished)
        self.canvas.polygon_edited.connect(self._polygon_edited)
        self.canvas.polygons_edited.connect(self._polygons_edited)
        self.canvas.polygon_context.connect(self._polygon_menu)
        self.canvas.draft_changed.connect(self._draft_changed)
        self.canvas.notice.connect(self._canvas_notice)

        from darsia.fluidflower.series_rig_frame_panel import RigFramePanel

        self.rig_panel = RigFramePanel(self, project, reference, scrollable=False)
        self.rig_panel.changed.connect(self._frame_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_steps())
        splitter.addWidget(self._build_canvas_column())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 1020])
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
        scroll.setMinimumWidth(500)
        content = QWidget()
        scroll.setWidget(content)
        self.steps_scroll = scroll
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(
            _note(
                "Outline the layers on the raw baseline photograph of the reference "
                "run, propose the boundaries, adjust and propose again until they "
                "fit; then group the layers into facies and export them."
            )
        )
        self.step_frame = StepSection(
            1, "Rig frame and injection points", self.rig_panel, expanded=False
        )
        self.step_paint = StepSection(2, "Mark layers", self._build_paint_step())
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
                "Choose a layer and draw it with Draw (L): left click adds a "
                "point, right click or Enter closes the polygon. Rough outlines "
                "are enough; the boundaries are placed within the snap distance "
                "(step 3). Draw only adds new polygons and never moves a point "
                "that is already there. To change existing polygons, use Edit "
                "(V): click a point to select it and drag to move it, Ctrl+click "
                "to select more points, or drag on empty space to select the "
                "points in a rectangle. Delete removes the selected points, the "
                "arrow keys move them a little (Shift: more) and Esc clears the "
                "selection. Double-click an edge to add a point there. Right-"
                "click a polygon to add or remove a point, delete the polygon or "
                "give it to another layer. The brush (B) paints strokes, for "
                "example as corrections. Mark the ColorChecker, tape and fittings "
                "as Excluded. Untick Show to hide a layer in the photograph; this "
                "changes only what is shown, and its points cannot be edited "
                "while it is hidden. Right-click the list to show only one layer, "
                "or all layers again. Background (T) shows another photo of the "
                "same run under the marks, for example one taken before the "
                "indicator was added; the proposal still uses the reference photo."
            )
        )
        self.layer_table = QTableWidget(0, 5)
        self.layer_table.setHorizontalHeaderLabels(
            ["", "Layer", "Polygons", "Strokes", "Show"]
        )
        show_header = self.layer_table.horizontalHeaderItem(_SHOW_COLUMN)
        if show_header is not None:
            show_header.setToolTip(_SHOW_TIP)
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
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            _SHOW_COLUMN, QHeaderView.ResizeMode.ResizeToContents
        )
        self.layer_table.itemSelectionChanged.connect(self._layer_selected)
        self.layer_table.itemChanged.connect(self._layer_item_changed)
        self.layer_table.cellDoubleClicked.connect(self._layer_double_clicked)
        self.layer_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_table.customContextMenuRequested.connect(self._layer_table_menu)
        layout.addWidget(self.layer_table)
        buttons = QHBoxLayout()
        self.add_layer_button = QPushButton("Add layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.remove_layer_button = QPushButton("Remove layer")
        self.remove_layer_button.clicked.connect(lambda: self.remove_layer())
        self.clear_strokes_button = QPushButton("Clear its marks")
        self.clear_strokes_button.setToolTip(
            "Remove the polygons and strokes of the selected layer"
        )
        self.clear_strokes_button.clicked.connect(lambda: self.clear_layer_marks())
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
                "A model learns from the marked pixels and labels the rest of the "
                "photograph inside the rig frame. Inside a polygon its layer is "
                "kept; near polygon edges the model places the boundary within the "
                "snap distance, choosing only between layers marked nearby. Where "
                "it is wrong, move or add a polygon point with Edit (V) or paint "
                "a short stroke of the right layer, and propose again."
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
        snap_row = QHBoxLayout()
        snap_row.addWidget(QLabel("Snap distance"))
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.0, lp.MAX_SNAP_MM)
        self.snap.setDecimals(0)
        self.snap.setSingleStep(1.0)
        self.snap.setSuffix(" mm")
        self.snap.setToolTip(
            "How far polygon edges may lie from the true boundary. Within this "
            "distance the model places the boundary; 0 uses the polygons exactly. "
            "Without the rig size, the photograph is taken as 1 m wide."
        )
        self.snap.editingFinished.connect(self._settings_changed)
        snap_row.addWidget(self.snap)
        snap_row.addSpacing(10)
        snap_row.addWidget(QLabel("Detail"))
        self.detail = QComboBox()
        for key, label in (("normal", "Normal"), ("fine", "Fine")):
            self.detail.addItem(label, key)
        self.detail.setToolTip(
            "The grid the model labels. Fine uses the photograph as shown on screen, "
            "with about twice the pixels per mm of Normal, so thin layers keep their "
            "width; a proposal then takes about twice as long. A layer can also get "
            "its own snap distance: right-click it in the layer list."
        )
        self.detail.currentIndexChanged.connect(self._settings_changed)
        snap_row.addWidget(self.detail)
        snap_row.addStretch(1)
        layout.addLayout(snap_row)
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
                "A facies groups layers that share role, porosity and permeability. "
                "Tick its layers in the Layers list; each layer belongs to one "
                "facies. Free water and excluded regions get no properties; "
                "everything outside the rig frame is added as a facies of its own."
            )
        )
        self.facies_table = QTableWidget(0, 5)
        self.facies_table.verticalHeader().setVisible(False)
        self.facies_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.facies_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.facies_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        header = self.facies_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.facies_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.facies_table.itemChanged.connect(self._facies_item_changed)
        self.facies_table.itemSelectionChanged.connect(self._facies_selection_changed)
        layout.addWidget(self.facies_table)
        facies_buttons = QHBoxLayout()
        self.add_facies_button = QPushButton("Add facies")
        self.add_facies_button.setToolTip(
            "Add an empty facies, then tick its layers in the Layers list"
        )
        self.add_facies_button.clicked.connect(lambda: self.add_facies())
        facies_buttons.addWidget(self.add_facies_button)
        self.remove_facies_button = QPushButton("Remove facies")
        self.remove_facies_button.setToolTip(
            "Remove the selected facies; each of its layers gets a facies of its own"
        )
        self.remove_facies_button.clicked.connect(lambda: self.remove_facies())
        facies_buttons.addWidget(self.remove_facies_button)
        facies_buttons.addStretch(1)
        layout.addLayout(facies_buttons)
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
            (
                "polygon",
                "Draw",
                (
                    "Draw a new polygon for the selected layer (L): left click adds "
                    "a point, right click or Enter closes it"
                ),
            ),
            (
                "edit",
                "Edit",
                (
                    "Edit existing polygons (V): click a point to select it and "
                    "drag to move; Ctrl+click selects more points; drag on empty "
                    "space selects points in a rectangle; Delete removes the "
                    "selected points; double-click or right-click an edge to add a "
                    "point"
                ),
            ),
            ("paint", "Paint", "Paint strokes of the selected layer (B)"),
            ("erase", "Erase", "Erase strokes (E); polygons are kept"),
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
        self.undo_button.setToolTip(
            "Undo the last change to the polygons or strokes (Ctrl+Z)"
        )
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
        self.show_strokes = QCheckBox("Marks")
        self.show_strokes.setToolTip("Polygons and strokes")
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
        display.addSpacing(8)
        background_label = QLabel("Background")
        background_label.setToolTip(_BACKGROUND_TIP)
        display.addWidget(background_label)
        self.background = QComboBox()
        self.background.setToolTip(_BACKGROUND_TIP)
        self.background.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.background.activated.connect(self._background_activated)
        display.addWidget(self.background)
        self._refresh_background_choice()
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
        self.canvas_hint = _note(_CANVAS_HINT)
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
            ("L", lambda: self.set_tool("polygon")),
            ("V", lambda: self.set_tool("edit")),
            ("B", lambda: self.set_tool("paint")),
            ("E", lambda: self.set_tool("erase")),
            ("H", lambda: self.set_tool("pan")),
            ("F", self.canvas.fit_image),
            ("P", self.propose),
            ("T", self.toggle_background),
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

    def _session_painted_on_raw_photo(self) -> bool:
        """Whether the run's saved session has marks on its raw reference photo.

        The marks fit the raw photo when the session's recorded image size is
        the raw photo's size. Nothing is written.
        """

        try:
            reference = self._reference()
            if reference is None or reference.baseline_image is None:
                return False
            if not lp.session_path(self.project, reference.run_id).is_file():
                return False
            saved = lp.load_painting_session(self.project, reference)
            raw_size = photo_size(Path(reference.baseline_image),
                                  camera_u=lp.reference_camera_u(self.project, reference))
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return (
            bool(saved.polygons or saved.strokes)
            and saved.image_size is not None
            and tuple(saved.image_size) == tuple(raw_size)
        )

    def reference(self) -> SeriesRunInputs | None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return workspace_frame.workspace_reference(self._workspace_frame_config, self._reference(), check_files=False)
        return self._reference()

    def current_frame(self) -> RigFrame | None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            reference = self.reference()
            return workspace_frame.corrected_rig_frame(self._workspace_frame_config, reference.run_id) if reference else None
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
        # Which layers are hidden is a display choice for the session shown.
        self._clear_background()
        self._hidden_layers = set()
        self._hidden_session = None
        self.canvas.set_hidden_layers(())
        self._raw_session_frame = False
        try:
            self._workspace_frame_config = workspace_frame.current_configuration(self.project)
        except (OSError, ValueError) as error:
            self.session = None
            self.image = None
            self.canvas.clear_photo()
            self.propose_status.setText(f"Could not read the workspace coordinate frame: {error}")
            self.propose_button.setEnabled(False)
            self.export_button.setEnabled(False)
            return
        if (
            workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
            and self._session_painted_on_raw_photo()
        ):
            # Marks painted on the raw reference photo only fit that photo:
            # this tab then works in the raw frame. The saved corrected maps
            # of the project are left as they are.
            self._raw_session_frame = True
            self._workspace_frame_config = {}
        self.rig_panel.setEnabled(workspace_frame.frame_kind(self._workspace_frame_config) == "raw")
        self._saved_corrected_boundaries = None
        self.show_proposal.setText("Saved layer boundaries" if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected" else "Proposal")
        self._loading = True
        try:
            self.session = None
            self.image = None
            self.proposal = None
            self._undo_stack = []
            self._redo_stack = []
            self.canvas.cancel_polygon()
            self.canvas.clear_selection()
            self._other_photo = None
            self._reference_token = None
            self._detail_plain = None
            self._detail_enhanced = None
            try:
                reference = self.reference()
            except (OSError, ValueError) as error:
                self.canvas.clear_photo()
                self.propose_status.setText(f"Could not read the selected run's corrected maps: {error}")
                self.propose_button.setEnabled(False)
                self.export_button.setEnabled(False)
                return
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
                self._restore_background()
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
            elif self._raw_session_frame and self.image is not None:
                self.propose_status.setText(_RAW_SESSION_NOTE)
            self.last_export = None
            self._refresh_layers()
            self._refresh_facies_table()
            self._render_marks()
            self._render_proposal()
            self._refresh_export_info()
            self._refresh_status()
            frame_done = (
                self.current_frame() is not None
                and self.current_frame().has_size()  # type: ignore[union-attr]
            )
            self.step_frame.set_expanded(not frame_done)
            self.set_tool("polygon" if self.session is not None else "pan")
        finally:
            self._loading = False

    def _load_reference_photo(self, reference: SeriesRunInputs) -> None:
        assert reference.baseline_image is not None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
                image = workspace_frame.load_painting_image(self._workspace_frame_config, reference)
                if (self.session is not None and self.session.image_size not in (None, image.image_size)
                        and (self.session.strokes or self.session.polygons)):
                    raise ValueError("Saved marks belong to another coordinate frame; the draft was preserved")
                self._saved_corrected_boundaries = workspace_frame.saved_boundaries(
                    self._workspace_frame_config, reference.run_id, image.display.shape[:2], project=self.project)
            else:
                image = lp.load_painting_image(
                    Path(reference.baseline_image),
                    camera_u=lp.reference_camera_u(self.project, reference))
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
        # The full-resolution photo for zooming in; the saved corrected maps
        # are shown at their display size only.
        self._detail_plain = (
            None
            if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
            else self._full_resolution(image.path, camera_u=image.camera_u)
        )
        self._update_enhanced()
        self._show_reference(reset_view=True)

    def _display_mask(self) -> np.ndarray | None:
        """Pixels used to stretch the display contrast (inside the rig frame)."""

        if self.image is None:
            return None
        frame = self.current_frame()
        shape = self.image.display.shape[:2]
        scale = (
            shape[0] / self.image.image_size[1],
            shape[1] / self.image.image_size[0],
        )
        if frame is not None:
            return lp.frame_mask(frame, shape, scale)
        # Without a frame, leave out the border where the rig frame usually is.
        mask = np.zeros(shape, dtype=bool)
        rows, cols = int(0.12 * shape[0]), int(0.08 * shape[1])
        mask[rows : shape[0] - rows, cols : shape[1] - cols] = True
        return mask

    def _full_resolution(self, path: Path | str, *, camera_u: Any = None) -> np.ndarray | None:
        """The photograph at up to ``DETAIL_LIMIT_PX`` as uint8 RGB.

        It is shown when zoomed in. None when it would be no larger than the
        display copy, or when it cannot be read. Display only: the marks, the
        proposal and saving never use it.
        """

        if self.image is None:
            return None
        from darsia.fluidflower.photograph_io import load_reduced_rgb8

        try:
            width, height = photo_size(Path(path), camera_u=camera_u)
            factor = max(1, int(math.ceil(max(width, height) / lp.DETAIL_LIMIT_PX)))
            detail = load_reduced_rgb8(Path(path), factor, camera_u=camera_u)
        except (OSError, ValueError):
            return None
        rows, cols = self.image.display.shape[:2]
        if detail.shape[0] <= rows or detail.shape[1] <= cols:
            return None
        return detail

    def _stretched_detail(
        self, detail: np.ndarray | None, display: np.ndarray
    ) -> np.ndarray | None:
        """``detail`` with the same contrast stretch as its display copy."""

        if detail is None:
            return None
        lower, upper = lp.contrast_limits(display, self._display_mask())
        return lp.stretch_contrast_rgb8(detail, lower, upper)

    def _update_enhanced(self) -> None:
        if self.image is None:
            self._enhanced = None
            self._detail_enhanced = None
            return
        self._enhanced = lp.enhance_contrast(self.image.display, self._display_mask())
        self._detail_enhanced = self._stretched_detail(self._detail_plain, self.image.display)
        self._update_background_enhanced()

    def _show_reference(self, *, reset_view: bool) -> None:
        if self.image is None or self._reference_token is None:
            return
        enhanced = self.enhance.isChecked()
        token = self._reference_token
        display = self._enhanced if enhanced else self._plain
        detail = self._detail_enhanced if enhanced else self._detail_plain
        if self._background_shown and self._background_token is not None:
            token = self._background_token
            stretched = enhanced and self._background_enhanced is not None
            display = self._background_enhanced if stretched else self._background_plain
            detail = (
                self._background_detail_enhanced
                if stretched
                else self._background_detail_plain
            )
        assert display is not None
        self.canvas.set_photo(
            display,
            self.image.image_size,
            token=token,
            reset_view=reset_view,
            detail=detail,
        )
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.canvas.paint_enabled = False
        else:
            self.canvas.paint_enabled = True
        self._other_photo = None
        self.rig_panel.use_host_photo(self.image.image_size, self.canvas.preview)
        self.photo_note_row.hide()
        self._render_marks()
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
        from darsia.fluidflower.photograph_io import load_reduced

        # A RAW of a run with a camera profile is shown on U, the grid of the marks.
        camera_u = self._camera_u_profile() if image_family(path) == "raw" else None
        try:
            width, height = photo_size(Path(path), camera_u=camera_u)
            factor = max(1, int(math.ceil(max(width, height) / lp.DISPLAY_LIMIT_PX)))
            rgb = load_reduced(Path(path), factor, camera_u=camera_u)
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

    # ----------------------------------------------------- background photo

    def background_available(self) -> bool:
        """A background photo needs the raw reference photograph of the run."""

        if self.image is None:
            return False
        try:
            return workspace_frame.frame_kind(self._workspace_frame_config) == "raw"
        except (AttributeError, TypeError, ValueError):
            return False

    @property
    def background_path(self) -> Path | None:
        """The chosen background photo of this run, if any."""

        return self._background_path

    @property
    def background_shown(self) -> bool:
        return bool(self._background_shown and self._background_token is not None)

    def set_background_photo(
        self, path: Path | str | None, *, show: bool = True, quiet: bool = False
    ) -> bool:
        """Show another photograph of the reference run under the marks.

        The photograph must have the raw size of the reference photo, for
        example one taken earlier in the same run before the indicator was
        added. Only the display changes: marks, strokes and the proposal stay
        on top, painting works as on the reference photo, and the proposal is
        always computed from the reference photo. ``None`` forgets the choice.
        The choice is remembered per user, series folder and run.
        """

        if path is None:
            self._clear_background()
            self._remember_background(None)
            if self.image is not None and self._other_photo is None:
                self._show_reference(reset_view=False)
            return True
        if not self.background_available():
            if not quiet:
                QMessageBox.information(
                    self,
                    "Background photo",
                    "A background photo can be used with the raw reference "
                    "photograph only.",
                )
            return False
        assert self.image is not None
        path = Path(path)
        expected = (int(self.image.image_size[0]), int(self.image.image_size[1]))
        # A camera RAW of a run with a configured camera profile is shown on the
        # camera grid U (N106). U is also the grid of the camera's JPEG and of a
        # RAW reference shown on U; otherwise both must be the same kind.
        camera_u = self._camera_u_profile() if image_family(path) == "raw" else None
        reference_on_u = getattr(self.image, "camera_u", None) is not None
        if image_family(path) != image_family(self.image.path):
            if camera_u is None and not reference_on_u:
                if not quiet:
                    QMessageBox.warning(
                        self,
                        "Background photo",
                        f"{path.name} and the reference photo {Path(self.image.path).name} "
                        "are not the same kind of file (camera RAW or JPEG/TIFF/PNG). "
                        "A camera's JPEG is cropped and lens-corrected, so the two do "
                        "not line up without a camera profile for this run. Choose a "
                        "photograph of the same kind.",
                    )
                return False
        try:
            width, height = photo_size(path, camera_u=camera_u)
        except (OSError, ValueError) as error:
            if not quiet:
                QMessageBox.warning(self, "Could not read the photograph", str(error))
            return False
        if (int(width), int(height)) != expected:
            if not quiet:
                QMessageBox.warning(
                    self,
                    "Background photo",
                    f"{path.name} is {width} x {height} pixels, the reference "
                    f"photo {expected[0]} x {expected[1]}. Choose a photograph "
                    "of the same run and camera position.",
                )
            return False
        from darsia.fluidflower.photograph_io import load_reduced

        shape = self.image.display.shape[:2]
        try:
            factor = max(1, int(math.ceil(max(width, height) / lp.DISPLAY_LIMIT_PX)))
            rgb = np.asarray(load_reduced(path, factor, camera_u=camera_u), dtype=np.float32)
        except (OSError, ValueError) as error:
            if not quiet:
                QMessageBox.warning(self, "Could not read the photograph", str(error))
            return False
        if rgb.shape[:2] != shape:
            import cv2

            rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        rgb = np.clip(rgb, 0.0, 1.0)
        return self._install_background(
            path, rgb, self._full_resolution(path, camera_u=camera_u), expected, show)

    def _camera_u_profile(self):
        """The reference run's configured camera-U profile, or None (cached per run)."""

        reference = self.reference()
        root = getattr(self.project, "root", None)
        if reference is None or root is None:
            return None
        key = (str(root), str(reference.run_id))
        if self._camera_u_key != key:
            try:
                self._camera_u = lp.run_camera_u(self.project, reference.run_id)
            except (OSError, ValueError, KeyError, TypeError):
                self._camera_u = None
            self._camera_u_key = key
        return self._camera_u

    def _install_background(self, path, rgb, detail, expected, show) -> bool:
        """Install a loaded background photo (display copy and detail) and show it."""

        self._background_path = path
        self._background_rgb = rgb
        self._background_plain = np.rint(255.0 * rgb).astype(np.uint8)
        self._background_detail_plain = detail
        self._update_background_enhanced()
        self._background_token = {
            "baseline": self._background_plain,
            "path": str(path),
            "raw_size": expected,
            "reference": False,
            "background": True,
        }
        self._background_shown = bool(show)
        self._remember_background(path)
        self._refresh_background_choice()
        if self._other_photo is None:
            self._show_reference(reset_view=False)
        return True

    def show_background(self, shown: bool) -> None:
        """Show the chosen background photo (True) or the reference photo."""

        self._background_shown = bool(shown) and self._background_token is not None
        self._refresh_background_choice()
        if self.image is not None:
            self._show_reference(reset_view=False)

    def toggle_background(self) -> None:
        """Switch between the reference photo and the chosen background (T)."""

        if self._background_token is None:
            self.choose_background_photo()
            return
        self.show_background(not self._background_shown)

    def choose_background_photo(self) -> bool:
        if not self.background_available():
            self._refresh_background_choice()
            return False
        assert self.image is not None
        start = Path(self.image.path).parent
        start = start.parent if start.parent != start else start
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Background photo of the same run",
            str(start),
            photo_dialog_filter(self.image.path),
        )
        if not path:
            self._refresh_background_choice()
            return False
        return self.set_background_photo(Path(path))

    def _update_background_enhanced(self) -> None:
        if self._background_rgb is None or self.image is None:
            self._background_enhanced = None
            self._background_detail_enhanced = None
            return
        self._background_enhanced = lp.enhance_contrast(
            self._background_rgb, self._display_mask()
        )
        self._background_detail_enhanced = self._stretched_detail(
            self._background_detail_plain, self._background_rgb
        )

    def _clear_background(self) -> None:
        self._background_path = None
        self._background_rgb = None
        self._background_plain = None
        self._background_enhanced = None
        self._background_detail_plain = None
        self._background_detail_enhanced = None
        self._background_token = None
        self._background_shown = False
        self._refresh_background_choice()

    def _refresh_background_choice(self) -> None:
        box = getattr(self, "background", None)
        if box is None:
            return
        box.blockSignals(True)
        try:
            box.clear()
            box.addItem("Reference photo", None)
            if self._background_path is not None:
                box.addItem(self._background_path.name, str(self._background_path))
            box.addItem("Choose photo...", _BACKGROUND_CHOOSE)
            box.setCurrentIndex(
                1 if self._background_shown and self._background_path is not None else 0
            )
            box.setEnabled(self.background_available())
        finally:
            box.blockSignals(False)

    def _background_activated(self, index: int) -> None:
        data = self.background.itemData(index)
        if data == _BACKGROUND_CHOOSE:
            self.choose_background_photo()
        else:
            self.show_background(data is not None)

    def _background_settings_key(self) -> str | None:
        reference = self.reference()
        if reference is None:
            return None
        root = str(getattr(self.project, "root", "") or "")
        digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]
        return f"background/{digest}/{reference.run_id}"

    def _remember_background(self, path: Path | None) -> None:
        key = self._background_settings_key()
        if key is None:
            return
        settings = QSettings(*_BACKGROUND_SETTINGS)
        if path is None:
            settings.remove(key)
        else:
            settings.setValue(key, str(path))
        settings.sync()

    def _restore_background(self) -> None:
        """Load the background photo last chosen for this run (not shown)."""

        key = self._background_settings_key()
        if key is None or not self.background_available():
            self._refresh_background_choice()
            return
        value = QSettings(*_BACKGROUND_SETTINGS).value(key, "")
        if value and Path(str(value)).is_file():
            self.set_background_photo(Path(str(value)), show=False, quiet=True)
        else:
            self._refresh_background_choice()

    def _sync_settings(self) -> None:
        session = self.session
        widgets = (self.smoothing, self.min_region, self.snap, self.detail, self.unit)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if session is not None:
                self.smoothing.setCurrentIndex(
                    max(0, self.smoothing.findData(session.smoothing))
                )
                self.min_region.setValue(float(session.min_region_cm2))
                self.snap.setValue(float(session.snap_mm))
                self.detail.setCurrentIndex(max(0, self.detail.findData(session.detail)))
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
            self.snap,
            self.detail,
            self.unit,
            self.add_facies_button,
            self.remove_facies_button,
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
        if (
            tool in {"polygon", "edit", "paint", "erase"}
            and self._other_photo is not None
        ):
            self.show_reference_photo()
        self.canvas.set_tool(tool)
        button = self.tool_buttons.get(tool)
        if button is not None:
            button.setChecked(True)
        painting = tool in {"paint", "erase"}
        self.brush_size.setEnabled(painting)
        self.brush_label.setEnabled(painting)

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
        self._prune_hidden_layers()
        table.blockSignals(True)
        try:
            current = self.canvas.layer_id
            session = self.session
            layers = session.layers if session is not None else []
            strokes = session.stroke_counts() if session is not None else {}
            polygons = session.polygon_counts() if session is not None else {}
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
                name.setToolTip(
                    (f"Key {row + 1}. " if row < 9 else "")
                    + f"Facies: {layer.facies}. Double-click to rename."
                )
                table.setItem(row, 1, name)
                for column, counts in ((2, polygons), (3, strokes)):
                    text = str(counts.get(layer.layer_id, 0))
                    if column == 2 and layer.snap_mm is not None:
                        # The layer's own snap distance, shown with its polygons.
                        text += f" · {layer.snap_mm:g} mm"
                    count = QTableWidgetItem(text)
                    count.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if column == 2 and layer.snap_mm is not None:
                        count.setToolTip(
                            f"This layer's own snap distance: {layer.snap_mm:g} mm. "
                            "Right-click to change it."
                        )
                    table.setItem(row, column, count)
                shown = QTableWidgetItem()
                shown.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                shown.setCheckState(
                    Qt.CheckState.Unchecked
                    if layer.layer_id in self._hidden_layers
                    else Qt.CheckState.Checked
                )
                shown.setToolTip(_SHOW_TIP)
                table.setItem(row, _SHOW_COLUMN, shown)
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
        self.canvas.set_layer(layer.layer_id, QColor(layer.color))

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
            self.set_tool("polygon")

    def _layer_item_changed(self, item: QTableWidgetItem) -> None:
        if self.session is None:
            return
        if item.column() == _SHOW_COLUMN:
            if item.row() < len(self.session.layers):
                self.set_layer_visible(
                    self.session.layers[item.row()].layer_id,
                    item.checkState() == Qt.CheckState.Checked,
                )
            return
        if item.column() != 1:
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
        self._render_marks()
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
        self.canvas.set_layer(layer.layer_id, QColor(layer.color))
        self._refresh_layers()
        self._refresh_facies_table()
        self._mark_dirty()
        self._refresh_status()
        row = len(self.session.layers) - 1
        self.layer_table.editItem(self.layer_table.item(row, 1))

    def _mark_text(self, layer_id: int) -> str:
        assert self.session is not None
        polygons = self.session.polygon_counts().get(layer_id, 0)
        strokes = self.session.stroke_counts().get(layer_id, 0)
        parts = []
        if polygons:
            parts.append(f"{polygons} polygon{'s' if polygons != 1 else ''}")
        if strokes:
            parts.append(f"{strokes} stroke{'s' if strokes != 1 else ''}")
        return " and ".join(parts)

    def remove_layer(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        assert layer is not None
        marks = self._mark_text(layer_id)
        if confirm:
            question = (
                f"Remove {layer.name} and its {marks}?"
                if marks
                else f"Remove {layer.name}?"
            )
            answer = QMessageBox.question(
                self, "Remove layer", question + " This cannot be undone."
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.canvas.cancel_polygon()
        self.canvas.clear_selection()
        self.session.remove_layer(layer_id)
        if layer_id in self._hidden_layers:
            # A later layer may get the same ID; it starts shown.
            self._hidden_layers.discard(layer_id)
            self.canvas.set_hidden_layers(self._hidden_layers)
        # Earlier snapshots may refer to the removed layer.
        self._undo_stack = []
        self._redo_stack = []
        if self.session.layers:
            first = self.session.layers[0]
            self.canvas.set_layer(first.layer_id, QColor(first.color))
        self._after_mark_change()
        self._refresh_facies_table()

    def clear_layer_marks(self, *, confirm: bool = True) -> None:
        layer_id = self.selected_layer_id()
        if self.session is None or layer_id is None:
            return
        layer = self.session.layer(layer_id)
        marks = self._mark_text(layer_id)
        if not marks or layer is None:
            return
        if confirm:
            answer = QMessageBox.question(
                self, "Clear marks", f"Remove the {marks} of {layer.name}?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # Removing polygons renumbers the others.
        self.canvas.clear_selection()
        self._push_undo()
        self.session.clear_layer_marks(layer_id)
        self._after_mark_change()

    def clear_layer_strokes(self, *, confirm: bool = True) -> None:
        """Former name of :meth:`clear_layer_marks`."""

        self.clear_layer_marks(confirm=confirm)

    # --------------------------------------------------- layer visibility

    @property
    def hidden_layers(self) -> frozenset[int]:
        """Layers hidden in the photograph (display only, never saved)."""

        return frozenset(self._hidden_layers)

    def layer_visible(self, layer_id: int) -> bool:
        return int(layer_id) not in self._hidden_layers

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        """Show or hide a layer in the photograph.

        Only the display changes: its polygons, strokes and proposed pixels.
        The session, the proposal and what is saved or exported stay the same.
        """

        hidden = set(self._hidden_layers)
        if visible:
            hidden.discard(int(layer_id))
        else:
            hidden.add(int(layer_id))
        self._set_hidden_layers(hidden)

    def show_only_layer(self, layer_id: int) -> None:
        if self.session is None or self.session.layer(layer_id) is None:
            return
        self._set_hidden_layers(
            row.layer_id for row in self.session.layers if row.layer_id != layer_id
        )

    def show_all_layers(self) -> None:
        self._set_hidden_layers(())

    def _set_hidden_layers(self, layer_ids: Iterable[int]) -> None:
        known = (
            {row.layer_id for row in self.session.layers}
            if self.session is not None
            else set()
        )
        hidden = {int(value) for value in layer_ids} & known
        if hidden != self._hidden_layers:
            self._hidden_layers = hidden
            self.canvas.set_hidden_layers(hidden)
            self._render_proposal()
        self._sync_show_column()

    def _prune_hidden_layers(self) -> None:
        """Hidden layers belong to the session shown: another session (another
        run or an imported map) shows every layer; removed layers are dropped.
        Another session also starts with no point selected."""

        session = self.session
        if session is not self._hidden_session:
            self._hidden_session = session
            hidden: set[int] = set()
            self.canvas.clear_selection()
        elif session is None:
            hidden = set()
        else:
            hidden = self._hidden_layers & {row.layer_id for row in session.layers}
        if hidden != self._hidden_layers:
            self._hidden_layers = hidden
            self.canvas.set_hidden_layers(hidden)

    def _sync_show_column(self) -> None:
        """Tick the Show box of every shown layer."""

        if self.session is None:
            return
        table = self.layer_table
        table.blockSignals(True)
        try:
            for row, layer in enumerate(self.session.layers):
                item = table.item(row, _SHOW_COLUMN)
                state = (
                    Qt.CheckState.Unchecked
                    if layer.layer_id in self._hidden_layers
                    else Qt.CheckState.Checked
                )
                if item is not None and item.checkState() != state:
                    item.setCheckState(state)
        finally:
            table.blockSignals(False)

    def _reveal_layer(self, layer_id: int) -> None:
        """A layer that gets a new mark is shown, so that the mark can be seen."""

        if int(layer_id) not in self._hidden_layers or self.session is None:
            return
        self.set_layer_visible(layer_id, True)
        layer = self.session.layer(layer_id)
        if layer is not None:
            self._canvas_notice(f"{layer.name} is shown again for its new mark.")

    def _layer_table_menu(self, position: QPoint) -> None:
        """Right-click menu of the layer list: which layers are shown."""

        if self.session is None:
            return
        row = self.layer_table.indexAt(position).row()
        layer = (
            self.session.layers[row] if 0 <= row < len(self.session.layers) else None
        )
        menu = QMenu(self)
        only = menu.addAction("Show only this layer")
        only.setEnabled(layer is not None)
        if layer is not None:
            only.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self.show_only_layer(
                    layer_id
                )
            )
        every = menu.addAction("Show all layers")
        every.setEnabled(bool(self._hidden_layers))
        every.triggered.connect(lambda _checked=False: self.show_all_layers())
        menu.addSeparator()
        own = menu.addAction("Snap distance of this layer…")
        own.setEnabled(layer is not None)
        common = menu.addAction("Use the common snap distance")
        common.setEnabled(layer is not None and layer.snap_mm is not None)
        if layer is not None:
            own.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self._ask_layer_snap(layer_id)
            )
            common.triggered.connect(
                lambda _checked=False, layer_id=layer.layer_id: self.set_layer_snap(
                    layer_id, None
                )
            )
        self._layer_table_menu_widget = menu
        menu.popup(self.layer_table.viewport().mapToGlobal(position))

    def _ask_layer_snap(self, layer_id: int) -> None:
        """Ask for one layer's own snap distance and set it."""

        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if layer is None:
            return
        value, accepted = QInputDialog.getDouble(
            self,
            "Snap distance of this layer",
            f"Snap distance for {layer.name} (mm). 0 keeps its polygons exactly, "
            "for thin layers; the common distance is "
            f"{self.session.snap_mm:g} mm.",
            self.session.layer_snap_mm(layer_id),
            0.0,
            lp.MAX_SNAP_MM,
            1,
        )
        if accepted:
            self.set_layer_snap(layer_id, float(value))

    def set_layer_snap(self, layer_id: int, snap_mm: float | None) -> None:
        """Give one layer its own snap distance, or the common one with None."""

        if self.session is None:
            return
        layer = self.session.layer(layer_id)
        if layer is None:
            return
        value = None if snap_mm is None else float(min(max(snap_mm, 0.0), lp.MAX_SNAP_MM))
        if layer.snap_mm == value:
            return
        layer.snap_mm = value
        self._mark_dirty()
        self._refresh_layers()
        self._refresh_status()

    # ------------------------------------------------------------- marks

    def _snapshot(self) -> tuple[list[lp.Polygon], list[lp.Stroke]]:
        assert self.session is not None
        return (
            [
                lp.Polygon(row.layer_id, list(row.points))
                for row in self.session.polygons
            ],
            [
                lp.Stroke(row.layer_id, row.radius_px, list(row.points))
                for row in self.session.strokes
            ],
        )

    def _restore(self, snapshot: tuple[list[lp.Polygon], list[lp.Stroke]]) -> None:
        assert self.session is not None
        polygons, strokes = snapshot
        self.session.polygons = [
            lp.Polygon(row.layer_id, list(row.points)) for row in polygons
        ]
        self.session.strokes = [
            lp.Stroke(row.layer_id, row.radius_px, list(row.points)) for row in strokes
        ]

    def _push_undo(self) -> None:
        if self.session is None:
            return
        self._undo_stack.append(self._snapshot())
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []

    def _stroke_finished(self, stroke: lp.Stroke) -> None:
        if self.session is None:
            return
        snapshot = self._snapshot()
        try:
            self.session.add_stroke(stroke)
        except ValueError:
            self._render_marks()
            return
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._reveal_layer(stroke.layer_id)
        self.canvas.add_stroke_to_image(stroke, self._layer_colors())
        self._after_mark_change(render=False)

    def add_stroke(self, stroke: lp.Stroke) -> None:
        """Add a stroke as if it had been painted (used by scripts and tests)."""

        self._stroke_finished(stroke)

    def _polygon_finished(self, polygon: lp.Polygon) -> None:
        if self.session is None:
            return
        snapshot = self._snapshot()
        try:
            self.session.add_polygon(polygon)
        except ValueError as error:
            self._canvas_notice(str(error))
            return
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._reveal_layer(polygon.layer_id)
        self._after_mark_change()

    def add_polygon(self, polygon: lp.Polygon) -> None:
        """Add a polygon as if it had been drawn (used by scripts and tests)."""

        self._polygon_finished(polygon)

    def _polygon_edited(
        self, index: int, points: Sequence[tuple[float, float]]
    ) -> None:
        self._polygons_edited({index: points})

    def _polygons_edited(
        self, changes: Mapping[int, Sequence[tuple[float, float]]]
    ) -> list[int]:
        """Give one or more polygons new points, as one change to undo.

        A polygon whose new points are not valid (fewer than three, or no
        area) keeps its points and the hint line says why; the others still
        change. Returns the indices of the polygons that changed.
        """

        if self.session is None:
            return []
        snapshot = self._snapshot()
        applied: list[int] = []
        problems: list[str] = []
        for index, points in sorted(dict(changes).items()):
            index = int(index)
            if not 0 <= index < len(self.session.polygons):
                continue
            layer_id = self.session.polygons[index].layer_id
            try:
                self.session.replace_polygon(index, lp.Polygon(layer_id, list(points)))
            except ValueError as error:
                problems.append(str(error))
                continue
            applied.append(index)
        if not applied:
            if problems:
                self._canvas_notice(problems[0])
                self._render_marks()
            return []
        self._undo_stack.append(snapshot)
        del self._undo_stack[:-_UNDO_LIMIT]
        self._redo_stack = []
        self._after_mark_change()
        if problems:
            self._canvas_notice(problems[0])
        return applied

    def delete_polygon(self, index: int) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        # Deleting a polygon renumbers the polygons after it.
        self.canvas.clear_selection()
        self._push_undo()
        self.session.remove_polygon(index)
        self._after_mark_change()

    def move_polygon(self, index: int, layer_id: int) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        if self.session.polygons[index].layer_id == layer_id:
            return
        self.canvas.clear_selection()
        self._push_undo()
        self.session.move_polygon(index, layer_id)
        self._reveal_layer(layer_id)
        self._after_mark_change()

    def remove_polygon_point(self, index: int, vertex: int) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        polygon = self.session.polygons[index]
        if len(polygon.points) <= 3 or not 0 <= vertex < len(polygon.points):
            return
        points = [
            point for number, point in enumerate(polygon.points) if number != vertex
        ]
        # The other selected points keep their selection.
        selection = self.canvas.selection_without(index, (vertex,))
        self._polygon_edited(index, points)
        if self.session.polygons[index].points == points:
            self.canvas.set_selection(selection)

    def remove_selected_points(self) -> int:
        """Remove the points selected with the Edit tool, as one change to undo.

        A polygon that would keep fewer than three points is left unchanged
        and the hint line says so. Returns how many points were removed.
        """

        return self.canvas.delete_selected_points()

    def add_polygon_point(self, index: int, row: float, col: float) -> int:
        """Add a point to a polygon on its edge nearest to (row, col).

        (row, col) is in raw pixels, for example where the polygon was
        right-clicked; the point goes where it projects onto that edge. Like a
        moved point, the change can be undone. With the Edit tool the new
        point is selected. Returns the index of the new point, or -1.
        """

        if self.session is None or not 0 <= index < len(self.session.polygons):
            return -1
        points, inserted = insert_point_on_nearest_edge(
            self.session.polygons[index].points, row, col
        )
        if inserted < 0:
            return -1
        self._polygon_edited(index, points)
        if self.session.polygons[index].points != points:
            return -1  # refused; the notice says why
        self.canvas.set_selection([(index, inserted)])
        return inserted

    def _polygon_menu(self, index: int, vertex: int, position: QPoint) -> None:
        if self.session is None or not 0 <= index < len(self.session.polygons):
            return
        polygon = self.session.polygons[index]
        layer = self.session.layer(polygon.layer_id)
        menu = QMenu(self)
        title = menu.addAction(f"Polygon of {layer.name if layer else 'a layer'}")
        title.setEnabled(False)
        if vertex >= 0:
            remove_point = menu.addAction("Remove this point")
            remove_point.setEnabled(len(polygon.points) > 3)
            remove_point.triggered.connect(
                lambda _checked=False: self.remove_polygon_point(index, vertex)
            )
            selected = self.canvas.selected_points
            if len(selected) > 1 and (index, vertex) in selected:
                remove_selected = menu.addAction(
                    f"Remove {len(selected)} selected points"
                )
                remove_selected.triggered.connect(
                    lambda _checked=False: self.remove_selected_points()
                )
        elif self.canvas.last_context_raw is not None:
            # Where the polygon was right-clicked, in raw pixels.
            clicked = self.canvas.last_context_raw
            add_point = menu.addAction("Add point here")
            add_point.triggered.connect(
                lambda _checked=False, at=clicked: self.add_polygon_point(
                    index, at[0], at[1]
                )
            )
        move = menu.addMenu("Move to layer")
        for row in self.session.layers:
            action = move.addAction(_swatch(row.color), row.name)
            action.setEnabled(row.layer_id != polygon.layer_id)
            action.triggered.connect(
                lambda _checked=False, target=row.layer_id: self.move_polygon(
                    index, target
                )
            )
        menu.addSeparator()
        delete = menu.addAction("Delete polygon")
        delete.triggered.connect(lambda _checked=False: self.delete_polygon(index))
        self._polygon_menu_widget = menu
        menu.popup(position)

    def _draft_changed(self, count: int) -> None:
        if count:
            self.canvas_hint.setText(
                _DRAFT_HINT.format(count=count, plural="s" if count != 1 else "")
            )
        else:
            self.canvas_hint.setText(_CANVAS_HINT)

    def _canvas_notice(self, message: str) -> None:
        self.canvas_hint.setText(str(message))

    def show_frame_hint(self, text: str) -> None:
        """What to click while the rig frame panel marks corners or points."""

        if getattr(self, "canvas_hint", None) is None:
            return  # the panel is created before the canvas column
        if text:
            self.canvas_hint.setText(text)
        elif self.canvas.drafting():
            self._draft_changed(len(self.canvas.draft_points))
        else:
            self.canvas_hint.setText(_CANVAS_HINT)

    def undo(self) -> None:
        if self.session is None:
            return
        if self.canvas.drafting():
            self.canvas.remove_last_point()
            return
        if not self._undo_stack:
            return
        current, snapshot = self._snapshot(), self._undo_stack.pop()
        if snapshot[0] != current[0]:
            self.canvas.clear_selection()  # the polygons change
        self._redo_stack.append(current)
        self._restore(snapshot)
        self._after_mark_change()

    def redo(self) -> None:
        if self.session is None or not self._redo_stack:
            return
        current, snapshot = self._snapshot(), self._redo_stack.pop()
        if snapshot[0] != current[0]:
            self.canvas.clear_selection()  # the polygons change
        self._undo_stack.append(current)
        self._restore(snapshot)
        self._after_mark_change()

    def _after_mark_change(self, *, render: bool = True) -> None:
        if render:
            self._render_marks()
        self._refresh_layers()
        self._mark_dirty()
        self._refresh_status()

    def _render_polygons(self) -> None:
        if self.session is None:
            self.canvas.render_polygons([], {})
            return
        self.canvas.render_polygons(self.session.polygons, self._layer_colors())

    def _render_marks(self) -> None:
        if self.session is None:
            self.canvas.render_strokes([], {})
            self.canvas.render_polygons([], {})
            return
        colors = self._layer_colors()
        self.canvas.render_strokes(self.session.strokes, colors)
        self.canvas.render_polygons(self.session.polygons, colors)

    def _render_strokes(self) -> None:
        """Former name of :meth:`_render_marks`."""

        self._render_marks()

    # ---------------------------------------------------------- proposal

    def _settings_changed(self, *_args: Any) -> None:
        if self.session is None or self._loading:
            return
        self.session.smoothing = str(self.smoothing.currentData() or "medium")
        self.session.min_region_cm2 = float(self.min_region.value())
        self.session.snap_mm = float(self.snap.value())
        self.session.detail = str(self.detail.currentData() or "normal")
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
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.propose_status.setText("Saved corrected rig maps are shown. The reference labels remain editable in Advanced; production registration is a separate approved step.")
            return
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
            self._proposal_image(),
            snapshot,
            self.current_frame(),
            lp.proposal_path(self.project, reference.run_id),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.propose_status.setText)
        worker.finished.connect(self._proposal_worker_finished)
        worker.failed.connect(self._proposal_worker_failed)
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
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

    def _proposal_image(self) -> lp.PaintingImage:
        """The reference photograph on the grid the session's detail asks for.

        Fine uses the displayed photograph as the analysis grid. That image is
        kept while the photograph stays, so its features are computed once.
        """

        assert self.image is not None and self.session is not None
        if self.session.detail != "fine":
            return self.image
        if self._fine_image is None or self._fine_image[0] is not self.image:
            fine = lp.with_analysis_width(self.image, int(self.image.display.shape[1]))
            self._fine_image = (self.image, fine)
        return self._fine_image[1]

    @Slot(object)
    def _proposal_worker_finished(self, proposal: lp.LayerProposal) -> None:
        """Hand a finished proposal to _proposal_ready in this view's thread.

        The worker's signal reaches a slot of the view, so Qt queues it into
        the GUI thread. _proposal_ready may be replaced by an extension's
        plain function, which Qt would call in the worker's thread, so the
        worker never connects to it directly.
        """

        self._proposal_ready(proposal)

    @Slot(str)
    def _proposal_worker_failed(self, message: str) -> None:
        """Hand a failed proposal's message to _proposal_failed in this view's thread."""

        self._proposal_failed(message)

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
        if (workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected"
                and self._other_photo is None and self.proposal is None):
            self.canvas.set_proposal(self._saved_corrected_boundaries)
            return
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
        # Hidden layers are fully transparent (display only).
        hidden = np.zeros(256, dtype=bool)
        for layer_id in self._hidden_layers:
            if 0 < layer_id < 256:
                hidden[layer_id] = True
        table[hidden] = 0
        rgba = table[classes]
        shown = ~hidden[classes]
        # A boundary pixel is drawn where it touches a shown layer (or the area
        # outside the rig frame); between hidden layers there is none.
        boundary = np.zeros(classes.shape, dtype=bool)
        boundary[:, 1:] |= (classes[:, 1:] != classes[:, :-1]) & (
            shown[:, 1:] | shown[:, :-1]
        )
        boundary[1:, :] |= (classes[1:, :] != classes[:-1, :]) & (
            shown[1:, :] | shown[:-1, :]
        )
        rgba[boundary] = (255, 255, 255, 235)
        if self.show_uncertain.isChecked():
            uncertain = cv2.resize(
                self.proposal.uncertain().astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            rows, cols = np.indices(classes.shape)
            hatch = ((rows + cols) // 3) % 2 == 0
            rgba[uncertain & hatch & shown] = (255, 0, 255, 200)
        self.canvas.set_proposal(rgba)
        self._update_polygon_fill()

    def _display_changed(self, *_args: Any) -> None:
        reference_shown = self._other_photo is None
        self.canvas.set_overlays_visible(
            self.show_strokes.isChecked() and reference_shown,
            self.show_proposal.isChecked() and reference_shown,
        )
        self._update_polygon_fill()
        sender = self.sender()
        if sender in (self.opacity, self.show_uncertain):
            self._render_proposal()

    def _update_polygon_fill(self) -> None:
        """Outlines only where the proposal already colours the layers."""

        self.canvas.set_polygon_fill(
            not (
                self.proposal is not None
                and self.show_proposal.isChecked()
                and self._other_photo is None
            )
        )

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
        """Rebuild the facies table: name, layers, role and properties."""

        if self._facies_menu_open:
            # A layer list is open; rebuild once it has closed.
            self._facies_refresh_pending = True
            return
        session = self.session
        names = session.facies_order() if session is not None else []
        selected = self.selected_facies()
        table = self.facies_table
        table.blockSignals(True)
        try:
            unit = str(self.unit.currentData() or "darcy")
            table.setHorizontalHeaderLabels(
                ["Facies", "Layers", "Role", "Porosity", f"Perm. [{unit}]"]
            )
            for column, tip in (
                (3, "Porosity [-], between 0 and 1"),
                (4, f"Permeability [{unit}]"),
            ):
                header_item = table.horizontalHeaderItem(column)
                if header_item is not None:
                    header_item.setToolTip(tip)
            self._clear_cell_widgets(table)
            self._facies_buttons = {}
            table.setRowCount(len(names))
            for row, name in enumerate(names):
                assert session is not None
                entry = session.facies.get(name, lp.FaciesEntry())
                members = session.layers_in_facies(name)
                item = QTableWidgetItem(
                    _swatch(members[0].color) if members else QIcon(), name
                )
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
                if members:
                    item.setToolTip(
                        "Double-click to rename; the name of another facies merges "
                        "the two."
                    )
                else:
                    item.setForeground(_muted(self))
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setToolTip(
                        "No layers: tick layers in the Layers list, or remove the "
                        "facies. Facies without layers are not exported."
                    )
                table.setItem(row, 0, item)
                # The cell behind the layer list keeps the facies for selection.
                holder = QTableWidgetItem()
                holder.setData(Qt.ItemDataRole.UserRole, name)
                holder.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                table.setItem(row, 1, holder)
                table.setCellWidget(row, 1, self._layers_button(name))
                role = QComboBox()
                for key, text in lp.LAYER_ROLES.items():
                    role.addItem(_ROLE_SHORT.get(key, text), key)
                    role.setItemData(
                        role.count() - 1, text, Qt.ItemDataRole.ToolTipRole
                    )
                role.setCurrentIndex(max(0, role.findData(entry.role)))
                role.setToolTip(
                    "Porous medium needs porosity and permeability. In mass "
                    "integrals free water has porosity 1 and excluded regions "
                    "(ColorChecker, tape, fittings) 0."
                )
                role.currentIndexChanged.connect(
                    lambda _index, combo=role, facies=name: self._role_edited(
                        combo, facies
                    )
                )
                table.setCellWidget(row, 2, role)
                porous = entry.role == "porous_medium"
                fixed = entry.effective_porosity()
                for column, value in ((3, entry.porosity), (4, entry.permeability)):
                    if porous:
                        text = _format_number(value)
                    elif column == 3 and fixed is not None:
                        text = _format_number(fixed)
                    else:
                        text = "—"
                    cell = QTableWidgetItem(text)
                    cell.setData(Qt.ItemDataRole.UserRole, name)
                    if not porous and column == 3 and fixed is not None:
                        cell.setToolTip(
                            "Free water fills the whole volume: porosity 1 in mass "
                            "integrals."
                            if fixed > 0.0
                            else "Left out of mass integrals: porosity 0."
                        )
                    if porous:
                        cell.setFlags(
                            Qt.ItemFlag.ItemIsEnabled
                            | Qt.ItemFlag.ItemIsSelectable
                            | Qt.ItemFlag.ItemIsEditable
                        )
                        invalid = (
                            value is None
                            or (column == 3 and not 0.0 < value <= 1.0)
                            or (column == 4 and value <= 0.0)
                        )
                        if invalid and members:
                            cell.setBackground(QColor(201, 138, 20, 70))
                            cell.setToolTip(
                                "Porosity between 0 and 1"
                                if column == 3
                                else "A positive permeability"
                            )
                    else:
                        cell.setFlags(
                            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                        )
                        cell.setForeground(_muted(self))
                    table.setItem(row, column, cell)
                if name == selected:
                    table.selectRow(row)
        finally:
            table.blockSignals(False)
        _fit_table_height(table, max_rows=10)
        self._facies_selection_changed()
        self._refresh_status()

    def _layers_text(self, facies: str) -> str:
        if self.session is None:
            return ""
        members = self.session.layers_in_facies(facies)
        if not members:
            return "No layers: tick…"
        return ", ".join(layer.name for layer in members)

    def _layers_button(self, facies: str) -> QToolButton:
        button = QToolButton()
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        button.setText(self._layers_text(facies))
        button.setToolTip(
            "Tick the layers of this facies. A layer belongs to one facies: ticking "
            "moves it here, unticking gives it a facies of its own."
        )
        menu = QMenu(button)
        menu.aboutToShow.connect(
            lambda target=menu, name=facies: self._fill_layer_menu(target, name)
        )
        menu.aboutToHide.connect(self._layer_menu_closed)
        button.setMenu(menu)
        self._facies_buttons[facies] = button
        return button

    def _fill_layer_menu(self, menu: QMenu, facies: str) -> None:
        self._facies_menu_open = True
        menu.clear()
        if self.session is None:
            return
        for layer in self.session.layers:
            box = QCheckBox(layer.name)
            box.setIcon(_swatch(layer.color))
            box.setChecked(layer.facies == facies)
            if layer.facies != facies and layer.facies != layer.name:
                box.setText(f"{layer.name}   (now in {layer.facies})")
            box.toggled.connect(
                lambda checked, layer_id=layer.layer_id, name=facies: (
                    self.set_layer_in_facies(layer_id, name, checked)
                )
            )
            holder = QWidget()
            row = QHBoxLayout(holder)
            row.setContentsMargins(10, 3, 12, 3)
            row.addWidget(box)
            action = QWidgetAction(menu)
            action.setDefaultWidget(holder)
            menu.addAction(action)

    def _layer_menu_closed(self) -> None:
        self._facies_menu_open = False
        # The menu belongs to a button in the table: rebuild after it has closed.
        if self._facies_refresh_pending:
            self._facies_refresh_pending = False
            QTimer.singleShot(0, self._refresh_facies_table)

    def selected_facies(self) -> str:
        rows = {index.row() for index in self.facies_table.selectedIndexes()}
        if not rows:
            return ""
        item = self.facies_table.item(min(rows), 0)
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _facies_selection_changed(self) -> None:
        self.remove_facies_button.setEnabled(
            self.session is not None and bool(self.selected_facies())
        )

    def _role_edited(self, box: QComboBox, facies: str) -> None:
        try:
            role = str(box.currentData())
        except RuntimeError:
            return
        self.set_facies_role(facies, role)

    def _facies_changed(self) -> None:
        self._mark_dirty()
        if self._facies_menu_open:
            self._facies_refresh_pending = True
        else:
            QTimer.singleShot(0, self._refresh_facies_table)
        self._refresh_layers()

    def set_layer_in_facies(self, layer_id: int, facies: str, member: bool) -> None:
        """Tick or untick a layer in a facies' layer list."""

        if self.session is None:
            return
        self.session.set_layer_in_facies(layer_id, facies, bool(member))
        button = self._facies_buttons.get(facies)
        if button is not None:
            try:
                button.setText(self._layers_text(facies))
            except RuntimeError:  # the table was rebuilt
                pass
        self._facies_changed()

    def set_layer_facies(self, layer_id: int, facies: str) -> None:
        """Move a layer into a facies (created when new)."""

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
        self._facies_changed()

    def add_facies(self, name: str | None = None) -> str:
        if self.session is None:
            return ""
        try:
            created = self.session.add_facies(name)
        except ValueError as error:
            QMessageBox.information(self, "Facies name in use", str(error))
            return ""
        self._mark_dirty()
        self._refresh_facies_table()
        row = self.session.facies_order().index(created)
        self.facies_table.selectRow(row)
        if name is None:
            self.facies_table.editItem(self.facies_table.item(row, 0))
        return created

    def remove_facies(self, name: str | None = None, *, confirm: bool = True) -> None:
        if self.session is None:
            return
        facies = name or self.selected_facies()
        if facies not in self.session.facies:
            return
        members = self.session.layers_in_facies(facies)
        if confirm and members:
            answer = QMessageBox.question(
                self,
                "Remove facies",
                f"Remove the facies {facies}? Its layers ("
                + ", ".join(layer.name for layer in members)
                + ") get a facies of their own.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.session.remove_facies(facies)
        self._facies_changed()

    def rename_facies(self, old: str, new: str) -> None:
        if self.session is None or not str(new).strip() or old == str(new).strip():
            return
        self.session.rename_facies(old, new)
        self._facies_changed()

    def set_facies_role(self, facies: str, role: str) -> None:
        if self.session is None or facies not in self.session.facies:
            return
        if self.session.facies[facies].role == role:
            return
        self.session.set_facies(facies, role=role)
        self._facies_changed()

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
        if item.column() not in (3, 4):
            return
        value = _parse_number(item.text())
        if item.column() == 3:
            self.session.set_facies(facies, porosity=value)
        else:
            self.session.set_facies(facies, permeability=value)
        self._mark_dirty()
        QTimer.singleShot(0, self._refresh_facies_table)

    def _present_layers(self) -> set[int] | None:
        if self.proposal is None:
            return None
        return {
            int(row["layer_id"]) for row in self.proposal.layers if row.get("pixels")
        }

    def _unit_changed(self, _index: int) -> None:
        if self.session is None or self._loading:
            return
        self.session.permeability_unit = str(self.unit.currentData() or "darcy")
        self._mark_dirty()
        self._refresh_facies_table()

    # ------------------------------------------------------------ export

    def export_problems_now(self) -> list[str]:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            return ["Corrected-frame production export and Build run model are not qualified yet."]
        if self._raw_session_frame:
            return [_RAW_SESSION_EXPORT_NOTE]
        if self.session is None:
            return ["Choose a reference run with a baseline photograph."]
        return lp.check_ready_for_revision(
            self.session, self.proposal, self.current_frame(), self.reference()
        )

    def save_and_export(self) -> None:
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.export_status.setText("Corrected-frame production export and Build run model are not qualified yet.")
            return
        if self._raw_session_frame:
            self.export_status.setText(_RAW_SESSION_EXPORT_NOTE)
            return
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
        # Direct: quit from the worker thread, so that wait() in the GUI
        # thread cannot block a queued quit.
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(thread.quit, Qt.ConnectionType.DirectConnection)
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
        marked = session.marked_layers()
        parts = [f"{len(marked)} of {len(session.layers)} layers marked"]
        if session.polygons:
            parts.append(f"{len(session.polygons)} polygon(s)")
        if session.strokes:
            parts.append(f"{len(session.strokes)} stroke(s)")
        self.step_paint.set_status(
            " · ".join(parts), "done" if len(marked) >= 2 else "warn"
        )
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))

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
        if workspace_frame.frame_kind(self._workspace_frame_config) == "rig-corrected":
            self.propose_button.setEnabled(False)
            self.rig_panel.setEnabled(False)
            if self.image is not None and self._saved_corrected_boundaries is not None:
                self.step_propose.set_status("Saved rig boundaries", "done")
                self.proposal_summary.setText("Read-only old rig maps; no classifier or registration was run.")
            else:
                self.step_propose.set_status("Saved rig maps unavailable", "warn")
                self.proposal_summary.setText("No raw-photo or common-map fallback was used.")

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
        sources = []
        if proposal.polygon_count:
            sources.append(f"{proposal.polygon_count} polygon(s)")
        if proposal.stroke_count or not sources:
            sources.append(f"{proposal.stroke_count} stroke(s)")
        text = (
            "From "
            + " and ".join(sources)
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


__all__ = [
    "LayerCanvas",
    "LayerPainterView",
    "StepSection",
    "insert_point_on_nearest_edge",
]
