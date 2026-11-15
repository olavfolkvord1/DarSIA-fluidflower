"""One viewer of stored results: run, frame, field and colour scale.

The Calibration page and the Mass page show their stored results in this one
viewer. A page declares what it shows: each frame's stored results
(:class:`Source`, e.g. the mass maps) with the fields they hold (:class:`Field`).
The viewer lists the runs, steps through a run's frames and shows one to three
panels, each a field of the shown frame, with a colour scale kept per run and
field.

It is display only: it computes no result, starts no job and writes nothing. Each
result is read through :func:`darsia.utils.results.read` at its location
(:class:`darsia.utils.locations.Locations`) with its method's own reader, and is
read and drawn off the window thread, the latest request first, while a busy bar
with the elapsed seconds shows that it is under way. A result that is not stored
is told apart from one the share does not answer for now
(:class:`darsia.utils.results.Unavailable`).

"""

import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PySide6.QtCore import QLocale, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QDoubleValidator, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.gui.ui.image_canvas import ImageCanvas
from darsia.gui.ui.widgets import NoScrollComboBox
from darsia.utils import results
from darsia.utils.locations import Frame, Locations

DISPLAY_PIXELS = 1280
"""The longest side of a shown map in cells; a larger stored grid shows every n-th
cell."""
PANEL_PX = (1280, 720)
"""Width and height in pixels of a drawn panel."""
CACHED = 8
"""Display copies of stored frames kept in memory."""
SETTLE_MS = 120
"""Milliseconds the frame slider rests before its frame is read."""
POLL_MS = 100
"""Milliseconds between two looks at a read under way."""
PLAY_MS = 600
"""Milliseconds between two frames while Play runs."""
BACKGROUND = "#e6edf2"
"""Colour of cells without a value."""
OTHER = ("Other label", "#555b61")
"""Legend name and colour of a categorical code the field does not declare."""
LAYOUTS = {
    "one": ("One panel", ((0, 0, 1, 2),)),
    "side_by_side": ("Side by side", ((0, 0, 1, 1), (0, 1, 1, 1))),
    "pyramid": ("Three panels, pyramid", ((0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1))),
}
"""The panel layouts: name, and each panel's row, column, row span and column
span."""

# Matplotlib's font and text caches are shared by every figure: one figure is
# drawn at a time, also by two viewers.
_RENDER = threading.Lock()


@dataclass(frozen=True)
class Category:
    """One code of a categorical field, e.g. one phase of a phase map."""

    code: int
    label: str
    colour: str


@dataclass(frozen=True)
class Field:
    """One array of a frame's stored result that a panel can show."""

    key: str
    """Its name in what the result's reader returns, a key or an attribute, e.g.
    ``mass`` of the official ``SimpleMassAnalysisResults``."""
    label: str
    unit: str = ""
    limit: float | None = None
    """The physical maximum the colour scale starts at, e.g. 1 for a saturation;
    without it the scale is fitted to the first frame shown."""
    classes: tuple[Category, ...] = ()
    """The codes of a categorical field, drawn in their colours; empty for a
    continuous field."""
    photo: bool = False
    """Whether the array is a colour image (rows, columns, 3), drawn as it is."""


@dataclass(frozen=True)
class Source:
    """A stored result of each frame, e.g. the mass maps, and the fields it holds."""

    kind: str
    """The kind of the result (:meth:`darsia.utils.locations.Locations.result`)."""
    label: str
    reader: Callable[[Path], Any]
    """The method's own reader of the result's folder; it holds no Qt object, since
    it runs off the window thread."""
    fields: tuple[Field, ...]


def _nice_limit(value: float) -> float:
    """A colour-scale maximum: the value rounded up to two significant digits.

    A float32 value just above a round number, e.g. 1.2000000477, stays at it.

    Parameters
    ----------
    value : float
        The largest value shown.

    Returns
    -------
    float
        The maximum; 1 for a value that is not finite and positive.

    """
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return 1.0
    step = 10.0 ** (math.floor(math.log10(value)) - 1)
    return float(f"{math.ceil(value / step * (1 - 1e-6)) * step:.3g}")


def _fit(values: np.ndarray) -> float:
    """The colour-scale maximum of the finite values shown (1 without any)."""
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return _nice_limit(finite.max()) if finite.size else 1.0


def _hours(hours: float | None) -> str:
    """An elapsed time as text."""
    if hours is None or not math.isfinite(hours):
        return "time not recorded"
    return f"{hours:.2f} h"


class _Worker:
    """Runs the latest request off the window thread, one at a time.

    A daemon thread runs while a request waits and ends when none does, so a
    read that waits for the share never holds up the window or its closing. A
    request is a function of plain values: nothing of Qt is used, kept or freed in
    the thread, and the window takes the answer when its timer looks.

    """

    def __init__(self) -> None:
        """No thread until the first request."""
        self._lock = threading.Lock()
        self._waiting: tuple[int, Callable[[], Any]] | None = None
        self._answer: tuple[int, Any, Exception | None] | None = None
        self._running = False
        self._number = 0

    def submit(self, work: Callable[[], Any]) -> int:
        """Run ``work`` after the request under way, in place of one still waiting;
        returns the request's number."""
        with self._lock:
            self._number += 1
            self._waiting = (self._number, work)
            if not self._running:
                self._running = True
                threading.Thread(target=self._run, daemon=True).start()
            return self._number

    def answer(self, number: int) -> tuple[int, Any, Exception | None] | None:
        """The answer to request ``number`` once it is there: the number, the
        value and the error; None before."""
        with self._lock:
            if self._answer is None or self._answer[0] != number:
                return None
            answer, self._answer = self._answer, None
            return answer

    def _run(self) -> None:
        """Run the waiting requests until none waits."""
        while True:
            with self._lock:
                if self._waiting is None:
                    self._running = False
                    return
                (number, work), self._waiting = self._waiting, None
            try:
                answer = (number, work(), None)
            except Exception as error:
                answer = (number, None, error)
            with self._lock:
                self._answer = answer


@dataclass
class _Drawn:
    """One drawn panel: its pixels or a message, its scale and the values shown."""

    pixels: np.ndarray | None
    message: str = ""
    scale: float | None = None
    values: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class _Request:
    """What drawing one frame needs, as plain values."""

    locations: Locations
    run: str
    index: int
    frames: tuple[tuple[Frame, float | None], ...]
    panels: tuple[tuple[Source, Field, float | None], ...]
    cache: OrderedDict


def _value(stored: Any, key: str) -> Any:
    """A field of a stored result, by key or attribute; None if it has none."""
    if isinstance(stored, Mapping):
        return stored.get(key)
    return getattr(stored, key, None)


def _display(value: Any) -> tuple[np.ndarray, tuple | None, int]:
    """A display copy of a stored array: every n-th cell, so that its longest side
    is at most :data:`DISPLAY_PIXELS`, with its extent in metres (left, right,
    bottom, top) when it is a ``darsia.Image``, and n."""
    extent = None
    if hasattr(value, "img"):
        extent = tuple(float(x) for x in value.domain)
        value = value.img
    array = np.asarray(value)
    step = max(1, math.ceil(max(array.shape[:2]) / DISPLAY_PIXELS))
    return np.ascontiguousarray(array[::step, ::step]), extent, step


def _cached(request: _Request, key: tuple, what: str, read: Callable[[], Any]) -> Any:
    """A display copy from the cache or from a read, or the message why there is
    none. The key holds the result's folder, so another results root is read anew;
    a failed read is not kept, so choosing the frame again reads again."""
    cache = request.cache
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    try:
        found = read()
    except FileNotFoundError:
        return f"{what} is not stored."
    except results.Unavailable as error:
        return (
            f"{what} is not available now: the results folder gave no answer "
            f"({error}). Choose the frame again to read it again."
        )
    except Exception as error:
        return f"{what} could not be read: {error}"
    cache[key] = found
    while len(cache) > CACHED:
        cache.popitem(last=False)
    return found


def _read_frame(request: _Request, source: Source, frame: Frame) -> dict | str:
    """The display copies of a frame's stored result, by field key, or the message
    why there are none."""
    location = request.locations.result(source.kind, run=request.run, frame=frame)

    def read() -> dict:
        stored = results.read(location, source.reader)
        values = {field.key: _value(stored, field.key) for field in source.fields}
        return {
            key: _display(value) for key, value in values.items() if value is not None
        }

    what = f"{source.label} of {request.run}, {frame.session}/{frame.name},"
    return _cached(request, (location,), what, read)


def _figure() -> tuple[Figure, Any]:
    """A figure of :data:`PANEL_PX` pixels and its axes."""
    figure = Figure(figsize=(PANEL_PX[0] / 100, PANEL_PX[1] / 100), dpi=100)
    figure.set_layout_engine("constrained")
    FigureCanvasAgg(figure)
    return figure, figure.subplots()


def _pixels(figure: Figure) -> np.ndarray:
    """A figure drawn as RGBA pixels."""
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def _class_colours(codes: np.ndarray, classes: tuple[Category, ...]) -> tuple:
    """An RGB image of categorical codes and the legend of the codes present."""
    rgb = np.empty(codes.shape + (3,))
    rgb[...] = to_rgb(OTHER[1])
    legend, known = [], np.zeros(codes.shape, bool)
    for category in classes:
        where = codes == category.code
        known |= where
        rgb[where] = to_rgb(category.colour)
        if where.any():
            legend.append(Patch(facecolor=category.colour, label=category.label))
    if not known.all():
        legend.append(Patch(facecolor=OTHER[1], label=OTHER[0]))
    return rgb, legend


def _draw_map(values, extent, field, scale, title) -> np.ndarray:
    """Draw one field: a continuous field on a fixed linear
    scale from 0 to ``scale``, a categorical one in its classes' colours and a
    photograph as it is. Cells without a value show :data:`BACKGROUND`."""
    with _RENDER:
        figure, axis = _figure()
        axis.set_title(title, fontsize=13)
        rows, columns = values.shape[:2]
        place = {
            "extent": extent or (-0.5, columns - 0.5, rows - 0.5, -0.5),
            "origin": "upper",
        }
        legend = []
        if field.photo:
            shown = values
            if values.dtype != np.uint8:
                shown = np.clip(np.nan_to_num(values.astype(np.float64)), 0, 1)
            axis.imshow(shown, interpolation="nearest", **place)
        elif field.classes:
            rgb, legend = _class_colours(values, field.classes)
            axis.imshow(rgb, interpolation="nearest", **place)
        else:
            colours = colormaps["viridis"].with_extremes(bad=BACKGROUND)
            image = axis.imshow(
                np.ma.masked_invalid(values.astype(np.float64)),
                cmap=colours,
                vmin=0.0,
                vmax=scale,
                interpolation="nearest",
                **place,
            )
            unit = f" ({field.unit})" if field.unit else ""
            label = f"{field.label}{unit} · fixed scale 0 to {scale:g}"
            figure.colorbar(image, ax=axis, fraction=0.035, label=label)
        if legend:
            figure.legend(handles=legend, loc="outside right upper", frameon=False)
        if extent:
            axis.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
        else:
            axis.set_axis_off()
        return _pixels(figure)


def _map_panel(read, source, field, scale, title) -> _Drawn:
    """One panel of a field of the frame."""
    found = read[source.kind]
    if isinstance(found, str):
        return _Drawn(None, found, scale)
    if field.key not in found:
        return _Drawn(None, f"This frame stores no {field.label.lower()}.", scale)
    values, extent, step = found[field.key]
    continuous = not field.classes and not field.photo
    if continuous and scale is None:
        scale = field.limit or _fit(values)
    if step > 1:
        title += f"\nshown at 1/{step} of the stored resolution"
    pixels = _draw_map(values, extent, field, scale, title)
    if not continuous:
        return _Drawn(pixels)
    return _Drawn(pixels, "", scale, values)


def _draw(request: _Request) -> list[_Drawn]:
    """Read and draw one frame's panels, off the window thread."""
    frame, hours = request.frames[request.index]
    sources = {s.kind: s for s, _, _ in request.panels}
    read = {kind: _read_frame(request, s, frame) for kind, s in sources.items()}
    drawn = []
    for source, field, scale in request.panels:
        # A panel that cannot be drawn says why; the others are still drawn.
        try:
            title = (
                f"{request.run} · {field.label} · {_hours(hours)}\nframe {request.index + 1}"
                f" of {len(request.frames)}, {frame.session}/{frame.name}"
            )
            drawn.append(_map_panel(read, source, field, scale, title))
        except Exception as error:
            message = f"{field.label} could not be drawn: {error}"
            drawn.append(_Drawn(None, message, scale))
    return drawn


class _Canvas(ImageCanvas):
    """The official image display, here showing drawn pixels."""

    def set_pixels(self, pixels: np.ndarray) -> None:
        """Show RGBA pixels, scaled to fit."""
        pixels = np.ascontiguousarray(pixels)
        height, width = pixels.shape[:2]
        image = QImage(
            pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888
        )
        self._original_pixmap = QPixmap.fromImage(image.copy())
        self._label.setText("")
        self._rescale()


class _Panel(QWidget):
    """One panel: its content, its colour scale and its drawn frame."""

    changed = Signal(int)
    scale_set = Signal(int, float)

    def __init__(self, number: int, parent: QWidget) -> None:
        """An empty panel; ``number`` names it in :attr:`changed` and
        :attr:`scale_set`."""
        super().__init__(parent)
        self.number = number
        self.values: np.ndarray | None = None
        """The continuous values shown, for «Fit to frame»."""
        self.pixels: np.ndarray | None = None
        """The drawn pixels shown."""
        self.message = ""
        """The message shown instead of the pixels, or the note under them."""
        self.shown: tuple[str, str] | None = None
        """The run and the content of what is shown; None while nothing is."""
        self.scale: float | None = None
        """The colour scale's maximum of what is shown."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.content = NoScrollComboBox()
        self.content.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.content.setMinimumContentsLength(12)
        self.content.currentIndexChanged.connect(self._content_changed)
        layout.addWidget(self.content)
        self.scale_row = QWidget()
        row = QHBoxLayout(self.scale_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Scale 0 to"))
        self.scale_edit = QLineEdit()
        # The C locale, as float() reads and :g writes: a decimal point on any
        # system, without group separators.
        validator = QDoubleValidator(0.0, 1e300, 12, self.scale_edit)
        locale = QLocale.c()
        locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
        validator.setLocale(locale)
        self.scale_edit.setValidator(validator)
        self.scale_edit.setMaximumWidth(100)
        self.scale_edit.setToolTip(
            "The colour scale's maximum, fixed for every frame of the run."
        )
        self.scale_edit.editingFinished.connect(self._edited)
        row.addWidget(self.scale_edit)
        self.fit_button = QPushButton("Fit to frame")
        self.fit_button.setToolTip("Fit the colour scale to the shown frame.")
        self.fit_button.clicked.connect(self.fit)
        row.addWidget(self.fit_button)
        row.addStretch(1)
        layout.addWidget(self.scale_row)
        self.canvas = _Canvas("Choose a run.")
        layout.addWidget(self.canvas, 1)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.hide()
        layout.addWidget(self.notice)
        self.clear("Choose a run.")

    def clear(self, message: str) -> None:
        """Show a message and nothing of an earlier frame; the colour scale cannot
        be changed until a frame is shown."""
        self.values = self.pixels = self.shown = None
        self.message = message
        self.canvas.set_message(message)
        self.notice.hide()
        self.scale_edit.setEnabled(False)
        self.fit_button.setEnabled(False)

    def show_drawn(self, drawn: _Drawn, shown: tuple[str, str]) -> None:
        """Show a drawn panel of a run and content, or its message."""
        self.values, self.pixels, self.message = (
            drawn.values,
            drawn.pixels,
            drawn.message,
        )
        self.shown, self.scale = shown, drawn.scale
        if drawn.pixels is None:
            self.canvas.set_message(drawn.message)
            self.notice.hide()
        else:
            self.canvas.set_pixels(drawn.pixels)
            self.notice.setText(drawn.message)
            self.notice.setVisible(bool(drawn.message))
        self.scale_edit.setText("" if drawn.scale is None else f"{drawn.scale:g}")
        self.scale_edit.setEnabled(drawn.values is not None)
        self.fit_button.setEnabled(drawn.values is not None)

    def set_continuous(self, continuous: bool) -> None:
        """Show the colour scale only for a continuous field."""
        self.scale_row.setVisible(continuous)

    def _content_changed(self) -> None:
        """Tell the viewer that this panel's content changed."""
        self.changed.emit(self.number)

    def _edited(self) -> None:
        """Keep a typed positive maximum for the run's frames; anything else gives
        the field back the scale shown."""
        try:
            value = float(self.scale_edit.text())
        except ValueError:
            value = math.nan
        if math.isfinite(value) and value > 0:
            self.scale_set.emit(self.number, value)
        else:
            self.scale_edit.setText("" if self.scale is None else f"{self.scale:g}")

    def fit(self) -> None:
        """Fit the colour scale to the largest value of the shown frame."""
        if self.values is not None:
            self.scale_set.emit(self.number, _fit(self.values))


class ResultViewer(QWidget):
    """The stored results of a project's runs, frame by frame, read only.

    Parameters
    ----------
    locations : Locations
        The project's locations of results.
    sources : sequence of Source
        Each frame's stored results the viewer shows.
    contents : sequence of str, optional
        The first content of each panel: ``<kind>/<key>`` of a field; by default
        the fields in order.
    layout : str, optional
        The first panel layout, one of :data:`LAYOUTS`.
    parent : QWidget, optional
        The parent widget.

    """

    def __init__(
        self,
        locations: Locations,
        sources: Sequence[Source],
        *,
        contents: Sequence[str] = (),
        layout: str = "one",
        parent: QWidget | None = None,
    ) -> None:
        """Build the viewer; :meth:`set_runs` gives it the runs to show."""
        super().__init__(parent)
        self.locations = locations
        self._worker = _Worker()
        self._pending: int | None = None
        self._asked: tuple = ()
        self._started = 0.0
        self._runs: dict[str, tuple[tuple[Frame, float | None], ...]] = {}
        self._run: str | None = None
        self._scales: dict[tuple[str, str], float] = {}
        self._layout = layout

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.runs_table = QTableWidget(0, 2)
        self.runs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.runs_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.verticalHeader().hide()
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.setMinimumWidth(160)
        self.runs_table.setMaximumWidth(320)
        self.runs_table.itemSelectionChanged.connect(self._run_selected)
        outer.addWidget(self.runs_table)
        right = QVBoxLayout()
        outer.addLayout(right, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Layout"))
        self.layout_choice = NoScrollComboBox()
        for name, (label, _) in LAYOUTS.items():
            self.layout_choice.addItem(label, name)
        self.layout_choice.currentIndexChanged.connect(self._layout_chosen)
        controls.addWidget(self.layout_choice)
        controls.addStretch(1)
        right.addLayout(controls)

        self.panels = [_Panel(number, self) for number in range(3)]
        for panel in self.panels:
            panel.changed.connect(self._panels_changed)
            panel.scale_set.connect(self._scale_set)
        self.grid = QGridLayout()
        right.addLayout(self.grid, 1)

        frames = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous_frame)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setToolTip("Step through the run's frames; stops at the last.")
        self.play_button.toggled.connect(self._play_toggled)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderPressed.connect(self.stop_playing)
        for widget in (self.previous_button, self.play_button, self.next_button):
            frames.addWidget(widget)
        frames.addWidget(self.slider, 1)
        right.addLayout(frames)
        self.frame_label = QLabel()
        self.frame_label.setWordWrap(True)
        right.addWidget(self.frame_label)
        status = QHBoxLayout()
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setMaximumSize(80, 12)
        self.busy_bar.hide()
        status.addWidget(self.busy_bar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        status.addWidget(self.status, 1)
        right.addLayout(status)

        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._request)
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._look)
        self._play = QTimer(self)
        self._play.setInterval(PLAY_MS)
        self._play.timeout.connect(self._play_step)

        self._fill_runs()
        self.set_results(sources, contents=contents)
        with QSignalBlocker(self.layout_choice):
            self.layout_choice.setCurrentIndex(self.layout_choice.findData(layout))
        self.set_layout(layout)

    # ------------------------------------------------------------ what is shown

    @property
    def run(self) -> str | None:
        """The run shown."""
        return self._run

    @property
    def busy(self) -> bool:
        """Whether a frame is being read or drawn."""
        return self._pending is not None

    def set_results(
        self,
        sources: Sequence[Source],
        *,
        contents: Sequence[str] = (),
        locations: Locations | None = None,
    ) -> None:
        """Show other stored results, e.g. of another calibration, or of another
        results root (``locations``, e.g. of another Setup variant); see the class.
        The colour scales and the frames kept in memory start anew.

        Raises
        ------
        ValueError
            If a content is none of the sources' fields; the viewer is then left as
            it was.

        """
        found = {
            f"{source.kind}/{field.key}": (source, field)
            for source in sources
            for field in source.fields
        }
        if any(key not in found for key in contents):
            raise ValueError("Each content is a source's field.")
        if locations is not None:
            self.locations = locations
        self.sources = tuple(sources)
        self._contents = found
        self._cache: OrderedDict = OrderedDict()
        self._scales = {}
        keys = list(contents) + [key for key in found if key not in contents]
        for number, panel in enumerate(self.panels):
            key = keys[number % len(keys)] if keys else None
            with QSignalBlocker(panel.content):
                panel.content.clear()
                for name, (source, field) in found.items():
                    panel.content.addItem(self._content_label(source, field), name)
                if key is not None:
                    panel.content.setCurrentIndex(panel.content.findData(key))
            panel.set_continuous(self._continuous(key))
            panel.clear("")
        self._request()

    def _content_label(self, source: Source, field: Field) -> str:
        """The name of a panel content in its choice."""
        return (
            field.label if len(self.sources) == 1 else f"{field.label} · {source.label}"
        )

    def _continuous(self, key: str | None) -> bool:
        """Whether a content is a continuous field, which has a colour scale."""
        source, field = self._contents.get(key, (None, None))
        return field is not None and not field.classes and not field.photo

    def set_runs(
        self, runs: Mapping[str, Sequence[tuple[Frame, float | None]]]
    ) -> None:
        """Give the runs to show: each run's frames in order, with their times [h].

        The run shown stays, at its frame, if it is still given, e.g. when a page
        lists newly stored frames; else the first run is shown.

        """
        self._runs = {
            run: tuple((Frame(*frame), hours) for frame, hours in frames)
            for run, frames in runs.items()
        }
        self._fill_runs()
        if self._run in self._runs:
            last = max(0, len(self._runs[self._run]) - 1)
            with QSignalBlocker(self.slider):
                self.slider.setRange(0, last)
                self.slider.setValue(min(self.slider.value(), last))
            self._request()
        elif self._runs:
            self.select_run(next(iter(self._runs)))
        else:
            self._run = None
            self._request()

    def _fill_runs(self) -> None:
        """List the runs with their frame counts."""
        table = self.runs_table
        with QSignalBlocker(table):
            table.setHorizontalHeaderLabels(["Run", "Frames"])
            table.setRowCount(len(self._runs))
            for row, (run, frames) in enumerate(self._runs.items()):
                texts = (run, str(len(frames)))
                for column, text in enumerate(texts):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, run)
                    table.setItem(row, column, item)
                if run == self._run:
                    table.selectRow(row)
            table.resizeColumnsToContents()

    def select_run(self, run: str) -> None:
        """Show a run from its first frame."""
        self._run = run
        with QSignalBlocker(self.runs_table):
            for row in range(self.runs_table.rowCount()):
                if self.runs_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == run:
                    self.runs_table.selectRow(row)
        with QSignalBlocker(self.slider):
            self.slider.setRange(0, max(0, len(self._runs[run]) - 1))
            self.slider.setValue(0)
        self._request()

    def _run_selected(self) -> None:
        """Show the run chosen in the list."""
        rows = self.runs_table.selectionModel().selectedRows()
        if rows:
            run = self.runs_table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
            if run != self._run:
                self.select_run(run)

    def set_layout(self, layout: str) -> None:
        """Arrange the panels in one of :data:`LAYOUTS`."""
        self._layout = layout
        for panel in self.panels:
            self.grid.removeWidget(panel)
            panel.hide()
        positions = LAYOUTS[layout][1]
        for panel, position in zip(self.panels, positions):
            self.grid.addWidget(panel, *position)
            panel.show()
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, int(len(positions) == 3))
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self._request()

    def _layout_chosen(self) -> None:
        """Arrange the panels as chosen."""
        self.set_layout(self.layout_choice.currentData())

    def _shown(self) -> list[_Panel]:
        """The panels of the layout."""
        return self.panels[: len(LAYOUTS[self._layout][1])]

    def _panels_changed(self, number: int) -> None:
        """Draw the frame again with a panel's new content."""
        panel = self.panels[number]
        panel.set_continuous(self._continuous(panel.content.currentData()))
        self._request()

    def _scale_set(self, number: int, value: float) -> None:
        """Keep a panel's colour scale for its field in every frame of the run,
        only from a frame of that run and field."""
        panel = self.panels[number]
        key = panel.content.currentData()
        if panel.shown != (self._run, key):
            return
        self._scales[(self._run, key)] = value
        self._request()

    # ------------------------------------------------------------------ frames

    def show_frame(self, index: int) -> None:
        """Show one frame of the run now."""
        with QSignalBlocker(self.slider):
            self.slider.setValue(index)
        self._request()

    def previous_frame(self) -> None:
        """Show the frame before."""
        self.show_frame(max(0, self.slider.value() - 1))

    def next_frame(self) -> None:
        """Show the frame after."""
        self.show_frame(min(self.slider.maximum(), self.slider.value() + 1))

    def _slider_moved(self, index: int) -> None:
        """Name the frame at once; read it when the slider rests."""
        self._name_frame(index)
        self._settle.start()

    def _name_frame(self, index: int) -> None:
        """The line under the frame slider: frame, photo and time."""
        frames = self._runs.get(self._run, ())
        if not frames:
            self.frame_label.setText("")
            return
        frame, hours = frames[index]
        self.frame_label.setText(
            f"Frame {index + 1} of {len(frames)} · {frame.session}/{frame.name} · "
            f"{_hours(hours)}"
        )

    def _request(self, *_: Any) -> None:
        """Read and draw the shown frame off the window thread."""
        self._settle.stop()
        frames = self._runs.get(self._run, ())
        if not frames or not self._contents:
            self._pending = None
            self._poll.stop()
            self.busy_bar.hide()
            if not self._contents:
                message = "Nothing to show."
            elif self._run is None:
                message = "Choose a run."
            else:
                message = f"{self._run} has no frames."
            for panel in self._shown():
                panel.clear(message)
            self.frame_label.setText("")
            self.status.setText("")
            return
        index = self.slider.value()
        self._name_frame(index)
        keys = [panel.content.currentData() for panel in self._shown()]
        # Nothing of another run or content stays shown while this one is read.
        for panel, key in zip(self._shown(), keys):
            if panel.shown != (self._run, key):
                panel.clear(f"Reading {self._run}…")
        panels = tuple(
            (*self._contents[key], self._scales.get((self._run, key))) for key in keys
        )
        request = _Request(
            self.locations, self._run, index, frames, panels, self._cache
        )
        self._pending = self._worker.submit(partial(_draw, request))
        self._asked = (self._run, index, len(frames), keys)
        self._started = monotonic()
        self.busy_bar.show()
        self._look()
        self._poll.start()

    def _look(self) -> None:
        """Show the elapsed time of the read under way, or its answer."""
        answer = self._worker.answer(self._pending)
        run, index, count, keys = self._asked
        if answer is None:
            seconds = monotonic() - self._started
            self.status.setText(
                f"Reading {run}, frame {index + 1} of {count}… {seconds:.0f} s"
            )
            return
        self._pending = None
        self._poll.stop()
        self.busy_bar.hide()
        _, value, error = answer
        self.status.setText("")
        if error is not None:
            for panel in self._shown():
                panel.clear(f"The frame could not be shown: {error}")
            return
        for panel, key, item in zip(self._shown(), keys, value):
            panel.show_drawn(item, (run, key))
            if item.scale is not None:
                self._scales.setdefault((run, key), item.scale)

    def stop_playing(self) -> None:
        """Stop Play; the frame shown stays."""
        self.play_button.setChecked(False)

    def _play_toggled(self, playing: bool) -> None:
        """Start or stop stepping through the frames."""
        if not playing:
            self._play.stop()
            self.play_button.setText("Play")
            return
        if self.slider.value() >= self.slider.maximum():
            self.show_frame(0)
        self.play_button.setText("Pause")
        self._play.start()

    def _play_step(self) -> None:
        """Show the next frame once the shown one is drawn; stop after the last."""
        if self.busy:
            return
        if self.slider.value() >= self.slider.maximum():
            self.stop_playing()
        else:
            self.next_frame()

    def showEvent(self, event: Any) -> None:
        """Look again for the answer of a read under way when the viewer was
        closed."""
        super().showEvent(event)
        if self._pending is not None:
            self._poll.start()

    def hideEvent(self, event: Any) -> None:
        """Stop Play while the viewer is hidden, e.g. behind another page."""
        self.stop_playing()
        super().hideEvent(event)

    def closeEvent(self, event: Any) -> None:
        """Stop the timers; a read under way ends in its own thread, and its
        frame is shown when the viewer is shown again."""
        for timer in (self._settle, self._poll, self._play):
            timer.stop()
        super().closeEvent(event)
