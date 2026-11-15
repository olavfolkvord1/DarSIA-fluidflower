"""The figure style shared by every figure and film, and the pixel canvas they draw on.

:class:`FigureStyle` holds the colours, the resolution of figures written as files
and the rule for a logarithmic time axis, as one configuration table. Films are
drawn in pixels: :func:`canvas` gives a Matplotlib figure of a size in pixels, and
:func:`pixels` draws it and returns its RGBA pixels.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

PIXEL_DPI = 100
"""Resolution of the pixel canvas: one inch is this many pixels."""


@dataclass(frozen=True)
class Curve:
    """One curve of a mass graph."""

    column: str
    """The column of the mass records it draws [g]."""
    name: str
    """Its name in the graph's key."""
    colour: str
    """Its colour (a Matplotlib colour)."""
    line: str
    """Its line style (a Matplotlib format, e.g. "-" or "--")."""


@dataclass(frozen=True)
class FigureStyle:
    """Colours, resolution and axis rule of every figure and film; one table."""

    dpi: float
    """Resolution of figures written as files [dots per inch]."""
    log_span: float
    """A time axis spanning more than this many times its first positive time is
    drawn logarithmic."""
    surface: str
    """Background of figures written as files (graph tiles are white)."""
    ink: str
    """Text and labels."""
    secondary: str
    """Ticks and secondary lines."""
    muted: str
    """Lines of single runs."""
    grid: str
    """Grid lines."""
    axis: str
    """Axis lines."""
    washes: tuple[str, str]
    """The two tones that shade neighbouring injections."""
    series: str
    """The colour of an ensemble."""
    frame: str
    """Background of film frames."""
    empty: str
    """A film cell without a tile."""
    no_value: str
    """Map cells without a value."""
    colour_map: str
    """Colour map of mass maps (a Matplotlib colour map)."""
    marker: str
    """The line marking a graph's frame time."""
    curves: tuple[Curve, ...]
    """The curves of a mass graph, in drawing order."""


def bgr(colour: str) -> tuple[int, int, int]:
    """A Matplotlib colour as 8-bit (blue, green, red)."""
    red, green, blue = (int(round(value * 255)) for value in to_rgb(colour))
    return blue, green, red


def canvas(size_px, *, constrained: bool = True) -> Figure:
    """A figure of ``size_px`` (width, height) pixels on its own Agg canvas.

    With ``constrained``, Matplotlib's constrained layout places its axes;
    without, the caller places them.

    """
    width, height = size_px
    figure = Figure(
        figsize=(width / PIXEL_DPI, height / PIXEL_DPI),
        dpi=PIXEL_DPI,
        layout="constrained",
    )
    if not constrained:
        figure.set_layout_engine("none")
    FigureCanvasAgg(figure)
    return figure


def pixels(figure: Figure) -> np.ndarray:
    """Draw ``figure`` and return its RGBA pixels; its drawing buffers are released."""
    try:
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba()).copy()
    finally:
        figure.clear()
        figure.canvas.renderer = None
