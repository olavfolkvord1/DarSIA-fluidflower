"""Figures of the ensemble statistics, drawn on Matplotlib's Agg canvas.

From the tables of :func:`ensemble_tables` (:func:`ensemble_figures`):

- per domain and component, every run's mass at the production target times, the
  median and interquartile range over the runs, and their coefficient of variation
  below, with the injections shaded (:func:`mass_figure`);
- per component, each run's fraction of the injected reference at each mark and
  their median (:func:`fraction_figure`);
- per domain, each run's centre of mass over time and the ensemble's mean centre,
  in mm of the grid (:func:`trajectory_figure`).

Each figure is written as ``<name>.png`` with its title and ``<name>_notitle.png``
without (:func:`save_figure`). Text is in English, one measure per axis. Runs are
drawn thin in a quiet grey and the ensemble in the first series colour.

"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

from darsia.presets.workflows.analysis.domains import COMPONENTS
from darsia.presets.workflows.comparison.ensemble import (
    PEAK,
    EnsembleRun,
    EnsembleSettings,
)

DPI = 150
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASHES = "#e1e0d9", "#c3c2b7", ("#ebeae4", "#dcdcd4")
"""Grid and axis lines, and the two tones that shade neighbouring injections."""
SERIES = "#2a78d6"
"""The ensemble's colour."""
LOG_SPAN = 50.0
"""A time axis spanning more than this many times its first positive time is
drawn logarithmic."""
COMPONENT_LABELS = dict(total="Total CO2", aqueous="Dissolved CO2", gas="Free gas")
"""Names of the components in the figures."""


def _figure(size_in: tuple[float, float]) -> Figure:
    """A figure on its own Agg canvas, without pyplot."""
    figure = Figure(figsize=size_in, dpi=DPI, layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure


def _style(axis) -> None:
    """Recessive axes: hairline grid, muted spines and ticks, text in ink."""
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    axis.set_axisbelow(True)
    for side, spine in axis.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
    axis.tick_params(colors=SECONDARY, labelcolor=INK, labelsize=8)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def _numbers(values) -> np.ndarray:
    """Values as a float array, None as NaN."""
    return np.asarray([np.nan if v is None else float(v) for v in values], np.float64)


def _stem(text: str) -> str:
    """A file name part for a domain's name: lower case, other characters as "-"."""
    return re.sub(r"[^0-9A-Za-z]+", "-", str(text)).strip("-").lower() or "domain"


def save_figure(figure: Figure, title: str, folder: Path, stem: str) -> list[str]:
    """Write a figure as ``<stem>_notitle.png`` and, with its title, ``<stem>.png``.

    Returns
    -------
    list of str
        The two file names.

    """
    folder = Path(folder)
    options = dict(dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    figure.savefig(folder / f"{stem}_notitle.png", **options)
    figure.suptitle(title, fontsize=10)
    figure.savefig(folder / f"{stem}.png", **options)
    return [f"{stem}.png", f"{stem}_notitle.png"]


def time_scale(times: np.ndarray) -> str:
    """ "log" where the positive times span more than :data:`LOG_SPAN` times their
    smallest, else "linear"."""
    values = times[np.isfinite(times) & (times > 0)]
    return "log" if values.size and values.max() / values.min() > LOG_SPAN else "linear"


def _plain_log_ticks(axis, times: np.ndarray) -> None:
    """Hour ticks 0.1, 0.3, 1, 3, 10, ... on a logarithmic time axis."""
    values = times[np.isfinite(times) & (times > 0)]
    if values.size == 0:
        return
    low = math.floor(math.log10(values.min()))
    high = math.ceil(math.log10(values.max()))
    ticks = [step * 10.0**power for power in range(low, high + 1) for step in (1, 3)]
    shown = [t for t in ticks if values.min() / 1.5 <= t <= values.max() * 1.5]
    axis.xaxis.set_major_locator(FixedLocator(shown))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_locator(NullLocator())


def injection_spans(runs: Sequence[EnsembleRun]) -> list[tuple[str, float, float]]:
    """Each injection's median start and end [h] over the runs, for shading."""
    numbers: dict[int, list[tuple[float, float]]] = {}
    for run in runs:
        for row in run.protocol.injections().itertuples():
            numbers.setdefault(row.number, []).append((row.start_h, row.end_h))
    return [
        (
            f"Injection {number}",
            float(np.median([start for start, _ in rows])),
            float(np.median([end for _, end in rows])),
        )
        for number, rows in sorted(numbers.items())
    ]


def mass_figure(curves, band, spans, domain_label: str, component_label: str):
    """Every run's mass thin, the median bold in its interquartile band, and the
    coefficient of variation in a panel below.

    Parameters
    ----------
    curves : mapping
        Per run its (time, mass) points at the production target times.
    band : sequence of dict
        The ``mass_by_time`` rows of the domain and component, of the mass.
    spans : sequence of tuple
        The injections as (label, start, end) hours, shaded in both panels.
    domain_label, component_label : str
        Names of the domain and component.

    Returns
    -------
    tuple
        The figure, without title, and its title. A logarithmic time axis
        (:func:`time_scale`) leaves out the times at or before zero.

    """
    figure = _figure((9.0, 5.8))
    grid = figure.add_gridspec(2, 1, height_ratios=(3.3, 1.0))
    top = figure.add_subplot(grid[0])
    bottom = figure.add_subplot(grid[1], sharex=top)
    times = _numbers(row["time_h"] for row in band)
    scale = time_scale(times)
    shown = times > 0 if scale == "log" else np.isfinite(times)
    for axis in (top, bottom):
        _style(axis)
        axis.set_xscale(scale)
        for number, (_, start, end) in enumerate(spans):
            if scale == "log" and shown.any():
                start = max(start, float(np.nanmin(times[shown])))
            if end > start:
                axis.axvspan(
                    start, end, color=WASHES[number % 2], linewidth=0, zorder=0
                )
    place = blended_transform_factory(top.transData, top.transAxes)
    for label, start, end in spans:
        log = scale == "log" and start > 0
        middle = math.sqrt(max(start, 1e-9) * end) if log else (start + end) / 2
        name = label.replace("Injection ", "I")
        text = dict(ha="center", va="top", fontsize=7, color=SECONDARY)
        top.text(middle, 0.99, name, transform=place, **text)
    for points in curves.values():
        x, y = _numbers(p[0] for p in points), _numbers(p[1] for p in points)
        keep = (x > 0) if scale == "log" else np.isfinite(x)
        top.plot(x[keep], y[keep], color=MUTED, linewidth=0.7, alpha=0.55, zorder=2)
    median, q25, q75 = (
        _numbers(row[k] for row in band) for k in ("median", "q25", "q75")
    )
    top.fill_between(
        times[shown],
        q25[shown],
        q75[shown],
        color=SERIES,
        alpha=0.16,
        linewidth=0,
        zorder=3,
    )
    top.plot(
        times[shown],
        median[shown],
        color=SERIES,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=4,
    )
    top.set_ylabel(f"{component_label} mass (g)")
    top.tick_params(labelbottom=False)
    cv = _numbers(None if row["cv"] is None else 100.0 * row["cv"] for row in band)
    bottom.plot(times[shown], cv[shown], color=SECONDARY, linewidth=1.5, zorder=3)
    bottom.set_ylim(bottom=0)
    bottom.set_ylabel("CV (%)")
    bottom.set_xlabel(
        "Hours since injection start" + (" (log scale)" if scale == "log" else "")
    )
    if scale == "log":
        _plain_log_ticks(bottom, times)
    handles = [
        Line2D(
            [], [], color=MUTED, linewidth=0.9, label=f"Each run (n = {len(curves)})"
        ),
        Line2D([], [], color=SERIES, linewidth=2.0, label="Median"),
        Patch(facecolor=SERIES, alpha=0.16, label="Interquartile range"),
    ]
    if spans:
        names = ", ".join(label.replace("Injection ", "I") for label, _, _ in spans)
        label = f"Injections {names} (median start to end)"
        handles.append(Patch(facecolor=WASHES[0], label=label))
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    return figure, f"{domain_label} · {component_label}: mass across runs"


def fraction_figure(rows, component_label: str):
    """Each run's fraction of the injected reference at each mark, and the median.

    Parameters
    ----------
    rows : sequence of dict
        The ``mass_at_marks`` rows of one component; each domain and mark is one
        column, in the order of the rows.
    component_label : str
        Name of the component.

    Returns
    -------
    tuple
        The figure, without title, and its title.

    """
    keys = ("domain", "domain_label", "mark", "mark_label")
    columns = list(dict.fromkeys(tuple(row[k] for k in keys) for row in rows))
    values = {column: [] for column in columns}
    for row in rows:
        if row["fraction_of_injected"] is not None:
            column = tuple(row[k] for k in keys)
            values[column].append(100.0 * row["fraction_of_injected"])
    figure = _figure((max(5.5, 1.05 * len(columns) + 2.0), 4.6))
    axis = figure.subplots()
    _style(axis)
    axis.grid(False, axis="x")
    axis.axhline(100.0, color=MUTED, linewidth=0.9, zorder=1)
    for position, column in enumerate(columns):
        points = np.asarray(values[column], np.float64)
        if points.size == 0:
            continue
        offsets = ((np.arange(points.size) * 0.618034) % 1.0 - 0.5) * 0.42
        axis.scatter(
            position + offsets,
            points,
            s=22,
            color=SERIES,
            alpha=0.8,
            edgecolors=SURFACE,
            linewidths=0.8,
            zorder=3,
        )
        axis.plot(
            [position - 0.28, position + 0.28],
            [np.median(points)] * 2,
            color=INK,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=4,
        )
    labels = [
        textwrap.fill(label, 18) + "\n" + textwrap.fill(mark, 18)
        for _, label, _, mark in columns
    ]
    axis.set_xticks(range(len(columns)), labels, fontsize=8)
    axis.set_xlim(-0.6, len(columns) - 0.4)
    axis.set_ylabel("Detected mass / injected (%)")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=SERIES, label="Each run"),
        Line2D([], [], color=INK, linewidth=2.0, label="Median"),
        Line2D([], [], color=MUTED, linewidth=0.9, label="100 % of injected"),
    ]
    figure.legend(
        handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8
    )
    title = f"{component_label}: detected fraction of the injected mass at the marks"
    return figure, title


def trajectory_figure(paths, mean, domain_label: str, component_label: str):
    """The centre of mass over time: each run's path thin, the mean path bold.

    Parameters
    ----------
    paths : mapping
        Per run its (hours, x, y) points [h, mm].
    mean : sequence of tuple
        The ensemble's mean centre (hours, x, y) per time.
    domain_label, component_label : str
        Names of the domain and component.

    Returns
    -------
    tuple
        The figure, without title, and its title.

    """
    figure = _figure((7.2, 5.6))
    axis = figure.subplots()
    _style(axis)
    for points in paths.values():
        points = sorted((p for p in points if p[0] is not None), key=lambda p: p[0])
        x, y = _numbers(p[1] for p in points), _numbers(p[2] for p in points)
        axis.plot(x, y, color=MUTED, linewidth=0.8, alpha=0.6, zorder=2)
        if x.size:
            axis.scatter(x[-1:], y[-1:], s=10, color=MUTED, zorder=2)
    mean = sorted(mean, key=lambda p: p[0])
    x, y = _numbers(p[1] for p in mean), _numbers(p[2] for p in mean)
    axis.plot(x, y, color=SERIES, linewidth=2.2, solid_joinstyle="round", zorder=4)
    if x.size:
        first = dict(s=64, facecolors=SURFACE, edgecolors=SERIES, linewidths=2.0)
        axis.scatter(x[:1], y[:1], zorder=5, **first)
        last = dict(s=64, color=SERIES, edgecolors=SURFACE, linewidths=2.0)
        axis.scatter(x[-1:], y[-1:], zorder=5, **last)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    handles = [
        Line2D(
            [], [], color=MUTED, linewidth=0.9, label=f"Each run (n = {len(paths)})"
        ),
        Line2D([], [], color=SERIES, linewidth=2.2, label="Ensemble mean"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            color=SERIES,
            label="First time",
        ),
        Line2D([], [], marker="o", linestyle="", color=SERIES, label="Last time"),
    ]
    figure.legend(
        handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8
    )
    return figure, f"{domain_label} · {component_label}: centre of mass over time"


def _curves(rows, domain: str, component: str) -> dict[str, list[tuple]]:
    """Per run, its mass at each production target time, from its first frame
    recorded for that target."""
    curves: dict[str, dict[float, float | None]] = {}
    for row in rows:
        if row["domain"] == domain and row["component"] == component:
            points = curves.setdefault(row["run"], {})
            if row["target_h"] is not None:
                points.setdefault(row["target_h"], row["mass_g"])
    return {run: sorted(points.items()) for run, points in curves.items()}


def ensemble_figures(
    folder: Path,
    tables: Mapping[str, list[dict]],
    runs: Sequence[EnsembleRun],
    settings: EnsembleSettings,
) -> list[str]:
    """Write the ensemble's figures of its tables.

    Parameters
    ----------
    folder : Path
        The folder of the figures.
    tables : mapping
        The tables of :func:`ensemble_tables`.
    runs : sequence of EnsembleRun
        The runs, whose injections the mass figures shade.
    settings : EnsembleSettings
        The component of the trajectory figures.

    Returns
    -------
    list of str
        The file names: ``mass_<domain>_<component>``, ``fractions_<component>``
        and ``trajectory_<domain>_<component>``, each with and without title.

    Raises
    ------
    ValueError
        Where two domains' names give one file name, so that a figure would
        overwrite another.

    """
    names = dict.fromkeys(
        (row["domain"], row["domain_label"])
        for table in ("mass_long", "centres_long")
        for row in tables[table]
    )
    stems = [_stem(name) for _, name in names]
    alike = sorted({name for (_, name), s in zip(names, stems) if stems.count(s) > 1})
    if alike:
        raise ValueError("Domains named alike would share figures: " + ", ".join(alike))
    spans = injection_spans(runs)
    files = []
    long = tables["mass_long"]
    for domain, name in dict.fromkeys((r["domain"], r["domain_label"]) for r in long):
        for component in COMPONENTS:
            curves = _curves(long, domain, component)
            if all(value is None for p in curves.values() for _, value in p):
                continue
            band = [
                row
                for row in tables["mass_by_time"]
                if (row["domain"], row["component"]) == (domain, component)
                and row["quantity"] == "mass_g"
            ]
            label = COMPONENT_LABELS[component]
            figure, title = mass_figure(curves, band, spans, name, label)
            files += save_figure(
                figure, title, folder, f"mass_{_stem(name)}_{component}"
            )
    for component in COMPONENTS:
        rows = [
            row
            for row in tables["mass_at_marks"]
            if row["component"] == component and row["mark"] != PEAK
        ]
        if any(row["fraction_of_injected"] is not None for row in rows):
            figure, title = fraction_figure(rows, COMPONENT_LABELS[component])
            files += save_figure(figure, title, folder, f"fractions_{component}")
    component = settings.centre_component
    centres = [row for row in tables["centres_long"] if row["component"] == component]
    for domain, name in dict.fromkeys(
        (r["domain"], r["domain_label"]) for r in centres
    ):
        paths: dict[str, list[tuple]] = {}
        for row in centres:
            if row["domain"] == domain:
                point = (row["elapsed_h"], row["centre_x_mm"], row["centre_y_mm"])
                paths.setdefault(row["run"], []).append(point)
        mean = [
            (row["time_h"], row["mean_x_mm"], row["mean_y_mm"])
            for row in tables["centres_by_time"]
            if (row["domain"], row["component"]) == (domain, component)
        ]
        label = COMPONENT_LABELS[component]
        figure, title = trajectory_figure(paths, mean, name, label)
        files += save_figure(
            figure, title, folder, f"trajectory_{_stem(name)}_{component}"
        )
    return files
