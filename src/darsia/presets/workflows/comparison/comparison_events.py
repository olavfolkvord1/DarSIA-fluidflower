"""Events: the first frame where a question about an analysis domain's CO2 holds.

A question (:class:`EventQuestion`) holds one or more conditions
(:class:`EventCondition`), each on one component of the CO2 (``total``,
``aqueous`` or ``gas``) in one analysis domain:

- ``detected``: its mass is above the noise limit;
- ``reaches``: its mass is at least a threshold;
- ``falls_below``: its mass is below a threshold it reached at an earlier frame;
- ``maximum``: the frame of its largest mass;
- ``gone``: it had been detected, and is detected neither at this frame nor later.

A threshold, and the noise limit, is in grams or in percent of the CO2 injected
up to the frame; a percentage cannot be reached before anything is injected. The
conditions combine with ``all`` (and) or ``any`` (or), and a question can be
asked to hold for several frames in a row. Thresholds are formed in double
precision, as configured, and compared with the masses as given.

A frame is unknown where its mass, or the injected mass a percentage needs, is
not known, e.g. in a domain that is not complete when the complete masses are
read (:meth:`RunSeries.from_domain_mass`). The states follow Kleene's logic, true,
false or unknown held as 1, 0 and 1/2, so that ``and`` is the minimum, ``or`` the
maximum and ``not`` one minus the state.

The answer of a run (:func:`run_answer`) is its first frame where the question
holds: its time in hours since the injection start, its hours after the start of
the run's last injection, and its resolution, the hours since the frame before,
at which the event had not yet happened. Per injection (:func:`injection_answers`)
it is the first frame in that injection's time, from its start to the next
injection's start (the last one's to the run's end), counted from its start; the
question is evaluated over the whole run, so one that already holds when the
injection starts is answered at its first frame. :func:`event_table` gives both
for every question and run, as the rows :func:`write_events` writes.

The answers read the masses per domain of a run's frames (:class:`RunSeries`); no
map is read. The official comparison's events (:func:`comparison_events`) are
questions of one condition: the mass of a region reaches a fraction of the run's
largest injected mass.

"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darsia.presets.workflows.analysis.domain_mass import RunSeries
from darsia.presets.workflows.analysis.domains import COMPONENTS, write_rows
from darsia.presets.workflows.config.multi_fluidflower_config import (
    EventConfig,
    MultiFluidFlowerConfig,
)
from darsia.presets.workflows.config.sections import required_sections

logger = logging.getLogger(__name__)

EVENTS = ("detected", "reaches", "falls_below", "maximum", "gone")
"""What a condition can ask of a component's mass."""
UNITS = ("g", "percent_injected")
"""Units of a threshold: grams, or percent of the CO2 injected up to the frame."""
COLUMNS = (
    "question",
    "mode",
    "run",
    "injection",
    "case_id",
    "elapsed_h",
    "after_injection_h",
    "resolution_h",
    "note",
)
"""Columns of the event table: a row per question, run and, per injection, injection."""
_TRUE, _UNKNOWN, _FALSE = 1.0, 0.5, 0.0
_OFFICIAL_MODES = {field: component for component, field in COMPONENTS.items()}
"""The official events' modes, the fields of the mass maps, and the component each
reads."""


@dataclass(frozen=True)
class EventCondition:
    """One condition: what happens to a component of the CO2 in a domain."""

    component: Literal["total", "aqueous", "gas"]
    """The total CO2, the dissolved CO2 or the free gas."""
    event: Literal["detected", "reaches", "falls_below", "maximum", "gone"]
    """What happens to its mass."""
    domain_id: str
    """The analysis domain, by its identity."""
    threshold: float | None = None
    """``reaches`` and ``falls_below``: the threshold in ``unit``, 0 or more."""
    unit: Literal["g", "percent_injected"] | None = None
    """``reaches`` and ``falls_below``: the unit of the threshold."""

    def __post_init__(self) -> None:
        """Refuse a condition that would be read as another one."""
        if self.component not in COMPONENTS or self.event not in EVENTS:
            raise ValueError(f"Unknown component or event: {self}")
        needs = self.event in ("reaches", "falls_below")
        given = (self.threshold, self.unit) != (None, None)
        if given and not needs:
            raise ValueError(f"{self.event} takes no threshold")
        if needs and not (self.unit in UNITS and self.threshold is not None):
            raise ValueError(f"{self.event} needs a threshold and its unit")
        if needs and not self.threshold >= 0:
            raise ValueError(f"{self.event} needs a threshold of 0 or more")


@dataclass(frozen=True)
class EventQuestion:
    """A named question: its conditions, how they combine and the noise filter.

    Conditions given as mappings, as read from a TOML file, become conditions.

    """

    name: str
    """Name of the question, unique in an analysis."""
    conditions: tuple[EventCondition, ...]
    """The conditions, at least one."""
    combine: Literal["all", "any"] = "all"
    """``all`` conditions hold (and), or ``any`` holds (or)."""
    noise_min: float = 0.0
    """The largest mass that does not count as detected, in ``noise_unit``."""
    noise_unit: Literal["g", "percent_injected"] = "g"
    """Unit of ``noise_min``."""
    hold_frames: int = 1
    """Frames in a row the question must hold; 1 takes its first frame."""

    def __post_init__(self) -> None:
        """Turn conditions given as mappings into conditions; check the rest."""
        conditions = tuple(
            EventCondition(**c) if isinstance(c, Mapping) else c
            for c in self.conditions
        )
        object.__setattr__(self, "conditions", conditions)
        if not self.name.strip() or not conditions:
            raise ValueError("A question needs its name and a condition")
        if self.combine not in ("all", "any") or self.noise_unit not in UNITS:
            raise ValueError(f"Unknown combination or noise unit: {self.name}")
        if not self.noise_min >= 0:
            raise ValueError(f"The noise limit of {self.name} must be 0 or more")
        if type(self.hold_frames) is not int or self.hold_frames < 1:
            raise ValueError(f"{self.name} must hold for a whole number of frames")


@dataclass(frozen=True)
class EventAnswer:
    """When a question first held in a run, or in one of its injections."""

    question: str
    """Name of the question."""
    run: str
    """The run."""
    injection: int | None = None
    """The injection ``after_injection_h`` counts from: the run's last one for an
    answer per run, its own for one per injection; None without injections."""
    case_id: str | None = None
    """The frame of the answer; None where the question never held."""
    elapsed_h: float | None = None
    """Its hours since the injection start."""
    after_injection_h: float | None = None
    """Its hours after the injection's start, negative before it."""
    resolution_h: float | None = None
    """Hours since the frame before, at which the question did not hold; None at
    the run's first frame and where the question held or was unknown there."""
    note: str = ""
    """Unknown frames before the answer, or why there is none."""


def _limit(value: float, unit: str, injected: np.ndarray) -> np.ndarray:
    """A threshold in grams per frame, in double precision as configured: NaN where
    a percentage lacks the injected mass, infinite before anything is injected."""
    injected = np.asarray(injected, np.float64)
    if unit == "g":
        return np.full(injected.shape, float(value))
    share = float(value) / 100 * injected
    return np.where(injected > 0, share, np.where(np.isnan(injected), np.nan, np.inf))


def _compare(compare, amounts: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Three-valued states of a comparison, unknown where a side is not known."""
    states = compare(amounts, limits).astype(np.float64)
    states[np.isnan(amounts) | np.isnan(limits)] = _UNKNOWN
    return states


def _earlier(states: np.ndarray) -> np.ndarray:
    """Whether the states held at an earlier frame: ``or`` over the frames before."""
    return np.r_[_FALSE, np.maximum.accumulate(states)][: states.size]


def condition_states(
    condition: EventCondition,
    series: RunSeries,
    noise_min: float = 0.0,
    noise_unit: str = "g",
) -> np.ndarray:
    """Whether a condition holds at each frame of a run.

    Parameters
    ----------
    condition : EventCondition
        The condition.
    series : RunSeries
        The run's masses per domain.
    noise_min : float
        The largest mass that does not count as detected.
    noise_unit : {"g", "percent_injected"}
        Its unit.

    Returns
    -------
    np.ndarray
        Per frame 1 (true), 0 (false) or 1/2 (unknown).

    """
    if condition.domain_id not in series.masses:
        raise ValueError(f"Run {series.run} has no masses of {condition.domain_id}")
    amounts = series.masses[condition.domain_id][condition.component]
    injected = series.injected_g
    detected = _compare(np.greater, amounts, _limit(noise_min, noise_unit, injected))
    if condition.event == "detected":
        return detected
    if condition.event in ("reaches", "falls_below"):
        limits = _limit(condition.threshold, condition.unit, injected)
        reached = _compare(np.greater_equal, amounts, limits)
        if condition.event == "reaches":
            return reached
        return np.minimum(_earlier(reached), 1 - reached)
    if condition.event == "maximum":
        known = ~np.isnan(amounts)
        states = np.full(amounts.size, _UNKNOWN)
        if known.any():
            states[known] = _FALSE
            states[np.nanargmax(amounts)] = _TRUE
        return states
    later = np.maximum.accumulate(detected[::-1])[::-1]
    return np.minimum(_earlier(detected), 1 - later)


def question_states(question: EventQuestion, series: RunSeries) -> np.ndarray:
    """Whether a question holds at each frame of a run, its noise filter applied.

    Returns
    -------
    np.ndarray
        Per frame 1 (true), 0 (false) or 1/2 (unknown); a frame where the question
        must still hold at frames after the run's end is unknown.

    """
    states = [
        condition_states(c, series, question.noise_min, question.noise_unit)
        for c in question.conditions
    ]
    combine = np.min if question.combine == "all" else np.max
    combined = combine(states, axis=0)
    hold = question.hold_frames
    if hold == 1:
        return combined
    padded = np.r_[combined, np.full(hold - 1, _UNKNOWN)]
    return sliding_window_view(padded, hold).min(axis=1)


def _first(question, series, states, frames, injection, never) -> EventAnswer:
    """The first of ``frames`` where ``states`` hold, counted from ``injection``."""
    number, start_h = (None, None) if injection is None else injection
    found = np.flatnonzero(states[frames] == _TRUE)
    if found.size == 0:
        unknown = np.any(states[frames] == _UNKNOWN)
        note = "Not found; some frames are unknown" if unknown else never
        return EventAnswer(question.name, series.run, number, note=note)
    frame = frames[found[0]]
    unknown = int(np.count_nonzero(states[frames[: found[0]]] == _UNKNOWN))
    hours = float(series.elapsed_h[frame])
    return EventAnswer(
        question=question.name,
        run=series.run,
        injection=number,
        case_id=series.case_ids[frame],
        elapsed_h=hours,
        after_injection_h=None if start_h is None else hours - start_h,
        resolution_h=(
            hours - float(series.elapsed_h[frame - 1])
            if frame and states[frame - 1] == _FALSE
            else None
        ),
        note=f"{unknown} earlier frame(s) unknown" if unknown else "",
    )


def run_answer(
    question: EventQuestion,
    series: RunSeries,
    injections: pd.DataFrame | None = None,
) -> EventAnswer:
    """The first frame of a run where a question holds.

    Parameters
    ----------
    question : EventQuestion
        The question.
    series : RunSeries
        The run's masses per domain.
    injections : pd.DataFrame, optional
        The run's injections (:meth:`darsia.InjectionProtocol.injections`); with
        them the answer counts its hours after the start of the last one.

    """
    last = None
    if injections is not None and len(injections):
        row = injections.iloc[-1]
        last = (int(row["number"]), float(row["start_h"]))
    frames = np.arange(len(series.case_ids))
    states = question_states(question, series)
    return _first(question, series, states, frames, last, "Never")


def injection_answers(
    question: EventQuestion, series: RunSeries, injections: pd.DataFrame
) -> list[EventAnswer]:
    """Per injection, the first frame in its time where a question holds.

    Parameters
    ----------
    question : EventQuestion
        The question, evaluated over the whole run.
    series : RunSeries
        The run's masses per domain.
    injections : pd.DataFrame
        The run's injections (:meth:`darsia.InjectionProtocol.injections`); each
        one's time runs from its start to the next one's start.

    """
    states = question_states(question, series)
    hours = series.elapsed_h
    starts = [float(start) for start in injections["start_h"]]
    answers = []
    for number, start, end in zip(injections["number"], starts, starts[1:] + [np.inf]):
        frames = np.flatnonzero((hours >= start) & (hours < end))
        if frames.size:
            never, injection = "Never in this injection's time", (int(number), start)
            answers.append(_first(question, series, states, frames, injection, never))
        else:
            note = "No frame in this injection's time"
            answers.append(
                EventAnswer(question.name, series.run, int(number), note=note)
            )
    return answers


def event_table(
    questions: Sequence[EventQuestion],
    series: Sequence[RunSeries],
    injections: Mapping[str, pd.DataFrame],
) -> list[dict]:
    """Every question's answers per run and per injection, as rows of the table.

    Parameters
    ----------
    questions : sequence of EventQuestion
        The questions.
    series : sequence of RunSeries
        The runs' masses per domain.
    injections : mapping
        Each run's injections (:meth:`darsia.InjectionProtocol.injections`).

    Returns
    -------
    list of dict
        Per question, the answers of every run (``mode`` "run"), then those of
        every run's injections (``mode`` "injection"), in the columns
        :data:`COLUMNS`.

    """
    rows = []
    for question in questions:
        for run in series:
            answer = run_answer(question, run, injections[run.run])
            rows.append(dict(asdict(answer), mode="run"))
        for run in series:
            for answer in injection_answers(question, run, injections[run.run]):
                rows.append(dict(asdict(answer), mode="injection"))
    return rows


def write_events(path: Path, rows: Iterable[dict]) -> None:
    """Write the event table; a value that is None is an empty cell.

    Parameters
    ----------
    path : Path
        The CSV file.
    rows : iterable of dict
        Rows of :func:`event_table`.

    """
    write_rows(path, rows, COLUMNS)


def read_events(path: Path) -> list[dict]:
    """Read the event table written by :func:`write_events`.

    Returns
    -------
    list of dict
        The rows as :func:`event_table` gives them: None for an empty cell, the
        injection a number and the hours numbers.

    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    kinds = dict(
        injection=int,
        case_id=str,
        elapsed_h=float,
        after_injection_h=float,
        resolution_h=float,
    )
    for row in rows:
        row.update(
            {key: kind(row[key]) if row[key] else None for key, kind in kinds.items()}
        )
    return rows


def integrated_events(
    table: pd.DataFrame, events: Iterable[EventConfig]
) -> dict[str, float | None]:
    """The official events of one run's table of integrated masses.

    Parameters
    ----------
    table : pd.DataFrame
        The run's ``integrated_mass.csv``: a ``time`` column, the detected masses
        of each region of interest (``<roi>_detected_mass``, ``_mass_g`` and
        ``_mass_aq``) and an ``exact_mass`` column of the injected mass; for
        events of a single-phase fluid's volume (a tracer, mode ``volume``), the
        detected volumes (``<roi>_detected_volume_m3``) and the injected volume
        (``exact_volume_total_m3``).
    events : iterable of EventConfig
        The events of the comparison's configuration.

    Returns
    -------
    dict
        Per event its first time where the detected mass of its region reaches its
        relative threshold of the largest injected mass (of 1 g without an exact
        mass column), or its detected volume that of the largest injected volume;
        None where it never does.

    """
    table = table.sort_values("time", kind="stable")
    exact = [column for column in table.columns if "exact_mass" in column]
    total = np.max(table[exact[0]]) if exact else 1.0
    found = {}
    for event in events:
        largest = total
        if event.mode == "volume":
            # Volume of a single-phase fluid (tracer), in m3, read by the same rule
            assert "exact_volume_total_m3" in table.columns, (
                "Volume events need the mass results of a single-phase fluid: "
                "column 'exact_volume_total_m3' not found."
            )
            largest = np.max(table["exact_volume_total_m3"])
            key = f"{event.roi_name}_detected_volume_m3"
            component = "total"
        elif event.mode in _OFFICIAL_MODES:
            key = f"{event.roi_name}_detected_{event.mode}"
            component = _OFFICIAL_MODES[event.mode]
        else:
            raise NotImplementedError(f"Event type {event.mode} not implemented.")
        assert key in table.columns, f"Key {key} not found in mass results."
        series = RunSeries(
            run="",
            case_ids=tuple(map(str, table.index)),
            elapsed_h=table["time"].to_numpy(np.float64),
            injected_g=np.full(len(table), np.nan),
            masses={event.roi_name: {component: table[key].to_numpy(np.float64)}},
        )
        threshold = event.relative_threshold * largest
        if np.isnan(threshold):  # no injected mass: the event's time is not known
            found[event.event_id] = None
            continue
        condition = EventCondition(component, "reaches", event.roi_name, threshold, "g")
        question = EventQuestion(event.event_id, (condition,))
        found[event.event_id] = run_answer(question, series).elapsed_h
    return found


@required_sections("events")
def comparison_events(
    path: Path,
):
    """Write the official comparison's events of each run (:func:`integrated_events`).

    Parameters
    ----------
    path : Path
        The comparison's configuration; its results are each run's
        ``sparse_data/integrated_mass.csv``, and the events are written to the
        events path, a row per run.

    """
    # This script does not depend on raw data, only results
    config = MultiFluidFlowerConfig(path, require_results=True)

    # Safety checks
    assert config.events is not None
    assert config.runs is not None

    # Prepare data frame to store results
    events = pd.DataFrame(columns=["run"] + list(config.events.events.keys()))

    for run, run_config in config.runs.config.items():
        assert run_config.data is not None
        mass_path = run_config.data.results / "sparse_data" / "integrated_mass.csv"
        times = integrated_events(pd.read_csv(mass_path), config.events.events.values())
        for event_id, time in times.items():
            events.at[run, event_id] = np.nan if time is None else time
            logger.info(f"Event {event_id} for run {run} occurred at time {time}.")

    # Store to file.
    events.to_csv(config.events.path)
