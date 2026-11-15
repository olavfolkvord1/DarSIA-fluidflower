"""The Comparison's steps: events, the ensemble's tables and the distances.

- ``events`` answers each configured question about the selected runs' masses per
  domain, per run and per injection (:func:`event_table`).
- ``ensemble`` summarises the selected runs' masses and centres per domain
  (:func:`ensemble_tables`) and draws its figures in the series' figure style
  (:func:`ensemble_figures`).
- ``distances`` (R9) gives, at each comparison time, the Wasserstein distance of
  every pair of selected runs in every domain, each run at its frame nearest the
  time (:func:`wasserstein_distances`); one part per time, named by the time as
  configured. The configured reference run's sand map and geometry are the
  reference.

Each frame's production target is its own time since the injection start, as in
the run statistics. The comparisons' domains are the selection's, defined in the
``[domains]`` table. A comparison is stored once per results folder, by its kind
and part, with the selection's runs, frames and domains in its scope; a
comparison over another selection is therefore refused there until the stored
one is removed, or it is made in another results folder.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import darsia
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, StepContext
from darsia.presets.workflows.analysis.domain_mass import RunSeries, read_domain_mass
from darsia.presets.workflows.analysis.domains import AnalysisDomain, write_rows
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS_TABLE, DomainSettings
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
    co2_state,
    run_grid,
)
from darsia.presets.workflows.comparison.comparison_events import (
    EventCondition,
    EventQuestion,
    event_table,
    write_events,
)
from darsia.presets.workflows.comparison.ensemble import (
    EnsembleRun,
    EnsembleSettings,
    ensemble_tables,
    write_tables,
)
from darsia.presets.workflows.comparison.ensemble_figures import ensemble_figures
from darsia.presets.workflows.comparison.mass_wasserstein import (
    WassersteinSettings,
    distance_scales,
    frame_cells,
    wasserstein_distances,
    wasserstein_reference,
)
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    protocol_files,
    read_rig,
    run_settings,
)

FIGURES = "figures"
"""The folder of the ensemble's figures, beside its tables."""


@dataclass(frozen=True)
class EventSettings:
    """The questions of the events; the ``[events]`` table."""

    questions: Mapping[str, Mapping]
    """Each question's conditions and how they combine (:class:`EventQuestion`),
    by name."""
    masses: str = "complete"
    """A domain's total, ``"complete"``, or its known part, ``"quantified"``."""


@dataclass(frozen=True)
class DistanceSettings:
    """The comparison times of the distances; the ``[distances]`` table."""

    times_h: Sequence[float]
    """The times [h since the injection start], one part each."""
    tolerance_h: float
    """Largest distance of a run's frame from a time; a run without one is
    missing at that time."""
    reference_run: str
    """The run whose sand map and geometry are the reference of every distance;
    one of the selection's runs."""
    chart_labels: Sequence[int] = ()
    """Labels of the colour checker in the rig's labels."""


def _domains(ctx: StepContext) -> list[AnalysisDomain]:
    """The selection's analysis domains."""
    defined = ctx.settings["domains"].domains
    return [AnalysisDomain(domain_id=n, **defined[n]) for n in ctx.selection.domains]


def _series(ctx: StepContext) -> dict:
    """Each selected run's rows of masses per domain, the CO2 injected up to each
    frame [g] by case id, and its injection protocol."""
    found = {}
    for run in ctx.selection.frames:
        rows = ctx.read(
            "domain-mass",
            lambda f: read_domain_mass(Path(f) / DOMAIN_MASS_TABLE),
            run=run,
        )
        config = run_settings(ctx, run, EXPERIMENT)
        protocol = make_experiment(config).injection_protocol
        injected = {
            row["case_id"]: protocol.injected_mass(time=float(row["elapsed_h"])) * 1000
            for row in rows
        }
        found[run] = (rows, injected, protocol)
    return found


def _events(ctx: StepContext) -> None:
    """Every question's answers per run and per injection."""
    settings = ctx.settings["events"]
    questions = [
        EventQuestion(
            name=name,
            conditions=tuple(EventCondition(**c) for c in question["conditions"]),
            **{k: v for k, v in question.items() if k != "conditions"},
        )
        for name, question in settings.questions.items()
    ]
    found = _series(ctx)
    series = [
        RunSeries.from_domain_mass(rows, injected, settings.masses)
        for rows, injected, _ in found.values()
    ]
    injections = {run: protocol.injections() for run, (_, _, protocol) in found.items()}
    rows = event_table(questions, series, injections)
    ctx.publish(lambda folder: write_events(Path(folder) / "events.csv", rows))


def _ensemble(ctx: StepContext) -> None:
    """The ensemble's tables of the selected runs' domains, and their figures in
    :data:`FIGURES`."""
    runs = [
        EnsembleRun(
            rows,
            injected,
            {row["case_id"]: float(row["elapsed_h"]) for row in rows},
            protocol,
        )
        for rows, injected, protocol in _series(ctx).values()
    ]
    settings = ctx.settings["ensemble"]
    tables = ensemble_tables(runs, _domains(ctx), settings)

    def write(folder):
        write_tables(Path(folder), tables)
        figures = Path(folder) / FIGURES
        figures.mkdir()
        ensemble_figures(figures, tables, runs, settings, ctx.settings["style"])

    ctx.publish(write)


def named_times(times_h: Sequence[float]) -> dict[str, float]:
    """The comparison times by the name of their part: the hours as configured,
    every digit kept (``repr``), e.g. ``"72.33333h"``.

    Raises
    ------
    ValueError
        If a time is configured twice.

    """
    named = {f"{float(t)!r}h": float(t) for t in times_h}
    if len(named) != len(times_h):
        raise ValueError(f"A comparison time is configured twice: {list(times_h)}.")
    return named


def _times(selection: Selection, tables: Mapping) -> tuple[str, ...]:
    """One part per comparison time, named by its hours (:func:`named_times`)."""
    return tuple(named_times(tables["distances"]["times_h"]))


def _distances(ctx: StepContext) -> None:
    """The distances of every pair of runs in every domain at one time, on the
    configured reference run's sand map, which a job must select
    (:attr:`Step.required`)."""
    settings, distances = ctx.settings["wasserstein"], ctx.settings["distances"]
    material, domains = ctx.settings["material"], _domains(ctx)
    time_h = named_times(distances.times_h)[ctx.task.part]
    run = distances.reference_run
    grid = run_grid(ctx.read("rig", read_rig, run=run), material)
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    reference = wasserstein_reference(
        grid.rig.labels,
        grid.rig.permeability.img,
        grid.rig.porosity.img,
        ctx.settings["domains"].sand_labels,
        distances.chart_labels,
        domains,
        settings,
    )
    scales = distance_scales(
        time_h,
        experiment.injection_protocol,
        co2_state(grid, experiment, experiment.experiment_start),
        grid.rig.depth.img,
        grid.rig.facies.dimensions,
        domains,
    )
    frames = {}
    for run, run_frames in ctx.selection.frames.items():
        grid = run_grid(ctx.read("rig", read_rig, run=run), material)
        experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
        hours = [
            experiment.time_since_start(experiment.get_datetime(ctx.photo(run, f)))
            for f in run_frames
        ]
        nearest = int(np.argmin([abs(h - time_h) for h in hours]))
        if abs(hours[nearest] - time_h) > distances.tolerance_h:
            frames[run] = None
            continue
        frame = run_frames[nearest]
        maps = ctx.read("mass-maps", darsia.read_mass_maps, run=run, frame=frame)
        labels = ctx.read("phase-maps", darsia.read_phase_maps, run=run, frame=frame)[
            "labels"
        ]
        frames[run] = frame_cells(
            maps,
            labels,
            grid.rig.labels,
            grid.facies,
            grid.geometry,
            reference,
            settings,
        )
    found = wasserstein_distances(time_h, frames, reference, scales, settings)
    rows = [asdict(distance) for distance in found]
    columns = list(rows[0]) if rows else []
    ctx.publish(
        lambda folder: write_rows(Path(folder) / "distances.csv", rows, columns)
    )


EVENTS = Step(
    "events",
    {"domain-mass": Scope.ALL},
    Order.PARTS,
    Memory(),
    _events,
    {"events": EventSettings},
    sections=EXPERIMENT,
    files=protocol_files,
    domains=True,
)
"""The events of the selected runs."""
ENSEMBLE = Step(
    "ensemble",
    {"domain-mass": Scope.ALL},
    Order.PARTS,
    Memory(),
    _ensemble,
    {"ensemble": EnsembleSettings, "domains": DomainSettings, "style": FigureStyle},
    sections=EXPERIMENT,
    files=protocol_files,
    domains=True,
)
"""The ensemble's tables and figures of the selected runs."""
DISTANCES = Step(
    "distances",
    {"rig": Scope.ALL, "mass-maps": Scope.ALL, "phase-maps": Scope.ALL},
    Order.PARTS,
    Memory(grid=24.0),
    _distances,
    {
        "wasserstein": WassersteinSettings,
        "distances": DistanceSettings,
        "domains": DomainSettings,
        "material": MaterialSettings,
    },
    _times,
    sections=EXPERIMENT,
    files=protocol_files,
    domains=True,
    required=lambda tables, runs: [tables["distances"]["reference_run"]],
)
"""The distances at each comparison time (R9)."""
STEPS = (EVENTS, ENSEMBLE, DISTANCES)
"""The Comparison's steps."""
