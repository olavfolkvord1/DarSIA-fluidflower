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

The ensemble's shared production targets are frozen by job authoring from the
selection's provenance, separately from each frame's actual time since injection.
The comparisons' domains are the selection's, defined in the
``[domains]`` table. A comparison is stored once per results folder, by its kind
and part, with the selection's runs, frames and domains in its scope; a
comparison over another selection is therefore refused there until the stored
one is removed, or it is made in another results folder.

"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import darsia
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, StepContext, Term
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from darsia.presets.workflows.analysis.domain_mass import RunSeries, read_domain_mass
from darsia.presets.workflows.analysis.domains import AnalysisDomain, write_rows
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS_TABLE, DomainSettings
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
    co2_state,
    frame_key,
    run_grid,
)
from darsia.presets.workflows.comparison.comparison_events import (
    EventCondition,
    EventQuestion,
    event_table,
    read_events,
    write_events,
)
from darsia.presets.workflows.comparison.ensemble import (
    EnsembleRun,
    EnsembleSettings,
    ensemble_tables,
    event_summary,
    write_tables,
)
from darsia.presets.workflows.comparison.ensemble_figures import ensemble_figures
from darsia.presets.workflows.comparison.mass_wasserstein import (
    ComparedFrame,
    WassersteinSettings,
    distance_rows,
    distance_scales,
    frame_cells,
    wasserstein_distances,
    wasserstein_reference,
)
from darsia.presets.workflows.media.style import FigureStyle
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    output_cells,
    protocol_files,
    read_rig,
    run_settings,
)
from darsia.utils.locations import Locations, normal_path

FIGURES = "figures"
"""The folder of the ensemble's figures, beside its tables."""


def _distance_cell_bytes(tables, run, context) -> int:
    """Retained per-run/domain cell maps and the comparison solver's array work.

    Each region retains a working-precision mass and float64 pixel counts per
    run, and the reference's float64 weight and two masks. The solver budget
    counts scalar/vector work and sparse stencil storage; library factorization
    fill-in depends on the region and is learned separately from observed peaks.
    """
    settings = tables["wasserstein"]
    reference = tables["distances"]["reference_run"]
    runs = context.runs or (reference,)
    count = output_cells(
        tables,
        runs,
        context,
        settings["cell_mm"],
        settings.get("extent_edge_mm"),
        reference,
    )
    domains = len(tables["domains"]["domains"])
    retained = domains * ((context.itemsize + 8) * len(runs) + 8 + 2)
    # Two faces per cell: Bregman flux/force/auxiliary arrays, cell scalars,
    # and eight sparse mixed/stencil operators with value/index storage.
    work = (2 * 12 + 12) * 8 + 8 * 5 * (8 + 4) + 8 * 4
    return count * (retained + work)


@dataclass(frozen=True)
class EventSettings:
    """The questions of the events; the ``[events]`` table."""

    questions: Mapping[str, Mapping]
    """Each question's conditions and how they combine (:class:`EventQuestion`),
    by name."""
    masses: str = "complete"
    """A domain's total, ``"complete"``, or its known part, ``"quantified"``."""


@dataclass(frozen=True)
class EnsembleInputSettings(EnsembleSettings):
    """The ensemble's method settings and its frozen selection targets.

    Job authoring supplies the targets from the selection's provenance, through
    :func:`ensemble_targets`; they are not inferred from acquisition times.

    """

    target_h: Mapping[str, Mapping[str, float]]
    """The exact shared target [h] of each selected case id, by run. Required for
    the workflow adapter; a legacy job without these assignments must be remade."""


def ensemble_targets(
    selection: Selection,
    locations: Locations,
    matches: Mapping[str, Sequence[tuple[Path, float]]],
) -> dict[str, dict[str, float]]:
    """Extract exact selected-photo targets for a job's ensemble configuration.

    Parameters
    ----------
    selection : Selection
        The job's frozen frames.
    locations : Locations
        Their source-photo locations.
    matches : mapping
        Per run, source-photo paths and their target hours, as recorded in the
        selection's pinned provenance. Job authoring verifies that provenance's
        content identity before passing its records. Paths may be relative to
        the project; comparison uses the normal source path, never a basename,
        acquisition time or nearest-time match.

    Returns
    -------
    dict
        Run to case id to target hour, for ``[ensemble].target_h``. The normal job
        configuration freeze makes these assignments part of result identity.

    Raises
    ------
    ValueError
        If a selected source has no target, a source is ambiguous, a record names
        an unselected source, or a target is not a finite number.

    """
    found = {}
    for run, frames in selection.frames.items():
        sources = {}
        for frame in frames:
            source = os.path.normcase(normal_path(locations.photo(run, frame)))
            if source in sources:
                raise ValueError(f"Ambiguous selected source in {run}: {source}.")
            sources[source] = frame_key(frame)
        targets = {}
        for path, target in matches.get(run, ()):
            source = os.path.normcase(normal_path(path, locations.project))
            if source not in sources:
                raise ValueError(f"Unselected target source in {run}: {source}.")
            case = sources[source]
            if case in targets:
                raise ValueError(f"Ambiguous target for {run}/{case}.")
            targets[case] = _target_hour(target, run, case)
        missing = set(sources.values()) - targets.keys()
        if missing:
            raise ValueError(f"Missing frozen targets in {run}: {sorted(missing)}.")
        found[run] = targets
    return found


def _target_hour(value, run, case) -> float:
    """One explicitly assigned, finite target hour."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Invalid frozen target for {run}/{case}: {value!r}.")
    return float(value)


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
    settings = ctx.settings["ensemble"]
    runs = []
    for run, (rows, injected, protocol) in _series(ctx).items():
        assigned = settings.target_h.get(run, {})
        cases = {frame_key(frame) for frame in ctx.selection.frames[run]}
        cases.update(row["case_id"] for row in rows)
        missing = cases - assigned.keys()
        if missing:
            raise ValueError(f"Missing frozen targets in {run}: {sorted(missing)}.")
        targets = {case: _target_hour(assigned[case], run, case) for case in cases}
        runs.append(EnsembleRun(rows, injected, targets, protocol))
    tables = ensemble_tables(runs, _domains(ctx), settings)
    tables["events_summary"] = event_summary(
        ctx.read("events", lambda folder: read_events(Path(folder) / "events.csv"))
    )

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


def _observed_geometry(geometry, labels, rules):
    """Remove the saved chart exclusion from this frame's observed support.

    The reference's continued sand and permeability are unchanged, as are the
    other run's observations. Only pixels explicitly marked COLOR_CHECKER in
    this frame are excluded; unrelated unknown phases remain unknown. Never
    reconstruct the region from current settings or turn its phase into water.
    Return the frame's geometry and the number of previously counted pixels
    excluded, for the result's observation-support diagnostic.
    """
    labels, rules = np.asarray(labels), np.asarray(rules)
    if (
        labels.shape != geometry.counted.shape
        or rules.shape != labels.shape
        or not np.issubdtype(rules.dtype, np.integer)
    ):
        raise ValueError(
            "Saved phase labels, rule map and geometry must share their grid"
        )
    hidden = rules == PhaseRule.COLOR_CHECKER
    if np.any(hidden & (labels != PhaseLabel.UNKNOWN)):
        raise ValueError("Saved COLOR_CHECKER pixels must have unknown phase")
    count = int(np.count_nonzero(hidden & geometry.counted))
    return replace(geometry, counted=geometry.counted & ~hidden), count


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
    frames, selected_frames, observation_pixels = {}, {}, {}
    for run, run_frames in ctx.selection.frames.items():
        grid = run_grid(ctx.read("rig", read_rig, run=run), material)
        experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
        dates = [experiment.get_datetime(ctx.photo(run, f)) for f in run_frames]
        hours = [experiment.time_since_start(date) for date in dates]
        if not hours:
            frames[run] = selected_frames[run] = None
            continue
        nearest = int(np.argmin([abs(h - time_h) for h in hours]))
        if abs(hours[nearest] - time_h) > distances.tolerance_h:
            frames[run] = None
            selected_frames[run] = None
            continue
        frame = run_frames[nearest]
        selected_frames[run] = ComparedFrame(
            frame_key(frame), dates[nearest].isoformat(), hours[nearest]
        )
        maps = ctx.read("mass-maps", darsia.read_mass_maps, run=run, frame=frame)
        phase = ctx.read("phase-maps", darsia.read_phase_maps, run=run, frame=frame)
        labels = phase["labels"]
        geometry, hidden = _observed_geometry(grid.geometry, labels, phase["rules"])
        del phase
        observation_pixels[run] = hidden
        frames[run] = frame_cells(
            maps,
            labels,
            grid.rig.labels,
            grid.facies,
            geometry,
            reference,
            settings,
        )
    found = wasserstein_distances(time_h, frames, reference, scales, settings)
    rows = distance_rows(found, selected_frames)
    columns = list(rows[0]) if rows else []
    ctx.publish(
        lambda folder: write_rows(Path(folder) / "distances.csv", rows, columns),
        measurements={"colorchecker_excluded_counted_pixels": observation_pixels},
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
    {"domain-mass": Scope.ALL, "events": Scope.ONE},
    Order.PARTS,
    Memory(camera_bytes=14.0),
    _ensemble,
    {
        "ensemble": EnsembleInputSettings,
        "domains": DomainSettings,
        "style": FigureStyle,
    },
    sections=EXPERIMENT,
    files=protocol_files,
    domains=True,
)
"""The ensemble's tables and figures of the selected runs."""
DISTANCES = Step(
    "distances",
    {"rig": Scope.ALL, "mass-maps": Scope.ALL, "phase-maps": Scope.ALL},
    Order.PARTS,
    Memory(
        camera=1.0,
        grid=28.0,
        camera_bytes=24.0,
        grid_bytes=174.0,
        terms=(Term(_distance_cell_bytes, fixed_bytes=1),),
    ),
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
