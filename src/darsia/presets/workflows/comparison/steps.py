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
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import darsia
from darsia.jobs.step import Memory, Order, Scope, Selection, Step, StepContext, Term
from darsia.multiphase.phase_rules import PhaseLabel, PhaseRule
from darsia.presets.workflows.analysis.domain_mass import (
    CENTRES,
    COLUMNS as DOMAIN_COLUMNS,
    RunSeries,
    read_domain_mass,
)
from darsia.presets.workflows.analysis.domains import COMPONENTS, AnalysisDomain, write_rows
from darsia.presets.workflows.analysis.steps import DOMAIN_MASS_TABLE, DomainSettings
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
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
    read_grid_model,
    read_rig,
    run_settings,
)
from darsia.utils.locations import Locations, normal_path

FIGURES = "figures"
"""The folder of the ensemble's figures, beside its tables."""


def _series_dimensions(tables, context):
    definitions = tables.get("domains", {}).get("domains", {})
    domains = context.domains or tuple(definitions)
    return definitions, domains


def _domain_series_bytes(tables, _run, context) -> int:
    """Parsed domain-mass rows and their dense float64 RunSeries arrays."""
    definitions, domains = _series_dimensions(tables, context)
    domain_rows = context.frames * (len(domains) + 1)  # the optional full-rig row
    # CSV values are independent str objects. The numeric spelling covers the
    # sign, float64 precision, and a three-digit scientific exponent.
    number = sys.getsizeof("-1.2345678901234567e-308")
    description = max(
        (
            sys.getsizeof(key)
            + sys.getsizeof(config.get("name", key))
            + sys.getsizeof(config.get("kind", "whole"))
            for key, config in definitions.items()
        ),
        default=3 * sys.getsizeof("whole"),
    )
    run = max(map(sys.getsizeof, context.runs), default=sys.getsizeof(""))
    case = context.case_id_bytes or sys.getsizeof("")
    pointers = np.dtype(np.intp).itemsize
    row = (
        sys.getsizeof(dict.fromkeys(DOMAIN_COLUMNS))
        + (len(DOMAIN_COLUMNS) - 5) * number
        + run + case + description
        + 2 * pointers  # the retained CSV list and the RunSeries row list
    )
    arrays = np.dtype(np.float64).itemsize * (
        domain_rows * len(COMPONENTS) * (1 + len(CENTRES)) + 2 * context.frames
    )
    return domain_rows * row + arrays


def _ensemble_tables_bytes(tables, _run, context) -> int:
    """The mass and centre long tables retained while rendering the ensemble."""
    _, domains = _series_dimensions(tables, context)
    rows = context.frames * len(COMPONENTS)
    pointer = np.dtype(np.intp).itemsize
    number = max(sys.getsizeof(0.0), sys.getsizeof(np.float64()))
    # Long rows reference existing IDs and labels. The numeric fields may be
    # new scalars. The centre time reducer also indexes rows by four fields.
    mass_row = sys.getsizeof(dict.fromkeys(range(11))) + 5 * number + pointer
    centre_row = (
        sys.getsizeof(dict.fromkeys(range(13))) + 7 * number + pointer
        + sys.getsizeof((None, None, None, None)) + 2 * pointer
    )
    return rows * ((len(domains) + 1) * mass_row + len(domains) * centre_row)


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
    domains = (
        1
        if tables["distances"].get("split_by_domain")
        else len(tables["domains"]["domains"])
    )
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
    excluded_runs: Mapping[str, Sequence[str]] = field(default_factory=dict)
    """Runs omitted from specific domains, by configured domain ID."""
    split_by_domain: bool = False
    """Give each selected domain its own task at each comparison time."""

    def __post_init__(self) -> None:
        if any(
            isinstance(runs, str)
            or not isinstance(runs, Sequence)
            or not all(isinstance(run, str) for run in runs)
            for runs in self.excluded_runs.values()
        ):
            raise ValueError("Distance exclusions must list run IDs for each domain.")


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


def distance_parts(
    times_h: Sequence[float], domains: Sequence[str], split_by_domain: bool = False
) -> dict[str, tuple[float, str | None]]:
    """Map a task part to its target hour and its optional single domain.

    Domain positions keep arbitrary domain IDs out of result directory names.
    """
    times = named_times(times_h)
    if not split_by_domain:
        return {part: (hour, None) for part, hour in times.items()}
    if not domains:
        raise ValueError("Choose at least one domain for the distance comparisons.")
    return {
        f"{part}--domain-{index:04d}": (hour, domain)
        for part, hour in times.items()
        for index, domain in enumerate(domains, 1)
    }


def _times(selection: Selection, tables: Mapping) -> tuple[str, ...]:
    """One part per comparison time, or per time and selected domain."""
    distances = tables["distances"]
    return tuple(
        distance_parts(
            distances["times_h"],
            selection.domains,
            distances.get("split_by_domain", False),
        )
    )


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


def start_co2(baseline: darsia.Image, experiment) -> darsia.CO2MassAnalysis:
    """The reference run's CO2 at the experiment's start, as our production's
    comparison took it for the distances' scales: the official
    :class:`darsia.CO2MassAnalysis` on the reference rig's baseline (its nominal
    heights), at the pressure and temperature of the experiment's protocol and
    their gradients at the start.

    Parameters
    ----------
    baseline : darsia.Image
        The reference rig's baseline.
    experiment : darsia.ProtocolledExperiment
        The reference run's experiment.

    Returns
    -------
    darsia.CO2MassAnalysis
        The CO2's solubility and gas density on the baseline's grid.

    """
    protocol = experiment.pressure_temperature_protocol
    state = protocol.get_state(experiment.experiment_start)
    gradient = protocol.get_gradient(experiment.experiment_start)
    return darsia.CO2MassAnalysis(
        baseline=baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )


def _distances(ctx: StepContext) -> None:
    """The selected pairs and domains at one time on the reference run's sand map."""
    settings, distances = ctx.settings["wasserstein"], ctx.settings["distances"]
    material, domains = ctx.settings["material"], _domains(ctx)
    missing_domains = (
        distances.excluded_runs.keys() - ctx.settings["domains"].domains.keys()
    )
    if missing_domains:
        raise ValueError(
            f"Unknown distance domains in excluded_runs: {sorted(missing_domains)}."
        )
    time_h, domain_id = distance_parts(
        distances.times_h, ctx.selection.domains, distances.split_by_domain
    )[ctx.task.part]
    if domain_id is not None:
        domains = [domain for domain in domains if domain.domain_id == domain_id]
    run = distances.reference_run
    grid = run_grid(ctx.read("rig", read_rig, run=run), material)
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    # With measured models, the cells cover the measured corners of every compared
    # run's measurement canvas.
    models = {r: ctx.read("rig", read_grid_model, run=r) for r in ctx.selection.frames}
    extent = None
    if any(model is not None for model, _ in models.values()):
        if settings.extent_edge_mm is None:
            raise ValueError("Measured models need [wasserstein] extent_edge_mm")
        dtype = grid.rig.pixel_area.img.dtype
        corners = [
            darsia.grid_corners(grid.rig.labels, model, dtype, canvas)
            for model, canvas in models.values()
        ]
        extent = darsia.measured_extent(
            corners, settings.cell_mm, settings.extent_edge_mm
        )
    # The reference's measured areas weight its means: by its model alone, in
    # double precision, as our production's comparison measured them.
    areas = None
    if grid.rig.grid_to_rig is not None:
        areas = darsia.grid_pixel_areas(
            grid.rig.labels, grid.rig.grid_to_rig, np.float64
        )
    reference = wasserstein_reference(
        grid.rig.labels,
        grid.rig.permeability.img,
        grid.rig.porosity.img,
        ctx.settings["domains"].sand_labels,
        distances.chart_labels,
        domains,
        settings,
        grid.positions,
        extent,
        areas,
    )
    scales = distance_scales(
        time_h,
        experiment.injection_protocol,
        start_co2(grid.rig.baseline, experiment),
        grid.rig.depth.img,
        grid.rig.facies.dimensions,
        domains,
    )
    del grid, areas
    frames, selected_frames, observation_pixels = {}, {}, {}
    for run, run_frames in ctx.selection.frames.items():
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
        grid = run_grid(ctx.read("rig", read_rig, run=run), material)
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
            grid.positions,
        )
        # Only the independent comparison cells are needed after this run.
        del maps, labels, geometry, grid
    found = wasserstein_distances(
        time_h,
        frames,
        reference,
        scales,
        settings,
        excluded_runs=distances.excluded_runs,
    )
    rows = distance_rows(found, selected_frames)
    columns = list(rows[0]) if rows else []
    measurements = {"colorchecker_excluded_counted_pixels": observation_pixels}
    if distances.excluded_runs:
        measurements["domain_excluded_runs"] = dict(distances.excluded_runs)
    ctx.publish(
        lambda folder: write_rows(Path(folder) / "distances.csv", rows, columns),
        measurements=measurements,
    )


EVENTS = Step(
    "events",
    {"domain-mass": Scope.ALL},
    Order.PARTS,
    Memory(terms=(Term(_domain_series_bytes, fixed_bytes=1),)),
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
    Memory(
        camera_bytes=14.0,
        terms=(
            Term(_domain_series_bytes, fixed_bytes=1),
            Term(_ensemble_tables_bytes, fixed_bytes=1),
        ),
    ),
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
