"""The Analysis's steps: each frame's mass maps and each run's mass per domain.

- ``mass-maps`` (R7) turns a frame's read signal, phase maps and free gas into its
  CO2 mass per pore volume (:func:`darsia.mass_maps`), with its run's calibration
  and the CO2 tables at the frame's pressure and temperature; each field of the
  maps is stored as an image.
- ``mass-copy`` copies each frame's mass maps onto metric cells, e.g. 1 mm
  (:class:`darsia.MassCopy`), a second result of the frame; each field of the copy,
  and the run's pore volume and domain area per cell, is stored as an image on
  the cells' grid.
- ``domain-mass`` (R8) sums each frame's maps over the analysis domains of the
  selection (:class:`DomainMass`) on the run's sand map, one table per run, each
  frame known by its session and name; the selected domains are part of its
  identity.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import darsia
from darsia.jobs.step import Memory, Order, Scope, Step, StepContext
from darsia.presets.workflows.analysis.domain_mass import DomainMass, write_domain_mass
from darsia.presets.workflows.analysis.domains import AnalysisDomain
from darsia.presets.workflows.calibration.steps import (
    MaterialSettings,
    ReadingSettings,
    co2_state,
    frame_key,
    read_signal,
    run_grid,
    sand_reading,
)
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    protocol_files,
    read_image,
    read_rig,
    run_settings,
)

DOMAIN_MASS_TABLE = "domain_mass.csv"
"""The file of a run's stored mass per domain (R8)."""


@dataclass(frozen=True)
class DomainSettings:
    """The analysis domains by id; the ``[domains]`` table, of which a selection
    names the ones a job covers."""

    domains: Mapping[str, Mapping]
    """Each domain's kind, name and, for a box, its bounds [mm] or, for a layer,
    its label (:class:`AnalysisDomain`)."""
    sand_labels: tuple[int, ...] = ()
    """The labels of the sand layers, which a domain of the sand reads."""


def _mass_maps(ctx: StepContext) -> None:
    """Each frame's CO2 mass per pore volume."""
    run, settings = ctx.task.run, ctx.settings
    grid = run_grid(ctx.read("rig", read_rig), settings["material"])
    calibration = ctx.read("calibration", darsia.read_mass_calibrations)[run]
    reading = sand_reading(settings["reading"])
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    start = experiment.experiment_start
    for frame in ctx.task.frames:
        if ctx.done(frame):
            continue
        image = ctx.read("corrected-image", read_image, frame=frame)
        maps = ctx.read("phase-maps", darsia.read_phase_maps, frame=frame)
        signal, measured = read_signal(
            grid, image.img, maps, reading, settings["run_statistics"]
        )
        date = experiment.get_datetime(ctx.photo(run, frame))
        relative = darsia.ScalarImage(
            darsia.relative_signal(signal, calibration),
            name=frame.name,
            date=date,
            reference_date=start,
            dimensions=image.dimensions,
            origin=image.origin,
        )
        hours = (date - start).total_seconds() / 3600
        factor = darsia.dilution_factor(
            hours,
            grid.geometry.pore_volume(grid.domain & maps["co2"]),
            calibration.anchor_h,
            calibration.anchor_pore_volume_m3,
            calibration.beta,
            calibration.beta_growth,
            signal.dtype,
        )
        result = darsia.mass_maps(
            relative,
            maps["labels"],
            maps["rules"],
            maps["saturation"],
            measured,
            grid.geometry,
            co2_state(grid, experiment, date),
            calibration,
            factor,
        )
        ctx.publish(lambda folder: darsia.write_mass_maps(folder, result), frame=frame)


def _mass_copy(ctx: StepContext) -> None:
    """Each frame's mass maps copied onto metric cells."""
    grid = run_grid(ctx.read("rig", read_rig), ctx.settings["material"])
    copy = darsia.MassCopy(grid.geometry, ctx.settings["mass_copy"])
    for frame in ctx.task.frames:
        if ctx.done(frame):
            continue
        maps = ctx.read("mass-maps", darsia.read_mass_maps, frame=frame)
        labels = ctx.read("phase-maps", darsia.read_phase_maps, frame=frame)["labels"]
        fields = {**copy(maps, labels), **copy.static}
        ctx.publish(
            lambda folder: darsia.write_mass_copy(
                folder, fields, grid.rig.facies, copy.cells, frame.name
            ),
            frame=frame,
        )


def _domain_mass(ctx: StepContext) -> None:
    """The mass of the run's frames per analysis domain of the selection, on the
    run's sand map, whose labels the layer domains name; a frame's rows carry its
    session and name."""
    run, settings = ctx.task.run, ctx.settings["domains"]
    grid = run_grid(ctx.read("rig", read_rig), ctx.settings["material"])
    domains = [
        AnalysisDomain(domain_id=name, **settings.domains[name])
        for name in ctx.selection.domains
    ]
    sums = DomainMass(domains, grid.geometry, grid.rig.labels, settings.sand_labels)
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    start = experiment.experiment_start
    rows = []
    for frame in ctx.task.frames:
        maps = ctx.read("mass-maps", darsia.read_mass_maps, frame=frame)
        date = experiment.get_datetime(ctx.photo(run, frame))
        hours = (date - start).total_seconds() / 3600
        for row in sums(maps, co2_state(grid, experiment, date)):
            case = frame_key(frame)
            rows.append({**row, "run": run, "case_id": case, "elapsed_h": hours})
    ctx.publish(
        lambda folder: write_domain_mass(Path(folder) / DOMAIN_MASS_TABLE, rows)
    )


MASS_MAPS = Step(
    "mass-maps",
    {
        "rig": Scope.OWN,
        "corrected-image": Scope.OWN,
        "phase-maps": Scope.OWN,
        "calibration": Scope.ONE,
    },
    Order.FRAMES,
    Memory(grid=40.0),
    _mass_maps,
    {
        "material": MaterialSettings,
        "reading": ReadingSettings,
        "run_statistics": darsia.RunStatisticsSettings,
    },
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each frame's mass maps (R7)."""
MASS_COPY = Step(
    "mass-copy",
    {"rig": Scope.OWN, "mass-maps": Scope.OWN, "phase-maps": Scope.OWN},
    Order.FRAMES,
    Memory(grid=12.0),
    _mass_copy,
    {"mass_copy": darsia.MassCopySettings, "material": MaterialSettings},
)
"""Each frame's mass maps on metric cells."""
DOMAIN_MASS = Step(
    "domain-mass",
    {"rig": Scope.OWN, "mass-maps": Scope.OWN},
    Order.RUN,
    Memory(grid=24.0),
    _domain_mass,
    {"domains": DomainSettings, "material": MaterialSettings},
    sections=EXPERIMENT,
    files=protocol_files,
    domains=True,
)
"""Each run's mass per analysis domain of the selection (R8)."""
STEPS = (MASS_MAPS, MASS_COPY, DOMAIN_MASS)
"""The Analysis's steps."""
