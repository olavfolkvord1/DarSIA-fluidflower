"""The Calibration's steps: the phase model, each run's overflow, phase maps and
statistics, and the calibration fit with its folds.

- ``phase-model`` (R3) stores the series' phase-model folder named in the
  configuration, so that every phase map records the model it was made with; the
  folder's content is part of its identity, so a model trained again into the same
  folder is not taken for the stored one.
- ``overflow`` gives each frame of a run the free-gas volume of the water its
  overflow displaced, from the run's own scale readings where they reach, else
  none (the model's gas).
- ``phase-maps`` (R4) runs the phase model on each frame and the phase rules and
  the gas allocation over the run in time order; each frame stores its phase and
  rule maps, its free-gas saturation and the CO2 of its labels before the rules,
  and a run resumes at its first missing frame.
- ``run-statistics`` (R5) gives each frame the histograms of its sands' dissolved
  CO2 signal, its gas statistics and the column estimate of its CO2 pixels
  without a measurement of their own (:func:`darsia.unknown_statistics`), as the
  fit reads them.
- ``calibration`` (R6) fits the gain and the dilution across the fit runs, and
  each fold leaves one run out.

"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import darsia
from darsia.jobs.step import (
    Each,
    Memory,
    Order,
    Scope,
    Selection,
    Step,
    StepContext,
    run_tables,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    protocol_files,
    read_image,
    read_rig,
    run_settings,
)


@dataclass(frozen=True)
class PhaseModelSource:
    """The series' phase-model folder; the ``[phase_model]`` table."""

    folder: str
    """The model folder, relative to the project."""
    pixel_size_mm: float | None = None
    """Pixel size of the training grid of a delivered folder."""
    accept_canvas: bool = False
    """Whether to apply a model drawn on another canvas, or recording none."""


@dataclass(frozen=True)
class MaterialSettings:
    """The sands' irreducible water and the colour checker; the ``[material]``
    table."""

    swi: Mapping[str, float]
    """Swi of each facies, by id; a facies without one holds no quantified gas."""
    measured_swi: Sequence[int] = ()
    """Facies whose Swi is measured; the fit may fit the others."""
    chart_label: int | None = None
    """Label of the colour checker in the rig's labels, if any."""


@dataclass(frozen=True)
class SandPath:
    """One sand's colour path and signal function; a table of ``[reading]``."""

    colors: Sequence[Sequence[float]]
    """Key colours of the path, relative to the baseline (RGB)."""
    values: Sequence[float]
    """Values of the path at its key colours."""
    supports: Sequence[float]
    """Supports of the signal function."""
    table: Sequence[float]
    """Signal at the supports."""
    factor: float
    """The path's scale over the sand's measured colour range."""


@dataclass(frozen=True)
class ReadingSettings:
    """The sands' reading of the dissolved CO2; the ``[reading]`` table."""

    sands: Mapping[str, SandPath]
    """Each sand's colour path, by facies id."""


@dataclass(frozen=True)
class FitRuns:
    """The runs of the calibration fit; the ``[calibration_runs]`` table."""

    fit: Sequence[str]
    """The runs fitted."""
    folds: Sequence[str] = ()
    """The fit runs left out one at a time."""


def frame_key(frame: darsia.Frame) -> str:
    """A frame's name in a run's tables: its session and its photo's name."""
    return f"{frame.session}/{frame.name}"


@dataclass
class RunGrid:
    """What a run's steps read of its rig, in the run grid's precision."""

    rig: Rig
    facies: np.ndarray
    permeability: dict[int, float]
    domain: np.ndarray
    geometry: darsia.RunGeometry
    chart: np.ndarray | None
    pixel_mm: float
    datum_m: float
    optical_pitch: tuple[float, float]
    pixel_area_mm2: np.ndarray
    """Each pixel's area [mm2], the rig's pixel areas in double precision times a
    million, as our production's phase rules took them."""


def run_grid(rig: Rig, material: MaterialSettings) -> RunGrid:
    """A run's grid: its facies and their permeability, the domain of the facies
    with one, the geometry with each facies' Swi, the colour checker, the pixel
    size, the datum of the hydrostatic head and the pitch on the sand plane.

    Depth, porosity and Swi take the precision of the rig's pixel areas, the run
    grid's.

    """
    facies = rig.facies.img
    dtype = rig.pixel_area.img.dtype
    permeability = {
        int(f): float(rig.permeability.img[facies == f].flat[0])
        for f in np.unique(facies)
    }
    domain = np.isin(facies, list(permeability))
    swi = np.full(facies.shape, np.nan, dtype)
    for name, value in material.swi.items():
        swi[facies == int(name)] = value
    chart = None
    if material.chart_label is not None:
        chart = rig.labels.img == material.chart_label
    geometry = darsia.run_geometry(
        rig.facies,
        rig.pixel_area.img,
        rig.depth.img.astype(dtype),
        rig.porosity.img.astype(dtype),
        swi,
        domain,
        chart,
    )
    scaling = [
        c
        for c in rig.shape_corrections
        if isinstance(c, darsia.MetricScalingCorrection)
    ]
    datum = scaling[0].native_origin[1] if scaling else rig.baseline.origin[1]
    return RunGrid(
        rig,
        facies,
        permeability,
        domain,
        geometry,
        chart,
        float(rig.baseline.voxel_size[0]) * 1000,
        float(datum),
        darsia.optical_pitch_m(rig.pixel_area, domain),
        np.asarray(rig.pixel_area.img, np.float64) * 1e6,
    )


def co2_state(grid: RunGrid, experiment, date) -> darsia.PrecisionCO2MassAnalysis:
    """Solubility and gas density of CO2 on a run's grid at a frame's pressure and
    temperature (the run's pressure-temperature protocol)."""
    state = experiment.pressure_temperature_protocol.get_state(date)
    gradient = experiment.pressure_temperature_protocol.get_gradient(date)
    return darsia.PrecisionCO2MassAnalysis(
        grid.rig.baseline,
        datum_m=grid.datum_m,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )


def sand_reading(settings: ReadingSettings) -> darsia.SandColorPaths:
    """The sands' colour paths of the reading settings."""
    paths, factors = {}, {}
    for name, sand in settings.sands.items():
        colors = [np.asarray(c, float) for c in sand.colors]
        path = darsia.ColorPath(colors=colors, base_color=np.zeros(3), mode="rgb")
        paths[int(name)] = darsia.ColorPathSignal(
            darsia.ColorPathInterpolation(
                path, darsia.ColorMode.RELATIVE, list(sand.values)
            ),
            darsia.PWTransformation(list(sand.supports), list(sand.table)),
        )
        factors[int(name)] = sand.factor
    return darsia.SandColorPaths(paths, factors)


def read_signal(grid, image, maps, reading, settings) -> tuple[np.ndarray, np.ndarray]:
    """A frame's read signal of its dissolved CO2, averaged inside each sand's
    dissolved CO2, as the run statistics read it, and the pixels whose colour is
    measured (finite in the corrected image)."""
    support = grid.domain & np.isfinite(image).all(axis=-1)
    labels, rules = maps["labels"], maps["rules"]
    dissolved = (
        support
        & (labels == darsia.PhaseLabel.DISSOLVED)
        & ~darsia.model_gas(labels, rules)
    )
    baseline = grid.rig.baseline.img
    signal = reading(image, baseline, support, dissolved, grid.facies)
    optical = darsia.optical_grid(
        grid.rig.facies, grid.optical_pitch, baseline.dtype.type
    )
    rev = darsia.REV(size=settings.mean_mm / 1000, img=optical)
    sands = {sand: grid.facies == sand for sand in reading.sands}
    return darsia.sand_mean(signal, dissolved, sands, rev), support


def _phase_model(ctx: StepContext) -> None:
    """Store the series' phase-model folder."""
    source = Path(ctx.settings["phase_model"].folder)
    ctx.publish(lambda folder: shutil.copytree(source, folder, dirs_exist_ok=True))


def _model_files(tables: Mapping, project, folders) -> list[str]:
    """The series' phase-model folder."""
    return [str(tables["phase_model"]["folder"])]


def _overflow_workbook(tables: Mapping) -> str | None:
    """A run's overflow workbook, ``[protocols] overflow`` of its tables."""
    return tables.get("protocols", {}).get("overflow")


def _overflow_files(tables: Mapping, project, folders) -> list[str]:
    """The run's protocols and its overflow workbook."""
    workbook = _overflow_workbook(tables)
    found = protocol_files(tables, project, folders)
    return found + ([str(workbook)] if workbook is not None else [])


def _overflow(ctx: StepContext) -> None:
    """Each frame's free-gas volume from the run's own scale readings."""
    run = ctx.task.run
    settings = ctx.settings["overflow"]
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    start = experiment.experiment_start
    path = _overflow_workbook(run_tables(ctx.tables, run))
    volume = None
    if path is not None:
        times, weights = darsia.OverflowProtocol(Path(path)).readings(
            start, list(settings.statuses)
        )
        volume = darsia.displaced_volume(times, weights, settings)
    volumes = {}
    for frame in ctx.task.frames:
        hours = (experiment.get_datetime(ctx.photo(run, frame)) - start).total_seconds()
        usable = (
            volume is not None and volume.volume_ml is not None and not volume.reasons
        )
        volumes[frame_key(frame)] = (
            darsia.measured_reference(volume, hours / 3600) if usable else None
        )
    reasons = [] if volume is None else list(volume.reasons)
    ctx.publish(lambda folder: darsia.write_frame_volumes(folder, volumes, reasons))


class _Probabilities(Sequence):
    """A run's phase probabilities by frame, computed when asked for, a window of
    them kept."""

    def __init__(self, ctx, model, grid, frames, kept: int) -> None:
        self.ctx, self.model, self.grid, self.frames = ctx, model, grid, frames
        self.image = lru_cache(maxsize=kept + 1)(self._image)
        self.probabilities = lru_cache(maxsize=kept)(self._probabilities)

    def _image(self, index: int) -> np.ndarray:
        return self.ctx.read(
            "corrected-image", read_image, frame=self.frames[index]
        ).img

    def _probabilities(self, index: int):
        previous = self.image(index - 1) if index > 0 else None
        return darsia.phase_probabilities(
            self.model,
            self.image(index),
            self.grid.rig.baseline.img,
            previous,
            pixel_size_mm=self.grid.pixel_mm,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        return self.probabilities(int(index))


def _phase_maps(ctx: StepContext) -> None:
    """The run's phase maps in time order, resumed at its first missing frame.

    Raises
    ------
    ValueError
        If the run's frames are not in time order, which the phase maps and the
        allocation read by index.

    """
    run, frames, settings = ctx.task.run, ctx.task.frames, ctx.settings
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    dates = [experiment.get_datetime(ctx.photo(run, frame)) for frame in frames]
    late = [f.name for f, a, b in zip(frames[1:], dates, dates[1:]) if not a < b]
    if late:
        raise ValueError(f"The frames of run {run} are not in time order: {late}.")
    rules, source = settings["phase_rules"], settings["phase_model"]
    grid = run_grid(ctx.read("rig", read_rig), settings["material"])
    model = ctx.read(
        "phase-model",
        lambda folder: darsia.read_phase_model(
            folder,
            source.pixel_size_mm,
            canvas=grid.rig.canvas,
            accept_canvas=source.accept_canvas,
        ),
    )
    volumes = ctx.read("overflow", darsia.read_frame_volumes)["volumes"]
    start = next((k for k, frame in enumerate(frames) if not ctx.done(frame)), None)
    if start is None:
        return
    kept = 2 * rules.neighbours_per_side + 2
    maps = darsia.free_gas(
        _Probabilities(ctx, model, grid, frames, kept),
        dates,
        grid.rig.facies,
        grid.domain,
        grid.pixel_area_mm2,
        grid.permeability,
        experiment.injection_protocol,
        rules,
        grid.geometry.pore_volume_m3,
        grid.geometry.swi,
        [volumes[frame_key(frame)] for frame in frames],
        settings["gas_allocation"],
        grid.chart,
        gas_threshold=model.gas_threshold,
        start=start,
        anchor_saturation=lambda k: ctx.own(darsia.read_phase_maps, frame=frames[k])[
            "saturation"
        ],
    )
    for frame, arrays in zip(frames[start:], maps):
        ctx.publish(lambda f: darsia.write_phase_maps(f, *arrays), frame=frame)


def _run_statistics(ctx: StepContext) -> None:
    """Each frame's sand histograms and gas statistics, as the fit reads them."""
    run, frames, settings = ctx.task.run, ctx.task.frames, ctx.settings
    statistics_settings = settings["run_statistics"]
    grid = run_grid(ctx.read("rig", read_rig), settings["material"])
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    protocol, start = experiment.injection_protocol, experiment.experiment_start
    volumes = ctx.read("overflow", darsia.read_frame_volumes)
    reading = sand_reading(settings["reading"])
    rows = []
    for index, frame in enumerate(frames):
        image = ctx.read("corrected-image", read_image, frame=frame)
        maps = ctx.read("phase-maps", darsia.read_phase_maps, frame=frame)
        date = experiment.get_datetime(ctx.photo(run, frame))
        co2 = co2_state(grid, experiment, date)
        capacity = darsia.dissolved_capacity(
            grid.geometry.pore_volume_m3, co2.solubility_co2, maps["saturation"]
        )
        support = grid.domain & np.isfinite(image.img).all(axis=-1)
        histograms = next(
            darsia.run_statistics(
                [(image.img, maps["labels"], maps["rules"], support, capacity)],
                grid.rig.baseline.img,
                grid.rig.facies,
                grid.optical_pitch,
                grid.geometry.counted,
                reading,
                statistics_settings,
            )
        )
        gas = darsia.gas_statistics(
            maps["co2"],
            darsia.model_gas(maps["labels"], maps["rules"]),
            maps["saturation"],
            grid.facies,
            grid.geometry.counted,
            grid.geometry.pore_volume_m3,
            co2.density_gaseous_co2,
            co2.solubility_co2,
        )
        signal, measured = read_signal(
            grid, image.img, maps, reading, statistics_settings
        )
        unknown = darsia.unknown_statistics(
            darsia.relative_signal(signal, statistics_settings),
            maps["labels"],
            maps["rules"],
            measured,
            grid.geometry,
            co2,
            settings["unknown_mass"],
        )
        hours = (date - start).total_seconds() / 3600
        volume = volumes["volumes"][frame_key(frame)]
        rows.append(
            darsia.CalibrationFrame(
                run=run,
                index=index,
                elapsed_h=hours,
                target_h=hours,
                injected_g=protocol.injected_mass(time=hours) * 1000,
                dissolved=histograms,
                gas=gas,
                column=unknown.column_estimate(),
                overflow_m3=volume,
                measured=volume is not None and volumes["measured"],
            )
        )
    dtype = grid.rig.pixel_area.img.dtype
    ctx.publish(lambda folder: darsia.write_frames(folder, run, rows, dtype))


def _calibration_parts(selection: Selection, tables: Mapping) -> tuple[str, ...]:
    """The global fit, then one fold per run left out."""
    folds = tables.get("calibration_runs", {}).get("folds", ())
    return ("fit",) + tuple(f"fold-{run}" for run in folds)


def _calibration(ctx: StepContext) -> None:
    """The fit across the fit runs, or one fold leaving a run out."""
    settings = ctx.settings
    runs, material = settings["calibration_runs"], settings["material"]
    frames = [
        frame
        for run in ctx.selection.frames
        for frame in ctx.read("run-statistics", darsia.read_frames, run=run)
    ]
    swi = {int(k): float(v) for k, v in material.swi.items()}
    measured = [int(k) for k in material.measured_swi]
    arguments = (frames, list(runs.fit), settings["calibration_fit"], swi, measured)
    predictions = []
    if ctx.task.part == "fit":
        calibration = darsia.fit_calibration(*arguments)
    else:
        held = ctx.task.part.removeprefix("fold-")
        calibration, predictions = darsia.calibration_fold(*arguments, held)
    masses = darsia.mass_calibrations(calibration, settings["run_statistics"])
    ctx.publish(
        lambda folder: darsia.write_calibration(
            folder, calibration, masses, predictions
        )
    )


PHASE_MODEL = Step(
    "phase-model",
    {},
    Order.PARTS,
    Memory(),
    _phase_model,
    {"phase_model": PhaseModelSource},
    files=_model_files,
    selected=False,
)
"""The series' phase model (R3), made from no run of the selection."""
OVERFLOW = Step(
    "overflow",
    {},
    Order.RUN,
    Memory(),
    _overflow,
    {"overflow": darsia.OverflowSettings},
    sections=EXPERIMENT,
    files=_overflow_files,
)
"""Each run's free-gas volume per frame from its overflow."""
PHASE_MAPS = Step(
    "phase-maps",
    {
        "rig": Scope.OWN,
        "corrected-image": Scope.OWN,
        "phase-model": Scope.ONE,
        "overflow": Scope.OWN,
    },
    Order.RUN,
    Memory(grid=40.0, held=0.0),
    _phase_maps,
    {
        "phase_model": PhaseModelSource,
        "phase_rules": darsia.PhaseRuleSettings,
        "gas_allocation": darsia.GasAllocationSettings,
        "material": MaterialSettings,
    },
    each=Each.FRAME,
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each frame's phase maps and free-gas saturation (R4)."""
RUN_STATISTICS = Step(
    "run-statistics",
    {
        "rig": Scope.OWN,
        "corrected-image": Scope.OWN,
        "phase-maps": Scope.OWN,
        "overflow": Scope.OWN,
    },
    Order.RUN,
    Memory(grid=24.0),
    _run_statistics,
    {
        "run_statistics": darsia.RunStatisticsSettings,
        "reading": ReadingSettings,
        "material": MaterialSettings,
        "unknown_mass": darsia.UnknownMassSettings,
    },
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each run's statistics of its frames (R5)."""
CALIBRATION = Step(
    "calibration",
    {"run-statistics": Scope.ALL},
    Order.PARTS,
    Memory(),
    _calibration,
    {
        "calibration_fit": darsia.CalibrationFitSettings,
        "calibration_runs": FitRuns,
        "material": MaterialSettings,
        "run_statistics": darsia.RunStatisticsSettings,
    },
    _calibration_parts,
)
"""The calibration fit and its folds (R6)."""
STEPS = (PHASE_MODEL, OVERFLOW, PHASE_MAPS, RUN_STATISTICS, CALIBRATION)
"""The Calibration's steps."""
