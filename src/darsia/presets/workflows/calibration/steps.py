"""The Calibration's steps: the phase model, each run's overflow, phase maps and
statistics, and the calibration fit with its folds.

- ``phase-model`` (R3) stores the series' phase-model folder named in the
  configuration, so that every phase map records the model it was made with; the
  folder's content is part of its identity, so a model trained again into the same
  folder is not taken for the stored one.
- ``displaced-volume`` gives each run's displaced water from its own scale
  readings, with its approved evaporation (:func:`darsia.approved_evaporation`),
  and what another run compares it by as a donor: the sand map, the recipe, the
  pressure record and the injection period.
- ``overflow`` gives each frame of a run the free-gas volume of the water its
  overflow displaced: from the run's own readings where they reach, else from
  the most similar fit run with readings (:func:`darsia.ranked_donors`), else
  none (the model's gas). Where that donor is held out of a calibration fold,
  the frames also get the next donor's volumes, which the fold reads.
- ``phase-maps`` (R4) runs the phase model on each frame and the phase rules and
  the gas allocation over the run in time order; each frame stores its phase and
  rule maps, its free-gas saturation and the CO2 of its labels before the rules,
  also without a held-out donor's volumes, and a run resumes at its first missing
  frame.
- ``color-ranges`` measures each sand's colour range on the configured runs'
  frames (:func:`darsia.color_ranges`), each run on its own colour paths.
- ``path-ratios`` measures each sand's path ratio to the reference sand where two
  sands meet, on the configured run's frames (:func:`darsia.path_ratios`), read
  on each sand's own path as a share of its measured colour range, in its
  ordinary light (:func:`darsia.shadow_light`), up to the ceiling of the
  concentration curve at the configured gain.
- ``run-statistics`` (R5) gives each frame the histograms of its sands' dissolved
  CO2 signal, its gas statistics and the column estimate of its CO2 pixels
  without a measurement of their own (:func:`darsia.unknown_statistics`), as the
  fit reads them. Every sand is read by Beer–Lambert as the reference sand
  (:func:`response_reading`): on the run's reference-sand background, with the
  measured path ratios and the reference sand's factor, its path's scale over
  its measured colour range.
- ``calibration`` (R6) fits the gain and the dilution across the fit runs, and
  each fold leaves one run out, reading every run's frames without its
  estimate. With calibration groups (:class:`FitRuns`), e.g. runs photographed
  with other camera settings, each group has its own fit and folds over its own
  runs, and a run's mass maps read its group's fit.

"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
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
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    corrected_photo,
    protocol_files,
    read_image,
    read_rig,
    run_settings,
)

READING = ("reading",)
"""The section of the sands' colour paths, which a step reads per run
(:func:`run_reading`)."""


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
    scale: float
    """The path's scale: the sand's factor is the scale over its measured colour
    range (``color-ranges``)."""


@dataclass(frozen=True)
class ReadingSettings:
    """The sands' colour paths; the ``[reading]`` table, a run's own
    (``[runs.<run>.reading]``) merged over it."""

    sands: Mapping[str, SandPath]
    """Each sand's colour path, by facies id."""


@dataclass(frozen=True)
class ResponseRuns:
    """The runs the sands' response is measured on; the ``[response]`` table."""

    range_runs: Sequence[str]
    """The runs whose frames give each sand's colour range."""
    ratio_run: str
    """The run whose frames give the sands' path ratios."""
    ratio_gain: float
    """The gain of the reading on the colour ranges alone: the top of the
    concentration curve over it is the ceiling signal of the path ratios."""


@dataclass(frozen=True)
class FitGroup:
    """One calibration group; a table of ``[calibration_runs.groups]``."""

    runs: Sequence[str]
    """The runs of the group: their mass maps read its fit; none for the one
    calibration of every run."""
    fit: Sequence[str]
    """The runs fitted, of the group's runs."""
    folds: Sequence[str] = ()
    """The fit runs left out one at a time."""

    def __post_init__(self) -> None:
        """Refuse fit runs outside the group's runs, and folds outside its fit."""
        outside = [run for run in self.fit if self.runs and run not in self.runs]
        outside += [run for run in self.folds if run not in self.fit]
        if outside:
            raise ValueError(
                f"Runs {outside} of a calibration's fit or folds are not its own."
            )


@dataclass(frozen=True)
class FitRuns:
    """The runs of the calibration fit; the ``[calibration_runs]`` table: the fit
    runs and folds of one calibration of every run, or one group of runs per
    calibration (``groups``), e.g. runs photographed with other camera settings.
    """

    fit: Sequence[str] = ()
    """The runs fitted, without groups."""
    folds: Sequence[str] = ()
    """The fit runs left out one at a time, without groups."""
    groups: Mapping[str, FitGroup] = field(default_factory=dict)
    """Each calibration group by its name; none for one calibration."""

    def __post_init__(self) -> None:
        """Refuse a table that names both the fit runs and groups, or neither, and
        folds outside the fit runs."""
        if bool(self.groups) == bool(self.fit):
            raise ValueError(
                "[calibration_runs] names either the fit runs or the groups."
            )
        FitGroup((), self.fit, self.folds)

    def group_of(self, run: str) -> str | None:
        """The name of a run's calibration group; None without groups.

        Raises
        ------
        ValueError
            If the run is in no group, or in more than one.

        """
        if not self.groups:
            return None
        found = [name for name, group in self.groups.items() if run in group.runs]
        if len(found) != 1:
            raise ValueError(f"Run {run} is in {len(found)} calibration groups.")
        return found[0]

    def group(self, run: str) -> FitGroup:
        """The calibration of a run: its group, or without groups the one fit
        (with no runs of its own)."""
        name = self.group_of(run)
        return FitGroup((), self.fit, self.folds) if name is None else self.groups[name]

    def parts(self, runs: Sequence[str]) -> dict[str, tuple[FitGroup, str | None]]:
        """The parts of the calibration: each fit and each fold, by name, with its
        group and its held-out run (None for the fit); a group is made only when
        one of its runs is among ``runs``.

        Without groups the parts are ``fit`` and ``fold-<run>``; with groups,
        ``fit-<group>`` and ``fold-<group>-<run>``.

        """
        if not self.groups:
            one = FitGroup((), self.fit, self.folds)
            return {"fit": (one, None)} | {f"fold-{r}": (one, r) for r in self.folds}
        found: dict[str, tuple[FitGroup, str | None]] = {}
        for name, group in self.groups.items():
            if any(run in group.runs for run in runs):
                found[f"fit-{name}"] = (group, None)
                found |= {f"fold-{name}-{r}": (group, r) for r in group.folds}
        return found


def fit_runs(tables: Mapping) -> FitRuns:
    """The calibration's runs from the configuration's ``[calibration_runs]``."""
    return settings_from_table(FitRuns, tables["calibration_runs"])


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
    positions: darsia.GridPositions
    """Each pixel's position on the rig and the grid's pitch, by the rig's measured
    model of its grid (:func:`darsia.grid_positions`), in the grid's precision."""
    pixel_area_mm2: np.ndarray
    """Each pixel's area [mm2], the rig's pixel areas in double precision times a
    million, as our production's phase rules took them."""


def run_grid(rig: Rig, material: MaterialSettings) -> RunGrid:
    """A run's grid: its facies and their permeability, the domain of the facies
    with one, the geometry with each facies' Swi, the colour checker, the pixel
    size, the datum of the hydrostatic head, the pitch on the sand plane, and each
    pixel's position on the rig by the rig's measured model of its grid.

    Depth, porosity, Swi and the measured positions take the precision of the
    rig's pixel areas, the run grid's.

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
    positions = darsia.grid_positions(
        rig.facies, rig.grid_to_rig, dtype, rig.measurement_canvas
    )
    geometry = darsia.run_geometry(
        rig.facies,
        rig.pixel_area.img,
        rig.depth.img.astype(dtype),
        rig.porosity.img.astype(dtype),
        swi,
        domain,
        chart,
        positions,
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
        positions,
        np.asarray(rig.pixel_area.img, np.float64) * 1e6,
    )


def co2_state(grid: RunGrid, experiment, date) -> darsia.PrecisionCO2MassAnalysis:
    """Solubility and gas density of CO2 on a run's grid at a frame's pressure and
    temperature and their gradients: the run's atmosphere at the frame's hours
    since the injection start, in the grid's dtype (:func:`run_atmosphere`), as
    our production took it."""
    hours, dtype = experiment.time_since_start(date), grid.rig.pixel_area.img.dtype
    pressure, temperature = run_atmosphere(experiment, hours, dtype)
    pressure_gradient, temperature_gradient = run_atmosphere(
        experiment, hours, dtype, gradient=True
    )
    return darsia.PrecisionCO2MassAnalysis(
        grid.rig.baseline,
        datum_m=grid.datum_m,
        atmospheric_pressure=pressure,
        atmospheric_temperature=temperature,
        atmospheric_pressure_gradient=pressure_gradient,
        atmospheric_temperature_gradient=temperature_gradient,
        y_m=grid.positions.y,
    )


def run_reading(ctx: StepContext, run: str) -> ReadingSettings:
    """A run's colour paths: the ``[reading]`` table with the run's own
    (``[runs.<run>.reading]``) merged over it, which a step reading it records as
    one of its sections."""
    return settings_from_table(ReadingSettings, run_tables(ctx.tables, run)["reading"])


def sand_paths(settings: ReadingSettings) -> dict[int, darsia.ColorPathSignal]:
    """Each sand's colour path and signal function, by facies id."""
    paths = {}
    for name, sand in settings.sands.items():
        colors = [np.asarray(c, float) for c in sand.colors]
        path = darsia.ColorPath(colors=colors, base_color=np.zeros(3), mode="rgb")
        paths[int(name)] = darsia.ColorPathSignal(
            darsia.ColorPathInterpolation(
                path, darsia.ColorMode.RELATIVE, list(sand.values)
            ),
            darsia.PWTransformation(list(sand.supports), list(sand.table)),
        )
    return paths


def range_reading(
    settings: ReadingSettings, ranges: Mapping[int, float]
) -> darsia.SandColorPaths:
    """Each sand read on its own colour path as a share of its measured colour
    range: its factor is its path's scale over the range."""
    factors = {
        int(k): sand.scale / ranges[int(k)] for k, sand in settings.sands.items()
    }
    return darsia.SandColorPaths(sand_paths(settings), factors)


def response_reading(
    ctx: StepContext, run: str, grid: RunGrid
) -> darsia.BeerLambertReading:
    """A run's reading of its sands' dissolved CO2, as our production read it for
    the calibration and the mass: every sand by Beer–Lambert as the reference
    sand, with the measured path ratios (``path-ratios``), on the reference sand's
    background over the run's known pixels outside its shadows
    (:func:`darsia.shadow_light` of the sands read), and with the reference sand's
    factor, its path's scale over its measured colour range (``color-ranges``).

    Raises
    ------
    ValueError
        If the ``[beer_lambert]`` table gives path ratios: they are measured.

    """
    settings = ctx.settings["beer_lambert"]
    if settings.path_ratio is not None:
        raise ValueError("The path ratios are measured: [beer_lambert] names none.")
    ratios = ctx.read("path-ratios", darsia.read_path_ratios).ratio
    settings = replace(settings, path_ratio=ratios)
    reading, reference = run_reading(ctx, run), settings.reference_sand
    baseline, facies, pitch = grid.rig.baseline.img, grid.rig.facies, grid.optical_pitch
    sands = [int(sand) for sand in reading.sands]
    light = darsia.shadow_light(baseline, facies, pitch, sands, ctx.settings["shadow"])
    background = darsia.reference_background(
        baseline, facies, pitch, grid.geometry.known, light, settings
    )
    ranges = ctx.read("color-ranges", darsia.read_color_ranges)
    factor = reading.sands[str(reference)].scale / ranges[reference]
    path = sand_paths(reading)[reference]
    return darsia.BeerLambertReading(path, factor, background, facies, pitch, settings)


def measured_pixels(grid: RunGrid, image: np.ndarray) -> np.ndarray:
    """A frame's pixels whose colour change is measured: in the run's domain and
    finite in the frame's corrected image."""
    return grid.domain & np.isfinite(image).all(axis=-1)


def native_reading(ctx: StepContext, run: str) -> darsia.SandColorPaths:
    """Each sand read on its own colour path with the factor one: the signal our
    production kept on a sand's measured pixels outside the dissolved CO2 it
    read (:func:`read_signal`)."""
    reading = run_reading(ctx, run)
    return darsia.SandColorPaths(
        sand_paths(reading), {int(sand): 1.0 for sand in reading.sands}
    )


def read_signal(
    grid, image, maps, reading, settings, native
) -> tuple[np.ndarray, np.ndarray]:
    """A frame's read signal, as our production read it, and the pixels whose
    colour is measured (finite in the corrected image).

    The dissolved CO2 of the frame's labels before the rules (``maps["before"]``,
    :func:`darsia.free_gas`) is read by ``reading`` and averaged inside each
    sand's dissolved CO2; every other measured pixel of a sand read keeps its own
    path's signal (``native``, :func:`native_reading`). The statistics and the
    mass maps take this signal on the dissolved CO2 of the phase map."""
    support = measured_pixels(grid, image)
    dissolved = support & (maps["before"] == darsia.PhaseLabel.DISSOLVED)
    baseline = grid.rig.baseline.img
    signal = reading(image, baseline, support, dissolved, grid.facies)
    optical = darsia.optical_grid(
        grid.rig.facies, grid.optical_pitch, baseline.dtype.type
    )
    rev = darsia.REV(size=settings.mean_mm / 1000, img=optical)
    sands = {sand: grid.facies == sand for sand in reading.sands}
    signal = darsia.sand_mean(signal, dissolved, sands, rev)
    sand_pixels = support & np.isin(grid.facies, reading.sands)
    own = native(image, baseline, support, sand_pixels, grid.facies)
    return np.where(dissolved, signal, own).astype(signal.dtype), support


def _phase_model(ctx: StepContext) -> None:
    """Store the series' phase-model folder."""
    source = Path(ctx.settings["phase_model"].folder)
    ctx.publish(lambda folder: shutil.copytree(source, folder, dirs_exist_ok=True))


def _model_files(tables: Mapping, project, folders) -> list[str]:
    """The series' phase-model folder."""
    return [str(tables["phase_model"]["folder"])]


def _overflow_workbook(tables: Mapping) -> Path | tuple[Path, str] | None:
    """A run's overflow protocol, ``[protocols] overflow`` of its tables: a file,
    or a workbook and its sheet (``[path, sheet]``), as the other protocols are
    given."""
    value = tables.get("protocols", {}).get("overflow")
    if isinstance(value, (list, tuple)):
        return Path(value[0]), str(value[1])
    return None if value is None else Path(value)


def _overflow_file(tables: Mapping) -> str | None:
    """The file of a run's overflow protocol (:func:`_overflow_workbook`): its
    workbook for a sheet."""
    source = _overflow_workbook(tables)
    if isinstance(source, tuple):
        source = source[0]
    return None if source is None else str(source)


def _sand_map_identity(tables: Mapping) -> str:
    """A run's sand map as a donor shares it: its drawing, as configured."""
    return str(tables.get("labeling", {}).get("sand_map", {}).get("drawing", ""))


def _displaced_files(tables: Mapping, project, folders) -> list[str]:
    """The run's protocols, its overflow workbook and its sand map's drawing."""
    workbook = _overflow_file(tables)
    drawing = tables.get("labeling", {}).get("sand_map", {}).get("drawing")
    found = protocol_files(tables, project, folders)
    return found + [str(path) for path in (workbook, drawing) if path is not None]


def _displaced_volume(ctx: StepContext) -> None:
    """The run's displaced volume from its scale readings, with its approved
    evaporation, and what a recipient compares it by."""
    run, settings = ctx.task.run, ctx.settings["overflow"]
    tables = run_tables(ctx.tables, run)
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    start = experiment.experiment_start
    path = _overflow_workbook(tables)
    volume = proposal = rate = None
    if path is not None:
        protocol = darsia.OverflowProtocol(path)
        # The readings count from the injection start in their own clock, the
        # sheet's, as our production read each reviewed curve; the injection
        # protocol's start is in the photographs' clock.
        origin = start if protocol.injection_start is None else protocol.injection_start
        times, weights = protocol.readings(origin, list(settings.statuses))
        volume = darsia.displaced_volume(times, weights, settings)
        if settings.evaporation is not None:
            proposal = darsia.proposed_evaporation(volume, settings.evaporation)
        rate = darsia.approved_evaporation(run, proposal, settings.evaporation)
        if rate is not None:
            volume = darsia.displaced_volume(times, weights, settings, rate)
    pressure = experiment.pressure_temperature_protocol.df
    steps = experiment.injection_protocol.df
    steps = steps[steps["rate_kg_s"] > 0]
    record = darsia.OverflowRun(
        sand_map=_sand_map_identity(tables),
        recipe=dict(tables.get("recipe", {})),
        pressure_h=np.asarray(
            [experiment.time_since_start(date) for date in pressure["datetime"]]
        ),
        pressure_bar=np.asarray(pressure["pressure_bar"], np.float64),
        injection_h=(
            experiment.time_since_start(steps["start"].min()),
            experiment.time_since_start(steps["end"].max()),
        ),
        volume=volume,
    )
    ctx.publish(
        lambda folder: darsia.write_overflow_run(folder, record, proposal, rate)
    )


def run_atmosphere(
    experiment, hours: float, dtype, gradient: bool = False
) -> tuple[float, float]:
    """A run's atmospheric pressure [bar] and temperature [°C] at a time [h since
    its injection start], or with ``gradient`` their upward gradients [bar/m] and
    [°C/m]: its pressure-temperature record interpolated over the record's hours
    since the start, those in the grid's dtype
    (:func:`darsia.interpolated_atmosphere`), as our production did."""
    record = experiment.pressure_temperature_protocol.df
    times = [experiment.time_since_start(date) for date in record["datetime"]]
    columns = ("pressure_bar", "temperature_celsius")
    if gradient:
        columns = ("pressure_gradient_bar", "temperature_gradient_celsius")
    return darsia.interpolated_atmosphere(
        times, *(record[column] for column in columns), hours, dtype
    )


def _gas_density(experiment, hours: float, dtype) -> float:
    """The gas density [kg/m3] at a run's atmosphere at a time [h since its
    injection start] (:func:`run_atmosphere`), at no hydrostatic head."""
    pressure, temperature = run_atmosphere(experiment, hours, dtype)
    return darsia.atmospheric_gas_density(pressure, temperature, dtype)


def _overflow(ctx: StepContext) -> None:
    """Each frame's free-gas volume: from the run's own readings, else from the
    most similar fit run with readings of the run's calibration; and, if that donor
    is left out of a fold, from the next one. The readings are interpolated in the
    grid's dtype, as our production interpolated them in the job's precision; the
    stored displaced volumes keep theirs."""
    run, frames = ctx.task.run, ctx.task.frames
    settings = ctx.settings["overflow"]
    fits = ctx.settings["calibration_runs"].group(run)
    runs = {
        name: ctx.read("displaced-volume", darsia.read_overflow_run, run=name)
        for name in ctx.selection.frames
    }
    dtype = ctx.read("rig", read_rig).pixel_area.img.dtype
    experiments = {name: None for name in runs}

    def experiment_of(name):
        if experiments[name] is None:
            experiments[name] = make_experiment(run_settings(ctx, name, EXPERIMENT))
        return experiments[name]

    own = experiment_of(run)
    dates = [own.get_datetime(ctx.photo(run, frame)) for frame in frames]
    hours = [own.time_since_start(date) for date in dates]
    mass = [own.injection_protocol.injected_mass(time=h) * 1000 for h in hours]
    density = [_gas_density(own, h, dtype) for h in hours]
    candidates = {name: runs[name] for name in runs if name in fits.fit or name == run}
    ranked = darsia.ranked_donors(run, candidates, settings)

    def state(donor: str, h: float) -> tuple[float, float]:
        other = experiment_of(donor)
        mass_g = other.injection_protocol.injected_mass(time=h) * 1000
        return mass_g, _gas_density(other, h, dtype)

    def readings(name: str) -> darsia.DisplacedVolume | None:
        volume = runs[name].volume
        return None if volume is None else volume.astype(dtype)

    def volumes(donor: str | None) -> dict[str, darsia.FrameVolume]:
        found = darsia.frame_volumes(
            readings(run),
            hours,
            None if donor is None else (donor, readings(donor)),
            mass,
            density,
            None if donor is None else partial(state, donor),
        )
        return {frame_key(frame): v for frame, v in zip(frames, found)}

    primary = volumes(ranked[0] if ranked else None)
    held_out = {}
    if ranked and ranked[0] in fits.folds:
        if any(v.source != darsia.MEASURED for v in primary.values()):
            held_out[ranked[0]] = volumes(ranked[1] if len(ranked) > 1 else None)
    volume = runs[run].volume
    reasons = [] if volume is None else list(volume.reasons)
    ctx.publish(
        lambda folder: darsia.write_frame_volumes(folder, primary, reasons, held_out)
    )


def previous_acquisition(photo: Path, sessions: Sequence[Path]) -> Path | None:
    """The run's acquisition just before a photo: the photo before it in its
    session's folder, in the order of the file names, or for a session's first
    photo the last photo of the run's previous session; None for the run's first
    photo. Our production read this acquisition, selected or not, as each frame's
    previous image.

    Parameters
    ----------
    photo : Path
        A frame's photo, in one of the run's session folders.
    sessions : sequence of Path
        The run's session folders, in time order (the order of the run's
        sessions in the configuration).

    Returns
    -------
    Path or None
        The previous acquisition's photo.

    """
    folders = [Path(folder) for folder in sessions]
    photo = Path(photo)
    position = folders.index(photo.parent)
    for folder in reversed(folders[: position + 1]):
        names = sorted(
            p.name
            for p in folder.iterdir()
            if p.suffix.lower() == photo.suffix.lower()
            and (folder != photo.parent or p.name < photo.name)
        )
        if names:
            return folder / names[-1]
    return None


class _Probabilities(Sequence):
    """A run's phase probabilities by frame, computed when asked for, a window of
    them kept; each frame's previous image is the run's acquisition just before
    it (:func:`previous_acquisition`), read through the rig's corrections as the
    corrected-image step reads a photo."""

    def __init__(self, ctx, model, grid, frames, kept: int, rig: Rig) -> None:
        self.ctx, self.model, self.grid, self.frames = ctx, model, grid, frames
        self.rig = rig
        self.image = lru_cache(maxsize=kept + 1)(self._image)
        self.probabilities = lru_cache(maxsize=kept)(self._probabilities)

    def _image(self, index: int) -> np.ndarray:
        return self.ctx.read(
            "corrected-image", read_image, frame=self.frames[index]
        ).img

    def _previous(self, index: int) -> np.ndarray | None:
        """The corrected image of a frame's previous acquisition, or None."""
        run = self.ctx.task.run
        photo = self.ctx.photo(run, self.frames[index])
        sessions = list(self.ctx.locations.sessions[run].values())
        previous = previous_acquisition(photo, sessions)
        if previous is None:
            return None
        return corrected_photo(self.rig, previous)[0].img

    def _probabilities(self, index: int):
        return darsia.phase_probabilities(
            self.model,
            self.image(index),
            self.grid.rig.baseline.img,
            self._previous(index),
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
    variants = (None, *ctx.read("overflow", darsia.held_out_runs))
    start = next((k for k, frame in enumerate(frames) if not ctx.done(frame)), None)
    if start is None:
        return
    kept = 2 * rules.neighbours_per_side + 2
    rig = ctx.read("rig", read_rig)
    rig.load_experiment(experiment)
    probabilities = _Probabilities(ctx, model, grid, frames, kept, rig)

    def maps(held_out):
        volumes = ctx.read("overflow", lambda f: darsia.read_frame_volumes(f, held_out))
        return darsia.free_gas(
            probabilities,
            dates,
            grid.rig.facies,
            grid.domain,
            grid.pixel_area_mm2,
            grid.permeability,
            experiment.injection_protocol,
            rules,
            grid.geometry.pore_volume_m3,
            grid.geometry.swi,
            [volumes[frame_key(frame)].volume_m3 for frame in frames],
            settings["gas_allocation"],
            grid.chart,
            gas_threshold=model.gas_threshold,
            start=start,
            anchor_saturation=lambda k: ctx.own(
                lambda f: darsia.read_phase_maps(f, held_out), frame=frames[k]
            )["saturation"],
        )

    for frame, *found in zip(frames[start:], *(maps(v) for v in variants)):

        def write(folder, found=found):
            for held_out, arrays in zip(variants, found):
                darsia.write_phase_maps(folder, *arrays, held_out=held_out)

        ctx.publish(write, frame=frame)


class _RunFrames(Sequence):
    """A run's selected frames as a measurement reads them, by index: each one's
    corrected image, its phase and rule maps, and its pixels whose colour is
    measured (in the domain and finite in the corrected image)."""

    def __init__(self, ctx: StepContext, run: str, grid: RunGrid) -> None:
        self.ctx, self.run, self.grid = ctx, run, grid
        self.frames = ctx.selection.frames[run]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        frame, run = self.frames[int(index)], self.run
        image = self.ctx.read("corrected-image", read_image, run=run, frame=frame).img
        maps = self.ctx.read("phase-maps", darsia.read_phase_maps, run=run, frame=frame)
        support = measured_pixels(self.grid, image)
        return image, maps["labels"], maps["rules"], support


def _color_ranges(ctx: StepContext) -> None:
    """Each sand's colour range on the configured runs, each run read on its own
    colour paths."""
    settings, runs = ctx.settings, []
    for run in settings["response"].range_runs:
        grid = run_grid(ctx.read("rig", read_rig, run=run), settings["material"])
        rig, paths = grid.rig, sand_paths(run_reading(ctx, run))
        frames = _RunFrames(ctx, run, grid)
        runs.append((frames, rig.baseline.img, rig.facies, grid.optical_pitch, paths))
    ranges = darsia.color_ranges(runs, settings["color_ranges"])
    ctx.publish(lambda folder: darsia.write_color_ranges(folder, ranges))


def _path_ratios(ctx: StepContext) -> None:
    """The sands' path ratios on the configured run, read on the colour ranges
    alone in each sand's ordinary light, up to the concentration curve's top at
    the configured gain."""
    settings = ctx.settings
    response, statistics = settings["response"], settings["run_statistics"]
    run = response.ratio_run
    grid = run_grid(ctx.read("rig", read_rig, run=run), settings["material"])
    ranges = ctx.read("color-ranges", darsia.read_color_ranges)
    reading = range_reading(run_reading(ctx, run), ranges)
    baseline, facies, pitch = grid.rig.baseline.img, grid.rig.facies, grid.optical_pitch
    light = darsia.shadow_light(
        baseline, facies, pitch, reading.sands, settings["shadow"]
    )
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    hours = [
        experiment.time_since_start(experiment.get_datetime(ctx.photo(run, frame)))
        for frame in ctx.selection.frames[run]
    ]
    span = statistics.saturation_signal - statistics.zero_signal
    ceiling = statistics.zero_signal + span / response.ratio_gain
    found = darsia.path_ratios(
        _RunFrames(ctx, run, grid),
        hours,
        baseline,
        facies,
        pitch,
        grid.geometry.known,
        reading,
        light,
        settings["beer_lambert"].reference_sand,
        ceiling,
        settings["path_ratios"],
    )
    ctx.publish(lambda folder: darsia.write_path_ratios(folder, found))


def _run_statistics(ctx: StepContext) -> None:
    """Each frame's sand histograms, gas statistics and column estimate, as the
    fit reads them; for a fold leaving out the donor of the run's free gas, also
    from the frames' maps without it."""
    run, frames, settings = ctx.task.run, ctx.task.frames, ctx.settings
    statistics_settings = settings["run_statistics"]
    grid = run_grid(ctx.read("rig", read_rig), settings["material"])
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    protocol = experiment.injection_protocol
    variants = (None, *ctx.read("overflow", darsia.held_out_runs))
    volumes = {
        held_out: ctx.read("overflow", lambda f: darsia.read_frame_volumes(f, held_out))
        for held_out in variants
    }
    reading = response_reading(ctx, run, grid)
    native = native_reading(ctx, run)
    rows = {held_out: [] for held_out in variants}
    for index, frame in enumerate(frames):
        image = ctx.read("corrected-image", read_image, frame=frame).img
        date = experiment.get_datetime(ctx.photo(run, frame))
        co2 = co2_state(grid, experiment, date)
        hours = experiment.time_since_start(date)
        for held_out in variants:
            maps = ctx.read(
                "phase-maps", lambda f: darsia.read_phase_maps(f, held_out), frame=frame
            )
            capacity = darsia.dissolved_capacity(
                grid.geometry.pore_volume_m3, co2.solubility_co2, maps["saturation"]
            )
            signal, measured = read_signal(
                grid, image, maps, reading, statistics_settings, native
            )
            dissolved = (
                measured
                & (maps["labels"] == darsia.PhaseLabel.DISSOLVED)
                & ~darsia.model_gas(maps["labels"], maps["rules"])
            )
            histograms = darsia.signal_histograms(
                signal,
                dissolved,
                grid.facies,
                grid.geometry.counted,
                capacity,
                reading.sands,
                statistics_settings,
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
            unknown = darsia.unknown_statistics(
                darsia.relative_signal(signal, statistics_settings),
                maps["labels"],
                maps["rules"],
                measured,
                grid.geometry,
                co2,
                settings["unknown_mass"],
            )
            volume = volumes[held_out][frame_key(frame)]
            rows[held_out].append(
                darsia.CalibrationFrame(
                    run=run,
                    index=index,
                    elapsed_h=hours,
                    target_h=hours,
                    injected_g=protocol.injected_mass(time=hours) * 1000,
                    dissolved=histograms,
                    gas=gas,
                    column=unknown.column_estimate(),
                    overflow_m3=volume.volume_m3,
                    measured=volume.source == darsia.MEASURED,
                )
            )
    dtype = grid.rig.pixel_area.img.dtype
    primary = rows.pop(None)
    ctx.publish(lambda folder: darsia.write_frames(folder, run, primary, dtype, rows))


def _calibration_parts(selection: Selection, tables: Mapping) -> tuple[str, ...]:
    """The global fit, then one fold per run left out; with groups, each group's
    fit and folds, of the groups with a selected run (:meth:`FitRuns.parts`)."""
    if "calibration_runs" not in tables:
        return ("fit",)
    return tuple(fit_runs(tables).parts(list(selection.frames)))


def _required_runs(tables: Mapping, selected: Sequence[str]) -> list[str]:
    """The runs a calibration job must select: the fit runs and folds of each
    calibration it makes (of each group with a selected run)."""
    if "calibration_runs" not in tables:
        return []
    parts = fit_runs(tables).parts(list(selected)).values()
    return [run for group, _ in parts for run in (*group.fit, *group.folds)]


def _donor_runs(tables: Mapping, selected: Sequence[str]) -> list[str]:
    """The runs an overflow job must select: the fit runs of each selected run's
    calibration, among which its donors are ranked."""
    fits = fit_runs(tables)
    return [donor for run in selected for donor in fits.group(run).fit]


def _run_part(run: str, tables: Mapping) -> str:
    """The fit that a run's mass maps read: its group's, or the one fit."""
    name = fit_runs(tables).group_of(run) if "calibration_runs" in tables else None
    return "fit" if name is None else f"fit-{name}"


def _calibration(ctx: StepContext) -> None:
    """The fit across the fit runs, or one fold leaving a run out, over the runs of
    the part's group; a fold reads each run's frames without the held-out run's
    estimated free gas."""
    settings = ctx.settings
    runs, material = settings["calibration_runs"], settings["material"]
    group, held = runs.parts(list(ctx.selection.frames))[ctx.task.part]
    members = [r for r in ctx.selection.frames if not group.runs or r in group.runs]
    frames = [
        frame
        for run in members
        for frame in ctx.read(
            "run-statistics", lambda f: darsia.read_frames(f, held), run=run
        )
    ]
    swi = {int(k): float(v) for k, v in material.swi.items()}
    measured = [int(k) for k in material.measured_swi]
    arguments = (frames, list(group.fit), settings["calibration_fit"], swi, measured)
    predictions = []
    if held is None:
        calibration = darsia.fit_calibration(*arguments)
    else:
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
DISPLACED_VOLUME = Step(
    "displaced-volume",
    {},
    Order.RUN,
    Memory(),
    _displaced_volume,
    {"overflow": darsia.OverflowSettings},
    sections=(*EXPERIMENT, "recipe"),
    files=_displaced_files,
)
"""Each run's displaced volume from its overflow, as a donor gives it."""
OVERFLOW = Step(
    "overflow",
    {"displaced-volume": Scope.ALL, "rig": Scope.OWN},
    Order.RUN,
    Memory(),
    _overflow,
    {"overflow": darsia.OverflowSettings, "calibration_runs": FitRuns},
    sections=EXPERIMENT,
    files=protocol_files,
    required=_donor_runs,
)
"""Each run's free-gas volume per frame, its own or a donor's."""
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
COLOR_RANGES = Step(
    "color-ranges",
    {"rig": Scope.ALL, "corrected-image": Scope.ALL, "phase-maps": Scope.ALL},
    Order.PARTS,
    Memory(grid=12.0),
    _color_ranges,
    {
        "color_ranges": darsia.ColorRangeSettings,
        "response": ResponseRuns,
        "material": MaterialSettings,
    },
    sections=READING,
    runs=lambda tables: tables["response"]["range_runs"],
)
"""Each sand's colour range, measured on the configured runs."""
PATH_RATIOS = Step(
    "path-ratios",
    {
        "rig": Scope.ALL,
        "corrected-image": Scope.ALL,
        "phase-maps": Scope.ALL,
        "color-ranges": Scope.ONE,
    },
    Order.PARTS,
    Memory(grid=24.0),
    _path_ratios,
    {
        "path_ratios": darsia.PathRatioSettings,
        "shadow": darsia.ShadowSettings,
        "beer_lambert": darsia.BeerLambertSettings,
        "response": ResponseRuns,
        "run_statistics": darsia.RunStatisticsSettings,
        "material": MaterialSettings,
    },
    sections=(*READING, *EXPERIMENT),
    files=protocol_files,
    runs=lambda tables: [tables["response"]["ratio_run"]],
)
"""The sands' path ratios, measured on the configured run."""
RESPONSE = {
    "beer_lambert": darsia.BeerLambertSettings,
    "shadow": darsia.ShadowSettings,
}
"""The settings of the sands' Beer–Lambert response (:func:`response_reading`)."""
RUN_STATISTICS = Step(
    "run-statistics",
    {
        "rig": Scope.OWN,
        "corrected-image": Scope.OWN,
        "phase-maps": Scope.OWN,
        "overflow": Scope.OWN,
        "color-ranges": Scope.ONE,
        "path-ratios": Scope.ONE,
    },
    Order.RUN,
    Memory(grid=24.0),
    _run_statistics,
    {
        "run_statistics": darsia.RunStatisticsSettings,
        "material": MaterialSettings,
        "unknown_mass": darsia.UnknownMassSettings,
        **RESPONSE,
    },
    sections=(*READING, *EXPERIMENT),
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
    run_part=_run_part,
    required=_required_runs,
)
"""The calibration fit and its folds (R6), or each group's."""
STEPS = (
    PHASE_MODEL,
    DISPLACED_VOLUME,
    OVERFLOW,
    PHASE_MAPS,
    COLOR_RANGES,
    PATH_RATIOS,
    RUN_STATISTICS,
    CALIBRATION,
)
"""The Calibration's steps."""
