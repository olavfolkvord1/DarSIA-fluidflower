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
  its measured colour range. Independent frames run in chunks, with one stored
  result per frame, so an interrupted chunk resumes at its missing frames.
- ``calibration`` (R6) fits the gain and the dilution across the fit runs, and
  each fold leaves one run out, reading every run's frames without its
  estimate. With calibration groups (:class:`FitRuns`), e.g. runs photographed
  with other camera settings, each group has its own fit and folds over its own
  runs, and a run's mass maps read its group's fit.

"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache, partial
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len

import darsia
from darsia.jobs.step import (
    Each,
    Memory,
    Order,
    Scope,
    Selection,
    Step,
    StepContext,
    Term,
    run_tables,
    table,
)
from darsia.multiphase.gas_allocation import injection_ports
from darsia.multiphase.gas_model_cache import GasModelCache
from darsia.multiphase.gas_parameters import parameters_at_depth
from darsia.multiphase.gas_zone_cache import GasZoneCache
from darsia.multi_image_analysis.calibration_fit import (
    calibration_memory,
    statistics_memory,
    stored_statistics_memory,
)
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.calibration.model_cache import model_cache_identity
from darsia.presets.workflows.config.expert_knowledge import ExpertKnowledgeConfig
from darsia.presets.workflows.config.protocols import InjectionProtocolConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    corrected_photo,
    grid_canvas,
    grid_mask,
    grid_model,
    protocol_files,
    read_image,
    read_rig,
    run_settings,
)

READING = ("reading",)
"""The section of the sands' colour paths, which a step reads per run
(:func:`run_reading`)."""
EXPERT = ("expert_knowledge", "roi")
"""The sections of a run's expert knowledge: where CO2 and free gas may be
(``[expert_knowledge]``) and the named regions it names (``[[roi]]``), which the
phase maps read per run (:func:`expert_limits`)."""


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
    """The pixels whose phase is decided and whose colour change is measured: the
    reporting domain's pixels of a material with porosity, or every pixel."""
    geometry: darsia.RunGeometry
    """The geometry over the reporting domain (:func:`darsia.run_geometry`)."""
    chart: np.ndarray | None
    pixel_mm: float
    datum_m: float
    optical_pitch: tuple[float, float]
    """The pitch [m] on the sand plane (:func:`darsia.optical_pitch_m` over the
    reporting domain); with a measurement canvas, with its counts, on which the
    optical grid is sized (:class:`darsia.OpticalPitch`)."""
    positions: darsia.GridPositions
    """Each pixel's position on the rig and the grid's pitch, by the rig's measured
    model of its grid (:func:`darsia.grid_positions`), in the grid's precision."""
    pixel_area_mm2: np.ndarray
    """Each pixel's area [mm2], the rig's pixel areas in double precision times a
    million, as our production's phase rules took them."""
    reporting: np.ndarray
    """The pixels the run reports on (``Rig.reporting_domain``), or every pixel."""


def run_grid(
    rig: Rig, material: MaterialSettings, pitch_stride: int | None = None
) -> RunGrid:
    """A run's grid: its facies and their permeability, the domains, the geometry
    with each facies' Swi, the colour checker, the pixel size, the datum of the
    hydrostatic head, the pitch on the sand plane, and each pixel's position on the
    rig by the rig's measured model of its grid, with the grid's pitch over the
    rig's measurement canvas where a pitch stride is given
    (:func:`darsia.grid_positions`; the phase maps take it).

    With the rig's reporting domain (``Rig.reporting_domain``), the run reports on
    it, as our production's mass did: the geometry and the optical pitch are taken
    over it, and the domain whose phase is decided is its pixels of a material with
    porosity, our production's analysis domain, and the colour checker is its
    pixels of the chart. Without one both domains are every pixel.
    With the rig's measurement canvas, the optical pitch is the voxel size of the
    optical grid sized on the canvas' pixels, as production sized it.

    Depth, porosity, Swi and the measured positions take the precision of the
    rig's pixel areas, the run grid's.

    """
    facies = rig.facies.img
    dtype = rig.pixel_area.img.dtype
    permeability = {
        int(f): float(rig.permeability.img[facies == f].flat[0])
        for f in np.unique(facies)
    }
    reporting = rig.reporting_domain
    if reporting is None:
        reporting = domain = np.isin(facies, list(permeability))
    else:
        domain = reporting & (rig.porosity.img > 0)
    swi = np.full(facies.shape, np.nan, dtype)
    for name, value in material.swi.items():
        swi[facies == int(name)] = value
    chart = None
    if material.chart_label is not None:
        chart = rig.labels.img == material.chart_label
        if rig.reporting_domain is not None:  # the chart the run reports on
            chart &= reporting
    positions = darsia.grid_positions(
        rig.facies, rig.grid_to_rig, dtype, rig.measurement_canvas, pitch_stride
    )
    geometry = darsia.run_geometry(
        rig.facies,
        rig.pixel_area.img,
        rig.depth.img.astype(dtype),
        rig.porosity.img.astype(dtype),
        swi,
        reporting,
        chart,
        positions,
    )
    measured = None if rig.grid_to_rig is None else positions
    optical = darsia.optical_pitch_m(rig.pixel_area, reporting, measured)
    if rig.measurement_canvas is not None:
        optical = darsia.OpticalPitch(optical, rig.measurement_canvas.shape)
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
        optical,
        positions,
        np.asarray(rig.pixel_area.img, np.float64) * 1e6,
        reporting,
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
        baseline,
        facies,
        pitch,
        grid.geometry.known,
        light,
        settings,
        canvas=grid.rig.measurement_canvas,
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


def _model_copy_bytes(tables: Mapping, run: str | None, context) -> int:
    """The copy buffer and directory-entry records retained by recursive copytree."""
    path = Path(tables["phase_model"]["folder"])
    if context.project is not None:
        path = context.project / path
    pointer = np.dtype(np.intp).itemsize
    entries = sum(sys.getsizeof(str(item)) + sys.getsizeof(item) + pointer for item in path.rglob("*"))
    return 2 * shutil.COPY_BUFSIZE + entries


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


@lru_cache(maxsize=128)
def _protocol_table_bytes(path: Path, size: int, stamp: int) -> int:
    """Bound a parsed protocol from table bytes and its decoder's live rows."""
    ref = np.dtype(np.intp).itemsize
    objects = max(sys.getsizeof(0.0), sys.getsizeof(pd.Timestamp(0)))
    per_cell = objects + 3 * ref
    if is_zipfile(path):
        with ZipFile(path) as archive:
            sheets = [entry.file_size for entry in archive.infolist()
                      if entry.filename.startswith("xl/worksheets/")
                      and not entry.filename.endswith(".rels")]
            shared = sum(entry.file_size for entry in archive.infolist()
                         if entry.filename in ("xl/sharedStrings.xml", "xl/styles.xml"))
        # A cell needs at least an opening tag and a delimiter. Strings and
        # tokenizer/XML buffers are counted separately from pandas' row arrays.
        expanded = max(sheets, default=0)
        return (expanded // 4) * per_cell + 5 * (expanded + shared) + size
    # A CSV date, number or text occupies at least one byte and a delimiter.
    # Tabular conversion retains both the temporary rows and numeric columns.
    return (size // 2) * per_cell + 5 * size


def _displaced_table_bytes(tables: Mapping, run: str | None, context) -> int:
    """The protocol frames, scale series and largest in-flight table decoder."""
    selected = run_tables(tables, run) if run is not None else tables
    if context.project is None:
        return 0
    protocols = selected.get("protocols", {})
    values = []
    for name in ("imaging", "blacklist", "injection", "pressure_temperature", "overflow"):
        value = protocols.get(name)
        values.extend(value.values() if isinstance(value, Mapping) else (value,))
    files = []
    for value in values:
        if value is None:
            continue
        location = Path(value[0] if isinstance(value, (list, tuple)) else value)
        location = context.project / location
        stat = location.stat()
        files.append(_protocol_table_bytes(location, stat.st_size, stat.st_mtime_ns))
    return sum(files)


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


def previous_acquisition(
    photo: Path,
    sessions: Sequence[Path],
    excluded: Callable[[Path], bool] | None = None,
) -> Path | None:
    """The run's acquisition just before a photo: the photo before it in its
    session's folder, in the order of the file names, or for a session's first
    photo the last photo of the run's previous session; None for the run's first
    photo. A photo the configuration excludes is no acquisition and is passed
    over. Our production read this acquisition, selected or not, as each frame's
    previous image, its history without the light screening's exclusions.

    Parameters
    ----------
    photo : Path
        A frame's photo, in one of the run's session folders.
    sessions : sequence of Path
        The run's session folders, in time order (the order of the run's
        sessions in the configuration).
    excluded : callable, optional
        Whether the configuration excludes a photo, e.g. the experiment's
        blacklist (:meth:`ProtocolledExperiment.is_blacklisted`); by default
        no photo is excluded.

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
        for name in reversed(names):
            if excluded is None or not excluded(folder / name):
                return folder / name
    return None


def _within(image: np.ndarray | None, support: np.ndarray | None) -> np.ndarray | None:
    """An image with its pixels outside a support unknown (NaN), in a copy, as our
    production's phase step read its images; the image itself without a support."""
    if image is None or support is None:
        return image
    within = np.array(image, copy=True)
    within[~support] = np.nan
    return within


def _phase_support(tables: Mapping) -> str | None:
    """The phase support a run's tables name (``[labeling.sand_map]
    phase_support``), or None."""
    return tables.get("labeling", {}).get("sand_map", {}).get("phase_support")


def expert_limits(
    tables: Mapping, rig: Rig
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Where a run's expert knowledge lets CO2, dissolved or as gas, and free gas
    be on its grid.

    The run's ``[expert_knowledge]`` table and its ``[[roi]]`` regions, read as the
    analysis reads them (:class:`ExpertKnowledgeConfig`, :class:`RoiRegistry`),
    become the masks of ``concentration_aq`` and ``saturation_g`` of an
    :class:`ExpertKnowledgeAdapter` on the rig's labels and facies; a region named
    in a ``no_`` list is excluded, a positive list keeps only its regions.

    Parameters
    ----------
    tables : Mapping
        The run's configuration tables (:func:`run_tables`).
    rig : Rig
        The run's rig, whose facies give the grid.

    Returns
    -------
    keep_co2, keep_gas : array or None
        Where CO2 and where free gas may be; None where nothing limits it, e.g.
        without the table.

    Raises
    ------
    ValueError
        If the table names phase maps as dynamic expert knowledge, which the phase
        maps cannot read, since they are what it limits.

    """
    section = tables.get("expert_knowledge")
    if not section:
        return None, None
    registry = RoiRegistry.from_entries(tables.get("roi", []))
    config = ExpertKnowledgeConfig.from_table(section, registry)
    if config.phase_maps is not None:
        raise ValueError(
            "The phase maps cannot be limited by phase maps: [expert_knowledge] "
            "phase_maps is the analysis' dynamic expert knowledge."
        )
    adapter = ExpertKnowledgeAdapter.from_config(
        config, registry, labels_image=rig.labels, facies_image=rig.facies
    )
    return (
        adapter.mask_for(rig.facies, "concentration_aq"),
        adapter.mask_for(rig.facies, "saturation_g"),
    )


def _expert_limited(tables: Mapping, run: str | None, context=None) -> int:
    """One where a run's expert knowledge names any region, whose two masks the
    phase maps hold; else none."""
    if run is None:
        return 0
    section = run_tables(tables, run).get("expert_knowledge") or {}
    keys = (
        "saturation_g",
        "concentration_aq",
        "no_saturation_g",
        "no_concentration_aq",
    )
    return int(any(section.get(key) for key in keys))


def _phase_map_files(tables: Mapping, project, folders) -> list[str]:
    """The run's protocols and its phase support, if it names one."""
    support = _phase_support(tables)
    found = protocol_files(tables, project, folders)
    return found + ([str(support)] if support is not None else [])


def window_frames(rules: darsia.PhaseRuleSettings) -> int:
    """The frames the phase maps keep at once: the rules' neighbours on each side
    of a frame, the frame itself and the one after it."""
    return 2 * rules.neighbours_per_side + 2


def _window(tables: Mapping, run: str | None, context=None) -> int:
    """The phase maps' window by the job's rules (:func:`window_frames`)."""
    tables = run_tables(tables, run) if run is not None else tables
    rules = settings_from_table(darsia.PhaseRuleSettings, table(tables, "phase_rules"))
    return window_frames(rules)


def _variants(tables: Mapping, run: str | None, context=None) -> int:
    """The held-out variants of a run's maps at most: one where its calibration
    group has folds, of which its first donor may be held out (:func:`_overflow`)."""
    if run is None or "calibration_runs" not in tables:
        return 0
    return int(bool(fit_runs(tables).group(run).folds))


def _supported(tables: Mapping, run: str | None, context=None) -> int:
    """One where a run names a phase support, whose masked images the phase maps
    hold beside the read ones; else none."""
    return int(run is not None and _phase_support(run_tables(tables, run)) is not None)


def reading_sands(tables: Mapping, run: str | None, context=None) -> int:
    """The sand masks read_signal and sand_mean keep simultaneously."""
    tables = run_tables(tables, run) if run is not None else tables
    return len(tables.get("reading", {}).get("sands", {}))


def _range_rigs(tables: Mapping, run: str | None, context=None) -> int:
    """Additional rigs retained in the list passed to color_ranges."""
    return max(0, len(tables.get("response", {}).get("range_runs", ())) - 1)


def _statistics_entries(tables: Mapping, run: str | None, context) -> int:
    """Stored histogram and column-group values for one frame and its variants.

    A sand histogram has relative/weight arrays of at most intervals + 1
    entries. Column estimates have relative/ratio/weight arrays of at most
    (intervals + 1)**2 groups; their own-signal component is zero. Neither the
    combined disjoint sand histograms nor the column groups can exceed pixels.
    """
    own = run_tables(tables, run) if run is not None else tables
    histograms = int(own["run_statistics"]["intervals"]) + 1
    columns = int(own["unknown_mass"]["intervals"]) + 1
    pixels = context.grid_pixels
    # Histograms cover measured dissolved pixels; column groups cover missing
    # measurements. They partition pixels, so reserve the larger three-value
    # column groups first and histogram groups only in the remaining pixels.
    occupied = min(columns**2, pixels)
    entries = 3 * occupied
    entries += 2 * min(histograms * reading_sands(tables, run), pixels - occupied)
    return entries * (1 + _variants(tables, run))


def _statistics_bytes(tables: Mapping, run: str | None, context) -> int:
    """One frame's retained statistics and Python records for the streaming writer.

    Histogram and column arrays follow capacity's dtype: a float64 geometry or
    CO2 field can promote them even with float32 images. The source arrays stay
    alive while _rows builds plain lists, and dataclasses.asdict copies a frame.
    Reserve both arrays at least at float64 width, plus the Python floats and
    list pointers. _array may also cast one array at a time; that array has at
    most one entry per pixel, without a sand or variant multiplier. Streaming
    JSON adds only traversal/chunk/TextIO buffers, covered by the process base.
    """
    array_bytes = max(context.itemsize, np.dtype(np.float64).itemsize)
    value_bytes = 2 * array_bytes + sys.getsizeof(0.0) + np.dtype(np.intp).itemsize
    return (
        _statistics_entries(tables, run, context) * value_bytes
        + context.grid_pixels * array_bytes
    )


def _gas_ports(own: Mapping, context) -> dict:
    """Read measured injecting ports and periods from one run's protocol."""
    config = InjectionProtocolConfig()
    config.load_injection(own["protocols"])
    source = config.injection
    if source is None:
        raise ValueError("The phase maps need an injection protocol for their memory.")
    path = source[0] if isinstance(source, tuple) else source
    if not path.is_absolute():
        if context.project is None:
            raise ValueError("Relative injection protocols need the job's project.")
        path = Path(context.project) / path
    source = (path, source[1]) if isinstance(source, tuple) else path
    protocol = darsia.InjectionProtocol(
        source, reference_density=config.injection_reference_density
    )
    return injection_ports(protocol)


def _injection_map_bytes(tables: Mapping, run: str | None, context) -> int:
    """Protocol and cache arrays retained across the prefix and allocation passes.

    Three model frames belong to the LRU; one prior frame can remain with an
    allocation while a skipped-neighbour miss evicts it. Two zone payloads can
    coexist across a read miss or while the next prefix is built. Each zone has
    a bool contour and at most one bool path/float32 blend per port.

    The builder holds each active port's CO2 mask and working maximum, the phase
    rules' port masks/reduction, twelve grid masks and the last float64 distance
    and normalized distance. Allocations keep their own port masks/reduction,
    current limit and float64 inward fade; they have no running signal maximum.
    Builder and allocation lifetimes are sequential, so reserve the larger.
    """
    own = run_tables(tables, run) if run is not None else tables
    ports, working = _gas_ports(own, context), context.itemsize
    count, variants = len(ports), 1 + _variants(tables, run)
    shared = (GasModelCache.DEFAULT_KEPT + 1) * (working + 3) + 2 * (1 + 13 * count)
    building = 11 * count + count * (working + 1) + 28
    # Hidden bridge profiles/transition (float32), mask, centreline coordinates
    # plus coupled threshold coefficients/fixed fractions/capacity scales.
    # Active meeting fallback adds a seed copy and component-selection overlap.
    allocations = (20 * count + 51) * variants
    if own.get("gas_allocation", {}).get("growth_only_from_injecting_ports", False):
        # The active set can only change at a protocol boundary. Each variant
        # holds one passable grid, cached reach grids, and saturation history.
        boundaries = {
            date for periods in ports.values() for span in periods for date in span
        }
        allocations += (4 + len(boundaries) + 3 * working) * variants
        # Port distances, reach scanning and volume peeling are sequential.
        # Only one generator is doing its transient allocation work at a time.
        shape = context.grid_shape
        if shape is None:
            canvas = grid_canvas(own)
            if canvas is None:
                raise ValueError("Gas growth memory needs a grid shape or complete canvas")
            shape = canvas.shape
        row_bytes = (12 * np.dtype(np.intp).itemsize + 12) * shape[-1]
        row_bytes_per_pixel = (row_bytes + context.grid_pixels - 1) // context.grid_pixels
        # Migration: masks, labels and group IDs, candidate indices and the
        # sorting workspace, pore volumes, capacity and prior/retained volumes.
        labels_bytes = np.dtype(np.int32).itemsize
        index_bytes = np.dtype(np.intp).itemsize
        volume_bytes = np.dtype(np.float64).itemsize
        migration = 14 + 4 * labels_bytes + 13 * index_bytes + 12 * volume_bytes
        allocations += max(33, 6 + row_bytes_per_pixel, migration)
    return shared + max(building, allocations)


def _gas_pitch(tables: Mapping, context) -> tuple[float, float]:
    """The allocation's coordinate pitch from grid metadata, without grid arrays.

    A configured measurement canvas supplies the sampling extent without a grid
    shape. Its pitch is independent of the cropped image dimensions; a one-pixel
    view only checks that the canvas contains the grid's origin. With a supplied
    shape, grid_pitch also checks that the canvas contains the whole grid.
    """
    model, canvas = grid_model(tables), grid_canvas(tables)
    shape = context.grid_shape
    if shape is None:
        if model is None or canvas is None:
            raise ValueError(
                "Phase-map gas memory needs Sizes.grid_shape (rows, columns), "
                "or a measured grid_to_rig with canvas_origin and canvas_shape."
            )
        shape = (1, 1)
    rig = tables["rig"]
    image = darsia.Image(
        np.broadcast_to(np.uint8(0), shape),
        dimensions=[rig["height"], rig["width"]],
        space_dim=2,
    )
    stride = tables.get("phase_rules", {}).get("pitch_stride")
    pitch = darsia.grid_pitch(
        image, model, np.dtype(f"f{context.itemsize}"), canvas, stride
    )
    if pitch is None or any(not np.isfinite(value) or value <= 0 for value in pitch):
        raise ValueError(
            "Phase-map gas memory needs a finite positive grid pitch; measured "
            "grids require phase_rules.pitch_stride and at least two samples per axis."
        )
    return pitch


def _gas_depth_bound_mm(tables: Mapping, context) -> float:
    """Bound the run-grid median by its configured depth source's maximum.

    Linear-in-hull/nearest depth interpolation and the rig's linear resize are
    convex combinations. Their output median cannot exceed the largest source
    depth. This reads only that source, never a complete rig or camera image.
    The source is also a declared rig input. Runtime lengths still use the
    actual median of the accepted run grid, not this admission upper bound.
    """
    depth = tables.get("depth", {})
    triangulated = depth.get("interpolation") == "triangulation"
    named = depth.get("measurements") if triangulated else depth.get("depth_map")
    if not named and not triangulated:
        results = tables.get("data", {}).get("results")
        if results:
            named = Path(results) / "setup" / "depth" / "depth_map.npz"
    if not named:
        raise ValueError("Gas memory requires a configured measured depth source")
    path = Path(named)
    if not path.is_absolute():
        if context.project is None:
            raise ValueError("Relative depth sources need the job's project")
        path = Path(context.project) / path
    values = (
        pd.read_csv(path, usecols=["depth"])["depth"].to_numpy()
        if triangulated
        else darsia.imread(path).img
    )
    valid = values[np.isfinite(values) & (values > 0)]
    if not valid.size:
        raise ValueError("Gas memory requires finite positive measured depth")
    # Round upward in the eventual working dtype, including interpolation's
    # conversion, rather than underestimating a float32 endpoint by one ulp.
    dtype = np.dtype(f"f{context.itemsize}")
    bound = np.nextafter(dtype.type(valid.max()), dtype.type(np.inf))
    return float(bound) * 1000


def _gas_route_bytes(tables: Mapping, run: str | None, context) -> int:
    """Retained ordered coordinates, including resampling and endpoint restoration.

    A simple MCP path has at most pixels-1 diagonal steps. Gaussian smoothing
    cannot increase total variation; restoring endpoints adds at most that
    variation again. Rounding adds at most one grid step per resampled segment.
    Four times the sample count plus a straight hidden segment therefore bounds
    the ordered raster, including repeated pixels. Three copies cover two zone
    payloads and the route currently being built. Fixed grid terms deliberately
    remain conservative instead of subtracting their ordinary coordinate count.
    """
    own = run_tables(tables, run) if run is not None else tables
    pitch = _gas_pitch(own, context)
    pixels = context.grid_pixels
    samples = int(np.ceil(max(0, pixels - 1) * np.hypot(*pitch) / min(pitch))) + 2
    shape = context.grid_shape
    if shape is None:
        canvas = grid_canvas(own)
        if canvas is None or int(np.prod(canvas.shape)) < pixels:
            raise ValueError("Gas route memory needs a grid shape or complete canvas")
        shape = canvas.shape
    vertices = 4 * samples + max(shape) + 1
    return 3 * 8 * vertices * len(_gas_ports(own, context))


def _gas_scratch_parts(tables: Mapping, run: str | None, context) -> dict[str, int]:
    """Array estimates for sequential contour, link and FFT-score operations.

    EDT counts its int8 input, two int32 feature/index grids, float64 coordinate
    differences and result, plus caller masks. MCP includes its float64 costs,
    cumulative costs, int16 offsets, int8 edges, indexed heap, Python traceback
    and arc-length resampling. FFT uses the actual fast lengths and half spectra;
    an additional real-grid workspace allowance is retained for the backend.
    These are implementation/configuration estimates, not universal bounds for
    external-library workspaces; host measured admission remains necessary.
    """
    own = run_tables(tables, run) if run is not None else tables
    pitch = _gas_pitch(own, context)
    shape = context.grid_shape
    if shape is None:
        canvas = grid_canvas(own)
        if canvas is None or int(np.prod(canvas.shape)) < context.grid_pixels:
            raise ValueError(
                "Gas scratch memory needs Sizes.grid_shape (rows, columns), "
                "or an explicit measurement canvas containing the complete grid."
            )
        # The full explicit canvas bounds a crop's shape; do not factor pixels.
        shape = canvas.shape
    shape = tuple(int(value) for value in shape)
    settings = asdict(
        parameters_at_depth(
            darsia.GasAllocationSettings(**own["gas_allocation"]),
            _gas_depth_bound_mm(own, context),
        )
    )
    window = settings["score_window_mm"]
    pixels, working = context.grid_pixels, context.itemsize
    widths = {}
    for name, length in (("contour_smoothing_mm", 2), ("contour_blend_mm", 3)):
        values = tuple(float(value) for value in settings[name])
        if len(values) != length or any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Gas {name} needs {length} finite nonnegative lengths.")
        widths[name] = values
    closing = float(settings["closing_length_mm"])
    if not np.isfinite(closing) or closing < 0:
        raise ValueError("Gas closing_length_mm must be finite and nonnegative.")
    margin, rounding = widths["contour_smoothing_mm"]
    _, blend_closing, blend_rounding = widths["contour_blend_mm"]
    reach = max(closing + margin + 4 * rounding, blend_closing + 4 * blend_rounding)
    padding = tuple(int(np.ceil(reach / step)) + 2 for step in pitch)
    padded = (shape[0] + 2 * padding[0]) * (shape[1] + 2 * padding[1])
    contour = 40 * padded + 12 * pixels + 64 * sum(shape)

    # The eight-neighbour path is simple. Its physical arc length cannot exceed
    # (pixels-1) diagonal steps, including a winding observed component.
    samples = int(np.ceil(max(0, pixels - 1) * np.hypot(*pitch) / min(pitch))) + 2
    pointer = np.dtype(np.intp).itemsize
    capacity = 1 << (max(128, pixels) - 1).bit_length()
    retained = (3 + 2 * working + 8 + 8 + 2 + 4 + pointer + 8) * pixels
    heap = (16 + pointer) * (capacity + capacity // 2)
    traceback = (
        sys.getsizeof((0, 0)) + 2 * sys.getsizeof(max(shape)) + 2 * pointer + 16
    ) * pixels
    # Original points/length/arc arrays use 32 B per vertex. Resampled points,
    # smooth/intermediate paths and raster conversion use at most 192 B/sample,
    # including repeated raster coordinates.
    # The solver remains alive through traceback and path smoothing, so its
    # allocated heap may coexist with either operation's temporary arrays.
    links = (
        retained
        + heap
        + max(48 * pixels, traceback, 32 * pixels + 192 * samples)
        + 4096
    )
    positions = working if grid_model(own) is not None else 8
    links = max(links, (3 * positions + 5) * pixels)

    sigma = window / np.sqrt(12)
    kernel_shape = tuple(2 * int(np.ceil(3 * sigma / step)) + 1 for step in pitch)
    cells = kernel_shape[0] * kernel_shape[1]
    fft_shape = tuple(
        next_fast_len(n + k - 1, real=True) for n, k in zip(shape, kernel_shape)
    )
    real_cells = fft_shape[0] * fft_shape[1]
    complex_cells = fft_shape[0] * (fft_shape[1] // 2 + 1)
    kernel_building = 25 * cells + 16 * sum(kernel_shape)
    retained_kernel = (9 + working) * cells + 8 * sum(kernel_shape)
    fft = 3 * working * pixels + 6 * working * complex_cells + 4 * working * real_cells
    exact_support = (2 * working + 36) * pixels
    score = max(kernel_building, retained_kernel + max(fft, exact_support))

    # Depth values are not loaded by admission. The implementation caps kernel
    # offsets at the grid edges, which bounds even an arbitrarily deep cell.
    # Quantised levels run sequentially; there is no number-of-levels multiplier.
    depth_kernel = tuple(2 * size - 1 for size in shape)
    depth_cells = depth_kernel[0] * depth_kernel[1]
    depth_fft = tuple(
        next_fast_len(n + k - 1, real=True) for n, k in zip(shape, depth_kernel)
    )
    depth_real = depth_fft[0] * depth_fft[1]
    depth_complex = depth_fft[0] * (depth_fft[1] // 2 + 1)
    depth_building = 12 * depth_cells + 16 * sum(depth_kernel)
    depth_filter = 5 * depth_cells + 4 * pixels + 24 * depth_complex + 16 * depth_real
    depth = 18 * pixels + max(depth_building, depth_filter)
    return {
        "contour": int(contour),
        "links": int(links),
        "score": int(score),
        "depth": int(depth),
    }


def _gas_scratch_bytes(tables: Mapping, run: str | None, context) -> int:
    """Largest scratch estimate; contour, link search and scoring are sequential."""
    return max(_gas_scratch_parts(tables, run, context).values())


class _Probabilities(Sequence):
    """A run's phase probabilities by frame, computed when asked for, a window of
    them kept; each frame's previous image is the run's acquisition just before
    it (:func:`previous_acquisition`), read through the rig's corrections as the
    corrected-image step reads a photo. With the run's phase support, the current
    image, the previous image and the baseline are read inside it, the rest
    unknown (NaN), as our production's phase step read them.

    With the run's expert knowledge (:func:`expert_limits`) the probabilities are
    limited to where CO2 and free gas may be (:func:`darsia.limit_probabilities`),
    and each frame's area that the model saw as CO2 and as free gas outside them
    is noted by index (``removed``)."""

    def __init__(
        self,
        ctx,
        model,
        grid,
        frames,
        kept: int,
        rig: Rig,
        support=None,
        limits: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
        co2_threshold: float | None = None,
    ) -> None:
        self.ctx, self.model, self.grid, self.frames = ctx, model, grid, frames
        self.rig, self.support = rig, support
        self.limits, self.co2_threshold = limits, co2_threshold
        self.removed: dict[int, dict[str, float]] = {}
        self.baseline = _within(grid.rig.baseline.img, support)
        self.image = lru_cache(maxsize=kept + 1)(self._image)
        self.probabilities = lru_cache(maxsize=kept)(self._probabilities)

    @property
    def limited(self) -> bool:
        """Whether expert knowledge limits the probabilities."""
        return any(limit is not None for limit in self.limits)

    def _image(self, index: int) -> np.ndarray:
        image = self.ctx.read("corrected-image", read_image, frame=self.frames[index])
        return _within(image.img, self.support)

    def _previous(self, index: int) -> np.ndarray | None:
        """The corrected image of a frame's previous acquisition, or None."""
        run = self.ctx.task.run
        photo = self.ctx.photo(run, self.frames[index])
        sessions = list(self.ctx.locations.sessions[run].values())
        previous = previous_acquisition(
            photo, sessions, self.rig.experiment.is_blacklisted
        )
        if previous is None:
            return None
        return corrected_photo(self.rig, previous)[0].img

    def _probabilities(self, index: int):
        found = darsia.phase_probabilities(
            self.model,
            self.image(index),
            self.baseline,
            _within(self._previous(index), self.support),
            pixel_size_mm=self.grid.pixel_mm,
        )
        if not self.limited:
            return found
        (co2, gas), (keep_co2, keep_gas) = found, self.limits
        seen = self.grid.domain & (co2 >= self.co2_threshold)
        seen_gas = seen & (gas >= self.model.gas_threshold)
        no_co2 = np.zeros(co2.shape, bool) if keep_co2 is None else ~keep_co2
        no_gas = no_co2 if keep_gas is None else no_co2 | ~keep_gas
        area = self.grid.pixel_area_mm2
        self.removed[index] = {
            "co2_area_mm2": float(np.sum(area, where=seen & no_co2)),
            "gas_area_mm2": float(np.sum(area, where=seen_gas & no_gas)),
        }
        return darsia.limit_probabilities(co2, gas, keep_co2, keep_gas)

    def removed_at(self, index: int) -> dict[str, float]:
        """The CO2 and free-gas area [mm2] that the model saw in a frame where its
        run's expert knowledge lets none be, before the phase rules."""
        if index not in self.removed:
            self.probabilities(index)
        return self.removed[index]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index):
        return self.probabilities(int(index))

    def clear(self) -> None:
        """Release the model's temporal arrays after the complete prepass."""
        self.probabilities.cache_clear()
        self.image.cache_clear()


def _phase_maps(ctx: StepContext) -> None:
    """One model pass and prefix-zone pass, then cached allocation in time order.

    Resume replays every selected frame to prepare the same prefix zones, then
    allocates from the first missing frame. Both private local caches are shared
    by primary and held-out volumes. Model frames are retained with exact local
    provenance for explicit later reuse; allocation-dependent zones are removed
    when this invocation ends.

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
    grid = run_grid(ctx.read("rig", read_rig), settings["material"], rules.pitch_stride)
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
    kept = window_frames(rules)
    rig = ctx.read("rig", read_rig)
    rig.load_experiment(experiment)
    support = grid_mask(
        run_tables(ctx.tables, run), "phase_support", grid.rig.baseline.img.shape
    )
    limits = expert_limits(run_tables(ctx.tables, run), grid.rig)
    limited = any(limit is not None for limit in limits)
    probabilities = _Probabilities(
        ctx,
        model,
        grid,
        frames,
        kept,
        rig,
        support,
        limits,
        rules.co2_threshold if limited else None,
    )
    models = darsia.gas_model_frames(
        probabilities,
        dates,
        grid.rig.facies,
        grid.domain,
        grid.pixel_area_mm2,
        grid.permeability,
        experiment.injection_protocol,
        rules,
        grid.chart,
        gas_threshold=model.gas_threshold,
        positions=grid.positions,
    )
    previous_photos = [
        previous_acquisition(
            ctx.photo(run, frame),
            list(ctx.locations.sessions[run].values()),
            experiment.is_blacklisted,
        )
        for frame in frames
    ]
    identity = model_cache_identity(ctx, frames, dates, previous_photos)

    def checked_models():
        yield from models
        after = [
            previous_acquisition(
                ctx.photo(run, frame),
                list(ctx.locations.sessions[run].values()),
                experiment.is_blacklisted,
            )
            for frame in frames
        ]
        if model_cache_identity(ctx, frames, dates, after) != identity:
            raise ValueError("Model cache provenance changed during the model pass.")

    with (
        GasModelCache(
            len(frames), retain=True, identity=identity, revision=ctx.revision
        ) as cached,
        GasZoneCache(len(frames)) as zones,
    ):
        print(f"Retained model cache: {cached.retained_path}", flush=True)
        prepared_zones = darsia.gas_model_zones(
            cached.store(checked_models()),
            dates,
            grid.rig.facies,
            grid.domain,
            grid.permeability,
            experiment.injection_protocol,
            rules,
            settings["gas_allocation"],
            positions=grid.positions,
            chart=grid.chart,
            depth_m=grid.rig.depth.img,
        )
        for _ in zones.store(prepared_zones):
            del _
        probabilities.clear()
        allocations = {}

        def saturation_before(held_out):
            """The free-gas saturation of the frame before a resumed run's start,
            for the allocation's growth rule; None from the run's first frame."""
            if not (
                start and settings["gas_allocation"].growth_only_from_injecting_ports
            ):
                return None
            return ctx.own(
                lambda folder: darsia.read_phase_maps(folder, held_out)["saturation"],
                frame=frames[start - 1],
            )

        def maps(held_out):
            volumes = ctx.read(
                "overflow", lambda f: darsia.read_frame_volumes(f, held_out)
            )
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
                model_frames=cached,
                zones=zones,
                depth_m=grid.rig.depth.img,
                positions=grid.positions,
                record=lambda index, value: allocations.__setitem__(
                    held_out, asdict(value)
                ),
                saturation_before=saturation_before(held_out),
            )

        found_maps = zip(frames[start:], *(maps(v) for v in variants))
        for index, (frame, *found) in enumerate(found_maps, start):

            def write(folder, found=found):
                for held_out, arrays in zip(variants, found):
                    darsia.write_phase_maps(folder, *arrays, held_out=held_out)

            measurements = {
                "gas_allocation": {
                    "primary": allocations[None],
                    "without": {
                        run: value
                        for run, value in allocations.items()
                        if run is not None
                    },
                }
            }
            if limited:
                measurements["expert_knowledge"] = probabilities.removed_at(index)
            ctx.publish(write, frame=frame, measurements=measurements)


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
    for frame in frames:
        if ctx.done(frame):
            continue
        rows = {held_out: [] for held_out in variants}
        image = ctx.read("corrected-image", read_image, frame=frame).img
        date = experiment.get_datetime(ctx.photo(run, frame))
        co2 = co2_state(grid, experiment, date)
        hours = experiment.time_since_start(date)
        before = None
        for held_out in variants:
            maps = ctx.read(
                "phase-maps", lambda f: darsia.read_phase_maps(f, held_out), frame=frame
            )
            capacity = darsia.dissolved_capacity(
                grid.geometry.pore_volume_m3, co2.solubility_co2, maps["saturation"]
            )
            # The donor changes the allocation, not the labels before its rules.
            # Older stored variants may differ; only reuse their same reading.
            if before is None or not np.array_equal(before, maps["before"]):
                signal, measured = read_signal(
                    grid, image, maps, reading, statistics_settings, native
                )
                before = maps["before"]
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
                    index=0,  # its index in the selected run is set when read
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
        measured = ctx.tables.get("jobs", {}).get("memory_admission") == "declared"
        facts = {"fit_memory": statistics_memory(primary, dtype, rows)} if measured else None
        ctx.publish(
            lambda folder: darsia.write_frames(folder, run, primary, dtype, rows),
            frame=frame,
            **({"measurements": facts} if measured else {}),
        )


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


def _calibration_ready_bytes(ctx: StepContext) -> dict[str, int]:
    """Inspect each accepted statistics input once for all calibration parts."""
    frames = {
        run: [ctx.read("run-statistics", stored_statistics_memory, run=run, frame=frame)
              for frame in selected]
        for run, selected in ctx.selection.frames.items()
    }
    settings = ctx.settings
    return {
        part: calibration_memory(
            [row for run, rows in frames.items() if not group.runs or run in group.runs for row in rows],
            held, settings["calibration_fit"], len(settings["material"].swi),
        )
        for part, (group, held) in settings["calibration_runs"].parts(list(frames)).items()
    }


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
        for index, selected in enumerate(ctx.selection.frames[run])
        for frame in ctx.read(
            "run-statistics",
            lambda f: darsia.read_frames(f, held, first_index=index),
            run=run,
            frame=selected,
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
    Memory(terms=(Term(_model_copy_bytes, fixed_bytes=1),)),
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
    Memory(terms=(Term(_displaced_table_bytes, fixed_bytes=1),)),
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
    Memory(grid=8.0, camera_bytes=20.0, grid_bytes=74.0),
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
    Memory(
        # Two loaded rigs and a corrected previous acquisition; feature
        # construction overlaps the rule/allocation generators. Fixed-width
        # terms cover labels, indices, masks and float64 distance/volume work.
        camera=13.0,
        grid=64.0,
        camera_bytes=46.0,
        grid_bytes=112.0,
        terms=(
            Term(_window, grid=3.0 + 2.0),  # RGB image and two probabilities
            Term(_variants, grid=4.0, grid_bytes=80.0),
            Term(_supported, grid=6.0),
            Term(_injection_map_bytes, grid_bytes=1.0),
            Term(_gas_scratch_bytes, fixed_bytes=1.0),
            Term(_gas_route_bytes, fixed_bytes=1.0),
            Term(_expert_limited, grid_bytes=2.0),  # two bool limit masks
        ),
        photos=1,
        photo_bytes=darsia.decoded_bytes,
    ),
    _phase_maps,
    {
        "phase_model": PhaseModelSource,
        "phase_rules": darsia.PhaseRuleSettings,
        "gas_allocation": darsia.GasAllocationSettings,
        "material": MaterialSettings,
    },
    each=Each.FRAME,
    sections=EXPERIMENT + EXPERT,
    files=_phase_map_files,
)
"""Each frame's phase maps and free-gas saturation (R4)."""
COLOR_RANGES = Step(
    "color-ranges",
    {"rig": Scope.ALL, "corrected-image": Scope.ALL, "phase-maps": Scope.ALL},
    Order.PARTS,
    Memory(
        grid=31.0,
        camera_bytes=37.0,
        grid_bytes=198.0,
        terms=(Term(_range_rigs, grid=12.0, camera_bytes=18.0, grid_bytes=64.0),),
    ),
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
    Memory(grid=26.0, camera_bytes=18.0, grid_bytes=115.0),
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
    Order.FRAMES,
    Memory(
        # Rig/geometry, RGB image, response/native signal, phase maps,
        # capacity, histogram and unknown-mass scratch. read_signal's mask
        # dictionary and sand_mean's list each hold one mask per sand.
        # Include the signal, measured mask and previous before labels cached
        # by per-frame statistics while another held-out variant is read.
        grid=46.0 + 1.0,
        camera_bytes=18.0,
        grid_bytes=88.0 + 2.0,
        terms=(
            Term(reading_sands, grid_bytes=2.0),
            # One frame's rows and variants coexist through publication.
            Term(_statistics_bytes, fixed_bytes=1),
        ),
    ),
    _run_statistics,
    {
        "run_statistics": darsia.RunStatisticsSettings,
        "material": MaterialSettings,
        "unknown_mass": darsia.UnknownMassSettings,
        **RESPONSE,
    },
    sections=(*READING, *EXPERIMENT),
    files=protocol_files,
    accelerated=True,  # The backend's sand means must agree bit for bit.
)
"""Each frame's statistics (R5), independent of other selected frames."""
CALIBRATION = Step(
    "calibration",
    {"run-statistics": Scope.ALL},
    Order.PARTS,
    Memory(ready=_calibration_ready_bytes),
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
