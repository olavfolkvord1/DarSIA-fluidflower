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
  CO2 signal and its gas statistics, as the fit reads them.
- ``calibration`` (R6) fits the gain and the dilution across the fit runs, and
  each fold leaves one run out.

"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
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
    Term,
    run_tables,
    table,
)
from darsia.multiphase.gas_allocation import injection_ports
from darsia.multiphase.gas_model_cache import GasModelCache
from darsia.presets.workflows.config.protocols import InjectionProtocolConfig
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_rig import make_experiment
from darsia.presets.workflows.setup.steps import (
    EXPERIMENT,
    grid_mask,
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


def _overflow_files(tables: Mapping, project, folders) -> list[str]:
    """The run's protocols and its overflow workbook."""
    workbook = _overflow_file(tables)
    found = protocol_files(tables, project, folders)
    return found + ([workbook] if workbook is not None else [])


def _overflow(ctx: StepContext) -> None:
    """Each frame's free-gas volume from the run's own scale readings."""
    run = ctx.task.run
    settings = ctx.settings["overflow"]
    experiment = make_experiment(run_settings(ctx, run, EXPERIMENT))
    start = experiment.experiment_start
    path = _overflow_workbook(run_tables(ctx.tables, run))
    volume = None
    if path is not None:
        protocol = darsia.OverflowProtocol(path)
        # The readings count from the injection start in their own clock, the
        # sheet's, as our production read each reviewed curve; the injection
        # protocol's start is in the photographs' clock.
        origin = start if protocol.injection_start is None else protocol.injection_start
        times, weights = protocol.readings(origin, list(settings.statuses))
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


def _supported(tables: Mapping, run: str | None, context=None) -> int:
    """One where a run names a phase support, whose masked images the phase maps
    hold beside the read ones; else none."""
    return int(run is not None and _phase_support(run_tables(tables, run)) is not None)


def reading_sands(tables: Mapping, run: str | None, context=None) -> int:
    """The sand masks read_signal and sand_mean keep simultaneously."""
    tables = run_tables(tables, run) if run is not None else tables
    return len(tables.get("reading", {}).get("sands", {}))


def _statistics_entries(tables: Mapping, run: str | None, context) -> int:
    """Histogram values retained as Python rows by the initial run writer."""
    own = run_tables(tables, run) if run is not None else tables
    bins = int(own["run_statistics"]["intervals"]) + 1
    return 2 * min(bins * reading_sands(tables, run), context.grid_pixels)


def _gas_port_count(tables: Mapping, run: str | None, context) -> int:
    """Distinct measured injecting ports, including per-run protocol overrides."""
    own = run_tables(tables, run) if run is not None else tables
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
    return len(injection_ports(protocol))


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
    count, working = _gas_port_count(tables, run, context), context.itemsize
    shared = (GasModelCache.DEFAULT_KEPT + 1) * (working + 3) + 2 * (1 + 13 * count)
    building = 11 * count + count * (working + 1) + 28
    # Hidden bridge profiles/transition (float32), mask, centreline coordinates
    # plus coupled threshold coefficients/fixed fractions/capacity scales.
    # Active meeting fallback adds a seed copy and component-selection overlap.
    allocations = (20 * count + 51)
    return shared + max(building, allocations)


class _Probabilities(Sequence):
    """A run's phase probabilities by frame, computed when asked for, a window of
    them kept; each frame's previous image is the run's acquisition just before
    it (:func:`previous_acquisition`), read through the rig's corrections as the
    corrected-image step reads a photo. With the run's phase support, the current
    image, the previous image and the baseline are read inside it, the rest
    unknown (NaN), as our production's phase step read them."""

    def __init__(
        self, ctx, model, grid, frames, kept: int, rig: Rig, support=None
    ) -> None:
        self.ctx, self.model, self.grid, self.frames = ctx, model, grid, frames
        self.rig, self.support = rig, support
        self.baseline = _within(grid.rig.baseline.img, support)
        self.image = lru_cache(maxsize=kept + 1)(self._image)
        self.probabilities = lru_cache(maxsize=kept)(self._probabilities)

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
        return self.rig.read_image(previous).img

    def _probabilities(self, index: int):
        return darsia.phase_probabilities(
            self.model,
            self.image(index),
            self.baseline,
            _within(self._previous(index), self.support),
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
    kept = window_frames(rules)
    rig = ctx.read("rig", read_rig)
    rig.load_experiment(experiment)
    support = grid_mask(
        run_tables(ctx.tables, run), "phase_support", grid.rig.baseline.img.shape
    )
    maps = darsia.free_gas(
        _Probabilities(ctx, model, grid, frames, kept, rig, support),
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
        depth_m=grid.rig.depth.img,
    )
    for frame, (labels, rule_map, saturation, co2, _before) in zip(frames[start:], maps):
        ctx.publish(
            lambda f: darsia.write_phase_maps(f, labels, rule_map, saturation, co2),
            frame=frame,
        )


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
                column=None,
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
            Term(_supported, grid=6.0),
            Term(_injection_map_bytes, grid_bytes=1.0),
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
    sections=EXPERIMENT,
    files=_phase_map_files,
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
            # The initial writer retains Python histogram rows for the run.
            Term(
                _statistics_entries,
                frame_bytes=sys.getsizeof(0.0) + np.dtype(np.intp).itemsize,
            ),
        ),
    ),
    _run_statistics,
    {
        "run_statistics": darsia.RunStatisticsSettings,
        "reading": ReadingSettings,
        "material": MaterialSettings,
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
