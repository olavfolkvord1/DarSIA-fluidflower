"""The minimal chain on a coarse run grid run as one job, against the direct calls.

Two synthetic runs of six frames each (the minimal chain's sensor images, mapping,
material light correction and delivered phase model) are set up, calibrated,
analysed and compared by the presets' steps. Their mass maps equal the methods
called directly, in single precision through the service and in double precision
in this process; a run whose phase maps were interrupted after the injection end
resumes at its first missing frame with the same maps.

"""

import datetime
import json
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import toml
from threadpoolctl import threadpool_info

import darsia
import darsia.jobs.__main__ as jobs_main
from darsia.jobs import queue
from darsia.jobs.job import create_job
from darsia.jobs.step import Selection, StepContext, table
from darsia.presets.workflows.analysis import steps as analysis
from darsia.presets.workflows.analysis.domain_mass import DomainMass, write_domain_mass
from darsia.presets.workflows.analysis.domains import AnalysisDomain, write_rows
from darsia.presets.workflows.calibration import steps as calibration
from darsia.presets.workflows.comparison import steps as comparison
from darsia.presets.workflows.comparison.mass_wasserstein import (
    WassersteinSettings,
    distance_scales,
    frame_cells,
    wasserstein_distances,
    wasserstein_reference,
)
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup import steps as setup
from darsia.presets.workflows.setup.setup_facies import make_facies
from darsia.presets.workflows.setup.setup_rig import make_experiment, make_rig
from darsia.presets.workflows.setup.setup_sand_map import make_sand_map
from darsia.utils.locations import Frame
from tests.unit.test_gas_allocation import TABLES as RULE_TABLES
from tests.unit.test_job_service import _serve, importable  # noqa: F401
from tests.unit.test_minimal_chain import COLOURS, _head

START = datetime.datetime(2026, 1, 1, 12)
HOUR = datetime.timedelta(hours=1)
STEPS = setup.STEPS + calibration.STEPS + analysis.STEPS + comparison.STEPS
KERNEL = sorted(
    {
        r["architecture"]
        for r in threadpool_info()
        if r.get("internal_api") == "openblas"
    }
)[0]
"""This process's BLAS kernel, so that the direct calls run under the job's."""
SETTINGS = """
[jobs]
blas_kernel = "{kernel}"
threads = 1
[data]
folder = "photos-r1"
baseline = "baseline.npz"
results = "results"
[depth]
measurements = "depth.csv"
depth_map = "depth.npz"
[labeling]
labels = "labels.npz"
[labeling.sand_map]
drawing = "drawing.json"
edge_fill_mm = 0.0
[facies]
path = "facies.npz"
props = "props.csv"
[facies.1]
labels = [1, 3]
[facies.2]
labels = [2, 4]
[protocols]
injection = "injection.csv"
pressure_temperature = "pt.csv"
[protocols.imaging]
photos-r1 = "imaging-r1.csv"
photos-r2 = "imaging-r2.csv"
[rig]
path = "rig"
width = 0.45
height = 0.3
dim = 2
[image_porosity]
mode = "full"
[corrections.grid]
precision = "{precision}"
pixel_size_mm = 5.0
area_factor = 1.0337
[corrections.mapping]
path = "mapping.npz"
[corrections.material_illumination]
fit_shape = [150, 225]
block_size = 6
edge_margin = 0.0
dn_scale = 1000.0
[corrections.material_illumination.targets]
{targets}
[corrections.material_illumination.fit]
spatial_ridge = 0.002
offset_ridge = 0.02
residual_constant_ridge = 0.02
residual_spatial_ridge = 0.01
iterations = 8
huber_threshold = 1.5
clip_level = 0.98
dark_level = 8.0
minimum_block_cells = 32
chroma_cutoff = 0.08
chroma_sigmas = 3.5
[phase_model]
folder = "model"
pixel_size_mm = 0.5
accept_canvas = true
[material]
swi = {{"1" = 0.1, "2" = 0.2}}
measured_swi = [1, 2]
[overflow]
statuses = ["Verified"]
recipe_tolerance = {{alkalinity = 0.05}}
variant = "circulation"
circulation = {{minimum_readings = 3}}
evaporation = {{minimum_readings = 3, rates_g_h = {{r1 = "measured"}}}}
[recipe]
alkalinity = 1.0
indicator = "bromocresol green"
[run_statistics]
mean_mm = 10.0
zero_signal = 0.0
saturation_signal = 1.2
intervals = 32768
[reading.sands.1]
colors = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]
values = [0.0, 1.0]
supports = [0.0, 1.0]
table = [0.0, 1.0]
factor = 1.0
[reading.sands.2]
colors = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]
values = [0.0, 1.0]
supports = [0.0, 1.0]
table = [0.0, 1.0]
factor = 1.0
[unknown_mass]
column_reach_mm = 20.0
intervals = 32768
[calibration_fit]
gain_window_h = [1.0, 3.0]
dilution_start_h = 3.5
dilution_bins_h = [3.5, 5.0]
anchor_h = 3.0
anchor_tolerance_h = 0.6
gain_grid = [[0.1, 10.0, 21]]
beta_grid = [[0.0, 1.0, 5]]
beta_growth_grid = [[0.0, 0.2, 3]]
tolerance = 1e-10
evaluations = 50
swi_limit = 1e-6
[calibration_runs]
fit = ["r1", "r2"]
folds = ["r1"]
[domains]
sand_labels = [1, 2, 3, 4]
[domains.domains.whole]
kind = "whole"
name = "Whole"
[domains.domains.upper]
kind = "layer"
name = "Upper sand, right"
label = 3
[domains.domains.lower]
kind = "layer"
name = "Lower sand, left"
label = 2
[events]
masses = "quantified"
[events.questions.arrival]
conditions = [{{component = "total", event = "detected", domain_id = "whole"}}]
[ensemble]
masses = "quantified"
centre_component = "total"
[wasserstein]
cell_mm = 15.0
hole_pixels = 3
empty_mass_kg = 1e-8
max_iterations = 60
distance_tolerance = 1e-4
increment_tolerance = 1e-2
bregman_updates = [0, 10, 20]
[mass_copy]
copy_mm = 10.0
[distances]
times_h = [3.0]
tolerance_h = 0.6
reference_run = "r2"
[runs.r2.data]
folder = "photos-r2"
[runs.r2.depth]
depth_map = "depth-r2.npz"
[runs.r2.recipe]
alkalinity = 1.04
[runs.r1.protocols]
overflow = "overflow-r1.csv"
"""


def _project(root: Path, precision: str) -> tuple[Path, Selection]:
    """Two runs of the minimal chain: the plume grows in each frame."""
    root.mkdir(parents=True)
    facies = np.ones((120, 180), int)
    facies[60:] = 2
    colour = COLOURS[np.pad(facies, ((0, 1), (0, 1)), mode="edge")]
    light = np.exp(0.2 * np.linspace(-1, 1, 181))[None, :, None]
    frames = {}
    for run, grow in (("r1", 4), ("r2", 5)):
        folder = root / f"photos-{run}"
        folder.mkdir()
        lines = ["path,datetime", f"baseline.npz,{START - 2 * HOUR}"]
        darsia.OpticalImage(colour * light, color_space="RGB").save(
            folder / "baseline.npz"
        )
        names = []
        for k in range(6):
            plume = np.zeros((121, 181, 3))
            r = 10 + grow * k
            plume[60 - r : 60 + r // 2, 90 - r : 90 + r, 0] = 0.3
            name = f"frame{k:02d}.npz"
            image = darsia.OpticalImage((colour + plume) * light, color_space="RGB")
            image.save(folder / name)
            lines.append(f"{name},{START + k * HOUR}")
            names.append(name)
        (root / f"imaging-{run}.csv").write_text("\n".join(lines) + "\n")
        # Water evaporates before the injection, 0.4 g/h, which the correction
        # adds back; the second run has no readings of its own.
        (root / f"overflow-{run}.csv").write_text(
            "image_time,weight_g,status\n"
            + "".join(
                f"{START + h * HOUR},"
                f"{1000 + max(0.0, h + 1) * grow / 4 - 0.4 * (h + 2)},Verified\n"
                for h in (-1.75, -1.5, -1.25, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
            )
        )
        frames[run] = tuple(Frame(f"photos-{run}", name) for name in names)
    i, j = np.mgrid[0:120, 0:180].astype(np.float32)
    darsia.ImageMappingCorrection(
        j, i, source_shape=(121, 181), dimensions=[0.3, 0.45]
    ).save(root / "mapping.npz")
    # Two sand layers of each facies, split at the plume's centre: the labels are
    # not the facies (no facies file: each run's facies follow its sand map).
    labels = facies + 2 * (np.arange(180) >= 90)[None, :]
    darsia.ScalarImage(labels, dimensions=[0.3, 0.45]).save(root / "labels.npz")
    depth = darsia.ScalarImage(np.full((120, 180), 0.01), dimensions=[0.3, 0.45])
    depth.save(root / "depth.npz")
    # The second run's rig is deeper, so that the distances' reference run counts.
    deeper = darsia.ScalarImage(np.full((120, 180), 0.012), dimensions=[0.3, 0.45])
    deeper.save(root / "depth-r2.npz")
    (root / "props.csv").write_text("id,porosity,permeability\n1,0.44,100\n2,0.4,50\n")
    corners = [(x, y) for y in (0, 0.15, 0.3) for x in (0, 0.225, 0.45)]
    drawing = {
        "anchors": [list(corner) for corner in corners],
        "polygons": [
            [2, [0, 1, 4, 3]],
            [4, [1, 2, 5, 4]],
            [1, [3, 4, 7, 6]],
            [3, [4, 5, 8, 7]],
        ],
    }
    (root / "drawing.json").write_text(json.dumps(drawing))
    (root / "injection.csv").write_text(
        "id,location_x,location_y,start,end,rate_g/s\n"
        f"1,0.225,0.15,{START - HOUR},{START + 2 * HOUR},0.0005\n"
    )
    (root / "pt.csv").write_text(
        "datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,"
        f"temperature_gradient_celsius\n{START - 3 * HOUR},1.013,20.0,-1.2e-4,0.0\n"
        f"{START + 1.4 * HOUR},1.0207,20.6,-1.1e-4,0.02\n"
        f"{START + 3.7 * HOUR},1.0061,19.7,-1.3e-4,-0.01\n"
        f"{START + 24 * HOUR},1.0153,20.2,-1.15e-4,0.03\n"
    )
    heads = root / "model" / "models" / "generic"
    heads.mkdir(parents=True)
    for role, head in (("co2_presence", (20, -3)), ("gas_within_co2", (6, -2.5))):
        payload = _head(*head)
        if role == "gas_within_co2":  # the change against the previous frame counts
            index = darsia.PHASE_FEATURE_NAMES.index("dynamic_delta_r")
            payload["coefficients"][index] = 12.0
        (heads / f"{role}_signal.json").write_text(json.dumps(payload))
    policy = {"gas_within_co2": {"presence_threshold": 0.66}}
    (root / "model" / "phase_decision_policy.json").write_text(
        json.dumps({"models": {"generic": policy}})
    )
    rules = toml.dumps({k: RULE_TABLES[k] for k in ("phase_rules", "gas_allocation")})
    targets = "".join(f'"{m}" = {np.log(COLOURS[m] * 1e3).tolist()}\n' for m in (1, 2))
    text = SETTINGS.format(kernel=KERNEL, precision=precision, targets=targets)
    (root / "config.toml").write_text(text + rules)
    sessions = {run: {f"photos-{run}": f"photos-{run}"} for run in frames}
    selection = Selection(frames, sessions, domains=("whole", "upper"))
    return root / "config.toml", selection


ATMOSPHERE = (
    "pressure_bar",
    "temperature_celsius",
    "pressure_gradient_bar",
    "temperature_gradient_celsius",
)


def _atmosphere(experiment, h: float, dtype) -> dict:
    """Our production's rule for a run's atmosphere and its gradients at a time:
    the record linear in its hours, those in the dtype."""
    record = experiment.pressure_temperature_protocol.df
    start = experiment.experiment_start
    times = [(date - start).total_seconds() / 3600 for date in record["datetime"]]
    times = np.asarray(times, dtype)
    return {name: float(np.interp(h, times, record[name])) for name in ATMOSPHERE}


def _co2_state(grid, experiment, date) -> darsia.PrecisionCO2MassAnalysis:
    """The CO2 state on a run's grid at a frame's atmosphere by our production's
    rule, in the grid's dtype."""
    h = (date - experiment.experiment_start).total_seconds() / 3600
    value = _atmosphere(experiment, h, grid.rig.pixel_area.img.dtype)
    return darsia.PrecisionCO2MassAnalysis(
        grid.rig.baseline,
        datum_m=grid.datum_m,
        atmospheric_pressure=value["pressure_bar"],
        atmospheric_temperature=value["temperature_celsius"],
        atmospheric_pressure_gradient=value["pressure_gradient_bar"],
        atmospheric_temperature_gradient=value["temperature_gradient_celsius"],
    )


def _direct(root: Path, selection: Selection) -> dict:
    """The mass maps of every frame by the methods called directly."""
    tables = tomllib.loads((root / "config.toml").read_text())

    def settings(kind, section):
        return settings_from_table(kind, table(tables, section))

    material = settings(calibration.MaterialSettings, "material")
    statistics = settings(darsia.RunStatisticsSettings, "run_statistics")
    reading = calibration.sand_reading(settings(calibration.ReadingSettings, "reading"))
    overflow = settings(darsia.OverflowSettings, "overflow")
    runs, donors = {}, {}
    for run, run_frames in selection.frames.items():
        own = root / f"{run}.toml"
        own.write_text(toml.dumps(tables["runs"][run]))
        config = FluidFlowerConfig(
            [root / "config.toml", own], require_data=False, require_results=False
        )
        labels = make_sand_map(config)
        labels.save(root / f"labels-{run}.npz")
        make_facies(config, labels).save(root / f"facies-{run}.npz")
        make_rig(
            Rig,
            config,
            labels=root / f"labels-{run}.npz",
            facies=root / f"facies-{run}.npz",
        ).save(root / run)
        rig = Rig.load(root / run)
        experiment = make_experiment(config)
        rig.load_experiment(experiment)
        images = []
        for frame in run_frames:
            rig.read_image(root / frame.session / frame.name).save(root / "image.npz")
            images.append(darsia.imread(root / "image.npz"))
        grid = calibration.run_grid(rig, material)
        start = experiment.experiment_start
        dates = [
            experiment.get_datetime(root / frame.session / frame.name)
            for frame in run_frames
        ]
        runs[run] = (grid, experiment, images, dates)
        volume = None
        workbook = tables["runs"][run].get("protocols", {}).get("overflow")
        if workbook is not None:  # the second run has no readings of its own
            readings = darsia.OverflowProtocol(root / workbook).readings(
                start, ["Verified"]
            )
            proposal = darsia.proposed_evaporation(
                darsia.displaced_volume(*readings, overflow), overflow.evaporation
            )
            rate = darsia.approved_evaporation(run, proposal, overflow.evaporation)
            assert rate is not None and rate > 0
            volume = darsia.displaced_volume(*readings, overflow, rate)
        pressure = experiment.pressure_temperature_protocol.df
        steps = experiment.injection_protocol.df
        donors[run] = darsia.OverflowRun(
            sand_map="drawing.json",
            recipe=tables["recipe"],
            pressure_h=np.asarray(
                [(d - start).total_seconds() / 3600 for d in pressure["datetime"]]
            ),
            pressure_bar=np.asarray(pressure["pressure_bar"], np.float64),
            injection_h=(
                (steps["start"].min() - start).total_seconds() / 3600,
                (steps["end"].max() - start).total_seconds() / 3600,
            ),
            volume=volume,
        )

    def density(experiment, h, dtype):
        value = _atmosphere(experiment, h, dtype)
        return darsia.atmospheric_gas_density(
            value["pressure_bar"], value["temperature_celsius"], dtype
        )

    def readings(run, dtype):
        # Our production interpolated the readings in the job's precision.
        volume = donors[run].volume
        if volume is None:
            return None
        return replace(
            volume,
            times_h=volume.times_h.astype(dtype),
            volume_ml=volume.volume_ml.astype(dtype),
        )

    volumes = {}
    for run, (grid, experiment, images, dates) in runs.items():
        start, dtype = experiment.experiment_start, grid.rig.pixel_area.img.dtype
        hours = [(date - start).total_seconds() / 3600 for date in dates]
        mass = [
            experiment.injection_protocol.injected_mass(time=h) * 1000 for h in hours
        ]
        here = [density(experiment, h, dtype) for h in hours]
        ranked = darsia.ranked_donors(run, donors, overflow)

        def state(donor, h, dtype=dtype):
            other = runs[donor][1]
            injected = other.injection_protocol.injected_mass(time=h) * 1000
            return injected, density(other, h, dtype)

        def frame_volumes(donor):
            return darsia.frame_volumes(
                readings(run, dtype),
                hours,
                None if donor is None else (donor, readings(donor, dtype)),
                mass,
                here,
                None if donor is None else lambda h: state(donor, h),
            )

        volumes[run] = {None: frame_volumes(ranked[0] if ranked else None)}
        if ranked:  # the second run's donor is held out of the fold: no next donor
            assert ranked == ["r1"] and run == "r2"
            assert {v.source for v in volumes[run][None]} == {"r1"}
            volumes[run]["r1"] = frame_volumes(None)
    rules = settings(darsia.PhaseRuleSettings, "phase_rules")
    allocation = settings(darsia.GasAllocationSettings, "gas_allocation")
    variants, frames = {}, {None: [], "r1": []}
    for run, (grid, experiment, images, dates) in runs.items():
        rig, start = grid.rig, experiment.experiment_start
        model = darsia.read_phase_model(
            root / "model", 0.5, canvas=rig.canvas, accept_canvas=True
        )
        probabilities = [
            darsia.phase_probabilities(
                model,
                image.img,
                rig.baseline.img,
                images[k - 1].img if k else None,
                pixel_size_mm=grid.pixel_mm,
            )
            for k, image in enumerate(images)
        ]
        variants[run] = {}
        for held_out, found in volumes[run].items():
            maps = darsia.free_gas(
                probabilities,
                dates,
                rig.facies,
                grid.domain,
                grid.pixel_area_mm2,
                grid.permeability,
                experiment.injection_protocol,
                rules,
                grid.geometry.pore_volume_m3,
                grid.geometry.swi,
                [v.volume_m3 for v in found],
                allocation,
                grid.chart,
                gas_threshold=model.gas_threshold,
            )
            variants[run][held_out] = [
                dict(zip(("labels", "rules", "saturation", "co2"), m)) for m in maps
            ]
        for held_out in (None, "r1"):
            found = volumes[run].get(held_out, volumes[run][None])
            maps = variants[run].get(held_out, variants[run][None])
            for index, (image, frame_maps, date) in enumerate(zip(images, maps, dates)):
                h = (date - start).total_seconds() / 3600
                co2 = _co2_state(grid, experiment, date)
                capacity = darsia.dissolved_capacity(
                    grid.geometry.pore_volume_m3,
                    co2.solubility_co2,
                    frame_maps["saturation"],
                )
                support = grid.domain & np.isfinite(image.img).all(axis=-1)
                histograms = next(
                    darsia.run_statistics(
                        [
                            (
                                image.img,
                                frame_maps["labels"],
                                frame_maps["rules"],
                                support,
                                capacity,
                            )
                        ],
                        rig.baseline.img,
                        rig.facies,
                        grid.optical_pitch,
                        grid.geometry.counted,
                        reading,
                        statistics,
                    )
                )
                signal, measured = calibration.read_signal(
                    grid, image.img, frame_maps, reading, statistics
                )
                unknown = darsia.unknown_statistics(
                    darsia.relative_signal(signal, statistics),
                    frame_maps["labels"],
                    frame_maps["rules"],
                    measured,
                    grid.geometry,
                    co2,
                    settings(darsia.UnknownMassSettings, "unknown_mass"),
                )
                gas = darsia.gas_statistics(
                    frame_maps["co2"],
                    darsia.model_gas(frame_maps["labels"], frame_maps["rules"]),
                    frame_maps["saturation"],
                    grid.facies,
                    grid.geometry.counted,
                    grid.geometry.pore_volume_m3,
                    co2.density_gaseous_co2,
                    co2.solubility_co2,
                )
                frames[held_out].append(
                    darsia.CalibrationFrame(
                        run,
                        index,
                        h,
                        h,
                        experiment.injection_protocol.injected_mass(time=h) * 1000,
                        histograms,
                        gas,
                        unknown.column_estimate(),
                        found[index].volume_m3,
                        found[index].source == darsia.MEASURED,
                    )
                )
    arguments = (
        settings(darsia.CalibrationFitSettings, "calibration_fit"),
        {1: 0.1, 2: 0.2},
        [1, 2],
    )
    fit = darsia.fit_calibration(frames[None], ["r1", "r2"], *arguments)
    fold, predictions = darsia.calibration_fold(
        frames["r1"], ["r1", "r2"], *arguments, "r1"
    )
    runs = {
        run: (grid, experiment, images, variants[run][None], dates)
        for run, (grid, experiment, images, dates) in runs.items()
    }
    masses = darsia.mass_calibrations(fit, statistics)
    made, copies = {}, {}
    for run, (grid, experiment, images, maps, dates) in runs.items():
        copy = darsia.MassCopy(grid.geometry, darsia.MassCopySettings(10.0))
        start = experiment.experiment_start
        for frame, image, frame_maps, date in zip(
            selection.frames[run], images, maps, dates
        ):
            h = (date - start).total_seconds() / 3600
            signal, measured = calibration.read_signal(
                grid, image.img, frame_maps, reading, statistics
            )
            relative = darsia.ScalarImage(
                darsia.relative_signal(signal, masses[run]),
                name=frame.name,
                date=date,
                reference_date=start,
                dimensions=image.dimensions,
                origin=image.origin,
            )
            calibrated = masses[run]
            factor = darsia.dilution_factor(
                h,
                grid.geometry.pore_volume(grid.domain & frame_maps["co2"]),
                calibrated.anchor_h,
                calibrated.anchor_pore_volume_m3,
                calibrated.beta,
                calibrated.beta_growth,
                signal.dtype,
            )
            made[(run, frame)] = darsia.mass_maps(
                relative,
                frame_maps["labels"],
                frame_maps["rules"],
                frame_maps["saturation"],
                measured,
                grid.geometry,
                _co2_state(grid, experiment, date),
                calibrated,
                factor,
            )
            copies[(run, frame)] = {
                **copy(made[(run, frame)], frame_maps["labels"]),
                **copy.static,
            }
    sand_labels = tables["domains"]["sand_labels"]
    defined = tables["domains"]["domains"]
    domains = [AnalysisDomain(domain_id=n, **defined[n]) for n in selection.domains]
    domain_mass = {}
    for run, (grid, experiment, images, maps, dates) in runs.items():
        sums = DomainMass(domains, grid.geometry, grid.rig.labels, sand_labels)
        start, rows = experiment.experiment_start, []
        for frame, date in zip(selection.frames[run], dates):
            h = (date - start).total_seconds() / 3600
            co2 = _co2_state(grid, experiment, date)
            for row in sums(made[(run, frame)], co2):
                case = f"{frame.session}/{frame.name}"
                rows.append({**row, "run": run, "case_id": case, "elapsed_h": h})
        domain_mass[run] = rows
    wasserstein = settings(WassersteinSettings, "wasserstein")
    chart = settings(comparison.DistanceSettings, "distances").chart_labels
    grid, experiment = runs["r2"][:2]  # the configured reference run
    reference = wasserstein_reference(
        grid.rig.labels,
        grid.rig.permeability.img,
        grid.rig.porosity.img,
        sand_labels,
        chart,
        domains,
        wasserstein,
    )
    scales = distance_scales(
        3.0,
        experiment.injection_protocol,
        _co2_state(grid, experiment, experiment.experiment_start),
        grid.rig.depth.img,
        grid.rig.facies.dimensions,
        domains,
    )
    found = {}
    for run, (grid, experiment, images, maps, dates) in runs.items():
        start = experiment.experiment_start
        hours = [(date - start).total_seconds() / 3600 for date in dates]
        nearest = int(np.argmin([abs(h - 3.0) for h in hours]))
        frame = selection.frames[run][nearest]
        found[run] = frame_cells(
            made[(run, frame)],
            maps[nearest]["labels"],
            grid.rig.labels,
            grid.facies,
            grid.geometry,
            reference,
            wasserstein,
        )
    distances = wasserstein_distances(3.0, found, reference, scales, wasserstein)
    return {
        "maps": made,
        "copies": copies,
        "fit": fit,
        "fold": (fold, predictions),
        "domain_mass": domain_mass,
        "distances": [asdict(distance) for distance in distances],
    }


def _compare(project: Path, selection: Selection, direct: dict) -> None:
    """Every stored mass map, its copy on metric cells and the fit equal the
    direct calls'."""
    results = project / "results"
    for (run, frame), maps in direct["maps"].items():
        stored = darsia.read_mass_maps(
            results / "mass-maps" / run / frame.session / frame.name
        )
        for name in darsia.MASS_FIELDS:
            np.testing.assert_array_equal(
                getattr(stored, name).img, getattr(maps, name).img
            )
            assert getattr(stored, name).img.dtype == getattr(maps, name).img.dtype
    for (run, frame), copy in direct["copies"].items():
        folder = results / "mass-copy" / run / frame.session / frame.name
        stored = darsia.read_mass_copy(folder)
        assert set(stored) == set(copy)
        for name, values in copy.items():
            np.testing.assert_array_equal(stored[name].img, values)
    record = darsia.read_calibration(results / "calibration" / "fit")
    assert record["calibration"]["gain"] == direct["fit"].gain
    assert record["calibration"]["beta"] == direct["fit"].beta
    # The fold leaving the second run's donor out read its frames without it.
    record = darsia.read_calibration(results / "calibration" / "fold-r1")
    fold, predictions = direct["fold"]
    assert record["calibration"] == json.loads(json.dumps(asdict(fold)))
    rows = [asdict(prediction) for prediction in predictions]
    assert record["predictions"] == json.loads(json.dumps(rows))
    for run, rows in direct["domain_mass"].items():  # on each run's sand map
        write_domain_mass(project / f"domain-mass-{run}.csv", rows)
        stored = results / "domain-mass" / run / "domain_mass.csv"
        assert stored.read_text() == (project / f"domain-mass-{run}.csv").read_text()
    rows = direct["distances"]
    assert rows
    write_rows(project / "distances.csv", rows, list(rows[0]))
    stored = results / "distances" / "3.0h" / "distances.csv"
    assert stored.read_text() == (project / "distances.csv").read_text()


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_the_chain_as_jobs_equals_the_direct_calls(
    tmp_path, monkeypatch, importable, precision  # noqa: F811
):
    """Every frame's mass maps of the job equal the direct calls'; the comparisons
    are made; single precision runs through the service's task processes, double
    precision in this process."""
    project = tmp_path / "project"
    config, selection = _project(project, precision)
    monkeypatch.chdir(project)
    direct = _direct(project, selection)
    jobs = tmp_path / "jobs"
    job = create_job(jobs, STEPS, selection, config)
    if precision == "float32":
        _serve(jobs, tmp_path, seconds=1200.0)
    else:
        _run(job)
    failed = queue.markers(job, queue.FAILED)
    assert not failed, [
        queue.read_marker(job / queue.FAILED / f"{t}.{k}")["traceback"]
        for t, k in failed.items()
    ]
    assert queue.waiting(job) == set()
    _compare(project, selection, direct)
    for name in ("events/events.csv", "ensemble/mass_long.csv"):
        assert (project / "results" / name).is_file()
    if precision == "float64":
        _reused_only_while_unchanged(jobs, project, config, selection)


def _reused_only_while_unchanged(jobs, project, config, selection):
    """The job made again is reused whole; a hand-edited protocol, a model trained
    again into its folder and another domain each refuse the results made from
    them; the phase model serves another selection of runs."""
    assert queue.tasks(create_job(jobs, STEPS, selection, config)) == {}
    frame = "corrected-image/r1/photos-r1/frame00.npz"
    edits = (
        ("injection.csv", frame, "file:injection.csv"),
        ("model/phase_decision_policy.json", "phase-model", "file:model"),
    )
    for name, key, why in edits:
        text = (project / name).read_text()
        (project / name).write_text(text + "\n")  # edited by hand
        with pytest.raises(ValueError, match=f"{key} is stored and does not") as e:
            create_job(jobs, STEPS, selection, config)
        assert f"{key} is stored and does not serve the job: Made with other " in str(
            e.value
        )
        assert why in str(e.value)
        (project / name).write_text(text)
    more = Selection(selection.frames, selection.sessions, ("whole", "upper", "lower"))
    with pytest.raises(ValueError, match="domain-mass/r1 is stored and does not") as e:
        create_job(jobs, STEPS, more, config)
    assert "Made with other scope" in str(e.value) and "mass-maps/" not in str(e.value)
    one = Selection({"r2": selection.frames["r2"]}, selection.sessions)
    assert queue.tasks(create_job(jobs, (calibration.PHASE_MODEL,), one, config)) == {}


def _run(job, skip=()):
    """Run a job's ready tasks in this process, but the skipped ones."""
    while ready := [t for t in queue.ready(job) if t.id not in skip]:
        for task in ready:
            claim = queue.claim(job, task.id, "here")
            ctx = jobs_main.context(job, task, claim.token)
            ctx.step.run(ctx)
            queue.finish(claim, ctx.adopted)


def _counting(publish, published):
    """A publish that notes the frames of the phase maps it publishes."""

    def counted(ctx, *args, **kwargs):
        if ctx.step.name == "phase-maps":
            published.append(kwargs["frame"])
        return publish(ctx, *args, **kwargs)

    return counted


def test_phase_maps_resumed_after_the_injection_end_equal_the_whole_run(
    tmp_path, monkeypatch
):
    """A run's phase maps interrupted after the injection end's anchor frame resume
    at the first missing frame; every map and saturation equals the whole run's."""
    project = tmp_path / "project"
    config, selection = _project(project, "float32")
    config.write_text(  # the first run alone, the one fit run
        config.read_text().replace('fit = ["r1", "r2"]', 'fit = ["r1"]')
    )
    monkeypatch.chdir(project)
    one = Selection({"r1": selection.frames["r1"]}, selection.sessions)
    early = setup.STEPS + (
        calibration.PHASE_MODEL,
        calibration.DISPLACED_VOLUME,
        calibration.OVERFLOW,
        calibration.PHASE_MAPS,
    )
    _run(create_job(tmp_path / "jobs", early, one, config))
    folder = project / "results" / "phase-maps" / "r1"
    whole = {
        f: darsia.read_phase_maps(folder / f.session / f.name) for f in one.frames["r1"]
    }
    again = project / "config-again.toml"
    again.write_text(
        config.read_text().replace('results = "results"', 'results = "again"')
    )
    job = create_job(tmp_path / "again", early, one, again)
    _run(job, skip={"phase-maps.r1"})
    task = queue.tasks(job)["phase-maps.r1"]
    claim = queue.claim(job, task.id, "here")
    ctx = jobs_main.context(job, task, claim.token)
    publish, published = ctx.publish, []

    def killed_after_four(*args, **kwargs):
        if len(published) == 4:  # frames 0 to 3 are stored; frame 2 is the anchor
            raise SystemExit("killed")
        published.append(kwargs["frame"])
        publish(*args, **kwargs)

    monkeypatch.setattr(ctx, "publish", killed_after_four)
    with pytest.raises(SystemExit):
        ctx.step.run(ctx)
    queue.release(claim)
    again_published = []
    monkeypatch.setattr(
        StepContext, "publish", _counting(StepContext.publish, again_published)
    )
    _run(job)
    assert claim.token in queue.accepted(job, task.id)
    assert again_published == list(one.frames["r1"][4:])  # from the first missing
    folder = project / "again" / "phase-maps" / "r1"
    for frame, maps in whole.items():
        resumed = darsia.read_phase_maps(folder / frame.session / frame.name)
        for name, array in maps.items():
            np.testing.assert_array_equal(resumed[name], array)


def test_a_frame_without_measured_colour_below_co2_gives_the_fit_a_column(
    tmp_path, monkeypatch
):
    """Dissolved CO2 whose colour is not measured, below measured CO2 in its
    column, reaches the fit as the frame's column estimate: the stored statistics
    hold the frame's unknown-mass statistics, read from above."""
    project = tmp_path / "project"
    config, selection = _project(project, "float64")
    monkeypatch.chdir(project)
    steps = setup.STEPS + calibration.STEPS[:5]
    job = create_job(tmp_path / "jobs", steps, selection, config)
    statistics = {"run-statistics.r1", "run-statistics.r2"}
    _run(job, skip=statistics)
    frame = selection.frames["r1"][-1]
    results = project / "results"
    maps = darsia.read_phase_maps(
        results / "phase-maps" / "r1" / frame.session / frame.name
    )
    dissolved = maps["labels"] == darsia.PhaseLabel.DISSOLVED
    rows, columns = np.nonzero(dissolved[1:] & dissolved[:-1])
    row, column = rows[len(rows) // 2] + 1, columns[len(rows) // 2]
    stored = results / "corrected-image" / "r1" / frame.session / frame.name
    image = darsia.imread(stored / "image.npz")
    image.img[row, column] = np.nan  # its colour is not measured
    image.save(stored / "image.npz")
    _run(job, skip={"run-statistics.r2"})
    found = darsia.read_frames(results / "run-statistics" / "r1")[-1]
    tables = tomllib.loads(config.read_text())
    material = settings_from_table(calibration.MaterialSettings, tables["material"])
    grid = calibration.run_grid(Rig.load(results / "rig" / "r1"), material)
    statistics = settings_from_table(
        darsia.RunStatisticsSettings, tables["run_statistics"]
    )
    reading = calibration.sand_reading(
        settings_from_table(calibration.ReadingSettings, tables["reading"])
    )
    signal, measured = calibration.read_signal(
        grid, image.img, maps, reading, statistics
    )
    assert not measured[row, column]
    experiment = make_experiment(
        FluidFlowerConfig([config], require_data=False, require_results=False)
    )
    co2 = _co2_state(
        grid, experiment, experiment.get_datetime(project / frame.session / frame.name)
    )
    wanted = darsia.unknown_statistics(
        darsia.relative_signal(signal, statistics),
        maps["labels"],
        maps["rules"],
        measured,
        grid.geometry,
        co2,
        settings_from_table(darsia.UnknownMassSettings, tables["unknown_mass"]),
    ).column_estimate()
    assert wanted is not None and found.column is not None
    assert wanted.weight.sum() > 0
    for name in ("relative", "ratio", "weight"):
        np.testing.assert_array_equal(
            getattr(found.column, name), getattr(wanted, name)
        )


def test_a_fold_without_a_runs_donor_takes_the_next_donor(tmp_path, monkeypatch):
    """A run without readings of its own takes its free gas from the most similar
    fit run with readings; where that donor is held out of a fold, the frames get
    the next donor's volumes too, which the fold reads."""
    project = tmp_path / "project"
    config, selection = _project(project, "float64")
    monkeypatch.chdir(project)
    tables = tomllib.loads(config.read_text())
    tables["runs"]["r3"] = {
        "data": {"folder": "photos-r1"},
        "protocols": {"overflow": "overflow-r3.csv"},
    }
    readings = (project / "overflow-r1.csv").read_text().splitlines()
    rows = [line.split(",") for line in readings[1:]]
    (project / "overflow-r3.csv").write_text(  # thrice the first run's water
        "\n".join(
            [readings[0]] + [f"{t},{3 * float(w) - 2000},{s}" for t, w, s in rows]
        )
        + "\n"
    )
    tables["calibration_runs"] = {"fit": ["r1", "r2", "r3"], "folds": ["r1"]}
    frames = {**selection.frames, "r3": selection.frames["r1"]}
    overflow = settings_from_table(darsia.OverflowSettings, tables["overflow"])
    fits = settings_from_table(calibration.FitRuns, tables["calibration_runs"])
    stored = {}

    def ctx(run, step):
        def publish(write):
            folder = tmp_path / step / run
            folder.mkdir(parents=True)
            write(folder)
            stored[step, run] = folder

        def read(kind, reader, run=run):
            if kind == "rig":
                return SimpleNamespace(pixel_area=SimpleNamespace(img=np.zeros(1)))
            return reader(stored[kind, run])

        return SimpleNamespace(
            task=SimpleNamespace(run=run, frames=frames[run]),
            selection=SimpleNamespace(frames=frames),
            config=(config,),
            tables=tables,
            settings={"overflow": overflow, "calibration_runs": fits},
            photo=lambda run, frame: project / frame.session / frame.name,
            publish=publish,
            read=read,
        )

    for run in frames:
        calibration._displaced_volume(ctx(run, "displaced-volume"))
    calibration._overflow(ctx("r2", "overflow"))
    folder = stored["overflow", "r2"]
    primary = darsia.read_frame_volumes(folder)
    without = darsia.read_frame_volumes(folder, held_out="r1")
    assert darsia.held_out_runs(folder) == ("r1",)
    assert {v.source for v in primary.values()} == {"r1"}  # ties go by name
    assert {v.source for v in without.values()} == {"r3"}
    for key, volume in primary.items():
        assert without[key].volume_m3 > volume.volume_m3 > 0
    assert darsia.read_frame_volumes(folder, held_out="r3") == primary
    calibration._overflow(ctx("r1", "overflow"))  # its own readings
    own = stored["overflow", "r1"]
    assert {v.source for v in darsia.read_frame_volumes(own).values()} == {"measured"}
    assert darsia.held_out_runs(own) == ()


def test_a_runs_atmosphere_is_its_record_over_its_hours(tmp_path):
    """A run's atmosphere and its upward gradients at a time, which its frames'
    CO2 state and its donor's gas densities take, are its pressure-temperature
    record interpolated over the hours since its start, in the grid's dtype: our
    production's values for this record."""
    start = START
    offsets = ((-30, 7), (17, 41), (52, 13), (96, 58))
    record = pd.DataFrame(
        {
            "datetime": [
                start + datetime.timedelta(minutes=m, seconds=s) for m, s in offsets
            ],
            "pressure_bar": [1.0131, 1.0207, 1.0061, 1.0153],
            "temperature_celsius": [20.0, 20.6, 19.7, 20.2],
            "pressure_gradient_bar": [-1.2e-4, -1.1e-4, -1.3e-4, -1.15e-4],
            "temperature_gradient_celsius": [0.0, 0.02, -0.01, 0.03],
        }
    )
    record.to_csv(tmp_path / "pt.csv", index=False)
    experiment = SimpleNamespace(
        pressure_temperature_protocol=darsia.PressureTemperatureProtocol(
            tmp_path / "pt.csv"
        ),
        time_since_start=lambda date: (date - start).total_seconds() / 3600,
    )
    wanted = {
        "float32": [
            (1.0189291519976085, 20.460196210337518),
            (1.0147317567027136, 20.232094591263177),
            (1.0110306515383032, 19.967970192299084),
        ],
        "float64": [
            (1.0189291520672739, 20.46019621583742),
            (1.0147317567567566, 20.232094594594596),
            (1.0110306517690875, 19.967970204841713),
        ],
    }
    gradients = {
        "float32": [
            (-0.00011233006316104141, 0.015339873677917185),
            (-0.00011817567574970722, 0.007736486375439165),
            (-0.00012196089423102742, 0.011437615383926873),
        ],
        "float64": [
            (-0.00011233006306937632, 0.015339873861247372),
            (-0.00011817567567567567, 0.007736486486486487),
            (-0.0001219608938547486, 0.011437616387337058),
        ],
    }
    for dtype, values in wanted.items():
        hours = (0.11, 0.53, 1.27)
        found = [calibration.run_atmosphere(experiment, h, dtype) for h in hours]
        assert found == values
        found = [
            calibration.run_atmosphere(experiment, h, dtype, gradient=True)
            for h in hours
        ]
        assert found == gradients[dtype]
        image = darsia.ScalarImage(np.zeros((2, 2), dtype), dimensions=[0.02, 0.02])
        grid = SimpleNamespace(
            rig=SimpleNamespace(baseline=image, pixel_area=image), datum_m=0.02
        )
        for h, value, gradient in zip(hours, values, gradients[dtype]):
            date = start + datetime.timedelta(hours=h)
            co2 = calibration.co2_state(grid, experiment, date)
            assert (co2.atmospheric_pressure, co2.atmospheric_temperature) == value
            assert (
                co2.atmospheric_pressure_gradient,
                co2.atmospheric_temperature_gradient,
            ) == gradient


def test_phase_maps_refuse_frames_out_of_time_order(tmp_path, monkeypatch):
    """A run's frames out of time order are refused before any map is made: the
    phase maps and the allocation read the frames by index."""
    project = tmp_path / "project"
    config, selection = _project(project, "float64")
    monkeypatch.chdir(project)
    frames = selection.frames["r1"]
    ctx = SimpleNamespace(
        task=SimpleNamespace(run="r1", frames=(frames[1], frames[0]) + frames[2:]),
        config=(config,),
        tables=tomllib.loads(config.read_text()),
        settings={},
        photo=lambda run, frame: project / frame.session / frame.name,
    )
    with pytest.raises(ValueError, match=r"not in time order: \['frame00.npz'\]"):
        calibration._phase_maps(ctx)


def test_a_comparison_time_keeps_its_digits():
    """A comparison time's part is named by the time as configured, every digit
    kept, and read back from the settings; a time given twice is refused."""
    named = comparison.named_times([72.33333, 72.333334, 3])
    assert named == {"72.33333h": 72.33333, "72.333334h": 72.333334, "3.0h": 3.0}
    with pytest.raises(ValueError, match="configured twice"):
        comparison.named_times([3, 3.0])


def test_the_rig_is_known_by_its_default_depth_map(tmp_path):
    """A depth table that names no depth map names the depth setup's map in the
    results folder, so that a map set up again is not taken for the one a stored
    rig read."""
    tables = {"data": {"results": "results"}, "depth": {"measurements": "depth.csv"}}
    found = [Path(path).as_posix() for path in setup._rig_stamped(tables, tmp_path, [])]
    assert found == ["results/setup/depth/depth_map.npz"]


def test_a_job_without_a_fit_run_or_the_reference_run_is_refused(tmp_path):
    """A job over runs without one of the configured fit runs, among which the
    overflow ranks its donors and over which the fit is made, is refused, naming
    it; so is a job of distances without their configured reference run."""
    config, selection = _project(tmp_path / "project", "float64")
    jobs, frames = tmp_path / "jobs", selection.frames
    alone = Selection({"r2": frames["r2"]}, selection.sessions)
    with pytest.raises(ValueError, match=r"overflow needs run\(s\) \['r1'\]"):
        create_job(jobs, STEPS, alone, config)
    alone = Selection({"r1": frames["r1"]}, selection.sessions)
    with pytest.raises(ValueError, match=r"distances needs run\(s\) \['r2'\]"):
        create_job(jobs, (comparison.DISTANCES,), alone, config)
