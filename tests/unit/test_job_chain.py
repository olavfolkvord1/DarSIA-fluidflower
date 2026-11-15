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
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

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
from darsia.utils.locations import Frame, normal_path
from tests.unit.test_gas_allocation import TABLES as RULE_TABLES
from tests.unit.test_job_service import _serve, importable, numpys_blas  # noqa: F401
from tests.unit.test_minimal_chain import COLOURS, _head

START = datetime.datetime(2026, 1, 1, 12)
HOUR = datetime.timedelta(hours=1)
STEPS = setup.STEPS + calibration.STEPS + analysis.STEPS + comparison.STEPS
KERNEL = sorted(
    {r["architecture"] for r in numpys_blas() if r.get("internal_api") == "openblas"}
)[0]
"""This process's BLAS kernel, NumPy's, so that the direct calls run under the job's:
the chain's numbers depend on NumPy's and SciPy's OpenBLAS, which take one kernel.
Another OpenBLAS of this process, e.g. OpenCV's, may take another, but no step of
the chain computes with it."""
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
recipe_tolerance = {{}}
variant = "circulation"
circulation = {{minimum_readings = 3}}
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
[runs.r2.data]
folder = "photos-r2"
[runs.r2.protocols]
overflow = "overflow-r2.csv"
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
        (root / f"overflow-{run}.csv").write_text(
            "image_time,weight_g,status\n"
            + "".join(
                f"{START + h * HOUR},{1000 + max(0.0, h + 1) * grow / 4},Verified\n"
                for h in (-1.5, -1.25, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
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
        f"temperature_gradient_celsius\n{START - 3 * HOUR},1.013,20.0,0.0,0.0\n"
        f"{START + 24 * HOUR},1.013,20.0,0.0,0.0\n"
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


def _direct(root: Path, selection: Selection) -> dict:
    """The mass maps of every frame by the methods called directly."""
    tables = tomllib.loads((root / "config.toml").read_text())

    def settings(kind, section):
        return settings_from_table(kind, table(tables, section))

    material = settings(calibration.MaterialSettings, "material")
    statistics = settings(darsia.RunStatisticsSettings, "run_statistics")
    reading = calibration.sand_reading(settings(calibration.ReadingSettings, "reading"))
    rules = settings(darsia.PhaseRuleSettings, "phase_rules")
    runs, frames = {}, []
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
        model = darsia.read_phase_model(
            root / "model", 0.5, canvas=rig.canvas, accept_canvas=True
        )
        # Each frame's previous image: the run's acquisition before it.
        previous = [
            calibration.previous_acquisition(
                root / frame.session / frame.name, [root / frame.session]
            )
            for frame in selection.frames[run]
        ]
        start = experiment.experiment_start
        dates = [
            experiment.get_datetime(root / frame.session / frame.name)
            for frame in run_frames
        ]
        hours = [(date - start).total_seconds() / 3600 for date in dates]
        overflow = settings(darsia.OverflowSettings, "overflow")
        readings = darsia.OverflowProtocol(root / f"overflow-{run}.csv").readings(
            start, ["Verified"]
        )
        volume = darsia.displaced_volume(*readings, overflow)
        volumes = [darsia.measured_reference(volume, h) for h in hours]
        probabilities = [
            darsia.phase_probabilities(
                model,
                image.img,
                rig.baseline.img,
                None if before is None else rig.read_image(before).img,
                pixel_size_mm=grid.pixel_mm,
            )
            for image, before in zip(images, previous)
        ]
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
            volumes,
            settings(darsia.GasAllocationSettings, "gas_allocation"),
            grid.chart,
            gas_threshold=model.gas_threshold,
        )
        maps = [dict(zip(("labels", "rules", "saturation", "co2"), m)) for m in maps]
        runs[run] = (grid, experiment, images, maps, dates)
        for index, (image, frame_maps, date) in enumerate(zip(images, maps, dates)):
            h = (date - start).total_seconds() / 3600
            co2 = calibration.co2_state(grid, experiment, date)
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
            frames.append(
                darsia.CalibrationFrame(
                    run,
                    index,
                    h,
                    h,
                    experiment.injection_protocol.injected_mass(time=h) * 1000,
                    histograms,
                    gas,
                    None,
                    volumes[index],
                    volumes[index] is not None,
                )
            )
    fit = darsia.fit_calibration(
        frames,
        ["r1", "r2"],
        settings(darsia.CalibrationFitSettings, "calibration_fit"),
        {1: 0.1, 2: 0.2},
        [1, 2],
    )
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
                calibration.co2_state(grid, experiment, date),
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
            co2 = calibration.co2_state(grid, experiment, date)
            for row in sums(made[(run, frame)], co2):
                case = f"{frame.session}/{frame.name}"
                rows.append({**row, "run": run, "case_id": case, "elapsed_h": h})
        domain_mass[run] = rows
    wasserstein = settings(WassersteinSettings, "wasserstein")
    chart = settings(comparison.DistanceSettings, "distances").chart_labels
    found, reference, scales = {}, None, None
    for run, (grid, experiment, images, maps, dates) in runs.items():
        start = experiment.experiment_start
        hours = [(date - start).total_seconds() / 3600 for date in dates]
        if reference is None:  # the first run's sand map
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
                calibration.co2_state(grid, experiment, start),
                grid.rig.depth.img,
                grid.rig.facies.dimensions,
                domains,
            )
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
    monkeypatch.chdir(project)
    one = Selection({"r1": selection.frames["r1"]}, selection.sessions)
    early = setup.STEPS + calibration.STEPS[:3]
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


def test_a_frames_previous_image_is_the_runs_acquisition_before_it(tmp_path):
    """A frame's previous image is the run's photo just before it, selected or not:
    the photo before it in its session, else the last photo of the session before;
    none for the run's first photo."""
    first, second = tmp_path / "first", tmp_path / "second"
    for folder, names in (
        (first, ("a1.ARW", "a2.ARW", "a3.ARW")),
        (second, ("b1.ARW",)),
    ):
        folder.mkdir()
        for name in names:
            (folder / name).touch()
    (first / "a2.json").touch()  # no photo
    sessions = [first, second]
    assert (
        calibration.previous_acquisition(first / "a3.ARW", sessions) == first / "a2.ARW"
    )
    assert (
        calibration.previous_acquisition(second / "b1.ARW", sessions)
        == first / "a3.ARW"
    )
    assert calibration.previous_acquisition(first / "a1.ARW", sessions) is None


def test_a_photo_the_configuration_excludes_is_no_previous_image(tmp_path):
    """A photo the configuration excludes (its blacklist, as the light screening
    excluded it) is passed over: a frame's previous image is the kept photo
    before it, in its session or the session before."""
    first, second = tmp_path / "first", tmp_path / "second"
    for folder, names in (
        (first, ("a1.ARW", "a2.ARW", "a3.ARW")),
        (second, ("b1.ARW", "b2.ARW")),
    ):
        folder.mkdir()
        for name in names:
            (folder / name).touch()
    (tmp_path / "imaging.csv").write_text("path,datetime\nfirst/a1.ARW,2026-01-01\n")
    (tmp_path / "blacklist.csv").write_text(
        "path\nfirst/a2.ARW\nfirst/a3.ARW\nsecond/b1.ARW\n"
    )
    protocol = darsia.ImagingProtocol(
        tmp_path / "imaging.csv", tmp_path / "blacklist.csv"
    )
    sessions = [first, second]

    def previous(photo):
        return calibration.previous_acquisition(
            photo, sessions, protocol.is_blacklisted
        )

    assert previous(second / "b2.ARW") == first / "a1.ARW"
    assert previous(first / "a3.ARW") == first / "a1.ARW"
    assert previous(first / "a1.ARW") is None
    assert (  # no exclusion by default
        calibration.previous_acquisition(second / "b2.ARW", sessions)
        == second / "b1.ARW"
    )


def test_a_phase_support_makes_the_phase_step_read_inside_it(tmp_path, monkeypatch):
    """With the run's phase support (``[labeling.sand_map] phase_support``), the
    phase step reads its current image, previous image and baseline inside it, the
    rest unknown (NaN), as our production's phase step read them; without one, as
    they are. The support is a boolean mask of the grid, or none."""
    shape = (4, 5)
    support = np.ones(shape, bool)
    support[:, :2] = False
    path = tmp_path / "support.npz"
    darsia.Image(support, dimensions=[0.1, 0.1]).save(path)
    tables = {"labeling": {"sand_map": {"phase_support": str(path)}}}
    np.testing.assert_array_equal(
        setup.grid_mask(tables, "phase_support", shape + (3,)), support
    )
    assert setup.grid_mask({}, "phase_support", shape) is None
    with pytest.raises(ValueError, match="phase_support is a boolean mask"):
        setup.grid_mask(tables, "phase_support", (5, 5))
    images = [np.full(shape + (3,), v, np.float32) for v in (0.2, 0.3, 0.4)]
    seen = []

    def features(model, *given, **kwargs):
        seen.append(given)

    monkeypatch.setattr(darsia, "phase_probabilities", features)
    monkeypatch.setattr(calibration._Probabilities, "_previous", lambda s, i: images[2])
    current = SimpleNamespace(img=images[0])
    ctx = SimpleNamespace(read=lambda kind, reader, frame: current)
    baseline = SimpleNamespace(img=images[1])
    grid = SimpleNamespace(rig=SimpleNamespace(baseline=baseline), pixel_mm=0.5)
    for given in (support, None):
        calibration._Probabilities(ctx, None, grid, ["frame"], 2, None, given)[0]
    inside, plain = seen
    for read, image in zip(inside, images):
        assert np.isnan(read[~support]).all()
        np.testing.assert_array_equal(read[support], image[support])
    assert all(read is image for read, image in zip(plain, images))


def test_a_data_folder_outside_the_project_gives_its_runs_protocols(tmp_path):
    """The imaging protocol of a data folder outside the project (``../raw``)
    is among the files of the run whose photos it holds, not of another run."""
    project = tmp_path / "project"
    imaging = {f"../raw/{r}": f"../raw/{r}/imaging.csv" for r in ("r1", "r2")}
    tables = {"protocols": {"imaging": imaging, "injection": "injection.csv"}}
    folders = [normal_path("../raw/r1/day1", project)]
    files = setup.protocol_files(tables, project, folders)
    assert files == ["../raw/r1/imaging.csv", "injection.csv"]
    assert setup.protocol_files(tables, project, [project / folders[0]]) == files


def test_the_experiment_takes_the_injections_reference_density(tmp_path, monkeypatch):
    """A run's experiment reads its injection protocol with the configured
    reference density of the flow controller, by which volume rates become mass."""
    project = tmp_path / "project"
    config, _ = _project(project, "float64")
    monkeypatch.chdir(project)
    (project / "injection-sccm.csv").write_text(
        "id,location_x,location_y,start,end,rate_sccm\n"
        f"1,0.225,0.15,{START - HOUR},{START + 2 * HOUR},2.0\n"
    )
    config.write_text(
        config.read_text().replace(
            'injection = "injection.csv"',
            'injection = "injection-sccm.csv"\ninjection_reference_density = 2.5',
        )
    )
    loaded = FluidFlowerConfig([config], require_data=False, require_results=False)
    made = make_experiment(loaded).injection_protocol
    wanted = darsia.InjectionProtocol(
        project / "injection-sccm.csv", reference_density=2.5
    )
    assert made.injected_mass(time=1.5) == wanted.injected_mass(time=1.5) > 0


def test_a_comparison_time_keeps_its_digits():
    """A comparison time's part is named by the time as configured, every digit
    kept, and read back from the settings; a time given twice is refused."""
    named = comparison.named_times([72.33333, 72.333334, 3])
    assert named == {"72.33333h": 72.33333, "72.333334h": 72.333334, "3.0h": 3.0}
    with pytest.raises(ValueError, match="configured twice"):
        comparison.named_times([3, 3.0])
