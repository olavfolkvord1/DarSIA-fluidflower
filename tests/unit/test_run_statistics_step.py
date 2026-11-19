"""Independent frame statistics, their stored indices and calibration inputs."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.jobs.step import Each, Order, Selection
from darsia.presets.workflows.calibration import steps
from darsia.utils.locations import Frame
from tests.unit.test_calibration_fit import NO_SWI, RUNS, SETTINGS, _run


class _Reading:
    sands = (1, 2)

    def __init__(self, channel):
        self.channel = channel

    def __call__(self, image, baseline, support, dissolved, facies):
        return image[..., self.channel].copy()


class _StatisticsContext:
    def __init__(self, inputs, output, frames, selected, settings, grid):
        self.inputs, self.output, self.grid = inputs, output, grid
        self.task = SimpleNamespace(run="run", frames=frames)
        self.selection = Selection({"run": selected}, {})
        self.settings = settings
        self.tables = {}
        self.published = []
        self.images_read = []

    def folder(self, frame):
        return self.output / frame.session / frame.name

    def read(self, kind, reader, *, frame=None):
        if kind == "rig":
            return self.grid
        folder = self.inputs / kind
        if frame is not None:
            folder /= Path(frame.session) / frame.name
        if kind == "corrected-image":
            self.images_read.append(frame)
        return reader(folder)

    def photo(self, run, frame):
        return self.selection.frames[run].index(frame) + 1

    def done(self, frame):
        return (self.folder(frame) / "statistics.json").exists()

    def publish(self, writer, *, frame):
        folder = self.folder(frame)
        folder.mkdir(parents=True)
        writer(folder)
        self.published.append(frame)


def _statistics_inputs(tmp_path, monkeypatch, dtype, different_before=False):
    shape = (8, 6)
    facies = np.ones(shape, np.uint8)
    facies[:, 3:] = 2
    sand_map = darsia.Image(facies, dimensions=[0.04, 0.03])
    areas = darsia.ScalarImage(np.full(shape, 25e-6, dtype), dimensions=[0.04, 0.03])
    domain = np.ones(shape, bool)
    geometry = darsia.run_geometry(
        sand_map,
        areas.img,
        np.full(shape, 0.01, dtype),
        np.full(shape, 0.4, dtype),
        np.full(shape, 0.1, dtype),
        domain,
    )
    grid = SimpleNamespace(
        rig=SimpleNamespace(
            pixel_area=areas,
            facies=sand_map,
            baseline=SimpleNamespace(img=np.zeros((*shape, 3), dtype)),
        ),
        geometry=geometry,
        domain=domain,
        facies=facies,
        optical_pitch=darsia.optical_pitch_m(areas, domain),
    )
    co2 = SimpleNamespace(
        solubility_co2=np.full(shape, 1.7, dtype),
        density_gaseous_co2=np.full(shape, 1.8, dtype),
    )
    experiment = SimpleNamespace(
        injection_protocol=SimpleNamespace(injected_mass=lambda time: time * 1e-3),
        get_datetime=lambda photo: photo,
        time_since_start=lambda date: date,
    )
    monkeypatch.setattr(steps, "run_grid", lambda rig, material: rig)
    monkeypatch.setattr(steps, "run_settings", lambda *args: None)
    monkeypatch.setattr(steps, "make_experiment", lambda config: experiment)
    monkeypatch.setattr(steps, "co2_state", lambda *args: co2)
    monkeypatch.setattr(steps, "response_reading", lambda *args: _Reading(0))
    monkeypatch.setattr(steps, "native_reading", lambda *args: _Reading(1))
    frames = tuple(
        Frame(f"session-{index // 2}", f"photo-{index % 2}") for index in range(6)
    )
    volumes = {}
    for index, frame in enumerate(frames):
        image = np.full((*shape, 3), 0.2, dtype)
        image[..., 0] = 0.4 + index / 20
        image[5, 1] = np.nan
        image_folder = tmp_path / "corrected-image" / frame.session / frame.name
        image_folder.mkdir(parents=True)
        darsia.Image(image, dimensions=[0.04, 0.03]).save(image_folder / "image.npz")
        before = np.full(shape, darsia.PhaseLabel.DISSOLVED, np.uint8)
        before[0] = darsia.PhaseLabel.GAS
        rules = np.zeros(shape, np.uint8)
        saturation = np.zeros(shape, dtype)
        saturation[0] = 0.8
        phase_folder = tmp_path / "phase-maps" / frame.session / frame.name
        phase_folder.mkdir(parents=True)
        darsia.write_phase_maps(phase_folder, before, rules, saturation, domain, before)
        labels = before.copy()
        labels[0] = darsia.PhaseLabel.DISSOLVED
        alternative = before.copy()
        if different_before:
            alternative[2:4] = darsia.PhaseLabel.WATER
        darsia.write_phase_maps(
            phase_folder,
            labels,
            rules,
            np.zeros(shape, dtype),
            domain,
            alternative,
            held_out="donor",
        )
        volumes[steps.frame_key(frame)] = darsia.FrameVolume(None, "model")
    (tmp_path / "overflow").mkdir()
    darsia.write_frame_volumes(
        tmp_path / "overflow", volumes, held_out={"donor": volumes}
    )
    settings = {
        "material": None,
        "run_statistics": darsia.RunStatisticsSettings(5.0, 0.0, 1.2, 32768),
        "unknown_mass": darsia.UnknownMassSettings(20.0, 32768),
    }
    return frames, settings, grid


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_statistics_chunks_and_resume_keep_each_frames_bytes(
    tmp_path, monkeypatch, dtype
):
    frames, settings, grid = _statistics_inputs(tmp_path / "inputs", monkeypatch, dtype)
    assert steps.RUN_STATISTICS.order is Order.FRAMES
    assert steps.RUN_STATISTICS.stored is Each.FRAME
    whole = _StatisticsContext(
        tmp_path / "inputs", tmp_path / "whole", frames, frames, settings, grid
    )
    steps.RUN_STATISTICS.run(whole)
    output = tmp_path / "chunked"
    chunks = (frames[4:], frames[:2], frames[2:4])
    for chunk in chunks:
        ctx = _StatisticsContext(
            tmp_path / "inputs", output, chunk, frames, settings, grid
        )
        steps.RUN_STATISTICS.run(ctx)
        assert ctx.published == list(chunk)
        # A retry reads and publishes none of the completed frames.
        ctx.published.clear()
        ctx.images_read.clear()
        steps.RUN_STATISTICS.run(ctx)
        assert ctx.published == ctx.images_read == []
    for frame in frames:
        path = Path(frame.session) / frame.name / "statistics.json"
        assert (output / path).read_bytes() == (whole.output / path).read_bytes()
        assert darsia.read_frames(output / path.parent)[0].index == 0
    # Only the absent frame of a partially completed chunk is read and published.
    resumed = _StatisticsContext(
        tmp_path / "inputs", output, frames[:2], frames, settings, grid
    )
    missing = frames[1]
    (resumed.folder(missing) / "statistics.json").unlink()
    resumed.folder(missing).rmdir()
    steps.RUN_STATISTICS.run(resumed)
    assert resumed.images_read == resumed.published == [missing]


@pytest.mark.parametrize("different_before", [False, True])
def test_signal_is_reused_only_for_variants_with_the_same_before(
    tmp_path, monkeypatch, different_before
):
    frames, settings, grid = _statistics_inputs(
        tmp_path / "inputs", monkeypatch, np.float32, different_before
    )
    original, calls = steps.read_signal, []

    def read_signal(*args):
        calls.append(args[2]["before"].copy())
        return original(*args)

    monkeypatch.setattr(steps, "read_signal", read_signal)
    ctx = _StatisticsContext(
        tmp_path / "inputs", tmp_path / "output", frames[:1], frames, settings, grid
    )
    steps.RUN_STATISTICS.run(ctx)
    assert len(calls) == (2 if different_before else 1)
    if different_before:
        assert not np.array_equal(calls[0], calls[1])
    # The guarded variant has exactly the histograms of an independent reading.
    frame = frames[0]
    image = ctx.read("corrected-image", steps.read_image, frame=frame).img
    maps = ctx.read(
        "phase-maps",
        lambda folder: darsia.read_phase_maps(folder, "donor"),
        frame=frame,
    )
    statistics = settings["run_statistics"]
    signal, measured = original(grid, image, maps, _Reading(0), statistics, _Reading(1))
    capacity = darsia.dissolved_capacity(
        grid.geometry.pore_volume_m3,
        np.full(grid.facies.shape, 1.7, np.float32),
        maps["saturation"],
    )
    expected = darsia.signal_histograms(
        signal,
        measured
        & (maps["labels"] == darsia.PhaseLabel.DISSOLVED)
        & ~darsia.model_gas(maps["labels"], maps["rules"]),
        grid.facies,
        grid.geometry.counted,
        capacity,
        _Reading.sands,
        statistics,
    )
    stored = darsia.read_frames(ctx.folder(frame), "donor")[0]
    for sand, histogram in expected.items():
        assert stored.dissolved[sand].relative.tobytes() == histogram.relative.tobytes()
        assert stored.dissolved[sand].weight.tobytes() == histogram.weight.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("selection_mode", ["all", "reversed", "subset"])
def test_frame_results_restore_the_selected_indices_and_the_exact_fit(
    tmp_path, monkeypatch, dtype, selection_mode
):
    selected, reference, folded = {}, [], []
    folders = {}
    for seed, run in enumerate(RUNS):
        frames = [
            dataclasses.replace(
                frame,
                column=darsia.ColumnEstimate(
                    np.array([0.2], dtype),
                    np.array([1.05], dtype),
                    np.array([1e-5 * (seed + 1)], dtype),
                ),
            )
            for frame in _run(run, seed, dtype)
        ]
        # One frame per result; duplicate photo names across sessions are distinct.
        selected[run] = tuple(
            Frame(f"session-{i // 2}", f"photo-{i % 2}") for i in range(len(frames))
        )
        alternative = [
            dataclasses.replace(
                frame,
                gas=dataclasses.replace(
                    frame.gas, free_gas_g=frame.gas.free_gas_g * 0.9
                ),
            )
            for frame in frames
        ]
        aggregate = tmp_path / "aggregate" / run
        aggregate.mkdir(parents=True)
        darsia.write_frames(aggregate, run, frames, dtype, {"b": alternative})
        old = darsia.read_frames(aggregate)
        old_fold = darsia.read_frames(aggregate, "b")
        assert [frame.index for frame in old] == list(range(len(frames)))
        for frame, value, other in reversed(
            list(zip(selected[run], frames, alternative))
        ):
            folder = tmp_path / "per-frame" / run / frame.session / frame.name
            folder.mkdir(parents=True)
            darsia.write_frames(
                folder,
                run,
                [dataclasses.replace(value, index=0)],
                dtype,
                {"b": [dataclasses.replace(other, index=0)]},
            )
            folders[run, frame] = folder
        positions = list(range(len(frames)))
        if selection_mode == "reversed":
            positions.reverse()
        elif selection_mode == "subset":
            positions = positions[::2]
        selected[run] = tuple(selected[run][index] for index in positions)
        reference += [
            dataclasses.replace(old[index], index=new)
            for new, index in enumerate(positions)
        ]
        folded += [
            dataclasses.replace(old_fold[index], index=new)
            for new, index in enumerate(positions)
        ]
    selection = Selection(selected, {})
    fit_runs = steps.FitRuns(RUNS, ("b",))
    settings = {
        "calibration_runs": fit_runs,
        "calibration_fit": SETTINGS,
        "run_statistics": darsia.RunStatisticsSettings(5.0, 0.0, 1.2, 32768),
        "material": steps.MaterialSettings({}, ()),
    }
    collected = []

    def read(kind, reader, *, run, frame):
        assert kind == "run-statistics"
        values = reader(folders[run, frame])
        collected.extend(values)
        return values

    for part, wanted in (("fit", reference), ("fold-b", folded)):
        collected.clear()
        output = tmp_path / part
        output.mkdir()
        ctx = SimpleNamespace(
            settings=settings,
            selection=selection,
            task=SimpleNamespace(part=part),
            read=read,
            publish=lambda writer: writer(output),
        )
        steps.CALIBRATION.run(ctx)
        # The reader reconstructs all original fields, indices and array bytes.
        assert len(collected) == len(wanted)
        for actual, expected in zip(collected, wanted):
            assert (actual.run, actual.index) == (expected.run, expected.index)
            assert json.dumps(
                dataclasses.asdict(actual),
                default=lambda value: value.tolist(),
                sort_keys=True,
            ) == json.dumps(
                dataclasses.asdict(expected),
                default=lambda value: value.tolist(),
                sort_keys=True,
            )
            for sand, histogram in expected.dissolved.items():
                found = actual.dissolved[sand]
                assert found.relative.dtype == histogram.relative.dtype
                assert found.relative.tobytes() == histogram.relative.tobytes()
                assert found.weight.tobytes() == histogram.weight.tobytes()
            for name in ("relative", "ratio", "weight"):
                found = getattr(actual.column, name)
                original = getattr(expected.column, name)
                assert found.dtype == original.dtype
                assert found.tobytes() == original.tobytes()
        if part == "fit":
            fit = darsia.fit_calibration(wanted, RUNS, SETTINGS, *NO_SWI)
            predictions = []
        else:
            fit, predictions = darsia.calibration_fold(
                wanted, RUNS, SETTINGS, *NO_SWI, "b"
            )
        record = darsia.read_calibration(output)
        expected = tmp_path / f"expected-{part}"
        expected.mkdir()
        darsia.write_calibration(
            expected,
            fit,
            darsia.mass_calibrations(fit, settings["run_statistics"]),
            predictions,
        )
        assert record == darsia.read_calibration(expected)
    # Reusing one stored frame in another selection assigns its new ordinal.
    stored = folders[RUNS[0], selected[RUNS[0]][5]]
    assert darsia.read_frames(stored)[0].index == 0
    assert darsia.read_frames(stored, first_index=3)[0].index == 3
