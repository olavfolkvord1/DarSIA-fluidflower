"""The workflow shares one complete private prepass, including on resume."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.multiphase import gas_model_cache
from darsia.multiphase.gas_model_cache import (
    GasModelCache,
    GasModelFrame,
    GasModelReader,
)
from darsia.multiphase.gas_zone import GasZone
from darsia.multiphase.gas_zone_cache import GasZoneCache
from darsia.presets.workflows.calibration import steps
from darsia.utils.locations import Frame


@pytest.mark.parametrize("resume", [0, 1])
@pytest.mark.parametrize("fail_publish", [False, True])
@pytest.mark.parametrize("changed_provenance", [False, True])
def test_phase_workflow_shares_prepass_retains_models_and_cleans_zones(
    monkeypatch, tmp_path, resume, fail_publish, changed_provenance
):
    frames = tuple(Frame("session", str(index)) for index in range(3))
    shape = (3, 4)
    chart = np.zeros(shape, bool)
    chart[0, 0] = True
    rig = SimpleNamespace(
        baseline=SimpleNamespace(img=np.zeros((*shape, 3), np.float32)),
        depth=SimpleNamespace(img=np.full(shape, 0.012, np.float32)),
        canvas=None,
        facies=SimpleNamespace(img=np.ones(shape, np.uint8)),
        load_experiment=lambda _: None,
    )
    protocol, positions = object(), object()
    experiment = SimpleNamespace(
        get_datetime=lambda value: datetime(2024, 1, 1) + timedelta(hours=int(value)),
        injection_protocol=protocol,
        is_blacklisted=lambda _: False,
    )
    grid = SimpleNamespace(
        rig=rig,
        domain=np.ones(shape, bool),
        pixel_area_mm2=1.0,
        permeability={1: 1.0},
        geometry=SimpleNamespace(
            pore_volume_m3=np.ones(shape, np.float32),
            swi=np.zeros(shape, np.float32),
        ),
        chart=chart,
        positions=positions,
    )
    model = SimpleNamespace(gas_threshold=0.5)
    settings = {
        "phase_rules": SimpleNamespace(pitch_stride=1),
        "phase_model": SimpleNamespace(pixel_size_mm=None, accept_canvas=False),
        "material": object(),
        "gas_allocation": SimpleNamespace(growth_only_from_injecting_ports=False),
    }
    seen = {
        "models": [],
        "allocations": [],
        "writes": [],
        "published": [],
        "directories": [],
    }

    class Cache(GasModelCache):
        def __enter__(self):
            result = super().__enter__()
            seen["directories"].append(self._directory)
            return result

    class ZoneCache(GasZoneCache):
        def __enter__(self):
            result = super().__enter__()
            seen["directories"].append(self._directory)
            return result

    class Probabilities:
        def __init__(self, *args):
            self.cleared = False

        def clear(self):
            self.cleared = True

        def __getitem__(self, index):
            raise AssertionError("Allocation must use the prepared model state")

    def models(probabilities, dates, *_args, **kwargs):
        assert not probabilities.cleared and kwargs["positions"] is positions
        assert len(dates) == len(frames)
        for index in range(len(frames)):
            seen["models"].append(index)
            labels = np.full(shape, index, np.uint8)
            yield GasModelFrame(
                np.full(shape, 0.5, np.float32), labels, labels.copy(), labels.copy()
            )

    def build_zones(values, dates, *_args, **kwargs):
        assert kwargs["chart"] is chart and kwargs["positions"] is positions
        assert kwargs["depth_m"] is rig.depth.img
        for _, value in zip(dates, values, strict=True):
            assert isinstance(value, GasModelFrame)
            yield GasZone(np.ones(shape, bool), {}, {"index": int(value.labels[0, 0])})

    @dataclass
    class Record:
        volume: float

    def allocate(probabilities, *args, **kwargs):
        assert probabilities.cleared
        assert "fixed_zone" not in kwargs
        assert "history_maps" not in kwargs
        cache = kwargs["model_frames"]
        assert len(cache) == len(frames) and kwargs["start"] == resume
        seen["allocations"].append((cache, kwargs["zones"]))
        volumes = args[9]
        for index in range(resume, len(frames)):
            state = cache[index]
            assert kwargs["zones"][index].diagnostics == {"index": index}
            assert not kwargs["zones"][index].contour.flags.writeable
            assert not state.labels.flags.writeable and state.labels[0, 0] == index
            labels = state.labels.copy()
            labels[:] = volumes[index]
            kwargs["record"](index, Record(volumes[index]))
            yield labels, state.rules, np.zeros(shape), np.ones(
                shape, bool
            ), state.before

    def volumes(_folder, held_out):
        return {
            steps.frame_key(frame): SimpleNamespace(volume_m3=3 if held_out else 2)
            for frame in frames
        }

    def write(_folder, *arrays, held_out):
        seen["writes"].append((held_out, int(arrays[0][0, 0])))

    def read(kind, reader):
        if kind == "rig":
            return rig
        if kind == "phase-model":
            return model
        assert kind == "overflow"
        return reader(None)

    def publish(writer, *, frame, measurements):
        if fail_publish:
            raise OSError("injected publication failure")
        writer(None)
        seen["published"].append((frame, measurements))

    ctx = SimpleNamespace(
        task=SimpleNamespace(run="run", frames=frames),
        settings=settings,
        tables={},
        photo=lambda run, frame: frame.name,
        read=read,
        done=lambda frame: frames.index(frame) < resume,
        publish=publish,
        locations=SimpleNamespace(sessions={"run": {"session": tmp_path}}),
        revision="full-source-audit-revision",
    )
    identity = {
        "frame_keys": ["/".join(frame) for frame in frames],
        "dates": [experiment.get_datetime(frame.name).isoformat() for frame in frames],
        "input_tokens": {"rig/run": "accepted"},
        "model": {"threshold": 0.5},
        "settings": {"rules": "fixed"},
        "backend": {"device": "cpu"},
        "source": {"model/rules": "same"},
    }
    monkeypatch.setattr(gas_model_cache, "_local_cache", lambda: tmp_path / "cache")
    calls = 0

    def identified(*_):
        nonlocal calls
        calls += 1
        if calls > 1 and changed_provenance:
            return {**identity, "source": {"model/rules": "changed"}}
        return identity

    monkeypatch.setattr(steps, "model_cache_identity", identified)
    monkeypatch.setattr(steps, "previous_acquisition", lambda *_: None)
    monkeypatch.setattr(steps, "GasModelCache", Cache)
    monkeypatch.setattr(steps, "GasZoneCache", ZoneCache)
    monkeypatch.setattr(steps, "_Probabilities", Probabilities)
    monkeypatch.setattr(steps, "make_experiment", lambda _: experiment)
    monkeypatch.setattr(steps, "run_settings", lambda *_args: None)
    monkeypatch.setattr(steps, "run_grid", lambda *_args: grid)
    monkeypatch.setattr(steps, "grid_mask", lambda *_args: None)
    monkeypatch.setattr(steps, "window_frames", lambda _: 1)
    monkeypatch.setattr(darsia, "gas_model_frames", models, raising=False)
    monkeypatch.setattr(darsia, "gas_model_zones", build_zones, raising=False)
    monkeypatch.setattr(darsia, "free_gas", allocate)
    monkeypatch.setattr(darsia, "held_out_runs", lambda _: ["held"])
    monkeypatch.setattr(darsia, "read_frame_volumes", volumes)
    monkeypatch.setattr(darsia, "write_phase_maps", write)
    if changed_provenance:
        with pytest.raises(ValueError, match="provenance changed"):
            steps._phase_maps(ctx)
        model_path, zone_path = seen["directories"]
        assert not zone_path.exists() and model_path.is_dir()
        assert seen["models"] == [0, 1, 2] and not seen["allocations"]
        with pytest.raises(ValueError, match="not complete"):
            GasModelReader.open(model_path, identity=identity)
        return
    if fail_publish:
        with pytest.raises(OSError, match="publication failure"):
            steps._phase_maps(ctx)
    else:
        steps._phase_maps(ctx)
        assert len(seen["published"]) == len(frames) - resume
        assert seen["writes"] == [(None, 2), ("held", 3)] * (len(frames) - resume)
        for _, measurements in seen["published"]:
            assert measurements["gas_allocation"] == {
                "primary": {"volume": 2},
                "without": {"held": {"volume": 3}},
            }
    assert seen["models"] == [0, 1, 2]
    assert len(seen["allocations"]) == 2
    assert all(a is b for a, b in zip(*seen["allocations"], strict=True))
    model_path, zone_path = seen["directories"]
    assert model_path.is_dir() and not zone_path.exists()
    manifest = json.loads((model_path / "manifest.json").read_bytes())
    assert manifest["status"] == "complete" and manifest["stored"] == 3
    assert manifest["producer_revision"] == ctx.revision
    reopened = GasModelReader.open(model_path, identity=identity)
    assert [int(state.labels[0, 0]) for state in reopened] == [0, 1, 2]
