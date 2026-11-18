"""Persistent caches require complete, exact model provenance and file bytes."""

import gc
import json
import pickle
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.multiphase import gas_model_cache as cache_module
from darsia.multiphase.gas_model_cache import (
    GasModelCache,
    GasModelFrame,
    GasModelReader,
)
from darsia.presets.workflows.calibration import model_cache as adapter
from darsia.utils.locations import Frame


def identity(count=2):
    return {
        "frame_keys": [f"session/{i}" for i in range(count)],
        "dates": [f"2024-01-01T0{i}:00:00" for i in range(count)],
        "input_tokens": {"rig/run": "accepted-rig", "phase-model": "accepted-model"},
        "model": {"pixel_size_mm": 0.1},
        "settings": {"phase_rules": {"window": 3}},
        "backend": {"device": "cpu", "blas": "Haswell"},
        "source": {"model-and-rules": "frozen-method-identity"},
    }


def frame(value=0.25, dtype=np.float32):
    p = np.full((3, 4), value, dtype)
    p[0, 0] = -0.0
    p[0, 1] = np.nan
    labels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    return GasModelFrame(p, labels, labels + 1, labels + 2)


@pytest.fixture(autouse=True)
def local_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cache_module, "_local_cache", lambda: tmp_path / "local-app-cache"
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_retention_survives_context_owner_gc_and_later_allocation_error(dtype):
    expected = [frame(0.25, dtype), frame(0.75, dtype)]
    with pytest.raises(RuntimeError, match="allocation failed"):
        with GasModelCache(
            2, retain=True, identity=identity(), revision="producer-before-gas-fix"
        ) as cache:
            list(cache.store(expected))
            directory = cache.retained_path
            reader = cache.reader()
            raise RuntimeError("allocation failed")
    del cache
    gc.collect()
    assert directory.is_dir()
    manifest_bytes = (directory / "manifest.json").read_bytes()
    reopened = GasModelReader.open(directory, identity=identity(), kept=1)
    assert json.loads(manifest_bytes)["producer_revision"] == "producer-before-gas-fix"
    assert not reopened._cached
    for actual, wanted in zip(reopened, expected, strict=True):
        for name in wanted.__dataclass_fields__:
            assert getattr(actual, name).tobytes() == getattr(wanted, name).tobytes()
            assert not getattr(actual, name).flags.writeable
    assert (
        pickle.loads(pickle.dumps(reader))[0].labels.tobytes()
        == expected[0].labels.tobytes()
    )
    assert (directory / "manifest.json").read_bytes() == manifest_bytes


def test_new_retained_invocations_never_overwrite_one_another():
    paths = []
    for _ in range(2):
        with GasModelCache(
            0, retain=True, identity=identity(0), revision="rev"
        ) as cache:
            list(cache.store([]))
            paths.append(cache.retained_path)
    assert paths[0] != paths[1] and all(p.is_dir() for p in paths)
    assert len(GasModelReader.open(paths[0], identity=identity(0))) == 0


def test_equal_acquisition_dates_keep_distinct_ordered_frame_keys():
    own = identity()
    own["dates"] = [own["dates"][0]] * 2
    with GasModelCache(2, retain=True, identity=own, revision="rev") as cache:
        list(cache.store([frame(0.25), frame(0.75)]))
        path = cache.retained_path
    reader = GasModelReader.open(path, identity=own)
    assert reader[0].gas_probability[1, 1] == 0.25
    assert reader[1].gas_probability[1, 1] == 0.75
    own["frame_keys"] = [own["frame_keys"][0]] * 2
    with pytest.raises(ValueError, match="frame_keys must be unique"):
        GasModelCache(2, retain=True, identity=own, revision="rev")


@pytest.mark.parametrize(
    "failure", ["short", "long", "consumer", "writer", "not_exhausted"]
)
def test_partial_retained_caches_remain_evidence_and_cannot_reopen(
    monkeypatch, failure
):
    with GasModelCache(2, retain=True, identity=identity(), revision="rev") as cache:
        directory = cache.retained_path
        if failure == "writer":

            def broken(*args):
                raise OSError("disk failure")

            monkeypatch.setattr(cache, "_write", broken)
        values = [frame()] * (
            1 if failure == "short" else 3 if failure == "long" else 2
        )
        storing = cache.store(values)
        if failure == "not_exhausted":
            next(storing)
            next(storing)
            assert (
                json.loads((directory / "manifest.json").read_bytes())["status"]
                == "building"
            )
            storing.close()
        elif failure == "consumer":
            next(storing)
            storing.close()
        else:
            with pytest.raises((ValueError, OSError)):
                list(storing)
    assert directory.is_dir()
    assert (
        json.loads((directory / "manifest.json").read_bytes())["status"] == "incomplete"
    )
    with pytest.raises(ValueError, match="not complete"):
        GasModelReader.open(directory, identity=identity())


@pytest.mark.parametrize("key", list(identity()))
def test_any_changed_model_provenance_refuses_reuse(key):
    with GasModelCache(2, retain=True, identity=identity(), revision="rev") as cache:
        list(cache.store([frame(), frame()]))
        directory = cache.retained_path
    changed = deepcopy(identity())
    if key in ("frame_keys", "dates"):
        changed[key].reverse()
    else:
        changed[key]["changed"] = True
    with pytest.raises(ValueError, match="provenance"):
        GasModelReader.open(directory, identity=changed)


@pytest.mark.parametrize(
    "damage", ["byte", "missing", "extra", "directory", "shape", "dtype", "count"]
)
def test_reopen_checks_every_file_and_manifest_contract(damage):
    with GasModelCache(2, retain=True, identity=identity(), revision="rev") as cache:
        list(cache.store([frame(), frame()]))
        directory = cache.retained_path
    path = directory / "00000001.npz"
    if damage == "byte":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif damage == "missing":
        path.unlink()
    elif damage == "extra":
        (directory / "extra.npz").write_bytes(b"extra")
    elif damage == "directory":
        path.unlink()
        path.mkdir()
    else:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        manifest[damage] = {"shape": [4, 3], "dtype": "<f8", "count": 3}[damage]
        (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        GasModelReader.open(directory, identity=identity())


def test_retention_requires_complete_identity_and_revision():
    for own, revision in [(None, "rev"), ({}, "rev"), (identity(), None)]:
        with pytest.raises(ValueError):
            GasModelCache(2, retain=True, identity=own, revision=revision)


def test_source_boundary_allows_only_allocation_body_changes(tmp_path):
    root = tmp_path / "source"
    gas = root / "multiphase/gas_allocation.py"
    workflow = root / "presets/workflows/calibration/steps.py"
    gas.parent.mkdir(parents=True)
    workflow.parent.mkdir(parents=True)
    gas.write_text(
        "from rules import phase_maps\ndef gas_model_frames(x):\n"
        " return phase_maps(x)\ndef free_gas(x):\n return x + 1\n"
    )
    workflow.write_text(
        "def _phase_maps(ctx):\n models = model(ctx)\n"
        " with cache() as c:\n  allocate(models, 1)\n"
    )
    original = adapter.model_source_identity(root)
    gas.write_text(gas.read_text().replace("x + 1", "x + 2"))
    workflow.write_text(
        workflow.read_text().replace("allocate(models, 1)", "allocate(models, 2)")
    )
    assert adapter.model_source_identity(root) == original
    gas.write_text(gas.read_text().replace("phase_maps(x)", "phase_maps(x + 1)"))
    assert adapter.model_source_identity(root) != original
    original = adapter.model_source_identity(root)
    workflow.write_text(workflow.read_text().replace("model(ctx)", "model(ctx, 2)"))
    assert adapter.model_source_identity(root) != original


def test_adapter_pins_tokens_previous_photo_and_keeps_actual_dates(
    monkeypatch, tmp_path
):
    frames = [Frame("session", "one"), Frame("session", "two")]
    inputs = [
        SimpleNamespace(kind=kind, key=kind, frame=None)
        for kind in ("rig", "phase-model")
    ]
    inputs += [
        SimpleNamespace(kind="corrected-image", key=f"corrected/{f.name}", frame=f)
        for f in frames
    ]
    inputs += [SimpleNamespace(kind="overflow", key="overflow", frame=None)]
    tokens = {i.key: "accepted-" + i.key for i in inputs}
    ctx = SimpleNamespace(
        task=SimpleNamespace(inputs=inputs, run="run"),
        token_of=lambda i: tokens[i.key],
        tables={"data": {}, "protocols": {}, "gas_allocation": {"window": 1}},
        settings={
            key: SimpleNamespace(value=1)
            for key in ("phase_rules", "material", "phase_model")
        },
        _files={"protocol": "file-sha"},
        _job={"environment": {"backend": "cpu"}},
        device="cpu",
    )
    monkeypatch.setattr(adapter, "model_source_identity", lambda: {"model": "sha"})
    monkeypatch.setattr(adapter.importlib.metadata, "version", lambda name: "1.0")
    previous = tmp_path / "previous.photo"
    previous.write_bytes(b"photo-before")
    dates = [datetime(2024, 1, 1, 1, 58), datetime(2024, 1, 1, 2, 3)]
    found = adapter.model_cache_identity(ctx, frames, dates, [None, previous])
    assert found["dates"] == [v.isoformat() for v in dates]
    assert "overflow" not in found["input_tokens"]
    ctx.tables["gas_allocation"]["window"] = 2
    assert adapter.model_cache_identity(ctx, frames, dates, [None, previous]) == found
    assert "expert_knowledge" not in found["settings"]
    ctx.tables["roi"] = [{"name": "top", "shape": "label", "label_index": 3}]
    ctx.tables["expert_knowledge"] = {"no_concentration_aq": ["top"]}
    limited = adapter.model_cache_identity(ctx, frames, dates, [None, previous])
    assert limited != found  # the run's expert knowledge limits its model frames
    assert limited["settings"]["expert_knowledge"] == {"no_concentration_aq": ["top"]}
    del ctx.tables["roi"], ctx.tables["expert_knowledge"]
    previous.write_bytes(b"photo-changed")
    assert adapter.model_cache_identity(ctx, frames, dates, [None, previous]) != found
    tokens["rig"] = "new-rig"
    assert (
        adapter.model_cache_identity(ctx, frames, dates, [None, previous])[
            "input_tokens"
        ]["rig"]
        == "new-rig"
    )
    inputs.pop(0)
    with pytest.raises(ValueError, match="one accepted rig"):
        adapter.model_cache_identity(ctx, frames, dates, [None, previous])
