"""Persistent caches preserve exact model provenance and file bytes."""

import gc
import json
import pickle
from copy import deepcopy

import numpy as np
import pytest

from darsia.multiphase import gas_model_cache as cache_module
from darsia.multiphase.gas_model_cache import GasModelCache, GasModelFrame, GasModelReader


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
