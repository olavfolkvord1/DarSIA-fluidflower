"""Private model-state storage preserves bytes and cannot accept a partial run."""

from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ProcessPoolExecutor
import gc
import multiprocessing
import pickle
import weakref
import zipfile

import numpy as np
import pytest

from darsia.multiphase.gas_model_cache import GasModelCache, GasModelFrame


def frame(dtype=np.float32, value=0.25):
    probability = np.full((3, 4), value, dtype=dtype)
    probability[0, 0] = -0.0
    probability[0, 1] = np.nan
    labels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    return GasModelFrame(probability, labels, labels + 1, labels + 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_model_cache_is_lossless_private_and_readonly(dtype):
    source = frame(dtype)
    originals = [getattr(source, name).copy() for name in source.__dataclass_fields__]
    with GasModelCache(1) as cache:
        directory = cache._directory
        yielded = list(cache.store([source]))
        assert yielded == [source] and yielded[0] is source
        with zipfile.ZipFile(directory / "00000000.npz") as archive:
            assert all(
                item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
            )
        source.gas_probability[:] = 0
        found = cache[0]
        for actual, expected in zip(
            (getattr(found, name) for name in found.__dataclass_fields__), originals
        ):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            assert not actual.flags.writeable
        assert cache[-1] is found
        with pytest.raises(ValueError, match="read-only"):
            found.labels[0, 0] = 0
        with pytest.raises(FrozenInstanceError):
            found.labels = found.labels.copy()
        with pytest.raises(IndexError):
            cache[1]
        with pytest.raises(TypeError):
            cache[0.5]
    assert not directory.exists()
    with pytest.raises(RuntimeError, match="not open"):
        cache[0]


def test_model_cache_evicts_before_loading_and_never_holds_all_frames(monkeypatch):
    with GasModelCache(6, kept=2) as cache:
        list(cache.store(frame(value=index) for index in range(6)))
        one, two = cache[0], cache[1]
        one_ref = weakref.ref(one.gas_probability)
        del one
        gc.collect()
        original = cache._load

        def load(index):
            assert one_ref() is None
            assert len(cache._cached) == 1
            return original(index)

        monkeypatch.setattr(cache, "_load", load)
        three = cache[2]
        assert cache[1] is two
        assert len(cache._cached) == 2 and len(cache) == 6
        assert three.gas_probability[1, 1] == 2


@pytest.mark.parametrize("produced", [1, 3])
def test_model_cache_count_mismatch_is_unreadable_and_cleaned(produced):
    with GasModelCache(2) as cache:
        directory = cache._directory
        with pytest.raises(ValueError, match="Expected 2 model frames"):
            list(cache.store(frame() for _ in range(produced)))
        with pytest.raises(RuntimeError, match="not open"):
            cache[0]
    assert not directory.exists()


def test_model_cache_requires_full_exhaustion_and_cannot_restart():
    with GasModelCache(1) as cache:
        storing = cache.store([frame()])
        next(storing)
        with pytest.raises(RuntimeError, match="not open"):
            cache[0]
        storing.close()
        with pytest.raises(RuntimeError, match="once"):
            list(cache.store([frame()]))


def test_model_cache_strict_zip_finishes_the_store():
    with GasModelCache(2) as cache:
        for _, _ in zip([0, 1], cache.store([frame(), frame()]), strict=True):
            pass
        assert cache[1].labels.shape == (3, 4)
    with GasModelCache(0) as empty:
        assert list(empty.store([])) == [] and len(empty) == 0
        with pytest.raises(IndexError):
            empty[0]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"gas_probability": np.zeros((3, 4), dtype=np.uint8)}, "floating dtype"),
        ({"gas_probability": np.zeros(12)}, "two-dimensional"),
        ({"before": np.zeros((3, 4), dtype=np.int32)}, "before map"),
        ({"labels": np.zeros((4, 3), dtype=np.uint8)}, "labels map"),
        ({"rules": np.zeros((3, 4), dtype=np.float32)}, "rules map"),
    ],
)
def test_model_cache_refuses_wrong_array_contract(change, message):
    with GasModelCache(1) as cache:
        with pytest.raises(ValueError, match=message):
            list(cache.store([replace(frame(), **change)]))


def test_model_cache_preserves_noncontiguous_values_and_rejects_precision_changes():
    source = frame()
    transposed = GasModelFrame(
        *(getattr(source, name).T for name in source.__dataclass_fields__)
    )
    with GasModelCache(1) as cache:
        list(cache.store([transposed]))
        assert (
            cache[0].gas_probability.tobytes() == transposed.gas_probability.tobytes()
        )
    with GasModelCache(2) as cache:
        with pytest.raises(ValueError, match="grid and precision"):
            list(cache.store([frame(np.float32), frame(np.float64)]))


def test_model_cache_cleanup_after_writer_failure(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("injected local write failure")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="local write failure"):
        with GasModelCache(1) as cache:
            directory = cache._directory
            list(cache.store([frame()]))
    assert not directory.exists()


def test_model_cache_private_files_cannot_gain_unknown_arrays():
    with GasModelCache(1) as cache:
        list(cache.store([frame()]))
        np.savez(cache._directory / "00000000.npz", unexpected=np.zeros(1))
        with pytest.raises(ValueError, match="array inventory"):
            cache[0]


def test_model_cache_context_cannot_be_reused_or_bypassed():
    cache = GasModelCache(1)
    with pytest.raises(RuntimeError, match="inside"):
        list(cache.store([frame()]))
    with cache:
        with pytest.raises(RuntimeError, match="only once"):
            cache.__enter__()
    with pytest.raises(RuntimeError, match="only once"):
        cache.__enter__()
    for count, kept in [(-1, 1), (0, 0)]:
        with pytest.raises(ValueError):
            GasModelCache(count, kept)


def _read_in_worker(reader, index):
    state = reader[index]
    return (
        state.gas_probability.tobytes(),
        state.labels.tobytes(),
        state.gas_probability.dtype.str,
        all(
            not getattr(state, name).flags.writeable
            for name in state.__dataclass_fields__
        ),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_model_reader_pickles_metadata_only_and_keeps_parent_ownership(dtype):
    source = frame(dtype)
    with GasModelCache(1) as cache:
        with pytest.raises(RuntimeError, match="not open"):
            cache.reader()
        list(cache.store([source]))
        reader = cache.reader()
        assert reader[0].gas_probability.dtype == dtype
        assert reader._cached and not cache._cached
        serialized = pickle.dumps(reader)
        assert len(serialized) < 2048
        restored = pickle.loads(serialized)
        assert not restored._cached
        assert restored[0].gas_probability.tobytes() == source.gas_probability.tobytes()
        with pytest.raises(TypeError, match="cache.reader"):
            pickle.dumps(cache)
        directory = cache._directory
        del restored
        gc.collect()
        assert directory.is_dir()
    assert not directory.exists()
    with pytest.raises(RuntimeError, match="not open"):
        reader[0]
    with pytest.raises(RuntimeError, match="no longer open"):
        pickle.loads(serialized)


def test_model_reader_validates_private_inventory_and_array_contract():
    with GasModelCache(1) as cache:
        list(cache.store([frame()]))
        reader = cache.reader()
        other = frame(np.float64)
        np.savez(cache._directory / "00000000.npz", **vars(other))
        with pytest.raises(ValueError, match="grid and precision"):
            reader[0]
        (cache._directory / "unexpected.npz").touch()
        with pytest.raises(ValueError, match="frame inventory"):
            cache.reader()


def test_spawned_readers_share_disk_bytes_without_cache_ownership():
    sources = [frame(np.float32, 1), frame(np.float32, 2)]
    with GasModelCache(2, kept=1) as cache:
        list(cache.store(sources))
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as workers:
            actual = list(workers.map(_read_in_worker, [cache.reader()] * 2, [0, 1]))
        assert cache._directory.is_dir() and not cache._cached
        for found, expected in zip(actual, sources, strict=True):
            assert found == (
                expected.gas_probability.tobytes(),
                expected.labels.tobytes(),
                expected.gas_probability.dtype.str,
                True,
            )
