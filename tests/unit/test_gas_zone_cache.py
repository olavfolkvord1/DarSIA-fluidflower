"""Private prefix-zone storage and worker views preserve exact arrays/records."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing
import pickle
import zipfile

import numpy as np
import pytest

from darsia.multiphase.gas_zone import GasLink, GasZone
from darsia.multiphase.gas_zone_cache import GasZoneCache


def zone(index=0):
    contour = np.ones((3, 4), bool)
    path = np.zeros_like(contour)
    path[1] = True
    blend = np.full(contour.shape, 0.25 + index / 8, np.float32)
    port = (0.01, 0.02)
    note = {"status": "linked", "port_pixel": [np.int64(1), 2], "zero": -0.0}
    coordinates = np.array([[1, col] for col in range(4)], np.int32)
    links = {} if index == 0 else {port: GasLink(path, blend, note, coordinates)}
    return GasZone(contour, links, {"ports": {port: note}, "index": index})


def _worker(reader, index):
    value = reader[index]
    coordinates = {
        port: (link.path_coordinates.tobytes(), link.path_coordinates.flags.writeable)
        for port, link in value.links.items()
    }
    return (
        value.contour.tobytes(),
        value.diagnostics,
        not value.contour.flags.writeable,
        coordinates,
    )


def test_zone_cache_lossless_uncompressed_and_cleanup():
    original = zone(1)
    with GasZoneCache(1) as cache:
        directory = cache._directory
        yielded = list(cache.store([original]))
        assert yielded[0] is original
        with zipfile.ZipFile(directory / "00000000.npz") as archive:
            assert all(
                item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
            )
        actual = cache[0]
        assert actual.diagnostics == original.diagnostics
        assert actual.contour.tobytes() == original.contour.tobytes()
        assert not actual.contour.flags.writeable
        for port, link in actual.links.items():
            assert link.diagnostics == original.links[port].diagnostics
            for name in ("path", "blend", "path_coordinates"):
                value, expected = getattr(link, name), getattr(
                    original.links[port], name
                )
                assert (
                    value.dtype == expected.dtype
                    and value.tobytes() == expected.tobytes()
                )
                assert not value.flags.writeable
    assert not directory.exists()


def test_zone_prefix_reader_is_explicit_and_cannot_read_future_indices():
    with GasZoneCache(3) as cache:
        with pytest.raises(RuntimeError, match="not been published"):
            cache.reader(through=0)
        stream = cache.store(zone(index) for index in range(3))
        next(stream)
        first = cache.reader(through=0)
        assert len(first) == 3 and first[0].diagnostics["index"] == 0
        with pytest.raises(RuntimeError, match="not open"):
            cache.reader()
        with pytest.raises(RuntimeError, match="outside the published"):
            first[1]
        with pytest.raises(RuntimeError, match="outside the published"):
            first[-1]
        next(stream)
        assert pickle.loads(pickle.dumps(first))[0].diagnostics["index"] == 0
        second = cache.reader(through=1)
        assert second[1].diagnostics["index"] == 1
        assert len(second._cached) == 1
        second[0]
        assert len(second._cached) == 1
        with pytest.raises(RuntimeError, match="outside the published"):
            first[1]
        list(stream)
        assert cache.reader()[2].diagnostics["index"] == 2
    with pytest.raises(RuntimeError, match="not open"):
        first[0]


def test_zone_prefix_reader_spawns_without_owning_parent_files():
    with GasZoneCache(2) as cache:
        stream = cache.store([zone(1), zone(2)])
        first = next(stream)
        reader = cache.reader(through=0)
        reader[0]
        serialized = pickle.dumps(reader)
        assert len(serialized) < 2048 and not pickle.loads(serialized)._cached
        with ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            result = pool.submit(_worker, reader, 0).result()
        expected_coordinates = {
            port: (link.path_coordinates.tobytes(), False)
            for port, link in first.links.items()
        }
        assert result == (
            first.contour.tobytes(),
            first.diagnostics,
            True,
            expected_coordinates,
        )
        assert cache._directory.is_dir()
        list(stream)


@pytest.mark.parametrize("count", [0, 2])
def test_zone_wrong_count_never_completes(count):
    with GasZoneCache(count) as cache:
        with pytest.raises(ValueError, match="Expected"):
            list(cache.store([zone()]))
        with pytest.raises(RuntimeError, match="not open"):
            cache.reader()


@pytest.mark.parametrize(
    "wrong",
    [
        replace(zone(), contour=np.ones((3, 4), np.uint8)),
        replace(
            zone(1),
            links={
                (0.01, 0.02): replace(
                    next(iter(zone(1).links.values())),
                    blend=np.ones((3, 4), np.float64),
                )
            },
        ),
    ],
)
def test_zone_contract_refuses_wrong_widths(wrong):
    with GasZoneCache(1) as cache:
        with pytest.raises(ValueError):
            list(cache.store([wrong]))


def test_zone_invalid_diagnostics_cleanup_without_object_pickle():
    for diagnostic in ({"invalid": float("nan")}, {"invalid": object()}):
        with GasZoneCache(1) as cache:
            directory = cache._directory
            with pytest.raises((ValueError, TypeError)):
                list(cache.store([replace(zone(), diagnostics=diagnostic)]))
        assert not directory.exists()


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([[1, 0]], np.int64),
        np.array([[1, 0]], np.float32),
        np.empty((0, 2), np.int32),
        np.array([[1, -1]], np.int32),
        np.array([[1, 4]], np.int32),
        np.array([[1, 0], [1, 2]], np.int32),
        np.array([[1, 0], [1, 0]], np.int32),
        np.array([[0, 0]], np.int32),
    ],
)
def test_zone_coordinates_refuse_wrong_type_bounds_and_pixel_order(coordinates):
    original = zone(1)
    port = next(iter(original.links))
    wrong = replace(
        original,
        links={port: replace(original.links[port], path_coordinates=coordinates)},
    )
    with GasZoneCache(1) as cache:
        with pytest.raises(ValueError, match="coordinates"):
            list(cache.store([wrong]))


def test_zone_reader_preserves_coordinate_direction_after_pickle():
    original = zone(1)
    port = next(iter(original.links))
    reverse = original.links[port].path_coordinates[::-1].copy()
    original = replace(
        original, links={port: replace(original.links[port], path_coordinates=reverse)}
    )
    with GasZoneCache(1) as cache:
        list(cache.store([original]))
        restored = pickle.loads(pickle.dumps(cache.reader()))[0].links[port]
        assert np.array_equal(restored.path_coordinates, reverse)
        assert not restored.path_coordinates.flags.writeable
