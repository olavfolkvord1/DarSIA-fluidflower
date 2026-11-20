"""Statistics memory for mixed image and capacity precisions, without jobs."""

import sys
import tracemalloc
from dataclasses import replace
from types import SimpleNamespace

import darsia
import numpy as np
import pytest
from darsia.jobs.step import MemoryContext
from darsia.presets.workflows.calibration.steps import _statistics_bytes


@pytest.mark.parametrize("shape", [(12, 16), (24, 32)])
@pytest.mark.parametrize("image_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("geometry_dtype", [np.float32, np.float64])
def test_statistics_memory_covers_capacity_dtype(shape, image_dtype, geometry_dtype):
    """Dense column groups retain capacity precision even for float32 signals."""
    rows, cols = shape
    relative = np.broadcast_to(
        np.linspace(0.1, 0.9, cols, dtype=image_dtype), shape
    ).copy()
    labels = np.full(shape, darsia.PhaseLabel.DISSOLVED, np.uint8)
    measured = np.zeros(shape, bool)
    measured[0] = True
    sand_map = darsia.ScalarImage(np.ones(shape, np.uint8), dimensions=[0.01, 0.02])

    def full(value):
        return np.full(shape, value, geometry_dtype)

    geometry = darsia.run_geometry(
        sand_map, full(1e-8), full(0.01), full(0.4), full(0.1), np.ones(shape, bool)
    )
    co2 = SimpleNamespace(
        solubility_co2=np.linspace(1, 2, rows * cols, dtype=image_dtype).reshape(shape),
        density_gaseous_co2=np.ones(shape, image_dtype),
    )
    capacity = darsia.dissolved_capacity(
        geometry.pore_volume_m3, co2.solubility_co2, np.zeros(shape, image_dtype)
    )
    histogram = darsia.signal_histograms(
        relative,
        measured,
        sand_map.img,
        geometry.counted,
        capacity,
        [1],
        darsia.RunStatisticsSettings(1, 0, 1, 32768),
    )[1]
    unknown = darsia.unknown_statistics(
        relative,
        labels,
        np.zeros(shape, np.uint8),
        measured,
        geometry,
        co2,
        darsia.UnknownMassSettings(100, 32768),
    )
    column = unknown.column_estimate()
    arrays = [
        histogram.relative,
        histogram.weight,
        column.relative,
        column.ratio,
        column.weight,
    ]
    expected_dtype = np.result_type(image_dtype, geometry_dtype)
    assert all(array.dtype == expected_dtype for array in arrays)
    # All source arrays coexist with the conversion of one array.
    held = sum(array.nbytes for array in arrays)
    held += max(
        array.size * (
            np.dtype(image_dtype).itemsize + sys.getsizeof(0.0) + np.dtype(np.intp).itemsize
        )
        for array in arrays
    )
    context = MemoryContext(
        np.dtype(image_dtype).itemsize, rows * cols, rows * cols, 99
    )
    tables = {
        "run_statistics": {"intervals": 32768},
        "unknown_mass": {"intervals": 32768},
        "reading": {"sands": {"one": {}}},
    }
    assert _statistics_bytes(tables, None, context) >= held


def test_statistics_writer_only_materializes_one_numeric_array(tmp_path):
    """Two dense variants fit the declaration and retain the same JSON values."""
    pixels = 32768
    values = np.linspace(0, 1, pixels, dtype=np.float64)
    column = darsia.ColumnEstimate(values, values * 0.5, values * 0.2)
    frame = darsia.CalibrationFrame(
        "one", 0, 2.5, 2.5, 10.0, {}, darsia.GasStatistics(1.0, 0.0, 0.002, {}),
        column, 0.001, True,
    )
    held_column = darsia.ColumnEstimate(values + 0.2, values * 0.25, values * 0.3)
    held = replace(frame, column=held_column, measured=False)
    settings = {
        "run_statistics": {"intervals": pixels}, "unknown_mass": {"intervals": pixels},
        "calibration_runs": {"fit": ["one"], "folds": ["one"]},
    }
    declared = _statistics_bytes(settings, "one", MemoryContext(4, pixels, pixels, 1))
    source_bytes = 6 * values.nbytes
    tracemalloc.start()
    try:
        darsia.write_frames(tmp_path, "one", [frame], np.float32, {"one": [held]})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert source_bytes + peak <= declared + 64 * 1024
    actual = darsia.read_frames(tmp_path, "one")[0]
    assert actual.measured is False
    np.testing.assert_array_equal(actual.column.relative, held_column.relative.astype(np.float32))
