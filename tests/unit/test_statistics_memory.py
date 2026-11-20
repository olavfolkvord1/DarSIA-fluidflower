"""Statistics memory for mixed image and capacity precisions, without jobs."""

import sys
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
    # The writer retains source arrays, asdict copies, and converted float lists.
    # One dtype-conversion array can also coexist while tolist builds its list.
    lists = [array.astype(image_dtype).tolist() for array in arrays]
    held = 2 * sum(array.nbytes for array in arrays)
    held += sum(
        sys.getsizeof(values) + sum(map(sys.getsizeof, values)) for values in lists
    )
    held += max(array.size for array in arrays) * np.dtype(image_dtype).itemsize
    context = MemoryContext(
        np.dtype(image_dtype).itemsize, rows * cols, rows * cols, 99
    )
    tables = {
        "run_statistics": {"intervals": 32768},
        "unknown_mass": {"intervals": 32768},
        "reading": {"sands": {"one": {}}},
    }
    assert _statistics_bytes(tables, None, context) >= held
