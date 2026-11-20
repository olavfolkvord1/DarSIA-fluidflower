"""Gas admission metadata against the actual coordinate and score algorithms."""

import gc
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.jobs.step import MemoryContext
from darsia.multiphase import gas_allocation, gas_zone
from darsia.multiphase.gas_parameters import gas_parameters
from darsia.presets.workflows.calibration.steps import (
    _gas_depth_bound_mm,
    _gas_pitch,
    _gas_scratch_bytes,
    _gas_scratch_parts,
)
from darsia.presets.workflows.setup.steps import grid_depth


@pytest.fixture(autouse=True)
def private_depth_files(tmp_path, monkeypatch):
    """All synthetic source files remain inside this test's temporary folder."""
    monkeypatch.chdir(tmp_path)


def tables(depth_mm=12.0):
    source = Path.cwd() / f"depth-{depth_mm:g}.npz"
    darsia.ScalarImage(np.full((3, 4), depth_mm / 1000), dimensions=[0.2, 0.3]).save(
        source
    )
    return {
        "rig": {"height": 0.2, "width": 0.3},
        "phase_rules": {"pitch_stride": 2},
        "depth": {"depth_map": str(source)},
        "gas_allocation": {
            "seed_mm2": 100.0,
            "persistent_seed_mm2": 25.0,
            "seed_window_h": 1.01,
            "fallback_seed_mm2": 25.0,
            "lattice_mm": 1.0,
            "cost_up_per_mm": 0.5,
            "cost_side_per_mm": 1.0,
            "cost_down_per_mm": 3.0,
        },
    }


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("scale", [1, 2])
def test_gas_nominal_shape_matches_positions_and_scales(itemsize, scale):
    shape = (20 * scale, 30 * scale)
    context = MemoryContext(itemsize, 0, int(np.prod(shape)), 1, grid_shape=shape)
    own = tables()
    image = darsia.Image(np.zeros(shape, dtype=np.uint8), dimensions=[0.2, 0.3])
    assert _gas_pitch(own, context) == darsia.grid_positions(image).pitch_mm
    assert darsia.grid_pitch(image) == darsia.grid_positions(image).pitch_mm
    assert _gas_scratch_bytes(own, "run", context) >= 4 * 8 * context.grid_pixels
    with pytest.raises(ValueError, match="Sizes.grid_shape"):
        _gas_scratch_bytes(own, "run", replace(context, grid_shape=None))


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("scale", [1, 2])
def test_gas_measured_pitch_uses_canvas_and_run_override(itemsize, scale):
    shape = (20 * scale, 30 * scale)
    context = MemoryContext(itemsize, 0, int(np.prod(shape)), 1, grid_shape=shape)
    own = tables()
    model = [[0.01 / scale, 0.0001, 0], [0.0001, -0.01 / scale, 0.2], [0.0002, 0, 1]]
    own["labeling"] = {"sand_map": {"grid_to_rig": model}}
    image = darsia.Image(np.zeros(shape, dtype=np.uint8), dimensions=[0.2, 0.3])
    dtype = np.dtype(f"f{itemsize}")
    expected = darsia.grid_positions(image, model, dtype, pitch_stride=2).pitch_mm
    assert _gas_pitch(own, context) == expected
    with pytest.raises(ValueError, match="Sizes.grid_shape"):
        _gas_pitch(own, replace(context, grid_shape=None))
    canvas = darsia.MeasurementCanvas((-2, -3), (shape[0] + 6, shape[1] + 4))
    own["labeling"]["sand_map"].update(
        canvas_origin=canvas.origin, canvas_shape=canvas.shape
    )
    expected = darsia.grid_positions(image, model, dtype, canvas, 2).pitch_mm
    assert _gas_pitch(own, context) == expected
    assert _gas_pitch(own, replace(context, grid_shape=None)) == expected
    changed = {**own, "runs": {"run": {"depth": tables(120)["depth"]}}}
    assert _gas_scratch_bytes(changed, "run", context) > _gas_scratch_bytes(
        own, "run", context
    )
    own["phase_rules"] = {}
    with pytest.raises(ValueError, match="pitch_stride"):
        _gas_pitch(own, context)


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("shape", [(20, 30), (40, 60)])
@pytest.mark.parametrize("depth_mm", [2.0, 80.0])
def test_gas_kernel_budget_covers_actual_materialized_kernel(
    monkeypatch, itemsize, shape, depth_mm
):
    """Intercept the actual filter kernel and its image buffers, including a
    depth-derived window wider than the image. FFT and EDT scratch are separate.
    """
    context = MemoryContext(itemsize, 0, int(np.prod(shape)), 1, grid_shape=shape)
    own = tables(depth_mm)
    settings = gas_parameters(
        darsia.GasAllocationSettings(**own["gas_allocation"]),
        np.full(shape, depth_mm / 1000),
    )
    pitch = _gas_pitch(own, context)
    seen = []
    original = gas_allocation.fftconvolve

    def record(source, kernel, **kwargs):
        target = original(source, kernel, **kwargs)
        seen.append((kernel.shape, kernel.nbytes, source.nbytes, target.nbytes))
        return target

    monkeypatch.setattr(gas_allocation, "fftconvolve", record)
    values = np.full(shape, 0.5, dtype=f"f{itemsize}")
    score = gas_allocation._round_score(
        np.ones(shape, bool), values, pitch, settings.score_window_mm
    )
    assert score.dtype == values.dtype
    assert len(seen) == 2
    reserved = _gas_scratch_bytes(own, "run", context)
    for _, kernel_bytes, source_bytes, target_bytes in seen:
        assert reserved >= 3 * kernel_bytes + source_bytes + target_bytes
    if depth_mm == 80:
        assert seen[0][0][0] > shape[0]


@pytest.mark.parametrize("depth_mm", [0, -1, np.inf, np.nan])
def test_gas_invalid_depth_source_cannot_make_small_estimate(depth_mm):
    with pytest.raises(ValueError, match="finite positive measured depth"):
        _gas_scratch_bytes(
            tables(depth_mm), "run", MemoryContext(4, 0, 600, 1, grid_shape=(20, 30))
        )


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("shape", [(180, 320), (360, 640)])
@pytest.mark.parametrize("measured", [False, True])
def test_prefix_geometry_and_fft_scratch_fit_declared_arrays(itemsize, shape, measured):
    """Exercise padded EDT, actual MCP and FFT at two sizes/precisions.

    Trace measures the tracked allocations of these helpers, not every native
    workspace or a whole-phase guarantee. Estimates retain separate library
    workspace allowances and host measured admission.
    """
    own = tables(12)
    if measured:
        own["labeling"] = {
            "sand_map": {
                "grid_to_rig": [
                    [0.3 / shape[1], 0, 0],
                    [0, -0.2 / shape[0], 0.2],
                    [0, 0, 1],
                ]
            }
        }
    dtype = np.dtype(f"f{itemsize}")
    image = darsia.Image(np.zeros(shape, np.uint8), dimensions=[0.2, 0.3])
    positions = darsia.grid_positions(
        image,
        own.get("labeling", {}).get("sand_map", {}).get("grid_to_rig"),
        dtype,
        pitch_stride=2,
    )
    path = np.ones(shape, bool)
    mask = np.zeros(shape, bool)
    mask[shape[0] // 4 : shape[0] // 2, shape[1] // 4 : 3 * shape[1] // 4] = True
    probability = np.full(shape, 0.5, dtype)
    depth = np.full(shape, 0.012, dtype)
    settings = gas_parameters(
        darsia.GasAllocationSettings(**own["gas_allocation"]), depth
    )
    context = MemoryContext(itemsize, 0, int(np.prod(shape)), 1, grid_shape=shape)
    reserved = _gas_scratch_parts(own, None, context)
    x = np.broadcast_to(positions.x, shape)
    y = np.broadcast_to(positions.y, shape)
    port = (float(x[-2, shape[1] // 2]), float(y[-2, shape[1] // 2]))
    operations = {
        "contour": lambda: gas_zone._envelope(mask, path, positions.pitch_mm, settings),
        "links": lambda: gas_zone._injector_link(
            port,
            mask,
            path,
            probability,
            path,
            positions,
            settings,
            current_co2=path,
            depth_m=np.full(np.shape(mask), 0.1),
        ),
        "score": lambda: gas_allocation._round_score(
            mask, probability, positions.pitch_mm, settings.score_window_mm
        ),
        "depth": lambda: gas_zone._depth_margin(mask, depth, path, positions.pitch_mm),
    }
    for name, operation in operations.items():
        gc.collect()
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            baseline = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            result = operation()
            peak = tracemalloc.get_traced_memory()[1] - baseline
        finally:
            if not tracing:
                tracemalloc.stop()
        if name == "links":
            assert result[0].any() and result[2]["status"] == "linked"
        elif name == "depth":
            assert result[0].any() and result[1]["added_pixels"] > 0
        else:
            assert result.any()
        assert 0 < peak <= reserved[name]


def test_source_depth_and_anisotropy_change_derived_storage():
    context = MemoryContext(4, 0, 600, 1, grid_shape=(20, 30))
    original = tables()
    larger = tables(120)
    assert (
        _gas_scratch_parts(larger, None, context)["contour"]
        > _gas_scratch_parts(original, None, context)["contour"]
    )
    assert (
        _gas_scratch_parts(larger, None, context)["score"]
        > _gas_scratch_parts(original, None, context)["score"]
    )
    taller = tables()
    taller["rig"]["height"] *= 10
    assert (
        _gas_scratch_parts(taller, None, context)["links"]
        > _gas_scratch_parts(original, None, context)["links"]
    )


@pytest.mark.parametrize("itemsize", [4, 8])
def test_depth_map_bound_covers_maximum_and_working_precision(itemsize, tmp_path):
    own = tables()
    values = np.array([[0.0100000001, 0.0240000001], [np.nan, -1.0]])
    source = tmp_path / "measured.npz"
    darsia.ScalarImage(values, dimensions=[0.2, 0.3]).save(source)
    own["depth"]["depth_map"] = str(source)
    context = MemoryContext(itemsize, 0, 600, 1, grid_shape=(20, 30))
    bound = _gas_depth_bound_mm(own, context)
    assert bound >= values[0].max() * 1000
    assert bound >= values[0].astype(f"f{itemsize}").max() * 1000
    assert bound < 24.001


@pytest.mark.parametrize("itemsize", [4, 8])
def test_measurement_maximum_bounds_actual_triangulated_run_grid(itemsize, tmp_path):
    source = tmp_path / "measurements.csv"
    source.write_text(
        "x,y,depth\n0.05,0.05,0.01\n0.25,0.05,0.02\n"
        "0.05,0.15,0.015\n0.25,0.15,0.03\n",
        encoding="utf-8",
    )
    own = tables()
    own["depth"] = {"interpolation": "triangulation", "measurements": source.name}
    shape = (20, 30)
    image = darsia.ScalarImage(
        np.zeros(shape, dtype=f"f{itemsize}"), dimensions=[0.2, 0.3]
    )
    rig = SimpleNamespace(baseline=image, pixel_area=image, grid_to_rig=None)
    context = MemoryContext(
        itemsize, 0, 600, 1, grid_shape=shape, project=str(tmp_path)
    )
    actual = grid_depth(rig, source).img
    bound = _gas_depth_bound_mm(own, context)
    assert bound >= 1000 * actual.max() >= 1000 * np.median(actual)
    assert 30 <= bound < 30.001


def test_relative_and_legacy_depth_sources_preserve_resolution(tmp_path):
    own = tables()
    context = MemoryContext(4, 0, 600, 1, grid_shape=(20, 30), project=str(tmp_path))
    own["depth"]["depth_map"] = Path(own["depth"]["depth_map"]).name
    expected = _gas_depth_bound_mm(own, context)
    with pytest.raises(ValueError, match="job's project"):
        _gas_depth_bound_mm(own, replace(context, project=None))
    target = tmp_path / "results/setup/depth/depth_map.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes((tmp_path / own["depth"]["depth_map"]).read_bytes())
    own.pop("depth")
    own["data"] = {"results": "results"}
    assert _gas_depth_bound_mm(own, context) == expected


def test_missing_or_invalid_sources_fail_before_memory_admission(tmp_path):
    own = tables()
    context = MemoryContext(4, 0, 600, 1, grid_shape=(20, 30), project=str(tmp_path))
    own.pop("depth")
    with pytest.raises(ValueError, match="configured measured depth source"):
        _gas_scratch_bytes(own, None, context)
    own["depth"] = {"depth_map": "missing.npz"}
    with pytest.raises((FileNotFoundError, ValueError)):
        _gas_scratch_bytes(own, None, context)
    source = tmp_path / "bad.csv"
    source.write_text("x,y,depth\n0,0,0\n1,1,-0.01\n", encoding="utf-8")
    own["depth"] = {"interpolation": "triangulation", "measurements": source.name}
    with pytest.raises(ValueError, match="finite positive measured depth"):
        _gas_scratch_bytes(own, None, context)
