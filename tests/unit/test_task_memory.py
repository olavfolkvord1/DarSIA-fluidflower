"""Array memory estimates without starting jobs, services or host locks."""

import copy
import tracemalloc
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

import darsia
from darsia.jobs.job import Sizes, task_bytes, task_memory, task_photo
from darsia.jobs.step import (
    Input,
    Memory,
    MemoryContext,
    Order,
    Selection,
    Step,
    Task,
    Term,
)
from darsia.presets.workflows.analysis.steps import MASS_COPY
from darsia.presets.workflows.calibration.steps import (
    _injection_map_bytes,
    _statistics_entries,
)
from darsia.presets.workflows.comparison.steps import DISTANCES
from darsia.presets.workflows.setup.steps import output_cells
from darsia.presets.workflows.setup.target_steps import (
    MATERIAL_TARGETS,
    MaterialTargetSettings,
    measure_rig_targets,
)
from darsia.utils.locations import Frame
from tests.unit.test_material_blocks import FIT
from tests.unit.test_sand_light_field import SETTINGS, photograph


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("pixels", [120, 480])
def test_memory_counts_precision_fixed_arrays_frames_and_config(itemsize, pixels):
    """Fixed-width maps stay fixed as precision changes; configured windows and
    retained frames multiply only the arrays whose lifetime they control."""
    memory = Memory(
        camera=3,
        grid=2,
        held=1,
        camera_bytes=8,
        grid_bytes=3,
        photos=1,
        photo_bytes=darsia.decoded_bytes,
        terms=(
            Term(
                lambda tables, run, context: tables[run], grid=2, grid_bytes=1, held=0.5
            ),
        ),
    )
    expected = itemsize * (3 * pixels + (2 + 5 + (2 + 0.5 * 5) * 4) * pixels)
    expected += (8 + 3 + 4 + darsia.decoded_bytes("image.dng")) * pixels
    assert (
        memory.bytes(itemsize, pixels, pixels, 5, {"run": 4}, "run", "image.dng")
        == expected
    )
    assert Memory(grid=0.1).bytes(4, 0, 1, 0) == 1


def test_task_memory_covers_mixed_formats_and_input_runs():
    """A part consuming another run counts its frames once and budgets the
    largest decoder, including when the first frame is a cheaper JPEG."""
    frames = (Frame("session", "one.jpg"), Frame("session", "two.dng"))
    selection = Selection({"run": frames})
    task = Task("part", "step", None, (), "all", (Input("images", "run"),), ())
    step = Step(
        "step",
        {},
        Order.PARTS,
        Memory(photos=1, photo_bytes=darsia.decoded_bytes, held=3),
        lambda ctx: None,
    )
    found = task_memory(step, task, selection, Sizes(4, 100, 25))
    assert found == dict(
        itemsize=4, camera=100, grid=25, frames=2, format=".dng", bytes=5000
    )
    assert task_bytes(step, task, selection, Sizes(4, 100, 25)) == found["bytes"]
    own = Task("run", "step", "run", frames, None, (), ())
    assert task_photo(own, darsia.decoded_bytes) == "two.dng"


def test_invalid_memory_counts_fail_before_admission():
    """Missing configuration, negative counts and non-finite counts must not
    silently turn an unknown task's memory into a small reservation."""
    memory = Memory(terms=(Term(lambda tables, run, context: tables["count"], grid=1),))
    with pytest.raises(ValueError, match="tables"):
        memory.bytes(4, 1, 1, 1)
    for count in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            memory.bytes(4, 1, 1, 1, {"count": count})


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("pixels", [120, 480])
def test_injection_memory_counts_shared_cache_and_run_protocol(
    tmp_path, itemsize, pixels
):
    """Shared model frames and per-port signal use working precision; bool/uint8
    maps and float32 link blends keep their own widths. Repeated episodes do not
    duplicate a port's state, and run overrides use the configured density."""
    (tmp_path / "injection.csv").write_text(
        "id,location_x,location_y,start,end,rate_g/s\n"
        "1,0,0,2024-01-01 00:00,2024-01-01 01:00,1\n"
        "2,0,0,2024-01-01 01:00,2024-01-01 02:00,1\n"
        "3,0,0,2024-01-01 03:00,2024-01-01 04:00,1\n"
        "4,1,0,2024-01-01 00:00,2024-01-01 02:00,1\n"
        "5,2,0,2024-01-01 00:00,2024-01-01 08:00,0\n"
    )
    (tmp_path / "volume.csv").write_text(
        "id,location_x,location_y,start,end,rate_sccm\n"
        "1,0,0,2024-01-01 00:00,2024-01-01 01:00,1\n"
    )
    tables = {
        "protocols": {"injection": "injection.csv"},
        "runs": {
            "r2": {
                "protocols": {
                    "injection": "volume.csv",
                    "injection_reference_density": 2.5,
                }
            }
        },
    }
    memory = Memory(terms=(Term(_injection_map_bytes, grid_bytes=1),))
    assert (
        memory.bytes(itemsize, 0, pixels, 0, tables, "r1", project=tmp_path)
        == {4: 173, 8: 189}[itemsize] * pixels
    )
    assert (
        memory.bytes(itemsize, 0, pixels, 0, tables, "r2", project=tmp_path)
        == {4: 127, 8: 143}[itemsize] * pixels
    )
    with pytest.raises(ValueError, match="project"):
        memory.bytes(itemsize, 0, pixels, 0, tables, "r1")


def test_task_memory_passes_project_and_counts_nonpixel_storage(tmp_path):
    """A declaration can read project metadata and count retained scalar arrays
    without making the generic engine aware of its scientific configuration."""
    (tmp_path / "count").write_text("7")
    memory = Memory(
        terms=(
            Term(
                lambda tables, run, context: int(
                    (context.project / tables[run]).read_text()
                ),
                fixed_bytes=8,
                frame_bytes=4,
            ),
        ),
        frame_bytes=2,
    )
    frames = (Frame("session", "one.jpg"), Frame("session", "two.jpg"))
    task = Task("run", "step", "run", frames, None, (), ())
    step = Step("step", {}, Order.RUN, memory, lambda ctx: None)
    found = task_memory(
        step,
        task,
        Selection({"run": frames}),
        Sizes(4, 1, 1),
        {"run": "count"},
        tmp_path,
    )
    assert found["bytes"] == 7 * (8 + 2 * 4) + 2 * 2


@pytest.mark.parametrize("itemsize", [4, 8])
@pytest.mark.parametrize("scale", [1, 2])
def test_output_cell_memory_handles_nominal_canvas_and_shape(itemsize, scale):
    """Finer output raises the reservation independently of the input size.
    A measured model uses its canvas or supplied shape, and fails without both."""
    shape = (20 * scale, 30 * scale)
    pixels = int(np.prod(shape))
    context = MemoryContext(itemsize, pixels, pixels, 1)
    tables = {"rig": {"height": 0.2, "width": 0.3}, "mass_copy": {"copy_mm": 10}}
    assert output_cells(tables, ("run",), context, 10) == 600
    assert output_cells(tables, ("run",), context, 5) == 2400
    coarse = MASS_COPY.memory.bytes(itemsize, pixels, pixels, 1, tables, "run")
    tables["mass_copy"]["copy_mm"] = 5
    fine = MASS_COPY.memory.bytes(itemsize, pixels, pixels, 1, tables, "run")
    assert fine - coarse == (2400 - 600) * 18 * 8
    model = [
        [0.01 / scale, 0, 0.005 / scale],
        [0, -0.01 / scale, 0.2 - 0.005 / scale],
        [0, 0, 1],
    ]
    tables["labeling"] = {"sand_map": {"grid_to_rig": model}}
    with pytest.raises(ValueError, match="Sizes.grid_shape"):
        output_cells(tables, ("run",), context, 10, 10)
    assert (
        output_cells(tables, ("run",), replace(context, grid_shape=shape), 10, 10)
        == 600
    )
    tables["labeling"]["sand_map"].update(canvas_origin=[0, 0], canvas_shape=shape)
    assert output_cells(tables, ("run",), context, 10, 10) == 600
    assert Sizes(itemsize, pixels, pixels, shape).to_json()["grid_shape"] == list(shape)
    for bad in ((1, pixels + 1), (0, pixels), (pixels,), (1.0, pixels)):
        with pytest.raises(ValueError, match="grid_shape"):
            Sizes(itemsize, pixels, pixels, bad)


def test_comparison_cells_count_each_selected_run_and_domain():
    """All frame cell maps stay alive through the pairwise comparisons."""
    tables = {
        "rig": {"height": 0.2, "width": 0.3},
        "wasserstein": {"cell_mm": 10},
        "distances": {"reference_run": "one"},
        "domains": {"domains": {"whole": {}, "upper": {}}},
    }
    memory = DISTANCES.memory
    first = memory.bytes(4, 600, 600, 1, tables, runs=("one",))
    both = memory.bytes(4, 600, 600, 2, tables, runs=("one", "two"))
    assert both - first == 600 * 2 * (4 + 8)


def test_statistics_counts_are_bounded_by_pixels_before_json_storage():
    """A high quantization count cannot create more occupied groups than pixels;
    disjoint sand histograms together also contain at most one bin per pixel."""
    tables = {
        "reading": {"sands": {"1": {}, "2": {}}},
        "run_statistics": {"intervals": 32768},
        "unknown_mass": {"intervals": 32768},
    }
    assert _statistics_entries(tables, "one", MemoryContext(4, 100, 100, 1)) == 300


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1, 2])
def test_material_target_memory_counts_configured_fit_work_and_measured_arrays(
    dtype, scale
):
    """Two resolutions and precisions cover this synthetic target fit's traced
    arrays; degree, fit-grid size and external-mask levels are separate terms.
    This check does not bound native LAPACK workspace or arbitrary experiments."""
    image, labels = photograph(np.random.default_rng(1))
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1).astype(dtype)
    labels = np.repeat(np.repeat(labels, scale, axis=0), scale, axis=1)
    baseline = darsia.OpticalImage(image, dimensions=[0.096, 0.16])
    rig = SimpleNamespace(
        shape_corrected_baseline=baseline, facies=darsia.ScalarImage(labels)
    )
    field = replace(SETTINGS, minimum_blocks=10)
    settings = MaterialTargetSettings([1, 2, 5], (10, 10), field, 2)
    correction = dict(
        fit_shape=[48 * scale, 80 * scale],
        block_size=4,
        edge_margin=0,
        dn_scale=1000,
        fit=asdict(replace(FIT, minimum_block_cells=4)),
    )
    tables = {
        "material_targets": asdict(settings),
        "corrections": {"material_illumination": correction},
    }
    pixels, itemsize = labels.size, np.dtype(dtype).itemsize
    estimate = MATERIAL_TARGETS.memory.bytes(
        itemsize, pixels, pixels, 1, tables, "run", grid_shape=labels.shape
    )
    tracemalloc.start()
    try:
        found = measure_rig_targets(rig, correction, settings)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert set(found["targets"]) == {"1", "2", "5"}
    assert 0 < peak <= estimate
    changed = copy.deepcopy(tables)
    changed["material_targets"]["light_field"]["degree"] = 4
    assert (
        MATERIAL_TARGETS.memory.bytes(
            itemsize, pixels, pixels, 1, changed, "run", grid_shape=labels.shape
        )
        > estimate
    )
