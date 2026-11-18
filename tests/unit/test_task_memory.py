"""Array budgets before job metadata or host admission is introduced."""

from dataclasses import asdict

import numpy as np
import pytest

import darsia
from darsia.jobs.step import Memory, MemoryContext, Term
from darsia.presets.workflows.analysis.steps import MASS_COPY
from darsia.presets.workflows.calibration.steps import _injection_map_bytes, _window
from darsia.presets.workflows.setup.steps import output_cells
from tests.unit.test_phase_rules import SETTINGS


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


def test_phase_window_memory_uses_the_configured_run():
    """The initial phase declaration reads its run's configured neighbour window."""
    tables = {
        "phase_rules": asdict(SETTINGS),
        "runs": {"wide": {"phase_rules": {"neighbours_per_side": 5}}},
    }
    assert _window(tables, "wide") == 12
    assert _window(tables, "ordinary") == 2 * SETTINGS.neighbours_per_side + 2
