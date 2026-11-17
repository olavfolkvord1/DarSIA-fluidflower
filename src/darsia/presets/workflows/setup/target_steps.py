"""The measurement of the material light correction's targets as a job step.

The step reads a run's stored rig, e.g. that of the reference run of the material
light correction (``[material_reference] run``): its shape-corrected baseline, its
facies and the fit mask of its correction (``[corrections.material_illumination]
fit_mask``). The light field shared by its sands
(:func:`darsia.fit_sand_light_field`) is divided out, keeping the light at a
configured point, e.g. the colour checker's centre, and the constant target colour
of each material is measured on the blocks of the correction's fitting grid
(:func:`darsia.measure_material_targets`). The targets are stored as
``targets.json``, in the form of ``[corrections.material_illumination.targets]``;
nothing is configured by the step. It is kept apart from the Setup's steps, so that
a Setup job does not need its settings.
"""

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import darsia
from darsia.jobs.step import Memory, Order, Scope, Step, StepContext, Term, run_tables
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.setup.steps import read_rig

TARGETS = "targets.json"
"""The stored targets' file in their result folder."""
CORRECTION = ("corrections", "material_illumination")
"""The table of the material light correction, whose fitting grid, blocks and
fit mask the measurement shares."""


@dataclass(frozen=True)
class MaterialTargetSettings:
    """How the targets of the material light correction are measured; the
    ``[material_targets]`` table."""

    materials: Sequence[int]
    """The materials (facies ids) whose targets are measured."""
    anchor: Sequence[float]
    """Column and row, in pixels of the run's grid, of the point whose light the
    targets keep, e.g. the centre of the colour checker."""
    light_field: darsia.SandLightFieldSettings
    """The light field shared by the sands, divided out before the measurement."""
    minimum_blocks: int
    """Fewest training blocks without colour outliers a material must keep."""


def read_targets(folder: Path) -> dict:
    """Stored targets (:data:`TARGETS`)."""
    return json.loads((Path(folder) / TARGETS).read_text(encoding="utf-8"))


def _correction(tables: Mapping[str, Any]) -> dict[str, Any]:
    """A run's table of the material light correction."""
    return tables.get(CORRECTION[0], {}).get(CORRECTION[1], {})


def measure_rig_targets(
    rig: Any, correction: Mapping[str, Any], settings: MaterialTargetSettings
) -> dict:
    """The targets of the material light correction measured on a rig.

    Parameters
    ----------
    rig : Rig
        The run's rig: its shape-corrected baseline and its facies.
    correction : Mapping[str, Any]
        The run's ``[corrections.material_illumination]`` table: its
        ``fit_shape``, ``block_size``, ``edge_margin``, ``dn_scale``, ``fit`` and,
        optionally, ``fit_mask`` (default: the pixels of the measured materials).
        Its targets, if any, are not read.
    settings : MaterialTargetSettings
        The measurement's settings.

    Returns
    -------
    dict
        ``targets`` (each material's natural logarithm of its target RGB, by id),
        ``blocks`` (each material's training blocks and those kept) and
        ``light_field`` (its coefficients, the level and spread of each sand, the
        number of blocks and the anchor).

    """
    baseline = rig.shape_corrected_baseline
    facies = np.asarray(rig.facies.img)
    mask_path = correction.get("fit_mask")
    if mask_path is None:
        fit_mask = np.isin(facies, list(settings.materials))
    else:
        fit_mask = np.asarray(darsia.imread(Path(mask_path)).img, bool)
    field = darsia.fit_sand_light_field(
        baseline.img, facies, fit_mask, settings.light_field
    )
    targets, blocks = darsia.measure_material_targets(
        baseline.img,
        field.flatten(baseline.img, settings.anchor),
        facies,
        fit_mask,
        list(settings.materials),
        dimensions=tuple(baseline.dimensions),
        fit_shape=tuple(correction["fit_shape"]),
        block_size=int(correction["block_size"]),
        edge_margin=float(correction["edge_margin"]),
        dn_scale=float(correction["dn_scale"]),
        settings=settings_from_table(darsia.MaterialFitSettings, correction["fit"]),
        minimum_blocks=settings.minimum_blocks,
    )
    return {
        "targets": {str(m): np.asarray(v).tolist() for m, v in targets.items()},
        "blocks": {
            str(m): {"training": training, "kept": kept}
            for m, (training, kept) in blocks.items()
        },
        "light_field": {
            "coefficients": field.coefficients.tolist(),
            "levels": {str(m): level for m, level in field.levels.items()},
            "spread": {str(m): list(spread) for m, spread in field.spread.items()},
            "samples": field.samples,
            "anchor": [float(value) for value in settings.anchor],
        },
    }


def _fit_mask(tables: Mapping[str, Any], project, folders) -> list[str]:
    """The fit mask of the run's material light correction, if it names one."""
    mask = _correction(tables).get("fit_mask")
    return [] if mask is None else [str(mask)]


def _material_targets(ctx: StepContext) -> None:
    """The targets measured on the run's rig."""
    run = ctx.task.run
    measured = measure_rig_targets(
        ctx.read("rig", read_rig),
        _correction(run_tables(ctx.tables, run)),
        ctx.settings["material_targets"],
    )

    def write(folder):
        (Path(folder) / TARGETS).write_text(json.dumps(measured, indent=1), "utf-8")

    ctx.publish(write)


def _target_work_bytes(tables, run, context) -> int:
    """Configured light-field design and material-target fitting-grid buffers.

    Count polynomial columns and fitted levels, reduced cells, blocks, dense
    designs and their solve copies. A default mask contains only the requested
    materials. An external mask may contain others; minimum_blocks bounds their
    number before any level/design matrix is allocated. Library solve workspace
    can vary, so this declaration is combined with the service's measured peaks.
    """
    own = run_tables(tables, run) if run is not None else tables
    correction, settings = _correction(own), own["material_targets"]
    field = settings["light_field"]
    cell, block, degree = (
        int(field[k]) for k in ("cell_pixels", "block_cells", "degree")
    )
    minimum = int(field["minimum_blocks"])
    if min(cell, block, minimum) <= 0 or degree < 1:
        raise ValueError(
            "Light-field memory needs positive cell/block/minimum counts and degree."
        )
    if context.grid_shape is None:
        cells = max(1, context.grid_pixels // cell)
        samples = cells // block**2
    else:
        shape = tuple(max(1, n // cell) for n in context.grid_shape)
        cells = math.prod(shape)
        samples = math.prod(n // block for n in shape)
    terms = degree * (degree + 3) // 2
    levels = len(settings["materials"])
    if correction.get("fit_mask") is not None:
        levels = samples // minimum
    unknowns = terms + levels
    # Cell labels, green/log-green, block copies and reduction scratch;
    # sample indices/scalars; polynomial/list stack; levels; design, weighted
    # design and solve copy; small dense solve factors.
    light = cells * (6 * 8 + 4 + 3) + samples * 10 * 8
    light += samples * (2 * terms * 8 + levels * 9 + 3 * unknowns * 8)
    light += 3 * unknowns**2 * 8
    fit_shape = tuple(int(n) for n in correction["fit_shape"])
    if context.grid_shape is not None:
        fit_shape = tuple(min(n, m) for n, m in zip(fit_shape, context.grid_shape))
    if len(fit_shape) != 2 or min(fit_shape) <= 0:
        raise ValueError(
            "Material-target memory needs two positive fit_shape dimensions."
        )
    fitted = math.prod(fit_shape)
    minimum_cells = int(correction["fit"]["minimum_block_cells"])
    if minimum_cells <= 0:
        raise ValueError("Material-target memory needs positive minimum_block_cells.")
    # Projection/colour/block RGB arrays, int64 indices, coordinates, masks,
    # and output sample vectors. Each record holds two RGB arrays and scalars.
    records = fitted // minimum_cells
    record_bytes = (
        sys.getsizeof((None,) * 6)
        + 4 * sys.getsizeof(np.float64(0))
        + 2 * (sys.getsizeof(np.empty(0, np.float32)) + 3 * 4)
        + 16
    )
    targets = fitted * (4 * 3 * 4 + 4 * 8 + 8) + records * (record_bytes + 12 * 8)
    return light + targets


MATERIAL_TARGETS = Step(
    "material-targets",
    {"rig": Scope.OWN},
    Order.RUN,
    Memory(
        # Loaded rig, flattened RGB and its copy; tiled float64 field work,
        # float32 projection input and masks. Fitting grids are counted below.
        grid=12 + 3 + 3,
        camera_bytes=18,
        grid_bytes=64 + 5 * 8 + 3 * 4 + 4,
        terms=(Term(_target_work_bytes, fixed_bytes=1),),
    ),
    _material_targets,
    {"material_targets": MaterialTargetSettings},
    sections=(CORRECTION[0],),
    stamped=_fit_mask,
)
"""The targets of the material light correction, measured on the run's rig."""
STEPS = (MATERIAL_TARGETS,)
"""The steps of this module."""
