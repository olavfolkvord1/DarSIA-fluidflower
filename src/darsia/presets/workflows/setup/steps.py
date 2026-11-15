"""The Setup's steps: each run's sand map, rig and corrected images.

Each run is set up from the job's configuration and its own tables: the tables
under ``[runs.<run>]`` are merged over the job's for that run, e.g. its photo
folder, baseline, protocols and sand-map adjustment. The rig is the official one
(:func:`darsia.presets.workflows.setup.setup_rig.make_rig`), with the run's sand
map as its labels and the facies of those labels
(:func:`darsia.presets.workflows.setup.setup_facies.make_facies`); each corrected
image is a photo read through the rig's corrections, once per Setup. A run whose
``[depth] interpolation`` is ``"triangulation"`` takes its depth from the rule
evaluated on its own grid (:func:`grid_depth`), as our production's mass did.

A step records the configuration sections it reads through the run's official
configuration (:attr:`darsia.jobs.step.Step.sections`) and the content of the
small files they name that it reads (:attr:`darsia.jobs.step.Step.files`), e.g.
the protocols of the run's photo folders, and the size and time of the large ones
(:attr:`darsia.jobs.step.Step.stamped`), the depth map and the image mapping, so
that a result made before one of them was changed is not reused.

"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import toml

import darsia
from darsia.jobs.step import RUNS, Memory, Order, Scope, Step, StepContext, run_tables
from darsia.presets.workflows.config.depth import DepthConfig
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.config.sections import list_required_sections
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_facies import make_facies
from darsia.presets.workflows.setup.setup_rig import (
    make_experiment,
    make_rig,
    setup_rig,
)
from darsia.presets.workflows.setup.setup_sand_map import make_sand_map, setup_sand_map
from darsia.utils.locations import normal_path

EXPERIMENT = ("data", "protocols")
"""The sections of a run's experiment: its photos and protocols."""
DEPTH = "depth"
"""The column of the measured depths in the depth measurements' table, as the
official depth setup reads it."""


def _file(value: Any) -> str:
    """The file of a configured protocol: a path, or a [path, sheet] pair."""
    return str(value[0]) if isinstance(value, (list, tuple)) else str(value)


def _holds(folder: Any, project: Path, folders: Sequence[Path]) -> bool:
    """Whether a configured data folder holds one of a run's photo folders."""
    base = os.path.normcase(normal_path(str(folder), project))
    for photos in folders:
        found = os.path.normcase(normal_path(photos))
        if found == base or found.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def grid_mask(
    tables: Mapping[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray | None:
    """A boolean image on a run's grid that its configuration names
    (``[labeling.sand_map] <key>``), if it names one: e.g. the pixels the run
    reports on (``reporting_domain``) or those its phase step reads
    (``phase_support``).

    Parameters
    ----------
    tables : Mapping[str, Any]
        The run's configuration tables.
    key : str
        The mask's key in ``[labeling.sand_map]``.
    shape : tuple of int
        The grid's shape; its first two entries are its rows and columns.

    Returns
    -------
    numpy.ndarray or None
        The mask, or None if the run names none.

    Raises
    ------
    ValueError
        If the image is not a boolean mask of the grid.

    """
    path = tables.get("labeling", {}).get("sand_map", {}).get(key)
    if path is None:
        return None
    mask = np.asarray(darsia.imread(Path(path)).img)
    if mask.dtype != bool or mask.shape != tuple(shape[:2]):
        raise ValueError(f"[labeling.sand_map] {key} is a boolean mask of the grid.")
    return mask


def protocol_files(
    tables: Mapping[str, Any], project: Path, folders: Sequence[Path]
) -> list[str]:
    """The protocol files a run's experiment reads (:func:`make_experiment`): the
    imaging protocol and blacklist of each data folder that holds the run's photos,
    and the injection and pressure-temperature protocols.

    Parameters
    ----------
    tables : Mapping[str, Any]
        The run's configuration tables.
    project : Path
        The project folder, against which relative folders are read.
    folders : Sequence[Path]
        The run's photo folders.

    Returns
    -------
    list of str
        The files, as configured.

    """
    protocols = tables.get("protocols", {})
    found = []
    for key in ("imaging", "blacklist"):
        value = protocols.get(key)
        if isinstance(value, Mapping):
            found += [_file(v) for k, v in value.items() if _holds(k, project, folders)]
        elif value is not None:
            found.append(_file(value))
    for key in ("injection", "pressure_temperature"):
        if protocols.get(key) is not None:
            found.append(_file(protocols[key]))
    return found


def run_config(ctx: StepContext, run: str, folder: Path) -> list[Path]:
    """The configuration files of one run: the job's, then the run's own tables
    (``[runs.<run>]``) as one more file written into a folder.

    Parameters
    ----------
    ctx : StepContext
        The task's context.
    run : str
        The run.
    folder : Path
        A folder for the run's file.

    Returns
    -------
    list of Path
        The files, merged in order.

    """
    paths = list(ctx.config)
    own = ctx.tables.get(RUNS, {}).get(run)
    if own:
        paths.append(Path(folder) / "run.toml")
        paths[-1].write_text(toml.dumps(own), encoding="utf-8")
    return paths


def run_settings(
    ctx: StepContext, run: str, sections: Sequence[str] = ()
) -> FluidFlowerConfig:
    """The official configuration of one run (:func:`run_config`), with the
    sections it needs checked."""
    with TemporaryDirectory() as folder:
        config = FluidFlowerConfig(
            run_config(ctx, run, Path(folder)),
            require_data=False,
            require_results=False,
        )
    config.check(*sections)
    return config


def read_labels(folder: Path) -> darsia.Image:
    """A stored sand map (labels on the run's grid)."""
    return darsia.imread(Path(folder) / "labels.npz")


def read_rig(folder: Path) -> Rig:
    """A stored rig (:meth:`Rig.load`)."""
    return Rig.load(Path(folder))


def read_image(folder: Path) -> darsia.Image:
    """A stored corrected image."""
    return darsia.imread(Path(folder) / "image.npz")


def _sand_map(ctx: StepContext) -> None:
    """The run's sand map from the series' drawing and the run's adjustment."""
    sections = list_required_sections(setup_sand_map)
    labels = make_sand_map(run_settings(ctx, ctx.task.run, sections))
    ctx.publish(lambda folder: labels.save(Path(folder) / "labels.npz"))


def grid_depth(rig: Rig, measurements: Path) -> darsia.Image:
    """A rig's depth by the depth rule of its measurements: linear inside their
    hull and the nearest one outside (:func:`darsia.interpolate_linear_in_hull`),
    at the pixel centres of the rig's grid (:func:`darsia.pixel_centres`, in
    metres, cast to the grid's precision) and in that precision, the precision of
    its pixel areas, as our production's mass evaluated it on its run grid.

    Parameters
    ----------
    rig : Rig
        The rig, on its run grid.
    measurements : Path
        The depth measurements' table: ``x``, ``y`` [m] and :data:`DEPTH`.

    Returns
    -------
    darsia.Image
        The depth [m] of each pixel, on the rig's grid.

    """
    table = pd.read_csv(measurements)
    columns = ["x" if "x" in table else "X", "y" if "y" in table else "Y", DEPTH]
    x, y = darsia.pixel_centres(rig.baseline)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values, _ = darsia.interpolate_linear_in_hull(
        tuple(table[c].to_numpy() for c in columns),
        points.astype(rig.pixel_area.img.dtype),
    )
    return darsia.ScalarImage(
        values.reshape(xx.shape),
        dimensions=rig.baseline.dimensions,
        origin=rig.baseline.origin,
    )


def output_cells(
    tables, runs, context, cell_mm: float, edge_mm=None, reference=None
) -> int:
    """Count nominal metric cells from configured physical rig dimensions."""
    if not math.isfinite(cell_mm) or cell_mm <= 0:
        raise ValueError("Memory estimation needs a finite positive output cell size.")
    own = {r: run_tables(tables, r) if r is not None else tables for r in runs}
    step = Decimal(str(float(cell_mm))) / Decimal(1000)
    rig = own[reference if reference is not None else next(iter(own))]["rig"]
    lengths = (rig["height"], rig["width"])
    if any(not math.isfinite(float(n)) or n <= 0 for n in lengths):
        raise ValueError("Memory estimation needs finite positive output dimensions.")
    return math.prod(
        int((Decimal(str(n)) / step).to_integral_value(rounding=ROUND_CEILING))
        for n in lengths
    )


def _rig(ctx: StepContext) -> None:
    """The run's rig, with its sand map as the labels and their facies, and with
    the depth rule evaluated on its grid where the run's configuration chooses it."""
    labels = ctx.read("sand-map", lambda folder: Path(folder) / "labels.npz")
    config = run_settings(ctx, ctx.task.run, list_required_sections(setup_rig))
    with TemporaryDirectory() as folder:
        facies = Path(folder) / "facies.npz"
        make_facies(config, darsia.imread(labels)).save(facies)
        # A depth rule evaluated on the grid needs no depth map.
        rule = config.depth.interpolation == "triangulation"
        rig = make_rig(Rig, config, labels=labels, facies=facies, depth=not rule)
    if rule:
        rig.depth = grid_depth(rig, Path(config.depth.measurements))
        rig.setup_geometry()
    ctx.publish(lambda folder: rig.save(Path(folder)))


def _corrected_image(ctx: StepContext) -> None:
    """The run's photos of the task read through the rig's corrections."""
    run = ctx.task.run
    rig = ctx.read("rig", read_rig)
    rig.load_experiment(make_experiment(run_settings(ctx, run, EXPERIMENT)))
    for frame in ctx.task.frames:
        if ctx.done(frame):
            continue
        image = rig.read_image(ctx.photo(run, frame))
        ctx.publish(lambda folder: image.save(Path(folder) / "image.npz"), frame=frame)


def _drawing_files(tables: Mapping[str, Any], project, folders) -> list[str]:
    """The sand-map drawing and the run's adjustment of it."""
    sand_map = tables.get("labeling", {}).get("sand_map", {})
    return [str(sand_map[key]) for key in ("drawing", "adjustment") if key in sand_map]


def _rig_files(tables: Mapping[str, Any], project, folders) -> list[str]:
    """The run's protocols, the facies' properties and the depth measurements of
    a depth rule evaluated on the run's grid."""
    depth = tables.get("depth", {})
    named = [tables.get("facies", {}).get("props")]
    if str(depth.get("interpolation", "")).lower() == "triangulation":
        named.append(depth.get("measurements"))
    found = protocol_files(tables, project, folders)
    return found + [str(path) for path in named if path is not None]


def _rig_stamped(tables: Mapping[str, Any], project, folders) -> list[str]:
    """The rig's depth map, as its depth configuration names it (the depth
    setup's map in the results folder by default), unless its depth rule is
    evaluated on the run's grid, and its image mapping."""
    mapping = tables.get("corrections", {}).get("mapping") or {}
    found = [mapping.get("path")]
    rule = str(tables.get("depth", {}).get("interpolation", "")).lower()
    if "depth" in tables and rule != "triangulation":
        results = tables.get("data", {}).get("results")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "depth.toml"
            path.write_text(toml.dumps({"depth": tables["depth"]}), encoding="utf-8")
            depth = DepthConfig().load(path, None if results is None else Path(results))
        found.insert(0, depth.depth_map)
    return [str(path) for path in found if path is not None]


SAND_MAP = Step(
    "sand-map",
    {},
    Order.RUN,
    Memory(camera_bytes=16.0),
    _sand_map,
    sections=tuple(list_required_sections(setup_sand_map)),
    files=_drawing_files,
)
"""The run's sand map (R2)."""
RIG = Step(
    "rig",
    {"sand-map": Scope.OWN},
    Order.RUN,
    Memory(camera=4.0, camera_bytes=164.0, photos=1, photo_bytes=darsia.decoded_bytes),
    _rig,
    sections=tuple(list_required_sections(setup_rig)),
    files=_rig_files,
    stamped=_rig_stamped,
)
"""The run's rig."""
CORRECTED_IMAGE = Step(
    "corrected-image",
    {"rig": Scope.OWN},
    Order.FRAMES,
    Memory(
        camera=20.0,
        grid=15.0,
        camera_bytes=18.0,
        grid_bytes=68.0,
        photos=1,
        photo_bytes=darsia.decoded_bytes,
    ),
    _corrected_image,
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each frame's corrected image (R1)."""
STEPS = (SAND_MAP, RIG, CORRECTED_IMAGE)
"""The Setup's steps."""
