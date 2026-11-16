"""The Setup's steps: each run's sand map, rig and corrected images.

Each run is set up from the job's configuration and its own tables: the tables
under ``[runs.<run>]`` are merged over the job's for that run, e.g. its photo
folder, baseline, protocols and sand-map adjustment. The rig is the official one
(:func:`darsia.presets.workflows.setup.setup_rig.make_rig`), with the run's sand
map as its labels and the facies of those labels
(:func:`darsia.presets.workflows.setup.setup_facies.make_facies`); each corrected
image is a photo read through the rig's corrections, once per Setup. A run whose
``[depth] interpolation`` is ``"triangulation"`` takes its depth from the rule
evaluated on its own grid (:func:`grid_depth`), at its pixels' positions on the rig
by its measured model where it has one, as our production's mass did.
Where ``[material_reference]`` names a reference run, every other run's material
light correction is fitted on the reference's fitting grid, from the reference's
stored rig (:func:`reference_homography`), as our production fitted it.

A step records the configuration sections it reads through the run's official
configuration (:attr:`darsia.jobs.step.Step.sections`) and the content of the
small files they name that it reads (:attr:`darsia.jobs.step.Step.files`), e.g.
the protocols of the run's photo folders, and the size and time of the large ones
(:attr:`darsia.jobs.step.Step.stamped`), the depth map and the image mapping, so
that a result made before one of them was changed is not reused.

"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
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
from darsia.presets.workflows.setup.sand_map import grid_homographies
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
MATERIAL_REFERENCE = "material_reference"
"""The section naming the reference run of the material illumination fit
(``run``): every other run is fitted on its fitting grid."""


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


def run_config(
    ctx: StepContext,
    run: str,
    folder: Path,
    made: Mapping[str, Any] | None = None,
) -> list[Path]:
    """The configuration files of one run: the job's, then the run's own tables
    (``[runs.<run>]``) as one more file written into a folder, and then the tables
    a step makes for the run, if any.

    Parameters
    ----------
    ctx : StepContext
        The task's context.
    run : str
        The run.
    folder : Path
        A folder for the run's files.
    made : Mapping[str, Any], optional
        Tables the step makes for the run, e.g. of its reference run's files.

    Returns
    -------
    list of Path
        The files, merged in order.

    """
    paths = list(ctx.config)
    own = ctx.tables.get(RUNS, {}).get(run)
    for name, tables in (("run.toml", own), ("made.toml", made)):
        if tables:
            paths.append(Path(folder) / name)
            paths[-1].write_text(toml.dumps(tables), encoding="utf-8")
    return paths


def run_settings(
    ctx: StepContext,
    run: str,
    sections: Sequence[str] = (),
    made: Mapping[str, Any] | None = None,
) -> FluidFlowerConfig:
    """The official configuration of one run (:func:`run_config`), with the
    sections it needs checked."""
    with TemporaryDirectory() as folder:
        config = FluidFlowerConfig(
            run_config(ctx, run, Path(folder), made),
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


def read_grid_model(
    folder: Path,
) -> tuple[list[list[float]] | None, darsia.MeasurementCanvas | None]:
    """A stored rig's measured model of its grid (``Rig.grid_to_rig``) and the
    canvas its pixels were measured on (``Rig.measurement_canvas``), from its meta
    data alone; None for a grid in rig metres and for the grid itself."""
    with open(Path(folder) / "meta_data.json", "r") as f:
        meta = json.load(f)
    canvas = meta.get("measurement_canvas")
    if canvas is not None:
        canvas = darsia.MeasurementCanvas(
            tuple(canvas["origin"]), tuple(canvas["shape"])
        )
    return meta.get("grid_to_rig"), canvas


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
    in the precision of the rig's grid, that of its pixel areas, as our
    production's mass evaluated it on its run grid, at each pixel's position on
    the rig (:func:`darsia.grid_positions` by the rig's measured model of its
    grid, ``Rig.grid_to_rig``, cast to the grid's precision): the model at the
    pixel's column and row, as our production's mass placed its pixels in metres,
    or without one the pixel centres.

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
    dtype = rig.pixel_area.img.dtype
    positions = darsia.grid_positions(rig.baseline, rig.grid_to_rig, dtype)
    xx, yy = positions.x, positions.y
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values, _ = darsia.interpolate_linear_in_hull(
        tuple(table[c].to_numpy() for c in columns), points.astype(dtype)
    )
    return darsia.ScalarImage(
        values.reshape(xx.shape),
        dimensions=rig.baseline.dimensions,
        origin=rig.baseline.origin,
    )


def reference_run(tables: Mapping[str, Any]) -> list[str]:
    """The reference run of the material illumination fit that the configuration
    names (``[material_reference] run``), if any."""
    named = tables.get(MATERIAL_REFERENCE, {}).get("run")
    return [] if named is None else [named]


def grid_model(tables: Mapping[str, Any]) -> list[list[float]] | None:
    """A run's measured model of its grid, the homography from its grid's pixels
    to rig metres (``[labeling.sand_map] grid_to_rig``), if it has one."""
    return tables.get("labeling", {}).get("sand_map", {}).get("grid_to_rig")


def grid_reporting_domain(
    tables: Mapping[str, Any], grid: darsia.Image
) -> np.ndarray | None:
    """The pixels of a run's grid it reports on, a boolean image on the grid
    (``[labeling.sand_map] reporting_domain``), if it has one.

    Raises
    ------
    ValueError
        If the image is not a boolean mask of the grid.

    """
    path = tables.get("labeling", {}).get("sand_map", {}).get("reporting_domain")
    if path is None:
        return None
    mask = np.asarray(darsia.imread(Path(path)).img)
    if mask.dtype != bool or mask.shape != grid.img.shape[:2]:
        raise ValueError(
            "[labeling.sand_map] reporting_domain is a boolean mask of the grid."
        )
    return mask


def grid_canvas(tables: Mapping[str, Any]) -> darsia.MeasurementCanvas | None:
    """The canvas a run's pixels were measured on, in its grid's pixels
    (``[labeling.sand_map] canvas_origin``, the column and row of its first
    pixel, and ``canvas_shape``, its rows and columns), if it has one.

    Raises
    ------
    ValueError
        If one of the two is given without the other, or a canvas without the
        grid's measured model.

    """
    table = tables.get("labeling", {}).get("sand_map", {})
    origin, shape = table.get("canvas_origin"), table.get("canvas_shape")
    if origin is None and shape is None:
        return None
    if origin is None or shape is None or table.get("grid_to_rig") is None:
        raise ValueError(
            "[labeling.sand_map] canvas_origin and canvas_shape go together, with "
            "grid_to_rig."
        )
    return darsia.MeasurementCanvas(
        (int(origin[0]), int(origin[1])), (int(shape[0]), int(shape[1]))
    )


def _grid_to_rig(tables: Mapping[str, Any], grid: darsia.Image | None) -> np.ndarray:
    """A run's homography from its grid's pixels to rig metres: its measured model,
    else that of its metric grid (:func:`grid_homographies`)."""
    return grid_homographies(grid_model(tables), grid)[0]


def reference_homography(
    reference: np.ndarray, own: np.ndarray, dtype: np.dtype
) -> np.ndarray:
    """The homography from a run's grid pixels to its reference run's, from each
    run's measured model of its grid: ``inverse(reference) @ own``, computed in
    ``dtype`` (the inverse solved in it), as our production computed it in its
    precision.

    Parameters
    ----------
    reference, own : array
        The reference run's and this run's homography from their grid's pixels
        (column, row, 1) to rig metres.
    dtype : numpy dtype
        Floating precision of the calculation, e.g. the run grid's.

    Returns
    -------
    array
        3 x 3 homography in ``dtype``.

    """
    reference = np.asarray(reference, dtype)
    inverse = darsia.linalg.solve(reference, np.eye(3, dtype=reference.dtype))
    return inverse @ np.asarray(own, dtype)


def _material_reference(
    ctx: StepContext, run: str, labels: darsia.Image, folder: Path
) -> dict[str, Any]:
    """The tables of a run's reference run in its material illumination fit: the
    reference run's stored shape-corrected baseline and facies, copied into a
    folder of the task in the read of the reference's rig, the fit mask its own
    fit read, and the homography from this run's grid to the reference's, in the
    rig's precision; none for the reference run itself, or without one."""
    named = reference_run(ctx.tables)
    if not named or named[0] == run:
        return {}
    own, other = run_tables(ctx.tables, run), run_tables(ctx.tables, named[0])
    configured = grid_model(other)

    files = ("shape_corrected_baseline.npz", "facies.npz")

    def stored(source) -> darsia.Image | None:
        """Copy the files of the reference's rig that the fit reads into the
        task's folder, and read the reference's grid (its labels) where no rig
        geometry of the reference is configured."""
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(Path(source) / name, folder / name)
        return None if configured is not None else read_labels(Path(source))

    grid = ctx.read("rig", stored, run=named[0])
    precision = run_settings(ctx, run, ("corrections",)).corrections.precision
    homography = reference_homography(
        _grid_to_rig(other, grid),
        _grid_to_rig(own, labels),
        np.dtype(precision or np.float64),
    )
    table = {
        "baseline": str(folder / files[0]),
        "facies": str(folder / files[1]),
        "homography": homography.tolist(),
    }
    fit = other.get("corrections", {}).get("material_illumination", {})
    if fit.get("fit_mask") is not None:
        table["fit_mask"] = str(fit["fit_mask"])
    return {"corrections": {"material_illumination": {"reference": table}}}


def _rig(ctx: StepContext) -> None:
    """The run's rig, with its sand map as the labels and their facies, fitted on
    its reference run's grid where the configuration names one, and with the run's
    measured model of its grid (``Rig.grid_to_rig``) and the canvas its pixels were
    measured on (``Rig.measurement_canvas``), by which the depth rule is evaluated
    on its grid where the run's configuration chooses it, and each pixel's area is
    measured (:func:`darsia.grid_pixel_areas`, with the run's optical area factor),
    and with the pixels the run reports on (``Rig.reporting_domain``). The model is
    that of the sand map's grid, so a rig on another grid (e.g. scaled to whole
    metric cells) with a model is refused, as is a canvas that does not hold the
    grid."""
    run = ctx.task.run
    labels = ctx.read("sand-map", lambda folder: Path(folder) / "labels.npz")
    sand_map = darsia.imread(labels)
    with TemporaryDirectory() as folder:
        reference = Path(folder) / "reference"
        made = _material_reference(ctx, run, sand_map, reference)
        config = run_settings(ctx, run, list_required_sections(setup_rig), made)
        facies = Path(folder) / "facies.npz"
        make_facies(config, sand_map).save(facies)
        # A depth rule evaluated on the grid needs no depth map.
        rule = config.depth.interpolation == "triangulation"
        rig = make_rig(Rig, config, labels=labels, facies=facies, depth=not rule)
    rig.grid_to_rig = grid_model(run_tables(ctx.tables, run))
    if rig.grid_to_rig is not None and rig.baseline.img.shape[:2] != sand_map.shape[:2]:
        raise ValueError(
            "[labeling.sand_map] grid_to_rig is the measured model of the sand "
            "map's grid; the rig's grid is another one."
        )
    rig.measurement_canvas = grid_canvas(run_tables(ctx.tables, run))
    if rig.measurement_canvas is not None and not rig.measurement_canvas.holds(
        *sand_map.shape[:2]
    ):
        raise ValueError("[labeling.sand_map] the canvas does not hold the grid.")
    if rig.grid_to_rig is not None and rig.pixel_area is not None:
        rig.pixel_area.img = darsia.grid_pixel_areas(
            rig.pixel_area,
            rig.grid_to_rig,
            rig.pixel_area.img.dtype,
            config.corrections.grid.area_factor,
        )
    rig.reporting_domain = grid_reporting_domain(run_tables(ctx.tables, run), sand_map)
    if rule:
        rig.depth = grid_depth(rig, Path(config.depth.measurements))
        rig.setup_geometry()
    ctx.publish(lambda folder: rig.save(Path(folder)))


def corrected_photo(
    rig: Rig, photo: Path
) -> tuple[darsia.Image, darsia.ProjectiveRegistration | None]:
    """A photo read through the rig's corrections, as the run's corrected images
    are read: with the rig's registration, registered to the setup's reference.

    Parameters
    ----------
    rig : Rig
        The run's rig, its experiment loaded.
    photo : Path
        The photo.

    Returns
    -------
    tuple
        The corrected image, and its registration (None without the rig's).

    """
    if rig.registration is None:
        return rig.read_image(photo), None
    return rig.read_registered_image(photo)


def _corrected_image(ctx: StepContext) -> None:
    """The run's photos of the task read through the rig's corrections; with the
    rig's registration, each one registered to the setup's reference, its
    homography stored beside it."""
    run = ctx.task.run
    rig = ctx.read("rig", read_rig)
    rig.load_experiment(make_experiment(run_settings(ctx, run, EXPERIMENT)))
    for frame in ctx.task.frames:
        if ctx.done(frame):
            continue
        image, found = corrected_photo(rig, ctx.photo(run, frame))

        def write(folder, image=image, found=found):
            image.save(Path(folder) / "image.npz")
            if found is not None:
                darsia.write_frame_registration(folder, found)

        ctx.publish(write, frame=frame)


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
    evaluated on the run's grid, its image mapping and its material fit mask."""
    corrections = tables.get("corrections", {})
    mapping = corrections.get("mapping") or {}
    mask = (corrections.get("material_illumination") or {}).get("fit_mask")
    found = [mapping.get("path"), mask]
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
    Memory(grid=2.0),
    _sand_map,
    sections=tuple(list_required_sections(setup_sand_map)),
    files=_drawing_files,
)
"""The run's sand map (R2)."""
RIG = Step(
    "rig",
    {"sand-map": Scope.OWN, "rig": Scope.ALL},
    Order.RUN,
    Memory(camera=6.0, grid=12.0),
    _rig,
    sections=(*list_required_sections(setup_rig), MATERIAL_REFERENCE),
    files=_rig_files,
    stamped=_rig_stamped,
    runs=reference_run,
)
"""The run's rig; a run other than the reference run of the material fit reads
the reference's rig."""
CORRECTED_IMAGE = Step(
    "corrected-image",
    {"rig": Scope.OWN},
    Order.FRAMES,
    Memory(camera=6.0, grid=12.0),
    _corrected_image,
    sections=EXPERIMENT,
    files=protocol_files,
)
"""Each frame's corrected image (R1)."""
STEPS = (SAND_MAP, RIG, CORRECTED_IMAGE)
"""The Setup's steps."""
