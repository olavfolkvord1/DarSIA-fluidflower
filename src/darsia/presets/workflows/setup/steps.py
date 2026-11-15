"""The Setup's steps: each run's sand map, rig and corrected images.

Each run is set up from the job's configuration and its own tables: the tables
under ``[runs.<run>]`` are merged over the job's for that run, e.g. its photo
folder, baseline, protocols and sand-map adjustment. The rig is the official one
(:func:`darsia.presets.workflows.setup.setup_rig.make_rig`), with the run's sand
map as its labels and the facies of those labels
(:func:`darsia.presets.workflows.setup.setup_facies.make_facies`); each corrected
image is a photo read through the rig's corrections, once per Setup.

A step records the configuration sections it reads through the run's official
configuration (:attr:`darsia.jobs.step.Step.sections`) and the content of the
small files they name that it reads (:attr:`darsia.jobs.step.Step.files`), e.g.
the protocols of the run's photo folders, so that a result made before one of
them was edited is not reused.

"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import toml

import darsia
from darsia.jobs.step import RUNS, Memory, Order, Scope, Step, StepContext
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

EXPERIMENT = ("data", "protocols")
"""The sections of a run's experiment: its photos and protocols."""


def _file(value: Any) -> str:
    """The file of a configured protocol: a path, or a [path, sheet] pair."""
    return str(value[0]) if isinstance(value, (list, tuple)) else str(value)


def _holds(folder: Any, project: Path, folders: Sequence[Path]) -> bool:
    """Whether a configured data folder holds one of a run's photo folders."""
    base = os.path.normcase(os.path.normpath(project / str(folder)))
    for photos in folders:
        found = os.path.normcase(os.path.normpath(photos))
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


def _rig(ctx: StepContext) -> None:
    """The run's rig, with its sand map as the labels and their facies."""
    labels = ctx.read("sand-map", lambda folder: Path(folder) / "labels.npz")
    config = run_settings(ctx, ctx.task.run, list_required_sections(setup_rig))
    with TemporaryDirectory() as folder:
        facies = Path(folder) / "facies.npz"
        make_facies(config, darsia.imread(labels)).save(facies)
        rig = make_rig(Rig, config, labels=labels, facies=facies)
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
    """The run's protocols and the facies' properties."""
    props = tables.get("facies", {}).get("props")
    found = protocol_files(tables, project, folders)
    return found + ([str(props)] if props is not None else [])


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
    {"sand-map": Scope.OWN},
    Order.RUN,
    Memory(camera=6.0, grid=12.0),
    _rig,
    sections=tuple(list_required_sections(setup_rig)),
    files=_rig_files,
)
"""The run's rig."""
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
