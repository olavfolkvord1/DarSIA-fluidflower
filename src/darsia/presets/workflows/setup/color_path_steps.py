"""The fit of each sand's colour path as job steps.

The step ``color-paths`` reads a run's stored rig, its corrected baseline and
facies, and the corrected images of the run's selected frames, and fits each
sand's colour path on their colour changes
(:func:`darsia.fit_run_color_paths`); the key colours are stored as
``color_paths.json`` in their precision. The step ``color-path-reading`` reads the
paths of every selected run and gives them one scale per sand
(:func:`darsia.color_path_readings`); each run's tables are stored as
``reading.json``, in the form of ``[runs.<run>.reading]``. Nothing is configured
by the steps, and the Setup's steps are unchanged.
"""

import json
from pathlib import Path

import numpy as np

import darsia
from darsia.jobs.step import Memory, Order, Scope, Step, StepContext
from darsia.presets.workflows.setup.steps import read_image, read_rig

PATHS = "color_paths.json"
"""A run's stored key colours in their result folder."""
READING = "reading.json"
"""The stored tables of the runs' readings in their result folder."""


def write_color_paths(
    folder: Path,
    paths: dict[int, np.ndarray],
    samples: dict[int, int],
    precision: np.dtype,
) -> None:
    """Store a run's key colours of each sand in their floating precision, and the
    number of samples of each sand (:data:`PATHS`)."""
    stored = {
        "precision": np.dtype(precision).name,
        "paths": {str(sand): knots.tolist() for sand, knots in paths.items()},
        "samples": {str(sand): count for sand, count in samples.items()},
    }
    (Path(folder) / PATHS).write_text(json.dumps(stored, indent=1), "utf-8")


def read_color_paths(folder: Path) -> dict[int, np.ndarray]:
    """A run's stored key colours of each sand (:data:`PATHS`), in their
    precision."""
    stored = json.loads((Path(folder) / PATHS).read_text(encoding="utf-8"))
    return {
        int(sand): np.asarray(knots, stored["precision"])
        for sand, knots in stored["paths"].items()
    }


def read_readings(folder: Path) -> dict:
    """The stored tables of the runs' readings (:data:`READING`)."""
    return json.loads((Path(folder) / READING).read_text(encoding="utf-8"))


def _color_paths(ctx: StepContext) -> None:
    """Each sand's colour path over the run's selected frames."""
    rig = ctx.read("rig", read_rig)
    images = (
        ctx.read("corrected-image", read_image, frame=frame).img
        for frame in ctx.task.frames
    )
    baseline = rig.baseline.img
    paths, samples = darsia.fit_run_color_paths(
        images, baseline, rig.facies.img, ctx.settings["color_path_fit"]
    )
    ctx.publish(
        lambda folder: write_color_paths(folder, paths, samples, baseline.dtype)
    )


def _color_path_reading(ctx: StepContext) -> None:
    """The selected runs' colour paths on one scale per sand."""
    paths = {
        run: ctx.read("color-paths", read_color_paths, run=run)
        for run in ctx.selection.frames
    }
    tables = {
        run: {"sands": {str(sand): table for sand, table in sands.items()}}
        for run, sands in darsia.color_path_readings(paths).items()
    }
    ctx.publish(
        lambda folder: (Path(folder) / READING).write_text(
            json.dumps(tables, indent=1), "utf-8"
        )
    )


COLOR_PATHS = Step(
    "color-paths",
    {"rig": Scope.OWN, "corrected-image": Scope.OWN},
    Order.RUN,
    Memory(grid=6.0),
    _color_paths,
    {"color_path_fit": darsia.ColorPathFitSettings},
)
"""Each sand's colour path of a run, over its selected frames."""
COLOR_PATH_READING = Step(
    "color-path-reading",
    {"color-paths": Scope.ALL},
    Order.PARTS,
    Memory(),
    _color_path_reading,
)
"""The selected runs' colour paths on one scale per sand."""
STEPS = (COLOR_PATHS, COLOR_PATH_READING)
"""The steps of this module."""
