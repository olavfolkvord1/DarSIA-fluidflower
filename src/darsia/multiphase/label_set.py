"""Labelled regions of a frame, and the canvas they are drawn on.

A label set holds the labelled regions of one frame of a run, each with a mask on
the frame's grid. The set records its canvas: the corrected images of the rig it
was drawn on, as the rig describes them (``Rig.canvas``), i.e., the source format,
the corrections as plain settings and the grid in rig metres. A phase-model
training samples a set only on the canvas of its images, or by the caller's
explicit choice, since a set drawn on another canvas would label the wrong pixels
or colours without notice.

Labels are classes, not colours. A set therefore moves between two corrected
canvases of one rig, e.g. from camera JPEGs to raw photographs, to another pixel
size or another lens model, by the geometry of their grids alone
(:func:`move_labels`).

"""

from dataclasses import dataclass

import numpy as np

from darsia.image.coordinatesystem import pixel_centres
from darsia.image.image import ScalarImage


@dataclass(frozen=True)
class LabelledRegion:
    """A labelled region of one frame of a run."""

    name: str
    """Unique name of the region; it seeds the choice of its sampled pixels."""
    run: str
    """The run of the frame; the cross-validation leaves out whole runs."""
    phase: str
    """Phase class of the region, one of :data:`~darsia.PHASE_CLASSES`."""
    weight: float
    """Total weight of the region (e.g. its label's confidence)."""


@dataclass(frozen=True)
class LabelSet:
    """The labelled regions of one frame, drawn on one canvas.

    A canvas is a series': its corrections, e.g. the crop, the curvature and the
    lens model, are the same for every run. A run's placement on its camera
    images stays outside the canvas, in the per-frame registration.

    """

    regions: list[LabelledRegion]
    """The labelled regions."""
    masks: list[np.ndarray]
    """One boolean mask per region on the canvas' grid: the pixels the region
    labels (e.g. the drawn region within the porous sand)."""
    canvas: dict | None
    """The canvas the regions are drawn on (``Rig.canvas``); None for a set that
    records none, e.g. a delivered one."""
    moved_from: dict | None = None
    """The canvas the set was moved from by :func:`move_labels`; None if it was
    drawn on its canvas."""


def _settings(record: dict, prefix: str = "") -> dict:
    """Settings of a record by their dotted names; an empty table is a setting."""
    settings = {}
    for key, value in record.items():
        if isinstance(value, dict) and value:
            settings.update(_settings(value, f"{prefix}{key}."))
        else:
            settings[f"{prefix}{key}"] = value
    return settings


def canvas_differences(canvas: dict | None, other: dict | None) -> list[str]:
    """Names of the settings in which two canvases differ.

    A setting is named by its path in the record, e.g.
    ``corrections.curvature.crop.width``; a list is compared as a whole.

    Parameters
    ----------
    canvas, other : dict or None
        Canvases (``Rig.canvas``); None for a canvas that is not recorded.

    Returns
    -------
    list of str
        The differing settings, sorted; ``["canvas"]`` if a canvas is not
        recorded, since an unknown canvas never matches.

    """
    if canvas is None or other is None:
        return ["canvas"]
    first, second = _settings(canvas), _settings(other)
    return sorted(
        name
        for name in first.keys() | second.keys()
        if name not in first or name not in second or first[name] != second[name]
    )


_RESOLUTION = (
    "grid.shape",
    "grid.dimensions",
    "grid.origin",
    "corrections.grid.pixel_size_mm",
)
"""Settings of the resolution of a canvas: the grid's shape and pixel size, and the
dimensions and origin of its whole metric cells, which follow from them. The
extent before the cells extend it (``grid.extent``) belongs to the canvas, so the
cells' dimensions and origin pass only when both canvases record the same extent."""


def _extent(record: dict | None) -> dict | None:
    """The extent a canvas records before whole metric cells extend it, or None."""
    return ((record or {}).get("grid") or {}).get("extent")


def check_canvas(
    recorded: dict | None,
    canvas: dict | None,
    *,
    accept: bool,
    subject: str,
    resolution: bool = False,
) -> bool:
    """Check the canvas recorded by labels or a model against the images' canvas.

    A recorded canvas equal to the images' passes. With ``resolution``, so does
    one that differs only in resolution: the grid's shape and pixel size, with the
    same extent in rig metres (``grid.extent``), recorded by both, and every other
    setting the same. Any other difference, or a canvas that is not recorded,
    passes only by the caller's explicit choice.

    Parameters
    ----------
    recorded : dict or None
        Canvas recorded by a label set or a phase model; None if not recorded.
    canvas : dict or None
        Canvas of the images (``Rig.canvas``).
    accept : bool
        Whether the caller accepts a canvas that differs or is not recorded.
    subject : str
        What recorded the canvas, for the message, e.g. "the labels".
    resolution : bool
        Whether another resolution passes, e.g. for a phase model, which carries
        its sizes in mm.

    Returns
    -------
    bool
        Whether the canvas passed only because the caller accepted it, so the
        caller can record it.

    Raises
    ------
    ValueError
        If the canvas differs or is not recorded and is not accepted; the message
        names the differing settings.

    """
    if recorded is not None and recorded == canvas:
        return False
    differences = canvas_differences(recorded, canvas)
    waived = set(_RESOLUTION)
    if _extent(recorded) is None or _extent(recorded) != _extent(canvas):
        # Without the same extent in both, other dimensions or another origin are
        # another canvas, not another resolution.
        waived -= {"grid.dimensions", "grid.origin"}
    if resolution and set(differences) <= waived:
        return False
    if not accept:
        raise ValueError(
            f"The canvas of {subject} differs from the images' canvas; differing "
            f"settings: {', '.join(differences)}. "
            "Accept it explicitly only if this is intended."
        )
    return True


def _grid_image(grid: dict) -> ScalarImage:
    """An image on a corrected grid, with the grid's coordinates."""
    return ScalarImage(
        np.zeros(grid["shape"], np.uint8),
        dimensions=grid["dimensions"],
        origin=grid["origin"],
    )


def move_labels(labels: LabelSet, canvas: dict) -> LabelSet:
    """Move a label set onto another corrected canvas of the same rig.

    Both grids are in rig metres. Each pixel centre of the new grid
    (:func:`darsia.pixel_centres`) takes the label of the pixel of the set's grid
    that holds it (nearest neighbour), and is unlabelled outside that grid. The
    regions and their phase classes are kept, so a move between the formats, pixel
    sizes or lens models of one rig is geometry alone.

    Parameters
    ----------
    labels : LabelSet
        Label set on a corrected canvas.
    canvas : dict
        Corrected canvas to move the set onto (``Rig.canvas``).

    Returns
    -------
    LabelSet
        The regions with their masks on the new canvas, recording the canvas
        they came from.

    Raises
    ------
    ValueError
        If either canvas has no grid in rig metres. Labels on a camera frame, or
        on a crop in pixels, must first be placed on a canvas in rig metres with
        the rig's mapping (lens model and registration), which this function
        does not do.

    """
    grids = [record and record["grid"] for record in (labels.canvas, canvas)]
    if not all(grids):
        raise ValueError(
            "Labels move only between canvases in rig metres; labels on a camera "
            "frame or a crop in pixels must first be placed on such a canvas with "
            "the rig's mapping (lens model and registration)."
        )
    source, target = (_grid_image(grid) for grid in grids)
    x, y = pixel_centres(target)
    # The grids share the rig's axes, so the pixel centres along one axis of the
    # new grid give the source pixels along that axis.
    rows = np.column_stack((np.full_like(y, x[0]), y))
    cols = np.column_stack((x, np.full_like(x, y[0])))
    picked, inside = [], []
    for axis, centres in enumerate((rows, cols)):
        index = source.coordinatesystem.voxel(centres)[:, axis]
        size = source.img.shape[axis]
        picked.append(np.clip(index, 0, size - 1))
        inside.append((index >= 0) & (index < size))
    keep = np.outer(*inside)
    return LabelSet(
        labels.regions,
        [mask[np.ix_(*picked)] & keep for mask in labels.masks],
        canvas,
        moved_from=labels.canvas,
    )
