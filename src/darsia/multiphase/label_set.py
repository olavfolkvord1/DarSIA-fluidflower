"""Labelled regions of a frame, and the canvas they are drawn on.

A label set holds the labelled regions of one frame of a run, each with a mask on
the frame's grid. The set records its canvas: the corrected images of the rig it
was drawn on, as the rig describes them (``Rig.canvas``), i.e., the source format,
the corrections as plain settings and the grid in rig metres. A phase-model
training samples a set only on the canvas of its images, or by the caller's
explicit choice, since a set drawn on another canvas would label the wrong pixels
or colours without notice.

"""

from dataclasses import dataclass

import numpy as np


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
