"""Locate an injection point from the first photographs of an injection.

Gas leaves the injection port and rises, and the indicator changes color
around it. The port is therefore at the foot of the plume that appears in the
photographs taken right after the injection starts. This module compares
those photographs with the photographs taken just before the start and
returns the foot of the plume in pixels of the raw photograph.

The module only needs numpy and Pillow (no darsia import, Python 3.10), so
series tools can load it from its source file.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FrameRecord",
    "InjectionPointEstimate",
    "detect_injection_point",
    "frames_around",
    "load_reduced",
    "read_imaging_protocol",
]


@dataclass(frozen=True)
class FrameRecord:
    path: Path
    time: dt.datetime


@dataclass
class InjectionPointEstimate:
    """Foot of the first plume, in pixels of the full-resolution photograph."""

    row: float
    col: float
    status: str
    frame: str
    plume_area_px: int
    foot_rows: tuple[float, float]
    foot_cols: tuple[float, float]
    first_change_frame: str
    threshold: float
    factor: int
    image_shape: tuple[int, int]
    reference_frames: list[str] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frames


def read_imaging_protocol(path: Path, folder: Path | None = None) -> list[FrameRecord]:
    """Read a native imaging protocol (``path``, ``datetime`` columns).

    Relative image paths are taken relative to ``folder`` (the image folder the
    protocol belongs to), or to the protocol's own folder.
    """

    protocol = Path(path)
    base = Path(folder) if folder is not None else protocol.parent
    rows: list[FrameRecord] = []
    with protocol.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            raw = str(row.get("path") or "").strip()
            stamp = str(row.get("datetime") or "").strip()
            if not raw or not stamp:
                continue
            image = Path(raw)
            rows.append(
                FrameRecord(
                    image if image.is_absolute() else base / image,
                    dt.datetime.fromisoformat(stamp),
                )
            )
    return rows


def frames_around(
    frames: Iterable[FrameRecord],
    start: dt.datetime,
    *,
    before_s: float = 90.0,
    after_s: float = 150.0,
    guard_s: float = 5.0,
) -> tuple[list[FrameRecord], list[FrameRecord]]:
    """Split frames into those before ``start`` and those just after it."""

    ordered = sorted(frames, key=lambda frame: frame.time)
    before = [
        frame
        for frame in ordered
        if start - dt.timedelta(seconds=before_s)
        <= frame.time
        <= start - dt.timedelta(seconds=guard_s)
    ]
    after = [
        frame
        for frame in ordered
        if start - dt.timedelta(seconds=guard_s)
        < frame.time
        <= start + dt.timedelta(seconds=after_s)
    ]
    return before, after


def load_reduced(path: Path, factor: int = 4) -> np.ndarray:
    """Read a photograph at 1/factor resolution as float32 RGB in [0, 1].

    JPEG files are decoded at reduced size directly (DCT scaling).
    """

    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        if factor > 1:
            image.draft("RGB", (width // factor, height // factor))
        image = image.convert("RGB")
        target = (max(1, width // factor), max(1, height // factor))
        if image.size != target:
            image = image.resize(target, Image.BOX)
        return np.asarray(image, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Image operations without scipy


def _box_mean(field_: np.ndarray, size: int) -> np.ndarray:
    """Mean over a size x size window (edge-padded), via an integral image."""

    if size <= 1:
        return field_.astype(np.float32, copy=True)
    pad = size // 2
    padded = np.pad(field_.astype(np.float64), pad, mode="edge")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(0).cumsum(1)
    rows, cols = field_.shape
    total = (
        integral[size : size + rows, size : size + cols]
        - integral[:rows, size : size + cols]
        - integral[size : size + rows, :cols]
        + integral[:rows, :cols]
    )
    return (total / (size * size)).astype(np.float32)


def _dilate(mask: np.ndarray, size: int) -> np.ndarray:
    return _box_mean(mask.astype(np.float32), size) > 0.0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a sparse mask, as (n, 2) row/col arrays."""

    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    components: list[np.ndarray] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        members = [seed]
        while queue:
            row, col = queue.popleft()
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbour = (row + d_row, col + d_col)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
                        members.append(neighbour)
        components.append(np.asarray(members, dtype=np.int64))
    return components


def _robust_threshold(difference: np.ndarray, valid: np.ndarray, k: float) -> float:
    values = difference[valid]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    return median + k * max(mad, 1e-4)


# ---------------------------------------------------------------------------
# Detection


def detect_injection_point(
    before: Sequence[Path],
    after: Sequence[Path],
    *,
    factor: int = 4,
    smoothing: int = 5,
    sigma: float = 8.0,
    min_area_fraction: float = 2e-5,
    foot_fraction: float = 0.02,
    plume_frames: int = 4,
    region: Mapping[str, int] | None = None,
    prior: tuple[float, float] | None = None,
    search_radius: float | None = None,
) -> InjectionPointEstimate:
    """Foot of the plume that appears after the injection start.

    ``before`` are photographs before the start (at least one), ``after`` the
    photographs that follow it, in time order. Changes that already fluctuate
    between the ``before`` photographs (water surface, reflections) are
    ignored, and a change counts only if it persists in all later photographs.
    The first photograph with such a change marks the arrival of gas. The plume
    is the persistent region, ``plume_frames`` photographs later, that contains
    this first change; its foot is the centroid of its lowest ``foot_fraction``
    of the image height.

    ``region`` (rows/cols of the full photograph: ``top``, ``bottom``, ``left``,
    ``right``) limits the search, for example to the rig interior. ``prior``
    (row, col) with ``search_radius`` (pixels of the full photograph) limits it
    to a disc, for example around the port found in an earlier injection when
    the water around the port is already colored.
    """

    if not before:
        raise ValueError("At least one photograph before the start is required")
    if len(after) < 2:
        raise ValueError("At least two photographs after the start are required")
    reference_stack = np.stack([load_reduced(path, factor) for path in before])
    reference = np.median(reference_stack, axis=0)
    shape = reference.shape[:2]
    valid = np.ones(shape, dtype=bool)
    if region:
        top = int(region.get("top", 0)) // factor
        bottom = int(region.get("bottom", shape[0] * factor)) // factor
        left = int(region.get("left", 0)) // factor
        right = int(region.get("right", shape[1] * factor)) // factor
        valid[:] = False
        valid[max(0, top) : max(0, bottom), max(0, left) : max(0, right)] = True
    if prior is not None and search_radius:
        rows_grid, cols_grid = np.ogrid[: shape[0], : shape[1]]
        centre_row = (float(prior[0]) + 0.5) / factor - 0.5
        centre_col = (float(prior[1]) + 0.5) / factor - 0.5
        valid &= (rows_grid - centre_row) ** 2 + (cols_grid - centre_col) ** 2 <= (
            float(search_radius) / factor
        ) ** 2

    def difference(image: np.ndarray) -> np.ndarray:
        return _box_mean(np.sqrt(((image - reference) ** 2).sum(axis=2)), smoothing)

    notes: list[str] = []
    noisy = np.zeros(shape, dtype=bool)
    if len(before) >= 2:
        for image in reference_stack:
            change = difference(image)
            noisy |= change > _robust_threshold(change, valid, sigma)
        noisy = _dilate(noisy, 4 * smoothing + 1)
        if noisy[valid].mean() > 0.2:
            notes.append(
                f"{noisy[valid].mean():.0%} of the search area already changes "
                "before the start"
            )
    usable = valid & ~noisy
    names = [Path(path).name for path in after]

    def result(status: str, **values) -> InjectionPointEstimate:
        defaults = dict(
            row=float("nan"),
            col=float("nan"),
            frame="",
            plume_area_px=0,
            foot_rows=(float("nan"), float("nan")),
            foot_cols=(float("nan"), float("nan")),
            first_change_frame="",
            threshold=float("nan"),
        )
        defaults.update(values)
        return InjectionPointEstimate(
            status=status,
            factor=factor,
            image_shape=(shape[0] * factor, shape[1] * factor),
            reference_frames=[Path(path).name for path in before],
            frames=names,
            notes=notes,
            **defaults,
        )

    if not usable.any():
        notes.append("No usable pixels in the search area")
        return result("no_plume")
    changes = [difference(load_reduced(path, factor)) for path in after]
    threshold = _robust_threshold(changes[0], usable, sigma)
    masks = [(change > threshold) & usable for change in changes]
    # A change persists if it is present in every later photograph.
    persistent = [mask.copy() for mask in masks]
    for index in range(len(persistent) - 2, -1, -1):
        persistent[index] &= persistent[index + 1]
    min_area = max(4, int(round(min_area_fraction * shape[0] * shape[1])))
    first_index = next(
        (index for index, mask in enumerate(persistent) if mask.sum() >= min_area),
        None,
    )
    if first_index is None:
        notes.append("No persistent change after the start")
        return result("no_plume", threshold=float(threshold))
    seeds = _dilate(persistent[first_index], 3)
    plume_index = min(first_index + max(0, plume_frames), len(after) - 2)
    plume_index = max(plume_index, first_index)
    candidates = [
        part
        for part in _components(persistent[plume_index])
        if len(part) >= min_area and seeds[part[:, 0], part[:, 1]].any()
    ]
    if not candidates:
        notes.append("The plume could not be followed from the first change")
        candidates = [
            part
            for part in _components(persistent[first_index])
            if len(part) >= min_area
        ] or _components(persistent[first_index])
        plume_index = first_index
    plume = max(candidates, key=len)
    status = "ok"
    if len(candidates) > 1 and len(sorted(candidates, key=len)[-2]) > 0.5 * len(plume):
        status = "check"
        notes.append("Several plume-like regions start at the first change")
    if notes and status == "ok" and any("before the start" in note for note in notes):
        status = "check"
    rows = plume[:, 0].astype(np.float64)
    cols = plume[:, 1].astype(np.float64)
    foot_height = max(2.0, foot_fraction * shape[0])
    foot = rows >= rows.max() - foot_height
    foot_row = float(rows[foot].mean())
    foot_col = float(cols[foot].mean())
    scale = float(factor)
    return result(
        status,
        row=(foot_row + 0.5) * scale - 0.5,
        col=(foot_col + 0.5) * scale - 0.5,
        frame=names[plume_index],
        plume_area_px=int(len(plume) * factor * factor),
        foot_rows=(
            float(rows[foot].min() * scale),
            float((rows[foot].max() + 1) * scale),
        ),
        foot_cols=(
            float(cols[foot].min() * scale),
            float((cols[foot].max() + 1) * scale),
        ),
        first_change_frame=names[first_index],
        threshold=float(threshold),
    )
