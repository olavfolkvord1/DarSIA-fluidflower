"""Display means of a mass map: the values a film or figure shows, never stored.

A map is shown on a display grid of every ``step``-th cell (:func:`display_step`).
Each shown cell holds a mean of the stored cells under it, so no single grain
stands for its neighbours:

- :func:`block_mean`, the mean of the cell's known values;
- the depth-scaled mean, for films and figures only (:func:`depth_scaled`): the
  water pixels inside each connected part of a sand layer are averaged over a
  disk as wide as the rig is deep (:func:`depth_window`), weighted by pore volume,
  and each part keeps its pore-volume-weighted total (:func:`layer_mean`). It
  never reaches through another sand nor across the boundary between gas and
  water, so the grain texture goes and every level difference between layers
  stays. :func:`pooled_mean` then gives each shown cell the pore-volume-weighted
  mean under it.

Pixels whose phase is unknown have no value (:func:`shown_values`).

"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure

from darsia.multiphase.phase_rules import PhaseLabel


def shown_values(values: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """The map's values, NaN where the phase map does not know the phase."""
    return np.where(phase == PhaseLabel.UNKNOWN, np.nan, values)


def display_step(shape, cells: int) -> int:
    """The step that keeps a grid's longest shown side at or below ``cells`` cells."""
    return max(1, math.ceil(max(int(size) for size in shape) / int(cells)))


def block_mean(values: np.ndarray, step: int) -> np.ndarray:
    """The mean of the finite values of each ``step`` x ``step`` block; NaN for a
    block without one.

    Edge blocks hold fewer cells, so every stored cell lies in exactly one shown
    cell. The sums run in double precision, as our production summed them; the
    means are returned in the values' floating type.

    """
    values = np.asarray(values)
    rows, columns = (math.ceil(size / step) for size in values.shape)
    padded = np.full((rows * step, columns * step), np.nan)
    padded[: values.shape[0], : values.shape[1]] = values
    blocks = padded.reshape(rows, step, columns, step)
    known = np.isfinite(blocks)
    counts = known.sum(axis=(1, 3))
    totals = np.where(known, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return means.astype(values.dtype if values.dtype.kind == "f" else np.float64)


def depth_window(
    depth_m: np.ndarray, pixel_area_m2: np.ndarray, sand: np.ndarray
) -> tuple[int, float]:
    """The window of the depth-scaled mean: the median rig depth of the sand [mm],
    and that depth over the sand's median pixel side as a width [pixels, odd]."""
    depth_mm = float(np.median(depth_m[sand])) * 1000.0
    pixel_mm = float(np.sqrt(np.median(pixel_area_m2[sand]))) * 1000.0
    return max(1, int(round(depth_mm / pixel_mm)) | 1), depth_mm


def layer_mean(
    values: np.ndarray,
    layers: np.ndarray,
    water: np.ndarray,
    pore_volume: np.ndarray,
    window: int,
) -> np.ndarray:
    """The depth-scaled mean of a map's water pixels inside each part of a sand layer.

    A part is a 4-connected piece of one sand-map label, found once for the frame:
    two parts of one layer separated by another sand never share a mean, however
    close they lie, and each part keeps its own total.

    Parameters
    ----------
    values : array
        The map; NaN where it has no value.
    layers : array
        The placed sand-map label of every pixel.
    water : array
        The pixels averaged: water in a sand layer. Every other pixel, gas
        included, keeps its value.
    pore_volume : array
        Each pixel's pore volume, the weight of its value.
    window : int
        The width of the disk averaged over [pixels] (:func:`depth_window`).

    Returns
    -------
    array
        The map with each part's known water pixels replaced by their
        pore-volume-weighted mean over the disk around them, taken over the same
        part's known water pixels only and scaled so that the part keeps its
        pore-volume-weighted total. It is computed in double precision, as our
        production computed it.

    Raises
    ------
    ValueError
        Where the mean of a part sums to zero while the part holds CO2, so that
        its total cannot be kept.

    """
    shown = np.asarray(values, dtype=np.float64)
    weight = np.broadcast_to(np.asarray(pore_volume, dtype=np.float64), shown.shape)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window))
    disk = disk.astype(np.float64)
    out = shown.copy()
    averaged = water & np.isfinite(shown)
    parts = measure.label(np.asarray(layers, dtype=np.int64) + 1, connectivity=1)
    for number, box in enumerate(ndimage.find_objects(parts), 1):
        inside = averaged[box] & (parts[box] == number)
        if not inside.any():
            continue
        rows, columns = np.nonzero(inside)
        top, left = box[0].start, box[1].start
        part = np.s_[
            top + rows.min() : top + rows.max() + 1,
            left + columns.min() : left + columns.max() + 1,
        ]
        keep = inside[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
        weights, known = weight[part], shown[part]
        border = cv2.BORDER_CONSTANT
        numerator = cv2.filter2D(
            np.where(keep, known * weights, 0), -1, disk, borderType=border
        )
        denominator = cv2.filter2D(
            np.where(keep, weights, 0), -1, disk, borderType=border
        )
        mean = numerator[keep] / denominator[keep]
        before = float(np.sum(known[keep] * weights[keep], dtype=np.float64))
        after = float(np.sum(mean * weights[keep], dtype=np.float64))
        if after != 0:
            mean *= before / after
        elif before != 0:
            raise ValueError("The display mean cannot keep the mass of a layer part")
        out[part][keep] = mean
    return out


def depth_scaled(
    values: np.ndarray,
    phase: np.ndarray,
    saturation: np.ndarray,
    layers: np.ndarray,
    sand_layers,
    pore_volume: np.ndarray,
    depth_m: np.ndarray,
    pixel_area_m2: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """The depth-scaled mean of a map (:func:`layer_mean`) and its window: its width
    [pixels] and the median rig depth [mm] (:func:`depth_window`).

    The sand is the pixels of the ``sand_layers`` (sand-map labels) with a positive
    pore volume, depth and pixel area; its water is where the phase is known and
    the free-gas ``saturation`` is zero. ``values`` are the map's shown values
    (:func:`shown_values`).

    """
    shape = values.shape
    volume = np.broadcast_to(np.asarray(pore_volume, dtype=np.float64), shape)
    depth = np.broadcast_to(np.asarray(depth_m), shape)
    area = np.broadcast_to(np.asarray(pixel_area_m2), shape)
    sand = np.isin(layers, sand_layers)
    for amount in (volume, depth, area):
        sand &= np.isfinite(amount) & (amount > 0)
    window, depth_mm = depth_window(depth, area, sand)
    water = sand & (phase != PhaseLabel.UNKNOWN)
    water &= np.isfinite(saturation) & (saturation <= 0)
    return layer_mean(values, layers, water, volume, window), window, depth_mm


def pooled_mean(values: np.ndarray, pore_volume: np.ndarray, step: int) -> np.ndarray:
    """The pore-volume-weighted mean of the known values of each ``step`` x ``step``
    block; NaN for a block without one. With ``step`` 1 the values themselves."""
    if step == 1:
        return values
    weight = np.broadcast_to(np.asarray(pore_volume, dtype=np.float64), values.shape)
    known = np.isfinite(values) & np.isfinite(weight) & (weight > 0)
    rows = np.arange(0, values.shape[0], step)
    columns = np.arange(0, values.shape[1], step)

    def blocks(array):
        """Sums of ``array`` over the blocks."""
        return np.add.reduceat(np.add.reduceat(array, rows, axis=0), columns, axis=1)

    totals = blocks(np.where(known, values * weight, 0))
    weights = blocks(np.where(known, weight, 0))
    means = np.full(weights.shape, np.nan)
    np.divide(totals, weights, out=means, where=weights > 0)
    return means
