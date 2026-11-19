"""Each sand's colour path fitted on the colour changes of a run's photographs.

A frame's colour change is its corrected image minus the corrected baseline,
sampled on a regular grid of pixels in each sand (:func:`sample_color_changes`).
Over a run's frames, the samples of a sand are ranked by the length of their
colour change, and the path's key colours are the channel-wise medians of the
samples in bands around given fractions of that ranking, from the baseline's
colour, zero, on (:func:`fit_color_path`). The paths of a series' runs share one
scale per sand, the longest of its paths, so that a path's value at each key
colour is its distance along the path over that scale
(:func:`color_path_readings`).

"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorPathFitSettings:
    """How each sand's colour path is fitted on a run's photographs."""

    sands: Sequence[int]
    """The facies ids whose colour paths are fitted."""
    stride_pixels: int
    """Distance in pixels, along rows and columns, of the sampled pixels."""
    minimum_samples: int
    """Fewest samples of a sand over the run for a path."""
    fractions: Sequence[float]
    """Fractions of the ranked lengths of the colour changes at which the key
    colours are taken, in increasing order."""
    band: float
    """Half width, as a fraction of the ranking, of the band of samples whose
    median is a key colour."""
    minimum_span: float
    """Length of the colour change at the last fraction that a path must
    exceed."""
    minimum_step: float
    """Length a key colour must lie from the one before, or it is left out."""


def sample_color_changes(
    current: np.ndarray,
    baseline: np.ndarray,
    facies: np.ndarray,
    sands: Sequence[int],
    stride: int,
) -> dict[int, np.ndarray]:
    """A frame's colour changes in each sand, on every ``stride``-th pixel of its
    rows and columns from the first, where the frame and the baseline are finite.

    Parameters
    ----------
    current : array
        The frame's corrected image (rows, cols, 3).
    baseline : array
        The corrected baseline on the same grid.
    facies : array
        Integer facies of each pixel (rows, cols).
    sands : sequence of int
        The facies sampled.
    stride : int
        Distance of the sampled pixels.

    Returns
    -------
    dict
        Each sand's colour changes (samples, 3), in row-major order of the
        sampled pixels, in the images' precision.

    """
    current, baseline = np.asarray(current), np.asarray(baseline)
    support = np.isfinite(current).all(axis=2) & np.isfinite(baseline).all(axis=2)
    rows, cols = np.mgrid[0 : facies.shape[0] : stride, 0 : facies.shape[1] : stride]
    samples = {}
    for sand in sorted(int(s) for s in sands):
        selected = support[rows, cols] & (facies[rows, cols] == sand)
        at = rows[selected], cols[selected]
        samples[sand] = current[at] - baseline[at]
    return samples


def fit_color_path(
    samples: Sequence[np.ndarray], settings: ColorPathFitSettings
) -> np.ndarray | None:
    """A sand's colour path from its colour changes over a run's frames.

    The samples are ranked by their length. Each key colour is the channel-wise
    median of the samples whose length lies between the quantiles at a fraction
    minus and plus the band; the path starts at zero, the baseline's colour, and
    a key colour closer than the minimum step to the one before is left out. The
    quantiles compute in the samples' floating precision.

    Parameters
    ----------
    samples : sequence of array
        The sand's colour changes (samples, 3) of each frame, in frame order.
    settings : ColorPathFitSettings
        Settings of the fit.

    Returns
    -------
    array or None
        The key colours (knots, 3), or None if the sand has too few samples, too
        short colour changes or no key colour beyond zero.

    """
    if sum(len(values) for values in samples) < settings.minimum_samples:
        return None
    values = np.concatenate(samples)
    dtype = np.result_type(values, np.float32).type
    norms = np.linalg.norm(values, axis=1)
    if np.quantile(norms, dtype(settings.fractions[-1])) <= settings.minimum_span:
        return None
    rows = []
    for fraction in settings.fractions:
        fraction, band = dtype(fraction), dtype(settings.band)
        lower, upper = np.quantile(
            norms,
            np.asarray((max(0.0, fraction - band), min(1.0, fraction + band)), dtype),
        )
        rows.append(np.median(values[(norms >= lower) & (norms <= upper)], axis=0))
    knots = np.vstack((np.zeros(3, values.dtype), np.asarray(rows, values.dtype)))
    keep = np.r_[
        True, np.linalg.norm(np.diff(knots, axis=0), axis=1) > settings.minimum_step
    ]
    knots = knots[keep]
    return knots if len(knots) > 1 else None


def fit_run_color_paths(
    images: Iterable[np.ndarray],
    baseline: np.ndarray,
    facies: np.ndarray,
    settings: ColorPathFitSettings,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Each sand's colour path over a run's frames (:func:`sample_color_changes`,
    :func:`fit_color_path`).

    Parameters
    ----------
    images : iterable of array
        The run's corrected images, in frame order; read one at a time.
    baseline : array
        The run's corrected baseline.
    facies : array
        Integer facies of each pixel.
    settings : ColorPathFitSettings
        Settings of the fit.

    Returns
    -------
    dict
        The key colours of each sand that has a path, by facies id.
    dict
        The number of samples of each sand.

    """
    samples: dict[int, list] = {int(sand): [] for sand in settings.sands}
    for image in images:
        found = sample_color_changes(
            image, baseline, facies, settings.sands, settings.stride_pixels
        )
        for sand, values in found.items():
            samples[sand].append(values)
    paths = {}
    for sand, rows in samples.items():
        knots = fit_color_path(rows, settings)
        if knots is not None:
            paths[sand] = knots
    counts = {
        sand: sum(len(values) for values in rows) for sand, rows in samples.items()
    }
    return paths, counts


def path_distances(knots: np.ndarray) -> np.ndarray:
    """Cumulative distances along a path's key colours, from zero, in their
    precision.

    Parameters
    ----------
    knots : array
        The key colours (knots, 3).

    Returns
    -------
    array
        The distance of each key colour along the path.

    """
    knots = np.asarray(knots)
    distances = np.zeros(len(knots), knots.dtype)
    np.cumsum(
        np.linalg.norm(np.diff(knots, axis=0), axis=1),
        dtype=knots.dtype,
        out=distances[1:],
    )
    return distances


def color_path_readings(
    paths: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, dict[int, dict]]:
    """Each run's colour paths read on one scale per sand: the longest of the
    sand's paths over the runs. A key colour's value is its distance along the
    path over the scale, and the signal function is the identity on [0, 1].

    Parameters
    ----------
    paths : Mapping
        Each run's key colours of each sand (:func:`fit_color_path`), by run.

    Returns
    -------
    dict
        Each run's table of each sand, by run and facies id, as the sands of a
        run's ``[reading]`` table: ``colors``, ``values``, ``supports``,
        ``table`` and ``scale``.

    Raises
    ------
    ValueError
        If a path has no positive length.

    """
    scales: dict[int, float] = {}
    for knots_of_run in paths.values():
        for sand, knots in knots_of_run.items():
            length = float(path_distances(knots)[-1])
            if not np.isfinite(length) or length <= 0:
                raise ValueError(f"The colour path of sand {sand} has no length.")
            scales[sand] = max(scales.get(sand, 0.0), length)
    return {
        run: {
            sand: {
                "colors": np.asarray(knots).tolist(),
                "values": (path_distances(knots) / scales[sand]).tolist(),
                "supports": [0.0, 1.0],
                "table": [0.0, 1.0],
                "scale": scales[sand],
            }
            for sand, knots in sorted(knots_of_run.items())
        }
        for run, knots_of_run in paths.items()
    }
