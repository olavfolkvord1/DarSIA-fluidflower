"""Phase-model training: two logistic heads from labelled regions of frames.

Training has two results. The *training samples* hold the optical features
(:class:`~darsia.PhaseFeatures`) of up to a set number of pixels of each labelled
region of a run's corrected frames. The *phase model* is fitted on the samples
and written as a folder that :func:`~darsia.read_phase_model` reads.

A labelled region has a phase class. For each of the two heads (CO2 presence,
and free gas within the CO2) a class gives a pixel target, a regional target, or
neither. A pixel target labels every sampled pixel of the region (e.g. water,
aqueous CO2, free gas). A regional target only says whether the region as a
whole holds the phase (e.g. a region of mixed gas and aqueous CO2 holds gas); its
pixels form a bag in which, for a positive target, the highest scoring pixels
must score high, and for a negative target all pixels must score low.

All arithmetic runs in the floating dtype of the training features.

"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from darsia.multiphase.phase_model import PHASE_FEATURE_NAMES, PhaseFeatures

_TARGETS = {
    "water": ((0, 0), (None, None)),
    "aqueous": ((1, 1), (0, 0)),
    "gas": ((1, 1), (1, 1)),
    "mixed_gas_aqueous": ((None, 1), (None, 1)),
    "mixed_aqueous_water": ((None, 1), (0, 0)),
    "mixed_gas_aqueous_water": ((None, 1), (None, 1)),
    "co2_present_region": ((None, 1), (None, None)),
    "co2_absent_region": ((None, 0), (None, None)),
    "gas_present_region": ((None, 1), (None, 1)),
    "aqueous_present_region": ((None, 1), (None, 0)),
    "gas_absent_region": ((None, None), (0, 0)),
    "aqueous_absent_region": ((None, None), (None, None)),
    "aqueous_present_gas_absent_region": ((None, 1), (0, 0)),
    "current_experiment_co2_artifact": ((0, 0), (None, None)),
}
"""Per phase class and head (CO2 presence, gas within CO2): the pixel target and
the regional target, None where the class gives none."""

PHASE_CLASSES = tuple(_TARGETS)
"""Phase classes of labelled regions."""


@dataclass(frozen=True)
class PhaseTrainingSettings:
    """Settings of the phase-model training; one configuration table."""

    pixels_per_region: int
    """Largest number of pixels sampled from one labelled region."""
    texture_sigma_mm: float
    """Standard deviation (mm) of the Gaussian defining the image texture."""


@dataclass(frozen=True)
class LabelledRegion:
    """A labelled region of one frame of a run."""

    name: str
    """Unique name of the region; it seeds the choice of its sampled pixels."""
    run: str
    """The run of the frame; the cross-validation leaves out whole runs."""
    phase: str
    """Phase class of the region, one of :data:`PHASE_CLASSES`."""
    weight: float
    """Total weight of the region (e.g. its label's confidence)."""


@dataclass(frozen=True)
class TrainingSamples:
    """Optical features of sampled pixels of labelled regions."""

    features: np.ndarray
    """Feature matrix (rows, 36), columns as :data:`~darsia.PHASE_FEATURE_NAMES`."""
    owners: np.ndarray
    """Index of each row's region in :attr:`regions`."""
    regions: tuple[LabelledRegion, ...]
    """The regions with at least one sampled pixel."""
    pixel_size_mm: float
    """Pixel size of the grid of the sampled frames."""
    texture_sigma_mm: float
    """Texture scale (mm) of the features."""


def _sample(mask: np.ndarray, limit: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of at most ``limit`` pixels of a mask, chosen by the name."""
    rows, columns = np.nonzero(mask)
    if rows.size > limit:
        seed = int.from_bytes(
            hashlib.sha256(name.encode("utf-8")).digest()[:8], "little"
        )
        chosen = np.sort(
            np.random.default_rng(seed).choice(rows.size, size=limit, replace=False)
        )
        rows, columns = rows[chosen], columns[chosen]
    return rows, columns


def frame_samples(
    current: np.ndarray,
    baseline: np.ndarray,
    previous: np.ndarray | None,
    regions: list[LabelledRegion],
    masks: list[np.ndarray],
    *,
    settings: PhaseTrainingSettings,
    pixel_size_mm: float,
) -> TrainingSamples:
    """Training samples of the labelled regions of one frame.

    Each region is sampled where its mask meets the support of the frame's
    features: all pixels there, or a reproducible choice of
    ``settings.pixels_per_region`` of them, seeded by the region's name. A region
    without a supported pixel gives no samples and is left out.

    Parameters
    ----------
    current, baseline : array
        Linear RGB images of shape (rows, cols, 3) on one grid.
    previous : array or None
        Linear RGB image of the run's previous frame on the same grid.
    regions : list of LabelledRegion
        The labelled regions of the frame.
    masks : list of array
        One boolean mask per region on the frame's grid: the pixels that may be
        sampled (e.g. the drawn region within the porous sand).
    settings : PhaseTrainingSettings
        Number of sampled pixels per region and texture scale.
    pixel_size_mm : float
        Pixel size of the images' grid.

    Returns
    -------
    TrainingSamples
        Features in the images' floating dtype.

    Raises
    ------
    ValueError
        If a region's phase class is not one of :data:`PHASE_CLASSES`.

    """
    features = PhaseFeatures(
        current,
        baseline,
        previous,
        texture_sigma_mm=settings.texture_sigma_mm,
        pixel_size_mm=pixel_size_mm,
    )
    kept, rows, columns = [], [], []
    for region, mask in zip(regions, masks, strict=True):
        if region.phase not in _TARGETS:
            raise ValueError(f"Unknown phase class {region.phase!r} of {region.name}.")
        sampled = _sample(
            mask & features.support, settings.pixels_per_region, region.name
        )
        if sampled[0].size:
            kept.append(region)
            rows.append(sampled[0])
            columns.append(sampled[1])
    if kept:
        matrix = features.matrix((np.concatenate(rows), np.concatenate(columns)))
    else:
        matrix = np.empty((0, len(PHASE_FEATURE_NAMES)), features.dtype)
    return TrainingSamples(
        features=matrix,
        owners=np.repeat(np.arange(len(kept), dtype=np.int32), [r.size for r in rows]),
        regions=tuple(kept),
        pixel_size_mm=pixel_size_mm,
        texture_sigma_mm=settings.texture_sigma_mm,
    )


def join_samples(parts: list[TrainingSamples]) -> TrainingSamples:
    """Join the training samples of several frames, in the given order.

    Raises
    ------
    ValueError
        If the parts differ in pixel size or texture scale.

    """
    first = parts[0]
    if any(
        (part.pixel_size_mm, part.texture_sigma_mm)
        != (first.pixel_size_mm, first.texture_sigma_mm)
        for part in parts
    ):
        raise ValueError("Training samples of different grids or texture scales.")
    offsets = np.cumsum([0] + [len(part.regions) for part in parts])
    return TrainingSamples(
        features=np.concatenate([part.features for part in parts]),
        owners=np.concatenate(
            [part.owners + offset for part, offset in zip(parts, offsets)]
        ).astype(np.int32),
        regions=tuple(region for part in parts for region in part.regions),
        pixel_size_mm=first.pixel_size_mm,
        texture_sigma_mm=first.texture_sigma_mm,
    )


def write_training_samples(path: str | Path, samples: TrainingSamples) -> None:
    """Write training samples to one compressed NumPy archive."""
    regions = samples.regions
    np.savez_compressed(
        path,
        features=samples.features,
        owners=samples.owners,
        names=np.array([region.name for region in regions], dtype=str),
        runs=np.array([region.run for region in regions], dtype=str),
        phases=np.array([region.phase for region in regions], dtype=str),
        weights=np.array([region.weight for region in regions], dtype=float),
        pixel_size_mm=samples.pixel_size_mm,
        texture_sigma_mm=samples.texture_sigma_mm,
    )


def read_training_samples(path: str | Path) -> TrainingSamples:
    """Read training samples written by :func:`write_training_samples`."""
    with np.load(path) as archive:
        return TrainingSamples(
            features=archive["features"],
            owners=archive["owners"],
            regions=tuple(
                LabelledRegion(str(name), str(run), str(phase), float(weight))
                for name, run, phase, weight in zip(
                    archive["names"],
                    archive["runs"],
                    archive["phases"],
                    archive["weights"],
                )
            ),
            pixel_size_mm=float(archive["pixel_size_mm"]),
            texture_sigma_mm=float(archive["texture_sigma_mm"]),
        )
