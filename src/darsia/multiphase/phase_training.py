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

A head is fitted in two steps: a weighted logistic regression on the pixel
targets by iteratively reweighted least squares (IRLS), then a refinement with
the bags by Adam steps. The weights balance runs and classes: each region has a
total weight independent of its pixel count, and each run's regions of one
target share one total. The features are standardized over the pixel rows and
the bags together.

The cross-validation leaves out one run at a time. A region is scored by a
quantile of its pixels' probabilities. For each regularization of the grid, the
held-out scores of all runs give one decision threshold (by the weighted
balanced accuracy of the regions); the regularization whose worst run is best
at that threshold is selected (then the mean, the stability of the per-fold
thresholds and the ROC area decide), and the head is fitted on all runs.

All arithmetic runs in the floating dtype of the training features.

"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import darsia
from darsia.multiphase.phase_model import (
    PHASE_FEATURE_NAMES,
    LogisticHead,
    PhaseFeatures,
    PhaseModel,
)

_HEADS = ("co2_presence", "gas_within_co2")

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

_ROWS_PER_SUM = 65536
"""Rows per partial sum of the feature standardization."""


@dataclass(frozen=True)
class PhaseTrainingSettings:
    """Settings of the phase-model training; one configuration table."""

    pixels_per_region: int
    """Largest number of pixels sampled from one labelled region."""
    texture_sigma_mm: float
    """Standard deviation (mm) of the Gaussian defining the image texture."""
    regularization_grid: list[float]
    """Penalties on the squared coefficients tried in the cross-validation."""
    weak_strength: float
    """Weight of the bags' loss relative to the loss of the pixel targets."""
    positive_top_fraction: float
    """Fraction of a positive bag's pixels, the highest scoring, that must score
    high."""
    mil_iterations: int
    """Largest number of Adam steps of the refinement with the bags."""
    mil_learning_rate: float
    """First Adam step size; step k uses it divided by sqrt(1 + k / 80)."""
    region_score_quantile: float
    """Quantile of a region's pixel probabilities that scores the region."""
    threshold_levels: tuple[float, float, int]
    """Fixed candidate thresholds of the region scores (first, last, count); the
    same quantiles of the scores are candidates too."""
    irls_steps: tuple[int, int]
    """Newton steps of the logistic regression; the longer run starts over when the
    shorter one does not converge."""
    irls_tolerance: float
    """The regression converges when no parameter changes by more than this."""
    mil_tolerance: float
    """The refinement stops when no parameter changes by more than this, and the
    gradient is below it too, for ``mil_stable_steps`` steps in a row."""
    mil_stable_steps: int
    """Steps in a row within the tolerance after which the refinement stops."""


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


@dataclass(frozen=True)
class PhaseTraining:
    """A trained phase model, the CO2 head's threshold and the cross-validation."""

    model: PhaseModel
    """The two heads, their texture scale and the gas head's threshold."""
    co2_threshold: float
    """Threshold of the CO2 head's region scores held out in the cross-validation;
    the phase rules take their CO2 threshold from their settings."""
    cross_validation: dict[str, dict]
    """Per head, the selected regularization and the scores of the grid."""


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


def _target(region: LabelledRegion, head: str, regional: bool) -> int | None:
    """The region's pixel or regional target for a head."""
    return _TARGETS[region.phase][_HEADS.index(head)][regional]


def _pixel_rows(samples, head, runs):
    """Rows with a pixel target of the head: features, targets and weights.

    Each region's total weight is spread evenly over its rows, each stratum of
    one run and one target is scaled to a total of one, and the weights are
    scaled to a mean of one.

    """
    dtype = samples.features.dtype.type
    order = np.argsort(samples.owners, kind="stable")
    owners = samples.owners[order]
    indices = np.arange(len(samples.regions), dtype=np.int32)
    starts = np.searchsorted(owners, indices, side="left")
    stops = np.searchsorted(owners, indices, side="right")
    rows, labels, weights, strata, codes = [], [], [], [], {}
    for index, region in enumerate(samples.regions):
        target = _target(region, head, regional=False)
        if region.run not in runs or target is None or region.weight <= 0.0:
            continue
        selected = order[starts[index] : stops[index]]
        rows.append(selected)
        labels.append(np.full(selected.size, target, dtype=np.int8))
        weights.append(np.full(selected.size, region.weight / selected.size, dtype))
        code = codes.setdefault((region.run, target), len(codes))
        strata.append(np.full(selected.size, code, dtype=np.int32))
    if not rows:
        raise ValueError(f"No pixel targets of {head} in the training runs.")
    rows, labels, weights = map(np.concatenate, (rows, labels, weights))
    strata = np.concatenate(strata)
    order = np.argsort(strata, kind="stable")
    bounds = np.flatnonzero(np.diff(strata[order])) + 1
    for positions in np.split(order, bounds):
        total = float(np.sum(weights[positions]))
        if total > 0.0:
            weights[positions] /= total
    weights *= dtype(weights.size) / max(np.sum(weights), dtype(1e-12))
    return samples.features[rows], labels, weights


def _bags(samples, head, runs):
    """Bags of the regions with only a regional target of the head.

    Returns a list of (rows, target, weight), each run's bags of one target
    sharing a total weight of one.

    """
    pending, totals = [], {}
    for index, region in enumerate(samples.regions):
        target = _target(region, head, regional=True)
        if (
            region.run not in runs
            or _target(region, head, regional=False) is not None
            or target is None
            or region.weight <= 0.0
        ):
            continue
        stratum = (region.run, target)
        rows = np.flatnonzero(samples.owners == index)
        pending.append((rows, target, stratum, region.weight))
        totals[stratum] = totals.get(stratum, 0.0) + region.weight
    return [
        (rows, target, weight / totals[stratum])
        for rows, target, stratum, weight in pending
        if totals[stratum] > 0.0
    ]


def _standardization(pure, pure_weights, features, bags, weak_strength):
    """Weighted mean and standard deviation of the features over the head's rows.

    The pixel rows carry a total weight of one and the bags ``weak_strength``,
    each bag its share spread evenly over its rows. A constant feature keeps the
    scale one.

    """
    dtype = features.dtype.type
    pure_total = np.sum(pure_weights, dtype=dtype)
    bag_total = sum((dtype(weight) for _, _, weight in bags), dtype(0))

    def parts():
        """Feature rows and their weights in partial sums."""
        for first in range(0, len(pure), _ROWS_PER_SUM):
            last = first + _ROWS_PER_SUM
            yield pure[first:last], pure_weights[first:last] / pure_total
        for rows, _, weight in bags:
            mass = dtype(weak_strength) * dtype(weight) / bag_total / dtype(len(rows))
            for first in range(0, len(rows), _ROWS_PER_SUM):
                part = rows[first : first + _ROWS_PER_SUM]
                yield features[part], np.full(len(part), mass, dtype=dtype)

    total = dtype(1) + dtype(weak_strength)
    mean = np.zeros(features.shape[1], dtype=dtype)
    for values, weights in parts():
        mean += np.sum(values * weights[:, None], axis=0)
    mean /= total
    variance = np.zeros_like(mean)
    for values, weights in parts():
        centered = values - mean
        variance += np.sum(centered * centered * weights[:, None], axis=0)
    variance /= total
    return mean, np.where(variance <= 1e-12, 1.0, np.sqrt(np.maximum(variance, 0.0)))


def _objective(matrix, labels, weights, coefficients, intercept, regularization):
    """Weighted mean logistic loss plus the coefficients' penalty."""
    logits = matrix @ coefficients + intercept
    loss = np.logaddexp(0.0, np.where(labels > 0.5, -logits, logits))
    return float(
        np.sum(weights * loss) / np.sum(weights)
        + 0.5 * regularization * np.dot(coefficients, coefficients)
    )


def _irls(matrix, labels, weights, regularization, steps, tolerance):
    """Weighted logistic regression by Newton steps with a halving line search.

    Each Newton system is solved in the features' precision
    (:func:`darsia.linalg.solve`).

    Returns the coefficients, the intercept and whether the steps converged.

    """
    dtype = matrix.dtype.type
    target = labels.astype(dtype)
    weight_sum = np.sum(weights, dtype=dtype)
    dimension = matrix.shape[1]
    coefficients = np.zeros(dimension, dtype=dtype)
    positive = np.sum(weights * target, dtype=dtype) / weight_sum
    intercept = dtype(
        np.log(np.clip(positive, 1e-6, 1 - 1e-6) / np.clip(1 - positive, 1e-6, 1))
    )
    identity = np.eye(dimension, dtype=dtype)
    objective = _objective(
        matrix, target, weights, coefficients, intercept, regularization
    )
    for _ in range(steps):
        logits = np.clip(matrix @ coefficients + intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = weights * (probability - target) / weight_sum
        curvature = (
            weights * np.maximum(probability * (1.0 - probability), 1e-7) / weight_sum
        )
        gradient = np.concatenate(
            (matrix.T @ residual + regularization * coefficients, [np.sum(residual)])
        )
        hessian = np.empty((dimension + 1, dimension + 1), dtype=dtype)
        hessian[:dimension, :dimension] = (
            matrix.T @ (matrix * curvature[:, None]) + regularization * identity
        )
        cross = matrix.T @ curvature
        hessian[:dimension, dimension] = cross
        hessian[dimension, :dimension] = cross
        hessian[dimension, dimension] = float(np.sum(curvature))
        hessian.flat[:: dimension + 2] += 1e-8
        step = darsia.linalg.solve(hessian, gradient)
        factor = dtype(1)
        for _ in range(12):
            candidate = (
                coefficients - factor * step[:dimension],
                intercept - factor * step[dimension],
            )
            value = _objective(matrix, target, weights, *candidate, regularization)
            if value <= dtype(objective) + dtype(1e-12):
                (coefficients, intercept), objective = candidate, value
                break
            factor *= 0.5
        else:
            return coefficients, intercept, False
        if float(np.max(np.abs(factor * step))) < tolerance:
            return coefficients, intercept, True
    return coefficients, intercept, False


@dataclass(frozen=True)
class _PackedBags:
    """The bags of one target, their standardized rows one after the other."""

    features: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray
    selected: np.ndarray | None
    target: int


def _pack(features, bags, target, mean, scale, weight_sum, top_fraction):
    """Standardized rows of the bags of one target; None without such bags.

    A positive bag selects ``top_fraction`` of its rows (at least one); a
    negative bag uses all rows.

    """
    dtype = features.dtype.type
    selected = [bag for bag in bags if bag[1] == target]
    if not selected:
        return None
    counts = np.asarray([rows.size for rows, _, _ in selected], dtype=np.intp)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    packed = np.empty((int(offsets[-1]), features.shape[1]), dtype=dtype)
    for index, (rows, _, _) in enumerate(selected):
        packed[offsets[index] : offsets[index + 1]] = (features[rows] - mean) / scale
    return _PackedBags(
        features=packed,
        counts=counts,
        offsets=offsets,
        weights=np.asarray(
            [dtype(weight) / dtype(weight_sum) for _, _, weight in selected], dtype
        ),
        selected=(
            np.asarray(
                [max(1, round(top_fraction * int(count))) for count in counts], np.intp
            )
            if target
            else None
        ),
        target=target,
    )


def _bag_terms(packed, coefficients, intercept):
    """Gradient, intercept gradient and loss of the packed bags.

    A positive bag's loss is the mean logistic loss of its selected highest
    scoring rows, chosen anew at each step; a negative bag's is that of all its
    rows.

    """
    dtype = coefficients.dtype.type
    logits = np.clip(packed.features @ coefficients + intercept, -30.0, 30.0)
    if not packed.target:
        weights = np.repeat(packed.weights / packed.counts.astype(dtype), packed.counts)
        residual = weights / (1.0 + np.exp(-logits))
        loss = dtype(np.dot(weights, np.logaddexp(0.0, logits)))
    else:
        residual = np.zeros_like(logits)
        loss = dtype(0)
        for index, count in enumerate(packed.selected):
            start, stop = packed.offsets[index : index + 2]
            bag = logits[start:stop]
            chosen = np.argsort(bag, kind="stable")[-count:]
            weight = packed.weights[index] / dtype(count)
            residual[start + chosen] = weight * (1.0 / (1.0 + np.exp(-bag[chosen])) - 1)
            loss += dtype(weight * np.sum(np.logaddexp(0.0, -bag[chosen])))
    return packed.features.T @ residual, np.sum(residual, dtype=dtype), loss


def _mil(pure, labels, weights, packed, coefficients, intercept, *, settings, penalty):
    """Refine the regression with the bags' loss by Adam steps.

    The gradient is capped at a norm of ten; the step size decays with the step
    count. The refinement stops early when ``settings.mil_stable_steps`` steps in
    a row change no parameter by more than ``settings.mil_tolerance`` and the
    gradient is below it too.

    """
    dtype = pure.dtype.type
    targets = labels.astype(dtype)
    weights = weights / np.sum(weights)
    strength = float(settings.weak_strength)
    coefficients = coefficients.copy()
    moments = [np.zeros_like(coefficients), np.zeros_like(coefficients)]
    bias_moments = [dtype(0), dtype(0)]
    beta1, beta2, epsilon = dtype(0.9), dtype(0.999), dtype(1e-7)
    stable = 0
    for step in range(1, settings.mil_iterations + 1):
        logits = np.clip(pure @ coefficients + intercept, -30.0, 30.0)
        residual = weights * (1.0 / (1.0 + np.exp(-logits)) - targets)
        gradient = pure.T @ residual + penalty * coefficients
        bias_gradient = np.sum(residual, dtype=dtype)
        loss = dtype(np.dot(weights, np.logaddexp(0.0, logits) - targets * logits))
        loss += dtype(0.5) * dtype(penalty) * np.dot(coefficients, coefficients)
        for bags in packed:
            if bags is not None:
                bag_gradient, bag_bias_gradient, bag_loss = _bag_terms(
                    bags, coefficients, intercept
                )
                gradient += strength * bag_gradient
                bias_gradient += strength * bag_bias_gradient
                loss += strength * bag_loss
        if (
            stable >= settings.mil_stable_steps
            and np.isfinite(loss)
            and float(np.max(np.abs(gradient))) <= settings.mil_tolerance
            and abs(float(bias_gradient)) <= settings.mil_tolerance
        ):
            break
        norm = np.linalg.norm(gradient)
        if norm > 10.0:
            gradient *= 10.0 / norm
            bias_gradient = np.clip(bias_gradient, -10.0, 10.0)
        moments[0] = beta1 * moments[0] + (1.0 - beta1) * gradient
        moments[1] = beta2 * moments[1] + (1.0 - beta2) * np.square(gradient)
        bias_moments[0] = beta1 * bias_moments[0] + (1.0 - beta1) * bias_gradient
        bias_moments[1] = beta2 * bias_moments[1] + (1.0 - beta2) * bias_gradient**2
        corrections = 1.0 - beta1**step, 1.0 - beta2**step
        rate = dtype(settings.mil_learning_rate) / np.sqrt(
            dtype(1) + dtype(step) / dtype(80)
        )
        coefficient_step = (
            rate
            * (moments[0] / corrections[0])
            / (np.sqrt(moments[1] / corrections[1]) + epsilon)
        )
        intercept_step = (
            rate
            * (bias_moments[0] / corrections[0])
            / (np.sqrt(bias_moments[1] / corrections[1]) + epsilon)
        )
        largest = max(
            float(np.max(np.abs(coefficient_step))), abs(float(intercept_step))
        )
        coefficients -= coefficient_step
        intercept -= intercept_step
        stable = stable + 1 if largest < settings.mil_tolerance else 0
    return coefficients, intercept


def _fit_head(samples, head, runs, penalty, settings):
    """One head fitted on the training runs' regions.

    Raises
    ------
    ValueError
        If the runs hold no pixel targets of both classes, or no bags.

    """
    features = samples.features
    pure, labels, weights = _pixel_rows(samples, head, runs)
    bags = _bags(samples, head, runs)
    if not bags:
        raise ValueError(f"No regional labels of {head} in the training runs.")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"The pixel targets of {head} need both classes.")
    mean, scale = _standardization(
        pure, weights, features, bags, settings.weak_strength
    )
    standardized = (pure - mean) / scale
    for steps in settings.irls_steps:
        coefficients, intercept, converged = _irls(
            standardized, labels, weights, penalty, steps, settings.irls_tolerance
        )
        if converged:
            break
    weight_sum = max(sum(weight for _, _, weight in bags), 1e-12)
    packed = [
        _pack(
            features,
            bags,
            target,
            mean,
            scale,
            weight_sum,
            float(settings.positive_top_fraction),
        )
        for target in (1, 0)
    ]
    coefficients, intercept = _mil(
        standardized,
        labels,
        weights,
        packed,
        coefficients,
        intercept,
        settings=settings,
        penalty=penalty,
    )
    return LogisticHead(mean, scale, coefficients, float(intercept))


def _region_scores(probabilities, owners, count, quantile):
    """Score of each region: a quantile of its rows' probabilities."""
    dtype = probabilities.dtype.type
    scores = np.full(count, np.nan, dtype=dtype)
    order = np.argsort(owners, kind="stable")
    grouped = owners[order]
    starts = np.r_[0, np.flatnonzero(grouped[1:] != grouped[:-1]) + 1]
    for start, stop in zip(starts, np.r_[starts[1:], owners.size]):
        scores[grouped[start]] = float(
            np.quantile(probabilities[order[start:stop]], dtype(quantile))
        )
    return scores


def _regional_view(scores, regions, head, runs):
    """Scores, targets and weights of the runs' regions with a regional target.

    Each stratum of one run and one target has a total weight of one.

    """
    dtype = scores.dtype.type
    values, targets, weights, strata = [], [], [], []
    for index, region in enumerate(regions):
        target = _target(region, head, regional=True)
        value = float(scores[index])
        if (
            region.run not in runs
            or target is None
            or region.weight <= 0.0
            or not np.isfinite(value)
        ):
            continue
        values.append(value)
        targets.append(target)
        weights.append(region.weight)
        strata.append((region.run, target))
    if not values:
        raise ValueError(f"No regional labels of {head} in the runs.")
    weights = np.asarray(weights, dtype=dtype)
    for stratum in sorted(set(strata)):
        mask = np.array([value == stratum for value in strata])
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            weights[mask] /= total
    return np.asarray(values, dtype=dtype), np.asarray(targets, dtype=np.int8), weights


def _balanced_accuracy(scores, targets, weights, threshold):
    """Weighted balanced accuracy of the regions scoring at least the threshold."""
    predictions = (scores >= threshold).astype(np.int8)
    recalls = []
    for value in (0, 1):
        mask = targets == value
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            hits = predictions[mask] == value
            recalls.append(float(np.sum(weights[mask] * hits) / total))
    return float(np.mean(recalls, dtype=weights.dtype)) if recalls else float("nan")


def _threshold(scores, targets, weights, levels):
    """Threshold of the best balanced accuracy, and that accuracy.

    Candidates are the fixed levels (first, last, count) and the same quantiles of
    the scores; of equally accurate thresholds the one nearest to 0.5 wins.

    """
    dtype = scores.dtype.type
    first, last, count = levels
    levels = np.linspace(dtype(first), dtype(last), count, dtype=dtype)
    best = (float("-inf"), 0.5)
    for threshold in np.unique(np.concatenate((levels, np.quantile(scores, levels)))):
        accuracy = _balanced_accuracy(scores, targets, weights, threshold)
        if (accuracy, -abs(float(threshold) - 0.5)) > (best[0], -abs(best[1] - 0.5)):
            best = (accuracy, float(threshold))
    return best[1], best[0]


def _roc_area(scores, targets, weights):
    """Weighted area under the ROC curve of the regions; NaN without both classes."""
    dtype = weights.dtype.type
    if set(np.unique(targets).tolist()) != {0, 1}:
        return float("nan")
    keep = weights > 0
    order = np.argsort(-scores[keep], kind="stable")
    target, weight, score = (
        targets[keep][order],
        weights[keep][order],
        scores[keep][order],
    )
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    true = np.cumsum(weight * target)[ends]
    false = np.cumsum(weight * (1 - target))[ends]
    if true[-1] == 0 or not false[-1]:
        return float("nan")
    recall = true / true[-1]
    previous = np.r_[dtype(0), recall[:-1]]
    rates = np.diff(np.r_[dtype(0), false / false[-1]])
    return float(np.sum(rates * ((previous + recall) / 2)))


def _cross_validate(samples, head, settings):
    """Leave-one-run-out cross-validation of one head over the regularization grid.

    Returns the head fitted on all runs at the selected regularization, its
    threshold and a summary.

    Raises
    ------
    ValueError
        If fewer than two runs have regional targets of the head.

    """
    regions = samples.regions
    dtype = samples.features.dtype.type
    runs = sorted(
        {r.run for r in regions if _target(r, head, regional=True) is not None}
    )
    if len(runs) < 2:
        raise ValueError(f"The cross-validation of {head} needs two runs.")
    grid, held_out = [], {}
    for penalty in settings.regularization_grid:
        thresholds, areas = [], []
        out_of_fold = np.full(len(regions), np.nan, dtype=dtype)
        for run in runs:
            training = set(runs) - {run}
            fit = _fit_head(samples, head, training, penalty, settings)
            scores = _region_scores(
                fit.probability(samples.features),
                samples.owners,
                len(regions),
                settings.region_score_quantile,
            )
            thresholds.append(
                _threshold(
                    *_regional_view(scores, regions, head, training),
                    settings.threshold_levels,
                )[0]
            )
            areas.append(_roc_area(*_regional_view(scores, regions, head, {run})))
            own = np.array([region.run == run for region in regions])
            out_of_fold[own] = scores[own]
        held_out[penalty] = out_of_fold
        threshold = _threshold(
            *_regional_view(out_of_fold, regions, head, set(runs)),
            settings.threshold_levels,
        )[0]
        accuracies = [
            _balanced_accuracy(
                *_regional_view(out_of_fold, regions, head, {run}), float(threshold)
            )
            for run in runs
        ]
        grid.append(
            {
                "regularization": float(penalty),
                "worst_run_balanced_accuracy": float(np.min(accuracies)),
                "mean_balanced_accuracy": float(np.mean(accuracies, dtype=dtype)),
                "fold_threshold_range": float(np.max(thresholds) - np.min(thresholds)),
                "mean_roc_area": float(np.nanmean(areas, dtype=dtype)),
            }
        )
    selected = max(
        grid,
        key=lambda row: (
            row["worst_run_balanced_accuracy"],
            row["mean_balanced_accuracy"],
            -row["fold_threshold_range"],
            row["mean_roc_area"],
            -abs(np.log10(row["regularization"]) + 3.0),
        ),
    )["regularization"]
    threshold, accuracy = _threshold(
        *_regional_view(held_out[selected], regions, head, set(runs)),
        settings.threshold_levels,
    )
    head_fit = _fit_head(samples, head, set(runs), selected, settings)
    summary = {
        "runs": runs,
        "selected_regularization": selected,
        "balanced_accuracy": accuracy,
        "grid": grid,
    }
    return head_fit, threshold, summary


def train_phase_model(
    samples: TrainingSamples, settings: PhaseTrainingSettings
) -> PhaseTraining:
    """Train the two heads of a phase model on training samples.

    Each head is cross-validated by leaving out one run at a time (see the
    module description) and fitted on all runs with a regional target of the
    head at the selected regularization; the threshold of its held-out region
    scores is its decision threshold. The model takes the samples' texture
    scale and the gas head's threshold.

    Parameters
    ----------
    samples : TrainingSamples
        Joined samples of the labelled frames of at least two runs.
    settings : PhaseTrainingSettings
        Settings of the fit and the cross-validation.

    Returns
    -------
    PhaseTraining
        The model, the CO2 head's threshold and the cross-validation.

    """
    fits = {head: _cross_validate(samples, head, settings) for head in _HEADS}
    return PhaseTraining(
        model=PhaseModel(
            co2_presence=fits["co2_presence"][0],
            gas_within_co2=fits["gas_within_co2"][0],
            texture_sigma_mm=samples.texture_sigma_mm,
            gas_threshold=fits["gas_within_co2"][1],
        ),
        co2_threshold=fits["co2_presence"][1],
        cross_validation={head: fit[2] for head, fit in fits.items()},
    )
