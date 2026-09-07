"""Shared training primitives for cross-series optical phase models.

The module keeps pixel supervision and regional phase-presence supervision
separate.  Pure reviewed regions train the gas and aqueous heads.  Mixed
regions constrain thresholds and validate phase presence without pretending
that every pixel in a mixed region belongs to both phases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PHASE_WATER = "water"
PHASE_AQUEOUS = "aqueous"
PHASE_GAS = "gas"
PHASE_MIXED_GAS_AQUEOUS = "mixed_gas_aqueous"
PHASE_MIXED_AQUEOUS_WATER = "mixed_aqueous_water"
PHASE_MIXED_ALL = "mixed_gas_aqueous_water"
PHASE_CO2_PRESENT_REGION = "co2_present_region"
PHASE_CO2_ABSENT_REGION = "co2_absent_region"
PHASE_GAS_PRESENT_REGION = "gas_present_region"
PHASE_AQUEOUS_PRESENT_REGION = "aqueous_present_region"
PHASE_GAS_ABSENT_REGION = "gas_absent_region"
PHASE_AQUEOUS_ABSENT_REGION = "aqueous_absent_region"
PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION = "aqueous_present_gas_absent_region"
PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT = "current_experiment_co2_artifact"
PHASE_IGNORED = "ignored"

PURE_PHASES = frozenset((PHASE_WATER, PHASE_AQUEOUS, PHASE_GAS))
MIXED_PHASES = frozenset(
    (PHASE_MIXED_GAS_AQUEOUS, PHASE_MIXED_AQUEOUS_WATER, PHASE_MIXED_ALL)
)
REGIONAL_EVIDENCE_PHASES = frozenset(
    (
        PHASE_CO2_PRESENT_REGION,
        PHASE_CO2_ABSENT_REGION,
        PHASE_GAS_PRESENT_REGION,
        PHASE_AQUEOUS_PRESENT_REGION,
        PHASE_GAS_ABSENT_REGION,
        PHASE_AQUEOUS_ABSENT_REGION,
        PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION,
        PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT,
    )
)

MANUAL_PHASE_CLASSES: Mapping[str, str] = {
    "confirmed_no_co2": PHASE_WATER,
    "confirmed_only_aqueous_co2": PHASE_AQUEOUS,
    "confirmed_predominantly_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_aqueous_co2": PHASE_AQUEOUS,
    "confirmed_predominantly_free_gas": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_only_free_gas": PHASE_GAS,
    "confirmed_free_gas": PHASE_GAS,
    "confirmed_mixed_free_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "confirmed_mixed_aqueous_water": PHASE_MIXED_AQUEOUS_WATER,
    "confirmed_mixed_free_aqueous_water": PHASE_MIXED_ALL,
    # Candidate attribution records which model exposed the problem. These
    # classes capture candidate-independent evidence in the source image and
    # therefore enter training as conservative regional constraints.
    "missed_visible_plume": PHASE_CO2_PRESENT_REGION,
    "missed_visible_co2": PHASE_CO2_PRESENT_REGION,
    "missed_visible_gas": PHASE_GAS_PRESENT_REGION,
    "missed_visible_aqueous_co2": PHASE_AQUEOUS_PRESENT_REGION,
    "isolated_noise": PHASE_CO2_ABSENT_REGION,
    "disconnected_noise": PHASE_CO2_ABSENT_REGION,
    "unsupported_free_gas_classification": PHASE_GAS_ABSENT_REGION,
    "unsupported_aqueous_co2_classification": PHASE_AQUEOUS_ABSENT_REGION,
    "likely_dense_aqueous_co2_misclassified_as_gas": (
        PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION
    ),
    "incomplete_wash_indicator_carryover": PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT,
    "phase_not_assessable": PHASE_IGNORED,
    "phase_signal_artifact": PHASE_IGNORED,
}

QUIZ_PHASE_CLASSES: Mapping[str, str] = {
    "water_or_no_visible_co2": PHASE_WATER,
    "only_aqueous_co2": PHASE_AQUEOUS,
    "predominantly_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "predominantly_free_gas": PHASE_MIXED_GAS_AQUEOUS,
    "free_gas_only": PHASE_GAS,
    "mixed_free_and_aqueous_co2": PHASE_MIXED_GAS_AQUEOUS,
    "mixed_aqueous_co2_and_water": PHASE_MIXED_AQUEOUS_WATER,
    "mixed_free_aqueous_co2_and_water": PHASE_MIXED_ALL,
    "co2_present_phase_uncertain": PHASE_CO2_PRESENT_REGION,
    "not_assessable": PHASE_IGNORED,
    "phase_signal_artifact": PHASE_IGNORED,
    "ignore_region": PHASE_IGNORED,
}

CONFIDENCE_WEIGHTS: Mapping[str, float] = {
    "high": 1.0,
    "medium": 0.55,
    "low": 0.0,
}

SOURCE_WEIGHTS: Mapping[str, float] = {
    "manual": 1.0,
    "quiz": 0.8,
}


@dataclass(frozen=True)
class ReviewedRegion:
    """One reviewed visible-2D region and its traceable provenance."""

    region_id: str
    run: str
    dataset: str
    frame_index: int
    elapsed_h: float
    phase_class: str
    confidence: str
    source: str
    reviewer: str
    payload: Mapping[str, Any]

    @property
    def supervision_weight(self) -> float:
        return float(CONFIDENCE_WEIGHTS.get(self.confidence, 0.0)) * float(
            SOURCE_WEIGHTS.get(self.source, 0.0)
        )


@dataclass(frozen=True)
class LogisticFit:
    """Serializable result from deterministic weighted logistic IRLS."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    backend: str
    precision: str
    iterations: int
    converged: bool
    objective: float
    regularization: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float32)
        standardized = (matrix - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32
        )
        logits = standardized @ np.asarray(self.coefficients, dtype=np.float32)
        logits += np.float32(self.intercept)
        logits = np.clip(logits, -30.0, 30.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float32)


@dataclass(frozen=True)
class _PackedCudaBags:
    """Padded weak-region tensors for fused CUDA MIL evaluation."""

    features: Any
    valid: Any
    counts: Any
    normalized_weights: Any
    selected_counts: Any | None
    target: int
    region_indices: tuple[int, ...]
    total_rows: int


def canonical_phase_class(item: Mapping[str, Any], source: str) -> str | None:
    """Map one UI record to the shared visible-phase vocabulary."""

    if source == "manual":
        category = str(item.get("category") or "").strip().lower()
        mapped = MANUAL_PHASE_CLASSES.get(category)
        if mapped is not None:
            return mapped
        declared = str(item.get("training_phase_class") or "").strip().lower()
        if declared in PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES:
            return declared
        return None
    if source == "quiz":
        choice = str(item.get("human_choice") or "").strip().lower()
        schema = str(item.get("phase_label_schema") or "").strip().lower()
        # V1/V2 combined "mostly" and "entirely" labels. Preserve their
        # stored canonical class instead of silently applying V3 semantics.
        if schema and schema != "fluidflower-visible-phase-v3":
            declared = str(item.get("phase_class") or "").strip().lower()
            if declared in (
                PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES | {PHASE_IGNORED}
            ):
                return declared
        mapped = QUIZ_PHASE_CLASSES.get(choice)
        if mapped is not None:
            return mapped
        declared = str(item.get("phase_class") or "").strip().lower()
        if declared in (
            PURE_PHASES | MIXED_PHASES | REGIONAL_EVIDENCE_PHASES | {PHASE_IGNORED}
        ):
            return declared
        return None
    raise ValueError(f"Unsupported review source {source!r}")


def pixel_target(phase_class: str, head: str) -> int | None:
    """Return a hard pixel target only where the review supports one."""

    if head == "gas":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "aqueous":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            # A gas-labelled region may also contain aqueous CO2 around or
            # behind the visible gas in the camera-facing slice.
            PHASE_GAS: None,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "co2_presence":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "gas_within_co2":
        return {
            PHASE_WATER: None,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
        }.get(phase_class)
    raise ValueError(f"Unsupported phase head {head!r}")


def regional_presence_target(phase_class: str, head: str) -> int | None:
    """Return whether a reviewed region contains evidence for one phase."""

    if head == "gas":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 0,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: 0,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: None,
            PHASE_GAS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 0,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "aqueous":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: None,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 1,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: None,
            PHASE_GAS_PRESENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_REGION: 1,
            PHASE_GAS_ABSENT_REGION: None,
            PHASE_AQUEOUS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "co2_presence":
        return {
            PHASE_WATER: 0,
            PHASE_AQUEOUS: 1,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 1,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: 1,
            PHASE_CO2_ABSENT_REGION: 0,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: 1,
            PHASE_GAS_ABSENT_REGION: None,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 1,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: 0,
        }.get(phase_class)
    if head == "gas_within_co2":
        return {
            PHASE_WATER: None,
            PHASE_AQUEOUS: 0,
            PHASE_GAS: 1,
            PHASE_MIXED_GAS_AQUEOUS: 1,
            PHASE_MIXED_AQUEOUS_WATER: 0,
            PHASE_MIXED_ALL: 1,
            PHASE_CO2_PRESENT_REGION: None,
            PHASE_CO2_ABSENT_REGION: None,
            PHASE_GAS_PRESENT_REGION: 1,
            PHASE_AQUEOUS_PRESENT_REGION: 0,
            # The reviewer only asserts that the region contains no free gas;
            # MIL therefore uses it as a negative bag without assigning a hard
            # per-pixel aqueous/water class.
            PHASE_GAS_ABSENT_REGION: 0,
            PHASE_AQUEOUS_ABSENT_REGION: None,
            PHASE_AQUEOUS_PRESENT_GAS_ABSENT_REGION: 0,
            PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT: None,
        }.get(phase_class)
    raise ValueError(f"Unsupported phase head {head!r}")


def balanced_pixel_view(
    features: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a run/class-balanced pure-region pixel training view.

    Every reviewed region first receives a total confidence/source weight
    independent of its pixel area.  Each ``(run, target)`` stratum is then
    normalized to equal total weight, preventing a heavily reviewed water
    class in one run from dominating the cross-series model.
    """

    matrix = np.asarray(features, dtype=np.float32)
    owners = np.asarray(region_indices, dtype=np.int32)
    if owners.shape != (matrix.shape[0],):
        raise ValueError("Region indices do not match feature rows")

    selected_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    weights_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    strata: list[tuple[str, int]] = []
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        target = pixel_target(region.phase_class, head)
        total_weight = region.supervision_weight
        if target is None or total_weight <= 0.0:
            continue
        rows = np.flatnonzero(owners == index)
        if rows.size == 0:
            continue
        selected_chunks.append(rows)
        labels_chunks.append(np.full(rows.size, target, dtype=np.int8))
        weights_chunks.append(
            np.full(rows.size, total_weight / rows.size, dtype=np.float64)
        )
        owner_chunks.append(np.full(rows.size, index, dtype=np.int32))
        strata.extend([(region.run, int(target))] * rows.size)

    if not selected_chunks:
        raise ValueError(f"No pixel supervision is available for {head}")
    selected = np.concatenate(selected_chunks)
    labels = np.concatenate(labels_chunks)
    weights = np.concatenate(weights_chunks)
    selected_owners = np.concatenate(owner_chunks)
    unique_strata = sorted(set(strata))
    for stratum in unique_strata:
        mask = np.asarray([value == stratum for value in strata], dtype=bool)
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            weights[mask] /= total
    weights *= float(weights.size) / max(float(np.sum(weights)), 1e-12)
    return matrix[selected], labels, weights, selected_owners


def weighted_standardization(
    features: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one shared float64 reference normalization."""

    matrix = np.asarray(features, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Training weights must have positive finite mass")
    mean = np.sum(matrix * weights[:, None], axis=0) / total
    centered = matrix - mean
    variance = np.sum(centered * centered * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale = np.maximum(scale, 1e-6)
    return mean, scale


def _objective_numpy(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    regularization: float,
) -> float:
    logits = matrix @ coefficients + intercept
    loss = np.logaddexp(0.0, logits) - labels * logits
    return float(
        np.sum(weights * loss) / np.sum(weights)
        + 0.5 * regularization * np.dot(coefficients, coefficients)
    )


def _fit_cpu64(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, bool, float]:
    matrix = np.asarray(standardized, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weight))
    dimension = matrix.shape[1]
    coefficients = np.zeros(dimension, dtype=np.float64)
    positive = float(np.sum(weight * target) / weight_sum)
    intercept = float(
        np.log(np.clip(positive, 1e-6, 1 - 1e-6) / np.clip(1 - positive, 1e-6, 1))
    )
    identity = np.eye(dimension, dtype=np.float64)
    converged = False
    objective = _objective_numpy(
        matrix, target, weight, coefficients, intercept, regularization
    )

    for iteration in range(1, max_iter + 1):
        logits = np.clip(matrix @ coefficients + intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        residual = weight * (probability - target) / weight_sum
        curvature = (
            weight * np.maximum(probability * (1.0 - probability), 1e-7) / weight_sum
        )
        gradient = np.concatenate(
            (matrix.T @ residual + regularization * coefficients, [np.sum(residual)])
        )
        weighted_matrix = matrix * curvature[:, None]
        hessian = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        hessian[:dimension, :dimension] = (
            matrix.T @ weighted_matrix + regularization * identity
        )
        cross = matrix.T @ curvature
        hessian[:dimension, dimension] = cross
        hessian[dimension, :dimension] = cross
        hessian[dimension, dimension] = float(np.sum(curvature))
        hessian.flat[:: dimension + 2] += 1e-8
        step = np.linalg.solve(hessian, gradient)

        factor = 1.0
        accepted = False
        for _ in range(12):
            candidate_coefficients = coefficients - factor * step[:dimension]
            candidate_intercept = intercept - factor * float(step[dimension])
            candidate_objective = _objective_numpy(
                matrix,
                target,
                weight,
                candidate_coefficients,
                candidate_intercept,
                regularization,
            )
            if candidate_objective <= objective + 1e-12:
                coefficients = candidate_coefficients
                intercept = candidate_intercept
                objective = candidate_objective
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(factor * step))) < tolerance:
            converged = True
            break
    return coefficients, intercept, iteration, converged, objective


def _fit_cuda32(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, float, int, bool, float]:
    import cupy as cp

    matrix = cp.asarray(standardized, dtype=cp.float32)
    target = cp.asarray(labels, dtype=cp.float32)
    weight = cp.asarray(weights, dtype=cp.float32)
    weight_sum = cp.sum(weight, dtype=cp.float32)
    dimension = matrix.shape[1]
    coefficients = cp.zeros(dimension, dtype=cp.float32)
    positive = cp.sum(weight * target, dtype=cp.float32) / weight_sum
    positive = cp.clip(positive, cp.float32(1e-6), cp.float32(1 - 1e-6))
    intercept = cp.log(positive / (cp.float32(1.0) - positive))
    identity = cp.eye(dimension, dtype=cp.float32)

    def objective_value(beta: Any, bias: Any) -> Any:
        logits = matrix @ beta + bias
        loss = cp.logaddexp(cp.float32(0.0), logits) - target * logits
        return cp.sum(weight * loss, dtype=cp.float32) / weight_sum + cp.float32(
            0.5 * regularization
        ) * cp.dot(beta, beta)

    objective = objective_value(coefficients, intercept)
    converged = False
    for iteration in range(1, max_iter + 1):
        logits = cp.clip(matrix @ coefficients + intercept, -30.0, 30.0)
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        residual = weight * (probability - target) / weight_sum
        curvature = (
            weight
            * cp.maximum(
                probability * (cp.float32(1.0) - probability),
                cp.float32(1e-7),
            )
            / weight_sum
        )
        gradient = cp.concatenate(
            (
                matrix.T @ residual + cp.float32(regularization) * coefficients,
                cp.sum(residual, dtype=cp.float32).reshape(1),
            )
        )
        weighted_matrix = matrix * curvature[:, None]
        hessian = cp.empty((dimension + 1, dimension + 1), dtype=cp.float32)
        hessian[:dimension, :dimension] = (
            matrix.T @ weighted_matrix + cp.float32(regularization) * identity
        )
        cross = matrix.T @ curvature
        hessian[:dimension, dimension] = cross
        hessian[dimension, :dimension] = cross
        hessian[dimension, dimension] = cp.sum(curvature, dtype=cp.float32)
        hessian += cp.float32(1e-6) * cp.eye(dimension + 1, dtype=cp.float32)
        step = cp.linalg.solve(hessian, gradient)

        factor = cp.float32(1.0)
        accepted = False
        for _ in range(12):
            candidate_coefficients = coefficients - factor * step[:dimension]
            candidate_intercept = intercept - factor * step[dimension]
            candidate_objective = objective_value(
                candidate_coefficients, candidate_intercept
            )
            if bool((candidate_objective <= objective + cp.float32(1e-7)).item()):
                coefficients = candidate_coefficients
                intercept = candidate_intercept
                objective = candidate_objective
                accepted = True
                break
            factor *= cp.float32(0.5)
        if not accepted:
            break
        if float(cp.max(cp.abs(factor * step)).item()) < tolerance:
            converged = True
            break

    result = cp.asnumpy(coefficients).astype(np.float64)
    result_intercept = float(intercept.item())
    result_objective = float(objective.item())
    del coefficients
    cp.get_default_memory_pool().free_all_blocks()
    return (
        result,
        result_intercept,
        iteration,
        converged,
        result_objective,
    )


def cuda_available() -> bool:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - optional runtime and driver failures vary
        return False


def fit_weighted_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float,
    backend: str = "auto",
    max_iter: int = 40,
    tolerance: float = 1e-5,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> LogisticFit:
    """Fit one deterministic weighted logistic model on CPU or CUDA."""

    matrix = np.asarray(features, dtype=np.float32)
    target = np.asarray(labels, dtype=np.int8)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("Training matrix and labels differ")
    if weights.shape != (target.size,):
        raise ValueError("Training weights and labels differ")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("Logistic training requires both target classes")
    finite = np.all(np.isfinite(matrix), axis=1) & np.isfinite(weights) & (weights > 0)
    matrix = matrix[finite]
    target = target[finite]
    weights = weights[finite]
    mean, scale = (
        weighted_standardization(matrix, weights)
        if normalization is None
        else (
            np.asarray(normalization[0], dtype=np.float64),
            np.asarray(normalization[1], dtype=np.float64),
        )
    )

    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested == "cuda":
        standardized = (matrix - mean.astype(np.float32)) / scale.astype(np.float32)
        coefficients, intercept, iterations, converged, objective = _fit_cuda32(
            standardized,
            target,
            weights,
            regularization=float(regularization),
            max_iter=int(max_iter),
            tolerance=float(tolerance),
        )
        precision = "float32"
    elif requested == "cpu":
        standardized = (matrix.astype(np.float64) - mean) / scale
        coefficients, intercept, iterations, converged, objective = _fit_cpu64(
            standardized,
            target,
            weights,
            regularization=float(regularization),
            max_iter=int(max_iter),
            tolerance=float(tolerance),
        )
        precision = "float64"
    else:
        raise ValueError(f"Unsupported logistic backend {backend!r}")

    return LogisticFit(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        intercept=float(intercept),
        backend=requested,
        precision=precision,
        iterations=int(iterations),
        converged=bool(converged),
        objective=float(objective),
        regularization=float(regularization),
    )


def _weak_region_bags(
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None,
) -> list[tuple[np.ndarray, int, float, int]]:
    """Return run/class-balanced weak bags not used as hard pixel labels."""

    owners = np.asarray(region_indices, dtype=np.int32)
    pending: list[tuple[np.ndarray, int, float, int, tuple[str, int]]] = []
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        if pixel_target(region.phase_class, head) is not None:
            continue
        target = regional_presence_target(region.phase_class, head)
        weight = region.supervision_weight
        if target is None or weight <= 0.0:
            continue
        rows = np.flatnonzero(owners == index)
        if rows.size == 0:
            continue
        pending.append(
            (rows, int(target), float(weight), index, (region.run, int(target)))
        )

    totals: dict[tuple[str, int], float] = {}
    for _rows, _target, weight, _index, stratum in pending:
        totals[stratum] = totals.get(stratum, 0.0) + weight
    bags: list[tuple[np.ndarray, int, float, int]] = []
    for rows, target, weight, index, stratum in pending:
        total = totals[stratum]
        if total > 0.0:
            bags.append((rows, target, weight / total, index))
    return bags


def _pack_weak_bags_cuda(
    matrix: np.ndarray,
    bags: Sequence[tuple[np.ndarray, int, float, int]],
    *,
    target: int,
    mean: np.ndarray,
    scale: np.ndarray,
    weight_sum: float,
    positive_top_fraction: float,
) -> _PackedCudaBags | None:
    """Pack equal-target bags once instead of transferring every bag per step."""

    selected = [bag for bag in bags if int(bag[1]) == int(target)]
    if not selected:
        return None

    import cupy as cp

    dimension = int(matrix.shape[1])
    maximum_rows = max(int(rows.size) for rows, *_rest in selected)
    host_features = np.zeros(
        (len(selected), maximum_rows, dimension),
        dtype=np.float32,
    )
    host_valid = np.zeros((len(selected), maximum_rows), dtype=bool)
    host_counts = np.empty(len(selected), dtype=np.int32)
    host_weights = np.empty(len(selected), dtype=np.float32)
    host_selected_counts = (
        np.empty(len(selected), dtype=np.int32) if int(target) == 1 else None
    )
    region_ids: list[int] = []
    total_rows = 0
    mean32 = np.asarray(mean, dtype=np.float32)
    scale32 = np.asarray(scale, dtype=np.float32)
    for bag_index, (rows, _target, bag_weight, region_index) in enumerate(selected):
        count = int(rows.size)
        host_features[bag_index, :count] = (
            np.asarray(matrix[rows], dtype=np.float32) - mean32
        ) / scale32
        host_valid[bag_index, :count] = True
        host_counts[bag_index] = count
        host_weights[bag_index] = np.float32(float(bag_weight) / float(weight_sum))
        if host_selected_counts is not None:
            host_selected_counts[bag_index] = max(
                1,
                round(float(positive_top_fraction) * count),
            )
        region_ids.append(int(region_index))
        total_rows += count

    return _PackedCudaBags(
        features=cp.asarray(host_features, dtype=cp.float32),
        valid=cp.asarray(host_valid),
        counts=cp.asarray(host_counts, dtype=cp.int32),
        normalized_weights=cp.asarray(host_weights, dtype=cp.float32),
        selected_counts=(
            None
            if host_selected_counts is None
            else cp.asarray(host_selected_counts, dtype=cp.int32)
        ),
        target=int(target),
        region_indices=tuple(region_ids),
        total_rows=int(total_rows),
    )


def _packed_weak_terms_cuda(
    packed: _PackedCudaBags,
    coefficients: Any,
    intercept: Any,
) -> tuple[Any, Any, Any]:
    """Evaluate one target group with a bounded number of CUDA operations."""

    import cupy as cp

    bag_count, maximum_rows, dimension = packed.features.shape
    flat_features = packed.features.reshape(bag_count * maximum_rows, dimension)
    logits = (flat_features @ coefficients).reshape(bag_count, maximum_rows)
    logits = cp.clip(logits + intercept, -30.0, 30.0)

    if packed.target == 0:
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        row_weights = packed.normalized_weights[:, None] / packed.counts[:, None]
        residual = cp.where(
            packed.valid,
            probability * row_weights,
            cp.float32(0.0),
        )
        safe_logits = cp.where(packed.valid, logits, cp.float32(0.0))
        per_bag_loss = (
            cp.sum(
                cp.where(
                    packed.valid,
                    cp.logaddexp(cp.float32(0.0), safe_logits),
                    cp.float32(0.0),
                ),
                axis=1,
                dtype=cp.float32,
            )
            / packed.counts
        )
        loss = cp.sum(
            packed.normalized_weights * per_bag_loss,
            dtype=cp.float32,
        )
    else:
        if packed.selected_counts is None:
            raise RuntimeError("Positive MIL bags require selected row counts")
        masked_logits = cp.where(packed.valid, logits, -cp.inf)
        order = cp.argsort(masked_logits, axis=1)
        ordered_logits = cp.take_along_axis(masked_logits, order, axis=1)
        positions = cp.arange(maximum_rows, dtype=cp.int32)[None, :]
        selected_mask = positions >= (
            cp.int32(maximum_rows) - packed.selected_counts[:, None]
        )
        safe_logits = cp.where(selected_mask, ordered_logits, cp.float32(0.0))
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-safe_logits))
        ordered_residual = cp.where(
            selected_mask,
            (probability - cp.float32(1.0))
            * packed.normalized_weights[:, None]
            / packed.selected_counts[:, None],
            cp.float32(0.0),
        )
        residual = cp.empty_like(ordered_residual)
        cp.put_along_axis(residual, order, ordered_residual, axis=1)
        per_bag_loss = (
            cp.sum(
                cp.where(
                    selected_mask,
                    cp.logaddexp(cp.float32(0.0), -safe_logits),
                    cp.float32(0.0),
                ),
                axis=1,
                dtype=cp.float32,
            )
            / packed.selected_counts
        )
        loss = cp.sum(
            packed.normalized_weights * per_bag_loss,
            dtype=cp.float32,
        )

    flat_residual = residual.reshape(bag_count * maximum_rows)
    gradient = flat_features.T @ flat_residual
    bias_gradient = cp.sum(flat_residual, dtype=cp.float32)
    return gradient, bias_gradient, loss


def fit_weighted_logistic_mil(
    features: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None,
    regularization: float,
    backend: str = "auto",
    weak_strength: float = 0.35,
    positive_top_fraction: float = 0.10,
    max_iter: int = 160,
    learning_rate: float = 0.035,
    tolerance: float = 2e-5,
) -> tuple[LogisticFit, dict[str, Any]]:
    """Fine-tune a pure-region logistic model with mixed-region bag losses.

    Positive mixed bags require a high-scoring subset, while negative bags
    constrain every sampled pixel. This lets pure regions anchor the optical
    classes without assigning one contradictory hard label to every pixel in
    a mixed region.
    """

    if not 0.0 < float(positive_top_fraction) <= 1.0:
        raise ValueError("positive_top_fraction must be in (0, 1]")
    if float(weak_strength) < 0.0:
        raise ValueError("weak_strength must be non-negative")
    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested != "cuda":
        raise RuntimeError(
            "MIL fitting currently requires CUDA/CuPy; CPU float64 remains "
            "the inference/parity reference"
        )

    matrix = np.asarray(features, dtype=np.float32)
    pure_matrix, pure_labels, pure_weights, _ = balanced_pixel_view(
        matrix,
        region_indices,
        regions,
        head=head,
        runs=runs,
    )
    initial = fit_weighted_logistic(
        pure_matrix,
        pure_labels,
        pure_weights,
        regularization=float(regularization),
        backend="cuda",
    )
    bags = _weak_region_bags(
        region_indices,
        regions,
        head=head,
        runs=runs,
    )
    if not bags or float(weak_strength) == 0.0:
        return initial, {
            "mode": "pure-region-irls",
            "weak_bags": len(bags),
            "iterations": 0,
            "converged": initial.converged,
        }

    import cupy as cp

    mean32 = np.asarray(initial.mean, dtype=np.float32)
    scale32 = np.asarray(initial.scale, dtype=np.float32)
    pure_standardized = (pure_matrix - mean32) / scale32
    x_pure = cp.asarray(pure_standardized, dtype=cp.float32)
    y_pure = cp.asarray(pure_labels, dtype=cp.float32)
    w_pure = cp.asarray(pure_weights, dtype=cp.float32)
    w_pure /= cp.sum(w_pure, dtype=cp.float32)
    coefficients = cp.asarray(initial.coefficients, dtype=cp.float32)
    intercept = cp.float32(initial.intercept)
    first_moment = cp.zeros_like(coefficients)
    second_moment = cp.zeros_like(coefficients)
    bias_first = cp.float32(0.0)
    bias_second = cp.float32(0.0)
    beta1 = cp.float32(0.9)
    beta2 = cp.float32(0.999)
    epsilon = cp.float32(1e-7)
    weak_weight_sum = max(sum(value[2] for value in bags), 1e-12)
    positive_payload = _pack_weak_bags_cuda(
        matrix,
        bags,
        target=1,
        mean=mean32,
        scale=scale32,
        weight_sum=weak_weight_sum,
        positive_top_fraction=float(positive_top_fraction),
    )
    negative_payload = _pack_weak_bags_cuda(
        matrix,
        bags,
        target=0,
        mean=mean32,
        scale=scale32,
        weight_sum=weak_weight_sum,
        positive_top_fraction=float(positive_top_fraction),
    )
    converged = False
    objective = float("nan")
    stable_steps = 0

    for iteration in range(1, int(max_iter) + 1):
        pure_logits = cp.clip(x_pure @ coefficients + intercept, -30.0, 30.0)
        pure_probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-pure_logits))
        pure_residual = w_pure * (pure_probability - y_pure)
        gradient = x_pure.T @ pure_residual
        bias_gradient = cp.sum(pure_residual, dtype=cp.float32)
        pure_loss = cp.sum(
            w_pure
            * (cp.logaddexp(cp.float32(0.0), pure_logits) - y_pure * pure_logits),
            dtype=cp.float32,
        )

        weak_loss = cp.float32(0.0)
        weak_gradient = cp.zeros_like(coefficients)
        weak_bias_gradient = cp.float32(0.0)
        for packed in (positive_payload, negative_payload):
            if packed is None:
                continue
            packed_gradient, packed_bias_gradient, packed_loss = (
                _packed_weak_terms_cuda(packed, coefficients, intercept)
            )
            weak_gradient += packed_gradient
            weak_bias_gradient += packed_bias_gradient
            weak_loss += packed_loss

        gradient += cp.float32(weak_strength) * weak_gradient
        bias_gradient += cp.float32(weak_strength) * weak_bias_gradient
        gradient += cp.float32(regularization) * coefficients
        gradient_norm = cp.linalg.norm(gradient)
        if float(gradient_norm.item()) > 10.0:
            gradient *= cp.float32(10.0) / gradient_norm
            bias_gradient = cp.clip(bias_gradient, -10.0, 10.0)

        first_moment = beta1 * first_moment + (cp.float32(1.0) - beta1) * gradient
        second_moment = beta2 * second_moment + (cp.float32(1.0) - beta2) * cp.square(
            gradient
        )
        bias_first = beta1 * bias_first + (cp.float32(1.0) - beta1) * bias_gradient
        bias_second = beta2 * bias_second + (cp.float32(1.0) - beta2) * cp.square(
            bias_gradient
        )
        correction1 = cp.float32(1.0) - beta1**iteration
        correction2 = cp.float32(1.0) - beta2**iteration
        rate = cp.float32(learning_rate / np.sqrt(1.0 + iteration / 80.0))
        coefficient_step = (
            rate
            * (first_moment / correction1)
            / (cp.sqrt(second_moment / correction2) + epsilon)
        )
        intercept_step = (
            rate
            * (bias_first / correction1)
            / (cp.sqrt(bias_second / correction2) + epsilon)
        )
        coefficients -= coefficient_step
        intercept -= intercept_step

        maximum_step = max(
            float(cp.max(cp.abs(coefficient_step)).item()),
            abs(float(intercept_step.item())),
        )
        stable_steps = stable_steps + 1 if maximum_step < float(tolerance) else 0
        objective = float(
            (
                pure_loss
                + cp.float32(weak_strength) * weak_loss
                + cp.float32(0.5 * regularization) * cp.dot(coefficients, coefficients)
            ).item()
        )
        if stable_steps >= 8:
            converged = True
            break

    result = LogisticFit(
        mean=np.asarray(initial.mean, dtype=np.float64),
        scale=np.asarray(initial.scale, dtype=np.float64),
        coefficients=cp.asnumpy(coefficients).astype(np.float64),
        intercept=float(intercept.item()),
        backend="cuda",
        precision="float32",
        iterations=int(iteration),
        converged=bool(converged),
        objective=float(objective),
        regularization=float(regularization),
    )
    diagnostics = {
        "mode": "pure-irls-plus-mixed-region-mil",
        "weak_bags": len(bags),
        "positive_weak_bags": sum(value[1] == 1 for value in bags),
        "negative_weak_bags": sum(value[1] == 0 for value in bags),
        "cuda_bag_evaluation": "padded-fused-v1",
        "positive_weak_rows": (
            0 if positive_payload is None else positive_payload.total_rows
        ),
        "negative_weak_rows": (
            0 if negative_payload is None else negative_payload.total_rows
        ),
        "weak_strength": float(weak_strength),
        "positive_top_fraction": float(positive_top_fraction),
        "iterations": int(iteration),
        "converged": bool(converged),
        "objective": float(objective),
    }
    del x_pure, y_pure, w_pure, coefficients, first_moment, second_moment
    del positive_payload, negative_payload
    cp.get_default_memory_pool().free_all_blocks()
    return result, diagnostics


def predict_logistic(
    fit: LogisticFit,
    features: np.ndarray,
    *,
    backend: str = "auto",
    chunk_rows: int = 250_000,
) -> np.ndarray:
    """Predict in bounded chunks, retaining GPU acceleration when available."""

    matrix = np.asarray(features, dtype=np.float32)
    requested = backend.strip().lower()
    if requested == "auto":
        requested = "cuda" if cuda_available() else "cpu"
    if requested == "cpu":
        return fit.predict(matrix)
    if requested != "cuda":
        raise ValueError(f"Unsupported prediction backend {backend!r}")

    import cupy as cp

    mean = cp.asarray(fit.mean, dtype=cp.float32)
    scale = cp.asarray(fit.scale, dtype=cp.float32)
    coefficients = cp.asarray(fit.coefficients, dtype=cp.float32)
    output = np.empty(matrix.shape[0], dtype=np.float32)
    step = max(1, int(chunk_rows))
    for start in range(0, matrix.shape[0], step):
        stop = min(matrix.shape[0], start + step)
        block = cp.asarray(matrix[start:stop], dtype=cp.float32)
        logits = ((block - mean) / scale) @ coefficients
        logits += cp.float32(fit.intercept)
        logits = cp.clip(logits, -30.0, 30.0)
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        output[start:stop] = cp.asnumpy(probability)
    del mean, scale, coefficients
    cp.get_default_memory_pool().free_all_blocks()
    return output


def region_probability_scores(
    probabilities: np.ndarray,
    region_indices: np.ndarray,
    region_count: int,
    *,
    quantile: float = 0.90,
) -> np.ndarray:
    """Aggregate pixel probabilities to robust phase-presence scores."""

    values = np.asarray(probabilities, dtype=np.float64)
    owners = np.asarray(region_indices, dtype=np.int32)
    scores = np.full(int(region_count), np.nan, dtype=np.float64)
    for index in np.unique(owners):
        selected = values[owners == index]
        if selected.size:
            scores[int(index)] = float(np.quantile(selected, quantile))
    return scores


def regional_evaluation_view(
    scores: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    runs: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an equal-run/class regional presence evaluation view."""

    score_rows: list[float] = []
    targets: list[int] = []
    weights: list[float] = []
    indices: list[int] = []
    strata: list[tuple[str, int]] = []
    for index, region in enumerate(regions):
        if runs is not None and region.run not in runs:
            continue
        target = regional_presence_target(region.phase_class, head)
        value = float(scores[index])
        weight = region.supervision_weight
        if target is None or weight <= 0.0 or not np.isfinite(value):
            continue
        score_rows.append(value)
        targets.append(int(target))
        weights.append(weight)
        indices.append(index)
        strata.append((region.run, int(target)))
    if not score_rows:
        raise ValueError(f"No regional evaluation labels for {head}")
    score_array = np.asarray(score_rows, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    for stratum in sorted(set(strata)):
        mask = np.fromiter(
            (value == stratum for value in strata), dtype=bool, count=len(strata)
        )
        total = float(np.sum(weight_array[mask]))
        if total > 0.0:
            weight_array[mask] /= total
    return score_array, target_array, weight_array, np.asarray(indices, dtype=np.int32)


def weighted_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    recalls: list[float] = []
    for value in (0, 1):
        mask = labels == value
        total = float(np.sum(weights[mask]))
        if total > 0.0:
            recalls.append(
                float(np.sum(weights[mask] * (predictions[mask] == value)) / total)
            )
    return float(np.mean(recalls)) if recalls else float("nan")


def tune_presence_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Tune a regional presence threshold by weighted balanced accuracy."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            (
                np.linspace(0.02, 0.98, 97),
                np.quantile(score_array, np.linspace(0.02, 0.98, 97)),
            )
        )
    )
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        accuracy = weighted_balanced_accuracy(
            label_array,
            (score_array >= threshold).astype(np.int8),
            weight_array,
        )
        candidate = (float(accuracy), -abs(float(threshold) - 0.5))
        incumbent = (best[0], -abs(best[1] - 0.5))
        if candidate > incumbent:
            best = (float(accuracy), float(threshold))
    return best[1], best[0]


def regional_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Return weighted, interpretable regional binary metrics."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    weight_array = np.asarray(weights, dtype=np.float64)
    predicted = (score_array >= float(threshold)).astype(np.int8)
    result = {
        "balanced_accuracy": weighted_balanced_accuracy(
            label_array, predicted, weight_array
        ),
        "threshold": float(threshold),
        "regions": int(label_array.size),
        "positive_fraction": float(
            np.sum(weight_array * label_array) / np.sum(weight_array)
        ),
    }
    if set(np.unique(label_array).tolist()) == {0, 1}:
        result["roc_auc"] = float(
            roc_auc_score(label_array, score_array, sample_weight=weight_array)
        )
        result["average_precision"] = float(
            average_precision_score(
                label_array, score_array, sample_weight=weight_array
            )
        )
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result
