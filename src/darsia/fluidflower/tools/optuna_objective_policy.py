"""Explicit search-objective policies for FF-AC calibration campaigns.

The calibrated physical result remains the value returned by
``evaluate_run``.  These policies only control what Optuna learns from each
trial.  In particular, ``staged-spatial`` preserves the first spatial search
contract, while ``aligned-spatial-v1`` uses the same mass/spatial score during
both Optuna search and final candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


DEFAULT_TRAINING_LIMITS = {
    "maximum_gas_envelope_excess_fraction": 0.005,
    "maximum_aqueous_decrease_fraction": 0.15,
    "minimum_ratio_4p1": 0.75,
    "maximum_ratio_4p1": 1.25,
    "minimum_late_mean_ratio": 0.85,
    "maximum_late_mean_ratio": 1.15,
    "minimum_terminal_ratio": 0.85,
    "maximum_terminal_ratio": 1.15,
    "maximum_worst_window_mae": 0.25,
}
INVALID_SCORE = 1.0e6
V2_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR = 0.50
V2_GAIN_PRIOR_WEIGHT = 0.01
V2_PHYSICAL_PENALTY_TERMS = (
    "post_injection_balance_penalty",
    "aqueous_diffusion_penalty",
    "late_gas_penalty",
    "overflow_phase_penalty",
    "overflow_post_coverage_penalty",
    "depth_prior_penalty",
)
V2_CONSTRAINT_NAMES = (
    "late_gas_envelope",
    "aqueous_decrease",
    "ratio_4p1",
    "late_mean_ratio",
    "terminal_ratio",
    "worst_window",
    "worst_relative_window",
)


@dataclass(frozen=True)
class SearchObjectiveScore:
    value: float
    policy: str
    stage: str
    guardrail_violation: float
    normalized_spatial_penalty: float
    mass_tie_breaker: float
    normalized_physical_penalty: float = 0.0
    gain_regularization_score: float = 0.0
    normalization_mass_kg: float = 0.0
    constraint_values: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _finite_or_invalid(self.value),
            "policy": self.policy,
            "stage": self.stage,
            "guardrail_violation": _finite_or_invalid(
                self.guardrail_violation
            ),
            "normalized_spatial_penalty": _finite_or_invalid(
                self.normalized_spatial_penalty
            ),
            "mass_tie_breaker": _finite_or_invalid(self.mass_tie_breaker),
            "normalized_physical_penalty": _finite_or_invalid(
                self.normalized_physical_penalty
            ),
            "gain_regularization_score": _finite_or_invalid(
                self.gain_regularization_score
            ),
            "normalization_mass_kg": _finite_or_invalid(
                self.normalization_mass_kg
            ),
            "constraint_values": [
                _finite_or_invalid(value) for value in self.constraint_values
            ],
        }


def normalize_policy(value: str | None) -> str:
    policy = str(value or "evaluation").strip().lower().replace("_", "-")
    aliases = {
        "mass": "evaluation",
        "legacy": "evaluation",
        "spatial": "staged-spatial",
        "staged": "staged-spatial",
        "aligned": "aligned-spatial-v1",
        "selection-aligned": "aligned-spatial-v1",
        "v2": "aligned-physical-v2",
        "physical-v2": "aligned-physical-v2",
        "objective-v2": "aligned-physical-v2",
        "overflow-first": "aligned-physical-v2-overflow-first",
        "physical-v2-overflow-first": "aligned-physical-v2-overflow-first",
        "objective-v2-overflow-first": "aligned-physical-v2-overflow-first",
    }
    policy = aliases.get(policy, policy)
    if policy not in {
        "evaluation",
        "staged-spatial",
        "aligned-spatial-v1",
        "aligned-physical-v2",
        "aligned-physical-v2-overflow-first",
    }:
        raise ValueError(
            f"Unknown Optuna objective policy {value!r}; expected evaluation "
            "staged-spatial, aligned-spatial-v1, aligned-physical-v2, or "
            "aligned-physical-v2-overflow-first."
        )
    return policy


def is_physical_v2_policy(policy: str | None) -> bool:
    return normalize_policy(policy) in {
        "aligned-physical-v2",
        "aligned-physical-v2-overflow-first",
    }


def uses_late_gas_guardrail(policy: str | None) -> bool:
    return normalize_policy(policy) != "aligned-physical-v2-overflow-first"


def normalized_spatial_penalty(
    objective_terms: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> float:
    raw = float(objective_terms.get("spatial_raw_penalty_mass_kg", 0.0) or 0.0)
    denominator = objective_normalization_mass(metrics)
    if denominator <= 0.0:
        return math.inf
    return max(0.0, raw) / denominator


def objective_normalization_mass(metrics: Mapping[str, Any]) -> float:
    maximum_injected = max(
        (_metric_value(row, "injected_full") for row in metrics.values()),
        default=0.0,
    )
    return maximum_injected * max(1, len(metrics))


def normalized_physical_penalty(
    objective_terms: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    include_late_gas: bool = True,
) -> float:
    denominator = objective_normalization_mass(metrics)
    if denominator <= 0.0:
        return math.inf
    penalty = sum(
        max(0.0, float(objective_terms.get(name, 0.0) or 0.0))
        for name in _physical_penalty_terms(include_late_gas=include_late_gas)
    )
    return penalty / denominator


def _physical_penalty_terms(*, include_late_gas: bool) -> tuple[str, ...]:
    if include_late_gas:
        return V2_PHYSICAL_PENALTY_TERMS
    return tuple(
        name for name in V2_PHYSICAL_PENALTY_TERMS if name != "late_gas_penalty"
    )


def gain_regularization_score(objective_terms: Mapping[str, Any]) -> float:
    z2 = max(
        0.0,
        float(objective_terms.get("layer_gain_regularization_z2", 0.0) or 0.0),
    )
    return V2_GAIN_PRIOR_WEIGHT * z2


def guardrail_constraint_values(
    diagnostics: Mapping[str, Any],
    *,
    limits: Mapping[str, float] = DEFAULT_TRAINING_LIMITS,
    include_late_gas: bool = True,
) -> tuple[float, ...]:
    """Signed Optuna constraints; values <= 0 are physically admissible."""

    required = (
        "post_injection_aqueous_decrease_fraction",
        "ratio_4p1",
        "late_mean_ratio",
        "terminal_ratio",
        "worst_window_mae",
        "worst_window_symmetric_error",
    )
    if include_late_gas:
        required = ("maximum_gas_envelope_excess_fraction", *required)
    if any(name not in diagnostics for name in required):
        return tuple(1.0 for _name in V2_CONSTRAINT_NAMES)
    return (
        (
            _upper_constraint(
                diagnostics,
                "maximum_gas_envelope_excess_fraction",
                limits["maximum_gas_envelope_excess_fraction"],
            )
            if include_late_gas
            else -1.0
        ),
        _upper_constraint(
            diagnostics,
            "post_injection_aqueous_decrease_fraction",
            limits["maximum_aqueous_decrease_fraction"],
        ),
        _interval_constraint(
            diagnostics,
            "ratio_4p1",
            limits["minimum_ratio_4p1"],
            limits["maximum_ratio_4p1"],
        ),
        _interval_constraint(
            diagnostics,
            "late_mean_ratio",
            limits["minimum_late_mean_ratio"],
            limits["maximum_late_mean_ratio"],
        ),
        _interval_constraint(
            diagnostics,
            "terminal_ratio",
            limits["minimum_terminal_ratio"],
            limits["maximum_terminal_ratio"],
        ),
        _upper_constraint(
            diagnostics,
            "worst_window_mae",
            limits["maximum_worst_window_mae"],
        ),
        _upper_constraint(
            diagnostics,
            "worst_window_symmetric_error",
            V2_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR,
        ),
    )


def guardrail_violation(
    diagnostics: Mapping[str, Any],
    *,
    limits: Mapping[str, float] = DEFAULT_TRAINING_LIMITS,
    include_late_gas: bool = True,
) -> float:
    """Continuous dimensionless distance outside the fixed guardrails."""

    violations = [
        _upper_violation(
            diagnostics,
            "post_injection_aqueous_decrease_fraction",
            limits["maximum_aqueous_decrease_fraction"],
        ),
        _interval_violation(
            diagnostics,
            "ratio_4p1",
            limits["minimum_ratio_4p1"],
            limits["maximum_ratio_4p1"],
        ),
        _interval_violation(
            diagnostics,
            "late_mean_ratio",
            limits["minimum_late_mean_ratio"],
            limits["maximum_late_mean_ratio"],
        ),
        _interval_violation(
            diagnostics,
            "terminal_ratio",
            limits["minimum_terminal_ratio"],
            limits["maximum_terminal_ratio"],
        ),
        _upper_violation(
            diagnostics,
            "worst_window_mae",
            limits["maximum_worst_window_mae"],
        ),
    ]
    if include_late_gas:
        violations.insert(
            0,
            _upper_violation(
                diagnostics,
                "maximum_gas_envelope_excess_fraction",
                limits["maximum_gas_envelope_excess_fraction"],
            ),
        )
    return float(sum(violations))


def score_search_objective(
    *,
    policy: str,
    evaluation_objective: float,
    status: str,
    feasible: bool,
    diagnostics: Mapping[str, Any],
    guardrail_failures: Sequence[str],
    objective_terms: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> SearchObjectiveScore:
    """Return the scalar Optuna sees, together with auditable components."""

    resolved = normalize_policy(policy)
    evaluation = float(evaluation_objective)
    if resolved == "evaluation":
        value = evaluation if math.isfinite(evaluation) else INVALID_SCORE
        return SearchObjectiveScore(
            value=value,
            policy=resolved,
            stage="evaluation-objective",
            guardrail_violation=0.0,
            normalized_spatial_penalty=0.0,
            mass_tie_breaker=0.0,
        )

    if status != "ok" or not feasible or not diagnostics or not metrics:
        return SearchObjectiveScore(
            value=INVALID_SCORE,
            policy=resolved,
            stage="invalid",
            guardrail_violation=1.0,
            normalized_spatial_penalty=INVALID_SCORE,
            mass_tie_breaker=INVALID_SCORE,
        )

    is_v2 = is_physical_v2_policy(resolved)
    include_late_gas = uses_late_gas_guardrail(resolved)
    constraints = (
        guardrail_constraint_values(
            diagnostics,
            include_late_gas=include_late_gas,
        )
        if is_v2
        else ()
    )
    violation = (
        sum(max(0.0, value) for value in constraints)
        if is_v2
        else guardrail_violation(
            diagnostics,
            include_late_gas=include_late_gas,
        )
    )
    diagnostic_name = "temporal_balanced_huber" if is_v2 else "selection_score"
    mass_tie_breaker = max(
        0.0, float(diagnostics.get(diagnostic_name, math.inf))
    )
    spatial = normalized_spatial_penalty(objective_terms, metrics)
    physical = (
        normalized_physical_penalty(
            objective_terms,
            metrics,
            include_late_gas=include_late_gas,
        )
        if is_v2
        else 0.0
    )
    gain = gain_regularization_score(objective_terms) if is_v2 else 0.0
    normalization_mass = objective_normalization_mass(metrics)
    if not all(
        math.isfinite(value)
        for value in (spatial, mass_tie_breaker, physical, gain)
    ):
        return SearchObjectiveScore(
            value=INVALID_SCORE,
            policy=resolved,
            stage="invalid-diagnostics",
            guardrail_violation=max(1.0, violation),
            normalized_spatial_penalty=_finite_or_invalid(spatial),
            mass_tie_breaker=_finite_or_invalid(mass_tie_breaker),
            normalized_physical_penalty=_finite_or_invalid(physical),
            gain_regularization_score=_finite_or_invalid(gain),
            normalization_mass_kg=_finite_or_invalid(normalization_mass),
            constraint_values=constraints,
        )
    if is_v2:
        combined = mass_tie_breaker + spatial + physical + gain
        failed = bool(guardrail_failures) or violation > 1e-12
        return SearchObjectiveScore(
            value=combined,
            policy=resolved,
            stage="constrained-infeasible" if failed else "physical-selection",
            guardrail_violation=violation if failed else 0.0,
            normalized_spatial_penalty=spatial,
            mass_tie_breaker=mass_tie_breaker,
            normalized_physical_penalty=physical,
            gain_regularization_score=gain,
            normalization_mass_kg=normalization_mass,
            constraint_values=constraints,
        )
    if resolved == "aligned-spatial-v1":
        aligned_score = spatial + mass_tie_breaker
        if guardrail_failures or violation > 1e-12:
            return SearchObjectiveScore(
                value=100.0 + violation + aligned_score,
                policy=resolved,
                stage="guardrail-recovery",
                guardrail_violation=violation,
                normalized_spatial_penalty=spatial,
                mass_tie_breaker=mass_tie_breaker,
            )
        return SearchObjectiveScore(
            value=aligned_score,
            policy=resolved,
            stage="spatial-selection",
            guardrail_violation=0.0,
            normalized_spatial_penalty=spatial,
            mass_tie_breaker=mass_tie_breaker,
        )

    if guardrail_failures or violation > 1e-12:
        return SearchObjectiveScore(
            value=100.0 + violation + 0.01 * mass_tie_breaker,
            policy=resolved,
            stage="guardrail-recovery",
            guardrail_violation=violation,
            normalized_spatial_penalty=spatial,
            mass_tie_breaker=mass_tie_breaker,
        )

    return SearchObjectiveScore(
        value=spatial + 0.01 * mass_tie_breaker,
        policy=resolved,
        stage="spatial-selection",
        guardrail_violation=0.0,
        normalized_spatial_penalty=spatial,
        mass_tie_breaker=mass_tie_breaker,
    )


def rescore_logged_components(
    row: Mapping[str, Any],
    *,
    policy: str = "aligned-spatial-v1",
) -> SearchObjectiveScore:
    """Rescore one compact trial row without evaluating its images again."""

    resolved = normalize_policy(policy)
    if resolved == "evaluation":
        evaluation = float(row.get("objective", math.inf))
        return SearchObjectiveScore(
            value=evaluation if math.isfinite(evaluation) else INVALID_SCORE,
            policy=resolved,
            stage="evaluation-objective",
            guardrail_violation=0.0,
            normalized_spatial_penalty=0.0,
            mass_tie_breaker=0.0,
        )

    terms = row.get("search_objective_terms", {})
    diagnostics = row.get("diagnostics", {})
    if (
        str(row.get("status", "")) != "ok"
        or not bool(row.get("feasible"))
        or not isinstance(terms, Mapping)
        or not isinstance(diagnostics, Mapping)
        or not diagnostics
    ):
        return SearchObjectiveScore(
            value=INVALID_SCORE,
            policy=resolved,
            stage="invalid",
            guardrail_violation=1.0,
            normalized_spatial_penalty=INVALID_SCORE,
            mass_tie_breaker=INVALID_SCORE,
        )

    is_v2 = is_physical_v2_policy(resolved)
    include_late_gas = uses_late_gas_guardrail(resolved)
    raw_constraints = terms.get("constraint_values", ())
    constraints = tuple(float(value) for value in raw_constraints or ())
    if is_v2 and (
        len(constraints) != len(V2_CONSTRAINT_NAMES)
        or not include_late_gas
    ):
        constraints = guardrail_constraint_values(
            diagnostics,
            include_late_gas=include_late_gas,
        )
    violation = (
        sum(max(0.0, value) for value in constraints)
        if is_v2
        else float(terms.get("guardrail_violation", math.inf))
    )
    spatial = float(terms.get("normalized_spatial_penalty", math.inf))
    mass = float(
        terms.get(
            "mass_tie_breaker",
            diagnostics.get(
                "temporal_balanced_huber" if is_v2 else "selection_score",
                diagnostics.get("selection_score", math.inf),
            ),
        )
    )
    physical = 0.0
    gain = 0.0
    normalization_mass = float(terms.get("normalization_mass_kg", 0.0) or 0.0)
    if is_v2:
        objective_terms = row.get("objective_terms", {})
        if not isinstance(objective_terms, Mapping):
            objective_terms = {}
        physical_value = terms.get("normalized_physical_penalty")
        if physical_value is None or not include_late_gas:
            if normalization_mass <= 0.0:
                raw_spatial = float(
                    objective_terms.get("spatial_raw_penalty_mass_kg", 0.0) or 0.0
                )
                if raw_spatial > 0.0 and spatial > 0.0:
                    normalization_mass = raw_spatial / spatial
            physical_value = (
                sum(
                    max(0.0, float(objective_terms.get(name, 0.0) or 0.0))
                    for name in _physical_penalty_terms(
                        include_late_gas=include_late_gas
                    )
                )
                / normalization_mass
                if normalization_mass > 0.0
                else 0.0
            )
        physical = float(physical_value)
        gain = float(
            terms.get(
                "gain_regularization_score",
                gain_regularization_score(objective_terms),
            )
        )
    if not all(
        math.isfinite(value)
        for value in (violation, spatial, mass, physical, gain)
    ):
        return SearchObjectiveScore(
            value=INVALID_SCORE,
            policy=resolved,
            stage="invalid-diagnostics",
            guardrail_violation=_finite_or_invalid(violation),
            normalized_spatial_penalty=_finite_or_invalid(spatial),
            mass_tie_breaker=_finite_or_invalid(mass),
            normalized_physical_penalty=_finite_or_invalid(physical),
            gain_regularization_score=_finite_or_invalid(gain),
            normalization_mass_kg=_finite_or_invalid(normalization_mass),
            constraint_values=constraints,
        )

    failures = list(row.get("guardrail_failures", []) or [])
    if not include_late_gas:
        failures = [name for name in failures if name != "late_gas_envelope"]
    failed = bool(failures) or violation > 1e-12
    if is_v2:
        combined = spatial + mass + physical + gain
        return SearchObjectiveScore(
            value=combined,
            policy=resolved,
            stage="constrained-infeasible" if failed else "physical-selection",
            guardrail_violation=violation if failed else 0.0,
            normalized_spatial_penalty=spatial,
            mass_tie_breaker=mass,
            normalized_physical_penalty=physical,
            gain_regularization_score=gain,
            normalization_mass_kg=normalization_mass,
            constraint_values=constraints,
        )
    mass_weight = 1.0 if resolved == "aligned-spatial-v1" else 0.01
    combined = spatial + mass_weight * mass
    return SearchObjectiveScore(
        value=(100.0 + violation + combined) if failed else combined,
        policy=resolved,
        stage="guardrail-recovery" if failed else "spatial-selection",
        guardrail_violation=violation if failed else 0.0,
        normalized_spatial_penalty=spatial,
        mass_tie_breaker=mass,
    )


def _metric_value(row: Any, name: str) -> float:
    value = row.get(name) if isinstance(row, Mapping) else getattr(row, name)
    return float(value)


def _finite_or_invalid(value: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else INVALID_SCORE


def _upper_violation(
    diagnostics: Mapping[str, Any],
    name: str,
    upper: float,
) -> float:
    value = float(diagnostics[name])
    return max(0.0, value - upper) / max(abs(float(upper)), 1e-6)


def _interval_violation(
    diagnostics: Mapping[str, Any],
    name: str,
    lower: float,
    upper: float,
) -> float:
    value = float(diagnostics[name])
    width = max(float(upper) - float(lower), 1e-6)
    return max(float(lower) - value, value - float(upper), 0.0) / width


def _upper_constraint(
    diagnostics: Mapping[str, Any],
    name: str,
    upper: float,
) -> float:
    value = float(diagnostics[name])
    return (value - float(upper)) / max(abs(float(upper)), 1e-6)


def _interval_constraint(
    diagnostics: Mapping[str, Any],
    name: str,
    lower: float,
    upper: float,
) -> float:
    value = float(diagnostics[name])
    width = max(float(upper) - float(lower), 1e-6)
    return max(float(lower) - value, value - float(upper)) / width
