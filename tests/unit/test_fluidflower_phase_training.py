from __future__ import annotations

import numpy as np
import pytest

from darsia.fluidflower.phase_features import (
    DARSIA_CHANGE_DETECTOR,
    NO_CHANGE_DETECTOR,
)
from darsia.fluidflower.phase_training.trainer import (
    CHANGE_ABLATION_VARIANT,
    change_evidence_ablation_report,
    model_variants,
)
from darsia.fluidflower.phase_training.logistic import (
    _pack_weak_bags_cuda,
    _packed_weak_terms_cuda,
    cuda_available,
)


def _head_report(score: float, run_score: float) -> dict[str, object]:
    return {
        "selected_regularization": 0.1,
        "oof_metrics": {
            "balanced_accuracy": score,
            "roc_auc": score + 0.05,
            "average_precision": score + 0.02,
        },
        "regularization_grid": [
            {
                "regularization": 0.1,
                "global_threshold_run_metrics": [
                    {
                        "held_out_run": "ac52",
                        "balanced_accuracy": run_score,
                    }
                ],
            }
        ],
    }


def test_change_evidence_ablation_is_explicit_and_non_default() -> None:
    default = model_variants()
    ablation = model_variants(include_change_ablation=True)

    assert CHANGE_ABLATION_VARIANT not in default
    assert default["generic"].change_detector == DARSIA_CHANGE_DETECTOR
    assert ablation[CHANGE_ABLATION_VARIANT].change_detector == NO_CHANGE_DETECTOR
    assert set(ablation[CHANGE_ABLATION_VARIANT].feature_names) < set(
        default["generic"].feature_names
    )


def test_change_evidence_ablation_report_uses_complete_run_holdout_metrics() -> None:
    report = change_evidence_ablation_report(
        {
            "generic": {"heads": {"co2_presence": _head_report(0.82, 0.80)}},
            CHANGE_ABLATION_VARIANT: {
                "heads": {"co2_presence": _head_report(0.70, 0.66)}
            },
        }
    )

    assert report["status"] == "ok"
    assert report["validation"] == "leave-one-complete-run-out"
    assert report["mean_balanced_accuracy_delta"] == pytest.approx(0.12)
    holdout = report["heads"]["co2_presence"]["leave_one_run_out"][0]
    assert holdout["run"] == "ac52"
    assert holdout["with_darsia_change"] == pytest.approx(0.80)
    assert holdout["without_darsia_change"] == pytest.approx(0.66)
    assert holdout["delta"] == pytest.approx(0.14)


@pytest.mark.skipif(not cuda_available(), reason="CUDA/CuPy is unavailable")
def test_packed_cuda_mil_terms_match_sequential_bag_evaluation() -> None:
    import cupy as cp

    rng = np.random.default_rng(20260903)
    matrix = rng.normal(size=(29, 5)).astype(np.float32)
    mean = rng.normal(size=5).astype(np.float32)
    scale = rng.uniform(0.5, 1.5, size=5).astype(np.float32)
    bags = [
        (np.arange(0, 8), 1, 0.6, 10),
        (np.arange(8, 13), 1, 0.4, 11),
        (np.arange(13, 23), 0, 0.7, 12),
        (np.arange(23, 29), 0, 0.3, 13),
    ]
    weight_sum = sum(bag[2] for bag in bags)
    top_fraction = 0.4
    coefficients = cp.asarray(rng.normal(size=5), dtype=cp.float32)
    intercept = cp.float32(0.17)

    packed_terms = []
    for target in (1, 0):
        packed = _pack_weak_bags_cuda(
            matrix,
            bags,
            target=target,
            mean=mean,
            scale=scale,
            weight_sum=weight_sum,
            positive_top_fraction=top_fraction,
        )
        assert packed is not None
        packed_terms.append(_packed_weak_terms_cuda(packed, coefficients, intercept))
    packed_gradient = packed_terms[0][0] + packed_terms[1][0]
    packed_bias = packed_terms[0][1] + packed_terms[1][1]
    packed_loss = packed_terms[0][2] + packed_terms[1][2]

    reference_gradient = cp.zeros_like(coefficients)
    reference_bias = cp.float32(0.0)
    reference_loss = cp.float32(0.0)
    for rows, target, bag_weight, _region_index in bags:
        weak_matrix = cp.asarray((matrix[rows] - mean) / scale, dtype=cp.float32)
        logits = cp.clip(weak_matrix @ coefficients + intercept, -30.0, 30.0)
        probability = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-logits))
        normalized_weight = cp.float32(bag_weight / weight_sum)
        if target == 1:
            count = max(1, round(top_fraction * int(logits.size)))
            selected = cp.argpartition(probability, -count)[-count:]
            selected_logits = logits[selected]
            residual = (
                cp.float32(1.0) / (cp.float32(1.0) + cp.exp(-selected_logits))
                - cp.float32(1.0)
            ) / cp.float32(count)
            reference_gradient += normalized_weight * (
                weak_matrix[selected].T @ residual
            )
            reference_bias += normalized_weight * cp.sum(residual, dtype=cp.float32)
            reference_loss += normalized_weight * cp.mean(
                cp.logaddexp(cp.float32(0.0), -selected_logits),
                dtype=cp.float32,
            )
        else:
            residual = probability / cp.float32(int(logits.size))
            reference_gradient += normalized_weight * (weak_matrix.T @ residual)
            reference_bias += normalized_weight * cp.sum(residual, dtype=cp.float32)
            reference_loss += normalized_weight * cp.mean(
                cp.logaddexp(cp.float32(0.0), logits),
                dtype=cp.float32,
            )

    np.testing.assert_allclose(
        cp.asnumpy(packed_gradient),
        cp.asnumpy(reference_gradient),
        rtol=2e-5,
        atol=2e-6,
    )
    assert float(packed_bias.item()) == pytest.approx(
        float(reference_bias.item()), rel=2e-5, abs=2e-6
    )
    assert float(packed_loss.item()) == pytest.approx(
        float(reference_loss.item()), rel=2e-5, abs=2e-6
    )
