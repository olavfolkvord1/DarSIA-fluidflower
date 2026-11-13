"""Unit tests for phase-model inference from a delivered model folder."""

import json

import numpy as np

import darsia

NAMES = darsia.PHASE_FEATURE_NAMES


def _head(coefficients, intercept=0.0) -> dict:
    """A stored head in the delivered format, on unit-scaled features."""
    return {
        "feature_names": list(NAMES),
        "feature_mean": [0.0] * len(NAMES),
        "feature_scale": [1.0] * len(NAMES),
        "coefficients": list(coefficients),
        "intercept": intercept,
        "texture_sigma": 2.0,
        "feature_contract": {"input_color_space": "camera-white-relative-linear-rgb"},
    }


def _model(folder, co2_head, gas_head):
    """Write a model folder in the delivered layout; read it as trained on 0.5 mm."""
    heads = folder / "models" / "generic"
    heads.mkdir(parents=True)
    for role, head in (("co2_presence", co2_head), ("gas_within_co2", gas_head)):
        (heads / f"{role}_signal.json").write_text(json.dumps(head))
    policy = {"models": {"generic": {"gas_within_co2": {"presence_threshold": 0.66}}}}
    (folder / "phase_decision_policy.json").write_text(json.dumps(policy))
    return darsia.read_phase_model(folder, pixel_size_mm=0.5)


def _unit(name):
    """Coefficients selecting one feature, so that the logit is its value."""
    return [float(other == name) for other in NAMES]


def test_read_phase_model_reads_the_two_heads(tmp_path):
    """Coefficients, intercepts, the texture scale in mm and the gas threshold
    come from the folder."""
    model = _model(tmp_path, _head(_unit("delta_r"), 1.5), _head(_unit("delta_g")))
    assert model.texture_sigma_mm == 1.0
    assert model.gas_threshold == 0.66
    assert model.co2_presence.intercept == 1.5
    assert model.co2_presence.coefficients[NAMES.index("delta_r")] == 1.0
    assert model.gas_within_co2.coefficients[NAMES.index("delta_g")] == 1.0


def test_head_probability_is_the_clipped_logistic_in_the_feature_dtype():
    """The logit is standardized, linear and clipped to [-30, 30]."""
    head = darsia.LogisticHead(
        feature_mean=np.array([1.0, 0.0]),
        feature_scale=np.array([2.0, 1.0]),
        coefficients=np.array([3.0, -1.0]),
        intercept=0.5,
    )
    features = np.array([[3.0, 1.0], [1.0, 100.0]], dtype=np.float32)
    probability = head.probability(features)
    assert probability.dtype == np.float32
    logit = (3.0 - 1.0) / 2.0 * 3.0 - 1.0 + 0.5
    np.testing.assert_allclose(
        probability, [1 / (1 + np.exp(-logit)), 1 / (1 + np.exp(30.0))], rtol=1e-6
    )
