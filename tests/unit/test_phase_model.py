"""Unit tests for phase-model inference from a delivered model folder."""

import json

import numpy as np

import darsia

NAMES = darsia.PHASE_FEATURE_NAMES
UNKNOWN = {"canvas": None, "accept_canvas": True}
"""A delivered model, which records no canvas, read for images without one by
explicit choice."""


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
    return darsia.read_phase_model(folder, pixel_size_mm=0.5, **UNKNOWN)


def _unit(name):
    """Coefficients selecting one feature, so that the logit is its value."""
    return [float(other == name) for other in NAMES]


def _feature(probability):
    """Feature value behind a probability of a head selecting one feature."""
    return np.log(probability / (1 - probability))


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


def test_probabilities_compare_with_baseline_and_previous_frame(tmp_path):
    """Change against the baseline and the previous frame, in the images' dtype."""
    model = _model(tmp_path, _head(_unit("delta_r")), _head(_unit("dynamic_delta_g")))
    baseline = np.full((40, 50, 3), 0.3)
    previous = baseline + [0.0, 0.05, 0.0]
    current = baseline + [0.12, 0.2, 0.0]
    co2, gas = darsia.phase_probabilities(
        model, current, baseline, previous, pixel_size_mm=0.5
    )
    assert co2.dtype == np.float64
    np.testing.assert_allclose(_feature(co2), 0.12)
    np.testing.assert_allclose(_feature(gas), 0.15)
    co2, gas = darsia.phase_probabilities(
        model,
        *(image.astype(np.float32) for image in (current, baseline)),
        pixel_size_mm=0.5,
    )
    assert gas.dtype == np.float32
    np.testing.assert_allclose(_feature(co2), 0.12, atol=1e-5)
    np.testing.assert_array_equal(gas, 0.5)


def test_probabilities_need_a_valid_texture_window(tmp_path):
    """An invalid pixel removes the texture window around it, scaled with the grid."""
    rng = np.random.default_rng(2)
    model = _model(
        tmp_path, _head(rng.normal(size=36), 0.1), _head(rng.normal(size=36), -0.2)
    )
    baseline = rng.uniform(0.1, 0.9, (60, 70, 3))
    current = baseline * rng.uniform(0.8, 1.2, baseline.shape)
    current[30, 40, 1] = np.nan
    # The texture sigma of 1 mm is 2 pixels on the training grid, 4 on the finer.
    for pixel_size_mm, half in ((0.5, 8), (0.25, 16)):
        co2, gas = darsia.phase_probabilities(
            model, current, baseline, pixel_size_mm=pixel_size_mm
        )
        unsupported = np.isnan(co2)
        window = np.s_[30 - half : 31 + half, 40 - half : 41 + half]
        assert unsupported[window].all()
        assert unsupported.sum() == (2 * half + 1) ** 2
        assert np.array_equal(unsupported, np.isnan(gas))
        assert ((co2[~unsupported] > 0) & (co2[~unsupported] < 1)).all()
