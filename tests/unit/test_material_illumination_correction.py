"""Unit tests for the material illumination correction and its gain fit."""

import dataclasses
import tomllib

import numpy as np
import pytest

import darsia
from darsia.corrections.color.materialilluminationcorrection import _polynomial

FIT = darsia.MaterialFitSettings(
    **tomllib.loads(
        """
        # The values the fit ran with in production.
        spatial_ridge = 0.002
        offset_ridge = 0.02
        residual_constant_ridge = 0.02
        residual_spatial_ridge = 0.01
        iterations = 8
        huber_threshold = 1.5
        clip_level = 0.98
        dark_level = 8.0
        minimum_block_cells = 32
        chroma_cutoff = 0.08
        chroma_sigmas = 3.5
        """
    )
)


def _correction(field_shape=(4, 6)) -> darsia.MaterialIlluminationCorrection:
    """Constant common gain (2, 3, 4) and a sloped gain of material 7 in rows 2-3."""
    common = np.zeros((6, 3))
    common[0] = np.log([2.0, 3.0, 4.0])
    field = np.zeros((6, 3))
    field[0] = np.log(0.5)
    field[1] = [0.1, 0.0, -0.1]
    material_map = np.zeros((4, 6), dtype=int)
    material_map[2:] = 7
    return darsia.MaterialIlluminationCorrection(
        common, {7: field}, material_map, field_shape
    )


def _sloped_gain() -> np.ndarray:
    """Expected gain of material 7 along a row of six pixels."""
    x = (np.arange(6) + 0.5) * 2 / 6 - 1
    return 0.5 * np.exp(np.outer(x, [0.1, 0.0, -0.1]))


def test_gain_of_each_material_and_of_the_common_field():
    """Pixels of material 7 use its field, all other pixels the common field."""
    corrected = _correction().correct_array(np.ones((4, 6, 3), dtype=np.float32))

    assert corrected.dtype == np.float32
    np.testing.assert_allclose(
        corrected[:2], np.broadcast_to([2, 3, 4], (2, 6, 3)), rtol=1e-6
    )
    np.testing.assert_allclose(corrected[2], _sloped_gain(), rtol=1e-6)
    np.testing.assert_allclose(corrected[3], _sloped_gain(), rtol=1e-6)


def test_coarse_fitting_grid_gives_the_same_gain():
    """Pixel centres mapped to a half-size fitting grid keep their positions."""
    coarse = _correction(field_shape=(2, 3))

    np.testing.assert_allclose(
        coarse.gain(np.float64), _correction().gain(np.float64), rtol=1e-12
    )


def test_invalid_pixels_become_nan_and_zero_stays_zero():
    """A non-finite or negative channel invalidates the whole pixel."""
    image = np.ones((4, 6, 3))
    image[0, 0, 1] = np.nan
    image[1, 1, 2] = -1.0
    image[3, 5] = 0.0

    corrected = _correction().correct_array(image)

    assert np.isnan(corrected[0, 0]).all() and np.isnan(corrected[1, 1]).all()
    np.testing.assert_array_equal(corrected[3, 5], 0.0)
    assert np.isfinite(corrected[0, 1:]).all()


def _samples(count: int = 40):
    """Noise-free samples of a known common field with sum-zero offsets."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(-0.9, 0.9, (3 * count, 2))
    materials = np.repeat([1, 2, 3], count)
    common = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.3, -0.2, 0.1],
            [-0.1, 0.2, 0.05],
            [0.05, 0, 0],
            [0, 0.05, 0],
            [0, 0, 0.05],
        ]
    )
    offsets = {1: [0.1, 0.0, -0.1], 2: [-0.05, 0.1, 0.0], 3: [-0.05, -0.1, 0.1]}
    log_gain = _polynomial(positions) @ common + np.array(
        [offsets[m] for m in materials]
    )
    target = np.log(np.full((3 * count, 3), 0.5))
    return positions, materials, np.exp(target - log_gain), target, common, log_gain


def test_fit_recovers_a_known_field():
    """Noise-free samples give back the generating field up to the ridge bias."""
    positions, materials, measured, target, common, log_gain = _samples()

    fitted, fields = darsia.fit_material_gain(
        positions,
        materials,
        measured,
        target,
        fit_mask=np.ones(len(materials), dtype=bool),
        materials=[1, 2, 3],
        settings=FIT,
    )

    np.testing.assert_allclose(fitted, common, atol=5e-3)
    for material in (1, 2, 3):
        keep = materials == material
        predicted = _polynomial(positions[keep]) @ fields[material]
        np.testing.assert_allclose(predicted, log_gain[keep], atol=1e-3)


def test_fit_down_weights_an_outlying_sample():
    """One sample three times too bright hardly moves its material's field."""
    positions, materials, measured, target, _, log_gain = _samples()
    measured[5] *= 3

    _, fields = darsia.fit_material_gain(
        positions,
        materials,
        measured,
        target,
        fit_mask=np.ones(len(materials), dtype=bool),
        materials=[1, 2, 3],
        settings=FIT,
    )

    error = np.abs(_polynomial(positions[:40]) @ fields[1] - log_gain[:40])
    assert np.delete(error, 5).max() < 5e-3


def test_the_fit_refuses_an_iteration_count_below_one():
    """Without a robust solve there are no coefficients, so zero is refused."""
    positions, materials, measured, target, _, _ = _samples()

    with pytest.raises(ValueError, match="positive integer"):
        darsia.fit_material_gain(
            positions,
            materials,
            measured,
            target,
            fit_mask=np.ones(len(materials), dtype=bool),
            materials=[1, 2, 3],
            settings=dataclasses.replace(FIT, iterations=0),
        )
