"""The light field shared by the sands of a photograph, with one level per sand."""

import numpy as np
import pytest

import darsia

SETTINGS = darsia.SandLightFieldSettings(
    cell_pixels=4,
    degree=2,
    block_cells=2,
    minimum_blocks=30,
    iterations=8,
    huber=1.345,
    samples_per_unknown=10,
)
"""Production's settings, but for the cells and blocks of the small image."""
FIELD = np.array([0.2, -0.1, 0.05, 0.02, -0.03])
"""The light field of the synthetic photograph."""
LEVELS = {1: np.log(0.3), 2: np.log(0.12), 5: np.log(0.05)}
"""The log green of the synthetic sands."""


def photograph(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Three sands in bands, lit by :data:`FIELD`, with a grain texture."""
    shape = (96, 160)
    materials = np.full(shape, 1)
    materials[32:64] = 2
    materials[64:] = 5
    field = darsia.SandLightField(
        coefficients=FIELD,
        levels={},
        degree=2,
        grid_shape=(24, 40),
        image_shape=shape,
        samples=0,
        spread={},
    )
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    x, y = field.coordinates(columns, rows)
    light = darsia.polynomial_terms(x, y, 2) @ FIELD
    level = np.vectorize(LEVELS.get)(materials)
    green = np.exp(level + light) * rng.lognormal(0.0, 0.002, shape)
    image = np.stack((0.5 * green, green, 0.8 * green), axis=-1).astype(np.float32)
    return image, materials


def test_the_terms_are_ordered_by_degree_then_power_of_x():
    """x, y; then y², xy, x²: x**i * y**(d - i) by degree d and power i."""
    terms = darsia.polynomial_terms(np.array(2.0), np.array(3.0), 2)
    np.testing.assert_array_equal(terms, [3.0, 2.0, 9.0, 6.0, 4.0])


def test_the_field_and_the_levels_are_recovered():
    """The field of a photograph lit by a known field, and each sand's level."""
    rng = np.random.default_rng(1)
    image, materials = photograph(rng)
    found = darsia.fit_sand_light_field(
        image, materials, np.ones(materials.shape, bool), SETTINGS
    )
    np.testing.assert_allclose(found.coefficients, FIELD, atol=2e-3)
    assert set(found.levels) == {1, 2, 5}
    for material, level in LEVELS.items():
        assert found.levels[material] == pytest.approx(level, abs=2e-3)
    assert found.grid_shape == (24, 40) and found.samples == 240
    before, after = found.spread[2]
    assert after < before


def test_flattening_keeps_the_light_at_the_anchor():
    """Divided by the field, each sand is uniform at its light at the anchor."""
    rng = np.random.default_rng(2)
    image, materials = photograph(rng)
    found = darsia.fit_sand_light_field(
        image, materials, np.ones(materials.shape, bool), SETTINGS
    )
    anchor = (150.0, 10.0)
    flat = found.flatten(image, anchor)
    assert flat.dtype == np.float32
    x, y = found.coordinates(*anchor)
    kept = darsia.polynomial_terms(x, y, 2) @ FIELD
    for material, level in LEVELS.items():
        green = flat[..., 1][materials == material]
        np.testing.assert_allclose(green, np.exp(level + kept), rtol=0.01)
    assert flat[10, 150, 1] == pytest.approx(image[10, 150, 1], rel=1e-6)


def test_too_little_sand_is_refused():
    """A mask of too few blocks leaves no fit."""
    rng = np.random.default_rng(3)
    image, materials = photograph(rng)
    mask = np.zeros(materials.shape, bool)
    mask[:16, :40] = True
    with pytest.raises(ValueError, match="Too few blocks"):
        darsia.fit_sand_light_field(image, materials, mask, SETTINGS)
