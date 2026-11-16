"""The measurement of the material light correction's targets on a run's rig."""

import tomllib
from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.config.utils import settings_from_table
from darsia.presets.workflows.setup.target_steps import (
    MaterialTargetSettings,
    measure_rig_targets,
)

SETTINGS = settings_from_table(
    MaterialTargetSettings,
    tomllib.loads(
        """
        # Production's settings, but for the cells and blocks of the small image.
        materials = [1, 2]
        anchor = [400.0, 20.0]
        minimum_blocks = 8
        [light_field]
        cell_pixels = 4
        degree = 2
        block_cells = 2
        minimum_blocks = 30
        iterations = 8
        huber = 1.345
        samples_per_unknown = 10
        """
    ),
)
CORRECTION = tomllib.loads(
    """
    # The material light correction's table: production's fit settings.
    fit_shape = [80, 120]
    block_size = 8
    edge_margin = 0.01
    dn_scale = 1000.0
    [fit]
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
COLOURS = {1: np.array([0.02, 0.04, 0.03]), 2: np.array([0.01, 0.03, 0.02])}
"""The sands' colours where the light is that of the anchor."""


def test_the_targets_are_the_sands_colours_in_the_anchors_light():
    """A rig whose sands are lit by a smooth field: the targets are each sand's
    colour in the light at the anchor, in the units of the correction."""
    shape = (320, 480)
    materials = np.ones(shape, int)
    materials[160:] = 2
    light = darsia.SandLightField(
        coefficients=np.array([0.3, -0.2, 0.05, 0.0, -0.04]),
        levels={},
        degree=2,
        grid_shape=(80, 120),
        image_shape=shape,
        samples=0,
        spread={},
    )
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    terms = darsia.polynomial_terms(*light.coordinates(columns, rows), 2)
    at_anchor = darsia.polynomial_terms(*light.coordinates(400.0, 20.0), 2)
    gain = np.exp(terms @ light.coefficients - at_anchor @ light.coefficients)
    rng = np.random.default_rng(7)
    colour = np.stack([COLOURS[m] for m in (1, 2)])[materials - 1]
    baseline = colour * gain[..., None] * rng.lognormal(0.0, 0.003, (*shape, 1))
    rig = SimpleNamespace(
        shape_corrected_baseline=darsia.Image(
            baseline.astype(np.float32), dimensions=[0.32, 0.48], series=False
        ),
        facies=darsia.Image(materials, dimensions=[0.32, 0.48], series=False),
    )

    found = measure_rig_targets(rig, CORRECTION, SETTINGS)

    assert set(found) == {"targets", "blocks", "light_field"}
    for material, rgb in COLOURS.items():
        np.testing.assert_allclose(
            found["targets"][str(material)], np.log(1000.0 * rgb), atol=5e-3
        )
    assert found["light_field"]["anchor"] == [400.0, 20.0]
    np.testing.assert_allclose(
        found["light_field"]["coefficients"], light.coefficients, atol=5e-3
    )
    for block in found["blocks"].values():
        assert block["kept"] == block["training"] > 0
