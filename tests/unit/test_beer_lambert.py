"""Unit tests for the Beer–Lambert response per sand on synthetic frames."""

import tomllib

import numpy as np

import darsia

CONFIGURATION = tomllib.loads(
    """
    # The values the final rerun ran with.
    [beer_lambert]
    reference_sand = 5
    mean_mm = 5.0
    background_degree = 6
    background_margin_mm = 3.0
    unshaded_light = 0.95
    background_fill = 0.25
    background_iterations = 6
    background_huber_threshold = 1.5

    [beer_lambert.path_ratio]
    1 = 0.7970970082154487
    2 = 0.8942802086755279
    13 = 0.9924228967813956

    [shadow_light]
    fine_mm = 5.0
    broad_mm = 25.0
    below = 0.75
    reach_mm = 12.0
    smallest_mm2 = 2000.0
    darkest = 0.2

    """
)
SETTINGS = darsia.BeerLambertSettings(**CONFIGURATION["beer_lambert"])
SHADOW = darsia.ShadowSettings(**CONFIGURATION["shadow_light"])

TONE = {1: (0.30, 0.28, 0.22), 2: (0.35, 0.33, 0.26), 5: (0.25, 0.23, 0.18)}


def test_an_empty_block_gives_no_block_mean():
    """Blocks without reference-sand pixels stay out of the background's fit, also
    when the fill share is zero."""
    shape = (130, 140)
    facies = np.full(shape, 5, np.uint8)
    facies[:, 105:] = 1
    sand_map = darsia.ScalarImage(facies, dimensions=[0.065, 0.07])
    baseline = np.array([TONE[label] for label in facies.ravel()]).reshape(shape + (3,))
    support = np.ones(shape, bool)
    support[:10, :10] = False  # one block without pixels inside the reference sand
    empty = darsia.BeerLambertSettings(
        **dict(CONFIGURATION["beer_lambert"], background_fill=0.0)
    )

    background = darsia.reference_background(
        baseline, sand_map, (0.5e-3, 0.5e-3), support, np.ones(shape), empty
    )

    np.testing.assert_allclose(background, np.broadcast_to(TONE[5], shape + (3,)))


def test_the_background_leaves_out_a_shadow():
    shape = (200, 300)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    facies = np.full(shape, 5, np.uint8)
    facies[150:, :] = 2
    sand_map = darsia.ScalarImage(facies, dimensions=[0.1, 0.15])
    pitch = (0.5e-3, 0.5e-3)
    light = 0.8 + 0.3 * np.exp(-((cols - 120) ** 2 + (rows - 70) ** 2) / 4e4)
    unshaded = np.array([TONE[label] for label in facies.ravel()]).reshape(shape + (3,))
    unshaded *= light[..., None]
    shade = np.ones(shape)
    shade[30:100, 60:200] = 0.4  # 35 x 70 mm
    baseline = unshaded * shade[..., None]
    support = np.ones(shape, bool)

    shadow = darsia.shadow_light(baseline, sand_map, pitch, [2, 5], SHADOW)
    background = darsia.reference_background(
        baseline, sand_map, pitch, support, shadow, SETTINGS
    )
    followed = darsia.reference_background(
        baseline, sand_map, pitch, support, np.ones(shape), SETTINGS
    )

    inside = shade < 1
    assert (shadow[inside] < SETTINGS.unshaded_light).all()
    own = facies == 5
    np.testing.assert_allclose(background[own], unshaded[own], rtol=0.02)
    assert (followed[inside] / unshaded[inside]).mean() < 0.9
