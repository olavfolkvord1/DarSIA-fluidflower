"""Unit tests for the Beer–Lambert response per sand on synthetic frames."""

import dataclasses
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

    [path_ratios]
    mean_mm = 5.0
    tile_mm = 3.0
    band_mm = [1.0, 3.0]
    far_mm = 6.0
    min_dissolved_share = 0.95
    gas_free_mm = 5.0
    min_band_mm2 = 0.55  # 20 pixels of 0.0279 mm²
    flat_tol_per_mm = 0.02
    arrived_share = 0.5
    equalise_h = 2.5
    min_level = 0.03
    ceiling_margin = 0.95
    min_tiles = 30
    block_mm = 50.0
    bootstrap = 400
    seed = 0
    """
)
SETTINGS = darsia.BeerLambertSettings(**CONFIGURATION["beer_lambert"])
SHADOW = darsia.ShadowSettings(**CONFIGURATION["shadow_light"])
RATIO_SETTINGS = darsia.PathRatioSettings(**CONFIGURATION["path_ratios"])

DIRECTION = np.array([0.8, 0.58, 0.02]) / np.linalg.norm([0.8, 0.58, 0.02])
"""Direction of the log colour change of dissolved CO2."""

TONE = {1: (0.30, 0.28, 0.22), 2: (0.35, 0.33, 0.26), 5: (0.25, 0.23, 0.18)}


def _path(background, ratio=1.0, amplitude=0.14):
    """Signal of a sand's colour path: the change of its background at a log change
    of ``ratio`` times the reference sand's, up to ``amplitude``."""
    knots = [
        background * np.expm1(ratio * amplitude * t * DIRECTION) for t in (0, 0.5, 1)
    ]
    path = darsia.ColorPath(colors=knots, base_color=np.zeros(3), mode="rgb")
    return darsia.ColorPathSignal(
        darsia.ColorPathInterpolation(path, darsia.ColorMode.RELATIVE, [0.0, 0.5, 1.0]),
        darsia.PWTransformation([0.0, 1.0], [0.0, 1.0]),
    )


def test_a_uniform_background_reads_the_reference_sands_linear_change():
    shape = (130, 140)
    facies = np.full(shape, 5, np.uint8)
    facies[:, 105:] = 1
    sand_map = darsia.ScalarImage(facies, dimensions=[0.065, 0.07])
    pitch = (0.5e-3, 0.5e-3)
    baseline = np.array([TONE[label] for label in facies.ravel()]).reshape(shape + (3,))
    ratio = np.where(facies == 1, 0.8, 1.0)[..., None]
    current = baseline * np.exp(ratio * 0.07 * DIRECTION)
    support = np.ones(shape, bool)
    dissolved = support.copy()
    dissolved[:20] = False
    path = _path(np.array(TONE[5]))
    settings = dataclasses.replace(SETTINGS, path_ratio={1: 0.8})
    background = darsia.reference_background(
        baseline, sand_map, pitch, support, np.ones(shape), settings
    )
    reading = darsia.BeerLambertReading(
        path, 1.25, background, sand_map, pitch, settings
    )

    signal = reading(current, baseline, support, dissolved, facies)

    own = facies == 5
    reference = np.broadcast_to(TONE[5], baseline.shape)
    np.testing.assert_allclose(background, reference, rtol=1e-12)
    linear = darsia.SandColorPaths({5: path}, {5: 1.25})
    expected = linear(current, baseline, support, dissolved, facies)
    np.testing.assert_allclose(signal[own], expected[own], rtol=1e-12)
    # The other sand's smaller change on its brighter background reads as the
    # reference sand's change at the same concentration.
    level = signal[dissolved & own].mean()
    np.testing.assert_allclose(signal[dissolved & ~own], level, rtol=1e-12)
    assert level > 0.6 and not signal[~dissolved].any()


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


def test_path_ratios_are_measured_where_two_sands_meet():
    shape = (300, 500)  # 0.2 mm pixels
    facies = np.full(shape, 1, np.uint8)
    facies[100:, :] = 5
    facies[200:, :] = 2
    sand_map = darsia.ScalarImage(facies, dimensions=[0.06, 0.1])
    ratio = {1: 0.8, 2: 0.9, 5: 1.0}
    baseline = np.array([TONE[label] for label in facies.ravel()]).reshape(shape + (3,))
    paths = {sand: _path(np.array(TONE[sand]), ratio[sand]) for sand in ratio}
    reading = darsia.SandColorPaths(paths, {sand: 1.0 for sand in ratio})
    path_ratio = np.vectorize(ratio.get)(facies)[..., None]
    hours = [0.0, 1.0, 3.0, 4.0, 5.0]
    frames = []
    for hour in hours:
        concentration = 0.5 * (1 - np.exp(-(hour + 0.5)))
        current = baseline * np.exp(path_ratio * 0.14 * concentration * DIRECTION)
        phase = np.full(shape, darsia.PhaseLabel.DISSOLVED, np.uint8)
        rules = np.zeros(shape, np.uint8)
        frames.append((current, phase, rules, np.ones(shape, bool)))

    measured = darsia.path_ratios(
        frames,
        hours,
        baseline,
        sand_map,
        (0.2e-3, 0.2e-3),
        np.ones(shape, bool),
        reading,
        np.ones(shape),
        5,
        1.2,
        RATIO_SETTINGS,
    )

    assert set(measured.pairs) == {(1, 5), (2, 5)}
    assert all(pair.tiles == 3 * 34 for pair in measured.pairs.values())
    np.testing.assert_allclose(
        [measured.ratio[1], measured.ratio[2]], [0.8, 0.9], rtol=1e-9
    )
