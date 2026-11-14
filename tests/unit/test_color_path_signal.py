"""Unit tests for reading colour changes along colour paths without a cap."""

import numpy as np

import darsia

KNOTS = [np.zeros(3), np.array([0.01, 0.01, 0.0]), np.array([0.02, 0.03, 0.0])]


def _signal(values=(0.0, 0.5, 1.0), supports=(0.0, 1.0), table=(0.0, 1.0)):
    """Signal of a two-segment relative colour path."""
    path = darsia.ColorPath(colors=KNOTS, base_color=np.zeros(3), mode="rgb")
    return darsia.ColorPathSignal(
        darsia.ColorPathInterpolation(path, darsia.ColorMode.RELATIVE, list(values)),
        darsia.PWTransformation(list(supports), list(table)),
    )


def test_a_change_beyond_the_path_end_goes_on_along_the_last_segment():
    beyond = KNOTS[2] + 0.5 * (KNOTS[2] - KNOTS[1])
    colors = np.array([KNOTS[1] / 2, KNOTS[2], beyond, -KNOTS[1]], np.float32)

    identity = _signal()(colors)
    assert identity.dtype == np.float32
    np.testing.assert_allclose(identity, [0.25, 1.0, 1.25, 0.0], rtol=1e-6)

    # Beyond its last support the signal function goes on along its last segment.
    table = _signal(supports=(0.0, 0.5, 1.0), table=(0.0, 0.4, 1.4))(colors)
    np.testing.assert_allclose(table, [0.2, 1.4, 1.9, 0.0], rtol=1e-6)


def test_a_path_of_one_segment_goes_on_beyond_its_end():
    end = KNOTS[2]
    path = darsia.ColorPath(colors=[KNOTS[0], end], base_color=np.zeros(3), mode="rgb")
    interpolation = darsia.ColorPathInterpolation(
        path, darsia.ColorMode.RELATIVE, [0.0, 1.0]
    )
    signal = darsia.ColorPathSignal(
        interpolation, darsia.PWTransformation([0.0, 1.0], [0.0, 1.0])
    )
    colors = np.array([end / 2, end, 2 * end, 3 * end, -end / 2], np.float32)

    values = signal(colors)

    np.testing.assert_allclose(values, [0.5, 1.0, 2.0, 3.0, 0.0], rtol=1e-6)
    # Up to its end the value is the path's own interpretation, bit for bit.
    np.testing.assert_array_equal(values[:2], interpolation(colors[:2]))


def test_each_sand_is_read_on_its_own_path_inside_the_dissolved_co2():
    rng = np.random.default_rng(0)
    facies = np.repeat(np.array([[1, 2, 7]]), 40, axis=0).repeat(20, axis=1)
    baseline = rng.uniform(0.2, 0.4, (*facies.shape, 3)).astype(np.float32)
    change = rng.uniform(0, 1.6, facies.shape)[..., None] * KNOTS[2]
    current = (baseline + change).astype(np.float32)
    support = np.ones(facies.shape, bool)
    support[0] = False
    current[0] = np.nan
    paths = {1: _signal(), 2: _signal(values=(0.0, 0.2, 1.0))}
    reading = darsia.SandColorPaths(paths, {1: 1.5, 2: 0.5, 7: 9.0})

    dissolved = rng.random(facies.shape) < 0.3
    signal = reading(current, baseline, support, dissolved, facies)
    relative = current - baseline
    for sand, factor in ((1, 1.5), (2, 0.5)):
        mask = dissolved & support & (facies == sand)
        np.testing.assert_allclose(
            signal[mask], factor * paths[sand](relative[mask]), rtol=1e-6
        )
    assert not signal[~dissolved | (facies == 7)].any()
    assert not signal[0].any()

    # A pixel's signal does not depend on which other pixels are dissolved.
    more = reading(current, baseline, support, dissolved | ~dissolved, facies)
    np.testing.assert_array_equal(more[dissolved], signal[dissolved])
