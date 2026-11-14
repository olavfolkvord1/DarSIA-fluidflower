"""Unit tests for reading colour changes along colour paths without a cap."""

import dataclasses
import tomllib

import numpy as np
import pytest

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


RANGES = darsia.ColorRangeSettings(
    **tomllib.loads(
        """
        # Our production's recipe (31, 16 and 30 pixels; at least 3,000 and 500
        # pixels) on the sand plane of the two runs it measured, pixels of
        # 0.0279495 and 0.0279564 mm²: 2,999.9 and 2,999.1 pixels inside, 499.47
        # and 499.35 beside the gas, so that a count passes as in production.
        [color_ranges]
        mean_mm = 5.18
        inner_mm = 2.67
        gas_mm = 5.0
        minimum_inner_mm2 = 83.845
        minimum_gas_mm2 = 13.96
        gas_percentile = 90.0
        """
    )["color_ranges"]
)


def test_the_colour_range_is_the_change_beside_the_free_gas():
    shape = (80, 100)  # 0.5 mm pixels
    sand_map = darsia.ScalarImage(np.ones(shape, np.uint8), dimensions=[0.04, 0.05])
    baseline = np.full((*shape, 3), 0.3)
    phase = np.zeros(shape, np.uint8)
    phase[10:70, 10:90] = darsia.PhaseLabel.DISSOLVED
    phase[36:44, 46:54] = darsia.PhaseLabel.GAS
    support = np.ones(shape, bool)
    end = KNOTS[2]

    rules = np.zeros(shape, np.uint8)

    def run(*reached, phase=phase, rules=rules):
        frames = [(baseline + t * end, phase, rules, support) for t in reached]
        return frames, baseline, sand_map, (0.5e-3, 0.5e-3), {1: _signal()}

    ranges = darsia.color_ranges([run(0.5, 1.0), run(1.4, 0.7)], RANGES)

    np.testing.assert_allclose(ranges[1], 1.2 * np.linalg.norm(end), rtol=1e-6)

    # The model's gas that a peel made dissolved CO2 is free gas here too.
    gas = phase == darsia.PhaseLabel.GAS
    peeled = np.where(gas, darsia.PhaseLabel.DISSOLVED, phase).astype(np.uint8)
    rule = np.where(gas, darsia.PhaseRule.PEELED, 0).astype(np.uint8)
    model = [run(*t, phase=peeled, rules=rule) for t in ((0.5, 1.0), (1.4, 0.7))]
    assert darsia.color_ranges(model, RANGES) == ranges

    # A length that rounds to zero pixels takes no step (SciPy repeats a zero until
    # nothing changes): no erosion, and no pixel beside the free gas.
    no_depth = darsia.color_ranges(
        [run(0.5, 1.0), run(1.4, 0.7)], dataclasses.replace(RANGES, inner_mm=0.2)
    )
    np.testing.assert_allclose(no_depth[1], ranges[1], rtol=1e-6)
    with pytest.raises(ValueError, match="No positive colour range"):
        darsia.color_ranges([run(1.0)], dataclasses.replace(RANGES, gas_mm=0.2))
