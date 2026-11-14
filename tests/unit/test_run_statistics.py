"""Unit tests for the run statistics on synthetic frames."""

import tomllib

import numpy as np
import pytest

import darsia

W, D, G, U = darsia.PhaseLabel

SETTINGS = darsia.RunStatisticsSettings(
    **tomllib.loads(
        """
        # The values the statistics ran with in production.
        [run_statistics]
        mean_mm = 5.0
        zero_signal = 0.0
        saturation_signal = 1.2
        intervals = 32768
        """
    )["run_statistics"]
)

SHAPE = (60, 80)
"""Rows and columns of a 0.5 mm grid."""
PITCH = darsia.optical_pitch_m(
    darsia.ScalarImage(np.full(SHAPE, 0.0005**2 * 1.0337), dimensions=[0.03, 0.04]),
    np.ones(SHAPE, bool),
)
"""Its pitch on the sand plane: the mean's square is 10 pixels wide."""


class Signal:
    """A sand reading giving each sand's dissolved CO2 a set signal map."""

    sands = [1, 2]

    def __init__(self, maps):
        self.maps = maps

    def __call__(self, current, baseline, support, dissolved, facies):
        signal = np.zeros(facies.shape, current.dtype)
        for sand in self.sands:
            mask = dissolved & (facies == sand)
            signal[mask] = self.maps[sand].astype(current.dtype)[mask]
        return signal


def _sand_map():
    """Sand 1 on the left half, sand 2 on the right half."""
    facies = np.ones(SHAPE, np.uint8)
    facies[:, 40:] = 2
    return darsia.ScalarImage(facies, dimensions=[0.03, 0.04])


def test_the_mean_stays_inside_each_sand_and_its_dissolved_co2():
    sand_map = _sand_map()
    rev = darsia.REV(size=SETTINGS.mean_mm / 1000, img=sand_map)
    grains = np.indices(SHAPE).sum(0) % 2  # a checkerboard of grains
    signal = np.where(sand_map.img == 1, 0.2 + 0.2 * grains, 0.9).astype(np.float32)
    dissolved = np.zeros(SHAPE, bool)
    dissolved[10:50, 10:70] = True
    masks = {sand: sand_map.img == sand for sand in (1, 2)}

    mean = darsia.sand_mean(signal, dissolved, masks, rev)

    assert mean.dtype == np.float32
    inner = dissolved & (sand_map.img == 1)
    np.testing.assert_allclose(mean[inner], 0.3, atol=0.011)  # the grains leave
    np.testing.assert_allclose(mean[dissolved & (sand_map.img == 2)], 0.9, rtol=1e-6)
    np.testing.assert_array_equal(mean[~dissolved], signal[~dissolved])


def _run(dtype):
    """Three frames whose dissolved CO2 grows; sand 2 beyond saturation."""
    frames = []
    for index in range(3):
        phase = np.full(SHAPE, W, np.uint8)
        phase[: 20 + 10 * index] = D
        phase[:4] = G
        phase[30:34, 0:8] = U
        rules = np.zeros(SHAPE, np.uint8)
        rules[10:14, 20:30] = darsia.PhaseRule.GRAVITY  # the model's gas
        support = np.ones(SHAPE, bool)
        support[25, 20] = False
        weight = np.linspace(1, 2, SHAPE[1], dtype=dtype)[None, :].repeat(SHAPE[0], 0)
        frames.append((np.ones((*SHAPE, 3), dtype), phase, rules, support, weight))
    counted = np.ones(SHAPE, bool)
    counted[:, 70:] = False
    maps = {
        1: np.full(SHAPE, 0.6 + 0.1 * np.arange(SHAPE[0])[:, None] / 60),
        2: np.full(SHAPE, 1.5),
    }
    return frames, counted, Signal(maps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_statistics_of_a_synthetic_run(dtype):
    frames, counted, reading = _run(dtype)
    baseline = np.zeros((*SHAPE, 3), dtype)
    sand_map = _sand_map()

    run = list(
        darsia.run_statistics(
            frames, baseline, sand_map, PITCH, counted, reading, SETTINGS
        )
    )

    assert len(run) == 3
    for (_, phase, rules, support, weight), statistics in zip(frames, run):
        gas = darsia.model_gas(phase, rules)
        assert (gas & (phase == D)).any()
        for sand in (1, 2):
            pixels = support & (phase == D) & ~gas & (sand_map.img == sand) & counted
            histogram = statistics[sand]
            assert histogram.weight.dtype == dtype
            np.testing.assert_allclose(
                histogram.weight.sum(), weight[pixels].sum(), rtol=1e-5
            )
            assert np.all(np.diff(histogram.relative) > 0)
    # Beyond saturation the relative signal is one; below it, the signal over 1.2.
    np.testing.assert_array_equal(run[0][2].relative, [1.0])
    assert run[2][1].relative.min() >= 0.6 / 1.2 - 1e-3
    assert run[2][1].relative.max() <= (0.6 + 0.1 * 39 / 60) / 1.2 + 1e-3


def test_a_frames_histograms_take_its_dissolved_pixels_that_count():
    """A frame's histograms of a signal read elsewhere take each sand's dissolved
    pixels that count, whatever the signal holds on the other pixels."""
    facies = _sand_map().img
    signal = np.full(SHAPE, 0.3, np.float32)  # a quarter of the curve
    signal[:, 40:] = 0.9  # three quarters in sand 2
    signal[:10] = 5.0  # beyond saturation, but not dissolved
    dissolved = np.zeros(SHAPE, bool)
    dissolved[10:40] = True
    counted = np.ones(SHAPE, bool)
    counted[:, 70:] = False
    weight = np.full(SHAPE, 0.5, np.float32)

    found = darsia.signal_histograms(
        signal, dissolved, facies, counted, weight, (1, 2), SETTINGS
    )

    assert set(found) == {1, 2}
    np.testing.assert_allclose(found[1].relative, [0.25])
    np.testing.assert_allclose(found[2].relative, [0.75])
    np.testing.assert_array_equal(found[1].weight, [np.float32(0.5 * 30 * 40)])
    np.testing.assert_array_equal(found[2].weight, [np.float32(0.5 * 30 * 30)])


def test_a_run_in_chunks_equals_the_run_whole():
    frames, counted, reading = _run(np.float32)
    inputs = (
        np.zeros((*SHAPE, 3), np.float32),
        _sand_map(),
        PITCH,
        counted,
        reading,
        SETTINGS,
    )

    whole = list(darsia.run_statistics(frames, *inputs))
    chunks = [
        statistics
        for chunk in (frames[:1], frames[1:])
        for statistics in darsia.run_statistics(chunk, *inputs)
    ]

    for one, other in zip(whole, chunks, strict=True):
        for sand in (1, 2):
            np.testing.assert_array_equal(one[sand].relative, other[sand].relative)
            np.testing.assert_array_equal(one[sand].weight, other[sand].weight)
