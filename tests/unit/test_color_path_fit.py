"""Each sand's colour path fitted on the colour changes of a run's photographs."""

import tomllib

import numpy as np
import pytest

import darsia

SETTINGS = darsia.ColorPathFitSettings(
    **tomllib.loads(
        """
        # The settings our production fitted the colour paths with.
        sands = [1, 2]
        stride_pixels = 8
        minimum_samples = 16
        fractions = [0.25, 0.55, 0.85, 0.98]
        band = 0.075
        minimum_span = 1e-8
        minimum_step = 1e-9
        """
    ),
)
DIRECTION = np.array([0.6, 0.8, 0.0], np.float32)
"""The direction of the synthetic colour changes."""


def test_a_frame_is_sampled_on_a_grid_in_each_sand():
    """Every eighth pixel of rows and columns, where frame and baseline are
    finite, as the frame minus the baseline."""
    current = np.full((40, 48, 3), 0.5, np.float32)
    baseline = np.full((40, 48, 3), 0.2, np.float32)
    facies = np.full((40, 48), 1)
    facies[24:] = 2
    current[8, 16] = np.nan
    baseline[32, 0] = np.inf
    found = darsia.sample_color_changes(current, baseline, facies, [1, 2], 8)
    assert len(found[1]) == 3 * 6 - 1 and len(found[2]) == 2 * 6 - 1
    np.testing.assert_allclose(found[1], 0.3, rtol=1e-6)
    assert found[1].dtype == np.float32


def test_the_key_colours_are_medians_of_ranked_bands():
    """Colour changes along one direction with uniformly spread lengths: each key
    colour lies at its fraction of the longest change, from zero."""
    rng = np.random.default_rng(4)
    lengths = rng.uniform(0.0, 1.0, (2, 5000)).astype(np.float32)
    frames = [length[:, None] * DIRECTION for length in lengths]
    knots = darsia.fit_color_path(frames, SETTINGS)
    assert knots.dtype == np.float32 and knots.shape == (5, 3)
    np.testing.assert_array_equal(knots[0], 0.0)
    np.testing.assert_allclose(
        knots[1:], np.outer([0.25, 0.55, 0.85, 0.9525], DIRECTION), atol=0.01
    )


def test_too_few_or_no_colour_changes_give_no_path():
    """Too few samples, or changes of no length, give no path."""
    few = [np.ones((15, 3), np.float32)]
    assert darsia.fit_color_path(few, SETTINGS) is None
    assert darsia.fit_color_path([np.zeros((100, 3), np.float32)], SETTINGS) is None


def test_a_runs_paths_over_its_frames():
    """A run's frames are sampled and pooled per sand; a sand without changes
    has no path, and every sand its count."""
    baseline = np.full((64, 64, 3), 0.2, np.float32)
    facies = np.full((64, 64), 1)
    facies[32:] = 2
    rng = np.random.default_rng(6)
    frames = []
    for _ in range(3):
        image = baseline.copy()
        image[:32] += rng.uniform(0, 0.1, (32, 64, 1)).astype(np.float32) * DIRECTION
        frames.append(image)
    paths, samples = darsia.fit_run_color_paths(
        iter(frames), baseline, facies, SETTINGS
    )
    assert set(paths) == {1} and samples == {1: 3 * 4 * 8, 2: 3 * 4 * 8}


def test_the_runs_share_one_scale_per_sand():
    """The scale of a sand is its longest path; each value is the distance along
    the path over it, and the signal function is the identity."""
    short = np.array([[0, 0, 0], [0.3, 0.4, 0], [0.6, 0.8, 0]], np.float32)
    long = np.array([[0, 0, 0], [0, 2, 0]], np.float32)
    readings = darsia.color_path_readings({"a": {1: short}, "b": {1: long}})
    assert readings["a"][1]["scale"] == readings["b"][1]["scale"] == 2.0
    np.testing.assert_allclose(readings["a"][1]["values"], [0, 0.25, 0.5], rtol=1e-6)
    assert readings["b"][1]["values"] == [0.0, 1.0]
    assert readings["a"][1]["supports"] == readings["a"][1]["table"] == [0.0, 1.0]
    assert readings["a"][1]["colors"] == short.tolist()
    with pytest.raises(ValueError, match="no length"):
        darsia.color_path_readings({"a": {1: np.zeros((2, 3), np.float32)}})


def test_distances_along_a_path_keep_its_precision():
    """Cumulative distances from zero, in the key colours' precision."""
    knots = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 12]], np.float32)
    distances = darsia.path_distances(knots)
    assert distances.dtype == np.float32
    np.testing.assert_array_equal(distances, [0, 5, 17])
