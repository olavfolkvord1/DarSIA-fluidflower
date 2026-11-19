"""The proposal of the light correction's reference run: the medoid of the runs'
optical corrections, from their stable sand along the frame and their chart."""

import dataclasses
import tomllib

import numpy as np
import pytest

from darsia.corrections.color.reference_proposal import (
    ReferenceProposalSettings,
    ReferenceSamples,
    frame_facing_sand_boundary,
    propose_reference,
    reference_samples,
    stable_band_mask,
)

SETTINGS = ReferenceProposalSettings(
    **tomllib.loads(
        """
        # Production's settings, but the band for the small images.
        stable_band_mm = 2.0
        use_lighting = true
        use_color = true
        grid_shape = [12, 12]
        minimum_stable_cells = 3
        minimum_chart_patches = 6
        """
    )
)
"""The settings of the tests, both parts used."""


def samples(run: str, light: float, cast: float = 0.0) -> ReferenceSamples:
    """A run whose stable cells are a grey of ``light`` and whose chart's patches
    are greys of ``light`` with a red cast."""
    cells = np.full((9, 3), light)
    chart = np.repeat(np.linspace(0.2, 0.8, 8)[:, None], 3, axis=1) * light / 0.5
    chart[:, 0] *= np.exp(cast)
    return ReferenceSamples(run, cells, np.ones(9, bool), np.clip(chart, 0.01, 0.99))


def test_the_run_with_the_least_mean_correction_is_proposed():
    """The middle of three exposures is the medoid; the ranking holds each run's
    mean, largest, light and colour corrections; a colour cast counts too."""
    found = propose_reference(
        [samples("dark", 0.3), samples("middle", 0.4), samples("bright", 0.5)],
        SETTINGS,
    )
    assert found["reference_run"] == "middle" and found["selected"] is False
    first = found["ranking"][0]
    expected = np.log(0.4 / 0.3)
    assert first["mean_light_correction"] == pytest.approx(
        (expected + np.log(0.5 / 0.4)) / 2
    )
    assert set(first) == {
        "run",
        "mean_log_correction",
        "maximum_log_correction",
        "mean_light_correction",
        "mean_colour_correction",
    }
    cast = propose_reference(
        [samples("a", 0.4), samples("b", 0.4, 0.3), samples("c", 0.4, 0.3)],
        dataclasses.replace(SETTINGS, use_lighting=False),
    )
    assert cast["reference_run"] == "b"
    assert cast["ranking"][-1]["run"] == "a"


def test_equal_corrections_are_ranked_by_the_largest_then_by_name():
    """Runs equally far from each other are ranked by name."""
    found = propose_reference([samples("b", 0.4), samples("a", 0.4)], SETTINGS)
    assert [r["run"] for r in found["ranking"]] == ["a", "b"]


def test_too_few_stable_cells_or_patches_are_refused():
    """A run with too few stable cells, or without a chart where colour counts,
    is refused, naming the run."""
    poor = samples("poor", 0.4)
    poor = ReferenceSamples("poor", poor.stable_rgb, np.eye(9, dtype=bool)[0], None)
    with pytest.raises(ValueError, match="poor"):
        propose_reference([samples("a", 0.4), poor], SETTINGS)
    with pytest.raises(ValueError, match="ColorChecker"):
        propose_reference(
            [
                samples("a", 0.4),
                ReferenceSamples("b", poor.stable_rgb, np.ones(9, bool)),
            ],
            SETTINGS,
        )


def test_the_stable_band_lies_along_the_frame_not_along_water():
    """The stable sand is within the band of the sand facing the frame; sand next
    to mapped water is not stable there."""
    sand = np.zeros((20, 20), bool)
    sand[5:20, :] = True
    water = np.zeros_like(sand)
    water[0:5, :] = True
    facing = frame_facing_sand_boundary(sand, water)
    assert not facing[5, 10] and facing[19, 10] and facing[10, 0]
    band = stable_band_mask(sand, 2.0, (1.0, 1.0), facing)
    assert band[19, 10] and band[10, 1] and not band[10, 10] and not band[5, 10]


def test_a_runs_samples_are_the_medians_of_its_stable_cells():
    """Each coarse cell holds the median of its usable stable pixels; cells
    without any stay empty, at their index."""
    image = np.full((4, 4, 3), 0.5)
    image[0, 0] = 0.9
    image[3, 3] = 1.0
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True
    mask[3, 3] = True
    found = reference_samples("a", image, mask, None, (2, 2))
    assert found.stable_valid.tolist() == [True, False, False, False]
    assert np.allclose(found.stable_rgb[0], 0.5)
