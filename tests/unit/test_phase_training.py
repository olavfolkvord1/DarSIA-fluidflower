"""Unit tests for the phase-model training on synthetic labelled frames."""

import dataclasses
import tomllib

import numpy as np

import darsia

SETTINGS = darsia.PhaseTrainingSettings(
    **tomllib.loads(
        """
        # The values the training of our phase model ran with in production; its
        # texture scale was 2 pixels of the training grid, here of 0.5 mm.
        [phase_training]
        pixels_per_region = 1024
        texture_sigma_mm = 1.0
        """
    )["phase_training"]
)


def _frame(seed, shape=(120, 160)):
    """Baseline and frame of a run: aqueous CO2 in a plume, free gas at its top."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    baseline = rng.uniform(0.9, 1.1, (*shape, 3)) * [0.3, 0.25, 0.18]
    plume = ((rows - 100) / 55.0) ** 2 + ((cols - 80) / 45.0) ** 2 < 1
    gas = plume & (rows < 70) & (np.abs(cols - 80) < 25)
    change = np.where(plume[..., None], [1.15, 0.9, 0.6], 1.0)
    change = np.where(gas[..., None], [1.5, 1.4, 1.2], change)
    current = baseline * change + rng.normal(0, 0.004, baseline.shape)
    return current, baseline, plume, gas


def test_frame_samples_take_supported_pixels_of_each_region():
    """A region gives its supported pixels, at most the set number, chosen by name."""
    current, baseline, _, _ = _frame(0)
    current[40, 40, 0] = np.nan
    settings = dataclasses.replace(SETTINGS, pixels_per_region=50)
    masks = [np.zeros(current.shape[:2], bool) for _ in range(4)]
    masks[0][38:43, 38:43] = True  # unsupported: inside the invalid pixel's window
    masks[1][10:15, 10:15] = True  # 25 pixels
    masks[2][60:80, 60:80] = True  # 400 pixels
    masks[3][60:80, 60:80] = True
    regions = [darsia.LabelledRegion(f"r{i}", "a", "water", 1.0) for i in range(4)]
    samples = darsia.frame_samples(
        current, baseline, None, regions, masks, settings=settings, pixel_size_mm=0.5
    )
    assert [region.name for region in samples.regions] == ["r1", "r2", "r3"]
    assert np.bincount(samples.owners).tolist() == [25, 50, 50]
    again = darsia.frame_samples(
        current, baseline, None, regions, masks, settings=settings, pixel_size_mm=0.5
    )
    np.testing.assert_array_equal(samples.features, again.features)
    assert not np.array_equal(samples.features[25:75], samples.features[75:])
    # A sample is the row of its pixel in the features of the whole frame.
    features = darsia.PhaseFeatures(
        current, baseline, texture_sigma_mm=1.0, pixel_size_mm=0.5
    )
    whole = features.matrix(np.s_[:]).reshape(*current.shape[:2], -1)
    np.testing.assert_array_equal(
        samples.features[:25], whole[10:15, 10:15].reshape(25, -1)
    )
