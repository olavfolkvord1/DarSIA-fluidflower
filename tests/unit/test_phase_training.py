"""Unit tests for the phase-model training on synthetic labelled frames."""

import dataclasses
import json
import tomllib

import numpy as np
from scipy.ndimage import binary_erosion

import darsia

SETTINGS = darsia.PhaseTrainingSettings(
    **tomllib.loads(
        """
        # The values the training of our phase model ran with in production; its
        # texture scale was 2 pixels of the training grid, here of 0.5 mm.
        [phase_training]
        pixels_per_region = 1024
        texture_sigma_mm = 1.0
        regularization_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        weak_strength = 0.35
        positive_top_fraction = 0.10
        mil_iterations = 160
        mil_learning_rate = 0.035
        region_score_quantile = 0.90
        threshold_levels = [0.02, 0.98, 97]
        irls_steps = [40, 120]
        irls_tolerance = 1e-5
        mil_tolerance = 2e-5
        mil_stable_steps = 8
        """
    )["phase_training"]
)
PIXEL_MM = 0.5
BOXES = {
    "water": np.s_[5:25, 5:35],
    "co2_absent_region": np.s_[5:25, 120:155],
    "aqueous": np.s_[80:100, 50:75],
    "gas": np.s_[52:62, 72:88],
    "mixed_gas_aqueous": np.s_[60:80, 60:100],
}
"""Labelled boxes of each run's frame, on the 0.5 mm grid."""


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


def _samples(runs=("a", "b", "c")):
    """Joined training samples of one labelled frame per run."""
    parts = []
    for seed, run in enumerate(runs):
        current, baseline, _, _ = _frame(seed)
        regions = [darsia.LabelledRegion(f"{run}:{p}", run, p, 1.0) for p in BOXES]
        masks = [np.zeros(current.shape[:2], bool) for _ in BOXES]
        for mask, box in zip(masks, BOXES.values()):
            mask[box] = True
        parts.append(
            darsia.frame_samples(
                current.astype(np.float32),
                baseline.astype(np.float32),
                None,
                regions,
                masks,
                settings=SETTINGS,
                pixel_size_mm=PIXEL_MM,
            )
        )
    return darsia.join_samples(parts)


def _phases(training, current, baseline, pixel_size_mm):
    """Water 0, aqueous CO2 1 and free gas 2 at the trained thresholds."""
    co2, gas = darsia.phase_probabilities(
        training.model, current, baseline, pixel_size_mm=pixel_size_mm
    )
    co2 = (co2 >= training.co2_threshold).astype(int)
    return co2 + co2 * (gas >= training.model.gas_threshold)


def _inner(phases, margin):
    """Pixels at least ``margin`` pixels away from another phase."""
    return np.any(
        [binary_erosion(phases == v, iterations=margin) for v in (0, 1, 2)], 0
    )


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


def test_train_phase_model_separates_the_labelled_phases():
    """Water, aqueous CO2 and free gas of a new frame are classified 4 mm away
    from their boundaries, in the dtype of the samples."""
    training = darsia.train_phase_model(_samples(), SETTINGS)
    assert training.model.co2_presence.coefficients.dtype == np.float32
    for head in ("co2_presence", "gas_within_co2"):
        summary = training.cross_validation[head]
        assert summary["runs"] == ["a", "b", "c"]
        assert summary["selected_regularization"] in SETTINGS.regularization_grid
    current, baseline, plume, gas = _frame(7)
    truth = plume.astype(int) + gas
    phases = _phases(
        training, *(i.astype(np.float32) for i in (current, baseline)), 0.5
    )
    inner = _inner(truth, 8)
    assert all((truth[inner] == v).sum() > 100 for v in (0, 1, 2))
    np.testing.assert_array_equal(phases[inner], truth[inner])


def test_trained_folder_reads_back_with_its_sizes_in_mm(tmp_path):
    """Samples and model are separate results; the folder needs no pixel size."""
    darsia.write_training_samples(tmp_path / "samples.npz", _samples())
    samples = darsia.read_training_samples(tmp_path / "samples.npz")
    trained = darsia.train_phase_model(samples, SETTINGS)
    darsia.write_phase_model(
        tmp_path / "model",
        trained.model,
        pixel_size_mm=samples.pixel_size_mm,
        co2_threshold=trained.co2_threshold,
        cross_validation=trained.cross_validation,
    )
    model = darsia.read_phase_model(tmp_path / "model")
    assert model.texture_sigma_mm == SETTINGS.texture_sigma_mm
    assert model.gas_threshold == trained.model.gas_threshold
    for head in ("co2_presence", "gas_within_co2"):
        np.testing.assert_array_equal(
            getattr(model, head).coefficients, getattr(trained.model, head).coefficients
        )
        assert getattr(model, head).intercept == getattr(trained.model, head).intercept
    policy = json.loads((tmp_path / "model" / "phase_decision_policy.json").read_text())
    assert policy["models"]["generic"]["co2_presence"]["presence_threshold"] == (
        trained.co2_threshold
    )


def test_a_model_trained_at_one_pixel_size_classifies_alike_at_another():
    """On a grid of twice the pixel size the texture window keeps its width in mm,
    and the phases 4 mm away from their boundaries are those of the training grid."""
    training = darsia.train_phase_model(_samples(), SETTINGS)
    current, baseline, plume, gas = _frame(7)
    coarse = [i.reshape(60, 2, 80, 2, 3).mean(axis=(1, 3)) for i in (current, baseline)]
    for images, pixel_size_mm in (((current, baseline), 0.5), (coarse, 1.0)):
        invalid = images[0].copy()
        invalid[20, 20, 1] = np.nan
        co2, _ = darsia.phase_probabilities(
            training.model, invalid, images[1], pixel_size_mm=pixel_size_mm
        )
        half_width = round(4 * SETTINGS.texture_sigma_mm / pixel_size_mm)
        assert np.isnan(co2).sum() == (2 * half_width + 1) ** 2
    fine = _phases(training, current, baseline, 0.5)[::2, ::2]
    phases = _phases(training, *coarse, 1.0)
    inner = _inner((plume.astype(int) + gas)[::2, ::2], 4)
    np.testing.assert_array_equal(phases[inner], fine[inner])
