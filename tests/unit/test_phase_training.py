"""Unit tests for the phase-model training on synthetic labelled frames."""

import dataclasses
import json
import shutil
import tomllib

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

import darsia
from darsia.presets.workflows.config.corrections import CorrectionsConfig
from darsia.presets.workflows.rig import Rig

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
BULGE = (
    "[corrections.curvature.bulge]\nhorizontal_bulge = {}\nvertical_bulge = 0.0\n"
    "horizontal_center_offset = 0\nvertical_center_offset = 0\n"
)
"""A curvature correction's bulge, with its horizontal bulge to be given."""


def _rig_canvas(folder, settings="", shape=(120, 160), origin=None):
    """Canvas of a rig that maps its photographs of 120 x 160 pixels onto a grid
    of the shape over 0.06 x 0.08 m, at the origin, with further settings."""
    folder.mkdir()
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    darsia.ImageMappingCorrection(
        cols * (160 / shape[1]),
        rows * (120 / shape[0]),
        source_shape=(120, 160),
        dimensions=[0.06, 0.08],
        origin=origin,
    ).save(folder / "grid.npz")
    darsia.OpticalImage(np.full((120, 160, 3), 0.3), color_space="RGB").save(
        folder / "baseline.npz"
    )
    (folder / "config.toml").write_text(
        f'[corrections.mapping]\npath = "{(folder / "grid.npz").as_posix()}"\n'
        + settings
    )
    rig = Rig()
    rig.setup_reading(
        folder / "baseline.npz",
        experiment=None,
        corrections_config=CorrectionsConfig().load(folder / "config.toml"),
    )
    return rig.canvas


@pytest.fixture(scope="module")
def canvas(tmp_path_factory):
    """Canvas of the rig whose frames are labelled."""
    return _rig_canvas(tmp_path_factory.mktemp("canvas") / "rig")


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


def _samples(canvas, runs=("a", "b", "c")):
    """Joined training samples of one labelled frame per run, on the canvas."""
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
                darsia.LabelSet(regions, masks, canvas),
                settings=SETTINGS,
                pixel_size_mm=PIXEL_MM,
                canvas=canvas,
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


def test_frame_samples_take_supported_pixels_of_each_region(canvas):
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
    labels = darsia.LabelSet(regions, masks, canvas)
    options = {"settings": settings, "pixel_size_mm": 0.5, "canvas": canvas}
    samples = darsia.frame_samples(current, baseline, None, labels, **options)
    assert [region.name for region in samples.regions] == ["r1", "r2", "r3"]
    assert np.bincount(samples.owners).tolist() == [25, 50, 50]
    again = darsia.frame_samples(current, baseline, None, labels, **options)
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


def test_train_phase_model_separates_the_labelled_phases(canvas):
    """Water, aqueous CO2 and free gas of a new frame are classified 4 mm away
    from their boundaries, in the dtype of the samples."""
    training = darsia.train_phase_model(_samples(canvas), SETTINGS)
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


def test_trained_folder_reads_back_with_its_sizes_in_mm(tmp_path, canvas):
    """Samples and model are separate results; the folder needs no pixel size and
    records the canvas."""
    darsia.write_training_samples(tmp_path / "samples.npz", _samples(canvas))
    samples = darsia.read_training_samples(tmp_path / "samples.npz")
    trained = darsia.train_phase_model(samples, SETTINGS)
    darsia.write_phase_model(
        tmp_path / "model",
        trained.model,
        pixel_size_mm=samples.pixel_size_mm,
        co2_threshold=trained.co2_threshold,
        cross_validation=trained.cross_validation,
        accepted_canvases=samples.accepted_canvases,
        moved_from=samples.moved_from,
    )
    model = darsia.read_phase_model(tmp_path / "model", canvas=canvas)
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
    canvases = json.loads((tmp_path / "model" / "canvas.json").read_text())
    assert canvases == {
        "canvas": canvas,
        "accepted_label_canvases": [],
        "labels_moved_from": [],
    }


def test_a_model_trained_at_one_pixel_size_classifies_alike_at_another(
    tmp_path, canvas
):
    """Read for the rig's canvas at twice the pixel size, or for whole metric cells
    that extend a canvas differently, also one whose origin is not at its height,
    the model passes without an explicit choice. It keeps its texture window's
    width in mm, and the phases 4 mm away from their boundaries are those of the
    training grid."""
    training = darsia.train_phase_model(_samples(canvas), SETTINGS)
    darsia.write_phase_model(
        tmp_path / "model",
        training.model,
        pixel_size_mm=PIXEL_MM,
        co2_threshold=training.co2_threshold,
        cross_validation=training.cross_validation,
        accepted_canvases=(),
        moved_from=(),
    )
    on_coarse = _rig_canvas(tmp_path / "coarse", shape=(60, 80))
    model = darsia.read_phase_model(tmp_path / "model", canvas=on_coarse)
    cells = "[corrections.grid]\npixel_size_mm = {}\narea_factor = 1.0\n"
    first, second = (
        _rig_canvas(tmp_path / f"cells{size}", cells.format(size), origin=[0.0, 0.5])
        for size in (0.7, 1.1)
    )
    assert first["grid"]["dimensions"] != second["grid"]["dimensions"]
    assert first["grid"]["extent"] == second["grid"]["extent"]
    assert not darsia.check_canvas(
        first, second, accept=False, subject="the model", resolution=True
    )
    current, baseline, plume, gas = _frame(7)
    coarse = [i.reshape(60, 2, 80, 2, 3).mean(axis=(1, 3)) for i in (current, baseline)]
    for images, pixel_size_mm in (((current, baseline), 0.5), (coarse, 1.0)):
        invalid = images[0].copy()
        invalid[20, 20, 1] = np.nan
        co2, _ = darsia.phase_probabilities(
            model, invalid, images[1], pixel_size_mm=pixel_size_mm
        )
        half_width = round(4 * SETTINGS.texture_sigma_mm / pixel_size_mm)
        assert np.isnan(co2).sum() == (2 * half_width + 1) ** 2
    fine = _phases(training, current, baseline, 0.5)[::2, ::2]
    phases = _phases(training, *coarse, 1.0)
    inner = _inner((plume.astype(int) + gas)[::2, ::2], 4)
    np.testing.assert_array_equal(phases[inner], fine[inner])


def test_other_dimensions_without_a_recorded_extent_are_another_canvas():
    """Canvases that record no extent before whole metric cells pass with other
    dimensions and another origin only by explicit choice."""
    record = {
        "source": {"format": "NPZ"},
        "corrections": {"mapping": {"path": "mapping.npz"}},
        "grid": {"shape": [4, 5], "dimensions": [0.4, 0.5], "origin": [0.0, 0.4]},
    }
    other = {
        **record,
        "grid": {**record["grid"], "dimensions": [0.8, 1.0], "origin": [0.0, 0.8]},
    }
    check = dict(subject="the model", resolution=True)

    with pytest.raises(
        ValueError, match=r"settings: grid\.dimensions, grid\.origin\. "
    ):
        darsia.check_canvas(record, other, accept=False, **check)
    assert darsia.check_canvas(record, other, accept=True, **check)


def test_labels_on_another_canvas_are_sampled_only_by_explicit_choice(tmp_path):
    """A changed curvature parameter is named and refused, and so is a label set
    without a canvas; accepting them is recorded with the samples."""
    rig, drawn = (
        _rig_canvas(tmp_path / name, BULGE.format(bulge))
        for name, bulge in (("rig", 1e-7), ("drawn", 2e-7))
    )
    current, baseline, _, _ = _frame(0)
    mask = np.zeros(current.shape[:2], bool)
    mask[BOXES["water"]] = True
    regions = [darsia.LabelledRegion("a:water", "a", "water", 1.0)]

    def sample(labels_canvas, accept=False):
        """Samples of the water box drawn on a canvas, from the rig's images."""
        return darsia.frame_samples(
            current,
            baseline,
            None,
            darsia.LabelSet(regions, [mask], labels_canvas),
            settings=SETTINGS,
            pixel_size_mm=PIXEL_MM,
            canvas=rig,
            accept_canvas=accept,
        )

    bulge = r"settings: corrections\.curvature\.bulge\.horizontal_bulge\. "
    with pytest.raises(ValueError, match=bulge):
        sample(drawn)
    with pytest.raises(ValueError, match=r"settings: canvas\. "):
        sample(None)
    parts = [sample(rig), sample(drawn, accept=True), sample(None, accept=True)]
    darsia.write_training_samples(tmp_path / "samples.npz", darsia.join_samples(parts))
    samples = darsia.read_training_samples(tmp_path / "samples.npz")
    assert parts[0].accepted_canvases == ()
    assert samples.canvas == rig
    assert samples.accepted_canvases == (drawn, None)


def test_a_crop_in_pixels_gives_no_grid_in_rig_metres(tmp_path, canvas):
    """A crop whose width and height are pixels leaves the images without rig
    metres: its canvas has no grid, and labels are not moved onto it."""
    crop = (
        "[corrections.curvature.crop]\ntop_left = [0, 0]\nbottom_left = [119, 0]\n"
        "bottom_right = [119, 159]\ntop_right = [0, 159]\nwidth = 160.0\n"
        "height = 120.0\nin_meters = false\n"
    )
    pixels = _rig_canvas(tmp_path / "pixels", crop)
    mask = np.zeros((120, 160), bool)
    mask[BOXES["water"]] = True
    regions = [darsia.LabelledRegion("a:water", "a", "water", 1.0)]
    assert pixels["grid"] is None
    with pytest.raises(ValueError, match="crop in pixels"):
        darsia.move_labels(darsia.LabelSet(regions, [mask], canvas), pixels)


def test_labels_move_between_corrected_canvases_of_a_rig(tmp_path):
    """A region on a 1 mm canvas, moved to a 0.5 mm canvas shifted by a fraction
    of a pixel, covers the same rig area within one target pixel and is sampled
    there with its move recorded; moving it back returns the original."""
    coarse = _rig_canvas(tmp_path / "coarse", shape=(60, 80))
    fine = _rig_canvas(tmp_path / "fine", origin=[0.0003, 0.0598])
    mask = np.zeros((60, 80), bool)
    mask[20:35, 10:50] = True  # x from 10 to 50 mm, y from 25 to 40 mm
    regions = [darsia.LabelledRegion("a:gas", "a", "gas", 1.0)]
    moved = darsia.move_labels(darsia.LabelSet(regions, [mask], coarse), fine)
    rows, cols = np.nonzero(moved.masks[0])
    x = 0.3 + 0.5 * np.array([cols.min(), cols.max() + 1])
    y = 59.8 - 0.5 * np.array([rows.max() + 1, rows.min()])
    np.testing.assert_allclose([*x, *y], [10, 50, 25, 40], atol=0.5)
    assert rows.size == (np.ptp(rows) + 1) * (np.ptp(cols) + 1)
    assert (moved.regions, moved.canvas, moved.moved_from) == (regions, fine, coarse)
    current, baseline, _, _ = _frame(0)
    options = {"settings": SETTINGS, "pixel_size_mm": PIXEL_MM, "canvas": fine}
    samples = darsia.frame_samples(current, baseline, None, moved, **options)
    assert (samples.accepted_canvases, samples.moved_from) == ((), (coarse,))
    back = darsia.move_labels(moved, coarse)
    np.testing.assert_array_equal(back.masks[0], mask)


def test_a_model_is_read_only_for_images_on_its_canvas(tmp_path, canvas):
    """A model read for images on its canvas keeps it. Read for images of another
    canvas, also of another origin at another resolution, or as a delivered folder
    without one, it is refused with the differing settings named; read by explicit
    choice it gives the same probabilities."""
    trained = darsia.train_phase_model(_samples(canvas), SETTINGS)
    folder = tmp_path / "model"
    darsia.write_phase_model(
        folder,
        trained.model,
        pixel_size_mm=PIXEL_MM,
        co2_threshold=trained.co2_threshold,
        cross_validation=trained.cross_validation,
        accepted_canvases=(),
        moved_from=(),
    )
    model = darsia.read_phase_model(folder, canvas=canvas)
    other = _rig_canvas(tmp_path / "other", BULGE.format(1e-7))
    bulge = r"settings: corrections\.curvature\.bulge\.horizontal_bulge, "
    with pytest.raises(ValueError, match=bulge):
        darsia.read_phase_model(folder, canvas=other)
    shifted = _rig_canvas(tmp_path / "shifted", shape=(60, 80), origin=[0.001, 0.06])
    moved = r"settings: grid\.extent\.origin, grid\.origin, grid\.shape\. "
    with pytest.raises(ValueError, match=moved):
        darsia.read_phase_model(folder, canvas=shifted)
    delivered = tmp_path / "delivered"
    shutil.copytree(folder, delivered)
    (delivered / "canvas.json").unlink()
    with pytest.raises(ValueError, match=r"settings: canvas\. "):
        darsia.read_phase_model(delivered, canvas=canvas)
    accepted = darsia.read_phase_model(folder, canvas=other, accept_canvas=True)
    assert model.canvas == accepted.canvas == canvas
    current, baseline, _, _ = _frame(7)
    np.testing.assert_array_equal(
        darsia.phase_probabilities(accepted, current, baseline, pixel_size_mm=0.5),
        darsia.phase_probabilities(model, current, baseline, pixel_size_mm=0.5),
    )
