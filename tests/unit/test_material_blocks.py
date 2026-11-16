"""Unit tests for the block fit of the material illumination correction."""

import tomllib

import numpy as np
import pytest

import darsia

FIT = darsia.MaterialFitSettings(
    **tomllib.loads(
        """
        # The values the fit ran with in production.
        spatial_ridge = 0.002
        offset_ridge = 0.02
        residual_constant_ridge = 0.02
        residual_spatial_ridge = 0.01
        iterations = 8
        huber_threshold = 1.5
        clip_level = 0.98
        dark_level = 8.0
        minimum_block_cells = 32
        chroma_cutoff = 0.08
        chroma_sigmas = 3.5
        """
    )
)


def _bands(shape=(40, 60)) -> np.ndarray:
    """Material 1 in the upper and material 2 in the lower half."""
    material_map = np.ones(shape, dtype=int)
    material_map[shape[0] // 2 :] = 2
    return material_map


def test_projection_to_the_fitting_grid():
    """Area means; unusable and clipped cells unsafe; pure cells eroded by one."""
    rgb = np.full((40, 60, 3), 0.5, dtype=np.float32)
    rgb[0, 0, 1] = -0.1
    rgb[8:12, 20:24, 0] = 0.99

    projection = darsia.project_to_fit_grid(
        rgb, _bands(), np.ones((40, 60), dtype=bool), (10, 15), [1, 2], FIT
    )

    expected = np.full((10, 15, 3), 0.5, dtype=np.float32)
    expected[0, 0] = np.nan
    expected[2, 5, 0] = 0.99
    np.testing.assert_allclose(projection["rgb"], expected, rtol=1e-6)
    np.testing.assert_array_equal(
        np.argwhere(~projection["fit_safe"]), [[0, 0], [2, 5]]
    )
    assert (projection["materials"][1:4, 1:-1] == 1).all()
    assert (projection["materials"][6:9, 1:-1] == 2).all()
    # Erosion at the material boundary and at the grid's edge.
    assert (projection["materials"][[0, 4, 5, 9]] == -1).all()
    assert (projection["materials"][:, [0, -1]] == -1).all()


def test_block_samples_positions_folds_and_margin():
    """Block medians carry their centre, fold and the target colour."""
    rgb = np.full((80, 120, 3), 0.5, dtype=np.float32)
    projection = darsia.project_to_fit_grid(
        rgb,
        _bands((80, 120)),
        np.ones((80, 120), dtype=bool),
        (20, 30),
        [1, 2],
        FIT,
    )
    targets = {1: np.log([10.0, 20.0, 30.0]), 2: np.log([5.0, 5.0, 5.0])}

    def samples(edge_margin):
        return darsia.collect_block_samples(
            projection,
            targets,
            dimensions=(0.8, 1.2),
            edge_margin=edge_margin,
            block_size=10,
            dn_scale=100.0,
            settings=FIT,
            dtype=np.float64,
        )

    # Two block rows and three block columns, one material each.
    blocks = samples(0.0)
    assert len(blocks["x"]) == 6
    np.testing.assert_allclose(blocks["before"], 50.0, rtol=1e-6)
    np.testing.assert_allclose(
        blocks["target"][blocks["material"] == 1], [[10, 20, 30]] * 3, rtol=1e-6
    )
    np.testing.assert_array_equal(blocks["fold"], [0, 2, 4, 3, 0, 2])
    # The margin leaves 30 cells in the outer blocks and 50 in the middle ones.
    assert len(samples(0.17)["x"]) == 2


def test_colour_outlier_block_is_flagged():
    """A block whose chromaticity deviates from its material's blocks is flagged."""
    count = 20
    before = np.full((count, 3), 50.0)
    before[7] = [80.0, 50.0, 50.0]
    samples = {
        "before": before,
        "target": np.full((count, 3), 40.0),
        "material": np.ones(count, dtype=np.int64),
        "fold": np.zeros(count, dtype=np.int64),
    }

    outliers = darsia.block_color_outliers(samples, [1], FIT)

    np.testing.assert_array_equal(np.flatnonzero(outliers), [7])


def test_fit_removes_a_known_illumination():
    """The corrected baseline matches the target colours of its materials."""
    rows, cols = np.mgrid[0:200, 0:300]
    y, x = (rows + 0.5) * 2 / 200 - 1, (cols + 0.5) * 2 / 300 - 1
    gain = np.exp(
        np.stack(
            [0.2 + 0.3 * x - 0.1 * y + 0.05 * x * x, 0.1 - 0.2 * x + 0.2 * y, 0.1 * x],
            axis=-1,
        )
    )
    material_map = np.ones((200, 300), dtype=int)
    material_map[70:140] = 2
    material_map[140:] = 3
    colours = {1: [0.3, 0.2, 0.1], 2: [0.2, 0.25, 0.3], 3: [0.4, 0.35, 0.2]}
    target_image = np.array([colours[m] for m in material_map.ravel()]).reshape(
        200, 300, 3
    )
    targets = {m: np.log(np.array(colour) * 1000.0) for m, colour in colours.items()}

    correction = darsia.fit_material_illumination(
        target_image / gain,
        material_map,
        np.ones((200, 300), dtype=bool),
        targets,
        dimensions=(0.2, 0.3),
        fit_shape=(100, 150),
        block_size=10,
        edge_margin=0.0,
        dn_scale=1000.0,
        settings=FIT,
    )

    corrected = correction.correct_array(target_image / gain)
    np.testing.assert_allclose(corrected, target_image, rtol=2e-3)


def _illuminated(columns: np.ndarray, rows: int = 200) -> tuple:
    """Three horizontal bands of material under a known illumination, seen at the
    given columns of a 300 column wide reference; with the targets."""
    x, y = np.meshgrid(
        (columns + 0.5) * 2 / 300 - 1, (np.arange(rows) + 0.5) * 2 / rows - 1
    )
    gain = np.exp(np.stack([0.2 + 0.3 * x - 0.1 * y, 0.1 - 0.2 * x + 0.2 * y, 0.1 * x]))
    material_map = np.ones((rows, len(columns)), dtype=int)
    material_map[70:140] = 2
    material_map[140:] = 3
    colours = {1: [0.3, 0.2, 0.1], 2: [0.2, 0.25, 0.3], 3: [0.4, 0.35, 0.2]}
    image = np.array([colours[m] for m in material_map.ravel()]).reshape(rows, -1, 3)
    targets = {m: np.log(np.array(colour) * 1000.0) for m, colour in colours.items()}
    return image / np.moveaxis(gain, 0, -1), image, material_map, targets


def _fit(baseline, material_map, targets, reference=None, dimensions=(0.2, 0.3)):
    """The fit of the tests, on a 100 x 150 grid of 10 cell blocks."""
    return darsia.fit_material_illumination(
        baseline,
        material_map,
        np.ones(material_map.shape, dtype=bool),
        targets,
        dimensions=dimensions,
        fit_shape=(100, 150),
        block_size=10,
        edge_margin=0.0,
        dn_scale=1000.0,
        settings=FIT,
        reference=reference,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_run_fitted_on_its_own_grid_as_reference_is_unchanged(dtype):
    """The reference run itself gives the fit without a reference, bit for bit."""
    baseline, _, material_map, targets = _illuminated(np.arange(300))
    baseline = baseline.astype(dtype)
    reference = darsia.MaterialFitReference(
        baseline, material_map, np.ones((200, 300), bool), (0.2, 0.3), np.eye(3)
    )

    plain = _fit(baseline, material_map, targets)
    fitted = _fit(baseline, material_map, targets, reference)

    np.testing.assert_array_equal(fitted.common_coefficients, plain.common_coefficients)
    for material, coefficients in plain.material_coefficients.items():
        np.testing.assert_array_equal(
            fitted.material_coefficients[material], coefficients
        )
    np.testing.assert_array_equal(fitted.gain(dtype), plain.gain(dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_projection_onto_another_fitting_grid(dtype):
    """A homography two cells to the right moves the image and its masks."""
    rgb = np.full((40, 60, 3), 0.5, dtype=dtype)
    rgb[0, 0, 1] = np.nan
    rgb[8:12, 20:24, 0] = 0.99
    mask = np.ones((40, 60), dtype=bool)
    mask[20:, 32:40] = False
    plain = darsia.project_to_fit_grid(rgb, _bands(), mask, (10, 15), [1, 2], FIT)
    resize = np.array([[0.25, 0, -0.375], [0, 0.25, -0.375], [0, 0, 1]])

    moved = darsia.project_to_fit_grid(
        rgb,
        _bands(),
        mask,
        (10, 15),
        [1, 2],
        FIT,
        resize + [[0, 0, 2], [0, 0, 0], [0] * 3],
    )

    np.testing.assert_array_equal(moved["rgb"][:, 2:], plain["rgb"][:, :-2])
    assert np.isnan(moved["rgb"][:, :2]).all()
    np.testing.assert_array_equal(moved["fit_safe"][:, 2:], plain["fit_safe"][:, :-2])
    assert not moved["fit_safe"][:, :2].any()
    # Eroded masks: the cells away from the edges of both grids.
    for key in ("frame_safe", "materials"):
        np.testing.assert_array_equal(moved[key][:, 3:14], plain[key][:, 1:12])


def test_only_cells_safe_and_pure_in_the_reference_are_fitted():
    """Cells clipped or of another material in the reference leave the blocks."""
    rgb = np.full((80, 120, 3), 0.5, dtype=np.float32)
    mask = np.ones((80, 120), dtype=bool)
    projection = darsia.project_to_fit_grid(
        rgb, _bands((80, 120)), mask, (20, 30), [1, 2], FIT
    )
    reference_rgb = rgb.copy()
    reference_rgb[:40, :40] = 0.99
    reference_map = _bands((80, 120))
    reference_map[40:, 80:] = 1
    reference = darsia.project_to_fit_grid(
        reference_rgb, reference_map, mask, (20, 30), [1, 2], FIT
    )
    targets = {1: np.log([10.0, 20.0, 30.0]), 2: np.log([5.0, 5.0, 5.0])}

    blocks = darsia.collect_block_samples(
        projection,
        targets,
        dimensions=(0.8, 1.2),
        edge_margin=0.0,
        block_size=10,
        dn_scale=100.0,
        settings=FIT,
        dtype=np.float64,
        reference=reference,
    )

    # Gone: the block clipped in the reference (upper left) and the block of
    # material 1 in the reference (lower right).
    np.testing.assert_array_equal(blocks["material"], [1, 1, 2, 2])
    np.testing.assert_array_equal(blocks["fold"], [2, 4, 3, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_reference_grid_places_the_field(dtype):
    """A run seen 30 columns to the right of the reference is corrected in place."""
    reference_baseline, _, material_map, targets = _illuminated(np.arange(300))
    baseline, image, _, _ = _illuminated(np.arange(300) + 30)
    baseline = baseline.astype(dtype)
    reference = darsia.MaterialFitReference(
        reference_baseline,
        material_map,
        np.ones((200, 300), bool),
        (0.2, 0.3),
        np.array([[1.0, 0, 30], [0, 1, 0], [0, 0, 1]]),
    )

    # The run's own dimensions (none here) are not used: the grid is the reference's.
    correction = _fit(baseline, material_map, targets, reference, dimensions=(0, 0))
    corrected = correction.correct_array(baseline)

    np.testing.assert_allclose(corrected[:, :270], image[:, :270], rtol=2e-3)


def _targets(baseline, corrected, minimum_blocks=8):
    """The targets measured on 32-pixel blocks of an image of 1 mm pixels, with a
    1 cm margin."""
    return darsia.measure_material_targets(
        baseline,
        corrected,
        _bands(baseline.shape[:2]),
        np.ones(baseline.shape[:2], bool),
        [1, 2],
        dimensions=(0.32, 0.48),
        fit_shape=(80, 120),
        block_size=8,
        edge_margin=0.01,
        dn_scale=1000.0,
        settings=FIT,
        minimum_blocks=minimum_blocks,
    )


def test_targets_are_measured_on_the_corrected_blocks():
    """The target is the median log colour of the clean training blocks of the
    corrected baseline; the baseline chooses the cells, so a block dark in the
    baseline is left out, also when the corrected baseline is bright there, and a
    block of another chromaticity is left out as a colour outlier."""
    rng = np.random.default_rng(5)
    baseline = (0.02 * rng.lognormal(0.0, 0.01, (320, 480, 3))).astype(np.float32)
    corrected = 2 * baseline
    targets, blocks = _targets(baseline, corrected)
    for material in (1, 2):
        np.testing.assert_allclose(targets[material], np.log(40.0), atol=2e-3)
        assert targets[material].dtype == np.float32
        assert blocks[material][0] == blocks[material][1]
    # The training block in rows 32-63 and columns 96-127 is dark in the
    # baseline only: it is left out.
    dark = baseline.copy()
    dark[32:64, 96:128] *= 0.2
    assert _targets(dark, corrected)[1][1][0] == blocks[1][0] - 1
    assert _targets(baseline, dark)[1][1] == blocks[1]
    # A red training block in rows 64-95 and columns 192-223 is an outlier.
    cast = corrected.copy()
    cast[64:96, 192:224, 0] *= 1.5
    found, counted = _targets(baseline, cast)
    assert counted[1] == (blocks[1][0], blocks[1][0] - 1)
    np.testing.assert_allclose(found[1], targets[1], atol=2e-3)
    with pytest.raises(ValueError, match="clean training blocks for material 1"):
        _targets(baseline, corrected, minimum_blocks=blocks[1][0] + 1)
