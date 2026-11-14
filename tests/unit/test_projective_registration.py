"""Unit tests for the projective registration to the run grid."""

import tomllib

import cv2
import numpy as np
import pytest

import darsia

SETTINGS = tomllib.loads(
    """
    # The values our production ran with. Production's positions are those of its
    # 2982 x 5472 run grid (fit columns [900, 4500, 180], rows [500, 2400, 180],
    # 100 matches); here they are placed on a 640 x 1100 test grid.
    [registration]
    fit_columns = [100, 1001, 100]
    fit_rows = [100, 541, 110]
    template_half = 40
    search_half = 48
    min_correlation = 0.5
    minimum_matches = 40
    ransac_threshold = 1.0
    ransac_confidence = 0.995
    ransac_iterations = 2000
    refinement_steps = 10
    refinement_tolerance = 1.1920928955078125e-07  # single-precision epsilon
    minimum_inlier_fraction = 0.9
    orientation_columns = [0, 275, 550, 825, 1099]
    orientation_rows = [0, 160, 320, 480, 639]
    heldout_zones = [
        ["upper_right", [870, 60, 1060, 220], 30],
        ["lower_left", [60, 430, 300, 600], 40],
    ]
    heldout_template_half = 25
    heldout_search_half = 10
    heldout_min_correlation = 0.55
    heldout_minimum_matches = 20
    sigma_small = 1.2
    sigma_large = 6.0
    minimum_template_std = 0.0001
    log_offset = 1.0
    """
)["registration"]

GRID = (640, 1100)
"""Rows and columns of the run grid."""

H = np.array(
    [
        [0.98755, 0.00516, 57.013],
        [-0.000136, 0.99112, 43.001],
        [2.5e-6, 9.7e-6, 1.0],
    ]
)
"""Homography from the run grid to the camera image, like a setup's."""

RANSAC = {
    "confidence": SETTINGS["ransac_confidence"],
    "iterations": SETTINGS["ransac_iterations"],
    "refinement_steps": SETTINGS["refinement_steps"],
    "refinement_tolerance": SETTINGS["refinement_tolerance"],
}
"""The RANSAC controls of the settings, as keywords of the RANSAC alone."""


def _project(h: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Homography applied to pixel centres, in double precision."""
    den = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / den, (
        h[1, 0] * x + h[1, 1] * y + h[1, 2]
    ) / den


@pytest.fixture(scope="module")
def images():
    """Textured camera images (camera units) of a setup and of a moved frame, and
    the run-grid reference of the setup's image under H."""
    rng = np.random.default_rng(3)
    shape = (730, 1200)
    texture = cv2.GaussianBlur(rng.standard_normal(shape), (0, 0), 2.5)
    texture += 0.5 * cv2.GaussianBlur(rng.standard_normal(shape), (0, 0), 9.0)
    camera = (3000.0 * np.exp(texture)).astype(np.float32)
    y, x = np.mgrid[0 : GRID[0], 0 : GRID[1]].astype(np.float64)
    mx, my = _project(H, x, y)
    reference = cv2.remap(
        camera, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_CUBIC
    )
    frame = cv2.warpAffine(
        camera,
        np.array([[1.0, 0.0, 1.4], [0.0, 1.0, -0.8]]),
        shape[::-1],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return camera, frame, reference


def test_ransac_recovers_a_known_homography_among_outliers():
    """The inliers are the true matches, and their homography is recovered."""
    rng = np.random.default_rng(1)
    source = rng.uniform([0, 0], [1100, 640], (300, 2))
    target = np.column_stack(_project(H, source[:, 0], source[:, 1]))
    target += 0.05 * rng.standard_normal(target.shape)
    outlier = rng.random(300) < 0.3
    target[outlier] += rng.uniform(20, 60, (outlier.sum(), 2))

    for dtype in (np.float32, np.float64):
        h, inliers = darsia.ransac_homography(
            source.astype(dtype), target.astype(dtype), 1.0, **RANSAC
        )
        assert h.dtype == dtype
        np.testing.assert_array_equal(inliers[:, 0] == 1, ~outlier)
        mapped = np.column_stack(_project(h.astype(float), *source.T))
        exact = np.column_stack(_project(H, *source.T))
        assert np.abs(mapped - exact).max() < 0.05


def test_rig_points_give_the_homography_of_the_grid_corners():
    """The run grid's corner pixel centres map onto the rig points."""
    corners = np.array([[43, 57], [679, 60], [672, 1145], [42, 1143]])
    h = darsia.rig_points_homography(corners, GRID, np.float64)
    rows, cols = GRID
    x, y = _project(
        h, np.array([0, 0, cols - 1, cols - 1]), np.array([0, rows - 1, rows - 1, 0])
    )
    np.testing.assert_allclose(np.column_stack((y, x)), corners, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_registration_recovers_the_homographies_of_a_setup_and_a_frame(images, dtype):
    """From the rig points for the setup, then from the setup's for a moved frame."""
    camera, frame, reference = (image.astype(dtype) for image in images)
    settings = darsia.ProjectiveRegistrationSettings(**SETTINGS)
    rows, cols = GRID
    x = np.array([0, 0, cols - 1, cols - 1], float)
    corners = np.column_stack(
        _project(H, x, np.array([0, rows - 1, rows - 1, 0], float))
    )
    # Rig points picked in the photograph: whole pixels, a few pixels off.
    rig_points = np.round(corners[:, ::-1] + [[2, -3], [-3, 2], [3, 3], [-2, -2]])

    setup = darsia.projective_registration(
        reference,
        camera,
        darsia.rig_points_homography(rig_points, GRID, dtype),
        settings,
    )
    moved = darsia.projective_registration(reference, frame, setup.homography, settings)

    assert setup.homography.dtype == dtype
    assert setup.matches == 50 and setup.inlier_fraction == 1.0
    assert set(setup.heldout) == {"upper_right", "lower_left"}
    # Within a tenth of a pixel over the whole run grid.
    y, x = np.mgrid[0:rows:20, 0:cols:20].astype(float)
    for registration, (dx, dy) in ((setup, (0, 0)), (moved, (1.4, -0.8))):
        ex, ey = _project(H, x, y)
        rx, ry = _project(registration.homography.astype(float), x, y)
        assert np.hypot(rx - ex - dx, ry - ey - dy).max() < 0.1


def test_the_mapping_of_a_registration_samples_the_image_onto_the_run_grid():
    """Replayed by the image mapping, the homography gives the run-grid image, with
    shared or per-channel coordinates."""
    rows, cols = GRID
    camera = np.mgrid[0:730, 0:1200].astype(np.float32)
    # Each channel holds a linear function of the camera position (column, row).
    image = np.stack([3 * camera[1] + camera[0], camera[1] - 2 * camera[0] + 900], -1)
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    u, v = _project(H, x, y)

    mapping = darsia.registration_mapping(
        H.astype(np.float32), GRID, image.shape[:2], dimensions=[0.64, 1.1]
    )
    shifted = darsia.registration_mapping(
        H.astype(np.float32),
        GRID,
        image.shape[:2],
        dimensions=[0.64, 1.1],
        coordinates=lambda u, v: (np.stack([u, u + 2], -1), np.stack([v, v], -1)),
    )

    # Bilinear sampling is exact on linear images, up to OpenCV's 1/32 pixel steps.
    mapped = mapping.correct_array(image)
    np.testing.assert_allclose(mapped[..., 0], 3 * u + v, atol=0.07)
    np.testing.assert_allclose(mapped[..., 1], u - 2 * v + 900, atol=0.07)
    channelwise = shifted.correct_array(image)
    np.testing.assert_array_equal(channelwise[..., 0], mapped[..., 0])
    np.testing.assert_allclose(channelwise[..., 1], u + 2 - 2 * v + 900, atol=0.07)
