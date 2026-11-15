"""Tests of the registration of raw photographs onto a setup's reference."""

import tomllib

import cv2
import numpy as np
import pytest

import darsia
from tests.unit.test_raw_decode import BLACK, LAYOUTS, WHITE, write_dng

pytest.importorskip("rawpy")

CONFIG = """
[corrections.raw]
demosaic = "edge_aware"
[corrections.grid]
precision = "float32"
area_factor = 1.0
[corrections.registration]
camera_shape = [730, 1200]
channel = 1
[corrections.registration.crop]
top_left = [43, 57]
bottom_left = [679, 60]
bottom_right = [672, 1145]
top_right = [42, 1143]
width = 1.1
height = 0.64
[corrections.registration.radial_lens]
centre = [600.0, 365.0]
radius = 700.0
coefficients = [[0.4, -0.2], [0.0, 0.0], [-0.3, 0.1]]
sensor_offset = [14.0, 12.0]
# Production's values (test_projective_registration.py) placed on this grid.
[corrections.registration.settings."698x1200"]
fit_columns = [100, 1101, 100]
fit_rows = [100, 601, 100]
template_half = 40
search_half = 48
min_correlation = 0.5
minimum_matches = 40
ransac_threshold = 1.0
ransac_confidence = 0.995
ransac_iterations = 2000
refinement_steps = 10
refinement_tolerance = 1.1920928955078125e-07
minimum_inlier_fraction = 0.9
orientation_columns = [0, 300, 600, 900, 1199]
orientation_rows = [0, 175, 350, 525, 697]
heldout_zones = [
    ["upper_right", [950, 60, 1150, 220], 30],
    ["lower_left", [60, 480, 300, 660], 40],
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
"""A camera grid of 730 x 1200 on a 760 x 1240 sensor, a crop to four rig points
onto a 698 x 1200 grid of 0.64 x 1.1 m, and a radial lens model."""
MOVE = (1.4, -0.8)
"""The camera's move between the baseline and the photograph, in sensor pixels."""


def _mosaic(scene: np.ndarray) -> np.ndarray:
    """The RGGB mosaic of a scene in camera units, above the black level."""
    phase = np.reshape(LAYOUTS["RGGB"][0], (2, 2))
    rows, cols = np.indices(scene.shape[:2])
    values = scene[rows, cols, phase[rows % 2, cols % 2]] + BLACK
    return np.clip(np.round(values), 0, WHITE).astype(np.uint16)


@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    """A textured baseline and a moved photograph with a clipped spot."""
    folder = tmp_path_factory.mktemp("photos")
    rng = np.random.default_rng(5)
    texture = cv2.GaussianBlur(rng.standard_normal((760, 1240)), (0, 0), 2.5)
    texture += 0.5 * cv2.GaussianBlur(rng.standard_normal((760, 1240)), (0, 0), 9.0)
    scene = np.stack([3000.0, 4000.0, 2500.0], -1) * np.exp(texture)[..., None]
    moved = cv2.warpAffine(
        scene,
        np.array([[1.0, 0.0, MOVE[0]], [0.0, 1.0, MOVE[1]]]),
        (1240, 760),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    mosaic = _mosaic(moved)
    mosaic[400:404, 640:644] = WHITE  # clipped photosites inside a fit patch
    write_dng(folder / "baseline.dng", _mosaic(scene), LAYOUTS["RGGB"][0])
    write_dng(folder / "frame.dng", mosaic, LAYOUTS["RGGB"][0])
    return folder


def _sampled(path, homography, registration, divide_first=False):
    """A photograph sampled once from its sensor through a homography and the lens
    model onto the reference's grid, divided by its white range after (or
    before) the interpolation."""
    sensor = darsia.read_raw_sensor(path, "edge_aware")
    values = sensor.values.astype(np.float32)
    values[sensor.invalid] = np.nan
    if divide_first:
        values /= np.float32(sensor.white_range)
    mapping = darsia.registration_mapping(
        homography,
        registration.reference.shape,
        values.shape[:2],
        list(registration.dimensions),
        list(registration.origin),
        coordinates=darsia.lens_coordinates(path, registration.lens),
    )
    sampled = mapping.correct_array(values)
    return sampled if divide_first else sampled / np.float32(sensor.white_range)


def test_a_raw_photograph_is_registered_to_the_reference_and_sampled_once(
    tmp_path, photos
):
    """The reference is the official crop of the baseline's camera view, its
    homography registered from the rig points; a photograph's camera view of all
    its values is registered from it, and its sensor values are sampled once
    through its homography and lens model, divided by the white range after."""
    table = tomllib.loads(CONFIG)["corrections"]["registration"]
    crop = table["crop"]
    corners = [
        crop[k] for k in ("top_left", "bottom_left", "bottom_right", "top_right")
    ]
    size = (crop["width"], crop["height"])
    settings = {
        grid: darsia.ProjectiveRegistrationSettings(**values)
        for grid, values in table["settings"].items()
    }
    models = (darsia.RadialLensSettings(**table["radial_lens"]),)
    registration, baseline, setup_quality = darsia.setup_frame_registration(
        photos / "baseline.dng",
        settings,
        models,
        (730, 1200),
        1,
        "edge_aware",
        np.float32,
        corners=corners,
        size=size,
    )
    sensor = darsia.read_raw_sensor(photos / "baseline.dng", "edge_aware")
    lens = darsia.lens_coordinates(photos / "baseline.dng", models)
    view = darsia.camera_view(
        sensor.values[..., 1:2].astype(np.float32), lens, (730, 1200), (1,)
    )[..., 0]
    white = np.float32(sensor.white_range)
    reference = darsia.extract_quadrilateral_ROI(
        view / white, pts_src=darsia.make_voxel(corners), width=1.1, height=0.64
    )
    np.testing.assert_array_equal(registration.reference, reference)
    assert registration.dimensions == (0.64, 1.1) and setup_quality.matches == 66
    np.testing.assert_array_equal(
        baseline,
        _sampled(photos / "baseline.dng", registration.homography, registration),
    )

    values, found = darsia.register_raw_frame(
        photos / "frame.dng", registration, "float32"
    )
    # Every fit point matches: the clipped spot is not left out of the views.
    assert found.matches == 66 and found.homography.dtype == np.float32
    y, x = np.mgrid[0:698:20, 0:1200:20].astype(np.float32)
    project = darsia.corrections.shape.registration._project
    sx, sy = project(registration.homography.astype(float), x, y)
    fx, fy = project(found.homography.astype(float), x, y)
    # Within a quarter pixel: the move is demosaiced from the moved mosaic.
    assert np.hypot(fx - sx - MOVE[0], fy - sy - MOVE[1]).max() < 0.25
    np.testing.assert_array_equal(
        values, _sampled(photos / "frame.dng", found.homography, registration)
    )
    first = _sampled(photos / "frame.dng", found.homography, registration, True)
    assert not np.array_equal(values, first, equal_nan=True)
    assert np.isnan(values).any(axis=-1).sum() > 0  # around the clipped spot
    with pytest.raises(ValueError, match="No registration settings for a 730 x 1200"):
        darsia.setup_frame_registration(
            photos / "baseline.dng",
            settings,
            models,
            (730, 1200),
            1,
            "edge_aware",
            np.float32,
        )
