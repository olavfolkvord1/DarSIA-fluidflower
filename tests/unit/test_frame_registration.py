"""Tests of the registration of raw photographs onto a setup's reference."""

import json
import tomllib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import darsia
from darsia.presets.workflows.config.corrections import CorrectionsConfig
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup import steps as setup
from darsia.presets.workflows.setup.setup_sand_map import make_sand_map
from darsia.utils.locations import Frame
from tests.unit.test_camera_lens import BLUE, DISTORTION, RED
from tests.unit.test_camera_lens import SETTINGS as LENS
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


def _config(tmp_path) -> CorrectionsConfig:
    """The corrections of the test photographs."""
    (tmp_path / "config.toml").write_text(CONFIG)
    return CorrectionsConfig().load(tmp_path / "config.toml")


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
    """The reference is the official curvature correction's crop of the baseline's
    camera view in single precision, its homography registered from the rig
    points; a photograph's camera view of all
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
    # The rig chain's crop, computed in single precision.
    crop = {**crop, "precision": "float32"}
    stages = {"init": None, "crop": crop, "bulge": None, "stretch": None}
    reference = darsia.CurvatureCorrection(config=stages).correct_array(view / white)
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
    project = darsia.project_xy
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


def test_a_rig_registers_its_photographs_and_the_step_stores_their_homography(
    tmp_path, photos, monkeypatch
):
    """With [corrections.registration], the rig's baseline and photographs are
    registered to its reference, also after a save and a load; the corrected-image
    step stores each photograph's homography beside its image. A stored rig records
    that it has a registration, so a registration not found is not read as none."""
    experiment = SimpleNamespace(
        get_datetime=lambda path: None,
        injection_protocol=None,
        pressure_temperature_protocol=None,
        experiment_start=None,
    )
    rig = Rig()
    rig.setup_reading(
        photos / "baseline.dng",
        experiment=experiment,
        corrections_config=_config(tmp_path),
    )
    registration = rig.registration
    np.testing.assert_array_equal(
        rig.baseline.img,
        _sampled(photos / "baseline.dng", registration.homography, registration),
    )
    assert rig.baseline.dimensions == [0.64, 1.1]
    assert rig.canvas["grid"]["dimensions"] == [0.64, 1.1]

    rig.baseline_path, rig.color_corrections = photos / "baseline.dng", []
    ones = darsia.ScalarImage(np.ones((698, 1200)), dimensions=rig.baseline.dimensions)
    rig.depth = rig.porosity = rig.permeability = rig.image_porosity = ones
    rig.labels = rig.facies = darsia.ScalarImage(
        np.zeros((698, 1200), dtype=int), dimensions=rig.baseline.dimensions
    )
    rig.save(tmp_path / "rig")
    loaded = Rig.load(tmp_path / "rig")
    np.testing.assert_array_equal(loaded.registration.reference, registration.reference)
    np.testing.assert_array_equal(
        loaded.registration.homography, registration.homography
    )
    assert loaded.registration.settings == registration.settings
    assert loaded.registration.lens == registration.lens
    meta_data = json.loads((tmp_path / "rig" / "meta_data.json").read_text())
    assert meta_data["registration"] is True

    stored = {}
    frame = Frame("photos", "frame.dng")
    ctx = SimpleNamespace(
        task=SimpleNamespace(run="r1", frames=(frame,)),
        read=lambda kind, reader: loaded,
        done=lambda frame: False,
        photo=lambda run, frame: photos / frame.name,
        publish=lambda write, frame: write(tmp_path) or stored.update(frame=frame),
    )
    monkeypatch.setattr(setup, "run_settings", lambda ctx, run, sections: None)
    monkeypatch.setattr(setup, "make_experiment", lambda config: experiment)
    loaded.reference_date = None
    setup._corrected_image(ctx)
    values, found = darsia.register_raw_frame(
        photos / "frame.dng", registration, "float32"
    )
    np.testing.assert_array_equal(setup.read_image(tmp_path).img, values)
    np.testing.assert_array_equal(loaded.read_image(photos / "frame.dng").img, values)
    kept = darsia.read_frame_registration(tmp_path)
    np.testing.assert_array_equal(kept.homography, found.homography)
    assert kept.homography.dtype == np.float32 and kept.matches == found.matches
    assert stored["frame"] == frame

    # A share that does not find the registration's files yet: the load raises a
    # file not found, which a result's read reads again.
    for name in ("registration.json", "registration_reference.npz"):
        (tmp_path / "rig" / name).rename(tmp_path / f"hidden-{name}")
        with pytest.raises(FileNotFoundError):
            Rig.load(tmp_path / "rig")
        (tmp_path / f"hidden-{name}").rename(tmp_path / "rig" / name)


SHAPE_CHAIN = {
    "mapping": '[corrections.mapping]\npath = "mapping.npz"\n',
    "resize": '[corrections.resize]\nmode = "scale"\nscale = 0.5\n',
    "drift": '[corrections.drift]\nmode = "colorchecker"\ncolorchecker = "upper_left"\n',
    "curvature": "[corrections.curvature.crop]\ntop_left = [43, 57]\n"
    "bottom_left = [679, 60]\nbottom_right = [672, 1145]\ntop_right = [42, 1143]\n"
    "width = 1.1\nheight = 0.64\n",
}
"""A table of each shape correction that the registration replaces."""


@pytest.mark.parametrize("name", ["raw", *SHAPE_CHAIN])
def test_a_registration_needs_raw_photographs_and_replaces_the_shape_chain(
    tmp_path, photos, name
):
    """The registration needs raw photographs, and a stored mapping, a resize, a
    drift correction or a curvature correction beside it is refused: its reference
    already is the camera grid cropped to the rig points."""
    raw = '[corrections.raw]\ndemosaic = "edge_aware"\n'
    text = CONFIG.replace(raw, "") if name == "raw" else CONFIG + SHAPE_CHAIN[name]
    (tmp_path / "config.toml").write_text(text)
    config = CorrectionsConfig().load(tmp_path / "config.toml")
    refusal = (
        r"needs \[corrections.raw\]"
        if name == "raw"
        else rf"replaces the shape chain: \[corrections.{name}\] cannot"
    )
    with pytest.raises(ValueError, match=refusal):
        Rig().setup_reading(
            photos / "baseline.dng",
            experiment=SimpleNamespace(get_datetime=lambda path: None),
            corrections_config=config,
        )


def test_a_baseline_with_other_lens_tables_than_the_calibrations_is_refused(
    tmp_path, photos, monkeypatch
):
    """A baseline that records other lens tables than those the recorded-table
    model was calibrated with is refused; tables without that model are refused
    when the configuration loads."""
    tables = "[corrections.registration.lens_tables]\n" + "".join(
        f"{key} = {list(values)}\n"
        for key, values in zip(("distortion", "red", "blue"), (DISTORTION, RED, BLUE))
    )
    (tmp_path / "config.toml").write_text(CONFIG + tables)
    with pytest.raises(ValueError, match="camera_lens model, which is not configured"):
        CorrectionsConfig().load(tmp_path / "config.toml")

    config = _config(tmp_path)
    registration = config.registration
    registration.camera_lens = darsia.CameraLensSettings(**LENS)
    registration.lens_tables = darsia.CameraLensTables(DISTORTION, RED, BLUE)
    read = []

    def recorded(path, settings):
        read.append((path, settings))
        return darsia.CameraLensTables(DISTORTION, RED, BLUE[::-1])

    monkeypatch.setattr(darsia, "read_camera_lens_tables", recorded)
    with pytest.raises(ValueError, match="records other lens tables"):
        Rig().setup_reading(
            photos / "baseline.dng",
            experiment=SimpleNamespace(get_datetime=lambda path: None),
            corrections_config=config,
        )
    assert read == [(photos / "baseline.dng", registration.camera_lens)]


def test_a_sand_map_is_placed_on_the_registrations_reference_grid(
    tmp_path, monkeypatch
):
    """Without a stored mapping, the sand map lies on the registration's grid: the
    official crop of the camera grid to the rig points, in metres."""
    monkeypatch.chdir(tmp_path)
    corners = [[0, 0], [0, 0.64], [1.1, 0], [1.1, 0.64]]
    drawing = {"anchors": corners, "polygons": [[1, [0, 1, 3, 2]]]}
    (tmp_path / "drawing.json").write_text(json.dumps(drawing))
    (tmp_path / "config.toml").write_text(
        CONFIG
        + '[labeling]\nlabels = "labels.npz"\n'
        + '[labeling.sand_map]\ndrawing = "drawing.json"\nedge_fill_mm = 0.0\n'
        + '[rig]\npath = "rig"\nwidth = 1.1\nheight = 0.64\ndim = 2\n'
    )
    config = FluidFlowerConfig(
        tmp_path / "config.toml", require_data=False, require_results=False
    )
    labels = make_sand_map(config)
    assert labels.img.shape == (698, 1200) and labels.dimensions == [0.64, 1.1]
    assert list(labels.origin) == [0.0, 0.64] and set(np.unique(labels.img)) == {1}
