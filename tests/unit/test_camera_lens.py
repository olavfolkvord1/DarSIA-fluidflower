"""Unit tests for the lens model from the tables a camera records in raw files."""

import tomllib

import numpy as np
import tifffile

import darsia

SETTINGS = tomllib.loads(
    """
    # The values our production ran with (one camera profile).
    [camera_lens]
    page = 0
    subpage = 0
    distortion_tag = 28727
    distortion_header = 11
    distortion_entries = [1, 12]
    chromatic_tag = 28725
    chromatic_header = 22
    red_entries = [1, 12]
    blue_entries = [12, 23]
    knots = [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001,
             0.7000000000000001, 0.8, 0.9, 1.0]
    distortion_scale = 16384.0
    chromatic_scale = 2097152.0
    centre = [2735.941390570268, 1540.0423206018281]
    radius = 3291.1963476237365
    amplitude = 1.0033805150829112
    shift = [0.008900502454740798, -0.008331840772507504]
    sensor_offset = [12, 296]
    """
)["camera_lens"]

DISTORTION = (720, 698, 627, 507, 337, 128, -111, -365, -610, -839, -1053)
RED = (0, 512, 384, 384, 384, 512, 512, 640, 768, 896, 896)
BLUE = (0, 512, 640, 768, 768, 640, 512, 384, 256, 128, 256)


def test_the_tables_are_read_from_the_record_of_a_raw_file(tmp_path):
    """The tables come from the configured sub-page and tags, without headers."""
    path = tmp_path / "frame.arw"
    with tifffile.TiffWriter(path) as tiff:
        tiff.write(np.zeros((4, 6), np.uint16), subifds=1)
        tiff.write(
            np.zeros((2, 3), np.uint16),
            extratags=[
                (28727, "h", 12, (11, *DISTORTION), True),
                (28725, "h", 23, (22, *RED, *BLUE), True),
            ],
        )

    tables = darsia.read_camera_lens_tables(path, darsia.CameraLensSettings(**SETTINGS))

    assert tables == darsia.CameraLensTables(DISTORTION, RED, BLUE)


def test_camera_coordinates_move_radially_and_per_colour_onto_the_sensor():
    """The distortion scales the distance from the centre; red and blue once more."""
    settings = darsia.CameraLensSettings(**SETTINGS)
    tables = darsia.CameraLensTables(DISTORTION, RED, BLUE)
    cu, cv = settings.centre
    # The centre, a point at 0.25 of the radius and one beyond the last knot.
    du = np.array([0.0, 0.25 * settings.radius * 0.6, 1.1 * settings.radius])
    dv = np.array([0.0, 0.25 * settings.radius * 0.8, 0.0])

    x, y = darsia.camera_lens_coordinates(cu + du, cv + dv, tables, settings)

    scale = 1 + settings.amplitude * np.array([720, 567, -1053]) / 16384.0
    red = scale * (1 + np.array([0, 384, 896]) / 2097152.0)
    blue = scale * (1 + np.array([0, 704, 256]) / 2097152.0)
    for channel, factor in enumerate((red, scale, blue)):
        np.testing.assert_allclose(
            x[:, channel], cu + 12 + settings.shift[0] + factor * du
        )
        np.testing.assert_allclose(
            y[:, channel], cv + 296 + settings.shift[1] + factor * dv
        )
    single, _ = darsia.camera_lens_coordinates(
        (cu + du).astype(np.float32), cv + dv.astype(np.float32), tables, settings, (1,)
    )
    assert single.dtype == np.float32 and single.shape == (3, 1)
