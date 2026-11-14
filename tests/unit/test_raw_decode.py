"""Tests of the decode of camera raw files."""

from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
import tifffile

import darsia

pytest.importorskip("rawpy")

BLACK, WHITE = 800, 16000
"""Black and white level of the test sensor."""

LAYOUTS = {
    "RGGB": ((0, 1, 1, 2), cv2.COLOR_BayerBG2RGB_EA),
    "BGGR": ((2, 1, 1, 0), cv2.COLOR_BayerRG2RGB_EA),
    "GRBG": ((1, 0, 2, 1), cv2.COLOR_BayerGB2RGB_EA),
    "GBRG": ((1, 2, 0, 1), cv2.COLOR_BayerGR2RGB_EA),
}
"""DNG colour filter pattern (0 red, 1 green, 2 blue) and OpenCV's code for it."""


def write_dng(path, mosaic, pattern):
    """Write a mosaic as a minimal DNG raw file, which LibRaw reads."""
    tifffile.imwrite(
        path,
        mosaic,
        photometric=32803,  # colour filter array
        extratags=[
            (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
            (33422, "B", 4, pattern, True),  # CFAPattern
            (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
            (50708, "s", 0, "Test camera", True),  # UniqueCameraModel
            (50714, "H", 1, (BLACK,), True),  # BlackLevel
            (50717, "H", 1, (WHITE,), True),  # WhiteLevel
            (50721, "f", 9, (1, 0, 0, 0, 1, 0, 0, 0, 1), True),  # ColorMatrix1
        ],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_decode_is_linear_camera_rgb_and_clipping_is_invalid(tmp_path, layout, dtype):
    """Edge-aware RGB of the black-subtracted mosaic over the white range, in the
    given dtype; values that draw on a clipped or outermost photosite are NaN."""
    pattern, code = LAYOUTS[layout]
    phase = np.reshape(pattern, (2, 2))
    mosaic = np.random.default_rng(0).integers(600, 15000, (32, 48), dtype=np.uint16)
    y, x = np.argwhere(phase == 0)[0] + (10, 20)
    mosaic[y, x] = WHITE  # a clipped red photosite
    write_dng(tmp_path / "image.dng", mosaic, pattern)

    image = darsia.imread(tmp_path / "image.dng", demosaic="edge_aware", dtype=dtype)
    img = image.img

    subtracted = np.maximum(mosaic.astype(int) - BLACK, 0)
    scaled = subtracted.astype(dtype) / dtype(WHITE - BLACK)
    expected = cv2.demosaicing(subtracted.astype(np.uint16), code).astype(dtype)
    expected /= dtype(WHITE - BLACK)
    valid = np.isfinite(img)
    assert img.dtype == dtype
    np.testing.assert_array_equal(img[valid], expected[valid])

    # Each photosite keeps its value in its own colour, so the channels are RGB.
    rows, cols = np.indices(mosaic.shape)
    own = img[rows, cols, phase[rows % 2, cols % 2]]
    scaled[y, x] = np.nan
    np.testing.assert_array_equal(own[1:-1, 1:-1], scaled[1:-1, 1:-1])

    # Inside, only the red values drawing on the clipped photosite are invalid.
    invalid = np.zeros(img.shape, dtype=bool)
    invalid[y - 1 : y + 2, x - 1 : x + 2, 0] = True
    np.testing.assert_array_equal(~valid[2:-2, 2:-2], invalid[2:-2, 2:-2])

    # Next to the top edge, only a photosite's own colour and, at a green
    # photosite, the colour of its horizontal neighbours are valid.
    cols = np.arange(2, mosaic.shape[1] - 2)
    colour = phase[1, cols % 2]
    green = colour == 1
    edge_valid = np.zeros((cols.size, 3), dtype=bool)
    edge_valid[np.arange(cols.size), colour] = True
    edge_valid[green, phase[1, (cols[green] + 1) % 2]] = True
    np.testing.assert_array_equal(valid[1, cols], edge_valid)

    # The outermost rows and columns copy their neighbours, as OpenCV does.
    np.testing.assert_array_equal(img[[0, -1]], img[[1, -2]])
    np.testing.assert_array_equal(img[:, [0, -1]], img[:, [1, -2]])


def test_a_list_or_folder_of_raw_files_is_a_time_series(tmp_path):
    """The decoded files along the time axis, with the given dates, transformed."""
    rng = np.random.default_rng(2)
    paths = [tmp_path / "a.dng", tmp_path / "b.dng"]
    for path in paths:
        mosaic = rng.integers(600, 15000, (32, 48), dtype=np.uint16)
        write_dng(path, mosaic, LAYOUTS["RGGB"][0])
    epoch = datetime(1970, 1, 1)
    dates = [epoch, epoch + timedelta(hours=1)]
    options = dict(demosaic="edge_aware", dtype=np.float32, date=dates)

    series = darsia.imread(
        paths, transformations=[darsia.TypeCorrection(np.float64)], **options
    )

    assert series.series and series.img.shape == (32, 48, 2, 3)
    assert series.img.dtype == np.float64
    assert series.date == dates and series.time == [0.0, 3600.0]
    for i, path in enumerate(paths):
        single = darsia.imread(path, demosaic="edge_aware", dtype=np.float32)
        np.testing.assert_array_equal(series.img[:, :, i], single.img)
    np.testing.assert_array_equal(darsia.imread(tmp_path, **options).img, series.img)
