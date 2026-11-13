"""Unit tests for the image mapping correction."""

import numpy as np

import darsia


def _ramp(rows: int, cols: int) -> np.ndarray:
    """RGB image whose channels are linear in row and column."""
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return np.stack([2 * r + 3 * c, r - c + 40, 5 * r], axis=-1)


def _affine_maps(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Source coordinates on a coarser, shifted lattice (exact in 1/32 steps)."""
    i, j = np.mgrid[0:rows, 0:cols].astype(np.float32)
    return 1.5 + 1.25 * j, 0.75 + 1.5 * i


def test_affine_mapping_samples_linear_image_exactly():
    """Bilinear sampling reproduces a linear image at the mapped coordinates."""
    map_x, map_y = _affine_maps(4, 5)
    mapping = darsia.ImageMappingCorrection(
        map_x, map_y, source_shape=(8, 10), dimensions=[0.4, 0.5]
    )
    image = darsia.OpticalImage(_ramp(8, 10), color_space="RGB", dimensions=[8, 10])

    mapped = mapping(image)

    expected = np.stack([2 * map_y + 3 * map_x, map_y - map_x + 40, 5 * map_y], -1)
    assert mapped.img.dtype == np.float32
    assert mapping.correct_array(_ramp(8, 10).astype(np.float64)).dtype == np.float64
    assert mapped.img.shape == (4, 5, 3)
    np.testing.assert_allclose(mapped.img, expected, rtol=1e-6)
    np.testing.assert_allclose(mapped.dimensions, [0.4, 0.5])
    np.testing.assert_allclose(mapped.origin, [0.0, 0.4])


def test_channelwise_maps_shift_each_channel():
    """Maps with a channel axis sample each channel at its own coordinates."""
    i, j = np.mgrid[0:3, 0:4].astype(np.float32)
    shift = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    map_x = j[..., None] + shift
    map_y = np.repeat(i[..., None], 3, axis=-1)
    mapping = darsia.ImageMappingCorrection(
        map_x, map_y, source_shape=(4, 8), dimensions=[3, 4]
    )
    source = np.tile(np.arange(8, dtype=np.float32)[None, :, None], (4, 1, 3))

    mapped = mapping.correct_array(source)

    np.testing.assert_array_equal(mapped[..., 0], j)
    np.testing.assert_array_equal(mapped[..., 1], j + 1)
    np.testing.assert_array_equal(mapped[..., 2], j + 2)


def test_invalid_source_pixels_and_outside_samples_become_nan():
    """Only nonzero interpolation weights on invalid pixels invalidate a target."""
    source = _ramp(6, 6)
    source[2, 3, 1] = np.nan
    # Coordinates are (x, y) = (column, row); the last target row lies outside.
    map_x = np.array([[2.5, 4.0], [3.0, 2.0], [-3.0, 1.0]], dtype=np.float32)
    map_y = np.array([[2.0, 2.0], [1.0, 2.0], [1.0, 9.0]], dtype=np.float32)
    mapping = darsia.ImageMappingCorrection(
        map_x, map_y, source_shape=(6, 6), dimensions=[3, 2]
    )

    mapped = mapping.correct_array(source)

    # (2, 2.5) draws on (2, 3) with weight 1/2: invalid in all channels.
    assert np.isnan(mapped[0, 0]).all()
    # Pixel (2, 4) is not in the footprint; (1, 3) and (2, 2) touch the invalid
    # pixel with zero weight only.
    np.testing.assert_array_equal(mapped[0, 1], source[2, 4])
    np.testing.assert_array_equal(mapped[1, 0], source[1, 3])
    np.testing.assert_array_equal(mapped[1, 1], source[2, 2])
    assert np.isnan(mapped[2]).all()
