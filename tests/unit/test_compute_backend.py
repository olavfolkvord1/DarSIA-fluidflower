"""Unit tests for the compute backends on small synthetic frames."""

import numpy as np
import pytest

import darsia


@pytest.fixture
def restore_backend():
    """Give the process's backend back after a test."""
    previous = darsia.compute_backend()
    yield
    darsia.use_compute_backend(previous)


def _mapping(dtype, channel_maps=False):
    """A mapping of a 61 x 83 RGB image with invalid pixels onto a 50 x 70 grid.

    The maps turn, scale and jitter the grid; some samples fall on exact steps of
    1/64 pixel, outside the image or on its last row and column, and two maps are
    not finite.
    """
    rng = np.random.default_rng(3)
    image = rng.uniform(0, 2, (61, 83, 3)).astype(dtype)
    image[10, 20] = np.nan
    image[30, 40, 1] = np.inf
    rows, cols = np.mgrid[0:50, 0:70].astype(np.float64)
    x, y = cols * 1.17 + rows * 0.08 - 2, rows * 1.21 - cols * 0.05 + 1.5
    if channel_maps:
        x, y = x[..., None], y[..., None]
    jitter = rng.normal(0, 0.4, (50, 70, 3) if channel_maps else (50, 70))
    map_x = np.asarray(x + jitter, np.float32)
    map_y = np.asarray(y - jitter, np.float32)
    map_x[::7, ::5] = np.round(map_x[::7, ::5] * 64) / 64
    map_y[3, 3], map_x[4, 4] = np.nan, np.inf
    map_y[5, 6] = 60.0  # the last source row
    mapping = darsia.ImageMappingCorrection(
        map_x, map_y, source_shape=(61, 83), dimensions=[0.05, 0.07]
    )
    return mapping, image


def _means(dtype, shape=(70, 90), size_mm=9.0, channels=3):
    """Values with invalid pixels outside the masks, three disjoint masks and an
    empty one, and a square of ``size_mm`` on a 1 mm grid."""
    rng = np.random.default_rng(5)
    values = rng.uniform(0, 1, (*shape, channels)).astype(dtype)
    values[1, 2] = np.nan
    rows, cols = np.indices(shape)
    region = (rows * 3 + cols) % 4
    dissolved = rng.uniform(size=shape) < 0.7
    masks = {sand: region == sand for sand in range(4)}
    masks[3] = np.zeros(shape, bool)
    grid = darsia.ScalarImage(
        np.zeros(shape), dimensions=[shape[0] / 1000, shape[1] / 1000]
    )
    rev = darsia.REV(size=size_mm / 1000, img=grid)
    dissolved[1, 2] = False  # an invalid value outside the masks
    values = values[..., 0] if channels == 1 else values
    return values, dissolved, masks, rev


class _Counting(darsia.CpuBackend):
    """The CPU backend, counting the calls of its operations."""

    def __init__(self):
        self.calls = []

    def remap(self, *args):
        self.calls.append("remap")
        return super().remap(*args)

    def sand_means(self, *args):
        self.calls.append("sand_means")
        return super().sand_means(*args)


def test_methods_compute_on_the_process_backend(restore_backend):
    """The image mapping and the sand means run on the backend in use, with the
    CPU's numbers; the CPU is the default."""
    assert darsia.compute_backend().label == "cpu"
    mapping, image = _mapping(np.float32)
    values, dissolved, masks, rev = _means(np.float32)
    expected = (
        mapping.correct_array(image),
        darsia.sand_mean(values, dissolved, masks, rev),
    )
    counting = _Counting()
    darsia.use_compute_backend(counting)

    computed = (
        mapping.correct_array(image),
        darsia.sand_mean(values, dissolved, masks, rev),
    )

    assert counting.calls == ["remap", "remap", "sand_means"]
    for ours, reference in zip(computed, expected):
        np.testing.assert_array_equal(ours, reference)
