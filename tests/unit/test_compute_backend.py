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


@pytest.fixture(scope="module")
def opencl_backends():
    """A backend on each OpenCL device that can give the CPU's bits."""
    backends = [darsia.OpenCLBackend(device) for device in darsia.opencl_devices()]
    if not backends:
        pytest.skip("No OpenCL device with IEEE single and double precision")
    return backends


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_opencl_maps_images_with_the_cpus_bits(opencl_backends, restore_backend, dtype):
    """The samples of the planes and of their invalid indicator, and so the mapped
    image, are the CPU's, with shared and with channel-wise maps."""
    for channel_maps in (False, True):
        mapping, image = _mapping(dtype, channel_maps)
        values = np.where(np.isfinite(image), image, 0).astype(dtype)
        cpu = darsia.CpuBackend()
        expected = cpu.remap(values, mapping.map_x, mapping.map_y)
        darsia.use_compute_backend(cpu)
        mapped = mapping.correct_array(image)
        for backend in opencl_backends:
            sampled = backend.remap(values, mapping.map_x, mapping.map_y)
            np.testing.assert_array_equal(sampled, expected, err_msg=backend.device)
            darsia.use_compute_backend(backend)
            np.testing.assert_array_equal(
                mapping.correct_array(image), mapped, err_msg=backend.device
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_opencl_sand_means_have_the_cpus_bits(opencl_backends, dtype):
    """RGB and scalar values, odd and even squares, a grid smaller than the square,
    and an invalid value outside the masks, which spreads as on the CPU."""
    cases = (
        _means(dtype),
        _means(dtype, size_mm=8.0, channels=1),
        _means(dtype, shape=(7, 5)),
    )
    for values, dissolved, masks, rev in cases:
        masks = [dissolved & mask for mask in masks.values()]
        expected = darsia.CpuBackend().sand_means(values, masks, rev)
        for backend in opencl_backends:
            means = backend.sand_means(values, masks, rev)
            np.testing.assert_array_equal(means, expected, err_msg=backend.device)


def test_choice_without_a_device_is_the_cpu(monkeypatch):
    """Without an OpenCL device the machine computes on the CPU, and says why."""
    monkeypatch.setattr(darsia.utils.gpu.opencl, "cl", object())
    monkeypatch.setattr(darsia.utils.gpu.opencl, "opencl_devices", lambda: [])

    backend, reason = darsia.choose_compute_backend((60, 80))

    assert backend.name == "cpu"
    assert reason == "cpu: no OpenCL GPU with IEEE single and double precision"


def test_choice_proposes_a_checked_device_of_a_measured_class(
    opencl_backends, monkeypatch
):
    """A device with the CPU's numbers is proposed only once its class has been
    measured on real frames, and is found again by its label; one that differs in
    a single value of the check is not used, and the reason names it."""
    labels = [backend.label for backend in opencl_backends]
    backend, reason = darsia.choose_compute_backend((60, 80))
    assert backend.name == "cpu" and "not yet measured on real frames" in reason
    backend, reason = darsia.choose_compute_backend((60, 80), measured=labels)
    assert backend.name == "opencl" and reason.startswith("opencl:")
    for label in labels:
        assert darsia.backend_named(label).label == label

    def nudged(self, values, masks, rev):
        means = sand_means(self, values, masks, rev)
        means.flat[0] = np.nextafter(means.flat[0], np.inf)
        return means

    sand_means = darsia.OpenCLBackend.sand_means
    monkeypatch.setattr(darsia.OpenCLBackend, "sand_means", nudged)
    backend, reason = darsia.choose_compute_backend((60, 80), measured=labels)

    assert backend.name == "cpu"
    assert reason.count("differs from the CPU") == len(opencl_backends)
    with pytest.raises(ValueError, match="none gives the CPU's numbers"):
        darsia.backend_named(labels[0])


def test_a_backend_is_found_by_its_label():
    """The CPU by its label; a label of no backend, and a device not on this
    machine, are refused."""
    assert darsia.backend_named("cpu").name == "cpu"
    with pytest.raises(ValueError, match="No compute backend"):
        darsia.backend_named("cuda:0")
    with pytest.raises(ValueError, match="no such device here"):
        darsia.backend_named("opencl:no such device")
