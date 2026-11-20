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


def _phase_frame(dtype):
    """A frame with a plume, its baseline with grain texture, its previous frame,
    invalid pixels, and a model of two heads with random coefficients on all
    features, centred on the frame's mean features, with feature scales spread
    log-uniformly from 1e-4 to 0.3, as small as a delivered model's."""
    rng = np.random.default_rng(7)
    rows, cols = np.mgrid[0:90, 0:120] / 100.0
    light = (0.4 + 0.1 * np.sin(3 * cols) * np.cos(2 * rows))[..., None]
    grains = rng.uniform(0.85, 1.15, (90, 120, 3))
    baseline = light * grains * np.array([0.9, 0.8, 0.6])
    plume = np.exp(-((cols - 0.6) ** 2 + (rows - 0.4) ** 2) / 0.05)[..., None]
    previous = baseline * (1 + 0.4 * plume * np.array([0.3, -0.1, -0.6]))
    current = baseline * (1 + 0.8 * plume * np.array([0.4, -0.15, -0.7]))
    current, baseline, previous = (
        x.astype(dtype) for x in (current, baseline, previous)
    )
    current[40:43, 60:64, 0] = np.nan
    baseline[70, 30] = np.nan
    features = darsia.PhaseFeatures(
        current, baseline, previous, texture_sigma_mm=0.334, pixel_size_mm=0.167
    )
    mean = features.matrix(np.s_[:])[features.support.ravel()].mean(axis=0)
    heads = [
        darsia.LogisticHead(
            feature_mean=mean.astype(np.float64),
            feature_scale=10 ** rng.uniform(-4, np.log10(0.3), 36),
            coefficients=scale * rng.normal(0, 0.6, 36),
            intercept=0.0,
        )
        for scale in (1.0, 0.08)
    ]
    model = darsia.PhaseModel(*heads, 0.334, 0.66, None)
    return model, current, baseline, previous


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

    def phase_probabilities(self, *args, **kwargs):
        self.calls.append("phase_probabilities")
        return super().phase_probabilities(*args, **kwargs)


def test_methods_compute_on_the_process_backend(restore_backend):
    """The image mapping, the sand means and the phase probabilities run on the
    backend in use, with the CPU's numbers; the CPU is the default."""
    assert darsia.compute_backend().label == "cpu"
    mapping, image = _mapping(np.float32)
    values, dissolved, masks, rev = _means(np.float32)
    model, *frame = _phase_frame(np.float32)

    def computed():
        return (
            mapping.correct_array(image),
            darsia.sand_mean(values, dissolved, masks, rev),
            *darsia.phase_probabilities(model, *frame, pixel_size_mm=0.167),
        )

    expected = computed()
    counting = _Counting()
    darsia.use_compute_backend(counting)

    ours = computed()

    assert counting.calls == ["remap", "remap", "sand_means", "phase_probabilities"]
    for result, reference in zip(ours, expected):
        np.testing.assert_array_equal(result, reference)


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_opencl_filters_have_opencvs_bits(opencl_backends, dtype):
    """The device's separable filters give gaussian_blur's bits on widths around
    OpenCV's vectors of eight: 17 and 37 taps, and in double precision also 3 and
    5; in single precision a filter of at most five taps is refused."""
    cl = darsia.utils.gpu.opencl.cl
    rng = np.random.default_rng(4)
    sigmas = (2.0, 4.5) + ((0.3, 0.5) if dtype == np.float64 else ())
    for cols in (7, 8, 9, 16, 24, 100):
        image = rng.uniform(0, 1, (23, cols)) * rng.choice([1, 1e-3], (23, cols))
        image = image.astype(dtype)
        for sigma in sigmas:
            taps = darsia.gaussian_taps(sigma, dtype)
            expected = darsia.gaussian_blur(image, sigma)
            for backend in opencl_backends:
                filtered = np.empty_like(image)
                with backend._operation() as made:
                    source = backend._upload(made, image)
                    target, scratch = (
                        backend._allocate(made, image.nbytes) for _ in range(2)
                    )
                    program = backend._program(dtype)
                    backend._filtered(
                        made,
                        program,
                        (source, 0),
                        (target, 0),
                        scratch,
                        taps,
                        image.shape,
                    )
                    cl.enqueue_copy(backend._queue, filtered, target)
                np.testing.assert_array_equal(
                    filtered, expected, err_msg=backend.device
                )
    if dtype == np.float32:
        model, *frame = _phase_frame(dtype)
        heads = (model.co2_presence, model.gas_within_co2)
        for backend in opencl_backends:
            with pytest.raises(ValueError, match="5 taps"):
                backend.phase_probabilities(
                    *frame, texture_sigma_mm=0.334, pixel_size_mm=1.0, heads=heads
                )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_opencl_phase_probabilities_within_the_tolerance(opencl_backends, dtype):
    """With and without a previous frame, the probabilities are the CPU's within
    the stated tolerance, on the CPU's support."""
    model, current, baseline, previous = _phase_frame(dtype)
    heads = (model.co2_presence, model.gas_within_co2)
    for earlier in (previous, None):
        frame = (current, baseline, earlier)
        sizes = {"texture_sigma_mm": 0.334, "pixel_size_mm": 0.167, "heads": heads}
        expected = darsia.CpuBackend().phase_probabilities(*frame, **sizes)
        for backend in opencl_backends:
            for ours, reference in zip(
                backend.phase_probabilities(*frame, **sizes), expected
            ):
                known = np.isfinite(reference)
                assert ours.dtype == dtype and known.sum() > 0.9 * known.size
                np.testing.assert_array_equal(np.isfinite(ours), known)
                np.testing.assert_allclose(
                    ours[known], reference[known], rtol=0, atol=darsia.PHASE_TOLERANCE
                )


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
