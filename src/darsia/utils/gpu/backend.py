"""Compute backends: the few operations of the methods that a device can take.

A method computes its heaviest operations on the process's backend
(:func:`compute_backend`), never on a backend it names itself; everything else stays
on the CPU. The CPU backend (:class:`darsia.CpuBackend`) is the reference and the
default. The backend is set once per process (:func:`use_compute_backend`), before
the work starts. A backend's label names its device class, which a job records and
locks; its device names the device and its driver, which each result records.
A machine's backend is proposed by a check and a short timing
(:func:`choose_compute_backend`), and a recorded one is found again, checked, by its
label (:func:`backend_named`).

"""

from __future__ import annotations

from collections.abc import Collection
from statistics import median
from time import perf_counter
from typing import Protocol

import numpy as np

import darsia
from darsia.utils.gpu import opencl
from darsia.utils.gpu.cpu import CpuBackend


class ComputeBackend(Protocol):
    """The operations a backend gives, with the numbers of the CPU backend.

    A new backend is one class with these members.

    """

    name: str
    """Name of the backend: ``"cpu"`` or ``"opencl"``."""
    label: str
    """The backend and its device class, which a job records and locks: ``"cpu"``,
    or ``"opencl:"`` and the device's name."""
    device: str
    """The device the operations run on, with its driver; recorded with each
    result."""

    def remap(
        self, planes: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """Bilinear samples of image planes at source coordinates
        (:meth:`darsia.CpuBackend.remap`)."""
        ...

    def sand_means(
        self, values: np.ndarray, masks: list[np.ndarray], rev: darsia.REV
    ) -> np.ndarray:
        """Each mask's mean of the values over a square, on the mask's pixels
        (:meth:`darsia.CpuBackend.sand_means`)."""
        ...

    def phase_probabilities(
        self,
        current: np.ndarray,
        baseline: np.ndarray,
        previous: np.ndarray | None,
        *,
        texture_sigma_mm: float,
        pixel_size_mm: float,
        heads: tuple[darsia.LogisticHead, darsia.LogisticHead],
    ) -> tuple[np.ndarray, np.ndarray]:
        """The probabilities of two logistic heads on a frame's phase features
        (:meth:`darsia.CpuBackend.phase_probabilities`)."""
        ...


_current: list[ComputeBackend] = [CpuBackend()]
"""The process's backend."""


def compute_backend() -> ComputeBackend:
    """The backend the methods of this process compute on.

    Returns
    -------
    ComputeBackend
        The backend set by :func:`use_compute_backend`, the CPU without one.

    """
    return _current[0]


def use_compute_backend(backend: ComputeBackend) -> None:
    """Compute the methods of this process on a backend from now on.

    Parameters
    ----------
    backend : ComputeBackend
        The backend, e.g. :class:`darsia.CpuBackend`.

    """
    _current[0] = backend


PHASE_TOLERANCE = 1e-4
"""Largest difference of a phase probability from the CPU's that a backend may
give. On three real frames of a run, two OpenCL devices gave both heads' probabilities
within 4.9e-5 of the CPU's and changed no label of the phase maps; a difference of
1e-4 in every probability changed at most 28 per million labels and 0.007 % of a
frame's CO2 mass, and that was accepted."""


_CHECK_SHAPE = (64, 97)
"""Rows and columns of the frame on which a device is checked against the CPU; the
last column lies beyond OpenCV's vectors of eight."""
_TIMED_PIXELS = 1 << 21
"""Most pixels of the frame on which devices are timed."""
_TIMED_SQUARE = 30
"""Side in pixels of the timed sand means (5 mm on a 0.17 mm grid)."""


def _square(shape: tuple[int, int], size: int) -> darsia.REV:
    """The square of a sand mean of ``size`` pixels on a grid of a shape."""
    grid = darsia.ScalarImage(
        np.zeros(shape, np.uint8), dimensions=[float(n) for n in shape]
    )
    return darsia.REV(size=float(size), img=grid)


def _masks(shape: tuple[int, int], rng: np.random.Generator) -> list[np.ndarray]:
    """Four disjoint masks of blocks, each of a random 60 % of its block's pixels."""
    rows, cols = np.indices(shape)
    region = (rows // 5 + cols // 7) % 4
    dissolved = rng.uniform(size=shape) < 0.6
    return [dissolved & (region == k) for k in range(4)]


def _check_frame(dtype: type):
    """The check's frame: planes of a smooth light and grains with invalid values,
    the maps of a turned, scaled and jittered remap with entries that are not
    finite, and the masks and square of the sand means."""
    rng = np.random.default_rng(0)
    rows, cols = np.indices(_CHECK_SHAPE)
    light = 0.4 + 0.1 * np.sin(3 * cols / 100) * np.cos(2 * rows / 100)
    grains = rng.uniform(0.85, 1.15, (*_CHECK_SHAPE, 3))
    planes = (light[..., None] * grains * np.array([0.9, 0.8, 0.6])).astype(dtype)
    planes[5, 7] = np.nan
    planes[20, 30, 1] = np.inf
    jitter = rng.normal(0, 0.3, _CHECK_SHAPE)
    map_x = np.asarray(cols * 1.03 + rows * 0.02 - 1 + jitter, np.float32)
    map_y = np.asarray(rows * 1.04 - cols * 0.01 + 1 - jitter, np.float32)
    map_x[3, 4], map_y[6, 8] = np.nan, np.inf
    return planes, map_x, map_y, _masks(_CHECK_SHAPE, rng), _square(_CHECK_SHAPE, 7)


def _timing_frame(shape: tuple[int, int], dtype: type):
    """Values, masks and the square of the timed sand means, on a frame of the
    grid's aspect with at most :data:`_TIMED_PIXELS` pixels."""
    scale = min(1.0, (_TIMED_PIXELS / (shape[0] * shape[1])) ** 0.5)
    small = tuple(max(1, int(n * scale)) for n in shape)
    rng = np.random.default_rng(1)
    values = rng.uniform(0, 1, (*small, 3)).astype(dtype)
    return values, _masks(small, rng), _square(small, _TIMED_SQUARE)


_CHECK_SIZES = {"texture_sigma_mm": 0.334, "pixel_size_mm": 0.167}
"""Sizes of the check's phase features: a texture of 2 pixels, filters of 17 taps."""


def _phase_frame(planes: np.ndarray):
    """The check's phase frame: the planes as baseline, and a current and a previous
    frame with a plume, each with one more invalid pixel."""
    dtype = planes.dtype.type
    rows, cols = np.indices(planes.shape[:2]) / 100
    plume = np.exp(-((cols - 0.5) ** 2 + (rows - 0.3) ** 2) / 0.03)[..., None]
    current = planes * (1 + 0.8 * plume * np.array([0.4, -0.15, -0.7]))
    previous = planes * (1 + 0.4 * plume * np.array([0.3, -0.1, -0.6]))
    current, previous = current.astype(dtype), previous.astype(dtype)
    current[40, 50, 2] = np.nan
    previous[50, 60, 0] = np.inf
    return current, planes, previous


def _check_heads(frame) -> tuple[darsia.LogisticHead, darsia.LogisticHead]:
    """Two heads with random coefficients, centred on the mean features of the
    check's phase frame, with feature scales spread log-uniformly from 1e-4 to 0.3,
    as small as a delivered model's, which amplify the rounding the most."""
    features = darsia.PhaseFeatures(*frame, **_CHECK_SIZES)
    matrix = features.matrix(np.s_[:]).astype(np.float64)
    mean = matrix[features.support.ravel()].mean(axis=0)
    rng = np.random.default_rng(3)
    return tuple(
        darsia.LogisticHead(
            feature_mean=mean,
            feature_scale=10 ** rng.uniform(-4, np.log10(0.3), 36),
            coefficients=weight * rng.normal(0, 0.6, 36),
            intercept=0.0,
        )
        for weight in (1.0, 0.08)
    )


def _agrees(
    backend: ComputeBackend, heads: tuple[darsia.LogisticHead, ...] | None
) -> bool:
    """Whether a backend gives the CPU's numbers on the check's frame, in single and
    double precision: the remap and the sand means bit for bit (values and NaN
    positions), the phase probabilities on the CPU's support and within
    :data:`PHASE_TOLERANCE`, with the given heads or, without, the check's own."""
    cpu = CpuBackend()
    for dtype in (np.float32, np.float64):
        planes, map_x, map_y, masks, rev = _check_frame(dtype)
        for operation, args in (
            ("remap", (planes, map_x, map_y)),
            ("sand_means", (planes, masks, rev)),
        ):
            ours = getattr(backend, operation)(*args)
            reference = getattr(cpu, operation)(*args)
            if not np.array_equal(ours, reference, equal_nan=True):
                return False
        frame = _phase_frame(planes)
        sizes = dict(_CHECK_SIZES, heads=heads or _check_heads(frame))
        for ours, reference in zip(
            backend.phase_probabilities(*frame, **sizes),
            cpu.phase_probabilities(*frame, **sizes),
        ):
            known = np.isfinite(reference)
            if not np.array_equal(known, np.isfinite(ours)) or (
                np.abs(ours[known] - reference[known]).max() > PHASE_TOLERANCE
            ):
                return False
    return True


def backend_named(
    label: str, *, heads: tuple[darsia.LogisticHead, darsia.LogisticHead] | None = None
) -> ComputeBackend:
    """The checked backend of a recorded device class, as a task starts on it.

    ``"cpu"`` gives the CPU backend. ``"opencl:<device name>"`` gives a backend on
    this machine's device of that name once it has passed the check of
    :func:`choose_compute_backend`; nothing is timed.

    Parameters
    ----------
    label : str
        The backend's label (:attr:`ComputeBackend.label`), e.g. a job's.
    heads : tuple of darsia.LogisticHead, optional
        The job's CO2-presence and gas-within-CO2 heads, with which the phase
        probabilities are checked; without them, the check's own heads.

    Returns
    -------
    ComputeBackend
        The backend.

    Raises
    ------
    ValueError
        If the label names no backend, no device of that name is on this machine,
        or none of them gives the CPU's numbers.

    """
    if label == "cpu":
        return CpuBackend()
    kind, _, name = label.partition(":")
    if kind != "opencl" or not name:
        raise ValueError(f"No compute backend is labelled {label!r}.")
    found = [device for device in opencl.opencl_devices() if device.name == name]
    for device in found:
        backend = opencl.OpenCLBackend(device)
        if _agrees(backend, heads):
            return backend
    reason = "none gives the CPU's numbers" if found else "no such device here"
    raise ValueError(f"{label}: {reason}.")


def choose_compute_backend(
    shape: tuple[int, int],
    dtype: type = np.float32,
    *,
    heads: tuple[darsia.LogisticHead, darsia.LogisticHead] | None = None,
    measured: Collection[str] = (),
) -> tuple[ComputeBackend, str]:
    """A machine's proposed backend: the fastest checked OpenCL device of a class
    measured on real frames, or the CPU.

    Each device of :func:`darsia.opencl_devices` first runs every operation on a
    small synthetic frame, in single and double precision, against the CPU: the
    remap and the sand means must give the CPU's bits, the phase probabilities the
    CPU's support and values within :data:`PHASE_TOLERANCE`, with the job's heads or,
    without them, heads whose feature scales reach 1e-4.
    A device that differs or fails is not used: its difference would otherwise
    enter every result unnoticed. The devices that pass are timed on the sand means
    of a synthetic RGB frame of the grid's aspect with at most
    :data:`_TIMED_PIXELS` pixels (one warm-up, then the median of three). Only a
    device whose class (its label) has been measured on real frames is proposed,
    the fastest of them; otherwise the proposal is the CPU.

    Parameters
    ----------
    shape : tuple of int
        Rows and columns of the frames the machine will compute, e.g. a run grid.
    dtype : type
        Floating dtype of those frames.
    heads : tuple of darsia.LogisticHead, optional
        The job's CO2-presence and gas-within-CO2 heads, with which the phase
        probabilities are checked; without them, the check's own heads.
    measured : collection of str
        Labels of the device classes measured on real frames, e.g. from the
        machine's record.

    Returns
    -------
    ComputeBackend
        The proposed backend.
    str
        Why, for the machine's record: the proposed device and its time, and each
        device not used with its reason.

    """
    if opencl.cl is None:
        return CpuBackend(), "cpu: PyOpenCL is not installed"
    devices = opencl.opencl_devices()
    if not devices:
        return CpuBackend(), "cpu: no OpenCL GPU with IEEE single and double precision"
    timed, refused, timing = [], [], None
    for device in devices:
        try:
            backend = opencl.OpenCLBackend(device)
            if not _agrees(backend, heads):
                refused.append(f"{backend.device} differs from the CPU")
                continue
            if timing is None:
                timing = _timing_frame(shape, dtype)
            seconds = []
            for _ in range(4):
                start = perf_counter()
                backend.sand_means(*timing)
                seconds.append(perf_counter() - start)
            timed.append((median(seconds[1:]), backend))
        except Exception as error:  # a failing driver must not stop the choice
            refused.append(f"{device.name} failed: {error}")
    timed.sort(key=lambda entry: entry[0])
    rows, cols = timing[0].shape[:2] if timing else shape
    notes = {}
    for seconds, backend in timed:
        note = (
            f"{backend.device}: {seconds:.3f} s for the sand means of a {rows} x {cols}"
        )
        unmeasured = backend.label not in measured
        notes[backend] = note + (
            " frame, not yet measured on real frames" if unmeasured else " frame"
        )
    proposed = next((b for _, b in timed if b.label in measured), None)
    others = "".join(f"; not used: {n}" for b, n in notes.items() if b is not proposed)
    others += "".join(f"; not used: {reason}" for reason in refused)
    if proposed is None:
        why = (
            "no checked OpenCL device of a measured class"
            if timed
            else ("no OpenCL device gives the CPU's numbers")
        )
        return CpuBackend(), f"cpu: {why}" + others
    return proposed, f"opencl: {notes[proposed]}" + others
