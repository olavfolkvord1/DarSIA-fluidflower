"""The OpenCL compute backend: the operations of a backend on a GPU.

The remap and the sand means repeat the CPU backend's arithmetic in the same order
(:class:`darsia.CpuBackend`):

- the remap OpenCV's bilinear sampling with float maps: each coordinate rounded to
  1/32 pixel, the four weights of that fraction in single precision, summed in a
  fixed order;
- the sand means SciPy's square filter of each axis in turn: a running sum in double
  precision along each line, its mean taken at each pixel, and the values of the
  dtype between the axes; then the mask's mean divided by the mean of the mask.

The kernels are built with contraction of multiply and add off and with correctly
rounded single-precision division, so every step is the IEEE operation the CPU
does, on the devices whose arithmetic guarantees these operations
(:func:`opencl_devices`); whether a device gives the CPU's numbers is a check
against the CPU. NaN positions are the CPU's; a GPU may give a NaN another payload.
Each call uploads its arrays, downloads its result and releases its device buffers.
One backend serves one call at a time, since each call sets its kernels' arguments.

PyOpenCL is an optional dependency (``pip install darsia[opencl]``).

"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np

import darsia

try:
    import pyopencl as cl
except ImportError:  # pragma: no cover - the optional dependency is missing
    cl = None

_SOURCE = r"""
#pragma OPENCL FP_CONTRACT OFF
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef DOUBLE_VALUES
typedef double real;
#else
typedef float real;
#endif

/* Sampling as cv2.remap, INTER_LINEAR with float maps and a constant NaN border. */
__kernel void remap_linear(__global const real *source, const int source_rows,
                           const int source_cols, __global const float *map_x,
                           __global const float *map_y, const int map_channels,
                           const int channels, const int rows, const int cols,
                           __global real *target)
{
    const int col = get_global_id(0), row = get_global_id(1);
    if (col >= cols || row >= rows) return;
    const int pixel = row * cols + col;
    for (int channel = 0; channel < channels; channel++) {
        const int at = map_channels == 1 ? pixel : pixel * channels + channel;
        const float x = map_x[at], y = map_y[at];
        real value = (real)NAN;
        if (isfinite(x) && isfinite(y)) {
            const int sx = convert_int_sat_rte(x * 32.0f);
            const int sy = convert_int_sat_rte(y * 32.0f);
            const int ix = clamp(sx >> 5, -32768, 32767);
            const int iy = clamp(sy >> 5, -32768, 32767);
            if (ix >= 0 && ix < source_cols - 1 && iy >= 0 && iy < source_rows - 1) {
                const float ax = (float)(sx & 31) * (1.0f / 32.0f);
                const float ay = (float)(sy & 31) * (1.0f / 32.0f);
                const real w0 = (real)((1.0f - ay) * (1.0f - ax));
                const real w1 = (real)((1.0f - ay) * ax);
                const real w2 = (real)(ay * (1.0f - ax));
                const real w3 = (real)(ay * ax);
                const int top = (iy * source_cols + ix) * channels + channel;
                const int bottom = top + source_cols * channels;
                value = ((source[top] * w0 + source[top + channels] * w1)
                         + source[bottom] * w2) + source[bottom + channels] * w3;
            }
        }
        target[pixel * channels + channel] = value;
    }
}

/* The index of a reflected border (d c b a | a b c d | d c b a), repeated. */
static inline int reflected(int index, const int length)
{
    if (index >= 0 && index < length) return index;
    const int period = 2 * length;
    index %= period;
    if (index < 0) index += period;
    return index < length ? index : period - 1 - index;
}

/* scipy.ndimage.uniform_filter1d along lines, mode reflect, origin 0. */
__kernel void box_lines(__global const real *source, __global real *target,
                        const int length, const int lines, const int along,
                        const int across, const int size)
{
    const int line = get_global_id(0);
    if (line >= lines) return;
    const int start = line * across, behind = size / 2;
    double sum = 0.0;
    for (int k = 0; k < size; k++)
        sum += (double)source[start + reflected(k - behind, length) * along];
    target[start] = (real)(sum / size);
    for (int i = 1; i < length; i++) {
        sum += (double)source[start + reflected(i + size - 1 - behind, length) * along]
               - (double)source[start + reflected(i - 1 - behind, length) * along];
        target[start + i * along] = (real)(sum / size);
    }
}

/* The weight of a mask: one on its pixels, zero elsewhere. */
__kernel void mask_weight(__global const uchar *mask, const int count,
                          __global real *weight)
{
    const int pixel = get_global_id(0);
    if (pixel >= count) return;
    weight[pixel] = mask[pixel] ? (real)1 : (real)0;
}

/* One channel of the values times the weight. */
__kernel void weighted_channel(__global const real *values, __global const real *weight,
                               const int count, const int channels,
                               const int channel, __global real *plane)
{
    const int pixel = get_global_id(0);
    if (pixel >= count) return;
    plane[pixel] = values[pixel * channels + channel] * weight[pixel];
}

/* On the mask's pixels, the mean of the weighted channel over the mean weight. */
__kernel void place_mean(__global const real *mean, __global const real *pore,
                         __global const uchar *mask, const int count,
                         const int channels, const int channel,
                         __global real *values)
{
    const int pixel = get_global_id(0);
    if (pixel >= count || !mask[pixel]) return;
    const real volume = pore[pixel];
    values[pixel * channels + channel] =
        volume < (real)1e-12 ? (real)0 : mean[pixel] / volume;
}
"""
"""Kernels of the operations, for float32 values or, with ``DOUBLE_VALUES``,
float64."""

_FLOATING = (np.dtype(np.float32), np.dtype(np.float64))
"""The value dtypes the kernels take."""

_GROUP = {1: (64,), 2: (32, 8)}
"""Work-group size of the kernels by the dimension of their work; the drivers'
own choice was up to 50 times slower on grids whose sizes have few divisors."""


def opencl_devices() -> list:
    """The OpenCL GPUs whose arithmetic can give the CPU's numbers.

    A device counts if its single precision rounds to nearest, keeps denormals,
    Inf and NaN and divides correctly rounded, and if it has double precision. A
    device whose properties cannot be read is left out. Whether a device gives the
    CPU's numbers is a check against the CPU, not these properties.

    Returns
    -------
    list of pyopencl.Device
        The devices in platform order; empty without PyOpenCL or a platform.

    """
    if cl is None:
        return []
    try:
        platforms = cl.get_platforms()
    except cl.Error:
        return []
    flags = cl.device_fp_config
    required = (
        flags.ROUND_TO_NEAREST
        | flags.DENORM
        | flags.INF_NAN
        | flags.CORRECTLY_ROUNDED_DIVIDE_SQRT
    )
    devices = []
    for platform in platforms:
        try:
            found = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for device in found:
            try:
                single, double = device.single_fp_config, device.double_fp_config
            except cl.Error:
                continue
            if single & required == required and double:
                devices.append(device)
    return devices


class OpenCLBackend:
    """The operations of a backend on one OpenCL device."""

    name = "opencl"
    """Name of the backend."""

    def __init__(self, device) -> None:
        """The backend of one device.

        Parameters
        ----------
        device : pyopencl.Device
            A device of :func:`opencl_devices`.

        """
        self.label = f"opencl:{device.name}"
        """The backend and its device class, which a job records and locks."""
        self.device = " | ".join(
            (device.platform.name, device.name, device.driver_version)
        )
        """The platform, the device and its driver, recorded with each result."""
        self._context = cl.Context([device])
        self._queue = cl.CommandQueue(self._context)
        self._programs = {}
        self._lock = threading.Lock()

    def _program(self, dtype: np.dtype):
        """The kernels for values of a dtype, by name, built once without a disk
        cache."""
        dtype = np.dtype(dtype)
        if dtype not in _FLOATING:
            raise ValueError(f"OpenCL values must be float32 or float64, not {dtype}.")
        if dtype not in self._programs:
            options = ["-cl-fp32-correctly-rounded-divide-sqrt"]
            if dtype == np.float64:
                options.append("-DDOUBLE_VALUES")
            program = cl.Program(self._context, _SOURCE).build(
                options=options, cache_dir=False
            )
            self._programs[dtype] = {
                kernel.function_name: kernel for kernel in program.all_kernels()
            }
        return self._programs[dtype]

    @contextmanager
    def _operation(self):
        """Hold the backend for one call, and release the call's device buffers,
        which it collects in the yielded list, at its end."""
        with self._lock:
            made = []
            try:
                yield made
            finally:
                for buffer in made:
                    buffer.release()

    def _allocate(self, made: list, size: int, array: np.ndarray | None = None):
        """A device buffer of a call: empty, or a copy of an array."""
        flags = cl.mem_flags.READ_WRITE
        if array is not None:
            flags |= cl.mem_flags.COPY_HOST_PTR
            array = np.ascontiguousarray(array)
        made.append(cl.Buffer(self._context, flags, size, hostbuf=array))
        return made[-1]

    def _upload(self, made: list, array: np.ndarray):
        """A device copy of an array, for a call."""
        array = np.ascontiguousarray(array)
        return self._allocate(made, array.nbytes, array)

    def _run(self, kernel, work: tuple[int, ...], *args) -> None:
        """Launch a kernel on work items of a shape, rounded up to whole groups."""
        group = _GROUP[len(work)]
        size = tuple(-(-n // g) * g for n, g in zip(work, group))
        kernel(self._queue, size, group, *args)

    def remap(
        self, planes: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """Bilinear samples of image planes at source coordinates
        (:meth:`darsia.CpuBackend.remap`).

        Parameters
        ----------
        planes : array
            Floating image planes (rows, cols, channels) on the source grid.
        map_x, map_y : array
            Source column and row (float32) of each target pixel, for all planes
            or one map per plane.

        Returns
        -------
        array
            The samples (target rows, target cols, channels).

        Raises
        ------
        ValueError
            If the planes are not of three dimensions, or the maps differ in shape
            or do not give one map per plane.

        """
        if (
            planes.ndim != 3
            or map_x.shape != map_y.shape
            or map_x.ndim not in (2, 3)
            or (map_x.ndim == 3 and map_x.shape[2] != planes.shape[2])
        ):
            raise ValueError("The maps do not fit the planes.")
        rows, cols = map_x.shape[:2]
        channels = planes.shape[2]
        result = np.empty((rows, cols, channels), planes.dtype)
        with self._operation() as made:
            program = self._program(planes.dtype)
            target = self._allocate(made, result.nbytes)
            self._run(
                program["remap_linear"],
                (cols, rows),
                self._upload(made, planes),
                np.int32(planes.shape[0]),
                np.int32(planes.shape[1]),
                self._upload(made, np.asarray(map_x, np.float32)),
                self._upload(made, np.asarray(map_y, np.float32)),
                np.int32(1 if map_x.ndim == 2 else channels),
                np.int32(channels),
                np.int32(rows),
                np.int32(cols),
                target,
            )
            cl.enqueue_copy(self._queue, result, target)
        return result

    def _square_mean(self, program, source, target, scratch, shape, size):
        """SciPy's square mean of a plane: along the columns, then the rows."""
        rows, cols = shape
        for lines, length, along, across, read, write in (
            (cols, rows, cols, 1, source, scratch),
            (rows, cols, 1, cols, scratch, target),
        ):
            self._run(
                program["box_lines"],
                (lines,),
                read,
                write,
                np.int32(length),
                np.int32(lines),
                np.int32(along),
                np.int32(across),
                np.int32(size),
            )

    def sand_means(
        self, values: np.ndarray, masks: list[np.ndarray], rev: darsia.REV
    ) -> np.ndarray:
        """Each mask's mean of the values over a square, on the mask's pixels
        (:meth:`darsia.CpuBackend.sand_means`).

        Parameters
        ----------
        values : array
            Values (rows, cols) or (rows, cols, channels).
        masks : list of array
            Disjoint boolean masks (rows, cols).
        rev : darsia.REV
            The square of the means.

        Returns
        -------
        array
            The values with each mask's pixels replaced by their mean.

        Raises
        ------
        ValueError
            If a mask is not of the values' grid.

        """
        shape = values.shape[:2]
        if values.ndim not in (2, 3) or any(np.shape(m) != shape for m in masks):
            raise ValueError("Each mask must have the shape of the values' grid.")
        count = shape[0] * shape[1]
        n = np.int32(count)
        channels = 1 if values.ndim == 2 else values.shape[2]
        given = np.ascontiguousarray(values)
        result = np.empty_like(given)
        with self._operation() as made:
            program = self._program(values.dtype)
            current = self._upload(made, given)
            plane_bytes = count * given.itemsize
            weight, pore, plane, mean, scratch = (
                self._allocate(made, plane_bytes) for _ in range(5)
            )
            for mask in masks:
                if not mask.any():
                    continue
                pixels = self._upload(made, mask.astype(np.uint8))
                self._run(program["mask_weight"], (count,), pixels, n, weight)
                self._square_mean(program, weight, pore, scratch, shape, rev.size)
                for channel in range(channels):
                    self._run(
                        program["weighted_channel"],
                        (count,),
                        current,
                        weight,
                        n,
                        np.int32(channels),
                        np.int32(channel),
                        plane,
                    )
                    self._square_mean(program, plane, mean, scratch, shape, rev.size)
                    self._run(
                        program["place_mean"],
                        (count,),
                        mean,
                        pore,
                        pixels,
                        n,
                        np.int32(channels),
                        np.int32(channel),
                        current,
                    )
            cl.enqueue_copy(self._queue, result, current)
        return result
