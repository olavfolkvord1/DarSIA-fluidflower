"""The OpenCL compute backend: the operations of a backend on a GPU.

The remap and the sand means repeat the CPU backend's arithmetic in the same order
(:class:`darsia.CpuBackend`):

- the remap OpenCV's bilinear sampling with float maps: each coordinate rounded to
  1/32 pixel, the four weights of that fraction in single precision, summed in a
  fixed order;
- the sand means SciPy's square filter of each axis in turn: a running sum in double
  precision along each line, its mean taken at each pixel, and the values of the
  dtype between the axes; then the mask's mean divided by the mean of the mask.

The phase probabilities repeat the phase features step by step: the images without
invalid pixels, their mean colours and changes, and the support, with the CPU's
bits; the Gaussian filters in OpenCV's order, with its bits; and the 36 features and
the heads, whose sums run in another order than BLAS's, so the probabilities differ
from the CPU's in the last bits (:data:`darsia.PHASE_TOLERANCE`).

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

import cv2
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

/* NumPy's maximum and minimum: a NaN first operand propagates. */
static inline real larger(const real a, const real b)
{
    return (a >= b || isnan(a)) ? a : b;
}

static inline real smaller(const real a, const real b)
{
    return (a <= b || isnan(a)) ? a : b;
}

/* The index of OpenCV's default border (gfedcb | abcdefgh | gfedcba). */
static inline int reflected_101(int index, const int length)
{
    if (length == 1) return 0;
    while (index < 0 || index >= length)
        index = index < 0 ? -index : 2 * (length - 1) - index;
    return index;
}

/* The images without invalid pixels, their validity, their mean colours, the norm
   of the change against the baseline and its square, and the norm of the change
   against the previous frame, as PhaseFeatures computes them. */
__kernel void phase_prepare(__global real *current, __global real *baseline,
                            __global real *previous, const int dynamic,
                            const int count, __global uchar *valid,
                            __global real *gray, __global real *norms)
{
    const int pixel = get_global_id(0);
    if (pixel >= count) return;
    uchar all_valid = 1;
    for (int i = 0; i < 2 + dynamic; i++) {
        __global real *rgb = (i == 0 ? current : i == 1 ? baseline : previous) + 3 * pixel;
        const uchar finite = isfinite(rgb[0]) && isfinite(rgb[1]) && isfinite(rgb[2]);
        all_valid &= finite;
        if (!finite) rgb[0] = rgb[1] = rgb[2] = (real)0;
        gray[i * count + pixel] = ((rgb[0] + rgb[1]) + rgb[2]) / (real)3;
    }
    valid[pixel] = all_valid;
    __global const real *now = current + 3 * pixel;
    for (int i = 0; i < 1 + dynamic; i++) {
        __global const real *other = (i == 0 ? baseline : previous) + 3 * pixel;
        const real d0 = now[0] - other[0], d1 = now[1] - other[1], d2 = now[2] - other[2];
        const real norm = sqrt(((d0 * d0) + (d1 * d1)) + (d2 * d2));
        norms[2 * i * count + pixel] = norm;
        if (i == 0) norms[count + pixel] = norm * norm;
    }
}

/* One pass of cv2.sepFilter2D with a symmetric kernel, in OpenCV's order: along
   the rows, every tap in turn; along the columns, the centre and then each pair of
   rows at the same distance. In the first `fused` columns OpenCV's vector code
   multiplies and adds in one rounding (FMA), in the others in two. */
__kernel void filter_pass(__global const real *source, const int source_at,
                          __global real *target, const int target_at,
                          __global const real *taps, const int width,
                          const int rows, const int cols, const int along_rows,
                          const int fused)
{
    const int col = get_global_id(0), row = get_global_id(1);
    if (col >= cols || row >= rows) return;
    source += source_at;
    target += target_at;
    const int reach = width / 2;
    const bool once = col < fused;
    real sum;
    if (along_rows) {
        __global const real *line = source + row * cols;
        sum = once ? fma(line[reflected_101(col - reach, cols)], taps[0], (real)0)
                   : taps[0] * line[reflected_101(col - reach, cols)];
        for (int k = 1; k < width; k++) {
            const real value = line[reflected_101(col + k - reach, cols)];
            sum = once ? fma(value, taps[k], sum) : sum + taps[k] * value;
        }
    } else {
        const real centre = source[row * cols + col];
        sum = once ? fma(centre, taps[reach], (real)0) : centre * taps[reach] + (real)0;
        for (int k = 1; k <= reach; k++) {
            const real pair = source[reflected_101(row + k, rows) * cols + col]
                              + source[reflected_101(row - k, rows) * cols + col];
            sum = once ? fma(pair, taps[reach + k], sum) : sum + pair * taps[reach + k];
        }
    }
    target[row * cols + col] = sum;
}

/* One pass of the erosion by a square; beyond the border nothing erodes. */
__kernel void erode_pass(__global const uchar *source, __global uchar *target,
                         const int width, const int rows, const int cols,
                         const int along_rows)
{
    const int col = get_global_id(0), row = get_global_id(1);
    if (col >= cols || row >= rows) return;
    const int reach = width / 2;
    uchar kept = 1;
    for (int k = -reach; k <= reach && kept; k++) {
        const int c = along_rows ? col + k : col, r = along_rows ? row : row + k;
        if (c >= 0 && c < cols && r >= 0 && r < rows) kept = source[r * cols + c];
    }
    target[row * cols + col] = kept;
}

/* The 36 features of PhaseFeatures.matrix and both heads' probabilities. */
__kernel void phase_heads(__global const real *current, __global const real *baseline,
                          __global const real *previous, const int dynamic,
                          __global const real *gray, __global const real *smooth,
                          __global const real *norms, __global const real *statistics,
                          __global const uchar *support, __global const real *heads,
                          const int count, __global real *co2, __global real *gas)
{
    const int pixel = get_global_id(0);
    if (pixel >= count) return;
    if (!support[pixel]) {
        co2[pixel] = (real)NAN;
        gas[pixel] = (real)NAN;
        return;
    }
    const real epsilon = (real)1e-4, floor_ = (real)0.05;
    __global const real *now = current + 3 * pixel, *then = baseline + 3 * pixel;
    __global const real *before = previous + 3 * pixel;
    real intensity[3], contrast[3], texture[3];
    for (int i = 0; i < 2 + dynamic; i++) {
        intensity[i] = gray[i * count + pixel];
        contrast[i] = intensity[i] - smooth[i * count + pixel];
        texture[i] = fabs(contrast[i]);
    }
    const real sum_now = ((now[0] + now[1]) + now[2]) + epsilon;
    const real sum_then = ((then[0] + then[1]) + then[2]) + epsilon;
    const real local_mean = statistics[pixel];
    const real spread = statistics[count + pixel] - local_mean * local_mean;
    const real local_std = sqrt(larger(spread, (real)0));
    const real bright_now = larger(contrast[0], (real)0);
    const real bright_then = larger(contrast[1], (real)0);
    const real rel_now = smaller(
        larger(contrast[0] / larger(smooth[pixel], floor_), (real)-1), (real)4);
    const real rel_then = smaller(
        larger(contrast[1] / larger(smooth[count + pixel], floor_), (real)-1), (real)4);
    const real rel_bright_now = larger(rel_now, (real)0);
    const real rel_bright_then = larger(rel_then, (real)0);
    const real persistent = smaller(rel_bright_then, rel_bright_now);
    real f[36];
    for (int c = 0; c < 3; c++) f[c] = now[c] - then[c];
    f[3] = intensity[0] - intensity[1];
    for (int c = 0; c < 3; c++) f[4 + c] = now[c] / sum_now - then[c] / sum_then;
    for (int k = 0; k < 3; k++) {
        const int i = k == 1 ? 1 : 0, j = k == 0 ? 1 : 2;
        const real ratio_now = larger(now[i], epsilon) / larger(now[j], epsilon);
        const real ratio_then = larger(then[i], epsilon) / larger(then[j], epsilon);
        f[7 + k] = log(ratio_now) - log(ratio_then);
    }
    const real texture_delta = texture[0] - texture[1];
    f[10] = texture_delta;
    f[11] = fabs(texture_delta);
    f[12] = norms[pixel];
    f[13] = local_mean;
    f[14] = local_std;
    f[15] = local_mean / (local_std + epsilon);
    f[16] = intensity[1];
    f[17] = contrast[1];
    f[18] = contrast[0];
    f[19] = contrast[0] - contrast[1];
    f[20] = smaller(bright_then, bright_now);
    f[21] = larger(bright_now - bright_then, (real)0);
    f[22] = rel_then;
    f[23] = rel_now;
    f[24] = rel_now - rel_then;
    f[25] = persistent;
    f[26] = larger(rel_bright_now - rel_bright_then, (real)0);
    f[27] = persistent * persistent / (larger(rel_bright_then, rel_bright_now) + floor_);
    for (int k = 28; k < 36; k++) f[k] = (real)0;
    if (dynamic) {
        f[28] = (real)1;
        for (int c = 0; c < 3; c++) f[29 + c] = now[c] - before[c];
        f[32] = intensity[0] - intensity[2];
        f[33] = texture[0] - texture[2];
        f[34] = norms[2 * count + pixel];
        f[35] = statistics[2 * count + pixel];
    }
    for (int h = 0; h < 2; h++) {
        __global const real *head = heads + h * 109;
        real logit = (real)0;
        for (int k = 0; k < 36; k++)
            logit += ((f[k] - head[k]) / head[36 + k]) * head[72 + k];
        logit = smaller(larger(logit + head[108], (real)-30), (real)30);
        const real probability = (real)1 / ((real)1 + exp(-logit));
        if (h == 0) co2[pixel] = probability;
        else gas[pixel] = probability;
    }
}
"""
"""Kernels of the operations, for float32 values or, with ``DOUBLE_VALUES``,
float64."""

_FLOATING = (np.dtype(np.float32), np.dtype(np.float64))
"""The value dtypes the kernels take."""

_GROUP = {1: (64,), 2: (32, 8)}
"""Work-group size of the kernels by the dimension of their work; the drivers'
own choice was up to 50 times slower on grids whose sizes have few divisors."""

_OPENCV_AVX2 = 11
"""OpenCV's number of the processor feature AVX2 (``cv::CpuFeatures``)."""

_SMALL_FILTER = 5
"""In single precision OpenCV sums symmetric filters of at most this many taps in
an order of their own, which these kernels do not repeat."""


def _opencv_fuses() -> bool:
    """Whether OpenCV's filters on this host multiply and add in one rounding: its
    AVX2 code, which OpenCV builds with FMA3, runs where AVX2 is enabled (not
    switched off by ``OPENCV_CPU_DISABLE``); its SSE code adds in two roundings."""
    return cv2.checkHardwareSupport(_OPENCV_AVX2)


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
        self.hardware = {
            "backend": self.name,
            "platform": device.platform.name,
            "model": device.name,
            "driver": device.driver_version,
        }
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

    def _filtered(self, made, program, source, target, scratch, taps, shape):
        """OpenCV's separable filter of a plane with taps: rows, then columns.

        The source and the target are a buffer and the index of a plane in it. In
        single precision OpenCV's vector code fuses multiply and add, where it runs
        its AVX2 code, in the columns of whole groups of eight; double precision has
        no vector code.
        """
        rows, cols = shape
        width = np.int32(len(taps))
        fused = taps.dtype == np.float32 and _opencv_fuses()
        fused = np.int32(cols // 8 * 8 if fused else 0)
        weights = self._upload(made, taps)
        count = rows * cols
        for (read, i), (write, j), along_rows in (
            (source, (scratch, 0), 1),
            ((scratch, 0), target, 0),
        ):
            self._run(
                program["filter_pass"],
                (cols, rows),
                read,
                np.int32(i * count),
                write,
                np.int32(j * count),
                weights,
                width,
                np.int32(rows),
                np.int32(cols),
                np.int32(along_rows),
                fused,
            )

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
        (:meth:`darsia.CpuBackend.phase_probabilities`).

        The images' preparation, the support and the filters give the CPU's bits;
        the heads sum their terms in another order than BLAS, and the device's
        logarithms and exponentials round otherwise, so the probabilities differ
        from the CPU's in the last bits (:data:`darsia.PHASE_TOLERANCE`), on the
        CPU's support.

        Parameters
        ----------
        current, baseline : array
            Linear RGB images of shape (rows, cols, 3) on one grid.
        previous : array or None
            Linear RGB image of the run's previous frame on the same grid.
        texture_sigma_mm : float
            Standard deviation (mm) of the Gaussian defining the image texture.
        pixel_size_mm : float
            Pixel size of the images' grid.
        heads : tuple of darsia.LogisticHead
            The CO2-presence and the gas-within-CO2 head.

        Returns
        -------
        tuple of array
            The two probabilities (rows, cols) in the images' floating dtype.

        Raises
        ------
        ValueError
            If the images differ in shape or are not RGB, or, in single precision,
            a filter has at most five taps (a texture of less than about 0.56
            pixels), which OpenCV sums in an order of its own.

        """
        given = [image for image in (current, baseline, previous) if image is not None]
        if len({image.shape for image in given}) != 1:
            raise ValueError("The images of a frame differ in shape.")
        if current.ndim != 3 or current.shape[2] != 3:
            raise ValueError("The images of a frame must be RGB.")
        dtype = np.result_type(*given).type
        shape = current.shape[:2]
        count = shape[0] * shape[1]
        sigma, spatial, width = darsia.phase_feature_sizes(
            current.shape, texture_sigma_mm, pixel_size_mm, dtype
        )
        texture_taps = darsia.gaussian_taps(sigma, dtype)
        spatial_taps = darsia.gaussian_taps(spatial, dtype)
        smallest = min(len(texture_taps), len(spatial_taps))
        if dtype == np.float32 and smallest <= _SMALL_FILTER:
            raise ValueError(
                f"A filter of {smallest} taps (texture of {sigma:.3g} pixels) is "
                "summed by OpenCV in an order of its own; compute on the CPU."
            )
        dynamic = previous is not None
        plane = count * np.dtype(dtype).itemsize
        co2, gas = (np.empty(shape, dtype) for _ in range(2))
        with self._operation() as made:
            program = self._program(dtype)
            images = [self._upload(made, np.asarray(image, dtype)) for image in given]
            images += images[:1] * (3 - len(images))  # no previous frame: never read

            def planes(number):
                """A device buffer of a number of the frame's planes."""
                return self._allocate(made, number * plane)

            gray, smooth, norms, local, scratch = (planes(n) for n in (3, 3, 3, 3, 1))
            valid, support, eroded = (self._allocate(made, count) for _ in range(3))
            n = np.int32(count)
            self._run(
                program["phase_prepare"],
                (count,),
                *images,
                np.int32(dynamic),
                n,
                valid,
                gray,
                norms,
            )
            for i in range(2 + dynamic):
                self._filtered(
                    made, program, (gray, i), (smooth, i), scratch, texture_taps, shape
                )
            for i in range(2 + dynamic):
                self._filtered(
                    made, program, (norms, i), (local, i), scratch, spatial_taps, shape
                )
            rows, cols = shape
            for read, write, along_rows in ((valid, eroded, 1), (eroded, support, 0)):
                self._run(
                    program["erode_pass"],
                    (cols, rows),
                    read,
                    write,
                    np.int32(width),
                    np.int32(rows),
                    np.int32(cols),
                    np.int32(along_rows),
                )
            parameters = np.concatenate(
                [
                    np.concatenate(
                        (
                            head.feature_mean,
                            head.feature_scale,
                            head.coefficients,
                            [head.intercept],
                        )
                    )
                    for head in heads
                ]
            ).astype(dtype)
            targets = [self._allocate(made, plane) for _ in range(2)]
            self._run(
                program["phase_heads"],
                (count,),
                *images,
                np.int32(dynamic),
                gray,
                smooth,
                norms,
                local,
                support,
                self._upload(made, parameters),
                n,
                *targets,
            )
            for result, target in zip((co2, gas), targets):
                cl.enqueue_copy(self._queue, result, target)
        return co2, gas
