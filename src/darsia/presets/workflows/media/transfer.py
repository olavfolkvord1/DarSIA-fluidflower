"""Measured display transfers and a shared, exact population colour range.

These operations produce display pixels or metadata, never scientific maps.
    The caller supplies stored frames and geometry; no experiment names, paths
    or material constants are assumed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb


def divided_size(shape, divisor: int) -> tuple[int, int]:
    """The grid's integer-divided width and height, with no resampling yet."""
    if isinstance(divisor, bool) or int(divisor) != divisor or divisor < 1:
        raise ValueError("The panel divisor must be a positive integer")
    height, width = (int(value) // divisor for value in shape)
    if min(height, width) < 1:
        raise ValueError("The panel divisor leaves an empty panel")
    return width, height


def probe_indices(count: int) -> tuple[int, ...]:
    """Four ordered, distinct probes: one eighth, one quarter, half and last."""
    if count < 1:
        raise ValueError("A film needs at least one frame")
    return tuple(sorted({count // 8, count // 4, count // 2, count - 1}))


def resized_photo(image, size) -> np.ndarray:
    """Downsample corrected RGB using area interpolation; retain unknown pixels."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A photograph needs three RGB channels")
    valid = np.isfinite(image).all(axis=-1)
    small = cv2.resize(
        np.where(valid[..., None], image, 0), size, interpolation=cv2.INTER_AREA
    )
    known = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    small[known < 1] = np.nan
    return small


def first_frame_limits(image, size, percentiles=(0.5, 99.5)) -> tuple[float, float]:
    """Measure the finite first-frame display samples once, across RGB channels."""
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("Photograph percentiles must be ordered in [0, 100]")
    small = resized_photo(image, size)
    values = small[np.isfinite(small)]
    if not values.size:
        raise ValueError("The first photograph has no known display values")
    low, high = np.percentile(values, percentiles)
    if not high > low:
        raise ValueError("The first photograph has no usable intensity range")
    return float(low), float(high)


def stretched_photo(image, size, limits, gamma: float, invalid_grey: int):
    """Use fixed first-frame limits, followed by a power of ``1/gamma``."""
    low, high = limits
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError("Photograph limits must be finite and strictly increasing")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Photograph gamma must be finite and positive")
    small = resized_photo(image, size)
    known = np.isfinite(small).all(axis=-1)
    safe = np.where(known[..., None], small, low)
    # Truncation, rather than rounding, matches the reviewed display producer.
    rgb = (255 * np.clip((safe - low) / (high - low), 0, 1) ** (1 / gamma)).astype(
        np.uint8
    )
    rgb[~known] = invalid_grey
    return rgb


def population_scale(
    samples: Iterable[np.ndarray], *, maximum_samples: int, percentile: float = 99.9
) -> dict:
    """Exact pooled percentile of finite positive samples, in their supplied unit.

    Every run's selected probes must be passed to this ONE call. Values are
    retained only in the upper tail needed for this percentile, bounded by the
    supplied sum of probe grid sizes. Each source array is scanned in chunks.
    The endpoints use a single minimum or maximum. Input arrays are unchanged.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("The positive percentile must be in [0, 100]")
    if (
        isinstance(maximum_samples, bool)
        or maximum_samples < 1
        or int(maximum_samples) != maximum_samples
    ):
        raise ValueError("The maximum sample count must be a positive integer")
    q = percentile / 100
    keep = min(maximum_samples, math.ceil((1 - q) * (maximum_samples - 1)) + 2)
    if q in (0, 1):
        keep = 1
    # One retained tail plus one survivor batch; no growing concatenations or
    # full-tail partition per source chunk. Dropped ties can replace only equal
    # values already in the retained tail, so discarding them is exact.
    buffer = np.empty(2 * keep, dtype=np.float64)
    used = merges = 0
    cutoff = -math.inf
    count = probes = scanned = 0
    minimum = math.inf
    maximum = -math.inf
    for sample in samples:
        probes += 1
        values = np.asarray(sample)
        scanned += values.size
        if scanned > maximum_samples:
            raise ValueError("Probe arrays exceed the declared maximum sample count")
        for chunk in np.nditer(
            values,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            buffersize=65536,
        ):
            positive = np.asarray(
                chunk[np.isfinite(chunk) & (chunk > 0)], dtype=np.float64
            )
            count += positive.size
            if not positive.size:
                continue
            if q == 0:
                minimum = min(minimum, float(positive.min()))
                continue
            if q == 1:
                maximum = max(maximum, float(positive.max()))
                continue
            survivors = positive[positive > cutoff]
            offset = 0
            while offset < survivors.size:
                take = min(buffer.size - used, survivors.size - offset)
                buffer[used : used + take] = survivors[offset : offset + take]
                used += take
                offset += take
                if used == buffer.size:
                    buffer.partition(keep)
                    buffer[:keep] = buffer[keep:]
                    used = keep
                    cutoff = float(buffer[:keep].min())
                    merges += 1
    if not count:
        raise ValueError("The pooled probes have no finite positive mass per area")
    if q == 0:
        value = minimum
    elif q == 1:
        value = maximum
    else:
        rank = (count - 1) * q
        lower, upper = math.floor(rank), math.ceil(rank)
        tail = buffer[:used]
        offset = count - used
        lower, upper = lower - offset, upper - offset
        tail.partition((lower, upper))
        a, b = tail[lower], tail[upper]
        weight = rank - math.floor(rank)
        value = float(
            a + (b - a) * weight if weight < 0.5 else b - (b - a) * (1 - weight)
        )
    return dict(
        colour_scale=value,
        percentile=percentile,
        positive_samples=count,
        probe_frames=probes,
        scope="pooled-population",
        unit="g/m²",
        maximum_samples=maximum_samples,
        retained_samples=used,
        buffer_capacity=buffer.size,
        tail_merges=merges,
    )


def area_map(values, pore_volume, size, scale, *, colour_map, no_value, over_range):
    """Area-resampled values on a fixed square-root scale; no-pore cells are grey.

    Unknown input values propagate through their entire resampling footprint.
    The final boolean is the known display-cell mask for honest clipping counts.
    """
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    small = cv2.resize(np.where(valid, values, 0), size, interpolation=cv2.INTER_AREA)
    known = (
        cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 1
    )
    volume = np.asarray(pore_volume)
    pore = (
        cv2.resize((volume > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
        > 0
    )
    geometry = (
        cv2.resize(
            (np.isfinite(volume) & (volume >= 0)).astype(np.float32),
            size,
            interpolation=cv2.INTER_AREA,
        )
        >= 1
    )
    known &= geometry
    index = (np.sqrt(np.clip(small / scale, 0, 1)) * 255).astype(np.uint8)
    lut = (colormaps[colour_map](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    rgb = lut[index]
    if over_range is not None:
        rgb[known & pore & (small > scale)] = np.rint(
            255 * np.asarray(to_rgb(over_range))
        ).astype(np.uint8)
    rgb[~pore] = 40
    rgb[~known] = np.rint(255 * np.asarray(to_rgb(no_value))).astype(np.uint8)
    return rgb, small, known & pore
