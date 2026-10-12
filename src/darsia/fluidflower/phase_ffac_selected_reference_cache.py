"""Bounded reuse of the unchanged scientific flat-field reference preparation.

This is opt-in execution machinery. Every lookup hashes the actual input array
bytes, including its validity mask. It never reuses a registered image, feature
map, probability, or temporal state. The authoritative preparation function is
called unchanged on a miss and its already-readonly output is reused on a hit.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from unittest.mock import patch

import numpy as np

CONTRACT = "ffac-exact-flat-reference-worker-cache-v1"


def array_key(value):
    """Hash logical C-order bytes without a full-image temporary copy."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Reference cache cannot hash object arrays")
    digest = sha256()
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    elif array.ndim == 0:
        digest.update(array.tobytes())
    else:
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
    return array.shape, array.dtype.str, digest.hexdigest()


class FlatReferenceCache:
    def __init__(self, prepare, *, maximum_bytes=512 * 1024**2):
        if not isinstance(maximum_bytes, int) or maximum_bytes < 1:
            raise ValueError("Reference cache needs a positive byte bound")
        self.prepare = prepare
        self.maximum_bytes = maximum_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def __call__(self, signal, *, source_valid_mask=None, tile_rows=128):
        # Keep the original public argument validation even on a warm hit.
        # None and an explicitly all-valid mask remain different keys.
        if not isinstance(tile_rows, int) or isinstance(tile_rows, bool) or tile_rows < 1:
            raise ValueError("tile_rows must be a positive integer")
        globals_ = getattr(self.prepare, "__globals__", {})
        display = globals_.get("display")
        matrix = None if display is None else getattr(display, "M_NATURAL", None)
        key = (array_key(signal), array_key(source_valid_mask), tile_rows, array_key(matrix))
        if key in self.entries:
            self.hits += 1
            result, size = self.entries.pop(key)
            self.entries[key] = (result, size)
            return result
        self.misses += 1
        result = self.prepare(signal, source_valid_mask=source_valid_mask, tile_rows=tile_rows)
        if result.linear_rgb.flags.writeable or result.valid.flags.writeable:
            raise ValueError("Authoritative reference preparation must return readonly arrays")
        size = result.linear_rgb.nbytes + result.valid.nbytes
        if size <= self.maximum_bytes:
            while self.entries and self.bytes + size > self.maximum_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
                self.evictions += 1
            self.entries[key] = (result, size)
            self.bytes += size
        return result

    def receipt(self):
        return dict(contract=CONTRACT, maximum_bytes=self.maximum_bytes,
                    resident_bytes=self.bytes, cache_hits=self.hits,
                    cache_misses=self.misses, evictions=self.evictions,
                    input_key="actual-C-order-bytes-shape-dtype-mask-tile-rows-and-colour-matrix",
                    cached_output="readonly-unregistered-linear-reference-and-valid-mask",
                    probabilities_cached=False, frame_geometry_cached=False,
                    numerical_function_changed=False)


@contextmanager
def flat_reference_cache(*, maximum_bytes=512 * 1024**2):
    from . import phase_gas_flatfield as flat
    if isinstance(flat.prepare_flatfield_reference, FlatReferenceCache):
        raise ValueError("Nested flat reference caches are not supported")
    cache = FlatReferenceCache(flat.prepare_flatfield_reference, maximum_bytes=maximum_bytes)
    with patch.object(flat, "prepare_flatfield_reference", cache):
        yield cache

