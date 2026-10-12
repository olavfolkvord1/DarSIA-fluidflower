"""Bounded, exact decoding reuse for the two fixed selected-image references.

Integration is explicit: NativeCentreEvaluator.infer's source_baseline_log
decode uses ``fixed_temporal_baseline`` when RGB retention is opted in. Its
discarded flat_baseline_log decode uses key-only ``validate_fixed``. Current and previous decodes
retain the original decoder. This helper does not install patches or change
any accepted inference module. Its hashes describe in-memory arrays, not
source files; existing source admission and SHA seals remain necessary.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Callable

import numpy as np

CONTRACT = "ffac-selected-fixed-baseline-exact-decode-cache-v1"
FIXED_ROLES = frozenset({"fixed_temporal_baseline", "fixed_native_flat_baseline"})


def _array_sha256(array: np.ndarray) -> str:
    """Hash exact C-order bytes without allocating a full strided-array copy."""
    digest = hashlib.sha256()
    if array.nbytes == 0:
        return digest.hexdigest()
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    else:
        iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"], order="C", buffersize=256 * 1024)
        for block in iterator:
            digest.update(block.tobytes())
    return digest.hexdigest()


class FixedBaselineDecodeCache:
    """One worker's two-entry byte-bounded cache; hits never share writable RGB."""

    def __init__(self, *, max_bytes: int = 512 * 1024 * 1024):
        if type(max_bytes) is not int or not 0 <= max_bytes <= 1024 * 1024 * 1024:
            raise ValueError("Fixed-reference decode cache requires a 0..1 GiB byte budget")
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._delegated = self._too_large = 0
        self._validated = OrderedDict()
        self._validation_hits = self._validation_misses = 0
        self._bindings = {}

    def decode(self, role: str, log, valid, *, decoder: Callable | None = None):
        """Use the original decoder on cold, invalid, current and previous calls.

        The exact input and mask byte hashes are checked on every lookup. A
        changed array cannot retrieve a value decoded from its previous bytes.
        Unsupported representations are delegated, preserving original errors.
        """
        if decoder is None:
            from .phase_ffac_selected_native_centre import decode_native_log
            decoder = decode_native_log
        source, support = np.asarray(log), np.asarray(valid)
        if (role not in FIXED_ROLES or self.max_bytes == 0
                or source.dtype != np.float32 or source.ndim != 3 or source.shape[0] != 3
                or support.dtype != bool or support.shape != source.shape[1:]):
            self._delegated += 1
            return decoder(log, valid)
        log_sha, support_sha = _array_sha256(source), _array_sha256(support)
        # Keep the decoder itself in the key so alternate conventions cannot
        # reuse results. It is strongly retained by the key for the cache life.
        key = (decoder, source.shape, source.dtype.str, log_sha, support_sha)
        self._bindings[role] = dict(log_array_sha256=log_sha,
            support_array_sha256=support_sha, shape_chw=list(source.shape),
            input_dtype=source.dtype.str, support_dtype=support.dtype.str)
        if key in self._entries:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key].copy()
        self._misses += 1
        result = decoder(log, valid)
        # The accepted decoder returns HWC float32. Never reinterpret another
        # result convention or an input changed during the cold call.
        if (not isinstance(result, np.ndarray) or result.dtype != np.float32
                or result.shape != (*support.shape, 3)
                or _array_sha256(source) != log_sha
                or _array_sha256(support) != support_sha):
            return result
        if result.nbytes > self.max_bytes:
            self._too_large += 1
            return result
        while self._entries and (len(self._entries) >= 2
                or self._bytes + result.nbytes > self.max_bytes):
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.nbytes
        saved = result.copy()
        saved.setflags(write=False)
        self._entries[key] = saved
        self._bytes += saved.nbytes
        return result

    def validate_fixed(self, role: str, log, valid, *, decoder: Callable | None = None):
        """Skip only an already validated, discarded fixed-flat decode.

        This retains two exact hash keys and no decoded RGB. The accepted
        decoder is still called for every cold, changed or invalid input.
        """
        if decoder is None:
            from .phase_ffac_selected_native_centre import decode_native_log
            decoder = decode_native_log
        source, support = np.asarray(log), np.asarray(valid)
        if (role != "fixed_native_flat_baseline" or source.dtype != np.float32
                or source.ndim != 3 or source.shape[0] != 3
                or support.dtype != bool or support.shape != source.shape[1:]):
            decoder(log, valid)
            return None
        log_sha, support_sha = _array_sha256(source), _array_sha256(support)
        key = (decoder, source.shape, source.dtype.str, log_sha, support_sha)
        self._bindings[role] = dict(log_array_sha256=log_sha,
            support_array_sha256=support_sha, shape_chw=list(source.shape),
            input_dtype=source.dtype.str, support_dtype=support.dtype.str)
        if key in self._validated:
            self._validated.move_to_end(key)
            self._validation_hits += 1
            return None
        self._validation_misses += 1
        decoder(log, valid)
        if _array_sha256(source) == log_sha and _array_sha256(support) == support_sha:
            while len(self._validated) >= 2:
                self._validated.popitem(last=False)
            self._validated[key] = None
        return None

    def clear(self):
        self._entries.clear()
        self._validated.clear()
        self._bytes = 0

    def receipt(self) -> dict:
        return dict(contract=CONTRACT, fixed_roles=sorted(FIXED_ROLES),
            cached_entries=len(self._entries), cached_bytes=self._bytes,
            max_bytes=self.max_bytes, hits=self._hits, misses=self._misses,
            delegated_calls=self._delegated, oversized_results=self._too_large,
            validation_hits=self._validation_hits, validation_misses=self._validation_misses,
            validation_keys=len(self._validated), validation_rgb_retained_bytes=0,
            array_bindings=dict(self._bindings), cached_values_readonly=True,
            hits_return_fresh_mutable_copy=True, source_file_verification_replaced=False,
            numerical_decoder_changed=False,
            accepted_running_snapshot_changed=False)


__all__ = ["FixedBaselineDecodeCache", "FIXED_ROLES", "CONTRACT"]
