"""Lossless source-backed border strips for moving FF_AC references.

The current photograph and annotations stay on their unchanged native grid.
Only a baseline or previous photograph may be reconstructed with the certified
RAW-derived border before it is aligned to the current frame.  No pixel is
inferred by nearest-edge padding or filled from a phase label.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from uuid import uuid4

import numpy as np


CONTRACT = "ffac-v5-source-backed-extended-reference-ring-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def split_extended_ring(padded_log: np.ndarray, native_log: np.ndarray, *,
                        pad_px: int, maximum_center_delta_log: float = 1e-5
                        ) -> tuple[dict[str, np.ndarray], float]:
    """Extract four exact RAW strips after checking native/padded parity."""
    padded = np.asarray(padded_log)
    native = np.asarray(native_log)
    pad = int(pad_px)
    if (pad < 1 or padded.dtype != np.float32 or native.dtype != np.float32
            or native.ndim != 3 or native.shape[0] != 3
            or padded.shape != (3, native.shape[1] + 2 * pad,
                                native.shape[2] + 2 * pad)):
        raise ValueError("Extended RAW reference has an invalid shape or dtype")
    if np.isinf(padded).any() or np.isinf(native).any():
        raise ValueError("Extended RAW reference contains infinite signal")
    center = padded[:, pad:-pad, pad:-pad]
    if not np.array_equal(np.isfinite(center), np.isfinite(native)):
        raise ValueError("Padded V4 correction changed native source validity")
    finite = np.isfinite(native)
    difference = (float(np.max(np.abs(center[finite] - native[finite])))
                  if np.any(finite) else 0.0)
    if difference > maximum_center_delta_log:
        raise ValueError("Padded V4 correction changed the native source centre")
    strips = {
        "top": np.ascontiguousarray(padded[:, :pad, :]),
        "bottom": np.ascontiguousarray(padded[:, -pad:, :]),
        "left": np.ascontiguousarray(padded[:, pad:-pad, :pad]),
        "right": np.ascontiguousarray(padded[:, pad:-pad, -pad:]),
    }
    return strips, difference


def write_extended_ring(path: Path, padded_log: np.ndarray,
                        native_log: np.ndarray, *, pad_px: int
                        ) -> tuple[str, float]:
    """Write one uncompressed, same-volume atomic ring and return its SHA."""
    path = Path(path)
    strips, difference = split_extended_ring(
        padded_log, native_log, pad_px=pad_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **strips, pad_px=np.asarray(pad_px, np.int16),
                     contract=np.asarray(CONTRACT))
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        if path.exists():
            if file_sha256(path) != digest:
                raise ValueError("Completed V5 source-backed ring changed")
        else:
            os.replace(temporary, path)
        return digest, difference
    finally:
        temporary.unlink(missing_ok=True)


def assemble_extended_log(native_log: np.ndarray, ring_path: Path, *,
                          ring_sha256: str, expected_pad_px: int
                          ) -> np.ndarray:
    """Rebuild the padded log signal from the exact native centre and strips."""
    core = np.asarray(native_log)
    if (core.dtype != np.float32 or core.ndim != 3 or core.shape[0] != 3
            or np.isinf(core).any()):
        raise ValueError("V5 native reference core is not float32 log RGB")
    path = Path(ring_path)
    if file_sha256(path) != ring_sha256:
        raise ValueError("V5 extended RAW ring changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"top", "bottom", "left", "right",
                                  "pad_px", "contract"}:
            raise ValueError("V5 extended RAW ring schema changed")
        pad = int(archive["pad_px"])
        if pad != int(expected_pad_px) or str(archive["contract"]) != CONTRACT:
            raise ValueError("V5 extended RAW ring has another padding contract")
        height, width = core.shape[1:]
        expected = dict(top=(3, pad, width + 2 * pad),
                        bottom=(3, pad, width + 2 * pad),
                        left=(3, height, pad), right=(3, height, pad))
        strips = {}
        for key, shape in expected.items():
            value = np.asarray(archive[key])
            if value.shape != shape or value.dtype != np.float32 or np.isinf(value).any():
                raise ValueError(f"V5 RAW {key} strip is invalid")
            strips[key] = value
    result = np.empty((3, height + 2 * pad, width + 2 * pad), np.float32)
    result[:, :pad, :] = strips["top"]
    result[:, -pad:, :] = strips["bottom"]
    result[:, pad:-pad, :pad] = strips["left"]
    result[:, pad:-pad, -pad:] = strips["right"]
    result[:, pad:-pad, pad:-pad] = core
    return result


__all__ = ["CONTRACT", "split_extended_ring", "write_extended_ring",
           "assemble_extended_log"]
