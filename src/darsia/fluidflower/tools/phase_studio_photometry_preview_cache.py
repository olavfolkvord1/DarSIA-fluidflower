"""Regenerable series-owned display cache for geometry-only photographs.

The cache contains lossless RGB PNGs on the *saved comparison grid*, beneath
each experiment series' NAS ``results/photometry_previews`` directory. It is a
viewing aid, not a phase map, optical calibration, or concentration product.
The caller must verify the frozen source/result relationship before asking for
a cached image and must byte-verify the inputs when generating a cache miss.

``key_info`` is a mapping with these fields::

    {
        "dataset": "ffum", "run": "run10", "category": "frame",
        "frame_index": 4, "display_shape_hw": (839, 1536),
        "native_manifest_path": Path(...),
        "native_manifest_sha256": "<frozen SHA-256>",
        "raw_sources": [{"path": Path(...), "sha256": "<frozen SHA-256>"}],
        "rig_path": Path(...),
        "rig_artifacts": [{"path": Path(...), "sha256": "<frozen SHA-256>"}],
        "config_files": [Path(...)],
        "contract_sha256": "<optional SHA-256 of frozen correction contract>",
    }

``rig_artifacts``, ``config_files``, and ``contract_sha256`` are optional. The
ordered raw-source list matters for averaged clear-water references. Current
source/configuration file stats and the complete serialized-rig fingerprint
join the frozen hashes in the key; a changed input yields a miss. A cache hit
avoids loading the rig and decoding its raw JPEG(s).
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image, PngImagePlugin

from darsia.fluidflower.series_paths import series_key, series_path
from darsia.fluidflower.tools.phase_studio_photometry_geometry import (
    serialized_rig_fingerprint,
)


_VERSION = 1
_META_KEY = "darsia_geometry_preview"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_RUN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_CATEGORIES = frozenset({"frame", "baseline", "clearwater"})


def _sha(value: object, field: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return result


def _path(value: object) -> Path:
    return Path(value).expanduser().resolve()


def _stat(path: Path) -> tuple[str, int, int]:
    state = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), state.st_size, state.st_mtime_ns


def _shape(value: object) -> tuple[int, int]:
    try:
        height, width = value
    except (TypeError, ValueError) as exc:
        raise ValueError("display_shape_hw must contain height and width") from exc
    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("display_shape_hw must contain positive integers")
    if int(height) != height or int(width) != width or height <= 0 or width <= 0:
        raise ValueError("display_shape_hw must contain positive integers")
    return int(height), int(width)


def _file_records(rows: object, *, required: bool, label: str) -> list[dict]:
    if rows is None:
        rows = []
    if not isinstance(rows, (list, tuple)) or (required and not rows):
        raise ValueError(f"{label} must be a nonempty ordered list")
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} entries must contain path and sha256")
        path = _path(row["path"])
        records.append({"sha256": _sha(row["sha256"], f"{label}.sha256"),
                        "stat": _stat(path)})
    return records


def _identity(key_info: Mapping) -> tuple[dict, str]:
    """Build the canonical key from frozen identities and current file state."""
    dataset = series_key(key_info["dataset"])
    run = str(key_info["run"]).strip().casefold()
    if not _RUN.fullmatch(run):
        raise ValueError(f"Invalid experiment name: {run!r}")
    category = str(key_info["category"]).strip().casefold()
    if category not in _CATEGORIES:
        raise ValueError(f"Invalid photograph category: {category!r}")
    frame_index = key_info.get("frame_index")
    if category == "frame":
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("Frame preview requires a nonnegative frame_index")
    elif frame_index is not None:
        raise ValueError("Reference previews must use frame_index=None")
    shape = _shape(key_info["display_shape_hw"])
    manifest = _path(key_info["native_manifest_path"])
    rig = _path(key_info["rig_path"])
    if not rig.is_dir():
        raise FileNotFoundError(rig)
    fingerprint = serialized_rig_fingerprint(rig)
    if not fingerprint:
        raise ValueError(f"Serialized geometry rig is empty: {rig}")
    config_paths = key_info.get("config_files") or []
    if not isinstance(config_paths, (list, tuple)):
        raise ValueError("config_files must be a list of paths")
    contract = key_info.get("contract_sha256")
    payload = {
        "version": _VERSION,
        "dataset": dataset,
        "run": run,
        "category": category,
        "frame_index": frame_index,
        "display_shape_hw": shape,
        "native_manifest": {
            "sha256": _sha(key_info["native_manifest_sha256"], "native_manifest_sha256"),
            "stat": _stat(manifest),
        },
        "raw_sources": _file_records(key_info["raw_sources"], required=True,
                                     label="raw_sources"),
        "rig": {"path": str(rig), "fingerprint": fingerprint},
        "rig_artifacts": _file_records(key_info.get("rig_artifacts"), required=False,
                                       label="rig_artifacts"),
        "config_files": [_stat(_path(path)) for path in config_paths],
        "contract_sha256": _sha(contract, "contract_sha256") if contract else None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode("utf-8")
    return payload, sha256(canonical).hexdigest()


def _cache_path(payload: dict, digest: str) -> Path:
    return (series_path(payload["dataset"], "results") / "photometry_previews" /
            payload["run"] / f"v{_VERSION}" / digest[:2] / f"{digest}.png")


def load(key_info: Mapping, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """Load a verified uint8 RGB preview, or return ``None`` for a cache miss.

    Missing/changed input files also give a miss, allowing the caller's normal
    frozen-source validation to produce the authoritative error on regeneration.
    Invalid key schemas and requested shapes raise ``ValueError``.
    """
    expected = _shape(expected_shape)
    if expected != _shape(key_info["display_shape_hw"]):
        raise ValueError("Requested preview shape differs from the saved display grid")
    try:
        payload, digest = _identity(key_info)
    except (FileNotFoundError, OSError):
        return None
    path = _cache_path(payload, digest)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            if (image.format != "PNG" or image.mode != "RGB" or
                    image.size != (expected[1], expected[0])):
                return None
            metadata = json.loads(image.info[_META_KEY])
            if (metadata.get("version") != _VERSION or
                    metadata.get("key_sha256") != digest):
                return None
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if pixels.shape != (*expected, 3):
            return None
        if sha256(pixels.tobytes()).hexdigest() != metadata.get("pixels_sha256"):
            return None
        # A source edited during PNG reading cannot produce a valid hit.
        if _identity(key_info)[1] != digest:
            return None
        return pixels
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def store(key_info: Mapping, rgb: np.ndarray) -> Path:
    """Atomically persist an exact-grid, lossless RGB display preview.

    The caller must have SHA-verified and decoded the source using the frozen
    geometric rig. ``store`` only caches those already verified display pixels.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("Preview must be a uint8 RGB ndarray")
    expected = _shape(key_info["display_shape_hw"])
    if rgb.shape != (*expected, 3):
        raise ValueError("Preview pixels differ from the saved display grid")
    pixels = np.ascontiguousarray(rgb)
    payload, digest = _identity(key_info)
    target = _cache_path(payload, digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": _VERSION,
        "key_sha256": digest,
        "pixels_sha256": sha256(pixels.tobytes()).hexdigest(),
    }
    info = PngImagePlugin.PngInfo()
    info.add_text(_META_KEY, json.dumps(metadata, sort_keys=True,
                                        separators=(",", ":")))
    descriptor, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".png",
                                             dir=target.parent)
    os.close(descriptor)
    try:
        Image.fromarray(pixels).save(
            temporary, format="PNG", pnginfo=info, compress_level=1,
        )
        if _identity(key_info)[1] != digest:
            raise RuntimeError("Preview source or rig changed while writing the cache")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
