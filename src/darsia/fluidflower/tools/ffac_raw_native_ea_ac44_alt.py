"""Source-exact AC44 DSC00002 alternate Sony lens-table renderer.

The existing FF_AC EA renderer and geometry modules remain byte-for-byte
unchanged.  One original ARW has a different embedded distortion/CA table.
Only its exact ARW/JPEG/native-review source triple may use that table; every
other source is delegated to the frozen renderer with its original checks.

The parent renderer bytecode is executed with a *call-local* geometry overlay,
not a process-global monkeypatch. This is safe for parallel materialization.
The early intrinsic parameters remain unchanged, as independently checked on
the source image in the source-only AC44 f18 diagnostic receipt.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import Mapping, Sequence

import numpy as np
import tifffile

from darsia.fluidflower.tools import ffac_raw_geometry as geo
from darsia.fluidflower.tools import ffac_raw_native_ea as original


_ORIGINAL_RENDERER_SHA256 = "041e65a5437ecae4237f518ef08b58e2becfbb4e62d45dc9bb8ead07c194b559"
_ORIGINAL_GEOMETRY_SHA256 = "66c8df7fb12c9d8f4c3b0baad180cfb4199467b97b1706a30125b7d90f25ca4d"
_ALT_SHA = {
    "arw_sha256": "d7ea7f5a986186adc27527e82f9abd10f568ac72aad439fa44af3b36bcf50ce8",
    "camera_jpeg_sha256": "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c",
    "review_native_jpeg_sha256": "5c4d2c2fe1f1a805396f809c356f2a63bdbcf3edfa8d85fa0a218b6718584d8e",
}
_ALT_DISTORTION = (704, 684, 615, 497, 331, 126, -110, -359, -600, -824, -1034)
_ALT_CA_RED = (0, 640, 512, 512, 512, 640, 640, 640, 768, 768, 768)
_ALT_CA_BLUE = (0, 384, 512, 640, 640, 512, 384, 256, 256, 256, 384)


def _check_parent_code() -> None:
    if geo._sha256(Path(original.__file__)) != _ORIGINAL_RENDERER_SHA256:
        raise ValueError("Frozen FF_AC EA renderer source changed")
    if geo._sha256(Path(geo.__file__)) != _ORIGINAL_GEOMETRY_SHA256:
        raise ValueError("Frozen FF_AC RAW geometry source changed")


def _exact_alternate_tags(arw_path: Path):
    if geo._sha256(arw_path) != _ALT_SHA["arw_sha256"]:
        raise ValueError("Alternate Sony lens table is not authorized for this ARW")
    with tifffile.TiffFile(arw_path) as tiff:
        sub = tiff.pages[0].pages[0]
        d = tuple(int(v) for v in sub.tags[0x7037].value)
        ca = tuple(int(v) for v in sub.tags[0x7035].value)
    if (d[0] != 11 or tuple(d[1:12]) != _ALT_DISTORTION
            or ca[0] != 22 or tuple(ca[1:12]) != _ALT_CA_RED
            or tuple(ca[12:23]) != _ALT_CA_BLUE):
        raise ValueError("AC44 source Sony distortion/CA tags differ from exact exception")
    return _ALT_DISTORTION, _ALT_CA_RED, _ALT_CA_BLUE


class _AlternateGeo:
    """Call-local replacement for two geometry attributes only."""

    _DISTORTION = {**geo._DISTORTION, "early": _ALT_DISTORTION}

    @staticmethod
    def _sony_tags(arw_path: Path):
        return _exact_alternate_tags(Path(arw_path))

    def __getattr__(self, name: str):
        return getattr(geo, name)


def _render_exact_alternate(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    _check_parent_code()
    if {key: str(expected_source_hashes.get(key, "")).lower() for key in _ALT_SHA} != _ALT_SHA:
        raise ValueError("Alternate Sony lens source triple differs from exact AC44 frame")
    # Reuse the frozen full-resolution EA renderer bytecode with an isolated
    # global dictionary. No process-global attr is changed, even transiently.
    globals_for_call = dict(original.registered_edge_aware_raw_linear.__globals__)
    globals_for_call["geo"] = _AlternateGeo()
    frozen_call = FunctionType(
        original.registered_edge_aware_raw_linear.__code__,
        globals_for_call,
        original.registered_edge_aware_raw_linear.__name__,
        original.registered_edge_aware_raw_linear.__defaults__,
        original.registered_edge_aware_raw_linear.__closure__,
    )
    image, certificate = frozen_call(
        arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
    )
    if (certificate["sources"] != _ALT_SHA
            or tuple(certificate["sony_distortion_table"]) != _ALT_DISTORTION
            or tuple(certificate["sony_ca_red_table"]) != _ALT_CA_RED
            or tuple(certificate["sony_ca_blue_table"]) != _ALT_CA_BLUE):
        raise ValueError("Alternate Sony lens certificate did not bind exact source/tags")
    certificate = dict(certificate)
    certificate["contract"] = "ffac-source-bound-full-res-edge-aware-camera-linear-ac44-alt-v1"
    certificate["source_exact_alternate_lens"] = {
        "source_arw_sha256": _ALT_SHA["arw_sha256"],
        "parent_renderer_sha256": _ORIGINAL_RENDERER_SHA256,
        "parent_geometry_sha256": _ORIGINAL_GEOMETRY_SHA256,
        "exception_module_sha256": geo._sha256(Path(__file__)),
        "profile_intrinsics": "unchanged early FF_AC profile",
        "parallel_execution": "call-local geometry namespace, no module mutation",
    }
    return image, certificate


def registered_edge_aware_raw_linear(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    """Drop-in v2 renderer: exact alternate frame or unchanged frozen v1."""
    _check_parent_code()
    arw_sha = str(expected_source_hashes.get("arw_sha256", "")).lower()
    if arw_sha == _ALT_SHA["arw_sha256"]:
        return _render_exact_alternate(
            arw_path, camera_jpeg_path, review_native_jpeg_path,
            expected_source_hashes=expected_source_hashes, content_box=content_box,
        )
    return original.registered_edge_aware_raw_linear(
        arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
    )


__all__ = ["registered_edge_aware_raw_linear"]
