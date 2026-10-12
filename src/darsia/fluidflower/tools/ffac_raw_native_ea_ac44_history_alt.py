"""Exact historical AC44 f21 alternate-lens renderer on its own review revision.

The ARW/JPEG acquisition is the same DSC00002 used by the current AC44 f18,
but this historical review image has different bytes and coordinates. This
module accepts only that historical source triple. It reuses the already
verified source-exact Sony tag overlay without modifying either frozen module.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import Mapping, Sequence

import numpy as np

from darsia.fluidflower.tools import ffac_raw_geometry as geo
from darsia.fluidflower.tools import ffac_raw_native_ea as original
from darsia.fluidflower.tools import ffac_raw_native_ea_ac44_alt as current_alt


_CURRENT_ALT_RENDERER_SHA256 = "d7d9e942a425a5d1526861737e63d9f4b1a1b0bb0e07c97db924f0f2a13d0497"
_HISTORY_SHA = {
    "arw_sha256": "d7ea7f5a986186adc27527e82f9abd10f568ac72aad439fa44af3b36bcf50ce8",
    "camera_jpeg_sha256": "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c",
    "review_native_jpeg_sha256": "61e1634ac874ed06edf3c5766e96b2e3a1451522dc9412bd5e63313220f12ea4",
}


def _check_dependencies() -> None:
    current_alt._check_parent_code()
    if geo._sha256(Path(current_alt.__file__)) != _CURRENT_ALT_RENDERER_SHA256:
        raise ValueError("Frozen current AC44 alternate-lens adapter changed")


def _historical_alternate(
    arw_path: str | Path,
    camera_jpeg_path: str | Path,
    review_native_jpeg_path: str | Path,
    *,
    expected_source_hashes: Mapping[str, str],
    content_box: Sequence[int],
) -> tuple[np.ndarray, dict]:
    _check_dependencies()
    if {key: str(expected_source_hashes.get(key, "")).lower() for key in _HISTORY_SHA} != _HISTORY_SHA:
        raise ValueError("Historical AC44 alternate lens source triple differs")
    globals_for_call = dict(original.registered_edge_aware_raw_linear.__globals__)
    globals_for_call["geo"] = current_alt._AlternateGeo()
    isolated = FunctionType(
        original.registered_edge_aware_raw_linear.__code__,
        globals_for_call,
        original.registered_edge_aware_raw_linear.__name__,
        original.registered_edge_aware_raw_linear.__defaults__,
        original.registered_edge_aware_raw_linear.__closure__,
    )
    image, certificate = isolated(
        arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
    )
    if (certificate["sources"] != _HISTORY_SHA
            or tuple(certificate["sony_distortion_table"]) != current_alt._ALT_DISTORTION
            or tuple(certificate["sony_ca_red_table"]) != current_alt._ALT_CA_RED
            or tuple(certificate["sony_ca_blue_table"]) != current_alt._ALT_CA_BLUE):
        raise ValueError("Historical AC44 lens certificate differs from exact source")
    certificate = dict(certificate)
    certificate["contract"] = "ffac-source-bound-full-res-edge-aware-camera-linear-ac44-history-alt-v1"
    certificate["source_exact_alternate_lens"] = {
        "arw_sha256": _HISTORY_SHA["arw_sha256"],
        "camera_jpeg_sha256": _HISTORY_SHA["camera_jpeg_sha256"],
        "historical_native_sha256": _HISTORY_SHA["review_native_jpeg_sha256"],
        "current_alt_adapter_sha256": _CURRENT_ALT_RENDERER_SHA256,
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
    """Drop-in: historical AC44 source triple only; all others frozen v1."""
    _check_dependencies()
    if str(expected_source_hashes.get("arw_sha256", "")).lower() == _HISTORY_SHA["arw_sha256"]:
        return _historical_alternate(
            arw_path, camera_jpeg_path, review_native_jpeg_path,
            expected_source_hashes=expected_source_hashes, content_box=content_box,
        )
    return original.registered_edge_aware_raw_linear(
        arw_path, camera_jpeg_path, review_native_jpeg_path,
        expected_source_hashes=expected_source_hashes, content_box=content_box,
    )


__all__ = ["registered_edge_aware_raw_linear"]
