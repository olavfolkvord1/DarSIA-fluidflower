"""Separate historical AC44 revision processor with exact f21 lens exception.

The source-bound historical V4-R5 correction remains bytecode-identical. Only
the exact DSC00002 frame21 review revision uses its Sony distortion/CA tables.
All 22 frames are rendered under a new processor identity in a new output root;
old partial output and the current-AC44 alternate processor are untouched.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import Any

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import ffac_track1_v4r5_historical_processor as base
from darsia.fluidflower.tools import ffac_raw_native_ea_ac44_history_alt as alt


CONTRACT = "ffac-v5-track1-source-bound-native-processor-v1"
VARIANT_CONTRACT = "ffac-v5-track1-ac44-historical-exact-alt-lens-processor-v1"
_BASE_PROCESSOR_SHA256 = "1d3e556529a44a306731a739938a698bda0b45d20bea10f08b3e2dfe2b2a51bc"
_SOURCE_MANIFEST_SHA256 = "e746d4bf8bc9a3d35a22cccf27f8cfd5788a5959e38b702e3269df6d83671a17"
_ALT_ARW_SHA256 = "d7ea7f5a986186adc27527e82f9abd10f568ac72aad439fa44af3b36bcf50ce8"
_ALT_CAMERA_SHA256 = "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c"
_ALT_NATIVE_SHA256 = "61e1634ac874ed06edf3c5766e96b2e3a1451522dc9412bd5e63313220f12ea4"


class Track1HistoricalAC44AltProcessor(base.Track1HistoricalV4R5SourceProcessor):
    """Rerender the 22-frame historical AC44 revision under a third identity."""

    def __init__(self, source_row: dict[str, Any], *,
                 historical_fit_manifest_path: Path,
                 canonical_fit_manifest_path: Path,
                 photo_aggregate_path: Path,
                 enclosed_hole_policy: str = "keep",
                 display_mode: str = "raw_log") -> None:
        self.bound_base_processor_sha256 = (
            _BASE_PROCESSOR_SHA256 if enclosed_hole_policy == "keep"
            and display_mode == "raw_log" else sha256_file(Path(base.__file__)))
        if sha256_file(Path(base.__file__)) != self.bound_base_processor_sha256:
            raise ValueError("Frozen historical V4-R5 processor source changed")
        if (source_row.get("run") != "ac44"
                or source_row.get("source_manifest_sha256") != _SOURCE_MANIFEST_SHA256):
            raise ValueError("Historical AC44 alt-lens processor requires exact revision")
        super().__init__(
            source_row,
            historical_fit_manifest_path=historical_fit_manifest_path,
            canonical_fit_manifest_path=canonical_fit_manifest_path,
            photo_aggregate_path=photo_aggregate_path,
            enclosed_hole_policy=enclosed_hole_policy,
            display_mode=display_mode,
        )
        frame21 = self.cohort.frame(21)
        if (frame21["camera_jpeg_sha256"] != _ALT_CAMERA_SHA256
                or frame21["native_review_jpeg_sha256"] != _ALT_NATIVE_SHA256):
            raise ValueError("Historical AC44 f21 source differs from exact exception")
        self.identity_value.update(
            contract=CONTRACT,
            processor_variant_contract=VARIANT_CONTRACT,
            processor_module_sha256=sha256_file(Path(__file__)),
            base_processor_module_sha256=self.bound_base_processor_sha256,
            renderer_module_sha256=sha256_file(Path(alt.__file__)),
            source_exact_alt_lens_exception={
                "source_manifest_sha256": _SOURCE_MANIFEST_SHA256,
                "frame_index": 21,
                "arw_sha256": _ALT_ARW_SHA256,
                "camera_jpeg_sha256": _ALT_CAMERA_SHA256,
                "native_review_jpeg_sha256": _ALT_NATIVE_SHA256,
                "all_other_frames_use_original_renderer": True,
            },
        )

    def _correct_one(self, **kwargs: Any):
        """Execute frozen historical correction with call-local renderer."""
        if sha256_file(Path(base.__file__)) != self.bound_base_processor_sha256:
            raise ValueError("Frozen historical V4-R5 processor source changed")
        parent = base.Track1HistoricalV4R5SourceProcessor._correct_one
        globals_for_call = dict(parent.__globals__)
        globals_for_call["registered_edge_aware_raw_linear"] = (
            alt.registered_edge_aware_raw_linear
        )
        isolated = FunctionType(parent.__code__, globals_for_call, parent.__name__,
                                parent.__defaults__, parent.__closure__)
        return isolated(self, **kwargs)


__all__ = ["Track1HistoricalAC44AltProcessor"]
