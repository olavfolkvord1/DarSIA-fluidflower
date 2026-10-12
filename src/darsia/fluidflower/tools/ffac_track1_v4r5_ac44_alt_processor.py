"""Isolated AC44 source processor with one exact alternate Sony lens frame.

Reuses the frozen V4-R5 correction *bytecode* and source admission logic. The
only renderer change is a call-local binding to the SHA-gated AC44 adapter.
This variant has a distinct identity and is intended for an isolated complete
AC44 revision root; old partial receipts are never rewritten or relabeled.
"""

from __future__ import annotations

from pathlib import Path
from types import FunctionType
from typing import Any

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import ffac_track1_v4r5_source_processor as base
from darsia.fluidflower.tools import ffac_raw_native_ea_ac44_alt as alt


CONTRACT = base.CONTRACT
VARIANT_CONTRACT = "ffac-v5-track1-ac44-source-exact-alt-lens-processor-v1"
_BASE_PROCESSOR_SHA256 = "af07d7b5e0fe4e7273c3abb0f2cbbf09763d32ca8b3a4e62cd6f58e041e2cce7"
_SOURCE_MANIFEST_SHA256 = "fdcd7f7ff6f8718180db82fbb25ffa788b185c51effefb25d9d8b20f2f01898e"
_ALT_ARW_SHA256 = "d7ea7f5a986186adc27527e82f9abd10f568ac72aad439fa44af3b36bcf50ce8"
_ALT_CAMERA_SHA256 = "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c"
_ALT_NATIVE_SHA256 = "5c4d2c2fe1f1a805396f809c356f2a63bdbcf3edfa8d85fa0a218b6718584d8e"


class Track1V4R5AC44AltProcessor(base.Track1V4R5SourceProcessor):
    """Rerender the complete current AC44 revision under one new identity."""

    def __init__(self, source_row: dict[str, Any], *,
                 calibration_manifest_path: Path,
                 photo_aggregate_path: Path,
                 enclosed_hole_policy: str = "keep",
                 display_mode: str = "raw_log") -> None:
        self.bound_base_processor_sha256 = (
            _BASE_PROCESSOR_SHA256 if enclosed_hole_policy == "keep"
            and display_mode == "raw_log" else sha256_file(Path(base.__file__)))
        if sha256_file(Path(base.__file__)) != self.bound_base_processor_sha256:
            raise ValueError("Frozen current43 V4-R5 processor source changed")
        if (source_row.get("run") != "ac44"
                or source_row.get("source_manifest_sha256") != _SOURCE_MANIFEST_SHA256):
            raise ValueError("AC44 alternate-lens processor is source-revision specific")
        super().__init__(source_row, calibration_manifest_path=calibration_manifest_path,
                         photo_aggregate_path=photo_aggregate_path,
                         enclosed_hole_policy=enclosed_hole_policy,
                         display_mode=display_mode)
        frame18 = self.cohort.frame(18)
        if (frame18["camera_jpeg_sha256"] != _ALT_CAMERA_SHA256
                or frame18["native_review_jpeg_sha256"] != _ALT_NATIVE_SHA256):
            raise ValueError("AC44 frame18 camera/native source differs from exact exception")
        self.identity_value.update(
            contract=CONTRACT,
            processor_variant_contract=VARIANT_CONTRACT,
            processor_module_sha256=sha256_file(Path(__file__)),
            base_processor_module_sha256=self.bound_base_processor_sha256,
            renderer_module_sha256=sha256_file(Path(alt.__file__)),
            source_exact_alt_lens_exception={
                "source_manifest_sha256": _SOURCE_MANIFEST_SHA256,
                "frame_index": 18,
                "arw_sha256": _ALT_ARW_SHA256,
                "camera_jpeg_sha256": _ALT_CAMERA_SHA256,
                "native_review_jpeg_sha256": _ALT_NATIVE_SHA256,
                "all_other_frames_use_original_renderer": True,
            },
        )

    def correct(self, **kwargs: Any):
        """Run exact base correction with a call-local renderer binding."""
        if sha256_file(Path(base.__file__)) != self.bound_base_processor_sha256:
            raise ValueError("Frozen current43 V4-R5 processor source changed")
        parent = base.Track1V4R5SourceProcessor.correct
        isolated_globals = dict(parent.__globals__)
        isolated_globals["registered_edge_aware_raw_linear"] = (
            alt.registered_edge_aware_raw_linear
        )
        isolated = FunctionType(parent.__code__, isolated_globals, parent.__name__,
                                parent.__defaults__, parent.__closure__)
        return isolated(self, **kwargs)


__all__ = ["Track1V4R5AC44AltProcessor"]
