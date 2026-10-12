"""Source-bound optical-only photo geometry for the first FF_AC review model.

This cohort intentionally has no physical facies, permeability, injection
position, or mass interpretation.  The source-bound B-photo sand core is moved
to each *native* frame by its certified source H; photos and review boxes stay
in their original coordinates.  Everything outside that core is unresolved.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.ffac_v5_frame_context import (
    SHAPE, V5FrameGeometryCohort, _normal_h, _source_photo_grid,
    project_source_photo_safe_core, photo_safe_core_policy_metadata,
)


CONTRACT = "ffac-v5-track1-photo-geometry-frame-v1"
COHORT_CONTRACT = "ffac-v5-track1-photo-geometry-cohort-v1"
PHOTO_CORE_CONTRACT = "ffac-track1-exact-revision-safe-B-photo-core-v1"
PHOTO_SCOPE = "transported_safe_B_sand_core_only"


def _marker_policy(marker: dict[str, Any]) -> str:
    policy = marker.get("enclosed_hole_policy", "keep")
    expected = photo_safe_core_policy_metadata(policy)
    for key, value in expected.items():
        if key in marker and marker[key] != value:
            raise ValueError("Track1 photo cohort support policy metadata changed")
    if policy != "keep" and not all(key in marker for key in expected):
        raise ValueError("Track1 filled photo cohort lacks its explicit support contract")
    return policy


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _check(path: Any, digest: Any, label: str) -> Path:
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Track1 {label} lacks exact source path/SHA")
    source = Path(path)
    if not source.is_file() or sha256_file(source) != digest:
        raise ValueError(f"Track1 {label} changed: {source}")
    return source


def _same_path(first: Any, second: Any) -> bool:
    return Path(str(first)).resolve() == Path(str(second)).resolve()


def _same_h(actual: np.ndarray, expected: np.ndarray) -> bool:
    return bool(np.allclose(actual, expected, rtol=0.0, atol=1e-7))


class Track1FramePhotoCohort:
    """One exact 52-revision cohort row, reconstructed in parent or worker."""

    def __init__(self, native_manifest_path: str | Path) -> None:
        native_path = Path(native_manifest_path)
        native = _read(native_path)
        marker = native.get("v5_track1_photo_geometry") or {}
        if marker.get("contract") != COHORT_CONTRACT:
            raise ValueError("Track1 native manifest lacks its photo-only cohort")
        self.enclosed_hole_policy = _marker_policy(marker)
        source_path = _check(marker.get("original_source_manifest_path"),
                             marker.get("original_source_manifest_sha256"),
                             "original native source manifest")
        source = _read(source_path)
        count = source.get("frame_count")
        rows = marker.get("frames")
        if (native.get("run") != source.get("run") or native.get("frame_count") != count
                or tuple(source.get("analysis_image_shape_hw") or ()) != SHAPE
                or (native.get("v5_photometry") or {}).get("source_manifest_sha256")
                   != marker["original_source_manifest_sha256"]
                or not isinstance(rows, list) or len(rows) != count
                or [row.get("frame_index") for row in rows] != list(range(count))):
            raise ValueError("Track1 photo cohort does not match its native revision")
        self.native_manifest_path = native_path
        self.native_manifest_sha256 = sha256_file(native_path)
        self.source_manifest_path = source_path
        self.source_manifest_sha256 = marker["original_source_manifest_sha256"]
        self.source = source
        self.rows = rows
        self.shape_hw = SHAPE

    def frame(self, index: int) -> dict[str, Any]:
        """Live-hash the receipt and exact frame/photo sources before use."""
        if not isinstance(index, int) or not 0 <= index < len(self.rows):
            raise ValueError("Track1 frame index is outside its source timeline")
        row = self.rows[index]
        receipt_path = _check(row.get("receipt_path"), row.get("receipt_sha256"),
                              "frame transport receipt")
        receipt = _read(receipt_path)
        original = self.source["frame_sources"][index]
        native_source = Path(original.get("frame_path") or
                             self.source_manifest_path.parent / f"frame_{index:03d}.jpg")
        if (receipt.get("contract") != CONTRACT
                or receipt.get("original_source_manifest_sha256") != self.source_manifest_sha256
                or not _same_path(receipt.get("original_source_manifest_path"),
                                  self.source_manifest_path)
                or receipt.get("frame_index") != index
                or receipt.get("native_shape_hw") != list(self.shape_hw)
                or receipt.get("validation_only") is not True
                or receipt.get("mass_eligible") is not False
                or receipt.get("photo_sand_transport_qa_pass") is not True
                or receipt.get("temporal_registration_pass") is not True
                or receipt.get("optical_geometry_qa_pass") is not True
                or receipt.get("photo_sand_qa_scope") != PHOTO_SCOPE
                or receipt.get("photo_sand_qa_pass") is not True
                or receipt.get("fixed_native_mask_erosion_px") != 8
                or receipt.get("training_label_eligibility_certified") is not False
                or receipt.get("phase_map_eligibility_certified") is not False
                or receipt.get("rig_metal_certified") is not False
                or receipt.get("dynamic_overflow_certified") is not False
                or not _same_path(receipt.get("native_review_jpeg_path"), native_source)
                or receipt.get("native_review_jpeg_sha256") !=
                   (original.get("native_jpeg_sha256") or original.get("frame_sha256"))):
            raise ValueError("Track1 frame receipt or source revision is inconsistent")
        if not (self.source["run"] == "ac61" and index == 0):
            if (not _same_path(receipt.get("camera_jpeg_path"), original.get("source_path"))
                    or receipt.get("camera_jpeg_sha256") !=
                       (original.get("source_sha256") or original.get("source_raw_sha256"))):
                raise ValueError("Track1 camera JPEG differs from exact source frame")
        else:
            # AC61 frame 0 is a separately documented three-ARW prepared
            # source, not one same-stem camera JPEG.  Keep the original frozen
            # source SHA live-bound as well as its actual camera member below.
            _check(original.get("source_path"),
                   original.get("source_sha256") or original.get("source_raw_sha256"),
                   "AC61 historical prepared frame")
        _check(receipt["native_review_jpeg_path"],
               receipt["native_review_jpeg_sha256"], "native review frame")
        _check(receipt.get("camera_jpeg_path"), receipt.get("camera_jpeg_sha256"),
               "camera JPEG acquisition")
        _check(receipt.get("v5_B_native_jpeg_path"),
               receipt.get("v5_B_native_jpeg_sha256"), "V5 B native review")
        _check(receipt.get("v5_B_registration_certificate_path"),
               receipt.get("v5_B_registration_certificate_sha256"),
               "RAW-to-B registration certificate")
        _check(receipt.get("v5_B_arw_path"), receipt.get("v5_B_arw_sha256"),
               "V5 B RAW acquisition")
        _check(receipt.get("v5_B_camera_jpeg_path"),
               receipt.get("v5_B_camera_jpeg_sha256"),
               "V5 B camera JPEG")
        temporal_path = _check(receipt.get("temporal_geometry_receipt_path"),
                               receipt.get("temporal_geometry_receipt_sha256"),
                               "temporal source registration")
        temporal = _read(temporal_path)
        frame_evidence = temporal.get("frames") or []
        if (index >= len(frame_evidence)
                or frame_evidence[index].get("frame_index") != index
                or not _same_h(_normal_h(frame_evidence[index].get(
                    "H_timeline0_to_frame_native")),
                    _normal_h(receipt.get("H_timeline0_to_frame_native")))):
            raise ValueError("Track1 temporal receipt differs from this frame H")
        if frame_evidence[index].get("temporal_registration_pass") is not True:
            # The one original quarter-res gate failure is preserved as FAIL.
            # A separately SHA-bound, source-only V3 exception accepts that
            # *same* H for AC51 frame 18 in validation-only scope.
            exception_path = _check(
                receipt.get("ac51_f18_explicit_v3_exception_receipt_path"),
                receipt.get("ac51_f18_explicit_v3_exception_receipt_sha256"),
                "AC51 frame-18 source-only H exception")
            exception = _read(exception_path)
            if (self.source.get("run") != "ac51" or index != 18
                    or exception.get("contract") !=
                    "ffac-v5-ac51-f18-temporal-fallback-result-v3"
                    or exception.get("source_manifest_sha256") !=
                    self.source_manifest_sha256
                    or exception.get("original_camera_frame_jpeg_sha256") !=
                    receipt["camera_jpeg_sha256"]
                    or exception.get("frame_native_sha256") !=
                    receipt["native_review_jpeg_sha256"]
                    or exception.get("original_primary_gate_pass") is not False
                    or exception.get("fallback_decision_pass") is not True
                    or exception.get("validation_only") is not True
                    or exception.get("mass_eligible") is not False
                    or not _same_h(_normal_h(exception.get(
                        "selected_H_timeline0_to_frame_native")),
                        _normal_h(receipt["H_timeline0_to_frame_native"]))):
                raise ValueError("Track1 temporal exception is absent or mis-scoped")
        photo_path = _check(receipt.get("photo_sand_reference_receipt_path"),
                            receipt.get("photo_sand_reference_receipt_sha256"),
                            "B-photo source acceptance")
        photo = _read(photo_path)
        mask_path = _check(receipt.get("photo_sand_reference_mask_path"),
                           receipt.get("photo_sand_reference_mask_sha256"),
                           "B-photo source mask")
        if (photo.get("contract") != PHOTO_CORE_CONTRACT
                or photo.get("source_manifest_sha256") != self.source_manifest_sha256
                or photo.get("B_native_review_jpeg_sha256") !=
                   receipt["v5_B_native_jpeg_sha256"]
                or photo.get("B_arw_sha256") != receipt["v5_B_arw_sha256"]
                or photo.get("B_camera_jpeg_sha256") !=
                   receipt["v5_B_camera_jpeg_sha256"]
                or photo.get("B_registration_certificate_sha256") !=
                   receipt["v5_B_registration_certificate_sha256"]
                or photo.get("mask_npz_sha256") !=
                   receipt["photo_sand_reference_mask_sha256"]
                or not _same_path(photo.get("mask_npz_path"), mask_path)
                or photo.get("photo_gate_pass") is not True
                or photo.get("photo_sand_qa_scope") != PHOTO_SCOPE
                or photo.get("per_frame_transport_qa_pass") is not False
                or photo.get("training_label_eligibility_certified") is not False
                or photo.get("phase_map_eligibility_certified") is not False
                or photo.get("rig_metal_certified") is not False
                or photo.get("dynamic_overflow_certified") is not False
                or photo.get("mass_eligible") is not False):
            raise ValueError("Track1 B-photo acceptance was promoted beyond its source scope")
        h_time = _normal_h(receipt.get("H_timeline0_to_frame_native"))
        h_b0 = _normal_h(receipt.get("H_v5_B_to_timeline0_native"))
        h_bf = _normal_h(receipt.get("H_v5_B_to_frame_native"))
        h_ref0 = _normal_h(receipt.get("H_ac28_to_timeline0_native"))
        h_reff = _normal_h(receipt.get("H_ac28_to_frame_native"))
        if (not _same_h(h_bf, _normal_h(h_time @ h_b0))
                or not _same_h(h_reff, _normal_h(h_time @ h_ref0))):
            raise ValueError("Track1 per-frame B/photo homography composition changed")
        receipt["receipt_path"] = str(receipt_path)
        receipt["receipt_sha256"] = row["receipt_sha256"]
        # Original receipts remain immutable evidence for the transported
        # conservative core.  The manifest explicitly binds the derived
        # support policy without upgrading that original QA scope.
        receipt.update(photo_safe_core_policy_metadata(self.enclosed_hole_policy))
        return receipt

    def photo(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the manifest's explicit B-sand support and its complement."""
        receipt = self.frame(index)
        sand = _source_photo_grid(receipt["photo_sand_reference_mask_path"],
                                  receipt["photo_sand_reference_mask_sha256"])
        safe, support = project_source_photo_safe_core(
            sand, receipt["H_ac28_to_frame_native"], shape_hw=self.shape_hw,
            enclosed_hole_policy=self.enclosed_hole_policy, return_metadata=True)
        support["photo_sand_support_mask_sha256"] = sha256(
            np.packbits(safe, bitorder="little").tobytes()).hexdigest()
        support["source_transport_receipt_sha256"] = receipt["receipt_sha256"]
        support["native_manifest_sha256"] = self.native_manifest_sha256
        support["frame_index"] = index
        receipt.update(support)
        receipt["photo_sand_support_receipt_sha256"] = sha256(json.dumps(
            support, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode()).hexdigest()
        return safe, ~safe, receipt

    def H_v5_B_to_frame(self, index: int) -> np.ndarray:
        return _normal_h(self.frame(index)["H_v5_B_to_frame_native"])

    def H_previous_to_frame(self, previous_index: int, index: int) -> np.ndarray:
        return _normal_h(self.H_v5_B_to_frame(index) @
                         np.linalg.inv(self.H_v5_B_to_frame(previous_index)))

    def feature_support(self, index: int, previous_index: int | None,
                        linear_input: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reuse the same SHA-bound B/previous support and 8px feature guard."""
        return V5FrameGeometryCohort.feature_support(
            self, index, previous_index, linear_input)


def bind_photo_only_marker(source_manifest_path: str | Path,
                           row: dict[str, Any], *,
                           enclosed_hole_policy: str = "keep") -> dict[str, Any]:
    """Preflight a frozen aggregate row before any corrected frame is written.

    This uses the same live-source verifier as the eventual materialized native
    cohort, without constructing a fake physical-context manifest.
    """
    source_path = Path(source_manifest_path)
    policy = photo_safe_core_policy_metadata(enclosed_hole_policy)
    source_sha = sha256_file(source_path)
    source = _read(source_path)
    frames = row.get("frames")
    if (row.get("source_manifest_sha256") != source_sha
            or not _same_path(row.get("source_manifest_path"), source_path)
            or row.get("run") != source.get("run")
            or not isinstance(frames, list)
            or len(frames) != int(source.get("frame_count") or -1)
            or [r.get("frame_index") for r in frames] != list(range(len(frames)))):
        raise ValueError("Track1 photo aggregate does not cover this exact source revision")
    probe = Track1FramePhotoCohort.__new__(Track1FramePhotoCohort)
    probe.source_manifest_path = source_path
    probe.source_manifest_sha256 = source_sha
    probe.source = source
    probe.rows = frames
    probe.shape_hw = tuple(source["analysis_image_shape_hw"])
    probe.enclosed_hole_policy = enclosed_hole_policy
    for index in range(len(frames)):
        probe.frame(index)
    marker = dict(contract=COHORT_CONTRACT,
                original_source_manifest_path=str(source_path),
                original_source_manifest_sha256=source_sha,
                frames=[dict(frame_index=index,
                             receipt_path=str(item["receipt_path"]),
                             receipt_sha256=str(item["receipt_sha256"]))
                        for index, item in enumerate(frames)])
    # Preserve the exact old marker when using the default conservative core.
    if enclosed_hole_policy != "keep":
        marker.update(policy)
    return marker


__all__ = ["CONTRACT", "COHORT_CONTRACT", "Track1FramePhotoCohort",
           "bind_photo_only_marker"]
