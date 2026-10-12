"""Exact historical-revision Track1 RAW source processor.

This module is separate from the SHA-frozen current-43 processor.  Its nine
baseline fits are keyed by original source-manifest SHA, never run name, and
use the accepted narrow Track1 B-photo core.  It renders source material only;
fit/evaluation decisions remain in the later 4048-ID admission receipt.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import ffac_v5_observed_sand as obs
from darsia.fluidflower.tools.ffac_raw_native_ea import registered_edge_aware_raw_linear
from darsia.fluidflower.tools.ffac_track1_v4r5_source_processor import (
    CONTRACT, FILLED_DISPLAY_CONTRACT, ISOLATED_DISPLAY_CONTRACT, _chart_xy_baseline_native, _h,
    _source_outputs, _validate_output_policy, _output_display_contract,
)
from darsia.fluidflower.tools.ffac_track1_photographic_display import display_contract
from darsia.fluidflower.tools.ffac_v4r5_native_pilot import correct_frame_native_v4r5
from darsia.fluidflower.tools.ffac_v5_chart_registration import register_frame_chart_xy8
from darsia.fluidflower.tools.ffac_v5_frame_context import photo_safe_core_policy_metadata
from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
    Track1FramePhotoCohort, _source_photo_grid, bind_photo_only_marker,
    project_source_photo_safe_core,
)
from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
    CorrectedFrame,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Track1HistoricalV4R5SourceProcessor:
    """Historical B fit and photo core selected by exact source manifest SHA."""

    def __init__(self, source_row: dict[str, Any], *,
                 historical_fit_manifest_path: Path,
                 canonical_fit_manifest_path: Path,
                 photo_aggregate_path: Path,
                 enclosed_hole_policy: str = "keep",
                 display_mode: str = "raw_log",
                 isolated_invalid_policy: str = "keep") -> None:
        _validate_output_policy(enclosed_hole_policy, display_mode, isolated_invalid_policy)
        self.enclosed_hole_policy = enclosed_hole_policy
        self.display_mode = display_mode
        self.isolated_invalid_policy = isolated_invalid_policy
        self.row = source_row
        self.run = str(source_row["run"])
        self.source_path = Path(source_row["source_manifest_path"])
        self.source_sha = source_row["source_manifest_sha256"]
        if sha256_file(self.source_path) != self.source_sha:
            raise ValueError("Historical source manifest changed")
        self.source = _read(self.source_path)
        self.history_fit_path = Path(historical_fit_manifest_path)
        self.canonical_fit_path = Path(canonical_fit_manifest_path)
        self.photo_aggregate_path = Path(photo_aggregate_path)
        history_fit = _read(self.history_fit_path)
        if (history_fit.get("contract") !=
                "ffac-v4r5-historical9-exact-B-accepted-core-v1"
                or history_fit.get("source_B_photo_core_qa_pass") is not True
                or history_fit.get("training_label_eligibility_certified") is not False
                or history_fit.get("canonical_installation_manifest_sha256") !=
                   sha256_file(self.canonical_fit_path)):
            raise ValueError("Historical B fit lacks accepted source-only core")
        matching = [row for row in history_fit["results"]
                    if row["source_manifest_sha256"] == self.source_sha]
        if len(history_fit["results"]) != 9 or len(matching) != 1 or matching[0]["run"] != self.run:
            raise ValueError("Historical B fit is absent or ambiguous")
        fit_row = matching[0]
        self.fit_path = Path(fit_row["baseline_path"])
        self.fit_sidecar_path = Path(fit_row["sidecar_path"])
        for path, expected in ((self.fit_path, fit_row["baseline_sha256"]),
                               (self.fit_sidecar_path, fit_row["sidecar_sha256"])):
            if sha256_file(path) != expected:
                raise ValueError("Historical exact B fit asset changed")
        self.fit, self.grid = obs.load_baseline(self.fit_path)
        self.reference, ref_grid = obs.load_baseline(
            self.canonical_fit_path.parent / "virtual_reference_baseline.npz")
        if self.grid != ref_grid:
            raise ValueError("Historical fit and canonical reference grids differ")
        marker = bind_photo_only_marker(self.source_path, source_row,
                                       enclosed_hole_policy=self.enclosed_hole_policy)
        cohort = Track1FramePhotoCohort.__new__(Track1FramePhotoCohort)
        cohort.source_manifest_path = self.source_path
        cohort.source_manifest_sha256 = self.source_sha
        cohort.source = self.source
        cohort.rows = marker["frames"]
        cohort.shape_hw = tuple(self.source["analysis_image_shape_hw"])
        cohort.enclosed_hole_policy = self.enclosed_hole_policy
        self.cohort = cohort
        frame0 = cohort.frame(0)
        side_source = _read(self.fit_sidecar_path)["source"]
        if (side_source["source_manifest_sha256"] != self.source_sha
                or side_source["source_arw_sha256"] != frame0["v5_B_arw_sha256"]
                or side_source["source_review_sha256"] != frame0["v5_B_native_jpeg_sha256"]
                or side_source["photo_sand_mask_sha256"] !=
                   frame0["photo_sand_reference_mask_sha256"]
                or side_source.get("photo_sand_qa_pass") is not True
                or fit_row["photo_sand_mask_sha256"] !=
                   frame0["photo_sand_reference_mask_sha256"]):
            raise ValueError("Historical B fit differs from exact accepted source/photo core")
        self.old_field = None
        self.old_field_sha = fit_row.get("old_B_field_sha256")
        if fit_row.get("old_B_field_path") is not None:
            field_path = Path(fit_row["old_B_field_path"])
            if sha256_file(field_path) != self.old_field_sha:
                raise ValueError("Historical old-light scalar field changed")
            with np.load(field_path, allow_pickle=False) as archive:
                self.old_field = np.asarray(archive["field"], np.float32)
        self.identity_value = dict(
            contract=CONTRACT,
            historical_fit_manifest_path=str(self.history_fit_path),
            historical_fit_manifest_sha256=sha256_file(self.history_fit_path),
            canonical_fit_manifest_path=str(self.canonical_fit_path),
            canonical_fit_manifest_sha256=sha256_file(self.canonical_fit_path),
            photo_source_aggregate_path=str(self.photo_aggregate_path),
            photo_source_aggregate_sha256=sha256_file(self.photo_aggregate_path),
            processor_module_sha256=sha256_file(Path(__file__)),
            renderer_module_sha256=sha256_file(Path(registered_edge_aware_raw_linear.__code__.co_filename)),
            photometry_module_sha256=sha256_file(Path(obs.__file__)),
            pilot_math_module_sha256=sha256_file(Path(correct_frame_native_v4r5.__code__.co_filename)),
            validation_only=True, scientific_mass_eligible=False,
            whole_run_B_H_pass=True, donor_tile_cv_pass=False,
            late_yellow_gate_pass=False, weak_phase_gate_status="pending-diagnostic",
            historical_source_photo_core_pass=True,
            historical_label_admission_status="pending-4048-id-partition",
        )
        if enclosed_hole_policy != "keep" or display_mode != "raw_log":
            self.identity_value.update(
                processor_variant_contract=FILLED_DISPLAY_CONTRACT,
                enclosed_hole_policy=enclosed_hole_policy,
                enclosed_hole_max_area_px=60000,
                enclosed_hole_vertical_gap_max_px=(64 if enclosed_hole_policy == "fill_enclosed_v1" else 0),
                photo_sand_mask_module_sha256=sha256_file(
                    Path(project_source_photo_safe_core.__code__.co_filename)),
                photo_sand_support_policy=photo_safe_core_policy_metadata(enclosed_hole_policy),
                display_mode=display_mode,
                display_transfer=_output_display_contract(display_mode),
                full_frame_correction_before_model_mask=True,
            )
        if isolated_invalid_policy != "keep":
            from .ffac_track1_isolated_invalid import policy_contract, fill_isolated_invalid
            self.identity_value.update(processor_variant_contract=ISOLATED_DISPLAY_CONTRACT,
                isolated_invalid_policy=isolated_invalid_policy,
                isolated_invalid_fill=policy_contract(),
                isolated_invalid_module_sha256=sha256_file(Path(fill_isolated_invalid.__code__.co_filename)))

    def identity(self) -> dict[str, Any]:
        return dict(self.identity_value)

    def _correct_one(self, *, frame: dict[str, Any], camera_jpeg: Path,
                     camera_arw: Path, review_jpeg: Path,
                     content_box_xywh: tuple[int, int, int, int],
                     h_for_source: np.ndarray,
                     exact_b: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        expected = dict(arw_sha256=sha256_file(camera_arw),
                        camera_jpeg_sha256=sha256_file(camera_jpeg),
                        review_native_jpeg_sha256=sha256_file(review_jpeg))
        linear, renderer = registered_edge_aware_raw_linear(
            camera_arw, camera_jpeg, review_jpeg,
            expected_source_hashes=expected, content_box=content_box_xywh)
        safe_grid = _source_photo_grid(
            frame["photo_sand_reference_mask_path"],
            frame["photo_sand_reference_mask_sha256"])
        safe, mask_metadata = project_source_photo_safe_core(
            safe_grid, h_for_source, shape_hw=(2979, 5472),
            enclosed_hole_policy=self.enclosed_hole_policy, return_metadata=True)
        chart_certificate = None
        xy = usable = None
        if not exact_b:
            bcert = _read(Path(frame["v5_B_registration_certificate_path"]))
            if bcert.get("sources", {}).get("review_native_jpeg_sha256") != frame[
                    "v5_B_native_jpeg_sha256"]:
                raise ValueError("Historical chart registration belongs to another B")
            h_bf = _h(frame["H_v5_B_to_frame_native"])
            h_ref_frame = _h(frame["H_ac28_to_frame_native"])
            h_ref_b = _h(np.linalg.inv(h_bf) @ h_ref_frame)
            xy, usable, chart_certificate = register_frame_chart_xy8(
                baseline_camera_jpeg=Path(frame["v5_B_camera_jpeg_path"]),
                baseline_camera_sha256=frame["v5_B_camera_jpeg_sha256"],
                frame_camera_jpeg=camera_jpeg,
                frame_camera_sha256=expected["camera_jpeg_sha256"],
                baseline_chart_xy8=_chart_xy_baseline_native(self.fit, self.grid, h_ref_b),
                baseline_chart_log_rgb=self.fit.CB,
                baseline_review_to_camera_h=np.asarray(bcert["review_to_camera_h"], np.float64),
                frame_review_to_camera_h=np.asarray(renderer["review_to_camera_h"], np.float64),
                frame_native_linear_rgb=linear)
        corrected, info = correct_frame_native_v4r5(
            linear, self.fit, self.reference, self.grid, h_for_source,
            (np.ones_like(safe) if self.enclosed_hole_policy != "keep"
             or self.display_mode in ("photographic_v1", "natural_modes_v1") else safe),
            exact_baseline_source=exact_b,
            frame_chart_xy8=xy, frame_chart_usable=usable,
            old_field_ref8=self.old_field)
        return corrected, safe, dict(expected=expected, renderer=renderer,
                                     chart_certificate=chart_certificate,
                                     native_info=info, mask_metadata=mask_metadata)

    def _finish(self, corrected: np.ndarray, safe: np.ndarray,
                certificate: dict[str, Any]) -> CorrectedFrame:
        finite = np.isfinite(corrected).all(axis=0)
        rgb, scientific, effective, auxiliary = _source_outputs(
            corrected, safe, display_mode=self.display_mode,
            isolated_invalid_policy=getattr(self, "isolated_invalid_policy", "keep"),
            return_auxiliary=True)
        if not effective.any() or np.nanmax(scientific) > np.log(65535.0):
            raise ValueError("Historical source lacks nonclipping corrected safe pixels")
        certificate.update(
            native_shape_hw=[2979, 5472],
            source_photo_receipt_sha256=self.cohort.frame(0)[
                "photo_sand_reference_receipt_sha256"],
            fit_baseline_sidecar_sha256=sha256_file(self.fit_sidecar_path),
            old_field_sha256=self.old_field_sha,
            required_valid_mask_sha256=sha256(np.packbits(
                effective, bitorder="little").tobytes()).hexdigest(),
            source_safe_pixels=int(safe.sum()),
            source_safe_corrected_pixels=int(effective.sum()),
            source_safe_unresolved_pixels=int(np.count_nonzero(safe & ~effective)),
            source_safe_corrected_fraction=float(effective.sum() / safe.sum()),
            provisional_gate_status=dict(whole_run_B_H_pass=True,
                donor_tile_cv_pass=False, late_yellow_gate_pass=False,
                weak_phase="pending-diagnostic"))
        if self.enclosed_hole_policy != "keep" or self.display_mode != "raw_log":
            certificate.update(
                display_mode=self.display_mode,
                photographic_display=(_output_display_contract(self.display_mode)
                    if self.display_mode in ("photographic_v1", "natural_modes_v1") else None),
                photographic_full_frame_finite_pixels=int(finite.sum()),
                scientific_sidecar_domain="source-safe-filled-and-finite",
                full_frame_correction_before_model_mask=True,
            )
        if auxiliary["isolated_invalid_fill"] is not None:
            certificate.update(isolated_invalid_fill=auxiliary["isolated_invalid_fill"],
                photo_sand_support_contract=auxiliary["isolated_invalid_fill"]["contract"])
        if auxiliary["display_modes_metadata"] is not None:
            certificate["display_modes_metadata"] = auxiliary["display_modes_metadata"]
        return CorrectedFrame(rgb=rgb,
                              log_linear_rgb=scientific,
                              certificate=certificate,
                              required_valid_mask=effective,
                              filled_invalid_mask=auxiliary["filled_invalid_mask"],
                              display_modes=auxiliary["display_modes"],
                              photo_sand_mask=auxiliary["photo_sand_mask"],
                              display_invalid_mask=auxiliary["display_invalid_mask"])

    def correct(self, *, run: str, frame_id: str, camera_jpeg: Path,
                camera_arw: Path, review_jpeg: Path,
                source_manifest_sha256: str, content_shape_hw: tuple[int, int],
                content_box_xywh: tuple[int, int, int, int]) -> CorrectedFrame:
        if (run != self.run or source_manifest_sha256 != self.source_sha
                or tuple(content_shape_hw) != (2979, 5472)
                or tuple(content_box_xywh) != (416, 496, 5472, 2979)):
            raise ValueError("Historical source or native geometry changed")
        reference = frame_id == "pre_injection_reference"
        if not reference and not frame_id.startswith("frame_"):
            raise ValueError("Unknown historical source frame ID")
        index = 0 if reference else int(frame_id.split("_")[1])
        frame = self.cohort.frame(index)
        expected_jpeg = (frame["v5_B_camera_jpeg_sha256"] if reference
                         else frame["camera_jpeg_sha256"])
        expected_review = (frame["v5_B_native_jpeg_sha256"] if reference
                           else frame["native_review_jpeg_sha256"])
        if (sha256_file(camera_jpeg) != expected_jpeg
                or sha256_file(review_jpeg) != expected_review
                or (reference and sha256_file(camera_arw) != frame["v5_B_arw_sha256"])):
            raise ValueError("Historical acquisition differs from exact source receipt")
        h_bf = _h(frame["H_v5_B_to_frame_native"])
        h_ref_frame = _h(frame["H_ac28_to_frame_native"])
        h_ref_b = _h(np.linalg.inv(h_bf) @ h_ref_frame)
        h_for_source = h_ref_b if reference else h_ref_frame
        exact_b = (sha256_file(camera_arw) == frame["v5_B_arw_sha256"] and
                   sha256_file(camera_jpeg) == frame["v5_B_camera_jpeg_sha256"])
        if reference and not exact_b:
            raise ValueError("Historical reference is not its fitted B acquisition")
        corrected, safe, detail = self._correct_one(
            frame=frame, camera_jpeg=camera_jpeg, camera_arw=camera_arw,
            review_jpeg=review_jpeg, content_box_xywh=content_box_xywh,
            h_for_source=h_for_source, exact_b=exact_b)
        expected = detail["expected"]
        certificate = dict(
            transform_sha256=sha256(json.dumps(detail["renderer"], sort_keys=True,
                                                default=str).encode()).hexdigest(),
            camera_jpeg_sha256=expected["camera_jpeg_sha256"],
            camera_arw_sha256=expected["arw_sha256"],
            original_review_jpeg_sha256=expected["review_native_jpeg_sha256"],
            renderer_certificate=detail["renderer"],
            chart_certificate=detail["chart_certificate"],
            v4r5_native_info=detail["native_info"],
            source_frame_transport_receipt_sha256=frame["receipt_sha256"])
        if self.enclosed_hole_policy != "keep" or self.display_mode != "raw_log":
            certificate.update(detail["mask_metadata"])
        return self._finish(corrected, safe, certificate)

    def correct_composite_baseline(
        self, *, run: str, frame_id: str, members: list[dict[str, Any]],
        review_jpeg: Path, source_manifest_sha256: str,
        content_shape_hw: tuple[int, int],
        content_box_xywh: tuple[int, int, int, int],
        historical_prepared_sha256: str,
    ) -> CorrectedFrame:
        """Distinct source-bound median of the three old AC61 B acquisitions.

        This does not claim pixel-exact reconstruction of the historical
        prepared JPEG.  Frame 0 has no frozen review labels.
        """
        if (self.run != run or run != "ac61" or frame_id != "frame_000"
                or source_manifest_sha256 != self.source_sha
                or len(members) != 3
                or tuple(content_shape_hw) != (2979, 5472)
                or tuple(content_box_xywh) != (416, 496, 5472, 2979)):
            raise ValueError("AC61 historical three-RAW source binding changed")
        frame = self.cohort.frame(0)
        h = _h(frame["H_ac28_to_frame_native"])
        corrected_members = []
        safe_members = []
        certificates = []
        expected_members = []
        for member in members:
            jpeg = Path(member["camera_jpeg"])
            arw = Path(member["camera_arw"])
            if (sha256_file(jpeg) != member["camera_jpeg_sha256"]
                    or sha256_file(arw) != member["camera_arw_sha256"]):
                raise ValueError("AC61 source member changed")
            exact_b = (member["camera_arw_sha256"] == frame["v5_B_arw_sha256"]
                       and member["camera_jpeg_sha256"] == frame["v5_B_camera_jpeg_sha256"])
            corrected, safe, detail = self._correct_one(
                frame=frame, camera_jpeg=jpeg, camera_arw=arw,
                review_jpeg=review_jpeg,
                content_box_xywh=content_box_xywh,
                h_for_source=h, exact_b=exact_b)
            corrected_members.append(corrected)
            safe_members.append(safe)
            certificates.append(detail)
            expected_members.append(dict(camera_jpeg_sha256=member["camera_jpeg_sha256"],
                                         camera_arw_sha256=member["camera_arw_sha256"]))
        all_valid = np.logical_and.reduce(
            [np.isfinite(value).all(axis=0) for value in corrected_members])
        corrected = np.median(np.stack(corrected_members, axis=0), axis=0).astype(np.float32)
        corrected[:, ~all_valid] = np.nan
        safe = np.logical_and.reduce(safe_members)
        certificate = dict(
            transform_sha256=sha256(json.dumps(
                [part["renderer"] for part in certificates], sort_keys=True,
                default=str).encode()).hexdigest(),
            camera_members=expected_members,
            source_kind="new-v5-median-of-three-corrected-camera-raws",
            historical_prepared_sha256=historical_prepared_sha256,
            original_review_jpeg_sha256=sha256_file(review_jpeg),
            exact_historical_composite_reconstruction=False,
            member_renderer_certificates=[part["renderer"] for part in certificates],
            source_frame_transport_receipt_sha256=frame["receipt_sha256"])
        if self.enclosed_hole_policy != "keep" or self.display_mode != "raw_log":
            certificate.update(certificates[0]["mask_metadata"])
        return self._finish(corrected, safe, certificate)


__all__ = ["Track1HistoricalV4R5SourceProcessor"]
