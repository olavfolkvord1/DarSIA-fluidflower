"""Source-exact full-native Track1 V4-R5 processor for provisional review.

This processor only supplies corrected camera-linear float signal.  Its identity
records the failed spatial/yellow gates; source/label admission is separate.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import ffac_v5_observed_sand as obs
from darsia.fluidflower.tools.ffac_raw_native_ea import registered_edge_aware_raw_linear
from darsia.fluidflower.tools.ffac_track1_photographic_display import (
    display_contract, photographic_display, validate_display_mode,
)
from darsia.fluidflower.tools.ffac_v4r5_native_pilot import correct_frame_native_v4r5
from darsia.fluidflower.tools.ffac_v5_chart_registration import register_frame_chart_xy8
from darsia.fluidflower.tools.ffac_v5_frame_context import photo_safe_core_policy_metadata
from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
    Track1FramePhotoCohort, bind_photo_only_marker,
)
from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
    _source_photo_grid, project_source_photo_safe_core,
)
from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
    CorrectedFrame, _raw_detail_display,
)


CONTRACT = "ffac-v5-track1-source-bound-native-processor-v1"
FILLED_DISPLAY_CONTRACT = "ffac-v5-track1-filled-sand-photographic-processor-v1"
ISOLATED_DISPLAY_CONTRACT = "ffac-v5-track1-filled-sand-isolated-invalid-display-modes-processor-v1"


def _validate_output_policy(enclosed_hole_policy: str, display_mode: str,
                            isolated_invalid_policy: str = "keep") -> None:
    if enclosed_hole_policy not in ("keep", "fill_enclosed_v1"):
        raise ValueError(f"Unknown Track1 enclosed-hole policy: {enclosed_hole_policy!r}")
    if display_mode != "natural_modes_v1":
        validate_display_mode(display_mode)
    from .ffac_track1_isolated_invalid import validate_policy
    validate_policy(isolated_invalid_policy)
    if isolated_invalid_policy != "keep" and enclosed_hole_policy != "fill_enclosed_v1":
        raise ValueError("Isolated-invalid repair requires the explicit filled-sand policy")


def _output_display_contract(mode: str) -> dict:
    if mode == "natural_modes_v1":
        from .ffac_track1_display_modes import display_contract as modes_contract
        return modes_contract()
    return (display_contract() if mode == "photographic_v1"
            else {"contract": "existing-fixed-log-camera-channel-view"})


def _source_outputs(corrected: np.ndarray, safe: np.ndarray, *,
                    display_mode: str, isolated_invalid_policy: str = "keep",
                    return_auxiliary: bool = False):
    """Separate full photographic presentation from scientific sand support."""
    finite = np.isfinite(corrected).all(axis=0)
    effective = safe & finite
    scientific = corrected.copy()
    scientific[:, ~effective] = np.nan
    auxiliary = dict(filled_invalid_mask=None, isolated_invalid_fill=None,
                     display_modes=None, photo_sand_mask=None, display_modes_metadata=None,
                     display_invalid_mask=None)
    if isolated_invalid_policy != "keep":
        from .ffac_track1_isolated_invalid import fill_isolated_invalid
        scientific, filled, metadata = fill_isolated_invalid(scientific, safe)
        effective = safe & np.isfinite(scientific).all(axis=0)
        auxiliary.update(filled_invalid_mask=filled, isolated_invalid_fill=metadata)
    if display_mode == "natural_modes_v1":
        from .ffac_track1_display_modes import render_display_modes
        modes, metadata = render_display_modes(corrected, safe)
        rgb = modes["natural"]
        auxiliary.update(display_modes=modes, display_modes_metadata=metadata,
                         photo_sand_mask=safe.copy(), display_invalid_mask=~finite)
    else:
        rgb = (photographic_display(corrected) if display_mode == "photographic_v1"
               else _raw_detail_display(scientific))
    output = (rgb, scientific, effective)
    return (*output, auxiliary) if return_auxiliary else output


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _h(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(matrix[2, 2]) < 1e-12):
        raise ValueError("Source-bound homography is invalid")
    return matrix / matrix[2, 2]


def _chart_xy_baseline_native(fit: Any, grid: Any,
                              h_ref_to_baseline: np.ndarray) -> np.ndarray:
    xy = np.asarray(fit.xy, dtype=np.float32)
    if xy.shape != (24, 2):
        raise ValueError("V4-R5 ColorChecker lattice is incomplete")
    native_ref = np.column_stack((
        (xy[:, 0] + grid.origin_cell_xy[0]) * 8 + 3.5,
        (xy[:, 1] + grid.origin_cell_xy[1]) * 8 + 3.5,
    )).astype(np.float32)
    native_baseline = cv2.perspectiveTransform(native_ref[None],
                                               h_ref_to_baseline)[0]
    return (native_baseline - 3.5) / 8


class Track1V4R5SourceProcessor:
    """Render one exact current source revision with the frozen 43-run fit."""

    def __init__(self, source_row: dict[str, Any], *,
                 calibration_manifest_path: Path,
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
        self.source = _read(self.source_path)
        self.calibration_path = Path(calibration_manifest_path)
        self.photo_aggregate_path = Path(photo_aggregate_path)
        calibration = _read(self.calibration_path)
        if (calibration.get("contract") !=
                "ffac-v4r5-multirun-observed-sand-calibration43-v1"):
            # The fit is frozen by its live manifest SHA even if a fitter uses
            # a more specific contract; reject missing contracts, not names.
            if not str(calibration.get("contract") or "").startswith("ffac-v4r5-"):
                raise ValueError("Not a V4-R5 calibration manifest")
        marker = bind_photo_only_marker(self.source_path, source_row,
                                       enclosed_hole_policy=self.enclosed_hole_policy)
        cohort = Track1FramePhotoCohort.__new__(Track1FramePhotoCohort)
        cohort.source_manifest_path = self.source_path
        cohort.source_manifest_sha256 = source_row["source_manifest_sha256"]
        cohort.source = self.source
        cohort.rows = marker["frames"]
        cohort.shape_hw = tuple(self.source["analysis_image_shape_hw"])
        cohort.enclosed_hole_policy = self.enclosed_hole_policy
        self.cohort = cohort
        fit_path = self.calibration_path.parent / f"{self.run}_baseline.npz"
        sidecar_path = fit_path.with_suffix(".json")
        self.fit, self.grid = obs.load_baseline(fit_path)
        self.reference, ref_grid = obs.load_baseline(
            self.calibration_path.parent / "virtual_reference_baseline.npz")
        if self.grid != ref_grid:
            raise ValueError("V4-R5 run fit and reference use different grids")
        fit_source = _read(sidecar_path)["source"]
        frame0 = cohort.frame(0)
        if (fit_source["source_manifest_sha256"] != source_row["source_manifest_sha256"]
                or fit_source["source_arw_sha256"] != frame0["v5_B_arw_sha256"]
                or fit_source["source_review_sha256"] != frame0["v5_B_native_jpeg_sha256"]):
            raise ValueError("Current source revision differs from its V4-R5 B fit")
        self.old_field = None
        self.old_field_sha = None
        for item in calibration.get("old_B_local_residual_fields") or []:
            if item["run"] != self.run:
                continue
            path = Path(item["field_path"])
            if not path.is_absolute():
                path = self.calibration_path.parents[3] / path
            if sha256_file(path) != item["field_sha256"]:
                raise ValueError("Frozen old-light field changed")
            with np.load(path, allow_pickle=False) as archive:
                self.old_field = archive["field"].astype(np.float32)
            self.old_field_sha = item["field_sha256"]
            break
        # This identity is deliberately shared by all 43 revisions. Per-run
        # fit/source SHA is bound in each frame certificate, not an identity
        # field that would fragment the aggregate.
        self.identity_value = dict(
            contract=CONTRACT,
            calibration_manifest_path=str(self.calibration_path),
            calibration_manifest_sha256=sha256_file(self.calibration_path),
            photo_source_aggregate_path=str(self.photo_aggregate_path),
            photo_source_aggregate_sha256=sha256_file(self.photo_aggregate_path),
            processor_module_sha256=sha256_file(Path(__file__)),
            renderer_module_sha256=sha256_file(Path(registered_edge_aware_raw_linear.__code__.co_filename)),
            photometry_module_sha256=sha256_file(Path(obs.__file__)),
            pilot_math_module_sha256=sha256_file(Path(correct_frame_native_v4r5.__code__.co_filename)),
            validation_only=True, scientific_mass_eligible=False,
            whole_run_B_H_pass=True, donor_tile_cv_pass=False,
            late_yellow_gate_pass=False, weak_phase_gate_status="pending-diagnostic",
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

    def correct(self, *, run: str, frame_id: str, camera_jpeg: Path,
                camera_arw: Path, review_jpeg: Path,
                source_manifest_sha256: str, content_shape_hw: tuple[int, int],
                content_box_xywh: tuple[int, int, int, int]) -> CorrectedFrame:
        if (run != self.run or source_manifest_sha256 !=
                self.row["source_manifest_sha256"] or
                tuple(content_shape_hw) != (2979, 5472) or
                tuple(content_box_xywh) != (416, 496, 5472, 2979)):
            raise ValueError("V4-R5 source/geometry binding changed")
        reference = frame_id == "pre_injection_reference"
        if not reference and not frame_id.startswith("frame_"):
            raise ValueError("Unknown V4-R5 source frame ID")
        index = 0 if reference else int(frame_id.split("_")[1])
        frame = self.cohort.frame(index)
        if reference:
            if (sha256_file(camera_arw) != frame["v5_B_arw_sha256"]
                    or sha256_file(camera_jpeg) != frame["v5_B_camera_jpeg_sha256"]
                    or sha256_file(review_jpeg) != frame["v5_B_native_jpeg_sha256"]):
                raise ValueError("Reference source differs from frozen V4-R5 B")
        else:
            if (sha256_file(camera_jpeg) != frame["camera_jpeg_sha256"]
                    or sha256_file(review_jpeg) != frame["native_review_jpeg_sha256"]):
                raise ValueError("Timeline source differs from photo/H receipt")
        safe_grid = _source_photo_grid(
            frame["photo_sand_reference_mask_path"],
            frame["photo_sand_reference_mask_sha256"])
        h_ref_frame = _h(frame["H_ac28_to_frame_native"])
        h_b_frame = _h(frame["H_v5_B_to_frame_native"])
        h_ref_b = _h(np.linalg.inv(h_b_frame) @ h_ref_frame)
        h_for_source = h_ref_b if reference else h_ref_frame
        safe, mask_metadata = project_source_photo_safe_core(
            safe_grid, h_for_source, shape_hw=(2979, 5472),
            enclosed_hole_policy=self.enclosed_hole_policy, return_metadata=True)
        expected = dict(arw_sha256=sha256_file(camera_arw),
                        camera_jpeg_sha256=sha256_file(camera_jpeg),
                        review_native_jpeg_sha256=sha256_file(review_jpeg))
        linear, renderer = registered_edge_aware_raw_linear(
            camera_arw, camera_jpeg, review_jpeg,
            expected_source_hashes=expected, content_box=content_box_xywh)
        exact_b = (expected["arw_sha256"] == frame["v5_B_arw_sha256"]
                   and expected["camera_jpeg_sha256"] == frame["v5_B_camera_jpeg_sha256"])
        if reference and not exact_b:
            raise ValueError("Reference correction did not use the exact B acquisition")
        chart_certificate = None
        xy = usable = None
        if not exact_b:
            bcert = _read(Path(frame["v5_B_registration_certificate_path"]))
            if bcert.get("sources", {}).get("review_native_jpeg_sha256") != frame[
                    "v5_B_native_jpeg_sha256"]:
                raise ValueError("Baseline chart registration belongs to another B")
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
        finite = np.isfinite(corrected).all(axis=0)
        # B-photo sand and fitted photometric field have distinct support.
        # Their exact intersection is the only eligible float source; keep
        # unresolved source sand transparent and count it explicitly.
        rgb, scientific, effective, auxiliary = _source_outputs(
            corrected, safe, display_mode=self.display_mode,
            isolated_invalid_policy=getattr(self, "isolated_invalid_policy", "keep"),
            return_auxiliary=True)
        if not effective.any():
            raise ValueError("V4-R5 has no source-supported corrected sand")
        if np.nanmax(scientific) > np.log(65535.0):
            raise ValueError("V4-R5 corrected linear signal exceeds fixed denominator")
        certificate = dict(
            transform_sha256=sha256(json.dumps(renderer, sort_keys=True,
                                                default=str).encode()).hexdigest(),
            camera_jpeg_sha256=expected["camera_jpeg_sha256"],
            camera_arw_sha256=expected["arw_sha256"],
            original_review_jpeg_sha256=expected["review_native_jpeg_sha256"],
            native_shape_hw=[2979, 5472], renderer_certificate=renderer,
            chart_certificate=chart_certificate, v4r5_native_info=info,
            source_photo_receipt_sha256=frame["photo_sand_reference_receipt_sha256"],
            source_frame_transport_receipt_sha256=frame["receipt_sha256"],
            fit_baseline_sidecar_sha256=sha256_file(
                self.calibration_path.parent / f"{self.run}_baseline.json"),
            old_field_sha256=self.old_field_sha,
            required_valid_mask_sha256=sha256(np.packbits(effective,
                bitorder="little").tobytes()).hexdigest(),
            source_safe_pixels=int(safe.sum()),
            source_safe_corrected_pixels=int(effective.sum()),
            source_safe_unresolved_pixels=int(np.count_nonzero(safe & ~effective)),
            source_safe_corrected_fraction=float(effective.sum() / safe.sum()),
            provisional_gate_status=dict(whole_run_B_H_pass=True,
                donor_tile_cv_pass=False, late_yellow_gate_pass=False,
                weak_phase="pending-diagnostic"),
        )
        if self.enclosed_hole_policy != "keep" or self.display_mode != "raw_log":
            certificate.update(mask_metadata)
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


__all__ = ["Track1V4R5SourceProcessor", "CONTRACT"]
