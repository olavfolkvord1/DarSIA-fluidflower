"""Materialize source-bound, native-resolution FF_AC v5 review images.

The numerical RAW registration and photometric correction live in an injected
processor. This module owns the immutable review-image layout and source
lineage. It never reads phase labels, fits a phase model, or declares labels
eligible for training. A separate native-boundary audit must do that.
"""

from __future__ import annotations

from darsia.fluidflower.native import sidecar as _sidecar
from darsia.fluidflower.native.config import require_parameters

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Protocol
from uuid import uuid4

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_training.v5_linear_input import EXACT_VALID_MASK_POLICY
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.tools.ffac_v5_geometry_policy import (
    accepted_geometry_overrides, verify_accepted_geometry,
)
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests
from darsia.fluidflower.tools.schema2_phase_transfer_campaign import _plain_frame


FORMAT = "ffac-v5-native-source-materialization-v1"
BATCH_FORMAT = "ffac-v5-native-materialization-batch-v1"
V4R3_FULL_SOURCE_QA_CONTRACT = "ffac-v4r3-full-source-label-admission-v1"
V4R3_REVIEW_PARTITION_CONTRACT = "ffac-v4r3-frozen-review-fit-evaluation-partition-v1"
V4R3_FLOAT_RANGE_PREFLIGHT_CONTRACT = "ffac-v4r3-native-float-range-preflight-v1"
TRACK1_PROCESSOR_CONTRACT = "ffac-v5-track1-source-bound-native-processor-v1"
TRACK1_FULL_SOURCE_QA_CONTRACT = "ffac-v5-track1-full-source-label-admission-v1"
TRACK1_REVIEW_PARTITION_CONTRACT = "ffac-v5-track1-frozen-review-fit-evaluation-partition-v1"
TRACK1_FLOAT_RANGE_PREFLIGHT_CONTRACT = "ffac-v5-track1-native-float-range-preflight-v1"
TRACK1_PHOTOMETRY_HELDOUT_CONTRACT = "ffac-v5-track1-photometry-source-heldout-approval-v1"
TRACK1_CURRENT43_ADMISSION_CONTRACT = (
    "ffac-v5-track1-current43-provisional-source-admission-v1")
TRACK1_PROVISIONAL_PHOTOMETRY_DECISION = (
    "ffac-v5-track1-provisional-photometry-decision-v1")
RAW_DETAIL_LUT = dict(contract="ffac-v5-fixed-log-raw-detail-display-v1",
    linear_floor=2.0, linear_ceiling=65535.0, exponent=0.75,
    same_curve_all_runs_and_channels=True, display_only=True)
AC61_HISTORICAL_SOURCE_SHA256 = "14bbf139f5cba7b449d2a6f6312a147785f467b8d2154ff83992783698e0dffe"
AC61_HISTORICAL_PREPARED_SHA256 = "18d0ba62bd789c90cd36f317239bd7f438f96a50e50afda4d84df017858df244"
AC61_HISTORICAL_REPAIR_SHA256 = "3243488b89a71dfc73663b91bebd0383263b96b25ba477b8cae52e6185a897a7"


def file_sha256(path: Path) -> str:
    from darsia.fluidflower.project import sha256_file
    return sha256_file(Path(path))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_v4r3_full_source_admission(
    identity: dict[str, Any], *, source_manifest_sha256: str | None = None,
    source_manifest_run: str | None = None,
    all_source_manifest_sha256s: set[str] | None = None,
    source_manifest_runs: Mapping[str, str] | None = None,
    training_request_sha256: str | None = None,
    admission_path: Path | None = None,
    admission_sha256: str | None = None,
) -> dict[str, Any]:
    """Require a SHA-pinned cohort QA receipt, never a processor boolean.

    A selected-frame diagnostic does not call this helper. Full source
    materialization and aggregation both do, so the same immutable decision
    applies to all 52 revisions and cannot be promoted by a local flag.
    """
    track1 = identity.get("contract") == TRACK1_PROCESSOR_CONTRACT
    expected_admission = (TRACK1_FULL_SOURCE_QA_CONTRACT if track1
                          else V4R3_FULL_SOURCE_QA_CONTRACT)
    expected_partition = (TRACK1_REVIEW_PARTITION_CONTRACT if track1
                          else V4R3_REVIEW_PARTITION_CONTRACT)
    expected_float_range = (TRACK1_FLOAT_RANGE_PREFLIGHT_CONTRACT if track1
                            else V4R3_FLOAT_RANGE_PREFLIGHT_CONTRACT)
    path_text = (str(admission_path) if admission_path is not None else
                 identity.get("full_source_qa_receipt_path"))
    expected_sha = (admission_sha256 if admission_sha256 is not None else
                    identity.get("full_source_qa_receipt_sha256"))
    if not isinstance(path_text, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("V4-R3 full source needs a SHA-bound source/label QA receipt")
    path = Path(path_text)
    if file_sha256(path) != expected_sha:
        raise ValueError("V4-R3 full source QA receipt changed")
    receipt = _read(path)
    if track1 and receipt.get("contract") == TRACK1_CURRENT43_ADMISSION_CONTRACT:
        return _verify_track1_current43_admission(
            receipt, source_manifest_sha256=source_manifest_sha256,
            source_manifest_run=source_manifest_run,
            all_source_manifest_sha256s=all_source_manifest_sha256s,
            source_manifest_runs=source_manifest_runs,
            training_request_sha256=training_request_sha256)
    gates = ("source_geometry_qa_pass", "temporal_feature_qa_pass",
             "photo_sand_qa_pass", "training_label_subset_qa_pass",
             "float_input_qa_pass")
    if (receipt.get("contract") != expected_admission
            or receipt.get("scope") != "studio-validation-only-light-only"
            or receipt.get("validation_only") is not True
            or receipt.get("scientific_mass_eligible") is not False
            or receipt.get("chemistry_alignment_applied") is not False
            or receipt.get("original_review_ids_preserved") != 4048
            or any(receipt.get(gate) is not True for gate in gates)):
        raise ValueError("V4-R3 full source/label QA is incomplete or mis-scoped")
    source_shas = receipt.get("source_manifest_sha256s")
    if (not isinstance(source_shas, list) or len(source_shas) != 52
            or len(set(source_shas)) != 52
            or any(not isinstance(value, str) or len(value) != 64 for value in source_shas)):
        raise ValueError("V4-R3 full source QA does not bind 52 unique revisions")
    if source_manifest_sha256 is not None and source_manifest_sha256 not in source_shas:
        raise ValueError("V4-R3 revision is absent from full source QA")
    if all_source_manifest_sha256s is not None and set(source_shas) != all_source_manifest_sha256s:
        raise ValueError("V4-R3 full source QA differs from the frozen source set")
    if (training_request_sha256 is not None and
            receipt.get("frozen_training_request_sha256") != training_request_sha256):
        raise ValueError("V4-R3 full source QA belongs to another training request")
    partition_path_text = receipt.get("review_partition_path")
    partition_sha = receipt.get("review_partition_sha256")
    if (not isinstance(partition_path_text, str) or not isinstance(partition_sha, str)
            or len(partition_sha) != 64):
        raise ValueError("V4-R3 full source QA omitted the frozen review partition")
    partition_path = Path(partition_path_text)
    if file_sha256(partition_path) != partition_sha:
        raise ValueError("V4-R3 frozen review partition changed")
    partition = _read(partition_path)
    rows = partition.get("rows")
    annotation_sha = receipt.get("original_annotation_contract_sha256")
    if (not isinstance(annotation_sha, str) or len(annotation_sha) != 64
            or partition.get("contract") != expected_partition
            or partition.get("frozen_training_request_sha256") !=
            receipt.get("frozen_training_request_sha256")
            or partition.get("original_annotation_contract_sha256") !=
            receipt.get("original_annotation_contract_sha256")
            or not isinstance(rows, list) or len(rows) != 4048):
        raise ValueError("V4-R3 review partition omitted frozen source IDs")
    ids = set()
    fit_ids = eval_ids = 0
    for row in rows:
        identifier = row.get("region_id")
        role = row.get("role")
        run = str(row.get("run") or "").casefold()
        if (not isinstance(identifier, str) or not identifier or identifier in ids
                or not (run.startswith("ac") and run[2:].isdigit())
                or row.get("source_manifest_sha256") not in source_shas):
            raise ValueError("V4-R3 review partition has duplicate or unbound IDs")
        if (source_manifest_sha256 is not None
                and row["source_manifest_sha256"] == source_manifest_sha256
                and source_manifest_run is not None
                and run != source_manifest_run.casefold()):
            raise ValueError("V4-R3 review ID points to another source run")
        if (source_manifest_runs is not None and
                run != str(source_manifest_runs[row["source_manifest_sha256"]]).casefold()):
            raise ValueError("V4-R3 review ID has a mismatched source revision")
        ids.add(identifier)
        if role == "fit":
            if run in {"ac20", "ac21", "ac22", "ac24", "ac27", "ac51"}:
                raise ValueError("Chemistry-shifted review entered V4-R3 source fit")
            fit_ids += 1
        elif role == "evaluation-only" and isinstance(row.get("reason"), str) and row["reason"]:
            eval_ids += 1
        else:
            raise ValueError("V4-R3 review ID has no explicit fit/evaluation role")
    if (fit_ids <= 0 or fit_ids + eval_ids != 4048
            or partition.get("fit_review_ids") != fit_ids
            or partition.get("evaluation_only_review_ids") != eval_ids
            or receipt.get("fit_review_ids") != fit_ids
            or receipt.get("evaluation_only_review_ids") != eval_ids):
        raise ValueError("V4-R3 review partition counts disagree with admission")
    range_path_text = receipt.get("float_range_preflight_path")
    range_sha = receipt.get("float_range_preflight_sha256")
    if (not isinstance(range_path_text, str) or not isinstance(range_sha, str)
            or len(range_sha) != 64):
        raise ValueError("V4-R3 full source lacks SHA-bound float range preflight")
    range_path = Path(range_path_text)
    if file_sha256(range_path) != range_sha:
        raise ValueError("V4-R3 float range preflight changed")
    range_receipt = _read(range_path)
    maximum = range_receipt.get("max_measured_corrected_linear")
    if (range_receipt.get("contract") != expected_float_range
            or range_receipt.get("frozen_training_request_sha256") !=
            receipt.get("frozen_training_request_sha256")
            or set(range_receipt.get("source_manifest_sha256s") or []) != set(source_shas)
            or range_receipt.get("unit_denominator") != 65535
            or range_receipt.get("input_dtype") != "float32"
            or range_receipt.get("display_jpeg_is_training_input") is not False
            or range_receipt.get("cpu_only_float_pipeline_pass") is not True
            or range_receipt.get("uint8_model_input_rejected") is not True
            or range_receipt.get("full_timeline_per_frame_fail_fast") is not True
            or int(range_receipt.get("source_frames_measured") or 0) < 52
            or not isinstance(maximum, (int, float))
            or not 0 < float(maximum) <= 65535):
        raise ValueError("V4-R3 float range preflight is incomplete or clipping")
    if track1:
        if receipt.get("photometry_heldout_qa_pass") is not True:
            raise ValueError("Track1 source admission lacks explicit light-heldout PASS")
        fit_path = Path(receipt.get("photometry_fit_manifest_path") or "")
        fit_sha = receipt.get("photometry_fit_manifest_sha256")
        qa_path = Path(receipt.get("photometry_heldout_qa_path") or "")
        qa_sha = receipt.get("photometry_heldout_qa_sha256")
        _check_sha_file(fit_path, fit_sha, "Track1 photometry fit")
        _check_sha_file(qa_path, qa_sha, "Track1 photometry heldout QA")
        fit = _read(fit_path)
        qa = _read(qa_path)
        if (not isinstance(fit.get("contract"), str) or not fit["contract"]
                or qa.get("contract") != TRACK1_PHOTOMETRY_HELDOUT_CONTRACT
                or qa.get("photometry_fit_manifest_sha256") != fit_sha
                or qa.get("photometry_fit_contract") != fit["contract"]
                or qa.get("source_holdouts_pass") is not True
                or qa.get("weak_phase_source_gate_pass") is not True
                or qa.get("late_light_gate_pass") is not True
                or qa.get("validation_only") is not True
                or qa.get("scientific_mass_eligible") is not False):
            raise ValueError("Track1 photometry has no accepted source-only heldouts")
    return receipt


def _verify_track1_current43_admission(
    receipt: dict[str, Any], *, source_manifest_sha256: str | None,
    source_manifest_run: str | None,
    all_source_manifest_sha256s: set[str] | None,
    source_manifest_runs: Mapping[str, str] | None,
    training_request_sha256: str | None,
) -> dict[str, Any]:
    """Accept only a documented, source-scoped provisional 43-run review fit."""
    gates = ("source_geometry_qa_pass", "temporal_feature_qa_pass",
             "photo_sand_qa_pass", "training_label_subset_qa_pass",
             "float_input_qa_pass")
    if (receipt.get("scope") != "studio-validation-only-light-only-provisional"
            or receipt.get("validation_only") is not True
            or receipt.get("scientific_mass_eligible") is not False
            or receipt.get("mass_eligible") is not False
            or receipt.get("chemistry_alignment_applied") is not False
            or receipt.get("provisional_studio_training_authorized") is not True
            or receipt.get("automatic_publication") is not False
            or receipt.get("original_review_ids_preserved") != 4048
            or any(receipt.get(gate) is not True for gate in gates)):
        raise ValueError("Current43 provisional admission overstates source eligibility")
    current = receipt.get("source_manifest_sha256s")
    historical = receipt.get("historical_source_manifest_sha256s")
    if (not isinstance(current, list) or len(current) != 43
            or len(set(current)) != 43
            or not isinstance(historical, list) or len(historical) != 9
            or len(set(historical)) != 9
            or set(current) & set(historical)
            or any(not isinstance(x, str) or len(x) != 64
                   for x in current + historical)):
        raise ValueError("Current43 admission does not bind 43+9 exact revisions")
    if (source_manifest_sha256 is not None and
            source_manifest_sha256 not in current):
        raise ValueError("Revision is outside the approved current43 source set")
    if (all_source_manifest_sha256s is not None and
            set(current) != all_source_manifest_sha256s):
        raise ValueError("Current43 source admission differs from the batch")
    if (training_request_sha256 is not None and
            receipt.get("frozen_training_request_sha256") != training_request_sha256):
        raise ValueError("Current43 source admission belongs to another request")
    partition_path = Path(receipt.get("review_partition_path") or "")
    _check_sha_file(partition_path, receipt.get("review_partition_sha256"),
                    "Current43 review partition")
    partition = _read(partition_path)
    rows = partition.get("rows")
    if (partition.get("contract") != TRACK1_REVIEW_PARTITION_CONTRACT
            or not isinstance(rows, list) or len(rows) != 4048
            or partition.get("frozen_training_request_sha256") !=
               receipt.get("frozen_training_request_sha256")):
        raise ValueError("Current43 partition omitted frozen review lineage")
    seen: set[str] = set()
    fit = evaluation = historical_ids = 0
    for row in rows:
        identifier = row.get("region_id")
        run = str(row.get("run") or "").casefold()
        digest = row.get("source_manifest_sha256")
        role = row.get("role")
        if (not isinstance(identifier, str) or not identifier or identifier in seen
                or digest not in set(current) | set(historical)
                or role not in {"fit", "evaluation-only"}
                or (source_manifest_runs is not None and digest in source_manifest_runs
                    and run != str(source_manifest_runs[digest]).casefold())):
            raise ValueError("Current43 review ID has ambiguous source or role")
        if (source_manifest_sha256 is not None and digest == source_manifest_sha256
                and source_manifest_run is not None
                and run != source_manifest_run.casefold()):
            raise ValueError("Current43 review ID belongs to another run")
        seen.add(identifier)
        if digest in historical:
            historical_ids += 1
            if (role != "evaluation-only" or row.get("reason") !=
                    "historical-source-not-materialized-current43-provisional"):
                raise ValueError("Historical source review entered current43 fit")
        if role == "fit":
            if digest not in current or run in {"ac20", "ac21", "ac22", "ac24", "ac27", "ac51"}:
                raise ValueError("Out-of-scope or chemistry review entered current43 fit")
            fit += 1
        else:
            if not isinstance(row.get("reason"), str) or not row["reason"]:
                raise ValueError("Evaluation-only review lacks explicit reason")
            evaluation += 1
    if (not fit or fit + evaluation != 4048 or historical_ids != 1719
            or partition.get("fit_review_ids") != fit
            or partition.get("evaluation_only_review_ids") != evaluation
            or receipt.get("fit_review_ids") != fit
            or receipt.get("evaluation_only_review_ids") != evaluation):
        raise ValueError("Current43 fit/evaluation partition counts differ")
    for stem in ("float_range_preflight", "photometry_fit_manifest",
                 "photometry_heldout_qa"):
        target = Path(receipt.get(f"{stem}_path") or "")
        _check_sha_file(target, receipt.get(f"{stem}_sha256"), stem)
    preflight = _read(Path(receipt["float_range_preflight_path"]))
    if (preflight.get("contract") != TRACK1_FLOAT_RANGE_PREFLIGHT_CONTRACT
            or set(preflight.get("source_manifest_sha256s") or []) != set(current)
            or preflight.get("unit_denominator") != 65535
            or preflight.get("input_dtype") != "float32"
            or preflight.get("display_jpeg_is_training_input") is not False
            or preflight.get("cpu_only_float_pipeline_pass") is not True
            or preflight.get("uint8_model_input_rejected") is not True
            or preflight.get("full_timeline_per_frame_fail_fast") is not True
            or not 0 < float(preflight.get("max_measured_corrected_linear") or 0) <= 65535):
        raise ValueError("Current43 float preflight lacks exact nonclipping contract")
    decision = _read(Path(receipt["photometry_heldout_qa_path"]))
    if (decision.get("contract") != TRACK1_PROVISIONAL_PHOTOMETRY_DECISION
            or decision.get("whole_run_B_H_pass") is not True
            or decision.get("donor_tile_cv_status") != "failed-warning"
            or decision.get("late_yellow_status") != "failed-warning"
            or decision.get("weak_phase_status") != "pending-diagnostic"
            or decision.get("photometry_fit_manifest_sha256") !=
               receipt.get("photometry_fit_manifest_sha256")):
        raise ValueError("Current43 provisional light warnings changed or were hidden")
    return receipt


def _check_sha_file(path: Path, digest: Any, label: str) -> None:
    if not isinstance(digest, str) or len(digest) != 64 or file_sha256(path) != digest:
        raise ValueError(f"{label} changed or lacks SHA binding")


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


from darsia.fluidflower.native.sidecar import CorrectedFrame


class FrameProcessor(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def correct(
        self,
        *,
        run: str,
        frame_id: str,
        camera_jpeg: Path,
        camera_arw: Path,
        review_jpeg: Path,
        source_manifest_sha256: str,
        content_shape_hw: tuple[int, int],
        content_box_xywh: tuple[int, int, int, int],
    ) -> CorrectedFrame: ...

    def correct_composite_baseline(
        self, *, run: str, frame_id: str, members: list[dict[str, Any]],
        review_jpeg: Path, source_manifest_sha256: str,
        content_shape_hw: tuple[int, int],
        content_box_xywh: tuple[int, int, int, int],
        historical_prepared_sha256: str,
    ) -> CorrectedFrame: ...


def _checked_source(
    row: dict[str, Any], original_native: Path, shape_hw: tuple[int, int],
) -> dict[str, Any]:
    jpeg = Path(row["source_path"])
    if jpeg.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"Source is not an original camera JPEG: {jpeg}")
    arw = jpeg.with_suffix(".ARW")
    if not arw.is_file() or arw.stem.casefold() != jpeg.stem.casefold():
        raise FileNotFoundError(f"No same-stem original ARW for {jpeg}")
    jpeg_hash = file_sha256(jpeg)
    expected_jpeg = row.get("source_sha256") or row.get("source_raw_sha256")
    if jpeg_hash != expected_jpeg:
        raise ValueError(f"Frozen original camera JPEG changed: {jpeg}")
    native_hash = file_sha256(original_native)
    expected_native = row.get("native_jpeg_sha256") or row.get("frame_sha256")
    if expected_native and native_hash != expected_native:
        raise ValueError(f"Frozen native review image changed: {original_native}")
    with Image.open(original_native) as image:
        if image.size[0] < shape_hw[1] or image.size[1] < shape_hw[0]:
            raise ValueError("Original review image is smaller than the native content")
    return dict(camera_jpeg=str(jpeg), camera_jpeg_sha256=jpeg_hash,
                camera_arw=str(arw), camera_arw_sha256=file_sha256(arw),
                original_review_jpeg=str(original_native),
                original_review_jpeg_sha256=native_hash)


def _checked_ac61_prepared_source(
    row: dict[str, Any], old: dict[str, Any], original_native: Path,
) -> dict[str, Any]:
    """Preserve old AC61 lineage while declaring a *new* 3-ARW V5 frame 0.

    The historic prepared JPEG cannot be reconstructed pixel-exactly from the
    three acquisitions: its old pipeline did not record that formula. The new
    image must therefore be identified as a distinct, QA-pending source.
    """
    marker = (old.get("two_arm_photometry") or {}).get("prepared_baseline_composite")
    arm = old.get("two_arm_photometry") or {}
    if (old.get("run") != "ac61" or row.get("frame_index") != 0
            or not isinstance(marker, dict)
            or arm.get("source_manifest_sha256") != AC61_HISTORICAL_SOURCE_SHA256
            or marker.get("original_pre_injection_sha256") != AC61_HISTORICAL_PREPARED_SHA256
            or marker.get("historical_repair_receipt_sha256") != AC61_HISTORICAL_REPAIR_SHA256
            or marker.get("reconstruction") != "calibrated_rig_baseline"):
        raise ValueError("AC61 source is not the frozen prepared-baseline exception")
    if (file_sha256(Path(arm["source_manifest"])) != AC61_HISTORICAL_SOURCE_SHA256
            or file_sha256(Path(marker["historical_repair_receipt_path"]))
            != AC61_HISTORICAL_REPAIR_SHA256
            or file_sha256(Path(marker["original_pre_injection_path"]))
            != AC61_HISTORICAL_PREPARED_SHA256
            or file_sha256(Path(row["source_path"])) != AC61_HISTORICAL_PREPARED_SHA256):
        raise ValueError("AC61 historical prepared source or repair receipt changed")
    native_hash = file_sha256(original_native)
    if native_hash != row.get("native_jpeg_sha256"):
        raise ValueError("AC61 historical native frame 0 changed")
    configured = row.get("configured_baseline_members")
    members = marker.get("members")
    if not isinstance(configured, list) or not isinstance(members, list) or len(members) != 3:
        raise ValueError("AC61 three-acquisition baseline inventory is missing")
    verified = []
    for original, member in zip(configured, members, strict=True):
        jpeg = Path(original["path"])
        arw = jpeg.with_suffix(".ARW")
        if (jpeg != Path(member["path"]) or original.get("elapsed_h") != member.get("elapsed_h")
                or original.get("timestamp") != member.get("timestamp")
                or file_sha256(jpeg) != member["sha256"] or not arw.is_file()):
            raise ValueError(f"AC61 baseline member changed: {jpeg}")
        verified.append(dict(camera_jpeg=str(jpeg), camera_jpeg_sha256=member["sha256"],
            camera_arw=str(arw), camera_arw_sha256=file_sha256(arw),
            elapsed_h=member["elapsed_h"], timestamp=member["timestamp"]))
    if abs(sum(m["elapsed_h"] for m in verified)/3 - row["elapsed_h"]) > 1e-12:
        raise ValueError("AC61 composite acquisition time changed")
    return dict(
        source_kind="new-v5-median-of-three-corrected-camera-raws",
        historical_prepared_jpeg=str(row["source_path"]),
        historical_prepared_sha256=AC61_HISTORICAL_PREPARED_SHA256,
        historical_source_manifest_sha256=AC61_HISTORICAL_SOURCE_SHA256,
        historical_repair_receipt_sha256=AC61_HISTORICAL_REPAIR_SHA256,
        original_review_jpeg=str(original_native),
        original_review_jpeg_sha256=native_hash,
        camera_members=verified,
        exact_historical_composite_reconstruction=False,
    )


def _reference_source(
    *, run: str, old: dict[str, Any], review_root: Path,
    reference_state: dict[str, Any], reference_state_sha256: str,
    source_manifest_sha256: str, binding_path: Path,
) -> dict[str, Any]:
    """Bind this exact revision's B acquisition, never infer from a run name."""
    state = reference_state.get(run)
    if not isinstance(state, dict) or state.get("kind") not in {
        "frame0", "pre-injection floor"}:
        raise ValueError(f"No recognized frozen V5 reference state for {run}")
    from darsia.fluidflower.tools.ffac_v5_reference_bindings import CONTRACT as BINDING_CONTRACT
    binding_path = Path(binding_path)
    binding_sha = file_sha256(binding_path)
    audit = _read(binding_path)
    if audit.get("contract") != BINDING_CONTRACT or audit.get("source_revisions") != 52:
        raise ValueError("V5 revision B binding audit has another contract/cohort")
    matches = [r for r in audit["rows"] if r["old_manifest_sha256"] == source_manifest_sha256]
    if len(matches) != 1 or matches[0]["run"] != run:
        raise ValueError("V5 B binding is missing or ambiguous for this exact revision")
    row = matches[0]
    for path_key, sha_key in (("camera_arw", "camera_arw_sha256"),
                              ("camera_jpeg", "camera_jpeg_sha256"),
                              ("original_review_jpeg", "original_review_jpeg_sha256")):
        if file_sha256(Path(row[path_key])) != row[sha_key]:
            raise ValueError(f"V5 revision B source changed: {path_key}")
    arw = Path(row["camera_arw"])
    jpeg = Path(row["camera_jpeg"])
    native = Path(row["original_review_jpeg"])
    if arw.stem.casefold() != jpeg.stem.casefold():
        raise ValueError("V5 revision B ARW and camera JPEG are different acquisitions")
    if row["is_canonical"] and arw.name.casefold() != str(state["file"]).casefold():
        raise ValueError("Canonical V5 B differs from Claude's fitted B acquisition")
    timeline = [int(r["frame_index"]) for r in old["frame_sources"]
                if str(Path(r["source_path"]).with_suffix(".ARW")).casefold()
                   == str(arw).casefold()]
    if len(timeline) > 1:
        raise ValueError(f"Reference acquisition duplicated in {run} source timeline")
    return dict(
        kind=state["kind"] if row["is_canonical"] else "historical-revision-B",
        reference_state_sha256=reference_state_sha256,
        revision_B_binding_sha256=binding_sha, is_canonical_B=row["is_canonical"],
        camera_arw=str(arw), camera_arw_sha256=row["camera_arw_sha256"],
        camera_jpeg=str(jpeg), camera_jpeg_sha256=row["camera_jpeg_sha256"],
        original_review_jpeg=str(native), original_review_jpeg_sha256=row["original_review_jpeg_sha256"],
        timeline_frame_index=timeline[0] if timeline else None,
        cross_frame_registration=not bool(timeline),
        historical_reference_array_sha256=(old.get("reference") or {}).get("corrected_array_sha256"),
    )


def _validate_output(
    frame: CorrectedFrame, source: dict[str, Any], expected_shape: tuple[int, int],
    *, exact_valid_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frame, CorrectedFrame):
        raise TypeError("V5 frame processor must return CorrectedFrame")
    rgb = np.asarray(frame.rgb)
    if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.float32:
        raise ValueError("V5 output must be float32 RGB on the exact native content grid")
    if not np.all(np.isfinite(rgb)) or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("V5 display output is nonfinite or outside RGB 0â€“1")
    log_linear = np.asarray(frame.log_linear_rgb)
    if log_linear.shape != (3, *expected_shape) or log_linear.dtype != np.float32:
        raise ValueError("V5 linear sidecar must be native 3Ã—HÃ—W float32")
    if np.isinf(log_linear).any():
        raise ValueError("V5 linear sidecar contains infinite RAW values")
    valid = np.all(np.isfinite(log_linear), axis=0)
    if exact_valid_mask:
        required = frame.required_valid_mask
        if (not isinstance(required, np.ndarray) or required.dtype != np.dtype("bool")
                or required.shape != expected_shape or not required.any()
                or np.any(required & ~valid)):
            raise ValueError("V4-R3 source-required pixels lack finite RAW correction")
        required_digest = sha256(np.packbits(required, bitorder="little").tobytes()).hexdigest()
        if frame.certificate.get("required_valid_mask_sha256") != required_digest:
            raise ValueError("V4-R3 source eligibility mask lacks certificate binding")
    elif frame.required_valid_mask is not None:
        raise ValueError("Legacy V5 output unexpectedly carries a source eligibility mask")
    elif np.count_nonzero(valid) < .999 * np.prod(expected_shape):
        raise ValueError("More than 0.1% of the V5 native linear signal is nonfinite")
    certificate = frame.certificate
    if not isinstance(certificate, dict) or not certificate.get("transform_sha256"):
        raise ValueError("V5 frame lacks its RAWâ†’native transform certificate")
    if source.get("camera_members"):
        expected_members = [{"camera_jpeg_sha256": r["camera_jpeg_sha256"],
                             "camera_arw_sha256": r["camera_arw_sha256"]}
                            for r in source["camera_members"]]
        if (certificate.get("camera_members") != expected_members
                or certificate.get("source_kind") != source["source_kind"]
                or certificate.get("historical_prepared_sha256")
                   != source["historical_prepared_sha256"]
                or certificate.get("original_review_jpeg_sha256")
                   != source["original_review_jpeg_sha256"]):
            raise ValueError("V5 composite certificate differs from three-ARW source")
    else:
        for key in ("camera_jpeg_sha256", "camera_arw_sha256", "original_review_jpeg_sha256"):
            if certificate.get(key) != source[key]:
                raise ValueError(f"V5 transform certificate differs from source: {key}")
    if certificate.get("native_shape_hw") != list(expected_shape):
        raise ValueError("V5 transform certificate binds another native grid")
    if frame.filled_invalid_mask is not None:
        from .ffac_track1_isolated_invalid import CONTRACT as FILL_CONTRACT
        filled = frame.filled_invalid_mask
        repair = certificate.get("isolated_invalid_fill") or {}
        if (not exact_valid_mask or not isinstance(filled, np.ndarray)
                or filled.dtype != np.dtype(bool) or filled.shape != expected_shape
                or np.any(filled & ~frame.required_valid_mask)
                or repair.get("contract") != FILL_CONTRACT
                or repair.get("filled_pixel_count") != int(filled.sum())
                or repair.get("filled_mask_packbits_sha256") != sha256(
                    np.packbits(filled, bitorder="little").tobytes()).hexdigest()):
            raise ValueError("Isolated-invalid repair lacks its exact native mask binding")
    elif certificate.get("isolated_invalid_fill"):
        raise ValueError("Isolated-invalid certificate lacks its repair mask")
    if frame.display_modes is not None:
        photo = frame.photo_sand_mask
        display_invalid = frame.display_invalid_mask
        if (certificate.get("display_mode") != "natural_modes_v1"
                or set(frame.display_modes) != {"natural", "lift1", "lift2"}
                or not isinstance(photo, np.ndarray) or photo.dtype != np.dtype(bool)
                or photo.shape != expected_shape
                or not isinstance(display_invalid, np.ndarray)
                or display_invalid.dtype != np.dtype(bool)
                or display_invalid.shape != expected_shape
                or certificate.get("display_modes_metadata", {}).get("invalid_cleanup", {}).get(
                    "invalid_input_pixels") != int(display_invalid.sum())
                or (exact_valid_mask and np.any(frame.required_valid_mask & ~photo))):
            raise ValueError("Photographic display modes lack their native photo-sand support")
        for mode, image in frame.display_modes.items():
            if (not isinstance(image, np.ndarray) or image.dtype != np.float32
                    or image.shape != rgb.shape or not np.isfinite(image).all()
                    or image.min() < 0 or image.max() > 1):
                raise ValueError(f"Photographic display mode has invalid RGB: {mode}")
        if not np.array_equal(frame.display_modes["natural"], rgb):
            raise ValueError("Main native photograph differs from fixed natural display")
    elif (frame.photo_sand_mask is not None or frame.display_invalid_mask is not None
            or certificate.get("display_mode") == "natural_modes_v1"):
        raise ValueError("Photographic display certificate lacks its modes or support mask")
    return rgb, log_linear


def _fixed_float_input_qa(log_linear, *, required_valid_mask=None):
    return _sidecar.fixed_float_input_qa(log_linear,required_valid_mask=required_valid_mask,
        parameters=require_parameters('sidecar'))


def _write_v4_ring(path, frame, native_log, *, pad_px):
    return _sidecar.write_frame_ring(path,frame,native_log,pad_px=pad_px,parameters=require_parameters('extended_ring'))


def _raw_detail_display(log_linear):
    return _sidecar.raw_detail_display(log_linear,parameters=require_parameters('sidecar'))


def _write_log_linear(path, values):
    return _sidecar.write_log_linear(path,values)


def _copy_verified(source,destination,expected_sha256):
    return _sidecar.copy_verified(source,destination,expected_sha256)


def _frame_output(rgb, *, run,index,count,elapsed_h,canvas_wh,destination):
    return _sidecar.write_display_jpeg(rgb,run=run,index=index,count=count,elapsed_h=elapsed_h,
        canvas_wh=canvas_wh,destination=destination,renderer=_plain_frame,parameters=require_parameters('sidecar'))


_AUXILIARY_OUTPUT_KEYS = ("v5_display_modes", "v5_photo_sand_mask_path",
    "v5_photo_sand_mask_sha256", "v5_filled_invalid_mask_path",
    "v5_filled_invalid_mask_sha256", "v5_display_invalid_mask_path",
    "v5_display_invalid_mask_sha256")


def _write_display_invalid_mask(path, mask):
    return _sidecar.write_display_invalid_mask(path,mask,parameters=require_parameters('sidecar'))


def _auxiliary_output_fields(record):
    return _sidecar.auxiliary_output_fields(record,parameters=require_parameters('sidecar'))


def _write_auxiliary_outputs(frame,target,native_sha256,*,run,index,count,elapsed_h,canvas_wh):
    return _sidecar.write_auxiliary_outputs(frame,target,native_sha256,run=run,index=index,count=count,
        elapsed_h=elapsed_h,canvas_wh=canvas_wh,renderer=_plain_frame,parameters=require_parameters('sidecar'))


def _verify_auxiliary_outputs(record: dict[str, Any], identity: dict[str, Any],
                              target: Path) -> None:
    """Default assets remain unchanged; opted-in assets are all SHA mandatory."""
    modes = record.get("v5_display_modes")
    required_display = identity.get("display_mode") == "natural_modes_v1"
    required_fill = identity.get("isolated_invalid_policy") == "fill_isolated_invalid_v1"
    if required_display:
        if (not isinstance(modes, dict) or set(modes) != {"natural", "lift1", "lift2"}
                or modes["natural"] != dict(jpeg_path=str(target),
                    jpeg_sha256=record.get("native_jpeg_sha256"))):
            raise ValueError("Completed photographic display modes lack their native binding")
    elif modes:
        raise ValueError("Legacy processor unexpectedly carries photographic modes")
    for field, required in (("photo_sand", required_display), ("filled_invalid", required_fill)):
        path, digest = record.get(f"v5_{field}_mask_path"), record.get(f"v5_{field}_mask_sha256")
        if required or path or digest:
            if not required or not path or not digest or file_sha256(Path(path)) != digest:
                raise ValueError(f"Completed V5 {field} mask changed or is unbound")
    invalid_path, invalid_sha = record.get("v5_display_invalid_mask_path"), record.get(
        "v5_display_invalid_mask_sha256")
    if required_display or invalid_path or invalid_sha:
        if (not required_display or not invalid_path or not invalid_sha
                or file_sha256(Path(invalid_path)) != invalid_sha):
            raise ValueError("Completed V5 display_invalid mask changed or is unbound")
    for mode, binding in (modes or {}).items():
        if (not isinstance(binding, dict) or not binding.get("jpeg_path")
                or file_sha256(Path(binding["jpeg_path"])) != binding.get("jpeg_sha256")):
            raise ValueError(f"Completed V5 photographic display changed: {mode}")


def _copy_auxiliary_outputs(record: dict[str, Any], target: Path,
                            native_sha256: str) -> dict[str, Any]:
    """Give a matched pre-injection reference its own immutable asset paths."""
    output = {}
    for field in ("photo_sand", "filled_invalid"):
        source = record.get(f"v5_{field}_mask_path")
        if source:
            path = target.with_name(f"{target.stem}.{field}_mask.npy")
            digest = _copy_verified(Path(source), path, record[f"v5_{field}_mask_sha256"])
            output[f"v5_{field}_mask_path"] = str(path)
            output[f"v5_{field}_mask_sha256"] = digest
    if record.get("v5_display_modes"):
        invalid_path = target.with_name(f"{target.stem}.display_invalid_mask.npz")
        output["v5_display_invalid_mask_path"] = str(invalid_path)
        output["v5_display_invalid_mask_sha256"] = _copy_verified(
            Path(record["v5_display_invalid_mask_path"]), invalid_path,
            record["v5_display_invalid_mask_sha256"])
        modes = {"natural": dict(jpeg_path=str(target), jpeg_sha256=native_sha256)}
        for mode in ("lift1", "lift2"):
            binding = record["v5_display_modes"][mode]
            path = target.with_name(f"{target.stem}.{mode}.jpg")
            digest = _copy_verified(Path(binding["jpeg_path"]), path, binding["jpeg_sha256"])
            modes[mode] = dict(jpeg_path=str(path), jpeg_sha256=digest)
        output["v5_display_modes"] = modes
    return output


def _bound_frame_geometry_marker(source_manifest: Path,
                                 row: dict[str, Any] | None) -> dict[str, Any]:
    """Bind every V5 frame to its original source and accepted temporal H.

    A full materialization must never inherit the old static run context when
    the sand moves.  Diagnostic selected-frame pilots do not emit a trainable
    manifest and may omit this marker.
    """
    from darsia.fluidflower.tools.ffac_v5_frame_context import (
        COHORT_CONTRACT, verify_frame_geometry,
    )

    if row is None:
        raise ValueError("Full V5 source needs an accepted per-frame geometry row")
    source = _read(source_manifest)
    source_sha = file_sha256(source_manifest)
    frames = row.get("frames") or []
    if (row.get("source_manifest_sha256") != source_sha
            or len(frames) != int(source.get("frame_count") or -1)
            or [item.get("frame_index") for item in frames] != list(range(len(frames)))):
        raise ValueError("V5 per-frame geometry row does not cover this exact revision")
    static_cache: dict[tuple[str, str], tuple[int, int]] = {}
    for index, item in enumerate(frames):
        path = Path(item.get("receipt_path") or "")
        digest = item.get("receipt_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or file_sha256(path) != digest:
            raise ValueError(f"V5 frame geometry receipt changed: {path}")
        binding = verify_frame_geometry(path, source_manifest,
            expected_shape_hw=tuple(source["analysis_image_shape_hw"]),
            static_hash_cache=static_cache)
        if binding["frame_index"] != index or binding["receipt_sha256"] != digest:
            raise ValueError("V5 frame geometry receipt has another frame/source")
    return dict(contract=COHORT_CONTRACT,
                original_source_manifest_path=str(source_manifest),
                original_source_manifest_sha256=source_sha,
                frames=[dict(frame_index=int(item["frame_index"]),
                             receipt_path=str(item["receipt_path"]),
                             receipt_sha256=str(item["receipt_sha256"]))
                        for item in frames])


def materialize_revision(
    revision: dict[str, Any], *, output_root: Path, processor: FrameProcessor,
    reference_state_path: Path, reference_binding_path: Path,
    geometry_row: dict[str, Any] | None = None,
    frame_geometry_row: dict[str, Any] | None = None,
    frame_indices: set[int] | None = None,
    admission_path: Path | None = None,
    admission_sha256: str | None = None,
    source_only_preparation: bool = False,
) -> dict[str, Any]:
    """Build one immutable revision; fail closed on absent paired ARW/geometry.

    Existing per-frame receipts permit deterministic resume. Any changed source,
    correction identity or output hash stops; no partial work is deleted.
    """
    run = revision["run"]
    source_manifest = Path(revision["manifest_path"])
    source_manifest_sha = revision["manifest_sha256"]
    if file_sha256(source_manifest) != source_manifest_sha:
        raise ValueError(f"Frozen native source manifest changed: {source_manifest}")
    old = _read(source_manifest)
    reference_state_path = Path(reference_state_path)
    reference_state_sha = file_sha256(reference_state_path)
    reference_state = _read(reference_state_path)["reference_state"]
    if old.get("run") != run or old.get("dataset") != "ffac":
        raise ValueError("Frozen manifest belongs to another run/series")
    identity = processor.identity()
    if not identity or not isinstance(identity, dict):
        raise ValueError("V5 processor has no immutable runtime identity")
    track1 = identity.get("contract") == TRACK1_PROCESSOR_CONTRACT
    if track1:
        if geometry_row is not None:
            raise ValueError("Track1 optical source cannot inherit physical run geometry")
        geometry = None
    else:
        if geometry_row is None:
            raise ValueError("V5 source omitted its accepted run geometry")
        geometry = accepted_geometry_overrides(source_manifest, geometry_row)
    frames = old["frame_sources"]
    if len(frames) != old["frame_count"] or [r["frame_index"] for r in frames] != list(range(len(frames))):
        raise ValueError("Native source frame list is incomplete or non-contiguous")
    if frame_indices is not None:
        frame_indices = set(frame_indices)
        if not frame_indices or not frame_indices.issubset(set(range(len(frames)))):
            raise ValueError("Selected pilot frames are outside this frozen revision")
    content_box = tuple(int(x) for x in old["content_box_xywh"])
    shape = tuple(int(x) for x in old["analysis_image_shape_hw"])
    canvas_wh = tuple(int(x) for x in old["canvas_wh"])
    if content_box[2:] != tuple(reversed(shape)) or content_box[0] < 0 or content_box[1] < 0:
        raise ValueError("Native content geometry differs from source manifest")
    if content_box[0] + shape[1] > canvas_wh[0] or content_box[1] + shape[0] > canvas_wh[1]:
        raise ValueError("Native content box escapes review canvas")
    v4 = identity.get("contract") == "ffac-v4-source-bound-native-processor-v1"
    v4r3 = identity.get("contract") in {
        "ffac-v4r3-source-bound-native-processor-v1", TRACK1_PROCESSOR_CONTRACT}
    if v4r3 and frame_geometry_row is None:
        raise ValueError("V4-R3 pilot and full source require exact per-frame geometry")
    if v4 and frame_indices is None and frame_geometry_row is None:
        raise ValueError("Full V4 source needs exact per-frame geometry")
    if source_only_preparation and (not track1 or frame_indices is not None):
        raise ValueError("Source-only preparation applies only to full Track1 revisions")
    if v4r3 and frame_indices is None and not source_only_preparation:
        verify_v4r3_full_source_admission(
            identity, source_manifest_sha256=source_manifest_sha,
            source_manifest_run=run,
            admission_path=admission_path,
            admission_sha256=admission_sha256)
    if not (v4 or v4r3) and frame_geometry_row is not None:
        raise ValueError("Legacy V5 processor cannot accept a V4 frame geometry row")
    if frame_geometry_row is None:
        frame_geometry = None
    elif track1:
        from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
            bind_photo_only_marker,
        )
        frame_geometry = bind_photo_only_marker(
            source_manifest, frame_geometry_row,
            enclosed_hole_policy=identity.get("enclosed_hole_policy", "keep"))
    else:
        frame_geometry = _bound_frame_geometry_marker(source_manifest,
                                                      frame_geometry_row)
    root = Path(output_root) / "revisions" / source_manifest_sha
    review_root = root / "manual_pairwise_review"
    frame_dir = review_root / "native_frames" / run
    baseline_dir = review_root / "native_references" / run
    job_path = root / "v5_materialization_job.json"
    job = dict(format=FORMAT, run=run, source_manifest_path=str(source_manifest),
               source_manifest_sha256=source_manifest_sha,
               reference_state_path=str(reference_state_path),
               reference_state_sha256=reference_state_sha,
               source_review_root=revision["review_root"],
               **({"v5_track1_photo_geometry": frame_geometry} if track1 else dict(
                   run_geometry_receipt_path=geometry["v5_run_geometry"]["receipt_path"],
                   run_geometry_receipt_sha256=geometry["v5_run_geometry"]["receipt_sha256"],
                   v5_frame_geometry=frame_geometry)),
               content_box_xywh=list(content_box), analysis_image_shape_hw=list(shape),
               canvas_wh=list(canvas_wh), frame_count=len(frames),
               processor=identity, training_eligibility="pending-native-annotation-boundary-qa")
    if job_path.is_file():
        if _read(job_path) != job:
            raise ValueError("Partial materialization belongs to a different immutable job")
    else:
        _write_new(job_path, job)
    output_rows = []
    revised_rows = []
    for row in frames:
        index = int(row["frame_index"])
        if frame_indices is not None and index not in frame_indices:
            continue
        original_native = Path(revision["review_root"]) / "native_frames" / run / f"frame_{index:03d}.jpg"
        prepared_ac61 = bool(run == "ac61" and index == 0 and
            (old.get("two_arm_photometry") or {}).get("prepared_baseline_composite"))
        source = (_checked_ac61_prepared_source(row, old, original_native)
                  if prepared_ac61 else _checked_source(row, original_native, shape))
        target = frame_dir / f"frame_{index:03d}.jpg"
        detail_target = frame_dir / f"frame_{index:03d}.raw_detail.jpg"
        linear_path = frame_dir / f"frame_{index:03d}.log_linear_rgb.npy"
        valid_path = frame_dir / f"frame_{index:03d}.log_linear_valid_mask.npy"
        ring_path = frame_dir / f"frame_{index:03d}.extended_ring.npz"
        record_path = frame_dir / f"frame_{index:03d}.v5.json"
        if record_path.is_file():
            record = _read(record_path)
            if (record.get("source") != source or record.get("elapsed_h") != row["elapsed_h"]
                    or record.get("processor") != identity or
                    record.get("signal_qa", {}).get("fixed_input_nonclipping_pass") is not True or
                    file_sha256(target) != record.get("native_jpeg_sha256") or
                    file_sha256(detail_target) != record.get("v5_raw_detail_jpeg_sha256") or
                    file_sha256(linear_path) != record.get("log_linear_rgb_sha256") or
                    (v4r3 and file_sha256(valid_path) != record.get(
                        "v5_log_linear_valid_mask_sha256")) or
                    (v4 and file_sha256(ring_path) != record.get(
                        "v5_extended_ring_sha256"))):
                raise ValueError(f"Completed V5 frame changed: {target}")
            _verify_auxiliary_outputs(record, identity, target)
        else:
            # An interruption may leave one or both finished outputs without
            # their receipt. Recompute and accept only byte-identical outputs.
            if prepared_ac61:
                corrected = processor.correct_composite_baseline(
                    run=run, frame_id="frame_000", members=source["camera_members"],
                    review_jpeg=original_native,
                    source_manifest_sha256=source_manifest_sha,
                    content_shape_hw=shape, content_box_xywh=content_box,
                    historical_prepared_sha256=source["historical_prepared_sha256"])
            else:
                corrected = processor.correct(run=run, frame_id=f"frame_{index:03d}",
                    camera_jpeg=Path(source["camera_jpeg"]),
                    camera_arw=Path(source["camera_arw"]),
                    review_jpeg=original_native,
                    source_manifest_sha256=source_manifest_sha,
                    content_shape_hw=shape, content_box_xywh=content_box)
            rgb, log_linear = _validate_output(
                corrected, source, shape, exact_valid_mask=v4r3)
            signal_qa = _fixed_float_input_qa(
                log_linear, required_valid_mask=(corrected.required_valid_mask
                                                 if v4r3 else None))
            ring_qa = (_write_v4_ring(ring_path, corrected, log_linear,
                       pad_px=int(identity["extended_pad_px"])) if v4 else {})
            content_hash = sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
            linear_hash = _write_log_linear(linear_path, log_linear)
            valid_hash = (_write_log_linear(valid_path,
                          np.isfinite(log_linear).all(axis=0).astype(bool))
                          if v4r3 else None)
            output_hash = _frame_output(rgb, run=run, index=index, count=len(frames),
                elapsed_h=float(row["elapsed_h"]), canvas_wh=canvas_wh, destination=target)
            detail_hash = _frame_output(_raw_detail_display(log_linear),
                run=run, index=index, count=len(frames),
                elapsed_h=float(row["elapsed_h"]), canvas_wh=canvas_wh,
                destination=detail_target)
            record = dict(format=FORMAT, run=run, frame_index=index,
                elapsed_h=row["elapsed_h"], source=source, processor=identity,
                certificate=corrected.certificate, native_jpeg_sha256=output_hash,
                v5_raw_detail_jpeg_path=str(detail_target),
                v5_raw_detail_jpeg_sha256=detail_hash,
                v5_raw_detail_display_lut=RAW_DETAIL_LUT,
                signal_qa=signal_qa,
                log_linear_rgb_path=str(linear_path),
                log_linear_rgb_sha256=linear_hash,
                log_linear_rgb_shape=[3, *shape],
                log_linear_rgb_dtype="float32-camera-native-log",
                corrected_content_sha256=content_hash,
                corrected_content_shape_hw=list(shape), corrected_content_dtype="float32-rgb-0-1",
                source_manifest_sha256=source_manifest_sha,
                training_eligibility="pending-native-annotation-boundary-qa")
            if v4r3:
                record.update(v5_log_linear_valid_mask_path=str(valid_path),
                              v5_log_linear_valid_mask_sha256=valid_hash)
            record.update(_write_auxiliary_outputs(corrected, target, output_hash,
                run=run, index=index, count=len(frames), elapsed_h=float(row["elapsed_h"]),
                canvas_wh=canvas_wh))
            record.update(ring_qa)
            _verify_auxiliary_outputs(record, identity, target)
            _write_new(record_path, record)
        output_rows.append(record)
        revised = {k: v for k, v in row.items() if k not in {
            "color_correction_active", "color_correction_flagged", "color_corrections"}}
        revised_rows.append({**revised, "frame_path": str(target),
            "native_jpeg_sha256": record["native_jpeg_sha256"],
            "v5_raw_detail_jpeg_path": record["v5_raw_detail_jpeg_path"],
            "v5_raw_detail_jpeg_sha256": record["v5_raw_detail_jpeg_sha256"],
            "frame_sha256": record["native_jpeg_sha256"],
            "corrected_array_sha256": record["corrected_content_sha256"],
            "v5_raw_path": source.get("camera_arw"),
            "v5_raw_sha256": source.get("camera_arw_sha256"),
            "v5_raw_members": source.get("camera_members"),
            "v5_historical_composite_reproduction": False if prepared_ac61 else None,
            "v5_log_linear_rgb_path": record["log_linear_rgb_path"],
            "v5_log_linear_rgb_sha256": record["log_linear_rgb_sha256"],
            **({} if not v4r3 else dict(
                v5_log_linear_valid_mask_path=record["v5_log_linear_valid_mask_path"],
                v5_log_linear_valid_mask_sha256=record["v5_log_linear_valid_mask_sha256"])),
            **({} if not v4 else dict(
                v5_extended_ring_path=record["v5_extended_ring_path"],
                v5_extended_ring_sha256=record["v5_extended_ring_sha256"],
                v5_extended_pad_px=record["v5_extended_pad_px"],
                v5_extended_origin_xy=record["v5_extended_origin_xy"])),
            "v5_signal_qa": record["signal_qa"],
            **_auxiliary_output_fields(record),
            "v5_frame_receipt_path": str(record_path),
            "v5_frame_receipt_sha256": file_sha256(record_path)})
    if frame_indices is not None:
        # A pilot creates source-exact, resumable per-frame assets but never a
        # partial revision manifest that a trainer could mistake for a series.
        return dict(format=FORMAT, run=run, source_manifest_sha256=source_manifest_sha,
            pilot_only=True, frame_count=len(output_rows), frames=output_rows,
            training_eligibility="pending-native-annotation-boundary-qa")
    ref = dict(old.get("reference") or {})
    bsource = _reference_source(run=run, old=old,
        review_root=Path(revision["review_root"]),
        reference_state=reference_state, reference_state_sha256=reference_state_sha,
        source_manifest_sha256=source_manifest_sha,
        binding_path=reference_binding_path)
    timeline_index = bsource["timeline_frame_index"]
    baseline_path = baseline_dir / "pre_injection.jpg"
    baseline_detail = baseline_dir / "pre_injection.raw_detail.jpg"
    baseline_linear = baseline_dir / "pre_injection.log_linear_rgb.npy"
    baseline_valid = baseline_dir / "pre_injection.log_linear_valid_mask.npy"
    baseline_ring = baseline_dir / "pre_injection.extended_ring.npz"
    baseline_receipt_path = baseline_dir / "pre_injection.v5.json"
    if baseline_receipt_path.is_file():
        baseline_record = _read(baseline_receipt_path)
        if (baseline_record.get("source") != bsource
                or baseline_record.get("processor") != identity
                or baseline_record.get("signal_qa", {}).get("fixed_input_nonclipping_pass") is not True
                or file_sha256(baseline_path) != baseline_record["native_jpeg_sha256"]
                or file_sha256(baseline_detail) != baseline_record["v5_raw_detail_jpeg_sha256"]
                or file_sha256(baseline_linear) != baseline_record["log_linear_rgb_sha256"]
                or (v4r3 and file_sha256(baseline_valid) != baseline_record.get(
                    "v5_log_linear_valid_mask_sha256"))
                or (v4 and file_sha256(baseline_ring) != baseline_record.get(
                    "v5_extended_ring_sha256"))):
            raise ValueError(f"Completed V5 baseline changed: {baseline_path}")
        _verify_auxiliary_outputs(baseline_record, identity, baseline_path)
    else:
        # A partial baseline copy can be resumed only when its hash matches the
        # completed timeline source (or an exact recomputation below).
        if (timeline_index is not None and
                output_rows[timeline_index]["source"]["original_review_jpeg_sha256"] ==
                bsource["original_review_jpeg_sha256"]):
            matched = output_rows[timeline_index]
            if (matched["source"]["camera_arw_sha256"] != bsource["camera_arw_sha256"]
                    or matched["source"]["camera_jpeg_sha256"] != bsource["camera_jpeg_sha256"]):
                raise ValueError("Reference-state acquisition differs from matched timeline")
            baseline_dir.mkdir(parents=True, exist_ok=True)
            _copy_verified(frame_dir / f"frame_{timeline_index:03d}.jpg",
                           baseline_path, matched["native_jpeg_sha256"])
            _copy_verified(frame_dir / f"frame_{timeline_index:03d}.raw_detail.jpg",
                           baseline_detail, matched["v5_raw_detail_jpeg_sha256"])
            _copy_verified(frame_dir / f"frame_{timeline_index:03d}.log_linear_rgb.npy",
                           baseline_linear, matched["log_linear_rgb_sha256"])
            if v4r3:
                _copy_verified(frame_dir / f"frame_{timeline_index:03d}.log_linear_valid_mask.npy",
                               baseline_valid, matched["v5_log_linear_valid_mask_sha256"])
            if v4:
                _copy_verified(frame_dir / f"frame_{timeline_index:03d}.extended_ring.npz",
                               baseline_ring, matched["v5_extended_ring_sha256"])
            baseline_record = dict(format=FORMAT, run=run,
                source=bsource, processor=identity,
                source_timeline_frame_index=timeline_index,
                source_timeline_frame_receipt_sha256=file_sha256(
                    frame_dir / f"frame_{timeline_index:03d}.v5.json"),
                certificate=matched["certificate"],
                signal_qa=matched["signal_qa"],
                corrected_content_sha256=matched["corrected_content_sha256"],
                native_jpeg_sha256=file_sha256(baseline_path),
                v5_raw_detail_jpeg_path=str(baseline_detail),
                v5_raw_detail_jpeg_sha256=file_sha256(baseline_detail),
                v5_raw_detail_display_lut=RAW_DETAIL_LUT,
                log_linear_rgb_sha256=file_sha256(baseline_linear),
                log_linear_rgb_path=str(baseline_linear))
            if v4r3:
                baseline_record.update(
                    v5_log_linear_valid_mask_path=str(baseline_valid),
                    v5_log_linear_valid_mask_sha256=file_sha256(baseline_valid))
            if v4:
                baseline_record.update(
                    v5_extended_ring_path=str(baseline_ring),
                    v5_extended_ring_sha256=file_sha256(baseline_ring),
                    v5_extended_pad_px=matched["v5_extended_pad_px"],
                    v5_extended_origin_xy=matched["v5_extended_origin_xy"],
                    extended_native_center_max_abs_delta_log=matched[
                        "extended_native_center_max_abs_delta_log"],
                    extended_valid_pixels=matched["extended_valid_pixels"],
                    extended_invalid_pixels=matched["extended_invalid_pixels"],
                    extended_valid_fraction=matched["extended_valid_fraction"],
                    extended_max_corrected_linear=matched[
                        "extended_max_corrected_linear"],
                    invalid_sensor_pixels_preserved=True)
            baseline_record.update(_copy_auxiliary_outputs(
                matched, baseline_path, baseline_record["native_jpeg_sha256"]))
        else:
            corrected = processor.correct(run=run, frame_id="pre_injection_reference",
                camera_jpeg=Path(bsource["camera_jpeg"]),
                camera_arw=Path(bsource["camera_arw"]),
                review_jpeg=Path(bsource["original_review_jpeg"]),
                source_manifest_sha256=source_manifest_sha,
                content_shape_hw=shape, content_box_xywh=content_box)
            rgb, log_linear = _validate_output(
                corrected, bsource, shape, exact_valid_mask=v4r3)
            signal_qa = _fixed_float_input_qa(
                log_linear, required_valid_mask=(corrected.required_valid_mask
                                                 if v4r3 else None))
            ring_qa = (_write_v4_ring(baseline_ring, corrected, log_linear,
                       pad_px=int(identity["extended_pad_px"])) if v4 else {})
            linear_hash = _write_log_linear(baseline_linear, log_linear)
            valid_hash = (_write_log_linear(baseline_valid,
                          np.isfinite(log_linear).all(axis=0).astype(bool))
                          if v4r3 else None)
            output_hash = _frame_output(rgb, run=run, index=0, count=1,
                elapsed_h=float(frames[0]["elapsed_h"]),
                canvas_wh=canvas_wh, destination=baseline_path)
            detail_hash = _frame_output(_raw_detail_display(log_linear),
                run=run, index=0, count=1,
                elapsed_h=float(frames[0]["elapsed_h"]),
                canvas_wh=canvas_wh, destination=baseline_detail)
            baseline_record = dict(format=FORMAT, run=run,
                source=bsource, processor=identity,
                source_timeline_frame_index=None,
                certificate=corrected.certificate,
                signal_qa=signal_qa,
                corrected_content_sha256=sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                native_jpeg_sha256=output_hash,
                v5_raw_detail_jpeg_path=str(baseline_detail),
                v5_raw_detail_jpeg_sha256=detail_hash,
                v5_raw_detail_display_lut=RAW_DETAIL_LUT,
                log_linear_rgb_sha256=linear_hash,
                log_linear_rgb_path=str(baseline_linear))
            if v4r3:
                baseline_record.update(
                    v5_log_linear_valid_mask_path=str(baseline_valid),
                    v5_log_linear_valid_mask_sha256=valid_hash)
            baseline_record.update(ring_qa)
            baseline_record.update(_write_auxiliary_outputs(corrected, baseline_path,
                output_hash, run=run, index=0, count=1,
                elapsed_h=float(frames[0]["elapsed_h"]), canvas_wh=canvas_wh))
        _verify_auxiliary_outputs(baseline_record, identity, baseline_path)
        _write_new(baseline_receipt_path, baseline_record)
    new_manifest = dict(old)
    # The V5 image is now the authoritative analysis source. Retain the old
    # lineage only in v5_photometry; analysis_review_root must stop here.
    prior_analysis_source = new_manifest.pop("analysis_source_review_root", None)
    prior_correction_contract = new_manifest.pop("correction_contract", None)
    prior_two_arm = new_manifest.pop("two_arm_photometry", None)
    prior_photometry_revision = new_manifest.pop("photometry_revision_id", None)
    new_manifest["photometry_revision_id"] = "ffac-v5-raw-photometry-20260929"
    prior_domain = new_manifest.get("domain")
    prior_injection_points = new_manifest.get("injection_points")
    prior_phase_quiz_domain = new_manifest.get("phase_quiz_domain")
    if track1:
        # Track1 has a B-photo sand core only.  Old facies/permeability and
        # injector positions are lineage, never accepted physical context.
        new_manifest["domain"] = None
        new_manifest["injection_points"] = None
        new_manifest["phase_quiz_domain"] = None
        new_manifest.pop("v5_run_geometry", None)
        new_manifest.pop("v5_frame_geometry", None)
        new_manifest["v5_track1_photo_geometry"] = frame_geometry
        new_manifest["v5_temporal_reference"] = dict(
            contract="ffac-v5-native-crop-source-supported-v1")
        new_manifest["geometry_revision_id"] = (
            "ffac-v5-track1-photo-" + source_manifest_sha[:20])
        if identity.get("enclosed_hole_policy", "keep") != "keep":
            new_manifest["geometry_revision_id"] += "-enclosed-filled-v1"
    else:
        new_manifest["domain"] = geometry["domain"]
        new_manifest["injection_points"] = geometry["injection_points"]
        new_manifest["phase_quiz_domain"] = None
        new_manifest["v5_run_geometry"] = geometry["v5_run_geometry"]
        if frame_geometry is not None:
            new_manifest["v5_frame_geometry"] = frame_geometry
        new_manifest["geometry_revision_id"] = (
            "ffac-v5-run-native-" + geometry["v5_run_geometry"]["receipt_sha256"][:20])
    new_manifest["jpeg_quality"] = 100
    new_manifest["jpeg_chroma_subsampling"] = "4:4:4"
    new_manifest["correction_contract"] = dict(
        kind="ffac-v5-full-native-camera-linear",
        processor=identity,
        **({"track1_photo_geometry_contract": frame_geometry["contract"]} if track1
           else {"run_native_geometry_receipt_sha256":
                 geometry["v5_run_geometry"]["receipt_sha256"]}),
        frame_sources_are_source_sha_bound=True,
        scientific_signal="log-camera-linear-float32-sidecar",
        chemistry_alignment_mode="none-light-only",
        current_trainer_input="sha-bound-float32-log-camera-linear-sidecar",
        unit_denominator=65535,
        display_jpeg_is_training_input=False)
    if frame_geometry is not None:
        new_manifest["correction_contract"].update(
            frame_geometry_contract=frame_geometry["contract"],
            frame_geometry_receipts_sha256=sha256(json.dumps(
                frame_geometry["frames"], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest())
    new_manifest["correction_contract"]["run_geometry_validation_only"] = (
        True if track1 else bool(
            geometry["v5_run_geometry"].get("validation_only", False)))
    new_manifest["correction_contract"]["scientific_mass_eligible"] = False
    new_manifest["frame_sources"] = revised_rows
    new_manifest["reference"] = {**ref,
        "kind": "v5-source-bound-run-pre-injection-reference",
        "included_in_timeline": timeline_index is not None,
        "timeline_frame_index": timeline_index,
        "materialized_native_sha256": baseline_record["native_jpeg_sha256"],
        "corrected_array_sha256": baseline_record["corrected_content_sha256"],
        "v5_reference_state": bsource,
        "v5_baseline_receipt_path": str(baseline_receipt_path),
        "v5_baseline_receipt_sha256": file_sha256(baseline_receipt_path),
        "v5_log_linear_rgb_path": str(baseline_linear),
        "v5_log_linear_rgb_sha256": baseline_record["log_linear_rgb_sha256"],
        **({} if not v4r3 else dict(
            v5_log_linear_valid_mask_path=str(baseline_valid),
            v5_log_linear_valid_mask_sha256=baseline_record[
                "v5_log_linear_valid_mask_sha256"])),
        **({} if not v4 else dict(
            v5_extended_ring_path=baseline_record["v5_extended_ring_path"],
            v5_extended_ring_sha256=baseline_record["v5_extended_ring_sha256"],
            v5_extended_pad_px=baseline_record["v5_extended_pad_px"],
            v5_extended_origin_xy=baseline_record["v5_extended_origin_xy"])),
        "v5_raw_detail_jpeg_path": str(baseline_detail),
        "v5_raw_detail_jpeg_sha256": baseline_record["v5_raw_detail_jpeg_sha256"],
        **_auxiliary_output_fields(baseline_record),
        "v5_signal_qa": baseline_record["signal_qa"]}
    new_manifest["v5_photometry"] = dict(format=FORMAT, source_manifest=str(source_manifest),
        source_manifest_sha256=source_manifest_sha, source_review_root=revision["review_root"],
        processor=identity, same_frame_order_timestamps_and_native_canvas=True,
        original_analysis_source_review_root=prior_analysis_source,
        original_correction_contract=prior_correction_contract,
        original_two_arm_photometry=prior_two_arm,
        original_photometry_revision_id=prior_photometry_revision,
        original_domain=prior_domain,
        original_injection_points=prior_injection_points,
        original_phase_quiz_domain=prior_phase_quiz_domain,
        **({"track1_photo_geometry": frame_geometry,
            "photo_sand_qa_scope": frame_geometry.get(
                "photo_sand_support_contract", "transported_safe_B_sand_core_only"),
            "source_photo_acceptance_is_not_label_or_map_acceptance": True}
           if track1 else {"run_native_geometry": geometry["v5_run_geometry"]}),
        run_geometry_validation_only=(True if track1 else bool(
            geometry["v5_run_geometry"].get("validation_only", False))),
        scientific_mass_eligible=False,
        trainer_feature_input="sha-bound-float32-log-camera-linear-sidecar",
        chemistry_alignment_mode="none-light-only",
        unit_denominator=65535,
        display_jpeg_is_training_input=False,
        raw_detail_display_lut=RAW_DETAIL_LUT,
        lossless_corrected_signal="native-float32-log-camera-linear-sidecar",
        label_transfer_status="pending-native-annotation-boundary-qa",
        training_eligible=False)
    if track1 and identity.get("display_mode") == "photographic_v1":
        new_manifest["v5_photometry"]["photographic_display"] = identity[
            "display_transfer"]
    if v4r3:
        new_manifest["v5_float_support_policy"] = EXACT_VALID_MASK_POLICY
    if frame_geometry is not None:
        new_manifest["v5_photometry"]["frame_geometry_contract"] = frame_geometry["contract"]
    manifest_path = frame_dir / "manifest.json"
    if manifest_path.is_file():
        if _read(manifest_path) != new_manifest:
            raise ValueError("Completed V5 native manifest differs from expected immutable content")
    else:
        _write_new(manifest_path, new_manifest)
    if track1:
        from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
            Track1FramePhotoCohort,
        )
        Track1FramePhotoCohort(manifest_path).frame(0)
    else:
        verify_accepted_geometry(source_manifest, manifest_path)
    receipt = dict(**job, materialized_review_root=str(review_root),
        materialized_manifest_path=str(manifest_path),
        materialized_manifest_sha256=file_sha256(manifest_path),
        materialized_baseline_path=str(baseline_path),
        materialized_baseline_sha256=file_sha256(baseline_path),
        materialized_baseline_receipt_path=str(baseline_receipt_path),
        materialized_baseline_receipt_sha256=file_sha256(baseline_receipt_path),
        frames=output_rows)
    receipt_path = root / "materialization_receipt.json"
    if receipt_path.is_file():
        if _read(receipt_path) != receipt:
            raise ValueError("Completed V5 receipt differs from expected immutable content")
    else:
        _write_new(receipt_path, receipt)
    return receipt


def source_inventory(request_path: Path) -> dict[str, Any]:
    """Read-only inventory of exact source revisions and RAW pairing gaps."""
    revisions = source_manifests(Path(request_path))
    rows = []
    for revision in revisions:
        manifest = _read(Path(revision["manifest_path"]))
        missing_arw = []
        for frame in manifest["frame_sources"]:
            source = Path(frame["source_path"])
            if not source.with_suffix(".ARW").is_file():
                missing_arw.append(int(frame["frame_index"]))
        reference = manifest.get("reference") or {}
        rows.append(dict(run=revision["run"], source_manifest_sha256=revision["manifest_sha256"],
            source_manifest_path=revision["manifest_path"], frame_count=manifest["frame_count"],
            missing_paired_arw=missing_arw,
            baseline_in_timeline=bool(reference.get("included_in_timeline")),
            baseline_frame_index=reference.get("timeline_frame_index"),
            source_photometry_revision=manifest.get("photometry_revision_id")))
    return dict(format=FORMAT, request_path=str(request_path),
        request_sha256=file_sha256(request_path), source_revisions=len(rows),
        timeline_frames=sum(row["frame_count"] for row in rows), rows=rows)


def aggregate_materialization(
    request_path: Path, output_root: Path, *,
    source_manifest_sha256s: set[str] | None = None,
    admission_path: Path | None = None,
    admission_sha256: str | None = None,
) -> dict[str, Any]:
    """Seal an exact admitted native source set after every revision is verified.

    The resulting immutable batch receipt is the only source the later
    annotation-rebinding job should accept. It does not mark labels eligible.
    """
    request_path, output_root = Path(request_path), Path(output_root)
    request = _read(request_path)
    frozen_rows = source_manifests(request_path)
    if source_manifest_sha256s is None:
        source_rows = frozen_rows
    else:
        if (len(source_manifest_sha256s) != 43 or
                not source_manifest_sha256s.issubset(
                    {row["manifest_sha256"] for row in frozen_rows})):
            raise ValueError("Current43 aggregate requires 43 frozen source revisions")
        source_rows = [row for row in frozen_rows
                       if row["manifest_sha256"] in source_manifest_sha256s]
        if (len(source_rows) != 43 or len({row["run"] for row in source_rows}) != 43
                or sum(_read(Path(row["manifest_path"]))["frame_count"]
                       for row in source_rows) != 864):
            raise ValueError("Current43 aggregate is not exact 43/864")
    revisions = []
    processor_identity = None
    geometry_contracts: set[str] = set()
    geometry_validation_scopes: set[bool] = set()
    frame_geometry_contracts: set[str] = set()
    for source in source_rows:
        root = output_root / "revisions" / source["manifest_sha256"]
        path = root / "materialization_receipt.json"
        if not path.is_file():
            raise FileNotFoundError(f"V5 source revision is not complete: {path}")
        row = _read(path)
        if (row.get("format") != FORMAT or row.get("run") != source["run"]
                or row.get("source_manifest_sha256") != source["manifest_sha256"]
                or file_sha256(Path(row["source_manifest_path"])) != source["manifest_sha256"]
                or file_sha256(Path(row["materialized_manifest_path"])) != row["materialized_manifest_sha256"]
                or file_sha256(Path(row["materialized_baseline_path"])) != row["materialized_baseline_sha256"]
                or len(row.get("frames") or []) != row["frame_count"]):
            raise ValueError(f"V5 source revision receipt is inconsistent: {path}")
        materialized_native = _read(Path(row["materialized_manifest_path"]))
        track1 = (row.get("processor") or {}).get("contract") == TRACK1_PROCESSOR_CONTRACT
        if track1:
            if (materialized_native.get("v5_run_geometry") is not None
                    or row.get("run_geometry_receipt_sha256") is not None
                    or materialized_native.get("domain") is not None
                    or materialized_native.get("injection_points") is not None):
                raise ValueError("Track1 optical revision inherited uncertified physics")
            physical_geometry = None
            geometry_contracts.add("ffac-v5-track1-photo-only-no-physical-geometry-v1")
            geometry_validation_scopes.add(True)
            marker = materialized_native.get("v5_track1_photo_geometry") or {}
        else:
            physical_geometry = verify_accepted_geometry(
                Path(source["manifest_path"]), Path(row["materialized_manifest_path"]))
            geometry_contracts.add(physical_geometry["contract"])
            geometry_validation_scopes.add(bool(physical_geometry["validation_only"]))
            if (physical_geometry["source_manifest_sha256"] != source["manifest_sha256"]
                    or row.get("run_geometry_receipt_sha256") !=
                    physical_geometry["receipt_sha256"]):
                raise ValueError(f"V5 revision has an unrelated run-native geometry: {path}")
            marker = materialized_native.get("v5_frame_geometry") or {}
        v4 = (row.get("processor") or {}).get("contract") == (
            "ffac-v4-source-bound-native-processor-v1")
        v4r3 = (row.get("processor") or {}).get("contract") in {
            "ffac-v4r3-source-bound-native-processor-v1", TRACK1_PROCESSOR_CONTRACT}
        if v4r3 and materialized_native.get("v5_float_support_policy") != EXACT_VALID_MASK_POLICY:
            raise ValueError("V4-R3 full source lacks accepted exact finite-mask policy")
        if track1:
            from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
                COHORT_CONTRACT as TRACK1_PHOTO_COHORT, bind_photo_only_marker,
            )
            if (row.get("v5_track1_photo_geometry") != marker
                    or marker.get("contract") != TRACK1_PHOTO_COHORT
                    or marker.get("original_source_manifest_sha256") !=
                    source["manifest_sha256"]
                    or len(marker.get("frames") or []) != row["frame_count"]):
                raise ValueError(f"Track1 revision lacks exact photo transport: {path}")
            bound = bind_photo_only_marker(Path(source["manifest_path"]),
                dict(run=source["run"], source_manifest_path=source["manifest_path"],
                     source_manifest_sha256=source["manifest_sha256"],
                     frames=marker["frames"]),
                enclosed_hole_policy=marker.get("enclosed_hole_policy", "keep"))
            if marker != bound:
                raise ValueError("Track1 materialized photo marker changed")
            frame_geometry_contracts.add(marker["contract"])
        elif v4 or v4r3:
            if (row.get("v5_frame_geometry") != marker
                    or marker.get("original_source_manifest_sha256") != source["manifest_sha256"]
                    or marker.get("contract") !=
                    "ffac-v5-validation-only-frame-geometry-cohort-v1"
                    or len(marker.get("frames") or []) != row["frame_count"]):
                raise ValueError(f"V4 revision lacks its complete frame geometry: {path}")
            _bound_frame_geometry_marker(Path(source["manifest_path"]), marker)
            frame_geometry_contracts.add(marker["contract"])
        elif marker:
            raise ValueError("Legacy V5 revision unexpectedly carries V4 frame geometry")
        reference_receipt = (Path(row["materialized_baseline_path"]).with_name(
            "pre_injection.v5.json"))
        reference = _read(reference_receipt)
        if (reference.get("signal_qa", {}).get("fixed_input_nonclipping_pass") is not True
                or file_sha256(Path(reference["log_linear_rgb_path"])) !=
                reference["log_linear_rgb_sha256"]
                or file_sha256(Path(reference["v5_raw_detail_jpeg_path"])) !=
                reference["v5_raw_detail_jpeg_sha256"]
                or file_sha256(Path(row["materialized_baseline_path"])) !=
                reference["native_jpeg_sha256"]):
            raise ValueError(f"V5 reference float signal is uncertified: {reference_receipt}")
        if (v4r3 and file_sha256(Path(reference.get(
                "v5_log_linear_valid_mask_path") or "")) != reference.get(
                "v5_log_linear_valid_mask_sha256")):
            raise ValueError(f"V4-R3 reference validity mask changed: {reference_receipt}")
        _verify_auxiliary_outputs(reference, row["processor"], Path(row["materialized_baseline_path"]))
        if (v4 and (reference.get("v5_extended_pad_px") != 32
                or reference.get("v5_extended_origin_xy") != [-32, -32]
                or file_sha256(Path(reference.get("v5_extended_ring_path") or "")) !=
                reference.get("v5_extended_ring_sha256"))):
            raise ValueError(f"V4 source-backed reference ring changed: {reference_receipt}")
        if processor_identity is None:
            processor_identity = row["processor"]
        elif row["processor"] != processor_identity:
            raise ValueError("V5 source revisions used different processor identities")
        frame_dir = Path(row["materialized_manifest_path"]).parent
        for index, frame in enumerate(row["frames"]):
            target = frame_dir / f"frame_{index:03d}.jpg"
            frame_receipt = frame_dir / f"frame_{index:03d}.v5.json"
            if (frame.get("frame_index") != index
                    or frame.get("signal_qa", {}).get("fixed_input_nonclipping_pass") is not True
                    or file_sha256(target) != frame["native_jpeg_sha256"]
                    or file_sha256(Path(frame["v5_raw_detail_jpeg_path"])) !=
                    frame["v5_raw_detail_jpeg_sha256"]
                    or file_sha256(Path(frame["log_linear_rgb_path"])) != frame["log_linear_rgb_sha256"]
                    or (v4r3 and file_sha256(Path(frame.get(
                        "v5_log_linear_valid_mask_path") or "")) != frame.get(
                        "v5_log_linear_valid_mask_sha256"))
                    or _read(frame_receipt) != frame):
                raise ValueError(f"V5 frame changed or is missing: {target}")
            _verify_auxiliary_outputs(frame, row["processor"], target)
            if (v4 and (frame.get("v5_extended_pad_px") != 32
                    or frame.get("v5_extended_origin_xy") != [-32, -32]
                    or file_sha256(Path(frame.get("v5_extended_ring_path") or "")) !=
                    frame.get("v5_extended_ring_sha256"))):
                raise ValueError(f"V4 source-backed frame ring changed: {target}")
        revisions.append(dict(run=source["run"],
            source_manifest_path=source["manifest_path"],
            source_manifest_sha256=source["manifest_sha256"],
            materialized_review_root=row["materialized_review_root"],
            materialized_manifest_path=row["materialized_manifest_path"],
            materialized_manifest_sha256=row["materialized_manifest_sha256"],
            materialized_baseline_path=row["materialized_baseline_path"],
            materialized_baseline_sha256=row["materialized_baseline_sha256"],
            frame_count=row["frame_count"], receipt_path=str(path),
            receipt_sha256=file_sha256(path),
            **({"track1_photo_geometry_contract": marker["contract"],
                "track1_photo_geometry_receipt_count": len(marker["frames"])}
               if track1 else dict(
                   run_geometry_receipt_path=physical_geometry["receipt_path"],
                   run_geometry_receipt_sha256=physical_geometry["receipt_sha256"],
                   run_native_context_sha256=physical_geometry["context_npz_sha256"])),
            **({} if not v4 else dict(
                frame_geometry_contract=marker["contract"],
                frame_geometry_receipt_count=len(marker["frames"])))))
    by_source_sha = {row["source_manifest_sha256"]: row for row in revisions}
    # A frozen two-arm source can be a derived review whose own manifest SHA
    # differs from the canonical prepared review used by Studio.  Its explicit
    # one-step parent binding is the only permitted additional match.
    by_canonical_parent_sha: dict[str, list[dict[str, Any]]] = {}
    for row in revisions:
        source_manifest = _read(Path(row["source_manifest_path"]))
        two_arm = source_manifest.get("two_arm_photometry") or {}
        parent_sha = two_arm.get("source_manifest_sha256")
        if parent_sha:
            by_canonical_parent_sha.setdefault(parent_sha, []).append(row)
    parent = Path(request["settings"]["parent_model"])
    jobs = _read(parent / "review_source" / "postflight_manifest.json")["jobs"]
    canonical = []
    for job in jobs:
        run = str(job["run"]).casefold()
        root = analysis_review_root(
            Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
        original_path = root / "native_frames" / run / "manifest.json"
        digest = file_sha256(original_path)
        candidates = [r for r in ([by_source_sha[digest]] if digest in by_source_sha else [])
                      + by_canonical_parent_sha.get(digest, []) if r["run"] == run]
        # A direct source and its derived child must not both be selected for
        # the same Studio campaign without an explicit, unique frozen mapping.
        if len(candidates) != 1:
            raise ValueError(f"Canonical Studio source absent from V5 materialization: {run}")
        row = candidates[0]
        canonical.append(dict(run=run, canonical_source_manifest_sha256=digest,
            frozen_source_manifest_sha256=row["source_manifest_sha256"],
            materialized_review_root=row["materialized_review_root"],
            materialized_manifest_path=row["materialized_manifest_path"],
            materialized_manifest_sha256=row["materialized_manifest_sha256"]))
    if len(canonical) != 43 or len({row["run"] for row in canonical}) != 43:
        raise ValueError("Expected 43 unique canonical FF_AC Studio runs")
    if processor_identity and processor_identity.get("contract") in {
            "ffac-v4r3-source-bound-native-processor-v1", TRACK1_PROCESSOR_CONTRACT}:
        verify_v4r3_full_source_admission(
            processor_identity,
            all_source_manifest_sha256s={row["manifest_sha256"] for row in source_rows},
            source_manifest_runs={row["manifest_sha256"]: row["run"] for row in source_rows},
            training_request_sha256=file_sha256(request_path),
            admission_path=admission_path,
            admission_sha256=admission_sha256)
    track1_batch = processor_identity.get("contract") == TRACK1_PROCESSOR_CONTRACT
    if (len(geometry_contracts) != 1 or len(geometry_validation_scopes) != 1
            or (processor_identity.get("contract") in {
                "ffac-v4-source-bound-native-processor-v1", TRACK1_PROCESSOR_CONTRACT}
                and len(frame_geometry_contracts) != 1)):
        raise ValueError("V5 source revisions mix physical-geometry acceptance scopes")
    receipt = dict(format=BATCH_FORMAT,
        frozen_training_request_path=str(request_path),
        frozen_training_request_sha256=file_sha256(request_path),
        processor_identity=processor_identity,
        source_revisions=len(revisions),
        timeline_frames=sum(row["frame_count"] for row in revisions),
        canonical_runs=len(canonical), revisions=revisions,
        **({"source_admission_receipt_path": str(admission_path),
            "source_admission_receipt_sha256": admission_sha256}
           if track1_batch and admission_path is not None else {}),
        accepted_run_native_geometry_revisions=(0 if track1_batch else len(revisions)),
        run_geometry_contract=next(iter(geometry_contracts)),
        run_geometry_validation_only=next(iter(geometry_validation_scopes)),
        **({"track1_photo_geometry_contract":
            "ffac-v5-track1-photo-geometry-cohort-v1",
            "track1_photo_geometry_revisions": len(revisions),
            "track1_photo_geometry_receipts": sum(
                row["track1_photo_geometry_receipt_count"] for row in revisions),
            "photo_sand_qa_scope": "transported_safe_B_sand_core_only",
            "training_label_eligibility": "requires-separate-4048-id-partition",
            "rig_metal_certified": False,
            "dynamic_overflow_certified": False}
           if track1_batch else {}),
        scientific_mass_eligible=False,
        canonical_mapping=canonical,
        training_eligibility="pending-native-annotation-boundary-qa")
    if frame_geometry_contracts:
        receipt.update(frame_geometry_contract=next(iter(frame_geometry_contracts)),
            frame_geometry_revisions=len(revisions),
            frame_geometry_receipts=sum(
                row["track1_photo_geometry_receipt_count"] if track1_batch else
                row["frame_geometry_receipt_count"] for row in revisions))
    path = output_root / "materialization_receipt.json"
    if path.is_file():
        if _read(path) != receipt:
            raise ValueError("Existing V5 aggregate receipt differs from current source set")
    else:
        _write_new(path, receipt)
    return receipt


from darsia.fluidflower.tools.ffac_track1_mixed52_materialization import (  # noqa: E402
    aggregate_track1_mixed52_materialization,
    verify_track1_mixed52_source_admission,
)
