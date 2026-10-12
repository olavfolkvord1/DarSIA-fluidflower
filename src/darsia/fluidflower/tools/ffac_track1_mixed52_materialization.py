"""Seal and verify source-exact Track1 processors as one 52-revision cohort.

The current and historical source renderers have different immutable identities.
This module never substitutes a run-level fit for a historical source revision.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_training.v5_linear_input import EXACT_VALID_MASK_POLICY
from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
    COHORT_CONTRACT, bind_photo_only_marker,
)
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests


COHORT_CONTRACT_MIXED = "ffac-v5-track1-mixed52-source-materialization-v1"
ADMISSION_CONTRACT = "ffac-v5-track1-mixed52-provisional-source-admission-v1"
PARTITION_CONTRACT = "ffac-v5-track1-frozen-review-fit-evaluation-partition-v1"
FLOAT_CONTRACT = "ffac-v5-track1-native-float-range-preflight-v1"
DECISION_CONTRACT = "ffac-v5-track1-provisional-photometry-decision-v1"
PROCESSOR_CONTRACT = "ffac-v5-track1-source-bound-native-processor-v1"
AC44_ALT_SOURCE_SHA = "fdcd7f7ff6f8718180db82fbb25ffa788b185c51effefb25d9d8b20f2f01898e"
AC44_ALT_ARW_SHA = "d7ea7f5a986186adc27527e82f9abd10f568ac72aad439fa44af3b36bcf50ce8"
AC44_ALT_CAMERA_SHA = "dbf9525c4cc73c427b086c4739cf9eb851c56d51ae4dc7a0de07ec63bf9a045c"
AC44_ALT_NATIVE_SHA = "5c4d2c2fe1f1a805396f809c356f2a63bdbcf3edfa8d85fa0a218b6718584d8e"
AC44_ALT_RENDERER_SHA = "d7d9e942a425a5d1526861737e63d9f4b1a1b0bb0e07c97db924f0f2a13d0497"
AC44_ALT_PROCESSOR_SHA = "2120e335d6924b09d954677b520fdd43a76c14b942eedd5490fb1822e911667b"
HISTORICAL_AC44_ALT_SOURCE_SHA = "e746d4bf8bc9a3d35a22cccf27f8cfd5788a5959e38b702e3269df6d83671a17"
HISTORICAL_AC44_ALT_NATIVE_SHA = "61e1634ac874ed06edf3c5766e96b2e3a1451522dc9412bd5e63313220f12ea4"
HISTORICAL_AC44_ALT_RENDERER_SHA = "d12c29d0ce056d499cfd052ca99cb803d2d11d8904811d471da68f2980bc87c6"
HISTORICAL_AC44_ALT_PROCESSOR_SHA = "bf22b7d2259878d0149702d6d94a242e9cf4c32f0a4d16ea57fb024271204e0f"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _identity_sha(identity: dict[str, Any]) -> str:
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _bound_file(path: Any, expected: Any, label: str) -> Path:
    if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} is not SHA-bound")
    target = Path(path)
    if not target.is_file() or _sha(target) != expected:
        raise ValueError(f"{label} changed or is missing")
    return target


def _sealed_payload(path: Any, expected: Any, label: str) -> Path:
    """Inspect a per-frame payload sealed by materialize_revision.

    Finalizing a revision already live-rehashes its complete float, mask and
    JPEG payload. RunV5LinearInput rehashes a frame again on every model load.
    The batch does not add a third 200-GB NAS scan between those two checks.
    """
    if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} has no recorded SHA")
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"{label} is absent from sealed revision")
    return target


def _source_groups(request_path: Path, current_shas: set[str]) -> tuple[list[dict], dict]:
    sources = source_manifests(request_path)
    by_sha = {row["manifest_sha256"]: row for row in sources}
    if (len(sources) != 52 or len(by_sha) != 52 or len(current_shas) != 43
            or not current_shas.issubset(by_sha)):
        raise ValueError("Mixed Track1 source inventory is not exact 43+9")
    current = [row for row in sources if row["manifest_sha256"] in current_shas]
    history = [row for row in sources if row["manifest_sha256"] not in current_shas]
    if (len(current) != 43 or len(history) != 9
            or sum(_read(Path(row["manifest_path"]))["frame_count"]
                   for row in current) != 864
            or sum(_read(Path(row["manifest_path"]))["frame_count"]
                   for row in history) != 181):
        raise ValueError("Mixed Track1 inventory is not exact 864+181")
    return sources, by_sha


def aggregate_track1_mixed52_materialization(
    request_path: Path, current_root: Path, historical_root: Path,
    batch_root: Path, *, current_source_manifest_sha256s: set[str],
    admission_path: Path, admission_sha256: str,
    ac44_alt_root: Path | None = None,
    historical_ac44_alt_root: Path | None = None,
) -> dict[str, Any]:
    """Verify every source/float receipt before writing a mixed-cohort batch."""
    from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
        BATCH_FORMAT, FORMAT, _write_new,
    )

    request_path = Path(request_path)
    sources, by_sha = _source_groups(request_path, current_source_manifest_sha256s)
    roots = {"current43": Path(current_root), "historical9": Path(historical_root)}
    if ac44_alt_root is not None:
        roots["current43_ac44_alt"] = Path(ac44_alt_root)
    if historical_ac44_alt_root is not None:
        roots["historical9_ac44_alt"] = Path(historical_ac44_alt_root)
    revisions: list[dict[str, Any]] = []
    identities: dict[str, dict[str, Any]] = {}
    for source in sources:
        digest = source["manifest_sha256"]
        group = ("current43_ac44_alt" if digest == AC44_ALT_SOURCE_SHA and
                 ac44_alt_root is not None else
                 "historical9_ac44_alt" if digest == HISTORICAL_AC44_ALT_SOURCE_SHA
                 and historical_ac44_alt_root is not None else
                 "current43" if digest in current_source_manifest_sha256s else
                 "historical9")
        receipt_path = roots[group] / "revisions" / digest / "materialization_receipt.json"
        receipt = _read(receipt_path)
        native_path = _bound_file(receipt.get("materialized_manifest_path"),
                                  receipt.get("materialized_manifest_sha256"),
                                  "materialized native manifest")
        original = _bound_file(source["manifest_path"], digest, "original source manifest")
        if (receipt.get("format") != FORMAT or receipt.get("run") != source["run"]
                or receipt.get("source_manifest_sha256") != digest
                or Path(receipt["source_manifest_path"]).resolve() != original.resolve()
                or receipt.get("frame_count") != len(receipt.get("frames") or [])
                or receipt.get("frame_count") != _read(original)["frame_count"]):
            raise ValueError(f"Mixed source revision receipt changed: {digest}")
        native = _read(native_path)
        identity = receipt.get("processor")
        if not isinstance(identity, dict) or identity.get("contract") != PROCESSOR_CONTRACT:
            raise ValueError(f"Mixed source lacks Track1 processor: {digest}")
        old_identity = identities.setdefault(group, identity)
        if old_identity != identity:
            raise ValueError(f"Mixed source group changed processor: {group}")
        if (native.get("v5_float_support_policy") != EXACT_VALID_MASK_POLICY
                or native.get("v5_run_geometry") is not None
                or native.get("v5_frame_geometry") is not None
                or native.get("domain") is not None):
            raise ValueError("Mixed Track1 source inherited physical or unmasked context")
        marker = native.get("v5_track1_photo_geometry") or {}
        if (marker.get("contract") != COHORT_CONTRACT
                or marker.get("original_source_manifest_sha256") != digest
                or len(marker.get("frames") or []) != receipt["frame_count"]
                or receipt.get("v5_track1_photo_geometry") != marker):
            raise ValueError(f"Mixed source lacks exact photo transport: {digest}")
        rebound = bind_photo_only_marker(original, dict(
            run=source["run"], source_manifest_path=str(original),
            source_manifest_sha256=digest, frames=marker["frames"]))
        if rebound != marker:
            raise ValueError(f"Mixed photo marker no longer binds source: {digest}")
        baseline = _bound_file(receipt.get("materialized_baseline_path"),
                               receipt.get("materialized_baseline_sha256"),
                               "materialized baseline JPEG")
        baseline_receipt = _read(_bound_file(
            receipt.get("materialized_baseline_receipt_path"),
            receipt.get("materialized_baseline_receipt_sha256"),
            "materialized baseline receipt"))
        if baseline_receipt.get("native_jpeg_sha256") != _sha(baseline):
            raise ValueError("Mixed baseline JPEG/receipt disagree")
        for frame_index, frame in enumerate(receipt["frames"]):
            frame_receipt = native_path.parent / f"frame_{frame_index:03d}.v5.json"
            if (frame.get("frame_index") != frame_index
                    or _read(frame_receipt) != frame
                    or frame.get("signal_qa", {}).get("fixed_input_nonclipping_pass") is not True):
                raise ValueError(f"Mixed frame receipt or range invalid: {digest}:{frame_index}")
            if group == "current43_ac44_alt" and frame_index == 18:
                acquisition = frame.get("source") or {}
                if (acquisition.get("camera_arw_sha256") != AC44_ALT_ARW_SHA
                        or acquisition.get("camera_jpeg_sha256") != AC44_ALT_CAMERA_SHA
                        or acquisition.get("original_review_jpeg_sha256") !=
                           AC44_ALT_NATIVE_SHA):
                    raise ValueError("AC44 alternate Sony lens applied to another acquisition")
            if group == "historical9_ac44_alt" and frame_index == 21:
                acquisition = frame.get("source") or {}
                if (acquisition.get("camera_arw_sha256") != AC44_ALT_ARW_SHA
                        or acquisition.get("camera_jpeg_sha256") != AC44_ALT_CAMERA_SHA
                        or acquisition.get("original_review_jpeg_sha256") !=
                           HISTORICAL_AC44_ALT_NATIVE_SHA):
                    raise ValueError("Historical AC44 lens applied to another acquisition")
            if (digest ==
                    "e6e25869e9757dd347b57fb18874d386fb92b00de80c138df9c08045ee95443f"
                    and frame_index == 0):
                source_members = (frame.get("source") or {}).get("camera_members") or []
                cert = frame.get("certificate") or {}
                if (group != "historical9" or len(source_members) != 3
                        or (frame.get("source") or {}).get("source_kind") !=
                           "new-v5-median-of-three-corrected-camera-raws"
                        or cert.get("camera_members") != [
                            {"camera_jpeg_sha256": member.get("camera_jpeg_sha256"),
                             "camera_arw_sha256": member.get("camera_arw_sha256")}
                            for member in source_members]
                        or cert.get("exact_historical_composite_reconstruction") is not False):
                    raise ValueError("AC61 historical composite lost its three-RAW lineage")
            _sealed_payload(frame.get("log_linear_rgb_path"),
                            frame.get("log_linear_rgb_sha256"), "float log RAW frame")
            _sealed_payload(frame.get("v5_log_linear_valid_mask_path"),
                            frame.get("v5_log_linear_valid_mask_sha256"), "float-valid mask")
            _sealed_payload(str(native_path.parent / f"frame_{frame_index:03d}.jpg"),
                            frame.get("native_jpeg_sha256"), "native display JPEG")
            _sealed_payload(frame.get("v5_raw_detail_jpeg_path"),
                            frame.get("v5_raw_detail_jpeg_sha256"),
                            "raw-detail display JPEG")
        revisions.append(dict(
            run=source["run"], source_group=group,
            source_manifest_path=str(original), source_manifest_sha256=digest,
            processor_identity_sha256=_identity_sha(identity),
            materialized_review_root=receipt["materialized_review_root"],
            materialized_manifest_path=str(native_path),
            materialized_manifest_sha256=receipt["materialized_manifest_sha256"],
            materialized_baseline_path=str(baseline),
            materialized_baseline_sha256=receipt["materialized_baseline_sha256"],
            frame_count=receipt["frame_count"], receipt_path=str(receipt_path),
            receipt_sha256=_sha(receipt_path),
            track1_photo_geometry_contract=COHORT_CONTRACT,
            track1_photo_geometry_receipt_count=len(marker["frames"])))
    if (set(identities) != set(roots) or
            len({_identity_sha(identity) for identity in identities.values()}) != len(identities)):
        raise ValueError("Mixed Track1 must have distinct source processors")
    selected = {row["source_manifest_sha256"]: row for row in revisions}
    request = _read(request_path)
    parent = Path(request["settings"]["parent_model"])
    jobs = _read(parent / "review_source" / "postflight_manifest.json")["jobs"]
    canonical = []
    for job in jobs:
        run = str(job["run"]).casefold()
        original_root = analysis_review_root(
            Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
        canonical_sha = _sha(original_root / "native_frames" / run / "manifest.json")
        candidates = []
        for row in revisions:
            if (row["run"] != run or row["source_group"] not in
                    {"current43", "current43_ac44_alt"}):
                continue
            original_manifest = _read(Path(row["source_manifest_path"]))
            parent_sha = (original_manifest.get("two_arm_photometry") or {}).get(
                "source_manifest_sha256")
            if canonical_sha in {row["source_manifest_sha256"], parent_sha}:
                candidates.append(row)
        if len(candidates) != 1:
            raise ValueError(f"Canonical Studio source has no unique current revision: {run}")
        row = candidates[0]
        canonical.append(dict(run=run,
            canonical_source_manifest_sha256=canonical_sha,
            frozen_source_manifest_sha256=row["source_manifest_sha256"],
            materialized_review_root=row["materialized_review_root"],
            materialized_manifest_path=row["materialized_manifest_path"],
            materialized_manifest_sha256=row["materialized_manifest_sha256"]))
    if len(canonical) != 43 or sum(row["frame_count"] for row in revisions) != 1045:
        raise ValueError("Mixed source count is not 52 revisions/1045 frames")
    receipt = dict(format=BATCH_FORMAT,
        source_cohort_contract=COHORT_CONTRACT_MIXED,
        frozen_training_request_path=str(request_path),
        frozen_training_request_sha256=_sha(request_path),
        processor_identities=identities,
        source_revisions=52, timeline_frames=1045, canonical_runs=43,
        revisions=revisions, canonical_mapping=canonical,
        source_admission_receipt_path=str(admission_path),
        source_admission_receipt_sha256=admission_sha256,
        frame_payload_sha_verified_at_revision_seal=True,
        frame_payload_sha_reverified_on_model_load=True,
        accepted_run_native_geometry_revisions=0,
        run_geometry_contract="ffac-v5-track1-photo-only-no-physical-geometry-v1",
        run_geometry_validation_only=True,
        track1_photo_geometry_contract=COHORT_CONTRACT,
        track1_photo_geometry_revisions=52,
        track1_photo_geometry_receipts=1045,
        photo_sand_qa_scope="transported_safe_B_sand_core_only",
        training_label_eligibility="separate-4048-id-fit-eval-partition",
        rig_metal_certified=False, dynamic_overflow_certified=False,
        scientific_mass_eligible=False,
        training_eligibility="studio-validation-only-provisional")
    verify_track1_mixed52_source_admission(
        receipt, admission_path=admission_path, admission_sha256=admission_sha256,
        frozen_training_request_sha256=_sha(request_path))
    target = Path(batch_root) / "materialization_receipt.json"
    if target.exists():
        if _read(target) != receipt:
            raise ValueError("Existing mixed Track1 batch differs from source set")
    else:
        _write_new(target, receipt)
    return receipt


def verify_track1_mixed52_source_admission(
    receipt: dict[str, Any], *, admission_path: Path, admission_sha256: str,
    frozen_training_request_sha256: str,
) -> dict[str, Any]:
    """Fail closed on exact mixed source/label/float lineage and visible warnings."""
    path = _bound_file(str(admission_path), admission_sha256, "mixed52 admission")
    admission = _read(path)
    rows = receipt.get("revisions") or []
    source_shas = {row.get("source_manifest_sha256") for row in rows}
    groups = {"current43": set(), "historical9": set()}
    identities = receipt.get("processor_identities") or {}
    if "current43_ac44_alt" in identities:
        groups["current43_ac44_alt"] = set()
    if "historical9_ac44_alt" in identities:
        groups["historical9_ac44_alt"] = set()
    if (receipt.get("source_cohort_contract") != COHORT_CONTRACT_MIXED
            or receipt.get("processor_identity") is not None
            or set(identities) != set(groups)
            or len(rows) != 52 or len(source_shas) != 52
            or sum(row.get("frame_count", -999) for row in rows) != 1045):
        raise ValueError("Mixed52 batch source cohort is incomplete")
    for row in rows:
        group = row.get("source_group")
        if (group not in groups or row.get("processor_identity_sha256") !=
                _identity_sha(identities[group])):
            raise ValueError("Mixed52 source revision has another processor")
        per_revision = _read(_bound_file(row.get("receipt_path"),
                                         row.get("receipt_sha256"),
                                         "mixed52 revision receipt"))
        if (per_revision.get("processor") != identities[group]
                or per_revision.get("source_manifest_sha256") !=
                   row.get("source_manifest_sha256")
                or per_revision.get("materialized_manifest_sha256") !=
                   row.get("materialized_manifest_sha256")):
            raise ValueError("Mixed52 revision source/processor changed")
        groups[group].add(row["source_manifest_sha256"])
    gates = ("source_geometry_qa_pass", "temporal_feature_qa_pass",
             "photo_sand_qa_pass", "training_label_subset_qa_pass",
             "float_input_qa_pass")
    current_shas = groups["current43"] | groups.get("current43_ac44_alt", set())
    historical_shas = groups["historical9"] | groups.get("historical9_ac44_alt", set())
    if "current43_ac44_alt" in identities:
        exception = identities["current43_ac44_alt"].get(
            "source_exact_alt_lens_exception") or {}
        if (identities["current43_ac44_alt"].get("processor_variant_contract") !=
                "ffac-v5-track1-ac44-source-exact-alt-lens-processor-v1"
                or identities["current43_ac44_alt"].get("renderer_module_sha256") !=
                AC44_ALT_RENDERER_SHA
                or identities["current43_ac44_alt"].get("processor_module_sha256") !=
                AC44_ALT_PROCESSOR_SHA
                or exception != dict(source_manifest_sha256=AC44_ALT_SOURCE_SHA,
                                     frame_index=18, arw_sha256=AC44_ALT_ARW_SHA,
                                     camera_jpeg_sha256=AC44_ALT_CAMERA_SHA,
                                     native_review_jpeg_sha256=AC44_ALT_NATIVE_SHA,
                                     all_other_frames_use_original_renderer=True)):
            raise ValueError("Mixed52 AC44 lens exception is not exact-source bound")
    if "historical9_ac44_alt" in identities:
        exception = identities["historical9_ac44_alt"].get(
            "source_exact_alt_lens_exception") or {}
        if (identities["historical9_ac44_alt"].get("processor_variant_contract") !=
                "ffac-v5-track1-ac44-historical-exact-alt-lens-processor-v1"
                or identities["historical9_ac44_alt"].get("renderer_module_sha256") !=
                HISTORICAL_AC44_ALT_RENDERER_SHA
                or identities["historical9_ac44_alt"].get("processor_module_sha256") !=
                HISTORICAL_AC44_ALT_PROCESSOR_SHA
                or exception != dict(source_manifest_sha256=HISTORICAL_AC44_ALT_SOURCE_SHA,
                                     frame_index=21, arw_sha256=AC44_ALT_ARW_SHA,
                                     camera_jpeg_sha256=AC44_ALT_CAMERA_SHA,
                                     native_review_jpeg_sha256=HISTORICAL_AC44_ALT_NATIVE_SHA,
                                     all_other_frames_use_original_renderer=True)):
            raise ValueError("Historical AC44 lens exception is not exact-source bound")
    if ((len(groups["current43"]) != (42 if "current43_ac44_alt" in groups else 43))
            or (groups.get("current43_ac44_alt", set()) != {AC44_ALT_SOURCE_SHA}
                if "current43_ac44_alt" in groups else False)
            or (len(groups["historical9"]) !=
                (8 if "historical9_ac44_alt" in groups else 9))
            or (groups.get("historical9_ac44_alt", set()) !=
                {HISTORICAL_AC44_ALT_SOURCE_SHA}
                if "historical9_ac44_alt" in groups else False)
            or len(current_shas) != 43 or len(historical_shas) != 9
            or admission.get("contract") != ADMISSION_CONTRACT
            or admission.get("scope") != "studio-validation-only-light-only-provisional"
            or admission.get("validation_only") is not True
            or admission.get("scientific_mass_eligible") is not False
            or admission.get("mass_eligible") is not False
            or admission.get("chemistry_alignment_applied") is not False
            or admission.get("automatic_publication") is not False
            or admission.get("provisional_studio_training_authorized") is not True
            or admission.get("original_review_ids_preserved") != 4048
            or admission.get("frozen_training_request_sha256") !=
               frozen_training_request_sha256
            or set(admission.get("source_manifest_sha256s") or []) != source_shas
            or set(admission.get("current_source_manifest_sha256s") or []) !=
               current_shas
            or set(admission.get("historical_source_manifest_sha256s") or []) !=
               historical_shas
            or any(admission.get(gate) is not True for gate in gates)):
        raise ValueError("Mixed52 provisional admission does not certify exact sources")
    partition = _read(_bound_file(admission.get("review_partition_path"),
                                  admission.get("review_partition_sha256"),
                                  "mixed52 review fit/evaluation partition"))
    label_rows = partition.get("rows") or []
    if (partition.get("contract") != PARTITION_CONTRACT
            or partition.get("frozen_training_request_sha256") !=
               frozen_training_request_sha256
            or set(partition.get("source_manifest_sha256s") or []) != source_shas
            or len(label_rows) != 4048
            or len({row.get("region_id") for row in label_rows}) != 4048
            or len({row.get("source_manifest_sha256") for row in label_rows}) != 45
            or not {row.get("source_manifest_sha256") for row in label_rows}.issubset(
                source_shas)):
        raise ValueError("Mixed52 review partition lacks exact 4048 source labels")
    fit_count = evaluation_count = historical_count = 0
    for row in label_rows:
        role = row.get("role")
        run = str(row.get("run") or "").casefold()
        if row.get("source_manifest_sha256") in historical_shas:
            historical_count += 1
        if role == "fit":
            if run in {"ac20", "ac21", "ac22", "ac24", "ac27", "ac51"}:
                raise ValueError("Chemistry-shifted review entered mixed52 source fit")
            fit_count += 1
        elif role == "evaluation-only" and isinstance(row.get("reason"), str) and row["reason"]:
            evaluation_count += 1
        else:
            raise ValueError("Mixed52 review has no explicit fit/evaluation role")
    if (historical_count != 1719 or fit_count == 0
            or fit_count + evaluation_count != 4048
            or partition.get("fit_review_ids") != fit_count
            or partition.get("evaluation_only_review_ids") != evaluation_count
            or admission.get("fit_review_ids") != fit_count
            or admission.get("evaluation_only_review_ids") != evaluation_count):
        raise ValueError("Mixed52 source label partition counts disagree")
    support = _read(_bound_file(admission.get("review_support_audit_path"),
                                admission.get("review_support_audit_sha256"),
                                "mixed52 native review support audit"))
    if (support.get("contract") != "ffac-v5-track1-mixed52-native-label-support-v1"
            or support.get("review_ids") != 4048
            or support.get("historical_review_ids") != 1719
            or support.get("fit_review_ids") != fit_count
            or not 0 < int(support.get("fit_hard_gas_positive_ids") or 0) <= 279
            or admission.get("fit_hard_gas_positive_ids") !=
               support.get("fit_hard_gas_positive_ids")
            or support.get("fit_region_support_pass") is not True
            or support.get("frozen_training_request_sha256") !=
               frozen_training_request_sha256
            or len(support.get("rows") or []) != 4048):
        raise ValueError("Mixed52 fit subset lacks native float/photo/temporal support")
    support_by_id = {row.get("region_id"): row for row in support["rows"]}
    if len(support_by_id) != 4048:
        raise ValueError("Mixed52 native support has duplicate review IDs")
    for row in label_rows:
        measured = support_by_id.get(row["region_id"])
        if (measured is None or measured.get("role") != row["role"]
                or measured.get("source_manifest_sha256") !=
                   row["source_manifest_sha256"]
                or measured.get("frame_index") != row.get("frame_index")
                or measured.get("reason") != row.get("reason")):
            raise ValueError("Mixed52 fit role was not measured on its exact source")
        if (row["role"] == "fit" and
                (float(measured.get("photo_safe_fraction") or 0) < .90
                 or float(measured.get(
                    "feature_retention_of_photo_safe_fraction") or 0) < .90)):
            raise ValueError("Mixed52 fit review has inadequate native source support")
    photo = _read(_bound_file(admission.get("photo_transport_aggregate_path"),
                              admission.get("photo_transport_aggregate_sha256"),
                              "mixed52 photo transport aggregate"))
    if (len(photo.get("rows") or []) != 52
            or {row.get("source_manifest_sha256") for row in photo["rows"]} != source_shas):
        raise ValueError("Mixed52 photo transport lacks exact revisions")
    if admission.get("ac51_f18_temporal_primary_gate_pass") is not False:
        raise ValueError("Mixed52 hid the AC51 frame18 temporal exception")
    _bound_file(admission.get("ac51_f18_explicit_v3_exception_receipt_path"),
                admission.get("ac51_f18_explicit_v3_exception_receipt_sha256"),
                "AC51 frame18 source-only geometry exception")
    historical_fit = _read(_bound_file(admission.get("historical_fit_manifest_path"),
                                       admission.get("historical_fit_manifest_sha256"),
                                       "historical source-specific B fits"))
    if (len(historical_fit.get("results") or []) != 9
            or {row.get("source_manifest_sha256") for row in historical_fit["results"]}
               != historical_shas):
        raise ValueError("Mixed52 history used wrong B fits")
    if (identities["current43"].get("calibration_manifest_sha256") !=
            admission.get("photometry_fit_manifest_sha256")
            or ("current43_ac44_alt" in identities and
                identities["current43_ac44_alt"].get("calibration_manifest_sha256") !=
                admission.get("photometry_fit_manifest_sha256"))
            or identities["historical9"].get("canonical_fit_manifest_sha256") !=
               admission.get("photometry_fit_manifest_sha256")
            or identities["historical9"].get("historical_fit_manifest_sha256") !=
               admission.get("historical_fit_manifest_sha256")
            or ("historical9_ac44_alt" in identities and
                identities["historical9_ac44_alt"].get("historical_fit_manifest_sha256") !=
                admission.get("historical_fit_manifest_sha256"))
            or any(identity.get("photo_source_aggregate_sha256") !=
                   admission.get("photo_transport_aggregate_sha256")
                   for identity in identities.values())):
        raise ValueError("Mixed52 processor math differs from admitted source fits")
    float_receipt = _read(_bound_file(admission.get("float_range_preflight_path"),
                                      admission.get("float_range_preflight_sha256"),
                                      "mixed52 float range receipt"))
    if (float_receipt.get("contract") != FLOAT_CONTRACT
            or set(float_receipt.get("source_manifest_sha256s") or []) != source_shas
            or float_receipt.get("unit_denominator") != 65535
            or float_receipt.get("input_dtype") != "float32"
            or float_receipt.get("display_jpeg_is_training_input") is not False
            or float_receipt.get("cpu_only_float_pipeline_pass") is not True
            or float_receipt.get("uint8_model_input_rejected") is not True
            or float_receipt.get("full_timeline_per_frame_fail_fast") is not True
            or not 0 < float(float_receipt.get("max_measured_corrected_linear") or 0)
                   <= 65535):
        raise ValueError("Mixed52 float input clips or lacks CPU float proof")
    cpu_proof = _read(_bound_file(float_receipt.get("cpu_float_proof_path"),
                                  float_receipt.get("cpu_float_proof_sha256"),
                                  "mixed52 CPU float/uint8 proof"))
    if (cpu_proof.get("cpu_only_float_pipeline_pass") is not True
            or cpu_proof.get("uint8_model_input_rejected") is not True
            or cpu_proof.get("display_jpeg_is_training_input") is not False):
        raise ValueError("Mixed52 CPU float/uint8 source proof changed")
    fit = _read(_bound_file(admission.get("photometry_fit_manifest_path"),
                            admission.get("photometry_fit_manifest_sha256"),
                            "mixed52 canonical photometry fit"))
    decision = _read(_bound_file(admission.get("photometry_heldout_qa_path"),
                                 admission.get("photometry_heldout_qa_sha256"),
                                 "mixed52 provisional light decision"))
    if (decision.get("contract") != DECISION_CONTRACT
            or decision.get("photometry_fit_manifest_sha256") !=
               admission.get("photometry_fit_manifest_sha256")
            or decision.get("whole_run_B_H_pass") is not True
            or decision.get("donor_tile_cv_status") != "failed-warning"
            or decision.get("late_yellow_status") != "failed-warning"
            or decision.get("weak_phase_status") != "pending-diagnostic"
            or not fit.get("results")):
        raise ValueError("Mixed52 provisional photometry warnings changed")
    return admission


__all__ = ["aggregate_track1_mixed52_materialization",
           "verify_track1_mixed52_source_admission"]
