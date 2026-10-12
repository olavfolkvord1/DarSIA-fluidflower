"""Prepare and fit an isolated FF_AC v5 RAW-photometry Studio candidate.

The source materializer owns pixel correction. This module only rebinds an
already frozen set of labels to SHA-verified full-native corrected images,
then uses the existing optical/regional trainer. It never changes live reviews
or a published model. Track1 uses a source-bound photo/float admission receipt;
physical candidates require a separate native-boundary certificate.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_training import trainer as training
from darsia.fluidflower.phase_training.v5_chemistry_fit_scope import (
    EXCLUDED_RUNS as CHEMISTRY_UNALIGNED_RUNS,
    scope_for_regions,
    verify_scope,
)
from darsia.fluidflower.phase_training.v5_linear_input import V5_UNIT_DENOMINATOR
from darsia.fluidflower.phase_workflow import prepare_phase_training_request
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools import train_ffac_two_arm_optical_region as previous
from darsia.fluidflower.tools import train_studio_optical_then_region_series as series
from darsia.fluidflower.tools import train_studio_series_update as regional
from darsia.fluidflower.tools import train_studio_region_update as update
from darsia.fluidflower.tools import train_region_coverage_overnight as regional_training
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests
from darsia.fluidflower.tools.ffac_v5_geometry_policy import verify_accepted_geometry
from darsia.fluidflower.tools.prepare_grain_gas_retraining import physics_files
from darsia.fluidflower.tools.publish_fitted_phase_candidate import (
    destination_volume_staging_root, publish_fitted, verify_fitted,
)


FORMAT = "ffac-v5-raw-full-native-training-v1"
BATCH_FORMAT = "ffac-v5-native-materialization-batch-v1"
OPTICAL_CHECKPOINT_FORMAT = "ffac-v5-raw-immutable-optical-checkpoint-v1"
TRACK1_PROCESSOR_CONTRACT = "ffac-v5-track1-source-bound-native-processor-v1"
TRACK1_CURRENT43_ADMISSION_CONTRACT = (
    "ffac-v5-track1-current43-provisional-source-admission-v1")
TRACK1_MIXED52_COHORT_CONTRACT = (
    "ffac-v5-track1-mixed52-source-materialization-v1")
TRACK1_MIXED52_ADMISSION_CONTRACT = (
    "ffac-v5-track1-mixed52-provisional-source-admission-v1")
TRACK1_AC44_ALT_SOURCE_SHA256 = (
    "fdcd7f7ff6f8718180db82fbb25ffa788b185c51effefb25d9d8b20f2f01898e")
TRACK1_HISTORICAL_AC44_ALT_SOURCE_SHA256 = (
    "e746d4bf8bc9a3d35a22cccf27f8cfd5788a5959e38b702e3269df6d83671a17")
TRACK1_PARTITION_CONTRACT = "ffac-v5-track1-frozen-review-fit-evaluation-partition-v1"
TRACK1_PHOTOMETRY_DECISION_CONTRACT = "ffac-v5-track1-provisional-photometry-decision-v1"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode()).hexdigest()


def _processor_identity_sha(identity: dict) -> str:
    # This uses the historical/current materializer's canonical JSON encoding.
    return sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _exact(path: str | Path) -> Path:
    return Path(path).resolve()


def _track1_scope(receipt: dict) -> str | None:
    """Distinguish the old 43-only review batch from a mixed-processor 52 batch."""
    if receipt.get("source_cohort_contract") == TRACK1_MIXED52_COHORT_CONTRACT:
        _require(receipt.get("processor_identity") is None,
                 "Mixed52 batch cannot claim one homogeneous processor")
        return "all52"
    if (receipt.get("processor_identity") or {}).get("contract") == (
            TRACK1_PROCESSOR_CONTRACT):
        _require(receipt.get("processor_identities") is None,
                 "Current43 batch cannot carry a second processor identity")
        return "current43"
    return None


def _verify_mixed52_processors(receipt: dict, selected: dict[str, dict],
                               canonical_shas: set[str],
                               expected_shas: set[str]) -> None:
    """Bind each revision to its own frozen current or historical processor."""
    identities = receipt.get("processor_identities")
    required_groups = {"current43", "historical9"}
    optional_groups = {"current43_ac44_alt", "historical9_ac44_alt"}
    _require(isinstance(identities, dict)
             and required_groups.issubset(identities)
             and set(identities).issubset(required_groups | optional_groups)
             and all(isinstance(value, dict) and value.get("contract") ==
                     TRACK1_PROCESSOR_CONTRACT for value in identities.values()),
             "Mixed52 needs exact current, historical, and AC44 processor groups")
    identity_shas = {group: _processor_identity_sha(identity)
                     for group, identity in identities.items()}
    _require(len(set(identity_shas.values())) == len(identity_shas),
             "Mixed52 source groups cannot share a processor identity")
    groups = {group: set() for group in identities}
    for digest, row in selected.items():
        group = row.get("source_group")
        _require(group in groups and row.get("processor_identity_sha256") ==
                 identity_shas[group],
                 f"Mixed52 revision has no exact processor binding: {digest}")
        if group == "current43_ac44_alt":
            _require(digest == TRACK1_AC44_ALT_SOURCE_SHA256
                     and str(row.get("run") or "").casefold() == "ac44",
                     "AC44 alternate renderer was assigned to another source")
        if group == "historical9_ac44_alt":
            _require(digest == TRACK1_HISTORICAL_AC44_ALT_SOURCE_SHA256
                     and str(row.get("run") or "").casefold() == "ac44",
                     "Historical AC44 alternate renderer was assigned to another source")
        per_revision = _read(_exact(row["receipt_path"]))
        _require(per_revision.get("processor") == identities[group],
                 f"Mixed52 revision used another frozen processor: {digest}")
        groups[group].add(digest)
    current_alt = groups.get("current43_ac44_alt", set())
    historical_alt = groups.get("historical9_ac44_alt", set())
    historical_shas = expected_shas - canonical_shas
    _require((current_alt == {TRACK1_AC44_ALT_SOURCE_SHA256}
              if "current43_ac44_alt" in identities else not current_alt)
             and (historical_alt == {TRACK1_HISTORICAL_AC44_ALT_SOURCE_SHA256}
                  if "historical9_ac44_alt" in identities else not historical_alt)
             and groups["current43"] == canonical_shas - current_alt
             and groups["historical9"] == historical_shas - historical_alt
             and len(groups["current43"]) + len(current_alt) == 43
             and len(groups["historical9"]) + len(historical_alt) == 9,
             "Mixed52 processor groups do not match the frozen source versions")


def _one_step_two_arm_child(child_manifest_path: Path,
                            parent_manifest_sha256: str) -> bool:
    """Verify a single geometry/time-preserving photometry revision by SHA."""
    child = _read(child_manifest_path)
    marker = child.get("two_arm_photometry") or {}
    parent_path = _exact(marker.get("source_manifest") or "")
    return (marker.get("format") == "ffac-two-arm-native-materialization-v1"
            and marker.get("source_manifest_sha256") == parent_manifest_sha256
            and marker.get("same_raw_geometry_and_timestamps") is True
            and marker.get("no_phase_label_or_map_in_target_selection") is True
            and parent_path.is_file()
            and sha256_file(parent_path) == parent_manifest_sha256)


def verify_materialization(frozen_request_path: Path, receipt_path: Path,
                           boundary_path: Path | None) -> tuple[dict, dict[str, dict]]:
    """Verify complete source set and independent native-label geometry gate."""
    frozen_request_path = _exact(frozen_request_path)
    receipt_path = _exact(receipt_path)
    receipt = _read(receipt_path)
    track1_scope = _track1_scope(receipt)
    track1 = track1_scope is not None
    mixed52 = track1_scope == "all52"
    if not track1:
        _require(boundary_path is not None, "Physical V5 batch requires native boundary QA")
        boundary_path = _exact(boundary_path)
    _require(receipt.get("format") == BATCH_FORMAT
             and _exact(receipt.get("frozen_training_request_path", "")) == frozen_request_path
             and receipt.get("frozen_training_request_sha256") == sha256_file(frozen_request_path),
             "V5 batch is not bound to the frozen reviewer request")
    expected = source_manifests(frozen_request_path)
    rows = receipt.get("revisions") or []
    expected_by_sha = {row["manifest_sha256"]: row for row in expected}
    _require(len(expected_by_sha) == 52 and receipt.get("canonical_runs") == 43,
             "Frozen V5 request does not describe its complete source inventory")
    expected_revisions = 43 if track1_scope == "current43" else 52
    expected_frames = 864 if track1_scope == "current43" else 1045
    _require(len(rows) == receipt.get("source_revisions") == expected_revisions
             and receipt.get("timeline_frames") == expected_frames,
             "V5 batch does not cover its declared native source scope")
    if track1:
        _require(receipt.get("scientific_mass_eligible") is False and boundary_path is None,
                 "Track1 optical batch cannot claim physical acceptance")
    else:
        _require(receipt.get("accepted_run_native_geometry_revisions") == 52,
                 "V5 batch still uses the shared physical/facies grid")
    by_sha = {row["source_manifest_sha256"]: dict(row) for row in rows}
    canonical = receipt.get("canonical_mapping") or []
    canonical_shas = {row["frozen_source_manifest_sha256"] for row in canonical}
    _require(len(canonical) == 43
             and len({str(row["run"]).casefold() for row in canonical}) == 43
             and len(canonical_shas) == 43
             and canonical_shas.issubset(by_sha)
             and canonical_shas.issubset(expected_by_sha)
             and all(str(row["run"]).casefold() ==
                     str(expected_by_sha[row["frozen_source_manifest_sha256"]]["run"]).casefold()
                     for row in canonical),
             "V5 canonical maps are incomplete or ambiguous")
    _require(len(by_sha) == expected_revisions
             and set(by_sha) == (canonical_shas if track1_scope == "current43"
                                    else set(expected_by_sha))
             and set(by_sha).issubset(expected_by_sha),
             "V5 batch omits or duplicates a source revision")
    frozen_request = _read(frozen_request_path)
    parent_jobs = _read(_exact(frozen_request["output_root"]) /
                        "review_source" / "postflight_manifest.json")["jobs"]
    parent_by_run = {str(job["run"]).casefold(): optical.native_source(job)[1]
                     for job in parent_jobs}
    _require(len(parent_jobs) == len(parent_by_run) == 43,
             "V5 canonical set differs from the frozen Studio timeline")
    for row in canonical:
        run = str(row["run"]).casefold()
        parent_path = parent_by_run.get(run)
        _require(parent_path is not None,
                 "V5 canonical run is absent from the frozen Studio timeline")
        parent_sha = sha256_file(parent_path)
        original_sha = row["canonical_source_manifest_sha256"]
        frozen_sha = row["frozen_source_manifest_sha256"]
        if original_sha != frozen_sha:
            _require(parent_sha in {original_sha, frozen_sha}
                     and _one_step_two_arm_child(
                         _exact(expected_by_sha[frozen_sha]["manifest_path"]),
                         original_sha),
                     "Canonical Studio source lacks its exact one-step frozen revision")
        else:
            _require(parent_sha == frozen_sha or
                     _one_step_two_arm_child(parent_path, frozen_sha),
                     "Canonical Studio source lacks its exact one-step parent revision")
    for digest in sorted(by_sha):
        source = expected_by_sha[digest]
        row = by_sha[source["manifest_sha256"]]
        original = _exact(source["manifest_path"])
        materialized = _exact(row["materialized_manifest_path"])
        per_revision = _exact(row["receipt_path"])
        _require(_exact(row["source_manifest_path"]) == original
                 and str(row["run"]).casefold() == source["run"]
                 and sha256_file(original) == source["manifest_sha256"]
                 and sha256_file(materialized) == row["materialized_manifest_sha256"]
                 and sha256_file(per_revision) == row["receipt_sha256"]
                 and _exact(row["materialized_review_root"])
                     == materialized.parents[2],
                 f"V5 source or derived manifest changed: {source['run']}")
        detail = _read(per_revision)
        if track1:
            from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
                COHORT_CONTRACT, Track1FramePhotoCohort,
            )
            cohort = Track1FramePhotoCohort(materialized)
            _require(cohort.source_manifest_sha256 == source["manifest_sha256"]
                     and cohort.native_manifest_sha256 == row["materialized_manifest_sha256"]
                     and len(cohort.rows) == row["frame_count"]
                     and row.get("track1_photo_geometry_contract") == COHORT_CONTRACT
                     and row.get("track1_photo_geometry_receipt_count") == len(cohort.rows)
                     and detail.get("source_manifest_sha256") == source["manifest_sha256"]
                     and detail.get("materialized_manifest_sha256") == row["materialized_manifest_sha256"]
                     and detail.get("frame_count") == row["frame_count"],
                     f"Track1 photo/source revision differs: {source['run']}")
        else:
            physical_geometry = verify_accepted_geometry(original, materialized)
            _require(detail.get("source_manifest_sha256") == source["manifest_sha256"]
                     and detail.get("materialized_manifest_sha256") == row["materialized_manifest_sha256"]
                     and detail.get("frame_count") == row["frame_count"]
                     and row.get("run_geometry_receipt_sha256")
                         == physical_geometry["receipt_sha256"]
                     and detail.get("run_geometry_receipt_sha256")
                         == physical_geometry["receipt_sha256"],
                     f"V5 per-revision receipt differs: {source['run']}")
        source_baseline = (_exact(source["review_root"]) / "native_references"
                           / source["run"] / "pre_injection.jpg")
        _require(source_baseline.is_file(),
                 f"Original pre-injection reference is missing: {source_baseline}")
        row["_original_baseline_sha256"] = sha256_file(source_baseline)
    if track1:
        admission_path = _exact(receipt.get("source_admission_receipt_path") or "")
        admission_sha = receipt.get("source_admission_receipt_sha256")
        _require(admission_path.is_file() and
                 sha256_file(admission_path) == admission_sha,
                 "Track1 source admission is not SHA-bound at batch level")
        if mixed52:
            from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
                verify_track1_mixed52_source_admission,
            )
            _verify_mixed52_processors(receipt, by_sha, canonical_shas,
                                       set(expected_by_sha))
            verify_track1_mixed52_source_admission(
                receipt, admission_path=admission_path,
                admission_sha256=admission_sha,
                frozen_training_request_sha256=sha256_file(frozen_request_path))
        else:
            from darsia.fluidflower.tools.materialize_ffac_v5_sources import (
                verify_v4r3_full_source_admission,
            )
            verify_v4r3_full_source_admission(
                receipt["processor_identity"],
                admission_path=admission_path,
                admission_sha256=admission_sha,
                all_source_manifest_sha256s=set(by_sha),
                source_manifest_runs={digest: row["run"] for digest, row in by_sha.items()},
                training_request_sha256=sha256_file(frozen_request_path),
            )
        admission = _read(admission_path)
        required_gates = ("source_geometry_qa_pass", "temporal_feature_qa_pass",
                          "photo_sand_qa_pass", "training_label_subset_qa_pass",
                          "float_input_qa_pass")
        _require(admission.get("contract") == (TRACK1_MIXED52_ADMISSION_CONTRACT
                                               if mixed52 else
                                               TRACK1_CURRENT43_ADMISSION_CONTRACT)
                 and admission.get("scope") == "studio-validation-only-light-only-provisional"
                 and all(admission.get(gate) is True for gate in required_gates)
                 and admission.get("validation_only") is True
                 and admission.get("scientific_mass_eligible") is False
                 and admission.get("mass_eligible") is False
                 and admission.get("provisional_studio_training_authorized") is True
                 and admission.get("automatic_publication") is False
                 and admission.get("original_review_ids_preserved") == 4048
                 and set(admission.get("source_manifest_sha256s") or []) == set(by_sha)
                 and len(admission.get("historical_source_manifest_sha256s") or []) == 9
                 and set(admission["historical_source_manifest_sha256s"]) ==
                     set(expected_by_sha) - canonical_shas,
                 "Track1 full-source/photo/label/float gates have not passed")
        partition_path = _exact(admission.get("review_partition_path") or "")
        _require(partition_path.is_file() and
                 sha256_file(partition_path) == admission.get("review_partition_sha256"),
                 "Track1 frozen fit/evaluation partition changed")
        partition = _read(partition_path)
        partition_rows = partition.get("rows") or []
        _require(partition.get("contract") == TRACK1_PARTITION_CONTRACT
                 and len(partition_rows) == 4048
                 and len({row.get("region_id") for row in partition_rows}) == 4048
                 and {row.get("source_manifest_sha256") for row in partition_rows}.issubset(expected_by_sha)
                 and all(row.get("role") in ("fit", "evaluation-only") and
                         (row.get("role") == "fit" or row.get("reason"))
                         for row in partition_rows),
                 "Track1 partition is incomplete or has ambiguous source roles")
        if not mixed52:
            _require(all(row["role"] == "evaluation-only" and
                         row.get("reason") ==
                         "historical-source-not-materialized-current43-provisional"
                         for row in partition_rows
                         if row["source_manifest_sha256"] not in by_sha),
                     "An unmaterialized historical review entered candidate fitting")
        for stem in ("float_range_preflight", "photometry_fit_manifest",
                     "photometry_heldout_qa"):
            path = _exact(admission.get(f"{stem}_path") or "")
            _require(path.is_file() and sha256_file(path) == admission.get(f"{stem}_sha256"),
                     f"Track1 admission lacks certified {stem}")
        warning = _read(_exact(admission["photometry_heldout_qa_path"]))
        _require(warning.get("contract") == TRACK1_PHOTOMETRY_DECISION_CONTRACT
                 and warning.get("whole_run_B_H_pass") is True
                 and warning.get("donor_tile_cv_status") == "failed-warning"
                 and warning.get("late_yellow_status") == "failed-warning"
                 and warning.get("weak_phase_status") == "pending-diagnostic",
                 "Track1 provisional light decision or explicit warnings changed")
    else:
        boundary = _read(boundary_path)
        _require(boundary.get("format") == "ffac-v5-native-annotation-boundary-qa-v1"
                 and boundary.get("passed") is True
                 and boundary.get("materialization_receipt_sha256") == sha256_file(receipt_path)
                 and boundary.get("frozen_training_request_sha256") == sha256_file(frozen_request_path)
                 and boundary.get("checked_annotation_ids") == 4048
                 and boundary.get("native_label_geometry_unchanged") is True
                 and boundary.get("run_native_physical_context_source_bound") is True
                 and boundary.get("prior_shared_facies_context_preserved") is False
                 and boundary.get("eligible_annotation_pixels_outside_corrected_domain") == 0,
                 "Full-native annotation-boundary QA has not certified this V5 batch")
    return receipt, by_sha


def _verified_source_alias(job: dict, selected: dict[str, dict],
                           checked_aliases: set[tuple[str, str]],
                           canonical_aliases: dict[str, dict] | None = None) -> dict:
    source_root, manifest_path, manifest = optical.native_source(job)
    digest = sha256_file(manifest_path)
    row = selected.get(digest)
    if row is None and canonical_aliases is not None:
        row = canonical_aliases.get(digest)
        if row is not None:
            frozen_path = _exact(row["source_manifest_path"])
            _require((_one_step_two_arm_child(frozen_path, digest) or
                      _one_step_two_arm_child(manifest_path,
                                              row["source_manifest_sha256"]))
                and str(job["run"]).casefold() == str(row["run"]).casefold(),
                "Canonical source has no exact one-step frozen photometry revision")
            return row
    _require(row is not None and str(job["run"]).casefold() == str(row["run"]).casefold(),
             f"No corrected V5 source for {job['run']} {digest}")
    alias = (str(_exact(source_root)).casefold(), digest)
    if alias in checked_aliases:
        return row
    # Historical copies can have another path but must have identical native
    # images and baseline. The manifest SHA alone does not prove pixel bytes.
    run = str(job["run"]).casefold()
    for frame in manifest["frame_sources"]:
        index = int(frame["frame_index"])
        path = Path(source_root) / "native_frames" / run / f"frame_{index:03d}.jpg"
        expected = frame.get("native_jpeg_sha256") or frame.get("frame_sha256")
        _require(path.is_file() and (not expected or sha256_file(path) == expected),
                 f"Historical source JPEG changed: {path}")
    baseline = Path(source_root) / "native_references" / run / "pre_injection.jpg"
    _require(baseline.is_file() and sha256_file(baseline)
             == row["_original_baseline_sha256"],
             f"Historical baseline differs from selected source: {baseline}")
    checked_aliases.add(alias)
    return row


def _bound_job(job: dict, selected: dict[str, dict], receipt_path: Path,
               checked_aliases: set[tuple[str, str]],
               canonical_aliases: dict[str, dict] | None = None) -> dict:
    if str(job.get("dataset") or "").casefold() != "ffac":
        return deepcopy(job)
    row = _verified_source_alias(job, selected, checked_aliases,
                                 canonical_aliases)
    result = deepcopy(job)
    result["review_output_dir"] = row["materialized_review_root"]
    result["analysis_source_review_root"] = row["materialized_review_root"]
    result["v5_raw_photometry"] = {
        "format": FORMAT,
        "batch_receipt_path": str(_exact(receipt_path)),
        "batch_receipt_sha256": sha256_file(receipt_path),
        "source_manifest_sha256": row["source_manifest_sha256"],
        "materialized_manifest_sha256": row["materialized_manifest_sha256"],
        "full_native_resolution": True,
        "scientific_input": "sha-bound-float32-log-camera-linear-sidecar",
        "unit_denominator": V5_UNIT_DENOMINATOR,
        "display_jpeg_is_training_input": False,
    }
    root, path, _ = optical.native_source(result)
    _require(_exact(root) == _exact(row["materialized_review_root"])
             and sha256_file(path) == row["materialized_manifest_sha256"],
             f"Trainer would read another V5 source: {job['run']}")
    return result


def _canonical_aliases(receipt: dict, selected: dict[str, dict],
                       canonical_jobs: list[dict]) -> dict[str, dict]:
    aliases: dict[str, dict] = {}
    jobs_by_run = {str(job["run"]).casefold(): job for job in canonical_jobs}
    _require(len(jobs_by_run) == len(canonical_jobs) == 43,
             "Canonical Studio jobs are incomplete or duplicated")
    for entry in receipt["canonical_mapping"]:
        run = str(entry["run"]).casefold()
        parent = entry["canonical_source_manifest_sha256"]
        frozen = entry["frozen_source_manifest_sha256"]
        observed_path = optical.native_source(jobs_by_run[run])[1]
        observed = sha256_file(observed_path)
        if observed == frozen:
            continue
        row = selected[frozen]
        _require(observed not in aliases and
                 str(row["run"]).casefold() == run and
                 ((_one_step_two_arm_child(_exact(row["source_manifest_path"]),
                                           observed) and observed == parent) or
                  (_one_step_two_arm_child(observed_path, frozen)
                   and parent == frozen)),
                 "Canonical parent alias is ambiguous")
        aliases[observed] = row
    return aliases


def _current43_bundles(old_pairs: list[training.ReviewBundle],
                       selected: dict[str, dict], receipt_path: Path,
                       destination: Path,
                       checked_aliases: set[tuple[str, str]]) -> tuple[
                           list[tuple[Path, Path]], dict[int, int]]:
    """Make a derived training view without opening unmaterialized old images.

    The immutable parent decisions remain untouched. The separate 4048-ID
    partition records every omitted historical review as unscorable here.
    """
    selected_shas = set(selected)
    pairs: list[tuple[Path, Path]] = []
    old_to_new: dict[int, int] = {}
    for index, pair in enumerate(old_pairs):
        combined = _read(pair.combined_manifest_path)
        decisions = _read(pair.decisions_path)
        candidate_jobs: dict[str, dict] = {}
        for job in combined.get("jobs") or []:
            if str(job.get("dataset") or "").casefold() != "ffac":
                continue
            run = str(job["run"]).casefold()
            digest = sha256_file(optical.native_source(job)[1])
            if digest not in selected_shas:
                continue
            rebound = _bound_job(job, selected, receipt_path, checked_aliases)
            previous_job = candidate_jobs.setdefault(run, rebound)
            _require(previous_job == rebound,
                     f"Current43 bundle has ambiguous sources for {run}")
        kept_decisions = {
            run: value for run, value in (decisions.get("decisions") or {}).items()
            if str(run).casefold() in candidate_jobs
        }
        if not kept_decisions:
            continue
        kept_jobs = [candidate_jobs[run] for run in sorted({
            str(run).casefold() for run in kept_decisions
        })]
        decisions_path = destination / f"{index:03d}_decisions.json"
        manifest_path = destination / f"{index:03d}_combined_manifest.json"
        _write_new(decisions_path, dict(decisions, decisions=kept_decisions))
        _write_new(manifest_path, dict(combined, jobs=kept_jobs))
        old_to_new[index] = len(pairs)
        pairs.append((decisions_path, manifest_path))
    _require(pairs, "Current43 training view has no reviewed source")
    return pairs, old_to_new


def _copy_physics(source_root: Path, destination: Path,
                  canonical_jobs: list[tuple[dict, dict]],
                  receipt_path: Path) -> str:
    if canonical_jobs and all(
            _read(optical.native_source(new)[1]).get("v5_track1_photo_geometry")
            for _old, new in canonical_jobs):
        return _copy_track1_photo_context(source_root, destination,
                                          canonical_jobs, receipt_path)
    profile = phase_runtime.load_physics_profile(source_root)
    _require(profile is not None, "Frozen request has no physical profile")
    for source in physics_files(source_root, profile):
        target = destination / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        _require(sha256_file(source) == sha256_file(target),
                 "Copied physical context changed")
    profile_path = destination / "phase_optical_regions.json"
    updated = _read(profile_path)
    for old, new in canonical_jobs:
        key = f"ffac/{str(old['run']).casefold()}"
        entry = updated["runs"].get(key)
        old_manifest_path = optical.native_source(old)[1]
        new_manifest_path = optical.native_source(new)[1]
        before = sha256_file(old_manifest_path)
        after = sha256_file(new_manifest_path)
        _require(entry is not None and entry["native_manifest_sha256"] == before,
                 f"Physical profile cannot bind new source for {key}")
        geometry = verify_accepted_geometry(old_manifest_path, new_manifest_path)
        _require(geometry["source_manifest_sha256"] == before
                 and geometry["materialized_manifest_sha256"] == after,
                 f"Run-native physical context does not bind this source: {key}")
        from darsia.fluidflower.tools.ffac_v5_frame_context import (
            COHORT_CONTRACT, V5FrameGeometryCohort,
        )
        frame_geometry = None
        native_document = _read(new_manifest_path)
        if native_document.get("v5_frame_geometry"):
            frame_geometry = V5FrameGeometryCohort(new_manifest_path)
            _require(frame_geometry.source_manifest_sha256 == before
                     and frame_geometry.native_manifest_sha256 == after
                     and len(frame_geometry.rows) == int(native_document["frame_count"]),
                     f"V5 per-frame geometry does not bind this source: {key}")
            for index in range(len(frame_geometry.rows)):
                frame_geometry.frame(index)
        source_grid = Path(geometry["context_npz_path"])
        target_grid = destination / "registered_context_v5" / f"{str(old['run']).casefold()}.npz"
        target_grid.parent.mkdir(parents=True, exist_ok=True)
        if target_grid.exists():
            _require(sha256_file(target_grid) == geometry["context_npz_sha256"],
                     f"Prepared V5 physical context changed: {target_grid}")
        else:
            shutil.copy2(source_grid, target_grid)
            _require(sha256_file(target_grid) == geometry["context_npz_sha256"],
                     f"Copied V5 physical context changed: {target_grid}")
        original_grid = dict(path=entry["path"], sha256=entry["sha256"])
        entry["path"] = str(target_grid.relative_to(destination)).replace("\\", "/")
        entry["sha256"] = geometry["context_npz_sha256"]
        entry["native_manifest_sha256"] = after
        entry.setdefault("provenance", {})["v5_raw_photometry"] = {
            "format": FORMAT,
            "receipt_path": str(_exact(receipt_path)),
            "receipt_sha256": sha256_file(receipt_path),
            "source_manifest_sha256": before,
            "materialized_manifest_sha256": after,
            "physical_grid_bytes_preserved": False,
        }
        entry["provenance"]["v5_run_geometry"] = {
            "contract": geometry["contract"],
            "receipt_path": geometry["receipt_path"],
            "receipt_sha256": geometry["receipt_sha256"],
            "validation_only": bool(geometry.get("validation_only", False)),
            "mass_eligible": False,
            "physical_scale_qa_pass": False,
            "bottom_gas_guard_allowed": False,
            "physical_scale_warning": (
                "Native 2D scale and injection coordinates remain provisional; "
                "bottom-gas distance rules and mass interpretation are disabled."),
            "original_context": original_grid,
            "exact_2d_coordinate_fields": ["x_m_native", "y_m_native", "pixel_area_m2"],
            "legacy_1d_axes_nominal_approximation_only": True,
            "scientific_and_mass_acceptance": False,
        }
        if frame_geometry is not None:
            entry["provenance"]["v5_frame_geometry"] = {
                "contract": COHORT_CONTRACT,
                "native_manifest_sha256": after,
                "original_source_manifest_sha256": before,
                "frame_receipt_count": len(frame_geometry.rows),
                "validation_only": True,
                "mass_eligible": False,
            }
    _write_new(profile_path.with_suffix(".v5.json"), updated)
    profile_path.with_suffix(".v5.json").replace(profile_path)
    runtime = phase_runtime.physics_runtime_identity(
        phase_runtime.load_physics_profile(destination))
    inherited = _read(source_root / "physics_inheritance.json")
    inherited["original_physics_runtime"] = inherited.get("physics_runtime")
    inherited["physics_runtime"] = runtime
    inherited["v5_raw_photometry"] = {
        "format": FORMAT, "receipt_path": str(_exact(receipt_path)),
        "receipt_sha256": sha256_file(receipt_path), "canonical_runs": len(canonical_jobs),
        "physical_grid_bytes_preserved": False,
        "per_run_2d_physical_context_copied": True,
    }
    for copy in inherited.get("copies") or []:
        old_target = _exact(copy["target"])
        if old_target.is_relative_to(source_root):
            new_target = destination / old_target.relative_to(source_root)
            copy["target"] = str(new_target)
            if sha256_file(new_target) != copy["sha256"]:
                copy["original_copied_sha256"] = copy["sha256"]
                copy["sha256"] = sha256_file(new_target)
                copy["v5_rebound_after_copy"] = True
    _write_new(destination / "physics_inheritance.json", inherited)
    return runtime


def _copy_track1_photo_context(source_root: Path, destination: Path,
                               canonical_jobs: list[tuple[dict, dict]],
                               receipt_path: Path) -> str:
    """Bind an optical-only Studio context, with no inherited facies/rig map.

    The regional runtime needs a domain, nominal pixel spacing and a structural
    injection timeline.  The per-frame photo cohort supplies the real support
    at inference.  No position-dependent rule or mass consumer is enabled.
    """
    from darsia.fluidflower.tools.ffac_v5_track1_photo_context import (
        COHORT_CONTRACT, Track1FramePhotoCohort,
    )

    inherited_profile = phase_runtime.load_physics_profile(source_root)
    _require(inherited_profile is not None and
             inherited_profile.get("format") == "baseline-relative-optical-regions-v1",
             "Track1 needs the existing optical-region model as an immutable parent")
    learned = inherited_profile["model"]
    learned_source = (source_root / learned["path"]).resolve()
    _require(learned_source.is_relative_to(source_root) and
             sha256_file(learned_source) == learned["sha256"],
             "Inherited optical-region artifact changed")
    learned_target = destination / learned["path"]
    learned_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(learned_source, learned_target)
    _require(sha256_file(learned_target) == learned["sha256"],
             "Copied optical-region artifact changed")

    rows: dict[str, dict] = {}
    for old, new in canonical_jobs:
        run = str(old["run"]).casefold()
        key = f"ffac/{run}"
        old_entry = inherited_profile["runs"].get(key)
        _require(old_entry is not None, f"Missing nominal axes for {run}")
        native_manifest = optical.native_source(new)[1]
        cohort = Track1FramePhotoCohort(native_manifest)
        _require(cohort.source_manifest_sha256 ==
                     new["v5_raw_photometry"]["source_manifest_sha256"]
                 and len(cohort.rows) == _read(native_manifest)["frame_count"],
                 f"Track1 photo cohort belongs to another source revision: {run}")
        # The static context intentionally contains no 8-facies/permeability or
        # exact physical coordinate fields. It is replaced by photo-safe support
        # separately for *each frame* before regional inference.
        old_grid_path = (source_root / old_entry["path"]).resolve()
        _require(old_grid_path.is_relative_to(source_root) and
                 sha256_file(old_grid_path) == old_entry["sha256"],
                 f"Nominal parent axes changed: {run}")
        with np.load(old_grid_path, allow_pickle=False) as grid:
            x_m = np.asarray(grid["x_m"], np.float64)
            y_m = np.asarray(grid["y_m"], np.float64)
        shape = tuple(cohort.shape_hw)
        _require(x_m.shape == (shape[1],) and y_m.shape == (shape[0],),
                 f"Nominal axes do not match native review grid: {run}")
        target = destination / "track1_optical_context" / f"{run}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez_compressed(stream,
                domain=np.ones(shape, dtype=bool),
                facies=np.ones(shape, dtype=np.uint8),
                x_m=x_m, y_m=y_m)
        rows[key] = {
            "path": target.relative_to(destination).as_posix(),
            "sha256": sha256_file(target),
            "native_manifest_sha256": cohort.native_manifest_sha256,
            "injection_stages": old_entry["injection_stages"],
            "permeability_available": False,
            "provenance": {
                "v5_raw_photometry": {
                    "format": FORMAT,
                    "receipt_path": str(_exact(receipt_path)),
                    "receipt_sha256": sha256_file(receipt_path),
                    "source_manifest_sha256": cohort.source_manifest_sha256,
                    "materialized_manifest_sha256": cohort.native_manifest_sha256,
                    "physical_grid_inherited": False,
                },
                "v5_track1_photo_geometry": {
                    "contract": COHORT_CONTRACT,
                    "native_manifest_sha256": cohort.native_manifest_sha256,
                    "original_source_manifest_sha256": cohort.source_manifest_sha256,
                    "frame_receipt_count": len(cohort.rows),
                    "validation_only": True,
                    "mass_eligible": False,
                    "facies_ids_used": False,
                    "nominal_axes_only": True,
                },
            },
        }
    _require(len(rows) == len(canonical_jobs) == 43,
             "Track1 nominal context lacks a canonical AC run")
    profile = {
        "format": "baseline-relative-optical-regions-v1",
        "model": learned,
        "runs": rows,
        "v5_track1_review_scope": {
            "contract": "ffac-v5-track1-photo-only-review-context-v1",
            "facies_ids_used": False,
            "physical_scale_qa_pass": False,
            "absolute_position_rules_enabled": False,
            "mass_analysis_allowed": False,
        },
    }
    _write_new(destination / "phase_optical_regions.json", profile)
    runtime = phase_runtime.physics_runtime_identity(
        phase_runtime.load_physics_profile(destination))
    _write_new(destination / "physics_inheritance.json", {
        "parent_model": str(source_root),
        "original_physics_runtime": phase_runtime.physics_runtime_identity(inherited_profile),
        "physics_runtime": runtime,
        "track1_photo_only": True,
        "old_physical_contexts_copied": False,
        "facies_ids_used": False,
        "validation_only": True,
        "mass_eligible": False,
    })
    return runtime


def prepare(frozen_request_path: Path, receipt_path: Path, boundary_path: Path | None,
            cache: Path, *, review_sampling_spec_path: Path,
            actor: str, backend: str = "auto",
            extraction_workers: int = 16) -> dict:
    """Create a fresh request, preserving all source-bound label coordinates."""
    frozen_request_path = _exact(frozen_request_path)
    receipt_path = _exact(receipt_path)
    review_sampling_spec_path = _exact(review_sampling_spec_path)
    review_sampling_spec = _read(review_sampling_spec_path)
    from darsia.fluidflower.phase_training import v5_review_sampling
    _require(
        review_sampling_spec.get("contract") ==
            "ffac-v5-source-bound-visible-review-sampling-spec-v1"
        and review_sampling_spec.get("review_sampling_policy") ==
            v5_review_sampling.POLICY
        and review_sampling_spec.get("frozen_training_request_sha256") ==
            sha256_file(frozen_request_path)
        and review_sampling_spec.get("review_sampling_code_sha256") ==
            sha256_file(Path(v5_review_sampling.__file__))
        and review_sampling_spec.get("source_mask_qa_required") is True,
        "V5 review sampling lacks its frozen visible-source/sand contract")
    frozen_label_cohort = _exact(review_sampling_spec["frozen_label_cohort_path"])
    _require(sha256_file(frozen_label_cohort) ==
             review_sampling_spec["frozen_label_cohort_sha256"]
             and int(_read(frozen_label_cohort).get("labels") or -1) == 4048,
             "V5 review sampling refers to another frozen label cohort")
    receipt, selected = verify_materialization(
        frozen_request_path, receipt_path, boundary_path)
    track1_scope = _track1_scope(receipt)
    track1 = track1_scope is not None
    current43 = track1_scope == "current43"
    admission_path = (_exact(receipt["source_admission_receipt_path"])
                      if track1 else None)
    admission = _read(admission_path) if admission_path is not None else None
    old_request = _read(frozen_request_path)
    old_root = _exact(old_request["output_root"])
    old_pairs = previous._frozen_pairs(old_request)
    old_records, old_regions = previous.region_records(old_pairs)
    _require(len(old_records) == 4048, "Frozen effective label cohort changed")
    sources = {}
    training._load_sources_and_regions(old_pairs, preserve_native_revisions=True,
                                       region_sources=sources)
    source_by_id = {key: str(source.asset_manifest_path.resolve())
                    for key, source in sources.items()}
    source_sha_by_path = {path: sha256_file(Path(path))
                         for path in set(source_by_id.values())}
    retained_ids = {key for key, path in source_by_id.items()
                    if source_sha_by_path[path] in selected}
    if track1:
        partition = _read(_exact(admission["review_partition_path"]))
        partition_by_id = {row["region_id"]: row for row in partition["rows"]}
        _require(set(partition_by_id) == set(old_records)
                 and all(partition_by_id[key]["source_manifest_sha256"] ==
                         source_sha_by_path[source_by_id[key]]
                         for key in old_records)
                 and all(partition_by_id[key]["role"] == "evaluation-only"
                         for key in set(old_records) - retained_ids),
                 "Track1 partition lost an original review or source role")
    else:
        retained_ids = set(old_records)
    active_old_records = {key: old_records[key] for key in retained_ids}
    active_old_regions = [row for row in old_regions if row.region_id in retained_ids]
    chemistry_scope = scope_for_regions(active_old_regions)
    workspace = _exact(cache) / "studio-v5-raw-continuations" / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=False)
    bindings = workspace / "bound-manifests"
    bindings.mkdir()
    checked_aliases: set[tuple[str, str]] = set()
    if current43:
        pairs, old_to_new = _current43_bundles(
            old_pairs, selected, receipt_path, bindings, checked_aliases)
    else:
        pairs = []
        for index, pair in enumerate(old_pairs):
            combined = _read(pair.combined_manifest_path)
            combined["jobs"] = [_bound_job(job, selected, receipt_path, checked_aliases)
                                for job in combined.get("jobs") or []]
            path = bindings / f"{index:03d}_combined_manifest.json"
            _write_new(path, combined)
            pairs.append((pair.decisions_path, path))
    settings = deepcopy(old_request["settings"])
    for stale in ("two_arm_photometry", "feature_cache_root", "completed_fit_source",
                  "imported_completed_heads", "resume_completed_heads"):
        settings.pop(stale, None)
    photometry_settings = {
        "format": FORMAT, "materialization_receipt_path": str(receipt_path),
        "materialization_receipt_sha256": sha256_file(receipt_path),
        "full_native_resolution": True, "ac28_light_anchor": True,
        "new_light_saturated_yellow_gain": 0.0,
        "scientific_input": "sha-bound-float32-log-camera-linear-sidecar",
        "unit_denominator": V5_UNIT_DENOMINATOR,
        "display_jpeg_is_training_input": False,
        "visible_review_sampling_spec_path": str(review_sampling_spec_path),
        "visible_review_sampling_spec_sha256": sha256_file(review_sampling_spec_path),
        "physical_scale_qa_pass": False,
        "bottom_gas_guard_allowed": False,
    }
    if track1:
        photometry_settings.update(
            track1_photo_only=True,
            full_source_qa_receipt_path=str(admission_path),
            full_source_qa_receipt_sha256=sha256_file(admission_path),
            provisional_studio_training=True,
            provisional_photometry_warning_receipt_path=admission["photometry_heldout_qa_path"],
            provisional_photometry_warning_receipt_sha256=admission["photometry_heldout_qa_sha256"],
            mass_analysis_allowed=False)
        settings["v5_review_fit_partition"] = {
            "contract": TRACK1_PARTITION_CONTRACT,
            "path": admission["review_partition_path"],
            "sha256": admission["review_partition_sha256"],
            "region_scope": track1_scope,
            "materialized_source_manifest_sha256s": sorted(selected),
        }
    else:
        photometry_settings.update(
            native_boundary_qa_path=str(_exact(boundary_path)),
            native_boundary_qa_sha256=sha256_file(boundary_path))
    settings.update(backend=backend, radiometry_profile="legacy",
                    stage_workers=8, extraction_workers=extraction_workers,
                    opencv_threads=1, scientific_acceptance=False,
                    human_review_required=True,
                    v5_chemistry_fit_scope=chemistry_scope,
                    v5_raw_photometry=photometry_settings)
    request = prepare_phase_training_request(
        output_parent=workspace / "models", cache_parent=workspace / "cache",
        bundles=pairs, actor=actor, settings=settings,
        withdrawal_snapshot=settings["review_retractions_snapshot"])
    _require(not request.get("locked_review_sessions"),
             "Detached V5 request unexpectedly locked live reviews")
    root = _exact(request["output_root"])
    new_pairs = previous._frozen_pairs(request)
    new_records, _ = previous.region_records(new_pairs)
    verified = {path.casefold(): {
        "source_manifest_sha256": sha256_file(Path(path)),
        "materialized_manifest_sha256": selected[sha256_file(Path(path))]["materialized_manifest_sha256"],
    } for path in {source_by_id[key] for key in retained_ids}}
    annotation = previous.verify_region_binding(
        active_old_records, new_records, verified,
        {key: source_by_id[key] for key in retained_ids})
    _require(annotation["annotation_ids"] == len(retained_ids)
             and annotation["all_ids_classes_times_geometry_and_payloads_unchanged"],
             "V5 request changed a saved label")
    old_collection = _read(old_root / "review_source" / "postflight_manifest.json")
    _require(len(old_collection.get("jobs") or []) == 43,
             "Frozen Studio collection does not have 43 runs")
    old_jobs = old_collection["jobs"]
    aliases = _canonical_aliases(receipt, selected, old_jobs)
    new_jobs = [_bound_job(job, selected, receipt_path, checked_aliases, aliases)
                for job in old_jobs]
    physical_runtime = _copy_physics(old_root, root,
                                     list(zip(old_jobs, new_jobs, strict=True)),
                                     receipt_path)
    old_collection["jobs"] = new_jobs
    _write_new(root / "review_source" / "postflight_manifest.json", old_collection)
    continuation = _read(old_root / "continuation_preparation.json")
    continuation["destination"] = str(root)
    continuation["v5_raw_photometry"] = settings["v5_raw_photometry"]
    if current43:
        original_origins_binding = continuation["annotation_origins"]
        original_origins_path = _exact(original_origins_binding["path"])
        _require(sha256_file(original_origins_path) ==
                 original_origins_binding["sha256"],
                 "Original 4048-ID annotation provenance changed")
        original_origins = _read(original_origins_path)
        _require(set(original_origins["regions"]) == set(old_records),
                 "Original annotation origins omit a frozen review")
        retained_origins = {}
        for identifier in sorted(retained_ids):
            origin = deepcopy(original_origins["regions"][identifier])
            old_index = int(origin["document_index"])
            _require(old_index in old_to_new,
                     f"Current review lost its source document: {identifier}")
            origin["document_index"] = old_to_new[old_index]
            origin["original_document_index"] = old_index
            retained_origins[identifier] = origin
        derived_origins_path = root / "track1_current43_annotation_origins.json"
        _write_new(derived_origins_path, {
            "format": "ffac-v5-track1-current43-annotation-origins-v1",
            "regions": retained_origins,
            "original_review_ids_preserved": 4048,
            "historical_review_ids_evaluation_only_unscorable":
                4048 - len(retained_origins),
            "original_annotation_origins_path": str(original_origins_path),
            "original_annotation_origins_sha256": sha256_file(original_origins_path),
            "source_documents_modified": False,
        })
        continuation["annotation_origins"] = {
            "path": str(derived_origins_path),
            "sha256": sha256_file(derived_origins_path),
        }
        continuation["original_4048_annotation_origins"] = original_origins_binding
    _write_new(root / "continuation_preparation.json", continuation)
    equivalences = [{
        "source_manifest_path": row["source_manifest_path"],
        "source_manifest_sha256": row["source_manifest_sha256"],
        "materialized_manifest_path": row["materialized_manifest_path"],
        "materialized_manifest_sha256": row["materialized_manifest_sha256"],
    } for row in receipt["revisions"]]
    if track1:
        _require(set(new_records) == retained_ids
                 and all(partition_by_id[key]["source_manifest_sha256"] in selected
                         for key in new_records),
                 "Track1 candidate acquired an uncertified review")
        domains = {
            "contract": "ffac-v5-track1-photo-only-regional-domain-preflight-v1",
            "effective_source_revisions": len(selected),
            "effective_annotations": len(new_records),
            "original_review_ids_preserved": 4048,
            "historical_evaluation_only_unscorable_annotations": 4048 - len(new_records),
            "fit_annotations": sum(partition_by_id[key]["role"] == "fit"
                                   for key in new_records),
            "evaluation_only_annotations": sum(row["role"] == "evaluation-only"
                                               for row in partition["rows"]
                                               if row["region_id"] in new_records),
            "partition_sha256": sha256_file(_exact(admission["review_partition_path"])),
            "photo_only": True,
            "physical_facies_used": False,
        }
        domains["domain_contract_sha256"] = _digest(domains)
    else:
        domains = previous.audit_domains(root / "training_request.json",
                                         _exact(settings["parent_model"]), equivalences)
        _require(domains["effective_annotations"] == 4048,
                 "Regional source binding lost saved labels")
    _write_new(root / "regional_domain_preflight.json", domains)
    binding = {
        "format": FORMAT,
        "frozen_request_path": str(frozen_request_path),
        "frozen_request_sha256": sha256_file(frozen_request_path),
        "prepared_request_path": str(root / "training_request.json"),
        "prepared_request_sha256": sha256_file(root / "training_request.json"),
        "materialization_receipt_path": str(receipt_path),
        "materialization_receipt_sha256": sha256_file(receipt_path),
        **({"track1_photo_only": True,
            "full_source_qa_receipt_path": str(admission_path),
            "full_source_qa_receipt_sha256": sha256_file(admission_path),
            "review_partition_path": admission["review_partition_path"],
            "review_partition_sha256": admission["review_partition_sha256"]}
           if track1 else
           {"native_boundary_qa_path": str(_exact(boundary_path)),
            "native_boundary_qa_sha256": sha256_file(boundary_path)}),
        "visible_review_sampling_spec_path": str(review_sampling_spec_path),
        "visible_review_sampling_spec_sha256": sha256_file(
            review_sampling_spec_path),
        "annotation_binding": annotation,
        "original_review_ids_preserved": len(old_records),
        "historical_review_ids_evaluation_only_unscorable": len(old_records) - len(new_records),
        "original_annotation_contract_sha256": _digest([
            {key: value for key, value in old_records[identifier].items()
             if key != "native_manifest_sha256"}
            for identifier in sorted(old_records)]),
        **({"current43_annotation_origins_sha256":
                sha256_file(derived_origins_path)} if current43 else {}),
        "physical_runtime": physical_runtime,
        "regional_domain_preflight_sha256": sha256_file(root / "regional_domain_preflight.json"),
        "regional_domain_contract_sha256": domains["domain_contract_sha256"],
        "parent_summary_sha256": sha256_file(
            _exact(settings["parent_model"]) / "summary.json"),
        "source_equivalences": equivalences,
        "source_revisions": len(selected), "canonical_runs": 43,
        "timeline_frames": receipt["timeline_frames"],
        "studio_published": False,
    }
    _write_new(root / "v5_raw_source_binding.json", binding)
    return binding


def verify_prepared(request_path: Path) -> tuple[dict, dict]:
    request_path = _exact(request_path)
    root = request_path.parent
    binding = _read(root / "v5_raw_source_binding.json")
    # Keep the checks explicit to make the immutable boundary auditable.
    _require(binding.get("format") == FORMAT
             and _exact(binding["prepared_request_path"]) == request_path
             and sha256_file(request_path) == binding["prepared_request_sha256"]
             and sha256_file(root / "regional_domain_preflight.json")
                 == binding["regional_domain_preflight_sha256"],
             "Prepared V5 request or physical preflight changed")
    track1 = binding.get("track1_photo_only") is True
    receipt, selected = verify_materialization(
        Path(binding["frozen_request_path"]),
        Path(binding["materialization_receipt_path"]),
        None if track1 else Path(binding["native_boundary_qa_path"]))
    track1_scope = _track1_scope(receipt)
    _require(track1 == (track1_scope is not None),
             "Prepared V5 Track1 scope differs from materialization")
    _require(sha256_file(_exact(binding["frozen_request_path"]))
             == binding["frozen_request_sha256"]
             and sha256_file(_exact(binding["materialization_receipt_path"]))
                 == binding["materialization_receipt_sha256"]
             and (sha256_file(_exact(binding["full_source_qa_receipt_path"]))
                  == binding["full_source_qa_receipt_sha256"] if track1 else
                  sha256_file(_exact(binding["native_boundary_qa_path"]))
                  == binding["native_boundary_qa_sha256"])
             and phase_runtime.physics_runtime_identity(
                 phase_runtime.load_physics_profile(root)) == binding["physical_runtime"],
             "Prepared V5 source, admission, or context runtime changed")
    request = _read(request_path)
    linear_input = request.get("settings", {}).get("v5_raw_photometry") or {}
    _require(linear_input.get("format") == FORMAT
             and linear_input.get("scientific_input")
                 == "sha-bound-float32-log-camera-linear-sidecar"
             and linear_input.get("unit_denominator") == V5_UNIT_DENOMINATOR
              and linear_input.get("display_jpeg_is_training_input") is False
              and linear_input.get("visible_review_sampling_spec_sha256") ==
                  binding.get("visible_review_sampling_spec_sha256")
              and linear_input.get("physical_scale_qa_pass") is False
              and linear_input.get("bottom_gas_guard_allowed") is False,
             "Prepared V5 model is not bound to the float32 RAW training basis")
    if track1:
        partition_path = _exact(binding["review_partition_path"])
        partition_binding = request["settings"].get("v5_review_fit_partition") or {}
        _require(sha256_file(partition_path) == binding["review_partition_sha256"]
                 and partition_binding.get("contract") == TRACK1_PARTITION_CONTRACT
                 and partition_binding.get("region_scope") == track1_scope
                 and set(partition_binding.get(
                     "materialized_source_manifest_sha256s") or []) == set(selected)
                 and _exact(partition_binding.get("path") or "") == partition_path
                 and partition_binding.get("sha256") == binding["review_partition_sha256"]
                 and linear_input.get("track1_photo_only") is True
                 and linear_input.get("provisional_studio_training") is True
                 and sha256_file(_exact(linear_input.get(
                     "provisional_photometry_warning_receipt_path") or "")) ==
                     linear_input.get("provisional_photometry_warning_receipt_sha256")
                 and linear_input.get("mass_analysis_allowed") is False,
                 "Track1 fit/evaluation partition or review-only scope changed")
    _require(sha256_file(_exact(request["settings"]["parent_model"])
                         / "summary.json") == binding["parent_summary_sha256"],
             "Original FF_AC parent changed since V5 preparation")
    sampling_spec_path = _exact(binding["visible_review_sampling_spec_path"])
    _require(sha256_file(sampling_spec_path) ==
             binding["visible_review_sampling_spec_sha256"]
             and _exact(linear_input["visible_review_sampling_spec_path"]) ==
                 sampling_spec_path,
             "Prepared V5 visible-source review sampling policy changed")
    old_request = _read(_exact(binding["frozen_request_path"]))
    old_pairs = previous._frozen_pairs(old_request)
    new_pairs = previous._frozen_pairs(request)
    old_records, old_regions = previous.region_records(old_pairs)
    new_records, new_regions = previous.region_records(new_pairs)
    _require(len(old_records) == binding["original_review_ids_preserved"] == 4048
             and binding["source_revisions"] == len(selected)
             and binding["timeline_frames"] == receipt["timeline_frames"],
             "V5 original review/source inventory changed")
    source_by_id = {}
    training._load_sources_and_regions(
        old_pairs, preserve_native_revisions=True, region_sources=source_by_id)
    source_path_by_id = {
        key: str(source.asset_manifest_path.resolve())
        for key, source in source_by_id.items()
    }
    source_sha_by_path = {path: sha256_file(Path(path))
                         for path in set(source_path_by_id.values())}
    retained_ids = ({key for key, path in source_path_by_id.items()
                     if source_sha_by_path[path] in selected}
                    if track1 else set(old_records))
    _require(set(new_records) == retained_ids
             and len(old_records) - len(new_records) ==
                 binding["historical_review_ids_evaluation_only_unscorable"]
             and _digest([
                 {key: value for key, value in old_records[identifier].items()
                  if key != "native_manifest_sha256"}
                 for identifier in sorted(old_records)]) ==
                 binding["original_annotation_contract_sha256"],
             "Track1 candidate silently changed original or active review IDs")
    chemistry = request["settings"].get("v5_chemistry_fit_scope") or {}
    _require(chemistry == scope_for_regions(
                 [row for row in old_regions if row.region_id in retained_ids]),
             "V5 chemistry fit exclusion differs from frozen review IDs")
    verify_scope(chemistry, new_regions)
    if track1:
        partition = _read(partition_path)
        partition_by_id = {row["region_id"]: row for row in partition["rows"]}
        _require(set(partition_by_id) == set(old_records)
                 and all(partition_by_id[key]["source_manifest_sha256"] ==
                         source_sha_by_path[source_path_by_id[key]] for key in old_records)
                 and all(partition_by_id[key]["role"] == "evaluation-only"
                         for key in set(old_records) - retained_ids),
                 "Track1 historical source/role partition changed")
    verified = {
        path.casefold(): {
            "source_manifest_sha256": sha256_file(Path(path)),
            "materialized_manifest_sha256": selected[sha256_file(Path(path))][
                "materialized_manifest_sha256"],
        }
        for path in {source_path_by_id[key] for key in retained_ids}
    }
    _require(previous.verify_region_binding(
        {key: old_records[key] for key in retained_ids}, new_records, verified,
        {key: source_path_by_id[key] for key in retained_ids})
        == binding["annotation_binding"],
        "Prepared V5 annotation identities, classes, times, or geometry changed")
    if track1_scope == "current43":
        continuation = _read(root / "continuation_preparation.json")
        origins_binding = continuation.get("annotation_origins") or {}
        origins_path = _exact(origins_binding.get("path") or "")
        _require(origins_path == root / "track1_current43_annotation_origins.json"
                 and sha256_file(origins_path) ==
                     binding["current43_annotation_origins_sha256"] ==
                     origins_binding.get("sha256")
                 and continuation.get("original_4048_annotation_origins") ==
                     _read(_exact(old_request["output_root"]) /
                           "continuation_preparation.json")["annotation_origins"],
                 "Current43 annotation-origin lineage changed")
        origins = _read(origins_path)
        _require(set(origins.get("regions") or {}) == retained_ids
                 and origins.get("original_review_ids_preserved") == 4048
                 and origins.get(
                     "historical_review_ids_evaluation_only_unscorable") ==
                     4048 - len(retained_ids)
                 and sha256_file(_exact(origins[
                     "original_annotation_origins_path"])) ==
                     origins["original_annotation_origins_sha256"],
                 "Current43 source indices or original ID inventory changed")
        _require(all(0 <= int(origin["document_index"]) < len(new_pairs)
                     for origin in origins["regions"].values()),
                 "Current43 regional origin points outside its derived pairs")
    old_collection = _read(_exact(old_request["output_root"])
                           / "review_source" / "postflight_manifest.json")
    new_collection = _read(root / "review_source" / "postflight_manifest.json")
    _require(len(old_collection["jobs"]) == len(new_collection["jobs"]) == 43,
             "Prepared V5 canonical Studio timeline changed")
    checked_aliases: set[tuple[str, str]] = set()
    aliases = _canonical_aliases(receipt, selected, old_collection["jobs"])
    for old_job, new_job in zip(old_collection["jobs"], new_collection["jobs"],
                                strict=True):
        _require(_bound_job(old_job, selected,
                            _exact(binding["materialization_receipt_path"]),
                            checked_aliases, aliases) == new_job,
                 f"Prepared V5 Studio source changed: {old_job['run']}")
    domain = (_read(root / "regional_domain_preflight.json") if track1 else
              previous.audit_domains(
                  request_path, _exact(request["settings"]["parent_model"]),
                  binding["source_equivalences"]))
    _require(domain["domain_contract_sha256"]
             == binding["regional_domain_contract_sha256"],
             "Prepared V5 photo support or physical domains changed")
    profile = phase_runtime.load_physics_profile(root)
    _require(profile is not None, "Prepared V5 physical profile disappeared")
    if track1:
        _require(profile.get("v5_track1_review_scope", {}).get("mass_analysis_allowed") is False
                 and all((entry.get("provenance") or {}).get("v5_track1_photo_geometry")
                         for entry in profile["runs"].values()),
                 "Track1 context acquired a physical or unbound profile")
    for key, entry in profile["runs"].items():
        context = (root / entry["path"]).resolve()
        _require(context.is_relative_to(root)
                 and sha256_file(context) == entry["sha256"],
                 f"Prepared V5 physical grid changed: {key}")
    request["request_path"] = str(request_path)
    request["frozen_pairs"] = [(p.decisions_path, p.combined_manifest_path)
                                for p in new_pairs]
    return request, binding


def verify_native_maps(jobs: list[dict]) -> dict:
    """Check combined and three separate full-canvas RGBA maps per frame."""
    totals = {"runs": 0, "frames": 0, "phase_pngs": 0,
              "phase_layer_pngs": 0}
    for job in jobs:
        run = str(job["run"]).casefold()
        _source_root, _manifest_path, native = optical.native_source(job)
        count = int(native["frame_count"])
        expected = [f"frame_{index:03d}.png" for index in range(count)]
        base = Path(job["phase_map_output_dir"]) / "native_phase_maps" / run
        expected_folder = f"A_{job['params_sha256'][:16]}"
        candidates = [path for path in base.iterdir() if path.is_dir()]
        _require(len(candidates) == 1 and candidates[0].name == expected_folder,
                 f"V5 native phase-map identity is ambiguous: {run}")
        folder = candidates[0]
        found = sorted(path.name for path in folder.glob("frame_*.png"))
        _require(found == expected,
                 f"V5 native phase maps are incomplete: {run}")
        for phase in ("aqueous", "gas", "mixed"):
            layer = folder / phase
            layer_found = sorted(path.name for path in layer.glob("frame_*.png"))
            _require(layer_found == expected,
                     f"V5 {phase} phase layer is incomplete: {run}")
        for index, filename in enumerate(expected):
            for path in (folder / filename,
                         *(folder / phase / filename
                           for phase in ("aqueous", "gas", "mixed"))):
                with Image.open(path) as image:
                    _require(image.format == "PNG" and image.mode == "RGBA"
                             and list(image.size) == native["canvas_wh"],
                             f"V5 phase map is not a full native RGBA PNG: {run}/{index}/{path.parent.name}")
        totals["phase_pngs"] += count
        totals["phase_layer_pngs"] += 3 * count
        totals["runs"] += 1
        totals["frames"] += count
    _require(totals == {"runs": 43, "frames": 864, "phase_pngs": 864,
                        "phase_layer_pngs": 2592},
             "V5 native phase-map set is not complete")
    return totals


def _checkpoint_optical_stage(request: dict, binding: dict, job: Path, *,
                              coupled_applicability: dict | None = None) -> dict:
    """Freeze an already fitted optical root outside the Studio model library."""
    root = _exact(request["output_root"])
    convergence = series.verify_optical_selected_convergence(root)
    fitted = verify_fitted(root, allow_nonconverged=True)
    _require(fitted["series"] == "ffac" and fitted["adapters"] == ["generic"]
             and not fitted["nonconverged_heads"]
             and fitted["physics_runtime"] == binding["physical_runtime"],
             "V5 optical fitting did not meet convergence or physical gates")
    hidden = job / "optical_parent" / root.name
    publish_fitted(root, hidden, destination_volume_staging_root(hidden),
                   allow_nonconverged=False)
    _require(not (hidden / "validation").exists(),
             "Intermediate optical model appeared in Studio")
    checked_hidden = verify_fitted(hidden, allow_nonconverged=False)
    _require(checked_hidden == fitted, "Hidden optical checkpoint differs from fit")
    _write_new(job / "optical_convergence.json", convergence)
    checkpoint = {
        "format": OPTICAL_CHECKPOINT_FORMAT,
        "prepared_request_path": str(_exact(request["request_path"])),
        "prepared_request_sha256": sha256_file(_exact(request["request_path"])),
        "source_binding_sha256": sha256_file(root / "v5_raw_source_binding.json"),
        "convergence_sha256": sha256_file(job / "optical_convergence.json"),
        "hidden_model_path": str(hidden),
        "publication_receipt_sha256": sha256_file(hidden / "publication_receipt.json"),
        "verified_fitted": checked_hidden,
        "validation_only": True,
        "native_phase_maps_built": False,
        "regional_layer_fitted": False,
        "studio_published": False,
    }
    if coupled_applicability is not None:
        checkpoint["coupled_aqueous_applicability"] = coupled_applicability
        checkpoint["coupled_applicability_receipt_sha256"] = sha256_file(
            job / "coupled_applicability.json")
        checkpoint["pre_coupled_training_request_sha256"] = sha256_file(
            job / "pre_coupled_training_request.json")
    _write_new(job / "optical_checkpoint.json", checkpoint)
    return checkpoint


def _fit_optical_stage(request: dict, binding: dict, job: Path) -> dict:
    """Fit once and freeze the optical heads outside the Studio model library."""
    optical.fit(request)
    return _checkpoint_optical_stage(request, binding, job)


def _coupled_applicability(root: Path, binding: dict, receipt: dict,
                           code_path: Path) -> dict:
    """Verify the source-bound receipt for a standard aqueous-refiner no-op."""
    target_classes = ("aqueous_absent_region", "aqueous_present_region")
    with (root / "region_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    positive = sum(row["phase_class"] in target_classes
                   and float(row["supervision_weight"]) > 0 for row in rows)
    _require(len(rows) == binding["annotation_binding"]["annotation_ids"] == 4048
             and positive == 0
             and receipt.get("contract") ==
                 "ffac-v5-track1-coupled-applicability-v1"
             and receipt.get("model_id") == root.name
             and receipt.get("request_sha256") ==
                 binding["prepared_request_sha256"] ==
                 sha256_file(root / "training_request.json")
             and receipt.get("inventory_sha256") ==
                 sha256_file(root / "region_inventory.csv")
             and receipt.get("coupled_code_sha256") == sha256_file(code_path)
             and receipt.get("target_classes") == list(target_classes)
             and receipt.get("target_positive_weight_regions") == 0
             and receipt.get("original_review_ids") == 4048
             and receipt.get("standard_refine_unpublished_result") ==
                 "no-op-no-target-regions"
             and receipt.get("request_mutated") is False,
             "Coupled no-op receipt differs from the frozen optical training inventory")
    return receipt


def verify_optical_checkpoint(request_path: Path, optical_job: Path,
                              binding: dict) -> tuple[Path, dict]:
    """Verify the immutable intermediate without refitting optical heads."""
    request_path, optical_job = _exact(request_path), _exact(optical_job)
    root = request_path.parent
    checkpoint_path = optical_job / "optical_checkpoint.json"
    checkpoint = _read(checkpoint_path)
    hidden = _exact(checkpoint["hidden_model_path"])
    _require(checkpoint.get("format") == OPTICAL_CHECKPOINT_FORMAT
             and _exact(checkpoint["prepared_request_path"]) == request_path
             and checkpoint["prepared_request_sha256"] == sha256_file(request_path)
             and checkpoint["source_binding_sha256"] ==
                 sha256_file(root / "v5_raw_source_binding.json"),
             "V5 optical checkpoint belongs to another prepared source")
    applicability = checkpoint.get("coupled_aqueous_applicability")
    if applicability is not None:
        applicability_path = optical_job / "coupled_applicability.json"
        code_snapshot = optical_job / "coupled_refiner_code.py"
        _require(checkpoint.get("coupled_applicability_receipt_sha256") ==
                     sha256_file(applicability_path)
                 and applicability == _coupled_applicability(
                     root, binding, _read(applicability_path), code_snapshot)
                 and checkpoint.get("pre_coupled_training_request_sha256") ==
                     sha256_file(optical_job / "pre_coupled_training_request.json") ==
                     binding["prepared_request_sha256"]
                 and not (root / "continuation_evidence" / "coupled_revision.json").exists()
                 and not (hidden / "continuation_evidence" / "coupled_revision.json").exists(),
                 "V5 checkpoint falsely claims coupled refinement was inapplicable")
    _require(hidden == optical_job / "optical_parent" / root.name
             and sha256_file(optical_job / "optical_convergence.json") ==
                 checkpoint["convergence_sha256"]
             and not (hidden / "validation").exists()
             and sha256_file(hidden / "publication_receipt.json") ==
                 checkpoint["publication_receipt_sha256"],
             "V5 optical checkpoint path, convergence, or publication changed")
    publication = _read(hidden / "publication_receipt.json")
    _require(publication.get("destination") == str(hidden)
             and publication.get("native_validation_pending") is True,
             "V5 intermediate optical model was not kept hidden")
    for relative, row in (publication.get("published_files") or {}).items():
        path = (hidden / relative).resolve()
        _require(path.is_relative_to(hidden) and sha256_file(path) == row["sha256"],
                 f"V5 optical checkpoint file changed: {relative}")
    checked = verify_fitted(hidden, allow_nonconverged=False)
    _require(checked == checkpoint["verified_fitted"]
             and checked["series"] == "ffac"
             and checked["adapters"] == ["generic"]
             and not checked["nonconverged_heads"]
             and checked["physics_runtime"] == binding["physical_runtime"],
             "V5 optical checkpoint fit or physical runtime changed")
    return hidden, checkpoint


def fit_optical(request_path: Path, job: Path) -> dict:
    """Make a resumable optical checkpoint before regional/facies QA is ready."""
    request, binding = verify_prepared(request_path)
    root, job = _exact(request["output_root"]), _exact(job)
    _require(not job.exists() and not job.is_relative_to(root),
             "Choose a new detached V5 optical checkpoint job")
    job.mkdir(parents=True, exist_ok=False)
    _write_new(job / "v5_raw_source_binding.json", binding)
    return _fit_optical_stage(request, binding, job)


def checkpoint_existing_optical(request_path: Path, job: Path, *,
                                coupled_applicability_sha256: str) -> dict:
    """Checkpoint the completed ordinary fit without launching another fit.

    This entry point only accepts the detached job that froze the prepared
    request before the attempted (inapplicable) aqueous refinement.
    """
    request, binding = verify_prepared(request_path)
    root, job = _exact(request["output_root"]), _exact(job)
    pre_coupled = job / "pre_coupled_training_request.json"
    bound = job / "v5_raw_source_binding.json"
    applicability_path = job / "coupled_applicability.json"
    _require(job.is_dir() and not job.is_relative_to(root)
             and not root.is_relative_to(job)
             and pre_coupled.is_file() and bound.is_file()
             and sha256_file(pre_coupled) ==
                 binding["prepared_request_sha256"] == sha256_file(_exact(request_path))
             and sha256_file(bound) ==
                 sha256_file(root / "v5_raw_source_binding.json")
             and not (job / "optical_checkpoint.json").exists()
             and not (job / "optical_convergence.json").exists()
             and not (job / "optical_parent").exists()
             and not (root / "continuation_evidence" / "coupled_revision.json").exists(),
             "Existing optical job lacks an exact unmodified prepared fit snapshot")
    _require(len(coupled_applicability_sha256) == 64
             and applicability_path.is_file()
             and sha256_file(applicability_path) == coupled_applicability_sha256,
             "Coupled no-op receipt is not the requested immutable evidence")
    applicability = _coupled_applicability(
        root, binding, _read(applicability_path),
        Path(__file__).with_name("train_coupled_aqueous_revision.py"))
    refiner_snapshot = job / "coupled_refiner_code.py"
    if not refiner_snapshot.exists():
        shutil.copy2(Path(__file__).with_name("train_coupled_aqueous_revision.py"),
                     refiner_snapshot)
    _require(sha256_file(refiner_snapshot) ==
             applicability["coupled_code_sha256"],
             "Coupled-refiner code changed while freezing the no-op decision")
    return _checkpoint_optical_stage(
        request, binding, job, coupled_applicability=applicability)


def _complete_stage(request: dict, binding: dict, hidden: Path, job: Path, *,
                    backend: str, cpus: int | None,
                    memory_gb: float | None, optical_checkpoint_path: Path,
                    optical_refitted_in_this_job: bool) -> dict:
    """Fit regional layer and native maps from a verified optical checkpoint."""
    root = _exact(request["output_root"])
    library = _exact(request["settings"]["parent_model"]).parent
    regional_args = regional.prepare_launch(hidden, library,
        Path(request["cache_root"]).parent, "ffac", request["created_by"],
        backend=backend, cpus=cpus, memory_gb=memory_gb)
    regional_args.output = job / "regional_candidate" / regional_args.output.name
    regional_args.job = job / "regional_run"
    regional_args.bottom_gas_guard = False
    regional_args.region_edge_guard = False
    regional_args.metric_postprocessing = False
    regional_args.nominal_review_gas_minimum = True
    regional_args.job.mkdir(parents=True, exist_ok=False)
    _write_new(regional_args.job / "launch_request.json", {
        "format": FORMAT,
        "parent": str(hidden), "output": str(regional_args.output),
        "job": str(regional_args.job),
        "bottom_gas_guard": False, "region_edge_guard": False,
        "output_guards": False, "free_gas_cleanup": True,
        "nominal_review_gas_minimum_mm2": 2.5,
        "metric_postprocessing": False,
        "physical_scale_qa_pass": False,
        "validation_only_physical_scale_warning": (
            "Native distance calibration is unverified; only the nominal "
            "2.5 mm2 gas-speckle minimum is applied for Studio review. "
            "Position guards and mass interpretation are disabled."),
        "review_candidate_only": True, "studio_published": False,
    })
    hidden_request = _read(hidden / "training_request.json")
    pairs = previous._frozen_pairs(hidden_request)
    _, regions = previous.region_records(pairs)
    _require(len(regions) == binding["annotation_binding"]["annotation_ids"],
             "Regional V5 input differs from optical labels")
    continuation = _read(root / "continuation_preparation.json")
    evidence = {
        "parent": str(hidden),
        "pairs": [[str(p.decisions_path), str(p.combined_manifest_path)] for p in pairs],
        "regions": [asdict(r) for r in regions],
        "audit": {
            "format": FORMAT,
            "family_models": continuation["family_models"],
            "annotation_origins": continuation["annotation_origins"],
            # Existing regional binder uses this key for exact source-revision
            # equivalence. Its contents carry explicit V5 provenance.
            "two_arm_source_binding": {
                "source_equivalences": binding["source_equivalences"]},
            "v5_raw_source_binding": {
                "receipt_sha256": binding["materialization_receipt_sha256"],
                "annotation_contract_sha256": binding["annotation_binding"]["annotation_contract_sha256"],
            },
            "v5_chemistry_fit_scope": hidden_request["settings"]["v5_chemistry_fit_scope"],
            **({"v5_review_fit_partition": hidden_request["settings"][
                "v5_review_fit_partition"]} if binding.get("track1_photo_only") else {}),
            "original_user_documents_modified": False,
        },
    }
    _write_new(regional_args.job / "evidence.json", evidence)
    _write_new(regional_args.job / "supervision_policy.json", {
        "policy": "missed-aqueous-is-dense-reviewer-clarification-20260920",
        "reviewer_statement": "The whole marked area is aqueous CO2",
        "corrected_regions": [r.region_id for r in regions if
                              r.source == "manual" and
                              r.payload.get("category") == "missed_visible_aqueous_co2"],
        "original_documents_modified": False,
        "mixed_phase_classes_unchanged": True,
    })
    plan, domains = regional_training.bind_domains(evidence, root / "review_source")
    preflight = _read(root / "regional_domain_preflight.json")
    _require(len(domains) == preflight["effective_source_revisions"]
             and len(plan) > 0,
             "V5 regional physical domains differ from frozen preflight")
    regional_contract = _digest([
        {key: value for key, value in record.items()
         if key != "native_manifest_sha256"}
        for _, record in sorted(previous.region_records(pairs)[0].items())
    ])
    _require(regional_contract
             == binding["annotation_binding"]["annotation_contract_sha256"],
             "Regional V5 snapshot differs from optical annotation contract")
    _write_new(regional_args.job / "snapshot_parity.json", {
        "format": FORMAT,
        "optical_annotation_contract_sha256": binding["annotation_binding"][
            "annotation_contract_sha256"],
        "regional_annotation_contract_sha256": regional_contract,
        "regions": len(regions), "same_frozen_pairs": True,
    })
    regional_args.status_callback = None
    update.run(regional_args)
    candidate = Path(regional_args.output)
    candidate_profile = phase_runtime.load_physics_profile(candidate)
    _require(candidate_profile is not None and not any(
        key in candidate_profile for key in (
            "bottom_gas_guard", "output_guards",
            "region_edge_guard")),
        "V5 validation-only candidate retained a metric-scale postprocessor")
    if binding.get("track1_photo_only"):
        _require(candidate_profile.get("v5_track1_review_scope") == {
                     "contract": "ffac-v5-track1-photo-only-review-context-v1",
                     "facies_ids_used": False,
                     "physical_scale_qa_pass": False,
                     "absolute_position_rules_enabled": False,
                     "mass_analysis_allowed": False,
                 },
                 "Track1 Studio candidate lost its photo-only review scope")
    _require(candidate_profile.get("v5_nominal_review_gas_minimum") == {
                 "contract": "ffac-v5-nominal-review-gas-minimum-v1",
                 "minimum_area_mm2": 2.5,
                 "physical_scale_qa_pass": False,
                 "mass_analysis_allowed": False,
             } and candidate_profile.get("free_gas_cleanup") == {
                 "format": "native-free-gas-minimum-v1",
                 "minimum_area_mm2": 2.5,
             }, "V5 Studio candidate lost its nominal 2.5 mm2 gas minimum")
    preparation = _read(candidate / "studio_preparation.json")
    collection = _read(candidate / "validation" / "postflight_manifest.json")
    jobs = collection.get("jobs") or []
    _require(preparation.get("status") == "ready"
             and len(jobs) == 43
             and len({str(row["run"]).casefold() for row in jobs}) == 43,
             "V5 candidate lacks a complete native Studio review campaign")
    canonical = {
        row["run"]: row for row in
        _read(_exact(binding["materialization_receipt_path"]))["canonical_mapping"]
    }
    for review_job in jobs:
        run = str(review_job["run"]).casefold()
        source_root, manifest_path, native = optical.native_source(review_job)
        selected = canonical.get(run)
        _require(selected is not None
                 and _exact(source_root) == _exact(selected["materialized_review_root"])
                 and sha256_file(manifest_path)
                     == selected["materialized_manifest_sha256"]
                 and int(native["frame_count"]) == len(native["frame_sources"])
                 and review_job.get("quality_flags", {}).get("human_review_required") is True
                 and review_job.get("phase_model_only") is True,
                 f"V5 Studio maps lost the exact corrected source or review-only policy: {run}")
    native_maps = verify_native_maps(jobs)
    result = {
        "format": FORMAT,
        "candidate_model": str(candidate),
        "optical_heads_refitted": bool(optical_refitted_in_this_job),
        "optical_checkpoint_path": str(_exact(optical_checkpoint_path)),
        "optical_checkpoint_sha256": sha256_file(optical_checkpoint_path),
        "regional_layer_refitted": True,
        "native_studio_maps_built": True,
        "native_studio_runs": native_maps["runs"],
        "native_studio_frames": native_maps["frames"],
        "native_phase_pngs": native_maps["phase_pngs"],
        "native_phase_layer_pngs": native_maps["phase_layer_pngs"],
        "review_candidate_only": True,
        "track1_photo_only": bool(binding.get("track1_photo_only")),
        "provisional_studio_training": bool(binding.get("track1_photo_only")),
        "candidate_label": ("FF_AC RAW · supported sand · NOT FOR MASS ANALYSIS"
                            if binding.get("track1_photo_only") else
                            "FF_AC RAW · native physical review candidate"),
        "photometry_warning_receipt_path": (
            _read(_exact(binding["full_source_qa_receipt_path"]))[
                "photometry_heldout_qa_path"]
            if binding.get("track1_photo_only") else None),
        "photometry_warnings": (["donor 4x4 tile CV failed",
                                  "24 h saturated-yellow comparison failed"]
                                 if binding.get("track1_photo_only") else []),
        "physical_scale_qa_pass": False,
        "position_dependent_metric_guards_disabled": True,
        "nominal_review_gas_minimum_mm2": 2.5,
        "mass_analysis_allowed": False,
        "studio_published": False,
        "qa_pending": True,
        "source_binding_sha256": sha256_file(root / "v5_raw_source_binding.json"),
    }
    _write_new(job / "result.json", result)
    return result


def complete_from_optical(request_path: Path, optical_job: Path, job: Path, *,
                          backend: str = "auto", cpus: int | None = None,
                          memory_gb: float | None = None) -> dict:
    """Finish regional training and maps without recomputing optical heads."""
    request, binding = verify_prepared(request_path)
    root, job = _exact(request["output_root"]), _exact(job)
    optical_job = _exact(optical_job)
    _require(not job.exists() and not job.is_relative_to(root)
             and not job.is_relative_to(optical_job)
             and not optical_job.is_relative_to(job),
             "Choose a new detached regional result job")
    hidden, _checkpoint = verify_optical_checkpoint(
        request_path, optical_job, binding)
    job.mkdir(parents=True, exist_ok=False)
    _write_new(job / "v5_raw_source_binding.json", binding)
    _write_new(job / "optical_checkpoint_source.json", {
        "contract": OPTICAL_CHECKPOINT_FORMAT,
        "optical_checkpoint_path": str(optical_job / "optical_checkpoint.json"),
        "optical_checkpoint_sha256": sha256_file(
            optical_job / "optical_checkpoint.json"),
        "hidden_model_path": str(hidden),
        "hidden_publication_sha256": sha256_file(
            hidden / "publication_receipt.json"),
    })
    return _complete_stage(
        request, binding, hidden, job, backend=backend, cpus=cpus,
        memory_gb=memory_gb,
        optical_checkpoint_path=optical_job / "optical_checkpoint.json",
        optical_refitted_in_this_job=False)


def execute(request_path: Path, job: Path, *, backend: str = "auto",
            cpus: int | None = None, memory_gb: float | None = None) -> dict:
    """Fit optical heads, then regional layer and maps in one detached job."""
    request, binding = verify_prepared(request_path)
    root, job = _exact(request["output_root"]), _exact(job)
    _require(not job.exists() and not job.is_relative_to(root),
             "Choose a new detached V5 result job")
    job.mkdir(parents=True, exist_ok=False)
    _write_new(job / "v5_raw_source_binding.json", binding)
    _fit_optical_stage(request, binding, job)
    hidden, _checkpoint = verify_optical_checkpoint(request_path, job, binding)
    return _complete_stage(
        request, binding, hidden, job, backend=backend, cpus=cpus,
        memory_gb=memory_gb, optical_checkpoint_path=job / "optical_checkpoint.json",
        optical_refitted_in_this_job=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--fit-optical", action="store_true")
    mode.add_argument("--checkpoint-existing-optical", action="store_true")
    mode.add_argument("--complete-optical", action="store_true")
    parser.add_argument("--frozen-request", type=Path)
    parser.add_argument("--materialization-receipt", type=Path)
    parser.add_argument("--boundary-qa", type=Path)
    parser.add_argument("--review-sampling-spec", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--actor", default="Olav Folkvord")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--optical-job", type=Path)
    parser.add_argument("--coupled-applicability-sha256")
    parser.add_argument("--backend", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--extraction-workers", type=int, default=16)
    parser.add_argument("--cpus", type=int, default=24)
    parser.add_argument("--memory-gb", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.prepare:
        value = prepare(args.frozen_request, args.materialization_receipt,
                        args.boundary_qa, args.cache,
                        review_sampling_spec_path=args.review_sampling_spec,
                        actor=args.actor,
                        backend=args.backend,
                        extraction_workers=args.extraction_workers)
    elif args.verify:
        value = verify_prepared(args.request)[1]
    elif args.fit_optical:
        value = fit_optical(args.request, args.job)
    elif args.checkpoint_existing_optical:
        value = checkpoint_existing_optical(
            args.request, args.job,
            coupled_applicability_sha256=args.coupled_applicability_sha256 or "")
    elif args.complete_optical:
        value = complete_from_optical(args.request, args.optical_job, args.job,
                                      backend=args.backend, cpus=args.cpus,
                                      memory_gb=args.memory_gb)
    else:
        value = execute(args.request, args.job, backend=args.backend,
                        cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
