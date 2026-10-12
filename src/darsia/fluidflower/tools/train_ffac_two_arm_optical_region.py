"""Detached FF_AC optical/regional training on exact raw-rig photometry arms.

The frozen reviewer request is an immutable source of labels. A materialization
receipt supplies new corrected images for the same raw frames. Preparation
creates a *new* request and binds every historical label and every registered
Studio timeline to that receipt; execution fits both layers in a detached job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_training import logistic, trainer as training
from darsia.fluidflower.phase_workflow import prepare_phase_training_request
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools import train_studio_optical_then_region as ac
from darsia.fluidflower.tools import train_studio_optical_then_region_series as series
from darsia.fluidflower.tools import train_studio_series_update as regional
from darsia.fluidflower.tools import train_studio_region_update as update
from darsia.fluidflower.tools import train_region_coverage_overnight as regional_training
from darsia.fluidflower.tools.publish_fitted_phase_candidate import publish_fitted, verify_fitted
from darsia.fluidflower.tools.prepare_grain_gas_retraining import physics_files


FORMAT = "ffac-two-arm-optical-regional-v1"
MATERIALIZATION_FORMAT = "ffac-two-arm-native-materialization-v1"
REFERENCE_FORMAT = "ffac-two-arm-reference-availability-v1"
FEATURE_CACHE_FORMAT = "ffac-two-arm-verified-feature-cache-v1"
EXTRACTION_CACHE_FORMAT = "ffac-two-arm-complete-extraction-cache-v1"
ARMS = ("within_run", "global", "sequential_global")
PAIRED_ARMS = ("within_run", "global")
GEOMETRY_FIELDS = (
    "schema", "dataset", "run", "kind", "frame_count", "canvas_wh",
    "analysis_image_shape_hw", "content_box_xywh", "full_analysis_resolution",
    "additional_spatial_resampling", "base_image_shared_between_candidates",
    "contours_stored_as_transparent_overlays", "phase_topology", "phase_quiz_domain",
    "injection_points", "reference_registration", "domain", "geometry_revision_id",
)
FRAME_PHOTOMETRY_FIELDS = {
    "native_jpeg_sha256", "corrected_array_sha256", "color_corrections",
    "color_correction_flagged", "color_correction_active", "frame_seconds",
    "source_sha256", "source_raw_sha256", "frame_sha256", "frame_path",
    "phase_map",
}
REFERENCE_PHOTOMETRY_FIELDS = {"corrected_array_sha256", "materialized_native_sha256"}
AC47_V4_SOURCE_SHA256 = "12ab32e1593ec49460428f7ae380d9673c1f9d8c5530656e112cc328b52318be"
AC47_V4_RECEIPT_SHA256 = "1c93f303d84ffc58874965af5a53611817f28fb1cc1596b79c31cd76573d2841"
AC47_V4_PARENT_SHA256 = "4ba36305071601e11716c86609ce86138ce8b5bee2bf33f5236dcb4c1f6f252e"
AC47_V4_REVISION_ID = "ac47-neutral-tone-only-v4-20260925"
AC47_V4_CHANGED_FRAMES = [1, 3, 8, 13, 15]
AC47_PRIOR_WITHIN_SOURCE_SHA256 = "106ed263de8372cc408544327d98dc2f64369e994b35a6cea2db5e2219c06448"
AC47_PRIOR_WITHIN_RECEIPT_SHA256 = "f343901375a91215d86001b00697a5596e91ace0b2131abf263949cb69e36fc2"
AC47_ARM_REVISION_IDS = {
    "within_run": "ffac-two-arm-within-run-legacy-v2-v1",
    "global": "ffac-two-arm-global-robust16-legacy-v2-v1",
    "sequential_global": "ffac-sequential-within-run-common-start-deterministic-chartqa-v3",
}
AC61_COMPOSITE_SOURCE_SHA256 = "14bbf139f5cba7b449d2a6f6312a147785f467b8d2154ff83992783698e0dffe"
AC61_COMPOSITE_PREPARED_SHA256 = "18d0ba62bd789c90cd36f317239bd7f438f96a50e50afda4d84df017858df244"
AC61_COMPOSITE_REPAIR_SHA256 = "3243488b89a71dfc73663b91bebd0383263b96b25ba477b8cae52e6185a897a7"
AC61_COMPOSITE_ARRAY_SHA256 = "36d171adfa57f295b13e85bcd9face9bfd4216e755a20bf49c6e7fd1b8bd8d8b"
AC61_COMPOSITE_MEMBER_SHAS = (
    ("DSC21674.JPG", "420bfa273bb83c8b8e1b97ad079b461336984d1af33e3d4caab89b3b2e5f8b4f"),
    ("DSC21675.JPG", "5c391817f1ef539328bf6fddcce4d796f2fdf8541d0f1e209847444be682593b"),
    ("DSC21676.JPG", "feae6b4de035bb19f2716d9bd4a3d40d7e8443dcd1d6200c8aaf979db236b985"),
)


def read(path: Path) -> dict:
    return json.loads(resolve_review_path(Path(path)).read_text(encoding="utf-8-sig"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(value: str | Path) -> Path:
    return resolve_review_path(Path(value)).resolve()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def extraction_feature_columns(settings: dict) -> int:
    """The trainer caches its maximal v5 matrix before selecting model heads."""
    return len(replace(training.MAXIMAL_CONTRACT,
                       generic_version=settings["feature_version"]).feature_names)


def _extraction_completion(root: Path, source: dict, inventory: dict,
                           feature_columns: int) -> tuple[list[dict], dict]:
    """Verify finished extraction directly; never synthesize a fitted summary."""
    import numpy as np
    from darsia.fluidflower.phase_training.radiometry import resolve_profile

    settings = source["settings"]
    require(settings.get("preserve_native_revisions") is True,
            "Extraction-only reuse requires frozen native source revisions")
    origins, label_audit = {}, {}
    _sources, regions, _ = training._load_sources_and_regions(_frozen_pairs(source),
        withdrawal_snapshot=settings.get("review_retractions_snapshot"),
        preserve_native_revisions=True, region_sources=origins, label_policy_audit=label_audit)
    selected_series = {str(value).casefold().replace("_", "") for value in settings.get("series", [])}
    if selected_series:
        regions = [region for region in regions if region.dataset.casefold().replace("_", "") in selected_series]
        origins = {region.region_id: origins[region.region_id] for region in regions}
    require(len(regions) == inventory.get("regions")
            and dict(Counter(row.phase_class for row in regions)) == inventory.get("phase_classes")
            and dict(Counter(f"{row.dataset}/{row.run}" for row in regions)) == inventory.get("regions_by_series_run")
            and label_audit == inventory.get("training_label_policy"),
            "Extraction inventory differs from frozen labels or label policy")
    groups = training._native_revision_task_groups(regions, origins)
    contract = replace(training.MAXIMAL_CONTRACT, generic_version=settings["feature_version"])
    require(len(contract.feature_names) == feature_columns, "Extraction feature contract changed")
    radiometry = resolve_profile(settings.get("radiometry_profile"))
    reports, staged_receipts, ordered_regions = [], [], []
    for run_source, run_regions in groups:
        manifest_sha = sha256_file(run_source.asset_manifest_path)
        suffix = training._canonical_hash(str(run_source.review_root.resolve()))[:16]
        staged_root = root / "assets" / f"{run_source.dataset}_{run_source.run}_{manifest_sha[:16]}_{suffix}"
        receipt_path = staged_root / "staged_assets.json"
        require(receipt_path.is_file() and not receipt_path.is_symlink(),
                f"Missing completed native staging receipt: {receipt_path}")
        staged_receipt = read(receipt_path)
        manifest = read(run_source.asset_manifest_path)
        count = int(manifest["frame_count"])
        local_files = [staged_root / "manifest.json", staged_root / "pre_injection.jpg",
                       *(staged_root / f"frame_{index:03d}.jpg" for index in range(count))]
        source_files = [run_source.asset_manifest_path,
                        run_source.review_root / "native_references" / run_source.run / "pre_injection.jpg",
                        *(run_source.review_root / "native_frames" / run_source.run / f"frame_{index:03d}.jpg"
                          for index in range(count))]
        rows = staged_receipt.get("files") or []
        require(staged_receipt.get("source_manifest_sha256") == manifest_sha
                and staged_receipt.get("source_review_root") == str(run_source.review_root)
                and staged_receipt.get("asset_sha256") == training._canonical_hash(rows)
                and len(rows) == len(local_files), "Native staging identity changed")
        for original, local, row in zip(source_files, local_files, rows, strict=True):
            require(local.is_file() and not local.is_symlink() and original.is_file()
                    and row.get("name") == local.name and row.get("bytes") == local.stat().st_size
                    and sha256_file(original) == row.get("sha256")
                    and sha256_file(local) == row.get("sha256"),
                    f"Original or staged native bytes changed: {local}")
        staged = training.StagedRun(run_source.run, run_source.dataset, staged_root,
            local_files[0], local_files[1], tuple(local_files[2:]), staged_receipt["asset_sha256"])
        cache_key = training._feature_cache_key(staged, run_regions,
            pixels_per_region=int(settings.get("pixels_per_region") or 1024),
            texture_sigma=float(settings.get("texture_sigma") or 2.),
            radiometry_profile=radiometry, feature_contract=contract)
        sample = root / "features" / staged.run / staged.asset_sha256[:16] / cache_key[:20] / "samples.npz"
        metadata_path = sample.with_name("metadata.json")
        require(sample.is_file() and metadata_path.is_file() and not sample.is_symlink()
                and not metadata_path.is_symlink(), f"Incomplete expected extraction task: {sample}")
        metadata = read(metadata_path)
        require(metadata.get("cache_key") == cache_key
                and metadata.get("asset_sha256") == staged.asset_sha256
                and _path(metadata.get("features_path", "")) == sample
                and metadata.get("region_count") == len(run_regions)
                and metadata.get("feature_columns") == feature_columns
                and metadata.get("features_sha256") == sha256_file(sample),
                f"Extraction key, labels, or sample bytes changed: {sample}")
        with np.load(sample, allow_pickle=False) as archive:
            values, owners = archive["features"], archive["region_indices"]
            require(values.dtype == np.dtype("float32") and values.ndim == 2
                    and values.shape == (metadata.get("pixel_rows"), feature_columns)
                    and tuple(archive["feature_names"]) == contract.feature_names
                    and owners.dtype == np.dtype("int32") and owners.shape == (len(values),)
                    and (not len(owners) or (owners.min() >= 0 and owners.max() < len(run_regions)))
                    and np.isfinite(values).all(), f"Extracted array shape or ownership changed: {sample}")
            owner_counts = np.bincount(owners, minlength=len(run_regions))
        expected_frames = []
        for frame in sorted({region.frame_index for region in run_regions}):
            indices = [i for i, region in enumerate(run_regions) if region.frame_index == frame]
            expected_frames.append(dict(frame_index=frame, regions=len(indices),
                                        pixels=int(owner_counts[indices].sum())))
        require(metadata.get("frames") == expected_frames,
                f"Extracted frame coverage differs from frozen labels: {sample}")
        reports.append(metadata)
        ordered_regions.extend(run_regions)
        staged_receipts.append(dict(path=receipt_path.relative_to(root).as_posix(),
                                    sha256=sha256_file(receipt_path)))
    require(groups and len(groups) == inventory.get("native_source_revisions"),
            "Extraction inventory native revision count differs from frozen labels")
    return reports, dict(stage="extraction_complete_fit_not_required", fit_completion_required=False,
        expected_tasks=len(groups), ordered_regions_sha256=_digest([asdict(row) for row in ordered_regions]),
        staged_receipts=staged_receipts)


def inspect_feature_cache(cache_root: Path, *, arm: str, frozen_request_sha256: str,
                          materialization_receipt_sha256: str,
                          parent_summary_sha256: str,
                          annotation_contract_sha256: str, annotation_ids: int,
                          feature_columns: int, evidence_format: str | None = None) -> dict:
    """SHA-bind completed extraction from the same immutable two-arm cohort.

    Only extracted samples and their metadata are shared. The trainer validates
    staged native assets against their original source again on every cache hit.
    """
    root = _path(cache_root)
    require(arm in ARMS and root.is_dir() and not root.is_symlink()
            and root.parent.name == "cache", "Select a completed same-arm feature cache")
    source_path = root.parent.parent / "models" / root.name / "training_request.json"
    source_binding_path = source_path.parent / "two_arm_source_binding.json"
    require(source_path.is_file() and source_binding_path.is_file(),
            "Feature cache has no frozen two-arm source request")
    source, source_binding = read(source_path), read(source_binding_path)
    require(_path(source["output_root"]) == source_path.parent
            and _path(source["cache_root"]) == root
            and source_binding.get("prepared_request_sha256") == sha256_file(source_path)
            and source_binding.get("arm") == arm
            and source_binding.get("frozen_request_sha256") == frozen_request_sha256
            and source_binding.get("materialization_receipt_sha256")
                == materialization_receipt_sha256
            and source_binding.get("parent_summary_sha256") == parent_summary_sha256
            and source_binding.get("annotation_binding", {}).get("annotation_contract_sha256")
                == annotation_contract_sha256
            and source_binding.get("annotation_binding", {}).get("annotation_ids")
                == annotation_ids
            and not source["settings"].get("feature_cache_root"),
            "Feature cache belongs to a different two-arm source or labels")
    staging = root / "result_staging"
    summary_path = staging / "summary.json"
    inventory_path = staging / "training_inventory.json"
    require(inventory_path.is_file(), "Feature cache has no completed extraction inventory")
    inventory = read(inventory_path)
    evidence_format = evidence_format or (FEATURE_CACHE_FORMAT if summary_path.is_file() else EXTRACTION_CACHE_FORMAT)
    require(evidence_format in {FEATURE_CACHE_FORMAT, EXTRACTION_CACHE_FORMAT}, "Unknown feature-cache evidence format")
    completion = None
    if evidence_format == FEATURE_CACHE_FORMAT:
        require(summary_path.is_file(), "Feature cache has no completed fit summary")
        summary = read(summary_path)
        require(summary.get("campaign_id") == source["model_id"], "Feature cache campaign identity changed")
        reports = summary.get("feature_extraction") or []
    else:
        reports, completion = _extraction_completion(root, source, inventory, feature_columns)
    require(inventory.get("regions") == annotation_ids
            and reports and inventory.get("pixel_rows") == sum(
                row.get("pixel_rows", 0) for row in reports),
            "Feature cache inventory differs from the frozen source request")
    files = []
    reported_paths = set()
    for report in reports:
        sample = _path(report["features_path"])
        metadata_path = sample.with_name("metadata.json")
        require(sample.is_relative_to(root / "features")
                and sample.name == "samples.npz" and sample not in reported_paths
                and sample.is_file() and metadata_path.is_file()
                and not sample.is_symlink() and not metadata_path.is_symlink(),
                "Feature cache sample or metadata is missing or ambiguous")
        reported_paths.add(sample)
        metadata = read(metadata_path)
        require(metadata == report and metadata.get("feature_columns") == feature_columns
                and metadata.get("features_sha256") == sha256_file(sample)
                and metadata.get("cache_key", "").startswith(sample.parent.name),
                f"Feature cache metadata or sample changed: {sample}")
        for path in (sample, metadata_path):
            files.append(dict(path=path.relative_to(root).as_posix(),
                              bytes=path.stat().st_size, sha256=sha256_file(path)))
    require(reported_paths == set((root / "features").rglob("samples.npz"))
            and len(reported_paths) == len(list((root / "features").rglob("metadata.json")))
            and sum(row["region_count"] for row in reports) == annotation_ids,
            "Feature cache is incomplete for the frozen annotations")
    result = dict(format=evidence_format, root=str(root),
                source_request_path=str(source_path),
                source_request_sha256=sha256_file(source_path),
                source_binding_sha256=sha256_file(source_binding_path),
                inventory_sha256=sha256_file(inventory_path),
                feature_samples=len(reported_paths), pixel_rows=inventory["pixel_rows"],
                feature_columns=feature_columns, annotation_ids=annotation_ids,
                files=sorted(files, key=lambda row: row["path"]))
    if completion is None:
        result["summary_sha256"] = sha256_file(summary_path)
    else:
        result["extraction_completion"] = completion
    return result


def verify_feature_cache_binding(request: dict, binding: dict) -> dict | None:
    """Reject cache or request drift before a new two-arm job is created."""
    feature_cache = binding.get("feature_cache")
    if feature_cache is None:
        require(not request["settings"].get("feature_cache_root"),
                "Prepared request has an unbound feature cache")
        return None
    require(feature_cache.get("format") in {FEATURE_CACHE_FORMAT, EXTRACTION_CACHE_FORMAT}
            and request["settings"].get("feature_cache_root") == feature_cache.get("root")
            and request["settings"].get("disable_resume_completed_heads") is True
            and request["settings"]["two_arm_photometry"].get("feature_cache_sha256")
                == _digest(feature_cache),
            "Prepared feature-cache binding changed")
    actual_cache = inspect_feature_cache(_path(feature_cache["root"]), arm=binding["arm"],
        frozen_request_sha256=binding["frozen_request_sha256"],
        materialization_receipt_sha256=binding["materialization_receipt_sha256"],
        parent_summary_sha256=binding["parent_summary_sha256"],
        annotation_contract_sha256=binding["annotation_binding"]["annotation_contract_sha256"],
        annotation_ids=binding["annotation_binding"]["annotation_ids"],
        feature_columns=extraction_feature_columns(request["settings"]),
        evidence_format=feature_cache["format"])
    require(actual_cache == feature_cache,
            "Frozen feature cache changed before the two-arm fit")
    source_settings = deepcopy(read(_path(feature_cache["source_request_path"]))["settings"])
    new_settings = deepcopy(request["settings"])
    new_settings.pop("feature_cache_root")
    new_settings.pop("disable_resume_completed_heads")
    new_settings["two_arm_photometry"].pop("feature_cache_sha256")
    # Resource selection is frozen separately from the optical feature identity.
    # Keep the comparison exact for every scientific/photometric setting.
    for settings in (source_settings, new_settings):
        for key in ("stage_workers", "extraction_workers", "opencv_threads", "optical_extraction_resources"):
            settings.pop(key, None)
    require(new_settings == source_settings,
            "Feature-cache retry changed optical settings or photometry")
    return feature_cache


def _drop_inherited_cache_for_sequential(settings: dict, arm: str) -> None:
    """A prior within-run request cannot lend samples to a new photometry arm."""
    if arm == "sequential_global":
        settings.pop("feature_cache_root", None)
        settings.pop("disable_resume_completed_heads", None)


def _unchanged(old: dict, new: dict, volatile: set[str]) -> bool:
    return {key: value for key, value in old.items() if key not in volatile} == {
        key: value for key, value in new.items() if key not in volatile}


def _verify_source_photometry_revision(source: dict, target: dict, *,
                                       source_root: Path, source_sha256: str,
                                       arm: str) -> dict | None:
    """Accept the one known historical v4 source only with its exact lineage."""
    old_id = source.get("photometry_revision_id")
    lineage = (target.get("two_arm_photometry") or {}).get("source_photometry_revision")
    if old_id is None:
        require(target.get("photometry_revision_id") is None and lineage is None,
                "Unexpected photometry revision appeared in two-arm source")
        return None
    if old_id == AC47_ARM_REVISION_IDS["within_run"] and arm == "sequential_global":
        prior_path = source_root.parent / "materialization_receipt.json"
        require(source_sha256 == AC47_PRIOR_WITHIN_SOURCE_SHA256
                and source.get("run", "").casefold() == "ac47"
                and sha256_file(prior_path) == AC47_PRIOR_WITHIN_RECEIPT_SHA256,
                "AC47 sequential source is not the frozen prior within-run revision")
        prior = read(prior_path)
        historical = (source.get("two_arm_photometry") or {}).get("source_photometry_revision")
        require(prior.get("format") == MATERIALIZATION_FORMAT
                and prior.get("arm") == "within_run" and prior.get("run") == "ac47"
                and prior.get("source_manifest_sha256") == AC47_V4_SOURCE_SHA256
                and prior.get("materialized_manifest_sha256") == source_sha256
                and _path(prior["materialized_manifest_path"]) ==
                    source_root / "native_frames" / "ac47" / "manifest.json"
                and prior.get("source_photometry_revision") == historical
                and isinstance(historical, dict)
                and historical.get("id") == AC47_V4_REVISION_ID
                and historical.get("receipt_sha256") == AC47_V4_RECEIPT_SHA256
                and historical.get("target_manifest_sha256") == AC47_V4_SOURCE_SHA256,
                "AC47 prior within-run receipt lost its exact v4 lineage")
        checked_prior = verify_revision(prior, arm="within_run", target_sha256=None)
        require(checked_prior["materialized_manifest_sha256"] == source_sha256
                and checked_prior["source_photometry_revision"]["changed_frame_indices"]
                    == AC47_V4_CHANGED_FRAMES,
                "AC47 prior within-run source failed its original receipt verification")
        require(isinstance(lineage, dict)
                and lineage.get("id") == old_id
                and _path(lineage.get("receipt_path", "")) == prior_path
                and lineage.get("receipt_sha256") == AC47_PRIOR_WITHIN_RECEIPT_SHA256
                and lineage.get("source_manifest_sha256") == AC47_V4_SOURCE_SHA256
                and lineage.get("target_manifest_sha256") == source_sha256
                and lineage.get("source_photometry_revision") == historical
                and target.get("photometry_revision_id") == AC47_ARM_REVISION_IDS[arm]
                and (target.get("two_arm_photometry") or {}).get("arm_photometry_revision_id")
                    == AC47_ARM_REVISION_IDS[arm]
                and (target.get("two_arm_photometry") or {}).get(
                    "arm_colorchecker_algorithm_revision") == "legacy-v2",
                "AC47 sequential photometry hop is stale or incomplete")
        old_environment = dict((source.get("correction_contract") or {}).get("environment") or {})
        current_environment = dict((target.get("correction_contract") or {}).get("environment") or {})
        require(current_environment == old_environment,
                "AC47 sequential correction changed the declared within-run pipeline")
        return dict(source_revision_id=old_id,
                    source_revision_receipt_sha256=AC47_PRIOR_WITHIN_RECEIPT_SHA256,
                    inherited_v4_receipt_sha256=AC47_V4_RECEIPT_SHA256,
                    arm_revision_id=target["photometry_revision_id"],
                    arm_colorchecker_algorithm_revision="legacy-v2",
                    changed_frame_indices=AC47_V4_CHANGED_FRAMES)
    require(source_sha256 == AC47_V4_SOURCE_SHA256
            and source.get("run", "").casefold() == "ac47"
            and old_id == AC47_V4_REVISION_ID,
            "Unrecognized inherited photometry revision")
    receipt_path = source_root / "photometry_revision.json"
    require(isinstance(lineage, dict)
            and target.get("photometry_revision_id") == AC47_ARM_REVISION_IDS[arm]
            and target["photometry_revision_id"] != old_id
            and (target.get("two_arm_photometry") or {}).get("arm_photometry_revision_id")
                == target["photometry_revision_id"]
            and (target.get("two_arm_photometry") or {}).get(
                "arm_colorchecker_algorithm_revision") == "legacy-v2"
            and lineage.get("id") == old_id
            and _path(lineage.get("receipt_path", "")) == receipt_path
            and lineage.get("receipt_sha256") == AC47_V4_RECEIPT_SHA256
            and lineage.get("changed_frame_indices") == AC47_V4_CHANGED_FRAMES
            and lineage.get("source_manifest_sha256") == AC47_V4_PARENT_SHA256
            and lineage.get("target_manifest_sha256") == source_sha256,
            "AC47 two-arm photometry revision lineage is stale or incomplete")
    require(sha256_file(receipt_path) == AC47_V4_RECEIPT_SHA256,
            "Frozen AC47 v4 photometry receipt changed")
    from darsia.fluidflower.tools.phase_training_photometry_revision import verify_revision as verify_v4
    verified = verify_v4(source_root)
    require(verified["receipt_sha256"] == AC47_V4_RECEIPT_SHA256
            and verified["source_manifest_sha256"] == AC47_V4_PARENT_SHA256
            and verified["target_manifest_sha256"] == source_sha256
            and verified["changed_frame_indices"] == AC47_V4_CHANGED_FRAMES,
            "Frozen AC47 v4 correction provenance changed")
    receipt = read(receipt_path)
    require(receipt.get("algorithm_revision") == "neutral-tone-only-v4"
            and receipt.get("target_manifest_sha256") == source_sha256,
            "Frozen AC47 v4 correction algorithm changed")
    old_environment = dict((source.get("correction_contract") or {}).get("environment") or {})
    require(old_environment.pop("FFAC_COLORCHECKER_ALGORITHM_REVISION", None)
            == "neutral-tone-only-v4", "Frozen AC47 v4 environment changed")
    current_environment = dict((target.get("correction_contract") or {}).get("environment") or {})
    require(current_environment.pop("FFAC_COLORCHECKER_ALGORITHM_REVISION", None)
                in (None, "legacy-v2")
            and current_environment == old_environment,
            "AC47 arms differ from the declared current photometry pipeline")
    return dict(source_revision_id=old_id,
                source_revision_receipt_sha256=AC47_V4_RECEIPT_SHA256,
                arm_revision_id=target["photometry_revision_id"],
                arm_colorchecker_algorithm_revision="legacy-v2",
                changed_frame_indices=AC47_V4_CHANGED_FRAMES)


def _verify_prepared_baseline_composite(source: dict, target: dict, row: dict,
                                        *, source_root: Path, arm: str) -> dict | None:
    """Bind the one historic AC61 baseline frame to its three raw acquisitions."""
    source_sha = row["source_manifest_sha256"]
    marker = (target.get("two_arm_photometry") or {}).get("prepared_baseline_composite")
    recorded = row.get("prepared_baseline_composite")
    if source_sha != AC61_COMPOSITE_SOURCE_SHA256:
        require(marker is None and recorded is None,
                "Unexpected prepared-baseline composite in two-arm source")
        return None
    old_frames, new_frames, records = (source.get("frame_sources") or [],
                                       target.get("frame_sources") or [], row.get("frames") or [])
    require(source.get("run", "").casefold() == "ac61"
            and len(old_frames) == len(new_frames) == len(records) == 22
            and isinstance(marker, dict) and marker == recorded
            and marker.get("source_manifest_sha256") == source_sha
            and marker.get("reconstruction") == "calibrated_rig_baseline",
            "AC61 prepared-baseline composite provenance is missing")
    original = old_frames[0]
    prepared = _path(original["source_path"])
    old_frame = source_root / "native_frames" / "ac61" / "frame_000.jpg"
    old_baseline = source_root / "native_references" / "ac61" / "pre_injection.jpg"
    repair_path = source_root.parents[2] / "repair_receipt.json"
    require(original.get("frame_index") == 0
            and original.get("selection_status") == "pre_injection_negative_control"
            and original.get("source_name") == "Pre-injection baseline"
            and original.get("source_sha256") == AC61_COMPOSITE_PREPARED_SHA256
            and source.get("reference", {}).get("corrected_array_sha256")
                == AC61_COMPOSITE_ARRAY_SHA256
            and sha256_file(prepared) == sha256_file(old_frame)
                == sha256_file(old_baseline) == AC61_COMPOSITE_PREPARED_SHA256
            and _path(marker.get("original_pre_injection_path", "")) == prepared
            and marker.get("original_pre_injection_sha256") == AC61_COMPOSITE_PREPARED_SHA256
            and _path(marker.get("historical_repair_receipt_path", "")) == repair_path
            and marker.get("historical_repair_receipt_sha256") == AC61_COMPOSITE_REPAIR_SHA256
            and sha256_file(repair_path) == AC61_COMPOSITE_REPAIR_SHA256
            and marker.get("original_corrected_array_sha256") == AC61_COMPOSITE_ARRAY_SHA256,
            "AC61 historical prepared baseline or repair receipt changed")
    repair = read(repair_path)
    copies = repair.get("unchanged_copies") or []
    require(repair.get("source_hashes", {}).get(str(prepared)) == AC61_COMPOSITE_PREPARED_SHA256
            and any(_path(item.get("source", "")) == prepared
                    and _path(item.get("target", "")) == old_frame
                    and item.get("sha256") == AC61_COMPOSITE_PREPARED_SHA256
                    for item in copies)
            and any(_path(item.get("source", "")) == prepared
                    and _path(item.get("target", "")) == old_baseline
                    and item.get("sha256") == AC61_COMPOSITE_PREPARED_SHA256
                    for item in copies),
            "AC61 historical repair does not bind the prepared baseline frame")
    configured, members = original.get("configured_baseline_members"), marker.get("members")
    require(isinstance(configured, list) and isinstance(members, list)
            and len(configured) == len(members) == len(AC61_COMPOSITE_MEMBER_SHAS)
            and original.get("time_basis") ==
                "mean time of configured baseline members; retained prepared baseline",
            "AC61 raw baseline member inventory changed")
    for expected, old_member, new_member in zip(AC61_COMPOSITE_MEMBER_SHAS,
                                                 configured, members, strict=True):
        path = _path(old_member["path"])
        require(path.name == expected[0] and sha256_file(path) == expected[1]
                and _path(new_member.get("path", "")) == path
                and new_member.get("sha256") == expected[1]
                and new_member.get("elapsed_h") == old_member.get("elapsed_h")
                and new_member.get("timestamp") == old_member.get("timestamp"),
                f"AC61 raw baseline member changed: {expected[0]}")
    require(abs(sum(item["elapsed_h"] for item in configured) / len(configured)
                - original["elapsed_h"]) < 1e-12,
            "AC61 prepared baseline acquisition time changed")
    corrected_sha = marker.get("materialized_corrected_array_sha256")
    require(corrected_sha == new_frames[0].get("corrected_array_sha256")
            == records[0].get("corrected_array_sha256")
            == target.get("reference", {}).get("corrected_array_sha256")
            and (arm != "within_run" or corrected_sha == AC61_COMPOSITE_ARRAY_SHA256),
            "AC61 corrected baseline and frame zero differ")
    return marker


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _verify_sequential_chart_qa(value: object, *, extractor: str,
                                usable_count: int) -> None:
    require(isinstance(value, dict)
            and value.get("extractor_revision") == extractor
            and value.get("usable_baseline_patches") == usable_count
            and all(_finite_number(value.get(key)) and 0 <= value[key] <= 1
                    for key in ("mean_all", "mean_baseline_usable", "max_all"))
            and value["max_all"] >= value["mean_all"],
            "Sequential common-chart QA is incomplete or invalid")


def _verify_sequential_provenance(row: dict, target: dict, *, target_sha256: str,
                                  target_content_sha256: str) -> dict:
    """Bind one accepted baseline-only fit and every frame's chart/range QA."""
    from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import (
        SEQUENTIAL_CHART_EXTRACTOR_REVISION, SEQUENTIAL_QA_REVISION,
    )
    from darsia.fluidflower.tools.phase_common_baseline_photometry import FIT_FORMAT, FitPolicy

    provenance = target.get("sequential_photometry")
    fit = row.get("sequential_fit")
    require(isinstance(target_content_sha256, str) and len(target_content_sha256) == 64
            and isinstance(provenance, dict) and isinstance(fit, dict)
            and provenance.get("within_run_first") is True
            and provenance.get("common_start_second") is True
            and provenance.get("baseline_target_sha256") == target_sha256
            and row.get("baseline_target_sha256") == target_sha256
            and provenance.get("chart_extractor_revision") == row.get("chart_extractor_revision")
                == SEQUENTIAL_CHART_EXTRACTOR_REVISION
            and provenance.get("sequential_qa_revision") == row.get("sequential_qa_revision")
                == SEQUENTIAL_QA_REVISION
            and provenance.get("fixed_run_fit") == fit
            and provenance.get("no_phase_label_or_map_in_target_selection") is True,
            "Sequential baseline target, extractor, or fit provenance is incomplete")
    shape = target.get("analysis_image_shape_hw")
    mask = fit.get("usable_chart_patch_mask")
    require(fit.get("format") == FIT_FORMAT and fit.get("status") == "accepted"
            and fit.get("shape_hw") == shape
            and fit.get("target_sha256") == target_content_sha256
            and fit.get("target_file_sha256") == target_sha256
            and fit.get("chart_extractor_revision") == SEQUENTIAL_CHART_EXTRACTOR_REVISION
            and isinstance(mask, list) and len(mask) == 4
            and all(isinstance(line, list) and len(line) == 6
                    and all(type(item) is bool for item in line) for line in mask),
            "Sequential fixed fit differs from the frozen baseline target")
    usable = sum(item for line in mask for item in line)
    require(fit.get("usable_chart_patches") == usable
            and fit.get("policy") == asdict(FitPolicy()),
            "Sequential fit chart mask or policy changed")
    parameters, qa = fit.get("parameters"), fit.get("qa")
    require(isinstance(parameters, dict) and isinstance(qa, dict)
            and isinstance(parameters.get("gains_linear_rgb"), list)
            and len(parameters["gains_linear_rgb"]) == 3
            and all(_finite_number(gain) and fit["policy"]["gain_min"] <= gain <= fit["policy"]["gain_max"]
                    for gain in parameters["gains_linear_rgb"])
            and _finite_number(parameters.get("shadow_alpha"))
            and fit["policy"]["shadow_alpha_min"] <= parameters["shadow_alpha"] <=
                fit["policy"]["shadow_alpha_max"]
            and _finite_number(parameters.get("shadow_cutoff_srgb"))
            and 0 < parameters["shadow_cutoff_srgb"] < 1
            and qa.get("usable_chart_patches") == usable
            and usable >= fit["policy"]["min_usable_chart_patches"]
            and type(qa.get("sand_cells")) is int
            and qa["sand_cells"] >= fit["policy"]["min_cells"]
            and type(qa.get("sand_sample_pixels")) is int
            and qa["sand_sample_pixels"] >= qa["sand_cells"] * fit["policy"]["min_cell_pixels"]
            and all(isinstance(qa.get(name), dict)
                    and all(_finite_number(qa[name].get(metric)) and 0 <= qa[name][metric] <= 1
                            for metric in ("mean_all", "mean_usable", "max_all"))
                    for name in ("chart_before", "chart_after"))
            and all(_finite_number(qa.get(name)) and 0 <= qa[name]
                    for name in ("sand_blue_log_error_before", "sand_blue_log_error_after",
                                 "median_blue_chromaticity_difference_before"))
            and qa["sand_blue_log_error_after"] <= fit["policy"]["sand_max_log_error_after"]
            and qa.get("facies_mask_used") is True
            and qa.get("domain_mask_used") is True
            and qa.get("interior_mask_used") is False,
            "Sequential fixed fit failed its baseline-only acceptance contract")
    _verify_sequential_chart_qa(row.get("baseline_common_chart_qa"),
        extractor=SEQUENTIAL_CHART_EXTRACTOR_REVISION, usable_count=usable)
    frames = row.get("frames") or []
    materialized = target.get("frame_sources") or []
    require(len(frames) == len(materialized) == target.get("frame_count"),
            "Sequential per-frame QA timeline is incomplete")
    for record, frame in zip(frames, materialized, strict=True):
        index = record["frame_index"]
        input_qa = record.get("sequential_input_range_qa")
        chart_qa = record.get("post_common_chart_qa")
        diagnostics = [entry for entry in frame.get("color_corrections") or []
                       if entry.get("type") == "FixedRunBaselineCorrection"]
        color_rows = [entry for entry in frame.get("color_corrections") or []
                      if entry.get("type") == "ColorCorrection"]
        require(len(diagnostics) == 1 and isinstance(input_qa, dict)
                and len(color_rows) == 1
                and color_rows[0].get("algorithm_revision") == "legacy-v2"
                and input_qa.get("format") == "ffac-fixed-frame-input-range-qa-v1"
                and input_qa.get("shape_hw") == shape
                and input_qa.get("channel_count") == shape[0] * shape[1] * 3
                and input_qa.get("within_tolerance") is True
                and input_qa.get("nonfinite_channels") == 0
                and type(input_qa.get("out_of_range_channels")) is int
                and input_qa["out_of_range_channels"] >= 0
                and _finite_number(input_qa.get("max_overshoot"))
                and input_qa["max_overshoot"] <= fit["policy"]["input_max_overshoot"]
                and _finite_number(input_qa.get("out_of_range_fraction"))
                and input_qa["out_of_range_fraction"] <= fit["policy"]["input_max_overshoot_fraction"]
                and diagnostics[0].get("target_sha256") == target_sha256
                and diagnostics[0].get("fitted_on") == "within-run-corrected-pre-injection-baseline"
                and diagnostics[0].get("reused_without_refitting") is True
                and diagnostics[0].get("input_range_qa") == input_qa
                and diagnostics[0].get("common_chart_qa") == chart_qa,
                f"Sequential fixed correction or input QA changed: {row['run']}/{index}")
        _verify_sequential_chart_qa(chart_qa,
            extractor=SEQUENTIAL_CHART_EXTRACTOR_REVISION, usable_count=usable)
    return dict(target_sha256=target_sha256, target_content_sha256=target_content_sha256,
                chart_extractor_revision=SEQUENTIAL_CHART_EXTRACTOR_REVISION,
                sequential_qa_revision=SEQUENTIAL_QA_REVISION,
                fitted_on_baseline_only=True, frames_with_qa=len(frames))


def verify_revision(row: dict, *, arm: str, target_sha256: str | None,
                    target_content_sha256: str | None = None) -> dict:
    """Prove a corrected review revision has the identical raw/geometry timeline."""
    run = str(row["run"]).casefold()
    source_path = _path(row["source_manifest_path"])
    target_path = _path(row["materialized_manifest_path"])
    source_root = _path(row["source_review_root"])
    target_root = _path(row["materialized_review_root"])
    require(source_path == source_root / "native_frames" / run / "manifest.json",
            f"Source manifest path differs from its review root: {run}")
    require(target_path == target_root / "native_frames" / run / "manifest.json"
            and source_root != target_root, f"Materialized review root is ambiguous: {run}")
    require(sha256_file(source_path) == row["source_manifest_sha256"]
            and sha256_file(target_path) == row["materialized_manifest_sha256"],
            f"Native manifest changed: {run}")
    source, target = read(source_path), read(target_path)
    require(source.get("dataset") == target.get("dataset") == "ffac"
            and str(source.get("run", "")).casefold() == str(target.get("run", "")).casefold() == run,
            f"Native series/run changed: {run}")
    for field in GEOMETRY_FIELDS:
        require(source.get(field) == target.get(field), f"Native {field} changed: {run}")
    provenance = target.get("two_arm_photometry") or {}
    require(provenance.get("format") == MATERIALIZATION_FORMAT
            and provenance.get("arm") == arm
            and _path(provenance.get("source_manifest", "")) == source_path
            and provenance.get("source_manifest_sha256") == row["source_manifest_sha256"]
            and provenance.get("common_target_sha256") ==
                (None if arm == "sequential_global" else target_sha256)
            and provenance.get("same_raw_geometry_and_timestamps") is True
            and provenance.get("no_phase_label_or_map_in_target_selection") is True,
            f"Native two-arm provenance is incomplete: {run}")
    source_photometry_revision = _verify_source_photometry_revision(
        source, target, source_root=source_root,
        source_sha256=row["source_manifest_sha256"], arm=arm)
    correction = target.get("correction_contract") or {}
    require(correction.get("source_revision_manifest_sha256") == row["source_manifest_sha256"],
            f"Corrected image contract is not bound to its source: {run}")
    # The source correction and chart target may differ. Rig registration and
    # spatial correction must remain on the same geometry.
    old_correction = source.get("correction_contract") or {}
    require(correction.get("mass_fit_loaded") is not True,
            f"Mass-fit processing appeared in photometry-only source: {run}")
    for field in ("rig_registration", "shape_correction", "light_correction", "spatial_light_correction"):
        if field in old_correction:
            require(correction.get(field) == old_correction[field],
                    f"Spatial correction {field} changed: {run}")
    old_reference, new_reference = source.get("reference") or {}, target.get("reference") or {}
    require(_unchanged(old_reference, new_reference, REFERENCE_PHOTOMETRY_FIELDS),
            f"Baseline identity or semantics changed: {run}")
    baseline = target_root / "native_references" / run / "pre_injection.jpg"
    original_baseline = source_root / "native_references" / run / "pre_injection.jpg"
    require(sha256_file(baseline) == row["materialized_baseline_sha256"]
            and new_reference.get("materialized_native_sha256") == row["materialized_baseline_sha256"],
            f"Materialized baseline JPEG changed: {run}")
    source_baseline_sha = sha256_file(original_baseline)
    old_frames, new_frames, records = (source.get("frame_sources") or [],
                                       target.get("frame_sources") or [], row.get("frames") or [])
    count = source.get("frame_count")
    require(type(count) is int and count > 0 and len(old_frames) == len(new_frames) == len(records) == count
            and row.get("frame_count") == count, f"Incomplete frame timeline: {run}")
    prepared_baseline_composite = _verify_prepared_baseline_composite(
        source, target, row, source_root=source_root, arm=arm)
    frames = []
    changed_source_photometry_frames = []
    for index, (old, new, record) in enumerate(zip(old_frames, new_frames, records, strict=True)):
        require(old.get("frame_index") == new.get("frame_index") == record.get("frame_index") == index,
                f"Frame order changed: {run}/{index}")
        require(_unchanged(old, new, FRAME_PHOTOMETRY_FIELDS),
                f"Raw frame identity, timing, or selection changed: {run}/{index}")
        registration_receipt = old.get("registration_receipt_path")
        registration_sha = old.get("registration_receipt_sha256")
        require((registration_receipt is None) == (registration_sha is None),
                f"Incomplete frozen registration receipt: {run}/{index}")
        if registration_receipt is not None:
            require(_path(new["registration_receipt_path"]) == _path(registration_receipt)
                    and new.get("registration_receipt_sha256") == registration_sha
                    and sha256_file(_path(registration_receipt)) == registration_sha,
                    f"Frozen registration receipt changed: {run}/{index}")
        raw = _path(old["source_path"])
        raw_sha = sha256_file(raw)
        require(raw_sha == record.get("raw_sha256")
                and (not old.get("source_sha256") or raw_sha == old["source_sha256"])
                and new.get("source_sha256") == raw_sha
                and new.get("source_raw_sha256", raw_sha) == raw_sha
                and _path(record.get("raw_path", "")) == raw,
                f"Raw frame bytes changed: {run}/{index}")
        old_jpeg = sha256_file(source_root / "native_frames" / run / f"frame_{index:03d}.jpg")
        new_jpeg = sha256_file(target_root / "native_frames" / run / f"frame_{index:03d}.jpg")
        require((not old.get("native_jpeg_sha256") or old["native_jpeg_sha256"] == old_jpeg)
                and record.get("original_native_sha256") in (None, old_jpeg)
                and record.get("native_jpeg_sha256") == new_jpeg == new.get("native_jpeg_sha256")
                and new.get("frame_sha256", new_jpeg) == new_jpeg
                and _path(new.get("frame_path", str(target_root / "native_frames" / run /
                    f"frame_{index:03d}.jpg"))) == target_root / "native_frames" / run /
                    f"frame_{index:03d}.jpg"
                and record.get("corrected_array_sha256") == new.get("corrected_array_sha256")
                and record.get("elapsed_h") == old.get("elapsed_h"),
                f"Corrected frame receipt changed: {run}/{index}")
        if source_photometry_revision is not None:
            corrections = new.get("color_corrections") or []
            if arm == "sequential_global":
                color_rows = [item for item in corrections if item.get("type") == "ColorCorrection"]
                valid_correction = (len(color_rows) == 1
                    and color_rows[0].get("algorithm_revision") == "legacy-v2")
            else:
                valid_correction = bool(corrections) and all(
                    item.get("algorithm_revision") == "legacy-v2" for item in corrections)
            require(valid_correction,
                    f"AC47 frame has stale or mixed correction algorithm: {run}/{index}")
            if old.get("corrected_array_sha256") != new.get("corrected_array_sha256"):
                changed_source_photometry_frames.append(index)
        frames.append(dict(index=index, raw_sha256=raw_sha, original_native_sha256=old_jpeg,
                           materialized_native_sha256=new_jpeg, elapsed_h=old["elapsed_h"]))
    if source_photometry_revision is not None and arm == "within_run":
        require(changed_source_photometry_frames == AC47_V4_CHANGED_FRAMES,
                "AC47 within-run correction changed frames outside frozen v4 revision")
    sequential = None
    if arm == "sequential_global":
        sequential = _verify_sequential_provenance(row, target,
            target_sha256=target_sha256, target_content_sha256=target_content_sha256)
    require(sha256_file(source_path) == row["source_manifest_sha256"]
            and sha256_file(target_path) == row["materialized_manifest_sha256"],
            f"Native source changed during verification: {run}")
    return dict(run=run, source_manifest_path=str(source_path),
                source_manifest_sha256=row["source_manifest_sha256"],
                materialized_manifest_path=str(target_path),
                materialized_manifest_sha256=row["materialized_manifest_sha256"],
                source_review_root=str(source_root), materialized_review_root=str(target_root),
                original_baseline_native_sha256=source_baseline_sha,
                materialized_baseline_native_sha256=row["materialized_baseline_sha256"],
                source_photometry_revision=source_photometry_revision,
                prepared_baseline_composite=prepared_baseline_composite,
                sequential_photometry=sequential,
                frames=frames)


def load_materialization(frozen_request_path: Path, receipt_path: Path,
                         expected_sha256: str, arm: str) -> tuple[dict, dict[str, dict]]:
    """Read only a complete, externally SHA-bound materialization receipt."""
    frozen_request_path, receipt_path = _path(frozen_request_path), _path(receipt_path)
    require(arm in ARMS and sha256_file(receipt_path) == expected_sha256,
            "Materialization receipt SHA differs from the selected arm")
    receipt = read(receipt_path)
    require(receipt.get("format") == MATERIALIZATION_FORMAT and receipt.get("arm") == arm
            and receipt.get("complete") is True
            and _path(receipt.get("frozen_request_path", "")) == frozen_request_path
            and receipt.get("frozen_request_sha256") == sha256_file(frozen_request_path),
            "Materialization does not cover the exact frozen request")
    from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests
    expected = source_manifests(frozen_request_path)
    expected_by_sha = {row["manifest_sha256"]: row for row in expected}
    require(len(expected_by_sha) == len(expected)
            and receipt.get("selection", {}).get("policy") ==
                "all-frozen-bundle-native-revisions-v1"
            and receipt["selection"].get("selected_source_manifest_sha256") ==
                sorted(expected_by_sha)
            and receipt.get("source_revisions") == len(expected)
            and len(receipt.get("revisions") or []) == len(expected),
            "Materialization omits or adds frozen native source revisions")
    recovery = receipt.get("recovery")
    if recovery is not None:
        require(recovery.get("format") == "ffac-two-arm-sha-verified-reuse-v1",
                "Unknown materialization recovery contract")
        composition = {item["source_manifest_sha256"]: item
                       for item in recovery.get("composition") or []}
        contracts = {item["source_manifest_sha256"]: item
                     for item in recovery.get("source_correction_contracts") or []}
        reused = recovery.get("reused_source_manifests") or []
        new = recovery.get("new_source_manifests") or []
        require(len(composition) == len(recovery.get("composition") or [])
                and len(contracts) == len(recovery.get("source_correction_contracts") or [])
                and set(composition) == set(contracts) == set(expected_by_sha)
                and len(reused) + len(new) == len(expected_by_sha)
                and set(reused).isdisjoint(new)
                and set(reused) | set(new) == set(expected_by_sha),
                "Recovery composition omits or duplicates a frozen source")
        for row in receipt["revisions"]:
            source_sha = row["source_manifest_sha256"]
            entry, contract = composition[source_sha], contracts[source_sha]
            per_revision = _path(entry["per_revision_receipt_path"])
            require(per_revision == _path(row["materialized_review_root"]).parent /
                        "materialization_receipt.json"
                    and sha256_file(per_revision) == entry["per_revision_receipt_sha256"]
                    and read(per_revision) == row
                    and entry["materialized_manifest_sha256"] ==
                        row["materialized_manifest_sha256"]
                    and entry["origin"] == ("reused" if source_sha in reused else "new"),
                    "Recovery per-revision receipt or origin changed")
            source_path = _path(row["source_manifest_path"])
            require(_path(contract["source_manifest_path"]) == source_path
                    and contract["source_correction_contract_sha256"] ==
                        _digest(read(source_path).get("correction_contract") or {}),
                    "Historical source correction provenance changed")
    target_sha = receipt.get("target_sha256")
    require((arm in ("global", "sequential_global")
             and isinstance(target_sha, str) and len(target_sha) == 64)
            or (arm == "within_run" and target_sha is None),
            "Photometry target does not match the selected arm")
    target_content_sha = None
    if arm == "global":
        from darsia.fluidflower.tools.phase_two_arm_photometry import load_target
        load_target(_path(receipt["target_path"]), expected_sha256=target_sha)
    elif arm == "sequential_global":
        from darsia.fluidflower.tools.ffac_common_start_target import validate_candidate
        target_path = _path(receipt["target_path"])
        require(sha256_file(target_path) == target_sha,
                "Sequential common-start target file changed")
        target = read(target_path)
        validate_candidate(target, verify_source_bytes=True)
        target_content_sha = _digest(target)
    verified: dict[str, dict] = {}
    for row in receipt.get("revisions") or []:
        selected = expected_by_sha.get(row.get("source_manifest_sha256"))
        require(selected is not None
                and str(row.get("run", "")).casefold() == selected["run"]
                and _path(row["source_manifest_path"]) == _path(selected["manifest_path"]),
                "Materialization revision does not match frozen source selection")
        checked = verify_revision(row, arm=arm, target_sha256=target_sha,
                                  target_content_sha256=target_content_sha)
        key = checked["source_manifest_path"].casefold()
        require(key not in verified, "Duplicate source revision in materialization receipt")
        verified[key] = checked
    require(len(verified) == receipt.get("source_revisions") and verified,
            "Materialization revision count differs")
    return receipt, verified


def extend_source_aliases(frozen_request: dict, verified: dict[str, dict]) -> dict[str, dict]:
    """Accept byte-identical historical copies only after checking their JPEGs."""
    by_identity = {(row["source_manifest_sha256"], row["run"]): row
                   for row in verified.values()}
    require(len(by_identity) == len(verified),
            "Materialization has ambiguous run and native manifest identities")
    extended = dict(verified)
    jobs = []
    for pair in _frozen_pairs(frozen_request):
        jobs.extend(read(pair.combined_manifest_path).get("jobs") or [])
    jobs.extend(read(Path(frozen_request["output_root"]) / "review_source" /
                     "postflight_manifest.json").get("jobs") or [])
    for job in jobs:
        if str(job.get("dataset") or "").casefold() != "ffac":
            continue
        root, manifest, _ = optical.native_source(job)
        manifest = _path(manifest)
        key = str(manifest).casefold()
        if key in extended:
            continue
        run = str(job["run"]).casefold()
        digest = sha256_file(manifest)
        primary = by_identity.get((digest, run))
        if primary is None:
            continue  # A source with no active labels need not be materialized.
        require(sha256_file(_path(root) / "native_references" / run / "pre_injection.jpg") ==
                primary["original_baseline_native_sha256"],
                f"Same native manifest SHA has another baseline JPEG: {run}")
        for frame in primary["frames"]:
            jpeg = _path(root) / "native_frames" / run / f"frame_{frame['index']:03d}.jpg"
            require(sha256_file(jpeg) == frame["original_native_sha256"],
                    f"Same native manifest SHA has another corrected JPEG: {run}/{frame['index']}")
        extended[key] = {**primary, "source_manifest_path": str(manifest),
                         "source_review_root": str(_path(root))}
    return extended


def verify_reference_audit(reference_path: Path, expected_sha256: str,
                           materialization_path: Path, materialization_sha256: str,
                           materialization: dict) -> dict:
    """Permit the arm baseline while keeping all other references deferred."""
    reference_path, materialization_path = _path(reference_path), _path(materialization_path)
    require(sha256_file(reference_path) == expected_sha256,
            "Reference-availability sidecar SHA changed")
    audit = read(reference_path)
    require(audit.get("format") == REFERENCE_FORMAT
            and audit.get("arm") == materialization["arm"]
            and _path(audit.get("materialization_receipt_path", "")) == materialization_path
            and audit.get("materialization_receipt_sha256") == materialization_sha256
            and audit.get("source_revisions") == materialization["source_revisions"]
            and audit.get("pre_injection_available") is True
            and audit.get("injection_ends_available") is False
            and audit.get("clearwater_available") is False,
            "Reference availability does not bind this complete photometry arm")
    inventory_root = _path(audit["chart_inventory_root"])
    inventory_path = _path(audit["chart_inventory_receipt_path"])
    require(inventory_path == inventory_root / "inventory.json"
            and sha256_file(inventory_path) == audit["chart_inventory_receipt_sha256"],
            "Frozen chart inventory receipt changed")
    inventory = read(inventory_path)
    require(not inventory.get("failures")
            and inventory.get("sampled_runs") == inventory.get("expected_runs"),
            "Frozen chart inventory is incomplete")
    inventory_rows = {str(item["run"]).casefold(): item
                      for item in inventory["run_receipts"]}
    require(len(inventory_rows) == len(inventory["run_receipts"]),
            "Frozen chart inventory has duplicate runs")
    revisions = {row["source_manifest_sha256"]: row for row in materialization["revisions"]}
    rows = {row["source_manifest_sha256"]: row for row in audit.get("references") or []}
    require(set(revisions) == set(rows) == set(audit.get("source_manifest_sha256") or []),
            "Reference sidecar omits or adds native source revisions")
    for source_sha, row in rows.items():
        revision = revisions[source_sha]
        run = str(revision["run"]).casefold()
        require(str(row.get("run", "")).casefold() == run
                and row.get("arm") == materialization["arm"]
                and row.get("format") == REFERENCE_FORMAT
                and _path(row["source_manifest_path"]) == _path(revision["source_manifest_path"])
                and _path(row["materialized_manifest_path"]) == _path(revision["materialized_manifest_path"])
                and row.get("materialized_manifest_sha256") == revision["materialized_manifest_sha256"]
                and row.get("old_reference_assets_reused") is False
                and row.get("injection_ends", {}).get("available") is False
                and row.get("clearwater", {}).get("available") is False,
                f"Reference availability row differs from materialization: {run}")
        per_revision = _path(row["materialization_receipt_path"])
        require(sha256_file(per_revision) == row["materialization_receipt_sha256"]
                and read(per_revision) == revision,
                f"Per-revision materialization receipt changed: {run}")
        baseline = row.get("baseline") or {}
        raw = _path(baseline["raw_path"])
        chart_path = _path(row["chart_inventory_path"])
        chart_row = inventory_rows.get(run)
        require(chart_row is not None
                and chart_path == inventory_root / "runs" / f"{run}.json"
                and _path(chart_row["path"]) == chart_path
                and sha256_file(chart_path) == row["chart_inventory_sha256"]
                == chart_row["sha256"],
                f"Frozen chart inventory row changed: {run}")
        chart = read(chart_path)
        require(_path(chart["raw_baseline_path"]) == raw
                and chart["raw_baseline_sha256"] == baseline["raw_sha256"],
                f"Raw baseline differs from frozen chart inventory: {run}")
        old_native = _path(baseline["original_native_path"])
        new_native = _path(baseline["materialized_native_path"])
        require(baseline.get("available") is True
                and sha256_file(raw) == baseline["raw_sha256"]
                and old_native == _path(revision["source_review_root"]) /
                    "native_references" / run / "pre_injection.jpg"
                and new_native == _path(revision["materialized_review_root"]) /
                    "native_references" / run / "pre_injection.jpg"
                and sha256_file(old_native) == baseline["original_native_sha256"]
                and sha256_file(new_native) == baseline["materialized_native_sha256"]
                == revision["materialized_baseline_sha256"],
                f"Raw or corrected baseline reference changed: {run}")
        sidecar = _path(row["reference_availability_path"])
        require(sidecar == new_native.parent / "reference_availability.json"
                and sha256_file(sidecar) == row["reference_availability_sha256"]
                and read(sidecar) == {key: value for key, value in row.items()
                                      if key not in {"reference_availability_path",
                                                     "reference_availability_sha256"}},
                f"Per-revision reference availability changed: {run}")
        require({path.name for path in new_native.parent.iterdir()} <=
                {"pre_injection.jpg", "reference_availability.json"},
                f"Unverified injection or clear-water reference asset exists: {run}")
    return dict(path=str(reference_path), sha256=expected_sha256,
                source_revisions=len(rows), pre_injection_available=True,
                injection_ends_available=False, clearwater_available=False)


def bound_job(job: dict, verified: dict[str, dict], *, arm: str,
              receipt_path: Path, receipt_sha256: str) -> dict:
    """Rebind one source only when its actual native manifest is in the receipt."""
    if str(job.get("dataset") or "").casefold() != "ffac":
        return deepcopy(job)
    source_root, source_manifest, _ = optical.native_source(job)
    key = str(_path(source_manifest)).casefold()
    require(key in verified and sha256_file(source_manifest) == verified[key]["source_manifest_sha256"],
            f"Unmaterialized native source: {job.get('run')}")
    row = verified[key]
    require(str(job["run"]).casefold() == row["run"]
            and _path(source_root) == _path(row["source_review_root"]),
            f"Materialized revision maps to a different run/root: {job.get('run')}")
    result = deepcopy(job)
    result["review_output_dir"] = row["materialized_review_root"]
    result["analysis_source_review_root"] = row["materialized_review_root"]
    result["two_arm_photometry"] = dict(format=FORMAT, arm=arm,
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha256,
        source_manifest_sha256=row["source_manifest_sha256"],
        materialized_manifest_sha256=row["materialized_manifest_sha256"],
        raw_geometry_and_time_equivalence_verified=True)
    target_root, target_manifest, _ = optical.native_source(result)
    require(_path(target_root) == _path(row["materialized_review_root"])
            and sha256_file(target_manifest) == row["materialized_manifest_sha256"],
            f"Trainer resolves a different materialized native source: {job.get('run')}")
    return result


def region_records(pairs: list[training.ReviewBundle]) -> tuple[dict[str, dict], list]:
    sources: dict = {}
    _, regions, _ = training._load_sources_and_regions(
        pairs, preserve_native_revisions=True, region_sources=sources)
    logistic.assert_safe_training_regions(regions)
    records = {}
    for region in regions:
        require(region.region_id not in records, f"Duplicate annotation ID: {region.region_id}")
        records[region.region_id] = dict(
            run=region.run, dataset=region.dataset, frame_index=region.frame_index,
            elapsed_h=region.elapsed_h, phase_class=region.phase_class,
            confidence=region.confidence, source=region.source,
            geometry=training._geometry_key(region.run, region.payload),
            payload_sha256=training._canonical_hash(region.payload),
            native_manifest_sha256=sha256_file(sources[region.region_id].asset_manifest_path))
    return records, regions


def verify_region_binding(original: dict[str, dict], rebound: dict[str, dict],
                          verified: dict[str, dict], source_by_id: dict[str, str]) -> dict:
    require(original.keys() == rebound.keys(), "Two-arm binding added or lost annotation IDs")
    image_changes = 0
    for identifier, before in original.items():
        after = rebound[identifier]
        key = source_by_id[identifier].casefold()
        require(key in verified, f"Annotation has no materialized source: {identifier}")
        expected = verified[key]
        require(before["native_manifest_sha256"] == expected["source_manifest_sha256"]
                and after["native_manifest_sha256"] == expected["materialized_manifest_sha256"],
                f"Annotation moved to a different raw source: {identifier}")
        require(_unchanged(before, after, {"native_manifest_sha256"}),
                f"Annotation ID, class, time, geometry or payload changed: {identifier}")
        image_changes += before["native_manifest_sha256"] != after["native_manifest_sha256"]
    return dict(annotation_ids=len(original), rebound_image_revisions=image_changes,
                annotation_contract_sha256=_digest([
                    {k: v for k, v in original[key].items() if k != "native_manifest_sha256"}
                    for key in sorted(original)]),
                all_ids_classes_times_geometry_and_payloads_unchanged=True)


def _frozen_pairs(request: dict) -> list[training.ReviewBundle]:
    pairs = []
    root = _path(request["output_root"])
    for bundle in request["bundles"]:
        paths = []
        for role in ("decisions", "combined_manifest"):
            item = bundle[role]
            path = _path(item["snapshot_path"])
            require(path.is_relative_to(root / "input_snapshot") and sha256_file(path) == item["sha256"],
                    f"Frozen {role} snapshot changed")
            paths.append(path)
        pairs.append(training.ReviewBundle(*paths))
    return pairs


def live_review_delta(before: dict[str, dict], current: dict[str, dict]) -> dict:
    """Compare every source-bound reviewer decision, including same-ID edits."""
    old_ids, new_ids = set(before), set(current)
    stable_fields = ("run", "dataset", "frame_index", "elapsed_h", "phase_class",
                     "confidence", "source", "geometry", "payload_sha256",
                     "native_manifest_sha256")
    changed = [key for key in sorted(old_ids & new_ids)
               if any(before[key][field] != current[key][field] for field in stable_fields)]
    return dict(added_annotation_ids=sorted(new_ids - old_ids),
                removed_annotation_ids=sorted(old_ids - new_ids),
                changed_annotation_ids=changed,
                same_effective_cohort=not (new_ids - old_ids or old_ids - new_ids or changed))


def audit_live_reviews(frozen_request_path: Path, parent: Path, cache: Path) -> dict:
    """Compare current saved reviews with the 22:07 frozen effective cohort."""
    from darsia.fluidflower.phase_workflow import discover_phase_review_sessions

    frozen_request_path, parent, cache = (_path(frozen_request_path), _path(parent), _path(cache))
    frozen_request = read(frozen_request_path)
    before, _ = region_records(_frozen_pairs(frozen_request))
    sessions = discover_phase_review_sessions()
    audit_root = cache / "two-arm-live-review-audits" / uuid.uuid4().hex
    _previous, pairs, _regions, capture = optical.capture_inputs(
        parent, "ffac", audit_root / "capture", sessions,
        preserve_native_revisions=True)
    current, _ = region_records([training.ReviewBundle(Path(a), Path(b)) for a, b in pairs])
    delta = live_review_delta(before, current)
    included = {row["session_path"] for row in capture.get("captured_sources") or []
                if row.get("session_path")}
    latest = max((session for session in sessions if session.get("session_path") in included),
                 key=lambda row: float(row.get("modified_timestamp") or 0), default=None)
    result = dict(format=FORMAT, frozen_request_path=str(frozen_request_path),
        frozen_request_sha256=sha256_file(frozen_request_path),
        checked_utc=datetime.now(timezone.utc).isoformat(),
        frozen_annotation_ids=len(before), current_annotation_ids=len(current),
        added_annotation_ids=delta["added_annotation_ids"],
        removed_annotation_ids=delta["removed_annotation_ids"],
        changed_annotation_ids=delta["changed_annotation_ids"],
        latest_included_review_utc=(latest.get("updated_utc") if latest else None),
        latest_included_review_session=(latest.get("session_path") if latest else None),
        current_saved_review_sessions=len(included),
        same_effective_cohort=delta["same_effective_cohort"],
        capture_root=str(audit_root / "capture"))
    write_new(audit_root / "live_review_audit.json", result)
    return result


def audit_domains(request_path: Path, parent: Path,
                  source_equivalences: list[dict] | None = None) -> dict:
    """Resolve every effective source's physical domain before any optical fit."""
    request_path, parent = _path(request_path), _path(parent)
    request = read(request_path)
    pairs = _frozen_pairs(request)
    _, regions = region_records(pairs)
    continuation = read(request_path.parent / "continuation_preparation.json")
    origins = continuation["annotation_origins"]
    require(sha256_file(_path(origins["path"])) == origins["sha256"],
            "Annotation-origin ledger changed before regional domain preflight")
    equivalences = source_equivalences or []
    audit = dict(annotation_origins=origins,
                 family_models=continuation["family_models"],
                 two_arm_source_binding=dict(source_equivalences=equivalences))
    evidence = dict(parent=str(parent), pairs=[[str(p.decisions_path),
        str(p.combined_manifest_path)] for p in pairs],
        regions=[asdict(region) for region in regions], audit=audit)
    plan, bindings = regional_training.bind_domains(
        evidence, request_path.parent / "review_source")
    labeled = {}
    for group in plan:
        key = str(group["native_path"])
        labeled[key] = labeled.get(key, 0) + len(group["records"])
    require(sum(labeled.values()) == len(regions) and set(labeled) == set(bindings),
            "Regional domain preflight lost a source revision or annotation")
    rows = []
    for path in sorted(bindings):
        native = read(Path(path))
        binding = bindings[path]
        rows.append(dict(run=native["run"], native_manifest_path=path,
            native_manifest_sha256=sha256_file(Path(path)),
            annotation_ids=labeled[path], kind=binding["kind"],
            physical_domain_path=binding["path"],
            physical_domain_sha256=binding["sha256"],
            inherited_source_sha256=binding.get("source_native_sha256"),
            original_rig_facies_preserved=binding.get("original_rig_facies_preserved", False),
            matching_physical_grid_sha256=binding.get("matching_physical_grid_sha256")))
    return dict(format=FORMAT, request_path=str(request_path),
        request_sha256=sha256_file(request_path), parent=str(parent),
        parent_summary_sha256=sha256_file(parent / "summary.json"),
        annotation_origins_sha256=origins["sha256"],
        effective_annotations=len(regions), effective_source_revisions=len(rows),
        source_equivalences=len(equivalences), sources=rows,
        domain_contract_sha256=_digest(rows))


def _copy_physics(source_root: Path, new_root: Path, canonical: list[tuple[dict, dict]],
                  *, receipt_path: Path, receipt_sha: str, arm: str) -> str:
    profile_path = source_root / "phase_optical_regions.json"
    require(profile_path.is_file(), "Expected registered FF_AC regional physical profile")
    for source in physics_files(source_root, phase_runtime.load_physics_profile(source_root)):
        target = new_root / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha256_file(target) == sha256_file(source), "Physical grid copy changed")
    profile = read(new_root / "phase_optical_regions.json")
    require(profile.get("format") == "baseline-relative-optical-regions-v1", "Unexpected FF_AC profile")
    for old_job, new_job in canonical:
        key = f"ffac/{str(old_job['run']).casefold()}"
        entry = profile["runs"].get(key)
        require(entry is not None, f"Missing physical grid for {key}")
        old_sha = sha256_file(optical.native_source(old_job)[1])
        new_sha = sha256_file(optical.native_source(new_job)[1])
        require(entry["native_manifest_sha256"] == old_sha,
                f"Registered context was already on another revision: {key}")
        if arm == "sequential_global" and key == "ffac/ac47":
            inherited = (entry.get("provenance") or {}).get("two_arm_photometry") or {}
            require(isinstance(inherited.get("receipt_path"), str)
                    and bool(inherited["receipt_path"]),
                    "AC47 physical grid lacks its prior within-run receipt")
            prior_path = _path(inherited.get("receipt_path", ""))
            require(old_sha == AC47_PRIOR_WITHIN_SOURCE_SHA256
                    and inherited.get("format") == FORMAT
                    and inherited.get("arm") == "within_run"
                    and inherited.get("source_manifest_sha256") == AC47_V4_SOURCE_SHA256
                    and inherited.get("materialized_manifest_sha256") == old_sha
                    and inherited.get("physical_grid_bytes_preserved") is True
                    and sha256_file(prior_path) == inherited.get("receipt_sha256"),
                    "AC47 physical grid lacks its exact prior within-run provenance")
            prior = read(prior_path)
            prior_rows = [row for row in prior.get("revisions") or []
                          if row.get("source_manifest_sha256") == AC47_V4_SOURCE_SHA256
                          and row.get("materialized_manifest_sha256") == old_sha]
            require(prior.get("format") == MATERIALIZATION_FORMAT
                    and prior.get("arm") == "within_run" and prior.get("complete") is True
                    and len(prior_rows) == 1
                    and _path(prior_rows[0]["materialized_manifest_path"])
                        == _path(optical.native_source(old_job)[1]),
                    "AC47 inherited within-run receipt does not contain the prior source")
            entry.setdefault("provenance", {})["source_two_arm_photometry"] = deepcopy(inherited)
        entry["native_manifest_sha256"] = new_sha
        entry.setdefault("provenance", {})["two_arm_photometry"] = dict(
            format=FORMAT, arm=arm, receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
            source_manifest_sha256=old_sha, materialized_manifest_sha256=new_sha,
            physical_grid_bytes_preserved=True)
    update.write(new_root / "phase_optical_regions.json", profile)
    runtime = phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(new_root))
    inheritance = read(source_root / "physics_inheritance.json")
    inheritance["original_physics_runtime"] = inheritance.get("physics_runtime")
    inheritance["physics_runtime"] = runtime
    inheritance["two_arm_photometry"] = dict(format=FORMAT, arm=arm,
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
        canonical_runs=len(canonical), physical_grid_bytes_preserved=True)
    for row in inheritance.get("copies") or []:
        old_target = _path(row["target"])
        if old_target.is_relative_to(source_root):
            new_target = new_root / old_target.relative_to(source_root)
            row["target"] = str(new_target)
            rebound_sha = sha256_file(new_target)
            if rebound_sha != row["sha256"]:
                row["original_copied_sha256"] = row["sha256"]
                row["two_arm_rebound_after_copy"] = True
                row["sha256"] = rebound_sha
    update.write(new_root / "physics_inheritance.json", inheritance)
    return runtime


def prepare(frozen_request_path: Path, materialization_receipt_path: Path,
            materialization_sha256: str, reference_audit_path: Path,
            reference_audit_sha256: str, parent: Path, library: Path, cache: Path,
            *, arm: str, actor: str, feature_cache_root: Path | None = None,
            cpus: int | None = None, memory_gb: float | None = None,
            extraction_workers: int | None = None) -> dict:
    """Make and verify a fresh request without fitting either model."""
    frozen_request_path, materialization_receipt_path = (_path(frozen_request_path),
                                                          _path(materialization_receipt_path))
    parent, library, cache = (_path(parent), _path(library), _path(cache))
    checked = ac.preflight(parent, library, cache)
    old_request = read(frozen_request_path)
    require(_path(old_request["output_root"]) == frozen_request_path.parent
            and old_request["settings"].get("parent_model") == str(parent),
            "Frozen request is not the selected FF_AC parent continuation")
    ac.verify_optical_request(old_request, checked)
    live = audit_live_reviews(frozen_request_path, parent, cache)
    require(live["same_effective_cohort"],
            "Saved FF_AC reviews changed since the frozen request; prepare a new immutable base request")
    receipt, verified = load_materialization(frozen_request_path,
        materialization_receipt_path, materialization_sha256, arm)
    reference_audit = verify_reference_audit(reference_audit_path,
        reference_audit_sha256, materialization_receipt_path, materialization_sha256, receipt)
    verified = extend_source_aliases(old_request, verified)
    old_pairs = _frozen_pairs(old_request)
    old_records, _ = region_records(old_pairs)
    feature_cache = None
    if feature_cache_root is not None:
        source_cache = _path(feature_cache_root)
        source_binding_path = (source_cache.parent.parent / "models" / source_cache.name /
                               "two_arm_source_binding.json")
        require(source_binding_path.is_file(), "Feature cache has no frozen two-arm binding")
        source_binding = read(source_binding_path)
        feature_cache = inspect_feature_cache(source_cache, arm=arm,
            frozen_request_sha256=sha256_file(frozen_request_path),
            materialization_receipt_sha256=materialization_sha256,
            parent_summary_sha256=checked["parent_summary_sha256"],
            annotation_contract_sha256=source_binding["annotation_binding"]["annotation_contract_sha256"],
            annotation_ids=len(old_records),
            feature_columns=extraction_feature_columns(old_request["settings"]))
    source_by_id = {}
    training._load_sources_and_regions(old_pairs, preserve_native_revisions=True,
                                       region_sources=source_by_id)
    source_paths = {key: str(value.asset_manifest_path.resolve())
                    for key, value in source_by_id.items()}
    required_source_paths = {path.casefold() for path in source_paths.values()}
    binding_root = cache / "two-arm-request-bindings" / uuid.uuid4().hex
    binding_root.mkdir(parents=True, exist_ok=False)
    pairs = []
    rebound_manifest_sha = []
    for index, pair in enumerate(old_pairs):
        manifest = read(pair.combined_manifest_path)
        rebound = deepcopy(manifest)
        rebound_jobs = []
        for job in manifest.get("jobs") or []:
            if str(job.get("dataset") or "").casefold() == "ffac":
                source_manifest = str(_path(optical.native_source(job)[1])).casefold()
                if source_manifest in required_source_paths:
                    rebound_jobs.append(bound_job(job, verified, arm=arm,
                        receipt_path=materialization_receipt_path,
                        receipt_sha256=materialization_sha256))
                    continue
            rebound_jobs.append(deepcopy(job))
        rebound["jobs"] = rebound_jobs
        target = binding_root / f"{index:03d}_combined_manifest.json"
        write_new(target, rebound)
        rebound_manifest_sha.append(sha256_file(target))
        pairs.append((pair.decisions_path, target))
    canonical_path = old_request["output_root"]
    collection = read(Path(canonical_path) / "review_source" / "postflight_manifest.json")
    old_jobs = collection.get("jobs") or []
    require(len(old_jobs) == 43 and len({str(j["run"]).casefold() for j in old_jobs}) == 43,
            "Expected complete 43-run FF_AC Studio timeline")
    new_jobs = [bound_job(job, verified, arm=arm,
        receipt_path=materialization_receipt_path, receipt_sha256=materialization_sha256)
        for job in old_jobs]
    settings = optical.with_extraction_resources(old_request["settings"],
        [verified[path.casefold()]["materialized_manifest_path"] for path in source_paths.values()],
        cpus=cpus, memory_gb=memory_gb, extraction_workers=extraction_workers,
        preserve_legacy_workers=True)
    _drop_inherited_cache_for_sequential(settings, arm)
    settings["two_arm_photometry"] = dict(format=FORMAT, arm=arm,
        frozen_request_sha256=sha256_file(frozen_request_path),
        materialization_receipt_sha256=materialization_sha256,
        reference_availability_receipt_sha256=reference_audit_sha256,
        common_target_sha256=(receipt.get("target_sha256") if arm == "global" else None),
        baseline_target_sha256=(receipt.get("target_sha256")
                                if arm == "sequential_global" else None),
        source_revisions=receipt["source_revisions"], canonical_runs=len(new_jobs),
        exact_raw_geometry_time_and_annotation_binding=True)
    if arm == "sequential_global":
        from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import (
            SEQUENTIAL_CHART_EXTRACTOR_REVISION, SEQUENTIAL_QA_REVISION,
        )
        settings["two_arm_photometry"].update(
            chart_extractor_revision=SEQUENTIAL_CHART_EXTRACTOR_REVISION,
            sequential_qa_revision=SEQUENTIAL_QA_REVISION,
            photometric_qa_status="pending_review; training_blocked")
    if feature_cache is not None:
        settings["feature_cache_root"] = feature_cache["root"]
        settings["disable_resume_completed_heads"] = True
        settings["two_arm_photometry"]["feature_cache_sha256"] = _digest(feature_cache)
    workspace = cache / "studio-two-arm-continuations" / uuid.uuid4().hex
    new_request = prepare_phase_training_request(output_parent=workspace / "models",
        cache_parent=workspace / "cache", bundles=pairs, actor=actor, settings=settings,
        withdrawal_snapshot=old_request["settings"]["review_retractions_snapshot"])
    root = Path(new_request["output_root"])
    new_pairs = _frozen_pairs(new_request)
    new_records, _ = region_records(new_pairs)
    annotation = verify_region_binding(old_records, new_records, verified, source_paths)
    if feature_cache is not None:
        require(feature_cache["annotation_ids"] == annotation["annotation_ids"]
                and source_binding["annotation_binding"]["annotation_contract_sha256"]
                    == annotation["annotation_contract_sha256"],
                "Feature cache labels differ from the new immutable request")
    runtime = _copy_physics(Path(old_request["output_root"]), root,
        list(zip(old_jobs, new_jobs, strict=True)), receipt_path=materialization_receipt_path,
        receipt_sha=materialization_sha256, arm=arm)
    collection["jobs"] = new_jobs
    update.write(root / "review_source" / "postflight_manifest.json", collection)
    continuation = read(Path(old_request["output_root"]) / "continuation_preparation.json")
    continuation["destination"] = str(library / "studio_retraining" / "ffac" / root.name)
    continuation["two_arm_photometry"] = settings["two_arm_photometry"]
    update.write(root / "continuation_preparation.json", continuation)
    equivalences = [dict(source_manifest_path=row["source_manifest_path"],
        source_manifest_sha256=row["source_manifest_sha256"],
        materialized_manifest_path=row["materialized_manifest_path"],
        materialized_manifest_sha256=row["materialized_manifest_sha256"])
        for row in receipt["revisions"]]
    domain_preflight = audit_domains(root / "training_request.json", parent, equivalences)
    require(domain_preflight["effective_annotations"] == annotation["annotation_ids"],
            "Regional physical-domain preflight differs from frozen optical labels")
    write_new(root / "regional_domain_preflight.json", domain_preflight)
    binding = dict(format=FORMAT, arm=arm, created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_request_path=str(frozen_request_path),
        frozen_request_sha256=sha256_file(frozen_request_path),
        materialization_receipt_path=str(materialization_receipt_path),
        materialization_receipt_sha256=materialization_sha256,
        reference_audit=reference_audit,
        prepared_request_path=str(root / "training_request.json"),
        prepared_request_sha256=sha256_file(root / "training_request.json"),
        parent=str(parent), parent_summary_sha256=checked["parent_summary_sha256"],
        physics_runtime=runtime, source_revisions=receipt["source_revisions"], canonical_runs=len(new_jobs),
        rebound_manifest_sha256=rebound_manifest_sha, annotation_binding=annotation,
        regional_domain_preflight_sha256=sha256_file(root / "regional_domain_preflight.json"),
        regional_domain_contract_sha256=domain_preflight["domain_contract_sha256"],
        original_frozen_request_and_source_documents_unchanged=True,
        studio_published=False)
    if feature_cache is not None:
        binding["feature_cache"] = feature_cache
    write_new(root / "two_arm_source_binding.json", binding)
    return binding


def verify_prepared(request_path: Path) -> tuple[dict, dict]:
    request_path = _path(request_path)
    binding = read(request_path.parent / "two_arm_source_binding.json")
    require(binding.get("format") == FORMAT
            and _path(binding["prepared_request_path"]) == request_path
            and sha256_file(request_path) == binding["prepared_request_sha256"],
            "Prepared two-arm request changed")
    receipt, verified = load_materialization(
        _path(binding["frozen_request_path"]), _path(binding["materialization_receipt_path"]),
        binding["materialization_receipt_sha256"], binding["arm"])
    require(verify_reference_audit(_path(binding["reference_audit"]["path"]),
        binding["reference_audit"]["sha256"],
        _path(binding["materialization_receipt_path"]),
        binding["materialization_receipt_sha256"], receipt) == binding["reference_audit"],
        "Prepared reference availability changed")
    request = read(request_path)
    old = read(_path(binding["frozen_request_path"]))
    verified = extend_source_aliases(old, verified)
    require(request["settings"].get("two_arm_photometry", {}).get("materialization_receipt_sha256")
            == binding["materialization_receipt_sha256"]
            and sha256_file(_path(binding["frozen_request_path"])) == binding["frozen_request_sha256"]
            and sha256_file(_path(binding["parent"]) / "summary.json") == binding["parent_summary_sha256"],
            "Prepared request provenance changed")
    prepared_photometry = request["settings"]["two_arm_photometry"]
    if binding["arm"] == "sequential_global":
        from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import (
            SEQUENTIAL_CHART_EXTRACTOR_REVISION, SEQUENTIAL_QA_REVISION,
        )
        require(prepared_photometry.get("common_target_sha256") is None
                and prepared_photometry.get("baseline_target_sha256") == receipt["target_sha256"]
                and prepared_photometry.get("chart_extractor_revision")
                    == SEQUENTIAL_CHART_EXTRACTOR_REVISION
                and prepared_photometry.get("sequential_qa_revision") == SEQUENTIAL_QA_REVISION
                and prepared_photometry.get("photometric_qa_status")
                    == "pending_review; training_blocked",
                "Prepared sequential request lost its target or QA gate")
    verify_feature_cache_binding(request, binding)
    old_pairs, new_pairs = _frozen_pairs(old), _frozen_pairs(request)
    before, _ = region_records(old_pairs)
    after, _ = region_records(new_pairs)
    sources = {}
    training._load_sources_and_regions(old_pairs, preserve_native_revisions=True, region_sources=sources)
    source_paths = {key: str(value.asset_manifest_path.resolve()) for key, value in sources.items()}
    require(verify_region_binding(before, after, verified, source_paths) == binding["annotation_binding"],
            "Prepared request annotation binding changed")
    old_collection = read(Path(old["output_root"]) / "review_source" / "postflight_manifest.json")
    new_collection = read(request_path.parent / "review_source" / "postflight_manifest.json")
    require(len(old_collection["jobs"]) == len(new_collection["jobs"]) == binding["canonical_runs"],
            "Prepared Studio timeline count changed")
    for old_job, new_job in zip(old_collection["jobs"], new_collection["jobs"], strict=True):
        require(bound_job(old_job, verified, arm=binding["arm"],
            receipt_path=_path(binding["materialization_receipt_path"]),
            receipt_sha256=binding["materialization_receipt_sha256"]) == new_job,
            f"Prepared Studio timeline changed: {old_job['run']}")
    profile = phase_runtime.load_physics_profile(request_path.parent)
    require(profile is not None and receipt["source_revisions"] == binding["source_revisions"],
            "Prepared physical profile or source revision set changed")
    for key, entry in profile["runs"].items():
        context = (request_path.parent / entry["path"]).resolve()
        require(context.is_relative_to(request_path.parent) and
                sha256_file(context) == entry["sha256"],
                f"Prepared physical grid changed: {key}")
    require(phase_runtime.physics_runtime_identity(profile) == binding["physics_runtime"],
            "Prepared physical runtime or source revision set changed")
    domain_preflight_path = request_path.parent / "regional_domain_preflight.json"
    require(sha256_file(domain_preflight_path) == binding["regional_domain_preflight_sha256"]
            and read(domain_preflight_path)["domain_contract_sha256"]
                == binding["regional_domain_contract_sha256"],
            "Prepared regional physical-domain preflight changed")
    request["request_path"] = str(request_path)
    request["frozen_pairs"] = [(pair.decisions_path, pair.combined_manifest_path) for pair in new_pairs]
    return request, binding


def verify_pair(first_request: Path, second_request: Path) -> dict:
    """Require both runnable arms to use one raw-source and label cohort."""
    _, first = verify_prepared(first_request)
    _, second = verify_prepared(second_request)
    require({first["arm"], second["arm"]} == set(PAIRED_ARMS),
            "Paired requests must contain one within-run and one global arm")
    for key in ("frozen_request_sha256", "parent_summary_sha256", "canonical_runs"):
        require(first[key] == second[key], f"Paired arms differ in {key}")
    require(first["annotation_binding"]["annotation_contract_sha256"] ==
            second["annotation_binding"]["annotation_contract_sha256"]
            and first["annotation_binding"]["annotation_ids"] ==
            second["annotation_binding"]["annotation_ids"],
            "Paired arms differ in effective annotation IDs or payloads")
    source_sets = []
    for binding in (first, second):
        receipt = read(_path(binding["materialization_receipt_path"]))
        source_sets.append({(str(_path(row["source_manifest_path"])).casefold(),
                             row["source_manifest_sha256"], str(row["run"]).casefold())
                            for row in receipt["revisions"]})
    require(source_sets[0] == source_sets[1],
            "Paired arms did not materialize the same native source revision set")
    return dict(format=FORMAT, arms=sorted(PAIRED_ARMS),
                source_revisions=len(source_sets[0]), canonical_runs=first["canonical_runs"],
                annotation_ids=first["annotation_binding"]["annotation_ids"],
                annotation_contract_sha256=first["annotation_binding"]["annotation_contract_sha256"],
                frozen_request_sha256=first["frozen_request_sha256"],
                source_sets_identical=True, prepared_requests=[str(_path(first_request)),
                    str(_path(second_request))])


def execute(request_path: Path, job: Path, *, backend: str = "auto",
            cpus: int | None = None, memory_gb: float | None = None) -> dict:
    """Fit fresh optical heads, regional layer, and native Studio maps serially."""
    request, binding = verify_prepared(request_path)
    require(binding["arm"] != "sequential_global",
            "Sequential common-start arm requires reviewed photometric QA before training")
    root = Path(request["output_root"])
    job = _path(job)
    parent = _path(binding["parent"])
    materialization = read(_path(binding["materialization_receipt_path"]))
    equivalents = [dict(source_manifest_path=row["source_manifest_path"],
        source_manifest_sha256=row["source_manifest_sha256"],
        materialized_manifest_path=row["materialized_manifest_path"],
        materialized_manifest_sha256=row["materialized_manifest_sha256"])
        for row in materialization["revisions"]]
    fresh_domains = audit_domains(request_path, parent, equivalents)
    require(fresh_domains["domain_contract_sha256"] ==
            binding["regional_domain_contract_sha256"],
            "Regional physical domains changed since preflight; refusing optical fit")
    library = parent.parent
    require(not job.exists() and not job.is_relative_to(library)
            and not job.is_relative_to(root) and not job.is_relative_to(parent),
            "Choose a fresh detached FF_AC results job")
    job.mkdir(parents=True, exist_ok=False)
    update.write(job / "two_arm_source_binding.json", binding)
    update.write(job / "optical_prepared.json", dict(request_path=str(_path(request_path)),
        request_sha256=binding["prepared_request_sha256"],
        annotation_binding=binding["annotation_binding"],
        physics_runtime=binding["physics_runtime"]))
    optical.fit(request)
    if binding.get("feature_cache") is not None:
        fitted_summary = read(root / "summary.json")
        extraction = fitted_summary.get("feature_extraction") or []
        require(len(extraction) == binding["feature_cache"]["feature_samples"]
                and all(row.get("cache_hit") is True for row in extraction),
                "Two-arm retry did not reuse every SHA-bound feature sample")
    convergence = series.verify_optical_selected_convergence(root)
    update.write(job / "optical_convergence.json", convergence)
    fitted = verify_fitted(root, allow_nonconverged=True)
    require(fitted["series"] == "ffac" and fitted["adapters"] == ["generic"]
            and not fitted["nonconverged_heads"]
            and fitted["physics_runtime"] == binding["physics_runtime"],
            "Two-arm optical fit or inherited physics failed verification")
    hidden = job / "optical_parent" / root.name
    publish_fitted(root, hidden, Path(request["cache_root"]).parent / "publication-stage",
                   allow_nonconverged=False)
    require(not (hidden / "validation").exists(), "Optical intermediate entered Studio")
    regional_args = regional.prepare_launch(hidden, library, Path(request["cache_root"]).parent,
        "ffac", request["created_by"], backend=backend, cpus=cpus, memory_gb=memory_gb)
    regional_args.output = job / "regional_candidate" / regional_args.output.name
    regional_args.job = job / "regional_run"
    regional_args.bottom_gas_guard = True
    regional_args.region_edge_guard = True
    regional_args.job.mkdir(parents=True, exist_ok=False)
    update.write(regional_args.job / "launch_request.json", dict(format=FORMAT,
        parent=str(hidden), output=str(regional_args.output), job=str(regional_args.job),
        bottom_gas_guard=True, region_edge_guard=True, review_candidate_only=True,
        studio_published=False))
    hidden_request = read(hidden / "training_request.json")
    pairs = _frozen_pairs(hidden_request)
    _, regions = region_records(pairs)
    require(len(regions) == binding["annotation_binding"]["annotation_ids"],
            "Regional evidence count differs from exact optical labels")
    audit = dict(format=FORMAT, family_models=read(root / "continuation_preparation.json")["family_models"],
        annotation_origins=read(root / "continuation_preparation.json")["annotation_origins"],
        two_arm_source_binding=dict(receipt_sha256=binding["materialization_receipt_sha256"],
            annotation_contract_sha256=binding["annotation_binding"]["annotation_contract_sha256"],
            source_equivalences=equivalents),
        original_documents_modified=False)
    evidence = dict(parent=str(hidden), pairs=[[str(pair.decisions_path),
        str(pair.combined_manifest_path)] for pair in pairs],
        regions=[asdict(region) for region in regions], audit=audit)
    update.write(regional_args.job / "evidence.json", evidence)
    update.write(regional_args.job / "supervision_policy.json", dict(
        policy="missed-aqueous-is-dense-reviewer-clarification-20260920",
        reviewer_statement="The whole marked area is aqueous CO2",
        previous_interpretation="aqueous_present_region",
        current_interpretation="aqueous",
        corrected_regions=[region.region_id for region in regions if
            region.source == "manual" and
            region.payload.get("category") == "missed_visible_aqueous_co2"],
        original_documents_modified=False, mixed_phase_classes_unchanged=True))
    update.write(regional_args.job / "snapshot_parity.json", dict(format=FORMAT,
        optical_annotation_contract_sha256=binding["annotation_binding"]["annotation_contract_sha256"],
        regional_annotation_contract_sha256=_digest([
            {k: v for k, v in record.items() if k != "native_manifest_sha256"}
            for _, record in sorted(region_records(pairs)[0].items())]),
        regions=len(regions), same_frozen_pairs=True))
    regional_args.status_callback = None
    update.run(regional_args)
    candidate = Path(regional_args.output)
    result = dict(format=FORMAT, completed_utc=datetime.now(timezone.utc).isoformat(),
        arm=binding["arm"], original_parent=str(parent), optical_parent=str(hidden),
        candidate_model=str(candidate), optical_heads_refitted=True,
        regional_layer_refitted=True, native_studio_maps_built=True,
        review_candidate_only=True, studio_published=False, qa_pending=True,
        chart_residual_flags_are_qa_warnings=True,
        source_binding_sha256=sha256_file(root / "two_arm_source_binding.json"),
        original_parent_unchanged=sha256_file(parent / "summary.json") == binding["parent_summary_sha256"])
    require(result["original_parent_unchanged"], "Original parent changed during two-arm fit")
    update.write(job / "candidate_qa_state.json", dict(candidate_model=str(candidate),
        review_candidate_only=True, studio_published=False,
        qa_required_before_publication=["source provenance", "ColorChecker chart residuals",
            "optical and regional convergence", "native visual review"]))
    update.write(job / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Freeze and verify a new two-arm request")
    mode.add_argument("--verify", action="store_true", help="Verify a prepared request read-only")
    mode.add_argument("--audit-live", action="store_true", help="Compare latest saved reviews with frozen request")
    mode.add_argument("--audit-domains", action="store_true", help="Resolve all effective physical domains before training")
    mode.add_argument("--verify-pair", action="store_true", help="Verify identical raw sources and labels across arms")
    mode.add_argument("--execute", action="store_true", help="Fit both layers and build detached Studio maps")
    parser.add_argument("--frozen-request", type=Path)
    parser.add_argument("--materialization-receipt", type=Path)
    parser.add_argument("--materialization-sha256")
    parser.add_argument("--reference-audit-receipt", type=Path)
    parser.add_argument("--reference-audit-sha256")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--feature-cache-root", type=Path,
                        help="SHA-verified completed within-run extraction cache for a new request")
    parser.add_argument("--actor", default="Olav Folkvord")
    parser.add_argument("--prepared-request", type=Path)
    parser.add_argument("--paired-prepared-request", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--extraction-workers", type=int,
                        help="Explicit process ceiling for --prepare; does not modify an existing request")
    args = parser.parse_args(argv)
    if args.prepare:
        needed = (args.frozen_request, args.materialization_receipt,
                  args.materialization_sha256, args.reference_audit_receipt,
                  args.reference_audit_sha256, args.parent, args.library, args.cache, args.arm)
        if not all(needed):
            parser.error("--prepare requires frozen request, materialization and reference receipts with SHAs, parent, library, cache, arm")
        result = prepare(args.frozen_request, args.materialization_receipt,
            args.materialization_sha256, args.reference_audit_receipt,
            args.reference_audit_sha256, args.parent, args.library, args.cache,
            arm=args.arm, actor=args.actor, feature_cache_root=args.feature_cache_root,
            cpus=args.cpus, memory_gb=args.memory_gb, extraction_workers=args.extraction_workers)
    elif args.audit_live:
        if args.frozen_request is None or args.parent is None or args.cache is None:
            parser.error("--audit-live requires --frozen-request, --parent and --cache")
        result = audit_live_reviews(args.frozen_request, args.parent, args.cache)
    elif args.audit_domains:
        if args.frozen_request is None or args.parent is None:
            parser.error("--audit-domains requires --frozen-request and --parent")
        result = audit_domains(args.frozen_request, args.parent)
    elif args.verify_pair:
        if args.prepared_request is None or args.paired_prepared_request is None:
            parser.error("--verify-pair requires both prepared requests")
        result = verify_pair(args.prepared_request, args.paired_prepared_request)
    elif args.verify:
        if args.prepared_request is None:
            parser.error("--verify requires --prepared-request")
        _request, result = verify_prepared(args.prepared_request)
    else:
        if args.extraction_workers is not None:
            parser.error("--extraction-workers applies only to --prepare; prepare a new immutable request")
        if args.prepared_request is None or args.job is None:
            parser.error("--execute requires --prepared-request and fresh --job")
        result = execute(args.prepared_request, args.job, backend=args.backend,
                         cpus=args.cpus, memory_gb=args.memory_gb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
