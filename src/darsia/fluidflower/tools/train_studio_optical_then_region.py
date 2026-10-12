"""Detached FF_AC optical refit followed by a regional refit and Studio maps.

The optical model is first fitted in scratch and copied to the series results
folder without validation. The ordinary regional continuation then uses that
fitted optical model as its parent. The combined QA candidate also remains in
the results folder until it passes review. This command never applies an old
regional fit to new scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

import darsia
from darsia.fluidflower import phase_features, phase_runtime, phase_review_retractions
from darsia.fluidflower.phase_training import logistic, trainer as training
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import series_path
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools import train_studio_series_update as regional
from darsia.fluidflower.tools import train_studio_region_update as update
from darsia.fluidflower.tools.publish_fitted_phase_candidate import fitted_files, publish_fitted, verify_fitted


FEATURE_VERSION = phase_features.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION
HEADS = ("co2_presence", "gas_within_co2")
FORMAT = "studio-optical-then-regional-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def effective_regions(root):
    request = read(Path(root) / "training_request.json")
    pairs = [training.ReviewBundle(
        Path(bundle["decisions"]["snapshot_path"]),
        Path(bundle["combined_manifest"]["snapshot_path"]))
        for bundle in request["bundles"]]
    _, regions, _ = training._load_sources_and_regions(pairs,
        preserve_native_revisions=bool(request["settings"].get("preserve_native_revisions")))
    logistic.assert_safe_training_regions(regions)
    return regions


def frozen_region_records(regions, source_by_id):
    """Keep each label's meaning and original image revision independently."""
    hashes = {}
    resolved_paths = {}

    def resolved(path):
        if path not in resolved_paths:
            resolved_paths[path] = path.resolve()
        return resolved_paths[path]

    records = {}
    for region in regions:
        if region.region_id in records:
            raise ValueError(f"Duplicate frozen annotation ID: {region.region_id}")
        source = source_by_id[region.region_id]
        baseline = source.review_root / "native_references" / source.run / "pre_injection.jpg"
        frames = [source.review_root / "native_frames" / source.run /
                  f"frame_{index:03d}.jpg" for index in
                  sorted({region.frame_index, max(0, region.frame_index - 1)})]
        assets = [source.asset_manifest_path, baseline, *frames]
        resolved_assets = [resolved(asset) for asset in assets]
        path = resolved_assets[0]
        for asset in resolved_assets:
            if asset not in hashes:
                hashes[asset] = sha256_file(asset)
        records[region.region_id] = dict(
            region_id=region.region_id, run=region.run, dataset=region.dataset,
            phase_class=region.phase_class, confidence=region.confidence,
            geometry=training._geometry_key(region.run, region.payload),
            elapsed_h=region.elapsed_h,
            # A changed category with the same class and box is still a changed
            # human decision. Inherited labels must retain their full payload.
            payload_sha256=training._canonical_hash(region.payload),
            original_native_manifest_sha256=hashes[path],
            original_native_assets_sha256=training._canonical_hash(
                [hashes[asset] for asset in resolved_assets]),
            original_review_root=str(resolved(source.review_root)))
    return records


def frozen_region_contract(regions, source_by_id):
    records = frozen_region_records(regions, source_by_id)
    return training._canonical_hash([records[key] for key in sorted(records)])


def verify_parent_region_records(parent_records, captured_records):
    """Allow new IDs while refusing loss or revision of inherited labels."""
    missing = set(parent_records) - set(captured_records)
    if missing:
        raise ValueError(f"Optical capture lost {len(missing)} parent annotations")
    changed = [key for key in parent_records if captured_records[key] != parent_records[key]]
    if changed:
        raise ValueError(f"Optical capture changed {len(changed)} parent annotation revisions or native sources")
    return sorted(set(captured_records) - set(parent_records))


def optical_source_alignment(parent, evidence, review_jobs):
    """Audit each effective label against its immutable original native source."""
    descriptor = evidence["audit"]["annotation_origins"]
    origin_path = Path(descriptor["path"])
    if sha256_file(origin_path) != descriptor["sha256"]:
        raise ValueError("Regional annotation-origin audit changed")
    origins = read(origin_path)["regions"]
    pairs = [training.ReviewBundle(Path(decisions), Path(manifest))
             for decisions, manifest in evidence["pairs"]]
    region_sources = {}
    _, regions, _ = training._load_sources_and_regions(pairs,
        preserve_native_revisions=True, region_sources=region_sources)
    expected = {row["region_id"]: row for row in evidence["regions"]}
    if len(expected) != len(evidence["regions"]) or set(expected) != set(region_sources):
        raise ValueError("The regional evidence and original-native optical evidence differ")
    origin_manifests = {}
    expected_source_paths = {}
    for region in regions:
        original = expected[region.region_id]
        if (region.phase_class != original["phase_class"] or region.run != original["run"]
                or training._geometry_key(region.run, region.payload) !=
                   training._geometry_key(original["run"], original["payload"])):
            raise ValueError(f"Native optical review changed a label: {region.region_id}")
        origin = origins.get(region.region_id)
        if origin is None:
            raise ValueError(f"Missing frozen annotation origin: {region.region_id}")
        index = origin["document_index"]
        if index not in origin_manifests:
            origin_manifests[index] = read(evidence["pairs"][index][1])
        source_key = (index, region.run)
        if source_key not in expected_source_paths:
            matches = [job for job in origin_manifests[index]["jobs"]
                       if str(job["run"]).casefold() == region.run]
            if not matches:
                raise ValueError(f"Frozen origin has no native job for {region.region_id}")
            expected_source_paths[source_key] = {str(optical.native_source(job)[1].resolve())
                                                 for job in matches}
        expected_paths = expected_source_paths[source_key]
        if str(region_sources[region.region_id].asset_manifest_path.resolve()) not in expected_paths:
            raise ValueError(f"Optical source differs from frozen annotation origin: {region.region_id}")
    grouped = training._native_revision_task_groups(regions, region_sources)
    source_records = []
    for source, rows in grouped:
        native = read(source.asset_manifest_path)
        frame_ids = sorted({frame for row in rows for frame in
                            (row.frame_index, max(0, row.frame_index - 1))})
        baseline = source.review_root / "native_references" / source.run / "pre_injection.jpg"
        frames = [source.review_root / "native_frames" / source.run /
                  f"frame_{frame:03d}.jpg" for frame in frame_ids]
        assets = [source.asset_manifest_path, baseline, *frames]
        if any(not path.is_file() for path in assets):
            missing = next(path for path in assets if not path.is_file())
            raise FileNotFoundError(f"Frozen native source asset missing: {missing}")
        digests = [sha256_file(path) for path in assets]
        source_records.append(dict(run=source.run, manifest=str(source.asset_manifest_path),
            manifest_sha256=digests[0], baseline_sha256=digests[1],
            reviewed_regions=len(rows), reviewed_frames=len(frame_ids),
            bound_source_digest=hashlib.sha256("".join(digests).encode()).hexdigest(),
            corrected_reference_sha256=(native.get("reference") or {}).get("corrected_array_sha256")))
    region_records = frozen_region_records(regions, region_sources)
    return dict(effective_regions=len(regions), bound_regions=sum(row["reviewed_regions"] for row in source_records),
        unalignable_regions=0, native_revisions=len(source_records),
        physical_runs=len({source.run for source, _ in grouped}),
        region_contract_sha256=training._canonical_hash(
            [region_records[key] for key in sorted(region_records)]),
        region_records=region_records,
        source_bindings=source_records,
        optical_trainer_supports_per_annotation_native_revision=True)


def preflight(parent, library, cache, *, series="ffac"):
    """Read-only checks that also document the exact runtime used for training."""
    if series != "ffac":
        raise ValueError("This two-stage command is limited to FF_AC until its AC47 source contract is generalized")
    parent, library, cache = (resolve_review_path(Path(value)).resolve()
                              for value in (parent, library, cache))
    expected_library = resolve_review_path(series_path(series, "models")).resolve()
    if library != expected_library:
        raise ValueError("Final models must be placed in the FF_AC NAS model library")
    if not parent.is_dir() or cache.is_relative_to(parent) or library.is_relative_to(parent):
        raise ValueError("Select an existing parent and detached cache/library folders")
    summary = read(parent / "summary.json")
    previous = read(parent / "training_request.json")
    if summary.get("inventory", {}).get("datasets") != [series]:
        raise ValueError("Parent is not an isolated FF_AC model")
    if previous.get("evaluation_only") or previous.get("settings", {}).get("evaluation_only"):
        raise ValueError("Transfer evaluation cannot be an optical training parent")
    if summary.get("optical_training_performed") is not False or summary.get("regional_training_performed") is not True:
        raise ValueError("Expected the ready regional-only FF_AC parent with inherited optical heads")
    contracts = {}
    for head in HEADS:
        path = parent / "models" / "generic" / f"{head}_signal.json"
        payload = read(path)
        contract = payload["feature_contract"]
        if contract["generic_version"] != FEATURE_VERSION or tuple(contract["feature_names"]) != tuple(payload["feature_names"]):
            raise ValueError(f"Parent {head} is not on the explicit v5 optical contract")
        contracts[head] = dict(sha256=sha256_file(path), feature_count=len(payload["feature_names"]),
                               generic_version=contract["generic_version"])
    if contracts[HEADS[0]]["feature_count"] != contracts[HEADS[1]]["feature_count"]:
        raise ValueError("The inherited optical heads have different feature dimensions")
    profile = phase_runtime.load_physics_profile(parent)
    if profile is None or profile.get("format") != "baseline-relative-optical-regions-v1":
        raise ValueError("Parent has no verified regional physical profile")
    if {key.split("/")[0] for key in profile["runs"]} != {series}:
        raise ValueError("Parent physical context contains another series")
    runtime = phase_runtime.physics_runtime_identity(profile)
    validation = read(parent / "validation" / "postflight_manifest.json")
    if validation["physics_runtime"] != runtime:
        raise ValueError("Python/numpy/OpenCV/DarSIA runtime differs from the bound Studio validation")
    source = read(parent / "review_source" / "postflight_manifest.json")
    jobs = source.get("jobs") or []
    ac47 = [job for job in jobs if str(job["run"]).casefold() == "ac47"]
    if len(jobs) != 43 or len(ac47) != 1 or len({str(job["run"]).casefold() for job in jobs}) != 43:
        raise ValueError("Expected the complete 43-run AC review timeline with exactly one AC47")
    revision = ac47[0].get("photometry_revision") or {}
    receipt = Path(revision.get("receipt_path") or "")
    native = Path(ac47[0]["review_output_dir"]) / "native_frames" / "ac47" / "manifest.json"
    if (not receipt.is_file() or sha256_file(receipt) != revision.get("receipt_sha256")
            or sha256_file(native) != revision.get("target_manifest_sha256")):
        raise ValueError("AC47 revised photometry receipt or native image manifest changed")
    # The regional evidence inventory is the minimum new optical training must
    # retain; candidate-specific unsupported boxes are never pixel labels.
    evidence = read(parent / "evidence.json")
    regional_evidence = evidence["regions"]
    logistic.assert_safe_training_regions(regional_evidence)
    dense_ids = {row["region_id"] for row in regional_evidence
                 if row["payload"].get("category") == "confirmed_only_aqueous_co2"}
    if not dense_ids:
        raise ValueError("The latest confirmed-only-aqueous AC evidence is missing")
    alignment = optical_source_alignment(parent, evidence, jobs)
    return dict(format=FORMAT, series=series, parent=str(parent), library=str(library), cache=str(cache),
        parent_summary_sha256=sha256_file(parent / "summary.json"),
        parent_request_sha256=sha256_file(parent / "training_request.json"),
        parent_regional_model_sha256=sha256_file(parent / profile["model"]["path"]),
        inherited_optical_heads=contracts, feature_version=FEATURE_VERSION,
        physics_runtime=runtime, python=str(Path(__import__("sys").executable).resolve()),
        darsia_source=str(Path(darsia.__file__).resolve()), numpy_version=np.__version__,
        opencv_version=cv2.__version__, review_jobs=len(jobs),
        ac47_photometry_receipt_sha256=revision["receipt_sha256"],
        effective_regional_regions=len(regional_evidence),
        regional_region_contract_sha256=alignment["region_contract_sha256"],
        optical_source_alignment=alignment,
        required_confirmed_only_aqueous_ids=sorted(dense_ids))


def verify_optical_request(request, parent_receipt):
    root = Path(request["output_root"]).resolve()
    settings = request["settings"]
    if (settings.get("feature_version") != FEATURE_VERSION
            or set(settings.get("fit_heads") or []) != set(HEADS)
            or settings.get("inherited_heads")
            or settings.get("preserve_native_revisions") is not True
            or settings.get("manual_training_label_policy_version") != logistic.MANUAL_TRAINING_LABEL_POLICY_VERSION):
        raise ValueError("Optical request lost its explicit V4 labels or fresh v5 two-head fit")
    if root.is_relative_to(Path(parent_receipt["library"])) or (root / "validation").exists():
        raise ValueError("Unfitted optical intermediate must remain outside the Studio library")
    regions = effective_regions(root)
    ids = {region.region_id for region in regions}
    missing = set(parent_receipt["required_confirmed_only_aqueous_ids"]) - ids
    if missing:
        raise ValueError(f"Cumulative optical capture lost {len(missing)} confirmed aqueous regions")
    for region in regions:
        if region.region_id in parent_receipt["required_confirmed_only_aqueous_ids"] and region.phase_class != "aqueous":
            raise ValueError("Confirmed-only-aqueous evidence changed classification")
    captured_contract = None
    added_ids = []
    if "regional_region_contract_sha256" in parent_receipt:
        pairs = [training.ReviewBundle(Path(bundle["decisions"]["snapshot_path"]),
                                       Path(bundle["combined_manifest"]["snapshot_path"]))
                 for bundle in request["bundles"]]
        sources = {}
        _, check_regions, _ = training._load_sources_and_regions(pairs,
            preserve_native_revisions=True, region_sources=sources)
        parent_records = parent_receipt["optical_source_alignment"]["region_records"]
        captured_records = frozen_region_records(check_regions, sources)
        added_ids = verify_parent_region_records(parent_records, captured_records)
        captured_contract = training._canonical_hash(
            [captured_records[key] for key in sorted(captured_records)])
        if len(captured_records) != len(parent_records) + len(added_ids):
            raise ValueError("Optical request has ambiguous cumulative annotation IDs")
    profile = phase_runtime.load_physics_profile(root)
    if phase_runtime.physics_runtime_identity(profile) != parent_receipt["physics_runtime"]:
        raise ValueError("Optical staging changed inherited physical runtime")
    return dict(root=str(root), regions=len(regions), retained_confirmed_aqueous=len(parent_receipt["required_confirmed_only_aqueous_ids"]),
                new_annotation_ids=added_ids, region_contract_sha256=captured_contract,
                regional_profile_copied=sha256_file(root / profile["model"]["path"]) == parent_receipt["parent_regional_model_sha256"])


def frozen_optical_request(path, checked, actor):
    """Read the original request without changing its saved settings or snapshots."""
    request_path = Path(path).resolve()
    if not request_path.is_file() or request_path.name != "training_request.json":
        raise FileNotFoundError("Choose the frozen optical training_request.json")
    request = read(request_path)
    if (request_path.parent != Path(request["output_root"]).resolve()
            or not request_path.parent.is_relative_to(Path(checked["cache"]).resolve())
            or request.get("created_by") != actor):
        raise ValueError("Prepared optical request is outside this isolated actor/cache")
    request["request_path"] = str(request_path)
    request["frozen_pairs"] = [
        (Path(row["decisions"]["snapshot_path"]),
         Path(row["combined_manifest"]["snapshot_path"]))
        for row in request["bundles"]]
    return request


def resume_plan(job, checked, request_path, actor):
    """Inspect an interrupted optical fit before permitting a same-job continuation."""
    job = resolve_review_path(Path(job)).resolve()
    if not job.is_dir() or job.is_relative_to(Path(checked["library"])):
        raise ValueError("Resume requires the existing detached FF_AC results job")
    for name in ("result.json", "optical_fitted.json", "optical_quality_block.json",
                 "regional_started.json"):
        if (job / name).exists():
            raise ValueError(f"Job progressed beyond the resumable optical fit: {name}")
    for name in ("optical_parent", "regional_run", "regional_candidate"):
        if (job / name).exists():
            raise ValueError(f"Job has a partial later-stage output: {name}")
    if read(job / "preflight.json") != checked:
        raise ValueError("Recorded preflight differs from current parent, source, or runtime")
    request = frozen_optical_request(request_path, checked, actor)
    optical_check = verify_optical_request(request, checked)
    if not optical_check["regional_profile_copied"] or read(job / "optical_prepared.json") != optical_check:
        raise ValueError("Recorded optical preparation differs from the frozen request")
    optical_root = Path(request["output_root"]).resolve()
    if (optical_root / "summary.json").exists() or (optical_root / "models").exists():
        raise ValueError("Optical output was partly published; inspect it before resuming")
    staging = Path(request["cache_root"]).resolve() / "result_staging"
    inventory_path = staging / "training_inventory.json"
    samples_path = staging / "training_samples.npz"
    if not inventory_path.is_file() or not samples_path.is_file():
        raise ValueError("The original optical training inventory or samples are missing")
    inventory = read(inventory_path)
    settings = request["settings"]
    variants = training.model_variants(
        include_change_ablation=bool(settings.get("change_evidence_ablation")),
        generic_version=settings["feature_version"])
    expected = [(variant, head, contract) for variant, contract in variants.items()
                for head in settings["fit_heads"]]
    completed = []
    seen_gap = False
    for variant, head, contract in expected:
        path = staging / "models" / variant / f"{head}_signal.json"
        if not path.is_file():
            seen_gap = True
            continue
        if seen_gap:
            raise ValueError("Optical head checkpoints are not an ordered prefix")
        previous_campaign = training.CAMPAIGN_ID
        try:
            training.CAMPAIGN_ID = request["model_id"]
            reusable = training._load_reusable_head(
                path, head=head, fit_mode=settings["fit_mode"], inventory=inventory)
        finally:
            training.CAMPAIGN_ID = previous_campaign
        if reusable is None or reusable[0].feature_contract != contract.to_dict():
            raise ValueError(f"Checkpoint cannot be reused with this request: {variant}/{head}")
        model, report, _threshold, parity = reusable
        provenance = model.training_provenance
        source = provenance.get("source_sha256") or {}
        if (not parity.get("passed") or
                source.get("trainer") != sha256_file(Path(training.__file__)) or
                source.get("training_primitives") != sha256_file(Path(logistic.__file__)) or
                source.get("optical_features") != sha256_file(Path(phase_features.__file__)) or
                [float(row["regularization"]) for row in report["regularization_grid"]] !=
                   [float(value) for value in settings["regularization_grid"]]):
            raise ValueError(f"Checkpoint code, grid, or parity differs: {variant}/{head}")
        completed.append(dict(variant=variant, head=head,
                              relative_path=str(path.relative_to(staging)),
                              sha256=sha256_file(path),
                              model_sha256=model.to_dict()["model_sha256"]))
    unexpected = {str(path.relative_to(staging)) for path in (staging / "models").rglob("*_signal.json")} - {
        row["relative_path"] for row in completed}
    if unexpected or not completed:
        raise ValueError("Missing or unexpected optical head checkpoints")
    plan = dict(format="studio-optical-then-regional-resume-v1", job=str(job),
                request_path=str(Path(request_path).resolve()),
                request_sha256=sha256_file(Path(request_path)),
                preflight_sha256=sha256_file(job / "preflight.json"),
                optical_prepared_sha256=sha256_file(job / "optical_prepared.json"),
                inventory_sha256=sha256_file(inventory_path),
                samples_sha256=sha256_file(samples_path),
                completed_heads=completed,
                next_head=(f"{expected[len(completed)][0]}/{expected[len(completed)][1]}"
                           if len(completed) < len(expected) else None))
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    current = {row["relative_path"]: row["sha256"] for row in completed}
    for previous in sorted((job / "resume_attempts").glob("*.json")):
        earlier = read(previous)
        for key in ("request_sha256", "preflight_sha256", "optical_prepared_sha256"):
            if earlier.get(key) != plan[key]:
                raise ValueError(f"Earlier resume attempt has a different {key}")
        for row in earlier.get("completed_heads") or []:
            if current.get(row["relative_path"]) != row["sha256"]:
                raise ValueError("A checkpoint changed since an earlier resume attempt")
    return plan, request


def post_optical_resume_plan(job, checked, request_path, actor, *, allow_empty_optical_parent=False):
    """Inspect a fully synced optical fit stopped before coupled refinement."""
    from darsia.fluidflower.tools.train_coupled_aqueous_revision import RECEIPT, SEED

    job = resolve_review_path(Path(job)).resolve()
    if not job.is_dir() or job.is_relative_to(Path(checked["library"])):
        raise ValueError("Post-optical resume requires the detached FF_AC results job")
    for name in ("result.json", "optical_fitted.json", "optical_quality_block.json",
                 "regional_started.json", "candidate_qa_state.json"):
        if (job / name).exists():
            raise ValueError(f"Job progressed beyond post-optical resume: {name}")
    for name in ("optical_parent", "regional_run", "regional_candidate"):
        path = job / name
        if path.exists() and (name != "optical_parent" or not allow_empty_optical_parent
                              or not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"Job has a partial later-stage output: {name}")
    if read(job / "preflight.json") != checked:
        raise ValueError("Recorded preflight differs from current parent, source, or runtime")
    request = frozen_optical_request(request_path, checked, actor)
    optical_check = verify_optical_request(request, checked)
    if (not optical_check["regional_profile_copied"]
            or read(job / "optical_prepared.json") != optical_check):
        raise ValueError("Recorded optical preparation differs from the frozen request")
    previous = sorted((job / "resume_attempts").glob("[0-9][0-9][0-9][0-9].json"))
    if not previous or previous[0].name != "0001.json":
        raise ValueError("Post-optical resume requires the recorded first optical attempt")
    first = read(previous[0])
    original_plan = {key: value for key, value in first.items()
                     if key not in {"plan_sha256", "resumed_utc"}}
    original_plan_sha = hashlib.sha256(json.dumps(
        original_plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if (first.get("format") != "studio-optical-then-regional-resume-v1"
            or first.get("job") != str(job)
            or first.get("plan_sha256") != original_plan_sha
            or first.get("next_head") != "btb-ratiometric-relative-v2/gas_within_co2"
            or len(first.get("completed_heads") or []) != 3):
        raise ValueError("First optical resume attempt does not match the inspected three-head state")
    request_path = Path(request_path).resolve()
    if first.get("request_path") != str(request_path):
        raise ValueError("First optical resume attempt belongs to another request")
    output = Path(request["output_root"]).resolve()
    staging = Path(request["cache_root"]).resolve() / "result_staging"
    source_hashes = dict(request_sha256=sha256_file(request_path),
        preflight_sha256=sha256_file(job / "preflight.json"),
        optical_prepared_sha256=sha256_file(job / "optical_prepared.json"),
        inventory_sha256=sha256_file(staging / "training_inventory.json"),
        samples_sha256=sha256_file(staging / "training_samples.npz"))
    if any(first.get(key) != value for key, value in source_hashes.items()):
        raise ValueError("Frozen request, preflight, or optical training inputs changed")
    if (output / RECEIPT).exists() or (output / SEED).exists():
        raise ValueError("Coupled refinement already started; inspect its partial output")
    if (output / "validation").exists() or (output / "publication_receipt.json").exists():
        raise ValueError("Optical output has already been published or validated")
    synced_inputs = {}
    for name in ("training_samples.npz", "training_inventory.json", "region_inventory.csv"):
        staged_input = staging / name
        output_input = output / name
        if not staged_input.is_file() or not output_input.is_file():
            raise ValueError(f"Missing synced optical training input: {name}")
        digest = sha256_file(staged_input)
        if sha256_file(output_input) != digest:
            raise ValueError(f"Synced optical training input differs from staging: {name}")
        synced_inputs[name] = digest
    summary_path = output / "summary.json"
    if (not summary_path.is_file() or
            sha256_file(summary_path) != sha256_file(staging / "summary.json")):
        raise ValueError("Synced optical summary differs from the fitted staging summary")
    summary = read(summary_path)
    if summary.get("campaign_id") != request["model_id"]:
        raise ValueError("Synced optical summary identifies another campaign")
    variants = training.model_variants(
        include_change_ablation=bool(request["settings"].get("change_evidence_ablation")),
        generic_version=request["settings"]["feature_version"])
    expected = [(variant, head, contract) for variant, contract in variants.items()
                for head in request["settings"]["fit_heads"]]
    if len(expected) != 4 or len({(variant, head) for variant, head, _ in expected}) != 4:
        raise ValueError("Post-optical resume requires exactly four fitted optical heads")
    inventory = read(staging / "training_inventory.json")
    heads = []
    for variant, head, contract in expected:
        relative = Path("models") / variant / f"{head}_signal.json"
        staged = staging / relative
        fitted_path = output / relative
        if not staged.is_file() or not fitted_path.is_file():
            raise ValueError(f"Missing fitted optical head: {variant}/{head}")
        digest = sha256_file(staged)
        if sha256_file(fitted_path) != digest:
            raise ValueError(f"Synced optical head differs from staging: {variant}/{head}")
        previous_campaign = training.CAMPAIGN_ID
        try:
            training.CAMPAIGN_ID = request["model_id"]
            reusable = training._load_reusable_head(staged, head=head,
                fit_mode=request["settings"]["fit_mode"], inventory=inventory)
        finally:
            training.CAMPAIGN_ID = previous_campaign
        if reusable is None or reusable[0].feature_contract != contract.to_dict():
            raise ValueError(f"Fitted head cannot be reused with this request: {variant}/{head}")
        model, report, _threshold, parity = reusable
        provenance = model.training_provenance
        sources = provenance.get("source_sha256") or {}
        if (not parity.get("passed") or
                sources.get("trainer") != sha256_file(Path(training.__file__)) or
                sources.get("training_primitives") != sha256_file(Path(logistic.__file__)) or
                sources.get("optical_features") != sha256_file(Path(phase_features.__file__)) or
                [float(row["regularization"]) for row in report["regularization_grid"]] !=
                   [float(value) for value in request["settings"]["regularization_grid"]]):
            raise ValueError(f"Fitted head provenance or grid differs: {variant}/{head}")
        row = dict(variant=variant, head=head, relative_path=str(relative),
                   sha256=digest, model_sha256=model.to_dict()["model_sha256"])
        heads.append(row)
    if {str(path.relative_to(staging)) for path in (staging / "models").rglob("*_signal.json")} != {
            row["relative_path"] for row in heads}:
        raise ValueError("Staging has unexpected optical heads")
    if [row for row in heads[:3]] != first["completed_heads"]:
        raise ValueError("The original three verified checkpoint identities changed")
    if set(summary.get("reused_heads") or []) != {
            f"{row['variant']}/{row['head']}" for row in first["completed_heads"]}:
        raise ValueError("Optical summary does not identify the three reused heads")
    fitted = verify_fitted(output, allow_nonconverged=True)
    if fitted["nonconverged_heads"] or fitted["physics_runtime"] != checked["physics_runtime"]:
        raise ValueError("Optical heads did not converge or inherited physics changed")
    plan = dict(format="studio-optical-then-regional-post-optical-resume-v1",
        job=str(job), request_path=str(request_path), **source_hashes,
        first_attempt_sha256=sha256_file(previous[0]),
        region_inventory_sha256=synced_inputs["region_inventory.csv"],
        optical_summary_sha256=sha256_file(summary_path),
        optical_model_sha256=fitted["model_sha256"], completed_heads=heads,
        next_step="coupled_aqueous_refinement")
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan, request


def post_fitted_resume_plan(job, checked, request_path, actor):
    """Inspect a failed optical publication without permitting another fit."""
    from darsia.fluidflower.tools.train_coupled_aqueous_revision import (
        AQ_CLASSES, RECEIPT, SEED, ordered_regions)

    job = resolve_review_path(Path(job)).resolve()
    optical_plan, request = post_optical_resume_plan(job, checked, request_path, actor,
        allow_empty_optical_parent=True)
    attempts = job / "resume_attempts"
    second = attempts / "0002.json"
    if not second.is_file() or {key: value for key, value in read(second).items()
                               if key != "resumed_utc"} != optical_plan:
        raise ValueError("Recorded second attempt differs from the completed optical fit")
    output = Path(request["output_root"]).resolve()
    if ((output / RECEIPT).exists() or (output / SEED).exists()
            or (output / "coupled_aqueous_fit_verification.json").exists()
            or request["settings"].get("coupled_aqueous_refinement")):
        raise ValueError("Coupled refinement has started; this no-coupled resume is unsafe")
    if any(region.phase_class in AQ_CLASSES and region.supervision_weight > 0
           for region in ordered_regions(output)):
        raise ValueError("Frozen labels require coupled refinement")
    fitted = verify_fitted(output, allow_nonconverged=False)
    if (fitted["model_sha256"] != optical_plan["optical_model_sha256"]
            or fitted["physics_runtime"] != checked["physics_runtime"]):
        raise ValueError("Completed optical identity differs from the recorded second attempt")
    hidden = job / "optical_parent" / output.name
    if hidden.exists():
        raise ValueError("The hidden optical parent already exists")
    source_files = {path.relative_to(output).as_posix(): sha256_file(path)
                    for path in fitted_files(output)}
    # A failed cross-volume rename left a complete local copy. Inspect it and
    # preserve it as evidence; the publisher handles the destination-volume move.
    orphan = []
    local_stage = Path(checked["cache"]) / "optical-regional-publication-stage"
    for candidate in sorted(local_stage.glob(f"fitted-candidate-*/{output.name}")):
        receipt_path = candidate / "publication_receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"Unidentified partial optical publication stage: {candidate}")
        receipt = read(receipt_path)
        if receipt.get("destination") != str(hidden):
            continue
        if (receipt.get("source") != str(output)
                or receipt.get("source_files") != source_files
                or (receipt.get("model_id"), receipt.get("model_sha256")) !=
                   (fitted["model_id"], fitted["model_sha256"])):
            raise ValueError("Orphan optical publication stage has different source identity")
        published = receipt.get("published_files") or {}
        if (set(published) != set(source_files)
                or any(not (candidate / relative).is_file()
                       or sha256_file(candidate / relative) != row["sha256"]
                       for relative, row in published.items())
                or verify_fitted(candidate, logical_root=hidden,
                                 allow_nonconverged=False) != fitted):
            raise ValueError("Orphan optical publication stage is incomplete or changed")
        orphan.append(dict(path=str(candidate), receipt_sha256=sha256_file(receipt_path)))
    if len(orphan) > 1:
        raise ValueError("Multiple optical publication stages identify this job")
    plan = dict(format="studio-optical-then-regional-post-fitted-resume-v1",
        job=str(job), request_path=str(Path(request_path).resolve()),
        second_attempt_sha256=sha256_file(second),
        optical_plan_sha256=optical_plan["plan_sha256"],
        output=str(output), hidden_parent=str(hidden), publication_staging_root=str(local_stage),
        fitted=fitted, source_files=source_files, orphan_local_stage=orphan,
        coupled_refinement="not_applicable_no_active_aqueous_region_labels",
        next_step="publish_fitted_optical_parent")
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan, request


def regional_capture_resume_plan(job, checked, request_path, actor):
    """Inspect a failed regional capture after the fitted optical parent was published.

    This continuation is deliberately limited to a failure before evidence or
    candidate output exists. It can therefore replay capture without repeating
    either fitted layer or changing the allocated candidate identity.
    """
    job = resolve_review_path(Path(job)).resolve()
    library = Path(checked["library"]).resolve()
    if not job.is_dir() or job.is_relative_to(library):
        raise ValueError("Regional capture resume requires a detached results job")
    if read(job / "preflight.json") != checked:
        raise ValueError("Recorded preflight differs from current parent, source, or runtime")
    request = frozen_optical_request(request_path, checked, actor)
    optical_check = verify_optical_request(request, checked)
    if (not optical_check["regional_profile_copied"]
            or read(job / "optical_prepared.json") != optical_check):
        raise ValueError("Frozen optical preparation changed")
    for name in ("result.json", "candidate_qa_state.json", "optical_quality_block.json"):
        if (job / name).exists():
            raise ValueError(f"Job progressed beyond failed regional capture: {name}")

    third_path = job / "resume_attempts" / "0003.json"
    third = read(third_path)
    third_body = {key: value for key, value in third.items()
                  if key not in {"plan_sha256", "resumed_utc"}}
    third_hash = hashlib.sha256(json.dumps(
        third_body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    second_path = job / "resume_attempts" / "0002.json"
    if (third.get("format") != "studio-optical-then-regional-post-fitted-resume-v1"
            or third.get("plan_sha256") != third_hash
            or third.get("second_attempt_sha256") != sha256_file(second_path)
            or third.get("job") != str(job)
            or third.get("request_path") != str(Path(request_path).resolve())):
        raise ValueError("Recorded post-fitted continuation changed")

    output = Path(request["output_root"]).resolve()
    hidden = job / "optical_parent" / output.name
    optical_receipt_path = job / "optical_fitted.json"
    optical_receipt = read(optical_receipt_path)
    publication_path = hidden / "publication_receipt.json"
    publication = read(publication_path)
    fitted = verify_fitted(output, allow_nonconverged=False)
    if (third.get("output") != str(output)
            or third.get("hidden_parent") != str(hidden)
            or third.get("fitted") != fitted
            or optical_receipt.get("fitted") != fitted
            or optical_receipt.get("hidden_parent") != str(hidden)
            or optical_receipt.get("receipt_sha256") != sha256_file(publication_path)
            or publication.get("source") != str(output)
            or publication.get("destination") != str(hidden)
            or publication.get("source_files") != third.get("source_files")
            or verify_fitted(hidden, allow_nonconverged=False) != fitted):
        raise ValueError("Published optical parent differs from frozen fitted output")
    if (hidden / "validation").exists() or hidden.is_relative_to(library):
        raise ValueError("Optical-only parent was exposed to Studio")
    published_files = publication.get("published_files") or {}
    if (set(published_files) != set(third["source_files"])
            or any(not (hidden / relative).is_file()
                   or sha256_file(hidden / relative) != row["sha256"]
                   for relative, row in published_files.items())):
        raise ValueError("Published optical files are missing or changed")

    started_path = job / "regional_started.json"
    started = read(started_path)
    regional_job = job / "regional_run"
    candidate = Path(started["output"]).resolve()
    if (Path(started["job"]).resolve() != regional_job
            or not candidate.is_relative_to(job / "regional_candidate")
            or not candidate.name.startswith("phase-model-")
            or candidate.exists() or (job / "regional_candidate").exists()):
        raise ValueError("Regional candidate allocation changed or already produced output")
    launch_path = regional_job / "launch_request.json"
    launch = read(launch_path)
    if (launch.get("format") != FORMAT
            or launch.get("parent") != str(hidden)
            or launch.get("output") != str(candidate)
            or launch.get("job") != str(regional_job)
            or launch.get("optical_receipt") != str(optical_receipt_path)
            or launch.get("optical_feature_version") != FEATURE_VERSION
            or launch.get("optical_heads") != list(HEADS)
            or not isinstance(launch.get("bottom_gas_guard", False), bool)
            or not isinstance(launch.get("region_edge_guard", False), bool)
            or launch.get("review_candidate_only") is not True
            or launch.get("studio_published") is not False):
        raise ValueError("Regional launch request differs from published optical parent")
    status_path, timing_path = regional_job / "status.json", regional_job / "timing.json"
    status, timing = read(status_path), read(timing_path)
    if (status.get("stage") != "failed"
            or "Target context provenance identifies another source manifest" not in status.get("error", "")
            or Path(status.get("model_root", "")).resolve() != candidate
            or timing.get("outcome") != "failed"
            or [stage["stage"] for stage in timing.get("completed_stages", [])] !=
               ["initialization", "capture"]):
        raise ValueError("Regional job did not stop at the known capture provenance failure")
    allowed = {"timing_events", "timing_runs", "execution_resources.json",
               "launch_request.json", "overnight_status.json", "status.json", "timing.json"}
    if ({path.name for path in regional_job.iterdir()} - allowed
            or (regional_job / "evidence.json").exists()):
        raise ValueError("Regional job has uninspected partial evidence or fitted output")
    archived_timing = {sha256_file(path) for path in
                       (regional_job / "timing_runs").glob("*.json")}
    if sha256_file(timing_path) not in archived_timing:
        raise ValueError("Current failed regional timing is not archived")
    plan = dict(format="studio-optical-then-regional-capture-resume-v1",
        job=str(job), request_path=str(Path(request_path).resolve()),
        preflight_sha256=sha256_file(job / "preflight.json"),
        third_attempt_sha256=sha256_file(third_path),
        optical_receipt_sha256=sha256_file(optical_receipt_path),
        publication_receipt_sha256=sha256_file(publication_path),
        optical_model_sha256=fitted["model_sha256"],
        hidden_parent=str(hidden), regional_job=str(regional_job),
        candidate_output=str(candidate), regional_launch_sha256=sha256_file(launch_path),
        failed_status_sha256=sha256_file(status_path),
        failed_timing_sha256=sha256_file(timing_path),
        next_step="retry_regional_capture_only")
    previous_paths = sorted((job / "resume_attempts").glob("[0-9][0-9][0-9][0-9].json"))
    previous_paths = [path for path in previous_paths if int(path.stem) >= 4]
    if [path.name for path in previous_paths] != [
            f"{index:04d}.json" for index in range(4, 4 + len(previous_paths))]:
        raise ValueError("Regional capture continuation receipts are not contiguous")
    stable_keys = ("format", "job", "request_path", "preflight_sha256",
                   "third_attempt_sha256", "optical_receipt_sha256",
                   "publication_receipt_sha256", "optical_model_sha256",
                   "hidden_parent", "regional_job", "candidate_output",
                   "regional_launch_sha256", "next_step")
    previous_capture_attempts = []
    for path in previous_paths:
        previous = read(path)
        body = {key: value for key, value in previous.items()
                if key not in {"plan_sha256", "resumed_utc"}}
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if (previous.get("plan_sha256") != digest
                or any(previous.get(key) != plan[key] for key in stable_keys)
                or previous.get("failed_timing_sha256") not in archived_timing):
            raise ValueError("Earlier regional capture continuation changed")
        previous_capture_attempts.append(dict(path=str(path), sha256=sha256_file(path)))
    plan["previous_capture_attempts"] = previous_capture_attempts
    plan["plan_sha256"] = hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return plan, request


@contextmanager
def resume_execution_lock(job):
    """Keep concurrent continuations out; the OS releases this lock on process death."""
    path = Path(job) / "resume_execution.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def freeze_regional_evidence(hidden, regional_args, optical_check):
    """Reuse the optical fit's frozen labels, even if review is still open."""
    expected = optical_check.get("region_contract_sha256")
    if not expected:
        raise ValueError("Frozen optical annotation/source contract is unavailable")
    root = Path(regional_args.job)
    if (root / "evidence.json").exists():
        raise ValueError("Regional evidence was already captured; inspect this job before resuming")
    hidden = Path(hidden)
    policy = phase_review_retractions.model_retractions(hidden, series="ffac")
    _previous, pairs, regions, audit = optical.capture_inputs(
        hidden, "ffac", root / "evidence_capture", [],
        withdrawal_snapshot=policy, preserve_native_revisions=True)
    frozen_pairs = [training.ReviewBundle(Path(decisions), Path(manifest))
                    for decisions, manifest in pairs]
    source_by_id = {}
    _, captured, _ = training._load_sources_and_regions(
        frozen_pairs, preserve_native_revisions=True, region_sources=source_by_id)
    actual = frozen_region_contract(captured, source_by_id)
    if (actual != expected or len(captured) != optical_check["regions"]
            or {row.region_id for row in captured} != {row.region_id for row in regions}):
        raise ValueError("Regional capture differs from the frozen optical labels or native sources")
    evidence = dict(parent=str(hidden),
        pairs=[[str(path) for path in pair] for pair in pairs],
        regions=[asdict(region) for region in regions], audit=audit)
    update.write(root / "evidence.json", evidence)
    update.write(root / "snapshot_parity.json", dict(format=FORMAT,
        source="frozen-optical-training-request", optical_region_contract_sha256=expected,
        regional_region_contract_sha256=actual, regions=len(captured),
        additional_annotation_ids=optical_check.get("new_annotation_ids", []),
        live_review_sessions_reopened=False))
    # The ordinary capture helper returns early when evidence.json exists. Keep
    # its companion audit so the detached job remains interpretable on its own.
    corrected = [row.region_id for row in regions if row.source == "manual"
                 and row.payload.get("category") == "missed_visible_aqueous_co2"]
    update.write(root / "supervision_policy.json", dict(
        policy="missed-aqueous-is-dense-reviewer-clarification-20260920",
        reviewer_statement="The whole marked area is aqueous CO2",
        previous_interpretation="aqueous_present_region", current_interpretation="aqueous",
        corrected_regions=corrected, original_documents_modified=False,
        mixed_phase_classes_unchanged=True))


def _finish_regional_job(job, checked, hidden, regional_args):
    regional_args.status_callback = None
    freeze_regional_evidence(hidden, regional_args, read(Path(job) / "optical_prepared.json"))
    update.run(regional_args)
    candidate = Path(regional_args.output)
    # Validation binds to the byte-level model identity, including summary.json.
    # Keep the materialized model immutable and record QA state outside it.
    write(job / "candidate_qa_state.json", dict(
        candidate_model=str(candidate), review_candidate_only=True,
        studio_published=False,
        qa_required_before_publication=["AC47 6 h", "AC51 24 h", "AC51 48 h", "AC51 71 h"],
    ))
    result = dict(format=FORMAT, completed_utc=datetime.now(timezone.utc).isoformat(),
        original_parent=checked["parent"], optical_parent=str(hidden),
        candidate_model=str(candidate), review_candidate_only=True,
        studio_published=False, qa_pending=True,
        regional_job=str(regional_args.job), optical_heads_refitted=True,
        regional_layer_refitted=True, shared_physics_runtime=checked["physics_runtime"],
        original_parent_unchanged=sha256_file(Path(checked["parent"]) / "summary.json") == checked["parent_summary_sha256"])
    if not result["original_parent_unchanged"]:
        raise ValueError("Original model changed during detached training")
    write(job / "result.json", result)
    return result


def _execute_from_checked(checked, job, actor, *, backend="auto", cpus=None, memory_gb=None,
                          prepared_optical_request=None, resume_existing=False,
                          resume_post_optical=False, resume_post_fitted=False,
                          resume_regional_capture=False,
                          expected_resume_plan_sha256=None):
    if checked["optical_source_alignment"]["unalignable_regions"]:
        raise ValueError("Optical extraction did not bind every original AC review image")
    job = resolve_review_path(Path(job)).resolve()
    if resume_regional_capture:
        if prepared_optical_request is None or not expected_resume_plan_sha256:
            raise ValueError("Regional capture resume requires the frozen request and inspected plan SHA")
        plan, _request = regional_capture_resume_plan(
            job, checked, prepared_optical_request, actor)
        if plan["plan_sha256"] != expected_resume_plan_sha256:
            raise ValueError("Regional capture resume plan changed since inspection")
        attempts = job / "resume_attempts"
        index = 1 + max((int(path.stem) for path in attempts.glob("[0-9][0-9][0-9][0-9].json")), default=0)
        with (attempts / f"{index:04d}.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(**plan, resumed_utc=datetime.now(timezone.utc).isoformat()),
                      stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        hidden = Path(plan["hidden_parent"])
        regional_args = regional.prepare_launch(hidden, checked["library"], checked["cache"],
            "ffac", actor, backend=backend, cpus=cpus, memory_gb=memory_gb)
        regional_args.output = Path(plan["candidate_output"])
        regional_args.job = Path(plan["regional_job"])
        regional_args.bottom_gas_guard = read(regional_args.job / "launch_request.json").get(
            "bottom_gas_guard", False)
        regional_args.region_edge_guard = read(regional_args.job / "launch_request.json").get(
            "region_edge_guard", False)
        return _finish_regional_job(job, checked, hidden, regional_args)
    if resume_post_fitted:
        if prepared_optical_request is None or not expected_resume_plan_sha256:
            raise ValueError("Post-fitted resume requires the frozen request and inspected plan SHA")
        plan, request = post_fitted_resume_plan(job, checked, prepared_optical_request, actor)
        if plan["plan_sha256"] != expected_resume_plan_sha256:
            raise ValueError("Post-fitted resume plan changed since inspection")
        attempts = job / "resume_attempts"
        index = 1 + max((int(path.stem) for path in attempts.glob("[0-9][0-9][0-9][0-9].json")), default=0)
        with (attempts / f"{index:04d}.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(**plan, resumed_utc=datetime.now(timezone.utc).isoformat()),
                      stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    elif resume_post_optical:
        if prepared_optical_request is None or not expected_resume_plan_sha256:
            raise ValueError("Post-optical resume requires the frozen request and inspected plan SHA")
        plan, request = post_optical_resume_plan(job, checked, prepared_optical_request, actor)
        if plan["plan_sha256"] != expected_resume_plan_sha256:
            raise ValueError("Post-optical resume plan changed since inspection")
        attempts = job / "resume_attempts"
        index = 1 + max((int(path.stem) for path in attempts.glob("[0-9][0-9][0-9][0-9].json")), default=0)
        with (attempts / f"{index:04d}.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(**plan, resumed_utc=datetime.now(timezone.utc).isoformat()),
                      stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        from darsia.fluidflower.tools.train_coupled_aqueous_revision import refine_unpublished
        refine_unpublished(request)
    elif resume_existing:
        if prepared_optical_request is None or not expected_resume_plan_sha256:
            raise ValueError("Resume requires the frozen request and inspected plan SHA")
        plan, request = resume_plan(job, checked, prepared_optical_request, actor)
        if plan["plan_sha256"] != expected_resume_plan_sha256:
            raise ValueError("Resume plan changed since inspection")
        attempts = job / "resume_attempts"
        attempts.mkdir(exist_ok=True)
        index = 1 + max((int(path.stem) for path in attempts.glob("[0-9][0-9][0-9][0-9].json")), default=0)
        with (attempts / f"{index:04d}.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(**plan, resumed_utc=datetime.now(timezone.utc).isoformat()),
                      stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        staging = Path(request["cache_root"]).resolve() / "result_staging"
        optical.fit(request, resume_completed_heads=True,
            required_resume_checkpoints={staging / row["relative_path"]: row["sha256"]
                                         for row in plan["completed_heads"]})
    else:
        if job.exists():
            raise FileExistsError("Choose a fresh FF_AC NAS results folder for this detached run")
        job.mkdir(parents=True)
        write(job / "preflight.json", checked)
        if prepared_optical_request is None:
            request, _unused_destination = optical.prepare(checked["parent"], checked["library"], checked["cache"],
                "ffac", actor, backend=backend, feature_version=FEATURE_VERSION,
                preserve_native_revisions=True)
        else:
            request = frozen_optical_request(prepared_optical_request, checked, actor)
        optical_check = verify_optical_request(request, checked)
        if not optical_check["regional_profile_copied"]:
            raise ValueError("Inherited regional artifact was not copied byte-for-byte")
        write(job / "optical_prepared.json", optical_check)
        optical.fit(request)
    # A nonconverged optical head is a research checkpoint, not a parent for
    # another fitted layer. verify_fitted also checks numerical parity, head
    # provenance, frozen evidence and the physical runtime.
    fitted = verify_fitted(request["output_root"], allow_nonconverged=True)
    if fitted["nonconverged_heads"]:
        write(job / "optical_quality_block.json", dict(
            reason="One or more optical heads did not converge; regional fit was not started",
            fitted=fitted))
        raise ValueError("Optical heads did not converge; regional fit and Studio publication are blocked")
    if fitted["physics_runtime"] != checked["physics_runtime"]:
        raise ValueError("Optical fitting changed physical runtime")
    hidden = job / "optical_parent" / Path(request["output_root"]).name
    # This verified copy is durable lineage on the NAS, but has no validation
    # and is deliberately outside the models folder scanned by Studio.
    staging = (Path(plan["publication_staging_root"]) if resume_post_fitted
               else Path(checked["cache"]) / "optical-regional-publication-stage")
    published = publish_fitted(request["output_root"], hidden, staging, allow_nonconverged=False)
    if (hidden / "validation").exists() or hidden.is_relative_to(Path(checked["library"])):
        raise ValueError("Optical-only intermediate was accidentally exposed to Studio")
    write(job / "optical_fitted.json", dict(fitted=fitted, hidden_parent=str(hidden),
        receipt_sha256=sha256_file(hidden / "publication_receipt.json"),
        nonconverged_heads=published["nonconverged_heads"]))
    regional_args = regional.prepare_launch(hidden, checked["library"], checked["cache"], "ffac", actor,
        backend=backend, cpus=cpus, memory_gb=memory_gb)
    regional_args.output = job / "regional_candidate" / regional_args.output.name
    regional_args.job = job / "regional_run"
    regional_args.bottom_gas_guard = True
    regional_args.region_edge_guard = True
    if regional_args.output.is_relative_to(Path(checked["library"])):
        raise ValueError("QA candidate must remain outside Studio's model library")
    regional_args.job.mkdir(parents=True, exist_ok=False)
    write(regional_args.job / "launch_request.json", dict(format=FORMAT, parent=str(hidden),
        output=str(regional_args.output), job=str(regional_args.job),
        optical_receipt=str(job / "optical_fitted.json"),
         optical_feature_version=FEATURE_VERSION, optical_heads=list(HEADS),
         bottom_gas_guard=regional_args.bottom_gas_guard,
         region_edge_guard=regional_args.region_edge_guard,
         review_candidate_only=True, studio_published=False))
    write(job / "regional_started.json", dict(output=str(regional_args.output), job=str(regional_args.job)))
    return _finish_regional_job(job, checked, hidden, regional_args)


def execute(parent, library, cache, job, actor, *, backend="auto", cpus=None, memory_gb=None,
            prepared_optical_request=None, resume_existing=False,
            resume_post_optical=False, resume_post_fitted=False,
            resume_regional_capture=False,
            expected_resume_plan_sha256=None):
    if sum((resume_existing, resume_post_optical, resume_post_fitted,
            resume_regional_capture)) > 1:
        raise ValueError("Select one same-job resume stage")
    checked = preflight(parent, library, cache)
    if resume_existing or resume_post_optical or resume_post_fitted or resume_regional_capture:
        job = resolve_review_path(Path(job)).resolve()
        if not job.is_dir():
            raise ValueError("Resume requires the existing detached FF_AC results job")
        with resume_execution_lock(job):
            return _execute_from_checked(checked, job, actor, backend=backend, cpus=cpus,
                memory_gb=memory_gb, prepared_optical_request=prepared_optical_request,
                resume_existing=resume_existing, resume_post_optical=resume_post_optical,
                resume_post_fitted=resume_post_fitted,
                resume_regional_capture=resume_regional_capture,
                expected_resume_plan_sha256=expected_resume_plan_sha256)
    return _execute_from_checked(checked, job, actor, backend=backend, cpus=cpus,
        memory_gb=memory_gb, prepared_optical_request=prepared_optical_request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "library", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--job", type=Path, help="Fresh FF_AC NAS results folder; required with --execute")
    parser.add_argument("--actor", default="Olav Folkvord")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-gb", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Launch optical fit, regional fit, then Studio maps")
    mode.add_argument("--inspect-resume", action="store_true", help="Read-only check of an interrupted optical fit")
    mode.add_argument("--resume-existing", action="store_true", help="Continue the inspected existing results job")
    mode.add_argument("--inspect-post-optical-resume", action="store_true",
                      help="Read-only check of a fitted optical output stopped before coupled refinement")
    mode.add_argument("--resume-post-optical", action="store_true",
                      help="Continue the inspected fitted optical output without training its heads again")
    mode.add_argument("--inspect-post-fitted-resume", action="store_true",
                      help="Read-only check after a failed optical publication")
    mode.add_argument("--resume-post-fitted", action="store_true",
                      help="Publish an inspected fitted optical output and continue the regional stage")
    mode.add_argument("--inspect-regional-capture-resume", action="store_true",
                      help="Read-only check of the same job after a failed regional evidence capture")
    mode.add_argument("--resume-regional-capture", action="store_true",
                      help="Retry only the inspected regional capture and later stages")
    parser.add_argument("--prepared-optical-request", type=Path,
                        help="Reuse an already verified frozen optical request and feature cache")
    parser.add_argument("--expected-resume-plan-sha256",
                        help="Exact plan_sha256 returned by --inspect-resume")
    args = parser.parse_args(argv)
    if (args.execute or args.resume_existing or args.resume_post_optical
            or args.resume_post_fitted or args.resume_regional_capture):
        if args.job is None:
            parser.error("--job is required with a training action")
        value = execute(args.parent, args.library, args.cache, args.job, args.actor,
                        backend=args.backend, cpus=args.cpus, memory_gb=args.memory_gb,
                        prepared_optical_request=args.prepared_optical_request,
                        resume_existing=args.resume_existing,
                        resume_post_optical=args.resume_post_optical,
                        resume_post_fitted=args.resume_post_fitted,
                        resume_regional_capture=args.resume_regional_capture,
                        expected_resume_plan_sha256=args.expected_resume_plan_sha256)
    elif (args.inspect_resume or args.inspect_post_optical_resume
          or args.inspect_post_fitted_resume or args.inspect_regional_capture_resume):
        if args.job is None or args.prepared_optical_request is None:
            parser.error("Resume inspection requires --job and --prepared-optical-request")
        checked = preflight(args.parent, args.library, args.cache)
        inspector = (regional_capture_resume_plan if args.inspect_regional_capture_resume else
                     post_fitted_resume_plan if args.inspect_post_fitted_resume else
                     post_optical_resume_plan if args.inspect_post_optical_resume else resume_plan)
        value, _request = inspector(args.job, checked, args.prepared_optical_request, args.actor)
    else:
        value = preflight(args.parent, args.library, args.cache)
        value.pop("required_confirmed_only_aqueous_ids")
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
