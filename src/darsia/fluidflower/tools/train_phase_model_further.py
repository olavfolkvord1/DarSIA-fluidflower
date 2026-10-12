"""Retrain a new version from cumulative reviews, then prepare native review maps.

This is a Studio companion job. It snapshots live evidence without locking it,
uses the installed trainer unchanged, and preserves the selected physical model.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter
from collections.abc import Mapping
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys
import uuid


from darsia.fluidflower import phase_features, phase_runtime
from darsia.fluidflower.phase_training import logistic, trainer as t
from darsia.fluidflower.phase_training.scope import partition_bundle, series_id, validate_training_series
from darsia.fluidflower.phase_validation import _model_identity, materialize_phase_validation_campaign
from darsia.fluidflower.phase_workflow import (
    build_phase_training_command, discover_phase_review_sessions, prepare_phase_training_request,
)
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.prepare_grain_gas_retraining import physics_files


EVENT_PREFIX = "PHASE_CONTINUATION_EVENT "
DEFAULT_OPTICAL_TRAINING_PROFILE = "standard-v1"
FAST_OPTICAL_TRAINING_PROFILE = "generic-co2-fast-v1"
OPTICAL_TRAINING_PROFILES = {
    DEFAULT_OPTICAL_TRAINING_PROFILE: {
        "adapters": ("generic", "btb-ratiometric-relative-v2"),
        "regularization_grid": (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    },
    FAST_OPTICAL_TRAINING_PROFILE: {
        "adapters": ("generic",),
        "regularization_grid": (1e-5, 1e-3, 1e-1),
    },
}


def event(stage, **values):
    print(EVENT_PREFIX + json.dumps(dict(stage=stage, **values)), flush=True)


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def record_key(run, field, item):
    identifier = item.get("id") or item.get("proposal_id")
    return (str(run).casefold(), field, identifier) if identifier else None


def revision_time(item):
    value = (item.get("updated_utc") or item.get("reclassified_utc")
             or item.get("answered_utc") or item.get("created_utc"))
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except ValueError:
        return None


def evidence_signature(item, field):
    kind = "manual" if field == "annotations" else "quiz"
    return json.dumps(dict(phase_class=t.canonical_phase_class(item, kind),
        confidence=item.get("confidence"), **{key: item.get(key) for key in (
            "frame_index", "elapsed_h", "shape", "x", "y", "width", "height", "points")}), sort_keys=True)


def merge_records(documents, *, unresolved_conflicts=None):
    """Latest captured revision of an ID replaces its older captured revisions.

    Original bundle bytes are preserved separately. Distinct IDs retain their
    geometry and are resolved by the production trainer's existing rules.
    """
    latest, values, ambiguous = {}, {}, set()
    for index, document in enumerate(documents):
        for run, decision in document.get("decisions", {}).items():
            for field in ("annotations", "phase_quiz_answers"):
                for position, item in enumerate(decision.get(field, [])):
                    key = record_key(run, field, item)
                    if key is not None:
                        if key in values:
                            old_time, new_time = revision_time(values[key]), revision_time(item)
                            different = evidence_signature(values[key], field) != evidence_signature(item, field)
                            if different and (old_time is None or new_time is None or old_time == new_time):
                                if unresolved_conflicts is None:
                                    raise ValueError(f"Conflicting revisions of annotation {key[2]} lack a trustworthy revision order")
                                ambiguous.add(key)
                            if old_time is not None and (new_time is None or new_time < old_time):
                                continue
                        latest[key], values[key] = (index, position), item
    result, replaced = [], []
    for index, document in enumerate(documents):
        updated = copy.deepcopy(document)
        for run, decision in updated.get("decisions", {}).items():
            for field in ("annotations", "phase_quiz_answers"):
                retained = []
                for position, item in enumerate(decision.get(field, [])):
                    key = record_key(run, field, item)
                    if key in ambiguous:
                        unresolved_conflicts.append(dict(run=run, source=field, id=key[2],
                            document_index=index, annotation=copy.deepcopy(item),
                            reason="Conflicting revisions have no trustworthy revision order; deferred for human review"))
                        continue
                    if key is not None and latest[key] != (index, position):
                        replaced.append(dict(run=run, source=field, id=key[2], old_bundle=index,
                                             selected_bundle=latest[key][0]))
                        continue
                    retained.append(item)
                if field in decision:
                    decision[field] = retained
        result.append(updated)
    return result, replaced


def annotation_origins(original_documents, aligned_documents, merged_documents, regions, captured):
    """Bind queue revision fingerprints to their exact pre-alignment records.

    Timeline alignment changes frame_index and may add original_frame_index.
    Those display/training coordinates must not replace the fingerprint of the
    source record that a later continuation verifies before timeline alignment.
    """
    from darsia.fluidflower.phase_evidence_audit import payload_hash
    effective = {region.region_id: region for region in regions}
    result = {}
    def identifier(run, kind, item):
        value = item.get("id") or item.get("proposal_id") or t._geometry_key(run, item)[:16]
        return f"{run}:{kind}:{value}"
    for index, (original, aligned, merged) in enumerate(zip(original_documents, aligned_documents, merged_documents, strict=True)):
        for run, decision in merged.get("decisions", {}).items():
            name = str(run).lower()
            for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                raw_items = original["decisions"][run].get(field, [])
                aligned_items = aligned["decisions"][run].get(field, [])
                candidates = list(zip(raw_items, aligned_items, strict=True))
                for retained in decision.get(field, []):
                    key = identifier(name, kind, retained)
                    if key not in effective:
                        continue
                    matches = [raw for raw, transformed in candidates if transformed == retained]
                    if not matches:
                        raise ValueError("A selected annotation has no exact pre-alignment source")
                    raw = matches[-1]
                    if key in result:
                        if result[key]["original_annotation"] == raw:
                            continue
                        raise ValueError("An effective annotation has multiple selected source records")
                    result[key] = dict(original_annotation=copy.deepcopy(raw),
                        original_annotation_sha256=payload_hash(raw),
                        original_region_id=identifier(name, kind, raw),
                        aligned_annotation_sha256=payload_hash(effective[key].payload),
                        document_index=index, collection=field,
                        source_decisions=captured[index]["decisions"],
                        source_decisions_sha256=captured[index]["decisions_sha256"],
                        source_manifest_sha256=captured[index]["manifest_sha256"],
                        representation="captured source after sampling-path resolution, before native timeline/geometry alignment")
    if set(result) != set(effective):
        raise ValueError("Could not bind every effective annotation to its selected captured source")
    return dict(format="phase-annotation-prealignment-origins-v1", regions=result,
                original_user_documents_modified=False)


def lineage(parent):
    """Resolve explicitly recorded parents; never infer a series from a name."""
    models, visited, pending = {}, set(), [parent]
    while pending:
        current = resolve_review_path(pending.pop())
        if current in visited or not (current / "summary.json").is_file():
            continue
        if len(visited) >= 64:
            raise ValueError("Parent lineage is unexpectedly large")
        visited.add(current)
        model_id, digest = _model_identity(current)
        runtime = phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(current))
        known = models.get(model_id)
        if known and (known["model_sha256"], known["physics_runtime"]) != (digest, runtime):
            raise ValueError(f"Conflicting immutable model identities in parent lineage: {model_id}")
        models[model_id] = dict(root=str(current), model_sha256=digest, physics_runtime=runtime)
        request_path = current / "training_request.json"
        if not request_path.is_file():
            continue
        settings = read(request_path).get("settings", {})
        pending.extend(settings[key] for key in ("parent_model", "evidence_parent_model", "physics_parent_model")
                       if settings.get(key))
    return models


def native_source(job):
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root
    run = str(job["run"]).lower()
    root = resolve_review_path(analysis_review_root(resolve_review_path(job.get("analysis_source_review_root") or job["review_output_dir"]), run))
    path = root / "native_frames" / run / "manifest.json"
    return root, path, read(path)


def _has_effective_labels(decision):
    return any(t.canonical_phase_class(item, source) not in (None, "ignored")
        for field, source in (("annotations", "manual"), ("phase_quiz_answers", "quiz"))
        for item in decision.get(field, []))


def _exact_prepared_timeline_binding(original_job, target_job):
    """Prove metadata-only rebinding preserves every optical/temporal input."""
    original_root, original_path, original = native_source(original_job)
    target_root, target_path, target = native_source(target_job)
    original_hash, target_hash = sha256_file(original_path), sha256_file(target_path)
    if original_hash == target_hash:
        return None
    for key in ("dataset", "run", "canvas_wh", "content_box_xywh", "analysis_image_shape_hw",
                "correction_contract", "reference_registration", "geometry_revision_id",
                "additional_spatial_resampling", "full_analysis_resolution"):
        if original.get(key) != target.get(key):
            raise ValueError(f"Promoted timeline changes its inherited {key}; exact optical equivalence is required")
    for key in ("kind", "timeline_frame_index"):
        if (original.get("reference") or {}).get(key) != (target.get("reference") or {}).get(key):
            raise ValueError("Promoted timeline changes its inherited baseline reference semantics")
    old_frames, new_frames = original["frame_sources"], target["frame_sources"]
    count = original.get("frame_count")
    if type(count) is not int or count < 1 or count != target.get("frame_count") or len(old_frames) != count or len(new_frames) != count:
        raise ValueError("Promoted timeline must retain every inherited frame in the same order")
    run = str(original_job["run"]).lower()
    files, frames = [], []
    def same_bytes(old_path, new_path, kind):
        old_sha, new_sha = sha256_file(old_path), sha256_file(new_path)
        if old_sha != new_sha:
            raise ValueError(f"Promoted timeline changes inherited {kind} bytes")
        files.append(dict(kind=kind, source_path=str(old_path), target_path=str(new_path), sha256=old_sha))
        return old_sha
    for index, (old, new) in enumerate(zip(old_frames, new_frames, strict=True)):
        old_time, new_time = float(old["elapsed_h"]), float(new["elapsed_h"])
        if (old.get("frame_index") != index or new.get("frame_index") != index
                or not old.get("source_name") or old.get("source_name") != new.get("source_name")
                or not math.isfinite(old_time) or not math.isfinite(new_time) or old_time != new_time):
            raise ValueError("Promoted timeline changes inherited frame identity, order or time")
        digest = same_bytes(original_root / "native_frames" / run / f"frame_{index:03d}.jpg",
                            target_root / "native_frames" / run / f"frame_{index:03d}.jpg", f"frame {index}")
        frames.append(dict(frame_index=index, elapsed_h=old_time, source_name=old["source_name"], jpeg_sha256=digest))
    baseline_sha = same_bytes(original_root / "native_references" / run / "pre_injection.jpg",
                              target_root / "native_references" / run / "pre_injection.jpg", "baseline")
    if sha256_file(original_path) != original_hash or sha256_file(target_path) != target_hash:
        raise ValueError("Native metadata changed while proving timeline equivalence")
    return dict(format="exact-optical-timeline-equivalence-v1", run=run,
        source_manifest=str(original_path), source_manifest_sha256=original_hash,
        target_manifest=str(target_path), target_manifest_sha256=target_hash,
        source_native_metadata=original, target_native_metadata=target, frames=frames, files=files,
        baseline_sha256=baseline_sha, all_corrected_images_and_baseline_byte_identical=True,
        frame_order_times_and_previous_image_history_unchanged=True,
        purpose="Bind own prepared physical context for previously trained run without registered physics",
        original_sources_and_annotations_modified=False)


def _continuation_geometry_binding(original_path, original, target_path, target):
    """Resolve an exact old-to-new receipt without guessing from canvas size."""
    from darsia.fluidflower.tools import phase_training_geometry_revision as geometry

    original_hash, target_hash = sha256_file(original_path), sha256_file(target_path)
    old_revision, new_revision = original.get("geometry_revision_id"), target.get("geometry_revision_id")
    if old_revision == new_revision:
        if original_hash != target_hash:
            raise ValueError("Same geometry revision has different native manifests; an explicit mapping is required")
        return None
    if not new_revision or target.get("source_manifest_sha256") != original_hash:
        raise ValueError("Review geometry has no exact source-manifest binding to the selected parent's revision")
    frames = target.get("frame_sources") or []
    frame_path = Path(str(frames[0].get("frame_path") or "")) if frames else Path()
    if not frame_path.is_absolute():
        raise ValueError("Registered native manifest has no absolute revision source")
    revision_root = frame_path.parent.resolve()
    revision_path = revision_root / "manifest.json"
    snapshot_path = revision_root / "source_manifest_snapshot.json"
    receipt_path = revision_root / "migration_receipts.json"
    if sha256_file(revision_path) != target_hash or sha256_file(snapshot_path) != original_hash:
        raise ValueError("Native geometry revision or its original manifest snapshot changed")
    status, receipts = read(revision_root / "status.json"), read(receipt_path)
    if (not status.get("complete") or not status.get("baseline_exact")
            or status.get("manifest_sha256") != target_hash
            or receipts.get("geometry_revision_id") != new_revision):
        raise ValueError("Native geometry revision is incomplete or has inconsistent receipts")
    run = str(target.get("run") or "").lower()
    bindings = geometry.migration._bind_receipts(receipts, {run: original}, {run: target},
        {run: original_hash}, {run: target_hash})
    if set(bindings) != {(run, int(row["frame_index"])) for row in original["frame_sources"]}:
        raise ValueError("Geometry receipts do not cover the complete original timeline")
    return dict(bindings=bindings, geometry_revision_id=new_revision,
        source_manifest=str(original_path), source_manifest_sha256=original_hash,
        target_manifest=str(target_path), target_manifest_sha256=target_hash,
        source_revision_root=str(revision_root), receipt_path=str(receipt_path),
        receipt_sha256=sha256_file(receipt_path))


def align_inputs(parent, selected, documents, manifests, origins, *, capture_root=None,
                 promoted_targets=()):
    """Pin source ownership to the selected parent's registered image timelines."""
    from darsia.fluidflower.phase_evidence_audit import superseded_evidence_ids
    evidence = []
    for document in documents:
        for run, decision in document.get("decisions", {}).items():
            for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                for item in decision.get(field, []):
                    identifier = item.get("id") or item.get("proposal_id") or t._geometry_key(run.lower(), item)[:16]
                    evidence.append((f"{run.lower()}:{kind}:{identifier}", item))
    superseded = superseded_evidence_ids(evidence)
    targets = {}
    for manifest, origin in zip(manifests, origins):
        if origin == "parent":
            targets.update({str(job["run"]).lower(): job for job in manifest["jobs"]})
    source_path = parent / "review_source/postflight_manifest.json"
    if not source_path.is_file():
        source_path = parent / "validation/postflight_manifest.json"
    selected_manifest = read(source_path)
    profile = phase_runtime.load_physics_profile(parent)
    for job in selected_manifest["jobs"]:
        if series_id(job["dataset"]) != selected:
            continue
        _, native, _ = native_source(job)
        if profile is not None and sha256_file(native) != profile["runs"][f'{selected}/{job["run"]}']["native_manifest_sha256"]:
            raise ValueError("Selected parent's review timeline differs from its frozen physical context")
        targets[str(job["run"]).lower()] = job
    # An explicit promotion may add a new experiment. Existing parent timelines
    # remain canonical, including when a new evaluation used a sparse subset.
    additions, parity_runs, parity_records = {}, set(), []
    for promotion in promoted_targets:
        target = promotion["target"]
        name = str(target["run"]).lower()
        if name in additions and additions[name] != target["native_manifest_sha256"]:
            raise ValueError(f"Promotions select different native revisions of {name}")
        additions[name] = target["native_manifest_sha256"]
        physical_key = f"{selected}/{name}"
        if name in targets and profile is not None and physical_key not in profile.get("runs", {}):
            # Historical training evidence is not itself a physical registration.
            # Rebind only after proving the new metadata describes the very same
            # full image sequence and baseline used by every inherited document.
            checked_paths = set()
            for source_manifest in manifests:
                for source_job in source_manifest["jobs"]:
                    if str(source_job["run"]).lower() != name:
                        continue
                    source_path = native_source(source_job)[1]
                    if source_path in checked_paths:
                        continue
                    checked_paths.add(source_path)
                    binding = _exact_prepared_timeline_binding(source_job, target["job"])
                    if binding is not None:
                        parity_records.append(binding)
            if parity_records and capture_root is None:
                raise ValueError("Timeline equivalence requires a detached capture_root for its receipt")
            targets[name] = copy.deepcopy(target["job"])
            parity_runs.add(name)
        elif name not in targets:
            targets[name] = copy.deepcopy(target["job"])
    if parity_records:
        artifact_root = Path(capture_root).resolve()
        protected = [parent, *(Path(row[key]).parent for row in parity_records
                               for key in ("source_manifest", "target_manifest"))]
        if any(artifact_root == Path(path).resolve() or artifact_root.is_relative_to(Path(path).resolve()) for path in protected):
            raise ValueError("Timeline equivalence receipt must be outside immutable model and native assets")
        write(artifact_root / "timeline_equivalence.json", dict(format="continuation-timeline-equivalence-v1",
            parent_model=str(parent), bindings=parity_records, only_previously_unregistered_physical_runs=True,
            source_documents_modified=False, native_images_modified=False))
    aligned, bound_manifests, legacy_warnings = [], [], []
    geometry_bindings, geometry_records = {}, []
    for document_index, (document, manifest, origin) in enumerate(zip(documents, manifests, origins)):
        result, bound = copy.deepcopy(document), copy.deepcopy(manifest)
        jobs = {str(job["run"]).lower(): job for job in manifest["jobs"]}
        for run, decision in result["decisions"].items():
            name = run.lower()
            if name not in targets:
                if origin == "collection_review" and not _has_effective_labels(decision):
                    # The normal session may list an untouched extra experiment.
                    # Preserve that record without inventing a training context.
                    continue
                raise ValueError(f"No parent-registered training timeline for {run}")
            _, original_path, original = native_source(jobs[name])
            target_root, target_path, target = native_source(targets[name])
            geometry = None
            if original.get("geometry_revision_id") or target.get("geometry_revision_id"):
                identity = (str(original_path), str(target_path))
                if identity not in geometry_bindings:
                    geometry_bindings[identity] = _continuation_geometry_binding(
                        original_path, original, target_path, target)
                geometry = geometry_bindings[identity]
                if geometry is not None:
                    if capture_root is None:
                        raise ValueError("Geometry migration requires a detached capture_root for its audit")
                    artifact_root = Path(capture_root).resolve()
                    if any(artifact_root.is_relative_to(Path(protected).resolve()) for protected in
                           (parent, original_path.parent, target_root, geometry["source_revision_root"])):
                        raise ValueError("Geometry migration audit must be outside parent and source assets")
                    if decision.get("reference_guides"):
                        raise ValueError("Reference guides require a separate geometry mapping")
                    if decision.get("frame_choices") and any(index != row["binding"]["target_frame_index"]
                            for (mapped_run, index), row in geometry["bindings"].items() if mapped_run == name):
                        raise ValueError("Geometry timeline change requires explicit frame-verdict mapping")
            for key in ("canvas_wh", "content_box_xywh"):
                if original.get(key) != target.get(key):
                    raise ValueError(f"Review geometry differs from the selected parent's registered frame: {run}")
            for field, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                for position, item in enumerate(decision.get(field, [])):
                    if geometry is not None:
                        from darsia.fluidflower.tools import phase_training_geometry_revision as geometry_revision
                        if item.get("geometry_migration"):
                            raise ValueError("Previously migrated evidence cannot be translated a second time")
                        frame = item.get("frame_index")
                        if type(frame) is not int or (name, frame) not in geometry["bindings"]:
                            raise ValueError("Reviewed region has no exact frame geometry receipt")
                        source_frame = geometry["bindings"][(name, frame)]["old_frame"]
                        elapsed = float(item.get("elapsed_h", source_frame["elapsed_h"]))
                        if abs(elapsed - float(source_frame["elapsed_h"])) > 1e-5:
                            raise ValueError("Reviewed region time differs from its geometry receipt")
                        # Anonymous historical regions retain the trainer's original
                        # geometry-derived identity after their coordinates move.
                        if not (item.get("id") or item.get("proposal_id")):
                            item["id"] = t._geometry_key(name, item)[:16]
                        item, record = geometry_revision.translate_training_annotation(
                            item, geometry["bindings"][(name, frame)])
                        item["geometry_migration"]["geometry_revision_id"] = geometry["geometry_revision_id"]
                        decision[field][position] = item
                        geometry_records.append(dict(document_index=document_index, run=name, collection=field,
                            id=item.get("id") or item.get("proposal_id"),
                            source_manifest_sha256=geometry["source_manifest_sha256"],
                            target_manifest_sha256=geometry["target_manifest_sha256"],
                            geometry_revision_id=geometry["geometry_revision_id"], **record))
                    phase_class = t.canonical_phase_class(item, kind)
                    identifier = item.get("id") or item.get("proposal_id") or t._geometry_key(name, item)[:16]
                    # Keep candidate-issue records aligned for revision merging
                    # and evaluation, even though V4 omits them from fitting.
                    legacy_candidate_issue = (kind == "manual" and
                        str(item.get("category") or "").strip().lower()
                        in logistic.CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES)
                    if (phase_class is None or (phase_class == "ignored" and not legacy_candidate_issue)
                            or f"{name}:{kind}:{identifier}" in superseded):
                        continue
                    if item.get("frame_index") is None:
                        continue
                    frame = int(item["frame_index"])
                    item_source = target if geometry is not None else original
                    if not 0 <= frame < len(item_source["frame_sources"]):
                        raise ValueError(f"Review frame outside registered timeline: {run}/{frame}")
                    elapsed = float(item.get("elapsed_h", item_source["frame_sources"][frame]["elapsed_h"]))
                    if abs(elapsed - float(item_source["frame_sources"][frame]["elapsed_h"])) > 1e-5:
                        if origin == "parent" and item_source["frame_sources"] == target["frame_sources"]:
                            legacy_warnings.append(dict(run=name, frame=frame, id=identifier,
                                annotation_elapsed_h=elapsed, native_elapsed_h=original["frame_sources"][frame]["elapsed_h"],
                                action="Preserved unchanged inherited record on its identical registered image"))
                        else:
                            raise ValueError(f"Review time differs from registered source frame: {run}/{frame}")
                    region = t.ReviewedRegion(region_id=str(item.get("id") or item.get("proposal_id") or position),
                        run=name, dataset=selected, frame_index=frame, elapsed_h=elapsed,
                        phase_class=phase_class,
                        confidence=str(item.get("confidence") or "low"), source=kind,
                        reviewer=str(item.get("reviewer") or "unknown"), payload=item)
                    decision[field][position] = t._align_region_timeline(region, item_source, target).payload
            for job in bound["jobs"]:
                if str(job["run"]).lower() == name:
                    if origin in {"promotion", "collection_review", "reclassification"} or name in parity_runs:
                        # Do not leave a sparse evaluation's context descriptor
                        # attached to the parent's aligned canonical timeline.
                        job.clear()
                        job.update(copy.deepcopy(targets[name]))
                    job["review_output_dir"] = str(target_root)
                    job["analysis_source_review_root"] = str(target_root)
                    if target.get("geometry_revision_id"):
                        job["geometry_revision_id"] = target["geometry_revision_id"]
                        job["geometry_source_manifest_sha256"] = sha256_file(target_path)
        aligned.append(result)
        bound_manifests.append(bound)
    if any(value is not None for value in geometry_bindings.values()):
        audit_path = Path(capture_root) / "geometry_alignment.json"
        write(audit_path, dict(format="continuation-geometry-alignment-v1", parent_model=str(parent),
            bindings=[{key: value for key, value in bound.items() if key != "bindings"}
                      for bound in geometry_bindings.values() if bound is not None],
            migrated_records=len(geometry_records), records=geometry_records,
            source_documents_modified=False, native_images_copied=False))
    return aligned, bound_manifests, legacy_warnings


def capture_inputs(parent, selected, capture, sessions, *, withdrawal_snapshot=None,
                   preserve_native_revisions=False):
    from darsia.fluidflower import phase_review_retractions as withdrawals
    withdrawal_snapshot = (withdrawals.load_retractions(series=selected) if withdrawal_snapshot is None
                           else withdrawals.validate_snapshot(withdrawal_snapshot))
    previous = read(parent / "training_request.json")
    summary = read(parent / "summary.json")
    datasets = {series_id(value) for value in summary["inventory"]["datasets"]}
    if selected not in datasets:
        raise ValueError("Selected model does not cover the requested experimental series")
    family = lineage(parent)
    from darsia.fluidflower.tools.phase_training_promotions import validated_promotions
    promotions = validated_promotions(parent, selected, family)
    inputs, captured, seen = [], [], set()
    promoted_inputs, collection_inputs, replacement_inputs = {}, {}, {}
    for bundle in previous["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            row = bundle[role]
            path = Path(row["snapshot_path"]).resolve()
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"Parent evidence changed: {path}")
            pair.append(path)
        inputs.append((pair, "parent", None))
    # Review ordering follows saved modification order. A later captured revision
    # can update the same annotation ID without retaining its stale geometry.
    for session in sorted(sessions, key=lambda row: float(row.get("modified_timestamp") or 0)):
        if session.get("evidence_role") == "blind_transfer_validation" or session.get("include_in_training") is False:
            continue
        model_id = session.get("target_model_id")
        if model_id not in family:
            continue
        decisions = resolve_review_path(session.get("decision_path") or "", context=session.get("session_path"))
        if not decisions.is_file():
            continue
        manifest = Path(session["combined_root"]) / "postflight_manifest.json"
        value = read(manifest)
        source = family[model_id]
        if value.get("format") == "fluidflower-phase-review-collection-v1":
            from darsia.fluidflower.tools.phase_review_collections import validated_collection
            if (value.get("source_model_id"), value.get("source_model_sha256"), value.get("source_physics_runtime")) != (
                    model_id, source["model_sha256"], source["physics_runtime"]):
                raise ValueError(f"Review collection is not bound to its recorded source model: {decisions}")
            checked = validated_collection(manifest, Path(source["root"]), full=True)
            key = (str(decisions.resolve()), str(manifest.resolve()))
            collection_inputs[key] = dict(checked, manifest_sha256=sha256_file(manifest))
            kind = "collection_review"
        else:
            if (value.get("model_id"), value.get("model_sha256"), value.get("physics_runtime")) != (
                    model_id, source["model_sha256"], source["physics_runtime"]):
                raise ValueError(f"Review is not bound to its recorded model: {decisions}")
            kind = "review"
        inputs.append(((decisions, manifest), kind, session.get("session_path")))
    for promotion in promotions:
        decisions = resolve_review_path(promotion["decision_path"], context=promotion.get("session", {}).get("session_path"))
        manifest = Path(promotion["combined_root"]) / "postflight_manifest.json"
        key = (str(decisions.resolve()), str(manifest.resolve()))
        promoted_inputs[key] = promotion
        inputs.append(((decisions, manifest), "promotion", promotion.get("session", {}).get("session_path")))
    for event, decisions, manifest, receipt in withdrawals.materialize_replacements(
            withdrawal_snapshot, capture / "evidence_actions"):
        replacement = event["replacement"]
        if replacement["series"] != selected:
            continue
        job = replacement["job"]
        descriptor = job.get("phase_transfer_context") or {}
        binding = (descriptor.get("provenance") or {}).get("target_binding") or {}
        source_manifest = binding.get("source_manifest") or replacement["combined_manifest"]["path"]
        target = dict(dataset=selected, run=replacement["run"], job=copy.deepcopy(job),
            native_manifest=replacement["native_manifest"]["path"],
            native_manifest_sha256=replacement["native_manifest"]["sha256"],
            source_manifest=source_manifest,
            review_root=job["analysis_source_review_root"],
            baseline=str(Path(job["analysis_source_review_root"]) / "native_references" / replacement["run"] / "pre_injection.jpg"))
        combined = replacement["combined_manifest_data"]
        selection = dict(target=target, receipt_path=str(receipt), receipt_sha256=sha256_file(receipt),
            source_model_id=combined.get("source_model_id") or combined.get("model_id") or job.get("candidate_id"),
            source_model_sha256=combined.get("source_model_sha256") or combined.get("model_sha256"),
            evaluation_root=str(Path(replacement["combined_manifest"]["path"]).parent),
            decision_path=str(decisions), combined_root=str(manifest.parent),
            selection_kind="explicit_historical_reclassification")
        replacement_inputs[(str(decisions.resolve()), str(manifest.resolve()))] = selection
        inputs.append(((decisions, manifest), "reclassification", None))
    pairs, documents, manifests, origins = [], [], [], []
    captured_promotions, captured_collections, inherited_run_names = [], [], set()
    for (decision_path, manifest_path), kind, session_path in inputs:
        decision_bytes, manifest_bytes = decision_path.read_bytes(), manifest_path.read_bytes()
        import hashlib
        digests = (hashlib.sha256(decision_bytes).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest())
        if digests in seen:
            continue
        seen.add(digests)
        decisions, manifest = partition_bundle(json.loads(decision_bytes.decode("utf-8-sig")),
            json.loads(manifest_bytes.decode("utf-8-sig")), selected)
        from darsia.fluidflower.phase_training.gas_negative_sampling import resolve_document_sources
        decisions = resolve_document_sources(decisions, Path(decision_path).parent)
        if not decisions["decisions"]:
            continue
        if kind == "parent":
            inherited_run_names.update(str(run).lower() for run in decisions["decisions"])
        input_key = (str(decision_path.resolve()), str(manifest_path.resolve()))
        collection = collection_inputs.get(input_key) if kind == "collection_review" else None
        collection_selections = []
        replacement_selection = replacement_inputs.get(input_key)
        if replacement_selection is not None:
            collection_selections.append(replacement_selection)
        if collection is not None:
            if digests[1] != collection["manifest_sha256"]:
                raise ValueError("Review collection changed during evidence capture")
            by_run = {str(run).lower(): decision for run, decision in decisions["decisions"].items()}
            for selection in collection.get("context_selections", []):
                target = selection["target"]
                if target["dataset"] != selected:
                    raise ValueError("Review collection context belongs to another experimental series")
                name = str(target["run"]).lower()
                if name not in by_run or not _has_effective_labels(by_run[name]):
                    continue
                if selection.get("selection_kind") != "normal_visible_review":
                    raise ValueError("Review collection has an invalid context-selection policy")
                row = dict(selection, decision_path=str(decision_path), combined_root=str(manifest_path.parent))
                row["receipt_sha256"] = row.get("receipt_sha256") or row.get("sha256")
                if not row["receipt_sha256"] or sha256_file(Path(row["receipt_path"])) != row["receipt_sha256"]:
                    raise ValueError("Review collection receipt changed during evidence capture")
                collection_selections.append(row)
        promotion = promoted_inputs.get((str(decision_path.resolve()), str(manifest_path.resolve()))) if kind == "promotion" else None
        if promotion is not None:
            name = str(promotion["target"]["run"]).lower()
            if set(decisions["decisions"]) != {name}:
                raise ValueError("Promoted decisions contain an experiment outside the explicit receipt")
            decision = decisions["decisions"][name]
            if not _has_effective_labels(decision):
                continue
            receipt_path = Path(promotion["receipt_path"])
            if sha256_file(receipt_path) != promotion["receipt_sha256"]:
                raise ValueError("Explicit training promotion changed during evidence capture")
            manifest = copy.deepcopy(manifest)
            manifest.update(include_in_training=True, evidence_role="explicitly_promoted_training",
                requires_explicit_training_selection=False,
                training_promotion=dict(path=str(receipt_path),sha256=promotion["receipt_sha256"]))
            manifest["jobs"] = [copy.deepcopy(promotion["target"]["job"])]
        index = len(documents)
        # Preserve each original input byte-for-byte before creating a derived
        # deduplicated training view. No review_session.json is copied here.
        folder = capture / "originals" / str(index)
        folder.mkdir(parents=True, exist_ok=False)
        for name, data in (("decisions.json", decision_bytes), ("manifest.json", manifest_bytes)):
            (folder / name).write_bytes(data)
        omitted_collection_runs = []
        if collection is not None:
            # Empty extra rows are part of normal review navigation, not training
            # sources. Keep their verdict/audit metadata in the original snapshot;
            # otherwise an untouched extra can break a single-run fit. Inherited
            # runs retain even non-training records for supersession resolution.
            extras = {str(row["target"]["run"]).lower() for row in collection.get("context_selections", [])}
            omitted_collection_runs = [run for run, decision in decisions["decisions"].items()
                if str(run).lower() in extras and str(run).lower() not in inherited_run_names
                and not _has_effective_labels(decision)]
            if omitted_collection_runs:
                decisions = copy.deepcopy(decisions)
                decisions["decisions"] = {run: decision for run, decision in decisions["decisions"].items()
                                         if run not in omitted_collection_runs}
        if promotion is not None:
            shutil.copy2(receipt_path, folder / "promotion_receipt.json")
            if sha256_file(folder / "promotion_receipt.json") != promotion["receipt_sha256"]:
                raise ValueError("Training promotion snapshot differs from its receipt")
            captured_promotions.append(dict(promotion, document_index=index,
                captured_receipt_path=str(folder / "promotion_receipt.json")))
        copied_receipts = {}
        for selection in collection_selections:
            receipt = Path(selection["receipt_path"])
            identity = (str(receipt.resolve()), selection["receipt_sha256"])
            if identity not in copied_receipts:
                destination = folder / f"collection_receipt_{len(copied_receipts):03d}.json"
                shutil.copy2(receipt, destination)
                if sha256_file(destination) != selection["receipt_sha256"]:
                    raise ValueError("Review collection snapshot differs from its receipt")
                copied_receipts[identity] = destination
            captured_collections.append(dict(selection, document_index=index,
                captured_receipt_path=str(copied_receipts[identity])))
        captured.append(dict(kind=kind, session_path=session_path, decisions=str(decision_path),
            combined_manifest=str(manifest_path), decisions_sha256=digests[0], manifest_sha256=digests[1],
            omitted_unreviewed_collection_runs=omitted_collection_runs))
        documents.append(decisions)
        manifests.append(manifest)
        origins.append(kind)
    original_documents = documents
    # Filter all copies before newest-revision merging, so an older parent copy
    # cannot resurrect a withdrawn annotation. Original bytes above stay intact.
    withdrawal_audit = []
    filtered_documents = []
    for index, (document, manifest) in enumerate(zip(documents, manifests)):
        filtered, removed = withdrawals.filter_document(document, manifest, withdrawal_snapshot)
        filtered_documents.append(filtered)
        withdrawal_audit.extend(dict(row, document_index=index) for row in removed)
    documents = filtered_documents
    original_documents = documents  # pre-alignment view; immutable original bytes are already captured
    write(capture / "review_retractions.json", withdrawal_snapshot)
    write(capture / "review_retraction_audit.json", dict(
        snapshot_digest=withdrawal_snapshot["digest"], excluded=withdrawal_audit,
        original_documents_unchanged=True))
    captured_context_selections = [*captured_promotions, *captured_collections]
    if preserve_native_revisions:
        # A spatial trainer can extract each annotation on its own native
        # revision, using annotation_origins.document_index as the binding.
        # It MUST NOT use the last RunSource per run as an image lookup. This
        # avoids moving boxes between different corrections or timelines.
        legacy_warnings = []
    else:
        documents, manifests, legacy_warnings = align_inputs(parent, selected, documents, manifests, origins,
            capture_root=capture, promoted_targets=captured_context_selections)
    unresolved = [] if preserve_native_revisions else None
    merged, replaced = merge_records(documents, unresolved_conflicts=unresolved)
    merged_before_geometry_rebind = merged
    geometry_audit_path = capture / "geometry_alignment.json"
    if geometry_audit_path.is_file():
        from darsia.fluidflower.tools.phase_training_geometry_revision import rebind_evidence_audits
        merged, supersession_changes = rebind_evidence_audits(original_documents, merged)
        write(capture / "geometry_supersession_alignment.json", dict(
            format="continuation-geometry-supersession-v1", changes=supersession_changes,
            human_verdicts_unchanged=True, source_documents_modified=False))
    for index, (decisions, manifest) in enumerate(zip(merged, manifests)):
        folder = capture / "training_view" / str(index)
        write(folder / "decisions.json", decisions)
        write(folder / "manifest.json", manifest)
        pairs.append((folder / "decisions.json", folder / "manifest.json"))
    label_policy_audit = {}
    sources, regions, conflicts = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs],
        preserve_native_revisions=preserve_native_revisions,
        label_policy_audit=label_policy_audit)
    label_policy_path = capture / "training_label_policy_audit.json"
    write(label_policy_path, label_policy_audit)
    validate_training_series(sources, regions, selected)
    if not regions:
        raise ValueError("No effective reviewed regions for this experimental series")
    origin_path = capture / "annotation_origins.json"
    write(origin_path, annotation_origins(original_documents, documents, merged_before_geometry_rebind, regions, captured))
    active_runs = {row.run for row in regions if row.supervision_weight > 0}
    promotion_audit, collection_audit = [], []
    for promotion in captured_context_selections:
        name = str(promotion["target"]["run"]).lower()
        if name in active_runs:
            effective_job = next(job for job in manifests[promotion["document_index"]]["jobs"]
                                 if str(job["run"]).lower() == name)
            row = {key:promotion[key] for key in (
                "receipt_path", "receipt_sha256", "captured_receipt_path", "source_model_id",
                "source_model_sha256", "evaluation_root", "target", "decision_path", "combined_root")}
            row.update(effective_job=effective_job, selection_kind=promotion.get("selection_kind", "explicit_promotion"))
            (collection_audit if row["selection_kind"] == "normal_visible_review" else promotion_audit).append(row)
    return previous, pairs, regions, dict(captured_sources=captured, replaced_duplicate_ids=replaced,
        native_revision_policy="per-annotation-origin" if preserve_native_revisions else "parent-aligned",
        deferred_revision_conflicts=unresolved or [],
        review_retractions=dict(snapshot_digest=withdrawal_snapshot["digest"], excluded=withdrawal_audit,
                                snapshot_path=str(capture / "review_retractions.json")),
        annotation_origins=dict(path=str(origin_path), sha256=sha256_file(origin_path)),
        training_label_policy=dict(path=str(label_policy_path),
                                   sha256=sha256_file(label_policy_path),
                                   **label_policy_audit),
        effective_regions=len(regions), classes=dict(Counter(row.phase_class for row in regions)),
        runs=sorted({row.run for row in regions}), conflicts=conflicts,
        inherited_timeline_warnings=legacy_warnings,
        geometry_alignment_audit=str(geometry_audit_path) if geometry_audit_path.is_file() else None,
        timeline_equivalence_audit=str(capture / "timeline_equivalence.json") if (capture / "timeline_equivalence.json").is_file() else None,
        family_models=family, explicit_promotions=promotion_audit,
        collection_context_selections=collection_audit,
        live_reviews_modified=False, source_documents_preserved=True)


def _profile_without_registries(profile):
    from darsia.fluidflower.tools.phase_transfer_context import SUPPORT_KEYS
    result = copy.deepcopy(profile)
    def visit(value):
        if not isinstance(value, dict):
            raise ValueError("Invalid nested frozen physics profile")
        if "format" in value and isinstance(value.get("runs"), dict):
            value.pop("runs")
        for name in SUPPORT_KEYS:
            if name in value:
                visit(value[name])
    visit(result)
    return result


def register_promoted_contexts(root, parent, selected, review_manifest, promotions):
    """Extend only an unpublished continuation's registries and review timeline."""
    from darsia.fluidflower.tools.phase_transfer_context import prepared_target_physics, PROFILE_FILES, SUPPORT_KEYS
    from darsia.fluidflower.tools.phase_transfer_review import registered_target_physics
    from darsia.fluidflower.phase_plume import freeze_context
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Promotion registration must be outside the immutable parent model")
    original = phase_runtime.load_physics_profile(root)
    if original is None:
        raise ValueError("Promotion requires a copied parent physical profile")
    updated, manifest = copy.deepcopy(original), copy.deepcopy(review_manifest)
    registered, added, seen = [], [], {}
    for promotion in promotions:
        receipt = Path(promotion["receipt_path"])
        if sha256_file(receipt) != promotion["receipt_sha256"]:
            raise ValueError("Explicit training promotion changed before context registration")
        target = promotion["target"]
        if target["dataset"] != selected:
            raise ValueError("Promoted context belongs to another experimental series")
        name, key = target["run"], f"{selected}/{target['run']}"
        if target["job"].get("phase_transfer_context"):
            checked = prepared_target_physics(parent, target)
        elif key in original["runs"]:
            registered_target_physics(parent, target)
            checked = None
        else:
            raise ValueError("A new promoted experiment requires its own prepared physical context")
        if key in seen:
            if seen[key] != target["native_manifest_sha256"]:
                raise ValueError("Promotions select conflicting target revisions")
            continue
        seen[key] = target["native_manifest_sha256"]
        job = copy.deepcopy(promotion["effective_job"])
        _, native_path, _ = native_source(job)
        if key in updated["runs"]:
            if updated["runs"][key]["native_manifest_sha256"] != sha256_file(native_path):
                raise ValueError("Existing promoted run must retain its parent-registered canonical context")
        else:
            if sha256_file(native_path) != target["native_manifest_sha256"]:
                raise ValueError("New promoted timeline differs from its exact prepared context")
            entry = freeze_context(root, selected, name, checked["context"],native_manifest=native_path,
                provenance=dict(checked["provenance"], training_promotion=dict(
                    path=str(receipt),sha256=promotion["receipt_sha256"],
                    selection_kind=promotion.get("selection_kind", "explicit_promotion"),
                    source_model_id=promotion["source_model_id"],source_model_sha256=promotion["source_model_sha256"])))
            def insert(value):
                if not isinstance(value, dict):
                    raise ValueError("Invalid nested frozen physics profile")
                if "format" in value and isinstance(value.get("runs"), dict):
                    if key in value["runs"]:
                        raise ValueError("Promoted context would replace an inherited nested registry entry")
                    value["runs"][key] = copy.deepcopy(entry)
                for support in SUPPORT_KEYS:
                    if support in value:
                        insert(value[support])
            insert(updated)
            added.append(key)
        jobs = [existing for existing in manifest["jobs"] if str(existing["run"]).lower() == name]
        if jobs:
            if any(sha256_file(native_source(existing)[1]) != sha256_file(native_path) for existing in jobs):
                raise ValueError("Review timeline conflicts with the promoted canonical context")
        else:
            manifest["jobs"].append(job)
        registered.append(key)
    if _profile_without_registries(updated) != _profile_without_registries(original):
        raise ValueError("Training promotion changed frozen physics settings or calibration")
    if added:
        profiles = [root/name for name in PROFILE_FILES if (root/name).is_file()]
        if len(profiles) != 1:
            raise ValueError("Continuation must contain one copied physics profile")
        profiles[0].write_text(json.dumps(updated,indent=2,allow_nan=False),encoding="utf-8")
        if phase_runtime.load_physics_profile(root) != updated:
            raise ValueError("Promoted physical context registry did not validate")
    return manifest, dict(registered_runs=registered,new_contexts=added,
        settings_unchanged=True,calibration_unchanged=True,
        original_parent_modified=False,receipts=[dict(path=p["receipt_path"],sha256=p["receipt_sha256"]) for p in promotions])


def detach_inherited_completion(root, parent, cache):
    """Archive only the new copy's old corrector before fitting fresh optics.

    Context registration must also operate on this unwrapped profile: its loader
    attaches runtime-only checkpoint paths, which must never enter profile JSON.
    """
    from darsia.fluidflower import phase_aqueous_context_completion as completion
    from darsia.fluidflower import phase_decision as decision, phase_decision_runtime as decision_runtime
    root, parent, cache = (Path(value).resolve() for value in (root, parent, cache))
    sidecar = root / completion.PROFILE_FILE
    if not sidecar.is_file():
        return None
    workspace = (cache / "studio-continuations").resolve()
    if (not root.is_relative_to(workspace) or root == workspace or root == parent
            or root.is_relative_to(parent) or parent.is_relative_to(root)):
        raise ValueError("Only a separate new continuation may detach inherited completion")
    profile = phase_runtime.load_physics_profile(root)
    evidence_profile = profile
    decision_specification = None
    if decision_runtime.PROFILE_KEY in profile:
        evidence_profile, decision_specification = decision_runtime.split_profile(profile)
    base, specification = completion.split_profile(evidence_profile)
    checkpoint = (root / specification["checkpoint"]).resolve()
    archive = (root / "inherited_completion_before_retraining").resolve()
    if archive.exists():
        raise FileExistsError("Inherited completion was already archived")
    sources = [sidecar, checkpoint]
    snapshot = root / "completion_code_snapshot"
    if snapshot.is_dir():
        sources.append(snapshot)
    if decision_specification is not None:
        sources.append(root / decision_runtime.PROFILE_FILE)
        decision_snapshot = root / "decision_code_snapshot"
        if decision_snapshot.is_dir():
            sources.append(decision_snapshot)
    moves = []
    for source in sources:
        source = source.resolve()
        target = (archive / source.relative_to(root)).resolve()
        if (not source.is_relative_to(root) or source == root
                or not target.is_relative_to(archive) or not archive.is_relative_to(root)):
            raise ValueError("Inherited completion archive escaped the new continuation")
        moves.append((source, target))
    receipt = dict(required=True, format=completion.VERSION, parent_model=str(parent),
        inherited_runtime=phase_runtime.physics_runtime_identity(profile),
        base_runtime=phase_runtime.physics_runtime_identity(base),
        inherited_checkpoint_sha256=specification["checkpoint_sha256"],
        archive=str(archive), parent_modified=False,
        reason="Fresh optical heads require a newly fitted contextual aqueous completion stage")
    if decision_specification is not None:
        from darsia.fluidflower.phase_decision_versions import implementation
        decision=implementation(decision_specification).core
        receipt['restore_unified_decision'] = dict(
            profile_sha256=sha256_file(root / decision_runtime.PROFILE_FILE),
            specification=decision_specification,
            decision_identity=decision.identity(decision.Settings(**decision_specification['settings'])))
    archive.mkdir()
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    if phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(root)) != receipt["base_runtime"]:
        raise ValueError("Detaching completion changed the inherited base physical runtime")
    write(root / "completion_retraining.json", receipt)
    return receipt


def with_extraction_resources(settings, source_manifests, *, cpus=None, memory_gb=None,
                              extraction_workers=None, preserve_legacy_workers=False):
    """Freeze only execution settings after native source selection is complete."""
    from darsia.fluidflower.phase_execution_resources import (
        OPTICAL_EXTRACTION_RESOURCE_FORMAT, optical_extraction_resources,
    )
    result = copy.deepcopy(settings)
    previous = settings.get("optical_extraction_resources") or {}
    if previous.get("format") == OPTICAL_EXTRACTION_RESOURCE_FORMAT:
        overrides = dict(previous.get("requested_overrides") or {})
    else:
        # Historical two-arm requests have no provenance distinguishing explicit
        # smaller limits from old defaults. Never silently raise those limits.
        overrides = ({key: settings[key] for key in
                     ("stage_workers", "extraction_workers", "opencv_threads") if key in settings}
                     if preserve_legacy_workers else {})
    overrides.update({key: value for key, value in dict(cpus=cpus, memory_gb=memory_gb,
                      extraction_workers=extraction_workers).items() if value is not None})
    manifests = {Path(path).resolve() for path in source_manifests}
    shapes = []
    for path in sorted(manifests):
        manifest = read(path)
        shape = manifest.get("analysis_image_shape_hw")
        if shape is None:
            shape = list(reversed(manifest.get("canvas_wh") or []))
        if len(shape) != 2 or min(map(int, shape)) < 1:
            raise ValueError(f"Native extraction source has no valid dimensions: {path}")
        shapes.append(tuple(map(int, shape)))
    if not shapes:
        raise ValueError("Optical extraction requires at least one native source")
    plan = optical_extraction_resources(max(shapes, key=lambda shape: shape[0] * shape[1]),
                                        tasks=len(manifests), **overrides)
    for key in ("stage_workers", "extraction_workers", "opencv_threads"):
        result[key] = plan[key]
    result["optical_extraction_resources"] = plan
    return result


def prepare(parent, library, cache, selected, actor, *, backend="auto", sessions=None,
            geometry_revision_root=None, feature_version=None,
            preserve_native_revisions=False,
            optical_training_profile=DEFAULT_OPTICAL_TRAINING_PROFILE,
            cpus=None, memory_gb=None, extraction_workers=None):
    parent, library, cache = (Path(value).resolve() for value in (parent, library, cache))
    selected = series_id(selected)
    if not selected or not actor.strip():
        raise ValueError("Select one experimental series and enter a reviewer name")
    if optical_training_profile not in OPTICAL_TRAINING_PROFILES:
        raise ValueError(f"Unknown optical training profile {optical_training_profile!r}")
    if cache.is_relative_to(parent) or library.is_relative_to(parent):
        raise ValueError("New versions and caches must be outside the selected model")
    previous_request = read(parent / "training_request.json")
    if previous_request.get("evaluation_only") or previous_request.get("settings", {}).get("evaluation_only"):
        raise ValueError("Select the fitted source model to train further; transfer labels require explicit selection as training evidence")
    parent_feature_version = previous_request.get("settings", {}).get("feature_version")
    # Studio supplies no override. Retain the parent's relative grain channels
    # instead of silently reverting a reviewed v5 model to the legacy default.
    selected_feature_version = feature_version or (
        parent_feature_version
        if parent_feature_version == phase_features.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION
        else phase_features.GENERIC_FEATURE_VERSION)
    optical_profile = OPTICAL_TRAINING_PROFILES[optical_training_profile]
    variants = t.model_variants(generic_version=selected_feature_version,
                                adapters=optical_profile["adapters"])
    profile = phase_runtime.load_physics_profile(parent)
    if profile is None or {key.split("/")[0] for key in profile["runs"]} != {selected}:
        raise ValueError("Train further requires a model with registered physics for this series only; select its series-specific version")
    workspace = cache / "studio-continuations" / uuid.uuid4().hex
    capture = workspace / "evidence_capture"
    from darsia.fluidflower import phase_review_retractions as withdrawals
    withdrawal_snapshot = withdrawals.load_retractions(series=selected)
    withdrawals.assert_calibration_can_be_inherited(parent, withdrawal_snapshot)
    previous, pairs, regions, audit = capture_inputs(parent, selected, capture,
        discover_phase_review_sessions() if sessions is None else sessions,
        withdrawal_snapshot=withdrawal_snapshot,
        preserve_native_revisions=preserve_native_revisions)
    context_selections = [*audit.get("explicit_promotions", []), *audit.get("collection_context_selections", [])]
    if preserve_native_revisions:
        # Historical labels for an already registered run use their own frozen
        # native source only for feature extraction. They must not replace that
        # run's canonical physical context or Studio review timeline.
        historical = [selection for selection in context_selections
                      if f"{selected}/{str(selection['target']['run']).lower()}" in profile["runs"]]
        context_selections = [selection for selection in context_selections
                              if selection not in historical]
        audit["historical_native_context_selections"] = len(historical)
    geometry_audit = None
    if geometry_revision_root is not None:
        if preserve_native_revisions:
            raise ValueError("Explicit geometry migration and original-native extraction require separate fits")
        if context_selections:
            raise ValueError("Promoted experiments and an explicit geometry revision require separate continuation preparation")
        if selected != "ffac":
            raise ValueError("The explicit registered image revision currently supports FF_AC only")
        from darsia.fluidflower.tools.phase_training_geometry_revision import migrate_training_pairs
        before_counts = Counter(row.phase_class for row in regions)
        pairs, geometry_audit = migrate_training_pairs(pairs,
            revision_root=geometry_revision_root, output_root=capture / "geometry_revision")
        _, regions, _ = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs])
        if Counter(row.phase_class for row in regions) != before_counts:
            raise ValueError("Geometry migration changed the number or meaning of reviewed regions")
        audit["geometry_revision"] = geometry_audit
    from darsia.fluidflower.phase_training import gas_negative_sampling as sampling
    sources, _, _ = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs],
        preserve_native_revisions=preserve_native_revisions)
    sampling_limit = max(64, int(previous["settings"].get("pixels_per_region") or 1024))
    if withdrawal_snapshot["targets"] or preserve_native_revisions:
        # Do not let old parent weights/maps influence which negative pixels
        # train the fresh model. filter_document also drops older sampling records.
        derived = [read(pair[0]) for pair in pairs]
        if preserve_native_revisions:
            for document in derived:
                for decision in (document.get("decisions") or {}).values():
                    for field in ("annotations", "phase_quiz_answers"):
                        for item in decision.get(field) or []:
                            item.pop(sampling.FIELD, None)
        sampling_audit = dict(policy=("original-native-uniform-region-sampling-v1"
                                      if preserve_native_revisions else "withdrawal-uniform-region-sampling-v1"),
            parent_predictions_used=False, inherited_sampling_removed=True,
            reason="Fresh fit uses deterministic uniform sampling on each original native revision")
    else:
        derived, sampling_audit = sampling.enrich_training_documents(
            [read(pair[0]) for pair in pairs], regions, sources, parent, uniform_limit=sampling_limit)
    sampled_pairs = []
    for index, (document, pair) in enumerate(zip(derived, pairs)):
        destination = capture / "sampling_training_view" / str(index) / "decisions.json"
        write(destination, document)
        sampled_pairs.append((destination, pair[1]))
    pairs = sampled_pairs
    region_sources = {}
    sources, regions, _ = t._load_sources_and_regions([t.ReviewBundle(*pair) for pair in pairs],
        preserve_native_revisions=preserve_native_revisions,
        region_sources=region_sources if preserve_native_revisions else None)
    write(capture / "sampling_audit.json", sampling_audit)
    audit["sampling"] = sampling_audit
    settings = dict(previous["settings"])
    settings["manual_training_label_policy_version"] = logistic.MANUAL_TRAINING_LABEL_POLICY_VERSION
    settings[withdrawals.DOCUMENT_KEY] = withdrawal_snapshot
    for key in ("feature_cache_root", "completed_fit_source", "imported_completed_heads", "excluded_adapters", "resume_completed_heads"):
        settings.pop(key, None)
    runs = {row.run for row in regions}
    settings.update(iteration="Studio retrain", parent_model=str(parent), model_scope="single-series",
        series=[selected], fit_mode="mil", backend=backend, fit_heads=["co2_presence", "gas_within_co2"],
        inherited_heads=[], model_initialization="Fresh retraining of both heads from cumulative evidence",
        feature_version=selected_feature_version,
        feature_contracts={name: contract.to_dict() for name, contract in variants.items()},
        adapters=list(variants), mil_normalization_policy=logistic.MIL_NORMALIZATION_POLICY,
        optical_training_profile=optical_training_profile,
        mil_supervision_policy=logistic.MIL_SUPERVISION_POLICY,
        gas_negative_sampling_policy=("original-native-uniform-region-sampling-v1"
            if preserve_native_revisions else
            ("withdrawal-uniform-region-sampling-v1" if withdrawal_snapshot["targets"] else sampling.POLICY)),
        preserve_native_revisions=bool(preserve_native_revisions),
        regularization_grid=list(optical_profile["regularization_grid"]),
        mil_max_iters=max(15360, int(settings.get("mil_max_iters") or 0)),
        evaluation_mode="whole-run" if len(runs) > 1 else "fit-only-single-experiment",
        change_evidence_ablation=False, scientific_acceptance=False, human_review_required=True)
    settings.setdefault("regularization", 1e-4)
    settings.setdefault("pixels_per_region", 1024)
    settings = with_extraction_resources(settings,
        [source.asset_manifest_path for source in
         (region_sources.values() if preserve_native_revisions else sources)],
        cpus=cpus, memory_gb=memory_gb, extraction_workers=extraction_workers)
    if geometry_audit is not None:
        settings["training_geometry_revision"] = {
            "format": geometry_audit["format"], "runs": geometry_audit["runs"],
            "old_user_documents_preserved": True, "sampling_rebuilt_after_migration": True}
    request = prepare_phase_training_request(output_parent=workspace / "models", cache_parent=workspace / "cache",
        bundles=pairs, actor=actor, settings=settings, withdrawal_snapshot=withdrawal_snapshot)
    if request.get("locked_review_sessions"):
        raise ValueError("Detached continuation snapshots unexpectedly locked a live review")
    root = Path(request["output_root"])
    copied = []
    if profile is not None:
        for source in physics_files(parent, profile):
            target = root / source.relative_to(parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256_file(source)
            if sha256_file(target) != digest:
                raise OSError(f"Physical context copy failed: {source}")
            copied.append(dict(source=str(source), target=str(target), sha256=digest))
    completion_retraining = detach_inherited_completion(root, parent, cache)
    if completion_retraining:
        for row in copied:
            original_target = Path(row["target"])
            archived_target = Path(completion_retraining["archive"]) / original_target.relative_to(root)
            if not original_target.exists() and archived_target.is_file():
                row["original_copied_target"] = str(original_target)
                row["target"] = str(archived_target)
    # Materialize the selected model's same registered validation timeline.
    source = parent / "review_source/postflight_manifest.json"
    if not source.is_file():
        source = parent / "validation/postflight_manifest.json"
    value = read(source)
    value["jobs"] = [job for job in value["jobs"] if series_id(job["dataset"]) == selected]
    if not value["jobs"]:
        raise ValueError("Selected parent has no registered review timeline for this series")
    geometry_inheritance = None
    if geometry_audit is not None:
        from darsia.fluidflower.tools.phase_training_geometry_revision import bind_review_manifest, rebind_copied_physics
        value = bind_review_manifest(value, geometry_audit)
        geometry_inheritance = rebind_copied_physics(root, geometry_audit, parent_root=parent)
    promotion_inheritance = None
    if context_selections:
        value, promotion_inheritance = register_promoted_contexts(root,parent,selected,value,context_selections)
    write(root / "review_source/postflight_manifest.json", value)
    runtime = phase_runtime.physics_runtime_identity(profile)
    new_runtime = phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(root))
    expected_base_runtime = completion_retraining["base_runtime"] if completion_retraining else runtime
    if geometry_inheritance is None and promotion_inheritance is None and new_runtime != expected_base_runtime:
        raise ValueError("Continuation changed the selected parent's physical runtime")
    if geometry_inheritance is not None and (not geometry_inheritance["settings_unchanged"]
                                             or not geometry_inheritance["calibration_unchanged"]):
        raise ValueError("Geometry migration changed physical calibration or model settings")
    write(root / "physics_inheritance.json", dict(parent_model=str(parent), physics_runtime=new_runtime,
        original_physics_runtime=runtime, copies=copied, settings_unchanged=True, calibration_unchanged=True,
        **({"completion_retraining": completion_retraining} if completion_retraining else {}),
        **({("context_extensions" if audit.get("collection_context_selections") else "explicit_promotions"):promotion_inheritance}
           if promotion_inheritance is not None else {}),
        **({"geometry_revision": geometry_inheritance} if geometry_inheritance is not None else {})))
    frozen = [t.ReviewBundle(Path(pair[0]), Path(pair[1])) for pair in request["frozen_pairs"]]
    _, check_regions, _ = t._load_sources_and_regions(frozen,
        preserve_native_revisions=preserve_native_revisions)
    def comparable(row):
        value = asdict(row)
        if sampling.FIELD in value["payload"]:
            value["payload"][sampling.FIELD] = sampling.contract(value["payload"][sampling.FIELD])
        return value
    if [comparable(row) for row in regions] != [comparable(row) for row in check_regions]:
        raise ValueError("Request freezer changed the effective cumulative evidence")
    shutil.copytree(capture, root / "continuation_evidence")
    destination = library / "studio_retraining" / selected / root.name
    write(root / "continuation_preparation.json", dict(parent_model=str(parent), series=selected,
        destination=str(destination), model_library=str(library), **audit))
    return request, destination


def fit(request, *, resume_completed_heads=False, required_resume_checkpoints=None):
    """Call the current production entry point in this cancellable job process."""
    if request["settings"]["evaluation_mode"] == "fit-only-single-experiment":
        from darsia.fluidflower.phase_training.single_run import train
        train(Path(request["request_path"]), heads=("co2_presence", "gas_within_co2"))
    else:
        command = build_phase_training_command(request)
        # Use the installed trainer parser/implementation without spawning an
        # orphan training child when Studio's Stop button terminates this job.
        start = command.index("--campaign-id")
        if required_resume_checkpoints is not None and not isinstance(required_resume_checkpoints, Mapping):
            raise TypeError("Required resume checkpoints must map paths to inspected SHA256 values")
        required = {Path(path).resolve(): str(digest).lower()
                    for path, digest in (required_resume_checkpoints or {}).items()}
        if required_resume_checkpoints and len(required) != len(required_resume_checkpoints):
            raise ValueError("Required resume checkpoints contain duplicate paths")
        if required and not resume_completed_heads:
            raise ValueError("Required head checkpoints need resume_completed_heads")
        original_loader = t._load_reusable_head
        reused = set()
        def require_checkpoint(path, **kwargs):
            resolved = Path(path).resolve()
            expected_sha = required.get(resolved)
            if expected_sha is not None and sha256_file(resolved) != expected_sha:
                raise ValueError(f"Required optical checkpoint SHA changed before reuse: {resolved}")
            loaded = original_loader(path, **kwargs)
            if resolved in required:
                if loaded is None:
                    raise ValueError(f"Required optical checkpoint cannot be reused: {resolved}")
                if sha256_file(resolved) != expected_sha:
                    raise ValueError(f"Required optical checkpoint SHA changed during reuse: {resolved}")
                reused.add(resolved)
            return loaded
        try:
            if required:
                t._load_reusable_head = require_checkpoint
            t.main([*command[start:],
                    *(["--resume-completed-heads"] if resume_completed_heads else [])])
        finally:
            t._load_reusable_head = original_loader
        if set(required) != reused:
            raise ValueError("Trainer skipped one or more required optical checkpoints")
        for path, digest in required.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Required optical checkpoint SHA changed after training: {path}")
    # A region excluding aqueous CO2 does not select gas or water. Finish both
    # newly fitted heads with their joint aqueous probability before publication.
    from darsia.fluidflower.tools.train_coupled_aqueous_revision import refine_unpublished
    refine_unpublished(request)


def materialize(request, *, backend="auto"):
    """Run the checked CUDA materializer in this owned process when available."""
    root = Path(request["output_root"])
    use_cuda = backend != "cpu" and logistic.cuda_available()
    mode = "cuda with CPU reference parity" if use_cuda else "cpu"
    event("materializing", model_id=request["model_id"], output_root=str(root),
          inference_backend=mode,
          message=f"Calculating native phase maps using {mode}.")
    if use_cuda:
        arguments = ["darsia.fluidflower.tools.materialize_phase_iteration", "--model", str(root),
                     "--source", str(root / "review_source"),
                     "--work", str(root.parent / f"native-validation-{uuid.uuid4().hex}")]
        previous_arguments = sys.argv
        try:
            sys.argv = arguments
            runpy.run_module("darsia.fluidflower.tools.materialize_phase_iteration", run_name="__main__")
        finally:
            sys.argv = previous_arguments
    else:
        materialize_phase_validation_campaign(root, root / "review_source", root / "validation",
            progress=lambda message: print(message, flush=True))


def run(parent, library, cache, selected, actor, *, backend="auto", feature_version=None,
        optical_training_profile=DEFAULT_OPTICAL_TRAINING_PROFILE,
        cpus=None, memory_gb=None, extraction_workers=None):
    request, destination = prepare(parent, library, cache, selected, actor, backend=backend,
                                   feature_version=feature_version,
                                   optical_training_profile=optical_training_profile,
                                   cpus=cpus, memory_gb=memory_gb, extraction_workers=extraction_workers)
    root = Path(request["output_root"])
    completion_required = (root / "completion_retraining.json").is_file()
    event("prepared", model_id=request["model_id"], output_root=str(root), request_path=request["request_path"])
    fit(request)
    event("fitted", model_id=request["model_id"], output_root=str(root))
    if completion_required:
        from darsia.fluidflower.tools import train_series_b_completion as contextual
        materialize(request, backend=backend)
        output = root / "completion_training"
        event("completion_training", model_id=request["model_id"], output_root=str(root),
              message="Training a new variant-B aqueous completion from the freshly fitted base maps.")
        contextual.prepare(series_id(selected), root, output, cache_root=cache)
        contextual.fit(series_id(selected), root, output, epochs=16, backend=backend)
        contextual.attach(root, output)
        materialize(request, backend=backend)
    from darsia.fluidflower.tools.publish_fitted_phase_candidate import publish_fitted, attach_validation
    staging = Path(library).resolve().parent / "phase_studio_publication_staging"
    publication = publish_fitted(root, destination, staging, allow_nonconverged=True)
    unconverged = publication.get("nonconverged_heads", [])
    event("published", model_id=request["model_id"], output_root=str(destination), nonconverged_heads=unconverged,
          **({"message": "The fitted model is visible with a convergence warning. Native review maps are being prepared."}
             if unconverged else {}))
    if not completion_required:
        materialize(request, backend=backend)
    attach_validation(root, destination, staging, allow_nonconverged=True)
    event("ready", model_id=request["model_id"], output_root=str(destination), nonconverged_heads=unconverged,
          **({"message": "Native phase maps are ready for inspection. Some fitted heads did not converge; inspect the training diagnostics before evaluating this candidate."}
             if unconverged else {}))
    return destination


if __name__ == "__main__":
    if os.name != "nt":
        os.setsid()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "library", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int, help="CPU ceiling for the new optical extraction request")
    parser.add_argument("--memory-gb", type=float, help="RAM ceiling for the new optical extraction request")
    parser.add_argument("--extraction-workers", type=int, help="Explicit extraction process ceiling")
    parser.add_argument("--feature-version", help="Explicit optical feature revision; defaults to the existing production version")
    parser.add_argument("--optical-training-profile", choices=tuple(OPTICAL_TRAINING_PROFILES),
                        default=DEFAULT_OPTICAL_TRAINING_PROFILE,
                        help="Opt-in adapter and regularization selection for this immutable request")
    args = parser.parse_args()
    run(args.parent, args.library, args.cache, args.series, args.actor, backend=args.backend,
        feature_version=args.feature_version,
        optical_training_profile=args.optical_training_profile,
        cpus=args.cpus, memory_gb=args.memory_gb, extraction_workers=args.extraction_workers)
