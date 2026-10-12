"""Move detached training evidence to a verified native photograph revision.

Studio documents and published models are never edited. The caller first merges
annotation IDs and aligns their timeline to the parent's native manifest. This
module then changes that one frozen view's geometry, binds the new photographs,
and explicitly rebinds copied physical contexts on the unchanged baseline grid.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from darsia.fluidflower.tools import phase_rig_annotation_migration as migration


FORMAT = "phase-training-geometry-revision-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _native(job):
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root
    run = str(job["run"]).lower()
    root = analysis_review_root(Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
    return root, root / "native_frames" / run / "manifest.json"


def translate_training_annotation(annotation, bound):
    """Preserve the trainer's old visible vertices before translating them.

    The installed annotation rasterizer clips vertices to pixel centers before
    rounding and filling polygons. We repeat that operation on the ORIGINAL
    canvas, then subtract the sampling displacement. Target clipping removes
    only unobserved registration borders and is always recorded. It is never
    used to trim an interior gas label or to change its phase semantics.
    """
    points, polygon = migration._points(annotation)
    cw, ch, ox, oy, width, height = bound["old_layout"]
    ncw, nch, nox, noy, _, _ = bound["new_layout"]
    source = np.asarray([(x*cw-ox, y*ch-oy) for x, y in points], dtype=np.float64)
    # Match the trainer's float32 conversion before its vertex clipping.
    source = source.astype(np.float32).astype(np.float64)
    visible = np.clip(source, [0, 0], [width-1, height-1])
    if np.ptp(visible[:, 0]) <= 0 or np.ptp(visible[:, 1]) <= 0:
        raise ValueError("Annotation has no two-dimensional visible source region")
    shifted = visible - np.asarray(bound["shift"])
    receipt = bound["binding"]["receipt"]
    valid = receipt.get("valid_bounds_xyxy", [0, 0, width, height])
    left, top, right, bottom = [int(value) for value in valid]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Invalid target registration bounds")
    target = np.clip(shifted, [left, top], [right-1, bottom-1])
    if np.ptp(target[:, 0]) <= 0 or np.ptp(target[:, 1]) <= 0:
        raise ValueError("Registration would discard an entire annotation; manual review required")
    normalized = (target + [nox, noy]) / [ncw, nch]
    result = deepcopy(annotation)
    original_sampling = result.pop("training_sampling", None)
    minimum, maximum = normalized.min(axis=0), normalized.max(axis=0)
    result.update(x=float(minimum[0]), y=float(minimum[1]),
                  width=float(maximum[0]-minimum[0]), height=float(maximum[1]-minimum[1]),
                  points=normalized.tolist() if polygon else [],
                  frame_index=bound["binding"]["target_frame_index"],
                  elapsed_h=bound["new_frame"]["elapsed_h"])
    def area(vertices):
        following = np.roll(vertices, -1, axis=0)
        return abs(float((vertices[:, 0]*following[:, 1]-vertices[:, 1]*following[:, 0]).sum())) / 2
    source_area, target_area = area(visible), area(target)
    audit = dict(source_geometry={key: deepcopy(annotation[key]) for key in migration.GEOMETRY_KEYS if key in annotation},
        source_visible_vertices_native_px=visible.tolist(),
        source_vertices_clipped=bool(np.any(source != visible)),
        target_border_vertices_clipped=bool(np.any(shifted != target)),
        target_border_clip_max_px=float(np.max(np.abs(shifted-target))),
        visible_source_area_px2=source_area, target_area_px2=target_area,
        target_area_loss_fraction=max(0., 1-target_area/source_area) if source_area else 0.,
        valid_bounds_xyxy=valid, shift_xy_native_px=list(bound["shift"]),
        stale_sampling_removed=original_sampling is not None,
        time_metadata_rebound=annotation.get("elapsed_h") != result["elapsed_h"],
        target_geometry={key: deepcopy(result[key]) for key in migration.GEOMETRY_KEYS if key in result})
    result["geometry_migration"] = dict(format=FORMAT,
        frame_receipt_sha256=digest(bound["binding"]), source_annotation_sha256=digest(annotation),
        candidate_attribution_applies_to_original_prediction_only=True, **audit)
    return result, audit


def bind_review_manifest(manifest, audit):
    """Return a copied job manifest using the new AC32 source only."""
    value = deepcopy(manifest)
    for job in value.get("jobs", []):
        run = str(job.get("run", "")).lower()
        if run not in audit.get("runs", {}):
            continue
        entry = audit["runs"][run]
        job["review_output_dir"] = entry["review_root"]
        job["analysis_source_review_root"] = entry["review_root"]
        job["geometry_revision_id"] = entry["geometry_revision_id"]
        job["geometry_source_manifest_sha256"] = entry["target_manifest_sha256"]
    return value


def rebind_evidence_audits(original_documents, transformed_documents):
    """Retain human supersession after transporting the copied original box.

    The reviewer fingerprint continues to identify the original review in the
    migration provenance. The trainer-facing fingerprint names its exact
    transported copy. No verdict, reviewer, timestamp or target ID changes.
    """
    from darsia.fluidflower.phase_evidence_audit import payload_hash, superseded_evidence_ids
    def inventory(documents, *, unique):
        records = {}
        for document in documents:
            for run, decision in document.get("decisions", {}).items():
                for collection, kind in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
                    for item in decision.get(collection, []):
                        identifier = item.get("id") or item.get("proposal_id")
                        if not identifier:
                            from darsia.fluidflower.phase_training.trainer import _geometry_key
                            identifier = _geometry_key(str(run).lower(), item)[:16]
                        key = f"{str(run).lower()}:{kind}:{identifier}"
                        if unique and key in records and payload_hash(records[key][0]) != payload_hash(item):
                            raise ValueError("Supersession rebinding requires unique merged evidence revisions")
                        records.setdefault(key, []).append(item)
        return records
    old = inventory(original_documents, unique=False)
    old_records = [(key, item) for key, items in old.items() for item in items]
    superseded_evidence_ids(old_records)
    values = deepcopy(transformed_documents)
    new = {key: items[0] for key, items in inventory(values, unique=True).items()}
    visiting, complete, changes = set(), set(), []
    def inherited_bridge(identifier, payload, target):
        """Prove an old live re-review through its already-migrated parent copy.

        A continuation may retain an unchanged live reviewer record while its
        original target exists only in the copied parent's registered geometry.
        Accept only a recorded bridge whose entire reviewer payload, exact
        source fingerprint, transform receipt, and captured target all match.
        """
        current_geometry = payload.get("geometry_migration") or {}
        if current_geometry.get("format") != FORMAT:
            return None
        for candidate in old.get(identifier, []):
            geometry = candidate.get("geometry_migration") or {}
            bridge = geometry.get("supersession_rebinding") or {}
            candidate_audit = candidate.get("evidence_audit") or {}
            if (geometry.get("format") != FORMAT
                    or not geometry.get("geometry_revision_id")
                    or bridge.get("human_verdict_and_target_id_unchanged") is not True
                    or bridge.get("original_evidence_audit") != payload["evidence_audit"]):
                continue
            keys = ("geometry_revision_id", "frame_receipt_sha256", "source_annotation_sha256")
            if any(not geometry.get(key) or geometry.get(key) != current_geometry.get(key) for key in keys):
                continue
            source_hash = geometry["source_annotation_sha256"]
            if not any(digest(item) == source_hash for item in old.get(identifier, [])):
                continue
            registered_hash = bridge.get("transformed_target_annotation_sha256")
            if (registered_hash != candidate_audit.get("original_annotation_sha256")
                    or registered_hash not in {payload_hash(item) for item in old.get(target, [])}):
                continue
            # Equality here includes category/class, confidence, reviewer,
            # timestamps, candidate identity and the transported coordinates.
            left, right = deepcopy(candidate), deepcopy(payload)
            left.pop("geometry_migration", None); right.pop("geometry_migration", None)
            left["evidence_audit"] = deepcopy(bridge["original_evidence_audit"])
            if left != right:
                continue
            return dict(kind="verified-captured-parent-supersession-bridge",
                parent_reviewer_payload_sha256=payload_hash(candidate),
                parent_target_payload_sha256=registered_hash,
                source_reviewer_canonical_sha256=source_hash,
                frame_receipt_sha256=geometry["frame_receipt_sha256"],
                geometry_revision_id=geometry["geometry_revision_id"])
        return None
    def visit(identifier):
        if identifier in complete:
            return
        if identifier in visiting:
            raise ValueError("Human evidence supersession contains a cycle")
        visiting.add(identifier)
        payload = new[identifier]
        audit = payload.get("evidence_audit")
        if audit:
            target = str(audit.get("original_region_id") or "")
            if target in new:
                visit(target)
                proof = None
                if target not in old or audit.get("original_annotation_sha256") not in {payload_hash(item) for item in old[target]}:
                    proof = inherited_bridge(identifier, payload, target)
                    if proof is None:
                        raise ValueError("Cannot prove the original human supersession before geometry rebinding")
                target_hash = payload_hash(new[target])
                if target_hash != audit["original_annotation_sha256"]:
                    old_audit = deepcopy(audit)
                    payload.setdefault("geometry_migration", {"format": FORMAT})["supersession_rebinding"] = dict(
                        original_evidence_audit=old_audit,
                        transformed_target_annotation_sha256=target_hash,
                        human_verdict_and_target_id_unchanged=True,
                        **({"inherited_proof": proof} if proof is not None else {}))
                    audit["original_annotation_sha256"] = target_hash
                    changes.append(dict(region_id=identifier, target=target,
                        original_target_sha256=old_audit["original_annotation_sha256"], transformed_target_sha256=target_hash,
                        **({"inherited_proof": proof} if proof is not None else {})))
        visiting.remove(identifier); complete.add(identifier)
    for identifier in new:
        visit(identifier)
    if superseded_evidence_ids(old_records) != superseded_evidence_ids(list(new.items())):
        raise ValueError("Geometry migration changed which human reviews supersede earlier labels")
    return values, changes


def migrate_training_pairs(pairs, *, revision_root, output_root):
    """Create derived evidence pairs and a native wrapper at new paths only."""
    revision_root, output_root = Path(revision_root).resolve(), Path(output_root).resolve()
    if output_root.exists() or output_root == revision_root or output_root.is_relative_to(revision_root):
        raise ValueError("Training geometry output must be a new detached directory")
    target_path = revision_root / "manifest.json"
    source_snapshot = revision_root / "source_manifest_snapshot.json"
    receipt_path = revision_root / "migration_receipts.json"
    target, original, receipts = read(target_path), read(source_snapshot), read(receipt_path)
    status = read(revision_root / "status.json")
    run = str(target.get("run", "")).lower()
    if run != "ac32" or target.get("dataset") != "ffac" or not status.get("complete") or not status.get("baseline_exact"):
        raise ValueError("Only the complete verified AC32 revision is supported")
    old_hash, new_hash = sha(source_snapshot), sha(target_path)
    if target.get("source_manifest_sha256") != old_hash or status.get("manifest_sha256") != new_hash:
        raise ValueError("Native revision provenance changed")
    bounds = target["correction_contract"]["rig_registration"]["valid_bounds_xyxy"]
    bound = migration._bind_receipts(receipts, {run: original}, {run: target}, {run: old_hash}, {run: new_hash})
    if set(bound) != {(run, index) for index in range(int(target["frame_count"]))}:
        raise ValueError("Migration must cover the complete registered frame timeline")
    protected = {str(path): sha(path) for path in (target_path, source_snapshot, receipt_path)}
    documents, source_documents, manifests, originals, records = [], [], [], [], []
    baseline_source = None
    seen = set()
    for decision_path, manifest_path in pairs:
        decision_path, manifest_path = Path(decision_path), Path(manifest_path)
        document, manifest = read(decision_path), read(manifest_path)
        if document.get("schema") != 2:
            raise ValueError("Expected schema-2 detached evidence")
        protected.update({str(path): sha(path) for path in (decision_path, manifest_path)})
        candidates = [job for job in manifest["jobs"] if str(job["run"]).lower() == run]
        if run in {str(k).lower() for k in document["decisions"]}:
            if not candidates:
                raise ValueError("AC32 decisions have no source job")
            for job in candidates:
                source_root, path = _native(job)
                if sha(path) != old_hash:
                    raise ValueError("AC32 annotations are not bound to the receipt's original native image revision")
                baseline_source = source_root / "native_references" / run / "pre_injection.jpg"
                protected[str(path)] = old_hash
        value = deepcopy(document)
        for name, decision in value["decisions"].items():
            if str(name).lower() != run:
                continue
            if decision.get("reference_guides"):
                raise ValueError("Reference guides need a separately specified geometry mapping")
            for collection in ("annotations", "phase_quiz_answers"):
                for position, item in enumerate(decision.get(collection, [])):
                    identifier = item.get("id") or item.get("proposal_id")
                    if not identifier:
                        # Anonymous historical regions receive a provenance-bound
                        # identity in the detached copy; never a newly invented label.
                        identifier = digest(item)[:32]
                        item["id"] = identifier
                    key = (name, collection, identifier)
                    if key in seen:
                        raise ValueError("Geometry migration requires merged unique annotation IDs")
                    seen.add(key)
                    index = item.get("frame_index")
                    if type(index) is not int or (run, index) not in bound:
                        raise ValueError(f"Annotation {identifier} has no exact frame receipt")
                    if item.get("geometry_migration"):
                        raise ValueError("Never translate previously migrated evidence a second time")
                    mapped, row = translate_training_annotation(item, bound[(run, index)])
                    mapped["geometry_migration"]["geometry_revision_id"] = receipts["geometry_revision_id"]
                    decision[collection][position] = mapped
                    from darsia.fluidflower.phase_training.trainer import canonical_phase_class
                    records.append(dict(run=run, collection=collection, id=identifier,
                        phase_class=canonical_phase_class(item, "manual" if collection == "annotations" else "quiz"),
                        frame_index=index, **row))
        documents.append(value); source_documents.append(document); manifests.append(manifest)
        originals.append(dict(decisions=str(decision_path), decisions_sha256=sha(decision_path),
                              manifest=str(manifest_path), manifest_sha256=sha(manifest_path)))
    if baseline_source is None:
        raise ValueError("No AC32 source was selected for geometry migration")
    documents, supersession_changes = rebind_evidence_audits(source_documents, documents)
    reference_index = target["reference"]["timeline_frame_index"]
    if bound[(run, reference_index)]["shift"] != [0., 0.]:
        raise ValueError("The frozen physical grid requires exact baseline identity")
    # Check actual JPEG content, not just pre-encoding array metadata.
    from PIL import Image
    _, _, ox, oy, width, height = migration._layout(target)
    with Image.open(baseline_source) as image:
        before = np.asarray(image.convert("RGB"))[oy:oy+height, ox:ox+width]
    baseline_frame = revision_root / f"frame_{reference_index:03d}.jpg"
    with Image.open(baseline_frame) as image:
        after = np.asarray(image.convert("RGB"))[oy:oy+height, ox:ox+width]
    if not np.array_equal(before, after):
        raise ValueError("Registered baseline JPEG differs from the parent physical reference")
    protected[str(baseline_source)] = sha(baseline_source)
    frame_paths = []
    for row in target["frame_sources"]:
        path = revision_root / f"frame_{row['frame_index']:03d}.jpg"
        expected = row.get("frame_sha256")
        if sha(path) != expected:
            raise ValueError(f"Registered photograph changed: {path}")
        protected[str(path)] = expected; frame_paths.append(path)
    native_root = output_root / "native_source"
    audit = dict(format=FORMAT, source_revision_root=str(revision_root), originals=originals,
        runs={run: dict(review_root=str(native_root), source_manifest_sha256=old_hash,
            target_manifest_sha256=new_hash, target_manifest_path=str(native_root / "native_frames" / run / "manifest.json"),
            geometry_revision_id=receipts["geometry_revision_id"], valid_bounds_xyxy=bounds,
            receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path))},
        migrated_regions=len(records), source_clipped_regions=sum(row["source_vertices_clipped"] for row in records),
        supersession_bindings=supersession_changes,
        target_border_clipped_regions=sum(row["target_border_vertices_clipped"] for row in records),
        maximum_target_area_loss_fraction=max((row["target_area_loss_fraction"] for row in records), default=0.),
        stale_sampling_removed=sum(row["stale_sampling_removed"] for row in records),
        baseline_jpeg_content_exact=True, other_runs_changed=False, duplicate_evidence_added=False,
        clipping_policy="Original trainer float32 vertex clip to [0,w-1]/[0,h-1], then translation, then accepted target visible pixel centers; every edge loss retained below.",
        sampling_policy="Rebuild sampler after migration; original saved gas maps are a different grid and must explicitly fall back to uniform sampling unless a separately validated transported selector is used.",
        physical_context_policy="Retain baseline-grid facies, axes, permeability and injection stages; restrict domain to the registration's known valid area; rebind every copied nested native-manifest/context hash.",
        records=records, live_reviews_modified=False, old_assets_modified=False)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f"Source changed during detached migration: {path}")
    # All semantic checks complete before materializing any derived artifact.
    frame_out = native_root / "native_frames" / run
    frame_out.mkdir(parents=True, exist_ok=False)
    for path in [target_path, *frame_paths]:
        destination = frame_out / path.name
        shutil.copy2(path, destination)
        if sha(destination) != protected[str(path)]:
            raise OSError("Photograph revision copy failed")
    ref_out = native_root / "native_references" / run / "pre_injection.jpg"
    ref_out.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(baseline_frame, ref_out)
    pairs_out = []
    for index, (document, manifest) in enumerate(zip(documents, manifests)):
        folder = output_root / "training_view" / str(index)
        document["training_geometry_revision"] = dict(format=FORMAT,
            audit_path=str(output_root / "geometry_migration.json"), runs=deepcopy(audit["runs"]))
        write_new(folder / "decisions.json", document)
        write_new(folder / "manifest.json", bind_review_manifest(manifest, audit))
        pairs_out.append((folder / "decisions.json", folder / "manifest.json"))
    write_new(output_root / "geometry_migration.json", audit)
    return pairs_out, audit


def rebind_copied_physics(model_root, audit, *, parent_root):
    """Explicitly revise only newly copied contexts/profile, retaining calibration."""
    from darsia.fluidflower import phase_runtime
    from darsia.fluidflower.tools.prepare_grain_gas_retraining import PROFILE_FILES
    root, parent = Path(model_root).resolve(), Path(parent_root).resolve()
    if root == parent or root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Physical context changes require a distinct new model directory")
    profile_paths = [root / name for name in PROFILE_FILES if (root / name).is_file()]
    if len(profile_paths) != 1:
        raise ValueError("Expected one copied physics profile")
    path = profile_paths[0]
    before = phase_runtime.load_physics_profile(root)
    original_profile_hash = sha(path)
    profile = deepcopy(before)
    updates, context_writes = {}, []
    for run, record in audit["runs"].items():
        key = f"ffac/{run}"
        entry = profile["runs"][key]
        if entry["native_manifest_sha256"] != record["source_manifest_sha256"]:
            raise ValueError("Copied physical context is not bound to original geometry")
        context_path = (root / entry["path"]).resolve()
        if not context_path.is_relative_to(root) or sha(context_path) != entry["sha256"]:
            raise ValueError("Copied context escaped its new model or changed")
        with np.load(context_path, allow_pickle=False) as arrays:
            data = {key: arrays[key].copy() for key in arrays.files}
        domain = data["domain"]
        left, top, right, bottom = record["valid_bounds_xyxy"]
        allowed = np.zeros(domain.shape, dtype=bool)
        allowed[top:bottom, left:right] = True
        removed = int(np.count_nonzero(domain & ~allowed))
        data["domain"] = domain & allowed
        if not data["domain"].any():
            raise ValueError("Registration eliminated the physical domain")
        context_writes.append((context_path, data))
        updates[key] = dict(source_manifest_sha256=entry["native_manifest_sha256"],
            target_manifest_sha256=record["target_manifest_sha256"], original_context_sha256=entry["sha256"],
            removed_unobserved_domain_pixels=removed, context_path=str(context_path),
            geometry_revision_id=record["geometry_revision_id"], valid_bounds_xyxy=record["valid_bounds_xyxy"])
    for context_path, data in context_writes:
        # A copied model is a private workspace, not the user's published parent.
        temporary = context_path.with_suffix(".geometry-rebound.npz")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(context_path)
    def update(value):
        if isinstance(value, dict):
            runs = value.get("runs")
            if isinstance(runs, dict):
                for key, details in updates.items():
                    if key not in runs:
                        continue
                    entry = runs[key]
                    if entry.get("native_manifest_sha256") != details["source_manifest_sha256"] or entry.get("sha256") != details["original_context_sha256"]:
                        raise ValueError("Nested inherited physics binding disagrees")
                    entry["native_manifest_sha256"] = details["target_manifest_sha256"]
                    entry["sha256"] = sha(details["context_path"])
                    entry["geometry_revision"] = deepcopy(details)
            for nested in value.values():
                update(nested)
        elif isinstance(value, list):
            for nested in value:
                update(nested)
    update(profile)
    temporary = path.with_suffix(".geometry-rebound.json")
    write_new(temporary, profile)
    temporary.replace(path)
    after = phase_runtime.load_physics_profile(root)
    return dict(format=FORMAT, profile_path=str(path), original_profile_sha256=original_profile_hash,
        profile_sha256=sha(path), original_runtime=phase_runtime.physics_runtime_identity(before),
        runtime=phase_runtime.physics_runtime_identity(after), runs=updates,
        calibration_unchanged=True, settings_unchanged=True, injection_stages_unchanged=True,
        baseline_grid_unchanged=True, context_domain_restricted_to_observed_pixels=True)
