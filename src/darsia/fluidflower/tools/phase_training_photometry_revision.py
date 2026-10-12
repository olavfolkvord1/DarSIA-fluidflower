"""Bind an immutable AC47 corrected-photo revision to detached regional training.

This bridge changes no reviewer payload, frame geometry, timing, parent model, or
published review. It verifies every source and replacement JPEG by hash before
rebinding copied evidence manifests and the new model's review collection.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools.train_phase_model_further import native_source


FORMAT = "phase-training-photometry-revision-v1"
RUN = "ac47"
SERIES = "ffac"
ALGORITHM_REVISION = "neutral-tone-only-v4"
GEOMETRY_FIELDS = (
    "dataset", "run", "frame_count", "canvas_wh", "analysis_image_shape_hw",
    "content_box_xywh", "full_analysis_resolution",
    "additional_spatial_resampling", "domain", "phase_topology",
    "injection_points", "reference",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def verify_revision(revision_root, *, parent_job=None):
    """Verify a complete photometry-only revision and its source binding."""
    root = resolve_review_path(Path(revision_root)).resolve()
    receipt_path = root / "photometry_revision.json"
    receipt = _read(receipt_path)
    if receipt.get("format") != FORMAT or receipt.get("dataset") != SERIES or receipt.get("run") != RUN:
        raise ValueError("Expected an AC47/FF_AC photometry revision receipt")
    if receipt.get("algorithm_revision") != ALGORITHM_REVISION or receipt.get("complete") is not True:
        raise ValueError("AC47 photometry revision is incomplete or uses an unknown correction")
    source_root = resolve_review_path(Path(receipt["source_review_root"])).resolve()
    source_path = source_root / "native_frames" / RUN / "manifest.json"
    target_path = root / "native_frames" / RUN / "manifest.json"
    if source_root == root or _sha(source_path) != receipt.get("source_manifest_sha256"):
        raise ValueError("Original AC47 corrected source changed")
    if _sha(target_path) != receipt.get("target_manifest_sha256"):
        raise ValueError("Revised AC47 manifest changed")
    source, target = _read(source_path), _read(target_path)
    if any(source.get(key) != target.get(key) for key in GEOMETRY_FIELDS):
        raise ValueError("Photometry revision changes geometry, domain, or baseline semantics")
    if source.get("dataset") != SERIES or str(source.get("run", "")).lower() != RUN:
        raise ValueError("Revision source is not AC47/FF_AC")
    if target.get("source_manifest_sha256") != receipt["source_manifest_sha256"]:
        raise ValueError("Revision manifest does not bind its original source")
    environment = target.get("correction_contract", {}).get("environment", {})
    if environment.get("FFAC_COLORCHECKER_ALGORITHM_REVISION") != ALGORITHM_REVISION:
        raise ValueError("Revised ColorChecker algorithm is not recorded in the native manifest")
    source_contract = deepcopy(source.get("correction_contract"))
    target_contract = deepcopy(target.get("correction_contract"))
    target_contract.get("environment", {}).pop("FFAC_COLORCHECKER_ALGORITHM_REVISION", None)
    if source_contract != target_contract:
        raise ValueError("Photometry revision changes unrelated correction settings")
    if parent_job is not None:
        _, parent_path, _ = native_source(parent_job)
        if _sha(parent_path) != receipt["source_manifest_sha256"]:
            raise ValueError("Selected parent AC47 source is not this revision's source")
    old_rows, new_rows = source.get("frame_sources", []), target.get("frame_sources", [])
    frame_receipts = receipt.get("frames", [])
    if len(old_rows) != 20 or len(new_rows) != 20 or len(frame_receipts) != 20:
        raise ValueError("Photometry revision must cover the complete 20-frame AC47 timeline")
    changed = []
    for index, (old, new, record) in enumerate(zip(old_rows, new_rows, frame_receipts, strict=True)):
        for key in ("frame_index", "elapsed_h", "source_name", "source_path", "source_sha256"):
            if old.get(key) != new.get(key) or old.get(key) != record.get(key):
                raise ValueError(f"Frame {index} source identity or time changed")
        if old.get("frame_index") != index:
            raise ValueError("AC47 frame order changed")
        old_jpeg = source_root / "native_frames" / RUN / f"frame_{index:03d}.jpg"
        new_jpeg = root / "native_frames" / RUN / f"frame_{index:03d}.jpg"
        old_sha, new_sha = _sha(old_jpeg), _sha(new_jpeg)
        if (old_sha != old.get("native_jpeg_sha256") or new_sha != new.get("native_jpeg_sha256")
                or old_sha != record.get("source_jpeg_sha256")
                or new_sha != record.get("target_jpeg_sha256")):
            raise ValueError(f"Frame {index} corrected JPEG bytes changed")
        if old_sha != new_sha:
            changed.append(index)
    if changed != receipt.get("changed_frame_indices") or 0 in changed:
        raise ValueError("Changed-frame inventory or pre-injection reference differs")
    old_baseline = source_root / "native_references" / RUN / "pre_injection.jpg"
    new_baseline = root / "native_references" / RUN / "pre_injection.jpg"
    baseline_sha = _sha(old_baseline)
    if baseline_sha != _sha(new_baseline) or baseline_sha != receipt.get("baseline_sha256"):
        raise ValueError("Photometry revision changes the registered baseline")
    reference_receipt = receipt.get("injection_references") or {}
    reference_dir = root / "native_references" / RUN
    reference_paths = {
        "acquisition_inventory_sha256": reference_dir / "acquisition_inventory.json",
        "injection_end_catalog_sha256": reference_dir / "injection_ends.json",
        "reference_registration_sha256": reference_dir / "reference_registration.json",
        "reference_builder_sha256": root / "complete_ac47_photometry_references.py",
    }
    if (reference_receipt.get("format") != "ac47-injection-reference-revision-v1"
            or any(_sha(path) != reference_receipt.get(key)
                   for key, path in reference_paths.items())):
        raise ValueError("Revised AC47 injection-reference assets differ from their receipt")
    inventory = _read(reference_paths["acquisition_inventory_sha256"])
    catalog = _read(reference_paths["injection_end_catalog_sha256"])
    registration = _read(reference_paths["reference_registration_sha256"])
    if (inventory.get("dataset") != SERIES or inventory.get("run") != RUN
            or inventory.get("complete") is not True or not inventory.get("frames")
            or catalog.get("dataset") != SERIES or catalog.get("run") != RUN
            or catalog.get("native_manifest_sha256") != receipt["target_manifest_sha256"]
            or catalog.get("photometry_revision_id") != target.get("photometry_revision_id")
            or registration.get("native_manifest_sha256") != receipt["target_manifest_sha256"]
            or registration.get("frame_count") != len(new_rows)
            or len(registration.get("reference_registration", [])) != len(new_rows)):
        raise ValueError("Revised AC47 reference inventory, catalog, or registration is unbound")
    catalog_rows = catalog.get("references") or []
    reference_rows = reference_receipt.get("reference_rows") or []
    if (len(catalog_rows) != len(reference_rows) or len(catalog_rows) != 2
            or len(catalog_rows) != reference_receipt.get("reference_count")
            or len(new_rows) != reference_receipt.get("registration_count")):
        raise ValueError("Revised AC47 injection-reference count differs")
    by_id = {row.get("id"): row for row in reference_rows}
    if len(by_id) != len(reference_rows):
        raise ValueError("Duplicate AC47 injection reference in receipt")
    for row in catalog_rows:
        record = by_id.get(row.get("id"))
        if (record is None or row.get("source_name") != record.get("source_name")
                or row.get("source_path") != record.get("source_path")
                or row.get("sha256") != record.get("target_reference_sha256")
                or row.get("correction_contract") != target.get("correction_contract")
                or row.get("available") is not True
                or row.get("needs_materialization") is not False):
            raise ValueError("Revised AC47 injection reference is unavailable or unbound")
        asset = resolve_review_path(Path(row["path"])).resolve()
        if (not asset.is_relative_to(reference_dir.resolve())
                or asset != resolve_review_path(Path(record["target_path"])).resolve()
                or _sha(asset) != row["sha256"]):
            raise ValueError("Revised AC47 injection reference JPEG differs")
    return dict(root=root, receipt_path=receipt_path, receipt_sha256=_sha(receipt_path),
                source_root=source_root, source_manifest_sha256=receipt["source_manifest_sha256"],
                target_manifest_sha256=receipt["target_manifest_sha256"],
                changed_frame_indices=changed, baseline_sha256=baseline_sha)


def _bound_job(job, verified):
    result = deepcopy(job)
    result["review_output_dir"] = str(verified["root"])
    result["analysis_source_review_root"] = str(verified["root"])
    result["photometry_revision"] = dict(format=FORMAT,
        receipt_path=str(verified["receipt_path"]),
        receipt_sha256=verified["receipt_sha256"],
        source_manifest_sha256=verified["source_manifest_sha256"],
        target_manifest_sha256=verified["target_manifest_sha256"])
    return result


def bind_captured_evidence(evidence, collection, output_root, verified):
    """Copy source manifests; retain exact original decision documents/regions."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    original_regions = deepcopy(evidence["regions"])
    value = deepcopy(evidence)
    changed_manifests = []
    for index, (decisions, manifest_path) in enumerate(value["pairs"]):
        manifest = _read(manifest_path)
        jobs = manifest.get("jobs", [])
        matching = [job for job in jobs if str(job.get("run", "")).lower() == RUN]
        if not matching:
            continue
        for job in matching:
            _, source_path, _ = native_source(job)
            if _sha(source_path) != verified["source_manifest_sha256"]:
                raise ValueError("Cumulative AC47 annotation comes from another photograph revision")
        updated = deepcopy(manifest)
        updated["jobs"] = [_bound_job(job, verified) if str(job.get("run", "")).lower() == RUN else job
                           for job in updated["jobs"]]
        target = output_root / "evidence_manifests" / f"{index:03d}.json"
        _write_new(target, updated)
        value["pairs"][index] = [decisions, str(target)]
        changed_manifests.append(index)
    if not changed_manifests or value["regions"] != original_regions:
        raise ValueError("AC47 evidence binding failed or modified reviewer labels")
    collection_manifest = _read(Path(collection) / "postflight_manifest.json")
    matches = [job for job in collection_manifest.get("jobs", []) if str(job.get("run", "")).lower() == RUN]
    if len(matches) != 1:
        raise ValueError("Expected exactly one AC47 job in parent review collection")
    _, source_path, _ = native_source(matches[0])
    if _sha(source_path) != verified["source_manifest_sha256"]:
        raise ValueError("Parent AC47 collection is not bound to revision source")
    collection_manifest["jobs"] = [
        _bound_job(job, verified) if str(job.get("run", "")).lower() == RUN else job
        for job in collection_manifest["jobs"]
    ]
    revised_collection = output_root / "collection"
    _write_new(revised_collection / "postflight_manifest.json", collection_manifest)
    value.setdefault("audit", {})["photometry_revision"] = dict(
        format=FORMAT, receipt_path=str(verified["receipt_path"]),
        receipt_sha256=verified["receipt_sha256"],
        source_manifest_sha256=verified["source_manifest_sha256"],
        target_manifest_sha256=verified["target_manifest_sha256"],
        changed_frame_indices=verified["changed_frame_indices"],
        rebound_manifest_indices=changed_manifests,
        reviewer_payloads_timestamps_and_geometry_unchanged=True,
        original_documents_modified=False)
    _write_new(output_root / "binding.json", value["audit"]["photometry_revision"])
    return value, revised_collection


def verify_context_binding(job, parent_native_sha, target_native_sha, *, parent_context=None):
    """Prove a context follows the exact AC47 photometry receipt lineage."""
    descriptor = job.get("photometry_revision") or {}
    if descriptor.get("format") != FORMAT:
        raise ValueError("Revised physical context requires an explicit photometry receipt")
    receipt_path = resolve_review_path(Path(descriptor["receipt_path"])).resolve()
    if _sha(receipt_path) != descriptor.get("receipt_sha256"):
        raise ValueError("Photometry revision receipt changed")
    verified = verify_revision(receipt_path.parent)
    source_sha = verified["source_manifest_sha256"]
    target_sha = verified["target_manifest_sha256"]
    if (descriptor.get("source_manifest_sha256") != source_sha
            or descriptor.get("target_manifest_sha256") != target_sha):
        raise ValueError("Physical context does not match the exact AC47 photometry revision")
    if target_native_sha != target_sha:
        # The two-arm correction starts at the v4 target and binds a new native
        # manifest while retaining the original v4 receipt as its first hop.
        return _verify_two_arm_context_binding(
            job, verified, parent_native_sha, target_native_sha,
            parent_context=parent_context)
    if parent_native_sha == source_sha:
        return verified
    if parent_native_sha != target_sha:
        raise ValueError("Physical context does not match the exact AC47 photometry revision")
    if parent_context is None:
        raise ValueError("Inherited AC47 physical context lacks the exact photometry revision provenance")
    # A later training generation inherits the already-revised physical grid.
    # Accept it only when its own provenance proves this exact receipt was used.
    inherited = parent_context.get("provenance", {}).get("photometry_revision", {})
    inherited_receipt = inherited.get("receipt_path")
    if (not inherited_receipt
            or resolve_review_path(Path(inherited_receipt)).resolve() != receipt_path
            or inherited.get("receipt_sha256") != verified["receipt_sha256"]
            or inherited.get("source_manifest_sha256") != source_sha
            or inherited.get("target_manifest_sha256") != target_sha
            or inherited.get("physical_grid_bytes_preserved") is not True):
        raise ValueError("Inherited AC47 physical context lacks the exact photometry revision provenance")
    return verified


def _verify_two_arm_context_binding(job, verified, parent_native_sha, target_native_sha,
                                    *, parent_context):
    """Accept a materialized AC47 revision only with both verified provenance hops."""
    from darsia.fluidflower.tools import train_ffac_two_arm_optical_region as two_arm

    descriptor = job.get("two_arm_photometry") or {}
    arm = descriptor.get("arm")
    if (descriptor.get("format") != two_arm.FORMAT
            or arm not in two_arm.ARMS
            or not descriptor.get("receipt_path")
            or parent_context is None
            or parent_native_sha != target_native_sha
            or parent_context.get("native_manifest_sha256") != parent_native_sha):
        raise ValueError("Physical context does not match the exact AC47 two-arm revision")
    receipt_path = resolve_review_path(Path(descriptor["receipt_path"])).resolve()
    if _sha(receipt_path) != descriptor.get("receipt_sha256"):
        raise ValueError("AC47 two-arm materialization receipt changed")
    receipt = _read(receipt_path)
    if (receipt.get("format") != two_arm.MATERIALIZATION_FORMAT
            or receipt.get("arm") != arm
            or receipt.get("complete") is not True):
        raise ValueError("AC47 two-arm materialization receipt is incomplete")
    if arm == "global":
        target_path = resolve_review_path(Path(receipt["target_path"])).resolve()
        if _sha(target_path) != receipt.get("target_sha256"):
            raise ValueError("AC47 global chart target changed")
    target_content_sha = None
    if arm == "sequential_global":
        from darsia.fluidflower.tools.ffac_common_start_target import validate_candidate
        target_path = resolve_review_path(Path(receipt["target_path"])).resolve()
        if _sha(target_path) != receipt.get("target_sha256"):
            raise ValueError("AC47 sequential common-start target changed")
        target = _read(target_path)
        validate_candidate(target, verify_source_bytes=True)
        target_content_sha = two_arm._digest(target)
    v4_sha = verified["target_manifest_sha256"]
    source_sha = (two_arm.AC47_PRIOR_WITHIN_SOURCE_SHA256
                  if arm == "sequential_global" else v4_sha)
    matches = [row for row in receipt.get("revisions") or []
               if str(row.get("run", "")).lower() == RUN
               and row.get("source_manifest_sha256") == source_sha]
    if len(matches) != 1:
        raise ValueError("AC47 two-arm materialization has no unique frozen source")
    row = matches[0]
    _, native_path, _ = native_source(job)
    source_path = (resolve_review_path(Path(row["source_manifest_path"])).resolve()
                   if arm == "sequential_global" else
                   verified["root"] / "native_frames" / RUN / "manifest.json")
    if (descriptor.get("source_manifest_sha256") != source_sha
            or descriptor.get("materialized_manifest_sha256") != target_native_sha
            or descriptor.get("raw_geometry_and_time_equivalence_verified") is not True
            or row.get("materialized_manifest_sha256") != target_native_sha
            or resolve_review_path(Path(row["source_manifest_path"])).resolve() != source_path
            or (arm == "sequential_global" and _sha(source_path) != source_sha)
            or resolve_review_path(Path(row["materialized_manifest_path"])).resolve() != native_path.resolve()
            or _sha(native_path) != target_native_sha):
        raise ValueError("Physical context does not match the exact AC47 two-arm revision")
    provenance = parent_context.get("provenance", {})
    inherited_v4 = provenance.get("photometry_revision") or {}
    inherited_two_arm = provenance.get("two_arm_photometry") or {}
    if (not inherited_v4.get("receipt_path")
            or resolve_review_path(Path(inherited_v4["receipt_path"])).resolve()
                != verified["receipt_path"]
            or inherited_v4.get("receipt_sha256") != verified["receipt_sha256"]
            or inherited_v4.get("source_manifest_sha256") != verified["source_manifest_sha256"]
            or inherited_v4.get("target_manifest_sha256") != v4_sha
            or inherited_v4.get("physical_grid_bytes_preserved") is not True
            or inherited_two_arm.get("format") != two_arm.FORMAT
            or inherited_two_arm.get("arm") != arm
            or not inherited_two_arm.get("receipt_path")
            or resolve_review_path(Path(inherited_two_arm["receipt_path"])).resolve()
                != receipt_path
            or inherited_two_arm.get("receipt_sha256") != descriptor["receipt_sha256"]
            or inherited_two_arm.get("source_manifest_sha256") != source_sha
            or inherited_two_arm.get("materialized_manifest_sha256") != target_native_sha
            or inherited_two_arm.get("physical_grid_bytes_preserved") is not True):
        raise ValueError("Inherited AC47 physical context lacks exact two-arm provenance")
    if arm == "sequential_global":
        inherited_prior = provenance.get("source_two_arm_photometry") or {}
        if not isinstance(inherited_prior.get("receipt_path"), str) or not inherited_prior["receipt_path"]:
            raise ValueError("Inherited AC47 context lost its prior within-run provenance")
        prior_path = resolve_review_path(Path(inherited_prior.get("receipt_path", ""))).resolve()
        if (inherited_prior.get("format") != two_arm.FORMAT
                or inherited_prior.get("arm") != "within_run"
                or inherited_prior.get("source_manifest_sha256") != v4_sha
                or inherited_prior.get("materialized_manifest_sha256") != source_sha
                or inherited_prior.get("physical_grid_bytes_preserved") is not True
                or _sha(prior_path) != inherited_prior.get("receipt_sha256")):
            raise ValueError("Inherited AC47 context lost its prior within-run provenance")
        prior = _read(prior_path)
        prior_rows = [item for item in prior.get("revisions") or []
                      if item.get("source_manifest_sha256") == v4_sha
                      and item.get("materialized_manifest_sha256") == source_sha]
        if (prior.get("format") != two_arm.MATERIALIZATION_FORMAT
                or prior.get("arm") != "within_run" or prior.get("complete") is not True
                or len(prior_rows) != 1
                or resolve_review_path(Path(prior_rows[0]["materialized_manifest_path"])).resolve()
                    != source_path):
            raise ValueError("Inherited AC47 prior within-run receipt changed")
    if arm == "sequential_global":
        checked = two_arm.verify_revision(
            row, arm=arm, target_sha256=receipt.get("target_sha256"),
            target_content_sha256=target_content_sha)
    else:
        checked = two_arm.verify_revision(
            row, arm=arm, target_sha256=receipt.get("target_sha256"))
    if (checked["source_manifest_sha256"] != source_sha
            or checked["materialized_manifest_sha256"] != target_native_sha):
        raise ValueError("Verified AC47 two-arm source differs from the inherited context")
    return verified
