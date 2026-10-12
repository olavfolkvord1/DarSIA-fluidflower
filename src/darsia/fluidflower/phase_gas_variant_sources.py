"""Frozen, source-preserving training inputs for the FF_AC gas variants.

This loader reads label and source metadata only.  It never stages images,
opens a held-out result, or creates negative labels from unmarked pixels.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .phase_training import trainer as training
from .phase_training.logistic import ReviewedRegion
from .phase_training.v5_review_fit_partition import verify_fit_partition
from .project import sha256_file
from .review_storage import resolve_review_path


HELD_OUT_RUNS = frozenset({
    "ac48", "ac53", "ac35", "ac38", "ac59", "ac22", "ac46", "ac57", "ac33",
})
NEW_LABEL_SHA256 = "56c0a0477e244f4ad9b15f5370b8d97ec9ddecf058a3053084426f4d589c7dbc"
NEW_LABEL_COUNTS = {"ac39": 446, "ac31": 234, "ac50": 191}
FORMAT = "ffac-gas-flatfield-source-inventory-v1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _path(value: str | Path) -> Path:
    return resolve_review_path(Path(value)).resolve()


def _sealed_bundles(request: dict[str, Any]) -> list[training.ReviewBundle]:
    bundles = []
    for binding in request["bundles"]:
        files = []
        for name in ("decisions", "combined_manifest"):
            record = binding[name]
            path = _path(record["snapshot_path"])
            _require(path.is_file() and sha256_file(path) == record["sha256"],
                     f"Frozen optical training snapshot changed: {path}")
            files.append(path)
        bundles.append(training.ReviewBundle(*files))
    _require(bool(bundles), "Frozen optical training has no source snapshots")
    return bundles


def _new_training_labels(
    candidate: Path, snapshot: Path,
) -> tuple[list[training.RunSource], list[ReviewedRegion], dict[str, Any]]:
    manifest = _read(snapshot / "SNAPSHOT_MANIFEST.json")
    annotations = snapshot / "pairwise_review_annotations.csv"
    actual_sha = sha256_file(annotations)
    _require(actual_sha == NEW_LABEL_SHA256
             and manifest["files"][annotations.name]["sha256"] == actual_sha
             and manifest["files"][annotations.name]["bytes"] == annotations.stat().st_size,
             "Sealed free-gas annotation CSV differs from the specified snapshot")
    session_path = snapshot / "review_session.json"
    _require(sha256_file(session_path) == manifest["files"][session_path.name]["sha256"],
             "Sealed training review session changed")
    session = _read(session_path)
    _require(session["target_model_id"] == candidate.name,
             "Free-gas labels refer to another candidate")
    review_root = _path(session["combined_root"])
    _require(review_root.is_relative_to(candidate / "validation" / "prepared_sets"),
             "Free-gas review root does not belong to the frozen candidate")
    validation_path = review_root / "validation_binding.json"
    validation = _read(validation_path)
    _require(validation["model_id"] == candidate.name
             and _path(validation["model_root"]) == candidate,
             "Free-gas review preparation belongs to another model")
    sources, regions, origins, source_receipts = [], [], {}, []
    manifests = {}
    native_sha_by_run = {}
    for run in sorted(NEW_LABEL_COUNTS):
        binding_path = candidate / "inference_run_bindings" / f"{run}.json"
        binding = _read(binding_path)
        native_path = _path(binding["materialized_manifest_path"])
        _require(binding["run"] == run
                 and binding["model_identity"] == [candidate.name, validation["model_sha256"]]
                 and sha256_file(native_path) == binding["materialized_manifest_sha256"],
                 f"Current training-label source binding changed: {run}")
        native = _read(native_path)
        _require(native["run"] == run
                 and native["dataset"] == "ffac"
                 and native["v5_photometry"]["source_manifest_sha256"] == binding["source_manifest_sha256"],
                 f"Training-label native source identity changed: {run}")
        source = training.RunSource(
            run=run, dataset="ffac", review_root=native_path.parent.parent.parent,
            asset_manifest_path=native_path,
            decisions_path=snapshot / "pairwise_review_decisions.json",
            combined_manifest_path=validation_path,
        )
        sources.append(source)
        manifests[run] = (native, source)
        native_sha_by_run[run] = binding["materialized_manifest_sha256"]
        source_receipts.append({
            "run": run, "binding_path": str(binding_path),
            "binding_sha256": sha256_file(binding_path),
            "native_manifest_path": str(native_path),
            "native_manifest_sha256": binding["materialized_manifest_sha256"],
            "original_source_manifest_sha256": binding["source_manifest_sha256"],
            "canvas_wh": native["canvas_wh"],
            "content_box_xywh": native["content_box_xywh"],
            "analysis_image_shape_hw": native["analysis_image_shape_hw"],
            "frames": [],
        })
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(dict(Counter(row["run"] for row in rows)) == NEW_LABEL_COUNTS,
             "Sealed training outline counts changed")
    for row in rows:
        run = row["run"]
        native, source = manifests[run]
        candidate_id = session["selected_candidates"][run]
        _require(row["category"] == "confirmed_only_free_gas"
                 and row["coordinate_space"] == "normalized_native_review_canvas_16x9"
                 and row["shape"] == "freehand"
                 and row["candidate_id"] == candidate_id
                 and validation["selected_candidates"][run] == candidate_id
                 and row["params_sha256"] == validation["pipeline_sha256"]
                 and row["contours_visible"].casefold() == "false",
                 f"Free-gas outline lost its blind candidate/source binding: {row['annotation_id']}")
        points = json.loads(row["points_json"])
        _require(len(points) >= 3 and all(
            isinstance(point, list) and len(point) == 2
            and all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in point)
            for point in points), "Invalid sealed freehand polygon")
        frame = int(row["frame_index"])
        elapsed = float(row["elapsed_h"])
        _require(0 <= frame < native["frame_count"]
                 and math.isclose(elapsed, float(native["frame_sources"][frame]["elapsed_h"]),
                                  abs_tol=1e-5, rel_tol=0),
                 "Sealed free-gas outline does not match its native frame time")
        payload = dict(row)
        payload.update(id=row["annotation_id"], points=points,
                       frame_index=frame, elapsed_h=elapsed,
                       phase_class="gas", training_phase_class="gas",
                       training_scope="whole-marked-region",
                       training_eligible=True,
                       gas_variant_annotation_sha256=actual_sha,
                       source_native_manifest_sha256=native_sha_by_run[run])
        for name in ("x", "y", "width", "height"):
            payload[name] = float(row[name])
        region = ReviewedRegion(
            region_id=f"{run}:manual:{row['annotation_id']}", run=run, dataset="ffac",
            frame_index=frame, elapsed_h=elapsed, phase_class="gas",
            confidence=row["confidence"].casefold(), source="manual",
            reviewer=row["reviewer"], payload=payload,
        )
        _require(region.supervision_weight > 0, "New free-gas outline has no supervision weight")
        regions.append(region)
        origins[region.region_id] = source
    _require(len(origins) == len(regions) == 871, "Duplicate sealed training outline ID")
    groups = training._native_revision_task_groups(regions, origins)
    aligned = [region for _source, group in groups for region in group]
    for receipt in source_receipts:
        native = manifests[receipt["run"]][0]
        receipt["frames"] = [
            {"frame_index": index, "elapsed_h": native["frame_sources"][index]["elapsed_h"],
             "sidecar_sha256": native["frame_sources"][index]["v5_log_linear_rgb_sha256"]}
            for index in sorted({r.frame_index for r in aligned if r.run == receipt["run"]})
        ]
    return sources, aligned, {
        "annotation_csv_path": str(annotations), "annotation_csv_sha256": actual_sha,
        "snapshot_manifest_sha256": sha256_file(snapshot / "SNAPSHOT_MANIFEST.json"),
        "review_session_sha256": sha256_file(session_path),
        "validation_binding_path": str(validation_path),
        "validation_binding_sha256": sha256_file(validation_path),
        "sources": source_receipts,
        "full_positive_outlines": len(aligned),
        "unmarked_pixels_are_negatives": False,
        "polygon_conversion": "x*canvas_width-content_x; y*canvas_height-content_y; existing annotation_mask rasterization",
    }


def load_training_sources(frozen_candidate: Path, snapshot: Path) -> dict[str, Any]:
    """Return extraction groups plus exact old/new IDs and safe provenance.

    ``groups`` contains ``(RunSource, tuple[ReviewedRegion, ...])``.  Old labels
    retain their exact original native revisions and frozen fit/evaluation role.
    Only the new sealed polygons bind to the current display source revisions.
    """
    candidate, snapshot = _path(frozen_candidate), _path(snapshot)
    regional = _read(candidate / "regional_model.json")
    optical_parent = _path(regional["parent_model"])
    checkpoint_path = optical_parent.parent.parent / "optical_checkpoint.json"
    checkpoint = _read(checkpoint_path)
    _require(checkpoint["format"] == "ffac-v5-raw-immutable-optical-checkpoint-v1"
             and _path(checkpoint["hidden_model_path"]) == optical_parent
             and checkpoint["verified_fitted"]["regions"] == 4048,
             "Candidate no longer binds the frozen 4048-region optical checkpoint")
    for name in ("co2_presence_signal.json", "gas_within_co2_signal.json"):
        _require(sha256_file(candidate / "models" / "generic" / name)
                 == sha256_file(optical_parent / "models" / "generic" / name),
                 f"Current candidate's optical head differs from its frozen parent: {name}")
    request_path = optical_parent / "training_request.json"
    request = _read(request_path)
    prepared_path = _path(checkpoint["prepared_request_path"])
    prepared = _read(prepared_path)
    _require(sha256_file(prepared_path) == checkpoint["prepared_request_sha256"]
             and sha256_file(optical_parent / "publication_receipt.json")
                 == checkpoint["publication_receipt_sha256"]
             and request["settings"] == prepared["settings"]
             and [tuple(b[k]["sha256"] for k in ("decisions", "combined_manifest"))
                  for b in request["bundles"]]
                 == [tuple(b[k]["sha256"] for k in ("decisions", "combined_manifest"))
                     for b in prepared["bundles"]]
             and sha256_file(request_path)
                 == sha256_file(candidate / "parent_provenance" / "training_request.json"),
             "Frozen optical training request or its publication binding changed")
    origins, audit = {}, {}
    sources, loaded, conflicts = training._load_sources_and_regions(
        _sealed_bundles(request),
        withdrawal_snapshot=request["settings"].get("review_retractions_snapshot"),
        preserve_native_revisions=True, region_sources=origins, label_policy_audit=audit,
    )
    inventory_path = optical_parent / "region_inventory.csv"
    with inventory_path.open(encoding="utf-8-sig", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    by_id = {r.region_id: r for r in loaded}
    _require(len(inventory) == len(by_id) == len(loaded) == 4048
             and {r["region_id"] for r in inventory} == set(by_id),
             "Old source snapshots differ from exact frozen training inventory")
    for row in inventory:
        region = by_id[row["region_id"]]
        _require(region.run == row["run"] and region.phase_class == row["phase_class"]
                 and math.isclose(region.supervision_weight, float(row["supervision_weight"])),
                 "Frozen old label class or supervision weight changed")
    partition = request["settings"]["v5_review_fit_partition"]
    allowed = verify_fit_partition(partition, loaded, origins)
    ood = frozenset(request["settings"]["v5_chemistry_fit_scope"]["excluded_runs"])
    old = [by_id[row["region_id"]] for row in inventory
           if row["region_id"] in allowed and row["run"] not in HELD_OUT_RUNS | ood]
    old_origins = {r.region_id: origins[r.region_id] for r in old}
    old_groups = training._native_revision_task_groups(old, old_origins)
    old = [r for _source, group in old_groups for r in group]
    new_sources, new, new_receipt = _new_training_labels(candidate, snapshot)
    old_ids, new_ids = {r.region_id for r in old}, {r.region_id for r in new}
    _require(not old_ids & new_ids and not ({r.run for r in old + new} & HELD_OUT_RUNS),
             "Held-out run or duplicate outline entered gas training")
    new_origins = {r.region_id: next(s for s in new_sources if s.run == r.run) for r in new}
    groups = training._native_revision_task_groups(old + new, {**old_origins, **new_origins})
    source_rows = [{
        "run": source.run, "native_manifest_path": str(source.asset_manifest_path),
        "native_manifest_sha256": sha256_file(source.asset_manifest_path),
        "regions": len(group), "region_ids_sha256": _digest(sorted(r.region_id for r in group)),
        "frames": sorted({r.frame_index for r in group}),
    } for source, group in old_groups]
    receipt = {
        "format": FORMAT, "frozen_candidate": str(candidate),
        "optical_parent": str(optical_parent),
        "optical_checkpoint_path": str(checkpoint_path),
        "optical_checkpoint_sha256": sha256_file(checkpoint_path),
        "old_training_request_sha256": sha256_file(request_path),
        "old_training_source_snapshots": [
            {name: {"path": binding[name]["snapshot_path"],
                    "sha256": binding[name]["sha256"]}
             for name in ("decisions", "combined_manifest")}
            for binding in request["bundles"]
        ],
        "old_inventory_path": str(inventory_path),
        "old_inventory_sha256": sha256_file(inventory_path),
        "old_inventory_regions": len(inventory),
        "old_fit_partition": partition,
        "old_fit_regions_after_exclusions": len(old),
        "old_fit_region_ids_sha256": _digest(sorted(old_ids)),
        "old_fit_annotations_sha256": _digest([
            {"region_id": r.region_id, "payload": r.payload}
            for r in sorted(old, key=lambda r: r.region_id)
        ]),
        "new_annotation_sha256": NEW_LABEL_SHA256,
        "new_region_ids": sorted(new_ids),
        "old_region_ids": sorted(old_ids),
        "training_runs": sorted({r.run for r in old + new}),
        "held_out_runs": sorted(HELD_OUT_RUNS),
        "held_out_runs_absent_from_training": True,
        "existing_chemistry_evaluation_only_runs": sorted(ood),
        "old_sources": source_rows, "new_labels": new_receipt,
        "training_counts": {
            name: {run: dict(sorted(Counter(r.phase_class for r in rows if r.run == run).items()))
                   for run in sorted({r.run for r in rows})}
            for name, rows in (("old", old), ("new", new), ("combined", old + new))
        },
        "new_region_ids_sha256": _digest(sorted(new_ids)),
        "held_out_evaluation_labels_read": False,
        "old_unsupported_issue_labels_remain_excluded": True,
        "source_loader_code_sha256": sha256_file(Path(__file__)),
    }
    return {"groups": groups, "old_region_ids": old_ids, "new_region_ids": new_ids,
            "old_regions": old, "new_regions": new, "receipt": receipt}


__all__ = ["HELD_OUT_RUNS", "NEW_LABEL_SHA256", "load_training_sources"]
