"""Compose a complete two-arm source set from verified revisions and new work.

Old per-revision files are read-only. Registered AC32 revisions are reused only
when their original frozen geometry contract survives in the new manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import (
    AC47_V4_CHANGED_FRAMES, AC47_V4_SOURCE_SHA256,
    AC61_PREPARED_BASELINE_ARRAY_SHA256, FORMAT,
    _frozen_registration_profile, _hash, _read, _require_matching_geometry,
    _prepared_baseline_source, _source_photometry_revision, _write_new,
    materialize_revision, source_manifests,
)


def _contract_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def verify_reusable(revision: dict[str, Any], receipt: dict[str, Any], *,
                    arm: str, target_sha256: str | None) -> None:
    """Rehash all inherited assets, IDs and geometry before crossing roots."""
    source_path = Path(revision["manifest_path"])
    source_sha = revision["manifest_sha256"]
    if _hash(source_path) != source_sha:
        raise ValueError("Frozen source manifest changed")
    original = _read(source_path)
    if (receipt.get("format") != FORMAT or receipt.get("arm") != arm
            or receipt.get("source_manifest_sha256") != source_sha
            or Path(receipt.get("source_manifest_path", "")).resolve() != source_path.resolve()
            or receipt.get("common_target_sha256") != target_sha256
            or receipt.get("run") != revision["run"]):
        raise ValueError("Per-revision receipt belongs to another source or arm")
    review_root = Path(receipt["materialized_review_root"])
    per_revision_path = review_root.parents[0] / "materialization_receipt.json"
    if not per_revision_path.is_file() or _read(per_revision_path) != receipt:
        raise ValueError("Per-revision receipt changed")
    manifest_path = Path(receipt["materialized_manifest_path"])
    if (_hash(manifest_path) != receipt["materialized_manifest_sha256"] or
            manifest_path.resolve() != (review_root / "native_frames" /
                                       revision["run"] / "manifest.json").resolve()):
        raise ValueError("Materialized native manifest changed")
    materialized = _read(manifest_path)
    marker = materialized.get("two_arm_photometry") or {}
    if (marker.get("arm") != arm or marker.get("source_manifest_sha256") != source_sha
            or marker.get("common_target_sha256") != target_sha256):
        raise ValueError("Materialized photometry identity changed")
    for field in ("run", "dataset", "canvas_wh", "content_box_xywh",
                  "analysis_image_shape_hw", "frame_count"):
        if materialized.get(field) != original.get(field):
            raise ValueError(f"Materialized {field} differs from original")
    source_photometry_revision = _source_photometry_revision(original, revision)
    if source_photometry_revision is not None:
        arm_revision_id = ("ffac-two-arm-within-run-legacy-v2-v1"
                           if arm == "within_run"
                           else "ffac-two-arm-global-robust16-legacy-v2-v1")
        if (marker.get("source_photometry_revision") != source_photometry_revision
                or receipt.get("source_photometry_revision") != source_photometry_revision
                or marker.get("arm_photometry_revision_id") != arm_revision_id
                or receipt.get("arm_photometry_revision_id") != arm_revision_id
                or materialized.get("photometry_revision_id") != arm_revision_id
                or marker.get("arm_colorchecker_algorithm_revision") != "legacy-v2"):
            raise ValueError("AC47 source-v4 and new arm photometry provenance differ")
        old_environment = dict((original.get("correction_contract") or {}).get("environment") or {})
        if old_environment.pop("FFAC_COLORCHECKER_ALGORITHM_REVISION", None) != "neutral-tone-only-v4":
            raise ValueError("AC47 source is not the frozen v4 correction")
        new_environment = (materialized.get("correction_contract") or {}).get("environment") or {}
        if (new_environment.get("FFAC_COLORCHECKER_ALGORITHM_REVISION") not in (None, "legacy-v2")
                or {key: value for key, value in new_environment.items()
                    if key != "FFAC_COLORCHECKER_ALGORITHM_REVISION"} != old_environment):
            raise ValueError("AC47 changed more than the declared ColorChecker revision")
    elif materialized.get("photometry_revision_id") != original.get("photometry_revision_id"):
        raise ValueError("An unreviewed active photometry revision appeared")
    _, frozen_registration = _frozen_registration_profile(original, revision["run"])
    actual_registration = (materialized.get("correction_contract") or {}).get(
        "rig_registration")
    _require_matching_geometry(frozen_registration, actual_registration,
                               revision["run"])
    if len(receipt["frames"]) != len(original["frame_sources"]):
        raise ValueError("Materialized frame count changed")
    baseline = review_root / "native_references" / revision["run"] / "pre_injection.jpg"
    if _hash(baseline) != receipt["materialized_baseline_sha256"]:
        raise ValueError("Materialized pre-injection baseline changed")
    if (materialized.get("reference") or {}).get("materialized_native_sha256") != _hash(baseline):
        raise ValueError("Materialized baseline manifest binding changed")
    prepared_baseline = _prepared_baseline_source(original, revision)
    marker = (materialized.get("two_arm_photometry") or {}).get("prepared_baseline_composite")
    recorded_marker = receipt.get("prepared_baseline_composite")
    if prepared_baseline is None:
        if marker is not None or recorded_marker is not None:
            raise ValueError("Unexpected prepared-baseline marker in another source")
    else:
        first_array_sha = receipt["frames"][0]["corrected_array_sha256"]
        prepared_baseline["materialized_corrected_array_sha256"] = first_array_sha
        if (marker != prepared_baseline or recorded_marker != prepared_baseline or
                materialized["frame_sources"][0].get("corrected_array_sha256") != first_array_sha or
                (materialized.get("reference") or {}).get("corrected_array_sha256") != first_array_sha or
                (arm == "within_run" and first_array_sha != AC61_PREPARED_BASELINE_ARRAY_SHA256)):
            raise ValueError("AC61 prepared baseline lost its calibrated-rig provenance")
    for original_frame, new_frame, recorded in zip(
            original["frame_sources"], materialized["frame_sources"],
            receipt["frames"], strict=True):
        index = original_frame["frame_index"]
        if (new_frame["frame_index"] != index or recorded["frame_index"] != index
                or new_frame["elapsed_h"] != original_frame["elapsed_h"]
                or recorded["elapsed_h"] != original_frame["elapsed_h"]
                or new_frame["source_path"] != original_frame["source_path"]
                or recorded["raw_path"] != original_frame["source_path"]):
            raise ValueError(f"Frame {index} raw identity or time changed")
        if (new_frame.get("source_sha256") != recorded["raw_sha256"] or
                (original_frame.get("source_sha256") or
                 original_frame.get("source_raw_sha256") or recorded["raw_sha256"])
                != recorded["raw_sha256"]):
            raise ValueError(f"Frame {index} raw SHA differs")
        if "phase_map" in new_frame:
            raise ValueError(f"Frame {index} carries stale phase-map provenance")
        for field in ("registration_receipt_path", "registration_receipt_sha256",
                      "geometry_revision_id"):
            if original_frame.get(field) != new_frame.get(field):
                raise ValueError(f"Frame {index} lost {field}")
        registration_path = original_frame.get("registration_receipt_path")
        if registration_path and _hash(Path(registration_path)) != original_frame.get(
                "registration_receipt_sha256"):
            raise ValueError(f"Frame {index} frozen registration receipt changed")
        native = review_root / "native_frames" / revision["run"] / f"frame_{index:03d}.jpg"
        if (_hash(native) != recorded["native_jpeg_sha256"] or
                new_frame.get("native_jpeg_sha256") != recorded["native_jpeg_sha256"]):
            raise ValueError(f"Frame {index} native JPEG changed")
        if arm == "within_run" and original_frame.get("corrected_array_sha256"):
            differs = (recorded["corrected_array_sha256"] !=
                       original_frame["corrected_array_sha256"])
            expected_v4_delta = (source_sha == AC47_V4_SOURCE_SHA256 and
                                 index in AC47_V4_CHANGED_FRAMES)
            if differs != expected_v4_delta:
                raise ValueError(f"Frame {index} within-run corrected array differs from certified source revision")


def compose(request_path: Path, *, arm: str, output_root: Path,
            reuse_roots: list[Path], target_path: Path | None,
            target_sha256: str | None) -> dict[str, Any]:
    if arm not in {"within_run", "global"}:
        raise ValueError("Unknown photometry arm")
    if (arm == "global") != (target_path is not None and target_sha256 is not None):
        raise ValueError("Only global arm has a frozen target")
    if target_path is not None and _hash(target_path) != target_sha256:
        raise ValueError("Frozen ColorChecker target changed")
    if output_root.exists():
        raise FileExistsError(f"Recovery root already exists: {output_root}")
    frozen_request_sha = _hash(request_path)
    revisions = source_manifests(request_path)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for root in reuse_roots:
        for path in (root / "revisions").glob("*/materialization_receipt.json"):
            row = _read(path)
            candidates.setdefault(str(row.get("source_manifest_sha256")), []).append(row)
    reused: list[str] = []
    new: list[str] = []
    rejected: list[dict[str, str]] = []
    reports = []
    composition = []
    output_root.mkdir(parents=True)
    common_target = ((target_path, target_sha256) if target_path is not None else None)
    for revision in revisions:
        chosen = None
        for candidate in candidates.get(revision["manifest_sha256"], []):
            try:
                verify_reusable(revision, candidate, arm=arm, target_sha256=target_sha256)
                chosen = candidate
                break
            except (ValueError, KeyError, FileNotFoundError) as error:
                rejected.append({"source_manifest_sha256": revision["manifest_sha256"],
                                 "reason": str(error)})
        if chosen is None:
            chosen = materialize_revision(revision, arm=arm, output_root=output_root,
                                          common_target=common_target)
            verify_reusable(revision, chosen, arm=arm, target_sha256=target_sha256)
            new.append(revision["manifest_sha256"])
        else:
            reused.append(revision["manifest_sha256"])
        reports.append(chosen)
        receipt_path = (Path(chosen["materialized_review_root"]).parent /
                        "materialization_receipt.json")
        composition.append({
            "source_manifest_sha256": revision["manifest_sha256"],
            "origin": "reused" if revision["manifest_sha256"] in reused else "new",
            "per_revision_receipt_path": str(receipt_path),
            "per_revision_receipt_sha256": _hash(receipt_path),
            "materialized_manifest_sha256": chosen["materialized_manifest_sha256"],
        })
    contracts = []
    for revision in revisions:
        original = _read(Path(revision["manifest_path"]))
        contracts.append({"source_manifest_path": revision["manifest_path"],
                          "source_manifest_sha256": revision["manifest_sha256"],
                          "source_correction_contract_sha256": _contract_sha(
                              original.get("correction_contract") or {})})
    result = dict(format=FORMAT, arm=arm,
        frozen_request_path=str(request_path), frozen_request_sha256=frozen_request_sha,
        target_path=str(target_path) if target_path else None,
        target_sha256=target_sha256,
        selection={"policy": "all-frozen-bundle-native-revisions-v1",
                   "selected_source_revisions": len(revisions),
                   "selected_source_manifest_sha256": sorted(
                       row["manifest_sha256"] for row in revisions)},
        source_revisions=len(reports), frames=sum(row["frame_count"] for row in reports),
        revisions=reports, complete=True,
        recovery={"format": "ffac-two-arm-sha-verified-reuse-v1",
                  "reused_source_manifests": reused,
                  "new_source_manifests": new,
                  "rejected_reuse_candidates": rejected,
                  "reuse_roots": [str(path) for path in reuse_roots],
                  "source_correction_contracts": contracts,
                  "composition": composition},
        created_utc=datetime.now(timezone.utc).isoformat())
    _write_new(output_root / "materialization_receipt.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-request", type=Path, required=True)
    parser.add_argument("--arm", choices=("within_run", "global"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--target-profile", type=Path)
    parser.add_argument("--target-sha256")
    args = parser.parse_args(argv)
    report = compose(args.frozen_request, arm=args.arm, output_root=args.output_root,
        reuse_roots=args.reuse_root, target_path=args.target_profile,
        target_sha256=args.target_sha256)
    print(json.dumps({"arm": report["arm"],
                      "source_revisions": report["source_revisions"],
                      "frames": report["frames"],
                      "reused": len(report["recovery"]["reused_source_manifests"]),
                      "new": len(report["recovery"]["new_source_manifests"])}), flush=True)


if __name__ == "__main__":
    main()
