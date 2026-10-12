"""Certify coordinate-preserving FF_AC v5 RAW materialization before fitting.

This checks the *frozen source revision of each annotation*. It does not infer
whether a phase label is scientifically correct. The resulting certificate is
an input to, not a replacement for, later phase-map QA.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_training import trainer as training
from darsia.fluidflower.phase_training.review_regions import annotation_mask
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_ffac_two_arm_optical_region as previous
from darsia.fluidflower.tools.ffac_v5_geometry_policy import verify_accepted_geometry
from darsia.fluidflower.tools.materialize_ffac_two_arm_sources import source_manifests


FORMAT = "ffac-v5-native-annotation-boundary-qa-v1"
BATCH_FORMAT = "ffac-v5-native-materialization-batch-v1"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _geometry(annotation: dict, canvas_wh: list[int],
              content_box_xywh: list[int]) -> float:
    """Return bounding-box fraction on the unchanged native content canvas."""
    _require(annotation.get("coordinate_space")
             == "normalized_native_review_canvas_16x9",
             "Frozen annotation has another coordinate space")
    points = annotation.get("points") or []
    if points:
        _require(annotation.get("shape") == "freehand" and len(points) >= 3,
                 "Frozen freehand annotation has no polygon")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    else:
        _require(annotation.get("shape") == "rectangle",
                 "Frozen annotation shape is unsupported")
        x, y = float(annotation["x"]), float(annotation["y"])
        width, height = float(annotation["width"]), float(annotation["height"])
        _require(width > 0 and height > 0, "Frozen rectangle has no area")
        xs, ys = [x, x + width], [y, y + height]
    _require(all(math.isfinite(value) and -1e-6 <= value <= 1 + 1e-6
                 for value in (*xs, *ys)),
             "Frozen label has invalid normalized coordinates")
    left, right = min(xs) * canvas_wh[0], max(xs) * canvas_wh[0]
    top, bottom = min(ys) * canvas_wh[1], max(ys) * canvas_wh[1]
    _require(right > left and bottom > top, "Frozen annotation has no area")
    cx, cy, width, height = content_box_xywh
    overlap = max(0.0, min(right, cx + width) - max(left, cx)) * max(
        0.0, min(bottom, cy + height) - max(top, cy))
    _require(overlap > 0, "Frozen annotation has no overlap with source image")
    return overlap / ((right - left) * (bottom - top))


def audit(frozen_request_path: Path, batch_receipt_path: Path) -> dict:
    frozen_request_path = Path(frozen_request_path).resolve()
    batch_receipt_path = Path(batch_receipt_path).resolve()
    batch = _read(batch_receipt_path)
    _require(batch.get("format") == BATCH_FORMAT
             and Path(batch.get("frozen_training_request_path", "")).resolve()
                 == frozen_request_path
             and batch.get("frozen_training_request_sha256")
                 == sha256_file(frozen_request_path)
             and batch.get("source_revisions") == 52
             and batch.get("timeline_frames") == 1045
             and batch.get("canonical_runs") == 43,
             "V5 batch does not seal the complete frozen source cohort")
    source_rows = source_manifests(frozen_request_path)
    rows = batch.get("revisions") or []
    selected = {row["source_manifest_sha256"]: row for row in rows}
    _require(len(selected) == len(rows) == len(source_rows) == 52
             and set(selected) == {row["manifest_sha256"] for row in source_rows},
             "V5 batch does not cover every frozen source revision")
    geometry = {}
    source_sha_by_path = {}
    frame_count = 0
    labelled_sources = Counter()
    for source in source_rows:
        digest = source["manifest_sha256"]
        row = selected[digest]
        original_path = Path(source["manifest_path"]).resolve()
        source_sha_by_path[str(original_path).casefold()] = digest
        target_path = Path(row["materialized_manifest_path"]).resolve()
        _require(sha256_file(original_path) == digest
                 and sha256_file(target_path)
                     == row["materialized_manifest_sha256"]
                 and sha256_file(row["receipt_path"]) == row["receipt_sha256"],
                 f"Frozen or derived V5 source changed: {source['run']}")
        original, target, receipt = (_read(original_path), _read(target_path),
                                     _read(Path(row["receipt_path"])))
        _require(receipt.get("source_manifest_sha256") == digest
                 and receipt.get("materialized_manifest_sha256")
                     == row["materialized_manifest_sha256"],
                 f"V5 revision receipt changed: {source['run']}")
        physical_geometry = verify_accepted_geometry(original_path, target_path)
        _require(physical_geometry["source_manifest_sha256"] == digest
                 and physical_geometry["materialized_manifest_sha256"]
                     == row["materialized_manifest_sha256"],
                 f"V5 per-run physical geometry differs from sealed source: {source['run']}")
        _require(target.get("v5_photometry", {}).get("source_manifest_sha256")
                 == digest
                 and target.get("analysis_source_review_root") is None
                 and target.get("correction_contract", {}).get("kind")
                     == "ffac-v5-full-native-camera-linear",
                 f"V5 target may still resolve old source: {source['run']}")
        old_frames = original["frame_sources"]
        new_frames = target["frame_sources"]
        _require(len(old_frames) == len(new_frames) == row["frame_count"]
                 == len(receipt["frames"]),
                 f"V5 frame timeline changed: {source['run']}")
        for index, (before, after, record) in enumerate(
                zip(old_frames, new_frames, receipt["frames"], strict=True)):
            _require(before.get("frame_index") == after.get("frame_index")
                     == record.get("frame_index") == index
                     and before.get("elapsed_h") == after.get("elapsed_h")
                     == record.get("elapsed_h"),
                     f"V5 source frame time/order changed: {source['run']}/{index}")
            for field in ("source_path", "source_sha256", "frame_seconds",
                          "registration_receipt_path", "registration_receipt_sha256"):
                _require(before.get(field) == after.get(field),
                         f"V5 source identity changed: {source['run']}/{index}/{field}")
            target_jpeg = (target_path.parent / f"frame_{index:03d}.jpg")
            _require(Path(after["frame_path"]).resolve() == target_jpeg
                     and sha256_file(target_jpeg) == after["native_jpeg_sha256"]
                     == record["native_jpeg_sha256"],
                     f"V5 trainer JPEG differs from sealed source: {source['run']}/{index}")
            certificate = record.get("certificate") or {}
            _require(certificate.get("native_shape_hw")
                     == target["analysis_image_shape_hw"]
                     and bool(certificate.get("transform_sha256")),
                     f"V5 RAW registration certificate absent: {source['run']}/{index}")
            with Image.open(target_jpeg) as image:
                _require(list(image.size) == target["canvas_wh"],
                         f"V5 native canvas size changed: {source['run']}/{index}")
            frame_count += 1
        geometry[digest] = (original, target, physical_geometry)
    _require(frame_count == 1045, "V5 frame set is incomplete")

    request = _read(frozen_request_path)
    pairs = previous._frozen_pairs(request)
    records, regions = previous.region_records(pairs)
    native_sources: dict = {}
    training._load_sources_and_regions(
        pairs, preserve_native_revisions=True,
        region_sources=native_sources)
    _require(len(records) == len(regions) == len(native_sources) == 4048
             and {r.region_id for r in regions} == set(records) == set(native_sources),
             "Frozen effective annotation IDs changed")
    counts = Counter()
    min_overlap = 1.0
    annotated_frames = defaultdict(list)
    for region in regions:
        source = native_sources[region.region_id]
        path = source.asset_manifest_path.resolve()
        key = str(path).casefold()
        digest = source_sha_by_path.get(key)
        if digest is None:
            digest = sha256_file(path)
            source_sha_by_path[key] = digest
        _require(digest == records[region.region_id]["native_manifest_sha256"]
                 and digest in geometry,
                 f"Annotation lost its exact source revision: {region.region_id}")
        original, target, _physical_geometry = geometry[digest]
        index = int(region.frame_index)
        _require(0 <= index < original["frame_count"]
                 and original["frame_sources"][index]["elapsed_h"]
                     == target["frame_sources"][index]["elapsed_h"],
                 f"Annotation moved to another timeline frame: {region.region_id}")
        overlap = _geometry(region.payload, original["canvas_wh"],
                            original["content_box_xywh"])
        min_overlap = min(min_overlap, overlap)
        labelled_sources[digest] += 1
        counts[(region.phase_class, region.payload["shape"])] += 1
        annotated_frames[(digest, index)].append(region)
    annotation_signal_pixels = 0
    annotations_outside_corrected_domain = 0
    for (digest, index), frame_regions in sorted(annotated_frames.items()):
        _original, target, physical_geometry = geometry[digest]
        row = target["frame_sources"][index]
        path = Path(row["v5_log_linear_rgb_path"])
        _require(sha256_file(path) == row["v5_log_linear_rgb_sha256"],
                 f"Frozen annotation float source changed: {digest}/{index}")
        signal = np.load(path, allow_pickle=False, mmap_mode="r")
        height = int(target["analysis_image_shape_hw"][0])
        width = int(target["analysis_image_shape_hw"][1])
        _require(signal.shape == (3, height, width) and signal.dtype == np.float32,
                 f"Annotation float source has wrong native shape: {digest}/{index}")
        union = np.zeros((height, width), dtype=bool)
        for region in frame_regions:
            union |= annotation_mask(region.payload,
                canvas_wh=target["canvas_wh"], content_box=target["content_box_xywh"])
        _require(bool(np.any(union)), f"Eligible labels have no native pixels: {digest}/{index}")
        with np.load(physical_geometry["context_npz_path"], allow_pickle=False) as archive:
            domain = archive["domain"].astype(bool)
        outside = int(np.count_nonzero(union & ~domain))
        annotations_outside_corrected_domain += outside
        _require(outside == 0,
                 f"Frozen annotation intersects excluded run-native domain: {digest}/{index}: {outside} pixels")
        annotation_signal_pixels += int(np.count_nonzero(union))
        _require(bool(np.isfinite(signal[:, union]).all()),
                 f"Eligible annotation intersects invalid V5 float pixels: {digest}/{index}")
    # Canvas/content geometry and every saved native label remain unchanged.
    # The accepted physical context is a new per-run 2D field, source-bound
    # separately. RAW texture alignment and phase truth need later QA.
    result = {
        "format": FORMAT,
        "passed": True,
        "frozen_training_request_path": str(frozen_request_path),
        "frozen_training_request_sha256": sha256_file(frozen_request_path),
        "materialization_receipt_path": str(batch_receipt_path),
        "materialization_receipt_sha256": sha256_file(batch_receipt_path),
        "checked_source_revisions": len(geometry),
        "checked_native_frames": frame_count,
        "checked_annotation_ids": len(regions),
        "labelled_source_revisions": len(labelled_sources),
        "class_and_shape_counts": {
            f"{phase}:{shape}": count for (phase, shape), count in sorted(counts.items())},
        "min_annotation_bbox_fraction_inside_native_content": min_overlap,
        "native_label_geometry_unchanged": True,
        "checked_annotated_frames": len(annotated_frames),
        "eligible_annotation_union_pixels_checked": annotation_signal_pixels,
        "invalid_eligible_annotation_signal_pixels": 0,
        "eligible_annotation_pixels_outside_corrected_domain": annotations_outside_corrected_domain,
        "run_native_physical_context_source_bound": True,
        "prior_shared_facies_context_preserved": False,
        "coordinate_system": "normalized-native-review-canvas; source-exact",
        "semantic_phase_alignment_certified": False,
        "scientific_acceptance": False,
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-request", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.frozen_request, args.materialization_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
