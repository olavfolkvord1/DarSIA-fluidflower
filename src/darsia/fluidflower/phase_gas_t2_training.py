"""Source-bound G3 gas refit with the sealed T2 aqueous boxes only.

Old G3 rows are imported verbatim from their sealed extraction, including
round-1 images that no longer occur in the camera-clock review timeline.
New boxes retain their own frame geometry. This module has no test-label,
blind-session, CO2-training, or model-publication entry point.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from .phase_training import logistic
from .phase_training.review_regions import sample_mask
from .phase_training.trainer import RunSource
from .project import sha256_file
from .review_storage import resolve_review_path

CONTRACT = "ffac-g3-t2-pixel-gas-refit-v1"
T2_MANIFEST_SHA256 = "2cf04230c35f8615f6ac3398de93c18382f59c6ad38a41d05a28032bf72bd484"
T2_ANNOTATIONS_SHA256 = "f0d8d400a75ce2020fe0b0077a210f9134d3e6fcf8f32a1908104b507be702e0"
G3_SHA256 = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
G3_LABEL_SHA256 = "02767870373795de7ea5a0f748f8e74239a4def8c85c196d84bba8b02ad3511b"
ROUND1_ANNOTATIONS_SHA256 = "56c0a0477e244f4ad9b15f5370b8d97ec9ddecf058a3053084426f4d589c7dbc"
CO2_SHA256 = "8c3727110be01dbef4849fad6849e50bcc8e28bdd1ee22a1c24c0ab8027c64e3"
T2_RUNS = frozenset(("ac25", "ac28", "ac39", "ac47", "ac50", "ac56"))
BLIND_RUNS = frozenset(("ac30", "ac49", "ac54"))
FLATFIELD_NAMES = ("ff_log_r", "ff_log_g", "ff_log_b", "ff_log_y")
HEAD = "gas_within_co2"


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value, *, progress=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = safe(value)
    if path.is_file() and not progress:
        if read(path) != value:
            raise ValueError(f"Existing immutable T2 receipt differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def checked(path, expected):
    path = resolve_review_path(path)
    if sha256_file(path) != expected:
        raise ValueError(f"Sealed T2/G3 artifact changed: {path}")
    return path


def label_digest(regions):
    return digest([dict(region_id=r.region_id, run=r.run.casefold(), phase_class=r.phase_class,
                        confidence=r.confidence, source=r.source, payload=dict(r.payload))
                   for r in regions])


def load_t2_source_bundle(candidate, snapshot):
    """Read only the SHA-whitelisted training snapshot and its own sources."""
    candidate, snapshot = resolve_review_path(candidate).resolve(), Path(snapshot).resolve()
    manifest_path = checked(snapshot / "MANIFEST.json", T2_MANIFEST_SHA256)
    manifest = read(manifest_path)
    annotations = checked(snapshot / "pairwise_review_annotations.csv", T2_ANNOTATIONS_SHA256)
    sums_path = checked(snapshot / "SHA256SUMS.txt", manifest["sha256sums_file_sha256"])
    sums = {}
    for line in sums_path.read_text(encoding="utf-8-sig").splitlines():
        value, name = line.split(None, 1)
        sums[name.strip().removeprefix("./")] = value
    session_path = checked(snapshot / "review_session.json", sums["review_session.json"])
    session = read(session_path)
    if (manifest["candidate_viewed"] != candidate.name or session["target_model_id"] != candidate.name
            or manifest["annotations_csv_sha256"] != T2_ANNOTATIONS_SHA256
            or set(manifest["runs"]) != T2_RUNS):
        raise ValueError("T2 snapshot belongs to another candidate or cohort")
    validation_path = resolve_review_path(session["combined_root"]) / "validation_binding.json"
    validation = read(validation_path)
    if validation["model_id"] != candidate.name or resolve_review_path(validation["model_root"]).resolve() != candidate:
        raise ValueError("T2 review preparation belongs to another candidate")
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    categories = Counter(row["category"] for row in rows)
    if categories != {"confirmed_only_aqueous_co2": 223, "incomplete_wash_indicator_carryover": 2}:
        raise ValueError("T2 categories or box count differs from the sealed declaration")
    groups, regions, sources, excluded = [], [], [], []
    for run in sorted(T2_RUNS):
        native_path = candidate / "source_alias" / "native_frames" / run / "manifest.json"
        native = read(native_path)
        if native["run"] != run or native["dataset"] != "ffac":
            raise ValueError("T2 native source belongs to another run")
        marker = native.get("ffac_rig_camera_sources", {})
        if marker.get("contract") != "ffac-rig-camera-native-sources-v1" or marker.get("inference_authorized") is not False:
            raise ValueError("T2 requires its exact separate rig/camera source contract")
        source = RunSource(run=run, dataset="ffac", review_root=candidate / "source_alias",
                           asset_manifest_path=native_path,
                           decisions_path=snapshot / "pairwise_review_decisions.json",
                           combined_manifest_path=validation_path)
        local = []
        for row in rows:
            if row["run"] != run:
                continue
            if row["category"] == "incomplete_wash_indicator_carryover":
                excluded.append(dict(annotation_id=row["annotation_id"], run=run,
                                     reason="carryover-is-not-a-gas-training-negative"))
                continue
            index, elapsed = int(row["frame_index"]), float(row["elapsed_h"])
            if not 0 <= index < len(native["frame_sources"]):
                raise ValueError("T2 image is outside its sealed own-source timeline")
            frame = native["frame_sources"][index]
            if (row["candidate_id"] != session["selected_candidates"][run]
                    or row["candidate_id"] != validation["selected_candidates"][run]
                    or row["params_sha256"] != validation["pipeline_sha256"]
                    or not math.isclose(elapsed, float(frame["elapsed_h"]), rel_tol=0, abs_tol=1e-8)
                    or row["coordinate_space"] != "normalized_native_review_canvas_16x9"
                    or row["shape"] != "rectangle"):
                raise ValueError(f"T2 annotation lost its exact source/geometry: {row['annotation_id']}")
            payload = dict(row)
            for name in ("x", "y", "width", "height"):
                value = float(row[name])
                if not math.isfinite(value) or value < 0 or value > 1:
                    raise ValueError("Invalid normalized T2 box coordinate")
                payload[name] = value
            payload.update(id=row["annotation_id"], points=[], frame_index=index, elapsed_h=elapsed,
                           phase_class="aqueous", training_phase_class="aqueous",
                           training_scope="whole-marked-region", training_eligible=True,
                           t2_annotation_sha256=T2_ANNOTATIONS_SHA256,
                           source_native_manifest_sha256=sha256_file(native_path),
                           source_jpeg_sha256=frame["source_sha256"],
                           training_revision="G3-T2", source_session_id=session["review_session_id"])
            region = logistic.ReviewedRegion(region_id=f"{run}:manual:{row['annotation_id']}",
                run=run, dataset="ffac", frame_index=index, elapsed_h=elapsed,
                phase_class="aqueous", confidence=row["confidence"].casefold(), source="manual",
                reviewer=row["reviewer"], payload=payload)
            if region.supervision_weight <= 0:
                raise ValueError("A sealed T2 explicit negative has zero supervision weight")
            local.append(region)
        groups.append((source, tuple(local)))
        regions.extend(local)
        sources.append(dict(run=run, native_manifest_path=str(native_path),
                            native_manifest_sha256=sha256_file(native_path),
                            frame_indices=sorted({r.frame_index for r in local}), regions=len(local)))
    if len(regions) != 223 or len({r.region_id for r in regions}) != 223 or len(excluded) != 2:
        raise ValueError("Duplicate or missing T2 training boxes")
    logistic.assert_safe_training_regions(regions)
    receipt = dict(contract=CONTRACT, snapshot_manifest_path=str(manifest_path),
                   snapshot_manifest_sha256=T2_MANIFEST_SHA256,
                   annotation_csv_path=str(annotations), annotation_csv_sha256=T2_ANNOTATIONS_SHA256,
                   session_path=str(session_path), session_sha256=sha256_file(session_path),
                   session_id=session["review_session_id"], validation_binding_path=str(validation_path),
                   validation_binding_sha256=sha256_file(validation_path),
                   source_bindings=sources, excluded_labels=excluded, box_count=223,
                   region_ids_sha256=digest([r.region_id for r in regions]),
                   rounds_2_3_4_read=False, new_blind_session_labels_read=False,
                   old_blind_run_inventory_policy="retain-existing-G3-regions-only",
                   normalized_canvas_geometry="own-native-photograph-canvas-before-rig-origin-translation")
    return dict(groups=groups, regions=regions, new_region_ids={r.region_id for r in regions}, receipt=receipt)


def load_frozen_g3_data(experiment, extraction):
    """Use exact fitted G3 matrices and owners; no historical re-extraction."""
    experiment, extraction = resolve_review_path(experiment), resolve_review_path(extraction)
    head_path = checked(experiment / "variants/G3/gas_head.json", G3_SHA256)
    head = read(head_path)
    report = read(experiment / "variants/G3/training_report.json")
    receipt = read(extraction / "extraction_receipt.json")
    regions_path = checked(extraction / "regions.json", receipt["region_inventory_sha256"])
    checked(extraction / "training_samples.npz", receipt["training_samples_sha256"])
    regions = [logistic.ReviewedRegion(**row) for row in read(regions_path)]
    if (label_digest(regions) != G3_LABEL_SHA256 or report["training_label_sha256"] != G3_LABEL_SHA256
            or head["training_provenance"]["training_label_sha256"] != G3_LABEL_SHA256
            or [r.region_id for r in regions] != report["training_region_ids"]
            or receipt["new_annotation_sha256"] != ROUND1_ANNOTATIONS_SHA256
            or len(receipt["new_region_ids"]) != 871 or len(regions) != 2849
            or tuple(head["feature_names"]) != FLATFIELD_NAMES):
        raise ValueError("Original G3 dataset/round-1 inventory changed")
    with np.load(extraction / "training_samples.npz", allow_pickle=False) as archive:
        features = archive["flatfield_features"].copy()
        owners = archive["region_indices"].copy()
    if features.dtype != np.float32 or features.shape[1:] != (4,) or owners.dtype != np.int32 or owners.shape != features.shape[:1] or not np.isfinite(features).all():
        raise ValueError("Frozen G3 features are not exact finite float32 rows")
    from .phase_gas_variant_sources import HELD_OUT_RUNS
    if {r.run for r in regions} & HELD_OUT_RUNS:
        raise ValueError("Original G3 inventory unexpectedly includes held-out runs")
    # Only this pre-G3 whitelist can be opened for historical supervision.
    label_sources = []
    for bundle in receipt["old_training_source_snapshots"]:
        for role, artifact in bundle.items():
            path = checked(artifact["path"], artifact["sha256"])
            label_sources.append(dict(role=role, path=str(path), sha256=artifact["sha256"]))
    round1 = receipt["new_labels"]
    checked(round1["annotation_csv_path"], ROUND1_ANNOTATIONS_SHA256)
    session_path = Path(round1["annotation_csv_path"]).parent / "review_session.json"
    checked(session_path, round1["review_session_sha256"])
    label_sources.append(dict(role="round1_gas_outlines", path=round1["annotation_csv_path"],
                              sha256=ROUND1_ANNOTATIONS_SHA256,
                              session_sha256=round1["review_session_sha256"]))
    return dict(features=features, owners=owners, regions=regions, receipt=receipt,
                head=head, head_path=head_path, report=report,
                experiment=experiment, extraction=extraction, label_sources=label_sources)


def ac31_off_timeline_audit(frozen, base_candidate):
    """Prove all round-1 AC31 labels stay on their own original camera image."""
    new_ids = set(frozen["receipt"]["new_region_ids"])
    source = next(row for row in frozen["receipt"]["new_labels"]["sources"] if row["run"] == "ac31")
    old_path = checked(source["native_manifest_path"], source["native_manifest_sha256"])
    old = read(old_path)
    current_path = Path(base_candidate) / "source_alias/native_frames/ac31/manifest.json"
    current = read(current_path)
    current_shas = {frame["source_sha256"] for frame in current["frame_sources"]}
    rows = []
    counts = np.bincount(frozen["owners"], minlength=len(frozen["regions"]))
    for i, region in enumerate(frozen["regions"]):
        if region.run != "ac31" or region.region_id not in new_ids:
            continue
        frame = old["frame_sources"][region.frame_index]
        if frame["source_sha256"] not in current_shas:
            if counts[i] == 0:
                raise ValueError("Off-timeline round-1 AC31 contour lost its frozen feature rows")
            rows.append(dict(region_id=region.region_id, original_frame_index=region.frame_index,
                original_elapsed_h=region.elapsed_h, source_sha256=frame["source_sha256"],
                scientific_sidecar_sha256=frame["v5_log_linear_rgb_sha256"],
                imported_sample_rows=int(counts[i])))
    if len(rows) != 34:
        raise ValueError(f"Expected 34 retained AC31 off-timeline labels, found {len(rows)}")
    return dict(round1_ac31_regions=234, off_timeline_regions=34, own_source_features_preserved=True,
                original_manifest_path=str(old_path), original_manifest_sha256=sha256_file(old_path),
                current_manifest_path=str(current_path), current_manifest_sha256=sha256_file(current_path),
                labels=rows)


def load_combined_training_sources(old_candidate, round1_snapshot, base_candidate, t2_snapshot):
    """Source groups for regional refitting in the original G3 order."""
    from .phase_gas_variant_sources import load_training_sources
    old = load_training_sources(Path(old_candidate), Path(round1_snapshot))
    old_regions = [region for _source, group in old["groups"] for region in group]
    if label_digest(old_regions) != G3_LABEL_SHA256:
        raise ValueError("Reconstructed source groups differ from frozen G3 inventory")
    new = load_t2_source_bundle(base_candidate, t2_snapshot)
    ids = {r.region_id for r in old_regions}
    if ids & new["new_region_ids"]:
        raise ValueError("T2 region ID duplicates an original G3 label")
    groups = [*old["groups"], *new["groups"]]
    all_regions = [r for _source, group in groups for r in group]
    receipt = dict(contract=CONTRACT, old_g3_label_sha256=G3_LABEL_SHA256,
                   old_source_bundle=old["receipt"], t2=new["receipt"],
                   combined_label_sha256=label_digest(all_regions),
                   combined_region_ids_sha256=digest([r.region_id for r in all_regions]),
                   old_blind_run_region_counts=dict(Counter(r.run for r in old_regions if r.run in BLIND_RUNS)),
                   new_blind_session_labels_read=False, rounds_2_3_4_read=False)
    return dict(groups=groups, regions=all_regions, old_regions=old_regions,
                new_regions=new["regions"], new_region_ids=new["new_region_ids"], receipt=receipt)


def supervision_weight_report(features, owners, regions, t2_ids):
    """Report actual run/class-balanced pixel loss mass, not box pixel area."""
    _matrix, targets, weights, selected_owners = logistic.balanced_pixel_view(
        features, owners, regions, head=HEAD)
    total = float(weights.sum())
    new = np.isin(selected_owners, [i for i, r in enumerate(regions) if r.region_id in t2_ids])
    groups = dict(t2_negatives=new, existing_negatives=(targets == 0) & ~new,
                  existing_positives=targets == 1)
    report = {}
    for name, selected in groups.items():
        indices = np.unique(selected_owners[selected])
        report[name] = dict(regions=int(len(indices)), sample_rows=int(selected.sum()),
            raw_box_weight=sum(regions[int(i)].supervision_weight for i in indices),
            balanced_loss_weight=float(weights[selected].sum()),
            fraction_of_direct_pixel_loss=float(weights[selected].sum() / total))
    report["policy"] = "confidence-times-source per box; equal (run,target) strata; uniform mass within each box"
    return report


def training_counts(regions, owners):
    rows = np.bincount(owners, minlength=len(regions))
    result = defaultdict(lambda: defaultdict(lambda: dict(regions=0, sample_rows=0, supervision_weight=0.)))
    for i, region in enumerate(regions):
        entry = result[region.run][region.phase_class]
        entry["regions"] += 1
        entry["sample_rows"] += int(rows[i])
        entry["supervision_weight"] += region.supervision_weight
    return {run: dict(classes) for run, classes in sorted(result.items())}


def scientific_descriptor_identity(descriptor):
    """Pixel identity survives certified copies made by the old recovery.

    Producer job receipts and NAS locations can differ. Source, grid,
    encoding, validity and every individual pixel-file SHA may not.
    """
    centre = descriptor["native_centre"]
    return digest(dict(contract=descriptor["contract"], source_sha256=descriptor["source_sha256"],
        phase_shape_hw=descriptor["phase_shape_hw"], native_origin_xy=descriptor["native_origin_xy"],
        unit_denominator=descriptor["unit_denominator"], storage=descriptor["storage"],
        centre_source_sha256=centre["source_sha256"],
        centre_log_sha256=centre["log_linear_rgb"]["sha256"],
        centre_valid_sha256=centre["valid_mask"]["sha256"],
        ring_windows=[dict(canvas_xyxy=row["canvas_xyxy"],
                           log_sha256=row["log_linear_rgb"]["sha256"])
                      for row in descriptor["ring_windows"]]))


def _extract_t2_run(candidate, run, serialized_regions, destination, pixels_per_region):
    """Production flat-field expression on SHA-bound full rig scientific input."""
    from .phase_ffac_rig_camera_runtime import bound_path
    from .phase_pixel_area import load_rig_classes
    from .phase_rig_domain_inference import masked_flatfield_features
    from .tools import run_ffac_rig_camera_inference as producer
    from .tools.phase_coupled_constraint_comparison import annotation_window
    cv2.setNumThreads(1)
    started = time.perf_counter()
    candidate, destination = Path(candidate), Path(destination)
    regions = [logistic.ReviewedRegion(**row) for row in serialized_regions]
    native = read(candidate / "source_alias/native_frames" / run / "manifest.json")
    matrices, ownership, coverage, frames = [], [], [], []
    baseline = baseline_valid = None
    baseline_sha = None
    verified_baseline_descriptors = set()
    for index in sorted({r.frame_index for r in regions}):
        path = candidate / "rig_camera_frames" / run.upper() / f"frame_{index:03d}.json"
        frame = read(path)
        record = native["frame_sources"][index]
        if (not frame.get("complete") or frame["source_sha256"] != record["source_sha256"]
                or frame["elapsed_h"] != record["elapsed_h"]):
            raise ValueError("T2 production receipt source differs from the annotated image")
        descriptor = frame["baseline_extended_scientific"]
        this_baseline = scientific_descriptor_identity(descriptor)
        full_descriptor_sha = digest(descriptor)
        if full_descriptor_sha not in verified_baseline_descriptors:
            for asset in [descriptor["native_centre"]["log_linear_rgb"],
                          descriptor["native_centre"]["valid_mask"],
                          *[row["log_linear_rgb"] for row in descriptor["ring_windows"]]]:
                bound_path(asset)
            verified_baseline_descriptors.add(full_descriptor_sha)
        if baseline is None:
            baseline, baseline_valid = producer.compose_extended_scientific(descriptor)
            baseline_sha = this_baseline
        elif baseline_sha != this_baseline:
            raise ValueError("T2 run changed its frozen pre-injection baseline")
        current, current_valid = producer.compose_extended_scientific(frame["full_extended_scientific"])
        source = frame["binding"]["source"]
        prepared = read(bound_path(frame["full_extended_scientific"]["binding"]["prepared_source"]))
        h = producer.canvas_homography(np.asarray(prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64))
        aligned, aligned_valid = producer.align_scientific(baseline, baseline_valid, h, target_shape_hw=current.shape[1:])
        classes, _classes_binding = load_rig_classes(frame["rig_domain_revision"])
        valid = classes["rig_domain"] & current_valid & aligned_valid
        predecessor = source.get("temporal_predecessor")
        if predecessor is not None:
            previous_rows = [row for row in native["frame_sources"]
                             if row["source_sha256"] == predecessor["source_sha256"]]
            if len(previous_rows) != 1:
                raise ValueError("T2 source predecessor requires its exact source descriptor")
            previous_row = previous_rows[0]
            previous_log, previous_valid = producer.compose_extended_scientific(
                previous_row["full_extended_scientific"])
            previous_prepared = read(bound_path(previous_row["full_extended_scientific"]["binding"]["prepared_source"]))
            h_previous = np.asarray(previous_prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
            h_native = np.asarray(prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
            previous_aligned, previous_support = producer.align_scientific(previous_log, previous_valid,
                producer.canvas_homography(h_native @ np.linalg.inv(h_previous)),
                target_shape_hw=current.shape[1:])
            valid &= previous_support
            del previous_log, previous_aligned, previous_valid, previous_support
        # Native G3 filtering uses this complete supported rig neighbourhood.
        # Existing unsupported/fill pixels are never admitted as sample rows.
        features, supported = masked_flatfield_features(current, aligned, valid,
            source_baseline_log=baseline, source_baseline_valid=baseline_valid,
            h_baseline_to_current=h)
        supported &= classes["modelled"]
        frame_coverage = []
        for owner, region in enumerate(regions):
            if region.frame_index != index:
                continue
            shape = current.shape[1:]
            bounds, mask = annotation_window(region.payload, canvas_wh=native["canvas_wh"],
                content_box=(0, 0, shape[1], shape[0]))
            x0, y0, x1, y1 = bounds
            selection = mask & supported[y0:y1, x0:x1]
            rows, columns = sample_mask(selection, limit=int(pixels_per_region), annotation_id=region.region_id)
            if not rows.size:
                raise ValueError(f"T2 explicit gas-negative box has no observed feature rows: {region.region_id}")
            values = features[rows+y0, columns+x0]
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("T2 gas features must remain finite float32")
            matrices.append(values.copy())
            ownership.append(np.full(len(values), owner, np.int32))
            item = dict(region_id=region.region_id, run=run, frame_index=index,
                source_sha256=record["source_sha256"], annotation_pixels=int(mask.sum()),
                supported_pixels=int(selection.sum()), sample_rows=len(values),
                sampling_method="same-G3-deterministic-uniform-1024-per-box-valid-native-pixels",
                stored_canvas_wh=native["canvas_wh"], phase_shape_hw=list(shape))
            coverage.append(item)
            frame_coverage.append(item)
        frames.append(dict(frame_index=index, source_sha256=record["source_sha256"],
                           producer_receipt_path=str(path), producer_receipt_sha256=sha256_file(path),
                           scientific_descriptor_sha256=digest(frame["full_extended_scientific"]),
                           baseline_descriptor_sha256=this_baseline,
                           baseline_producer_declaration_sha256=full_descriptor_sha,
                           H_baseline_canvas_to_frame_canvas=h.tolist(), regions=len(frame_coverage)))
        del current, aligned, features, classes, supported, valid
    matrix, owners = np.concatenate(matrices), np.concatenate(ownership)
    destination.mkdir(parents=True, exist_ok=True)
    sample_path = destination / "samples.npz"
    if sample_path.exists():
        raise ValueError("T2 extraction output exists without a committed matching receipt")
    with sample_path.open("xb") as stream:
        np.savez(stream, features=matrix, region_indices=owners)
    receipt = dict(contract=CONTRACT, run=run, sample_path=str(sample_path),
                   samples_sha256=sha256_file(sample_path), frame_bindings=frames, coverage=coverage,
                   region_count=len(regions), sample_rows=len(matrix), wall_seconds=time.perf_counter()-started)
    write(destination / "receipt.json", receipt)
    return receipt


def prepare_dataset(experiment, extraction, base_candidate, t2_snapshot, output, *, workers=6):
    """Import unchanged G3 features and append source-exact T2 negatives."""
    output = Path(output)
    frozen = load_frozen_g3_data(experiment, extraction)
    new = load_t2_source_bundle(base_candidate, t2_snapshot)
    regions = [*frozen["regions"], *new["regions"]]
    if len({r.region_id for r in regions}) != len(regions):
        raise ValueError("T2 duplicates an original G3 region")
    initial = dict(contract=CONTRACT, frozen_g3_head_sha256=G3_SHA256,
                   old_extraction_receipt_sha256=sha256_file(Path(extraction) / "extraction_receipt.json"),
                   old_region_inventory_sha256=frozen["receipt"]["region_inventory_sha256"],
                   old_training_samples_sha256=frozen["receipt"]["training_samples_sha256"],
                   t2=new["receipt"], code_sha256=sha256_file(Path(__file__)))
    write(output / "initial_binding.json", initial)
    regions_path = output / "regions.json"
    write(regions_path, [asdict(r) for r in regions])
    reports, pending = {}, []
    for source, group in new["groups"]:
        destination = output / "t2_extraction" / source.run
        receipt_path = destination / "receipt.json"
        if receipt_path.is_file():
            receipt = read(receipt_path)
            checked(receipt["sample_path"], receipt["samples_sha256"])
            if [r["region_id"] for r in receipt["coverage"]] != [r.region_id for r in sorted(group, key=lambda r: r.frame_index)]:
                raise ValueError("Resumed T2 extraction region order changed")
            reports[source.run] = receipt
        else:
            pending.append((source.run, [asdict(r) for r in group], str(destination)))
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(max(1, int(workers)), len(pending)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_extract_t2_run, str(base_candidate), run, group, destination,
                                  frozen["receipt"]["sampling"]["pixels_per_region"]): run
                       for run, group, destination in pending}
            for future in as_completed(futures):
                report = future.result()
                reports[report["run"]] = report
                write(output / "progress.json", dict(stage="t2_extracting", completed=len(reports),
                      total=len(new["groups"]), last_run=report["run"], seconds=time.perf_counter()-started), progress=True)
                print(json.dumps(dict(stage="t2_run_extracted", run=report["run"], rows=report["sample_rows"])), flush=True)
    matrices, owners = [frozen["features"]], [frozen["owners"]]
    offset = len(frozen["regions"])
    for source, group in new["groups"]:
        with np.load(reports[source.run]["sample_path"], allow_pickle=False) as archive:
            matrices.append(archive["features"].copy())
            owners.append(archive["region_indices"].copy() + offset)
        offset += len(group)
    features, ownership = np.concatenate(matrices), np.concatenate(owners)
    if not np.array_equal(features[:len(frozen["features"])], frozen["features"]) or not np.array_equal(ownership[:len(frozen["owners"])], frozen["owners"]):
        raise ValueError("Imported G3 samples changed during T2 append")
    for name, array in (("flatfield.npy", features), ("owners.npy", ownership)):
        path = output / name
        if path.is_file():
            if not np.array_equal(np.load(path, allow_pickle=False), array):
                raise ValueError("Resumed T2 merged dataset differs")
        else:
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
    receipt = dict(**initial, combined_label_sha256=label_digest(regions),
                   combined_regions=len(regions), old_regions=len(frozen["regions"]), t2_regions=223,
                   original_round1_region_count=871, old_sample_rows=len(frozen["features"]),
                   t2_sample_rows=len(features)-len(frozen["features"]),
                   old_features_and_owner_indices_bit_identical=True,
                   original_ac31_off_timeline=ac31_off_timeline_audit(frozen, base_candidate),
                   training_label_sources=frozen["label_sources"],
                   region_inventory_path=str(regions_path), region_inventory_sha256=sha256_file(regions_path),
                   feature_path=str(output / "flatfield.npy"), feature_sha256=sha256_file(output / "flatfield.npy"),
                   owner_path=str(output / "owners.npy"), owner_sha256=sha256_file(output / "owners.npy"),
                   new_region_ids=sorted(new["new_region_ids"]),
                   training_counts=training_counts(regions, ownership),
                   weights=supervision_weight_report(features, ownership, regions, new["new_region_ids"]),
                   old_blind_run_region_counts=dict(Counter(r.run for r in frozen["regions"] if r.run in BLIND_RUNS)),
                   source_feature_reports=[reports[run] for run in sorted(reports)],
                   training_input_policy="exact-frozen-G3-rows-plus-own-source-T2-production-flatfield-rows",
                   rounds_2_3_4_read=False, new_blind_session_labels_read=False,
                   extraction_seconds=time.perf_counter()-started)
    write(output / "dataset_receipt.json", receipt)
    write(output / "progress.json", dict(stage="dataset_ready", regions=len(regions), rows=len(features)), progress=True)
    return receipt


def fit_pixel_g3_t2(experiment, dataset_root, output, *, threads=6):
    """One CPU MIL fit at the fixed G3 regularization and stopping tolerance."""
    from threadpoolctl import threadpool_limits
    dataset_root, output, experiment = Path(dataset_root), Path(output), resolve_review_path(experiment)
    receipt = read(dataset_root / "dataset_receipt.json")
    head_path = checked(experiment / "variants/G3/gas_head.json", G3_SHA256)
    old_head = read(head_path)
    checked(receipt["feature_path"], receipt["feature_sha256"])
    checked(receipt["owner_path"], receipt["owner_sha256"])
    checked(receipt["region_inventory_path"], receipt["region_inventory_sha256"])
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    regions = [logistic.ReviewedRegion(**r) for r in read(receipt["region_inventory_path"])]
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 4 or owners.dtype != np.int32 or owners.shape != features.shape[:1] or not np.isfinite(features).all():
        raise ValueError("T2 MIL input must be finite float32 scientific feature rows")
    if label_digest(regions) != receipt["combined_label_sha256"]:
        raise ValueError("Merged G3/T2 label SHA differs from extraction")
    logistic.assert_safe_training_regions(regions)
    settings = dict(old_head["training_provenance"]["fit_settings"])
    if settings["backend"] != "cpu" or not 1 <= int(threads) <= 32:
        raise ValueError("T2 MIL fit requires a bounded CPU-only allocation")
    initial = dict(contract=CONTRACT, dataset_receipt_sha256=sha256_file(dataset_root / "dataset_receipt.json"),
                   parent_head_sha256=G3_SHA256, fit_settings=settings,
                   cpu_only=True, threads=int(threads), code_sha256={
                       "trainer": sha256_file(Path(__file__)), "logistic": sha256_file(Path(logistic.__file__))})
    write(output / "initial_binding.json", initial)
    write(output / "progress.json", dict(stage="pixel_MIL_fit_running", started_utc=datetime.now(timezone.utc).isoformat()), progress=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    cv2.setNumThreads(1)
    with threadpool_limits(limits=int(threads)):
        fit, diagnostics = logistic.fit_weighted_logistic_mil(features, owners, regions,
            head=HEAD, runs=None, **settings)
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    model = dict(format=old_head["format"], head=HEAD, variant="G3", training_revision="G3-T2",
                 feature_contract=old_head["feature_contract"], feature_names=list(FLATFIELD_NAMES),
                 fit=safe(asdict(fit)), gas_display_threshold=.6,
                 training_provenance=dict(**initial,
                     combined_label_sha256=receipt["combined_label_sha256"],
                     dataset_receipt_path=str(dataset_root / "dataset_receipt.json"),
                     unchanged_co2_head_sha256=CO2_SHA256,
                     pixel_threshold=.7, regional_threshold=.6, gas_minimum_mm2=2.5,
                     connectivity=4, rounds_2_3_4_read=False, new_blind_session_labels_read=False,
                     training_scope="same-G3-inventory-plus-223-sealed-T2-aqueous-boxes",
                     mass_eligible=False))
    write(output / "gas_head.json", model)
    result = dict(**initial, converged=bool(fit.converged), fit_wall_seconds=wall, fit_cpu_seconds=cpu,
                  iterations=fit.iterations, old_fit=old_head["fit"], new_fit=safe(asdict(fit)),
                  diagnostics=diagnostics, model_path=str(output / "gas_head.json"),
                  model_sha256=sha256_file(output / "gas_head.json"),
                  weights=receipt["weights"], training_counts=receipt["training_counts"],
                  training_label_sha256=receipt["combined_label_sha256"],
                  training_region_ids=[r.region_id for r in regions],
                  training_region_inventory_path=receipt["region_inventory_path"],
                  training_region_inventory_sha256=receipt["region_inventory_sha256"],
                  old_blind_run_region_counts=receipt["old_blind_run_region_counts"],
                  rounds_2_3_4_read=False, new_blind_session_labels_read=False,
                  created_utc=datetime.now(timezone.utc).isoformat())
    write(output / "training_report.json", result)
    write(output / "fit_receipt.json", result)
    write(output / "progress.json", dict(stage="pixel_fit_complete", converged=bool(fit.converged),
          model_sha256=result["model_sha256"], wall_seconds=wall), progress=True)
    if not fit.converged:
        raise ValueError("T2 gas MIL fit did not converge; diagnostics preserved and downstream blocked")
    return result


def pixel_sample_in_sample_report(experiment, dataset_root, fitted_head, output):
    """Sample-based §4b supplement; final map QA is reported separately.

    The frozen deterministic training coordinates are evaluated without any
    threshold selection. A per-box mean avoids making large boxes heavier.
    Off-timeline round-1 contours retain the same original source feature rows.
    """
    from .tools.train_ffac_gas_flatfield_variants import gas_probability
    dataset_root = Path(dataset_root)
    receipt = read(dataset_root / "dataset_receipt.json")
    for path_key, sha_key in (("feature_path", "feature_sha256"),
                              ("owner_path", "owner_sha256"),
                              ("region_inventory_path", "region_inventory_sha256")):
        checked(receipt[path_key], receipt[sha_key])
    old_path = checked(Path(experiment) / "variants/G3/gas_head.json", G3_SHA256)
    old, new = read(old_path), read(fitted_head)
    if new.get("training_revision") != "G3-T2" or new["training_provenance"]["combined_label_sha256"] != receipt["combined_label_sha256"]:
        raise ValueError("In-sample QA head belongs to another sealed training dataset")
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    regions = [logistic.ReviewedRegion(**r) for r in read(receipt["region_inventory_path"])]
    old_scores, new_scores = [gas_probability(head, np.empty((len(features), 0), np.float32), features)
                              for head in (old, new)]
    counts = np.bincount(owners, minlength=len(regions))
    old_gas_counts, new_gas_counts = [np.bincount(owners, weights=(scores >= .7), minlength=len(regions))
                                    for scores in (old_scores, new_scores)]
    t2_ids = set(receipt["new_region_ids"])
    round1_ids = {r.region_id for r in regions if r.payload.get("gas_variant_annotation_sha256") == ROUND1_ANNOTATIONS_SHA256}
    rows = []
    for index, region in enumerate(regions):
        kind = "T2-explicit-aqueous-negative" if region.region_id in t2_ids else "round1-gas-positive" if region.region_id in round1_ids else None
        if kind is None:
            continue
        if counts[index] == 0:
            raise ValueError("Sealed T2/round1 region has no imported native feature sample")
        rows.append(dict(region_id=region.region_id, run=region.run, frame_index=region.frame_index,
            elapsed_h=region.elapsed_h, kind=kind, sample_rows=int(counts[index]),
            old_gas_fraction=float(old_gas_counts[index] / counts[index]),
            new_gas_fraction=float(new_gas_counts[index] / counts[index]),
            source_native_manifest_sha256=region.payload.get("source_native_manifest_sha256"),
            source_jpeg_sha256=region.payload.get("source_jpeg_sha256")))
    aggregates = []
    keys = sorted({(r["kind"], r["run"], r["frame_index"]) for r in rows})
    for kind, run, frame in keys:
        group = [r for r in rows if (r["kind"], r["run"], r["frame_index"]) == (kind, run, frame)]
        aggregates.append(dict(kind=kind, run=run, frame_index=frame, regions=len(group),
            old_equal_box_gas_fraction=float(np.mean([r["old_gas_fraction"] for r in group])),
            new_equal_box_gas_fraction=float(np.mean([r["new_gas_fraction"] for r in group]))))
    if Counter(r["kind"] for r in rows) != {"T2-explicit-aqueous-negative": 223, "round1-gas-positive": 871}:
        raise ValueError("In-sample QA omitted T2 or round1 supervision")
    report = dict(contract=CONTRACT, scope="in-sample scientific training coordinates; pixel threshold .7 only; not downstream final maps",
        independent_validation=False, final_map_qa=False, old_head_sha256=G3_SHA256,
        new_head_sha256=sha256_file(fitted_head), dataset_receipt_sha256=sha256_file(dataset_root / "dataset_receipt.json"),
        regions=rows, per_run_frame=aggregates, original_ac31_off_timeline=receipt["original_ac31_off_timeline"],
        rounds_2_3_4_read=False, new_blind_session_labels_read=False)
    write(output, report)
    return report
