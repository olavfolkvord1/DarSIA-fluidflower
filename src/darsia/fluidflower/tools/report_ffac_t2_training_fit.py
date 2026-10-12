"""Source-bound full native §4b training-box QA after the T2 candidate exists.

Only sealed G3 round-1 and T2 training annotations are read. Native final-map
metrics are separate from the four-feature, sampled pixel-head supplement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .. import phase_gas_t2_training as training
from ..phase_pixel_area import load_rig_classes
from ..phase_training.logistic import ReviewedRegion
from .phase_coupled_constraint_comparison import annotation_window

CONTRACT = "ffac-g3-t2-native-training-map-qa-v1"


def source_index(native):
    """Camera-source SHA is the identity; a changed timeline index is not."""
    result = {}
    for frame in native["frame_sources"]:
        sha = frame["source_sha256"]
        if sha in result:
            raise ValueError("Ambiguous duplicate camera source in training-map QA")
        result[sha] = frame
    return result


def phase_metrics(old, new, mask, shared_modelled):
    """Keep unknown coverage visible rather than silently changing denominator."""
    if old.shape != new.shape or mask.shape != old.shape or shared_modelled.shape != old.shape:
        raise ValueError("Training-map comparison changed native geometry")
    pixels = int(mask.sum())
    shared = mask & shared_modelled
    observed = int(shared.sum())
    if pixels == 0:
        raise ValueError("Training annotation rasterized to no native pixels")
    result = dict(annotation_pixels=pixels, shared_modelled_pixels=observed,
                  shared_modelled_fraction=observed / pixels)
    for name, labels in (("old", old), ("new", new)):
        gas = int(np.count_nonzero(mask & (labels == 2)))
        result.update({f"{name}_gas_pixels": gas,
                       f"{name}_gas_fraction": gas / pixels,
                       f"{name}_unresolved_pixels": int(np.count_nonzero(mask & ~np.isin(labels, (0, 1, 2)))),
                       f"{name}_shared_modelled_gas_fraction": (
                           float(np.count_nonzero(shared & (labels == 2)) / observed)
                           if observed else None)})
    return result


def _labels(asset):
    path = training.checked(asset["path"], asset["sha256"])
    if path.suffix.casefold() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            value = np.array(image)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Final-map QA requires the categorical phase grid")
    return value


def _frame_receipt(candidate, run, index):
    path = Path(candidate) / "rig_camera_frames" / run.upper() / f"frame_{index:03d}.json"
    # Dedicated runners have historically used upper-case run directories.
    # A lower-case directory is permitted only when that exact path exists.
    if not path.is_file():
        path = Path(candidate) / "rig_camera_frames" / run / f"frame_{index:03d}.json"
    result = training.read(path)
    if result.get("complete") is not True:
        raise ValueError("Training-map QA encountered an uncommitted frame")
    return result, dict(path=str(path), sha256=training.sha256_file(path))


def assess_frame(base_candidate, new_candidate, task):
    cv2.setNumThreads(1)
    run, index, sha = task["run"], task["frame_index"], task["source_sha256"]
    old_receipt, old_binding = _frame_receipt(base_candidate, run, index)
    new_receipt, new_binding = _frame_receipt(new_candidate, run, task["new_frame_index"])
    for receipt in (old_receipt, new_receipt):
        if receipt.get("source_sha256", receipt["binding"]["source"]["source_sha256"]) != sha:
            raise ValueError("Native map no longer belongs to the annotation's source")
    old, new = _labels(old_receipt["labels_map"]), _labels(new_receipt["labels_map"])
    old_classes, _ = load_rig_classes(old_receipt["rig_domain_revision"])
    new_classes, _ = load_rig_classes(new_receipt["rig_domain_revision"])
    if (old.shape != new.shape or not np.array_equal(old_classes["rig_domain"], new_classes["rig_domain"])
            or old_receipt["rig_domain_revision"]["area_map"]["sha256"]
            != new_receipt["rig_domain_revision"]["area_map"]["sha256"]):
        raise ValueError("T2 training-map QA altered geometry, domain or approved area")
    shared = old_classes["modelled"] & new_classes["modelled"]
    rows = []
    for region in task["regions"]:
        bounds, mask = annotation_window(region["payload"], canvas_wh=task["canvas_wh"],
                                         content_box=(0, 0, old.shape[1], old.shape[0]))
        x0, y0, x1, y1 = bounds
        metrics = phase_metrics(old[y0:y1, x0:x1], new[y0:y1, x0:x1], mask,
                                shared[y0:y1, x0:x1])
        rows.append(dict(region_id=region["region_id"], kind=region["kind"], run=run,
            original_annotation_frame_index=region["frame_index"],
            original_annotation_elapsed_h=region["elapsed_h"], frame_index=index,
            camera_elapsed_h=task["elapsed_h"], source_sha256=sha,
            annotation_sha256=region["annotation_sha256"], bounds_xyxy=list(bounds), **metrics))
    return dict(run=run, frame_index=index, source_sha256=sha,
                old_producer_receipt=old_binding, new_producer_receipt=new_binding,
                old_labels=old_receipt["labels_map"], new_labels=new_receipt["labels_map"], rows=rows)


def aggregate(rows):
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["run"], row["frame_index"])].append(row)
    for (kind, run, frame), group in sorted(groups.items()):
        item = dict(kind=kind, run=run, frame_index=frame, regions=len(group),
                    camera_elapsed_h=group[0]["camera_elapsed_h"])
        for name in ("old", "new"):
            item[f"{name}_equal_box_gas_fraction"] = float(np.mean([r[f"{name}_gas_fraction"] for r in group]))
            item[f"{name}_pixel_weighted_gas_fraction"] = sum(r[f"{name}_gas_pixels"] for r in group) / sum(r["annotation_pixels"] for r in group)
        item["shared_modelled_fraction"] = sum(r["shared_modelled_pixels"] for r in group) / sum(r["annotation_pixels"] for r in group)
        result.append(item)
    return result


def aggregate_runs(rows):
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["run"])].append(row)
    for (kind, run), group in sorted(groups.items()):
        item = dict(kind=kind, run=run, regions=len(group),
                    frames=len({row["frame_index"] for row in group}))
        for name in ("old", "new"):
            item[f"{name}_equal_box_gas_fraction"] = float(np.mean([r[f"{name}_gas_fraction"] for r in group]))
            item[f"{name}_pixel_weighted_gas_fraction"] = sum(r[f"{name}_gas_pixels"] for r in group) / sum(r["annotation_pixels"] for r in group)
        item["shared_modelled_fraction"] = sum(r["shared_modelled_pixels"] for r in group) / sum(r["annotation_pixels"] for r in group)
        result.append(item)
    return result


def report_native_training_maps(base_candidate, new_candidate, dataset_root,
                                old_extraction, output, *, workers=4):
    """Compare exact native final maps, matching round-1 source SHA rather than index."""
    base_candidate, new_candidate, dataset_root, output = map(Path,
        (base_candidate, new_candidate, dataset_root, output))
    data = training.read(dataset_root / "dataset_receipt.json")
    training.checked(data["region_inventory_path"], data["region_inventory_sha256"])
    regions = [ReviewedRegion(**row) for row in training.read(data["region_inventory_path"])]
    if training.label_digest(regions) != data["combined_label_sha256"]:
        raise ValueError("§4b dataset label binding changed")
    t2 = data["t2"]
    for pathkey, shakey in (("snapshot_manifest_path", "snapshot_manifest_sha256"),
                            ("annotation_csv_path", "annotation_csv_sha256"),
                            ("session_path", "session_sha256")):
        training.checked(t2[pathkey], t2[shakey])
    old_receipt_path = training.checked(Path(old_extraction) / "extraction_receipt.json",
                                        data["old_extraction_receipt_sha256"])
    original = training.read(old_receipt_path)
    round1_sources = {row["run"]: row for row in original["new_labels"]["sources"]}
    training.checked(original["new_labels"]["annotation_csv_path"], training.ROUND1_ANNOTATIONS_SHA256)
    selected = []
    t2_ids = set(data["new_region_ids"])
    for region in regions:
        if region.region_id in t2_ids:
            kind, labelsha = "T2-explicit-aqueous-negative", training.T2_ANNOTATIONS_SHA256
        elif region.payload.get("gas_variant_annotation_sha256") == training.ROUND1_ANNOTATIONS_SHA256:
            kind, labelsha = "round1-gas-positive", training.ROUND1_ANNOTATIONS_SHA256
        else:
            continue
        if region.run in training.BLIND_RUNS:
            raise ValueError("New blind-run labels may not enter §4b")
        selected.append((region, kind, labelsha))
    if Counter(kind for _, kind, _ in selected) != {"T2-explicit-aqueous-negative": 223, "round1-gas-positive": 871}:
        raise ValueError("§4b training inventory count changed")
    manifests, tasks, absent, manifest_bindings = {}, {}, [], []
    t2_sources = {row["run"]: row for row in t2["source_bindings"]}
    for region, kind, labelsha in selected:
        run = region.run
        if run not in manifests:
            old_path = base_candidate / "source_alias/native_frames" / run / "manifest.json"
            new_path = new_candidate / "source_alias/native_frames" / run / "manifest.json"
            if run in t2_sources:
                training.checked(old_path, t2_sources[run]["native_manifest_sha256"])
            old, new = training.read(old_path), training.read(new_path)
            if old["canvas_wh"] != new["canvas_wh"]:
                raise ValueError("The T2 candidate changed annotation canvas geometry")
            historical = None
            if run in round1_sources:
                binding = round1_sources[run]
                historical = training.read(training.checked(binding["native_manifest_path"], binding["native_manifest_sha256"]))
                if historical["canvas_wh"] != old["canvas_wh"]:
                    raise ValueError("Round-1 annotation canvas needs a verified transform")
            manifests[run] = (old, new, source_index(old), source_index(new), historical)
            manifest_bindings.append(dict(run=run,
                base_native_manifest=dict(path=str(old_path), sha256=training.sha256_file(old_path)),
                new_native_manifest=dict(path=str(new_path), sha256=training.sha256_file(new_path)),
                original_round1_native_manifest=round1_sources.get(run)))
        old, new, old_index, new_index, historical = manifests[run]
        source = old["frame_sources"][region.frame_index] if kind.startswith("T2") else historical["frame_sources"][region.frame_index]
        sha = source["source_sha256"]
        if kind.startswith("T2") and region.payload["source_jpeg_sha256"] != sha:
            raise ValueError("T2 source binding differs from the training seal")
        if sha not in old_index or sha not in new_index:
            absent.append(dict(region_id=region.region_id, run=run, source_sha256=sha,
                original_frame_index=region.frame_index, kind=kind,
                reason="own source is outside current 864-frame timeline; final-map recall not evaluated"))
            continue
        key = (run, sha)
        frame, newframe = old_index[sha], new_index[sha]
        task = tasks.setdefault(key, dict(run=run, frame_index=frame["frame_index"],
            new_frame_index=newframe["frame_index"], elapsed_h=frame["elapsed_h"],
            source_sha256=sha, canvas_wh=old["canvas_wh"], regions=[]))
        task["regions"].append(dict(**asdict(region), kind=kind, annotation_sha256=labelsha))
    expected_absent = {row["region_id"] for row in data["original_ac31_off_timeline"]["labels"]}
    if {row["region_id"] for row in absent} != expected_absent or len(absent) != 34:
        raise ValueError("§4b missing native sources differ from preserved AC31 off-timeline audit")
    output.mkdir(parents=True, exist_ok=True)
    results, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(max(1, int(workers)), len(tasks))) as pool:
        futures = {pool.submit(assess_frame, str(base_candidate), str(new_candidate), task): key
                   for key, task in tasks.items()}
        for future in as_completed(futures):
            value = future.result()
            training.write(output / "frames" / value["run"] / f"frame_{value['frame_index']:03d}.json", value)
            results.append(value)
    rows = sorted((row for value in results for row in value["rows"]), key=lambda row: row["region_id"])
    report = dict(contract=CONTRACT, scope="in-sample full native final maps; not independent test-label validation",
        final_map_qa=True, independent_validation=False, base_candidate=str(base_candidate),
        new_candidate=str(new_candidate), dataset_receipt_sha256=training.sha256_file(dataset_root / "dataset_receipt.json"),
        training_label_sha256=data["combined_label_sha256"], per_region=rows,
        per_run_frame=aggregate(rows), per_run=aggregate_runs(rows),
        source_manifest_bindings=manifest_bindings, t2_label_session_seals=t2,
        round1_label_seals=original["new_labels"],
        evaluated_region_counts=dict(Counter(row["kind"] for row in rows)),
        original_region_counts=dict(Counter(kind for _, kind, _ in selected)), off_timeline_not_evaluated=absent,
        denominator_policy="All rasterized annotation pixels; shared-modelled coverage and rates reported separately. Unknown pixels are never dropped silently.",
        per_box_weighting="Equal-box mean and pixel-weighted fractions both reported; these metrics do not select thresholds.",
        rounds_2_3_4_read=False, new_blind_session_labels_read=False, wall_seconds=time.perf_counter()-started)
    training.write(output / "native_training_map_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-candidate", "new-candidate", "dataset-root", "old-extraction",
                 "experiment", "new-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 24:
        parser.error("CPU QA workers must be within 1--24")
    report_native_training_maps(args.base_candidate, args.new_candidate, args.dataset_root,
                                args.old_extraction, args.output, workers=args.workers)
    training.pixel_sample_in_sample_report(args.experiment, args.dataset_root, args.new_model,
                                          args.output / "sampled_pixel_supplement.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
