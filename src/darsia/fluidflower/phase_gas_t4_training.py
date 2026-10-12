"""Two independent CPU G3 gas MIL fits with sealed L-cap positive boxes.

The original G3 matrices are imported verbatim. Epoch membership is read from
the series project table. No T2/B-T2/B-T3 loader is called and no other review
decisions are opened by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import math
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_gas_t2_training as g3
from .phase_training import logistic
from .phase_training.trainer import RunSource
from .project import sha256_file
from .review_storage import resolve_review_path

CONTRACT = "ffac-g3-t4-light-epoch-pixel-gas-refit-v1"
SPEC_SHA256 = "28040816088dfb4d08d9ce0cf53708f5bfe497adf57f36891b8da9a116e7a39d"
LCAP_ANNOTATIONS_SHA256 = "93645a04957468390592ccec5b9d045b3c83cd4294dc6d1b4528daa9418c7d1d"
LCAP_MANIFEST_SHA256 = "7b059be01d426b67a600c541c5c3ebe1dd055b62c9bf62c6bf085eb220a5728a"
LCAP_SESSION_SHA256 = "45a1bcafed6585fe5d9960d54c98d5beac08000a7ac77be6afbb59a3a088065d"
LCAP_GAS_COUNTS = {"ac42": 41, "ac44": 41, "ac58": 30, "ac60": 44}
FLATFIELD_NAMES, HEAD, CO2_SHA256 = g3.FLATFIELD_NAMES, g3.HEAD, g3.CO2_SHA256
G3_SHA256, G3_LABEL_SHA256 = g3.G3_SHA256, g3.G3_LABEL_SHA256
ROUND1_ANNOTATIONS_SHA256 = g3.ROUND1_ANNOTATIONS_SHA256
read, safe, digest, checked, label_digest = g3.read, g3.safe, g3.digest, g3.checked, g3.label_digest
load_frozen_g3_data = g3.load_frozen_g3_data


def write(path, value, *, progress=False):
    """Commit a new immutable T4 receipt, allowing matching restart evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = safe(value)
    if path.is_file() and not progress:
        if read(path) != value:
            raise ValueError(f"Existing immutable T4 receipt differs: {path}")
        return
    import json
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_light_epochs(path):
    """Load source-declared epoch membership; never infer it from run numbers."""
    path = resolve_review_path(path).resolve()
    table = read(path)
    if (table.get("contract") != "ffac-light-epoch-project-table-v1"
            or table.get("series") != "FF_AC"
            or table.get("epochs") != ["rig_light", "new_lamps"]
            or table.get("provenance", {}).get("spec_sha256") != SPEC_SHA256):
        raise ValueError("T4 requires the authorized FF_AC project light-epoch table")
    mapping = {}
    for row in table["runs"]:
        run, epoch = row["run"].casefold(), row["epoch"]
        if run in mapping or epoch not in table["epochs"] or not run.startswith("ac"):
            raise ValueError("Duplicate run or undeclared epoch in project table")
        mapping[run] = epoch
    if not mapping or set(mapping.values()) != set(table["epochs"]):
        raise ValueError("Project epoch table lacks one of the two light epochs")
    return dict(epochs=list(table["epochs"]), run_to_epoch=mapping,
        receipt=dict(contract=table["contract"], path=str(path), sha256=sha256_file(path),
            spec_sha256=SPEC_SHA256, run_to_epoch=mapping, epochs=list(table["epochs"]),
            provenance=table["provenance"]))


def epoch_for_run(light_epochs, run):
    try:
        return light_epochs["run_to_epoch"][str(run).casefold()]
    except KeyError as exc:
        raise ValueError(f"Run has no declared light epoch: {run}") from exc


def load_lcap_source_bundle(candidate, snapshot):
    """Read only the promoted 156 gas boxes and their sealed source metadata."""
    candidate, snapshot = resolve_review_path(candidate).resolve(), Path(snapshot).resolve()
    manifest_path = checked(snapshot / "MANIFEST.json", LCAP_MANIFEST_SHA256)
    manifest = read(manifest_path)
    annotations = checked(snapshot / "lcap_annotations.csv", LCAP_ANNOTATIONS_SHA256)
    session_path = checked(snapshot / "session_copy/review_session.json", LCAP_SESSION_SHA256)
    session = read(session_path)
    if (manifest["lcap_annotations_sha256"] != LCAP_ANNOTATIONS_SHA256
            or session["target_model_id"] != candidate.name):
        raise ValueError("L-cap labels belong to another declared base candidate")
    validation_path = resolve_review_path(session["combined_root"]) / "validation_binding.json"
    validation = read(validation_path)
    if validation["model_id"] != candidate.name or resolve_review_path(validation["model_root"]).resolve() != candidate:
        raise ValueError("L-cap review preparation belongs to another candidate")
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if Counter(r["category"] for r in rows) != {"confirmed_only_free_gas": 156, "confirmed_only_aqueous_co2": 137}:
        raise ValueError("L-cap sealed class inventory changed")
    positives = [r for r in rows if r["category"] == "confirmed_only_free_gas"]
    if Counter(r["run"] for r in positives) != LCAP_GAS_COUNTS or len({r["annotation_id"] for r in rows}) != 293:
        raise ValueError("L-cap sealed run/unique-region inventory changed")
    groups, regions, sources = [], [], []
    for run in sorted(LCAP_GAS_COUNTS):
        native_path = candidate / "source_alias/native_frames" / run / "manifest.json"
        native = read(native_path)
        native_sha = sha256_file(native_path)
        marker = native.get("ffac_rig_camera_sources", {})
        if (native["run"] != run or native["dataset"] != "ffac"
                or marker.get("contract") != "ffac-rig-camera-native-sources-v1"
                or marker.get("inference_authorized") is not False):
            raise ValueError("L-cap requires exact base rig/camera native source metadata")
        source = RunSource(run=run, dataset="ffac", review_root=candidate / "source_alias",
            asset_manifest_path=native_path,
            decisions_path=snapshot / "session_copy/pairwise_review_decisions.json",
            combined_manifest_path=validation_path)
        local = []
        for row in positives:
            if row["run"] != run:
                continue
            index, elapsed = int(row["frame_index"]), float(row["elapsed_h"])
            if not 0 <= index < len(native["frame_sources"]):
                raise ValueError("L-cap box lost its original camera-clock frame")
            frame = native["frame_sources"][index]
            if (row["candidate_id"] != session["selected_candidates"][run]
                    or row["candidate_id"] != validation["selected_candidates"][run]
                    or row["params_sha256"] != validation["pipeline_sha256"]
                    or not math.isclose(elapsed, float(frame["elapsed_h"]), rel_tol=0, abs_tol=1e-8)
                    or row["coordinate_space"] != "normalized_native_review_canvas_16x9"
                    or row["shape"] != "rectangle" or row["contours_visible"].casefold() != "false"):
                raise ValueError(f"L-cap box lost its sealed source/geometry: {row['annotation_id']}")
            payload = dict(row)
            for name in ("x", "y", "width", "height"):
                value = float(row[name])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Invalid normalized L-cap coordinate")
                payload[name] = value
            payload.update(id=row["annotation_id"], points=[], frame_index=index, elapsed_h=elapsed,
                phase_class="gas", training_phase_class="gas", training_scope="whole-marked-region",
                training_eligible=True, lcap_annotation_sha256=LCAP_ANNOTATIONS_SHA256,
                source_native_manifest_sha256=native_sha, source_jpeg_sha256=frame["source_sha256"],
                training_revision="G3-T4", source_session_id=session["review_session_id"])
            region = logistic.ReviewedRegion(region_id=f"{run}:manual:{row['annotation_id']}",
                run=run, dataset="ffac", frame_index=index, elapsed_h=elapsed, phase_class="gas",
                confidence=row["confidence"].casefold(), source="manual", reviewer=row["reviewer"], payload=payload)
            if region.supervision_weight <= 0:
                raise ValueError("Promoted L-cap gas box has zero supervision weight")
            local.append(region)
        groups.append((source, tuple(local)))
        regions.extend(local)
        sources.append(dict(run=run, native_manifest_path=str(native_path), native_manifest_sha256=native_sha,
            frame_indices=sorted({r.frame_index for r in local}), regions=len(local)))
    logistic.assert_safe_training_regions(regions)
    receipt = dict(contract=CONTRACT, snapshot_manifest_path=str(manifest_path),
        snapshot_manifest_sha256=LCAP_MANIFEST_SHA256, annotation_csv_path=str(annotations),
        annotation_csv_sha256=LCAP_ANNOTATIONS_SHA256, session_path=str(session_path),
        session_sha256=LCAP_SESSION_SHA256, session_id=session["review_session_id"],
        validation_binding_path=str(validation_path), validation_binding_sha256=sha256_file(validation_path),
        source_bindings=sources, box_count=156, excluded_aqueous_boxes=137,
        excluded_region_ids=[r["annotation_id"] for r in rows if r["category"] == "confirmed_only_aqueous_co2"],
        region_ids_sha256=digest([r.region_id for r in regions]),
        geometry="normalized-own-native-review-canvas-16x9; clip-to-rig-and-observed-feature-support",
        authorization_spec_sha256=SPEC_SHA256, shared_session_other_decisions_read=False,
        t2_bt2_bt3_labels_read=False, rounds_2_3_4_read=False, sealed_scorer_read=False)
    return dict(groups=groups, regions=regions, new_region_ids={r.region_id for r in regions}, receipt=receipt)


def _source_name(region, round1_ids, lcap_ids):
    return "lcap_positive" if region.region_id in lcap_ids else "round1" if region.region_id in round1_ids else "old_G3"


def training_inventory(regions, light_epochs, *, round1_ids=(), lcap_ids=(), owners=None):
    """Report distinct supported regions, raw box weights, and run counts."""
    sample_counts = None if owners is None else np.bincount(owners, minlength=len(regions))
    report = {epoch: dict(by_source={}, by_run={}, positive=dict(regions=0, direct_regions=0, weak_regions=0, raw_weight=0., runs=[]),
        negative=dict(regions=0, direct_regions=0, weak_regions=0, raw_weight=0., runs=[]), unused_for_gas=dict(regions=0, raw_weight=0.))
        for epoch in light_epochs["epochs"]}
    for i, region in enumerate(regions):
        epoch = epoch_for_run(light_epochs, region.run)
        target = logistic.regional_presence_target(region.phase_class, HEAD)
        active = region.supervision_weight > 0 and (sample_counts is None or sample_counts[i] > 0)
        kind = "positive" if target == 1 and active else "negative" if target == 0 and active else "unused_for_gas"
        weight = region.supervision_weight
        entry = report[epoch][kind]
        entry["regions"] += 1
        entry["raw_weight"] += weight
        if kind != "unused_for_gas":
            entry["direct_regions" if logistic.pixel_target(region.phase_class, HEAD) is not None else "weak_regions"] += 1
            entry["runs"].append(region.run)
        source = _source_name(region, set(round1_ids), set(lcap_ids))
        for column, key in (("by_source", source), ("by_run", region.run)):
            row = report[epoch][column].setdefault(key, {})
            cell = row.setdefault(region.phase_class, dict(regions=0, active_regions=0, raw_weight=0., sample_rows=0))
            cell["regions"] += 1
            cell["active_regions"] += int(active)
            cell["raw_weight"] += weight
            cell["sample_rows"] += 0 if sample_counts is None else int(sample_counts[i])
    for epoch in report.values():
        for kind in ("positive", "negative"):
            epoch[kind]["runs"] = sorted(set(epoch[kind]["runs"]))
            epoch[kind]["run_count"] = len(epoch[kind]["runs"])
    return report


def minimum_data_gate(inventory, *, raise_on_failure=True):
    new = inventory["new_lamps"]
    passed = new["positive"]["regions"] >= 100 and new["negative"]["regions"] >= 100
    gate = dict(passed=passed, epoch="new_lamps", required_positive_regions=100, required_negative_regions=100,
        positive_regions=new["positive"]["regions"], negative_regions=new["negative"]["regions"],
        direct_positive_regions=new["positive"]["direct_regions"],
        weak_positive_regions=new["positive"]["weak_regions"], positive_runs=new["positive"]["runs"], negative_runs=new["negative"]["runs"])
    if not passed and raise_on_failure:
        raise ValueError(f"T4 minimum-data gate failed before fitting: {gate}")
    return gate


def load_combined_training_sources(old_candidate, round1_snapshot, base_candidate, lcap_snapshot, light_epochs):
    """Return original G3 source groups plus promoted L-cap groups only."""
    from .phase_gas_variant_sources import load_training_sources
    epochs = load_light_epochs(light_epochs) if not isinstance(light_epochs, dict) else light_epochs
    old = load_training_sources(Path(old_candidate), Path(round1_snapshot))
    old_regions = [r for _source, group in old["groups"] for r in group]
    if label_digest(old_regions) != G3_LABEL_SHA256:
        raise ValueError("Reconstructed sources differ from exact original G3 labels")
    new = load_lcap_source_bundle(base_candidate, lcap_snapshot)
    regions = [*old_regions, *new["regions"]]
    if len(regions) != 3005 or len({r.region_id for r in regions}) != len(regions):
        raise ValueError("T4 inventory is not exactly 2849 G3 plus156 unique L-cap gas boxes")
    inventory = training_inventory(regions, epochs, round1_ids=old["new_region_ids"], lcap_ids=new["new_region_ids"])
    gate = minimum_data_gate(inventory)
    receipt = dict(contract=CONTRACT, old_g3_label_sha256=G3_LABEL_SHA256, old_source_bundle=old["receipt"],
        lcap=new["receipt"], combined_label_sha256=label_digest(regions),
        combined_region_ids_sha256=digest([r.region_id for r in regions]), light_epochs=epochs["receipt"],
        training_counts=inventory, minimum_data_gate=gate,
        t2_bt2_bt3_labels_read=False, rounds_2_3_4_read=False, sealed_scorer_read=False)
    return dict(groups=[*old["groups"], *new["groups"]], regions=regions, old_regions=old_regions,
        new_regions=new["regions"], new_region_ids=new["new_region_ids"], light_epochs=epochs, receipt=receipt)


def preflight(experiment, extraction, base_candidate, lcap_snapshot, light_epochs, output, *, spec):
    """SHA-prove label/source inventory and gate before feature extraction/fit."""
    checked(spec, SPEC_SHA256)
    checked(resolve_review_path(base_candidate) / "models/generic/co2_presence_signal.json", CO2_SHA256)
    epochs = load_light_epochs(light_epochs)
    frozen = load_frozen_g3_data(experiment, extraction)
    new = load_lcap_source_bundle(base_candidate, lcap_snapshot)
    regions = [*frozen["regions"], *new["regions"]]
    if len(regions) != 3005 or len({r.region_id for r in regions}) != len(regions):
        raise ValueError("Unexpected/duplicate T4 label inventory")
    inventory = training_inventory(regions, epochs, round1_ids=frozen["receipt"]["new_region_ids"], lcap_ids=new["new_region_ids"])
    result = dict(contract=CONTRACT, spec_sha256=SPEC_SHA256, light_epochs=epochs["receipt"],
        original_g3_label_sha256=G3_LABEL_SHA256, original_g3_regions=2849, original_round1_regions=871,
        g3_head_sha256=G3_SHA256, g3_extraction_receipt_path=str(resolve_review_path(extraction) / "extraction_receipt.json"),
        g3_extraction_receipt_sha256=sha256_file(resolve_review_path(extraction) / "extraction_receipt.json"),
        original_ac31_off_timeline=g3.ac31_off_timeline_audit(frozen, base_candidate),
        training_label_sources=frozen["label_sources"], lcap=new["receipt"],
        combined_label_sha256=label_digest(regions), combined_region_count=len(regions), training_counts=inventory,
        minimum_data_gate=minimum_data_gate(inventory, raise_on_failure=False),
        t2_bt2_bt3_labels_read=False, rounds_2_3_4_read=False, sealed_scorer_read=False, fit_started=False)
    write(Path(output) / "preflight_receipt.json", result)
    minimum_data_gate(inventory)
    return result


def _extract_lcap_run(candidate, run, serialized_regions, destination, pixels_per_region):
    """Reuse the validated source-bound scientific extraction, with T4 labels."""
    destination = Path(destination)
    old = g3._extract_t2_run(candidate, run, serialized_regions, destination / "scientific_samples", pixels_per_region)
    receipt = dict(**{k: v for k, v in old.items() if k != "contract"}, contract=CONTRACT,
        source_extractor_path=str(Path(g3.__file__)), source_extractor_sha256=sha256_file(Path(g3.__file__)),
        source_extractor_input="only-sealed-Lcap-gas-positive-regions", t2_labels_loaded=False)
    write(destination / "receipt.json", receipt)
    return receipt


def _epoch_view(features, owners, regions, light_epochs, epoch):
    """Physically isolate matrices and remap owners before any normalization."""
    indices = np.array([i for i, r in enumerate(regions) if epoch_for_run(light_epochs, r.run) == epoch], np.int32)
    mapping = np.full(len(regions), -1, np.int32)
    mapping[indices] = np.arange(len(indices), dtype=np.int32)
    selection = mapping[owners] >= 0
    return features[selection].copy(), mapping[owners[selection]], [regions[int(i)] for i in indices]


def _weight_report(features, owners, regions, round1_ids, lcap_ids):
    _matrix, targets, weights, selected = logistic.balanced_pixel_view(features, owners, regions, head=HEAD)
    report = defaultdict(dict)
    for source in ("old_G3", "round1", "lcap_positive"):
        for target, name in ((0, "negative"), (1, "positive")):
            mask = (targets == target) & np.array([_source_name(regions[int(i)], round1_ids, lcap_ids) == source for i in selected])
            report[source][name] = dict(direct_regions=len(np.unique(selected[mask])), sample_rows=int(mask.sum()),
                balanced_direct_pixel_loss_weight=float(weights[mask].sum()))
    bags = logistic._weak_region_bags(owners, regions, head=HEAD, runs=None)
    for source in report:
        report[source]["weak_bag_loss_weight"] = sum(weight for _rows, _target, weight, owner in bags
            if _source_name(regions[int(owner)], round1_ids, lcap_ids) == source)
    return dict(report)


def prepare_dataset(experiment, extraction, base_candidate, lcap_snapshot, light_epochs, output, *, spec, workers=1):
    """Append L-cap native samples to bit-identical original G3 rows."""
    if not 1 <= int(workers) <= 4:
        raise ValueError("T4 extraction requires one to four CPU workers")
    output = Path(output)
    proof = preflight(experiment, extraction, base_candidate, lcap_snapshot, light_epochs, output, spec=spec)
    frozen = load_frozen_g3_data(experiment, extraction)
    new = load_lcap_source_bundle(base_candidate, lcap_snapshot)
    epochs = load_light_epochs(light_epochs)
    regions = [*frozen["regions"], *new["regions"]]
    reports, pending = {}, []
    for source, group in new["groups"]:
        destination = output / "lcap_extraction" / source.run
        if (destination / "receipt.json").is_file():
            receipt = read(destination / "receipt.json")
            checked(receipt["sample_path"], receipt["samples_sha256"])
            if receipt["contract"] != CONTRACT or [r["region_id"] for r in receipt["coverage"]] != [r.region_id for r in sorted(group, key=lambda r: r.frame_index)]:
                raise ValueError("Resumed L-cap extraction changed its source/region order")
            reports[source.run] = receipt
        else:
            pending.append((source.run, [asdict(r) for r in group], str(destination)))
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(pending)), mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_extract_lcap_run, str(base_candidate), run, rows, destination,
                frozen["receipt"]["sampling"]["pixels_per_region"]) for run, rows, destination in pending]
            for future in as_completed(futures):
                report = future.result()
                reports[report["run"]] = report
                write(output / "progress.json", dict(stage="lcap_extracting", completed=len(reports), total=4), progress=True)
    matrices, owners = [frozen["features"]], [frozen["owners"]]
    offset = len(frozen["regions"])
    for source, group in new["groups"]:
        with np.load(reports[source.run]["sample_path"], allow_pickle=False) as archive:
            matrices.append(archive["features"].copy())
            owners.append(archive["region_indices"].copy() + offset)
        offset += len(group)
    features, ownership = np.concatenate(matrices), np.concatenate(owners)
    if not np.array_equal(features[:len(frozen["features"])], frozen["features"]) or not np.array_equal(ownership[:len(frozen["owners"])], frozen["owners"]):
        raise ValueError("Original G3 features or owners changed during T4 append")
    output.mkdir(parents=True, exist_ok=True)
    for name, array in (("flatfield.npy", features), ("owners.npy", ownership)):
        path = output / name
        if path.is_file():
            if not np.array_equal(np.load(path, allow_pickle=False), array):
                raise ValueError("Resumed T4 merged dataset differs")
        else:
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
    region_path = output / "regions.json"
    write(region_path, [asdict(r) for r in regions])
    counts = training_inventory(regions, epochs, round1_ids=frozen["receipt"]["new_region_ids"], lcap_ids=new["new_region_ids"], owners=ownership)
    weights = {}
    for epoch in epochs["epochs"]:
        x, owner, local = _epoch_view(features, ownership, regions, epochs, epoch)
        weights[epoch] = _weight_report(x, owner, local, set(frozen["receipt"]["new_region_ids"]), new["new_region_ids"])
    receipt = dict(**{k: v for k, v in proof.items() if k not in ("training_counts", "minimum_data_gate")},
        training_counts=counts, minimum_data_gate=minimum_data_gate(counts), weights=weights,
        region_inventory_path=str(region_path), region_inventory_sha256=sha256_file(region_path),
        feature_path=str(output / "flatfield.npy"), feature_sha256=sha256_file(output / "flatfield.npy"),
        owner_path=str(output / "owners.npy"), owner_sha256=sha256_file(output / "owners.npy"),
        new_region_ids=sorted(new["new_region_ids"]), original_round1_region_ids=frozen["receipt"]["new_region_ids"],
        old_features_and_owner_indices_bit_identical=True, old_sample_rows=len(frozen["features"]),
        lcap_sample_rows=len(features)-len(frozen["features"]), source_feature_reports=[reports[r] for r in sorted(reports)],
        extraction_seconds=time.perf_counter()-started, training_input_policy="exact-original-G3-plus156-Lcap-positive-boxes")
    write(output / "dataset_receipt.json", receipt)
    return receipt


def fit_pixel_g3_t4(experiment, dataset_root, output, *, threads=4, epoch=None):
    """Fit two isolated CPU MIL heads at the fixed G3 settings."""
    from threadpoolctl import threadpool_limits
    if not 1 <= int(threads) <= 32:
        raise ValueError("T4 needs a bounded CPU thread allocation")
    dataset_root, output = Path(dataset_root), Path(output)
    receipt_path = dataset_root / "dataset_receipt.json"
    receipt = read(receipt_path)
    if receipt.get("contract") != CONTRACT:
        raise ValueError("T4 will not fit any T2/T3 or other dataset")
    if (receipt.get("original_g3_regions") != 2849 or receipt.get("original_round1_regions") != 871
            or receipt.get("old_features_and_owner_indices_bit_identical") is not True
            or receipt.get("t2_bt2_bt3_labels_read") is not False or receipt.get("rounds_2_3_4_read") is not False
            or receipt.get("lcap", {}).get("annotation_csv_sha256") != LCAP_ANNOTATIONS_SHA256
            or receipt.get("lcap", {}).get("box_count") != 156):
        raise ValueError("T4 dataset is not the exact authorized original G3 plus L-cap set")
    for source in receipt["training_label_sources"]:
        checked(source["path"], source["sha256"])
        if source.get("session_sha256"):
            checked(Path(source["path"]).parent / "review_session.json", source["session_sha256"])
    for key in ("snapshot_manifest", "annotation_csv", "session"):
        checked(receipt["lcap"][f"{key}_path"], receipt["lcap"][f"{key}_sha256"])
    epochs = load_light_epochs(receipt["light_epochs"]["path"])
    if epoch is not None and epoch not in epochs["epochs"]:
        raise ValueError("Requested T4 fit epoch is absent from the project table")
    if epochs["receipt"]["sha256"] != receipt["light_epochs"]["sha256"]:
        raise ValueError("Project epoch table changed after extraction")
    for name in ("feature", "owner", "region_inventory"):
        checked(receipt[f"{name}_path"], receipt[f"{name}_sha256"])
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    regions = [logistic.ReviewedRegion(**r) for r in read(receipt["region_inventory_path"])]
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 4
            or owners.dtype != np.int32 or owners.shape != features.shape[:1] or not np.isfinite(features).all()
            or not owners.size or owners.min() < 0 or owners.max() >= len(regions)
            or label_digest(regions) != receipt["combined_label_sha256"] or len(regions) != 3005):
        raise ValueError("T4 fit lost its exact finite scientific samples or labels")
    if (label_digest(regions[:2849]) != G3_LABEL_SHA256
            or {r.region_id for r in regions[2849:]} != set(receipt["new_region_ids"])
            or any(r.phase_class != "gas" or r.payload.get("lcap_annotation_sha256") != LCAP_ANNOTATIONS_SHA256
                   for r in regions[2849:])):
        raise ValueError("T4 fit inventory lost its exact G3 prefix or L-cap-only gas suffix")
    counts = training_inventory(regions, epochs, owners=owners)
    gate = minimum_data_gate(counts)
    old_head = read(checked(resolve_review_path(experiment) / "variants/G3/gas_head.json", G3_SHA256))
    settings = dict(old_head["training_provenance"]["fit_settings"])
    if settings["backend"] != "cpu":
        raise ValueError("T4 MIL is CPU only")
    cv2.setNumThreads(1)
    fitted = {}
    selected_epochs = epochs["epochs"] if epoch is None else [epoch]
    for epoch_id in selected_epochs:
        local_output = output / epoch_id
        if (local_output / "fit_receipt.json").is_file():
            result = read(local_output / "fit_receipt.json")
            checked(result["model_path"], result["model_sha256"])
            if (result["dataset_receipt_sha256"] != sha256_file(receipt_path)
                    or result["light_epoch_id"] != epoch_id or not result["converged"]):
                raise ValueError("Existing T4 epoch fit belongs to another dataset")
            fitted[epoch_id] = result
            continue
        x, owner, local = _epoch_view(features, owners, regions, epochs, epoch_id)
        logistic.assert_safe_training_regions(local)
        initial = dict(contract=CONTRACT, training_revision="G3-T4", light_epoch_id=epoch_id,
            training_runs=sorted({r.run for r in local}), training_label_sha256=label_digest(local),
            epoch_label_sha256=label_digest(local),
            dataset_receipt_sha256=sha256_file(receipt_path), light_epochs=epochs["receipt"],
            parent_head_sha256=G3_SHA256, fit_settings=settings, cpu_only=True, threads=int(threads),
            own_epoch_sample_rows=len(x), minimum_data_gate=gate, normalization_isolated_by_epoch=True,
            code_sha256=dict(trainer=sha256_file(Path(__file__)), logistic=sha256_file(Path(logistic.__file__))))
        write(local_output / "initial_binding.json", initial)
        write(local_output / "progress.json", dict(stage="pixel_MIL_fit_running", epoch=epoch_id, started_utc=datetime.now(timezone.utc).isoformat()), progress=True)
        started, cpu_started = time.perf_counter(), time.process_time()
        with threadpool_limits(limits=int(threads)):
            fit, diagnostics = logistic.fit_weighted_logistic_mil(x, owner, local, head=HEAD, runs=None, **settings)
        wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
        model = dict(format=old_head["format"], head=HEAD, variant="G3", training_revision="G3-T4",
            light_epoch_id=epoch_id, feature_contract=old_head["feature_contract"], feature_names=list(FLATFIELD_NAMES),
            fit=safe(asdict(fit)), gas_display_threshold=.6,
            training_provenance=dict(**initial, combined_label_sha256=label_digest(local),
                unchanged_co2_head_sha256=CO2_SHA256, pixel_threshold=.7, regional_threshold=.6,
                gas_minimum_mm2=2.5, connectivity=4, t2_bt2_bt3_labels_read=False,
                rounds_2_3_4_read=False, sealed_scorer_read=False, mass_eligible=False))
        write(local_output / "gas_head.json", model)
        result = dict(**initial, converged=bool(fit.converged), fit_wall_seconds=wall, fit_cpu_seconds=cpu,
            old_fit=old_head["fit"], new_fit=safe(asdict(fit)), diagnostics=diagnostics,
            model_path=str(local_output / "gas_head.json"), model_sha256=sha256_file(local_output / "gas_head.json"),
            fit_receipt_path=str(local_output / "fit_receipt.json"), weights=receipt["weights"][epoch_id],
            training_counts=receipt["training_counts"][epoch_id], training_region_ids=[r.region_id for r in local])
        write(local_output / "training_report.json", result)
        write(local_output / "fit_receipt.json", result)
        if not fit.converged:
            raise ValueError("T4 gas MIL did not converge; downstream blocked")
        fitted[epoch_id] = result
        del x, owner
    result = dict(contract=CONTRACT, epochs=fitted, converged=all(v["converged"] for v in fitted.values()), light_epochs=epochs["receipt"], cpu_only=True,
        dataset_receipt_path=str(receipt_path), dataset_receipt_sha256=sha256_file(receipt_path),
        combined_label_sha256=receipt["combined_label_sha256"], minimum_data_gate=gate,
        common_frozen_co2_head_sha256=CO2_SHA256, t2_bt2_bt3_labels_read=False, rounds_2_3_4_read=False)
    write(output / ("fit_receipt.json" if epoch is None else f"fit_receipt_{epoch}.json"), result)
    if epoch is None:
        write(output / "progress.json", dict(stage="two_epoch_pixel_fit_complete", epochs=list(fitted)), progress=True)
    return result
