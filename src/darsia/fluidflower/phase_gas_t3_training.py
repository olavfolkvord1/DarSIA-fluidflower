"""Append the spec-authorized sealed B-T2 boxes to the exact frozen T2 set.

The gas-only CPU MIL fit keeps G3's feature expression and fit settings.
Historical samples are imported verbatim, including the original AC31 source
images. Only the promoted B-T2 CSV and its sealed session metadata are opened;
live sessions and the later regression/blind label files are never consulted.
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

from . import phase_gas_t2_training as t2
from .phase_training import logistic
from .phase_training.trainer import RunSource
from .project import sha256_file
from .review_storage import resolve_review_path

CONTRACT = "ffac-g3-t3-pixel-gas-refit-v1"
SPEC_SHA256 = "f2ca85416cb0069b8c50207912ab0434b4cef9397cda1fb196dfa3d3db7d874e"
BT2_MANIFEST_SHA256 = "f1b5020599ef5da2419c0237023b0b55ebd2f43e4bc64b9d8e97386812f92c76"
BT2_ANNOTATIONS_SHA256 = "4b08f5e1a75b01dab349dfd2f9826d3fe4791ad7bc6e39cae24b2a962de22680"
BT2_SUMS_SHA256 = "4a39a8ca2d32aacbcf4b8c8dec471278818cc5e920a140ba5ae4bc045244d419"
BT2_SESSION_SHA256 = "2988273a8d68e52c81b7ad94efc14c8457fb4960edf45b9199e8466bcf488f07"
T2_DATASET_SHA256 = "9380f073512d0668e88c12895fd2101da84a3b712ada14bc6004b8a69c4e6840"
BT2_RUNS = frozenset(("ac30", "ac49", "ac54"))
HIDDEN_RUNS = frozenset(("ac26", "ac55"))
FLATFIELD_NAMES, HEAD, CO2_SHA256 = t2.FLATFIELD_NAMES, t2.HEAD, t2.CO2_SHA256
G3_SHA256, G3_LABEL_SHA256 = t2.G3_SHA256, t2.G3_LABEL_SHA256
ROUND1_ANNOTATIONS_SHA256 = t2.ROUND1_ANNOTATIONS_SHA256
read, safe, digest, write, checked, label_digest = (
    t2.read, t2.safe, t2.digest, t2.write, t2.checked, t2.label_digest)


def _check_bt2_row(row, session, validation, native):
    run = row["run"]
    index, elapsed = int(row["frame_index"]), float(row["elapsed_h"])
    if run not in BT2_RUNS or index not in (5, 8, 11, 13):
        raise ValueError("B-T2 training box is outside the sealed cohort/frames")
    frame = native["frame_sources"][index]
    if (row["candidate_id"] != session["selected_candidates"][run]
            or row["candidate_id"] != validation["selected_candidates"][run]
            or row["params_sha256"] != validation["pipeline_sha256"]
            or not math.isclose(elapsed, float(frame["elapsed_h"]), rel_tol=0, abs_tol=1e-8)
            or row["coordinate_space"] != "normalized_native_review_canvas_16x9"
            or row["shape"] != "rectangle" or row["contours_visible"].casefold() != "false"):
        raise ValueError(f"B-T2 box lost its sealed source/geometry: {row['annotation_id']}")
    for name in ("x", "y", "width", "height"):
        value = float(row[name])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid normalized B-T2 box coordinate")
    return index, elapsed, frame


def load_bt2_source_bundle(candidate, snapshot):
    """Promote only the spec's 321 SHA-whitelisted scored B-T2 boxes."""
    candidate, snapshot = resolve_review_path(candidate).resolve(), Path(snapshot).resolve()
    manifest_path = checked(snapshot / "MANIFEST.json", BT2_MANIFEST_SHA256)
    manifest = read(manifest_path)
    annotations = checked(snapshot / "blind_annotations_ac30_ac49_ac54.csv", BT2_ANNOTATIONS_SHA256)
    sums_path = checked(snapshot / "SHA256SUMS.txt", BT2_SUMS_SHA256)
    sums = {}
    for line in sums_path.read_text(encoding="utf-8-sig").splitlines():
        value, name = line.split(None, 1)
        sums[name.strip().removeprefix("./")] = value
    if (manifest["contract"] != "claude-ffac-blind-bt2-label-snapshot-v1"
            or manifest["blind_annotations_sha256"] != BT2_ANNOTATIONS_SHA256
            or manifest["sha256sums_file_sha256"] != BT2_SUMS_SHA256
            or sums.get("session_copy/review_session.json") != BT2_SESSION_SHA256):
        raise ValueError("B-T2 snapshot seal differs from the T3 specification")
    session_path = checked(snapshot / "session_copy/review_session.json", BT2_SESSION_SHA256)
    session = read(session_path)
    if session["target_model_id"] != candidate.name:
        raise ValueError("B-T2 labels belong to another base candidate")
    validation_path = resolve_review_path(session["combined_root"]) / "validation_binding.json"
    validation = read(validation_path)
    if (validation["model_id"] != candidate.name
            or resolve_review_path(validation["model_root"]).resolve() != candidate):
        raise ValueError("B-T2 review preparation belongs to another candidate")
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    categories = {"confirmed_only_free_gas": 140, "confirmed_only_aqueous_co2": 181}
    if (Counter(r["category"] for r in rows) != categories
            or Counter(r["run"] for r in rows) != {"ac30": 103, "ac49": 106, "ac54": 112}
            or len({r["annotation_id"] for r in rows}) != 321):
        raise ValueError("B-T2 class/run/unique-box inventory differs from the sealed declaration")
    groups, regions, sources = [], [], []
    for run in sorted(BT2_RUNS):
        native_path = candidate / "source_alias/native_frames" / run / "manifest.json"
        native = read(native_path)
        native_sha = sha256_file(native_path)
        marker = native.get("ffac_rig_camera_sources", {})
        if (native["run"] != run or native["dataset"] != "ffac"
                or marker.get("contract") != "ffac-rig-camera-native-sources-v1"
                or marker.get("inference_authorized") is not False):
            raise ValueError("B-T2 requires the base's exact separate rig/camera source contract")
        source = RunSource(run=run, dataset="ffac", review_root=candidate / "source_alias",
            asset_manifest_path=native_path,
            decisions_path=snapshot / "session_copy/pairwise_review_decisions.json",
            combined_manifest_path=validation_path)
        local = []
        for row in rows:
            if row["run"] != run:
                continue
            index, elapsed, frame = _check_bt2_row(row, session, validation, native)
            phase = "gas" if row["category"] == "confirmed_only_free_gas" else "aqueous"
            payload = dict(row)
            payload.update({name: float(row[name]) for name in ("x", "y", "width", "height")})
            payload.update(id=row["annotation_id"], points=[], frame_index=index, elapsed_h=elapsed,
                phase_class=phase, training_phase_class=phase, training_scope="whole-marked-region",
                training_eligible=True, bt2_annotation_sha256=BT2_ANNOTATIONS_SHA256,
                source_native_manifest_sha256=native_sha,
                source_jpeg_sha256=frame["source_sha256"], source_session_id=session["review_session_id"],
                training_revision="G3-T3", training_authorization_spec_sha256=SPEC_SHA256,
                training_box_clip="rig-domain-and-observed-feature-support")
            region = logistic.ReviewedRegion(region_id=f"{run}:manual:{row['annotation_id']}",
                run=run, dataset="ffac", frame_index=index, elapsed_h=elapsed, phase_class=phase,
                confidence=row["confidence"].casefold(), source="manual", reviewer=row["reviewer"],
                payload=payload)
            if region.supervision_weight <= 0:
                raise ValueError("A sealed B-T2 explicit box has zero supervision weight")
            local.append(region)
        groups.append((source, tuple(local)))
        regions.extend(local)
        sources.append(dict(run=run, native_manifest_path=str(native_path),
            native_manifest_sha256=native_sha,
            frame_indices=sorted({r.frame_index for r in local}), regions=len(local)))
    logistic.assert_safe_training_regions(regions)
    receipt = dict(contract=CONTRACT, snapshot_manifest_path=str(manifest_path),
        snapshot_manifest_sha256=BT2_MANIFEST_SHA256, annotation_csv_path=str(annotations),
        annotation_csv_sha256=BT2_ANNOTATIONS_SHA256, session_path=str(session_path),
        session_sha256=BT2_SESSION_SHA256, session_id=session["review_session_id"],
        validation_binding_path=str(validation_path), validation_binding_sha256=sha256_file(validation_path),
        source_bindings=sources, box_count=321, class_counts=categories,
        region_ids_sha256=digest([r.region_id for r in regions]),
        training_authorization_spec_sha256=SPEC_SHA256,
        promotion_reason="B-T2 already scored; Olav's T3 specification authorizes training use",
        boxes_clipped_to_rig_domain=True, live_session_read=False,
        rounds_2_3_4_read=False, new_blind_session_labels_read=False)
    return dict(groups=groups, regions=regions, new_region_ids={r.region_id for r in regions}, receipt=receipt)


def load_frozen_t2_dataset(dataset_root):
    """Import the complete exact successful T2 dataset, without re-extraction."""
    root = Path(dataset_root)
    receipt_path = checked(root / "dataset_receipt.json", T2_DATASET_SHA256)
    receipt = read(receipt_path)
    for path_key, sha_key in (("feature_path", "feature_sha256"), ("owner_path", "owner_sha256"),
                              ("region_inventory_path", "region_inventory_sha256")):
        checked(receipt[path_key], receipt[sha_key])
    if (receipt["contract"] != t2.CONTRACT or receipt["combined_regions"] != 3072
            or receipt["old_regions"] != 2849 or receipt["t2_regions"] != 223
            or receipt["t2"]["annotation_csv_sha256"] != t2.T2_ANNOTATIONS_SHA256
            or receipt["original_round1_region_count"] != 871
            or receipt["old_features_and_owner_indices_bit_identical"] is not True
            or receipt["rounds_2_3_4_read"] is not False
            or receipt["new_blind_session_labels_read"] is not False):
        raise ValueError("Frozen T2 input is not the exact authorized complete training set")
    for source in receipt["training_label_sources"]:
        checked(source["path"], source["sha256"])
        if source.get("session_sha256"):
            checked(Path(source["path"]).parent / "review_session.json", source["session_sha256"])
    for path_key, sha_key in (("snapshot_manifest_path", "snapshot_manifest_sha256"),
                              ("annotation_csv_path", "annotation_csv_sha256"), ("session_path", "session_sha256")):
        checked(receipt["t2"][path_key], receipt["t2"][sha_key])
    regions = [logistic.ReviewedRegion(**row) for row in read(receipt["region_inventory_path"])]
    if (label_digest(regions) != receipt["combined_label_sha256"]
            or label_digest(regions[:2849]) != G3_LABEL_SHA256
            or {r.region_id for r in regions[2849:]} != set(receipt["new_region_ids"])):
        raise ValueError("Imported T2 label inventory/order changed")
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    _check_arrays(features, owners, regions)
    return dict(features=features, owners=owners, regions=regions, receipt=receipt,
                receipt_path=receipt_path, root=root)


def _check_arrays(features, owners, regions):
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 4
            or owners.dtype != np.int32 or owners.shape != features.shape[:1]
            or not np.isfinite(features).all() or owners.min() < 0 or owners.max() >= len(regions)):
        raise ValueError("T3 MIL input must be finite float32 scientific rows with exact int32 owners")


def load_combined_training_sources(old_candidate, round1_snapshot, base_candidate, t2_snapshot, bt2_snapshot):
    """Source groups in frozen G3, T2, then B-T2 order for regional refitting."""
    old = t2.load_combined_training_sources(old_candidate, round1_snapshot, base_candidate, t2_snapshot)
    new = load_bt2_source_bundle(base_candidate, bt2_snapshot)
    if {r.region_id for r in old["regions"]} & new["new_region_ids"]:
        raise ValueError("B-T2 region ID duplicates an existing G3/T2 label")
    groups = [*old["groups"], *new["groups"]]
    regions = [r for _source, group in groups for r in group]
    receipt = dict(old["receipt"], contract=CONTRACT,
        bt2=new["receipt"], combined_label_sha256=label_digest(regions),
        combined_region_ids_sha256=digest([r.region_id for r in regions]),
        old_hidden_run_inventory_present={run: any(r.run == run for r in old["old_regions"]) for run in sorted(HIDDEN_RUNS)})
    return dict(groups=groups, regions=regions, old_regions=old["old_regions"],
        new_regions=old["new_regions"], new_region_ids=old["new_region_ids"],
        bt2_regions=new["regions"], bt2_region_ids=new["new_region_ids"], receipt=receipt)


def source_family(region, t2_ids, bt2_ids):
    if region.region_id in bt2_ids:
        return "bt2_positives" if region.phase_class == "gas" else "bt2_negatives"
    if region.region_id in t2_ids:
        return "t2_negatives"
    if region.payload.get("gas_variant_annotation_sha256") == ROUND1_ANNOTATIONS_SHA256:
        return "round1"
    target = logistic.regional_presence_target(region.phase_class, HEAD)
    return "g3_positives" if target == 1 else "g3_negatives" if target == 0 else "g3_not_gas_supervision"


def supervision_weight_report(features, owners, regions, t2_ids, bt2_ids):
    """Separate raw per-box weights from actual run/class-balanced loss mass."""
    _matrix, _targets, weights, selected_owners = logistic.balanced_pixel_view(features, owners, regions, head=HEAD)
    loss = np.bincount(selected_owners, weights=weights, minlength=len(regions))
    counts = np.bincount(owners, minlength=len(regions))
    result = {name: dict(regions=0, sample_rows=0, raw_box_weight=0., gas_supervision_box_weight=0.,
                        balanced_direct_pixel_loss_weight=0., mil_presence_regions=0)
              for name in ("g3_positives", "g3_negatives", "round1", "t2_negatives", "bt2_positives", "bt2_negatives", "g3_not_gas_supervision")}
    for index, region in enumerate(regions):
        item = result[source_family(region, t2_ids, bt2_ids)]
        target = logistic.regional_presence_target(region.phase_class, HEAD)
        item["regions"] += 1
        item["sample_rows"] += int(counts[index])
        item["raw_box_weight"] += region.supervision_weight
        item["gas_supervision_box_weight"] += region.supervision_weight if target is not None and counts[index] else 0.
        item["balanced_direct_pixel_loss_weight"] += float(loss[index])
        item["mil_presence_regions"] += int(logistic.pixel_target(region.phase_class, HEAD) is None and target is not None and region.supervision_weight > 0 and counts[index] > 0)
    for item in result.values():
        item["fraction_of_direct_pixel_loss"] = item["balanced_direct_pixel_loss_weight"] / float(weights.sum())
    result["policy"] = "confidence-times-source per box; equal (run,target) direct strata; uniform sample mass within each box; MIL settings unchanged"
    return result


def training_counts(regions, owners, t2_ids, bt2_ids):
    counts = np.bincount(owners, minlength=len(regions))
    by_source = defaultdict(Counter)
    visible_regions = [r for r in regions if r.run not in HIDDEN_RUNS]
    by_run = t2.training_counts(regions, owners)
    for run in HIDDEN_RUNS:
        if run in by_run:
            by_run[run] = {"retained_historical_inventory": True, "hidden_pass_fail_only": True}
    for index, region in enumerate(regions):
        key = source_family(region, t2_ids, bt2_ids)
        by_source[key][region.phase_class] += 1
    return dict(by_run=by_run, by_source={k: dict(v) for k, v in sorted(by_source.items())},
                visible_runs=sorted({r.run for r in visible_regions}), hidden_runs_pass_fail_only=True)


def extraction_worker_allocation(requested, tasks, *, available_bytes=None):
    """Bound frame parallelism by free RAM, reserving capacity for the laptop."""
    if int(requested) < 1 or tasks < 1:
        raise ValueError("T3 extraction needs a positive requested worker and frame count")
    if available_bytes is None:
        try:
            import psutil
            available_bytes = int(psutil.virtual_memory().available)
        except ImportError:
            # No guessed memory permits an aggressive allocation.
            available_bytes = 7 * 1024**3
    estimated_worker_bytes, reserve_bytes = 3 * 1024**3, 4 * 1024**3
    capacity = max(1, (int(available_bytes) - reserve_bytes) // estimated_worker_bytes)
    actual = min(int(requested), int(tasks), capacity)
    return dict(requested=int(requested), frame_tasks=int(tasks), actual=int(actual),
        available_bytes=int(available_bytes), reserved_bytes=reserve_bytes,
        estimated_peak_bytes_per_worker=estimated_worker_bytes,
        policy="at-most-requested/source-frame-count; three-GiB-estimate-per-worker plus four-GiB-reserve")


def prepare_dataset(experiment, t2_dataset, base_candidate, bt2_snapshot, output, *, spec, workers=3):
    """Keep every T2 feature/owner byte; append source-exact rig-clipped B-T2."""
    output = Path(output)
    checked(spec, SPEC_SHA256)
    checked(resolve_review_path(experiment) / "variants/G3/gas_head.json", G3_SHA256)
    checked(resolve_review_path(base_candidate) / "models/generic/co2_presence_signal.json", CO2_SHA256)
    frozen, new = load_frozen_t2_dataset(t2_dataset), load_bt2_source_bundle(base_candidate, bt2_snapshot)
    regions = [*frozen["regions"], *new["regions"]]
    if len({r.region_id for r in regions}) != 3393:
        raise ValueError("T3 duplicates/misses an existing G3/T2 or promoted B-T2 region")
    initial = dict(contract=CONTRACT, spec_path=str(spec), spec_sha256=SPEC_SHA256,
        frozen_g3_head_sha256=G3_SHA256, frozen_t2_dataset_receipt_path=str(frozen["receipt_path"]),
        frozen_t2_dataset_receipt_sha256=T2_DATASET_SHA256, bt2=new["receipt"],
        frozen_co2_head_sha256=CO2_SHA256, code_sha256=dict(
            t3_training=sha256_file(Path(__file__)), t2_source_extractor=sha256_file(Path(t2.__file__))))
    write(output / "initial_binding.json", initial)
    regions_path = output / "regions.json"
    write(regions_path, [asdict(r) for r in regions])
    reports, pending = {}, []
    for source, group in new["groups"]:
        for index in sorted({r.frame_index for r in group}):
            local = [r for r in group if r.frame_index == index]
            key = f"{source.run}/frame_{index:03d}"
            destination = output / "bt2_extraction" / key
            receipt_path = destination / "receipt.json"
            if receipt_path.is_file():
                receipt = read(receipt_path)
                checked(receipt["sample_path"], receipt["samples_sha256"])
                if [row["region_id"] for row in receipt["coverage"]] != [r.region_id for r in local]:
                    raise ValueError("Resumed B-T2 extraction region order changed")
                reports[key] = receipt
            else:
                pending.append((key, source.run, [asdict(r) for r in local], str(destination)))
    allocation = extraction_worker_allocation(workers, 12)
    write(output / "extraction_allocation.json", allocation, progress=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(allocation["actual"], len(pending)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(t2._extract_t2_run, str(base_candidate), run, group, destination, 1024): key
                       for key, run, group, destination in pending}
            for future in as_completed(futures):
                report = future.result()
                key = futures[future]
                reports[key] = report
                write(output / "progress.json", dict(stage="bt2_extracting", completed=len(reports), total=12,
                    last_frame=key, seconds=time.perf_counter()-started), progress=True)
                print(f"B-T2 extracted {key}: {report['sample_rows']} scientific sample rows", flush=True)
    matrices, ownership = [frozen["features"]], [frozen["owners"]]
    global_indices = {r.region_id: i for i, r in enumerate(regions)}
    for source, group in new["groups"]:
        for index in sorted({r.frame_index for r in group}):
            local = [r for r in group if r.frame_index == index]
            key = f"{source.run}/frame_{index:03d}"
            with np.load(reports[key]["sample_path"], allow_pickle=False) as archive:
                matrices.append(archive["features"].copy())
                mapping = np.asarray([global_indices[r.region_id] for r in local], np.int32)
                ownership.append(mapping[archive["region_indices"]])
    features, owners = np.concatenate(matrices), np.concatenate(ownership)
    _check_arrays(features, owners, regions)
    if (not np.array_equal(features[:len(frozen["features"])], frozen["features"])
            or not np.array_equal(owners[:len(frozen["owners"])], frozen["owners"])):
        raise ValueError("Imported T2 samples changed during the T3 append")
    for name, array in (("flatfield.npy", features), ("owners.npy", owners)):
        path = output / name
        if path.is_file():
            if not np.array_equal(np.load(path, allow_pickle=False), array):
                raise ValueError("Resumed T3 merged dataset differs")
        else:
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
    t2_ids, bt2_ids = set(frozen["receipt"]["new_region_ids"]), new["new_region_ids"]
    receipt = dict(**initial, combined_label_sha256=label_digest(regions), combined_regions=3393,
        old_regions=2849, t2_regions=223, bt2_regions=321, bt2_positives=140, bt2_negatives=181,
        original_round1_region_count=871, imported_t2_sample_rows=len(frozen["features"]),
        bt2_sample_rows=len(features)-len(frozen["features"]),
        old_features_and_owner_indices_bit_identical=True,
        original_ac31_off_timeline=frozen["receipt"]["original_ac31_off_timeline"],
        old_extraction_receipt_sha256=frozen["receipt"]["old_extraction_receipt_sha256"],
        training_label_sources=frozen["receipt"]["training_label_sources"], t2=frozen["receipt"]["t2"],
        region_inventory_path=str(regions_path), region_inventory_sha256=sha256_file(regions_path),
        feature_path=str(output / "flatfield.npy"), feature_sha256=sha256_file(output / "flatfield.npy"),
        owner_path=str(output / "owners.npy"), owner_sha256=sha256_file(output / "owners.npy"),
        new_region_ids=sorted(t2_ids), bt2_region_ids=sorted(bt2_ids),
        training_counts=training_counts(regions, owners, t2_ids, bt2_ids),
        weights=supervision_weight_report(features, owners, regions, t2_ids, bt2_ids),
        old_hidden_run_inventory_present={run: any(r.run == run for r in frozen["regions"][:2849]) for run in sorted(HIDDEN_RUNS)},
        source_feature_reports=[reports[key] for key in sorted(reports)],
        extraction_allocation=allocation,
        training_input_policy="bit-identical-complete-T2-rows-plus-promoted-sealed-B-T2-production-flatfield-rows",
        rounds_2_3_4_read=False, new_blind_session_labels_read=False, live_session_read=False,
        source_clipping="rig_domain intersect observed current/baseline/previous support intersect modelled",
        extraction_seconds=time.perf_counter()-started)
    write(output / "dataset_receipt.json", receipt)
    write(output / "progress.json", dict(stage="dataset_ready", regions=3393, rows=len(features)), progress=True)
    return receipt


def fit_pixel_g3_t3(experiment, dataset_root, output, *, threads=6):
    """One CPU MIL fit with exactly G3's fixed settings; no parameter search."""
    from threadpoolctl import threadpool_limits
    dataset_root, output, experiment = Path(dataset_root), Path(output), resolve_review_path(experiment)
    receipt = read(dataset_root / "dataset_receipt.json")
    old_head = read(checked(experiment / "variants/G3/gas_head.json", G3_SHA256))
    if (receipt["contract"] != CONTRACT or receipt["frozen_t2_dataset_receipt_sha256"] != T2_DATASET_SHA256
            or receipt["spec_sha256"] != SPEC_SHA256 or receipt["combined_regions"] != 3393
            or receipt["bt2"]["annotation_csv_sha256"] != BT2_ANNOTATIONS_SHA256):
        raise ValueError("T3 pixel fit received a different dataset contract/seal")
    for path_key, sha_key in (("feature_path", "feature_sha256"), ("owner_path", "owner_sha256"),
                              ("region_inventory_path", "region_inventory_sha256")):
        checked(receipt[path_key], receipt[sha_key])
    features = np.load(receipt["feature_path"], mmap_mode="r", allow_pickle=False)
    owners = np.load(receipt["owner_path"], mmap_mode="r", allow_pickle=False)
    regions = [logistic.ReviewedRegion(**row) for row in read(receipt["region_inventory_path"])]
    _check_arrays(features, owners, regions)
    if label_digest(regions) != receipt["combined_label_sha256"]:
        raise ValueError("Merged G3/T2/B-T2 label SHA differs from the extraction")
    logistic.assert_safe_training_regions(regions)
    settings = dict(old_head["training_provenance"]["fit_settings"])
    if settings["backend"] != "cpu" or not 1 <= int(threads) <= 32:
        raise ValueError("T3 MIL fit requires a bounded CPU-only allocation")
    initial = dict(contract=CONTRACT, training_revision="G3-T3",
        dataset_receipt_sha256=sha256_file(dataset_root / "dataset_receipt.json"), parent_head_sha256=G3_SHA256,
        fit_settings=settings, cpu_only=True, no_regularization_reselection=True, threads=int(threads),
        code_sha256={"trainer": sha256_file(Path(__file__)), "source_extractor": sha256_file(Path(t2.__file__)),
                     "logistic": sha256_file(Path(logistic.__file__))})
    write(output / "initial_binding.json", initial)
    write(output / "progress.json", dict(stage="pixel_MIL_fit_running", started_utc=datetime.now(timezone.utc).isoformat()), progress=True)
    started, cpu_started = time.perf_counter(), time.process_time()
    cv2.setNumThreads(1)
    with threadpool_limits(limits=int(threads)):
        fit, diagnostics = logistic.fit_weighted_logistic_mil(features, owners, regions, head=HEAD, runs=None, **settings)
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    model = dict(format=old_head["format"], head=HEAD, variant="G3", training_revision="G3-T3",
        feature_contract=old_head["feature_contract"], feature_names=list(FLATFIELD_NAMES),
        fit=safe(asdict(fit)), gas_display_threshold=.6,
        training_provenance=dict(**initial, combined_label_sha256=receipt["combined_label_sha256"],
            dataset_receipt_path=str(dataset_root / "dataset_receipt.json"), unchanged_co2_head_sha256=CO2_SHA256,
            pixel_threshold=.7, regional_threshold=.6, gas_minimum_mm2=2.5, connectivity=4,
            rounds_2_3_4_read=False, new_blind_session_labels_read=False, live_session_read=False,
            training_scope="exact-G3-T2-plus-321-sealed-scored-B-T2-boxes", mass_eligible=False))
    write(output / "gas_head.json", model)
    result = dict(**initial, converged=bool(fit.converged), fit_wall_seconds=wall, fit_cpu_seconds=cpu,
        iterations=fit.iterations, old_fit=old_head["fit"], new_fit=safe(asdict(fit)), diagnostics=diagnostics,
        model_path=str(output / "gas_head.json"), model_sha256=sha256_file(output / "gas_head.json"),
        weights=receipt["weights"], training_counts=receipt["training_counts"],
        training_label_sha256=receipt["combined_label_sha256"], training_region_ids=[r.region_id for r in regions],
        training_region_inventory_path=receipt["region_inventory_path"],
        training_region_inventory_sha256=receipt["region_inventory_sha256"],
        old_hidden_run_inventory_present=receipt["old_hidden_run_inventory_present"],
        exclusions_by_session_and_label_sha=dict(allowed_label_sources=receipt["training_label_sources"],
            t2=receipt["t2"], promoted_bt2=receipt["bt2"], exact_frozen_t2_dataset_sha256=T2_DATASET_SHA256,
            rounds_2_3_4_read=False, new_blind_session_labels_read=False),
        rounds_2_3_4_read=False, new_blind_session_labels_read=False, live_session_read=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    write(output / "training_report.json", result)
    write(output / "fit_receipt.json", result)
    write(output / "progress.json", dict(stage="pixel_fit_complete", converged=bool(fit.converged),
        model_sha256=result["model_sha256"], wall_seconds=wall), progress=True)
    if not fit.converged:
        raise ValueError("T3 gas MIL fit did not converge; diagnostics preserved and downstream blocked")
    return result
