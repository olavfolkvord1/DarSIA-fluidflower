"""Source-exact CPU refit of only the regional G3 gas confirmation head.

The frozen pixel heads, the regional CO2 head, and every regional setting are
immutable inputs.  Historical labels retain their own native revisions.  Only
the sealed round-1 contours are appended; held-out labels are never opened.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_optical_regions as regional
from .phase_gas_variant_sources import HELD_OUT_RUNS, load_training_sources
from .phase_training import logistic
from .project import sha256_file

FORMAT = "ffac-g3-regional-gas-only-refit-v1"
HEAD = "gas_within_co2"
G3_SHA256 = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
CO2_SHA256 = "8c3727110be01dbef4849fad6849e50bcc8e28bdd1ee22a1c24c0ab8027c64e3"
TRAINING_LABEL_SHA256 = "02767870373795de7ea5a0f748f8e74239a4def8c85c196d84bba8b02ad3511b"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    """Commit a complete metadata file; reject any changed existing receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = _json_safe(value)
    if path.is_file():
        if _read(path) != value:
            raise ValueError(f"Existing regional refit receipt differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _progress(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def inventory_contract(regions, g3_report):
    """Verify the exact G3 region identities and explicit supervised payloads."""
    if not regions or len({r.region_id for r in regions}) != len(regions):
        raise ValueError("Regional gas inventory is empty or contains duplicate IDs")
    if {r.run.casefold() for r in regions} & HELD_OUT_RUNS:
        raise ValueError("A held-out run entered regional gas training")
    logistic.assert_safe_training_regions(regions)
    labels = [dict(region_id=r.region_id, run=r.run.casefold(),
                   phase_class=r.phase_class, confidence=r.confidence,
                   source=r.source, payload=dict(r.payload)) for r in regions]
    label_sha = _digest(labels)
    if (label_sha != TRAINING_LABEL_SHA256
            or label_sha != g3_report["training_label_sha256"]
            or [r.region_id for r in regions] != g3_report["training_region_ids"]):
        raise ValueError("Regional labels differ from the exact G3 training inventory")
    return dict(region_count=len(regions), training_label_sha256=label_sha,
                training_region_ids_sha256=_digest([r.region_id for r in regions]),
                unmarked_pixels_are_negatives=False,
                round2_labels_read=False,
                held_out_runs=sorted(HELD_OUT_RUNS))


def _record_weight(record):
    """Use the optimiser's confidence/source policy, not confidence alone."""
    return logistic.ReviewedRegion(**record).supervision_weight


def _majority_cell_fallback(prepared, records, native, native_support):
    """The frozen majority-covered cell sampler, restricted to valid G3 input.

    Only original marked native pixels with current G3 support enter pooling.
    The original >0.5 coverage, working-domain gate, contradictory phase bits,
    512-row cap and original-ID seed remain unchanged. Four neighbouring
    working cells are not required because these are direct working-cell
    features, not interpolated native-point features.
    """
    from .tools.phase_coupled_constraint_comparison import annotation_window
    from .tools.train_series_b_completion import pool_annotation
    restrictions = dict(water=1, aqueous=2, gas=4, co2_absent_region=1,
                        current_experiment_co2_artifact=1, gas_absent_region=3,
                        aqueous_absent_region=5, aqueous_present_gas_absent_region=3,
                        mixed_aqueous_water=3)
    shape = prepared["domain"].shape
    allowed = np.full(shape, 7, np.uint8)
    windows = []
    for owner, record in records:
        if _record_weight(record) <= 0:
            continue
        bounds, mask = annotation_window(record["payload"], canvas_wh=native["canvas_wh"],
                                         content_box=native["content_box_xywh"])
        nx0, ny0, nx1, ny1 = bounds
        mask &= native_support[ny0:ny1, nx0:nx1]
        pooled, (x0, y0, x1, y1) = pool_annotation(
            mask, bounds, native["analysis_image_shape_hw"], shape)
        selected = (pooled > .5) & prepared["domain"][y0:y1, x0:x1]
        windows.append((owner, record, (x0, y0, x1, y1), selected))
        if record["phase_class"] in restrictions:
            allowed[y0:y1, x0:x1][selected] &= restrictions[record["phase_class"]]
    result = {}
    for owner, record, (x0, y0, x1, y1), selected in windows:
        conflict = selected & (allowed[y0:y1, x0:x1] == 0)
        rows, cols = np.nonzero(selected & ~conflict)
        if len(rows) > 512:
            seed = int.from_bytes(hashlib.sha256(record["region_id"].encode()).digest()[:8], "little")
            take = np.random.default_rng(seed).choice(len(rows), 512, replace=False)
            rows, cols = rows[take], cols[take]
        result[owner] = (prepared["features"][rows+y0, cols+x0], dict(
            sampling_method="frozen-majority-covered-regional-cells-fallback",
            selected_working_pixels=int(selected.sum()),
            conflicting_working_pixels=int(conflict.sum()),
            fallback_validity_policy="pool-only-original-label-pixels-with-native-G3-support"))
    return result


def supervision_completeness(regions, owners, coverage):
    """Zero-weight labels remain in inventory but need no training rows."""
    counts = np.bincount(owners, minlength=len(regions))
    missing, zero_weight = [], []
    for index, region in enumerate(regions):
        weight = region.supervision_weight
        if weight <= 0:
            zero_weight.append(dict(region_id=region.region_id, run=region.run,
                                   phase_class=region.phase_class, confidence=region.confidence,
                                   source=region.source, supervision_weight=float(weight),
                                   training_rows=int(counts[index]),
                                   reason="zero-optimizer-supervision-weight"))
        elif counts[index] == 0:
            missing.append(region.region_id)
    fallbacks = [dict(row) for row in coverage if row.get("fallback_used")]
    return dict(missing_positive_weight_region_ids=missing,
                zero_weight_exclusions=zero_weight, fallback_regions=fallbacks,
                supervision_weight_policy="ReviewedRegion.supervision_weight:confidence-times-source")


def prepare_training_samples(frame, records):
    """Keep all 20 contextual fields; restrict sampling to G3-supported pixels.

    The evaluator's ``prepared`` contains the original 18 contextual columns
    and pure G3 ``gas_joint_mean`` columns. Invalid G3 values are zero only in
    the masked probability calculation, never an old-head fallback or label.
    Use the existing deterministic native-point sampler for every region. This
    preserves tiny contours without requiring their entire 1 mm cell to be
    covered, and excludes invalid G3 points before interpolation. Positive-
    weight regions with no native rows may retain the frozen majority-covered
    working-cell samples. Zero-weight labels remain audit inventory only.
    """
    from .tools.train_phase_optical_regions import small_region_samples
    from .tools.phase_coupled_constraint_comparison import annotation_window
    prepared = frame.prepared
    domain = np.asarray(frame.domain, bool)
    flat_valid = np.asarray(frame.flat_valid, bool)
    if domain.shape != flat_valid.shape or prepared["features"].shape[-1] != 20:
        raise ValueError("Regional G3 fields or native support differ")
    native_support = domain & flat_valid
    photo_sand = np.asarray(frame.photo_sand_domain, bool)
    restrictions = dict(water=1, aqueous=2, gas=4, co2_absent_region=1,
                        current_experiment_co2_artifact=1, gas_absent_region=3,
                        aqueous_absent_region=5, aqueous_present_gas_absent_region=3,
                        mixed_aqueous_water=3)
    windows = []
    positive_records = [(owner, record) for owner, record in records if _record_weight(record) > 0]
    for _owner, record in positive_records:
        bounds, _mask = annotation_window(record["payload"], canvas_wh=frame.native["canvas_wh"],
                                          content_box=frame.native["content_box_xywh"])
        windows.append((record, bounds))
    matrices, owner_arrays, coverage_rows = [], [], []
    majority_fallbacks = None
    for owner, record in records:
        bounds, mask = annotation_window(record["payload"], canvas_wh=frame.native["canvas_wh"],
                                         content_box=frame.native["content_box_xywh"])
        x0, y0, x1, y1 = bounds
        weight = _record_weight(record)
        fallback_used = False
        if weight <= 0:
            matrix = np.empty((0, len(regional.FEATURE_NAMES)), np.float32)
            sampling = dict(sampling_method="excluded-zero-weight-region",
                            native_pixels_in_domain=0, conflicting_native_pixels=0)
        else:
            matrix, sampling = small_region_samples(
                prepared, record, frame.native, native_support, windows, restrictions)
            if not len(matrix):
                if majority_fallbacks is None:
                    majority_fallbacks = _majority_cell_fallback(
                        prepared, positive_records, frame.native, native_support)
                fallback_matrix, fallback_sampling = majority_fallbacks[owner]
                if len(fallback_matrix):
                    matrix = fallback_matrix
                    sampling = {**sampling, **fallback_sampling,
                                "native_sampling_method": sampling["sampling_method"],
                                "native_training_rows_before_fallback": 0,
                                "fallback_reason": "positive-weight-region-has-zero-native-interpolated-rows"}
                    fallback_used = True
        if len(matrix):
            matrices.append(matrix)
            owner_arrays.append(np.full(len(matrix), owner, np.int32))
        coverage_rows.append(dict(region_id=record["region_id"], run=record["run"],
            frame_index=record["frame_index"], phase_class=record["phase_class"],
            confidence=record["confidence"], supervision_weight=float(weight),
            training_rows=len(matrix), fallback_used=fallback_used, **sampling,
            label_pixels_observable=int(mask.sum()),
            label_pixels_photo_certified_sand=int(np.count_nonzero(
                mask & photo_sand[y0:y1, x0:x1])),
            label_pixels_g3_supported=int(np.count_nonzero(
                mask & native_support[y0:y1, x0:x1])),
            status=("included" if len(matrix) else "excluded-zero-weight-region"
                    if weight <= 0 else "no-unambiguous-supervision")))
    values = (np.concatenate(matrices) if matrices else
              np.empty((0, len(regional.FEATURE_NAMES)), np.float32))
    owners = np.concatenate(owner_arrays) if owner_arrays else np.empty(0, np.int32)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Regional G3 samples must be finite float32")
    return values, owners, coverage_rows


def replace_only_gas_head(frozen_model, gas_fit):
    result = deepcopy(frozen_model)
    result["heads"][HEAD] = _json_safe(asdict(gas_fit))
    for key in frozen_model:
        if key != "heads" and result[key] != frozen_model[key]:
            raise ValueError("Refit changed a frozen regional setting")
    for head in frozen_model["heads"]:
        if head != HEAD and result["heads"][head] != frozen_model["heads"][head]:
            raise ValueError("Refit changed an immutable regional head")
    return result


def _verify_sample_receipt(path, binding):
    receipt = _read(path)
    if receipt["binding"] != binding:
        raise ValueError("Regional sample cache belongs to another source or model")
    for key in ("samples",):
        if sha256_file(Path(receipt[key]["path"])) != receipt[key]["sha256"]:
            raise ValueError("Regional sample cache changed after sealing")
    return receipt


def _source_worker(task, *, evaluator_factory=None):
    """Own one immutable source revision; resume at complete frame receipts."""
    from threadpoolctl import threadpool_limits
    from .phase_g3_probability_layers import NativeG3FrameEvaluator
    source, group, owner_start, binding, folder, threads = task
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source_receipt = folder / "source_receipt.json"
    if source_receipt.is_file():
        return _verify_sample_receipt(source_receipt, binding)
    cv2.setNumThreads(int(threads))
    records = [(owner_start + i, asdict(region)) for i, region in enumerate(group)]
    frames = defaultdict(list)
    for owner, record in records:
        frames[int(record["frame_index"])].append((owner, record))
    evaluator = None
    started = time.perf_counter()
    frame_receipts, matrices, owner_arrays, coverage = [], [], [], []
    with threadpool_limits(limits=int(threads)):
        for index, frame_records in sorted(frames.items()):
            frame_binding = {**binding, "frame_index": index,
                             "region_ids": [r["region_id"] for _, r in frame_records]}
            receipt_path = folder / f"frame_{index:03d}.json"
            if receipt_path.is_file():
                receipt = _verify_sample_receipt(receipt_path, frame_binding)
            else:
                if evaluator is None:
                    if evaluator_factory is not None:
                        evaluator = evaluator_factory(binding, source, int(threads))
                    elif binding.get("frame_evaluator") == "g3-t2-source-exact-v1":
                        from .phase_gas_t2_regional_inputs import source_evaluator
                        evaluator = source_evaluator(binding, source, int(threads))
                    else:
                        evaluator = NativeG3FrameEvaluator(
                            candidate=Path(binding["frozen_candidate"]),
                            g3_model_path=Path(binding["g3_model_path"]),
                            native_manifest_path=source.asset_manifest_path,
                            run=source.run, threads=int(threads), source=source.review_root,
                            independent_reference_frames=source.independent_reference_frames)
                frame_started = time.perf_counter()
                frame = evaluator.evaluate(index, export_frozen_co2=False)
                matrix, owners, frame_coverage = prepare_training_samples(frame, frame_records)
                sample_path = folder / f"frame_{index:03d}.npz"
                temporary = sample_path.with_suffix(".npz.partial")
                with temporary.open("wb") as stream:
                    np.savez(stream, features=matrix, owners=owners)
                temporary.replace(sample_path)
                receipt = dict(binding=frame_binding,
                               samples=dict(path=str(sample_path), sha256=sha256_file(sample_path)),
                               coverage=frame_coverage, source_context=frame.metadata,
                               seconds=time.perf_counter() - frame_started)
                _write(receipt_path, receipt)
                del frame, matrix, owners
            with np.load(receipt["samples"]["path"], allow_pickle=False) as archive:
                matrices.append(archive["features"])
                owner_arrays.append(archive["owners"])
            coverage.extend(receipt["coverage"])
            frame_receipts.append(dict(path=str(receipt_path), sha256=sha256_file(receipt_path)))
    sample_path = folder / "source_samples.npz"
    temporary = sample_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, features=np.concatenate(matrices), owners=np.concatenate(owner_arrays))
    temporary.replace(sample_path)
    receipt = dict(binding=binding, run=source.run,
                   samples=dict(path=str(sample_path), sha256=sha256_file(sample_path)),
                   coverage=coverage, frames=frame_receipts,
                   seconds=time.perf_counter() - started)
    _write(source_receipt, receipt)
    return receipt


def _training_counts(regions, owners):
    counts = np.bincount(owners, minlength=len(regions))
    result = defaultdict(lambda: defaultdict(lambda: dict(regions=0, sample_rows=0)))
    for index, region in enumerate(regions):
        row = result[region.run][region.phase_class]
        row["regions"] += 1
        row["sample_rows"] += int(counts[index])
    return {run: dict(classes) for run, classes in sorted(result.items())}


def extract_regional_training(frozen_candidate, experiment, snapshot, output, *,
                              workers=8, threads=1, source_bundle=None):
    """Extract native-source regional samples in parallel, caching each frame."""
    candidate, experiment, output = map(Path, (frozen_candidate, experiment, output))
    output.mkdir(parents=True, exist_ok=True)
    g3_path = experiment / "variants/G3/gas_head.json"
    if sha256_file(g3_path) != G3_SHA256:
        raise ValueError("G3 pixel head changed before regional extraction")
    if sha256_file(candidate / "models/generic/co2_presence_signal.json") != CO2_SHA256:
        raise ValueError("Frozen CO2 presence head changed")
    fit_receipt = _read(experiment / "fit_receipt.json")
    report = _read(experiment / "variants/G3/training_report.json")
    bundle = source_bundle or load_training_sources(candidate, Path(snapshot))
    groups = bundle["groups"]
    regions = [r for _, group in groups for r in group]
    inventory = inventory_contract(regions, report)
    frozen_model_path = candidate / "regional_model.json"
    implementation = dict(refit_code_sha256=sha256_file(Path(__file__)),
                          regional_code_sha256=sha256_file(Path(regional.__file__)))
    from . import phase_g3_probability_layers as evaluator_module
    from .tools import train_phase_optical_regions as sampling
    implementation.update(evaluator_code_sha256=sha256_file(Path(evaluator_module.__file__)),
                          regional_sampling_code_sha256=sha256_file(Path(sampling.__file__)))
    binding = dict(format=FORMAT, frozen_candidate=str(candidate),
                   g3_model_path=str(g3_path), g3_model_sha256=G3_SHA256,
                   co2_head_sha256=CO2_SHA256,
                   frozen_regional_sha256=sha256_file(frozen_model_path),
                   original_fit_receipt_sha256=sha256_file(experiment / "fit_receipt.json"),
                   source_inventory_sha256=_digest(bundle["receipt"]),
                   inventory=inventory, implementation=implementation,
                   feature_names=list(regional.FEATURE_NAMES),
                   feature_policy="18-frozen-context-columns-plus-pure-G3-gas-joint-means",
                   sampling_policy="native-points-max512-original-ID-seed-with-frozen-majority-cell-fallback-v2",
                   invalid_g3_policy="zero-for-masked-pooling-only;never-training-support;no-old-head-fallback")
    if fit_receipt["frozen_files_sha256"]["regional_model.json"] != binding["frozen_regional_sha256"]:
        raise ValueError("Frozen regional model differs from G3 experiment")
    _write(output / "extraction_binding.json", binding)
    _write(output / "source_preflight.json", bundle["receipt"])
    _write(output / "regions.json", [asdict(r) for r in regions])
    final_receipt = output / "extraction_receipt.json"
    if final_receipt.is_file():
        return _verify_sample_receipt(final_receipt, binding)
    tasks, offset = [], 0
    for source, group in groups:
        local_binding = {**binding, "native_manifest_path": str(source.asset_manifest_path),
                         "native_manifest_sha256": sha256_file(source.asset_manifest_path),
                         "source_region_ids_sha256": _digest([r.region_id for r in group]),
                         "owner_start": offset}
        folder = output / "source_samples" / f"{offset:04d}-{source.run}-{_digest(local_binding)[:16]}"
        tasks.append((source, group, offset, local_binding, str(folder), max(1, int(threads))))
        offset += len(group)
    started = time.perf_counter()
    reports = {}
    _progress(output / "progress.json", dict(stage="regional_extraction", completed=0,
              source_groups=len(tasks), workers=max(1, int(workers)), cpu_only=True))
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), len(tasks))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_source_worker, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            reports[futures[future]] = future.result()
            elapsed = time.perf_counter() - started
            _progress(output / "progress.json", dict(stage="regional_extraction",
                      completed=len(reports), source_groups=len(tasks), seconds=elapsed,
                      eta_seconds=elapsed / len(reports) * (len(tasks)-len(reports))))
            print(json.dumps(dict(stage="regional_training_sources", completed=len(reports),
                                  total=len(tasks), seconds=elapsed)), flush=True)
    matrices, owners, coverage, receipts = [], [], [], []
    for i, task in enumerate(tasks):
        report = reports[i]
        with np.load(report["samples"]["path"], allow_pickle=False) as archive:
            matrices.append(archive["features"])
            owners.append(archive["owners"])
        coverage.extend(report["coverage"])
        receipt_path = Path(task[4]) / "source_receipt.json"
        receipts.append(dict(path=str(receipt_path), sha256=sha256_file(receipt_path)))
    matrix, owners = np.concatenate(matrices), np.concatenate(owners)
    if matrix.dtype != np.float32 or matrix.shape != (len(owners), 20) or not np.isfinite(matrix).all():
        raise ValueError("Regional extraction produced invalid model input")
    if owners.dtype != np.int32 or (owners < 0).any() or (owners >= len(regions)).any():
        raise ValueError("Regional extraction lost exact region owners")
    support_audit = supervision_completeness(regions, owners, coverage)
    missing = support_audit["missing_positive_weight_region_ids"]
    if missing:
        _write(output / "missing_region_support.json", dict(region_ids=missing, coverage=coverage,
                                                            **support_audit))
        raise ValueError(f"Regional G3 extraction has no supported samples for {len(missing)} training regions")
    sample_path = output / "training_samples.npz"
    temporary = sample_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, features=matrix, owners=owners)
    temporary.replace(sample_path)
    receipt = dict(binding=binding,
                   samples=dict(path=str(sample_path), sha256=sha256_file(sample_path)),
                   regions_sha256=sha256_file(output / "regions.json"),
                   coverage=coverage, source_receipts=receipts,
                   **support_audit,
                   training_counts=_training_counts(regions, owners),
                   rows=len(owners), region_count=len(regions),
                   extraction_seconds=time.perf_counter() - started,
                   cpu_only=True, round2_labels_read=False)
    _write(final_receipt, receipt)
    return receipt


def _verify_finished_refit(root, binding, frozen_model):
    receipt = _read(root / "regional_refit_receipt.json")
    if receipt["binding"] != binding:
        raise ValueError("Completed regional refit belongs to another source or model")
    path = Path(receipt["regional_model_path"])
    if sha256_file(path) != receipt["regional_model_sha256"]:
        raise ValueError("Completed regional refit model changed")
    report_path = Path(receipt["training_report_path"])
    if sha256_file(report_path) != receipt["training_report_sha256"]:
        raise ValueError("Completed regional refit training report changed")
    model = _read(path)
    frozen_except_gas, refit_except_gas = deepcopy(frozen_model), deepcopy(model)
    frozen_except_gas["heads"].pop(HEAD)
    refit_except_gas["heads"].pop(HEAD)
    if frozen_except_gas != refit_except_gas or not model["heads"][HEAD]["converged"]:
        raise ValueError("Completed regional refit changed frozen settings or did not converge")
    return receipt


def fit_regional_g3(frozen_candidate, experiment, snapshot, local_output, *,
                   workers=8, threads=1, fit_threads=None):
    """Task A: extract exact G3 owners, then replace only regional gas weights."""
    root = Path(local_output)
    candidate, experiment = Path(frozen_candidate), Path(experiment)
    started = time.perf_counter()
    extraction_root = root / "extraction"
    extraction = extract_regional_training(candidate, experiment, snapshot, extraction_root,
                                           workers=workers, threads=threads)
    frozen = _read(candidate / "regional_model.json")
    # Same region-label objective and regularisation as the existing regional
    # seed head. A longer fixed iteration ceiling certifies convergence without
    # changing loss, labels, thresholds, or regularisation after seeing results.
    settings = dict(regularization=float(frozen["heads"][HEAD]["regularization"]),
                    weak_strength=.35, positive_top_fraction=.10,
                    max_iter=15000, tolerance=2e-5, backend="cpu")
    binding = dict(extraction_binding=extraction["binding"],
                   extraction_receipt_sha256=sha256_file(extraction_root / "extraction_receipt.json"),
                   fitting=settings, original_regional_fit_report_inherited=True,
                   refit_objective="gas-only-region-MIL;CO2-head-not-part-of-optimisation",
                   original_final_objective="joint-aqueous-CPU64-polish;inherited-report-is-parent-report")
    _write(root / "refit_binding.json", binding)
    if (root / "regional_refit_receipt.json").is_file():
        return _verify_finished_refit(root, binding, frozen)
    regions = [logistic.ReviewedRegion(**r) for r in _read(extraction_root / "regions.json")]
    if sha256_file(extraction_root / "regions.json") != extraction["regions_sha256"]:
        raise ValueError("Regional refit label inventory changed")
    if sha256_file(extraction["samples"]["path"]) != extraction["samples"]["sha256"]:
        raise ValueError("Regional refit samples changed")
    with np.load(extraction["samples"]["path"], allow_pickle=False) as archive:
        matrix, owners = archive["features"], archive["owners"]
    from threadpoolctl import threadpool_limits
    fit_threads = max(1, int(fit_threads if fit_threads is not None else workers * threads))
    cv2.setNumThreads(fit_threads)
    _progress(root / "progress.json", dict(stage="regional_gas_fit", rows=len(owners),
              regions=len(regions), threads=fit_threads, cpu_only=True))
    report_path = root / "regional_gas_training_report.json"
    if report_path.is_file():
        # A process may stop after saving the fit and before committing the
        # model/receipt. Reuse the complete fitted checkpoint, never fit again.
        report = _read(report_path)
        if report.get("binding") != binding:
            raise ValueError("Interrupted regional fit checkpoint lost its source binding")
        fit_payload = dict(report["fit"])
        for name in ("mean", "scale", "coefficients"):
            fit_payload[name] = np.asarray(fit_payload[name], np.float64)
        fit = logistic.LogisticFit(**fit_payload)
    else:
        fit_started = time.perf_counter()
        with threadpool_limits(limits=fit_threads):
            fit, diagnostics = logistic.fit_weighted_logistic_mil(
                matrix, owners, regions, head=HEAD, runs=None, **settings)
        report = dict(binding=binding, fit=_json_safe(asdict(fit)), diagnostics=diagnostics,
                      fit_seconds=time.perf_counter()-fit_started,
                      training_counts=extraction["training_counts"],
                      training_label_sha256=TRAINING_LABEL_SHA256,
                      supervision_weights_sha256=_digest([
                          {"region_id": r.region_id, "weight": r.supervision_weight}
                          for r in regions]),
                      gas_head_weights_sha256=_digest({
                          k: v for k, v in asdict(fit).items()
                          if k in ("mean", "scale", "coefficients", "intercept")}),
                      region_count=len(regions), new_regions=871, cpu_only=True)
        _write(report_path, report)
    if not fit.converged:
        raise ValueError("Regional G3 gas refit did not converge; no model was exported")
    model = replace_only_gas_head(frozen, fit)
    model_path = root / "regional_model.json"
    _write(model_path, model)
    receipt = dict(format=FORMAT, binding=binding,
                   regional_model_path=str(model_path), regional_model_sha256=sha256_file(model_path),
                   regional_co2_sha256=_digest(frozen["heads"]["co2_presence"]),
                   frozen_regional_model_sha256=sha256_file(candidate / "regional_model.json"),
                   g3_pixel_sha256=G3_SHA256, co2_pixel_sha256=CO2_SHA256,
                   changed_head=HEAD, all_other_regional_fields_exact=True,
                   training_label_sha256=TRAINING_LABEL_SHA256,
                   training_counts=extraction["training_counts"],
                   training_report_path=str(root / "regional_gas_training_report.json"),
                   training_report_sha256=sha256_file(root / "regional_gas_training_report.json"),
                   converged=True, cpu_only=True, round2_labels_read=False,
                   final_maps_generated=False, studio_candidate_created=False,
                   fit_seconds=report["fit_seconds"], total_seconds=time.perf_counter()-started)
    _write(root / "regional_refit_receipt.json", receipt)
    _progress(root / "progress.json", dict(stage="regional_refit_complete", **receipt))
    return receipt


__all__ = ["fit_regional_g3", "extract_regional_training", "inventory_contract",
           "prepare_training_samples", "replace_only_gas_head", "supervision_completeness"]
