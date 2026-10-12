"""Sealed G3+T2+scored-BT2 source extraction and CPU refit of the regional gas head only.

Historical G3 extraction/fit defaults stay unchanged. Each new cache binds the
new pixel head, exact old inventory, T2 snapshot, source revision and code.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_g3_regional_refit as inherited
from . import phase_gas_t3_regional_inputs as inputs
from . import phase_gas_t3_training as training
from . import phase_gas_t2_regional_inputs as shared_inputs
from . import phase_optical_regions as regional
from .phase_gas_variant_sources import HELD_OUT_RUNS
from .phase_training import logistic
from .project import sha256_file

FORMAT = "ffac-g3-t3-regional-gas-only-refit-v1"
HEAD = inherited.HEAD


def inventory_contract(bundle, pixel_model):
    """Admit only the three exact scored/training snapshots in their fixed order."""
    regions = [r for _source, group in bundle["groups"] for r in group]
    old, t2, bt2 = bundle["old_regions"], bundle["new_regions"], bundle["bt2_regions"]
    receipt = bundle["receipt"]
    if (len(old) != 2849 or len(t2) != 223 or len(bt2) != 321 or len(regions) != 3393
            or len({r.region_id for r in regions}) != len(regions)
            or [r.region_id for r in regions] != [r.region_id for r in [*old, *t2, *bt2]]
            or training.label_digest(old) != inherited.TRAINING_LABEL_SHA256
            or receipt["old_g3_label_sha256"] != inherited.TRAINING_LABEL_SHA256
            or receipt["t2"]["annotation_csv_sha256"] != training.t2.T2_ANNOTATIONS_SHA256
            or receipt["t2"]["snapshot_manifest_sha256"] != training.t2.T2_MANIFEST_SHA256
            or receipt["t2"]["box_count"] != 223
            or len(receipt["t2"]["excluded_labels"]) != 2
            or receipt["bt2"]["annotation_csv_sha256"] != training.BT2_ANNOTATIONS_SHA256
            or receipt["bt2"]["box_count"] != 321
            or receipt.get("rounds_2_3_4_read") is not False
            or receipt.get("new_blind_session_labels_read") is not False):
        raise ValueError("Regional G3-T3 inventory lost its exact scored/training sources/order/exclusions")
    if {r.run.casefold() for r in regions} & HELD_OUT_RUNS:
        raise ValueError("A regression-test run entered regional G3-T3 training")
    if (any(r.phase_class != "aqueous" or r.run.casefold() not in training.t2.T2_RUNS for r in t2)
            or set(bundle["new_region_ids"]) != {r.region_id for r in t2}
            or any(r.phase_class not in ("aqueous", "gas") or r.run.casefold() not in training.BT2_RUNS for r in bt2)
            or sum(r.phase_class == "gas" for r in bt2) != 140
            or sum(r.phase_class == "aqueous" for r in bt2) != 181
            or set(bundle["bt2_region_ids"]) != {r.region_id for r in bt2}):
        raise ValueError("G3-T3 additions differ from 223 T2 negatives plus 140/181 scored B-T2 boxes")
    logistic.assert_safe_training_regions(regions)
    combined = training.label_digest(regions)
    if (combined != receipt["combined_label_sha256"]
            or combined != pixel_model["training_provenance"]["combined_label_sha256"]):
        raise ValueError("Regional labels differ from the fitted G3-T3 pixel inventory")
    return dict(training_label_sha256=combined, original_g3_label_sha256=inherited.TRAINING_LABEL_SHA256,
        training_region_ids_sha256=training.digest([r.region_id for r in regions]),
        region_count=len(regions), original_g3_regions=len(old), original_round1_regions=871,
        t2_regions=len(t2), bt2_regions=len(bt2), bt2_positive_regions=140, bt2_negative_regions=181,
        unmarked_pixels_are_negatives=False, rounds_2_3_4_read=False, new_blind_session_labels_read=False,
        held_out_runs=sorted(HELD_OUT_RUNS), t2_snapshot=receipt["t2"], bt2_snapshot=receipt["bt2"],
        coming_blind_runs=["ac26", "ac55"],
        old_inventory_coming_blind_counts={run: sum(r.run.casefold() == run for r in old) for run in ("ac26", "ac55")})


def _source_worker(task):
    from .tools.refit_ffac_g3_downstream import below_normal_priority
    below_normal_priority()
    return inherited._source_worker(task, evaluator_factory=inputs.source_evaluator)


def source_task_metadata(tasks,binding,region_count):
    """Serialize real RunSource Path fields without touching any frame pixels."""
    return training.safe(dict(binding=binding,source_groups=len(tasks),region_count=region_count,
        tasks=[dict(source=asdict(task[0]),regions=[asdict(r) for r in task[1]],
            owner_start=task[2],binding=task[3],canonical_folder=task[4],threads=task[5]) for task in tasks],
        cpu_only=True,extraction_performed=False,fitting_performed=False))


def extract_regional_training(frozen_candidate, pixel_model_path, source_bundle, output, *,
                              pixel_model_sha256, legacy_context_candidate,
                              workers=24, threads=1, plan_only=False,
                              memory_per_worker_mb=6144, memory_headroom_mb=2048):
    """Extract exact sources once; native points and majority fallback inherited."""
    root, candidate, pixel_path = map(Path, (output, frozen_candidate, pixel_model_path))
    root.mkdir(parents=True, exist_ok=True)
    pixel = inputs.verify_pixel_model(pixel_path, pixel_model_sha256)
    inventory = inventory_contract(source_bundle, pixel)
    groups = source_bundle["groups"]
    regions = [r for _source, group in groups for r in group]
    if not groups or not 1 <= int(threads) <= 16:
        raise ValueError("Regional extraction requires sources and bounded CPU threads")
    if sha256_file(candidate / "models/generic/co2_presence_signal.json") != inherited.CO2_SHA256:
        raise ValueError("Regional G3-T3 changed the frozen CO2 pixel head")
    frozen = inherited._read(candidate / "regional_model.json")
    legacy = inherited._read(Path(legacy_context_candidate) / "regional_model.json")
    from .phase_g3_probability_layers import _validate_refit
    _validate_refit(legacy, frozen)
    from . import phase_g3_probability_layers as original_inputs
    from .tools import train_phase_optical_regions as sampling
    binding = dict(format=FORMAT, frame_evaluator="g3-t3-source-exact-v1",
        frozen_candidate=str(candidate.resolve()), legacy_context_candidate=str(Path(legacy_context_candidate).resolve()),
        g3_model_path=str(pixel_path.resolve()), g3_model_sha256=pixel_model_sha256,
        co2_head_sha256=inherited.CO2_SHA256,
        frozen_regional_sha256=sha256_file(candidate / "regional_model.json"),
        legacy_regional_sha256=sha256_file(Path(legacy_context_candidate) / "regional_model.json"),
        source_inventory_sha256=training.digest(source_bundle["receipt"]), inventory=inventory,
        implementation={"extension": sha256_file(Path(__file__)),
            "inherited_refit": sha256_file(Path(inherited.__file__)),
            "source_inputs": sha256_file(Path(inputs.__file__)),
            "shared_source_inputs": sha256_file(Path(shared_inputs.__file__)),
            "legacy_evaluator": sha256_file(Path(original_inputs.__file__)),
            "regional": sha256_file(Path(regional.__file__)),
            "sampling": sha256_file(Path(sampling.__file__)),
            "source_inventory": sha256_file(Path(training.__file__)),
            "logistic": sha256_file(Path(logistic.__file__))},
        feature_names=list(regional.FEATURE_NAMES),
        feature_policy="18-frozen-source-context-columns-plus-two-G3-T3-gas-joint-means",
        sampling_policy="native-points-max512-original-ID-seed-with-frozen-majority-cell-fallback-v2",
        invalid_g3_policy="zero-for-masked-pooling-only;never-training-support;no-old-head-fallback",
        source_policy="legacy-original-native-revisions;T2-and-scored-BT2-source-exact-full-rig-scientific;no-timeline-remapping")
    inherited._write(root / "extraction_binding.json", binding)
    inherited._write(root / "source_preflight.json", source_bundle["receipt"])
    inherited._write(root / "regions.json", [asdict(r) for r in regions])
    receipt_path = root / "extraction_receipt.json"
    if receipt_path.is_file():
        return inherited._verify_sample_receipt(receipt_path, binding)
    tasks, offset = [], 0
    for source, group in groups:
        local = dict(binding, native_manifest_path=str(source.asset_manifest_path),
            native_manifest_sha256=sha256_file(source.asset_manifest_path),
            source_region_ids_sha256=training.digest([r.region_id for r in group]), owner_start=offset)
        folder = root / "source_samples" / f"{offset:04d}-{source.run}-{training.digest(local)[:16]}"
        tasks.append((source, group, offset, local, str(folder), int(threads)))
        offset += len(group)
    if plan_only:
        staged = source_task_metadata(tasks,binding,len(regions))
        inherited._write(root / "source_task_plan.json", staged)
        return staged
    import os, psutil
    from .tools.distribute_ffac_t2_l1zone_candidate import resource_admission
    process = psutil.Process()
    cpus = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    admission = resource_admission(int(workers), len(tasks), memory_per_worker_mb,
        memory_headroom_mb, logical_cpus=cpus, available_bytes=psutil.virtual_memory().available)
    reports, started = {}, time.perf_counter()
    inherited._progress(root / "progress.json", dict(stage="regional_extraction", completed=0,
        source_groups=len(tasks), workers=admission["admitted_workers"], admission=admission, cpu_only=True))
    with ProcessPoolExecutor(max_workers=admission["admitted_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_source_worker, task): i for i, task in enumerate(tasks)}
        for future in as_completed(futures):
            reports[futures[future]] = future.result()
            inherited._progress(root / "progress.json", dict(stage="regional_extraction",
                completed=len(reports), source_groups=len(tasks), seconds=time.perf_counter()-started))
    matrices, owner_arrays, coverage, receipts = [], [], [], []
    for i, task in enumerate(tasks):
        report = reports[i]
        with np.load(report["samples"]["path"], allow_pickle=False) as archive:
            matrices.append(archive["features"])
            owner_arrays.append(archive["owners"])
        coverage.extend(report["coverage"])
        path = Path(task[4]) / "source_receipt.json"
        receipts.append(dict(path=str(path), sha256=sha256_file(path)))
    matrix, owners = np.concatenate(matrices), np.concatenate(owner_arrays)
    if (matrix.dtype != np.float32 or matrix.shape != (len(owners), 20)
            or not np.isfinite(matrix).all() or owners.dtype != np.int32
            or (owners < 0).any() or (owners >= len(regions)).any()):
        raise ValueError("Regional G3-T3 lost finite original20-column rows or exact owners")
    support = inherited.supervision_completeness(regions, owners, coverage)
    if support["missing_positive_weight_region_ids"]:
        inherited._write(root / "missing_region_support.json", dict(coverage=coverage, **support))
        raise ValueError("Regional G3-T3 has positive-weight regions without supported samples")
    sample_path = root / "training_samples.npz"
    temporary = sample_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, features=matrix, owners=owners)
    temporary.replace(sample_path)
    receipt = dict(binding=binding, samples=dict(path=str(sample_path), sha256=sha256_file(sample_path)),
        regions_sha256=sha256_file(root / "regions.json"), coverage=coverage,
        source_receipts=receipts, **support, rows=len(owners), region_count=len(regions),
        training_counts=training.training_counts(regions, owners, source_bundle["new_region_ids"], source_bundle["bt2_region_ids"]),
        supervision_weights=training.supervision_weight_report(matrix, owners, regions, source_bundle["new_region_ids"], source_bundle["bt2_region_ids"]),
        extraction_seconds=time.perf_counter()-started, cpu_only=True,
        rounds_2_3_4_read=False, new_blind_session_labels_read=False)
    inherited._write(receipt_path, receipt)
    return receipt


def fit_regional_g3_t3(frozen_candidate, pixel_model_path, source_bundle, output, *,
                        pixel_model_sha256, legacy_context_candidate,
                        workers=24, threads=1, fit_threads=6):
    """One unchanged MIL objective; replace only gas_within_co2 after convergence."""
    from threadpoolctl import threadpool_limits
    root, candidate = Path(output), Path(frozen_candidate)
    started = time.perf_counter()
    extraction_root = root / "extraction"
    extraction = extract_regional_training(candidate, pixel_model_path, source_bundle, extraction_root,
        pixel_model_sha256=pixel_model_sha256, legacy_context_candidate=legacy_context_candidate,
        workers=workers, threads=threads)
    frozen = inherited._read(candidate / "regional_model.json")
    settings = dict(regularization=float(frozen["heads"][HEAD]["regularization"]),
        weak_strength=.35, positive_top_fraction=.10, max_iter=15000, tolerance=2e-5, backend="cpu")
    if not 1 <= int(fit_threads) <= 32:
        raise ValueError("Regional G3-T3 fitting requires bounded CPU threads")
    binding = dict(extraction_binding=extraction["binding"],
        extraction_receipt_sha256=sha256_file(extraction_root / "extraction_receipt.json"),
        fitting=settings, original_regional_fit_report_inherited=True,
        refit_objective="gas-only-region-MIL;CO2-head-not-part-of-optimisation")
    inherited._write(root / "refit_binding.json", binding)
    if (root / "regional_refit_receipt.json").is_file():
        return inherited._verify_finished_refit(root, binding, frozen)
    regions = [logistic.ReviewedRegion(**r) for r in inherited._read(extraction_root / "regions.json")]
    if (sha256_file(extraction_root / "regions.json") != extraction["regions_sha256"]
            or sha256_file(Path(extraction["samples"]["path"])) != extraction["samples"]["sha256"]):
        raise ValueError("Regional G3-T3 fit labels or samples changed")
    with np.load(extraction["samples"]["path"], allow_pickle=False) as archive:
        matrix, owners = archive["features"], archive["owners"]
    report_path = root / "regional_gas_training_report.json"
    cv2.setNumThreads(int(fit_threads))
    inherited._progress(root / "progress.json", dict(stage="regional_gas_fit", rows=len(owners),
        regions=len(regions), threads=int(fit_threads), cpu_only=True))
    if report_path.is_file():
        report = inherited._read(report_path)
        if report.get("binding") != binding:
            raise ValueError("Regional G3-T3 fit checkpoint lost its source binding")
        payload = dict(report["fit"])
        for key in ("mean", "scale", "coefficients"):
            payload[key] = np.asarray(payload[key], np.float64)
        fit = logistic.LogisticFit(**payload)
    else:
        fit_started = time.perf_counter()
        fit_cpu_started = time.process_time()
        with threadpool_limits(limits=int(fit_threads)):
            fit, diagnostics = logistic.fit_weighted_logistic_mil(matrix, owners, regions,
                                                                  head=HEAD, runs=None, **settings)
        report = dict(binding=binding, fit=inherited._json_safe(asdict(fit)), diagnostics=diagnostics,
            fit_seconds=time.perf_counter()-fit_started, fit_cpu_seconds=time.process_time()-fit_cpu_started, training_counts=extraction["training_counts"],
            supervision_weights=extraction["supervision_weights"],
            training_label_sha256=extraction["binding"]["inventory"]["training_label_sha256"],
            region_count=len(regions), original_round1_regions=871, t2_regions=223, bt2_regions=321, cpu_only=True)
        inherited._write(report_path, report)
    if not fit.converged:
        raise ValueError("Regional G3-T3 did not converge; no regional model exported")
    model = inherited.replace_only_gas_head(frozen, fit)
    model_path = root / "regional_model.json"
    inherited._write(model_path, model)
    receipt = dict(format=FORMAT, binding=binding, regional_model_path=str(model_path),
        regional_model_sha256=sha256_file(model_path), frozen_regional_model_sha256=sha256_file(candidate / "regional_model.json"),
        regional_co2_sha256=inherited._digest(frozen["heads"]["co2_presence"]),
        g3_t3_pixel_sha256=pixel_model_sha256, co2_pixel_sha256=inherited.CO2_SHA256,
        changed_head=HEAD, all_other_regional_fields_exact=True,
        training_label_sha256=report["training_label_sha256"], training_counts=extraction["training_counts"],
        training_report_path=str(report_path), training_report_sha256=sha256_file(report_path),
        converged=True, cpu_only=True, rounds_2_3_4_read=False, new_blind_session_labels_read=False,
        final_maps_generated=False, studio_candidate_created=False,
        fit_seconds=report["fit_seconds"], fit_cpu_seconds=report["fit_cpu_seconds"], total_seconds=time.perf_counter()-started)
    inherited._write(root / "regional_refit_receipt.json", receipt)
    inherited._progress(root / "progress.json", dict(stage="regional_refit_complete", **receipt))
    return receipt


__all__ = ["FORMAT", "inventory_contract", "extract_regional_training", "fit_regional_g3_t3"]
