"""One all-run T4b regional gas MIL fit with frozen G3 epoch weight balance."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_g3_regional_refit as inherited
from . import phase_gas_t4_regional_refit as t4
from . import phase_gas_t4_regional_inputs as t4_inputs
from . import phase_gas_t4b_regional_inputs as inputs
from . import phase_gas_t4_training as training
from . import phase_optical_regions as regional
from .phase_training import logistic
from .project import sha256_file

FORMAT = "ffac-g3-t4b-joint-regional-gas-only-refit-v1"
HEAD = inherited.HEAD
BASE_REGIONAL_SHA256 = "d33b3eea47392268d3ddebe2d9e9c695bfc640eb6e252d83670a3eaf3e7428be"
compact_candidate_source_plan = t4.compact_candidate_source_plan
source_task_metadata = t4.source_task_metadata
minimum_data_gate = t4.minimum_data_gate
_weight_report = t4._weight_report


def inventory_contract(bundle, pixel):
    regions = [r for _source, group in bundle["groups"] for r in group]
    provenance = pixel["training_provenance"]
    runs = sorted({r.run.casefold() for r in regions})
    if (provenance["combined_label_sha256"] != training.label_digest(regions)
            or sorted(provenance["training_runs"]) != runs
            or provenance["light_epochs"]["sha256"] != bundle["light_epochs"]["receipt"]["sha256"]):
        raise ValueError("T4b regional labels differ from the all-run joint pixel inventory")
    # Reuse the complete T4 source/order/exclusion guards. These metadata views
    # select no features and represent no additional fitted pixel heads.
    epochs = {}
    for epoch in bundle["light_epochs"]["epochs"]:
        own = t4.epoch_regions(bundle, epoch)
        view = deepcopy(pixel)
        view["light_epoch_id"] = epoch
        view["training_provenance"].update(epoch_label_sha256=training.label_digest(own),
                                          training_runs=sorted({r.run.casefold() for r in own}))
        epochs[epoch] = t4.inventory_contract(bundle, view, epoch)
    return dict(training_label_sha256=training.label_digest(regions),
        training_region_ids_sha256=training.digest([r.region_id for r in regions]),
        training_runs=runs, region_count=len(regions), original_g3_regions=2849,
        original_round1_regions=871, lcap_regions=156, minimum_data_gate=minimum_data_gate(bundle),
        epoch_inventories=epochs, unmarked_pixels_are_negatives=False,
        rounds_2_3_4_read=False, test_labels_read=False, t2_bt2_bt3_labels_read=False,
        lcap_snapshot=bundle["receipt"]["lcap"])


def direct_weight_ratio(features, owners, regions):
    _x, target, weight, _selected = logistic.balanced_pixel_view(features, owners, regions, head=HEAD)
    positive, negative = float(weight[target == 1].sum()), float(weight[target == 0].sum())
    if positive <= 0 or negative <= 0:
        raise ValueError("Frozen regional G3 needs both direct positive and negative weight")
    return dict(base_ratio=positive / negative, positive_direct_weight=positive,
                negative_direct_weight=negative, direct_sample_rows=len(weight))


def load_base_regional_ratio(base_regional_refit, source_bundle):
    root = Path(base_regional_refit)
    fit_path, extraction_path = root / "regional_refit_receipt.json", root / "extraction/extraction_receipt.json"
    fit, extraction = inherited._read(fit_path), inherited._read(extraction_path)
    regions_path = root / "extraction/regions.json"
    regions = [logistic.ReviewedRegion(**r) for r in inherited._read(regions_path)]
    if (sha256_file(root / "regional_model.json") != BASE_REGIONAL_SHA256
            or fit["regional_model_sha256"] != BASE_REGIONAL_SHA256 or fit.get("converged") is not True
            or sha256_file(extraction_path) != fit["binding"]["extraction_receipt_sha256"]
            or sha256_file(regions_path) != extraction["regions_sha256"]
            or extraction["binding"]["g3_model_sha256"] != inherited.G3_SHA256
            or training.label_digest(regions) != inherited.TRAINING_LABEL_SHA256
            or training.label_digest(source_bundle["old_regions"]) != inherited.TRAINING_LABEL_SHA256
            or [r.region_id for r in regions] != [r.region_id for r in source_bundle["old_regions"]]):
        raise ValueError("T4b R_reg is not bound to the exact frozen d33b3eea G3 refit")
    sample = extraction["samples"]
    if sha256_file(Path(sample["path"])) != sample["sha256"]:
        raise ValueError("Frozen regional G3 samples changed before R_reg recomputation")
    with np.load(sample["path"], allow_pickle=False) as archive:
        ratio = direct_weight_ratio(archive["features"], archive["owners"], regions)
    return dict(**ratio, base_regional_refit=str(root.resolve()),
        base_regional_sha256=BASE_REGIONAL_SHA256, base_g3_model_path=extraction["binding"]["g3_model_path"],
        base_fit_receipt_path=str(fit_path.resolve()), base_fit_receipt_sha256=sha256_file(fit_path),
        base_extraction_receipt_path=str(extraction_path.resolve()), base_extraction_receipt_sha256=sha256_file(extraction_path),
        base_training_label_sha256=inherited.TRAINING_LABEL_SHA256,
        base_samples=sample, weighting_rule="unchanged-all-run-balanced_pixel_view;direct-loss-only")


def _source_worker(task):
    return inherited._source_worker(task, evaluator_factory=inputs.source_evaluator)


def _context_plan_path(compact, roots):
    for root in roots:
        for path in Path(root).glob("source_context/*.json"):
            if inherited._read(path) == compact:
                return path.resolve()
    return None


def extract_regional_training(frozen_candidate, pixel_model_path, source_bundle, output, *,
                              pixel_model_sha256, legacy_context_candidate, base_regional_refit, source_context_roots=(),
                              workers=1, threads=1, plan_only=False,
                              memory_per_worker_mb=6144, memory_headroom_mb=2048):
    root, candidate, pixel_path = map(Path, (output, frozen_candidate, pixel_model_path))
    root.mkdir(parents=True, exist_ok=True)
    pixel = inputs.verify_pixel_model(pixel_path, pixel_model_sha256)
    inventory = inventory_contract(source_bundle, pixel)
    regions = [r for _source, group in source_bundle["groups"] for r in group]
    groups = source_bundle["groups"]
    base_ratio = load_base_regional_ratio(base_regional_refit, source_bundle)
    if not groups or not 1 <= int(threads) <= 16:
        raise ValueError("Regional T4b extraction requires sources and bounded CPU threads")
    if sha256_file(candidate / "models/generic/co2_presence_signal.json") != inherited.CO2_SHA256:
        raise ValueError("Regional T4b changed the frozen common CO2 pixel head")
    frozen = inherited._read(candidate / "regional_model.json")
    legacy = inherited._read(Path(legacy_context_candidate) / "regional_model.json")
    from .phase_g3_probability_layers import _validate_refit
    _validate_refit(legacy, frozen)
    from . import phase_gas_t2_regional_inputs as shared_inputs
    from . import phase_g3_probability_layers as original_inputs
    from .tools import train_phase_optical_regions as sampling
    table_receipt = source_bundle["light_epochs"]["receipt"]
    binding = dict(format=FORMAT, frame_evaluator="g3-t4b-source-exact-v1", 
        light_epoch_table_sha256=table_receipt["sha256"], light_epoch_table_path=table_receipt["path"],
        base_regional_ratio=base_ratio, base_g3_model_path=base_ratio["base_g3_model_path"],
        frozen_candidate=str(candidate.resolve()), legacy_context_candidate=str(Path(legacy_context_candidate).resolve()),
        g3_model_path=str(pixel_path.resolve()), g3_model_sha256=pixel_model_sha256,
        co2_head_sha256=inherited.CO2_SHA256, frozen_regional_sha256=sha256_file(candidate / "regional_model.json"),
        legacy_regional_sha256=sha256_file(Path(legacy_context_candidate) / "regional_model.json"),
        source_inventory_sha256=training.digest(source_bundle["receipt"]), inventory=inventory,
        implementation={"extension": sha256_file(Path(__file__)), "inherited_refit": sha256_file(Path(inherited.__file__)),
            "source_inputs": sha256_file(Path(inputs.__file__)), "shared_source_inputs": sha256_file(Path(shared_inputs.__file__)),
            "legacy_evaluator": sha256_file(Path(original_inputs.__file__)), "regional": sha256_file(Path(regional.__file__)),
            "sampling": sha256_file(Path(sampling.__file__)), "source_inventory": sha256_file(Path(training.__file__)),
            "logistic": sha256_file(Path(logistic.__file__))},
        feature_names=list(regional.FEATURE_NAMES), feature_policy="18-frozen-source-context-columns-plus-two-joint-T4b-gas-means;no-epoch-feature",
        sampling_policy="native-points-max512-original-ID-seed-with-frozen-majority-cell-fallback-v2",
        invalid_g3_policy="zero-for-masked-pooling-only;never-training-support;no-old-head-fallback",
        source_policy="legacy-original-native-revisions;Lcap-source-exact-full-rig;no-timeline-remapping")
    inherited._write(root / "extraction_binding.json", binding)
    inherited._write(root / "source_preflight.json", source_bundle["receipt"])
    inherited._write(root / "regions.json", [asdict(r) for r in regions])
    receipt_path = root / "extraction_receipt.json"
    if receipt_path.is_file():
        return inherited._verify_sample_receipt(receipt_path, binding)
    from .phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    tasks, offset = [], 0
    for source, group in groups:
        local = dict(binding, native_manifest_path=str(source.asset_manifest_path),
            native_manifest_sha256=sha256_file(source.asset_manifest_path),
            source_region_ids_sha256=training.digest([r.region_id for r in group]), owner_start=offset)
        native = inherited._read(source.asset_manifest_path)
        if (native.get(SOURCE_KEY) or {}).get("contract") == SOURCE_CONTRACT:
            compact = compact_candidate_source_plan(candidate, source.run, {r.frame_index for r in group})
            path = _context_plan_path(compact, source_context_roots)
            if path is None:
                path = root / "source_context" / f"{source.run}-{training.digest(compact)[:16]}.json"
                inherited._write(path, compact)
            local.update(source_context_plan_path=str(path.resolve()), source_context_plan_sha256=sha256_file(path))
        folder = root / "source_samples" / f"{offset:04d}-{source.run}-{training.digest(local)[:16]}"
        tasks.append((source, group, offset, local, str(folder), int(threads)))
        offset += len(group)
    if plan_only:
        staged = source_task_metadata(tasks, binding, len(regions))
        inherited._write(root / "source_task_plan.json", staged)
        return staged
    import os, psutil
    from .tools.distribute_ffac_t2_l1zone_candidate import resource_admission
    process = psutil.Process()
    cpus = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    admission = resource_admission(int(workers), len(tasks), memory_per_worker_mb,
        memory_headroom_mb, logical_cpus=max(1, cpus // int(threads)), available_bytes=psutil.virtual_memory().available)
    admission["threads_per_worker"] = int(threads)
    reports, started = {}, time.perf_counter()
    inherited._progress(root / "progress.json", dict(stage="regional_extraction", 
        completed=0, source_groups=len(tasks), workers=admission["admitted_workers"], admission=admission, cpu_only=True))
    if admission["admitted_workers"] == 1:
        for i, task in enumerate(tasks):
            reports[i] = _source_worker(task)
            inherited._progress(root / "progress.json", dict(stage="regional_extraction", 
                completed=len(reports), source_groups=len(tasks), seconds=time.perf_counter()-started))
    else:
        with ProcessPoolExecutor(max_workers=admission["admitted_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
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
    if (matrix.dtype != np.float32 or matrix.shape != (len(owners), 20) or not np.isfinite(matrix).all()
            or owners.dtype != np.int32 or (owners < 0).any() or (owners >= len(regions)).any()):
        raise ValueError("Regional T4b lost finite original 20-column rows or exact joint owners")
    support = inherited.supervision_completeness(regions, owners, coverage)
    if support["missing_positive_weight_region_ids"]:
        inherited._write(root / "missing_region_support.json", dict(coverage=coverage, **support))
        raise ValueError("Regional T4b has positive-weight regions without supported samples")
    sample_path = root / "training_samples.npz"
    temporary = sample_path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, features=matrix, owners=owners)
    temporary.replace(sample_path)
    receipt = dict(binding=binding, samples=dict(path=str(sample_path), sha256=sha256_file(sample_path)),
        regions_sha256=sha256_file(root / "regions.json"), coverage=coverage, source_receipts=receipts,
        **support, rows=len(owners), region_count=len(regions),
        training_counts=inherited._training_counts(regions, owners),
        supervision_weights=_weight_report(matrix, owners, regions, source_bundle["new_region_ids"]),
        extraction_seconds=time.perf_counter()-started, cpu_only=True,
        rounds_2_3_4_read=False, test_labels_read=False, t2_bt2_bt3_labels_read=False)
    inherited._write(receipt_path, receipt)
    return receipt


def fit_regional_g3_t4b(frozen_candidate, pixel_model_path, source_bundle, output, *,
                       pixel_model_sha256, legacy_context_candidate, base_regional_refit,
                       workers=1, threads=1, fit_threads=2, source_context_roots=(),
                       memory_per_worker_mb=6144, memory_headroom_mb=2048):
    from threadpoolctl import threadpool_limits
    from . import phase_gas_t4b_training as joint
    if not 1 <= int(fit_threads) <= 32:
        raise ValueError("T4b regional fit needs bounded CPU threads")
    root, candidate = Path(output), Path(frozen_candidate)
    started = time.perf_counter()
    extraction_root = root / "extraction"
    extraction = extract_regional_training(candidate, pixel_model_path, source_bundle, extraction_root,
        pixel_model_sha256=pixel_model_sha256, legacy_context_candidate=legacy_context_candidate,
        base_regional_refit=base_regional_refit, source_context_roots=source_context_roots,
        workers=workers, threads=threads, memory_per_worker_mb=memory_per_worker_mb,
        memory_headroom_mb=memory_headroom_mb)
    frozen = inherited._read(candidate / "regional_model.json")
    settings = dict(regularization=float(frozen["heads"][HEAD]["regularization"]), weak_strength=.35,
                   positive_top_fraction=.10, max_iter=15000, tolerance=2e-5, backend="cpu")
    binding = dict(extraction_binding=extraction["binding"],
        extraction_receipt_sha256=sha256_file(extraction_root / "extraction_receipt.json"), fitting=settings,
        joint_weighting_sha256=sha256_file(Path(joint.__file__)),
        refit_objective="one-all-run-gas-MIL;epoch-positive-balance;no-epoch-regional-input",
        original_regional_fit_report_inherited=True)
    inherited._write(root / "refit_binding.json", binding)
    if (root / "regional_refit_receipt.json").is_file():
        return inherited._verify_finished_refit(root, binding, frozen)
    regions_path = extraction_root / "regions.json"
    regions = [logistic.ReviewedRegion(**r) for r in inherited._read(regions_path)]
    if (sha256_file(regions_path) != extraction["regions_sha256"]
            or sha256_file(Path(extraction["samples"]["path"])) != extraction["samples"]["sha256"]
            or training.label_digest(regions) != extraction["binding"]["inventory"]["training_label_sha256"]):
        raise ValueError("T4b joint regional labels or samples changed")
    with np.load(extraction["samples"]["path"], allow_pickle=False) as archive:
        matrix, owners = archive["features"], archive["owners"]
    if matrix.shape != (len(owners), 20):
        raise ValueError("T4b regional features must contain no epoch indicator")
    report_path = root / "regional_gas_training_report.json"
    cv2.setNumThreads(int(fit_threads))
    inherited._progress(root / "progress.json", dict(stage="joint_regional_gas_fit", rows=len(owners),
        regions=len(regions), threads=int(fit_threads), cpu_only=True))
    if report_path.is_file():
        report = inherited._read(report_path)
        if report.get("binding") != binding:
            raise ValueError("T4b regional fit checkpoint changed its source binding")
        payload = dict(report["fit"])
        for key in ("mean", "scale", "coefficients"):
            payload[key] = np.asarray(payload[key], np.float64)
        fit = logistic.LogisticFit(**payload)
    else:
        fit_started, cpu_started = time.perf_counter(), time.process_time()
        ratio = extraction["binding"]["base_regional_ratio"]
        with threadpool_limits(limits=int(fit_threads)):
            fit, diagnostics, balancing = joint.fit_balanced_mil(matrix, owners, regions,
                source_bundle["light_epochs"], ratio["base_ratio"], head=HEAD,
                round1_ids=source_bundle["receipt"]["old_source_bundle"]["new_region_ids"],
                lcap_ids=source_bundle["new_region_ids"], **settings)
        report = dict(binding=binding, fit=inherited._json_safe(asdict(fit)), diagnostics=diagnostics,
            balancing=balancing, R_reg=ratio["base_ratio"], base_ratio_proof=ratio,
            base_gas_head=frozen["heads"][HEAD], regional_epoch_input=False,
            fit_seconds=time.perf_counter()-fit_started, fit_cpu_seconds=time.process_time()-cpu_started,
            training_counts=extraction["training_counts"], supervision_weights=extraction["supervision_weights"],
            training_label_sha256=extraction["binding"]["inventory"]["training_label_sha256"],
            region_count=len(regions), cpu_only=True)
        inherited._write(report_path, report)
    if not fit.converged:
        raise ValueError("T4b regional MIL did not converge; no model exported")
    model_path = root / "regional_model.json"
    inherited._write(model_path, inherited.replace_only_gas_head(frozen, fit))
    receipt = dict(format=FORMAT, binding=binding,
        regional_model_path=str(model_path.resolve()), regional_model_sha256=sha256_file(model_path),
        frozen_regional_model_sha256=sha256_file(candidate / "regional_model.json"),
        regional_co2_sha256=inherited._digest(frozen["heads"]["co2_presence"]),
        g3_t4b_pixel_sha256=pixel_model_sha256, co2_pixel_sha256=inherited.CO2_SHA256,
        changed_head=HEAD, all_other_regional_fields_exact=True, regional_epoch_input=False,
        training_label_sha256=report["training_label_sha256"], training_counts=extraction["training_counts"],
        R_reg=report["R_reg"], balancing=report["balancing"],
        training_report_path=str(report_path.resolve()), training_report_sha256=sha256_file(report_path),
        converged=True, cpu_only=True, rounds_2_3_4_read=False, test_labels_read=False,
        t2_bt2_bt3_labels_read=False, sealed_scorer_read=False,
        final_maps_generated=False, studio_candidate_created=False,
        fit_seconds=report["fit_seconds"], fit_cpu_seconds=report["fit_cpu_seconds"], total_seconds=time.perf_counter()-started)
    inherited._write(root / "regional_refit_receipt.json", receipt)
    inherited._progress(root / "progress.json", dict(stage="joint_regional_refit_complete", **receipt))
    return receipt


def assemble_joint_bundle(frozen_candidate, pixel_fit_receipt, source_bundle, output):
    root, parent, pixel_path = Path(output), Path(frozen_candidate), Path(pixel_fit_receipt)
    pixel, receipt = inherited._read(pixel_path), inherited._read(root / "regional_refit_receipt.json")
    model = inputs.verify_pixel_model(pixel["model_path"], pixel["model_sha256"])
    inventory_contract(source_bundle, model)
    if (pixel.get("converged") is not True or pixel.get("cpu_only") is not True
            or receipt.get("converged") is not True
            or receipt["g3_t4b_pixel_sha256"] != pixel["model_sha256"]):
        raise ValueError("T4b bundle requires completed joint pixel and regional fits")
    table = source_bundle["light_epochs"]["receipt"]
    bundle = dict(contract=inputs.BUNDLE_CONTRACT, converged=True, cpu_only=True,
        frozen_candidate=str(parent.resolve()), frozen_regional_sha256=sha256_file(parent / "regional_model.json"),
        common_frozen_co2_head_sha256=inherited.CO2_SHA256,
        light_epoch_table_path=table["path"], light_epoch_table_sha256=table["sha256"],
        pixel_fit_receipt_path=str(pixel_path.resolve()), pixel_fit_receipt_sha256=sha256_file(pixel_path),
        pixel_model_path=pixel["model_path"], pixel_model_sha256=pixel["model_sha256"],
        regional_model_path=receipt["regional_model_path"], regional_model_sha256=receipt["regional_model_sha256"],
        regional_fit_receipt_path=str((root / "regional_refit_receipt.json").resolve()),
        regional_fit_receipt_sha256=sha256_file(root / "regional_refit_receipt.json"),
        training_runs=receipt["binding"]["extraction_binding"]["inventory"]["training_runs"],
        training_label_sha256=receipt["training_label_sha256"], R_reg=receipt["R_reg"],
        regional_epoch_input=False, t2_bt2_bt3_labels_read=False, rounds_2_3_4_read=False,
        test_labels_read=False, sealed_scorer_read=False)
    inputs.verify_regional_model(bundle["regional_model_path"], bundle["regional_model_sha256"], parent)
    path = root / "regional_joint_bundle.json"
    inherited._write(path, bundle)
    inputs.verify_joint_bundle(path, Path(table["path"]), parent)
    return bundle
