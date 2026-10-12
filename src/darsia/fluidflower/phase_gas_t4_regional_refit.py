"""Two independent CPU regional gas refits on the exact G3 plus L-cap set.

The native point sampler, zero-weight exemption and majority-cell fallback
are inherited directly from G3. Every source/frame cache binds its epoch's
pixel head and immutable original source revision.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np

from . import phase_g3_regional_refit as inherited
from . import phase_gas_t4_regional_inputs as inputs
from . import phase_gas_t4_training as training
from . import phase_optical_regions as regional
from .phase_gas_variant_sources import HELD_OUT_RUNS
from .phase_training import logistic
from .project import sha256_file

FORMAT = "ffac-g3-t4-epoch-regional-gas-only-refit-v1"
HEAD = inherited.HEAD
LCAP_ANNOTATIONS_SHA256 = "93645a04957468390592ccec5b9d045b3c83cd4294dc6d1b4528daa9418c7d1d"


def epoch_regions(bundle, epoch_id):
    table = bundle["light_epochs"]
    if epoch_id not in table["epochs"]:
        raise ValueError("Unknown project light epoch")
    regions = [r for _source, group in bundle["groups"] for r in group]
    if any(r.run.casefold() not in table["run_to_epoch"] for r in regions):
        raise ValueError("A training run has no declared project light epoch")
    return [r for r in regions if table["run_to_epoch"][r.run.casefold()] == epoch_id]


def minimum_data_gate(bundle):
    """Count positive-weight MIL regions, including explicit weak gas bags."""
    counts = {}
    for epoch_id in bundle["light_epochs"]["epochs"]:
        regions = epoch_regions(bundle, epoch_id)
        targets = {target: [r for r in regions if r.supervision_weight > 0
                   and logistic.regional_presence_target(r.phase_class, HEAD) == target]
                   for target in (0, 1)}
        counts[epoch_id] = dict(positive_regions=len(targets[1]), negative_regions=len(targets[0]),
            positive_runs=sorted({r.run.casefold() for r in targets[1]}),
            negative_runs=sorted({r.run.casefold() for r in targets[0]}),
            all_runs=sorted({r.run.casefold() for r in regions}), region_count=len(regions),
            training_label_sha256=training.label_digest(regions))
    # The table declares which epoch received the protocol's new lamps.
    new_epoch = "new_lamps"
    if new_epoch not in counts:
        raise ValueError("Project epoch table has no new-lamp declaration")
    admitted = counts[new_epoch]["positive_regions"] >= 100 and counts[new_epoch]["negative_regions"] >= 100
    return dict(epochs=counts, new_lamp_epoch=new_epoch, minimum_positive_regions=100,
                minimum_negative_regions=100, passed=admitted, cpu_only=True)


def inventory_contract(bundle, pixel_model, epoch_id):
    regions = [r for _source, group in bundle["groups"] for r in group]
    old, added, receipt = bundle["old_regions"], bundle["new_regions"], bundle["receipt"]
    if (len(old) != 2849 or len(added) != 156 or len(regions) != 3005
            or [r.region_id for r in regions] != [r.region_id for r in [*old, *added]]
            or len({r.region_id for r in regions}) != len(regions)
            or training.label_digest(old) != inherited.TRAINING_LABEL_SHA256
            or receipt["old_g3_label_sha256"] != inherited.TRAINING_LABEL_SHA256
            or receipt["lcap"]["annotation_csv_sha256"] != LCAP_ANNOTATIONS_SHA256
            or receipt["lcap"]["box_count"] != 156
            or any(r.phase_class != "gas" for r in added)
            or {r.region_id for r in added} != set(bundle["new_region_ids"])
            or receipt.get("rounds_2_3_4_read") is not False
            or receipt.get("test_labels_read", False) is not False
            or receipt.get("t2_bt2_bt3_labels_read") is not False
            or receipt.get("sealed_scorer_read") is not False
            or any(key in receipt for key in ("t2", "bt2", "bt3"))
            or any(r.payload.get("training_revision") in ("G3-T2", "G3-T3") for r in added)):
        raise ValueError("Regional T4 lost the exact G3 plus 156 positive L-cap sources/order/exclusions")
    if {r.run.casefold() for r in regions} & HELD_OUT_RUNS:
        raise ValueError("A regression-test run entered regional T4 training")
    logistic.assert_safe_training_regions(regions)
    combined = training.label_digest(regions)
    own = epoch_regions(bundle, epoch_id)
    own_sha = training.label_digest(own)
    provenance = pixel_model["training_provenance"]
    if (combined != receipt["combined_label_sha256"]
            or provenance["epoch_label_sha256"] != own_sha
            or pixel_model["light_epoch_id"] != epoch_id):
        raise ValueError("Regional T4 labels differ from the fitted own-epoch pixel inventory")
    gate = minimum_data_gate(bundle)
    if not gate["passed"]:
        raise ValueError("T4 new-lamp set requires at least 100 positive and 100 negative regions before fitting")
    old_ids = {r.region_id for r in old}
    if (provenance.get("training_runs") != sorted({r.run.casefold() for r in own})
            or provenance.get("light_epochs", {}).get("sha256") != bundle["light_epochs"]["receipt"]["sha256"]):
        raise ValueError("Regional T4 pixel head lost its own runs or project epoch table")
    return dict(training_label_sha256=own_sha, combined_label_sha256=combined,
        original_g3_label_sha256=inherited.TRAINING_LABEL_SHA256,
        training_region_ids_sha256=training.digest([r.region_id for r in own]),
        training_runs=sorted({r.run.casefold() for r in own}), light_epoch_id=epoch_id,
        region_count=len(own), original_g3_regions=sum(r.region_id in old_ids for r in own),
        lcap_regions=sum(r.region_id in bundle["new_region_ids"] for r in own),
        combined_region_count=3005, original_round1_regions=871,
        unmarked_pixels_are_negatives=False, rounds_2_3_4_read=False,
        test_labels_read=False, t2_bt2_bt3_labels_read=False,
        held_out_runs=sorted(HELD_OUT_RUNS), lcap_snapshot=receipt["lcap"], minimum_data_gate=gate)


def compact_source_plan(plan, run, frame_indices):
    """Select exact source and predecessor metadata without a worker-wide plan."""
    from .phase_ffac_rig_camera_runtime import bound_path
    selected_keys = {f"{run.casefold()}/frame_{int(index):03d}" for index in frame_indices}
    producer = {key: plan["source_producer_frames"][key] for key in sorted(selected_keys)}
    source_by_task = {row["scientific_task_id"]: row for row in plan["source_inventory"]}
    needed_tasks = set()
    previous = {}
    extended = {}
    for key in sorted(selected_keys):
        original = inherited._read(bound_path(producer[key]))
        source = original["binding"]["source"]
        needed_tasks.add(source["scientific_task_id"])
        task = source["previous_scientific_task_id"]
        if task is None:
            continue
        if key in plan.get("previous_extended_scientific_sources", {}):
            previous[key] = plan["previous_extended_scientific_sources"][key]
        elif task in plan["extended_scientific_sources"]:
            extended[task] = plan["extended_scientific_sources"][task]
        else:
            # Exact off-timeline descriptors need not have a campaign frame
            # in source_inventory. Only the producer-receipt fallback uses
            # that inventory to resolve the declared predecessor task.
            needed_tasks.add(task)
            row = source_by_task[task]
            previous_key = f"{row['run']}/frame_{row['frame_index']:03d}"
            producer[previous_key] = plan["source_producer_frames"][previous_key]
    return dict(source_producer_frames=producer, previous_extended_scientific_sources=previous,
        extended_scientific_sources=extended,
        source_inventory=[source_by_task[task] for task in sorted(needed_tasks)])


def compact_candidate_source_plan(candidate, run, frame_indices, *, fallback_plan=None):
    """Read small bound frame receipts, retaining each exact source predecessor.

    The seam frame already seals its original producer receipt. A predecessor
    is accepted only when that producer's scientific task ID matches the
    current source's declared predecessor. Rare off-timeline predecessors use
    the frozen plan's original lookup; they never use a nearby acquisition.
    """
    from .phase_ffac_rig_camera_runtime import bound_path
    candidate, run = Path(candidate), str(run).casefold()
    indices = sorted({int(index) for index in frame_indices})
    cache, producer, sources, seals = {}, {}, {}, {}

    def load(index):
        if index not in cache:
            path = candidate / "rig_camera_frames" / run / f"frame_{index:03d}.json"
            frame = inherited._read(path)
            asset = frame["source_producer_receipt"]
            original = inherited._read(bound_path(asset))
            if (frame.get("complete") is not True or original.get("complete") is not True
                    or original["binding"]["source"] != frame["binding"]["source"]):
                raise ValueError("T4 compact source differs from its sealed producer")
            key = f"{run}/frame_{index:03d}"
            cache[index] = original
            producer[key] = asset
            source = original["binding"]["source"]
            sources[source["scientific_task_id"]] = source
            seals[key] = dict(path=str(path.resolve()), sha256=sha256_file(path))
        return cache[index]

    for index in indices:
        original = load(index)
        source = original["binding"]["source"]
        task = source["previous_scientific_task_id"]
        if task is None:
            continue
        previous_index = (source.get("temporal_predecessor") or {}).get("frame_index")
        if previous_index is not None:
            try:
                prior = load(int(previous_index))["binding"]["source"]
            except FileNotFoundError:
                prior = None
            if prior is not None and prior["scientific_task_id"] == task:
                continue
        plan = fallback_plan if fallback_plan is not None else inherited._read(candidate / "old_domain_seam_plan.json")
        compact = compact_source_plan(plan, run, indices)
        compact["compact_source_provenance"] = dict(policy="frozen-plan-exact-off-timeline-predecessor",
            full_plan_sha256=sha256_file(candidate / "old_domain_seam_plan.json"))
        return compact
    return dict(source_producer_frames=producer, previous_extended_scientific_sources={},
        extended_scientific_sources={}, source_inventory=[sources[task] for task in sorted(sources)],
        compact_source_provenance=dict(policy="sealed-frame-producers-and-exact-source-task-predecessors",
                                       parent_frame_receipts=seals))


def _source_worker(task):
    from .tools.refit_ffac_g3_downstream import below_normal_priority
    below_normal_priority()
    return inherited._source_worker(task, evaluator_factory=inputs.source_evaluator)


def source_task_metadata(tasks, binding, region_count):
    return training.safe(dict(binding=binding, source_groups=len(tasks), region_count=region_count,
        tasks=[dict(source=asdict(task[0]), regions=[asdict(r) for r in task[1]], owner_start=task[2],
                    binding=task[3], canonical_folder=task[4], threads=task[5]) for task in tasks],
        cpu_only=True, extraction_performed=False, fitting_performed=False))


def _weight_report(features, owners, regions, added_ids):
    _view, _targets, weights, selected_owners = logistic.balanced_pixel_view(features, owners, regions, head=HEAD)
    loss = np.bincount(selected_owners, weights=weights, minlength=len(regions))
    rows = np.bincount(owners, minlength=len(regions))
    report = defaultdict(lambda: defaultdict(lambda: dict(regions=0, sample_rows=0, raw_box_weight=0.,
        gas_supervision_box_weight=0., balanced_direct_pixel_loss_weight=0., mil_presence_regions=0)))
    for i, region in enumerate(regions):
        source = ("lcap" if region.region_id in added_ids else "round1"
                  if region.payload.get("gas_variant_annotation_sha256") == training.ROUND1_ANNOTATIONS_SHA256
                  else "historical_g3")
        target = logistic.regional_presence_target(region.phase_class, HEAD)
        row = report[source]["positive" if target == 1 else "negative" if target == 0 else "uninformative"]
        row["regions"] += 1
        row["sample_rows"] += int(rows[i])
        row["raw_box_weight"] += region.supervision_weight
        row["gas_supervision_box_weight"] += region.supervision_weight if target is not None and rows[i] else 0.
        row["balanced_direct_pixel_loss_weight"] += float(loss[i])
        row["mil_presence_regions"] += int(logistic.pixel_target(region.phase_class, HEAD) is None
            and target is not None and region.supervision_weight > 0 and rows[i] > 0)
    return dict(by_source={source: dict(classes) for source, classes in report.items()},
        policy="confidence-times-source per box; equal (run,target) strata; uniform mass within each box")


def extract_regional_training(frozen_candidate, pixel_model_path, source_bundle, output, *,
                              pixel_model_sha256, legacy_context_candidate, light_epoch_id,
                              workers=1, threads=1, plan_only=False,
                              memory_per_worker_mb=6144, memory_headroom_mb=2048):
    root, candidate, pixel_path = map(Path, (output, frozen_candidate, pixel_model_path))
    root.mkdir(parents=True, exist_ok=True)
    pixel = inputs.verify_pixel_model(pixel_path, pixel_model_sha256, light_epoch_id=light_epoch_id)
    inventory = inventory_contract(source_bundle, pixel, light_epoch_id)
    regions = epoch_regions(source_bundle, light_epoch_id)
    groups = [(source, tuple(r for r in group if source_bundle["light_epochs"]["run_to_epoch"][r.run.casefold()] == light_epoch_id))
              for source, group in source_bundle["groups"]]
    groups = [(source, group) for source, group in groups if group]
    if not groups or not 1 <= int(threads) <= 16:
        raise ValueError("Regional T4 extraction requires sources and bounded CPU threads")
    if sha256_file(candidate / "models/generic/co2_presence_signal.json") != inherited.CO2_SHA256:
        raise ValueError("Regional T4 changed the frozen common CO2 pixel head")
    frozen = inherited._read(candidate / "regional_model.json")
    legacy = inherited._read(Path(legacy_context_candidate) / "regional_model.json")
    from .phase_g3_probability_layers import _validate_refit
    _validate_refit(legacy, frozen)
    from . import phase_gas_t2_regional_inputs as shared_inputs
    from . import phase_g3_probability_layers as original_inputs
    from .tools import train_phase_optical_regions as sampling
    table_receipt = source_bundle["light_epochs"]["receipt"]
    binding = dict(format=FORMAT, frame_evaluator="g3-t4-source-exact-v1", light_epoch_id=light_epoch_id,
        light_epoch_table_sha256=table_receipt["sha256"],
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
        feature_names=list(regional.FEATURE_NAMES), feature_policy="18-frozen-source-context-columns-plus-two-own-epoch-gas-joint-means",
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
    inherited._progress(root / "progress.json", dict(stage="regional_extraction", light_epoch_id=light_epoch_id,
        completed=0, source_groups=len(tasks), workers=admission["admitted_workers"], admission=admission, cpu_only=True))
    if admission["admitted_workers"] == 1:
        for i, task in enumerate(tasks):
            reports[i] = _source_worker(task)
            inherited._progress(root / "progress.json", dict(stage="regional_extraction", light_epoch_id=light_epoch_id,
                completed=len(reports), source_groups=len(tasks), seconds=time.perf_counter()-started))
    else:
        with ProcessPoolExecutor(max_workers=admission["admitted_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_source_worker, task): i for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                reports[futures[future]] = future.result()
                inherited._progress(root / "progress.json", dict(stage="regional_extraction", light_epoch_id=light_epoch_id,
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
        raise ValueError("Regional T4 lost finite original 20-column rows or exact epoch owners")
    support = inherited.supervision_completeness(regions, owners, coverage)
    if support["missing_positive_weight_region_ids"]:
        inherited._write(root / "missing_region_support.json", dict(coverage=coverage, **support))
        raise ValueError("Regional T4 has positive-weight regions without supported samples")
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


def fit_regional_g3_t4_epoch(frozen_candidate, pixel_model_path, source_bundle, output, *,
                           pixel_model_sha256, legacy_context_candidate, light_epoch_id,
                           workers=1, threads=1, fit_threads=2,
                           memory_per_worker_mb=6144, memory_headroom_mb=2048):
    """Fit only one epoch's regional gas head, retaining every other field."""
    from threadpoolctl import threadpool_limits
    root, candidate = Path(output), Path(frozen_candidate)
    started = time.perf_counter()
    extraction_root = root / "extraction"
    extraction = extract_regional_training(candidate, pixel_model_path, source_bundle, extraction_root,
        pixel_model_sha256=pixel_model_sha256, legacy_context_candidate=legacy_context_candidate,
        light_epoch_id=light_epoch_id, workers=workers, threads=threads,
        memory_per_worker_mb=memory_per_worker_mb, memory_headroom_mb=memory_headroom_mb)
    frozen = inherited._read(candidate / "regional_model.json")
    settings = dict(regularization=float(frozen["heads"][HEAD]["regularization"]), weak_strength=.35,
                   positive_top_fraction=.10, max_iter=15000, tolerance=2e-5, backend="cpu")
    if not 1 <= int(fit_threads) <= 32:
        raise ValueError("Regional T4 fitting requires bounded CPU threads")
    binding = dict(extraction_binding=extraction["binding"],
        extraction_receipt_sha256=sha256_file(extraction_root / "extraction_receipt.json"), fitting=settings,
        original_regional_fit_report_inherited=True, refit_objective="gas-only-own-epoch-region-MIL;CO2-frozen")
    inherited._write(root / "refit_binding.json", binding)
    if (root / "regional_refit_receipt.json").is_file():
        return inherited._verify_finished_refit(root, binding, frozen)
    regions = [logistic.ReviewedRegion(**r) for r in inherited._read(extraction_root / "regions.json")]
    if (sha256_file(extraction_root / "regions.json") != extraction["regions_sha256"]
            or sha256_file(Path(extraction["samples"]["path"])) != extraction["samples"]["sha256"]):
        raise ValueError("Regional T4 fit labels or samples changed")
    if any(source_bundle["light_epochs"]["run_to_epoch"].get(r.run.casefold()) != light_epoch_id for r in regions):
        raise ValueError("Another epoch entered the regional T4 fit")
    with np.load(extraction["samples"]["path"], allow_pickle=False) as archive:
        matrix, owners = archive["features"], archive["owners"]
    report_path = root / "regional_gas_training_report.json"
    cv2.setNumThreads(int(fit_threads))
    inherited._progress(root / "progress.json", dict(stage="regional_gas_fit", light_epoch_id=light_epoch_id,
        rows=len(owners), regions=len(regions), threads=int(fit_threads), cpu_only=True))
    if report_path.is_file():
        report = inherited._read(report_path)
        if report.get("binding") != binding:
            raise ValueError("Regional T4 fit checkpoint lost its source binding")
        payload = dict(report["fit"])
        for key in ("mean", "scale", "coefficients"):
            payload[key] = np.asarray(payload[key], np.float64)
        fit = logistic.LogisticFit(**payload)
    else:
        fit_started, cpu_started = time.perf_counter(), time.process_time()
        with threadpool_limits(limits=int(fit_threads)):
            fit, diagnostics = logistic.fit_weighted_logistic_mil(matrix, owners, regions,
                head=HEAD, runs=None, **settings)
        report = dict(binding=binding, fit=inherited._json_safe(asdict(fit)), diagnostics=diagnostics,
            base_gas_head=frozen["heads"][HEAD], light_epoch_id=light_epoch_id,
            fit_seconds=time.perf_counter()-fit_started, fit_cpu_seconds=time.process_time()-cpu_started,
            training_counts=extraction["training_counts"], supervision_weights=extraction["supervision_weights"],
            training_label_sha256=extraction["binding"]["inventory"]["training_label_sha256"],
            region_count=len(regions), cpu_only=True)
        inherited._write(report_path, report)
    if not fit.converged:
        raise ValueError("Regional T4 did not converge; no regional model exported")
    model_path = root / "regional_model.json"
    inherited._write(model_path, inherited.replace_only_gas_head(frozen, fit))
    receipt = dict(format=FORMAT, binding=binding, light_epoch_id=light_epoch_id,
        regional_model_path=str(model_path.resolve()), regional_model_sha256=sha256_file(model_path),
        frozen_regional_model_sha256=sha256_file(candidate / "regional_model.json"),
        regional_co2_sha256=inherited._digest(frozen["heads"]["co2_presence"]),
        g3_t4_pixel_sha256=pixel_model_sha256, co2_pixel_sha256=inherited.CO2_SHA256,
        changed_head=HEAD, all_other_regional_fields_exact=True,
        training_label_sha256=report["training_label_sha256"], training_counts=extraction["training_counts"],
        training_report_path=str(report_path.resolve()), training_report_sha256=sha256_file(report_path),
        converged=True, cpu_only=True, rounds_2_3_4_read=False, test_labels_read=False,
        t2_bt2_bt3_labels_read=False, final_maps_generated=False, studio_candidate_created=False,
        fit_seconds=report["fit_seconds"], fit_cpu_seconds=report["fit_cpu_seconds"], total_seconds=time.perf_counter()-started)
    inherited._write(root / "regional_refit_receipt.json", receipt)
    inherited._progress(root / "progress.json", dict(stage="regional_refit_complete", **receipt))
    return receipt


def assemble_epoch_bundle(frozen_candidate, pixel_fit_receipt, source_bundle, output, *, epoch_results=None):
    """Seal two completed epoch fits, also accepting SHA-exact transported files.

Each result directory contains regional_model.json, its training report and
regional_refit_receipt.json. Transport does not change their committed bytes.
"""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    pixel_receipt_path = Path(pixel_fit_receipt)
    pixels = training.read(pixel_receipt_path)
    table = source_bundle["light_epochs"]
    if (pixels.get("converged") is not True or pixels.get("cpu_only") is not True
            or set(pixels["epochs"]) != set(table["epochs"])
            or pixels["combined_label_sha256"] != source_bundle["receipt"]["combined_label_sha256"]):
        raise ValueError("Regional T4 bundle requires both completed own-epoch CPU pixel heads")
    result_roots = (epoch_results if epoch_results is not None else
                    {epoch_id: root / "epochs" / epoch_id for epoch_id in table["epochs"]})
    if set(result_roots) != set(table["epochs"]):
        raise ValueError("Regional T4 bundle needs one result directory per project epoch")
    gate, epochs = minimum_data_gate(source_bundle), {}
    if not gate["passed"]:
        raise ValueError("T4 new-lamp set requires at least 100 positive and 100 negative regions before fitting")
    for epoch_id in table["epochs"]:
        pixel = pixels["epochs"][epoch_id]
        path = training.checked(pixel["model_path"], pixel["model_sha256"])
        model = inputs.verify_pixel_model(path, pixel["model_sha256"], light_epoch_id=epoch_id)
        expected_inventory = inventory_contract(source_bundle, model, epoch_id)
        epoch_root = Path(result_roots[epoch_id]).resolve()
        fit_receipt = epoch_root / "regional_refit_receipt.json"
        receipt = inherited._read(fit_receipt)
        extraction_binding = receipt["binding"]["extraction_binding"]
        if (receipt.get("light_epoch_id") != epoch_id or receipt.get("converged") is not True
                or receipt.get("cpu_only") is not True or extraction_binding["inventory"] != expected_inventory
                or extraction_binding["g3_model_sha256"] != pixel["model_sha256"]
                or extraction_binding["light_epoch_table_sha256"] != table["receipt"]["sha256"]
                or extraction_binding["frozen_regional_sha256"] != sha256_file(Path(frozen_candidate) / "regional_model.json")
                or sha256_file(epoch_root / "regional_gas_training_report.json") != receipt["training_report_sha256"]):
            raise ValueError("Regional T4 epoch result changed its exact inventory or fit provenance")
        regional_path = epoch_root / "regional_model.json"
        inputs.verify_regional_model(regional_path, receipt["regional_model_sha256"], frozen_candidate)
        epochs[epoch_id] = dict(pixel_model_path=str(path.resolve()), pixel_model_sha256=pixel["model_sha256"],
            regional_model_path=str(regional_path), regional_model_sha256=receipt["regional_model_sha256"],
            regional_fit_receipt_path=str(fit_receipt), regional_fit_receipt_sha256=sha256_file(fit_receipt),
            training_label_sha256=receipt["training_label_sha256"], training_runs=gate["epochs"][epoch_id]["all_runs"])
    result = dict(contract=inputs.BUNDLE_CONTRACT, frozen_candidate=str(Path(frozen_candidate).resolve()),
        frozen_regional_sha256=sha256_file(Path(frozen_candidate) / "regional_model.json"),
        co2_pixel_sha256=inherited.CO2_SHA256, light_epoch_table_path=table["receipt"]["path"],
        light_epoch_table_sha256=table["receipt"]["sha256"], epochs=epochs,
        pixel_fit_receipt_path=str(pixel_receipt_path.resolve()), pixel_fit_receipt_sha256=sha256_file(pixel_receipt_path),
        minimum_data_gate=gate, independent_own_epoch_heads=True, common_co2_frozen=True,
        converged=True, cpu_only=True, rounds_2_3_4_read=False, test_labels_read=False, t2_bt2_bt3_labels_read=False)
    inherited._write(root / "regional_epoch_bundle.json", result)
    inputs.verify_epoch_bundle(root / "regional_epoch_bundle.json", table["receipt"]["path"], frozen_candidate)
    return result


def fit_regional_g3_t4(frozen_candidate, pixel_fit_receipt, source_bundle, output, *,
                     legacy_context_candidate, workers=1, threads=1, fit_threads=2,
                     memory_per_worker_mb=6144, memory_headroom_mb=2048):
    """Fit epochs sequentially and seal a routing bundle for candidate inference."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    pixel_receipt_path = Path(pixel_fit_receipt)
    pixels = training.read(pixel_receipt_path)
    table = source_bundle["light_epochs"]
    if (pixels.get("converged") is not True or pixels.get("cpu_only") is not True
            or set(pixels["epochs"]) != set(table["epochs"])
            or pixels["combined_label_sha256"] != source_bundle["receipt"]["combined_label_sha256"]):
        raise ValueError("Regional T4 requires both completed own-epoch CPU pixel heads")
    gate = minimum_data_gate(source_bundle)
    inherited._write(root / "minimum_data_gate.json", gate)
    if not gate["passed"]:
        raise ValueError("T4 new-lamp set requires at least 100 positive and 100 negative regions before fitting")
    for epoch_id in table["epochs"]:
        pixel = pixels["epochs"][epoch_id]
        path = training.checked(pixel["model_path"], pixel["model_sha256"])
        fit_regional_g3_t4_epoch(frozen_candidate, path, source_bundle, root / "epochs" / epoch_id,
            pixel_model_sha256=pixel["model_sha256"], legacy_context_candidate=legacy_context_candidate,
            light_epoch_id=epoch_id, workers=workers, threads=threads, fit_threads=fit_threads,
            memory_per_worker_mb=memory_per_worker_mb, memory_headroom_mb=memory_headroom_mb)
    return assemble_epoch_bundle(frozen_candidate, pixel_fit_receipt, source_bundle, root)


__all__ = ["FORMAT", "HEAD", "epoch_regions", "minimum_data_gate", "inventory_contract",
           "compact_source_plan", "compact_candidate_source_plan", "source_task_metadata", "extract_regional_training",
           "fit_regional_g3_t4_epoch", "fit_regional_g3_t4", "assemble_epoch_bundle"]
