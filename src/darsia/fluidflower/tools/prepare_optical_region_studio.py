"""Prepare the trained regional color model through ordinary Studio contracts."""
from __future__ import annotations
import os


import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import re
import shutil
import numpy as np
import cv2

from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower import phase_optical_regions_runtime as runtime
from darsia.fluidflower import phase_runtime
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.phase_validation import materialize_phase_validation_campaign, _model_identity
from darsia.fluidflower.tools.phase_optical_cleanup_preview import read, load


def write(path, value):
    from darsia.fluidflower.tools.render_concentration_movie import write as publish
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    publish(path, value)


def _is_detached_review_candidate(root: Path) -> bool:
    """Honor an explicit detached-run receipt before registering a Studio model."""
    if (root / "inference_incremental_plan.json").is_file():
        from darsia.fluidflower.tools.ffac_track1_incremental_inference import verify_incremental
        verify_incremental(root)
        return True
    from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import (
        RECEIPT_FILE, verify_inference_rerun,
    )
    if (root / RECEIPT_FILE).is_file():
        # Fail closed if the receipt, frozen fit, or exact inference sources
        # changed. This is a new map revision, never a new training launch.
        verify_inference_rerun(root)
        return True
    request_path = root.parent.parent / "regional_run" / "launch_request.json"
    if not request_path.is_file():
        return False
    request = read(request_path)
    return (request.get("review_candidate_only") is True
            and Path(str(request.get("output", ""))).resolve() == root.resolve())


def _ensure_published_optical_review_sources(root: Path) -> None:
    """Split a published optical parent's frozen campaign for per-run preparation.

    Regional candidates already have distinct original and model-bound jobs, so
    their review sources must never be reconstructed from the campaign manifest.
    """
    root = Path(root)
    if root.parent.name != "optical_parent":
        return
    receipt_path = root / "publication_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Optical parent has no fitted publication receipt")
    receipt = read(receipt_path)
    if receipt.get("artifact_kind") != "fitted-phase-review-candidate":
        raise ValueError("Optical parent has an unexpected publication kind")
    if (receipt.get("model_id"), receipt.get("model_sha256")) != _model_identity(root):
        raise ValueError("Optical parent differs from its fitted publication identity")
    source = root / "review_source" / "postflight_manifest.json"
    expected = (receipt.get("published_files") or {}).get(
        "review_source/postflight_manifest.json", {}).get("sha256")
    if not expected or sha256_file(source) != expected:
        raise ValueError("Optical parent review source differs from its fitted publication")
    jobs = read(source).get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Optical parent has no frozen review jobs")
    seen = set()
    missing = []
    for job in jobs:
        run = job.get("run") if isinstance(job, dict) else None
        if (not isinstance(run, str) or re.fullmatch(r"ac\d+", run) is None
                or job.get("dataset") != receipt.get("series") or run in seen):
            raise ValueError("Optical parent has duplicate or invalid frozen review jobs")
        seen.add(run)
        for name, value in (("original_job.json", job),
                            ("postflight_manifest.json", {"jobs": [job]})):
            path = root / "review_source" / run / name
            if path.is_file():
                if read(path) != value:
                    raise ValueError(f"{run}: per-run review source differs from the frozen campaign")
            else:
                missing.append((path, value))
    for path, value in missing:
        write(path, value)


def _regional_request_settings(series: str, parent: Path, fitted: dict,
                               v5_settings: dict | None) -> dict:
    """Preserve the optical input contract when publishing regional support."""
    settings = dict(series_scope=series, model_scope="single-series", series=[series],
        series_filter=[series], parent_model=str(parent),
        iteration=f"FF_{series[2:].upper()} · Color regions · {'Full aqueous corrections' if fitted.get('dense_aqueous_corrections') else 'Regional evidence'} · test",
        native_revision_policy="Each annotation kept on its original registered image")
    if v5_settings is not None:
        settings['v5_raw_photometry'] = dict(v5_settings)
        scope = fitted.get('v5_chemistry_fit_scope')
        if not isinstance(scope, dict):
            raise ValueError('V5 regional model lacks frozen chemistry fit scope')
        settings['v5_chemistry_fit_scope'] = dict(scope)
    return settings


def _verify_track1_nominal_context(entry: dict, native_sha: str,
                                   source_sha: str) -> None:
    """Keep a photo-only context tied to its exact materialized source.

    An inherited AC47 two-arm physical-grid claim is irrelevant to Track1's
    nominal optical context. This check never promotes it to physical QA.
    """
    marker = (entry.get("provenance") or {}).get("v5_track1_photo_geometry") or {}
    if (entry.get("native_manifest_sha256") != native_sha
            or marker.get("contract") !=
                "ffac-v5-track1-photo-geometry-cohort-v1"
            or marker.get("native_manifest_sha256") != native_sha
            or marker.get("original_source_manifest_sha256") != source_sha
            or marker.get("validation_only") is not True
            or marker.get("mass_eligible") is not False
            or marker.get("facies_ids_used") is not False
            or marker.get("nominal_axes_only") is not True):
        raise ValueError("Track1 nominal context differs from exact photo-only source")


def prepare(experiment, collection, root, *, fresh_optical=False,
            bottom_gas_guard=False, region_edge_guard=False,
            metric_postprocessing=True, nominal_review_gas_minimum=False):
    if root.exists():
        raise FileExistsError(root)
    fitted = read(experiment / "regional_model.json")
    evidence_path = Path(fitted["evidence_path"])
    if sha256_file(evidence_path) != fitted["evidence_sha256"] or not fitted["fit_report"]["converged"]:
        raise ValueError("Training evidence changed or fit is incomplete")
    evidence = read(evidence_path)
    parent = Path(fitted["parent_model"])
    original = read(collection / "postflight_manifest.json")
    series = {j['dataset'] for j in original['jobs']}
    if len(series) != 1:
        raise ValueError('A regional candidate must belong to exactly one series')
    series = series.pop()
    if {r['dataset'] for r in evidence['regions']} != {series}:
        raise ValueError('Training evidence and review images belong to different series')
    # A regional continuation inherits the optical heads and their input
    # representation. Dropping this marker would silently run the V5 heads on
    # the display JPEG rather than the source-bound float RAW sidecars.
    from darsia.fluidflower.phase_training.v5_linear_input import (
        SOURCE_FORMAT, V5_UNIT_DENOMINATOR, model_v5_settings,
    )
    v5_settings = model_v5_settings(parent)
    v5_canonical = None
    if v5_settings is not None:
        if series != 'ffac' or not fresh_optical:
            raise ValueError('V5 RAW regional maps require FF_AC fresh optical inference')
        receipt_path = Path(v5_settings['materialization_receipt_path'])
        if sha256_file(receipt_path) != v5_settings['materialization_receipt_sha256']:
            raise ValueError('V5 RAW materialization receipt changed')
        receipt = read(receipt_path)
        canonical = receipt.get('canonical_mapping') or []
        if len(canonical) != 43:
            raise ValueError('V5 RAW canonical source collection is incomplete')
        v5_canonical = {row['run'].casefold(): row['materialized_manifest_sha256']
                        for row in canonical}
        if len(v5_canonical) != 43:
            raise ValueError('V5 RAW canonical source runs are ambiguous')
    root.mkdir(parents=True)
    (root / "parent_provenance").mkdir()
    for name in ("summary.json", "training_request.json"):
        shutil.copy2(parent / name, root / "parent_provenance" / name)
    shutil.copytree(parent / "models", root / "models")
    shutil.copy2(experiment / "regional_model.json", root / "regional_model.json")
    profile = dict(format=regional.VERSION, model=dict(path="regional_model.json",
        sha256=sha256_file(root / "regional_model.json")), runs={})
    parent_profile = phase_runtime.load_physics_profile(parent) if fresh_optical else None
    inherited_scope = ((parent_profile or {}).get("v5_track1_review_scope")
                       if fresh_optical else None)
    fit_seen_runs = {str(row["run"]).casefold() for row in evidence["regions"]}
    if inherited_scope is not None:
        if not nominal_review_gas_minimum or metric_postprocessing or bottom_gas_guard or region_edge_guard:
            raise ValueError("Track1 photo-only Studio model cannot enable physical guards")
        profile["v5_track1_review_scope"] = dict(inherited_scope)
        if v5_settings.get("provisional_studio_training") is not True:
            raise ValueError("Track1 Studio model lacks provisional review status")
        warning_path = Path(v5_settings.get("provisional_photometry_warning_receipt_path") or "")
        if (not warning_path.is_file() or sha256_file(warning_path) !=
                v5_settings.get("provisional_photometry_warning_receipt_sha256")):
            raise ValueError("Track1 Studio photometry warnings changed")
        warning = read(warning_path)
        if (warning.get("contract") != "ffac-v5-track1-provisional-photometry-decision-v1"
                or warning.get("donor_tile_cv_status") != "failed-warning"
                or warning.get("late_yellow_status") != "failed-warning"):
            raise ValueError("Track1 Studio photometry warnings were suppressed")
        partition_binding = (evidence.get("audit") or {}).get("v5_review_fit_partition") or {}
        partition_path = Path(partition_binding.get("path") or "")
        if (partition_binding.get("contract") !=
                "ffac-v5-track1-frozen-review-fit-evaluation-partition-v1"
                or not partition_path.is_file()
                or sha256_file(partition_path) != partition_binding.get("sha256")):
            raise ValueError("Track1 Studio source lacks its frozen fit/evaluation partition")
        partition = read(partition_path)
        partition_rows = partition.get("rows") or []
        if (partition.get("contract") != partition_binding["contract"]
                or len(partition_rows) != 4048
                or len({row.get("region_id") for row in partition_rows}) != 4048):
            raise ValueError("Track1 Studio fit/evaluation review IDs changed")
        fit_seen_runs = {str(row["run"]).casefold() for row in partition_rows
                         if row.get("role") == "fit"}
    jobs = []
    for original_job in original["jobs"]:
        job = dict(original_job)
        old = parent if fresh_optical else Path(job["raw_phase_map_output_dir"]).parent.parent
        for head in (() if fresh_optical else ("co2_presence", "gas_within_co2")):
            relative = Path("models/generic") / f"{head}_signal.json"
            if sha256_file(old / relative) != sha256_file(root / relative):
                raise ValueError("Cached optical classifier differs from inherited classifier")
        inherited = phase_runtime.load_physics_profile(old)
        key = f"{job['dataset']}/{job['run']}"
        from darsia.fluidflower.tools.train_phase_model_further import native_source
        _, native_path, native = native_source(job)
        native_sha = sha256_file(native_path)
        if v5_canonical is not None:
            source_v5 = native.get('v5_photometry') or {}
            if (v5_canonical.get(job['run'].casefold()) != native_sha
                    or source_v5.get('format') != SOURCE_FORMAT
                    or source_v5.get('unit_denominator') != V5_UNIT_DENOMINATOR
                    or source_v5.get('display_jpeg_is_training_input') is not False):
                raise ValueError(f"V5 RAW source differs from fitted parent: {job['run']}")
        photometry_revision = job.get('photometry_revision')
        if inherited_scope is not None:
            if key not in inherited['runs']:
                raise ValueError('Track1 parent has no nominal optical context')
            _verify_track1_nominal_context(
                inherited['runs'][key], native_sha,
                (native.get('v5_photometry') or {}).get('source_manifest_sha256'))
        elif photometry_revision is not None:
            if not fresh_optical or key != 'ffac/ac47':
                raise ValueError('AC47 photometry revision requires fresh optical inference')
            from darsia.fluidflower.tools.phase_training_photometry_revision import verify_context_binding
            if key not in inherited['runs']:
                raise ValueError('No parent AC47 physical context to bind to photometry revision')
            source_native_sha = inherited['runs'][key]['native_manifest_sha256']
            verified = verify_context_binding(job, source_native_sha, native_sha,
                                              parent_context=inherited['runs'][key])
        elif key not in inherited['runs'] or inherited['runs'][key]['native_manifest_sha256'] != native_sha:
            old = Path(job['raw_phase_map_output_dir']).parent.parent
            inherited = phase_runtime.load_physics_profile(old)
        entry = dict(inherited["runs"][key])
        if photometry_revision is not None and inherited_scope is None:
            entry['native_manifest_sha256'] = native_sha
            entry['provenance'] = {**entry.get('provenance', {}),
                'photometry_revision': dict(
                    receipt_path=str(verified['receipt_path']),
                    receipt_sha256=verified['receipt_sha256'],
                    source_manifest_sha256=verified['source_manifest_sha256'],
                    target_manifest_sha256=verified['target_manifest_sha256'],
                    physical_grid_bytes_preserved=True)}
        elif entry['native_manifest_sha256'] != native_sha:
            raise ValueError('Context belongs to a different native image revision')
        src = old / entry["path"]
        if sha256_file(src) != entry["sha256"]:
            raise ValueError("Registered context changed")
        dest = root / "registered_context" / f"{job['run']}.npz"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        entry["path"] = dest.relative_to(root).as_posix()
        profile["runs"][key] = entry
        write(root / "review_source" / job["run"] / "original_job.json", original_job)
        job["phase_support_format"] = regional.VERSION
        job["phase_model_only"] = True
        job["quality_flags"] = dict(dataset=series, human_review_required=True,
            independent_holdout=False, independent_validation=False,
            training_seen=job["run"].casefold() in fit_seen_runs,
            evidence_role="trained_optical_region_candidate", review_mode="visible")
        if inherited_scope is not None:
            job["quality_flags"].update(photo_only_supported_sand=True,
                                         mass_analysis_allowed=False,
                                         unresolved_outside_photo_support=True,
                                         provisional_studio_training=True,
                                         photometry_yellow_gate_failed=True,
                                         photometry_tile_cv_failed=True,
                                         weak_phase_qa_pending=True)
        write(root / "review_source" / job["run"] / "postflight_manifest.json", dict(jobs=[job]))
        jobs.append(job)
    from dataclasses import asdict
    if nominal_review_gas_minimum:
        if (v5_settings is None or metric_postprocessing or bottom_gas_guard
                or region_edge_guard):
            raise ValueError("Nominal gas minimum requires V5 review-only maps without other metric guards")
        from darsia.fluidflower.phase_free_gas_output_guard import Settings as GasSettings
        profile["free_gas_cleanup"] = asdict(GasSettings(minimum_area_mm2=2.5))
        profile["v5_nominal_review_gas_minimum"] = {
            "contract": "ffac-v5-nominal-review-gas-minimum-v1",
            "minimum_area_mm2": 2.5,
            "physical_scale_qa_pass": False,
            "mass_analysis_allowed": False,
        }
    elif metric_postprocessing:
        from darsia.fluidflower.phase_region_output_guards import Settings
        profile["output_guards"] = asdict(Settings())
        from darsia.fluidflower.phase_free_gas_output_guard import Settings as GasSettings
        profile["free_gas_cleanup"] = asdict(GasSettings())
    elif bottom_gas_guard or region_edge_guard:
        raise ValueError("Metric guards require independently verified physical scale")
    from darsia.fluidflower.phase_regional_gas import Settings as RegionalGasSettings
    profile["regional_gas"] = asdict(RegionalGasSettings())
    if bottom_gas_guard:
        from darsia.fluidflower.phase_bottom_gas_guard import Settings as BottomGasSettings
        profile["bottom_gas_guard"] = asdict(BottomGasSettings())
    if region_edge_guard:
        from darsia.fluidflower.phase_region_edge_guard import Settings as EdgeSettings
        profile["region_edge_guard"] = asdict(EdgeSettings())
    write(root / runtime.PROFILE_FILE, profile)
    write(root / "review_source/postflight_manifest.json", dict(jobs=jobs))
    bundles = []
    for index, pair in enumerate(evidence["pairs"]):
        bundle = {}
        for kind, source in zip(("decisions", "combined_manifest"), pair):
            target = root / "input_snapshot" / f"{index:03d}_{kind}.json"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            bundle[kind] = dict(source_path=str(source), snapshot_path=str(target), sha256=sha256_file(target))
        bundles.append(bundle)
    for source, name in ((evidence_path, "evidence.json"),
                         (evidence_path.parent / "extraction.json", "extraction.json"),
                         (experiment / "protocol.json", "regional_training_protocol.json")):
        shutil.copy2(source, root / name)
    from darsia.fluidflower.phase_contributions import publish_regional_inventory
    publish_regional_inventory(root)
    inventory = dict(datasets=[series], series_scope=series, regions=len(evidence["regions"]),
        runs=sorted({r["run"] for r in evidence["regions"]}),
        phase_classes=dict(Counter(r["phase_class"] for r in evidence["regions"])))
    now = utc_now()
    request = dict(model_id=root.name, created_utc=now, output_root=str(root), bundles=bundles,
        artifact_kind="optical-region-training-candidate",
        settings=_regional_request_settings(series, parent, fitted, v5_settings))
    retractions = evidence['audit'].get('review_retractions')
    if retractions and retractions.get('snapshot_path'):
        from darsia.fluidflower.phase_review_retractions import validate_snapshot
        source = Path(retractions['snapshot_path'])
        snapshot = validate_snapshot(read(source))
        if snapshot['digest'] != retractions['snapshot_digest']:
            raise ValueError('Frozen evidence withdrawal policy changed')
        target = root / 'review_retractions.json'
        shutil.copy2(source, target)
        request['review_retractions'] = dict(path=str(target), sha256=sha256_file(target),
            snapshot_digest=retractions['snapshot_digest'])
    write(root / "training_request.json", request)
    summary = dict(schema=1, model_id=root.name, campaign_id=root.name, created_utc=now,
        training_completed_utc=now, artifact_kind="optical-region-training-candidate",
        optical_training_performed=False, regional_training_performed=True,
        inherited_optical_model_id=parent.name, inherited_heads=["co2_presence", "gas_within_co2"],
        regional_model_sha256=sha256_file(root / "regional_model.json"),
        regional_fit=fitted["fit_report"], inventory=inventory, runs=inventory["runs"],
        models={}, evaluation=dict(kind="training-only", generalization_validated=False),
        human_validation=None, deployment=None)
    write(root / "summary.json", summary)
    return root


class CachedOptical:
    def __init__(self, root, job, baseline):
        self.root, self.job, self.baseline = root, job, baseline
        self.index = 0
        self.receipts = []

    def __call__(self, current, previous):
        p, raw, domain, spacing, native, identity = load(self.job["review_output_dir"],
            self.job["run"], json.dumps(self.job, sort_keys=True), self.index)
        task = identity["worker_task"]
        from darsia.fluidflower.tools.train_phase_optical_regions import crop
        if not np.array_equal(current, crop(Path(task["current"]), native["content_box_xywh"])):
            raise ValueError("Cached optical photo differs")
        if not np.array_equal(self.baseline, np.load(task["baseline"], mmap_mode="r")):
            raise ValueError("Cached optical baseline differs")
        expected_previous = None if task["previous"] is None else crop(Path(task["previous"]), native["content_box_xywh"])
        if (previous is None) != (expected_previous is None) or (previous is not None and not np.array_equal(previous, expected_previous)):
            raise ValueError("Cached optical temporal input differs")
        with np.load(identity["score_archive"]) as archive:
            heads = archive["heads"]
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import partition
        output = partition(heads)
        self.receipts.append(identity)
        self.index += 1
        load.cache_clear()
        return output

    def close(self):
        write(self.root / "optical_reuse_receipts.json", self.receipts)


def references(*, execution_backend="auto", execution_threads=1, verify_numerics=False, **kw):
    from darsia.fluidflower.tools.phase_reference_single_pass import SinglePassReferences, Endpoint
    from darsia.fluidflower.tools.phase_injection_references import injection_end_references, resolve_reference_asset
    run, review, native = kw["run_id"], kw["review_root"], kw["native_manifest"]
    baseline_time = (native.get("reference") or {}).get("elapsed_h")
    baseline_time = float(baseline_time) if baseline_time is not None else -1e-6
    baseline_photo = review / "native_references" / run / "pre_injection.jpg"
    endpoints = [Endpoint("baseline", baseline_time, baseline_photo, sha256_file(baseline_photo))]
    for row in injection_end_references(dict(run=run, dataset=kw["dataset"]), review):
        if row.get("available") and not row.get("needs_materialization") and row.get("frame_index") is None:
            photo = resolve_reference_asset(row, review, run)
            endpoints.append(Endpoint(row["id"], float(row["elapsed_h"]), photo, sha256_file(photo)))
    def predict(current, previous, reference_id):
        from darsia.fluidflower.tools.phase_studio_execution import make_predictor, CpuPredictor
        report = {"runs": []}
        shared = kw.get("predictor")
        if hasattr(shared, "predict_reference"):
            result = shared.predict_reference(current, previous, report)
        elif isinstance(shared, CpuPredictor):
            result = shared(current, previous)
            report['backend'] = 'cpu-reference-tiles' if verify_numerics else 'cpu-tiles'
        else:
            predictor = make_predictor(kw['model'], kw['baseline'], report,
                                       backend=execution_backend, threads=execution_threads,
                                       verify_numerics=verify_numerics)
            try:
                result = predictor(current, previous)
            finally:
                predictor.close()
        write(kw['destination'] / 'reference_backend_checks' / (reference_id + '.json'), report)
        return result
    observer = SinglePassReferences(endpoints, run=run, destination=kw["destination"],
        canvas_shape=kw["canvas_shape"], content_box=kw["content_box"],
        pipeline_sha256=kw["pipeline_sha256"], model_sha256=kw["model_sha256"], predict_reference=predict)
    observer.initial_reference_time = baseline_time
    return observer


def _registered_v5_backend(model, baseline, requested):
    """Select a registered-frame backend without constructing a legacy predictor."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Backend must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu", "CPU requested"
    try:
        import cupy as cp
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor
        CudaFeaturePredictor.checked_contract(model)
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device")
        free, _ = cp.cuda.runtime.memGetInfo()
        needed = baseline.shape[0] * baseline.shape[1] * 260 + 512 * 1024 ** 2
        if free < needed:
            raise RuntimeError(f"CUDA memory estimate {needed} exceeds available {free}")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if requested == "cuda":
            raise RuntimeError(f"Requested registered CUDA backend unavailable: {error}") from error
        return "cpu", f"{type(error).__name__}: {error}"
    return "cuda", None


def run_one(root, run, fresh_optical=False, threads=None, memory_gb=None, execution_backend="auto", defer_references=False,
            generate_review_proposals=False, *, output_mode='review', numeric_writer_factory=None,
            verify_numerics=False, eager_references=False):
    from darsia.fluidflower.phase_execution_resources import execution_resources
    from darsia.fluidflower.tools.train_phase_model_further import native_source
    from darsia.fluidflower.tools.phase_studio_execution import make_predictor
    root = Path(root)
    if not (root / "review_source" / run / "original_job.json").is_file():
        _ensure_published_optical_review_sources(root)
    original = read(root / "review_source" / run / "original_job.json")
    _, native_path, _ = native_source(original)
    native = read(native_path)
    registered_v5 = bool(native.get("v5_track1_photo_geometry") or native.get("v5_frame_geometry"))
    if native.get("v5_track1_photo_geometry") and native.get("v5_frame_geometry"):
        raise ValueError("A native source cannot combine Track1 and physical V5 geometry")
    if registered_v5 and not fresh_optical:
        raise ValueError("Moving V5 geometry requires fresh registered optical inference")
    _, _, width, height = native["content_box_xywh"]
    resources = execution_resources((height, width), cpus=threads, memory_gb=memory_gb)
    from darsia.fluidflower.tools.phase_throughput_resources import (throughput_resources, apply_phase_threads,
                                                   apply_calibrated_output_resources)
    calibration_path = root.parent.parent / 'config' / 'phase_execution_calibration.json'
    resources, output_resource_calibration = apply_calibrated_output_resources(
        resources, (height, width), calibration_path,
        enabled=fresh_optical and execution_backend != 'cpu',
        generate_review_proposals=generate_review_proposals)
    resources.apply()
    throughput = throughput_resources(resources, (height, width),
        calibration_path=calibration_path, generate_review_proposals=generate_review_proposals)
    if output_mode not in {'review', 'numerical'}:
        raise ValueError('Studio preparation supports review or numerical output')
    destination = root / ('numerical/run_packages' if output_mode == 'numerical' else 'validation/run_packages') / run
    from darsia.fluidflower.tools import materialize_phase_iteration as materializer, phase_inference_benchmark_backends as backend
    from darsia.fluidflower.tools import phase_reference_single_pass as single
    from darsia.fluidflower.tools.phase_reference_readiness import (mark_references_pending,
        prepare_validation_references, defer_validation_references)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    from darsia.fluidflower.tools import phase_studio_execution, phase_optical_cache, phase_optical_cache_compatibility
    from darsia.fluidflower import (phase_execution_resources, phase_validation,
                                   phase_learning, phase_proposal_transport)
    for module in (materializer, backend, single, phase_studio_execution, phase_execution_resources,
                   phase_validation, phase_learning, phase_proposal_transport, phase_optical_cache,
                   phase_optical_cache_compatibility):
        shutil.copy2(module.__file__, snapshot / Path(module.__file__).name)
    execution_sources = {Path(module.__file__).name: sha256_file(Path(module.__file__))
                         for module in (phase_validation, phase_learning, phase_proposal_transport,
                                        phase_optical_cache, phase_optical_cache_compatibility)}
    if output_mode == 'numerical' or numeric_writer_factory is not None:
        from darsia.fluidflower import phase_numeric_fields
        path = Path(phase_numeric_fields.__file__)
        shutil.copy2(path, snapshot / path.name)
        execution_sources[path.name] = sha256_file(path)
    if throughput.enabled:
        from darsia.fluidflower.tools import (phase_throughput_resources, phase_optical_prefetch,
                             phase_regional_pipeline, phase_regional_fused_execution)
        for module in (phase_throughput_resources, phase_optical_prefetch,
                       phase_regional_pipeline, phase_regional_fused_execution):
            path = Path(module.__file__)
            shutil.copy2(path, snapshot / path.name)
            execution_sources[path.name] = sha256_file(path)
    if registered_v5:
        from darsia.fluidflower.tools import (phase_frame_workers, phase_v5_registered_frame,
            ffac_v5_temporal_alignment, ffac_v5_track1_photo_context, ffac_v5_frame_context)
        from darsia.fluidflower.phase_training import v5_linear_input
        for module in (phase_frame_workers, phase_v5_registered_frame,
                       ffac_v5_temporal_alignment, ffac_v5_track1_photo_context,
                       ffac_v5_frame_context, v5_linear_input):
            path = Path(module.__file__)
            shutil.copy2(path, snapshot / path.name)
            execution_sources[path.name] = sha256_file(path)
    provenance = dict(backend="cuda-float32-cached-features-checked", requested_backend="auto",
        checked_cuda_sha256=sha256_file(Path(materializer.__file__)), backend_sha256=sha256_file(Path(backend.__file__)),
        single_pass_reference_sha256=sha256_file(Path(single.__file__)),
        optical_reuse="Exact archived score identity and pixel map verified; inherited classifiers are byte-identical",
        adapter_sha256=sha256_file(Path(__file__)))
    if fresh_optical:
        provenance.update(backend="selected-at-runtime", optical_reuse=False)
    write(destination / "execution_resources.json", resources.to_dict())
    write(destination / "execution_resource_calibration.json", output_resource_calibration)
    provenance["execution_resources"] = resources.to_dict()
    provenance['output_resource_calibration'] = output_resource_calibration
    provenance["requested_backend"] = execution_backend
    provenance["review_proposals"] = "requested" if generate_review_proposals else "deferred-until-quiz"
    provenance['verify_numerics'] = bool(verify_numerics)
    provenance['reference_preparation'] = 'eager' if eager_references else 'on-demand'
    provenance['throughput_execution'] = dict(requested=throughput.enabled,
        plan=throughput.to_dict(), sources=execution_sources)
    if output_mode == 'review':
        mark_references_pending(destination)
    def progress(message):
        write(destination / "progress.json", dict(message=message, updated_utc=utc_now()))
        print(message, flush=True)
    backend_report = {'runs': []}
    active_throughput = False
    def predictor_factory(model, baseline):
        nonlocal active_throughput
        if not fresh_optical:
            return CachedOptical(destination, original, baseline)
        if registered_v5:
            if verify_numerics:
                raise ValueError("Registered V5 Studio inference requires its frame-bound parity receipts")
            from darsia.fluidflower.tools.phase_frame_workers import ENV as FRAME_ENV, ParallelFramePredictor
            if os.environ.get(FRAME_ENV):
                selected, reason = _registered_v5_backend(model, baseline, execution_backend)
                predictor = ParallelFramePredictor(root, baseline, backend_report,
                    selected, destination / "registered_frame_cache")
                backend_report.update(backend=f"{selected}-registered-frame-broker",
                                      verify_numerics=False)
                if reason:
                    backend_report["cpu_selection_reason"] = reason
            else:
                if execution_backend == "cuda":
                    raise ValueError("Registered CUDA inference requires a frame broker")
                predictor = None
                backend_report.update(backend="cpu-registered-frame-direct",
                                      cpu_selection_reason="No registered frame broker",
                                      verify_numerics=False)
            active_throughput = False
            write(destination / "throughput_resources.json", dict(throughput.to_dict(),
                active=False, optical_prefetch=bool(predictor is not None),
                parallel_phase=False, backend=backend_report["backend"],
                frame_execution="source-bound-registered-v5"))
            provenance.update(backend=backend_report["backend"],
                              frame_execution="source-bound-registered-v5",
                              registered_v5_sources=execution_sources)
            return predictor
        predictor = make_predictor(model, baseline, backend_report,
                                   backend=execution_backend, threads=resources.compute_threads,
                                   prefetch=throughput.enabled,
                                   parity_threads=throughput.optical_threads or 1,
                                   verify_numerics=verify_numerics)
        active_throughput = throughput.enabled and backend_report['backend'] in {'cuda-float32', 'cuda-float32-checked'}
        if active_throughput:
            apply_phase_threads(throughput)
        actual = dict(throughput.to_dict(), active=active_throughput,
                      optical_prefetch=active_throughput, parallel_phase=False,
                      backend=backend_report['backend'])
        if not active_throughput:
            actual.update(phase_workers=1, phase_threads=resources.compute_threads,
                          pending_frames=1, reason=backend_report.get('cpu_selection_reason') or throughput.reason)
        write(destination / 'throughput_resources.json', actual)
        provenance['backend'] = backend_report['backend']
        return predictor
    def reference_factory(**kw):
        return references(execution_backend=execution_backend,
                          execution_threads=resources.compute_threads,
                          verify_numerics=verify_numerics, **kw)
    def decision_factory(physics, predictor):
        if not active_throughput:
            return None
        from darsia.fluidflower.tools.phase_regional_pipeline import create, unsupported_reason
        from darsia.fluidflower.tools.phase_regional_fused_execution import apply
        executor = create(physics, workers=throughput.phase_workers, pending=throughput.pending_frames,
                          apply_function=apply)
        actual = dict(throughput.to_dict(), active=True, optical_prefetch=True,
                      parallel_phase=executor is not None, backend=backend_report['backend'])
        if executor is None:
            from dataclasses import replace
            serial_threads = max(1, resources.compute_threads - throughput.optical_threads - 1)
            apply_phase_threads(replace(throughput, phase_threads=serial_threads))
            actual.update(phase_workers=1, phase_threads=serial_threads, pending_frames=1,
                          reason=unsupported_reason(physics) or 'serial phase execution')
        write(destination / 'throughput_resources.json', actual)
        return executor
    materialize_phase_validation_campaign(root, root / "review_source" / run, destination,
        predictor_factory=predictor_factory,
        inference_provenance=provenance,
        reference_factory=reference_factory if output_mode == 'review' and eager_references else None, progress=progress,
        decision_executor_factory=decision_factory,
        generate_review_proposals=generate_review_proposals, output_mode=output_mode,
        numeric_writer_factory=numeric_writer_factory)
    if fresh_optical:
        write(destination/'backend_checks.json', backend_report)
    if output_mode == 'review':
        if not eager_references:
            defer_validation_references(destination)
        elif not defer_references:
            prepare_validation_references(destination, progress=progress)
    return str(destination)


def complete_run(root, run, threads, allow_inference=False, *, eager_presentation=False,
                 eager_references=False):
    """CPU-only receipt/media work can overlap the next run's GPU inference."""
    from threadpoolctl import threadpool_limits
    from darsia.fluidflower.tools.phase_reference_batch import ReferenceInferenceRequired
    from darsia.fluidflower.tools.phase_reference_readiness import prepare_validation_references, defer_validation_references
    from darsia.fluidflower.tools.finish_optical_region_studio import media
    import time
    started = time.perf_counter()
    root = Path(root)
    package = root / 'validation/run_packages' / run
    cv2.setNumThreads(threads)
    with threadpool_limits(limits=threads):
        try:
            if eager_references:
                prepare_validation_references(package, progress=lambda s: print(s, flush=True),
                                              allow_inference=allow_inference)
            else:
                defer_validation_references(package)
        except ReferenceInferenceRequired:
            return dict(run=run, needs_inference=True)
        receipt_path = media(root, run, threads, eager=eager_presentation)
        # Publish this experiment as soon as its native timeline is usable.
        # Reference identities are rebound only for this presentation-path change.
        from darsia.fluidflower.tools.phase_reference_readiness import RECEIPT, require_reference_readiness
        from darsia.fluidflower.tools.phase_reference_maps import _asset
        require_reference_readiness(package, allow_pending=not eager_references)
        manifest = read(package / 'postflight_manifest.json')
        manifest['jobs'] = [read(Path(receipt_path))['job']]
        write(package / 'postflight_manifest.json', manifest)
        readiness = read(package / RECEIPT)
        readiness['campaign'] = _asset(package / 'postflight_manifest.json')
        write(package / RECEIPT, readiness)
    result = dict(run=run, package=str(package), needs_inference=False,
                  finalization_seconds=time.perf_counter()-started, completed_utc=utc_now())
    write(root / 'validation/finalization_timings' / (run+'.json'), result)
    return result


def prepare_pipelined(root, jobs, *, cpus, memory_gb, fresh_optical=True, backend='auto', generate_review_proposals=False,
                      verify_numerics=False, eager_references=False):
    """One GPU owner, one bounded CPU finalizer, unchanged per-run chronology."""
    from darsia.fluidflower.phase_execution_resources import pipeline_resources
    from darsia.fluidflower.tools.finish_optical_region_studio import finish
    from concurrent.futures import wait, FIRST_COMPLETED
    allocation = pipeline_resources(cpus, memory_gb)
    if not allocation['overlap']:
        raise ValueError('Pipeline requires a separate finalizer resource budget')
    write(root / 'preparation_resources.json', allocation)
    all_jobs = read(root / 'review_source/postflight_manifest.json')['jobs']
    native_done = len(all_jobs)-len(jobs)
    resumed = set(j['run'] for j in all_jobs)-set(j['run'] for j in jobs)
    finalized = set()
    published = set()
    def progress():
        nonlocal published
        if finalized and finalized != published:
            assemble(root, [str(root / 'validation/run_packages' / run) for run in sorted(finalized)],
                     partial=len(finalized) < len(all_jobs))
            published = set(finalized)
        status_path = root / 'studio_preparation.json'
        status = read(status_path) if status_path.is_file() else {}
        write(status_path, dict(status, status='preparing',
            runs_done=len(finalized), native_runs_done=native_done, runs_total=len(all_jobs),
            updated_utc=utc_now()))
    completed, deferred = [], []
    try:
        with ProcessPoolExecutor(max_workers=1) as finalizers:
            for run in sorted(resumed):
                completed.append(finalizers.submit(complete_run, root, run, allocation['finalizer_cpus'],
                                                   eager_references=eager_references))
            with ProcessPoolExecutor(max_workers=1) as inference:
                for job in jobs:
                    # Propagate CPU failures before starting another run.
                    for task in completed:
                        if task.done():
                            result = task.result()
                            if not result['needs_inference']:
                                finalized.add(result['run'])
                    if finalized != published:
                        progress()
                    active = inference.submit(run_one, root, job['run'], fresh_optical,
                        allocation['map_cpus'], allocation['map_memory_gb'], backend, True,
                        generate_review_proposals, verify_numerics=verify_numerics,
                        eager_references=eager_references)
                    while not active.done():
                        waiting = [task for task in completed if not task.done()]
                        wait([active, *waiting], return_when=FIRST_COMPLETED)
                        for task in completed:
                            if task.done():
                                result = task.result()
                                if not result['needs_inference'] and result['run'] not in finalized:
                                    finalized.add(result['run'])
                                    progress()
                    active.result()
                    native_done += 1
                    progress()
                    completed.append(finalizers.submit(complete_run, root, job['run'],
                                                       allocation['finalizer_cpus'], eager_references=eager_references))
            # The GPU owner has now exited. Unusual missing reference acquisitions
            # may need fresh inference; they must never compete for its VRAM.
            for task in completed:
                result = task.result()
                if result['needs_inference']:
                    deferred.append(result['run'])
                else:
                    finalized.add(result['run'])
                    progress()
            for run in deferred:
                finalizers.submit(complete_run, root, run, allocation['finalizer_cpus'], True,
                                  eager_references=eager_references).result()
                finalized.add(run)
                progress()
        packages = [str(root / 'validation/run_packages' / j['run']) for j in all_jobs]
        assemble(root, packages)
    except Exception:
        import traceback
        write(root / 'studio_preparation.json', dict(status='failed', error=traceback.format_exc()))
        raise


def assemble(root, results, *, partial=False):
    from darsia.fluidflower.tools.phase_reference_maps import _asset
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness, RECEIPT, FORMAT
    packages = [Path(v) for v in results]
    if not packages:
        raise ValueError('Cannot publish an empty validation selection')
    for package in packages:
        require_reference_readiness(package, allow_pending=True)
    manifests = [read(p / "postflight_manifest.json") for p in packages]
    expected_path = root / 'expected_review_runs.json'
    if expected_path.is_file():
        from darsia.fluidflower.tools.extend_optical_region_studio import require_run_set
        selected_jobs = [j for manifest in manifests for j in manifest['jobs']]
        expected_runs = read(expected_path)['runs']
        if partial:
            actual = [j['run'] for j in selected_jobs]
            if len(set(actual)) != len(actual) or not set(actual) <= set(expected_runs):
                raise ValueError('Partial validation contains duplicate or unexpected experiments')
        else:
            require_run_set(selected_jobs, expected_runs)
    bindings = [read(p / "validation_binding.json") for p in packages]
    for key in ("model_sha256", "pipeline_sha256", "physics_runtime"):
        if len({b[key] for b in bindings}) != 1:
            raise ValueError("Run packages do not identify one model")
    source_root = root / "review_source"
    source_path = source_root / "postflight_manifest.json"
    source_digest = sha256_file(source_path)
    source_jobs = read(source_path)["jobs"]
    source_by_run = {job["run"]: job for job in source_jobs}
    if len(source_by_run) != len(source_jobs):
        raise ValueError("Frozen source campaign has duplicate experiments")
    for package, manifest, binding in zip(packages, manifests, bindings):
        jobs = manifest["jobs"]
        if len(jobs) != 1:
            raise ValueError("A validation run package must contain exactly one experiment")
        run = jobs[0]["run"]
        per_run_source = source_root / run / "postflight_manifest.json"
        per_run_digest = sha256_file(per_run_source)
        if (manifest.get("source_manifest_sha256") != per_run_digest
                or binding.get("source_manifest_sha256") != per_run_digest
                or read(per_run_source)["jobs"] != [source_by_run.get(run)]):
            raise ValueError(f"Validation package does not bind frozen source for {run}")
    destination = root / "validation"
    # A session opened midway keeps immutable sources as further runs finish.
    # The main binding is only a discovery pointer to this prepared subset.
    import hashlib
    snapshot_material = 'aggregate-source-v2:' + source_digest + ':' + ''.join(
        sha256_file(p / name) for p in packages
        for name in ('postflight_manifest.json', 'validation_binding.json', RECEIPT))
    snapshot_id = hashlib.sha256(snapshot_material.encode()).hexdigest()[:20]
    destination = destination / 'prepared_sets' / snapshot_id
    manifest = dict(manifests[0], jobs=[j for m in manifests for j in m["jobs"]],
                    source_campaign_root=str(source_root), source_manifest_sha256=source_digest)
    binding = dict(bindings[0], ffac_root=str(destination), ffum_root=str(destination),
        combined_root=str(destination), selected_candidates={k:v for b in bindings for k,v in b["selected_candidates"].items()},
        source_manifest_sha256=source_digest, run_count=len(packages), completed_utc=utc_now())
    receipts = [read(p / RECEIPT) for p in packages]
    reference_status = 'ready' if all(r.get('status') == 'ready' for r in receipts) else 'deferred'
    if (destination / 'validation_binding.json').is_file():
        require_reference_readiness(destination, allow_pending=True)
        existing = read(destination / 'validation_binding.json')
        if (read(destination / 'postflight_manifest.json') != manifest
                or any(existing.get(key) != binding.get(key) for key in
                        ('model_sha256', 'pipeline_sha256', 'physics_runtime', 'selected_candidates',
                         'combined_root', 'source_manifest_sha256'))):
            raise ValueError('Existing prepared snapshot changed')
        binding = existing
    else:
        write(destination / "postflight_manifest.json", manifest)
        write(destination / "validation_binding.json", binding)
        write(destination / RECEIPT, dict(format=FORMAT, status=reference_status, output_mode="review",
            binding=_asset(destination / "validation_binding.json"), campaign=_asset(destination / "postflight_manifest.json"),
            references=[r for receipt in receipts for r in receipt["references"]],
            unavailable_references=[r for receipt in receipts for r in receipt["unavailable_references"]]))
    from darsia.fluidflower.tools.phase_review_collections import refresh_prepared_collection
    collection = refresh_prepared_collection(root, destination)
    if collection is None:
        if (root / 'validation/registered_base_frozen.json').exists():
            raise ValueError('Registered validation base is preserved; restore its collection discovery before publication')
        for filename in ('postflight_manifest.json', 'validation_binding.json'):
            write(root / 'validation' / filename, read(destination / filename))
        receipt = read(destination / RECEIPT)
        receipt.update(binding=_asset(root / 'validation/validation_binding.json'),
                       campaign=_asset(root / 'validation/postflight_manifest.json'))
        write(root / 'validation' / RECEIPT, receipt)
    settings_file = None
    if not _is_detached_review_candidate(root):
        from PySide6.QtCore import QSettings
        settings = QSettings("DarSIA", "DarSIA Phase Model Studio")
        saved = json.loads(str(settings.value("modelHistory/completedModelRoots", "{}")))
        saved[root.name] = str(root)
        settings.setValue("modelHistory/completedModelRoots", json.dumps(saved))
        settings.sync()
        settings_file = settings.fileName()
    write(root / "studio_preparation.json", dict(status="preparing" if partial else "ready", runs=binding["run_count"],
        runs_done=binding['run_count'], presentation_ready=True, partial=partial,
        model_identity=list(_model_identity(root)), validation=str(destination), settings_file=settings_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--fresh-optical', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cpus', type=int, default=None)
    parser.add_argument('--memory-gb', type=float, default=None)
    parser.add_argument('--backend', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-mode', choices=('review', 'numerical'), default='review',
                        help='Numerical emits final float32 fields for mass analysis without Studio assets or reference-map rendering')
    parser.add_argument('--serial', action='store_true', help='Disable overlap for profiling or constrained machines')
    parser.add_argument('--generate-review-proposals', action='store_true',
                        help='Generate quiz proposals explicitly; normally deferred until opening the quiz.')
    parser.add_argument('--verify-numerics', action='store_true',
                        help='Explicit diagnostic run comparing predictions with CPU references.')
    parser.add_argument('--eager-references', action='store_true',
                        help='Prepare reference phase maps now; normally calculated when selected in Studio.')
    args = parser.parse_args()
    if args.serial:
        os.environ['PHASE_THROUGHPUT'] = '0'
    if not args.resume:
        prepare(args.experiment, args.collection, args.root, fresh_optical=args.fresh_optical)
    else:
        _ensure_published_optical_review_sources(args.root)
    jobs = read(args.root / "review_source/postflight_manifest.json")["jobs"]
    results, pending = [], []
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness, RECEIPT
    for job in jobs:
        if args.output_mode == 'numerical':
            pending.append(job)
            continue
        package = args.root / 'validation/run_packages' / job['run']
        if args.resume and (package / RECEIPT).is_file() and read(package / RECEIPT).get('status') in {'ready', 'deferred'}:
            # Complete the requested references below without regenerating maps.
            require_reference_readiness(package, allow_pending=True)
            results.append(str(package))
        else:
            pending.append(job)
    from darsia.fluidflower.phase_execution_resources import available_resources
    available_cpus, available_memory = available_resources()
    total_cpus = args.cpus if args.cpus is not None else int(os.environ.get('PHASE_CPU_LIMIT', available_cpus))
    cpu_selection = None
    if args.cpus is None and 'PHASE_CPU_LIMIT' not in os.environ and args.backend != 'cpu' and not args.serial:
        from darsia.fluidflower.tools.phase_throughput_resources import total_cpu_recommendation
        cpu_selection = total_cpu_recommendation(available_cpus,
            args.root.parent.parent / 'config' / 'phase_execution_calibration.json',
            generate_review_proposals=args.generate_review_proposals)
        total_cpus = cpu_selection['cpu_limit']
    write(args.root / 'preparation_cpu_selection.json', dict(cpus=total_cpus,
        selection=cpu_selection, explicit=args.cpus is not None or 'PHASE_CPU_LIMIT' in os.environ))
    total_memory = args.memory_gb if args.memory_gb is not None else float(os.environ.get('PHASE_MEMORY_GB', available_memory / 1024**3))
    if total_cpus < 1 or total_memory <= 0 or not 1 <= args.workers <= total_cpus:
        raise ValueError('Workers must fit a positive CPU and memory budget')
    if args.backend != 'cpu' and args.workers != 1:
        raise ValueError('Use one run worker for GPU/auto, or select --backend cpu')
    if args.output_mode == 'numerical':
        # Each run keeps chronological delivery; GPU ownership remains singular.
        # Physical mass inputs can consume each frame in memory via run_one's
        # numeric_writer_factory or replay these verified numeric archives.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, args.root, j['run'], args.fresh_optical,
                max(1, total_cpus//args.workers), total_memory/args.workers, args.backend,
                output_mode='numerical', verify_numerics=args.verify_numerics): j['run'] for j in pending}
            for future in as_completed(futures):
                results.append(future.result())
        write(args.root / 'numerical/manifest.json', dict(format='phase-numeric-campaign-v1',
            model_identity=list(_model_identity(args.root)), run_packages=sorted(results),
            completed_utc=utc_now(), output_mode='numerical'))
        raise SystemExit(0)
    from darsia.fluidflower.phase_execution_resources import pipeline_resources
    if not args.serial and args.workers == 1 and pipeline_resources(total_cpus, total_memory)['overlap']:
        prepare_pipelined(args.root, pending, cpus=total_cpus, memory_gb=total_memory,
                          fresh_optical=args.fresh_optical, backend=args.backend,
                          generate_review_proposals=args.generate_review_proposals,
                          verify_numerics=args.verify_numerics, eager_references=args.eager_references)
        raise SystemExit(0)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for package in results:
            result = complete_run(args.root, Path(package).name, max(1, total_cpus//args.workers),
                                  eager_references=args.eager_references)
            if result['needs_inference']:
                complete_run(args.root, Path(package).name, max(1, total_cpus//args.workers), True,
                             eager_references=args.eager_references)
        futures = {pool.submit(run_one, args.root, j["run"], args.fresh_optical, max(1, total_cpus//args.workers), total_memory/args.workers, args.backend,
                               False, args.generate_review_proposals, verify_numerics=args.verify_numerics,
                               eager_references=args.eager_references):j["run"] for j in pending}
        for future in as_completed(futures):
            result = future.result()
            complete_run(args.root, futures[future], max(1, total_cpus//args.workers),
                         eager_references=args.eager_references)
            results.append(result)
            assemble(args.root, sorted(results), partial=len(results) < len(jobs))
            write(args.root / "studio_preparation.json", dict(status="preparing", runs_done=len(results), runs_total=len(jobs)))
    assemble(args.root, sorted(results))
