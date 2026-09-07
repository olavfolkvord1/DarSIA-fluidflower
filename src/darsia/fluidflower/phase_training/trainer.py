"""Train transferable gas/aqueous optical heads from reviewed FF-AC and FF-UM.

The command consumes immutable review snapshots, stages native corrected
review images locally, extracts the shared optical feature contract in
parallel, and performs leave-one-run-out validation.  Repeated logistic fits
run on CUDA float32 when available; a deterministic CPU float64 fit remains
the numerical parity reference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import multiprocessing
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from darsia.fluidflower.phase_features import (
    DARSIA_CHANGE_DETECTOR,
    GENERIC_FEATURE_VERSION,
    NO_CHANGE_DETECTOR,
    OpticalFeatureContract,
)
from darsia.fluidflower.phase_training.logistic import (
    LogisticFit,
    ReviewedRegion,
    balanced_pixel_view,
    canonical_phase_class,
    cuda_available,
    fit_weighted_logistic,
    fit_weighted_logistic_mil,
    predict_logistic,
    region_probability_scores,
    regional_evaluation_view,
    regional_metrics,
    regional_presence_target,
    tune_presence_threshold,
)
from darsia.fluidflower.phase_training.model import (
    TRANSFERABLE_MODEL_ALGORITHM,
    TRANSFERABLE_MODEL_SCHEMA,
    DualPhaseSignalModel,
    feature_samples,
)
from darsia.fluidflower.phase_training.review_regions import (
    _annotation_mask,
    _load_rgb,
    _sample_mask,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
FLUIDFLOWER_ROOT = PACKAGE_ROOT.parent


CAMPAIGN_ID = "crossseries_phase_model_v3_20260830"
DEFAULT_OUTPUT = Path(rf"Z:\Albus\Autokalibrering_log\{CAMPAIGN_ID}")
DEFAULT_CACHE = Path(rf"C:\FFAC_cache\{CAMPAIGN_ID}")
DEFAULT_SNAPSHOT = DEFAULT_OUTPUT / "input_snapshot"
DEFAULT_BUNDLES = (
    (
        DEFAULT_SNAPSHOT / "training_decisions_ac18_run15.json",
        DEFAULT_SNAPSHOT / "training_combined_manifest.json",
    ),
    (
        DEFAULT_SNAPSHOT / "completed_validation_ac52_run16.json",
        DEFAULT_SNAPSHOT / "validation_combined_manifest.json",
    ),
)

MODEL_VARIANTS: Mapping[str, OpticalFeatureContract] = {
    "generic": OpticalFeatureContract(
        chemistry_adapter="none",
        reference_adapter="none",
        include_dynamic=True,
        generic_version=GENERIC_FEATURE_VERSION,
        change_detector=DARSIA_CHANGE_DETECTOR,
    ),
    "btb-ratiometric-relative-v2": OpticalFeatureContract(
        chemistry_adapter="btb-ratiometric-relative-v2",
        reference_adapter="none",
        include_dynamic=True,
        generic_version=GENERIC_FEATURE_VERSION,
        change_detector=DARSIA_CHANGE_DETECTOR,
    ),
}
CHANGE_ABLATION_VARIANT = "generic-no-darsia-change"
CHANGE_ABLATION_CONTRACT = OpticalFeatureContract(
    chemistry_adapter="none",
    reference_adapter="none",
    include_dynamic=True,
    generic_version=GENERIC_FEATURE_VERSION,
    change_detector=NO_CHANGE_DETECTOR,
)
MAXIMAL_CONTRACT = MODEL_VARIANTS["btb-ratiometric-relative-v2"]


def model_variants(
    *, include_change_ablation: bool = False
) -> dict[str, OpticalFeatureContract]:
    """Return production variants plus an optional non-deployable ablation."""

    variants = dict(MODEL_VARIANTS)
    if include_change_ablation:
        variants[CHANGE_ABLATION_VARIANT] = CHANGE_ABLATION_CONTRACT
    return variants


@dataclass(frozen=True)
class ReviewBundle:
    decisions_path: Path
    combined_manifest_path: Path


@dataclass(frozen=True)
class RunSource:
    run: str
    dataset: str
    review_root: Path
    asset_manifest_path: Path
    decisions_path: Path
    combined_manifest_path: Path


@dataclass(frozen=True)
class StagedRun:
    run: str
    dataset: str
    root: Path
    asset_manifest_path: Path
    baseline_path: Path
    frame_paths: tuple[Path, ...]
    asset_sha256: str


@dataclass(frozen=True)
class ExtractionTask:
    staged: StagedRun
    regions: tuple[ReviewedRegion, ...]
    output_path: Path
    metadata_path: Path
    cache_key: str
    pixels_per_region: int
    texture_sigma: float
    opencv_threads: int
    radiometry_profile: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _status(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _source_priority(region: ReviewedRegion) -> tuple[int, int, str, str]:
    source = 1 if region.source == "manual" else 0
    confidence = {"high": 2, "medium": 1, "low": 0}.get(region.confidence, 0)
    timestamp = str(
        region.payload.get("updated_utc") or region.payload.get("answered_utc") or ""
    )
    return source, confidence, timestamp, region.region_id


def _geometry_key(run: str, item: Mapping[str, Any]) -> str:
    points = item.get("points") or []
    if points:
        geometry: Any = [
            [round(float(point[0]), 7), round(float(point[1]), 7)] for point in points
        ]
    else:
        geometry = [
            round(float(item.get(name) or 0.0), 7)
            for name in ("x", "y", "width", "height")
        ]
    return _canonical_hash(
        {
            "run": run,
            "frame_index": int(item.get("frame_index") or 0),
            "geometry": geometry,
        }
    )


def _load_sources_and_regions(
    bundles: Sequence[ReviewBundle],
) -> tuple[list[RunSource], list[ReviewedRegion], list[dict[str, Any]]]:
    sources: dict[str, RunSource] = {}
    selected_by_geometry: dict[str, ReviewedRegion] = {}
    conflicts: list[dict[str, Any]] = []

    for bundle in bundles:
        decisions_payload = _read_json(bundle.decisions_path)
        combined = _read_json(bundle.combined_manifest_path)
        jobs_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for job in combined.get("jobs") or []:
            jobs_by_run[str(job.get("run") or "").lower()].append(job)

        for run, decision in sorted((decisions_payload.get("decisions") or {}).items()):
            normalized_run = str(run).lower()
            jobs = jobs_by_run.get(normalized_run) or []
            if not jobs:
                raise RuntimeError(
                    f"No review job maps {normalized_run} in {bundle.combined_manifest_path}"
                )
            dataset = str(jobs[0].get("dataset") or "unknown").lower()
            review_root = Path(str(jobs[0].get("analysis_source_review_root") or jobs[0]["review_output_dir"]))
            manifest_path = (
                review_root / "native_frames" / normalized_run / "manifest.json"
            )
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            from .radiometry import analysis_review_root
            review_root = analysis_review_root(review_root, normalized_run)
            manifest_path = review_root / "native_frames" / normalized_run / "manifest.json"
            source = RunSource(
                run=normalized_run,
                dataset=dataset,
                review_root=review_root,
                asset_manifest_path=manifest_path,
                decisions_path=Path(bundle.decisions_path),
                combined_manifest_path=Path(bundle.combined_manifest_path),
            )
            existing_source = sources.get(normalized_run)
            if existing_source is not None and existing_source.dataset != dataset:
                raise RuntimeError(
                    f"Conflicting dataset assignment for {normalized_run}"
                )
            sources[normalized_run] = source

            reviewer = str(decision.get("reviewer") or "unknown")
            records = (
                ("manual", decision.get("annotations") or []),
                ("quiz", decision.get("phase_quiz_answers") or []),
            )
            for review_source, items in records:
                for item in items:
                    phase_class = canonical_phase_class(item, review_source)
                    if phase_class is None or phase_class == "ignored":
                        continue
                    confidence = str(item.get("confidence") or "low").lower()
                    identifier = str(
                        item.get("id")
                        or item.get("proposal_id")
                        or _geometry_key(normalized_run, item)[:16]
                    )
                    region = ReviewedRegion(
                        region_id=f"{normalized_run}:{review_source}:{identifier}",
                        run=normalized_run,
                        dataset=dataset,
                        frame_index=int(item.get("frame_index") or 0),
                        elapsed_h=float(item.get("elapsed_h") or 0.0),
                        phase_class=phase_class,
                        confidence=confidence,
                        source=review_source,
                        reviewer=str(item.get("reviewer") or reviewer),
                        payload=dict(item),
                    )
                    key = _geometry_key(normalized_run, item)
                    incumbent = selected_by_geometry.get(key)
                    if incumbent is None:
                        selected_by_geometry[key] = region
                        continue
                    if incumbent.phase_class != region.phase_class:
                        conflicts.append(
                            {
                                "geometry_key": key,
                                "run": normalized_run,
                                "kept": incumbent.region_id,
                                "kept_class": incumbent.phase_class,
                                "candidate": region.region_id,
                                "candidate_class": region.phase_class,
                            }
                        )
                    if _source_priority(region) > _source_priority(incumbent):
                        selected_by_geometry[key] = region

    regions = sorted(
        selected_by_geometry.values(),
        key=lambda region: (
            region.dataset,
            region.run,
            region.frame_index,
            region.region_id,
        ),
    )
    return sorted(sources.values(), key=lambda value: value.run), regions, conflicts


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    if not destination.is_file() or destination.stat().st_size != source_stat.st_size:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return {
        "name": destination.name,
        "bytes": int(destination.stat().st_size),
        "sha256": _sha256(destination),
    }


def _stage_run(source: RunSource, cache_root: Path) -> StagedRun:
    manifest_sha = _sha256(source.asset_manifest_path)
    local_root = (
        Path(cache_root)
        / "assets"
        / f"{source.dataset}_{source.run}_{manifest_sha[:16]}"
    )
    local_manifest = local_root / "manifest.json"
    local_baseline = local_root / "pre_injection.jpg"
    source_baseline = (
        source.review_root / "native_references" / source.run / "pre_injection.jpg"
    )
    manifest = _read_json(source.asset_manifest_path)
    frame_count = int(manifest.get("frame_count") or 0)
    source_frame_root = source.review_root / "native_frames" / source.run
    local_frames = tuple(
        local_root / f"frame_{index:03d}.jpg" for index in range(frame_count)
    )
    source_frames = tuple(
        source_frame_root / f"frame_{index:03d}.jpg" for index in range(frame_count)
    )
    complete_path = local_root / "staged_assets.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        required = (local_manifest, local_baseline, *local_frames)
        if str(complete.get("source_manifest_sha256")) == manifest_sha and all(
            path.is_file() for path in required
        ):
            return StagedRun(
                run=source.run,
                dataset=source.dataset,
                root=local_root,
                asset_manifest_path=local_manifest,
                baseline_path=local_baseline,
                frame_paths=local_frames,
                asset_sha256=str(complete["asset_sha256"]),
            )

    local_root.mkdir(parents=True, exist_ok=True)
    files = [
        _copy_file(source.asset_manifest_path, local_manifest),
        _copy_file(source_baseline, local_baseline),
    ]
    for source_frame, local_frame in zip(source_frames, local_frames, strict=True):
        files.append(_copy_file(source_frame, local_frame))
    asset_sha = _canonical_hash(files)
    _write_json(
        complete_path,
        {
            "schema": 1,
            "run": source.run,
            "dataset": source.dataset,
            "source_review_root": str(source.review_root),
            "source_manifest_sha256": manifest_sha,
            "asset_sha256": asset_sha,
            "files": files,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return StagedRun(
        run=source.run,
        dataset=source.dataset,
        root=local_root,
        asset_manifest_path=local_manifest,
        baseline_path=local_baseline,
        frame_paths=local_frames,
        asset_sha256=asset_sha,
    )


def _extract_run_features(task: ExtractionTask) -> dict[str, Any]:
    cv2.setNumThreads(max(1, int(task.opencv_threads)))
    if task.output_path.is_file() and task.metadata_path.is_file():
        metadata = _read_json(task.metadata_path)
        if str(metadata.get("cache_key")) == task.cache_key:
            if _sha256(task.output_path) != metadata.get("features_sha256"):
                raise ValueError(f"Corrupt feature cache for {task.staged.dataset}/{task.staged.run}")
            if task.radiometry_profile is not None and (metadata.get("radiometry") or {}).get("profile") != task.radiometry_profile:
                raise ValueError("Feature cache belongs to a different radiometric profile")
            return {**metadata, "cache_hit": True, "seconds_current": 0.0}

    started = time.perf_counter()
    manifest = _read_json(task.staged.asset_manifest_path)
    canvas_wh = tuple(int(value) for value in manifest["canvas_wh"])
    content_box = tuple(int(value) for value in manifest["content_box_xywh"])
    radiometry = None
    if task.radiometry_profile is not None:
        from .radiometry import RunRadiometry
        radiometry = RunRadiometry(
            task.radiometry_profile, series_id=task.staged.dataset,
            run_id=task.staged.run, baseline_path=task.staged.baseline_path,
            content_box=content_box, cache_root=task.staged.root / "uniform24",
            source_manifest=task.staged.asset_manifest_path,
        )
        baseline = radiometry.load(task.staged.baseline_path, "pre_injection")
    else:
        baseline = _load_rgb(task.staged.baseline_path, content_box)
    by_frame: dict[int, list[tuple[int, ReviewedRegion]]] = defaultdict(list)
    for local_index, region in enumerate(task.regions):
        by_frame[int(region.frame_index)].append((local_index, region))

    feature_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    for frame_index in sorted(by_frame):
        current = (
            _load_rgb(task.staged.frame_paths[frame_index], content_box)
            if radiometry is None else radiometry.load(task.staged.frame_paths[frame_index], f"frame_{frame_index:03d}")
        )
        previous = None
        if frame_index > 0:
            previous = (
                _load_rgb(task.staged.frame_paths[frame_index - 1], content_box)
                if radiometry is None else radiometry.load(task.staged.frame_paths[frame_index - 1], f"frame_{frame_index - 1:03d}")
            )

        row_chunks: list[np.ndarray] = []
        column_chunks: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for local_index, region in by_frame[frame_index]:
            mask = _annotation_mask(
                region.payload,
                canvas_wh=canvas_wh,
                content_box=content_box,
            )
            rows, columns = _sample_mask(
                mask,
                limit=max(1, int(task.pixels_per_region)),
                annotation_id=region.region_id,
            )
            row_chunks.append(rows)
            column_chunks.append(columns)
            owners.append(np.full(rows.size, local_index, dtype=np.int32))
        all_rows = np.concatenate(row_chunks)
        all_columns = np.concatenate(column_chunks)
        extracted = feature_samples(
            current,
            baseline,
            all_rows,
            all_columns,
            texture_sigma=float(task.texture_sigma),
            previous_rgb=previous,
            feature_contract=MAXIMAL_CONTRACT,
        )
        feature_chunks.append(np.asarray(extracted, dtype=np.float32))
        owner_chunks.append(np.concatenate(owners))
        frame_rows.append(
            {
                "frame_index": int(frame_index),
                "regions": len(by_frame[frame_index]),
                "pixels": int(all_rows.size),
            }
        )
        del current, previous, extracted, all_rows, all_columns
        gc.collect()

    matrix = np.vstack(feature_chunks)
    region_indices = np.concatenate(owner_chunks)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = task.output_path.with_suffix(task.output_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            features=np.asarray(matrix, dtype=np.float32),
            region_indices=np.asarray(region_indices, dtype=np.int32),
            feature_names=np.asarray(MAXIMAL_CONTRACT.feature_names),
        )
    temporary.replace(task.output_path)
    metadata = {
        "schema": 1,
        "run": task.staged.run,
        "dataset": task.staged.dataset,
        "cache_key": task.cache_key,
        "radiometry": None if radiometry is None else radiometry.report(),
        "asset_sha256": task.staged.asset_sha256,
        "features_path": str(task.output_path),
        "features_sha256": _sha256(task.output_path),
        "region_count": len(task.regions),
        "pixel_rows": int(matrix.shape[0]),
        "feature_columns": int(matrix.shape[1]),
        "pixels_per_region": int(task.pixels_per_region),
        "texture_sigma": float(task.texture_sigma),
        "opencv_threads": int(task.opencv_threads),
        "frames": frame_rows,
        "seconds": float(time.perf_counter() - started),
        "cache_hit": False,
        "seconds_current": float(time.perf_counter() - started),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(task.metadata_path, metadata)
    return metadata


def _feature_cache_key(
    staged: StagedRun,
    regions: Sequence[ReviewedRegion],
    *,
    pixels_per_region: int,
    texture_sigma: float,
    radiometry_profile: dict[str, Any] | None = None,
) -> str:
    region_contract = [
        {
            "region_id": region.region_id,
            "phase_class": region.phase_class,
            "confidence": region.confidence,
            "source": region.source,
            "geometry": _geometry_key(region.run, region.payload),
        }
        for region in regions
    ]
    return _canonical_hash(
        {
            "asset_sha256": staged.asset_sha256,
            "regions": region_contract,
            "pixels_per_region": int(pixels_per_region),
            "texture_sigma": float(texture_sigma),
            "feature_contract": MAXIMAL_CONTRACT.to_dict(),
            "optical_phase_features_sha256": _sha256(
                FLUIDFLOWER_ROOT / "phase_features.py"
            ),
            "dual_phase_signal_sha256": _sha256(PACKAGE_ROOT / "model.py"),
            "annotation_geometry_sha256": _sha256(PACKAGE_ROOT / "review_regions.py"),
            # Model fitting, validation, reporting, and backend changes must
            # not invalidate expensive optical feature extraction.
            "feature_extraction_contract_version": 1,
            "radiometry_profile": radiometry_profile,
            "radiometry_runtime": _radiometry_runtime(radiometry_profile),
        }
    )


def _load_feature_caches(
    sources: Sequence[RunSource],
    regions: Sequence[ReviewedRegion],
    *,
    cache_root: Path,
    stage_workers: int,
    extraction_workers: int,
    opencv_threads: int,
    pixels_per_region: int,
    texture_sigma: float,
    radiometry_profile: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ReviewedRegion], list[dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=max(1, int(stage_workers))) as executor:
        future_to_source = {
            executor.submit(_stage_run, source, cache_root): source
            for source in sources
        }
        staged = []
        for future in as_completed(future_to_source):
            result = future.result()
            staged.append(result)
            _status(f"Staged native corrected assets for {result.run}")
    staged.sort(key=lambda value: value.run)

    tasks: list[ExtractionTask] = []
    regions_by_run: dict[str, tuple[ReviewedRegion, ...]] = {}
    for staged_run in staged:
        run_regions = tuple(
            region for region in regions if region.run == staged_run.run
        )
        if not run_regions:
            continue
        regions_by_run[staged_run.run] = run_regions
        cache_key = _feature_cache_key(
            staged_run,
            run_regions,
            pixels_per_region=pixels_per_region,
            texture_sigma=texture_sigma,
            radiometry_profile=radiometry_profile,
        )
        run_cache = Path(cache_root) / "features" / staged_run.run / cache_key[:20]
        tasks.append(
            ExtractionTask(
                staged=staged_run,
                regions=run_regions,
                output_path=run_cache / "samples.npz",
                metadata_path=run_cache / "metadata.json",
                cache_key=cache_key,
                pixels_per_region=int(pixels_per_region),
                texture_sigma=float(texture_sigma),
                opencv_threads=int(opencv_threads),
                radiometry_profile=radiometry_profile,
            )
        )

    context = multiprocessing.get_context("spawn")
    reports: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, min(int(extraction_workers), len(tasks))),
        mp_context=context,
    ) as executor:
        future_to_task = {
            executor.submit(_extract_run_features, task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            report = future.result()
            reports.append(report)
            _status(
                f"Feature extraction ready for {report['run']}: "
                f"{report['pixel_rows']} rows"
                + (
                    " (cache hit)"
                    if report.get("cache_hit")
                    else f" in {report['seconds_current']:.1f} s"
                )
            )
    reports.sort(key=lambda row: str(row["run"]))

    combined_features: list[np.ndarray] = []
    combined_owners: list[np.ndarray] = []
    combined_regions: list[ReviewedRegion] = []
    for task in sorted(tasks, key=lambda value: value.staged.run):
        with np.load(task.output_path, allow_pickle=False) as archive:
            features = np.asarray(archive["features"], dtype=np.float32)
            local_owners = np.asarray(archive["region_indices"], dtype=np.int32)
        offset = len(combined_regions)
        combined_features.append(features)
        combined_owners.append(local_owners + offset)
        combined_regions.extend(task.regions)
    return (
        np.vstack(combined_features),
        np.concatenate(combined_owners),
        combined_regions,
        reports,
    )


def _variant_matrix(
    maximal_features: np.ndarray,
    contract: OpticalFeatureContract,
) -> np.ndarray:
    maximal_names = list(MAXIMAL_CONTRACT.feature_names)
    columns = [maximal_names.index(name) for name in contract.feature_names]
    return np.asarray(maximal_features[:, columns], dtype=np.float32)


def _fit_model_object(
    fit: LogisticFit,
    *,
    feature_contract: OpticalFeatureContract,
    provenance: Mapping[str, Any],
    texture_sigma: float,
) -> DualPhaseSignalModel:
    return DualPhaseSignalModel(
        feature_mean=tuple(float(value) for value in fit.mean),
        feature_scale=tuple(float(value) for value in fit.scale),
        coefficients=tuple(float(value) for value in fit.coefficients),
        intercept=float(fit.intercept),
        texture_sigma=float(texture_sigma),
        training_provenance=dict(provenance),
        feature_names=feature_contract.feature_names,
        feature_contract=feature_contract.to_dict(),
        schema=TRANSFERABLE_MODEL_SCHEMA,
        algorithm=TRANSFERABLE_MODEL_ALGORITHM,
    )


def _parity_check(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
    limit: int,
) -> dict[str, Any]:
    if not cuda_available():
        return {
            "status": "skipped",
            "reason": "CUDA/CuPy unavailable",
            "passed": False,
        }
    rng = np.random.default_rng(20260830)
    selected_chunks: list[np.ndarray] = []
    per_class = max(1000, int(limit) // 2)
    for value in (0, 1):
        candidates = np.flatnonzero(labels == value)
        if candidates.size > per_class:
            candidates = np.sort(rng.choice(candidates, per_class, replace=False))
        selected_chunks.append(candidates)
    selected = np.concatenate(selected_chunks)
    subset = matrix[selected]
    subset_labels = labels[selected]
    subset_weights = weights[selected]
    cpu = fit_weighted_logistic(
        subset,
        subset_labels,
        subset_weights,
        regularization=regularization,
        backend="cpu",
    )
    cuda = fit_weighted_logistic(
        subset,
        subset_labels,
        subset_weights,
        regularization=regularization,
        backend="cuda",
        normalization=(cpu.mean, cpu.scale),
    )
    cpu_probability = cpu.predict(subset)
    cuda_probability = cuda.predict(subset)
    difference = np.abs(cpu_probability - cuda_probability)
    agreement = float(np.mean((cpu_probability >= 0.5) == (cuda_probability >= 0.5)))
    report = {
        "status": "ok",
        "rows": int(subset.shape[0]),
        "cpu_precision": cpu.precision,
        "cuda_precision": cuda.precision,
        "median_probability_abs_difference": float(np.median(difference)),
        "p99_probability_abs_difference": float(np.quantile(difference, 0.99)),
        "max_probability_abs_difference": float(np.max(difference)),
        "classification_agreement_at_0p5": agreement,
    }
    report["passed"] = bool(
        report["p99_probability_abs_difference"] <= 0.005
        and report["max_probability_abs_difference"] <= 0.03
        and agreement >= 0.995
    )
    return report


def _inference_parity_check(
    fit: LogisticFit,
    matrix: np.ndarray,
    *,
    limit: int,
) -> dict[str, Any]:
    if not cuda_available():
        return {
            "status": "skipped",
            "reason": "CUDA/CuPy unavailable",
            "passed": False,
        }
    rng = np.random.default_rng(20260830)
    if matrix.shape[0] > int(limit):
        selected = np.sort(rng.choice(matrix.shape[0], int(limit), replace=False))
        subset = matrix[selected]
    else:
        subset = matrix
    cpu_probability = fit.predict(subset)
    cuda_probability = predict_logistic(fit, subset, backend="cuda")
    difference = np.abs(cpu_probability - cuda_probability)
    agreement = float(np.mean((cpu_probability >= 0.5) == (cuda_probability >= 0.5)))
    report = {
        "status": "ok",
        "kind": "serialized-model-cpu64-cuda32-inference-parity",
        "rows": int(subset.shape[0]),
        "median_probability_abs_difference": float(np.median(difference)),
        "p99_probability_abs_difference": float(np.quantile(difference, 0.99)),
        "max_probability_abs_difference": float(np.max(difference)),
        "classification_agreement_at_0p5": agreement,
    }
    report["passed"] = bool(
        report["p99_probability_abs_difference"] <= 0.005
        and report["max_probability_abs_difference"] <= 0.03
        and agreement >= 0.995
    )
    return report


def _eligible_runs_for_head(
    regions: Sequence[ReviewedRegion],
    head: str,
) -> list[str]:
    return sorted(
        {
            region.run
            for region in regions
            if regional_presence_target(region.phase_class, head) is not None
        }
    )


def _cross_validate_head(
    matrix: np.ndarray,
    region_indices: np.ndarray,
    regions: Sequence[ReviewedRegion],
    *,
    head: str,
    regularization_grid: Sequence[float],
    backend: str,
    fit_mode: str = "pure",
    weak_strength: float = 0.35,
    positive_top_fraction: float = 0.10,
    mil_max_iters: int = 160,
) -> tuple[dict[str, Any], LogisticFit, float, dict[str, Any]]:
    runs = _eligible_runs_for_head(regions, head)
    if len(runs) < 2:
        raise ValueError(
            f"{head} requires regional evaluation labels from at least two runs; "
            f"found {runs}"
        )
    grid_records: list[dict[str, Any]] = []
    score_cache: dict[tuple[float, str], np.ndarray] = {}
    for regularization in regularization_grid:
        fold_rows: list[dict[str, Any]] = []
        for held_out in runs:
            training_runs = set(runs) - {held_out}
            train_matrix, train_labels, train_weights, _owners = balanced_pixel_view(
                matrix,
                region_indices,
                regions,
                head=head,
                runs=training_runs,
            )
            if fit_mode == "mil":
                fit, fit_diagnostics = fit_weighted_logistic_mil(
                    matrix,
                    region_indices,
                    regions,
                    head=head,
                    runs=training_runs,
                    regularization=float(regularization),
                    backend=backend,
                    weak_strength=float(weak_strength),
                    positive_top_fraction=float(positive_top_fraction),
                    max_iter=int(mil_max_iters),
                )
            elif fit_mode == "pure":
                fit = fit_weighted_logistic(
                    train_matrix,
                    train_labels,
                    train_weights,
                    regularization=float(regularization),
                    backend=backend,
                )
                fit_diagnostics = {"mode": "pure-region-irls", "weak_bags": 0}
            else:
                raise ValueError(f"Unsupported fit mode {fit_mode!r}")
            probabilities = predict_logistic(fit, matrix, backend=backend)
            scores = region_probability_scores(
                probabilities,
                region_indices,
                len(regions),
            )
            score_cache[(float(regularization), held_out)] = scores
            train_scores, train_targets, train_region_weights, _ = (
                regional_evaluation_view(
                    scores,
                    regions,
                    head=head,
                    runs=training_runs,
                )
            )
            threshold, _ = tune_presence_threshold(
                train_scores,
                train_targets,
                train_region_weights,
            )
            held_scores, held_targets, held_weights, held_indices = (
                regional_evaluation_view(
                    scores,
                    regions,
                    head=head,
                    runs={held_out},
                )
            )
            metrics = regional_metrics(
                held_scores,
                held_targets,
                held_weights,
                threshold=threshold,
            )
            fold_rows.append(
                {
                    "held_out_run": held_out,
                    "held_out_dataset": regions[int(held_indices[0])].dataset,
                    "fit_backend": fit.backend,
                    "fit_precision": fit.precision,
                    "fit_iterations": fit.iterations,
                    "fit_converged": fit.converged,
                    "fit_diagnostics": fit_diagnostics,
                    **metrics,
                }
            )
        balanced = [float(row["balanced_accuracy"]) for row in fold_rows]
        auc = [float(row["roc_auc"]) for row in fold_rows]
        alpha_oof_scores = np.full(len(regions), np.nan, dtype=np.float64)
        for held_out in runs:
            held_scores = score_cache[(float(regularization), held_out)]
            for index, region in enumerate(regions):
                if region.run == held_out:
                    alpha_oof_scores[index] = held_scores[index]
        all_scores, all_targets, all_weights, _ = regional_evaluation_view(
            alpha_oof_scores,
            regions,
            head=head,
            runs=set(runs),
        )
        global_threshold, _ = tune_presence_threshold(
            all_scores,
            all_targets,
            all_weights,
        )
        global_metrics = regional_metrics(
            all_scores,
            all_targets,
            all_weights,
            threshold=global_threshold,
        )
        global_run_metrics: list[dict[str, Any]] = []
        for held_out in runs:
            run_scores, run_targets, run_weights, _ = regional_evaluation_view(
                alpha_oof_scores,
                regions,
                head=head,
                runs={held_out},
            )
            global_run_metrics.append(
                {
                    "held_out_run": held_out,
                    **regional_metrics(
                        run_scores,
                        run_targets,
                        run_weights,
                        threshold=global_threshold,
                    ),
                }
            )
        global_balanced = [
            float(row["balanced_accuracy"]) for row in global_run_metrics
        ]
        grid_records.append(
            {
                "regularization": float(regularization),
                "mean_balanced_accuracy": float(np.mean(balanced)),
                "minimum_balanced_accuracy": float(np.min(balanced)),
                "mean_roc_auc": float(np.nanmean(auc)),
                "global_oof_threshold": float(global_threshold),
                "global_oof_metrics": global_metrics,
                "global_threshold_run_metrics": global_run_metrics,
                "minimum_global_threshold_balanced_accuracy": float(
                    np.min(global_balanced)
                ),
                "mean_global_threshold_balanced_accuracy": float(
                    np.mean(global_balanced)
                ),
                "fold_threshold_range": float(
                    np.max([float(row["threshold"]) for row in fold_rows])
                    - np.min([float(row["threshold"]) for row in fold_rows])
                ),
                "folds": fold_rows,
            }
        )
        _status(
            f"{head}: regularization={regularization:g}, "
            f"global-threshold mean={np.mean(global_balanced):.3f}, "
            f"worst={np.min(global_balanced):.3f}"
        )

    selected_record = max(
        grid_records,
        key=lambda row: (
            float(row["minimum_global_threshold_balanced_accuracy"]),
            float(row["mean_global_threshold_balanced_accuracy"]),
            -float(row["fold_threshold_range"]),
            float(row["mean_roc_auc"]),
            -abs(np.log10(float(row["regularization"])) + 3.0),
        ),
    )
    selected_regularization = float(selected_record["regularization"])

    oof_scores = np.full(len(regions), np.nan, dtype=np.float64)
    for held_out in runs:
        fold_scores = score_cache[(selected_regularization, held_out)]
        for index, region in enumerate(regions):
            if region.run == held_out:
                oof_scores[index] = fold_scores[index]
    eval_scores, eval_targets, eval_weights, _ = regional_evaluation_view(
        oof_scores,
        regions,
        head=head,
        runs=set(runs),
    )
    final_threshold, threshold_accuracy = tune_presence_threshold(
        eval_scores,
        eval_targets,
        eval_weights,
    )
    oof_metrics = regional_metrics(
        eval_scores,
        eval_targets,
        eval_weights,
        threshold=final_threshold,
    )

    full_matrix, full_labels, full_weights, _ = balanced_pixel_view(
        matrix,
        region_indices,
        regions,
        head=head,
        runs=set(runs),
    )
    if fit_mode == "mil":
        final_fit, final_fit_diagnostics = fit_weighted_logistic_mil(
            matrix,
            region_indices,
            regions,
            head=head,
            runs=set(runs),
            regularization=selected_regularization,
            backend=backend,
            weak_strength=float(weak_strength),
            positive_top_fraction=float(positive_top_fraction),
            max_iter=int(mil_max_iters),
        )
        parity = _inference_parity_check(final_fit, full_matrix, limit=120_000)
    else:
        parity = _parity_check(
            full_matrix,
            full_labels,
            full_weights,
            regularization=selected_regularization,
            limit=120_000,
        )
        final_fit = fit_weighted_logistic(
            full_matrix,
            full_labels,
            full_weights,
            regularization=selected_regularization,
            backend=backend,
        )
        final_fit_diagnostics = {"mode": "pure-region-irls", "weak_bags": 0}
    report = {
        "head": head,
        "eligible_runs": runs,
        "fit_mode": fit_mode,
        "selection_policy": (
            "maximize the worst held-out balanced accuracy with one shared "
            "global OOF threshold, then mean accuracy, threshold stability, "
            "and ROC AUC"
        ),
        "regularization_grid": grid_records,
        "selected_regularization": selected_regularization,
        "oof_threshold": float(final_threshold),
        "oof_threshold_balanced_accuracy": float(threshold_accuracy),
        "oof_metrics": oof_metrics,
        "cpu64_cuda32_parity": parity,
        "final_fit": {
            "backend": final_fit.backend,
            "precision": final_fit.precision,
            "iterations": final_fit.iterations,
            "converged": final_fit.converged,
            "objective": final_fit.objective,
            "pixel_rows": int(full_matrix.shape[0]),
            "positive_pixel_fraction": float(np.mean(full_labels)),
            "diagnostics": final_fit_diagnostics,
        },
    }
    return report, final_fit, final_threshold, parity


def _write_region_inventory(path: Path, regions: Sequence[ReviewedRegion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "region_index",
                "region_id",
                "dataset",
                "run",
                "frame_index",
                "elapsed_h",
                "phase_class",
                "confidence",
                "source",
                "reviewer",
                "supervision_weight",
            ),
        )
        writer.writeheader()
        for index, region in enumerate(regions):
            writer.writerow(
                {
                    "region_index": index,
                    "region_id": region.region_id,
                    "dataset": region.dataset,
                    "run": region.run,
                    "frame_index": region.frame_index,
                    "elapsed_h": region.elapsed_h,
                    "phase_class": region.phase_class,
                    "confidence": region.confidence,
                    "source": region.source,
                    "reviewer": region.reviewer,
                    "supervision_weight": region.supervision_weight,
                }
            )


def _sync_result(local_result: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for source in local_result.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(local_result)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def _load_reusable_head(
    path: Path,
    *,
    head: str,
    fit_mode: str,
    inventory: Mapping[str, Any],
) -> tuple[DualPhaseSignalModel, dict[str, Any], float, dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        model = DualPhaseSignalModel.load(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    provenance = model.training_provenance
    if (
        provenance.get("campaign_id") != CAMPAIGN_ID
        or provenance.get("target_kind") != f"visible_{head}_phase_probability"
        or provenance.get("fit_mode") != fit_mode
        or provenance.get("training_inventory") != _json_safe(inventory)
    ):
        return None
    report = provenance.get("cross_validation")
    backend_contract = provenance.get("backend_contract")
    if not isinstance(report, Mapping) or not isinstance(backend_contract, Mapping):
        return None
    parity = backend_contract.get("parity")
    if not isinstance(parity, Mapping):
        return None
    threshold = report.get("oof_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        return None
    restored_report = dict(report)
    restored_report["deployment_allowed"] = bool(provenance.get("deployment_allowed"))
    restored_report["model_path"] = str(path)
    restored_report["model_sha256"] = model.to_dict()["model_sha256"]
    restored_report["reused_from_verified_staging_checkpoint"] = True
    return model, restored_report, float(threshold), dict(parity)


def _selected_grid_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = float(report.get("selected_regularization", float("nan")))
    records = list(report.get("regularization_grid") or [])
    for record in records:
        if np.isclose(float(record.get("regularization", float("nan"))), selected):
            return record
    return {}


def change_evidence_ablation_report(
    model_reports: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare full-run holdout metrics with and without DarSIA change evidence."""

    reference = dict(model_reports.get("generic") or {})
    ablated = dict(model_reports.get(CHANGE_ABLATION_VARIANT) or {})
    if not reference or not ablated:
        return {
            "status": "not_requested",
            "reference_variant": "generic",
            "ablated_variant": CHANGE_ABLATION_VARIANT,
        }
    heads: dict[str, Any] = {}
    balanced_deltas: list[float] = []
    reference_heads = dict(reference.get("heads") or {})
    ablated_heads = dict(ablated.get("heads") or {})
    for head in sorted(set(reference_heads).intersection(ablated_heads)):
        full_report = dict(reference_heads[head])
        ablated_report = dict(ablated_heads[head])
        full_metrics = dict(full_report.get("oof_metrics") or {})
        ablated_metrics = dict(ablated_report.get("oof_metrics") or {})
        metric_delta: dict[str, float | None] = {}
        for metric in ("balanced_accuracy", "roc_auc", "average_precision"):
            left = full_metrics.get(metric)
            right = ablated_metrics.get(metric)
            metric_delta[metric] = (
                None
                if not isinstance(left, (int, float))
                or not isinstance(right, (int, float))
                or not np.isfinite(left)
                or not np.isfinite(right)
                else float(left) - float(right)
            )
        if metric_delta["balanced_accuracy"] is not None:
            balanced_deltas.append(float(metric_delta["balanced_accuracy"]))
        full_runs = {
            str(row["held_out_run"]): row
            for row in _selected_grid_record(full_report).get(
                "global_threshold_run_metrics", []
            )
        }
        ablated_runs = {
            str(row["held_out_run"]): row
            for row in _selected_grid_record(ablated_report).get(
                "global_threshold_run_metrics", []
            )
        }
        holdouts = []
        for run in sorted(set(full_runs).intersection(ablated_runs)):
            full_score = float(full_runs[run]["balanced_accuracy"])
            ablated_score = float(ablated_runs[run]["balanced_accuracy"])
            holdouts.append(
                {
                    "run": run,
                    "with_darsia_change": full_score,
                    "without_darsia_change": ablated_score,
                    "delta": full_score - ablated_score,
                }
            )
        heads[head] = {
            "with_darsia_change": full_metrics,
            "without_darsia_change": ablated_metrics,
            "delta_with_minus_without": metric_delta,
            "leave_one_run_out": holdouts,
        }
    mean_delta = float(np.mean(balanced_deltas)) if balanced_deltas else None
    return {
        "status": "ok",
        "reference_variant": "generic",
        "ablated_variant": CHANGE_ABLATION_VARIANT,
        "scope": "registered visible 2D optical classification only",
        "validation": "leave-one-complete-run-out",
        "heads": heads,
        "mean_balanced_accuracy_delta": mean_delta,
        "interpretation": (
            "Positive values support retaining DarSIA change evidence; this report "
            "does not evaluate concentration, depth, saturation, or mass."
        ),
    }


def _radiometry_runtime(profile):
    if profile is None:
        return None
    from .radiometry import runtime_identity
    return runtime_identity()


def train_campaign(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    bundles = [
        ReviewBundle(Path(decisions), Path(manifest))
        for decisions, manifest in args.bundle
    ]
    sources, regions, conflicts = _load_sources_and_regions(bundles)
    series_filter = {str(value).casefold().replace("_", "") for value in getattr(args, "series", []) or []}
    if series_filter:
        sources = [s for s in sources if s.dataset.casefold().replace("_", "") in series_filter]
        regions = [r for r in regions if r.dataset.casefold().replace("_", "") in series_filter]
    if len({s.run for s in sources}) != len(sources):
        raise ValueError("Run IDs collide across series; use distinct run IDs before training")
    from .radiometry import resolve_profile
    radiometry_profile = resolve_profile(getattr(args, "radiometry_profile", None))
    if len(sources) < 4:
        raise RuntimeError(
            "Cross-series v3 training requires at least four reviewed runs"
        )
    run_names = sorted(source.run for source in sources)
    _status(
        f"Loaded {len(regions)} deduplicated reviewed regions from "
        f"{', '.join(run_names)}"
    )
    local_result = Path(args.cache_root) / "result_staging"
    if local_result.exists() and not args.resume_completed_heads:
        shutil.rmtree(local_result)
    local_result.mkdir(parents=True, exist_ok=True)
    if radiometry_profile is not None:
        _write_json(local_result / "radiometry_profile.json", radiometry_profile)

    maximal_features, region_indices, ordered_regions, extraction_reports = (
        _load_feature_caches(
            sources,
            regions,
            cache_root=Path(args.cache_root),
            stage_workers=args.stage_workers,
            extraction_workers=args.extraction_workers,
            opencv_threads=args.opencv_threads,
            pixels_per_region=args.pixels_per_region,
            texture_sigma=args.texture_sigma,
            radiometry_profile=radiometry_profile,
        )
    )
    _status(
        f"Combined feature matrix: {maximal_features.shape[0]} x "
        f"{maximal_features.shape[1]}"
    )
    _write_region_inventory(local_result / "region_inventory.csv", ordered_regions)
    with (local_result / "training_samples.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            features=np.asarray(maximal_features, dtype=np.float32),
            region_indices=np.asarray(region_indices, dtype=np.int32),
            feature_names=np.asarray(MAXIMAL_CONTRACT.feature_names),
        )

    inventory = {
        "radiometry_profile": radiometry_profile,
        "series_filter": sorted(series_filter),
        "regions_by_series_run": dict(Counter(f"{r.dataset}/{r.run}" for r in ordered_regions)),
        "regions": len(ordered_regions),
        "pixel_rows": int(maximal_features.shape[0]),
        "runs": run_names,
        "datasets": sorted({region.dataset for region in ordered_regions}),
        "phase_classes": dict(
            Counter(region.phase_class for region in ordered_regions)
        ),
        "confidence": dict(Counter(region.confidence for region in ordered_regions)),
        "sources": dict(Counter(region.source for region in ordered_regions)),
        "reviewers": dict(Counter(region.reviewer for region in ordered_regions)),
        "deduplication_conflicts": conflicts,
    }
    _write_json(local_result / "training_inventory.json", inventory)

    model_reports: dict[str, Any] = {}
    reused_heads: list[str] = []
    selected_variants = model_variants(
        include_change_ablation=bool(getattr(args, "change_evidence_ablation", False))
    )
    decision_policy: dict[str, Any] = {
        "schema": 1,
        "region_score": "90th percentile of pixel probability",
        "models": {},
    }
    for variant, feature_contract in selected_variants.items():
        matrix = _variant_matrix(maximal_features, feature_contract)
        variant_root = local_result / "models" / variant
        variant_report: dict[str, Any] = {
            "feature_contract": feature_contract.to_dict(),
            "heads": {},
        }
        decision_policy["models"][variant] = {}
        for head in args.heads:
            model_path = variant_root / f"{head}_signal.json"
            reusable = (
                _load_reusable_head(
                    model_path,
                    head=head,
                    fit_mode=args.fit_mode,
                    inventory=inventory,
                )
                if args.resume_completed_heads
                else None
            )
            if reusable is not None:
                model, report, threshold, _parity = reusable
                variant_report["heads"][head] = report
                decision_policy["models"][variant][head] = {
                    "presence_threshold": float(threshold),
                    "model_sha256": model.to_dict()["model_sha256"],
                }
                reused_heads.append(f"{variant}/{head}")
                _status(f"Reused verified staging checkpoint for {variant}/{head}")
                continue
            _status(f"Cross-validating {variant}/{head} on {args.backend}")
            report, fit, threshold, parity = _cross_validate_head(
                matrix,
                region_indices,
                ordered_regions,
                head=head,
                regularization_grid=args.regularization_grid,
                backend=args.backend,
                fit_mode=args.fit_mode,
                weak_strength=args.weak_strength,
                positive_top_fraction=args.positive_top_fraction,
                mil_max_iters=args.mil_max_iters,
            )
            deployment_allowed = bool(
                feature_contract.change_detector != NO_CHANGE_DETECTOR
                and parity.get("passed")
                and float(report["oof_metrics"]["balanced_accuracy"])
                >= float(args.minimum_balanced_accuracy)
            )
            provenance = {
                "radiometry_profile": radiometry_profile,
                "radiometry_runtime": _radiometry_runtime(radiometry_profile),
                "campaign_id": CAMPAIGN_ID,
                "target_kind": f"visible_{head}_phase_probability",
                "fit_mode": args.fit_mode,
                "deployment_allowed": deployment_allowed,
                "deployment_scope": (
                    "Radiometrically corrected RGB frames with this exact "
                    "feature contract; output is optical probability, not mass "
                    "or saturation."
                ),
                "runs": run_names,
                "head_training_runs": list(report["eligible_runs"]),
                "datasets": sorted({source.dataset for source in sources}),
                "reviewers": sorted({region.reviewer for region in ordered_regions}),
                "training_inventory": inventory,
                "cross_validation": _json_safe(report),
                "backend_contract": {
                    "production_backend": fit.backend,
                    "production_precision": fit.precision,
                    "authoritative_reference": "cpu-float64",
                    "parity": _json_safe(parity),
                },
                "source_sha256": {
                    "trainer": _sha256(Path(__file__)),
                    "training_primitives": _sha256(PACKAGE_ROOT / "logistic.py"),
                    "optical_features": _sha256(FLUIDFLOWER_ROOT / "phase_features.py"),
                },
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            model = _fit_model_object(
                fit,
                feature_contract=feature_contract,
                provenance=provenance,
                texture_sigma=args.texture_sigma,
            )
            model_path = model.save(model_path)
            report["deployment_allowed"] = deployment_allowed
            report["model_path"] = str(model_path)
            report["model_sha256"] = model.to_dict()["model_sha256"]
            variant_report["heads"][head] = report
            decision_policy["models"][variant][head] = {
                "presence_threshold": float(threshold),
                "model_sha256": model.to_dict()["model_sha256"],
            }
            _status(
                f"Finished {variant}/{head}: OOF balanced accuracy "
                f"{report['oof_metrics']['balanced_accuracy']:.3f}, "
                f"deployment_allowed={deployment_allowed}"
            )
        model_reports[variant] = variant_report

    ablation = change_evidence_ablation_report(model_reports)
    _write_json(local_result / "cross_validation.json", _json_safe(model_reports))
    _write_json(local_result / "change_evidence_ablation.json", _json_safe(ablation))
    _write_json(local_result / "phase_decision_policy.json", decision_policy)
    source_snapshot = [
        {
            "decisions": str(bundle.decisions_path),
            "decisions_sha256": _sha256(bundle.decisions_path),
            "combined_manifest": str(bundle.combined_manifest_path),
            "combined_manifest_sha256": _sha256(bundle.combined_manifest_path),
        }
        for bundle in bundles
    ]
    summary = {
        "radiometry_profile": radiometry_profile,
        "radiometry_runtime": _radiometry_runtime(radiometry_profile),
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend_requested": args.backend,
        "cuda_available": cuda_available(),
        "runs": run_names,
        "inventory": inventory,
        "source_snapshot": source_snapshot,
        "feature_extraction": extraction_reports,
        "models": model_reports,
        "change_evidence_ablation": ablation,
        "reused_heads": reused_heads,
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(local_result / "summary.json", _json_safe(summary))
    _write_json(local_result / "radiometry_inventory.json", {
        "profile": radiometry_profile,
        "series_filter": sorted(series_filter),
        "runs": [{"series_id": r["dataset"], "run_id": r["run"],
                  "source_asset_sha256": r["asset_sha256"],
                  "radiometry": r.get("radiometry")} for r in extraction_reports],
    })
    _status(f"Syncing final result to {args.output_root}")
    _sync_result(local_result, Path(args.output_root))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id",
        default=CAMPAIGN_ID,
        help="Immutable model-version identifier recorded in every model head.",
    )
    parser.add_argument(
        "--bundle",
        nargs=2,
        action="append",
        metavar=("DECISIONS_JSON", "COMBINED_MANIFEST_JSON"),
        help="Repeat for each frozen review bundle.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--radiometry-profile", default="legacy",
                        help="image:PATH, uniform24-after2014, a frozen profile JSON, or explicit legacy input")
    parser.add_argument("--series", nargs="+", help="Optional series subset, e.g. FF_AC FF_UM")
    parser.add_argument("--pixels-per-region", type=int, default=1024)
    parser.add_argument("--texture-sigma", type=float, default=2.0)
    parser.add_argument("--stage-workers", type=int, default=4)
    parser.add_argument("--extraction-workers", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int, default=6)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--regularization-grid",
        nargs="+",
        type=float,
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    )
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.65)
    parser.add_argument(
        "--change-evidence-ablation",
        action="store_true",
        help=(
            "Also fit a non-deployable generic control without DarSIA structured "
            "change-evidence channels and compare complete-run holdouts."
        ),
    )
    parser.add_argument(
        "--fit-mode",
        choices=("pure", "mil"),
        default="pure",
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=("gas", "aqueous", "co2_presence", "gas_within_co2"),
        default=("gas", "aqueous", "co2_presence", "gas_within_co2"),
    )
    parser.add_argument("--weak-strength", type=float, default=0.35)
    parser.add_argument("--positive-top-fraction", type=float, default=0.10)
    parser.add_argument("--mil-max-iters", type=int, default=160)
    parser.add_argument(
        "--resume-completed-heads",
        action="store_true",
        help=(
            "Reuse a valid per-head model in local result_staging when its "
            "campaign, fit mode, target, and training inventory match exactly."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    global CAMPAIGN_ID
    parser = build_parser()
    args = parser.parse_args(argv)
    CAMPAIGN_ID = str(args.campaign_id).strip() or CAMPAIGN_ID
    if not args.bundle:
        args.bundle = [
            (str(decisions), str(manifest)) for decisions, manifest in DEFAULT_BUNDLES
        ]
    summary = train_campaign(args)
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "output_root": str(Path(args.output_root).resolve()),
                "runs": summary["runs"],
                "regions": summary["inventory"]["regions"],
                "pixel_rows": summary["inventory"]["pixel_rows"],
                "seconds": summary["seconds"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
