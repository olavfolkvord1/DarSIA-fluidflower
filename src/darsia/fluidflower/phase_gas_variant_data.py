"""Source-exact native RAW samples shared by the four isolated gas variants.

No held-out review payload or prediction is consumed here. Historical labels
retain their original source revisions; the sealed new freehand labels use
their own current native source manifests. Generic and flat-field features
are evaluated together at the same deterministic native pixel coordinates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .phase_features import OpticalFeatureContract
from .phase_gas_flatfield import (
    FLATFIELD_FEATURE_NAMES, extract_flatfield_features, flatfield_contract,
    prepare_flatfield_reference, register_flatfield_reference,
)
from .phase_training import trainer as t
from .project import sha256_file


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(t._json_safe(document), indent=2,
                                   ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _manifest_staged(source):
    """Use immutable existing sidecars rather than copy entire source series."""
    manifest = _read(source.asset_manifest_path)
    if (manifest.get("v5_photometry") or {}).get("format") != (
            "ffac-v5-native-source-materialization-v1"):
        raise ValueError("Gas variants require source-bound float RAW sidecars")
    reference = manifest["reference"]
    frames = manifest["frame_sources"]
    def path(row, key):
        value = row.get(key)
        return Path(value) if value else None
    baseline = path(reference, "v5_log_linear_rgb_path")
    if baseline is None or not frames:
        raise ValueError("Native manifest is missing scientific RAW input")
    frame_paths = tuple(path(row, "v5_log_linear_rgb_path") for row in frames)
    if any(value is None for value in frame_paths):
        raise ValueError("A native frame is missing its scientific RAW sidecar")
    valid_paths = tuple(path(row, "v5_log_linear_valid_mask_path") for row in frames)
    if manifest.get("v5_float_support_policy") == "exact_per_frame_sha_mask_v1":
        if (path(reference, "v5_log_linear_valid_mask_path") is None
                or any(value is None for value in valid_paths)):
            raise ValueError("Exact native RAW support requires every source validity mask")
    if any(value is None for value in valid_paths):
        valid_paths = ()
    return t.StagedRun(
        run=source.run, dataset=source.dataset, root=source.review_root,
        asset_manifest_path=source.asset_manifest_path, baseline_path=baseline,
        frame_paths=frame_paths, asset_sha256=sha256_file(source.asset_manifest_path),
        independent_reference_frames=source.independent_reference_frames,
        baseline_valid_mask_path=path(reference, "v5_log_linear_valid_mask_path"),
        frame_valid_mask_paths=valid_paths,
    )


class _FlatfieldAugmenter:
    feature_names = FLATFIELD_FEATURE_NAMES

    def __init__(self, staged):
        self.staged = staged
        self.reference = None

    def __call__(self, *, frame_index, rows, columns, current_source_valid,
                 baseline_source_valid, frame_geometry):
        if frame_geometry is None:
            raise ValueError("Gas features require source-exact per-frame registration")
        if self.reference is None:
            log = np.load(self.staged.baseline_path, allow_pickle=False, mmap_mode="r")
            self.reference = prepare_flatfield_reference(
                log, source_valid_mask=baseline_source_valid)
        log = np.load(self.staged.frame_paths[frame_index],
                      allow_pickle=False, mmap_mode="r")
        registered = register_flatfield_reference(
            self.reference, frame_geometry.H_v5_B_to_frame(frame_index),
            target_shape_hw=log.shape[1:])
        extra, usable = extract_flatfield_features(
            log, registered, current_valid_mask=current_source_valid,
            selection=(rows, columns))
        metadata = {
            "contract": flatfield_contract()["generic_version"],
            "requested_native_pixels": int(rows.size),
            "supported_native_pixels": int(np.count_nonzero(usable)),
            "unsupported_native_pixels": int(np.count_nonzero(~usable)),
            "same_rows_for_all_variants": True,
            "baseline_registration": registered.alignment,
        }
        return extra, usable, metadata


def _extract_task(task):
    return t._extract_run_features(task, feature_augmenter=_FlatfieldAugmenter(task.staged))


def extract_training_data(frozen_candidate, snapshot, output, *, workers=4,
                          source_bundle=None):
    """Extract one auditable, CPU-only sample set; never refit the CO2 head."""
    from .phase_gas_variant_sources import load_training_sources
    from .tools.train_ffac_gas_flatfield_variants import TrainingDataset

    started = time.perf_counter()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    frozen_candidate = Path(frozen_candidate)
    snapshot = Path(snapshot)
    source_bundle = (load_training_sources(frozen_candidate, snapshot)
                     if source_bundle is None else source_bundle)
    groups = source_bundle["groups"]
    gas = _read(frozen_candidate / "models/generic/gas_within_co2_signal.json")
    contract = OpticalFeatureContract.from_dict(gas["feature_contract"])
    names = tuple(contract.feature_names)
    provenance = source_bundle["receipt"]
    optical_request = Path(provenance["optical_parent"]) / "training_request.json"
    settings = _read(optical_request)["settings"]
    pixels_per_region = int(settings["pixels_per_region"])
    texture_sigma = float(settings.get("texture_sigma", 2))
    _write(root / "source_preflight.json", provenance)
    implementation = {
        "flatfield": flatfield_contract(),
        "extractor_sha256": sha256_file(Path(__file__)),
        "trainer_sha256": sha256_file(Path(t.__file__)),
        "source_resolver_sha256": sha256_file(Path(__file__).with_name(
            "phase_gas_variant_sources.py")),
    }
    _write(root / "feature_contract.json", implementation)
    tasks = []
    for index, (source, regions) in enumerate(groups):
        staged = _manifest_staged(source)
        key = hashlib.sha256(json.dumps({
            "manifest_sha256": staged.asset_sha256,
            "regions": [asdict(region) for region in regions],
            "generic_contract": contract.to_dict(),
            "pixels_per_region": pixels_per_region,
            "texture_sigma": texture_sigma,
            "implementation": implementation,
        }, sort_keys=True, default=str).encode()).hexdigest()
        directory = root / "feature_samples" / f"{index:03d}-{source.run}-{key[:16]}"
        tasks.append(t.ExtractionTask(
            staged=staged, regions=tuple(regions),
            output_path=directory / "samples.npz",
            metadata_path=directory / "metadata.json", cache_key=key,
            pixels_per_region=pixels_per_region, texture_sigma=texture_sigma,
            opencv_threads=1, feature_contract=contract,
        ))
    _write(root / "progress.json", {
        "stage": "extracting", "source_groups": len(tasks), "completed": 0,
        "device": "cpu", "workers": int(workers),
    })
    with ProcessPoolExecutor(
        max_workers=max(1, min(int(workers), len(tasks))),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=t._initialize_extraction_worker, initargs=(1,),
    ) as pool:
        futures = {pool.submit(_extract_task, task): task for task in tasks}
        reports = []
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            _write(root / "progress.json", {
                "stage": "extracting", "source_groups": len(tasks),
                "completed": len(reports), "last_training_run": report["run"],
                "seconds": time.perf_counter() - started, "device": "cpu",
            })
            print(f"Training RAW features: {len(reports)}/{len(tasks)} groups ready", flush=True)
    features, owners, regions = [], [], []
    for task in tasks:
        with np.load(task.output_path, allow_pickle=False) as archive:
            matrix = archive["features"]
            sample_names = tuple(str(name) for name in archive["feature_names"])
            if (matrix.dtype != np.float32 or not np.isfinite(matrix).all()
                    or sample_names != (*names, *FLATFIELD_FEATURE_NAMES)):
                raise ValueError("Shared gas samples differ from the declared feature contract")
            features.append(matrix)
            owners.append(archive["region_indices"].astype(np.int32) + len(regions))
        regions.extend(task.regions)
    matrix = np.vstack(features)
    region_indices = np.concatenate(owners)
    # No optimizer reads a holdout run, a zero-weight label or an unmarked
    # pixel. The established MIL implementation interprets explicit regions.
    _write(root / "regions.json", [asdict(region) for region in regions])
    sample_file = root / "training_samples.npz"
    with sample_file.open("wb") as stream:
        np.savez(stream, current_features=matrix[:, :len(names)],
                 flatfield_features=matrix[:, len(names):],
                 region_indices=region_indices,
                 feature_names=np.asarray(names))
    provenance = {
        **provenance, "implementation": implementation,
        "sampling": {"pixels_per_region": pixels_per_region,
                     "texture_sigma": texture_sigma, "native_resolution": True,
                     "zero_unlabelled_negative_pixels": True},
        "source_feature_reports": sorted(reports, key=lambda r: r["features_path"]),
        "training_samples_sha256": sha256_file(sample_file),
        "region_inventory_sha256": sha256_file(root / "regions.json"),
        "extraction_seconds": time.perf_counter() - started,
    }
    _write(root / "extraction_receipt.json", provenance)
    _write(root / "progress.json", {
        "stage": "extracted", "source_groups": len(tasks),
        "completed": len(tasks), "seconds": time.perf_counter() - started,
    })
    return TrainingDataset(
        current_features=matrix[:, :len(names)], flatfield_features=matrix[:, len(names):],
        region_indices=region_indices, regions=regions,
        new_region_ids=frozenset(source_bundle["new_region_ids"]),
        provenance=provenance, feature_names=names)


def load_extracted_data(output):
    """Resume a completed source-bound extraction without reading labels again."""
    from .tools.train_ffac_gas_flatfield_variants import TrainingDataset
    from .phase_training.logistic import ReviewedRegion
    root = Path(output)
    receipt = _read(root / "extraction_receipt.json")
    if (sha256_file(root / "training_samples.npz") != receipt["training_samples_sha256"]
            or sha256_file(root / "regions.json") != receipt["region_inventory_sha256"]):
        raise ValueError("Source-exact gas extraction changed after sealing")
    with np.load(root / "training_samples.npz", allow_pickle=False) as archive:
        current = archive["current_features"]
        flat = archive["flatfield_features"]
        owners = archive["region_indices"]
        names = tuple(str(name) for name in archive["feature_names"])
    regions = [ReviewedRegion(**row) for row in _read(root / "regions.json")]
    return TrainingDataset(
        current_features=current, flatfield_features=flat, region_indices=owners,
        regions=regions, feature_names=names, provenance=receipt,
        new_region_ids=frozenset(receipt["new_region_ids"]))
