"""Isolated CPU gas-head variants with frozen CO2 support and blind test outputs.

This experiment is deliberately separate from Studio's two-head model loader:
G2 and G3 have a gas-specific feature contract, while the CO2 file is copied
byte for byte.  No publication or full-cohort inference entry point exists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from darsia.fluidflower.phase_training import logistic
from darsia.fluidflower.project import sha256_file


CONTRACT = "ffac-gas-head-flatfield-variants-v1"
VARIANTS = ("G0prime", "G1", "G2", "G3")
HELD_OUT_RUNS = frozenset({"ac48", "ac53", "ac35", "ac38", "ac59", "ac22",
                          "ac46", "ac57", "ac33"})
INFERENCE_RUNS = HELD_OUT_RUNS | {"ac28"}
FLATFIELD_NAMES = ("ff_log_r", "ff_log_g", "ff_log_b", "ff_log_y")
ANNOTATION_SHA256 = "56c0a0477e244f4ad9b15f5370b8d97ec9ddecf058a3053084426f4d589c7dbc"
HEAD = "gas_within_co2"


@dataclass(frozen=True)
class TrainingDataset:
    current_features: np.ndarray
    flatfield_features: np.ndarray
    region_indices: np.ndarray
    regions: Sequence[logistic.ReviewedRegion]
    new_region_ids: frozenset[str]
    provenance: Mapping[str, Any]
    feature_names: tuple[str, ...]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def validate_dataset(data: TrainingDataset) -> None:
    current, flat, owners = (np.asarray(data.current_features),
                             np.asarray(data.flatfield_features),
                             np.asarray(data.region_indices))
    if (current.dtype != np.float32 or current.ndim != 2 or not current.size
            or current.shape[1] != len(data.feature_names)
            or flat.dtype != np.float32 or flat.shape != (current.shape[0], 4)
            or owners.dtype != np.int32 or owners.shape != (current.shape[0],)
            or not np.isfinite(current).all() or not np.isfinite(flat).all()):
        raise ValueError("Gas training requires finite, aligned float32 feature matrices")
    if not data.regions or owners.min() < 0 or owners.max() >= len(data.regions):
        raise ValueError("Gas training sample ownership is invalid")
    identifiers = [item.region_id for item in data.regions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gas training region identifiers must be unique")
    if not data.new_region_ids.issubset(identifiers):
        raise ValueError("New gas label identifiers are absent from the inventory")
    leaked = sorted({item.run.casefold() for item in data.regions} & HELD_OUT_RUNS)
    if leaked:
        raise ValueError("Whole-run held-out labels reached training: " + ", ".join(leaked))
    for item in data.regions:
        if item.region_id in data.new_region_ids and item.phase_class != "gas":
            raise ValueError("Confirmed-only free gas outlines must be full-positive gas regions")
        if item.payload.get("evaluation_only") is True:
            raise ValueError("Evaluation-only label reached gas training")
    logistic.assert_safe_training_regions(data.regions)


def variant_training_view(data: TrainingDataset, variant: str):
    """Preserve MIL region meaning; unlabelled pixels never become negatives."""
    validate_dataset(data)
    if variant not in VARIANTS:
        raise ValueError("Unknown gas variant")
    selected = [index for index, region in enumerate(data.regions)
                if variant != "G0prime" or region.region_id not in data.new_region_ids]
    translation = np.full(len(data.regions), -1, dtype=np.int32)
    translation[selected] = np.arange(len(selected), dtype=np.int32)
    rows = translation[data.region_indices] >= 0
    old = np.asarray(data.current_features[rows], np.float32)
    flat = np.asarray(data.flatfield_features[rows], np.float32)
    if variant in ("G0prime", "G1"):
        features, names = old, data.feature_names
    elif variant == "G2":
        features, names = np.concatenate((old, flat), axis=1), (*data.feature_names, *FLATFIELD_NAMES)
    else:
        features, names = flat, FLATFIELD_NAMES
    regions = [data.regions[index] for index in selected]
    return features, translation[data.region_indices[rows]], regions, tuple(names)


def training_counts(regions, owners) -> dict:
    rows_per_region = np.bincount(owners, minlength=len(regions))
    per_run = defaultdict(lambda: defaultdict(lambda: {"regions": 0, "sample_rows": 0}))
    for index, region in enumerate(regions):
        group = per_run[region.run.casefold()][region.phase_class]
        group["regions"] += 1
        group["sample_rows"] += int(rows_per_region[index])
    return {run: dict(classes) for run, classes in sorted(per_run.items())}


def frozen_fit_settings(candidate: Path) -> dict:
    source = read_json(candidate / "models/generic/gas_within_co2_signal.json")
    provenance = source["training_provenance"]
    checkpoint = provenance["fit_checkpoint_contract"]
    if checkpoint["fit_mode"] != "mil":
        raise ValueError("Frozen gas head was not trained with the required MIL fit")
    return dict(regularization=float(provenance["cross_validation"]["selected_regularization"]),
                weak_strength=float(checkpoint["weak_strength"]),
                positive_top_fraction=float(checkpoint["positive_top_fraction"]),
                max_iter=int(checkpoint["mil_max_iters"]), tolerance=2e-5,
                backend="cpu")


def _fit_worker(variant, sample_paths, regions, names, settings, threads):
    from threadpoolctl import threadpool_limits
    import cv2
    cv2.setNumThreads(threads)
    old = np.load(sample_paths["current"], mmap_mode="r", allow_pickle=False)
    flat = np.load(sample_paths["flatfield"], mmap_mode="r", allow_pickle=False)
    owners = np.load(sample_paths["owners"], mmap_mode="r", allow_pickle=False)
    data = TrainingDataset(old, flat, owners, regions, frozenset(sample_paths["new_region_ids"]),
                           {}, tuple(names))
    matrix, selected_owners, selected_regions, feature_names = variant_training_view(data, variant)
    started = time.perf_counter()
    with threadpool_limits(limits=threads):
        fit, diagnostics = logistic.fit_weighted_logistic_mil(
            matrix, selected_owners, selected_regions, head=HEAD, runs=None, **settings)
    seconds = time.perf_counter() - started
    payload = dict(mean=fit.mean.tolist(), scale=fit.scale.tolist(),
                   coefficients=fit.coefficients.tolist(), intercept=float(fit.intercept),
                   backend=fit.backend, precision=fit.precision, iterations=fit.iterations,
                   converged=fit.converged, objective=float(fit.objective),
                   regularization=fit.regularization,
                   convergence_certificate=fit.convergence_certificate)
    labels = [dict(region_id=r.region_id, run=r.run.casefold(), phase_class=r.phase_class,
                   confidence=r.confidence, source=r.source, payload=dict(r.payload))
              for r in selected_regions]
    return dict(variant=variant, feature_names=list(feature_names), fit=payload,
                diagnostics=diagnostics, fit_seconds=seconds,
                training_counts=training_counts(selected_regions, selected_owners),
                training_runs=sorted({r.run.casefold() for r in selected_regions}),
                training_label_sha256=canonical_sha256(labels),
                training_region_ids=[r.region_id for r in selected_regions],
                new_label_count=sum(r.region_id in data.new_region_ids for r in selected_regions))


def fit_variants(data: TrainingDataset, frozen_candidate: Path, output: Path, *,
                 workers: int = 4, threads_per_worker: int = 6) -> dict:
    """Fit exactly four new gas heads; freeze all original CO2 and regional files."""
    validate_dataset(data)
    candidate, output = Path(frozen_candidate).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(candidate) or candidate.is_relative_to(output):
        raise ValueError("Variant output must be an absent detached folder")
    if len(data.new_region_ids) != 871 or data.provenance.get("new_annotation_sha256") != ANNOTATION_SHA256:
        raise ValueError("Gas variants require the exact sealed 871-outline snapshot")
    if not 1 <= workers <= 4 or not 1 <= threads_per_worker <= 32:
        raise ValueError("Invalid bounded CPU fit allocation")
    source_gas = read_json(candidate / "models/generic/gas_within_co2_signal.json")
    if tuple(source_gas["feature_names"]) != data.feature_names:
        raise ValueError("Original gas feature names differ from extracted generic features")
    frozen = {name: sha256_file(candidate / name) for name in (
        "models/generic/co2_presence_signal.json", "models/generic/gas_within_co2_signal.json",
        "regional_model.json", "phase_optical_regions.json")}
    settings = frozen_fit_settings(candidate)
    output.mkdir(parents=True, exist_ok=False)
    samples = output / "fit_samples"
    samples.mkdir()
    paths = {}
    for name, array in (("current", data.current_features), ("flatfield", data.flatfield_features),
                        ("owners", data.region_indices)):
        path = samples / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        paths[name] = str(path)
    paths["new_region_ids"] = sorted(data.new_region_ids)
    write_new_json(samples / "provenance.json", dict(data.provenance))
    started = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_fit_worker, variant, paths, data.regions, data.feature_names,
                               settings, threads_per_worker): variant for variant in VARIANTS}
        for future in as_completed(futures):
            result = future.result()
            variant = result["variant"]
            folder = output / "variants" / variant
            for name in ("models/generic/co2_presence_signal.json", "regional_model.json",
                         "phase_optical_regions.json"):
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate / name, target)
                if sha256_file(target) != frozen[name]:
                    raise ValueError("Frozen model artifact copy differs")
            feature_version = ("baseline-relative-bright-grain-v5+flatfield-v1" if variant == "G2"
                               else "flatfield-gas-v1" if variant == "G3"
                               else "baseline-relative-bright-grain-v5")
            provenance = dict(data.provenance, held_out_runs=sorted(HELD_OUT_RUNS),
                              frozen_co2_sha256=frozen["models/generic/co2_presence_signal.json"],
                              frozen_candidate=str(candidate), frozen_files_sha256=frozen,
                              fit_settings=settings, training_label_sha256=result["training_label_sha256"],
                              shared_sampling_policy="same-native-label-pixels-with-valid-old-and-flatfield-features",
                              unlabelled_pixels_are_negative=False,
                              inference_invalid_policy="retain-frozen-current-phase-at-invalid-flatfield-pixels",
                              code_sha256={"runner": sha256_file(Path(__file__)),
                                           "logistic": sha256_file(Path(logistic.__file__))})
            model = dict(format=CONTRACT, head=HEAD, variant=variant,
                         feature_contract=dict(version=feature_version, feature_names=result["feature_names"]),
                         feature_names=result["feature_names"], fit=result["fit"],
                         gas_display_threshold=.6, training_provenance=provenance)
            write_new_json(folder / "gas_head.json", model)
            write_new_json(folder / "training_report.json", result)
            results[variant] = dict(model_path=str(folder / "gas_head.json"),
                                    model_sha256=sha256_file(folder / "gas_head.json"),
                                    **{key: result[key] for key in ("fit_seconds", "training_counts",
                                        "training_runs", "training_label_sha256", "new_label_count")},
                                    converged=bool(result["fit"]["converged"]))
            print(json.dumps(dict(stage="gas_fit_complete", variant=variant,
                                  converged=results[variant]["converged"])), flush=True)
    receipt = dict(contract=CONTRACT, created_utc=datetime.now(timezone.utc).isoformat(),
                   stage="gas_fit_complete", frozen_candidate=str(candidate),
                   frozen_files_sha256=frozen, variants=results,
                   held_out_runs=sorted(HELD_OUT_RUNS), allowed_inference_runs=sorted(INFERENCE_RUNS),
                   cpu_only=True, workers=workers, threads_per_worker=threads_per_worker,
                   wall_seconds=time.perf_counter() - started,
                   test_outputs_visibility="sealed_for_Claude_until_labels_locked",
                   full_cohort_inference_authorized=False, published_to_studio=False)
    write_new_json(output / "fit_receipt.json", receipt)
    for name, digest in frozen.items():
        if sha256_file(candidate / name) != digest:
            raise ValueError("Original model artifact changed during gas fitting")
    if not all(item["converged"] for item in results.values()):
        raise ValueError("One or more gas fits did not converge; diagnostics are saved, inference is blocked")
    return receipt


def checked_inference_runs(runs: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(str(run).casefold() for run in runs)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(INFERENCE_RUNS):
        raise ValueError("Inference is restricted to AC28 and the nine held-out runs")
    return selected


def gas_probability(model: Mapping[str, Any], current: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    variant = model["variant"]
    features = (current if variant in ("G0prime", "G1") else flatfield if variant == "G3"
                else np.concatenate((current, flatfield), axis=-1))
    fit = model["fit"]
    values = np.asarray(features, np.float64)
    mean, scale, coefficients = (np.asarray(fit[key], np.float64)
                                 for key in ("mean", "scale", "coefficients"))
    if values.shape[-1] != len(model["feature_names"]) or scale.shape != coefficients.shape or (scale <= 0).any():
        raise ValueError("Gas-head feature dimensions or normalization differ")
    from scipy.special import expit
    return expit(((values - mean) / scale) @ coefficients + float(fit["intercept"])).astype(np.float32)


def replace_gas_partition(frozen_labels: np.ndarray, gas: np.ndarray, eligible: np.ndarray,
                          *, threshold: float = .6) -> np.ndarray:
    """Only relabel frozen CO2 pixels; outside support and invalid pixels are exact."""
    labels = np.asarray(frozen_labels)
    probability = np.asarray(gas, np.float32)
    valid = np.asarray(eligible, bool)
    if labels.ndim != 2 or probability.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("Frozen phase labels and new gas scores must share a grid")
    if not np.isin(labels, (0, 1, 2, 255)).all() or not np.isfinite(probability[valid]).all():
        raise ValueError("Invalid frozen labels or eligible gas probabilities")
    if ((probability[valid] < 0) | (probability[valid] > 1)).any():
        raise ValueError("Gas probabilities are outside the unit interval")
    changed = valid & np.isin(labels, (1, 2))
    result = labels.astype(np.uint8, copy=True)
    result[changed] = np.where(probability[changed] >= threshold, 2, 1)
    if not np.array_equal(np.isin(result, (1, 2)), np.isin(labels, (1, 2))):
        raise ValueError("Gas partition changed frozen CO2 support")
    return result


def frozen_inference_eligibility(domain, flatfield_valid, frozen_labels) -> np.ndarray:
    """Unresolved frozen pixels cannot enter confirmation or cleanup decisions."""
    domain = np.asarray(domain, bool)
    flatfield_valid = np.asarray(flatfield_valid, bool)
    frozen_labels = np.asarray(frozen_labels)
    if domain.shape != flatfield_valid.shape or domain.shape != frozen_labels.shape:
        raise ValueError("Frozen inference masks must share the native label grid")
    return domain & flatfield_valid & (frozen_labels != 255)


def finalize_gas_labels(frozen_labels, cleaned_labels, eligible) -> np.ndarray:
    """Restore immutable pixels after categorical-only cleanup, including NPZ labels."""
    frozen = np.asarray(frozen_labels, np.uint8)
    result = np.asarray(cleaned_labels, np.uint8).copy()
    valid = np.asarray(eligible, bool)
    if frozen.shape != result.shape or frozen.shape != valid.shape:
        raise ValueError("Final gas labels and frozen masks must share a grid")
    result[~valid] = frozen[~valid]
    if not np.array_equal(np.isin(result, (1, 2)), np.isin(frozen, (1, 2))):
        raise ValueError("Variant inference changed the frozen CO2 boundary")
    if (not np.array_equal(result[~valid], frozen[~valid])
            or not np.array_equal(result[frozen == 255], frozen[frozen == 255])):
        raise ValueError("Variant inference changed invalid or unsupported pixels")
    return result


def _read_frozen_map(path: Path, content_box) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
    x, y, width, height = map(int, content_box)
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA")).copy()
    local = rgba[y:y + height, x:x + width]
    labels = np.full((height, width), 255, np.uint8)
    labels[local[..., 3] == 0] = 0
    recognized = local[..., 3] == 0
    for phase, label in (("aqueous", 1), ("gas", 2), ("unresolved", 255)):
        match = np.all(local == np.asarray(PHASE_FILL_COLORS[phase], np.uint8), axis=-1)
        labels[match] = label
        recognized |= match
    if not recognized.all():
        raise ValueError("Frozen phase map contains an unsupported categorical colour")
    return labels, rgba


def _write_variant_map(path: Path, frozen_rgba, labels, content_box) -> None:
    from PIL import Image
    from darsia.fluidflower.phase_validation import PHASE_FILL_COLORS
    x, y, width, height = map(int, content_box)
    rgba = np.asarray(frozen_rgba, np.uint8).copy()
    local = rgba[y:y + height, x:x + width]
    for phase, label in (("aqueous", 1), ("gas", 2)):
        local[labels == label] = PHASE_FILL_COLORS[phase]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with Image.open(path) as image:
            if not np.array_equal(np.asarray(image.convert("RGBA")), rgba):
                raise ValueError("Existing variant output differs; never overwrite it")
        return
    with path.open("xb") as stream:
        Image.fromarray(rgba).save(stream, format="PNG", compress_level=1)


def _infer_run(candidate: str, output: str, run: str, threads: int,
               frame_indices: Sequence[int] | None = None,
               inference_output: str | None = None) -> dict:
    """One disjoint frame chunk owns RAW readers and evaluates all four gas heads."""
    import cv2
    from threadpoolctl import threadpool_limits
    from darsia.fluidflower import phase_gas_flatfield as flat
    from darsia.fluidflower import phase_optical_regions as regional
    from darsia.fluidflower.phase_gas_regional_context import (
        build_shared_regional_context, prepare_conditional_gas, shared_context_contract)
    from darsia.fluidflower.phase_gas_inference_chunks import previous_frame_indices
    from darsia.fluidflower.phase_free_gas_output_guard import clean_labels, Settings as GasMinimum
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
    from darsia.fluidflower.phase_regional_gas import confirm_gas, Settings as GasConfirmation
    from darsia.fluidflower.phase_training.v5_linear_input import RunV5LinearInput
    from darsia.fluidflower.tools.ffac_v5_temporal_alignment import align_temporal_inputs
    from darsia.fluidflower.tools.ffac_v5_track1_photo_context import Track1FramePhotoCohort
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import CpuFeaturePredictor
    from darsia.fluidflower.tools.train_phase_model_further import native_source
    from darsia.fluidflower.tools.train_phase_optical_regions import track1_frame_domain
    from scipy.special import expit

    cv2.setNumThreads(threads)
    candidate, output = Path(candidate), Path(output)
    selected = checked_inference_runs([run])[0]
    profile = read_json(candidate / "phase_optical_regions.json")
    if set(profile) - {"format", "model", "runs", "regional_gas", "free_gas_cleanup",
                       "v5_track1_review_scope", "v5_nominal_review_gas_minimum",
                       "v5_track1_incremental_inference"}:
        raise ValueError("Current candidate contains an unhandled downstream gas policy")
    regional_model = read_json(candidate / profile["model"]["path"])
    if sha256_file(candidate / profile["model"]["path"]) != profile["model"]["sha256"]:
        raise ValueError("Frozen regional model changed")
    models = {variant: read_json(output / "variants" / variant / "gas_head.json")
              for variant in VARIANTS}
    original = HierarchicalPhaseModel.load(candidate)
    package = candidate / "validation/run_packages" / selected
    package_manifest = read_json(package / "postflight_manifest.json")
    jobs = [job for job in package_manifest["jobs"] if job["run"].casefold() == selected]
    if len(jobs) != 1:
        raise ValueError("Frozen run package has no unique bound source job")
    job = jobs[0]
    source, native_path, native = native_source(job)
    count = len(native["frame_sources"])
    indices = tuple(range(count)) if frame_indices is None else tuple(frame_indices)
    if not indices or indices != tuple(range(indices[0], indices[-1] + 1)) or not 0 <= indices[0] <= indices[-1] < count:
        raise ValueError("Inference workers require one contiguous native frame chunk")
    independent = set(job.get("independent_reference_frames") or [])
    predecessors = previous_frame_indices(count, independent)
    binding = read_json(package / "validation_binding.json")
    frozen_maps = (Path(job["phase_map_output_dir"]) / "native_phase_maps" / selected /
                   ("A_" + binding["pipeline_sha256"][:16]))
    cohort = Track1FramePhotoCohort(native_path)
    shape = tuple(native["analysis_image_shape_hw"])
    reader = RunV5LinearInput(native_manifest_path=native_path, run_id=selected,
                              content_shape_hw=shape)
    baseline_photo = source / "native_references" / selected / "pre_injection.jpg"
    # This load verifies both the scientific reference and its display binding.
    baseline_core = reader.load(baseline_photo, "pre_injection")
    baseline_log_path = Path(reader.reference["v5_log_linear_rgb_path"])
    baseline_log = np.load(baseline_log_path, allow_pickle=False, mmap_mode="r")
    baseline_valid, _ = reader.load_validity(baseline_photo, "pre_injection")
    flat_reference = flat.prepare_flatfield_reference(baseline_log,
                                                      source_valid_mask=baseline_valid)
    temporal = (native.get("v5_temporal_reference") or {}).get("contract")
    if temporal == "ffac-v5-source-backed-extended-ring-v1":
        baseline, baseline_origin, baseline_valid_extended = reader.load_extended(
            baseline_photo, "pre_injection")
    elif temporal == "ffac-v5-native-crop-source-supported-v1":
        baseline, baseline_origin, baseline_valid_extended = baseline_core, (0, 0), baseline_valid
    else:
        raise ValueError("Frozen run has an unsupported temporal RAW source contract")
    predictor = None
    previous = None
    previous_index = predecessors[indices[0]]
    if previous_index is not None:
        previous_photo = source / "native_frames" / selected / f"frame_{previous_index:03d}.jpg"
        previous = reader.load(previous_photo, f"frame_{previous_index:03d}")
    frames = []
    namespace = Path(inference_output) if inference_output is not None else output
    destination = namespace / ("qa_ac28" if selected == "ac28" else "scoring_blind") / selected
    initial = dict(contract=CONTRACT, frozen_candidate=str(candidate), run=selected,
                   gas_models_sha256={variant: sha256_file(output / "variants" / variant / "gas_head.json")
                                      for variant in VARIANTS},
                   native_manifest_sha256=sha256_file(native_path))
    if destination.exists():
        if read_json(destination / "initial_binding.json") != initial:
            raise ValueError("Partial inference output belongs to a different source or model")
    elif frame_indices is not None:
        raise ValueError("The parent must bind the run namespace before submitting frame chunks")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_new_json(destination / "initial_binding.json", initial)
    started = time.perf_counter()
    with threadpool_limits(limits=threads):
        for index in indices:
            frame_source = native["frame_sources"][index]
            if previous_index != predecessors[index]:
                raise ValueError("Chunk predecessor differs from the original sequential RAW timeline")
            frame_started = time.perf_counter()
            current_photo = source / "native_frames" / selected / f"frame_{index:03d}.jpg"
            domains, spacing, domain_binding = track1_frame_domain(
                native, cohort, index, nominal_profile=profile, nominal_profile_root=candidate)
            feature_support, feature_receipt = cohort.feature_support(index, previous_index, reader)
            domain = domains["photo_certified_sand_domain"] & feature_support
            current = reader.load(current_photo, f"frame_{index:03d}",
                                  required_valid_mask=feature_support)
            current_log = np.load(Path(frame_source["v5_log_linear_rgb_path"]),
                                  allow_pickle=False, mmap_mode="r")
            current_valid, _ = reader.load_validity(current_photo, f"frame_{index:03d}")
            previous_origin, previous_valid = (0, 0), None
            previous_input = previous
            if previous_index is not None and temporal == "ffac-v5-source-backed-extended-ring-v1":
                previous_photo = source / "native_frames" / selected / f"frame_{previous_index:03d}.jpg"
                previous_input, previous_origin, previous_valid = reader.load_extended(
                    previous_photo, f"frame_{previous_index:03d}")
            aligned_baseline, aligned_previous, alignment = align_temporal_inputs(
                baseline=baseline, current=current, previous=previous_input,
                h_baseline_to_current=cohort.H_v5_B_to_frame(index),
                h_baseline_to_previous=(None if previous_index is None else
                                        cohort.H_v5_B_to_frame(previous_index)),
                required_valid_mask=feature_support,
                baseline_origin_xy=baseline_origin, previous_origin_xy=previous_origin,
                baseline_source_valid_mask=baseline_valid_extended,
                previous_source_valid_mask=previous_valid)
            if predictor is None:
                predictor = CpuFeaturePredictor(original, aligned_baseline,
                                                 block_rows=128, input_encoding="v5-unit-float32")
            else:
                predictor.set_baseline(aligned_baseline)
            registered_flat = flat.register_flatfield_reference(
                flat_reference, cohort.H_v5_B_to_frame(index), target_shape_hw=shape)
            frozen_path = frozen_maps / f"frame_{index:03d}.png"
            frozen_labels, frozen_rgba = _read_frozen_map(frozen_path, native["content_box_xywh"])
            frozen_map_sha = sha256_file(frozen_path)
            current_features = predictor.prepare_rgb(current)
            previous_features = (predictor.baseline if aligned_previous is None
                                 else predictor.prepare_rgb(aligned_previous))
            presence = np.empty(shape, np.float32)
            gas_fields = {variant: np.empty(shape, np.float32) for variant in VARIANTS}
            flat_valid = np.zeros(shape, bool)
            flat_tiles = iter(flat.iter_flatfield_feature_tiles(
                current_log, registered_flat, current_valid_mask=current_valid, tile_rows=128))
            for first, last, feature_block in predictor.feature_blocks(
                    current_features, previous_features, dynamic_available=aligned_previous is not None):
                ff_first, ff_last, flat_block, valid_block = next(flat_tiles)
                if (first, last) != (ff_first, ff_last):
                    raise ValueError("Old and flat-field feature row tiles differ")
                # The frozen current CO2 head is evaluated with its exact CPU
                # float32 packed expression, as in registered native inference.
                presence[first:last] = expit(feature_block @ predictor.weights[:, 0]
                    + predictor.bias[0]).reshape(last - first, shape[1])
                frozen_gas = expit(feature_block @ predictor.weights[:, 1]
                                   + predictor.bias[1]).reshape(last - first, shape[1])
                safe_flat = np.where(valid_block[..., None], flat_block, 0).reshape(-1, 4)
                flat_valid[first:last] = valid_block
                for variant, model in models.items():
                    new_gas = gas_probability(model, feature_block, safe_flat).reshape(
                        last - first, shape[1])
                    gas_fields[variant][first:last] = np.where(valid_block, new_gas, frozen_gas)
            del current_features, previous_features
            audits = {}
            # Require exactly the same support for every variant. Flat-field
            # invalid locations retain their frozen current phase labels.
            eligible = frozen_inference_eligibility(domain, flat_valid, frozen_labels)
            shared_regional = build_shared_regional_context(
                current, aligned_baseline, presence, domain, spacing,
                regional.Settings(**regional_model["settings"]))
            for variant, model in models.items():
                conditional = gas_fields[variant]
                prepared = prepare_conditional_gas(shared_regional, conditional)
                regional_score = regional.probability(
                    prepared["features"], regional_model["heads"][HEAD])
                native_gas = np.where(presence >= .5, conditional, 0).astype(np.float32)
                labels = replace_gas_partition(frozen_labels, native_gas, eligible,
                                                threshold=float(model["gas_display_threshold"]))
                safe_labels = np.where(labels == 255, 0, labels).astype(np.uint8)
                confirmed, rejected = confirm_gas(safe_labels, regional_score, eligible,
                                                   GasConfirmation(**profile["regional_gas"]))
                cleaned, cleanup = clean_labels(confirmed, spacing,
                                                 GasMinimum(**profile["free_gas_cleanup"]))
                cleaned = finalize_gas_labels(frozen_labels, cleaned, eligible)
                target = destination / variant / f"frame_{index:03d}.png"
                _write_variant_map(target, frozen_rgba, cleaned, native["content_box_xywh"])
                npz_path = target.with_suffix(".npz")
                if npz_path.is_file():
                    with np.load(npz_path, allow_pickle=False) as archived:
                        if (set(archived.files) != {"labels", "photo_sand", "feature_eligible"}
                                or not np.array_equal(archived["labels"], cleaned)
                                or not np.array_equal(archived["photo_sand"], domain)
                                or not np.array_equal(archived["feature_eligible"], eligible)):
                            raise ValueError("Existing variant label archive differs; never overwrite it")
                else:
                    with npz_path.open("xb") as stream:
                        np.savez_compressed(stream, labels=cleaned, photo_sand=domain,
                                            feature_eligible=eligible)
                audits[variant] = dict(map_path=str(target), map_sha256=sha256_file(target),
                                       labels_path=str(npz_path), labels_sha256=sha256_file(npz_path),
                                       co2_support_exact=True, unsupported_pixels_exact=True,
                                       regional_weights_frozen=True, gas_cleanup=cleanup)
            frames.append(dict(frame_index=index, elapsed_h=frame_source["elapsed_h"],
                               frozen_map_path=str(frozen_path), frozen_map_sha256=frozen_map_sha,
                               domain_binding=domain_binding,
                               feature_support_receipt_sha256=feature_receipt["receipt_sha256"],
                               alignment=alignment, variants=audits,
                               seconds=time.perf_counter() - frame_started))
            if index not in independent:
                previous, previous_index = current, index
            if selected == "ac28":
                print(json.dumps(dict(stage="qa_inference", run=selected, frame=index)), flush=True)
            del current, aligned_baseline, aligned_previous, conditional, prepared, shared_regional, gas_fields, frozen_rgba
    receipt = dict(contract=CONTRACT, run=selected, cpu_only=True,
                   frozen_candidate=str(candidate), native_manifest_path=str(native_path),
                   native_manifest_sha256=sha256_file(native_path),
                   downstream_policy="frozen-regional-gas-weights-and-native-gas-minimum",
                   co2_support_policy="exact-frozen-current-final-map-union",
                   invalid_flatfield_regional_input="frozen-original-gas-probability",
                   shared_regional_context=shared_context_contract(),
                   flatfield_contract=flat.flatfield_contract(), frames=frames,
                   wall_seconds=time.perf_counter() - started,
                   visibility="qa" if selected == "ac28" else "blind_until_Claude_seals_labels")
    filename = ("inference_receipt.json" if frame_indices is None else
                f"chunk_{indices[0]:03d}_{indices[-1]:03d}.json")
    receipt_path = destination / filename
    write_new_json(receipt_path, receipt)
    return dict(run=selected, status="complete", receipt_path=str(receipt_path))


def _verify_chunk_outputs(receipt, destination: Path, expected_indices) -> None:
    observed = [int(frame["frame_index"]) for frame in receipt["frames"]]
    if observed != list(expected_indices):
        raise ValueError("Inference receipt does not cover its exact assigned native frames")
    for frame in receipt["frames"]:
        if set(frame["variants"]) != set(VARIANTS):
            raise ValueError("Inference receipt omitted a gas variant")
        for details in frame["variants"].values():
            for path_key, sha_key in (("map_path", "map_sha256"), ("labels_path", "labels_sha256")):
                path = Path(details[path_key]).resolve()
                if not path.is_relative_to(destination.resolve()) or sha256_file(path) != details[sha_key]:
                    raise ValueError("Completed inference output changed or escaped its run namespace")


def infer_variants(frozen_candidate: Path, output: Path, *, runs: Sequence[str],
                   threads: int = 3, workers: int = 16,
                   inference_output: Path | None = None, chunk_frames: int = 4,
                   reuse_complete_runs: bool = True) -> dict:
    """Schedule disjoint source-frame chunks; keep every blind artifact isolated."""
    from darsia.fluidflower.phase_gas_inference_chunks import (
        frame_chunks, assert_complete_chunk_frames)
    from darsia.fluidflower.tools.train_phase_model_further import native_source
    selected = checked_inference_runs(runs)
    candidate, output = Path(frozen_candidate).resolve(), Path(output).resolve()
    namespace = output if inference_output is None else Path(inference_output).resolve()
    if (namespace.is_relative_to(candidate) or candidate.is_relative_to(namespace)
            or (namespace != output and not namespace.is_relative_to(output))):
        raise ValueError("Inference namespace must be detached from frozen inputs and inside the new experiment")
    if not 1 <= workers <= 16 or not 1 <= threads <= 32 or not 1 <= chunk_frames <= 32:
        raise ValueError("Invalid bounded CPU chunk inference allocation")
    fit_receipt = read_json(output / "fit_receipt.json")
    if not all(item["converged"] for item in fit_receipt["variants"].values()):
        raise ValueError("Unconverged gas variants cannot run inference")
    if fit_receipt["frozen_candidate"] != str(candidate):
        raise ValueError("Gas inference candidate differs from its frozen fit")
    for name, digest in fit_receipt["frozen_files_sha256"].items():
        if sha256_file(candidate / name) != digest:
            raise ValueError("Frozen model file changed before gas inference")
    for variant, details in fit_receipt["variants"].items():
        if sha256_file(Path(details["model_path"])) != details["model_sha256"]:
            raise ValueError("Trained gas variant changed before inference")
    model_shas = {variant: details["model_sha256"] for variant, details in fit_receipt["variants"].items()}
    completed, pending, run_plans = [], [], {}
    for run in selected:
        package = candidate / "validation/run_packages" / run
        jobs = [job for job in read_json(package / "postflight_manifest.json")["jobs"]
                if job["run"].casefold() == run]
        if len(jobs) != 1:
            raise ValueError("Frozen run package has no unique bound source job")
        _source, native_path, native = native_source(jobs[0])
        count = len(native["frame_sources"])
        chunks = frame_chunks(count, chunk_frames)
        destination = namespace / ("qa_ac28" if run == "ac28" else "scoring_blind") / run
        initial = dict(contract=CONTRACT, frozen_candidate=str(candidate), run=run,
                       gas_models_sha256=model_shas, native_manifest_sha256=sha256_file(native_path))
        if destination.exists():
            if read_json(destination / "initial_binding.json") != initial:
                raise ValueError("Inference namespace already binds a different run or gas model")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            write_new_json(destination / "initial_binding.json", initial)
        receipt_path = destination / "inference_receipt.json"
        run_plans[run] = dict(destination=destination, count=count, chunks=chunks, completed_chunks=[])
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt["native_manifest_sha256"] != initial["native_manifest_sha256"]:
                raise ValueError("Completed run receipt binds another native source")
            _verify_chunk_outputs(receipt, destination, range(count))
            completed.append(dict(run=run, status="already_complete", receipt_path=str(receipt_path)))
        else:
            old_destination = output / ("qa_ac28" if run == "ac28" else "scoring_blind") / run
            old_receipt_path = old_destination / "inference_receipt.json"
            if namespace != output and reuse_complete_runs and old_receipt_path.is_file():
                if read_json(old_destination / "initial_binding.json") != initial:
                    raise ValueError("Completed prior run binds a different frozen model or source")
                old_receipt = read_json(old_receipt_path)
                if (old_receipt["native_manifest_sha256"] != initial["native_manifest_sha256"]
                        or old_receipt["frozen_candidate"] != str(candidate)):
                    raise ValueError("Completed prior run receipt changed its frozen source")
                _verify_chunk_outputs(old_receipt, old_destination, range(count))
                reference = dict(run=run, status="complete", reused_complete_run=True,
                                 receipt_path=str(old_receipt_path),
                                 receipt_sha256=sha256_file(old_receipt_path))
                reused_path = destination / "reused_complete_run.json"
                if not reused_path.exists():
                    write_new_json(reused_path, reference)
                elif read_json(reused_path) != reference:
                    raise ValueError("Reused prior run binding changed")
                completed.append(reference)
                continue
            for indices in chunks:
                chunk_path = destination / f"chunk_{indices[0]:03d}_{indices[-1]:03d}.json"
                if chunk_path.is_file():
                    receipt = read_json(chunk_path)
                    if receipt["native_manifest_sha256"] != initial["native_manifest_sha256"]:
                        raise ValueError("Completed chunk binds another native source")
                    _verify_chunk_outputs(receipt, destination, indices)
                    run_plans[run]["completed_chunks"].append(receipt)
                else:
                    pending.append((run, indices))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(pending))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_infer_run, str(candidate), str(output), run, threads,
                               indices, str(namespace)) for run, indices in pending]
        for future in as_completed(futures):
            result = future.result()
            run = result["run"]
            receipt = read_json(Path(result["receipt_path"]))
            run_plans[run]["completed_chunks"].append(receipt)
            if run == "ac28":
                print(json.dumps(dict(stage="qa_chunk_complete", run=run)), flush=True)
            plan = run_plans[run]
            if len(plan["completed_chunks"]) == len(plan["chunks"]):
                assert_complete_chunk_frames(plan["count"], plan["completed_chunks"])
                template = dict(plan["completed_chunks"][0])
                frames = sorted((frame for chunk in plan["completed_chunks"] for frame in chunk["frames"]),
                                key=lambda frame: frame["frame_index"])
                template.update(frames=frames, frame_chunks=len(plan["chunks"]),
                                wall_seconds=time.perf_counter() - started,
                                sum_chunk_worker_seconds=sum(chunk["wall_seconds"] for chunk in plan["completed_chunks"]))
                final_path = plan["destination"] / "inference_receipt.json"
                write_new_json(final_path, template)
                completed.append(dict(run=run, status="complete", receipt_path=str(final_path)))
                print(json.dumps(dict(stage="inference_run_complete", run=run,
                                      visibility="qa" if run == "ac28" else "blind")), flush=True)
    # A fully resumed set of chunk receipts may need assembly without any
    # worker future, for example after interruption immediately before sealing.
    assembled = {row["run"] for row in completed}
    for run, plan in run_plans.items():
        if run in assembled:
            continue
        assert_complete_chunk_frames(plan["count"], plan["completed_chunks"])
        template = dict(plan["completed_chunks"][0])
        template.update(frames=sorted((frame for chunk in plan["completed_chunks"] for frame in chunk["frames"]),
                                      key=lambda frame: frame["frame_index"]),
                        frame_chunks=len(plan["chunks"]), wall_seconds=time.perf_counter() - started,
                        sum_chunk_worker_seconds=sum(chunk["wall_seconds"] for chunk in plan["completed_chunks"]))
        final_path = plan["destination"] / "inference_receipt.json"
        write_new_json(final_path, template)
        completed.append(dict(run=run, status="complete", receipt_path=str(final_path)))
    receipt = dict(contract=CONTRACT, stage="restricted_inference_complete", cpu_only=True,
                   inference_namespace=str(namespace), fit_root=str(output),
                   workers=workers, threads_per_worker=threads, chunk_frames=chunk_frames,
                   runs=sorted(completed, key=lambda row: row["run"]),
                   test_results_inspected=False, test_results_displayed=False,
                   studio_registration_performed=False,
                   full_cohort_inference_authorized=False)
    aggregate_path = namespace / ("inference_receipt_" + canonical_sha256(sorted(selected))[:12] + ".json")
    if not aggregate_path.exists():
        write_new_json(aggregate_path, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-candidate", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=6)
    parser.add_argument("--extraction-workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--inference-workers", type=int, default=4)
    parser.add_argument("--inference-output", type=Path)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--no-reuse-complete-runs", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--runs", nargs="+", default=["ac28", *sorted(HELD_OUT_RUNS)])
    args = parser.parse_args(argv)
    runs = checked_inference_runs(args.runs)
    if args.fit_only and args.inference_only:
        parser.error("Choose fit-only or inference-only")
    if not args.inference_only:
        from ..phase_gas_variant_data import extract_training_data, load_extracted_data
        extraction = args.output.with_name(args.output.name + "-extraction")
        data = (load_extracted_data(extraction) if (extraction / "extraction_receipt.json").is_file()
                else extract_training_data(args.frozen_candidate, args.snapshot,
                                           extraction, workers=args.extraction_workers))
        fit_variants(data, args.frozen_candidate, args.output, workers=args.workers,
                     threads_per_worker=args.threads_per_worker)
    if not args.fit_only:
        infer_variants(args.frozen_candidate, args.output, runs=runs,
                       threads=args.threads_per_worker, workers=args.inference_workers,
                       inference_output=args.inference_output, chunk_frames=args.chunk_frames,
                       reuse_complete_runs=not args.no_reuse_complete_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
