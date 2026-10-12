"""Source-exact CPU G3 probability export, with no final-map production.

Scientific pixels remain floating-point RAW.  The legacy regional context uses
the old conditional score only where G3's flat-field features are invalid,
solely to reproduce the frozen G3 experiment.  The refitted regional context
uses G3 alone: invalid values are zero for arithmetic, never training support.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import time
import uuid
from typing import Any

import cv2
import numpy as np

from .project import sha256_file
from .tools.train_ffac_gas_flatfield_variants import (
    HEAD, gas_probability, read_json, _read_frozen_map, frozen_inference_eligibility,
)


CONTRACT = "ffac-g3-downstream-probability-layers-v1"
MASK_NAMES = ("co2", "photo_sand_domain", "feature_support", "regional_native_domain",
              "flatfield_valid", "feature_eligible", "optical_presence_ge_half",
              "frozen_gas", "frozen_unresolved")
LABELLED_FRAMES = {
    "ac39": (1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 15),
    "ac50": (1, 2, 3, 7, 12, 13, 16), "ac31": (1, 2, 3, 9, 10),
    "ac48": (3, 7, 9), "ac53": (3, 7, 9), "ac35": (3, 4, 5, 7, 9),
    "ac28": (8,),
}


@dataclass
class FrameLayers:
    current: np.ndarray
    aligned_baseline: np.ndarray
    presence: np.ndarray
    g3_pixel: np.ndarray
    flat_valid: np.ndarray
    domain: np.ndarray
    photo_sand_domain: np.ndarray
    feature_support: np.ndarray
    spacing_m: tuple[float, float]
    native: dict
    prepared: dict
    prepared_legacy: dict
    regional_frozen: np.ndarray
    regional_refit: np.ndarray | None
    metadata: dict


def _validate_refit(frozen, refit):
    if frozen["settings"] != refit["settings"]:
        raise ValueError("Regional refit changed frozen settings")
    if frozen.get("feature_names") != refit.get("feature_names"):
        raise ValueError("Regional refit changed feature names")
    if set(frozen["heads"]) != set(refit["heads"]):
        raise ValueError("Regional refit changed the set of heads")
    for name in frozen["heads"]:
        if name != HEAD and frozen["heads"][name] != refit["heads"][name]:
            raise ValueError("Regional refit changed a frozen non-gas head")


class NativeG3FrameEvaluator:
    """Cache one exact source baseline while evaluating independent frames."""
    def __init__(self, candidate, g3_model_path, native_manifest_path, *, run=None,
                 threads=2, source=None, independent_reference_frames=None):
        from .phase_learning import HierarchicalPhaseModel
        from .phase_training.v5_linear_input import RunV5LinearInput
        from .tools.ffac_v5_track1_photo_context import Track1FramePhotoCohort
        from .phase_gas_flatfield import prepare_flatfield_reference
        from .phase_gas_inference_chunks import previous_frame_indices

        if not 1 <= threads <= 16:
            raise ValueError("Invalid bounded CPU thread allocation")
        self.threads = int(threads)
        self.candidate, self.g3_path = Path(candidate).resolve(), Path(g3_model_path).resolve()
        self.native_path = Path(native_manifest_path).resolve()
        self.native = read_json(self.native_path)
        self.run = str(run or self.native.get("run") or "").casefold()
        if not self.run or (self.native.get("run") and self.native["run"].casefold() != self.run):
            raise ValueError("Native source run differs from the requested run")
        self.source = Path(source).resolve() if source else self.native_path.parent.parent.parent
        self.shape = tuple(self.native["analysis_image_shape_hw"])
        self.profile = read_json(self.candidate / "phase_optical_regions.json")
        self.regional_path = self.candidate / self.profile["model"]["path"]
        if sha256_file(self.regional_path) != self.profile["model"]["sha256"]:
            raise ValueError("Frozen regional model changed")
        self.regional = read_json(self.regional_path)
        self.g3_model = read_json(self.g3_path)
        if self.g3_model["variant"] != "G3" or len(self.g3_model["feature_names"]) != 4:
            raise ValueError("Probability export requires the frozen four-feature G3 head")
        if not self.g3_model["fit"].get("converged", False):
            raise ValueError("Unconverged G3 head cannot be exported")
        original = HierarchicalPhaseModel.load(self.candidate)
        self.original = original
        self.reader = RunV5LinearInput(native_manifest_path=self.native_path,
                                      run_id=self.run, content_shape_hw=self.shape)
        self.cohort = Track1FramePhotoCohort(self.native_path)
        baseline_photo = self.source / "native_references" / self.run / "pre_injection.jpg"
        baseline_core = self.reader.load(baseline_photo, "pre_injection")
        self.baseline_valid, _ = self.reader.load_validity(baseline_photo, "pre_injection")
        baseline_log_path = Path(self.reader.reference["v5_log_linear_rgb_path"])
        self.flat_reference = prepare_flatfield_reference(
            np.load(baseline_log_path, allow_pickle=False, mmap_mode="r"),
            source_valid_mask=self.baseline_valid)
        self.temporal = (self.native.get("v5_temporal_reference") or {}).get("contract")
        if self.temporal == "ffac-v5-source-backed-extended-ring-v1":
            self.baseline, self.baseline_origin, self.baseline_valid_extended = self.reader.load_extended(
                baseline_photo, "pre_injection")
        elif self.temporal == "ffac-v5-native-crop-source-supported-v1":
            self.baseline, self.baseline_origin, self.baseline_valid_extended = (
                baseline_core, (0, 0), self.baseline_valid)
        else:
            raise ValueError("Unsupported temporal RAW source contract")
        independent = (self.native.get("independent_reference_frames") or ()) if (
            independent_reference_frames is None) else independent_reference_frames
        self.independent = tuple(int(value) for value in independent)
        self.predecessors = previous_frame_indices(len(self.native["frame_sources"]), self.independent)
        self.predictor = None
        self.binding = dict(native_manifest_path=str(self.native_path),
                            native_manifest_sha256=sha256_file(self.native_path),
                            g3_model_path=str(self.g3_path), g3_model_sha256=sha256_file(self.g3_path),
                            regional_frozen_path=str(self.regional_path),
                            regional_frozen_sha256=sha256_file(self.regional_path),
                            co2_head_sha256=sha256_file(self.candidate / "models/generic/co2_presence_signal.json"),
                            independent_reference_frames=list(self.independent))

    def evaluate(self, frame_index, *, refit_regional_path=None, export_frozen_co2=False):
        from scipy.special import expit
        from threadpoolctl import threadpool_limits
        from . import phase_gas_flatfield as flat, phase_optical_regions as regional
        from .phase_gas_regional_context import build_shared_regional_context, prepare_conditional_gas
        from .tools.ffac_v5_temporal_alignment import align_temporal_inputs
        from .tools.phase_inference_benchmark_backends import CpuFeaturePredictor
        from .tools.train_phase_optical_regions import track1_frame_domain

        if export_frozen_co2:
            raise ValueError("Training-source evaluation does not load current frozen maps; use export_frame")
        index = int(frame_index)
        if index != frame_index or not 0 <= index < len(self.predecessors):
            raise ValueError("Frame index is outside its exact source timeline")
        cv2.setNumThreads(self.threads)
        started = time.perf_counter()
        with threadpool_limits(limits=self.threads):
            prior = self.predecessors[index]
            current_photo = self.source / "native_frames" / self.run / f"frame_{index:03d}.jpg"
            domains, spacing, domain_binding = track1_frame_domain(
                self.native, self.cohort, index, nominal_profile=self.profile,
                nominal_profile_root=self.candidate)
            feature_support, feature_receipt = self.cohort.feature_support(index, prior, self.reader)
            photo_sand = np.asarray(domains["photo_certified_sand_domain"], bool)
            domain = photo_sand & feature_support
            current = self.reader.load(current_photo, f"frame_{index:03d}",
                                       required_valid_mask=feature_support)
            current_source = self.native["frame_sources"][index]
            current_log = np.load(Path(current_source["v5_log_linear_rgb_path"]),
                                  allow_pickle=False, mmap_mode="r")
            current_valid, _ = self.reader.load_validity(current_photo, f"frame_{index:03d}")
            previous, previous_origin, previous_valid = None, (0, 0), None
            if prior is not None:
                photo = self.source / "native_frames" / self.run / f"frame_{prior:03d}.jpg"
                if self.temporal == "ffac-v5-source-backed-extended-ring-v1":
                    previous, previous_origin, previous_valid = self.reader.load_extended(photo, f"frame_{prior:03d}")
                else:
                    previous = self.reader.load(photo, f"frame_{prior:03d}")
            aligned_baseline, aligned_previous, alignment = align_temporal_inputs(
                baseline=self.baseline, current=current, previous=previous,
                h_baseline_to_current=self.cohort.H_v5_B_to_frame(index),
                h_baseline_to_previous=None if prior is None else self.cohort.H_v5_B_to_frame(prior),
                required_valid_mask=feature_support, baseline_origin_xy=self.baseline_origin,
                previous_origin_xy=previous_origin,
                baseline_source_valid_mask=self.baseline_valid_extended,
                previous_source_valid_mask=previous_valid)
            if self.predictor is None:
                self.predictor = CpuFeaturePredictor(self.original, aligned_baseline,
                                                    block_rows=128, input_encoding="v5-unit-float32")
            else:
                self.predictor.set_baseline(aligned_baseline)
            registered = flat.register_flatfield_reference(
                self.flat_reference, self.cohort.H_v5_B_to_frame(index), target_shape_hw=self.shape)
            current_features = self.predictor.prepare_rgb(current)
            previous_features = self.predictor.baseline if aligned_previous is None else self.predictor.prepare_rgb(aligned_previous)
            presence = np.empty(self.shape, np.float32)
            legacy_gas = np.empty(self.shape, np.float32)
            g3 = np.full(self.shape, np.nan, np.float32)
            ff_valid = np.zeros(self.shape, bool)
            tiles = iter(flat.iter_flatfield_feature_tiles(current_log, registered,
                                                          current_valid_mask=current_valid, tile_rows=128))
            for first, last, features in self.predictor.feature_blocks(
                    current_features, previous_features, dynamic_available=aligned_previous is not None):
                ff_first, ff_last, flat_features, valid = next(tiles)
                if (first, last) != (ff_first, ff_last):
                    raise ValueError("RAW feature tile rows differ")
                presence[first:last] = expit(features @ self.predictor.weights[:, 0] + self.predictor.bias[0]).reshape(last-first, self.shape[1])
                old_gas = expit(features @ self.predictor.weights[:, 1] + self.predictor.bias[1]).reshape(last-first, self.shape[1])
                safe = np.where(valid[..., None], flat_features, 0).reshape(-1, 4)
                score = gas_probability(self.g3_model, features, safe).reshape(last-first, self.shape[1])
                ff_valid[first:last] = valid
                g3[first:last] = np.where(valid, score, np.nan)
                legacy_gas[first:last] = np.where(valid, score, old_gas)
            shared = build_shared_regional_context(current, aligned_baseline, presence, domain,
                                                   spacing, regional.Settings(**self.regional["settings"]))
            pure_prepared = prepare_conditional_gas(shared, np.nan_to_num(g3, nan=0.0))
            legacy_prepared = prepare_conditional_gas(shared, legacy_gas)
            frozen_score = regional.probability(legacy_prepared["features"], self.regional["heads"][HEAD])
            refit_score = None
            refit_binding = {}
            if refit_regional_path is not None:
                path = Path(refit_regional_path).resolve()
                refit = read_json(path)
                _validate_refit(self.regional, refit)
                refit_score = regional.probability(pure_prepared["features"], refit["heads"][HEAD])
                refit_binding = dict(regional_refit_path=str(path), regional_refit_sha256=sha256_file(path))
        metadata = dict(self.binding, **refit_binding, run=self.run, frame_index=index,
                        previous_frame_index=prior, elapsed_h=float(current_source["elapsed_h"]),
                        native_shape_hw=list(self.shape), spacing_m=list(spacing),
                        pixel_area_mm2=float(np.prod(spacing) * 1e6), domain_binding=domain_binding,
                        feature_support_receipt_sha256=feature_receipt["receipt_sha256"],
                        alignment=alignment, seconds=time.perf_counter()-started,
                        regional_frozen_input="G3 where FF-valid; frozen old conditional gas elsewhere",
                        regional_refit_input="G3; zero for invalid arithmetic, invalid excluded from labelled support",
                        regional_context_domain="photo_certified_sand_domain AND frozen source feature_support",
                        unchanged_context_columns=[i for i in range(20) if i not in (1, 11)])
        return FrameLayers(current, aligned_baseline, presence, g3, ff_valid, domain,
                           photo_sand, feature_support, tuple(spacing), self.native,
                           pure_prepared, legacy_prepared, frozen_score, refit_score, metadata)

    frame = evaluate


def evaluate_source_frame(frozen_candidate, g3_model_path, native_manifest_path,
                         frame_index, *, threads=2, source=None, run=None,
                         independent_reference_frames=None, refit_regional_path=None):
    evaluator = NativeG3FrameEvaluator(frozen_candidate, g3_model_path, native_manifest_path,
                                     run=run, threads=threads, source=source,
                                     independent_reference_frames=independent_reference_frames)
    return evaluator.evaluate(frame_index, refit_regional_path=refit_regional_path)


def _current_run_source(candidate, run):
    from .tools.train_phase_model_further import native_source
    package = Path(candidate) / "validation/run_packages" / run
    jobs = [row for row in read_json(package / "postflight_manifest.json")["jobs"]
            if row["run"].casefold() == run]
    if len(jobs) != 1:
        raise ValueError("Frozen current run has no unique source")
    source, native_path, native = native_source(jobs[0])
    binding = read_json(package / "validation_binding.json")
    maps = Path(jobs[0]["phase_map_output_dir"]) / "native_phase_maps" / run / ("A_" + binding["pipeline_sha256"][:16])
    return jobs[0], source, native_path, native, maps


def resolve_current_g3_map(experiment, run, frame_index):
    """Resolve already-produced restricted maps without reading labels or scores."""
    root, run = Path(experiment).resolve(), str(run).casefold()
    subfolder = "qa_ac28" if run == "ac28" else "scoring_blind"
    choices = [root / subfolder / run / "inference_receipt.json"]
    choices.extend(path / subfolder / run / "inference_receipt.json"
                   for path in sorted(root.glob("parallel-inference*")))
    for receipt_path in choices:
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            frames = [row for row in receipt["frames"] if row["frame_index"] == frame_index]
            if len(frames) == 1:
                row = frames[0]["variants"]["G3"]
                path = Path(row["map_path"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("Existing G3 map escaped its experiment or is absent")
                return str(path)
    return None


def inventory_frames(frozen_candidate, experiment=None):
    candidate = Path(frozen_candidate).resolve()
    packages = candidate / "validation/run_packages"
    rows = []
    for package in sorted(packages.iterdir(), key=lambda p: p.name):
        if not package.is_dir() or not (package / "postflight_manifest.json").is_file():
            continue
        run = package.name.casefold()
        _job, _source, native_path, native, maps = _current_run_source(candidate, run)
        digest = sha256_file(native_path)
        for index, frame in enumerate(native["frame_sources"]):
            frozen = maps / f"frame_{index:03d}.png"
            if not frozen.is_file():
                raise ValueError("Frozen current CO2 map is missing")
            rows.append(dict(run=run, frame_index=index, elapsed_h=float(frame["elapsed_h"]),
                             native_manifest_path=str(native_path), native_manifest_sha256=digest,
                             native_shape_hw=list(native["analysis_image_shape_hw"]),
                             frozen_map_path=str(frozen), current_g3_map_path=(
                                 resolve_current_g3_map(experiment, run, index) if experiment else None)))
    if len({row["run"] for row in rows}) != 43 or len(rows) != 864:
        raise ValueError("Overnight inventory must bind exactly 43 current runs and 864 frames")
    return rows


def _atomic_npy(path, array, *, compressed=False):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            if compressed:
                np.savez_compressed(stream, masks=array)
            else:
                np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if sha256_file(path) != sha256_file(temporary):
                raise ValueError("Existing probability file differs; it will not be overwritten")
        else:
            # Rename is atomic and never replaces an existing file on Windows.
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), sha256=sha256_file(path), dtype=str(array.dtype),
                shape=list(array.shape), bytes=path.stat().st_size)


def _atomic_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if read_json(path) != value:
                raise ValueError("Existing probability receipt differs; it will not be overwritten")
        else:
            temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_export_receipt(receipt_path, *, expected_binding=None):
    path = Path(receipt_path).resolve()
    receipt = read_json(path)
    if receipt.get("contract") != CONTRACT or receipt.get("complete") is not True:
        raise ValueError("Probability receipt is not complete")
    if expected_binding is not None and receipt["binding"] != expected_binding:
        raise ValueError("Probability receipt binds different inputs or storage")
    for info in receipt["files"].values():
        artifact = Path(info["path"]).resolve()
        if not artifact.is_relative_to(path.parent) or sha256_file(artifact) != info["sha256"]:
            raise ValueError("Probability artifact changed or escaped its frame folder")
    return dict(receipt, receipt_path=str(path), receipt_sha256=sha256_file(path))


_EVALUATOR = None
_EVALUATOR_KEY = None


def export_frame(frozen_candidate, experiment, refit_model, local_output, run, frame_index,
                 *, labelled, threads=2, provisional=False):
    """Write only probability arrays, atomically, and skip verified completions."""
    global _EVALUATOR, _EVALUATOR_KEY
    candidate, experiment = Path(frozen_candidate).resolve(), Path(experiment).resolve()
    run, index = str(run).casefold(), int(frame_index)
    job, source, native_path, native, frozen_maps = _current_run_source(candidate, run)
    if not 0 <= index < len(native["frame_sources"]):
        raise ValueError("Probability frame is outside the exact current timeline")
    g3_path = experiment / "variants/G3/gas_head.json"
    refit_path = Path(refit_model).resolve()
    folder = Path(local_output).resolve() / "layers" / run / f"frame_{index:03d}"
    if folder.is_relative_to(candidate) or folder.is_relative_to(experiment):
        raise ValueError("Probability export must use a new detached output folder")
    frozen_path = frozen_maps / f"frame_{index:03d}.png"
    binding = dict(frozen_candidate=str(candidate), g3_model_sha256=sha256_file(g3_path),
                   regional_refit_sha256=sha256_file(refit_path),
                   native_manifest_sha256=sha256_file(native_path),
                   frozen_map_sha256=sha256_file(frozen_path), run=run, frame_index=index,
                   labelled=bool(labelled), provisional=bool(provisional),
                   code_sha256=sha256_file(Path(__file__)))
    receipt_path = folder / "receipt.json"
    if receipt_path.is_file():
        return validate_export_receipt(receipt_path, expected_binding=binding)
    folder.mkdir(parents=True, exist_ok=True)
    key = (str(candidate), str(g3_path), str(native_path), int(threads),
           tuple(job.get("independent_reference_frames") or ()))
    if _EVALUATOR_KEY != key:
        _EVALUATOR = None
        _EVALUATOR = NativeG3FrameEvaluator(candidate, g3_path, native_path, run=run,
                                          threads=threads, source=source,
                                          independent_reference_frames=key[-1])
        _EVALUATOR_KEY = key
    layers = _EVALUATOR.evaluate(index, refit_regional_path=refit_path)
    frozen, _rgba = _read_frozen_map(frozen_path, native["content_box_xywh"])
    del _rgba
    eligible = frozen_inference_eligibility(layers.domain, layers.flat_valid, frozen)
    masks = dict(co2=np.isin(frozen, (1, 2)), photo_sand_domain=layers.photo_sand_domain,
                 feature_support=layers.feature_support, regional_native_domain=layers.domain,
                 flatfield_valid=layers.flat_valid, feature_eligible=eligible,
                 optical_presence_ge_half=layers.presence >= .5,
                 frozen_gas=frozen == 2, frozen_unresolved=frozen == 255)
    files = {}
    if labelled:
        g3 = layers.g3_pixel.astype(np.float16)
        reg_frozen = cv2.resize(layers.regional_frozen, layers.domain.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float16)
        reg_refit = cv2.resize(layers.regional_refit, layers.domain.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float16)
        layout = "native-float16"
    else:
        g3 = np.full(layers.domain.shape, 255, np.uint8)
        g3[layers.flat_valid] = np.rint(254 * layers.g3_pixel[layers.flat_valid]).astype(np.uint8)
        reg_frozen = layers.regional_frozen.astype(np.float16)
        reg_refit = layers.regional_refit.astype(np.float16)
        layout = "compact-g3-u8-regional-own-grid-float16"
    for name, array in (("g3_pixel.npy", g3), ("regional_frozen.npy", reg_frozen),
                        ("regional_refit.npy", reg_refit)):
        files[name] = _atomic_npy(folder / name, array)
    files["co2.packbits.npy"] = _atomic_npy(folder / "co2.packbits.npy", np.packbits(masks["co2"].reshape(-1), bitorder="little"))
    packed = np.stack([np.packbits(masks[name].reshape(-1), bitorder="little") for name in MASK_NAMES[1:]])
    files["extra_masks.packbits.npz"] = _atomic_npy(folder / "extra_masks.packbits.npz", packed, compressed=True)
    metadata = dict(layers.metadata)
    metadata.update(current_g3_map_path=resolve_current_g3_map(experiment, run, index),
                    frozen_map_path=str(frozen_path), frozen_map_sha256=binding["frozen_map_sha256"],
                    storage_layout=layout, regional_working_shape_hw=list(layers.regional_frozen.shape),
                    regional_working_spacing_m=list(layers.prepared["spacing_m"]),
                    resize_rule="cv2.resize(values.astype(float32), native_shape[::-1], interpolation=cv2.INTER_LINEAR); threshold AFTER resize",
                    g3_compact_rule="round-to-nearest-even(254*p); 255 invalid; decode code/254",
                    bitorder="little", masks_order=list(MASK_NAMES[1:]),
                    mask_shapes_hw=list(layers.domain.shape),
                    native_g3_invalid_count=int(np.count_nonzero(~layers.flat_valid)),
                    photo_sand_definition="pure photo_certified_sand_domain; no feature support intersection",
                    feature_eligible_definition="photo_sand AND feature_support AND flatfield_valid AND frozen_label !=255",
                    legacy_extra_operators=["pixel gas also requires optical_presence>=0.5",
                         "frozen labels restored outside feature_eligible",
                         "4-connected gas minimum calculated before restoring ineligible frozen labels"],
                    final_maps_created=False, studio_candidate_created=False,
                    cpu_only=True, mass_eligible=False)
    receipt = dict(contract=CONTRACT, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                   binding=binding, metadata=metadata, files=files)
    _atomic_json(receipt_path, receipt)
    return validate_export_receipt(receipt_path, expected_binding=binding)


def load_exported_layers(receipt_path):
    receipt = validate_export_receipt(receipt_path)
    shape = tuple(receipt["metadata"]["native_shape_hw"])
    count = int(np.prod(shape))
    files = receipt["files"]
    gas = np.load(files["g3_pixel.npy"]["path"], allow_pickle=False)
    if gas.dtype == np.uint8:
        g3 = gas.astype(np.float32) / 254
        g3[gas == 255] = np.nan
    else:
        g3 = gas.astype(np.float32)
    masks = {"co2": np.unpackbits(np.load(files["co2.packbits.npy"]["path"], allow_pickle=False),
                                 bitorder="little", count=count).reshape(shape).astype(bool)}
    with np.load(files["extra_masks.packbits.npz"]["path"], allow_pickle=False) as archive:
        packed = archive["masks"]
    for name, row in zip(receipt["metadata"]["masks_order"], packed):
        masks[name] = np.unpackbits(row, bitorder="little", count=count).reshape(shape).astype(bool)
    arrays = dict(g3_pixel=g3, **masks)
    for key in ("regional_frozen", "regional_refit"):
        values = np.load(files[key + ".npy"]["path"], allow_pickle=False).astype(np.float32)
        if values.shape != shape:
            values = cv2.resize(values, shape[::-1], interpolation=cv2.INTER_LINEAR)
        arrays[key] = values
    return arrays, receipt


def reconstruct_exported_gas(arrays, pixel_area_mm2, *, threshold=.6,
                            regional="frozen", minimum_area_mm2=2.5, legacy=True):
    """Diagnostic reconstruction only; never write categorical maps."""
    if regional not in ("frozen", "refit", "off") or not 0 <= minimum_area_mm2 <= 25:
        raise ValueError("Unsupported downstream diagnostic configuration")
    co2 = np.asarray(arrays["co2"], bool)
    result = co2 & (np.asarray(arrays["g3_pixel"]) >= threshold)
    if legacy:
        eligible = np.asarray(arrays["feature_eligible"], bool)
        result &= np.asarray(arrays["optical_presence_ge_half"], bool)
        result = np.where(eligible & co2, result, np.asarray(arrays["frozen_gas"], bool))
    if regional != "off":
        rejection = result & (np.asarray(arrays["regional_" + regional]) < .6)
        if legacy:
            rejection &= eligible
        result[rejection] = False
    if minimum_area_mm2:
        if not np.isfinite(pixel_area_mm2) or pixel_area_mm2 <= 0:
            raise ValueError("Gas minimum requires positive nominal pixel area")
        _, components, stats, _ = cv2.connectedComponentsWithStats(result.astype(np.uint8), connectivity=4)
        small = stats[:, cv2.CC_STAT_AREA] * pixel_area_mm2 < minimum_area_mm2
        small[0] = False
        result[small[components]] = False
    if legacy:
        result[~eligible] = np.asarray(arrays["frozen_gas"], bool)[~eligible]
    return result & co2


def qa_ac28_frame8(export_receipt, existing_g3_map_path=None):
    path = Path(export_receipt["receipt_path"] if isinstance(export_receipt, dict) else export_receipt)
    arrays, receipt = load_exported_layers(path)
    metadata = receipt["metadata"]
    if metadata["run"] != "ac28" or metadata["frame_index"] != 8 or not receipt["binding"]["labelled"]:
        raise ValueError("Scientific reconstruction QA is restricted to AC28 frame 8 native float16")
    native = read_json(Path(metadata["native_manifest_path"]))
    current_path = Path(existing_g3_map_path or metadata["current_g3_map_path"])
    old, _rgba = _read_frozen_map(current_path, native["content_box_xywh"])
    co2 = arrays["co2"]
    if not np.array_equal(np.isin(old, (1, 2)), co2):
        raise ValueError("Existing G3 map and exported frozen CO2 union differ")
    reports = {}
    for mode, legacy in (("literal_section3", False), ("legacy_exact_operators", True)):
        gas = reconstruct_exported_gas(arrays, metadata["pixel_area_mm2"], legacy=legacy)
        count = int(np.count_nonzero(co2))
        wrong = int(np.count_nonzero(co2 & (gas != (old == 2))))
        agreement = 1. if count == 0 else 1. - wrong/count
        reports[mode] = dict(co2_pixels=count, disagreements=wrong,
                            agreement=agreement, passes_99_9_percent=agreement >= .999)
    explanation = ("Frozen variant inference does not call regional.predict; it additionally gates gas "
                   "by frozen optical presence>=.5 and restores frozen labels outside exact eligibility. "
                   "Masks for all such operators are exported.")
    return dict(contract=CONTRACT, run="ac28", frame_index=8, current_g3_map_path=str(current_path),
                current_g3_map_sha256=sha256_file(current_path), exported_receipt_sha256=sha256_file(path),
                **reports, float16_probability_storage=True,
                regional_predict_applied=False,
                simple_pixel_agreement=reports["literal_section3"]["agreement"],
                faithful_pixel_agreement=reports["legacy_exact_operators"]["agreement"],
                qa_pass=reports["legacy_exact_operators"]["passes_99_9_percent"],
                difference_explanation=explanation, explanation=explanation,
                required_reproduction_layers=["feature_eligible", "optical_presence_ge_half", "frozen_gas", "frozen_unresolved"])


__all__ = ["FrameLayers", "NativeG3FrameEvaluator", "evaluate_source_frame", "export_frame",
           "inventory_frames", "resolve_current_g3_map", "validate_export_receipt",
           "load_exported_layers", "reconstruct_exported_gas", "qa_ac28_frame8", "LABELLED_FRAMES"]
