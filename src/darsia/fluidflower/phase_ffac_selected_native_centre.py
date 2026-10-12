"""Source-exact historical native-centre inference for selected RAW frames.

This isolated adapter retains the original unmasked optical feature operators,
the original regional CO2 reconstruction, and the literal stored-probability
encoding of the approved G3 experiment. Rig extension and later P1 rules belong
to the caller. Neither old categorical maps nor annotations are model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any

import cv2
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .project import sha256_file

CONTRACT = "ffac-selected-frozen-native-centre-v1"
ORIGINAL_REGIONAL_SHA256 = "66b6092db5470f5e8751e86d24c5195a653e739ce986efee70b9b0e5aa96bd3e"
ORIGINAL_GENERIC_GAS_SHA256 = "74e7d6e166b04d158eb04f09cdc310ec38dbbd2b148bcd00a25aedb8962020e3"
ORIGINAL_PROFILE_SHA256 = "9aae8b113306846c3d237ce74a0765c03787b671c87ea76380ccb09527117658"


def decode_native_log(log, valid):
    """Match RunV5LinearInput's channel-wise float32 decoder exactly."""
    source, mask = np.asarray(log), np.asarray(valid)
    if (source.dtype != np.float32 or source.ndim != 3 or source.shape[0] != 3
            or mask.dtype != bool or mask.shape != source.shape[1:]
            or not np.array_equal(mask, np.isfinite(source).all(axis=0))):
        raise ValueError("Native-centre RAW requires CHW float32 and exact finite HW support")
    result = np.empty((*mask.shape, 3), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for channel in range(3):
            np.exp(source[channel], out=result[..., channel])
            result[..., channel] /= np.float32(65535)
    if (not np.isfinite(result[mask]).all() or np.any(result[mask] < 0)
            or np.any(result[mask] > 1)):
        raise ValueError("Native-centre RAW exceeds the frozen unit representation")
    result[~mask] = 0
    return result


def encode_decode_probabilities(g3, regional, flat_valid, *, labelled):
    """Literal export/decoding order; do not substitute float32 probabilities."""
    pixel, scores, valid = map(np.asarray, (g3, regional, flat_valid))
    if (pixel.dtype != np.float32 or pixel.ndim != 2 or scores.dtype != np.float32
            or scores.ndim != 2 or valid.dtype != bool or valid.shape != pixel.shape
            or not np.isfinite(pixel[valid]).all() or not np.isfinite(scores).all()
            or np.any((pixel[valid] < 0) | (pixel[valid] > 1))
            or np.any((scores < 0) | (scores > 1)) or type(labelled) is not bool):
        raise ValueError("Native-centre encoding requires valid float32 probability grids")
    if labelled:
        stored_g3 = pixel.astype(np.float16)
        stored_regional = cv2.resize(scores, pixel.shape[::-1],
                                    interpolation=cv2.INTER_LINEAR).astype(np.float16)
        decoded_g3 = stored_g3.astype(np.float32)
        decoded_regional = stored_regional.astype(np.float32)
        layout = "native-float16"
    else:
        stored_g3 = np.full(pixel.shape, 255, np.uint8)
        stored_g3[valid] = np.rint(254 * pixel[valid]).astype(np.uint8)
        decoded_g3 = stored_g3.astype(np.float32) / np.float32(254)
        decoded_g3[stored_g3 == 255] = np.nan
        stored_regional = scores.astype(np.float16)
        decoded_regional = stored_regional.astype(np.float32)
        layout = "compact-g3-u8-regional-own-grid-float16"
    return decoded_g3, decoded_regional, dict(storage_layout=layout,
        native_shape_hw=list(pixel.shape), regional_working_shape_hw=list(scores.shape),
        g3_compact_rule="round-to-nearest-even(254*p); 255 invalid; decode code/254",
        regional_resize="float32 cv2.INTER_LINEAR to native grid before threshold",
        labelled=labelled, encoding_performed=True)


def original_co2_labels(current, baseline, presence, generic_gas, domain,
                        spacing_m, regional_model, profile):
    """The original regional chain, with original generic gas as evidence."""
    from . import phase_optical_regions as regional
    from . import phase_regional_gas as gas
    from . import phase_region_output_guards as guards
    from . import phase_free_gas_output_guard as cleanup

    settings = regional.Settings(**regional_model["settings"])
    raw = np.where(presence >= .5, np.where(generic_gas >= .6, 2, 1), 0).astype(np.uint8)
    prepared = regional.prepare(current, baseline, presence, generic_gas,
                                domain, spacing_m, settings)
    labels, audit = regional.predict(prepared, regional_model, raw, domain)
    if profile.get(gas.PROFILE_KEY) is not None:
        conditional = regional.probability(prepared["features"],
                                           regional_model["heads"]["gas_within_co2"])
        labels, rejected = gas.confirm_gas(labels, conditional, domain,
                                           gas.Settings(**profile[gas.PROFILE_KEY]))
        audit["regional_gas_reclassified_pixels"] = int(rejected.sum())
    if profile.get(guards.PROFILE_KEY) is not None:
        labels, audit["output_guards"] = guards.guard_labels(labels, domain,
            spacing_m, guards.Settings(**profile[guards.PROFILE_KEY]))
    if profile.get(cleanup.PROFILE_KEY) is not None:
        labels, audit["free_gas_cleanup"] = cleanup.clean_labels(labels, spacing_m,
            cleanup.Settings(**profile[cleanup.PROFILE_KEY]))
    labels = labels.copy()
    labels[~domain] = 255
    audit.update(generic_gas_used_for_co2_context=True,
                 g3_used_for_co2_context=False, original_profile_guards_only=True)
    return labels, audit


def preminimum_gas_selection(decoded_g3, decoded_regional):
    """Retained probability selection before the caller's corrected CO2 union.

    A later seam release can create CO2 in the old rejected footprint. Do not
    erase its gas evidence by intersecting the old reconstructed CO2 here.
    """
    pixel, regional = np.asarray(decoded_g3), np.asarray(decoded_regional)
    if (pixel.ndim != 2 or regional.ndim != 2 or pixel.dtype != np.float32
            or regional.dtype != np.float32):
        raise ValueError("Pre-minimum selection needs decoded float32 probability grids")
    native = cv2.resize(regional, pixel.shape[::-1], interpolation=cv2.INTER_LINEAR)
    return (pixel >= .7) & (native >= .6)


@dataclass(frozen=True)
class NativeCentreResult:
    labels: np.ndarray
    feature_support: np.ndarray
    raw_feature_support: np.ndarray
    domain: np.ndarray
    preminimum_gas: np.ndarray
    g3_pixel: np.ndarray
    regional_refit: np.ndarray
    flatfield_valid: np.ndarray
    provenance: dict[str, Any]


class NativeCentreEvaluator:
    """One worker's frozen models; cache only its unmodified RAW flat reference."""
    def __init__(self, frozen_candidate, g3_model_path, refit_regional_path, *,
                 profile=None, threads=1, block_rows=128,
                 cache_maximum_bytes=0):
        from .phase_learning import HierarchicalPhaseModel
        from .phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
        from .phase_g3_probability_layers import _validate_refit
        from .phase_ffac_selected_fixed_baseline_decode_cache import FixedBaselineDecodeCache

        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError("Native-centre threads must be bounded to 1..16")
        if type(block_rows) is not int or block_rows < 1:
            raise ValueError("Native-centre tile rows must be positive")
        self.candidate = Path(frozen_candidate).resolve()
        profile_path = self.candidate / "phase_optical_regions.json"
        stored_profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if profile is not None and {k: v for k, v in profile.items() if not k.startswith("_")} != stored_profile:
            raise ValueError("Native-centre profile override differs from the frozen original")
        self.profile = stored_profile
        original_path = self.candidate / self.profile["model"]["path"]
        g3_path, refit_path = Path(g3_model_path), Path(refit_regional_path)
        co2_path = self.candidate / "models/generic/co2_presence_signal.json"
        if self.profile["model"]["sha256"] != ORIGINAL_REGIONAL_SHA256:
            raise ValueError("Native-centre original regional profile changed")
        generic_gas_path = self.candidate / "models/generic/gas_within_co2_signal.json"
        expected = {"original_regional": ORIGINAL_REGIONAL_SHA256,
                    "g3": G3_SHA256, "regional_refit": REFIT_SHA256, "co2": CO2_SHA256,
                    "original_generic_gas": ORIGINAL_GENERIC_GAS_SHA256,
                    "original_profile": ORIGINAL_PROFILE_SHA256}
        paths = dict(original_regional=original_path, g3=g3_path,
                     regional_refit=refit_path, co2=co2_path,
                     original_generic_gas=generic_gas_path, original_profile=profile_path)
        self.binding = {}
        for name, path in paths.items():
            digest = sha256_file(path)
            if digest != expected[name]:
                raise ValueError("Frozen native-centre artifact changed: " + name)
            self.binding[name] = dict(path=str(path), sha256=digest)
        self.original = HierarchicalPhaseModel.load(self.candidate)
        self.regional = json.loads(original_path.read_text(encoding="utf-8-sig"))
        self.g3 = json.loads(g3_path.read_text(encoding="utf-8-sig"))
        self.refit = json.loads(refit_path.read_text(encoding="utf-8-sig"))
        _validate_refit(self.regional, self.refit)
        if (self.g3.get("variant") != "G3" or len(self.g3["feature_names"]) != 4
                or self.g3["fit"].get("converged") is not True):
            raise ValueError("Native-centre requires the frozen converged G3 head")
        self.threads, self.block_rows = threads, block_rows
        self._flat_key = None
        self._flat_reference = None
        self._flat_sources = None
        self._fixed_baseline_decode_cache = FixedBaselineDecodeCache(
            max_bytes=cache_maximum_bytes)

    def infer(self, current_log, *, current_valid, source_baseline_log,
              baseline_valid, H_B_current, photo_sand_mask, spacing_m,
              source_origin_xy=(0, 0), flat_baseline_log=None,
              flat_baseline_valid=None, previous_log=None, previous_valid=None,
              H_B_previous=None, previous_origin_xy=(0, 0),
              labelled_probability_encoding=False, probability_receipt=None,
              input_provenance=None):
        from .tools.ffac_v5_temporal_alignment import align_temporal_inputs
        from .tools.ffac_v5_frame_context import compute_feature_support
        from .tools.phase_inference_benchmark_backends import CpuFeaturePredictor
        from . import phase_gas_flatfield as flat
        from . import phase_optical_regions as regional
        from .phase_gas_regional_context import build_shared_regional_context, prepare_conditional_gas
        from .tools.train_ffac_gas_flatfield_variants import gas_probability
        from .phase_g3_production import apply_g3_configuration
        from .phase_retained_source_inference import probability_encoding_contract

        started = time.perf_counter()
        current = decode_native_log(current_log, current_valid)
        baseline = self._fixed_baseline_decode_cache.decode("fixed_temporal_baseline",
            source_baseline_log, baseline_valid)
        previous = None if previous_log is None else decode_native_log(previous_log, previous_valid)
        shape = current.shape[:2]
        photo_sand = np.asarray(photo_sand_mask)
        if photo_sand.dtype != bool or photo_sand.shape != shape:
            raise ValueError("Native-centre requires exact photo-sand HW bool")
        spacing = np.asarray(spacing_m, float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Native-centre requires frozen nominal y/x spacing")
        if (previous_log is None) != (previous_valid is None) or (previous_log is None) != (H_B_previous is None):
            raise ValueError("Native-centre predecessor must provide log/support/H together")
        if flat_baseline_log is None:
            if source_origin_xy != (0, 0) or source_baseline_log.shape[1:] != shape:
                raise ValueError("Extended temporal baseline requires its separate native flat-field core")
            flat_baseline_log, flat_baseline_valid = source_baseline_log, baseline_valid
        if flat_baseline_valid is None or flat_baseline_log.shape[1:] != shape:
            raise ValueError("Native flat-field baseline must be same-grid source-supported log RAW")
        # Validate separately: temporal extension and native flat-field core are
        # deliberately different historical operators.
        self._fixed_baseline_decode_cache.validate_fixed("fixed_native_flat_baseline",
            flat_baseline_log, flat_baseline_valid)
        encoding_receipt = None
        if probability_receipt is not None:
            encoding_receipt = probability_encoding_contract(probability_receipt, shape)
            if encoding_receipt["layout"] != ("native-float16" if labelled_probability_encoding else
                    "compact-g3-u8-regional-own-grid-float16"):
                raise ValueError("Requested native encoding differs from the matched parent receipt")
        cv2.setNumThreads(self.threads)
        with threadpool_limits(limits=self.threads):
            support, raw_support = compute_feature_support(current_valid=current_valid,
                baseline_valid=baseline_valid, h_baseline_to_current=H_B_current,
                previous_valid=previous_valid, h_baseline_to_previous=H_B_previous,
                baseline_origin_xy=source_origin_xy, previous_origin_xy=previous_origin_xy)
            domain = photo_sand & support
            aligned_base, aligned_previous, alignment = align_temporal_inputs(
                baseline=baseline, current=current, previous=previous,
                h_baseline_to_current=H_B_current, h_baseline_to_previous=H_B_previous,
                required_valid_mask=support, baseline_origin_xy=source_origin_xy,
                previous_origin_xy=previous_origin_xy, baseline_source_valid_mask=baseline_valid,
                previous_source_valid_mask=previous_valid)
            predictor = CpuFeaturePredictor(self.original, aligned_base,
                block_rows=self.block_rows, input_encoding="v5-unit-float32")
            flat_key = (id(flat_baseline_log), id(flat_baseline_valid), shape)
            if self._flat_key != flat_key:
                self._flat_reference = flat.prepare_flatfield_reference(flat_baseline_log,
                    source_valid_mask=flat_baseline_valid, tile_rows=self.block_rows)
                self._flat_sources = (flat_baseline_log, flat_baseline_valid)
                self._flat_key = flat_key
            registered = flat.register_flatfield_reference(self._flat_reference,
                H_B_current, target_shape_hw=shape)
            current_features = predictor.prepare_rgb(current)
            previous_features = (predictor.baseline if aligned_previous is None else
                                 predictor.prepare_rgb(aligned_previous))
            generic = np.empty((*shape, 2), np.float32)
            g3_presence = np.empty(shape, np.float32)
            g3 = np.full(shape, np.nan, np.float32)
            flat_valid = np.zeros(shape, bool)
            tiles = iter(flat.iter_flatfield_feature_tiles(current_log, registered,
                current_valid_mask=current_valid, tile_rows=self.block_rows))
            for first, last, features in predictor.feature_blocks(current_features,
                    previous_features, dynamic_available=aligned_previous is not None):
                ff_first, ff_last, ff_features, valid = next(tiles)
                if (first, last) != (ff_first, ff_last):
                    raise ValueError("Native-centre optical and flat-field tile rows differ")
                # Original CpuFeaturePredictor.predict uses one two-head matrix
                # multiplication. NativeG3FrameEvaluator uses a single-head
                # operation for its presence context. Keep both exact orders.
                generic[first:last] = expit(features @ predictor.weights + predictor.bias).reshape(last-first, shape[1], 2)
                g3_presence[first:last] = expit(features @ predictor.weights[:, 0] + predictor.bias[0]).reshape(last-first, shape[1])
                safe = np.where(valid[..., None], ff_features, 0).reshape(-1, 4)
                scores = gas_probability(self.g3, features, safe).reshape(last-first, shape[1])
                g3[first:last] = np.where(valid, scores, np.nan)
                flat_valid[first:last] = valid
            inherited, co2_audit = original_co2_labels(current, aligned_base,
                generic[..., 0], generic[..., 1], domain, spacing, self.regional, self.profile)
            shared = build_shared_regional_context(current, aligned_base, g3_presence,
                domain, spacing, regional.Settings(**self.regional["settings"]))
            pure = prepare_conditional_gas(shared, np.nan_to_num(g3, nan=0.0))
            reg = regional.probability(pure["features"], self.refit["heads"]["gas_within_co2"])
            decoded_g3, decoded_reg, encoding = encode_decode_probabilities(g3, reg,
                flat_valid, labelled=labelled_probability_encoding)
            nominal_area = float(np.prod(spacing) * 1e6)
            labels, gas_audit = apply_g3_configuration(inherited, decoded_g3,
                                                     decoded_reg, nominal_area)
            preminimum = preminimum_gas_selection(decoded_g3, decoded_reg)
        provenance = dict(contract=CONTRACT, inference_performed=True,
            scientific_input="CHW float32 log camera RAW", input_provenance=input_provenance or {},
            models=self.binding, code_sha256=sha256_file(Path(__file__)),
            feature_operators="original CpuFeaturePredictor; unmasked Gaussian blur; float32 weights",
            decoder="np.exp(float32 channel,out=float32); /np.float32(65535); invalid zero",
            temporal_alignment=alignment, native_shape_hw=list(shape), spacing_m=spacing.tolist(),
            nominal_pixel_area_mm2=nominal_area, current_pixels_resampled=False,
            co2_context=co2_audit, gas_configuration=gas_audit, probability_encoding=encoding,
            matched_encoding_receipt=encoding_receipt, feature_support_pixels=int(support.sum()),
            domain_pixels=int(domain.sum()), flat_valid_pixels=int(flat_valid.sum()),
            preminimum_selection="decoded pixel>=0.7 AND resized regional>=0.6 before corrected CO2 union intersection",
            source_labels_used_as_model_input=False, labels_changed=False, training=False,
            cpu_only=True, threads=self.threads, seconds=time.perf_counter()-started)
        provenance["fixed_baseline_decode_cache"] = self._fixed_baseline_decode_cache.receipt()
        return NativeCentreResult(labels, support, raw_support, domain, preminimum,
                                  decoded_g3, decoded_reg, flat_valid, provenance)


def infer_native_centre(frozen_candidate, current_log, *, g3_model_path,
                        refit_regional_path, evaluator=None, profile=None,
                        threads=1, **kwargs):
    """Convenience entry point; retain an evaluator per worker for reference reuse."""
    if evaluator is None:
        evaluator = NativeCentreEvaluator(frozen_candidate, g3_model_path,
                                         refit_regional_path, profile=profile, threads=threads)
    return evaluator.infer(current_log, **kwargs)
