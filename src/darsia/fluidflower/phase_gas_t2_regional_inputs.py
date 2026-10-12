"""Source-exact CPU inputs and gas-only decisions for the sealed G3-T2 refit.

CO2 categories are supplied by the sealed candidate. Frozen optical presence
is evaluated only as regional context; this module never reconstructs CO2.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.special import expit

from . import phase_optical_regions as regional
from .phase_g3_production import CO2_SHA256
from .phase_gas_flatfield import FLATFIELD_FEATURE_NAMES
from .phase_gas_regional_context import build_shared_regional_context, prepare_conditional_gas
from .phase_pixel_area import area_filtered_gas, load_rig_classes, load_rig_pixel_area, validate_pixel_area
from .phase_rig_domain_inference import MaskedCpuFeaturePredictor, log_to_unit_rgb, masked_flatfield_features
from .project import sha256_file

CONTRACT = "ffac-g3-t2-source-exact-regional-inputs-v1"
SPACING_M = (.00016448, .00016448)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def verify_pixel_model(path, expected_sha256, *, training_revision="G3-T2"):
    if sha256_file(Path(path)) != expected_sha256:
        raise ValueError("G3-T2 pixel head changed")
    model = _read(path)
    if (training_revision not in ("G3-T2", "G3-T3")
            or model.get("variant") != "G3" or model.get("training_revision") != training_revision
            or model.get("feature_names") != list(FLATFIELD_FEATURE_NAMES)
            or model.get("fit", {}).get("converged") is not True):
        raise ValueError("G3-T2 needs its converged, unchanged four-feature expression")
    return model


def prepare_gas_inputs(frozen_candidate, pixel_model, current_log, aligned_baseline_log,
                       domain, *, aligned_previous_log=None, spacing_m=SPACING_M,
                       source_baseline_log=None, h_baseline_to_current=None,
                       source_baseline_valid=None, feature_shape_hw=(2979, 5472),
                       settings=None, input_contract=CONTRACT):
    """Original masked rig context and two new conditional-gas columns only."""
    from .phase_learning import HierarchicalPhaseModel
    from .tools.train_ffac_gas_flatfield_variants import gas_probability
    root = Path(frozen_candidate)
    if sha256_file(root / "models/generic/co2_presence_signal.json") != CO2_SHA256:
        raise ValueError("G3-T2 context changed the frozen CO2 head")
    valid = np.asarray(domain)
    if valid.dtype != bool or valid.ndim != 2:
        raise ValueError("G3-T2 context requires the exact boolean rig grid")
    for values in (current_log, aligned_baseline_log, aligned_previous_log):
        if values is not None and (values.dtype != np.float32 or values.shape != (3, *valid.shape)):
            raise ValueError("G3-T2 context requires unchanged CHW float32 scientific pixels")
    valid = valid & np.isfinite(current_log).all(0) & np.isfinite(aligned_baseline_log).all(0)
    if aligned_previous_log is not None:
        valid &= np.isfinite(aligned_previous_log).all(0)
    current, _ = log_to_unit_rgb(current_log, valid)
    baseline, _ = log_to_unit_rgb(aligned_baseline_log, valid)
    previous = None if aligned_previous_log is None else log_to_unit_rgb(aligned_previous_log, valid)[0]
    model = HierarchicalPhaseModel.load(root)
    wrapper = MaskedCpuFeaturePredictor(model, baseline, valid, feature_shape_hw=feature_shape_hw)
    predictor = wrapper.predictor
    current_features = predictor.prepare_rgb(current)
    previous_features = predictor.baseline if previous is None else predictor.prepare_rgb(previous)
    presence = np.empty(valid.shape, np.float32)
    for first, last, features in predictor.feature_blocks(current_features, previous_features,
                                                          dynamic_available=previous is not None):
        presence[first:last] = expit(features @ predictor.weights[:, 0] + predictor.bias[0]).reshape(
            last-first, valid.shape[1])
    ff, ff_valid = masked_flatfield_features(current_log, aligned_baseline_log, valid,
        source_baseline_log=source_baseline_log, h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid)
    pixel = gas_probability(pixel_model, np.empty((*valid.shape, 0), np.float32),
                            np.where(ff_valid[..., None], ff, 0))
    pixel[~ff_valid] = np.nan
    if settings is None:
        settings = regional.Settings(**_read(root / "regional_model.json")["settings"])
    shared = build_shared_regional_context(current, baseline, presence, valid, spacing_m, settings)
    prepared = prepare_conditional_gas(shared, np.nan_to_num(pixel, nan=0.))
    return SimpleNamespace(prepared=prepared, presence=presence, g3_pixel=pixel,
        flat_valid=ff_valid, domain=valid, photo_sand_domain=np.asarray(domain, bool),
        feature_support=valid, spacing_m=tuple(spacing_m), metadata=dict(
            contract=input_contract, cpu_only=True, frozen_co2_head_sha256=CO2_SHA256,
            co2_inference_performed=False, co2_presence_context_only=True,
            context_columns_unchanged=[i for i in range(20) if i not in (1, 11)],
            gas_dependent_columns=[1, 11], optical_context_native_shape_hw=list(feature_shape_hw),
            optical_spatial_sigma_px=float(predictor.spatial_sigma),
            regional_context_shape_hw=list(prepared["domain"].shape)))


def replace_gas_from_scores(base_labels, pixel_score, regional_score, modelled, pixel_area_mm2,
                            *, filled_support=None):
    """Threshold and minimize observed gas, preserving frozen CO2/support."""
    labels, pixel, score, support = map(np.asarray, (base_labels, pixel_score, regional_score, modelled))
    if (labels.dtype != np.uint8 or labels.ndim != 2 or support.dtype != bool
            or any(v.shape != labels.shape for v in (pixel, score, support))
            or not np.isin(labels, (0, 1, 2, 255)).all()
            or not np.isfinite(score[support]).all()
            or not np.isfinite(pixel[support]).all()):
        raise ValueError("Gas replacement lost the frozen label/support/probability grid")
    area = validate_pixel_area(pixel_area_mm2, labels.shape)
    co2 = np.isin(labels, (1, 2))
    selected = co2 & support & (pixel >= .7) & (score >= .6)
    filled = np.zeros(labels.shape, bool) if filled_support is None else np.asarray(filled_support)
    if filled.dtype != bool or filled.shape != labels.shape or np.any(filled & support):
        raise ValueError("Inherited fill flags must be disjoint from fresh modelled gas support")
    filled_co2 = filled & co2
    if filled_co2.any():
        from scipy.ndimage import distance_transform_edt
        if not support.any():
            raise ValueError("Gas partition filling needs a supported source donor")
        _distance, indices = distance_transform_edt(~support, return_indices=True)
        nearest = selected[tuple(indices)]
        selected[filled_co2] = nearest[filled_co2]
    gas, minimum = area_filtered_gas(selected, area, minimum_area_mm2=2.5, connectivity=4)
    result = labels.copy()
    revised = co2 & (support | filled)
    result[revised] = np.where(gas[revised], 2, 1)
    if not np.array_equal(np.isin(result, (1, 2)), co2) or not np.array_equal(result[~revised], labels[~revised]):
        raise AssertionError("Gas-only replacement changed frozen CO2 or unsupported categories")
    audit = dict(contract=CONTRACT, frozen_co2_union_reused=True,
        co2_union_changed_pixels=0, unsupported_unfilled_categories_preserved=True,
        unsupported_gas_pixels_preserved=int(np.count_nonzero((labels == 2) & ~(support | filled))),
        inherited_filled_co2_pixels=int(filled_co2.sum()),
        fill_policy="nearest fresh modelled conditional partition inside frozen CO2 only; support flags unchanged",
        minimum_scope="full combined modelled-and-declared-filled gas graph before B",
        pixel_threshold=.7, regional_threshold=.6, gas_minimum=minimum,
        gas_minimum_before_rule_b=True, rule_b_applied=False, top_clip_applied=False,
        changed_phase_pixels=int(np.count_nonzero(result != labels)),
        selected_pre_minimum_pixels=int(selected.sum()), selected_post_minimum_pixels=int(gas.sum()))
    return result, audit, dict(frozen_co2=co2, modelled_gas_support=support,
                               inherited_gas_fill=filled_co2,
                               pre_minimum_gas=selected, post_minimum_gas=gas)


def evaluate_gas_frame(frozen_candidate, pixel_model_path, regional_model_path,
                       current_log, aligned_baseline_log, base_labels, modelled,
                       pixel_area_mm2, *, expected_pixel_sha256, expected_regional_sha256,
                       aligned_previous_log=None, context_domain=None, spacing_m=SPACING_M,
                       source_baseline_log=None, h_baseline_to_current=None,
                       source_baseline_valid=None, feature_shape_hw=(2979, 5472),
                       base_frame_binding=None, filled_support=None,
                       training_revision="G3-T2", input_contract=CONTRACT):
    """Return gas labels before B/top/floor; never invoke CO2 reconstruction.

    Root orchestration supplies already verified source floats, labels, classes
    and area. Unsupported base classes remain exact and are separately audited.
    """
    from .phase_g3_probability_layers import _validate_refit
    pixel = verify_pixel_model(pixel_model_path, expected_pixel_sha256, training_revision=training_revision)
    path = Path(regional_model_path)
    if sha256_file(path) != expected_regional_sha256:
        raise ValueError("G3-T2 regional gas head changed")
    model, frozen = _read(path), _read(Path(frozen_candidate) / "regional_model.json")
    _validate_refit(frozen, model)
    from copy import deepcopy
    original_other, revised_other = deepcopy(frozen), deepcopy(model)
    original_other["heads"].pop("gas_within_co2")
    revised_other["heads"].pop("gas_within_co2")
    if original_other != revised_other:
        raise ValueError("G3-T2 changed frozen regional metadata or another head")
    if model["heads"]["gas_within_co2"].get("converged") is not True:
        raise ValueError("G3-T2 regional head did not converge")
    inputs = prepare_gas_inputs(frozen_candidate, pixel, current_log, aligned_baseline_log,
        modelled if context_domain is None else context_domain,
        aligned_previous_log=aligned_previous_log, spacing_m=spacing_m,
        source_baseline_log=source_baseline_log, h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid, feature_shape_hw=feature_shape_hw,
        settings=regional.Settings(**model["settings"]), input_contract=input_contract)
    support = np.asarray(modelled, bool) & inputs.domain & inputs.flat_valid
    score = regional.probability(inputs.prepared["features"], model["heads"]["gas_within_co2"])
    score = cv2.resize(score, np.asarray(base_labels).shape[::-1], interpolation=cv2.INTER_LINEAR)
    result, audit, masks = replace_gas_from_scores(base_labels, inputs.g3_pixel, score, support,
                                                 pixel_area_mm2, filled_support=filled_support)
    audit.update(inputs.metadata, pixel_model_sha256=expected_pixel_sha256,
                 regional_model_sha256=expected_regional_sha256,
                 base_frame_binding=base_frame_binding,
                 context_domain_mask_sha256=sha256(np.packbits(inputs.domain, bitorder="little").tobytes()).hexdigest(),
                 frozen_co2_union_sha256=sha256(np.packbits(masks["frozen_co2"], bitorder="little").tobytes()).hexdigest())
    if training_revision != "G3-T2":
        audit["training_revision"] = training_revision
    return result, audit, masks


def prepare_frame_inputs(base_candidate, frame, *, original_plan=None, baseline_cache=None,
                          load_area=True):
    """Load the sealed seam frame's exact current/B/predecessor RAW lineage.

    ``frame`` is its already source-bound receipt. The predecessor descriptor
    comes from the source plan even when that acquisition is off the timeline.
    Current pixels are never warped. Area/classes are verified on the full grid.
    """
    from .phase_ffac_rig_camera_runtime import bound_path
    from .tools import run_ffac_rig_camera_inference as producer
    root = Path(base_candidate)
    plan = _read(root / "old_domain_seam_plan.json") if original_plan is None else original_plan
    source = frame["binding"]["source"]
    key = f"{source['run']}/frame_{source['frame_index']:03d}"
    original = _read(bound_path(plan["source_producer_frames"][key]))
    original_source = original["binding"]["source"]
    if (frame.get("complete") is not True or original.get("complete") is not True
            or frame["source_sha256"] != original_source["source_sha256"]
            or frame["elapsed_h"] != original_source["elapsed_h"]
            or original.get("source_sha256", original_source["source_sha256"]) != frame["source_sha256"]
            or original.get("elapsed_h", original_source["elapsed_h"]) != frame["elapsed_h"]
            or frame["full_extended_scientific"] != original["full_extended_scientific"]
            or source != original["binding"]["source"]):
        raise ValueError("G3-T2 frame differs from its source-exact seam lineage")
    classes, classes_binding = load_rig_classes(frame["rig_domain_revision"])
    rig = classes["rig_domain"]
    revision = frame["rig_domain_revision"]
    area = load_rig_pixel_area(revision, rig.shape, origin_xy=revision["native_origin_xy"]) if load_area else None
    current, current_valid = producer.compose_extended_scientific(original["full_extended_scientific"])
    baseline_key = _digest(original["baseline_extended_scientific"])
    if baseline_cache is not None and baseline_cache.get("key") == baseline_key:
        baseline, baseline_valid = baseline_cache["log"], baseline_cache["valid"]
    else:
        baseline, baseline_valid = producer.compose_extended_scientific(original["baseline_extended_scientific"])
        if baseline_cache is not None:
            baseline_cache.clear()
            baseline_cache.update(key=baseline_key, log=baseline, valid=baseline_valid)
    prepared = _read(bound_path(original["full_extended_scientific"]["binding"]["prepared_source"]))
    h = np.asarray(prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
    h_full = producer.canvas_homography(h)
    aligned, aligned_valid = producer.align_scientific(baseline, baseline_valid, h_full, target_shape_hw=rig.shape)
    previous = None
    previous_valid = np.ones(rig.shape, bool)
    previous_binding = None
    task = source["previous_scientific_task_id"]
    if task is not None:
        previous_asset = plan.get("previous_extended_scientific_sources", {}).get(key,
            plan["extended_scientific_sources"].get(task))
        if previous_asset is None:
            # The seam replay omitted source-changed frames from its numerical
            # replay inventory. Their predecessor must still be an exact saved
            # producer source, never a neighbouring frame chosen by index.
            matches = [row for row in plan["source_inventory"] if row["scientific_task_id"] == task]
            if len(matches) != 1:
                raise ValueError("G3-T2 frame lost its unique exact source predecessor")
            previous_source = matches[0]
            previous_key = f"{previous_source['run']}/frame_{previous_source['frame_index']:03d}"
            previous_receipt_asset = plan["source_producer_frames"][previous_key]
            previous_receipt = _read(bound_path(previous_receipt_asset))
            if (previous_receipt.get("complete") is not True
                    or previous_receipt["binding"]["source"] != previous_source):
                raise ValueError("G3-T2 predecessor producer receipt changed its exact source")
            descriptor = previous_receipt["full_extended_scientific"]
            if descriptor["source_sha256"] != previous_source["source_sha256"]:
                raise ValueError("G3-T2 predecessor RAW belongs to another acquisition")
            previous_binding = dict(source_producer_frame=previous_receipt_asset,
                                    component="full_extended_scientific")
        else:
            descriptor = _read(bound_path(previous_asset))
            previous_binding = previous_asset
        previous_log, supported = producer.compose_extended_scientific(descriptor)
        previous_prepared = _read(bound_path(descriptor["binding"]["prepared_source"]))
        h_previous = np.asarray(previous_prepared["geometry_frame"]["H_v5_B_to_frame_native"], np.float64)
        previous, previous_valid = producer.align_scientific(previous_log, supported,
            producer.canvas_homography(h @ np.linalg.inv(h_previous)), target_shape_hw=rig.shape)
    context_domain = rig & current_valid & aligned_valid & previous_valid
    return dict(current_log=current, aligned_baseline_log=aligned,
        source_baseline_log=baseline, source_baseline_valid=baseline_valid,
        h_baseline_to_current=h_full, aligned_previous_log=previous,
        context_domain=context_domain, modelled=classes["modelled"],
        filled_support=classes["filled_observed"] | classes["filled_unobserved"],
        pixel_area_mm2=None if area is None else area.values_mm2, classes=classes,
        base_frame_binding=dict(source_sha256=frame["source_sha256"], elapsed_h=frame["elapsed_h"],
            current_extended_scientific_sha256=_digest(original["full_extended_scientific"]),
            baseline_extended_scientific_sha256=_digest(original["baseline_extended_scientific"]),
            previous_extended_scientific=previous_binding, rig_classes=classes_binding,
            area=None if area is None else area.provenance, source_geometry_receipt=source["rig_geometry"]["source_geometry_receipt"],
            source_timeline_unchanged=True, current_pixels_unwarped=True))


class NativeT2FrameEvaluator:
    """One bound rig source revision, keeping only its original baseline."""
    def __init__(self, candidate, g3_model_path, native_manifest_path, *, run=None,
                 threads=1, source=None, independent_reference_frames=None,
                 training_revision="G3-T2", input_contract=CONTRACT):
        from .phase_ffac_rig_camera_runtime import load_native_source
        if not 1 <= int(threads) <= 16:
            raise ValueError("Invalid bounded CPU thread allocation")
        self.candidate, self.path = Path(candidate), Path(native_manifest_path)
        self.source = Path(source) if source else self.path.parent.parent.parent
        self.run, self.threads = str(run).casefold(), int(threads)
        self.state = load_native_source(self.source, self.run, verify_geometry=False)
        self.native = self.state["native"]
        if self.state["native_path"].resolve() != self.path.resolve():
            raise ValueError("Regional T2 evaluator lost its exact source revision")
        self.pixel_path = Path(g3_model_path)
        self.pixel = verify_pixel_model(self.pixel_path, sha256_file(self.pixel_path), training_revision=training_revision)
        self.input_contract = input_contract
        self.plan = _read(self.candidate / "old_domain_seam_plan.json")
        self.baseline_cache = {}

    def evaluate(self, frame_index, *, export_frozen_co2=False):
        if export_frozen_co2:
            raise ValueError("Training evaluation cannot read frozen maps")
        index = int(frame_index)
        if index != frame_index or not 0 <= index < len(self.native["frame_sources"]):
            raise ValueError("Regional T2 frame is outside its source timeline")
        record, row = self.native["frame_sources"][index], self.state["inputs"]["frames"][index]
        receipt = _read(self.candidate / "rig_camera_frames" / self.run.upper() / f"frame_{index:03d}.json")
        if (receipt["source_sha256"], receipt["elapsed_h"]) != (record["source_sha256"], record["elapsed_h"]):
            raise ValueError("Regional T2 photograph and source receipt differ")
        values = prepare_frame_inputs(self.candidate, receipt, original_plan=self.plan,
                                       baseline_cache=self.baseline_cache, load_area=False)
        classes = values["classes"]
        shape = classes["rig_domain"].shape
        origin = receipt["rig_domain_revision"]["native_origin_xy"]
        frame = prepare_gas_inputs(self.candidate, self.pixel, values["current_log"],
            values["aligned_baseline_log"], values["context_domain"],
            aligned_previous_log=values["aligned_previous_log"], source_baseline_log=values["source_baseline_log"],
            source_baseline_valid=values["source_baseline_valid"], h_baseline_to_current=values["h_baseline_to_current"],
            input_contract=self.input_contract)
        # Review normalization remains photographic; sampling indexes the full
        # phase grid directly, including its explicitly declared extra rows.
        frame.native = dict(canvas_wh=list(self.native["canvas_wh"]),
            content_box_xywh=[0, 0, shape[1], shape[0]], analysis_image_shape_hw=list(shape))
        frame.flat_valid &= classes["modelled"]
        frame.metadata.update(run=self.run, frame_index=index,
            elapsed_h=float(record["elapsed_h"]), source_sha256=record["source_sha256"],
            native_manifest_path=str(self.path), native_manifest_sha256=sha256_file(self.path),
            source_context=values["base_frame_binding"],
            native_origin_xy=list(origin), phase_shape_hw=list(shape),
            annotation_policy="original normalized photographic canvas; full phase-grid sampling origin zero",
            pixel_model_sha256=sha256_file(self.pixel_path))
        return frame


def source_evaluator(binding, source, threads, *, training_revision="G3-T2", input_contract=CONTRACT):
    """Dispatch from a declared source contract, never from a run-name guess."""
    from .phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    native = _read(source.asset_manifest_path)
    cls = NativeT2FrameEvaluator
    candidate = binding["frozen_candidate"]
    if (native.get(SOURCE_KEY) or {}).get("contract") != SOURCE_CONTRACT:
        from .phase_g3_probability_layers import NativeG3FrameEvaluator
        cls = NativeG3FrameEvaluator
        candidate = binding["legacy_context_candidate"]
    kwargs = {}
    if cls is NativeT2FrameEvaluator:
        kwargs.update(training_revision=training_revision, input_contract=input_contract)
    elif training_revision != "G3-T2":
        verify_pixel_model(binding["g3_model_path"], binding["g3_model_sha256"], training_revision=training_revision)
    return cls(candidate=Path(candidate), g3_model_path=Path(binding["g3_model_path"]),
               native_manifest_path=source.asset_manifest_path, run=source.run,
               threads=threads, source=source.review_root,
               independent_reference_frames=source.independent_reference_frames, **kwargs)


__all__ = ["CONTRACT", "NativeT2FrameEvaluator", "prepare_gas_inputs", "prepare_frame_inputs", "evaluate_gas_frame",
           "replace_gas_from_scores", "source_evaluator", "verify_pixel_model"]
