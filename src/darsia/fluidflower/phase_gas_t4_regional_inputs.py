"""Epoch-bound T4 inputs using the unchanged source-exact G3 gas graph.

The series project's epoch table selects the gas heads. CO2 categories,
optical context, geometry and measured area remain frozen parent inputs.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from . import phase_gas_t2_regional_inputs as shared
from . import phase_optical_regions as regional
from .project import sha256_file

CONTRACT = "ffac-g3-t4-source-exact-regional-inputs-v1"
BUNDLE_CONTRACT = "ffac-g3-t4-epoch-regional-bundle-v1"
SPACING_M = shared.SPACING_M
FLATFIELD_FEATURE_NAMES = shared.FLATFIELD_FEATURE_NAMES
prepare_frame_inputs = shared.prepare_frame_inputs


def verify_pixel_model(path, expected_sha256, *, light_epoch_id=None):
    if sha256_file(Path(path)) != expected_sha256:
        raise ValueError("G3-T4 pixel head changed")
    model = shared._read(path)
    if (model.get("variant") != "G3" or model.get("training_revision") != "G3-T4"
            or model.get("feature_names") != list(FLATFIELD_FEATURE_NAMES)
            or model.get("fit", {}).get("converged") is not True
            or not model.get("light_epoch_id")
            or (light_epoch_id is not None and model["light_epoch_id"] != light_epoch_id)):
        raise ValueError("G3-T4 needs its own epoch's converged four-feature expression")
    return model


def verify_regional_model(path, expected_sha256, parent):
    from .phase_g3_probability_layers import _validate_refit
    if sha256_file(Path(path)) != expected_sha256:
        raise ValueError("G3-T4 regional gas head changed")
    frozen = shared._read(Path(parent) / "regional_model.json")
    model = shared._read(path)
    _validate_refit(frozen, model)
    before, after = deepcopy(frozen), deepcopy(model)
    before["heads"].pop("gas_within_co2")
    after["heads"].pop("gas_within_co2")
    if before != after:
        raise ValueError("G3-T4 changed frozen regional metadata or another head")
    if model["heads"]["gas_within_co2"].get("converged") is not True:
        raise ValueError("G3-T4 regional head did not converge")
    return model


def verify_epoch_bundle(bundle_path, project, parent):
    """Verify both gas pairs and their exact project-owned epoch declaration."""
    from . import phase_gas_t4_training as training
    bundle = shared._read(bundle_path)
    parent = Path(parent).resolve()
    table_path = Path(bundle["light_epoch_table_path"]).resolve()
    project = Path(project).resolve()
    if (not project.exists() or (project.is_file() and table_path != project)
            or (project.is_dir() and not table_path.is_relative_to(project))):
        raise ValueError("T4 light epoch table is outside its series project")
    if (bundle.get("contract") != BUNDLE_CONTRACT or bundle.get("cpu_only") is not True
            or bundle.get("converged") is not True
            or Path(bundle["frozen_candidate"]).resolve() != parent
            or sha256_file(table_path) != bundle["light_epoch_table_sha256"]
            or sha256_file(Path(bundle["pixel_fit_receipt_path"])) != bundle["pixel_fit_receipt_sha256"]
            or sha256_file(parent / "models/generic/co2_presence_signal.json") != shared.CO2_SHA256
            or sha256_file(parent / "regional_model.json") != bundle["frozen_regional_sha256"]):
        raise ValueError("T4 regional bundle changed its parent, table or frozen CO2")
    table = training.load_light_epochs(table_path)
    epochs = bundle["epochs"]
    if set(epochs) != set(table["epochs"]) or len(epochs) != 2:
        raise ValueError("T4 regional bundle needs exactly the two project epochs")
    for epoch_id, entry in epochs.items():
        pixel = verify_pixel_model(entry["pixel_model_path"], entry["pixel_model_sha256"],
                                   light_epoch_id=epoch_id)
        verify_regional_model(entry["regional_model_path"], entry["regional_model_sha256"], parent)
        receipt_path = Path(entry["regional_fit_receipt_path"])
        if sha256_file(receipt_path) != entry["regional_fit_receipt_sha256"]:
            raise ValueError("T4 regional fit receipt changed")
        receipt = shared._read(receipt_path)
        binding = receipt["binding"]["extraction_binding"]
        inventory = binding["inventory"]
        runs = inventory["training_runs"]
        if (receipt.get("converged") is not True or receipt.get("cpu_only") is not True
                or receipt.get("rounds_2_3_4_read") is not False
                or receipt.get("test_labels_read") is not False
                or receipt.get("t2_bt2_bt3_labels_read") is not False
                or receipt.get("light_epoch_id") != epoch_id
                or receipt["regional_model_sha256"] != entry["regional_model_sha256"]
                or binding["g3_model_sha256"] != entry["pixel_model_sha256"]
                or binding["light_epoch_table_sha256"] != bundle["light_epoch_table_sha256"]
                or not runs or any(table["run_to_epoch"].get(run.casefold()) != epoch_id for run in runs)
                or pixel["training_provenance"]["epoch_label_sha256"] != inventory["training_label_sha256"]
                or pixel["training_provenance"].get("light_epochs", {}).get("sha256") != bundle["light_epoch_table_sha256"]
                or sorted(pixel["training_provenance"].get("training_runs", [])) != sorted(runs)):
            raise ValueError("T4 gas heads lost independent own-epoch training provenance")
    return dict(light_epoch_table=table, light_epoch_table_path=str(table_path),
                light_epoch_table_sha256=bundle["light_epoch_table_sha256"], epochs=epochs)


def prepare_gas_inputs(*args, **kwargs):
    return shared.prepare_gas_inputs(*args, **kwargs, input_contract=CONTRACT)


def replace_gas_from_scores(*args, **kwargs):
    result, audit, masks = shared.replace_gas_from_scores(*args, **kwargs)
    audit["contract"] = CONTRACT
    return result, audit, masks


def evaluate_gas_frame(frozen_candidate, pixel_model_path, regional_model_path,
                       current_log, aligned_baseline_log, base_labels, modelled,
                       pixel_area_mm2, *, expected_pixel_sha256, expected_regional_sha256,
                       aligned_previous_log=None, context_domain=None, spacing_m=SPACING_M,
                       source_baseline_log=None, h_baseline_to_current=None,
                       source_baseline_valid=None, feature_shape_hw=(2979, 5472),
                       base_frame_binding=None, filled_support=None, light_epoch_id=None):
    """Evaluate the selected epoch's gas pair inside the frozen CO2 union."""
    pixel = verify_pixel_model(pixel_model_path, expected_pixel_sha256,
                               light_epoch_id=light_epoch_id)
    model = verify_regional_model(regional_model_path, expected_regional_sha256, frozen_candidate)
    frame = prepare_gas_inputs(frozen_candidate, pixel, current_log, aligned_baseline_log,
        modelled if context_domain is None else context_domain,
        aligned_previous_log=aligned_previous_log, spacing_m=spacing_m,
        source_baseline_log=source_baseline_log, h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid, feature_shape_hw=feature_shape_hw,
        settings=regional.Settings(**model["settings"]))
    support = np.asarray(modelled, bool) & frame.domain & frame.flat_valid
    score = regional.probability(frame.prepared["features"], model["heads"]["gas_within_co2"])
    score = cv2.resize(score, np.asarray(base_labels).shape[::-1], interpolation=cv2.INTER_LINEAR)
    result, audit, masks = replace_gas_from_scores(base_labels, frame.g3_pixel, score, support,
                                                 pixel_area_mm2, filled_support=filled_support)
    from hashlib import sha256
    audit.update(frame.metadata, training_revision="G3-T4", light_epoch_id=pixel["light_epoch_id"],
        pixel_model_sha256=expected_pixel_sha256, regional_model_sha256=expected_regional_sha256,
        base_frame_binding=base_frame_binding,
        context_domain_mask_sha256=sha256(np.packbits(frame.domain, bitorder="little").tobytes()).hexdigest(),
        frozen_co2_union_sha256=sha256(np.packbits(masks["frozen_co2"], bitorder="little").tobytes()).hexdigest())
    return result, audit, masks


class NativeT4FrameEvaluator(shared.NativeT2FrameEvaluator):
    """Reuse exact rig evaluation with only a compact per-source context plan."""
    def __init__(self, candidate, g3_model_path, native_manifest_path, *, run, threads,
                 source, independent_reference_frames=None, context_plan_path,
                 context_plan_sha256, light_epoch_id):
        from .phase_ffac_rig_camera_runtime import load_native_source
        if not 1 <= int(threads) <= 16:
            raise ValueError("Invalid bounded CPU thread allocation")
        self.candidate, self.path = Path(candidate), Path(native_manifest_path)
        self.source = Path(source)
        self.run, self.threads = str(run).casefold(), int(threads)
        self.state = load_native_source(self.source, self.run, verify_geometry=False)
        self.native = self.state["native"]
        if self.state["native_path"].resolve() != self.path.resolve():
            raise ValueError("Regional T4 evaluator lost its exact source revision")
        self.pixel_path = Path(g3_model_path)
        self.pixel = verify_pixel_model(self.pixel_path, sha256_file(self.pixel_path),
                                       light_epoch_id=light_epoch_id)
        self.input_contract = CONTRACT
        if sha256_file(Path(context_plan_path)) != context_plan_sha256:
            raise ValueError("T4 compact source context plan changed")
        self.plan = shared._read(context_plan_path)
        self.baseline_cache = {}


def source_evaluator(binding, source, threads):
    """Source-contract dispatch, with no run ranges embedded in the code."""
    from .phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    verify_pixel_model(binding["g3_model_path"], binding["g3_model_sha256"],
                       light_epoch_id=binding["light_epoch_id"])
    if source.run.casefold() not in binding["inventory"]["training_runs"]:
        raise ValueError("Regional T4 source belongs to another light epoch")
    native = shared._read(source.asset_manifest_path)
    kwargs = dict(candidate=Path(binding["legacy_context_candidate"]),
        g3_model_path=Path(binding["g3_model_path"]), native_manifest_path=source.asset_manifest_path,
        run=source.run, threads=threads, source=source.review_root,
        independent_reference_frames=source.independent_reference_frames)
    if (native.get(SOURCE_KEY) or {}).get("contract") == SOURCE_CONTRACT:
        return NativeT4FrameEvaluator(**dict(kwargs, candidate=Path(binding["frozen_candidate"]),
            context_plan_path=binding["source_context_plan_path"],
            context_plan_sha256=binding["source_context_plan_sha256"],
            light_epoch_id=binding["light_epoch_id"]))
    from .phase_g3_probability_layers import NativeG3FrameEvaluator
    return NativeG3FrameEvaluator(**kwargs)


__all__ = ["CONTRACT", "BUNDLE_CONTRACT", "verify_pixel_model", "verify_regional_model",
           "verify_epoch_bundle", "prepare_gas_inputs", "prepare_frame_inputs", "evaluate_gas_frame",
           "replace_gas_from_scores", "source_evaluator", "NativeT4FrameEvaluator"]
