"""Joint T4b pixel boundary with a lamp offset and an epoch-free regional head."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from . import phase_gas_t2_regional_inputs as shared
from . import phase_gas_t4_regional_inputs as t4
from . import phase_gas_t4_training as sources
from . import phase_optical_regions as regional
from .project import sha256_file

CONTRACT = "ffac-g3-t4b-source-exact-regional-inputs-v1"
BUNDLE_CONTRACT = "ffac-g3-t4b-joint-regional-bundle-v1"
FEATURE_NAMES = (*shared.FLATFIELD_FEATURE_NAMES, "new_lamp_indicator")
SPACING_M = shared.SPACING_M
prepare_frame_inputs = shared.prepare_frame_inputs
replace_gas_from_scores = shared.replace_gas_from_scores


def compact_candidate_source_plan(*args, **kwargs):
    from .phase_gas_t4_regional_refit import compact_candidate_source_plan as compact
    return compact(*args, **kwargs)


def verify_pixel_model(path, expected_sha256):
    if sha256_file(Path(path)) != expected_sha256:
        raise ValueError("T4b joint pixel model changed")
    model = shared._read(path)
    fit = model.get("fit", {})
    if (model.get("variant") != "G3" or model.get("training_revision") != "G3-T4b"
            or model.get("feature_names") != list(FEATURE_NAMES)
            or model.get("lamp_shift_only") is not True or fit.get("converged") is not True or fit.get("backend") != "cpu"
            or any(np.asarray(fit.get(k, [])).shape != (5,) for k in ("mean", "scale", "coefficients"))
            or not all(np.isfinite(fit[k]).all() for k in ("mean", "scale", "coefficients"))
            or np.any(np.asarray(fit["scale"]) <= 0) or not np.isfinite(fit["intercept"])):
        raise ValueError("T4b requires one converged five-feature shift-only pixel head")
    return model


def epoch_pixel_model(model, light_epoch_id):
    """Fold a known indicator into the intercept, without fitting another head."""
    if light_epoch_id not in ("rig_light", "new_lamps"):
        raise ValueError("T4b needs a project-declared light epoch")
    result = deepcopy(model)
    fit = result["fit"]
    fit["intercept"] += fit["coefficients"][4] * (
        float(light_epoch_id == "new_lamps") - fit["mean"][4]) / fit["scale"][4]
    for key in ("mean", "scale", "coefficients"):
        fit[key] = fit[key][:4]
    result["feature_names"] = list(shared.FLATFIELD_FEATURE_NAMES)
    return result


def prepare_gas_inputs(*args, light_epoch_id, **kwargs):
    args = list(args)
    args[1] = epoch_pixel_model(args[1], light_epoch_id)
    frame = shared.prepare_gas_inputs(*args, **kwargs, input_contract=CONTRACT)
    frame.metadata.update(training_revision="G3-T4b", light_epoch_id=light_epoch_id,
                          lamp_indicator_in_regional_features=False,
                          pixel_feature_names=list(FEATURE_NAMES))
    return frame


def verify_regional_model(path, expected_sha256, parent):
    model = t4.verify_regional_model(path, expected_sha256, parent)
    if model["feature_names"] != list(regional.FEATURE_NAMES) or len(model["feature_names"]) != 20:
        raise ValueError("T4b regional model must keep its original20 features with no epoch input")
    return model


def verify_joint_bundle(bundle_path, project, parent):
    bundle = shared._read(bundle_path)
    parent, project = Path(parent).resolve(), Path(project).resolve()
    table_path = Path(bundle["light_epoch_table_path"]).resolve()
    if (not project.exists() or (project.is_file() and table_path != project)
            or (project.is_dir() and not table_path.is_relative_to(project))):
        raise ValueError("T4b epoch table is outside the series project")
    if (bundle.get("contract") != BUNDLE_CONTRACT or bundle.get("cpu_only") is not True
            or bundle.get("converged") is not True
            or Path(bundle["frozen_candidate"]).resolve() != parent
            or sha256_file(table_path) != bundle["light_epoch_table_sha256"]
            or sha256_file(parent / "regional_model.json") != bundle["frozen_regional_sha256"]
            or sha256_file(parent / "models/generic/co2_presence_signal.json") != shared.CO2_SHA256):
        raise ValueError("T4b bundle changed its frozen parent or project table")
    table = sources.load_light_epochs(table_path)
    pixel = verify_pixel_model(bundle["pixel_model_path"], bundle["pixel_model_sha256"])
    pixel_receipt_path = Path(bundle["pixel_fit_receipt_path"])
    if sha256_file(pixel_receipt_path) != bundle["pixel_fit_receipt_sha256"]:
        raise ValueError("T4b pixel fit receipt changed")
    pixel_receipt = shared._read(pixel_receipt_path)
    if (pixel_receipt.get("converged") is not True or pixel_receipt.get("cpu_only") is not True
            or pixel_receipt["model_sha256"] != bundle["pixel_model_sha256"]):
        raise ValueError("T4b bundle changed its joint pixel fit receipt")
    verify_regional_model(bundle["regional_model_path"], bundle["regional_model_sha256"], parent)
    receipt_path = Path(bundle["regional_fit_receipt_path"])
    if sha256_file(receipt_path) != bundle["regional_fit_receipt_sha256"]:
        raise ValueError("T4b regional fit receipt changed")
    receipt = shared._read(receipt_path)
    binding = receipt["binding"]["extraction_binding"]
    inventory = binding["inventory"]
    runs = inventory["training_runs"]
    provenance = pixel["training_provenance"]
    if (receipt.get("converged") is not True or receipt.get("cpu_only") is not True or receipt.get("regional_epoch_input") is not False
            or any(receipt.get(k) is not False for k in ("rounds_2_3_4_read", "test_labels_read", "t2_bt2_bt3_labels_read"))
            or receipt["regional_model_sha256"] != bundle["regional_model_sha256"]
            or binding["g3_model_sha256"] != bundle["pixel_model_sha256"]
            or binding["light_epoch_table_sha256"] != bundle["light_epoch_table_sha256"]
            or provenance["combined_label_sha256"] != inventory["training_label_sha256"]
            or provenance["light_epochs"]["sha256"] != bundle["light_epoch_table_sha256"]
            or sorted(provenance["training_runs"]) != sorted(runs)
            or not runs or any(run not in table["run_to_epoch"] for run in runs)):
        raise ValueError("T4b joint heads lost their all-run training provenance")
    return dict(light_epoch_table=table, light_epoch_table_path=str(table_path),
                light_epoch_table_sha256=bundle["light_epoch_table_sha256"], **{
                    k: bundle[k] for k in ("pixel_model_path", "pixel_model_sha256", "regional_model_path", "regional_model_sha256",
                        "regional_fit_receipt_path", "regional_fit_receipt_sha256", "pixel_fit_receipt_path", "pixel_fit_receipt_sha256")})


def evaluate_gas_frame(frozen_candidate, pixel_model_path, regional_model_path,
                       current_log, aligned_baseline_log, base_labels, modelled,
                       pixel_area_mm2, *, expected_pixel_sha256, expected_regional_sha256,
                       light_epoch_id=None, run=None, light_epochs=None, aligned_previous_log=None, context_domain=None,
                       spacing_m=SPACING_M, source_baseline_log=None, h_baseline_to_current=None,
                       source_baseline_valid=None, feature_shape_hw=(2979, 5472),
                       base_frame_binding=None, filled_support=None):
    if run is not None:
        if light_epochs is None or "receipt" not in light_epochs:
            raise ValueError("T4b requires the SHA-bound project light table with run selection")
        declared = light_epochs["run_to_epoch"][str(run).casefold()]
        if light_epoch_id is not None and light_epoch_id != declared:
            raise ValueError("T4b run and declared light epoch disagree")
        light_epoch_id = declared
    pixel = verify_pixel_model(pixel_model_path, expected_pixel_sha256)
    if run is None or pixel["training_provenance"]["light_epochs"]["sha256"] != light_epochs["receipt"]["sha256"]:
        raise ValueError("T4b inference changed the joint pixel head's project light table")
    model = verify_regional_model(regional_model_path, expected_regional_sha256, frozen_candidate)
    frame = prepare_gas_inputs(frozen_candidate, pixel, current_log, aligned_baseline_log,
        modelled if context_domain is None else context_domain, light_epoch_id=light_epoch_id,
        aligned_previous_log=aligned_previous_log, spacing_m=spacing_m,
        source_baseline_log=source_baseline_log, h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid, feature_shape_hw=feature_shape_hw,
        settings=regional.Settings(**model["settings"]))
    support = np.asarray(modelled, bool) & frame.domain & frame.flat_valid
    score = regional.probability(frame.prepared["features"], model["heads"]["gas_within_co2"])
    score = cv2.resize(score, np.asarray(base_labels).shape[::-1], interpolation=cv2.INTER_LINEAR)
    result, audit, masks = replace_gas_from_scores(base_labels, frame.g3_pixel, score, support,
                                                 pixel_area_mm2, filled_support=filled_support)
    audit.update(frame.metadata, contract=CONTRACT, pixel_model_sha256=expected_pixel_sha256,
                 regional_model_sha256=expected_regional_sha256, base_frame_binding=base_frame_binding)
    return result, audit, masks


class NativeT4bFrameEvaluator(shared.NativeT2FrameEvaluator):
    def __init__(self, candidate, g3_model_path, native_manifest_path, *, run, threads,
                 source, context_plan_path, context_plan_sha256, light_epochs,
                 independent_reference_frames=None):
        from .phase_ffac_rig_camera_runtime import load_native_source
        if not 1 <= int(threads) <= 16:
            raise ValueError("Invalid bounded CPU thread allocation")
        self.candidate, self.path, self.source = Path(candidate), Path(native_manifest_path), Path(source)
        self.run, self.threads = str(run).casefold(), int(threads)
        self.state = load_native_source(self.source, self.run, verify_geometry=False)
        self.native = self.state["native"]
        if self.state["native_path"].resolve() != self.path.resolve():
            raise ValueError("T4b lost its exact native source revision")
        self.pixel_path = Path(g3_model_path)
        model = verify_pixel_model(self.pixel_path, sha256_file(self.pixel_path))
        self.pixel = epoch_pixel_model(model, light_epochs["run_to_epoch"][self.run])
        self.input_contract = CONTRACT
        if sha256_file(Path(context_plan_path)) != context_plan_sha256:
            raise ValueError("T4b compact source context changed")
        self.plan = shared._read(context_plan_path)
        self.baseline_cache = {}


def source_evaluator(binding, source, threads):
    from .phase_ffac_rig_camera_metadata import SOURCE_KEY, SOURCE_CONTRACT
    from .phase_g3_probability_layers import NativeG3FrameEvaluator
    table = sources.load_light_epochs(binding["light_epoch_table_path"])
    if table["receipt"]["sha256"] != binding["light_epoch_table_sha256"]:
        raise ValueError("T4b project epoch table changed")
    if source.run.casefold() not in binding["inventory"]["training_runs"]:
        raise ValueError("T4b source is outside the all-run training inventory")
    model = verify_pixel_model(binding["g3_model_path"], binding["g3_model_sha256"])
    if model["training_provenance"]["light_epochs"]["sha256"] != table["receipt"]["sha256"]:
        raise ValueError("T4b source and joint pixel head use different project light tables")
    kwargs = dict(candidate=Path(binding["legacy_context_candidate"]),
        g3_model_path=Path(binding["g3_model_path"]), native_manifest_path=source.asset_manifest_path,
        run=source.run, threads=threads, source=source.review_root,
        independent_reference_frames=source.independent_reference_frames)
    native = shared._read(source.asset_manifest_path)
    if (native.get(SOURCE_KEY) or {}).get("contract") == SOURCE_CONTRACT:
        return NativeT4bFrameEvaluator(**dict(kwargs, candidate=Path(binding["frozen_candidate"]),
            context_plan_path=binding["source_context_plan_path"],
            context_plan_sha256=binding["source_context_plan_sha256"], light_epochs=table))
    # The frozen four-input constructor supplies unchanged optical/source
    # preparation. Its scorer then receives the exact algebraic joint fit.
    from .phase_g3_regional_refit import G3_SHA256
    if sha256_file(Path(binding["base_g3_model_path"])) != G3_SHA256:
        raise ValueError("T4b legacy source lost the exact G3 constructor input")
    evaluator = NativeG3FrameEvaluator(**dict(kwargs, g3_model_path=Path(binding["base_g3_model_path"])))
    evaluator.g3_path = Path(binding["g3_model_path"])
    evaluator.g3_model = epoch_pixel_model(model, table["run_to_epoch"][source.run.casefold()])
    evaluator.binding.update(g3_model_path=str(evaluator.g3_path), g3_model_sha256=binding["g3_model_sha256"],
        training_revision="G3-T4b", lamp_indicator_in_regional_features=False,
        light_epoch_table_sha256=binding["light_epoch_table_sha256"])
    return evaluator
