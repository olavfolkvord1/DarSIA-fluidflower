"""Strict read-only Studio admission for a sealed stored-map L1/floor revision."""
from functools import lru_cache
import json
from pathlib import Path

from .project import sha256_file
from .review_storage import resolve_review_path


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("L1 neighbourhood review metadata must be an object")
    return value


def _require(value, message):
    if not value:
        raise ValueError(message)


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    from .phase_validation import _model_identity
    from .phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    from .phase_l1_neighbourhood_gas import approved_spec_settings
    from .phase_rig_bottom_review_policy import load_policy, T3_PURPOSE, T3_BLIND_RUNS
    from .tools.refit_ffac_g3_downstream import canonical_sha256
    from .tools.register_track1_studio_candidate import registration_warning
    from .tools.run_ffac_l1_neighbourhood_candidate import CONTRACT, KIND, PLAN, SEAL, WARNING

    root, registration_path = Path(root_text), Path(registration_text)
    registration = _read(registration_path)
    plan_path, seal_path = root / PLAN, root / SEAL
    plan, seal = _read(plan_path), _read(seal_path)
    identity = list(_model_identity(root))
    plan_sha, seal_sha = sha256_file(plan_path), sha256_file(seal_path)
    metadata = registration.get("metadata_sha256") or {}
    policy = load_policy(root)
    _require(registration.get("registration_kind") == KIND
             and registration_warning(root) == WARNING
             and registration.get("publication_scope") == "studio_validation_only"
             and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
             and registration.get("model_id") == identity[0]
             and registration.get("model_sha256") == identity[1]
             and all(registration.get(key) is False for key in
                     ("mass_eligible", "bottom_mass_allowed", "scientific_acceptance",
                      "deployment_allowed", "automatic_activation")),
             "L1 neighbourhood saved registration changed its identity or review-only restrictions")
    _require(metadata.get(PLAN) == plan_sha == registration.get("source_plan_sha256")
             and metadata.get(SEAL) == seal_sha == registration.get("production_receipt_sha256")
             and metadata.get("rig_bottom_review_policy.json") == sha256_file(root / "rig_bottom_review_policy.json"),
             "L1 neighbourhood registration lost its exact plan, seal or blind-policy binding")
    _require(plan.get("contract") == seal.get("contract") == CONTRACT
             and seal.get("complete") is True
             and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
             and seal.get("model_identity") == plan.get("model_identity") == identity
             and seal.get("pipeline_sha256") == plan.get("pipeline_sha256")
             and seal.get("source_plan_sha256") == plan_sha
             and plan.get("job_sha256") == canonical_sha256(plan["binding"])
             and plan.get("pipeline_sha256") == canonical_sha256(dict(binding=plan["binding"], model_identity=identity))
             and all(seal.get(key) is False for key in
                     ("mass_eligible", "bottom_mass_allowed", "scientific_acceptance",
                      "automatic_activation", "labels_changed")),
             "L1 neighbourhood saved maps differ from their sealed stored-map revision")
    heads = plan["binding"].get("heads") or {}
    _require((heads.get("co2") or {}).get("sha256") == CO2_SHA256
             and (heads.get("g3") or {}).get("sha256") == G3_SHA256
             and (heads.get("regional") or {}).get("sha256") == REFIT_SHA256,
             "L1 neighbourhood changed one of the frozen base heads")
    inputs = plan["binding"]["inputs"]
    policy_plan = policy.get("source_plan") if policy else {}
    spec_sha=(inputs.get("spec") or {}).get("sha256")
    try:
        settings=approved_spec_settings(spec_sha)
    except (TypeError,ValueError) as error:
        raise ValueError("L1 neighbourhood lost the approved stored-map specification binding") from error
    bound_spec=plan["binding"].get("spec")
    _require(bound_spec is None or (isinstance(bound_spec,dict) and bound_spec.get("sha256")==spec_sha),
             "L1 neighbourhood lost the approved stored-map specification binding")
    _require(inputs.get("cpu_only") is True
             and inputs.get("training") is False
             and inputs.get("model_inference_performed") is False
             and inputs.get("annotations_read") is False
             and inputs.get("stored_map_rule") == dict(activation_h=.935, radius_px=937.5,
                    box_hw=[51, 51], gas_fraction=settings['gas_fraction_threshold'], iterations=1)
             and inputs.get("mass_eligible") is False and inputs.get("bottom_mass_allowed") is False
             and inputs.get("fixed_configuration") == dict(regional_threshold=.6,
                     pixel_threshold=.7, minimum_mm2=2.5, connectivity=4)
             and set(inputs.get("blind_hold_runs") or ()) == set(T3_BLIND_RUNS)
             and set(seal.get("blind_hold_runs") or ()) == set(T3_BLIND_RUNS)
             and policy is not None and policy.get("purpose") == T3_PURPOSE
             and tuple(policy["blind_runs"]) == T3_BLIND_RUNS
             and policy_plan.get("sha256") == plan_sha
             and resolve_review_path(policy_plan.get("path", "")).resolve().samefile(plan_path)
             and policy.get("old_blind_reveal_does_not_release") is True
             and policy.get("candidate_blind_scoring_required") is True,
             "L1 neighbourhood lost its fixed settings or new candidate-specific blind restrictions")
    return dict(registration=registration, pipeline_sha256=plan["pipeline_sha256"], model_identity=identity,
                rule_settings=settings)


def l1_neighbourhood_saved_review(model_root):
    from .tools.register_track1_studio_candidate import registration_path
    from .tools.run_ffac_l1_neighbourhood_candidate import PLAN, SEAL
    from .phase_validation import _model_identity
    root = resolve_review_path(model_root).resolve()
    registration = resolve_review_path(registration_path(root.name)).resolve()
    metadata = _read(registration).get("metadata_sha256") or {}
    paths = sorted({registration, root / PLAN, root / SEAL, root / "rig_bottom_review_policy.json",
                    *(root / name for name in metadata)}, key=str)
    stamps = (tuple(_model_identity(root)),
              tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths))
    return _admitted(str(root), str(registration), stamps)
