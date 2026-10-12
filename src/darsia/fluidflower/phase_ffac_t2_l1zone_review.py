"""Read-only Studio admission for the explicitly sealed combined L1 candidate.

This route does not authorize inference or release blind predictions. Large
immutable plans are parsed once per metadata revision while cataloguing runs.
"""
from functools import lru_cache
import json
from pathlib import Path
import re

from .review_storage import resolve_review_path
from .project import sha256_file


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Combined saved-review metadata must be an object")
    return value


def _samefile(left, right):
    return resolve_review_path(Path(left)).samefile(right)


def _require(value, message):
    if not value:
        raise ValueError(message)


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    from .phase_validation import _model_identity
    from .tools.refit_ffac_g3_downstream import canonical_sha256
    from .tools.register_track1_studio_candidate import registration_warning
    from .tools.run_ffac_t2_l1zone_candidate import CONTRACT, KIND, PLAN, SEAL, HIDDEN, SPEC_SHA256, WARNING
    root, registration_path = Path(root_text), Path(registration_text)
    registration = _read(registration_path)
    _require(registration.get("registration_kind") == KIND and registration_warning(root) == WARNING,
             "Combined saved review requires its intact dedicated registration")
    seal_path, plan_path = root / SEAL, root / PLAN
    seal, plan, policy = _read(seal_path), _read(plan_path), _read(root / "rig_bottom_review_policy.json")
    identity = list(_model_identity(root))
    plan_sha, seal_sha = sha256_file(plan_path), sha256_file(seal_path)
    pipeline = plan.get("pipeline_sha256")
    _require(registration.get("publication_scope") == "studio_validation_only"
             and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
             and all(registration.get(key) is False for key in
                     ("mass_eligible", "bottom_mass_allowed", "scientific_acceptance", "deployment_allowed", "automatic_activation"))
             and registration.get("model_id") == identity[0] and registration.get("model_sha256") == identity[1],
             "Combined saved registration changed its identity or review-only restrictions")
    metadata = registration.get("metadata_sha256") or {}
    _require(metadata.get(PLAN) == plan_sha == registration.get("source_plan_sha256")
             and metadata.get(SEAL) == seal_sha == registration.get("production_receipt_sha256")
             and metadata.get("rig_bottom_review_policy.json") == sha256_file(root / "rig_bottom_review_policy.json"),
             "Combined registration lost its exact plan, seal or blind-policy binding")
    _require(seal.get("contract") == plan.get("contract") == CONTRACT and seal.get("complete") is True
             and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
             and seal.get("model_identity") == plan.get("model_identity") == identity
             and isinstance(pipeline, str) and re.fullmatch(r"[0-9a-f]{64}", pipeline)
             and seal.get("pipeline_sha256") == pipeline and seal.get("source_plan_sha256") == plan_sha
             and _samefile(seal["plan"]["path"], plan_path) and seal["plan"]["sha256"] == plan_sha
             and plan.get("job_sha256") == canonical_sha256(plan["binding"])
             and pipeline == canonical_sha256(dict(binding=plan["binding"], model_identity=identity)),
             "Combined saved maps differ from their sealed composition plan")
    inputs = plan["binding"]["inputs"]
    _require(plan["binding"]["spec"]["sha256"] == SPEC_SHA256
             and all(seal.get(key) is False for key in
                     ("mass_eligible", "bottom_mass_allowed", "scientific_acceptance", "automatic_activation", "training", "labels_read", "labels_changed"))
             and all(inputs.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed", "automatic_activation", "training", "labels_read"))
             and set(seal.get("blind_hold_runs", [])) == set(inputs.get("blind_hold_runs", [])) == HIDDEN
             and set(policy.get("blind_runs", [])) == HIDDEN and policy.get("candidate_id") == identity[0]
             and policy.get("no_mass_until_blind_bottom_check") is True
             and policy.get("old_blind_reveal_does_not_release") is True
             and policy.get("candidate_blind_scoring_required") is True,
             "Combined saved review lost its no-training, mass or candidate-specific blind restrictions")
    return dict(registration=registration, pipeline_sha256=pipeline, model_identity=identity)


def combined_saved_review(model_root):
    """Return the checked registration/pipeline, or raise without changing files."""
    from .tools.register_track1_studio_candidate import registration_path
    from .tools.run_ffac_t2_l1zone_candidate import PLAN, SEAL
    from .phase_validation import _model_identity
    root = resolve_review_path(Path(model_root)).resolve()
    registration = resolve_review_path(registration_path(root.name)).resolve()
    metadata = _read(registration).get("metadata_sha256") or {}
    paths = sorted({registration, root / PLAN, root / SEAL, root / "rig_bottom_review_policy.json",
                    *(root / name for name in metadata)}, key=str)
    stamps = (tuple(_model_identity(root)),
              tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths))
    return _admitted(str(root), str(registration), stamps)
