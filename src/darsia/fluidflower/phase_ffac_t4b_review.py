"""Strict read-only Studio admission for the sealed joint lamp-shift T4b candidate."""
from functools import lru_cache
import json
from pathlib import Path

from .project import sha256_file
from .review_storage import resolve_review_path


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("T4b review metadata must be an object")
    return value


def _require(value, message):
    if not value:
        raise ValueError(message)


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


@lru_cache(maxsize=16)
def _plan_dependencies(root_text, plan_signature, expected_plan_sha):
    """Retain only small asset bindings, keyed by the current registered plan."""
    from .tools.run_ffac_t4b_floor_candidate import PLAN
    plan = _read(Path(root_text) / PLAN)
    heads = plan["binding"]["heads"]
    assets = [heads[key] for key in ("co2", "g3", "regional")]
    assets += list(heads["joint"].values())
    assets += list(plan["binding"]["inputs"]["joint"].values())
    assets.append(plan["binding"]["inputs"]["regional_bundle"])
    assets.append(plan["binding"]["inputs"]["light_epoch_table"])
    return tuple((asset["path"], asset["sha256"]) for asset in assets)


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    from .phase_validation import _model_identity
    from .phase_g3_production import CO2_SHA256
    from .tools.refit_ffac_g3_downstream import canonical_sha256
    from .tools.register_track1_studio_candidate import registration_warning
    from .tools.run_ffac_t4b_floor_candidate import (
        CONTRACT, KIND, PLAN, SEAL, WARNING, SPEC_SHA256, FIXED_CONFIGURATION,
        validate_runtime_heads, validate_source_inventory,
    )
    root, registration_path = Path(root_text), Path(registration_text)
    registration, plan, seal = _read(registration_path), _read(root / PLAN), _read(root / SEAL)
    identity = list(_model_identity(root))
    plan_sha, seal_sha = sha256_file(root / PLAN), sha256_file(root / SEAL)
    metadata = registration.get("metadata_sha256") or {}
    _require(registration.get("registration_kind") == KIND
        and registration_warning(root) == WARNING
        and registration.get("publication_scope") == "studio_validation_only"
        and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
        and registration.get("model_id") == identity[0] and registration.get("model_sha256") == identity[1]
        and all(registration.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "deployment_allowed", "automatic_activation")),
        "T4b saved registration changed its identity or review-only restrictions")
    _require(metadata.get(PLAN) == plan_sha == registration.get("source_plan_sha256")
        and metadata.get(SEAL) == seal_sha == registration.get("production_receipt_sha256"),
        "T4b registration lost its exact plan or seal binding")
    _require(plan.get("contract") == seal.get("contract") == CONTRACT
        and seal.get("complete") is True and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
        and seal.get("model_identity") == plan.get("model_identity") == identity
        and seal.get("pipeline_sha256") == plan.get("pipeline_sha256")
        and seal.get("source_plan_sha256") == plan_sha
        and plan.get("job_sha256") == canonical_sha256(plan["binding"])
        and plan.get("pipeline_sha256") == canonical_sha256(dict(binding=plan["binding"], model_identity=identity))
        and all(seal.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "automatic_activation", "labels_changed")),
        "T4b saved maps differ from their sealed training revision")
    inputs = plan["binding"]["inputs"]
    _require((plan["binding"]["heads"].get("co2") or {}).get("sha256") == CO2_SHA256,
        "T4b changed the frozen CO2-presence head")
    _require((inputs.get("spec") or {}).get("sha256") == SPEC_SHA256
        and inputs.get("cpu_only") is True
        and inputs.get("mass_eligible") is False and inputs.get("bottom_mass_allowed") is False
        and inputs.get("fixed_configuration") == FIXED_CONFIGURATION
        and inputs.get("blind_hold_runs") == seal.get("blind_hold_runs") == []
        and inputs.get("l1_rules_applied") == []
        and not (root / "rig_bottom_review_policy.json").exists(),
        "T4b lost its fixed settings, visibility or bottom-mass restrictions")
    validate_source_inventory(plan["source_inventory"])
    _require(inputs.get("source_inventory_sha256") == canonical_sha256(plan["source_inventory"]),
        "T4b source inventory changed")
    profile = _read(root / "phase_rig_camera.json")
    validate_runtime_heads(root, profile["g3_t4b_floor_revision"], profile["frozen_heads"], verified_plan=plan)
    _require(profile["frozen_heads"] == seal.get("heads"), "T4b seal lost its joint-head bindings")
    return dict(registration=registration, pipeline_sha256=plan["pipeline_sha256"], model_identity=identity,
        light_epochs=plan["run_to_epoch"], blind_hold_runs=[])


def t4b_saved_review(model_root):
    from .tools.register_track1_studio_candidate import registration_path
    from .tools.run_ffac_t4b_floor_candidate import PLAN, SEAL
    from .phase_validation import _model_identity
    from .phase_ffac_rig_camera_runtime import bound_path
    root = resolve_review_path(model_root).resolve()
    registration = resolve_review_path(registration_path(root.name)).resolve()
    current_registration = _read(registration)
    metadata = current_registration.get("metadata_sha256") or {}
    expected_plan_sha = metadata.get(PLAN)
    _require(isinstance(expected_plan_sha, str)
        and expected_plan_sha == current_registration.get("source_plan_sha256"),
        "T4b registration lost its exact plan binding")
    assets = _plan_dependencies(str(root), _signature(root / PLAN), expected_plan_sha)
    paths = sorted({registration, root / PLAN, root / SEAL, *(root / name for name in metadata),
                    *(bound_path(dict(path=path, sha256=sha), base=root) for path, sha in assets)}, key=str)
    bottom_policy = root / "rig_bottom_review_policy.json"
    stamps = (tuple(_model_identity(root)), expected_plan_sha,
              tuple((str(path), _signature(path)) for path in paths),
              _signature(bottom_policy) if bottom_policy.exists() else None)
    return _admitted(str(root), str(registration), stamps)
