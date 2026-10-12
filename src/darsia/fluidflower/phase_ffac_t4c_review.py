"""Exact, read-only Studio admission of calibrated T4c saved placements."""
from functools import lru_cache
import json
import math
from pathlib import Path

from .project import sha256_file
from .review_storage import resolve_review_path


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("T4c review metadata must be an object")
    return value


def _require(value, message):
    if not value:
        raise ValueError(message)


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _assets(value):
    """Only compact path/SHA pairs are retained, never scientific frame data."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _require(isinstance(value["path"], str) and isinstance(value["sha256"], str)
                     and len(value["sha256"]) == 64, "T4c has an invalid bound asset")
            yield value["path"], value["sha256"]
        else:
            for child in value.values():
                yield from _assets(child)
    elif isinstance(value, list):
        for child in value:
            yield from _assets(child)


@lru_cache(maxsize=16)
def _plan_dependencies(root_text, plan_signature, expected_plan_sha):
    from .tools.run_ffac_t4c_floor_candidate import PLAN
    from .phase_ffac_rig_camera_runtime import bound_path
    binding = _read(Path(root_text) / PLAN)["binding"]
    dependencies = list(_assets(dict(heads=binding["heads"], inputs=binding["inputs"])))
    shift = _read(bound_path(binding["inputs"]["calibrated_shift"], base=Path(root_text)))
    dependencies.extend(_assets(shift))
    return tuple(sorted(set(dependencies)))


@lru_cache(maxsize=256)
def _checked_asset(path_text, expected_sha, signature):
    _require(sha256_file(Path(path_text)) == expected_sha, "T4c bound source artifact changed")
    return path_text


def review_state_label(placement):
    descriptions = {"P1": "Optical", "P2": "Neighbourhood-weighted, caps only", "P3": "T4b boundary"}
    _require(placement in descriptions, "T4c saved review has an unknown placement")
    return f"G3-T4c {placement} gas · New lamps calibrated to overflow · {descriptions[placement]} · Floor contact · Studio review only"


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    from .phase_validation import _model_identity
    from .phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
    from .phase_ffac_rig_camera_runtime import bound_path
    from .tools.refit_ffac_g3_downstream import canonical_sha256
    from .tools.register_track1_studio_candidate import registration_warning
    from .tools import run_ffac_t4c_floor_candidate as runner
    root, registration_path = Path(root_text), Path(registration_text)
    registration, plan, seal = _read(registration_path), _read(root / runner.PLAN), _read(root / runner.SEAL)
    identity = list(_model_identity(root))
    placement, delta = plan.get("placement"), plan.get("delta")
    _require(placement in ("P1", "P2", "P3") and type(delta) in (int, float) and math.isfinite(delta),
             "T4c saved review lost its calibrated placement or finite shift")
    expected_warning = (runner.warning_for_placement(placement) if hasattr(runner, "warning_for_placement") else runner.WARNING)
    plan_sha, seal_sha = sha256_file(root / runner.PLAN), sha256_file(root / runner.SEAL)
    metadata = registration.get("metadata_sha256") or {}
    _require(registration.get("registration_kind") == runner.KIND
        and registration_warning(root) == expected_warning
        and registration.get("publication_scope") == "studio_validation_only"
        and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
        and registration.get("model_id") == identity[0] and registration.get("model_sha256") == identity[1]
        and all(registration.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "deployment_allowed", "automatic_activation")),
        "T4c saved registration changed its identity or review-only restrictions")
    _require(metadata.get(runner.PLAN) == plan_sha == registration.get("source_plan_sha256")
        and metadata.get(runner.SEAL) == seal_sha == registration.get("production_receipt_sha256"),
        "T4c registration lost its exact plan or seal binding")
    _require(all(sha256_file(root / relative) == digest for relative, digest in metadata.items()),
             "T4c registered candidate metadata changed")
    _require(plan.get("contract") == seal.get("contract") == runner.CONTRACT
        and seal.get("complete") is True and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
        and seal.get("model_identity") == plan.get("model_identity") == identity
        and seal.get("pipeline_sha256") == plan.get("pipeline_sha256")
        and seal.get("source_plan_sha256") == plan_sha
        and plan.get("job_sha256") == canonical_sha256(plan["binding"])
        and plan.get("pipeline_sha256") == canonical_sha256(dict(binding=plan["binding"], model_identity=identity))
        and all(seal.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "automatic_activation", "labels_changed")),
        "T4c saved maps differ from their exact sealed placement")
    inputs = plan["binding"]["inputs"]
    _require((plan["binding"]["heads"].get("co2") or {}).get("sha256") == CO2_SHA256,
             "T4c changed the frozen CO2-presence head")
    _require(plan["binding"]["heads"]["g3"]["sha256"] == G3_SHA256
        and plan["binding"]["heads"]["regional"]["sha256"] == REFIT_SHA256,
        "T4c changed its exact original G3 pixel or regional head")
    _require((inputs.get("spec") or {}).get("sha256") == runner.SPEC_SHA256
        and inputs.get("cpu_only") is True and inputs.get("mass_eligible") is False
        and inputs.get("bottom_mass_allowed") is False and inputs.get("fixed_configuration") == runner.FIXED_CONFIGURATION
        and inputs.get("blind_hold_runs") == seal.get("blind_hold_runs") == []
        and inputs.get("l1_rules_applied") == [] and not (root / "rig_bottom_review_policy.json").exists(),
        "T4c lost its fixed settings, visibility or bottom-mass restrictions")
    _require(inputs.get("placement") == seal.get("placement") == placement
        and inputs.get("delta") == seal.get("delta") == delta
        and isinstance(inputs.get("calibrated_shift"), dict)
        and inputs["calibrated_shift"].get("path") and inputs["calibrated_shift"].get("sha256")
        and seal.get("calibrated_shift") == inputs["calibrated_shift"],
        "T4c lost its exact calibration, placement or shift binding")
    shift = _read(bound_path(inputs["calibrated_shift"], base=root))
    raw = _read(bound_path(shift["script_output"], base=root))
    _require(shift.get("placement") == placement and shift.get("delta") == delta
        and type(shift.get("delta_star_unrounded")) in (float, int)
        and round(float(shift["delta_star_unrounded"]), 2) == delta
        and raw.get("script_sha256") == runner.CALIBRATOR_SHA256
        and raw.get("counts_sha256") == shift["counts"]["sha256"]
        and raw["variants"][placement]["delta_star"] == shift["delta_star_unrounded"],
        "T4c calibrated shift differs from the frozen calibrator output")
    _require(inputs.get("code") == runner.code_binding(), "T4c current numerical pipeline differs from its frozen code binding")
    runner.validate_source_inventory(plan["source_inventory"])
    _require(inputs.get("source_inventory_sha256") == canonical_sha256(plan["source_inventory"]),
             "T4c source inventory changed")
    profile = _read(root / "phase_rig_camera.json")
    other_markers = {"g3_t4b_floor_revision", "g3_t4_floor_revision", "g3_t3_floor_revision",
                     "g3_t2_floor_revision", "g3_t2_l1zone_revision", "l1_neighbourhood_floor_revision"}
    _require("g3_t4c_floor_revision" in profile and not other_markers.intersection(profile),
             "T4c profile mixed another gas revision")
    runner.validate_runtime_heads(root, profile["g3_t4c_floor_revision"], profile["frozen_heads"], verified_plan=plan)
    _require(profile["frozen_heads"] == seal.get("heads"), "T4c seal lost its exact frozen-head bindings")
    return dict(registration=registration, pipeline_sha256=plan["pipeline_sha256"], model_identity=identity,
        light_epochs=plan["run_to_epoch"], blind_hold_runs=[], placement=placement, delta=delta,
        calibrated_shift=inputs["calibrated_shift"], state_label=review_state_label(placement))


def t4c_saved_review(model_root):
    from .tools.register_track1_studio_candidate import registration_path
    from .tools.run_ffac_t4c_floor_candidate import PLAN, SEAL
    from .phase_validation import _model_identity
    from .phase_ffac_rig_camera_runtime import bound_path
    root = resolve_review_path(model_root).resolve()
    registration = resolve_review_path(registration_path(root.name)).resolve()
    current_registration = _read(registration)
    metadata = current_registration.get("metadata_sha256") or {}
    expected_plan_sha = metadata.get(PLAN)
    _require(isinstance(expected_plan_sha, str) and expected_plan_sha == current_registration.get("source_plan_sha256"),
             "T4c registration lost its exact plan binding")
    assets = _plan_dependencies(str(root), _signature(root / PLAN), expected_plan_sha)
    bound_assets = []
    for path, sha in assets:
        resolved = bound_path(dict(path=path, sha256=sha), base=root, verify=False)
        _checked_asset(str(resolved), sha, _signature(resolved))
        bound_assets.append(resolved)
    paths = sorted({registration, root / PLAN, root / SEAL, *(root / name for name in metadata), *bound_assets}, key=str)
    bottom_policy = root / "rig_bottom_review_policy.json"
    stamps = (tuple(_model_identity(root)), expected_plan_sha,
              tuple((str(path), _signature(path)) for path in paths),
              _signature(bottom_policy) if bottom_policy.exists() else None)
    return _admitted(str(root), str(registration), stamps)


__all__ = ["t4c_saved_review", "review_state_label"]
