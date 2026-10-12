"""Explicitly include one completed visible experiment in future continuation.

Selection records are separate from models and editable reviews. They do not
train, change labels, or rewrite an evaluation's original exclusion policy.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
from darsia.fluidflower.phase_workflow import discover_phase_review_sessions
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import phase_transfer_review as transfer

FORMAT = "explicit-phase-training-promotion-v1"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def promotion_root():
    return default_execution_cache_root().resolve() / "phase-training-promotions"


def _command_value(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Completed review command must specify {flag} exactly once")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Missing completed review command value: {flag}")
    return str(command[index + 1])


def _selection(evaluation_root, model_root, reviewer=None, save_root=None):
    root, model = Path(evaluation_root).resolve(), Path(model_root).resolve()
    plan_path, binding_path = root / "transfer_review.json", root / "validation/validation_binding.json"
    plan, result = _read(plan_path), _read(root / "transfer_result.json")
    if result.get("stage") != "ready":
        raise ValueError("The target's model predictions are not complete")
    identity = _model_identity(model)
    if (plan.get("source_model_id"), plan.get("source_model_sha256")) != identity:
        raise ValueError("This experiment was evaluated with another source model")
    if (result.get("model_id"), result.get("source_model_sha256")) != identity:
        raise ValueError("Completed result differs from its source model")
    if Path(plan["evaluation_root"]).resolve() != root or Path(result["evaluation_root"]).resolve() != root:
        raise ValueError("Completed result identifies another evaluation directory")
    if plan.get("review_mode") != "visible" or result.get("review_mode") != "visible":
        raise ValueError("Only a visible model review can be included through this action")
    if plan.get("physics_mode") not in {"prepared-target", "registered-target"}:
        raise ValueError("Further training requires the full model and the experiment's own context")
    source_runtime = physics_runtime_identity(load_physics_profile(model))
    if source_runtime is None or source_runtime != plan.get("source_physics_runtime"):
        raise ValueError("This experiment was evaluated with different source physical rules")
    command = result.get("reopen_command") or result.get("command") or []
    actor = _command_value(command, "--reviewer").strip()
    saved = Path(_command_value(command, "--local-review-root")).resolve()
    session_id = _command_value(command, "--review-session-id")
    if not actor or not session_id:
        raise ValueError("Completed review has no reviewer or session identity")
    if reviewer is not None and str(reviewer).strip().casefold() != actor.casefold():
        raise ValueError(f"This saved review belongs to {actor}; select its reviewer before including it")
    if save_root is not None and Path(save_root).resolve() != saved:
        raise ValueError("Training selection must use the completed review's actual save directory")
    combined = root / "validation"
    if Path(_command_value(command, "--combined-root")).resolve() != combined:
        raise ValueError("Completed review command has a different annotation geometry source")
    if _command_value(command, "--target-model-id") != plan["evaluation_model_id"]:
        raise ValueError("Completed review command has a different evaluated model")
    if Path(_command_value(command, "--transfer-plan")).resolve() != plan_path:
        raise ValueError("Completed review command has a different evaluation plan")
    targets = [row for row in transfer.target_sources(plan["source_manifest"], plan["dataset"])
               if row["run"] == plan["run"] and Path(row["native_manifest"]).resolve()
               == Path(plan["target_native_manifest"]).resolve()]
    if len(targets) != 1:
        raise ValueError("Training selection needs exactly one bound native target revision")
    target = targets[0]
    expected = plan["protected_sha256"].get(str(Path(target["native_manifest"]).resolve()))
    if not expected or target["native_manifest_sha256"] != expected:
        raise ValueError("The experiment's native image revision changed after evaluation")
    # The frozen plan supplies the provenance path. A caller may reach the same
    # immutable model via a mapped drive or UNC alias; its verified identity,
    # not that caller's spelling of the path, determines the selected model.
    source_model_root = Path(plan.get("source_model") or model).resolve()
    record = dict(format=FORMAT, source_model_id=identity[0], source_model_sha256=identity[1],
        source_model_root=str(source_model_root), evaluation_root=str(root),
        evaluation_model_id=plan["evaluation_model_id"],
        evaluation_model_sha256=plan["evaluation_model_sha256"],
        dataset=plan["dataset"], run=plan["run"], physics_mode=plan["physics_mode"],
        target_native_manifest=str(Path(target["native_manifest"]).resolve()),
        target_native_manifest_sha256=expected,
        source_manifest=str(Path(plan["source_manifest"]).resolve()),
        source_manifest_sha256=sha256_file(Path(plan["source_manifest"])),
        plan_path=str(plan_path), plan_sha256=sha256_file(plan_path),
        binding_path=str(binding_path), binding_sha256=sha256_file(binding_path),
        combined_root=str(combined), reviewer=actor, save_root=str(saved), review_session_id=session_id)
    return record, plan, target


def _key(record):
    # The path is stable across idempotent button presses. All scientific and
    # review identities participate; timestamps and comments do not.
    body = {key: value for key, value in record.items()
            if key not in {"created_utc", "receipt_path", "receipt_sha256", "selection", "training_performed"}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _path(record):
    return promotion_root() / (_key(record) + ".json")


def _validate_science(record, plan, target, parent):
    transfer.verify_hashes(plan["protected_sha256"])
    transfer.validate_review_binding(plan, _read(record["binding_path"]))
    from darsia.fluidflower.tools.phase_transfer_context import prepared_target_physics
    # This also checks categorical facies, domain, native grid and own baseline.
    if target.get("job", {}).get("phase_transfer_context"):
        source_digest = plan["protected_sha256"].get(str(target["source_manifest"]))
        prepared_target_physics(Path(parent).resolve(), target,
                                expected_source_manifest_sha256=source_digest)
    else:
        transfer.registered_target_physics(Path(parent).resolve(), target)


def registered_promotion(evaluation_root, model_root, reviewer=None, save_root=None):
    """Read a matching selection; full image validation happens before capture."""
    record, _, _ = _selection(evaluation_root, model_root, reviewer, save_root)
    path = _path(record)
    if not path.is_file():
        return None
    stored = _read(path)
    if any(stored.get(key) != value for key, value in record.items()) or _key(stored) != path.stem:
        raise ValueError("Saved training selection changed")
    return dict(stored, receipt_path=str(path), receipt_sha256=sha256_file(path))


def register_promotion(evaluation_root, model_root, reviewer, save_root):
    """Select one editable visible session for future training, without fitting."""
    record, plan, target = _selection(evaluation_root, model_root, reviewer, save_root)
    _validate_science(record, plan, target, model_root)
    path = _path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(record, created_utc=datetime.now(timezone.utc).isoformat(),
        selection="Include this session's saved labels in future full-model continuation",
        training_performed=False)
    from darsia.fluidflower.tools.calibration_pairwise_review import _exclusive_path_lock
    with _exclusive_path_lock(path.with_suffix(".lock"), timeout_s=120, stale_after_s=600):
        if not path.exists():
            # Atomic, Windows sharing-safe publication. Never replace a previous
            # selection or mutate the underlying evaluation or review session.
            from darsia.fluidflower.tools.render_concentration_movie import write
            write(path, stored)
        elif _key(_read(path)) != _key(record):
            raise ValueError("Conflicting training selection already exists")
    return registered_promotion(evaluation_root, model_root, reviewer, save_root)


def validated_promotions(parent, selected, lineage):
    """Resolve explicitly included, editable reviews for detached input capture.

    Empty enrollments wait for a review to be saved. Unrelated experiments and
    actors remain excluded. A matching stale selection fails instead of quietly
    dropping the user's intended evidence.
    """
    output, seen_sessions = [], set()
    for path in sorted(promotion_root().glob("*.json")):
        stored = _read(path)
        if stored.get("dataset") != selected or stored.get("source_model_id") not in lineage:
            continue
        ancestor = lineage[stored["source_model_id"]]
        if stored.get("format") != FORMAT or _key(stored) != path.stem:
            raise ValueError(f"Training selection has changed: {path}")
        if stored["source_model_sha256"] != ancestor["model_sha256"]:
            raise ValueError("Included review was made with different ancestor weights")
        record, plan, target = _selection(stored["evaluation_root"], ancestor["root"],
                                         stored["reviewer"], stored["save_root"])
        if any(stored.get(key) != value for key, value in record.items()):
            raise ValueError(f"Included experiment changed after selection: {path}")
        _validate_science(record, plan, target, parent)
        sessions = discover_phase_review_sessions([Path(stored["save_root"])]) + discover_phase_review_sessions()
        for session in sessions:
            if session.get("review_session_id") != stored["review_session_id"]:
                continue
            if (session.get("target_model_id") != stored["evaluation_model_id"]
                    or str(session.get("reviewer") or "").casefold() != stored["reviewer"].casefold()
                    or Path(session.get("combined_root") or "").resolve() != Path(stored["combined_root"])
                    or session.get("review_surface") != "open_correction"):
                raise ValueError("Included review session changed identity or source geometry")
            decisions = resolve_review_path(session["decision_path"], context=session.get("session_path"))
            selected_store = resolve_review_path(stored["save_root"], context=session.get("session_path"))
            if not decisions.resolve().is_relative_to(selected_store):
                raise ValueError("Included decisions are outside the selected review store")
            if not decisions.is_file():
                continue
            identity = str(decisions.resolve()).casefold()
            if identity in seen_sessions:
                continue
            seen_sessions.add(identity)
            output.append(dict(record, receipt_path=str(path), receipt_sha256=sha256_file(path),
                target=target, session=session, decision_path=str(decisions),
                combined_root=stored["combined_root"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("register",))
    for flag in ("evaluation-root", "model", "save-root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    result = register_promotion(args.evaluation_root, args.model, args.reviewer, args.save_root)
    print("PHASE_PROMOTION_EVENT " + json.dumps(dict(stage="registered", **result)), flush=True)


if __name__ == "__main__":
    main()
