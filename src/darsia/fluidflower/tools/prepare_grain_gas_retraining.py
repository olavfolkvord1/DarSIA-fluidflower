"""Freeze series-isolated reviewed grain exclusions for a gas-only fit.

Live reviews remain editable. Detached, verified input copies prevent the
standard request freezer from locking their source sessions. This script never
fits models, copies inherited weights, or publishes validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_training.scope import requested_series, series_id, validate_training_series
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.phase_workflow import (
    PHASE_REVIEW_SESSION_FORMAT, build_phase_training_command,
    prepare_phase_training_request,
)
from darsia.fluidflower.project import sha256_file


REVIEWER = "Olav Folkvord"
SUPPORTED_SERIES = {"ffac", "ffbm", "ffum"}
PROFILE_FILES = ("phase_optical_regions.json", "phase_native_reference.json", "phase_facies_reference.json",
                 "phase_reviewed_reference.json", "phase_reference_support.json",
                 "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")


def parent_series(settings, selected=None):
    """Never silently partition or reuse another series' optical evidence."""
    actual = requested_series(settings)
    if actual not in SUPPORTED_SERIES:
        raise ValueError("An explicitly single-series FF_AC, FF_BM or FF_UM parent is required")
    if selected is not None and series_id(selected) != actual:
        raise ValueError(f"Parent series {actual} differs from requested series {selected}")
    return actual


def physics_files(parent, profile):
    """Preserve the selected parent's profile and provenance without upgrading it."""
    parent = Path(parent).resolve()
    profiles = [parent / name for name in PROFILE_FILES if (parent / name).is_file()]
    if len(profiles) != 1:
        raise ValueError("A single explicit parent physics profile is required")
    files = set(profiles)
    # Regional profiles bind a separately fitted model by its content hash.
    # Optical continuation keeps this profile only as frozen provenance until
    # the regional layer is refitted; omitting its model makes even a detached
    # intermediate impossible to verify, and must never silently drop it.
    regional = profile.get("model") if profile.get("format") == "baseline-relative-optical-regions-v1" else None
    if regional is not None:
        relative = Path(regional["path"])
        artifact = (parent / relative).resolve()
        if relative.is_absolute() or not artifact.is_relative_to(parent) or not artifact.is_file():
            raise ValueError("Regional model escaped or is missing from its parent bundle")
        if sha256_file(artifact) != regional["sha256"]:
            raise ValueError("Regional model hash mismatch while copying")
        files.add(artifact)
    from darsia.fluidflower import phase_decision_runtime as decision
    if decision.PROFILE_KEY in profile:
        files.add(parent / decision.PROFILE_FILE)
    from darsia.fluidflower import phase_aqueous_context_completion as completion
    if completion.PROFILE_KEY in profile:
        # The optional learned stage is part of the frozen physical runtime;
        # copying only the inherited context would silently lose its weights.
        _, specification = completion.split_profile(profile)
        checkpoint = (parent / specification["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(parent) or not checkpoint.is_file():
            raise ValueError("Aqueous completion checkpoint escaped its bundle")
        if sha256_file(checkpoint) != specification["checkpoint_sha256"]:
            raise ValueError("Aqueous completion checkpoint hash mismatch while copying")
        files.update((parent / completion.PROFILE_FILE, checkpoint))
    for directory in parent.iterdir():
        if directory.is_dir() and (directory.name == "plume_context"
                or directory.name.endswith(("_calibration", "_code_snapshot"))):
            files.update(path for path in directory.rglob("*") if path.is_file())
    for entry in profile["runs"].values():
        relative = Path(entry["path"])
        context = (parent / relative).resolve()
        if relative.is_absolute() or not context.is_relative_to(parent):
            raise ValueError("Parent context escaped its bundle")
        files.add(context)
    return sorted(files)


def _read(path, protected):
    path = Path(path).resolve()
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if str(path) in protected and protected[str(path)] != digest:
        raise ValueError(f"Protected input changed: {path}")
    protected[str(path)] = digest
    return json.loads(contents.decode("utf-8-sig"))


def _check_hashes(records):
    for name, expected in records.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"Protected input changed: {name}")


def _copy(source, destination, expected):
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise OSError(f"Copied input differs: {destination}")


def gas_only_command(request):
    command = build_phase_training_command(request)
    if request['settings'].get('evaluation_mode') == 'fit-only-single-experiment':
        return [*command, '--heads', 'gas_within_co2']
    index = command.index("--heads") + 1
    end = index
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    if command[index:end] != ["co2_presence", "gas_within_co2"]:
        raise ValueError("Unexpected standard phase-head command")
    return command[:index] + ["gas_within_co2"] + command[end:]


def inspect(parent, reviews=(), selected=None):
    """Check the complete input contract without creating or modifying files."""
    parent = Path(parent).resolve()
    protected = {}
    previous = _read(parent / "training_request.json", protected)
    summary = _read(parent / "summary.json", protected)
    settings = previous["settings"]
    selected = parent_series(settings, selected)
    if settings.get("evaluation_mode") not in {"whole-run", "fit-only-single-experiment"}:
        raise ValueError("A supported explicit parent evaluation mode is required")
    profile = phase_runtime.load_physics_profile(parent)
    if profile is None:
        raise ValueError("An explicit parent physics profile is required")
    copy_paths = physics_files(parent, profile)
    for path in copy_paths:
        protected[str(path.resolve())] = sha256_file(path)
    binding = _read(parent / "validation/validation_binding.json", protected)
    manifest = _read(parent / "validation/postflight_manifest.json", protected)
    model_id, model_digest = _model_identity(parent)
    runtime = phase_runtime.physics_runtime_identity(profile)
    if model_id != previous["model_id"] or model_id != summary["model_id"]:
        raise ValueError("Parent model identities differ")
    for record in (binding, manifest):
        if record.get("model_id") != model_id or record.get("model_sha256") != model_digest:
            raise ValueError("Parent validation is not bound to its current optical weights")
    if manifest.get("physics_runtime") != runtime:
        raise ValueError("Parent physics runtime changed")
    if binding.get("physics_runtime", runtime) != runtime:
        raise ValueError("Parent validation physics binding changed")

    bundles = []
    for bundle in previous["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            path = Path(record["snapshot_path"]).resolve()
            _read(path, protected)
            if protected[str(path)] != record["sha256"]:
                raise ValueError(f"Parent frozen evidence changed: {path}")
            pair.append(path)
        bundles.append(tuple(pair))
    parent_sources, parent_regions, _ = t._load_sources_and_regions([t.ReviewBundle(*p) for p in bundles])
    validate_training_series(parent_sources, parent_regions, selected)

    new_pairs, review_records = [], []
    for value in reviews:
        decision_path = Path(value).resolve()
        session_path = decision_path.parent / "review_session.json"
        session = _read(session_path, protected)
        decisions = _read(decision_path, protected)
        if session.get("format") != PHASE_REVIEW_SESSION_FORMAT or session.get("editing_locked"):
            raise ValueError(f"An unlocked phase review session is required: {session_path}")
        if session.get("reviewer") != REVIEWER or session.get("target_model_id") != model_id:
            raise ValueError(f"Review identity differs from the selected reviewer/model: {session_path}")
        source_manifest = Path(session["combined_root"]) / "postflight_manifest.json"
        source_manifest = source_manifest.resolve()
        reviewed = _read(source_manifest, protected)
        if (reviewed.get("model_id"), reviewed.get("model_sha256"), reviewed.get("physics_runtime")) != (model_id, model_digest, runtime):
            raise ValueError("Live review is not bound to the exact parent optical/physics model")
        if {series_id(j.get("dataset", "")) for j in reviewed["jobs"]} != {selected}:
            raise ValueError("Review manifest includes another experimental series")
        jobs = {(j["run"], j["candidate_id"]): j for j in reviewed["jobs"]}
        for run, decision in decisions.get("decisions", {}).items():
            if decision.get("reviewer") != REVIEWER or decision.get("phase_quiz_answers"):
                raise ValueError("Expected manual grain exclusions from the submitting reviewer")
            for item in decision.get("annotations", []):
                candidate = session.get("selected_candidates", {}).get(run)
                job = jobs.get((run, item.get("candidate_id")))
                if (job is None or series_id(job.get("dataset", "")) != selected or item.get("candidate_id") != candidate
                        or item.get("params_sha256") != job.get("params_sha256")):
                    raise ValueError("Grain annotation has a different displayed candidate/digest")
                frame = item.get("frame_index")
                if (item.get("reviewer") != REVIEWER or type(frame) is not int or frame < 0
                        or item.get("category") != "unsupported_free_gas_classification"
                        or item.get("training_phase_class") != "gas_absent_region"):
                    raise ValueError("Unexpected reviewer, frame or grain exclusion semantics")
        pair = (decision_path, source_manifest)
        new_pairs.append(pair)
        review_records.append({"decisions": str(decision_path), "session": str(session_path),
                               "combined_manifest": str(source_manifest)})
    if new_pairs:
        new_sources, new_regions, _ = t._load_sources_and_regions([t.ReviewBundle(*p) for p in new_pairs])
        validate_training_series(new_sources, new_regions, selected)
    else:
        new_regions = []
    new_ids = {r.region_id for r in new_regions}
    parent_ids = {r.region_id for r in parent_regions}
    if len(new_regions) != len(new_ids):
        raise ValueError("New grain exclusion identifiers must be unique")
    if new_ids & parent_ids or any(r.phase_class != "gas_absent_region" for r in new_regions):
        raise ValueError("New exclusions repeat parent identifiers or contain other classes")
    bundles.extend(new_pairs)
    sources, regions, conflicts = t._load_sources_and_regions([t.ReviewBundle(*p) for p in bundles])
    validate_training_series(sources, regions, selected)
    if {r.region_id for r in regions} != parent_ids | new_ids:
        raise ValueError("Combining evidence dropped or superseded reviewed regions")
    for source in sources:
        _read(source.asset_manifest_path, protected)

    jobs = _source_jobs(parent / "validation")
    if {series_id(j["dataset"]) for j in jobs} != {selected}:
        raise ValueError("Parent validation includes another experimental series")
    if {f"{j['dataset']}/{j['run']}" for j in jobs} != set(profile["runs"]):
        raise ValueError("Parent review jobs and physical contexts differ")
    for job in jobs:
        entry = profile["runs"][f"{job['dataset']}/{job['run']}"]
        context = (parent / entry["path"]).resolve()
        if not context.is_relative_to(parent) or sha256_file(context) != entry["sha256"]:
            raise ValueError("Parent context escaped its bundle or changed")
        review_root = analysis_review_root(Path(job.get("analysis_source_review_root") or job["review_output_dir"]), job["run"])
        timeline = (review_root / "native_frames" / job["run"] / "manifest.json").resolve()
        _read(timeline, protected)
        if protected[str(timeline)] != entry["native_manifest_sha256"]:
            raise ValueError(f"Registered source timeline changed: {job['run']}")
    for directory in ("models",):
        if not (parent / directory).is_dir():
            raise FileNotFoundError(parent / directory)
        for path in sorted((parent / directory).rglob("*")):
            if path.is_file():
                protected[str(path.resolve())] = sha256_file(path)
    _check_hashes(protected)
    return {"parent": parent, "previous": previous, "profile": profile, "runtime": runtime,
            "series": selected, "physics_files": copy_paths, "ready_to_train": bool(new_ids),
            "bundles": bundles, "regions": regions, "conflicts": conflicts, "protected": protected,
            "new_region_ids": sorted(new_ids), "parent_regions": len(parent_regions),
            "reviews": review_records, "model_sha256": model_digest}


def prepare(parent, reviews, output, cache, iteration="V13-G1", selected=None):
    checked = inspect(parent, reviews, selected)
    if not checked["ready_to_train"]:
        raise ValueError("No new series-specific grain exclusions; use --preflight-only to inspect readiness")
    parent, protected = checked["parent"], checked["protected"]
    selected = checked["series"]
    output, cache = Path(output).resolve(), Path(cache).resolve()
    if output == cache or output.is_relative_to(parent) or cache.is_relative_to(parent):
        raise ValueError("Use separate output/cache locations outside the parent model")
    if (output / "iteration.json").exists():
        raise FileExistsError(output / "iteration.json")
    capture = output / "preparation_inputs" / uuid.uuid4().hex
    capture.mkdir(parents=True, exist_ok=False)
    detached, captured = [], []
    for index, pair in enumerate(checked["bundles"], 1):
        copied = []
        for role, source in zip(("decisions", "combined_manifest"), pair):
            destination = capture / f"{index:02d}_{role}.json"
            _copy(source, destination, protected[str(source)])
            copied.append(destination)
            captured.append({"source": str(source), "detached": str(destination),
                             "sha256": protected[str(source)]})
        detached.append(tuple(copied))
    settings = dict(checked["previous"]["settings"], parent_model=str(parent), iteration=iteration,
                    fit_heads=["gas_within_co2"], inherited_heads=["co2_presence"],
                    model_initialization="fresh gas-head fits; parent CO2 heads installed separately after fitting",
                    manual_approval_required=True,
                    phase_support_extension=checked["profile"]["format"])
    request = prepare_phase_training_request(output_parent=output / selected, cache_parent=cache / selected,
                                            bundles=detached, actor=REVIEWER, settings=settings)
    root = Path(request["output_root"])
    if request.get("locked_review_sessions"):
        raise RuntimeError("Detached preparation unexpectedly locked a review session")
    _, frozen, frozen_conflicts = t._load_sources_and_regions([t.ReviewBundle(*p) for p in request["frozen_pairs"]])
    if [asdict(r) for r in frozen] != [asdict(r) for r in checked["regions"]] or frozen_conflicts != checked["conflicts"]:
        raise ValueError("Frozen inputs differ from the verified cumulative evidence")
    copies = []
    for source in checked["physics_files"]:
        destination = root / source.relative_to(parent)
        digest = protected[str(source.resolve())]
        _copy(source, destination, digest)
        copies.append({"source": str(source), "target": str(destination), "sha256": digest})
    source_manifest = parent / "validation/postflight_manifest.json"
    destination = root / "review_source/postflight_manifest.json"
    _copy(source_manifest, destination, protected[str(source_manifest)])
    copies.append({"source": str(source_manifest), "target": str(destination),
                   "sha256": protected[str(source_manifest)]})
    if phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(root)) != checked["runtime"]:
        raise ValueError("Copied parent physics/calibration runtime differs")
    inventory = {"series_scope": selected, "parent_model": str(parent),
                 "parent_model_sha256": checked["model_sha256"], "parent_regions": checked["parent_regions"],
                 "new_session_regions": len(checked["new_region_ids"]), "new_region_ids": checked["new_region_ids"],
                 "cumulative_regions": len(frozen), "fit_heads": ["gas_within_co2"],
                 "inherited_heads": ["co2_presence"], "inherited_heads_pending_copy": True,
                 "regions_by_series_run": dict(Counter(f"{r.dataset}/{r.run}" for r in frozen)),
                 "phase_classes": dict(Counter(r.phase_class for r in frozen)),
                 "deduplication_conflicts": frozen_conflicts, "parent_and_review_protected_sha256": protected,
                 "review_sources": checked["reviews"], "detached_inputs": captured,
                 "physics_runtime": checked["runtime"], "physics_copies": copies,
                 "physics_and_calibration_unchanged": True, "live_reviews_modified": False,
                 "historical_calibration_receipts_are_parent_provenance": True,
                 "optical_and_gas_predictions_unchanged": False, "generalization_validated": False}
    t._write_json(root / "evidence_preflight.json", inventory)
    t._write_json(root / "physics_inheritance.json", {"parent_model": str(parent),
                  "physics_runtime": checked["runtime"], "copies": copies,
                  "settings_unchanged": True, "context_entries_unchanged": True,
                  "calibration_unchanged": True})
    _copy(Path(__file__).resolve(), root / "code_snapshot" / Path(__file__).name, sha256_file(Path(__file__)))
    result = {"model_id": request["model_id"], "root": str(root), "cache": request["cache_root"],
              "parent": str(parent), "source": str(root / "review_source"),
              "command": gas_only_command(request), "preflight": inventory}
    _check_hashes(protected)
    if (root / "summary.json").exists() or (root / "models").exists():
        raise ValueError("Preparation must not install model summaries or inherited weights")
    t._write_json(root / "series_iteration.json", result)
    if (output / "iteration.json").exists():
        raise FileExistsError(output / "iteration.json")
    t._write_json(output / "iteration.json", {selected: result})
    print(f"PREPARED {iteration} {selected}: {len(frozen)} regions; {len(checked['new_region_ids'])} new grain exclusions; {root}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    for name in ("output", "cache"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--review", type=Path, action="append", default=[])
    parser.add_argument("--series", help="Optional safety check against the parent's declared series")
    parser.add_argument("--iteration", default="V13-G1")
    parser.add_argument("--preflight-only", action="store_true", help="Verify inputs without creating any files")
    args = parser.parse_args()
    if args.preflight_only:
        checked = inspect(args.parent, args.review, args.series)
        print(json.dumps({"series": checked["series"], "ready_to_train": checked["ready_to_train"],
                          "reason": None if checked["ready_to_train"] else "No new series-specific grain exclusions supplied",
                          "parent_regions": checked["parent_regions"], "new_region_ids": checked["new_region_ids"],
                          "cumulative_regions": len(checked["regions"]), "physics_profile": checked["profile"]["format"],
                          "evaluation_mode": checked["previous"]["settings"]["evaluation_mode"],
                          "physics_runtime": checked["runtime"]}, indent=2))
    else:
        if args.output is None or args.cache is None or not args.review:
            parser.error("Preparation requires --output, --cache and at least one --review; otherwise use --preflight-only")
        prepare(args.parent, args.review, args.output, args.cache, args.iteration, args.series)


if __name__ == "__main__":
    main()
