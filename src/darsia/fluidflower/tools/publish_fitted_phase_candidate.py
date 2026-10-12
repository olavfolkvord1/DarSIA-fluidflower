"""Expose finished fits immediately; attach complete native review maps later.

Both operations stage outside the scanned library and rename complete folders.
They preserve model identity and never imply human approval or deployment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower.phase_training.scope import requested_series, validate_training_series
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.materialize_phase_iteration import relocate
from darsia.fluidflower.tools.nas_publish import publish_immutable_directory, remove_owned_stage
from darsia.fluidflower.tools.prepare_grain_gas_retraining import physics_files
from darsia.fluidflower.tools.publish_normalized_grain_candidate import checked_hash, path_at, require, verify_inference
from darsia.fluidflower.tools.run_baseline_grain_iteration import verify_backend_report


ROOT_FILES = {
    "summary.json", "training_request.json", "training_inventory.json", "region_inventory.csv",
    "training_samples.npz", "phase_decision_policy.json", "cross_validation.json",
    "change_evidence_ablation.json", "radiometry_inventory.json", "evidence_preflight.json",
    "physics_inheritance.json", "execution_code_sha256.json", "series_iteration.json",
    "training_status.json", "continuation_preparation.json", "review_retractions.json",
}
FROZEN_DIRECTORIES = {"models", "input_snapshot", "code_snapshot", "execution_code_snapshot",
                      "fit_source_snapshot", "reference_code_snapshot", "review_source",
                      "continuation_evidence"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def verify_candidate_backend_report(report, *, allow_nonconverged=False):
    """Waive only a known optimizer-convergence failure for human review."""
    try:
        backend = verify_backend_report(report, allowed_backends=("cpu", "cuda"))
    except ValueError as error:
        # The strict verifier checks backend and numerical/parity evidence first.
        # Never fall back on an unrelated report, artifact or numerical failure.
        if (not allow_nonconverged or str(error) != "Final fit did not converge"
                or report.get("final_fit", {}).get("converged") is not False):
            raise
        backend = report["final_fit"]["backend"]
    require(math.isfinite(float(report["final_fit"]["objective"])), "Nonfinite fitted objective")
    return backend


def verify_fitted(source, *, logical_root=None, allow_nonconverged=False):
    source = Path(source).resolve()
    logical_root = source if logical_root is None else Path(logical_root).resolve()
    request, summary = read(source / "training_request.json"), read(source / "summary.json")
    from darsia.fluidflower.phase_review_retractions import model_retractions
    withdrawal_snapshot = model_retractions(source)  # verifies the frozen policy, including after relocation
    identity = _model_identity(source)
    series = requested_series(request["settings"])
    require(series in {"ffac", "ffum", "ffbm"}, "Expected an explicit experimental series")
    require(request["model_id"] == identity[0] == summary["campaign_id"], "Fitted model identity differs")
    require(summary["inventory"]["datasets"] == [series], "Model inventory contains another series")
    pairs = []
    for bundle in request["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            path = path_at(record["snapshot_path"], source, logical_root)
            require(path.is_relative_to(source / "input_snapshot"), "Frozen input escaped its model")
            checked_hash(path, record["sha256"], "frozen review input")
            pair.append(path)
        pairs.append(t.ReviewBundle(*pair))
    sources, regions, _ = t._load_sources_and_regions(
        pairs, withdrawal_snapshot=withdrawal_snapshot,
        preserve_native_revisions=bool(request["settings"].get("preserve_native_revisions", False)))
    validate_training_series(sources, regions, series)
    require(len(regions) == summary["inventory"]["regions"] > 0, "Fitted evidence count differs")
    require(set(summary["models"]) == {p.name for p in (source / "models").iterdir() if p.is_dir()},
            "Model adapters differ from their summary")
    nonconverged_heads = []
    for adapter, variant in summary["models"].items():
        require(set(variant["heads"]) == {"co2_presence", "gas_within_co2"}, "Missing fitted optical head")
        for head, report in variant["heads"].items():
            verify_candidate_backend_report(report, allow_nonconverged=allow_nonconverged)
            if report["final_fit"].get("converged") is False:
                nonconverged_heads.append(dict(adapter=adapter, head=head,
                    iterations=report["final_fit"]["iterations"], converged=False,
                    objective=report["final_fit"]["objective"]))
            path = source / "models" / adapter / f"{head}_signal.json"
            payload = read(path)
            require(all(math.isfinite(float(value)) for name in
                    ("coefficients", "feature_mean", "feature_scale") for value in payload[name])
                    and all(float(value) > 0 for value in payload["feature_scale"])
                    and math.isfinite(float(payload["intercept"])), "Invalid fitted numerical parameters")
            provenance = payload["training_provenance"]
            require(provenance["campaign_id"] == identity[0]
                    and provenance.get("series_scope") == series
                    and provenance["training_inventory"] == summary["inventory"], "Head provenance differs from the request")
            require(payload["model_sha256"] == report["model_sha256"], "Head digest differs from its report")
    inspect_phase_model_bundle(source)
    profile = load_physics_profile(source)
    require(profile is not None, "Candidate requires its inherited physics context")
    runtime = physics_runtime_identity(profile)
    if (source / "physics_inheritance.json").is_file():
        inheritance = read(source / "physics_inheritance.json")
        require(inheritance["physics_runtime"] == runtime, "Inherited physics runtime changed")
    for entry in profile["runs"].values():
        context = (source / entry["path"]).resolve()
        require(context.is_relative_to(source), "Physical context escaped its model")
        checked_hash(context, entry["sha256"], "inherited physical context")
    require((source / "review_source/postflight_manifest.json").is_file(), "Missing review source manifest")
    for receipt_path in source.glob("*_fit_verification.json"):
        receipt = read(receipt_path)
        require((receipt["model_id"], receipt["model_sha256"]) == identity,
                "Completed-fit receipt identifies another model")
    return dict(model_id=identity[0], model_sha256=identity[1], series=series,
        iteration=request["settings"].get("iteration"), regions=len(regions),
        physics_runtime=runtime, adapters=sorted(summary["models"]), nonconverged_heads=nonconverged_heads)


def fitted_files(source):
    """An allowlist prevents copying in-progress validation and scratch folders."""
    source = Path(source).resolve()
    paths = {p for p in source.iterdir() if p.is_file()
             and (p.name in ROOT_FILES or p.name.endswith("_fit_verification.json"))}
    for name in FROZEN_DIRECTORIES:
        if (source / name).is_dir():
            paths.update(p for p in (source / name).rglob("*") if p.is_file())
    paths.update(physics_files(source, load_physics_profile(source)))
    return sorted(paths)


def publication_library(destination):
    """Find the enclosing DarSIA model/results library, including shallow models."""
    for parent in destination.parents:
        if (parent.name.casefold() in {"models", "phase_models", "results"}
                and parent.parent.name.casefold() == "darsia"):
            return parent
    # Older layouts use iteration/series/model beneath a dedicated library.
    return destination.parents[2]


def destination_volume_staging_root(destination):
    """Keep incomplete copies beside, never inside, the scanned library."""
    return publication_library(destination).parent / "publication-staging"


def validate_locations(source, destination, staging_root, *, attaching=False):
    require(source != destination and not source.is_relative_to(destination)
            and not destination.is_relative_to(source), "Source and published model must be separate")
    library = publication_library(destination)
    require(not staging_root.is_relative_to(library) and staging_root not in {library, library.parent}
            and not staging_root.is_relative_to(source), "Stage outside the scanned model library and source")
    if not attaching and destination.exists():
        raise FileExistsError(destination)
    if attaching and (destination / "validation").exists():
        raise FileExistsError(destination / "validation")


def copy_fitted(source, destination, staging, paths, hashes):
    written = {}
    for path in paths:
        relative = path.relative_to(source)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # All physics/calibration and code directories retain exact source bytes.
        # Only explicitly listed root bookkeeping documents relocate paths.
        frozen = relative.as_posix() in {"summary.json", "review_retractions.json"} or relative.parts[0] not in ROOT_FILES
        changed = False
        if path.suffix == ".json" and not frozen:
            value = read(path)
            relocated = relocate(value, source, destination)
            if relocated != value:
                write(target, relocated)
                changed = True
        if not changed:
            shutil.copy2(path, target)
            checked_hash(target, hashes[relative.as_posix()], "staged fitted artifact")
        written[relative.as_posix()] = dict(sha256=sha256_file(target), paths_relocated=changed)
    return written


def publish_fitted(source, destination, staging_root, *, allow_nonconverged=False):
    source, destination, staging_root = (Path(p).resolve() for p in (source, destination, staging_root))
    validate_locations(source, destination, staging_root)
    checked = verify_fitted(source, allow_nonconverged=allow_nonconverged)
    paths = fitted_files(source)
    hashes = {p.relative_to(source).as_posix(): sha256_file(p) for p in paths}
    # Older summaries derive their ID from the folder name; keep it stable even
    # while the complete candidate is hidden outside the scanned library.
    stage_parent = staging_root / f"fitted-candidate-{uuid.uuid4().hex}"
    staging = stage_parent / destination.name
    staging.mkdir(parents=True, exist_ok=False)
    try:
        written = copy_fitted(source, destination, staging, paths, hashes)
        staged = verify_fitted(staging, logical_root=destination, allow_nonconverged=allow_nonconverged)
        require(staged == checked, "Staging changed the completed model or evidence")
        for relative, expected in hashes.items():
            checked_hash(source / relative, expected, "original fitted artifact")
        receipt = dict(checked, created_utc=utc_now(), source=str(source), destination=str(destination),
            artifact_kind="fitted-phase-review-candidate", native_validation_pending=True,
            human_review_required=True, human_approval=False, scientific_acceptance=False,
            automatic_activation=False, training_during_publication=False, original_files_unchanged=True,
            nonconverged_review_candidate=bool(checked["nonconverged_heads"]),
            source_files=hashes, published_files=written)
        write(staging / "publication_receipt.json", receipt)

        def verify_published_copy(path):
            require(verify_fitted(path, logical_root=destination,
                                  allow_nonconverged=allow_nonconverged) == checked,
                    "Destination-volume copy changed the completed model or evidence")

        # The caller's stage can be on C: while the final model is on an SMB
        # share. A second checked copy outside the scanned library makes the
        # final rename stay on the destination volume.
        publish_immutable_directory(staging, destination, destination_volume_staging_root(destination),
                                    verify=verify_published_copy)
        return receipt
    finally:
        # This invocation alone created this UUID folder.
        remove_owned_stage(staging_root, stage_parent, "fitted-candidate-")


def validation_path(path, validation, logical_root):
    path = Path(path).resolve()
    logical = logical_root / "validation"
    return validation / path.relative_to(logical) if path.is_relative_to(logical) else path


def verify_validation(model, validation, *, logical_root=None, qa=None):
    model, validation = Path(model).resolve(), Path(validation).resolve()
    logical_root = model if logical_root is None else Path(logical_root).resolve()
    identity = _model_identity(model)
    runtime = physics_runtime_identity(load_physics_profile(model))
    binding, manifest = read(validation / "validation_binding.json"), read(validation / "postflight_manifest.json")
    for value in (binding, manifest):
        require((value["model_id"], value["model_sha256"]) == identity
                and value["physics_runtime"] == runtime, "Review maps identify another fitted model/physics revision")
        checked_hash(model / "review_source/postflight_manifest.json", value["source_manifest_sha256"], "review source")
    require(binding.get("completed_utc") and Path(binding["combined_root"]).resolve() == logical_root / "validation",
            "Validation binding is incomplete or points elsewhere")
    jobs, metadata, frames = _source_jobs(validation), [], 0
    profile = load_physics_profile(model)
    require({f'{j["dataset"]}/{j["run"]}' for j in jobs} == set(profile["runs"])
            and len(jobs) == binding["run_count"], "Native validation run coverage differs")
    for job in jobs:
        native = validation_path(Path(job["review_output_dir"]) / "native_frames" / job["run"] / "manifest.json", validation, logical_root)
        meta = read(native)
        checked_hash(native, profile["runs"][f'{job["dataset"]}/{job["run"]}']["native_manifest_sha256"], "native timeline")
        require(meta["frame_count"] > 0 and len(meta["frame_sources"]) == meta["frame_count"], "Incomplete native timeline")
        for index in range(meta["frame_count"]):
            suffix = Path("native_phase_maps") / job["run"] / f'A_{job["params_sha256"][:16]}' / f'frame_{index:03d}.png'
            path = validation_path(Path(job["phase_map_output_dir"]) / suffix, validation, logical_root)
            require(path.is_relative_to(validation) and path.is_file(), "Missing final native phase map")
            if job.get("raw_phase_map_output_dir"):
                raw = validation_path(Path(job["raw_phase_map_output_dir"]) / "native_phase_maps" / job["run"]
                    / f'A_{identity[1][:16]}' / f'frame_{index:03d}.png', validation, logical_root)
                require(raw.is_relative_to(validation) and raw.is_file(), "Missing raw native optical map")
        metadata.append(meta)
        frames += meta["frame_count"]
    if qa is not None:
        report = read(Path(qa) / "report.json")
        require(Path(report["model"]).resolve() == logical_root and (report["model_id"], report["model_sha256"]) == identity
                and report["physics_runtime"] == runtime and report["model_and_binding_unchanged"] is True,
                "Studio QA identifies another revision or is incomplete")
        for name, key in (("validation_binding.json", "validation_binding_sha256"),
                          ("postflight_manifest.json", "validation_manifest_sha256")):
            checked_hash(validation / name, report[key], "Studio QA binding")
        require(set(report["ui"]) == {"open_correction", "blind_region_labeling"}
                and all(row["returncode"] == 0 for row in report["ui"].values()), "Studio QA did not pass")
    # The checked CUDA materializer records actual per-frame parity. A regular
    # CPU materialization may instead supply only the standard complete binding.
    if validation == model / "validation" and (validation / "inference_report.json").is_file():
        verify_inference(model, binding, jobs, metadata)
    return dict(model_id=identity[0], model_sha256=identity[1], physics_runtime=runtime,
        runs=[j["run"] for j in jobs], frames=frames)


def attach_validation(source, destination, staging_root, qa=None, *, allow_nonconverged=False):
    from darsia.fluidflower.tools.phase_reference_readiness import RECEIPT, mark_references_pending, prepare_validation_references
    source, destination, staging_root = (Path(p).resolve() for p in (source, destination, staging_root))
    pending_receipt = destination / "validation/attachment_receipt.json"
    if pending_receipt.is_file() and read(pending_receipt).get("native_validation_pending") is True:
        receipt = read(pending_receipt)
        require(Path(receipt["source"]).resolve() == source, "Pending attachment belongs to another source")
        require(verify_fitted(source, allow_nonconverged=allow_nonconverged)
                == verify_fitted(destination, allow_nonconverged=allow_nonconverged),
                "Pending fitted candidate differs from its source")
        require(verify_validation(source, source / "validation", qa=qa)
                == verify_validation(destination, destination / "validation"),
                "Pending native validation differs from its source")
        for relative, expected in receipt["published_files"].items():
            checked_hash(destination / "validation" / relative, expected, "pending review asset")
        receipt["reference_phase_maps"] = prepare_validation_references(
            destination / "validation", lambda message: print(message, flush=True))
        receipt["native_validation_pending"] = False
        write(pending_receipt, receipt)
        return receipt
    validate_locations(source, destination, staging_root, attaching=True)
    original = verify_fitted(source, allow_nonconverged=allow_nonconverged)
    published = verify_fitted(destination, allow_nonconverged=allow_nonconverged)
    require(original == published, "Published fitted candidate differs from the materialized source")
    checked = verify_validation(source, source / "validation", qa=qa)
    validation = source / "validation"
    # Reference readiness binds final validation paths and is regenerated there.
    # Copying the source receipt would also freeze a hash we replace on retry.
    paths = sorted(p for p in validation.rglob("*") if p.is_file() and p != validation / RECEIPT)
    hashes = {p.relative_to(validation).as_posix(): sha256_file(p) for p in paths}
    staging = staging_root / f"review-maps-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        written = {}
        for path in paths:
            relative = path.relative_to(validation)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                write(target, relocate(read(path), source, destination))
            else:
                shutil.copy2(path, target)
                checked_hash(target, hashes[relative.as_posix()], "staged review asset")
            written[relative.as_posix()] = sha256_file(target)
        if qa is not None:
            for path in Path(qa).rglob("*"):
                if path.is_file():
                    target = staging / "studio_verification" / path.relative_to(qa)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    checked_hash(target, sha256_file(path), "Studio QA copy")
            report_path = staging / "studio_verification/report.json"
            report = read(report_path)
            write(report_path.with_name("source_report.json"), report)
            report = relocate(report, source, destination)
            for name, key in (("validation_binding.json", "validation_binding_sha256"),
                              ("postflight_manifest.json", "validation_manifest_sha256")):
                report["source_" + key] = report[key]
                report[key] = sha256_file(staging / name)
            write(report_path, report)
        staged = verify_validation(destination, staging, logical_root=destination,
                                   qa=staging / "studio_verification" if qa is not None else None)
        require(staged == checked, "Staging changed native review coverage or identity")
        for relative, expected in hashes.items():
            checked_hash(validation / relative, expected, "original native validation")
        receipt = dict(checked, created_utc=utc_now(), source=str(source), destination=str(destination),
            native_validation_pending=False, human_review_required=True, human_approval=False,
            nonconverged_heads=original["nonconverged_heads"],
            scientific_acceptance=False, automatic_activation=False, source_files=hashes, published_files=written)
        write(staging / "attachment_receipt.json", {**receipt, "native_validation_pending": True})
        mark_references_pending(staging)

        def verify_published_copy(path):
            require(verify_validation(destination, path, logical_root=destination,
                                      qa=path / "studio_verification" if qa is not None else None) == checked,
                    "Destination-volume copy changed native review coverage or identity")

        publish_immutable_directory(staging, destination / "validation",
                                    destination_volume_staging_root(destination), verify=verify_published_copy)
        receipt["reference_phase_maps"] = prepare_validation_references(
            destination / "validation", lambda message: print(message, flush=True))
        write(destination / "validation/attachment_receipt.json", receipt)
        return receipt
    finally:
        # The completed validation may remain pending for a safe retry.
        remove_owned_stage(staging_root, staging, "review-maps-")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fitted", "validation"))
    for name in ("source", "destination", "staging-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--allow-nonconverged", action="store_true",
        help="Expose numerically checked but nonconverged fits for human review, with warnings")
    args = parser.parse_args()
    result = (publish_fitted(args.source, args.destination, args.staging_root,
                            allow_nonconverged=args.allow_nonconverged) if args.operation == "fitted"
              else attach_validation(args.source, args.destination, args.staging_root, args.qa,
                                     allow_nonconverged=args.allow_nonconverged))
    print(json.dumps({k: result[k] for k in ("destination", "model_id", "model_sha256", "native_validation_pending")}, indent=2))


if __name__ == "__main__":
    main()
