"""Publish a complete normalized full-head candidate for human review.

The completed generic CO2 and gas fits may both change. Publication verifies
their exact exported parameters, frozen inputs, native assets and Studio checks;
it does not confer scientific acceptance or activate a deployed model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower.phase_learning import SerializedPhaseHead, inspect_phase_model_bundle
from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.export_normalized_grain_candidate import HEADS, exact_parameters, logistic_fit, validate_fit_report
from darsia.fluidflower.tools.materialize_phase_iteration import relocate
from darsia.fluidflower.tools.nas_publish import clear_hidden_attribute


def read(path):
    return Path(path).read_text(encoding="utf-8-sig")


def document(path):
    return json.loads(read(path))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_at(path, source, logical_root):
    """Resolve staged references written for the final destination."""
    path = Path(path).resolve()
    return source / path.relative_to(logical_root) if path.is_relative_to(logical_root) else path


def checked_hash(path, digest, label):
    require(Path(path).is_file() and sha256_file(path) == digest, f"Changed or missing {label}: {path}")


def verify_reload(checked, head):
    # The fitted CPU predictor uses float32 and the serialized predictor uses
    # float64 before its float32 output. Match the exporter's explicit contract.
    require(checked["fitted_parameters_exact"] is True and type(checked["rows"]) is int
            and checked["rows"] > 0 and math.isfinite(checked["max_probability_abs_difference"])
            and 0 <= checked["max_probability_abs_difference"] <= 1e-5
            and math.isfinite(checked["classification_agreement"])
            and .99999 <= checked["classification_agreement"] <= 1
            and checked["display_threshold"] == (.6 if head == "gas_within_co2" else .5),
            "Export reload verification failed")


def verify_export(source, logical_root):
    summary = document(source / "summary.json")
    receipt = document(source / "export_verification.json")
    identity = _model_identity(source)
    require((receipt["model_id"], receipt["model_sha256"]) == identity,
            "Export receipt belongs to another model revision")
    require(receipt["fitted_parameters_preserved"] is True and receipt["training_performed"] is False,
            "Candidate does not preserve the completed fits")
    require(receipt["adapters"] == ["generic"] and set(summary["models"]) == {"generic"},
            "Candidate must contain only the completed generic adapter")
    require(summary.get("human_review_required") is True
            and summary.get("development_cv_not_independent_acceptance") is True
            and not summary.get("human_validation") and not summary.get("deployment"),
            "Candidate status must require human review")
    series = receipt["series"]
    require(series in {"ffac", "ffum", "ffbm"} and summary["inventory"]["datasets"] == [series],
            "Export series differs from model inventory")
    require(receipt["regions"] == summary["inventory"]["regions"] > 0,
            "Export evidence count differs from model inventory")
    require(set(summary["models"]["generic"]["heads"]) == set(HEADS)
            and set(receipt["verification"]) == set(HEADS), "Export is missing an optical head")
    require(bool(receipt["protected_sha256"]) and bool(receipt["copied_sources"]),
            "Export lacks frozen input provenance")
    for path, digest in receipt["protected_sha256"].items():
        checked_hash(Path(path), digest, "protected source")
    for row in receipt["copied_sources"]:
        checked_hash(Path(row["source"]), row["sha256"], "copied source")
        checked_hash(path_at(row["target"], source, logical_root), row["sha256"], "frozen copy")
    request = document(source / "training_request.json")
    require(request["model_id"] == identity[0] and request["settings"]["iteration"] == receipt["iteration"],
            "Training request differs from exported identity")
    require(request["bundles"], "No frozen review bundles")
    for bundle in request["bundles"]:
        for role in ("decisions", "combined_manifest"):
            row = bundle[role]
            checked_hash(path_at(row["snapshot_path"], source, logical_root), row["sha256"], "review evidence")
    for head in HEADS:
        folder = source / "fit_source_snapshot/generic" / head
        fit_payload, report = document(folder / "selected_fit.json"), document(folder / "head_report.json")
        loaded = SerializedPhaseHead.load(source / "models/generic" / f"{head}_signal.json")
        validate_fit_report(fit_payload, report, head, summary["runs"], len(loaded.mean))
        exact_parameters(loaded, logistic_fit(fit_payload))
        checked = receipt["verification"][head]
        verify_reload(checked, head)
        checked_hash(folder / "selected_fit.json", checked["selected_fit_sha256"], "selected fit")
    profile = load_physics_profile(source)
    runtime = physics_runtime_identity(profile)
    inheritance = document(source / "physics_inheritance.json")
    require(runtime == inheritance["physics_runtime"] and inheritance["settings_unchanged"] is True
            and inheritance["calibration_unchanged"] is True, "Inherited physics changed")
    for entry in profile["runs"].values():
        checked_hash(source / entry["path"], entry["sha256"], "physical context")
    inspect_phase_model_bundle(source)
    return identity, runtime, series, profile, receipt["iteration"]


def verify_inference(source, binding, jobs, metadata):
    """Check actual saved parity values, rather than accepting a success flag."""
    report = document(source / "validation/inference_report.json")
    require(bool(report.get("completed_utc")), "Native inference report is incomplete")
    require(len(report["runs"]) == len(jobs), "Inference run coverage is incomplete")
    require(report.get("model_hashes"), "Inference report lacks model hashes")
    for path, digest in report["model_hashes"].items():
        checked_hash(source / path, digest, "materialized model")
    provenance = binding["inference_provenance"]
    require(provenance["backend"] == "cuda-float32-cached-features-checked"
            and provenance["cpu_reference_frames"] == [0, 1, 10], "Missing fresh CPU/CUDA parity")
    for key, name in (("materializer_sha256", "materialize_phase_iteration.py"),
                      ("backend_sha256", "phase_inference_benchmark_backends.py")):
        checked_hash(source / "validation/code_snapshot" / name, provenance[key], "inference code snapshot")
    for saved, job, meta in zip(report["runs"], jobs, metadata):
        expected = list(range(meta["frame_count"]))
        require([row["frame"] for row in saved["frames"]] == expected,
                f"Incomplete inference timeline: {job['run']}")
        require(saved["shape_hw"] == meta["content_box_xywh"][2:][::-1],
                f"Inference geometry differs: {job['run']}")
        expected_parity = {frame for frame in (0, 1, 10) if frame < meta["frame_count"]}
        require(len(saved["parity"]) == len(expected_parity)
                and {row["frame"] for row in saved["parity"]} == expected_parity,
                f"Incomplete CPU/CUDA parity coverage: {job['run']}")
        for row in saved["parity"]:
            require(set(row["heads"]) == {"co2_presence_probability", "gas_probability_conditional_on_co2"},
                    "CPU/CUDA parity is missing a head")
            for values in row["heads"].values():
                require(all(math.isfinite(values[key]) and values[key] >= 0
                            for key in ("q99_abs_error", "max_abs_error", "disagreement")),
                        "Nonfinite CPU/CUDA parity")
                require(values["q99_abs_error"] <= .001 and values["disagreement"] <= .0001,
                        "CPU/CUDA parity exceeds the materializer tolerances")


def verify(source, qa, *, logical_root=None):
    source, qa = Path(source).resolve(), Path(qa).resolve()
    logical_root = source if logical_root is None else Path(logical_root).resolve()
    identity, runtime, series, profile, iteration = verify_export(source, logical_root)
    binding = document(source / "validation/validation_binding.json")
    manifest = document(source / "validation/postflight_manifest.json")
    for value in (binding, manifest):
        require((value["model_id"], value["model_sha256"]) == identity
                and value["physics_runtime"] == runtime, "Native validation belongs to another revision")
        checked_hash(source / "review_source/postflight_manifest.json", value["source_manifest_sha256"],
                     "review source manifest")
    require(binding["inference_provenance"] == manifest["inference_provenance"],
            "Native inference provenance differs between binding and manifest")
    require(path_at(binding["combined_root"], source, logical_root) == source / "validation",
            "Studio binding selects another validation campaign")
    checked = document(qa / "report.json")
    require(Path(checked["model"]).resolve() == logical_root
            and (checked["model_id"], checked["model_sha256"]) == identity
            and checked["physics_runtime"] == runtime
            and checked.get("model_and_binding_unchanged") is True, "Studio QA belongs to another revision or is incomplete")
    for name, key in (("validation_binding.json", "validation_binding_sha256"),
                      ("postflight_manifest.json", "validation_manifest_sha256")):
        checked_hash(source / "validation" / name, checked[key], "QA validation binding")
    require(set(checked["ui"]) == {"open_correction", "blind_region_labeling"}
            and all(type(row["returncode"]) is int and row["returncode"] == 0 for row in checked["ui"].values()),
            "Both Studio entry points must pass")
    jobs = _source_jobs(source / "validation")
    require(jobs and {j["dataset"] for j in jobs} == {series}, "Native jobs differ from the candidate series")
    keys = {f'{j["dataset"]}/{j["run"]}' for j in jobs}
    require(len(keys) == len(jobs) and keys == set(profile["runs"])
            and keys == {f'{j["dataset"]}/{j["run"]}' for j in _source_jobs(source / "review_source")}
            and {j["run"] for j in jobs} == set(checked["runs"]), "Native run coverage is incomplete")
    metadata, total = [], 0
    for job in jobs:
        native = path_at(job["review_output_dir"], source, logical_root) / "native_frames" / job["run"] / "manifest.json"
        meta = document(native)
        checked_hash(native, profile["runs"][f'{series}/{job["run"]}']["native_manifest_sha256"], "native source manifest")
        rows = checked["runs"][job["run"]]
        require(type(meta["frame_count"]) is int and meta["frame_count"] > 0
                and len(meta["frame_sources"]) == meta["frame_count"]
                and [r["frame"] for r in rows] == list(range(meta["frame_count"])), "QA native timeline is incomplete")
        require(rows[0]["elapsed_h"] <= 0 and rows[0]["gas"] == rows[0]["aqueous"] == 0,
                "Missing or nonempty pre-injection frame")
        for row, frame in zip(rows, meta["frame_sources"]):
            require(row["elapsed_h"] == frame["elapsed_h"]
                    and all(type(row[k]) is int and row[k] >= 0 for k in ("gas", "aqueous")),
                    "QA frame time/counts differ from the native timeline")
            suffix = Path("native_phase_maps") / job["run"] / f'A_{job["params_sha256"][:16]}' / f'frame_{row["frame"]:03d}.png'
            path = path_at(job["phase_map_output_dir"], source, logical_root) / suffix
            require(path.is_relative_to(source / "validation"), "Final map points outside this materialized candidate")
            checked_hash(path, row["map_sha256"], "native phase map")
            # Raw maps precede physical support and are keyed by optical-model
            # identity; final maps use the combined optical/physical pipeline.
            raw = (path_at(job["raw_phase_map_output_dir"], source, logical_root)
                   / "native_phase_maps" / job["run"] / f"A_{identity[1][:16]}"
                   / f'frame_{row["frame"]:03d}.png')
            require(raw.is_relative_to(source / "validation") and raw.is_file(), "Missing raw optical native map")
        metadata.append(meta)
        total += len(rows)
    require(binding["run_count"] == len(jobs), "Studio binding has incomplete run coverage")
    verify_inference(source, binding, jobs, metadata)
    return identity, runtime, total, iteration


def validate_locations(source, destination, staging_root):
    require(source != destination and not source.is_relative_to(destination)
            and not destination.is_relative_to(source), "Source and destination must be separate model folders")
    if destination.exists():
        raise FileExistsError(destination)
    require(not staging_root.is_relative_to(source), "Staging must be outside the source candidate")
    library = destination.parents[2]
    require(staging_root not in (library, library.parent) and not staging_root.is_relative_to(library),
            "Staging must be outside the scanned model library")


def publish(source, destination, staging_root, qa):
    source, destination, staging_root, qa = (Path(p).resolve() for p in (source, destination, staging_root, qa))
    validate_locations(source, destination, staging_root)
    identity, runtime, frames, iteration = verify(source, qa)
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    hashes = {p.relative_to(source).as_posix(): sha256_file(p) for p in paths}
    qa_paths = sorted(p for p in qa.rglob("*") if p.is_file())
    qa_hashes = {p.relative_to(qa).as_posix(): sha256_file(p) for p in qa_paths}
    staging = staging_root / f"normalized-grain-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    written = {}
    for path in paths:
        relative = path.relative_to(source)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frozen = (relative.as_posix() == "summary.json" or relative.parts[0] in {
            "models", "input_snapshot", "code_snapshot", "reference_code_snapshot", "fit_source_snapshot", "review_source"})
        changed = False
        if path.suffix == ".json" and not frozen:
            value = document(path)
            updated = relocate(value, source, destination)
            if value != updated:
                target.write_text(json.dumps(updated, indent=2, allow_nan=False), encoding="utf-8")
                changed = True
        if not changed:
            shutil.copy2(path, target)
            checked_hash(target, hashes[relative.as_posix()], "staged file")
        written[relative.as_posix()] = dict(sha256=sha256_file(target), paths_relocated=changed)
    require(not (staging / "studio_verification").exists(), "Candidate already contains Studio publication QA")
    for path in qa_paths:
        target = staging / "studio_verification" / path.relative_to(qa)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        checked_hash(target, qa_hashes[path.relative_to(qa).as_posix()], "copied Studio QA")
    report_path = staging / "studio_verification/report.json"
    original = document(report_path)
    (report_path.parent / "source_report.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
    report = relocate(original, source, destination)
    for name, key in (("validation_binding.json", "validation_binding_sha256"),
                      ("postflight_manifest.json", "validation_manifest_sha256")):
        report["source_" + key] = report[key]
        report[key] = sha256_file(staging / "validation" / name)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    require(verify(staging, report_path.parent, logical_root=destination) == (identity, runtime, frames, iteration),
            "Staged candidate changed identity or verification")
    for relative, digest in hashes.items():
        checked_hash(source / relative, digest, "original candidate file")
    for relative, digest in qa_hashes.items():
        checked_hash(qa / relative, digest, "original Studio QA")
    for path in (staging / "studio_verification").rglob("*"):
        if path.is_file():
            written[path.relative_to(staging).as_posix()] = dict(sha256=sha256_file(path), paths_relocated=path == report_path)
    receipt = dict(created_utc=utc_now(), source=str(source), destination=str(destination),
        model_id=identity[0], model_sha256=identity[1], physics_runtime=runtime, iteration=iteration,
        artifact_kind="normalized-full-head-review-candidate", verified_frames=frames,
        training_during_publication=False, human_approval=False, scientific_acceptance=False,
        automatic_activation=False, original_files_unchanged=True, source_files=hashes,
        source_qa_files=qa_hashes, published_files=written)
    (staging / "publication_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    clear_hidden_attribute(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    require(_model_identity(destination) == identity
            and physics_runtime_identity(load_physics_profile(destination)) == runtime, "Published identity changed")
    print(json.dumps(dict(published=str(destination), verified_frames=frames, iteration=iteration,
                          human_review_required=True), indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", "--model", dest="source", type=Path, required=True)
    for name in ("destination", "staging-root", "qa"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.destination, args.staging_root, args.qa)
