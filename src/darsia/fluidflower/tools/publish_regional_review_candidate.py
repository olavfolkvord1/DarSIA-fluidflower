"""Publish a completed, reviewed regional candidate into its Studio library.

The training coordinator deliberately writes under ``results``. This separate,
opt-in command verifies its complete native review package, copies it to
a staging directory outside ``models``, rewrites only movable bookkeeping paths,
and atomically renames the fully checked folder into ``models``. It never fits
models, makes phase maps, or activates a model for deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.phase_features import RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION
from darsia.fluidflower.phase_runtime import load_physics_profile, physics_runtime_identity
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.series_paths import series_path
from darsia.fluidflower.tools.materialize_phase_iteration import relocate
from darsia.fluidflower.tools.nas_publish import (
    _rename_directory_no_replace, clear_hidden_attribute, remove_owned_stage,
)
from darsia.fluidflower.tools.phase_reference_maps import _asset


FORMAT = "regional-review-publication-v1"
QA_FORMAT = "regional-review-publication-qa-v1"
FAILED_QA_FORMAT = "ffac-regional-review-candidate-qa-decision-v1"
FAILED_V5_QA_FORMAT = "ffac-review-candidate-qa-decision-v1"
FAILED_UM_QA_FORMAT = "ffum-fast-regional-review-candidate-qa-decision-v1"
FAILED_UM_QUANT_FORMAT = "fluidflower-series-phase-candidate-comparison-v1"
FAILED_UM_MODEL_ID = "phase-model-20260926T174424Z-c6f5d253-ffum-unified"
FAILED_UM_MODEL_SHA256 = "89c6b26370abc01806527d584eeb8fa8ae547c87869c15b740efea48288d6911"
FAILED_BM_QA_FORMAT = "ffbm-fast-regional-review-candidate-qa-decision-v1"
FAILED_BM_JOB = "optical-regional-fast-v1-bm-certified-20260926T2028Z"
FAILED_BM_MODEL_ID = "phase-model-20260926T203541Z-5392e612-ffbm-unified"
FAILED_BM_MODEL_SHA256 = "005d7f69e40f5e57eb0a058e8f8b33720a492e307febba83ece5c754ef7e38a6"
FAILED_BM_QUANT_SHA256 = "f9c881df1c8ea6bbf881ff7a5499b442c4d4e8f1fa6a52d422cefd3fe808a212"
FAILED_BM_VISUAL_SHA256 = "94fa5379fe8835bd9e31e52a30a125816ee28150809950bd430317b22de001ec"
FAILED_BM_ASSESSMENT_SHA256 = "d8b486610f7d31d9e74c381ee7f54f7bd6b0815f489c49930ef5778186288d48"
VALIDATION_ONLY_QA = "validation_only_qa"
FAILED_QA_REPORTS = {
    "candidate_comparison/qa_failure_report.md",
    "visual_qa_ac47_ac51/visual_qa_report.md",
    "provenance_qa_receipt.json",
}
QA_ROOT_DOCUMENTS = {
    "candidate_qa_decision.json", "candidate_qa_state.json",
    "provenance_qa_receipt.json", "provenance_qa.md", "qa_plan.md",
    "ac51_stagewise_gas_audit.json", "ac51_stagewise_gas_audit.md",
    "result.json", "preflight.json", "followup.md",
}
QA_DIRECTORIES = {"candidate_comparison", "visual_qa_ac47_ac51"}
EXPECTED_ADAPTERS = {"generic", "btb-ratiometric-relative-v2"}
GENERIC_ONLY_PROFILE = "generic-co2-fast-v1"
EXPECTED_RUN_COUNTS = {"ffac": 43, "ffum": 13, "ffbm": 5}
FROZEN_TOP_LEVEL = {
    "models", "input_snapshot", "code_snapshot", "execution_code_snapshot",
    "fit_source_snapshot", "reference_code_snapshot", "review_source",
    "continuation_evidence", "physics_context", "regional_model",
    "parent_provenance",
}
FROZEN_ROOT_JSON = {
    "summary.json", "phase_optical_regions.json", "regional_model.json",
    "review_retractions.json", "native_source_receipt.json",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _physical(path: str | Path, root: Path, logical_root: Path) -> Path:
    """Locate an internal final-path reference inside a still-staged copy."""
    value = Path(path).resolve()
    return root / value.relative_to(logical_root) if value.is_relative_to(logical_root) else value


def _inside(path: Path, parent: Path, label: str) -> Path:
    result = path.resolve()
    _require(result.is_relative_to(parent.resolve()), f"{label} escaped the model: {path}")
    return result


def _check_reference_receipt(package: Path, root: Path, logical_root: Path) -> str:
    receipt_path = package / "phase_reference_readiness.json"
    _require(receipt_path.is_file(), f"Missing reference readiness: {package}")
    receipt = _read(receipt_path)
    _require(receipt.get("format") == "phase-reference-readiness-v1"
             and receipt.get("status") in {"ready", "deferred"},
             f"Reference readiness is incomplete: {package}")
    for key, name in (("binding", "validation_binding.json"),
                      ("campaign", "postflight_manifest.json")):
        entry = receipt.get(key) or {}
        expected = logical_root / package.relative_to(root) / name
        _require(Path(entry.get("path", "")).resolve() == expected
                 and entry.get("sha256") == sha256_file(package / name),
                 f"Reference readiness identifies another {name}: {package}")
    if receipt["status"] == "ready":
        # A ready receipt is meaningful only when each saved reference layer
        # still points to its original, hash-verified image after relocation.
        for reference in receipt.get("references") or []:
            for layer in (reference.get("layers") or {}).values():
                for asset in (layer.values() if isinstance(layer, dict) and "path" not in layer else [layer]):
                    if isinstance(asset, dict) and "path" in asset and "sha256" in asset:
                        target = _physical(asset["path"], root, logical_root)
                        _require(target.is_file() and sha256_file(target) == asset["sha256"],
                                 f"Ready reference layer changed: {target}")
    return receipt["status"]


def verify_candidate(root: Path, *, logical_root: Path | None = None,
                     allow_historical_source_path: bool = False) -> dict:
    """Check model identity and every native timeline, map and baseline photo."""
    root = Path(root).resolve()
    logical_root = root if logical_root is None else Path(logical_root).resolve()
    _require(root.is_dir() and (root / "validation").is_dir(), "Candidate has no complete validation directory")
    summary = _read(root / "summary.json")
    identity = _model_identity(root)
    _require(summary.get("model_id") == summary.get("campaign_id") == identity[0] == logical_root.name,
             "Candidate identity differs from its folder or summary")
    datasets = summary.get("inventory", {}).get("datasets")
    _require(isinstance(datasets, list) and len(datasets) == 1
             and datasets[0] in EXPECTED_RUN_COUNTS
             and summary.get("regional_training_performed") is True,
             "Expected a completed isolated AC, UM or BM regional model")
    series = datasets[0]
    inspected = inspect_phase_model_bundle(root)
    adapters = set(inspected["adapters"])
    optical_parent = None
    if series == "ffac":
        _require(adapters == EXPECTED_ADAPTERS,
                 "AC candidate must preserve both generic and BTB optical adapters")
    else:
        request = _read(root / "parent_provenance" / "training_request.json")
        settings = request.get("settings") or {}
        candidate_request = _read(root / "training_request.json")
        optical_parent = Path(candidate_request.get("settings", {}).get("parent_model") or "").resolve()
        _require(settings.get("optical_training_profile") == GENERIC_ONLY_PROFILE
                 and settings.get("adapters") == ["generic"]
                 and settings.get("feature_version") == RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION
                 and settings.get("fit_heads") == ["co2_presence", "gas_within_co2"]
                 and settings.get("regularization_grid") == [1e-5, 1e-3, 1e-1]
                 and adapters == {"generic"}
                 and optical_parent.is_dir()
                 and optical_parent.parent.name == "optical_parent"
                 and not optical_parent.is_relative_to(series_path(series, "models").resolve())
                 and Path(request.get("output_root") or "").resolve() == optical_parent
                 and sha256_file(optical_parent / "training_request.json") ==
                     sha256_file(root / "parent_provenance" / "training_request.json"),
                 "UM/BM candidate lacks the exact opt-in generic-only optical profile")
        _require(sha256_file(optical_parent / "summary.json") ==
                 sha256_file(root / "parent_provenance" / "summary.json")
                 and summary.get("regional_fit", {}).get("converged") is True
                 and all(sha256_file(root / "models" / "generic" / f"{head}_signal.json") ==
                         sha256_file(optical_parent / "models" / "generic" /
                                     f"{head}_signal.json")
                         for head in ("co2_presence", "gas_within_co2")),
                 "UM/BM regional model changed its verified optical parent heads")
    profile = load_physics_profile(root)
    _require(profile is not None and profile.get("format") == "baseline-relative-optical-regions-v1",
             "Candidate has no regional physical profile")
    runtime = physics_runtime_identity(profile)
    regional = profile.get("model") or {}
    relative = Path(regional.get("path") or "")
    _require(not relative.is_absolute() and relative != Path("."), "Regional model must be internal")
    regional_file = _inside(root / relative, root, "Regional model")
    _require(regional_file.is_file() and sha256_file(regional_file) == regional.get("sha256"),
             "Regional fitted model is missing or changed")
    manifest_path = root / "validation/postflight_manifest.json"
    binding_path = root / "validation/validation_binding.json"
    binding, manifest = _read(binding_path), _read(manifest_path)
    source_manifest = root / "review_source/postflight_manifest.json"
    _require(source_manifest.is_file(), "Frozen source campaign is missing")
    source_digest = sha256_file(source_manifest)
    native_receipt = None
    if series != "ffac":
        native_receipt = _read(root / "native_source_receipt.json")
        _require(native_receipt.get("format") == "studio-native-source-parity-v1"
                 and native_receipt.get("model_id") == identity[0]
                 and native_receipt.get("series") == series
                 and native_receipt.get("source_manifest_sha256") == source_digest
                 and native_receipt.get("parent_source_manifest_sha256") ==
                     sha256_file(optical_parent / "review_source" / "postflight_manifest.json"),
                 "Native source parity receipt does not bind optical and regional source campaigns")
    for value in (binding, manifest):
        _require((value.get("model_id"), value.get("model_sha256")) == identity
                 and value.get("physics_runtime") == runtime
                 and value.get("source_manifest_sha256") == source_digest,
                 "Validation does not bind the exact fitted model, physics and source")
    _require(binding.get("completed_utc") and binding.get("output_mode", "review") == "review"
             and binding.get("run_count") == EXPECTED_RUN_COUNTS[series],
             "Native review campaign is incomplete")
    prepared_root = _physical(binding.get("combined_root", ""), root, logical_root)
    _inside(prepared_root, root / "validation", "Prepared collection")
    _require(prepared_root.is_dir() and (prepared_root / "validation_binding.json").is_file(),
             "Selected prepared collection is missing")
    prepared_binding = _read(prepared_root / "validation_binding.json")
    _require(prepared_binding.get("run_count") == EXPECTED_RUN_COUNTS[series]
             and (prepared_binding.get("model_id"), prepared_binding.get("model_sha256")) == identity
             and prepared_binding.get("physics_runtime") == runtime
             and _physical(prepared_binding.get("combined_root", ""), root, logical_root) == prepared_root,
             "Selected prepared collection identifies a different model or map package")
    reference_status = {
        "validation": _check_reference_receipt(root / "validation", root, logical_root),
        "prepared": _check_reference_receipt(prepared_root, root, logical_root),
    }
    for receipt in (root / "validation").rglob("phase_reference_readiness.json"):
        _check_reference_receipt(receipt.parent, root, logical_root)
    jobs = _source_jobs(root / "validation")
    source_jobs = _source_jobs(root / "review_source")
    keys = {f'{job["dataset"]}/{job["run"]}' for job in jobs}
    expected_runs = EXPECTED_RUN_COUNTS[series]
    _require(len(jobs) == len(keys) == expected_runs
             and len(source_jobs) == expected_runs
             and all(job["dataset"] == series for job in jobs + source_jobs)
             and keys == set(profile["runs"])
             and keys == {f'{job["dataset"]}/{job["run"]}' for job in source_jobs}
             and {job["run"] for job in jobs} == set(binding.get("selected_candidates") or {}),
             "Native map inventory differs from the model/source binding")
    source_by_run = {job["run"]: job for job in source_jobs}
    if native_receipt is None and not allow_historical_source_path:
        _require(all(job.get("review_output_dir") ==
                     source_by_run[job["run"]].get("review_output_dir") for job in jobs),
                 "Validated native source differs from frozen review campaign")
    if series == "ffbm":
        _require({job["run"].casefold() for job in jobs} == {f"c{i}" for i in range(1, 6)},
                 "BM requires the complete C1-C5 native map inventory")
    if native_receipt is not None:
        _require(native_receipt.get("runs") == expected_runs
                 and set(native_receipt.get("images") or {}) == {job["run"] for job in jobs},
                 "Native source parity receipt omits a registered run")
        # Native materialization may copy the registered review folder to a
        # series-owned prepared_review location. Bind both physical folders to
        # the same prefit JPEG hashes instead of requiring identical path text.
        for run, source_job in source_by_run.items():
            original = _physical(source_job["review_output_dir"], root, logical_root)
            expected = native_receipt["images"][run]
            manifest = original / "native_frames" / run / "manifest.json"
            _require(manifest.is_file() and sha256_file(manifest) == expected["manifest_sha256"],
                     f"Frozen review source manifest changed: {run}")
            baseline = original / "native_references" / run / "pre_injection.jpg"
            _require(baseline.is_file() and sha256_file(baseline) == expected["baseline_sha256"],
                     f"Frozen review baseline changed: {run}")
            for frame_index, digest in enumerate(expected["frame_sha256"]):
                photo = original / "native_frames" / run / f"frame_{frame_index:03d}.jpg"
                _require(photo.is_file() and sha256_file(photo) == digest,
                         f"Frozen review photograph changed: {run}/{frame_index}")
    frames = 0
    baselines = {}
    for job in jobs:
        run, key = job["run"], f'{job["dataset"]}/{job["run"]}'
        entry = profile["runs"][key]
        context = _inside(root / entry["path"], root, "Run context")
        _require(context.is_file() and sha256_file(context) == entry["sha256"],
                 f"Physical context changed: {run}")
        review = _physical(job["review_output_dir"], root, logical_root)
        native = review / "native_frames" / run / "manifest.json"
        _require(native.is_file() and sha256_file(native) == entry["native_manifest_sha256"],
                 f"Native photo manifest differs from physical context: {run}")
        metadata = _read(native)
        image_receipt = native_receipt["images"][run] if native_receipt else None
        if image_receipt is not None:
            _require(image_receipt.get("manifest_sha256") == sha256_file(native),
                     f"Native source manifest changed after optical training: {run}")
        count = metadata.get("frame_count")
        rows = metadata.get("frame_sources") or []
        _require(type(count) is int and count > 0 and len(rows) == count,
                 f"Incomplete native photo timeline: {run}")
        baseline = metadata.get("reference") or {}
        index = baseline.get("timeline_frame_index")
        _require(baseline.get("included_in_timeline") is True and type(index) is int
                 and 0 <= index < count
                 and (baseline.get("source_sha256") == rows[index].get("source_sha256")
                      if baseline.get("source_sha256") else image_receipt is not None),
                 f"Unbound clear-water baseline: {run}")
        photo = review / "native_references" / run / "pre_injection.jpg"
        expected_baseline = (image_receipt["baseline_sha256"] if image_receipt else
                             rows[index].get("native_jpeg_sha256"))
        _require(photo.is_file() and sha256_file(photo) == expected_baseline,
                 f"Clear-water baseline photograph changed: {run}")
        if rows[index].get("native_jpeg_sha256"):
            _require(expected_baseline == rows[index]["native_jpeg_sha256"],
                     f"Baseline native manifest changed: {run}")
        if image_receipt is not None:
            _require(len(image_receipt.get("frame_sha256") or []) == count,
                     f"Native source parity receipt has incomplete frames: {run}")
            for frame_index, row in enumerate(rows):
                native_photo = review / "native_frames" / run / f"frame_{frame_index:03d}.jpg"
                _require(native_photo.is_file()
                         and sha256_file(native_photo) == image_receipt["frame_sha256"][frame_index]
                         and (not row.get("native_jpeg_sha256")
                              or row["native_jpeg_sha256"] ==
                                 image_receipt["frame_sha256"][frame_index]),
                         f"Native review photograph changed: {run}/{frame_index}")
        baselines[run] = expected_baseline
        params = str(job.get("params_sha256") or "")
        _require(len(params) == 64, f"Missing native map policy hash: {run}")
        maps = _physical(job["phase_map_output_dir"], root, logical_root)
        raw_root = job.get("raw_phase_map_output_dir")
        _require(series == "ffac" or raw_root,
                 f"Missing raw optical native maps: {run}")
        raw = _physical(raw_root, root, logical_root) if raw_root else None
        for index in range(count):
            suffix = Path("native_phase_maps") / run / f"A_{params[:16]}" / f"frame_{index:03d}.png"
            target = _inside(maps / suffix, root / "validation", "Native phase map")
            _require(target.is_file() and target.stat().st_size > 8,
                     f"Missing native phase map: {run}/{index}")
            if raw is not None:
                optical = str(job.get("optical_model_sha256") or identity[1])
                optical_map = _inside(raw / "native_phase_maps" / run / f"A_{optical[:16]}"
                                      / f"frame_{index:03d}.png", root / "validation", "Raw optical map")
                _require(optical_map.is_file() and optical_map.stat().st_size > 8,
                         f"Missing raw optical map: {run}/{index}")
        frames += count
    if native_receipt is not None:
        _require(native_receipt.get("frames") == frames,
                 "Native source parity receipt has the wrong frame count")
    return dict(model_id=identity[0], model_sha256=identity[1], series=series,
                adapters=sorted(adapters), physics_runtime=runtime,
                runs=expected_runs, run_names=sorted(job["run"] for job in jobs),
                frames=frames, reference_status=reference_status,
                baselines=baselines, source_manifest_sha256=source_digest)


def _qa_gate(path: Path, checked: dict) -> dict:
    qa = _read(path)
    _require(qa.get("format") == QA_FORMAT and qa.get("status") == "pass"
             and qa.get("model_id") == checked["model_id"]
             and qa.get("model_sha256") == checked["model_sha256"]
             and qa.get("physics_runtime") == checked["physics_runtime"]
             and qa.get("runs_checked") == checked["runs"],
             "QA does not approve this exact model and native review")
    if checked["series"] == "ffac":
        _require(set(qa.get("focus_checked") or []) ==
                 {"ac47-6h", "ac51-24h", "ac51-48h", "ac51-71h"},
                 "QA does not approve the AC47/AC51 focus review")
    else:
        _require(qa.get("source_manifest_sha256") == checked["source_manifest_sha256"]
                 and qa.get("run_names_checked") == checked["run_names"]
                 and all(qa.get(name) is True for name in (
                     "native_image_parity", "training_coverage_passed",
                     "optical_convergence_passed", "regional_convergence_passed",
                     "visual_review_passed")),
                 "QA does not bind complete UM/BM source, coverage, convergence and visual review")
    report = Path(qa.get("comparison_report") or "").resolve()
    _require(report.is_file() and sha256_file(report) == qa.get("comparison_report_sha256"),
             "Matched-frame QA comparison changed")
    return qa


def _validation_only_qa_gate(source: Path, decision_path: Path, checked: dict) -> tuple[dict, dict[str, str]]:
    """Bind the recorded failed decision and every copied QA asset to this candidate."""
    job = source.parent.parent
    _require(source.parent.name == "regional_candidate"
             and decision_path == job / "candidate_qa_decision.json",
             "Validation-only QA must be the decision in this candidate's results job")
    _require(not (source / "publication_receipt.json").exists()
             and not (source / VALIDATION_ONLY_QA).exists(),
             "Validation-only source must be the unpublished candidate")
    qa = _read(decision_path)
    _require(qa.get("format") == FAILED_QA_FORMAT
             and qa.get("status") == "fail"
             and qa.get("studio_publication_allowed") is False
             and qa.get("scientific_acceptance") is False
             and Path(qa.get("candidate_model") or "").resolve() == source
             and qa.get("candidate_model_id") == checked["model_id"]
             and qa.get("candidate_model_sha256") == checked["model_sha256"]
             and qa.get("runs_checked") == checked["runs"]
             and qa.get("provenance_integrity") == "passed"
             and qa.get("visual_qa") == "failed_ac51_late_broad_speckled_gas",
             "Failed QA decision does not identify this exact model and failure")
    result = _read(job / "result.json")
    parent = Path(qa.get("parent_model") or "").resolve()
    _require(Path(result.get("candidate_model") or "").resolve() == source
             and Path(result.get("original_parent") or "").resolve() == parent
             and result.get("review_candidate_only") is True
             and result.get("studio_published") is False
             and parent.is_dir() and parent.parent == series_path("ffac", "models").resolve()
             and parent != source,
             "Failed QA decision does not bind the original parent and detached result")
    provenance_path = job / "provenance_qa_receipt.json"
    provenance = _read(provenance_path)
    _require(provenance.get("format") == "ffac-regional-candidate-provenance-qa-v1"
             and provenance.get("status") == "integrity_passed_publication_qa_pending"
             and Path(provenance.get("candidate_model") or "").resolve() == source
             and provenance.get("model_id") == checked["model_id"]
             and provenance.get("model_sha256") == checked["model_sha256"]
             and provenance.get("candidate_summary_sha256") == sha256_file(source / "summary.json")
             and provenance.get("physics_runtime") == checked["physics_runtime"]
             and provenance.get("source_manifest_sha256") == checked["source_manifest_sha256"]
             and provenance.get("runs") == checked["runs"]
             and provenance.get("frames") == checked["frames"]
             and provenance.get("per_run_source_and_receipts_verified") == checked["runs"]
             and provenance.get("studio_published") is False
             and provenance.get("publication_approved") is False,
             "Provenance QA does not bind the exact 43-run failed candidate")
    for name, digest in (provenance.get("parent_hashes_verified_unchanged") or {}).items():
        _require(name in {"summary.json", "training_request.json", "regional_model.json"}
                 and sha256_file(parent / name) == digest,
                 f"Original parent changed since provenance QA: {name}")
    _require(len(provenance.get("parent_hashes_verified_unchanged") or {}) == 3,
             "Provenance QA lacks original parent hashes")
    reports = qa.get("reports") or []
    _require({Path(row.get("path") or "").resolve().relative_to(job).as_posix()
              for row in reports} == FAILED_QA_REPORTS,
             "Failed QA report inventory is incomplete or outside this job")
    for row in reports:
        report = Path(row["path"]).resolve()
        _require(report.is_file() and sha256_file(report) == row.get("sha256"),
                 f"Failed QA report changed: {report}")
    evidence: dict[str, str] = {}
    for name in QA_ROOT_DOCUMENTS:
        path = job / name
        _require(path.is_file() and not path.is_symlink(),
                 f"Missing or linked failed QA record: {path}")
        evidence[name] = sha256_file(path)
    for name in QA_DIRECTORIES:
        folder = job / name
        _require(folder.is_dir() and not folder.is_symlink()
                 and not (hasattr(folder, "is_junction") and folder.is_junction()),
                 f"Missing or linked failed QA evidence: {folder}")
        for path in folder.rglob("*"):
            _require(not path.is_symlink()
                     and not (hasattr(path, "is_junction") and path.is_junction()),
                     f"Failed QA evidence contains a filesystem link: {path}")
            if path.is_file():
                evidence[path.relative_to(job).as_posix()] = sha256_file(path)
    _require(FAILED_QA_REPORTS <= set(evidence), "Failed QA reports were not copied")
    return qa, evidence


def _v5_validation_only_qa_gate(source: Path, decision_path: Path,
                                checked: dict) -> tuple[dict, dict[str, str]]:
    """Permit human Studio review of the *failed* AC v5 maps, never deployment.

    The v5 job retained an older historical source campaign path while preparing
    its maps from current native photographs. The complete 864-frame comparison
    binds the current photographs and maps to the parent; an equal source path is
    therefore not required for this narrowly identified failed-QA candidate.
    """
    job = source.parent.parent
    _require(source.parent.name == "regional_candidate"
             and decision_path == job / "qa_decision_unpublished.json"
             and checked["series"] == "ffac", "Expected the AC v5 detached QA decision")
    _require(not (source / "publication_receipt.json").exists()
             and not (source / VALIDATION_ONLY_QA).exists(),
             "Validation-only source must remain the unpublished candidate")
    qa = _read(decision_path)
    result_path = job / "result.json"
    quant_path = job / "qa_full_native_20260926" / "qa_comparison.json"
    visual_path = job / "visual_qa_subagent" / "assessment.md"
    _require(qa.get("format") == FAILED_V5_QA_FORMAT
             and qa.get("status") == "fail_unpublished"
             and qa.get("scientific_acceptance") is False
             and qa.get("studio_publication") is False
             and qa.get("candidate_model_id") == checked["model_id"]
             and qa.get("source_result_sha256") == sha256_file(result_path)
             and qa.get("quant_report_sha256") == sha256_file(quant_path)
             and qa.get("visual_report_sha256") == sha256_file(visual_path)
             and len(qa.get("blocking_findings") or []) >= 2,
             "AC v5 failed-QA decision or its evidence changed")
    result = _read(result_path)
    parent = Path(result.get("original_parent") or "").resolve()
    _require(Path(result.get("candidate_model") or "").resolve() == source
             and result.get("review_candidate_only") is True
             and result.get("studio_published") is False
             and parent.is_dir()
             and parent.parent == series_path("ffac", "models").resolve()
             and parent != source,
             "AC v5 result no longer binds the detached candidate and published parent")
    quant = _read(quant_path)
    area = quant.get("area") or {}
    _require(quant.get("format") == "ffac-optical-regional-v5-saved-map-qa-v1"
             and quant.get("status") == "incomplete"
             and quant.get("independent_validation") is False
             and quant.get("training_seen") is True
             and Path(quant.get("candidate") or "").name == checked["model_id"]
             and quant.get("provenance", {}).get("candidate_summary_sha256") ==
                 sha256_file(source / "summary.json")
             and quant.get("provenance", {}).get("parent_summary_sha256") ==
                 sha256_file(parent / "summary.json")
             and quant.get("candidate_validation_binding_sha256") ==
                 sha256_file(source / "validation" / "validation_binding.json")
             and quant.get("candidate_postflight_manifest_sha256") ==
                 sha256_file(source / "validation" / "postflight_manifest.json")
             and area.get("matched_runs") == checked["runs"] == 43
             and area.get("matched_frames") == area.get("existing_map_pairs") ==
                 checked["frames"] == 864,
             "AC v5 comparison does not bind all native frames and this exact model")
    run_rows = area.get("runs") or {}
    _require(set(run_rows) == set(checked["run_names"]),
             "AC v5 comparison omits a prepared experiment")
    compared = 0
    for run, record in run_rows.items():
        frames = record.get("frames") or []
        _require(record.get("expected_frames") == record.get("matched_frames") == len(frames)
                 and record.get("parent_native_manifest_sha256") ==
                     record.get("candidate_native_manifest_sha256"),
                 f"AC v5 source parity failed for {run}")
        for index, frame in enumerate(frames):
            map_path = Path(frame.get("candidate_map") or "")
            _require(frame.get("run") == run and frame.get("native_index_0based") == index
                     and len(str(frame.get("source_sha256") or "")) == 64
                     and len(str(frame.get("native_jpeg_sha256") or "")) == 64
                     and len(str(frame.get("corrected_array_sha256") or "")) == 64
                     and map_path.is_file()
                     and sha256_file(map_path) == frame.get("candidate_map_sha256"),
                     f"AC v5 native map or source receipt changed: {run}/{index}")
            compared += 1
    _require(compared == checked["frames"], "AC v5 comparison is incomplete")
    evidence: dict[str, str] = {}
    for name in ("qa_decision_unpublished.json", "qa_decision_unpublished.md",
                 "result.json", "qa_gas_loss_stage_diagnosis.md"):
        path = job / name
        _require(path.is_file() and not path.is_symlink(), f"Missing QA record: {path}")
        evidence[name] = sha256_file(path)
    for name in ("qa_full_native_20260926", "visual_qa_subagent"):
        folder = job / name
        _require(folder.is_dir() and not folder.is_symlink()
                 and not (hasattr(folder, "is_junction") and folder.is_junction()),
                 f"Missing QA evidence: {folder}")
        for path in folder.rglob("*"):
            _require(not path.is_symlink()
                     and not (hasattr(path, "is_junction") and path.is_junction()),
                     f"QA evidence contains a filesystem link: {path}")
            if path.is_file():
                evidence[path.relative_to(job).as_posix()] = sha256_file(path)
    _require({"qa_full_native_20260926/qa_comparison.json",
              "qa_full_native_20260926/qa_coverage_addendum.md",
              "visual_qa_subagent/assessment.md"} <= set(evidence),
             "AC v5 failed-QA report inventory is incomplete")
    return qa, evidence


def _um_validation_only_qa_gate(source: Path, decision_path: Path,
                                checked: dict) -> tuple[dict, dict[str, str]]:
    """Allow this failed UM candidate in Studio solely for human validation."""
    job = source.parent.parent
    _require(source.parent.name == "regional_candidate"
             and decision_path == job / "qa_decision_unpublished.json"
             and checked["series"] == "ffum"
             and checked["model_id"] == FAILED_UM_MODEL_ID
             and checked["model_sha256"] == FAILED_UM_MODEL_SHA256
             and checked["runs"] == 13 and checked["frames"] == 261,
             "Expected only the exact failed FF_UM fast candidate")
    _require(not (source / "publication_receipt.json").exists()
             and not (source / VALIDATION_ONLY_QA).exists(),
             "Validation-only source must remain the unpublished candidate")
    decision = _read(decision_path)
    result_path = job / "result.json"
    preflight_path = job / "preflight.json"
    result, preflight = _read(result_path), _read(preflight_path)
    parent = Path(result.get("original_parent") or "").resolve()
    required_parent = {
        "summary.json": "parent_summary_sha256",
        "training_request.json": "parent_request_sha256",
        "regional_model.json": "parent_regional_model_sha256",
        "review_source/postflight_manifest.json": "parent_source_manifest_sha256",
    }
    _require(parent.is_dir() and parent.parent == series_path("ffum", "models").resolve()
             and all(sha256_file(parent / name) == preflight.get(key)
                     for name, key in required_parent.items()),
             "Original UM parent differs from the frozen preflight")
    _require(Path(result.get("candidate_model") or "").resolve() == source
             and result.get("review_candidate_only") is True
             and result.get("studio_published") is False
             and result.get("qa_pending") is True
             and result.get("original_parent_unchanged") is True
             and result.get("series") == "ffum",
             "UM result does not bind this detached failed candidate")
    report = Path(decision.get("quant_report_path") or "").resolve()
    visual = Path(decision.get("visual_report_path") or "").resolve()
    _require(report.is_relative_to(job) and visual.is_relative_to(job)
             and report.is_file() and visual.is_file()
             and sha256_file(report) == decision.get("quant_report_sha256")
             and sha256_file(visual) == decision.get("visual_report_sha256"),
             "UM quantitative or visual failed-QA evidence changed")
    _require(decision.get("format") == FAILED_UM_QA_FORMAT
             and decision.get("status") == "fail_unpublished"
             and decision.get("scientific_acceptance") is False
             and decision.get("studio_publication") is False
             and Path(decision.get("candidate_model") or "").resolve() == source
             and decision.get("candidate_model_id") == checked["model_id"]
             and decision.get("candidate_model_sha256") == checked["model_sha256"]
             and decision.get("candidate_summary_sha256") == sha256_file(source / "summary.json")
             and decision.get("source_manifest_sha256") == checked["source_manifest_sha256"]
             and decision.get("native_source_receipt_sha256") ==
                 sha256_file(source / "native_source_receipt.json")
             and decision.get("physics_runtime") == checked["physics_runtime"]
             and decision.get("runs_checked") == checked["runs"]
             and decision.get("frames_checked") == checked["frames"]
             and Path(decision.get("parent_model") or "").resolve() == parent
             and decision.get("parent_summary_sha256") == sha256_file(parent / "summary.json")
             and decision.get("result_sha256") == sha256_file(result_path)
             and {"run19_late_sidewall_false_aqueous",
                  "run20_late_sidewall_false_aqueous",
                  "run16_late_aqueous_positive_lost_all",
                  "run11_gas_negative_region_increase"} <=
                 set(decision.get("blocking_findings") or []),
             "UM failed-QA decision does not bind the exact model and sidewall failure")
    quant = _read(report)
    native_receipt = _read(source / "native_source_receipt.json")
    area = quant.get("area") or {}
    labels = quant.get("labels") or {}
    label_summary = labels.get("summary") or {}
    _require(quant.get("format") == FAILED_UM_QUANT_FORMAT
             and quant.get("status") == "comparison_complete_with_unaligned_reviews"
             and quant.get("series") == "ffum"
             and Path(quant.get("parent") or "").resolve() == parent
             and Path(quant.get("candidate") or "").resolve() == source
             and Path(quant.get("job") or "").resolve() == job
             and quant.get("job_result_sha256") == sha256_file(result_path)
             and quant.get("parent_validation_binding_sha256") ==
                 sha256_file(parent / "validation" / "validation_binding.json")
             and quant.get("candidate_validation_binding_sha256") ==
                 sha256_file(source / "validation" / "validation_binding.json")
             and quant.get("parent_postflight_manifest_sha256") ==
                 sha256_file(parent / "validation" / "postflight_manifest.json")
             and quant.get("candidate_postflight_manifest_sha256") ==
                 sha256_file(source / "validation" / "postflight_manifest.json")
             and quant.get("independent_validation") is False
             and quant.get("training_seen") is True
             and quant.get("publication_qa_pass") is False
             and area.get("matched_runs") == checked["runs"]
             and area.get("matched_frames") == checked["frames"]
             and not area.get("issues")
             and Path(labels.get("frozen_request") or "").resolve() ==
                 (source / "training_request.json")
             and labels.get("frozen_request_sha256") ==
                 sha256_file(source / "training_request.json")
             and label_summary.get("frozen_effective_regions") == 2686
             and label_summary.get("aligned_regions") == 719
             and label_summary.get("unaligned_regions") == 1967
             and label_summary.get("gas_positive_lost_all") == 0
             and label_summary.get("aqueous_positive_lost_all") == 3
             and label_summary.get("gas_negative_more_evidence") == 28,
             "UM quantitative failed-QA report does not bind this candidate, parent and frozen labels")
    run_rows = area.get("runs") or {}
    _require(set(run_rows) == set(checked["run_names"]),
             "UM quantitative failed-QA report omits a prepared experiment")
    compared = 0
    for run, row in run_rows.items():
        frames = row.get("frames") or []
        receipt_images = (native_receipt.get("images") or {}).get(run) or {}
        native_hashes = receipt_images.get("frame_sha256") or []
        _require(row.get("expected_frames") == row.get("matched_frames") ==
                 len(frames) == len(native_hashes),
                 f"UM native timeline differs from the frozen receipt: {run}")
        for index, frame in enumerate(frames):
            parent_map = Path(frame.get("parent_map") or "").resolve()
            candidate_map = Path(frame.get("candidate_map") or "").resolve()
            _require(frame.get("run") == run
                     and frame.get("parent_index_0based") == index
                     and frame.get("candidate_index_0based") == index
                     and frame.get("studio_frame_1based") == index + 1
                     and len(str(frame.get("source_sha256") or "")) == 64
                     and frame.get("native_jpeg_sha256") == native_hashes[index]
                     and parent_map.is_relative_to(parent / "validation")
                     and candidate_map.is_relative_to(source / "validation")
                     and parent_map.is_file() and candidate_map.is_file()
                     and frame.get("parent_map_sha256") == sha256_file(parent_map)
                     and frame.get("candidate_map_sha256") == sha256_file(candidate_map),
                     f"UM native source or mapped frame changed: {run}/{index}")
            compared += 1
    _require(compared == checked["frames"], "UM quantitative comparison is incomplete")
    observations = {row.get("region_id"): row for row in
                    labels.get("observations") or []}
    regressions = {
        "run16:manual:51756df9f9334109858e4e5a58586de8":
            ("run16", 20, "aqueous", "aqueous_evidence", 41209, 0),
        "run16:manual:84f854099eb148e39fcc61434b8d00fc":
            ("run16", 21, "aqueous", "aqueous_evidence", 21731, 0),
        "run9:manual:80cc10afa24e4aba8e96cca478486a32":
            ("run9", 15, "mixed_gas_aqueous", "aqueous_evidence", 13, 0),
        "run11:manual:5d92e570051342c8b04b7b3da9e245aa":
            ("run11", 18, None, "gas_evidence", 5133, 14346),
    }
    for region_id, (run, frame, phase_class, measure, before, after) in regressions.items():
        row = observations.get(region_id) or {}
        _require(row.get("run") == run
                 and row.get("studio_frame_1based") == frame
                 and (phase_class is None or row.get("phase_class") == phase_class)
                 and row.get("parent", {}).get(measure) == before
                 and row.get("candidate", {}).get(measure) == after
                 and (row.get("aqueous_role") == "positive"
                      if measure == "aqueous_evidence" else
                      row.get("gas_role") == "negative"),
                 f"UM aligned regional regression changed: {region_id}")
    _require(decision.get("material_regressions") == {
                 region_id: {"run": run, "studio_frame_1based": frame,
                             "measure": measure, "parent": before, "candidate": after}
                 for region_id, (run, frame, _, measure, before, after)
                 in regressions.items()},
             "UM decision omits source-aligned aqueous loss or gas increase")
    evidence = {}
    mandatory = {"result.json",
                 report.relative_to(job).as_posix(), visual.relative_to(job).as_posix()}
    declared = decision.get("evidence_files") or {}
    _require(isinstance(declared, dict) and mandatory <= set(declared),
             "UM failed-QA evidence inventory omits decision, result or reports")
    for name, digest in declared.items():
        relative = Path(name)
        original = job / relative
        target = original.resolve()
        _require(not relative.is_absolute() and target.is_relative_to(job)
                 and target.is_file() and not original.is_symlink()
                 and sha256_file(target) == digest,
                 f"UM failed-QA evidence changed or escaped the job: {name}")
        evidence[relative.as_posix()] = digest
    evidence["qa_decision_unpublished.json"] = sha256_file(decision_path)
    return decision, evidence


def _bm_validation_only_qa_gate(source: Path, decision_path: Path,
                                checked: dict) -> tuple[dict, dict[str, str]]:
    """Permit only the documented failed BM model for Studio validation."""
    job = source.parent.parent
    _require(source.parent.name == "regional_candidate"
             and job == series_path("ffbm", "results").resolve() / FAILED_BM_JOB
             and decision_path == job / "qa_decision_unpublished.json"
             and checked["series"] == "ffbm"
             and checked["model_id"] == FAILED_BM_MODEL_ID
             and checked["model_sha256"] == FAILED_BM_MODEL_SHA256
             and checked["runs"] == 5 and checked["frames"] == 134
             and checked["run_names"] == [f"c{number}" for number in range(1, 6)],
             "Expected only the exact failed FF_BM fast candidate and job")
    _require(not (source / "publication_receipt.json").exists()
             and not (source / VALIDATION_ONLY_QA).exists(),
             "Validation-only source must remain the unpublished candidate")
    decision = _read(decision_path)
    result_path = job / "result.json"
    preflight_path = job / "preflight.json"
    state_path = job / "candidate_qa_state.json"
    visual_path = job / "native_visual_qa.md"
    report_path = job / "quantitative_qa" / "qa_comparison.json"
    index_path = job / "quantitative_qa" / "qa_comparison_index.json"
    assessment_path = job / "quantitative_qa" / "qa_assessment.md"
    result, preflight, state = (_read(path) for path in
                                (result_path, preflight_path, state_path))
    parent = Path(result.get("original_parent") or "").resolve()
    required_parent = {
        "summary.json": "parent_summary_sha256",
        "training_request.json": "parent_request_sha256",
        "regional_model.json": "parent_regional_model_sha256",
        "review_source/postflight_manifest.json": "parent_source_manifest_sha256",
    }
    _require(parent.is_dir() and parent.parent == series_path("ffbm", "models").resolve()
             and all(sha256_file(parent / name) == preflight.get(key)
                     for name, key in required_parent.items()),
             "Original BM parent differs from the frozen preflight")
    _require(Path(result.get("candidate_model") or "").resolve() == source
             and result.get("review_candidate_only") is True
             and result.get("studio_published") is False
             and result.get("qa_pending") is True
             and result.get("original_parent_unchanged") is True
             and result.get("series") == "ffbm"
             and Path(state.get("candidate_model") or "").resolve() == source
             and state.get("review_candidate_only") is True
             and state.get("studio_published") is False,
             "BM result or QA state does not bind this detached failed candidate")
    _require(report_path.is_file() and visual_path.is_file() and index_path.is_file()
             and assessment_path.is_file()
             and sha256_file(report_path) == FAILED_BM_QUANT_SHA256
             and sha256_file(visual_path) == FAILED_BM_VISUAL_SHA256
             and sha256_file(assessment_path) == FAILED_BM_ASSESSMENT_SHA256,
             "BM quantitative or visual failed-QA evidence changed")
    _require(decision.get("format") == FAILED_BM_QA_FORMAT
             and decision.get("status") == "fail_unpublished"
             and decision.get("scientific_acceptance") is False
             and decision.get("studio_publication") is False
             and decision.get("deployment_allowed") is False
             and Path(decision.get("candidate_model") or "").resolve() == source
             and decision.get("candidate_model_id") == checked["model_id"]
             and decision.get("candidate_model_sha256") == checked["model_sha256"]
             and decision.get("candidate_summary_sha256") == sha256_file(source / "summary.json")
             and decision.get("source_manifest_sha256") == checked["source_manifest_sha256"]
             and decision.get("native_source_receipt_sha256") ==
                 sha256_file(source / "native_source_receipt.json")
             and decision.get("physics_runtime") == checked["physics_runtime"]
             and decision.get("runs_checked") == 5
             and decision.get("frames_checked") == 134
             and decision.get("run_names_checked") == checked["run_names"]
             and Path(decision.get("parent_model") or "").resolve() == parent
             and decision.get("parent_summary_sha256") == sha256_file(parent / "summary.json")
             and decision.get("result_sha256") == sha256_file(result_path)
             and decision.get("preflight_sha256") == sha256_file(preflight_path)
             and decision.get("candidate_qa_state_sha256") == sha256_file(state_path)
             and Path(decision.get("quant_report_path") or "").resolve() == report_path
             and decision.get("quant_report_sha256") == FAILED_BM_QUANT_SHA256
             and Path(decision.get("visual_report_path") or "").resolve() == visual_path
             and decision.get("visual_report_sha256") == FAILED_BM_VISUAL_SHA256
             and {"c1_native_010_false_aqueous_bottom_sides",
                  "c1_native_024_plume_dropout"} <= set(decision.get("blocking_findings") or []),
             "BM failed-QA decision omits the exact model, evidence or C1 failures")
    index, quant = _read(index_path), _read(report_path)
    _require(index.get("format") == "fluidflower-series-comparison-index-v1"
             and Path(index.get("report") or "").resolve() == report_path
             and index.get("report_sha256") == FAILED_BM_QUANT_SHA256
             and index.get("publication_qa_pass") is False,
             "BM quantitative index does not bind the exact failed report")
    area, labels = quant.get("area") or {}, quant.get("labels") or {}
    summary = labels.get("summary") or {}
    _require(quant.get("format") == FAILED_UM_QUANT_FORMAT
             and quant.get("status") == "comparison_complete"
             and quant.get("series") == "ffbm"
             and Path(quant.get("parent") or "").resolve() == parent
             and Path(quant.get("candidate") or "").resolve() == source
             and Path(quant.get("job") or "").resolve() == job
             and quant.get("job_result_sha256") == sha256_file(result_path)
             and quant.get("parent_validation_binding_sha256") ==
                 sha256_file(parent / "validation/validation_binding.json")
             and quant.get("candidate_validation_binding_sha256") ==
                 sha256_file(source / "validation/validation_binding.json")
             and quant.get("parent_postflight_manifest_sha256") ==
                 sha256_file(parent / "validation/postflight_manifest.json")
             and quant.get("candidate_postflight_manifest_sha256") ==
                 sha256_file(source / "validation/postflight_manifest.json")
             and quant.get("independent_validation") is False
             and quant.get("training_seen") is True
             and quant.get("publication_qa_pass") is False
             and area.get("matched_runs") == 5 and area.get("matched_frames") == 134
             and not area.get("issues")
             and Path(labels.get("frozen_request") or "").resolve() ==
                 source / "training_request.json"
             and labels.get("frozen_request_sha256") ==
                 sha256_file(source / "training_request.json")
             and summary.get("frozen_effective_regions") == 1000
             and summary.get("aligned_regions") == 1000
             and summary.get("unaligned_regions") == 0
             and summary.get("gas_positive_lost_all") == 6
             and summary.get("aqueous_positive_lost_all") == 41,
             "BM quantitative failed-QA report does not bind the exact model and frozen labels")
    native = _read(source / "native_source_receipt.json")
    run_rows = area.get("runs") or {}
    _require(set(run_rows) == set(checked["run_names"]),
             "BM quantitative report omits a prepared experiment")
    source_jobs = {row["run"]: row for row in _source_jobs(source / "review_source")}
    compared = 0
    for run, row in run_rows.items():
        frames = row.get("frames") or []
        native_hashes = native["images"][run]["frame_sha256"]
        source_manifest = _read(Path(source_jobs[run]["review_output_dir"]) /
                                "native_frames" / run / "manifest.json")
        source_frames = source_manifest.get("frame_sources") or []
        _require(row.get("expected_frames") == row.get("matched_frames") ==
                 len(frames) == len(native_hashes) == len(source_frames),
                 f"BM native timeline differs from the frozen receipt: {run}")
        for index_0based, frame in enumerate(frames):
            parent_map = Path(frame.get("parent_map") or "").resolve()
            candidate_map = Path(frame.get("candidate_map") or "").resolve()
            _require(frame.get("run") == run
                     and frame.get("parent_index_0based") == index_0based
                     and frame.get("candidate_index_0based") == index_0based
                     and frame.get("studio_frame_1based") == index_0based + 1
                     and frame.get("source_sha256") ==
                         source_frames[index_0based].get("source_sha256")
                     and frame.get("native_jpeg_sha256") == native_hashes[index_0based]
                     and parent_map.name == f"frame_{index_0based:03d}.png"
                     and candidate_map.name == f"frame_{index_0based:03d}.png"
                     and parent_map.is_relative_to(parent / "validation")
                     and candidate_map.is_relative_to(source / "validation")
                     and parent_map.is_file() and candidate_map.is_file()
                     and frame.get("parent_map_sha256") == sha256_file(parent_map)
                     and frame.get("candidate_map_sha256") == sha256_file(candidate_map),
                     f"BM native source or mapped frame changed: {run}/{index_0based}")
            compared += 1
    _require(compared == 134, "BM quantitative comparison is incomplete")
    c1 = run_rows["c1"]["frames"]
    _require(c1[10]["candidate"]["counts"].get("aqueous") == 1544323
             and c1[10]["candidate"]["border"].get("left_2pct_aqueous") == 178920
             and c1[10]["candidate"]["border"].get("right_2pct_aqueous") == 126473
             and c1[10]["candidate"]["border"].get("bottom_2pct_aqueous") == 237408
             and [c1[i]["candidate"]["counts"].get("aqueous") for i in (23, 24, 25)] ==
                 [11136898, 3508796, 14512570]
             and c1[24]["parent"]["counts"].get("aqueous") == 17049054,
             "BM C1 false aqueous or plume-dropout evidence changed")
    observations = labels.get("observations") or []
    _require(len(observations) == 1000
             and len({row.get("region_id") for row in observations}) == 1000,
             "BM source-aligned label observations are incomplete")
    evidence = {}
    mandatory = {"result.json", "preflight.json", "candidate_qa_state.json",
                 "native_visual_qa.md", "quantitative_qa/qa_comparison.json",
                 "quantitative_qa/qa_comparison_index.json",
                 "quantitative_qa/qa_comparison.md",
                 "quantitative_qa/qa_assessment.md"}
    declared = decision.get("evidence_files") or {}
    _require(isinstance(declared, dict) and mandatory <= set(declared),
             "BM failed-QA evidence inventory omits a required report")
    for name, digest in declared.items():
        relative = Path(name)
        original = job / relative
        target = original.resolve()
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() == name and target.is_relative_to(job)
                 and target.is_file() and not original.is_symlink()
                 and sha256_file(target) == digest,
                 f"BM failed-QA evidence changed or escaped the job: {name}")
        evidence[relative.as_posix()] = digest
    for preview in quant.get("previews") or []:
        path = Path(preview.get("path") or "").resolve()
        relative = path.relative_to(job) if path.is_relative_to(job) else None
        _require(relative is not None and relative.as_posix() in evidence
                 and preview.get("sha256") == evidence[relative.as_posix()],
                 "BM quantitative preview changed or is missing from QA evidence")
    evidence["qa_decision_unpublished.json"] = sha256_file(decision_path)
    return decision, evidence


def _check_validation_only_evidence(root: Path, expected: dict[str, str]) -> None:
    evidence_root = root / VALIDATION_ONLY_QA
    actual = {path.relative_to(evidence_root).as_posix(): sha256_file(path)
              for path in evidence_root.rglob("*") if path.is_file()}
    _require(actual == expected, "Published failed-QA evidence differs from the source job")


def _frozen(relative: Path) -> bool:
    return relative.as_posix() in FROZEN_ROOT_JSON or relative.parts[0] in FROZEN_TOP_LEVEL


def _rebind_references(staging: Path, destination: Path) -> None:
    for path in (staging / "validation").rglob("phase_reference_readiness.json"):
        receipt = _read(path)
        _require(receipt.get("format") == "phase-reference-readiness-v1"
                 and receipt.get("status") in {"ready", "deferred"},
                 f"Cannot publish incomplete references: {path}")
        for key, name in (("binding", "validation_binding.json"),
                          ("campaign", "postflight_manifest.json")):
            receipt[key] = {"path": str(destination / path.parent.relative_to(staging) / name),
                            "sha256": sha256_file(path.parent / name)}
        _write(path, receipt)


def publish(source: Path, destination: Path, staging_root: Path, qa_report: Path,
            *, validation_only: bool = False) -> dict:
    """Opt-in atomic publication; refuses overwrite and leaves source unchanged."""
    source, destination, staging_root, qa_report = (Path(path).resolve() for path in
        (source, destination, staging_root, qa_report))
    qa_format = _read(qa_report).get("format") if validation_only else None
    failed_v5 = validation_only and qa_format == FAILED_V5_QA_FORMAT
    failed_um = validation_only and qa_format == FAILED_UM_QA_FORMAT
    failed_bm = validation_only and qa_format == FAILED_BM_QA_FORMAT
    checked = verify_candidate(source, allow_historical_source_path=failed_v5)
    series = checked["series"]
    library = series_path(series, "models").resolve()
    _require(destination.parent == library and destination.name == source.name,
             "Destination must be this exact model ID in the selected Studio library")
    _require(source.is_relative_to(series_path(series, "results").resolve())
             and not source.is_relative_to(library), "Source must be a detached results candidate")
    _require(staging_root != library and not staging_root.is_relative_to(library)
             and not staging_root.is_relative_to(source) and not source.is_relative_to(staging_root),
             "Staging must be outside both source and Studio model library")
    _require(staging_root.anchor == destination.anchor, "Staging and final model must use the same volume")
    if destination.exists():
        raise FileExistsError(destination)
    _require(validation_only or not (series == "ffbm" and
             checked["model_id"] == FAILED_BM_MODEL_ID),
             "The documented failed FF_BM candidate cannot receive ordinary QA-pass publication")
    failed_qa, qa_evidence = (None, {})
    if validation_only:
        _require(series == "ffac" or (series == "ffum" and failed_um)
                 or (series == "ffbm" and failed_bm),
                 "No validation-only failed-QA policy exists for this series and receipt")
        if failed_um:
            failed_qa, qa_evidence = _um_validation_only_qa_gate(source, qa_report, checked)
        elif failed_bm:
            failed_qa, qa_evidence = _bm_validation_only_qa_gate(source, qa_report, checked)
        elif failed_v5:
            failed_qa, qa_evidence = _v5_validation_only_qa_gate(source, qa_report, checked)
        else:
            failed_qa, qa_evidence = _validation_only_qa_gate(source, qa_report, checked)
    else:
        _qa_gate(qa_report, checked)
    source_paths = sorted(path for path in source.rglob("*") if path.is_file())
    _require(source_paths and not any(path.is_symlink()
             or (hasattr(path, "is_junction") and path.is_junction())
             for path in source.rglob("*")),
             "Candidate is empty or contains filesystem links")
    hashes = {path.relative_to(source).as_posix(): sha256_file(path) for path in source_paths}
    stage_parent = staging_root / f"regional-review-publication-{uuid.uuid4().hex}"
    staging = stage_parent / destination.name
    staging.mkdir(parents=True, exist_ok=False)
    try:
        relocated = []
        for path in source_paths:
            relative = path.relative_to(source)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            changed = False
            if path.suffix.lower() == ".json" and not _frozen(relative):
                value = _read(path)
                result = relocate(value, source, destination)
                if result != value:
                    _write(target, result)
                    relocated.append(relative.as_posix())
                    changed = True
                else:
                    shutil.copy2(path, target)
            else:
                shutil.copy2(path, target)
            if not changed:
                _require(sha256_file(target) == hashes[relative.as_posix()],
                         f"Staged file differs from source: {relative}")
        _rebind_references(staging, destination)
        for relative, digest in qa_evidence.items():
            original = source.parent.parent / relative
            target = staging / VALIDATION_ONLY_QA / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            _require(sha256_file(target) == digest,
                     f"Staged failed-QA evidence changed: {relative}")
        staged = verify_candidate(staging, logical_root=destination,
                                  allow_historical_source_path=failed_v5)
        _require(staged == checked, "Staging changed model identity, native maps or baseline references")
        if validation_only:
            _check_validation_only_evidence(staging, qa_evidence)
        for name in ("summary.json", *(
            f"models/{adapter}/{head}_signal.json"
            for adapter in checked["adapters"]
            for head in ("co2_presence", "gas_within_co2")
        )):
            _require(sha256_file(staging / name) == hashes[name], f"Model identity bytes changed: {name}")
        for relative, digest in hashes.items():
            _require(sha256_file(source / relative) == digest, f"Source changed during staging: {relative}")
        for relative, digest in qa_evidence.items():
            _require(sha256_file(source.parent.parent / relative) == digest,
                     f"Failed-QA source changed during staging: {relative}")
        receipt = dict(format=FORMAT, created_utc=utc_now(), source=str(source), destination=str(destination),
                       qa_report=str(qa_report), qa_report_sha256=sha256_file(qa_report),
                       model_id=checked["model_id"], model_sha256=checked["model_sha256"],
                       physics_runtime=checked["physics_runtime"], runs=checked["runs"], frames=checked["frames"],
                       source_files=hashes, relocated_json=relocated, original_files_unchanged=True,
                       training_during_publication=False, automatic_activation=False,
                       human_review_required=True, scientific_acceptance=False)
        if validation_only:
            warning = ("QA FAILED: FF_UM RUN19/RUN20 false aqueous sidewall lines, "
                       "RUN16 late aqueous loss and RUN11 gas-negative increase. "
                       "Studio validation only; scientific acceptance "
                       "and deployment are not approved." if failed_um else
                       "QA FAILED: FF_BM C1 frame 010 false aqueous at the bottom and sides; "
                       "frame 024 visible plume dropout. Studio validation only; scientific "
                       "acceptance and deployment are not approved." if failed_bm else
                       "QA FAILED: AC51 false aqueous sidewall strips and lost gas-positive "
                       "regions. Studio validation only; scientific acceptance and deployment "
                       "are not approved." if failed_v5 else
                       "QA FAILED: AC51 late broad speckled gas; Studio validation only. "
                       "This candidate is not scientifically accepted or approved for deployment.")
            receipt.update(publication_scope="studio_validation_only", qa_status="fail",
                           failed_qa_decision_sha256=qa_evidence[qa_report.name],
                           failed_qa_decision_name=qa_report.name,
                           qa_evidence_files=qa_evidence,
                            original_parent=(
                                _read(source.parent.parent / "result.json")["original_parent"]
                                if failed_v5 or failed_um or failed_bm else failed_qa["parent_model"]),
                           deployment_allowed=False,
                           explicit_failed_qa_review_exception=True,
                           warning=warning)
        else:
            receipt.update(publication_scope="review", qa_status="pass")
        _write(staging / "publication_receipt.json", receipt)
        receipt_sha256 = sha256_file(staging / "publication_receipt.json")
        clear_hidden_attribute(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        _rename_directory_no_replace(staging, destination)
        try:
            _require(sha256_file(destination / "publication_receipt.json") == receipt_sha256,
                     "Published QA and model publication receipt changed")
            _require(verify_candidate(destination,
                                      allow_historical_source_path=failed_v5) == checked,
                     "Published model differs from fully checked stage")
            if validation_only:
                _check_validation_only_evidence(destination, qa_evidence)
        except BaseException:
            # A post-rename I/O failure must not leave an unchecked model in
            # the scanned library. The verified source remains untouched.
            _rename_directory_no_replace(
                destination, staging_root / f"quarantined-regional-review-{uuid.uuid4().hex}")
            raise
        return receipt
    finally:
        # Never remove anything outside the one freshly generated staging folder.
        if stage_parent.exists():
            remove_owned_stage(staging_root, stage_parent, "regional-review-publication-")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--qa-report", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true", help="Publish only after passing QA")
    mode.add_argument("--publish-for-validation-only", action="store_true",
                      help="Explicitly copy a documented failed-QA AC, UM or BM candidate for Studio validation only")
    args = parser.parse_args(argv)
    if args.publish or args.publish_for_validation_only:
        if args.destination is None or args.staging_root is None or args.qa_report is None:
            parser.error("Publication requires --destination, --staging-root and --qa-report")
        result = publish(args.source, args.destination, args.staging_root, args.qa_report,
                         validation_only=args.publish_for_validation_only)
    else:
        result = verify_candidate(args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
