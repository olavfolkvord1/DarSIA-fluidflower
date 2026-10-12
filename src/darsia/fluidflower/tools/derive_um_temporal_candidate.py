"""Freeze a detached FF_UM temporal *method* candidate without fitting heads.

The input is a complete published Studio model. This builder copies its frozen
optical/regional weights, physical contexts and native source campaign, then
adds only an opt-in aqueous temporal profile. It never materializes maps,
publishes a model or registers one in Studio. A separate, reviewed mapping job
may later run ``prepare_optical_region_studio --resume`` on the returned root.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_aqueous_temporal_support import PROFILE_KEY, Settings
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.phase_validation import _source_jobs
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.publish_regional_review_candidate import verify_candidate


FORMAT = "ffum-aqueous-temporal-derived-candidate-v1"
_COPY_DIRECTORIES = ("models", "registered_context", "review_source",
                     "parent_provenance", "input_snapshot")
_COPY_FILES = ("regional_model.json", "evidence.json", "extraction.json",
               "regional_training_protocol.json", "review_retractions.json",
               "region_inventory.csv", "expected_review_runs.json")
_PROVENANCE_FILES = ("summary.json", "training_request.json",
                     "phase_optical_regions.json", "native_source_receipt.json",
                     "publication_receipt.json")
_RUNTIME_CODE_BASE = ("phase_runtime.py", "phase_optical_regions_runtime.py")
_SOURCE_ANCHORED_V2 = "aqueous-source-anchored-temporal-v2"


def _method_settings(value: dict) -> tuple[dict, str]:
    """Validate a registered, opt-in temporal method and name its code source."""
    if value.get("format") == Settings().format:
        return asdict(Settings(**value)), "phase_aqueous_temporal_support.py"
    if value.get("format") == _SOURCE_ANCHORED_V2:
        from darsia.fluidflower.phase_aqueous_source_history import Settings as V2Settings
        return asdict(V2Settings(**value)), "phase_aqueous_source_history.py"
    raise ValueError("Unregistered FF_UM aqueous temporal-support method format")


def _profile_input(path: Path, parent_profile: dict) -> tuple[dict, str, str]:
    value = _read(path)
    if PROFILE_KEY in value:
        inherited = dict(value)
        specification = inherited.pop(PROFILE_KEY)
        if inherited != parent_profile:
            raise ValueError("Registered method profile changes inherited optical/regional settings")
        kind = "complete-regional-profile"
    else:
        specification = value
        kind = "temporal-settings-only"
    settings, module = _method_settings(specification)
    return settings, module, kind


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _replace_json(path: Path, value: dict) -> None:
    replacement = path.with_name(path.name + ".tmp")
    _write_new(replacement, value)
    os.replace(replacement, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _review_evidence(path: Path) -> bool:
    """Ignore session navigation state; treat every saved correction as evidence."""
    if path.name == "review_session.json":
        return False
    if "manual_pairwise_review" in path.parts:
        return path.suffix.casefold() in {".json", ".jsonl", ".csv"}
    return (path.name.casefold().startswith("pairwise_review_")
            or path.name.casefold().endswith(".history.jsonl"))


def review_inventory(reviews_root: Path, frozen_utc: str) -> dict:
    """Fail if the active canonical review tree gained saved evidence."""
    reviews_root = reviews_root.resolve()
    if not reviews_root.is_dir():
        raise FileNotFoundError(f"Canonical FF_UM reviews are unavailable: {reviews_root}")
    cutoff = datetime.fromisoformat(frozen_utc.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("Frozen request has no time zone")
    cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
    rows = []
    for path in reviews_root.rglob("*"):
        if not path.is_file() or not _review_evidence(path):
            continue
        stat = path.stat()
        rows.append(dict(path=path.relative_to(reviews_root).as_posix(),
                         bytes=stat.st_size, modified_ns=stat.st_mtime_ns,
                         sha256=sha256_file(path)))
    rows.sort(key=lambda row: row["path"])
    changed = [row for row in rows if row["modified_ns"] > cutoff_ns]
    if changed:
        names = ", ".join(row["path"] for row in changed[:4])
        raise ValueError(f"Saved FF_UM reviews changed after the frozen optical request: {names}. "
                         "Prepare a fresh source/fit plan; do not inherit stale fitted heads.")
    return dict(root=str(reviews_root), frozen_request_utc=frozen_utc,
                files=len(rows), inventory_sha256=_digest_json(rows),
                latest_evidence_utc=(datetime.fromtimestamp(max(row["modified_ns"] for row in rows) / 1e9,
                                  timezone.utc).isoformat() if rows else None))


def _project_paths(parent: Path, job: Path) -> tuple[Path, Path]:
    parent, job = parent.resolve(), job.resolve()
    if parent.parent.name != "models" or not parent.name.startswith("phase-model-"):
        raise ValueError("Select a published FF_UM phase model under DarSIA/models")
    project = parent.parent.parent
    if not (project / "series.toml").is_file() or project.parent.name.casefold() != "ff_um":
        raise ValueError("The parent does not belong to the FF_UM DarSIA project")
    results = (project / "results").resolve()
    if job.parent != results or not job.name.startswith("aqueous-temporal-method-"):
        raise ValueError("Use a fresh aqueous-temporal-method-* job directly under FF_UM/DarSIA/results")
    if job.exists():
        raise FileExistsError(f"Derived job already exists: {job}")
    return project, results


def _qsettings_registered(candidate: Path) -> bool:
    from PySide6.QtCore import QSettings

    settings = QSettings("DarSIA", "DarSIA Phase Model Studio")
    value = json.loads(str(settings.value("modelHistory/completedModelRoots", "{}")))
    if not isinstance(value, dict):
        raise ValueError("Studio model history has an invalid format")
    return candidate.name in value


def preflight(parent: Path, job: Path, *, candidate_name: str | None = None,
              reviews_root: Path | None = None,
              settings_file: Path | None = None) -> dict:
    parent, job = Path(parent).resolve(), Path(job).resolve()
    project, _ = _project_paths(parent, job)
    checked = verify_candidate(parent)
    if checked["series"] != "ffum" or checked["runs"] != 13 or checked["frames"] != 261:
        raise ValueError("Expected the complete published FF_UM 13-run/261-frame model")
    publication = _read(parent / "publication_receipt.json")
    if (publication.get("destination") != str(parent)
            or publication.get("model_id") != checked["model_id"]
            or publication.get("model_sha256") != checked["model_sha256"]
            or publication.get("publication_scope") != "studio_validation_only"):
        raise ValueError("The selected parent has no matching published Studio identity")
    profile = _read(parent / "phase_optical_regions.json")
    if PROFILE_KEY in profile:
        raise ValueError("Parent already contains temporal aqueous support")
    source_sha = sha256_file(parent / "review_source/postflight_manifest.json")
    if checked["source_manifest_sha256"] != source_sha:
        raise ValueError("Parent source campaign differs from verified native validation")
    frozen = _read(parent / "parent_provenance/training_request.json")
    frozen_utc = frozen.get("created_utc")
    if not isinstance(frozen_utc, str):
        raise ValueError("Parent has no dated frozen optical request")
    canonical_reviews = (project / "reviews").resolve()
    if reviews_root is not None and Path(reviews_root).resolve() != canonical_reviews:
        raise ValueError("Review inventory must use the canonical FF_UM DarSIA/reviews root")
    reviews = review_inventory(canonical_reviews, frozen_utc)
    name = candidate_name or ("phase-model-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                              + "-" + uuid.uuid4().hex[:8] + "-ffum-unified")
    if not re.fullmatch(r"phase-model-\d{8}T\d{6}Z-[0-9a-f]{8}-ffum-unified", name):
        raise ValueError("Candidate name must be a new timestamped FF_UM unified model identity")
    candidate = job / "regional_candidate" / name
    if candidate.exists() or _qsettings_registered(candidate):
        raise ValueError("Candidate identity already exists in results or Studio history")
    settings_path = Path(settings_file).resolve() if settings_file else None
    settings, method_module, input_kind = (
        _profile_input(settings_path, profile) if settings_path else
        (asdict(Settings()), "phase_aqueous_temporal_support.py", "implicit-dry-run-default"))
    code = {module: sha256_file(Path(phase_runtime.__file__).with_name(module))
            for module in (*_RUNTIME_CODE_BASE, method_module)}
    return dict(format=FORMAT, parent=str(parent), parent_model_id=checked["model_id"],
                parent_model_sha256=checked["model_sha256"],
                parent_physics_runtime=checked["physics_runtime"],
                parent_profile_sha256=sha256_file(parent / "phase_optical_regions.json"),
                parent_source_manifest_sha256=source_sha,
                parent_regional_model_sha256=sha256_file(parent / "regional_model.json"),
                published_parent_receipt_sha256=sha256_file(parent / "publication_receipt.json"),
                native_source_receipt_sha256=sha256_file(parent / "native_source_receipt.json"),
                reviews=reviews, job=str(job), candidate=str(candidate),
                candidate_id=name, runs=checked["run_names"], frames=checked["frames"],
                settings=settings, settings_source=str(settings_path) if settings_path else None,
                settings_source_sha256=sha256_file(settings_path) if settings_path else None,
                settings_source_kind=input_kind,
                method_module=method_module,
                runtime_code_sha256=code,
                temporal_format=settings["format"], preflight_utc=_utc_now())


def verify_prepared(candidate: Path, *, allow_materialized: bool = False) -> dict:
    """Check a detached, map-free derived bundle before any inference starts."""
    candidate = Path(candidate).resolve()
    job = candidate.parent.parent
    receipt = _read(candidate / "method_revision_receipt.json")
    if (receipt.get("format") != FORMAT or receipt.get("candidate") != str(candidate)
            or receipt.get("candidate_id") != candidate.name):
        raise ValueError("Derived candidate receipt identifies a different model")
    parent = Path(receipt["parent"])
    if (not parent.is_dir() or _model_identity(parent) !=
            (receipt["parent_model_id"], receipt["parent_model_sha256"])):
        raise ValueError("Published method parent changed")
    profile = _read(candidate / "phase_optical_regions.json")
    parent_profile = _read(parent / "phase_optical_regions.json")
    method_settings, method_module = _method_settings(receipt["settings"])
    if (method_settings != receipt["settings"] or method_module != receipt["method_module"]
            or set(receipt["runtime_code_sha256"]) !=
            set((*_RUNTIME_CODE_BASE, method_module))):
        raise ValueError("Derived method uses an unregistered temporal runtime")
    expected = dict(parent_profile, **{PROFILE_KEY: receipt["settings"]})
    if profile != expected or sha256_file(candidate / "phase_optical_regions.json") != receipt["profile_sha256"]:
        raise ValueError("Derived profile changed more than the opt-in temporal setting")
    if phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(candidate)) != receipt["physics_runtime"]:
        raise ValueError("Derived runtime or its source code changed")
    if _model_identity(candidate) != (candidate.name, receipt["model_sha256"]):
        raise ValueError("Derived model identity changed")
    for relative, key in (
            ("summary.json", "summary_sha256"),
            ("training_request.json", "training_request_sha256"),
            ("native_source_receipt.json", "native_source_receipt_sha256_derived")):
        if sha256_file(candidate / relative) != receipt[key]:
            raise ValueError(f"Derived method provenance changed: {relative}")
    if sha256_file(candidate / "review_source/postflight_manifest.json") != receipt["parent_source_manifest_sha256"]:
        raise ValueError("Frozen review source differs from parent")
    for relative, digest in receipt["copied_files_sha256"].items():
        target = candidate / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Frozen inherited artifact changed: {relative}")
    for module, digest in receipt["runtime_code_sha256"].items():
        if sha256_file(Path(phase_runtime.__file__).with_name(module)) != digest:
            raise ValueError(f"Runtime source changed after method freeze: {module}")
    launch = _read(job / "regional_run/launch_request.json")
    if (launch.get("review_candidate_only") is not True
            or launch.get("output") != str(candidate)
            or sha256_file(job / "regional_run/launch_request.json") !=
            receipt["detached_launch_request_sha256"]):
        raise ValueError("Candidate would auto-register in Studio during materialization")
    if not allow_materialized and ((candidate / "validation").exists()
                                   or (candidate / "studio_preparation.json").exists()):
        raise ValueError("Builder must not copy or create phase maps or Studio preparation")
    if _qsettings_registered(candidate):
        raise ValueError("Unreviewed derived candidate is already registered in Studio")
    return receipt


def verify_materialized(candidate: Path) -> dict:
    """Read-only native map/source check, independent of publication policy.

    The regular regional publisher requires a new regional fit. This method
    revision deliberately has none, so its QA must use a separate validator.
    """
    candidate = Path(candidate).resolve()
    receipt = verify_prepared(candidate, allow_materialized=True)
    from darsia.fluidflower.tools.publish_regional_review_candidate import (
        _check_reference_receipt, _inside, _physical,
    )

    identity = (candidate.name, receipt["model_sha256"])
    source_root = candidate / "review_source"
    source_sha = receipt["parent_source_manifest_sha256"]
    runtime = receipt["physics_runtime"]
    prep = _read(candidate / "studio_preparation.json")
    if (prep.get("status") != "ready" or prep.get("runs") != 13
            or prep.get("runs_done") != 13 or prep.get("partial") is not False
            or prep.get("settings_file") is not None
            or prep.get("model_identity") != list(identity)):
        raise ValueError("Studio preparation is incomplete, registered or bound to another model")
    summary = _read(candidate / "summary.json")
    if (summary.get("optical_training_performed") is not False
            or summary.get("regional_training_performed") is not False
            or summary.get("method_revision", {}).get("optical_and_regional_refit_performed") is not False):
        raise ValueError("A derived method candidate must disclose inherited fitted heads")
    native = _read(candidate / "native_source_receipt.json")
    if (native.get("format") != "studio-native-source-parity-v1"
            or native.get("series") != "ffum" or native.get("model_id") != candidate.name
            or native.get("source_manifest_sha256") != source_sha
            or native.get("inherited_receipt_sha256") != receipt["native_source_receipt_sha256"]
            or native.get("runs") != 13 or native.get("frames") != 261):
        raise ValueError("Inherited native-source receipt is incomplete or identifies another campaign")
    binding = _read(candidate / "validation/validation_binding.json")
    manifest = _read(candidate / "validation/postflight_manifest.json")
    for item in (binding, manifest):
        if ((item.get("model_id"), item.get("model_sha256")) != identity
                or item.get("physics_runtime") != runtime
                or item.get("source_manifest_sha256") != source_sha):
            raise ValueError("Native map package identifies another method, model or source")
    if (binding.get("run_count") != 13 or binding.get("output_mode", "review") != "review"
            or not binding.get("completed_utc")):
        raise ValueError("Derived candidate has no complete 13-run review binding")
    prepared = _inside(_physical(binding.get("combined_root", ""), candidate, candidate),
                       candidate / "validation", "Prepared review collection")
    if Path(str(prep.get("validation") or "")).resolve() != prepared:
        raise ValueError("Studio preparation points to another native review package")
    prepared_binding = _read(prepared / "validation_binding.json")
    if (prepared_binding.get("run_count") != 13
            or (prepared_binding.get("model_id"), prepared_binding.get("model_sha256")) != identity
            or prepared_binding.get("physics_runtime") != runtime):
        raise ValueError("Selected prepared review collection differs from frozen method")
    _check_reference_receipt(candidate / "validation", candidate, candidate)
    _check_reference_receipt(prepared, candidate, candidate)
    source_jobs = _source_jobs(source_root)
    jobs = _source_jobs(candidate / "validation")
    source_by_run = {row["run"]: row for row in source_jobs}
    runs = {row["run"] for row in jobs}
    if (len(jobs) != len(source_jobs) or len(runs) != 13
            or runs != set(source_by_run) or runs != set(native.get("images") or {})
            or runs != set(binding.get("selected_candidates") or {})):
        raise ValueError("Native review run inventory differs from the frozen 13-run source")
    profile = phase_runtime.load_physics_profile(candidate)
    if set(profile["runs"]) != {f"ffum/{run}" for run in runs}:
        raise ValueError("Frozen regional context omits a reviewed run")
    frames = 0
    for row in jobs:
        run = row["run"]
        if row.get("dataset") != "ffum" or source_by_run[run].get("dataset") != "ffum":
            raise ValueError(f"Wrong series in native package: {run}")
        package = candidate / "validation" / "run_packages" / run
        per_source = source_root / run / "postflight_manifest.json"
        per_digest = sha256_file(per_source)
        per_binding, per_manifest = (_read(package / name) for name in
                                     ("validation_binding.json", "postflight_manifest.json"))
        if (per_manifest.get("jobs") != [row]
                or per_manifest.get("source_manifest_sha256") != per_digest
                or per_binding.get("source_manifest_sha256") != per_digest
                or (per_binding.get("model_id"), per_binding.get("model_sha256")) != identity
                or per_binding.get("physics_runtime") != runtime
                or _read(per_source).get("jobs") != [source_by_run[run]]):
            raise ValueError(f"Per-run source or method receipt differs: {run}")
        _check_reference_receipt(package, candidate, candidate)
        source_review = Path(source_by_run[run]["review_output_dir"]).resolve()
        materialized_review = Path(row["review_output_dir"]).resolve()
        image_receipt = native["images"][run]
        native_manifest = source_review / "native_frames" / run / "manifest.json"
        if sha256_file(native_manifest) != image_receipt["manifest_sha256"]:
            raise ValueError(f"Native photo manifest changed: {run}")
        if sha256_file(materialized_review / "native_frames" / run / "manifest.json") != image_receipt["manifest_sha256"]:
            raise ValueError(f"Materialized photo manifest differs from frozen source: {run}")
        context = profile["runs"][f"ffum/{run}"]
        context_path = _inside(candidate / context["path"], candidate, "Run context")
        if (sha256_file(context_path) != context["sha256"]
                or context.get("native_manifest_sha256") != image_receipt["manifest_sha256"]):
            raise ValueError(f"Frozen registered context differs from native source: {run}")
        timeline = _read(native_manifest)
        count = timeline.get("frame_count")
        if type(count) is not int or count <= 0 or len(timeline.get("frame_sources") or []) != count:
            raise ValueError(f"Native photo timeline is incomplete: {run}")
        baseline = source_review / "native_references" / run / "pre_injection.jpg"
        materialized_baseline = materialized_review / "native_references" / run / "pre_injection.jpg"
        if (sha256_file(baseline) != image_receipt["baseline_sha256"]
                or sha256_file(materialized_baseline) != image_receipt["baseline_sha256"]):
            raise ValueError(f"Native pre-injection reference changed: {run}")
        if len(image_receipt.get("frame_sha256") or []) != count:
            raise ValueError(f"Native photo receipt omits frames: {run}")
        maps = _inside(Path(row["phase_map_output_dir"]), candidate / "validation", "Phase maps")
        optical_root = _inside(Path(row["raw_phase_map_output_dir"]), candidate / "validation",
                               "Raw optical maps")
        params = str(row.get("params_sha256") or "")
        optical_sha = str(row.get("optical_model_sha256") or "")
        if len(params) != 64 or len(optical_sha) != 64:
            raise ValueError(f"Native map policy lacks an exact hash: {run}")
        for index, expected in enumerate(image_receipt["frame_sha256"]):
            photo = source_review / "native_frames" / run / f"frame_{index:03d}.jpg"
            materialized_photo = materialized_review / "native_frames" / run / f"frame_{index:03d}.jpg"
            if sha256_file(photo) != expected or sha256_file(materialized_photo) != expected:
                raise ValueError(f"Native review photograph changed: {run}/{index}")
            for base, marker, label in (
                    (maps, params, "Final native phase map"),
                    (optical_root, optical_sha, "Raw optical phase map")):
                target = _inside(base / "native_phase_maps" / run / f"A_{marker[:16]}" /
                                 f"frame_{index:03d}.png", candidate / "validation", label)
                if not target.is_file() or target.stat().st_size <= 8:
                    raise ValueError(f"{label} is missing: {run}/{index}")
        frames += count
    if frames != 261:
        raise ValueError(f"Expected 261 native frames, found {frames}")
    reviews = review_inventory(Path(receipt["reviews"]["root"]),
                               receipt["reviews"]["frozen_request_utc"])
    if reviews["inventory_sha256"] != receipt["reviews"]["inventory_sha256"]:
        raise ValueError("Saved reviewer evidence changed after candidate source freeze")
    if _qsettings_registered(candidate):
        raise ValueError("Unreviewed method candidate is registered in Studio")
    return dict(format=FORMAT, candidate=str(candidate), model_id=candidate.name,
                model_sha256=identity[1], physics_runtime=runtime,
                source_manifest_sha256=source_sha, runs=13, frames=frames,
                optical_training_performed=False, regional_training_performed=False,
                studio_registered=False, qa_status="not_assessed")


def create(parent: Path, job: Path, *, candidate_name: str | None = None,
           reviews_root: Path | None = None,
           settings_file: Path | None = None) -> dict:
    if settings_file is None:
        raise ValueError("Freezing a method candidate requires an explicit --settings-file")
    checked = preflight(parent, job, candidate_name=candidate_name,
                        reviews_root=reviews_root, settings_file=settings_file)
    parent, job, candidate = (Path(checked[key]) for key in ("parent", "job", "candidate"))
    job.mkdir(parents=False, exist_ok=False)
    _write_new(job / "candidate_build_status.json", dict(status="preparing", candidate=str(candidate),
                                                    created_utc=_utc_now()))
    candidate.mkdir(parents=True, exist_ok=False)
    for name in _COPY_DIRECTORIES:
        shutil.copytree(parent / name, candidate / name)
    for name in _COPY_FILES:
        if (parent / name).is_file():
            shutil.copy2(parent / name, candidate / name)
    provenance = candidate / "method_parent_provenance"
    provenance.mkdir()
    for name in _PROVENANCE_FILES:
        shutil.copy2(parent / name, provenance / name)
    inherited = {path.relative_to(candidate).as_posix(): sha256_file(path)
                 for path in candidate.rglob("*") if path.is_file()}
    for relative, digest in inherited.items():
        original = (parent / relative.removeprefix("method_parent_provenance/")
                    if relative.startswith("method_parent_provenance/") else parent / relative)
        if not original.is_file() or sha256_file(original) != digest:
            raise ValueError(f"Inherited source changed while copying: {relative}")
    # A reviewer can save a new box while copying the frozen bundle. A second
    # inventory gate prevents a candidate with stale inherited weights.
    current_reviews = review_inventory(Path(checked["reviews"]["root"]),
                                       checked["reviews"]["frozen_request_utc"])
    if current_reviews["inventory_sha256"] != checked["reviews"]["inventory_sha256"]:
        raise ValueError("Canonical saved FF_UM review inventory changed during method freeze")
    if checked["settings_source"] and (
            sha256_file(Path(checked["settings_source"])) != checked["settings_source_sha256"]
            or _profile_input(Path(checked["settings_source"]),
                              _read(parent / "phase_optical_regions.json")) !=
            (checked["settings"], checked["method_module"], checked["settings_source_kind"])):
        raise ValueError("Method profile settings changed during candidate freeze")
    profile = _read(parent / "phase_optical_regions.json")
    profile[PROFILE_KEY] = checked["settings"]
    _write_new(candidate / "phase_optical_regions.json", profile)
    source_summary = _read(parent / "summary.json")
    now = _utc_now()
    summary = dict(schema=source_summary["schema"], model_id=candidate.name,
                   campaign_id=candidate.name, created_utc=now,
                   artifact_kind="optical-region-method-revision-candidate",
                   optical_training_performed=False, regional_training_performed=False,
                   inherited_optical_model_id=source_summary.get("inherited_optical_model_id"),
                   inherited_regional_model_id=parent.name,
                   inherited_heads=["co2_presence", "gas_within_co2"],
                   regional_model_sha256=checked["parent_regional_model_sha256"],
                   inventory=source_summary["inventory"], runs=source_summary["runs"],
                   models={}, evaluation=dict(kind="method-revision-only",
                                              generalization_validated=False),
                   human_validation=None, deployment=None,
                   method_revision=dict(format=FORMAT, parent_model_sha256=checked["parent_model_sha256"],
                                        temporal_format=checked["temporal_format"],
                                        inherited_fitted_heads=True,
                                        optical_and_regional_refit_performed=False))
    _write_new(candidate / "summary.json", summary)
    request = dict(model_id=candidate.name, output_root=str(candidate), created_utc=now,
                   artifact_kind="optical-region-method-revision-candidate", bundles=[],
                   settings=dict(series_scope="ffum", model_scope="single-series", series=["ffum"],
                                 parent_model=str(parent),
                                 source_model_sha256=checked["parent_model_sha256"],
                                 source_manifest_sha256=checked["parent_source_manifest_sha256"],
                                 method_profile=checked["settings"],
                                 fit_requested=False, no_new_saved_regions=True,
                                 native_revision_policy="Inherited exact parent photo and geometry sources"))
    _write_new(candidate / "training_request.json", request)
    native_receipt = _read(parent / "native_source_receipt.json")
    if (native_receipt.get("model_id") != parent.name
            or native_receipt.get("source_manifest_sha256") != checked["parent_source_manifest_sha256"]):
        raise ValueError("Parent native-source receipt does not match published review campaign")
    _write_new(candidate / "native_source_receipt.json",
               dict(native_receipt, model_id=candidate.name,
                    derived_without_new_native_images=True,
                    inherited_receipt_sha256=checked["native_source_receipt_sha256"]))
    model_id, model_sha = _model_identity(candidate)
    physics = phase_runtime.load_physics_profile(candidate)
    runtime = phase_runtime.physics_runtime_identity(physics)
    if model_id != candidate.name or runtime == checked["parent_physics_runtime"]:
        raise ValueError("Derived method has no new model/runtime identity")
    _write_new(job / "regional_run/launch_request.json",
               dict(review_candidate_only=True, output=str(candidate), model_id=candidate.name,
                    method_revision_format=FORMAT, fit_requested=False,
                    source_parent=str(parent), created_utc=now))
    receipt = dict(checked, format=FORMAT, created_utc=now, model_sha256=model_sha,
                   physics_runtime=runtime,
                   profile_sha256=sha256_file(candidate / "phase_optical_regions.json"),
                   summary_sha256=sha256_file(candidate / "summary.json"),
                   training_request_sha256=sha256_file(candidate / "training_request.json"),
                   native_source_receipt_sha256_derived=sha256_file(candidate / "native_source_receipt.json"),
                   detached_launch_request_sha256=sha256_file(job / "regional_run/launch_request.json"),
                   copied_files_sha256=inherited,
                   optical_training_performed=False, regional_training_performed=False,
                   maps_materialized=False, studio_registered=False, qa_status="not_run")
    _write_new(candidate / "method_revision_receipt.json", receipt)
    verify_prepared(candidate)
    _replace_json(job / "candidate_build_status.json",
                  dict(status="ready_for_isolated_map_qa", candidate=str(candidate),
                       model_sha256=model_sha, created_utc=_utc_now()))
    _write_new(job / "candidate_ready.json", dict(status="ready_for_isolated_map_qa",
                                                   candidate=str(candidate), model_sha256=model_sha,
                                                   physics_runtime=runtime, created_utc=_utc_now()))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--candidate-name")
    parser.add_argument("--reviews-root", type=Path)
    parser.add_argument("--settings-file", type=Path,
                        help="Immutable JSON for the temporal profile; validated against current runtime Settings")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--create", action="store_true",
                        help="Freeze a detached method candidate; never run maps")
    action.add_argument("--verify-prepared", type=Path, metavar="CANDIDATE",
                        help="Read-only verification before maps")
    action.add_argument("--verify-materialized", type=Path, metavar="CANDIDATE",
                        help="Read-only verification of full native maps, sources and detached Studio state")
    args = parser.parse_args(argv)
    if args.verify_prepared:
        result = verify_prepared(args.verify_prepared)
    elif args.verify_materialized:
        result = verify_materialized(args.verify_materialized)
    else:
        if args.parent is None or args.job is None:
            parser.error("--parent and --job are required for a dry-run or --create")
        result = (create if args.create else preflight)(args.parent, args.job,
            candidate_name=args.candidate_name, reviews_root=args.reviews_root,
            settings_file=args.settings_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
