"""Test a frozen model on another same-series experiment, visibly or blind.

The selected optical weights are unchanged. A registered-target evaluation
also keeps its fitted physical rules and the exact target's registered context;
otherwise explicit optical-only mode uses target images, baseline and domain.
Temporal state starts afresh for the target. Blind evidence is excluded from
later training unless the user explicitly selects it.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid


from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.execution import default_execution_cache_root
from darsia.fluidflower.phase_runtime import load_physics_profile, load_run_context, physics_runtime_identity
from darsia.fluidflower.phase_training import trainer as t
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_training.scope import requested_series, series_id, validate_training_series
from darsia.fluidflower.phase_validation import _candidate_paths, _model_identity
from darsia.fluidflower.phase_workflow import build_phase_review_command, discover_phase_review_sessions
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.storage_paths import configured_tool_path
from darsia.fluidflower.tools.phase_transfer_provenance import audit_calibration_scope, target_exposure
from darsia.fluidflower.tools.phase_transfer_materialization import materialize_transfer_predictions


EVENT_PREFIX = "PHASE_TRANSFER_EVENT "
EVIDENCE_ROLE = "blind_transfer_validation"
VISIBLE_EVIDENCE_ROLE = "model_transfer_evaluation"
PHYSICS_FILES = ("phase_native_reference.json", "phase_facies_reference.json", "phase_reviewed_reference.json",
    "phase_reference_support.json", "phase_presence_context.json", "phase_physics.json", "phase_plume_support.json")
class _RawSeriesRoots(dict[str, Path]):
    """Keep legacy mapping callers while resolving defaults only on access."""

    def __missing__(self, series: str) -> Path:
        return configured_tool_path(series, "raw_inputs")


# Explicit caller/test overrides remain supported. Resolve only the requested
# series lazily, so discovering one project never probes the others' storage.
RAW_SERIES_ROOTS: dict[str, Path] = _RawSeriesRoots()


_DISCOVERY_CACHE = None


def _cached(bucket, key, produce):
    """Memoise one pure read for the duration of a single discovery pass.

    Discovery asks the same questions once per candidate experiment: the same manifests, the
    same training scope, the same physics profile, the same digests. It never writes to those
    files while it runs, so reading each of them once gives the same answer for a fraction of
    the time, which matters when the model lives on a share and every miss is a network round
    trip. Outside a discovery pass nothing is cached, so the batch still sees each file as it
    is at that moment. Deep copies are handed out so a caller that modifies what it received
    cannot reach into the cache.
    """
    cache = _DISCOVERY_CACHE
    if cache is None:
        return produce()
    store = cache.setdefault(bucket, {})
    if key not in store:
        store[key] = produce()
    return copy.deepcopy(store[key])


def _read_uncached(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def read(path):
    return _cached("read", str(path), lambda: _read_uncached(path))


_sha256_file_uncached = sha256_file
_load_physics_profile_uncached = load_physics_profile


def sha256_file(path):  # noqa: F811 - shadows the import with the same behaviour
    return _cached("sha256", str(path), lambda: _sha256_file_uncached(resolve_review_path(path)))


def load_physics_profile(root):  # noqa: F811 - shadows the import with the same behaviour
    return _cached("physics", str(root), lambda: _load_physics_profile_uncached(resolve_review_path(root)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def event(stage, **values):
    print(EVENT_PREFIX + json.dumps(dict(stage=stage, **values)), flush=True)


def progress_event(stage, **values):
    """Report progress to the window that spawned this process; importers get silence.

    The batch imports this module and runs discovery once per experiment. It has its own
    progress reporting and must not have its log filled with thousands of these lines.
    """
    if __name__ == "__main__":
        event(stage, **values)


def model_training_scope(model_root):
    """Use frozen reviewed regions, not directory names or invented holdouts."""
    return _cached("scope", str(model_root), lambda: _model_training_scope_uncached(model_root))


def _model_training_scope_uncached(model_root):
    model_root = resolve_review_path(model_root)
    request = read(model_root / "training_request.json")
    selected = requested_series(request["settings"])
    if selected not in {"ffac", "ffum", "ffbm"}:
        raise ValueError("Blind transfer requires an explicitly single-series model")
    pairs, protected = [], {str(model_root / "training_request.json"): sha256_file(model_root / "training_request.json")}
    for bundle in request["bundles"]:
        pair = []
        for role in ("decisions", "combined_manifest"):
            record = bundle[role]
            path = resolve_review_path(record["snapshot_path"])
            if sha256_file(path) != record["sha256"]:
                raise ValueError("Selected model's frozen evidence changed")
            protected[str(path)] = record["sha256"]
            pair.append(path)
        pairs.append(t.ReviewBundle(*pair))
    preserve_revisions = (bool(request["settings"].get("preserve_native_revisions"))
                          or request.get("artifact_kind") == "optical-region-training-candidate")
    options = {"preserve_native_revisions": True} if preserve_revisions else {}
    sources, regions, _ = t._load_sources_and_regions(pairs, **options)
    validate_training_series(sources, regions, selected)
    return selected, {r.run.casefold() for r in regions}, request, protected


def target_sources(manifest_path, selected):
    """Inspect prepared target metadata; do not load frames or predictions."""
    return _cached("targets", (str(manifest_path), str(selected)),
                   lambda: _target_sources_uncached(manifest_path, selected))


def _prepared_native_photographs(review_root, run, native):
    """Check the fixed photographic files while retaining the shadow identity."""
    if "ffac_rig_camera_sources" in native:
        from darsia.fluidflower.phase_ffac_rig_camera_runtime import prepared_photographs
        return prepared_photographs(review_root,run,native)
    from darsia.fluidflower.phase_v5_source_alias import KEY, resolve_original_native_source

    aliased = KEY in native
    photo_root, _original_path, original = resolve_original_native_source(review_root, run, native)

    def photograph(record, directory, stem):
        expected = photo_root / directory / run / f"{stem}.jpg"
        declaration = (record.get("v5_display_modes") or {}).get("natural")
        if aliased or declaration is not None:
            if (not isinstance(declaration, dict)
                    or not isinstance(declaration.get("jpeg_path"), str)
                    or not isinstance(declaration.get("jpeg_sha256"), str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", declaration["jpeg_sha256"])):
                raise ValueError("Prepared native photograph lacks its frozen natural-display binding")
            path = resolve_review_path(declaration["jpeg_path"])
            if path.resolve() != expected.resolve() or not path.is_file():
                raise ValueError("Prepared native photograph is missing or escaped its fixed source path")
            return path
        path = expected.with_suffix(".png") if expected.with_suffix(".png").is_file() else expected
        if not path.is_file():
            raise ValueError("Prepared native photograph is missing")
        return path

    # The alias validates the immutable manifest, geometry and unchanged source
    # declarations. Availability checks file presence; the selected display
    # loader verifies its JPEG hash when opened, without reading every photo here.
    baseline = photograph(original.get("reference") or {}, "native_references", "pre_injection")
    for index, record in enumerate(original.get("frame_sources") or []):
        photograph(record, "native_frames", f"frame_{index:03d}")
    return baseline


def _target_sources_uncached(manifest_path, selected):
    manifest_path = resolve_review_path(manifest_path)
    manifest = read(manifest_path)
    rows = []
    jobs = manifest.get("jobs", [])
    for job_index, job in enumerate(jobs, 1):
        if not isinstance(job, dict):
            continue
        dataset = job.get("dataset") or manifest.get("dataset")
        if dataset and series_id(dataset) != selected:
            continue
        run = str(job.get("run") or "").strip().casefold()
        if not run:
            continue
        progress_event("target", run=run, job_index=job_index, job_total=len(jobs))
        candidate_root, review_root = _candidate_paths(manifest_path.parent, job)
        if job.get("analysis_source_review_root"):
            review_root = Path(job["analysis_source_review_root"])
            if not review_root.is_absolute():
                review_root = manifest_path.parent / review_root
        try:
            review_root = resolve_review_path(analysis_review_root(resolve_review_path(review_root), run))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        native_path = review_root / "native_frames" / run / "manifest.json"
        if not native_path.is_file():
            continue
        native = read(native_path)
        legacy_provenance = None
        if not dataset:
            # Older FF_AC review exports omitted dataset/path fields. Accept
            # their known schema only when every source image identifies the
            # same experiment in the actual FF_AC raw-data tree.
            sources = native.get("frame_sources") or []
            source_pattern = rf"(?:^|/)albus/raw data/{re.escape(run)}(?:/|$)"
            if not (selected == "ffac" and manifest.get("schema") == 1
                    and str(manifest.get("calibration_profile", "")).startswith("expanded25-")
                    and native.get("kind") == "native_resolution_corrected_review_frames"
                    and str(native.get("run", "")).casefold() == run
                    and re.fullmatch(r"ac\d+", run) and sources
                    and all(re.search(source_pattern, str(row.get("source_path", "")).replace("\\", "/").casefold())
                            for row in sources)):
                continue
            dataset = selected
            legacy_provenance = "Legacy FF_AC campaign schema and every native raw-image path agree"
        count = int(native.get("frame_count") or 0)
        if count <= 0 or len(native.get("frame_sources", [])) != count:
            continue
        frames = native["frame_sources"]
        try:
            times = [float(frame["elapsed_h"]) for frame in frames]
            if (any(not math.isfinite(value) for value in times)
                    or any(later <= earlier for earlier, later in zip(times, times[1:]))
                    or any(frame.get("frame_index") != index for index, frame in enumerate(frames))):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        controls = job.get("independent_reference_frames") or []
        if controls and (controls != [0] or times[0] >= 0
                or native.get("timeline_revision", {}).get("kind") != "prepend-evaluated-baseline-v1"):
            continue
        try:
            baseline = _prepared_native_photographs(review_root, run, native)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        # A copied manifest lives elsewhere: resolve every asset root now, while
        # the target campaign still supplies the relative-path origin.
        target_job = dict(job, run=run, dataset=selected,
            candidate_folder=str(candidate_root), review_output_dir=str(review_root),
            analysis_source_review_root=str(review_root))
        rows.append(dict(run=run, dataset=selected, source_manifest=str(manifest_path),
            review_root=str(review_root), native_manifest=str(native_path),
            native_manifest_sha256=sha256_file(native_path), baseline=str(baseline),
            frame_count=count, job=target_job, prepared=True,
            has_registered_domain=bool((native.get("domain") or {}).get("path")),
            dataset_provenance=legacy_provenance or "Explicit campaign dataset"))
    return rows


def registered_target_physics(model_root, target):
    """Require the chosen model's own frozen context for this exact target grid."""
    root = Path(model_root).resolve()
    profile = load_physics_profile(root)
    key = f"{target['dataset']}/{target['run']}"
    entry = None if profile is None else profile.get("runs", {}).get(key)
    if entry is None or entry.get("native_manifest_sha256") != target["native_manifest_sha256"]:
        raise ValueError("Target physical context is not registered for this native source revision")
    profiles = [root / name for name in PHYSICS_FILES if (root / name).is_file()]
    if len(profiles) != 1:
        raise ValueError("Expected one frozen physical profile")
    context = (root / entry["path"]).resolve()
    if Path(entry["path"]).is_absolute() or not context.is_relative_to(root):
        raise ValueError("Registered context escaped the selected model")
    if sha256_file(context) != entry["sha256"]:
        raise ValueError("Registered target physical context changed")
    return dict(profile=profile, profile_path=profiles[0], context_path=context,
        source_runtime=physics_runtime_identity(profile), key=key)


def restrict_profile_to_target(profile, key):
    """Restrict context registries; every fitted rule/calibration value is retained."""
    result = copy.deepcopy(profile)
    def visit(value):
        if not isinstance(value, dict):
            return
        if "format" in value and isinstance(value.get("runs"), dict):
            if key not in value["runs"]:
                raise ValueError("A nested physics profile does not register the target")
            value["runs"] = {key: value["runs"][key]}
        for name in ("reference_support", "presence_support", "gas_support"):
            if name in value:
                visit(value[name])
    visit(result)
    physics_runtime_identity(result)  # validators check all inherited registries
    return result


def discover_transfer_runs(model_root, extra_source_manifests=(), *, own_validation_only=False):
    selected, trained_runs, request, _ = model_training_scope(model_root)
    calibration_audit = audit_calibration_scope(model_root)
    own_manifest = Path(model_root) / "validation" / "postflight_manifest.json"
    selected_candidates = None
    if own_validation_only:
        # Automatic opening is bounded to this model. Older models without their
        # own prepared maps need an explicit Find experiments for historical sources.
        manifests = [Path(value) for value in extra_source_manifests]
        if own_manifest.is_file():
            prepared = read(own_manifest)
            binding = read(Path(model_root) / "validation" / "validation_binding.json")
            identity = _model_identity(Path(model_root))
            if (prepared.get("model_id"), prepared.get("model_sha256")) != identity:
                raise ValueError("The model's prepared validation manifest has another model identity")
            if (binding.get("model_id"), binding.get("model_sha256")) != identity:
                raise ValueError("The model's prepared validation binding has another model identity")
            selected_candidates = {str(run).casefold(): str(candidate)
                for run, candidate in (binding.get("selected_candidates") or {}).items()}
            if (not selected_candidates or len(selected_candidates) != int(binding.get("run_count") or 0)):
                raise ValueError("The model's prepared validation selection is incomplete")
            manifests.insert(0, own_manifest)
    else:
        manifests = [Path(value) for value in extra_source_manifests]
        manifests += list((default_execution_cache_root() / "phase_transfer_targets").glob("*/postflight_manifest.json"))
        manifests += [Path(model_root) / name / "postflight_manifest.json" for name in ("review_source", "validation")]
        manifests += [Path(row["combined_manifest"]["snapshot_path"]) for row in request["bundles"]]
        # This historical native-image export predates dataset-tagged manifests.
        # Find it beside the configured model library, without crawling raw images.
        for location in (Path(model_root), Path(request.get("settings", {}).get("parent_model") or model_root)):
            for ancestor in location.parents:
                if ancestor.name == "phase_models":
                    manifests.append(ancestor.parent / "objective_v2_manual_review_20260823/postflight_manifest.json")
                    break
        for session in discover_phase_review_sessions():
            if session.get("combined_root"):
                manifests.append(Path(session["combined_root"]) / "postflight_manifest.json")
    output, seen = [], set()
    # Reading every prepared source takes tens of seconds and Studio shows empty lists meanwhile,
    # so report what is being read. Each experiment is checked against its own manifest here,
    # which is where most of the time goes.
    sources = list(dict.fromkeys(p.resolve() for p in manifests))
    for index, manifest in enumerate(sources, 1):
        progress_event("discovery", experiments=len(output),
                       manifests_done=index, manifests_total=len(sources))
        if not manifest.is_file():
            continue
        try:
            rows = target_sources(manifest, selected)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if selected_candidates is not None and manifest == own_manifest.resolve():
            rows = [row for row in rows if
                str(row.get("job", {}).get("candidate_id") or "") == selected_candidates.get(row["run"])]
            if {row["run"] for row in rows} != set(selected_candidates):
                raise ValueError("The model's prepared experiments do not match its selected validation candidates")
        for row in rows:
            key = (row["dataset"], row["run"], row["native_manifest_sha256"])
            if key in seen:
                continue
            seen.add(key)
            progress_event("discovery", experiments=len(output), run=row["run"],
                           manifests_done=index, manifests_total=len(sources))
            training_seen = row["run"] in trained_runs
            status = "In optical training" if training_seen else "Not in listed optical training runs"
            exposure = target_exposure(model_root, row["run"], trained_runs, calibration_audit)
            try:
                registered_target_physics(model_root, row)
                modes = ["registered-target", "optical-only"]
                physics_status = "Full model: target physical context registered"
            except (OSError, ValueError, KeyError, TypeError):
                try:
                    from darsia.fluidflower.tools.phase_transfer_context import prepared_target_physics
                    prepared_target_physics(model_root, row)
                    modes = ["prepared-target", "optical-only"]
                    physics_status = "Full frozen model: own prepared target context; no target refitting"
                except (OSError, ValueError, KeyError, TypeError):
                    modes = ["optical-only"]
                    domain_status = "target domain" if row["has_registered_domain"] else "full image crop; no porous-domain mask"
                    physics_status = f"Optical classifier only ({domain_status}); full-model test needs registered target geometry and injection protocol"
            row.update(training_seen=training_seen, **exposure,
                independent_holdout=False, physics_mode=modes[0], available_physics_modes=modes,
                physics_status=physics_status,
                label=f"{row['run'].upper()} · {status} · {row['frame_count']} frames",
                evidence_role=EVIDENCE_ROLE, include_in_training=False)
            output.append(row)
    return sorted(output, key=lambda row: (row["training_seen"], row["run"], row["source_manifest"]))


def completed_transfer_results(model_root):
    """Recover one verified result per exact target revision and physics mode."""
    root = Path(model_root).resolve()
    identity = _model_identity(root)
    results = (default_execution_cache_root() / "phase-transfer-review").glob("*/transfer_result.json")
    output, seen = [], set()
    for path in sorted(results, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            result = read(path)
            if (result.get("model_id"), result.get("source_model_sha256")) != identity:
                continue
            plan = read(path.parent / "transfer_review.json")
            binding = read(path.parent / "validation/validation_binding.json")
            validate_review_binding(plan, binding)
            if plan["source_model_sha256"] != identity[1] or plan["source_model_id"] != identity[0]:
                continue
            native = str(Path(plan["target_native_manifest"]).resolve())
            native_sha = plan["protected_sha256"].get(native)
            if not native_sha or sha256_file(Path(native)) != native_sha:
                continue
            key = (plan["run"], native_sha, plan["physics_mode"])
            if key in seen:
                continue
            seen.add(key)
            # The complete receipt keeps its binding. The UI only needs the
            # verified identity and reopening command, not a second full model.
            result.pop("binding", None)
            if result.get("review_mode") == "visible" and plan["physics_mode"] in {"prepared-target","registered-target"}:
                from darsia.fluidflower.tools.phase_training_promotions import registered_promotion
                try:
                    promotion = registered_promotion(path.parent, root)
                    if promotion:
                        result["promotion_receipt"] = promotion["receipt_path"]
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result["promotion_error"] = str(error)
            output.append(dict(result, model_root=str(root), target_run=plan["run"],
                target_native_manifest_sha256=native_sha, physics_mode=plan["physics_mode"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return output


def latest_transfer_result(model_root, *, run=None, native_manifest_sha256=None, physics_mode=None):
    """Recover this model's latest matching test without running inference."""
    return next((row for row in completed_transfer_results(model_root)
        if (run is None or row["target_run"] == run)
        and (native_manifest_sha256 is None or row["target_native_manifest_sha256"] == native_manifest_sha256)
        and (physics_mode is None or row["physics_mode"] == physics_mode)), None)


def discover_raw_series_runs(series, roots=None):
    """List experiment folders without treating raw photographs as prepared inputs."""
    overrides = roots or RAW_SERIES_ROOTS
    root = (Path(overrides[series]) if overrides
            else configured_tool_path(series, "raw_inputs"))
    if not root.is_dir():
        return []
    runs = {}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name.casefold()
        pattern = (r"ac(\d+)" if series == "ffac" else
                   r"run(\d+)(?:_.*)?" if series == "ffum" else r"c(\d+)")
        match = re.fullmatch(pattern, name)
        if match is None:
            continue
        number = int(match.group(1))
        if series == "ffac" and number < 17:
            continue
        run = ("ac" if series == "ffac" else "run" if series == "ffum" else "c") + str(number)
        runs.setdefault(run, str(folder.resolve()))
    return [dict(run=run, dataset=series, raw_source_root=path)
            for run, path in sorted(runs.items(), key=lambda item: int(re.search(r"\d+", item[0]).group()))]


def _session_frame_progress(session, run, frame_count):
    """Count valid explicit frame judgments in one saved session."""
    try:
        decisions = read(session["decision_path"])["decisions"]
        decision = decisions.get(run)
        if not isinstance(decision, dict):
            return 0, False, False
        from darsia.fluidflower.tools.calibration_pairwise_review import normalize_frame_choices
        choices = normalize_frame_choices(decision.get("frame_choices"),
            frame_count=frame_count, comparison_mode="single_candidate")
    except (OSError, KeyError, TypeError, ValueError):
        return 0, False, False
    activity = bool(choices or decision.get("annotations") or decision.get("phase_quiz_answers"))
    resolved = sum(row["choice"] != "unresolved" for row in choices)
    return resolved, resolved == frame_count, activity


def _review_sessions_for_sources(model_root, completed, extra_review_roots):
    roots = [Path(path) for path in extra_review_roots]
    for result in completed:
        command = result.get("command") or []
        if "--save-root" in command:
            try:
                roots.append(Path(command[command.index("--save-root") + 1]))
            except (IndexError, TypeError, ValueError):
                pass
    sessions = discover_phase_review_sessions()
    if roots:
        sessions += discover_phase_review_sessions(roots)
    return list({str(row.get("session_path")): row for row in sessions}.values())


def _review_binding_for_result(result):
    if not result.get("evaluation_root"):
        return None
    root = Path(result["evaluation_root"])
    try:
        plan = read(root / "transfer_review.json")
        binding = read(root / "validation/validation_binding.json")
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return dict(combined_root=str(root / "validation"),
        target_model_id=plan.get("evaluation_model_id"),
        candidate=(binding.get("selected_candidates") or {}).get(result.get("target_run")))


def _review_binding_for_session(session, source_identity):
    """Recover older visible evaluations even when a newer test supersedes them."""
    if session.get("evidence_role") != VISIBLE_EVIDENCE_ROLE:
        return None
    try:
        combined = Path(session["combined_root"]).resolve()
        root = combined.parent
        plan = read(root / "transfer_review.json")
        binding = read(combined / "validation_binding.json")
        if (plan.get("review_mode") != "visible"
                or (plan.get("source_model_id"), plan.get("source_model_sha256")) != source_identity
                or plan.get("evaluation_model_id") != session.get("target_model_id")
                or str(combined) != str((root / "validation").resolve())):
            return None
        run = str(plan["run"]).casefold()
        native = str(Path(plan["target_native_manifest"]).resolve())
        native_sha = plan["protected_sha256"][native]
        candidate = (binding.get("selected_candidates") or {}).get(run)
        if candidate != (session.get("selected_candidates") or {}).get(run):
            return None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return (run, native_sha), dict(combined_root=str(combined),
        target_model_id=plan["evaluation_model_id"], candidate=candidate)


def _review_binding_for_parent(model_root, row):
    root = Path(model_root)
    try:
        from darsia.fluidflower.phase_evidence_audit import verified_review_binding
        binding = _cached("saved_parent_binding", str(root), lambda:
            verified_review_binding(root, saved_comparison=True))
        candidate = (binding.get("selected_candidates") or {}).get(row["run"])
        if not candidate:
            return None
        target = next((source for source in target_sources(root / "validation/postflight_manifest.json", row["dataset"])
            if source["run"] == row["run"] and source["native_manifest_sha256"] == row["native_manifest_sha256"]
            and source.get("job", {}).get("candidate_id") == candidate), None)
        if target is None:
            return None
        job = target["job"]
        for field, expected in (("params_sha256", binding.get("pipeline_sha256")),
                                ("optical_model_sha256", binding.get("model_sha256")),
                                ("physics_runtime", binding.get("physics_runtime"))):
            if field in job and job[field] != expected:
                return None
        profile = load_physics_profile(root)
        entry = (profile or {}).get("runs", {}).get(f"{row['dataset']}/{row['run']}")
        if entry is not None and entry.get("native_manifest_sha256") != row["native_manifest_sha256"]:
            return None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return dict(combined_root=str(binding["combined_root"]),
        target_model_id=binding.get("model_id"),
        candidate=candidate)


def _matches_review_session(session, binding, run, source_model_id):
    if not binding or not binding.get("candidate"):
        return False
    if session.get("evidence_role") == VISIBLE_EVIDENCE_ROLE:
        if session.get("source_model_id") != source_model_id:
            return False
    elif session.get("target_model_id") != source_model_id:
        return False
    try:
        return (Path(session.get("combined_root") or "").resolve() == Path(binding["combined_root"]).resolve()
            and session.get("target_model_id") == binding["target_model_id"]
            and (session.get("selected_candidates") or {}).get(run) == binding["candidate"])
    except (OSError, TypeError, ValueError):
        return False


def _annotate_review_status(model_root, rows, completed, sessions):
    try:
        source_identity = _model_identity(Path(model_root))
    except (OSError, KeyError, TypeError, ValueError):
        source_identity = (None, None)
    source_model_id = source_identity[0]
    bindings = {}
    for result in completed:
        key = (result.get("target_run"), result.get("target_native_manifest_sha256"))
        binding = _review_binding_for_result(result)
        if binding:
            bindings.setdefault(key, []).append(binding)
    if source_model_id:
        for session in sessions:
            recovered = _review_binding_for_session(session, source_identity)
            if recovered:
                key, binding = recovered
                bindings.setdefault(key, []).append(binding)
    for row in rows:
        total = int(row.get("frame_count") or 0)
        row["preparation_status"] = "ready"
        row["total_frames"] = total
        row["reviewed_frames"] = 0
        row["review_status"] = "not_reviewed"
        if not source_model_id or total <= 0:
            continue
        key = (row["run"], row["native_manifest_sha256"])
        candidates = [*bindings.get(key, [])]
        parent_binding = _review_binding_for_parent(model_root, row)
        if parent_binding:
            candidates.append(parent_binding)
        complete = activity = False
        for session in sessions:
            if not any(_matches_review_session(session, binding, row["run"], source_model_id)
                       for binding in candidates):
                continue
            count, resolved, active = _session_frame_progress(session, row["run"], total)
            row["reviewed_frames"] = max(row["reviewed_frames"], count)
            complete = complete or resolved
            activity = activity or active
        if complete:
            row["review_status"] = "reviewed"
        elif activity:
            row["review_status"] = "in_progress"


def transfer_discovery(model_root, extra_source_manifests=(), extra_review_roots=(), *, own_validation_only=False):
    """Attach exact predictions and manual review status, including unprepared raw runs."""
    global _DISCOVERY_CACHE
    outer = _DISCOVERY_CACHE
    _DISCOVERY_CACHE = {} if outer is None else outer
    try:
        return _transfer_discovery(model_root, extra_source_manifests, extra_review_roots,
                                   own_validation_only=own_validation_only)
    finally:
        _DISCOVERY_CACHE = outer


def saved_validation_sources(model_root, binding, review_sessions=()):
    """List already-verified saved maps without reconstructing training inputs.

    The Studio catalogue supplies its validated model/pipeline binding. Small
    native manifests are still hash checked here; individual photographs and
    numeric maps retain their full checks when the selected frame is opened.
    Older exports without source hashes use the existing discovery route.
    """
    if (binding.get("model_root")
            and resolve_review_path(binding["model_root"]).resolve()
            != resolve_review_path(model_root).resolve()):
        raise ValueError("Saved experiment binding identifies another selected model")
    combined = resolve_review_path(binding["combined_root"])
    manifest_path = combined / "postflight_manifest.json"
    manifest = read(manifest_path)
    expected = (binding.get("model_id"), binding.get("model_sha256"),
                binding.get("pipeline_sha256"), binding.get("physics_runtime"))
    actual = tuple(manifest.get(key) for key in (
        "model_id", "model_sha256", "pipeline_sha256", "physics_runtime"))
    if actual != expected or not binding.get("completed_utc"):
        raise ValueError("Saved experiment manifest differs from its verified model binding")
    selected = {str(run).casefold(): str(candidate)
                for run, candidate in (binding.get("selected_candidates") or {}).items()}
    if not selected or len(selected) != int(binding.get("run_count") or 0):
        raise ValueError("Saved experiment selection is incomplete")
    jobs = [job for job in manifest.get("jobs", []) if
            str(job.get("candidate_id") or "") == selected.get(str(job.get("run") or "").casefold())]
    if len(jobs) != len(selected) or {str(job["run"]).casefold() for job in jobs} != set(selected):
        raise ValueError("Saved experiments do not match their selected candidates")
    if any(not job.get("native_manifest_sha256") for job in jobs):
        return None
    rows = []
    for job in jobs:
        run = str(job["run"]).casefold()
        for key, wanted in (("optical_model_sha256", expected[1]),
                            ("params_sha256", expected[2]), ("physics_runtime", expected[3])):
            if job.get(key) != wanted:
                raise ValueError("Saved experiment has another model or pipeline: " + run)
        directory = Path(job.get("analysis_source_review_root") or job["review_output_dir"])
        if not directory.is_absolute():
            directory = combined / directory
        directory = resolve_review_path(directory)
        native_path = directory / "native_frames" / run / "manifest.json"
        if sha256_file(native_path) != job["native_manifest_sha256"]:
            raise ValueError("Saved experiment's source manifest changed: " + run)
        native = read(native_path)
        frames = native.get("frame_sources") or []
        count = int(native.get("frame_count") or 0)
        if count <= 0 or count != len(frames):
            raise ValueError("Saved experiment's frame count is incomplete: " + run)
        times = [float(frame["elapsed_h"]) for frame in frames]
        if (any(not math.isfinite(value) for value in times)
                or any(right <= left for left, right in zip(times, times[1:]))
                or any(frame.get("frame_index") != index for index, frame in enumerate(frames))):
            raise ValueError("Saved experiment's frame/time binding is invalid: " + run)
        dataset = series_id(job.get("dataset") or manifest.get("dataset"))
        parent = dict(combined_root=str(combined), target_model_id=binding["model_id"],
                      candidate=selected[run])
        row = dict(run=run, dataset=dataset, source_manifest=str(manifest_path),
            review_root=str(directory), native_manifest=str(native_path),
            native_manifest_sha256=job["native_manifest_sha256"], frame_count=count,
            total_frames=count, job=dict(job), prepared=True, preparation_status="ready",
            parent_review_binding=parent, completed_evaluations={},
            available_physics_modes=[], physics_mode="", training_seen=None,
            optical_training_status="unverified", calibration_evidence_status="unverified",
            review_status="not_reviewed", reviewed_frames=0,
            label=f"{run.upper()} · saved phase maps · {count} frames")
        activity = complete = False
        for session in review_sessions:
            if _matches_review_session(session, parent, run, binding["model_id"]):
                reviewed, resolved, active = _session_frame_progress(session, run, count)
                row["reviewed_frames"] = max(row["reviewed_frames"], reviewed)
                complete |= resolved
                activity |= active
        row["review_status"] = "reviewed" if complete else "in_progress" if activity else "not_reviewed"
        rows.append(row)
    return sorted(rows, key=lambda row: (int(re.search(r"\d+", row["run"]).group()), row["run"]))


def _transfer_discovery(model_root, extra_source_manifests=(), extra_review_roots=(), *, own_validation_only=False):
    rows = (discover_transfer_runs(model_root, extra_source_manifests, own_validation_only=True)
        if own_validation_only else discover_transfer_runs(model_root, extra_source_manifests))
    completed = completed_transfer_results(model_root)
    for row in rows:
        matches = {result["physics_mode"]: result for result in completed
            if (result["target_run"], result["target_native_manifest_sha256"])
            == (row["run"], row["native_manifest_sha256"])}
        row["completed_evaluations"] = matches
        parent_binding = _review_binding_for_parent(model_root, row)
        row["parent_review_binding"] = parent_binding
        if row["physics_mode"] in matches:
            row["label"] += " · predictions ready"
            if matches[row["physics_mode"]].get("promotion_receipt"):
                row["label"] += " · included for next training"
    sessions = _review_sessions_for_sources(model_root, completed, extra_review_roots)
    _annotate_review_status(model_root, rows, completed, sessions)
    if not own_validation_only:
        try:
            selected = model_training_scope(model_root)[0]
            prepared_runs = {row["run"] for row in rows}
            for raw in discover_raw_series_runs(selected):
                if raw["run"] in prepared_runs:
                    continue
                run = raw["run"]
                rows.append(dict(**raw, source_manifest="", native_manifest="",
                    native_manifest_sha256="", frame_count=0, total_frames=0,
                    reviewed_frames=0, review_status="not_reviewed",
                    preparation_status="needs_preparation", prepared=False,
                    training_seen=None, optical_training_status="unverified",
                    calibration_evidence_status="unverified",
                    available_physics_modes=[], physics_mode="", completed_evaluations={},
                    label=f"{run.upper()} · needs preparation"))
        except (OSError, KeyError, TypeError, ValueError):
            pass
    def run_order(row):
        run = str(row.get("run") or "")
        number = re.search(r"\d+", run)
        return (int(number.group()) if number else math.inf, run)
    rows.sort(key=run_order)
    return dict(sources=rows, last_evaluation=completed[0] if completed else None)


def verify_hashes(records):
    for name, expected in records.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"Evaluation input changed: {name}")


def prepare_transfer_review(model_root, target_run, source_manifest, output_parent, *, physics_mode, review_mode="blind"):
    """Create a separate evaluation bundle; never modify or fit the source model."""
    if physics_mode not in {"optical-only", "registered-target", "prepared-target"}:
        raise ValueError("Choose explicit optical-only, registered-target or prepared-target physical mode")
    if review_mode not in {"visible", "blind"}:
        raise ValueError("Choose visible prediction testing or blind labeling")
    evidence_role = VISIBLE_EVIDENCE_ROLE if review_mode == "visible" else EVIDENCE_ROLE
    model_root, source_manifest, output_parent = (Path(p).resolve() for p in (model_root, source_manifest, output_parent))
    if output_parent == model_root or output_parent.is_relative_to(model_root):
        raise ValueError("Transfer evaluation must be separate from the selected model")
    selected, trained_runs, request, protected = model_training_scope(model_root)
    calibration_audit = audit_calibration_scope(model_root)
    exposure = target_exposure(model_root, target_run, trained_runs, calibration_audit)
    protected.update({row["path"]: row["sha256"] for row in calibration_audit["evidence"]})
    inspect_phase_model_bundle(model_root)
    source_identity = _model_identity(model_root)
    targets = [row for row in target_sources(source_manifest, selected) if row["run"] == target_run.strip().casefold()]
    if not targets:
        raise ValueError("No complete same-series native source and pre-injection baseline for the selected experiment")
    revisions = {row["native_manifest_sha256"] for row in targets}
    if len(revisions) != 1:
        raise ValueError("The selected campaign has multiple native revisions of this experiment")
    target = targets[0]
    physics = None
    if physics_mode == "registered-target":
        physics = registered_target_physics(model_root, target)
        context, _ = load_run_context(model_root, physics["profile"], selected, target["run"],
            native_manifest_sha256=target["native_manifest_sha256"])
        box = read(target["native_manifest"])["content_box_xywh"]
        if context.domain.shape != (box[3], box[2]):
            raise ValueError("Registered target context differs from native image geometry")
    elif physics_mode == "prepared-target":
        from darsia.fluidflower.tools.phase_transfer_context import prepared_target_physics
        physics = prepared_target_physics(model_root, target)
    native = read(target["native_manifest"])
    source_files = [model_root / "summary.json", source_manifest, Path(target["native_manifest"]), Path(target["baseline"])]
    source_files += [Path(target["native_manifest"]).parent / f"frame_{index:03d}.jpg" for index in range(target["frame_count"])]
    domain_path = (native.get("domain") or {}).get("path")
    if domain_path:
        domain = Path(domain_path)
        if not domain.is_absolute():
            domain = Path(target["review_root"]) / domain
        source_files.append(domain.resolve())
    weight_paths = sorted(p for p in (model_root / "models").rglob("*.json") if p.is_file())
    source_files.extend(weight_paths)
    physics_provenance = []
    if physics is not None:
        source_files.extend((physics["profile_path"], physics["context_path"]))
        source_files.extend(physics.get("source_files", []))
        physics_provenance = [physics["profile_path"]]
        for directory in model_root.iterdir():
            if directory.is_dir() and directory.name.endswith(("_calibration", "_code_snapshot")):
                physics_provenance.extend(path for path in directory.rglob("*") if path.is_file())
        source_files.extend(physics_provenance)
    protected.update({str(path.resolve()): sha256_file(path) for path in source_files})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evaluation_id = f"phase-transfer-{stamp}-{uuid.uuid4().hex[:8]}"
    root = output_parent / evaluation_id
    root.mkdir(parents=True, exist_ok=False)
    write(root / "source_snapshot/calibration_scope.json", calibration_audit)
    for index, record in enumerate(calibration_audit["evidence"]):
        origin = Path(record["path"])
        destination = root / "source_snapshot/calibration_evidence" / f"{index:03d}_{origin.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        if sha256_file(destination) != record["sha256"]:
            raise ValueError("Copied calibration evidence changed")
    if physics is not None:
        if physics_mode == "prepared-target":
            from darsia.fluidflower.tools.phase_transfer_context import freeze_target_physics
            freeze_target_physics(root, model_root, target, physics)
            original_context = root / "source_snapshot/target_context/input.npz"
            original_context.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(physics["context_path"], original_context)
            if sha256_file(original_context) != protected[str(physics["context_path"].resolve())]:
                raise ValueError("Copied prepared target input changed")
            write(original_context.with_suffix(".json"), physics["descriptor"])
        else:
            filtered = restrict_profile_to_target(physics["profile"], physics["key"])
            write(root / physics["profile_path"].name, filtered)
            context_copy = root / physics["context_path"].relative_to(model_root)
            context_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(physics["context_path"], context_copy)
            if sha256_file(context_copy) != protected[str(physics["context_path"])]:
                raise ValueError("Copied target context changed")
        for path in physics_provenance:
            destination = root / "source_snapshot/physics" / path.relative_to(model_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            if sha256_file(destination) != protected[str(path.resolve())]:
                raise ValueError("Copied physical calibration provenance changed")
    for source in weight_paths:
        destination = root / source.relative_to(model_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != protected[str(source.resolve())]:
            raise ValueError("Copied optical weights differ from the selected model")
    for name, path in (("training_request.json", model_root / "training_request.json"),
                       ("summary.json", model_root / "summary.json"), ("source_manifest.json", source_manifest)):
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        if sha256_file(destination) != protected[str(path.resolve())]:
            raise ValueError("Source snapshot changed")
    evaluation_job = dict(target["job"], quality_flags=dict(dataset=selected,
        training_seen_review=target["run"] in trained_runs, independent_holdout=False,
        human_review_required=True, evidence_role=evidence_role, include_in_training=False, **exposure))
    source_document = dict(schema=1, campaign_id=evaluation_id,
        source_campaign_manifest=str(source_manifest), jobs=[evaluation_job],
        evidence_role=evidence_role, include_in_training=False)
    write(root / "review_source/postflight_manifest.json", source_document)
    summary = copy.deepcopy(read(model_root / "summary.json"))
    summary.update(model_id=evaluation_id, campaign_id=evaluation_id, created_utc=utc_now(),
        artifact_kind="transfer-evaluation", evaluation_parent=str(model_root),
        evaluation_scope=f"{selected}/{target['run']}; {physics_mode}; {exposure['calibration_history']}",
        evaluation_training_seen=target["run"] in trained_runs,
        source_model_id=source_identity[0], source_model_sha256=source_identity[1],
        optical_weights_unchanged=True, training_performed=False, human_review_required=True,
        independent_validation=False, deployment=None, human_validation=None, **exposure)
    write(root / "summary.json", summary)
    evaluation_request = copy.deepcopy(request)
    evaluation_request.update(model_id=evaluation_id, output_root=str(root),
        cache_root=str(root / "evaluation_cache"), created_utc=utc_now())
    evaluation_request["settings"].update(parent_model=str(model_root),
        iteration="Transfer prediction test" if review_mode == "visible" else "Blind transfer evaluation",
        fit_heads=[], inherited_heads=[], evaluation_only=True, include_in_training=False,
        physics_mode=physics_mode, review_mode=review_mode)
    write(root / "training_request.json", evaluation_request)
    evaluation_profile = load_physics_profile(root)
    if physics_mode == "optical-only" and evaluation_profile is not None:
        raise ValueError("Optical-only evaluation unexpectedly inherited source-run physical context")
    physics_runtime = physics_runtime_identity(evaluation_profile)
    inspect_phase_model_bundle(root)
    evaluation_identity = _model_identity(root)
    training_seen = target["run"] in trained_runs
    plan = dict(format="phase-transfer-review-v2", evaluation_root=str(root), review_mode=review_mode,
        evaluation_model_id=evaluation_identity[0], evaluation_model_sha256=evaluation_identity[1],
        source_model=str(model_root), source_model_id=source_identity[0], source_model_sha256=source_identity[1],
        run=target["run"], dataset=selected, training_seen=training_seen,
        listed_training_runs=sorted(trained_runs), **exposure, independent_holdout=False,
        listed_calibration_runs=calibration_audit["fitted_runs"], calibration_audit=calibration_audit,
        label="In optical training" if training_seen else "Not in listed optical training runs",
        physics_mode=physics_mode, physics_runtime=physics_runtime,
        source_physics_runtime=None if physics is None else physics["source_runtime"],
        physics_status=("Full chosen model: exact target context and unchanged fitted rules; fresh target-only temporal state"
            if physics is not None else "Optical classifier only: target baseline/images/domain; physical rules omitted"),
        target_native_manifest=target["native_manifest"], source_manifest=str(source_manifest),
        target_baseline=target["baseline"], protected_sha256=protected,
        evaluation_sha256={str(path): sha256_file(path) for path in root.rglob("*") if path.is_file()},
        evidence_role=evidence_role, include_in_training=False, requires_explicit_training_selection=True,
        review_surface="open_correction" if review_mode == "visible" else "blind_region_labeling",
        blind_surface="blind_region_labeling" if review_mode == "blind" else None,
        prediction_display_before_blind_review=False, predictions_visible=review_mode == "visible")
    verify_hashes(protected)
    write(root / "transfer_review.json", plan)
    return plan


def materialize_transfer_review(plan, *, progress=None, backend="auto", output_mode="review"):
    root = Path(plan["evaluation_root"])
    verify_hashes(plan["protected_sha256"])
    verify_hashes(plan["evaluation_sha256"])
    if _model_identity(root) != (plan["evaluation_model_id"], plan["evaluation_model_sha256"]):
        raise ValueError("Prepared evaluation model changed")
    if physics_runtime_identity(load_physics_profile(root)) != plan["physics_runtime"]:
        raise ValueError("Transfer acquired a different physics profile or runtime")
    binding = materialize_transfer_predictions(plan, progress=progress, backend=backend, output_mode=output_mode)
    if set(binding["selected_candidates"]) != {plan["run"]} or binding.get("physics_runtime") != plan["physics_runtime"]:
        raise ValueError("Transfer validation changed experiment or physics mode")
    verify_hashes(plan["protected_sha256"])
    verify_hashes(plan["evaluation_sha256"])
    return binding


def validate_review_binding(plan, binding):
    from darsia.fluidflower.tools.phase_reference_readiness import require_reference_readiness
    if binding.get("output_mode", "review") != "review":
        raise ValueError("Production phase output has no Studio review media; prepare a review evaluation")
    root = Path(plan["evaluation_root"])
    require_reference_readiness(root / "validation", allow_pending=True)
    verify_hashes(plan["evaluation_sha256"])
    if physics_runtime_identity(load_physics_profile(root)) != plan["physics_runtime"]:
        raise ValueError("Transfer physical runtime changed after preparation")
    if set(binding["selected_candidates"]) != {plan["run"]} or binding.get("physics_runtime") != plan["physics_runtime"]:
        raise ValueError("Blind review binding identifies another experiment or physics mode")
    if (binding["model_id"], binding["model_sha256"]) != (plan["evaluation_model_id"], plan["evaluation_model_sha256"]):
        raise ValueError("Blind review binding identifies another evaluation model")


def transfer_review_command(plan, binding, *, mode, reviewer, save_root, cache_root, use_gpu=True):
    if mode not in {"blind_region_labeling", "open_correction"}:
        raise ValueError("Choose blind labeling or an explicit separate comparison")
    root = Path(plan["evaluation_root"])
    validate_review_binding(plan, binding)
    visible_test = plan.get("review_mode", "blind") == "visible"
    if visible_test and mode != "open_correction":
        raise ValueError("A visible prediction test cannot later be described as blind")
    command = build_phase_review_command(mode=mode, reviewer=reviewer,
        save_root=Path(save_root), cache_root=Path(cache_root), campaign_root=root / "validation",
        selected_candidates=binding["selected_candidates"], target_model_id=plan["evaluation_model_id"],
        review_session_id=f"{'visible' if visible_test else 'blind'}-transfer-{uuid.uuid4().hex[:12]}", use_gpu=use_gpu,
        program=Path(__file__).with_name("transfer_prediction_review_desktop.py" if visible_test
                                        else "blind_transfer_review_desktop.py"))
    command += ["--transfer-plan", str(root / "transfer_review.json")]
    if mode == "open_correction" and not visible_test:
        command.append("--open-correction")
    return command


def blind_review_command(plan, binding, **options):
    if plan.get("review_mode", "blind") != "blind":
        raise ValueError("Visible prediction tests cannot be opened as blind evaluations")
    return transfer_review_command(plan, binding, mode="blind_region_labeling", **options)


def visible_review_command(plan, binding, **options):
    if plan.get("review_mode", "blind") != "visible":
        raise ValueError("Use the explicit comparison action to reveal a blind evaluation")
    return transfer_review_command(plan, binding, mode="open_correction", **options)


def correction_review_command(plan, binding, *, blind_session_id, **options):
    """Explicit later comparison, retaining original model and evaluation identity."""
    if plan.get("review_mode", "blind") != "blind":
        raise ValueError("A direct prediction test has no blind-label snapshot")
    command = transfer_review_command(plan, binding, mode="open_correction", **options)
    return [*command, "--blind-session-id", blind_session_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("list", "prepare"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--review-root", type=Path, action="append", default=[])
    parser.add_argument("--own-validation-only", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--physics-mode", choices=("optical-only", "registered-target", "prepared-target"))
    parser.add_argument("--review-mode", choices=("visible", "blind"), default="visible")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reviewer")
    parser.add_argument("--save-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    if args.operation == "list":
        event("sources", **transfer_discovery(args.model, args.source_manifest, args.review_root,
                                              own_validation_only=args.own_validation_only))
        return
    if len(args.source_manifest) != 1 or not all((args.run, args.output_parent, args.physics_mode, args.reviewer, args.save_root, args.cache_root)):
        parser.error("Prepare requires one source manifest, run, output parent, explicit physics mode, reviewer, save root and cache root")
    plan = prepare_transfer_review(args.model, args.run, args.source_manifest[0], args.output_parent,
        physics_mode=args.physics_mode, review_mode=args.review_mode)
    event("prepared", evaluation_root=plan["evaluation_root"], training_seen=plan["training_seen"], label=plan["label"])
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(4)
    with threadpool_limits(limits=4):
        binding = materialize_transfer_review(plan, progress=lambda message: event("progress", message=message), backend=args.backend)
    options = dict(reviewer=args.reviewer, save_root=args.save_root, cache_root=args.cache_root)
    command = (visible_review_command if args.review_mode == "visible" else blind_review_command)(plan, binding, **options)
    correction_command = (None if args.review_mode == "visible" else
        correction_review_command(plan, binding, **options,
            blind_session_id=command[command.index("--review-session-id") + 1]))
    result = dict(stage="ready", evaluation_root=plan["evaluation_root"], binding=binding,
        completed_utc=utc_now(), model_root=str(args.model.resolve()), model_id=plan["source_model_id"],
        source_model_sha256=plan["source_model_sha256"],
        target_run=plan["run"],
        target_native_manifest_sha256=plan["protected_sha256"][str(Path(plan["target_native_manifest"]).resolve())],
        command=command, correction_command=correction_command,
        reopen_command=command if args.review_mode == "visible" else None, review_mode=args.review_mode,
        training_seen=plan["training_seen"], label=plan["label"], physics_mode=plan["physics_mode"],
        **{name: plan[name] for name in ("optical_training_status", "calibration_evidence_status", "exposure_label", "calibration_history")})
    write(Path(plan["evaluation_root"]) / "transfer_result.json", result)
    event(**result)


if __name__ == "__main__":
    main()
