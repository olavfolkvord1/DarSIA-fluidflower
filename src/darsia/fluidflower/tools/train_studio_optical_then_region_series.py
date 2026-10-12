"""Detached, QA-gated optical and regional continuation for FF_UM and FF_BM.

The fresh optical fit and its regional candidate remain outside Studio's model
library. Publication is a separate, explicit operation after source and visual
QA. The older FF_AC coordinator retains its own AC47-specific resume contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

import darsia
from darsia.fluidflower import phase_features, phase_runtime, phase_review_retractions
from darsia.fluidflower.phase_training import logistic, trainer as training
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.tools._paths import series_path
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools import train_studio_series_update as regional
from darsia.fluidflower.tools import train_studio_region_update as update
from darsia.fluidflower.tools import train_studio_optical_then_region as ac
from darsia.fluidflower.tools.publish_fitted_phase_candidate import publish_fitted, verify_fitted


FORMAT = "studio-generic-optical-then-regional-v1"
FAST_PROFILE = "generic-co2-fast-v1"
FEATURE_VERSION = phase_features.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION
HEADS = ("co2_presence", "gas_within_co2")
GRID = (1e-5, 1e-3, 1e-1)
EXPECTED_RUN_COUNTS = {"ffum": 13, "ffbm": 5}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def preflight(parent: Path, library: Path, cache: Path, series: str) -> dict:
    """Check source, model, native revisions and inherited labels read-only."""
    series = str(series).casefold()
    _require(series in EXPECTED_RUN_COUNTS, "Select FF_UM or FF_BM")
    parent, library, cache = (resolve_review_path(Path(value)).resolve()
                              for value in (parent, library, cache))
    _require(library == resolve_review_path(series_path(series, "models")).resolve()
             and parent.parent == library and parent.is_dir(),
             "Parent and Studio library must belong to the selected series")
    _require(not cache.is_relative_to(parent) and not library.is_relative_to(parent),
             "Cache and library must be detached from the parent")
    summary = _read(parent / "summary.json")
    previous = _read(parent / "training_request.json")
    _require(summary.get("inventory", {}).get("datasets") == [series]
             and summary.get("optical_training_performed") is False
             and summary.get("regional_training_performed") is True
             and not previous.get("evaluation_only")
             and not previous.get("settings", {}).get("evaluation_only"),
             "Expected an isolated regional parent with inherited optical heads")
    inherited = {}
    for head in HEADS:
        path = parent / "models" / "generic" / f"{head}_signal.json"
        payload = _read(path)
        contract = payload["feature_contract"]
        _require(contract["generic_version"] == FEATURE_VERSION
                 and tuple(contract["feature_names"]) == tuple(payload["feature_names"]),
                 f"Inherited {head} is not on the v5 feature contract")
        inherited[head] = dict(sha256=sha256_file(path),
                               feature_count=len(payload["feature_names"]))
    _require(len({row["feature_count"] for row in inherited.values()}) == 1,
             "Inherited optical heads have different feature dimensions")
    profile = phase_runtime.load_physics_profile(parent)
    _require(profile is not None
             and profile.get("format") == "baseline-relative-optical-regions-v1"
             and {key.split("/")[0] for key in profile["runs"]} == {series},
             "Parent has no isolated physical regional profile")
    runtime = phase_runtime.physics_runtime_identity(profile)
    validation = _read(parent / "validation" / "postflight_manifest.json")
    _require(validation.get("physics_runtime") == runtime,
             "Parent validation does not bind its physical runtime")
    source_manifest_path = parent / "review_source" / "postflight_manifest.json"
    source = _read(source_manifest_path)
    jobs = source.get("jobs") or []
    runs = [str(job.get("run", "")).casefold() for job in jobs]
    _require(len(jobs) == len(set(runs)) == EXPECTED_RUN_COUNTS[series]
             and all(job.get("dataset") == series for job in jobs)
             and {f"{series}/{run}" for run in runs} == set(profile["runs"]),
             "Parent review timeline differs from its complete physical run set")
    if series == "ffbm":
        _require(set(runs) == {f"c{i}" for i in range(1, 6)},
                 "FF_BM requires the complete C1-C5 timeline")
    native_images = {}
    for job in jobs:
        key = f'{series}/{job["run"]}'
        review = Path(job["review_output_dir"])
        native = review / "native_frames" / job["run"] / "manifest.json"
        _require(native.is_file()
                 and sha256_file(native) == profile["runs"][key]["native_manifest_sha256"],
                 f"Original native timeline changed: {key}")
        metadata = _read(native)
        count = metadata.get("frame_count")
        rows = metadata.get("frame_sources") or []
        _require(type(count) is int and count > 0 and len(rows) == count,
                 f"Incomplete original native timeline: {key}")
        frame_digests = []
        for index, row in enumerate(rows):
            frame = review / "native_frames" / job["run"] / f"frame_{index:03d}.jpg"
            digest = sha256_file(frame)
            _require(not row.get("native_jpeg_sha256")
                     or digest == row["native_jpeg_sha256"],
                     f"Original native JPEG differs from its manifest: {key}/{index}")
            frame_digests.append(digest)
        baseline = review / "native_references" / job["run"] / "pre_injection.jpg"
        baseline_digest = sha256_file(baseline)
        index = (metadata.get("reference") or {}).get("timeline_frame_index")
        _require(type(index) is int and 0 <= index < count
                 and baseline_digest == frame_digests[index],
                 f"Original baseline differs from native timeline: {key}")
        native_images[job["run"]] = dict(manifest_sha256=sha256_file(native),
                                         baseline_sha256=baseline_digest,
                                         frame_sha256=frame_digests)
    evidence = _read(parent / "evidence.json")
    logistic.assert_safe_training_regions(evidence["regions"])
    alignment = ac.optical_source_alignment(parent, evidence, jobs)
    _require(alignment["unalignable_regions"] == 0,
             "One or more inherited labels lack their original native source")
    return dict(format=FORMAT, series=series, parent=str(parent), library=str(library),
                cache=str(cache), parent_summary_sha256=sha256_file(parent / "summary.json"),
                parent_request_sha256=sha256_file(parent / "training_request.json"),
                parent_regional_model_sha256=sha256_file(parent / profile["model"]["path"]),
                parent_source_manifest_sha256=sha256_file(source_manifest_path),
                inherited_optical_heads=inherited, feature_version=FEATURE_VERSION,
                physics_runtime=runtime, python=str(Path(__import__("sys").executable).resolve()),
                darsia_source=str(Path(darsia.__file__).resolve()),
                numpy_version=np.__version__, opencv_version=cv2.__version__,
                review_jobs=len(jobs), review_runs=sorted(runs),
                native_images=native_images,
                native_frames=sum(len(row["frame_sha256"]) for row in native_images.values()),
                effective_regional_regions=len(evidence["regions"]),
                optical_source_alignment=alignment)


def verify_prepared_request(request: dict, checked: dict, actor: str) -> dict:
    """Bind an immutable fresh request to inherited labels and native assets."""
    root = Path(request["output_root"]).resolve()
    path = root / "training_request.json"
    settings = request["settings"]
    _require(path.is_file() and path.is_relative_to(Path(checked["cache"]))
             and not root.is_relative_to(Path(checked["library"]))
             and not (root / "validation").exists()
             and request.get("created_by") == actor,
             "Optical request is not a detached immutable request for this reviewer")
    frozen = _read(path)
    _require(all(frozen.get(key) == request.get(key) for key in
                 ("model_id", "output_root", "settings", "bundles", "created_by")),
             "In-memory optical request differs from the frozen source")
    _require(settings.get("optical_training_profile") == FAST_PROFILE
             and settings.get("adapters") == ["generic"]
             and settings.get("fit_heads") == list(HEADS)
             and settings.get("inherited_heads") == []
             and settings.get("feature_version") == FEATURE_VERSION
             and settings.get("preserve_native_revisions") is True
             and settings.get("manual_training_label_policy_version") ==
                 logistic.MANUAL_TRAINING_LABEL_POLICY_VERSION
             and settings.get("regularization_grid") == list(GRID)
             and int(settings.get("mil_max_iters") or 0) >= 15360,
             "Request lost its explicit fast generic two-head v5 MIL profile")
    preparation = _read(root / "continuation_preparation.json")
    _require(Path(preparation.get("parent_model", "")).resolve() == Path(checked["parent"])
             and preparation.get("series") == checked["series"],
             "Frozen continuation points to another parent or series")
    profile = phase_runtime.load_physics_profile(root)
    _require(profile is not None
             and phase_runtime.physics_runtime_identity(profile) == checked["physics_runtime"]
             and sha256_file(root / profile["model"]["path"]) ==
                 checked["parent_regional_model_sha256"]
             and sha256_file(root / "review_source" / "postflight_manifest.json") ==
                 checked["parent_source_manifest_sha256"],
             "Optical request changed inherited physics or registered review sources")
    pairs = []
    for bundle in request["bundles"]:
        values = []
        for role in ("decisions", "combined_manifest"):
            entry = bundle[role]
            snapshot = Path(entry["snapshot_path"]).resolve()
            _require(snapshot.is_relative_to(root / "input_snapshot")
                     and sha256_file(snapshot) == entry["sha256"],
                     "Frozen review snapshot changed or escaped the request")
            values.append(snapshot)
        pairs.append(training.ReviewBundle(*values))
    sources = {}
    _, regions, _ = training._load_sources_and_regions(
        pairs, preserve_native_revisions=True, region_sources=sources)
    logistic.assert_safe_training_regions(regions)
    captured = ac.frozen_region_records(regions, sources)
    inherited = checked["optical_source_alignment"]["region_records"]
    added_ids = ac.verify_parent_region_records(inherited, captured)
    classes = Counter(row.phase_class for row in regions)
    head_labels = dict(
        co2_positive=sum(classes[name] for name in
                         ("gas", "aqueous", "gas_present_region", "aqueous_present_region",
                          "mixed_gas_aqueous", "mixed_gas_aqueous_water",
                          "mixed_aqueous_water")),
        co2_negative=classes["water"] + classes["co2_absent_region"],
        gas_positive=sum(classes[name] for name in
                         ("gas", "gas_present_region", "mixed_gas_aqueous",
                          "mixed_gas_aqueous_water")),
        gas_negative=classes["aqueous"] + classes["aqueous_present_region"] +
                     classes["water"] + classes["co2_absent_region"],
    )
    _require(len(regions) == len(captured) and added_ids,
             "Optical request needs unique cumulative labels and new reviewed evidence")
    _require(sum(classes.values()) == len(regions)
             and all(value > 0 for value in head_labels.values()),
             "Optical request lacks positive or negative labels for one of its heads")
    return dict(root=str(root), regions=len(regions),
                inherited_regions=len(inherited), new_annotation_ids=added_ids,
                new_regions=len(added_ids), class_counts=dict(classes),
                head_label_coverage=head_labels,
                region_contract_sha256=training._canonical_hash(
                    [captured[key] for key in sorted(captured)]),
                request_sha256=sha256_file(path),
                parent_summary_sha256=checked["parent_summary_sha256"],
                source_manifest_sha256=checked["parent_source_manifest_sha256"],
                regional_profile_copied=True)


def _frozen_request(path: Path) -> dict:
    path = Path(path).resolve()
    _require(path.is_file() and path.name == "training_request.json",
             "Select a frozen training_request.json")
    request = _read(path)
    _require(path.parent == Path(request["output_root"]).resolve(),
             "Request path differs from its frozen output root")
    request["request_path"] = str(path)
    request["frozen_pairs"] = [(Path(bundle["decisions"]["snapshot_path"]),
                                Path(bundle["combined_manifest"]["snapshot_path"]))
                               for bundle in request["bundles"]]
    return request


def verify_optical_selected_convergence(root: Path) -> dict:
    """Reject an apparently fitted model if its selected CV folds did not converge."""
    reports = _read(Path(root) / "cross_validation.json")
    generic = reports.get("generic", {}).get("heads", {})
    result = {}
    for head in HEADS:
        report = generic.get(head) or {}
        selected = report.get("selected_regularization")
        _require(type(selected) in (int, float) and np.isfinite(selected),
                 f"{head} has no finite selected regularization")
        matching = [row for row in report.get("regularization_grid") or []
                    if np.isclose(float(row.get("regularization", float("nan"))),
                                  float(selected), rtol=1e-12, atol=0)]
        _require(len(matching) == 1, f"{head} has no unique selected CV grid row")
        folds = matching[0].get("folds") or []
        final = report.get("final_fit") or {}
        _require(bool(folds) and all(row.get("fit_converged") is True for row in folds)
                 and final.get("converged") is True,
                 f"{head} selected CV folds or final MIL fit did not converge")
        result[head] = dict(selected_regularization=float(selected),
                            selected_cv_folds=len(folds),
                            all_selected_cv_folds_converged=True,
                            final_fit_converged=True,
                            final_iterations=int(final["iterations"]))
    return result


def _freeze_regional_evidence(hidden: Path, regional_args, optical_check: dict,
                              series: str) -> None:
    """Force regional fitting to reuse the exact optical native revisions."""
    root = Path(regional_args.job)
    _require(not (root / "evidence.json").exists(),
             "Regional evidence was already captured")
    policy = phase_review_retractions.model_retractions(hidden, series=series)
    _previous, pairs, regions, audit = optical.capture_inputs(
        hidden, series, root / "evidence_capture", [],
        withdrawal_snapshot=policy, preserve_native_revisions=True)
    bundles = [training.ReviewBundle(Path(a), Path(b)) for a, b in pairs]
    sources = {}
    _, captured, _ = training._load_sources_and_regions(
        bundles, preserve_native_revisions=True, region_sources=sources)
    contract = ac.frozen_region_contract(captured, sources)
    _require(contract == optical_check["region_contract_sha256"]
             and len(captured) == optical_check["regions"]
             and {row.region_id for row in captured} == {row.region_id for row in regions},
             "Regional capture differs from frozen optical labels or native sources")
    _write(root / "evidence.json", dict(parent=str(hidden),
           pairs=[[str(a), str(b)] for a, b in pairs],
           regions=[asdict(row) for row in regions], audit=audit))
    _write(root / "snapshot_parity.json", dict(format=FORMAT,
           optical_region_contract_sha256=optical_check["region_contract_sha256"],
           regional_region_contract_sha256=contract, regions=len(captured),
           new_annotation_ids=optical_check["new_annotation_ids"],
           live_review_sessions_reopened=False))


def prepare_only(parent: Path, library: Path, cache: Path, actor: str,
                 series: str, *, backend: str = "auto") -> dict:
    """Make a new immutable fast-profile request without fitting any model."""
    checked = preflight(parent, library, cache, series)
    request, _destination = optical.prepare(
        checked["parent"], checked["library"], checked["cache"], series,
        actor, backend=backend, feature_version=FEATURE_VERSION,
        preserve_native_revisions=True, optical_training_profile=FAST_PROFILE)
    verified = verify_prepared_request(request, checked, actor)
    return dict(format=FORMAT, series=series, profile=FAST_PROFILE,
                request_path=str(Path(request["output_root"]) / "training_request.json"),
                output_root=str(request["output_root"]), regions=verified["regions"],
                inherited_regions=verified["inherited_regions"],
                new_regions=verified["new_regions"], class_counts=verified["class_counts"],
                head_label_coverage=verified["head_label_coverage"],
                request_sha256=verified["request_sha256"],
                parent_summary_sha256=checked["parent_summary_sha256"],
                native_frames=checked["native_frames"], fitted=False,
                studio_published=False)


def execute(parent: Path, library: Path, cache: Path, job: Path, actor: str,
            series: str, *, prepared_optical_request: Path | None = None,
            backend: str = "auto", cpus: int | None = None,
            memory_gb: float | None = None) -> dict:
    """Run one serial optical/regional/native-map job; keep candidate hidden."""
    checked = preflight(parent, library, cache, series)
    job = resolve_review_path(Path(job)).resolve()
    _require(job.is_relative_to(resolve_review_path(series_path(series, "results")).resolve())
             and not job.is_relative_to(Path(checked["library"])),
             "Job must be a detached selected-series NAS results folder")
    if job.exists():
        raise FileExistsError(job)
    if prepared_optical_request is None:
        request, _destination = optical.prepare(
            checked["parent"], checked["library"], checked["cache"], series,
            actor, backend=backend, feature_version=FEATURE_VERSION,
            preserve_native_revisions=True, optical_training_profile=FAST_PROFILE)
    else:
        request = _frozen_request(prepared_optical_request)
    optical_check = verify_prepared_request(request, checked, actor)
    retry = bool((request.get("settings") or {}).get("bm_mil_retry"))
    if retry:
        from darsia.fluidflower.tools.prepare_bm_mil_retry import verify_retry_request
        _require(series == "ffbm"
                 and optical_check["region_contract_sha256"] ==
                     request["settings"]["bm_mil_retry"]["source_region_contract_sha256"],
                 "BM retry changed the source region contract")
        verify_retry_request(request)
    job.mkdir(parents=True, exist_ok=False)
    _write(job / "preflight.json", checked)
    _write(job / "optical_prepared.json", optical_check)
    start = datetime.now(timezone.utc)
    _write(job / "stage_timing.json", dict(format=FORMAT,
           optical_started_utc=start.isoformat(), stage="optical"))
    optical.fit(request)
    if retry:
        verify_retry_request(request, allow_fitted=True)
    convergence = verify_optical_selected_convergence(request["output_root"])
    _write(job / "optical_convergence.json", dict(format=FORMAT,
           selected_cv_and_final_fit=convergence))
    fitted = verify_fitted(request["output_root"], allow_nonconverged=True)
    _require(fitted["series"] == series and fitted["adapters"] == ["generic"]
             and fitted["physics_runtime"] == checked["physics_runtime"]
             and not fitted["nonconverged_heads"],
             "Optical fit, adapters or inherited physics failed verification")
    hidden = job / "optical_parent" / Path(request["output_root"]).name
    publish_fitted(request["output_root"], hidden,
                   Path(checked["cache"]) / "optical-regional-publication-stage",
                   allow_nonconverged=False)
    _require(not (hidden / "validation").exists()
             and not hidden.is_relative_to(Path(checked["library"])),
             "Intermediate optical parent was exposed to Studio")
    _write(job / "optical_fitted.json", dict(fitted=fitted, hidden_parent=str(hidden),
           publication_receipt_sha256=sha256_file(hidden / "publication_receipt.json")))
    optical_finished = datetime.now(timezone.utc)
    regional_args = regional.prepare_launch(hidden, checked["library"], checked["cache"],
                                            series, actor, backend=backend,
                                            cpus=cpus, memory_gb=memory_gb)
    regional_args.output = job / "regional_candidate" / regional_args.output.name
    regional_args.job = job / "regional_run"
    regional_args.bottom_gas_guard = False
    regional_args.region_edge_guard = False
    _require(not regional_args.output.is_relative_to(Path(checked["library"])),
             "Regional candidate must stay outside the Studio library")
    regional_args.job.mkdir(parents=True, exist_ok=False)
    _write(regional_args.job / "launch_request.json", dict(format=FORMAT,
           parent=str(hidden), output=str(regional_args.output), job=str(regional_args.job),
           series=series, optical_feature_version=FEATURE_VERSION,
           optical_training_profile=FAST_PROFILE, optical_heads=list(HEADS),
           optical_receipt=str(job / "optical_fitted.json"),
           review_candidate_only=True, studio_published=False))
    _freeze_regional_evidence(hidden, regional_args, optical_check, series)
    regional_args.status_callback = None
    update.run(regional_args)
    candidate = Path(regional_args.output)
    _write(candidate / "native_source_receipt.json", dict(
           format="studio-native-source-parity-v1", model_id=candidate.name,
           series=series, parent_source_manifest_sha256=checked["parent_source_manifest_sha256"],
           source_manifest_sha256=sha256_file(candidate / "review_source" / "postflight_manifest.json"),
           images=checked["native_images"],
           runs=checked["review_jobs"],
           frames=sum(len(row["frame_sha256"]) for row in checked["native_images"].values())))
    _write(job / "candidate_qa_state.json", dict(candidate_model=str(candidate),
           review_candidate_only=True, studio_published=False,
           qa_required_before_publication=["source provenance", "training coverage",
               "optical and regional convergence", "native image parity", "visual review"]))
    result = dict(format=FORMAT, completed_utc=datetime.now(timezone.utc).isoformat(),
                  original_parent=checked["parent"], optical_parent=str(hidden),
                  candidate_model=str(candidate), review_candidate_only=True,
                  studio_published=False, qa_pending=True, series=series,
                  optical_heads_refitted=True, regional_layer_refitted=True,
                  shared_physics_runtime=checked["physics_runtime"],
                  original_parent_unchanged=all(
                      sha256_file(Path(checked["parent"]) / name) == checked[key]
                      for name, key in (("summary.json", "parent_summary_sha256"),
                                        ("training_request.json", "parent_request_sha256"),
                                        ("review_source/postflight_manifest.json",
                                         "parent_source_manifest_sha256"))))
    _require(result["original_parent_unchanged"],
             "Original parent changed during detached training")
    _write(job / "result.json", result)
    _write(job / "stage_timing.json", dict(format=FORMAT, stage="complete",
           optical_started_utc=start.isoformat(), optical_finished_utc=optical_finished.isoformat(),
           optical_seconds=(optical_finished - start).total_seconds(),
           regional_finished_utc=result["completed_utc"],
           regional_and_maps_seconds=(datetime.fromisoformat(result["completed_utc"])
                                      - optical_finished).total_seconds(),
           regional_timing_path=str(regional_args.job / "timing.json")))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", choices=tuple(EXPECTED_RUN_COUNTS), required=True)
    for name in ("parent", "library", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--actor", default="Olav Folkvord")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--prepared-optical-request", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        if args.job is None:
            parser.error("--execute requires a fresh --job")
        result = execute(args.parent, args.library, args.cache, args.job, args.actor,
                         args.series, prepared_optical_request=args.prepared_optical_request,
                         backend=args.backend, cpus=args.cpus, memory_gb=args.memory_gb)
    elif args.prepare_only:
        if args.prepared_optical_request is not None:
            parser.error("--prepare-only creates a new immutable request")
        result = prepare_only(args.parent, args.library, args.cache, args.actor,
                              args.series, backend=args.backend)
    else:
        result = preflight(args.parent, args.library, args.cache, args.series)
        result = {key: value for key, value in result.items()
                  if key not in {"optical_source_alignment", "native_images"}}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
