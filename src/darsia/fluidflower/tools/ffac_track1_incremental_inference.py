"""Bind sealed individual inference runs to one unchanged Track1 fit.

The source batch may still be running. Each admitted run requires its worker's
actual source-SHA seal and an immutable native timeline. The shared model and
static runtime profile never change as subsequent runs become ready.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import uuid

from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import (
    _bound, _copied_files, _new, _read, _require,
)
from darsia.fluidflower.tools.train_phase_model_further import native_source

CONTRACT = "ffac-track1-incremental-inference-v1"
BINDING_CONTRACT = "ffac-track1-incremental-run-binding-v1"
PLAN_FILE = "inference_incremental_plan.json"


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_incremental(old_candidate: Path, source_job_path: Path, output: Path,
                        *, spec_path: Path) -> dict:
    old, job_path, root = map(lambda p: Path(p).resolve(),
                              (old_candidate, source_job_path, output))
    _require(not root.exists(), "Use an absent incremental candidate")
    source = _read(job_path)
    _require(source.get("inference_only") is True and source.get("retraining") is False
             and source.get("labels_changed") is False,
             "Source job is not the frozen inference-only batch")
    source_processor = source.get("processor") or {}
    display_mode = source_processor.get("display_mode", "photographic_v1")
    _require(display_mode in {"photographic_v1", "natural_modes_v1"},
             "Unsupported inference display contract")
    v2 = display_mode == "natural_modes_v1"
    if v2:
        _require(source_processor.get("isolated_invalid_policy") == "fill_isolated_invalid_v1",
                 "V2 inference requires the specified isolated-invalid fill")
    summary = _read(old / "summary.json")
    learned = _read(old / "regional_model.json")
    profile = _read(old / "phase_optical_regions.json")
    request = _read(old / "training_request.json")
    _require((summary.get("inventory") or {}).get("datasets") == ["ffac"]
             and (summary.get("regional_fit") or {}).get("converged") is True
             and (learned.get("fit_report") or {}).get("converged") is True
             and (profile.get("v5_track1_review_scope") or {}).get("mass_analysis_allowed") is False
             and (request.get("settings", {}).get("v5_raw_photometry") or {}).get("provisional_studio_training") is True,
             "Source is not the converged provisional Track1 fit")
    jobs = _read(old / "review_source/postflight_manifest.json")["jobs"]
    by_run = {job["run"]: job for job in jobs}
    versions = {row["run"]: row["sha256"] for row in source["source_versions"]}
    _require(len(by_run) == len(versions) == 43 and set(by_run) == set(versions),
             "Source plan must contain the original 43 current runs")
    expected = []
    originals = {}
    for run in sorted(by_run, key=lambda r: int(r[2:])):
        path = old / "review_source" / run / "original_job.json"
        original = _read(path)
        _, _, native = native_source(original)
        _require(native["v5_track1_photo_geometry"]["original_source_manifest_sha256"] == versions[run],
                 f"{run}: planned source revision differs from the trained candidate")
        expected.append(dict(run=run, source_manifest_sha256=versions[run],
                             frame_count=native["frame_count"]))
        originals[run] = original
    _require(sum(row["frame_count"] for row in expected) == 864,
             "Expected native timeline differs from43/864")
    root.mkdir(parents=True, exist_ok=False)
    copied = _copied_files(old, root)
    now = utc_now()
    new_summary = dict(summary, model_id=root.name, campaign_id=root.name,
        created_utc=now, inference_revision_created_utc=now,
        inherited_training_candidate_id=old.name, training_performed_in_this_revision=False,
        inference_only_rerun=True, incremental_inference=True,
        display_name=("FF_AC Track1 · naturlig visning / ×4.3 / ×7.1 · 30.09.2026" if v2 else
                      "FF_AC Track1 · fotografisk visning + fylte sandhull · 30.09.2026"),
        mass_eligible=False, scientific_acceptance=False)
    _new(root / "summary.json", new_summary)
    copied["summary.json"] = sha256_file(root / "summary.json")
    original_sha = {}
    for run, original in originals.items():
        path = root / "incremental_original_sources" / run / "original_job.json"
        _new(path, original)
        original_sha[run] = sha256_file(path)
    plan = dict(contract=CONTRACT, created_utc=now, candidate=str(root), old_candidate=str(old),
        old_model_identity=list(_model_identity(old)), model_identity=list(_model_identity(root)),
        source_job_path=str(job_path), source_job_sha256=sha256_file(job_path),
        expected_sources=expected, run_count=43, frame_count=864,
        original_sources_root=str(root / "incremental_original_sources"),
        original_source_job_sha256_by_run=original_sha,
        trained_artifacts_preserved=copied,
        old_profile_path=str(old / "phase_optical_regions.json"),
        old_profile_sha256=sha256_file(old / "phase_optical_regions.json"),
        spec_path=str(Path(spec_path).resolve()), spec_sha256=sha256_file(spec_path),
        optical_training_performed=False, regional_training_performed=False,
        mass_eligible=False, scientific_acceptance=False,
        enclosed_hole_policy="fill_enclosed_v1", enclosed_hole_max_area_px=60000,
        vertical_gap_max_length_px=64, vertical_gap_rule="same_column_supported_above_and_below_v1",
        display_mode=display_mode)
    if v2:
        plan.update(isolated_invalid_policy="fill_isolated_invalid_v1",
                    isolated_invalid_max_component_px=50,
                    display_transfer=source_processor.get("display_transfer"))
    _new(root / PLAN_FILE, plan)
    profile = deepcopy(profile)
    profile["v5_track1_incremental_inference"] = dict(plan_path=PLAN_FILE,
        plan_sha256=sha256_file(root / PLAN_FILE), binding_directory="inference_run_bindings")
    _new(root / "phase_optical_regions.json", profile)
    _new(root / "inference_incremental_profile_seal.json", dict(
        profile_sha256=sha256_file(root / "phase_optical_regions.json"),
        plan_sha256=sha256_file(root / PLAN_FILE)))
    _new(root / "review_source/postflight_manifest.json", {"jobs": []})
    return plan


def verify_incremental(root: Path, runs: list[str] | None = None) -> dict:
    root = Path(root).resolve()
    plan = _read(root / PLAN_FILE)
    _require(plan.get("contract") == CONTRACT and Path(plan["candidate"]).resolve() == root
             and plan.get("model_identity") == list(_model_identity(root))
             and plan.get("run_count") == 43 and plan.get("frame_count") == 864
             and plan.get("optical_training_performed") is False
             and plan.get("regional_training_performed") is False
             and plan.get("mass_eligible") is False,
             "Incremental frozen fit plan is invalid")
    _bound(plan["source_job_path"], plan["source_job_sha256"], "frozen inference source job")
    for relative, digest in plan["trained_artifacts_preserved"].items():
        _bound(str(root / relative), digest, f"preserved {relative}")
    seal = _read(root / "inference_incremental_profile_seal.json")
    _bound(str(root / PLAN_FILE), seal["plan_sha256"], "incremental plan")
    _bound(str(root / "phase_optical_regions.json"), seal["profile_sha256"], "static runtime profile")
    allowed = {row["run"] for row in plan["expected_sources"]}
    selected = list(runs or [])
    _require(set(selected) <= allowed, "Requested run is outside the frozen cohort")
    for run in selected:
        binding = _read(root / "inference_run_bindings" / f"{run}.json")
        _require(binding.get("contract") == BINDING_CONTRACT and binding.get("run") == run
                 and binding.get("plan_sha256") == seal["plan_sha256"]
                 and binding.get("model_identity") == plan["model_identity"],
                 f"{run}: immutable run binding is invalid")
        _bound(binding["source_receipt_path"], binding["source_receipt_sha256"], f"{run} source seal")
        _bound(binding["materialized_manifest_path"], binding["materialized_manifest_sha256"], f"{run} native timeline")
        _bound(binding["review_source_manifest_path"], binding["review_source_manifest_sha256"], f"{run} map source")
    return dict(plan, source_bindings=[_read(root / "inference_run_bindings" / f"{run}.json")
                                     for run in selected])


def bind_completed_run(root: Path, event: dict) -> dict:
    root = Path(root).resolve()
    plan = verify_incremental(root)
    run = str(event.get("run") or "").lower()
    expected = {row["run"]: row for row in plan["expected_sources"]}.get(run)
    _require(expected is not None and event.get("event") == "run_complete"
             and (event.get("source_verification_cache") or {}).get("sealed_actual_sha_rechecks") is True
             and event.get("source_manifest_sha256") == expected["source_manifest_sha256"]
             and event.get("frame_count") == expected["frame_count"],
             "Only an exact completed worker source-SHA seal can admit a run")
    binding_path = root / "inference_run_bindings" / f"{run}.json"
    if binding_path.exists():
        old_binding = _read(binding_path)
        _require(old_binding.get("worker_completed_event") == event,
                 f"{run}: already bound to a different worker seal")
        verify_incremental(root, [run])
        return old_binding
    manifest_path = _bound(event["materialized_manifest_path"], event["materialized_manifest_sha256"], f"{run} new native")
    receipt_path = _bound(event["receipt_path"], event["receipt_sha256"], f"{run} worker materialization")
    native = _read(manifest_path)
    original_path = Path(plan["original_sources_root"]) / run / "original_job.json"
    _bound(str(original_path), plan["original_source_job_sha256_by_run"][run], f"{run} original job")
    original = _read(original_path)
    _, old_path, before = native_source(original)
    old_marker, marker = before["v5_track1_photo_geometry"], native["v5_track1_photo_geometry"]
    processor = native["v5_photometry"]["processor"]
    old_processor = before["v5_photometry"]["processor"]
    _require(marker["original_source_manifest_sha256"] == old_marker["original_source_manifest_sha256"]
             and marker["frames"] == old_marker["frames"]
             and marker.get("enclosed_hole_policy") == "fill_enclosed_v1"
             and marker.get("vertical_gap_max_length_px") == 64
             and marker.get("vertical_gap_rule") == "same_column_supported_above_and_below_v1"
             and processor.get("enclosed_hole_policy") == "fill_enclosed_v1"
             and processor.get("enclosed_hole_vertical_gap_max_px") == 64
             and processor.get("display_mode") == plan.get("display_mode", "photographic_v1")
             and (plan.get("display_mode") != "natural_modes_v1" or
                  processor.get("isolated_invalid_policy") == plan.get("isolated_invalid_policy"))
             and native["v5_photometry"].get("display_jpeg_is_training_input") is False,
             f"{run}: source-photo geometry or filled/display policy changed")
    for key in ("calibration_manifest_sha256", "photo_source_aggregate_sha256"):
        _require(processor.get(key) == old_processor.get(key), f"{run}: fitted normalization changed")
    for key in ("frame_count", "analysis_image_shape_hw", "canvas_wh", "content_box_xywh"):
        _require(native.get(key) == before.get(key), f"{run}: native grid changed")
    _require(len(native["frame_sources"]) == expected["frame_count"], f"{run}: incomplete timeline")
    for a, b in zip(before["frame_sources"], native["frame_sources"]):
        for key in ("frame_index", "elapsed_h", "source_sha256", "source_raw_sha256", "v5_raw_sha256", "v5_raw_members"):
            _require(a.get(key) == b.get(key), f"{run}: acquisition/timestamp changed")
    profile = _read(root / "phase_optical_regions.json")
    entry = deepcopy(profile["runs"][f"ffac/{run}"])
    _bound(str(root / entry["path"]), entry["sha256"], f"{run} unchanged nominal context")
    entry["native_manifest_sha256"] = event["materialized_manifest_sha256"]
    photo = entry["provenance"]["v5_track1_photo_geometry"]
    photo.update(native_manifest_sha256=event["materialized_manifest_sha256"],
        enclosed_hole_policy="fill_enclosed_v1", vertical_gap_max_length_px=64,
        vertical_gap_rule="same_column_supported_above_and_below_v1")
    entry["provenance"]["v5_raw_photometry"] = dict(
        entry["provenance"].get("v5_raw_photometry") or {},
        receipt_path=str(receipt_path), receipt_sha256=event["receipt_sha256"],
        materialized_manifest_sha256=event["materialized_manifest_sha256"],
        source_manifest_sha256=expected["source_manifest_sha256"], physical_grid_inherited=False)
    review_root = Path(event["materialized_review_root"]).resolve()
    _require(manifest_path == review_root / "native_frames" / run / "manifest.json", "Native manifest is outside the review root")
    job = deepcopy(original)
    job.update(review_output_dir=str(review_root), analysis_source_review_root=str(review_root))
    job["v5_raw_photometry"] = dict(job.get("v5_raw_photometry") or {},
        source_revision_receipt_path=str(receipt_path), source_revision_receipt_sha256=event["receipt_sha256"],
        source_manifest_sha256=expected["source_manifest_sha256"],
        materialized_manifest_sha256=event["materialized_manifest_sha256"],
        scientific_input="sha-bound-float32-log-camera-linear-sidecar", display_jpeg_is_training_input=False)
    job["quality_flags"] = dict(job.get("quality_flags") or {}, human_review_required=True,
        mass_analysis_allowed=False, inference_only_display_mask_revision=True)
    _new(root / "review_source" / run / "original_job.json", job)
    source_path = root / "review_source" / run / "postflight_manifest.json"
    _new(source_path, {"jobs": [job]})
    binding = dict(contract=BINDING_CONTRACT, created_utc=utc_now(), run=run,
        plan_sha256=sha256_file(root / PLAN_FILE), model_identity=plan["model_identity"],
        context_entry=entry, materialized_manifest_path=str(manifest_path),
        materialized_manifest_sha256=event["materialized_manifest_sha256"],
        source_receipt_path=str(receipt_path), source_receipt_sha256=event["receipt_sha256"],
        frame_count=expected["frame_count"], source_manifest_sha256=expected["source_manifest_sha256"],
        previous_materialized_manifest_path=str(old_path), previous_materialized_manifest_sha256=sha256_file(old_path),
        nominal_context_sha256=entry["sha256"], review_source_manifest_path=str(source_path),
        review_source_manifest_sha256=sha256_file(source_path), worker_completed_event=event,
        optical_training_performed=False, regional_training_performed=False, mass_eligible=False)
    # Root is the only binder. Publish the source table before the ready receipt;
    # no worker may claim this run until its final immutable binding appears.
    ready = [_read(path)["run"] for path in (root / "inference_run_bindings").glob("*.json")]
    jobs = [_read(root / "review_source" / r / "original_job.json")
            for r in sorted(ready + [run], key=lambda r: int(r[2:]))]
    _atomic(root / "review_source/postflight_manifest.json", {"jobs": jobs})
    _atomic(binding_path, binding)
    verify_incremental(root, [run])
    return binding


def seal_incremental(root: Path, batch_receipt: Path) -> dict:
    root = Path(root).resolve()
    plan = verify_incremental(root)
    batch_receipt = Path(batch_receipt).resolve()
    batch = _read(batch_receipt)
    batch_rows = batch.get("canonical_mapping") or []
    rows = {row["run"]: row for row in batch_rows}
    expected = {row["run"]: row for row in plan["expected_sources"]}
    source_job = _read(Path(plan["source_job_path"]))
    _require(batch_receipt == Path(source_job["settings"]["output_root"]).resolve() / "materialization_receipt.json"
             and Path(batch.get("output_root") or "").resolve() == batch_receipt.parent
             and batch.get("inference_only") is True and batch.get("retraining") is False
             and batch.get("labels_changed") is False
             and batch.get("source_revisions") == 43 and batch.get("timeline_frames") == 864
             and batch.get("revisions") == batch_rows
             and batch.get("processor") == source_job.get("processor")
             and set(rows) == set(expected) and len(rows) == len(batch_rows) == 43
             and sum(row["frame_count"] for row in rows.values()) == 864
             and batch.get("source_job_sha256") == plan["source_job_sha256"],
             "Final batch does not seal the exact43/864 source plan")
    bindings = verify_incremental(root, list(expected))["source_bindings"]
    for binding in bindings:
        row = rows[binding["run"]]
        for key in ("source_manifest_sha256", "materialized_manifest_sha256", "frame_count"):
            _require(row[key] == binding[key], "Final batch differs from an admitted run")
        _require(row["source_manifest_sha256"] == expected[binding["run"]]["source_manifest_sha256"]
                 and row["frame_count"] == expected[binding["run"]]["frame_count"]
                 and row["materialized_manifest_path"] == binding["materialized_manifest_path"],
                 "Final batch native path or acquisition differs from the frozen plan")
        _require(row["receipt_path"] == binding["source_receipt_path"]
                 and row["receipt_sha256"] == binding["source_receipt_sha256"], "Final source receipt changed")
    source_manifest = root / "review_source/postflight_manifest.json"
    source_jobs = _read(source_manifest).get("jobs") or []
    by_run = {job.get("run"): job for job in source_jobs}
    _require(len(source_jobs) == len(by_run) == 43 and set(by_run) == set(expected)
             and all(_read(root / "review_source" / run / "postflight_manifest.json") ==
                     {"jobs": [by_run[run]]} for run in expected),
             "Final source campaign differs from the sealed per-run map sources")
    # Validate the complete opt-in overlay before publishing its final receipt.
    # This verifies unchanged original acquisition/H chains and context arrays;
    # it never fits, renders or rewrites a source or runtime profile.
    from darsia.fluidflower.phase_optical_regions_runtime import load_profile
    loaded = load_profile(root)
    _require(set(loaded.get("_incremental_sealed_runs") or []) == set(expected),
             "Final runtime profile lacks a sealed acquisition")
    receipt = dict(contract=CONTRACT, created_utc=utc_now(), candidate=str(root),
        model_identity=plan["model_identity"], run_count=43, frame_count=864,
        old_candidate=plan["old_candidate"], old_model_identity=plan["old_model_identity"],
        plan_sha256=sha256_file(root / PLAN_FILE),
        materialization_receipt_path=str(batch_receipt),
        materialization_receipt_sha256=sha256_file(batch_receipt),
        source_bindings=bindings, stage="source_sealed",
        run_binding_sha256={binding["run"]: sha256_file(
            root / "inference_run_bindings" / (binding["run"] + ".json")) for binding in bindings},
        review_source_sha256=sha256_file(source_manifest),
        inference_profile_sha256=sha256_file(root / "phase_optical_regions.json"),
        frozen_training_request_sha256=plan["trained_artifacts_preserved"]["training_request.json"],
        frozen_regional_candidate_sha256=plan["trained_artifacts_preserved"]["regional_model.json"],
        user_authorized_spec={"path": plan["spec_path"], "sha256": plan["spec_sha256"]},
        enclosed_hole_policy="fill_enclosed_v1", enclosed_hole_max_area_px=60000,
        vertical_gap_max_length_px=64, vertical_gap_rule="same_column_supported_above_and_below_v1",
        display_mode=plan.get("display_mode", "photographic_v1"), cpu_path_required=True,
        optical_training_performed=False, regional_training_performed=False,
        mass_eligible=False, scientific_acceptance=False,
        trained_artifacts_preserved=plan["trained_artifacts_preserved"])
    if plan.get("display_mode") == "natural_modes_v1":
        receipt.update(isolated_invalid_policy="fill_isolated_invalid_v1",
                       isolated_invalid_max_component_px=50,
                       display_transfer=plan.get("display_transfer"))
    target = root / "inference_rerun_receipt.json"
    if target.exists():
        saved = _read(target)
        _require({key: value for key, value in saved.items() if key != "created_utc"} ==
                 {key: value for key, value in receipt.items() if key != "created_utc"},
                 "Existing final inference receipt differs from the sealed source batch")
        receipt = saved
    else:
        _new(target, receipt)
    verify_sealed_incremental(root)
    return receipt


def verify_sealed_incremental(root: Path) -> dict:
    """Require complete, unchanged43/864 inference provenance before registration.

    The pending incremental plan is insufficient. This branch accepts only the
    real finalized source batch and every already sealed per-run source. It
    does not change masks, coefficients, QA eligibility or runtime identities.
    """
    root = Path(root).resolve()
    receipt = _read(root / "inference_rerun_receipt.json")
    plan = verify_incremental(root)
    _require(receipt.get("contract") == CONTRACT and receipt.get("stage") == "source_sealed"
             and Path(receipt.get("candidate") or "").resolve() == root
             and receipt.get("model_identity") == plan["model_identity"]
             and receipt.get("old_candidate") == plan["old_candidate"]
             and receipt.get("old_model_identity") == plan["old_model_identity"]
             and receipt.get("run_count") == 43 and receipt.get("frame_count") == 864
             and receipt.get("plan_sha256") == sha256_file(root / PLAN_FILE)
             and receipt.get("trained_artifacts_preserved") == plan["trained_artifacts_preserved"]
             and receipt.get("frozen_training_request_sha256") == plan["trained_artifacts_preserved"]["training_request.json"]
             and receipt.get("frozen_regional_candidate_sha256") == plan["trained_artifacts_preserved"]["regional_model.json"]
             and receipt.get("optical_training_performed") is False
             and receipt.get("regional_training_performed") is False
             and receipt.get("mass_eligible") is False
             and receipt.get("scientific_acceptance") is False
             and receipt.get("enclosed_hole_policy") == "fill_enclosed_v1"
             and receipt.get("enclosed_hole_max_area_px") == 60000
             and receipt.get("vertical_gap_max_length_px") == 64
             and receipt.get("vertical_gap_rule") == "same_column_supported_above_and_below_v1"
             and receipt.get("display_mode") == plan.get("display_mode", "photographic_v1")
             and (plan.get("display_mode") != "natural_modes_v1" or
                  receipt.get("isolated_invalid_policy") == "fill_isolated_invalid_v1")
             and receipt.get("cpu_path_required") is True,
             "Completed incremental inference receipt is invalid")
    old = Path(plan["old_candidate"]).resolve()
    _require(list(_model_identity(old)) == plan["old_model_identity"],
             "Original fitted model identity changed")
    _bound(plan["old_profile_path"], plan["old_profile_sha256"], "original fitted runtime profile")
    for relative, digest in plan["trained_artifacts_preserved"].items():
        # The new summary names this inference revision; every inherited fit,
        # request, input-label snapshot and context array remains byte-identical.
        if relative != "summary.json":
            _bound(str(old / relative), digest, f"original preserved {relative}")
    _require(receipt.get("user_authorized_spec") == {
        "path": plan["spec_path"], "sha256": plan["spec_sha256"]},
        "Completed inference receipt differs from the authorized specification")
    _bound(plan["spec_path"], plan["spec_sha256"], "frozen authorized specification")
    expected = {row["run"]: row for row in plan["expected_sources"]}
    bindings = verify_incremental(root, list(expected))["source_bindings"]
    _require(receipt.get("source_bindings") == bindings
             and set(receipt.get("run_binding_sha256") or {}) == set(expected),
             "Completed inference sources differ from their immutable run bindings")
    for binding in bindings:
        run = binding["run"]
        _bound(str(root / "inference_run_bindings" / (run + ".json")),
               receipt["run_binding_sha256"][run], f"{run} frozen inference binding")
        _require(_read(old / "review_source" / run / "original_job.json") ==
                 _read(Path(plan["original_sources_root"]) / run / "original_job.json"),
                 f"{run}: inherited source job changed")
    batch_path = _bound(receipt["materialization_receipt_path"],
                        receipt["materialization_receipt_sha256"], "completed source batch")
    batch = _read(batch_path)
    source_job = _read(Path(plan["source_job_path"]))
    rows = batch.get("canonical_mapping") or []
    by_run = {row["run"]: row for row in rows}
    _require(batch_path == Path(source_job["settings"]["output_root"]).resolve() / "materialization_receipt.json"
             and Path(batch.get("output_root") or "").resolve() == batch_path.parent
             and batch.get("source_job_sha256") == plan["source_job_sha256"]
             and batch.get("inference_only") is True and batch.get("retraining") is False
             and batch.get("labels_changed") is False
             and batch.get("source_revisions") == 43 and batch.get("timeline_frames") == 864
             and batch.get("revisions") == rows and batch.get("processor") == source_job.get("processor")
             and len(rows) == len(by_run) == 43 and set(by_run) == set(expected)
             and sum(row["frame_count"] for row in rows) == 864,
             "Completed batch differs from the frozen43/864 acquisition plan")
    for binding in bindings:
        row, expected_row = by_run[binding["run"]], expected[binding["run"]]
        _require(row["source_manifest_sha256"] == binding["source_manifest_sha256"] == expected_row["source_manifest_sha256"]
                 and row["frame_count"] == binding["frame_count"] == expected_row["frame_count"]
                 and row["materialized_manifest_path"] == binding["materialized_manifest_path"]
                 and row["materialized_manifest_sha256"] == binding["materialized_manifest_sha256"]
                 and row["receipt_path"] == binding["source_receipt_path"]
                 and row["receipt_sha256"] == binding["source_receipt_sha256"],
                 "Completed batch differs from an immutable inference run")
    source_path = root / "review_source/postflight_manifest.json"
    _bound(str(source_path), receipt["review_source_sha256"], "completed review source campaign")
    _bound(str(root / "phase_optical_regions.json"), receipt["inference_profile_sha256"], "static inference profile")
    jobs = _read(source_path).get("jobs") or []
    jobs_by_run = {job.get("run"): job for job in jobs}
    _require(len(jobs) == len(jobs_by_run) == 43 and set(jobs_by_run) == set(expected)
             and all(_read(root / "review_source" / run / "postflight_manifest.json") ==
                     {"jobs": [jobs_by_run[run]]} for run in expected),
             "Completed review campaign differs from per-run source bindings")
    from darsia.fluidflower.phase_optical_regions_runtime import load_profile
    loaded = load_profile(root)
    _require(set(loaded.get("_incremental_sealed_runs") or []) == set(expected),
             "Completed inference runtime lacks sealed sources")
    return receipt
