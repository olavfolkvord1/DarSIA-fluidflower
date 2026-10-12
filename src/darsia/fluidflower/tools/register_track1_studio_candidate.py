"""Register an exact FF_AC Track-1 result for manual Studio review.

The registration is a small, immutable NAS receipt pointing at the completed
candidate.  It neither copies the 278-GB inference cache nor changes the
candidate, its labels, model weights, or the active phase-model default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from PIL import Image

from darsia.fluidflower.phase_learning import inspect_phase_model_bundle
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.series_paths import series_path


FORMAT = "ffac-track1-studio-validation-registration-v1"
WARNING = (
    "QA FAILED/INCOMPLETE: FF_AC Track 1 saved maps are for manual Studio "
    "validation only. Regional confirmation loses gas in AC50/AC53 weak-positive "
    "regions; no independent postfit phase labels are available. Some historical "
    "labels are not exact-source comparable. Physical scale and mass calibration "
    "are unverified. Not for mass analysis or deployment."
)
RERUN_QA_FILES = {"acceptance_report.json"}
DISPLAY_MODES_V2_ACCEPTANCE_CONTRACT = "ffac-track1-display-modes-v2-acceptance-v1"
DISPLAY_MODES_V2_CHECKS = {"a", "b", "c", "d", "e", "f", "g"}
RERUN_KINDS = {"inference_only_display_mask_rerun", "inference_only_display_modes_v2"}
ORIGINAL_QA_FILES = {
    "assessment.md", "source_exact_with_mismatched_historical_unscored.json",
    "weak_late_native_focus.json", "focus_optical_regional_parity.json",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def registration_path(model_id: str, *, results: Path | None = None) -> Path:
    results = series_path("ffac", "results") if results is None else Path(results)
    return results / "phase_model_review_registrations" / f"{model_id}.json"


def registered_roots(*, results: Path | None = None) -> list[Path]:
    """Discover all explicitly registered candidates regardless of QA status."""
    folder = registration_path("placeholder", results=results).parent
    try:
        receipts = list(folder.glob("phase-model-*.json"))
    except OSError:
        return []
    roots = []
    for path in receipts:
        try:
            value = _read(path)
            if isinstance(value.get("model_root"), str):
                # The immutable receipt records the producing machine's path.
                # A byte-preserved NAS mirror may be bound by the review-storage
                # alias registry without editing that receipt or its hashes.
                roots.append(resolve_review_path(value["model_root"]))
        except (OSError, ValueError, TypeError):
            # A corrupt receipt is reported at the exact selected model if its
            # root can be recovered; it never changes the registered default.
            continue
    return roots


def registration_warning(model: Path, *, results: Path | None = None) -> str:
    """Return the prominent QA warning when a registered pointer is intact."""
    model = resolve_review_path(model)
    path = registration_path(model.name, results=results)
    if not path.is_file():
        return ""
    invalid = "QA FAILED: Track-1 Studio registration has changed or is incomplete; do not use for mass analysis or deployment."
    try:
        value = _read(path)
        if value.get("registration_kind") == "ffac_selected_camera_policy_a_p1_final3_review":
            from darsia.fluidflower.tools.register_ffac_selected_policy_a_studio_candidate import registered_warning
            return registered_warning(model, results=results) or invalid
        if value.get("registration_kind") == "ffac_g3_t4c_p1_final3_review":
            from darsia.fluidflower.phase_ffac_p1_final3_review import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t4c_p1_final2_review":
            from darsia.fluidflower.phase_ffac_p1_final2_review import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t4c_p1_final_review":
            from darsia.fluidflower.phase_ffac_p1_final_review import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t4c_floor_review":
            from darsia.fluidflower.tools.run_ffac_t4c_floor_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t4b_floor_review":
            from darsia.fluidflower.tools.run_ffac_t4b_floor_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t4_floor_review":
            from darsia.fluidflower.tools.run_ffac_t4_floor_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_l1_neighbourhood_floor_review":
            from darsia.fluidflower.tools.run_ffac_l1_neighbourhood_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t3_floor_review":
            from darsia.fluidflower.tools.run_ffac_t3_floor_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t2_l1zone_review":
            from darsia.fluidflower.tools.run_ffac_t2_l1zone_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "ffac_g3_t2_floor_review":
            from darsia.fluidflower.tools.run_ffac_t2_floor_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_old_domain_seam_revision":
            from darsia.fluidflower.tools.revise_ffac_old_domain_seam import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_area_revision":
            from darsia.fluidflower.tools.revise_ffac_pixel_area import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_rig_camera":
            from darsia.fluidflower.tools.register_ffac_rig_camera_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_barrier_rule_b":
            from darsia.fluidflower.tools.register_ffac_barrier_rule_b_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_top_surface":
            from darsia.fluidflower.tools.register_ffac_g3_top_surface_candidate import registered_warning
            return registered_warning(model, value)
        if value.get("registration_kind") == "approved_g3_downstream":
            from darsia.fluidflower.tools.register_ffac_g3_studio_candidate import registered_warning
            return registered_warning(model, value)
        rerun = value.get("registration_kind") in RERUN_KINDS
        expected_qa = RERUN_QA_FILES if rerun else ORIGINAL_QA_FILES
        if not (value.get("format") == FORMAT and
                resolve_review_path(value.get("model_root") or "") == model and
                value.get("model_id") == model.name and
                value.get("model_sha256") == _model_identity(model)[1] and
                value.get("publication_scope") == "studio_validation_only" and
                value.get("qa_status") == "fail_or_incomplete" and
                value.get("scientific_acceptance") is False and
                value.get("mass_eligible") is False and
                value.get("deployment_allowed") is False and
                value.get("automatic_activation") is False and
                value.get("run_count") == 43 and value.get("frame_count") == 864 and
                len(value.get("native_run_manifest_sha256") or {}) == 43 and
                len(value.get("map_sha256") or {}) == 864 and
                set(value.get("qa_evidence_sha256") or {}) == expected_qa):
            return invalid
        if rerun:
            from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import RECEIPT_FILE
            if sha256_file(model / RECEIPT_FILE) != value.get("inference_rerun_receipt_sha256"):
                return invalid
            if value.get("registration_kind") == "inference_only_display_modes_v2":
                qa_report = _read(resolve_review_path(value["qa_directory"]) / "acceptance_report.json")
                if (qa_report.get("contract") != DISPLAY_MODES_V2_ACCEPTANCE_CONTRACT or
                        qa_report.get("model_id") != model.name or
                        qa_report.get("model_sha256") != value["model_sha256"] or
                        set(qa_report.get("section4") or {}) != DISPLAY_MODES_V2_CHECKS or
                        qa_report.get("acceptance_checks") != value.get("acceptance_checks")):
                    return invalid
                for evidence in (qa_report.get("evidence") or {}).values():
                    if (not isinstance(evidence, dict) or
                            sha256_file(resolve_review_path(evidence["path"])) != evidence["sha256"]):
                        return invalid
        for key, relative in (
            ("model_summary_sha256", "summary.json"),
            ("model_request_sha256", "training_request.json"),
            ("native_source_manifest_sha256", "review_source/postflight_manifest.json"),
            ("validation_binding_sha256", "validation/validation_binding.json"),
            ("postflight_manifest_sha256", "validation/postflight_manifest.json"),
        ):
            if sha256_file(model / relative) != value.get(key):
                return invalid
        qa = resolve_review_path(value["qa_directory"])
        if any(sha256_file(qa / name) != digest for name, digest in
               (value.get("qa_evidence_sha256") or {}).items()):
            return invalid
        return str(value.get("qa_warning") or invalid)
    except (OSError, TypeError, ValueError, KeyError):
        return invalid


def inspect_candidate(model: Path, qa_dir: Path) -> dict:
    """Verify exact saved maps and the failed/incomplete QA evidence."""
    model, qa_dir = Path(model).resolve(), Path(qa_dir).resolve()
    summary = _read(model / "summary.json")
    identity = _model_identity(model)
    _require(summary.get("model_id") == summary.get("campaign_id") ==
             identity[0] == model.name, "Model identity differs from its folder")
    _require(summary.get("inventory", {}).get("datasets") == ["ffac"]
             and summary.get("regional_training_performed") is True
             and summary.get("regional_fit", {}).get("converged") is True,
             "Track-1 regional fit is incomplete")
    inspect_phase_model_bundle(model)
    request = _read(model / "training_request.json")
    _require((request.get("settings", {}).get("v5_raw_photometry") or {}).get(
        "provisional_studio_training") is True,
        "Candidate is not the source-bound provisional RAW Track-1 fit")
    preparation = _read(model / "studio_preparation.json")
    _require(preparation.get("status") == "ready" and
             preparation.get("runs") == preparation.get("runs_done") == 43 and
             preparation.get("presentation_ready") is True and
             preparation.get("partial") is False and
             preparation.get("model_identity") == list(identity),
             "Studio preparation is not complete for this exact model")
    binding = _read(model / "validation" / "validation_binding.json")
    manifest = _read(model / "validation" / "postflight_manifest.json")
    source_manifest = model / "review_source" / "postflight_manifest.json"
    source_sha = sha256_file(source_manifest)
    for value in (binding, manifest):
        _require((value.get("model_id"), value.get("model_sha256")) == identity
                 and value.get("source_manifest_sha256") == source_sha,
                 "Native map campaign is not bound to the exact model and source")
    _require(binding.get("run_count") == 43 and binding.get("completed_utc")
             and binding.get("output_mode") == "review",
             "Native review campaign is incomplete")
    _require(Path(binding.get("combined_root") or "").resolve() ==
             Path(preparation.get("validation") or "").resolve(),
             "Studio preparation selects a different map package")
    jobs = _source_jobs(model / "validation")
    _require(len(jobs) == 43 and len({job["run"] for job in jobs}) == 43 and
             all(job.get("dataset") == "ffac" for job in jobs),
             "Native map campaign must include 43 distinct FF_AC runs")
    _require(set(binding.get("selected_candidates") or {}) ==
             {job["run"] for job in jobs}, "Prepared run selection is incomplete")
    map_hashes: dict[str, str] = {}
    native_sources: dict[str, str] = {}
    frames = 0
    for job in jobs:
        run = job["run"]
        review = Path(job["review_output_dir"]).resolve()
        native = review / "native_frames" / run / "manifest.json"
        _require(native.is_file(), f"Native source manifest is missing: {run}")
        native_sources[run] = sha256_file(native)
        metadata = _read(native)
        rows = metadata.get("frame_sources") or []
        count = metadata.get("frame_count")
        _require(type(count) is int and count > 0 and len(rows) == count and
                 all(isinstance(row.get("source_sha256"), str) and
                     len(row["source_sha256"]) == 64 for row in rows),
                 f"Native source identities are incomplete: {run}")
        params = str(job.get("params_sha256") or "")
        maps = Path(job["phase_map_output_dir"]).resolve()
        _require(len(params) == 64 and
                 maps.is_relative_to((model / "validation").resolve()),
                 f"Phase map output is not inside this candidate: {run}")
        _require((job.get("quality_flags") or {}).get("mass_analysis_allowed") is False,
                 f"Mass-use prohibition is missing: {run}")
        for index in range(count):
            path = maps / "native_phase_maps" / run / f"A_{params[:16]}" / f"frame_{index:03d}.png"
            _require(path.is_file(), f"Native phase map missing: {run}/{index}")
            with Image.open(path) as image:
                image.verify()
            map_hashes[f"{run}/{index:03d}"] = sha256_file(path)
        frames += count
    _require(frames == 864 and len(map_hashes) == 864,
             "Track-1 result does not contain exactly 864 native phase maps")
    from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import RECEIPT_FILE
    if (model / RECEIPT_FILE).is_file():
        return _inspect_rerun_registration(model, qa_dir, identity, map_hashes,
            native_sources, source_sha, frames)
    qa_files = (
        "assessment.md",
        "source_exact_with_mismatched_historical_unscored.json",
        "weak_late_native_focus.json",
        "focus_optical_regional_parity.json",
    )
    qa_hashes = {name: sha256_file(qa_dir / name) for name in qa_files}
    source_exact = _read(qa_dir / qa_files[1])
    weak = _read(qa_dir / qa_files[2])
    focus = _read(qa_dir / qa_files[3])
    binding_sha = sha256_file(model / "validation" / "validation_binding.json")
    for report in (source_exact, weak, focus):
        _require(report.get("model_sha256") == identity[1] and
                 report.get("postflight_sha256") == sha256_file(
                     model / "validation" / "postflight_manifest.json") and
                 (report.get("validation_binding_sha256") in (None, binding_sha)) and
                 report.get("scientific_acceptance") is False,
                 "QA report does not bind the exact failed/incomplete native maps")
    _require(source_exact.get("validation_binding_sha256") == binding_sha and
             weak.get("validation_binding_sha256") == binding_sha,
             "Exact-source and weak/late QA must bind the validation campaign")
    _require(source_exact.get("candidate_id") == identity[0] and
             source_exact.get("independent_postfit_supplied") is False and
             weak.get("candidate_id") == identity[0] and
             weak.get("independent_postfit") is False and
             weak.get("scored_labels") == 165,
             "QA does not document the required weak/late and independent-label limits")
    return dict(
        format=FORMAT, created_utc=utc_now(), model_root=str(model),
        model_id=identity[0], model_sha256=identity[1], series="ffac",
        publication_scope="studio_validation_only", qa_status="fail_or_incomplete",
        qa_warning=WARNING, qa_directory=str(qa_dir), qa_evidence_sha256=qa_hashes,
        run_count=43, frame_count=frames, native_source_manifest_sha256=source_sha,
        native_run_manifest_sha256=native_sources,
        validation_binding_sha256=sha256_file(model / "validation" / "validation_binding.json"),
        postflight_manifest_sha256=sha256_file(model / "validation" / "postflight_manifest.json"),
        model_summary_sha256=sha256_file(model / "summary.json"),
        model_request_sha256=sha256_file(model / "training_request.json"),
        map_sha256=map_hashes, storage_scope="local_candidate_pointer_on_this_machine",
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False,
        automatic_activation=False, model_and_labels_changed=False,
    )


def _inspect_rerun_registration(model: Path, qa_dir: Path, identity: tuple,
                               map_hashes: dict, native_sources: dict,
                               source_sha: str, frames: int) -> dict:
    """Register rerun Section-4 evidence, including failures, for human review.

    The unchanged fit retains its existing phase/generalization warnings.  This
    variant does not manufacture the old phase-QA reports with new identities
    and does not require every display/mask report to pass before it is visible.
    """
    from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import (
        ACCEPTANCE_CONTRACT, QA_FILE, RECEIPT_FILE, verify_inference_rerun,
    )
    rerun = verify_inference_rerun(model)
    qa = _read(qa_dir / QA_FILE)
    binding_sha = sha256_file(model / "validation/validation_binding.json")
    postflight_sha = sha256_file(model / "validation/postflight_manifest.json")
    checks = qa.get("acceptance_checks") or {}
    is_v2 = rerun.get("isolated_invalid_policy") == "fill_isolated_invalid_v1"
    expected_contract = (DISPLAY_MODES_V2_ACCEPTANCE_CONTRACT if is_v2
                         else ACCEPTANCE_CONTRACT)
    _require(qa.get("contract") == expected_contract and
             qa.get("model_id") == identity[0] and
             qa.get("model_sha256") == identity[1] and
             qa.get("postflight_sha256") == postflight_sha and
             qa.get("validation_binding_sha256") == binding_sha and
             qa.get("scientific_acceptance") is False and
             set(checks) == {"sidecar", "display", "phase_maps", "studio"},
             "Rerun Section-4 report is not bound to these exact native maps")
    if is_v2:
        _require(set(qa.get("section4") or {}) == DISPLAY_MODES_V2_CHECKS,
                 "V2 report does not contain all seven requested Section-4 checks")
        evidence = qa.get("evidence") or {}
        _require(set(evidence) == {"sidecar", "display", "overlay", "phase_maps", "studio"},
                 "V2 Section-4 report lacks a source-bound QA evidence set")
        for name, item in evidence.items():
            _require(isinstance(item, dict) and isinstance(item.get("path"), str) and
                     isinstance(item.get("sha256"), str) and
                     sha256_file(resolve_review_path(item["path"])) == item["sha256"],
                     f"V2 {name} QA evidence changed or is missing")
    # Failed/pending checks stay in the receipt; registration is visibility,
    # never acceptance or automatic activation.
    warning = WARNING + (
        " This inference-only revision fills enclosed sand holes, repairs isolated "
        "invalid model-input pixels, and offers three fixed photographic display "
        "modes. It does not constitute new model training or independent scientific "
        "validation. Section-4 results are in the rerun report."
        if is_v2 else
        " This inference-only revision fills enclosed sand holes and repairs the "
        "photographic display; it does not constitute new model training or "
        "independent scientific validation. Section-4 results are in the rerun report."
    )
    return dict(
        format=FORMAT, created_utc=utc_now(), model_root=str(model),
        model_id=identity[0], model_sha256=identity[1], series="ffac",
        registration_kind=("inference_only_display_modes_v2" if is_v2 else
                           "inference_only_display_mask_rerun"),
        publication_scope="studio_validation_only", qa_status="fail_or_incomplete",
        qa_warning=warning, qa_directory=str(qa_dir),
        qa_evidence_sha256={QA_FILE: sha256_file(qa_dir / QA_FILE)},
        acceptance_checks=checks,
        inference_rerun_receipt_sha256=sha256_file(model / RECEIPT_FILE),
        inherited_training_candidate_id=Path(rerun["old_candidate"]).name,
        inherited_training_model_identity=rerun["old_model_identity"],
        run_count=43, frame_count=frames, native_source_manifest_sha256=source_sha,
        native_run_manifest_sha256=native_sources,
        validation_binding_sha256=binding_sha, postflight_manifest_sha256=postflight_sha,
        model_summary_sha256=sha256_file(model / "summary.json"),
        model_request_sha256=sha256_file(model / "training_request.json"),
        map_sha256=map_hashes, storage_scope="local_candidate_pointer_on_this_machine",
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False,
        automatic_activation=False, model_and_labels_changed=False,
        trained_model_weights_changed=False, optical_training_performed=False,
        regional_training_performed=False,
    )


def register(model: Path, qa_dir: Path, *, results: Path | None = None) -> Path:
    receipt = inspect_candidate(model, qa_dir)
    target = registration_path(receipt["model_id"], results=results)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Immutable Studio registration already exists: {target}")
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    from ..phase_model_library import index_registered_model
    try:
        index_registered_model(model, results=target.parent.parent)
    except (OSError, ValueError, TypeError) as error:
        # The saved registration is complete. A separate index failure must not
        # revoke it or substitute another model; legacy discovery still works.
        import warnings
        warnings.warn(f"Phase library index was not updated: {error}", RuntimeWarning)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    print(register(args.model, args.qa_dir, results=args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
