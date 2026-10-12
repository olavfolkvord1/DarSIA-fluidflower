"""Register approved G3 saved maps for Studio, without activating a model.

Production parity is a technical assertion. It does not establish mass
eligibility, physical scale, or independence of the fixed CO2 classifier.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from PIL import Image

from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path

KIND = "approved_g3_downstream"
FORMAT = "ffac-track1-studio-validation-registration-v1"
SPEC_SHA = "9d773b61c370dfa2e7f1116c4b858ea143ef8545581847810c52a96bf19f1bc9"
G3_SHA = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
REGIONAL_SHA = "d33b3eea47392268d3ddebe2d9e9c695bfc640eb6e252d83670a3eaf3e7428be"
CONFIG = dict(regional_threshold=0.6, pixel_threshold=0.7,
              minimum_area_mm2=2.5, connectivity=4)
WARNING = (
    "Studio review candidate: approved G3 gas configuration (regional refit 0.6, "
    "pixel 0.7, minimum 2.5 mm²). Production parity passed; the CO2 union is "
    "unchanged. Physical scale and mass calibration remain unverified. "
    "Not for mass analysis or deployment."
)
INVALID = "QA FAILED: approved G3 Studio registration has changed or is incomplete; do not use for mass analysis or deployment."
FILES = {
    "model_summary_sha256": "summary.json",
    "model_request_sha256": "training_request.json",
    "native_source_manifest_sha256": "review_source/postflight_manifest.json",
    "validation_binding_sha256": "validation/validation_binding.json",
    "postflight_manifest_sha256": "validation/postflight_manifest.json",
    "production_receipt_sha256": "production_candidate_receipt.json",
    "phase_profile_sha256": "phase_optical_regions.json",
    "g3_model_sha256": "models/g3/gas_head.json",
    "regional_model_sha256": "regional_model.json",
    "co2_model_sha256": "models/generic/co2_presence_signal.json",
}
QA_FILES = ("parity_labelled_frames.json", "co2_union_identity.json")
QA_FRAMES = {
    "ac39": (1, 2, 3, 4, 6, 7, 9, 10, 12, 14, 15),
    "ac50": (1, 2, 3, 7, 12, 13, 16), "ac31": (1, 2, 3, 9, 10),
    "ac48": (3, 7, 9), "ac53": (3, 7, 9),
    "ac35": (3, 4, 5, 7, 9), "ac28": (8,),
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reports(model: Path) -> tuple[dict, dict]:
    reports = [_read(model / "production_qa" / name) for name in QA_FILES]
    for report in reports:
        _require(report.get("configuration") == CONFIG and
                 report.get("spec_sha256") == SPEC_SHA and
                 report.get("g3_sha256") == G3_SHA and
                 report.get("regional_refit_sha256") == REGIONAL_SHA,
                 "Approved gas configuration or immutable head/spec binding differs")
    parity, union = reports
    _require(parity.get("contract") == "ffac-g3-approved-labelled-parity-v1",
             "Wrong labelled parity contract")
    rows = parity.get("rows") or []
    keys = {(row.get("run"), row.get("frame_index")) for row in rows}
    expected = {(run, index) for run, indices in QA_FRAMES.items() for index in indices}
    _require(len(rows) == len(keys) == 35 and keys == expected and
             parity.get("all_pass") is True,
             "The exact 35 specified labelled/QA frames must have passing parity rows")
    for row in rows:
        score = row.get("agreement_inside_co2")
        _require(isinstance(score, (int, float)) and .999 <= score <= 1.,
                 "Production/literal agreement is below 99.9%")
        for key in ("production_gas_area_mm2", "literal_gas_area_mm2"):
            area = row.get(key)
            _require(isinstance(area, (int, float)) and 0 <= area < float("inf"),
                     "Gas areas are missing or invalid")
    _require(union.get("contract") == "ffac-g3-approved-co2-union-identity-v1",
             "Wrong CO2 identity contract")
    rows = union.get("rows") or []
    keys = {(row.get("run"), row.get("frame_index")) for row in rows}
    _require(len(rows) == len(keys) == 864 and len({key[0] for key in keys}) == 43,
             "CO2 identity does not cover all 43 runs / 864 frames")
    _require(all(row.get("exact") is True and
                 isinstance(row.get("export_co2_sha256"), str) and
                 len(row["export_co2_sha256"]) == 64 and
                 row["export_co2_sha256"] == row.get("production_co2_sha256")
                 for row in rows), "CO2 union is not bit-identical for every frame")
    return parity, union


def inspect_candidate(model: Path) -> dict:
    """Inspect actual prepared PNGs, their source bindings and parity evidence."""
    model = resolve_review_path(model).resolve()
    identity = _model_identity(model)
    summary = _read(model / "summary.json")
    _require(identity[0] == summary.get("campaign_id") == model.name,
             "Candidate identity differs from its folder")
    _require(summary.get("mass_eligible") is False and
             summary.get("scientific_acceptance") is False,
             "Review candidate must preserve mass/scientific restrictions")
    parity, union = _reports(model)
    _require(sha256_file(model / "models/g3/gas_head.json") == G3_SHA and
             sha256_file(model / "regional_model.json") == REGIONAL_SHA,
             "Actual G3/regional model files differ from the approved immutable heads")
    completed = _read(model / "production_candidate_receipt.json")
    _require(completed.get("complete") is True and completed.get("run_count") == 43 and
             completed.get("frame_count") == 864 and completed.get("model_identity") == list(identity),
             "Completed production receipt does not bind all43/864 maps")
    for report in (parity, union):
        _require((report.get("model_id"), report.get("model_sha256")) == identity,
                 "Production QA identifies another model")
    prep = _read(model / "studio_preparation.json")
    _require(prep.get("status") == "ready" and prep.get("runs") ==
             prep.get("runs_done") == 43 and prep.get("partial") is False and
             prep.get("presentation_ready") is True and
             prep.get("model_identity") == list(identity),
             "Studio preparation is incomplete or binds another model")
    binding = _read(model / "validation/validation_binding.json")
    manifest = _read(model / "validation/postflight_manifest.json")
    source_sha = sha256_file(model / "review_source/postflight_manifest.json")
    for value in (binding, manifest):
        _require((value.get("model_id"), value.get("model_sha256")) == identity and
                 value.get("source_manifest_sha256") == source_sha,
                 "Prepared campaign differs from model/source identity")
    _require(binding.get("run_count") == 43 and binding.get("output_mode") == "review",
             "Native campaign is not complete")
    _require(resolve_review_path(binding.get("combined_root") or "").resolve() ==
             resolve_review_path(prep.get("validation") or "").resolve(),
             "Studio selects a different map package")
    jobs = _source_jobs(model / "validation")
    _require(len(jobs) == 43 and all(job.get("dataset") == "ffac" for job in jobs),
             "Expected 43 distinct FF_AC source jobs")
    _require(set(binding.get("selected_candidates") or {}) == {j["run"] for j in jobs},
             "Prepared run selection is incomplete")
    maps, native_sources = {}, {}
    for job in jobs:
        run = job["run"]
        native = resolve_review_path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        native_sources[run] = sha256_file(native)
        metadata = _read(native)
        count = metadata.get("frame_count")
        sources = metadata.get("frame_sources") or []
        _require(type(count) is int and count > 0 and len(sources) == count and
                 all(isinstance(row.get("source_sha256"), str) and
                     len(row["source_sha256"]) == 64 for row in sources),
                 f"Missing native image source binding: {run}")
        params = str(job.get("params_sha256") or "")
        output = resolve_review_path(job["phase_map_output_dir"]).resolve()
        _require(len(params) == 64 and output.is_relative_to(model / "validation") and
                 (job.get("quality_flags") or {}).get("mass_analysis_allowed") is False,
                 f"Invalid saved map scope or mass policy: {run}")
        for index in range(count):
            path = output / "native_phase_maps" / run / f"A_{params[:16]}" / f"frame_{index:03d}.png"
            with Image.open(path) as image:
                _require(image.mode == "RGBA" and
                         image.size == tuple(metadata["canvas_wh"]),
                         f"Wrong Studio overlay encoding or canvas dimensions: {run}/{index}")
                image.verify()
            maps[f"{run}/{index:03d}"] = sha256_file(path)
    _require(len(maps) == 864, "Expected exactly 864 saved native maps")
    for row in union["rows"]:
        key = f"{row['run']}/{row['frame_index']:03d}"
        _require(maps.get(key) == row.get("studio_map_sha256"),
                 f"CO2 identity report binds a different Studio overlay: {key}")
        proof = resolve_review_path(row["map_path"]).resolve()
        _require(proof.is_relative_to(model / "production_labels") and
                 sha256_file(proof) == row.get("map_sha256"),
                 f"CO2 identity report binds a different native categorical map: {key}")
        with Image.open(proof) as image:
            _require(image.mode in ("L", "P") and image.size == (5472, 2979),
                     f"Wrong native categorical map dimensions: {key}")
            image.verify()
    return dict(
        format=FORMAT, registration_kind=KIND, created_utc=utc_now(),
        model_root=str(model), model_id=identity[0], model_sha256=identity[1],
        series="ffac", publication_scope="studio_validation_only",
        qa_status="passed_production_parity", qa_warning=WARNING,
        configuration=CONFIG, spec_sha256=SPEC_SHA, g3_sha256=G3_SHA,
        regional_refit_sha256=REGIONAL_SHA,
        run_count=43, frame_count=864, map_sha256=maps,
        native_run_manifest_sha256=native_sources,
        qa_directory=str(model / "production_qa"),
        qa_evidence_sha256={name: sha256_file(model / "production_qa" / name) for name in QA_FILES},
        **{key: sha256_file(model / relative) for key, relative in FILES.items()},
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False,
        automatic_activation=False, optical_training_performed=False,
        labels_changed=False, co2_union_changed=False,
    )


def registered_warning(model: Path, value: dict) -> str:
    """Check the immutable metadata chain on opening; do not rehash 864 PNGs."""
    try:
        _require(value.get("format") == FORMAT and value.get("registration_kind") == KIND and
                 resolve_review_path(value.get("model_root") or "") == model and
                 (value.get("model_id"), value.get("model_sha256")) == _model_identity(model) and
                 value.get("publication_scope") == "studio_validation_only" and
                 value.get("qa_status") == "passed_production_parity" and
                 value.get("configuration") == CONFIG and value.get("spec_sha256") == SPEC_SHA and
                 value.get("g3_sha256") == G3_SHA and value.get("regional_refit_sha256") == REGIONAL_SHA and
                 value.get("run_count") == 43 and value.get("frame_count") == 864 and
                 len(value.get("map_sha256") or {}) == 864 and
                 len(value.get("native_run_manifest_sha256") or {}) == 43 and
                 all(value.get(key) is False for key in
                     ("scientific_acceptance", "deployment_allowed", "mass_eligible", "automatic_activation")),
                 "Registration identity or policy changed")
        for key, relative in FILES.items():
            _require(sha256_file(model / relative) == value.get(key), "Candidate evidence changed")
        qa = resolve_review_path(value["qa_directory"])
        _require(set(value.get("qa_evidence_sha256") or {}) == set(QA_FILES), "QA evidence incomplete")
        for name, digest in value["qa_evidence_sha256"].items():
            _require(sha256_file(qa / name) == digest, "QA evidence changed")
        _reports(model)
        return WARNING
    except (OSError, ValueError, TypeError, KeyError):
        return INVALID


def saved_review_candidate(selection_dir: Path, job: dict):
    """Adapt registered saved maps with their native timeline, without mass files."""
    from darsia.fluidflower.tools.calibration_pairwise_review import Candidate
    from darsia.fluidflower.tools.register_track1_studio_candidate import (
        registration_path, registration_warning,
    )
    selection_dir = resolve_review_path(selection_dir).resolve()
    def path(value):
        value = Path(value)
        return resolve_review_path(value if value.is_absolute() else selection_dir / value).resolve()
    def same_directory(left, right):
        try:
            return Path(left).samefile(right)
        except OSError:
            return False
    root = path(job["candidate_folder"])
    warning = registration_warning(root)
    seam_registration = None
    combined_registration = None
    if warning.startswith("Studio review candidate:"):
        value = _read(registration_path(root.name))
        if value.get("registration_kind") == "ffac_g3_t4_floor_review":
            from darsia.fluidflower.phase_ffac_t4_review import t4_saved_review
            checked = t4_saved_review(root)
            _require(checked["pipeline_sha256"] == job.get("params_sha256"),
                     "T4 saved job differs from its sealed epoch-specific training revision")
            combined_registration = checked["registration"]
        elif value.get("registration_kind") == "ffac_l1_neighbourhood_floor_review":
            from darsia.fluidflower.phase_ffac_l1_neighbourhood_review import l1_neighbourhood_saved_review
            checked = l1_neighbourhood_saved_review(root)
            _require(checked["pipeline_sha256"] == job.get("params_sha256"),
                     "L1-neighbourhood saved job differs from its sealed post-processing revision")
            combined_registration = checked["registration"]
        elif value.get("registration_kind") == "ffac_g3_t3_floor_review":
            from darsia.fluidflower.phase_ffac_t3_review import t3_saved_review
            checked = t3_saved_review(root)
            _require(checked["pipeline_sha256"] == job.get("params_sha256"),
                     "T3 saved job differs from its sealed training revision")
            combined_registration = checked["registration"]
    if warning.startswith("Studio review candidate:") and not warning.startswith("Studio review candidate: approved G3"):
        value = _read(registration_path(root.name))
        if value.get("registration_kind") == "approved_g3_old_domain_seam_revision":
            seam_registration = value
        elif value.get("registration_kind") == "ffac_g3_t2_l1zone_review":
            from darsia.fluidflower.phase_ffac_t2_l1zone_review import combined_saved_review
            combined = combined_saved_review(root)
            _require(combined["pipeline_sha256"] == job.get("params_sha256"),
                     "Combined saved job differs from its sealed composition pipeline")
            combined_registration = combined["registration"]
    _require(warning.startswith("Studio review candidate: approved G3") or seam_registration is not None or combined_registration is not None,
             "Approved saved review requires an intact approved G3 registration")
    identity = _model_identity(root)
    if seam_registration is not None:
        seal = _read(root / "rig_camera_candidate_receipt.json")
        plan = seal.get("old_domain_seam_plan") or {}
        # The dispatcher has just hashed the registered metadata and bound plan.
        # Read its sealed pipeline instead of parsing the large source manifest
        # again for each experiment in the catalog.
        _require(seal.get("complete") is True and seal.get("model_identity") == list(identity)
                 and seal.get("pipeline_sha256") == job.get("params_sha256")
                 and same_directory(path(plan.get("path") or ""), root / "old_domain_seam_plan.json")
                 and plan.get("sha256") == seam_registration.get("source_plan_sha256")
                 and seal.get("mass_eligible") is False and seal.get("bottom_mass_allowed") is False,
                 "Saved seam maps differ from their sealed review plan or mass restrictions")
    binding = _read(root / "validation/validation_binding.json")
    manifest = _read(root / "validation/postflight_manifest.json")
    run = str(job["run"]).casefold()
    flags = job.get("quality_flags") or {}
    _require(flags.get("approved_g3_downstream") is True and
             flags.get("mass_analysis_allowed") is False and
             flags.get("scientific_acceptance") is False and
             job.get("output_mode") == "review" and job.get("phase_model_only") is True,
             "Approved saved maps require review-only scope and unchanged mass restrictions")
    _require((binding.get("model_id"), binding.get("model_sha256")) == identity and
             (manifest.get("model_id"), manifest.get("model_sha256")) == identity and
             binding.get("pipeline_sha256") == manifest.get("pipeline_sha256") == job.get("params_sha256") and
             job.get("optical_model_sha256") == identity[1] and
             (binding.get("selected_candidates") or {}).get(run) == job.get("candidate_id"),
             "Approved saved maps identify another model, pipeline or selected campaign")
    matches = [value for value in manifest["jobs"] if str(value["run"]).casefold() == run]
    _require(len(matches) == 1 and matches[0] == job,
             "Approved saved candidate differs from the immutable prepared job")
    selected_binding = _read(selection_dir / "validation_binding.json")
    selected_manifest = _read(selection_dir / "postflight_manifest.json")
    if same_directory(selection_dir, path(binding["combined_root"])) or same_directory(selection_dir, root / "validation"):
        _require(selected_binding == binding and selected_manifest == manifest and
                 _read(path(binding["combined_root"]) / "validation_binding.json") == binding and
                 _read(path(binding["combined_root"]) / "postflight_manifest.json") == manifest,
                 "Approved review publication pointer differs from its prepared metadata")
    else:
        package = root / "validation/run_packages" / run
        source_sha = sha256_file(root / "review_source" / run / "postflight_manifest.json")
        _require(same_directory(selection_dir, package) and
                 same_directory(path(selected_binding["combined_root"]), package) and
                 selected_manifest.get("jobs") == [job] and
                 selected_binding.get("selected_candidates") == {run:job["candidate_id"]} and
                 all(value.get("model_id") == identity[0] and value.get("model_sha256") == identity[1] and
                     value.get("pipeline_sha256") == binding["pipeline_sha256"] and
                     value.get("physics_runtime") == binding.get("physics_runtime") and
                     value.get("output_mode") == "review" and value.get("source_manifest_sha256") == source_sha
                     for value in (selected_binding, selected_manifest)),
                 "Approved per-run review package differs from its bound model, job or native source")
    review, maps = path(job["review_output_dir"]), path(job["phase_map_output_dir"])
    _require(review.is_dir() and maps.is_dir() and maps.is_relative_to(root / "validation"),
             "Approved saved native source or phase map folder is absent or unbound")
    native = review / "native_frames" / run / "manifest.json"
    registration = (combined_registration if combined_registration is not None else
                    seam_registration if seam_registration is not None else _read(registration_path(identity[0])))
    _require(sha256_file(native) == (registration.get("native_run_manifest_sha256") or {}).get(run),
             "Approved saved native timeline differs from its registration")
    source = _read(native)
    count = source.get("frame_count")
    _require(type(count) is int and count > 0 and len(source.get("frame_sources") or []) == count and
             source.get("dataset") == job.get("dataset") == "ffac" and source.get("run") == run,
             "Approved native timeline has an invalid run, series or frame count")
    return Candidate(run=run, role=job["role"], family=job.get("family", "unknown"),
        variant=job["variant"], candidate_id=job["candidate_id"], grid_order=job.get("grid_order", 1),
        params_sha256=job["params_sha256"], normalized_mae=None,
        mass_plot=root / "calibration_mass_fit.png", contour_movie=root / "calibration_review.mp4",
        per_frame_csv=native, summary_path=root / "summary.json", plot_seconds=0., image_seconds=0., fps=0.,
        frame_count=count, quality_flags=flags, dataset=job["dataset"], review_output_dir=review,
        phase_map_output_dir=maps, phase_model_only=True, optical_model_sha256=job["optical_model_sha256"],
        raw_phase_map_output_dir=None, presence_diagnostics_dir=None,
        phase_support_format=job.get("phase_support_format", ""))


def register(model: Path, *, results: Path | None = None) -> Path:
    from darsia.fluidflower.tools.register_track1_studio_candidate import registration_path
    receipt = inspect_candidate(model)
    target = registration_path(receipt["model_id"], results=results)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    print(register(args.model, results=args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
