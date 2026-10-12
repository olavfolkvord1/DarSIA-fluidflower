"""Register Rule B maps for manual Studio review without activating a model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from PIL import Image

from darsia.fluidflower.phase_g3_production import CO2_SHA256
from darsia.fluidflower.phase_validation import _model_identity, _source_jobs
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.review_storage import resolve_review_path
from .register_ffac_g3_studio_candidate import CONFIG, FORMAT, G3_SHA, REGIONAL_SHA, _read, _require

KIND = "approved_g3_barrier_rule_b"
RULE_FORMAT = "ffac-gas-barrier-rule-b-v1"
SPEC_SHA = "41fa22e6bbf91b10086eaf9eb5a198039d4453c1687370338221ca010322a3f8"
QA_CONTRACT = "ffac-gas-barrier-rule-b-qa-v1"
WARNING = (
    "Studio review candidate: approved G3 gas configuration with Barrier rule B. "
    "Unsupported gas beyond an unbreached barrier becomes aqueous CO2; the CO2 union is unchanged. "
    "Physical scale and mass calibration remain unverified. Not for mass analysis or deployment."
)
INVALID = "QA FAILED: Barrier rule B Studio registration has changed or is incomplete; do not use for mass analysis or deployment."
FILES = {
    "model_summary_sha256": "summary.json",
    "model_request_sha256": "training_request.json",
    "native_source_manifest_sha256": "review_source/postflight_manifest.json",
    "validation_binding_sha256": "validation/validation_binding.json",
    "postflight_manifest_sha256": "validation/postflight_manifest.json",
    "production_receipt_sha256": "barrier_rule_b_candidate_receipt.json",
    "source_plan_sha256": "barrier_rule_b_candidate_plan.json",
    "phase_profile_sha256": "phase_optical_regions.json",
    "component_manifest_sha256": "gas_barrier_rule_b/manifest.json",
    "barrier_derivation_sha256": "gas_barrier_rule_b/barrier_derivation.json",
    "studio_preparation_sha256": "studio_preparation.json",
    "g3_model_sha256": "models/g3/gas_head.json",
    "regional_model_sha256": "regional_model.json",
    "co2_model_sha256": "models/generic/co2_presence_signal.json",
    "qa_evidence_sha256": "production_qa/barrier_rule_b_qa.json",
}


def _reports(model: Path) -> dict:
    report = _read(model / FILES["qa_evidence_sha256"])
    _require(report.get("contract") == QA_CONTRACT and report.get("all_pass") is True and
             report.get("configuration") == CONFIG and report.get("spec_sha256") == SPEC_SHA and
             report.get("g3_sha256") == G3_SHA and report.get("regional_refit_sha256") == REGIONAL_SHA and
             report.get("co2_sha256") == CO2_SHA256 and report.get("run_count") == 43 and
             report.get("frame_count") == 864, "Incomplete Rule B QA or changed approved heads")
    rows = report.get("rows") or []
    keys = {(row.get("run"), row.get("frame_index")) for row in rows}
    _require(len(rows) == len(keys) == 864 and len({run for run, _ in keys}) == 43,
             "Rule B QA must cover all 43 runs / 864 distinct frames")
    for row in rows:
        _require(row.get("co2_union_exact") is True and row.get("only_gas_to_aqueous") is True,
                 "Rule B changed the CO2 union or a decision other than gas to aqueous")
        before, after, removed = (row.get(key) for key in (
            "before_gas_pixels", "after_gas_pixels", "removed_gas_pixels"))
        _require(all(type(value) is int for value in (before, after, removed)) and
                 0 <= after <= before and removed == before - after,
                 "Invalid Rule B before/after gas counts")
        _require(isinstance(row.get("co2_sha256_before"), str) and
                 len(row["co2_sha256_before"]) == 64 and
                 row["co2_sha256_before"] == row.get("co2_sha256_after"),
                 "Rule B CO2 bit masks differ")
        area = row.get("pixel_area_mm2")
        _require(isinstance(area, (int, float)) and 0 < area < float("inf"), "Invalid nominal pixel scale")
        _require(isinstance(row.get("breached_components"), list) and
                 isinstance(row.get("bridging_co2_components"), list), "Missing Rule B breakthrough receipt")
    return report


def inspect_candidate(model: Path) -> dict:
    model = resolve_review_path(model).resolve()
    identity = _model_identity(model)
    summary = _read(model / "summary.json")
    approved = summary.get("approved_g3_production") or {}
    _require(identity[0] == summary.get("model_id") == summary.get("campaign_id") == model.name and
             summary.get("mass_eligible") is False and summary.get("scientific_acceptance") is False and
             (approved.get("barrier_rule") or {}).get("format") == RULE_FORMAT and
             (approved.get("domain_revision") or {}).get("format") == "ffac-photo-sand-top-revision-v1",
             "Rule B candidate identity, parent boundary or review scope differs")
    qa = _reports(model)
    _require(qa.get("model_identity") == list(identity), "Rule B QA binds another model")
    for relative, expected in (("models/g3/gas_head.json", G3_SHA),
                               ("regional_model.json", REGIONAL_SHA),
                               ("models/generic/co2_presence_signal.json", CO2_SHA256)):
        _require(sha256_file(model / relative) == expected, "Actual immutable G3, regional or CO2 head changed")
    receipt = _read(model / "barrier_rule_b_candidate_receipt.json")
    _require(receipt.get("complete") is True and receipt.get("run_count") == 43 and
             receipt.get("frame_count") == 864 and receipt.get("model_identity") == list(identity),
             "Rule B map batch is incomplete")
    _require(sha256_file(resolve_review_path(receipt["parent_registration"]["path"])) ==
             receipt["parent_registration"]["sha256"], "Top-surface parent registration changed")
    _require(qa["component_manifest_sha256"] == sha256_file(model / "gas_barrier_rule_b/manifest.json") and
             qa["barrier_derivation_sha256"] == sha256_file(model / "gas_barrier_rule_b/barrier_derivation.json"),
             "Rule B QA binds different components or barrier derivation")
    prep = _read(model / "studio_preparation.json")
    _require(prep.get("status") == "ready" and prep.get("runs") == prep.get("runs_done") == 43 and
             prep.get("presentation_ready") is True and prep.get("partial") is False and
             prep.get("model_identity") == list(identity), "Studio preparation is incomplete")
    binding = _read(model / "validation/validation_binding.json")
    manifest = _read(model / "validation/postflight_manifest.json")
    source_sha = sha256_file(model / "review_source/postflight_manifest.json")
    _require(all((value.get("model_id"), value.get("model_sha256")) == identity and
                 value.get("source_manifest_sha256") == source_sha for value in (binding, manifest)) and
             binding.get("run_count") == 43 and binding.get("output_mode") == "review",
             "Saved campaign is not bound to the Rule B candidate and exact sources")
    _require(resolve_review_path(binding["combined_root"]).samefile(resolve_review_path(prep["validation"])),
             "Studio selects a different Rule B map package")
    jobs = _source_jobs(model / "validation")
    _require(len(jobs) == 43 and len({job["run"] for job in jobs}) == 43 and
             set(binding["selected_candidates"]) == {job["run"] for job in jobs}, "Missing source runs")
    qa_rows = {(row["run"], row["frame_index"]): row for row in qa["rows"]}
    native_sources, maps = {}, {}
    for job in jobs:
        run = job["run"]
        _require(job.get("dataset") == "ffac" and
                 (job.get("quality_flags") or {}).get("mass_analysis_allowed") is False,
                 "Unexpected run or changed mass policy")
        native = resolve_review_path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
        metadata = _read(native)
        native_sources[run] = sha256_file(native)
        count = metadata.get("frame_count")
        _require(type(count) is int and count > 0 and len(metadata.get("frame_sources") or []) == count,
                 "Incomplete photographic timeline")
        output = resolve_review_path(job["phase_map_output_dir"]).resolve()
        params = job.get("params_sha256") or ""
        _require(output.is_relative_to(model / "validation") and len(params) == 64,
                 "Saved overlays are outside the new Rule B candidate")
        for index in range(count):
            row = qa_rows[(run, index)]
            path = output / "native_phase_maps" / run / f"A_{params[:16]}" / f"frame_{index:03d}.png"
            _require(sha256_file(path) == row["studio_map_sha256"] and
                     native_sources[run] == row["native_manifest_sha256"], "QA binds another map or photograph")
            with Image.open(path) as image:
                _require(image.mode == "RGBA" and image.size == tuple(metadata["canvas_wh"]), "Wrong Studio map encoding")
                image.verify()
            proof = resolve_review_path(row["map_path"]).resolve()
            _require(proof.is_relative_to(model / "production_labels") and
                     sha256_file(proof) == row["map_sha256"], "Native categorical proof differs")
            with Image.open(proof) as image:
                _require(image.mode in ("L", "P") and image.size == (5472, 2979), "Wrong native proof grid")
                image.verify()
            _require(sha256_file(resolve_review_path(row["components_path"])) == row["components_sha256"],
                     "Source-bound open-component map changed after QA")
            maps[f"{run}/{index:03d}"] = row["studio_map_sha256"]
    _require(len(maps) == 864, "Missing native Rule B maps")
    return dict(format=FORMAT, registration_kind=KIND, created_utc=utc_now(), model_root=str(model),
        model_id=identity[0], model_sha256=identity[1], series="ffac", publication_scope="studio_validation_only",
        qa_status="passed_barrier_rule_b", qa_warning=WARNING, configuration=CONFIG, spec_sha256=SPEC_SHA,
        g3_sha256=G3_SHA, regional_refit_sha256=REGIONAL_SHA, run_count=43, frame_count=864,
        map_sha256=maps, native_run_manifest_sha256=native_sources,
        **{key: sha256_file(model / relative) for key, relative in FILES.items()},
        scientific_acceptance=False, deployment_allowed=False, mass_eligible=False, automatic_activation=False,
        optical_training_performed=False, labels_changed=False, co2_union_changed=False,
        operation="gas-to-aqueous-after-approved-g3-top-surface")


def registered_warning(model: Path, value: dict) -> str:
    try:
        _require(value.get("format") == FORMAT and value.get("registration_kind") == KIND and
                 resolve_review_path(value["model_root"]).samefile(model) and
                 (value.get("model_id"), value.get("model_sha256")) == _model_identity(model) and
                 value.get("publication_scope") == "studio_validation_only" and
                 value.get("qa_status") == "passed_barrier_rule_b" and value.get("configuration") == CONFIG and
                 value.get("spec_sha256") == SPEC_SHA and value.get("run_count") == 43 and value.get("frame_count") == 864 and
                 len(value.get("map_sha256") or {}) == 864 and len(value.get("native_run_manifest_sha256") or {}) == 43 and
                 all(value.get(key) is False for key in (
                     "scientific_acceptance", "deployment_allowed", "mass_eligible", "automatic_activation", "co2_union_changed")),
                 "Rule B registration identity or review scope changed")
        for key, relative in FILES.items():
            _require(sha256_file(model / relative) == value.get(key), "Rule B evidence changed")
        _reports(model)
        return WARNING
    except (OSError, ValueError, TypeError, KeyError):
        return INVALID


def register(model: Path, *, results: Path | None = None) -> Path:
    from .register_track1_studio_candidate import registration_path
    value = inspect_candidate(model)
    target = registration_path(value["model_id"], results=results)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        if target.exists():
            raise FileExistsError(target)
        os.rename(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    print(register(args.model, results=args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
