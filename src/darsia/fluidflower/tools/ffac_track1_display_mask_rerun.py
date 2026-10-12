"""Prepare a detached Track1 inference revision without changing trained artifacts.

The frozen training request continues to identify the original training images.
An explicit second receipt binds the new *inference* images, filled sand support,
and photographic display.  No extraction, optimization or map worker is launched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.train_phase_model_further import native_source


CONTRACT = "ffac-track1-display-mask-inference-rerun-v1"
ACCEPTANCE_CONTRACT = "ffac-track1-display-mask-rerun-acceptance-v1"
RECEIPT_FILE = "inference_rerun_receipt.json"
QA_FILE = "acceptance_report.json"
PRIORITY = {"ac28": 0, "ac45": 1, "ac56": 2, "ac22": 3}
RUN_COUNT = 43
FRAME_COUNT = 864


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _bound(path: Any, digest: Any, label: str) -> Path:
    _require(isinstance(path, str) and isinstance(digest, str) and len(digest) == 64,
             f"{label} lacks its immutable path and SHA")
    result = Path(path).resolve()
    _require(result.is_file() and sha256_file(result) == digest,
             f"{label} changed or is absent: {result}")
    return result


def _copied_files(old: Path, output: Path) -> dict[str, str]:
    """Copy small scientific lineage, never the old maps or derived caches."""
    mandatory = ("training_request.json", "regional_model.json")
    optional = ("evidence.json", "extraction.json", "regional_training_protocol.json",
                "region_inventory.csv", "expected_review_runs.json")
    for name in mandatory:
        _require((old / name).is_file(), f"Frozen candidate is missing {name}")
    files = [old / name for name in (*mandatory, *optional) if (old / name).is_file()]
    for directory in ("models", "input_snapshot", "parent_provenance", "registered_context"):
        _require((old / directory).is_dir(), f"Frozen candidate is missing {directory}")
        files.extend(path for path in (old / directory).rglob("*") if path.is_file())
    copied = {}
    for source in files:
        relative = source.relative_to(old)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256_file(source)
        _require(sha256_file(target) == digest, f"Copied artifact differs: {relative}")
        copied[relative.as_posix()] = digest
    return copied


def _inference_rows(batch: dict) -> list[dict]:
    rows = batch.get("canonical_mapping") or batch.get("revisions") or []
    _require(isinstance(rows, list) and len(rows) == RUN_COUNT and
             len({str(row.get("run") or "").casefold() for row in rows}) == RUN_COUNT,
             "Inference batch must bind 43 unique current runs")
    return rows


def prepare_inference_rerun(
    old_candidate: Path, materialization_receipt: Path, output: Path, *,
    optical_checkpoint: Path | None = None,
    frozen_regional_model: Path | None = None,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    """Clone the frozen fit and bind a new 43/864 source-only inference batch.

    The old training request is copied byte-for-byte, including its old training
    sources.  The new source binding is only in inference_rerun_receipt and the
    per-run review jobs/nominal context.  Call the ordinary --resume map runner
    on this fresh candidate after preparation.
    """
    old = Path(old_candidate).resolve()
    batch_path = Path(materialization_receipt).resolve()
    output = Path(output).resolve()
    _require(old.is_dir() and batch_path.is_file(), "Frozen candidate or new batch is absent")
    _require(not output.exists() and not output.is_relative_to(old)
             and not old.is_relative_to(output), "Use an absent detached candidate folder")
    # The ordinary Studio runner recognizes the inference receipt in this
    # candidate itself; its folder need not pretend to be a new training job.
    launch_path = output / "inference_launch_request.json"
    request = _read(old / "training_request.json")
    summary = _read(old / "summary.json")
    profile = _read(old / "phase_optical_regions.json")
    learned = _read(old / "regional_model.json")
    _require((summary.get("inventory") or {}).get("datasets") == ["ffac"] and
             (summary.get("regional_fit") or {}).get("converged") is True and
             (learned.get("fit_report") or {}).get("converged") is True and
             (request.get("settings", {}).get("v5_raw_photometry") or {}).get(
                 "provisional_studio_training") is True and
             (profile.get("v5_track1_review_scope") or {}).get(
                 "mass_analysis_allowed") is False,
             "Source candidate is not the converged provisional Track1 fit")
    old_jobs = _read(old / "review_source/postflight_manifest.json")["jobs"]
    _require(len(old_jobs) == RUN_COUNT and
             len({row["run"].casefold() for row in old_jobs}) == RUN_COUNT,
             "Frozen candidate does not contain 43 unique current runs")
    old_by_run = {job["run"].casefold(): job for job in old_jobs}
    batch = _read(batch_path)
    rows = _inference_rows(batch)
    _require({row["run"].casefold() for row in rows} == set(old_by_run),
             "Inference batch has a different current run set")
    recovery_path = old.parent.parent / "recovery_receipt.json"
    recovery = _read(recovery_path) if recovery_path.is_file() else {}
    if optical_checkpoint is None and recovery.get("optical_job"):
        optical_checkpoint = Path(recovery["optical_job"]) / "optical_checkpoint.json"
    if frozen_regional_model is None and recovery.get("regional_fit_inputs"):
        frozen_regional_model = Path(recovery["regional_fit_inputs"]["fit_path"])
    lineage = {}
    for label, path, expected in (
        ("optical_checkpoint", optical_checkpoint, recovery.get("optical_checkpoint_sha256")),
        ("frozen_regional_model", frozen_regional_model,
         (recovery.get("regional_fit_inputs") or {}).get("fit_sha256")),
    ):
        if path is not None:
            source = Path(path).resolve()
            _require(source.is_file(), f"Frozen {label} is absent")
            digest = sha256_file(source)
            _require(expected is None or digest == expected, f"Frozen {label} changed")
            lineage[label] = {"path": str(source), "sha256": digest}
    if frozen_regional_model is not None:
        fitted = _read(Path(frozen_regional_model))
        without_metadata = {key: value for key, value in learned.items()
                            if key not in {"v5_chemistry_fit_scope", "v5_review_fit_partition"}}
        _require(without_metadata == fitted, "Regional coefficients differ from the frozen fit")
    inferred_jobs, source_bindings = [], []
    frame_count = 0
    new_profile = deepcopy(profile)
    revisions = {str(row.get("run") or "").casefold(): row
                 for row in batch.get("revisions") or []}
    for row in rows:
        run = row["run"].casefold()
        new_path = _bound(row.get("materialized_manifest_path"),
                          row.get("materialized_manifest_sha256"), f"{run} new manifest")
        new_native = _read(new_path)
        new_review = Path(row.get("materialized_review_root") or
                          new_path.parent.parent.parent).resolve()
        _require(new_path == new_review / "native_frames" / run / "manifest.json",
                 f"{run} new native manifest is outside its review root")
        original_job = _read(old / "review_source" / run / "original_job.json")
        _old_root, old_path, old_native = native_source(original_job)
        old_marker = old_native.get("v5_track1_photo_geometry") or {}
        new_marker = new_native.get("v5_track1_photo_geometry") or {}
        photometry = new_native.get("v5_photometry") or {}
        processor = photometry.get("processor") or {}
        old_processor = (old_native.get("v5_photometry") or {}).get("processor") or {}
        # The filled support and display are the only intentional changes.
        # Pin the fitted normalization and source-photo geometry independently
        # of module hashes that legitimately change for the new mask/display.
        for key in ("calibration_manifest_sha256", "photo_source_aggregate_sha256"):
            if old_processor.get(key) is not None:
                _require(processor.get(key) == old_processor[key],
                         f"{run} fitted normalization or source-photo aggregate changed")
        _require(new_native.get("run") == run and
                 processor.get("enclosed_hole_policy") == "fill_enclosed_v1" and
                 processor.get("display_mode") == "photographic_v1" and
                 processor.get("enclosed_hole_max_area_px") == 60000 and
                 processor.get("enclosed_hole_vertical_gap_max_px") == 64 and
                 photometry.get("display_jpeg_is_training_input") is False and
                 photometry.get("unit_denominator") == 65535 and
                 new_marker.get("enclosed_hole_policy") == "fill_enclosed_v1" and
                 new_marker.get("vertical_gap_max_length_px") == 64 and
                 new_marker.get("vertical_gap_rule") ==
                     "same_column_supported_above_and_below_v1" and
                 new_marker.get("original_source_manifest_sha256") ==
                     old_marker.get("original_source_manifest_sha256") and
                 new_marker.get("frames") == old_marker.get("frames"),
                 f"{run} source, frame geometry or filled photographic policy changed")
        for name in ("frame_count", "analysis_image_shape_hw", "canvas_wh", "content_box_xywh"):
            _require(new_native.get(name) == old_native.get(name),
                     f"{run} native grid or frame count changed")
        old_frames, new_frames = old_native["frame_sources"], new_native["frame_sources"]
        _require(len(old_frames) == len(new_frames) == new_native["frame_count"],
                 f"{run} native timeline is incomplete")
        for before, after in zip(old_frames, new_frames):
            for key in ("frame_index", "elapsed_h", "source_sha256", "source_raw_sha256",
                        "v5_raw_sha256", "v5_raw_members"):
                _require(before.get(key) == after.get(key),
                         f"{run} original acquisition or timestamp changed: {key}")
        revision = revisions.get(run, row)
        if revision.get("receipt_path"):
            _bound(revision["receipt_path"], revision.get("receipt_sha256"),
                   f"{run} materialization receipt")
        entry = new_profile["runs"][f"ffac/{run}"]
        _bound(str(old / entry["path"]), entry["sha256"], f"{run} nominal context")
        entry["native_manifest_sha256"] = row["materialized_manifest_sha256"]
        provenance = entry.get("provenance") or {}
        photo_binding = provenance["v5_track1_photo_geometry"]
        photo_binding["native_manifest_sha256"] = row["materialized_manifest_sha256"]
        photo_binding["enclosed_hole_policy"] = "fill_enclosed_v1"
        photo_binding["vertical_gap_max_length_px"] = 64
        photo_binding["vertical_gap_rule"] = "same_column_supported_above_and_below_v1"
        provenance["v5_raw_photometry"] = dict(
            provenance.get("v5_raw_photometry") or {},
            receipt_path=str(batch_path), receipt_sha256=sha256_file(batch_path),
            materialized_manifest_sha256=row["materialized_manifest_sha256"],
            source_manifest_sha256=new_marker["original_source_manifest_sha256"],
            physical_grid_inherited=False,
        )
        # The geometry arrays and nominal axes remain byte-identical. Only
        # manifest identity is rebound; no physical QA is implied.
        entry["provenance"] = provenance
        def rebind(job: dict) -> dict:
            job = deepcopy(job)
            job.update(review_output_dir=str(new_review),
                       analysis_source_review_root=str(new_review))
            job["v5_raw_photometry"] = dict(job.get("v5_raw_photometry") or {},
                batch_receipt_path=str(batch_path), batch_receipt_sha256=sha256_file(batch_path),
                source_manifest_sha256=new_marker["original_source_manifest_sha256"],
                materialized_manifest_sha256=row["materialized_manifest_sha256"],
                scientific_input="sha-bound-float32-log-camera-linear-sidecar",
                display_jpeg_is_training_input=False)
            job["quality_flags"] = dict(job.get("quality_flags") or {},
                human_review_required=True, mass_analysis_allowed=False,
                inference_only_display_mask_revision=True)
            return job
        inferred_jobs.append((rebind(original_job), rebind(old_by_run[run])))
        source_bindings.append(dict(run=run, frame_count=len(new_frames),
            original_source_manifest_sha256=new_marker["original_source_manifest_sha256"],
            previous_materialized_manifest_path=str(old_path),
            previous_materialized_manifest_sha256=sha256_file(old_path),
            materialized_review_root=str(new_review),
            materialized_manifest_path=str(new_path),
            materialized_manifest_sha256=row["materialized_manifest_sha256"],
            frame_geometry_receipts_preserved=True,
            nominal_context_sha256=entry["sha256"]))
        frame_count += len(new_frames)
    _require(frame_count == FRAME_COUNT, "Inference batch must contain 864 current frames")
    output.mkdir(parents=True, exist_ok=False)
    copied = _copied_files(old, output)
    _new(output / "phase_optical_regions.json", new_profile)
    inferred_jobs.sort(key=lambda pair: (PRIORITY.get(pair[1]["run"].casefold(), 4),
                                         int(pair[1]["run"][2:])))
    for original_job, job in inferred_jobs:
        run = job["run"]
        _new(output / "review_source" / run / "original_job.json", original_job)
        _new(output / "review_source" / run / "postflight_manifest.json", {"jobs": [job]})
    _new(output / "review_source/postflight_manifest.json",
         {"jobs": [job for _original, job in inferred_jobs]})
    if not (output / "expected_review_runs.json").is_file():
        _new(output / "expected_review_runs.json", {"runs": [job["run"] for _, job in inferred_jobs]})
    now = utc_now()
    new_summary = dict(summary, model_id=output.name, campaign_id=output.name,
        created_utc=now, inference_revision_created_utc=now,
        inherited_training_candidate_id=old.name,
        training_performed_in_this_revision=False,
        inference_only_rerun=True,
        display_name="FF_AC Track1 · fotografisk visning + fylte sandhull · 30.09.2026",
        mass_eligible=False, scientific_acceptance=False)
    _new(output / "summary.json", new_summary)
    _new(launch_path, {"contract": CONTRACT, "output": str(output),
        "review_candidate_only": True, "inference_only": True,
        "optical_training_performed": False, "regional_training_performed": False,
        "created_utc": now})
    receipt = dict(contract=CONTRACT, created_utc=now, candidate=str(output),
        old_candidate=str(old), old_model_identity=list(_model_identity(old)),
        model_identity=list(_model_identity(output)),
        materialization_receipt_path=str(batch_path),
        materialization_receipt_sha256=sha256_file(batch_path),
        frozen_training_request_sha256=copied["training_request.json"],
        frozen_regional_candidate_sha256=copied["regional_model.json"],
        trained_artifacts_preserved=copied,
        **lineage,
        run_count=RUN_COUNT, frame_count=FRAME_COUNT, source_bindings=source_bindings,
        review_source_sha256=sha256_file(output / "review_source/postflight_manifest.json"),
        inference_profile_sha256=sha256_file(output / "phase_optical_regions.json"),
        stage="prepared",
        optical_training_performed=False, regional_training_performed=False,
        enclosed_hole_policy="fill_enclosed_v1", enclosed_hole_max_area_px=60000,
        vertical_gap_max_length_px=64,
        vertical_gap_rule="same_column_supported_above_and_below_v1",
        display_mode="photographic_v1", cpu_path_required=True,
        native_maps_built=False, mass_eligible=False, scientific_acceptance=False,
        old_candidate_unchanged=True)
    if spec_path is not None:
        spec = Path(spec_path).resolve()
        receipt["user_authorized_spec"] = {"path": str(spec), "sha256": sha256_file(spec)}
    _new(output / RECEIPT_FILE, receipt)
    return receipt


def verify_inference_rerun(model: Path) -> dict[str, Any]:
    """Verify immutable fit/source provenance without replaying model fitting."""
    model = Path(model).resolve()
    receipt = _read(model / RECEIPT_FILE)
    if receipt.get("contract") == "ffac-track1-incremental-inference-v1":
        from darsia.fluidflower.tools.ffac_track1_incremental_inference import verify_sealed_incremental
        return verify_sealed_incremental(model)
    _require(receipt.get("contract") == CONTRACT and
             Path(receipt.get("candidate") or "").resolve() == model and
             receipt.get("model_identity") == list(_model_identity(model)) and
             receipt.get("run_count") == RUN_COUNT and receipt.get("frame_count") == FRAME_COUNT and
             receipt.get("optical_training_performed") is False and
             receipt.get("regional_training_performed") is False and
             receipt.get("vertical_gap_max_length_px") == 64 and
             receipt.get("vertical_gap_rule") ==
                 "same_column_supported_above_and_below_v1" and
             receipt.get("mass_eligible") is False,
             "Inference rerun receipt is invalid")
    for relative, digest in receipt["trained_artifacts_preserved"].items():
        _bound(str(model / relative), digest, f"preserved {relative}")
    _bound(receipt["materialization_receipt_path"], receipt["materialization_receipt_sha256"],
           "inference materialization receipt")
    _require(sha256_file(model / "review_source/postflight_manifest.json") ==
                 receipt["review_source_sha256"] and
             sha256_file(model / "phase_optical_regions.json") == receipt["inference_profile_sha256"],
             "Inference source or nominal context binding changed")
    for binding in receipt["source_bindings"]:
        _bound(binding["materialized_manifest_path"], binding["materialized_manifest_sha256"],
               f"{binding['run']} inference native manifest")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-candidate", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optical-checkpoint", type=Path)
    parser.add_argument("--frozen-regional-model", type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    result = prepare_inference_rerun(args.old_candidate, args.materialization_receipt,
        args.output, optical_checkpoint=args.optical_checkpoint,
        frozen_regional_model=args.frozen_regional_model, spec_path=args.spec)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
