"""Materialize source-identical FF_AC review images under a frozen photometry arm.

This command creates only corrected native inputs. It neither reads phase labels
for fitting nor trains a model. Each historical native revision retains its own
frame index, raw photo, timestamp and canvas geometry. A target warning is
recorded, never silently converted into an exclusion of difficult CO2 frames.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.tools.schema2_phase_transfer_campaign import (
    _array_sha256, _correction_diagnostics, _open_ffac, _plain_frame, _write_jpeg,
)
from darsia.fluidflower.phase_training.radiometry import analysis_review_root
from darsia.fluidflower.phase_training import trainer as training


FORMAT = "ffac-two-arm-native-materialization-v1"
SEQUENTIAL_CHART_EXTRACTOR_REVISION = "central-median-swatches-v1"
SEQUENTIAL_QA_REVISION = "post-common-chart-every-frame-v1"
AC47_V4_SOURCE_SHA256 = "12ab32e1593ec49460428f7ae380d9673c1f9d8c5530656e112cc328b52318be"
AC47_V4_RECEIPT_SHA256 = "1c93f303d84ffc58874965af5a53611817f28fb1cc1596b79c31cd76573d2841"
AC47_V4_CHANGED_FRAMES = [1, 3, 8, 13, 15]
AC61_PREPARED_BASELINE_SOURCE_SHA256 = "14bbf139f5cba7b449d2a6f6312a147785f467b8d2154ff83992783698e0dffe"
AC61_PREPARED_BASELINE_JPEG_SHA256 = "18d0ba62bd789c90cd36f317239bd7f438f96a50e50afda4d84df017858df244"
AC61_PREPARED_BASELINE_ARRAY_SHA256 = "36d171adfa57f295b13e85bcd9face9bfd4216e755a20bf49c6e7fd1b8bd8d8b"
AC61_PREPARED_BASELINE_REPAIR_SHA256 = "3243488b89a71dfc73663b91bebd0383263b96b25ba477b8cae52e6185a897a7"
AC61_PREPARED_BASELINE_MEMBERS = (
    ("DSC21674.JPG", "420bfa273bb83c8b8e1b97ad079b461336984d1af33e3d4caab89b3b2e5f8b4f"),
    ("DSC21675.JPG", "5c391817f1ef539328bf6fddcce4d796f2fdf8541d0f1e209847444be682593b"),
    ("DSC21676.JPG", "feae6b4de035bb19f2716d9bd4a3d40d7e8443dcd1d6200c8aaf979db236b985"),
)


def _source_photometry_revision(original: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any] | None:
    """Verify the one historical corrected-photo revision before replacing it."""
    source_id = original.get("photometry_revision_id")
    if source_id is None:
        return None
    if source_id == "ffac-two-arm-within-run-legacy-v2-v1":
        if revision["run"] != "ac47":
            raise ValueError("Unexpected chained photometry revision outside AC47")
        prior_path = Path(revision["review_root"]).parent / "materialization_receipt.json"
        prior_sha = _hash(prior_path)
        prior = _read(prior_path)
        prior_source_path = Path(prior["source_manifest_path"])
        prior_source_sha = prior["source_manifest_sha256"]
        if (prior.get("format") != FORMAT or prior.get("arm") != "within_run"
                or prior.get("run") != "ac47"
                or prior_source_sha != AC47_V4_SOURCE_SHA256
                or _hash(prior_source_path) != AC47_V4_SOURCE_SHA256
                or prior.get("materialized_manifest_sha256") != revision["manifest_sha256"]
                or Path(prior["materialized_manifest_path"]).resolve() !=
                    Path(revision["manifest_path"]).resolve()
                or original.get("two_arm_photometry", {}).get("arm") != "within_run"
                or original["two_arm_photometry"].get("source_manifest_sha256") != prior_source_sha
                or original["two_arm_photometry"].get("arm_photometry_revision_id") != source_id):
            raise ValueError("AC47 prior within-run photometry receipt is inconsistent")
        historical = _source_photometry_revision(
            _read(prior_source_path),
            dict(run="ac47", manifest_sha256=prior_source_sha,
                 review_root=prior["source_review_root"]),
        )
        if (prior.get("source_photometry_revision") != historical
                or original["two_arm_photometry"].get("source_photometry_revision") != historical):
            raise ValueError("AC47 historical v4 lineage was lost in prior arm")
        return dict(id=source_id, receipt_path=str(prior_path),
                    receipt_sha256=prior_sha,
                    source_manifest_sha256=prior_source_sha,
                    target_manifest_sha256=revision["manifest_sha256"],
                    source_photometry_revision=historical)
    if (revision["run"] != "ac47" or revision["manifest_sha256"] != AC47_V4_SOURCE_SHA256
            or source_id != "ac47-neutral-tone-only-v4-20260925"):
        raise ValueError("Unknown historical photometry revision in frozen native source")
    from darsia.fluidflower.tools.phase_training_photometry_revision import verify_revision
    verified = verify_revision(Path(revision["review_root"]))
    receipt_path = Path(verified["receipt_path"])
    if (verified["receipt_sha256"] != AC47_V4_RECEIPT_SHA256
            or verified["target_manifest_sha256"] != AC47_V4_SOURCE_SHA256
            or verified["changed_frame_indices"] != AC47_V4_CHANGED_FRAMES):
        raise ValueError("Frozen AC47 photometry revision differs from certified receipt")
    receipt = _read(receipt_path)
    return {
        "id": source_id,
        "receipt_path": str(receipt_path),
        "receipt_sha256": AC47_V4_RECEIPT_SHA256,
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "target_manifest_sha256": AC47_V4_SOURCE_SHA256,
        "changed_frame_indices": AC47_V4_CHANGED_FRAMES,
    }


def _prepared_baseline_source(original: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any] | None:
    """Bind the one prepared AC61 control to its frozen image and acquisition records.

    The three configured photos establish acquisition/time provenance. We do not
    infer that their pixels were averaged. The calibrated rig baseline itself is
    verified against the historical corrected-array digest in the within-run arm.
    """
    if revision["manifest_sha256"] != AC61_PREPARED_BASELINE_SOURCE_SHA256:
        return None
    if revision["run"] != "ac61" or original.get("frame_count") != 22:
        raise ValueError("Unexpected AC61 prepared-baseline source contract")
    first = original["frame_sources"][0]
    prepared = Path(first["source_path"])
    source_root = Path(revision["review_root"])
    old_frame = source_root / "native_frames" / "ac61" / "frame_000.jpg"
    old_baseline = source_root / "native_references" / "ac61" / "pre_injection.jpg"
    repair_path = source_root.parents[2] / "repair_receipt.json"
    if (first.get("frame_index") != 0 or
            first.get("selection_status") != "pre_injection_negative_control" or
            first.get("source_name") != "Pre-injection baseline" or
            first.get("source_sha256") != AC61_PREPARED_BASELINE_JPEG_SHA256 or
            original.get("reference", {}).get("corrected_array_sha256") !=
            AC61_PREPARED_BASELINE_ARRAY_SHA256 or
            any(_hash(path) != AC61_PREPARED_BASELINE_JPEG_SHA256
                for path in (prepared, old_frame, old_baseline)) or
            _hash(repair_path) != AC61_PREPARED_BASELINE_REPAIR_SHA256):
        raise ValueError("Frozen AC61 prepared baseline or repair receipt changed")
    members = first.get("configured_baseline_members")
    if (first.get("time_basis") !=
            "mean time of configured baseline members; retained prepared baseline" or
            not isinstance(members, list) or
            len(members) != len(AC61_PREPARED_BASELINE_MEMBERS) or
            abs(sum(float(row["elapsed_h"]) for row in members) / len(members) -
                float(first["elapsed_h"])) > 1e-12):
        raise ValueError("Frozen AC61 baseline acquisition times changed")
    verified_members = []
    for row, (name, expected_sha) in zip(members, AC61_PREPARED_BASELINE_MEMBERS, strict=True):
        path = Path(row["path"])
        if path.name != name or _hash(path) != expected_sha:
            raise ValueError(f"Frozen AC61 baseline acquisition changed: {path}")
        verified_members.append(dict(path=str(path), sha256=expected_sha,
                                     elapsed_h=row["elapsed_h"], timestamp=row["timestamp"]))
    return dict(source_manifest_sha256=revision["manifest_sha256"],
        original_pre_injection_path=str(prepared),
        original_pre_injection_sha256=AC61_PREPARED_BASELINE_JPEG_SHA256,
        historical_repair_receipt_path=str(repair_path),
        historical_repair_receipt_sha256=AC61_PREPARED_BASELINE_REPAIR_SHA256,
        members=verified_members,
        original_corrected_array_sha256=AC61_PREPARED_BASELINE_ARRAY_SHA256,
        reconstruction="calibrated_rig_baseline")


def _hash(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _frozen_registration_profile(manifest: dict[str, Any], run: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Use the exact source revision's registration, never a current policy lookup."""
    frozen = (manifest.get("correction_contract") or {}).get("rig_registration")
    if not frozen:
        return None, None
    if frozen.get("active") is not True:
        raise ValueError(f"{run}: malformed historical registration contract")
    profile = frozen.get("profile")
    if not isinstance(profile, dict) or profile.get("enabled") is not True:
        raise ValueError(f"{run}: missing frozen enabled registration profile")
    if profile.get("series") != "ffac" or profile.get("run") != run:
        raise ValueError(f"{run}: frozen registration belongs to another series or run")
    path = Path(profile["policy_source"])
    if not path.is_file() or _hash(path) != profile["policy_file_sha256"]:
        raise ValueError(f"{run}: frozen registration policy file changed or is missing")
    document = _read(path)
    policy_run = (document.get("runs") or {}).get(run)
    if not isinstance(policy_run, dict):
        raise ValueError(f"{run}: frozen policy lacks this registered run")
    for key, value in policy_run.items():
        if profile.get(key) != value:
            raise ValueError(f"{run}: embedded registration profile differs from frozen policy")
    if not frozen.get("geometry_revision_sha256"):
        raise ValueError(f"{run}: frozen geometry revision SHA is missing")
    return profile, frozen


def _require_matching_geometry(expected: dict[str, Any] | None,
                               actual: dict[str, Any] | None, run: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{run}: an unregistered source unexpectedly gained registration")
        return
    if not isinstance(actual, dict) or actual.get("active") is not True:
        raise ValueError(f"{run}: frozen registration was not installed")
    for field in ("geometry_revision_sha256", "reference_sha256", "domain_sha256",
                  "valid_bounds_xyxy", "shape_hw", "implementation_sha256",
                  "host_implementation_sha256"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{run}: registered geometry differs on {field}")


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def source_manifests(request_path: Path) -> list[dict[str, Any]]:
    """Find every original-native revision frozen in one prepared request."""
    request = _read(request_path)
    if request["settings"].get("preserve_native_revisions") is not True:
        raise ValueError("Expected an original-native frozen FF_AC request")
    by_sha: dict[str, dict[str, Any]] = {}
    for bundle in request["bundles"]:
        combined_path = Path(bundle["combined_manifest"]["snapshot_path"])
        if _hash(combined_path) != bundle["combined_manifest"]["sha256"]:
            raise ValueError(f"Frozen combined manifest changed: {combined_path}")
        combined = _read(combined_path)
        for job in combined.get("jobs") or []:
            if str(job.get("dataset") or "").casefold() != "ffac":
                continue
            run = str(job["run"]).casefold()
            review_root = analysis_review_root(
                Path(job.get("analysis_source_review_root") or job["review_output_dir"]), run)
            manifest_path = review_root / "native_frames" / run / "manifest.json"
            digest = _hash(manifest_path)
            previous = by_sha.get(digest)
            if previous is not None and previous["run"] != run:
                raise ValueError("One source manifest SHA identifies different runs")
            by_sha[digest] = dict(run=run, review_root=str(review_root),
                                  manifest_path=str(manifest_path), manifest_sha256=digest)
    if not by_sha:
        raise ValueError("Frozen request has no FF_AC native source revisions")
    return sorted(by_sha.values(), key=lambda item: (item["run"], item["manifest_sha256"]))


def selected_source_manifests(request_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select all effective-label native revisions plus every current Studio run."""
    request = _read(request_path)
    all_rows = source_manifests(request_path)
    pairs = [training.ReviewBundle(Path(bundle["decisions"]["snapshot_path"]),
                                   Path(bundle["combined_manifest"]["snapshot_path"]))
             for bundle in request["bundles"]]
    sources: dict[str, Any] = {}
    _, regions, _ = training._load_sources_and_regions(
        pairs, preserve_native_revisions=True, region_sources=sources)
    active_paths = {str(source.asset_manifest_path.resolve()).casefold()
                    for source in sources.values()}
    parent = Path(request["settings"]["parent_model"])
    timeline_jobs = _read(parent / "review_source" / "postflight_manifest.json")["jobs"]
    if len(timeline_jobs) != 43 or len({str(job["run"]).casefold() for job in timeline_jobs}) != 43:
        raise ValueError("Expected 43 distinct canonical FF_AC Studio runs")
    timeline_paths = {
        str((analysis_review_root(
            Path(job.get("analysis_source_review_root") or job["review_output_dir"]),
            str(job["run"]).casefold()) / "native_frames" / str(job["run"]).casefold() /
            "manifest.json").resolve()).casefold()
        for job in timeline_jobs
    }
    selected_paths = active_paths | timeline_paths
    selected = [row for row in all_rows
                if str(Path(row["manifest_path"]).resolve()).casefold() in selected_paths]
    if len(selected) != len(selected_paths):
        raise ValueError("Frozen bundles do not cover every active and canonical source revision")
    ids = sorted(region.region_id for region in regions)
    selection = dict(policy="effective-label-revisions-plus-43-canonical-studio-runs-v1",
        effective_label_count=len(ids),
        effective_label_ids_sha256=sha256("\n".join(ids).encode()).hexdigest(),
        active_source_revisions=len(active_paths), canonical_timeline_runs=len(timeline_jobs),
        canonical_timeline_revisions=len(timeline_paths),
        selected_source_revisions=len(selected),
        selected_source_manifest_sha256=sorted(row["manifest_sha256"] for row in selected),
        unused_frozen_bundle_revisions=[row["manifest_sha256"] for row in all_rows
                                        if str(Path(row["manifest_path"]).resolve()).casefold()
                                        not in selected_paths])
    return selected, selection


def _chart_qc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [row for row in rows if row.get("type") == "ColorCorrection"]
    if len(checked) != 1:
        raise ValueError("Expected exactly one Uniform24 color correction")
    row = checked[0]
    for field in ("residual", "max_swatch_residual"):
        if not np.isfinite(float(row[field])):
            raise ValueError(f"ColorChecker {field} is not finite")
    return dict(residual=float(row["residual"]),
                max_swatch_residual=float(row["max_swatch_residual"]),
                flagged=bool(row.get("flagged", False)),
                strategy=str(row.get("strategy") or ""),
                affine_rejection_reason=row.get("affine_rejection_reason"))


def _frame_record(path: Path, original_native_path: Path, raw: Path, original: dict[str, Any],
                  corrected: np.ndarray, qc: dict[str, Any]) -> dict[str, Any]:
    raw_sha = _hash(raw)
    expected = original.get("source_sha256") or original.get("source_raw_sha256")
    if expected and raw_sha != expected:
        raise ValueError(f"Frozen raw photo changed: {raw}")
    old_native_sha = _hash(original_native_path)
    expected_native = original.get("native_jpeg_sha256") or original.get("frame_sha256")
    if expected_native and old_native_sha != expected_native:
        raise ValueError(f"Frozen review image changed: {original_native_path}")
    if not np.isfinite(corrected).all():
        raise ValueError(f"Nonfinite corrected image: {raw}")
    return dict(frame_index=int(original["frame_index"]), elapsed_h=float(original["elapsed_h"]),
                raw_path=str(raw), raw_sha256=raw_sha,
                original_native_sha256=old_native_sha,
                corrected_array_sha256=_array_sha256(corrected),
                native_jpeg_sha256=_hash(path), colorchecker=qc)


def _sequential_baseline_fit(
    source: Any, target: dict[str, Any], target_sha256: str,
) -> dict[str, Any]:
    """Fit the run's second-stage transform on its start image alone.

    The first stage is the existing within-run ColorChecker correction. The
    frozen fit returned here is reused for every frame, without consulting any
    phase label, predicted plume or later frame.
    """
    from darsia.fluidflower.tools.phase_common_baseline_photometry import (
        fit_fixed_run_transform,
    )
    from darsia.fluidflower.tools.colorchecker_refinement import extract_swatches

    context_path = Path(target["facies_context_path"])
    if _hash(context_path) != target["facies_context_sha256"]:
        raise ValueError("Common-start facies context changed")
    if list(source.baseline.shape[:2]) != target["shape_hw"]:
        raise ValueError("Common-start target and corrected baseline grids differ")
    with np.load(context_path) as context:
        facies = np.asarray(context["facies"])
        domain = np.asarray(context["domain"])
    if facies.shape != source.baseline.shape[:2] or domain.shape != facies.shape:
        raise ValueError("Common-start facies/domain context has another grid")
    corrections = source.rig.color_corrections or []
    if len(corrections) != 1:
        raise ValueError("Expected one within-run ColorChecker correction")
    correction = corrections[0]
    observed = extract_swatches(correction._restrict_to_roi(source.baseline))
    raw_baseline = np.asarray(source.rig.shape_corrected_baseline.img, dtype=np.float32)[..., :3]
    raw_chart = extract_swatches(correction._restrict_to_roi(raw_baseline))
    usable = np.all(np.isfinite(raw_chart) & (raw_chart < 0.995), axis=2)
    fit = fit_fixed_run_transform(
        source.baseline, target,
        observed_chart_rgb=observed,
        chart_usable_patches=usable,
        facies_map=facies,
        domain_mask=domain & source.valid_mask,
    )
    fit["target_file_sha256"] = target_sha256
    fit["usable_chart_patches"] = int(usable.sum())
    fit["usable_chart_patch_mask"] = usable.tolist()
    return fit


def _common_chart_qc(corrected: np.ndarray, correction: Any,
                     target: dict[str, Any], usable_mask: list[list[bool]]) -> dict[str, Any]:
    """Measure every final frame against the frozen shared chart gauge."""
    from darsia.fluidflower.tools.colorchecker_refinement import extract_swatches

    observed = extract_swatches(correction._restrict_to_roi(corrected))
    expected = np.asarray(target["chart_target_rgb"], dtype=np.float32)
    usable = np.asarray(usable_mask, dtype=bool)
    if observed.shape != (4, 6, 3) or expected.shape != observed.shape or usable.shape != (4, 6):
        raise ValueError("Common ColorChecker QA shape is invalid")
    patch_error = np.mean(np.abs(observed - expected), axis=2)
    if not np.all(np.isfinite(patch_error)) or not np.any(usable):
        raise ValueError("Common ColorChecker QA is nonfinite or lacks usable patches")
    return dict(extractor_revision=SEQUENTIAL_CHART_EXTRACTOR_REVISION,
                mean_all=float(np.mean(patch_error)),
                mean_baseline_usable=float(np.mean(patch_error[usable])),
                max_all=float(np.max(patch_error)),
                usable_baseline_patches=int(np.count_nonzero(usable)))


def materialize_revision(revision: dict[str, Any], *, arm: str, output_root: Path,
                         common_target: tuple[Path, str] | None,
                         baseline_target: tuple[dict[str, Any], str] | None = None) -> dict[str, Any]:
    run = revision["run"]
    source_path = Path(revision["manifest_path"])
    source_sha = revision["manifest_sha256"]
    if _hash(source_path) != source_sha:
        raise ValueError(f"Original native manifest changed: {source_path}")
    original = _read(source_path)
    if original.get("run") != run or original.get("dataset") != "ffac":
        raise ValueError("Original manifest run/series changed")
    source_photometry_revision = _source_photometry_revision(original, revision)
    prepared_baseline = _prepared_baseline_source(original, revision)
    frames = original["frame_sources"]
    if len(frames) != int(original["frame_count"]):
        raise ValueError("Native frame count differs from frozen frame list")
    if [int(row["frame_index"]) for row in frames] != list(range(len(frames))):
        raise ValueError("Native frame indexes are non-contiguous")
    revision_id = source_sha[:24]
    review_root = output_root / "revisions" / revision_id / "manual_pairwise_review"
    frames_root = review_root / "native_frames" / run
    reference_root = review_root / "native_references" / run
    receipt_path = output_root / "revisions" / revision_id / "materialization_receipt.json"
    if receipt_path.is_file():
        receipt = _read(receipt_path)
        if (receipt.get("format") != FORMAT or receipt.get("arm") != arm
                or receipt.get("source_manifest_sha256") != source_sha
                or receipt.get("common_target_sha256") != (common_target[1] if common_target else None)
                or receipt.get("baseline_target_sha256") != (baseline_target[1] if baseline_target else None)
                or (arm == "sequential_global" and
                    (receipt.get("chart_extractor_revision") != SEQUENTIAL_CHART_EXTRACTOR_REVISION
                     or receipt.get("sequential_qa_revision") != SEQUENTIAL_QA_REVISION))
                or Path(receipt["materialized_manifest_path"]).resolve() !=
                    (frames_root / "manifest.json").resolve()
                or _hash(Path(receipt["materialized_manifest_path"])) != receipt["materialized_manifest_sha256"]):
            raise ValueError(f"Completed revision receipt changed: {receipt_path}")
        baseline_path = reference_root / "pre_injection.jpg"
        if _hash(baseline_path) != receipt.get("materialized_baseline_sha256"):
            raise ValueError(f"Completed materialization baseline changed: {baseline_path}")
        reference = original.get("reference") or {}
        if reference.get("source_sha256"):
            source_baseline = Path(reference["source_path"])
            if _hash(source_baseline) != reference["source_sha256"]:
                raise ValueError(f"Frozen raw baseline changed: {source_baseline}")
        records = receipt.get("frames") or []
        if receipt.get("frame_count") != len(frames) or len(records) != len(frames):
            raise ValueError(f"Completed revision frame count changed: {receipt_path}")
        for index, (frame, original_row) in enumerate(zip(records, frames, strict=True)):
            raw = Path(original_row["source_path"])
            path = frames_root / f"frame_{index:03d}.jpg"
            original_native = (Path(revision["review_root"]) / "native_frames" /
                               run / f"frame_{index:03d}.jpg")
            if (frame.get("frame_index") != index
                    or frame.get("raw_path") != str(raw)
                    or float(frame.get("elapsed_h")) != float(original_row["elapsed_h"])
                    or _hash(raw) != frame.get("raw_sha256")
                    or frame.get("raw_sha256") !=
                        (original_row.get("source_sha256") or original_row.get("source_raw_sha256"))
                    or _hash(original_native) != frame.get("original_native_sha256")
                    or _hash(path) != frame.get("native_jpeg_sha256")):
                raise ValueError(f"Completed materialization frame or source changed: {path}")
        return receipt
    if frames_root.exists() or reference_root.exists():
        raise ValueError(f"Unreceipted partial revision requires manual audit: {review_root}")

    registration_profile, frozen_registration = _frozen_registration_profile(original, run)
    with _open_ffac(run, registration_profile=registration_profile,
                    disable_registration=registration_profile is None,
                    common_colorchecker_target=common_target) as source:
        color_corrections = source.rig.color_corrections or []
        if (len(color_corrections) != 1 or
                getattr(color_corrections[0], "algorithm_revision", None) != "legacy-v2"):
            raise ValueError(f"{run}: two-arm ColorChecker pipeline is not the shared legacy-v2 correction")
        actual_registration = source.correction_contract.get("rig_registration")
        _require_matching_geometry(frozen_registration, actual_registration, run)
        expected_shape = tuple(original["analysis_image_shape_hw"])
        if source.baseline.shape[:2] != expected_shape:
            raise ValueError(f"Historical {run} image geometry differs: {source.baseline.shape[:2]} vs {expected_shape}")
        canvas_wh = tuple(original["canvas_wh"])
        content_box = tuple(original["content_box_xywh"])
        if content_box[2:] != tuple(reversed(expected_shape)):
            raise ValueError("Historical content box is incompatible with raw-rig image")
        baseline_raw = Path(getattr(source.rig, "baseline_path"))
        reference = original.get("reference") or {}
        expected_baseline_sha = (reference.get("source_sha256") or
                                 reference.get("source_raw_sha256"))
        if expected_baseline_sha and _hash(baseline_raw) != expected_baseline_sha:
            raise ValueError(f"Frozen baseline raw photo changed for {run}")
        if (prepared_baseline is not None and
                baseline_raw.resolve() != Path(prepared_baseline["members"][0]["path"]).resolve()):
            raise ValueError("AC61 rig baseline differs from frozen first acquisition")
        sequential_fit = None
        if arm == "sequential_global":
            if baseline_target is None:
                raise ValueError("Sequential arm requires a frozen common-start target")
            sequential_fit = _sequential_baseline_fit(
                source, baseline_target[0], baseline_target[1]
            )
            sequential_fit["chart_extractor_revision"] = SEQUENTIAL_CHART_EXTRACTOR_REVISION
            from darsia.fluidflower.tools.phase_common_baseline_photometry import (
                apply_fixed_run_transform,
            )
        frames_root.mkdir(parents=True)
        reference_root.mkdir(parents=True)
        within_run_baseline = source.baseline
        baseline = (apply_fixed_run_transform(within_run_baseline, sequential_fit)
                    if sequential_fit is not None else source.baseline)
        baseline_common_chart_qc = (
            _common_chart_qc(baseline, source.rig.color_corrections[0],
                             baseline_target[0], sequential_fit["usable_chart_patch_mask"])
            if sequential_fit is not None else None)
        baseline_frame = _plain_frame(baseline, run=run, index=0, count=1, elapsed_h=0.0,
                                      width=canvas_wh[0], height=canvas_wh[1], native=True)
        baseline_path = reference_root / "pre_injection.jpg"
        _write_jpeg(baseline_path, baseline_frame)
        baseline_diagnostics = _correction_diagnostics(source.rig)
        baseline_qc = _chart_qc(baseline_diagnostics)
        if prepared_baseline is not None:
            prepared_baseline["materialized_corrected_array_sha256"] = _array_sha256(baseline)
            if (arm == "within_run" and
                    prepared_baseline["materialized_corrected_array_sha256"] !=
                    AC61_PREPARED_BASELINE_ARRAY_SHA256):
                raise ValueError("AC61 within-run calibrated baseline differs from frozen source")
        output_frames = []
        updated_frames = []
        for row in frames:
            index = int(row["frame_index"])
            sequential_input_qc = None
            if frozen_registration is not None:
                geometry_id = row.get("geometry_revision_id")
                if geometry_id and geometry_id != frozen_registration.get("geometry_revision_id"):
                    raise ValueError(f"{run}: frame {index} references another registered geometry")
                registration_receipt = row.get("registration_receipt_path")
                if registration_receipt:
                    registration_receipt_path = Path(registration_receipt)
                    if (_hash(registration_receipt_path) !=
                            row.get("registration_receipt_sha256")):
                        raise ValueError(f"{run}: frame {index} registration receipt changed")
            raw = Path(row["source_path"])
            if not raw.is_file():
                raise FileNotFoundError(raw)
            if prepared_baseline is not None and index == 0:
                # The historical timeline prepended a copy of an already corrected
                # baseline JPEG. It is not a raw image in the experiment protocol.
                corrected, diagnostics = within_run_baseline, baseline_diagnostics
            else:
                corrected, diagnostics = source.read(raw)
            if corrected.shape[:2] != expected_shape:
                raise ValueError(f"Raw frame geometry changed: {raw}")
            if sequential_fit is not None:
                from darsia.fluidflower.tools.phase_common_baseline_photometry import (
                    inspect_fixed_frame_input,
                )

                sequential_input_qc = inspect_fixed_frame_input(corrected, sequential_fit)
                corrected = apply_fixed_run_transform(corrected, sequential_fit)
                post_common_chart_qc = _common_chart_qc(
                    corrected, source.rig.color_corrections[0],
                    baseline_target[0], sequential_fit["usable_chart_patch_mask"])
                diagnostics = [*diagnostics, dict(type="FixedRunBaselineCorrection",
                    target_sha256=baseline_target[1],
                    fitted_on="within-run-corrected-pre-injection-baseline",
                    reused_without_refitting=True,
                    input_range_qa=sequential_input_qc,
                    common_chart_qa=post_common_chart_qc)]
            native = _plain_frame(corrected, run=run, index=index, count=len(frames),
                                  elapsed_h=float(row["elapsed_h"]), width=canvas_wh[0],
                                  height=canvas_wh[1], native=True)
            target = frames_root / f"frame_{index:03d}.jpg"
            _write_jpeg(target, native)
            original_native_path = Path(revision["review_root"]) / "native_frames" / run / f"frame_{index:03d}.jpg"
            record = _frame_record(target, original_native_path, raw, row,
                                   corrected, _chart_qc(diagnostics))
            if sequential_input_qc is not None:
                record["sequential_input_range_qa"] = sequential_input_qc
                record["post_common_chart_qa"] = post_common_chart_qc
            output_frames.append(record)
            updated = {key: value for key, value in row.items()
                       if key not in {"color_corrections", "color_correction_flagged",
                                      "color_correction_active", "phase_map"}}
            updated.update(source_sha256=record["raw_sha256"],
                           source_raw_sha256=record["raw_sha256"],
                           native_jpeg_sha256=record["native_jpeg_sha256"],
                           frame_sha256=record["native_jpeg_sha256"],
                           frame_path=str(target),
                           corrected_array_sha256=record["corrected_array_sha256"],
                           color_correction_active=True,
                           color_correction_flagged=record["colorchecker"]["flagged"],
                           color_corrections=diagnostics)
            updated_frames.append(updated)
        new_manifest = dict(original)
        new_manifest["frame_sources"] = updated_frames
        new_manifest["reference"] = {**reference,
                                     "corrected_array_sha256": _array_sha256(baseline),
                                     "materialized_native_sha256": _hash(baseline_path)}
        new_manifest["correction_contract"] = {**source.correction_contract,
                                                 "source_revision_manifest_sha256": source_sha}
        arm_photometry_revision_id = (
            "ffac-two-arm-within-run-legacy-v2-v1" if arm == "within_run"
            else "ffac-two-arm-global-robust16-legacy-v2-v1" if arm == "global"
            else "ffac-sequential-within-run-common-start-deterministic-chartqa-v3")
        if source_photometry_revision is not None:
            new_manifest["photometry_revision_id"] = arm_photometry_revision_id
        new_manifest["two_arm_photometry"] = dict(format=FORMAT, arm=arm,
            source_manifest=str(source_path), source_manifest_sha256=source_sha,
            common_target_sha256=(common_target[1] if common_target else None),
            arm_photometry_revision_id=arm_photometry_revision_id,
            arm_colorchecker_algorithm_revision="legacy-v2",
            source_photometry_revision=source_photometry_revision,
            prepared_baseline_composite=prepared_baseline,
            same_raw_geometry_and_timestamps=True,
            no_phase_label_or_map_in_target_selection=True)
        if sequential_fit is not None:
            new_manifest["sequential_photometry"] = dict(
                within_run_first=True, common_start_second=True,
                baseline_target_sha256=baseline_target[1],
                chart_extractor_revision=SEQUENTIAL_CHART_EXTRACTOR_REVISION,
                sequential_qa_revision=SEQUENTIAL_QA_REVISION,
                fixed_run_fit=sequential_fit,
                no_phase_label_or_map_in_target_selection=True)
        manifest_path = frames_root / "manifest.json"
        _write_new(manifest_path, new_manifest)
        result = dict(format=FORMAT, arm=arm, run=run,
            source_manifest_path=str(source_path), source_manifest_sha256=source_sha,
            source_review_root=revision["review_root"],
            materialized_review_root=str(review_root),
            materialized_manifest_path=str(manifest_path),
            materialized_manifest_sha256=_hash(manifest_path),
            arm_photometry_revision_id=arm_photometry_revision_id,
            source_photometry_revision=source_photometry_revision,
            prepared_baseline_composite=prepared_baseline,
            original_baseline_sha256=reference.get("source_sha256"),
            materialized_baseline_sha256=_hash(baseline_path),
            baseline_colorchecker=baseline_qc,
            baseline_common_chart_qa=baseline_common_chart_qc,
            common_target_sha256=(common_target[1] if common_target else None),
            baseline_target_sha256=(baseline_target[1] if baseline_target else None),
            chart_extractor_revision=(SEQUENTIAL_CHART_EXTRACTOR_REVISION
                                      if sequential_fit is not None else None),
            sequential_qa_revision=(SEQUENTIAL_QA_REVISION
                                    if sequential_fit is not None else None),
            sequential_fit=sequential_fit,
            registered_geometry=actual_registration is not None,
            geometry_revision_sha256=(actual_registration or {}).get("geometry_revision_sha256"),
            content_box_xywh=list(content_box), analysis_image_shape_hw=list(expected_shape),
            canvas_wh=list(canvas_wh), frame_count=len(output_frames), frames=output_frames,
            created_utc=datetime.now(timezone.utc).isoformat())
        _write_new(receipt_path, result)
        return result


def materialize(request_path: Path, *, arm: str, output_root: Path,
                target_path: Path | None = None, target_sha256: str | None = None,
                revision_limit: int | None = None,
                only_source_sha256: str | None = None,
                only_runs: tuple[str, ...] | None = None,
                workers: int = 1) -> dict[str, Any]:
    if arm not in {"within_run", "global", "sequential_global"}:
        raise ValueError("Unknown photometry arm")
    if (arm != "within_run") != (target_path is not None and target_sha256 is not None):
        raise ValueError("Global and sequential arms require one SHA-bound target")
    common_target = None
    baseline_target = None
    if arm == "global":
        from darsia.fluidflower.tools.phase_two_arm_photometry import load_target
        load_target(target_path, expected_sha256=target_sha256)
        common_target = (target_path, target_sha256)
    elif arm == "sequential_global":
        if _hash(target_path) != target_sha256:
            raise ValueError("Common-start target file differs from requested SHA")
        baseline_target = (_read(target_path), target_sha256)
        from darsia.fluidflower.tools.ffac_common_start_target import validate_candidate

        validate_candidate(baseline_target[0], verify_source_bytes=True)
    request_sha = _hash(request_path)
    revisions = source_manifests(request_path)
    selection = dict(policy="all-frozen-bundle-native-revisions-v1",
                     selected_source_revisions=len(revisions),
                     selected_source_manifest_sha256=sorted(row["manifest_sha256"]
                                                            for row in revisions))
    if only_source_sha256 is not None:
        revisions = [row for row in revisions if row["manifest_sha256"] == only_source_sha256]
        if len(revisions) != 1:
            raise ValueError("Requested source manifest SHA was not unique in frozen request")
    if only_runs is not None:
        wanted = {run.casefold() for run in only_runs}
        if len(wanted) != len(only_runs):
            raise ValueError("Requested run names are duplicated")
        revisions = [row for row in revisions if row["run"] in wanted]
        if {row["run"] for row in revisions} != wanted or len(revisions) != len(wanted):
            raise ValueError("Each requested run must identify exactly one frozen source revision")
    if revision_limit is not None:
        revisions = revisions[:revision_limit]
    if not 1 <= workers <= 24:
        raise ValueError("Materialization workers must be between 1 and 24")
    if workers == 1 or len(revisions) < 2:
        reports = [materialize_revision(row, arm=arm, output_root=output_root,
                                        common_target=common_target,
                                        baseline_target=baseline_target) for row in revisions]
    else:
        # Each native source revision owns a different SHA-keyed directory. A
        # process has its own rig environment, unlike unsafe concurrent threads.
        reports = [None] * len(revisions)
        with ProcessPoolExecutor(max_workers=min(workers, len(revisions))) as pool:
            pending = {
                pool.submit(materialize_revision, row, arm=arm,
                            output_root=output_root, common_target=common_target,
                            baseline_target=baseline_target): index
                for index, row in enumerate(revisions)
            }
            for future in as_completed(pending):
                reports[pending[future]] = future.result()
    result = dict(format=FORMAT, arm=arm, frozen_request_path=str(request_path),
                  frozen_request_sha256=request_sha, target_path=str(target_path) if target_path else None,
                  target_sha256=target_sha256, selection=selection,
                  source_revisions=len(reports),
                  frames=sum(row["frame_count"] for row in reports), revisions=reports,
                  complete=revision_limit is None and only_source_sha256 is None and only_runs is None,
                  created_utc=datetime.now(timezone.utc).isoformat())
    summary = output_root / "materialization_receipt.json"
    if summary.is_file():
        previous = _read(summary)
        # Per-revision replay already rehashes every saved native frame. The
        # top-level creation time belongs to the first completed materialization,
        # so a verified resume must not replace or compare it to today's time.
        comparable = {key: value for key, value in result.items()
                      if key != "created_utc"}
        if {key: value for key, value in previous.items()
                if key != "created_utc"} != comparable:
            raise FileExistsError(f"Materialization summary already exists: {summary}")
        result = previous
    else:
        _write_new(summary, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-request", type=Path, required=True)
    parser.add_argument("--arm", choices=("within_run", "global", "sequential_global"), required=True)
    parser.add_argument("--target-profile", type=Path)
    parser.add_argument("--target-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision-limit", type=int)
    parser.add_argument("--only-source-sha256")
    parser.add_argument("--only-runs", nargs="+")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent native source revisions processed in separate CPU processes (1-24)")
    args = parser.parse_args(argv)
    result = materialize(args.frozen_request, arm=args.arm, output_root=args.output_root,
                         target_path=args.target_profile, target_sha256=args.target_sha256,
                         revision_limit=args.revision_limit,
                         only_source_sha256=args.only_source_sha256,
                         only_runs=tuple(args.only_runs) if args.only_runs is not None else None,
                         workers=args.workers)
    print(json.dumps({key: result[key] for key in ("format", "arm", "source_revisions", "frames", "complete")}), flush=True)


if __name__ == "__main__":
    main()
