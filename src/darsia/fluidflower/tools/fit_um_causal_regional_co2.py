"""One isolated FF_UM grouped-run regional CO2 feature ablation.

Freeze an exact-source dense cohort; replay frozen optical cache entries for
every native frame in each run; extract paired static20/causal30 samples; fit
only grouped out-of-fold CO2-presence heads. No final model, phase map, Studio
registration, optical fit, gas-head fit, or publication occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from PIL import Image

from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower import phase_plume
from darsia.fluidflower import phase_causal_temporal_features as causal
from darsia.fluidflower.phase_training import logistic
from darsia.fluidflower.phase_validation import _model_identity
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import audit_um_temporal_training_cohort as cohort_audit
from darsia.fluidflower.tools.derive_um_temporal_candidate import review_inventory
from darsia.fluidflower.tools.publish_regional_review_candidate import verify_candidate
from darsia.fluidflower.tools.train_phase_optical_regions import region_samples


FORMAT = "ffum-causal-regional-co2-grouped-cv-v1"
PARENT_SHA256 = "89c6b26370abc01806527d584eeb8fa8ae547c87869c15b740efea48288d6911"
SOURCE_SHA256 = "ec7f4b4a01344e10f2bc4e7859b8ce1ac4b1f2a9911eb2586ee0a9ab51720684"
AUDIT_SHA256 = "e1999d08d9a59fe7c1fadcc652adbc47d9f7843bb212647bc378d5cbe05a3f0a"
CAUSAL_SHA256 = "55067245a247294e8907cf9d8e5f358272d548e7576cfd5defdb54d1621a1ae8"
OPTICAL_IDENTITY_SHA256 = "9c8daa05c38e5152a3f6d6b2d03c7ceee1a3f0f7dafd3d2e425c86a566535a63"
FOLDS = {
    "fold1": ("run7", "run10", "run20"),
    "fold2": ("run9", "run11", "run16", "run19"),
    "fold3": ("run6", "run12", "run14", "run15", "run17", "run18"),
}
REGULARIZATION = (0.001, 0.01)
EXPECTED_DENSE = Counter(aqueous=496, gas=32, current_experiment_co2_artifact=45, water=72)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return sha256_file(path)


def _frozen_code_paths() -> list[Path]:
    shared = Path(regional.__file__).parent
    return [Path(__file__), Path(cohort_audit.__file__), Path(regional.__file__),
            Path(causal.__file__), Path(logistic.__file__), Path(phase_plume.__file__),
            Path(region_samples.__code__.co_filename),
            *(shared / name for name in (
                "phase_regional_gas.py", "phase_region_output_guards.py",
                "phase_free_gas_output_guard.py", "phase_rendering.py"))]


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _write_status(root: Path, **status) -> None:
    value = dict(format=FORMAT, updated_utc=datetime.now(timezone.utc).isoformat(), **status)
    path = root / "status.json"
    temporary = root / "status.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _crop_rgb(path: Path, box: list[int]) -> np.ndarray:
    x, y, w, h = map(int, box)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").crop((x, y, x+w, y+h)))


def _verified_cached_heads(cache_root: Path, key: str, shape: tuple[int, int]):
    """Return native memmap and its exact optical payload receipt, read-only."""
    return cohort_audit._verified_heads(cache_root, key, shape)


def load_verified_run(model: Path, run: str, cache_root: Path, receipt: dict) -> dict:
    """Bind one full native timeline, registered context, and all cache keys."""
    model, cache_root = Path(model), Path(cache_root)
    images = receipt["images"][run]
    postflight = _read(model / "validation/run_packages" / run / "postflight_manifest.json")
    if (len(postflight["jobs"]) != 1 or postflight["jobs"][0]["run"] != run
            or postflight["model_sha256"] != PARENT_SHA256):
        raise ValueError(f"Published native validation binding changed: {run}")
    native_dir = Path(postflight["jobs"][0]["review_output_dir"]) / "native_frames" / run
    manifest_path = native_dir / "manifest.json"
    if _sha(manifest_path) != images["manifest_sha256"]:
        raise ValueError(f"Current native manifest changed: {run}")
    manifest = _read(manifest_path)
    count = int(manifest["frame_count"])
    if (count != len(manifest["frame_sources"])
            or count != len(images["frame_sha256"])
            or tuple(manifest["analysis_image_shape_hw"]) !=
               (int(manifest["content_box_xywh"][3]), int(manifest["content_box_xywh"][2]))):
        raise ValueError(f"Native frame/geometry inventory changed: {run}")
    backend = _read(model / "validation/run_packages" / run / "backend_checks.json")
    if len(backend["runs"]) != 1:
        raise ValueError(f"Unexpected native optical backend report: {run}")
    keys = {int(item["frame"]): item["optical_cache_key"]
            for item in backend["runs"][0]["frames"]}
    if set(keys) != set(range(count)):
        raise ValueError(f"Optical cache does not cover all native frames: {run}")
    profile = _read(model / "phase_optical_regions.json")
    context, context_record = phase_plume.load_context(
        model, profile, "ffum", run, native_manifest_sha256=images["manifest_sha256"])
    if tuple(context.domain.shape) != tuple(manifest["analysis_image_shape_hw"]):
        raise ValueError(f"Registered context/native shape differs: {run}")
    times = [float(item["elapsed_h"]) for item in manifest["frame_sources"]]
    first_injection = min(float(stage["start_h"]) for stage in context.injection_stages)
    if (not times or not times[0] < first_injection
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError(f"No exact ordered pre-injection prefix: {run}")
    for index, item in enumerate(manifest["frame_sources"]):
        if (int(item["frame_index"]) != index
                or item["native_jpeg_sha256"] != images["frame_sha256"][index]):
            raise ValueError(f"Native frame source identity changed: {run} {index}")
    return dict(run=run, native_dir=native_dir, manifest_path=manifest_path,
                manifest=manifest, keys=keys, context=context,
                context_record=context_record, times=times,
                baseline_sha256=images["baseline_sha256"],
                frame_sha256=images["frame_sha256"], cache_root=cache_root)


def _freeze_preflight(model: Path, comparison_path: Path, audit_path: Path,
                      cache_root: Path, output: Path, protocol_path: Path) -> tuple[dict, dict]:
    model, comparison_path, audit_path = (Path(p).resolve() for p in (model, comparison_path, audit_path))
    output, cache_root = Path(output).resolve(), Path(cache_root).resolve()
    if (output.parent != (model.parent.parent / "results").resolve()
            or not output.name.startswith("learned-regional-temporal-co2-")
            or output.exists()):
        raise ValueError("Use a new isolated FF_UM results/learned-regional-temporal-co2-* job")
    if _sha(audit_path) != AUDIT_SHA256 or _sha(Path(causal.__file__)) != CAUSAL_SHA256:
        raise ValueError("Frozen aligned cohort or causal feature module changed")
    if not cache_root.is_dir():
        raise FileNotFoundError("Frozen local FF_UM optical cache unavailable")
    checked = verify_candidate(model)
    model_id, model_sha = _model_identity(model)
    if (checked["series"] != "ffum" or checked["runs"] != 13 or checked["frames"] != 261
            or model_sha != PARENT_SHA256 or checked["model_sha256"] != model_sha
            or checked["source_manifest_sha256"] != SOURCE_SHA256):
        raise ValueError("Published FF_UM parent/source identity changed")
    comparison, audit = _read(comparison_path), _read(audit_path)
    if (comparison["series"] != "ffum"
            or comparison["labels"]["summary"]["aligned_regions"] != 719
            or audit["format"] != cohort_audit.FORMAT
            or audit["comparison_sha256"] != _sha(comparison_path)
            or audit["native_source_manifest_sha256"] != SOURCE_SHA256
            or Counter(audit["counts"]) != Counter({
                "aqueous_positive": 496, "gas_co2_positive": 32,
                "carryover_negative": 45, "water_negative": 72})):
        raise ValueError("Source-aligned FF_UM QA cohort changed")
    request_path = Path(comparison["labels"]["frozen_request"])
    if (_sha(request_path) != audit["frozen_request_sha256"]
            or comparison["labels"]["frozen_request_sha256"] != audit["frozen_request_sha256"]):
        raise ValueError("Frozen regional training request changed")
    source_receipt = _read(model / "native_source_receipt.json")
    if (source_receipt["source_manifest_sha256"] != SOURCE_SHA256
            or source_receipt["runs"] != 13 or source_receipt["frames"] != 261):
        raise ValueError("Published 13/261 native source receipt changed")
    frozen_optical = _read(model / "parent_provenance/training_request.json")
    reviews = review_inventory(model.parent.parent / "reviews", frozen_optical["created_utc"])
    expected_runs = set(sum((list(v) for v in FOLDS.values()), []))
    if expected_runs != set(source_receipt["images"]):
        raise ValueError("Predeclared grouped folds do not cover exact 13 runs")
    if tuple(regional.FEATURE_NAMES) != tuple(_read(model / "regional_model.json")["feature_names"]):
        raise ValueError("Existing 20-feature contract changed")
    if len(causal.FEATURE_NAMES) != 10 or causal.VERSION != "causal-temporal-regional-features-v1":
        raise ValueError("Causal 10-feature contract changed")
    code_paths = _frozen_code_paths()
    protocol = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        output=str(output), parent=str(model), parent_model_id=model_id,
        parent_model_sha256=model_sha, parent_source_manifest_sha256=SOURCE_SHA256,
        parent_profile_sha256=_sha(model / "phase_optical_regions.json"),
        parent_regional_model_sha256=_sha(model / "regional_model.json"),
        parent_native_receipt_sha256=_sha(model / "native_source_receipt.json"),
        frozen_optical_request_sha256=_sha(model / "parent_provenance/training_request.json"),
        frozen_regional_request_sha256=_sha(request_path),
        comparison=str(comparison_path), comparison_sha256=_sha(comparison_path),
        cohort_audit=str(audit_path), cohort_audit_sha256=_sha(audit_path),
        code_sha256={p.name: _sha(p) for p in code_paths},
        protocol_document=str(protocol_path.resolve()),
        protocol_document_sha256=_sha(protocol_path),
        review_inventory=reviews,
        optical_cache_root=str(cache_root),
        feature_contract=dict(static20=list(regional.FEATURE_NAMES),
                              causal30=list(regional.FEATURE_NAMES)+list(causal.FEATURE_NAMES),
                              causal_format=causal.VERSION, causal_settings=asdict(causal.Settings())),
        folds={key: list(value) for key, value in FOLDS.items()},
        regularization=list(REGULARIZATION),
        label_policy=dict(dense_classes=dict(cohort_audit.CLASSES),
            preinjection_water_excluded_from_paired_fit=4,
            model_specific_issue_boxes_are_not_dense_truth=True),
        optical_fitted=False, gas_head_fitted=False,
        phase_maps_generated=False, studio_registered=False,
        independent_validation=False, training_seen_optics=True)
    return protocol, audit


def verify_all_prefix_sources(protocol: dict, audit: dict) -> dict:
    """Verify every native and cached optical prefix before creating a job."""
    model = Path(protocol["parent"])
    cache_root = Path(protocol["optical_cache_root"])
    source_receipt = _read(model / "native_source_receipt.json")
    audited_payloads = audit["verified_cache_payloads"]
    verified_frames, audited_seen = 0, set()
    for run in sorted(source_receipt["images"]):
        source = load_verified_run(model, run, cache_root, source_receipt)
        shape = tuple(np.asarray(source["context"].domain).shape)
        for index, expected in enumerate(source["frame_sha256"]):
            photo = source["native_dir"] / f"frame_{index:03d}.jpg"
            if _sha(photo) != expected:
                raise ValueError(f"Exact native photo changed: {run} {index}")
            key = source["keys"][index]
            heads, cache_receipt = _verified_cached_heads(cache_root, key, shape)
            if cache_receipt["identity_sha256"] != OPTICAL_IDENTITY_SHA256:
                raise ValueError(f"Frozen optical identity changed: {run} {index}")
            if key in audited_payloads:
                if cache_receipt["payload_sha256"] != audited_payloads[key]:
                    raise ValueError(f"Audited optical cache payload changed: {run} {index}")
                audited_seen.add(key)
            if tuple(heads.shape) != (*shape, 2):
                raise ValueError(f"Frozen optical payload geometry changed: {run} {index}")
            verified_frames += 1
            del heads
        print(json.dumps(dict(stage="prefix_preflight", run=run,
                              frames_done=verified_frames, frames_total=261)), flush=True)
    if verified_frames != 261 or audited_seen != set(audited_payloads):
        raise ValueError("Complete 13/261 native and audited-cache prefix parity failed")
    return dict(runs=13, frames=verified_frames,
                audited_cache_payloads=len(audited_seen),
                optical_identity_sha256=OPTICAL_IDENTITY_SHA256)


def _post_launch_source_check(protocol: dict) -> dict:
    """Keep the frozen fit immutable while allowing later user reviews."""
    model = Path(protocol["parent"])
    frozen = {
        model / "phase_optical_regions.json": protocol["parent_profile_sha256"],
        model / "regional_model.json": protocol["parent_regional_model_sha256"],
        model / "native_source_receipt.json": protocol["parent_native_receipt_sha256"],
        model / "parent_provenance/training_request.json":
            protocol["frozen_optical_request_sha256"],
        Path(protocol["comparison"]): protocol["comparison_sha256"],
        Path(protocol["cohort_audit"]): protocol["cohort_audit_sha256"],
    }
    for path, expected in frozen.items():
        if _sha(path) != expected:
            raise ValueError(f"Frozen source input changed during exploratory fit: {path}")
    for filename, expected in protocol["code_sha256"].items():
        paths = _frozen_code_paths()
        matches = [path for path in paths if path.name == filename]
        if len(matches) != 1 or _sha(matches[0]) != expected:
            raise ValueError(f"Frozen extraction/fit code changed: {filename}")
    previous = protocol["review_inventory"]
    try:
        current = review_inventory(Path(previous["root"]), previous["frozen_request_utc"])
    except ValueError as error:
        # The fit already froze its exact cohort. Saved review additions made
        # while the user keeps validating belong to the next fit, not this one.
        if not str(error).startswith("Saved FF_UM reviews changed after the frozen optical request:"):
            raise
        current = dict(root=previous["root"],
                       frozen_request_utc=previous["frozen_request_utc"],
                       later_saved_evidence=str(error))
    return dict(frozen_review_inventory=previous, latest_review_inventory=current,
                later_saved_reviews_seen=current != previous,
                later_reviews_excluded_from_this_frozen_fit=True)


def _sample_records(audit: dict, evidence: dict) -> tuple[list[dict], dict[tuple[str, int], list[tuple[int, dict]]], list[dict]]:
    by_id = {row["region_id"]: row for row in evidence["regions"]}
    regions, groups, fallback = [], defaultdict(list), []
    for item in audit["regions"]:
        record = dict(by_id[item["region_id"]])
        record["frame_index"] = int(item["studio_frame_1based"]) - 1
        if (record["phase_class"] != item["phase_class"]
                or record["run"] != item["run"]
                or record["confidence"] == "low"):
            raise ValueError("Frozen aligned dense annotation changed")
        if record["frame_index"] == 0:
            if record["phase_class"] != "water":
                raise ValueError("Unexpected positive/other pre-injection annotation")
            fallback.append(record)
            continue
        index = len(regions)
        regions.append(record)
        groups[(record["run"], record["frame_index"])].append((index, record))
    if Counter(r["phase_class"] for r in regions) != EXPECTED_DENSE - Counter(water=4):
        raise ValueError("Dense post-injection label inventory changed")
    if len(fallback) != 4:
        raise ValueError("Pre-injection static fallback control count changed")
    return regions, groups, fallback


def _extract(root: Path, protocol: dict, audit: dict) -> dict:
    model = Path(protocol["parent"])
    cache_root = Path(protocol["optical_cache_root"])
    source_receipt = _read(model / "native_source_receipt.json")
    regional_model = _read(model / "regional_model.json")
    settings = regional.Settings(**regional_model["settings"])
    evidence = _read(model / "evidence.json")
    regions, groups, fallback = _sample_records(audit, evidence)
    _write_new(root / "regions.json", dict(format=FORMAT,
        regions=regions, preinjection_fallback_controls=fallback))
    static_chunks, causal_chunks, owner_chunks = [], [], []
    frame_receipts, coverage = [], []
    identity_sha = None
    processed = 0
    for run in sorted(source_receipt["images"]):
        source = load_verified_run(model, run, cache_root, source_receipt)
        manifest = source["manifest"]
        box = manifest["content_box_xywh"]
        domain = np.asarray(source["context"].domain, bool)
        spacing = (abs(float(np.diff(source["context"].y_m)[0])),
                   float(np.diff(source["context"].x_m)[0]))
        baseline_path = source["native_dir"] / "frame_000.jpg"
        if _sha(baseline_path) != source["baseline_sha256"]:
            raise ValueError(f"Exact corrected baseline changed: {run}")
        baseline = _crop_rgb(baseline_path, box)
        state = causal.CausalTemporalFeatureState(source["context"], run=run,
            native_manifest_sha256=_sha(source["manifest_path"]))
        for index, frame in enumerate(manifest["frame_sources"]):
            photo = source["native_dir"] / f"frame_{index:03d}.jpg"
            if _sha(photo) != source["frame_sha256"][index]:
                raise ValueError(f"Exact corrected native photo changed: {run} {index}")
            key = source["keys"][index]
            heads, cache_receipt = _verified_cached_heads(cache_root, key, tuple(domain.shape))
            if identity_sha is None:
                identity_sha = cache_receipt["identity_sha256"]
            elif cache_receipt["identity_sha256"] != identity_sha:
                raise ValueError("Frozen optical weights/runtime identity changes within campaign")
            current = _crop_rgb(photo, box)
            prepared = regional.prepare(current, baseline, heads[..., 0], heads[..., 1],
                                        domain, spacing, settings)
            extra, temporal_audit = state.peek(frame_index=index,
                elapsed_h=float(frame["elapsed_h"]), prepared=prepared,
                raw_presence=heads[..., 0])
            records = groups.get((run, index), [])
            if records and extra is None:
                raise ValueError(f"Aligned training label has no causal prefix: {run} {index}")
            if extra is not None and records:
                paired = dict(prepared,
                    features=np.concatenate((prepared["features"], extra), axis=-1))
                x, owners, rows = region_samples(paired, records, manifest,
                                                  native_domain=domain)
                if len(x) and (x.shape[1] != 30 or not np.isfinite(x).all()):
                    raise ValueError("Invalid paired 20+10 regional sample matrix")
                static_chunks.append(x[:, :20].copy())
                causal_chunks.append(x.copy())
                owner_chunks.append(owners.copy())
                coverage.extend(rows)
                del paired, x, owners
            state.commit(frame_index=index)
            frame_receipts.append(dict(run=run, frame_index=index,
                elapsed_h=float(frame["elapsed_h"]), source_name=frame["source_name"],
                source_sha256=frame["source_sha256"],
                native_jpeg_sha256=source["frame_sha256"][index],
                native_manifest_sha256=_sha(source["manifest_path"]),
                optical_cache_key=key,
                optical_cache_payload_sha256=cache_receipt["payload_sha256"],
                temporal=temporal_audit,
                sampled_regions=len(records)))
            processed += 1
            _write_status(root, stage="extracting", frames_done=processed, frames_total=261,
                          run=run, frame_index=index)
            del heads, current, prepared, extra
            gc.collect()
        del baseline, state, source
        gc.collect()
    if processed != 261 or len(frame_receipts) != 261:
        raise ValueError("Complete 13-run/261-frame replay was not performed")
    if len(coverage) != len(regions) or any(row["training_rows"] <= 0 for row in coverage):
        raise ValueError("At least one dense aligned annotation has no unambiguous training rows")
    if Counter(row["phase_class"] for row in coverage) != EXPECTED_DENSE - Counter(water=4):
        raise ValueError("Extracted dense class coverage differs")
    static = np.concatenate(static_chunks, axis=0)
    augmented = np.concatenate(causal_chunks, axis=0)
    owners = np.concatenate(owner_chunks, axis=0)
    if (static.shape[0] != augmented.shape[0] or owners.shape != (len(static),)
            or static.shape[1] != 20 or augmented.shape[1] != 30
            or not np.array_equal(static, augmented[:, :20])):
        raise ValueError("Paired static and causal sample rows differ")
    samples = root / "samples.npz"
    with samples.open("xb") as stream:
        np.savez_compressed(stream, static20=static, causal30=augmented, owners=owners)
    result = dict(format=FORMAT, frames=frame_receipts, coverage=coverage,
        frames_done=processed, runs_done=13,
        sampled_regions=len(coverage), sampled_rows=len(static),
        classes=dict(Counter(row["phase_class"] for row in coverage)),
        preinjection_static_fallback_controls=len(fallback),
        optical_identity_sha256=identity_sha,
        samples=str(samples), samples_sha256=_sha(samples),
        regions_sha256=_sha(root / "regions.json"),
        static20_exact_prefix_of_causal30=True,
        optical_fitted=False, gas_head_fitted=False, phase_maps_generated=False)
    _write_new(root / "extraction.json", result)
    return result


def _serialize_fit(fit: logistic.LogisticFit, diagnostics: dict) -> dict:
    values = asdict(fit)
    for key in ("mean", "scale", "coefficients"):
        values[key] = np.asarray(values[key]).tolist()
    return dict(fit=values, diagnostics=diagnostics)


def _certify_pure_fit(fit: logistic.LogisticFit, diagnostics: dict) -> bool:
    """A dense-only CV arm must finish the unchanged CPU64 IRLS objective."""
    return (diagnostics.get("mode") == "pure-region-irls"
            and diagnostics.get("weak_bags") == 0
            and fit.backend == "cpu" and fit.precision == "float64"
            and bool(fit.converged) and diagnostics.get("converged") is True
            and diagnostics.get("pure_irls_retry", {}).get("final_converged") is True)


def _fit_cv(root: Path, protocol: dict, extraction: dict) -> dict:
    samples = Path(extraction["samples"])
    if _sha(samples) != extraction["samples_sha256"]:
        raise ValueError("Frozen paired sample payload changed")
    regions_data = _read(root / "regions.json")
    if _sha(root / "regions.json") != extraction["regions_sha256"]:
        raise ValueError("Frozen dense sampled region ownership changed")
    regions = [logistic.ReviewedRegion(**record) for record in regions_data["regions"]]
    with np.load(samples, allow_pickle=False) as arrays:
        matrices = {"static20": arrays["static20"], "causal30": arrays["causal30"]}
        owners = arrays["owners"]
    if (len(matrices["static20"]) != len(owners)
            or len(matrices["causal30"]) != len(owners)
            or not np.array_equal(matrices["static20"], matrices["causal30"][:, :20])):
        raise ValueError("Frozen paired sample rows changed")
    all_runs = set(sum((list(v) for v in FOLDS.values()), []))
    fits = {}
    for regularization in REGULARIZATION:
        lam_key = f"{regularization:.3f}"
        fits[lam_key] = {}
        for fold, held_out in FOLDS.items():
            train_runs = all_runs - set(held_out)
            folds = {}
            for arm in ("static20", "causal30"):
                _write_status(root, stage="fitting", lambda_value=regularization,
                              fold=fold, arm=arm)
                fit, diagnostics = logistic.fit_weighted_logistic_mil(
                    matrices[arm], owners, regions,
                    head="co2_presence", runs=train_runs,
                    regularization=regularization, backend="cpu",
                    max_iter=1500, tolerance=2e-5)
                if not _certify_pure_fit(fit, diagnostics):
                    raise ValueError(f"Nonconverged regional CO2 fit: {lam_key} {fold} {arm}")
                folds[arm] = _serialize_fit(fit, diagnostics)
            fits[lam_key][fold] = dict(held_out_runs=list(held_out),
                                       train_runs=sorted(train_runs), arms=folds)
    result = dict(format=FORMAT, protocol_sha256=_sha(root / "protocol.json"),
                  extraction_sha256=_sha(root / "extraction.json"),
                  regions_sha256=_sha(root / "regions.json"),
                  samples_sha256=_sha(samples),
                  folds=fits, selected_production_model=None,
                  native_oof_qa_pending=True,
                  optical_fitted=False, gas_head_fitted=False,
                  phase_maps_generated=False, studio_registered=False)
    _write_new(root / "fold_fits.json", result)
    return result


def run(model: Path, comparison: Path, audit: Path, cache_root: Path,
        output: Path, protocol_document: Path) -> dict:
    protocol, source_audit = _freeze_preflight(model, comparison, audit, cache_root,
                                               output, protocol_document)
    protocol["full_prefix_preflight"] = verify_all_prefix_sources(protocol, source_audit)
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    _write_new(output / "protocol.json", protocol)
    try:
        _write_status(output, stage="preflight_complete", frames_done=0, frames_total=261)
        extracted = _extract(output, protocol, source_audit)
        _write_new(output / "post_extraction_source_check.json",
                   _post_launch_source_check(protocol))
        _write_status(output, stage="extraction_complete", frames_done=261,
                      frames_total=261, sampled_rows=extracted["sampled_rows"])
        fitted = _fit_cv(output, protocol, extracted)
        final_source_check = _post_launch_source_check(protocol)
        final_source_check["full_prefix_recheck"] = verify_all_prefix_sources(
            protocol, source_audit)
        _write_new(output / "post_fit_source_check.json", final_source_check)
        outcome = dict(format=FORMAT, status="exploratory_cv_fitted_pending_native_qa",
            protocol_sha256=_sha(output / "protocol.json"),
            extraction_sha256=_sha(output / "extraction.json"),
            fold_fits_sha256=_sha(output / "fold_fits.json"),
            frames=261, runs=13, sampled_regions=extracted["sampled_regions"],
            sampled_rows=extracted["sampled_rows"],
            later_saved_reviews_seen=final_source_check["later_saved_reviews_seen"],
            no_phase_maps=True, no_publication=True, no_frozen_head_refit=True)
        _write_new(output / "result.json", outcome)
        _write_status(output, stage="exploratory_cv_fitted_pending_native_qa",
                      frames_done=261, frames_total=261)
        return outcome
    except BaseException as error:
        _write_status(output, stage="failed", error=f"{type(error).__name__}: {error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-document", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="Verify frozen sources and all 261 optical payloads without creating a job")
    args = parser.parse_args()
    if args.preflight_only:
        protocol, audit = _freeze_preflight(args.model, args.comparison, args.audit,
                                            args.cache_root, args.output,
                                            args.protocol_document)
        print(json.dumps(dict(status="preflight_only", output_created=False,
            parent_model_sha256=protocol["parent_model_sha256"],
            review_inventory=protocol["review_inventory"],
            full_prefix_preflight=verify_all_prefix_sources(protocol, audit)), indent=2),
            flush=True)
    else:
        print(json.dumps(run(args.model, args.comparison, args.audit,
                             args.cache_root, args.output, args.protocol_document),
                         indent=2), flush=True)


if __name__ == "__main__":
    main()
