"""Freeze one fresh FF_BM MIL retry from the failed fast-profile inputs.

Only the optimizer budget changes. The original request, fitted heads and failed
job are never modified or copied into the new fitting destination. Feature and
staged-image caches are shared read-only and bound by a file-level SHA receipt.
This command prepares inputs; it does not train or schedule a process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower import phase_features, phase_workflow
from darsia.fluidflower.phase_training import logistic, trainer
from darsia.fluidflower.tools import train_studio_optical_then_region_series as series_job


FORMAT = "ffbm-frozen-mil-budget-retry-v1"
ITERATIONS = 30720
COPY_DIRECTORIES = ("input_snapshot", "continuation_evidence", "registered_context", "review_source")
COPY_FILES = ("phase_optical_regions.json", "regional_model.json", "review_retractions.json",
              "physics_inheritance.json", "continuation_preparation.json")


def _implementation_hashes() -> dict[str, str]:
    return {str(Path(module.__file__).resolve()): sha256_file(Path(module.__file__).resolve())
            for module in (logistic, trainer, phase_features, phase_workflow, series_job)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _cache_inventory(root: Path) -> list[dict]:
    """Bind only reusable extraction inputs, never previous optimizer output."""
    _require(root.is_dir(), "Failed request feature cache is missing")
    files = sorted(path for folder in ("assets", "features")
                   for path in (root / folder).rglob("*") if path.is_file())
    _require(files and all(not path.is_symlink() for path in files),
             "Feature cache is empty or contains links")
    features = list((root / "features").rglob("samples.npz"))
    _require(bool(features), "No reusable feature samples were extracted")
    for sample in features:
        metadata = sample.with_name("metadata.json")
        _require(metadata.is_file() and _read(metadata).get("features_sha256") == sha256_file(sample),
                 f"Feature cache metadata and samples differ: {sample}")
    return [dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path),
                 size_bytes=path.stat().st_size) for path in files]


def verify_retry_request(request: dict, *, allow_fitted: bool = False) -> dict:
    """Recheck frozen labels and every cache byte before fitting or scheduling."""
    settings = request.get("settings") or {}
    info = settings.get("bm_mil_retry") or {}
    _require(info.get("format") == FORMAT and settings.get("mil_max_iters") == ITERATIONS
             and settings.get("disable_resume_completed_heads") is True,
             "Request is not the explicit no-checkpoint BM retry")
    root = Path(request["output_root"]).resolve()
    source_path = Path(info["source_request_path"]).resolve()
    source = _read(source_path)
    _require(root != source_path.parent and source_path.parent.name == source["model_id"]
             and sha256_file(source_path) == info["source_request_sha256"],
             "Original frozen request changed or was reused as the retry destination")
    receipt_path = Path(info["receipt_path"]).resolve()
    _require(receipt_path == root / "bm_mil_retry_receipt.json"
             and sha256_file(receipt_path) == info["receipt_sha256"],
             "Retry source and cache receipt changed")
    receipt = _read(receipt_path)
    _require(receipt.get("format") == FORMAT
             and receipt["source_request_sha256"] == info["source_request_sha256"]
             and receipt["source_region_contract_sha256"] == info["source_region_contract_sha256"]
             and receipt["failed_job"] == info["failed_job"],
             "Retry receipt is not bound to the failed source")
    if info.get("strict_implementation_binding"):
        _require(receipt.get("implementation_sha256") == _implementation_hashes(),
                 "Optimizer or coordinator changed after the retry was frozen")
    normalized = dict(settings)
    for key in ("feature_cache_root", "disable_resume_completed_heads", "bm_mil_retry"):
        normalized.pop(key, None)
    normalized["mil_max_iters"] = source["settings"]["mil_max_iters"]
    _require(normalized == source["settings"],
             "Retry changed optical settings other than the MIL iteration budget")
    cache = Path(receipt["feature_cache_root"]).resolve()
    _require(str(cache) == settings.get("feature_cache_root")
             and cache == Path(source["cache_root"]).resolve(),
             "Retry feature cache does not belong to the original request")
    files = receipt["feature_cache_files"]
    _require(files and len(files) == len({row["path"] for row in files}),
             "Retry cache receipt is empty or ambiguous")
    for row in files:
        path = (cache / row["path"]).resolve()
        _require(path.is_relative_to(cache) and path.is_file() and not path.is_symlink()
                 and path.stat().st_size == row["size_bytes"]
                 and sha256_file(path) == row["sha256"],
                 f"Frozen extraction cache changed: {path}")
    _require({row["path"] for row in files} ==
             {path.relative_to(cache).as_posix() for folder in ("assets", "features")
              for path in (cache / folder).rglob("*") if path.is_file()},
             "Frozen extraction cache gained or lost input files")
    for relative, digest in receipt["copied_inputs"].items():
        path = root / relative
        original = source_path.parent / relative
        _require(path.is_file() and original.is_file()
                 and sha256_file(path) == sha256_file(original) == digest,
                 f"Retry input snapshot changed: {relative}")
    if allow_fitted:
        summary = _read(root / "summary.json")
        _require(summary.get("reused_heads") == []
                 and all(row.get("cache_hit") is True
                         for row in summary.get("feature_extraction") or []),
                 "Retry reused a fitted head or failed to reuse the verified feature cache")
    else:
        _require(not (root / "models").exists()
                 and not (Path(request["cache_root"]) / "result_staging").exists(),
                 "A fitted-head checkpoint exists in the fresh retry destination")
    return dict(source_request_sha256=info["source_request_sha256"],
                source_region_contract_sha256=info["source_region_contract_sha256"],
                feature_cache_files=len(files), feature_cache_root=str(cache),
                fitted_heads_reused=False)


def prepare(source_request: Path, failed_job: Path, series_cache: Path,
            parent: Path, library: Path, actor: str,
            *, expected_source_sha256: str) -> dict:
    """Clone only prefit evidence, then verify the exact training-region contract."""
    source_request = source_request.resolve()
    failed_job = failed_job.resolve()
    series_cache = series_cache.resolve()
    _require(source_request.is_file() and source_request.name == "training_request.json"
             and source_request.is_relative_to(series_cache / "studio-continuations"),
             "Select the frozen failed FF_BM request in its series cache")
    _require(sha256_file(source_request) == expected_source_sha256,
             "Failed request SHA does not match the explicit retry selection")
    source = _read(source_request)
    old_root = source_request.parent
    old_cache = Path(source["cache_root"]).resolve()
    checked = series_job.preflight(parent, library, series_cache, "ffbm")
    original = series_job._frozen_request(source_request)
    old_check = series_job.verify_prepared_request(original, checked, actor)
    prepared = _read(failed_job / "optical_prepared.json")
    _require(not (failed_job / "result.json").exists()
             and prepared.get("request_sha256") == expected_source_sha256
             and prepared.get("root") == str(old_root)
             and prepared.get("regions") == old_check["regions"]
             and prepared.get("region_contract_sha256") == old_check["region_contract_sha256"],
             "Failed job is not bound to the selected frozen inputs")
    report = _read(old_root / "cross_validation.json")
    gas = report["generic"]["heads"]["gas_within_co2"]
    _require(gas["final_fit"].get("converged") is False,
             "Retry source does not have the expected nonconverged gas fit")
    old_settings = source["settings"]
    _require(old_settings.get("optical_training_profile") == series_job.FAST_PROFILE
             and old_settings.get("feature_version") == series_job.FEATURE_VERSION
             and old_settings.get("regularization_grid") == list(series_job.GRID)
             and int(old_settings.get("mil_max_iters") or 0) == 15360,
             "Retry would change more than the documented BM MIL budget")
    cache_files = _cache_inventory(old_cache)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_id = f"phase-model-{stamp}-{uuid.uuid4().hex[:8]}"
    workspace = series_cache / "studio-continuations" / uuid.uuid4().hex
    root = workspace / "models" / model_id
    root.mkdir(parents=True, exist_ok=False)
    new_cache = workspace / "cache" / model_id
    for name in COPY_DIRECTORIES:
        shutil.copytree(old_root / name, root / name)
    for name in COPY_FILES:
        shutil.copy2(old_root / name, root / name)
    copied_inputs = {}
    for name in (*COPY_DIRECTORIES, *COPY_FILES):
        source_item = old_root / name
        target_item = root / name
        for src in (source_item.rglob("*") if source_item.is_dir() else (source_item,)):
            if not src.is_file():
                continue
            relative = src.relative_to(old_root)
            digest = sha256_file(src)
            _require(sha256_file(target_item / src.relative_to(source_item)) == digest
                     if source_item.is_dir() else sha256_file(target_item) == digest,
                     f"Copied retry input changed: {relative}")
            copied_inputs[relative.as_posix()] = digest
    receipt = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
                   source_request_path=str(source_request),
                   source_request_sha256=expected_source_sha256,
                   source_region_contract_sha256=old_check["region_contract_sha256"],
                   failed_job=str(failed_job), feature_cache_root=str(old_cache),
                   feature_cache_files=cache_files, copied_inputs=copied_inputs,
                   implementation_sha256=_implementation_hashes(),
                   old_mil_max_iters=15360, new_mil_max_iters=ITERATIONS,
                   old_fitted_heads_copied=False)
    receipt_path = root / "bm_mil_retry_receipt.json"
    _write(receipt_path, receipt)
    retry = json.loads(json.dumps(source))
    retry.update(model_id=model_id, output_root=str(root), cache_root=str(new_cache),
                 created_utc=datetime.now(timezone.utc).isoformat())
    retry["settings"].update(mil_max_iters=ITERATIONS,
                             feature_cache_root=str(old_cache),
                             disable_resume_completed_heads=True,
                             bm_mil_retry=dict(format=FORMAT,
                                 source_request_path=str(source_request),
                                 source_request_sha256=expected_source_sha256,
                                 source_region_contract_sha256=old_check["region_contract_sha256"],
                                 failed_job=str(failed_job),
                                 strict_implementation_binding=True,
                                 receipt_path=str(receipt_path),
                                 receipt_sha256=sha256_file(receipt_path)))
    for bundle in retry["bundles"]:
        for role in ("decisions", "combined_manifest"):
            for field in ("snapshot_path", "original_snapshot_path"):
                if field in bundle[role]:
                    path = Path(bundle[role][field])
                    _require(path.is_relative_to(old_root),
                             "Frozen bundle snapshot escaped the source request")
                    bundle[role][field] = str(root / path.relative_to(old_root))
    if retry.get("review_retractions"):
        retry["review_retractions"]["path"] = str(root / "review_retractions.json")
    _write(root / "training_request.json", retry)
    # The copied physical files are byte-identical; only receipt paths change.
    physics = _read(root / "physics_inheritance.json")
    for row in physics.get("copies", []):
        path = Path(row["target"])
        _require(path.is_relative_to(old_root), "Physical copy target escaped the failed request")
        row["target"] = str(root / path.relative_to(old_root))
    _write(root / "physics_inheritance.json", physics)
    continuation = _read(root / "continuation_preparation.json")
    if continuation.get("destination"):
        continuation["destination"] = str(Path(continuation["destination"]).parent / model_id)
    _write(root / "continuation_preparation.json", continuation)
    # Regenerated metadata is deliberately excluded from byte-identical copies.
    for name in ("physics_inheritance.json", "continuation_preparation.json"):
        receipt["copied_inputs"].pop(name)
    _write(receipt_path, receipt)
    retry["settings"]["bm_mil_retry"]["receipt_sha256"] = sha256_file(receipt_path)
    _write(root / "training_request.json", retry)
    verified = series_job.verify_prepared_request(series_job._frozen_request(root / "training_request.json"),
                                                  checked, actor)
    _require(verified["region_contract_sha256"] == old_check["region_contract_sha256"]
             and verified["regions"] == old_check["regions"]
             and verified["class_counts"] == old_check["class_counts"],
             "Retry changed the frozen labels or source geometry")
    source_after = sha256_file(source_request)
    _require(source_after == expected_source_sha256, "Source request changed during retry preparation")
    cache_check = verify_retry_request(retry)
    return dict(format=FORMAT, prepared=True, fitted=False, scheduled=False,
                request_path=str(root / "training_request.json"),
                request_sha256=sha256_file(root / "training_request.json"),
                model_id=model_id, regions=verified["regions"],
                region_contract_sha256=verified["region_contract_sha256"],
                cache=cache_check, source_request_unchanged=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--failed-job", type=Path, required=True)
    parser.add_argument("--series-cache", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--actor", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_request, args.failed_job, args.series_cache,
                             args.parent, args.library, args.actor,
                             expected_source_sha256=args.source_sha256),
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
