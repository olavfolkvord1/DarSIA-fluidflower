"""Resume a source-bound FF_AC V5 regional fit at Studio preparation.

This tool never extracts features or fits heads. It consumes one completed,
SHA-verified regional fit whose original Studio preparation failed, creates a
new detached candidate, and then generates the native maps on request.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from darsia.fluidflower import phase_runtime
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools import train_ffac_v5_raw_candidate as candidate
from darsia.fluidflower.tools import train_phase_model_further as optical
from darsia.fluidflower.tools.prepare_optical_region_studio import (
    _is_detached_review_candidate, prepare,
)
from darsia.fluidflower.tools.train_studio_region_update import read, write


CONTRACT = "ffac-v5-track1-regional-fit-recovery-v1"
EXPECTED_FAILURE = "Physical context does not match the exact AC47 two-arm revision"
PRIORITY = {"ac28": 0, "ac45": 1, "ac56": 2, "ac22": 3}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact(path: str | Path) -> Path:
    return Path(path).resolve()


def _frame_backend(requested: str, output: Path, frozen: str | None) -> tuple[str, dict]:
    """Choose once, before submitting SHA-bound registered frame tasks."""
    _require(requested in {"auto", "cpu", "cuda"}, "Unknown frame backend")
    if frozen is not None:
        _require(frozen in {"cpu", "cuda"} and
                 requested in {"auto", frozen},
                 "A resumed map cannot change its frozen optical backend")
        requested = frozen
    if requested == "cpu":
        return "cpu", {"selected": "cpu", "reason": "explicit CPU selection"}
    manifest = read(output / "review_source/postflight_manifest.json")
    _require(bool(manifest.get("jobs")), "No native run to size CUDA input")
    _source_root, _native_path, native = optical.native_source(manifest["jobs"][0])
    _x, _y, width, height = native["content_box_xywh"]
    required = int(width) * int(height) * 260 + 512 * 1024**2
    try:
        import cupy as cp
        devices = int(cp.cuda.runtime.getDeviceCount())
        if devices < 1:
            raise RuntimeError("No CUDA device")
        free, total = (int(value) for value in cp.cuda.runtime.memGetInfo())
        if free < required:
            raise RuntimeError(f"CUDA free memory {free} is below {required} required bytes")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if requested == "cuda":
            raise RuntimeError(f"Frozen CUDA map backend unavailable: {error}") from error
        return "cpu", {"selected": "cpu", "reason":
                       f"CUDA preflight unavailable: {type(error).__name__}: {error}",
                       "required_bytes": required}
    return "cuda", {"selected": "cuda", "devices": devices,
                    "free_bytes": free, "total_bytes": total,
                    "required_bytes": required, "reason": "CUDA preflight passed"}


def _metadata_only_fit(request: dict, fitted: dict, hashes: dict,
                       experiment: Path) -> dict:
    """Add missing Track1 scope to a copy, never to the converged fit."""
    extracted = read(_exact(hashes["extraction_path"]))
    evidence = read(_exact(hashes["evidence_path"]))
    audit = evidence.get("audit") or {}
    settings = request.get("settings") or {}
    scope = audit.get("v5_chemistry_fit_scope")
    partition = audit.get("v5_review_fit_partition")
    _require(isinstance(scope, dict) and isinstance(partition, dict) and
             scope == extracted.get("v5_chemistry_fit_scope") ==
                 settings.get("v5_chemistry_fit_scope") and
             partition == extracted.get("v5_review_fit_partition"),
             "Frozen regional evidence, extraction, and optical chemistry scope differ")
    request_partition = settings.get("v5_review_fit_partition") or {}
    _require(all(request_partition.get(key) == partition.get(key)
                 for key in ("contract", "path", "sha256", "region_scope")) and
             partition.get("contract") ==
                 "ffac-v5-track1-frozen-review-fit-evaluation-partition-v1" and
             partition.get("region_scope") == "all52" and
             sha256_file(_exact(partition["path"])) == partition["sha256"],
             "Frozen 4048-region fit/evaluation partition differs")
    rows = read(_exact(partition["path"]))["rows"]
    roles = {row["region_id"]: row["role"] for row in rows}
    coverage = extracted["coverage"]
    excluded_runs = set(scope["excluded_runs"])
    _require(len(rows) == len(roles) == len(coverage) == 4048 and
             {row["region_id"] for row in coverage} == set(roles) and
             all((int(row["training_rows"]) > 0) ==
                 (row["status"] == "included") for row in coverage) and
             all(int(row["training_rows"]) == 0 or
                 (roles[row["region_id"]] == "fit" and
                  row["run"].casefold() not in excluded_runs)
                 for row in coverage) and
             sum(int(row["training_rows"]) for row in coverage) ==
                 int(fitted["fit_report"]["active_feature_rows"]),
             "Converged regional rows include evaluation-only or excluded chemistry")
    _require("v5_chemistry_fit_scope" not in fitted and
             "v5_review_fit_partition" not in fitted,
             "Source regional fit already contains V5 metadata; refuse rewriting it")
    experiment.mkdir(parents=True, exist_ok=False)
    copy_model = dict(fitted)
    copy_model["v5_chemistry_fit_scope"] = scope
    copy_model["v5_review_fit_partition"] = partition
    copied_path = experiment / "regional_model.json"
    write(copied_path, copy_model)
    shutil.copy2(_exact(hashes["fit_path"]).parent / "protocol.json",
                 experiment / "protocol.json")
    saved = read(copied_path)
    _require({key: value for key, value in saved.items()
              if key not in ("v5_chemistry_fit_scope", "v5_review_fit_partition")}
             == fitted and saved["v5_chemistry_fit_scope"] == scope and
             saved["v5_review_fit_partition"] == partition,
             "Metadata-only regional fit copy changed trained coefficients")
    return {
        "contract": "ffac-v5-track1-regional-fit-metadata-copy-v1",
        "source_regional_model_path": hashes["fit_path"],
        "source_regional_model_sha256": hashes["fit_sha256"],
        "copied_regional_model_path": str(copied_path),
        "copied_regional_model_sha256": sha256_file(copied_path),
        "protocol_sha256": sha256_file(experiment / "protocol.json"),
        "evidence_sha256": hashes["evidence_sha256"],
        "extraction_sha256": hashes["extraction_sha256"],
        "chemistry_fit_scope": scope,
        "review_fit_partition_sha256": partition["sha256"],
        "original_model_and_coefficients_preserved": True,
        "included_regions": sum(row["status"] == "included" for row in coverage),
        "sampled_rows": sum(int(row["training_rows"]) for row in coverage),
        "evaluation_or_unsupervised_regions": sum(
            int(row["training_rows"]) == 0 for row in coverage),
    }


def _fit_inputs(request_path: Path, optical_job: Path,
                regional_job: Path) -> tuple[dict, dict, Path, Path, dict, dict]:
    request, binding = candidate.verify_prepared(request_path)
    hidden, _checkpoint = candidate.verify_optical_checkpoint(
        request_path, optical_job, binding)
    regional_job = _exact(regional_job)
    fit_path = regional_job / "full_aqueous" / "regional_model.json"
    extraction_path = regional_job / "extraction.json"
    evidence_path = regional_job / "evidence.json"
    parity_path = regional_job / "snapshot_parity.json"
    launch = read(regional_job / "launch_request.json")
    failed = read(regional_job / "status.json")
    fitted = read(fit_path)
    extracted = read(extraction_path)
    parity = read(parity_path)
    evidence = read(evidence_path)
    samples = _exact(extracted["samples"])
    old_output = _exact(launch["output"])
    _require(failed.get("stage") == "failed" and
             EXPECTED_FAILURE in str(failed.get("error") or "") and
             _exact(launch["parent"]) == hidden and
             _exact(launch["job"]) == regional_job and
             old_output.exists() and
             launch.get("review_candidate_only") is True and
             launch.get("metric_postprocessing") is False and
             launch.get("bottom_gas_guard") is False and
             launch.get("region_edge_guard") is False and
             launch.get("free_gas_cleanup") is True and
             launch.get("nominal_review_gas_minimum_mm2") == 2.5,
             "Regional recovery is not the exact failed Track1 Studio-preparation job")
    fit_report = fitted.get("fit_report") or {}
    _require(_exact(fitted["parent_model"]) == hidden and
             _exact(fitted["evidence_path"]) == evidence_path and
             fitted["evidence_sha256"] == sha256_file(evidence_path) and
             fitted["protocol_sha256"] == sha256_file(
                 regional_job / "full_aqueous" / "protocol.json") and
             fit_report.get("converged") is True and
             fit_report.get("new_oof_validation") is False and
             fit_report.get("inherited_oof_metrics_not_reused") is True and
             int(fit_report.get("active_feature_rows") or 0) > 0 and
             float(fit_report["maximum_absolute_gradient"]) <=
                 float(fit_report["gradient_tolerance"]),
             "Recovered regional head lacks its converged source-bound fit")
    _require(samples.is_file() and sha256_file(samples) ==
             extracted["samples_sha256"] and
             len(extracted.get("frames") or []) == 236 and
             len(extracted.get("coverage") or []) == 4048 and
             parity.get("same_frozen_pairs") is True and
             parity.get("regions") == 4048 and
             parity.get("optical_annotation_contract_sha256") ==
                 parity.get("regional_annotation_contract_sha256") ==
                 binding["annotation_binding"]["annotation_contract_sha256"] and
             _exact(evidence["parent"]) == hidden and
             (evidence.get("audit") or {}).get("v5_raw_source_binding", {}).get(
                 "receipt_sha256") == binding["materialization_receipt_sha256"],
             "Recovered regional samples differ from the frozen optical evidence")
    return request, binding, hidden, old_output, fitted, {
        "fit_path": str(fit_path), "fit_sha256": sha256_file(fit_path),
        "extraction_path": str(extraction_path),
        "extraction_sha256": sha256_file(extraction_path),
        "samples_path": str(samples), "samples_sha256": extracted["samples_sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": fitted["evidence_sha256"],
        "parity_path": str(parity_path), "parity_sha256": sha256_file(parity_path),
        "launch_path": str(regional_job / "launch_request.json"),
        "launch_sha256": sha256_file(regional_job / "launch_request.json"),
        "failed_status_path": str(regional_job / "status.json"),
        "failed_status_sha256": sha256_file(regional_job / "status.json"),
    }


def prepare_recovery(request_path: Path, optical_job: Path,
                     regional_job: Path, job: Path) -> dict:
    request_path, optical_job = _exact(request_path), _exact(optical_job)
    request, binding, hidden, old_output, fitted, hashes = _fit_inputs(
        request_path, optical_job, regional_job)
    job = _exact(job)
    root = _exact(request["output_root"])
    _require(not job.exists() and not job.is_relative_to(root)
             and not job.is_relative_to(optical_job)
             and not job.is_relative_to(_exact(regional_job))
             and not job.is_relative_to(old_output),
             "Choose a new detached recovery job; leave the failed candidate intact")
    output = job / "regional_candidate" / f"{old_output.name}-recovered"
    _require(not output.exists(), "Recovery candidate already exists")
    job.mkdir(parents=True, exist_ok=False)
    receipt = {
        "contract": CONTRACT, "stage": "preparing",
        "prepared_request_path": str(request_path),
        "prepared_request_sha256": sha256_file(request_path),
        "source_binding_sha256": sha256_file(root / "v5_raw_source_binding.json"),
        "optical_job": str(optical_job),
        "optical_checkpoint_sha256": sha256_file(
            optical_job / "optical_checkpoint.json"),
        "hidden_optical_parent": str(hidden),
        "regional_job": str(_exact(regional_job)),
        "failed_candidate_preserved": str(old_output),
        "candidate": str(output),
        "regional_fit_inputs": hashes,
        "regional_fit_reused": True,
        "regional_extraction_repeated": False,
        "regional_fit_repeated": False,
        "native_maps_built": False,
        "studio_published": False,
        "mass_analysis_allowed": False,
        "prepare_code_sha256": sha256_file(Path(prepare.__code__.co_filename)),
    }
    candidate._write_new(job / "recovery_receipt.json", receipt)
    collection = hidden / "review_source"
    experiment = job / "regional_run" / "full_aqueous"
    metadata = _metadata_only_fit(request, fitted, hashes, experiment)
    write(job / "fit_metadata_copy_receipt.json", metadata)
    old_launch = read(_exact(hashes["launch_path"]))
    detached_launch = dict(old_launch, output=str(output),
                           job=str(job / "regional_run"))
    write(job / "regional_run" / "launch_request.json", detached_launch)
    receipt["fit_metadata_copy_receipt_sha256"] = sha256_file(
        job / "fit_metadata_copy_receipt.json")
    receipt["detached_launch_sha256"] = sha256_file(
        job / "regional_run" / "launch_request.json")
    write(job / "recovery_receipt.json", receipt)
    prepare(experiment, collection, output,
            fresh_optical=True, bottom_gas_guard=False,
            region_edge_guard=False, metric_postprocessing=False,
            nominal_review_gas_minimum=True)
    manifest_path = output / "review_source" / "postflight_manifest.json"
    manifest = read(manifest_path)
    _require(len(manifest["jobs"]) == 43 and len({
        str(row["run"]).casefold() for row in manifest["jobs"]}) == 43,
        "Recovered Studio campaign lost a canonical run")
    manifest["jobs"] = sorted(manifest["jobs"],
                              key=lambda row: PRIORITY.get(
                                  str(row["run"]).casefold(), 4))
    write(manifest_path, manifest)
    write(output / "studio_review_priority.json", {
        "contract": "ffac-v5-track1-first-maps-priority-v1",
        "requested_runs": list(PRIORITY),
        "ordered_runs": [row["run"] for row in manifest["jobs"]],
        "detached_candidate_only": True,
    })
    write(output / "expected_review_runs.json", {
        "runs": [row["run"] for row in read(collection /
            "postflight_manifest.json")["jobs"]],
    })
    candidate_request = read(output / "training_request.json")
    candidate_request["created_by"] = request["created_by"]
    candidate_request["settings"]["iteration"] = (
        "FF_AC V5 RAW · Photo-only supported sand · NOT FOR MASS ANALYSIS · Studio review")
    write(output / "training_request.json", candidate_request)
    profile = phase_runtime.load_physics_profile(output)
    _require((profile or {}).get("v5_track1_review_scope", {}).get(
                 "mass_analysis_allowed") is False,
             "Recovered candidate lost its provisional photo-only scope")
    receipt.update(stage="prepared", candidate_request_sha256=sha256_file(
        output / "training_request.json"),
        candidate_regional_model_sha256=sha256_file(
            output / "regional_model.json"),
        candidate_manifest_sha256=sha256_file(manifest_path))
    write(job / "recovery_receipt.json", receipt)
    return receipt


def map_recovery(job: Path, *, backend: str = "auto", cpus: int = 24,
                 memory_gb: float = 180.0, workers: int = 1,
                 broker_cpu_workers: int = 8) -> dict:
    job = _exact(job)
    receipt_path = job / "recovery_receipt.json"
    receipt = read(receipt_path)
    _require(receipt.get("contract") == CONTRACT and
             receipt.get("stage") in {"prepared", "mapping"} and
             receipt.get("regional_fit_reused") is True and
             receipt.get("regional_extraction_repeated") is False and
             receipt.get("regional_fit_repeated") is False,
             "Regional recovery was not prepared from a frozen fit")
    output = _exact(receipt["candidate"])
    _require(output.is_relative_to(job) and output.is_dir() and
             _is_detached_review_candidate(output) and
             sha256_file(job / "fit_metadata_copy_receipt.json") ==
                 receipt["fit_metadata_copy_receipt_sha256"] and
             sha256_file(job / "regional_run/launch_request.json") ==
                 receipt["detached_launch_sha256"] and
             sha256_file(output / "training_request.json") ==
                 receipt["candidate_request_sha256"] and
             sha256_file(output / "regional_model.json") ==
                 receipt["candidate_regional_model_sha256"] and
             sha256_file(output / "review_source/postflight_manifest.json") ==
                 receipt["candidate_manifest_sha256"],
             "Recovered candidate changed before native map generation")
    hashes = receipt["regional_fit_inputs"]
    for stem in ("fit", "extraction", "samples", "evidence", "parity",
                 "launch", "failed_status"):
        _require(sha256_file(_exact(hashes[f"{stem}_path"])) ==
                 hashes[f"{stem}_sha256"],
                 f"Frozen regional {stem} changed before maps")
    copy_receipt = read(job / "fit_metadata_copy_receipt.json")
    _require(copy_receipt.get("source_regional_model_sha256") == hashes["fit_sha256"]
             and sha256_file(_exact(copy_receipt["copied_regional_model_path"])) ==
                 copy_receipt["copied_regional_model_sha256"] ==
                 receipt["candidate_regional_model_sha256"],
             "Recovered model is not the frozen-fit metadata-only copy")
    request_path = _exact(receipt["prepared_request_path"])
    request, binding = candidate.verify_prepared(request_path)
    hidden, _checkpoint = candidate.verify_optical_checkpoint(
        request_path, _exact(receipt["optical_job"]), binding)
    _require(sha256_file(request_path) == receipt["prepared_request_sha256"]
             and sha256_file(_exact(request["output_root"]) /
                 "v5_raw_source_binding.json") == receipt["source_binding_sha256"]
             and hidden == _exact(receipt["hidden_optical_parent"]),
             "Recovered maps no longer bind to the fitted optical source")
    _require(workers == 1 and 1 <= broker_cpu_workers <= min(14, cpus)
             and cpus >= 1 and memory_gb > 0,
             "Invalid CPU worker or memory budget")
    selected_backend, preflight = _frame_backend(
        backend, output, receipt.get("frame_backend"))
    from darsia.fluidflower.tools import phase_frame_workers
    receipt["frame_backend"] = selected_backend
    receipt["frame_backend_preflight"] = preflight
    receipt["frame_broker_contract"] = phase_frame_workers.VERSION
    receipt["frame_broker_code_sha256"] = sha256_file(
        Path(phase_frame_workers.__file__))
    receipt["frame_broker_cpu_workers"] = broker_cpu_workers
    receipt["stage"] = "mapping"
    write(receipt_path, receipt)
    command = [sys.executable, "-u", "-m",
               "darsia.fluidflower.tools.prepare_optical_region_studio",
               "--root", str(output), "--resume", "--fresh-optical",
               "--workers", str(workers), "--cpus", str(cpus),
               "--memory-gb", str(memory_gb), "--backend", selected_backend]
    # The child creates its run coordinator with Windows spawn. It inherits the
    # localhost manager address while this owning process keeps the broker alive.
    with phase_frame_workers.serve_frame_workers(
            broker_cpu_workers, int(selected_backend == "cuda")) as broker:
        environment = dict(os.environ, **broker.connection_env,
                           PHASE_FRAME_CPU_OPTICAL_WITH_GPU="0")
        initial = broker.snapshot()
        _require(initial["optical_policy"] ==
                 ("gpu-first" if selected_backend == "cuda" else "cpu-only"),
                 "Registered frame broker selected an unexpected optical policy")
        receipt["frame_broker_initial"] = initial
        write(receipt_path, receipt)
        try:
            subprocess.run(command, check=True, env=environment)
        except BaseException as error:
            receipt["frame_broker_last"] = broker.snapshot()
            receipt["last_mapping_error"] = f"{type(error).__name__}: {error}"
            write(receipt_path, receipt)
            raise
        receipt["frame_broker_last"] = broker.snapshot()
        receipt.pop("last_mapping_error", None)
        write(receipt_path, receipt)
    preparation = read(output / "studio_preparation.json")
    if not preparation.get("presentation_ready"):
        subprocess.run([sys.executable, "-u", "-m",
                        "darsia.fluidflower.tools.finish_optical_region_studio",
                        "--root", str(output), "--cpus", str(cpus),
                        "--memory-gb", str(memory_gb)], check=True)
    collection = read(output / "validation/postflight_manifest.json")
    jobs = collection.get("jobs") or []
    _require(len(jobs) == 43 and len({str(row["run"]).casefold()
                                     for row in jobs}) == 43,
             "Recovered maps lack the full 43-run Studio collection")
    canonical = {row["run"].casefold(): row for row in read(
        _exact(binding["materialization_receipt_path"]))["canonical_mapping"]}
    for row in jobs:
        run = str(row["run"]).casefold()
        source_root, manifest_path, _native = optical.native_source(row)
        _require(_exact(source_root) == _exact(canonical[run]["materialized_review_root"])
                 and sha256_file(manifest_path) ==
                     canonical[run]["materialized_manifest_sha256"]
                 and row.get("phase_model_only") is True
                 and row.get("quality_flags", {}).get(
                     "human_review_required") is True,
                 f"Recovered map uses another native source or review policy: {run}")
    native = candidate.verify_native_maps(jobs)
    result = {
        "contract": CONTRACT, "candidate": str(output),
        "regional_fit_reused": True, "regional_extraction_repeated": False,
        "regional_fit_repeated": False,
        "native_maps_built": True, "native_studio_runs": native["runs"],
        "native_studio_frames": native["frames"],
        "native_phase_pngs": native["phase_pngs"],
        "native_phase_layer_pngs": native["phase_layer_pngs"],
        "validation_only": True, "mass_analysis_allowed": False,
        "qa_pending": True, "studio_published": False,
        "regional_model_sha256": hashes["fit_sha256"],
        "optical_checkpoint_sha256": receipt["optical_checkpoint_sha256"],
    }
    candidate._write_new(job / "result.json", result)
    receipt["stage"] = "mapped"
    receipt["native_maps_built"] = True
    write(receipt_path, receipt)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--map", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--optical-job", type=Path)
    parser.add_argument("--regional-job", type=Path)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpus", type=int, default=24)
    parser.add_argument("--memory-gb", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--broker-cpu-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.prepare:
        _require(all(value is not None for value in (
            args.request, args.optical_job, args.regional_job)),
            "Prepare requires request, optical job, and fitted regional job")
        result = prepare_recovery(args.request, args.optical_job,
                                  args.regional_job, args.job)
    else:
        result = map_recovery(args.job, backend=args.backend, cpus=args.cpus,
                              memory_gb=args.memory_gb, workers=args.workers,
                              broker_cpu_workers=args.broker_cpu_workers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
