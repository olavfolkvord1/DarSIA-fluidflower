"""Map disjoint Track1 runs on one host without fitting or publishing aggregates.

Claims are permanent host ownership records on the shared candidate. A resumed
host may reuse its own completed packages or finish its incomplete runs. Root
controllers alone assemble partial/full Studio campaigns after reading these
receipts; worker hosts never publish root result or Studio registration files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import uuid

from darsia.fluidflower.project import sha256_file, utc_now

CONTRACT = "ffac-track1-inference-shard-v1"
CLAIM_CONTRACT = "ffac-track1-inference-run-claim-v1"


class RunBusyError(RuntimeError):
    """An earlier worker still owns this run's output paths."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _host_lease(path, *, busy_error=None):
    """Nonblocking OS lock releases after a crash, including on the NAS share."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise (busy_error or RuntimeError(
                "This host already has an active inference shard")) from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _claim(path, *, host, model_identity, source_sha, session):
    value = dict(contract=CLAIM_CONTRACT, run=path.stem, host=host,
                 machine=socket.gethostname(), pid=os.getpid(), session=session,
                 model_identity=model_identity, source_sha256=source_sha,
                 created_utc=utc_now())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        saved = _read(path)
        if (saved.get("contract") != CLAIM_CONTRACT or saved.get("run") != path.stem
                or saved.get("host") != host):
            raise ValueError(f"{path.stem}: run already claimed by another host")
        if (saved.get("model_identity") != model_identity
                or saved.get("source_sha256") != source_sha):
            raise ValueError(f"{path.stem}: claim identifies another model/source")
        return saved, False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return value, True


def _completed_package(candidate, run, identity):
    """Reuse only finalized maps bound to this candidate's exact run source."""
    from darsia.fluidflower.tools.phase_reference_readiness import (
        RECEIPT, require_reference_readiness,
    )
    from PIL import Image
    package = candidate / "validation/run_packages" / run
    required = [package / "validation_binding.json", package / "postflight_manifest.json",
                package / RECEIPT, candidate / "validation/presentation" / run / "media.json"]
    if not all(path.is_file() for path in required):
        return None
    binding, campaign = map(_read, required[:2])
    source_sha = sha256_file(candidate / "review_source" / run / "postflight_manifest.json")
    for value in (binding, campaign):
        if ((value.get("model_id"), value.get("model_sha256")) != tuple(identity)
                or value.get("source_manifest_sha256") != source_sha):
            raise ValueError(f"{run}: completed package identifies another model/source")
    provenance = binding.get("inference_provenance") or {}
    checks_path = package / "backend_checks.json"
    checks = _read(checks_path) if checks_path.is_file() else {}
    backend = provenance.get("backend")
    if (provenance.get("requested_backend") != "cpu" or not isinstance(backend, str)
            or not backend.startswith("cpu-") or checks.get("backend") != backend):
        raise ValueError(f"{run}: completed package lacks CPU inference provenance")
    jobs = campaign.get("jobs") or []
    if (len(jobs) != 1 or jobs[0].get("run") != run or binding.get("run_count") != 1
            or binding.get("selected_candidates") != {run: jobs[0].get("candidate_id")}
            or not binding.get("completed_utc")):
        raise ValueError(f"{run}: completed package is not a complete single-run timeline")
    readiness = _read(package / RECEIPT)
    if readiness.get("status") not in {"ready", "deferred"}:
        return None
    require_reference_readiness(package, allow_pending=True)
    job = jobs[0]
    media_job = _read(required[3]).get("job") or {}
    if (media_job.get("run") != run or media_job.get("candidate_id") != job.get("candidate_id")):
        raise ValueError(f"{run}: presentation belongs to another package")
    for field in ("summary", "per_frame_csv"):
        path = Path(media_job.get(field) or "")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{run}: completed presentation lacks {field}")
    native = _read(Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json")
    count, parameters = native.get("frame_count"), str(job.get("params_sha256") or "")
    maps = Path(job.get("phase_map_output_dir") or "").resolve()
    if (type(count) is not int or count <= 0 or len(parameters) != 64
            or not maps.is_relative_to(package.resolve())):
        raise ValueError(f"{run}: completed map timeline is invalid")
    folder = maps / "native_phase_maps" / run / f"A_{parameters[:16]}"
    for layer in (folder, folder / "aqueous", folder / "gas", folder / "mixed"):
        for index in range(count):
            path = layer / f"frame_{index:03d}.png"
            if not path.is_file():
                return None
            with Image.open(path) as image:
                image.verify()
    return dict(run=run, package=str(package), frame_count=count, resumed_complete=True,
                binding_sha256=sha256_file(required[0]),
                postflight_sha256=sha256_file(required[1]))


def _map_run_unlocked(candidate, run, threads, memory_gb, host, session, identity):
    # An inherited local broker address must never route a shard to another
    # controller's GPU or to a broker inaccessible from the remote host.
    os.environ.pop("PHASE_FRAME_BROKER", None)
    from darsia.fluidflower.tools.prepare_optical_region_studio import run_one, complete_run
    candidate = Path(candidate)
    try:
        cached = _completed_package(candidate, run, identity)
        if cached is None:
            run_one(candidate, run, fresh_optical=True, threads=threads,
                    memory_gb=memory_gb, execution_backend="cpu",
                    generate_review_proposals=False, eager_references=False)
            result = complete_run(candidate, run, threads, allow_inference=False,
                                  eager_presentation=False, eager_references=False)
            if result.get("needs_inference"):
                raise ValueError(f"{run}: finalization unexpectedly requested inference")
            cached = _completed_package(candidate, run, identity)
            if cached is None:
                raise ValueError(f"{run}: mapping did not produce a finalized native package")
            cached["resumed_complete"] = False
        result = dict(cached, contract=CONTRACT, status="succeeded", host=host,
                      session=session, pid=os.getpid(), completed_utc=utc_now(),
                      execution_backend="cpu", optical_training_performed=False,
                      regional_training_performed=False)
    except Exception as error:
        result = dict(contract=CONTRACT, status="failed", run=run, host=host,
                      session=session, pid=os.getpid(), completed_utc=utc_now(),
                      error=f"{type(error).__name__}: {error}")
        _atomic(candidate / "inference_shards/hosts" / host / "runs" / (run + ".json"), result)
        raise
    _atomic(candidate / "inference_shards/hosts" / host / "runs" / (run + ".json"), result)
    return result


def _map_run(candidate, run, threads, memory_gb, host, session, identity):
    """Keep an orphaned worker from racing a resumed host on the same run."""
    candidate = Path(candidate)
    with _host_lease(candidate / "inference_shards/run_locks" / (run + ".lock"),
                     busy_error=RunBusyError(f"{run}: inference worker is still active")):
        return _map_run_unlocked(candidate, run, threads, memory_gb, host, session, identity)


def map_shard(candidate, host, runs, cpus=24, memory_gb=180, workers=4):
    """Infer explicit disjoint runs; return packages for the root's assembly."""
    from darsia.fluidflower.tools.ffac_track1_display_mask_rerun import verify_inference_rerun
    candidate = Path(candidate).resolve()
    host = str(host)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", host):
        raise ValueError("Host must be a distinct simple filesystem name")
    runs = [str(run).lower() for run in runs]
    if (not runs or len(set(runs)) != len(runs)
            or any(not re.fullmatch(r"ac\d+", run) for run in runs)):
        raise ValueError("Shard requires a nonempty explicit list of distinct FF_AC runs")
    if (type(cpus) is not int or cpus < 1 or type(workers) is not int or workers < 1
            or not math.isfinite(float(memory_gb)) or float(memory_gb) <= 0):
        raise ValueError("CPU, memory and worker budgets must be positive")
    incremental = (candidate / "inference_incremental_plan.json").is_file()
    if incremental:
        from darsia.fluidflower.tools.ffac_track1_incremental_inference import verify_incremental
        receipt = verify_incremental(candidate, runs)
    else:
        receipt = verify_inference_rerun(candidate)
    identity = receipt["model_identity"]
    allowed = {row["run"] for row in (receipt["expected_sources"] if incremental else receipt["source_bindings"])}
    if not set(runs) <= allowed:
        raise ValueError("Shard contains runs outside the frozen inference cohort")
    actual_cpus = min(cpus, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    active_workers = min(workers, actual_cpus, len(runs))
    threads = max(1, actual_cpus // active_workers)
    per_worker_memory = float(memory_gb) / active_workers
    host_root = candidate / "inference_shards/hosts" / host
    session = uuid.uuid4().hex
    record = dict(contract=CONTRACT, candidate=str(candidate), host=host,
                  machine=socket.gethostname(), controller_pid=os.getpid(), session=session,
                  model_identity=identity, runs=runs, status="preparing", started_utc=utc_now(),
                  backend="cpu", requested_cpus=cpus, cpu_limit=actual_cpus,
                  workers=active_workers, threads_per_worker=threads,
                  memory_gb=float(memory_gb), memory_gb_per_worker=per_worker_memory,
                  optical_training_performed=False, regional_training_performed=False,
                  root_campaign_published=False, completed=[], failed=[])
    with _host_lease(host_root / "controller.lock"):
        created = []
        try:
            for run in runs:
                source_sha = sha256_file(candidate / "review_source" / run / "postflight_manifest.json")
                path = candidate / "inference_shards/claims" / (run + ".json")
                _, is_new = _claim(path, host=host, model_identity=identity,
                                   source_sha=source_sha, session=session)
                if is_new:
                    created.append(path)
        except Exception:
            # No work has started. Undo only the claims this invocation created.
            for path in created:
                if _read(path).get("session") == session:
                    path.unlink()
            raise
        record["status"] = "running"
        _atomic(host_root / "result.json", record)
        if active_workers == 1:
            for run in runs:
                try:
                    record["completed"].append(_map_run(candidate, run, threads,
                        per_worker_memory, host, session, identity))
                except Exception as error:
                    record["failed"].append(dict(run=run, error=f"{type(error).__name__}: {error}"))
                _atomic(host_root / "result.json", record)
        else:
            with ProcessPoolExecutor(max_workers=active_workers,
                                     mp_context=mp.get_context("spawn")) as pool:
                futures = {pool.submit(_map_run, str(candidate), run, threads,
                    per_worker_memory, host, session, identity): run for run in runs}
                for future in as_completed(futures):
                    try:
                        record["completed"].append(future.result())
                    except Exception as error:
                        record["failed"].append(dict(run=futures[future],
                            error=f"{type(error).__name__}: {error}"))
                    _atomic(host_root / "result.json", record)
        try:
            if incremental:
                verify_incremental(candidate, runs)
            else:
                verify_inference_rerun(candidate)
        except Exception as error:
            record["failed"].append(dict(run=None,
                error=f"Frozen candidate changed: {type(error).__name__}: {error}"))
        record.update(status="failed" if record["failed"] else "succeeded",
                      completed_utc=utc_now(), packages=[row["package"] for row in record["completed"]])
        _atomic(host_root / "result.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--runs", required=True, help="Explicit comma-separated disjoint run names")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpus", type=int, default=24)
    parser.add_argument("--memory-gb", type=float, default=180)
    args = parser.parse_args(argv)
    result = map_shard(args.root, args.host, [part.strip() for part in args.runs.split(",")],
                       cpus=args.cpus, memory_gb=args.memory_gb, workers=args.workers)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
