"""Map one host's frozen run partition as individual source seals appear."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import time
import uuid

from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.ffac_track1_incremental_inference import _atomic, verify_incremental
from darsia.fluidflower.tools import run_ffac_track1_inference_shard as shard


POLL_SECONDS = 10
STATUS_WRITE_ATTEMPTS = 20
STATUS_WRITE_RETRY_SECONDS = 0.05


def _write_hot_status(path, value):
    """Retry only transient SMB sharing errors on progress-file replacement."""
    for attempt in range(STATUS_WRITE_ATTEMPTS):
        try:
            return _atomic(path, value)
        except PermissionError:
            if attempt == STATUS_WRITE_ATTEMPTS - 1:
                raise
            time.sleep(STATUS_WRITE_RETRY_SECONDS)


def run_host(root: Path, host: str):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", host):
        raise ValueError("Host must be a distinct simple filesystem name")
    host_path = root / "inference_host_plans" / (host + ".json")
    host_bytes = host_path.read_bytes()
    host_sha = hashlib.sha256(host_bytes).hexdigest()
    plan = json.loads(host_bytes.decode("utf-8-sig"))
    frozen = verify_incremental(root)
    if (plan.get("contract") != "ffac-track1-incremental-host-plan-v1"
            or plan.get("host") != host or Path(plan.get("candidate", "")).resolve() != root
            or plan.get("model_identity") != frozen["model_identity"]
            or plan.get("plan_sha256") != sha256_file(root / "inference_incremental_plan.json")):
        raise ValueError("Host partition differs from the immutable model/source plan")
    runs = plan["runs"]
    if (not runs or len(set(runs)) != len(runs)
            or any(not re.fullmatch(r"ac\d+", run) for run in runs)
            or not set(runs) <= {row["run"] for row in frozen["expected_sources"]}):
        raise ValueError("Host partition is not an explicit distinct subset")
    cpus, workers, memory_gb = plan["cpus"], plan["workers"], plan["memory_gb"]
    if (type(cpus) is not int or cpus < 1 or type(workers) is not int or workers < 1
            or not math.isfinite(float(memory_gb)) or float(memory_gb) <= 0):
        raise ValueError("CPU, memory and worker budgets must be positive")
    actual_cpus = min(cpus, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    active_workers = min(workers, actual_cpus, len(runs))
    threads = max(1, actual_cpus // active_workers)
    per_worker_memory = float(memory_gb) / active_workers
    host_root = root / "inference_shards/hosts" / host
    destination = host_root / "dispatcher_result.json"
    shard_destination = host_root / "result.json"
    session = uuid.uuid4().hex
    record = dict(contract="ffac-track1-incremental-host-execution-v1", host=host,
        machine=socket.gethostname(), candidate=str(root), dispatcher_pid=os.getpid(),
        host_plan_sha256=host_sha, runs=runs, completed=[], failed=[],
        status="running", started_utc=utc_now(), workers=plan["workers"],
        cpus=plan["cpus"], memory_gb=plan["memory_gb"], training_performed=False)
    shard_record = dict(contract=shard.CONTRACT, candidate=str(root), host=host,
        machine=socket.gethostname(), controller_pid=os.getpid(), session=session,
        model_identity=frozen["model_identity"], runs=runs, status="running",
        started_utc=utc_now(), backend="cpu", requested_cpus=cpus,
        cpu_limit=actual_cpus, workers=active_workers, threads_per_worker=threads,
        memory_gb=float(memory_gb), memory_gb_per_worker=per_worker_memory,
        optical_training_performed=False, regional_training_performed=False,
        root_campaign_published=False, completed=[], failed=[])

    def publish(stage):
        record.update(stage=stage, active_runs=list(active.values()), updated_utc=utc_now())
        _write_hot_status(shard_destination, shard_record)
        _write_hot_status(destination, record)

    # One controller owns this host for the full queue lifetime. The pool keeps
    # spare slots available for source seals that arrive during slow maps.
    with shard._host_lease(host_root / "controller.lock"):
        active = {}
        retry_after = {}
        publish("awaiting_individual_source_seals")
        try:
            with ProcessPoolExecutor(max_workers=active_workers,
                                     mp_context=mp.get_context("spawn")) as pool:
                while len(record["completed"]) != len(runs):
                    if sha256_file(host_path) != host_sha:
                        raise ValueError("Host partition changed while inference was running")
                    for future, run in list(active.items()):
                        if not future.done():
                            continue
                        del active[future]
                        try:
                            result = future.result()
                            if (result.get("status") != "succeeded" or result.get("run") != run
                                    or result.get("host") != host or result.get("session") != session
                                    or result.get("execution_backend") != "cpu"):
                                raise ValueError(f"{run}: worker returned an invalid CPU map receipt")
                            verify_incremental(root, [run])
                        except shard.RunBusyError:
                            # A worker may outlive a crashed controller. Its run
                            # lease prevents concurrent map writes on resume.
                            retry_after[run] = time.monotonic() + POLL_SECONDS
                        except Exception as error:
                            failure = dict(run=run, error=f"{type(error).__name__}: {error}")
                            record["failed"].append(failure)
                            shard_record["failed"].append(failure)
                        else:
                            record["completed"].append(run)
                            shard_record["completed"].append(result)
                    admitted = []
                    if not record["failed"]:
                        occupied = set(active.values()) | set(record["completed"])
                        for run in runs:
                            if len(active) == active_workers:
                                break
                            if (run in occupied or time.monotonic() < retry_after.get(run, 0)
                                    or not (root / "inference_run_bindings" / (run + ".json")).is_file()):
                                continue
                            binding = verify_incremental(root, [run])["source_bindings"][0]
                            source_sha = sha256_file(root / "review_source" / run / "postflight_manifest.json")
                            if source_sha != binding["review_source_manifest_sha256"]:
                                raise ValueError(f"{run}: source changed after its exact run seal")
                            shard._claim(root / "inference_shards/claims" / (run + ".json"),
                                         host=host, model_identity=frozen["model_identity"],
                                         source_sha=source_sha, session=session)
                            future = pool.submit(shard._map_run, str(root), run, threads,
                                                 per_worker_memory, host, session,
                                                 frozen["model_identity"])
                            active[future] = run
                            occupied.add(run)
                            admitted.append(run)
                    if admitted:
                        print(json.dumps(dict(event="mapping_ready_runs", host=host,
                                              runs=admitted)), flush=True)
                    stage = ("draining_after_failure" if record["failed"] else
                             "mapping" if active else "awaiting_individual_source_seals")
                    publish(stage)
                    if record["failed"] and not active:
                        raise RuntimeError(f"{host}: exact native inference failed: {record['failed']}")
                    if len(record["completed"]) == len(runs):
                        break
                    if active:
                        wait(active, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)
                    else:
                        time.sleep(POLL_SECONDS)
            verify_incremental(root, runs)
            shard_record.update(status="succeeded", completed_utc=utc_now(),
                                packages=[row["package"] for row in shard_record["completed"]])
            record.update(status="succeeded", completed_utc=utc_now())
            publish("complete")
        except BaseException as error:
            shard_record.update(status="failed", completed_utc=utc_now(),
                                packages=[row["package"] for row in shard_record["completed"]])
            record.update(status="failed", error=repr(error), completed_utc=utc_now())
            publish("failed")
            raise
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    print(json.dumps(run_host(args.root, args.host), indent=2), flush=True)


if __name__ == "__main__":
    main()
