"""SHA-bound, disjoint whole-run compositing with CPU/RAM-aware admission.

Workers only produce their assigned run receipts. A separate coordinator verifies
every source and seals the candidate after all worker claims have disappeared.
No model fitting, training-label loading, default change, or blind reveal occurs here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from fractions import Fraction
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import socket
import uuid

from darsia.fluidflower.tools.refit_ffac_g3_downstream import (
    atomic_json, canonical_sha256, read_json, sha256_file, utc_now,
)

CONTRACT = "ffac-l1-neighbourhood-distributed-postprocess-v1"
ASSIGNMENT = "distributed_l1_neighbourhood/assignment.json"
CLAIMS = "distributed_l1_neighbourhood/claims"
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_MIB = 1024 ** 2


def _runner():
    return importlib.import_module("darsia.fluidflower.tools.run_ffac_l1_neighbourhood_candidate")


def parse_owner_workers(value):
    result = {}
    for part in value.split(","):
        owner, separator, number = part.partition("=")
        if not separator or not _NAME.fullmatch(owner) or owner in result:
            raise ValueError("Distinct portable owner=worker-count entries are required")
        try:
            count = int(number)
        except ValueError as error:
            raise ValueError("Worker counts must be positive integers") from error
        if count < 1:
            raise ValueError("Worker counts must be positive integers")
        result[owner] = count
    if not result:
        raise ValueError("At least one computer is required")
    return result


def partition_runs(sources, owner_workers):
    """Balance frame work per admitted host capacity; never split a whole run."""
    owners = list(owner_workers)
    if (not owners or any(not _NAME.fullmatch(owner) for owner in owners)
            or any(type(n) is not int or n < 1 for n in owner_workers.values())):
        raise ValueError("Distinct portable owners with positive worker budgets are required")
    counts, keys = {}, set()
    for row in sources:
        run, index = row["run"], row["frame_index"]
        if (not re.fullmatch(r"ac\d+", run) or type(index) is not int or index < 0
                or (run, index) in keys):
            raise ValueError("Duplicate or invalid source-owned native frame")
        keys.add((run, index))
        counts[run] = counts.get(run, 0) + 1
    assignments, load = {owner: [] for owner in owners}, dict.fromkeys(owners, 0)
    for run in sorted(counts, key=lambda name: (-counts[name], name)):
        owner = min(owners, key=lambda name: (
            Fraction(load[name], owner_workers[name]), owners.index(name)))
        assignments[owner].append(run)
        load[owner] += counts[run]
    if any(not runs for runs in assignments.values()):
        raise ValueError("Every assigned computer needs at least one whole run")
    return {owner: dict(runs=sorted(assignments[owner]), frame_count=load[owner],
                        max_workers=owner_workers[owner]) for owner in owners}


def resource_admission(requested, run_count, memory_per_worker_mb,
                       memory_headroom_mb, *, logical_cpus, available_bytes):
    """Bound the pool by actual CPU/RAM and measured worker high-water RSS.

    The caller supplies a conservative measurement including transient arrays.
    This has no fixed hardware-specific worker ceiling and works on CPU laptops.
    """
    if any(type(n) is not int or n < 1 for n in (requested, run_count, logical_cpus)):
        raise ValueError("Requested workers, run count and logical CPU count must be positive")
    if (not math.isfinite(memory_per_worker_mb) or memory_per_worker_mb <= 0
            or not math.isfinite(memory_headroom_mb) or memory_headroom_mb < 0
            or available_bytes < 0):
        raise ValueError("A positive worker-memory measurement and nonnegative reserve are required")
    usable = available_bytes - math.ceil(memory_headroom_mb * _MIB)
    memory_workers = max(0, usable // math.ceil(memory_per_worker_mb * _MIB))
    admitted = min(requested, run_count, logical_cpus, memory_workers)
    if admitted < 1:
        raise RuntimeError("Insufficient available RAM for one worker plus the declared headroom")
    return dict(requested_workers=requested, assigned_runs=run_count,
                logical_cpus=logical_cpus, available_bytes=available_bytes,
                memory_per_worker_mb=memory_per_worker_mb,
                memory_headroom_mb=memory_headroom_mb,
                memory_capacity_workers=int(memory_workers), admitted_workers=int(admitted),
                threads_per_worker=1, cpu_only=True, below_normal_priority=True)


def candidate_state(candidate):
    """The stored-map runner owns the exact base/source/head validation."""
    return _runner().candidate_state(Path(candidate).resolve())


def prepare_assignment(candidate, owner_workers, *, memory_per_worker_mb,
                       memory_headroom_mb=2048, memory_measurement=None):
    root, plan, sources = candidate_state(candidate)
    runner = _runner()
    if len(sources) != 864 or len({row["run"] for row in sources}) != 43:
        raise ValueError("Distributed work requires the exact sealed 43/864 inventory")
    # Validate the memory policy without assuming either computer's RAM/CPU.
    resource_admission(1, 1, memory_per_worker_mb, memory_headroom_mb,
                       logical_cpus=1, available_bytes=2 ** 63 - 1)
    if (root / "job.pidlock").exists():
        raise RuntimeError("An existing full-run lease must finish before assigning workers")
    partitions = partition_runs(sources, owner_workers)
    binding = dict(candidate=str(root), candidate_plan=runner.artifact(root / runner.PLAN),
        pipeline_sha256=plan["pipeline_sha256"], scientific_binding=plan["binding"],
        orchestrator_sha256=sha256_file(__file__),
        source_inventory_sha256=canonical_sha256(sources), assignments=partitions,
        memory_policy=dict(memory_per_worker_mb=memory_per_worker_mb,
                           memory_headroom_mb=memory_headroom_mb,
                           measurement=memory_measurement),
        threads_per_worker=1, cpu_only=True, below_normal_priority=True,
        bottom_mass_allowed=False, mass_eligible=False,
        blind_hold_runs=sorted(runner.HIDDEN))
    path = root / ASSIGNMENT
    if path.exists():
        previous = read_json(path)
        if previous.get("contract") != CONTRACT or previous.get("binding") != binding:
            raise ValueError("Refusing to change an existing distributed assignment")
        return path
    atomic_json(path, dict(contract=CONTRACT, binding=binding,
        assignment_sha256=canonical_sha256(binding), created_utc=utc_now()))
    return path


def load_assignment(path, owner=None):
    value = read_json(path)
    binding = value["binding"]
    if (value.get("contract") != CONTRACT
            or value.get("assignment_sha256") != canonical_sha256(binding)):
        raise ValueError("Distributed assignment identity changed")
    if binding["orchestrator_sha256"] != sha256_file(__file__):
        raise ValueError("Distributed wrapper differs from the assigned copy")
    root, plan, sources = candidate_state(binding["candidate"])
    runner = _runner()
    if runner.bound_path(binding["candidate_plan"]) != (root / runner.PLAN).resolve():
        raise ValueError("Assignment belongs to another candidate plan")
    if (plan["pipeline_sha256"] != binding["pipeline_sha256"]
            or plan["binding"] != binding["scientific_binding"]
            or canonical_sha256(sources) != binding["source_inventory_sha256"]):
        raise ValueError("Distributed source/model/code binding changed")
    assigned = [run for row in binding["assignments"].values() for run in row["runs"]]
    if (len(assigned) != 43 or len(set(assigned)) != 43
            or set(assigned) != {row["run"] for row in sources}):
        raise ValueError("Runs must be assigned to exactly one computer")
    if owner is not None and owner not in binding["assignments"]:
        raise ValueError("Unassigned worker owner")
    return root, plan, sources, value


@contextmanager
def exclusive_claim(root, name, assignment_sha256, owner):
    """NAS no-clobber claim; never guess foreign PIDs or recover stale claims."""
    if not _NAME.fullmatch(name) or not _NAME.fullmatch(owner):
        raise ValueError("Invalid claim name")
    path = Path(root) / CLAIMS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(owner=owner, hostname=socket.gethostname(), pid=os.getpid(),
                 token=uuid.uuid4().hex, assignment_sha256=assignment_sha256,
                 acquired_utc=utc_now())
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"Run/owner already claimed; inspect the existing claim: {path}") from error
    try:
        yield
    finally:
        if path.exists() and read_json(path).get("token") == value["token"]:
            path.unlink()


def worker_initializer():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import psutil
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        process.nice(10)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def execute_run(assignment_path, owner, run):
    root, plan, sources, assignment = load_assignment(assignment_path, owner)
    if run not in assignment["binding"]["assignments"][owner]["runs"]:
        raise ValueError("Worker attempted another computer's run")
    if (root / "job.pidlock").exists():
        raise RuntimeError("A full-run candidate lease conflicts with distributed work")
    if (root / CLAIMS / "seal.json").exists():
        raise RuntimeError("Coordinator seal is active")
    with exclusive_claim(root, run, assignment["assignment_sha256"], owner):
        runner = _runner()
        # load_assignment already checked this exact source/model/code state.
        # Reuse its parsed inventory instead of reading the 267 MB plan again.
        result = runner.run_one_run(root, run, verified_state=(root, plan, sources))
        runner.validate_run_result(root, plan,
            [row for row in sources if row["run"] == run], result)
        return result


def work(assignment_path, owner, *, workers=None):
    root, _plan, _sources, assignment = load_assignment(assignment_path, owner)
    owner_row = assignment["binding"]["assignments"][owner]
    requested = owner_row["max_workers"] if workers is None else workers
    if type(requested) is not int or not 1 <= requested <= owner_row["max_workers"]:
        raise ValueError("Worker allocation exceeds the assigned CPU budget")
    import psutil
    process = psutil.Process()
    cpus = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    memory = assignment["binding"]["memory_policy"]
    admission = resource_admission(requested, len(owner_row["runs"]),
        memory["memory_per_worker_mb"], memory["memory_headroom_mb"],
        logical_cpus=cpus, available_bytes=psutil.virtual_memory().available)
    status_root = root / "distributed_l1_neighbourhood/owners"
    started = dict(contract=CONTRACT, owner=owner, hostname=socket.gethostname(),
        pid=os.getpid(), process_created=process.create_time(),
        assignment_sha256=assignment["assignment_sha256"], admission=admission,
        started_utc=utc_now(), complete=False)
    results = []
    try:
        with exclusive_claim(root, "owner_" + owner, assignment["assignment_sha256"], owner):
            if (root / "job.pidlock").exists() or (root / CLAIMS / "seal.json").exists():
                raise RuntimeError("Owner conflicts with a full run or coordinator seal")
            atomic_json(status_root / f"{owner}-started.json", started)
            with ProcessPoolExecutor(max_workers=admission["admitted_workers"],
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=worker_initializer) as pool:
                futures = {pool.submit(execute_run, str(assignment_path), owner, run): run
                           for run in owner_row["runs"]}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    # Blind-held run identities and areas are not printed.
                    print(json.dumps(dict(event="distributed_run_complete", owner=owner,
                                          completed_runs=len(results))), flush=True)
        runner = _runner()
        output = status_root / f"{owner}.json"
        atomic_json(output, dict(started, complete=True, finished_utc=utc_now(),
            results=[runner.artifact(root / runner.RUN_DIRECTORY / f"{row['run']}.json")
                     for row in sorted(results, key=lambda row: row["run"])]))
        return output
    except BaseException as error:
        atomic_json(status_root / f"{owner}-failed.json", dict(started,
            failed_utc=utc_now(), error_type=type(error).__name__, error=str(error)))
        raise


def seal(assignment_path):
    root, plan, sources, assignment = load_assignment(assignment_path)
    if any((root / CLAIMS).glob("*.json")):
        raise RuntimeError("Cannot seal while any owner/run claim is active")
    with exclusive_claim(root, "seal", assignment["assignment_sha256"], "coordinator"):
        if any(path.name != "seal.json" for path in (root / CLAIMS).glob("*.json")):
            raise RuntimeError("A worker claimed work while the coordinator acquired its seal")
        runner, rows = _runner(), []
        for run in sorted({row["run"] for row in sources}):
            row = read_json(root / runner.RUN_DIRECTORY / f"{run}.json")
            runner.validate_run_result(root, plan,
                [source for source in sources if source["run"] == run], row)
            rows.append(row)
        if (root / runner.SEAL).exists():
            saved = read_json(root / runner.SEAL)
            if (saved.get("complete") is not True
                    or saved.get("pipeline_sha256") != plan["pipeline_sha256"]
                    or saved.get("run_count") != 43 or saved.get("frame_count") != 864):
                raise ValueError("Existing candidate seal differs from the assignment")
            return saved
        return runner.seal_candidate(root, plan, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("assign")
    create.add_argument("--candidate", type=Path, required=True)
    create.add_argument("--owner-workers", required=True,
                        help="Computer budgets, e.g. local=32,remote=24")
    create.add_argument("--memory-per-worker-mb", required=True, type=float)
    create.add_argument("--memory-headroom-mb", default=2048, type=float)
    create.add_argument("--memory-measurement", type=Path)
    worker = commands.add_parser("worker")
    worker.add_argument("--assignment", type=Path, required=True)
    worker.add_argument("--owner", required=True)
    worker.add_argument("--workers", type=int)
    final = commands.add_parser("seal")
    final.add_argument("--assignment", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "assign":
        measurement = None if args.memory_measurement is None else _runner().artifact(args.memory_measurement)
        print(prepare_assignment(args.candidate, parse_owner_workers(args.owner_workers),
            memory_per_worker_mb=args.memory_per_worker_mb,
            memory_headroom_mb=args.memory_headroom_mb, memory_measurement=measurement))
    elif args.command == "worker":
        print(work(args.assignment, args.owner, workers=args.workers))
    else:
        seal(args.assignment)


if __name__ == "__main__":
    main()
