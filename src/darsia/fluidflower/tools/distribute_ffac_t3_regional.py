"""Transport-only sharding of an existing frozen regional extraction.

Invoke this file directly with ``python -I`` and --frozen-code-root. Scientific
modules are imported from that original snapshot, never from this extension.
Prepare requires an explicit quiescence declaration. Host workspaces mirror
the canonical extraction/pixel paths; receipts are transported byte for byte.
Workers extract sources only. The original coordinator CLI performs the fit.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys

CONTRACT = "ffac-g3-t3-frozen-regional-source-distribution-v1"


def parse_owner_workers(value):
    result = {}
    for part in value.split(","):
        host, sep, count = part.partition("=")
        if not sep or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", host) is None or host in result:
            raise ValueError("Distinct portable host=worker-count entries are required")
        result[host] = int(count)
        if result[host] < 1:
            raise ValueError("Worker budgets must be positive")
    return result


def validate_memory_policy(memory_per_worker_mb, memory_headroom_mb):
    import math
    if (not math.isfinite(memory_per_worker_mb) or memory_per_worker_mb <= 0
            or not math.isfinite(memory_headroom_mb) or memory_headroom_mb < 0):
        raise ValueError("A positive worker-memory bound and nonnegative reserve are required")


def worker_admission(requested, source_count, memory_per_worker_mb, memory_headroom_mb):
    import psutil
    from darsia.fluidflower.tools.distribute_ffac_t2_l1zone_candidate import resource_admission
    process = psutil.Process()
    cpus = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    return resource_admission(requested, source_count, memory_per_worker_mb, memory_headroom_mb,
                              logical_cpus=cpus, available_bytes=psutil.virtual_memory().available)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != safe(value):
            raise ValueError("Immutable distribution receipt changed: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if digest_file(source) != expected:
        raise ValueError("Transport source changed: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest_file(destination) != expected:
            raise ValueError("Transport destination already contains different bytes: " + str(destination))
        return
    temporary = destination.with_suffix(destination.suffix + ".transport-partial")
    shutil.copyfile(source, temporary)
    if digest_file(temporary) != expected:
        raise ValueError("Transport copy lost its SHA binding")
    temporary.replace(destination)


def frozen_imports(code_root):
    root = Path(code_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from darsia.fluidflower import phase_gas_t3_regional_refit as worker
    from darsia.fluidflower import phase_gas_t3_training as training
    from darsia.fluidflower.phase_training.trainer import RunSource
    from darsia.fluidflower.phase_training.logistic import ReviewedRegion
    for module in (worker, training):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("Scientific modules were imported before the original snapshot")
    return worker, training, RunSource, ReviewedRegion


def serialized_task(source, group, offset, binding, extraction_root, training):
    local = dict(binding, native_manifest_path=str(source.asset_manifest_path),
        native_manifest_sha256=digest_file(source.asset_manifest_path),
        source_region_ids_sha256=training.digest([r.region_id for r in group]), owner_start=offset)
    name = f"{offset:04d}-{source.run}-{training.digest(local)[:16]}"
    return dict(task_id=name, source=safe(asdict(source)), regions=[asdict(r) for r in group],
        owner_start=offset, binding=local, canonical_folder=str(Path(extraction_root) / "source_samples" / name),
        source_frames=len({r.frame_index for r in group}), host=None, imported_files=[])


def receipt_files(folder, binding, *, require_source=False):
    """Only complete SHA-bound commits, excluding interrupted temporary files."""
    folder = Path(folder)
    files = {}
    receipts = sorted(folder.glob("frame_*.json"))
    source_path = folder / "source_receipt.json"
    if source_path.is_file():
        receipts.append(source_path)
    elif require_source:
        raise ValueError("Source transport requires its final source receipt")
    for path in receipts:
        value = read(path)
        expected = binding if path == source_path else dict(binding,
            frame_index=value["binding"]["frame_index"], region_ids=value["binding"]["region_ids"])
        if value["binding"] != expected:
            raise ValueError("Source/frame transport changed its scientific binding")
        sample = value["samples"]
        sample_path = Path(sample["path"])
        # Receipt paths remain literal canonical paths on both hosts.
        if sample_path.parent.resolve() != folder.resolve() or digest_file(sample_path) != sample["sha256"]:
            raise ValueError("Source/frame transport sample is outside its canonical folder or changed")
        files[sample_path.name] = sample["sha256"]
        files[path.name] = digest_file(path)
        if path == source_path:
            for frame in value["frames"]:
                frame_path = Path(frame["path"])
                if frame_path.parent.resolve() != folder.resolve() or digest_file(frame_path) != frame["sha256"]:
                    raise ValueError("Completed source lost a frame receipt")
    return [dict(name=name, sha256=sha) for name, sha in sorted(files.items())]


def assign_sources(tasks, hosts, owner_workers):
    if (len(hosts) != 2 or hosts[0] == hosts[1]
            or any(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", host) is None for host in hosts)):
        raise ValueError("Two distinct source-owner hosts are required")
    if set(owner_workers) != set(hosts) or any(type(n) is not int or n < 1 for n in owner_workers.values()):
        raise ValueError("Every exclusive host needs a positive worker budget")
    load = {host: 0 for host in hosts}
    pending = [t for t in tasks if not t.get("source_complete")]
    for task in sorted(pending, key=lambda t: (-t["remaining_frames"], t["task_id"])):
        host = min(hosts, key=lambda h: (Fraction(load[h], owner_workers[h]), hosts.index(h)))
        task["host"] = host
        load[host] += task["remaining_frames"]
    return load


def inventory(job_root, destination):
    """Metadata-only task staging; allowed while original extraction is active."""
    job, destination = Path(job_root), Path(destination)
    config = read(job / "config.json")
    stage = next(row for row in config["stages"] if row["name"] == "regional")
    args = dict(zip(stage["args"][::2], stage["args"][1::2]))
    extraction = Path(args["--output"]) / "extraction"
    binding_path = extraction / "extraction_binding.json"
    binding = read(binding_path)
    worker, training, _source_cls, _region_cls = frozen_imports(job / "code")
    if digest_file(worker.__file__) != binding["implementation"]["extension"]:
        raise ValueError("Distribution changed the original numerical T3 worker")
    staged = read(extraction / "source_task_plan.json")
    if (staged["binding"] != binding or staged.get("extraction_performed") is not False
            or staged.get("fitting_performed") is not False):
        raise ValueError("Distribution requires the exact metadata-only T3 extraction plan")
    tasks = []
    for row in staged["tasks"]:
        name = Path(row["canonical_folder"]).name
        if name != f"{row['owner_start']:04d}-{row['source']['run']}-{training.digest(row['binding'])[:16]}":
            raise ValueError("Staged source task differs from numerical cache identity")
        tasks.append(dict(row, task_id=name, source_frames=len({r["frame_index"] for r in row["regions"]}),
                          host=None, imported_files=[]))
    result = dict(contract=CONTRACT, inventory_only=True, original_job_root=str(job.resolve()),
        frozen_code_root=str((job / "code").resolve()), extraction_root=str(extraction.resolve()),
        original_binding=binding, original_extraction_binding_sha256=digest_file(binding_path),
        original_code_manifest_sha256=digest_file(job / "code_manifest.json"), tasks=tasks,
        total_source_groups=len(tasks), total_source_frames=sum(t["source_frames"] for t in tasks),
        cpu_only=True, extraction_performed=False, fitting_performed=False)
    write(destination, result)
    return result


def prepare(job_root, destination, hosts, *, source_job_stopped, task_inventory=None,
            owner_workers=None, memory_per_worker_mb=6144, memory_headroom_mb=2048):
    if not source_job_stopped:
        raise ValueError("Exclusive ownership requires the original source job to be stopped first")
    job, destination = Path(job_root), Path(destination)
    task_inventory = Path(task_inventory) if task_inventory else destination.parent / "task_inventory.json"
    staged = read(task_inventory) if task_inventory.exists() else inventory(job, task_inventory)
    extraction = Path(staged["extraction_root"])
    binding_path = extraction / "extraction_binding.json"
    binding = read(binding_path)
    if (staged.get("contract") != CONTRACT or staged.get("inventory_only") is not True
            or staged["original_job_root"] != str(job.resolve()) or staged["original_binding"] != binding
            or staged["original_extraction_binding_sha256"] != digest_file(binding_path)
            or staged["original_code_manifest_sha256"] != digest_file(job / "code_manifest.json")):
        raise ValueError("Pre-staged source tasks differ from the original frozen job")
    tasks = staged["tasks"]
    for task in tasks:
        folder = Path(task["canonical_folder"])
        task["imported_files"] = receipt_files(folder, task["binding"])
        task["source_complete"] = (folder / "source_receipt.json").is_file()
        task["completed_frames"] = sum(item["name"].startswith("frame_") and item["name"].endswith(".json")
                                        for item in task["imported_files"])
        task["remaining_frames"] = task["source_frames"] - task["completed_frames"]
        if task["remaining_frames"] < 0:
            raise ValueError("Imported frame inventory exceeds exact source frames")
    owner_workers = owner_workers or {hosts[0]: 32, hosts[1]: 24}
    validate_memory_policy(memory_per_worker_mb, memory_headroom_mb)
    load = assign_sources(tasks, hosts, owner_workers)
    # Source imports are explicit byte-identical copies. The source tree is
    # never renamed, removed or overwritten by this transport extension.
    for task in tasks:
        for item in task["imported_files"]:
            copy_verified(Path(task["canonical_folder"]) / item["name"],
                destination.parent / "imports" / task["task_id"] / item["name"], item["sha256"])
    plan = dict(contract=CONTRACT, created_utc=datetime.now(timezone.utc).isoformat(),
        original_job_root=str(job.resolve()), frozen_code_root=str((job / "code").resolve()),
        extraction_root=str(extraction.resolve()), original_binding=binding,
        original_extraction_binding_sha256=digest_file(binding_path),
        original_code_manifest_sha256=digest_file(job / "code_manifest.json"),
        task_inventory_sha256=digest_file(task_inventory),
        canonical_paths_must_be_mirrored=True, source_job_stopped_before_assignment=True,
        cpu_only=True, threads_per_worker=1, owner_workers=owner_workers,
        memory_policy=dict(memory_per_worker_mb=memory_per_worker_mb, memory_headroom_mb=memory_headroom_mb),
        hosts=hosts, remaining_frame_load=load, tasks=tasks,
        total_source_groups=len(tasks), imported_completed_sources=sum(t["source_complete"] for t in tasks),
        imported_frame_commits=sum(t["completed_frames"] for t in tasks),
        coordinator_fit="original frozen regional CLI only after all source receipts imported")
    write(destination, plan)
    return plan


def restore_imports(plan, plan_path, host):
    for task in plan["tasks"]:
        if task["host"] != host:
            continue
        for item in task["imported_files"]:
            copy_verified(Path(plan_path).parent / "imports" / task["task_id"] / item["name"],
                          Path(task["canonical_folder"]) / item["name"], item["sha256"])


def export_source(task, export_root):
    files = receipt_files(task["canonical_folder"], task["binding"], require_source=True)
    destination = Path(export_root) / "tasks" / task["task_id"]
    for item in files:
        copy_verified(Path(task["canonical_folder"]) / item["name"], destination / item["name"], item["sha256"])
    receipt = dict(contract=CONTRACT, task_id=task["task_id"], host=task["host"],
        scientific_binding_sha256=digest(task["binding"]), files=files,
        canonical_folder=task["canonical_folder"], cpu_only=True, threads=1,
        source_receipt_sha256=next(x["sha256"] for x in files if x["name"] == "source_receipt.json"))
    write(destination / "transport_receipt.json", receipt)
    return receipt


def execute(task, frozen_code_root):
    worker, _training, source_cls, region_cls = frozen_imports(frozen_code_root)
    source = dict(task["source"])
    for key in ("review_root", "asset_manifest_path", "decisions_path", "combined_manifest_path"):
        source[key] = Path(source[key])
    source["independent_reference_frames"] = tuple(source["independent_reference_frames"])
    return worker._source_worker((source_cls(**source), tuple(region_cls(**r) for r in task["regions"]),
        task["owner_start"], task["binding"], task["canonical_folder"], 1))


def verify_worker_snapshot(plan):
    """Check the mirrored original code and heads before claiming any source."""
    code = Path(plan["frozen_code_root"]).resolve()
    manifest_path = Path(plan["original_job_root"]) / "code_manifest.json"
    if digest_file(manifest_path) != plan["original_code_manifest_sha256"]:
        raise ValueError("Mirrored original code manifest changed")
    manifest = read(manifest_path)
    for relative, expected in manifest["files_sha256"].items():
        path = (code / relative).resolve()
        if not path.is_relative_to(code) or digest_file(path) != expected:
            raise ValueError("Mirrored original snapshot file changed: " + relative)
    binding = plan["original_binding"]
    heads = [(Path(binding["g3_model_path"]), binding["g3_model_sha256"]),
        (Path(binding["frozen_candidate"]) / "regional_model.json", binding["frozen_regional_sha256"]),
        (Path(binding["legacy_context_candidate"]) / "regional_model.json", binding["legacy_regional_sha256"]),
        (Path(binding["frozen_candidate"]) / "models/generic/co2_presence_signal.json", binding["co2_head_sha256"])]
    if any(digest_file(path) != expected for path, expected in heads):
        raise ValueError("Mirrored regional/pixel/CO2 scientific head changed")
    return dict(original_code_manifest_sha256=plan["original_code_manifest_sha256"],
                 snapshot_files_verified=len(manifest["files_sha256"]), scientific_heads_verified=len(heads))


def worker(plan_path, host, export_root, *, workers=None):
    plan = read(plan_path)
    if (plan.get("contract") != CONTRACT or host not in plan["hosts"]
            or (workers is not None and (type(workers) is not int or workers < 1))
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", host) is None):
        raise ValueError("Worker host/allocation differs from its exclusive source plan")
    if read(Path(plan["extraction_root"]) / "extraction_binding.json") != plan["original_binding"]:
        raise ValueError("Worker lost the original extraction binding")
    snapshot = verify_worker_snapshot(plan)
    frozen_imports(plan["frozen_code_root"])
    tasks = [t for t in plan["tasks"] if t["host"] == host]
    budget = plan["owner_workers"][host]
    requested = budget if workers is None else min(workers, budget)
    admission = worker_admission(requested, max(1, len(tasks)), **plan["memory_policy"])
    # One canonical host lock even when a second invocation chooses another
    # export directory. A failed run leaves the lock for explicit operator
    # recovery after confirming its process stopped; no lease expiry overlap.
    lock = Path(plan["extraction_root"]) / "distribution_owners" / ("owner-" + host)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir()  # Refuse concurrent host orchestrators; no expired-lease guess.
    write(lock / "owner.json", dict(pid=os.getpid(), plan_sha256=digest_file(plan_path), host=host))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    restore_imports(plan, plan_path, host)
    tasks = [t for t in plan["tasks"] if t["host"] == host]
    completed = []
    with ProcessPoolExecutor(max_workers=admission["admitted_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(execute, t, plan["frozen_code_root"]): t for t in tasks}
        for future in as_completed(futures):
            future.result()
            completed.append(export_source(futures[future], export_root))
    result = dict(contract=CONTRACT, host=host, plan_sha256=digest_file(plan_path),
                  source_receipts=completed, cpu_only=True, fitting_performed=False,
                  original_snapshot_verification=snapshot, admission=admission)
    write(Path(export_root) / (host + "-complete.json"), result)
    (lock / "owner.json").unlink()
    lock.rmdir()
    return result


def merge(plan_path, export_roots):
    plan = read(plan_path)
    if plan.get("contract") != CONTRACT:
        raise ValueError("Unknown source distribution plan")
    imported = []
    for task in plan["tasks"]:
        if task["source_complete"]:
            receipt_files(task["canonical_folder"], task["binding"], require_source=True)
            continue
        paths = [Path(root) / "tasks" / task["task_id"] / "transport_receipt.json" for root in export_roots]
        found = [p for p in paths if p.is_file()]
        if len(found) != 1:
            raise ValueError("Every pending source needs exactly one declared owner export")
        path = found[0]
        value = read(path)
        if (value.get("contract") != CONTRACT or value["host"] != task["host"]
                or value["task_id"] != task["task_id"] or value["canonical_folder"] != task["canonical_folder"]
                or value["scientific_binding_sha256"] != digest(task["binding"])):
            raise ValueError("Source export lost exclusive ownership or scientific binding")
        for item in value["files"]:
            if Path(item["name"]).name != item["name"] or item["name"] in (".", ".."):
                raise ValueError("Source export path escapes its task folder")
            copy_verified(path.parent / item["name"], Path(task["canonical_folder"]) / item["name"], item["sha256"])
        receipt_files(task["canonical_folder"], task["binding"], require_source=True)
        imported.append(dict(task_id=task["task_id"], owner=task["host"],
                             transport_receipt_sha256=digest_file(path), source_receipt_sha256=value["source_receipt_sha256"]))
    result = dict(contract=CONTRACT, plan_sha256=digest_file(plan_path),
        imported_sources=imported, all_original_sources_complete=True,
        original_scientific_binding_unchanged=True, fitting_performed=False)
    write(Path(plan_path).parent / "merge_receipt.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("inventory")
    stage.add_argument("--job-root", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--job-root", type=Path, required=True)
    prep.add_argument("--plan", type=Path, required=True)
    prep.add_argument("--hosts", nargs=2, required=True)
    prep.add_argument("--source-job-stopped", action="store_true")
    prep.add_argument("--task-inventory", type=Path)
    prep.add_argument("--owner-workers", default="Moderskipet=32,Olav=24")
    prep.add_argument("--memory-per-worker-mb", type=float, default=6144)
    prep.add_argument("--memory-headroom-mb", type=float, default=2048)
    work = sub.add_parser("worker")
    work.add_argument("--plan", type=Path, required=True)
    work.add_argument("--host", required=True)
    work.add_argument("--export-root", type=Path, required=True)
    work.add_argument("--workers", type=int)
    collect = sub.add_parser("merge")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--export-roots", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        result = inventory(args.job_root, args.output)
    elif args.command == "prepare":
        result = prepare(args.job_root, args.plan, args.hosts, source_job_stopped=args.source_job_stopped,
                         task_inventory=args.task_inventory, owner_workers=parse_owner_workers(args.owner_workers),
                         memory_per_worker_mb=args.memory_per_worker_mb, memory_headroom_mb=args.memory_headroom_mb)
    elif args.command == "worker":
        try:
            result = worker(args.plan, args.host, args.export_root, workers=args.workers)
        except Exception as error:
            if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.host) is not None:
                write(args.export_root / (args.host + "-failed.json"), dict(contract=CONTRACT,
                    host=args.host, plan_sha256=digest_file(args.plan), pid=os.getpid(),
                    cpu_only=True, fitting_performed=False, source_artifacts_preserved=True,
                    automatic_owner_reclaim=False, error_type=type(error).__name__, error=str(error)))
            raise
    else:
        result = merge(args.plan, args.export_roots)
    print(json.dumps({k: v for k, v in result.items() if k not in ("tasks", "original_binding", "source_receipts", "imported_sources")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
