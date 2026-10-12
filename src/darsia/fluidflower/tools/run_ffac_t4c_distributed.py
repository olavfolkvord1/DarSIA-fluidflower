"""Cooperating CPU hosts produce one source-bound T4c bundle in small chunks.

Each new-lamp photograph is prepared once and reused by every placement. NAS
claims use exclusive creation; failed or abandoned claims remain for inspection
and are never recovered by guessing a remote PID. Sealing is a separate step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import socket
import time
import traceback
import uuid

from .distribute_ffac_t4b_floor_candidate import resource_admission, worker_initializer
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, read_json, sha256_file, utc_now

CONTRACT = "ffac-t4c-distributed-bundle-v1"
PLACEMENTS = ("P1", "P2", "P3")
DIRECTORY = "distributed_t4c"
CHUNK_SIZE = 4
NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_BUNDLE_CACHE = {}


def _runner():
    return importlib.import_module("darsia.fluidflower.tools.run_ffac_t4c_floor_candidate")


def chunk_sources(sources, run_to_epoch):
    grouped, seen = {}, set()
    for source in sources:
        run, index = source["run"], source["frame_index"]
        if not re.fullmatch(r"ac\d+", run) or type(index) is not int or index < 0 or (run, index) in seen:
            raise ValueError("T4c chunks require unique source-owned frame keys")
        seen.add((run, index)); grouped.setdefault(run, []).append(source)
    if set(grouped) != set(run_to_epoch):
        raise ValueError("T4c chunk sources differ from the project light epochs")
    result = []
    for run, frames in sorted(grouped.items()):
        epoch = run_to_epoch[run]
        if epoch not in ("rig_light", "new_lamps"):
            raise ValueError("T4c requires declared project light epochs")
        frames.sort(key=lambda source: source["frame_index"])
        count = CHUNK_SIZE if epoch == "new_lamps" else len(frames)
        for offset in range(0, len(frames), count):
            group = frames[offset:offset+count]
            result.append(dict(id=f"{run}-{group[0]['frame_index']:03d}-{group[-1]['frame_index']:03d}",
                run=run, light_epoch_id=epoch, sources=group,
                frame_indices=[source["frame_index"] for source in group],
                source_sha256=canonical_sha256(group)))
    # Start expensive work first; the short rig-light copies fill remaining gaps.
    return sorted(result, key=lambda chunk: (chunk["light_epoch_id"] != "new_lamps", chunk["run"], chunk["frame_indices"][0]))


def prepare_bundle(candidates, path, *, memory_per_worker_mb=10240, memory_headroom_mb=2048):
    runner = _runner()
    if not candidates or set(candidates) - set(PLACEMENTS):
        raise ValueError("A bundle needs calibrated P1/P2/P3 candidates")
    state, descriptors = {}, {}
    for name in PLACEMENTS:
        if name not in candidates: continue
        root, plan, sources = runner.candidate_state(candidates[name])
        if plan["placement"] != name:
            raise ValueError("T4c bundle candidate has another placement")
        state[name] = (root, plan, sources)
        descriptors[name] = dict(root=str(root), plan=runner.artifact(root/runner.PLAN), pipeline_sha256=plan["pipeline_sha256"])
    first = next(iter(state.values()))
    config = first[1]["evaluation_config"]
    for root, plan, sources in state.values():
        if sources != first[2] or plan["evaluation_config"] != config or plan["run_to_epoch"] != first[1]["run_to_epoch"]:
            raise ValueError("T4c bundle placements do not share the exact source/config/epoch inventory")
        if (root/"job.pidlock").exists():
            raise RuntimeError("A candidate full-run lease conflicts with distributed production")
    resource_admission(1, 1, memory_per_worker_mb, memory_headroom_mb, logical_cpus=1, available_bytes=2**63-1)
    binding = dict(candidates=descriptors, evaluation_config_sha256=canonical_sha256(config),
        source_inventory_sha256=canonical_sha256(first[2]), run_to_epoch=first[1]["run_to_epoch"],
        chunks=chunk_sources(first[2], first[1]["run_to_epoch"]), chunk_size_new_lamps=CHUNK_SIZE,
        wrapper_sha256=sha256_file(__file__),
        memory_policy=dict(memory_per_worker_mb=memory_per_worker_mb, memory_headroom_mb=memory_headroom_mb),
        cpu_only=True, threads_per_worker=1, prepare_once_per_frame=True,
        mass_eligible=False, bottom_mass_allowed=False, training=False, annotations_read=False)
    path = Path(path).resolve()
    receipt = dict(contract=CONTRACT, binding=binding, bundle_sha256=canonical_sha256(binding), created_utc=utc_now())
    if path.exists():
        previous = read_json(path)
        if previous.get("contract") != CONTRACT or previous.get("binding") != binding:
            raise ValueError("Refusing to change the immutable distributed bundle")
        return path
    atomic_json(path, receipt)
    return path


def _load_bundle_uncached(path):
    runner = _runner()
    path = Path(path).resolve(); value = read_json(path); binding = value["binding"]
    if (value.get("contract") != CONTRACT or value.get("bundle_sha256") != canonical_sha256(binding)
            or binding.get("wrapper_sha256") != sha256_file(__file__)
            or binding.get("cpu_only") is not True or binding.get("threads_per_worker") != 1):
        raise ValueError("T4c bundle/wrapper identity changed")
    state = {}
    for name in PLACEMENTS:
        if name not in binding["candidates"]: continue
        descriptor = binding["candidates"][name]
        root, plan, sources = runner.candidate_state(descriptor["root"])
        if (plan.get("placement") != name or runner.bound_path(descriptor["plan"]) != (root/runner.PLAN).resolve()
                or plan["pipeline_sha256"] != descriptor["pipeline_sha256"]
                or canonical_sha256(plan["evaluation_config"]) != binding["evaluation_config_sha256"]
                or canonical_sha256(sources) != binding["source_inventory_sha256"]
                or plan["run_to_epoch"] != binding["run_to_epoch"]):
            raise ValueError("T4c bundle differs from a saved placement's source/config/plan")
        state[name] = (root, plan, sources)
    if not state: raise ValueError("T4c bundle has no placement candidates")
    first = next(iter(state.values()))
    if binding["chunks"] != chunk_sources(first[2], binding["run_to_epoch"]):
        raise ValueError("T4c bundle deterministic frame chunk inventory changed")
    return path, value, state


def _verified_file_signature(path, value, state):
    paths = {Path(path).resolve(), Path(__file__).resolve()}
    def descriptors(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                paths.add(Path(value["path"]).resolve())
            for child in value.values(): descriptors(child)
        elif isinstance(value, list):
            for child in value: descriptors(child)
    descriptors(value["binding"]["candidates"])
    first = next(iter(state.values()))
    config_binding = first[1]["evaluation_config"].get("binding", {})
    descriptors(config_binding)
    for module_name in config_binding.get("code", {}):
        paths.add(Path(importlib.import_module(module_name).__file__).resolve())
    for root, _, _ in state.values():
        for name in ("profile_template.json", "summary.json"):
            if (root/name).exists(): paths.add((root/name).resolve())
    return tuple((str(file), file.stat().st_size, file.stat().st_mtime_ns)
        for file in sorted(paths, key=str))


def load_bundle(path):
    """Cache verified immutable metadata within a worker, watching bound files.

    Initial validation retains the scientific runner's complete SHA checks.
    The saved snapshot is immutable; any bound input, plan or code file signature
    change forces those checks again before another frame is produced. This
    avoids repeatedly hashing the 577 MB frozen parent plan for every chunk.
    """
    key = str(Path(path).resolve())
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        signature, result = cached
        if _verified_file_signature(*result) == signature:
            return result
    result = _load_bundle_uncached(path)
    _BUNDLE_CACHE[key] = (_verified_file_signature(*result), result)
    return result


def _exclusive_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def _chunk_paths(bundle_path, chunk_id):
    if not NAME.fullmatch(chunk_id): raise ValueError("Invalid T4c chunk identifier")
    root = Path(bundle_path).parent / DIRECTORY
    return root/"claims"/(chunk_id+".json"), root/"results"/(chunk_id+".json"), root/"logs"/(chunk_id+".jsonl")


def _read_claim(path):
    # Exclusive creation owns the claim before the tiny JSON write completes.
    # Wait only for that metadata publication, never for a failed task retry.
    for attempt in range(20):
        try: return read_json(path)
        except json.JSONDecodeError:
            if attempt == 19: raise
            time.sleep(.05)


def _assert_result_claim(claim_path, bundle):
    if claim_path.exists():
        claim = _read_claim(claim_path)
        if claim.get("bundle_sha256") != bundle["bundle_sha256"]:
            raise ValueError("T4c result claim belongs to another bundle")
        if claim.get("status") == "failed":
            raise RuntimeError(f"T4c preserved failed chunk claim requires inspection: {claim_path}")


def _update_owned_claim(path, value):
    """Publish only this worker's mutable lease status, retaining its owner token."""
    if _read_claim(path).get("token") != value["token"]:
        raise RuntimeError("T4c chunk lease changed owner during production")
    temporary = path.with_name(path.name+".status-"+uuid.uuid4().hex)
    _exclusive_json(temporary, value)
    os.replace(temporary, path)


def verify_chunk_result(bundle, chunk, result):
    runner = _runner()
    if (result.get("contract") != CONTRACT or result.get("complete") is not True
            or result.get("bundle_sha256") != bundle["bundle_sha256"] or result.get("chunk") != chunk
            or set(result.get("placements", {})) != set(bundle["binding"]["candidates"])):
        raise ValueError("T4c completed chunk receipt changed")
    expected = {(source["run"], source["frame_index"]): source for source in chunk["sources"]}
    for name, frames in result["placements"].items():
        seen = set()
        for frame in frames:
            row = read_json(runner.bound_path(frame["receipt"]))
            key = (row["run"], row["frame_index"])
            if (key in seen or key not in expected or row.get("complete") is not True
                    or row.get("placement") != name or row["binding"]["source"] != expected[key]
                    or row["binding"]["pipeline_sha256"] != bundle["binding"]["candidates"][name]["pipeline_sha256"]):
                raise ValueError("T4c chunk result does not bind every placement's exact frame")
            for asset in (row["labels_map"], row["floor_contact_fill"]): runner.bound_path(asset)
            seen.add(key)
        if seen != set(expected): raise ValueError("T4c chunk receipt misses a placement frame")
    return result


def execute_chunk(bundle_path, host, chunk_id):
    if not NAME.fullmatch(host): raise ValueError("Host needs a portable distinct name")
    bundle_path, bundle, state = load_bundle(bundle_path)
    chunk = next((item for item in bundle["binding"]["chunks"] if item["id"] == chunk_id), None)
    if chunk is None: raise ValueError("Unassigned T4c chunk")
    claim_path, result_path, log_path = _chunk_paths(bundle_path, chunk_id)
    if result_path.exists():
        _assert_result_claim(claim_path, bundle)
        verify_chunk_result(bundle, chunk, read_json(result_path))
        return dict(chunk_id=chunk_id, status="complete", resumed=True, receipt=_runner().artifact(result_path))
    claim = dict(contract=CONTRACT, bundle_sha256=bundle["bundle_sha256"], chunk_id=chunk_id,
        host=host, hostname=socket.gethostname(), pid=os.getpid(), token=uuid.uuid4().hex,
        status="running", acquired_utc=utc_now())
    try:
        _exclusive_json(claim_path, claim)
    except FileExistsError:
        existing = _read_claim(claim_path)
        if existing.get("bundle_sha256") != bundle["bundle_sha256"] or existing.get("chunk_id") != chunk_id:
            raise ValueError("T4c chunk claim belongs to another immutable bundle")
        if existing.get("status") == "failed":
            raise RuntimeError(f"T4c chunk previously failed; inspect preserved claim/log: {claim_path}")
        if result_path.exists():
            verify_chunk_result(bundle, chunk, read_json(result_path))
            return dict(chunk_id=chunk_id, status="complete", resumed=True, receipt=_runner().artifact(result_path))
        return dict(chunk_id=chunk_id, status="claimed", owner=existing.get("host"), claim=str(claim_path))
    runner = _runner()
    started, cpu = time.perf_counter(), time.process_time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            def event(stage, **fields):
                value = dict(utc=utc_now(), stage=stage, host=host, chunk_id=chunk_id, **fields)
                log.write(json.dumps(value, allow_nan=False)+"\n"); log.flush()
            event("start")
            outputs, cache = {name: [] for name in state}, {}
            first = next(iter(state.values()))
            config = first[1]["evaluation_config"]
            prepared_count = 0
            for source in chunk["sources"]:
                prepared = None
                if chunk["light_epoch_id"] == "new_lamps":
                    prepared = runner.prepare_evaluation_frame(config, source["run"], source["frame_index"], baseline_cache=cache)
                    prepared_count += 1
                for name, (root, plan, _) in state.items():
                    if (root/"job.pidlock").exists(): raise RuntimeError("A full-run candidate lease conflicts with chunk production")
                    outputs[name].append(runner.infer_frame(root, plan, source, frame_state=prepared))
                event("frame_complete", run=source["run"], frame_index=source["frame_index"], placements=list(state))
            result = dict(contract=CONTRACT, complete=True, bundle_sha256=bundle["bundle_sha256"],
                chunk=chunk, host=host, claim_token=claim["token"], placements=outputs,
                raw_preparation_count=prepared_count, cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-started,
                cpu_only=True, threads_per_worker=1, created_utc=utc_now())
            verify_chunk_result(bundle, chunk, result)
            _exclusive_json(result_path, result)
            claim.update(status="complete", completed_utc=utc_now(), result=runner.artifact(result_path))
            _update_owned_claim(claim_path, claim)
            event("complete", raw_preparation_count=prepared_count, cpu_s=result["cpu_s"], wall_s=result["wall_s"])
    except BaseException as error:
        claim.update(status="failed", failed_utc=utc_now(), error=str(error), traceback=traceback.format_exc())
        _update_owned_claim(claim_path, claim)
        raise
    return dict(chunk_id=chunk_id, status="complete", resumed=False, receipt=runner.artifact(result_path))


def completed_chunks(bundle_path, bundle):
    complete, pending = [], []
    for chunk in bundle["binding"]["chunks"]:
        claim, path, _ = _chunk_paths(bundle_path, chunk["id"])
        if path.exists():
            _assert_result_claim(claim, bundle)
            verify_chunk_result(bundle, chunk, read_json(path)); complete.append(chunk["id"])
        else: pending.append(chunk["id"])
    return complete, pending


def run_host(bundle_path, workers, host, chunks=None):
    import psutil
    if not NAME.fullmatch(host): raise ValueError("Host needs a portable distinct name")
    bundle_path, bundle, _ = load_bundle(bundle_path)
    available = [chunk["id"] for chunk in bundle["binding"]["chunks"]]
    assigned = available if chunks is None else list(chunks)
    if not assigned or len(set(assigned)) != len(assigned) or set(assigned)-set(available):
        raise ValueError("Select distinct deterministic chunk identifiers from this bundle")
    policy = bundle["binding"]["memory_policy"]
    process = psutil.Process(); logical = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    admission = resource_admission(workers, len(assigned), policy["memory_per_worker_mb"], policy["memory_headroom_mb"],
        logical_cpus=logical, available_bytes=psutil.virtual_memory().available)
    root = bundle_path.parent / DIRECTORY
    lease_path = root/"hosts"/(host+".lock.json")
    token = uuid.uuid4().hex
    _exclusive_json(lease_path, dict(host=host, pid=os.getpid(), hostname=socket.gethostname(), token=token,
        bundle_sha256=bundle["bundle_sha256"], created_utc=utc_now()))
    receipt_path = root/"hosts"/(host+"-"+token+".json")
    results, started, cpu = [], time.perf_counter(), time.process_time()
    try:
        if (root/"seal.lock.json").exists(): raise RuntimeError("T4c coordinator seal is active")
        with ProcessPoolExecutor(max_workers=admission["admitted_workers"], initializer=worker_initializer,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(execute_chunk, str(bundle_path), host, chunk) for chunk in assigned]
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                print(json.dumps(dict(host=host, completed=len(results), assigned=len(assigned), **result)), flush=True)
        complete, pending = completed_chunks(bundle_path, bundle)
        receipt = dict(contract=CONTRACT, complete=True, host=host, bundle_sha256=bundle["bundle_sha256"],
            resource_admission=admission, selected_chunks=assigned, results=results,
            bundle_complete=not pending, completed_chunks=complete, pending_chunks=pending,
            parent_cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-started, created_utc=utc_now())
        _exclusive_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        _exclusive_json(receipt_path, dict(contract=CONTRACT, complete=False, host=host,
            bundle_sha256=bundle["bundle_sha256"], resource_admission=admission, results=results,
            error=str(error), traceback=traceback.format_exc(), created_utc=utc_now()))
        raise
    finally:
        if lease_path.exists() and read_json(lease_path).get("token") == token: lease_path.unlink()


def seal_bundle(bundle_path):
    bundle_path, bundle, state = load_bundle(bundle_path)
    seal_path = bundle_path.parent/DIRECTORY/"seal.lock.json"
    token = uuid.uuid4().hex
    _exclusive_json(seal_path, dict(token=token, pid=os.getpid(), bundle_sha256=bundle["bundle_sha256"], created_utc=utc_now()))
    try:
        if list((bundle_path.parent/DIRECTORY/"hosts").glob("*.lock.json")):
            raise RuntimeError("T4c host leases must finish before assembly/sealing")
        _, pending = completed_chunks(bundle_path, bundle)
        if pending: raise RuntimeError("T4c bundle still has incomplete chunks: "+",".join(pending))
        runner = _runner(); result = {}
        for name, (root, plan, sources) in state.items():
            for run in sorted({source["run"] for source in sources}): runner.assemble_run(root, plan, run)
            result[name] = runner.seal_candidate(root)
        return result
    finally:
        if seal_path.exists() and read_json(seal_path).get("token") == token: seal_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--chunks", help="comma-separated deterministic chunk IDs; omitted means shared claim queue")
    args = parser.parse_args(argv)
    if args.workers < 1: parser.error("Workers must be positive")
    run_host(args.bundle, args.workers, args.host, args.chunks.split(",") if args.chunks else None)


if __name__ == "__main__":
    main()
