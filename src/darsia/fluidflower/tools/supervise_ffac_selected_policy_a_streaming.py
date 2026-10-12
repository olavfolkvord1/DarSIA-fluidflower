"""Operational ready-run dispatch into the immutable main-CPU Policy A driver.

Reads metadata only; each run has its own unchanged producer invocation,
ordered evaluator, process evidence and stage receipt. No scientific imports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import socket
from pathlib import Path
import subprocess
import time


DRIVER = "darsia.fluidflower.tools.run_ffac_selected_policy_a_streaming"
_MACHINE = r"""$o=Get-CimInstance Win32_OperatingSystem; $rows=@(Get-CimInstance Win32_Process | ForEach-Object {$v=Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if($v){[pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;working_gib=$v.WorkingSet64/1GB;peak_gib=$v.PeakWorkingSet64/1GB;creation_utc=$_.CreationDate.ToUniversalTime().ToString('o')}}}); [pscustomobject]@{free_gib=$o.FreePhysicalMemory*1KB/1GB;total_gib=$o.TotalVisibleMemorySize*1KB/1GB;processes=$rows}|ConvertTo-Json -Depth 4 -Compress"""


def now():
    return datetime.now(timezone.utc).isoformat()


def asset(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(item):
    path = Path(item["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Operational metadata SHA changed: " + str(path))
    return json.loads(raw)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stop_new_work(path, reason, plan_descriptor):
    try:
        write_new(path, {"contract": "ffac-selected-policy-a-operational-stop-v1",
            "stop": True, "stop_new_work": True, "reason": reason,
            "stopped_utc": now(), "pid": os.getpid(), "plan": plan_descriptor,
            "production_allowed": False})
    except FileExistsError:
        pass


def tree_rss(processes, root_pid):
    known = {int(root_pid)}
    previous = 0
    while len(known) != previous:
        previous = len(known)
        known.update(int(row["pid"]) for row in processes if int(row["parent"]) in known)
    return sum(float(row["working_gib"]) for row in processes if int(row["pid"]) in known)


def admitted_slots(*, free_gib, reserve_gib, phase_budget_gib, active_phase_rss,
                   source_workers, source_budget_gib, source_rss, max_workers):
    # Current RSS is already removed from free RAM. Reserve the remaining
    # projected peak of every active writer before admitting a new evaluator.
    future = sum(max(0., phase_budget_gib-rss) for rss in active_phase_rss)
    future += max(0., source_workers*source_budget_gib-source_rss)
    available = free_gib-reserve_gib-future
    return max(0, min(max_workers-len(active_phase_rss),
                      math.floor(available/phase_budget_gib)))


class _ReadyMetadataReader:
    """One session SHA verification of small immutable metadata descriptors."""
    def __init__(self):
        self.cache = {}
    def record(self, descriptor):
        key = (descriptor["path"], descriptor["sha256"])
        if key not in self.cache:
            self.cache[key] = bound(descriptor)
        return self.cache[key]


def ready_run(plan, run, inventories, cache):
    from darsia.fluidflower.phase_ffac_selected_execution_admission import verify_ready_source, prefix_rows
    root = Path(plan["scientific_source_root"])
    reader = cache.setdefault("metadata_reader", _ReadyMetadataReader())
    for source in prefix_rows(inventories[run]):
        if not any(source.get(flag) for flag in ("selected", "baseline", "studio_anchor")):
            continue
        path = root/"selected_sources"/run/source["source_sha256"]/"receipt.json"
        if not path.is_file():
            return False
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        try:
            if key not in cache:
                raw = path.read_bytes()
                descriptor = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
                json.loads(raw)  # A metadata publisher may still be closing.
                verify_ready_source(descriptor, source, plan, reader)
                cache[key] = descriptor
        except json.JSONDecodeError:
            return False
    return True


def command(python, root, plan, run, stage_receipt):
    return [python, "-m", DRIVER, "--root", str(root), "--plan", str(plan),
            "--runs", run, "--workers", "1", "--source-wait-seconds", "21600",
            "--stage-receipt", str(stage_receipt)]


def require_main_host(plan):
    """Reject another machine before dispatch or heavy context allocation."""
    descriptor = plan["phase_campaign_binding"]["approved_main_host"]
    approved = bound(descriptor)
    if (socket.gethostname() != approved["hostname"]
            or platform.machine() != approved["machine"]):
        raise ValueError("Policy A supervisor requires the accepted Main CPU host")
    return descriptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "root", "operations", "python", "runtime-src", "powershell"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--phase-budget-gib", type=float, required=True)
    parser.add_argument("--reserve-gib", type=float, default=40.)
    parser.add_argument("--source-process-json")
    parser.add_argument("--source-workers", type=int, default=0)
    parser.add_argument("--source-budget-gib", type=float, default=0.)
    parser.add_argument("--poll-seconds", type=float, default=15.)
    parser.add_argument("--wait-hours", type=float, default=12.)
    args = parser.parse_args(argv)
    if (args.phase_budget_gib <= 0 or args.reserve_gib < 0 or args.max_workers < 1
            or args.source_workers < 0 or args.poll_seconds <= 0
            or args.source_workers and (not args.source_process_json or args.source_budget_gib <= 0)):
        raise ValueError("Measured positive resource budgets and source-owner evidence are required")
    if os.name != "nt":
        raise RuntimeError("This operational launch is restricted to the approved Windows main host")
    plan_descriptor = asset(args.plan)
    plan = bound(plan_descriptor)
    if plan.get("phase_campaign_binding", {}).get("backend") != "cpu":
        raise ValueError("Only the admitted CPU campaign can be dispatched")
    require_main_host(plan)
    from darsia.fluidflower.phase_ffac_selected_execution_admission import require_operation
    require_operation(plan, args.root, _ReadyMetadataReader())
    stop = Path(plan["execution_operation"]["shared_stop_path"])
    if stop.exists():
        raise ValueError("Shared stop already exists; no operational retry")
    operations = Path(args.operations)
    operations.mkdir(parents=True, exist_ok=False)
    inventories = {run: bound(state["source_inventory"]) for run,state in plan["runs"].items()}
    environment = dict(os.environ, PYTHONPATH=args.runtime_src,
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    pending, active, results, cache = set(plan["runs"]), {}, [], {}
    started = now()
    write_new(operations/"supervisor_started.json", dict(started_utc=started, pid=os.getpid(),
        plan=plan_descriptor, arguments=vars(args), script=asset(__file__),
        prefix_ready_runs_only=True, execution_sha256=plan["execution_sha256"], numerical_modules_changed=False))
    deadline = time.monotonic()+args.wait_hours*3600
    failure = None
    try:
        while pending or active:
            for run, state in list(active.items()):
                exit_code = state["process"].poll()
                if exit_code is None:
                    continue
                state["stdout"].close(); state["stderr"].close()
                stage_path = state["stage_path"]
                row = dict(run=run, exit_code=exit_code, finished_utc=now(), pid=state["process"].pid)
                if stage_path.is_file():
                    row["stage_receipt"] = asset(stage_path)
                    stage = bound(row["stage_receipt"])
                    row["complete"] = stage.get("complete") is True and stage.get("plan") == plan_descriptor and stage.get("runs") == [run]
                else:
                    row["complete"] = False
                write_new(state["folder"]/"done.json", row)
                results.append(row); del active[run]
                if exit_code != 0 or not row["complete"]:
                    failure = "Immutable run producer failed: " + run
                    stop_new_work(stop, failure, plan_descriptor)
            if stop.exists():
                failure = "Shared scientific/operational stop"
            if failure:
                pending.clear()
                if active:
                    time.sleep(args.poll_seconds); continue
                break
            if time.monotonic() >= deadline:
                failure = "Source/readiness dispatch timeout"
                stop_new_work(stop, failure, plan_descriptor)
                pending.clear()
                if active:
                    time.sleep(args.poll_seconds); continue
                break
            machine = json.loads(subprocess.check_output([args.powershell, "-NoProfile", "-Command", _MACHINE], text=True))
            processes = machine["processes"]
            phase_rss = [tree_rss(processes, state["process"].pid) for state in active.values()]
            source_workers, source_rss = 0, 0.
            if args.source_workers:
                owner = json.loads(Path(args.source_process_json).read_text(encoding="utf-8-sig"))
                source_pid = int(owner["pid"])
                if any(int(row["pid"]) == source_pid for row in processes):
                    source_workers, source_rss = args.source_workers, tree_rss(processes, source_pid)
            slots = admitted_slots(free_gib=machine["free_gib"], reserve_gib=args.reserve_gib,
                phase_budget_gib=args.phase_budget_gib, active_phase_rss=phase_rss,
                source_workers=source_workers, source_budget_gib=args.source_budget_gib,
                source_rss=source_rss, max_workers=args.max_workers)
            ready = [run for run in sorted(pending) if ready_run(plan, run, inventories, cache)]
            # Prefer already fully prepared runs to avoid holding reference contexts idle.
            source_root = Path(plan["scientific_source_root"])
            ready.sort(key=lambda run: (sum(not (source_root/"selected_sources"/run/row["source_sha256"]/"receipt.json").is_file()
                for row in inventories[run]["sources"]), run))
            for run in ready[:slots]:
                if stop.exists():
                    break
                folder = operations/run
                folder.mkdir(exist_ok=False)
                stage_path = folder/"stage.json"
                args_list = command(args.python, args.root, args.plan, run, stage_path)
                stdout = (folder/"stdout.log").open("w", encoding="utf-8")
                stderr = (folder/"stderr.log").open("w", encoding="utf-8")
                child = subprocess.Popen(args_list, env=environment, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
                active[run] = dict(process=child, folder=folder, stage_path=stage_path, stdout=stdout, stderr=stderr)
                pending.remove(run)
                write_new(folder/"process.json", dict(run=run, pid=child.pid, started_utc=now(),
                    arguments=args_list, plan=plan_descriptor, phase_budget_gib=args.phase_budget_gib,
                    admission={"free_gib": machine["free_gib"], "phase_rss": phase_rss,
                        "source_workers": source_workers, "source_rss": source_rss,
                        "reserve_gib": args.reserve_gib, "admitted_slots": slots}))
                print(json.dumps(dict(event="ready_run_dispatched", run=run, pid=child.pid,
                    active_runs=len(active), pending_runs=len(pending))), flush=True)
            time.sleep(args.poll_seconds)
    except Exception as error:
        failure = repr(error)
        stop_new_work(stop, failure, plan_descriptor)
        # The immutable core checks this marker before dispatch and after each
        # numerical evaluation; let its owned writers close preserved evidence.
        while active:
            for run, state in list(active.items()):
                code = state["process"].poll()
                if code is not None:
                    state["stdout"].close(); state["stderr"].close()
                    results.append(dict(run=run, exit_code=code, complete=False, stopped_by_supervisor=True))
                    del active[run]
            if active:
                time.sleep(args.poll_seconds)
    complete = failure is None and len(results) == len(plan["runs"]) and all(row["complete"] for row in results)
    write_new(operations/"supervisor_done.json", dict(started_utc=started, finished_utc=now(),
        complete=complete, failure=failure, results=results, plan=plan_descriptor,
        all_runs=len(plan["runs"]), numerical_modules_changed=False, scorer_pending=True,
        mass_admitted=False, registered=False))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
