"""Display-only preparation/finalization of committed selected RAW sources."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

from darsia.fluidflower.phase_ffac_selected_display_preparation import SourceReadyDisplayPreparer
from darsia.fluidflower.phase_ffac_selected_policy_a_baseline_qa import EvidenceReader, asset
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, utc_now

_CACHE = None


def worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import cv2
    cv2.setNumThreads(1)


def work_batch(root, plan_asset, run, digests, stage, backend, stop_path):
    global _CACHE
    key = (root, plan_asset["sha256"], run, backend)
    if _CACHE is None or _CACHE[0] != key:
        reader = EvidenceReader();plan=reader.record(plan_asset)
        _CACHE = (key, SourceReadyDisplayPreparer(root,plan,run,backend=backend,reader=reader))
    renderer = _CACHE[1]
    outputs=[]
    for digest in digests:
        if stop_path and Path(stop_path).exists():
            return dict(run=run,stopped=True,outputs=outputs)
        if stage in ("prepare","both"):
            outputs.append(dict(source_sha256=digest,stage="prepare",receipt=renderer.prepare(digest)))
        if stage in ("finalize","both"):
            phase=Path(root)/"selected_phase_frames"/run/(digest+".json")
            if phase.is_file():
                outputs.append(dict(source_sha256=digest,stage="finalize",receipt=renderer.finalize(digest)))
    return dict(run=run,stopped=False,outputs=outputs)


def pending_batches(root, plan, inventories, stage, batch_size):
    source_root=Path(plan["scientific_source_root"]);root=Path(root)
    batches=[];missing=0;total=0;done=0
    for run, inventory in inventories.items():
        pending=[]
        for row in sorted(inventory["sources"],key=lambda item:float(item["elapsed_h"])):
            if not (row.get("selected") or row.get("baseline")):continue
            digest=row["source_sha256"];total+=1
            source=source_root/"selected_sources"/run/digest/"receipt.json"
            preparation=root/"display_preparations"/run/digest/"receipt.json"
            phase=root/"selected_phase_frames"/run/(digest+".json")
            published=root/"display_renderings"/run/digest/"receipt.json"
            if not source.is_file():missing+=1;continue
            wanted=(not preparation.is_file()) if stage=="prepare" else (
                not published.is_file() and phase.is_file() and (preparation.is_file() or stage=="both"))
            if stage=="both" and not preparation.is_file():wanted=True
            if wanted:pending.append(digest)
            else:done+=int(preparation.is_file() if stage=="prepare" else published.is_file())
        if pending:batches.append((run,pending[:batch_size]))
    return batches,dict(total=total,completed=done,missing_sources=missing)


def execute(root, plan_asset, runs, *, workers=2, stage="prepare", backend="auto", watch=False,
            poll_seconds=10, batch_size=2, stop_path=None, report=None):
    if report and Path(report).exists():
        raise FileExistsError("Display operation reports are immutable")
    reader=EvidenceReader();plan=reader.record(plan_asset);root=Path(root)
    inventories={run:reader.record(plan["runs"][run]["source_inventory"]) for run in runs}
    completed=[];started=utc_now();round_number=0
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_init) as pool:
        while True:
            if stop_path and Path(stop_path).exists():break
            batches,counts=pending_batches(root,plan,inventories,stage,batch_size)
            if not batches:
                if not watch or counts["completed"]==counts["total"]:break
                time.sleep(poll_seconds);continue
            round_number+=1
            futures={pool.submit(work_batch,str(root),plan_asset,run,digests,stage,backend,stop_path):run
                for run,digests in batches}
            for future in as_completed(futures):
                try:
                    result=future.result()
                except Exception:
                    for pending in futures:pending.cancel()
                    raise
                completed.extend(result["outputs"])
                print(json.dumps(dict(run=result["run"],new_receipts=len(result["outputs"]),stopped=result["stopped"])),flush=True)
            if not watch:
                # Drain the current committed snapshot; never wait for phase
                # production or an unavailable source in a one-shot command.
                batches,_=pending_batches(root,plan,inventories,stage,batch_size)
                if not batches:break
    _,counts=pending_batches(root,plan,inventories,stage,batch_size)
    result=dict(contract="ffac-selected-display-preparation-operation-v1",complete=True,
        started_utc=started,finished_utc=utc_now(),stage=stage,plan=plan_asset,runs=list(runs),
        backend=backend,workers=workers,outputs=completed,counts=counts,
        all_requested_displays_complete=counts["completed"]==counts["total"],
        stopped=bool(stop_path and Path(stop_path).exists()),
        model_inference=False,scientific_inputs_changed=False,maps_changed=False,labels_changed=False)
    if report:
        if Path(report).exists():raise FileExistsError("Display operation reports are immutable")
        atomic_json(report,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True);parser.add_argument("--plan",required=True)
    parser.add_argument("--runs",nargs="+");parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--stage",choices=("prepare","finalize","both"),default="prepare")
    parser.add_argument("--backend",choices=("cpu","auto"),default="auto")
    parser.add_argument("--watch",action="store_true");parser.add_argument("--poll-seconds",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=2);parser.add_argument("--stop-path");parser.add_argument("--report",required=True)
    args=parser.parse_args()
    if args.workers<1 or args.batch_size<1 or args.poll_seconds<1:parser.error("Positive worker/batch/poll values required")
    descriptor=asset(args.plan);plan=EvidenceReader().record(descriptor)
    result=execute(args.root,descriptor,args.runs or sorted(plan["runs"]),workers=args.workers,
        stage=args.stage,backend=args.backend,watch=args.watch,poll_seconds=args.poll_seconds,
        batch_size=args.batch_size,stop_path=args.stop_path,report=args.report)
    print(json.dumps(result["counts"]),flush=True)
    return 0


if __name__=="__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
