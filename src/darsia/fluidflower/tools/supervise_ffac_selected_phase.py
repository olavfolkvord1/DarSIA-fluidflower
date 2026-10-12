"""Operational readiness supervisor for an unchanged selected phase campaign.

Runs wait for their own source/predecessor receipts while preparation uses the
other allocated workers. A failed fixed replay gate creates a shared stop;
no production or candidate registration is performed by this supervisor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
import traceback

from darsia.fluidflower.phase_ffac_selected_phase import asset,bound_json
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json,utc_now
from darsia.fluidflower.tools import run_ffac_selected_phase as core

CONTRACT="ffac-selected-phase-readiness-supervisor-v1"
_OPS={}


def publish_stop(path,reason):
    path=Path(path)
    if path.exists():return bound_json(asset(path))
    record=dict(contract="ffac-selected-phase-replay-stop-v1",created_utc=utc_now(),stop=True,
        stop_new_work=True,production_allowed=False,reason=reason,
        run=reason.get("run"),source_sha256=reason.get("source_sha256"),
        receipt=reason.get("receipt"),phase_plan=reason.get("phase_plan"))
    try:atomic_json(path,record)
    except (FileExistsError,ValueError):
        if not path.exists():raise
    return bound_json(asset(path))


def source_ready(root,run,source,plan):
    path=Path(root)/"selected_sources"/run/(source["source_sha256"])/"receipt.json"
    if not path.is_file():return False
    descriptor=asset(path);receipt=bound_json(descriptor)
    if (receipt.get("complete") is not True or receipt.get("run")!=run
        or receipt.get("source_sha256")!=source["source_sha256"]
        or receipt.get("elapsed_h")!=source["elapsed_h"]):
        raise ValueError("Prepared source readiness lost exact camera identity/time")
    if receipt["binding"]["job_sha256"]!=plan["job_sha256"]:
        imports=plan.get("source_imports",{}).get(run,{})
        if imports.get(source["source_sha256"])!=descriptor:
            raise ValueError("Prepared source comes from an unbound producer/import")
    return True


def wait_sources(root,run,sources,plan,stop_path,*,poll_seconds=2.,timeout_seconds=21600.):
    started=time.monotonic();last_log=0.
    while True:
        if Path(stop_path).exists():return False
        missing=[r for r in sources if not source_ready(root,run,r,plan)]
        if not missing:return True
        elapsed=time.monotonic()-started
        if elapsed>timeout_seconds:
            raise TimeoutError("Source-preparation prerequisites remain missing: "+
                ",".join(r["source_sha256"] for r in missing))
        if elapsed-last_log>=30. or last_log==0.:
            print(json.dumps(dict(event="waiting_for_bound_sources",run=run,
                missing_source_sha256=[r["source_sha256"] for r in missing],wait_seconds=elapsed)),flush=True)
            last_log=max(.001,elapsed)
        time.sleep(poll_seconds)


def initialise(config):
    core.initialise(config["plan"],config["output_root"],config["backend"])
    _OPS.update(config)


def replay_run(run):
    from darsia.fluidflower.phase_ffac_selected_inference import SelectedPhaseEvaluator
    plan=core._STATE["plan"];state=plan["runs"][run]
    inventory=bound_json(state["source_inventory"])
    lookup={row["source_sha256"]:row for row in inventory["sources"]}
    source_root=plan["scientific_source_root"]
    baseline=lookup[inventory["baseline_source_sha256"]]
    receipts=[];started=time.perf_counter()
    try:
        if not wait_sources(source_root,run,[baseline],plan,_OPS["stop_path"],
                timeout_seconds=_OPS["source_wait_timeout_seconds"]):
            return dict(run=run,complete=False,stopped="shared strict replay stop",receipts=[])
        evaluator=SelectedPhaseEvaluator(_OPS["output_root"],plan,run,backend=_OPS["backend"])
        for row in core.select_rows(inventory,"replay"):
            if Path(_OPS["stop_path"]).exists():
                return dict(run=run,complete=False,stopped="shared strict replay stop",receipts=receipts)
            required=[row]
            previous=row["previous_source_sha256"]
            if previous is not None:required.append(lookup[previous])
            if not wait_sources(source_root,run,required,plan,_OPS["stop_path"],
                timeout_seconds=_OPS["source_wait_timeout_seconds"]):
                return dict(run=run,complete=False,stopped="shared strict replay stop",receipts=receipts)
            resumed=evaluator.saved(row["source_sha256"])
            saved_asset=resumed or evaluator.save(evaluator.evaluate(row["source_sha256"]))
            saved=bound_json(saved_asset);replay=saved["qa"]["strict_studio_replay"]
            receipts.append(saved_asset)
            print(json.dumps(dict(event="studio_anchor_replayed",run=run,
                studio_frame_index=row["studio_frame_index"],source_sha256=row["source_sha256"],
                elapsed_h=row["elapsed_h"],seconds=saved["qa"]["seconds"],resume=resumed is not None,
                co2_changed_fraction=replay["co2"]["changed_area_fraction"],
                gas_changed_fraction=replay["gas"]["changed_area_fraction"],all_pass=replay["all_pass"])),flush=True)
            if not replay["all_pass"]:
                publish_stop(_OPS["stop_path"],dict(kind="fixed strict Studio replay gate failed",
                    run=run,source_sha256=row["source_sha256"],receipt=saved_asset,
                    replay=replay,phase_plan=_OPS["plan"]))
                return dict(run=run,complete=False,stopped="fixed strict Studio replay gate failed",
                    receipts=receipts,failed_replay=replay,seconds=time.perf_counter()-started)
        return dict(run=run,complete=True,receipts=receipts,seconds=time.perf_counter()-started)
    except Exception as error:
        failure=dict(kind="phase readiness/inference error",run=run,error=repr(error),
            source_sha256=row["source_sha256"] if "row" in locals() else baseline["source_sha256"],
            traceback=traceback.format_exc(),phase_plan=_OPS["plan"])
        publish_stop(_OPS["stop_path"],failure)
        return dict(run=run,complete=False,error=failure,receipts=receipts,
            seconds=time.perf_counter()-started)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    config_asset=asset(args.config);config=bound_json(config_asset)
    if config.get("contract")!=CONTRACT or config.get("stage")!="replay":
        raise ValueError("This operational supervisor may only replay Studio anchors")
    if config["workers"]<1 or len(set(config["runs"]))!=len(config["runs"]):
        raise ValueError("Invalid disjoint bounded run shard")
    plan=bound_json(config["plan"]);core.verify_code(plan)
    if any(run not in plan["runs"] for run in config["runs"]):
        raise ValueError("Operational run shard escapes the frozen source/phase plan")
    output=Path(config["stage_receipt"])
    if output.exists():raise FileExistsError("Operational completion receipt already exists")
    started=utc_now();results=[]
    with ProcessPoolExecutor(max_workers=min(config["workers"],len(config["runs"])),
        mp_context=multiprocessing.get_context("spawn"),initializer=initialise,
        initargs=(config,)) as executor:
        pending={executor.submit(replay_run,run):run for run in config["runs"]}
        for future in as_completed(pending):
            if future.cancelled():continue
            try:result=future.result()
            except Exception as error:
                result=dict(run=pending[future],complete=False,error=repr(error))
                publish_stop(config["stop_path"],dict(kind="operational worker failure",result=result))
            results.append(result)
            if not result["complete"]:
                for task in pending:task.cancel()
    complete=(len(results)==len(config["runs"]) and all(row["complete"] for row in results))
    atomic_json(output,dict(contract=CONTRACT,complete=complete,stage="replay",config=config_asset,
        phase_plan=config["plan"],host=socket.gethostname(),pid=os.getpid(),started_utc=started,
        finished_utc=utc_now(),runs=config["runs"],workers=config["workers"],internal_threads=1,
        results=results,shared_stop=asset(config["stop_path"]) if Path(config["stop_path"]).exists() else None,
        production_started=False,models_or_labels_changed=False))
    return 0 if complete else 2


if __name__=="__main__":
    raise SystemExit(main())
