"""Recover only the missing SHA-bound AC44 sources; compose an honest seal."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

from darsia.fluidflower.phase_ffac_rig_raw_dispatch import create_rig_raw_source_processor
from darsia.fluidflower import phase_ffac_rig_raw_dispatch as dispatch
from . import run_ffac_rig_camera_candidate as producer
from .refit_ffac_g3_downstream import atomic_json, below_normal_priority, canonical_sha256, read_json, sha256_file, utc_now

CONTRACT = "ffac-rig-ac44-scientific-recovery-v1"
COMPOSED_CONTRACT = "ffac-rig-camera-scientific-preparation-composed-v1"


def _recovery_code_binding(plan):
    current=producer._code_binding()
    for name in ("producer_sha256","raw_processor_sha256"):
        if current[name] != plan["binding"]["code"][name]:
            raise ValueError("Original numerical source preparation module changed: "+name)
    metadata=producer.metadata
    scientific=dict(source_shape=list(metadata.SOURCE_SHAPE),source_box=list(metadata.SOURCE_BOX),
        scientific_contract=metadata.SCIENTIFIC_CONTRACT,unit_denominator=65535,
        native_dtype="float32",source_receipt_contract=producer.SOURCE_RECEIPT_CONTRACT)
    expected=dict(source_shape=[2979,5472],source_box=[416,496,5472,2979],
        scientific_contract="ffac-camera-clock-full-native-scientific-v1",unit_denominator=65535,
        native_dtype="float32",source_receipt_contract="ffac-rig-camera-prepared-source-v1")
    if scientific != expected:
        raise ValueError("Recovery metadata changed the frozen native scientific layout/contract")
    return dict(current_code=current,metadata=_asset(metadata.__file__),
        original_metadata_sha256=plan["binding"]["code"]["metadata_sha256"],
        scientific_invariants=scientific,
        metadata_substitution_explicit=current["metadata_sha256"]!=plan["binding"]["code"]["metadata_sha256"],
        original_whole_code_binding_pass=current==plan["binding"]["code"],
        original_numerical_producer_and_raw_unchanged=True)


def _asset(path):
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path))


def _receipt_inventory(root, tasks, job_sha):
    present, missing = [], []
    for task in tasks:
        path = producer._source_folder(root, task) / "prepared_source.json"
        if not path.exists():
            missing.append(task)
            continue
        value = read_json(path)
        if (value.get("contract") != producer.SOURCE_RECEIPT_CONTRACT or value.get("complete") is not True
                or value.get("job_sha256") != job_sha or value.get("task") != task):
            raise ValueError("Existing source receipt differs from the immutable preparation plan")
        present.append(dict(task_id=task["task_id"], receipt=_asset(path)))
    return present, missing


def create_recovery_plan(prepared_plan, recovery_root):
    path, recovery_root = Path(prepared_plan), Path(recovery_root)
    plan = read_json(path)
    recovery_binding=_recovery_code_binding(plan)
    root = path.parent
    existing, missing = _receipt_inventory(root, plan["source_inventory"]["scientific_tasks"],plan["job_sha256"])
    if (len(existing) != 889 or len(missing) != 21 or {task["run"] for task in missing} != {"ac44"}):
        raise ValueError("Recovery scope is exactly 889 preserved and 21 missing AC44 source tasks")
    if (root / "scientific_preparation_receipt.json").exists():
        raise ValueError("An original complete preparation seal already exists")
    processor = create_rig_raw_source_processor(missing[0]["parent_native_manifest"]["path"])
    value = dict(contract=CONTRACT, created_utc=utc_now(), original_plan=_asset(path),
        original_job_sha256=plan["job_sha256"], original_code=plan["binding"]["code"],
        original_failure_logs=[_asset(root / name) for name in
                               ("scientific_preparation.stdout.log","scientific_preparation.stderr.log")],
        recovery_code=dict(tool=_asset(__file__),dispatcher=_asset(dispatch.__file__)),
        recovery_metadata_binding=recovery_binding,
        scope=dict(preserved_source_count=889,recovered_source_count=21,total_source_count=910),
        preserved_sources=existing, missing_tasks=missing,
        constructor_preflight=processor.identity, original_failed_job_relabelled=False,
        canonical_modules_unchanged=True, frozen_numerical_bytecode=True, cpu_only=True,
        training=False, labels_changed=False, source_guards_preserved=True,sha_checks_relaxed=False)
    value["recovery_job_sha256"]=canonical_sha256(dict(original_plan=value["original_plan"],
        recovery_code=value["recovery_code"],recovery_metadata_binding=recovery_binding,
        missing_tasks=missing,preserved_sources=existing))
    recovery_root.mkdir(parents=True, exist_ok=False)
    atomic_json(recovery_root / "recovery_plan.json",value)
    return value


def _recover_chunk(root, tasks, job_sha):
    processor = create_rig_raw_source_processor(tasks[0]["parent_native_manifest"]["path"])
    completed=[]
    for task in tasks:
        destination=producer._source_folder(root,task)/"prepared_source.json"
        if destination.exists():
            raise ValueError("Recovery refuses to overwrite or silently resume another source receipt")
        producer.prepare_source(root,task,job_sha,processor=processor)
        completed.append(dict(task_id=task["task_id"],receipt=_asset(destination)))
        print(json.dumps(dict(event="ac44_source_recovered",task_id=task["task_id"],utc=utc_now())),flush=True)
    return completed


def run_recovery(recovery_root, workers=12):
    recovery_root=Path(recovery_root)
    recovery_path=recovery_root/"recovery_plan.json"
    recovery=read_json(recovery_path)
    if recovery.get("contract") != CONTRACT:
        raise ValueError("Unknown targeted recovery plan")
    for asset in [recovery["original_plan"], *recovery["recovery_code"].values(),*recovery["original_failure_logs"]]:
        if sha256_file(asset["path"]) != asset["sha256"]:
            raise ValueError("Recovery's bound code/plan/failure evidence changed")
    plan=read_json(recovery["original_plan"]["path"])
    if _recovery_code_binding(plan) != recovery["recovery_metadata_binding"]:
        raise ValueError("Declared recovery numerical/metadata code changed")
    root=Path(recovery["original_plan"]["path"]).parent
    existing, missing=_receipt_inventory(root,plan["source_inventory"]["scientific_tasks"],plan["job_sha256"])
    if existing != recovery["preserved_sources"] or missing != recovery["missing_tasks"]:
        raise ValueError("Source preparation changed after recovery preflight")
    workers=producer.admitted_workers(workers)
    below_normal_priority()
    atomic_json(recovery_root/"started.json",dict(pid=os.getpid(),workers=workers,threads=1,
        started_utc=utc_now(),recovery_plan=_asset(recovery_path),source_count=len(missing)))
    chunks=[missing[index::workers] for index in range(workers)]
    recovered=[]
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=producer.worker_initializer) as pool:
            futures=[pool.submit(_recover_chunk,str(root),chunk,plan["job_sha256"]) for chunk in chunks if chunk]
            for future in as_completed(futures):
                recovered.extend(future.result())
                print(json.dumps(dict(event="ac44_recovery_progress",completed_sources=len(recovered),
                                      total_sources=len(missing),utc=utc_now())),flush=True)
        # Immutable receipt hashes are the composition boundary. The original
        # 889 numerical files are not re-materialized, and their certificates
        # retain their original producer hashes. Consumers still verify the
        # bound array hashes when reading each source.
        final, absent=_receipt_inventory(root,plan["source_inventory"]["scientific_tasks"],plan["job_sha256"])
        if absent or len(final)!=910:
            raise ValueError("Recovery did not complete the exact scientific source inventory")
        final_by_id={row["task_id"]:row for row in final}
        if any(final_by_id[row["task_id"]]!=row for row in recovery["preserved_sources"]):
            raise ValueError("A preserved scientific source receipt changed during recovery")
        recovered.sort(key=lambda row:row["task_id"])
        if {row["task_id"] for row in recovered}!={task["task_id"] for task in missing}:
            raise ValueError("Recovery source tasks differ from the exact missing inventory")
        result_path=recovery_root/"recovery_result.json"
        atomic_json(result_path,dict(contract=CONTRACT,complete=True,recovery_plan=_asset(recovery_path),
            recovered_sources=recovered,preserved_sources_unchanged=True,cpu_only=True,
            training=False,labels_changed=False,completed_utc=utc_now()))
        seal=dict(contract=COMPOSED_CONTRACT,complete=True,stage="scientific-preparation",
            job_sha256=plan["job_sha256"],source_count=910,run_count=43,frame_count=864,
            sources=sorted(final,key=lambda row:row["task_id"]),original_plan=recovery["original_plan"],
            composition=dict(original_failed_preparation=recovery["original_failure_logs"],
                preserved_source_count=889,recovered_source_count=21,
                recovery_plan=_asset(recovery_path),recovery_result=_asset(result_path),
                recovery_code=recovery["recovery_code"],
                recovery_job_sha256=recovery["recovery_job_sha256"],
                recovery_metadata_binding=recovery["recovery_metadata_binding"],
                original_failed_job_relabelled=False),
            phase_labels_produced=False,source_metadata_finalized=False,
            geometry_acceptance_required_for_inference=True,cpu_only=True,training=False,
            labels_changed=False,automatic_activation=False)
        seal_path=root/"scientific_preparation_composed_receipt.json"
        if seal_path.exists():
            raise ValueError("Composed scientific seal already exists")
        atomic_json(seal_path,seal)
        atomic_json(recovery_root/"exit.json",dict(status="succeeded",completed_utc=utc_now(),
            composed_preparation=_asset(seal_path),recovered_sources=21,preserved_sources=889))
        print(json.dumps(dict(event="ac44_recovery_completed",composed_preparation=_asset(seal_path))),flush=True)
    except BaseException as error:
        import traceback
        atomic_json(recovery_root/"exit.json",dict(status="failed",completed_utc=utc_now(),
            error=repr(error),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepared-plan",type=Path)
    parser.add_argument("--recovery-root",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        if args.prepared_plan is None: parser.error("--prepared-plan required")
        result=create_recovery_plan(args.prepared_plan,args.recovery_root)
        print(json.dumps(dict(event="ac44_recovery_preflight_pass",scope=result["scope"],
                             recovery_plan=_asset(args.recovery_root/"recovery_plan.json"))),flush=True)
    if args.run: run_recovery(args.recovery_root,args.workers)


if __name__=="__main__": main()
