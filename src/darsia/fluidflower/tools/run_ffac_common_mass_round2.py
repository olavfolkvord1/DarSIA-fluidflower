"""Bound, resumable R2 extraction, LORO, freeze and native map production.

No held-out/descriptive scientific pixels are admitted before the new freeze.
The production stage is unavailable unless the fixed final check passes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import time
import uuid

for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"

import numpy as np
from darsia.fluidflower import phase_ffac_common_mass_inputs as source
from darsia.fluidflower import phase_ffac_common_mass_round2_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_round2_production as output
from darsia.fluidflower import phase_ffac_common_mass_production as saved

CONTRACT="ffac-common-mass-round2-campaign-v1"
_SESSION={}


def engine():
    return importlib.import_module("darsia.fluidflower.phase_ffac_common_mass_round2")


def supported_candidates():
    return tuple(name for name, record in engine().candidate_availability().items() if record["available"])


def bound_modules():
    names=("phase_ffac_common_mass_round2_backend","phase_ffac_common_mass",
        "concentration_joint","concentration_native","tools.phase_mass_joint_optics")
    return (engine(),inputs,output,saved,source)+tuple(importlib.import_module("darsia.fluidflower."+name) for name in names)


def verify_worker_code(campaign_path):
    binding=output.read_json(campaign_path)
    if saved.sha_file(__file__)!=binding["driver_sha256"]:
        raise ValueError("The worker driver differs from the immutable campaign")
    for name,expected in binding["code_sha256"].items():
        module=importlib.import_module(name)
        if saved.sha_file(module.__file__)!=expected:
            raise ValueError(f"The worker scientific dependency changed: {name}")


def new_json(path,value):
    return saved.write_new_json(path,value)


def unchanged_or_new(path,value):
    path=Path(path)
    if path.is_file():
        if output.read_json(path)!=saved._jsonable(value):
            raise ValueError(f"Immutable R2 binding differs: {path}")
    else:new_json(path,value)


def commit(folder,value):
    folder=Path(folder);folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists():raise FileExistsError(folder)
    temporary=folder.with_name(f".{folder.name}.{uuid.uuid4().hex}.partial")
    temporary.mkdir();new_json(temporary / "receipt.json",dict(value,complete=True))
    temporary.rename(folder)


def resume(folder,binding):
    path=Path(folder) / "receipt.json"
    if not path.is_file():return None
    data=output.read_json(path)
    if data.get("complete") is not True or data.get("campaign_binding_sha256")!=binding:
        raise ValueError(f"Old/partial receipt cannot be rebound to this R2 job: {path}")
    if "summary" in data and data.get("summary_sha256")!=output.digest(data["summary"]):
        raise ValueError(f"Saved reduced statistics changed: {path}")
    return data


def init_worker(plan_path,round1_freeze,backend,l1_reference=None,campaign_path=None):
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    if campaign_path is not None:verify_worker_code(campaign_path)
    plan=output.read_json(plan_path);inputs.validate_split(plan["split"])
    r1=output.read_json(round1_freeze)
    _SESSION.update(plan=plan,index=inputs.metadata_index(plan),reader=inputs.VerifiedAssetReader(),
        round1_freeze=str(round1_freeze),path_folder=r1["path_folder"],backend=backend,
        limits=plan["physical_limits"],threads=threadpool_limits(limits=1),l1_reference=l1_reference)


def rows_for_run(run):
    return sorted((row for row in _SESSION["index"]["frames"] if row["run"]==run),key=lambda r:r["frame_index"])


def prepare(metadata,freeze=None):
    # The input guard consumes the verified record; the native executor binds
    # and verifies the actual immutable freeze file, never an unbound dict.
    evidence=output.FrozenEvidence.verify(freeze) if isinstance(freeze,(str,Path)) else freeze
    frozen_record=evidence.record if evidence is not None else None
    frozen_path=evidence.path if evidence is not None else None
    prepared=inputs.load_common_frame(metadata,_SESSION["plan"],
        session_reader=_SESSION["reader"],frozen_calibration=frozen_record,
        clearwater_transport=_SESSION["plan"]["clearwater_transports"][metadata["run"]])
    neighbourhood=None
    if _SESSION.get("l1_reference") and metadata["elapsed_h"]>=4:
        neighbourhood,proxy=source.l1_protocol_proxy_neighbourhood(metadata,_SESSION["l1_reference"],session_reader=_SESSION["reader"])
        proxy["l1_position_source_sha256"]=_SESSION["l1_reference"]["sha256"]
        prepared=replace(prepared,provenance=dict(prepared.provenance,**proxy))
    frame=inputs.to_native_frame(prepared,l1_neighbourhood_mask=neighbourhood)
    # R1's executor used only measurable excluded volumes. R2 also needs the
    # *geometric* strip masks at missing-depth pixels to identify each finite U
    # part. No geometry is imputed into the main mass by retaining these masks.
    bottom=np.asarray(prepared.bottom_strip_mask,bool) & frame.domain
    below=np.asarray(prepared.below_photo_filled_mask,bool) & frame.domain & ~bottom
    frame=replace(frame,excluded_masks=dict(frame.excluded_masks or {},bottom_strip=bottom,below_photo_filled=below))
    response=engine().prepare_round2_response(frame,_SESSION["path_folder"],
        swi_by_facies=_SESSION["limits"]["swi_by_facies"],r1_frozen_calibration_path=_SESSION["round1_freeze"],frozen_calibration=frozen_path)
    return frame,response


def no_outflow_proxy(frame,metadata):
    occupied=frame.domain.any(axis=0);first=frame.domain.argmax(axis=0)
    contact=int(np.count_nonzero(occupied & np.isin(frame.phase_labels[first,np.arange(frame.domain.shape[1])],(1,2))))
    return dict(no_co2_left_rig=contact==0,source=dict(metadata["receipt"]),co2_contact_columns=contact,
        status="inferred_locked_phase_open_top_proxy",spatial_margin_px=0,
        limitation="No measured CO2 outlet; fixed phase/top rules may weaken optical evidence",
        measured_water_displacement_is_not_CO2_outlet=True)


def summarize(frame,response,metadata):
    return engine().summarize_round2_frame(response,no_outflow_evidence=no_outflow_proxy(frame,metadata),
        light_epoch=frame.provenance.get("light_epoch",output.light_group(frame.run_id)),
        max_depth_m=_SESSION["limits"]["max_depth_m"],max_porosity=_SESSION["limits"]["max_porosity"],backend=_SESSION["backend"])


def extract_run(task):
    run,job,binding=task
    if run not in output.CALIBRATION:raise ValueError("R2 extraction is calibration26 only")
    _SESSION["reader"]=inputs.VerifiedAssetReader()
    started=time.perf_counter();cpu=time.process_time();records=[]
    for metadata in rows_for_run(run):
        folder=Path(job) / "work/extraction" / run / f"frame_{metadata['frame_index']:03d}"
        cached=resume(folder,binding)
        if cached:
            records.append(dict(frame_index=metadata["frame_index"],cached=True));continue
        tick=time.perf_counter();frame,response=prepare(metadata)
        summary=summarize(frame,response,metadata)
        source_binding={k:metadata.get(k) for k in ("run","frame_index","elapsed_h","source_sha256","labels","area_map","receipt")}
        commit(folder,dict(contract=CONTRACT,stage="extract",campaign_binding_sha256=binding,summary=summary,
            summary_sha256=output.digest(summary),
            source_binding=source_binding,source_binding_sha256=output.digest(source_binding),
            wall_seconds=time.perf_counter()-tick,created_utc=datetime.now(timezone.utc).isoformat(),
            full_resolution=True,RAW_downsampled=False,model_or_labels_changed=False))
        records.append(dict(frame_index=metadata["frame_index"],cached=False))
        del frame,response,summary;gc.collect()
        print(json.dumps(dict(stage="extract",run=run,frame_index=metadata["frame_index"],committed=True)),flush=True)
    return dict(run=run,frame_count=len(records),cached_frames=sum(r["cached"] for r in records),
        wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu)


def read_extraction(job,binding):
    rows=[]
    for path in sorted((Path(job) / "work/extraction").glob("*/frame_*/receipt.json")):
        value=resume(path.parent,binding);row=value["summary"]
        if row["run"] not in output.CALIBRATION:raise ValueError("Out-of-fit run leaked into extraction")
        rows.append(row)
    if {r["run"] for r in rows}!=output.CALIBRATION:
        raise ValueError("All26 calibration runs require complete extraction")
    return rows


def validate_full_extraction(rows,index):
    expected={(r["run"],r["frame_index"]) for r in index["frames"] if r["run"] in output.CALIBRATION}
    actual={(r["run"],r["frame_index"]) for r in rows}
    if actual!=expected or len(actual)!=len(rows):raise ValueError("Reduced statistics must cover every calibration Studio frame exactly once")


def fold_task(task):
    candidate,leftout,job,binding=task
    folder=Path(job) / "work/loro" / candidate / leftout
    cached=resume(folder,binding)
    if cached:return dict(candidate=candidate,leftout=leftout,cached=True)
    started=time.perf_counter();rows=read_extraction(job,binding)
    fit=engine().fit_round2_candidate(rows,candidate,excluded_run=leftout)
    predicted=[]
    for row in rows:
        if row["run"]!=leftout:continue
        result=engine().evaluate_round2_summary(row,fit["parameters"],fit["anchors_by_run"][leftout])
        predicted.append(output.mass_row(result,dict(candidate=candidate,leftout_run=leftout,
            calibration_runs=sorted(output.CALIBRATION-{leftout}),light_group=output.light_group(leftout,supplied=row.get("light_epoch")),
            source_binding=row.get("source_binding"),source_sha256=row.get("source_sha256"))))
    commit(folder,dict(contract=CONTRACT,stage="loro",campaign_binding_sha256=binding,candidate=candidate,leftout_run=leftout,
        fit=fit,predictions=predicted,selection_uses_heldout=False,colour_paths_unchanged=True,wall_seconds=time.perf_counter()-started))
    return dict(candidate=candidate,leftout=leftout,cached=False,frame_count=len(predicted),wall_seconds=time.perf_counter()-started)


def score_stage(args,binding,index):
    scores=[];rows=[]
    for candidate in supported_candidates():
        predictions=[]
        for run in sorted(output.CALIBRATION):
            value=resume(args.job / "work/loro" / candidate / run,binding)
            if value is None:raise ValueError("All26 LORO folds required for every supported candidate")
            predictions.extend(value["predictions"])
        validate_full_extraction(predictions,index)
        scored=output.score_predictions(predictions);scores.append(dict(scored,candidate=candidate))
        rows.extend(predictions)
    selection=output.choose_candidate(scores)
    record=dict(contract=CONTRACT,scores=scores,selection=selection,N3_status=engine().candidate_availability()["N3"]["reason"],
        selection_reads_heldout=False,campaign_binding_sha256=binding)
    unchanged_or_new(args.job / "selection.json",record)
    report=args.job / "selection_report"
    if not report.exists():output.write_tables(report,rows,stage="calibration26-LORO-selection")
    return record


def require_workflow_plan(args):
    binding=output.read_json(args.workflow_binding)
    descriptor=binding["calibration_plan"]
    plan=source.read_bound_json(descriptor)
    config=plan["configuration"];contract=config["native_fixed_phase_contract"]
    if contract.get("contract")!="native-round2-fixed-phase-v1" or contract.get("input_plan",{}).get("sha256")!=source.asset(args.input_plan)["sha256"]:
        raise ValueError("Saved existing workflow plan is not this source-bound R2 procedure")
    if set(contract["calibration_runs"])!=output.CALIBRATION or set(contract["heldout_runs"])!=output.HELDOUT or set(contract["descriptive_runs"])!=output.DESCRIPTIVE:
        raise ValueError("Saved R2 workflow split changed")
    if contract.get("approved_spec_sha256")!=args.spec_sha256 or contract.get("gas_rule")!="fixed 1-Swi; remaining water at local solubility; neither diluted nor fitted":
        raise ValueError("R2 specification or fixed gas rule differs")
    if contract.get("map_rescaling_allowed") is not False or contract.get("overflow_fit_weight")!=0 or config["overflow_policy"]!="compare_only":
        raise ValueError("R2 workflow cannot fit overflow or rescale maps")
    from darsia.fluidflower.calibration import validate_calibration_plan
    from darsia.fluidflower.project import FluidFlowerProject
    checked=validate_calibration_plan(FluidFlowerProject.load(binding["project_root"]),config)
    if not checked["valid"]:raise ValueError("Existing R2 workflow plan no longer validates")
    return binding,plan


def fit_stage(args,binding,index):
    require_workflow_plan(args)
    selected=output.read_json(args.job / "selection.json")["selection"]
    if selected.get("freeze_allowed") is not True:raise ValueError("Pre-registered selection requires stop; do not fit/freeze for held-out work")
    rows=read_extraction(args.job,binding);validate_full_extraction(rows,index)
    fit=engine().fit_round2_candidate(rows,selected["chosen_candidate"])
    unchanged_or_new(args.job / "calibration/shared_fit.json",dict(fit,campaign_binding_sha256=binding))
    return fit


def freeze_stage(args,binding):
    workflow,_=require_workflow_plan(args)
    choice=output.read_json(args.job / "selection.json")["selection"]
    if choice.get("freeze_allowed") is not True:raise ValueError("Selection stop must precede freeze")
    fit_path=args.job / "calibration/shared_fit.json";fit=output.read_json(fit_path)
    if fit.get("campaign_binding_sha256")!=binding:raise ValueError("Fit belongs to another R2 campaign")
    if fit.get("parameters",{}).get("candidate")!=choice["chosen_candidate"]:raise ValueError("Freeze candidate differs from selection")
    # No inferred proof from optimizer success: bind the explicit calibration-only
    # exact arithmetic audit before permitting a final-check image.
    audit_path=args.job / "calibration/exact_validation/validation.json"
    audit=output.read_json(audit_path)
    if audit.get("complete") is not True or audit.get("heldout_pixels_read") is not False or audit.get("summary_precision_bound_verified") is not True:
        raise ValueError("Calibration-only full-resolution precision audit required before R2 freeze")
    if audit.get("parameters")!=fit["parameters"] or audit.get("fit",{}).get("sha256")!=saved.sha_file(fit_path):
        raise ValueError("Exact audit does not bind the selected common parameters")
    r1=output.read_json(args.round1_freeze)
    inventory={str(Path(name)):sha for name,sha in r1["calibration_files"].items() if Path(name).is_file() and str(Path(name)).startswith(str(Path(r1["path_folder"])))}
    for asset in (args.input_plan,args.round1_freeze,args.job / "campaign_binding.json",args.job / "selection.json",fit_path,audit_path,Path(workflow["calibration_plan"]["path"])):
        inventory[str(asset.resolve())]=saved.sha_file(asset)
    frozen=dict(contract=output.FREEZE_CONTRACT,frozen=True,frozen_utc=datetime.now(timezone.utc).isoformat(),
        calibration_runs=sorted(output.CALIBRATION),heldout_runs=sorted(output.HELDOUT),descriptive_runs=sorted(output.DESCRIPTIVE),excluded_runs=sorted(output.EXCLUDED),
        calibration_files=inventory,parameters=fit["parameters"],anchors_by_run=fit["anchors_by_run"],path_folder=r1["path_folder"],
        round1_colour_path_source=source.asset(args.round1_freeze),spec_sha256=args.spec_sha256,input_plan=source.asset(args.input_plan),
        campaign_binding_sha256=binding,map_mass_rescaling=False,overflow_in_objective=False,default_changed=False,
        bottom_mass_allowed=False,conditional_validation_scope=output.SCOPE_CAVEAT)
    folder=args.job / "calibration/freeze";folder.mkdir(parents=True,exist_ok=False)
    new_json(folder / "frozen_calibration.json",frozen)
    return str(folder / "frozen_calibration.json")


def scientific_metadata(metadata,frame):
    return dict(run=metadata["run"],frame_index=metadata["frame_index"],elapsed_h=metadata["elapsed_h"],
        source_sha256=metadata["source_sha256"],phase_sha256=metadata["labels"]["sha256"],pixel_area_sha256=metadata["area_map"]["sha256"],
        pressure_bar=frame.pressure_bar,temperature_c=frame.temperature_c,recipe=frame.recipe,
        light_group=output.light_group(frame.run_id,supplied=frame.provenance.get("light_epoch")),
        dose_source=frame.provenance.get("dose_source"),dose_qa=frame.provenance.get("dose_qa"),
        nominal_setpoint_mass_g=frame.provenance.get("nominal_setpoint_mass_g"),input_provenance=frame.provenance,
        phase_caveat=saved.PHASE_CAVEAT,run_caveat=saved.CAVEATS.get(frame.run_id,""))


def produce_run(task):
    run,job,binding,freeze_path=task;freeze=output.FrozenEvidence.verify(freeze_path)
    freeze.check_frame_run(run);_SESSION["reader"]=inputs.VerifiedAssetReader()
    rows=rows_for_run(run);anchor_row=output.nearest(rows,2.25)
    frame,response=prepare(anchor_row,freeze)
    anchor=engine().anchor_from_response(response)
    del frame,response;gc.collect()
    started=time.perf_counter();count=0;cached_count=0
    for metadata in rows:
        folder=Path(job) / "production" / run / f"frame_{metadata['frame_index']:03d}"
        if (folder / "receipt.json").is_file():
            receipt=output.read_json(folder / "receipt.json")
            if (receipt.get("complete") is not True or receipt.get("calibration_freeze_sha256")!=freeze.sha256
                    or receipt.get("source_sha256")!=metadata["source_sha256"]
                    or receipt.get("phase_sha256")!=metadata["labels"]["sha256"]
                    or receipt.get("pixel_area_sha256")!=metadata["area_map"]["sha256"]
                    or receipt.get("elapsed_h")!=metadata["elapsed_h"]):
                raise ValueError("Saved map source/freeze differs; do not overwrite/rebind")
            cached_count+=1;continue
        frame,response=prepare(metadata,freeze)
        result=engine().evaluate_round2_frame(response,freeze.record["parameters"],anchor,fields=True)
        receipt=output.save_frame(folder,result,scientific_metadata(metadata,frame),freeze)
        count+=1;del frame,response,result;gc.collect()
        print(json.dumps(dict(stage="produce",run=run,frame_index=metadata["frame_index"],committed=True)),flush=True)
    return dict(run=run,frame_count=count+cached_count,cached_frames=cached_count,wall_seconds=time.perf_counter()-started)


def production_rows(job):
    return [output.read_json(path) for path in sorted((Path(job) / "production").glob("*/frame_*/receipt.json"))]


def check_stage(args,index):
    freeze=output.FrozenEvidence.verify(args.job / "calibration/freeze/frozen_calibration.json")
    rows=production_rows(args.job)
    expected={(r["run"],r["frame_index"]) for r in index["frames"] if r["run"] in output.HELDOUT|output.DESCRIPTIVE}
    actual={(r["run"],r["frame_index"]) for r in rows}
    if actual!=expected:raise ValueError("Final check requires every12 held-out run and descriptiveAC38 frame")
    check=output.score_predictions([r for r in rows if r["run"] in output.HELDOUT],final=True)
    unchanged_or_new(args.job / "final_check.json",dict(check,freeze=source.asset(freeze.path)))
    return check


def parallel(tasks,fn,args):
    records=[];failures=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
        initargs=(str(args.input_plan),str(args.round1_freeze),args.backend,
            source.asset(args.l1_reference) if args.l1_reference else None,str(args.job / "campaign_binding.json"))) as pool:
        futures={pool.submit(fn,task):task for task in tasks}
        for future in as_completed(futures):
            task=futures[future]
            try:record=future.result();records.append(record);print(json.dumps(dict(completed=record)),flush=True)
            except Exception as exc:
                failure=dict(task=list(task),error=repr(exc));failures.append(failure);print(json.dumps(dict(failure=failure)),flush=True)
    record=dict(contract=CONTRACT,stage=args.stage,completed=records,failures=failures,workers=args.workers,threads_per_worker=1,
        wall_seconds=time.perf_counter()-started,created_utc=datetime.now(timezone.utc).isoformat())
    new_json(args.job / "stage_receipts" / f"{args.stage}_{uuid.uuid4().hex}.json",record)
    return record


def campaign_binding(args):
    modules=bound_modules()
    binding=dict(contract=CONTRACT,input_plan=source.asset(args.input_plan),round1_freeze=source.asset(args.round1_freeze),
        spec_sha256=args.spec_sha256,driver_sha256=saved.sha_file(__file__),
        code_sha256={m.__name__:saved.sha_file(m.__file__) for m in modules},
        source_adapter_sha256=saved.sha_file(source.__file__),map_rescaling=False,overflow_fit=False,
        candidate_gas_fixed=True,candidates=list(supported_candidates()),selection_reads_heldout=False)
    binding["l1_reference"]=source.asset(args.l1_reference) if args.l1_reference else None
    approved=args.job / "approved_spec_fixed_gas_v2.md"
    if saved.sha_file(approved)!=args.spec_sha256:raise ValueError("Archived active specification differs")
    binding["approved_spec"]=source.asset(approved)
    unchanged_or_new(args.job / "campaign_binding.json",binding)
    return output.digest(binding)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--input-plan",type=Path,required=True)
    parser.add_argument("--round1-freeze",type=Path,required=True);parser.add_argument("--spec-sha256",required=True)
    parser.add_argument("--workflow-binding",type=Path,required=True)
    parser.add_argument("--l1-reference",type=Path,help="Optional source-bound protocol proxy for the separate 20mm sensitivity")
    parser.add_argument("--stage",choices=("extract","loro","select","fit","freeze","final-maps","check","produce","report"),required=True)
    parser.add_argument("--workers",type=int,default=12);parser.add_argument("--backend",choices=("auto","cpu","gpu"),default="auto")
    parser.add_argument("--runs",nargs="*");args=parser.parse_args(argv)
    if not 1<=args.workers<=24:parser.error("Use1–24 bounded workers and one internal thread")
    if len(args.spec_sha256)!=64:parser.error("Complete approved spec SHA required")
    plan=output.read_json(args.input_plan);inputs.validate_split(plan["split"]);index=inputs.metadata_index(plan)
    require_workflow_plan(args);binding=campaign_binding(args)
    selected=set(args.runs or ())
    if args.stage in {"extract","loro","select","fit","freeze"} and selected-output.CALIBRATION:raise ValueError("No held-out/descriptive pixels before freeze")
    if args.stage=="extract":
        record=parallel([(r,str(args.job),binding) for r in sorted(selected or output.CALIBRATION)],extract_run,args)
    elif args.stage=="loro":
        rows=read_extraction(args.job,binding);validate_full_extraction(rows,index)
        record=parallel([(c,r,str(args.job),binding) for c in supported_candidates() for r in sorted(selected or output.CALIBRATION)],fold_task,args)
    elif args.stage=="select":record=score_stage(args,binding,index)
    elif args.stage=="fit":record=fit_stage(args,binding,index)
    elif args.stage=="freeze":record=dict(freeze=freeze_stage(args,binding))
    elif args.stage in {"final-maps","produce"}:
        freeze_path=args.job / "calibration/freeze/frozen_calibration.json";output.FrozenEvidence.verify(freeze_path)
        if args.stage=="produce" and output.read_json(args.job / "final_check.json").get("full_production_allowed") is not True:
            raise ValueError("Final check failed/unassessable: stop, no39 production")
        allowed=output.HELDOUT|output.DESCRIPTIVE if args.stage=="final-maps" else output.CALIBRATION|output.HELDOUT|output.DESCRIPTIVE
        if selected-allowed:raise ValueError("Wrong cohort for map stage")
        record=parallel([(r,str(args.job),binding,str(freeze_path)) for r in sorted(selected or allowed)],produce_run,args)
    elif args.stage=="check":record=check_stage(args,index)
    else:
        rows=production_rows(args.job);folder=args.job / f"report_{uuid.uuid4().hex[:8]}"
        record=dict(report=str(output.write_tables(folder,rows,stage="saved-scientific-output")),full_production_complete=len(rows)==782)
    print(json.dumps(saved._jsonable(record),allow_nan=False),flush=True)
    return 2 if record.get("failures") or record.get("outcome","").startswith("stop") or record.get("outcome")=="fail" else 0


if __name__=="__main__":
    raise SystemExit(main())
