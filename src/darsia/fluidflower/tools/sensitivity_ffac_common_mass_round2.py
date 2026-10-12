"""Post-check R2 sensitivity fits over frozen, source-bound reduced statistics.

This companion never chooses a model, rewrites a phase map or reopens a failed
check. Extra final-check statistics require that exact check to have passed.
Each changed calibration uses fit-run data only and is saved before predictions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import gc
import json
import uuid

import numpy as np

from darsia.fluidflower.tools import run_ffac_common_mass_round2 as driver
from darsia.fluidflower import phase_ffac_common_mass_round2 as native
from darsia.fluidflower import phase_ffac_common_mass_round2_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_round2_production as out

CONTRACT="ffac-common-mass-round2-postcheck-sensitivity-v1"
VARIANTS=("gas_sg1","beta_zero","include_ac38","include_quarter_hour","nominal_dose","l1_aqueous","l1_saturated_upper")
_CONTEXT={}


def parameter_freeze(path,value):
    """Resume a saved fit without moving its freeze time or rebinding bytes."""
    path=Path(path)
    if path.exists():
        saved=out.read_json(path)
        if {k:v for k,v in saved.items() if k!="frozen_utc"}!=driver.saved._jsonable(value):
            raise ValueError("Previously frozen sensitivity parameters differ")
        return saved
    record=dict(value,frozen_utc=datetime.now(timezone.utc).isoformat())
    driver.new_json(path,record);return record


def context(job,destination):
    job=Path(job);destination=Path(destination)
    campaign=out.read_json(job / "campaign_binding.json")
    freeze=out.FrozenEvidence.verify(job / "calibration/freeze/frozen_calibration.json")
    check_path=job / "final_check.json";check=out.read_json(check_path)
    if check.get("final") is not True or check.get("full_production_allowed") is not True or check.get("freeze",{}).get("sha256")!=freeze.sha256:
        raise ValueError("Sensitivities require the chosen frozen final check to pass first")
    driver.verify_worker_code(job / "campaign_binding.json")
    plan=driver.source.read_bound_json(campaign["input_plan"])
    value=dict(contract=CONTRACT,campaign_sha256=out.digest(campaign),freeze=driver.source.asset(freeze.path),
        final_check=driver.source.asset(check_path),helper_sha256=driver.saved.sha_file(__file__),
        selection_or_main_outputs_changed=False,source_math=campaign["code_sha256"])
    driver.unchanged_or_new(destination / "binding.json",value)
    return dict(job=job,destination=destination,campaign=campaign,freeze=freeze,check=value["final_check"],
        plan=plan,binding=out.digest(value),dose=driver.source.read_bound_json(plan["dose_inventory"]))


def init(job,destination,backend):
    value=context(job,destination);campaign=value["campaign"]
    driver.init_worker(campaign["input_plan"]["path"],campaign["round1_freeze"]["path"],backend,
        campaign.get("l1_reference"),str(Path(job) / "campaign_binding.json"))
    _CONTEXT.update(value)


def cache_run(run):
    if run not in out.HELDOUT|out.DESCRIPTIVE:raise ValueError("Only frozen out-of-fit frames need a post-check cache")
    c=_CONTEXT;driver._SESSION["reader"]=inputs.VerifiedAssetReader();committed=0
    for metadata in driver.rows_for_run(run):
        folder=c["destination"] / "cache" / run / f"frame_{metadata['frame_index']:03d}"
        if driver.resume(folder,c["binding"]):continue
        frame,response=driver.prepare(metadata,c["freeze"])
        summary=native.summarize_round2_frame(response,no_outflow_evidence=driver.no_outflow_proxy(frame,metadata),
            light_epoch=frame.provenance["light_epoch"],max_depth_m=c["plan"]["physical_limits"]["max_depth_m"],
            max_porosity=c["plan"]["physical_limits"]["max_porosity"],backend=driver._SESSION["backend"],
            descriptive_sensitivity=run in out.DESCRIPTIVE,postcheck_sensitivity=c["check"])
        driver.commit(folder,dict(contract=CONTRACT,campaign_binding_sha256=c["binding"],summary=summary,
            summary_sha256=out.digest(summary),original_final_check=driver.source.asset(c["job"] / "final_check.json"),
            source_binding=frame_binding(metadata),
            scientific_pixels_after_final_check=True,selection_input=False))
        committed+=1;del frame,response,summary;gc.collect()
        print(json.dumps(dict(stage="postcheck-cache",run=run,frame_index=metadata["frame_index"],committed=True)),flush=True)
    return dict(run=run,committed_frames=committed)


def frame_binding(metadata):
    return {key:metadata.get(key) for key in ("run","frame_index","elapsed_h","source_sha256","labels","area_map","receipt")}


def validate_inventory(rows,index,runs,*,scope):
    expected={(r["run"],r["frame_index"]):r for r in index["frames"] if r["run"] in runs}
    actual=[(r["run"],r["frame_index"]) for r in rows]
    if len(set(actual))!=len(actual) or set(actual)!=set(expected):
        raise ValueError(f"{scope} statistics must cover every exact Studio frame once")
    for row in rows:
        metadata=expected[(row["run"],row["frame_index"])]
        if row["elapsed_h"]!=metadata["elapsed_h"] or row.get("source_binding",{}).get("source_receipt")!=metadata["receipt"]:
            raise ValueError(f"{scope} source/camera-time binding differs")
    return expected


def all_rows(c,*,require_extra=False):
    index=inputs.metadata_index(c["plan"])
    calibration=driver.read_extraction(c["job"],out.digest(c["campaign"]))
    expected=validate_inventory(calibration,index,out.CALIBRATION,scope="Calibration26")
    for key,metadata in expected.items():
        value=driver.resume(c["job"] / "work/extraction" / key[0] / f"frame_{key[1]:03d}",out.digest(c["campaign"]))
        if value.get("source_binding")!=frame_binding(metadata):
            raise ValueError("Calibration statistics belong to another source/phase/area")
    extra=[]
    for path in sorted((c["destination"] / "cache").glob("*/frame_*/receipt.json")):
        value=driver.resume(path.parent,c["binding"]);row=value["summary"]
        if row["run"] not in out.HELDOUT|out.DESCRIPTIVE:raise ValueError("Unexpected post-check cache run")
        metadata=next((r for r in index["frames"] if (r["run"],r["frame_index"])==(row["run"],row["frame_index"])),None)
        if metadata is None or value.get("source_binding")!=frame_binding(metadata):
            raise ValueError("Post-check statistics belong to another source/phase/area")
        extra.append(row)
    if require_extra:validate_inventory(extra,index,out.HELDOUT|out.DESCRIPTIVE,scope="Post-check13")
    return calibration,extra


def nominal_rows(rows,dose_inventory):
    by_run={r["run"]:r for r in dose_inventory["runs"]}
    output=[]
    for row in rows:
        record=by_run[row["run"]]
        dose=float(inputs.dose_at(record,row["elapsed_h"],nominal=True));qa=record.get("qa")
        output.append(dict(row,injected_mass_g=dose,nominal_injected_mass_g=dose,
            sensitivity_dose=dict(policy="nominal setpoint counterfactual",qa=qa,
                original_measured_or_nominal_dose_g=row["injected_mass_g"])))
    return output


def options(variant):
    if variant not in VARIANTS:raise ValueError("Unknown predeclared sensitivity")
    return {"gas_sg1":variant=="gas_sg1","no_dilution":variant=="beta_zero",
        "include_ac38":variant=="include_ac38","include_quarter_hour":variant=="include_quarter_hour"}


def evaluation(row,parameters,anchor,variant):
    if variant=="l1_saturated_upper" and row["elapsed_h"]>=4:
        base=native.evaluate_round2_summary(row,parameters,anchor)
        l1=row["l1_sensitivity"]
        if not (l1.get("verified") or (l1.get("source_bound") and l1.get("status")=="protocol_proxy_unverified" and l1.get("source_sha256"))):
            raise ValueError("Saturated L1 upper diagnostic requires its original bound proxy")
        mass=base["total_mass_g"]-l1["gas_mass_g"]-l1["gas_water_mass_g"]+l1["aqueous_equilibrium_mass_g"]
        gas=base["gas_mass_g"]-l1["gas_mass_g"]
        upper=max(0.,base["missing_capacity_upper_g"]+l1["missing_gas_to_aqueous_capacity_delta_g"])
        return out.mass_row(dict(base,total_mass_g=mass,gas_mass_g=gas,aqueous_mass_g=mass-gas,
            gas_water_mass_g=base["gas_water_mass_g"]-l1["gas_water_mass_g"],gas_mass_share=gas/mass if mass else 0.,
            missing_capacity_upper_g=upper,provisional_l1_location=not l1.get("verified"),
            sensitivity_convention="Separate upper assumption: reclassified L1 water at local solubility, undiluted"))
    result=native.evaluate_round2_summary(row,parameters,anchor,gas_sg1=variant=="gas_sg1",
        port_as_aqueous=variant=="l1_aqueous",allow_provisional_l1_sensitivity=variant=="l1_aqueous")
    return out.mass_row(result)


def sensitivity_task(task):
    variant,leftout=task;c=_CONTEXT
    folder=c["destination"] / "fits" / variant / (leftout or "all26")
    cached=driver.resume(folder,c["binding"])
    if cached:return dict(variant=variant,leftout=leftout,cached=True)
    calibration,extra=all_rows(c,require_extra=leftout is None or variant=="include_ac38")
    chosen=c["freeze"].record["parameters"]["candidate"]
    rows=calibration+[r for r in extra if r["run"]=="ac38"] if variant=="include_ac38" else calibration
    if variant=="nominal_dose":rows=nominal_rows(rows,c["dose"])
    if variant.startswith("l1_"):
        if leftout:
            main=driver.resume(c["job"] / "work/loro" / chosen / leftout,out.digest(c["campaign"]))["fit"]
            fit=main
        else:fit=out.read_json(c["job"] / "calibration/shared_fit.json")
    else:fit=native.fit_round2_candidate(rows,chosen,excluded_run=leftout,**options(variant))
    # Persist the common sensitivity parameters before any frozen out-of-fit
    # prediction. A sensitivity never replaces the chosen main freeze.
    parameter_freeze(folder.with_name(folder.name+".parameters.json"),dict(parameters=fit["parameters"],
        calibration_runs=fit.get("calibration_runs"),source_rows_sha256=out.digest(rows),
        heldout_used_in_fit=False,
        main_calibration_unchanged=True,variant=variant))
    predictions=[]
    if leftout:
        targets=[r for r in rows if r["run"]==leftout];anchors=fit["anchors_by_run"]
    else:
        targets=[r for r in extra if r["run"] in out.HELDOUT]
        if variant=="nominal_dose":targets=nominal_rows(targets,c["dose"])
        anchors=native.anchors_by_run(targets)
    for row in targets:
        result=evaluation(row,fit["parameters"],anchors[row["run"]],variant)
        predictions.append(dict(result,variant=variant,leftout_run=leftout,
            recipe=row.get("recipe"),dose_provenance=row.get("source_binding",{}).get("dose_qa"),
            evaluation_scope="calibration26-LORO" if leftout else "frozen calibration-only sensitivity on final-check runs",
            conditional_validation_scope=out.SCOPE_CAVEAT,main_outputs_changed=False,
            sensitivity_uses_final_check_for_choice=False))
    driver.commit(folder,dict(contract=CONTRACT,campaign_binding_sha256=c["binding"],variant=variant,leftout_run=leftout,
        fit=fit,predictions=predictions,main_outputs_changed=False,selection_changed=False))
    return dict(variant=variant,leftout=leftout,frames=len(predictions),cached=False)


def transfer_fit(rows,candidate,fit_runs):
    """Same native two-stage solver restricted explicitly to one light group."""
    stages,decisions=native.select_fit_frames(rows);anchors=native.anchors_by_run(rows)
    stages={stage:[r for r in values if r["run"] in fit_runs] for stage,values in stages.items()}
    first,second=stages["stage1"],stages["stage2"]
    if not first:raise ValueError("No eligible source-bound transfer fit frames")
    weights=native._weights(first)
    def residual1(vector):
        params=native._decode_stage1(candidate,vector)
        return weights*np.array([native.evaluate_round2_summary(r,params,anchors[r["run"]])["signed_interval_residual"] for r in first])
    if candidate=="N3":starts=[np.array([np.log(g),b]) for g,b in ((1.,0.),(20.,0.),(1.,1.),(5.,.5))];bounds=([-700.,0.],[700.,1.])
    elif candidate=="J":
        starts=[np.array([np.log(e),a,np.log(g)]) for e,a,g in ((1.,1.,1.),(.5,.8,.5),(2.,1.,2.),(4.,.5,1.))]
        bounds=([-700.,np.finfo(float).eps,-700.],[700.,1.,700.])
    else:starts=[np.array([np.log(g)]) for g in (1.,20.)];bounds=([-700.],[700.])
    fit1,audit1=native._optimize_multistart(residual1,starts,bounds);params=native._decode_stage1(candidate,fit1.x)
    audit2=dict(assessable=False,beta_fitted=False)
    if second:
        weights2=native._weights(second)
        def residual2(vector):
            p=dict(params,beta=float(vector[0]))
            return weights2*np.array([native.evaluate_round2_summary(r,p,anchors[r["run"]])["signed_interval_residual"] for r in second])
        fit2,audit2=native._optimize_multistart(residual2,[np.array([0.]),np.array([1.]),np.array([3.])],([0.],[np.inf]))
        params["beta"]=float(fit2.x[0])
    return dict(parameters=params,anchors_by_run=anchors,fit_runs=sorted(fit_runs),
        optimizer_audit=dict(stage1=audit1,stage2=audit2),frame_decisions=decisions,
        source_rows_sha256=out.digest(rows),selection_changed=False,heldout_used=False,
        solver_policy="same common native starts/bounds/weights; explicit fit-group filter, no proxy/evidence rewritten")


def transfer(c):
    calibration,_=all_rows(c);chosen=c["freeze"].record["parameters"]["candidate"]
    output=[]
    for group in ("rig_light","new_lamps"):
        fit_runs={r["run"] for r in calibration if out.light_group(r["run"],supplied=r["light_epoch"])==group}
        folder=c["destination"] / "transfer" / group
        if driver.resume(folder,c["binding"]):continue
        fit=transfer_fit(calibration,chosen,fit_runs)
        parameter_freeze(folder.with_name(folder.name+".parameters.json"),fit)
        predictions=[dict(out.mass_row(native.evaluate_round2_summary(r,fit["parameters"],fit["anchors_by_run"][r["run"]])),
            fit_light_group=group,predict_light_group=out.light_group(r["run"],supplied=r["light_epoch"]),
            conditional_validation_scope=out.SCOPE_CAVEAT) for r in calibration if r["run"] not in fit_runs]
        driver.commit(folder,dict(contract=CONTRACT,campaign_binding_sha256=c["binding"],fit=fit,predictions=predictions))
        output.append(dict(fit_group=group,frames=len(predictions)))
    return output


def report(c):
    predictions=[];scores=[];transfer_rows=[]
    for variant in VARIANTS:
        rows=[]
        for path in sorted((c["destination"] / "fits" / variant).glob("*/receipt.json")):
            value=driver.resume(path.parent,c["binding"]);predictions.extend(value["predictions"])
            if value["leftout_run"]:rows.extend(value["predictions"])
        if {r["run"] for r in rows}==out.CALIBRATION:
            scores.append(dict(out.score_predictions(rows),variant=variant,used_for_selection=False))
    for path in sorted((c["destination"] / "transfer").glob("*/receipt.json")):
        transfer_rows.extend(driver.resume(path.parent,c["binding"])["predictions"])
    folder=c["destination"] / ("report_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    driver.saved.write_csv_new(folder / "sensitivity_predictions.csv",predictions)
    driver.saved.write_csv_new(folder / "sensitivity_loro_scores_not_selection.csv",scores)
    driver.saved.write_csv_new(folder / "light_group_transfer.csv",transfer_rows)
    driver.new_json(folder / "report.json",dict(contract=CONTRACT,sensitivities=list(VARIANTS),
        binding=driver.source.asset(c["destination"] / "binding.json"),freeze=driver.source.asset(c["freeze"].path),
        final_check=c["check"],
        saved_tables={path.name:driver.source.asset(path) for path in folder.glob("*.csv")},
        available_prediction_count=len(predictions),conditional_validation_scope=out.SCOPE_CAVEAT,
        objective_only_control="Requires source-exact old R1 gas-response histograms and finite U; separate calibration-only control, never approximated by fixed gas",
        missing_controls=["objective-only R1 pair refit not performed by this tool"],
        bottom_or_unquantified_mass_added_to_main=False,selection_changed=False,main_maps_changed=False))
    return dict(report=str(folder),prediction_count=len(predictions))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--stage",choices=("cache","loro","heldout","transfer","report"),required=True)
    parser.add_argument("--workers",type=int,default=12);parser.add_argument("--backend",choices=("auto","cpu","gpu"),default="auto")
    parser.add_argument("--variants",nargs="*",choices=VARIANTS);args=parser.parse_args(argv)
    if not 1<=args.workers<=24:parser.error("Use 1–24 workers, one internal thread")
    c=context(args.job,args.destination)
    if args.stage in {"transfer","report"}:
        _CONTEXT.update(c);result=transfer(c) if args.stage=="transfer" else report(c)
    else:
        variants=args.variants or VARIANTS
        if args.stage=="cache":tasks=sorted(out.HELDOUT|out.DESCRIPTIVE);function=cache_run
        else:
            tasks=[(v,r) for v in variants for r in sorted(out.CALIBRATION)] if args.stage=="loro" else [(v,None) for v in variants]
            function=sensitivity_task
        result=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(args.job),str(args.destination),args.backend)) as pool:
            for future in as_completed([pool.submit(function,t) for t in tasks]):
                value=future.result();result.append(value);print(json.dumps(value),flush=True)
    print(json.dumps(driver.saved._jsonable(result),allow_nan=False),flush=True)


if __name__=="__main__":main()
