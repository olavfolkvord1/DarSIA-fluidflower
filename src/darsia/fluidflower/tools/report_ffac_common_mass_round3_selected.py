"""Report frozen selected-image R3 evidence without fitting or changing maps.

The optional three photo figures reopen exact frozen inputs only to draw the
already-defined excluded-part masks. This does not recompute phase inference,
fit concentrations or change the committed mass predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import gc
import gzip
import json
from pathlib import Path

import numpy as np

from darsia.fluidflower import phase_ffac_common_mass_inputs as assets
from darsia.fluidflower import phase_ffac_common_mass_round3 as engine
from . import run_ffac_common_mass_round3_selected as runner

CONTRACT="ffac-round3-selected-evidence-report-v1"
PART_COLOURS={"bottom_strip": "#f39c12", "below_photo_filled": "#d73027",
    "other_not_mass_eligible": "#984ea3", "unquantified_facies": "#377eb8"}


def _write(root,name,value):
    path=Path(root)/name
    assets.write_new_json(path,value)
    return assets.asset(path)


def _csv(root,name,keys,rows):
    path=Path(root)/name
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return assets.asset(path)


def describe_checks(score):
    """Use the saved literal gates; missing geometry never becomes zero error."""
    early=[]
    for row in score["check1"]:
        early.append(dict(target_h=row["target_h"],status="pass" if row["pass_gate"] else "fail",
            median_absolute_ratio_error=row["median_absolute_ratio_error"],group_medians=row["median_ratio_by_group"]))
    late={}
    for key in ("late_check","after48_check"):
        result=score[key]
        late[key]=dict(status="unassessable" if not result["assessable"] else "pass" if result["pass_gate"] else "fail",
            assessable_time_count=result["assessable_time_count"],required_assessable_times=result["required_assessable_times"],
            times=[dict(target_h=row["target_h"],status="unassessable" if not row["assessable"] else "pass" if row["pass_gate"] else "fail",
                eligible_run_count=row["eligible_run_count"],required_run_count=row["required_run_count"],
                available_run_count=row["available_run_count"],median_delta=row["median_delta"],
                eligible_runs=row["eligible_runs"]) for row in result["times"]])
    return dict(early=early,**late,all_pass=bool(score["pass_gate"]))


def fit_tables(records):
    parameters,decisions,counts,profiles=[],[],[],[]
    for name,fit in sorted(records.items()):
        for stage,audit in fit.get("grid_audits",{}).items():
            optimizer=fit.get("optimizer_audits",{}).get(stage,{})
            parameters.append(dict(fit=name,omitted_run=fit.get("heldout_fit_run"),stage=stage,
                gain=fit["parameters"]["gain"],beta=fit["parameters"]["beta"],
                stage_frames=fit.get("stages",{}).get(stage,{}).get("frames"),
                jacobian_rank=optimizer.get("jacobian_rank"),
                jacobian_singular_values=json.dumps(optimizer.get("jacobian_singular_values")),
                multistart=json.dumps(optimizer.get("multistart")),
                fitted_objective=audit["fitted_objective"],grid_minimum_objective=audit["grid_minimum_objective"],
                grid_minimum_value=audit["grid_minimum_value"],grid_stop_required=audit["stop_required"],
                numerical_comparison_tolerance=audit["numerical_comparison_tolerance"],
                stage2_eligibility_policy=fit.get("stage2_eligibility_policy"),fit_stop_reason=fit.get("reason")))
            profiles.extend(dict(fit=name,stage=stage,parameter=audit["parameter"],value=value,objective=objective,
                fitted_value=audit["fitted_value"],fitted_objective=audit["fitted_objective"])
                for value,objective in zip(audit["values"],audit["objectives"]))
        counter=Counter()
        for decision in fit.get("frame_decisions",[]):
            row=dict(decision,fit=name,omitted_run=fit.get("heldout_fit_run"))
            row["source_sha256"]=row["source_binding"]["source_sha256"]
            row["omitted_reasons"]=json.dumps(row["omitted_reasons"])
            decisions.append(row)
            counter[(row["stage"],row["selected_target_h"],bool(row["included"]))]+=1
        counts.extend(dict(fit=name,stage=stage,selected_target_h=target,included=included,image_count=count)
            for (stage,target,included),count in sorted(counter.items()))
    return parameters,decisions,counts,profiles


def _audits(value,prefix=""):
    if isinstance(value,dict):
        for key,item in value.items():
            name=prefix+"/"+key
            if key=="reduction_audit" and isinstance(item,dict):yield name,item
            elif isinstance(item,dict):yield from _audits(item,name)


def backend_report(job,reader,*,expand=False):
    complete_path=Path(job)/"extraction/complete.json"
    if not complete_path.is_file():
        return dict(status="extraction_not_complete",fit_backend="CPU float64 SciPy"),[],{}
    descriptor=assets.asset(complete_path)
    complete=reader.json(descriptor)
    results=complete.get("run_results",[])
    worker_wall=sum(float(row.get("wall_seconds",0)) for row in results)
    rows,files=[],{str(complete_path):descriptor}
    if expand:
        # Each producer receipt/statistics asset is verified once, then read
        # once. Keep only scalar audit rows, never copies of histogram arrays.
        for run in engine.FIT_RUNS:
            for path in sorted((Path(job)/"extraction"/run).glob("frame_*/receipt.json")):
                descriptor=assets.asset(path);record=reader.json(descriptor)
                if record.get("complete") is not True:raise ValueError("Backend report found an uncommitted frame")
                stats=record["native_statistics"]
                with gzip.open(reader.path(stats),"rt",encoding="utf-8") as stream:value=json.load(stream)
                for name,audit in _audits(value):
                    rows.append(dict(run=record["run"],frame_index=record["frame_index"],audit=name,**audit))
                files[str(path)]=descriptor;files[stats["path"]]=stats
                del value
    backend_counts=Counter(str(row.get("used_backend") or row.get("backend") or "unknown") for row in rows)
    return dict(status="full_scalar_audit" if expand else "producer_worker_wall_only",
        fit_backend="CPU float64 SciPy; GPU/CPU source-statistic reductions do not alter the objective",
        source_statistics_backend_counts=dict(backend_counts),
        successful_extraction_worker_wall_seconds=worker_wall,
        worker_wall_is_not_campaign_elapsed_or_CPU_seconds=True,
        campaign_wall_seconds=None,cpu_seconds=None,
        unrecorded_costs="No invented CPU-hours; concurrent worker sums are not elapsed wall time. Phase generation/display rendering are separate."),rows,files


def _grid_figures(output,records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=[]
    folder=Path(output)/"grid_profiles";folder.mkdir()
    for name,fit in sorted(records.items()):
        for stage,audit in fit.get("grid_audits",{}).items():
            figure,axis=plt.subplots(figsize=(7,3.8))
            axis.plot(audit["values"],audit["objectives"],linewidth=1)
            axis.axvline(audit["fitted_value"],color="#d95f02",label="optimizer")
            axis.axvline(audit["grid_minimum_value"],color="#377eb8",linestyle="--",label="grid minimum")
            if audit["parameter"]=="gain":axis.set_xscale("log")
            axis.set(title=f"{name}: {stage}; grid stop = {audit['stop_required']}",xlabel=audit["parameter"],ylabel="Identical weighted two-sided objective")
            axis.grid(alpha=.2);axis.legend();figure.tight_layout()
            path=folder/(name+"-"+stage+".png");figure.savefig(path,dpi=130);plt.close(figure)
            figures.append(assets.asset(path))
    return figures


def render_part_figures(job,output,*,reader=None):
    """Frozen-input descriptive masks on photos; no concentration evaluation."""
    reader=reader or assets.VerifiedAssetReader()
    job,output=Path(job),Path(output)
    freeze=job/"calibration/freeze/frozen_calibration.json"
    engine.verify_round3_freeze(freeze)
    runner.init_worker(assets.asset(job/"campaign.json"))
    candidate=Path(runner.SESSION["plan"]["selected_phase_candidate"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    from darsia.fluidflower import phase_ffac_common_mass_round2 as native
    figures,missing=[],[]
    folder=output/"excluded_part_figures";folder.mkdir()
    for run in ("ac19","ac47","ac56"):
        matches=[row for row in runner.SESSION["entries"] if row["run"]==run and row["selected_target_h"]==24.]
        if len(matches)!=1:
            missing.append(dict(run=run,target_h=24.,reason="No unique accepted selected 24 h source"));continue
        entry=matches[0]
        display_path=candidate/"display_renderings"/run/entry["source_binding"]["source_sha256"]/"receipt.json"
        if not display_path.is_file():
            missing.append(dict(run=run,target_h=24.,reason="Source-bound full-rig photo is not rendered"));continue
        display_descriptor=assets.asset(display_path);display=reader.json(display_descriptor)
        if (display.get("binding",{}).get("phase_receipt")!=entry["receipt"]
                or display.get("binding",{}).get("scientific_receipt")!=entry["scientific_receipt"]
                or display.get("source_sha256")!=entry["source_binding"]["source_sha256"]
                or display.get("elapsed_h")!=entry["elapsed_h"]):
            raise ValueError("The excluded-part figure photo belongs to another acquisition")
        loaded=runner._load_entry(entry,scope="frozen_evaluation",frozen_calibration=freeze)
        response=loaded["response"];frame=response["frame"]
        parts=engine._parts(frame,native._physical_partition(response)["missing"])
        photo=Image.open(reader.path(display["photographs"]["natural"])).convert("RGB")
        if (photo.height,photo.width)!=frame.domain.shape:raise ValueError("Figure photo and parts are not in one rig grid")
        target_width=min(1800,photo.width);size=(target_width,round(photo.height*target_width/photo.width))
        background=photo.resize(size,Image.Resampling.LANCZOS)
        figure,axis=plt.subplots(figsize=(11,6))
        axis.imshow(background)
        handles=[];counts={}
        colours=("#f39c12","#d73027","#984ea3","#377eb8")
        for index,(name,mask) in enumerate(parts.items()):
            colour=PART_COLOURS.get(name,colours[index%len(colours)])
            reduced=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(size,Image.Resampling.NEAREST))/255
            from matplotlib.colors import to_rgb
            overlay=np.empty((*reduced.shape,4),np.float32);overlay[...,:3]=to_rgb(colour);overlay[...,3]=reduced*.6
            axis.imshow(overlay);handles.append(Patch(color=colour,label=name.replace("_"," ")))
            counts[name]=int(mask.sum())
        axis.axis("off");axis.set_title(f"{run.upper()} · camera {entry['elapsed_h']:.3f} h · outside main M (descriptive only)")
        axis.legend(handles=handles,loc="lower right",fontsize=8);figure.tight_layout()
        path=folder/(run+"-24h.png");figure.savefig(path,dpi=150);plt.close(figure)
        figures.append(dict(run=run,target_h=24.,elapsed_h=entry["elapsed_h"],source_binding=entry["source_binding"],
            photograph=display["photographs"]["natural"],display_receipt=display_descriptor,phase_receipt=entry["receipt"],
            figure=assets.asset(path),part_pixels=counts,display_resizing_only=True,mass_recomputed=False,maps_changed=False))
        del loaded,response,frame,parts,photo,background
        gc.collect()
    return dict(complete=not missing,figures=figures,missing=missing,scope="frozen-source descriptive masks only")


def build(job,output,*,backend_audit=False,figures=False,reader=None):
    reader=reader or assets.VerifiedAssetReader()
    job,output=Path(job),Path(output)
    campaign_descriptor=assets.asset(job/"campaign.json");campaign=reader.json(campaign_descriptor)
    files={str(job/"campaign.json"):campaign_descriptor}
    records={}
    for path in sorted((job/"fits").glob("*.json")):
        if path.name in {"complete.json","STOP.json"}:continue
        descriptor=assets.asset(path);value=reader.json(descriptor)
        if "parameters" not in value:continue
        records[path.stem]=value;files[str(path)]=descriptor
    stop_path=job/"fits/STOP.json";check_path=job/"report/checks.json"
    stop=reader.json(assets.asset(stop_path)) if stop_path.is_file() else None
    checks=reader.json(assets.asset(check_path)) if check_path.is_file() else None
    if stop is not None:
        files[str(stop_path)]=assets.asset(stop_path)
        if checks is not None:raise ValueError("A stopped fit cannot acquire a downstream check report")
    elif checks is None:
        raise ValueError("Publish only the fixed fit-stop evidence or completed frozen checks")
    if checks is not None:
        files[str(check_path)]=assets.asset(check_path)
        if checks.get("complete") is not True or checks.get("production_automatically_allowed") is not False:
            raise ValueError("Checks must be complete without automatic production approval")
        freeze=reader.json(checks["frozen_calibration"])
        if (freeze.get("frozen") is not True or freeze.get("parameters")!=records["global"]["parameters"]
                or freeze.get("spec_sha256")!=runner.SPEC_SHA256
                or checks.get("spec_sha256")!=runner.SPEC_SHA256):
            raise ValueError("Report combines a different fit, freeze or approved specification")
        files[checks["frozen_calibration"]["path"]]=checks["frozen_calibration"]
    parameters,decisions,counts,profiles=fit_tables(records)
    backend,backend_rows,backend_files=backend_report(job,reader,expand=backend_audit)
    files.update(backend_files)
    output.mkdir(parents=True,exist_ok=False)
    tables={}
    for name,rows,keys in (
        ("fit_parameters",parameters,("fit","omitted_run","stage","gain","beta","stage_frames","jacobian_rank","jacobian_singular_values","multistart","fitted_objective","grid_minimum_objective","grid_minimum_value","grid_stop_required","numerical_comparison_tolerance","stage2_eligibility_policy","fit_stop_reason")),
        ("fit_images",decisions,("fit","omitted_run","stage","run","frame_index","selected_target_h","elapsed_h","source_sha256","included","omitted_reasons","baseline_eligibility_gain","baseline_eligibility_beta","baseline_W_over_d","fitted_beta_W_over_d")),
        ("fit_images_by_time",counts,("fit","stage","selected_target_h","included","image_count")),
        ("grid_profiles",profiles,("fit","stage","parameter","value","objective","fitted_value","fitted_objective"))):
        tables[name]=_csv(output,name+".csv",keys,rows)
    if backend_rows:
        tables["backend_reductions"]=_csv(output,"backend_reductions.csv",
            ("run","frame_index","audit","backend","input_rows","wall_seconds","gpu_fallback","gpu_status",
             "parity_checked_this_call","session_gpu_parity_verified","parity_max_absolute_difference"),backend_rows)
    grid_figures=_grid_figures(output,records)
    verdicts=None if checks is None else {name:describe_checks(checks[name]) for name in ("B0","B1")}
    part_figures=(render_part_figures(job,output,reader=reader) if figures and checks is not None else
        dict(complete=False,figures=[],missing=[dict(run=run,target_h=24.,reason="Not requested in this metadata report")
            for run in ("ac19","ac47","ac56")]))
    table_sources={}
    if checks is not None:
        for name in ("per_run_time.csv","excluded_parts.csv","excluded_distances.csv"):
            descriptor=assets.asset(job/"report"/name);reader.path(descriptor);table_sources[name]=descriptor
        # The scientific writer already preserves None as empty CSV fields;
        # do not rewrite it as a fabricated mass or finite interval.
    result=dict(contract=CONTRACT,created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,scope="fixed_fit_stop" if stop is not None else "frozen_selected_checks",
        approved_spec_sha256=runner.SPEC_SHA256,campaign=campaign_descriptor,fit_count=len(records),
        fit_stop=stop,check_decision=None if checks is None else checks["outcome"],check_verdicts=verdicts,
        production_automatically_allowed=False,manual_Olav_go_still_required=True,
        main_mass_definition_unchanged=True,undefined_same_column_geometry_is_unbounded_not_zero=True,
        estimates_are_separate_from_main_M=True,gas_branch_fitted=False,
        source_tables=table_sources,tables=tables,grid_figures=grid_figures,
        excluded_part_figures=part_figures,backend_and_cost=backend,source_files=files,
        mass_fitting_performed=False,scientific_maps_changed=False,labels_changed=False,
        AC32_AC44_caveat=True,AC38_descriptive=True,
        report_sections_complete=bool(stop is not None or part_figures["complete"]))
    _write(output,"report.json",result)
    lines=["# FF_AC round 3 — selected-image evidence",""]
    if stop is not None:
        lines.extend([f"**Stopped before freeze/held-out evaluation:** {stop['reason']}.",
            "The global/fold grid evidence is preserved; no calibration or mass production is approved.",""])
    else:
        lines.extend([f"**Decision:** {checks['outcome']}.",
            "Main M is unchanged in definition. Excluded estimates E_lo, E_hi and U_rest are reported separately.",
            "Undefined geometry remains unbounded; an unassessable time is not a passing mass-balance test.",""])
        for variant,verdict in verdicts.items():
            lines.append(f"{variant}: injection " + ", ".join(f"{row['target_h']:g} h {row['status']} ({row['median_absolute_ratio_error']:.2%})" for row in verdict['early'])+".")
            for key,label in (("late_check","4–48 h"),("after48_check",">48 h")):
                section=verdict[key]
                lines.append(f"{variant} {label}: {section['status']}; {section['assessable_time_count']}/{section['required_assessable_times']} required times assessable.")
                for row in section["times"]:
                    delta="unknown" if row["median_delta"] is None else f"{row['median_delta']:.2%}"
                    lines.append(f"- {row['target_h']:g} h: {row['status']}; {row['eligible_run_count']} eligible runs, median delta {delta}.")
            lines.append("")
        lines.extend(["AC32 and AC44 are caveat runs; AC38 remains descriptive. Fit-run scores use their source-bound omitted-run folds.",
            "No map is rescaled to injected mass; overflow did not enter fitting. No automatic production or default activation.",""])
    lines.extend([f"Global/fold records: {len(records)}. Gain/beta, multistart/rank, per-target inclusion and grid profiles are attached.",
        f"Successful extraction worker wall sum: {backend.get('successful_extraction_worker_wall_seconds','unknown')} s; this is not campaign elapsed time or CPU-hours.",
        "Optimizer: CPU float64. The recorded weighted reductions use the GPU where available with the validated CPU fallback.",
        f"Three selected-photo excluded-part figures complete: {part_figures['complete']}.",
        "Further production requires Olav's go after the fixed decision rules."])
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return assets.asset(output/"report.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",required=True)
    parser.add_argument("--destination",required=True)
    parser.add_argument("--backend-audit",action="store_true")
    parser.add_argument("--figures",action="store_true")
    args=parser.parse_args(argv)
    print(json.dumps(build(args.job,args.destination,backend_audit=args.backend_audit,figures=args.figures)))


if __name__=="__main__":main()
