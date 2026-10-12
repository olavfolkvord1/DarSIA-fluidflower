"""Read saved R2 results, make reports and append pending existing-workflow review.

No RAW, mass executor, fit, phase edit or approval runs here. A failed selection
or check is a diagnostic report, never a complete concentration registration.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import shutil
import uuid

from darsia.fluidflower import phase_ffac_common_mass_round2_production as out
from darsia.fluidflower import phase_ffac_common_mass_inputs as inputs
from darsia.fluidflower import phase_ffac_common_mass_review as viewer
from darsia.fluidflower import phase_ffac_common_mass_production as files

CONTRACT="ffac-common-mass-round2-saved-report-v1"


def inventory(job,input_plan,freeze=None):
    index=inputs.read_bound_json(input_plan["source_metadata"])
    expected={(r["run"],r["frame_index"]):r for r in index["frames"] if r["run"] in out.CALIBRATION|out.HELDOUT|out.DESCRIPTIVE}
    rows=[];seen=set()
    for path in sorted((Path(job) / "production").glob("*/frame_*/receipt.json")):
        row=out.read_json(path);key=row["run"],row["frame_index"]
        source=expected.get(key)
        if source is None or key in seen or row.get("complete") is not True:raise ValueError("Unexpected/duplicate/partial saved production frame")
        if freeze is None:raise ValueError("Saved map without a verified R2 freeze")
        if (row.get("calibration_freeze_sha256")!=freeze.sha256 or row.get("source_sha256")!=source["source_sha256"]
                or row.get("phase_sha256")!=source["labels"]["sha256"] or row.get("pixel_area_sha256")!=source["area_map"]["sha256"]
                or row.get("elapsed_h")!=source["elapsed_h"]):raise ValueError("Saved source/time/phase/area differs from frozen report inputs")
        if datetime.fromisoformat(row["created_utc"])<datetime.fromisoformat(freeze.record["frozen_utc"]):
            raise ValueError("Scientific result predates its R2 freeze")
        if row.get("bottom_mass_allowed") is not False or row.get("no_map_mass_rescaling") is not True:
            raise ValueError("Saved result violates the bottom block/no-rescaling contract")
        row=dict(row,frame_folder=str(path.parent),receipt_asset=inputs.asset(path))
        for name in ("c_aq_g_l_display","s_g_display"):
            descriptor=row["assets"][name]
            if files.sha_file(path.parent / descriptor["path"])!=descriptor["sha256"]:
                raise ValueError("Saved concentration preview changed")
        rows.append(row);seen.add(key)
    return index,expected,rows,seen==set(expected)


def era(run):
    n=int(run[2:]);return "AC17-38" if n<=38 else "AC39-49" if n<=49 else "AC50-61"


def enrich_saved_rows(rows,plan):
    """Join frozen recipe/atmosphere metadata without reading image pixels."""
    base=inputs.read_bound_json(plan["base_input_plan"])
    atmospheres=inputs.read_bound_json(base["atmosphere"])
    recipes=inputs.read_bound_json(base["recipes"])
    by_atmosphere={r["run"]:r for r in atmospheres["runs"]}
    by_recipe={r["run"]:r for r in recipes["runs"]}
    output=[]
    for row in rows:
        atmosphere=inputs.atmosphere_at(by_atmosphere[row["run"]],row["elapsed_h"])
        provenance=row.get("source_binding",row.get("native_result",{}).get("source_binding",{}))
        output.append(dict(row,**atmosphere,recipe=by_recipe[row["run"]],
            dose_source=provenance.get("dose_source"),dose_qa=provenance.get("dose_qa"),
            measured_mfc_source=provenance.get("measured_mfc_source"),
            dose_nominal_fallback_reasons=provenance.get("dose_nominal_fallback_reasons",[]),
            nominal_setpoint_mass_g=provenance.get("nominal_setpoint_mass_g"),
            historical_R1_dose_source=provenance.get("dose_target_source"),
            historical_R1_protocol_dose_g=provenance.get("nominal_protocol_injected_mass_g"),
            atmosphere_source=base["atmosphere"],recipe_source=base["recipes"],
            light_epoch_source=plan["light_epochs"],dose_inventory_source=plan["dose_inventory"],
            scientific_metadata_recalculated=False,
            metadata_join="Exact saved camera time; original source-bound atmosphere interpolation, no scientific image or fit"))
    return output


def descriptive_rows(rows):
    perframe=[];late=[];baseline=[]
    for run in sorted({r["run"] for r in rows}):
        timeline=[r for r in rows if r["run"]==run]
        finaldose=max(r["injected_mass_g"] for r in timeline)
        for row in timeline:
            perframe.append(dict(row,era=era(run),gas_mass_g=row.get("gas_mass_g"),gas_mass_share=row.get("gas_mass_share")))
        before=[r for r in timeline if r["elapsed_h"]<=0]
        for row in before:
            baseline.append(dict(run=run,frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],
                main_mass_g=row["main_mass_g"],gas_mass_g=row.get("gas_mass_g"),gas_mass_share=row.get("gas_mass_share"),
                final_dose_g=finaldose,fraction_of_final_dose=row["main_mass_g"]/finaldose if finaldose else None,
                pass_1_percent=(row["main_mass_g"]<=.01*finaldose) if finaldose else None,
                zero_mass_imposed=False,conditional_validation_scope=out.SCOPE_CAVEAT))
        for kind,target in (("nearest48h",48.),("last",max(r["elapsed_h"] for r in timeline))):
            row=out.nearest(timeline,target)
            late.append(dict(row,selection=kind,target_h=target,era=era(run)))
    eras=[]
    for cohort,runs in (("calibration",out.CALIBRATION),("heldout",out.HELDOUT),("descriptive",out.DESCRIPTIVE)):
        for target in (1.,2.25,4.,8.,12.,24.):
            picked=[out.nearest([r for r in rows if r["run"]==run],target) for run in sorted(runs) if any(r["run"]==run for r in rows)]
            for group in ("AC17-38","AC39-49","AC50-61"):
                values=[r for r in picked if era(r["run"])==group and r.get("ratio") is not None]
                if not values:continue
                eras.append(dict(cohort=cohort,era=group,target_h=target,run_count=len(values),
                    median_ratio=statistics.median(r["ratio"] for r in values),
                    median_gas_mass_g=statistics.median(r["gas_mass_g"] for r in values),
                    median_gas_mass_share=statistics.median(r["gas_mass_share"] for r in values),
                    conditional_validation_scope=out.SCOPE_CAVEAT))
    return perframe,baseline,late,eras


def overflow_rows(rows,descriptor):
    data=inputs.read_bound_json(descriptor)
    if data.get("fit_usage")!="never" or data.get("water_displacement_not_CO2_outlet") is not True:
        raise ValueError("Overflow must remain a gas-inventory comparison, never a dose objective or CO2-outlet measurement")
    index={(r["run"],r["frame_index"]):r for r in data["frames"]}
    output=[]
    for row in rows:
        archived=index.get((row["run"],row["frame_index"]))
        observation=archived.get("overflow") if archived else None
        if not observation:continue
        if archived.get("source_sha256")!=row["source_sha256"] or archived.get("camera_time_h")!=row["elapsed_h"]:
            raise ValueError("Overflow observation belongs to another exact source/camera frame")
        density=row.get("overflow_gas_density_g_l",row.get("native_result",{}).get("overflow_gas_density_g_l"))
        volume=observation["gas_in_rig_volume_ml"]
        reference_mass=volume/1000*density if density is not None else None
        output.append(dict(run=row["run"],frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],
            source_sha256=row["source_sha256"],gas_mass_g=row["gas_mass_g"],gas_mass_share=row["gas_mass_share"],
            compare_only_gas_inventory_ml=volume,native_atmospheric_density_g_l=density,
            inventory_equivalent_mass_g=reference_mass,gas_mass_minus_inventory_equivalent_g=row["gas_mass_g"]-reference_mass if reference_mass is not None else None,
            measurement_kind=observation["measurement_kind"],measured_vs_imputed=observation["measured_vs_imputed"],
            overflow_source_sha256=observation["source_sha256"],overflow_assumptions=observation["assumptions"],
            archived_observation=observation,
            target_volume_zero_censored=observation.get("target_volume_zero_censored"),
            uncensored_weight_balance_volume_ml=observation.get("uncensored_weight_balance_volume_ml"),
            native_density_assumption="Native density at atmospheric/top-row p,T; local gas hydrostatic variation and blocked regions are not silently corrected",
            comparison_support_caveat="Main gas excludes blocked/unknown regions; overflow is whole-rig inventory, so this is descriptive, not a like-for-like completeness proof",
            used_in_fit=False,map_rescaled=False,CO2_outlet_measurement=False,
            conditional_validation_scope=out.SCOPE_CAVEAT))
    return output


def missing_support_rows(rows):
    """Capacity/excluded-region bounds remain separate from main production mass."""
    output=[]
    for row in rows:
        parts=row.get("missing_capacity_parts",row.get("native_result",{}).get("missing_capacity_parts",{}))
        excluded=row.get("excluded_mass",row.get("native_result",{}).get("excluded_mass",{}))
        for name,part in sorted(parts.items()):
            estimate=excluded.get(name,{})
            dose=row["injected_mass_g"];capacity=part["capacity_upper_g"]
            output.append(dict(run=row["run"],frame_index=row["frame_index"],elapsed_h=row["elapsed_h"],region=name,
                main_mass_g=row["main_mass_g"],gas_mass_g=row["gas_mass_g"],gas_mass_share=row["gas_mass_share"],
                region_capacity_upper_g=capacity,region_U_over_d=capacity/dose if dose>0 else None,
                entire_missing_capacity_upper_g=row["missing_capacity_upper_g"],
                main_plus_all_capacity_ratio=(row["main_mass_g"]+row["missing_capacity_upper_g"])/dose if dose>0 else None,
                excluded_region_calculated_mass_g=estimate.get("total_mass_g"),
                excluded_region_calculated_gas_mass_g=estimate.get("gas_mass_g"),
                phase_pixels=part.get("phase_pixels"),unknown_geometry_phase_pixels=part.get("unknown_geometry_phase_pixels"),
                capacity_is_not_mass=True,excluded_mass_added_to_main=False,bottom_mass_allowed=False,
                conditional_validation_scope=out.SCOPE_CAVEAT))
    return output


def ingest_sensitivity_report(folder,path,*,binding,freeze,check_path):
    """Copy immutable saved diagnostic tables after checking their exact parent."""
    descriptor=inputs.asset(path);record=inputs.read_bound_json(descriptor)
    sensitivity_binding=inputs.read_bound_json(record["binding"])
    if (record.get("contract")!="ffac-common-mass-round2-postcheck-sensitivity-v1"
            or sensitivity_binding.get("campaign_sha256")!=out.digest(binding)
            or record.get("freeze",{}).get("sha256")!=freeze.sha256
            or record.get("final_check",{}).get("sha256")!=inputs.asset(check_path)["sha256"]
            or record.get("selection_changed") is not False or record.get("main_maps_changed") is not False):
        raise ValueError("Sensitivity report belongs to another campaign/freeze/check or changes selection")
    target=folder / "postcheck_sensitivities";target.mkdir(exist_ok=False)
    for name,asset in record["saved_tables"].items():
        if Path(name).name!=name or Path(name).suffix!=".csv":raise ValueError("Unsafe sensitivity report table name")
        source=Path(asset["path"])
        if inputs.sha256(source)!=asset["sha256"]:raise ValueError("Saved sensitivity table changed")
        shutil.copyfile(source,target / name)
    files.write_new_json(target / "source_report.json",dict(source=descriptor,record=record,source_math_recomputed=False))
    return dict(source=descriptor,table_count=len(record["saved_tables"]),missing_controls=record.get("missing_controls",[]))


def ingest_legacy_report(folder,path,*,binding,freeze,check_path):
    descriptor=inputs.asset(path);record=inputs.read_bound_json(descriptor)
    parent=inputs.read_bound_json(record["binding"])
    if (record.get("contract")!="ffac-round2-legacy-objective-only-early-control-v1"
            or parent.get("campaign_sha256")!=out.digest(binding)
            or parent.get("round2_freeze",{}).get("sha256")!=freeze.sha256
            or parent.get("final_check",{}).get("sha256")!=inputs.asset(check_path)["sha256"]
            or record.get("main_round2_changed") is not False or record.get("calibration_run_count")!=27):
        raise ValueError("Legacy objective diagnostic belongs to another campaign or changes the main calibration")
    target=folder / "legacy_objective_early_control";target.mkdir(exist_ok=False)
    for name,asset in record["saved_tables"].items():
        if Path(name).name!=name or Path(name).suffix!=".csv":raise ValueError("Unsafe legacy report table name")
        source=Path(asset["path"])
        if inputs.sha256(source)!=asset["sha256"]:raise ValueError("Legacy objective table changed")
        shutil.copyfile(source,target / name)
    files.write_new_json(target / "source_report.json",dict(source=descriptor,record=record,source_math_recomputed=False))
    return dict(source=descriptor,scope=record["scope"],table_count=len(record["saved_tables"]),
        conservation_or_heldout_claim=False,original_frozen_baseline_is_in_fit=True)


def ingest_metadata_diagnostic(folder,path,*,binding,campaign_path):
    descriptor=inputs.asset(path);record=inputs.read_bound_json(descriptor)
    if (record.get("complete") is not True or record.get("scientific_pixels_read") is not False
            or record.get("fit_computed") is not False or record.get("selection_used") is not False
            or record.get("input_plan",{}).get("sha256")!=binding["input_plan"]["sha256"]
            or record.get("source_campaign_binding",{}).get("sha256")!=inputs.asset(campaign_path)["sha256"]):
        raise ValueError("Metadata diagnostic does not match this immutable input/campaign or was used scientifically")
    target=folder / "metadata_diagnostics" / Path(path).parent.name;target.mkdir(parents=True,exist_ok=False)
    for name,asset in record["products"].items():
        if Path(name).name!=name:raise ValueError("Unsafe metadata diagnostic product name")
        source=Path(asset["path"])
        if inputs.sha256(source)!=asset["sha256"]:raise ValueError("Saved metadata diagnostic product changed")
        shutil.copyfile(source,target / name)
    files.write_new_json(target / "source_receipt.json",dict(source=descriptor,record=record,
        used_for_objective=False,science_recomputed=False))
    return dict(source=descriptor,products_copied=len(record["products"]),summary=record.get("summary"))


def render_timeline(path,rows,*,context="frozen common R2 result"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    timeline=sorted(rows,key=lambda r:r["elapsed_h"])
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for field,label in (("main_mass_g","Main total"),("aqueous_mass_g","Main aqueous"),("gas_mass_g","Main gas"),("injected_mass_g","Measured/flagged nominal dose")):
        axes[0].plot([r["elapsed_h"] for r in timeline],[r[field] for r in timeline],"o-",ms=3,label=label)
    axes[0].fill_between([r["elapsed_h"] for r in timeline],[r["main_mass_g"] for r in timeline],
        [r["main_mass_g"]+r["missing_capacity_upper_g"] for r in timeline],alpha=.15,label="Finite physical M..M+U capacity (not confidence)")
    axes[1].plot([r["elapsed_h"] for r in timeline],[r.get("ratio") for r in timeline],"o-",ms=3,label="M / dose")
    axes[1].plot([r["elapsed_h"] for r in timeline],[r.get("delta") for r in timeline],"o-",ms=3,label="Two-sided interval distance")
    axes[1].axhline(1,color="grey",ls="--",lw=.8)
    axes[0].set_ylabel("CO2 mass (g)");axes[1].set_ylabel("Ratio / distance");axes[1].set_xlabel("Camera time (h)")
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(timeline[0]["run"].upper()+" · "+context+" · bottom excluded")
    fig.text(.04,.015,out.SCOPE_CAVEAT+" Capacity includes blocked/unknown regions; it is not added to main mass.",fontsize=7,wrap=True)
    fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(path,dpi=130);plt.close(fig)


def render_selection_scores(path,scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(11,8));positions=np.arange(len(scores));names=[r["candidate"] for r in scores]
    for ax,index in zip(axes.flat[:2],(0,1)):
        values=[r["injection"][index]["median_absolute_ratio_error"] for r in scores]
        ax.bar(positions,values,color=["#467dad" if r["injection"][index]["pass_gate"] else "#bc655c" for r in scores])
        ax.axhline(.1,color="black",ls="--",lw=.8);ax.set_title(f"Median |M/d − 1| at {scores[0]['injection'][index]['target_h']:g} h")
        ax.set_xticks(positions,names);ax.set_ylabel("Ratio error");ax.grid(axis="y",alpha=.15)
    ax=axes[1,0]
    for index,target in enumerate((1.,2.25)):
        for group,marker in (("rig_light","o"),("new_lamps","s")):
            values=[r["injection"][index]["median_ratio_by_group"][group] for r in scores]
            ax.plot(positions,values,marker+"-",label=f"{group} · {target:g} h")
    ax.axhspan(.9,1.1,color="#467dad",alpha=.1);ax.set_title("Four fixed group-median gates");ax.set_ylabel("Median M/d");ax.set_xticks(positions,names);ax.legend(fontsize=8)
    ax=axes[1,1]
    values=[r["E_post"] if r["E_post"] is not None else float("nan") for r in scores]
    ax.bar(positions,values,color="#467dad");ax.axhline(.1,color="black",ls="--",lw=.8)
    for i,r in enumerate(scores):
        if r["E_post"] is None:ax.text(i,0.,"Not\nassessable",ha="center",fontsize=8)
    ax.set_title("Largest assessable median interval distance");ax.set_ylabel("E post");ax.set_xticks(positions,names)
    fig.suptitle("Calibration26 leave-one-run-out · fixed choice gates · no held-out images")
    fig.text(.03,.015,out.SCOPE_CAVEAT+" U/d ≤ 5% and minimum run counts are preserved.",fontsize=7)
    fig.subplots_adjust(left=.09,right=.97,bottom=.12,top=.88,wspace=.30,hspace=.40)
    fig.savefig(path,dpi=130);plt.close(fig)


def register_layer(job,index,rows,expected,freeze,check_path):
    if {(r["run"],r["frame_index"]) for r in rows}!=set(expected):raise ValueError("Studio registration requires exact complete39 source inventory")
    check=out.read_json(check_path)
    if check.get("full_production_allowed") is not True or check.get("freeze",{}).get("sha256")!=freeze.sha256:
        raise ValueError("No Studio concentration registration before the frozen final check passes")
    if {r[0] for r in expected}!=out.CALIBRATION|out.HELDOUT|out.DESCRIPTIVE:
        raise ValueError("Studio concentration registration requires all39 exact runs")
    from darsia.fluidflower.phase_validation import _model_identity
    candidate=Path(index["candidate_root"]);model_id,model_sha=_model_identity(candidate)
    series_root=next(p.parent for p in candidate.parents if p.name=="results")
    path=series_root / "results/concentration_layer_registrations" / (Path(job).name+".json")
    frames=[dict(run=r["run"],frame_index=r["frame_index"],elapsed_h=r["elapsed_h"],frame_folder=r["frame_folder"],receipt=r["receipt_asset"]) for r in rows]
    files.write_new_json(path,dict(contract=viewer.ROUND2_CONTRACT,candidate_id=model_id,candidate_sha256=model_sha,
        calibration_freeze=inputs.asset(freeze.path),final_check=inputs.asset(check_path),complete=True,default=False,
        run_count=39,frame_count=len(rows),frames=frames,production_root=str(Path(job) / "production"),
        bottom_mass_allowed=False,no_per_run_mass_fit=True,map_rescaled_to_injected_mass=False,
        scientific_status="Pending human review; frozen conditional validation, fixed physical gas rule",
        conditional_validation_scope=out.SCOPE_CAVEAT))
    return path


def publish_review(folder,workflow,freeze,check_path,assets,summary):
    from darsia.fluidflower.project import FluidFlowerProject
    from darsia.fluidflower.calibration import CalibrationPlanStore
    from darsia.fluidflower.calibration_review import CalibrationResultReviewStore,default_calibration_review_config
    from darsia.fluidflower.mass_analysis import CALIBRATION_RESULT_FORMAT
    from darsia.fluidflower.mass_inputs import MASS_PARAMETER_ARTIFACT_FORMAT
    project=FluidFlowerProject.load(workflow["project_root"])
    plan=CalibrationPlanStore(project).current()
    if inputs.asset(Path(workflow["calibration_plan"]["path"]))["sha256"]!=workflow["calibration_plan"]["sha256"]:
        raise ValueError("Saved workflow plan changed")
    expectedplan=inputs.read_bound_json(workflow["calibration_plan"])
    if plan["plan_id"]!=expectedplan["plan_id"]:raise ValueError("Current existing-workflow plan changed")
    candidate=freeze.record["parameters"]["candidate"]
    model={"N1":"titration","N2":"simple_flash","N3":"titration"}.get(candidate)
    if not model:raise ValueError("Joint choice needs explicit workflow integration before native review")
    audits=[Path(name) for name in freeze.record["calibration_files"] if str(name).replace("\\","/").endswith("calibration/exact_validation/validation.json")]
    if len(audits)!=1:raise ValueError("Pending review requires the frozen calibration-only precision evidence")
    audit=out.read_json(audits[0])
    if (audit.get("complete") is not True or audit.get("summary_precision_bound_verified") is not True
            or audit.get("heldout_pixels_read") is not False or audit.get("parameters")!=freeze.record["parameters"]):
        raise ValueError("Native review cannot fabricate a passed precision/parity reference")
    parameter=folder / "mass_parameters.json"
    files.write_new_json(parameter,dict(format=MASS_PARAMETER_ARTIFACT_FORMAT,native_contract="native-round2-fixed-phase-v1",
        selected_model=model,parameter_scope="shared_across_series",shared_parameters=freeze.record["parameters"],run_parameters={},
        phase_support_source="production_labels",map_mass_rescaling=False,native_frozen_calibration=inputs.asset(freeze.path),
        final_check=inputs.asset(check_path),bottom_mass_allowed=False))
    manifest=folder / "calibration_result.json"
    files.write_new_json(manifest,dict(format=CALIBRATION_RESULT_FORMAT,calibration_plan_id=plan["plan_id"],
        objective_contract_sha256=workflow["selection_contract_sha256"],selected_model=model,parameter_artifact=inputs.asset(parameter),
        native_frozen_calibration=inputs.asset(freeze.path),final_check=inputs.asset(check_path),
        parity=dict(reference_backend="cpu-float64",passed=True,
            evidence=inputs.asset(audits[0]),method="Full physical masses use CPU float64; GPU reducers independently require session CPU parity"),
        report=str(folder),review_assets=assets,diagnostic_summary=summary,approved=False,default_changed=False,
        conditional_validation_scope=out.SCOPE_CAVEAT))
    config=default_calibration_review_config(project);config["calibration_result_manifest_path"]=str(manifest)
    config["run_reviews"]={run:dict(status="pending",reviewer="",note=out.SCOPE_CAVEAT,
        concentration_preview_path=a["concentration_preview"]["path"],mass_plot_path=a["mass_plot"]["path"]) for run,a in assets.items()}
    revision=CalibrationResultReviewStore(project).save_review(config,actor="Codex",reason="Saved R2 concentration/mass review; all runs pending, no defaults or labels changed")
    return dict(manifest=str(manifest),review_id=revision["review_id"],approved_run_count=0)


def render_complete_figures(folder,index,rows,expected):
    # Direct script execution uses the campaign snapshot on PYTHONPATH.
    from darsia.fluidflower.tools.run_ffac_common_mass_campaign import _photo_path
    return files.render_report_figures(folder,rows,
        photo_resolver=lambda r:_photo_path(Path(index["candidate_root"]),r["run"],r["frame_index"]),
        phase_resolver=lambda r:expected[(r["run"],r["frame_index"])]["labels"]["path"])


def build(job,workflow_binding,*,destination=None,register=False,publish=True,figures=True,overflow_inputs=None,sensitivity_report=None,legacy_objective_report=None):
    job=Path(job);binding=out.read_json(job / "campaign_binding.json")
    plan=inputs.read_bound_json(binding["input_plan"])
    workflow=out.read_json(workflow_binding)
    folder=Path(destination) if destination else job / ("saved_report_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    selection=out.read_json(job / "selection.json")
    files.write_new_json(folder / "selection.json",selection)
    loro=[];eligibility=[]
    for path in sorted((job / "work/loro").glob("*/*/receipt.json")):
        value=out.read_json(path)
        if value.get("campaign_binding_sha256")!=out.digest(binding):raise ValueError("LORO receipt has another campaign")
        loro.extend(value["predictions"])
        eligibility.extend(dict(row,candidate=value["candidate"],leftout_run=value["leftout_run"],
            producer_receipt=inputs.asset(path),conditional_validation_scope=out.SCOPE_CAVEAT,
            heldout_used_for_selection=False) for row in value["fit"]["frame_decisions"])
    files.write_csv_new(folder / "calibration26_loro_predictions.csv",loro)
    files.write_csv_new(folder / "candidate_scores.csv",selection["scores"])
    files.write_csv_new(folder / "fit_frame_eligibility_by_fold_stage.csv",eligibility)
    if figures and loro:
        plotfolder=folder / "selection_timelines";plotfolder.mkdir()
        for candidate in sorted({r["candidate"] for r in loro}):
            candidatefolder=plotfolder / candidate;candidatefolder.mkdir()
            for run in sorted(out.CALIBRATION):
                values=[r for r in loro if r["candidate"]==candidate and r["run"]==run]
                if values:render_timeline(candidatefolder / (run+".png"),values,
                    context=f"{candidate} calibration-only LORO; pooled colour paths fixed")
        render_selection_scores(folder / "candidate_selection_summary.png",selection["scores"])
    dose=inputs.read_bound_json(plan["dose_inventory"])
    files.write_new_json(folder / "dose_inventory.json",dose)
    files.write_new_json(folder / "source_bindings.json",dict(campaign_binding=inputs.asset(job / "campaign_binding.json"),
        input_plan=binding["input_plan"],workflow_binding=inputs.asset(workflow_binding),
        scientific_input_plan=plan,code_snapshot_expected=True,code_snapshot_is_execution_not_approval=True))
    costs=[out.read_json(path) for path in sorted((job / "stage_receipts").glob("*.json"))]
    files.write_csv_new(folder / "stage_costs.csv",costs)
    freeze_path=job / "calibration/freeze/frozen_calibration.json"
    freeze=out.FrozenEvidence.verify(freeze_path) if freeze_path.exists() else None
    index,expected,rows,complete=inventory(job,plan,freeze)
    if rows:rows=enrich_saved_rows(rows,plan)
    check_path=job / "final_check.json";check=out.read_json(check_path) if check_path.exists() else None
    if complete and (not check or check.get("full_production_allowed") is not True):raise ValueError("Full39 production exists despite a stopping final check")
    summary=dict(contract=CONTRACT,selection=selection["selection"],final_check=check,
        frozen_parameters=freeze.record["parameters"] if freeze else None,
        available_frame_count=len(rows),available_run_count=len({r["run"] for r in rows}),
        expected_frame_count=len(expected),expected_run_count=39,complete_inventory=complete,
        approved=False,default_changed=False,conditional_validation_scope=out.SCOPE_CAVEAT)
    clock_diagnostic=job / "diagnostics/dose_clock_sensitivity_v1/receipt.json"
    if clock_diagnostic.is_file():summary["dose_clock_diagnostic"]=ingest_metadata_diagnostic(folder,clock_diagnostic,
        binding=binding,campaign_path=job / "campaign_binding.json")
    if sensitivity_report:
        if freeze is None or not check or check.get("full_production_allowed") is not True:
            raise ValueError("Post-check sensitivity tables require the passed frozen check")
        summary["sensitivity_report"]=ingest_sensitivity_report(folder,sensitivity_report,binding=binding,freeze=freeze,check_path=check_path)
    if legacy_objective_report:
        if freeze is None or not check or check.get("full_production_allowed") is not True:
            raise ValueError("Legacy control tables require the passed frozen check")
        summary["legacy_objective_early_control"]=ingest_legacy_report(folder,legacy_objective_report,binding=binding,freeze=freeze,check_path=check_path)
    assets={};review=None;registration=None
    if rows:
        perframe,baseline,late,eras=descriptive_rows(rows)
        out.write_tables(folder / "saved_native_tables",perframe,stage="saved-frozen-output")
        for name,table in (("pre_injection.csv",baseline),("late_and_last.csv",late),("era_ratios.csv",eras)):
            files.write_csv_new(folder / name,table)
        files.write_csv_new(folder / "missing_support_capacity_not_added.csv",missing_support_rows(rows))
        if overflow_inputs:
            descriptor=inputs.asset(overflow_inputs)
            files.write_csv_new(folder / "overflow_compare_only.csv",overflow_rows(rows,descriptor))
            files.write_new_json(folder / "overflow_source.json",descriptor)
            files.write_new_json(folder / "overflow_archived_inputs.json",inputs.read_bound_json(descriptor))
        plots=folder / "timelines";plots.mkdir()
        for run in sorted({r["run"] for r in rows}):
            runrows=[r for r in rows if r["run"]==run];path=plots / (run+".png");render_timeline(path,runrows)
            preview=out.nearest(runrows,2.25);descriptor=preview["assets"]["c_aq_g_l_display"]
            assets[run]=dict(mass_plot=inputs.asset(path),concentration_preview=inputs.asset(Path(preview["frame_folder"]) / descriptor["path"]))
        if publish:review=publish_review(folder,workflow,freeze,check_path,assets,summary)
        if complete and figures:render_complete_figures(folder / "figures",index,rows,expected)
        if register:registration=register_layer(job,index,rows,expected,freeze,check_path)
    elif register:raise ValueError("No scientific maps available for registration")
    files.write_new_json(folder / "summary.json",summary)
    (folder / "README.md").write_text("# FF_AC common mass round 2\n\n"
        +f"Selection: {selection['selection']['outcome']}. Frozen final check: {check.get('outcome') if check else 'not performed'}. "
        +f"Saved scientific inventory: {len(rows)} / {len(expected)} frames, {len(assets)} / 39 runs. No missing map or sensitivity is claimed.\n\n"
        +out.SCOPE_CAVEAT+"\n\nGas uses fixed per-facies 1−Swi; remaining pore water in gas pixels is at local solubility without gain or dilution. "
        +"Bottom strips and below-photo fill stay outside main mass; finite U is a physical capacity, not fitted missing mass or confidence. "
        +"No map is rescaled to the injected dose. Overflow is gas inventory comparison only, not a measured CO2 outlet.\n\n"
        +"The dose inventory retains units/clock QA and nominal fallbacks, AC38 reconstructed clock/borrowed atmosphere/missing recipe, AC32 recipe fallback, "
        +"AC41's AC42 clear-water reference and AC45 transport. Input physical geometry is source-bound but remains provisional at boundaries. "
        +"All reviews remain pending; neither phase models, labels, sessions nor default candidates are changed.\n",encoding="utf-8")
    files.write_new_json(folder / "report_receipt.json",dict(summary,review=review,registration=str(registration) if registration else None,
        files={str(p.relative_to(folder)):inputs.asset(p) for p in folder.rglob("*") if p.is_file()},
        scientific_executor_used=False,RAW_read=False,report_tool=inputs.asset(__file__)))
    return dict(report=str(folder),review=review,registered=registration is not None,**summary)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--workflow-binding",type=Path,required=True)
    parser.add_argument("--destination",type=Path);parser.add_argument("--register",action="store_true")
    parser.add_argument("--overflow-inputs",type=Path)
    parser.add_argument("--sensitivity-report",type=Path)
    parser.add_argument("--legacy-objective-report",type=Path)
    parser.add_argument("--no-publish",action="store_true");parser.add_argument("--no-figures",action="store_true")
    args=parser.parse_args(argv)
    print(json.dumps(build(args.job,args.workflow_binding,destination=args.destination,register=args.register,
        publish=not args.no_publish,figures=not args.no_figures,overflow_inputs=args.overflow_inputs,
        sensitivity_report=args.sensitivity_report,legacy_objective_report=args.legacy_objective_report),allow_nan=False))


if __name__=="__main__":main()
