"""Save the new pre-registered R2 procedure in the existing isolated workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from darsia.fluidflower.calibration import CalibrationPlanStore, CALIBRATION_EVALUATION_MANIFEST_FORMAT, validate_calibration_plan
from darsia.fluidflower.phase_interpretation import PhaseInterpretationRevisionStore, default_phase_interpretation_config
from darsia.fluidflower.phase_ffac_common_mass_inputs import asset, read_bound_json
from darsia.fluidflower.phase_ffac_common_mass_round2_production import CALIBRATION, HELDOUT, DESCRIPTIVE, EXCLUDED
from darsia.fluidflower.project import FluidFlowerProject
from .prepare_ffac_common_mass_plan import native_plan, write_new


def round2_plan(project, *, phase_binding, input_plan, spec_sha256, evaluation_manifest,
                round1_freeze, swi_binding, physical_limits_binding):
    config = native_plan(project, phase_binding=phase_binding, input_plan=input_plan,
        spec_sha256=spec_sha256, evaluation_manifest=evaluation_manifest)
    config["optimizer"].update(engine="native-round2-two-stage-common", trial_budget_per_model=1)
    config["native_fixed_phase_contract"] = dict(contract="native-round2-fixed-phase-v1",
        locked_phase_binding=phase_binding, input_plan=input_plan, approved_spec_sha256=spec_sha256,
        round1_freeze=round1_freeze, swi_binding=swi_binding, physical_limits_binding=physical_limits_binding,
        calibration_runs=sorted(CALIBRATION), heldout_runs=sorted(HELDOUT), descriptive_runs=sorted(DESCRIPTIVE), excluded_runs=sorted(EXCLUDED),
        parameter_scope="shared_across_series",
        candidate_parameters={"N1":["gain","beta"], "N2":["gain","beta"], "N3":["blend","gain","beta"], "J":["endpoint","amplitude","gamma","beta"]},
        N3_policy="include only if native workflow blend exists without new model code",
        gas_rule="fixed 1-Swi; remaining water at local solubility; neither diluted nor fitted",
        stages={"injection_h":[.497,2.247], "post_injection_h":[2.747,24.], "stage2_fitted":["beta"], "exclude_quarter_hour":True},
        selection_rule="round2-spec-section3", capacity_policy="finite two-sided dose interval; measured max depth/porosity only bound missing geometry; fit U/d<=0.05",
        dose_rule="measured MFC camera-time integral after units/clock QA; nominal fallback flagged for all runs",
        dilution_rule="reuse concentration_joint.dilution_factor unchanged; aqueous support only after fixed end anchor",
        map_rescaling_allowed=False, overflow_fit_weight=0, bottom_strip_mass_allowed=False, below_photo_fill_mass_allowed=False,
        freeze_before_heldout=True, heldout_final_check_before_production=True,
        no_outflow_frame_policy="same fixed open-top optical contact proxy, valid at fit-stage times; no measured CO2 outlet",
        main_geometry_imputation_allowed=False, manual_phase_review_allowed=False,
        representation="same source-bound full-resolution float32 RAW scientific input as R1",
        pooled_paths_policy="reuse frozen R1 paths exactly; no path refit, no held-out colour-path inputs")
    return config


def prepare(job, input_plan, *, spec_path=None):
    job = Path(job)
    bootstrap = json.loads((job / "job.json").read_text(encoding="utf-8-sig"))
    plan_binding = asset(input_plan)
    data = read_bound_json(plan_binding)
    approved = Path(spec_path) if spec_path else job / "approved_spec_fixed_gas_v2.md"
    candidate = Path(bootstrap["source_phase"])
    phase_binding = asset(candidate / "rig_camera_candidate_receipt.json")
    project = FluidFlowerProject.create(job / "calibration_project", name="FF_AC common mass round 2 · P1 final 3",
        series_id="ff_ac", template_id="ff_ac", rig_type="medium", overflow_mode="scale_images",
        created_by="Olav-authorized R2 spec, 9 Oct 2026")
    production = sorted(CALIBRATION | HELDOUT | DESCRIPTIVE)
    project.set_runs([dict(run_id=run) for run in production], updated_by="Codex", reason="Fixed 39-run source cohort; AC38 descriptive")
    campaign = project.root / "phase_campaign.json"
    write_new(campaign, dict(schema=2, campaign_id="ffac-round2-fixed-p1-final3", dataset="ffac", locked_phase_binding=phase_binding,
        jobs=[dict(run=run, role="locked-production-labels", candidate_id=candidate.name, numerical_products=dict(source_receipt=phase_binding)) for run in production],
        design=dict(interpretation_scope="visible_registered_camera_plane_2d", mass_calibration_active=False, depth_used=False, deployment_allowed=False)))
    phase_config = default_phase_interpretation_config()
    phase_config.update(campaign_manifest_path=str(campaign), selected_candidates={run:candidate.name for run in production},
        run_reviews={run:dict(status="approved",candidate_id=candidate.name,reviewer="Olav",source="R2 spec locks existing P1 final 3 phase support") for run in production})
    phase = PhaseInterpretationRevisionStore(project).save_revision(phase_config,actor="Olav (source-bound approval)",reason="Existing phase support only; parent candidate unchanged")
    base = read_bound_json(data["base_input_plan"])
    source = read_bound_json(base["source_metadata"])
    evaluation = project.root / "heldout_evaluation_manifest.json"
    write_new(evaluation,dict(schema=1,format=CALIBRATION_EVALUATION_MANIFEST_FORMAT,evaluator="native-round2-fixed-phase-v1",
        phase_revision_sha256=phase["content_sha256"],supported_objectives=["injected_mass_balance","overflow_phase_constraint","temporal_physics","spatial_artifacts"],supported_parameters=[],
        runs=[dict(run_id=run,data_path=str(candidate / "source_alias/native_frames" / run / "manifest.json"),
            data_sha256=asset(candidate / "source_alias/native_frames" / run / "manifest.json")["sha256"]) for run in sorted(HELDOUT)],
        access="Only source metadata until R2 freeze; no held-out science for selection"))
    limits = data["physical_limits"]
    # The whole finite-capacity policy and its measured source tables are one
    # small immutable certificate, rather than a machine-specific literal.
    limits_path=project.root / "physical_limits_certificate.json"
    for key in ("depth_source","facies_source"):
        if asset(limits[key]["path"])["sha256"]!=limits[key]["sha256"]:
            raise ValueError("A measured physical source changed before plan save")
    write_new(limits_path,limits)
    swi_binding=limits["facies_source"]
    limits_binding=asset(limits_path)
    config = round2_plan(project,phase_binding=phase_binding,input_plan=plan_binding,spec_sha256=asset(approved)["sha256"],
        evaluation_manifest=evaluation,round1_freeze=asset(bootstrap["round1_freeze"]),swi_binding=swi_binding,physical_limits_binding=limits_binding)
    validation = validate_calibration_plan(project,config)
    write_new(job / "calibration_plan_validation.json",validation)
    if not validation["valid"]:
        raise ValueError("New R2 existing-workflow plan has blocking errors")
    revision = CalibrationPlanStore(project).save_plan(config,actor="Codex",reason="Pre-registered R2 candidate LORO, fixed gas, two-stage native mass procedure; no defaults changed")
    descriptor=asset(project.root / "results/calibration/plans" / revision["plan_id"] / "plan.json")
    result=dict(project_root=str(project.root),calibration_plan=descriptor,selection_contract_sha256=revision["selection_contract_sha256"],
        input_plan=plan_binding,existing_workflow="Mass calibration from phase model",default_changed=False,phase_parent_changed=False)
    write_new(job / "calibration_plan_binding.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--input-plan",type=Path,required=True)
    parser.add_argument("--spec-path",type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.job,args.input_plan,spec_path=args.spec_path),indent=2))


if __name__ == "__main__":
    main()
