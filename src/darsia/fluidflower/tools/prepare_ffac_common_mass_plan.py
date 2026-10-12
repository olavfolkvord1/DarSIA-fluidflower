"""Save the common native procedure in the existing CalibrationPlanStore.

The project is an isolated campaign package. It neither activates a Studio
phase candidate nor changes the source project's current calibration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from darsia.fluidflower.calibration import (
    CalibrationPlanStore, CALIBRATION_EVALUATION_MANIFEST_FORMAT,
    default_calibration_plan, validate_calibration_plan,
)
from darsia.fluidflower.phase_interpretation import (
    PhaseInterpretationRevisionStore, default_phase_interpretation_config,
)
from darsia.fluidflower.phase_ffac_common_mass import (
    CALIBRATION_RUNS, HELDOUT_RUNS, EXCLUDED_RUNS, ALL_RUNS,
)
from darsia.fluidflower.phase_ffac_common_mass_inputs import asset, read_bound_json
from darsia.fluidflower.project import FluidFlowerProject


def write_new(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def native_plan(project, *, phase_binding, input_plan, spec_sha256, evaluation_manifest):
    config = default_calibration_plan(project)
    for row in config["parameter_space"]["parameters"]:
        row.update(enabled=False, scope="shared_across_series")
    config["fixed_transfer_contract"] = dict(
        transfer_shape="locked_categorical", source="production_labels",
        c_aq_zero_labels=[0, 255], s_g_positive_labels=[2],
        phase_probabilities_are_visible_2d_support=False,
    )
    for row in config["objectives"]:
        row["enabled"] = row["key"] != "manual_phase_review"
        row["weight"] = 1. if row["key"] == "injected_mass_balance" else 0.
    config["chemistry"]["model"] = "compare_simple_flash_and_titration"
    config["overflow_policy"] = "compare_only"
    config["evaluation_manifest_path"] = str(evaluation_manifest)
    config["execution_profile"]["workers"] = 12
    config["optimizer"].update(engine="scipy-least-squares-native-shared-pair", trial_budget_per_model=1)
    config["native_fixed_phase_contract"] = dict(
        contract="native-darsia-fixed-phase-mass-v1",
        locked_phase_binding=phase_binding, input_plan=input_plan,
        approved_spec_sha256=spec_sha256,
        calibration_runs=list(CALIBRATION_RUNS), heldout_runs=list(HELDOUT_RUNS),
        excluded_runs=list(EXCLUDED_RUNS),
        shared_response_parameters=[dict(key=key, scope="shared_across_series", lower=0., upper=None,
                                        domain="positive", parameterization="log")
                                    for key in ("aqueous_scale", "gas_scale")],
        shared_scale_domain_policy="positive gains; native relative concentrations and saturation remain clipped to [0,1]",
        plateau_tie_break="first successful start [1,1] when objective-equivalent; smallest source-derived plateau gain only if every calibration prediction is identical; report nonunique parameters",
        representation="corrected linear current/clearwater minus registered baseline/clearwater",
        flash_main="TitrationFlash", flash_sensitivity="SimpleFlash",
        gas_water_rule="relative c_aq=1 (local solubility), not 1 g/L",
        measured_recipe_inputs=True, map_rescaling_allowed=False,
        dilution_rule="none; late mass growth is reported, never forced to dose",
        overflow_fit_weight=0, bottom_strip_mass_allowed=False,
        below_photo_fill_mass_allowed=False,
        freeze_before_heldout=True,
        no_outflow_frame_policy=dict(
            name="early-open-top-proxy", min_time_exclusive_h=0., max_time_h=2.25,
            reject_co2_contact_with_top_domain=True,
            reject_co2_in_excluded_strips=False,
            excluded_strip_fit_policy="physical-dose-interval-v1",
            excluded_strip_capacity="label 1: solubility; label 2: max(solubility, gas density); measured pore volume only",
            unknown_physical_capacity="unbounded upper; lower-bound constraint only; never fabricate geometry",
            evidence="INFERRED from locked surface support; no measured CO2 outlet; affected by phase/top-rule uncertainty",
        ),
    )
    return config


def prepare(batch: Path, input_plan_path: Path) -> dict:
    job = json.loads((batch / "job.json").read_text(encoding="utf-8-sig"))
    input_binding = asset(input_plan_path)
    inputs = read_bound_json(input_binding)
    candidate = Path(job["candidate"])
    phase_binding = asset(candidate / "rig_camera_candidate_receipt.json")
    production = [run for run in ALL_RUNS if run not in EXCLUDED_RUNS]
    project = FluidFlowerProject.create(
        batch / "calibration_project", name="FF_AC · common native · P1 final 3",
        series_id="ff_ac", template_id="ff_ac", rig_type="medium",
        overflow_mode="scale_images", created_by="Olav-authorized common calibration, 9 Oct 2026",
    )
    project.set_runs([dict(run_id=run) for run in production], updated_by="Codex",
                     reason="Exact pre-registered 39-run cohort; source project and labels unchanged")
    campaign_path = project.root / "phase_campaign.json"
    write_new(campaign_path, dict(
        schema=2, campaign_id="ffac-p1-final3-locked-common-mass", dataset="ffac",
        locked_phase_binding=phase_binding,
        jobs=[dict(run=run, role="locked-production-labels", candidate_id=candidate.name,
                   numerical_products=dict(source_receipt=phase_binding)) for run in production],
        design=dict(interpretation_scope="visible_registered_camera_plane_2d",
                    mass_calibration_active=False, depth_used=False, deployment_allowed=False),
    ))
    phase_config = default_phase_interpretation_config()
    phase_config.update(
        campaign_manifest_path=str(campaign_path),
        selected_candidates={run: candidate.name for run in production},
        run_reviews={run: dict(status="approved", candidate_id=candidate.name,
                              reviewer="Olav", source="User locks P1 final 3 in approved specification, 9 Oct 2026")
                     for run in production},
    )
    phase = PhaseInterpretationRevisionStore(project).save_revision(
        phase_config, actor="Olav (explicit source-bound approval)",
        reason="Approved spec locks visible phase labels for this new calibration; parent flags remain unchanged",
    )
    evaluation_path = project.root / "heldout_evaluation_manifest.json"
    write_new(evaluation_path, dict(
        schema=1, format=CALIBRATION_EVALUATION_MANIFEST_FORMAT,
        evaluator="native-darsia-fixed-phase-mass-v1", phase_revision_sha256=phase["content_sha256"],
        supported_objectives=["injected_mass_balance", "overflow_phase_constraint", "temporal_physics", "spatial_artifacts"],
        supported_parameters=[],
        runs=[dict(run_id=run, data_path=str(candidate / "source_alias" / "native_frames" / run / "manifest.json"),
                   data_sha256=asset(candidate / "source_alias" / "native_frames" / run / "manifest.json")["sha256"])
              for run in HELDOUT_RUNS],
        access="Metadata only until calibration freeze; full scientific loader enforces the split",
    ))
    config = native_plan(project, phase_binding=phase_binding, input_plan=input_binding,
                         spec_sha256=job["spec_sha256"], evaluation_manifest=evaluation_path)
    validation = validate_calibration_plan(project, config)
    write_new(batch / "calibration_plan_validation.json", validation)
    if not validation["valid"]:
        raise ValueError("Existing calibration plan validation failed; inspect calibration_plan_validation.json")
    revision = CalibrationPlanStore(project).save_plan(
        config, actor="Codex", reason="One native shared series calibration, exactly as approved; no new phase inference",
    )
    descriptor = asset(project.root / "results" / "calibration" / "plans" / revision["plan_id"] / "plan.json")
    result = dict(project_root=str(project.root), calibration_plan=descriptor,
                  selection_contract_sha256=revision["selection_contract_sha256"],
                  input_plan=input_binding, existing_workflow="Mass calibration from phase model",
                  default_changed=False, phase_parent_changed=False)
    write_new(batch / "calibration_plan_binding.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--append-revision", action="store_true")
    args = parser.parse_args()
    if args.append_revision:
        job = json.loads((args.batch / "job.json").read_text(encoding="utf-8-sig"))
        project = FluidFlowerProject.load(args.batch / "calibration_project")
        previous = CalibrationPlanStore(project).current()
        config = native_plan(project,
            phase_binding=asset(Path(job["candidate"]) / "rig_camera_candidate_receipt.json"),
            input_plan=asset(args.input_plan), spec_sha256=job["spec_sha256"],
            evaluation_manifest=previous["configuration"]["evaluation_manifest_path"])
        revision = CalibrationPlanStore(project).save_plan(config, actor="Codex",
            reason="Measured inputs with physical dose intervals for excluded support; main mass excludes blocked strips; previous draft retained")
        result = dict(project_root=str(project.root), calibration_plan=asset(
            project.root / "results" / "calibration" / "plans" / revision["plan_id"] / "plan.json"),
            selection_contract_sha256=revision["selection_contract_sha256"],
            input_plan=asset(args.input_plan), existing_workflow="Mass calibration from phase model",
            default_changed=False, phase_parent_changed=False)
        write_new(args.batch / "calibration_plan_binding_final.json", result)
    else:
        result = prepare(args.batch, args.input_plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
