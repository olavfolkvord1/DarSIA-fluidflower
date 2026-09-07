from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

from darsia.fluidflower.calibration import (
    CALIBRATION_EVALUATION_MANIFEST_FORMAT,
    CALIBRATION_PLAN_FORMAT,
    CALIBRATION_REVISION_FORMAT,
    OBJECTIVE_CONTRACT_VERSION,
    CalibrationPlanStore,
    calibration_objective_contract,
    default_calibration_plan,
    validate_calibration_plan,
)
from darsia.fluidflower.phase_interpretation import (
    PhaseInterpretationRevisionStore,
    default_phase_interpretation_config,
    validate_phase_interpretation_config,
)
from darsia.fluidflower.project import FluidFlowerProject


def _project_with_phase_revision(tmp_path: Path) -> tuple[FluidFlowerProject, dict]:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Calibration contract",
        series_id="calibration-contract",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    review_file = review_dir / "review.json"
    review_file.write_text('{"reviewer":"Reviewer"}', encoding="utf-8")
    campaign_path = tmp_path / "campaign_manifest.json"
    campaign_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "campaign_id": "calibration-campaign",
                "dataset": "ffac",
                "jobs": [
                    {
                        "job_id": "ac17-physics",
                        "run": "ac17",
                        "role": "physics",
                        "variant": "physics-v1",
                        "family": "hierarchical-phase",
                        "candidate_id": "candidate-ac17-physics",
                        "quality_flags": {
                            "dataset": "ffac",
                            "hierarchical_phase_partition_active": True,
                        },
                    }
                ],
                "design": {
                    "interpretation_scope": "visible-2d-slice-only",
                    "mass_calibration_active": False,
                    "depth_used": False,
                    "deployment_allowed": False,
                },
                "review": {"output_dir": str(review_dir)},
            }
        ),
        encoding="utf-8",
    )
    phase_config = default_phase_interpretation_config()
    phase_config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review_dir),
            "selected_candidates": {"ac17": "candidate-ac17-physics"},
            "run_reviews": {
                "ac17": {
                    "status": "approved",
                    "candidate_id": "candidate-ac17-physics",
                    "reviewer": "Reviewer",
                }
            },
            "manual_review_files": [str(review_file)],
        }
    )
    phase_report = validate_phase_interpretation_config(phase_config)
    assert phase_report["valid"]
    phase_revision = PhaseInterpretationRevisionStore(project).save_revision(
        phase_config,
        actor="Reviewer",
        reason="Ready for calibration planning",
        validation=phase_report,
    )
    return project, phase_revision


def test_calibration_plan_freezes_objective_and_phase_provenance(
    tmp_path: Path,
) -> None:
    project, phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    config["execution_profile"].update(
        {"primary_backend": "cuda-float32", "workers": 2}
    )
    config["quality_profile"]["precision"]["compute_dtype"] = "float32"
    report = validate_calibration_plan(project, config)

    assert report["valid"]
    assert report["summary"]["warning_count"] == 0
    assert report["objective_contract"]["format"] == OBJECTIVE_CONTRACT_VERSION
    assert report["objective_contract"]["score_direction"] == "minimize"
    assert all(
        row["normalization"] and row["direction"] == "minimize_penalty"
        for row in report["objective_contract"]["objectives"]
    )

    revision = CalibrationPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Freeze one reproducible Optuna contract",
        validation=report,
    )
    assert revision["format"] == CALIBRATION_REVISION_FORMAT
    assert revision["configuration"]["format"] == CALIBRATION_PLAN_FORMAT
    assert revision["upstream"]["phase_revision_id"] == phase_revision["revision_id"]
    assert (
        revision["upstream"]["phase_revision_sha256"]
        == phase_revision["content_sha256"]
    )
    assert revision["upstream"]["phase_upstream"] == phase_revision["upstream"]
    assert (
        revision["selection_contract_sha256"]
        == revision["objective_contract"]["content_sha256"]
    )
    assert revision["execution"] == {
        "requested": False,
        "started": False,
        "study_id": None,
    }
    assert not revision["scientific_scope"]["trains_facies_model"]
    assert not revision["scientific_scope"]["trains_phase_model"]
    assert not revision["scientific_scope"]["contains_mass_results"]
    assert (
        project.root
        / "results"
        / "calibration"
        / "plans"
        / revision["plan_id"]
        / "plan.json"
    ).is_file()


def test_calibration_plan_rejects_execution_and_contract_drift(tmp_path: Path) -> None:
    project, _phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    config["execution_requested"] = True
    config["optimizer"]["selection_uses_identical_objective"] = False
    config["execution_profile"]["reference_backend"] = "cuda-float32"
    config["chemistry"]["initial_ph_bounds"] = [9.0, 8.0]
    report = validate_calibration_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert {
        "execution_not_supported",
        "selection_objective_mismatch",
        "invalid_reference_backend",
        "invalid_ph_bounds",
    } <= codes


def test_objective_digest_changes_with_scientific_inputs(tmp_path: Path) -> None:
    project, _phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    baseline = calibration_objective_contract(config)
    changed = deepcopy(config)
    changed["chemistry"]["temperature_c"] = 24.0

    assert (
        calibration_objective_contract(changed)["content_sha256"]
        != baseline["content_sha256"]
    )


def test_evaluation_manifest_makes_complete_contract_execution_ready(
    tmp_path: Path,
) -> None:
    project, phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    data_path = tmp_path / "evaluation-data.npz"
    data_path.write_bytes(b"immutable prepared evaluator data")
    parameter_keys = [
        str(row["key"])
        for row in config["parameter_space"]["parameters"]
        if row["enabled"]
    ]
    objective_keys = [
        str(row["key"]) for row in config["objectives"] if row["enabled"]
    ]
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": CALIBRATION_EVALUATION_MANIFEST_FORMAT,
                "evaluator": "test-evaluator-v1",
                "phase_revision_sha256": phase_revision["content_sha256"],
                "supported_objectives": objective_keys,
                "supported_parameters": parameter_keys,
                "runs": [
                    {
                        "run_id": "ac17",
                        "data_path": data_path.name,
                        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config["evaluation_manifest_path"] = str(manifest_path)

    report = validate_calibration_plan(project, config)

    assert report["valid"]
    assert report["summary"]["execution_ready"]
    assert report["summary"]["evaluation_run_count"] == 1
    assert report["objective_contract"]["evaluation_manifest"]["sha256"]


def test_evaluator_must_support_every_enabled_parameter(tmp_path: Path) -> None:
    project, phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    diffusion = next(
        row
        for row in config["parameter_space"]["parameters"]
        if row["key"] == "diffusion_coefficient_m2_s"
    )
    diffusion["enabled"] = True
    data_path = tmp_path / "evaluation-data.npz"
    data_path.write_bytes(b"immutable prepared evaluator data")
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": CALIBRATION_EVALUATION_MANIFEST_FORMAT,
                "evaluator": "test-evaluator-v1",
                "phase_revision_sha256": phase_revision["content_sha256"],
                "supported_objectives": [
                    str(row["key"])
                    for row in config["objectives"]
                    if row["enabled"]
                ],
                "supported_parameters": [
                    str(row["key"])
                    for row in config["parameter_space"]["parameters"]
                    if row["enabled"]
                    and row["key"] != "diffusion_coefficient_m2_s"
                ],
                "runs": [
                    {
                        "run_id": "ac17",
                        "data_path": data_path.name,
                        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config["evaluation_manifest_path"] = str(manifest_path)

    report = validate_calibration_plan(project, config)
    issues = {str(row["code"]): row for row in report["issues"]}

    assert not report["valid"]
    assert not report["summary"]["execution_ready"]
    assert issues["parameter_not_evaluable"]["details"]["unsupported"] == [
        "diffusion_coefficient_m2_s"
    ]


def test_save_revalidates_phase_pointer_at_commit_time(tmp_path: Path) -> None:
    project, phase_revision = _project_with_phase_revision(tmp_path)
    config = default_calibration_plan(project)
    preview_report = validate_calibration_plan(project, config)
    phase_config = deepcopy(phase_revision["configuration"])
    PhaseInterpretationRevisionStore(project).save_revision(
        phase_config,
        actor="Second reviewer",
        reason="Replace current phase interpretation",
    )

    try:
        CalibrationPlanStore(project).save_plan(
            config,
            actor="Reviewer",
            reason="This plan is now stale",
            validation=preview_report,
        )
    except ValueError as exc:
        assert "blocking validation errors" in str(exc)
    else:
        raise AssertionError("A plan bound to a stale phase revision was saved")
