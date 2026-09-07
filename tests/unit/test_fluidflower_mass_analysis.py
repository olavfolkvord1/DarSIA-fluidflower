from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.calibration import (
    CalibrationPlanStore,
    default_calibration_plan,
    validate_calibration_plan,
)
from darsia.fluidflower.calibration_review import (
    CalibrationResultReviewStore,
    default_calibration_review_config,
)
from darsia.fluidflower.image_correction import (
    CorrectionRevisionStore,
    identity_correction_config,
)
from darsia.fluidflower.mass_analysis import (
    CALIBRATION_RESULT_FORMAT,
    MASS_ANALYSIS_PLAN_FORMAT,
    MASS_ANALYSIS_REVISION_FORMAT,
    MASS_ANALYSIS_TASK_CONTRACT_FORMAT,
    MassAnalysisPlanStore,
    default_mass_analysis_plan,
    validate_mass_analysis_plan,
)
from darsia.fluidflower.mass_artifacts import (
    write_field_frame,
    write_field_index,
    write_preview_index,
)
from darsia.fluidflower.mass_inputs import (
    MASS_INPUT_MANIFEST_FORMAT,
    MASS_PARAMETER_ARTIFACT_FORMAT,
)
from darsia.fluidflower.phase_interpretation import (
    PhaseInterpretationRevisionStore,
    default_phase_interpretation_config,
    validate_phase_interpretation_config,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file
from darsia.fluidflower.wasserstein import (
    MASS_ANALYSIS_RESULT_FORMAT,
    WASSERSTEIN_PLAN_FORMAT,
    WASSERSTEIN_REVISION_FORMAT,
    WASSERSTEIN_TASK_CONTRACT_FORMAT,
    WassersteinPlanStore,
    default_wasserstein_plan,
    validate_wasserstein_plan,
)


def _project_with_calibration_plan(
    tmp_path: Path,
) -> tuple[FluidFlowerProject, dict, dict]:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Mass task contract",
        series_id="mass-task-contract",
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
                "campaign_id": "mass-campaign",
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
                    },
                    {
                        "job_id": "ac18-physics",
                        "run": "ac18",
                        "role": "physics",
                        "variant": "physics-v1",
                        "family": "hierarchical-phase",
                        "candidate_id": "candidate-ac18-physics",
                        "quality_flags": {
                            "dataset": "ffac",
                            "hierarchical_phase_partition_active": True,
                        },
                    },
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
    CorrectionRevisionStore(project).save_revision(
        identity_correction_config(),
        actor="Reviewer",
        reason="Synthetic no-correction branch",
    )
    phase_config = default_phase_interpretation_config()
    phase_config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review_dir),
            "selected_candidates": {
                "ac17": "candidate-ac17-physics",
                "ac18": "candidate-ac18-physics",
            },
            "run_reviews": {
                "ac17": {
                    "status": "approved",
                    "candidate_id": "candidate-ac17-physics",
                    "reviewer": "Reviewer",
                },
                "ac18": {
                    "status": "approved",
                    "candidate_id": "candidate-ac18-physics",
                    "reviewer": "Reviewer",
                },
            },
            "manual_review_files": [str(review_file)],
        }
    )
    phase_report = validate_phase_interpretation_config(phase_config)
    phase_revision = PhaseInterpretationRevisionStore(project).save_revision(
        phase_config,
        actor="Reviewer",
        reason="Phase selection complete",
        validation=phase_report,
    )
    calibration_config = default_calibration_plan(project)
    calibration_report = validate_calibration_plan(project, calibration_config)
    calibration_plan = CalibrationPlanStore(project).save_plan(
        calibration_config,
        actor="Reviewer",
        reason="Mass-analysis calibration contract",
        validation=calibration_report,
    )
    return project, phase_revision, calibration_plan


def _confirmed_mass_plan(project: FluidFlowerProject) -> dict:
    config = default_mass_analysis_plan(project)
    config["optical_branch"]["selection_confirmed"] = True
    return config


def _approve_upstream(project: FluidFlowerProject) -> None:
    for module in (
        "setup",
        "data_integrity",
        "overflow",
        "temporal_sync",
        "image_correction",
        "geometry_facies",
        "optical_calibration",
        "phase_interpretation",
        "calibration",
    ):
        project.set_module_status(
            module,
            "approved",
            updated_by="Reviewer",
            reason="Synthetic mass-plan readiness test",
        )


def _calibration_result(tmp_path: Path, calibration_plan: dict) -> Path:
    parameter_path = tmp_path / "calibrated_parameters.json"
    parameter_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": MASS_PARAMETER_ARTIFACT_FORMAT,
                "selected_model": "titration",
                "shared_parameters": {
                    "transfer_shape": "smoothstep",
                    "acid_low": 0.1,
                    "acid_high": 0.9,
                    "co2_onset": 0.1,
                    "co2_high": 0.7,
                    "gas_onset": 0.2,
                    "gas_high": 0.8,
                    "max_gas_saturation": 0.45,
                    "initial_pH": 8.1,
                    "aqueous_blend": 0.0,
                    "diffusion_beta": 0.0,
                },
                "run_parameters": {},
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "calibration_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": CALIBRATION_RESULT_FORMAT,
                "calibration_plan_id": calibration_plan["plan_id"],
                "objective_contract_sha256": calibration_plan[
                    "selection_contract_sha256"
                ],
                "selected_model": "titration",
                "parameter_artifact": {
                    "path": parameter_path.name,
                    "sha256": sha256_file(parameter_path),
                },
                "parity": {
                    "reference_backend": "cpu-float64",
                    "tested_backend": "cuda-float32",
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return result_path


def _approve_calibration_result(
    project: FluidFlowerProject,
    tmp_path: Path,
    result_path: Path,
) -> dict:
    preview = tmp_path / "concentration.png"
    mass_plot = tmp_path / "mass_plot.png"
    preview.write_bytes(b"concentration preview")
    mass_plot.write_bytes(b"mass plot")
    result_sha256 = sha256_file(result_path)
    config = default_calibration_review_config(project)
    config["calibration_result_manifest_path"] = str(result_path)
    config["run_reviews"] = {
        run_id: {
            "status": "approved",
            "reviewer": "Reviewer",
            "calibration_result_sha256": result_sha256,
            "concentration_preview_path": str(preview),
            "mass_plot_path": str(mass_plot),
        }
        for run_id in ("ac17", "ac18")
    }
    return CalibrationResultReviewStore(project).save_review(
        config,
        actor="Reviewer",
        reason="Synthetic concentration and mass review",
    )


def _mass_input_manifest(
    tmp_path: Path,
    phase_revision: dict,
    run_bindings: list[dict[str, str]],
) -> Path:
    root = tmp_path / "prepared-mass-inputs"
    times = [0.0, 1.0, 2.0]
    shape = (8, 12)
    grid = {
        "coordinate_system": "corrected_registered_rig_grid",
        "shape_rc": list(shape),
        "width_m": 1.2,
        "height_m": 0.8,
    }
    run_inputs = []
    for binding_index, binding in enumerate(run_bindings):
        run_id = binding["run_id"]
        run_root = root / run_id
        products = {}
        dynamic_values = {
            "optical_acidification": [],
            "phase_co2_probability": [],
            "phase_free_gas_probability": [],
        }
        for frame_index in range(len(times)):
            acid = np.zeros(shape, dtype=np.float64)
            co2 = np.zeros(shape, dtype=np.float64)
            gas = np.zeros(shape, dtype=np.float64)
            if frame_index >= 1:
                column_shift = binding_index
                acid[2:7, 2 + column_shift : 9 + column_shift] = 0.65 + (
                    0.1 * frame_index
                )
                co2[2:7, 2 + column_shift : 9 + column_shift] = 0.85
                gas[3:5, 4 + column_shift : 7 + column_shift] = (
                    0.75 if frame_index == 1 else 0.35
                )
            dynamic_values["optical_acidification"].append(acid)
            dynamic_values["phase_co2_probability"].append(co2)
            dynamic_values["phase_free_gas_probability"].append(gas)

        for product, values in dynamic_values.items():
            product_root = run_root / product
            frames = [
                write_field_frame(
                    product_root / f"{index:04d}.npz",
                    values[index],
                    run_id=run_id,
                    product=product,
                    frame_id=f"frame-{index:04d}",
                    time_hours=times[index],
                )
                for index in range(len(times))
            ]
            products[product] = write_field_index(
                product_root / "index.json",
                run_id=run_id,
                product=product,
                quantity=product,
                unit="1",
                frames=frames,
                grid=grid,
            )

        static_values = {
            "depth": (np.full(shape, 0.01, dtype=np.float64), "m"),
            "porosity": (np.full(shape, 0.36, dtype=np.float64), "1"),
            "permeability": (
                np.full(shape, 1.0e-10, dtype=np.float64),
                "m2",
            ),
        }
        for product, (values, unit) in static_values.items():
            product_root = run_root / product
            frame = write_field_frame(
                product_root / "0000.npz",
                values,
                run_id=run_id,
                product=product,
                frame_id="static-geometry",
                time_hours=0.0,
            )
            products[product] = write_field_index(
                product_root / "index.json",
                run_id=run_id,
                product=product,
                quantity=product,
                unit=unit,
                frames=[frame],
                grid=grid,
            )

        preview_root = run_root / "preview_images"
        preview_frames = []
        for index, time_hours in enumerate(times):
            image = np.zeros((*shape, 3), dtype=np.uint8)
            image[:, :, 2] = 100
            image[:, :, 1] = np.asarray(
                dynamic_values["optical_acidification"][index] * 220.0,
                dtype=np.uint8,
            )
            preview_path = preview_root / f"{index:04d}.png"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(
                str(preview_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            preview_frames.append(
                {
                    "frame_id": f"frame-{index:04d}",
                    "time_hours": time_hours,
                    "path": preview_path.name,
                    "sha256": sha256_file(preview_path),
                }
            )
        preview_reference = write_preview_index(
            preview_root / "index.json",
            run_id=run_id,
            frames=preview_frames,
        )
        run_inputs.append(
            {
                "run_id": run_id,
                "phase_candidate_id": binding["phase_candidate_id"],
                "target_times_hours": times,
                "injected_mass_g": [0.0, 0.5, 0.5],
                "geometry": {"width_m": 1.2, "height_m": 0.8},
                "thermodynamic_state": {
                    "pressure_bar": [1.01, 1.01, 1.01],
                    "temperature_c": [20.0, 20.0, 20.0],
                    "pressure_gradient_bar_per_m": [0.0, 0.0, 0.0],
                    "temperature_gradient_c_per_m": [0.0, 0.0, 0.0],
                },
                "products": products,
                "preview_images": preview_reference,
            }
        )
    manifest_path = root / "mass_input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": MASS_INPUT_MANIFEST_FORMAT,
                "upstream": {
                    "phase_revision_id": phase_revision["revision_id"],
                    "phase_revision_sha256": phase_revision["content_sha256"],
                    "image_correction_revision_id": phase_revision["upstream"][
                        "image_correction_revision_id"
                    ],
                    "image_correction_revision_sha256": phase_revision["upstream"][
                        "image_correction_revision_sha256"
                    ],
                },
                "run_inputs": run_inputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_mass_plan_is_valid_draft_without_calibration_result(tmp_path: Path) -> None:
    project, phase_revision, calibration_plan = _project_with_calibration_plan(tmp_path)
    config = _confirmed_mass_plan(project)
    report = validate_mass_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"]
    assert not report["execution_ready"]
    assert "calibration_result_missing" in codes
    assert "mass_input_manifest_missing" in codes
    assert "upstream_gates_incomplete" in codes
    assert report["summary"]["run_count"] == 2
    assert report["task_contract"]["format"] == MASS_ANALYSIS_TASK_CONTRACT_FORMAT
    assert config["optical_branch"]["compatible"]
    assert report["task_contract"]["optical_branch"] == config["optical_branch"]
    assert config["run_bindings"] == [
        {"run_id": "ac17", "phase_candidate_id": "candidate-ac17-physics"},
        {"run_id": "ac18", "phase_candidate_id": "candidate-ac18-physics"},
    ]

    revision = MassAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Prepare mass analysis while calibration result is pending",
        validation=report,
    )
    assert revision["format"] == MASS_ANALYSIS_REVISION_FORMAT
    assert revision["configuration"]["format"] == MASS_ANALYSIS_PLAN_FORMAT
    assert revision["upstream"]["calibration_plan_id"] == calibration_plan["plan_id"]
    assert revision["upstream"]["phase_revision_id"] == phase_revision["revision_id"]
    assert revision["upstream"]["optical_branch"] == config["optical_branch"]
    assert not revision["execution"]["ready"]
    assert not revision["execution"]["started"]
    assert not revision["scientific_scope"]["contains_mass_results"]
    assert revision["scientific_scope"]["raw_mass_is_authoritative"]


def test_mass_plan_requires_explicit_matching_optical_branch(tmp_path: Path) -> None:
    project, _phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = default_mass_analysis_plan(project)
    config["optical_branch"]["selection_confirmed"] = False
    report = validate_mass_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "optical_branch_not_selected" in codes

    config = _confirmed_mass_plan(project)
    config["optical_branch"]["correction_revision_sha256"] = "wrong"
    report = validate_mass_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}
    assert not report["valid"]
    assert "optical_branch_changed" in codes


def test_mass_plan_becomes_ready_with_matching_result_and_gates(
    tmp_path: Path,
) -> None:
    project, phase_revision, calibration_plan = _project_with_calibration_plan(tmp_path)
    _approve_upstream(project)
    result_path = _calibration_result(tmp_path, calibration_plan)
    _approve_calibration_result(project, tmp_path, result_path)
    config = _confirmed_mass_plan(project)
    config["calibration_result_manifest_path"] = str(result_path)
    config["mass_input_manifest_path"] = str(
        _mass_input_manifest(tmp_path, phase_revision, config["run_bindings"])
    )
    config["execution_profile"]["primary_backend"] = "cuda-float32"
    config["exports"]["numerical_dtype"] = "float32"
    config["quality_profile"]["precision"]["compute_dtype"] = "float32"
    report = validate_mass_analysis_plan(project, config)

    assert report["valid"]
    assert report["execution_ready"]
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["calibration_review_ready"]
    assert report["calibration_result"]["reference"]["sha256"] == sha256_file(
        result_path
    )
    assert report["task_contract"]["calibration_result"]["sha256"] == sha256_file(
        result_path
    )
    assert report["task_contract"]["calibration_result"]["parameter_artifact"]["sha256"]
    assert report["task_contract"]["calibration_review"]["review_id"]
    revision = MassAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Ready for a later task runner",
        validation=report,
    )
    assert revision["execution"]["ready"]
    assert revision["upstream"]["calibration_result"]["parameter_artifact"]["sha256"]
    assert revision["upstream"]["calibration_review"]["review_id"]
    assert (
        project.root
        / "results"
        / "mass_analysis"
        / "plans"
        / revision["plan_id"]
        / "plan.json"
    ).is_file()


def test_mass_plan_rejects_forced_conservation_and_dimensionless_cache(
    tmp_path: Path,
) -> None:
    project, _phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = _confirmed_mass_plan(project)
    config["physical_conversion"]["force_mass_conservation"] = True
    config["exports"]["wasserstein_cache"].update(
        {
            "enabled": True,
            "resolution_policy": "legacy_scale_factor_0.1",
            "target_cell_size_m": None,
        }
    )
    report = validate_mass_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "physical_conversion_changed" in codes
    assert "wasserstein_resolution_not_physical" in codes
    assert "wasserstein_cell_size_invalid" in codes


def test_mass_task_digest_changes_with_frame_selection(tmp_path: Path) -> None:
    project, _phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = _confirmed_mass_plan(project)
    baseline = validate_mass_analysis_plan(project, config)["task_contract"]
    changed = deepcopy(config)
    changed["image_selection"]["maximum_images_per_run"] = 120

    assert (
        validate_mass_analysis_plan(project, changed)["task_contract"]["content_sha256"]
        != baseline["content_sha256"]
    )


def test_mass_save_revalidates_current_calibration_plan(tmp_path: Path) -> None:
    project, _phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = _confirmed_mass_plan(project)
    preview_report = validate_mass_analysis_plan(project, config)
    replacement_config = default_calibration_plan(project)
    replacement_report = validate_calibration_plan(project, replacement_config)
    CalibrationPlanStore(project).save_plan(
        replacement_config,
        actor="Second reviewer",
        reason="Replace current calibration plan",
        validation=replacement_report,
    )

    try:
        MassAnalysisPlanStore(project).save_plan(
            config,
            actor="Reviewer",
            reason="This mass plan is now stale",
            validation=preview_report,
        )
    except ValueError as exc:
        assert "blocking validation errors" in str(exc)
    else:
        raise AssertionError("A mass plan bound to stale calibration was saved")


def _project_with_mass_plan(
    tmp_path: Path, *, ready: bool
) -> tuple[FluidFlowerProject, dict]:
    project, phase_revision, calibration_plan = _project_with_calibration_plan(tmp_path)
    if ready:
        _approve_upstream(project)
        result_path = _calibration_result(tmp_path, calibration_plan)
        _approve_calibration_result(project, tmp_path, result_path)
    config = _confirmed_mass_plan(project)
    if ready:
        config["calibration_result_manifest_path"] = str(result_path)
        config["mass_input_manifest_path"] = str(
            _mass_input_manifest(tmp_path, phase_revision, config["run_bindings"])
        )
    report = validate_mass_analysis_plan(project, config)
    mass_plan = MassAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic Wasserstein upstream",
        validation=report,
    )
    if ready:
        assert mass_plan["execution"]["ready"]
        project.set_module_status(
            "mass_analysis",
            "approved",
            updated_by="Reviewer",
            reason="Synthetic completed mass analysis",
        )
    return project, mass_plan


def _mass_result(tmp_path: Path, mass_plan: dict) -> Path:
    run_results = []
    for binding in mass_plan["configuration"]["run_bindings"]:
        run_id = binding["run_id"]
        products = {}
        for component in (
            "mass_total",
            "mass_aqueous",
            "mass_free_gas",
            "transport_weight",
        ):
            product_path = tmp_path / f"{run_id}-{component}-index.json"
            product_path.write_text(
                json.dumps({"run_id": run_id, "component": component}),
                encoding="utf-8",
            )
            products[component] = {
                "path": product_path.name,
                "sha256": sha256_file(product_path),
            }
        run_results.append(
            {
                "run_id": run_id,
                "complete": True,
                "target_times_hours": [0.0, 1.0, 2.0],
                "injected_mass_g": [0.0, 5.0, 5.0],
                "reference_plume_radius_m": [0.0, 0.2, 0.2],
                "domains": [
                    "whole_active_domain",
                    "analysis_box_1",
                    "analysis_box_2",
                ],
                "geometry": {
                    "width_m": 2.8,
                    "height_m": 1.5,
                    "native_cell_size_m": 0.0005,
                },
                "products": products,
            }
        )
    result_path = tmp_path / "mass_analysis_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": MASS_ANALYSIS_RESULT_FORMAT,
                "mass_analysis_plan_id": mass_plan["plan_id"],
                "mass_task_contract_sha256": mass_plan["task_contract"][
                    "content_sha256"
                ],
                "complete": True,
                "run_results": run_results,
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_wasserstein_plan_is_valid_draft_without_mass_result(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_wasserstein_plan(project)
    report = validate_wasserstein_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"]
    assert not report["execution_ready"]
    assert report["summary"]["run_count"] == 2
    assert report["summary"]["pair_count"] == 1
    assert "mass_result_missing" in codes
    assert "physical_cell_size_missing" in codes
    assert report["task_contract"]["format"] == WASSERSTEIN_TASK_CONTRACT_FORMAT

    revision = WassersteinPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Prepare synchronized pairwise analysis",
        validation=report,
    )
    assert revision["format"] == WASSERSTEIN_REVISION_FORMAT
    assert revision["configuration"]["format"] == WASSERSTEIN_PLAN_FORMAT
    assert revision["upstream"]["mass_analysis_plan_id"] == mass_plan["plan_id"]
    assert not revision["execution"]["ready"]
    assert not revision["execution"]["started"]
    assert not revision["scientific_scope"]["contains_wasserstein_results"]
    assert revision["scientific_scope"]["preserves_raw_mass_difference"]


def test_wasserstein_plan_becomes_ready_with_matching_mass_result(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    result_path = _mass_result(tmp_path, mass_plan)
    config = default_wasserstein_plan(project)
    config["mass_result_manifest_path"] = str(result_path)
    config["spatial_discretization"]["target_cell_size_m"] = 0.001
    report = validate_wasserstein_plan(project, config)

    assert report["valid"]
    assert report["execution_ready"]
    assert report["summary"]["pair_count"] == 1
    assert report["summary"]["time_count"] == 3
    assert report["summary"]["task_count"] == 9
    assert report["mass_result"]["reference"]["sha256"] == sha256_file(result_path)
    assert len(report["task_contract"]["mass_result"]["product_artifacts"]) == 8

    revision = WassersteinPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Ready for the separate CPU task runner",
        validation=report,
    )
    assert revision["execution"]["ready"]
    assert (
        project.root
        / "results"
        / "wasserstein"
        / "plans"
        / revision["plan_id"]
        / "plan.json"
    ).is_file()


def test_wasserstein_budget_freezes_a_deterministic_time_subset(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    result_path = _mass_result(tmp_path, mass_plan)
    config = default_wasserstein_plan(project)
    config["mass_result_manifest_path"] = str(result_path)
    config["spatial_discretization"]["target_cell_size_m"] = 0.001
    config["comparison"]["time_selection_budget"]["maximum_time_points"] = 2

    report = validate_wasserstein_plan(project, config)

    assert report["execution_ready"]
    assert report["summary"]["source_time_count"] == 3
    assert report["summary"]["time_count"] == 2
    assert report["summary"]["task_count"] == 6
    assert report["task_contract"]["resolved_time_selection"][
        "selected_time_indices"
    ] == [0, 2]


def test_wasserstein_rejects_dimensionless_scale_and_upsampling(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    config = default_wasserstein_plan(project)
    config["mass_result_manifest_path"] = str(_mass_result(tmp_path, mass_plan))
    config["spatial_discretization"]["target_cell_size_m"] = 0.0001
    config["spatial_discretization"]["dimensionless_resize_factor"] = 0.1
    report = validate_wasserstein_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "spatial_contract_changed" in codes
    assert "physical_grid_would_upsample" in codes


def test_wasserstein_digest_changes_with_component_selection(
    tmp_path: Path,
) -> None:
    project, _mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_wasserstein_plan(project)
    baseline = validate_wasserstein_plan(project, config)["task_contract"]
    changed = deepcopy(config)
    changed["comparison"]["mass_components"] = ["mass_total"]

    assert (
        validate_wasserstein_plan(project, changed)["task_contract"]["content_sha256"]
        != baseline["content_sha256"]
    )


def test_wasserstein_save_revalidates_current_mass_plan(tmp_path: Path) -> None:
    project, _mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_wasserstein_plan(project)
    preview_report = validate_wasserstein_plan(project, config)
    replacement_config = _confirmed_mass_plan(project)
    replacement_report = validate_mass_analysis_plan(project, replacement_config)
    MassAnalysisPlanStore(project).save_plan(
        replacement_config,
        actor="Second reviewer",
        reason="Replace current mass plan",
        validation=replacement_report,
    )

    try:
        WassersteinPlanStore(project).save_plan(
            config,
            actor="Reviewer",
            reason="This Wasserstein plan is stale",
            validation=preview_report,
        )
    except ValueError as exc:
        assert "blocking validation errors" in str(exc)
    else:
        raise AssertionError("A Wasserstein plan bound to stale mass data was saved")
