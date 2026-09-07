from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from darsia.fluidflower.mass_analysis import (
    MassAnalysisPlanStore,
    validate_mass_analysis_plan,
)
from darsia.fluidflower.post_analysis import (
    POST_ANALYSIS_PLAN_FORMAT,
    POST_ANALYSIS_RESULT_FORMAT,
    POST_ANALYSIS_REVISION_FORMAT,
    POST_ANALYSIS_TASK_CONTRACT_FORMAT,
    REQUIRED_POST_ANALYSIS_PRODUCTS,
    PostAnalysisPlanStore,
    default_post_analysis_plan,
    expected_post_analysis_artifacts,
    inspect_post_analysis_result,
    validate_post_analysis_plan,
)
from darsia.fluidflower.project import sha256_file
from tests.unit.test_fluidflower_mass_analysis import (
    _confirmed_mass_plan,
    _mass_result,
    _project_with_mass_plan,
)


def _mass_result_with_phase_products(tmp_path: Path, mass_plan: dict) -> Path:
    result_path = _mass_result(tmp_path, mass_plan)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for row in payload["run_results"]:
        run_id = row["run_id"]
        for product in (
            "concentration_aqueous",
            "saturation_free_gas",
            "preview_images",
        ):
            path = tmp_path / f"{run_id}-{product}-index.json"
            path.write_text(
                json.dumps({"run_id": run_id, "product": product}),
                encoding="utf-8",
            )
            row["products"][product] = {
                "path": path.name,
                "sha256": sha256_file(path),
            }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def _post_analysis_result(tmp_path: Path, post_plan: dict) -> Path:
    artifacts = []
    for index, expected in enumerate(
        expected_post_analysis_artifacts(post_plan["configuration"])
    ):
        product = expected["product"]
        run_key = "-".join(expected["run_ids"]) or "global"
        artifact_path = tmp_path / f"post-{index:03d}-{product}-{run_key}.dat"
        artifact_path.write_text(
            json.dumps({"product": product, "run_ids": expected["run_ids"]}),
            encoding="utf-8",
        )
        artifacts.append(
            {
                "artifact_id": f"artifact-{index:03d}",
                "product": product,
                "run_ids": expected["run_ids"],
                "path": artifact_path.name,
                "sha256": sha256_file(artifact_path),
                "media_type": "application/octet-stream",
            }
        )
    result_path = tmp_path / "post_analysis_result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": POST_ANALYSIS_RESULT_FORMAT,
                "post_analysis_plan_id": post_plan["plan_id"],
                "post_analysis_task_contract_sha256": post_plan["task_contract"][
                    "content_sha256"
                ],
                "complete": True,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return result_path


def test_post_analysis_plan_is_valid_draft_without_mass_result(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_post_analysis_plan(project)
    report = validate_post_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert report["valid"]
    assert not report["execution_ready"]
    assert report["summary"]["run_count"] == 2
    assert "mass_result_missing" in codes
    assert report["task_contract"]["format"] == POST_ANALYSIS_TASK_CONTRACT_FORMAT

    revision = PostAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Define products while mass analysis is pending",
        validation=report,
    )
    assert revision["format"] == POST_ANALYSIS_REVISION_FORMAT
    assert revision["configuration"]["format"] == POST_ANALYSIS_PLAN_FORMAT
    assert revision["upstream"]["mass_analysis_plan_id"] == mass_plan["plan_id"]
    assert not revision["execution"]["ready"]
    assert not revision["execution"]["started"]
    assert not revision["scientific_scope"]["contains_rendered_products"]
    assert revision["scientific_scope"]["raw_detected_mass_is_authoritative"]


def test_post_analysis_plan_becomes_ready_with_phase_products(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    result_path = _mass_result_with_phase_products(tmp_path, mass_plan)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(result_path)
    report = validate_post_analysis_plan(project, config)

    assert report["valid"]
    assert report["execution_ready"]
    assert report["summary"]["warning_count"] == 0
    assert report["summary"]["verified_input_artifact_count"] == 6
    assert report["mass_result"]["reference"]["sha256"] == sha256_file(result_path)

    revision = PostAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Ready for external post-analysis runner",
        validation=report,
    )
    assert revision["execution"]["ready"]
    assert (
        project.root
        / "results"
        / "post_analysis"
        / "plans"
        / revision["plan_id"]
        / "plan.json"
    ).is_file()


def test_phase_products_require_phase_fields_and_corrected_frames(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(_mass_result(tmp_path, mass_plan))
    report = validate_post_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "post_analysis_input_missing" in codes


def test_mass_only_post_products_need_no_phase_visual_inputs(tmp_path: Path) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(_mass_result(tmp_path, mass_plan))
    config["products"] = [
        *REQUIRED_POST_ANALYSIS_PRODUCTS,
        "mass_curve_full_domain",
        "mass_curve_analysis_domains",
        "cross_run_mass_grid",
    ]
    report = validate_post_analysis_plan(project, config)

    assert report["valid"]
    assert report["execution_ready"]
    assert report["summary"]["verified_input_artifact_count"] == 0


def test_cross_run_product_requires_two_selected_runs(tmp_path: Path) -> None:
    project, _mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_post_analysis_plan(project)
    config["run_ids"] = config["run_ids"][:1]
    report = validate_post_analysis_plan(project, config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "cross_run_product_without_runs" in codes


def test_post_analysis_digest_changes_with_rendering_contract(
    tmp_path: Path,
) -> None:
    project, _mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_post_analysis_plan(project)
    baseline = validate_post_analysis_plan(project, config)["task_contract"]
    changed = deepcopy(config)
    changed["rendering"]["movie_fps"] = 12.0

    assert (
        validate_post_analysis_plan(project, changed)["task_contract"]["content_sha256"]
        != baseline["content_sha256"]
    )


def test_post_analysis_save_revalidates_current_mass_plan(tmp_path: Path) -> None:
    project, _mass_plan = _project_with_mass_plan(tmp_path, ready=False)
    config = default_post_analysis_plan(project)
    preview_report = validate_post_analysis_plan(project, config)
    replacement_config = _confirmed_mass_plan(project)
    replacement_report = validate_mass_analysis_plan(project, replacement_config)
    MassAnalysisPlanStore(project).save_plan(
        replacement_config,
        actor="Second reviewer",
        reason="Replace current mass plan",
        validation=replacement_report,
    )

    try:
        PostAnalysisPlanStore(project).save_plan(
            config,
            actor="Reviewer",
            reason="This post-analysis plan is stale",
            validation=preview_report,
        )
    except ValueError as exc:
        assert "blocking validation errors" in str(exc)
    else:
        raise AssertionError("A post-analysis plan bound to stale mass data was saved")


def test_post_analysis_result_verifies_exact_product_coverage(tmp_path: Path) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(
        _mass_result_with_phase_products(tmp_path, mass_plan)
    )
    report = validate_post_analysis_plan(project, config)
    post_plan = PostAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic completed post-analysis task",
        validation=report,
    )
    result_path = _post_analysis_result(tmp_path, post_plan)
    inspection = inspect_post_analysis_result(str(result_path), post_plan)

    assert inspection["valid"]
    assert inspection["complete"]
    assert inspection["reference"]["artifact_count"] == len(
        expected_post_analysis_artifacts(config)
    )


def test_complete_post_result_rejects_missing_planned_artifact(tmp_path: Path) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(
        _mass_result_with_phase_products(tmp_path, mass_plan)
    )
    post_plan = PostAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic completed post-analysis task",
    )
    result_path = _post_analysis_result(tmp_path, post_plan)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["artifacts"].pop()
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    inspection = inspect_post_analysis_result(str(result_path), post_plan)

    assert not inspection["valid"]
    assert "post_analysis_artifact_missing" in {
        issue["code"] for issue in inspection["issues"]
    }
