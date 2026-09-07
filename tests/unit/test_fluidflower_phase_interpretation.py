from __future__ import annotations

import json
from pathlib import Path

from darsia.fluidflower.phase_interpretation import (
    PHASE_INTERPRETATION_CONFIG_FORMAT,
    PHASE_INTERPRETATION_REVISION_FORMAT,
    PhaseInterpretationRevisionStore,
    default_phase_interpretation_config,
    inspect_phase_campaign,
    parse_review_file_list,
    validate_phase_interpretation_config,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file


def _campaign(path: Path) -> Path:
    payload = {
        "schema": 2,
        "campaign_id": "phase-campaign-test",
        "dataset": "ffac",
        "jobs": [
            {
                "job_id": "ac17-optical",
                "run": "ac17",
                "role": "optical",
                "variant": "optical-v1",
                "family": "hierarchical-phase",
                "candidate_id": "candidate-ac17-optical",
                "quality_flags": {
                    "dataset": "ffac",
                    "hierarchical_phase_partition_active": True,
                },
            },
            {
                "job_id": "ac17-physics",
                "run": "ac17",
                "role": "physics",
                "variant": "physics-v1",
                "family": "hierarchical-phase",
                "candidate_id": "candidate-ac17-physics",
                "numerical_products": {
                    "phase_co2_probability": {
                        "path": "ac17/co2/index.json",
                        "sha256": "a" * 64,
                    }
                },
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
        "models": {
            "physics": {
                "gas_path": "models/gas.json",
                "gas_model_sha256": "b" * 64,
            }
        },
        "design": {
            "comparison": "optical versus physics",
            "interpretation_scope": "visible-2d-slice-only",
            "mass_calibration_active": False,
            "depth_used": False,
            "deployment_allowed": False,
        },
        "review": {"output_dir": str(path.parent / "review")},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_campaign_intake_is_read_only_and_candidate_aware(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    campaign_path = _campaign(tmp_path / "campaign_manifest.json")
    summary = inspect_phase_campaign(campaign_path)

    assert summary["campaign_id"] == "phase-campaign-test"
    assert summary["runs"] == ["ac17", "ac18"]
    assert len(summary["candidates_by_run"]["ac17"]) == 2
    assert summary["models"]["physics"]["gas_model_sha256"] == "b" * 64
    assert (
        summary["candidates_by_run"]["ac17"][1]["numerical_products"]
        ["phase_co2_probability"]["sha256"]
        == "a" * 64
    )
    assert not summary["mass_calibration_active"]
    assert not summary["depth_used"]
    assert not summary["deployment_allowed"]
    assert summary["manifest"]["sha256"] == sha256_file(campaign_path)


def test_phase_revision_freezes_reviews_candidates_and_upstream_state(
    tmp_path: Path,
) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Phase intake",
        series_id="phase-intake",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    review = tmp_path / "review"
    review.mkdir()
    manual_review = review / "reviewer.json"
    manual_review.write_text('{"reviewer":"Reviewer"}', encoding="utf-8")
    campaign_path = _campaign(tmp_path / "campaign_manifest.json")
    config = default_phase_interpretation_config()
    config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review),
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
            "manual_review_files": [str(manual_review)],
        }
    )
    report = validate_phase_interpretation_config(config)

    assert report["valid"]
    assert report["summary"] == {
        "error_count": 0,
        "warning_count": 0,
        "run_count": 2,
        "candidate_count": 3,
        "selected_run_count": 2,
        "reviewed_run_count": 2,
        "approved_run_count": 2,
        "needs_correction_run_count": 0,
        "manual_review_file_count": 1,
    }
    store = PhaseInterpretationRevisionStore(project)
    revision = store.save_revision(
        config,
        actor="Reviewer",
        reason="Candidate review complete",
        validation=report,
    )

    assert revision["format"] == PHASE_INTERPRETATION_REVISION_FORMAT
    assert revision["configuration"]["format"] == PHASE_INTERPRETATION_CONFIG_FORMAT
    assert not revision["scientific_scope"]["trains_model"]
    assert not revision["scientific_scope"]["contains_depth_or_3d_extrapolation"]
    assert revision["validation"]["manual_review_files"][0]["sha256"] == sha256_file(
        manual_review
    )
    assert revision["upstream"]["timeline"]["event_count"] == 0
    assert store.current()["revision_id"] == revision["revision_id"]
    assert (
        project.root
        / "results"
        / "phase_interpretation"
        / revision["revision_id"]
        / "revision.json"
    ).is_file()


def test_phase_intake_rejects_training_and_unknown_candidates(tmp_path: Path) -> None:
    review = tmp_path / "review"
    review.mkdir()
    campaign_path = _campaign(tmp_path / "campaign_manifest.json")
    config = default_phase_interpretation_config()
    config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review),
            "selected_candidates": {"ac17": "not-in-campaign"},
            "training_requested": True,
        }
    )
    report = validate_phase_interpretation_config(config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "training_not_supported" in codes
    assert "unknown_selected_candidate" in codes


def test_phase_approval_is_invalidated_when_selected_candidate_changes(
    tmp_path: Path,
) -> None:
    review = tmp_path / "review"
    review.mkdir()
    campaign_path = _campaign(tmp_path / "campaign_manifest.json")
    config = default_phase_interpretation_config()
    config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review),
            "selected_candidates": {"ac17": "candidate-ac17-optical"},
            "run_reviews": {
                "ac17": {
                    "status": "approved",
                    "candidate_id": "candidate-ac17-physics",
                    "reviewer": "Reviewer",
                }
            },
        }
    )

    report = validate_phase_interpretation_config(config)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["valid"]
    assert "run_review_candidate_changed" in codes


def test_review_file_parser_accepts_newlines_semicolons_and_deduplicates(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    value = f"{first}\n{second};{first}"
    assert parse_review_file_list(value) == [
        str(first.resolve()),
        str(second.resolve()),
    ]
