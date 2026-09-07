from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from darsia.fluidflower.phase_workflow import (
    PHASE_REVIEW_SESSION_FORMAT,
    PHASE_TRAINING_REQUEST_FORMAT,
    build_phase_review_command,
    build_phase_review_resume_command,
    build_phase_training_command,
    discover_phase_review_sessions,
    parse_training_bundles,
    phase_python_executable,
    prepare_phase_training_request,
)


def test_companion_process_prefers_virtual_environment_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / ("Scripts/pythonw.exe" if os.name == "nt" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))

    assert Path(phase_python_executable(gui=True)) == executable.resolve()


def test_single_model_review_command_retains_explicit_paths(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "postflight_manifest.json").write_text("{}", encoding="utf-8")
    program = tmp_path / "review.py"
    program.write_text("pass\n", encoding="utf-8")

    command = build_phase_review_command(
        mode="model_validation",
        campaign_root=campaign,
        reviewer="Olav Folkvord",
        save_root=tmp_path / "reviews",
        cache_root=tmp_path / "cache",
        selected_candidates={"ac52": "candidate-52"},
        target_model_id="phase-model-v6",
        review_session_id="review-001",
        program=program,
    )

    assert "--workspace" in command
    assert command[command.index("--workspace") + 1] == "current_comparison"
    assert "--ffac-root" in command
    assert "--validation-ffac-root" not in command
    assert "--single-model" in command
    assert "--approval-only" in command
    assert "--online" in command
    assert "ac52=candidate-52" in command
    assert command[command.index("--target-model-id") + 1] == "phase-model-v6"
    assert command[command.index("--review-session-id") + 1] == "review-001"
    assert str(campaign.resolve()) in command


def test_review_commands_keep_blind_labeling_and_open_correction_distinct(
    tmp_path: Path,
) -> None:
    roots = []
    for name in ("ffac", "ffum", "combined"):
        root = tmp_path / name
        root.mkdir()
        (root / "postflight_manifest.json").write_text("{}", encoding="utf-8")
        roots.append(root)
    program = tmp_path / "review.py"
    program.write_text("pass\n", encoding="utf-8")
    common = {
        "reviewer": "Olav Folkvord",
        "save_root": tmp_path / "reviews",
        "cache_root": tmp_path / "cache",
        "ffac_root": roots[0],
        "ffum_root": roots[1],
        "combined_root": roots[2],
        "program": program,
    }

    blind = build_phase_review_command(mode="blind_region_labeling", **common)
    correction = build_phase_review_command(mode="open_correction", **common)

    assert blind[blind.index("--workspace") + 1] == "training"
    assert blind[blind.index("--review-surface") + 1] == "blind_regions"
    assert "--approval-only" not in blind
    assert correction[correction.index("--workspace") + 1] == "current_comparison"
    assert correction[correction.index("--review-surface") + 1] == ("open_correction")
    assert "--approval-only" not in correction


def test_review_session_discovery_counts_and_resumes_exact_store(
    tmp_path: Path,
) -> None:
    review_root = tmp_path / "reviews"
    decision_dir = (
        review_root
        / "olav-folkvord"
        / "v5-campaign"
        / "current_comparison"
        / "manual_pairwise_review"
    )
    decision_dir.mkdir(parents=True)
    campaigns = {}
    for name in ("ffac", "ffum", "combined"):
        root = tmp_path / name
        root.mkdir()
        (root / "postflight_manifest.json").write_text("{}", encoding="utf-8")
        campaigns[name] = root
    decisions = {
        "decisions": {
            "ac32": {
                "annotations": [{"id": "one"}, {"id": "two"}],
                "phase_quiz_answers": [{"proposal_id": "three"}],
                "frame_choices": [{"frame_index": 0}],
            }
        }
    }
    (decision_dir / "pairwise_review_decisions.json").write_text(
        json.dumps(decisions), encoding="utf-8"
    )
    session_payload = {
        "schema": 1,
        "format": PHASE_REVIEW_SESSION_FORMAT,
        "reviewer": "Olav Folkvord",
        "reviewer_id": "olav-folkvord",
        "campaign_id": "v5-campaign",
        "workspace": "current_comparison",
        "workspace_title": "Open V5 model correction",
        "decision_dir": str(decision_dir),
        "ffac_root": str(campaigns["ffac"]),
        "ffum_root": str(campaigns["ffum"]),
        "combined_root": str(campaigns["combined"]),
        "review_surface": "open_correction",
        "single_model": True,
    }
    (decision_dir / "review_session.json").write_text(
        json.dumps(session_payload), encoding="utf-8"
    )
    program = tmp_path / "review.py"
    program.write_text("pass\n", encoding="utf-8")

    sessions = discover_phase_review_sessions((review_root,))

    assert len(sessions) == 1
    session = sessions[0]
    assert session["manual_regions"] == 2
    assert session["blind_region_labels"] == 1
    assert session["frame_verdicts"] == 1
    assert Path(session["local_review_root"]) == review_root.resolve()
    command = build_phase_review_resume_command(
        session,
        cache_root=tmp_path / "cache",
        program=program,
    )
    assert command[command.index("--workspace") + 1] == "current_comparison"
    assert command[command.index("--local-review-root") + 1] == str(
        review_root.resolve()
    )
    assert command[command.index("--review-surface") + 1] == "open_correction"
    assert "--single-model" in command


def test_training_request_snapshots_sources_and_builds_command(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.json"
    manifest = tmp_path / "combined.json"
    decisions.write_text('{"decisions": {}}', encoding="utf-8")
    manifest.write_text('{"jobs": []}', encoding="utf-8")
    bundles = parse_training_bundles(f"{decisions} | {manifest}")
    request = prepare_phase_training_request(
        output_parent=tmp_path / "models",
        cache_parent=tmp_path / "cache",
        bundles=bundles,
        actor="Reviewer",
        settings={"backend": "cpu", "stage_workers": 2},
    )
    trainer = tmp_path / "trainer.py"
    trainer.write_text("pass\n", encoding="utf-8")
    command = build_phase_training_command(request, program=trainer)

    assert request["format"] == PHASE_TRAINING_REQUEST_FORMAT
    assert Path(request["request_path"]).is_file()
    assert Path(request["frozen_pairs"][0][0]).read_text(encoding="utf-8") == (
        decisions.read_text(encoding="utf-8")
    )
    assert "--campaign-id" in command
    assert request["model_id"] in command
    assert "--resume-completed-heads" in command
    saved = json.loads(Path(request["request_path"]).read_text(encoding="utf-8"))
    assert saved["bundles"][0]["decisions"]["sha256"]


def test_training_request_locks_live_review_session(tmp_path: Path) -> None:
    decision_dir = tmp_path / "reviews" / "manual_pairwise_review"
    decision_dir.mkdir(parents=True)
    decisions = decision_dir / "pairwise_review_decisions.json"
    manifest = tmp_path / "combined.json"
    decisions.write_text('{"decisions": {}}', encoding="utf-8")
    manifest.write_text('{"jobs": []}', encoding="utf-8")
    session_path = decision_dir / "review_session.json"
    session_path.write_text(
        json.dumps(
            {
                "format": PHASE_REVIEW_SESSION_FORMAT,
                "review_session_id": "review-001",
                "target_model_id": "phase-model-v5",
            }
        ),
        encoding="utf-8",
    )

    request = prepare_phase_training_request(
        output_parent=tmp_path / "models",
        cache_parent=tmp_path / "cache",
        bundles=((decisions, manifest),),
        actor="Reviewer",
        settings={"backend": "cpu"},
    )

    session = json.loads(session_path.read_text(encoding="utf-8"))
    assert session["editing_locked"] is True
    assert session["training_locks"][0]["model_id"] == request["model_id"]
    assert session["training_locks"][0]["source_matches_frozen_revision"] is True
    assert request["locked_review_sessions"][0]["session_path"] == str(
        session_path.resolve()
    )


def test_default_training_command_uses_native_trainer_and_optional_ablation(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "decisions.json"
    manifest = tmp_path / "combined.json"
    decisions.write_text('{"decisions": {}}', encoding="utf-8")
    manifest.write_text('{"jobs": []}', encoding="utf-8")
    request = prepare_phase_training_request(
        output_parent=tmp_path / "models",
        cache_parent=tmp_path / "cache",
        bundles=((decisions, manifest),),
        actor="Reviewer",
        settings={
            "backend": "cpu",
            "change_evidence_ablation": True,
        },
    )

    command = build_phase_training_command(request)

    assert command[1:3] == ["-m", "darsia.fluidflower.phase_training.trainer"]
    assert "--change-evidence-ablation" in command
    assert command[command.index("--campaign-id") + 1] == request["model_id"]
