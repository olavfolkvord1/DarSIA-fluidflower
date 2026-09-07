from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from darsia.fluidflower.frame_quality import (
    FrameQualityStore,
    assess_frame_quality_rows,
    frame_quality_signature,
    scan_frame_quality,
)
from darsia.fluidflower.image_correction import identity_correction_config
from darsia.fluidflower.project import FluidFlowerProject, sha256_file


def _base_image() -> np.ndarray:
    rows, columns = 120, 180
    x = np.linspace(0.38, 0.62, columns, dtype=np.float32)
    image = np.repeat(x[None, :, None], rows, axis=0)
    return np.repeat(image, 3, axis=2)


def _quality_row(
    index: int,
    original: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, object]:
    return {
        "frame_id": f"frame-{index}",
        "run_id": "run18",
        "source_path": f"frame-{index}.png",
        "source_sha256": f"sha-{index}",
        "correction_dependency_sha256": "correction-sha",
        "raw_signature": frame_quality_signature(original),
        "corrected_signature": frame_quality_signature(corrected),
        "correction_qc": {"flagged": False},
        "analysis_error": "",
    }


def test_light_failure_can_be_flagged_as_corrected() -> None:
    base = _base_image()
    rows = []
    for index in range(7):
        original = base.copy()
        if index == 3:
            original[:, : original.shape[1] // 2] *= 0.35
        rows.append(_quality_row(index, original, base))

    assessment = assess_frame_quality_rows(rows)
    failed_light = assessment["frames"][3]

    assert failed_light["automatic_status"] == "corrected"
    assert "raw_left_right_imbalance" in failed_light["automatic_flags"]
    assert failed_light["blacklist_candidate"] is False


def test_persistent_transient_obstruction_suggests_exclusion() -> None:
    base = _base_image()
    rows = []
    for index in range(7):
        image = base.copy()
        if index == 3:
            image[15:105, 45:135] = 0.0
        rows.append(_quality_row(index, image, image))

    assessment = assess_frame_quality_rows(rows)
    obstruction = assessment["frames"][3]

    assert obstruction["automatic_status"] == "exclusion_suggested"
    assert any(
        "occlusion_or_obstruction" in flag for flag in obstruction["automatic_flags"]
    )
    assert obstruction["automatically_excluded"] is False


def test_only_manual_decisions_enter_the_derived_blacklist(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Quality review",
        series_id="quality-review",
        template_id="ff_um",
        rig_type="medium",
        created_by="Reviewer",
    )
    base = _base_image()
    obstruction = base.copy()
    obstruction[15:105, 45:135] = 0.0
    rows = [
        _quality_row(
            index,
            obstruction if index == 3 else base,
            obstruction if index == 3 else base,
        )
        for index in range(7)
    ]
    store = FrameQualityStore(project)
    project.set_runs([{"run_id": "run18", "label": "RUN18"}])
    assessment = store.save_assessment(
        assess_frame_quality_rows(rows),
        actor="Reviewer",
    )

    assert store.active_blacklist()["entry_count"] == 0
    assert assessment["frames"][3]["blacklist_candidate"] is True

    store.set_decision(
        "frame-3",
        "exclude",
        actor="Reviewer",
        reason="Person in front of camera",
    )
    blacklist = store.active_blacklist()
    assert blacklist["entry_count"] == 1
    assert blacklist["entries"][0]["frame_id"] == "frame-3"
    assert (store.run_blacklist_root / "run18.csv").is_file()
    assert (
        project.manifest["runs"][0]["workflow"]["image_correction"]["status"]
        == "review_required"
    )

    store.set_decision("frame-3", "clear", actor="Reviewer")
    assert store.active_blacklist()["entry_count"] == 0
    assert len(store.decisions()) == 2


def test_exclusions_require_named_reviewer_and_reason(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Quality review",
        series_id="quality-review",
        template_id="ff_um",
        rig_type="medium",
        created_by="Reviewer",
    )
    store = FrameQualityStore(project)
    assessment = assess_frame_quality_rows(
        [_quality_row(0, _base_image(), _base_image())]
    )
    with pytest.raises(ValueError, match="reviewer name"):
        store.save_assessment(assessment, actor="")

    store.save_assessment(assessment, actor="Reviewer")
    with pytest.raises(ValueError, match="reviewer name"):
        store.set_decision("frame-0", "keep", actor="")
    with pytest.raises(ValueError, match="review note"):
        store.set_decision("frame-0", "exclude", actor="Reviewer")


def test_filter_frames_uses_normalized_paths(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Quality review",
        series_id="quality-review",
        template_id="ff_um",
        rig_type="medium",
        created_by="Reviewer",
    )
    source = tmp_path / "images" / "frame.png"
    source.parent.mkdir()
    source.write_bytes(b"frame")
    row = _quality_row(0, _base_image(), _base_image())
    row["source_path"] = str(source)
    store = FrameQualityStore(project)
    store.save_assessment(assess_frame_quality_rows([row]), actor="Reviewer")
    store.set_decision(
        "frame-0",
        "exclude",
        actor="Reviewer",
        reason="Unrecoverable lighting failure",
    )

    relative = Path(os.path.relpath(source, Path.cwd()))
    assert store.filter_frames([{"source_path": str(relative)}]) == []


def test_identity_scan_builds_run_aware_quality_report(tmp_path: Path) -> None:
    image_root = tmp_path / "images" / "run18"
    image_root.mkdir(parents=True)
    frames = []
    base = _base_image()
    for index in range(5):
        path = image_root / f"frame-{index:03d}.png"
        assert cv2.imwrite(
            str(path),
            cv2.cvtColor(
                np.rint(base * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        frames.append(
            {
                "frame_id": f"frame-{index}",
                "run_id": "run18",
                "source_path": str(path),
                "relative_path": path.name,
                "sha256": sha256_file(path),
            }
        )

    report = scan_frame_quality(
        frames,
        identity_correction_config(),
        workers=2,
    )

    assert report["frame_count"] == 5
    assert report["run_count"] == 1
    assert report["runs"]["run18"]["ok"] == 5
    assert not any(row["automatically_excluded"] for row in report["frames"])
    assert json.dumps(report)


def test_worker_limit_changes_are_applied_without_restarting_scan(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images" / "run18"
    image_root.mkdir(parents=True)
    frames = []
    base = _base_image()
    for index in range(12):
        path = image_root / f"frame-{index:03d}.png"
        assert cv2.imwrite(
            str(path),
            cv2.cvtColor(
                np.rint(base * 255.0).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        frames.append(
            {
                "frame_id": f"frame-{index}",
                "run_id": "run18",
                "source_path": str(path),
                "relative_path": path.name,
                "sha256": sha256_file(path),
            }
        )

    completed = 0
    statuses: list[tuple[int, int]] = []

    def progress(done: int, _total: int) -> None:
        nonlocal completed
        completed = done

    def desired_workers() -> int:
        if completed < 1:
            return 1
        if completed < 4:
            return 4
        return 2

    report = scan_frame_quality(
        frames,
        identity_correction_config(),
        workers=1,
        max_workers=6,
        worker_limit=desired_workers,
        worker_status=lambda active, desired: statuses.append((active, desired)),
        progress=progress,
    )

    requested = [row["requested_workers"] for row in report["worker_history"]]
    assert requested == [1, 4, 2]
    assert any(active == 4 and desired == 4 for active, desired in statuses)
    assert any(desired == 2 for _active, desired in statuses)
    assert report["initial_worker_count"] == 1
    assert report["maximum_worker_count"] == 6
    assert report["dynamic_worker_limit"] is True
