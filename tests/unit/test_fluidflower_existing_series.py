from __future__ import annotations

from pathlib import Path

from darsia.fluidflower.existing_series import (
    create_existing_series_project,
    discover_ffac_series,
    discover_ffum_series,
)


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_discover_ffac_series_uses_only_analysis_image_folders(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    run = raw / "AC52"
    early_flush = run / "230515_144800_AC52_flush_30s"
    injection = run / "230516_090400_AC52_injection_30s"
    resting = run / "230516_121510_AC52_resting_5min"
    late_flush = run / "230517_120000_AC52_flush_after"
    overflow = run / "AC52_overflow"
    _image(early_flush / "DSC0001.JPG")
    _image(early_flush / "DSC0002.JPG")
    _image(injection / "DSC0003.JPG")
    _image(resting / "DSC0004.JPG")
    _image(late_flush / "DSC0005.JPG")
    _image(overflow / "scale.JPG")
    (run / "Albus protocol 52.xlsx").write_bytes(b"protocol")
    reviews = tmp_path / "reviews" / "ac52"
    reviews.mkdir(parents=True)
    (reviews / "ac52_overflow_scale_review.json").write_text("{}")

    rows = discover_ffac_series(raw, overflow_review_root=tmp_path / "reviews")

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "ac52"
    assert row["image_count"] == 2
    assert row["baseline_candidate"].endswith("DSC0002.JPG")
    assert row["clearwater_candidate"].endswith("DSC0001.JPG")
    assert "flush_after" not in row["baseline_candidate"]
    assert row["overflow_review"].endswith("ac52_overflow_scale_review.json")
    assert row["candidate_status"] == "requires_user_confirmation"


def test_discover_ffum_series_orders_runs_numerically(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    _image(experiments / "Run20" / "DCS200.JPG")
    _image(experiments / "Run6" / "DCS060.JPG")
    clear = tmp_path / "clearwater"
    _image(clear / "run20" / "clear.JPG")
    configs = tmp_path / "config"
    configs.mkdir()
    (configs / "run20.toml").write_text("[run]\n")

    rows = discover_ffum_series(
        experiments,
        clearwater_root=clear,
        configuration_root=configs,
    )

    assert [row["run_id"] for row in rows] == ["run6", "run20"]
    assert rows[0]["clearwater_candidate"] == ""
    assert rows[1]["clearwater_candidate"].endswith("clear.JPG")
    assert rows[1]["run_configuration"].endswith("run20.toml")


def test_create_existing_series_project_is_reference_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image = _image(source / "Run1" / "frame.JPG")
    project_root = tmp_path / "project"
    before = image.read_bytes()

    result = create_existing_series_project(
        project_root,
        name="Reference project",
        series_id="reference-project",
        template_id="ff_um",
        rig_type="large",
        runs=[
            {
                "run_id": "run1",
                "experiment_folder": str(image.parent),
                "image_count": 1,
                "first_image": str(image),
                "last_image": str(image),
                "baseline_candidate": str(image),
                "clearwater_candidate": "",
                "candidate_status": "requires_user_confirmation",
                "mapping_status": "ready_for_review",
            }
        ],
        protocol_path=None,
        source_paths={"experiment_images": source},
        protocol_organization="test",
        created_by="Reviewer",
    )

    assert image.read_bytes() == before
    assert result.report["source_files_modified"] is False
    assert result.report["reference_candidates_approved"] is False
    assert result.project.manifest["tools"]["source_mapping"][
        "baseline_and_clearwater_candidates_approved"
    ] is False
    assert (project_root / "existing_series_import.json").is_file()
    assert (project_root / "README_IMPORT.md").is_file()
