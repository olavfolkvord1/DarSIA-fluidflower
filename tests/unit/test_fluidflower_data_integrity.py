from __future__ import annotations

import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

import darsia.fluidflower.data_integrity as data_integrity
from darsia.fluidflower.data_integrity import (
    IntegrityReportStore,
    IntegrityScanner,
    IntegrityScanOptions,
    ScanCancelled,
    recommend_integrity_read_workers,
)
from darsia.fluidflower.project import FluidFlowerProject


def create_project(tmp_path: Path) -> tuple[FluidFlowerProject, Path]:
    images = tmp_path / "raw-images"
    images.mkdir()
    for number in (1, 2, 4, 5):
        Image.new("RGB", (32, 24), (number * 20, 80, 140)).save(
            images / f"frame{number:03d}.jpg",
            quality=95,
        )
    shutil.copyfile(images / "frame004.jpg", images / "frame005.jpg")
    Image.new("RGB", (16, 12), (30, 40, 50)).save(
        images / "frame006.jpg",
        quality=95,
    )
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Integrity test",
        series_id="integrity-test",
        template_id="ff_ac",
        rig_type="medium",
        image_sources={"experiment_images": images},
        created_by="Reviewer",
    )
    return project, images


def test_fast_scan_finds_duplicates_resolution_and_sequence_gaps(
    tmp_path: Path,
) -> None:
    project, _images = create_project(tmp_path)
    report = IntegrityScanner(project).scan(actor="Reviewer")
    codes = {issue["code"] for issue in report["issues"]}

    assert report["summary"]["file_count"] == 5
    assert report["summary"]["image_count"] == 5
    assert "exact_duplicate_files" in codes
    assert "resolution_outliers" in codes
    assert "missing_sequence_numbers" in codes
    assert "missing_image_timestamps" in codes
    duplicate = next(
        issue for issue in report["issues"] if issue["code"] == "exact_duplicate_files"
    )
    assert len(duplicate["details"]["paths"]) == 2


def test_verified_scan_hashes_every_file(tmp_path: Path) -> None:
    project, _images = create_project(tmp_path)
    report = IntegrityScanner(
        project,
        IntegrityScanOptions(mode="verified"),
    ).scan(actor="Reviewer")
    assert all(row["sha256"] for row in report["inventory"])


def test_integrity_scan_excludes_non_preferred_paired_camera_files(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    jpg = images / "frame001.JPG"
    raw = images / "frame001.ARW"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(jpg)
    raw.write_bytes(b"camera raw payload")
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Paired camera files",
        series_id="paired-camera-files",
        template_id="ff_ac",
        rig_type="medium",
        image_sources={"experiment_images": images},
        created_by="Reviewer",
    )

    processed_report = IntegrityScanner(project).scan(actor="Reviewer")
    assert {Path(row["path"]).suffix for row in processed_report["inventory"]} == {
        ".JPG"
    }

    project.update_metadata(
        "tools",
        {"source_mapping": {"image_file_policy": "prefer_raw"}},
        updated_by="Reviewer",
    )
    raw_report = IntegrityScanner(project).scan(actor="Reviewer")
    assert {Path(row["path"]).suffix for row in raw_report["inventory"]} == {
        ".ARW"
    }
    assert raw_report["inventory"][0]["is_camera_raw"] is True


def test_only_verified_mode_decodes_complete_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, _images = create_project(tmp_path)
    verification_flags: list[bool] = []
    original = data_integrity._image_metadata

    def record_mode(path: Path, *, verify: bool, read_exif: bool) -> dict[str, object]:
        verification_flags.append(verify)
        return original(path, verify=verify, read_exif=read_exif)

    monkeypatch.setattr(data_integrity, "_image_metadata", record_mode)
    IntegrityScanner(project).scan(actor="Reviewer")
    assert verification_flags == [False] * 5

    verification_flags.clear()
    IntegrityScanner(
        project,
        IntegrityScanOptions(mode="verified"),
    ).scan(actor="Reviewer")
    assert verification_flags == [True] * 5


def test_report_store_publishes_summary_and_compressed_inventory(
    tmp_path: Path,
) -> None:
    project, _images = create_project(tmp_path)
    report = IntegrityScanner(project).scan(actor="Reviewer")
    store = IntegrityReportStore(project)
    destination = store.save(report, actor="Reviewer")

    assert (destination / "report.json").is_file()
    assert (destination / "inventory.jsonl.gz").is_file()
    current = store.current()
    assert current is not None
    assert current["scan_id"] == report["scan_id"]
    assert len(store.inventory(current)) == 5
    pointer = json.loads(store.current_path.read_text(encoding="utf-8"))
    assert pointer["summary"]["file_count"] == 5


def test_missing_source_is_reported_without_crashing_scan(tmp_path: Path) -> None:
    project, images = create_project(tmp_path)
    shutil.rmtree(images)
    report = IntegrityScanner(
        project,
        IntegrityScanOptions(retry_count=1),
    ).scan(actor="Reviewer")
    assert report["summary"]["error_count"] == 1
    assert report["issues"][0]["code"] == "source_unavailable"


def test_scan_can_be_cancelled_before_reading_files(tmp_path: Path) -> None:
    project, _images = create_project(tmp_path)
    with pytest.raises(ScanCancelled):
        IntegrityScanner(project).scan(cancelled=lambda: True)


def test_parallel_read_limit_can_change_during_scan(tmp_path: Path) -> None:
    project, _images = create_project(tmp_path)
    requested = {"value": 1}
    statuses: list[tuple[int, int]] = []

    def worker_status(active: int, desired: int) -> None:
        statuses.append((active, desired))
        if active == 1 and desired == 1:
            requested["value"] = 4
        elif desired == 4:
            requested["value"] = 2

    report = IntegrityScanner(
        project,
        IntegrityScanOptions(parallel_reads=1),
    ).scan(
        max_parallel_reads=6,
        worker_limit=lambda: requested["value"],
        worker_status=worker_status,
    )

    scheduling = report["worker_scheduling"]
    assert scheduling["initial_worker_count"] == 1
    assert scheduling["maximum_worker_count"] == 6
    assert scheduling["dynamic_worker_limit"] is True
    assert [row["requested_workers"] for row in scheduling["history"]] == [1, 4, 2]
    assert any(desired == 4 for _active, desired in statuses)
    assert statuses[-1] == (0, 2)


def test_parallel_read_validation_has_no_machine_specific_fixed_ceiling() -> None:
    IntegrityScanOptions(parallel_reads=64).validate()
    with pytest.raises(ValueError, match="positive"):
        IntegrityScanOptions(parallel_reads=0).validate()


def test_integrity_worker_recommendation_uses_cpu_and_available_memory() -> None:
    large_host = recommend_integrity_read_workers(
        {
            "cpu": {"logical_cores": 32, "physical_cores": 16},
            "memory": {"available_bytes": 256 * 1024**3},
        }
    )
    assert large_host["recommended"] == 8
    assert large_host["maximum"] == 64

    memory_limited_host = recommend_integrity_read_workers(
        {
            "cpu": {"logical_cores": 8, "physical_cores": 4},
            "memory": {"available_bytes": 2 * 1024**3},
        }
    )
    assert memory_limited_host["recommended"] == 2
    assert memory_limited_host["maximum"] == 4


def test_run_reference_records_do_not_scan_nested_files_twice(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Run sources",
        series_id="run-sources",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    run = tmp_path / "series" / "AC17"
    injection = run / "CO2 injection"
    clearwater = run / "BTB flush"
    overflow = run / "overflow"
    for folder in (injection, clearwater, overflow):
        folder.mkdir(parents=True)
    baseline = injection / "frame001.jpg"
    clear = clearwater / "clear.jpg"
    scale = overflow / "scale.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(baseline)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(clear)
    Image.new("RGB", (8, 8), (70, 80, 90)).save(scale)
    table_a = overflow / "overflow-a.csv"
    table_b = overflow / "overflow-b.xlsx"
    table_a.write_text("time_h,weight_g\n0,10\n", encoding="utf-8")
    table_b.write_bytes(b"workbook")
    project.set_runs(
        [
            {
                "run_id": "ac17",
                "source_mapping": {
                    "run_id": "ac17",
                    "experiment_folder": str(run),
                    "experiment_folders": [str(injection)],
                    "image_count": 1,
                    "baseline_candidate": str(baseline),
                    "clearwater_candidate": str(clear),
                    "clearwater_folders": [str(clearwater)],
                    "overflow_image_folders": [str(overflow)],
                    "overflow_tables": [str(table_a), str(table_b)],
                    "overflow_table": str(table_a),
                    "candidate_status": "user_confirmed",
                },
            }
        ],
        updated_by="Reviewer",
    )

    report = IntegrityScanner(project).scan(actor="Reviewer")
    paths = [row["path"] for row in report["inventory"]]

    assert len(paths) == 5
    assert len(set(paths)) == 5
    assert any(
        source.get("scan_policy") == "reference_only"
        for source in project.source_manifest()["sources"]
    )


def test_overflow_screenshots_are_not_treated_as_camera_sequences(
    tmp_path: Path,
) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Overflow integrity semantics",
        series_id="overflow-integrity-semantics",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    run = tmp_path / "series" / "AC17"
    injection = run / "CO2 injection"
    overflow = run / "overflow"
    injection.mkdir(parents=True)
    overflow.mkdir(parents=True)
    baseline = injection / "frame001.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(baseline)
    first = overflow / "snapshot_2026-09-03-10-00-01.bmp"
    second = overflow / "snapshot_2026-09-03-10-00-59.bmp"
    Image.new("RGB", (8, 8), (40, 50, 60)).save(first)
    shutil.copyfile(first, second)
    project.set_runs(
        [
            {
                "run_id": "ac17",
                "source_mapping": {
                    "run_id": "ac17",
                    "experiment_folder": str(run),
                    "experiment_folders": [str(injection)],
                    "image_count": 1,
                    "baseline_candidate": str(baseline),
                    "overflow_image_folders": [str(overflow)],
                    "candidate_status": "user_confirmed",
                },
            }
        ],
        updated_by="Reviewer",
    )
    overflow_source = next(
        source
        for source in project.current_source_records()
        if source.get("kind") == "overflow_image_folder"
    )

    report = IntegrityScanner(project).scan(
        source_ids=[str(overflow_source["source_id"])],
        actor="Reviewer",
    )

    assert report["summary"]["image_count"] == 2
    assert report["issues"] == []


def test_series_container_and_excluded_runs_are_not_scanned(tmp_path: Path) -> None:
    series = tmp_path / "series"
    included = series / "AC17" / "CO2 injection"
    excluded = series / "AC14" / "CO2 injection"
    included.mkdir(parents=True)
    excluded.mkdir(parents=True)
    included_image = included / "frame001.jpg"
    excluded_image = excluded / "frame001.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(included_image)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(excluded_image)

    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Selected runs",
        series_id="selected-runs",
        template_id="ff_ac",
        rig_type="medium",
        image_sources={"experiment_images": series},
        created_by="Reviewer",
    )
    project.set_runs(
        [
            {
                "run_id": "ac17",
                "source_mapping": {
                    "run_id": "ac17",
                    "experiment_folder": str(series / "AC17"),
                    "experiment_folders": [str(included)],
                    "image_count": 1,
                    "baseline_candidate": str(included_image),
                    "candidate_status": "user_confirmed",
                },
            }
        ],
        updated_by="Reviewer",
    )

    current_sources = project.current_source_records()
    container = next(
        source
        for source in current_sources
        if source.get("kind") == "experiment_images" and not source.get("run_id")
    )
    assert container["scan_policy"] == "reference_only"
    assert container["effective_role"] == "series_container"

    report = IntegrityScanner(project).scan(actor="Reviewer")
    paths = {row["path"] for row in report["inventory"]}

    assert str(included_image.resolve()) in paths
    assert str(excluded_image.resolve()) not in paths
    assert all("AC14" not in path for path in paths)


def test_superseded_run_sources_are_not_scanned(tmp_path: Path) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Current source revision",
        series_id="current-source-revision",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    run = tmp_path / "series" / "AC17"
    old_images = run / "old"
    current_images = run / "current"
    old_images.mkdir(parents=True)
    current_images.mkdir(parents=True)
    old_image = old_images / "frame001.jpg"
    current_image = current_images / "frame001.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(old_image)
    Image.new("RGB", (8, 8), (40, 50, 60)).save(current_image)

    def mapping(folder: Path, image: Path) -> dict[str, object]:
        return {
            "run_id": "ac17",
            "experiment_folder": str(run),
            "experiment_folders": [str(folder)],
            "image_count": 1,
            "baseline_candidate": str(image),
            "candidate_status": "user_confirmed",
        }

    project.set_runs(
        [{"run_id": "ac17", "source_mapping": mapping(old_images, old_image)}],
        updated_by="Reviewer",
    )
    project.update_run_sources(
        "ac17",
        mapping(current_images, current_image),
        updated_by="Reviewer",
    )

    report = IntegrityScanner(project).scan(actor="Reviewer")
    paths = {row["path"] for row in report["inventory"]}

    assert str(current_image.resolve()) in paths
    assert str(old_image.resolve()) not in paths


def test_integrity_store_streams_camera_sequence_endpoints(tmp_path: Path) -> None:
    project, _images = create_project(tmp_path)
    report = IntegrityScanner(project).scan(actor="Reviewer")
    store = IntegrityReportStore(project)
    store.save(report, actor="Reviewer")

    endpoints = store.camera_sequence_endpoints()

    assert len(endpoints) == 1
    assert endpoints[0]["image_count"] == 5
    assert Path(endpoints[0]["first"]["path"]).name == "frame001.jpg"
    assert Path(endpoints[0]["last"]["path"]).name == "frame006.jpg"
