from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from darsia.fluidflower.overflow import (
    OverflowRevisionStore,
    apply_registered_overflow_mode,
    default_overflow_config,
    normalize_overflow_config,
    scan_overflow_run,
    validate_overflow_config,
)
from darsia.fluidflower.overflow_workflow import (
    scale_reader_review_progress,
    write_scale_reader_config,
)
from darsia.fluidflower.pages.overflow.page import OverflowPage
from darsia.fluidflower.project import FluidFlowerProject
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _project_with_overflow_sources(
    tmp_path: Path,
) -> tuple[FluidFlowerProject, Path, Path, Path]:
    raw = tmp_path / "raw" / "AC52"
    scale_a = raw / "overflow-a"
    scale_b = raw / "overflow-b"
    injection = raw / "co2"
    for folder in (scale_a, scale_b, injection):
        folder.mkdir(parents=True)
    Image.new("RGB", (40, 24), (30, 40, 50)).save(injection / "2026-08-01_09-00-00.jpg")
    Image.new("RGB", (40, 24), (100, 110, 120)).save(
        scale_a / "2026-08-01_09-05-00.jpg"
    )
    Image.new("RGB", (40, 24), (120, 110, 100)).save(
        scale_b / "2026-08-01_10-05-00.jpg"
    )
    table = raw / "overflow-reviewed.csv"
    table.write_text(
        "experiment_time_h,weight_g,validated\n0,812.4,true\n1,799.1,true\n",
        encoding="utf-8",
    )
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Overflow test",
        series_id="overflow-test",
        template_id="ff_ac",
        rig_type="medium",
        image_sources={"experiment_images": raw.parent},
        created_by="Olav Folkvord",
    )
    project.set_runs(
        [
            {
                "run_id": "ac52",
                "label": "AC52",
                "source_mapping": {
                    "run_id": "ac52",
                    "experiment_folder": str(raw),
                    "experiment_folders": [str(injection)],
                    "baseline_candidate": str(injection / "2026-08-01_09-00-00.jpg"),
                    "overflow_image_folders": [str(scale_a), str(scale_b)],
                    "overflow_tables": [str(table)],
                    "overflow_table": str(table),
                    "candidate_status": "user_confirmed",
                },
            }
        ],
        updated_by="Olav Folkvord",
    )
    return project, scale_a, scale_b, table


def _write_scale_review(project: FluidFlowerProject, run_id: str = "ac52") -> Path:
    output = (
        project.root
        / "results"
        / "overflow"
        / "scale_review"
        / run_id
        / f"{run_id}_overflow_scale_review.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": 1,
                "metadata": {
                    "run": run_id,
                    "updated_at": "2026-09-03T20:54:40",
                },
                "records": [
                    {
                        "status": "Corrected",
                        "selected_for_review": True,
                        "time_from_injection_h": -0.25,
                        "image_time": "2023-07-11T10:02:57",
                        "reviewer": "Olav Folkvord",
                    },
                    {
                        "status": "Partial",
                        "selected_for_review": True,
                        "time_from_injection_h": 4.34,
                        "image_time": "2023-07-11T14:38:46",
                        "reviewer": "Olav Folkvord",
                    },
                    {
                        "status": "Pending",
                        "selected_for_review": True,
                        "time_from_injection_h": 8.0,
                        "image_time": "2023-07-11T18:18:10",
                    },
                    {
                        "status": "Not selected",
                        "selected_for_review": False,
                        "time_from_injection_h": 9.0,
                        "image_time": "2023-07-11T19:18:10",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def test_default_selection_prefers_discovered_scale_images_and_scans_all_roots(
    tmp_path: Path,
) -> None:
    project, scale_a, scale_b, _table = _project_with_overflow_sources(tmp_path)

    config = default_overflow_config(project)
    row = config["runs"][0]
    assert row["mode"] == "scale_images"
    assert row["image_folders"] == [str(scale_a.resolve()), str(scale_b.resolve())]

    scan = scan_overflow_run(row)
    assert scan["status"] == "ready"
    assert scan["image_count"] == 2
    assert scan["timestamped_image_count"] == 2


def test_overflow_revision_can_switch_from_images_to_reviewed_table(
    tmp_path: Path,
) -> None:
    project, _scale_a, _scale_b, table = _project_with_overflow_sources(tmp_path)
    store = OverflowRevisionStore(project)
    config = default_overflow_config(project)
    config["runs"][0]["scan"] = scan_overflow_run(config["runs"][0])

    try:
        store.save_revision(config, actor="Olav Folkvord", reason="Too early")
    except ValueError as exc:
        assert "Data and integrity" in str(exc)
    else:
        raise AssertionError("Overflow save must require approved data integrity")

    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    store.save_draft(config, actor="Olav Folkvord", reason="Ready to review")
    first = store.save_revision(
        config, actor="Olav Folkvord", reason="Use camera images"
    )
    assert first["configuration"]["runs"][0]["mode"] == "scale_images"
    assert not store.has_draft()
    assert (
        project.manifest["runs"][0]["workflow"]["overflow"]["status"]
        == "review_required"
    )

    updated = normalize_overflow_config(project, store.current_config())
    updated_row = updated["runs"][0]
    updated_row["mode"] = "table"
    updated_row["table_path"] = str(table)
    updated_row["scan"] = scan_overflow_run(updated_row)
    second = store.save_revision(
        updated, actor="Olav Folkvord", reason="Use reviewed table"
    )

    assert second["revision_id"] != first["revision_id"]
    assert second["configuration"]["runs"][0]["mode"] == "table"
    assert second["configuration"]["runs"][0]["scan"]["row_count"] == 2
    assert len(store.revisions()) == 2
    assert store.revisions()[0]["configuration"]["runs"][0]["mode"] == ("scale_images")
    assert project.manifest["overflow"]["source_modes_by_run"] == {"ac52": "table"}


def test_scale_reader_adapter_preserves_multiple_image_roots(tmp_path: Path) -> None:
    project, scale_a, scale_b, _table = _project_with_overflow_sources(tmp_path)
    config = default_overflow_config(project)
    config["runs"][0]["scan"] = scan_overflow_run(config["runs"][0])

    path = write_scale_reader_config(project, config, run_ids=["ac52"])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["runs"]["ac52"]["image_roots"] == [
        str(scale_a.resolve()),
        str(scale_b.resolve()),
    ]
    assert payload["runs"]["ac52"]["injection_start"].startswith("2026-08-01")


def test_scale_reader_progress_reports_manual_count_and_reviewed_time_span(
    tmp_path: Path,
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    _write_scale_review(project)

    progress = scale_reader_review_progress(project.root, "AC52")

    assert progress["reviewed_count"] == 2
    assert progress["selected_count"] == 3
    assert progress["exact_count"] == 1
    assert progress["partial_count"] == 1
    assert progress["unreadable_count"] == 0
    assert progress["first_time_h"] == -0.25
    assert progress["last_time_h"] == 4.34
    assert progress["first_image_time"] == "2023-07-11T10:02:57"
    assert progress["last_image_time"] == "2023-07-11T14:38:46"
    assert progress["reviewers"] == ["Olav Folkvord"]


def test_none_is_an_explicit_valid_per_run_decision(tmp_path: Path) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    config = default_overflow_config(project)
    row = config["runs"][0]
    row["mode"] = "none"
    row["scan"] = {}

    report = validate_overflow_config(config, expected_run_ids=["ac52"])

    assert report["summary"]["error_count"] == 0
    assert report["summary"]["active_run_count"] == 0


def test_approved_no_overflow_revision_satisfies_run_quality_gate(
    qt_app: QApplication, tmp_path: Path
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    project.set_run_module_status(
        ["ac52"],
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    config = default_overflow_config(project)
    config["runs"][0].update({"mode": "none", "scan": {}})
    OverflowRevisionStore(project).save_revision(
        config,
        actor="Olav Folkvord",
        reason="No overflow measurements exist for this experiment.",
    )

    page = OverflowPage(project)
    page._set_status("complete")

    assert project.module_states["overflow"]["status"] == "complete"
    assert (
        project.manifest["runs"][0]["workflow"]["overflow"]["status"]
        == "complete"
    )
    page.close()


def test_none_is_presented_as_resolved_and_excluded_from_scans(
    qt_app: QApplication, tmp_path: Path
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    page = OverflowPage(project)
    row = page.config["runs"][0]
    row["mode"] = "none"
    row["source_origin"] = "individual_override"
    row["scan"] = {}

    page.set_overflow_scope("global")
    page._populate_table("ac52")
    assert not page.scan_selected.isEnabled()
    assert not page.scan_all.isEnabled()

    page.set_overflow_scope("individual")
    page._populate_table("ac52")

    assert page.run_table.item(0, 3).text() == "Not applicable"
    assert "no source scan is required" in page.run_table.item(0, 7).text()
    assert page.scan_progress.value() == page.scan_progress.maximum()
    assert "source scan is not required" in page.scan_message.text()
    assert "valid, explicit decision" in page.scan_report.toPlainText()
    assert page._scan_targets(True) == []
    assert page._scan_targets(False) == []
    assert not page.scan_selected.isEnabled()
    assert not page.scan_all.isEnabled()
    page.close()


def test_bulk_source_policy_uses_each_runs_registered_table_and_skips_missing(
    tmp_path: Path,
) -> None:
    project, _scale_a, _scale_b, table = _project_with_overflow_sources(tmp_path)
    config = default_overflow_config(project)
    missing = deepcopy(config["runs"][0])
    missing.update(
        {
            "run_id": "ac53",
            "mode": "scale_images",
            "discovered_tables": [],
            "table_path": "",
        }
    )
    config["runs"].append(missing)

    updated, summary = apply_registered_overflow_mode(
        config, mode="table", run_ids=["ac52", "ac53"]
    )

    rows = {row["run_id"]: row for row in updated["runs"]}
    assert rows["ac52"]["mode"] == "table"
    assert rows["ac52"]["table_path"] == str(table.resolve())
    assert rows["ac52"]["source_origin"] == "global_registered_source"
    assert rows["ac53"]["mode"] == "scale_images"
    assert summary["changed_run_ids"] == ["ac52"]
    assert "ac53" in summary["skipped"]


def test_normalization_refreshes_registered_sources_without_replacing_active_choice(
    tmp_path: Path,
) -> None:
    project, scale_a, _scale_b, table = _project_with_overflow_sources(tmp_path)
    config = default_overflow_config(project)
    config["runs"][0].update(
        {
            "mode": "scale_images",
            "image_folders": [str(scale_a)],
            "discovered_tables": [],
        }
    )

    normalized = normalize_overflow_config(project, config)

    row = normalized["runs"][0]
    assert row["mode"] == "scale_images"
    assert row["image_folders"] == [str(scale_a)]
    assert row["discovered_tables"] == [str(table.resolve())]


def test_incomplete_overflow_draft_is_restored_without_creating_a_revision(
    tmp_path: Path,
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    store = OverflowRevisionStore(project)
    config = default_overflow_config(project)
    config["runs"][0]["scan"] = scan_overflow_run(config["runs"][0])

    store.save_draft(config, actor="Olav Folkvord", reason="Scanned AC52")

    assert store.has_draft()
    assert store.revisions() == []
    restored = store.current_config()
    assert restored["runs"][0]["scan"]["image_count"] == 2


def test_overflow_page_exposes_per_run_source_switching(
    qt_app: QApplication, tmp_path: Path
) -> None:
    project, _scale_a, _scale_b, table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )

    page = OverflowPage(project)
    assert page.run_table.rowCount() == 1
    assert page.mode.currentData() == "scale_images"
    assert page.image_folders.count() == 2

    page.mode.setCurrentIndex(page.mode.findData("table"))
    page.table_path.setText(str(table))
    row = page.config["runs"][0]
    assert row["mode"] == "table"
    assert row["table_path"] == str(table)
    assert row["scan"] == {}
    page.close()


def test_overflow_page_bulk_assigns_registered_tables_and_keeps_individual_editor(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _scale_a, _scale_b, table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    page = OverflowPage(project)
    assert not page.global_group.isHidden()
    assert page.individual_splitter.isHidden()
    assert page.global_table.rowCount() == 1
    assert page.global_table.item(0, 0).checkState() == Qt.CheckState.Checked
    assert str(table.resolve()) in page.global_table.item(0, 3).toolTip()

    page._apply_global_source()

    assert page.config["runs"][0]["mode"] == "table"
    assert page.config["runs"][0]["table_path"] == str(table.resolve())
    assert page.store.has_draft()
    page.set_overflow_scope("individual")
    assert page.global_group.isHidden()
    assert not page.individual_splitter.isHidden()
    page.close()


def test_overflow_run_list_refreshes_reader_progress_when_reader_closes(
    qt_app: QApplication, tmp_path: Path
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )
    page = OverflowPage(project)
    assert page.run_table.horizontalHeaderItem(5).text() == "Reviewed images"
    assert page.run_table.horizontalHeaderItem(6).text() == "Reviewed time span"
    assert page.run_table.item(0, 5).text() == "-"

    _write_scale_review(project)
    page._reader_closed()

    assert page.run_table.item(0, 5).text() == "2 / 3"
    assert page.run_table.item(0, 6).text() == "-0.25 to 4.34 h"
    tooltip = page.run_table.item(0, 5).toolTip()
    assert "Exact readings: 1" in tooltip
    assert "Uncertain readings: 1" in tooltip
    assert "Olav Folkvord" in tooltip
    page.close()


def test_overflow_progress_describes_the_selected_run(
    qt_app: QApplication, tmp_path: Path
) -> None:
    project, _scale_a, _scale_b, _table = _project_with_overflow_sources(tmp_path)
    project.set_module_status(
        "data_integrity",
        "approved",
        updated_by="Olav Folkvord",
        reason="Test sources reviewed",
    )

    page = OverflowPage(project)
    scanned = page.config["runs"][0]
    scanned["scan"] = scan_overflow_run(scanned)
    pending = deepcopy(scanned)
    pending["run_id"] = "ac53"
    pending["scan"] = {}
    page.config["runs"].append(pending)

    page._populate_table("ac52")
    assert page.scan_progress.value() == page.scan_progress.maximum()
    assert "AC52" in page.scan_message.text()
    assert "2 images" in page.scan_message.text()

    page._populate_table("ac53")
    assert page.scan_progress.value() == 0
    assert page.scan_progress.maximum() == 1
    assert page.scan_message.text() == ("AC53: selected source has not been scanned.")
    page.close()
