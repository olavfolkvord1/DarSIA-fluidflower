from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from darsia.fluidflower.modules import MODULE_BY_KEY, available_module_keys
from darsia.fluidflower.project import (
    FluidFlowerProject,
    ProjectRegistry,
    sha256_file,
)
from darsia.fluidflower.templates import template_by_id


def create_project(tmp_path: Path) -> FluidFlowerProject:
    protocol = tmp_path / "protocol.csv"
    protocol.write_text("time_h,rate_ml_min\n0,2\n", encoding="utf-8")
    images = tmp_path / "raw-images"
    images.mkdir()
    (images / "frame001.jpg").write_bytes(b"unchanged-source-image")
    return FluidFlowerProject.create(
        tmp_path / "project",
        name="AC test series",
        series_id="AC Test",
        template_id="ff_ac",
        rig_type="medium",
        protocol_path=protocol,
        image_sources={"experiment_images": images},
        overflow_mode="none",
        created_by="Olav Folkvord",
    )


def create_run_mapping(tmp_path: Path, run_id: str) -> dict[str, object]:
    run_root = tmp_path / "experiments" / run_id.upper()
    injection = run_root / "CO2 injection"
    clearwater = run_root / "BTB flush"
    indicator_references = run_root / "pH buffers"
    overflow = run_root / "overflow"
    for folder in (injection, clearwater, indicator_references, overflow):
        folder.mkdir(parents=True, exist_ok=True)
    baseline = injection / "frame001.jpg"
    baseline.write_bytes(b"baseline")
    (injection / "frame002.jpg").write_bytes(b"co2")
    clear = clearwater / "clear.jpg"
    clear.write_bytes(b"clear")
    (indicator_references / "pH_7.2.jpg").write_bytes(b"buffer")
    (overflow / "scale.jpg").write_bytes(b"scale")
    overflow_a = overflow / "overflow-a.csv"
    overflow_b = overflow / "overflow-b.xlsx"
    overflow_a.write_text("time_h,weight_g\n0,100\n", encoding="utf-8")
    overflow_b.write_bytes(b"workbook")
    protocol = run_root / "experimental_protocol.xlsx"
    protocol.write_bytes(b"protocol")
    return {
        "run_id": run_id,
        "experiment_folder": str(run_root),
        "experiment_folders": [str(injection)],
        "co2_image_folders": [str(injection)],
        "co2_sequences": [
            {
                "order": 1,
                "kind": "injection",
                "folder": str(injection),
                "image_count": 2,
                "first_image": str(baseline),
                "last_image": str(injection / "frame002.jpg"),
            }
        ],
        "image_count": 2,
        "baseline_candidate": str(baseline),
        "clearwater_candidate": str(clear),
        "clearwater_folders": [str(clearwater)],
        "indicator_reference_folders": [str(indicator_references)],
        "protocol_tables": [str(protocol)],
        "protocol_table": str(protocol),
        "overflow_image_folders": [str(overflow)],
        "overflow_tables": [str(overflow_a), str(overflow_b)],
        "overflow_table": str(overflow_a),
        "candidate_status": "user_confirmed",
        "mapping_status": "ready_for_review",
    }


def test_known_templates_keep_ffum_protocol_schema() -> None:
    assert template_by_id("ff_um").protocol_template == "fluidflower_timeline_v2"
    assert template_by_id("ff_ac").protocol_template == "fluidflower_timeline_v2"
    assert template_by_id("ff_ac").default_rig_type == "medium"


def test_project_creation_registers_but_does_not_copy_or_modify_sources(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.csv"
    protocol.write_text("time_h,rate_ml_min\n0,2\n", encoding="utf-8")
    source_hash = sha256_file(protocol)
    images = tmp_path / "images"
    images.mkdir()
    image = images / "frame001.jpg"
    image.write_bytes(b"raw-image")

    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Series",
        series_id="series",
        template_id="ff_ac",
        rig_type="medium",
        protocol_path=protocol,
        image_sources={"experiment_images": images},
        created_by="Reviewer",
    )

    sources = project.source_manifest()["sources"]
    assert len(sources) == 2
    protocol_record = next(row for row in sources if row["kind"] == "protocol")
    assert protocol_record["sha256"] == source_hash
    assert protocol.read_text(encoding="utf-8") == "time_h,rate_ml_min\n0,2\n"
    assert image.read_bytes() == b"raw-image"
    assert not (project.root / "raw-images").exists()
    assert project.manifest["modules"]["setup"]["status"] == "approved"
    assert project.manifest["modules"]["data_integrity"]["status"] == "ready"


def test_project_keeps_default_results_below_portable_project_root(
    tmp_path: Path,
) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "projects" / "series",
        name="Storage test",
        series_id="storage-test",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )

    assert project.default_result_root() == project.root / "results"
    assert project.manifest["storage"]["default_result_root"] == {
        "scope": "project_relative",
        "path": "results",
    }

    moved_root = tmp_path / "moved" / "series"
    moved_root.parent.mkdir()
    project.root.rename(moved_root)
    reloaded = FluidFlowerProject.load(moved_root)
    assert reloaded.default_result_root() == moved_root.resolve() / "results"


def test_legacy_project_without_storage_contract_uses_results_folder(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    project.manifest.pop("storage")
    project._write_manifest()

    reloaded = FluidFlowerProject.load(project.root)
    assert reloaded.default_result_root() == project.root / "results"


def test_apparatus_class_update_is_versioned_and_reopens_geometry_review(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    project.update_metadata(
        "rig",
        {"depth": {"source": "measured-depth-map.npz"}},
        updated_by="Reviewer",
    )
    project.set_runs(
        [{"run_id": "ac17", "source_mapping": create_run_mapping(tmp_path, "ac17")}],
        updated_by="Reviewer",
    )
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    project.set_module_status("image_correction", "approved", updated_by="Reviewer")
    project.set_module_status("geometry_facies", "approved", updated_by="Reviewer")
    project.set_run_module_status(
        ["ac17"], "data_integrity", "approved", updated_by="Reviewer"
    )
    project.set_run_module_status(
        ["ac17"], "image_correction", "approved", updated_by="Reviewer"
    )
    project.set_run_module_status(
        ["ac17"], "geometry_facies", "approved", updated_by="Reviewer"
    )

    report = project.update_rig_type(
        "large", updated_by="Reviewer", reason="Corrected setup selection."
    )

    assert report["previous_rig_type"] == "medium"
    assert report["rig_type"] == "large"
    assert report["class_revision"] == 2
    assert project.manifest["rig"]["depth"] == {"source": "measured-depth-map.npz"}
    assert project.manifest["rig"]["class_history"][0]["type"] == "medium"
    assert project.module_states["geometry_facies"]["status"] == "review_required"
    assert (
        project.manifest["runs"][0]["workflow"]["geometry_facies"]["status"]
        == "review_required"
    )
    assert FluidFlowerProject.load(project.root).manifest["rig"]["type"] == "large"


def test_incremental_run_import_preserves_existing_runs_and_reopens_integrity(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    first = create_run_mapping(tmp_path, "ac17")
    project.set_runs(
        [{"run_id": "ac17", "label": "AC17", "source_mapping": first}],
        updated_by="Reviewer",
    )
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    result = project.root / "results" / "existing-result.txt"
    result.write_text("do not replace", encoding="utf-8")
    old_run = deepcopy(project.manifest["runs"][0])
    old_states = deepcopy(project.module_states)

    second = create_run_mapping(tmp_path, "ac18")
    report = project.append_runs(
        [second], updated_by="Reviewer", reason="Later experiment"
    )

    assert report["run_ids"] == ["ac18"]
    assert [row["run_id"] for row in project.manifest["runs"]] == ["ac17", "ac18"]
    assert project.manifest["runs"][0] == old_run
    assert old_states["data_integrity"]["status"] == "approved"
    assert project.module_states["data_integrity"]["status"] == "ready"
    assert report["global_module_states_changed"] is True
    assert "data_integrity" in report["global_modules_invalidated"]
    assert result.read_text(encoding="utf-8") == "do not replace"
    added = project.manifest["runs"][1]
    assert added["workflow"]["data_integrity"]["status"] == "ready"
    assert added["workflow"]["phase_interpretation"]["status"] == "not_started"
    assert len(added["source_mapping"]["source_record_ids"]) >= 7
    assert (project.root / "runs" / "ac18" / "run.json").is_file()
    assert (project.root / "runs" / "ac18" / "sources.json").is_file()
    sources = project.source_manifest()["sources"]
    overflow_tables = [
        row
        for row in sources
        if row.get("run_id") == "ac18" and row.get("kind") == "overflow_table"
    ]
    assert len(overflow_tables) == 2
    run_protocols = [
        row
        for row in sources
        if row.get("run_id") == "ac18" and row.get("kind") == "protocol"
    ]
    assert len(run_protocols) == 1
    indicator_sources = [
        row
        for row in sources
        if row.get("run_id") == "ac18"
        and row.get("kind") == "indicator_reference_image_folder"
    ]
    assert len(indicator_sources) == 1
    assert project.manifest["protocol"]["runs_with_protocol_workbooks"] == [
        "ac17",
        "ac18",
    ]
    transaction = (
        project.root / "history" / "run-source-changes" / report["revision_id"]
    )
    assert json.loads((transaction / "transaction.json").read_text())["status"] == (
        "committed"
    )


def test_incremental_run_import_rejects_duplicate_id_or_folder_without_changes(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    first = create_run_mapping(tmp_path, "ac17")
    project.set_runs(
        [{"run_id": "ac17", "source_mapping": first}],
        updated_by="Reviewer",
    )
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    before_manifest = deepcopy(project.manifest)
    before_sources = deepcopy(project.source_manifest())
    duplicate_id = create_run_mapping(tmp_path, "ac18")
    duplicate_id["run_id"] = "ac17"

    with pytest.raises(ValueError, match="Run ID already exists"):
        project.append_runs([duplicate_id], updated_by="Reviewer")

    duplicate_folder = deepcopy(first)
    duplicate_folder["run_id"] = "ac99"
    with pytest.raises(ValueError, match="Run folder is already registered"):
        project.append_runs([duplicate_folder], updated_by="Reviewer")

    assert project.manifest == before_manifest
    assert project.source_manifest() == before_sources


def test_updating_one_run_sources_keeps_previous_mapping_and_other_runs(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    first = create_run_mapping(tmp_path, "ac17")
    second = create_run_mapping(tmp_path, "ac18")
    project.set_runs(
        [
            {"run_id": "ac17", "source_mapping": first},
            {"run_id": "ac18", "source_mapping": second},
        ],
        updated_by="Reviewer",
    )
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    other_run = deepcopy(project.manifest["runs"][1])
    global_states = deepcopy(project.module_states)
    additional = Path(str(first["experiment_folder"])) / "late rest"
    additional.mkdir()
    (additional / "frame003.jpg").write_bytes(b"late")
    updated = deepcopy(first)
    updated["experiment_folders"] = [*first["experiment_folders"], str(additional)]
    updated["image_count"] = 3

    report = project.update_run_sources(
        "ac17", updated, updated_by="Reviewer", reason="Late images recovered"
    )

    changed = project.manifest["runs"][0]
    assert report["source_revision"] == 2
    assert changed["source_revision"] == 2
    assert len(changed["source_history"]) == 1
    assert changed["source_history"][0]["source_mapping"]["image_count"] == 2
    assert changed["source_mapping"]["image_count"] == 3
    assert project.manifest["runs"][1] == other_run
    assert global_states["data_integrity"]["status"] == "approved"
    assert project.module_states["data_integrity"]["status"] == "ready"
    assert report["global_module_states_changed"] is True
    assert project.manifest["tools"]["source_mapping"]["rows"][0]["image_count"] == 3


def test_shared_folder_structure_versions_only_selected_run_mappings(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    first = create_run_mapping(tmp_path, "ac17")
    second = create_run_mapping(tmp_path, "ac18")
    project.set_runs(
        [
            {"run_id": "ac17", "source_mapping": first},
            {"run_id": "ac18", "source_mapping": second},
        ],
        updated_by="Reviewer",
    )
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    second_before = deepcopy(project.manifest["runs"][1])
    raw_before = (
        Path(str(first["experiment_folders"][0])) / "frame001.jpg"
    ).read_bytes()
    continuation = Path(str(first["experiment_folder"])) / "resting"
    continuation.mkdir()
    (continuation / "frame003.jpg").write_bytes(b"resting")
    remapped = deepcopy(first)
    remapped["experiment_folders"] = [
        *first["experiment_folders"],
        str(continuation),
    ]
    remapped["image_count"] = 3
    template = {
        "schema": 1,
        "format": "darsia-fluidflower-run-layout-v1",
        "example_run_name": "AC17",
        "folders": [
            {
                "relative_path": "CO2 injection",
                "signature": "co_injection",
                "role": "co2_injection_images",
                "sequence_order": 1,
            },
            {
                "relative_path": "resting",
                "signature": "resting",
                "role": "experiment_images",
                "sequence_order": 2,
            },
        ],
    }

    report = project.apply_run_layout_template(
        [remapped],
        template,
        updated_by="Reviewer",
        reason="Reviewed common run structure.",
    )

    first_after = project.manifest["runs"][0]
    assert report["template_revision"] == 1
    assert report["changed_run_ids"] == ["ac17"]
    assert report["global_module_states_changed"] is True
    assert project.module_states["data_integrity"]["status"] == "ready"
    assert first_after["source_revision"] == 2
    assert first_after["source_mapping"]["layout_template_revision"] == 1
    assert first_after["source_mapping"]["image_count"] == 3
    assert first_after["source_history"][0]["source_mapping"]["image_count"] == 2
    assert project.manifest["runs"][1] == second_before
    assert (
        project.manifest["tools"]["source_mapping"]["run_layout_template"] == template
    )
    assert (
        project.manifest["tools"]["source_mapping"]["run_layout_template_revision"] == 1
    )
    assert (
        Path(str(first["experiment_folders"][0])) / "frame001.jpg"
    ).read_bytes() == raw_before

    reloaded = FluidFlowerProject.load(project.root)
    assert reloaded.manifest["runs"][0]["source_revision"] == 2
    assert reloaded.manifest["runs"][1] == second_before


def test_run_quality_gate_update_does_not_invalidate_other_runs_or_global_gate(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    first = create_run_mapping(tmp_path, "ac17")
    second = create_run_mapping(tmp_path, "ac18")
    project.set_runs(
        [
            {"run_id": "ac17", "source_mapping": first},
            {"run_id": "ac18", "source_mapping": second},
        ],
        updated_by="Reviewer",
    )
    other_run = deepcopy(project.manifest["runs"][1])
    global_states = deepcopy(project.module_states)

    project.set_run_module_status(
        ["ac17"],
        "data_integrity",
        "review_required",
        updated_by="Reviewer",
    )
    project.set_run_module_status(
        ["ac17"],
        "data_integrity",
        "approved",
        updated_by="Reviewer",
    )

    assert (
        project.manifest["runs"][0]["workflow"]["data_integrity"]["status"]
        == "approved"
    )
    assert project.manifest["runs"][1] == other_run
    assert project.module_states == global_states


def test_quality_gate_change_invalidates_completed_downstream_modules(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    for key in (
        "data_integrity",
        "overflow",
        "temporal_sync",
        "image_correction",
        "geometry_facies",
        "optical_calibration",
        "phase_interpretation",
        "calibration",
        "mass_analysis",
        "post_analysis",
    ):
        project.set_module_status(key, "complete", updated_by="Reviewer")

    invalidated = project.set_module_status(
        "image_correction",
        "in_progress",
        updated_by="Reviewer",
        reason="New ColorChecker correction",
    )

    assert "geometry_facies" in invalidated
    assert "optical_calibration" in invalidated
    assert "phase_interpretation" in invalidated
    assert "calibration" in invalidated
    assert "mass_analysis" in invalidated
    assert "post_analysis" in invalidated
    assert project.module_states["mass_analysis"]["status"] == "stale"
    assert project.module_states["geometry_facies"]["status"] == "stale"


def test_available_modules_follow_dependency_gates(tmp_path: Path) -> None:
    project = create_project(tmp_path)
    assert available_module_keys(project.module_states) == (
        "setup",
        "data_integrity",
    )
    project.set_module_status("data_integrity", "approved")
    available = available_module_keys(project.module_states)
    assert "overflow" in available
    assert "image_correction" in available
    assert "geometry_facies" not in available
    assert "temporal_sync" not in available
    assert "phase_interpretation" not in available
    project.set_module_status("overflow", "approved")
    assert "temporal_sync" in available_module_keys(project.module_states)
    project.set_module_status("image_correction", "approved")
    available = available_module_keys(project.module_states)
    assert "geometry_facies" in available


def test_project_rejects_approval_before_dependencies_pass(tmp_path: Path) -> None:
    project = create_project(tmp_path)

    with pytest.raises(ValueError, match="Image correction"):
        project.set_module_status("geometry_facies", "approved")

    project.set_module_status("data_integrity", "approved")
    project.set_module_status("image_correction", "approved")
    project.set_module_status("geometry_facies", "approved")
    assert project.module_states["geometry_facies"]["status"] == "approved"


def test_registry_is_only_an_index_and_project_remains_authoritative(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    registry = ProjectRegistry(tmp_path / "registry.json")
    registry.remember(project)

    rows = registry.entries()
    assert len(rows) == 1
    assert Path(rows[0]["path"]) == project.root
    loaded = FluidFlowerProject.load(Path(rows[0]["path"]))
    assert loaded.project_id == project.project_id

    registry_payload = json.loads(registry.path.read_text(encoding="utf-8"))
    assert "modules" not in registry_payload["projects"][0]


def test_module_registry_contains_complete_requested_workflow() -> None:
    assert set(MODULE_BY_KEY) == {
        "setup",
        "data_integrity",
        "overflow",
        "geometry_facies",
        "image_correction",
        "temporal_sync",
        "optical_calibration",
        "phase_interpretation",
        "calibration",
        "mass_analysis",
        "wasserstein",
        "post_analysis",
    }
    assert [definition.key for definition in MODULE_BY_KEY.values()] == [
        "setup",
        "data_integrity",
        "overflow",
        "temporal_sync",
        "image_correction",
        "geometry_facies",
        "optical_calibration",
        "phase_interpretation",
        "calibration",
        "mass_analysis",
        "wasserstein",
        "post_analysis",
    ]
