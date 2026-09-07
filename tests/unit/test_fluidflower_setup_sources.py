from __future__ import annotations

from pathlib import Path

from darsia.fluidflower.setup_drafts import SetupDraftStore
from darsia.fluidflower.source_discovery import (
    ROLE_ALTERNATE_COLOR_REFERENCE,
    ROLE_CLEARWATER,
    ROLE_EXPERIMENT,
    ROLE_IGNORE,
    ROLE_INDICATOR_REFERENCE,
    ROLE_INJECTION,
    ROLE_OVERFLOW_IMAGES,
    ROLE_OVERFLOW_TABLE,
    ROLE_PROTOCOL_TABLE,
    build_run_layout_template,
    discover_image_source_mapping,
    discover_run_candidates,
    inspect_run_structure,
)


def test_setup_draft_roundtrip_and_delete(tmp_path: Path) -> None:
    store = SetupDraftStore(tmp_path / "drafts")
    saved = store.save(
        {"name": "Series", "source_mapping": [{"run_id": "run18"}]},
        draft_id="draft-1",
    )

    assert saved["draft_id"] == "draft-1"
    assert store.latest()["values"]["name"] == "Series"
    assert store.load("draft-1")["values"]["source_mapping"][0]["run_id"] == "run18"

    store.delete("draft-1")
    assert store.drafts() == []


def test_discard_archives_draft_without_leaving_it_active(tmp_path: Path) -> None:
    store = SetupDraftStore(tmp_path / "drafts")
    store.save({"name": "Incomplete series"}, draft_id="draft-1")

    discarded = store.discard("draft-1")

    assert discarded["draft_id"] == "draft-1"
    assert discarded["discarded_utc"]
    assert store.drafts() == []
    assert (tmp_path / "drafts" / "discarded" / "draft-1.json").is_file()


def test_image_source_discovery_proposes_per_run_references(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    baselines = tmp_path / "baselines"
    clearwater = tmp_path / "clearwater"
    for run_id in ("run18", "run20"):
        (experiments / run_id).mkdir(parents=True)
        (baselines / run_id).mkdir(parents=True)
        (clearwater / run_id).mkdir(parents=True)
        (experiments / run_id / "frame.jpg").write_bytes(b"frame")
        (baselines / run_id / "baseline.jpg").write_bytes(b"baseline")
        (clearwater / run_id / "wash.jpg").write_bytes(b"wash")

    rows = discover_image_source_mapping(
        experiments,
        structure="run_subfolders",
        baseline_root=baselines,
        clearwater_root=clearwater,
    )

    assert [row["run_id"] for row in rows] == ["run18", "run20"]
    assert all(row["image_count"] == 1 for row in rows)
    assert all(row["baseline_candidate"].endswith("baseline.jpg") for row in rows)
    assert all(row["clearwater_candidate"].endswith("wash.jpg") for row in rows)
    assert all(row["candidate_status"] == "requires_user_confirmation" for row in rows)


def test_image_source_discovery_uses_one_confirmed_image_family(tmp_path: Path) -> None:
    injection = tmp_path / "series" / "AC17" / "CO2 injection"
    injection.mkdir(parents=True)
    for stem in ("frame001", "frame002"):
        (injection / f"{stem}.JPG").write_bytes(b"processed")
        (injection / f"{stem}.ARW").write_bytes(b"raw")

    processed = discover_image_source_mapping(tmp_path / "series")[0]
    raw = discover_image_source_mapping(
        tmp_path / "series", image_file_policy="prefer_raw"
    )[0]
    both = discover_image_source_mapping(tmp_path / "series", image_file_policy="all")[
        0
    ]

    assert processed["image_count"] == 2
    assert processed["first_image"].endswith(".JPG")
    assert processed["image_file_counts"] == {"processed": 2, "raw": 2}
    assert raw["image_count"] == 2
    assert raw["first_image"].endswith(".ARW")
    assert both["image_count"] == 4


def test_run_layout_maps_nested_per_run_sources_and_first_image_baseline(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    run17 = experiments / "AC17"
    run18 = experiments / "AC18"
    for run, stamp in ((run17, "240101"), (run18, "240102")):
        injection = run / f"{stamp}_120000_CO2_injection"
        resting = run / f"{stamp}_140000_resting"
        flush = run / f"{stamp}_090000_BTB_flush"
        overflow = run / "overflow_scale"
        for folder in (injection, resting, flush, overflow):
            folder.mkdir(parents=True, exist_ok=True)
        (injection / "frame001.jpg").write_bytes(b"first")
        (injection / "frame002.jpg").write_bytes(b"second")
        (resting / "frame003.jpg").write_bytes(b"rest")
        (flush / "DCS001_clear.jpg").write_bytes(b"clear")
        (flush / "DCS002_blue.jpg").write_bytes(b"blue")
        (overflow / "scale001.jpg").write_bytes(b"scale")
        (overflow / "overflow.xlsx").write_bytes(b"table")
        (run / "experimental_protocol.xlsx").write_bytes(b"protocol")

    inspected = inspect_run_structure(run17)
    assignments = {
        str(row["relative_path"]): str(row["suggested_role"]) for row in inspected
    }
    assert ROLE_EXPERIMENT in assignments.values()
    assert ROLE_INJECTION in assignments.values()
    assert ROLE_CLEARWATER in assignments.values()
    assert ROLE_OVERFLOW_IMAGES in assignments.values()
    table_roles = {
        candidate["suggested_role"]
        for row in inspected
        for candidate in row["table_candidates"]
    }
    assert ROLE_OVERFLOW_TABLE in table_roles
    assert ROLE_PROTOCOL_TABLE in table_roles
    template = build_run_layout_template(run17, assignments=assignments)

    rows = discover_image_source_mapping(
        experiments,
        structure="run_subfolders",
        layout_template=template,
    )

    assert [row["run_id"] for row in rows] == ["ac17", "ac18"]
    assert all(row["image_count"] == 3 for row in rows)
    assert all(row["baseline_candidate"].endswith("frame001.jpg") for row in rows)
    assert all(row["clearwater_candidate"].endswith("DCS001_clear.jpg") for row in rows)
    assert all(row["overflow_image_count"] == 1 for row in rows)
    assert all(row["overflow_table"].endswith("overflow.xlsx") for row in rows)
    assert all(
        row["protocol_table"].endswith("experimental_protocol.xlsx") for row in rows
    )
    assert all(row["mapping_status"] == "ready_for_review" for row in rows)


def test_run_layout_keeps_ph_buffers_separate_from_clearwater(tmp_path: Path) -> None:
    run = tmp_path / "series" / "AC52"
    injection = run / "CO2 injection"
    clearwater = run / "BTB flush"
    buffers = run / "indicator pH buffers"
    alternate = run / "alternate colorstate"
    for folder in (injection, clearwater, buffers, alternate):
        folder.mkdir(parents=True)
    (injection / "frame.jpg").write_bytes(b"co2")
    (clearwater / "clear.jpg").write_bytes(b"clear")
    (buffers / "pH_7.2.jpg").write_bytes(b"buffer")
    (alternate / "red_dye.jpg").write_bytes(b"alternate")

    inspected = inspect_run_structure(run)
    roles = {str(row["relative_path"]): str(row["suggested_role"]) for row in inspected}
    assert roles["BTB flush"] == ROLE_CLEARWATER
    assert roles["indicator pH buffers"] == ROLE_INDICATOR_REFERENCE
    assert roles["alternate colorstate"] == ROLE_ALTERNATE_COLOR_REFERENCE

    row = discover_image_source_mapping(tmp_path / "series")[0]
    assert row["indicator_reference_image_count"] == 1
    assert row["indicator_reference_folders"] == [str(buffers.resolve())]
    assert row["alternate_color_reference_image_count"] == 1
    assert row["alternate_color_reference_folders"] == [str(alternate.resolve())]


def test_per_run_protocol_layout_reports_missing_workbooks(tmp_path: Path) -> None:
    run = tmp_path / "series" / "Run10" / "CO2 injection"
    run.mkdir(parents=True)
    (run / "frame01.jpg").write_bytes(b"image")

    row = discover_image_source_mapping(
        tmp_path / "series",
        protocol_layout="per_run_files",
    )[0]

    assert row["protocol_table"] == ""
    assert "No per-experiment protocol workbook was found." in row["discovery_warnings"]


def test_run_layout_can_use_last_clearwater_frame_and_report_missing_role(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    example = experiments / "Run10"
    incomplete = experiments / "Run11"
    for run in (example, incomplete):
        injection = run / "CO2 injection"
        flush = run / "BTB flush"
        injection.mkdir(parents=True)
        flush.mkdir()
        (injection / "frame01.jpg").write_bytes(b"frame")
        (flush / "frame01.jpg").write_bytes(b"clear")
        (flush / "frame02.jpg").write_bytes(b"blue")
    overflow = example / "overflow"
    overflow.mkdir()
    (overflow / "display.jpg").write_bytes(b"scale")

    template = build_run_layout_template(example, clearwater_selector="last")
    rows = discover_image_source_mapping(
        experiments,
        structure="run_subfolders",
        layout_template=template,
    )

    assert all(row["clearwater_candidate"].endswith("frame02.jpg") for row in rows)
    assert rows[1]["overflow_image_count"] == 0
    assert any(
        "Expected folder was not matched" in warning
        for warning in rows[1]["discovery_warnings"]
    )


def test_run_layout_ignores_unassigned_additional_sources(
    tmp_path: Path,
) -> None:
    experiments = tmp_path / "experiments"
    example = experiments / "AC17"
    target = experiments / "AC18"
    for folder in (
        example / "240101_120000_CO2",
        example / "240101_140000_rest",
        example / "240101_090000_flush",
        target / "240102_120000_injection",
        target / "240102_140000_resting",
        target / "240102_090000_flush",
        target / "overflow" / "snapshots",
    ):
        folder.mkdir(parents=True)
    for index, folder in enumerate(
        (
            example / "240101_120000_CO2",
            example / "240101_140000_rest",
            example / "240101_090000_flush",
            target / "240102_120000_injection",
            target / "240102_140000_resting",
            target / "240102_090000_flush",
            target / "overflow" / "snapshots",
        )
    ):
        (folder / f"frame{index:02d}.jpg").write_bytes(b"image")
    (target / "overflow" / "overflow.xlsx").write_bytes(b"table")

    template = build_run_layout_template(example)
    rows = discover_image_source_mapping(
        experiments,
        structure="run_subfolders",
        layout_template=template,
    )
    target_row = rows[1]

    assert target_row["image_count"] == 2
    assert len(target_row["experiment_folders"]) == 2
    assert target_row["baseline_candidate"].endswith("frame03.jpg")
    assert target_row["overflow_image_count"] == 0
    assert target_row["overflow_table"] == ""
    assert any(
        match["match"] == "ignored-extra" and match["role"] == ROLE_IGNORE
        for match in target_row["source_folder_matches"]
    )
    assert any(
        "Unassigned folder was ignored" in warning
        for warning in target_row["discovery_warnings"]
    )
    assert any(
        "Unassigned table file was ignored" in warning
        for warning in target_row["discovery_warnings"]
    )


def test_explicitly_ignored_folder_is_not_rediscovered_as_an_extra_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "series"
    for run_id in ("AC17", "AC18"):
        injection = root / run_id / "CO2 injection"
        overflow = root / run_id / "overflow" / "snapshots"
        injection.mkdir(parents=True)
        overflow.mkdir(parents=True)
        (injection / "frame001.jpg").write_bytes(b"co2")
        (overflow / "scale001.jpg").write_bytes(b"scale")

    example = root / "AC17"
    template = build_run_layout_template(
        example,
        assignments={
            "CO2 injection": ROLE_INJECTION,
            "overflow/snapshots": ROLE_IGNORE,
        },
    )
    rows = discover_image_source_mapping(root, layout_template=template)

    ignored_spec = next(
        row
        for row in template["folders"]
        if row["relative_path"] == "overflow/snapshots"
    )
    assert ignored_spec["role"] == ROLE_IGNORE
    assert all(row["overflow_image_count"] == 0 for row in rows)
    assert all(
        any(
            match["role"] == ROLE_IGNORE
            and match["relative_path"] == "overflow/snapshots"
            for match in row["source_folder_matches"]
        )
        for row in rows
    )


def test_reviewed_run_definitions_can_rename_and_exclude_folders(
    tmp_path: Path,
) -> None:
    root = tmp_path / "series"
    for name in ("AC17", "AC18", "notes"):
        folder = root / name / "CO2 injection"
        folder.mkdir(parents=True)
        (folder / "frame01.jpg").write_bytes(b"image")

    candidates = discover_run_candidates(root)
    definitions = [
        {
            **row,
            "run_id": "run-seventeen" if row["run_id"] == "ac17" else row["run_id"],
            "included": row["run_id"] != "notes",
        }
        for row in candidates
    ]
    rows = discover_image_source_mapping(root, run_definitions=definitions)

    assert [row["run_id"] for row in rows] == ["run-seventeen", "ac18"]
    assert all(row["baseline_candidate"].endswith("frame01.jpg") for row in rows)


def test_multiple_injection_folders_follow_explicit_sequence_order(
    tmp_path: Path,
) -> None:
    run = tmp_path / "series" / "Run10"
    alphabetically_first = run / "A_second_injection"
    chronologically_first = run / "Z_first_injection"
    for folder, image in (
        (alphabetically_first, "frame101.jpg"),
        (chronologically_first, "frame001.jpg"),
    ):
        folder.mkdir(parents=True)
        (folder / image).write_bytes(b"image")
    assignments = {
        "A_second_injection": ROLE_INJECTION,
        "Z_first_injection": ROLE_INJECTION,
    }
    template = build_run_layout_template(
        run,
        assignments=assignments,
        sequence_orders={"Z_first_injection": 1, "A_second_injection": 2},
    )

    row = discover_image_source_mapping(
        run,
        structure="single_run",
        flat_run_id="run10",
        layout_template=template,
    )[0]

    assert row["image_count"] == 2
    assert row["baseline_candidate"].endswith("frame001.jpg")
    assert [Path(item["folder"]).name for item in row["co2_sequences"]] == [
        "Z_first_injection",
        "A_second_injection",
    ]
    assert [item["order"] for item in row["co2_sequences"]] == [1, 2]


def test_selected_protocol_file_is_matched_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "series"
    for run_id in ("AC17", "AC18"):
        injection = root / run_id / "CO2 injection"
        protocol = root / run_id / "protocol"
        injection.mkdir(parents=True)
        protocol.mkdir()
        (injection / "frame001.jpg").write_bytes(b"image")
        (protocol / f"{run_id}_notes.xlsx").write_bytes(b"notes")
        (protocol / f"{run_id}_protocol_darsia.xlsx").write_bytes(b"protocol")

    example = root / "AC17"
    selected = example / "protocol" / "AC17_protocol_darsia.xlsx"
    template = build_run_layout_template(
        example,
        assignments={
            "CO2 injection": ROLE_INJECTION,
            "protocol": ROLE_PROTOCOL_TABLE,
        },
        selected_table_files={"protocol": str(selected)},
    )
    rows = discover_image_source_mapping(root, layout_template=template)

    assert [Path(row["protocol_table"]).name for row in rows] == [
        "AC17_protocol_darsia.xlsx",
        "AC18_protocol_darsia.xlsx",
    ]
    assert all(len(row["protocol_tables"]) == 1 for row in rows)


def test_protocol_and_overflow_workbooks_are_independent_in_same_folder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "series"
    for run_id in ("AC17", "AC18"):
        run = root / run_id
        injection = run / "CO2 injection"
        injection.mkdir(parents=True)
        (injection / "frame001.jpg").write_bytes(b"image")
        (run / f"{run_id}_protocol_darsia.xlsx").write_bytes(b"protocol")
        (run / f"{run_id}_overflow_darsia.xlsx").write_bytes(b"overflow")
        (run / f"{run_id}_notes.xlsx").write_bytes(b"notes")
        if run_id == "AC18":
            (run / "AC18_overflow.xlsx").write_bytes(b"legacy overflow")

    example = root / "AC17"
    template = build_run_layout_template(
        example,
        table_assignments={
            "AC17_protocol_darsia.xlsx": ROLE_PROTOCOL_TABLE,
            "AC17_overflow_darsia.xlsx": ROLE_OVERFLOW_TABLE,
            "AC17_notes.xlsx": ROLE_IGNORE,
        },
    )
    rows = discover_image_source_mapping(root, layout_template=template)

    assert template["schema"] == 2
    assert len(template["table_files"]) == 3
    assert [Path(row["protocol_table"]).name for row in rows] == [
        "AC17_protocol_darsia.xlsx",
        "AC18_protocol_darsia.xlsx",
    ]
    assert [Path(row["overflow_table"]).name for row in rows] == [
        "AC17_overflow_darsia.xlsx",
        "AC18_overflow_darsia.xlsx",
    ]
    assert all(
        any(match["role"] == ROLE_IGNORE for match in row["source_table_matches"])
        for row in rows
    )
    target_matches = rows[1]["source_table_matches"]
    assert (
        next(
            match
            for match in target_matches
            if Path(match["path"]).name == "AC18_overflow_darsia.xlsx"
        )["role"]
        == ROLE_OVERFLOW_TABLE
    )
    assert (
        next(
            match
            for match in target_matches
            if Path(match["path"]).name == "AC18_overflow.xlsx"
        )["role"]
        == ROLE_IGNORE
    )
