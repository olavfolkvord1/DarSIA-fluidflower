from __future__ import annotations

from pathlib import Path

import pytest

from darsia.fluidflower.migration import (
    COPY_MANIFEST_FORMAT,
    build_copy_plan,
    execute_copy_plan,
    validate_copy_plan,
    verify_copy_manifest,
)
from darsia.fluidflower.project import sha256_file


def test_copy_plan_publishes_verified_files_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"immutable-source")
    source_hash = sha256_file(source)
    destination = tmp_path / "new-project"
    plan = build_copy_plan(
        destination,
        [
            {
                "source": source,
                "destination": "AC52/CO2 injection/frame.jpg",
                "kind": "experiment_image",
                "run_id": "ac52",
            }
        ],
        title="Two-run demo",
        created_by="Reviewer",
    )

    dry_run = execute_copy_plan(
        plan,
        generated_files={"README.txt": b"demo\n"},
        dry_run=True,
    )
    assert dry_run["status"] == "dry-run"
    assert not destination.exists()

    result = execute_copy_plan(
        plan,
        generated_files={"README.txt": b"demo\n"},
    )
    assert result["valid"]
    assert result["verified_files"] == 2
    assert source.read_bytes() == b"immutable-source"
    assert sha256_file(source) == source_hash
    assert (destination / "AC52" / "CO2 injection" / "frame.jpg").read_bytes() == (
        b"immutable-source"
    )
    assert (destination / "README.txt").read_text(encoding="utf-8") == "demo\n"
    assert verify_copy_manifest(destination)["valid"]
    assert COPY_MANIFEST_FORMAT in (
        destination / "migration_manifest.json"
    ).read_text(encoding="utf-8")


def test_copy_plan_refuses_existing_destination_and_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"value")
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_copy_plan(
            destination,
            [{"source": source, "destination": "source.bin"}],
            title="Rejected",
        )

    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="Unsafe"):
        build_copy_plan(
            absent,
            [{"source": source, "destination": "../source.bin"}],
            title="Rejected",
        )


def test_copy_plan_detects_source_change_before_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    destination = tmp_path / "destination"
    plan = build_copy_plan(
        destination,
        [{"source": source, "destination": "source.bin"}],
        title="Changed source",
    )
    source.write_bytes(b"second")

    report = validate_copy_plan(plan)
    assert not report["valid"]
    assert any("changed" in error.casefold() for error in report["errors"])
    with pytest.raises(ValueError, match="changed"):
        execute_copy_plan(plan)
    assert not destination.exists()

