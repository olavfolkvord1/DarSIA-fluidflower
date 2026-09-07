from __future__ import annotations

from pathlib import Path

from darsia.fluidflower.indicator_references import (
    REFERENCE_KIND_ALTERNATE_STATE,
    REFERENCE_KIND_PH_BUFFER,
    build_indicator_reference_contract,
    discover_indicator_reference_images,
    infer_ph,
)


def test_ph_is_inferred_only_from_explicit_path_label() -> None:
    assert infer_ph("buffers/pH_7,25/frame.jpg") == 7.25
    assert infer_ph("buffers/8.1-pH/frame.jpg") == 8.1
    assert infer_ph("buffers/yellow/frame_72.jpg") is None


def test_reference_contract_separates_chemistry_from_radiometry(
    tmp_path: Path,
) -> None:
    first = tmp_path / "pH_6.5"
    second = tmp_path / "pH_8.0"
    alternate = tmp_path / "different rig color"
    for folder in (first, second, alternate):
        folder.mkdir()
        (folder / "frame.jpg").write_bytes(b"image")

    entries = discover_indicator_reference_images(
        [
            (first, "series"),
            (second, "series"),
            (alternate, "run18"),
        ]
    )
    contract = build_indicator_reference_contract(entries, indicator_system="BTB")

    assert [row["kind"] for row in entries] == [
        REFERENCE_KIND_PH_BUFFER,
        REFERENCE_KIND_PH_BUFFER,
        REFERENCE_KIND_ALTERNATE_STATE,
    ]
    assert contract["status"] == "ready"
    assert contract["distinct_ph_values"] == [6.5, 8.0]
    assert contract["ph_span"] == [6.5, 8.0]
    assert contract["known_ph_buffer_count"] == 2
    assert contract["alternate_state_count"] == 1
    assert entries[0]["kind"] == REFERENCE_KIND_PH_BUFFER
    assert contract["radiometric_normalization_allowed"] is False


def test_incomplete_buffer_metadata_is_preserved_and_flagged(tmp_path: Path) -> None:
    image = tmp_path / "buffer_unknown.jpg"
    image.write_bytes(b"image")

    contract = build_indicator_reference_contract(
        [
            {
                "kind": REFERENCE_KIND_PH_BUFFER,
                "ph": "",
                "label": "unknown buffer",
                "scope": "series",
                "image_path": str(image),
            }
        ],
        indicator_system="BTB",
    )

    assert contract["status"] == "metadata_required"
    assert contract["entry_count"] == 1
    assert contract["entries"][0]["validation_error"]


def test_explicit_source_kind_overrides_filename_inference(tmp_path: Path) -> None:
    alternate = tmp_path / "pH_6.5_but_actually_a_dye_control"
    alternate.mkdir()
    image = alternate / "frame.jpg"
    image.write_bytes(b"image")

    rows = discover_indicator_reference_images(
        [(alternate, "series")],
        default_kind=REFERENCE_KIND_ALTERNATE_STATE,
    )

    assert rows[0]["kind"] == REFERENCE_KIND_ALTERNATE_STATE
    assert rows[0]["ph"] is None
    assert rows[0]["label"] == alternate.name
