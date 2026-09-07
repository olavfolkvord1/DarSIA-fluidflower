from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from darsia.fluidflower.mass_artifacts import (
    FIELD_INDEX_FORMAT,
    PREVIEW_INDEX_FORMAT,
    load_field_index,
    load_preview_index,
    write_field_frame,
    write_field_index,
    write_preview_index,
)
from darsia.fluidflower.project import sha256_file


def test_indexed_field_round_trip_and_frame_hash(tmp_path: Path) -> None:
    rows = []
    for index, time_hours in enumerate((0.0, 1.0, 2.0)):
        values = np.full((12, 18), index + 0.25, dtype=np.float32)
        display = np.clip(values / 3.0, 0.0, 1.0)
        rows.append(
            write_field_frame(
                tmp_path / f"frame-{index:03d}.npz",
                values,
                run_id="run01",
                product="concentration_aqueous",
                frame_id=f"frame-{index:03d}",
                time_hours=time_hours,
                display_values=display,
            )
        )
    index_path = tmp_path / "concentration-index.json"
    reference = write_field_index(
        index_path,
        run_id="run01",
        product="concentration_aqueous",
        quantity="aqueous_co2_concentration",
        unit="kg m-3",
        frames=rows,
        grid={"width_m": 2.8, "height_m": 1.5, "coordinate_system": "rig"},
    )

    product = load_field_index(
        reference["path"],
        reference["sha256"],
        run_id="run01",
        product="concentration_aqueous",
        expected_times_hours=[0.0, 1.0, 2.0],
    )
    values, display = product.load_frame(2)
    assert reference["format"] == FIELD_INDEX_FORMAT
    assert product.shape_rc == (12, 18)
    assert np.allclose(values, 2.25)
    assert display is not None and np.allclose(display, 0.75)

    (tmp_path / "frame-002.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256"):
        product.load_frame(2)


def test_preview_index_round_trip(tmp_path: Path) -> None:
    rows = []
    for index, time_hours in enumerate((0.0, 1.0)):
        path = tmp_path / f"preview-{index:03d}.png"
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, :, index] = 180
        assert cv2.imwrite(str(path), image)
        rows.append(
            {
                "frame_id": f"frame-{index:03d}",
                "time_hours": time_hours,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    index_path = tmp_path / "preview-index.json"
    reference = write_preview_index(index_path, run_id="run01", frames=rows)
    preview = load_preview_index(
        reference["path"],
        reference["sha256"],
        run_id="run01",
        expected_times_hours=[0.0, 1.0],
    )

    assert reference["format"] == PREVIEW_INDEX_FORMAT
    assert preview.load_frame(0).shape == (20, 30, 3)


def test_field_index_rejects_different_time_grid(tmp_path: Path) -> None:
    frame = write_field_frame(
        tmp_path / "mass.npz",
        np.ones((4, 5), dtype=np.float64),
        run_id="run01",
        product="mass_total",
        frame_id="frame-000",
        time_hours=0.0,
    )
    index_path = tmp_path / "mass-index.json"
    reference = write_field_index(
        index_path,
        run_id="run01",
        product="mass_total",
        quantity="co2_mass_per_pixel",
        unit="g pixel-1",
        frames=[frame],
        grid={"width_m": 1.0, "height_m": 1.0},
    )

    with pytest.raises(ValueError, match="time grid mismatch"):
        load_field_index(
            reference["path"],
            reference["sha256"],
            run_id="run01",
            product="mass_total",
            expected_times_hours=[1.0],
        )
