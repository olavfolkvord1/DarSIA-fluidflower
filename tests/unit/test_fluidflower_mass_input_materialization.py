from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from darsia.fluidflower.mass_input_materialization import (
    MASS_INPUT_MATERIALIZATION_FORMAT,
    MassInputRunWriter,
    PreparedMassInputFrame,
    write_mass_input_manifest,
)
from darsia.fluidflower.mass_inputs import inspect_mass_input_manifest


def _frame(index: int, shape: tuple[int, int]) -> PreparedMassInputFrame:
    co2 = np.zeros(shape, dtype=np.float32)
    gas = np.zeros(shape, dtype=np.float32)
    acid = np.zeros(shape, dtype=np.float32)
    if index:
        co2[1:-1, 2:-2] = 0.8
        gas[2:4, 4:7] = 0.55
        acid[1:-1, 2:-2] = 0.65
    image = np.zeros((*shape, 3), dtype=np.float32)
    image[..., 2] = 0.4
    return PreparedMassInputFrame(
        frame_id=f"frame-{index:04d}",
        time_hours=float(index),
        injected_mass_g=0.5 * index,
        pressure_bar=1.01,
        temperature_c=20.0,
        pressure_gradient_bar_per_m=0.0,
        temperature_gradient_c_per_m=0.0,
        optical_acidification=acid,
        phase_co2_probability=co2,
        phase_free_gas_probability=gas,
        corrected_rgb=image,
        source_reference={"path": f"source-{index}.jpg", "sha256": "a" * 64},
    )


def test_streaming_mass_input_materialization_round_trip(tmp_path: Path) -> None:
    shape = (8, 12)
    grid = {
        "coordinate_system": "corrected_registered_rig_grid",
        "shape_rc": list(shape),
        "width_m": 1.2,
        "height_m": 0.8,
    }
    writer = MassInputRunWriter(
        tmp_path / "prepared",
        run_id="AC17",
        phase_candidate_id="candidate-17",
        grid=grid,
        depth=np.full(shape, 0.01, dtype=np.float32),
        porosity=np.full(shape, 0.36, dtype=np.float32),
        permeability=np.full(shape, 1.0e-10, dtype=np.float32),
        permeability_unit="m2",
        provenance={"phase_model_sha256": "b" * 64},
    )
    for index in range(3):
        writer.add_frame(_frame(index, shape))
    run = writer.finalize()
    manifest_path = write_mass_input_manifest(
        tmp_path / "prepared" / "mass_input_manifest.json",
        run_inputs=[run],
        upstream={
            "phase_revision_id": "phase-1",
            "phase_revision_sha256": "c" * 64,
            "image_correction_revision_id": "correction-1",
            "image_correction_revision_sha256": "d" * 64,
            "geometry_revision_id": "geometry-1",
            "geometry_revision_sha256": "e" * 64,
        },
        provenance={"producer": "unit-test"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["materialization_format"] == MASS_INPUT_MATERIALIZATION_FORMAT
    report = inspect_mass_input_manifest(
        str(manifest_path),
        run_bindings=[
            {"run_id": "ac17", "phase_candidate_id": "candidate-17"}
        ],
        maximum_images_per_run=3,
        require_preview_images=True,
    )
    assert report["valid"], report["issues"]
    assert report["shared_target_times_hours"] == [0.0, 1.0, 2.0]


def test_streaming_mass_input_rejects_gas_outside_co2(tmp_path: Path) -> None:
    shape = (4, 6)
    writer = MassInputRunWriter(
        tmp_path,
        run_id="run18",
        phase_candidate_id="candidate-18",
        grid={"shape_rc": list(shape), "width_m": 1.0, "height_m": 0.5},
        depth=np.ones(shape, dtype=np.float32),
        porosity=np.ones(shape, dtype=np.float32),
        permeability=np.ones(shape, dtype=np.float32),
        permeability_unit="m2",
    )
    frame = _frame(0, shape)
    invalid = PreparedMassInputFrame(
        **{
            **frame.__dict__,
            "phase_free_gas_probability": np.ones(shape, dtype=np.float32),
        }
    )
    try:
        writer.add_frame(invalid)
    except ValueError as exc:
        assert "cannot exceed total CO2" in str(exc)
    else:
        raise AssertionError("Gas outside CO2 support was accepted")
