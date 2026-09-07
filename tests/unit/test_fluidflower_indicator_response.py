from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from darsia.fluidflower.calibration import (
    apply_indicator_response_constraints,
    default_calibration_plan,
    validate_calibration_plan,
)
from darsia.fluidflower.indicator_response import (
    INDICATOR_RESPONSE_MODEL_FORMAT,
    INDICATOR_RESPONSE_REFERENCE_FORMAT,
    IndicatorResponseModelStore,
    default_indicator_response_config,
    fit_indicator_response_from_arrays,
    predict_indicator_dic,
    predict_indicator_ph,
)
from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    import_darsia_rig,
    validate_geometry_config,
)
from darsia.fluidflower.image_correction import (
    CorrectionRevisionStore,
    identity_correction_config,
)
from darsia.fluidflower.indicator_references import build_indicator_reference_contract
from darsia.fluidflower.phase_interpretation import (
    PhaseInterpretationRevisionStore,
    default_phase_interpretation_config,
)
from darsia.fluidflower.project import FluidFlowerProject


def _project_with_phase_revision(tmp_path: Path) -> FluidFlowerProject:
    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Indicator response",
        series_id="indicator-response",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    review_file = review_dir / "review.json"
    review_file.write_text('{"reviewer":"Reviewer"}', encoding="utf-8")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "campaign_id": "indicator-response-campaign",
                "dataset": "ffac",
                "jobs": [
                    {
                        "job_id": "ac17-physics",
                        "run": "ac17",
                        "role": "physics",
                        "variant": "physics-v1",
                        "family": "hierarchical-phase",
                        "candidate_id": "candidate-ac17",
                        "quality_flags": {
                            "dataset": "ffac",
                            "hierarchical_phase_partition_active": True,
                        },
                    }
                ],
                "design": {
                    "interpretation_scope": "visible-2d-slice-only",
                    "mass_calibration_active": False,
                    "depth_used": False,
                    "deployment_allowed": False,
                },
                "review": {"output_dir": str(review_dir)},
            }
        ),
        encoding="utf-8",
    )
    phase_config = default_phase_interpretation_config()
    phase_config.update(
        {
            "campaign_manifest_path": str(campaign_path),
            "review_package_path": str(review_dir),
            "selected_candidates": {"ac17": "candidate-ac17"},
            "run_reviews": {
                "ac17": {
                    "status": "approved",
                    "candidate_id": "candidate-ac17",
                    "reviewer": "Reviewer",
                }
            },
            "manual_review_files": [str(review_file)],
        }
    )
    PhaseInterpretationRevisionStore(project).save_revision(
        phase_config,
        actor="Reviewer",
        reason="Ready for response-bound calibration",
    )
    return project


def _synthetic_model() -> dict:
    height, width = 36, 48
    facies = np.ones((height, width), dtype=np.int32)
    facies[:, width // 2 :] = 2
    active = np.ones((height, width), dtype=bool)
    observations = []
    for index, ph in enumerate((6.2, 6.8, 7.4, 8.0)):
        fraction = (ph - 6.2) / (8.0 - 6.2)
        base = np.asarray([0.92, 0.72, 0.08]) * (1.0 - fraction)
        base += np.asarray([0.08, 0.30, 0.78]) * fraction
        image = np.broadcast_to(base, (height, width, 3)).copy()
        image[:, width // 2 :] *= np.asarray([0.92, 1.04, 0.96])
        row_noise = np.linspace(-0.004, 0.004, height)[:, None, None]
        image = np.clip(image + row_noise, 0.0, 1.0).astype(np.float32)
        observations.append(
            {
                "image_id": f"buffer-{index}",
                "ph": ph,
                "scope": "series",
                "image": image,
            }
        )
    return fit_indicator_response_from_arrays(
        observations,
        facies=facies,
        active_mask=active,
        facies_names={"1": "Upper sand", "2": "Lower sand"},
        chemistry={
            "adapter": "btb_carbonate_charge_balance_v1",
            "alkalinity_M": 1.25e-3,
            "btb_M": 0.726e-3,
            "pKa1": 6.35,
            "pKa2": 10.33,
            "pKaB": 7.10,
            "co2_saturation_M": 0.034,
            "initial_ph_bounds": [7.5, 8.2],
            "measured_bounds": {
                "initial_ph": [7.5, 8.2],
                "alkalinity_M": [1.0e-3, 1.5e-3],
                "btb_M": [0.726e-3, 0.726e-3],
                "co2_saturation_M": [0.034, 0.034],
            },
            "bounds_reviewed": True,
        },
    )


def test_full_color_response_is_monotone_and_holdout_validated() -> None:
    model = _synthetic_model()

    assert model["status"] == "ready"
    assert model["feature_contract"]["assumed_color_direction"] is None
    assert model["validation"]["holdout_record_count"] > 0
    assert model["validation"]["mean_group_mae_ph"] < 0.45

    colors = np.asarray(
        [
            [0.92, 0.72, 0.08],
            [0.50, 0.51, 0.43],
            [0.08, 0.30, 0.78],
        ],
        dtype=np.float32,
    )
    prediction = predict_indicator_ph(model, colors)
    assert np.all(np.diff(prediction["ph"]) > 0.0)
    assert np.all(prediction["ph_lower"] <= prediction["ph"])
    assert np.all(prediction["ph"] <= prediction["ph_upper"])


def test_carbonate_adapter_assigns_more_dic_to_lower_ph() -> None:
    model = _synthetic_model()
    colors = np.asarray(
        [[0.92, 0.72, 0.08], [0.08, 0.30, 0.78]], dtype=np.float32
    )

    prediction = predict_indicator_dic(model, colors)

    assert prediction["ph"][0] < prediction["ph"][1]
    assert prediction["dic_M"][0] > prediction["dic_M"][1]
    assert np.all(prediction["dic_lower_M"] <= prediction["dic_M"])
    assert np.all(prediction["dic_M"] <= prediction["dic_upper_M"])


def test_response_model_freezes_optics_and_bounds_optuna(
    tmp_path: Path,
) -> None:
    project = _project_with_phase_revision(tmp_path)
    model = _synthetic_model()
    model.update({"format": INDICATOR_RESPONSE_MODEL_FORMAT, "model_id": "response-1"})
    model_path = tmp_path / "response.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    reference = {
        "schema": 1,
        "format": INDICATOR_RESPONSE_REFERENCE_FORMAT,
        "model_id": "response-1",
        "path": str(model_path),
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "status": "ready",
    }
    config = default_calibration_plan(project)
    config["chemistry"]["model"] = "titration"
    constrained = apply_indicator_response_constraints(config, reference)
    parameters = {
        str(row["key"]): row
        for row in constrained["parameter_space"]["parameters"]
    }

    assert not parameters["aqueous_transition_low_fraction"]["enabled"]
    assert not parameters["aqueous_transition_width_fraction"]["enabled"]
    assert not parameters["aqueous_titration_blend"]["enabled"]
    assert parameters["initial_ph"]["enabled"]
    assert parameters["initial_ph"]["lower"] == 7.5
    assert parameters["initial_ph"]["upper"] == 8.2
    assert parameters["alkalinity_M"]["enabled"]
    assert parameters["alkalinity_M"]["lower"] == 1.0e-3
    assert parameters["alkalinity_M"]["upper"] == 1.5e-3
    assert not parameters["indicator_concentration_M"]["enabled"]
    assert not parameters["co2_saturation_M"]["enabled"]
    assert validate_calibration_plan(project, constrained)["valid"]

    parameters["aqueous_transition_low_fraction"]["enabled"] = True
    report = validate_calibration_plan(project, constrained)
    codes = {str(row["code"]) for row in report["issues"]}
    assert "measured_optical_response_is_searchable" in codes


def _save_geometry_field(path: Path, array: np.ndarray) -> None:
    np.savez_compressed(
        path,
        array=array,
        metadata={
            "space_dim": 2,
            "indexing": "ij",
            "dimensions": [0.3, 0.4],
            "origin": "[0.0, 0.3]",
        },
    )


def test_project_fitter_keeps_sources_immutable_and_versions_model(
    tmp_path: Path,
) -> None:
    project = FluidFlowerProject.create(
        tmp_path / "project-store",
        name="Response store",
        series_id="response-store",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    rig = tmp_path / "rig"
    rig.mkdir()
    height, width = 30, 40
    labels = np.ones((height, width), dtype=np.int16)
    labels[:, width // 2 :] = 2
    baseline = np.zeros((height, width, 3), dtype=np.uint8)
    baseline[..., 2] = 150
    for name, array in (
        ("baseline.npz", baseline),
        ("labels.npz", labels),
        ("facies.npz", labels),
        ("depth.npz", np.full(labels.shape, 0.01, dtype=np.float32)),
        ("porosity.npz", np.full(labels.shape, 0.4, dtype=np.float32)),
        ("permeability.npz", np.full(labels.shape, 100.0, dtype=np.float32)),
    ):
        _save_geometry_field(rig / name, array)
    properties = tmp_path / "properties.csv"
    pd.DataFrame(
        [
            {"id": 1, "name": "Sand A", "porosity": 0.4, "permeability": 100},
            {"id": 2, "name": "Sand B", "porosity": 0.4, "permeability": 100},
        ]
    ).to_csv(properties, index=False)
    protocol = tmp_path / "protocol.csv"
    pd.DataFrame(
        [{"location_x": 0.2, "location_y": 0.1, "rate_sccm": 2.0}]
    ).to_csv(protocol, index=False)
    geometry = import_darsia_rig(
        rig,
        facies_properties_path=properties,
        protocol_path=protocol,
        rig_type="medium",
    )
    geometry_report = validate_geometry_config(geometry)
    assert geometry_report["summary"]["error_count"] == 0
    GeometryRevisionStore(project).save_revision(
        geometry,
        actor="Reviewer",
        reason="Synthetic approved geometry",
        validation=geometry_report,
    )
    CorrectionRevisionStore(project).save_revision(
        identity_correction_config(),
        actor="Reviewer",
        reason="Synthetic identity correction",
    )

    entries = []
    source_hashes = {}
    for index, ph in enumerate((6.2, 6.8, 7.4, 8.0)):
        fraction = (ph - 6.2) / 1.8
        color = np.asarray([0.92, 0.72, 0.08]) * (1.0 - fraction)
        color += np.asarray([0.08, 0.30, 0.78]) * fraction
        image = np.broadcast_to(color, (height, width, 3)).copy()
        image[:, width // 2 :] *= np.asarray([0.92, 1.04, 0.96])
        path = tmp_path / f"buffer_ph_{ph:.1f}.png"
        Image.fromarray(np.rint(255.0 * image).astype(np.uint8)).save(path)
        source_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "kind": "ph_buffer",
                "ph": ph,
                "label": "",
                "scope": "series",
                "image_path": str(path),
            }
        )
    config = default_indicator_response_config(project)
    config["reference_contract"] = build_indicator_reference_contract(
        entries, indicator_system="BTB"
    )
    config["chemistry"]["bounds_reviewed"] = True
    config["chemistry"]["measured_bounds"]["alkalinity_M"] = [1.0e-3, 1.5e-3]
    revision = IndicatorResponseModelStore(project).fit_and_save(
        config,
        actor="Reviewer",
        reason="Test immutable buffer fitting",
    )

    assert revision["status"] == "ready"
    assert Path(revision["model_path"]).is_file()
    assert IndicatorResponseModelStore(project).current_reference()["sha256"] == (
        revision["model_sha256"]
    )
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in source_hashes.items()
    )
