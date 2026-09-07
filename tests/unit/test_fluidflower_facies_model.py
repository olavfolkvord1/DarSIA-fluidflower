from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from darsia.fluidflower.facies_learning import FaciesAnnotationStore
from darsia.fluidflower.facies_model import (
    FACIES_MODEL_FORMAT,
    FACIES_SPLIT_PLAN_FORMAT,
    FaciesModelConfig,
    FaciesModelStore,
    extract_facies_features,
    facies_feature_names,
    train_facies_baseline,
)
from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    validate_geometry_config,
)
from darsia.fluidflower.project import FluidFlowerProject


def _export_bundle(
    tmp_path: Path,
    dataset_root: Path,
    *,
    name: str,
    split: str,
    group_id: str,
    include_unseen_gravel: bool = False,
) -> dict:
    height, width = 48, 64
    labels = np.zeros((height, width), dtype=np.int16)
    labels[12:24] = 1
    labels[24:] = 2
    if include_unseen_gravel:
        labels[34:, 36:] = 3
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    colours = {
        0: np.asarray((42, 91, 151), dtype=np.float32),
        1: np.asarray((102, 102, 107), dtype=np.float32),
        2: np.asarray((181, 145, 72), dtype=np.float32),
        3: np.asarray((137, 91, 64), dtype=np.float32),
    }
    image = np.zeros((height, width, 3), dtype=np.float32)
    for label_id, colour in colours.items():
        image[labels == label_id] = colour
    image += rng.normal(0.0, 3.0, image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    source_root = tmp_path / "sources" / name
    source_root.mkdir(parents=True)
    baseline_path = source_root / "baseline.png"
    labels_path = source_root / "labels.npy"
    depth_path = source_root / "depth.npy"
    Image.fromarray(image).save(baseline_path)
    np.save(labels_path, labels)
    np.save(depth_path, np.full(labels.shape, 0.01, dtype=np.float32))

    project = FluidFlowerProject.create(
        tmp_path / "projects" / name,
        name=name,
        series_id=name,
        template_id="ff_ac",
        rig_type="medium",
        created_by="Test reviewer",
    )
    regions = [
        {
            "label_id": 0,
            "facies_id": 0,
            "name": "Free water",
            "analysis_role": "free_water",
            "porosity": None,
            "permeability": None,
            "enabled": True,
        },
        {
            "label_id": 1,
            "facies_id": 1,
            "name": "Seal",
            "analysis_role": "seal",
            "porosity": 0.40,
            "permeability": 44.0,
            "enabled": True,
        },
        {
            "label_id": 2,
            "facies_id": 2,
            "name": "Sand",
            "analysis_role": "porous_medium",
            "porosity": 0.45,
            "permeability": 473.0,
            "enabled": True,
        },
    ]
    if include_unseen_gravel:
        regions.append(
            {
                "label_id": 3,
                "facies_id": 3,
                "name": "Gravel",
                "analysis_role": "porous_medium",
                "porosity": 0.48,
                "permeability": 980.0,
                "enabled": True,
            }
        )
    config = {
        "schema": 1,
        "format": "darsia-fluidflower-geometry-config-v1",
        "rig_type": "medium",
        "baseline_path": str(baseline_path),
        "depth_path": str(depth_path),
        "labels_path": str(labels_path),
        "facies_source_path": "",
        "width_m": 0.9,
        "height_m": 0.49,
        "coordinate_convention": "origin_lower_left_x_right_y_up",
        "grouping_mode": "shared_facies",
        "permeability_unit": "mD",
        "regions": regions,
        "injectors": [],
    }
    revision = GeometryRevisionStore(project).save_revision(
        config,
        actor="Test reviewer",
        reason="Facies model test",
        validation=validate_geometry_config(config),
    )
    project.set_module_status("data_integrity", "approved", updated_by="Test reviewer")
    project.set_module_status(
        "image_correction", "approved", updated_by="Test reviewer"
    )
    project.set_module_status("geometry_facies", "approved", updated_by="Test reviewer")
    return FaciesAnnotationStore(dataset_root).export(
        project,
        revision,
        split=split,
        group_id=group_id,
        actor="Test reviewer",
    )


def test_generic_feature_contract_is_dense_finite_and_position_optional() -> None:
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    image[..., 0] = np.arange(24, dtype=np.uint8)[None, :] * 7
    image[..., 1] = np.arange(20, dtype=np.uint8)[:, None] * 9
    image[..., 2] = 80
    indices = np.asarray([0, 23, 240, 479])

    generic = extract_facies_features(image, flat_indices=indices)
    positioned = extract_facies_features(
        image,
        flat_indices=indices,
        include_normalized_position=True,
    )

    assert generic.shape == (4, len(facies_feature_names()))
    assert positioned.shape == (
        4,
        len(facies_feature_names(include_normalized_position=True)),
    )
    assert positioned.shape[1] == generic.shape[1] + 2
    assert np.isfinite(generic).all()
    assert np.allclose(positioned[[0, -1], -2:], ((0.0, 0.0), (1.0, 1.0)))


def test_training_uses_group_isolated_splits_and_reports_unseen_materials(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "annotations"
    for index in range(3):
        _export_bundle(
            tmp_path,
            dataset_root,
            name=f"train-{index}",
            split="pool",
            group_id=f"train-rig-{index}",
        )
    _export_bundle(
        tmp_path,
        dataset_root,
        name="validation",
        split="pool",
        group_id="validation-rig",
    )
    holdout = _export_bundle(
        tmp_path,
        dataset_root,
        name="holdout",
        split="pool",
        group_id="holdout-rig",
        include_unseen_gravel=True,
    )

    result = train_facies_baseline(
        [dataset_root, dataset_root],
        tmp_path / "models",
        config=FaciesModelConfig(
            n_estimators=16,
            max_pixels_per_bundle=600,
            min_samples_leaf=1,
            cpu_workers=1,
            prediction_tile_size=64,
            store_probabilities=True,
        ),
        split_plan={
            "format": FACIES_SPLIT_PLAN_FORMAT,
            "name": "synthetic-transfer-test",
            "purpose": "Exercise immutable pool bundles with group-level assignment",
            "assignments": {
                "train-rig-0": "train",
                "train-rig-1": "train",
                "train-rig-2": "train",
                "validation-rig": "validation",
                "holdout-rig": "holdout",
            },
        },
    )
    model_root = Path(result["model_path"])
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
    metrics_text = (model_root / "metrics.json").read_text(encoding="utf-8")
    metrics = json.loads(metrics_text)

    assert manifest["format"] == FACIES_MODEL_FORMAT
    assert not manifest["dataset"]["holdout_used_for_fitting"]
    assert manifest["dataset"]["group_assignment"]["train"] == [
        "train-rig-0",
        "train-rig-1",
        "train-rig-2",
    ]
    assert manifest["dataset"]["group_assignment"]["holdout"] == ["holdout-rig"]
    assert manifest["dataset"]["split_plan"]["name"] == "synthetic-transfer-test"
    assert {row["source_split"] for row in manifest["dataset"]["bundles"]} == {"pool"}
    assert manifest["dataset"]["duplicate_records_skipped"] == 5
    assert len(manifest["dataset"]["bundles"]) == 5
    assert manifest["heads"]["role"]["calibration"]["applied"]
    assert manifest["heads"]["material"]["enabled"]
    assert manifest["heads"]["hydraulic"]["enabled"]
    assert not manifest["feature_contract"]["uses_absolute_position"]
    assert manifest["scientific_scope"]["domain"] == "registered_camera_plane_2d"
    assert not manifest["scientific_scope"]["contains_3d_extrapolation"]
    assert "NaN" not in metrics_text

    holdout_metrics = next(
        row for row in metrics["bundles"] if row["bundle_id"] == holdout["bundle_id"]
    )
    assert holdout_metrics["material"]["unseen_truth_fraction"] > 0.0
    assert holdout_metrics["material"]["seen_material_coverage"] < 1.0
    prediction_path = (
        model_root / "predictions" / holdout["bundle_id"] / "prediction.npz"
    )
    with np.load(prediction_path, allow_pickle=False) as prediction:
        assert prediction["role_prediction"].shape == (48, 64)
        assert prediction["role_probabilities"].dtype == np.float16
        assert prediction["hydraulic_prediction"].shape == (48, 64)
        assert prediction["hydraulic_probabilities"].dtype == np.float16
        assert prediction["role_uncertainty"].min() >= 0.0
        assert prediction["role_uncertainty"].max() <= 1.0
    assert (model_root / "model.joblib").is_file()
    assert (model_root / "model_card.md").is_file()
    assert (model_root / "predictions" / holdout["bundle_id"] / "preview.png").is_file()
    model_store = FaciesModelStore(tmp_path / "models")
    records = model_store.records()
    assert len(records) == 1
    assert model_store.verify(records[0])["model_id"] == result["model_id"]
