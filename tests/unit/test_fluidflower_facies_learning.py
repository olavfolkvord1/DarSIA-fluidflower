from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.facies_learning import (
    FaciesAnnotationStore,
    collect_facies_annotation_bundles,
)
from darsia.fluidflower.geometry_facies import (
    GeometryRevisionStore,
    validate_geometry_config,
)
from darsia.fluidflower.project import FluidFlowerProject


def _project_with_geometry(
    tmp_path: Path, name: str
) -> tuple[FluidFlowerProject, dict]:
    root = tmp_path / name
    baseline = tmp_path / f"{name}-baseline.png"
    Image.new("RGB", (10, 8), (30, 80, 130)).save(baseline)
    labels = np.zeros((8, 10), dtype=np.int16)
    labels[2:5] = 1
    labels[5:] = 2
    labels_path = tmp_path / f"{name}-labels.npy"
    depth_path = tmp_path / f"{name}-depth.npy"
    np.save(labels_path, labels)
    np.save(depth_path, np.full(labels.shape, 0.01, dtype=np.float32))
    project = FluidFlowerProject.create(
        root,
        name=name,
        series_id=name,
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )
    config = {
        "schema": 1,
        "format": "darsia-fluidflower-geometry-config-v1",
        "rig_type": "medium",
        "baseline_path": str(baseline),
        "depth_path": str(depth_path),
        "labels_path": str(labels_path),
        "facies_source_path": "",
        "width_m": 0.9,
        "height_m": 0.49,
        "coordinate_convention": "origin_lower_left_x_right_y_up",
        "grouping_mode": "shared_facies",
        "permeability_unit": "mD",
        "regions": [
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
        ],
        "injectors": [],
    }
    validation = validate_geometry_config(config)
    revision = GeometryRevisionStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="Facies annotation test",
        validation=validation,
    )
    return project, revision


def test_facies_annotation_bundle_is_portable_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    project, revision = _project_with_geometry(tmp_path, "series-a")
    project.set_module_status("data_integrity", "approved", updated_by="Reviewer")
    project.set_module_status("image_correction", "approved", updated_by="Reviewer")
    project.set_module_status("geometry_facies", "approved", updated_by="Reviewer")
    store = FaciesAnnotationStore(tmp_path / "central-dataset")

    record = store.export(
        project,
        revision,
        split="train",
        group_id="rig-a",
        actor="Reviewer",
    )
    repeated = store.export(
        project,
        revision,
        split="train",
        group_id="rig-a",
        actor="Reviewer",
    )
    manifest = store.verify(record)

    assert not record["reused"]
    assert repeated["reused"]
    assert repeated["bundle_id"] == record["bundle_id"]
    assert manifest["eligible_for_training"]
    assert manifest["split"] == "train"
    assert manifest["project"]["series_id"] == "series-a"
    assert manifest["annotation_scope"]["domain"] == "registered_camera_plane_2d"
    assert not manifest["annotation_scope"]["contains_3d_extrapolation"]
    assert {row["canonical_key"] for row in manifest["material_classes"]} == {
        "porous_medium:sand",
        "seal:seal",
    }
    assert all(
        row["eligible_for_cross_series_training"]
        for row in manifest["material_classes"]
    )
    assert manifest["frames"][0]["recommended_for_facies_training"]
    assert manifest["hydraulic_classes"]["unit"] == "mD"
    assert manifest["hydraulic_classes"]["eligible"]
    bundle = store.root / record["bundle_relpath"]
    with np.load(bundle / "targets.npz", allow_pickle=False) as archive:
        assert archive["labels"].shape == (8, 10)
        assert set(np.unique(archive["roles"]).tolist()) == {1, 2, 3}
        assert np.count_nonzero(archive["training_mask"]) == 60
        assert np.count_nonzero(archive["context_mask"]) == 80
        assert set(np.unique(archive["material_classes"]).tolist()) == {0, 1, 2}
        assert set(np.unique(archive["hydraulic_classes"]).tolist()) == {0, 1, 3}
    assert (bundle / "preview.png").is_file()
    assert store.summary()["split_conflicts"] == {}

    with pytest.raises(ValueError, match="already assigned"):
        store.export(
            project,
            revision,
            split="holdout",
            group_id="rig-a",
            actor="Reviewer",
        )


def test_collection_excludes_drafts_and_detects_cross_store_leakage(
    tmp_path: Path,
) -> None:
    first_project, first_revision = _project_with_geometry(tmp_path, "series-first")
    first_store = FaciesAnnotationStore(tmp_path / "dataset-first")
    draft = first_store.export(
        first_project,
        first_revision,
        split="train",
        group_id="shared-rig",
        actor="Reviewer",
    )
    assert not draft["eligible_for_training"]
    assert collect_facies_annotation_bundles([first_store.root]) == []
    assert (
        len(collect_facies_annotation_bundles([first_store.root], include_drafts=True))
        == 1
    )

    second_project, second_revision = _project_with_geometry(tmp_path, "series-second")
    second_project.set_module_status(
        "data_integrity", "approved", updated_by="Reviewer"
    )
    second_project.set_module_status(
        "image_correction", "approved", updated_by="Reviewer"
    )
    second_project.set_module_status(
        "geometry_facies", "approved", updated_by="Reviewer"
    )
    second_store = FaciesAnnotationStore(tmp_path / "dataset-second")
    second_store.export(
        second_project,
        second_revision,
        split="holdout",
        group_id="shared-rig",
        actor="Reviewer",
    )

    with pytest.raises(ValueError, match="split leakage"):
        collect_facies_annotation_bundles(
            [first_store.root, second_store.root],
            include_drafts=True,
        )
