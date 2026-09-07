from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from darsia.fluidflower.facies_examples import (
    FACIES_EXAMPLE_PREDICTION_FORMAT,
    FaciesExampleStore,
    create_facies_example_prediction,
    default_facies_example_session,
    facies_config_from_example_prediction,
    facies_example_quality,
)
from darsia.fluidflower.facies_model import FACIES_MODEL_FORMAT
from darsia.fluidflower.project import FluidFlowerProject, sha256_file


def _project(tmp_path: Path) -> FluidFlowerProject:
    return FluidFlowerProject.create(
        tmp_path / "project",
        name="Facies material example test",
        series_id="facies-example-test",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )


def _image(path: Path) -> Path:
    yy, xx = np.mgrid[:80, :120]
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :60] = (32, 87, 126)
    image[:, 60:] = (186, 159, 86)
    image[..., 1] = np.clip(image[..., 1] + (yy % 7), 0, 255)
    image[..., 2] = np.clip(image[..., 2] + (xx % 5), 0, 255)
    Image.fromarray(image).save(path)
    return path


def _ready_session(project: FluidFlowerProject, image_path: Path) -> dict:
    session = default_facies_example_session(project)
    session["source_image_path"] = str(image_path)
    session["materials"] = [
        {
            "material_id": "blue-sand",
            "name": "Blue sand",
            "canonical_key": "blue-sand",
            "analysis_role": "porous_medium",
        },
        {
            "material_id": "yellow-seal",
            "name": "Yellow seal",
            "canonical_key": "yellow-seal",
            "analysis_role": "seal",
        },
    ]
    examples = []
    for material_id, x_base in (("blue-sand", 0.04), ("yellow-seal", 0.56)):
        for index in range(8):
            examples.append(
                {
                    "example_id": f"{material_id}-{index}",
                    "material_id": material_id,
                    "source_image_path": str(image_path),
                    "x": x_base + 0.04 * (index % 3),
                    "y": 0.03 + 0.29 * (index % 4),
                    "width": 0.10,
                    "height": 0.10,
                    "created_by": "Reviewer",
                }
            )
    session["examples"] = examples
    return session


def test_material_examples_are_atomic_versioned_and_prediction_is_unreviewed(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _image(tmp_path / "source.png")
    source_hash = sha256_file(source)
    session = _ready_session(project, source)

    quality = facies_example_quality(session)
    assert quality["ready_for_guided_prediction"] is True
    assert quality["example_count"] == 16

    store = FaciesExampleStore(project)
    saved = store.save(session, actor="Reviewer", reason="Initial examples")
    assert store.current_path.is_file()
    assert (store.revisions_root / f"{saved['revision_id']}.json").is_file()
    assert len(store.history_path.read_text(encoding="utf-8").splitlines()) == 1

    result = create_facies_example_prediction(
        store.current(),
        tmp_path / "predictions",
        maximum_width=120,
        pixels_per_box=30,
        n_estimators=8,
        cpu_workers=1,
    )
    assert result["format"] == FACIES_EXAMPLE_PREDICTION_FORMAT
    assert result["status"] == "unreviewed_model_proposal"
    assert result["scientific_scope"]["original_sources_modified"] is False
    assert all(region["enabled"] is False for region in result["regions"])
    with np.load(Path(result["prediction_path"]) / "facies.npz") as archive:
        assert archive["array"].shape == (80, 120)
    assert sha256_file(source) == source_hash

    original_config = {"width_m": 1.0, "height_m": 0.5, "regions": []}
    untouched = deepcopy(original_config)
    updated = facies_config_from_example_prediction(
        original_config, Path(result["manifest_path"])
    )
    assert original_config == untouched
    assert updated["facies_generation"]["status"] == "unreviewed_model_proposal"
    assert Path(updated["labels_path"]).is_file()


def test_blind_first_uses_a_verified_compatible_model(tmp_path: Path) -> None:
    project = _project(tmp_path)
    source = _image(tmp_path / "source.png")
    session = _ready_session(project, source)
    guided = create_facies_example_prediction(
        session,
        tmp_path / "guided",
        maximum_width=120,
        pixels_per_box=30,
        n_estimators=8,
        cpu_workers=1,
    )
    local_payload = joblib.load(
        Path(guided["prediction_path"]) / "local_model.joblib"
    )
    model_root = tmp_path / "shared-model"
    model_root.mkdir()
    model_path = model_root / "model.joblib"
    joblib.dump(
        {
            **local_payload,
            "format": FACIES_MODEL_FORMAT,
            "config": {"prediction_tile_size": 64, "prediction_halo": 14},
        },
        model_path,
    )
    manifest = {
        "schema": 1,
        "format": FACIES_MODEL_FORMAT,
        "model_id": "verified-shared-model",
        "artifacts": {
            "model": {
                "relative_path": model_path.name,
                "sha256": sha256_file(model_path),
            }
        },
    }
    manifest_path = model_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    blind_session = default_facies_example_session(project)
    blind_session["source_image_path"] = str(source)
    blind_session["start_strategy"] = "blind_model_first"
    blind_session["pretrained_model_manifest"] = str(manifest_path)
    result = create_facies_example_prediction(
        blind_session,
        tmp_path / "blind",
        maximum_width=120,
        cpu_workers=1,
    )
    assert result["mode"] == "blind_model_first"
    assert result["source_model"]["model_id"] == "verified-shared-model"
    assert result["quality"]["example_count"] == 0
