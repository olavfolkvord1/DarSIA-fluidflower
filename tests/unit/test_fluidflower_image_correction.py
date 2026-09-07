from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import darsia
from darsia.fluidflower.execution import (
    ExecutionQueue,
    build_task_document,
    default_adapter_registry,
    execute_task_lease,
)
from darsia.fluidflower.image_correction import (
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    CorrectionRevisionStore,
    apply_affine,
    apply_darsia_baseline_array_correction,
    apply_neutral_tone,
    apply_uniform24_array_transform,
    darsia_baseline_correction_config,
    default_correction_config,
    discover_correction_frames,
    discover_project_correction_frames,
    fit_neutral_tone_model,
    fit_regularized_affine,
    identity_correction_config,
    residual_summary,
)
from darsia.fluidflower.image_correction_materialization import (
    ImageCorrectionAdapter,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file


def create_project(tmp_path: Path) -> FluidFlowerProject:
    return FluidFlowerProject.create(
        tmp_path / "project",
        name="Correction test",
        series_id="correction-test",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )


def synthetic_checker() -> np.ndarray:
    values = np.linspace(0.06, 0.94, 72, dtype=np.float32)
    checker = values.reshape(4, 6, 3)
    checker[-1] = np.linspace(0.92, 0.08, 18, dtype=np.float32).reshape(6, 3)
    return checker


def test_uniform24_components_reduce_synthetic_residual() -> None:
    reference = synthetic_checker()
    source = np.clip(reference * np.array([0.78, 0.88, 1.06]) + 0.025, 0.0, 1.0)
    before = residual_summary(source, reference)["mean"]
    tone = fit_neutral_tone_model(source, reference)
    corrected = apply_neutral_tone(source, tone)
    scaling, translation, _ = fit_regularized_affine(
        corrected,
        reference,
        ridge=0.01,
        huber_delta=0.025,
    )
    corrected = apply_affine(corrected, scaling, translation)
    after = residual_summary(corrected, reference)["mean"]
    assert after < before


def test_original_darsia_correction_is_an_explicit_legacy_comparator() -> None:
    height, width = 326, 500
    reference = np.full((height, width, 3), 0.15, dtype=np.float32)
    rows = (12, 93, 175, 255)
    columns = (12, 95, 177, 260, 344, 427)
    swatches = synthetic_checker()
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            reference[row : row + 50, column : column + 50] = swatches[
                row_index, column_index
            ]
    target = np.clip(
        reference * np.asarray([0.80, 0.90, 1.05], dtype=np.float32) + 0.02,
        0.0,
        1.0,
    )

    result = apply_darsia_baseline_array_correction(
        reference,
        target,
        roi_rc=[[0, 0], [height, 0], [height, width], [0, width]],
    )

    assert result["transform"]["algorithm"] == DARSIA_BASELINE_COLOR_ALGORITHM
    assert result["corrected"].shape == target.shape
    assert result["diagnostics"]["after"]["mean"] < result["diagnostics"]["before"]["mean"]
    assert (
        darsia_baseline_correction_config()["settings"]["clip"] is False
    )


def test_correction_revisions_are_append_only(tmp_path: Path) -> None:
    project = create_project(tmp_path)
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"placeholder")
    store = CorrectionRevisionStore(project)
    first = default_correction_config()
    first["reference_image"] = str(reference)
    revision_1 = store.save_revision(
        first,
        actor="Reviewer",
        reason="Initial setup",
    )
    second = default_correction_config()
    second["reference_image"] = str(reference)
    second["settings"]["ridge"] = 0.02
    revision_2 = store.save_revision(
        second,
        actor="Reviewer",
        reason="Stronger regularization",
    )

    assert revision_1["revision_id"] != revision_2["revision_id"]
    assert len(store.revisions()) == 2
    assert store.current()["settings"]["ridge"] == 0.02
    assert (store.artifact_root / revision_2["revision_id"] / "plan.json").is_file()


def test_original_darsia_correction_revision_is_execution_ready(tmp_path: Path) -> None:
    project = create_project(tmp_path)
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"placeholder")
    config = darsia_baseline_correction_config()
    config["reference_image"] = str(reference)
    config["batch"]["frames"] = [
        {
            "frame_id": "frame-1",
            "run_id": "ac52",
            "source_path": "frame.jpg",
            "sha256": "unused-in-plan-test",
        }
    ]

    revision = CorrectionRevisionStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="Preserve the original DarSIA correction as a comparator",
    )

    assert revision["algorithm"] == DARSIA_BASELINE_COLOR_ALGORITHM
    assert revision["validation"]["execution_ready"] is True
    assert revision["task_contract"]["reference"]["sha256"]


def test_changed_correction_invalidates_project_and_run_descendants(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    project.set_runs([{"run_id": "ac17", "label": "AC17"}])
    store = CorrectionRevisionStore(project)
    first_config = identity_correction_config()
    first = store.save_revision(
        first_config,
        actor="Reviewer",
        reason="Initial identity control",
    )
    workflow = (
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
    )
    for module in workflow:
        project.set_module_status(module, "complete", updated_by="Reviewer")
        project.set_run_module_status(
            ["ac17"], module, "complete", updated_by="Reviewer"
        )

    changed_config = identity_correction_config()
    changed_config["batch"]["output_encoding"] = "npy-float32"
    second = store.save_revision(
        changed_config,
        actor="Reviewer",
        reason="Alternative correction output contract",
    )

    assert first["dependency_sha256"] != second["dependency_sha256"]
    assert project.module_states["image_correction"]["status"] == "review_required"
    assert project.module_states["geometry_facies"]["status"] == "stale"
    assert project.module_states["optical_calibration"]["status"] == "stale"
    assert project.module_states["phase_interpretation"]["status"] == "stale"
    assert project.module_states["calibration"]["status"] == "stale"
    assert project.module_states["mass_analysis"]["status"] == "stale"
    assert project.module_states["wasserstein"]["status"] == "stale"
    run_workflow = project.manifest["runs"][0]["workflow"]
    assert run_workflow["image_correction"]["status"] == "review_required"
    assert run_workflow["phase_interpretation"]["status"] == "stale"
    assert run_workflow["mass_analysis"]["status"] == "stale"
    assert (store.artifact_root / first["revision_id"] / "plan.json").is_file()


def test_no_external_color_target_is_an_explicit_executable_revision(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    config = identity_correction_config()
    config["representative_run_id"] = "ac52"
    config["representative_preview_role"] = "sequence_1_last"
    config["target_image"] = "frame.jpg"
    config["batch"]["frames"] = [
        {
            "frame_id": "frame-1",
            "run_id": "ac52",
            "source_path": "frame.jpg",
            "sha256": "unused-in-plan-test",
        }
    ]

    revision = CorrectionRevisionStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="No ColorChecker was acquired",
    )

    assert revision["algorithm"] == IDENTITY_ALGORITHM
    assert revision["task_contract"]["application_scope"] == "series"
    assert revision["task_contract"]["representative_run_id"] == "ac52"
    assert (
        revision["task_contract"]["representative_preview_role"]
        == "sequence_1_last"
    )
    assert revision["task_contract"]["reference"] == {"path": "", "sha256": ""}
    assert revision["validation"]["execution_ready"] is True


def test_no_target_materialization_preserves_native_image_values(
    tmp_path: Path,
) -> None:
    project = create_project(tmp_path)
    source_root = tmp_path / "images" / "ac52"
    source_root.mkdir(parents=True)
    source = source_root / "frame.png"
    source_bgr = np.array(
        [
            [[4, 32, 220], [17, 155, 81]],
            [[240, 90, 11], [101, 73, 52]],
        ],
        dtype=np.uint8,
    )
    assert cv2.imwrite(str(source), source_bgr)

    config = identity_correction_config()
    config["batch"].update(
        {
            "source_root": str(source_root.parent),
            "output_encoding": "npy-float32",
            "frames": discover_correction_frames(source_root.parent),
        }
    )
    revision = CorrectionRevisionStore(project).save_revision(
        config,
        actor="Reviewer",
        reason="No external color target exists",
    )
    plan_path = (
        project.root
        / "results"
        / "image_correction"
        / "plans"
        / revision["revision_id"]
        / "plan.json"
    )
    task = build_task_document(
        project=project,
        module="image_correction",
        operation="execute_saved_plan",
        adapter=ImageCorrectionAdapter.descriptor,
        plan_reference={
            "plan_id": revision["revision_id"],
            "path": str(plan_path),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": revision["content_sha256"],
            "contract_format": revision["task_contract"]["format"],
            "contract_sha256": revision["task_contract"]["content_sha256"],
        },
        backend="cpu-float64",
        resource_request={"cpu_workers": 1, "preparation_workers": 1},
        output_root=project.root / "results" / "execution" / "image-correction",
        created_by="Reviewer",
        max_attempts=1,
    )
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "correction-worker",
        backends=["cpu-float64"],
        adapters=[ImageCorrectionAdapter.descriptor.key],
    )
    assert lease is not None
    execution = execute_task_lease(
        lease,
        registry=default_adapter_registry(),
        local_cache_root=tmp_path / "cache",
        inventory={
            "schema": 1,
            "format": "darsia-fluidflower-hardware-v1",
            "content_sha256": "test-inventory",
            "host": "test-host",
            "cpu": {"logical_cores": 4, "physical_cores": 2},
            "memory": {},
            "accelerators": [],
            "available_backends": ["cpu-float64"],
        },
        heartbeat_seconds=0.01,
    )

    destination = Path(execution["destination"])
    index = json.loads((destination / "corrected_frames.json").read_text())
    quality = json.loads((destination / "frame_quality.json").read_text())
    output = np.load(destination / index["frames"][0]["corrected_path"])
    expected = np.asarray(darsia.imread(source).img, dtype=np.float32)
    assert np.array_equal(output, expected)
    assert quality["frame_count"] == 1
    assert quality["frames"][0]["automatic_status"] == "ok"
    assert index["frames"][0]["frame_quality"]["automatically_excluded"] is False
    assert index["quality_assessment"]["advisory_only"] is True
    provenance = json.loads((destination / "provenance.json").read_text())
    assert (
        provenance["scientific_contract"][
            "indicator_buffer_reference_used_for_radiometry"
        ]
        is False
    )


def test_batch_inventory_is_run_aware_and_content_addressed(tmp_path: Path) -> None:
    root = tmp_path / "images"
    (root / "run18").mkdir(parents=True)
    (root / "run20").mkdir(parents=True)
    (root / "run18" / "frame.jpg").write_bytes(b"frame-18")
    (root / "run20" / "frame.jpg").write_bytes(b"frame-20")

    frames = discover_correction_frames(root, recursive=True)

    assert [row["run_id"] for row in frames] == ["run18", "run20"]
    assert len({row["frame_id"] for row in frames}) == 2
    assert all(len(row["sha256"]) == 64 for row in frames)


def test_project_inventory_uses_only_current_included_run_mappings(
    tmp_path: Path,
) -> None:
    series = tmp_path / "series"
    included = series / "AC17" / "CO2 injection"
    excluded = series / "AC14" / "CO2 injection"
    included.mkdir(parents=True)
    excluded.mkdir(parents=True)
    included_jpg = included / "frame001.JPG"
    included_raw = included / "frame001.ARW"
    excluded_jpg = excluded / "frame001.JPG"
    included_jpg.write_bytes(b"included jpeg")
    included_raw.write_bytes(b"included raw")
    excluded_jpg.write_bytes(b"excluded jpeg")
    project = FluidFlowerProject.create(
        tmp_path / "project-mapped",
        name="Mapped correction",
        series_id="mapped-correction",
        template_id="ff_ac",
        rig_type="medium",
        image_sources={"experiment_images": series},
        created_by="Reviewer",
    )
    project.set_runs(
        [
            {
                "run_id": "ac17",
                "source_mapping": {
                    "run_id": "ac17",
                    "experiment_folder": str(series / "AC17"),
                    "experiment_folders": [str(included)],
                    "candidate_status": "user_confirmed",
                },
            }
        ],
        updated_by="Reviewer",
    )

    frames = discover_project_correction_frames(project)

    assert [Path(row["source_path"]) for row in frames] == [included_jpg.resolve()]
    assert all("AC14" not in str(row["source_path"]) for row in frames)

    project.update_metadata(
        "tools",
        {"source_mapping": {"image_file_policy": "prefer_raw"}},
        updated_by="Reviewer",
    )
    with pytest.raises(ValueError, match="validated RAW decoder"):
        discover_project_correction_frames(project)


def test_serialized_uniform24_transform_preserves_native_shape() -> None:
    image = np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(4, 5, 3)
    transform = {
        "neutral_tone_enabled": False,
        "affine_enabled": True,
        "affine_scaling": np.eye(3).tolist(),
        "affine_translation": [0.0, 0.0, 0.0],
        "clip": True,
    }

    corrected = apply_uniform24_array_transform(image, transform)

    assert corrected.shape == image.shape
    assert corrected.dtype == np.float32
    assert np.allclose(corrected, image, atol=2.0e-6)
