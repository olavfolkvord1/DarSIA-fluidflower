from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from darsia.fluidflower.execution import (
    ExecutionQueue,
    build_task_document,
    default_adapter_registry,
    execute_task_lease,
)
from darsia.fluidflower.mass_artifacts import (
    write_field_frame,
    write_field_index,
    write_preview_index,
)
from darsia.fluidflower.post_analysis import (
    REQUIRED_POST_ANALYSIS_PRODUCTS,
    PostAnalysisPlanStore,
    default_post_analysis_plan,
    expected_post_analysis_artifacts,
    inspect_post_analysis_result,
    validate_post_analysis_plan,
)
from darsia.fluidflower.post_materialization import PostAnalysisAdapter
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.wasserstein import MASS_ANALYSIS_RESULT_FORMAT
from tests.unit.test_fluidflower_mass_analysis import _project_with_mass_plan


def _field_values(
    product: str,
    frame_index: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    rows, columns = shape
    yy, xx = np.mgrid[:rows, :columns]
    radius = min(rows, columns) * (0.12 + frame_index * 0.08)
    plume = (xx - columns * 0.45) ** 2 + (yy - rows * 0.58) ** 2 <= radius**2
    gas = (xx - columns * 0.45) ** 2 + (yy - rows * 0.58) ** 2 <= (radius * 0.42) ** 2
    if product == "mass_total":
        values = plume.astype(np.float64) * frame_index * 0.001
        return values, None
    if product == "mass_aqueous":
        values = (plume & ~gas).astype(np.float64) * frame_index * 0.001
        return values, None
    if product == "mass_free_gas":
        values = gas.astype(np.float64) * frame_index * 0.001
        return values, None
    if product == "concentration_aqueous":
        display = (plume & ~gas).astype(np.float32) * (0.35 + frame_index * 0.15)
        return display * 1.7, display
    if product == "saturation_free_gas":
        values = gas.astype(np.float32) * (0.18 + frame_index * 0.08)
        return values, values
    if product == "transport_weight":
        values = np.ones(shape, dtype=np.float64)
        values[:, columns // 2 :] = 2.0
        return values, None
    raise ValueError(product)


def _indexed_product(
    root: Path,
    run_id: str,
    product: str,
    times: list[float],
    shape: tuple[int, int],
) -> dict:
    folder = root / run_id / product
    frames = []
    for frame_index, time_hours in enumerate(times):
        values, display = _field_values(product, frame_index, shape)
        if run_id == "ac18" and product.startswith("mass_"):
            values = np.roll(values, 3, axis=1)
        frames.append(
            write_field_frame(
                folder / f"frame-{frame_index:03d}.npz",
                values,
                run_id=run_id,
                product=product,
                frame_id=f"frame-{frame_index:03d}",
                time_hours=time_hours,
                display_values=display,
            )
        )
    unit = "g pixel-1" if product.startswith("mass_") else "dimensionless"
    return write_field_index(
        folder / "index.json",
        run_id=run_id,
        product=product,
        quantity=product,
        unit=unit,
        frames=frames,
        grid={
            "coordinate_system": "corrected_registered_rig_grid",
            "width_m": 2.8,
            "height_m": 1.5,
            "shape_rc": list(shape),
        },
    )


def _preview_product(
    root: Path,
    run_id: str,
    times: list[float],
    shape: tuple[int, int],
) -> dict:
    folder = root / run_id / "previews"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_index, time_hours in enumerate(times):
        image = np.zeros((*shape, 3), dtype=np.uint8)
        image[:, :] = (80 + frame_index * 4, 70, 35)
        path = folder / f"frame-{frame_index:03d}.png"
        assert cv2.imwrite(str(path), image)
        frames.append(
            {
                "frame_id": f"frame-{frame_index:03d}",
                "time_hours": time_hours,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    return write_preview_index(folder / "index.json", run_id=run_id, frames=frames)


def _mass_result_with_indexed_products(
    tmp_path: Path,
    mass_plan: dict,
) -> Path:
    root = tmp_path / "mass-products"
    times = [0.0, 1.0, 2.0]
    shape = (48, 72)
    run_results = []
    for binding in mass_plan["configuration"]["run_bindings"]:
        run_id = binding["run_id"]
        products = {
            product: _indexed_product(root, run_id, product, times, shape)
            for product in (
                "mass_total",
                "mass_aqueous",
                "mass_free_gas",
                "concentration_aqueous",
                "saturation_free_gas",
                "transport_weight",
            )
        }
        products["preview_images"] = _preview_product(root, run_id, times, shape)
        run_results.append(
            {
                "run_id": run_id,
                "complete": True,
                "target_times_hours": times,
                "injected_mass_g": [0.0, 5.0, 5.0],
                "reference_plume_radius_m": [0.0, 0.2, 0.2],
                "domains": ["whole_active_domain"],
                "geometry": {
                    "width_m": 2.8,
                    "height_m": 1.5,
                    "native_cell_size_m": 0.01,
                },
                "products": products,
            }
        )
    path = tmp_path / "mass_analysis_result.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "format": MASS_ANALYSIS_RESULT_FORMAT,
                "mass_analysis_plan_id": mass_plan["plan_id"],
                "mass_task_contract_sha256": mass_plan["task_contract"][
                    "content_sha256"
                ],
                "complete": True,
                "run_results": run_results,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_post_analysis_adapter_materializes_exact_planned_coverage(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    mass_result = _mass_result_with_indexed_products(tmp_path, mass_plan)
    config = default_post_analysis_plan(project)
    config["mass_result_manifest_path"] = str(mass_result)
    config["products"] = [
        *REQUIRED_POST_ANALYSIS_PRODUCTS,
        "mass_curve_full_domain",
        "mass_curve_analysis_domains",
        "phase_map_movie_corrected_background",
        "phase_contour_contact_sheet",
        "cross_run_mass_grid",
        "cross_run_phase_grid_movie",
    ]
    config["rendering"]["movie_width_pixels"] = 640
    config["rendering"]["movie_fps"] = 2.0
    config["execution_profile"]["frame_workers"] = 2
    validation = validate_post_analysis_plan(project, config)
    assert validation["execution_ready"]
    plan = PostAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic generic post-analysis materialization",
        validation=validation,
    )

    task = build_task_document(
        project=project,
        module="post_analysis",
        operation="execute_saved_plan",
        adapter=PostAnalysisAdapter.descriptor,
        plan_reference={
            "plan_id": plan["plan_id"],
            "path": str(
                project.root
                / "results"
                / "post_analysis"
                / "plans"
                / plan["plan_id"]
                / "plan.json"
            ),
            "file_sha256": sha256_file(
                project.root
                / "results"
                / "post_analysis"
                / "plans"
                / plan["plan_id"]
                / "plan.json"
            ),
            "content_sha256": plan["content_sha256"],
            "contract_format": plan["task_contract"]["format"],
            "contract_sha256": plan["task_contract"]["content_sha256"],
        },
        backend="cpu-float64",
        resource_request={"cpu_workers": 2, "preparation_workers": 1},
        output_root=project.root / "results" / "execution" / "post-analysis",
        created_by="Reviewer",
        max_attempts=1,
    )
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "post-worker",
        backends=["cpu-float64"],
        adapters=["post-analysis-v1"],
    )
    assert lease is not None
    result = execute_task_lease(
        lease,
        registry=default_adapter_registry(),
        local_cache_root=tmp_path / "cache",
        inventory={
            "schema": 1,
            "format": "darsia-fluidflower-hardware-v1",
            "content_sha256": "test-inventory",
            "host": "test-host",
            "cpu": {"logical_cores": 8, "physical_cores": 4},
            "memory": {},
            "accelerators": [],
            "available_backends": ["cpu-float64"],
        },
        heartbeat_seconds=0.01,
    )
    destination = Path(result["destination"])
    result_path = destination / "post_analysis_result.json"
    inspection = inspect_post_analysis_result(str(result_path), plan)
    assert inspection["valid"]
    assert inspection["reference"]["artifact_count"] == len(
        expected_post_analysis_artifacts(config)
    )

    csv_path = destination / "runs" / config["run_ids"][0] / "integrated_mass.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    final = [row for row in rows if row["domain_id"] == "whole_active_domain"][-1]
    assert float(final["mass_total_g"]) < float(final["injected_mass_reference_g"])
    movie = destination / "runs" / config["run_ids"][0] / "phase_map_corrected.mp4"
    capture = cv2.VideoCapture(str(movie))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None and frame.shape[1] == 640
