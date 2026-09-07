from __future__ import annotations

import csv
import json
from pathlib import Path

from darsia.fluidflower.execution import (
    ExecutionQueue,
    build_task_document,
    default_adapter_registry,
    execute_task_lease,
)
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.wasserstein import (
    WassersteinPlanStore,
    default_wasserstein_plan,
    validate_wasserstein_plan,
)
from darsia.fluidflower.wasserstein_materialization import (
    WASSERSTEIN_RESULT_FORMAT,
    WassersteinAnalysisAdapter,
)
from tests.unit.test_fluidflower_mass_analysis import _project_with_mass_plan
from tests.unit.test_fluidflower_post_materialization import (
    _mass_result_with_indexed_products,
)


def test_wasserstein_adapter_materializes_physical_pairwise_distances(
    tmp_path: Path,
) -> None:
    project, mass_plan = _project_with_mass_plan(tmp_path, ready=True)
    mass_result = _mass_result_with_indexed_products(tmp_path, mass_plan)
    config = default_wasserstein_plan(project)
    config["mass_result_manifest_path"] = str(mass_result)
    config["spatial_discretization"]["target_cell_size_m"] = 0.1
    config["comparison"]["time_selection_budget"]["maximum_time_points"] = 2
    config["execution_profile"]["workers"] = 1
    validation = validate_wasserstein_plan(project, config)
    assert validation["execution_ready"]
    assert validation["summary"]["task_count"] == 6
    plan = WassersteinPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic weighted Wasserstein materialization",
        validation=validation,
    )
    plan_path = (
        project.root
        / "results"
        / "wasserstein"
        / "plans"
        / plan["plan_id"]
        / "plan.json"
    )
    task = build_task_document(
        project=project,
        module="wasserstein",
        operation="execute_saved_plan",
        adapter=WassersteinAnalysisAdapter.descriptor,
        plan_reference={
            "plan_id": plan["plan_id"],
            "path": str(plan_path),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
            "contract_format": plan["task_contract"]["format"],
            "contract_sha256": plan["task_contract"]["content_sha256"],
        },
        backend="cpu-float64",
        resource_request={"cpu_workers": 1, "preparation_workers": 1},
        output_root=project.root / "results" / "execution" / "wasserstein",
        created_by="Reviewer",
        max_attempts=1,
    )
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "wasserstein-worker",
        backends=["cpu-float64"],
        adapters=[WassersteinAnalysisAdapter.descriptor.key],
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
    result_manifest = json.loads(
        (destination / "wasserstein_result.json").read_text(encoding="utf-8")
    )
    assert result_manifest["format"] == WASSERSTEIN_RESULT_FORMAT
    assert result_manifest["complete"]
    assert result_manifest["comparison_count"] == 6
    assert (destination / "wasserstein_distances.xlsx").is_file()
    assert (destination / "intermediate" / "index.json").is_file()

    with (destination / "wasserstein_distances.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    baseline = [row for row in rows if row["time_index"] == "0"]
    assert all(row["status"] == "both_zero_mass" for row in baseline)
    shifted = next(
        row
        for row in rows
        if row["time_index"] == "2" and row["mass_component"] == "mass_total"
    )
    assert shifted["status"] in {"success", "solver_not_converged"}
    assert float(shifted["transport_cost_mass_m"]) > 0.0
    assert abs(float(shifted["absolute_mass_difference_g"])) < 1.0e-12
    assert float(shifted["mass_based_distance_m"]) > 0.0
