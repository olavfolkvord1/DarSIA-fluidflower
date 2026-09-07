from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.execution import (
    ExecutionQueue,
    build_task_document,
    default_adapter_registry,
    execute_task_lease,
)
from darsia.fluidflower.mass_acceleration import NumpyMassComputeSession
from darsia.fluidflower.mass_analysis import (
    MassAnalysisPlanStore,
    default_mass_analysis_plan,
    validate_mass_analysis_plan,
)
from darsia.fluidflower.mass_artifacts import load_field_index
from darsia.fluidflower.mass_materialization import (
    MASS_PROVENANCE_FORMAT,
    AcceleratedMassAnalysisAdapter,
    CpuFloat32MassAnalysisAdapter,
    MassAnalysisAdapter,
    materialize_mass_run,
)
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.quality_budget import default_quality_profile
from darsia.fluidflower.wasserstein import inspect_mass_analysis_result
from tests.unit.test_fluidflower_mass_analysis import (
    _approve_calibration_result,
    _approve_upstream,
    _calibration_result,
    _confirmed_mass_plan,
    _mass_input_manifest,
    _project_with_calibration_plan,
)


def test_mass_adapter_materializes_raw_phase_resolved_mass(tmp_path: Path) -> None:
    project, phase_revision, calibration_plan = _project_with_calibration_plan(tmp_path)
    _approve_upstream(project)
    result_path = _calibration_result(tmp_path, calibration_plan)
    _approve_calibration_result(project, tmp_path, result_path)
    config = _confirmed_mass_plan(project)
    config["calibration_result_manifest_path"] = str(result_path)
    config["mass_input_manifest_path"] = str(
        _mass_input_manifest(tmp_path, phase_revision, config["run_bindings"])
    )
    config["execution_profile"]["primary_backend"] = "cpu-float64"
    config["execution_profile"]["workers"] = 2
    config["exports"]["numerical_dtype"] = "float64"
    validation = validate_mass_analysis_plan(project, config)
    assert validation["execution_ready"]
    plan = MassAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic generic mass materialization",
        validation=validation,
    )
    plan_path = (
        project.root
        / "results"
        / "mass_analysis"
        / "plans"
        / plan["plan_id"]
        / "plan.json"
    )
    task = build_task_document(
        project=project,
        module="mass_analysis",
        operation="execute_saved_plan",
        adapter=MassAnalysisAdapter.descriptor,
        plan_reference={
            "plan_id": plan["plan_id"],
            "path": str(plan_path),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
            "contract_format": plan["task_contract"]["format"],
            "contract_sha256": plan["task_contract"]["content_sha256"],
        },
        backend="cpu-float64",
        resource_request={"cpu_workers": 2, "preparation_workers": 1},
        output_root=project.root / "results" / "execution" / "mass-analysis",
        created_by="Reviewer",
        max_attempts=1,
    )
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "mass-worker",
        backends=["cpu-float64"],
        adapters=[MassAnalysisAdapter.descriptor.key],
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
    manifest_path = destination / "mass_analysis_result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"]
    assert manifest["raw_mass_is_authoritative"]
    assert not manifest["mass_conservation_forced"]
    assert len(manifest["run_results"]) == 2
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["format"] == MASS_PROVENANCE_FORMAT
    assert provenance["scientific_contract"]["chemistry_engine"] == "darsia_titration"

    inspection = inspect_mass_analysis_result(str(manifest_path), plan)
    assert inspection["valid"]
    for run in manifest["run_results"]:
        products = run["products"]
        loaded = {}
        for product in ("mass_total", "mass_aqueous", "mass_free_gas"):
            reference = products[product]
            loaded[product] = load_field_index(
                destination / reference["path"],
                reference["sha256"],
                run_id=run["run_id"],
                product=product,
                expected_times_hours=run["target_times_hours"],
            )
        total_0, _ = loaded["mass_total"].load_frame(0)
        total_1, _ = loaded["mass_total"].load_frame(1)
        aqueous_1, _ = loaded["mass_aqueous"].load_frame(1)
        gas_1, _ = loaded["mass_free_gas"].load_frame(1)
        assert np.allclose(total_0, 0.0)
        assert float(total_1.sum()) > 0.0
        assert float(aqueous_1.sum()) > 0.0
        assert float(gas_1.sum()) > 0.0
        assert np.allclose(total_1, aqueous_1 + gas_1, rtol=1.0e-12, atol=1.0e-14)

        transport_reference = products["transport_weight"]
        transport = load_field_index(
            destination / transport_reference["path"],
            transport_reference["sha256"],
            run_id=run["run_id"],
            product="transport_weight",
        )
        transport_values, _ = transport.load_frame(0)
        assert np.all(np.isfinite(transport_values))
        assert np.all(transport_values > 0.0)

        integrated_path = destination / products["integrated_mass_csv"]["path"]
        with integrated_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert float(rows[1]["detected_mass_total_g"]) != float(
            rows[1]["injected_mass_g"]
        )


@pytest.mark.parametrize(
    ("backend", "adapter", "resource_key"),
    [
        ("cpu-float32", CpuFloat32MassAnalysisAdapter, "cpu_workers"),
        ("cuda-float32", AcceleratedMassAnalysisAdapter, "accelerator_workers"),
    ],
)
def test_nonreference_adapter_requires_and_records_cpu_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    adapter: type[MassAnalysisAdapter],
    resource_key: str,
) -> None:
    project, phase_revision, calibration_plan = _project_with_calibration_plan(tmp_path)
    _approve_upstream(project)
    result_path = _calibration_result(tmp_path, calibration_plan)
    _approve_calibration_result(project, tmp_path, result_path)
    config = _confirmed_mass_plan(project)
    config["calibration_result_manifest_path"] = str(result_path)
    config["mass_input_manifest_path"] = str(
        _mass_input_manifest(tmp_path, phase_revision, config["run_bindings"])
    )
    config["execution_profile"]["primary_backend"] = backend
    config["execution_profile"]["workers"] = 2
    config["exports"]["numerical_dtype"] = "float32"
    config["quality_profile"]["precision"]["compute_dtype"] = "float32"
    config["quality_profile"]["precision"]["cpu_float64_parity_fraction"] = 0.34
    validation = validate_mass_analysis_plan(project, config)
    assert validation["execution_ready"]
    plan = MassAnalysisPlanStore(project).save_plan(
        config,
        actor="Reviewer",
        reason="Synthetic accelerated mass materialization",
        validation=validation,
    )
    plan_path = (
        project.root
        / "results"
        / "mass_analysis"
        / "plans"
        / plan["plan_id"]
        / "plan.json"
    )
    task = build_task_document(
        project=project,
        module="mass_analysis",
        operation="execute_saved_plan",
        adapter=adapter.descriptor,
        plan_reference={
            "plan_id": plan["plan_id"],
            "path": str(plan_path),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
            "contract_format": plan["task_contract"]["format"],
            "contract_sha256": plan["task_contract"]["content_sha256"],
        },
        backend=backend,
        resource_request={
            resource_key: 2,
            "preparation_workers": 2,
            "device_id": "" if backend.startswith("cpu-") else "0",
        },
        output_root=project.root / "results" / "execution" / "accelerated-mass",
        created_by="Reviewer",
        max_attempts=1,
    )
    monkeypatch.setattr(
        "darsia.fluidflower.mass_materialization.create_mass_compute_session",
        lambda **kwargs: NumpyMassComputeSession(**kwargs),
    )
    queue = ExecutionQueue(tmp_path / "accelerated-queue")
    assert queue.enqueue(task)
    lease = queue.claim(
        "accelerated-mass-worker",
        backends=[backend],
        adapters=[adapter.descriptor.key],
    )
    assert lease is not None
    result = execute_task_lease(
        lease,
        registry=default_adapter_registry(),
        local_cache_root=tmp_path / "accelerated-cache",
        inventory={
            "schema": 1,
            "format": "darsia-fluidflower-hardware-v1",
            "content_sha256": "accelerated-test-inventory",
            "host": "gpu-test-host",
            "cpu": {"logical_cores": 8, "physical_cores": 4},
            "memory": {},
            "accelerators": [
                {
                    "backend": "cuda-float32",
                    "supported_backends": ["cuda-float32", "cuda-float64"],
                    "device_id": "0",
                }
            ],
            "available_backends": ["cpu-float64", "cpu-float32", backend],
        },
        heartbeat_seconds=0.01,
    )

    destination = Path(result["destination"])
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["backend"]["name"] == backend
    assert (
        provenance["scientific_contract"]["quality_summary"]["compute_dtype"]
        == "float32"
    )
    assert provenance["cpu_float64_parity"]["passed"]
    assert provenance["cpu_float64_parity"]["sample_count"] == 4
    manifest = json.loads((destination / "mass_analysis_result.json").read_text())
    assert manifest["quality_contract"]["backend"] == backend
    assert all(run["cpu_float64_parity"]["passed"] for run in manifest["run_results"])
    first_mass = load_field_index(
        destination / manifest["run_results"][0]["products"]["mass_total"]["path"],
        manifest["run_results"][0]["products"]["mass_total"]["sha256"],
        run_id=manifest["run_results"][0]["run_id"],
        product="mass_total",
        expected_times_hours=manifest["run_results"][0]["target_times_hours"],
    )
    values, _ = first_mass.load_frame(0)
    assert values.dtype == np.float32


def test_single_run_materialization_honors_resolved_source_indices(
    tmp_path: Path,
) -> None:
    project, phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = default_mass_analysis_plan(project)
    manifest_path = _mass_input_manifest(
        tmp_path, phase_revision, config["run_bindings"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["run_inputs"][0]
    parameters = {
        "transfer_shape": "smoothstep",
        "acid_low": 0.1,
        "acid_high": 0.9,
        "co2_onset": 0.1,
        "co2_high": 0.7,
        "gas_onset": 0.2,
        "gas_high": 0.8,
        "max_gas_saturation": 0.45,
        "initial_pH": 8.1,
        "aqueous_blend": 0.0,
        "diffusion_beta": 0.0,
    }

    result = materialize_mass_run(
        staging=tmp_path / "selected-output",
        manifest_path=manifest_path,
        row=row,
        model="titration",
        parameters=parameters,
        products_requested={
            "mass_total",
            "mass_aqueous",
            "mass_free_gas",
            "transport_weight",
            "integrated_mass_csv",
        },
        selected_time_indices=[0, 2],
    )

    assert result["source_time_indices"] == [0, 2]
    assert result["target_times_hours"] == [0.0, 2.0]
    assert len(result["integrated_mass"]) == 2
    assert [row["source_time_index"] for row in result["integrated_mass"]] == [
        0,
        2,
    ]


def test_single_run_materialization_uses_recorded_reduced_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, phase_revision, _calibration_plan = _project_with_calibration_plan(
        tmp_path
    )
    config = default_mass_analysis_plan(project)
    manifest_path = _mass_input_manifest(
        tmp_path, phase_revision, config["run_bindings"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = default_quality_profile("exploratory")
    quality["precision"]["compute_dtype"] = "float32"
    monkeypatch.setattr(
        "darsia.fluidflower.mass_materialization.create_mass_compute_session",
        lambda **kwargs: NumpyMassComputeSession(**kwargs),
    )
    staging = tmp_path / "reduced-output"
    result = materialize_mass_run(
        staging=staging,
        manifest_path=manifest_path,
        row=manifest["run_inputs"][0],
        model="titration",
        parameters={
            "transfer_shape": "smoothstep",
            "acid_low": 0.1,
            "acid_high": 0.9,
            "co2_onset": 0.1,
            "co2_high": 0.7,
            "gas_onset": 0.2,
            "gas_high": 0.8,
            "max_gas_saturation": 0.45,
            "initial_pH": 8.1,
            "aqueous_blend": 0.0,
            "diffusion_beta": 0.0,
        },
        products_requested={
            "mass_total",
            "mass_aqueous",
            "mass_free_gas",
            "transport_weight",
            "integrated_mass_csv",
        },
        selected_time_indices=[0, 2],
        quality_profile=quality,
        numerical_dtype="float32",
        backend_name="cuda-float32",
    )

    assert result["geometry"]["source_shape_rc"] == [8, 12]
    assert result["geometry"]["shape_rc"] == [4, 6]
    assert result["geometry"]["resolution_policy"] == "scale_factor"
    assert result["geometry"]["numerical_dtype"] == "float32"
    assert result["cpu_float64_parity"]["passed"]
    assert result["cpu_float64_parity"]["sample_count"] == 1
    field_reference = result["products"]["mass_total"]
    index_path = staging / str(field_reference["path"])
    index = json.loads(index_path.read_text())
    frame_path = index_path.parent / str(index["frames"][0]["path"])
    with np.load(frame_path, allow_pickle=False) as archive:
        assert archive["values"].dtype == np.float32
