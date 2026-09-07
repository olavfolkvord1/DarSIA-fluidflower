from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from darsia.fluidflower.execution import (
    BACKEND_CONTRACTS,
    AdapterDescriptor,
    ExecutionQueue,
    PlanVerificationAdapter,
    TaskExecutionContext,
    artifact_manifest,
    build_task_document,
    current_plan_reference,
    execute_task_lease,
    publish_directory_atomic,
    recommend_resources,
    spawn_worker_processes,
    task_for_current_plan,
    validate_task_document,
)
from darsia.fluidflower.project import FluidFlowerProject, sha256_file


def _project(tmp_path: Path) -> FluidFlowerProject:
    return FluidFlowerProject.create(
        tmp_path / "project",
        name="Execution test",
        series_id="execution-test",
        template_id="ff_ac",
        rig_type="medium",
        created_by="Reviewer",
    )


def _inventory() -> dict:
    return {
        "schema": 1,
        "format": "darsia-fluidflower-hardware-v1",
        "content_sha256": "inventory-test",
        "host": "test-host",
        "cpu": {"logical_cores": 16, "physical_cores": 8},
        "memory": {"total_bytes": 32 * 1024**3, "available_bytes": 24 * 1024**3},
        "accelerators": [
            {
                "backend": "cuda-float32",
                "device_id": "0",
                "name": "Test GPU",
                "memory_bytes": 16 * 1024**3,
            }
        ],
        "available_backends": ["cpu-float64", "cuda-float32"],
    }


def _task(
    tmp_path: Path,
    *,
    maximum_attempts: int = 3,
) -> tuple[FluidFlowerProject, dict]:
    project = _project(tmp_path)
    plan_path = (
        project.root / "results" / "calibration" / "plans" / "plan-1" / "plan.json"
    )
    plan_path.parent.mkdir(parents=True)
    plan = {
        "schema": 1,
        "format": "test-plan-v1",
        "plan_id": "plan-1",
        "task_contract": {
            "format": "test-contract-v1",
            "content_sha256": "contract-test",
        },
        "content_sha256": "saved-plan-contract",
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    task = build_task_document(
        project=project,
        module="calibration",
        operation="verify_saved_plan",
        adapter=PlanVerificationAdapter.descriptor,
        plan_reference={
            "plan_id": "plan-1",
            "path": str(plan_path),
            "file_sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
            "contract_format": "test-contract-v1",
            "contract_sha256": "contract-test",
        },
        backend="cpu-float64",
        resource_request={"cpu_workers": 1, "preparation_workers": 1},
        output_root=project.root / "results" / "execution" / "verified",
        created_by="Reviewer",
        max_attempts=maximum_attempts,
    )
    return project, task


def test_resource_recommendation_keeps_gpu_headroom_and_matching_preparation() -> None:
    recommendation = recommend_resources("calibration", _inventory())

    assert recommendation["selected_backend"] == "cuda-float32"
    assert recommendation["reference_backend"] == "cpu-float64"
    assert recommendation["accelerator_workers"] == 2
    assert recommendation["preparation_workers"] == 2

    wasserstein = recommend_resources("wasserstein", _inventory())
    assert wasserstein["selected_backend"] == "cpu-float64"
    assert wasserstein["accelerator_workers"] == 0
    assert wasserstein["cpu_workers"] <= 12


def test_task_contract_detects_mutation(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    validate_task_document(task)

    changed = dict(task)
    changed["priority"] = 999
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_task_document(changed)


@pytest.mark.parametrize(
    ("module", "adapter_key"),
    [
        ("post_analysis", "post-analysis-v1"),
        ("wasserstein", "wasserstein-cpu-float64-v1"),
    ],
)
def test_real_task_uses_saved_plan_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    adapter_key: str,
) -> None:
    project = _project(tmp_path)
    output_root = tmp_path / "planned-output" / module
    plan_path = tmp_path / f"{module}-plan.json"
    plan_path.write_text(
        json.dumps({"task_contract": {"output_root": str(output_root)}}),
        encoding="utf-8",
    )
    reference = {
        "plan_id": f"{module}-plan",
        "path": str(plan_path),
        "file_sha256": sha256_file(plan_path),
        "content_sha256": "plan-contract",
        "contract_format": "test-contract-v1",
        "contract_sha256": "task-contract",
    }
    monkeypatch.setattr(
        "darsia.fluidflower.execution.current_plan_reference",
        lambda _project, _module: reference,
    )

    task = task_for_current_plan(
        project,
        module,
        adapter_key=adapter_key,
        backend="cpu-float64",
        created_by="Reviewer",
        resources={"cpu_workers": 1, "preparation_workers": 1},
    )

    assert Path(task["output"]["destination"]).parent == output_root.resolve()

    assert task["backend"] == {
        "name": BACKEND_CONTRACTS["cpu-float64"].name,
        "family": "cpu",
        "precision": "float64",
        "semantic_contract": "fluidflower-cpu-float64-reference-v1",
        "authoritative_reference": True,
    }


def test_accelerator_float64_backends_have_explicit_precision_contracts() -> None:
    for backend in ("cuda-float64", "opencl-float64"):
        contract = BACKEND_CONTRACTS[backend]
        assert contract.precision == "float64"
        assert not contract.authoritative_reference
        assert "float64" in contract.semantic_contract


def test_cpu_float32_is_available_but_not_an_authoritative_reference() -> None:
    contract = BACKEND_CONTRACTS["cpu-float32"]

    assert contract.family == "cpu"
    assert contract.precision == "float32"
    assert not contract.authoritative_reference
    assert "float32" in contract.semantic_contract


def test_queue_snapshot_exposes_immutable_plan_quality_contract(tmp_path: Path) -> None:
    project, task = _task(tmp_path)
    quality = {
        "backend": "cpu-float64",
        "compute_dtype": "float64",
        "scientific_grid": "native registered grid",
        "preview": "max 1920 px, JPEG 92",
        "cpu_reference_dtype": "float64",
        "label": "reference quality",
    }
    plan_reference = dict(task["plan_reference"])
    plan_reference["quality_contract"] = quality
    rebuilt = build_task_document(
        project=project,
        module=str(task["module"]),
        operation=str(task["operation"]),
        adapter=PlanVerificationAdapter.descriptor,
        plan_reference=plan_reference,
        backend="cpu-float64",
        resource_request=dict(task["resources"]),
        output_root=Path(task["output"]["destination"]).parent,
        created_by=str(task["created_by"]),
    )
    queue = ExecutionQueue(tmp_path / "quality-queue")
    queue.enqueue(rebuilt)

    row = queue.snapshot()["tasks"][0]

    assert row["quality_contract"] == quality


def test_calibration_plan_reference_resolves_backend_from_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = {
        "schema": 1,
        "format": "darsia-fluidflower-quality-profile-v1",
        "name": "production",
        "numerical_resolution": {
            "policy": "native_registered_grid",
        },
        "precision": {"cpu_reference_dtype": "float64"},
        "preview": {"maximum_dimension_px": 1920, "jpeg_quality": 92},
    }
    plan_id = "calibration-plan"
    artifact_root = tmp_path / "plans"
    plan_path = artifact_root / plan_id / "plan.json"
    plan_path.parent.mkdir(parents=True)
    plan = {
        "plan_id": plan_id,
        "content_sha256": "saved-plan",
        "objective_contract": {
            "format": "objective-v1",
            "content_sha256": "objective-sha",
            "quality_profile": profile,
        },
        "configuration": {
            "quality_profile": profile,
            "execution_profile": {"primary_backend": "cpu-float32"},
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    class FakeStore:
        def __init__(self) -> None:
            self.artifact_root = artifact_root

        def current(self) -> dict:
            return plan

    monkeypatch.setattr(
        "darsia.fluidflower.execution._plan_store",
        lambda _project, _module: FakeStore(),
    )

    reference = current_plan_reference(_project(tmp_path / "project"), "calibration")

    assert reference["quality_contract"]["backend"] == "cpu-float32"
    assert reference["quality_contract"]["compute_dtype"] == "float32"
    assert reference["quality_contract"]["scientific_grid"] == (
        "native registered grid"
    )


def test_queue_is_idempotent_and_claims_only_supported_tasks(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")

    assert queue.enqueue(task)
    assert not queue.enqueue(task)
    assert (
        queue.claim(
            "gpu-only",
            backends=["cuda-float32"],
            adapters=["plan-verification-v1"],
        )
        is None
    )

    lease = queue.claim(
        "cpu-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert lease is not None
    assert lease.task["task_id"] == task["task_id"]
    lease.complete(
        {
            "format": "darsia-fluidflower-task-result-v1",
            "task_sha256": task["content_sha256"],
        }
    )
    assert queue.counts() == {
        "pending": 0,
        "in_progress": 0,
        "paused": 0,
        "completed": 1,
        "failed": 0,
    }


def test_stale_lease_is_requeued_for_work_stealing(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "lost-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert lease is not None
    old = time.time() - 120
    os.utime(lease.path, (old, old))

    assert queue.requeue_stale(lease_timeout_seconds=30) == 1
    assert queue.counts()["pending"] == 1
    recovered = queue.claim(
        "replacement-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert recovered is not None
    assert recovered.record["attempts"] == 1


def test_verification_adapter_publishes_atomically(tmp_path: Path) -> None:
    project, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "cpu-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert lease is not None

    from darsia.fluidflower.execution import default_adapter_registry

    result = execute_task_lease(
        lease,
        registry=default_adapter_registry(),
        local_cache_root=tmp_path / "cache",
        inventory=_inventory(),
        heartbeat_seconds=0.01,
    )
    destination = Path(result["destination"])
    assert destination.is_dir()
    assert (destination / "verification.json").is_file()
    assert (destination / "plan.json").is_file()
    assert (destination / ".darsia-publish.json").is_file()
    assert artifact_manifest(destination)["file_count"] == 4
    assert queue.counts()["completed"] == 1
    assert not (tmp_path / "cache" / task["task_id"] / "attempt-001").exists()
    assert project.root == Path(task["project"]["root"])


def test_plan_file_change_fails_without_publishing(tmp_path: Path) -> None:
    _, task = _task(tmp_path, maximum_attempts=1)
    plan_path = Path(task["plan_reference"]["path"])
    plan_path.write_text("{}", encoding="utf-8")
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    lease = queue.claim(
        "cpu-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert lease is not None

    from darsia.fluidflower.execution import default_adapter_registry

    with pytest.raises(RuntimeError, match="changed"):
        execute_task_lease(
            lease,
            registry=default_adapter_registry(),
            local_cache_root=tmp_path / "cache",
            inventory=_inventory(),
            heartbeat_seconds=0.01,
        )
    lease.fail("plan changed")
    assert queue.counts()["failed"] == 1


def test_atomic_publish_is_idempotent_for_identical_task(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "result.txt").write_text("stable", encoding="utf-8")
    destination = tmp_path / "published"

    first = publish_directory_atomic(
        staging,
        destination,
        task_id="task-1",
        task_sha256="task-contract",
    )
    second = publish_directory_atomic(
        staging,
        destination,
        task_id="task-1",
        task_sha256="task-contract",
    )
    assert first["artifacts"]["content_sha256"] == second["artifacts"]["content_sha256"]


def test_worker_stop_does_not_cancel_active_task(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.ensure_layout()
    context = TaskExecutionContext(
        task=task,
        queue=queue,
        worker_id="worker-1",
        workspace=tmp_path / "workspace",
        inventory=_inventory(),
    )

    queue.request_worker_stop("worker-1", actor="Reviewer")
    assert not context.cancellation_requested()
    queue.request_task_cancellation(task["task_id"], actor="Reviewer")
    assert context.cancellation_requested()


def test_queue_pause_and_resume_preserve_attempt_count(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)

    queue.request_task_pause(task["task_id"], actor="Reviewer")
    assert queue.counts()["paused"] == 1
    assert queue.claim(
        "cpu-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    ) is None

    assert queue.resume_task(task["task_id"], actor="Reviewer")
    lease = queue.claim(
        "cpu-worker",
        backends=["cpu-float64"],
        adapters=["plan-verification-v1"],
    )
    assert lease is not None
    assert int(lease.record.get("attempts") or 0) == 0


def test_resumable_chunk_is_verified_and_restored(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.ensure_layout()
    context = TaskExecutionContext(
        task=task,
        queue=queue,
        worker_id="worker-1",
        workspace=tmp_path / "workspace",
        inventory=_inventory(),
    )
    chunk = context.prepare_chunk("run-1")
    (chunk / "runs" / "run-1").mkdir(parents=True)
    (chunk / "runs" / "run-1" / "mass.csv").write_text(
        "time,mass\n0,0\n", encoding="utf-8"
    )

    committed = context.commit_chunk(
        "run-1", chunk, state={"run_id": "run-1", "complete": True}
    )
    assert committed["state"]["complete"] is True
    restored = tmp_path / "restored"
    context.restore_chunk("run-1", restored)
    assert (restored / "runs" / "run-1" / "mass.csv").read_text(
        encoding="utf-8"
    ) == "time,mass\n0,0\n"

    durable_file = (
        Path(committed["root"]) / "runs" / "run-1" / "mass.csv"
    )
    durable_file.write_text("corrupt", encoding="utf-8")
    assert context.committed_chunk("run-1") == {}


def test_registry_rejects_unknown_backend() -> None:
    from darsia.fluidflower.execution import AdapterRegistry

    class InvalidAdapter:
        descriptor = AdapterDescriptor(
            key="invalid",
            version="1",
            title="Invalid",
            modules=("calibration",),
            backends=("silent-mixed-precision",),
        )

        def execute(self, task, context):  # noqa: ANN001
            return {}

    with pytest.raises(ValueError, match="unknown backends"):
        AdapterRegistry().register(InvalidAdapter())


def test_spawned_worker_runs_same_queue_contract(tmp_path: Path) -> None:
    _, task = _task(tmp_path)
    queue = ExecutionQueue(tmp_path / "queue")
    queue.enqueue(task)
    handles = spawn_worker_processes(
        queue_root=queue.root,
        cache_root=tmp_path / "cache",
        workers=1,
        backends=["cpu-float64"],
        idle_exit_seconds=0.25,
    )
    try:
        handles[0].process.wait(timeout=15)
    finally:
        if handles[0].process.poll() is None:
            handles[0].process.terminate()
            handles[0].process.wait(timeout=5)

    assert handles[0].process.returncode == 0
    assert queue.counts()["completed"] == 1
    assert (queue.directory("logs") / f"{handles[0].worker_id}.log").is_file()
