"""Local/NAS and kill/resume acceptance with small native DarSIA products."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import errno
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from darsia.fluidflower.production_io import NasIO
import darsia.fluidflower.production_io as production_io
from darsia.fluidflower.production_job import (
    LocalProductionJob, code_identity, create_manifest, environment_lock,
    plan_batches, require_workers_retired, worker_identity,
    validate_manifest,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fixture_repo(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "native.txt").write_text("synthetic native acceptance fixture\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "native.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Local acceptance test",
                    "-c", "user.email=local-test@example.invalid", "commit", "-q",
                    "-m", "Synthetic test fixture"], check=True)
    return root


def make_job(tmp_path, fixture_repo, *, purpose="file_handling_acceptance", precision="float64", name="job", source_review=None):
    root = tmp_path / name
    root.mkdir()
    io = NasIO()
    binding = root / "fixture_input.txt"
    binding.write_text("Immutable synthetic input, not experimental data.\n")
    asset = io.asset(binding)
    manifest = create_manifest(root / "manifest.json", worktree=fixture_repo,
        settings={"host": "test", "precision": precision, "backend": "cpu",
                  "threads": 1, "workers": 1, "logical_cpus": 1},
        calibration=asset, specs={"synthetic_fixture": asset},
        inputs={"fixture": asset, **({"source_review": source_review} if source_review is not None else {})},
        scientific_input_roles=["fixture"],
        assurance_input_roles=["source_review"] if source_review is not None else [],
        purpose=purpose, frames=["fixture/frame_0000", "fixture/frame_0001", "fixture/frame_0002"],
        environment=environment_lock(freeze_text="synthetic-test==1\n"), io=io)
    return LocalProductionJob(manifest, root / "local", root / "nas", io=io)


def test_fix_restart_keeps_manifest_and_binds_both_source_reviews(tmp_path, fixture_repo):
    io = NasIO()
    initial = code_identity(fixture_repo)["commit"]
    original_document = {"contract": "darsia-production-io-source-review-v1",
                         "complete": True, "io_only_diff_passed": True,
                         "execution_commit": initial}
    original_path = tmp_path / "original_source_review.json"
    io.write_json(original_path, original_document)
    original_asset = io.asset(original_path)
    job = make_job(tmp_path, fixture_repo, source_review=original_asset)
    manifest_before = io.asset(job.manifest_asset["path"])
    (fixture_repo / "native.txt").write_text("synthetic IO metadata correction\n")
    subprocess.run(["git", "-C", str(fixture_repo), "add", "native.txt"], check=True)
    subprocess.run(["git", "-C", str(fixture_repo), "-c", "user.name=Local acceptance test",
                    "-c", "user.email=local-test@example.invalid", "commit", "-q",
                    "-m", "Synthetic IO fix"], check=True)
    current = code_identity(fixture_repo)["commit"]
    fix_path = tmp_path / "fix_source_review.json"
    io.write_json(fix_path, dict(original_document, execution_commit=current))
    fix_asset = io.asset(fix_path)
    options = dict(workers=[], stop={"point": "metadata before arrays"},
                   current_worktree=fixture_repo, fix_commit=current)
    with pytest.raises(ValueError, match="both"):
        job.restart(**options, original_source_review=original_asset)
    with pytest.raises(ValueError, match="actual commit"):
        job.restart(**options, original_source_review=original_asset,
                    fix_source_review=original_asset)
    unrelated_path = tmp_path / "unbound_original_review.json"
    io.write_json(unrelated_path, original_document)
    with pytest.raises(ValueError, match="manifest assurance"):
        job.restart(**options, original_source_review=io.asset(unrelated_path),
                    fix_source_review=fix_asset)
    record = job.restart(**options, original_source_review=original_asset,
                         fix_source_review=fix_asset)
    assert record["original_source_review"] == original_asset
    assert record["fix_source_review"] == fix_asset
    assert record["initial_commit"] == initial
    assert record["fix_commit"] == current
    assert record["recompute_frames"] == job.manifest["frames"]
    assert record["carried_frames"] == []
    assert io.asset(job.manifest_asset["path"]) == manifest_before


def write_native_products(temporary, frame_index):
    import darsia
    signal = darsia.ScalarImage(np.linspace(0, 1, 48).reshape(6, 8) + frame_index * 0.001)
    aqueous, saturation = darsia.SimpleFlash(0.0, 0.75, 0.75, 1.0)(signal)
    mass = np.asarray(aqueous.img, dtype=np.float64) * 0.02 + np.asarray(saturation.img, dtype=np.float64) * 0.1
    for name, values in [("mass.npy", mass), ("aqueous.npy", aqueous.img), ("saturation.npy", saturation.img)]:
        with (temporary / name).open("xb") as f:
            np.save(f, values, allow_pickle=False)
    NasIO().write_json(temporary / "statistics.json", {
        "mass_g": float(np.sum(mass, dtype=np.float64)),
        "aqueous_sum": float(np.sum(aqueous.img, dtype=np.float64)),
        "gas_sum": float(np.sum(saturation.img, dtype=np.float64)),
    })


def finish_frame(job, index):
    frame_id = f"fixture/frame_{index:04d}"
    temporary = job.begin_frame(frame_id)
    write_native_products(temporary, index)
    return job.commit_frame(frame_id, temporary, product_metadata={"precision": "float64"})


def interrupted_native_worker(manifest, local, nas, ready):
    job = LocalProductionJob(manifest, local, nas)
    temporary = job.begin_frame("fixture/frame_0002")
    write_native_products(temporary, 2)
    NasIO().write_json(ready, {"worker": worker_identity(), "temporary": str(temporary)})
    # This is a test-owned stop point, after computation and before publication.
    time.sleep(30)
    job.commit_frame("fixture/frame_0002", temporary, product_metadata={"precision": "float64"})


def product_hashes(job):
    result = {}
    for frame_id in job.manifest["frames"]:
        receipt = job.verify_frame(frame_id)
        result[frame_id] = {name: value["sha256"] for name, value in receipt["files"].items()}
    return result


def test_manifest_written_once_and_clean_code_required(tmp_path, fixture_repo):
    job = make_job(tmp_path, fixture_repo)
    before = digest(job.manifest_asset["path"])
    with pytest.raises(FileExistsError):
        NasIO().write_json(job.manifest_asset["path"], job.manifest)
    assert digest(job.manifest_asset["path"]) == before
    (fixture_repo / "native.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="clean"):
        code_identity(fixture_repo)


def test_local_staging_exact_input_and_production_go(tmp_path, fixture_repo):
    job = make_job(tmp_path, fixture_repo, purpose="production")
    with pytest.raises(PermissionError, match="GO"):
        job.assert_runtime(fixture_repo)
    staged = job.stage_inputs()["fixture"]
    assert staged["sha256"] == job.manifest["inputs"]["fixture"]["sha256"]
    assert Path(staged["path"]).is_relative_to(job.local_root)
    assert job.controls_root.is_relative_to(job.local_root)


def test_environment_and_input_role_bindings_are_complete(tmp_path, fixture_repo):
    job = make_job(tmp_path, fixture_repo)
    document = dict(job.manifest)
    assert document["environment"]["python_sha256"] == digest(document["environment"]["python_executable"])
    assert document["scientific_input_roles"] == ["fixture"]
    document["assurance_input_roles"] = ["fixture"]
    with pytest.raises(ValueError, match="partition"):
        validate_manifest(document)
    document["assurance_input_roles"] = []
    document["environment"] = dict(document["environment"], python_sha256="not a SHA")
    with pytest.raises(ValueError, match="environment"):
        validate_manifest(document)


def test_precision_and_committed_inventory_are_not_relaxed(tmp_path, fixture_repo):
    job = make_job(tmp_path, fixture_repo)
    temporary = job.begin_frame("fixture/frame_0000")
    write_native_products(temporary, 0)
    with pytest.raises(ValueError, match="precision"):
        job.commit_frame("fixture/frame_0000", temporary, product_metadata={"precision": "float32"})
    job.commit_frame("fixture/frame_0000", temporary, product_metadata={"precision": "float64"})
    with pytest.raises(FileExistsError):
        job.begin_frame("fixture/frame_0000")
    (job.frames_root / "fixture/frame_0000/unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="inventory"):
        job.verify_frame("fixture/frame_0000")


def test_pid_epoch_blocks_live_worker_but_not_unrelated_pid_reuse():
    actual = worker_identity()
    with pytest.raises(RuntimeError, match="not exited"):
        require_workers_retired([actual])
    unrelated = {**actual, "process_create_time": str(float(actual["process_create_time"]) - 1000)}
    require_workers_retired([unrelated])
    with pytest.raises(ValueError, match="start-time"):
        require_workers_retired([{"pid": actual["pid"], "process_create_time": 1.0}])


def test_native_kill_resume_preserves_products_manifest_and_partial(tmp_path, fixture_repo):
    baseline = make_job(tmp_path, fixture_repo, name="uninterrupted")
    resumed = make_job(tmp_path, fixture_repo, name="resumed")
    for index in range(3):
        finish_frame(baseline, index)
    for index in range(2):
        finish_frame(resumed, index)
    carried_before = product_hashes_subset(resumed, 2)
    manifest_before = digest(resumed.manifest_asset["path"])
    ready = resumed.controls_root / "test_worker_ready.json"
    child = multiprocessing.get_context("spawn").Process(target=interrupted_native_worker,
        args=(resumed.manifest_asset, resumed.local_root, resumed.nas_root, ready))
    child.start()
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and child.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Native test worker did not reach its owned stop point"
        checkpoint = NasIO().read_json(ready)
        assert checkpoint["worker"]["pid"] == child.pid
        resumed.register_worker(checkpoint["worker"], frame_id="fixture/frame_0002")
        with pytest.raises(RuntimeError, match="not exited"):
            resumed.restart(workers=[], stop={"reason": "omitted live worker"}, current_worktree=fixture_repo)
        child.terminate()  # Only the just-created, identified synthetic test worker.
        child.join(timeout=10)
        assert not child.is_alive()
        partial = Path(checkpoint["temporary"])
        partial_stat = partial.stat()
        record = resumed.restart(workers=[checkpoint["worker"]],
            stop={"original_stop_utc": "2026-10-12T00:00:00Z", "reason": "test-owned kill"},
            current_worktree=fixture_repo)
        assert record["carried_frames"] == resumed.manifest["frames"][:2]
        assert record["recompute_frames"] == resumed.manifest["frames"][2:]
        assert partial.stat().st_mtime_ns == partial_stat.st_mtime_ns
        assert partial.exists()  # Its product contents are never opened by recovery.
        finish_frame(resumed, 2)
        assert product_hashes(resumed) == product_hashes(baseline)
        assert product_hashes_subset(resumed, 2) == carried_before
        assert digest(resumed.manifest_asset["path"]) == manifest_before
        assert partial.exists()
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)


def product_hashes_subset(job, count):
    return {frame_id: job.verify_frame(frame_id)["files"] for frame_id in job.manifest["frames"][:count]}


def test_local_to_nas_products_have_exact_hashes_and_local_is_retained(tmp_path, fixture_repo):
    job = make_job(tmp_path, fixture_repo)
    finish_frame(job, 0)
    receipt = job.copy_frame_to_nas("fixture/frame_0000")
    for relative, expected in receipt["files"].items():
        assert digest(Path(receipt["nas_directory"]) / relative) == expected["sha256"]
    assert job.verify_frame("fixture/frame_0000")["complete"]
    assert receipt["local_retained"]


def test_copy_to_real_nas_when_requested(tmp_path, fixture_repo):
    destination = os.environ.get("FFAC_ACCEPTANCE_NAS_DIRECTORY")
    if not destination:
        pytest.skip("Real NAS acceptance destination not requested")
    job = make_job(tmp_path, fixture_repo)
    job.nas_root = Path(destination) / job.manifest["job_id"]
    finish_frame(job, 0)
    copied = job.copy_frame_to_nas("fixture/frame_0000")
    for relative, expected in copied["files"].items():
        assert digest(Path(copied["nas_directory"]) / relative) == expected["sha256"]
    assert job.verify_frame("fixture/frame_0000")["complete"]


def test_disk_space_batches_keep_twenty_percent_margin():
    assert plan_batches(available_bytes=700, input_bytes=100, output_bytes_per_frame=100, frame_count=7) == [5, 2]
    with pytest.raises(OSError, match="20 percent"):
        plan_batches(available_bytes=219, input_bytes=100, output_bytes_per_frame=100, frame_count=1)


@pytest.mark.parametrize("duration", [1, 2, 4, 8, 10])
@pytest.mark.parametrize("operation", ["read", "rename", "replace"])
def test_native_products_are_identical_with_transient_NAS_faults(tmp_path, fixture_repo, monkeypatch, duration, operation):
    baseline = make_job(tmp_path, fixture_repo, name="no_fault")
    faulted = make_job(tmp_path, fixture_repo, name="with_fault")
    finish_frame(baseline, 0)
    finish_frame(faulted, 0)
    clock = [0.0]
    retries = []
    policy = production_io.RetryPolicy(clock=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), on_retry=retries.append)
    faulted.io = NasIO(policy)
    # This tests the same NAS branch with a deterministic 1--10 s fault window.
    # The real NAS copy is tested separately, without monkeypatching its identity.
    monkeypatch.setattr(production_io, "is_network_path", lambda p: Path(p).is_relative_to(faulted.nas_root))
    original_open, original_move, original_replace = Path.open, production_io._move_no_replace, os.replace
    enabled = [False]
    faulted.nas_root.mkdir(parents=True)
    control = faulted.nas_root / "control.json"
    faulted.io.write_json(control, {"sequence": 1})

    def fail_if_active(path):
        if enabled[0] and Path(path).is_relative_to(faulted.nas_root) and clock[0] < duration:
            raise PermissionError(errno.EACCES, "Injected transient NAS access denial")

    def open_with_fault(path, mode="r", *args, **kwargs):
        if operation == "read" and "r" in mode:
            fail_if_active(path)
        return original_open(path, mode, *args, **kwargs)

    def move_with_fault(source, destination):
        if operation == "rename":
            fail_if_active(destination)
        return original_move(source, destination)

    def replace_with_fault(source, destination):
        if operation == "replace":
            fail_if_active(destination)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "open", open_with_fault)
    monkeypatch.setattr(production_io, "_move_no_replace", move_with_fault)
    monkeypatch.setattr(os, "replace", replace_with_fault)
    if operation == "rename":
        enabled[0] = True
    copied = faulted.copy_frame_to_nas("fixture/frame_0000")
    if operation == "read":
        enabled[0] = True
        faulted.io.read_json(Path(copied["nas_directory"]) / "commit.json")
    elif operation == "replace":
        enabled[0] = True
        faulted.io.replace_json(control, lambda: {"sequence": 2})
    assert retries and clock[0] >= duration and clock[0] <= 60
    assert baseline.verify_frame("fixture/frame_0000")["files"] == faulted.verify_frame("fixture/frame_0000")["files"]
    for relative, expected in copied["files"].items():
        assert digest(Path(copied["nas_directory"]) / relative) == expected["sha256"]
