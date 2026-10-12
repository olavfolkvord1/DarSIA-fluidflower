"""Operational distributed application of a frozen native fit.

One task owns an entire selected run timeline. The existing native scientific
worker performs exact evaluation; only the coordinator publishes verified runs.
Use ``python -m darsia.fluidflower.tools.studio_distributed_mass_service --help``.
Shared fitting and accelerators are unavailable. Production additionally requires
an explicit passing model receipt and measured matching two-host pilot outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import threading
import time
import uuid

from . import studio_distributed_mass_queue as transport

ADAPTER = "studio-native-frozen-apply-v1"
MODULE = "darsia.fluidflower.tools.studio_distributed_mass_service"
CAPABILITIES = {"adapter": ADAPTER, "native_frozen_apply": True,
                "measured_overflow_compare": True,
                "shared_fit": False, "photometry_fit": False,
                "accelerators": False, "production": "requires_model_acceptance_and_measured_host_parity"}


def digest(value):
    return transport._sha_bytes(transport._json_bytes(value))


def verify_production_gate(request):
    from .studio_distributed_mass_parity import verify_production_gate as verify
    return verify(request)


def source_identity():
    """Portable source identity includes DarSIA and its native extension code."""
    src = Path(__file__).resolve().parents[3]
    files = {}
    for package in ("darsia", "darsia_extensions"):
        for path in sorted((src / package).rglob("*.py")):
            files[path.relative_to(src).as_posix()] = transport._sha_file(path)
    return files


def environment_identity():
    return {"backend": "native-cpu", "python": platform.python_version(),
            "system": platform.system(), "machine": platform.machine(),
            "libraries": {name: importlib.metadata.version(name) for name in
                          ("numpy", "scipy", "pillow", "matplotlib", "opencv-python")}}


def _file_inventory(paths):
    return {str(Path(path).resolve()): transport._sha_file(Path(path))
            for path in sorted(set(map(str, paths)))}


def _verify_inventory(files):
    if not files:
        raise transport.QueueError("Frozen input inventory is empty")
    for path, expected in files.items():
        if transport._sha_file(Path(path)) != expected:
            raise transport.QueueError("Frozen input changed: " + path)


def _probe_write(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (".mass-access-" + uuid.uuid4().hex)
    try:
        with target.open("xb") as stream:
            stream.write(b"access probe")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        target.unlink(missing_ok=True)


def build_request(native_request, queue_root, *, purpose="diagnostic"):
    """Read/verify the actual selected inputs, with no image or mass calculation.

    A caller must inspect this returned request before ``create_campaign``.
    Dense registered physical arrays are loaded for the existing calibration
    verifier; phase inference and parameter fitting are never called.
    """
    from darsia.fluidflower import concentration_inputs as inputs
    from darsia.fluidflower import concentration_configuration as configuration
    from darsia.fluidflower import concentration_native as native
    from . import phase_studio_native_mass as worker
    from .phase_studio_calibration_compare import select_frames

    if purpose not in {"diagnostic", "production"}:
        raise ValueError("Purpose must be diagnostic or production")
    if native_request.get("mode") != "frozen":
        raise ValueError("Distributed calculation supports Apply saved native fit only; fitting is unavailable")
    overflow_mode = native_request.get("overflow_mode", "off")
    if overflow_mode not in {"off", "compare"}:
        raise ValueError("Distributed frozen native apply supports overflow comparison, not overflow refitting")
    overflow = native_request.get("overflow") or {}
    if not isinstance(overflow, dict) or (overflow_mode == "off" and overflow):
        raise ValueError("Overflow targets require Compare with measured overflow")
    catalog = inputs.discover_model_inputs(native_request["model_root"], include_saved_results=False)
    jobs = worker.selected_jobs(native_request, catalog)
    results_root = Path(catalog["results_root"]).resolve()
    queue_root = Path(queue_root).resolve()
    if queue_root == results_root or not queue_root.is_relative_to(results_root):
        raise ValueError("Queue must be a new directory inside the selected series results root")
    if queue_root.exists():
        raise ValueError("Campaign already exists; attach/resume it instead")
    model_shas = {job["phase_binding"]["model_sha256"] for job in jobs}
    if len(model_shas) != 1:
        raise ValueError("Selected runs do not share one frozen phase-model content identity")
    fit_path = Path(native_request["calibration_file"]).resolve()
    fit = worker.read(fit_path)
    if fit.get("format") != native.VERSION or fit.get("dataset") != catalog["dataset"]:
        raise ValueError("Frozen calibration belongs to another method or series")
    profile = Path(native_request.get("concentration_profile_path") or
                   results_root.parent.parent / "concentration.toml").resolve()
    for job in jobs:
        job["concentration_profile_path"] = str(profile)
    inspected = {job["run"]: inputs.inspect_run(job) for job in jobs}
    if any(run not in inspected for run in overflow):
        raise ValueError("Overflow targets refer to an unselected experiment")
    overflow = {run: str(Path(path).resolve()) for run, path in overflow.items()}
    overflow_references = []
    if overflow_mode == "compare":
        # Use the same time-linked reader as local Studio, before freezing a
        # campaign. A run without measured overflow is explicitly recorded as
        # unreferenced below; its numerical frozen fit is unchanged.
        overflow_references, _ = worker.overflow_references(
            {"mode": "frozen", "overflow_mode": "compare", "overflow": overflow}, inspected)
    selection_request = dict(native_request)
    if "frame_indices" in selection_request:
        indices = selection_request.pop("frame_indices")
        selection_request["frame_indices_by_run"] = {job["run"]: indices for job in jobs}
    if "frame_indices_by_run" in selection_request:
        selection_request.setdefault("frame_exclusion_reasons", {
            job["run"]: "Explicit distributed frame selection" for job in jobs})
    exclusions = select_frames(selection_request, inspected)
    common_paths = [profile, fit_path]
    for key in ("physical_profile_receipt_path", "image_selection_receipt_path"):
        if native_request.get(key):
            common_paths.append(native_request[key])
    common_paths.extend(overflow.values())
    common_files = _file_inventory(common_paths)
    tasks, native_manifests, phase_manifests = [], {}, {}
    for job in jobs:
        run = job["run"]
        metadata = inspected[run]
        config = configuration.load_run_configuration(job)
        analysis = worker.create_analysis(config)
        provenance = analysis.provenance()
        frozen = fit.get("native_calibrations", {}).get(run, {})
        for key in ("calibration_files", "native_implementations"):
            if provenance[key] != frozen.get(key):
                raise ValueError(f"{run}: frozen native calibration differs in {key}")
        if worker.physical_binding(config) != fit.get("physical_bindings", {}).get(run):
            raise ValueError(f"{run}: physical inputs or baseline differ from the frozen calibration")
        indices = metadata["selected_frame_indices"]
        selected_references = [reference for reference in overflow_references
                               if reference["run"] == run and reference["frame_index"] in indices]
        if run in overflow and not selected_references:
            raise ValueError(f"{run}: measured overflow has no target at a selected image time")
        selected = [metadata["frames"][index] for index in indices]
        paths = [metadata["native_manifest"], metadata["baseline"], job["source_manifest"]]
        if metadata.get("context", {}).get("path"):
            paths.append(metadata["context"]["path"])
        paths.extend(provenance["calibration_files"])
        series_root = results_root.parent.parent
        for artifact in config.get("provenance", {}).get("artifacts", []):
            path = Path(artifact["path"])
            paths.append(path if path.is_absolute() else series_root / path)
        artifact_record = config.get("provenance", {}).get("artifact_record")
        if artifact_record:
            path = Path(artifact_record)
            paths.append(path if path.is_absolute() else series_root / path)
        for row in selected:
            paths.extend((row["photo"], row["phase"]))
        files = _file_inventory(paths)
        native_manifests[run] = metadata["native_manifest_sha256"]
        phase_manifests[run] = {row["phase"]: files[str(Path(row["phase"]).resolve())] for row in selected}
        payload = {"files": files, "candidate": job["candidate"],
                   "phase_binding": job["phase_binding"],
                   "physical_binding": worker.physical_binding(config),
                   "overflow_mode": "compare" if run in overflow else "off",
                   "overflow_source": (overflow.get(run)),
                   "overflow_references": selected_references,
                   "frame_indices": indices,
                   "frame_times_h": [float(row["elapsed_h"]) for row in selected]}
        tasks.append({"task_id": "render_" + run, "stage": "render", "run": run,
                      "input_sha256": digest(payload), "available_frame_count": len(metadata["frames"]), **payload})
        del config, analysis
    code = source_identity()
    model_sha = next(iter(model_shas))
    request = {"schema": 1, "campaign_id": str(uuid.uuid4()), "purpose": purpose,
               "max_attempts": 3, "adapter": ADAPTER, "dataset": catalog["dataset"],
               "shared_paths": {"queue_root": str(queue_root), "results_root": str(results_root)},
               "selection": {"runs": [j["run"] for j in jobs], "physical_profile": "registered",
                             "excluded_frames": exclusions, "overflow_mode": overflow_mode,
                             "overflow_runs": sorted(overflow)},
               "native_request": {"model_root": catalog["model_root"], "model_id": catalog["model_id"],
                                  "mode": "frozen", "overflow_mode": overflow_mode, "overflow": overflow,
                                  "calibration_file": str(fit_path),
                                  "concentration_profile_path": str(profile),
                                  "unknown_policy": native_request.get("unknown_policy", "complete_frames")},
               "common_files": common_files, "code_files": code,
               "environment": environment_identity(), "tasks": tasks,
               "bindings": {"model_id": catalog["model_id"], "model_sha256": model_sha,
                            "native_manifest_sha256": digest(native_manifests),
                            "phase_map_manifest_sha256": digest(phase_manifests),
                            "concentration_toml_sha256": common_files[str(profile)],
                            "calibration_sha256": common_files[str(fit_path)],
                            "overflow_sources_sha256": digest({run: common_files[path]
                                                                for run, path in sorted(overflow.items())}),
                            "method": "native_frozen", "gas_policy": "registered-native-fixed-phase",
                            "code_sha256": digest(code)}}
    for key in ("physical_profile",):
        if native_request.get(key):
            request["selection"][key] = native_request[key]
    for kind in ("physical_profile", "image_selection"):
        path = native_request.get(kind + "_receipt_path")
        if path:
            request["bindings"][kind + "_receipt_sha256"] = common_files[str(Path(path).resolve())]
    # A conservative restriction receipt is NOT a scientific approval or QA pass.
    request["eligibility_restriction"] = {
        "model_id": catalog["model_id"], "model_sha256": model_sha,
        "qa_status": "not_certified_for_distributed_mass", "deployment_allowed": False,
        "publication_scope": "studio_validation_only", "scientific_acceptance": False,
        "reason": "Diagnostic adapter only; model presence and frozen fit do not establish scientific acceptance"}
    if purpose == "production":
        eligibility_path = Path(native_request.get("model_eligibility_receipt_path", ""))
        parity_path = Path(native_request.get("backend_parity_receipt_path", ""))
        if not eligibility_path.is_file() or not parity_path.is_file():
            raise ValueError("Production requires passing model eligibility and measured backend-parity receipts")
        eligibility = transport._read_json(eligibility_path)
        request["model_eligibility"] = {key: eligibility.get(key) for key in
                                       ("qa_status", "deployment_allowed", "publication_scope")}
        request["model_eligibility"].update(receipt_path=str(eligibility_path.resolve()),
                                          receipt_sha256=transport._sha_file(eligibility_path))
        transport._verify_eligibility(request)
        if (eligibility.get("qa_status") != "pass" or eligibility.get("deployment_allowed") is not True
                or eligibility.get("publication_scope") == "studio_validation_only"):
            raise ValueError("QA-fail or validation-only models cannot enter production")
        request["production_gate"] = {"parity_receipt_path": str(parity_path.resolve()),
                                      "parity_receipt_sha256": transport._sha_file(parity_path)}
        request.pop("eligibility_restriction")
        if os.name == "nt" and any(not str(path).startswith("\\\\") for path in
            [*request["common_files"], str(queue_root), str(results_root),
             str(eligibility_path), str(parity_path),
             *(path for task in tasks for path in task["files"]) ]):
            raise ValueError("Production inputs and receipts must use shared UNC paths")
        verify_production_gate(request)
    return request


def create_campaign(request):
    """Create the reviewed immutable request; numerical work still needs Start."""
    request = json.loads(transport._json_bytes(request))
    if request.get("adapter") != ADAPTER or request.get("purpose") not in {"diagnostic", "production"}:
        raise ValueError("Only the frozen native adapter can create campaigns")
    root = Path(request["shared_paths"]["queue_root"])
    if root.exists():
        raise ValueError("Campaign already exists; attach/resume it instead")
    _verify_inventory(request["common_files"])
    for task in request["tasks"]:
        _verify_inventory(task["files"])
    if source_identity() != request["code_files"]:
        raise ValueError("Source changed after preflight; repeat preflight")
    _probe_write(root.parent)
    if request["purpose"] == "production":
        verify_production_gate(request)
    else:
        eligibility_path = root.parent / (request["campaign_id"] + ".eligibility.json")
        eligibility = request.pop("eligibility_restriction")
        with eligibility_path.open("xb") as stream:
            stream.write(transport._json_bytes(eligibility))
        request["model_eligibility"] = {key: eligibility[key] for key in
                                       ("qa_status", "deployment_allowed", "publication_scope")}
        request["model_eligibility"].update(receipt_path=str(eligibility_path),
                                          receipt_sha256=transport._sha_file(eligibility_path))
    campaign = transport.CampaignQueue.create(root, request)
    (root / "services").mkdir(exist_ok=True)
    return campaign


def _verify_model_binding(campaign):
    from darsia.fluidflower.concentration_inputs import discover_model_inputs
    request = campaign.request
    catalog = discover_model_inputs(request["native_request"]["model_root"], include_saved_results=False)
    by_run = {j["run"]: j for j in catalog["jobs"]}
    if catalog["model_id"] != request["bindings"]["model_id"]:
        raise transport.QueueError("Selected model identity changed")
    for task in request["tasks"]:
        current = by_run.get(task["run"], {})
        if (current.get("candidate") != task["candidate"] or
                current.get("phase_binding") != task["phase_binding"]):
            raise transport.QueueError("Selected phase input binding changed: " + task["run"])


def verify_host(campaign, scratch, *, full=True, host=None, numerical=True):
    campaign._assert_campaign()
    request = campaign.request
    if request.get("adapter") != ADAPTER:
        raise transport.QueueError("Unsupported adapter; worker cannot join")
    if host is not None and host != socket.gethostname():
        raise transport.QueueError("Numerical service host must be this computer's actual hostname")
    if request["purpose"] == "production":
        certificate = verify_production_gate(request)
        if numerical and certificate["hosts"].get(host or socket.gethostname()) != environment_identity():
            raise transport.QueueError("This host/environment has no matching measured parity pilot")
    elif environment_identity() != request["environment"]:
        raise transport.QueueError("Numerical environment differs from the frozen campaign")
    if source_identity() != request["code_files"]:
        raise transport.QueueError("Numerical source differs from the frozen campaign")
    _verify_inventory(request["common_files"])
    if full:
        for task in request["tasks"]:
            _verify_inventory(task["files"])
        _verify_model_binding(campaign)
    _probe_write(scratch)
    _probe_write(campaign.root / "services")
    _probe_write(request["shared_paths"]["results_root"])
    return {"host": socket.gethostname(), "environment": request["environment"],
            "campaign_sha256": campaign.request_sha256, "capabilities": CAPABILITIES,
            "cross_host_parity_certified": request["purpose"] == "production"}


def execute_task(campaign, task, output):
    """Exact native calculation in an attempt directory; no catalog writes."""
    from . import phase_studio_native_mass as native_worker
    tick = time.monotonic()
    _verify_inventory(campaign.request["common_files"])
    _verify_inventory(task["files"])
    payload = {key: task[key] for key in ("files", "candidate", "phase_binding", "physical_binding",
                                        "frame_indices", "frame_times_h", "overflow_mode",
                                        "overflow_source", "overflow_references") if key in task}
    if digest(payload) != task["input_sha256"]:
        raise transport.QueueError("Task input binding changed")
    request = dict(campaign.request["native_request"], runs=[task["run"]],
                   frame_indices=task["frame_indices"], output=str(output / "native"))
    if "overflow_mode" in task:
        source = task.get("overflow_source")
        request["overflow_mode"] = task["overflow_mode"]
        request["overflow"] = {task["run"]: source} if source else {}
    path = output / "native_request.json"
    transport._atomic_json(path, request)
    delivered = native_worker.run(path, publish=False)
    _verify_inventory(campaign.request["common_files"])
    _verify_inventory(task["files"])
    receipt = {"adapter": ADAPTER, "task_id": task["task_id"],
               "campaign_sha256": campaign.request_sha256, "input_sha256": task["input_sha256"],
               "frame_indices": task["frame_indices"], "frame_times_h": task["frame_times_h"],
               "run": task["run"], "experiments": delivered,
               "execution_host": socket.gethostname(),
               "environment": environment_identity(), "seconds": time.monotonic() - tick}
    transport._atomic_json(output / "scientific_receipt.json", receipt)
    return receipt


def verify_scientific_receipt(campaign, task, offer):
    """Re-read coverage and mass diagnostics before transport acceptance."""
    campaign._verify_files(offer)
    output = Path(offer["output_dir"])
    receipt = transport._read_json(output / "scientific_receipt.json")
    for key, expected in (("adapter", ADAPTER), ("campaign_sha256", campaign.request_sha256),
                          ("input_sha256", task["input_sha256"]), ("task_id", task["task_id"]),
                          ("run", task["run"]), ("frame_indices", task["frame_indices"]),
                          ("frame_times_h", task["frame_times_h"])):
        if receipt.get(key) != expected:
            raise transport.QueueError("Scientific receipt mismatch: " + key)
    expected_environment = (verify_production_gate(campaign.request)["hosts"].get(offer["host"])
                            if campaign.request["purpose"] == "production" else campaign.request["environment"])
    if receipt.get("environment") != expected_environment:
        raise transport.QueueError("Scientific environment lacks matching campaign/parity binding")
    if receipt.get("execution_host") != offer["host"]:
        raise transport.QueueError("Scientific execution host differs from its lease")
    records = receipt.get("experiments", [])
    if len(records) != 1 or records[0].get("run") != task["run"]:
        raise transport.QueueError("Scientific receipt must contain exactly its selected run")
    result_path = Path(records[0]["result"])
    if not result_path.resolve().is_relative_to(output.resolve()):
        raise transport.QueueError("Scientific result escapes its attempt")
    result = transport._read_json(result_path)
    if (not result.get("complete") or result.get("selected_inputs", {}).get("candidate") != task["candidate"]
            or result.get("phase_inference_repeated") is not False
            or result.get("phase_labels_modified") is not False
            or result.get("calibration", {}).get("fit_performed") is not False):
        raise transport.QueueError("Scientific result lost its fixed-input/no-refit binding")
    fit = transport._read_json(Path(campaign.request["native_request"]["calibration_file"]))
    result_fit = result["calibration"]
    if (result_fit.get("parameters") != fit.get("parameters") or
            result_fit.get("physical_bindings", {}).get(task["run"]) != fit.get("physical_bindings", {}).get(task["run"]) or
            result_fit.get("native_calibrations", {}).get(task["run"]) != fit.get("native_calibrations", {}).get(task["run"])):
        raise transport.QueueError("Scientific result differs from selected frozen calibration")
    frames = result.get("frames", [])
    if ([row["frame_index"] for row in frames] != task["frame_indices"] or
            [row["elapsed_h"] for row in frames] != task["frame_times_h"]):
        raise transport.QueueError("Scientific frame coverage/order differs from request")
    for row in frames:
        masses = [row[key] for key in ("aqueous_mass_g", "gas_mass_g", "total_mass_g")]
        if (any(not math.isfinite(value) or value < -1e-9 for value in masses)
                or not math.isclose(masses[0] + masses[1], masses[2], rel_tol=1e-7, abs_tol=1e-8)):
            raise transport.QueueError("Nonfinite, negative or inconsistent phase mass")
        for key in ("fields", "preview"):
            path = Path(row[key])
            if not path.is_file() or not path.resolve().is_relative_to(output.resolve()):
                raise transport.QueueError("Missing or external numerical/display output")
        source = row["source_reference"]
        for key in ("photo", "phase"):
            if task["files"].get(str(Path(source[key]).resolve())) != source[key + "_sha256"]:
                raise transport.QueueError("Result used a different photograph or fixed phase map")
    if "overflow_mode" in task:
        if result.get("overflow_mode") != task["overflow_mode"]:
            raise transport.QueueError("Result used another overflow comparison policy")
        expected_refs = {item["frame_index"]: item for item in task["overflow_references"]}
        for row in frames:
            if row.get("gas_reference") != expected_refs.get(row["frame_index"]):
                raise transport.QueueError("Measured overflow comparison differs from frozen target")
        source = task.get("overflow_source")
        if source and result.get("source_bindings", {}).get(source) != campaign.request["common_files"].get(source):
            raise transport.QueueError("Result lost measured overflow source binding")
    return receipt


def coordinator_tick(campaign, *, grace_seconds=30):
    """Caller holds the coordinator OS lock; accept/recover/finalize once."""
    campaign._assert_campaign()
    if not transport._exists(campaign.root / "started.json"):
        return False
    if transport._exists(campaign.root / "cancelled.json"):
        return False
    for task in campaign.request["tasks"]:
        task_id = task["task_id"]
        if transport._exists(campaign.root / "accepted" / (task_id + ".json")):
            continue
        generation = campaign._generation(task_id)["generation"]
        offer = campaign.root / "offers" / task_id / f"g{generation:06d}.json"
        if transport._exists(offer):
            try:
                verify_scientific_receipt(campaign, task, transport._read_json(offer))
            except (ValueError, KeyError, TypeError, transport.QueueError) as exc:
                # Storage errors remain recoverable; no success/acceptance is inferred.
                if isinstance(exc.__cause__, OSError):
                    raise
                campaign.reject_offer(task_id, str(exc))
            else:
                campaign.accept(task_id)
    campaign.recover(grace_seconds=grace_seconds)
    counts = campaign.status()["counts"]
    if counts["accepted"] != len(campaign.request["tasks"]):
        return False
    _verify_inventory(campaign.request["common_files"])
    if source_identity() != campaign.request["code_files"]:
        raise transport.QueueError("Source changed before finalization")
    _verify_model_binding(campaign)
    records, accepted = [], []
    for task in campaign.request["tasks"]:
        _verify_inventory(task["files"])
        offer = campaign.accept(task["task_id"])
        scientific = verify_scientific_receipt(campaign, task, offer)
        records.extend(scientific["experiments"])
        accepted.append({"task_id": task["task_id"], "receipt_sha256": digest(offer)})
    delivery = {"format": ADAPTER, "complete": True, "purpose": campaign.request["purpose"],
                "publication_scope": campaign.request["model_eligibility"]["publication_scope"],
                "campaign_sha256": campaign.request_sha256, "experiments": records,
                "accepted": accepted, "frame_count": sum(len(t["frame_indices"]) for t in campaign.request["tasks"])}
    path = campaign.root / "delivery.json"
    if transport._exists(path):
        if transport._read_json(path) != delivery:
            raise transport.QueueError("Completed campaign delivery changed")
    else:
        transport._atomic_json(path, delivery)
    from .phase_studio_mass_inputs import register_variant
    for record in records:
        register_variant(record["result"], campaign.request["shared_paths"]["results_root"])
    transport._atomic_json(campaign.root / "published.json", {"campaign_sha256": campaign.request_sha256,
                                                            "delivery_sha256": transport._sha_file(path)})
    return True


@contextmanager
def _process_lock(path):
    """OS-held SMB lock, automatically released on a process crash."""
    path = Path(path)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise transport.QueueError("A service already holds " + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _spawn(args, log):
    from darsia.fluidflower.phase_workflow import phase_python_executable
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with Path(log).open("ab") as stream:
        return subprocess.Popen([phase_python_executable(), "-m", MODULE, *args], stdout=stream,
                                stderr=subprocess.STDOUT, env=env, **options)


def run_slot(root, host, slot, scratch):
    campaign = transport.CampaignQueue.open(root)
    verify_host(campaign, scratch, host=host, full=False)
    if (not transport._exists(campaign.root / "started.json")
            or transport._exists(campaign.root / "cancelled.json")):
        return 0
    # One task per process bounds memory and reloads source/configuration on retry.
    return transport.run_worker(campaign, lambda task, output: execute_task(campaign, task, output),
                                host=host, slot=slot, worker_id=f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
                                stop_when_idle=True, max_tasks=1)


def start_campaign(campaign):
    path = campaign.root / "started.json"
    if transport._exists(campaign.root / "cancelled.json"):
        raise transport.QueueError("Cancelled campaigns cannot be restarted")
    payload = {"campaign_sha256": campaign.request_sha256}
    if not transport._exists(path):
        transport._atomic_json(path, payload)
    elif transport._read_json(path) != payload:
        raise transport.QueueError("Campaign start binding changed")


def supervise(root, host, scratch, *, coordinator=False, poll_seconds=2, launch_token=None):
    host = transport._safe_id(host, "host")
    scratch = Path(scratch).resolve()
    campaign = transport.CampaignQueue.open(root)
    running = {}
    role = "coordinator" if coordinator else "worker"
    def launch(slot):
        return _spawn(["slot", "--queue", str(root), "--host", host, "--slot", str(slot),
                       "--scratch", str(scratch)], scratch / f"slot-{slot}.log")
    with _process_lock(campaign.root / ("coordinator.lock" if coordinator else f"worker-{host}.lock")):
        with _process_lock(campaign.root / f"slots-{host}.lock"):
            import psutil
            identity = {"host": host, "pid": os.getpid(), "role": role,
                        "campaign_sha256": campaign.request_sha256,
                        "launch_token": launch_token,
                        "process_started_at": psutil.Process().create_time()}
            transport._atomic_json(campaign.root / "services" / (host + ".json"),
                                   {**identity, "seen_at": time.time(), "active_slots": [], "state": "verifying_inputs"})
            try:
                verify_host(campaign, scratch, host=host, numerical=campaign.worker_limit(host) > 0)
            except Exception as exc:
                state = {**identity, "seen_at": time.time(), "active_slots": [],
                         "state": "verification_failed", "error": str(exc)}
                transport._atomic_json(campaign.root / "services" / (host + ".json"), state)
                transport._atomic_json(scratch / "service-status.json", state)
                raise
            if coordinator:
                start_campaign(campaign)
            while True:
                try:
                    campaign._assert_campaign()
                    finished = coordinator_tick(campaign) if coordinator else transport._exists(campaign.root / "published.json")
                    cancelled = transport._exists(campaign.root / "cancelled.json")
                    failed = campaign.status()["counts"]["failed"] > 0
                    if finished or cancelled or failed:
                        campaign.set_worker_limit(host, 0)
                    if transport._exists(campaign.root / "started.json"):
                        transport.reconcile_host_slots(campaign, host, running, launch)
                    state = {**identity,
                             "seen_at": time.time(), "active_slots": sorted(running),
                             "desired_slots": campaign.worker_limit(host),
                             "state": "complete" if finished else "cancelled" if cancelled else "failed" if failed else "running"}
                    transport._atomic_json(campaign.root / "services" / (host + ".json"), state)
                    transport._atomic_json(scratch / "service-status.json", state)
                    if (finished or cancelled or failed) and not running:
                        return
                except (OSError, transport.QueueError) as exc:
                    # Keep reconnecting; the lease generation fences outputs from lost heartbeats.
                    transport._atomic_json(scratch / "service-status.json", {
                        "host": host, "pid": os.getpid(), "state": "storage_or_verification_error",
                        "error": str(exc), "seen_at": time.time(), "active_slots": sorted(running)})
                time.sleep(poll_seconds)


class LocalWorkerService:
    """GUI bridge: actual detached host supervisors with dynamic draining slots."""
    def __init__(self, scratch=None):
        self.scratch = Path(scratch or Path.home() / ".cache" / "darsia" / "distributed-mass")
        self.processes = {}
        self._mutex = threading.Lock()

    @staticmethod
    def _live_service(campaign, host):
        """Reconnect a restarted GUI to this host's real, still-running process."""
        import psutil
        path = campaign.root / "services" / (host + ".json")
        if not transport._exists(path):
            return None
        value = transport._read_json(path)
        if value.get("campaign_sha256") != campaign.request_sha256 or value.get("host") != host:
            raise transport.QueueError("Host service lost its campaign binding")
        try:
            process = psutil.Process(value["pid"])
            if (process.is_running() and math.isclose(process.create_time(), value["process_started_at"], rel_tol=0, abs_tol=.01)
                    and value.get("state") not in {"complete", "cancelled", "failed", "verification_failed"}):
                return value
        except (psutil.NoSuchProcess, KeyError):
            pass
        return None

    def launch(self, campaign, host, count, *, coordinator=False, scratch=None):
        with self._mutex:
            folder = Path(scratch or self.scratch) / campaign.request["campaign_id"] / host
            live = self._live_service(campaign, host)
            role = "coordinator" if coordinator else "worker"
            if live is not None and live["role"] != role:
                raise transport.QueueError("This host already runs a " + live["role"] +
                                           " service; keep its role when reconnecting")
            verify_host(campaign, folder, host=host, numerical=count > 0, full=live is None)
            campaign.set_worker_limit(host, count)
            if live is not None:
                return live["pid"]
            key = (str(campaign.root), host)
            child = self.processes.get(key)
            if child is None or child.poll() is not None:
                launch_token = uuid.uuid4().hex
                child = _spawn([role, "--queue", str(campaign.root), "--host", host,
                                "--scratch", str(folder), "--launch-token", launch_token], folder / "supervisor.log")
                self.processes[key] = child
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    live = self._live_service(campaign, host)
                    # uv's Windows interpreter trampoline can give Popen and
                    # the real interpreter different PIDs. Bind the handshake
                    # to a unique launch token plus the verified live PID.
                    if live is not None and live.get("launch_token") == launch_token:
                        return live["pid"]
                    if child.poll() is not None:
                        raise transport.QueueError("Host service exited during launch; inspect " + str(folder / "supervisor.log"))
                    time.sleep(.1)
                raise transport.QueueError("Host service startup is unconfirmed; inspect " + str(folder / "supervisor.log"))
            return child.pid

    def apply_limit(self, campaign, host, count):
        key = (str(campaign.root), host)
        child = self.processes.get(key)
        if count and (child is None or child.poll() is not None):
            return self.launch(campaign, host, count)
        campaign._assert_campaign()
        campaign.set_worker_limit(host, count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "create", "coordinator", "worker", "slot",
                                             "status", "stop-local", "set-limit", "cancel", "certify-parity"))
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--request", type=Path, help="Native request for preflight/create (mode frozen)")
    parser.add_argument("--scratch", type=Path, default=Path.home() / ".cache/darsia/distributed-mass")
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--reason", default="")
    parser.add_argument("--purpose", choices=("diagnostic", "production"), default="diagnostic")
    parser.add_argument("--peer-queue", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch-token", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "certify-parity":
        if args.peer_queue is None or args.output is None:
            parser.error("--peer-queue and --output are required")
        from .studio_distributed_mass_parity import create_receipt
        print(json.dumps(create_receipt(args.queue, args.peer_queue, args.output), indent=2))
        return
    if args.command in {"preflight", "create"}:
        if args.request is None:
            parser.error("--request is required")
        request = build_request(transport._read_json(args.request), args.queue, purpose=args.purpose)
        if args.command == "preflight":
            print(json.dumps(request, indent=2))
        else:
            campaign = create_campaign(request)
            print(json.dumps(campaign.status()))
        return
    campaign = transport.CampaignQueue.open(args.queue)
    if args.command == "status":
        print(json.dumps(campaign.status(), indent=2))
    elif args.command in {"stop-local", "set-limit"}:
        if args.command == "set-limit" and args.workers is None:
            parser.error("--workers is required")
        campaign.set_worker_limit(args.host, 0 if args.command == "stop-local" else args.workers)
    elif args.command == "cancel":
        if not args.reason.strip():
            parser.error("--reason is required")
        if transport._exists(campaign.root / "delivery.json"):
            raise transport.QueueError("Cannot cancel a completed campaign")
        # Cancellation is an explicit coordinator command, never worker shutdown.
        transport._atomic_json(campaign.root / "cancelled.json", {
            "campaign_sha256": campaign.request_sha256, "reason": args.reason,
            "host": args.host, "time": time.time()})
    elif args.command == "slot":
        run_slot(args.queue, args.host, args.slot, args.scratch)
    else:
        if args.workers is not None:
            campaign.set_worker_limit(args.host, args.workers)
        elif not transport._exists(campaign.root / "control" / (args.host + ".json")):
            campaign.set_worker_limit(args.host, 0)
        supervise(args.queue, args.host, args.scratch, coordinator=args.command == "coordinator",
                  launch_token=args.launch_token)


if __name__ == "__main__":
    main()
