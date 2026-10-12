"""Measured CPU parity certificates from two completed distributed pilots.

No caller-provided pass flag or tolerance can certify a host. Certification
re-reads both accepted native outputs and compares every scalar and dense field.
Production verification recomputes that comparison from the hashed evidence.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import studio_distributed_mass_queue as queue

FORMAT = "studio-native-measured-host-parity-v1"
RTOL = 1e-6
ATOL = 1e-8


def compare_campaigns(reference_root, peer_root):
    from . import studio_distributed_mass_service as service
    reference = queue.CampaignQueue.open(reference_root)
    peer = queue.CampaignQueue.open(peer_root)
    if reference.root.resolve() == peer.root.resolve():
        raise ValueError("Parity requires two independent completed pilot campaigns")
    for campaign in (reference, peer):
        if (campaign.request.get("adapter") != service.ADAPTER
                or campaign.request.get("purpose") != "diagnostic"
                or not queue._exists(campaign.root / "published.json")
                or campaign.status()["counts"]["accepted"] != len(campaign.request["tasks"])):
            raise ValueError("Parity requires complete, published diagnostic frozen-native pilots")
        service._verify_inventory(campaign.request["common_files"])
        for task in campaign.request["tasks"]:
            service._verify_inventory(task["files"])
    for key in ("bindings", "native_request", "common_files", "code_files", "tasks"):
        if reference.request[key] != peer.request[key]:
            raise ValueError("Pilot selections/sources differ in " + key)
    host_sets = [set(), set()]
    measurements, evidence, run_specs = [], [], {}
    for task in reference.request["tasks"]:
        results = []
        for index, campaign in enumerate((reference, peer)):
            offer = campaign.accept(task["task_id"])
            scientific = service.verify_scientific_receipt(campaign, task, offer)
            host_sets[index].add(offer["host"])
            results.append(queue._read_json(Path(scientific["experiments"][0]["result"])))
            evidence.append({"campaign_sha256": campaign.request_sha256, "task_id": task["task_id"],
                             "accepted_sha256": service.digest(offer)})
        for left, right in zip(results[0]["frames"], results[1]["frames"]):
            scalars = {key for key, value in left.items() if type(value) in (int, float)}
            if scalars != {key for key, value in right.items() if type(value) in (int, float)}:
                raise ValueError("Pilot numeric scalar inventory differs")
            scalar_errors = {}
            for key in sorted(scalars):
                a, b = float(left[key]), float(right[key])
                if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
                    raise ValueError(f"Pilot scalar parity failed: {task['run']} frame {left['frame_index']} {key}")
                scalar_errors[key] = abs(a-b)
            field_errors = {}
            with np.load(left["fields"], allow_pickle=False) as first, np.load(right["fields"], allow_pickle=False) as second:
                if set(first.files) != set(second.files) or not first.files:
                    raise ValueError("Pilot numerical field inventory differs or is empty")
                for key in sorted(first.files):
                    a, b = first[key], second[key]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        raise ValueError("Pilot field grid/dtype differs: " + key)
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError("Pilot fields contain nonfinite values: " + key)
                    exact = a.dtype.kind in "biu"
                    if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=RTOL, atol=ATOL)):
                        raise ValueError(f"Pilot field parity failed: {task['run']} frame {left['frame_index']} {key}")
                    field_errors[key] = float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0))
            measurements.append({"run": task["run"], "frame_index": left["frame_index"],
                                 "elapsed_h": left["elapsed_h"], "scalar_max_abs_errors": scalar_errors,
                                 "field_max_abs_errors": field_errors})
        run_specs[task["run"]] = {key: task[key] for key in
            ("candidate", "phase_binding", "physical_binding", "frame_indices", "frame_times_h")}
        run_specs[task["run"]].update({key: task.get(key) for key in
            ("overflow_mode", "overflow_source", "overflow_references")})
    if any(len(hosts) != 1 for hosts in host_sets) or host_sets[0] == host_sets[1]:
        raise ValueError("Each pilot must run on one distinct host to certify that host pair")
    return {"format": FORMAT, "reference_root": str(reference.root), "peer_root": str(peer.root),
            "reference_sha256": reference.request_sha256, "peer_sha256": peer.request_sha256,
            "bindings": reference.request["bindings"], "run_specs": run_specs,
            "hosts": {next(iter(host_sets[0])): reference.request["environment"],
                      next(iter(host_sets[1])): peer.request["environment"]},
            "rtol": RTOL, "atol": ATOL, "measurements": measurements, "evidence": evidence}


def create_receipt(reference_root, peer_root, output):
    value = compare_campaigns(reference_root, peer_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Parity receipt exists; use a new path")
    # Exclusive publication; a failed generation never leaves a claimed pass.
    with output.open("xb") as stream:
        stream.write(queue._json_bytes(value))
    return value


def verify_production_gate(request):
    queue._verify_eligibility(request)
    queue._validate_request(request)
    gate = request.get("production_gate") or {}
    path = Path(gate.get("parity_receipt_path", ""))
    if not path.is_file() or queue._sha_file(path) != gate.get("parity_receipt_sha256"):
        raise queue.QueueError("Production parity receipt missing or changed")
    receipt = queue._read_json(path)
    if receipt.get("format") != FORMAT:
        raise queue.QueueError("Unsupported production parity receipt")
    actual = compare_campaigns(receipt["reference_root"], receipt["peer_root"])
    if actual != receipt:
        raise queue.QueueError("Production parity evidence changed")
    for key in ("model_id", "model_sha256", "concentration_toml_sha256", "calibration_sha256",
                "method", "gas_policy", "code_sha256"):
        if actual["bindings"][key] != request["bindings"][key]:
            raise queue.QueueError("Parity certificate belongs to another " + key)
    if actual["bindings"].get("overflow_sources_sha256") != request["bindings"].get("overflow_sources_sha256"):
        raise queue.QueueError("Parity certificate belongs to another measured overflow selection")
    for task in request["tasks"]:
        spec = actual["run_specs"].get(task["run"])
        if spec is None:
            raise queue.QueueError("No measured pilot for production run " + task["run"])
        for key in ("candidate", "phase_binding", "physical_binding"):
            if spec[key] != task[key]:
                raise queue.QueueError("Parity pilot physical/model binding differs: " + key)
        for key in ("overflow_mode", "overflow_source"):
            if spec.get(key) != task.get(key):
                raise queue.QueueError("Parity pilot measured overflow policy differs: " + key)
        production_refs = {row["frame_index"]: row for row in task.get("overflow_references") or []}
        if any(production_refs.get(row["frame_index"]) != row
               for row in spec.get("overflow_references") or []):
            raise queue.QueueError("Parity pilot measured overflow targets differ")
        timeline = dict(zip(task["frame_indices"], task["frame_times_h"]))
        if not spec["frame_indices"] or any(timeline.get(i) != t for i, t in
                                          zip(spec["frame_indices"], spec["frame_times_h"])):
            raise queue.QueueError("Measured pilot frames must belong to the production selection")
    queue._verify_eligibility(request)
    return actual
