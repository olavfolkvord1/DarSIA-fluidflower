"""Fail-closed physical recovery routing around an unchanged scientific plan."""
from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path

from darsia.fluidflower.tools.refit_ffac_g3_downstream import canonical_sha256
from darsia.fluidflower.tools import run_ffac_selected_policy_a_baseline_inventory as original

CONTRACT = "ffac-selected-policy-a-execution-operation-v1"


def same_asset(left, right):
    return Path(left["path"]) == Path(right["path"]) and left["sha256"] == right["sha256"]


def require_operation(plan, root, reader, *, verify_modules=True):
    operation = plan.get("execution_operation", {})
    if (operation.get("contract") != CONTRACT
            or operation.get("scientific_pipeline_unchanged") is not True
            or canonical_sha256(operation) != plan.get("execution_sha256")
            or Path(operation.get("phase_output_root", "")) != Path(root)
            or Path(operation.get("scientific_source_root", "")) != Path(plan["scientific_source_root"])
            or Path(operation["shared_stop_path"]).parent.parent != Path(operation["scientific_source_root"])
            or operation.get("backend") != "cpu" or operation.get("main_only") is not True
            or operation.get("internal_threads") != 1):
        raise ValueError("Execution operation changed physical routing or CPU admission")
    parent = reader.record(operation["original_phase_plan"])
    if (parent["phase_campaign_binding"] != plan["phase_campaign_binding"]
            or parent["pipeline_sha256"] != plan["pipeline_sha256"]
            or parent["job_sha256"] != plan["job_sha256"]
            or parent["runs"] != plan["runs"]
            or parent["identity_gate"] != plan["identity_gate"]
            or Path(parent["scientific_source_root"]) == Path(plan["scientific_source_root"])
            or Path(operation["shared_stop_path"]) == original.shared_stop(root, parent)):
        raise ValueError("Recovery must preserve science and use distinct physical roots/stop")
    if verify_modules:
        for name, descriptor in operation["code"].items():
            if original.asset(importlib.import_module(name).__file__)["sha256"] != descriptor["sha256"]:
                raise ValueError("Execution-only module changed: " + name)
    return operation, parent


def source_import_rows(operation, reader):
    proof = reader.record(operation["source_import_proof"])
    if (proof.get("contract") != "ffac-selected-source-physical-import-index-v1"
            or proof.get("complete") is not True
            or proof.get("byte_exact_receipt_copies") is not True
            or proof.get("arrays_copied_or_recomputed") is not False
            or proof.get("original_phase_plan") != operation["original_phase_plan"]
            or proof.get("expanded_proof_index_verified") is not True):
        raise ValueError("Physical source imports lack unchanged producer evidence")
    rows = {}
    for row in proof["imports"]:
        key = (row["run"], row["source_sha256"])
        if key in rows:
            raise ValueError("Physical source import duplicated")
        rows[key] = row
    return rows


def verify_ready_source(descriptor, source, plan, reader):
    operation, parent = require_operation(plan, operation_root(plan), reader, verify_modules=False)
    imports = getattr(reader, "physical_source_import_rows", None)
    if imports is None:
        imports = reader.physical_source_import_rows = source_import_rows(operation, reader)
    row = imports.get((source["run"], source["source_sha256"]))
    expected_path = Path(operation["scientific_source_root"])/"selected_sources"/source["run"]/source["source_sha256"]/"receipt.json"
    if Path(descriptor["path"]) != expected_path:
        raise ValueError("Scientific source is outside the admitted physical source root")
    if row is None:
        return original.verify_ready_source(descriptor, source, parent, reader)
    if (not same_asset(row["staged_receipt"], descriptor)
            or descriptor["sha256"] != row["original_receipt"]["sha256"]):
        raise ValueError("Physical source receipt differs from preserved producer bytes")
    saved, old = reader.record(descriptor), reader.record(row["original_receipt"])
    if saved != old or canonical_sha256(saved.get("binding")) != row["producer_binding_sha256"]:
        raise ValueError("Physical source import changed its producer binding")
    declared = parent.get("source_imports", {}).get(source["run"], {}).get(source["source_sha256"], row["original_receipt"])
    if not same_asset(declared, row["original_receipt"]):
        raise ValueError("Preserved source does not match its original plan import")
    return original.verify_ready_source(declared, source, parent, reader)


def verify_phase_import(descriptor, source, plan, reader):
    """A copied receipt retains its original inference producer and assets."""
    operation = plan["execution_operation"]
    proof = reader.record(operation["phase_import_proof"])
    if (proof.get("contract") != "ffac-selected-phase-physical-import-v1"
            or proof.get("complete") is not True
            or proof.get("byte_exact_receipt_copies") is not True
            or proof.get("maps_copied_or_recomputed") is not False
            or proof.get("original_phase_plan") != operation["original_phase_plan"]):
        raise ValueError("Phase imports lack preserved original producer evidence")
    rows = [r for r in proof["imports"] if r["run"] == source["run"] and r["source_sha256"] == source["source_sha256"]]
    if not rows:
        saved = reader.record(descriptor)
        if saved.get("qa", {}).get("execution_operation", {}).get("sha256") != plan["execution_sha256"]:
            raise ValueError("Completed phase lacks current execution or original import proof")
        return saved
    if len(rows) != 1 or rows[0]["staged_receipt"] != descriptor or rows[0]["original_receipt"]["sha256"] != descriptor["sha256"]:
        raise ValueError("Copied phase receipt changed original producer bytes")
    saved = reader.record(descriptor)
    if saved != reader.record(rows[0]["original_receipt"]) or saved.get("binding", {}).get("source") != source:
        raise ValueError("Copied phase changed source or original producer binding")
    return saved


def operation_root(plan):
    return plan["execution_operation"]["phase_output_root"]


def prefix_rows(inventory):
    """Baseline and chronological prefix through the first selected acquisition."""
    rows = original.select_rows(inventory)
    for index, row in enumerate(rows):
        if row.get("selected") and not row.get("baseline"):
            return rows[:index+1]
    raise ValueError("Selected run has no positive selected acquisition")


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def require_active_source_owner(plan, reader, plan_descriptor=None):
    path = Path(plan["execution_operation"]["source_owner_process_path"])
    if not path.is_file():
        return None
    owner = reader.record(original.asset(path))
    if (owner.get("common_stop") != plan["execution_operation"]["shared_stop_path"]
            or owner.get("internal_threads") != 1
            or owner.get("phase_plan") != plan_descriptor and plan_descriptor is not None):
        raise ValueError("Source owner changed the admitted execution plan/stop")
    approved = reader.record(plan["phase_campaign_binding"]["approved_main_host"])
    if owner.get("host") != approved["hostname"]:
        raise ValueError("Source owner is not on the approved Main host")
    return process_alive(owner["pid"])
