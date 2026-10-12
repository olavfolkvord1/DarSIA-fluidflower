"""Admit actual fresh and source-equivalent old CPU identity receipts.

An imported receipt remains the old producer's computation. Only its explicit
consumer equivalence may refer to the new plan. No predicted pixels are loaded.
"""
from __future__ import annotations

SPEC_SHA256 = "62bf2970077c9a270f85fe3f046e48811af70d5e1c1761cd729d6289782bef32"
CONTRACT = "ffac-selected-baseline-role-required-identity-seal-v1"
CONTROL_KEYS = {("ac19", 0), ("ac19", 1), ("ac54", 0), ("ac54", 1), ("ac54", 2)}
PROBABILITIES = {"g3_pixel.npy", "regional_frozen.npy", "regional_refit.npy", "co2.packbits.npy", "extra_masks.packbits.npz"}
MASKS = {"photo_sand_domain", "feature_support", "regional_native_domain", "flatfield_valid",
    "feature_eligible", "optical_presence_ge_half", "frozen_gas", "frozen_unresolved"}


def _required_products(row, source, audit_descriptor):
    if (row.get("labels", {}).get("bit_identical") is not True
            or not row.get("generated_inherited_map", {}).get("sha256")):
        raise ValueError("Required labels or freshly generated inherited evidence is missing")
    products = row.get("stored_probabilities", {})
    comparisons = products.get("comparisons", {})
    if (products.get("all_bit_identical") is not True or set(comparisons) != PROBABILITIES
            or any(comparisons[name].get("bit_identical") is not True
                for name in PROBABILITIES - {"extra_masks.packbits.npz"})
            or products.get("receipt") != source["original_probability_receipt"]
            or products.get("frozen_gas_audit") != audit_descriptor):
        raise ValueError("A required original probability product differs or is unbound")
    masks = products.get("extra_masks", {})
    checks = masks.get("comparisons", {})
    if (set(checks) != MASKS or masks.get("frozen_gas_required") is not False
            or masks.get("all_required_bit_identical") is not True
            or any(checks[name].get("required") is not True or checks[name].get("bit_identical") is not True
                for name in MASKS - {"frozen_gas"})):
        raise ValueError("A required nondiagnostic mask differs or was suppressed")
    diagnostic = checks["frozen_gas"]
    if (diagnostic.get("required") is not False or type(diagnostic.get("changed_pixels")) is not int
            or diagnostic["changed_pixels"] < 0 or row.get("frozen_gas") != diagnostic):
        raise ValueError("Diagnostic frozen_gas counts were lost or used as an acceptance mask")
    inherited = products.get("inherited_map", {})
    inherited_checks = inherited.get("comparisons", {})
    if (inherited.get("computed_from_scientific_images") is not True
            or inherited.get("archived_labels_used_as_inference_input") is not False
            or inherited.get("all_bit_identical") is not True
            or set(inherited_checks) != {"co2_union", "water", "unresolved"}
            or any(value.get("bit_identical") is not True for value in inherited_checks.values())
            or inherited.get("expected_original_stage_map") != source["original_inherited_map"]):
        raise ValueError("Fresh inherited CO2/water/unresolved identity was not demonstrated")
    return diagnostic


def require_composite_identity_gate(descriptor, reader, *, expected_runs):
    gate = reader.record(descriptor)
    flags = ("complete", "all_39_complete", "required_identity_passed", "cpu_only")
    if (gate.get("contract") != CONTRACT or gate.get("spec_sha256") != SPEC_SHA256
            or any(gate.get(key) is not True for key in flags)
            or set(gate.get("runs", [])) != set(expected_runs) or len(gate["runs"]) != len(expected_runs)):
        raise ValueError("The real all-run composite Main CPU identity seal is required")
    plan = reader.record(gate["identity_plan"])
    binding = plan["phase_campaign_binding"]
    if (set(plan["runs"]) != set(expected_runs) or binding.get("spec_sha256") != SPEC_SHA256
            or binding.get("stage_conventions_independently_bound") is not True
            or binding.get("fresh_inherited_map_required") is not True
            or binding.get("archived_inherited_labels_used_as_inference_input") is not False
            or binding.get("identity_inventory") != gate["identity_inventory"]
            or binding.get("approved_main_host") != gate["approved_main_host"]):
        raise ValueError("Composite identity binds another source cohort or constructor convention")
    approved_host = reader.record(gate["approved_main_host"])
    inventory = reader.record(gate["identity_inventory"])
    if (inventory.get("complete") is not True or set(inventory["runs"]) != set(expected_runs)
            or inventory.get("every_sequence_starts_from_baseline") is not True
            or inventory.get("only_original_parent_history") is not True
            or inventory.get("selected_history_used") is not False
            or inventory.get("inherited_map_must_be_computed_not_loaded") is not True):
        raise ValueError("Original own-history inventory is incomplete")
    expected = {}
    for run, state in inventory["runs"].items():
        indices = [row["studio_frame_index"] for row in state["frames"]]
        if (indices != list(range(len(indices))) or len(indices) < 3
                or run not in ("ac18", "ac52") and len(indices) != 3
                or state.get("start_from_baseline") is not True):
            raise ValueError("A whole required original history is missing")
        expected.update({(run, row["studio_frame_index"]): row for row in state["frames"]})
    if len(expected_runs) == 39 and len(expected) != 153:
        raise ValueError("The composite seal must include all 153 original own-history frames")
    current_audit_descriptor = binding["frozen_gas_audit"]
    current_audit = reader.record(current_audit_descriptor)
    from .phase_ffac_selected_required_identity import require_frozen_gas_audit
    require_frozen_gas_audit(current_audit)
    helper = "darsia.fluidflower.phase_ffac_selected_required_identity_baseline"
    if (current_audit.get("required_helper_binding_pending") is not False
            or current_audit.get("runtime_runner_binding_pending") is not False
            or current_audit.get("spec_sha256") != SPEC_SHA256
            or current_audit.get("required_identity_helper_sha256") != binding["code"][helper]["sha256"]):
        raise ValueError("The actual corrected constructor and frozen_gas consumer audit are unbound")
    used_audit = reader.record(gate["used_input_audit"])
    if (used_audit.get("contract") != "ffac-selected-baseline-role-comprehensive-used-input-array-audit-v1"
            or used_audit.get("complete") is not True or used_audit.get("full_closure_checked") is not True):
        raise ValueError("The comprehensive used-input array audit is incomplete")
    checked, diagnostics, origins = {}, [], {}
    for entry in gate.get("rows", []):
        key = (entry["run"], entry["original_studio_frame_index"])
        if key not in expected or key in checked:
            raise ValueError("Duplicate or unexpected composite identity frame")
        source = expected[key]
        row = reader.record(entry["receipt"])
        if (any(row.get(flag) is not True for flag in ("complete", "cpu_only", "all_pass"))
                or row.get("run") != key[0] or row.get("original_studio_frame_index") != key[1]
                or row.get("source_sha256") != source["source_sha256"]
                or row.get("elapsed_h") != source["elapsed_h"]
                or row.get("previous_source_sha256") != source["original_previous_source_sha256"]
                or entry.get("source_sha256") != row["source_sha256"]
                or entry.get("previous_source_sha256") != row["previous_source_sha256"]):
            raise ValueError("Composite producer receipt has a changed source/history or required failure")
        producer_plan = reader.record(row["identity_plan"])
        producer_audit_descriptor = producer_plan["phase_campaign_binding"]["frozen_gas_audit"]
        require_frozen_gas_audit(reader.record(producer_audit_descriptor))
        diagnostic = _required_products(row, source, producer_audit_descriptor)
        if entry.get("origin") == "fresh":
            host = row.get("phase_execution_host", {})
            if (row.get("contract") != "ffac-selected-baseline-role-required-cpu-identity-v1"
                    or row["identity_plan"] != gate["identity_plan"] or entry.get("import_equivalence") is not None
                    or host.get("hostname") != approved_host["hostname"]
                    or host.get("machine") != approved_host["machine"]):
                raise ValueError("Fresh identity was not produced by the bound new Main CPU runtime")
        elif entry.get("origin") == "imported":
            if (row.get("contract") != "ffac-selected-required-cpu-identity-v2"
                    or key in CONTROL_KEYS or row["identity_plan"] == gate["identity_plan"]):
                raise ValueError("Fresh controls cannot be replaced by imported or relabelled receipts")
            equivalent = reader.record(entry["import_equivalence"])
            index = equivalent.get("used_input_audit_row_index")
            if type(index) is not int or not 0 <= index < len(used_audit["rows"]):
                raise ValueError("Imported identity lacks a precise comprehensive array-audit row")
            proof = used_audit["rows"][index]
            if (equivalent.get("original_receipt") != entry["receipt"]
                    or equivalent.get("new_consumer_identity_plan") != gate["identity_plan"]
                    or equivalent.get("used_input_audit") != gate["used_input_audit"]
                    or equivalent.get("original_identity_plan") != row["identity_plan"]
                    or equivalent.get("original_runtime_and_numerical_producer_preserved") is not True
                    or equivalent.get("all_used_input_arrays_sha256_unchanged") is not True
                    or equivalent.get("no_new_inference") is not True
                    or equivalent.get("source_sha256") != row["source_sha256"]
                    or equivalent.get("previous_source_sha256") != row["previous_source_sha256"]
                    or proof.get("run") != key[0] or proof.get("original_studio_frame_index") != key[1]
                    or proof.get("old_identity_receipt") != entry["receipt"]
                    or proof.get("full_closure_checked") is not True
                    or key[0] in used_audit.get("rerun_runs", [])
                    or proof.get("all_used_input_arrays_sha256_unchanged") is not True):
                raise ValueError("Old producer cannot be admitted without unchanged actual used-input SHA evidence")
        else:
            raise ValueError("Composite identity row origin is unknown")
        checked[key] = entry["receipt"]
        origins[key] = entry["origin"]
        diagnostics.append(dict(run=key[0], original_studio_frame_index=key[1], **diagnostic))
    if set(checked) != set(expected) or any(origins.get(key) != "fresh" for key in CONTROL_KEYS if key in expected):
        raise ValueError("Actual composite evidence omits a required whole history or fresh direct control")
    return dict(identity_gate=descriptor, identity_plan=gate["identity_plan"],
        required_identity_stages=dict(composite_seal=descriptor, used_input_audit=gate["used_input_audit"]),
        runs=list(expected_runs), compared_row_receipts=[checked[key] for key in sorted(checked)],
        frozen_gas_diagnostics=diagnostics, frozen_gas_audit=current_audit_descriptor,
        verified=True, mode="historical-stage-conventions", code=binding["code"],
        source_preparation=binding["source_preparation"], approved_main_host=gate["approved_main_host"],
        old_receipts_relabelled=False, scorer_pending=True, round3_allowed=False)


__all__ = ["require_composite_identity_gate", "CONTRACT", "SPEC_SHA256"]
