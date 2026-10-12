"""Prepare exact original-source metadata for the widened CPU identity replay.

This tool copies source wrapper receipts byte for byte. It never reads RAW,
scientific arrays, labels, inherited maps, or model probabilities. Descriptors
for those inputs are retained so the separate replay computes every stage.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .prepare_ffac_selected_corrected_inputs import (
    asset, bound_json, canonical_sha, native_record, verify_import, write_new,
)
from .prepare_ffac_selected_policy_a_inputs import copy_exact_metadata, original_predecessor


def required_indices(run, anchors):
    """Every AC18/52 frame; baseline through frame 2 in every other run."""
    indices = sorted(int(index) for index in anchors)
    if not indices or indices != list(range(indices[-1] + 1)):
        raise ValueError("Original Studio chronology is incomplete")
    through = indices[-1] if run in ("ac18", "ac52") else 2
    if through not in anchors:
        raise ValueError("Original Studio frame 2 is required in every run")
    return list(range(through + 1))


def identity_row(run, index, source, state, native, imported):
    parent = source["parent_source"]
    original_index = parent["parent_source_frame_index"]
    cache = parent.get("retained_source_cache")
    probability = cache["parent_probability_receipt"] if cache else None
    document = bound_json(probability) if probability else None
    metadata = document.get("metadata", {}) if document else {}
    producer_asset = state["original_producer_receipts"][source["source_sha256"]]
    producer = bound_json(producer_asset)
    inherited_map = None
    if metadata.get("frozen_map_path") and metadata.get("frozen_map_sha256"):
        inherited_map = {"path": metadata["frozen_map_path"], "sha256": metadata["frozen_map_sha256"]}
    return dict(
        run=run, studio_frame_index=index, elapsed_h=source["elapsed_h"],
        source_sha256=source["source_sha256"],
        source_record=dict(inventory=state["source_inventory"], work_index=source["work_index"]),
        source_receipt=imported.get(source["source_sha256"]),
        original_previous_source_sha256=original_predecessor(source),
        original_previous_scientific_task_id=parent.get("previous_scientific_task_id"),
        original_temporal_predecessor=parent.get("temporal_predecessor"),
        temporal_input_policy="original-parent-own-history-only",
        selected_predecessor_must_not_be_used=True,
        parent_native_manifest=state["parent_native_manifest"],
        current_native=(native_record(native["frame_sources"][original_index]) if original_index is not None else None),
        baseline_native=native_record(native["frame_sources"][0]),
        original_probability_receipt=probability,
        original_probability_files=document["files"] if document else None,
        original_probability_metadata=metadata if document else None,
        original_inherited_map=inherited_map,
        original_inherited_map_is_comparison_only=True,
        reconstructed_inherited_map_must_be_computed=True,
        original_stage_conventions=dict(
            probability_stage_receipt=probability,
            probability_spacing_m=metadata.get("spacing_m"),
            probability_pixel_area_mm2=metadata.get("pixel_area_mm2"),
            regional_working_spacing_m=metadata.get("regional_working_spacing_m"),
            original_alignment=metadata.get("alignment"),
            storage_layout=metadata.get("storage_layout"),
            masks_order=metadata.get("masks_order"),
            bitorder=metadata.get("bitorder"),
            inherited_source_candidate=(document.get("binding", {}).get("frozen_candidate") if document else None),
            one_stage_convention_must_not_be_applied_to_another=True),
        original_producer_receipt=producer_asset,
        original_unscaled_area=producer["rig_domain_revision"]["area_map"],
        original_parent_frame_receipt=source["parent_receipt"],
        original_source_changed=parent["source_changed"])


def prepare(parent_job, new_job, candidate_name, approved_spec, expected_spec_sha):
    parent_job, new_job = Path(parent_job), Path(new_job)
    if parent_job.resolve() == new_job.resolve():
        raise ValueError("A new campaign folder is required")
    folder = new_job / "source_preparation"
    if (folder / "plan.json").exists():
        raise FileExistsError("The new source plan already exists")
    parent_asset = asset(parent_job / "source_preparation/plan.json")
    parent = bound_json(parent_asset)
    spec = asset(approved_spec)
    if spec["sha256"] != expected_spec_sha:
        raise ValueError("Approved identity specification changed")
    selection = bound_json(parent["binding"]["selection"])
    if len(parent["runs"]) != 39 or selection["runs"] != list(parent["runs"]):
        raise ValueError("The completed 39-run selection must stay unchanged")
    imports, copied, identities, missing = {}, [], {}, []
    for run, state in parent["runs"].items():
        inventory = bound_json(state["source_inventory"])
        native = bound_json(state["parent_native_manifest"])
        anchors = {int(row["studio_frame_index"]): row for row in inventory["sources"] if row["studio_anchor"]}
        for source in inventory["sources"]:
            old = parent_job / "selected_sources" / run / source["source_sha256"] / "receipt.json"
            if not old.is_file():
                continue
            saved = bound_json(asset(old))
            verify_import(saved, source, parent["parent_chain"]["parent_seal"])
            original, staged = copy_exact_metadata(old, new_job / "selected_sources" / run / source["source_sha256"] / "receipt.json")
            imports.setdefault(run, {})[source["source_sha256"]] = staged
            copied.append(dict(run=run, source_sha256=source["source_sha256"], producer_receipt=original,
                staged_receipt=staged, producer_binding=saved["binding"], arrays_copied=False,
                inference_reperformed=False, imported_metadata_alias=bool(saved.get("metadata_alias_provenance"))))
        rows = []
        for index in required_indices(run, anchors):
            source = anchors[index]
            row = identity_row(run, index, source, state, native, imports.get(run, {}))
            rows.append(row)
            if row["source_receipt"] is None:
                missing.append(dict(run=run, frame_index=index, source_sha256=source["source_sha256"],
                    original_parent_receipt=source["parent_receipt"], source_record=row["source_record"]))
        identities[run] = dict(through_studio_frame=rows[-1]["studio_frame_index"], frames=rows,
            parent_history_only=True, source_inventory=state["source_inventory"], start_from_baseline=True)
    import_asset = write_new(folder / "source_imports/receipt.json", dict(
        contract="ffac-selected-required-identity-source-imports-v1", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), parent_source_plan=parent_asset,
        imported_count=len(copied), imports=copied, byte_exact_receipt_copies=True,
        producer_code_and_source_binding_preserved=True, arrays_copied_or_recomputed=False,
        prior_jobs_changed=False))
    identities_asset = write_new(folder / "identity_inventory.json", dict(
        contract="ffac-selected-original-history-identity-inventory-v1", complete=True,
        approved_spec=spec, original_parent_seal=parent["parent_chain"]["parent_seal"],
        run_count=39, frame_count=sum(len(run["frames"]) for run in identities.values()),
        first_probe_runs=["ac19", "ac18", "ac52"], runs=identities,
        execution_stages=[dict(runs=["ac19"], through_studio_frame=1),
            dict(runs=["ac18", "ac52"], through_studio_frame="all"),
            dict(runs=list(parent["runs"]), through_studio_frame=2)],
        every_sequence_starts_from_baseline=True, backend="cpu",
        required_bit_exact=dict(final_labels=True, probability_layers=True, co2=True,
            extra_masks_except_frozen_gas=True, inherited_co2_water_unresolved=True),
        frozen_gas_compare_and_report_only_if_no_other_consumers=True,
        inherited_map_must_be_computed_not_loaded=True,
        only_original_parent_history=True, selected_history_used=False,
        identity_outputs_not_policy_a_outputs=True, model_evaluation_performed=False))
    missing_asset = write_new(folder / "required_alias_availability.json", dict(
        contract="ffac-required-identity-alias-availability-v1", complete=True,
        identity_inventory=identities_asset, imported_count=len(copied), missing_count=len(missing),
        missing=missing, no_raw_or_model_compute=True))
    saved_parts = []
    for record in bound_json(parent["saved_policy_a_parts"])["saved_parts"]:
        old = record["staged_receipt"]
        src = Path(old["path"])
        original, staged = copy_exact_metadata(src,
            new_job / candidate_name / "imported_policy_a_outputs" / src.parent.name / src.name)
        preserved = deepcopy(record)
        preserved.update(imported_receipt=original, staged_receipt=staged,
            new_identity_evidence=False, accepted_under_new_policy=False)
        saved_parts.append(preserved)
    parts_asset = write_new(folder / "saved_policy_a_parts.json", dict(
        contract="ffac-selected-policy-a-saved-parts-v1", complete=True, saved_parts=saved_parts,
        count=len(saved_parts), no_part_is_own_history_evidence=True, no_new_phase_computation=True))
    plan = deepcopy(parent)
    plan["source_imports"] = imports
    plan["binding"].update(approved_spec=spec, required_identity_parent_source_plan=parent_asset,
        required_identity_campaign_root=str(new_job), consumer_candidate=candidate_name,
        source_imports=imports, exact_existing_source_imports=import_asset,
        own_history_identity_inventory=identities_asset,
        own_history_identity_gate_required_before_dispatch=True,
        required_identity_alias_availability=missing_asset,
        selected_policy_a_coarse_gate=dict(co2_relative=0.01, gas_relative=0.02, absolute_cm2=0.5),
        saved_policy_a_outputs=parts_asset, original_selection_and_project_TOMLs_changed=False,
        continuation_planner=asset(Path(__file__)))
    plan["job_sha256"] = canonical_sha(plan["binding"])
    plan.update(source_import_proof=import_asset, identity_inventory=identities_asset,
                saved_policy_a_parts=parts_asset)
    plan_asset = write_new(folder / "plan.json", plan)
    ready = write_new(folder / "readiness.json", dict(
        contract="ffac-selected-required-identity-source-readiness-v1", metadata_complete=True,
        approved_spec=spec, source_plan=plan_asset, identity_inventory=identities_asset,
        source_imports=import_asset, saved_policy_a_parts=parts_asset, missing_aliases=missing_asset,
        selection=parent["binding"]["selection"], source_count=parent["totals"]["sources"],
        selected_count=parent["totals"]["selected"], imported_count=len(copied),
        source_workers_dispatched=False, phase_workers_dispatched=False,
        mass_or_round3_started=False, automatic_registration=False,
        dispatch_requires_full_required_original_history_identity_pass=True))
    return dict(plan=plan_asset, identity_inventory=identities_asset, source_imports=import_asset,
        missing_aliases=missing_asset, missing_count=len(missing), imported_count=len(copied),
        identity_frame_count=sum(len(run["frames"]) for run in identities.values()), readiness=ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", type=Path, required=True)
    parser.add_argument("--new-job", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--approved-spec", type=Path, required=True)
    parser.add_argument("--expected-spec-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_job, args.new_job, args.candidate_name,
        args.approved_spec, args.expected_spec_sha), indent=2))


if __name__ == "__main__":
    main()
