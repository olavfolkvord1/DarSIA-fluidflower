"""Preserve selected scientific sources in a new identity/Policy A campaign.

This metadata-only command never decodes RAW or evaluates a model. The two
histories are deliberately separate: imported source wrappers retain their
original producer binding; identity probes use the parent's declared history.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from .prepare_ffac_selected_corrected_inputs import (
    asset, bound_json, canonical_sha, native_record, sha, verify_import, write_new,
)


def original_predecessor(source):
    """Read original history without consulting the selected predecessor."""
    parent = source["parent_source"]
    task = parent.get("previous_scientific_task_id")
    temporal = parent.get("temporal_predecessor")
    task_sha = None
    if task is not None:
        prefix = source["run"] + "/source-"
        if not task.startswith(prefix) or len(task[len(prefix):]) != 64:
            raise ValueError("Original predecessor task has another run/source")
        task_sha = task[len(prefix):]
    temporal_sha = temporal.get("source_sha256") if temporal else None
    if task_sha and temporal_sha and task_sha != temporal_sha:
        raise ValueError("Parent predecessor declarations disagree")
    return task_sha or temporal_sha


def copy_exact_metadata(source, target):
    source, target = Path(source), Path(target)
    expected = asset(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(source.read_bytes())
    actual = asset(target)
    if actual["sha256"] != expected["sha256"]:
        raise ValueError("Immutable metadata import lost byte equivalence")
    return expected, actual


def prepare(parent_job, new_job, candidate_name, approved_spec):
    parent_job, new_job = Path(parent_job), Path(new_job)
    if parent_job.resolve() == new_job.resolve():
        raise ValueError("A new campaign directory is required")
    input_dir = new_job / "source_preparation"
    if (input_dir / "plan.json").exists():
        raise FileExistsError("The new source plan already exists")
    parent_plan_asset = asset(parent_job / "source_preparation/plan.json")
    parent = bound_json(parent_plan_asset)
    spec_asset = asset(approved_spec)
    selection_asset = parent["binding"]["selection"]
    selection = bound_json(selection_asset)
    if len(parent["runs"]) != 39 or selection["runs"] != list(parent["runs"]):
        raise ValueError("The exact completed 39-run selection must be retained")
    imported, imports, identities = [], {}, {}
    for run, state in parent["runs"].items():
        inventory = bound_json(state["source_inventory"])
        native = bound_json(state["parent_native_manifest"])
        anchor_rows = {int(s["studio_frame_index"]): s for s in inventory["sources"]
                       if s["studio_anchor"]}
        through = 5 if run == "ac18" else 1
        identity_rows = []
        for source in inventory["sources"]:
            source_sha = source["source_sha256"]
            existing = parent_job / "selected_sources" / run / source_sha / "receipt.json"
            if existing.is_file():
                saved = bound_json(asset(existing))
                verify_import(saved, source, parent["parent_chain"]["parent_seal"])
                producer, staged = copy_exact_metadata(
                    existing, new_job / "selected_sources" / run / source_sha / "receipt.json")
                imports.setdefault(run, {})[source_sha] = staged
                imported.append(dict(run=run, source_sha256=source_sha,
                    producer_receipt=producer, staged_receipt=staged,
                    producer_binding=saved["binding"], arrays_copied=False,
                    inference_reperformed=False))
        for index in range(through + 1):
            source = anchor_rows[index]
            parent_source = source["parent_source"]
            previous_sha = original_predecessor(source)
            original_index = parent_source["parent_source_frame_index"]
            cache = parent_source.get("retained_source_cache")
            probability = cache["parent_probability_receipt"] if cache else None
            probability_document = bound_json(probability) if probability else None
            producer_receipt = state["original_producer_receipts"][source["source_sha256"]]
            producer_document = bound_json(producer_receipt)
            baseline_native = native_record(native["frame_sources"][0])
            input_native = (native_record(native["frame_sources"][original_index])
                            if original_index is not None else None)
            identity_rows.append(dict(
                run=run, studio_frame_index=index, elapsed_h=source["elapsed_h"],
                source_sha256=source["source_sha256"],
                source_record=dict(inventory=state["source_inventory"],
                                   work_index=source["work_index"]),
                source_receipt=imports.get(run, {}).get(source["source_sha256"]),
                original_previous_source_sha256=previous_sha,
                original_previous_scientific_task_id=parent_source.get("previous_scientific_task_id"),
                original_temporal_predecessor=parent_source.get("temporal_predecessor"),
                temporal_input_policy="original-parent-own-history-only",
                selected_predecessor_must_not_be_used=True,
                parent_native_manifest=state["parent_native_manifest"],
                current_native=input_native, baseline_native=baseline_native,
                original_probability_receipt=probability,
                original_probability_files=(probability_document["files"]
                                            if probability_document else None),
                original_probability_metadata=(probability_document.get("metadata")
                                               if probability_document else None),
                original_producer_receipt=producer_receipt,
                original_unscaled_area=producer_document["rig_domain_revision"]["area_map"],
                original_parent_frame_receipt=source["parent_receipt"],
                original_source_changed=parent_source["source_changed"]))
        identities[run] = dict(through_studio_frame=through, frames=identity_rows,
            parent_history_only=True, source_inventory=state["source_inventory"])
    import_proof = write_new(input_dir / "source_imports/receipt.json", dict(
        contract="ffac-selected-policy-a-source-imports-v1", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), parent_source_plan=parent_plan_asset,
        imported_count=len(imported), imports=imported,
        byte_exact_receipt_copies=True, producer_code_and_source_binding_preserved=True,
        arrays_copied_or_recomputed=False, prior_jobs_changed=False))
    identity_asset = write_new(input_dir / "identity_inventory.json", dict(
        contract="ffac-selected-original-history-identity-inventory-v1", complete=True,
        approved_spec=spec_asset, original_parent_seal=parent["parent_chain"]["parent_seal"],
        run_count=39, frame_count=sum(len(v["frames"]) for v in identities.values()),
        first_probe_runs=["ac19", "ac18"], runs=identities,
        backend="cpu", bit_exact_labels_and_probabilities_required=True,
        only_original_parent_history=True, selected_history_used=False,
        identity_outputs_not_policy_a_outputs=True, model_evaluation_performed=False))
    saved_phase_parts = []
    old_candidate = parent_job / parent["binding"]["consumer_candidate"]
    for existing in sorted((old_candidate / "selected_phase_frames").glob("*/*.json")):
        producer, staged = copy_exact_metadata(existing,
            new_job / candidate_name / "imported_policy_a_outputs" / existing.parent.name / existing.name)
        saved = bound_json(producer)
        saved_phase_parts.append(dict(producer_receipt=producer, staged_receipt=staged,
            source_sha256=saved.get("source_sha256"), run=existing.parent.name,
            classification="selected-history-policy-a-only",
            original_binding=saved.get("binding"), original_QA=saved.get("qa"),
            new_identity_evidence=False, accepted_under_new_policy=False,
            numerical_assets_copied=False, reuse_requires_consumer_equivalence=True))
    phase_import_proof = write_new(input_dir / "saved_policy_a_parts.json", dict(
        contract="ffac-selected-policy-a-saved-parts-v1", complete=True,
        saved_parts=saved_phase_parts, count=len(saved_phase_parts),
        no_part_is_own_history_evidence=True, no_new_phase_computation=True))
    plan = deepcopy(parent)
    plan["source_imports"] = imports
    plan["binding"].update(approved_spec=spec_asset,
        identity_policy_a_parent_source_plan=parent_plan_asset,
        identity_policy_a_campaign_root=str(new_job), consumer_candidate=candidate_name,
        source_imports=imports, exact_existing_source_imports=import_proof,
        own_history_identity_inventory=identity_asset,
        own_history_identity_gate_required_before_dispatch=True,
        selected_policy_a_coarse_gate=dict(co2_relative=0.01, gas_relative=0.02, absolute_cm2=0.5),
        saved_policy_a_outputs=phase_import_proof,
        original_selection_and_project_TOMLs_changed=False,
        continuation_planner=asset(Path(__file__)))
    plan["job_sha256"] = canonical_sha(plan["binding"])
    plan.update(source_import_proof=import_proof, identity_inventory=identity_asset,
                saved_policy_a_parts=phase_import_proof)
    plan_asset = write_new(input_dir / "plan.json", plan)
    readiness = write_new(input_dir / "readiness.json", dict(
        contract="ffac-selected-policy-a-source-readiness-v1", complete=True,
        approved_spec=spec_asset, source_plan=plan_asset, identity_inventory=identity_asset,
        source_imports=import_proof, saved_policy_a_parts=phase_import_proof,
        selection=selection_asset, source_count=parent["totals"]["sources"],
        selected_count=parent["totals"]["selected"], imported_count=len(imported),
        source_workers_dispatched=False, phase_workers_dispatched=False,
        mass_or_round3_started=False, automatic_registration=False,
        dispatch_requires_all39_original_history_identity_pass=True))
    return dict(plan=plan_asset, readiness=readiness, identity_inventory=identity_asset,
                source_imports=import_proof, imported_count=len(imported),
                saved_policy_a_parts=phase_import_proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job", type=Path, required=True)
    parser.add_argument("--new-job", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--approved-spec", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parent_job, args.new_job, args.candidate_name,
                             args.approved_spec), indent=2))


if __name__ == "__main__":
    main()
