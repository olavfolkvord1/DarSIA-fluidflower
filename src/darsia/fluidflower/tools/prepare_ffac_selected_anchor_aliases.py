"""Make metadata-only aliases for original Studio scientific inputs.

This reproduces the selected source preparer's original-anchor descriptor
branch without constructing a RAW processor, decoding pixels, or inference.
Producer receipts and scientific assets remain immutable at their old paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .prepare_ffac_selected_corrected_inputs import (
    asset, bound_json, canonical_sha, write_new,
)

CONTRACT = "ffac-selected-frozen-scientific-source-v1"


def alias_payload(plan, run, state, source, parent, prepared, baseline,
                  *, tool, original_prepared_asset):
    digest = source["source_sha256"]
    if not source.get("studio_anchor"):
        raise ValueError("Only an original Studio source can be aliased")
    if prepared["task"]["source"]["sha256"] != digest:
        raise ValueError("Original scientific producer has another acquisition")
    full = parent["full_extended_scientific"]
    geometry = parent["binding"]["source"]["rig_geometry"]
    if geometry["source_sha256"] != digest or geometry["run"] != run:
        raise ValueError("Original geometry has another run/source")
    binding = dict(contract=CONTRACT, job_sha256=plan["job_sha256"], source=source,
        frozen_parent=plan["parent_chain"]["parent_seal"],
        source_inventory=state["source_inventory"])
    scientific_fields = dict(
        native_scientific=prepared["full_native_scientific"],
        prepared_source=full["binding"]["prepared_source"],
        full_extended_scientific=full, baseline_extended_scientific=baseline,
        geometry_frame=prepared["geometry_frame"], rig_geometry=geometry)
    receipt = dict(contract=CONTRACT, complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), binding=binding,
        run=run, source_sha256=digest, elapsed_h=source["elapsed_h"],
        **scientific_fields,
        source_audit=dict(reused_exact_scientific_source=True,
            parent_receipt=source["parent_receipt"], native_centre_changed=False,
            rig_geometry_changed=False, inference_performed=False,
            metadata_only_original_source_alias=True, scientific_arrays_generated=False),
        metadata_alias_provenance=dict(
            contract="ffac-selected-original-anchor-metadata-alias-v1",
            tool=tool, original_parent_receipt=source["parent_receipt"],
            original_prepared_receipt=original_prepared_asset,
            original_producer_descriptor_fields_sha256=canonical_sha(scientific_fields),
            descriptor_fields_copied_exactly=True, scientific_arrays_changed=False,
            RAW_decoding=False, original_code_executed=False,
            original_branch="SelectedRunSourcePreparer.prepare: studio_anchor descriptor reuse"),
        training=False, labels_read=False, phase_inference=False,
        bottom_mass_allowed=False)
    return receipt


def prepare(root, plan_asset, *, through=1):
    root = Path(root)
    plan = bound_json(plan_asset)
    tool = asset(Path(__file__))
    rows = []
    existing_count = 0
    referenced_assets = {}
    for run, state in plan["runs"].items():
        inventory = bound_json(state["source_inventory"])
        baseline_parent = bound_json(state["reference_parent_receipt"])
        baseline = baseline_parent["baseline_extended_scientific"]
        for source in inventory["sources"]:
            if not source.get("studio_anchor") or int(source["studio_frame_index"]) > through:
                continue
            destination = root / "selected_sources" / run / source["source_sha256"] / "receipt.json"
            if destination.exists():
                old = bound_json(asset(destination))
                if (old.get("complete") is not True or old["binding"]["source"] != source
                        or old["binding"]["frozen_parent"] != plan["parent_chain"]["parent_seal"]):
                    raise ValueError("Existing source wrapper is not this original acquisition")
                existing_count += 1
                continue
            parent = bound_json(source["parent_receipt"])
            full = parent["full_extended_scientific"]
            prepared_asset = full["binding"]["prepared_source"]
            prepared = bound_json(prepared_asset)
            saved = alias_payload(plan, run, state, source, parent, prepared, baseline,
                                  tool=tool, original_prepared_asset=prepared_asset)
            native = saved["native_scientific"]
            for key in ("log_linear_rgb", "valid_mask", "receipt"):
                item = native[key]
                if not Path(item["path"]).is_file():
                    raise FileNotFoundError("An original native scientific asset is missing")
                referenced_assets[item["path"]] = item
            aliased = write_new(destination, saved)
            rows.append(dict(run=run, frame_index=source["studio_frame_index"],
                source_sha256=source["source_sha256"], alias_receipt=aliased,
                original_parent_receipt=source["parent_receipt"],
                original_prepared_receipt=prepared_asset,
                scientific_fields_sha256=saved["metadata_alias_provenance"]
                    ["original_producer_descriptor_fields_sha256"]))
    proof = write_new(root / "source_preparation/original_anchor_aliases/receipt.json", dict(
        contract="ffac-selected-original-anchor-metadata-alias-inventory-v1", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_plan=plan_asset,
        tool=tool, through_studio_frame=through, existing_wrappers=existing_count,
        new_metadata_aliases=len(rows), total_wrappers=existing_count + len(rows),
        aliases=rows, referenced_original_assets=list(referenced_assets.values()),
        source_plan_modified=False, original_scientific_arrays_modified=False,
        scientific_arrays_read_or_recomputed=False, source_workers_dispatched=False,
        original_producer_code_executed=False, numerical_model_evaluated=False,
        historical_identity_outputs_claimed=False))
    return dict(proof=proof, new_metadata_aliases=len(rows),
                total_wrappers=existing_count + len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--through", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, asset(args.plan), through=args.through), indent=2))


if __name__ == "__main__":
    main()
