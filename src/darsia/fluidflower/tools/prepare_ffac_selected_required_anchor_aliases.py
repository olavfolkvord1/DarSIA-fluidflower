"""Create missing original-source aliases required by the widened identity plan."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .prepare_ffac_selected_anchor_aliases import alias_payload
from .prepare_ffac_selected_corrected_inputs import asset, bound_json, write_new


def required_acquisitions(identity):
    rows = {}
    for run, state in identity["runs"].items():
        for row in state["frames"]:
            key = (run, row["source_sha256"])
            if key in rows:
                raise ValueError("A required original acquisition was listed twice")
            rows[key] = row
    return rows


def prepare(root, plan_asset):
    root = Path(root)
    plan = bound_json(plan_asset)
    identity = bound_json(plan["identity_inventory"])
    required = required_acquisitions(identity)
    tool = asset(Path(__file__))
    aliases = []
    existing = []
    referenced = {}
    for run, state in plan["runs"].items():
        inventory = bound_json(state["source_inventory"])
        baseline = bound_json(state["reference_parent_receipt"])["baseline_extended_scientific"]
        sources = {row["source_sha256"]: row for row in inventory["sources"]}
        for (required_run, digest), row in required.items():
            if required_run != run:
                continue
            source = sources[digest]
            if not source["studio_anchor"] or int(source["studio_frame_index"]) != int(row["studio_frame_index"]):
                raise ValueError("Identity inventory does not select this original Studio acquisition")
            target = root / "selected_sources" / run / digest / "receipt.json"
            if target.exists():
                saved = bound_json(asset(target))
                if (saved.get("complete") is not True or saved["binding"]["source"] != source
                        or saved["binding"]["frozen_parent"] != plan["parent_chain"]["parent_seal"]):
                    raise ValueError("An existing wrapper has another source or frozen parent")
                imported = plan.get("source_imports", {}).get(run, {}).get(digest)
                if saved["binding"]["job_sha256"] != plan["job_sha256"] and imported != asset(target):
                    raise ValueError("An older wrapper must have explicit byte-exact import binding")
                existing.append(dict(run=run, frame_index=row["studio_frame_index"], source_sha256=digest,
                    receipt=asset(target), classification="byte-exact-imported-existing-wrapper"))
                continue
            original = bound_json(source["parent_receipt"])
            prepared_asset = original["full_extended_scientific"]["binding"]["prepared_source"]
            prepared = bound_json(prepared_asset)
            payload = alias_payload(plan, run, state, source, original, prepared, baseline,
                tool=tool, original_prepared_asset=prepared_asset)
            for field in ("log_linear_rgb", "valid_mask", "receipt"):
                item = payload["native_scientific"][field]
                if not Path(item["path"]).is_file():
                    raise FileNotFoundError("An original native scientific asset is missing")
                referenced[item["path"]] = item
            alias = write_new(target, payload)
            aliases.append(dict(run=run, frame_index=row["studio_frame_index"], source_sha256=digest,
                alias_receipt=alias, original_parent_receipt=source["parent_receipt"],
                original_prepared_receipt=prepared_asset,
                scientific_fields_sha256=payload["metadata_alias_provenance"]["original_producer_descriptor_fields_sha256"]))
    proof = write_new(root / "source_preparation/original_anchor_aliases/receipt.json", dict(
        contract="ffac-selected-original-anchor-metadata-alias-inventory-v1", complete=True,
        created_utc=datetime.now(timezone.utc).isoformat(), source_plan=plan_asset,
        identity_inventory=plan["identity_inventory"], tool=tool,
        required_scope="AC18/AC52 all original Studio frames; every one of 39 runs baseline through frame 2",
        existing_wrappers=len(existing), existing=existing,
        new_metadata_aliases=len(aliases), total_wrappers=len(required), aliases=aliases,
        referenced_original_assets=list(referenced.values()),
        source_plan_modified=False, original_scientific_arrays_modified=False,
        scientific_arrays_read_or_recomputed=False, source_workers_dispatched=False,
        original_producer_code_executed=False, numerical_model_evaluated=False,
        historical_identity_outputs_claimed=False))
    return dict(proof=proof, new_metadata_aliases=len(aliases), total_wrappers=len(required))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, asset(args.plan)), indent=2))


if __name__ == "__main__":
    main()
