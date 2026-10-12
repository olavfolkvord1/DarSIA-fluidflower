"""Metadata-only admission of a preserved producer's narrower inventory.

The producer receipt is never relabelled. Imported scope is admissible only
when the sealed byte-copy proof and its original inventory contain the exact
current source object, including its predecessor and all scientific assets.
"""
from __future__ import annotations


def verify_ready_source_scope(descriptor, source, plan, reader):
    saved = reader.record(descriptor)
    binding = saved.get("binding", {})
    if (saved.get("contract") != "ffac-selected-frozen-scientific-source-v1"
            or saved.get("complete") is not True or saved.get("run") != source["run"]
            or saved.get("source_sha256") != source["source_sha256"]
            or saved.get("elapsed_h") != source["elapsed_h"]
            or binding.get("source") != source
            or binding.get("frozen_parent") != plan["parent_chain"]["parent_seal"]):
        raise ValueError("Ready scientific source changed source/time/parent/assets")
    imported = plan.get("source_imports", {}).get(source["run"], {}).get(source["source_sha256"])
    current_inventory = plan["runs"][source["run"]]["source_inventory"]
    if imported is None:
        if (binding.get("source_inventory") != current_inventory
                or binding.get("job_sha256") != plan["job_sha256"]):
            raise ValueError("Current scientific producer changed inventory/job")
        return saved
    evidence = plan["phase_campaign_binding"]["original_source_alias_audit"]
    proof = reader.record(evidence["import_proof"])
    if (proof.get("complete") is not True or proof.get("byte_exact_receipt_copies") is not True
            or proof.get("producer_code_and_source_binding_preserved") is not True
            or proof.get("arrays_copied_or_recomputed") is not False):
        raise ValueError("Imported source requires the sealed byte-exact producer proof")
    matches = [row for row in proof["imports"] if row["run"] == source["run"]
               and row["source_sha256"] == source["source_sha256"]]
    if (len(matches) != 1 or descriptor != imported
            or matches[0]["staged_receipt"] != descriptor
            or matches[0]["producer_receipt"]["sha256"] != descriptor["sha256"]
            or matches[0]["producer_binding"] != binding):
        raise ValueError("Imported source is not its exact original producer binding")
    original = reader.record(matches[0]["producer_receipt"])
    if original != saved:
        raise ValueError("Imported source differs from the preserved producer receipt")
    producer_inventory = reader.record(binding["source_inventory"])
    rows = [row for row in producer_inventory["sources"]
            if row.get("source_sha256") == source["source_sha256"]]
    if len(rows) != 1 or rows[0] != source:
        raise ValueError("Original inventory does not contain the exact current source/predecessor")
    return saved


__all__ = ["verify_ready_source_scope"]
