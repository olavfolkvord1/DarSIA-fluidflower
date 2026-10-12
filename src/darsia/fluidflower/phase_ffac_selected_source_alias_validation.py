"""Validate metadata aliases against immutable original scientific producers.

Only JSON and the small alias producer tool are read. Dense scientific inputs
retain their original descriptors and are verified by the scientific loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class AliasMetadataVerifier:
    """Verify small immutable metadata once per file version in a worker."""
    def __init__(self):
        self._files = {}

    def read(self, descriptor):
        path = Path(descriptor["path"])
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        item = self._files.get(key)
        if item is None:
            payload = path.read_bytes()
            item = (hashlib.sha256(payload).hexdigest(), payload)
            self._files[key] = item
        if item[0] != descriptor["sha256"]:
            raise ValueError("Changed original-source alias metadata: " + str(path))
        return item[1]

    def json(self, descriptor):
        return json.loads(self.read(descriptor).decode("utf-8-sig"))


def verify_scientific_alias_fields(saved, original_parent, prepared, baseline):
    """Compare all original-anchor descriptor fields, without loading pixels."""
    source = saved["binding"]["source"]
    full = original_parent["full_extended_scientific"]
    fields = dict(native_scientific=prepared["full_native_scientific"],
        prepared_source=full["binding"]["prepared_source"],
        full_extended_scientific=full, baseline_extended_scientific=baseline,
        geometry_frame=prepared["geometry_frame"],
        rig_geometry=original_parent["binding"]["source"]["rig_geometry"])
    if prepared["task"]["source"]["sha256"] != source["source_sha256"]:
        raise ValueError("Original prepared source identifies another acquisition")
    if any(saved.get(key) != value for key, value in fields.items()):
        raise ValueError("Metadata alias changed an original scientific descriptor")
    expected = saved["metadata_alias_provenance"]["original_producer_descriptor_fields_sha256"]
    if _canonical(fields) != expected:
        raise ValueError("Original-anchor descriptor certificate disagrees")
    return expected


def verify_original_anchor_alias(receipt_asset, source, plan, run, *,
                                alias_inventory, tool_equivalence, verifier=None):
    verifier = verifier or AliasMetadataVerifier()
    saved = verifier.json(receipt_asset)
    if (saved.get("complete") is not True or saved.get("run") != run
            or saved.get("source_sha256") != source["source_sha256"]
            or saved["binding"]["source"] != source
            or saved["binding"]["job_sha256"] != plan["job_sha256"]
            or saved["binding"]["frozen_parent"] != plan["parent_chain"]["parent_seal"]):
        raise ValueError("Metadata alias has another exact source or consumer plan")
    proof = verifier.json(alias_inventory)
    row = next((row for row in proof["aliases"] if row["run"] == run
                and row["source_sha256"] == source["source_sha256"]), None)
    if row is None or row["alias_receipt"] != receipt_asset:
        raise ValueError("Metadata alias is not in the bound alias inventory")
    lineage = saved["metadata_alias_provenance"]
    if (lineage["original_parent_receipt"] != source["parent_receipt"]
            or lineage["original_prepared_receipt"] != row["original_prepared_receipt"]
            or saved["source_audit"].get("metadata_only_original_source_alias") is not True
            or saved["source_audit"].get("scientific_arrays_generated") is not False):
        raise ValueError("Alias producer lineage is incomplete")
    transport = verifier.json(tool_equivalence)
    if (transport.get("complete") is not True or transport.get("byte_exact") is not True
            or transport["original"] != lineage["tool"]
            or transport["portable"]["sha256"] != transport["original"]["sha256"]):
        raise ValueError("Alias tool transport lost exact producer equivalence")
    verifier.read(transport["portable"])
    parent = verifier.json(source["parent_receipt"])
    prepared = verifier.json(row["original_prepared_receipt"])
    baseline_parent = verifier.json(plan["runs"][run]["reference_parent_receipt"])
    result = verify_scientific_alias_fields(saved, parent, prepared,
                                           baseline_parent["baseline_extended_scientific"])
    if result != row["scientific_fields_sha256"]:
        raise ValueError("Alias inventory descriptor checksum disagrees")
    return dict(complete=True, source_sha256=source["source_sha256"],
        scientific_fields_sha256=result, producer_proof_verified=True,
        scientific_arrays_read=False, old_artifacts_modified=False)
