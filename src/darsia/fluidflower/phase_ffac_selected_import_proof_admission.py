"""Verify an append-only metadata enrichment of an existing import proof.

The accepted source plan and every original producer receipt remain unchanged.
No scientific assets, model predictions, annotations or sealed scorer are read.
"""
from __future__ import annotations


def require_import_proof(plan, source_descriptor, descriptor, reader):
    original_descriptor = plan['source_import_proof']
    if descriptor is None or descriptor == original_descriptor:
        return original_descriptor
    enriched = reader.record(descriptor)
    original = reader.record(original_descriptor)
    if (enriched.get('contract') != 'ffac-selected-byte-exact-imported-source-producer-equivalence-v1'
            or enriched.get('complete') is not True
            or enriched.get('source_plan') != source_descriptor
            or enriched.get('supersedes_metadata_interface') != original_descriptor
            or enriched.get('original_minimal_copy_proof_preserved') is not True
            or enriched.get('byte_exact_receipt_copies') is not True
            or enriched.get('producer_code_and_source_binding_preserved') is not True
            or enriched.get('arrays_copied_or_recomputed') is not False
            or enriched.get('scientific_arrays_read') is not False
            or enriched.get('scientific_arrays_generated') is not False
            or enriched.get('old_artifacts_modified') is not False
            or original.get('complete') is not True):
        raise ValueError('Import metadata override lacks the exact original proof/source-plan lineage')
    declared = {(run, digest): item for run, items in plan['source_imports'].items()
                for digest, item in items.items()}
    prior = {}
    for row in original['imports']:
        key = (row['run'], row['source_sha256'])
        if key in prior or key not in declared or row['staged_receipt'] != declared[key]:
            raise ValueError('Original import proof duplicates or changes a declared source')
        prior[key] = row
    checked = set()
    for row in enriched['imports']:
        key = (row['run'], row['source_sha256'])
        if key in checked or key not in prior:
            raise ValueError('Enriched import proof duplicates or adds a source')
        old = prior[key]
        if (set(row) != set(old) | {'producer_binding'}
                or any(row[name] != value for name, value in old.items())
                or row['producer_receipt']['sha256'] != row['staged_receipt']['sha256']):
            raise ValueError('Enrichment changed an original producer/staged descriptor or receipt field')
        # These are small source receipts, never their dense scientific assets.
        producer = reader.record(row['producer_receipt'])
        staged = reader.record(row['staged_receipt'])
        if producer != staged or row['producer_binding'] != producer['binding']:
            raise ValueError('Derived import binding is not the literal unchanged source producer')
        checked.add(key)
    if (checked != set(prior) or checked != set(declared)
            or enriched.get('imported_count') != len(checked)):
        raise ValueError('Metadata override omits an existing source import')
    return descriptor


__all__ = ['require_import_proof']
