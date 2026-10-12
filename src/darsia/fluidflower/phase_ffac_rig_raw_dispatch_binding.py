"""Verify frozen source-code aliases before reusing a prepared RAW identity.

A detached snapshot has the same numerical bytes at a new path. The prepared
scientific receipt remains authoritative; only code paths with equal verified
SHAs may be aliased, and every non-path identity field must match exactly.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .project import sha256_file

CONTRACT = "ffac-rig-prepared-frozen-code-path-alias-v1"


def _normalised(value):
    if isinstance(value,list):
        return [_normalised(item) for item in value]
    if isinstance(value,dict):
        return {key:_normalised(item) for key,item in value.items()
                if not (key=="path" and "sha256" in value)}
    return value


def _code_assets(value):
    if isinstance(value,list):
        for item in value: yield from _code_assets(item)
    elif isinstance(value,dict):
        if "path" in value and "sha256" in value:
            yield dict(path=value["path"],sha256=value["sha256"])
        for item in value.values(): yield from _code_assets(item)


def bind_prepared_source_processor(processor, producer_audit, *, scientific_receipt):
    """Return explicit code aliases and bind the verified prepared identity.

    ``scientific_receipt`` is the exact path/SHA asset already bound by the
    immutable source inventory. No native/source/calibration path is aliased.
    This does not alter prepared bytes or weaken the ring's identity equality.
    """
    path=Path(scientific_receipt["path"])
    if sha256_file(path)!=scientific_receipt["sha256"]:
        raise ValueError("Prepared scientific receipt changed before frozen-code alias binding")
    receipt=json.loads(path.read_text(encoding="utf-8-sig"))
    if receipt.get("producer_audit")!=producer_audit:
        raise ValueError("RAW producer audit does not belong to the bound scientific receipt")
    expected=producer_audit.get("processor")
    local=getattr(processor,"_ffac_unaliased_numerical_identity",processor.identity)
    if not isinstance(expected,dict) or not isinstance(local,dict):
        raise ValueError("Frozen code aliases require exact source processor identities")
    proof_key="explicit_frozen_variant_dispatch"
    original_proof=expected.get(proof_key)
    local_proof=local.get(proof_key)
    expected_base={key:value for key,value in expected.items() if key!=proof_key}
    local_base={key:value for key,value in local.items() if key!=proof_key}
    if expected_base!=local_base or _normalised(original_proof)!=_normalised(local_proof):
        raise ValueError("Prepared and local frozen RAW identities differ beyond code paths")
    if (original_proof is None)!=(local_proof is None):
        raise ValueError("Frozen RAW variant dispatch is missing")
    expected_assets=list(_code_assets(original_proof))
    local_assets=list(_code_assets(local_proof))
    if len(expected_assets)!=len(local_assets):
        raise ValueError("Frozen code inventory differs")
    aliases=[]
    for previous,current in zip(expected_assets,local_assets):
        if previous["sha256"]!=current["sha256"] or sha256_file(current["path"])!=previous["sha256"]:
            raise ValueError("A source-code alias has different numerical bytes")
        if previous["path"]!=current["path"]:
            aliases.append(dict(original_path=previous["path"],verified_local_path=current["path"],
                                sha256=current["sha256"]))
    binding=dict(contract=CONTRACT,scientific_receipt=deepcopy(scientific_receipt),
        verified_aliases=aliases,non_path_identity_exact=True,all_code_sha_verified=True,
        numerical_code_changed=False,source_or_calibration_aliases=False)
    if not hasattr(processor,"_ffac_unaliased_numerical_identity"):
        processor._ffac_unaliased_numerical_identity=deepcopy(local)
    processor.identity=deepcopy(expected)
    processor._ffac_verified_code_path_aliases=binding
    return binding


__all__=["bind_prepared_source_processor","CONTRACT"]
