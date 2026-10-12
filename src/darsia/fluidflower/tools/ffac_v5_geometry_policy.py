"""Dispatch immutable V5 source geometry without weakening either contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.ffac_v5_geometry_binding import (
    COHORT_CONTRACT as SCIENTIFIC_COHORT,
    CONTRACT as SCIENTIFIC_CONTRACT,
    geometry_overrides as scientific_overrides,
    verify_run_geometry,
)
from darsia.fluidflower.tools.ffac_v5_validation_geometry import (
    COHORT_CONTRACT as VALIDATION_COHORT,
    CONTRACT as VALIDATION_CONTRACT,
    validation_geometry_overrides,
    verify_validation_geometry,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def accepted_geometry_overrides(source_manifest_path: Path,
                                row: dict[str, Any], *,
                                expected_shape_hw: tuple[int, int] = (2979, 5472)) -> dict[str, Any]:
    """Verify exactly one receipt selected by its source-manifest SHA."""
    receipt_path = Path(row.get("receipt_path") or "")
    if (not receipt_path.is_file() or sha256_file(receipt_path) != row.get("receipt_sha256")):
        raise ValueError("V5 run-native geometry receipt changed")
    contract = _read(receipt_path).get("contract")
    if contract == SCIENTIFIC_CONTRACT:
        return scientific_overrides(source_manifest_path, row)
    if contract == VALIDATION_CONTRACT:
        return validation_geometry_overrides(source_manifest_path, row,
                                             expected_shape_hw=expected_shape_hw)
    raise ValueError("Unknown V5 run-native geometry receipt contract")


def verify_accepted_geometry(source_manifest_path: Path,
                             materialized_manifest_path: Path, *,
                             expected_shape_hw: tuple[int, int] = (2979, 5472)) -> dict[str, Any]:
    marker = _read(Path(materialized_manifest_path)).get("v5_run_geometry") or {}
    contract = marker.get("contract")
    if contract == SCIENTIFIC_CONTRACT:
        result = verify_run_geometry(source_manifest_path, materialized_manifest_path,
                                     expected_shape_hw=expected_shape_hw)
        # A geometric certificate alone cannot scientifically accept a mass
        # model, even when all physical boundary checks pass.
        result.update(validation_only=False, mass_eligible=False,
                      physical_geometry_eligible=True)
        return result
    if contract == VALIDATION_CONTRACT:
        return verify_validation_geometry(source_manifest_path,
                                          materialized_manifest_path,
                                          expected_shape_hw=expected_shape_hw)
    raise ValueError("Unknown V5 materialized geometry contract")


def cohort_contract_from_rows(rows: list[dict[str, Any]]) -> str:
    """Require one homogeneous scientific or validation-only 52-row cohort."""
    contracts = set()
    for row in rows:
        path = Path(row.get("receipt_path") or "")
        if not path.is_file() or sha256_file(path) != row.get("receipt_sha256"):
            raise ValueError("V5 cohort row receipt changed")
        contracts.add(_read(path).get("contract"))
    if contracts == {SCIENTIFIC_CONTRACT}:
        return SCIENTIFIC_COHORT
    if contracts == {VALIDATION_CONTRACT}:
        return VALIDATION_COHORT
    raise ValueError("V5 cohort mixes or lacks geometry acceptance contracts")


__all__ = ["accepted_geometry_overrides", "verify_accepted_geometry",
           "cohort_contract_from_rows"]
