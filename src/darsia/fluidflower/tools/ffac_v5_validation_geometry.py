"""Source-bound, validation-only FF_AC V5 run-native geometry.

This contract is deliberately separate from the stricter scientific geometry
contract.  The visible absolute facies boundary test failed, so V4 optical
photometry may use only a fixed source-derived safe interior.  The full porous
domain remains available to preserve native annotation coordinates; the safe
mask is never substituted for that domain when extracting labelled pixels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.ffac_v5_geometry_binding import (
    IMAGE_GEOMETRY_FIELDS, _source_frame_sha, _verify_original_context_lineage,
)


CONTRACT = "ffac-v5-run-native-geometry-validation-only-v1"
COHORT_CONTRACT = "ffac-v5-run-native-geometry-validation-only-52-v1"
REFERENCE_CONTRACT = "ffac-v5-physical-facies-safe-interior-provisional-v1"
SAFE_MASK_KEY = "facies_safe_interior"
SHAPE = (2979, 5472)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _checked(path: Path, digest: str, description: str) -> None:
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"V4 geometry {description} lacks SHA")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"V4 geometry {description} changed: {path}")


def _receipt(source_path: Path, row: dict[str, Any], *,
             expected_shape_hw: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_sha = sha256_file(source_path)
    source = _read(source_path)
    if row.get("source_manifest_sha256") != source_sha:
        raise ValueError("V4 geometry row belongs to another source revision")
    receipt_path = Path(row.get("receipt_path") or "")
    _checked(receipt_path, row.get("receipt_sha256"), "receipt")
    receipt = _read(receipt_path)
    if (receipt.get("contract") != CONTRACT
            or receipt.get("source_manifest_sha256") != source_sha
            or receipt.get("run") != str(source.get("run") or "").casefold()
            or receipt.get("reference_run") != "ac28"
            or receipt.get("native_shape_hw") != list(expected_shape_hw)
            or receipt.get("native_frame0_sha256") != _source_frame_sha(source)):
        raise ValueError("V4 geometry receipt has another source/run/native grid")
    if (receipt.get("validation_only") is not True
            or receipt.get("mass_eligible") is not False
            or receipt.get("absolute_boundary_qa_pass") is not False
            or receipt.get("safe_interior_mask_key") != SAFE_MASK_KEY
            or receipt.get("safe_interior_projection_pass") is not True
            or receipt.get("heldout_sand_qa_pass") is not True
            or receipt.get("domain_frame_exclusion_pass") is not True):
        raise ValueError("V4 source-only geometry approval is absent or misrepresented")
    if (not isinstance(receipt.get("heldout_sand_p95_px"), (float, int))
            or not math.isfinite(float(receipt["heldout_sand_p95_px"]))
            or int(receipt.get("heldout_sand_n") or 0) <= 0):
        raise ValueError("V4 geometry lacks independent sand registration")
    h = np.asarray(receipt.get("H_ac28_to_native"), np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()
            or abs(float(np.linalg.det(h))) <= 1e-9
            or not math.isclose(float(h[2, 2]), 1, abs_tol=1e-6)):
        raise ValueError("V4 geometry homography is invalid")
    for prefix, key in (("source_frame0", "native_frame0_sha256"),
                        ("reference_ac28_frame0", "reference_ac28_frame0_sha256")):
        _checked(Path(receipt.get(prefix + "_path") or ""), receipt.get(key), prefix)
    reference_path = Path(receipt.get("physical_facies_ref_receipt_path") or "")
    _checked(reference_path, receipt.get("physical_facies_ref_receipt_sha256"),
             "physical-facies safe-interior reference receipt")
    reference = _read(reference_path)
    if (reference.get("contract") != REFERENCE_CONTRACT
            or reference.get("absolute_facies_boundary_pass") is not False
            or reference.get("scientific_mass_eligible") is not False
            or reference.get("reference_run") != "ac28"
            or reference.get("uncertainty_exclusion_px") != 32):
        raise ValueError("V4 safe-interior reference does not preserve failed boundary QA")
    _checked(Path(reference.get("asset_path") or ""),
             reference.get("asset_sha256"), "physical safe-interior asset")
    context_path = Path(receipt.get("context_npz_path") or "")
    context_sha = receipt.get("context_npz_sha256")
    _checked(context_path, context_sha, "run-native context")
    source_context = Path(receipt.get("source_context_npz_path") or "")
    source_context_sha = receipt.get("source_context_npz_sha256")
    _checked(source_context, source_context_sha, "original source context")
    _verify_original_context_lineage(source, receipt, source_context,
                                     source_context_sha)
    with np.load(context_path, allow_pickle=False) as archive:
        needed = {"domain", "facies", SAFE_MASK_KEY, "permeability",
                  "x_m_native", "y_m_native", "pixel_area_m2", "x_m", "y_m"}
        if not needed.issubset(archive.files):
            raise ValueError("V4 context lacks porous domain, safe interior or 2D fields")
        shape = tuple(expected_shape_hw)
        if any(archive[key].shape != shape for key in needed - {"x_m", "y_m"}):
            raise ValueError("V4 context is not on the native source grid")
        if archive["x_m"].shape != (shape[1],) or archive["y_m"].shape != (shape[0],):
            raise ValueError("V4 nominal spacing axes are not on the source grid")
        domain = archive["domain"]
        facies = archive["facies"]
        safe = archive[SAFE_MASK_KEY]
        if (domain.dtype != np.dtype("bool") or safe.dtype != np.dtype("bool")
                or not np.issubdtype(facies.dtype, np.integer)
                or np.any(safe & (~domain | (facies == 0)))
                or np.any(domain & (facies == 0))
                or not np.any(safe)):
            raise ValueError("V4 safe interior is not a nonempty porous-domain subset")
        for key in ("x_m_native", "y_m_native", "pixel_area_m2", "permeability"):
            if not np.isfinite(archive[key][domain]).all():
                raise ValueError(f"V4 context {key} is nonfinite in the porous domain")
        if np.any(archive["pixel_area_m2"][domain] <= 0):
            raise ValueError("V4 context has nonpositive physical pixel area")
    old_points = source.get("injection_points") or []
    new_points = receipt.get("injection_points_native")
    if not isinstance(new_points, list) or len(new_points) != len(old_points):
        raise ValueError("V4 geometry lacks source-bound injector projection")
    for before, after in zip(old_points, new_points, strict=True):
        if any(before.get(key) != after.get(key) for key in
               ("x_m", "y_m", "active_windows_h", "label")):
            raise ValueError("V4 geometry changed physical injection protocol")
        xy = after.get("analysis_xy_px")
        if not isinstance(xy, list) or len(xy) != 2 or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in xy):
            raise ValueError("V4 injector has no finite native image position")
    return source, receipt


def validation_geometry_overrides(source_manifest_path: Path,
                                  row: dict[str, Any], *,
                                  expected_shape_hw: tuple[int, int] = SHAPE) -> dict[str, Any]:
    """Preflight one exact revision, without approving physical facies/mass."""
    source_path = Path(source_manifest_path)
    source, receipt = _receipt(source_path, row,
                               expected_shape_hw=expected_shape_hw)
    marker = dict(contract=CONTRACT,
                  original_source_manifest_sha256=sha256_file(source_path),
                  receipt_path=str(Path(row["receipt_path"])),
                  receipt_sha256=row["receipt_sha256"],
                  validation_only=True, mass_eligible=False)
    return dict(domain=dict(path=receipt["context_npz_path"],
                            sha256=receipt["context_npz_sha256"],
                            excluded_facies=[0]),
                injection_points=receipt["injection_points_native"],
                v5_run_geometry=marker,
                original_domain=source.get("domain"),
                original_injection_points=source.get("injection_points"))


def verify_validation_geometry(source_manifest_path: Path,
                               materialized_manifest_path: Path, *,
                               expected_shape_hw: tuple[int, int] = SHAPE) -> dict[str, Any]:
    """Verify a derived V4 manifest and retain its validation-only status."""
    original = Path(source_manifest_path)
    target = Path(materialized_manifest_path)
    source, materialized = _read(original), _read(target)
    for field in IMAGE_GEOMETRY_FIELDS:
        if source.get(field) != materialized.get(field):
            raise ValueError(f"V4 materialization changed native image geometry: {field}")
    marker = materialized.get("v5_run_geometry") or {}
    row = dict(source_manifest_sha256=sha256_file(original),
               receipt_path=marker.get("receipt_path"),
               receipt_sha256=marker.get("receipt_sha256"))
    if (marker.get("contract") != CONTRACT
            or marker.get("original_source_manifest_sha256") != row["source_manifest_sha256"]
            or marker.get("validation_only") is not True
            or marker.get("mass_eligible") is not False):
        raise ValueError("V4 derived source lacks validation-only geometry marker")
    _, receipt = _receipt(original, row, expected_shape_hw=expected_shape_hw)
    if (materialized.get("domain") != dict(path=receipt["context_npz_path"],
                                            sha256=receipt["context_npz_sha256"],
                                            excluded_facies=[0])
            or materialized.get("injection_points") != receipt["injection_points_native"]):
        raise ValueError("V4 derived source changed its certified context/injectors")
    return dict(contract=CONTRACT, run=receipt["run"],
                source_manifest_sha256=row["source_manifest_sha256"],
                materialized_manifest_sha256=sha256_file(target),
                receipt_path=row["receipt_path"],
                receipt_sha256=row["receipt_sha256"],
                context_npz_path=receipt["context_npz_path"],
                context_npz_sha256=receipt["context_npz_sha256"],
                safe_interior_mask_key=SAFE_MASK_KEY,
                native_image_geometry_unchanged=True,
                source_labels_preserved=True,
                validation_only=True, mass_eligible=False,
                absolute_facies_boundary_pass=False)


__all__ = ["CONTRACT", "COHORT_CONTRACT", "SAFE_MASK_KEY",
           "validation_geometry_overrides", "verify_validation_geometry"]
