"""Verify immutable, run-native FF_AC V5 physical geometry.

The old FF_AC rig facies and registered-context archives use one shared mask
and separable physical axes.  They are not an acceptable geometry source for
V5 materialization.  A V5-derived native manifest must instead bind an exact
source revision to a separately audited two-dimensional physical context.

This module only verifies a candidate context.  It does not transform source
images, labels, facies, or physical coordinates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from darsia.fluidflower.project import sha256_file


CONTRACT = "ffac-v5-run-native-geometry-v1"
COHORT_CONTRACT = "ffac-v5-run-native-geometry-52-v1"
IMAGE_GEOMETRY_FIELDS = (
    "schema", "dataset", "run", "kind", "frame_count", "canvas_wh",
    "analysis_image_shape_hw", "content_box_xywh", "full_analysis_resolution",
    "additional_spatial_resampling", "base_image_shared_between_candidates",
    "contours_stored_as_transparent_overlays", "phase_topology",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha(path: Path, expected: str, description: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"V5 geometry {description} changed: {path}")


def _source_frame_sha(manifest: dict[str, Any]) -> str:
    rows = manifest.get("frame_sources") or []
    if not rows or int(rows[0].get("frame_index", -1)) != 0:
        raise ValueError("V5 geometry source has no exact frame zero")
    value = rows[0].get("native_jpeg_sha256") or rows[0].get("frame_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("V5 geometry source frame zero lacks an image SHA")
    return value


def _verify_original_context_lineage(
    source: dict[str, Any], receipt: dict[str, Any],
    context_path: Path, context_sha: str,
) -> None:
    original = source.get("domain") or {}
    lineage = receipt.get("source_context_lineage") or {}
    if original:
        if (lineage.get("kind") != "same-revision-domain"
                or original.get("path") != str(context_path)
                or original.get("sha256") != context_sha):
            raise ValueError("V5 geometry source context differs from this native source")
        return
    if lineage.get("kind") != "same-run-frozen-sibling":
        raise ValueError("Historical V5 source lacks an independently bound facies sibling")
    sibling_path = Path(lineage.get("sibling_manifest_path") or "")
    sibling_sha = lineage.get("sibling_manifest_sha256")
    if not isinstance(sibling_sha, str) or len(sibling_sha) != 64:
        raise ValueError("Historical V5 facies sibling has no SHA")
    _sha(sibling_path, sibling_sha, "historical facies sibling")
    sibling = _read(sibling_path)
    if (sibling.get("run") != source.get("run")
            or sibling.get("analysis_image_shape_hw") != source.get("analysis_image_shape_hw")
            or sibling.get("content_box_xywh") != source.get("content_box_xywh")
            or (sibling.get("domain") or {}).get("path") != str(context_path)
            or (sibling.get("domain") or {}).get("sha256") != context_sha):
        raise ValueError("Historical V5 facies sibling is not source/geometry compatible")


def geometry_overrides(source_manifest_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Verify one frozen receipt before RAW rendering and return native overrides.

    The aggregate 52-revision manifest selects ``row`` by *source manifest SHA*;
    a run name alone is never enough because historical revisions differ.
    """
    source_path = Path(source_manifest_path)
    source_sha = sha256_file(source_path)
    source = _read(source_path)
    if row.get("source_manifest_sha256") != source_sha:
        raise ValueError("V5 geometry row belongs to another source revision")
    receipt_path = Path(row.get("receipt_path") or "")
    receipt_sha = row.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
        raise ValueError("V5 geometry row has no receipt SHA")
    _sha(receipt_path, receipt_sha, "receipt")
    receipt = _read(receipt_path)
    if (receipt.get("contract") != CONTRACT
            or receipt.get("source_manifest_sha256") != source_sha
            or receipt.get("run") != str(source.get("run") or "").casefold()
            or receipt.get("native_frame0_sha256") != _source_frame_sha(source)
            or receipt.get("native_shape_hw") != source.get("analysis_image_shape_hw")
            or receipt.get("reference_run") != "ac28"):
        raise ValueError("V5 geometry receipt has another run/frame source")
    for prefix, expected_key in (
        ("source_frame0", "native_frame0_sha256"),
        ("reference_ac28_frame0", "reference_ac28_frame0_sha256"),
    ):
        expected = receipt.get(expected_key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"V5 geometry lacks {prefix} SHA")
        _sha(Path(receipt.get(prefix + "_path") or ""), expected, prefix)
    h = np.asarray(receipt.get("H_ac28_to_native"), dtype=np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()
            or not math.isclose(float(h[2, 2]), 1.0, abs_tol=1e-6)
            or abs(float(np.linalg.det(h))) <= 1e-9):
        raise ValueError("V5 geometry homography is not finite and invertible")
    if (not isinstance(receipt.get("heldout_sand_p95_px"), (int, float))
            or not math.isfinite(float(receipt["heldout_sand_p95_px"]))
            or int(receipt.get("heldout_sand_n") or 0) <= 0):
        raise ValueError("V5 geometry lacks independent sand registration QA")
    if not all(receipt.get(key) is True for key in (
        "heldout_sand_qa_pass", "geometry_acceptance_pass",
        "absolute_facies_boundary_pass", "domain_frame_exclusion_pass",
        "physical_grid_2d_exact", "legacy_1d_axes_nominal_approximation_only",
    )):
        raise ValueError("V5 geometry receipt has not passed source-only gates")
    path = Path(receipt.get("context_npz_path") or "")
    digest = receipt.get("context_npz_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("V5 geometry row lacks derived context SHA")
    _sha(path, digest, "2D context")
    source_context_path = Path(receipt.get("source_context_npz_path") or "")
    source_context_sha = receipt.get("source_context_npz_sha256")
    if not isinstance(source_context_sha, str) or len(source_context_sha) != 64:
        raise ValueError("V5 geometry lacks original context lineage")
    _sha(source_context_path, source_context_sha, "original context")
    _verify_original_context_lineage(source, receipt,
                                     source_context_path, source_context_sha)
    injectors = receipt.get("injection_points_native")
    if not isinstance(injectors, list) or len(injectors) != len(source.get("injection_points") or []):
        raise ValueError("V5 geometry lacks projected injection coordinates")
    for before, after in zip(source.get("injection_points") or [], injectors, strict=True):
        for key in ("x_m", "y_m", "active_windows_h", "label"):
            if before.get(key) != after.get(key):
                raise ValueError("V5 geometry changed physical injection protocol")
    with np.load(path, allow_pickle=False) as archive:
        required = {"domain", "facies", "permeability", "x_m_native",
                    "y_m_native", "pixel_area_m2", "x_m", "y_m"}
        if not required.issubset(archive.files):
            raise ValueError("V5 geometry context lacks full 2D physical fields")
        shape = tuple(source["analysis_image_shape_hw"])
        if any(archive[key].shape != shape for key in required - {"x_m", "y_m"}):
            raise ValueError("V5 geometry context is not on the native source grid")
        if archive["x_m"].shape != (shape[1],) or archive["y_m"].shape != (shape[0],):
            raise ValueError("V5 geometry nominal 1D axes are not on the source grid")
    return dict(
        domain=dict(path=str(path), sha256=digest, excluded_facies=[0]),
        injection_points=injectors,
        v5_run_geometry=dict(contract=CONTRACT,
                             original_source_manifest_sha256=source_sha,
                             receipt_path=str(receipt_path),
                             receipt_sha256=receipt_sha),
        original_domain=source.get("domain"),
        original_injection_points=source.get("injection_points"),
    )


def verify_run_geometry(
    source_manifest_path: Path,
    materialized_manifest_path: Path,
    *,
    expected_shape_hw: tuple[int, int] = (2979, 5472),
    require_acceptance: bool = True,
) -> dict[str, Any]:
    """Return the SHA-bound 2D context, or fail before any regional fit.

    A receipt may exist during diagnostic work with acceptance flags false.
    The default is deliberately strict: no training/domain binding can use it.
    Native image/annotation coordinates remain unchanged while the domain,
    facies, injection pixels and 2D metric coordinates are new derived data.
    """
    source_path = Path(source_manifest_path)
    target_path = Path(materialized_manifest_path)
    source_sha = sha256_file(source_path)
    source, target = _read(source_path), _read(target_path)
    for field in IMAGE_GEOMETRY_FIELDS:
        if source.get(field) != target.get(field):
            raise ValueError(f"V5 materialization changed native image geometry: {field}")
    binding = target.get("v5_run_geometry") or {}
    if (binding.get("contract") != CONTRACT
            or binding.get("original_source_manifest_sha256") != source_sha):
        raise ValueError("V5 materialized source lacks exact-revision 2D geometry binding")
    receipt_path = Path(binding.get("receipt_path") or "")
    receipt_sha = binding.get("receipt_sha256")
    if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
        raise ValueError("V5 geometry receipt has no SHA")
    _sha(receipt_path, receipt_sha, "receipt")
    receipt = _read(receipt_path)
    if (receipt.get("contract") != CONTRACT
            or receipt.get("run") != str(source.get("run") or "").casefold()
            or receipt.get("source_manifest_sha256") != source_sha
            or receipt.get("native_shape_hw") != list(expected_shape_hw)
            or receipt.get("reference_run") != "ac28"):
        raise ValueError("V5 geometry receipt belongs to another run/source/grid")
    if receipt.get("native_frame0_sha256") != _source_frame_sha(source):
        raise ValueError("V5 geometry did not use this source revision's frame zero")
    original_injectors = source.get("injection_points") or []
    derived_injectors = target.get("injection_points") or []
    if (receipt.get("injection_points_native") != derived_injectors
            or len(original_injectors) != len(derived_injectors)):
        raise ValueError("V5 injection pixels are not source-bound to the run-native geometry")
    for before, after in zip(original_injectors, derived_injectors, strict=True):
        for key in ("x_m", "y_m", "active_windows_h", "label"):
            if before.get(key) != after.get(key):
                raise ValueError("V5 geometry changed physical injection protocol")
        xy = after.get("analysis_xy_px")
        if (not isinstance(xy, list) or len(xy) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in xy)):
            raise ValueError("V5 geometry lacks native injection pixel coordinates")
    for prefix, expected_key in (
        ("source_frame0", "native_frame0_sha256"),
        ("reference_ac28_frame0", "reference_ac28_frame0_sha256"),
    ):
        expected = receipt.get(expected_key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"V5 geometry lacks {prefix} SHA")
        _sha(Path(receipt.get(prefix + "_path") or ""), expected, prefix)
    h = np.asarray(receipt.get("H_ac28_to_native"), dtype=np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all()
            or not math.isclose(float(h[2, 2]), 1.0, abs_tol=1e-6)
            or abs(float(np.linalg.det(h))) <= 1e-9):
        raise ValueError("V5 geometry homography is not finite and invertible")
    if (not isinstance(receipt.get("heldout_sand_p95_px"), (int, float))
            or not math.isfinite(float(receipt["heldout_sand_p95_px"]))
            or receipt["heldout_sand_p95_px"] < 0
            or int(receipt.get("heldout_sand_n") or 0) <= 0):
        raise ValueError("V5 geometry has no independent sand registration result")
    if require_acceptance and not all(
        receipt.get(key) is True for key in (
            "heldout_sand_qa_pass", "geometry_acceptance_pass",
            "absolute_facies_boundary_pass", "domain_frame_exclusion_pass",
            "physical_grid_2d_exact", "legacy_1d_axes_nominal_approximation_only",
        )
    ):
        raise ValueError("V5 run-native geometry is diagnostic, not training eligible")
    context_path = Path(receipt.get("context_npz_path") or "")
    context_sha = receipt.get("context_npz_sha256")
    if (not isinstance(context_sha, str) or len(context_sha) != 64
            or target.get("domain", {}).get("path") != str(context_path)
            or target.get("domain", {}).get("sha256") != context_sha):
        raise ValueError("V5 manifest domain does not bind the accepted 2D context")
    _sha(context_path, context_sha, "2D context")
    source_context_path = Path(receipt.get("source_context_npz_path") or "")
    source_context_sha = receipt.get("source_context_npz_sha256")
    if not isinstance(source_context_sha, str) or len(source_context_sha) != 64:
        raise ValueError("V5 geometry lacks source-context lineage")
    _sha(source_context_path, source_context_sha, "original context")
    _verify_original_context_lineage(source, receipt,
                                     source_context_path, source_context_sha)
    with np.load(context_path, allow_pickle=False) as archive:
        required = {"domain", "facies", "permeability", "x_m_native",
                    "y_m_native", "pixel_area_m2", "x_m", "y_m"}
        if not required.issubset(archive.files):
            raise ValueError("V5 geometry context lacks full 2D physical fields")
        if any(archive[key].shape != expected_shape_hw for key in required - {"x_m", "y_m"}):
            raise ValueError("V5 geometry context is not on the full native grid")
        if archive["x_m"].shape != (expected_shape_hw[1],) or archive["y_m"].shape != (expected_shape_hw[0],):
            raise ValueError("V5 nominal 1D spacing axes do not match the native grid")
        domain = archive["domain"]
        facies = archive["facies"]
        if domain.dtype != np.dtype("bool") or not np.issubdtype(facies.dtype, np.integer):
            raise ValueError("V5 geometry domain/facies have invalid types")
        if np.any(domain & (facies == 0)):
            raise ValueError("V5 geometry calls excluded rig pixels porous")
        for key in ("x_m_native", "y_m_native", "pixel_area_m2", "permeability"):
            field = archive[key]
            if not np.isfinite(field[domain]).all():
                raise ValueError(f"V5 geometry {key} is nonfinite in the porous domain")
        if np.any(archive["pixel_area_m2"][domain] <= 0):
            raise ValueError("V5 geometry has nonpositive physical pixel area")
    return dict(
        contract=CONTRACT, run=receipt["run"],
        source_manifest_sha256=source_sha,
        materialized_manifest_sha256=sha256_file(target_path),
        receipt_path=str(receipt_path), receipt_sha256=receipt_sha,
        context_npz_path=str(context_path), context_npz_sha256=context_sha,
        native_image_geometry_unchanged=True,
        physical_context_replaced=True,
        physical_coordinates_2d=True,
    )


__all__ = ["CONTRACT", "IMAGE_GEOMETRY_FIELDS", "verify_run_geometry"]
