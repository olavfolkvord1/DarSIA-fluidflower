"""Optional FF_AC AC28-anchored dye-chroma alignment on camera-linear RAW.

The source-only fit is performed separately from this module on predeclared
run, time and spatial partitions. This transform changes log R and log B in
the porous sand, while log G and pixels outside the mapped facies are copied
bit-for-bit. It is not a concentration or mass calibration. No phase labels,
model predictions or GPU are used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CONTRACT = "ffac-ac28-chroma-curve-transform-bundle-v1"
MAP_RUNS = frozenset(("ac20", "ac21", "ac24", "ac27", "ac51"))
DIAGNOSTIC_ONLY = frozenset(("ac22",))
KNOTS = np.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=np.float64)


def load_chroma_bundle(path: str | Path) -> dict:
    """Load a source-bound transform bundle; caller also verifies its SHA."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("contract") != CONTRACT:
        raise ValueError("Unknown FF_AC chemistry transform contract")
    if bundle.get("anchor_run") != "ac28":
        raise ValueError("Chemistry transform has another light/recipe anchor")
    if set(bundle.get("enabled_runs", ())) != MAP_RUNS:
        raise ValueError("Chemistry transform run set is incomplete or broadened")
    if set(bundle.get("diagnostic_only_runs", ())) != DIAGNOSTIC_ONLY:
        raise ValueError("Chemistry diagnostic run set changed")
    if bundle.get("luminance_contract") != "logG-bit-exact":
        raise ValueError("Chemistry mapping could alter log G")
    curves = bundle.get("curves")
    if not isinstance(curves, dict) or set(curves) != MAP_RUNS | {"ac28"}:
        raise ValueError("Chemistry bundle has missing or extra production curves")
    for run, curve in curves.items():
        v = np.asarray(curve["vector"], np.float64)
        k = np.asarray(curve["knots"], np.float64)
        b = curve["blue_by_facies"]
        if (v.shape != (2,) or k.shape != (5, 2)
                or not np.isfinite(v).all() or not np.isfinite(k).all()
                or float(np.dot(v, v)) < .05**2
                or not isinstance(b, dict) or not b):
            raise ValueError(f"Malformed chemistry curve for {run}")
        for facies, colour in b.items():
            if int(facies) not in range(1, 9) or np.shape(colour) != (2,):
                raise ValueError(f"Malformed chemistry facies blue for {run}")
        projection = k @ v / float(np.dot(v, v))
        if not np.all(np.diff(projection) > 0):
            raise ValueError(f"Non-monotone chemistry curve for {run}")
    return bundle


def _segment_jacobian_limits(source: dict, reference: dict) -> tuple[float, float]:
    vector = np.asarray(source["vector"], np.float64)
    sk = np.asarray(source["knots"], np.float64)
    rk = np.asarray(reference["knots"], np.float64)
    singular = []
    determinants = []
    for segment in range(4):
        delta = (rk[segment + 1] - rk[segment]
                 - sk[segment + 1] + sk[segment]) / .25
        J = np.eye(2) + np.outer(delta, vector) / float(np.dot(vector, vector))
        singular.append(float(np.linalg.svd(J, compute_uv=False)[0]))
        determinants.append(float(np.linalg.det(J)))
    return max(singular), min(determinants)


def apply_chroma_alignment(
    corrected_log_rgb: np.ndarray,
    source_facies: np.ndarray,
    run: str,
    bundle: dict,
    *,
    tile_rows: int = 384,
) -> tuple[np.ndarray, dict]:
    """Map selected chemistry-run sand chroma, preserving log G exactly.

    `corrected_log_rgb` is the EA-v2 photometric output, camera-linear log RGB
    with shape (3,H,W). `source_facies` is the *same* source-bound native or
    1/8 facies grid used by photometry. Charts/rig and unsupported facies stay
    unchanged. The function returns a new float32 array and a compact receipt.
    """
    if bundle.get("contract") != CONTRACT:
        raise ValueError("Invalid chemistry mapping bundle")
    arr = np.asarray(corrected_log_rgb)
    facies = np.asarray(source_facies)
    if (arr.dtype != np.float32 or arr.ndim != 3 or arr.shape[0] != 3
            or facies.shape != arr.shape[1:] or tile_rows <= 0):
        raise ValueError("Chemistry mapping requires RGB float32 and matching facies")
    if arr.shape[1:] not in ((372, 684), (2979, 5472)):
        raise ValueError("Chemistry mapping requires an exact source grid")
    run = str(run).lower()
    if run not in MAP_RUNS:
        return arr.copy(), dict(
            contract="ffac-ac28-chroma-transform-application-v1", run=run,
            status="identity", mapped_pixels=0, preserved_G=True,
            reason=("diagnostic-only" if run in DIAGNOSTIC_ONLY else
                    "not-a-predeclared-chemistry-mapping-run"))
    source = bundle["curves"][run]
    reference = bundle["curves"]["ac28"]
    max_singular, min_det = _segment_jacobian_limits(source, reference)
    if max_singular > 2.0 or min_det <= 0.0:
        raise ValueError("Chemistry mapping would over-amplify or reverse signal")
    source_blue = source["blue_by_facies"]
    reference_blue = reference["blue_by_facies"]
    common = sorted(set(source_blue) & set(reference_blue), key=int)
    if not common:
        raise ValueError("Chemistry mapping has no source-bound common facies")
    v = np.asarray(source["vector"], np.float64)
    denominator = float(np.dot(v, v))
    sk = np.asarray(source["knots"], np.float64)
    rk = np.asarray(reference["knots"], np.float64)
    out = arr.copy()
    counts = {f: 0 for f in common}
    for y0 in range(0, arr.shape[1], tile_rows):
        y1 = min(y0 + tile_rows, arr.shape[1])
        tile = arr[:, y0:y1]
        ftile = facies[y0:y1]
        for key in common:
            mask = (ftile == int(key)) & np.all(np.isfinite(tile), axis=0)
            if not np.any(mask):
                continue
            r = tile[0][mask].astype(np.float64)
            g = tile[1][mask].astype(np.float64)
            b = tile[2][mask].astype(np.float64)
            c0, c1 = r - g, b - g
            own_blue = np.asarray(source_blue[key], np.float64)
            ref_blue = np.asarray(reference_blue[key], np.float64)
            s = np.clip(((c0 - own_blue[0]) * v[0]
                        + (c1 - own_blue[1]) * v[1]) / denominator, 0, 1)
            mapped0 = (c0 + ref_blue[0] - own_blue[0]
                       + np.interp(s, KNOTS, rk[:, 0])
                       - np.interp(s, KNOTS, sk[:, 0]))
            mapped1 = (c1 + ref_blue[1] - own_blue[1]
                       + np.interp(s, KNOTS, rk[:, 1])
                       - np.interp(s, KNOTS, sk[:, 1]))
            out[0, y0:y1][mask] = (g + mapped0).astype(np.float32)
            out[2, y0:y1][mask] = (g + mapped1).astype(np.float32)
            counts[key] += int(mask.sum())
    if not np.array_equal(out[1], arr[1], equal_nan=True):
        raise AssertionError("Chemistry mapping changed log G")
    return out, dict(
        contract="ffac-ac28-chroma-transform-application-v1", run=run,
        status="mapped", mapped_pixels=sum(counts.values()),
        mapped_pixels_by_facies=counts,
        max_jacobian_singular_value=max_singular,
        minimum_jacobian_determinant=min_det,
        preserved_G=True, source_shape=list(arr.shape),
        identity_outside_mapped_facies=True)
