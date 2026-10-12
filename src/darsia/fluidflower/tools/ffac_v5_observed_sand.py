"""Source-observed FF_AC photometry on an expanded reference grid.

This module contains numerical operations only.  Callers must bind every RAW
grid, homography, chart and per-run static-sand mask to exact source receipts.
No phase annotation, model output, or physical facies ID enters the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import ffac_v5_photometry as v5


@dataclass(frozen=True)
class ReferenceGrid:
    """1/8-grid cell coordinates in the original AC28 native frame."""

    shape_hw: tuple[int, int] = (382, 710)
    origin_cell_xy: tuple[int, int] = (-16, 0)
    original_shape_hw: tuple[int, int] = v5.GRID

    def __post_init__(self) -> None:
        if self.shape_hw[0] <= 0 or self.shape_hw[1] <= 0:
            raise ValueError("Grid shape must be positive")
        if self.original_shape_hw[0] <= 0 or self.original_shape_hw[1] <= 0:
            raise ValueError("Original grid shape must be positive")

    def normalized_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[: self.shape_hw[0], : self.shape_hw[1]]
        x = (xx + self.origin_cell_xy[0] - self.original_shape_hw[1] / 2) / (
            self.original_shape_hw[1] / 2
        )
        y = (yy + self.origin_cell_xy[1] - self.original_shape_hw[0] / 2) / (
            self.original_shape_hw[0] / 2
        )
        return x.astype(np.float64), y.astype(np.float64)

    def basis(self, *, include_constant: bool = False) -> np.ndarray:
        x, y = self.normalized_coordinates()
        first = 0 if include_constant else 1
        return np.stack(
            [x**ax * y ** (degree - ax) for degree in range(first, 5)
             for ax in range(degree + 1)]
        ).astype(np.float64)


def require_green_donor_only_receipt(receipt: dict) -> None:
    """Gate optical fitting on its narrow photo-sand scope, never label QA."""
    if (receipt.get("photo_sand_qa_scope") != "green_donor_mask_only"
            or receipt.get("green_donor_only_optical_fit_eligible") is not True
            or receipt.get("photo_sand_qa_pass") is not True):
        raise ValueError("Source-photo green donor mask is not fit-eligible")


def _rgb(array: np.ndarray, grid: ReferenceGrid) -> np.ndarray:
    raw = np.asarray(array)
    if raw.dtype != np.float32:
        raise TypeError("V4-R3 requires float32 log-linear RAW; uint8/JPEG is invalid")
    return v5._log_rgb(raw, grid.shape_hw)


def _fit_field(
    green: np.ndarray, fac: np.ndarray, keep: np.ndarray, grid: ReferenceGrid
) -> tuple[np.ndarray, dict[int, float], dict[int, int]]:
    basis = grid.basis().reshape(14, -1)
    idx = np.flatnonzero((keep & (fac >= 1) & np.isfinite(green)).ravel())
    if len(idx) < 1000:
        raise ValueError(f"Too few source-certified sand cells: {len(idx)}")
    ff = fac.ravel()[idx]
    # The input has been restricted to 0/1 source-observed sand.  Keep the
    # optical intercept explicitly single-class; physical facies are separate.
    classes = [1]
    X = np.column_stack([basis[:, idx].T] + [(ff == f).astype(float) for f in classes])
    z = green.ravel()[idx].astype(float)
    weight = np.ones(len(z))
    for _ in range(6):
        sw = np.sqrt(weight)
        coef = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        weight = np.minimum(1, .05 / np.maximum(np.abs(z - X @ coef), 1e-9))
    return coef[:14], {f: float(coef[14+i]) for i, f in enumerate(classes)}, {
        f: int(np.count_nonzero(ff == f)) for f in classes
    }


def fit_baseline_grid(
    *, run: str, logB: np.ndarray, facies: np.ndarray,
    chart_xy: np.ndarray, chart_log_rgb: np.ndarray,
    ref_chart_log_rgb: np.ndarray, chart_usable: np.ndarray,
    matrix: np.ndarray, rig_consensus: np.ndarray,
    grid: ReferenceGrid, flat: np.ndarray | None = None,
) -> v5.Baseline:
    """Fit a V5 baseline using this run's independently certified sand cells."""
    logB = _rgb(logB, grid)
    cons = _rgb(rig_consensus, grid)
    fac = np.asarray(facies, np.int16)
    xy = np.asarray(chart_xy, np.float64)
    M = np.asarray(matrix, np.float64)
    if fac.shape != grid.shape_hw or xy.shape != (24, 2) or M.shape != (3, 3):
        raise ValueError("Sand mask, chart lattice or matrix shape mismatch")
    if not np.array_equal(np.unique(fac), np.array([0, 1], np.int16)) and not (
        np.unique(fac).size == 1 and int(np.unique(fac)[0]) in (0, 1)
    ):
        raise ValueError("Observed-sand optical fit accepts only 0/1 classes")
    Minv = np.linalg.inv(M)
    B1 = v5._colour(logB, Minv)
    F = None if flat is None else _rgb(flat, grid)
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    validxy = np.all(np.isfinite(xy), 1)
    if validxy.sum() < 6:
        raise ValueError("Missing source-bound ColorChecker lattice")
    chart = ((xx >= xy[validxy, 0].min() - 7) &
             (xx <= xy[validxy, 0].max() + 7) &
             (yy >= xy[validxy, 1].min() - 7) &
             (yy <= xy[validxy, 1].max() + 7))
    native_x_cell = xx + grid.origin_cell_xy[0]
    native_y_cell = yy + grid.origin_cell_xy[1]
    fixed_chart = (native_x_cell >= 569) & (native_y_cell < 131)
    sand = (fac == 1) & ~chart & ~fixed_chart
    keep = sand & v5._same_fac(fac)
    L = B1 - (F if F is not None else 0)
    dbg = (L[2] - L[1]) - (cons[2] - cons[1])
    finite = keep & np.isfinite(dbg)
    num = cv2.GaussianBlur(np.where(finite, dbg, 0).astype(np.float32), (0, 0), 3)
    den = cv2.GaussianBlur(finite.astype(np.float32), (0, 0), 3)
    dbg = np.where(finite, num / np.maximum(den, 1e-6), np.nan)
    m = keep & np.isfinite(dbg)
    if m.sum() > 100:
        dbg[fac == 1] -= np.median(dbg[m])
    fitkeep = keep & np.isfinite(dbg) & (dbg > -.08)
    P, a, n = _fit_field(B1[1], fac, fitkeep, grid)
    fraction = float(1 - fitkeep.sum() / max(keep.sum(), 1))
    dist = cv2.distanceTransform((~(sand | chart)).astype(np.uint8), cv2.DIST_L2, 5)
    taper = np.exp(-dist**2 / (2 * 8.0**2)).astype(np.float32)
    cpl = v5._chart_plane(chart_log_rgb, ref_chart_log_rgb, chart_usable, M)
    X = np.column_stack([np.ones(validxy.sum()), xy[validxy]])
    patch = np.arange(24)
    au = np.linalg.lstsq(X, ((patch % 4) - 1.5)[validxy], rcond=None)[0]
    av = np.linalg.lstsq(X, ((patch // 4) - 2.5)[validxy], rcond=None)[0]
    Ug = (au[0] + au[1] * xx + au[2] * yy).astype(np.float32)
    Vg = (av[0] + av[1] * xx + av[2] * yy).astype(np.float32)
    return v5.Baseline(
        run, M, Minv, P, a, n, fac, sand, chart, cpl, Ug, Vg, taper,
        logB, B1[1] - B1[2], xy, np.asarray(chart_log_rgb, np.float32),
        F, fraction,
    )


def baseline_field(
    fit: v5.Baseline, reference: v5.Baseline, grid: ReferenceGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the V5 achromatic field on the source-observed reference grid."""
    if fit.fac.shape != grid.shape_hw or reference.fac.shape != grid.shape_hw:
        raise ValueError("Baseline fit geometry does not match reference grid")
    poly = np.einsum("i,ihw->hw", reference.P - fit.P, grid.basis())
    common = [f for f in fit.a if f in reference.a]
    if not common:
        raise ValueError("No shared source-observed optical class")
    values = np.array([reference.a[f] - fit.a[f] for f in common])
    weights = np.array([fit.n[f] for f in common])
    meanoff = float(np.average(values, weights=weights))
    offset = np.full(grid.shape_hw, np.nan)
    for fac, val in zip(common, values):
        offset[(fit.fac == fac) & fit.sand & reference.sand] = val
    insand = np.isfinite(offset)
    total = np.where(insand, poly + offset, np.nan)
    global_level = float(np.nanmedian(total))
    field = np.where(insand, total, fit.taper * (poly + meanoff) +
                     (1 - fit.taper) * global_level)
    dist = cv2.distanceTransform((~fit.chart).astype(np.uint8), cv2.DIST_L2, 5)
    wc = np.exp(-dist**2 / (2 * 20.0**2))
    area = ~insand & ~fit.chart & (dist < 60)
    field[area] = wc[area] * (-fit.cpl[0]) + (1 - wc[area]) * field[area]
    field[fit.chart] = -(fit.cpl[0] + fit.cpl[1] * fit.Ug[fit.chart] +
                         fit.cpl[2] * fit.Vg[fit.chart])
    return field.astype(np.float32), insand


def apply_grid(
    log_rgb: np.ndarray, fit: v5.Baseline, reference: v5.Baseline,
    grid: ReferenceGrid, *, light_field: np.ndarray | None = None,
    source_eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Correct a grid, with optional fail-closed source-eligible output mask.

    The analytic field is evaluable outside observed sand, but a finite number
    there is not an eligibility claim. Callers preparing model inputs must
    provide their independently certified output-domain mask.
    """
    src = _rgb(log_rgb, grid)
    field, _ = baseline_field(fit, reference, grid)
    if light_field is not None:
        light = np.asarray(light_field, np.float32)
        if light.shape != grid.shape_hw:
            raise ValueError("Frame-light field has wrong reference geometry")
        field = field - light
    rgb = v5._colour(src, fit.Minv)
    rgb += field[None]
    if fit.F is not None:
        sand = fit.sand & reference.sand & ~fit.chart
        alpha = np.clip((v5.GB_SAT - (rgb[1] - rgb[2])) /
                        np.maximum(v5.GB_SAT - fit.gb0, .2), 0, 1)
        alpha = np.where(np.isfinite(alpha), alpha, 1)
        tap = np.where(sand, 1.0, fit.taper * ~fit.chart)
        neutral = fit.F[1:2]
        rgb -= (neutral + alpha[None] * (fit.F - neutral)) * tap[None]
    if source_eligible_mask is not None:
        eligible = np.asarray(source_eligible_mask, bool)
        if eligible.shape != grid.shape_hw:
            raise ValueError("Output source-eligibility mask has wrong grid shape")
        rgb[:, ~eligible] = np.nan
    return rgb.astype(np.float32)


def apply_native(
    log_rgb_run: np.ndarray, fit: v5.Baseline, reference: v5.Baseline,
    grid: ReferenceGrid, h_ref_to_run: np.ndarray, *,
    source_eligible_mask: np.ndarray,
    light_field_ref8: np.ndarray | None = None,
    destination_origin_xy: tuple[int, int] = (0, 0),
    tile_rows: int = 128,
) -> np.ndarray:
    """Project the observed-sand correction onto exact run-native RAW pixels.

    The image and its labels remain on their original grid.  `h_ref_to_run`
    maps native AC28 reference coordinates to the current run frame; a padded
    RAW ring is represented by `destination_origin_xy`.  The caller must
    supply an independently source-certified eligibility mask.  Neither the
    smooth field nor the sensor's finite footprint promotes disputed sand,
    metal or frame pixels to eligible output.
    """
    source = np.asarray(log_rgb_run)
    if source.dtype != np.float32:
        raise TypeError("V4-R3 requires float32 log-linear RAW; uint8/JPEG is invalid")
    source = v5._log_rgb(source)
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("Expected channel-first native log-linear RAW")
    height, width = source.shape[1:]
    eligible = np.asarray(source_eligible_mask, bool)
    if eligible.shape != (height, width):
        raise ValueError("Native source eligibility mask shape mismatch")
    if fit.fac.shape != grid.shape_hw or reference.fac.shape != grid.shape_hw:
        raise ValueError("Fit and virtual reference use a different grid")
    H = np.asarray(h_ref_to_run, np.float64)
    if (H.shape != (3, 3) or not np.all(np.isfinite(H)) or
            abs(np.linalg.det(H)) < 1e-10):
        raise ValueError("Expected finite, invertible native homography")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    light = None
    if light_field_ref8 is not None:
        light = np.asarray(light_field_ref8, np.float32)
        if light.shape != grid.shape_hw:
            raise ValueError("Frame-light field uses a different grid")
    field_ref, _ = baseline_field(fit, reference, grid)
    inverse = np.linalg.inv(H)
    output = np.empty_like(source)

    def sample(array: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        planes = np.asarray(array, np.float32)
        if planes.ndim == 2:
            return cv2.remap(planes, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        return np.stack([sample(plane, mx, my) for plane in planes])

    ox, oy = destination_origin_xy
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        yy, xx = np.mgrid[start:stop, :width]
        xrun = xx.astype(np.float64) + ox
        yrun = yy.astype(np.float64) + oy
        den = inverse[2, 0] * xrun + inverse[2, 1] * yrun + inverse[2, 2]
        if np.any(np.abs(den) < 1e-9):
            raise ValueError("Native pixel maps to reference infinity")
        xref = (inverse[0, 0] * xrun + inverse[0, 1] * yrun + inverse[0, 2]) / den
        yref = (inverse[1, 0] * xrun + inverse[1, 1] * yrun + inverse[1, 2]) / den
        mx = ((xref - 3.5) / 8 - grid.origin_cell_xy[0]).astype(np.float32)
        my = ((yref - 3.5) / 8 - grid.origin_cell_xy[1]).astype(np.float32)
        in_field = ((mx >= -.5) & (mx <= grid.shape_hw[1] - .5) &
                    (my >= -.5) & (my <= grid.shape_hw[0] - .5))
        field = sample(field_ref, mx, my)
        if light is not None:
            field -= sample(light, mx, my)
        corrected = v5._colour(source[:, start:stop], fit.Minv)
        corrected += field[None]
        if fit.F is not None:
            flat = sample(fit.F, mx, my)
            gb0 = sample(fit.gb0, mx, my)
            taper = sample(fit.taper, mx, my)
            chart = sample(fit.chart.astype(np.float32), mx, my)
            alpha = np.clip((v5.GB_SAT - (corrected[1] - corrected[2])) /
                            np.maximum(v5.GB_SAT - gb0, .2), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            neutral = flat[1:2]
            # taper is one on observed sand and fades continuously outward;
            # no categorical native sand override can introduce a hard seam.
            tap = taper * (1 - np.clip(chart, 0, 1))
            corrected -= (neutral + alpha[None] * (flat - neutral)) * tap[None]
        valid = (eligible[start:stop] & in_field &
                 np.all(np.isfinite(source[:, start:stop]), axis=0) &
                 np.all(np.isfinite(corrected), axis=0))
        corrected[:, ~valid] = np.nan
        output[:, start:stop] = corrected
    return output


def estimate_frame_light_expanded_diagnostic(
    fit: v5.Baseline, frame_log_ref8: np.ndarray, grid: ReferenceGrid, *,
    frame_chart_xy_ref8: np.ndarray, frame_chart_usable: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit V5 light on its exact old centre and evaluate its polynomial outside.

    The central 372x684 result is numerically the frozen V5 light estimator.
    The added ring is a polynomial forecast, not source-certified by this API;
    callers must pass separate late-frame and spatial edge gates before use.
    """
    frame = _rgb(frame_log_ref8, grid)
    x0, y0 = -grid.origin_cell_xy[0], -grid.origin_cell_xy[1]
    h, w = grid.original_shape_hw
    if x0 < 0 or y0 < 0 or x0 + w > grid.shape_hw[1] or y0 + h > grid.shape_hw[0]:
        raise ValueError("Expanded grid does not contain the frozen V5 centre")
    centre = (slice(y0, y0 + h), slice(x0, x0 + w))
    xy = np.asarray(frame_chart_xy_ref8, np.float32)
    usable = np.asarray(frame_chart_usable, bool)
    if xy.shape != (24, 2) or usable.shape != (24,):
        raise ValueError("Exact per-frame source-bound chart required")
    local_xy = xy - np.array([x0, y0], np.float32)
    central_fit = replace(
        fit,
        fac=fit.fac[centre],
        sand=fit.sand[centre],
        chart=fit.chart[centre],
        Ug=fit.Ug[centre],
        Vg=fit.Vg[centre],
        taper=fit.taper[centre],
        logB=fit.logB[(slice(None), *centre)],
        gb0=fit.gb0[centre],
        xy=fit.xy - np.array([x0, y0], np.float32),
        F=None if fit.F is None else fit.F[(slice(None), *centre)],
    )
    original_field, info = v5.estimate_frame_light_grid(
        central_fit, frame[(slice(None), *centre)],
        frame_chart_xy8=local_xy, frame_chart_usable=usable)
    degree = int(info["degree"])
    coef = np.asarray(info["coefficients"], np.float64)
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    old_x = xx + grid.origin_cell_xy[0]
    old_y = yy + grid.origin_cell_xy[1]
    full = v5._light_basis((old_x - 1.5) / 4,
                           (old_y - 1.5) / 4, degree) @ coef
    full = full.astype(np.float32)
    if not np.array_equal(full[centre], original_field):
        raise ArithmeticError("Expanded frame light changed frozen central field")
    info.update({
        "coordinate_frame": "ac28-reference-expanded-1of8",
        "central_frozen_v5_parity": True,
        "added_ring_status": "diagnostic polynomial forecast; independent QA required",
        "grid_origin_cell_xy": list(grid.origin_cell_xy),
    })
    return full, info


def make_local_flat(
    corrected_new: list[np.ndarray], consensus: np.ndarray,
    reference: v5.Baseline, common_new_support: np.ndarray,
    grid: ReferenceGrid,
) -> np.ndarray:
    """Frozen V4 local F estimator, masked by independent new-donor support."""
    if not corrected_new:
        raise ValueError("At least one corrected new-light donor required")
    _rgb(consensus, grid)
    difference = np.nanmedian(np.stack([a - consensus for a in corrected_new]), axis=0)
    valid = (reference.sand & np.asarray(common_new_support, bool) &
             np.all(np.isfinite(difference), axis=0))
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 2)
    F = np.zeros_like(difference, dtype=np.float32)
    for c in range(3):
        num = cv2.GaussianBlur(np.where(valid, difference[c], 0).astype(np.float32),
                               (0, 0), 2)
        F[c] = num / np.maximum(den, 1e-6)
    F[:, den < .05] = 0
    if valid.sum() > 100:
        F[:, valid] -= np.median(F[1, valid])
    return F


def save_baseline(
    path: Path, fit: v5.Baseline, grid: ReferenceGrid, provenance: dict,
) -> dict:
    """Serialize a source-bound numerical fit without pickle or old-grid alias."""
    path = Path(path)
    if path.suffix.lower() != ".npz" or fit.fac.shape != grid.shape_hw:
        raise ValueError("Expected .npz and exact reference grid")
    required = ("source_arw_sha256", "source_review_sha256",
                "source_manifest_sha256", "sand_mask_sha256")
    if any(not provenance.get(k) for k in required):
        raise ValueError("Missing source, geometry or sand-mask SHA binding")
    keys = np.asarray(sorted(fit.a), dtype=np.int16)
    arrays = {
        "run": np.asarray(fit.run), "M": fit.M, "Minv": fit.Minv,
        "P": fit.P, "facies_keys": keys,
        "facies_offsets": np.asarray([fit.a[int(k)] for k in keys]),
        "facies_counts": np.asarray([fit.n[int(k)] for k in keys]),
        "fac": fit.fac, "sand": fit.sand, "chart": fit.chart,
        "cpl": fit.cpl, "Ug": fit.Ug, "Vg": fit.Vg,
        "taper": fit.taper, "logB": fit.logB, "gb0": fit.gb0,
        "xy": fit.xy, "CB": fit.CB,
        "has_F": np.asarray(fit.F is not None),
        "F": fit.F if fit.F is not None else np.zeros((0,), np.float32),
        "residual_exclusion_fraction": np.asarray(fit.residual_exclusion_fraction),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "contract": "ffac-v5-observed-sand-expanded-source-bound-baseline-v1",
        "run": fit.run, "npz_sha256": digest,
        "grid_shape_hw": list(grid.shape_hw),
        "grid_origin_cell_xy": list(grid.origin_cell_xy),
        "original_grid_shape_hw": list(grid.original_shape_hw),
        "source": provenance,
        "residual_exclusion_fraction": fit.residual_exclusion_fraction,
    }
    sidecar = path.with_suffix(".json")
    temp_json = sidecar.with_name(sidecar.stem + ".tmp.json")
    temp_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(temp_json, sidecar)
    return receipt


def load_baseline(path: Path, *, verify: bool = True) -> tuple[v5.Baseline, ReferenceGrid]:
    """Load a V4-R3 fit; reject old shared-facies or different-grid assets."""
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt.get("contract") != "ffac-v5-observed-sand-expanded-source-bound-baseline-v1":
        raise ValueError("Unknown observed-sand baseline contract")
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("npz_sha256"):
        raise ValueError("Baseline NPZ SHA mismatch")
    grid = ReferenceGrid(tuple(receipt["grid_shape_hw"]),
                         tuple(receipt["grid_origin_cell_xy"]),
                         tuple(receipt["original_grid_shape_hw"]))
    fit = v5.load_baseline(path, verify=False)
    if fit.fac.shape != grid.shape_hw:
        raise ValueError("Baseline mask does not match bound reference grid")
    return fit, grid
