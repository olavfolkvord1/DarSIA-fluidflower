"""FF_AC v5 RAW photometry; geometry and source binding are caller-owned.

Input is 3 x H x W log-linear RGB registered to the DarSIA review canvas.
Calibration is fitted once per run on its pre-injection baseline at 1/8
resolution. The caller must verify source SHA, chart geometry and facies.
This module has no CUDA dependency and does not access model/label files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

GRID = (372, 684)
NATIVE = (2979, 5472)
GB_SAT = 2.044  # Claude v5 yellow-pixel-derived dye endpoint.
LIGHT_WEIGHTS = np.array([0.0, 0.28, 0.72], dtype=np.float64)


def _log_rgb(a, shape=None):
    a = np.asarray(a, np.float32)
    if a.ndim != 3 or a.shape[0] != 3 or (shape is not None and a.shape[1:] != shape):
        raise ValueError(f"Expected RGB log array (3,H,W), got {a.shape}")
    return a


def _basis(h=372, w=684):
    yy, xx = np.mgrid[:h, :w]
    x, y = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    return np.stack([x**a * y**b for d in range(1, 5)
                     for a in range(d + 1) for b in [d - a]]).astype(np.float64)


def _same_fac(f):
    h, w = f.shape
    p = np.pad(f, 1, mode="edge")
    out = np.ones(f.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1+dy:1+dy+h, 1+dx:1+dx+w] == f
    return out


def _fit_field(g, fac, keep):
    basis = _basis().reshape(14, -1)
    idx = np.flatnonzero((keep & (fac >= 1) & np.isfinite(g)).ravel())
    if len(idx) < 1000:
        raise ValueError(f"Too few baseline sand cells: {len(idx)}")
    ff = fac.ravel()[idx]
    classes = [f for f in range(1, 9) if np.count_nonzero(ff == f) >= 30]
    X = np.column_stack([basis[:, idx].T] + [(ff == f).astype(float) for f in classes])
    y = g.ravel()[idx].astype(float)
    wt = np.ones(len(y))
    for _ in range(6):
        sw = np.sqrt(wt)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        wt = np.minimum(1, .05 / np.maximum(np.abs(y - X @ coef), 1e-9))
    return coef[:14], {f: float(coef[14+i]) for i, f in enumerate(classes)}, {
        f: int(np.count_nonzero(ff == f)) for f in classes}


def _colour(log_rgb, inv):
    bad = ~np.all(np.isfinite(log_rgb), 0)
    linear = np.exp(np.nan_to_num(log_rgb, nan=-30)).astype(np.float32)
    out = np.log(np.maximum(np.einsum("ij,jhw->ihw", inv, linear, optimize=True), 1e-3))
    out[:, bad] = np.nan
    return out.astype(np.float32)


def _chart_plane(C, CR, usable, M):
    C, CR = np.asarray(C), np.asarray(CR)
    u = np.asarray(usable, bool) & np.all(np.isfinite(C), 0) & np.all(np.isfinite(CR), 0)
    if C.shape != (3, 24) or CR.shape != (3, 24) or np.count_nonzero(u) < 6:
        raise ValueError("Invalid chart samples or too few common patches")
    k = np.arange(24)
    U, V = (k % 4) - 1.5, (k // 4) - 2.5
    X = np.column_stack([np.ones(u.sum()), U[u], V[u]])
    y = (C[:, u] - np.log(np.maximum(M @ np.exp(CR[:, u]), 1e-9))).mean(0)
    wt = np.ones(len(y))
    for _ in range(8):
        sw = np.sqrt(wt)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        wt = np.minimum(1, .03 / np.maximum(np.abs(y - X @ b), 1e-9))
    return b


@dataclass
class Baseline:
    run: str
    M: np.ndarray
    Minv: np.ndarray
    P: np.ndarray
    a: dict[int, float]
    n: dict[int, int]
    fac: np.ndarray
    sand: np.ndarray
    chart: np.ndarray
    cpl: np.ndarray
    Ug: np.ndarray
    Vg: np.ndarray
    taper: np.ndarray
    logB: np.ndarray
    gb0: np.ndarray
    xy: np.ndarray
    CB: np.ndarray
    F: np.ndarray | None
    residual_exclusion_fraction: float


def fit_baseline_grid(*, run, logB, facies, chart_xy, chart_log_rgb,
                      ref_chart_log_rgb, chart_usable, matrix,
                      rig_consensus, flat=None):
    """Fit the pre-injection run correction on the registered 1/8 RAW grid."""
    logB = _log_rgb(logB, GRID)
    cons = _log_rgb(rig_consensus, GRID)
    fac = np.asarray(facies, np.int16)
    xy = np.asarray(chart_xy, np.float64)
    M = np.asarray(matrix, np.float64)
    if fac.shape != GRID or xy.shape != (24, 2) or M.shape != (3, 3):
        raise ValueError("Facies, chart or matrix shape mismatch")
    Minv = np.linalg.inv(M)
    B1 = _colour(logB, Minv)
    F = None if flat is None else _log_rgb(flat, GRID)
    yy, xx = np.mgrid[:GRID[0], :GRID[1]]
    validxy = np.all(np.isfinite(xy), 1)
    if validxy.sum() < 6:
        raise ValueError("Missing ColorChecker lattice")
    cb = ((xx >= xy[validxy, 0].min() - 7) &
          (xx <= xy[validxy, 0].max() + 7) &
          (yy >= xy[validxy, 1].min() - 7) &
          (yy <= xy[validxy, 1].max() + 7))
    fixchart = (xx >= 569) & (yy < 131)
    sand = (fac >= 1) & ~cb & ~fixchart
    keep = sand & _same_fac(fac)
    L = B1 - (F if F is not None else 0)
    dbg = (L[2] - L[1]) - (cons[2] - cons[1])
    fin = keep & np.isfinite(dbg)
    num = cv2.GaussianBlur(np.where(fin, dbg, 0).astype(np.float32), (0, 0), 3)
    den = cv2.GaussianBlur(fin.astype(np.float32), (0, 0), 3)
    dbg = np.where(fin, num / np.maximum(den, 1e-6), np.nan)
    for f in range(1, 9):
        m = keep & (fac == f) & np.isfinite(dbg)
        if m.sum() > 100:
            dbg[fac == f] -= np.median(dbg[m])
    fitkeep = keep & np.isfinite(dbg) & (dbg > -.08)
    P, a, n = _fit_field(B1[1], fac, fitkeep)
    frac = float(1 - fitkeep.sum() / max(keep.sum(), 1))
    dist = cv2.distanceTransform((~(sand | cb)).astype(np.uint8), cv2.DIST_L2, 5)
    taper = np.exp(-dist**2 / (2 * 8.0**2)).astype(np.float32)
    cpl = _chart_plane(chart_log_rgb, ref_chart_log_rgb, chart_usable, M)
    X = np.column_stack([np.ones(validxy.sum()), xy[validxy]])
    k = np.arange(24)
    au = np.linalg.lstsq(X, ((k % 4) - 1.5)[validxy], rcond=None)[0]
    av = np.linalg.lstsq(X, ((k // 4) - 2.5)[validxy], rcond=None)[0]
    Ug = (au[0] + au[1] * xx + au[2] * yy).astype(np.float32)
    Vg = (av[0] + av[1] * xx + av[2] * yy).astype(np.float32)
    return Baseline(run, M, Minv, P, a, n, fac, sand, cb, cpl, Ug, Vg,
                    taper, logB, B1[1] - B1[2], xy,
                    np.asarray(chart_log_rgb, np.float32), F, frac)


def baseline_field(fit: Baseline, ref: Baseline):
    poly = np.einsum("i,ihw->hw", ref.P - fit.P, _basis())
    common = [f for f in fit.a if f in ref.a]
    if not common:
        raise ValueError("No common facies with AC28")
    vals = np.array([ref.a[f] - fit.a[f] for f in common])
    weights = np.array([fit.n[f] for f in common])
    meanoff = float(np.average(vals, weights=weights))
    off = np.full(GRID, np.nan)
    for f, v in zip(common, vals):
        off[(fit.fac == f) & fit.sand] = v
    insand = np.isfinite(off)
    total = np.where(insand, poly + off, np.nan)
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


def _resize(a, shape):
    return cv2.resize(np.asarray(a, np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def _native_sand_field_tile(fit, ref, facies_tile, lo, width):
    """Evaluate smooth field and facies offsets at native pixel centres."""
    yy = (np.arange(lo, lo + facies_tile.shape[0], dtype=np.float64) - 3.5) / 8
    xx = (np.arange(width, dtype=np.float64) - 3.5) / 8
    x = ((xx - GRID[1] / 2) / (GRID[1] / 2))[None, :]
    y = ((yy - GRID[0] / 2) / (GRID[0] / 2))[:, None]
    coef = ref.P - fit.P
    out = np.zeros(facies_tile.shape, dtype=np.float32)
    k = 0
    for d in range(1, 5):
        for a in range(d + 1):
            b = d - a
            out += (coef[k] * (x**a) * (y**b)).astype(np.float32)
            k += 1
    for f in fit.a:
        if f in ref.a:
            out[facies_tile == f] += ref.a[f] - fit.a[f]
    return out


def apply_v5(log_rgb, fit: Baseline, ref: Baseline, *, facies_native=None,
             light_field8=None, tile_rows=384):
    """Correct registered 1/8 or full-native RAW; return log-linear RGB."""
    source = _log_rgb(log_rgb)
    shape = source.shape[1:]
    if shape not in (GRID, NATIVE):
        raise ValueError(f"Not a registered 1/8/native frame: {shape}")
    if shape == NATIVE and (facies_native is None or np.shape(facies_native) != NATIVE):
        raise ValueError("Native correction requires source-bound native facies")
    field8, _ = baseline_field(fit, ref)
    field = field8 if shape == GRID else _resize(field8, shape)
    dlight_native = None
    if light_field8 is not None:
        dlight = np.asarray(light_field8, np.float32)
        if dlight.shape != GRID:
            raise ValueError("Light field must be on 1/8 grid")
        dlight_native = dlight if shape == GRID else _resize(dlight, shape)
        field = field - dlight_native
    fac = fit.fac if shape == GRID else np.asarray(facies_native)
    chart = fit.chart if shape == GRID else _resize(fit.chart.astype(np.float32), shape) > .5
    sand = (fac >= 1) & ~chart
    taper = fit.taper if shape == GRID else _resize(fit.taper, shape)
    tap = np.where(sand, 1.0, taper * ~chart).astype(np.float32)
    F = fit.F if (shape == GRID or fit.F is None) else np.stack(
        [_resize(f, shape) for f in fit.F])
    gb0 = fit.gb0 if shape == GRID else _resize(fit.gb0, shape)
    out = np.empty_like(source)
    for lo in range(0, shape[0], tile_rows):
        hi = min(lo + tile_rows, shape[0])
        rgb = _colour(source[:, lo:hi], fit.Minv)
        field_tile = field[lo:hi].copy()
        if shape == NATIVE:
            yy, xx = np.mgrid[lo:hi, :shape[1]]
            fixed_chart = (xx >= 569 * 8) & (yy < 131 * 8)
            common = [f for f in fit.a if f in ref.a]
            exact_sand = sand[lo:hi] & ~fixed_chart & np.isin(fac[lo:hi], common)
            exact = _native_sand_field_tile(fit, ref, fac[lo:hi], lo, shape[1])
            if dlight_native is not None:
                exact -= dlight_native[lo:hi]
            field_tile[exact_sand] = exact[exact_sand]
        rgb += field_tile[None]
        if F is not None:
            f = F[:, lo:hi]
            gb = rgb[1] - rgb[2]
            alpha = np.clip((GB_SAT - gb) /
                            np.maximum(GB_SAT - gb0[lo:hi], .2), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            fa = f[1:2]
            rgb -= (fa + alpha[None] * (f - fa)) * tap[None, lo:hi]
        out[:, lo:hi] = rgb
    return out


def render_ac28_response(corrected_log_linear, response):
    """Claude's AC28 linear-RAW camera response; RGB float32 in [0,1]."""
    src = _log_rgb(corrected_log_linear)
    M = np.asarray(response["M"], np.float64)
    luts = response["luts"]
    if M.shape != (3, 3) or len(luts) != 3:
        raise ValueError("Invalid AC28 response")
    bad = ~np.all(np.isfinite(src), 0)
    lin = np.exp(np.nan_to_num(src, nan=0)).astype(np.float32)
    z = np.maximum(np.einsum("ij,jhw->ihw", M, lin, optimize=True), 1e-3)
    out = np.stack([np.interp(np.log(z[c]), np.asarray(luts[c][0]),
                              np.asarray(luts[c][1])) for c in range(3)])
    out[:, bad] = 255
    return np.clip(out.transpose(1, 2, 0) / 255, 0, 1).astype(np.float32)


def load_ac28_response(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["response"]


def save_baseline(path: Path, fit: Baseline, provenance: dict):
    """Save numerical fit without pickle; sidecar binds immutable source SHA."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".npz":
        raise ValueError("Baseline calibration must be a .npz")
    required = ("source_arw_sha256", "source_review_sha256",
                "source_manifest_sha256", "facies_sha256")
    missing = [k for k in required if not provenance.get(k)]
    if missing:
        raise ValueError(f"Missing baseline source binding: {missing}")
    keys = np.array(sorted(fit.a), dtype=np.int16)
    arrays = {
        "run": np.asarray(fit.run),
        "M": fit.M,
        "Minv": fit.Minv,
        "P": fit.P,
        "facies_keys": keys,
        "facies_offsets": np.array([fit.a[int(k)] for k in keys]),
        "facies_counts": np.array([fit.n[int(k)] for k in keys]),
        "fac": fit.fac,
        "sand": fit.sand,
        "chart": fit.chart,
        "cpl": fit.cpl,
        "Ug": fit.Ug,
        "Vg": fit.Vg,
        "taper": fit.taper,
        "logB": fit.logB,
        "gb0": fit.gb0,
        "xy": fit.xy,
        "CB": fit.CB,
        "has_F": np.asarray(fit.F is not None),
        "F": fit.F if fit.F is not None else np.zeros((0,), np.float32),
        "residual_exclusion_fraction": np.asarray(fit.residual_exclusion_fraction),
    }
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "contract": "ffac-v5-source-bound-baseline-calibration-v1",
        "run": fit.run,
        "npz_sha256": sha,
        "source": provenance,
        "grid_shape": list(GRID),
        "residual_exclusion_fraction": fit.residual_exclusion_fraction,
    }
    sidecar = path.with_suffix(".json")
    temporary_json = sidecar.with_name(sidecar.stem + ".tmp.json")
    temporary_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    os.replace(temporary_json, sidecar)
    return receipt


def load_baseline(path: Path, *, verify=True):
    """Load numerical fit and fail closed on missing/mismatched provenance."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if verify:
        receipt = json.loads(sidecar.read_text(encoding="utf-8"))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.get("contract") != "ffac-v5-source-bound-baseline-calibration-v1":
            raise ValueError("Unknown FF_AC calibration contract")
        if sha != receipt.get("npz_sha256"):
            raise ValueError("Calibration SHA does not match sidecar")
    with np.load(path, allow_pickle=False) as z:
        keys = z["facies_keys"].astype(int)
        fit = Baseline(
            run=str(z["run"]),
            M=z["M"],
            Minv=z["Minv"],
            P=z["P"],
            a={int(k): float(v) for k, v in zip(keys, z["facies_offsets"])},
            n={int(k): int(v) for k, v in zip(keys, z["facies_counts"])},
            fac=z["fac"],
            sand=z["sand"],
            chart=z["chart"],
            cpl=z["cpl"],
            Ug=z["Ug"],
            Vg=z["Vg"],
            taper=z["taper"],
            logB=z["logB"],
            gb0=z["gb0"],
            xy=z["xy"],
            CB=z["CB"],
            F=z["F"] if bool(z["has_F"]) else None,
            residual_exclusion_fraction=float(z["residual_exclusion_fraction"]),
        )
    return fit


def registered_raw_grid_from_arw(arw_path, review_to_camera_h, profile,
                                 geometry_params):
    """Recreate Claude v5's 1/8 RAW sample, including pre-warp CFA blur.

    This is intentionally *not* an 8x8 box average of the registered native
    frame. It Gaussian-blurs each half-resolution RAW colour plane before
    mapping 1/8-grid centre samples through the camera and Sony lens model.
    It runs on CPU and returns log-linear RGB with invalid values as NaN.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("RAW source processing requires rawpy") from exc
    arw_path = Path(arw_path)
    H = np.asarray(review_to_camera_h, np.float64)
    if H.shape != (3, 3):
        raise ValueError("Expected 3x3 review-to-camera homography")
    if profile not in ("early", "late"):
        raise ValueError(f"Unknown Sony geometry profile: {profile}")
    with rawpy.imread(str(arw_path)) as raw:
        if raw.raw_image_visible.shape != (3672, 5496):
            raise ValueError(f"Unexpected RAW visible size: {raw.raw_image_visible.shape}")
        if not np.array_equal(raw.raw_pattern, [[0, 1], [3, 2]]):
            raise ValueError(f"Unexpected RAW CFA pattern: {raw.raw_pattern}")
        v = raw.raw_image_visible.astype(np.float32).copy()
    planes = {
        "R": v[0::2, 0::2] - 800.0,
        "G": .5 * (v[0::2, 1::2] + v[1::2, 0::2]) - 800.0,
        "B": v[1::2, 1::2] - 800.0,
    }
    planes = {c: cv2.GaussianBlur(p, (0, 0), 1.8) for c, p in planes.items()}
    h, w = GRID
    gy, gx = np.mgrid[:h, :w]
    points = np.column_stack([
        (8 * gx + 3.5).ravel(), (8 * gy + 3.5).ravel()
    ]).astype(np.float32)
    cam = cv2.perspectiveTransform(points[None], H)[0].astype(np.float64)
    u, vv = cam[:, 0], cam[:, 1]
    p = geometry_params["fitted_all_runs_no_chart"][profile]
    table = geometry_params["distortion_tables"][profile]
    ca = geometry_params["ca_tables"]
    du, dv = u - p["cu"], vv - p["cv"]
    radius = np.hypot(du, dv)
    radial = np.interp(radius / p["R"], np.linspace(0, 1, 11), table)
    scale = 1 + p["amp"] * radial / 16384.0
    phase = {"R": (0.0, 0.0), "G": (.5, .5), "B": (1.0, 1.0)}
    channels = []
    for colour in "RGB":
        ca_factor = (
            np.interp(radius / p["R"], np.linspace(0, 1, 11), ca[colour])
            / 2**21 if colour in ca else 0
        )
        x = p["cu"] + 12 + p["tx"] + scale * (1 + ca_factor) * du
        y = p["cv"] + 296 + p["ty"] + scale * (1 + ca_factor) * dv
        oy, ox = phase[colour]
        mx = ((x - ox) / 2).reshape(h, w).astype(np.float32)
        my = ((y - oy) / 2).reshape(h, w).astype(np.float32)
        channels.append(cv2.remap(
            planes[colour], mx, my, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
        ))
    linear = np.stack(channels).astype(np.float32)
    linear = np.where(
        np.isfinite(linear) & (linear > 2) & (linear < 15100),
        linear,
        np.nan,
    )
    return np.log(linear).astype(np.float32)


def sample_chart_log(log8, xy8):
    """Median of the same 3 x 3 patch-centre probes as the v5 reference."""
    arr = _log_rgb(log8, GRID)
    xy = np.asarray(xy8, np.float32)
    if xy.shape != (24, 2):
        raise ValueError("Chart coordinates must be (24,2)")
    out = []
    for c in range(3):
        vals = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x = (xy[:, 0] + dx).reshape(1, -1)
                y = (xy[:, 1] + dy).reshape(1, -1)
                vals.append(cv2.remap(arr[c], x, y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=np.nan)[0])
        out.append(np.nanmedian(np.stack(vals), axis=0))
    return np.asarray(out, np.float32)


def _light_basis(x, y, degree):
    x, y = (np.asarray(x, float) - 85) / 85, (np.asarray(y, float) - 46) / 46
    return np.stack([x**a * y**b for d in range(degree + 1)
                     for a in range(d + 1) for b in [d - a]], axis=-1)


def estimate_frame_light_grid(
    fit: Baseline,
    frame_log8,
    *,
    frame_chart_xy8=None,
    frame_chart_usable=None,
    minimum_weak_sand_cells=200,
):
    """Fit Claude v5's per-frame achromatic light drift relative to start B.

    The frame and baseline must share the source-bound 1/8 review grid.
    Returns a 372 x 684 log correction and diagnostics. This fit is adaptive
    on the current frame, so quality estimates must use held-out regions.
    """
    frame = _log_rgb(frame_log8, GRID)
    xy = fit.xy if frame_chart_xy8 is None else np.asarray(frame_chart_xy8, np.float32)
    current_chart = sample_chart_log(frame, xy)
    valid_chart = np.all(np.isfinite(current_chart), 0) & np.all(np.isfinite(fit.CB), 0)
    if frame_chart_usable is not None:
        valid_chart &= np.asarray(frame_chart_usable, bool)
    if valid_chart.sum() < 6:
        raise ValueError(f"Only {valid_chart.sum()} valid frame ColorChecker patches")
    with np.errstate(all="ignore"):
        coarse = np.nanmean(np.exp(frame).reshape(3, 93, 4, 171, 4), axis=(2, 4))
        coarse_b = np.nanmean(np.exp(fit.logB).reshape(3, 93, 4, 171, 4), axis=(2, 4))
        delta = np.log(coarse) - np.log(coarse_b)
    finite = np.all(np.isfinite(delta), 0)
    chroma_rg = delta[0] - delta[1]
    chroma_bg = delta[2] - delta[1]
    dchart = LIGHT_WEIGHTS @ (current_chart[:, valid_chart] - fit.CB[:, valid_chart])
    chart_centre = float(np.median(dchart))
    weak = (
        finite
        & (np.abs(chroma_bg) < .10)
        & (np.abs(chroma_rg) < .15)
        & (np.linalg.norm(np.nan_to_num(delta - chart_centre), axis=0) < .3)
    )
    fac = fit.fac
    yy8, xx8 = np.mgrid[:GRID[0], :GRID[1]]
    fixed_chart = (xx8 >= 569) & (yy8 < 131)
    top = np.argmax(fac >= 1, axis=0)
    upper8 = (
        (fac == 0) & (yy8 < top[None, :] - 6) & (yy8 > 28)
        & ~fit.chart & ~fixed_chart & (xx8 > 8) & (xx8 < 560)
    )
    f4 = fac.reshape(93, 4, 171, 4)
    chart4 = fit.chart.reshape(93, 4, 171, 4).any((1, 3))
    fixed4 = fixed_chart.reshape(93, 4, 171, 4).any((1, 3))
    sand = ((f4 >= 1).mean((1, 3)) > .99) & ~chart4 & ~fixed4
    upper = upper8.reshape(93, 4, 171, 4).mean((1, 3)) > .99
    y = np.tensordot(LIGHT_WEIGHTS, delta, axes=(0, 0))
    weak_sand = sand & weak
    degree = 2 if weak_sand.sum() >= minimum_weak_sand_cells else 1
    gy, gx = np.mgrid[:93, :171]
    X, Y, W = [], [], []
    cx, cy = xy[valid_chart, 0] / 4 - .375, xy[valid_chart, 1] / 4 - .375
    X.append(_light_basis(cx, cy, degree))
    Y.append(dchart)
    W.append(np.full(len(dchart), .25 / len(dchart)))
    upper_weak = upper & weak
    if upper_weak.sum() > 20:
        X.append(_light_basis(gx[upper_weak], gy[upper_weak], degree))
        Y.append(y[upper_weak])
        W.append(np.full(upper_weak.sum(), .25 / upper_weak.sum()))
    if weak_sand.sum() > 50:
        X.append(_light_basis(gx[weak_sand], gy[weak_sand], degree))
        Y.append(y[weak_sand])
        W.append(np.full(weak_sand.sum(), .5 / weak_sand.sum()))
    design, target, weights = np.vstack(X), np.concatenate(Y), np.concatenate(W)
    if len(target) < 3 * design.shape[1]:
        raise ValueError("Insufficient independent light references")
    huber = np.ones(len(target))
    for _ in range(8):
        sw = np.sqrt(weights * huber)
        coef = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)[0]
        huber = np.minimum(1, .02 / np.maximum(np.abs(target - design @ coef), 1e-9))
    iy, ix = np.mgrid[:GRID[0], :GRID[1]]
    field = _light_basis((ix - 1.5) / 4, (iy - 1.5) / 4, degree) @ coef
    info = {
        "degree": degree,
        "n_chart": int(valid_chart.sum()),
        "n_weak_sand": int(weak_sand.sum()),
        "fraction_weak_sand": float(weak_sand.sum() / max(sand.sum(), 1)),
        "n_weak_upper": int(upper_weak.sum()),
        "chart_log_drift": chart_centre,
        "coefficients": coef.tolist(),
    }
    return field.astype(np.float32), info
