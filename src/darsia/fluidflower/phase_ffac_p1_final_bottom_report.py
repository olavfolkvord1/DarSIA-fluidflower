"""Read-only photographic bottom-edge measurements for FF_AC P1-final.

The image texture locates a *candidate* sand/structure transition. The bound
rig polygon is only the search limit; it is never treated as observed sand.
Weak or conflicting photographs produce unknown samples, not an invented edge.
"""
from __future__ import annotations

import copy
import math
import warnings

import cv2
import numpy as np

CONTRACT = "ffac-p1-final-photo-bottom-measurement-v1"
METHOD = {
    "contract": CONTRACT,
    "role": "descriptive report only; never a domain or label correction",
    "sample_spacing_mm": 10.0,
    "sampling_direction": "metric x along the median rig-domain bottom row, not arc length of stepped mask sides",
    "non_floor_column_guard_mm": 30.0,
    "search_height_mm": 30.0,
    "profile_half_width_mm": 4.0,
    "comparison_window_mm": 2.0,
    "comparison_gap_mm": 0.5,
    "bottom_profile_context": "observed RAW/native photo is read up to 5 mm below the current domain edge when it exists; no unobserved window is filled",
    "texture": "median absolute first spatial difference of camera-log luminance; horizontal medians in a fixed 8 mm window",
    "luminance": "mean of three already-normalised camera-log RGB channels; no JPEG gamma or contrast fit",
    "candidate_score": "log(above/below texture) + max(0, below-minus-above log luminance)",
    "transition_location": "midpoint of the contiguous >=90% winning-score plateau containing the strongest score; avoids selecting one grain-noise maximum within a step window",
    "evidence_gate": "above/below texture >= 1.75 and log brightness rise >= 0.01, or texture ratio >= 3",
    "minimum_valid_share_per_window": 0.9,
    "native_check_tolerance_mm": 1.0,
    "uncertainty": "range of admissible transition rows with score within 90% of the winning score, plus one native pixel; not a statistical confidence interval",
    "interpolation": "linear only between adjacent accepted 1 cm samples; never bridge a rejected or unobserved sample",
    "hump": "longest contiguous interval at least 2 mm thicker than the 10th-percentile measured band; reported descriptive height above that baseline",
    "physical_scale": "stored approved 900 mm metric transform and its isotropic linear factor; vertical scale is not independently verified",
    "coordinate_conversion": "project native pixel centre [canvas_x+origin_x,canvas_y+origin_y,1] through per-source H_frame_native_to_metric_pixel, divide homogeneous denominator, then multiply by sqrt(metric_pixel_area_mm2)*physical_scale_revision.linear_factor",
    "x_origin": "the inherited common metric-plane origin; hump x is not measured anew from a metal wall; bottom_span_mm is current domain support extent, not a new wall-width measurement",
    "missing": "unknown, never zero; extrapolated/fill pixels do not count as observed RAW",
    "mass_eligible": False,
    "bottom_mass_allowed": False,
    "limitations": [
        "Brightness and texture changes can be caused by sand, seal, dye, reflections or RAW registration; no material identity is asserted.",
        "The reference and last photograph are measured independently; between them the reference curve is transported by the existing source-bound homography.",
        "Native photographs cannot verify a transition outside their camera footprint; those samples are explicitly RAW-only.",
        "Unmeasured columns are omitted from band areas, with omitted coverage reported; such areas are not full-rig totals.",
    ],
}


def method_document():
    return copy.deepcopy(METHOD)


def project(H, x, y):
    H = np.asarray(H, dtype=np.float64)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if H.shape != (3, 3) or np.any(~np.isfinite(den)) or np.any(np.abs(den) < 1e-10):
        raise ValueError("Bottom report metric homography is invalid")
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
            (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)


def local_metric_mm(H, x, y, factor):
    a, b = project(H, x, y)
    c, d = project(H, x + 1, y)
    e, f = project(H, x, y - 1)
    return np.hypot(c-a, d-b) * factor, np.hypot(e-a, f-b) * factor


def sample_columns(domain, H, origin_xy, mm_factor):
    """Sample the bound bottom edge in physical x, every centimetre."""
    d = np.asarray(domain, dtype=bool)
    occupied = d.any(axis=0)
    xs = np.flatnonzero(occupied)
    if not len(xs):
        raise ValueError("The bottom report requires a nonempty rig domain")
    bottom = d.shape[0] - 1 - np.argmax(d[::-1], axis=0)
    ox, oy = origin_xy
    u, _ = project(H, xs+ox, np.full(len(xs), np.median(bottom[xs])+oy))
    distance = (u-u[0]) * mm_factor
    if np.any(np.diff(distance) <= 0):
        raise ValueError("Bottom report requires monotone physical x across the rig")
    targets = np.arange(5., distance[-1], 10.)
    indices = np.searchsorted(distance, targets)
    indices = np.minimum(indices, len(xs)-1)
    previous = np.maximum(indices-1, 0)
    indices = np.where(np.abs(distance[previous]-targets) <= np.abs(distance[indices]-targets), previous, indices)
    return np.unique(xs[indices]), bottom, distance[-1]


def texture_luminance(log_rgb, observed):
    a = np.asarray(log_rgb)
    observed = np.asarray(observed, bool)
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1:] != observed.shape:
        raise ValueError("Bottom texture requires CHW log RGB on observed support")
    L = np.mean(np.nan_to_num(a, nan=0.), axis=0, dtype=np.float32)
    valid = observed & np.isfinite(a).all(axis=0)
    dx = np.zeros_like(L)
    dy = np.zeros_like(L)
    dx[:, 1:] = np.abs(np.diff(L, axis=1))
    dy[1:] = np.abs(np.diff(L, axis=0))
    texture = (dx + dy) * .5
    valid &= np.roll(valid, 1, axis=1) & np.roll(valid, 1, axis=0)
    valid[0] = False
    valid[:, 0] = False
    texture[~valid] = np.nan
    L[~valid] = np.nan
    return texture, L, valid


def measure_profile(texture, luminance, valid, *, x, bottom_y, roi_y0, pixel_x_mm, pixel_y_mm):
    """One fixed local change-point measurement; never a model prediction."""
    hp = max(2, round(METHOD["profile_half_width_mm"] / pixel_x_mm))
    window = max(2, round(METHOD["comparison_window_mm"] / pixel_y_mm))
    gap = max(1, round(METHOD["comparison_gap_mm"] / pixel_y_mm))
    a, b = max(0, x-hp), min(valid.shape[1], x+hp+1)
    last = min(valid.shape[0]-window-gap-1, int(math.floor(bottom_y-roi_y0)))
    first = max(window+gap, int(math.floor(bottom_y-roi_y0-METHOD["search_height_mm"]/pixel_y_mm)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t = np.nanmedian(texture[:, a:b], axis=1)
        l = np.nanmedian(luminance[:, a:b], axis=1)
    v = np.mean(valid[:, a:b], axis=1)
    rows, scores, contrasts = [], [], []
    for y in range(first, last+1):
        above = slice(y-gap-window, y-gap)
        below = slice(y+gap+1, y+gap+1+window)
        if min(float(np.mean(v[above])), float(np.mean(v[below]))) < METHOD["minimum_valid_share_per_window"]:
            continue
        ta, tb = float(np.nanmedian(t[above])), float(np.nanmedian(t[below]))
        la, lb = float(np.nanmedian(l[above])), float(np.nanmedian(l[below]))
        if not np.isfinite([ta, tb, la, lb]).all() or ta <= 1e-5:
            continue
        ratio, delta = ta / max(tb, 1e-5), lb-la
        if not ((ratio >= 1.75 and delta >= .01) or ratio >= 3.):
            continue
        rows.append(y+roi_y0)
        scores.append(math.log(ratio)+max(0., delta))
        contrasts.append((ratio, delta))
    result = dict(column=int(x), bottom_canvas_y=float(bottom_y), status="unknown: no supported texture/brightness transition", sand_bottom_canvas_y=None, uncertainty_px=None,
                  native_check="not yet checked", profile_rows=list(range(roi_y0, roi_y0+len(t))),
                  texture_profile=np.nan_to_num(t, nan=-1.).tolist(), luminance_profile=np.nan_to_num(l, nan=-1.).tolist())
    if not scores:
        return result
    winner = int(np.argmax(scores))
    rows_array=np.asarray(rows)
    near_indices=np.flatnonzero(np.asarray(scores) >= .9*scores[winner])
    groups=np.split(near_indices,np.flatnonzero(np.diff(rows_array[near_indices])>1)+1)
    selected=next(group for group in groups if winner in group)
    near=rows_array[selected]
    row=float((near.min()+near.max())*.5)
    result.update(status="RAW-supported descriptive transition", sand_bottom_canvas_y=float(row),
                  uncertainty_px=float(max(row-near.min(), near.max()-row)+1),
                  texture_ratio=float(contrasts[winner][0]), log_brightness_rise=float(contrasts[winner][1]), score=float(scores[winner]))
    return result


def native_check(sample, native_measurement, *, pixel_y_mm, native_available):
    result = dict(sample)
    if result["sand_bottom_canvas_y"] is None:
        result["native_check"] = "unknown RAW transition"
    elif not native_available:
        result["native_check"] = "RAW-only: transition outside native photographic support"
    elif native_measurement.get("sand_bottom_canvas_y") is None:
        result.update(status="unknown: native photograph has no corroborating transition", sand_bottom_canvas_y=None, native_check="native texture check ambiguous")
    elif abs(result["sand_bottom_canvas_y"]-native_measurement["sand_bottom_canvas_y"]) * pixel_y_mm > METHOD["native_check_tolerance_mm"]:
        result.update(status="unknown: RAW and native edge disagree", sand_bottom_canvas_y=None, native_check="RAW/native disagreement exceeds 1 mm")
    else:
        result["native_check"] = "native photograph corroborates RAW within 1 mm"
        result["native_sand_bottom_canvas_y"] = native_measurement["sand_bottom_canvas_y"]
    return result


def interpolate_supported(samples, width):
    curve = np.full(width, np.nan)
    good = sorted((s["column"], s["sand_bottom_canvas_y"]) for s in samples if s.get("sand_bottom_canvas_y") is not None)
    all_x = sorted(s["column"] for s in samples)
    nominal = np.median(np.diff(all_x)) if len(all_x)>1 else 0
    for x, y in good:
        curve[int(x)] = y
    for (a, u), (b, v) in zip(good, good[1:]):
        if 0 < b-a <= 1.05*nominal:
            curve[a:b+1] = np.linspace(u, v, b-a+1)
    return curve


def measured_band(domain, sand_bottom, *, row_offset=0):
    d = np.asarray(domain, bool)
    curve = np.asarray(sand_bottom, dtype=float)
    if curve.shape != (d.shape[1],):
        raise ValueError("Measured sand curve has another rig width")
    rows = np.arange(d.shape[0], dtype=float)[:, None] + row_offset
    return d & np.isfinite(curve)[None, :] & (rows > curve[None, :])


def distance_summary(samples, H, origin_xy, mm_factor):
    good = [s for s in samples if s.get("sand_bottom_canvas_y") is not None]
    distances = []
    ox, oy = origin_xy
    for s in good:
        x, y = s["column"]+ox, s["sand_bottom_canvas_y"]+oy
        u, v = project(H, x, y)
        a, b = project(H, x, s["bottom_canvas_y"]+oy)
        value = float(np.hypot(a-u, b-v) * mm_factor)
        s["band_distance_mm"] = value
        distances.append(value)
    return dict(accepted_samples=len(good), sample_count=len(samples), unknown_samples=len(samples)-len(good),
                median_mm=float(np.median(distances)) if distances else None,
                p10_mm=float(np.percentile(distances, 10)) if distances else None,
                p90_mm=float(np.percentile(distances, 90)) if distances else None)


def hump_summary(samples, H, origin_xy, mm_factor):
    good = [s for s in samples if s.get("band_distance_mm") is not None]
    if len(good)<3:
        return dict(status="unknown: insufficient accepted samples", x_range_mm=None, height_mm=None)
    baseline = float(np.percentile([s["band_distance_mm"] for s in good], 10))
    groups, active = [], []
    for s in samples:
        if s.get("band_distance_mm", -np.inf) >= baseline+2.:
            active.append(s)
        elif active:
            groups.append(active); active=[]
    if active:
        groups.append(active)
    if not groups:
        return dict(status="no >=2 mm photographic hump supported", x_range_mm=None, height_mm=0., baseline_mm=baseline)
    selected = max(groups, key=lambda g: (len(g), max(s["band_distance_mm"] for s in g)))
    if len(selected) < 2:
        return dict(status="isolated thicker-band sample; hump extent unknown at 1 cm sampling", x_range_metric_mm=None,
                    height_mm=selected[0]["band_distance_mm"]-baseline, baseline_mm=baseline, sample_count=1,
                    column=int(selected[0]["column"]))
    ox, oy = origin_xy
    xy = [project(H, s["column"]+ox, s["sand_bottom_canvas_y"]+oy) for s in (selected[0], selected[-1])]
    xvals = [float(p[0] * mm_factor) for p in xy]
    return dict(status="descriptive photographic hump; material identity unverified", x_range_metric_mm=xvals,
                columns=[selected[0]["column"], selected[-1]["column"]], height_mm=max(s["band_distance_mm"] for s in selected)-baseline,
                baseline_mm=baseline, sample_count=len(selected))


def transport_curve(curve, *, H_source_to_target, source_origin_xy, target_origin_xy, target_width):
    """Transport only observed curve segments; never bridge missing evidence."""
    good = np.isfinite(curve)
    result = np.full(target_width, np.nan)
    cuts = np.flatnonzero(np.diff(np.r_[False, good, False].astype(np.int8)))
    for a, b in zip(cuts[::2], cuts[1::2]):
        xs = np.arange(a, b, dtype=float) + source_origin_xy[0]
        tx, ty = project(H_source_to_target, xs, curve[a:b]+source_origin_xy[1])
        tx, ty = tx-target_origin_xy[0], ty-target_origin_xy[1]
        order = np.argsort(tx)
        tx, ty = tx[order], ty[order]
        if len(tx)>1 and np.all(np.diff(tx)>0):
            low, high = max(0, math.ceil(tx[0])), min(target_width-1, math.floor(tx[-1]))
            result[low:high+1] = np.interp(np.arange(low, high+1), tx, ty)
        elif len(tx)==1 and 0 <= round(tx[0]) < target_width:
            result[round(tx[0])] = ty[0]
    return result
