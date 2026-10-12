"""Read-only floor-zone diagnostics; colour and predictions are descriptive evidence.

The physical floor comes from the bound inner-metal polygon, never from colour,
image clipping, or the predicted phase. Distances follow native image columns
through the recorded projective metric mapping; they are not floor normals.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import cv2
import numpy as np

CONTRACT = "ffac-source-bound-bottom-band-diagnostics-v1"
BLIND_RUNS = frozenset(("ac30", "ac49", "ac54"))
METHOD = {
    "contract": CONTRACT,
    "primary_search_height_mm": 20.0,
    "sensitivity_search_heights_mm": [10.0, 30.0],
    "floor": "SHA-bound reference polygon bottom segments (ordered right-wall to bottom to left-wall, dx < 0 and |dy/dx| <= 1), transported by recorded reference-to-frame homography; vertical intersections of those segments only",
    "floor_segment_maximum_abs_slope_in_reference": 1.0,
    "distance": "metric length along native image columns, with the approved 900 mm isotropic scale",
    "pixel_convention": "integer native pixel centres; whole-pixel measured area; floor contact within one native pixel",
    "carryover_proxy": {
        "representation": "exact finite camera-log RAW centre/ring pixels; frozen M_NATURAL, gain 1, nonnegative clamp, clip and sRGB8; OpenCV float sRGB to CIELAB D65",
        "hue_degrees_inclusive": [45.0, 125.0],
        "minimum_chroma": 12.0,
        "lightness_inclusive": [5.0, 95.0],
        "connectivity": 4,
        "selection": "colour-positive pixels inside four-connected observed sand support reaching an observed physical floor pixel within the search zone",
        "strict_contact_sensitivity": "colour-positive pixels themselves four-connected to a colour-positive floor pixel",
        "morphology": "none; no RAW fill, no chroma filter, no adaptive transfer",
        "rig_boundary_guard": "primary colour pixels exclude one pixel adjacent to the inner-rig or hardware boundary (3 by 3 sand-mask erosion); observation connectivity remains unchanged",
        "meaning": "yellow/green photographic proxy, not a finding of incomplete washing or CO2",
    },
    "gap": {
        "selection": "contiguous observed resolved label-0 prefix from the floor, bounded by the search height",
        "current_co2": "saved candidate categorical labels 1 or 2",
        "contact": "first pixel immediately above the label-0 prefix is observed label 1 or 2; no missing or unresolved pixels bridged",
        "empty_column": "right-censored at the search limit, never extended to a distant plume",
    },
    "thickness": "sum of exact metric heights of selected whole pixels per column, with no neutral gaps filled; positive-column and observable-floor zero-inclusive medians; colour span and maximum floor height reported separately",
    "missing_observation": "unknown, never zero and never filled from extrapolated RAW",
    "area": "SHA-bound approved measured pixel-area weights, summed once and divided by 100 for cm2",
    "acceptance_threshold": None,
    "wash_truth_claimed": False,
    "mass_eligible": False,
    "bottom_mass_allowed": False,
    "limitations": [
        "Colour thresholds are a fixed descriptive proxy; sand, lighting, camera calibration, and indicator chemistry can confound it.",
        "A later carryover annotation is not propagated to a pre-injection photograph.",
        "The 900 mm width supplies an isotropic scale; vertical scale was not independently verified.",
        "Pixel rasterisation adds approximately one local pixel to contact and thickness resolution; no numerical confidence interval is claimed.",
        "Saved label 0 is a classifier result, not proof of absence of current CO2.",
        "Unobserved floor columns and interrupted columns have unknown full gap and band thickness.",
    ],
}


def method_document():
    return copy.deepcopy(METHOD)


def bottom_edge_segments(polygon_reference):
    """Select bottom segments from the bound, documented clockwise wall order."""
    p = np.asarray(polygon_reference, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("The floor requires a finite bound inner-metal polygon")
    q = np.roll(p, -1, axis=0)
    delta = q - p
    bottom = (delta[:, 0] < 0) & (np.abs(delta[:, 1]) <= np.abs(delta[:, 0]))
    if not bottom.any():
        raise ValueError("Bound polygon has no bottom segments in its declared wall order")
    return np.stack([p[bottom], q[bottom]], axis=1), np.flatnonzero(bottom)


def floor_intersections(polygon_native, width, origin_x, *, bottom_segments_native=None):
    """Intersect only the bound bottom curve; sidewalls never become floor."""
    p = np.asarray(polygon_native, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("The floor requires a finite bound inner-metal polygon")
    x = np.arange(width, dtype=np.float64) + float(origin_x)
    floor = np.full(width, -np.inf)
    segments = bottom_edge_segments(p)[0] if bottom_segments_native is None else np.asarray(bottom_segments_native, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or not np.isfinite(segments).all():
        raise ValueError("Bound bottom segments have invalid geometry")
    for a, b in segments:
        dx = b[0] - a[0]
        if abs(dx) < 1e-12:
            hit = np.isclose(x, a[0], atol=1e-9, rtol=0)
            floor[hit] = np.maximum(floor[hit], max(a[1], b[1]))
        else:
            t = (x - a[0]) / dx
            hit = (t >= 0) & (t <= 1)
            floor[hit] = np.maximum(floor[hit], a[1] + t[hit] * (b[1] - a[1]))
    floor[~np.isfinite(floor)] = np.nan
    return floor


def _project(H, x, y):
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if np.any(~np.isfinite(den)) or np.any(np.abs(den) < 1e-10):
        raise ValueError("Metric mapping has a horizon in the floor zone")
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
            (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den)


@dataclass
class FloorZone:
    y0: int
    y1: int
    floor_canvas_y: np.ndarray
    base_canvas_y: np.ndarray
    distance_mm: np.ndarray
    sand: np.ndarray
    raw_observed: np.ndarray
    area_mm2: np.ndarray
    modelled: np.ndarray
    geometric_columns: np.ndarray
    floor_in_canvas: np.ndarray
    hardware_floor_columns: np.ndarray
    observable_floor_columns: np.ndarray
    local_pixel_mm: np.ndarray
    pixel_height_mm: np.ndarray

    def selection(self, height_mm):
        if height_mm not in (10.0, 20.0, 30.0):
            raise ValueError("Use the declared 10, 20, or 30 mm search height")
        return self.sand & (self.distance_mm >= 0) & (self.distance_mm <= height_mm)

    def coverage(self, height_mm):
        z = self.selection(height_mm)
        n = int(self.geometric_columns.sum())
        observed = int(self.observable_floor_columns.sum())
        zone_pixels = int(z.sum())
        return {
            "geometric_floor_columns": n,
            "floor_in_canvas_columns": int(self.floor_in_canvas.sum()),
            "hardware_floor_columns_excluded": int(self.hardware_floor_columns.sum()),
            "observable_floor_columns": observed,
            "unknown_or_excluded_floor_columns": n - observed,
            "observable_floor_column_share": observed / n if n else None,
            "floor_zone_sand_pixels": zone_pixels,
            "floor_zone_raw_observed_pixels": int((z & self.raw_observed).sum()),
            "floor_zone_raw_observed_area_cm2": float(self.area_mm2[z & self.raw_observed].sum(dtype=np.float64)) / 100,
            "floor_zone_raw_missing_area_cm2": float(self.area_mm2[z & ~self.raw_observed].sum(dtype=np.float64)) / 100,
            "local_pixel_height_median_mm": float(np.nanmedian(self.local_pixel_mm[self.geometric_columns])) if n else None,
        }


def prepare_floor_zone(*, polygon_native, origin_xy, H_native_to_metric_pixel,
                       metric_pixel_area_mm2, linear_scale, rig_domain,
                       hardware_excluded, raw_observed, pixel_area_mm2,
                       modelled=None, bottom_segments_native=None):
    """Keep a small bottom ROI and the independently bound observation masks."""
    rig = np.asarray(rig_domain, dtype=bool)
    hardware = np.asarray(hardware_excluded, dtype=bool)
    observed = np.asarray(raw_observed, dtype=bool)
    area = np.asarray(pixel_area_mm2)
    if rig.ndim != 2 or any(a.shape != rig.shape for a in (hardware, observed, area)):
        raise ValueError("Floor context masks and measured area must share a 2-D canvas")
    if not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError("Measured pixel areas must be finite and nonnegative")
    if metric_pixel_area_mm2 <= 0 or linear_scale <= 0:
        raise ValueError("The source-bound physical scale must be positive")
    H = np.asarray(H_native_to_metric_pixel, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("A finite 3 by 3 recorded metric homography is required")
    h, w = rig.shape
    ox, oy = map(float, origin_xy)
    floor_native = floor_intersections(polygon_native, w, ox, bottom_segments_native=bottom_segments_native)
    geometric = np.isfinite(floor_native)
    if not geometric.any():
        raise ValueError("Inner-metal polygon has no floor intersections on this canvas")
    x = np.arange(w, dtype=np.float64) + ox
    safe_floor = np.where(geometric, floor_native, 0.0)
    fx, fy = _project(H, x, safe_floor)
    px, py = _project(H, x, safe_floor - 1)
    mm_factor = math.sqrt(float(metric_pixel_area_mm2)) * float(linear_scale)
    step = np.hypot(fx - px, fy - py) * mm_factor
    if np.any(step[geometric] <= 0):
        raise ValueError("Metric floor distance is degenerate")
    floor_canvas = floor_native - oy
    # This bound is conservative for the recorded rig homographies. The actual
    # projective distance, rather than this bound, defines every selected pixel.
    rows = int(math.ceil(30 / float(step[geometric].min()) * 1.1)) + 4
    y0 = max(0, min(h, int(math.floor(np.nanmin(floor_canvas))) - rows))
    y1 = max(y0, min(h, int(math.ceil(np.nanmax(floor_canvas))) + 2))
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] + oy
    qx, qy = _project(H, x[None, :], yy)
    distance = np.hypot(qx - fx[None, :], qy - fy[None, :]) * mm_factor
    den = H[2, 0] * x[None, :] + H[2, 1] * yy + H[2, 2]
    # Exact difference of projective positions at y +/- 0.5, one whole pixel.
    cell_den = den * den - .25 * H[2, 1] ** 2
    cell_dx = (H[0, 1] * den - H[2, 1] * qx * den) / cell_den
    cell_dy = (H[1, 1] * den - H[2, 1] * qy * den) / cell_den
    pixel_height = np.hypot(cell_dx, cell_dy) * mm_factor
    distance[(yy > safe_floor[None, :]) | ~geometric[None, :]] = np.nan
    sand = rig[y0:y1] & ~hardware[y0:y1]
    mod = np.asarray(modelled, dtype=bool) if modelled is not None else rig
    if mod.shape != rig.shape:
        raise ValueError("Modelled mask must share the floor canvas")
    base = np.full(w, -1, dtype=int)
    inside = geometric & (floor_canvas >= 0) & (floor_canvas < h)
    base[inside] = np.floor(floor_canvas[inside]).astype(int)
    floor_hardware = np.zeros(w, dtype=bool)
    floor_observed = np.zeros(w, dtype=bool)
    for col in np.flatnonzero(inside):
        row = base[col]
        # Allow at most one pixel of polygon rasterisation discrepancy. Never
        # substitute the last photograph row for a physical floor outside it.
        if not rig[row, col] and row > 0 and rig[row - 1, col]:
            base[col] = row = row - 1
        floor_hardware[col] = bool(hardware[row, col])
        floor_observed[col] = bool(rig[row, col] and not hardware[row, col] and observed[row, col])
    # Ensure the ROI contains the entire 30 mm zone; no silently truncated bound.
    if y0 > 0 and np.any(sand[0] & geometric & (distance[0] < 30)):
        raise ValueError("Conservative ROI failed to contain the 30 mm floor zone")
    return FloorZone(y0, y1, floor_canvas, base, distance, sand, observed[y0:y1],
                     area[y0:y1], mod[y0:y1], geometric, inside, floor_hardware,
                     floor_observed, np.where(geometric, step, np.nan), pixel_height)


def colour_proxy_mask(rgb_uint8):
    """Apply the declared fixed photographic proxy, with no model inputs."""
    rgb = np.asarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("The photographic proxy requires uint8 RGB Natural pixels")
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    L, a, b = np.moveaxis(lab, -1, 0)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    return (hue >= 45) & (hue <= 125) & (chroma >= 12) & (L >= 5) & (L <= 95)


def fixed_natural_rgb(log_camera_rgb, raw_observed):
    """Use the existing frozen colour transfer without any display repair."""
    from .tools.ffac_track1_display_modes import M_NATURAL
    log = np.asarray(log_camera_rgb)
    observed = np.asarray(raw_observed, dtype=bool)
    if log.dtype != np.float32 or log.shape != (3, *observed.shape):
        raise ValueError("Natural proxy requires source-native float32 camera log RGB")
    if np.any(observed & ~np.isfinite(log).all(0)):
        raise ValueError("Declared observed RAW contains nonfinite camera values")
    camera = np.exp(np.moveaxis(np.where(observed[None], log, 0), 0, -1).astype(np.float64))
    linear = np.clip(camera @ M_NATURAL, 0, 1)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - .055)
    rgb = np.rint(255 * srgb).astype(np.uint8)
    rgb[~observed] = 0
    return rgb


def floor_connected_band(zone, positive_roi, *, height_mm, photographic_observed_roi=None):
    """Measure only connected observed pixels; unobserved columns stay unknown."""
    positive = np.asarray(positive_roi, dtype=bool)
    if positive.shape != zone.sand.shape:
        raise ValueError("Band evidence must share the floor ROI")
    observed = zone.raw_observed.copy()
    if photographic_observed_roi is not None:
        photo = np.asarray(photographic_observed_roi, dtype=bool)
        if photo.shape != observed.shape:
            raise ValueError("Photographic coverage must share the floor ROI")
        observed &= photo
    eligible_cols = zone.observable_floor_columns.copy()
    seeds = np.zeros_like(positive)
    for col in np.flatnonzero(eligible_cols):
        row = zone.base_canvas_y[col] - zone.y0
        eligible_cols[col] = 0 <= row < len(observed) and observed[row, col]
        if eligible_cols[col]:
            seeds[row, col] = True
    candidates = positive & observed & zone.selection(height_mm) & eligible_cols[None, :]
    _, components = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=4)
    seed_ids = np.unique(components[candidates & seeds])
    seed_ids = seed_ids[seed_ids != 0]
    band = np.isin(components, seed_ids) if len(seed_ids) else np.zeros_like(candidates)
    positive_cols = band.any(axis=0)
    thickness = np.zeros(band.shape[1], dtype=np.float64)
    if positive_cols.any():
        thickness[positive_cols] = np.nanmax(np.where(band[:, positive_cols], zone.distance_mm[:, positive_cols], np.nan), axis=0)
    contact_to_missing = band & cv2.dilate((~observed & zone.selection(height_mm)).astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)).astype(bool)
    selected_n = int(eligible_cols.sum())
    result = {
        "search_height_mm": float(height_mm),
        "status": "observed proxy" if selected_n else "unknown: physical floor has no photographic RAW observation",
        "observable_photographic_floor_columns": selected_n,
        "band_columns": int(positive_cols.sum()) if selected_n else None,
        "band_column_share": float(positive_cols.sum() / selected_n) if selected_n else None,
        "band_pixels": int(band.sum()) if selected_n else None,
        "band_area_cm2": float(zone.area_mm2[band].sum(dtype=np.float64)) / 100 if selected_n else None,
        "median_thickness_positive_columns_mm": float(np.median(thickness[positive_cols])) if positive_cols.any() else None,
        "median_thickness_zero_inclusive_mm": float(np.median(thickness[eligible_cols])) if selected_n else None,
        "band_columns_reaching_search_limit": int((positive_cols & (thickness >= height_mm - 2 * zone.local_pixel_mm)).sum()),
        "band_pixels_adjacent_to_missing_raw": int(contact_to_missing.sum()),
        "full_band_area_unknown_when_missing_or_search_censored": bool(contact_to_missing.any() or np.any(positive_cols & (thickness >= height_mm - 2 * zone.local_pixel_mm)) or selected_n < int(zone.geometric_columns.sum())),
        "thickness_definition": "strict contact sensitivity: maximum height of colour-positive floor-connected pixels above the geometric floor",
    }
    return result, band


def colour_in_floor_connected_sand(zone, positive_roi, *, height_mm):
    """Primary proxy: neutral floor rows do not erase observed colour above them."""
    positive = np.asarray(positive_roi, dtype=bool)
    if positive.shape != zone.sand.shape:
        raise ValueError("Band evidence must share the floor ROI")
    observed_sand = zone.selection(height_mm) & zone.raw_observed
    _, components = cv2.connectedComponents(observed_sand.astype(np.uint8), connectivity=4)
    seeds = np.zeros_like(observed_sand)
    for col in np.flatnonzero(zone.observable_floor_columns):
        row = zone.base_canvas_y[col] - zone.y0
        if 0 <= row < len(seeds):
            seeds[row, col] = True
    seed_ids = np.unique(components[seeds & observed_sand])
    seed_ids = seed_ids[seed_ids != 0]
    support = np.isin(components, seed_ids) if len(seed_ids) else np.zeros_like(observed_sand)
    # A whole pixel touching the photographed inner metal boundary can mix
    # yellow hardware with sand. Do not classify that boundary pixel as band.
    colour_sand = cv2.erode(zone.sand.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    band = positive & support & colour_sand
    eligible = zone.observable_floor_columns
    n = int(eligible.sum())
    cols = band.any(axis=0)
    support_cols = support.any(axis=0)
    thickness = np.where(band, zone.pixel_height_mm, 0).sum(axis=0)
    extent = np.zeros(band.shape[1])
    span = np.zeros(band.shape[1])
    if cols.any():
        upper = np.nanmax(np.where(band[:, cols], zone.distance_mm[:, cols] + .5 * zone.pixel_height_mm[:, cols], np.nan), axis=0)
        lower = np.nanmin(np.where(band[:, cols], zone.distance_mm[:, cols] - .5 * zone.pixel_height_mm[:, cols], np.nan), axis=0)
        extent[cols] = upper
        span[cols] = upper - lower
    missing_neighbour = cv2.dilate((~zone.raw_observed & zone.selection(height_mm)).astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)).astype(bool)
    near_missing = band & missing_neighbour
    cap = cols & (extent >= height_mm - 2 * zone.local_pixel_mm)
    result = {
        "search_height_mm": float(height_mm),
        "status": "observed colour proxy in floor-connected sand" if n else "unknown: physical floor has no RAW observation",
        "observable_photographic_floor_columns": n,
        "floor_connected_observed_sand_columns": int(support_cols.sum()),
        "band_columns": int(cols.sum()) if n else None,
        "band_column_share": float(cols.sum() / support_cols.sum()) if support_cols.any() else None,
        "band_column_share_denominator": "columns containing floor-connected observed sand within the search zone",
        "band_pixels": int(band.sum()) if n else None,
        "band_area_cm2": float(zone.area_mm2[band].sum(dtype=np.float64)) / 100 if n else None,
        "median_thickness_positive_columns_mm": float(np.median(thickness[cols])) if cols.any() else None,
        "median_thickness_zero_inclusive_mm": float(np.median(thickness[eligible])) if n else None,
        "median_colour_span_positive_columns_mm": float(np.median(span[cols])) if cols.any() else None,
        "median_maximum_colour_height_above_floor_positive_columns_mm": float(np.median(extent[cols])) if cols.any() else None,
        "band_columns_reaching_search_limit": int(cap.sum()),
        "band_pixels_adjacent_to_missing_raw": int(near_missing.sum()),
        "full_band_area_unknown_when_missing_or_search_censored": bool(near_missing.any() or cap.any() or n < int(zone.geometric_columns.sum())),
        "thickness_definition": "sum of selected physical whole-pixel heights in each column; neutral gaps are not filled",
        "area_interpretation": "observed colour-proxy area inside a bounded connected support; missing and search-censored regions remain unknown",
        "rig_boundary_colour_pixels_excluded": int((positive & support & ~colour_sand).sum()),
    }
    return result, band


def floor_gap(zone, labels_roi, *, height_mm):
    """Bound the observed label-0 prefix; first adjacent CO2 defines contact."""
    labels = np.asarray(labels_roi)
    if labels.shape != zone.sand.shape:
        raise ValueError("Saved phase labels must share the floor ROI")
    search = zone.selection(height_mm)
    gap = np.zeros_like(search)
    statuses = {k: 0 for k in ("co2_at_floor", "positive_gap_with_co2_above", "search_censored", "interrupted_unknown")}
    positive_thickness = []
    contact_thickness = []
    contacts = np.zeros(labels.shape[1], dtype=bool)
    positive_prefix_columns = 0
    for col in np.flatnonzero(zone.observable_floor_columns):
        base = zone.base_canvas_y[col] - zone.y0
        if not (0 <= base < len(search)):
            statuses["interrupted_unknown"] += 1
            continue
        water_rows = []
        status = "search_censored"
        for row in range(base, -1, -1):
            if zone.distance_mm[row, col] > height_mm:
                break
            if not search[row, col] or not zone.raw_observed[row, col] or labels[row, col] not in (0, 1, 2):
                status = "interrupted_unknown"
                break
            if labels[row, col] in (1, 2):
                status = "positive_gap_with_co2_above" if water_rows else "co2_at_floor"
                contacts[col] = True
                contact_thickness.append(float(zone.distance_mm[row, col]) if water_rows else 0.0)
                if water_rows:
                    positive_thickness.append(float(zone.distance_mm[row, col]))
                break
            water_rows.append(row)
        if water_rows:
            gap[water_rows, col] = True
            positive_prefix_columns += 1
        statuses[status] += 1
    n = int(zone.observable_floor_columns.sum())
    total_contact = statuses["co2_at_floor"] + statuses["positive_gap_with_co2_above"]
    co2_bounded_gap = gap & contacts[None, :]
    result = {
        "search_height_mm": float(height_mm),
        "status": "bounded observed classifier diagnostic" if n else "unknown: physical floor has no RAW observation",
        "observable_floor_columns": n,
        "bounded_observed_gap_area_cm2": float(zone.area_mm2[gap].sum(dtype=np.float64)) / 100 if n else None,
        "bounded_observed_gap_pixels": int(gap.sum()) if n else None,
        "gap_area_in_modelled_sand_cm2": float(zone.area_mm2[gap & zone.modelled].sum(dtype=np.float64)) / 100 if n else None,
        "gap_area_in_observed_extension_cm2": float(zone.area_mm2[gap & ~zone.modelled].sum(dtype=np.float64)) / 100 if n else None,
        "co2_bounded_gap_area_cm2": float(zone.area_mm2[co2_bounded_gap].sum(dtype=np.float64)) / 100 if n else None,
        "co2_bounded_gap_pixels": int(co2_bounded_gap.sum()) if n else None,
        "co2_bounded_gap_area_definition": "only contiguous observed label-0 prefixes with immediate observed current-CO2 contact above; search-censored and interrupted columns excluded",
        "columns": statuses,
        "co2_immediately_above_gap_column_share": statuses["positive_gap_with_co2_above"] / n if n else None,
        "co2_at_floor_or_immediately_above_gap_column_share": total_contact / n if n else None,
        "positive_observed_gap_columns": positive_prefix_columns,
        "co2_immediately_above_among_positive_observed_gap_column_share": statuses["positive_gap_with_co2_above"] / positive_prefix_columns if positive_prefix_columns else None,
        "median_gap_thickness_contact_positive_columns_mm": float(np.median(positive_thickness)) if positive_thickness else None,
        "median_gap_thickness_contact_zero_inclusive_mm": float(np.median(contact_thickness)) if contact_thickness else None,
        "unknown_full_gap_columns": int(zone.geometric_columns.sum()) - n + statuses["interrupted_unknown"] + statuses["search_censored"],
        "unknown_or_interrupted_floor_columns": int(zone.geometric_columns.sum()) - n + statuses["interrupted_unknown"],
        "search_censored_columns": statuses["search_censored"],
        "area_interpretation": "observed lower bound in a bounded floor zone; missing columns do not contribute zero",
    }
    return result, gap, contacts


def report_document(frame_rows, run_rows, *, public):
    """Filter blind contributions before building every public aggregate."""
    ids = [(r["run"].casefold(), r["frame_index"]) for r in frame_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate diagnostic frame identities")
    if len({r["run"].casefold() for r in run_rows}) != len(run_rows):
        raise ValueError("Duplicate pre-injection run identities")
    frames = sorted([r for r in frame_rows if not public or r["run"].casefold() not in BLIND_RUNS], key=lambda r: (r["run"], r["frame_index"]))
    runs = sorted([r for r in run_rows if not public or r["run"].casefold() not in BLIND_RUNS], key=lambda r: r["run"])
    primary = [r["gap"]["20"] for r in frames]
    bands = [r["carryover_proxy"]["20"] for r in runs if r.get("carryover_proxy")]
    area = [r["bounded_observed_gap_area_cm2"] for r in primary if r["bounded_observed_gap_area_cm2"] is not None]
    contact_area = [r["co2_bounded_gap_area_cm2"] for r in primary if r["co2_bounded_gap_area_cm2"] is not None]
    band_area = [r["band_area_cm2"] for r in bands if r["band_area_cm2"] is not None]
    return {
        "method": method_document(), "scope": "public nonblind" if public else "internal all selected",
        "frame_count": len(frames), "run_count": len(runs),
        "aggregate": {
            "frames_with_observable_floor": sum(r["observable_floor_columns"] > 0 for r in primary),
            "frames_without_observable_floor": sum(r["observable_floor_columns"] == 0 for r in primary),
            "bounded_observed_gap_area_sum_across_time_cm2": sum(area) if area else None,
            "co2_bounded_gap_area_sum_across_time_cm2": sum(contact_area) if contact_area else None,
            "gap_area_sum_is_one_physical_area": False,
            "preinject_runs_with_observable_photographic_floor": sum(r["observable_photographic_floor_columns"] > 0 for r in bands),
            "preinject_band_observed_area_sum_across_runs_cm2": sum(band_area) if band_area else None,
            "co2_immediately_above_gap_columns": sum(r["columns"]["positive_gap_with_co2_above"] for r in primary),
            "observable_floor_columns_across_frames": sum(r["observable_floor_columns"] for r in primary),
        },
        "runs": runs, "frames": frames,
    }
