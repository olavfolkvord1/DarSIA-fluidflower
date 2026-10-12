"""Flagged floor-contact completion after frozen CO2 and gas interpretation.

Only an intact no-CO2 column below a sufficiently large aqueous component can
be filled. This is a declared interpretation rule, not measured concentration.
The bound diagnostic floor geometry and approved per-pixel metric are reused.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .phase_ffac_bottom_band_diagnostics import FloorZone, bottom_edge_segments, prepare_floor_zone

CONTRACT = "ffac-t2-floor-contact-v1"
METHOD = {
    "contract": CONTRACT,
    "order": "after gas thresholds, 4-connected minimum area, top-surface clip and barrier rule B",
    "elapsed_time": "strictly after injection start (elapsed_h > 0)",
    "floor": "same SHA-bound reference bottom segments and per-native-column projective metric as the bottom diagnostic; sidewalls excluded",
    "maximum_gap_mm": 20.0,
    "minimum_trigger_aqueous_component_area_mm2": 2.5,
    "component_connectivity": 4,
    "support": "observed or explicitly filled, inside the rig domain, excluding hardware, missing and unresolved pixels",
    "trigger": "first CO2 pixel above the intact floor gap is aqueous, never gas",
    "label_change": "no-CO2 to aqueous only; every changed pixel flagged floor_contact_fill",
    "mass_eligible": False,
    "bottom_mass_allowed": False,
    "future_concentration_policy": "nearest measured aqueous pixel above in the same native column; concentration imputed and flagged, not measured; no concentration computed by this rule",
    "future_preinjection_balance_policy": "indicator colour present before injection from the fixed 20 mm diagnostic proxy is reported separately and subtracted from injected-CO2 balance; no mass computed here",
}


@dataclass
class FloorContactResult:
    labels: np.ndarray
    floor_contact_fill: np.ndarray
    trigger_canvas_y: np.ndarray
    report: dict


def prepare_bound_floor_zone(frame_receipt, classes, pixel_area_mm2, observed_support):
    """Reuse the exact diagnostic bottom transport and approved measured metric.

    The caller verifies the frame receipt and input asset SHA bindings before
    loading them. This helper also checks the recorded polygon transport and
    physical approval; it never estimates new geometry from a phase map.
    """
    revision = frame_receipt["rig_domain_revision"]
    parameters = revision["parameters"]
    scale = parameters["physical_scale_revision"]
    if not scale.get("physical_scale_approved") or scale.get("inner_width_mm") != 900:
        raise ValueError("Floor contact requires the approved measured 900 mm physical scale")
    geometry = frame_receipt["binding"]["source"]["rig_geometry"]
    polygon = np.asarray(geometry["reference_rig"]["polygon_reference_native"], np.float64)
    H = np.asarray(geometry["H_reference_rig_to_frame_native"], np.float64)
    homogeneous = np.c_[polygon, np.ones(len(polygon))] @ H.T
    transported = homogeneous[:, :2] / homogeneous[:, 2:]
    declared = np.asarray(geometry["frame_inner_polygon_native"], np.float64)
    if transported.shape != declared.shape or not np.allclose(transported, declared, atol=1e-6, rtol=0):
        raise ValueError("Floor-contact polygon transport differs from the bound geometry")
    segments, indices = bottom_edge_segments(polygon)
    p = np.concatenate([segments, np.ones((*segments.shape[:2], 1))], axis=2) @ H.T
    segments_native = p[..., :2] / p[..., 2:]
    zone = prepare_floor_zone(polygon_native=declared, origin_xy=revision["native_origin_xy"],
        H_native_to_metric_pixel=parameters["H_frame_native_to_metric_pixel"],
        metric_pixel_area_mm2=parameters["metric_pixel_area_mm2"], linear_scale=scale["linear_factor"],
        rig_domain=classes["rig_domain"], hardware_excluded=classes["hardware_excluded"],
        raw_observed=observed_support, pixel_area_mm2=pixel_area_mm2,
        modelled=classes["modelled"], bottom_segments_native=segments_native)
    return zone, {"reference_segment_indices": indices.tolist(),
        "selection": "same source-bound diagnostic bottom segments; sidewalls excluded",
        "maximum_abs_reference_slope": 1.0,
        "measured_support_intervals_native": geometry["reference_rig"].get("measured_support_intervals_native"),
        "extrapolation": geometry["reference_rig"].get("extrapolation"),
        "physical_scale_revision": scale}


def apply_floor_contact(labels, *, elapsed_hours, zone: FloorZone,
                        observed_support, filled_support, unresolved,
                        pixel_area_mm2, maximum_gap_mm=20.0,
                        minimum_aqueous_area_mm2=2.5):
    """Return a new label array and exact flag; never mutate input labels.

    ``zone`` is prepared from the bound physical bottom segments. Support masks
    and measured area share the full label canvas. Labels 0/1/2 mean no-CO2,
    aqueous and gas; any other code interrupts a column. Whole-pixel metric
    heights bound the gap length; exact measured area weights the component.
    """
    source = np.asarray(labels)
    shape = source.shape
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("Floor-contact labels must be a two-dimensional integer array")
    masks = [np.asarray(v) for v in (observed_support, filled_support, unresolved)]
    if any(v.dtype != bool or v.shape != shape for v in masks):
        raise ValueError("Floor-contact support masks must be boolean on the exact label canvas")
    area = np.asarray(pixel_area_mm2)
    if area.shape != shape or not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError("Floor-contact requires finite nonnegative measured pixel areas")
    if (zone.sand.shape != (zone.y1 - zone.y0, shape[1])
            or not 0 <= zone.y0 <= zone.y1 <= shape[0]):
        raise ValueError("Floor zone belongs to a different label canvas")
    if (not np.isfinite(elapsed_hours) or maximum_gap_mm != 20.0
            or minimum_aqueous_area_mm2 != 2.5):
        raise ValueError("Use finite elapsed time and the fixed 20 mm / 2.5 mm2 rule")
    if not np.array_equal(area[zone.y0:zone.y1], zone.area_mm2):
        raise ValueError("Component and floor areas must use the same bound measured metric")
    observed, filled, unknown = masks
    supported = (observed | filled) & ~unknown
    full_fill = np.zeros(shape, dtype=bool)
    triggers = np.full(shape[1], -1, dtype=np.int32)
    eligible = zone.sand & supported[zone.y0:zone.y1]
    floor_columns = np.zeros(shape[1], dtype=bool)
    raw_floor_columns = np.zeros(shape[1], dtype=bool)
    for col in np.flatnonzero(zone.floor_in_canvas):
        row = int(zone.base_canvas_y[col])
        if zone.y0 <= row < zone.y1:
            floor_columns[col] = bool(eligible[row-zone.y0, col])
            raw_floor_columns[col] = bool(eligible[row-zone.y0, col] and observed[row, col])
    reasons = {key: 0 for key in ("filled", "co2_at_floor", "gas_first", "aqueous_too_small",
                                  "interrupted_unknown", "search_censored", "gap_exceeds_20mm")}
    if elapsed_hours > 0:
        # Only eligible aqueous pixels participate; missing or unresolved pixels
        # cannot connect two components for the minimum measured area rule.
        aqueous = (source == 1) & supported
        count, components = cv2.connectedComponents(aqueous.astype(np.uint8), connectivity=4)
        component_area = np.bincount(components.ravel(), weights=area.ravel(), minlength=count)
        component_area[0] = 0
        roi = source[zone.y0:zone.y1]
        for col in np.flatnonzero(floor_columns):
            base = int(zone.base_canvas_y[col] - zone.y0)
            if roi[base, col] in (1, 2):
                reasons["co2_at_floor"] += 1
                continue
            selected = []
            length_mm = 0.0
            reason = "search_censored"
            for row in range(base, -1, -1):
                distance = zone.distance_mm[row, col]
                if not math.isfinite(distance) or distance > maximum_gap_mm:
                    break
                if not eligible[row, col] or roi[row, col] not in (0, 1, 2):
                    reason = "interrupted_unknown"
                    break
                value = int(roi[row, col])
                if value == 0:
                    selected.append(row)
                    length_mm += float(zone.pixel_height_mm[row, col])
                    if length_mm > maximum_gap_mm + 1e-10:
                        reason = "gap_exceeds_20mm"
                        break
                    continue
                if value == 2:
                    reason = "gas_first"
                    break
                component = components[row + zone.y0, col]
                if component_area[component] + 1e-10 < minimum_aqueous_area_mm2:
                    reason = "aqueous_too_small"
                    break
                if selected:
                    full_fill[np.asarray(selected) + zone.y0, col] = True
                    triggers[col] = row + zone.y0
                    reason = "filled"
                else:
                    reason = "co2_at_floor"
                break
            reasons[reason] += 1
    output = source.copy()
    output[full_fill] = 1
    changed = output != source
    co2_before = (source == 1) | (source == 2)
    co2_after = (output == 1) | (output == 2)
    filled_columns = triggers >= 0
    denominator = int(floor_columns.sum())
    observed_denominator = int(raw_floor_columns.sum())
    before_injection = elapsed_hours <= 0
    report = {
        "contract": CONTRACT, "elapsed_h": float(elapsed_hours),
        "maximum_gap_mm": maximum_gap_mm,
        "minimum_aqueous_component_area_mm2": minimum_aqueous_area_mm2,
        "filled_pixels": int(full_fill.sum()),
        "filled_area_cm2": float(area[full_fill].sum(dtype=np.float64)) / 100,
        "floor_columns_with_observed_or_filled_support": denominator,
        "observable_floor_columns": observed_denominator,
        "filled_floor_columns": int(filled_columns.sum()),
        "filled_supported_floor_column_share": int(filled_columns.sum()) / denominator if denominator else None,
        "filled_observable_floor_columns": int((filled_columns & raw_floor_columns).sum()),
        "filled_observable_floor_column_share": int((filled_columns & raw_floor_columns).sum()) / observed_denominator if observed_denominator else None,
        "columns": reasons,
        "unflagged_label_changes": int((changed & ~full_fill).sum()),
        "co2_union_changes_outside_floor_contact_fill": int(((co2_before != co2_after) & ~full_fill).sum()),
        "filled_under_gas_pixels": 0,  # Every stored trigger was verified aqueous above.
        "preinjection_fill_pixels": int(full_fill.sum()) if before_injection else 0,
        "mass_eligible": False, "bottom_mass_allowed": False,
    }
    if not np.array_equal(changed, full_fill) or np.any(source[full_fill] != 0):
        raise AssertionError("Floor-contact output altered a non-gap pixel")
    if np.any(source[triggers[filled_columns], np.flatnonzero(filled_columns)] != 1):
        raise AssertionError("Gas cannot trigger the floor-contact rule")
    return FloorContactResult(output, full_fill, triggers, report)
