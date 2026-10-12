"""Independent quantitative reports for the camera-clock/rig-domain revision.

These reports measure actual categorical outputs and their physical weights.
They do not fit a model, move annotations, approve a scale or activate Studio.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .phase_pixel_area import FIXED_AC28_MM2, sum_pixel_area, validate_pixel_area

TARGET_TIMES_H = (1., 2.25, 4., 12., 24., 48.)


def _labels(values, shape=None):
    array = np.asarray(values)
    if (array.dtype != np.uint8 or array.ndim != 2 or
            (shape is not None and array.shape != shape) or
            not np.isin(array, (0, 1, 2, 255)).all()):
        raise ValueError("Rig QA needs the actual categorical native grid")
    return array


def phase_areas(labels, area_mm2):
    values = _labels(labels)
    area = validate_pixel_area(area_mm2, values.shape)
    return {key: sum_pixel_area(mask, area, unit="cm2") for key, mask in (
        ("water", values == 0), ("aqueous", values == 1),
        ("gas", values == 2), ("co2", np.isin(values, (1, 2))),
        ("resolved", values != 255))}


def frame_area_report(parent_labels, new_labels, area_mm2, classes, *,
                      hidden_chart_mask=None, hardware_masks=None):
    """Report the added strip separately from physically reweighted old sand.

    Both grids use the extended phase canvas; parent255 is not counted as
    water or as porous sand. All three support classes must partition the
    actual rig domain exactly. A hidden chart is an excluded, unfilled area.
    """
    old, new = _labels(parent_labels), _labels(new_labels)
    if old.shape != new.shape:
        raise ValueError("Old/new labels must share explicitly placed canvas coordinates")
    area = validate_pixel_area(area_mm2, new.shape)
    masks = {}
    for name in ("rig_domain", "modelled", "filled_observed", "filled_unobserved"):
        value = np.asarray(classes[name])
        if value.dtype != bool or value.shape != new.shape:
            raise ValueError("Rig QA support classes must match the phase canvas")
        masks[name] = value
    cover = sum(masks[name].astype(np.uint8) for name in
                ("modelled", "filled_observed", "filled_unobserved"))
    if not np.array_equal(cover, masks["rig_domain"].astype(np.uint8)):
        raise ValueError("Rig support classes overlap or leave holes")
    if np.any((new != 255) != masks["rig_domain"]):
        raise ValueError("Final phases do not cover exactly the declared rig domain")
    old_support = old != 255
    added = masks["rig_domain"] & ~old_support
    co2 = np.isin(new, (1, 2))
    co2_area = sum_pixel_area(co2, area, unit="cm2")
    added_co2 = sum_pixel_area(added & co2, area, unit="cm2")
    result = dict(old_fixed_cm2=phase_areas(old, FIXED_AC28_MM2),
        old_reweighted_cm2=phase_areas(old, area), new_cm2=phase_areas(new, area),
        added_domain_cm2=sum_pixel_area(added, area, unit="cm2"),
        added_cm2_by_class={name: sum_pixel_area(added & masks[name], area, unit="cm2")
                           for name in ("modelled", "filled_observed", "filled_unobserved")},
        support_cm2_by_class={name: sum_pixel_area(mask, area, unit="cm2")
                             for name, mask in masks.items()},
        added_co2_cm2=added_co2,
        added_strip_co2_share=(added_co2 / co2_area if co2_area else 0.))
    if hidden_chart_mask is not None:
        hidden = np.asarray(hidden_chart_mask)
        if hidden.dtype != bool or hidden.shape != new.shape or np.any(hidden & masks["rig_domain"]):
            raise ValueError("The hidden chart must be excluded, never filled")
        border = binary_dilation(hidden, structure=np.ones((3, 3), bool)) & ~hidden
        result["hidden_chart"] = dict(area_cm2=sum_pixel_area(hidden, area, unit="cm2"),
            co2_touches_boundary=bool(np.any(co2 & border)),
            touching_boundary_cm2=sum_pixel_area(co2 & border, area, unit="cm2"), filled_pixels=0)
    result["excluded_hardware_cm2"] = {}
    for name, mask in (hardware_masks or {}).items():
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != new.shape or np.any(mask & masks["rig_domain"]):
            raise ValueError("Hardware area is not an actual excluded pixel mask")
        result["excluded_hardware_cm2"][name] = sum_pixel_area(mask, area, unit="cm2")
    return result


def nearest_effect_rows(frame_rows, target_times=TARGET_TIMES_H):
    """Select actual frames by camera time; report offsets instead of interpolation."""
    if not frame_rows:
        raise ValueError("No completed frames exist for this run")
    indices = [row["frame_index"] for row in frame_rows]
    times = np.asarray([row["elapsed_h"] for row in frame_rows], float)
    if len(indices) != len(set(indices)) or not np.isfinite(times).all():
        raise ValueError("Effect rows have duplicate indices or invalid camera times")
    result = []
    for target in target_times:
        row = min(frame_rows, key=lambda item: (abs(item["elapsed_h"] - target), item["frame_index"]))
        result.append(dict(target_h=float(target), camera_offset_h=float(row["elapsed_h"] - target), **row))
    return result


def cohort_statistics(run_values, *, cohort_runs, rig_light_runs, new_light_runs,
                      other_light_runs=(), cv_ddof=0):
    """Use explicit protocol-bound lighting groups; never infer them from colours."""
    cohort, rig, new, other = map(set, (cohort_runs, rig_light_runs, new_light_runs, other_light_runs))
    if (not cohort or not rig or not new or rig & new or rig & other or new & other or
            rig | new | other != cohort or cv_ddof not in (0,1) or len(cohort)<=cv_ddof or
            not cohort <= set(run_values)):
        raise ValueError("Cohort lighting groups must explicitly partition completed runs")
    def values(runs):
        data = np.asarray([run_values[name] for name in sorted(runs)], float)
        if not np.isfinite(data).all() or np.any(data < 0):
            raise ValueError("Cohort area contains invalid values")
        return data
    all_values, rig_values, new_values = values(cohort), values(rig), values(new)
    mean, rig_mean, new_mean = float(all_values.mean()), float(rig_values.mean()), float(new_values.mean())
    return dict(run_count=len(cohort), coefficient_of_variation=(float(all_values.std(ddof=cv_ddof)) / mean if mean else None),
        cv_ddof=cv_ddof, other_light_runs=sorted(other),
        rig_light_mean_cm2=rig_mean, new_light_mean_cm2=new_mean,
        new_to_rig_ratio=(new_mean / rig_mean if rig_mean else None),
        cohort_runs=sorted(cohort), rig_light_runs=sorted(rig), new_light_runs=sorted(new))
