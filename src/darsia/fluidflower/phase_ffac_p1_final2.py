"""Locked final-round FF_AC categorical rules; pure arrays, never inference.

The reference scripts define distances on the nominal AC28 phase raster.
Approved variable pixel areas weight reports; they never replace its spacing.
Photographic registration, curve construction, and file binding belong to the
producer. This module changes only the four explicitly recorded step masks.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy import ndimage as cpu_ndimage

from .phase_ffac_p1_final import select_backend

CONTRACT = "ffac-g3-t4c-p1-final2-candidate-v1"
SPEC_SHA256 = "e843e61751b795984550cfe853548ea7ca4e50364a5cb44681621d485abd2c0b"
NOMINAL_AREA_MM2 = .027053
MM_PER_PIXEL = NOMINAL_AREA_MM2 ** .5
COLUMN_GATE_PIXELS = int(30 / MM_PER_PIXEL)
FLOOR_BAND_PIXELS = int(20 / MM_PER_PIXEL)
SHORT_FILL_PIXELS = int(1.5 / MM_PER_PIXEL)
RULE = {
    "order": ["A_sand_floor", "B_floor_connectivity", "C_floor_contact", "D_gas_envelope"],
    "nominal_pixel_area_mm2": NOMINAL_AREA_MM2,
    "nominal_mm_per_phase_pixel": MM_PER_PIXEL,
    "metric": "fixed nominal AC28 raster; measured variable areas weight cm2 only",
    "acting_columns": "abs(own last domain row - median last domain row) <= floor(30/MM)",
    "column_gate_mm": 30., "column_gate_pixels": COLUMN_GATE_PIXELS,
    "floor_band_mm": 20., "floor_band_pixels": FLOOR_BAND_PIXELS,
    "floor_line": "last corrected rig-domain row in each column; unresolved labels do not shift it",
    "floor_height": "own-column floor row minus pixel row",
    "connectivity_disc_radius_mm": 1.5,
    "connectivity_disc_radius_pixels": 1.5 / MM_PER_PIXEL,
    "connectivity": 8,
    "group_attributes": "original CO2 pixels only, not dilated pixels",
    "labels_dilated": False,
    "short_fill_limit_mm": 1.5, "short_fill_limit_pixels": SHORT_FILL_PIXELS,
    "carryover_tolerance": "Chebyshev one pixel; one 3x3 square dilation with zero border",
    "fill_reason_precedence": "a (short supported run), then b (fully inside tolerated carryover)",
    "fill_time": "elapsed_h >0", "fill_under_gas": False,
    "injection_end_h": [0.935, 2.389],
    "envelope_selection": "P1-final gas after A, inclusive through first frame at/after each end",
    "envelope_radius_mm": 5., "envelope_radius_pixels": 5. / MM_PER_PIXEL,
    "distance_transform": "exact float64 Euclidean EDT; empty union has empty envelope",
    "envelope_time": "strictly later frame index than the corresponding inclusive union frame",
    "size_filter_reapplied": False,
    "overlap_policy": "each final changed pixel belongs to exactly one step; unchanged stage restorations are reported",
    "training": False, "inference": False,
}


@dataclass
class FinalRoundResult:
    labels: np.ndarray
    step_masks: dict[str, np.ndarray]
    previous_labels: dict[str, np.ndarray]
    fill_reason_a: np.ndarray
    fill_reason_b: np.ndarray
    qa: dict


def _bool(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be a bool array on the exact phase grid")
    return array


def _labels(value):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2 or not all(array.shape):
        raise ValueError("Parent labels must be a nonempty two-dimensional uint8 grid")
    if not set(np.unique(array)).issubset({0, 1, 2, 255}):
        raise ValueError("Parent labels contain an unsupported category")
    return array


def _modules(backend):
    if backend == "cuda":
        import cupy as xp
        import cupyx.scipy.ndimage as ndi
        return xp, ndi
    return np, cpu_ndimage


def _numpy(value, xp):
    return xp.asnumpy(value) if xp is not np else np.asarray(value).copy()


def _is_gpu_failure(error):
    return type(error).__module__.startswith(("cupy", "cuda"))


def _edt(mask, xp, ndi):
    # Only distances are used. Equal-distance feature-index tie choices cannot
    # change the categorical result. Keep double precision on both backends.
    return (ndi.distance_transform_edt(mask, float64_distances=True)
            if xp is not np else ndi.distance_transform_edt(mask))


def _dilate(mask, radius_pixels, xp, ndi):
    if not bool(mask.any()):
        return xp.zeros(mask.shape, dtype=bool)
    return _edt(~mask, xp, ndi) <= radius_pixels


def _label(mask, xp, ndi):
    if xp is not np:
        labels, count = ndi.label(mask, structure=xp.ones((3, 3), dtype=xp.int8))
        return labels, int(count)
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, int(count)-1


def apply_floor_boundary(labels, below_floor_mask):
    """Apply A on its frozen producer-supplied boundary, without other rules."""
    source = _labels(labels)
    below = _bool(below_floor_mask, source.shape, "below_floor_mask")
    result = source.copy()
    result[below] = 255
    return result


def _envelopes_core(union1, union2, actual):
    xp, ndi = _modules(actual)
    unions = (xp.asarray(union1), xp.asarray(union2))
    return tuple(_numpy(_dilate(mask, 5. / MM_PER_PIXEL, xp, ndi), xp) for mask in unions)


def build_envelopes(union1, union2, backend="auto"):
    """Return two exact 5 mm discs of the externally bound inclusive unions."""
    first = np.asarray(union1)
    if first.dtype != bool or first.ndim != 2 or not all(first.shape):
        raise ValueError("Gas unions must be nonempty bool grids")
    second = _bool(union2, first.shape, "union2")
    wall, cpu = time.perf_counter(), time.process_time()
    actual = select_backend(backend)
    gpu, failure = 0., None
    try:
        if actual == "cuda":
            import cupy as cp
            begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record()
        env1, env2 = _envelopes_core(first, second, actual)
        if actual == "cuda":
            end.record(); end.synchronize(); gpu = float(cp.cuda.get_elapsed_time(begin, end))/1000.
    except Exception as error:
        if actual != "cuda" or not _is_gpu_failure(error):
            raise
        failure = f"{type(error).__name__}: {error}"
        actual = "cpu"
        env1, env2 = _envelopes_core(first, second, actual)
    audit = dict(backend=actual, requested_backend=backend, cuda_fallback_reason=failure,
        union1_pixels=int(first.sum()), union2_pixels=int(second.sum()),
        envelope1_pixels=int(env1.sum()), envelope2_pixels=int(env2.sum()),
        radius_mm=5., radius_pixels=5. / MM_PER_PIXEL,
        empty_union_envelope_policy="empty, not an implicit outside-grid source",
        cpu_seconds=time.process_time()-cpu, gpu_seconds=gpu,
        wall_seconds=time.perf_counter()-wall)
    return env1, env2, audit


def _core(parent, below, support, carryover, l1, env1, env2,
          elapsed_h, frame_index, u1_index, u2_index, corrected_domain, actual):
    xp, ndi = _modules(actual)
    source = xp.asarray(parent).copy()
    height, width = source.shape
    rows = xp.arange(height, dtype=xp.int32)[:, None]
    columns = xp.arange(width, dtype=xp.int32)

    # A. Boundary masks can include already-excluded pixels, but flags record
    # only actual label changes; the producer separately accounts new classes.
    below = xp.asarray(below)
    mask_a = below & (source != 255)
    prior_a = xp.where(mask_a, source, 0).astype(xp.uint8)
    source[below] = 255
    domain = ((source != 255) if corrected_domain is None
              else xp.asarray(corrected_domain) & ~below)
    has = domain.any(axis=0)
    floor = xp.where(has, height-1-domain[::-1].argmax(axis=0), -1).astype(xp.int32)
    median = float(xp.median(floor[has])) if bool(has.any()) else None
    acting = has & (xp.abs(floor-(median if median is not None else -1)) <= COLUMN_GATE_PIXELS)
    floor_height = floor[None, :]-rows

    # B. Dilated components decide grouping; all component attributes and flags
    # are measured exclusively on the original CO2 pixels.
    co2 = (source == 1) | (source == 2)
    dilated = _dilate(co2, 1.5 / MM_PER_PIXEL, xp, ndi)
    components, count = _label(dilated, xp, ndi)
    touches = co2 & acting[None, :] & (floor_height >= 0) & (floor_height <= FLOOR_BAND_PIXELS)
    deeper = co2 & (floor_height > FLOOR_BAND_PIXELS)
    def present(mask):
        selected = components[mask]
        return (xp.bincount(selected, minlength=count+1)>0 if selected.size
                else xp.zeros(count+1, dtype=bool))
    remove_ids = present(touches) & ~present(deeper)
    remove_ids[0] = False
    mask_b = co2 & remove_ids[components]
    prior_b = xp.where(mask_b, source, 0).astype(xp.uint8)
    kept_co2 = co2 & ~mask_b
    removed_ids = _numpy(xp.flatnonzero(remove_ids), xp).tolist()
    group_distances = []
    if removed_ids and bool(kept_co2.any()):
        distances = _edt(~kept_co2, xp, ndi)*MM_PER_PIXEL
        # CuPy minimum.at currently rounds float64 input through float32. The
        # ndimage reduction preserves double precision and original CO2-only
        # selection on both backends.
        minima = ndi.minimum(xp.where(mask_b, distances, xp.inf), labels=components,
                             index=xp.asarray(removed_ids))
        group_distances = _numpy(minima, xp).tolist()
    source[mask_b] = 0

    # C. Only a complete supported label-0 column run is eligible. No lateral
    # propagation, no fill under gas, and no unobserved carryover is invented.
    co2_after_b = (source == 1) | (source == 2)
    lowest = xp.max(xp.where(co2_after_b, rows, -1), axis=0)
    spans = floor-lowest
    trigger = source[xp.maximum(lowest, 0), columns]
    eligible = acting & (lowest >= 0) & (spans > 0) & (trigger == 1)
    run = (rows > lowest[None, :]) & (rows <= floor[None, :])
    intact = ~xp.any(run & ((source != 0) | ~xp.asarray(support)), axis=0)
    short = eligible & intact & (spans <= SHORT_FILL_PIXELS)
    colour = xp.zeros(width, dtype=bool)
    if carryover is not None and elapsed_h > 0:
        tolerated = ndi.binary_dilation(xp.asarray(carryover), structure=xp.ones((3, 3), dtype=bool),
                                       iterations=1, border_value=0)
        colour = eligible & intact & ~xp.any(run & ~tolerated, axis=0)
    if elapsed_h <= 0:
        short[:] = False
        colour[:] = False
    reason_a = run & short[None, :]
    reason_b = run & (colour & ~short)[None, :]
    mask_c = reason_a | reason_b
    prior_c = xp.where(mask_c, source, 0).astype(xp.uint8)
    source[mask_c] = 1

    # D. Envelopes come from P1-final after A, not this frame's changed gas.
    gas = source == 2
    mask_d = xp.zeros(source.shape, dtype=bool)
    if frame_index > u1_index:
        mask_d |= gas & xp.asarray(l1) & ~xp.asarray(env1)
    if frame_index > u2_index:
        mask_d |= gas & ~xp.asarray(env2)
    prior_d = xp.where(mask_d, source, 0).astype(xp.uint8)
    source[mask_d] = 1

    diagnostics = dict(
        floor_median_row=median, acting_columns=int(acting.sum()), excluded_columns=int((~acting).sum()),
        co2_connectivity_groups=count, floor_connectivity_removed_groups=len(removed_ids),
        floor_connectivity_removed_group_min_distance_to_kept_co2_mm=group_distances,
        floor_connectivity_distance_status=("known" if group_distances else
            "unknown: no kept CO2" if removed_ids else "not applicable: no removed group"),
        floor_connectivity_distance_median_mm=float(np.median(group_distances)) if group_distances else None,
        floor_connectivity_distance_max_mm=float(np.max(group_distances)) if group_distances else None,
        floor_connectivity_removed_gas_pixels=int((mask_b & (prior_b == 2)).sum()),
        fill_reason_a_columns=int(short.sum()), fill_reason_b_columns=int((colour & ~short).sum()),
        carryover_mask_known=carryover is not None,
        filled_under_gas_pixels=int((mask_c & (trigger == 2)[None, :]).sum()),
        fill_invalid_trigger_columns=int(((mask_c.any(axis=0)) & (trigger != 1)).sum()),
        fill_unsupported_pixels=int((mask_c & ~xp.asarray(support)).sum()),
        preinjection_fill_pixels=int(mask_c.sum()) if elapsed_h <= 0 else 0,
    )
    arrays = (source, mask_a, mask_b, mask_c, mask_d, prior_a, prior_b, prior_c, prior_d, reason_a, reason_b)
    return tuple(_numpy(array, xp) for array in arrays), diagnostics


def apply_final_round(parent_labels, below_floor_mask, supported_mask, carryover_mask_orNone,
                      l1_zone, env1, env2, elapsed_h, frame_index, u1_index, u2_index,
                      pixel_area_mm2, backend="auto", corrected_domain_mask=None):
    """Apply A–D literally; changed-pixel identity conflicts fail QA."""
    parent = _labels(parent_labels)
    shape = parent.shape
    below = _bool(below_floor_mask, shape, "below_floor_mask")
    supported = _bool(supported_mask, shape, "supported_mask")
    carryover = None if carryover_mask_orNone is None else _bool(carryover_mask_orNone, shape, "carryover_mask")
    corrected_domain = (None if corrected_domain_mask is None else
                        _bool(corrected_domain_mask, shape, "corrected_domain_mask"))
    l1, first, second = (_bool(value, shape, name) for value, name in
                        ((l1_zone, "l1_zone"), (env1, "env1"), (env2, "env2")))
    area = np.asarray(pixel_area_mm2)
    if (area.ndim not in (0, 2) or (area.ndim == 2 and area.shape != shape)
            or not np.issubdtype(area.dtype, np.number) or not np.isfinite(area).all() or np.any(area < 0)):
        raise ValueError("Pixel areas must be finite nonnegative weights on the exact phase grid")
    if (not np.isfinite(elapsed_h) or any(type(value) is not int or value < 0
            for value in (frame_index, u1_index, u2_index)) or u2_index < u1_index):
        raise ValueError("Frame time and inclusive injection-end indices must be valid and ordered")
    wall, cpu = time.perf_counter(), time.process_time()
    actual, gpu_seconds, failure = select_backend(backend), 0., None
    arguments = (parent, below, supported, carryover, l1, first, second,
                 float(elapsed_h), frame_index, u1_index, u2_index, corrected_domain)
    try:
        if actual == "cuda":
            import cupy as cp
            begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record()
        arrays, audit = _core(*arguments, actual)
        if actual == "cuda":
            end.record(); end.synchronize(); gpu_seconds = float(cp.cuda.get_elapsed_time(begin, end))/1000.
    except Exception as error:
        if actual != "cuda" or not _is_gpu_failure(error):
            raise
        failure = f"{type(error).__name__}: {error}"
        actual = "cpu"
        arrays, audit = _core(*arguments, actual)
    labels, *parts = arrays
    masks = dict(zip("ABCD", parts[:4]))
    previous = dict(zip("ABCD", parts[4:8]))
    reason_a, reason_b = parts[8:]
    overlaps = {a+b: int(np.count_nonzero(masks[a] & masks[b]))
                for i, a in enumerate("ABCD") for b in "ABCD"[i+1:]}
    flagged = np.logical_or.reduce(list(masks.values()))
    changed = parent != labels
    overlap_changed = {a+b: int(np.count_nonzero(masks[a] & masks[b] & changed))
                for i, a in enumerate("ABCD") for b in "ABCD"[i+1:]}
    gas_changes = (parent == 2) != (labels == 2)
    co2_changes = np.isin(parent, (1, 2)) != np.isin(labels, (1, 2))
    def measure(mask):
        return (float(np.count_nonzero(mask)*area)/100. if area.ndim == 0
                else float(np.sum(area, where=mask, dtype=np.float64))/100.)
    qa = dict(audit, backend=actual, requested_backend=backend, cuda_fallback_reason=failure,
        step_pixels={name: int(mask.sum()) for name, mask in masks.items()},
        step_area_cm2={name: measure(mask) for name, mask in masks.items()},
        step_previous_aqueous_cm2={name: measure(mask & (previous[name] == 1)) for name, mask in masks.items()},
        step_previous_gas_cm2={name: measure(mask & (previous[name] == 2)) for name, mask in masks.items()},
        fill_reason_a_cm2=measure(reason_a), fill_reason_b_cm2=measure(reason_b),
        mask_overlaps_pixels=overlaps, mask_overlaps_on_changed_pixels=overlap_changed,
        unchanged_stage_restoration_pixels=int((flagged & ~changed).sum()),
        changed_pixels=int(changed.sum()), flag_not_final_changed_pixels=int((flagged & ~changed).sum()),
        changed_unflagged_pixels=int((changed & ~flagged).sum()),
        labels_outside_masks_identical=bool(np.array_equal(parent[~flagged], labels[~flagged])),
        gas_changed_outside_a_d_pixels=int((gas_changes & ~(masks["A"] | masks["D"])).sum()),
        co2_union_changed_outside_a_b_c_pixels=int((co2_changes & ~(masks["A"] | masks["B"] | masks["C"])).sum()),
        domain_changed_outside_a_pixels=int(((parent == 255) != (labels == 255))[~masks["A"]].sum()),
        floor_boundary_matches=bool(np.array_equal(labels == 255, (parent == 255) | below)),
        fill_flags_are_new_aqueous=bool(np.all(previous["C"][masks["C"]] == 0) and np.all(labels[masks["C"]] == 1)),
        gas_flags_are_gas_to_aqueous=bool(np.all(previous["D"][masks["D"]] == 2) and np.all(labels[masks["D"]] == 1)),
        size_filter_reapplied=False, training=False, inference=False,
        cpu_seconds=time.process_time()-cpu, gpu_seconds=gpu_seconds,
        wall_seconds=time.perf_counter()-wall)
    qa["all_pass"] = bool(
        not any(overlap_changed.values()) and not (changed & ~flagged).any()
        and qa["floor_connectivity_removed_gas_pixels"] == 0
        and qa["labels_outside_masks_identical"] and qa["floor_boundary_matches"]
        and qa["gas_changed_outside_a_d_pixels"] == 0
        and qa["co2_union_changed_outside_a_b_c_pixels"] == 0
        and qa["domain_changed_outside_a_pixels"] == 0
        and qa["filled_under_gas_pixels"] == 0 and qa["fill_invalid_trigger_columns"] == 0
        and qa["fill_unsupported_pixels"] == 0 and qa["preinjection_fill_pixels"] == 0
        and qa["fill_flags_are_new_aqueous"] and qa["gas_flags_are_gas_to_aqueous"])
    return FinalRoundResult(labels, masks, previous, reason_a, reason_b, qa)
