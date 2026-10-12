"""Measure added rig strips without changing a phase, mask or photograph."""
from __future__ import annotations

import cv2
import numpy as np

from .phase_pixel_area import validate_pixel_area

SIDES = ("top", "bottom", "left", "right")
CLASSES = ("modelled", "filled_observed", "filled_unobserved")


def added_strip_report(parent_labels, new_labels, pixel_area_mm2, classes, *,
                       inner_polygon_native, photo_top_keep, native_origin_xy,
                       parent_photo_domain=None):
    """Partition additions by their nearest photographed/transported boundary.

    The metal polygon supplies left/right bounds at each row and the bottom
    at each column. The independently photo-registered surface supplies the
    top at each column. Raster-axis distances assign a side; they are not a
    claim of a calibrated physical distance. Ties use top, bottom, left, right.
    Internal support recovery is reported separately from extension beyond
    the parent photo domain, when that domain is supplied. Nothing is masked
    out because of its predicted class.
    """
    old, new = np.asarray(parent_labels), np.asarray(new_labels)
    if (old.dtype != np.uint8 or new.dtype != np.uint8 or old.ndim != 2 or
            old.shape != new.shape or not np.isin(old, (0, 1, 2, 255)).all() or
            not np.isin(new, (0, 1, 2, 255)).all()):
        raise ValueError("Edge QA needs matching actual categorical grids")
    shape = new.shape
    area = validate_pixel_area(pixel_area_mm2, shape)
    masks = {}
    for name in ("rig_domain", *CLASSES):
        value = np.asarray(classes[name])
        if value.dtype != bool or value.shape != shape:
            raise ValueError("Edge classes must share the full phase grid")
        masks[name] = value
    cover = sum(masks[name].astype(np.uint8) for name in CLASSES)
    if (not np.array_equal(cover, masks["rig_domain"].astype(np.uint8)) or
            not np.array_equal(new != 255, masks["rig_domain"])):
        raise ValueError("Edge classes must partition the actual final domain")
    top = np.asarray(photo_top_keep)
    if (top.dtype != bool or top.shape != shape or not top[-1].all() or
            np.any(top[:-1] & ~top[1:])):
        raise ValueError("Photo surface must remove a prefix only")
    polygon = np.asarray(inner_polygon_native, np.float64)
    origin = np.asarray(native_origin_xy, np.float64)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4 or
            origin.shape != (2,) or not np.isfinite(polygon).all() or
            not np.isfinite(origin).all()):
        raise ValueError("Own certified inner-wall polygon/origin required")
    outer = np.zeros(shape, np.uint8)
    cv2.fillPoly(outer, [np.rint((polygon-origin)*256).astype(np.int32)], 1, shift=8)
    outer = outer.astype(bool)
    if np.any(masks["rig_domain"] & (~outer | ~top)):
        raise ValueError("Final domain exceeds its independently defined boundaries")
    photo = None if parent_photo_domain is None else np.asarray(parent_photo_domain)
    if photo is not None and (photo.dtype != bool or photo.shape != shape):
        raise ValueError("Parent photo footprint must share the full phase grid")
    height, width = shape
    left = outer.argmax(axis=1)
    right = width-1-outer[:, ::-1].argmax(axis=1)
    bottom = height-1-outer[::-1].argmax(axis=0)
    surface = top.argmax(axis=0)
    xs = np.arange(width, dtype=np.float32)[None]
    by_side = {name: dict(added_pixels=0, added_domain_cm2=0., co2_cm2=0.,
        aqueous_cm2=0., gas_cm2=0.,
        co2_cm2_by_class={key: 0. for key in CLASSES},
        added_cm2_by_class={key: 0. for key in CLASSES},
        outer_photo_extension_co2_cm2=None if photo is None else 0.,
        internal_support_recovery_co2_cm2=None if photo is None else 0.) for name in SIDES}
    added = masks["rig_domain"] & (old == 255)
    for first in range(0, height, 128):
        last = min(first+128, height)
        ys = np.arange(first, last, dtype=np.float32)[:, None]
        distances = np.stack(np.broadcast_arrays(
            ys-surface[None], bottom[None]-ys,
            xs-left[first:last, None], right[first:last, None]-xs))
        nearest = distances.argmin(axis=0)
        part = added[first:last]
        values = new[first:last]
        weights = area[first:last]
        co2 = (values == 1) | (values == 2)
        def cm2(mask):
            return float(weights[mask].sum(dtype=np.float64)/100)
        for index, name in enumerate(SIDES):
            selected = part & (nearest == index)
            row = by_side[name]
            row["added_pixels"] += int(selected.sum())
            row["added_domain_cm2"] += cm2(selected)
            row["co2_cm2"] += cm2(selected & co2)
            row["aqueous_cm2"] += cm2(selected & (values == 1))
            row["gas_cm2"] += cm2(selected & (values == 2))
            for key in CLASSES:
                classified = selected & masks[key][first:last]
                row["added_cm2_by_class"][key] += cm2(classified)
                row["co2_cm2_by_class"][key] += cm2(classified & co2)
            if photo is not None:
                row["outer_photo_extension_co2_cm2"] += cm2(selected & co2 & ~photo[first:last])
                row["internal_support_recovery_co2_cm2"] += cm2(selected & co2 & photo[first:last])
    direct_area = float(area[added].sum(dtype=np.float64)/100)
    direct_co2 = float(area[added & np.isin(new, (1, 2))].sum(dtype=np.float64)/100)
    if (sum(row["added_pixels"] for row in by_side.values()) != int(added.sum()) or
            not np.isclose(sum(row["co2_cm2"] for row in by_side.values()), direct_co2, rtol=1e-10, atol=1e-8) or
            not np.isclose(sum(row["added_domain_cm2"] for row in by_side.values()), direct_area, rtol=1e-10, atol=1e-8)):
        raise ValueError("Side sums lost or duplicated an added pixel")
    return dict(contract="ffac-rig-added-strip-side-qa-v1", sides=by_side,
        added_domain_cm2=direct_area, added_co2_cm2=direct_co2,
        partition="nearest raster-axis distance to inner left/right/bottom polygon or independent photo top; tie order top,bottom,left,right",
        internal_support_recovery_separated=photo is not None,
        photo_footprint_comparison="source-changed rows require the caller's explicit geometry qualification",
        predictions_or_geometry_changed=False)
