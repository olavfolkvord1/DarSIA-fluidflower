"""Frozen gas-decision composition around L1, with no training or label reads.

The disc is the declared native-pixel disc of the sealed blind scorer. Physical
areas are used only for the inherited gas-fragment minimum and area reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .phase_pixel_area import sum_pixel_area, validate_pixel_area

CONTRACT = "ffac-g3-t2-l1zone-composition-v1"
REFERENCE_POINT_XY = (2786.0, 2897.0)
PHASE_OFFSET_XY = (416.0, 496.0)
RADIUS_PX = 375
ACTIVATION_H = .935


@dataclass
class L1Zone:
    mask: np.ndarray
    point_xy: tuple[float, float]
    active: bool

    def metadata(self):
        return dict(contract=CONTRACT, point_xy=list(self.point_xy), active=self.active,
                    radius_px=RADIUS_PX, activation_h=ACTIVATION_H,
                    reference_point_xy=list(REFERENCE_POINT_XY), phase_offset_xy=list(PHASE_OFFSET_XY),
                    pixel_coordinates="integer phase-grid column,row; sealed scorer disc")


@dataclass
class L1Composition:
    labels: np.ndarray
    t2_without_floor: np.ndarray
    l1_zone_edge_fragment: np.ndarray
    zone: L1Zone
    report: dict


def l1_zone(shape_hw, H_ac28_to_frame_native, elapsed_hours):
    """Transport the declared point with the current source's photographic H."""
    if len(shape_hw) != 2 or any(type(v) not in (int, np.int64, np.int32) or v <= 0 for v in shape_hw):
        raise ValueError("L1 zone requires a positive integer phase-grid shape")
    H = np.asarray(H_ac28_to_frame_native, np.float64)
    if (H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15
            or not np.isfinite(elapsed_hours)):
        raise ValueError("L1 zone requires a finite source-specific homography and camera time")
    mapped = H @ np.array((*REFERENCE_POINT_XY, 1.), np.float64)
    if abs(mapped[2]) < 1e-12:
        raise ValueError("L1 point lies on the photographic projective horizon")
    xy = mapped[:2] / mapped[2] + PHASE_OFFSET_XY
    yy, xx = np.ogrid[:shape_hw[0], :shape_hw[1]]
    mask = (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= RADIUS_PX ** 2
    return L1Zone(mask, tuple(map(float, xy)), bool(elapsed_hours >= ACTIVATION_H))


def zone_edge(mask):
    """Both sides of an in-grid four-neighbour disc boundary (no outer padding)."""
    edge = np.zeros(mask.shape, bool)
    vertical = mask[1:] != mask[:-1]
    edge[1:] |= vertical
    edge[:-1] |= vertical
    horizontal = mask[:, 1:] != mask[:, :-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    return edge


def compose_l1_decisions(base_labels, t2_labels, t2_floor_contact_fill, *,
                         H_ac28_to_frame_native, elapsed_hours, pixel_area_mm2):
    """Combine categorical predictions, undoing only certified T2 floor fills.

    Every input remains unmodified. The only gas-to-aqueous minimum-area
    changes are whole four-connected components touching the active disc edge.
    """
    old, t2, fill = map(np.asarray, (base_labels, t2_labels, t2_floor_contact_fill))
    if (old.ndim != 2 or old.dtype != np.uint8 or t2.dtype != np.uint8
            or t2.shape != old.shape or fill.dtype != bool or fill.shape != old.shape
            or not np.isin(old, (0, 1, 2, 255)).all() or not np.isin(t2, (0, 1, 2, 255)).all()):
        raise ValueError("L1 composition requires exact uint8 categorical grids and boolean floor flag")
    if np.any(fill & ((t2 != 1) | (old != 0))):
        raise ValueError("T2 floor flag includes a non-aqueous or originally CO2 pixel")
    area = validate_pixel_area(pixel_area_mm2, old.shape)
    t = t2.copy()
    t[fill] = 0
    domain_diff = int(np.count_nonzero((old == 255) != (t == 255)))
    union = (old == 1) | (old == 2)
    union_diff = int(np.count_nonzero(union != ((t == 1) | (t == 2))))
    if domain_diff or union_diff:
        raise ValueError(f"Base/T2 frozen domain or CO2 union mismatch: domain={domain_diff}, union={union_diff}")
    zone = l1_zone(old.shape, H_ac28_to_frame_native, elapsed_hours)
    result = old.copy()
    fragment = np.zeros(old.shape, bool)
    withdrawn_components = 0
    if zone.active:
        replace = zone.mask & union
        result[replace] = t[replace]
        gas = result == 2
        edge = zone_edge(zone.mask)
        if np.any(gas & edge):
            count, components = cv2.connectedComponents(gas.astype(np.uint8), connectivity=4)
            sizes = (np.bincount(components.ravel(), minlength=count) * area if np.isscalar(area)
                     else np.bincount(components.ravel(), weights=area.ravel(), minlength=count))
            touching = np.unique(components[gas & edge])
            small = touching[sizes[touching] < 2.5]
            withdrawn_components = int(len(small))
            lookup = np.zeros(count, bool)
            lookup[small] = True
            fragment = lookup[components]
            result[fragment] = 1
    inactive_diff = int(np.count_nonzero(result != old)) if not zone.active else 0
    outside_diff = int(np.count_nonzero((result != old) & ~zone.mask & ~fragment)) if zone.active else 0
    inside_diff = int(np.count_nonzero((result != t) & zone.mask & ~fragment)) if zone.active else 0
    union_diff_after = int(np.count_nonzero(((result == 1) | (result == 2)) != union))
    if inactive_diff or outside_diff or inside_diff or union_diff_after:
        raise AssertionError("Frozen L1 composition identity failed")
    report = dict(**zone.metadata(), elapsed_h=float(elapsed_hours),
        base_t2_domain_differences=domain_diff, base_t2_union_differences=union_diff,
        inactive_base_differences=inactive_diff, active_outside_differences=outside_diff,
        active_inside_differences=inside_diff, co2_union_differences=union_diff_after,
        changed_pixels=int(np.count_nonzero(result != old)), fragment_components=withdrawn_components,
        fragment_pixels=int(fragment.sum()), fragment_area_cm2=sum_pixel_area(fragment, area, unit="cm2"),
        minimum_mm2=2.5, connectivity=4, area_measure="unchanged base minimum-area asset; reporting scale is separate",
        training=False, labels_read=False, model_inference_performed=False)
    return L1Composition(result, t, fragment, zone, report)
