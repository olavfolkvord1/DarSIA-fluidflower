"""Read-only, fixed-definition QA for the one-pixel old-domain seam.

The domain is the recorded original photo-sand AND feature-support mask.
Neither labels nor support are inferred from a rendered photograph. This
module reports measurements; it declares no scientific acceptance threshold.
"""
from __future__ import annotations

import cv2
import numpy as np

from .phase_pixel_area import sum_pixel_area, validate_pixel_area

CONTRACT = "ffac-old-domain-one-pixel-ring-qa-v1"
RADIUS_PX = 6
CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def _mask(value, shape, name):
    result = np.asarray(value)
    if result.dtype != bool or result.shape != shape:
        raise ValueError(f"{name} must be a bool mask on the exact categorical grid")
    return result


def _erode(mask, radius):
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8),
                    borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def outermost_old_domain_ring(old_domain):
    """Four-neighbor inner boundary touching the exterior-connected complement.

    Complement pixels are flood-filled from a padded outside border using
    four-connectivity. Closed internal holes therefore are not the outermost
    boundary. This selection depends only on OLD, never on predicted classes.
    """
    old = np.asarray(old_domain)
    if old.dtype != bool or old.ndim != 2:
        raise ValueError("Old domain must be an exact two-dimensional bool mask")
    complement = np.pad((~old).astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(complement, None, (0, 0), 2, flags=4)
    exterior = complement[1:-1, 1:-1] == 2
    adjacent = cv2.dilate(exterior.astype(np.uint8), CROSS,
                         borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    return old & adjacent, exterior


def _opposing_half_neighborhoods(co2):
    """CO2 in opposite open half-squares of the 13x13 neighborhood."""
    offsets = np.arange(-RADIUS_PX, RADIUS_PX + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    result = np.zeros_like(co2)
    for coordinate in (yy, xx):
        # Dilation reflects the footprint, but an opposite-side AND is invariant.
        a = cv2.dilate(co2.astype(np.uint8), (coordinate < 0).astype(np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        b = cv2.dilate(co2.astype(np.uint8), (coordinate > 0).astype(np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        result |= a & b
    return result


def ring_populations(labels, old_domain, rig_domain):
    """Return fixed radius-six two-sided ring and interior populations.

    The 1 px ring is the exterior four-connected complement's four-neighbor
    OLD boundary. CO2 is required within the 13x13 neighborhood from strict
    OLD interior and exterior OLD complement inside the final rig. The reference
    interior is OLD eroded by 13x13, so all six-step rays remain within OLD.
    Its comparable two-sided population requires CO2 in opposite open half
    squares, above/below or left/right. The center is never its own witness.
    """
    values = np.asarray(labels)
    if (values.dtype != np.uint8 or values.ndim != 2 or
            not np.isin(values, (0, 1, 2, 255)).all()):
        raise ValueError("Ring QA needs native uint8 categorical labels 0/1/2/255")
    old = _mask(old_domain, values.shape, "Old domain")
    rig = _mask(rig_domain, values.shape, "Rig domain")
    ring, exterior = outermost_old_domain_ring(old)
    strict = old & ~ring
    interior = _erode(old, RADIUS_PX)
    co2 = np.isin(values, (1, 2)) & rig
    inside_co2 = co2 & strict
    outside_co2 = co2 & exterior
    square = np.ones((2 * RADIUS_PX + 1,) * 2, np.uint8)
    inside_near = cv2.dilate(inside_co2.astype(np.uint8), square,
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    outside_near = cv2.dilate(outside_co2.astype(np.uint8), square,
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    ring_two_sided = inside_near & outside_near
    interior_two_sided = _opposing_half_neighborhoods(co2)
    near_co2 = cv2.dilate(co2.astype(np.uint8), square,
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return dict(ring=ring, interior=interior,
                eligible_ring=ring & rig & ring_two_sided,
                eligible_interior=interior & rig & interior_two_sided,
                interior_near_co2=interior & rig & near_co2)


def _population_report(values, selected, area):
    unresolved = selected & (values == 255)
    resolved = selected & (values != 255)
    denominator = int(resolved.sum())
    no_co2 = resolved & (values == 0)
    count = int(no_co2.sum())
    return dict(eligible_pixels=int(selected.sum()), resolved_pixels=denominator,
                unresolved_pixels=int(unresolved.sum()), no_co2_pixels=count,
                no_co2_share=count / denominator if denominator else None,
                no_co2_area_cm2=sum_pixel_area(no_co2, area, unit="cm2"))


def old_domain_ring_report(labels, old_domain, rig_domain, pixel_area_mm2):
    """Measure both rates and their difference; zero denominator stays null."""
    values = np.asarray(labels)
    populations = ring_populations(values, old_domain, rig_domain)
    area = validate_pixel_area(pixel_area_mm2, values.shape)
    ring = _population_report(values, populations["eligible_ring"], area)
    interior = _population_report(values, populations["eligible_interior"], area)
    a, b = ring["no_co2_share"], interior["no_co2_share"]
    return dict(contract=CONTRACT, radius_px=RADIUS_PX,
        distance_definition="Chebyshev radius6, 13x13 neighborhood; center excluded as its own witness",
        old_domain_definition="recorded original photo_sand_domain AND feature_support",
        ring_definition="four-neighbor inner OLD boundary touching four-connected exterior complement",
        interior_definition="erode(OLD,13x13,constant0)",
        two_sided_definition="ring: CO2 in strict OLD interior and exterior OLD complement inside rig; interior: CO2 in opposing open half-squares on either cardinal axis",
        ring_pixels=int(populations["ring"].sum()), interior_pixels=int(populations["interior"].sum()),
        ring=ring, interior=interior,
        interior_near_co2=_population_report(values, populations["interior_near_co2"], area),
        no_co2_share_difference=(a - b if a is not None and b is not None else None),
        no_co2_share_ratio=(a / b if a is not None and b is not None and b > 0 else None),
        acceptance_threshold=None, acceptance_claimed=False, mass_eligible=False)


def categorical_change_report(before, after, pixel_area_mm2, *, pre_gas_minimum=None,
                              minimum_pixel_area_mm2=None):
    """Report final phase changes and optional actual four-connected cutoff.

    pre_gas_minimum, when supplied, is the actual affected pre-cutoff gas
    selection before Rule B on the same grid, not the global final gas mask.
    minimum_pixel_area_mm2 is the original inference area map; the reporting
    area may have been revised subsequently. Cutoff effects are limited to
    this selection and are not asserted to match the later final gas mask.
    An absent pre-cutoff selection gives null cutoff effects.
    """
    old, new = np.asarray(before), np.asarray(after)
    if (old.dtype != np.uint8 or new.dtype != np.uint8 or old.ndim != 2 or
            old.shape != new.shape or not np.isin(old, (0, 1, 2, 255)).all() or
            not np.isin(new, (0, 1, 2, 255)).all()):
        raise ValueError("Change QA needs paired exact uint8 categorical grids")
    area = validate_pixel_area(pixel_area_mm2, new.shape)
    changed = old != new
    phases = {}
    for name, codes in (("co2", (1, 2)), ("gas", (2,)), ("aqueous", (1,))):
        a, b = np.isin(old, codes), np.isin(new, codes)
        before_area, after_area = (sum_pixel_area(mask, area, unit="cm2") for mask in (a, b))
        phases[name] = dict(before_pixels=int(a.sum()), after_pixels=int(b.sum()),
            signed_change_pixels=int(b.sum()) - int(a.sum()),
            before_area_cm2=before_area, after_area_cm2=after_area,
            signed_change_area_cm2=after_area - before_area)
    minimum = None
    if minimum_pixel_area_mm2 is not None and pre_gas_minimum is None:
        raise ValueError("A minimum-area map requires the actual pre-cutoff gas selection")
    if pre_gas_minimum is not None:
        from .phase_pixel_area import filter_components_by_area
        pre = _mask(pre_gas_minimum, new.shape, "Pre-minimum gas")
        minimum_area = (area if minimum_pixel_area_mm2 is None else
                        validate_pixel_area(minimum_pixel_area_mm2, new.shape))
        post, audit = filter_components_by_area(pre, minimum_area, 2.5, connectivity=4)
        removed = pre & ~post
        minimum = dict(connectivity=4, minimum_area_mm2=2.5,
            minimum_scope="supplied affected pre-minimum gas selection before Rule B",
            component_minimum_area_weights=("phase reporting area" if minimum_pixel_area_mm2 is None
                                           else "separately supplied original inference area"),
            pre_minimum_gas_pixels=int(pre.sum()),
            withdrawn_components=audit["withdrawn_components"], withdrawn_pixels=int(removed.sum()),
            withdrawn_area_cm2=sum_pixel_area(removed, area, unit="cm2"),
            withdrawn_original_inference_area_cm2=sum_pixel_area(removed, minimum_area, unit="cm2"),
            cutoff_only_post_gas_pixels=int(post.sum()),
            comparison_to_final_gas_performed=False, later_rule_b_effects_included=False)
    transitions = {f"{a}->{b}": int(np.count_nonzero((old == a) & (new == b)))
                   for a in (0, 1, 2, 255) for b in (0, 1, 2, 255) if a != b}
    return dict(changed_pixels=int(changed.sum()),
                changed_area_cm2=sum_pixel_area(changed, area, unit="cm2"),
                transitions=transitions, phases=phases, four_connected_minimum=minimum)


def frame_ring_revision_report(run, frame_index, elapsed_h, before, after,
                               old_domain, rig_domain, pixel_area_mm2, *,
                               pre_gas_minimum=None, minimum_pixel_area_mm2=None,
                               provenance=None):
    """Apply the identical predeclared QA to both categorical frame versions."""
    if (not isinstance(run, str) or not run.startswith("ac") or
            type(frame_index) is not int or frame_index < 0 or
            not np.isfinite(elapsed_h)):
        raise ValueError("Frame QA needs an explicit run/index and finite actual camera time")
    return dict(run=run.casefold(), frame_index=frame_index, elapsed_h=float(elapsed_h),
        before=old_domain_ring_report(before, old_domain, rig_domain, pixel_area_mm2),
        after=old_domain_ring_report(after, old_domain, rig_domain, pixel_area_mm2),
        change=categorical_change_report(before, after, pixel_area_mm2,
            pre_gas_minimum=pre_gas_minimum,
            minimum_pixel_area_mm2=minimum_pixel_area_mm2),
        provenance=provenance or {})
