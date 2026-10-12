"""Scoped frozen-head reassembly of the old FF_AC model-domain boundary.

This pure operation uses a predeclared boundary release and source-bound gas
selections. CO2 changes only in that release. Gas components incident to it are
reassembled before the original-area four-connected minimum and then Rule B.
"""
from __future__ import annotations

from hashlib import sha256

import cv2
import numpy as np

from .phase_gas_barrier_rule import apply_barrier_rule_b, array_identity
from .phase_pixel_area import FIXED_AC28_MM2, area_filtered_gas, sum_pixel_area, validate_pixel_area


CONTRACT = "ffac-old-domain-seam-scoped-reassembly-v1"


def _mask_binding(mask):
    return dict(pixels=int(mask.sum()), sha256=sha256(
        np.packbits(mask.ravel(), bitorder="little").tobytes()).hexdigest())


def revise_seam_labels(old_labels, fresh_labels, ring, old_domain,
                       old_pre_min, fresh_pre_min, modelled, original_area_mm2,
                       *, open_components, source_component_ids, barrier_mask):
    """Recompute incident gas components while preserving all other categories.

    ``old_pre_min`` is the decoded original G3/refit threshold selection before
    its CO2 intersection or minimum. ``fresh_pre_min`` is the fresh threshold
    selection. Both are boolean masks on the complete unchanged canvas.
    ``ring`` is the caller-sealed native outer ring or source-derived exterior
    coarse-domain interpolation footprint. ``fresh_labels`` already includes
    explicit nearest fill there. Filled pixels can change CO2 in the release
    but cannot supply modelled gas.

    The old/new pre-minimum union graph identifies incident components even
    when a ring change splits a formerly connected gas candidate. Unaffected
    components retain their saved post-minimum, post-Rule-B categories. A new
    Rule B reachability change that would affect that retained population is
    an explicit error, requiring a larger declared revision scope.
    """
    old, fresh = np.asarray(old_labels), np.asarray(fresh_labels)
    if (old.dtype != np.uint8 or old.ndim != 2 or not old.size
            or fresh.dtype != np.uint8 or fresh.shape != old.shape
            or not np.isin(old, (0, 1, 2, 255)).all()
            or not np.isin(fresh, (0, 1, 2, 255)).all()):
        raise ValueError("Seam labels require matching native uint8 categorical grids")
    ring, domain, inherited, evaluated, supported = tuple(np.asarray(value) for value in
        (ring, old_domain, old_pre_min, fresh_pre_min, modelled))
    if any(mask.dtype != bool or mask.shape != old.shape for mask in
           (ring, domain, inherited, evaluated, supported)):
        raise ValueError("Seam masks require matching explicit bool grids")
    if np.any(ring & ~domain) or np.any(ring & (fresh == 255)):
        raise ValueError("The released old-domain ring must have resolved fresh or filled labels")
    area = validate_pixel_area(original_area_mm2, old.shape)
    sources = tuple(source_component_ids)
    old_co2 = (old == 1) | (old == 2)
    result = old.copy()
    result[ring] = fresh[ring]
    new_co2 = (result == 1) | (result == 2)
    old_selection = np.where(domain, inherited, evaluated)
    new_selection = np.where(domain & ~ring, inherited, evaluated)
    old_gas_candidates = old_selection & old_co2 & supported
    new_gas_candidates = new_selection & new_co2 & supported
    union = old_gas_candidates | new_gas_candidates
    count, components = cv2.connectedComponents(union.astype(np.uint8), connectivity=4)
    incident = np.zeros(count, bool)
    incident[np.unique(components[ring & union])] = True
    incident[0] = False
    affected = incident[components]
    scoped_candidates = new_gas_candidates & affected
    kept, minimum_audit = area_filtered_gas(scoped_candidates, area,
        minimum_area_mm2=2.5, connectivity=4)
    commit = supported & new_co2 & (affected | ring)
    result[commit] = 1
    result[kept] = 2

    # Reachability depends on the corrected CO2 union, so evaluate Rule B on
    # that entire union after the scoped gas minimum. Its nominal area serves
    # only the inherited diagnostic; report physical removal with original area.
    _, old_rule_b = apply_barrier_rule_b(old, open_components, sources,
        pixel_area_mm2=FIXED_AC28_MM2, barrier_mask=barrier_mask)
    gas_before_b = result == 2
    revised, rule_b = apply_barrier_rule_b(result, open_components, sources,
        pixel_area_mm2=FIXED_AC28_MM2, barrier_mask=barrier_mask)
    changed_reachability = sorted(set(old_rule_b["allowed_component_ids"])
                                  ^ set(rule_b["allowed_component_ids"]))
    retained_scope = ~ring & ~affected
    retained_gas_or_candidates = retained_scope & ((old == 2) | new_gas_candidates)
    reachability_outside_scope = (retained_gas_or_candidates &
                                  np.isin(open_components, changed_reachability))
    changed_by_b_outside_scope = retained_scope & (revised != result)
    if reachability_outside_scope.any() or changed_by_b_outside_scope.any():
        raise ValueError("Rule B requires a wider declared gas scope: "
            f"reachable component IDs changed={changed_reachability}; "
            f"outside-scope candidates={int(reachability_outside_scope.sum())}; "
            f"outside-scope gas removals={int(changed_by_b_outside_scope.sum())}")
    result = revised
    changed = result != old
    changed_co2 = old_co2 != ((result == 1) | (result == 2))
    changed_gas = (old == 2) != (result == 2)
    if (np.any(changed_co2 & ~ring) or np.any(changed & retained_scope)
            or np.any((old == 255) & ~ring & (result != 255))):
        raise AssertionError("Seam revision exceeded its predeclared CO2/gas scope")
    removed_by_b = gas_before_b & (result != 2)
    rule_b["removed_gas_area_original_variable_cm2"] = sum_pixel_area(removed_by_b, area, unit="cm2")
    masks = dict(old_pre_minimum_gas=old_gas_candidates,
        corrected_pre_minimum_gas=new_gas_candidates, affected_gas_graph=affected,
        affected_gas_after_minimum=kept, changed_labels=changed,
        changed_co2=changed_co2, changed_gas=changed_gas)
    audit = dict(contract=CONTRACT, cpu_only=True, training=False,
        ring=_mask_binding(ring), old_domain=_mask_binding(domain),
        native_shape_hw=list(old.shape), mask_hash_encoding="little-endian packbits, row-major native grid",
        co2_outside_ring_exact=True, categories_outside_ring_and_affected_gas_exact=True,
        released_mask=_mask_binding(ring), co2_outside_released_mask_exact=True,
        ring_argument_definition="caller-sealed native boundary ring or source-derived exterior coarse-domain interpolation footprint; no output-derived dilation",
        unresolved_outside_ring_exact=True,
        affected_union_component_count=int(incident.sum()),
        union_component_count=count - 1,
        affected_graph_policy="four-connected union of old and corrected pre-minimum modelled gas candidates; components incident to fixed ring, including splits",
        minimum=minimum_audit,
        minimum_scope="corrected pre-minimum gas inside the affected union graph only",
        filtering_area_policy="original rig variable area before measured-area scaling; no global refilter",
        original_area=(dict(representation="constant", value_mm2=float(area)) if np.isscalar(area)
                       else dict(representation="native-array", sha256=array_identity(area))),
        operator_order=["ring CO2 correction", "combined literal gas threshold selection",
                        "affected-component original-area four-connected minimum", "Rule B"],
        old_rule_b_allowed_component_ids=old_rule_b["allowed_component_ids"],
        changed_rule_b_reachable_component_ids=changed_reachability,
        rule_b=rule_b, masks={name: _mask_binding(mask) for name, mask in masks.items()},
        mass_eligible=False, bottom_mass_allowed=False)
    return result, audit, masks


__all__ = ["CONTRACT", "revise_seam_labels"]
