"""Fixed L1 neighbourhood relabelling of stored FF_AC G3 categorical maps.

This CPU-only operation reads no training data and performs no inference. The
gas fraction is computed once from the unmodified base map. Rule B eligibility
is reconstructed from its frozen component decision, never from revised gas.
Top clipping and the unchanged floor-contact rule are separate caller steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np

from .phase_gas_barrier_rule import VERSION as RULE_B_CONTRACT, array_identity

CONTRACT = "ffac-g3-l1-neighbourhood-gas-v1"
SPEC_SHA256 = "7135a087e9c02267336a6ad31f2bf2a04e7d2fcd66a868b5fe96cef2651ade17"
F05_SPEC_SHA256 = "d3f600baeb94bc1e578d0dceaa572c695f55295cbd6aa77267e6fbd14b7085a1"
REFERENCE_POINT_XY = (2786.0, 2897.0)
PHASE_OFFSET_XY = (416.0, 496.0)
RADIUS_PX = 937.5
ACTIVATION_H = 0.935
BOX_SIZE = (51, 51)
GAS_FRACTION_THRESHOLD = 0.3


def approved_settings(gas_fraction_threshold=GAS_FRACTION_THRESHOLD):
    """Bind either approved gas fraction to its exact specification, never tune it."""
    if isinstance(gas_fraction_threshold, (bool, np.bool_)) or not isinstance(gas_fraction_threshold, Real):
        raise ValueError("L1 neighbourhood gas fraction must be the approved 0.3 or 0.5")
    threshold = float(gas_fraction_threshold)
    if threshold == 0.3:
        spec_sha256 = SPEC_SHA256
    elif threshold == 0.5:
        spec_sha256 = F05_SPEC_SHA256
    else:
        raise ValueError("L1 neighbourhood gas fraction must be the approved 0.3 or 0.5")
    return dict(gas_fraction_threshold=threshold, spec_sha256=spec_sha256)


def approved_spec_settings(spec_sha256):
    """Derive the fixed settings only from one of the two approved source SHAs."""
    if not isinstance(spec_sha256, str):
        raise ValueError("L1 neighbourhood requires an exact approved specification SHA")
    if spec_sha256 == SPEC_SHA256:
        return approved_settings(0.3)
    if spec_sha256 == F05_SPEC_SHA256:
        return approved_settings(0.5)
    raise ValueError("L1 neighbourhood requires an exact approved specification SHA")


@dataclass
class L1NeighbourhoodZone:
    mask: np.ndarray
    point_xy: tuple[float, float]
    active: bool

    def metadata(self):
        return dict(contract=CONTRACT, point_xy=list(self.point_xy), active=self.active,
                    radius_px=RADIUS_PX, activation_h=ACTIVATION_H,
                    reference_point_xy=list(REFERENCE_POINT_XY), phase_offset_xy=list(PHASE_OFFSET_XY),
                    radius_definition="150 mm at fixed scorer scale 0.16 mm/px",
                    pixel_coordinates="integer phase-grid column,row")


@dataclass
class L1NeighbourhoodGas:
    labels: np.ndarray
    l1_neighbourhood_gas: np.ndarray
    zone: L1NeighbourhoodZone
    report: dict


def _shape(shape_hw):
    if (len(shape_hw) != 2 or any(isinstance(v, (bool, np.bool_)) or
            not isinstance(v, (int, np.integer)) or v <= 0 for v in shape_hw)):
        raise ValueError("L1 neighbourhood requires a positive integer phase-grid shape")
    return tuple(map(int, shape_hw))


def _time(elapsed_hours):
    if isinstance(elapsed_hours, (bool, np.bool_)):
        raise ValueError("L1 neighbourhood requires a finite camera-clock elapsed time")
    try:
        elapsed = float(elapsed_hours)
    except (TypeError, ValueError):
        raise ValueError("L1 neighbourhood requires a finite camera-clock elapsed time") from None
    if not np.isfinite(elapsed):
        raise ValueError("L1 neighbourhood requires a finite camera-clock elapsed time")
    return elapsed


def l1_neighbourhood_zone_at_point(shape_hw, l1_phase_xy, elapsed_hours):
    """The inclusive scorer disc, without resampling or physical-area scaling."""
    shape = _shape(shape_hw)
    xy = np.asarray(l1_phase_xy, np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("L1 neighbourhood requires a finite phase-grid injector point")
    elapsed = _time(elapsed_hours)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask = (xx - xy[0]) ** 2 + (yy - xy[1]) ** 2 <= RADIUS_PX ** 2
    return L1NeighbourhoodZone(mask, tuple(map(float, xy)), elapsed >= ACTIVATION_H)


def l1_neighbourhood_zone(shape_hw, H_ac28_to_frame_native, elapsed_hours):
    """Transport the protocol L1 point by the current frame's photographic H."""
    _shape(shape_hw)
    H = np.asarray(H_ac28_to_frame_native, np.float64)
    if (H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15):
        raise ValueError("L1 neighbourhood requires a finite nonsingular source homography")
    mapped = H @ np.array((*REFERENCE_POINT_XY, 1.0), np.float64)
    if abs(mapped[2]) < 1e-12:
        raise ValueError("L1 point lies on the photographic projective horizon")
    point = mapped[:2] / mapped[2] + PHASE_OFFSET_XY
    return l1_neighbourhood_zone_at_point(shape_hw, point, elapsed_hours)


def frozen_rule_b_forbidden_mask(open_components, barrier_mask, rule_b_receipt):
    """Recover the exact base Rule B spatial decision from its bound receipt.

    For the old-domain-seam candidate use ``seam_reassembly.rule_b`` and the
    source producer's ``rule_b_geometry.asset``. That receipt certifies the
    component-array identity and the final allowed IDs *after* the seam fix.
    A missing decision or different geometry fails rather than recalculating
    breakthroughs. The caller binds the receipt and NPZ file SHA themselves.
    """
    components, barrier = np.asarray(open_components), np.asarray(barrier_mask)
    if (components.ndim != 2 or not components.size or components.dtype.kind not in "ui" or
            components.min() < 0 or components.max() > 255):
        raise ValueError("Frozen Rule B requires a nonnegative component grid with IDs <=255")
    if (barrier.dtype != np.bool_ or barrier.shape != components.shape or
            np.any(barrier & (components > 0))):
        raise ValueError("Frozen Rule B barrier must be a disjoint boolean grid of the same shape")
    if (not isinstance(rule_b_receipt, dict) or rule_b_receipt.get("contract") != RULE_B_CONTRACT or
            rule_b_receipt.get("co2_union_changed_pixels") != 0 or
            rule_b_receipt.get("only_gas_to_aqueous") is not True):
        raise ValueError("Missing or incompatible frozen base Rule B decision")
    actual = array_identity(components)
    if rule_b_receipt.get("open_components_array_sha256") != actual:
        raise ValueError("Frozen Rule B spatial decision is bound to different component geometry")

    def ids(key):
        values = rule_b_receipt.get(key)
        if (not isinstance(values, list) or not values or
                any(type(v) is not int or not 0 < v <= 255 for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"Frozen Rule B requires explicit positive {key}")
        return sorted(values)

    sources, allowed = ids("source_component_ids"), ids("allowed_component_ids")
    if not set(sources).issubset(allowed):
        raise ValueError("Frozen Rule B allowed components omit a protocol source component")
    forbidden = (components > 0) & ~np.isin(components, allowed)
    audit = dict(contract="ffac-frozen-rule-b-spatial-decision-v1",
                 open_components_array_sha256=actual, barrier_array_sha256=array_identity(barrier),
                 forbidden_array_sha256=array_identity(forbidden),
                 source_component_ids=sources, allowed_component_ids=allowed,
                 forbidden_pixels=int(forbidden.sum()), barrier_pixels=int(barrier.sum()),
                 rule_b_decision_recomputed=False, revised_gas_used=False)
    return forbidden, audit


def select_frozen_base_rule_b(base_frame, source_producer_frame):
    """Select the final recorded base decision, including certified seam bypass.

    Both input JSON files must first be verified by the caller against their
    bound file SHAs. Normally the seam revision's own final decision wins. Its
    source-changed no-op branch has no decision of its own: only that exact
    unchanged branch, identical stored-label SHA, complete source identity and
    identical categorical geometry may inherit the bound producer's decision.
    No image, classifier or breakthrough detector is consulted here.
    """
    if not isinstance(base_frame, dict) or not isinstance(source_producer_frame, dict):
        raise ValueError("Frozen base Rule B selection requires bound frame receipts")
    seam = base_frame.get("seam_reassembly")
    if not isinstance(seam, dict):
        raise ValueError("Frozen base Rule B selection requires the seam receipt")

    def asset(value, description):
        if (not isinstance(value, dict) or not isinstance(value.get("path"), str) or not value["path"] or
                not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64 or
                any(char not in "0123456789abcdef" for char in value["sha256"])):
            raise ValueError(f"Frozen Rule B selection lacks an exact {description} asset")
        return value

    def decision(value):
        if (not isinstance(value, dict) or value.get("contract") != RULE_B_CONTRACT or
                type(value.get("co2_union_changed_pixels")) is not int or value["co2_union_changed_pixels"] != 0 or
                value.get("only_gas_to_aqueous") is not True):
            raise ValueError("Frozen Rule B selection lacks a compatible recorded decision")
        for key in ("source_component_ids", "allowed_component_ids"):
            ids = value.get(key)
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
                    any(type(identifier) is not int or not 0 < identifier <= 255 for identifier in ids)):
                raise ValueError("Frozen Rule B selection has invalid recorded component IDs")
        if not set(value["source_component_ids"]).issubset(value["allowed_component_ids"]):
            raise ValueError("Frozen Rule B selection omits a protocol source component")
        digest = value.get("open_components_array_sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Frozen Rule B selection lacks its component-array identity")
        return value

    if "rule_b" in seam:
        selected = decision(seam["rule_b"])
        return selected, dict(contract="ffac-frozen-base-rule-b-selection-v1",
            decision_origin="base.seam_reassembly.rule_b", certified_unchanged_seam_bypass=False,
            open_components_array_sha256=selected["open_components_array_sha256"],
            allowed_component_ids=list(selected["allowed_component_ids"]), rule_b_decision_recomputed=False)

    required_marker_keys = {"source_changed", "inference_performed", "co2_outside_ring_changed_pixels", "changed_pixels"}
    if (set(seam) != required_marker_keys or seam.get("source_changed") is not True or
            seam.get("inference_performed") is not False or
            any(type(seam.get(key)) is not int or seam[key] != 0
                for key in ("co2_outside_ring_changed_pixels", "changed_pixels"))):
        raise ValueError("A missing seam Rule B decision is not a certified unchanged source bypass")
    if base_frame.get("complete") is not True or source_producer_frame.get("complete") is not True:
        raise ValueError("Frozen Rule B bypass requires complete bound base and producer receipts")
    source = (base_frame.get("binding") or {}).get("source")
    original_source = (source_producer_frame.get("binding") or {}).get("source")
    if (not isinstance(source, dict) or not source or source != original_source or
            source.get("source_changed") is not True or not isinstance(source.get("run"), str) or not source["run"] or
            type(source.get("frame_index")) is not int or source["frame_index"] < 0):
        raise ValueError("Frozen Rule B bypass lost its full exact current-source identity")
    sha = source.get("source_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("Frozen Rule B bypass lacks an exact source SHA")
    _time(source.get("elapsed_h"))
    for key in ("run", "frame_index", "elapsed_h", "source_sha256"):
        if base_frame.get(key) != source[key]:
            raise ValueError("Frozen Rule B bypass receipt and source frame/time differ")
    source_asset = asset(base_frame.get("source_producer_receipt"), "source producer receipt")
    before = asset(base_frame.get("labels_map"), "base labels")
    original = asset(source_producer_frame.get("labels_map"), "producer labels")
    if before["sha256"] != original["sha256"]:
        raise ValueError("Frozen Rule B bypass changed the stored categorical labels")
    revisions = [row.get("rig_domain_revision") for row in (base_frame, source_producer_frame)]
    if any(not isinstance(revision, dict) for revision in revisions):
        raise ValueError("Frozen Rule B bypass lacks bound categorical geometry")
    for key in ("contract", "canvas_wh", "phase_shape_hw", "native_origin_xy"):
        if key not in revisions[0] or revisions[0][key] != revisions[1].get(key):
            raise ValueError("Frozen Rule B bypass changed the categorical phase-grid geometry")
    classes = [asset(revision.get("classes"), "rig categorical classes") for revision in revisions]
    if classes[0]["sha256"] != classes[1]["sha256"]:
        raise ValueError("Frozen Rule B bypass changed the categorical rig/domain classes")
    # Measured area maps and H_frame_native_to_metric_pixel legitimately differ
    # after Olav's global scale revision. The photographic placement must not.
    geometry_keys = ("format", "origin_xy", "canvas_wh", "content_box_xywh",
                     "H_reference_rig_to_frame_native", "reference_rig_identity")
    parameters = [revision.get("parameters") for revision in revisions]
    if any(not isinstance(value, dict) for value in parameters):
        raise ValueError("Frozen Rule B bypass lacks the bound photographic geometry")
    for key in geometry_keys:
        if key not in parameters[0] or parameters[0][key] != parameters[1].get(key):
            raise ValueError("Frozen Rule B bypass changed the bound photographic geometry")
    geometry = source_producer_frame.get("rule_b_geometry")
    if not isinstance(geometry, dict):
        raise ValueError("Frozen Rule B bypass lacks its original barrier geometry")
    geometry_asset = asset(geometry.get("asset"), "Rule B geometry")
    selected = decision(source_producer_frame.get("rule_b"))
    if geometry.get("source_component_ids") != selected["source_component_ids"]:
        raise ValueError("Frozen Rule B bypass barrier and decision source IDs differ")
    return selected, dict(contract="ffac-frozen-base-rule-b-selection-v1",
        decision_origin="bound_source_producer.rule_b", certified_unchanged_seam_bypass=True,
        source_producer_receipt=source_asset, base_labels=before, unchanged_producer_labels=original,
        unchanged_labels_sha256=before["sha256"], unchanged_classes_sha256=classes[0]["sha256"],
        source_sha256=sha, run=source["run"], frame_index=source["frame_index"], elapsed_h=source["elapsed_h"],
        rule_b_geometry=geometry_asset, open_components_array_sha256=selected["open_components_array_sha256"],
        allowed_component_ids=list(selected["allowed_component_ids"]),
        measured_area_revision_used_for_decision=False, rule_b_decision_recomputed=False)


def _apply(base_labels, zone, elapsed_hours, barrier_mask, rule_b_forbidden_mask, gas_fraction_threshold):
    settings = approved_settings(gas_fraction_threshold)
    labels, barrier, forbidden = map(np.asarray, (base_labels, barrier_mask, rule_b_forbidden_mask))
    if (labels.ndim != 2 or not labels.size or labels.dtype != np.uint8 or
            not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError("L1 neighbourhood requires uint8 stored labels 0/1/2/255")
    if (barrier.dtype != np.bool_ or forbidden.dtype != np.bool_ or
            barrier.shape != labels.shape or forbidden.shape != labels.shape or zone.mask.shape != labels.shape):
        raise ValueError("L1 neighbourhood exclusions must be explicit matching boolean grids")
    revised = labels.copy()
    flag = np.zeros(labels.shape, bool)
    unrestricted_count = excluded_barrier = excluded_rule_b = excluded_either = 0
    if zone.active:
        # Deliberately use the literal OpenCV call. Default BORDER_REFLECT_101
        # and outside255-as-not-gas are part of the pre-registered operation.
        frac = cv2.blur((labels == 2).astype(np.float32), BOX_SIZE)
        qualifying = (labels == 1) & zone.mask & (frac >= settings["gas_fraction_threshold"])
        unrestricted_count = int(qualifying.sum())
        excluded_barrier = int(np.count_nonzero(qualifying & barrier))
        excluded_rule_b = int(np.count_nonzero(qualifying & forbidden))
        excluded_either = int(np.count_nonzero(qualifying & (barrier | forbidden)))
        flag = qualifying & ~barrier & ~forbidden
        revised[flag] = 2
    changed = revised != labels
    only_1_to_2 = not np.any(changed & ~((labels == 1) & (revised == 2)))
    union_differences = int(np.count_nonzero(np.isin(revised, (1, 2)) != np.isin(labels, (1, 2))))
    if (not np.array_equal(changed, flag) or not only_1_to_2 or union_differences or
            np.any(flag & (~zone.mask | barrier | forbidden)) or (not zone.active and np.any(changed))):
        raise AssertionError("L1 neighbourhood stored-map identity failed")
    report = dict(**zone.metadata(), elapsed_h=float(elapsed_hours), **settings,
                  local_fraction_box_hw=list(BOX_SIZE),
                  border="OpenCV default BORDER_REFLECT_101", iterations=1,
                  fraction_source="unmodified base stored labels; outside255 counts as not gas",
                  zone_pixels=int(zone.mask.sum()), changed_pixels=int(flag.sum()),
                  unrestricted_qualifying_pixels=unrestricted_count,
                  excluded_barrier_pixels=excluded_barrier, excluded_rule_b_pixels=excluded_rule_b,
                  excluded_either_pixels=excluded_either,
                  co2_union_changed_pixels=union_differences, only_aqueous_to_gas=only_1_to_2,
                  gas_minimum_reapplied=False, rule_b_decision_recomputed=False,
                  training=False, labels_read=False, model_inference_performed=False)
    return L1NeighbourhoodGas(revised, flag, zone, report)


def apply_l1_neighbourhood(base_labels, *, H_ac28_to_frame_native, elapsed_hours,
                          barrier_mask, rule_b_forbidden_mask,
                          gas_fraction_threshold=GAS_FRACTION_THRESHOLD):
    """One fixed pass on source-bound categorical maps, before top/floor checks."""
    labels = np.asarray(base_labels)
    zone = l1_neighbourhood_zone(labels.shape, H_ac28_to_frame_native, elapsed_hours)
    return _apply(labels, zone, elapsed_hours, barrier_mask, rule_b_forbidden_mask, gas_fraction_threshold)


def apply_l1_neighbourhood_at_point(base_labels, *, l1_phase_xy, elapsed_hours,
                                   barrier_mask, rule_b_forbidden_mask,
                                   gas_fraction_threshold=GAS_FRACTION_THRESHOLD):
    """Same operation with an already source-bound transported phase-grid point."""
    labels = np.asarray(base_labels)
    zone = l1_neighbourhood_zone_at_point(labels.shape, l1_phase_xy, elapsed_hours)
    return _apply(labels, zone, elapsed_hours, barrier_mask, rule_b_forbidden_mask, gas_fraction_threshold)
