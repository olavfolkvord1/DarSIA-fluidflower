"""CPU-only, frame-local numerical placements for the frozen T4c experiment.

No training or annotation reader is used here. Callers supply the sealed base
labels and source-bound geometry, and apply the flagged floor rule afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit, logit
from scipy.spatial import cKDTree

from . import phase_gas_t2_regional_inputs as shared
from . import phase_gas_t4_regional_inputs as t4
from . import phase_gas_t4b_regional_inputs as t4b
from . import phase_optical_regions as regional
from .phase_gas_barrier_rule import array_identity, apply_barrier_rule_b
from .phase_gas_regional_context import build_shared_regional_context, prepare_conditional_gas
from .phase_l1_neighbourhood_gas import frozen_rule_b_forbidden_mask
from .phase_pixel_area import validate_pixel_area, FIXED_AC28_MM2
from .phase_rig_domain_inference import MaskedCpuFeaturePredictor, log_to_unit_rgb, masked_flatfield_features
from .project import sha256_file

CONTRACT = "ffac-g3-t4c-frozen-logit-placements-v1"
SPEC_SHA256 = "53aad00f20bec56092c160477210613e7849f50467303aa0efd5243d90337daf"
BASE_PIXEL_SHA256 = "90f1d4608506273974e4cc758573f4c6467a823243abdaa02ef214b7286c9692"
BASE_REGIONAL_SHA256 = "d33b3eea47392268d3ddebe2d9e9c695bfc640eb6e252d83670a3eaf3e7428be"
SPACING_M = shared.SPACING_M
PIXEL_THRESHOLD = .7
REGIONAL_THRESHOLD = .6
MINIMUM_AREA_MM2 = 2.5
P2_FRACTION_KERNEL = (51, 51)


@dataclass
class ShiftContext:
    """One photograph's immutable optics and reusable regional preparation."""
    light_epoch_id: str
    domain: np.ndarray
    flat_valid: np.ndarray | None = None
    base_pixel: np.ndarray | None = None
    logits: dict = field(default_factory=dict)
    regional_models: dict = field(default_factory=dict)
    shared_regional: object | None = None
    metadata: dict = field(default_factory=dict)
    base_fraction: np.ndarray | None = None
    p2_score: np.ndarray | None = None
    _base_gas_sha256: str | None = None


def _labels(value):
    labels = np.asarray(value)
    if labels.dtype != np.uint8 or labels.ndim != 2 or not np.isin(labels, (0, 1, 2, 255)).all():
        raise ValueError("T4c requires unchanged categorical base labels on a two-dimensional grid")
    return labels


def _bool_grid(value, shape, name):
    array = np.asarray(value)
    if array.dtype != bool or array.shape != shape:
        raise ValueError(f"T4c {name} must be a boolean mask on the exact base grid")
    return array


def _epoch(run, light_epochs):
    receipt = light_epochs.get("receipt", {}) if isinstance(light_epochs, dict) else {}
    if len(receipt.get("sha256", "")) != 64:
        raise ValueError("T4c requires the SHA-bound series-project light epoch table")
    epoch = light_epochs["run_to_epoch"][str(run).casefold()]
    if epoch not in ("rig_light", "new_lamps"):
        raise ValueError("T4c requires a declared rig_light or new_lamps epoch")
    return epoch


def verify_base_pixel_model(path, expected_sha256):
    if expected_sha256 != BASE_PIXEL_SHA256 or sha256_file(Path(path)) != expected_sha256:
        raise ValueError("T4c changed the exact frozen base G3 pixel head")
    model = shared._read(path)
    fit = model.get("fit", {})
    if (model.get("variant") != "G3" or model.get("feature_names") != list(shared.FLATFIELD_FEATURE_NAMES)
            or fit.get("converged") is not True or fit.get("backend") != "cpu"
            or any(np.asarray(fit.get(k, [])).shape != (4,) for k in ("mean", "scale", "coefficients"))
            or not all(np.isfinite(fit[k]).all() for k in ("mean", "scale", "coefficients"))
            or np.any(np.asarray(fit["scale"]) <= 0) or not np.isfinite(fit["intercept"])):
        raise ValueError("T4c requires the converged four-feature base G3 expression")
    return model


def pixel_logits(pixel_model, features4, *, chunk_rows=128):
    """Original float64 G3 standardization, without a probability/logit round trip.

    Row chunks avoid allocating a native-grid four-column float64 duplicate.
    The T4b indicator must already have been folded into this four-feature head.
    """
    x = np.asarray(features4)
    fit = pixel_model["fit"]
    mean, scale, coefficients = (np.asarray(fit[k], np.float64) for k in ("mean", "scale", "coefficients"))
    if (x.ndim != 3 or x.shape[2] != 4 or mean.shape != (4,) or scale.shape != (4,)
            or coefficients.shape != (4,) or np.any(scale <= 0) or not np.isfinite(x).all()
            or not all(np.isfinite(v).all() for v in (mean, scale, coefficients))
            or not np.isfinite(fit["intercept"]) or int(chunk_rows) != chunk_rows or chunk_rows < 1):
        raise ValueError("T4c logit scoring needs finite four-feature inputs and a bounded positive row chunk")
    result = np.empty(x.shape[:2], np.float64)
    for first in range(0, x.shape[0], int(chunk_rows)):
        last = min(first + int(chunk_rows), x.shape[0])
        values = np.asarray(x[first:last], np.float64)
        result[first:last] = ((values - mean) / scale) @ coefficients + fit["intercept"]
    return result


def shifted_probability(logits, delta, valid=None):
    if not np.isfinite(delta):
        raise ValueError("T4c shift must be finite")
    values = np.asarray(logits)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("T4c needs finite cached native logits")
    probability = expit(values + float(delta)).astype(np.float32)
    if valid is not None:
        probability[~_bool_grid(valid, values.shape, "flat-field support")] = np.nan
    return probability


def prepare_shift_context(frozen_candidate, base_pixel_path, base_regional_path,
                          current_log, aligned_baseline_log, context_domain, *,
                          run, light_epochs, expected_base_pixel_sha256,
                          expected_base_regional_sha256, t4b_pixel_path=None,
                          t4b_regional_path=None, expected_t4b_pixel_sha256=None,
                          expected_t4b_regional_sha256=None, aligned_previous_log=None,
                          spacing_m=SPACING_M, source_baseline_log=None,
                          h_baseline_to_current=None, source_baseline_valid=None,
                          feature_shape_hw=(2979, 5472), base_labels=None):
    """Prepare optics once, keeping the base CO2 union outside this evaluator."""
    epoch = _epoch(run, light_epochs)
    valid = np.asarray(context_domain)
    if valid.dtype != bool or valid.ndim != 2:
        raise ValueError("T4c context requires the exact boolean native grid")
    metadata = dict(contract=CONTRACT, spec_sha256=SPEC_SHA256, run=str(run).casefold(),
        light_epoch_id=epoch, light_epoch_table=light_epochs["receipt"], cpu_only=True,
        fitting_performed=False, training_labels_read=False, validation_annotations_read=False, floor_rule_applied=False,
        frozen_co2_head_sha256=shared.CO2_SHA256, co2_inference_performed=False)
    # Rig light never passes through an optical or regional re-evaluation.
    if epoch == "rig_light":
        return ShiftContext(epoch, valid, metadata=metadata)
    if expected_base_regional_sha256 != BASE_REGIONAL_SHA256:
        raise ValueError("T4c changed the exact frozen base regional head")
    pixel = verify_base_pixel_model(base_pixel_path, expected_base_pixel_sha256)
    base_model = t4.verify_regional_model(base_regional_path, expected_base_regional_sha256, frozen_candidate)
    models = {"P1": base_model}
    projected = None
    supplied = (t4b_pixel_path, t4b_regional_path, expected_t4b_pixel_sha256, expected_t4b_regional_sha256)
    if any(v is not None for v in supplied):
        if not all(v is not None for v in supplied):
            raise ValueError("T4c P3 requires both exact T4b heads and their SHAs")
        joint = t4b.verify_pixel_model(t4b_pixel_path, expected_t4b_pixel_sha256)
        if joint["training_provenance"]["light_epochs"]["sha256"] != light_epochs["receipt"]["sha256"]:
            raise ValueError("T4c changed the T4b head's series-project epoch table")
        projected = t4b.epoch_pixel_model(joint, epoch)
        models["P3"] = t4b.verify_regional_model(t4b_regional_path, expected_t4b_regional_sha256, frozen_candidate)
        if models["P3"]["settings"] != base_model["settings"]:
            raise ValueError("T4c P3 changed the frozen regional context settings")
    root = Path(frozen_candidate)
    if sha256_file(root / "models/generic/co2_presence_signal.json") != shared.CO2_SHA256:
        raise ValueError("T4c context changed the frozen CO2 head")
    for values in (current_log, aligned_baseline_log, aligned_previous_log):
        if values is not None and (values.dtype != np.float32 or values.shape != (3, *valid.shape)):
            raise ValueError("T4c requires unchanged CHW float32 scientific source pixels")
    valid = valid & np.isfinite(current_log).all(0) & np.isfinite(aligned_baseline_log).all(0)
    if aligned_previous_log is not None:
        valid &= np.isfinite(aligned_previous_log).all(0)
    current, _ = log_to_unit_rgb(current_log, valid)
    baseline, _ = log_to_unit_rgb(aligned_baseline_log, valid)
    previous = None if aligned_previous_log is None else log_to_unit_rgb(aligned_previous_log, valid)[0]
    from .phase_learning import HierarchicalPhaseModel
    wrapper = MaskedCpuFeaturePredictor(HierarchicalPhaseModel.load(root), baseline, valid,
                                        feature_shape_hw=feature_shape_hw)
    predictor = wrapper.predictor
    current_features = predictor.prepare_rgb(current)
    previous_features = predictor.baseline if previous is None else predictor.prepare_rgb(previous)
    presence = np.empty(valid.shape, np.float32)
    for first, last, features in predictor.feature_blocks(current_features, previous_features,
                                                          dynamic_available=previous is not None):
        presence[first:last] = expit(features @ predictor.weights[:, 0] + predictor.bias[0]).reshape(
            last-first, valid.shape[1])
    ff, ff_valid = masked_flatfield_features(current_log, aligned_baseline_log, valid,
        source_baseline_log=source_baseline_log, h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid)
    ff[~ff_valid] = 0
    logits = {"P1": pixel_logits(pixel, ff)}
    if projected is not None:
        logits["P3"] = pixel_logits(projected, ff)
    base_probability = shifted_probability(logits["P1"], 0., ff_valid)
    context = build_shared_regional_context(current, baseline, presence, valid, spacing_m,
                                            regional.Settings(**base_model["settings"]))
    metadata.update(base_pixel_model_sha256=expected_base_pixel_sha256,
        base_regional_model_sha256=expected_base_regional_sha256,
        t4b_pixel_model_sha256=expected_t4b_pixel_sha256,
        t4b_regional_model_sha256=expected_t4b_regional_sha256,
        t4b_indicator_applied_once=projected is not None,
        context_columns_unchanged=[i for i in range(20) if i not in (1, 11)],
        gas_dependent_columns=[1, 11], regional_context_shape_hw=list(context.prepared["domain"].shape),
        optical_context_native_shape_hw=list(feature_shape_hw),
        optical_spatial_sigma_px=float(predictor.spatial_sigma))
    result = ShiftContext(epoch, valid, ff_valid, base_probability, logits, models, context, metadata)
    if base_labels is not None:
        labels = _labels(base_labels)
        if labels.shape != valid.shape:
            raise ValueError("T4c base labels do not match the prepared native grid")
        _cached_fraction(result, labels)
    return result


def base_gas_fraction(base_labels):
    """The fixed, one-pass OpenCV neighbourhood specified for T4c P2 and QA."""
    return cv2.blur((_labels(base_labels) == 2).astype(np.float32), P2_FRACTION_KERNEL)


def _cached_fraction(context, base_labels):
    identity = array_identity(base_labels == 2)
    if context._base_gas_sha256 is None:
        context.base_fraction = base_gas_fraction(base_labels)
        context._base_gas_sha256 = identity
        if context.base_pixel is not None:
            p = np.asarray(context.base_pixel, np.float64)
            if p.shape != base_labels.shape:
                raise ValueError("T4c cached base probability belongs to another native grid")
            context.p2_score = logit(np.clip(p, 1e-6, 1.-1e-6)) + logit(
                np.clip(np.asarray(context.base_fraction, np.float64), .01, .99))
    elif context._base_gas_sha256 != identity:
        raise ValueError("T4c cached neighbourhood belongs to another frozen base gas mask")
    return context.base_fraction


def prepare_p2_allowed(base_labels, zones, barrier_mask, open_components, frozen_base_rule_b):
    """Four frozen candidate constraints, with no distance or regional gate."""
    labels = _labels(base_labels)
    zone = np.asarray(zones)
    if zone.shape != labels.shape or zone.dtype.kind not in "ui" or not np.isin(zone, (0, 1, 2, 3, 4)).all():
        raise ValueError("T4c P2 needs the transported frozen QA zone grid")
    barrier = _bool_grid(barrier_mask, labels.shape, "P2 barrier facies")
    forbidden, decision = frozen_rule_b_forbidden_mask(open_components, barrier, frozen_base_rule_b)
    if forbidden.shape != labels.shape:
        raise ValueError("T4c P2 frozen Rule B belongs to another grid")
    caps = np.isin(zone, (1, 3))
    allowed = (labels == 1) & caps & ~barrier & ~forbidden
    audit = dict(contract=CONTRACT, placement="P2", candidates="base aqueous in frozen l1_cap/l2_caps, outside barriers and frozen Rule B forbidden",
        p2_allowed_array_sha256=array_identity(allowed), p2_barrier_array_sha256=array_identity(barrier),
        p2_rule_b_forbidden_array_sha256=array_identity(forbidden), zone_array_sha256=array_identity(zone),
        allowed_pixels=int(allowed.sum()), regional_head_used=False, distance_eligibility_used=False,
        frozen_rule_b=decision)
    return allowed, audit, dict(p2_allowed=allowed, p2_barrier=barrier, p2_rule_b_forbidden=forbidden)


def _minimum_additions(base_gas, selected_additions, pixel_area_mm2):
    """Drop only small final 4-connected components having no base gas pixel."""
    area = validate_pixel_area(pixel_area_mm2, base_gas.shape)
    final = base_gas | selected_additions
    count, components = cv2.connectedComponents(final.astype(np.uint8), connectivity=4)
    sizes = np.bincount(components.ravel(), minlength=count)
    areas = (sizes * area if np.isscalar(area) else
             np.bincount(components.ravel(), weights=area.ravel(), minlength=count))
    protected = np.zeros(count, bool)
    protected[np.unique(components[base_gas])] = True
    keep = (areas >= MINIMUM_AREA_MM2) | protected
    keep[0] = False
    retained = final & keep[components]
    return retained, dict(minimum_area_mm2=MINIMUM_AREA_MM2, connectivity=4,
        scope="final gas components; every component containing frozen base gas protected",
        components=int(count-1), protected_base_components=int(protected[1:].sum()),
        removed_isolated_addition_pixels=int((final & ~retained).sum()),
        base_gas_removed_pixels=int((base_gas & ~retained).sum()), variable_pixel_area=not np.isscalar(area))


def evaluate_placement(context, placement, delta, base_labels, modelled, pixel_area_mm2, *,
                       filled_support=None, top_keep, open_components, barrier_mask,
                       frozen_base_rule_b, p2_allowed=None, source_component_ids=None):
    """Return labels/audit/masks before the unchanged caller-owned floor rule."""
    labels = _labels(base_labels)
    if placement not in ("P1", "P2", "P3") or not np.isfinite(delta):
        raise ValueError("T4c needs a declared placement and finite logit shift")
    audit = dict(context.metadata, contract=CONTRACT, placement=placement, delta=float(delta),
                 floor_rule_applied=False, frozen_co2_union_reused=True, co2_union_changed_pixels=0)
    if context.light_epoch_id == "rig_light":
        audit.update(rig_light_labels_bit_identical=True, changed_phase_pixels=0,
                     gas_added_pixels=0, gas_removed_pixels=0, optical_evaluation_bypassed=True)
        return labels.copy(), audit, {}
    if context.light_epoch_id != "new_lamps" or context.domain.shape != labels.shape:
        raise ValueError("T4c shift context does not match the declared new-lamp frame")
    support = _bool_grid(modelled, labels.shape, "fresh modelled support") & context.domain & context.flat_valid
    top = _bool_grid(top_keep, labels.shape, "top keep")
    forbidden, decision = frozen_rule_b_forbidden_mask(open_components, barrier_mask, frozen_base_rule_b)
    if source_component_ids is not None and sorted(source_component_ids) != decision["source_component_ids"]:
        raise ValueError("T4c source components disagree with the frozen base Rule B decision")
    fraction = _cached_fraction(context, labels)
    masks = dict(base_fraction=fraction, frozen_co2=np.isin(labels, (1, 2)))
    if placement == "P2":
        allowed = _bool_grid(p2_allowed, labels.shape, "P2 allowed")
        barrier = _bool_grid(barrier_mask, labels.shape, "P2 barrier facies")
        if np.any(allowed & ((labels != 1) | barrier | forbidden)):
            raise ValueError("T4c P2 allowed mask violates the frozen base constraints")
        p = np.asarray(context.base_pixel, np.float64)
        if p.shape != labels.shape:
            raise ValueError("T4c P2 lost the unshifted base G3 probability grid")
        score = context.p2_score
        if score is None:
            raise ValueError("T4c P2 lost its cached unshifted optical/neighbourhood score")
        additions = allowed & np.isfinite(score) & (score + float(delta) >= logit(PIXEL_THRESHOLD))
        final, minimum = _minimum_additions(labels == 2, additions, pixel_area_mm2)
        result = labels.copy()
        result[additions & final] = 2
        clipped = (result == 2) & (labels != 2) & ~top
        result[clipped] = labels[clipped]
        audit.update(regional_head_used=False, shifted_pixel_probability_used=False,
            neighbourhood_computed_from="frozen base gas only; never updated", neighbourhood_kernel=list(P2_FRACTION_KERNEL),
            neighbourhood_border="OpenCV default BORDER_DEFAULT", neighbourhood_fraction_clip=[.01, .99],
            base_pixel_probability_clip=[1e-6, 1.-1e-6], pixel_threshold=PIXEL_THRESHOLD,
            p2_allowed_array_sha256=array_identity(allowed), allowed_pixels=int(allowed.sum()),
            allowed_nonfinite_probability_pixels=int((allowed & ~np.isfinite(p)).sum()),
            selected_pre_minimum_pixels=int(additions.sum()), gas_minimum=minimum,
            top_clip_removed_pixels=int(clipped.sum()), top_clip_applied=True,
            frozen_rule_b=decision, rule_b_decision_recomputed=False, rule_b_applied=True)
        masks.update(p2_allowed=allowed, p2_barrier=barrier, p2_rule_b_forbidden=forbidden,
                     pre_minimum_additions=additions, post_minimum_gas=final)
        if np.any((labels == 2) & (result != 2)) or not np.array_equal(result[~allowed], labels[~allowed]):
            raise AssertionError("T4c P2 changed frozen base gas or a non-candidate label")
    else:
        if placement not in context.logits or placement not in context.regional_models or context.shared_regional is None:
            raise ValueError("T4c placement lost its exact frozen pixel/regional pair")
        pixel = shifted_probability(context.logits[placement], delta, context.flat_valid)
        prepared = prepare_conditional_gas(context.shared_regional, np.nan_to_num(pixel, nan=0.))
        model = context.regional_models[placement]
        regional_score = regional.probability(prepared["features"], model["heads"]["gas_within_co2"])
        regional_score = cv2.resize(regional_score, labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
        result, threshold_audit, threshold_masks = shared.replace_gas_from_scores(
            labels, pixel, regional_score, support, pixel_area_mm2, filled_support=filled_support)
        clipped = (result == 2) & ~top
        result[clipped] = 1
        result, applied_b = apply_barrier_rule_b(result, open_components,
            decision["source_component_ids"] if source_component_ids is None else source_component_ids,
            pixel_area_mm2=FIXED_AC28_MM2, barrier_mask=barrier_mask)
        audit.update(threshold_audit)
        audit.update(context.metadata, contract=CONTRACT, placement=placement, delta=float(delta),
            regional_head_used=True, shifted_pixel_probability_used=True, floor_rule_applied=False,
            top_clip_removed_pixels=int(clipped.sum()), top_clip_applied=True,
            rule_b_application=applied_b, rule_b_decision_recomputed=True, rule_b_applied=True,
            rule_b_removed_pixels=applied_b["removed_gas_pixels"], t4b_indicator_applied_once=placement == "P3")
        masks.update(threshold_masks, frozen_rule_b_forbidden=forbidden)
    if not np.array_equal(np.isin(result, (1, 2)), np.isin(labels, (1, 2))):
        raise AssertionError("T4c gas placement changed the frozen CO2 union")
    audit.update(changed_phase_pixels=int((result != labels).sum()),
        gas_added_pixels=int(((result == 2) & (labels != 2)).sum()),
        gas_removed_pixels=int(((result != 2) & (labels == 2)).sum()),
        base_fraction_array_sha256=array_identity(fraction))
    return result, audit, masks


def distance_to_base_gas_mm(base_labels, H_ac28_to_frame_native, native_origin_xy, *,
                           reference_spacing_mm, query_mask=None, chunk_rows=128):
    """Physical nearest-base-gas distance at the report's integer pixel centres.

    The caller reads the approved scale measurement and supplies the measured
    reference spacing. Frozen cap-zone geometry is a separate QA policy. No
    pixel-area approximation, half-pixel offset, or eligibility gate is used.
    """
    labels = _labels(base_labels)
    spacing = float(reference_spacing_mm)
    h = np.asarray(H_ac28_to_frame_native, np.float64)
    origin = np.asarray(native_origin_xy, np.float64)
    if (h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-15
            or origin.shape != (2,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing) or spacing <= 0
            or int(chunk_rows) != chunk_rows or chunk_rows < 1):
        raise ValueError("T4c distance needs nonsingular source geometry, measured positive spacing and bounded row chunks")
    query = np.ones(labels.shape, bool) if query_mask is None else _bool_grid(query_mask, labels.shape, "distance query")
    distances = np.full(labels.shape, np.inf, np.float32)
    if not np.any(labels == 2) or not query.any():
        return distances
    inverse = np.linalg.inv(h)
    def reference_points(rows, columns):
        x, y = columns.astype(np.float64) + origin[0], rows.astype(np.float64) + origin[1]
        denominator = inverse[2, 0]*x + inverse[2, 1]*y + inverse[2, 2]
        if np.any(np.abs(denominator) < 1e-12):
            raise ValueError("T4c distance queries cross the source projective horizon")
        return np.column_stack(((inverse[0, 0]*x + inverse[0, 1]*y + inverse[0, 2])/denominator,
                                (inverse[1, 0]*x + inverse[1, 1]*y + inverse[1, 2])/denominator)) * spacing
    rows, columns = np.nonzero(labels == 2)
    tree = cKDTree(reference_points(rows, columns))
    for first in range(0, labels.shape[0], int(chunk_rows)):
        last = min(first + int(chunk_rows), labels.shape[0])
        rows, columns = np.nonzero(query[first:last])
        rows += first
        if rows.size:
            distances[rows, columns] = tree.query(reference_points(rows, columns), workers=1)[0].astype(np.float32)
    return distances


reference_distance_to_base_mm = distance_to_base_gas_mm


__all__ = ["CONTRACT", "SPEC_SHA256", "BASE_PIXEL_SHA256", "BASE_REGIONAL_SHA256", "ShiftContext",
           "prepare_shift_context", "pixel_logits", "shifted_probability", "prepare_p2_allowed",
           "evaluate_placement", "base_gas_fraction", "distance_to_base_gas_mm", "reference_distance_to_base_mm"]
