"""CPU-only frozen-head inference at newly observed FF_AC rig boundaries.

The photograph is never spatially resampled here. Neighbourhood operators
ignore hardware; unobserved/unsupported pixels receive categorical nearest
neighbours only after inference, with separate flags. This is not a mass fit.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.special import expit

from .phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256

CONTRACT = "ffac-rig-domain-frozen-inference-v1"
LEGACY_CO2_CONTEXT_GAS_SHA256 = "74e7d6e166b04d158eb04f09cdc310ec38dbbd2b148bcd00a25aedb8962020e3"


def require_frozen_heads(candidate):
    root = Path(candidate)
    for name, expected in (("models/generic/co2_presence_signal.json", CO2_SHA256),
                           ("models/generic/gas_within_co2_signal.json", LEGACY_CO2_CONTEXT_GAS_SHA256),
                           ("models/g3/gas_head.json", G3_SHA256),
                           ("regional_model.json", REFIT_SHA256)):
        path = root / name
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Rig inference cannot change the frozen head: {name}")


def log_to_unit_rgb(log_rgb, valid):
    values, mask = np.asarray(log_rgb), np.asarray(valid)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[0] != 3:
        raise TypeError("Rig model input requires CHW float32 scientific log RGB")
    if mask.dtype != bool or mask.shape != values.shape[1:]:
        raise ValueError("Rig validity must match the native scientific grid")
    usable = mask & np.isfinite(values).all(axis=0)
    rgb = np.zeros((*usable.shape, 3), np.float32)
    rgb[usable] = np.exp(values[:, usable].T.astype(np.float64)) / 65535.
    if not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("Rig float input exceeds its fixed RAW denominator; no clipping allowed")
    return rgb, usable


def normalized_gaussian(values, valid, sigma, *, denominator=None):
    data, mask = np.asarray(values, np.float32), np.asarray(valid, bool)
    if data.shape[:2] != mask.shape or not np.isfinite(data).all():
        raise ValueError("Masked optical filter needs finite native values and a matching mask")
    weights = mask.astype(np.float32)
    if denominator is None:
        denominator = cv2.GaussianBlur(weights, (0, 0), float(np.float32(sigma)),
                                     borderType=cv2.BORDER_CONSTANT)
    elif denominator.dtype != np.float32 or denominator.shape != mask.shape:
        raise ValueError("Cached optical normalization must share the native float32 grid")
    weighted = data * (weights if data.ndim == 2 else weights[..., None])
    numerator = cv2.GaussianBlur(weighted, (0, 0), float(np.float32(sigma)),
                               borderType=cv2.BORDER_CONSTANT)
    denominator = denominator if data.ndim == 2 else denominator[..., None]
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 1e-6)


class MaskedCpuFeaturePredictor:
    """Use frozen optical features, replacing only boundary convolution.

    The inner implementation is the authoritative CPU feature extractor.
    No values belonging to the frame/hardware are admitted to its filters.
    """
    def __init__(self, model, baseline, valid, *, feature_shape_hw=None):
        from .tools.phase_inference_benchmark_backends import CpuFeaturePredictor, SPATIAL_CONTEXT_FRACTION
        owner = self

        class Predictor(CpuFeaturePredictor):
            def blur(self, values, sigma):
                # The hardware/support mask is fixed for this frame. Reuse its
                # Gaussian normalization across baseline, current, previous and
                # scalar features without changing numerator or division order.
                key = float(np.float32(sigma))
                if key not in owner._denominators:
                    owner._denominators[key] = cv2.GaussianBlur(
                        owner.valid.astype(np.float32), (0, 0), key,
                        borderType=cv2.BORDER_CONSTANT)
                return normalized_gaussian(values, owner.valid, sigma,
                    denominator=owner._denominators[key])

        self.valid = np.array(valid, dtype=bool, copy=True)
        self.valid.setflags(write=False)
        self._denominators = {}
        self.predictor = Predictor(model, baseline, block_rows=128,
                                   input_encoding="v5-unit-float32")
        if feature_shape_hw is not None:
            feature_shape = np.asarray(feature_shape_hw)
            if feature_shape.shape != (2,) or np.any(feature_shape <= 0):
                raise ValueError("Frozen optical context needs its original native shape")
            self.predictor.spatial_sigma = max(self.predictor.texture_sigma,
                SPATIAL_CONTEXT_FRACTION * float(feature_shape.min()))


def masked_flatfield_features(current_log, aligned_baseline_log, valid, *,
                              source_baseline_log=None, h_baseline_to_current=None,
                              source_baseline_valid=None):
    """Frozen G3's four log colour ratios, with a normalized 5×5 mean.

    The linear photographic matrix and channel/luminance definitions are
    unchanged. Only missing neighbours are excluded instead of introducing
    frame pixels or an artificial eight-pixel erosion.
    """
    from .tools.ffac_track1_display_modes import M_NATURAL
    from . import phase_gas_flatfield as flat
    shape = valid.shape
    transformed = []
    denominator = cv2.blur(valid.astype(np.float32), (5, 5),
                          borderType=cv2.BORDER_CONSTANT)
    for index, signal in enumerate((current_log, aligned_baseline_log)):
        if index == 1 and source_baseline_log is not None:
            # G3 calibrates colour on the original baseline before warping.
            # max(matrix @ RGB,0) is nonlinear: calibrating after a warp is
            # not equivalent near a negative channel or a support boundary.
            prepared = flat.prepare_flatfield_reference(source_baseline_log,
                source_valid_mask=source_baseline_valid)
            if h_baseline_to_current is None:
                raise ValueError("Flat-field baseline needs its exact frame homography")
            h = np.asarray(h_baseline_to_current,np.float64)
            values = cv2.warpPerspective(prepared.linear_rgb,h,shape[::-1],
                flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            luminance = cv2.warpPerspective(prepared.linear_rgb @ flat._LUMINANCE,h,shape[::-1],
                flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            footprint = cv2.warpPerspective(prepared.valid.astype(np.float32),h,shape[::-1],
                flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            if np.any(valid & (footprint < np.float32(1-1e-6))):
                raise ValueError("G3 uses an unsupported registered baseline pixel")
            values[~valid] = 0
            luminance[~valid] = 0
        else:
            if signal.dtype != np.float32 or signal.shape != (3, *shape):
                raise TypeError("G3 requires unchanged native float32 log input")
            values = np.zeros((*shape, 3), np.float32)
            values[valid] = np.exp(signal[:, valid].T.astype(np.float64)) @ M_NATURAL
            np.maximum(values, 0, out=values)
            luminance = values @ flat._LUMINANCE
        if values.dtype != np.float32:
            raise TypeError("G3 requires unchanged native float32 log input")
        values = np.concatenate((values,luminance[...,None]),axis=-1)
        numerator = cv2.blur(values, (5, 5), borderType=cv2.BORDER_CONSTANT)
        transformed.append(np.divide(numerator, denominator[..., None],
                                     out=np.zeros_like(numerator),
                                     where=denominator[..., None] > 1e-6))
    current, baseline = transformed
    c,b = current,baseline
    usable = valid & (c > 0).all(-1)
    features = np.full((*shape, 4), np.nan, np.float32)
    # The frozen feature is log(current / registered pre-injection).
    features[usable] = np.log(c[usable] / np.maximum(b[usable],flat.DENOMINATOR_FLOOR))
    return features, usable


def frozen_native_partition(presence, conditional_gas, valid):
    """The existing regional runtime uses p>=.5 and conditional gas>=.6.

    An argmax over water/aqueous/gas is a different classifier, so it must
    never be substituted for this frozen decision rule.
    """
    presence, conditional_gas = np.asarray(presence), np.asarray(conditional_gas)
    valid = np.asarray(valid, bool)
    if presence.shape != valid.shape or conditional_gas.shape != valid.shape:
        raise ValueError("Frozen partition inputs do not share the native grid")
    active = (presence >= .5) & valid
    labels = active.astype(np.uint8)
    labels[active & (conditional_gas >= .6)] = 2
    return labels


def require_categorical_feature_support(labels, optical_valid, gas_feature_valid):
    """An unavailable gas feature is unresolved, never aqueous evidence.

    Both frozen stages are required to assign a final water/aqueous/gas class.
    Filling happens later and has an explicit observed/unobserved flag.
    """
    labels = np.asarray(labels)
    optical_valid, gas_feature_valid = (np.asarray(mask) for mask in
                                        (optical_valid, gas_feature_valid))
    if (labels.dtype != np.uint8 or optical_valid.dtype != bool or
            gas_feature_valid.dtype != bool or labels.shape != optical_valid.shape or
            labels.shape != gas_feature_valid.shape):
        raise ValueError("Categorical feature support must share the native label grid")
    supported = optical_valid & gas_feature_valid
    result = labels.copy()
    result[~supported] = 255
    return result, supported, dict(
        optical_supported_pixels=int(optical_valid.sum()),
        gas_feature_invalid_optical_supported_pixels=int(np.count_nonzero(
            optical_valid & ~gas_feature_valid)),
        final_modelled_feature_pixels=int(supported.sum()),
        invalid_feature_policy="unresolved before explicitly flagged nearest-label filling")


def weighted_co2_output_guard(labels, valid, pixel_area_mm2):
    """Preserve the inherited eight-connected island and raster-seam rules."""
    from .phase_pixel_area import validate_pixel_area
    values = np.asarray(labels)
    valid = np.asarray(valid)
    if values.dtype != np.uint8 or valid.dtype != bool or values.shape != valid.shape:
        raise ValueError("CO2 output guard requires the exact native label/mask grid")
    area = validate_pixel_area(pixel_area_mm2, values.shape)
    co2 = np.isin(values,(1,2)) & valid
    count,components,stats,_ = cv2.connectedComponentsWithStats(co2.astype(np.uint8),connectivity=8)
    if np.isscalar(area):
        physical = stats[:,cv2.CC_STAT_AREA].astype(np.float64)*area
    else:
        physical = np.bincount(components.ravel(),weights=np.asarray(area).ravel(),minlength=count)
    keep = physical >= 2.5
    h,w = values.shape
    seams = (((stats[:,cv2.CC_STAT_WIDTH] == 1) & np.isin(stats[:,cv2.CC_STAT_LEFT],(0,w-1))) |
             ((stats[:,cv2.CC_STAT_HEIGHT] == 1) & np.isin(stats[:,cv2.CC_STAT_TOP],(0,h-1))))
    seams[0] = False
    keep[seams] = False
    keep[0] = False
    result = np.where(keep[components],values,0).astype(np.uint8)
    return result,dict(connectivity=8,minimum_island_area_mm2=2.5,
        removed_border_seams=int(seams.sum()),removed_islands=int((~keep[1:]).sum()),
        removed_island_pixels=int(np.count_nonzero(co2 & ~keep[components])),
        area_weighted=True,added_pixels=0)


def infer_native(candidate, current_log, aligned_baseline_log, domain, *,
                 aligned_previous_log=None, spacing_m=(.00016448, .00016448),
                 pixel_area_mm2=None, source_baseline_log=None,
                 h_baseline_to_current=None, source_baseline_valid=None,
                 feature_shape_hw=(2979,5472), return_decision_layers=False):
    """Evaluate frozen optical/CO2-regional/G3-regional heads on real pixels."""
    from .phase_learning import HierarchicalPhaseModel
    from . import phase_optical_regions as regional
    from .phase_gas_regional_context import build_shared_regional_context, prepare_conditional_gas
    from .tools.train_ffac_gas_flatfield_variants import gas_probability
    from .phase_pixel_area import area_filtered_gas

    require_frozen_heads(candidate)
    if pixel_area_mm2 is None:
        raise ValueError("Rig inference needs its recorded per-pixel physical area")
    root = Path(candidate)
    model = HierarchicalPhaseModel.load(root)
    regional_model = json.loads((root / "regional_model.json").read_text())
    g3_model = json.loads((root / "models/g3/gas_head.json").read_text())
    valid = np.asarray(domain, bool) & np.isfinite(current_log).all(0) & np.isfinite(aligned_baseline_log).all(0)
    if aligned_previous_log is not None:
        valid &= np.isfinite(aligned_previous_log).all(0)
    current, valid = log_to_unit_rgb(current_log, valid)
    baseline, _ = log_to_unit_rgb(aligned_baseline_log, valid)
    previous = None if aligned_previous_log is None else log_to_unit_rgb(aligned_previous_log, valid)[0]
    wrapper = MaskedCpuFeaturePredictor(model, baseline, valid,
                                       feature_shape_hw=feature_shape_hw)
    predictor = wrapper.predictor
    prepared_current = predictor.prepare_rgb(current)
    prepared_previous = predictor.baseline if previous is None else predictor.prepare_rgb(previous)
    shape = valid.shape
    presence, legacy_gas = np.empty(shape, np.float32), np.empty(shape, np.float32)
    for first, last, features in predictor.feature_blocks(prepared_current, prepared_previous,
                                                         dynamic_available=previous is not None):
        scores = expit(features @ predictor.weights + predictor.bias).reshape(last-first, shape[1], 2)
        presence[first:last], legacy_gas[first:last] = scores[..., 0], scores[..., 1]
    ff, ff_valid = masked_flatfield_features(current_log, aligned_baseline_log, valid,
        source_baseline_log=source_baseline_log,h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid)
    g3 = gas_probability(g3_model, np.empty((*shape, 0), np.float32),
                         np.where(ff_valid[..., None], ff, 0)).astype(np.float32)
    g3[~ff_valid] = np.nan
    settings = regional.Settings(**regional_model["settings"])
    shared = build_shared_regional_context(current, baseline, presence, valid, spacing_m, settings)
    # The parent's CO2 union was frozen BEFORE replacing its gas partition
    # with G3. Its native seeds and regional gas_joint feature columns must
    # therefore continue to use the generic legacy conditional gas head.
    # G3 belongs only to the downstream gas refit below.
    co2_prepared = prepare_conditional_gas(shared, legacy_gas)
    # Native frozen partition precedes the existing regional CO2 reconstruction.
    raw = frozen_native_partition(presence, legacy_gas, valid)
    labels, co2_audit = regional.predict(co2_prepared, regional_model, raw, valid)
    # Retain the inherited disconnected CO2-island guard. Its component
    # connectivity is eight, across both phases; G3 gas uses four below.
    labels, island_audit = weighted_co2_output_guard(labels,valid,pixel_area_mm2)
    prepared_gas = prepare_conditional_gas(shared, np.nan_to_num(g3, nan=0.))
    gas_region = regional.probability(prepared_gas["features"], regional_model["heads"]["gas_within_co2"])
    gas_region = cv2.resize(gas_region, shape[::-1], interpolation=cv2.INTER_LINEAR)
    co2 = np.isin(labels, (1, 2)) & valid
    gas_selected = co2 & ff_valid & (g3 >= .7) & (gas_region >= .6)
    pre_minimum_gas = gas_selected.copy() if return_decision_layers else None
    if pixel_area_mm2 is None:
        raise ValueError("Rig inference needs its recorded per-pixel physical area")
    gas_selected, minimum_audit = area_filtered_gas(gas_selected, pixel_area_mm2,
                                                   minimum_area_mm2=2.5, connectivity=4)
    labels[co2] = np.where(gas_selected[co2], 2, 1)
    labels, final_support, support_audit = require_categorical_feature_support(
        labels, valid, ff_valid)
    result = (labels, final_support, dict(contract=CONTRACT, cpu_only=True, training=False,
        float_input_end_to_end=True, hardware_in_filters=False,
        convolution="native normalized Gaussian; normalized5x5 G3; fixed trained weights",
        optical_context_native_shape_hw=list(feature_shape_hw),
        optical_spatial_sigma_px=float(predictor.spatial_sigma),
        co2=co2_audit, co2_islands=island_audit, gas_minimum=minimum_audit,
        categorical_feature_support=support_audit))
    if return_decision_layers:
        return (*result, dict(co2=co2, pre_minimum_gas=pre_minimum_gas))
    return result


def fill_rig_labels(model_labels, rig_domain, observed, *, hidden=None,
                    tie_break_reference=None, tie_break_mask=None, audit=None):
    """Fill nearest categorical labels, recording observed/unobserved support."""
    labels, domain, visible = np.asarray(model_labels), np.asarray(rig_domain), np.asarray(observed)
    if labels.dtype != np.uint8 or domain.dtype != bool or visible.dtype != bool or labels.shape != domain.shape or visible.shape != domain.shape:
        raise TypeError("Rig classes and labels must share one explicit grid")
    hidden = np.zeros(domain.shape, bool) if hidden is None else np.asarray(hidden, bool)
    if hidden.shape != domain.shape:
        raise ValueError("Hidden ColorChecker mask differs from the rig grid")
    if np.any(hidden & domain):
        raise ValueError("Hidden ColorChecker/hardware must be excluded from the rig domain")
    supported = domain & visible & np.isin(labels, (0, 1, 2))
    if not supported.any():
        raise ValueError("No modelled source pixel exists for nearest-label filling")
    filled = domain & ~supported
    result = np.full(domain.shape, 255, np.uint8)
    result[supported] = labels[supported]
    if filled.any():
        distances, indices = distance_transform_edt(~supported, return_distances=True, return_indices=True)
        result[filled] = labels[indices[0, filled], indices[1, filled]]
        tie_audit = dict(contract="ffac-nearest-fill-equal-distance-parent-category-v1",
            applicable=False, preference_candidates=0, equal_distance_preferences=0,
            unique_nearest_unchanged=True, modelled_support_unchanged=True)
        if tie_break_reference is not None or tie_break_mask is not None:
            preferred, eligible = np.asarray(tie_break_reference), np.asarray(tie_break_mask)
            if (preferred.dtype != np.uint8 or preferred.shape != domain.shape or
                    eligible.dtype != bool or eligible.shape != domain.shape or
                    np.any(eligible & ~np.isin(preferred, (0, 1, 2)))):
                raise ValueError("Nearest-fill tie preference requires a source-bound categorical reference/mask")
            # The tie policy is fixed before filling. An inherited category
            # may win ONLY if at least one modelled pixel with that category
            # is at the exact same minimum Euclidean raster distance. It
            # never substitutes a farther neighbour or calls a fill modelled.
            candidates = np.argwhere(filled & eligible & (result != preferred))
            count = 0
            for yy, xx in candidates:
                minimum_squared = int((int(yy)-int(indices[0,yy,xx]))**2 +
                                      (int(xx)-int(indices[1,yy,xx]))**2)
                radius = int(np.ceil(distances[yy,xx]))
                y0,y1 = max(0,int(yy)-radius),min(domain.shape[0],int(yy)+radius+1)
                x0,x1 = max(0,int(xx)-radius),min(domain.shape[1],int(xx)+radius+1)
                iy,ix = np.ogrid[y0:y1,x0:x1]
                exact_distance = (iy-int(yy))**2+(ix-int(xx))**2 == minimum_squared
                matches = (supported[y0:y1,x0:x1] & exact_distance &
                           (labels[y0:y1,x0:x1] == preferred[yy,xx]))
                if matches.any():
                    result[yy,xx] = preferred[yy,xx]
                    count += 1
            tie_audit.update(applicable=True,preference_candidates=len(candidates),
                             equal_distance_preferences=count)
        if audit is not None:
            if not isinstance(audit,dict):
                raise TypeError("Fill receipt audit must be a dictionary")
            audit.update(tie_audit,modelled_pixels=int(supported.sum()),
                filled_observed_pixels=int((filled & visible).sum()),
                filled_unobserved_pixels=int((filled & ~visible).sum()))
    return result, dict(modelled=supported, filled_observed=filled & visible,
                        filled_unobserved=filled & ~visible)


def interior_difference(before, after, old_domain, pixel_spacing_m, *, source_changed=False):
    if source_changed:
        return dict(excluded_source_changed=True)
    old, new, domain = np.asarray(before), np.asarray(after), np.asarray(old_domain, bool)
    if old.shape != new.shape or domain.shape != old.shape:
        raise ValueError("Interior QA requires the same original pixel grid")
    padded = np.pad(domain, 1, constant_values=False)
    distance = distance_transform_edt(padded, sampling=pixel_spacing_m)[1:-1, 1:-1] * 1000
    changed = (old != new) & domain
    bands = {"0_10_mm": (distance <= 10), "10_30_mm": (distance > 10) & (distance <= 30),
             "over_30_mm": distance > 30}
    counts = {key: int(np.count_nonzero(changed & band)) for key, band in bands.items()}
    return dict(excluded_source_changed=False, changed_pixels=counts,
                interior_exact=counts["over_30_mm"] == 0)
