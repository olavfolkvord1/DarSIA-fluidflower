"""Opt-in, baseline-backed completion next to an already detected plume.

The regional color signal is evaluated before any phase mask. Additions need a
connected path to this frame's inherited plume. They never feed its history,
remove an existing phase, or convert gas to water. This is phase support, not a
concentration or a calibrated probability estimate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

VERSION = "local-reference-aqueous-support-v1"
PROFILE_FILE = "phase_local_aqueous_support.json"
PROFILE_KEY = "local_aqueous_support"


@dataclass(frozen=True)
class Settings:
    working_spacing_m: float = .001
    smoothing_m: float = .002
    search_radius_m: float = .025
    minimum_chromatic_change: float = .006
    noise_multiplier: float = 3.
    minimum_direction_cosine: float = .5
    minimum_support_area_m2: float = .000004
    minimum_controls: int = 64
    minimum_seeds: int = 12
    minimum_model_aqueous_probability: float = .2
    use_model_negative_veto: bool = False
    continuity_radius_m: float = .003
    contour_smoothing_m: float = .001

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "use_model_negative_veto":
                if not isinstance(value, bool): raise ValueError("Model veto must be an explicit boolean")
                continue
            if key in {"continuity_radius_m", "contour_smoothing_m"} and value == 0:
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite local-support setting required: {key}")
        if self.minimum_direction_cosine > 1:
            raise ValueError("Direction cosine cannot exceed one")
        if self.minimum_model_aqueous_probability > 1:
            raise ValueError("Model probability threshold cannot exceed one")


def attach_profile(root, inherited):
    path = Path(root) / PROFILE_FILE
    if not path.exists():
        return inherited
    spec = json.loads(path.read_text(encoding="utf-8"))
    if inherited is None or set(spec) != {"format", "settings"} or spec["format"] != VERSION:
        raise ValueError("Local aqueous support requires an inherited model and a supported profile")
    Settings(**spec["settings"])
    return {**inherited, PROFILE_KEY: spec}


def split_profile(profile):
    return {k: v for k, v in profile.items() if k != PROFILE_KEY}, profile[PROFILE_KEY]


def runtime_identity(inherited, specification):
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps({"inherited": inherited, "specification": specification,
                                     "source_sha256": source}, sort_keys=True).encode()).hexdigest()


def regional_support(current, baseline, labels, domain, facies, spacing_m, settings=Settings(), *, aqueous_probability=None):
    """Return continuous support on the native grid plus interpretable diagnostics.

    Chromaticity suppresses common brightness variation; unchanged controls in
    each sand type estimate its remaining drift and noise. With insufficient
    controls or a measurable aqueous color direction, that facies abstains.
    """
    labels, domain, facies = np.asarray(labels), np.asarray(domain, bool), np.asarray(facies)
    shape = domain.shape
    if labels.shape != shape or facies.shape != shape or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Registered domain, facies and categorical phase labels are required")
    for rgb in (current, baseline):
        if np.asarray(rgb).shape != (*shape, 3) or np.asarray(rgb).dtype != np.uint8:
            raise ValueError("Local support requires registered native uint8 RGB references")
    dy, dx = map(float, spacing_m)
    if not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0:
        raise ValueError("Positive registered pixel spacing is required")
    target = (max(1, round(shape[1]*dx/settings.working_spacing_m)),
              max(1, round(shape[0]*dy/settings.working_spacing_m)))
    area = lambda a: cv2.resize(np.asarray(a, np.float32), target, interpolation=cv2.INTER_AREA)
    model_support = np.ones(target[::-1], bool)
    if aqueous_probability is not None:
        probability = np.asarray(aqueous_probability, np.float32)
        if probability.shape not in (shape, target[::-1]) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Aqueous model probabilities must use the native or working grid")
        if probability.shape != target[::-1]: probability = area(probability)
        model_support = probability >= settings.minimum_model_aqueous_probability
    dom = area(domain) > .95
    types = cv2.resize(facies.astype(np.float32), target, interpolation=cv2.INTER_NEAREST)
    seed = area((labels > 0) & domain) >= .5
    aqueous = area((labels == 1) & domain) >= .8
    empty = np.zeros(shape, bool)
    if not seed.any():
        return empty, {"added_pixels": 0, "reason": "no_inherited_plume", "facies": []}
    step_y, step_x = shape[0]*dy/target[1], shape[1]*dx/target[0]
    distance = ndi.distance_transform_edt(~seed, sampling=(step_y, step_x))
    near = distance <= settings.search_radius_m
    sigma = (settings.smoothing_m/step_y, settings.smoothing_m/step_x)
    images = [area(a)/255. for a in (current, baseline)]
    # Pool numerator and denominator separately: individual dark grains cannot
    # produce a large ratio before the regional average is taken.
    chroma = [a / np.maximum(a.sum(axis=-1, keepdims=True), .08) for a in images]
    delta = chroma[0] - chroma[1]
    eligible = np.zeros(dom.shape, bool)
    records = []
    for label in np.unique(types[dom]):
        mask = dom & (types == label)
        weight = ndi.gaussian_filter(mask.astype(np.float32), sigma)
        smooth = np.stack([ndi.gaussian_filter(delta[..., c]*mask, sigma) /
                           np.maximum(weight, 1e-6) for c in range(3)], axis=-1)
        controls = mask & ~near
        core = mask & aqueous
        record = {"facies": int(label), "controls": int(controls.sum()), "seeds": int(core.sum())}
        if controls.sum() < settings.minimum_controls or core.sum() < settings.minimum_seeds:
            records.append({**record, "status": "insufficient_controls_or_seeds"})
            continue
        drift = np.median(smooth[controls], axis=0)
        residual = smooth - drift
        direction = np.median(residual[core], axis=0)
        length = float(np.linalg.norm(direction))
        if length < settings.minimum_chromatic_change:
            records.append({**record, "status": "no_measurable_aqueous_direction"})
            continue
        direction /= length
        projection = residual @ direction
        samples = projection[controls]
        noise = 1.4826*float(np.median(np.abs(samples-np.median(samples))))
        threshold = max(settings.minimum_chromatic_change, settings.noise_multiplier*noise)
        cosine = projection / np.maximum(np.linalg.norm(residual, axis=-1), 1e-8)
        eligible |= mask & near & model_support & (projection >= threshold) & (cosine >= settings.minimum_direction_cosine)
        records.append({**record, "status": "evaluated", "threshold": threshold,
                        "noise": noise, "drift": drift.tolist(), "direction": direction.tolist()})
    # Propagation cannot jump a no-signal gap, cross the rig boundary, or grow
    # repeatedly from a previous addition. Four-neighbour topology is explicit.
    connected = ndi.binary_propagation(seed & dom, mask=(eligible | seed) & dom)
    candidate = connected & eligible & ~seed
    components, count = ndi.label(candidate)
    sizes = np.bincount(components.ravel(), minlength=count+1)
    keep = sizes*step_y*step_x >= settings.minimum_support_area_m2
    keep[0] = False
    candidate &= keep[components]
    # Interpolating a regional support field gives sub-grid boundaries without
    # re-applying a grain-scale threshold that recreates holes and speckles.
    candidate |= connected & eligible & seed
    native = cv2.resize(candidate.astype(np.float32), shape[::-1], interpolation=cv2.INTER_LINEAR) >= .5
    added = native & domain & (labels == 0)
    if settings.continuity_radius_m and (connected & eligible).any():
        # Complete the UNION of inherited and newly recovered phase. Smoothing
        # only the added mask leaves the inherited grain-scale fringe intact.
        union = ((labels > 0) | added) & domain
        continuous, continuity = continuous_boundary(union, connected & eligible, dom,
            (step_y, step_x), settings, model_support=model_support)
        added |= continuous & domain & (labels == 0)
    else:
        continuity = {"added_pixels": 0, "reason": "no_supported_boundary_or_disabled"}
    return added, {"added_pixels": int(added.sum()), "working_shape": list(dom.shape),
                   "model_negative_veto": aqueous_probability is not None,
                   "search_radius_m": settings.search_radius_m, "facies": records,
                   "continuity": continuity}


def continuous_boundary(native_union, optical_support, domain, spacing_m, settings, *, model_support=None):
    """Fill millimetre-scale interruptions without erasing the detected fringe.

    Geometry regularizes the whole detected region, not individual RGB pixels.
    Closing and a smooth signed-distance contour are restricted to a narrow
    band next to reference-supported phase. Large holes, separated plumes and
    gas labels are not reinterpreted by this operation.
    """
    sy, sx = spacing_m
    target = domain.shape[::-1]
    fraction = cv2.resize(native_union.astype(np.float32), target, interpolation=cv2.INTER_AREA)
    # Preserve the spatial reach of weak fringe detections. Majority pooling
    # would discard precisely those pixels the local search recovered.
    support = (fraction > .01) & domain
    radius = settings.continuity_radius_m
    ry, rx = max(1, int(np.ceil(radius/sy))), max(1, int(np.ceil(radius/sx)))
    yy, xx = np.ogrid[-ry:ry+1, -rx:rx+1]
    kernel = ((yy*sy)**2+(xx*sx)**2 <= radius**2+1e-12).astype(np.uint8)
    padded = np.pad(support.astype(np.uint8), ((ry+2, ry+2), (rx+2, rx+2)))
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
    closed = closed[ry+2:-ry-2, rx+2:-rx-2].astype(bool) | support
    holes = ndi.binary_fill_holes(closed) & ~closed & domain
    hole_labels, count = ndi.label(holes)
    hole_areas = np.bincount(hole_labels.ravel(), minlength=count+1)*sy*sx
    fill = hole_areas <= np.pi*radius**2
    fill[0] = False
    closed |= fill[hole_labels]
    signed = ndi.distance_transform_edt(closed, sampling=spacing_m)-ndi.distance_transform_edt(~closed, sampling=spacing_m)
    if settings.contour_smoothing_m:
        signed = ndi.gaussian_filter(signed, (settings.contour_smoothing_m/sy, settings.contour_smoothing_m/sx))
    # The half-cell allowance encloses the pooled weak fringe instead of
    # shrinking it to a majority contour. It is a spatial prior, not evidence
    # of a calibrated optical detection limit.
    signed += .5*max(sy, sx)
    band_m = radius+2*settings.contour_smoothing_m
    active = ndi.distance_transform_edt(~optical_support, sampling=spacing_m) <= band_m
    nearby = ndi.distance_transform_edt(~support, sampling=spacing_m) <= band_m
    allowed = domain & active & nearby
    if model_support is not None: allowed &= model_support
    field = cv2.resize(signed.astype(np.float32), native_union.shape[::-1], interpolation=cv2.INTER_LINEAR) >= 0
    allowed_native = cv2.resize(allowed.astype(np.uint8), native_union.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    field &= allowed_native
    return field, dict(method="unified-plume-metric-closing-signed-distance-v1",
        radius_m=radius, smoothing_m=settings.contour_smoothing_m, maximum_completion_band_m=band_m,
        maximum_enclosed_hole_area_m2=float(np.pi*radius**2),
        added_pixels=int(np.count_nonzero(field & ~native_union)), removed_pixels=0)


class LocalAqueousSequence:
    def __init__(self, inherited, specification):
        self.inherited, self.context = inherited, inherited.context
        self.settings = Settings(**specification["settings"])
        self.baseline = None
        self.last_diagnostics = {}
        self._captured_completion = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "inherited"), name)

    def bind_reference(self, baseline_rgb):
        if self.baseline is not None:
            raise ValueError("The local-support baseline is immutable")
        if hasattr(self.inherited, "bind_reference"):
            self.inherited.bind_reference(baseline_rgb)
        self.baseline = np.array(baseline_rgb, copy=True)

    def apply(self, predictions, elapsed_h):
        raise ValueError("Local aqueous support requires current and pre-injection RGB")

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        from .phase_aqueous_context_completion import delivered_labels, apply_probability
        if self.baseline is None:
            raise ValueError("Bind the pre-injection reference first")
        # Capture the already evaluated learned map. No second network call or
        # duplicate optical features are required for the local evidence test.
        from .phase_aqueous_context_completion import AqueousCompletionSequence, _Predictor
        if self.settings.use_model_negative_veto and isinstance(self.inherited, AqueousCompletionSequence) and self._captured_completion is None:
            self._captured_completion = _CapturedProbability(self.inherited._predictor or _Predictor(self.inherited.specification))
            self.inherited._predictor = self._captured_completion
        if hasattr(self.inherited, "apply_with_reference"):
            output, report = self.inherited.apply_with_reference(predictions, elapsed_h, current_rgb)
        else:
            output, report = self.inherited.apply(predictions, elapsed_h)
        added = np.zeros(self.context.domain.shape, bool)
        details = {"reason": "pre_injection_negative_control", "added_pixels": 0}
        if not report.get("pre_injection_negative_control", False) and elapsed_h > 0:
            spacing = (abs(float(self.context.y_m[1]-self.context.y_m[0])),
                       abs(float(self.context.x_m[1]-self.context.x_m[0])))
            added, details = regional_support(current_rgb, self.baseline,
                delivered_labels(output, self.context.domain), self.context.domain,
                self.context.facies, spacing, self.settings,
                aqueous_probability=((self._captured_completion.last if self._captured_completion is not None
                    else output.get("aqueous_co2_probability", np.asarray(output["co2_presence_probability"])*(1-np.asarray(output["gas_probability_conditional_on_co2"]))))
                    if self.settings.use_model_negative_veto else None))
            # .5 represents the categorical decision only, never confidence.
            output, added = apply_probability(output, added.astype(np.float32)*.5, self.context.domain)
        self.last_diagnostics = {**self.inherited.last_diagnostics, "local_reference_added_aqueous": added}
        return output, {**report, PROFILE_KEY: {"format": VERSION, **details,
            "removed_pixels": 0, "gas_unchanged": True, "history_updates": False,
            "calibrated_probability": False}}

    def write_diagnostics(self, *args, **kwargs):
        from .phase_presence_context import PresenceSequence
        return PresenceSequence.write_diagnostics(self, *args, **kwargs)


class _CapturedProbability:
    def __init__(self, predictor):
        self.predictor, self.last = predictor, None

    def __call__(self, inputs):
        self.last = self.predictor(inputs)
        return self.last

    def __deepcopy__(self, memo):
        # Reference replays may share immutable weights but must not share a
        # mutable last-frame capture with the timeline they were copied from.
        return type(self)(self.predictor)
