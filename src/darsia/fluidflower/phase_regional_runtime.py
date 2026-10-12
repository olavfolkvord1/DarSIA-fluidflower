"""Explicit multiresolution candidate: native gas, regional liquid, one contour.

This is a new numerical method, not a transparent acceleration of old models.
The frozen completion network sees different conservative proposals and must
be validated (or refitted) before a candidate is promoted. Existing profiles
continue to dispatch their original providers and retain their identities.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_decision_runtime as legacy
from . import phase_decision_versions as decisions
from . import phase_native_reference as native
from . import phase_reviewed_reference as reviewed, phase_facies_reference as facies_reference
from . import phase_plume, phase_plume_envelope
from . import phase_aqueous_context_completion as completion

VERSION = "regional-evidence-single-contour-runtime-v1"
PROFILE_FILE = legacy.PROFILE_FILE
PROFILE_KEY = legacy.PROFILE_KEY
split_profile = legacy.split_profile


@dataclass(frozen=True)
class RegionalSettings:
    spacing_m: float = .001

    def __post_init__(self):
        if not np.isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("Positive finite regional spacing required")


def validate(spec):
    if set(spec) != {"format", "settings", "decision"} or spec["format"] != VERSION:
        raise ValueError("Invalid regional decision specification")
    RegionalSettings(**spec["settings"])
    # No nested regional wrapper or legacy/local postprocessing stack.
    from . import phase_decision_age_runtime as age
    if spec["decision"].get("format") not in (legacy.VERSION, age.VERSION):
        raise ValueError("Regional runtime requires one supported final decision")
    impl = decisions.implementation(spec["decision"])
    if set(spec["decision"]) != {"format", "settings"}:
        raise ValueError("Invalid final decision specification")
    impl.core.Settings(**spec["decision"]["settings"])
    return spec


def candidate_profile(profile, *, spacing_m=.001):
    """Create an explicit candidate without editing its frozen parent profile."""
    if PROFILE_KEY not in profile:
        raise ValueError("A unified parent model is required")
    candidate = dict(profile)
    candidate[PROFILE_KEY] = validate(dict(format=VERSION,
        settings=asdict(RegionalSettings(spacing_m)), decision=copy.deepcopy(profile[PROFILE_KEY])))
    return candidate


def attach_profile(root, inherited):
    path = Path(root) / PROFILE_FILE
    if not path.is_file():
        return inherited
    spec = validate(json.loads(path.read_text(encoding="utf-8")))
    if inherited is None or completion.PROFILE_KEY not in inherited:
        raise ValueError("Regional decision requires a frozen completion provider")
    if "local_aqueous_support" in inherited:
        raise ValueError("Local search cannot be stacked after a regional decision")
    return {**inherited, PROFILE_KEY: spec}


def runtime_identity(inherited_identity, spec):
    validate(spec)
    payload = dict(format=VERSION, specification=spec,
        parent_identity=inherited_identity,
        final_identity=decisions.runtime_identity(inherited_identity, spec["decision"]),
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def regional_context(context, spacing_m):
    """Pool fields once; labels stay categorical and domain holes fail closed."""
    h, w = context.domain.shape
    dy, dx = abs(np.diff(context.y_m)[0]), np.diff(context.x_m)[0]
    target = (max(2, min(w, round(w*dx/spacing_m))),
              max(2, min(h, round(h*dy/spacing_m))))
    def pool(a):
        return cv2.resize(np.asarray(a, np.float32), target, interpolation=cv2.INTER_AREA)
    domain = pool(context.domain) >= .999
    facies = cv2.resize(np.asarray(context.facies, np.float32), target,
                        interpolation=cv2.INTER_NEAREST)
    permeability = None
    if context.permeability is not None:
        permeability = np.exp(pool(np.log(np.where(context.domain, context.permeability, 1.))))
    return phase_plume.PlumeContext(domain, facies,
        np.linspace(context.x_m[0], context.x_m[-1], target[0]),
        np.linspace(context.y_m[0], context.y_m[-1], target[1]),
        context.injection_stages, permeability)


class _PooledGasSource:
    """Regional geometry adapter; native gas physics runs exactly once/frame."""
    def __init__(self, geometry):
        self.base = geometry
        self.pending = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def apply(self, predictions, elapsed_h):
        if self.pending is None or self.pending[0] != float(elapsed_h):
            raise ValueError("Missing same-frame native gas evidence")
        _, fields, report = self.pending
        self.pending = None
        return fields, dict(report)


class RegionalEvidenceSequence:
    """One liquid evidence provider; old native fringe/contour stages are absent."""
    def __init__(self, context, profile, settings):
        providers = {native.VERSION: (native, native.NativeReferenceSequence),
                     reviewed.VERSION: (reviewed, reviewed.ReviewedReferenceSequence),
                     facies_reference.VERSION: (facies_reference, facies_reference.FaciesReferenceSequence)}
        if profile.get("format") not in providers:
            raise ValueError("Regional candidate requires a calibrated reference parent")
        module, provider_class = providers[profile["format"]]
        self.context = context
        self.settings = settings
        inherited = copy.deepcopy({k: v for k, v in profile.items() if k != "boundary_refinement"})
        module.validate_profile(inherited)
        gas_spec = inherited["reference_support"]["presence_support"]["gas_support"]
        cls = (phase_plume_envelope.LocalFloorSequence
               if gas_spec["format"] == phase_plume_envelope.VERSION else phase_plume.PlumeSequence)
        self.base = cls(context, phase_plume.PlumeSettings(**gas_spec["settings"]))
        reduced = regional_context(context, settings.spacing_m)
        # Preserve the physical scale of the causal graph instead of shrinking
        # it a second time by the old native-pixel reduction factor.
        scale = min(1., self.base.target[0] / reduced.domain.shape[1])
        nodes = [inherited, inherited["reference_support"],
                 inherited["reference_support"]["presence_support"], gas_spec]
        for item in nodes:
            item["settings"]["structure_scale"] = scale
        if "boundary_settings" in inherited:
            inherited["boundary_settings"]["maximum_band_cells"] = 0.
        self.parent = provider_class(reduced, inherited)
        self.parent.parent.base = _PooledGasSource(self.parent.parent.base)
        self.target = reduced.domain.shape[::-1]
        self.last_diagnostics = {}
        self.last_time = None

    def small(self, values):
        return cv2.resize(np.asarray(values, np.float32), self.target, interpolation=cv2.INTER_AREA)

    def full(self, mask):
        return cv2.resize(np.asarray(mask, np.uint8), self.context.domain.shape[::-1],
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    def bind_reference(self, baseline_rgb):
        # Pool before quantization/chromatic ratios, keeping sub-byte changes.
        self.parent.bind_reference(self.regional_rgb(baseline_rgb))

    def regional_rgb(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (*self.context.domain.shape, 3):
            raise ValueError("Registered native uint8 RGB required")
        # INTER_AREA accumulation can overshoot 255 by a float32 ulp.
        return np.clip(self.small(rgb)/255., 0., 1.)

    def apply_with_reference(self, predictions, elapsed_h, current_rgb):
        time = float(elapsed_h)
        if not np.isfinite(time) or (self.last_time is not None and time <= self.last_time):
            raise ValueError("Regional observations require strictly increasing times")
        gas_fields, report = self.base.apply(predictions, time)
        names = ("co2_presence_probability", "gas_probability_conditional_on_co2")
        inherited = {key: self.small(gas_fields[key]) for key in names}
        raw = {key: self.small(predictions[key]) for key in names}
        self.parent.parent.base.pending = (time, inherited, report)
        evidence, report = self.parent.apply_with_reference(raw, time, self.regional_rgb(current_rgb))
        domain = self.context.domain
        native_gas = domain & (gas_fields[names[0]] >= .5) & (gas_fields[names[1]] >= .6)
        liquid = self.full(evidence[names[0]] >= .5) & domain
        presence = liquid | native_gas
        co2, gas = presence.astype(np.float32), native_gas.astype(np.float32)
        self.last_diagnostics = dict(spatial_only_presence=(
            self.full(self.parent.last_diagnostics["spatial_only_presence"]) & domain) | native_gas)
        self.last_time = time
        report["regional_evidence"] = dict(format=VERSION, spacing_m=self.settings.spacing_m,
            native_shape=list(domain.shape), regional_shape=list(self.parent.context.domain.shape),
            gas_evaluations=1, gas_kept_at_native_resolution=True,
            inherited_native_boundary_refinement=False, final_contour_owner="unified_decision",
            completion_input_changed=True, production_validated=False)
        return {**predictions, names[0]: co2, names[1]: gas,
            "aqueous_co2_probability": co2-gas, "free_co2_gas_probability": gas,
            "untouched_water_probability": 1-co2}, report


def create_sequence(context, conservative_profile, specification, checkpoint):
    validate(specification)
    provider = RegionalEvidenceSequence(context, conservative_profile,
                                        RegionalSettings(**specification["settings"]))
    # The existing decision owns the ONLY final contour and learned completion.
    return decisions.DecisionSequence(context, provider, specification["decision"], checkpoint)
