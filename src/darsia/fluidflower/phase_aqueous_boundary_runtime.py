"""Explicit aqueous-boundary output revision over a frozen C1--C5 sequence.

The parent receives the original optical fields and owns all temporal history.
Only this frame's final aqueous boundary is revised. Profiles without the
explicit top-level ``boundary_refinement`` retain their original runtime hash.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_aqueous_boundary as boundary
from . import phase_presence_context as presence
from .project import sha256_file

VERSION = "native-aqueous-boundary-v1"


def split_profile(profile):
    """Validate the explicit revision and return a non-mutating parent view."""
    from . import phase_reference_support, phase_reviewed_reference
    from . import phase_facies_reference, phase_native_reference
    supported = {presence.VERSION, phase_reference_support.VERSION,
                 phase_reviewed_reference.VERSION, phase_facies_reference.VERSION,
                 phase_native_reference.VERSION}
    if profile.get("format") not in supported:
        raise ValueError("Aqueous boundary refinement requires a C1--C5 presence profile")
    specification = profile.get("boundary_refinement")
    if not isinstance(specification, dict) or specification.get("format") != VERSION:
        raise ValueError("Unknown aqueous boundary-refinement specification")
    if set(specification) - {"format", "settings"}:
        raise ValueError("Unknown boundary-refinement fields")
    values = specification.get("settings", {})
    if not isinstance(values, dict):
        raise ValueError("Boundary-refinement settings must be an object")
    settings = boundary.AqueousBoundarySettings(**values)
    inherited = {key: value for key, value in profile.items() if key != "boundary_refinement"}
    return inherited, settings


def runtime_identity(inherited_identity, settings):
    """Bind the inherited runtime, canonical settings, and both new modules."""
    payload = {"format": VERSION, "inherited_runtime": inherited_identity,
               "settings": asdict(settings)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for module_file in (boundary.__file__, __file__):
        digest.update(sha256_file(Path(module_file)).encode())
    return digest.hexdigest()


def _presence_sequence(sequence):
    current, seen = sequence, set()
    while id(current) not in seen:
        if isinstance(current, presence.PresenceSequence):
            return current
        seen.add(id(current))
        current = getattr(current, "parent", None)
        if current is None:
            break
    raise ValueError("Aqueous boundary refinement requires an inherited PresenceSequence")


def continuous_supported_mask(sequence):
    """Interpolate certified support continuously without crossing facies.

    Same-facies normalization excludes other materials and invalid coarse cells;
    native cells with no corresponding coarse evidence remain unsupported.
    This reads the parent's just-computed support and never changes its history.
    """
    parent = _presence_sequence(sequence)
    context = parent.context
    native_shape = context.domain.shape
    coarse = np.asarray(parent.previous_supported, bool)
    valid = np.asarray(parent.valid, bool)
    if coarse.shape != valid.shape:
        raise ValueError("Inherited support and valid coarse grid must agree")
    facies = np.asarray(context.facies)
    coarse_facies = cv2.resize(facies.astype(np.float32), coarse.shape[::-1],
                              interpolation=cv2.INTER_NEAREST)
    own_valid = cv2.resize(valid.astype(np.uint8), native_shape[::-1],
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    own_facies = cv2.resize(coarse_facies, native_shape[::-1],
                            interpolation=cv2.INTER_NEAREST)
    eligible = context.domain & own_valid & (own_facies == facies)
    result = np.zeros(native_shape, bool)
    for label in np.unique(facies[context.domain]):
        same = valid & (coarse_facies == label)
        weights = cv2.resize(same.astype(np.float32), native_shape[::-1],
                             interpolation=cv2.INTER_LINEAR)
        weighted = cv2.resize((same & coarse).astype(np.float32), native_shape[::-1],
                              interpolation=cv2.INTER_LINEAR)
        selected = eligible & (facies == label) & (weights > 1e-6)
        result[selected] = weighted[selected] / weights[selected] >= .5
    return result


class AqueousBoundarySequence:
    def __init__(self, inherited, settings):
        _presence_sequence(inherited)
        self.inherited = inherited
        self.context = inherited.context
        self.settings = settings
        self.last_diagnostics = {}

    def __getattr__(self, name):
        # Capability detection remains exact: C1 must not accidentally expose
        # apply_with_reference/bind_reference, which it does not implement.
        inherited = object.__getattribute__(self, "inherited")
        method = getattr(inherited, name)
        if name in ("apply_with_reference", "apply_rgb"):
            def wrapped(predictions, elapsed_h, *args, **kwargs):
                output, report = method(predictions, elapsed_h, *args, **kwargs)
                return self._finish(predictions, output, report)
            return wrapped
        return method

    def apply(self, predictions, elapsed_h):
        output, report = self.inherited.apply(predictions, elapsed_h)
        return self._finish(predictions, output, report)

    def _finish(self, original, inherited, report):
        domain = self.context.domain
        before = domain & (np.asarray(inherited["co2_presence_probability"]) >= .5)
        negative = bool(report.get("pre_injection_negative_control", False))
        if negative:
            result = inherited
            details = dict(format=VERSION, settings=asdict(self.settings),
                bypassed_pre_injection=True, band_pixels=0, added_aqueous_pixels=0,
                removed_aqueous_pixels=0, gas_labels_unchanged=True,
                history_updates=False, calibrated_probability=False)
        else:
            support = continuous_supported_mask(self.inherited) | before
            result, details = boundary.refine(inherited,
                original["co2_presence_probability"], self.context, support, self.settings)
            details = {**details, "bypassed_pre_injection": False}
        after = domain & (np.asarray(result["co2_presence_probability"]) >= .5)
        self.last_diagnostics = {**self.inherited.last_diagnostics,
            "boundary_before_co2": before,
            "boundary_before_gas": before & (np.asarray(inherited["gas_probability_conditional_on_co2"]) >= .6),
            "boundary_added_aqueous": after & ~before,
            "boundary_removed_aqueous": before & ~after}
        return result, {**report, "boundary_refinement": details}

    def write_diagnostics(self, *args, **kwargs):
        # Diagnostic additions are not observed history and cannot seed any
        # parent sequence. Keep inherited diagnostic names/counts untouched.
        self.inherited.last_diagnostics.update({key: value for key, value in self.last_diagnostics.items()
                                               if key.startswith("boundary_")})
        return self.inherited.write_diagnostics(*args, **kwargs)
