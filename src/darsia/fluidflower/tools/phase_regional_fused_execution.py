"""Exact execution adapter for the frozen regional + two output-guard chain.

The scientific functions and thresholds are unchanged. Categorical guards run
before constructing five full-resolution score fields, avoiding redundant
score-to-label conversions and copies between wrappers. Only the verified
float32 chain is eligible; other implementations retain their original path.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from darsia.fluidflower import phase_optical_regions as regional
from darsia.fluidflower import phase_optical_regions_runtime as runtime
from darsia.fluidflower import phase_regional_gas as regional_gas
from darsia.fluidflower import phase_region_output_guards as guards
from darsia.fluidflower import phase_free_gas_output_guard as gas_guard
from darsia.fluidflower import phase_rendering as rendering
from darsia.fluidflower.project import sha256_file


VERIFIED_SOURCES = {
    "phase_optical_regions.py": "afa4868d5eca7a4ceed46d82d4ea86a4c081dd02ed01de644224519db935b9e9",
    "phase_optical_regions_runtime.py": "89df4b55c7f517aea086799f4c19e528267655d119cdf318b9c651113816a1eb",
    "phase_regional_gas.py": "f5b7607bc9be520888332a6a776b8ed0f83c7de4c12c26ee39710976a663b750",
    "phase_region_output_guards.py": "d4d3a0a0cce59423e2666d3f1f363392826a389d97ea8d3b88e07b1950da9cb8",
    "phase_free_gas_output_guard.py": "6cfeaa6fcdd9c01b18d85eda5ba5bdf9e88674fd2370e33655223bae39bf5fff",
    "phase_rendering.py": "3407989fdd65db07bd261c3f519bbf2b4c6129b696c4d681ea1253a3d3365a94",
}


@lru_cache(maxsize=1)
def verified_sources():
    return all(sha256_file(Path(module.__file__)) == VERIFIED_SOURCES[Path(module.__file__).name]
               for module in (regional, runtime, regional_gas, guards, gas_guard, rendering))


def supports(sequence):
    if type(sequence) is not gas_guard.FreeGasSequence:
        return False
    middle = sequence.inherited
    return (type(middle) is guards.GuardedRegionSequence
            and type(middle.inherited) is regional_gas.RegionalGasSequence
            and sequence.context is middle.context is middle.inherited.context
            and verified_sources())


def apply(sequence, predictions, elapsed_h, current):
    """Return original scores, reports and diagnostics with fewer full-frame copies."""
    p = predictions.get("co2_presence_probability")
    g = predictions.get("gas_probability_conditional_on_co2")
    if (not supports(sequence) or not isinstance(p, np.ndarray) or not isinstance(g, np.ndarray)
            or p.dtype != np.float32 or g.dtype != np.float32):
        return sequence.apply_with_reference(predictions, elapsed_h, current)
    middle = sequence.inherited
    base = middle.inherited
    if base.baseline is None:
        raise ValueError("Bind the registered pre-injection reference before prediction")
    domain = base.context.domain
    spacing = (abs(float(base.context.y_m[1]-base.context.y_m[0])),
               abs(float(base.context.x_m[1]-base.context.x_m[0])))
    active = p >= .5
    raw = active.astype(np.uint8)
    raw[active & (g >= .6)] = 2
    prepared = regional.prepare(current, base.baseline, p, g, domain, spacing, base.settings)
    labels, report = regional.predict(prepared, base.model, raw, domain)
    conditional = regional.probability(prepared["features"], base.model["heads"]["gas_within_co2"])
    labels, rejected_gas = regional_gas.confirm_gas(labels, conditional, domain, base.gas_settings)
    del prepared, conditional, active
    base.last_diagnostics = dict(optical_region_added=(labels == 1) & (raw == 0),
        optical_region_removed=(labels == 0) & (raw == 1), regional_gas_reclassified=rejected_gas)
    report = dict(report, format=regional.VERSION, elapsed_h=elapsed_h,
        gas_changed_pixels=int(((labels == 2) != (raw == 2)).sum()),
        regional_gas=dict(format=regional_gas.VERSION, threshold=base.gas_settings.threshold,
            reclassified_gas_pixels=int(rejected_gas.sum()), co2_support_unchanged=True,
            policy="pixel gas requires learned regional gas confirmation"),
        reference_basis="registered pre-injection RGB", context_completion_used=False,
        mass_constraints_applied=False, classification_scores_not_concentration=True)
    guarded, domain_audit = guards.guard_labels(labels, domain, spacing, middle.settings)
    removed = (labels > 0) & (guarded == 0)
    rejected = ~np.asarray(domain, bool) | removed
    middle.last_diagnostics = {**base.last_diagnostics, "region_guard_removed": removed}
    cleaned, gas_audit = gas_guard.clean_labels(guarded, spacing, sequence.settings)
    changed = (guarded == 2) & (cleaned == 1)
    sequence.last_diagnostics = {**middle.last_diagnostics, "small_free_gas_reclassified": changed}
    co2 = np.where(labels > 0, np.maximum(p, .5), np.minimum(p, .49)).astype(np.float32, copy=False)
    gas = np.where(labels == 1, np.minimum(g, .4), g).astype(np.float32, copy=False)
    co2[rejected] = 0.
    gas[rejected | changed] = 0.
    result = dict(co2_presence_probability=co2, gas_probability_conditional_on_co2=gas,
        untouched_water_probability=1-co2, aqueous_co2_probability=co2*(1-gas),
        free_co2_gas_probability=co2*gas)
    return result, {**report, "output_guards": domain_audit, "free_gas_cleanup": gas_audit}
