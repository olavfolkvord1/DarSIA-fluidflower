"""Explicit v2 dispatch; all inherited temporal history remains untouched."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from . import phase_aqueous_boundary_runtime as v1
from . import phase_aqueous_boundary_v2 as boundary
from .project import sha256_file

VERSION = boundary.VERSION


def split_profile(profile):
    specification = profile.get("boundary_refinement", {})
    if specification.get("format") != VERSION:
        raise ValueError("Unknown aqueous boundary v2 specification")
    # Reuse all established profile and setting checks without altering v1.
    inherited, checked = v1.split_profile({**profile, "boundary_refinement": {
        **specification, "format": v1.VERSION}})
    return inherited, boundary.AqueousBoundarySettings(**asdict(checked))


def runtime_identity(inherited_identity, settings):
    payload = dict(format=VERSION, inherited_runtime=inherited_identity,
        base_boundary_runtime=v1.runtime_identity(inherited_identity, settings),
        settings=asdict(settings))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in (boundary.__file__, __file__):
        digest.update(sha256_file(Path(path)).encode())
    return digest.hexdigest()


class AqueousBoundarySequence(v1.AqueousBoundarySequence):
    def __getattr__(self, name):
        inherited = object.__getattribute__(self, "inherited")
        method = getattr(inherited, name)
        if name in ("apply_with_reference", "apply_rgb"):
            def wrapped(predictions, elapsed_h, *args, **kwargs):
                output, report = method(predictions, elapsed_h, *args, **kwargs)
                result, complete = self._finish(predictions, output, report)
                if report.get("pre_injection_negative_control") or not args:
                    return result, complete
                calibration = getattr(inherited, "calibration", None)
                baseline = getattr(inherited, "calibration_baseline", None)
                controls = report.get("reference_support", {}).get("facies_controls", [])
                if calibration is None or baseline is None or not controls:
                    return self._clean(result, predictions, complete)
                co2 = np.asarray(result["co2_presence_probability"]) >= .5
                gas = co2 & (np.asarray(result["gas_probability_conditional_on_co2"]) >= .6)
                score, eligible = boundary.reference_evidence(baseline, args[0],
                    self.context, gas, controls, inherited.settings, calibration, self.settings)
                support = np.asarray(inherited.last_diagnostics["trusted_history"], bool)
                result, detail = boundary.complete_reference_gaps(result, self.context,
                    score, eligible, support, self.settings)
                after = np.asarray(result["co2_presence_probability"]) >= .5
                self.last_diagnostics["boundary_added_aqueous"] |= after & ~co2
                self.last_diagnostics["boundary_reference_added_aqueous"] = after & ~co2
                details = complete["boundary_refinement"]
                details.update(format=VERSION, reference_completion=detail,
                    added_aqueous_pixels=details["added_aqueous_pixels"] + detail["reference_added_aqueous_pixels"])
                return self._clean(result, predictions, complete, eligible & (score >= 1))
            return wrapped
        return method

    def apply(self, predictions, elapsed_h):
        output, report = self.inherited.apply(predictions, elapsed_h)
        result, report = self._finish(predictions, output, report)
        return self._clean(result, predictions, report)

    def _clean(self, result, original, report, protected_reference=None):
        if report.get('pre_injection_negative_control'):
            return result, report
        before = np.asarray(result['co2_presence_probability']) >= .5
        result, detail = boundary.prune_uncertain_islands(result,
            original['co2_presence_probability'], self.context,
            protected_reference=protected_reference, settings=self.settings)
        after = np.asarray(result['co2_presence_probability']) >= .5
        removed = before & ~after
        # Completion can create attached pixels only, nevertheless reconstruct
        # the complete diagnostic change against the inherited frame exactly.
        inherited_before = self.last_diagnostics['boundary_before_co2']
        self.last_diagnostics['boundary_added_aqueous'] = after & ~inherited_before
        self.last_diagnostics['boundary_removed_aqueous'] = inherited_before & ~after
        self.last_diagnostics['boundary_uncertain_islands_removed'] = removed
        report['boundary_refinement']['uncertain_island_cleanup'] = detail
        report['boundary_refinement']['added_aqueous_pixels'] = int(self.last_diagnostics['boundary_added_aqueous'].sum())
        report['boundary_refinement']['removed_aqueous_pixels'] = int(self.last_diagnostics['boundary_removed_aqueous'].sum())
        return result, report

    def _finish(self, original, inherited, report):
        result, report = super()._finish(original, inherited, report)
        report["boundary_refinement"]["format"] = VERSION
        report["boundary_refinement"]["reference_completion"] = dict(
            reference_added_aqueous_pixels=0, history_updates=False)
        return result, report
