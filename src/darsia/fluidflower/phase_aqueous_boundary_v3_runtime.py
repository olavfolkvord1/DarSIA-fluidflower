"""Explicit v3 fringe dispatch over the unchanged v1 and causal sequence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from . import phase_aqueous_boundary_runtime as legacy
from . import phase_aqueous_boundary_v3 as boundary
from .project import sha256_file

VERSION = boundary.VERSION


def split_profile(profile):
    specification = profile.get('boundary_refinement')
    if not isinstance(specification, dict) or specification.get('format') != VERSION:
        raise ValueError('Unknown aqueous boundary-v3 specification')
    if set(specification) - {'format', 'settings'} or not isinstance(specification.get('settings', {}), dict):
        raise ValueError('Unknown aqueous boundary-v3 fields or settings')
    settings = boundary.AqueousBoundarySettings(**specification.get('settings', {}))
    inherited, _ = legacy.split_profile({**profile, 'boundary_refinement': dict(
        format=legacy.VERSION, settings=asdict(boundary.legacy_settings(settings)))})
    return inherited, settings


def runtime_identity(inherited_identity, settings):
    payload = dict(format=VERSION, inherited_runtime=inherited_identity,
        legacy_runtime=legacy.runtime_identity(inherited_identity, boundary.legacy_settings(settings)),
        settings=asdict(settings))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for filename in (__file__, boundary.__file__):
        digest.update(sha256_file(Path(filename)).encode())
    return digest.hexdigest()


class AqueousBoundarySequence(legacy.AqueousBoundarySequence):
    def _finish(self, original, inherited, report):
        result, report = super()._finish(original, inherited, report)
        detail = report['boundary_refinement']
        detail.update(format=VERSION, settings=asdict(self.settings))
        if not report.get('pre_injection_negative_control'):
            result, fringe = boundary.trim_fringe(result,
                original['co2_presence_probability'], self.context, self.settings)
            detail['fringe_cleanup'] = fringe
            result, gaps = boundary.complete_gaps(result, original['co2_presence_probability'],
                self.context, legacy.continuous_supported_mask(self.inherited), self.settings)
            detail['gap_completion'] = gaps
        before = self.last_diagnostics['boundary_before_co2']
        after = self.context.domain & (result['co2_presence_probability'] >= .5)
        self.last_diagnostics['boundary_added_aqueous'] = after & ~before
        self.last_diagnostics['boundary_removed_aqueous'] = before & ~after
        detail['added_aqueous_pixels'] = int((after & ~before).sum())
        detail['removed_aqueous_pixels'] = int((before & ~after).sum())
        weak = before & (original['co2_presence_probability'] < .5)
        detail['inherited_weak_promotions_before'] = int(weak.sum())
        detail['inherited_weak_promotions_protected'] = int((weak & after).sum())
        return result, report
