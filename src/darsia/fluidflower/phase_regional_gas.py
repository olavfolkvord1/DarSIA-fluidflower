"""Opt-in learned confirmation of free gas within the regional CO2 map.

Dense aqueous labels and gas exclusions train the conditional gas head. This
stage makes that head effective at inference, retaining gas only where both
the pixel classifier and the learned regional head support it. It neither
expands total CO2 support nor introduces a rule based on image height.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from . import phase_optical_regions as regional
from .phase_optical_regions_runtime import OpticalRegionSequence
from .project import sha256_file

VERSION = 'learned-regional-gas-confirmation-v1'
PROFILE_KEY = 'regional_gas'


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    threshold: float = .6

    def __post_init__(self):
        if self.format != VERSION or not np.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError('Invalid learned regional gas settings')


def runtime_identity(inherited, specification):
    return hashlib.sha256(json.dumps(dict(inherited=inherited,
        settings=asdict(Settings(**specification)), code_sha256=sha256_file(Path(__file__)),
        opencv=cv2.__version__), sort_keys=True).encode()).hexdigest()


def confirm_gas(labels, regional_probability, domain, settings=Settings()):
    labels = np.asarray(labels)
    values = np.asarray(regional_probability, np.float32)
    domain = np.asarray(domain, bool)
    if (labels.ndim != 2 or labels.shape != domain.shape or not np.isin(labels, (0, 1, 2)).all()
            or values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()):
        raise ValueError('Invalid native labels, domain or regional gas probabilities')
    # Interpolate continuous scores before thresholding, as for aqueous support.
    scores = cv2.resize(values, labels.shape[::-1], interpolation=cv2.INTER_LINEAR)
    rejected = domain & (labels == 2) & (scores < settings.threshold)
    result = labels.astype(np.uint8, copy=True)
    result[rejected] = 1
    return result, rejected


class RegionalGasSequence(OpticalRegionSequence):
    def __init__(self, context, profile):
        super().__init__(context, profile)
        self.gas_settings = Settings(**profile[PROFILE_KEY])
        if 'gas_within_co2' not in self.model['heads']:
            raise ValueError('Learned gas confirmation requires the trained conditional gas head')

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.baseline is None:
            raise ValueError('Bind the registered pre-injection reference before prediction')
        p = predictions['co2_presence_probability']
        g = predictions['gas_probability_conditional_on_co2']
        raw = np.where(p >= .5, np.where(g >= .6, 2, 1), 0).astype(np.uint8)
        spacing = (abs(float(self.context.y_m[1]-self.context.y_m[0])),
                   abs(float(self.context.x_m[1]-self.context.x_m[0])))
        prepared = regional.prepare(current, self.baseline, p, g, self.context.domain, spacing, self.settings)
        labels, report = regional.predict(prepared, self.model, raw, self.context.domain)
        conditional = regional.probability(prepared['features'], self.model['heads']['gas_within_co2'])
        labels, rejected = confirm_gas(labels, conditional, self.context.domain, self.gas_settings)
        co2 = np.where(labels > 0, np.maximum(p, .5), np.minimum(p, .49)).astype(np.float32)
        gas = np.where(labels == 1, np.minimum(g, .4), g).astype(np.float32)
        result = dict(co2_presence_probability=co2, gas_probability_conditional_on_co2=gas,
            untouched_water_probability=1-co2, aqueous_co2_probability=co2*(1-gas),
            free_co2_gas_probability=co2*gas)
        self.last_diagnostics = dict(optical_region_added=(labels == 1) & (raw == 0),
            optical_region_removed=(labels == 0) & (raw == 1), regional_gas_reclassified=rejected)
        return result, dict(report, format=regional.VERSION, elapsed_h=elapsed_h,
            gas_changed_pixels=int(((labels == 2) != (raw == 2)).sum()),
            regional_gas=dict(format=VERSION, threshold=self.gas_settings.threshold,
                reclassified_gas_pixels=int(rejected.sum()), co2_support_unchanged=True,
                policy='pixel gas requires learned regional gas confirmation'),
            reference_basis='registered pre-injection RGB', context_completion_used=False,
            mass_constraints_applied=False, classification_scores_not_concentration=True)
