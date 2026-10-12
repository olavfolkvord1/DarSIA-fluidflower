"""Opt-in native free-gas minimum, separate from combined-CO2 island removal."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .phase_rendering import two_phase_masks
from .project import sha256_file

VERSION = 'native-free-gas-minimum-v1'
PROFILE_KEY = 'free_gas_cleanup'


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    minimum_area_mm2: float = 2.5

    def __post_init__(self):
        if (self.format != VERSION or not np.isfinite(self.minimum_area_mm2)
                or not 0 < self.minimum_area_mm2 <= 25):
            raise ValueError('Invalid free-gas minimum settings')


def runtime_identity(inherited, specification):
    return hashlib.sha256(json.dumps(dict(inherited=inherited,
        settings=asdict(Settings(**specification)), code_sha256=sha256_file(Path(__file__)),
        opencv=cv2.__version__), sort_keys=True).encode()).hexdigest()


def clean_labels(labels, spacing_m, settings=Settings()):
    labels, spacing = np.asarray(labels), np.asarray(spacing_m, float)
    if labels.ndim != 2 or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError('Free-gas cleanup requires categorical phase labels')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Free-gas cleanup requires positive registered spacing')
    # Four-neighbour connectivity matches phase_decision.withdraw_small_free_gas.
    _, components, stats, _ = cv2.connectedComponentsWithStats((labels == 2).astype(np.uint8), connectivity=4)
    area = float(np.prod(spacing) * 1e6)
    small = stats[:, cv2.CC_STAT_AREA] * area < settings.minimum_area_mm2
    small[0] = False
    withdrawn = small[components]
    result = labels.astype(np.uint8, copy=True)
    result[withdrawn] = 1
    return result, dict(format=VERSION, minimum_area_mm2=settings.minimum_area_mm2,
        connectivity=4, pixel_area_mm2=area, withdrawn_gas_components=int(small.sum()),
        reclassified_gas_pixels=int(withdrawn.sum()), co2_support_unchanged=True)


class FreeGasSequence:
    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        self.inherited.bind_reference(baseline)

    def apply_with_reference(self, predictions, elapsed_h, current):
        result, report = self.inherited.apply_with_reference(predictions, elapsed_h, current)
        masks = two_phase_masks(result)
        labels = np.where(masks['gas'], 2, masks['aqueous']).astype(np.uint8)
        spacing = (abs(float(self.context.y_m[1]-self.context.y_m[0])),
                   abs(float(self.context.x_m[1]-self.context.x_m[0])))
        cleaned, audit = clean_labels(labels, spacing, self.settings)
        changed = (labels == 2) & (cleaned == 1)
        updated = dict(result)
        # These are display classification scores, not calibrated concentrations.
        gas = np.where(changed, 0., result['gas_probability_conditional_on_co2']).astype(np.float32)
        updated['gas_probability_conditional_on_co2'] = gas
        co2 = result['co2_presence_probability']
        for key, values in (('aqueous_co2_probability', co2*(1-gas)),
                            ('free_co2_gas_probability', co2*gas)):
            if key in result:
                updated[key] = np.where(changed, values, result[key]).astype(np.float32)
        self.last_diagnostics = {**self.inherited.last_diagnostics, 'small_free_gas_reclassified': changed}
        return updated, {**report, 'free_gas_cleanup': audit}
