"""Phase rules: phase maps of a run from its phase probabilities.

A phase map gives each pixel of a frame one category: water, dissolved CO2, free
gas or unknown. The labels of a frame follow from its CO2-presence and
gas-within-CO2 probabilities, the CO2 threshold of the settings and the gas
threshold of the phase model; CO2 is only where the phase model found it.

Areas are physical. Every setting is a field of :class:`PhaseRuleSettings`.

"""

from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np


class PhaseLabel(IntEnum):
    """Categories of a phase map (uint8)."""

    WATER = 0
    DISSOLVED = 1
    GAS = 2
    UNKNOWN = 255


@dataclass(frozen=True)
class PhaseRuleSettings:
    """Settings of the phase labels and rules; one configuration table."""

    co2_threshold: float
    """Smallest CO2-presence probability of CO2."""
    smallest_gas_patch_mm2: float
    """Free gas in 4-connected patches below this area is dissolved CO2."""


def _summed_areas(components: np.ndarray, count: int, pixel_area: float | np.ndarray):
    """Area of each component (index 0: background): its pixels' areas summed."""
    if np.ndim(pixel_area) == 0:
        return np.bincount(components.ravel(), minlength=count) * pixel_area
    return np.bincount(components.ravel(), weights=pixel_area.ravel(), minlength=count)


def _areas(mask: np.ndarray, connectivity: int, pixel_area: float | np.ndarray):
    """Components of a mask, their count and their areas (index 0: background)."""
    count, components = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=connectivity
    )
    return components, _summed_areas(components, count, pixel_area)


def phase_labels(
    co2: np.ndarray,
    gas: np.ndarray,
    domain: np.ndarray,
    pixel_area_mm2: float | np.ndarray,
    settings: PhaseRuleSettings,
    *,
    gas_threshold: float,
) -> np.ndarray:
    """Phase labels of one frame from its two phase probabilities.

    A domain pixel is CO2 where its CO2-presence probability reaches the CO2
    threshold, and free gas where its gas-within-CO2 probability also reaches
    the gas threshold; elsewhere it is water, including pixels without
    probabilities (NaN). Free gas in 4-connected patches smaller than
    ``smallest_gas_patch_mm2`` is dissolved CO2. Pixels outside the domain are
    unknown.

    Parameters
    ----------
    co2, gas : array
        CO2-presence and gas-within-CO2 probabilities of shape (rows, cols).
    domain : array
        Pixels whose phase is decided.
    pixel_area_mm2 : float or array
        Area of each pixel: one value for a uniform grid, or an array of shape
        (rows, cols), e.g. measured pixel areas. The area of a patch is the sum
        of its pixels' areas.
    settings : PhaseRuleSettings
        The CO2 threshold and the smallest gas patch.
    gas_threshold : float
        Gas threshold of the phase model that gave the probabilities
        (``PhaseModel.gas_threshold``).

    Returns
    -------
    array
        Labels (uint8) of :class:`PhaseLabel`.

    """
    present = domain & (co2 >= settings.co2_threshold)
    labels = np.where(domain, PhaseLabel.WATER, PhaseLabel.UNKNOWN).astype(np.uint8)
    labels[present] = PhaseLabel.DISSOLVED
    patches, area = _areas(present & (gas >= gas_threshold), 4, pixel_area_mm2)
    kept = area >= settings.smallest_gas_patch_mm2
    kept[0] = False
    labels[kept[patches]] = PhaseLabel.GAS
    return labels
