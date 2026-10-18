"""Per-pixel flags for dissolved-CO2 products: saturated and not quantified.

Two honest states that a concentration map must carry instead of a number:

* **saturated** – the indicator has used (almost) its whole colour range, so
  the colour no longer measures the dissolved CO2. The dissolved mass is then
  an interval: at least the value at the saturation threshold, at most the
  solubility at the lab temperature.
* **not quantified** – the pixel has no colour path (free water, excluded
  areas, outside the rig frame) or holds free gas, so the colour does not
  measure dissolved CO2 at all. Reported as a mask, never as zero.

The functions are pure numpy and series agnostic. The defaults (threshold
``f_min = 0.05``) are proposals awaiting Olav's decision (Claude, 18 Oct 2026).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from darsia.fluidflower.titration import TitrationModel

__all__ = [
    "DEFAULT_SATURATION_BLUE_FRACTION",
    "DissolvedInterval",
    "saturated_mask",
    "dissolved_interval",
    "not_quantified_mask",
    "flag_fractions",
]

DEFAULT_SATURATION_BLUE_FRACTION = 0.05
"""Blue fraction below which a pixel counts as saturated (95 % of the colour
range used; about 5 mM DIC for the FF_RE recipe)."""


def saturated_mask(blue_fraction, f_min: float = DEFAULT_SATURATION_BLUE_FRACTION):
    """True where the indicator is saturated (blue fraction below f_min).
    NaN blue fractions are never saturated (they are not quantified)."""
    f = np.asarray(blue_fraction, dtype=float)
    with np.errstate(invalid="ignore"):
        return (f < f_min) & np.isfinite(f)


@dataclass(frozen=True)
class DissolvedInterval:
    """Lower and upper bound of the dissolved CO2 (mol/L) per pixel, with the
    saturation mask that explains where the two differ."""

    lower: np.ndarray
    upper: np.ndarray
    saturated: np.ndarray


def dissolved_interval(blue_fraction, model: TitrationModel,
                       f_min: float = DEFAULT_SATURATION_BLUE_FRACTION,
                       p_co2_atm: float = 1.0) -> DissolvedInterval:
    """Interval of the dissolved CO2 per pixel.

    Below the threshold the colour is only a lower bound: ``lower`` is the DIC
    at ``f_min`` and ``upper`` the solubility at ``p_co2_atm``. Elsewhere both
    bounds equal the closed-form DIC of the blue fraction. NaN stays NaN."""
    f = np.asarray(blue_fraction, dtype=float)
    sat = saturated_mask(f, f_min)
    f_eff = np.where(sat, f_min, f)
    with np.errstate(invalid="ignore"):
        value = np.where(np.isfinite(f_eff), model.dic_from_blue_fraction(np.where(np.isfinite(f_eff), f_eff, 0.5)), np.nan)
    upper = np.where(sat, model.saturation_dic(p_co2_atm), value)
    return DissolvedInterval(lower=value, upper=upper, saturated=sat)


def not_quantified_mask(labels, quantified_labels, gas_mask=None, outside_mask=None):
    """True where the colour does not measure dissolved CO2: labels without a
    colour path (anything not in ``quantified_labels``), free gas, and pixels
    outside the rig frame."""
    labels = np.asarray(labels)
    quantified = np.isin(labels, np.asarray(list(quantified_labels)))
    mask = ~quantified
    if gas_mask is not None:
        mask |= np.asarray(gas_mask, dtype=bool)
    if outside_mask is not None:
        mask |= np.asarray(outside_mask, dtype=bool)
    return mask


def flag_fractions(saturated, not_quantified, sand_mask) -> dict:
    """Area fractions for the result table: saturated and not-quantified shares
    of the sand area (the area that is meant to be quantified)."""
    sand = np.asarray(sand_mask, dtype=bool)
    n = int(sand.sum())
    if n == 0:
        return {"saturated_fraction": float("nan"), "not_quantified_fraction": float("nan"), "sand_pixels": 0}
    sat = np.asarray(saturated, dtype=bool) & sand
    nq = np.asarray(not_quantified, dtype=bool) & sand
    return {
        "saturated_fraction": float(sat.sum() / n),
        "not_quantified_fraction": float(nq.sum() / n),
        "sand_pixels": n,
    }
