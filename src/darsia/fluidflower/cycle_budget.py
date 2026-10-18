"""Mass budget per injection cycle, series agnostic.

For every injection cycle (one injection and the waiting time until the next
one, or until the end of imaging) the budget closes

    injected = new dissolved (in the quantified sand) + free gas in the rig
               + left through the overflow + rest,

where

* ``injected`` comes from the injection protocol (MFC rate, duration and a
  gas density; the density's reference conditions are a declared input),
* ``new dissolved`` is the image product of the cycle relative to the cycle
  baseline, as a lower and an upper bound (saturated pixels, see
  ``concentration_flags``),
* ``free gas`` comes from an external scale record where one exists: the
  water displaced from the rig equals the free-gas volume, which times the gas
  density gives the mass. Runs without a scale (FF_UM, FF_RE e0) get NaN and
  the rest absorbs it,
* ``left`` is dissolved CO2 that left with the overflow water (two-way
  overflow runs only; a declared estimate),
* ``rest`` is what the images and the scale cannot see: free water, the open
  top, outside the frame, and measurement error. It is reported, never hidden.

Everything is in grams. The functions take plain numbers and tables so that
they can be used from the pipeline, from a notebook and from tests alike.
Defaults are proposals awaiting Olav's decision (Claude, 18 Oct 2026).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

__all__ = [
    "GAS_DENSITY_KG_PER_M3",
    "injected_mass_g",
    "free_gas_mass_g",
    "CycleBudgetInputs",
    "CycleBudget",
    "cycle_budget",
    "budget_rows",
]

GAS_DENSITY_KG_PER_M3 = {
    # CO2 at 1 atm; the MFC reference conditions decide which one applies
    "0C": 1.977,
    "15C": 1.872,
    "20C": 1.842,
    "25C": 1.811,
}


def injected_mass_g(rate_ml_per_min: float, duration_min: float, gas_density_kg_per_m3: float) -> float:
    """Injected mass from the MFC setpoint: mL/min x min x kg/m3 -> g."""
    volume_ml = rate_ml_per_min * duration_min
    return volume_ml * 1e-6 * gas_density_kg_per_m3 * 1e3


def free_gas_mass_g(displaced_water_g: float, gas_density_kg_per_m3: float,
                    water_density_kg_per_m3: float = 997.0) -> float:
    """Free gas in the rig from the water it displaced onto the scale.
    NaN in gives NaN out (no scale record)."""
    displaced_water_g = _as_term(displaced_water_g)
    if math.isnan(displaced_water_g):
        return float("nan")
    if not math.isfinite(displaced_water_g) or not (0.0 < water_density_kg_per_m3 < math.inf) \
            or not (0.0 < gas_density_kg_per_m3 < math.inf):
        raise ValueError("displaced water and densities must be finite and the densities positive")
    volume_m3 = displaced_water_g * 1e-3 / water_density_kg_per_m3
    return volume_m3 * gas_density_kg_per_m3 * 1e3


def _as_term(value) -> float:
    """Input contract of the budget: a term is a float; ``None`` means unknown
    and becomes NaN here, so callers may pass either."""
    if value is None:
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class CycleBudgetInputs:
    cycle_run: str
    injected_g: float
    dissolved_new_lower_g: float
    dissolved_new_upper_g: float
    free_gas_g: float = float("nan")
    left_overflow_g: float = float("nan")
    background_dissolved_g: float = float("nan")
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("injected_g", "dissolved_new_lower_g", "dissolved_new_upper_g"):
            v = _as_term(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, v)
        for name in ("free_gas_g", "left_overflow_g", "background_dissolved_g"):
            v = _as_term(getattr(self, name))
            if math.isinf(v):
                raise ValueError(f"{name} must be finite or unknown (NaN / None), not infinite")
            object.__setattr__(self, name, v)
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))


@dataclass(frozen=True)
class CycleBudget:
    cycle_run: str
    injected_g: float
    dissolved_new_lower_g: float
    dissolved_new_upper_g: float
    free_gas_g: float
    left_overflow_g: float
    background_dissolved_g: float
    rest_lower_g: float
    rest_upper_g: float
    dissolved_fraction_lower: float
    dissolved_fraction_upper: float
    free_gas_known: bool
    overflow_known: bool
    quality_flags: tuple[str, ...]

    def as_row(self) -> dict:
        row = asdict(self)
        row["quality_flags"] = " | ".join(self.quality_flags)
        return row


def _nan_to_zero(x: float) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def cycle_budget(inputs: CycleBudgetInputs) -> CycleBudget:
    """Close the budget. Unknown terms (NaN) count as zero in the rest, and the
    flags ``free_gas_known`` / ``overflow_known`` say so."""
    if inputs.injected_g < 0:
        raise ValueError("injected mass must be non-negative")
    if inputs.dissolved_new_upper_g < inputs.dissolved_new_lower_g:
        raise ValueError("upper bound of the dissolved mass below the lower bound")
    free_gas_known = not math.isnan(inputs.free_gas_g)
    overflow_known = not math.isnan(inputs.left_overflow_g)
    gas = _nan_to_zero(inputs.free_gas_g)
    left = _nan_to_zero(inputs.left_overflow_g)
    rest_upper = inputs.injected_g - inputs.dissolved_new_lower_g - gas - left
    rest_lower = inputs.injected_g - inputs.dissolved_new_upper_g - gas - left
    inj = inputs.injected_g
    return CycleBudget(
        cycle_run=inputs.cycle_run,
        injected_g=inj,
        dissolved_new_lower_g=inputs.dissolved_new_lower_g,
        dissolved_new_upper_g=inputs.dissolved_new_upper_g,
        free_gas_g=inputs.free_gas_g,
        left_overflow_g=inputs.left_overflow_g,
        background_dissolved_g=inputs.background_dissolved_g,
        rest_lower_g=rest_lower,
        rest_upper_g=rest_upper,
        dissolved_fraction_lower=inputs.dissolved_new_lower_g / inj if inj > 0 else float("nan"),
        dissolved_fraction_upper=inputs.dissolved_new_upper_g / inj if inj > 0 else float("nan"),
        free_gas_known=free_gas_known,
        overflow_known=overflow_known,
        quality_flags=tuple(inputs.quality_flags),
    )


def budget_rows(inputs: list[CycleBudgetInputs]) -> list[dict]:
    """One table row per cycle, in the order given."""
    return [cycle_budget(x).as_row() for x in inputs]
