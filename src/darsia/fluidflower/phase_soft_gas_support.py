"""Analysis-only, smooth gas plausibility prior for registered phase images.

This is a hypothesis to compare with reviewed images, not a 3-D gas-transport
solution or a production phase rule. It changes only the *conditional* free-gas
score; CO2 presence and visible-2D review labels remain separate. The end-of-
injection and previous-frame masks are observations, never accumulated output
from this prior. No Studio/runtime code imports this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class SoftGasSettings:
    post_injection_grace_h: float = 0.25
    end_snapshot_max_delay_h: float = 0.5
    below_source_tolerance_mm: float = 5.0
    below_source_transition_mm: float = 7.0
    end_footprint_tolerance_mm: float = 8.0
    end_footprint_transition_mm: float = 12.0
    facies_edge_exemption_mm: float = 5.0
    facies_edge_transition_mm: float = 8.0
    recent_gas_radius_mm: float = 8.0
    history_max_gap_h: float = 4.0
    maximum_odds_divisor: float = 6.0
    independent_optical_protect_start: float = 0.6
    independent_optical_protect_full: float = 0.95

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("Soft gas settings must be finite and positive")
        if self.maximum_odds_divisor < 1:
            raise ValueError("Odds divisor cannot increase unsupported gas")
        if not (0 < self.independent_optical_protect_start <
                self.independent_optical_protect_full <= 1):
            raise ValueError("Invalid independent optical protection interval")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def _registered_grid(x_m: np.ndarray, y_m: np.ndarray, facies: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2 or
        facies.shape != (len(y), len(x)) or not np.isfinite(x).all() or
        not np.isfinite(y).all()):
        raise ValueError("Expected one finite registered image grid")
    dx, dy = np.diff(x), np.diff(y)
    for delta in (dx, dy):
        if (np.any(delta == 0) or np.any(np.sign(delta) != np.sign(delta[0])) or
            not np.allclose(delta, delta[0], rtol=1e-3, atol=1e-9)):
            raise ValueError("Image coordinates must be regular and monotonic")
    return abs(float(dy[0])) * 1000., abs(float(dx[0])) * 1000.


def _distance_to(mask: np.ndarray, spacing_mm: tuple[float, float]) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return distance_transform_edt(~mask, sampling=spacing_mm).astype(np.float32)


def prepare_soft_gas_geometry(*, x_m: np.ndarray, y_m: np.ndarray,
                              facies: np.ndarray, stages: list[dict],
                              end_snapshots: list[np.ndarray | None],
                              end_snapshot_times_h: list[float | None],
                              settings: SoftGasSettings = SoftGasSettings()) -> dict:
    """Precompute physical distances; never assign a pixel to one injector.

    Missing or late snapshots disable the prior rather than inventing absent
    gas. The caller provides the corresponding photographed elapsed times.
    """
    facies = np.asarray(facies)
    spacing_mm = _registered_grid(x_m, y_m, facies)
    if (not stages or len(stages) != len(end_snapshots) or
        len(stages) != len(end_snapshot_times_h)):
        raise ValueError("One timed end snapshot per injection is required")
    for stage in stages:
        if not all(np.isfinite(float(stage[key])) for key in ("y_m", "end_h")):
            raise ValueError("Invalid injection geometry or time")
    if any(snapshot is None or timestamp is None or
           not np.isfinite(timestamp) or
           not (float(stage["end_h"]) <= timestamp <=
                float(stage["end_h"]) + settings.end_snapshot_max_delay_h)
           for stage, snapshot, timestamp in zip(stages, end_snapshots,
                                                  end_snapshot_times_h)):
        return {"shape": facies.shape, "missing_end_snapshot": True,
                "snapshot_times_h": tuple(end_snapshot_times_h),
                "end_snapshot_max_delay_h": settings.end_snapshot_max_delay_h}
    masks = [np.asarray(snapshot, bool) for snapshot in end_snapshots]
    if any(mask.shape != facies.shape for mask in masks) or not any(mask.any() for mask in masks):
        raise ValueError("End snapshots must have registered gas support")
    edge = np.zeros(facies.shape, bool)
    edge[1:, :] |= facies[1:, :] != facies[:-1, :]
    edge[:, 1:] |= facies[:, 1:] != facies[:, :-1]
    end_union = np.logical_or.reduce(masks)
    return {"shape": facies.shape, "missing_end_snapshot": False,
            "snapshot_times_h": tuple(end_snapshot_times_h),
            "end_snapshot_max_delay_h": settings.end_snapshot_max_delay_h,
            "spacing_mm": spacing_mm,
            "end_distance_mm": _distance_to(end_union, spacing_mm),
            "facies_edge_distance_mm": _distance_to(edge, spacing_mm)}


def soft_gas_risk(*, elapsed_h: float, x_m: np.ndarray, y_m: np.ndarray,
                  facies: np.ndarray, stages: list[dict],
                  end_snapshots: list[np.ndarray | None],
                  end_snapshot_times_h: list[float | None],
                  previous_observed_gas: np.ndarray | None = None,
                  previous_h: float | None = None,
                  reviewed_gas_positive: np.ndarray | None = None,
                  settings: SoftGasSettings = SoftGasSettings(),
                  prepared: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return a smooth [0,1] risk that a gas prediction lacks physical support.

    Recent gas can *reduce* risk if its time gap is reliable. A long gap adds
    no negative temporal evidence. Spatial risk still uses the observed
    injection-end footprint. Position is in metres, distances in millimetres.
    """
    facies = np.asarray(facies)
    spacing_mm = _registered_grid(x_m, y_m, facies)
    if not np.isfinite(elapsed_h) or not stages:
        raise ValueError("Finite time and at least one injection are required")
    geom = prepared if prepared is not None else prepare_soft_gas_geometry(
        x_m=x_m, y_m=y_m, facies=facies, stages=stages,
        end_snapshots=end_snapshots, end_snapshot_times_h=end_snapshot_times_h,
        settings=settings)
    if (geom["shape"] != facies.shape or
        geom["snapshot_times_h"] != tuple(end_snapshot_times_h) or
        geom["end_snapshot_max_delay_h"] != settings.end_snapshot_max_delay_h or
        (not geom["missing_end_snapshot"] and
         not np.allclose(geom["spacing_mm"], spacing_mm))):
        raise ValueError("Prepared gas geometry differs from the current grid")
    zero = np.zeros(facies.shape, np.float32)
    if geom["missing_end_snapshot"]:
        return zero, {"skipped": "missing_injection_end_evidence"}
    if elapsed_h < max(float(value) for value in end_snapshot_times_h):
        return zero, {"skipped": "end_snapshot_not_yet_observed"}
    if elapsed_h < max(float(stage["end_h"]) for stage in stages) + settings.post_injection_grace_h:
        return zero, {"skipped": "active_injection_or_grace"}

    # Only positions below *all* injection heights are exposed to this
    # conservative gravity argument. No x-coordinate or injector partition is
    # used. Smooth transitions avoid a categorical height cutoff.
    lower_than_all_mm = min(float(stage["y_m"]) for stage in stages) * 1000. - np.asarray(y_m)[:, None] * 1000.
    below = _smoothstep((lower_than_all_mm - settings.below_source_tolerance_mm)
                        / settings.below_source_transition_mm)
    far_end = _smoothstep((geom["end_distance_mm"] - settings.end_footprint_tolerance_mm)
                          / settings.end_footprint_transition_mm)
    far_edge = _smoothstep((geom["facies_edge_distance_mm"] - settings.facies_edge_exemption_mm)
                           / settings.facies_edge_transition_mm)
    risk = (below * far_end * far_edge).astype(np.float32)

    temporal = "none"
    if previous_observed_gas is not None:
        if previous_h is None or not np.isfinite(previous_h) or previous_h >= elapsed_h:
            raise ValueError("A previous gas observation needs an earlier finite time")
        previous = np.asarray(previous_observed_gas, bool)
        if previous.shape != facies.shape:
            raise ValueError("Previous gas observation is not registered")
        gap_h = elapsed_h - previous_h
        if gap_h <= settings.history_max_gap_h and previous.any():
            recent_distance = _distance_to(previous, spacing_mm)
            recent_support = np.exp(-recent_distance / settings.recent_gas_radius_mm)
            reliability = 1. - gap_h / settings.history_max_gap_h
            risk *= (1. - reliability * recent_support).astype(np.float32)
            temporal = "recent_observation_reduces_risk"
        else:
            temporal = "long_gap_or_no_recent_gas_is_neutral"
    if reviewed_gas_positive is not None:
        positive = np.asarray(reviewed_gas_positive, bool)
        if positive.shape != facies.shape:
            raise ValueError("Reviewed gas mask is not registered")
        risk[positive] = 0.
    return risk, {"skipped": None, "temporal": temporal,
                  "nonzero_risk_pixels": int(np.count_nonzero(risk)),
                  "mean_risk": float(risk.mean()), "maximum_risk": float(risk.max())}


def downweight_gas_probability(gas_conditional_on_co2: np.ndarray,
                               independent_optical_support: np.ndarray,
                               risk: np.ndarray,
                               settings: SoftGasSettings = SoftGasSettings()) -> tuple[np.ndarray, dict]:
    """Apply a bounded odds penalty, preserving independently strong optics.

    ``independent_optical_support`` must be a separately assessed image signal,
    e.g. registered baseline-relative color change. Reusing this same model's
    gas probability as support would make a confident false positive protect
    itself, as can happen in late AC51. This function is for analysis only.
    """
    gas = np.asarray(gas_conditional_on_co2, np.float32)
    optical = np.asarray(independent_optical_support, np.float32)
    risk = np.asarray(risk, np.float32)
    if (gas.ndim != 2 or optical.shape != gas.shape or risk.shape != gas.shape or
        not np.isfinite(gas).all() or not np.isfinite(optical).all() or
        not np.isfinite(risk).all() or min(float(gas.min()), float(optical.min()), float(risk.min())) < 0 or
        max(float(gas.max()), float(optical.max()), float(risk.max())) > 1):
        raise ValueError("Expected aligned finite [0,1] probability maps")
    protected = _smoothstep((optical - settings.independent_optical_protect_start)
                            / (settings.independent_optical_protect_full -
                               settings.independent_optical_protect_start))
    effective_risk = risk * (1. - protected)
    divisor = 1. + (settings.maximum_odds_divisor - 1.) * effective_risk
    # p/(p + (1-p)*divisor) is the exact odds update, also defined at 0 and 1.
    adjusted = gas / (gas + (1. - gas) * divisor)
    audit = {"softened_pixels": int(np.count_nonzero(adjusted < gas - 1e-6)),
             "crossed_half_probability": int(np.count_nonzero((gas >= .5) & (adjusted < .5))),
             "mean_probability_change": float(np.mean(gas - adjusted)),
             "maximum_probability_change": float(np.max(gas - adjusted))}
    return adjusted.astype(np.float32), audit
