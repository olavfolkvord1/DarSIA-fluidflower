"""Derive and apply an overflow-scale constraint on free CO2 mass.

The scale records cumulative liquid outflow. For an approximately fixed cell
volume, the change in free-gas volume is the cumulative liquid outflow minus
the cumulative liquid feed. The derived curve is deliberately frozen before
Optuna starts; the optimizer cannot tune the feed rate to improve image fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_NAME = "ffac-overflow-phase-balance"
MODEL_VERSION = "1.3.0"
REFERENCE_MODEL_NAME = "ffac-overflow-series-reference"
REFERENCE_MODEL_VERSION = "1.1.0"
HYBRID_MODEL_NAME = "ffac-overflow-hybrid-phase-balance"
HYBRID_MODEL_VERSION = "1.1.0"
CONFIRMED_STATUSES = {"Verified", "Corrected"}
DEFAULT_MATCH_TOLERANCE_H = 0.03
DEFAULT_LIQUID_DENSITY_KG_M3 = 1000.0
DEFAULT_FEED_RATE_SIGMA_G_H = 2.0
DEFAULT_SCALE_SIGMA_G = 1.0
DEFAULT_MODEL_VOLUME_SIGMA_ML = 5.0
MAX_ZERO_GAS_ANCHOR_GAP_S = 900.0
DEFAULT_TRACE_RATE_CORRECTION = 1.0125
DEFAULT_TRACE_RATE_CORRECTION_SOURCE = (
    "calibrated against AC20/AC24/AC27/AC60 independently anchored feed rates"
)
DEFAULT_CONSTRAINT_MODE = "two-sided-interval"
VALID_CONSTRAINT_MODES = {DEFAULT_CONSTRAINT_MODE, "gas-upper-bound"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OverflowPhaseTarget:
    time_h: float
    injected_mass_kg: float
    gas_mass_kg: float
    aqueous_mass_kg: float
    phase_mass_sigma_kg: float
    cumulative_outflow_g: float
    inferred_gas_volume_ml: float
    target_source: str = ""


@dataclass(frozen=True)
class OverflowPhaseConstraint:
    run: str
    path: Path
    sha256: str
    targets: tuple[OverflowPhaseTarget, ...]
    match_tolerance_h: float
    provenance: Mapping[str, Any]
    constraint_mode: str = DEFAULT_CONSTRAINT_MODE

    def target_at(self, time_h: float) -> OverflowPhaseTarget | None:
        if not self.targets:
            return None
        nearest = min(self.targets, key=lambda target: abs(target.time_h - time_h))
        if abs(nearest.time_h - float(time_h)) > self.match_tolerance_h:
            return None
        return nearest

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "path": str(self.path),
            "sha256": self.sha256,
            "match_tolerance_h": self.match_tolerance_h,
            "target_count": len(self.targets),
            "target_times_h": [target.time_h for target in self.targets],
            "constraint_mode": self.constraint_mode,
            "provenance": dict(self.provenance),
        }


def normalize_constraint_mode(value: Any) -> str:
    mode = str(value or DEFAULT_CONSTRAINT_MODE).strip().lower().replace("_", "-")
    aliases = {
        "interval": DEFAULT_CONSTRAINT_MODE,
        "two-sided": DEFAULT_CONSTRAINT_MODE,
        "upper": "gas-upper-bound",
        "upper-bound": "gas-upper-bound",
        "gas-upper": "gas-upper-bound",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_CONSTRAINT_MODES:
        raise ValueError(
            f"Unsupported overflow constraint mode {value!r}; expected one of "
            f"{sorted(VALID_CONSTRAINT_MODES)}"
        )
    return mode


def load_overflow_phase_constraint(
    path: str | Path,
    *,
    run: str,
) -> OverflowPhaseConstraint:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if int(payload.get("schema", 0)) != 1:
        raise ValueError(f"Unsupported overflow phase-target schema in {source}")
    target_run = str(payload.get("run", "")).strip().lower()
    if target_run != str(run).strip().lower():
        raise ValueError(
            f"Overflow phase targets are for {target_run!r}, not {run!r}: {source}"
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(f"Overflow phase target file has no targets: {source}")
    targets = tuple(
        OverflowPhaseTarget(
            time_h=float(row["time_h"]),
            injected_mass_kg=float(row["injected_mass_kg"]),
            gas_mass_kg=float(row["gas_mass_kg"]),
            aqueous_mass_kg=float(row["aqueous_mass_kg"]),
            phase_mass_sigma_kg=float(row["phase_mass_sigma_kg"]),
            cumulative_outflow_g=float(row["cumulative_outflow_g"]),
            inferred_gas_volume_ml=float(row["inferred_gas_volume_ml"]),
            target_source=str(row.get("target_source", "")),
        )
        for row in raw_targets
    )
    times = [target.time_h for target in targets]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError(f"Overflow phase target times must be unique and sorted: {source}")
    for target in targets:
        values = (
            target.time_h,
            target.injected_mass_kg,
            target.gas_mass_kg,
            target.aqueous_mass_kg,
            target.phase_mass_sigma_kg,
            target.cumulative_outflow_g,
            target.inferred_gas_volume_ml,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Non-finite overflow phase target in {source}: {target}")
        if target.time_h < 0.0 or target.injected_mass_kg < 0.0:
            raise ValueError(f"Invalid overflow phase target in {source}: {target}")
        if not 0.0 <= target.gas_mass_kg <= target.injected_mass_kg + 1e-12:
            raise ValueError(f"Gas target is outside injected mass in {source}: {target}")
        if target.phase_mass_sigma_kg < 0.0:
            raise ValueError(f"Negative target uncertainty in {source}: {target}")
    tolerance = float(
        payload.get("match_tolerance_h", DEFAULT_MATCH_TOLERANCE_H)
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"Invalid match tolerance in {source}: {tolerance}")
    constraint_mode = normalize_constraint_mode(payload.get("constraint_mode"))
    return OverflowPhaseConstraint(
        run=target_run,
        path=source.resolve(),
        sha256=_sha256(source),
        targets=targets,
        match_tolerance_h=tolerance,
        constraint_mode=constraint_mode,
        provenance={
            "model": payload.get("model"),
            "model_version": payload.get("model_version"),
            "source": payload.get("source"),
            "physics": payload.get("physics"),
            "derivation": payload.get("derivation"),
        },
    )


def resolve_overflow_phase_constraint(
    *,
    run: str,
    path: str | Path | None = None,
    penalty_weight: float | str | None = None,
) -> tuple[OverflowPhaseConstraint | None, float]:
    raw_path = (
        str(path)
        if path is not None
        else os.environ.get("FFAC_OVERFLOW_PHASE_TARGETS", "")
    ).strip()
    raw_weight = (
        penalty_weight
        if penalty_weight is not None
        else os.environ.get("FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT", "0")
    )
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Overflow phase penalty weight must be numeric, got {raw_weight!r}"
        ) from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("Overflow phase penalty weight must be finite and non-negative")
    if raw_path.lower() in {"", "off", "none"}:
        if weight > 0.0:
            raise ValueError(
                "FFAC_OVERFLOW_PHASE_TARGETS is required when the overflow "
                "phase penalty is active"
            )
        return None, 0.0
    constraint = load_overflow_phase_constraint(raw_path, run=run)
    return constraint, weight


def _parse_protocol_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def _load_injection_protocol(path: Path) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            stages.append(
                {
                    "start": _parse_protocol_time(row["start"]),
                    "end": _parse_protocol_time(row["end"]),
                    "rate_sccm": float(row["rate_sccm"]),
                    "density_kg_m3": float(row["density kg/m3"]),
                }
            )
    if not stages:
        raise ValueError(f"Injection protocol is empty: {path}")
    return stages


def _injected_mass_kg(stages: Sequence[Mapping[str, Any]], at: datetime) -> float:
    mass = 0.0
    for stage in stages:
        start = stage["start"]
        end = stage["end"]
        if at <= start:
            continue
        duration_minutes = max(
            0.0,
            (min(at, end) - start).total_seconds() / 60.0,
        )
        volume_m3 = float(stage["rate_sccm"]) * duration_minutes * 1e-6
        mass += volume_m3 * float(stage["density_kg_m3"])
    return mass


def _integrated_piecewise_feed_g(
    time_h: float,
    segments: Sequence[Mapping[str, Any]],
    *,
    rate_key: str,
) -> float:
    """Integrate a feed schedule relative to injection start.

    The first segment is assumed to be active immediately before injection as
    well. Its pre-injection contribution is negative and is used only while
    estimating the scale reading at t=0.
    """

    value = float(time_h)
    first_rate = float(segments[0][rate_key])
    if value < 0.0:
        return first_rate * value
    total = 0.0
    for index, segment in enumerate(segments):
        start_h = float(segment["start_h"])
        end_h = (
            float(segments[index + 1]["start_h"])
            if index + 1 < len(segments)
            else value
        )
        duration_h = max(0.0, min(value, end_h) - start_h)
        total += float(segment[rate_key]) * duration_h
        if value <= end_h:
            break
    return total


def _pairwise_window_slopes(
    times_h: np.ndarray,
    weights_g: np.ndarray,
    *,
    start_h: float,
    end_h: float,
    min_span_h: float = 0.60,
    max_span_h: float = 1.40,
) -> np.ndarray:
    slopes: list[float] = []
    for left in range(times_h.size):
        for right in range(left + 1, times_h.size):
            span_h = float(times_h[right] - times_h[left])
            midpoint_h = float((times_h[right] + times_h[left]) / 2.0)
            if (
                min_span_h <= span_h <= max_span_h
                and start_h <= midpoint_h <= end_h
            ):
                slopes.append(float((weights_g[right] - weights_g[left]) / span_h))
    return np.asarray(slopes, dtype=float)


def _derive_trace_feed_schedule(
    *,
    times_h: np.ndarray,
    weights_g: np.ndarray,
    injection_end_h: float,
    documented_feed_segments: Sequence[Mapping[str, Any]],
    trace_rate_correction: float,
    trace_rate_correction_source: str,
    effective_feed_rate_overrides_g_h: Sequence[float] | None = None,
    effective_feed_rate_override_source: str | None = None,
) -> tuple[list[dict[str, Any]], float, float, int, dict[str, Any]]:
    """Estimate effective feed rates while retaining documented changepoints.

    After injection, shrinking free-gas volume makes the outflow slope approach
    the liquid-feed rate from below. A high robust slope quantile is therefore
    used and corrected by the independently anchored control experiments. The
    documented nominal rate is an upper physical guard, not the fitted value.
    """

    if not math.isfinite(trace_rate_correction) or trace_rate_correction <= 0.0:
        raise ValueError("Trace feed-rate correction must be finite and positive")
    raw_segments = [dict(segment) for segment in documented_feed_segments]
    if not raw_segments:
        raise ValueError("At least one documented feed segment is required")
    raw_segments.sort(key=lambda segment: float(segment["start_h"]))
    if not math.isclose(float(raw_segments[0]["start_h"]), 0.0, abs_tol=1e-9):
        raise ValueError("The first documented feed segment must start at t=0")
    starts = [float(segment["start_h"]) for segment in raw_segments]
    if any(right <= left for left, right in zip(starts, starts[1:])):
        raise ValueError("Documented feed-segment starts must be strictly increasing")
    rate_overrides = (
        tuple(float(value) for value in effective_feed_rate_overrides_g_h)
        if effective_feed_rate_overrides_g_h is not None
        else None
    )
    if rate_overrides is not None and len(rate_overrides) != len(raw_segments):
        raise ValueError("Effective feed-rate overrides must match feed segments")

    derived: list[dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        start_h = float(segment["start_h"])
        nominal_rate = float(segment["nominal_rate_g_per_h"])
        if not math.isfinite(nominal_rate) or nominal_rate <= 0.0:
            raise ValueError(f"Invalid nominal feed rate: {nominal_rate!r}")
        next_start_h = starts[index + 1] if index + 1 < len(starts) else None
        if next_start_h is None:
            fit_start_h = max(5.0, injection_end_h + 2.0, start_h + 0.55)
            fit_end_h = min(float(times_h[-1]), max(9.5, start_h + 5.0))
        else:
            fit_start_h = max(4.5, injection_end_h + 1.5, start_h + 0.55)
            fit_end_h = min(float(times_h[-1]), next_start_h - 0.40)
        if fit_end_h - fit_start_h < 0.8:
            # Partial traces such as AC50 still contain enough post-injection
            # data to estimate the initial feed segment, but not the preferred
            # late 5-9.5 h window.
            fit_start_h = max(injection_end_h + 0.50, start_h + 0.55)
            fit_end_h = float(times_h[-1])
        slopes = _pairwise_window_slopes(
            times_h,
            weights_g,
            start_h=fit_start_h,
            end_h=fit_end_h,
        )
        if slopes.size < 3:
            mask = (times_h >= fit_start_h) & (times_h <= fit_end_h)
            minimum_fit_points = 2 if rate_overrides is not None else 3
            if int(np.count_nonzero(mask)) < minimum_fit_points:
                raise ValueError(
                    "Too few validated readings to estimate feed segment "
                    f"starting at {start_h:.3f} h"
                )
            slope_center = float(np.polyfit(times_h[mask], weights_g[mask], 1)[0])
            slope_mad = 0.0
            slope_count = int(np.count_nonzero(mask))
            estimator = "least-squares slope fallback"
        else:
            slope_center = float(np.quantile(slopes, 0.90))
            slope_median = float(np.median(slopes))
            slope_mad = float(1.4826 * np.median(np.abs(slopes - slope_median)))
            slope_count = int(slopes.size)
            estimator = "90th percentile of 0.6-1.4 h validated-trace slopes"
        trace_effective_rate = min(
            nominal_rate * 1.02,
            slope_center * float(trace_rate_correction),
        )
        effective_rate = (
            float(rate_overrides[index])
            if rate_overrides is not None
            else trace_effective_rate
        )
        if not math.isfinite(effective_rate) or effective_rate <= 0.0:
            raise ValueError("Derived feed rate is not finite and positive")
        nominal_ratio = effective_rate / nominal_rate
        schedule_anomaly = not 0.85 <= nominal_ratio <= 1.05
        rate_sigma = max(3.0, 0.0125 * effective_rate, slope_mad)
        if rate_overrides is not None:
            rate_sigma = max(
                rate_sigma,
                abs(effective_rate - trace_effective_rate),
                0.025 * effective_rate,
            )
        if schedule_anomaly:
            rate_sigma = max(rate_sigma, 0.05 * effective_rate)
        derived.append(
            {
                "start_h": start_h,
                "end_h": next_start_h,
                "nominal_rate_g_per_h": nominal_rate,
                "effective_rate_g_per_h": effective_rate,
                "effective_rate_sigma_g_per_h": rate_sigma,
                "effective_to_nominal_ratio": nominal_ratio,
                "schedule_anomaly": schedule_anomaly,
                "fit_start_h": fit_start_h,
                "fit_end_h": fit_end_h,
                "slope_estimator": estimator,
                "raw_slope_g_per_h": slope_center,
                "raw_slope_mad_g_per_h": slope_mad,
                "slope_sample_count": slope_count,
                "trace_effective_rate_before_override_g_per_h": (
                    trace_effective_rate if rate_overrides is not None else None
                ),
                "effective_rate_override_source": (
                    str(effective_feed_rate_override_source)
                    if rate_overrides is not None
                    else None
                ),
            }
        )

    pre_mask = times_h < 0.0
    pre_count = int(np.count_nonzero(pre_mask))
    baseline_slope_g_per_h: float | None = None
    if pre_count >= 2:
        design = np.column_stack(
            (times_h[pre_mask], np.ones(np.count_nonzero(pre_mask)))
        )
        baseline_slope_g_per_h, baseline_weight_g = np.linalg.lstsq(
            design,
            weights_g[pre_mask],
            rcond=None,
        )[0]
        pre_residual = weights_g[pre_mask] - (
            baseline_slope_g_per_h * times_h[pre_mask] + baseline_weight_g
        )
        baseline_sigma_g = max(
            1.0,
            float(np.std(pre_residual, ddof=2))
            if pre_residual.size > 2
            else 1.0,
        )
        selected = baseline_candidates = weights_g[pre_mask]
        baseline_source = "linear fit to validated pre-injection readings"
    else:
        feed_at_rows = np.asarray(
            [
                _integrated_piecewise_feed_g(
                    float(time_h), derived, rate_key="effective_rate_g_per_h"
                )
                for time_h in times_h
            ],
            dtype=float,
        )
        baseline_candidates = weights_g - feed_at_rows
    if pre_count == 1:
        selected = baseline_candidates[pre_mask]
        baseline_source = (
            "single validated pre-injection reading extrapolated with initial "
            "effective feed rate"
        )
        baseline_weight_g = float(np.median(selected))
        baseline_sigma_g = max(
            5.0,
            abs(float(times_h[pre_mask][0]))
            * float(derived[0]["effective_rate_sigma_g_per_h"]),
        )
    elif pre_count == 0:
        lower_mask = (times_h >= max(5.0, injection_end_h + 2.0)) & (
            times_h <= min(9.5, float(times_h[-1]))
        )
        selected = baseline_candidates[lower_mask]
        if selected.size < 3:
            raise ValueError("No pre-injection or late lower-envelope baseline is available")
        baseline_weight_g = float(np.quantile(selected, 0.05))
        lower_tail = selected[selected <= np.quantile(selected, 0.25)]
        baseline_sigma_g = max(
            5.0,
            float(1.4826 * np.median(np.abs(lower_tail - np.median(lower_tail))))
            if lower_tail.size
            else 5.0,
        )
        baseline_source = (
            "5th-percentile post-injection lower envelope; zero-gas anchor assumed"
        )

    diagnostics = {
        "mode": "trace-derived-with-documented-segments",
        "trace_rate_correction": float(trace_rate_correction),
        "trace_rate_correction_source": str(trace_rate_correction_source),
        "effective_rate_override_source": (
            str(effective_feed_rate_override_source)
            if rate_overrides is not None
            else None
        ),
        "baseline_source": baseline_source,
        "baseline_candidate_count": int(selected.size),
        "baseline_slope_g_per_h": (
            float(baseline_slope_g_per_h)
            if baseline_slope_g_per_h is not None
            else None
        ),
        "schedule_anomaly": any(
            bool(segment["schedule_anomaly"]) for segment in derived
        ),
    }
    return derived, baseline_weight_g, baseline_sigma_g, pre_count, diagnostics


def build_overflow_phase_targets(
    *,
    review_state_path: str | Path,
    injection_protocol_path: str | Path,
    evaluation_times_h: Sequence[float],
    overrides_by_index: Mapping[int, float] | None = None,
    excluded_indices: Sequence[int] | None = None,
    liquid_density_kg_m3: float = DEFAULT_LIQUID_DENSITY_KG_M3,
    feed_rate_sigma_g_h: float = DEFAULT_FEED_RATE_SIGMA_G_H,
    scale_sigma_g: float = DEFAULT_SCALE_SIGMA_G,
    model_volume_sigma_ml: float = DEFAULT_MODEL_VOLUME_SIGMA_ML,
    feed_schedule_mode: str = "verified",
    documented_feed_segments: Sequence[Mapping[str, Any]] | None = None,
    feed_schedule_provenance: Mapping[str, Any] | None = None,
    trace_rate_correction: float = DEFAULT_TRACE_RATE_CORRECTION,
    trace_rate_correction_source: str = DEFAULT_TRACE_RATE_CORRECTION_SOURCE,
    effective_feed_rate_overrides_g_h: Sequence[float] | None = None,
    effective_feed_rate_override_source: str | None = None,
    constraint_mode: str = DEFAULT_CONSTRAINT_MODE,
) -> dict[str, Any]:
    review_path = Path(review_state_path)
    protocol_path = Path(injection_protocol_path)
    review = json.loads(review_path.read_text(encoding="utf-8-sig"))
    metadata = review.get("metadata") or {}
    run = str(metadata.get("run", "")).strip().lower()
    if not run:
        raise ValueError(f"Overflow review state has no run: {review_path}")
    if metadata.get("weight_data_valid") is False:
        reason = str(metadata.get("weight_data_reason") or "unspecified reason")
        raise ValueError(
            f"Overflow-weight data are marked invalid for {run}: {reason}"
        )
    schedule_mode = str(feed_schedule_mode).strip().lower()
    if schedule_mode not in {"verified", "trace-derived"}:
        raise ValueError(f"Unsupported feed schedule mode: {feed_schedule_mode!r}")
    resolved_constraint_mode = normalize_constraint_mode(constraint_mode)
    if (
        schedule_mode == "verified"
        and metadata.get("gravity_schedule_verified") is False
    ):
        raise ValueError(
            f"Overflow gravity-feed schedule is not verified for {run}; "
            "manual readings cannot be used as phase targets yet"
        )
    injection_start = _parse_protocol_time(metadata["injection_start"])
    overrides = {int(key): float(value) for key, value in (overrides_by_index or {}).items()}
    exclusions = {int(value) for value in (excluded_indices or ())}

    rows: list[tuple[datetime, float, int]] = []
    override_rows: list[dict[str, Any]] = []
    for record in review.get("records") or []:
        if record.get("status") not in CONFIRMED_STATUSES:
            continue
        if record.get("weight_g") is None:
            continue
        index = int(record["index"])
        if index in exclusions:
            continue
        original = float(record["weight_g"])
        weight = overrides.get(index, original)
        if index in overrides:
            override_rows.append(
                {
                    "index": index,
                    "image_id": record.get("image_id"),
                    "image_time": record.get("image_time"),
                    "original_weight_g": original,
                    "replacement_weight_g": weight,
                    "reason": "verified transcription correction for target derivation",
                }
            )
        rows.append((_parse_protocol_time(record["image_time"]), weight, index))
    rows.sort(key=lambda row: row[0])
    if len(rows) < 10:
        raise ValueError(f"Too few confirmed overflow readings for {run}: {len(rows)}")
    times_h = np.asarray(
        [(timestamp - injection_start).total_seconds() / 3600.0 for timestamp, _, _ in rows],
        dtype=float,
    )
    weights_g = np.asarray([weight for _, weight, _ in rows], dtype=float)
    if np.any(np.diff(times_h) <= 0.0):
        raise ValueError("Confirmed overflow reading times must be strictly increasing")
    if np.any(np.diff(weights_g) < -0.2):
        raise ValueError(
            "Confirmed cumulative overflow weights are not monotone after overrides"
        )

    stages = _load_injection_protocol(protocol_path)
    gas_density_kg_m3 = float(stages[0]["density_kg_m3"])
    if not all(
        math.isclose(float(stage["density_kg_m3"]), gas_density_kg_m3)
        for stage in stages
    ):
        raise ValueError("Injection protocol uses more than one gas density")

    gravity_end: datetime | None = None
    gravity_end_h: float | None = None
    endpoint_time_h: float | None = None
    endpoint_weight_g: float | None = None
    endpoint_gap_s: float | None = None
    zero_gas_anchor_assumption: str | None = None
    trace_diagnostics: dict[str, Any] | None = None
    effective_feed_segments: list[dict[str, Any]]
    baseline_slope_g_h: float | None
    if schedule_mode == "verified":
        gravity_end = _parse_protocol_time(metadata["gravity_end"])
        pre_mask = times_h < 0.0
        pre_count = int(np.count_nonzero(pre_mask))
        if pre_count < 2:
            raise ValueError("At least two pre-injection scale readings are required")
        design = np.column_stack(
            (times_h[pre_mask], np.ones(np.count_nonzero(pre_mask)))
        )
        baseline_slope_g_h, baseline_weight_g = np.linalg.lstsq(
            design,
            weights_g[pre_mask],
            rcond=None,
        )[0]
        pre_residual = weights_g[pre_mask] - (
            baseline_slope_g_h * times_h[pre_mask] + baseline_weight_g
        )
        baseline_sigma_g = max(
            0.25,
            float(np.std(pre_residual, ddof=2)) if pre_residual.size > 2 else 1.0,
        )

        gravity_end_h = (gravity_end - injection_start).total_seconds() / 3600.0
        if gravity_end_h <= 0.0:
            raise ValueError("Gravity feed must end after injection starts")
        nearest_endpoint_index = int(np.argmin(np.abs(times_h - gravity_end_h)))
        endpoint_time_h = float(times_h[nearest_endpoint_index])
        endpoint_weight_g = float(weights_g[nearest_endpoint_index])
        endpoint_gap_s = abs(endpoint_time_h - gravity_end_h) * 3600.0
        feed_rate_source = "zero-gas endpoint anchor"
        effective_feed_rate_sigma_g_h = float(feed_rate_sigma_g_h)
        if endpoint_gap_s <= MAX_ZERO_GAS_ANCHOR_GAP_S:
            effective_feed_g_h = (
                endpoint_weight_g - float(baseline_weight_g)
            ) / gravity_end_h
            zero_gas_anchor_assumption = (
                "free gas is negligible at the confirmed reading nearest "
                "gravity-feed end"
            )
        else:
            configured_rate = metadata.get("gravity_feed_g_per_h")
            if configured_rate is None:
                raise ValueError(
                    "No confirmed scale reading lies within "
                    f"{MAX_ZERO_GAS_ANCHOR_GAP_S:.0f} s of gravity-feed end, and "
                    "the review metadata has no verified feed rate"
                )
            effective_feed_g_h = float(configured_rate)
            if not np.isfinite(effective_feed_g_h) or effective_feed_g_h <= 0.0:
                raise ValueError(
                    f"Invalid verified gravity-feed rate for {run}: {configured_rate!r}"
                )
            feed_rate_source = "verified review metadata"
            effective_feed_rate_sigma_g_h = max(
                float(feed_rate_sigma_g_h),
                0.01 * effective_feed_g_h,
            )
        effective_feed_segments = [
            {
                "start_h": 0.0,
                "end_h": gravity_end_h,
                "nominal_rate_g_per_h": float(
                    metadata.get("gravity_feed_g_per_h") or effective_feed_g_h
                ),
                "effective_rate_g_per_h": effective_feed_g_h,
                "effective_rate_sigma_g_per_h": effective_feed_rate_sigma_g_h,
                "schedule_anomaly": False,
            }
        ]
        coverage_end_h = min(gravity_end_h, float(times_h[-1]))
    else:
        if documented_feed_segments is None:
            raise ValueError(
                "Trace-derived feed schedules require documented feed segments"
            )
        injection_end_h = (
            max(stage["end"] for stage in stages) - injection_start
        ).total_seconds() / 3600.0
        (
            effective_feed_segments,
            baseline_weight_g,
            baseline_sigma_g,
            pre_count,
            trace_diagnostics,
        ) = _derive_trace_feed_schedule(
            times_h=times_h,
            weights_g=weights_g,
            injection_end_h=injection_end_h,
            documented_feed_segments=documented_feed_segments,
            trace_rate_correction=trace_rate_correction,
            trace_rate_correction_source=trace_rate_correction_source,
            effective_feed_rate_overrides_g_h=(
                effective_feed_rate_overrides_g_h
            ),
            effective_feed_rate_override_source=(
                effective_feed_rate_override_source
            ),
        )
        baseline_slope_g_h = trace_diagnostics.get("baseline_slope_g_per_h")
        effective_feed_g_h = float(
            effective_feed_segments[0]["effective_rate_g_per_h"]
        )
        effective_feed_rate_sigma_g_h = float(
            effective_feed_segments[0]["effective_rate_sigma_g_per_h"]
        )
        feed_rate_source = "validated trace constrained by documented segments"
        coverage_end_h = float(times_h[-1])

    targets: list[dict[str, Any]] = []
    for raw_time in sorted({float(value) for value in evaluation_times_h}):
        if raw_time < 0.0 or raw_time > coverage_end_h:
            continue
        cumulative_outflow_g = float(np.interp(raw_time, times_h, weights_g))
        if schedule_mode == "verified":
            cumulative_feed_g = effective_feed_g_h * raw_time
            cumulative_feed_sigma_g = effective_feed_rate_sigma_g_h * raw_time
        else:
            cumulative_feed_g = _integrated_piecewise_feed_g(
                raw_time,
                effective_feed_segments,
                rate_key="effective_rate_g_per_h",
            )
            segment_sigma_terms: list[float] = []
            for index, segment in enumerate(effective_feed_segments):
                start_h = float(segment["start_h"])
                end_h = (
                    float(effective_feed_segments[index + 1]["start_h"])
                    if index + 1 < len(effective_feed_segments)
                    else raw_time
                )
                duration_h = max(0.0, min(raw_time, end_h) - start_h)
                segment_sigma_terms.append(
                    float(segment["effective_rate_sigma_g_per_h"]) * duration_h
                )
            cumulative_feed_sigma_g = math.sqrt(
                sum(value * value for value in segment_sigma_terms)
            )
        gas_liquid_equivalent_g = max(
            0.0,
            cumulative_outflow_g
            - float(baseline_weight_g)
            - cumulative_feed_g,
        )
        gas_volume_m3 = (
            gas_liquid_equivalent_g / 1000.0 / float(liquid_density_kg_m3)
        )
        gas_mass_kg = gas_volume_m3 * gas_density_kg_m3
        at = injection_start + timedelta(hours=raw_time)
        injected_mass_kg = _injected_mass_kg(stages, at)
        gas_mass_kg = min(gas_mass_kg, injected_mass_kg)
        volume_sigma_g_equivalent = math.sqrt(
            float(scale_sigma_g) ** 2
            + float(baseline_sigma_g) ** 2
            + float(model_volume_sigma_ml) ** 2
            + cumulative_feed_sigma_g**2
        )
        phase_mass_sigma_kg = (
            volume_sigma_g_equivalent
            / 1000.0
            / float(liquid_density_kg_m3)
            * gas_density_kg_m3
        )
        target = asdict(
                OverflowPhaseTarget(
                    time_h=raw_time,
                    injected_mass_kg=injected_mass_kg,
                    gas_mass_kg=gas_mass_kg,
                    aqueous_mass_kg=max(0.0, injected_mass_kg - gas_mass_kg),
                    phase_mass_sigma_kg=phase_mass_sigma_kg,
                    cumulative_outflow_g=cumulative_outflow_g,
                    inferred_gas_volume_ml=gas_volume_m3 * 1e6,
                )
            )
        target.update(
            {
                "target_source": "measured-run-specific",
                "cumulative_outflow_observed": True,
                "cumulative_feed_g": cumulative_feed_g,
                "cumulative_feed_sigma_g": cumulative_feed_sigma_g,
            }
        )
        targets.append(target)
    if not targets:
        raise ValueError("No requested evaluation time is covered by overflow data")

    return {
        "schema": 1,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "run": run,
        "constraint_mode": resolved_constraint_mode,
        "match_tolerance_h": DEFAULT_MATCH_TOLERANCE_H,
        "source": {
            "review_state_path": str(review_path),
            "review_state_sha256": _sha256(review_path),
            "injection_protocol_path": str(protocol_path),
            "injection_protocol_sha256": _sha256(protocol_path),
            "confirmed_reading_count": len(rows),
            "manual_overrides": override_rows,
            "excluded_confirmed_indices": sorted(exclusions),
            "feed_schedule_mode": schedule_mode,
            "feed_schedule_provenance": dict(feed_schedule_provenance or {}),
        },
        "physics": {
            "balance": "delta_V_gas = delta_mass_out/rho_liquid - integral(Q_feed dt)",
            "liquid_density_kg_m3": float(liquid_density_kg_m3),
            "gas_density_kg_m3": gas_density_kg_m3,
            "effective_feed_g_per_h": effective_feed_g_h,
            "effective_feed_segments": effective_feed_segments,
            "feed_rate_source": feed_rate_source,
            "feed_rate_sigma_g_per_h": effective_feed_rate_sigma_g_h,
            "scale_sigma_g": float(scale_sigma_g),
            "model_volume_sigma_ml": float(model_volume_sigma_ml),
        },
        "derivation": {
            "injection_start": injection_start.isoformat(),
            "gravity_feed_end": gravity_end.isoformat() if gravity_end else None,
            "baseline_weight_at_injection_g": float(baseline_weight_g),
            "baseline_slope_g_per_h": (
                float(baseline_slope_g_h)
                if baseline_slope_g_h is not None
                else None
            ),
            "baseline_fit_sigma_g": baseline_sigma_g,
            "baseline_fit_reading_count": pre_count,
            "zero_gas_anchor_time_h": endpoint_time_h,
            "zero_gas_anchor_weight_g": endpoint_weight_g,
            "zero_gas_anchor_gap_to_feed_end_s": endpoint_gap_s,
            "zero_gas_anchor_assumption": zero_gas_anchor_assumption,
            "trace_feed_diagnostics": trace_diagnostics,
            "phase_balance_coverage_end_h": coverage_end_h,
            "interpolation": "linear between confirmed cumulative scale readings",
            "uncertainty": (
                "quadrature of scale, baseline fit, feed-rate, and model-volume terms"
            ),
        },
        "targets": targets,
    }


def build_overflow_fraction_reference(
    donor_payloads: Mapping[str, Mapping[str, Any]],
    *,
    minimum_donors: int = 3,
    relative_sigma_floor: float = 0.10,
) -> dict[str, Any]:
    """Build a robust, uncertainty-aware phase prior from measured runs.

    Gas mass is normalized by injected mass before runs are combined.  The
    median is used as the center, while the uncertainty includes both the
    donor measurement uncertainty and robust between-run dispersion.
    """

    if minimum_donors < 2:
        raise ValueError("At least two donors are required for a series reference")
    if not math.isfinite(relative_sigma_floor) or relative_sigma_floor < 0.0:
        raise ValueError("Relative sigma floor must be finite and non-negative")

    by_time: dict[float, list[dict[str, float | str]]] = {}
    donor_rows: list[dict[str, Any]] = []
    for raw_run, payload in sorted(donor_payloads.items()):
        run = str(raw_run).strip().lower()
        if int(payload.get("schema", 0)) != 1:
            raise ValueError(f"Unsupported donor overflow schema for {run}")
        payload_run = str(payload.get("run", "")).strip().lower()
        if payload_run != run:
            raise ValueError(
                f"Donor payload run mismatch: key={run!r}, payload={payload_run!r}"
            )
        targets = payload.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"Donor overflow payload has no targets: {run}")
        donor_rows.append(
            {
                "run": run,
                "payload_sha256": _mapping_sha256(payload),
                "target_count": len(targets),
                "assignment_mode": (payload.get("source") or {}).get(
                    "assignment_mode", "measured"
                ),
            }
        )
        for target in targets:
            injected = float(target["injected_mass_kg"])
            if not math.isfinite(injected) or injected <= 0.0:
                continue
            time_h = round(float(target["time_h"]), 6)
            gas_fraction = min(
                1.0,
                max(0.0, float(target["gas_mass_kg"]) / injected),
            )
            relative_sigma = max(
                0.0,
                float(target["phase_mass_sigma_kg"]) / injected,
            )
            by_time.setdefault(time_h, []).append(
                {
                    "run": run,
                    "gas_fraction": gas_fraction,
                    "relative_sigma": relative_sigma,
                }
            )

    rows: list[dict[str, Any]] = []
    for time_h in sorted(by_time):
        samples = by_time[time_h]
        if len(samples) < minimum_donors:
            continue
        fractions = np.asarray(
            [float(sample["gas_fraction"]) for sample in samples], dtype=float
        )
        donor_sigmas = np.asarray(
            [float(sample["relative_sigma"]) for sample in samples], dtype=float
        )
        center = float(np.median(fractions))
        mean = float(np.mean(fractions))
        robust_between_run_sigma = float(
            1.4826 * np.median(np.abs(fractions - center))
        )
        typical_measurement_sigma = float(np.median(donor_sigmas))
        combined_sigma = max(
            float(relative_sigma_floor),
            math.hypot(typical_measurement_sigma, robust_between_run_sigma),
        )
        rows.append(
            {
                "time_h": time_h,
                "gas_fraction_median": center,
                "gas_fraction_mean": mean,
                "relative_sigma": combined_sigma,
                "typical_measurement_relative_sigma": typical_measurement_sigma,
                "between_run_relative_sigma": robust_between_run_sigma,
                "donor_count": len(samples),
                "donors": samples,
            }
        )
    if not rows:
        raise ValueError(
            "No overflow time has enough measured donors for a series reference"
        )
    return {
        "schema": 1,
        "model": REFERENCE_MODEL_NAME,
        "model_version": REFERENCE_MODEL_VERSION,
        "center_statistic": "median gas fraction normalized by injected mass",
        "uncertainty": (
            "quadrature of median donor measurement sigma and 1.4826*MAD "
            "between runs, with a fixed relative floor"
        ),
        "minimum_donors": int(minimum_donors),
        "relative_sigma_floor": float(relative_sigma_floor),
        "donor_runs": [row["run"] for row in donor_rows],
        "donor_payloads": donor_rows,
        "targets": rows,
    }


def build_imputed_overflow_phase_targets(
    *,
    run: str,
    injection_protocol_path: str | Path,
    reference: Mapping[str, Any],
    reason: str,
    constraint_mode: str = DEFAULT_CONSTRAINT_MODE,
) -> dict[str, Any]:
    """Scale a series gas-fraction reference to one run's injection protocol."""

    run = str(run).strip().lower()
    if not run:
        raise ValueError("Run is required for imputed overflow targets")
    if reference.get("model") != REFERENCE_MODEL_NAME:
        raise ValueError("Unsupported overflow series reference")
    resolved_constraint_mode = normalize_constraint_mode(constraint_mode)
    protocol_path = Path(injection_protocol_path)
    stages = _load_injection_protocol(protocol_path)
    injection_start = min(stage["start"] for stage in stages)
    gas_density_kg_m3 = float(stages[0]["density_kg_m3"])
    if not all(
        math.isclose(float(stage["density_kg_m3"]), gas_density_kg_m3)
        for stage in stages
    ):
        raise ValueError("Injection protocol uses more than one gas density")

    targets: list[dict[str, Any]] = []
    for row in reference.get("targets") or []:
        time_h = float(row["time_h"])
        at = injection_start + timedelta(hours=time_h)
        injected_mass_kg = _injected_mass_kg(stages, at)
        if injected_mass_kg <= 0.0:
            continue
        gas_fraction = min(
            1.0,
            max(0.0, float(row["gas_fraction_median"])),
        )
        relative_sigma = max(0.0, float(row["relative_sigma"]))
        gas_mass_kg = injected_mass_kg * gas_fraction
        gas_volume_m3 = gas_mass_kg / gas_density_kg_m3
        target = asdict(
            OverflowPhaseTarget(
                time_h=time_h,
                injected_mass_kg=injected_mass_kg,
                gas_mass_kg=gas_mass_kg,
                aqueous_mass_kg=max(0.0, injected_mass_kg - gas_mass_kg),
                phase_mass_sigma_kg=injected_mass_kg * relative_sigma,
                cumulative_outflow_g=0.0,
                inferred_gas_volume_ml=gas_volume_m3 * 1e6,
            )
        )
        target.update(
            {
                "target_source": "imputed-series-reference",
                "cumulative_outflow_observed": False,
                "reference_donor_count": int(row["donor_count"]),
                "reference_gas_fraction_mean": float(row["gas_fraction_mean"]),
                "reference_gas_fraction_median": gas_fraction,
                "reference_relative_sigma": relative_sigma,
            }
        )
        targets.append(target)
    if not targets:
        raise ValueError(f"No imputed overflow targets could be built for {run}")

    reference_sha256 = _mapping_sha256(reference)
    return {
        "schema": 1,
        "model": REFERENCE_MODEL_NAME,
        "model_version": REFERENCE_MODEL_VERSION,
        "run": run,
        "constraint_mode": resolved_constraint_mode,
        "match_tolerance_h": DEFAULT_MATCH_TOLERANCE_H,
        "source": {
            "assignment_mode": "imputed-series-reference",
            "reason": str(reason).strip(),
            "reference_sha256": reference_sha256,
            "donor_runs": list(reference.get("donor_runs") or []),
            "injection_protocol_path": str(protocol_path),
            "injection_protocol_sha256": _sha256(protocol_path),
            "cumulative_outflow_observed": False,
        },
        "physics": {
            "balance": (
                "robust measured-series gas fraction scaled by the run-specific "
                "injected CO2 mass"
            ),
            "gas_density_kg_m3": gas_density_kg_m3,
            "feed_rate_source": "not used for imputed series reference",
        },
        "derivation": {
            "injection_start": injection_start.isoformat(),
            "center_statistic": reference.get("center_statistic"),
            "uncertainty": reference.get("uncertainty"),
            "relative_sigma_floor": reference.get("relative_sigma_floor"),
            "interpretation": (
                "population prior, not a run-specific overflow measurement"
            ),
        },
        "targets": targets,
    }


def build_hybrid_overflow_phase_targets(
    *,
    run: str,
    measured_payload: Mapping[str, Any],
    imputed_payload: Mapping[str, Any],
    reason: str,
    constraint_mode: str | None = None,
) -> dict[str, Any]:
    """Combine measured early targets with an imputed tail for one run."""

    normalized_run = str(run).strip().lower()
    for label, payload in (
        ("measured", measured_payload),
        ("imputed", imputed_payload),
    ):
        payload_run = str(payload.get("run", "")).strip().lower()
        if payload_run != normalized_run:
            raise ValueError(
                f"{label.title()} overflow payload is for {payload_run!r}, "
                f"not {normalized_run!r}"
            )
    measured_by_time = {
        round(float(target["time_h"]), 6): dict(target)
        for target in measured_payload.get("targets") or []
    }
    imputed_by_time = {
        round(float(target["time_h"]), 6): dict(target)
        for target in imputed_payload.get("targets") or []
    }
    if not measured_by_time or not imputed_by_time:
        raise ValueError("Hybrid overflow targets require measured and imputed data")
    inherited_modes = {
        normalize_constraint_mode(measured_payload.get("constraint_mode")),
        normalize_constraint_mode(imputed_payload.get("constraint_mode")),
    }
    resolved_constraint_mode = (
        normalize_constraint_mode(constraint_mode)
        if constraint_mode is not None
        else next(iter(inherited_modes))
    )
    if constraint_mode is None and len(inherited_modes) != 1:
        raise ValueError("Measured and imputed overflow constraint modes differ")

    targets: list[dict[str, Any]] = []
    measured_times: list[float] = []
    imputed_times: list[float] = []
    for time_h in sorted(set(measured_by_time) | set(imputed_by_time)):
        if time_h in measured_by_time:
            target = measured_by_time[time_h]
            target["target_source"] = "measured-run-specific"
            target["cumulative_outflow_observed"] = True
            measured_times.append(time_h)
        else:
            target = imputed_by_time[time_h]
            target["target_source"] = "imputed-cohort-tail"
            target["cumulative_outflow_observed"] = False
            imputed_times.append(time_h)
        targets.append(target)

    measured_source = dict(measured_payload.get("source") or {})
    imputed_source = dict(imputed_payload.get("source") or {})
    return {
        "schema": 1,
        "model": HYBRID_MODEL_NAME,
        "model_version": HYBRID_MODEL_VERSION,
        "run": normalized_run,
        "constraint_mode": resolved_constraint_mode,
        "match_tolerance_h": min(
            float(
                measured_payload.get(
                    "match_tolerance_h", DEFAULT_MATCH_TOLERANCE_H
                )
            ),
            float(
                imputed_payload.get(
                    "match_tolerance_h", DEFAULT_MATCH_TOLERANCE_H
                )
            ),
        ),
        "source": {
            "assignment_mode": "hybrid-run-specific-plus-imputed-tail",
            "reason": str(reason).strip(),
            "measured_payload_sha256": _mapping_sha256(measured_payload),
            "imputed_payload_sha256": _mapping_sha256(imputed_payload),
            "review_state_path": measured_source.get("review_state_path"),
            "review_state_sha256": measured_source.get("review_state_sha256"),
            "injection_protocol_path": measured_source.get(
                "injection_protocol_path"
            ),
            "injection_protocol_sha256": measured_source.get(
                "injection_protocol_sha256"
            ),
            "donor_runs": list(imputed_source.get("donor_runs") or []),
            "cumulative_outflow_observed": "measured targets only",
        },
        "physics": {
            "balance": (
                "run-specific measured phase balance through available scale "
                "coverage; normalized cohort gas fraction thereafter"
            ),
            "measured": measured_payload.get("physics"),
            "imputed": imputed_payload.get("physics"),
        },
        "derivation": {
            "measured_target_times_h": measured_times,
            "imputed_target_times_h": imputed_times,
            "measured_coverage_end_h": max(measured_times),
            "transition_rule": (
                "use measured targets whenever available; otherwise use the "
                "precomputed cohort reference without interpolation across modes"
            ),
        },
        "targets": targets,
    }


def write_overflow_phase_targets(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-state", required=True)
    parser.add_argument("--injection-protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument(
        "--constraint-mode",
        choices=sorted(VALID_CONSTRAINT_MODES),
        default=DEFAULT_CONSTRAINT_MODE,
        help=(
            "Use gas-upper-bound when overflow is an upper limit on free gas "
            "rather than a phase-partition target."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="INDEX=WEIGHT_G",
    )
    args = parser.parse_args(argv)
    overrides: dict[int, float] = {}
    for item in args.override:
        key, value = str(item).split("=", 1)
        overrides[int(key)] = float(value)
    payload = build_overflow_phase_targets(
        review_state_path=args.review_state,
        injection_protocol_path=args.injection_protocol,
        evaluation_times_h=args.times,
        overrides_by_index=overrides,
        constraint_mode=args.constraint_mode,
    )
    path = write_overflow_phase_targets(args.output, payload)
    print(
        json.dumps(
            {
                "path": str(path),
                "targets": len(payload["targets"]),
                "constraint_mode": payload["constraint_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
