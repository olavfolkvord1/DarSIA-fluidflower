"""Re-rank and validate the final AC calibration candidates.

The Optuna objective used during the screen is deliberately composite.  This
utility makes the final decision separately and reproducibly:

1. Re-rank every feasible point-wise trial from the two compatible searches.
2. Reject candidates that violate fixed phase and temporal guardrails.
3. Replay a small candidate set on the exact calibration frames.
4. Accept the first training-ranked candidate that passes untouched holdouts.

No calibration stored below ``Z:\\Albus\\Results`` is modified.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


REPO_ROOT = series_root("ffac")
CAMPAIGN_ID = "final_method_selection_20260814"
DEFAULT_OUTPUT_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
RUNS = ("ac20", "ac24", "ac27", "ac60")
SOURCE_VARIANT = "balance_gas24_diff_w0p25"
FINAL_VARIANT = "reranked_balance_gas24_diff_w0p25"
SOURCE_ROOTS = (
    (
        "diffusion_5000",
        configured_tool_path('ffac', 'calibration_logs') / 'mass_conserving_diffusion_screen_20260813_5000/balance_gas24_diff_w0p25',
    ),
    (
        "pointwise_8000",
        configured_tool_path('ffac', 'calibration_logs') / 'temporal_balance_width_screen_20260813_8000/pointwise_gaswidth085',
    ),
)
OVERFLOW_TARGET_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'mass_conserving_diffusion_screen_20260813_5000/overflow_targets'

# These times are not part of the 17-frame calibration schedule.  They span
# both injections and the complete post-injection interval.
HOLDOUT_TIMES_H = (
    0.333,
    0.500,
    1.000,
    1.600,
    2.200,
    2.750,
    3.500,
    4.750,
    5.500,
    7.000,
    10.500,
    13.500,
    18.000,
    25.500,
    32.000,
    38.000,
    45.000,
)

SCORE_WEIGHTS = {
    "normalized_mae": 0.25,
    "worst_window_mae": 0.20,
    "ratio_4p1_abs_bias": 0.20,
    "late_mean_abs_bias": 0.20,
    "post_injection_total_variation": 0.15,
}

# V2 gives every operational window equal influence while measuring error
# relative to the mass that is actually present at that time.  The symmetric
# denominator is stable close to zero and the Huber transform prevents a
# single very early frame from dominating the complete search.
V2_RELATIVE_FLOOR_FRACTION = 0.02
V2_HUBER_DELTA = 0.25
V2_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR = 0.50
V2_HOLDOUT_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR = 0.60

TRAINING_GUARDRAILS = {
    "maximum_gas_envelope_excess_fraction": 0.005,
    "maximum_aqueous_decrease_fraction": 0.15,
    "minimum_ratio_4p1": 0.75,
    "maximum_ratio_4p1": 1.25,
    "minimum_late_mean_ratio": 0.85,
    "maximum_late_mean_ratio": 1.15,
    "minimum_terminal_ratio": 0.85,
    "maximum_terminal_ratio": 1.15,
    "maximum_worst_window_mae": 0.25,
}

HOLDOUT_GUARDRAILS = {
    "maximum_normalized_mae": 0.15,
    "maximum_worst_window_mae": 0.30,
    "maximum_absolute_error_fraction": 0.35,
    "minimum_late_mean_ratio": 0.80,
    "maximum_late_mean_ratio": 1.20,
    "minimum_terminal_ratio": 0.80,
    "maximum_terminal_ratio": 1.20,
    "maximum_gas_envelope_excess_fraction": 0.01,
}


@dataclass(frozen=True)
class CandidateSource:
    name: str
    root: Path


@dataclass
class Candidate:
    run: str
    source: str
    history_path: str
    history_iter: int
    history_objective: float
    params: dict[str, Any]
    metrics: dict[str, Any]
    params_sha256: str
    diagnostics: dict[str, float]
    guardrail_failures: list[str]
    source_final: bool = False
    rank: int | None = None
    candidate_path: str | None = None

    @property
    def passes_guardrails(self) -> bool:
        return not self.guardrail_failures


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _params_sha256(params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(params),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _metric_value(value: Any, name: str) -> float:
    raw = value.get(name) if isinstance(value, Mapping) else getattr(value, name)
    return float(raw)


def _metric_series(metrics: Mapping[str, Any]) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for time_key, metric in metrics.items():
        time_h = float(str(time_key).removesuffix("h"))
        rows.append(
            (
                time_h,
                _metric_value(metric, "injected_full"),
                _metric_value(metric, "total_full"),
                _metric_value(metric, "gaseous_full"),
                _metric_value(metric, "aqueous_full"),
            )
        )
    return sorted(rows)


def _window_name(time_h: float) -> str:
    if time_h <= 0.92:
        return "I1"
    if time_h < 4.0:
        return "I2"
    if time_h <= 8.25:
        return "early_redistribution"
    return "late_redistribution"


def _huber_loss(value: float, *, delta: float = V2_HUBER_DELTA) -> float:
    magnitude = abs(float(value))
    if magnitude <= delta:
        return 0.5 * magnitude * magnitude / delta
    return magnitude - 0.5 * delta


def candidate_diagnostics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Calculate the fixed selection diagnostics from time-resolved masses."""

    rows = _metric_series(metrics)
    if not rows:
        raise ValueError("Candidate has no time-resolved metrics.")
    maximum_injected = max(row[1] for row in rows)
    if not maximum_injected > 0.0:
        raise ValueError("Candidate has no positive injected mass.")

    normalized_errors = [
        abs(total - injected) / maximum_injected
        for _time, injected, total, _gas, _aqueous in rows
    ]
    relative_floor = V2_RELATIVE_FLOOR_FRACTION * maximum_injected
    symmetric_errors = [
        2.0 * abs(total - injected)
        / max(abs(total) + abs(injected), relative_floor)
        for _time, injected, total, _gas, _aqueous in rows
    ]
    huber_errors = [_huber_loss(value) for value in symmetric_errors]
    windows: dict[str, list[float]] = {
        "I1": [],
        "I2": [],
        "early_redistribution": [],
        "late_redistribution": [],
    }
    symmetric_windows = {name: [] for name in windows}
    huber_windows = {name: [] for name in windows}
    for row, error, symmetric, huber in zip(
        rows,
        normalized_errors,
        symmetric_errors,
        huber_errors,
        strict=True,
    ):
        window = _window_name(row[0])
        windows[window].append(error)
        symmetric_windows[window].append(symmetric)
        huber_windows[window].append(huber)
    if any(not values for values in windows.values()):
        raise ValueError("Candidate does not cover all four operational windows.")
    window_mae = {
        name: sum(values) / len(values) for name, values in windows.items()
    }
    window_symmetric = {
        name: sum(values) / len(values)
        for name, values in symmetric_windows.items()
    }
    window_huber = {
        name: sum(values) / len(values)
        for name, values in huber_windows.items()
    }

    near_4p1 = min(rows, key=lambda row: abs(row[0] - 4.1))
    ratio_4p1 = near_4p1[2] / max(near_4p1[1], 1e-15)
    late = [row for row in rows if row[0] > 8.25]
    late_ratios = [row[2] / max(row[1], 1e-15) for row in late]
    late_mean_ratio = sum(late_ratios) / len(late_ratios)
    terminal_ratio = rows[-1][2] / max(rows[-1][1], 1e-15)

    post = [row for row in rows if row[0] >= 4.083 - 1e-9]
    total_variation = sum(
        abs(current[2] - previous[2])
        for previous, current in zip(post, post[1:])
    ) / maximum_injected
    aqueous_decrease = sum(
        max(0.0, previous[4] - current[4])
        for previous, current in zip(post, post[1:])
    ) / maximum_injected
    gas_increase = sum(
        max(0.0, current[3] - previous[3])
        for previous, current in zip(post, post[1:])
    ) / maximum_injected

    gas_envelope_excess = 0.0
    for time_h, injected, _total, gas, _aqueous in rows:
        if time_h < 24.0 or injected <= 0.0:
            continue
        allowed_fraction = 0.05 * 2.0 ** (-(time_h - 24.0) / 12.0)
        gas_envelope_excess = max(
            gas_envelope_excess,
            gas / injected - allowed_fraction,
        )
    gas_envelope_excess = max(0.0, gas_envelope_excess)

    diagnostics = {
        "normalized_mae": sum(normalized_errors) / len(normalized_errors),
        "maximum_absolute_error_fraction": max(normalized_errors),
        **{f"window_mae_{name}": value for name, value in window_mae.items()},
        **{
            f"window_symmetric_error_{name}": value
            for name, value in window_symmetric.items()
        },
        **{
            f"window_huber_{name}": value
            for name, value in window_huber.items()
        },
        "worst_window_mae": max(window_mae.values()),
        "temporal_balanced_symmetric_error": (
            sum(window_symmetric.values()) / len(window_symmetric)
        ),
        "temporal_balanced_huber": sum(window_huber.values()) / len(window_huber),
        "worst_window_symmetric_error": max(window_symmetric.values()),
        "ratio_4p1": ratio_4p1,
        "ratio_4p1_abs_bias": abs(ratio_4p1 - 1.0),
        "late_mean_ratio": late_mean_ratio,
        "late_mean_abs_bias": abs(late_mean_ratio - 1.0),
        "terminal_ratio": terminal_ratio,
        "late_ratio_range": max(late_ratios) - min(late_ratios),
        "post_injection_total_variation": total_variation,
        "post_injection_aqueous_decrease_fraction": aqueous_decrease,
        "post_injection_gas_increase_fraction": gas_increase,
        "maximum_gas_envelope_excess_fraction": gas_envelope_excess,
    }
    diagnostics["selection_score"] = sum(
        SCORE_WEIGHTS[name] * diagnostics[name]
        for name in SCORE_WEIGHTS
    )
    return diagnostics


def training_guardrail_failures(
    diagnostics: Mapping[str, float],
    *,
    include_late_gas: bool = True,
) -> list[str]:
    limits = TRAINING_GUARDRAILS
    failures: list[str] = []
    if include_late_gas and (
        diagnostics["maximum_gas_envelope_excess_fraction"]
        > limits["maximum_gas_envelope_excess_fraction"]
    ):
        failures.append("late_gas_envelope")
    if (
        diagnostics["post_injection_aqueous_decrease_fraction"]
        > limits["maximum_aqueous_decrease_fraction"]
    ):
        failures.append("aqueous_decrease")
    if not (
        limits["minimum_ratio_4p1"]
        <= diagnostics["ratio_4p1"]
        <= limits["maximum_ratio_4p1"]
    ):
        failures.append("ratio_4p1")
    if not (
        limits["minimum_late_mean_ratio"]
        <= diagnostics["late_mean_ratio"]
        <= limits["maximum_late_mean_ratio"]
    ):
        failures.append("late_mean_ratio")
    if not (
        limits["minimum_terminal_ratio"]
        <= diagnostics["terminal_ratio"]
        <= limits["maximum_terminal_ratio"]
    ):
        failures.append("terminal_ratio")
    if (
        diagnostics["worst_window_mae"]
        > limits["maximum_worst_window_mae"]
    ):
        failures.append("worst_window")
    return failures


def training_guardrail_failures_v2(
    diagnostics: Mapping[str, float],
    *,
    include_late_gas: bool = True,
) -> list[str]:
    """V1 physical gates plus an injection-sensitive temporal gate."""

    failures = training_guardrail_failures(
        diagnostics,
        include_late_gas=include_late_gas,
    )
    if (
        diagnostics["worst_window_symmetric_error"]
        > V2_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR
    ):
        failures.append("worst_relative_window")
    return failures


def holdout_guardrail_failures(
    diagnostics: Mapping[str, float],
    *,
    include_late_gas: bool = True,
) -> list[str]:
    limits = HOLDOUT_GUARDRAILS
    failures: list[str] = []
    for metric, limit_name, label in (
        ("normalized_mae", "maximum_normalized_mae", "mean_error"),
        ("worst_window_mae", "maximum_worst_window_mae", "worst_window"),
        (
            "maximum_absolute_error_fraction",
            "maximum_absolute_error_fraction",
            "maximum_error",
        ),
        (
            "maximum_gas_envelope_excess_fraction",
            "maximum_gas_envelope_excess_fraction",
            "late_gas_envelope",
        ),
    ):
        if label == "late_gas_envelope" and not include_late_gas:
            continue
        if diagnostics[metric] > limits[limit_name]:
            failures.append(label)
    if not (
        limits["minimum_late_mean_ratio"]
        <= diagnostics["late_mean_ratio"]
        <= limits["maximum_late_mean_ratio"]
    ):
        failures.append("late_mean_ratio")
    if not (
        limits["minimum_terminal_ratio"]
        <= diagnostics["terminal_ratio"]
        <= limits["maximum_terminal_ratio"]
    ):
        failures.append("terminal_ratio")
    return failures


def holdout_guardrail_failures_v2(
    diagnostics: Mapping[str, float],
    *,
    include_late_gas: bool = True,
) -> list[str]:
    failures = holdout_guardrail_failures(
        diagnostics,
        include_late_gas=include_late_gas,
    )
    if (
        diagnostics["worst_window_symmetric_error"]
        > V2_HOLDOUT_MAXIMUM_WORST_WINDOW_SYMMETRIC_ERROR
    ):
        failures.append("worst_relative_window")
    return failures


def _valid_history_row(row: Mapping[str, str]) -> bool:
    try:
        objective = float(row.get("objective", "nan"))
    except ValueError:
        return False
    feasible = str(row.get("feasible", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    status = str(row.get("status", "")).strip().lower()
    return feasible and status == "ok" and math.isfinite(objective)


def _history_path(root: Path, run: str) -> Path:
    final = root / run / f"auto_calibration_{run}.csv"
    temporary = root / run / f"tmp_auto_calibration_{run}.csv"
    return final if final.is_file() else temporary


def _source_final_hashes(run: str) -> set[str]:
    hashes: set[str] = set()
    for _name, root in SOURCE_ROOTS:
        path = root / run / f"final_full_scale_{run}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        params = payload.get("params")
        if isinstance(params, Mapping) and params:
            hashes.add(_params_sha256(params))
    return hashes


def collect_candidates(run: str) -> list[Candidate]:
    """Read and de-duplicate compatible history rows for one run."""

    final_hashes = _source_final_hashes(run)
    by_hash: dict[str, Candidate] = {}
    for source_name, source_root in SOURCE_ROOTS:
        history = _history_path(source_root, run)
        if not history.is_file():
            raise FileNotFoundError(history)
        with history.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if not _valid_history_row(row):
                    continue
                params = _parse_mapping(row.get("params"))
                metrics = _parse_mapping(row.get("metrics"))
                if not params or not metrics:
                    continue
                try:
                    diagnostics = candidate_diagnostics(metrics)
                    history_iter = int(row["iter"])
                    objective = float(row["objective"])
                except (KeyError, TypeError, ValueError):
                    continue
                digest = _params_sha256(params)
                candidate = Candidate(
                    run=run,
                    source=source_name,
                    history_path=str(history),
                    history_iter=history_iter,
                    history_objective=objective,
                    params=params,
                    metrics=metrics,
                    params_sha256=digest,
                    diagnostics=diagnostics,
                    guardrail_failures=training_guardrail_failures(diagnostics),
                    source_final=digest in final_hashes,
                )
                previous = by_hash.get(digest)
                if previous is None or (
                    candidate.diagnostics["selection_score"],
                    candidate.history_objective,
                    candidate.source,
                    candidate.history_iter,
                ) < (
                    previous.diagnostics["selection_score"],
                    previous.history_objective,
                    previous.source,
                    previous.history_iter,
                ):
                    by_hash[digest] = candidate
                elif candidate.source_final:
                    previous.source_final = True
    return sorted(
        by_hash.values(),
        key=lambda item: (
            not item.passes_guardrails,
            item.diagnostics["selection_score"],
            item.history_objective,
            item.source,
            item.history_iter,
        ),
    )


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "schema": 1,
        "run": candidate.run,
        "objective_full_scale": candidate.history_objective,
        "metrics": candidate.metrics,
        "params": candidate.params,
        "source": "fixed_physics_reranking_candidate",
        "selection_candidate": {
            "created_utc": _utc_now(),
            "source_campaign": candidate.source,
            "history_path": candidate.history_path,
            "history_iter": candidate.history_iter,
            "history_objective": candidate.history_objective,
            "params_sha256": candidate.params_sha256,
            "source_final": candidate.source_final,
            "diagnostics": candidate.diagnostics,
            "guardrail_failures": candidate.guardrail_failures,
            "score_weights": SCORE_WEIGHTS,
            "training_guardrails": TRAINING_GUARDRAILS,
        },
    }


def rank_histories(output_root: Path, *, replay_top: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    ranking_rows: list[dict[str, Any]] = []
    manifest_runs: dict[str, Any] = {}
    for run in RUNS:
        candidates = collect_candidates(run)
        accepted = [candidate for candidate in candidates if candidate.passes_guardrails]
        if not accepted:
            raise RuntimeError(f"No {run} candidates pass the training guardrails.")
        for rank, candidate in enumerate(accepted, start=1):
            candidate.rank = rank

        replay_pool = accepted[: max(1, replay_top)]
        replay_hashes = {candidate.params_sha256 for candidate in replay_pool}
        replay_pool.extend(
            candidate
            for candidate in candidates
            if candidate.source_final and candidate.params_sha256 not in replay_hashes
        )
        candidate_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(replay_pool, start=1):
            path = (
                output_root
                / "replay_candidates"
                / run
                / f"candidate_{index:02d}_{candidate.params_sha256[:12]}.json"
            )
            _atomic_json(path, _candidate_payload(candidate))
            candidate.candidate_path = str(path)
            candidate_rows.append(
                {
                    "rank": candidate.rank,
                    "source": candidate.source,
                    "history_iter": candidate.history_iter,
                    "history_objective": candidate.history_objective,
                    "params_sha256": candidate.params_sha256,
                    "source_final": candidate.source_final,
                    "candidate_path": str(path),
                    "diagnostics": candidate.diagnostics,
                }
            )

        for candidate in candidates:
            ranking_rows.append(
                {
                    "run": run,
                    "rank": candidate.rank,
                    "passes_guardrails": candidate.passes_guardrails,
                    "guardrail_failures": ";".join(candidate.guardrail_failures),
                    "source": candidate.source,
                    "source_final": candidate.source_final,
                    "history_iter": candidate.history_iter,
                    "history_objective": candidate.history_objective,
                    "params_sha256": candidate.params_sha256,
                    **candidate.diagnostics,
                }
            )
        manifest_runs[run] = {
            "unique_candidate_count": len(candidates),
            "guardrail_pass_count": len(accepted),
            "replay_candidates": candidate_rows,
        }
        print(
            f"{run}: {len(accepted)}/{len(candidates)} candidates pass; "
            f"{len(candidate_rows)} queued for exact replay",
            flush=True,
        )

    ranking_path = output_root / "candidate_ranking.csv"
    fieldnames = list(ranking_rows[0])
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranking_rows)
    manifest = {
        "schema": 1,
        "created_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "source_variant": SOURCE_VARIANT,
        "source_roots": [
            {"name": name, "root": str(root)} for name, root in SOURCE_ROOTS
        ],
        "runs": manifest_runs,
        "score_weights": SCORE_WEIGHTS,
        "training_guardrails": TRAINING_GUARDRAILS,
        "holdout_guardrails": HOLDOUT_GUARDRAILS,
        "holdout_times_h": HOLDOUT_TIMES_H,
        "ranking_csv": str(ranking_path),
    }
    _atomic_json(output_root / "selection_manifest.json", manifest)
    return manifest


@contextmanager
def _selection_environment(run: str) -> Iterator[tuple[Any, Any]]:
    from darsia.fluidflower.tools import ac_clearwater_phase_pilot as pilot
    from darsia.fluidflower.tools import ac_mass_conserving_diffusion_screen as campaign

    variant = next(
        item for item in campaign.VARIANTS if item.name == SOURCE_VARIANT
    )
    environment = campaign._campaign_env(variant)
    target = OVERFLOW_TARGET_ROOT / f"{run}.json"
    if not target.is_file():
        raise FileNotFoundError(target)
    environment["FFAC_OVERFLOW_PHASE_TARGETS"] = str(target)
    with pilot.campaign_environment(variant, environment):
        yield campaign, variant


def _build_evaluation_context(
    run: str,
    variant: Any,
    *,
    times: Sequence[float] | None,
    allow_partial_evaluation_times: bool = False,
    quality_dtype: str = "float32",
) -> Any:
    from darsia.presets.workflows.rig import Rig
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import build_context, load_bounds_map

    return build_context(
        run=run,
        config_dir=project_path('config_seg6/run_ac'),
        rig_cls=Rig,
        ref_config_path=None,
        use_facies=bool(variant.use_facies),
        bounds_map=load_bounds_map(REPO_ROOT / variant.bounds_file),
        enforce_lower=False,
        per_label_params=True,
        use_label_weights=False,
        label_weights=None,
        quality_scale=1.0,
        quality_dtype=quality_dtype,
        objective_integral="off",
        static_light_correction="off",
        signal_parameterization=variant.signal_parameterization,
        phase_separation=variant.phase_separation,
        color_path_anchor="off",
        color_path_anchor_weight=0.75,
        color_path_anchor_strict=True,
        evaluation_times_hours=times,
        evaluation_time_tolerance_seconds=600.0,
        allow_partial_evaluation_times=allow_partial_evaluation_times,
    )


def _close_context(context: Any) -> None:
    for name in (
        "_cuda_evaluator",
        "_cuda_float32_evaluator",
        "_opencl_evaluator",
        "_opencl_float32_evaluator",
        "_metal_evaluator",
    ):
        evaluator = getattr(context, name, None)
        if evaluator is not None:
            try:
                evaluator.close()
            finally:
                setattr(context, name, None)
    del context
    gc.collect()


def _disable_holdout_objective_terms(context: Any) -> None:
    """Keep holdout mass fields independent of training-only penalties."""

    context.overflow_phase_penalty_weight = 0.0
    context.overflow_post_coverage_penalty_weight = 0.0
    context.overflow_phase_constraint = None
    context.aqueous_diffusion_penalty_weight = 0.0
    context.post_injection_balance_weight = 0.0
    context.late_gas_penalty_weight = 0.0
    context.spatial_objective_weight = 0.0
    context.layer_gain_regularization_weight = 0.0


def _metrics_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): asdict(value) if hasattr(value, "__dataclass_fields__") else vars(value)
        for key, value in metrics.items()
    }


def _mass_metric_delta(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> float:
    maximum = 0.0
    for key, expected_metric in expected.items():
        actual_metric = actual.get(key)
        if actual_metric is None:
            return float("inf")
        for name in ("injected_full", "total_full", "gaseous_full", "aqueous_full"):
            expected_value = _metric_value(expected_metric, name)
            actual_value = _metric_value(actual_metric, name)
            scale = max(abs(expected_value), 1e-12)
            maximum = max(maximum, abs(actual_value - expected_value) / scale)
    return maximum


def _evaluate_candidates(
    context: Any,
    candidates: Sequence[dict[str, Any]],
    *,
    backend: str,
    holdout: bool,
) -> list[dict[str, Any]]:
    from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (
        evaluate_run,
        prepare_evaluation_context,
    )

    prepare_evaluation_context(context, backend=backend, release_images=True)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        payload = json.loads(
            Path(candidate["candidate_path"]).read_text(encoding="utf-8-sig")
        )
        started = time.perf_counter()
        result = evaluate_run(context, dict(payload["params"]))
        elapsed = time.perf_counter() - started
        metrics = _metrics_payload(result.metrics)
        diagnostics = candidate_diagnostics(metrics) if metrics else {}
        history_delta = (
            None
            if holdout
            else _mass_metric_delta(payload.get("metrics") or {}, metrics)
        )
        objective_delta = (
            None
            if holdout
            else abs(
                float(result.objective)
                - float(payload.get("objective_full_scale", result.objective))
            )
        )
        rows.append(
            {
                **candidate,
                "status": result.status,
                "feasible": bool(result.feasible),
                "objective": float(result.objective),
                "objective_terms": dict(result.objective_terms),
                "metrics": metrics,
                "diagnostics": diagnostics,
                "guardrail_failures": (
                    holdout_guardrail_failures(diagnostics)
                    if holdout and diagnostics
                    else training_guardrail_failures(diagnostics)
                    if diagnostics
                    else ["missing_metrics"]
                ),
                "history_max_relative_mass_delta": history_delta,
                "history_objective_abs_delta": objective_delta,
                "evaluation_seconds": elapsed,
            }
        )
    return rows


def validate_candidates(
    output_root: Path,
    *,
    backend: str,
    finalist_count: int,
) -> dict[str, Any]:
    manifest_path = output_root / "selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run the rank action first; missing {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    selected_dir = output_root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}

    for run in RUNS:
        replay_candidates = list(manifest["runs"][run]["replay_candidates"])
        existing_report_path = output_root / "validation" / f"{run}.json"
        existing_report = (
            json.loads(existing_report_path.read_text(encoding="utf-8-sig"))
            if existing_report_path.is_file()
            else None
        )
        training_context = None
        holdout_context = None
        with _selection_environment(run) as (_campaign, variant):
            if existing_report is not None and existing_report.get(
                "training_replay"
            ):
                replay = list(existing_report["training_replay"])
                print(f"{run}: reusing completed exact replay", flush=True)
            else:
                try:
                    print(f"{run}: building exact-replay context", flush=True)
                    training_context = _build_evaluation_context(
                        run, variant, times=None
                    )
                    replay = _evaluate_candidates(
                        training_context,
                        replay_candidates,
                        backend=backend,
                        holdout=False,
                    )
                finally:
                    if training_context is not None:
                        _close_context(training_context)

            reproducible = [
                row
                for row in replay
                if row["status"] == "ok"
                and row["feasible"]
                and not row["guardrail_failures"]
                and float(row["history_max_relative_mass_delta"]) <= 5e-5
                and float(row["history_objective_abs_delta"])
                <= max(1e-10, 5e-5 * abs(float(row["objective"])))
            ]
            reproducible.sort(
                key=lambda row: (
                    row["diagnostics"]["selection_score"],
                    row["objective"],
                    row["params_sha256"],
                )
            )
            if not reproducible:
                raise RuntimeError(f"No reproducible finalists for {run}.")
            finalists = reproducible[: max(1, finalist_count)]

            try:
                print(f"{run}: building {len(HOLDOUT_TIMES_H)}-frame holdout", flush=True)
                holdout_context = _build_evaluation_context(
                    run, variant, times=HOLDOUT_TIMES_H
                )
                _disable_holdout_objective_terms(holdout_context)
                holdout = _evaluate_candidates(
                    holdout_context,
                    finalists,
                    backend=backend,
                    holdout=True,
                )
            finally:
                if holdout_context is not None:
                    _close_context(holdout_context)

        holdout_by_hash = {row["params_sha256"]: row for row in holdout}
        selected_training: dict[str, Any] | None = None
        selected_holdout: dict[str, Any] | None = None
        for training in finalists:
            candidate_holdout = holdout_by_hash[training["params_sha256"]]
            if (
                candidate_holdout["status"] == "ok"
                and candidate_holdout["feasible"]
                and not candidate_holdout["guardrail_failures"]
            ):
                selected_training = training
                selected_holdout = candidate_holdout
                break
        validation_passed = selected_training is not None
        if selected_training is None:
            selected_training = finalists[0]
            selected_holdout = holdout_by_hash[selected_training["params_sha256"]]

        source_payload = json.loads(
            Path(selected_training["candidate_path"]).read_text(
                encoding="utf-8-sig"
            )
        )
        destination = selected_dir / f"final_full_scale_{run}.json"
        final_payload = {
            "schema": 2,
            "run": run,
            "objective_full_scale": selected_training["objective"],
            "objective_terms": selected_training["objective_terms"],
            "metrics": selected_training["metrics"],
            "params": source_payload["params"],
            "source": "fixed_physics_reranking_with_exact_replay_and_holdout",
            "final_selection": {
                "created_utc": _utc_now(),
                "campaign_id": CAMPAIGN_ID,
                "validation_passed": validation_passed,
                "source_candidate": selected_training["candidate_path"],
                "params_sha256": selected_training["params_sha256"],
                "training_rank": selected_training.get("rank"),
                "training_diagnostics": selected_training["diagnostics"],
                "training_reproduction": {
                    "maximum_relative_mass_delta": selected_training[
                        "history_max_relative_mass_delta"
                    ],
                    "objective_absolute_delta": selected_training[
                        "history_objective_abs_delta"
                    ],
                },
                "holdout_times_h": HOLDOUT_TIMES_H,
                "holdout_diagnostics": selected_holdout["diagnostics"],
                "holdout_guardrail_failures": selected_holdout[
                    "guardrail_failures"
                ],
                "score_weights": SCORE_WEIGHTS,
                "training_guardrails": TRAINING_GUARDRAILS,
                "holdout_guardrails": HOLDOUT_GUARDRAILS,
            },
        }
        _atomic_json(destination, final_payload)
        report = {
            "run": run,
            "validation_passed": validation_passed,
            "selected": str(destination),
            "selected_params_sha256": selected_training["params_sha256"],
            "training_replay": replay,
            "holdout": holdout,
        }
        reports[run] = report
        _atomic_json(output_root / "validation" / f"{run}.json", report)
        print(
            f"{run}: selected training rank {selected_training.get('rank')} "
            f"holdout_pass={validation_passed} "
            f"holdout_MAE={selected_holdout['diagnostics']['normalized_mae']:.3%}",
            flush=True,
        )

    summary = {
        "schema": 1,
        "completed_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "backend": backend,
        "runs": reports,
        "all_holdouts_passed": all(
            report["validation_passed"] for report in reports.values()
        ),
        "selected_dir": str(selected_dir),
    }
    _atomic_json(output_root / "validation_summary.json", summary)
    return summary


def _final_variant() -> Any:
    from darsia.fluidflower.tools import ac_mass_conserving_diffusion_screen as campaign
    from darsia.fluidflower.tools.ac_phase_curve_campaign import Variant

    source = next(
        item for item in campaign.VARIANTS if item.name == SOURCE_VARIANT
    )
    return Variant(
        name=FINAL_VARIANT,
        titration=source.titration,
        optimize_aqueous_blend=source.optimize_aqueous_blend,
        bounds_file=source.bounds_file,
        phase_separation=source.phase_separation,
        use_facies=source.use_facies,
        signal_parameterization=source.signal_parameterization,
        aqueous_noise_gate=source.aqueous_noise_gate,
    )


def run_postflight(output_root: Path, *, backend: str) -> dict[str, Any]:
    from darsia.fluidflower.tools import ac_clearwater_phase_pilot as pilot
    from darsia.fluidflower.tools import ac_mass_conserving_diffusion_screen as campaign

    validation_path = output_root / "validation_summary.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if not validation.get("all_holdouts_passed"):
        raise RuntimeError("Postflight requires all four fixed holdout gates to pass.")

    variant = _final_variant()
    campaign_root = output_root / "postflight_input"
    postflight_root = output_root / "phase_postflight"
    summaries: list[dict[str, Any]] = []
    for run in RUNS:
        selected = output_root / "selected" / f"final_full_scale_{run}.json"
        if not selected.is_file():
            raise FileNotFoundError(selected)
        candidate = (
            campaign_root
            / variant.name
            / run
            / f"final_full_scale_{run}.json"
        )
        _atomic_json(
            candidate,
            json.loads(selected.read_text(encoding="utf-8-sig")),
        )
        summary_path = postflight_root / variant.name / run / "summary.json"
        movie_path = (
            postflight_root / variant.name / run / "calibration_review.mp4"
        )
        if summary_path.is_file() and movie_path.is_file():
            summaries.append(
                json.loads(summary_path.read_text(encoding="utf-8-sig"))
            )
            print(f"{run}: postflight already complete", flush=True)
            continue

        source_variant = next(
            item for item in campaign.VARIANTS if item.name == SOURCE_VARIANT
        )
        environment = campaign._campaign_env(source_variant)
        target = OVERFLOW_TARGET_ROOT / f"{run}.json"
        environment["FFAC_OVERFLOW_PHASE_TARGETS"] = str(target)
        print(f"{run}: rendering exact spatial postflight", flush=True)
        _frame, _chemistry, summary = pilot.analyse_candidate(
            run=run,
            variant=variant,
            campaign_root=campaign_root,
            output_root=postflight_root,
            raw_root=configured_tool_path('ffac', 'raw_inputs'),
            backend=backend,
            environment=environment,
            diagnostic_workers=6,
        )
        summaries.append(summary)

    aggregate_rows = [
        {
            "run": summary["run"],
            "variant": summary["variant"],
            "normalized_mae": summary["normalized_mae"],
            "ratio_near_4p1h": summary["ratio_near_4p1h"],
            "gas_fraction_near_4p1h": summary["gas_fraction_near_4p1h"],
            "late_mass_ratio": summary["late_mass_ratio"],
            "late_gas_fraction": summary["late_gas_fraction"],
            "median_late_aqueous_trace_small_component_fraction": summary[
                "median_late_aqueous_trace_small_component_fraction"
            ],
            "median_late_gas_core_small_component_fraction": summary[
                "median_late_gas_core_small_component_fraction"
            ],
            "calibration_review_movie": summary["calibration_review_movie"][
                "movie"
            ],
        }
        for summary in summaries
    ]
    csv_path = postflight_root / "final_candidate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    result = {
        "schema": 1,
        "completed_utc": _utc_now(),
        "variant": variant.name,
        "backend": backend,
        "rows": aggregate_rows,
        "summary_csv": str(csv_path),
    }
    _atomic_json(postflight_root / "POSTFLIGHT_COMPLETE.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("rank", "validate", "postflight", "all")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replay-top", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=4)
    parser.add_argument(
        "--backend", choices=("cuda", "opencl", "prepared"), default="cuda"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action in {"rank", "all"}:
        rank_histories(args.output_root, replay_top=args.replay_top)
    if args.action in {"validate", "all"}:
        summary = validate_candidates(
            args.output_root,
            backend=args.backend,
            finalist_count=args.finalists,
        )
        print(json.dumps(summary, indent=2, default=str))
    if args.action in {"postflight", "all"}:
        result = run_postflight(args.output_root, backend=args.backend)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
