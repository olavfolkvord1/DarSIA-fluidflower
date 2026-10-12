"""Run the selected physical calibration method for all production AC runs.

The campaign uses the final Titration model selected on AC20/24/27/60:
hierarchical facies shape plus layer gain, run-native colour paths, fixed AC14
depth, soft 0.5 cm expert support, overflow phase evidence, post-injection mass
balance, a 24 h decaying free-gas envelope, and diffusion weight 0.25.

Manually validated run-specific overflow traces are used for 34 runs. The
documented feed-segment structure is retained while effective rates are fitted
from each trace. AC50 uses its measured early trace plus an imputed tail, and
only five runs without usable scale data receive matched-cohort priors. Every
target records its measured or imputed origin and uncertainty.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import csv
import hashlib
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_mass_conserving_diffusion_screen as selected  # noqa: E402
from darsia.fluidflower.tools import ac_phase_curve_campaign as base  # noqa: E402
from darsia.fluidflower.tools.ac_production_campaign import FINAL_PRODUCTION_RUNS  # noqa: E402
from darsia.fluidflower.tools.overflow_phase_constraint import (  # noqa: E402
    build_hybrid_overflow_phase_targets,
    build_imputed_overflow_phase_targets,
    build_overflow_fraction_reference,
    build_overflow_phase_targets,
    load_overflow_phase_constraint,
    write_overflow_phase_targets,
)


CAMPAIGN_ID = "final_production_overflow_corrected_20260814_5000"
VARIANT_NAME = "final_titration_balance_gas24_diff_w0p25"
MAX_ITERS = 5000
WARMUP_ITERS = 150
OPTUNA_SEED = 17

DIRECT_OVERFLOW_WEIGHT = 5.0
HYBRID_OVERFLOW_WEIGHT = 3.5
IMPUTED_OVERFLOW_WEIGHT = 2.5
OVERFLOW_REFERENCE_SIGMA_FLOOR = 0.10
OVERFLOW_REFERENCE_MIN_DONORS = 3
EVALUATION_TIMES_H = (
    0.164,
    0.747,
    1.331,
    1.914,
    2.497,
    2.997,
    4.083,
    6.000,
    8.000,
    9.417,
)
MANUAL_SCALE_OVERRIDES_G = {
    "ac22": {1450: 4692.3, 1460: 4692.4},
    "ac53": {580: 2074.0},
    "ac60": {1031: 3717.7},
}
MANUAL_SCALE_EXCLUSIONS = {
    "ac45": {
        0: "pre-injection reading precedes an evident scale tare/reset",
    }
}

POST_INJECTION_BALANCE_WEIGHT = 1.0
POST_INJECTION_START_H = 4.083
DIFFUSION_WEIGHT = 0.25
DIFFUSION_START_H = 4.083
DIFFUSION_SUPPORT_LEVEL = 0.03
DIFFUSION_PAIR_MIN_DT_H = 1.0
LATE_GAS_START_H = 24.0
LATE_GAS_START_FRACTION = 0.05
LATE_GAS_DECAY_HALF_LIFE_H = 12.0
LATE_GAS_PENALTY_WEIGHT = 200.0
EXTRA_CALIBRATION_TIMES_H = (12.0,)
EXTRA_CALIBRATION_TOLERANCE_SECONDS = 900.0

DEFAULT_LOGS_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_QUEUE_ROOT = rf"\\Moderskipet\Darsia_Queue\Kalibrering_AC_{CAMPAIGN_ID}"
OVERFLOW_REVIEW_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'overflow_weight_review'
FINAL5_SEED_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'final5_20260729/selected_sharedpath'
VALIDATED_FINAL_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'final_method_selection_20260814/selected'

ALL_RUNS = tuple(str(run).lower() for run in FINAL_PRODUCTION_RUNS)
FULLY_IMPUTED_OVERFLOW_RUNS = ("ac17", "ac18", "ac23", "ac26", "ac32")
HYBRID_OVERFLOW_RUNS = ("ac50",)
MEASURED_OVERFLOW_RUNS = tuple(
    run
    for run in ALL_RUNS
    if run not in {*FULLY_IMPUTED_OVERFLOW_RUNS, *HYBRID_OVERFLOW_RUNS}
)
OVERFLOW_COHORT_DONORS = {
    "ac17": ("ac19", "ac20", "ac21"),
    "ac18": ("ac19", "ac20", "ac21"),
    "ac23": ("ac20", "ac21", "ac22", "ac24", "ac25"),
    "ac26": ("ac24", "ac25", "ac27", "ac28"),
    "ac32": ("ac30", "ac31", "ac33", "ac34"),
    "ac50": ("ac48", "ac49", "ac52", "ac53"),
}
TRACE_NOMINAL_RATE_G_PER_H = {
    run: (300.0 if run in {"ac20", "ac21"} else 400.0)
    for run in (*MEASURED_OVERFLOW_RUNS, *HYBRID_OVERFLOW_RUNS)
}
VERIFIED_FEED_SCHEDULE_RUNS = {"ac20", "ac24", "ac27", "ac60"}
TRACE_NOMINAL_RATE_SOURCE_UNVERIFIED = {"ac19", "ac42", "ac61"}
EXPECTED_TRACE_SCHEDULE_ANOMALIES = {"ac31"}
_MODERSKIPET_CONTROLS = ("ac20", "ac40", "ac52")
_OLAV_CONTROLS = ("ac24", "ac44", "ac60")
_REMAINING_RUNS = tuple(
    run
    for run in ALL_RUNS
    if run not in {*_MODERSKIPET_CONTROLS, *_OLAV_CONTROLS}
)


def _weighted_assignments() -> dict[str, tuple[str, ...]]:
    """Allocate 22/18 runs to match the observed CUDA/OpenCL throughput."""

    moderskipet = list(_MODERSKIPET_CONTROLS)
    olav = list(_OLAV_CONTROLS)
    for run in _REMAINING_RUNS:
        if len(moderskipet) / 22.0 <= len(olav) / 18.0:
            moderskipet.append(run)
        else:
            olav.append(run)
    assignments = {
        "moderskipet": tuple(moderskipet),
        "olav": tuple(olav),
    }
    assigned = [run for values in assignments.values() for run in values]
    if (
        len(assignments["moderskipet"]) != 22
        or len(assignments["olav"]) != 18
        or len(assigned) != len(ALL_RUNS)
        or set(assigned) != set(ALL_RUNS)
    ):
        raise RuntimeError(f"Invalid final production assignment: {assignments}")
    return assignments


RUN_ASSIGNMENTS = _weighted_assignments()
RUNS = tuple(run for values in RUN_ASSIGNMENTS.values() for run in values)


VARIANT = base.Variant(
    name=VARIANT_NAME,
    titration=True,
    optimize_aqueous_blend=True,
    bounds_file="config/bounds_seg6_phase_curve_titration_hierarchical.json",
    phase_separation="shared-signal",
    use_facies=False,
    signal_parameterization="facies-shape-layer-gain",
    aqueous_noise_gate="off",
)
VARIANTS = (VARIANT,)

_ACTIVE_TARGET_PATHS: dict[str, Path] = {
    run: DEFAULT_LOGS_ROOT / "overflow_targets" / f"{run}.json" for run in RUNS
}
_ACTIVE_TARGET_METADATA: dict[str, dict[str, Any]] = {}

_ORIGINAL_RUN_PREFLIGHT = base.run_preflight
_ORIGINAL_MASTER_COMMAND = base._master_command


def _params_sha256(params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(params), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_seed(raw_params: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in raw_params.items()
        if not str(name).startswith("depth.")
        and not str(name).startswith("gas.hybrid_")
        and str(name) not in {"gas.clear_onset", "gas.clear_width"}
    }


def _build_seed_bundles(logs_root: Path) -> dict[str, Path]:
    params_by_run: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for run in RUNS:
        candidates: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        validated = VALIDATED_FINAL_ROOT / f"final_full_scale_{run}.json"
        if validated.is_file():
            payload = json.loads(validated.read_text(encoding="utf-8-sig"))
            params = _clean_seed(payload.get("params") or {})
            if not params:
                raise RuntimeError(f"Validated seed has no parameters: {validated}")
            digest = _params_sha256(params)
            seen.add(digest)
            candidates.append(
                {
                    "params": params,
                    "source": str(validated),
                    "source_family": "validated-final-method",
                    "params_sha256": digest,
                }
            )
            source_rows.append(
                {
                    "family": "validated-final-method",
                    "path": str(validated),
                    "sha256": base._sha256(validated),
                }
            )

        source = FINAL5_SEED_ROOT / f"final_full_scale_{run}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Production seed unavailable: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        raw_params = payload.get("params")
        if not isinstance(raw_params, dict) or not raw_params:
            raise RuntimeError(f"Production seed has no parameters: {source}")
        for blend in (0.0, 0.2, 0.4):
            params = base._convert_seed(
                raw_params,
                variant=VARIANT,
                aqueous_blend=blend,
            )
            digest = _params_sha256(params)
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append(
                {
                    "params": params,
                    "source": str(source),
                    "source_family": "final5-selected-converted-hierarchical",
                    "seed_aqueous_blend": blend,
                    "params_sha256": digest,
                }
            )
        source_rows.append(
            {
                "family": "final5-selected-converted-hierarchical",
                "path": str(source),
                "sha256": base._sha256(source),
                "source_objective_full_scale": payload.get("objective_full_scale"),
            }
        )
        params_by_run[run] = candidates
        sources[run] = source_rows

    path = logs_root / f"seed_params_{VARIANT_NAME}_{socket.gethostname().upper()}.json"
    base._atomic_json(
        path,
        {
            "metadata": {
                "campaign_id": CAMPAIGN_ID,
                "variant": VARIANT_NAME,
                "fixed_depth": True,
                "sources": sources,
            },
            "params_by_run": params_by_run,
        },
    )
    return {VARIANT_NAME: path}


def _campaign_env(variant: base.Variant) -> dict[str, str]:
    env = selected._base_campaign_env(variant)
    env.update(
        {
            "FFAC_EXPERT_CONSTRAINT_MODE": "soft",
            "FFAC_EXPERT_CONSTRAINT_EXPANSION_M": "0.005",
            "FFAC_DEPTH_PARAMETERIZATION": "fixed",
            "FFAC_DEPTH_PRIOR_WEIGHT": "0",
            "FFAC_POST_INJECTION_BALANCE_WEIGHT": str(
                POST_INJECTION_BALANCE_WEIGHT
            ),
            "FFAC_POST_INJECTION_BALANCE_START_H": str(POST_INJECTION_START_H),
            "FFAC_AQUEOUS_DIFFUSION_PENALTY_WEIGHT": str(DIFFUSION_WEIGHT),
            "FFAC_AQUEOUS_DIFFUSION_START_H": str(DIFFUSION_START_H),
            "FFAC_AQUEOUS_DIFFUSION_SUPPORT_LEVEL": str(
                DIFFUSION_SUPPORT_LEVEL
            ),
            "FFAC_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H": str(
                DIFFUSION_PAIR_MIN_DT_H
            ),
            "FFAC_LATE_GAS_START_H": str(LATE_GAS_START_H),
            "FFAC_LATE_GAS_MAX_FRACTION": str(LATE_GAS_START_FRACTION),
            "FFAC_LATE_GAS_DECAY_HALF_LIFE_H": str(
                LATE_GAS_DECAY_HALF_LIFE_H
            ),
            "FFAC_LATE_GAS_FLOOR_FRACTION": "0",
            "FFAC_LATE_GAS_PENALTY_WEIGHT": str(LATE_GAS_PENALTY_WEIGHT),
            "FFAC_LATE_GAS_CONSTRAINT_MODE": "gas-only",
            "FFAC_EXTRA_CALIBRATION_TIMES_H": ",".join(
                str(value) for value in EXTRA_CALIBRATION_TIMES_H
            ),
            "FFAC_EXTRA_CALIBRATION_TIME_TOLERANCE_SECONDS": str(
                EXTRA_CALIBRATION_TOLERANCE_SECONDS
            ),
            "FFAC_STATIC_LIGHT_CORRECTION": "off",
            "FFAC_AQUEOUS_NOISE_GATE": "off",
            "FFAC_COLORCHECKER_REFINEMENT": "off",
            "FFAC_COLORCHECKER_AUTO_ORIENT": "off",
            "FFAC_COLORCHECKER_STRATEGY": "neutral-tone-affine",
        }
    )
    return env


def _documented_feed_segments(run: str) -> tuple[dict[str, float], ...]:
    rate = float(TRACE_NOMINAL_RATE_G_PER_H[run])
    if run == "ac46":
        # Protocol: 400 mL/h from 10:30, changed to 150 mL/h at 19:00.
        # The calibration time origin is 10:41:10.
        return (
            {"start_h": 0.0, "nominal_rate_g_per_h": 400.0},
            {
                "start_h": 8.0 + 18.0 / 60.0 + 50.0 / 3600.0,
                "nominal_rate_g_per_h": 150.0,
            },
        )
    return ({"start_h": 0.0, "nominal_rate_g_per_h": rate},)


def _feed_schedule_provenance(run: str) -> dict[str, Any]:
    segments = [dict(segment) for segment in _documented_feed_segments(run)]
    if run == "ac60":
        source = project_path('config/overflow_scale_runs.json')
        return {
            "source_kind": "verified experiment configuration",
            "source_path": str(source),
            "source_sha256": base._sha256(source),
            "documented_feed_segments": segments,
        }

    raw_run_root = configured_tool_path('ffac', 'raw_inputs') / run.upper()
    workbooks = sorted(
        (
            path
            for path in raw_run_root.glob("*.xlsx")
            if not path.name.startswith("~$") and "protocol" in path.name.lower()
        ),
        key=lambda path: ("fromminidisc" in path.name.lower(), len(path.name)),
    )
    workbook = workbooks[0] if workbooks else None
    if run in TRACE_NOMINAL_RATE_SOURCE_UNVERIFIED:
        provenance: dict[str, Any] = {
            "source_kind": "cohort nominal structure plus validated run trace",
            "source_note": (
                "The workbook does not independently state a complete nominal "
                "rate schedule; the 400 g/h structure follows adjacent AC runs, "
                "while the effective rate is estimated from this run's manually "
                "validated cumulative-weight trace."
            ),
            "documented_feed_segments": segments,
        }
    else:
        provenance = {
            "source_kind": "experiment protocol workbook",
            "source_note": (
                "Nominal segment structure transcribed from the experiment "
                "protocol; effective rates are estimated from validated weights."
            ),
            "documented_feed_segments": segments,
        }
    if workbook is not None:
        provenance.update(
            {
                "source_path": str(workbook),
                "source_sha256": base._sha256(workbook),
            }
        )
    return provenance


def _build_measured_overflow_payload(
    run: str,
    *,
    evaluation_times_h: Sequence[float] = EVALUATION_TIMES_H,
    effective_feed_rate_overrides_g_h: Sequence[float] | None = None,
    effective_feed_rate_override_source: str | None = None,
) -> dict[str, Any]:
    review = OVERFLOW_REVIEW_ROOT / run / f"{run}_overflow_scale_review.json"
    protocol = project_path('protocols') / run / "injection_protocol.csv"
    provenance = _feed_schedule_provenance(run)
    kwargs: dict[str, Any] = {
        "review_state_path": review,
        "injection_protocol_path": protocol,
        "evaluation_times_h": evaluation_times_h,
        "overrides_by_index": MANUAL_SCALE_OVERRIDES_G.get(run),
        "excluded_indices": tuple(MANUAL_SCALE_EXCLUSIONS.get(run, {})),
        "feed_schedule_provenance": provenance,
    }
    if run in VERIFIED_FEED_SCHEDULE_RUNS and effective_feed_rate_overrides_g_h is None:
        kwargs["feed_schedule_mode"] = "verified"
    else:
        kwargs.update(
            {
                "feed_schedule_mode": "trace-derived",
                "documented_feed_segments": _documented_feed_segments(run),
                "effective_feed_rate_overrides_g_h": (
                    effective_feed_rate_overrides_g_h
                ),
                "effective_feed_rate_override_source": (
                    effective_feed_rate_override_source
                ),
            }
        )
    payload = build_overflow_phase_targets(**kwargs)
    payload["source"]["assignment_mode"] = "measured-run-specific"
    payload["source"]["manual_exclusions"] = [
        {"index": int(index), "reason": reason}
        for index, reason in MANUAL_SCALE_EXCLUSIONS.get(run, {}).items()
    ]
    payload["source"]["review_state_path"] = str(review)
    payload["source"]["injection_protocol_path"] = (
        f"protocols/{run}/injection_protocol.csv"
    )
    return payload


def _overflow_fallback_reason(run: str) -> str:
    review = OVERFLOW_REVIEW_ROOT / run / f"{run}_overflow_scale_review.json"
    if not review.is_file():
        return "no run-specific overflow review file"
    try:
        payload = json.loads(review.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"overflow review could not be read: {exc}"
    metadata = payload.get("metadata") or {}
    if metadata.get("weight_data_valid") is False:
        return "run-specific scale images rejected: " + str(
            metadata.get("weight_data_reason") or "unspecified reason"
        )
    confirmed = sum(
        1
        for record in payload.get("records") or []
        if record.get("status") in {"Verified", "Corrected"}
        and record.get("weight_g") is not None
    )
    return f"only {confirmed} usable manually confirmed scale readings"


def _write_assignment_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    fields = (
        "run",
        "assignment_mode",
        "penalty_weight",
        "target_count",
        "reason",
        "donor_runs",
        "measured_target_times_h",
        "imputed_target_times_h",
        "feed_schedule_anomaly",
        "target_path",
        "target_sha256",
    )
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_target_payloads(
    payloads: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_times = [round(value, 6) for value in EVALUATION_TIMES_H]
    mode_counts = {
        "measured-run-specific": 0,
        "hybrid-run-specific-plus-imputed-tail": 0,
        "imputed-cohort-reference": 0,
    }
    rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for run in ALL_RUNS:
        payload = payloads.get(run)
        info = metadata.get(run)
        if payload is None or info is None:
            errors.append(f"{run}: missing payload or assignment metadata")
            continue
        mode = str(info["assignment_mode"])
        if mode not in mode_counts:
            errors.append(f"{run}: unsupported assignment mode {mode!r}")
            continue
        mode_counts[mode] += 1
        targets = list(payload.get("targets") or [])
        times = [round(float(target["time_h"]), 6) for target in targets]
        if times != expected_times:
            errors.append(f"{run}: target times differ from campaign contract")
        gas_fractions: list[float] = []
        relative_sigmas: list[float] = []
        for target in targets:
            values = (
                float(target["time_h"]),
                float(target["injected_mass_kg"]),
                float(target["gas_mass_kg"]),
                float(target["aqueous_mass_kg"]),
                float(target["phase_mass_sigma_kg"]),
            )
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{run}: non-finite target value")
                break
            _, injected, gas, aqueous, sigma = values
            if injected <= 0.0 or gas < 0.0 or gas > injected + 1e-12:
                errors.append(f"{run}: gas target outside injected mass")
                break
            if aqueous < 0.0 or sigma < 0.0:
                errors.append(f"{run}: negative aqueous mass or uncertainty")
                break
            gas_fractions.append(gas / injected)
            relative_sigmas.append(sigma / injected)
        late_fractions = [
            fraction
            for time_h, fraction in zip(times, gas_fractions, strict=True)
            if time_h >= POST_INJECTION_START_H - 1e-6
        ]
        if any(
            right > left + 1e-9
            for left, right in zip(late_fractions, late_fractions[1:])
        ):
            errors.append(f"{run}: overflow gas fraction rises after 4.083 h")
        measured_count = sum(
            str(target.get("target_source", "")).startswith("measured")
            for target in targets
        )
        imputed_count = sum(
            str(target.get("target_source", "")).startswith("imputed")
            for target in targets
        )
        expected_source_counts = {
            "measured-run-specific": (10, 0),
            "hybrid-run-specific-plus-imputed-tail": (7, 3),
            "imputed-cohort-reference": (0, 10),
        }[mode]
        if (measured_count, imputed_count) != expected_source_counts:
            errors.append(
                f"{run}: target source counts {(measured_count, imputed_count)} "
                f"!= {expected_source_counts}"
            )
        rows[run] = {
            "assignment_mode": mode,
            "target_count": len(targets),
            "measured_target_count": measured_count,
            "imputed_target_count": imputed_count,
            "max_gas_fraction": max(gas_fractions, default=0.0),
            "final_gas_fraction": gas_fractions[-1] if gas_fractions else None,
            "max_relative_sigma": max(relative_sigmas, default=0.0),
            "feed_schedule_anomaly": bool(info["feed_schedule_anomaly"]),
        }

    expected_mode_counts = {
        "measured-run-specific": 34,
        "hybrid-run-specific-plus-imputed-tail": 1,
        "imputed-cohort-reference": 5,
    }
    if mode_counts != expected_mode_counts:
        errors.append(
            f"assignment counts {mode_counts} != {expected_mode_counts}"
        )
    anomalies = {
        run for run, row in rows.items() if row["feed_schedule_anomaly"]
    }
    if anomalies != EXPECTED_TRACE_SCHEDULE_ANOMALIES:
        errors.append(
            "feed schedule anomalies changed: "
            f"{sorted(anomalies)} != {sorted(EXPECTED_TRACE_SCHEDULE_ANOMALIES)}"
        )
    result = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "passed" if not errors else "failed",
        "expected_target_times_h": list(EVALUATION_TIMES_H),
        "mode_counts": mode_counts,
        "expected_feed_schedule_anomalies": sorted(
            EXPECTED_TRACE_SCHEDULE_ANOMALIES
        ),
        "runs": rows,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Overflow target validation failed: " + "; ".join(errors))
    return result


def _write_target_artifacts(logs_root: Path) -> dict[str, dict[str, Any]]:
    global _ACTIVE_TARGET_PATHS, _ACTIVE_TARGET_METADATA
    target_root = logs_root / "overflow_targets"
    target_root.mkdir(parents=True, exist_ok=True)

    measured_payloads = {
        run: _build_measured_overflow_payload(run)
        for run in MEASURED_OVERFLOW_RUNS
    }
    partial_payloads: dict[str, dict[str, Any]] = {}
    for run in HYBRID_OVERFLOW_RUNS:
        donor_rates = [
            float(
                measured_payloads[donor]["physics"]["effective_feed_segments"][0][
                    "effective_rate_g_per_h"
                ]
            )
            for donor in OVERFLOW_COHORT_DONORS[run]
        ]
        ordered_rates = sorted(donor_rates)
        midpoint = len(ordered_rates) // 2
        donor_rate = (
            ordered_rates[midpoint]
            if len(ordered_rates) % 2
            else (ordered_rates[midpoint - 1] + ordered_rates[midpoint]) / 2.0
        )
        partial_payloads[run] = _build_measured_overflow_payload(
            run,
            effective_feed_rate_overrides_g_h=(donor_rate,),
            effective_feed_rate_override_source=(
                "median effective rate from matched donors "
                + ",".join(OVERFLOW_COHORT_DONORS[run])
            ),
        )

    references: dict[str, dict[str, Any]] = {}
    reference_paths: dict[str, Path] = {}
    for run in (*FULLY_IMPUTED_OVERFLOW_RUNS, *HYBRID_OVERFLOW_RUNS):
        donors = OVERFLOW_COHORT_DONORS[run]
        reference = build_overflow_fraction_reference(
            {donor: measured_payloads[donor] for donor in donors},
            minimum_donors=OVERFLOW_REFERENCE_MIN_DONORS,
            relative_sigma_floor=OVERFLOW_REFERENCE_SIGMA_FLOOR,
        )
        reference["reference_for_runs"] = [run]
        reference["cohort_selection"] = (
            "nearest valid experiments in the same protocol/chronological block"
        )
        reference_path = target_root / f"{run}_cohort_reference.json"
        write_overflow_phase_targets(reference_path, reference)
        references[run] = reference
        reference_paths[run] = reference_path

    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    metadata: dict[str, dict[str, Any]] = {}
    assignment_rows: list[dict[str, Any]] = []
    for run in RUNS:
        if run in measured_payloads:
            payload = measured_payloads[run]
            mode = "measured-run-specific"
            penalty_weight = DIRECT_OVERFLOW_WEIGHT
            reason = (
                "validated run-specific scale trace with protocol-constrained, "
                "trace-derived effective feed rate"
            )
            donor_runs = [run]
        elif run in partial_payloads:
            reason = (
                "39 confirmed readings cover the run through 4.83 h; later "
                "images are unreadable because of reflections"
            )
            imputed_tail = build_imputed_overflow_phase_targets(
                run=run,
                injection_protocol_path=(
                    project_path('protocols') / run / "injection_protocol.csv"
                ),
                reference=references[run],
                reason=reason,
            )
            payload = build_hybrid_overflow_phase_targets(
                run=run,
                measured_payload=partial_payloads[run],
                imputed_payload=imputed_tail,
                reason=reason,
            )
            mode = "hybrid-run-specific-plus-imputed-tail"
            penalty_weight = HYBRID_OVERFLOW_WEIGHT
            donor_runs = list(references[run]["donor_runs"])
        else:
            reason = _overflow_fallback_reason(run)
            payload = build_imputed_overflow_phase_targets(
                run=run,
                injection_protocol_path=(
                    project_path('protocols') / run / "injection_protocol.csv"
                ),
                reference=references[run],
                reason=reason,
            )
            mode = "imputed-cohort-reference"
            payload["source"]["assignment_mode"] = mode
            for target in payload["targets"]:
                target["target_source"] = mode
            penalty_weight = IMPUTED_OVERFLOW_WEIGHT
            donor_runs = list(references[run]["donor_runs"])
        path = target_root / f"{run}.json"
        write_overflow_phase_targets(path, payload)
        constraint = load_overflow_phase_constraint(path, run=run)
        payloads[run] = payload
        paths[run] = path
        measured_target_times = [
            float(target["time_h"])
            for target in payload["targets"]
            if str(target.get("target_source", "")).startswith("measured")
        ]
        imputed_target_times = [
            float(target["time_h"])
            for target in payload["targets"]
            if str(target.get("target_source", "")).startswith("imputed")
        ]
        measured_physics = (
            payload.get("physics", {}).get("measured", {})
            if mode.startswith("hybrid")
            else payload.get("physics", {})
        )
        feed_schedule_anomaly = any(
            bool(segment.get("schedule_anomaly"))
            for segment in measured_physics.get("effective_feed_segments", [])
        )
        metadata[run] = {
            "assignment_mode": mode,
            "penalty_weight": penalty_weight,
            "reason": reason,
            "target_count": len(payload["targets"]),
            "target_path": str(path),
            "target_sha256": constraint.sha256,
            "donor_runs": donor_runs,
            "measured_target_times_h": measured_target_times,
            "imputed_target_times_h": imputed_target_times,
            "feed_schedule_anomaly": feed_schedule_anomaly,
        }
        assignment_rows.append(
            {
                "run": run,
                "assignment_mode": mode,
                "penalty_weight": penalty_weight,
                "target_count": len(payload["targets"]),
                "reason": reason,
                "donor_runs": ",".join(metadata[run]["donor_runs"]),
                "measured_target_times_h": ",".join(
                    f"{value:g}" for value in measured_target_times
                ),
                "imputed_target_times_h": ",".join(
                    f"{value:g}" for value in imputed_target_times
                ),
                "feed_schedule_anomaly": feed_schedule_anomaly,
                "target_path": str(path),
                "target_sha256": constraint.sha256,
            }
        )

    _ACTIVE_TARGET_PATHS = paths
    _ACTIVE_TARGET_METADATA = metadata
    _write_assignment_csv(logs_root / "overflow_assignment.csv", assignment_rows)
    validation = _validate_target_payloads(payloads, metadata)
    validation_path = logs_root / "overflow_validation.json"
    base._atomic_json(validation_path, validation)
    base._atomic_json(
        logs_root / "overflow_assignment.json",
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "cohort_references": {
                run: {
                    "path": str(path),
                    "sha256": base._sha256(path),
                    "donor_runs": list(references[run]["donor_runs"]),
                }
                for run, path in reference_paths.items()
            },
            "measured_runs": list(MEASURED_OVERFLOW_RUNS),
            "hybrid_runs": list(HYBRID_OVERFLOW_RUNS),
            "fully_imputed_runs": list(FULLY_IMPUTED_OVERFLOW_RUNS),
            "validation_path": str(validation_path),
            "validation_sha256": base._sha256(validation_path),
            "assignments": metadata,
        },
    )
    return payloads


def _master_command(**kwargs: Any) -> tuple[str, ...]:
    return tuple(_ORIGINAL_MASTER_COMMAND(**kwargs))


def _build_specs(
    *,
    role: str,
    logs_root: Path,
    queue_root: str,
    seed_paths: dict[str, Path],
) -> tuple[base.ProcessSpec, ...]:
    backend = "cuda" if role == "moderskipet" else "opencl"
    local_logs = project_path('logs') / CAMPAIGN_ID / role
    specs: list[base.ProcessSpec] = []
    for run in base._candidate_runs(role):
        env = _campaign_env(VARIANT)
        env["FFAC_OVERFLOW_PHASE_TARGETS"] = str(_ACTIVE_TARGET_PATHS[run])
        env["FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT"] = str(
            _ACTIVE_TARGET_METADATA[run]["penalty_weight"]
        )
        env["FFAC_OVERFLOW_ASSIGNMENT_MODE"] = str(
            _ACTIVE_TARGET_METADATA[run]["assignment_mode"]
        )
        job_id = f"{VARIANT_NAME}_{run}"
        queue = f"{queue_root}_{job_id}"
        specs.append(
            base.ProcessSpec(
                job_id=job_id,
                variant=VARIANT_NAME,
                run=run,
                queue=queue,
                command=_master_command(
                    variant=VARIANT,
                    run=run,
                    backend=backend,
                    queue=queue,
                    logs_root=logs_root,
                    seed_path=seed_paths[VARIANT_NAME],
                ),
                stdout=str(local_logs / f"{job_id}.log"),
                env=env,
            )
        )
    return tuple(specs)


def _run_preflight(
    logs_root: Path,
    role: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    target_payloads = _write_target_artifacts(logs_root)
    payload, seed_paths = _ORIGINAL_RUN_PREFLIGHT(logs_root, role)
    config_paths = (
        project_path('config_seg6/common.toml'),
        project_path('config_seg6/coloron.toml'),
        *(
            project_path('config_seg6/run_ac') / f"{run}.toml"
            for run in base._candidate_runs(role)
        ),
    )
    for path in config_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Campaign configuration unavailable: {path}")
        payload["source_hashes"][str(path.relative_to(REPO_ROOT))] = (
            base._sha256(path)
        )
    payload["overflow_phase_constraints"] = {
        run: {
            **_ACTIVE_TARGET_METADATA[run],
            "target_count": len(target_payloads[run]["targets"]),
        }
        for run in base._candidate_runs(role)
    }
    base._atomic_json(
        logs_root / f"preflight_{socket.gethostname().upper()}.json", payload
    )
    return payload, seed_paths


def validate_plan() -> dict[str, Any]:
    assigned = [run for values in RUN_ASSIGNMENTS.values() for run in values]
    if len(assigned) != 40 or len(set(assigned)) != 40:
        raise RuntimeError(f"Expected 40 unique production runs, got {assigned}")
    if set(assigned) != set(ALL_RUNS):
        raise RuntimeError("Production run assignment differs from canonical list")
    overflow_assigned = {
        *MEASURED_OVERFLOW_RUNS,
        *HYBRID_OVERFLOW_RUNS,
        *FULLY_IMPUTED_OVERFLOW_RUNS,
    }
    if overflow_assigned != set(ALL_RUNS):
        raise RuntimeError("Overflow assignment differs from canonical run list")
    if (
        len(MEASURED_OVERFLOW_RUNS) != 34
        or len(HYBRID_OVERFLOW_RUNS) != 1
        or len(FULLY_IMPUTED_OVERFLOW_RUNS) != 5
    ):
        raise RuntimeError("Expected overflow assignment counts 34/1/5")
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "variant": VARIANT_NAME,
        "runs": list(ALL_RUNS),
        "excluded_runs": {
            "ac29": "physical/chemical initial-state outlier",
            "ac51": "physical/chemical initial-state outlier",
        },
        "run_assignments": {
            role: list(runs) for role, runs in RUN_ASSIGNMENTS.items()
        },
        "calibrations": len(ALL_RUNS),
        "optuna_trials": len(ALL_RUNS) * MAX_ITERS,
        "structured_random_warmups": len(ALL_RUNS) * WARMUP_ITERS,
        "max_iters_per_calibration": MAX_ITERS,
        "warmup_iters_per_calibration": WARMUP_ITERS,
        "recommended_max_parallel": {"moderskipet": 3, "olav": 3},
        "scheduling": {
            "mode": "dynamic shared run pool",
            "preferred_assignments": {
                role: list(runs) for role, runs in RUN_ASSIGNMENTS.items()
            },
            "steal_unstarted_runs": True,
            "migrate_active_runs": False,
        },
        "method": {
            "aqueous_model": "run-specific hierarchical TitrationFlash",
            "phase_transition": "coupled and non-overlapping",
            "color_paths": "run-native; no AC60 color anchor",
            "depth": "fixed measured AC14 per-pixel map",
            "template_registration": "AC14 partial affine",
            "expert_support": "soft 0.5 cm",
            "post_injection_balance_weight": POST_INJECTION_BALANCE_WEIGHT,
            "diffusion_weight": DIFFUSION_WEIGHT,
            "late_gas_start_h": LATE_GAS_START_H,
            "late_gas_start_fraction": LATE_GAS_START_FRACTION,
            "late_gas_decay_half_life_h": LATE_GAS_DECAY_HALF_LIFE_H,
            "calibration_frames": 17,
            "selection_after_completion": (
                "fixed physical re-ranking followed by 17 untouched holdout times"
            ),
        },
        "overflow": {
            "measured_runs": list(MEASURED_OVERFLOW_RUNS),
            "measured_count": len(MEASURED_OVERFLOW_RUNS),
            "hybrid_runs": list(HYBRID_OVERFLOW_RUNS),
            "hybrid_count": len(HYBRID_OVERFLOW_RUNS),
            "fully_imputed_runs": list(FULLY_IMPUTED_OVERFLOW_RUNS),
            "imputed_count": len(FULLY_IMPUTED_OVERFLOW_RUNS),
            "cohort_donors": {
                run: list(donors)
                for run, donors in OVERFLOW_COHORT_DONORS.items()
            },
            "reference_center": "matched-cohort median measured gas fraction",
            "reference_sigma_floor_fraction": OVERFLOW_REFERENCE_SIGMA_FLOOR,
            "direct_penalty_weight": DIRECT_OVERFLOW_WEIGHT,
            "hybrid_penalty_weight": HYBRID_OVERFLOW_WEIGHT,
            "imputed_penalty_weight": IMPUTED_OVERFLOW_WEIGHT,
            "provenance_files": (
                "overflow_assignment.csv and one immutable target JSON per run"
            ),
        },
    }


def _configure_base() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.RUN_ASSIGNMENTS = RUN_ASSIGNMENTS
    base.RUNS = RUNS
    base.DYNAMIC_RUN_POOL = True
    base.MAX_ITERS = MAX_ITERS
    base.WARMUP_ITERS = WARMUP_ITERS
    base.OPTUNA_SEED = OPTUNA_SEED
    base.DEFAULT_LOGS_ROOT = DEFAULT_LOGS_ROOT
    base.DEFAULT_QUEUE_ROOT = DEFAULT_QUEUE_ROOT
    base.VARIANTS = VARIANTS
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        SCRIPT_DIR / "ac_mass_conserving_diffusion_screen.py",
        SCRIPT_DIR / "overflow_phase_constraint.py",
        SCRIPT_DIR / "temporal_gas_support.py",
        SCRIPT_DIR / "temporal_aqueous_support.py",
        SCRIPT_DIR / "expert_constraint_support.py",
        SCRIPT_DIR / "ac_locked_titration_flash.py",
        SCRIPT_DIR / "ac_production_campaign.py",
        SCRIPT_DIR / "verify_ac_depth_maps.py",
        project_path('config/bounds_seg6_phase_curve_titration_hierarchical.json'),
        project_path('external/darsia/src/darsia/corrections/color/colorcorrection.py'),
        project_path('external/darsia/src/darsia/multiphase/flash.py'),
        project_path('external/darsia/src/darsia/signals/models/color_path_interpolation.py'),
    )
    base.PROCESS_MARKERS = (Path(__file__).name, CAMPAIGN_ID)
    base.EXPECTED_COLOR_STATE = "on"
    base._build_seed_bundles = _build_seed_bundles
    base._campaign_env = _campaign_env
    base._master_command = _master_command
    base.validate_plan = validate_plan
    base.run_preflight = _run_preflight
    base.build_specs = _build_specs


def main(argv: Sequence[str] | None = None) -> None:
    _configure_base()
    base.main(argv)


if __name__ == "__main__":
    main()
