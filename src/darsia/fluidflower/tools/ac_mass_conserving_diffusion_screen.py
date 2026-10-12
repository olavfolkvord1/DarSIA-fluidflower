"""Screen mass-conserving aqueous redistribution around the night winner.

The control reproduces the historical late-phase closure. Six candidates use
the same run-native hierarchical TitrationFlash model, fixed AC14 depth map,
overflow evidence, and soft 0.5 cm expert support. They form a 2 x 3 screen of
late-gas onset (12 or 24 h) and mass-normalized diffusion weight (0, 0.25, 1).
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import hashlib
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools import ac_phase_curve_campaign as base  # noqa: E402
from darsia.fluidflower.tools.overflow_phase_constraint import (  # noqa: E402
    build_overflow_phase_targets,
    load_overflow_phase_constraint,
    write_overflow_phase_targets,
)


CAMPAIGN_ID = "mass_conserving_diffusion_screen_20260813_5000"
MAX_ITERS = 5000
WARMUP_ITERS = 150
OVERFLOW_PHASE_PENALTY_WEIGHT = 5.0
POST_INJECTION_BALANCE_WEIGHT = 1.0
POST_INJECTION_START_H = 4.083
DIFFUSION_START_H = 4.083
DIFFUSION_SUPPORT_LEVEL = 0.03
DIFFUSION_PAIR_MIN_DT_H = 1.0
EXTRA_CALIBRATION_TIMES_H = (12.0,)
EXTRA_CALIBRATION_TOLERANCE_SECONDS = 900.0
LATE_GAS_DECAY_HALF_LIFE_H = 12.0
LATE_GAS_PENALTY_WEIGHT = 200.0
DIFFUSION_WEIGHTS = (0.0, 0.25, 1.0)
LATE_GAS_SETTINGS = {
    12.0: 0.10,
    24.0: 0.05,
}

DEFAULT_LOGS_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_QUEUE_ROOT = rf"\\Moderskipet\Darsia_Queue\Kalibrering_AC_{CAMPAIGN_ID}"
OVERFLOW_REVIEW_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'overflow_weight_review'
NIGHT_WINNER = configured_tool_path('ffac', 'calibration_logs') / 'ac60_overflow_phase_screen_20260807_5000/titration_layers_soft05_overflow/ac60/final_full_scale_ac60.json'
RUN_SEED_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'late_gas_depth_screen_20260807_5000/titration_layers_soft05_depth'
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
MANUAL_SCALE_OVERRIDES_G = {"ac60": {1031: 3717.7}}

RUN_ASSIGNMENTS = {
    "moderskipet": ("ac20", "ac24", "ac27", "ac60"),
    "olav": (),
}
RUNS = tuple(run for assigned in RUN_ASSIGNMENTS.values() for run in assigned)


@dataclass(frozen=True)
class Candidate:
    variant: base.Variant
    diffusion_weight: float
    late_gas_start_h: float
    late_gas_start_fraction: float
    historical_control: bool = False


def _variant(name: str) -> base.Variant:
    return base.Variant(
        name=name,
        titration=True,
        optimize_aqueous_blend=True,
        bounds_file="config/bounds_seg6_phase_curve_titration_hierarchical.json",
        phase_separation="shared-signal",
        use_facies=False,
        signal_parameterization="facies-shape-layer-gain",
        aqueous_noise_gate="off",
    )


def _weight_suffix(weight: float) -> str:
    return str(weight).replace(".", "p")


CANDIDATES = (
    Candidate(
        _variant("night_method_control_fixed_depth"),
        diffusion_weight=0.0,
        late_gas_start_h=24.0,
        late_gas_start_fraction=0.05,
        historical_control=True,
    ),
    *(
        Candidate(
            _variant(
                f"balance_gas{int(start_h)}_diff_w{_weight_suffix(weight)}"
            ),
            diffusion_weight=weight,
            late_gas_start_h=start_h,
            late_gas_start_fraction=start_fraction,
        )
        for start_h, start_fraction in LATE_GAS_SETTINGS.items()
        for weight in DIFFUSION_WEIGHTS
    ),
)
VARIANTS = tuple(candidate.variant for candidate in CANDIDATES)
_CANDIDATE_BY_NAME = {
    candidate.variant.name: candidate for candidate in CANDIDATES
}
_ACTIVE_TARGET_PATHS = {
    run: DEFAULT_LOGS_ROOT / "overflow_targets" / f"{run}.json" for run in RUNS
}

_ORIGINAL_CAMPAIGN_ENV = base._campaign_env
_ORIGINAL_MASTER_COMMAND = base._master_command
_ORIGINAL_RUN_PREFLIGHT = base.run_preflight


def _base_campaign_env(variant: base.Variant) -> dict[str, str]:
    """Build the canonical phase-campaign environment without shared globals."""

    env = dict(os.environ)
    reset_names = {
        "FFAC_TITRATION_RECIPE",
        "FFAC_MASTER_LIGHT_CONTEXT",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
        "FFAC_PHOTOMETRIC_ANCHOR",
        "FFAC_PHOTOMETRIC_ANCHOR_MODE",
        "FFAC_SIGNAL_PARAMETERIZATION",
        "FFAC_PHASE_SEPARATION",
        "FFAC_AQUEOUS_NOISE_GATE",
        "FFAC_EXPERT_CONSTRAINT_MODE",
        "FFAC_EXPERT_CONSTRAINT_EXPANSION_M",
        "FFAC_LATE_GAS_START_H",
        "FFAC_LATE_GAS_MAX_FRACTION",
        "FFAC_LATE_GAS_PENALTY_WEIGHT",
        "FFAC_LATE_GAS_DECAY_HALF_LIFE_H",
        "FFAC_LATE_GAS_FLOOR_FRACTION",
        "FFAC_LATE_GAS_CONSTRAINT_MODE",
        "FFAC_POST_INJECTION_BALANCE_WEIGHT",
        "FFAC_POST_INJECTION_BALANCE_START_H",
        "FFAC_AQUEOUS_DIFFUSION_PENALTY_WEIGHT",
        "FFAC_AQUEOUS_DIFFUSION_START_H",
        "FFAC_AQUEOUS_DIFFUSION_SUPPORT_LEVEL",
        "FFAC_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H",
        "FFAC_OVERFLOW_PHASE_TARGETS",
        "FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT",
        "FFAC_DEPTH_PARAMETERIZATION",
        "FFAC_DEPTH_MIN_M",
        "FFAC_DEPTH_MAX_M",
        "FFAC_DEPTH_PRIOR_WEIGHT",
        "FFAC_COLORCHECKER_REFINEMENT",
        "FFAC_COLORCHECKER_AUTO_ORIENT",
        "FFAC_COLORCHECKER_STRATEGY",
        "FFAC_COLORCHECKER_REFINEMENT_STEPS",
        "FFAC_COLORCHECKER_REFINEMENT_RIDGE",
        "FFAC_COLORCHECKER_REFINEMENT_HUBER_DELTA",
        "FFAC_COLORCHECKER_REFINEMENT_MIN_IMPROVEMENT",
        "FFAC_COLORCHECKER_REFINEMENT_MAX_CONDITION",
        "FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MIN",
        "FFAC_COLORCHECKER_REFINEMENT_DIAGONAL_MAX",
        "FFAC_COLORCHECKER_REFINEMENT_OFF_DIAGONAL_MAX_ABS",
        "FFAC_COLORCHECKER_REFINEMENT_TRANSLATION_MAX_ABS",
        "FFAC_COLORCHECKER_RESIDUAL_WARN_THRESHOLD",
    }
    for name in reset_names:
        env.pop(name, None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FFAC_REQUIRE_VARYING_DEPTH": "on",
            "FFAC_EXPECTED_DEPTH_SHA256": base.EXPECTED_RUNTIME_DEPTH_SHA256,
            "FFAC_TITRATION_FLASH": "on" if variant.titration else "off",
            "FFAC_COUPLE_AQ_GAS": "on",
            "FFAC_PARAMETERIZE_GAS_WIDTH": "on",
            "FFAC_OPTIMIZE_AQUEOUS_BLEND": (
                "on" if variant.optimize_aqueous_blend else "off"
            ),
            "FFAC_PHASE_SEPARATION": variant.phase_separation,
            "FFAC_SIGNAL_PARAMETERIZATION": variant.signal_parameterization,
            "FFAC_AQUEOUS_NOISE_GATE": variant.aqueous_noise_gate,
            "FFAC_TEMPLATE_REGISTRATION": "ac14_template",
            "FFAC_TEMPLATE_REGISTRATION_MODE": "partial_affine",
            "FFAC_TEMPLATE_REGISTRATION_STRICT": "on",
            "FFAC_STATIC_LIGHT_CORRECTION": "off",
            "FFAC_FLAGGED_FRAME_POLICY": "exact",
            "FFAC_COLOR_PATH_ANCHOR": "off",
            "FFAC_COLOR_PATH_ANCHOR_WEIGHT": "0.75",
            "FFAC_COLOR_PATH_ANCHOR_STRICT": "on",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENCV_NUM_THREADS": "1",
        }
    )
    return env


def _source_candidate(run: str) -> Path:
    if run == "ac60":
        return NIGHT_WINNER
    return RUN_SEED_ROOT / run / f"final_full_scale_{run}.json"


def _candidate_digest(params: dict[str, float]) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _clean_seed(raw_params: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(value)
        for name, value in raw_params.items()
        if not str(name).startswith("depth.")
        and not str(name).startswith("gas.hybrid_")
        and str(name) not in {"gas.clear_onset", "gas.clear_width"}
    }


def _seed_candidates(raw_params: dict[str, Any]) -> list[dict[str, Any]]:
    base_params = _clean_seed(raw_params)
    selected_blend = float(base_params.get("flash.aqueous_blend", 0.2))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for blend in dict.fromkeys((selected_blend, 0.0, 0.2, 0.4)):
        params = dict(base_params)
        params["flash.aqueous_blend"] = float(blend)
        digest = _candidate_digest(params)
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append(
            {
                "params": params,
                "source_family": "run-specific hierarchical titration winner",
                "seed_aqueous_blend": float(blend),
                "params_sha256": digest,
            }
        )
    return candidates


def _build_seed_bundles(logs_root: Path) -> dict[str, Path]:
    source_rows: dict[str, dict[str, Any]] = {}
    params_by_run: dict[str, list[dict[str, Any]]] = {}
    for run in RUNS:
        source = _source_candidate(run)
        if not source.is_file():
            raise FileNotFoundError(f"Calibration seed unavailable: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        raw_params = payload.get("params")
        if not isinstance(raw_params, dict) or not raw_params:
            raise RuntimeError(f"Calibration seed has no parameters: {source}")
        params_by_run[run] = _seed_candidates(raw_params)
        source_rows[run] = {
            "path": str(source),
            "sha256": base._sha256(source),
            "objective_full_scale": payload.get("objective_full_scale"),
        }

    host = socket.gethostname().upper()
    output: dict[str, Path] = {}
    for definition in CANDIDATES:
        variant = definition.variant
        path = logs_root / f"seed_params_{variant.name}_{host}.json"
        base._atomic_json(
            path,
            {
                "metadata": {
                    "campaign_id": CAMPAIGN_ID,
                    "variant": variant.name,
                    "sources": source_rows,
                    "fixed_depth": True,
                    "candidate_count_per_run": len(
                        next(iter(params_by_run.values()))
                    ),
                },
                "params_by_run": params_by_run,
            },
        )
        output[variant.name] = path
    return output


def _campaign_env(variant: base.Variant) -> dict[str, str]:
    definition = _CANDIDATE_BY_NAME[variant.name]
    env = _base_campaign_env(variant)
    env.update(
        {
            "FFAC_EXPERT_CONSTRAINT_MODE": "soft",
            "FFAC_EXPERT_CONSTRAINT_EXPANSION_M": "0.005",
            "FFAC_OVERFLOW_PHASE_PENALTY_WEIGHT": str(
                OVERFLOW_PHASE_PENALTY_WEIGHT
            ),
            "FFAC_DEPTH_PARAMETERIZATION": "fixed",
            "FFAC_DEPTH_PRIOR_WEIGHT": "0",
            "FFAC_AQUEOUS_DIFFUSION_PENALTY_WEIGHT": str(
                definition.diffusion_weight
            ),
            "FFAC_AQUEOUS_DIFFUSION_START_H": str(DIFFUSION_START_H),
            "FFAC_AQUEOUS_DIFFUSION_SUPPORT_LEVEL": str(
                DIFFUSION_SUPPORT_LEVEL
            ),
            "FFAC_AQUEOUS_DIFFUSION_PAIR_MIN_DT_H": str(
                DIFFUSION_PAIR_MIN_DT_H
            ),
            "FFAC_EXTRA_CALIBRATION_TIMES_H": ",".join(
                str(value) for value in EXTRA_CALIBRATION_TIMES_H
            ),
            "FFAC_EXTRA_CALIBRATION_TIME_TOLERANCE_SECONDS": str(
                EXTRA_CALIBRATION_TOLERANCE_SECONDS
            ),
            "FFAC_STATIC_LIGHT_CORRECTION": "off",
            "FFAC_AQUEOUS_NOISE_GATE": "off",
        }
    )
    if definition.historical_control:
        env.update(
            {
                "FFAC_LATE_GAS_START_H": "24",
                "FFAC_LATE_GAS_MAX_FRACTION": "0.05",
                "FFAC_LATE_GAS_DECAY_HALF_LIFE_H": "0",
                "FFAC_LATE_GAS_FLOOR_FRACTION": "0",
                "FFAC_LATE_GAS_PENALTY_WEIGHT": "100",
                "FFAC_LATE_GAS_CONSTRAINT_MODE": "phase-closure",
                "FFAC_POST_INJECTION_BALANCE_WEIGHT": "0",
                "FFAC_POST_INJECTION_BALANCE_START_H": str(
                    POST_INJECTION_START_H
                ),
            }
        )
    else:
        env.update(
            {
                "FFAC_LATE_GAS_START_H": str(definition.late_gas_start_h),
                "FFAC_LATE_GAS_MAX_FRACTION": str(
                    definition.late_gas_start_fraction
                ),
                "FFAC_LATE_GAS_DECAY_HALF_LIFE_H": str(
                    LATE_GAS_DECAY_HALF_LIFE_H
                ),
                "FFAC_LATE_GAS_FLOOR_FRACTION": "0",
                "FFAC_LATE_GAS_PENALTY_WEIGHT": str(
                    LATE_GAS_PENALTY_WEIGHT
                ),
                "FFAC_LATE_GAS_CONSTRAINT_MODE": "gas-only",
                "FFAC_POST_INJECTION_BALANCE_WEIGHT": str(
                    POST_INJECTION_BALANCE_WEIGHT
                ),
                "FFAC_POST_INJECTION_BALANCE_START_H": str(
                    POST_INJECTION_START_H
                ),
            }
        )
    return env


def _master_command(**kwargs: Any) -> tuple[str, ...]:
    return tuple(_ORIGINAL_MASTER_COMMAND(**kwargs))


def _write_target_artifacts(logs_root: Path) -> dict[str, dict[str, Any]]:
    global _ACTIVE_TARGET_PATHS
    target_root = logs_root / "overflow_targets"
    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for run in RUNS:
        review = OVERFLOW_REVIEW_ROOT / run / f"{run}_overflow_scale_review.json"
        protocol = project_path('protocols') / run / "injection_protocol.csv"
        payload = build_overflow_phase_targets(
            review_state_path=review,
            injection_protocol_path=protocol,
            evaluation_times_h=EVALUATION_TIMES_H,
            overrides_by_index=MANUAL_SCALE_OVERRIDES_G.get(run),
        )
        payload["source"]["review_state_path"] = str(review)
        payload["source"]["injection_protocol_path"] = (
            f"protocols/{run}/injection_protocol.csv"
        )
        path = target_root / f"{run}.json"
        write_overflow_phase_targets(path, payload)
        load_overflow_phase_constraint(path, run=run)
        payloads[run] = payload
        paths[run] = path
    _ACTIVE_TARGET_PATHS = paths
    return payloads


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
    for definition in CANDIDATES:
        variant = definition.variant
        for run in RUN_ASSIGNMENTS[role]:
            env = _campaign_env(variant)
            env["FFAC_OVERFLOW_PHASE_TARGETS"] = str(_ACTIVE_TARGET_PATHS[run])
            job_id = f"{variant.name}_{run}"
            queue = f"{queue_root}_{job_id}"
            specs.append(
                base.ProcessSpec(
                    job_id=job_id,
                    variant=variant.name,
                    run=run,
                    queue=queue,
                    command=_master_command(
                        variant=variant,
                        run=run,
                        backend=backend,
                        queue=queue,
                        logs_root=logs_root,
                        seed_path=seed_paths[variant.name],
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
    payload["overflow_phase_constraints"] = {
        run: {
            "path": str(_ACTIVE_TARGET_PATHS[run]),
            "sha256": base._sha256(_ACTIVE_TARGET_PATHS[run]),
            "target_count": len(target_payloads[run]["targets"]),
            "feed_rate_source": target_payloads[run]["physics"][
                "feed_rate_source"
            ],
        }
        for run in RUNS
    }
    base._atomic_json(
        logs_root / f"preflight_{socket.gethostname().upper()}.json", payload
    )
    return payload, seed_paths


def validate_plan() -> dict[str, Any]:
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "runs": list(RUNS),
        "run_assignments": {
            role: list(runs) for role, runs in RUN_ASSIGNMENTS.items()
        },
        "variants": [
            {
                "name": definition.variant.name,
                "historical_control": definition.historical_control,
                "diffusion_weight": definition.diffusion_weight,
                "late_gas_start_h": definition.late_gas_start_h,
                "late_gas_start_fraction": definition.late_gas_start_fraction,
                "late_gas_mode": (
                    "phase-closure"
                    if definition.historical_control
                    else "gas-only"
                ),
            }
            for definition in CANDIDATES
        ],
        "calibrations": len(CANDIDATES) * len(RUNS),
        "optuna_trials": len(CANDIDATES) * len(RUNS) * MAX_ITERS,
        "random_warmups": len(CANDIDATES) * len(RUNS) * WARMUP_ITERS,
        "controlled_question": (
            "Can a mass-conserving redistribution prior reduce late mass drift, "
            "and does constraining free gas from 12 h add value beyond 24 h?"
        ),
        "common_method": {
            "base": "night winner titration_layers_soft05_overflow",
            "aqueous_model": "TitrationFlash with optimized bounded blend",
            "spatial_response": "hierarchical facies shape plus layer gain",
            "expert_constraint": "soft 0.5 cm",
            "depth": "fixed measured AC14 per-pixel map; no depth parameters",
            "color_paths": "run-native; no AC60 color anchor",
            "overflow_phase_penalty_weight": OVERFLOW_PHASE_PENALTY_WEIGHT,
            "post_injection_balance_weight": POST_INJECTION_BALANCE_WEIGHT,
            "post_injection_start_h": POST_INJECTION_START_H,
            "diffusion_metric": (
                "non-increasing concentration peaks and non-decreasing "
                "mass-normalized effective aqueous pore volume"
            ),
            "late_gas_envelope": (
                "10% at 12 h, 5% at 24 h, 1.25% at 48 h when enabled at 12 h; "
                "identical envelope from 24 h for both onset variants"
            ),
        },
    }


def _configure_base() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.RUN_ASSIGNMENTS = RUN_ASSIGNMENTS
    base.RUNS = RUNS
    base.MAX_ITERS = MAX_ITERS
    base.WARMUP_ITERS = WARMUP_ITERS
    base.DEFAULT_LOGS_ROOT = DEFAULT_LOGS_ROOT
    base.DEFAULT_QUEUE_ROOT = DEFAULT_QUEUE_ROOT
    base.VARIANTS = VARIANTS
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        SCRIPT_DIR / "overflow_phase_constraint.py",
        SCRIPT_DIR / "temporal_gas_support.py",
        SCRIPT_DIR / "expert_constraint_support.py",
        SCRIPT_DIR / "ac_locked_titration_flash.py",
        SCRIPT_DIR / "ac_production_campaign.py",
        SCRIPT_DIR / "verify_ac_depth_maps.py",
        project_path('config/bounds_seg6_phase_curve_titration_hierarchical.json'),
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
