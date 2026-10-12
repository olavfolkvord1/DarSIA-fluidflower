"""Run the coupled phase-width and aqueous-curve calibration screen.

The campaign compares a linear SimpleFlash control with a TitrationFlash whose
aqueous curve may blend toward a linear response. Both variants use the same
structural phase transition: T = max_value_aq = min_value_g and
max_value_g = T + W. Six runs are split between CUDA and OpenCL, with three
resident GPU contexts per host and one logical in-flight trial per context.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import psutil
except Exception:
    psutil = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.ac_locked_titration_flash import (  # noqa: E402
    COLOR_PATH_SOURCE_ROOT,
    EXPECTED_RUNTIME_DEPTH_SHA256,
    MEASUREMENTS_PATH,
    RESULTS_ROOT,
)
from darsia.fluidflower.tools.verify_ac_depth_maps import (  # noqa: E402
    EXPECTED_DEPTH_ARRAY_SHA256,
    verify_depth_maps,
)


CAMPAIGN_ID = "phase_curve_screen_20260806_5000"
RUN_ASSIGNMENTS = {
    "moderskipet": ("ac20", "ac24", "ac26"),
    "olav": ("ac60", "ac32", "ac27"),
}
RUNS = tuple(run for assigned in RUN_ASSIGNMENTS.values() for run in assigned)
MAX_ITERS = 5000
WARMUP_ITERS = 150
OPTUNA_SEED = 17
DEFAULT_LOGS_ROOT = configured_tool_path('ffac', 'calibration_logs') / f'{CAMPAIGN_ID}'
DEFAULT_QUEUE_ROOT = (
    rf"\\Moderskipet\Darsia_Queue\Kalibrering_AC_{CAMPAIGN_ID}"
)
LOCKED_SIMPLE_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'locked_simple_ffum_20260805_5000/simple_locked_ffum_runpath'
LOCKED_TITRATION_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'locked_titration_ffum_20260804_3200/titration_locked_ffum_runpath'
FACTORIAL_ROOT = configured_tool_path('ffac', 'calibration_logs') / 'flash_anchor_factorial_20260804_3200_pairedseeds'

DETAILED_LABEL_TO_FACIES = {
    0: 8,
    3: 1,
    4: 7,
    5: 7,
    6: 7,
    7: 2,
    8: 7,
    9: 5,
    10: 2,
    11: 5,
    12: 1,
    13: 8,
    14: 1,
}


@dataclass(frozen=True)
class Variant:
    name: str
    titration: bool
    optimize_aqueous_blend: bool
    bounds_file: str
    phase_separation: str = "shared-signal"
    use_facies: bool = True
    signal_parameterization: str = "per-label"
    aqueous_noise_gate: str = "off"


VARIANTS = (
    Variant(
        name="simple_coupled_width_runpath",
        titration=False,
        optimize_aqueous_blend=False,
        bounds_file="config/bounds_seg6_phase_curve_simple.json",
    ),
    Variant(
        name="titration_blend_coupled_width_runpath",
        titration=True,
        optimize_aqueous_blend=True,
        bounds_file="config/bounds_seg6_phase_curve_titration.json",
    ),
)

# Thin campaign wrappers can add their own implementation files to the
# preflight fingerprint without duplicating the supervisor implementation.
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()
PROCESS_MARKERS: tuple[str, ...] = (Path(__file__).name,)
EXPECTED_COLOR_STATE: str | None = None
SEED_PARAM_DEFAULTS: dict[str, float] = {}
# Opt-in for campaigns where both hosts can execute every run. Each host keeps
# its preferred ordering, but atomically claims work from a shared pool when a
# GPU lane becomes available.
DYNAMIC_RUN_POOL = False


@dataclass(frozen=True)
class ProcessSpec:
    job_id: str
    variant: str
    run: str
    queue: str
    command: tuple[str, ...]
    stdout: str
    env: dict[str, str]


def _candidate_runs(role: str) -> tuple[str, ...]:
    """Return preferred runs followed by stealable runs from the other hosts."""

    preferred = tuple(RUN_ASSIGNMENTS[role])
    if not DYNAMIC_RUN_POOL:
        return preferred
    ordered = list(preferred)
    seen = set(preferred)
    for assignment_role, runs in RUN_ASSIGNMENTS.items():
        if assignment_role == role:
            continue
        for run in runs:
            if run not in seen:
                ordered.append(run)
                seen.add(run)
    return tuple(ordered)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return completed.stdout.strip()


def _source_folder(run: str) -> Path:
    return (
        COLOR_PATH_SOURCE_ROOT
        / run
        / "calibration"
        / "color_path_interpretation"
    )


def _variant_sources(variant: Variant, run: str) -> tuple[tuple[str, Path], ...]:
    locked_simple = LOCKED_SIMPLE_ROOT / run / f"final_full_scale_{run}.json"
    locked_titration = (
        LOCKED_TITRATION_ROOT / run / f"final_full_scale_{run}.json"
    )
    if variant.titration:
        return (
            ("locked-titration", locked_titration),
            (
                "flexible-titration-coupled",
                FACTORIAL_ROOT
                / "titration_coupled_runpath"
                / f"final_full_scale_{run}.json",
            ),
            ("locked-simple-control", locked_simple),
        )
    return (
        ("locked-simple", locked_simple),
        (
            "flexible-simple-coupled",
            FACTORIAL_ROOT
            / "simple_coupled_runpath"
            / f"final_full_scale_{run}.json",
        ),
        ("locked-titration-control", locked_titration),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _convert_seed(
    raw_params: dict[str, Any],
    *,
    variant: Variant,
    aqueous_blend: float = 0.0,
    clear_onset: float = 0.20,
    clear_width: float = 0.40,
) -> dict[str, float]:
    source_signal = {
        str(name): float(value)
        for name, value in raw_params.items()
        if str(name).startswith("signal.")
    }
    if variant.signal_parameterization == "facies-shape-layer-gain":
        params: dict[str, float] = {}
        for facies in sorted(set(DETAILED_LABEL_TO_FACIES.values())):
            values = [
                float(source_signal.get(f"signal.label{facies}.value{idx}", 0.0))
                for idx in range(7)
            ]
            amplitude = _clamp(values[-1], 0.5, 2.0)
            for idx, value in enumerate(values):
                fraction = 0.0 if idx == 0 else _clamp(value / amplitude, 0.0, 1.0)
                if idx == 6:
                    fraction = 1.0
                params[f"signal.facies{facies}.value{idx}"] = fraction
            for label, material in DETAILED_LABEL_TO_FACIES.items():
                if material == facies:
                    params[f"signal.label{label}.gain"] = amplitude
    elif not variant.use_facies:
        params = {}
        for label, facies in DETAILED_LABEL_TO_FACIES.items():
            for idx in range(7):
                name = f"signal.label{facies}.value{idx}"
                params[f"signal.label{label}.value{idx}"] = float(
                    source_signal.get(name, 0.0)
                )
    else:
        params = dict(source_signal)
    transition = _clamp(
        float(raw_params.get("flash.max_value_aq", 1.0)),
        0.85,
        1.10,
    )
    source_max_g = float(raw_params.get("flash.max_value_g", transition + 0.5))
    width = _clamp(source_max_g - transition, 0.40, 0.85)
    params.update(
        {
            "flash.min_value_aq": 0.0,
            "flash.max_value_aq": transition,
            "flash.gas_width": width,
        }
    )
    if variant.phase_separation == "clearwater-direct":
        params.pop("flash.gas_width", None)
    if variant.phase_separation.startswith("clearwater-"):
        params.update(
            {
                "gas.clear_onset": _clamp(clear_onset, 0.02, 0.75),
                "gas.clear_width": _clamp(clear_width, 0.05, 0.90),
            }
        )
    if variant.optimize_aqueous_blend:
        params["flash.aqueous_blend"] = _clamp(aqueous_blend, 0.0, 0.40)
    params.update({name: float(value) for name, value in SEED_PARAM_DEFAULTS.items()})
    return params


def _clearwater_seed_pairs(variant: Variant) -> tuple[tuple[float, float], ...]:
    if not variant.phase_separation.startswith("clearwater-"):
        return ((0.20, 0.40),)
    return (
        (0.10, 0.20),
        (0.20, 0.40),
        (0.40, 0.40),
    )


def _build_seed_bundles(logs_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    host = socket.gethostname().upper()
    for variant in VARIANTS:
        params_by_run: dict[str, list[dict[str, Any]]] = {}
        sources_by_run: dict[str, list[dict[str, Any]]] = {}
        for run in RUNS:
            candidates: list[dict[str, Any]] = []
            source_rows: list[dict[str, Any]] = []
            seen: set[str] = set()
            for family, source in _variant_sources(variant, run):
                if not source.is_file():
                    raise FileNotFoundError(f"Calibration seed unavailable: {source}")
                payload = json.loads(source.read_text(encoding="utf-8-sig"))
                raw_params = payload.get("params")
                if not isinstance(raw_params, dict) or not raw_params:
                    raise RuntimeError(f"Seed has no parameter mapping: {source}")
                blend_values = (
                    (0.0, 0.20, 0.40)
                    if variant.optimize_aqueous_blend
                    else (0.0,)
                )
                converted_hashes: list[str] = []
                for blend in blend_values:
                    for clear_onset, clear_width in _clearwater_seed_pairs(variant):
                        params = _convert_seed(
                            raw_params,
                            variant=variant,
                            aqueous_blend=blend,
                            clear_onset=clear_onset,
                            clear_width=clear_width,
                        )
                        digest = hashlib.sha256(
                            json.dumps(
                                params, sort_keys=True, separators=(",", ":")
                            ).encode("utf-8")
                        ).hexdigest()
                        converted_hashes.append(digest)
                        if digest in seen:
                            continue
                        seen.add(digest)
                        candidates.append(
                            {
                                "params": params,
                                "source": str(source),
                                "source_family": family,
                                "seed_aqueous_blend": blend,
                                "seed_clear_onset": clear_onset,
                                "seed_clear_width": clear_width,
                            }
                        )
                source_rows.append(
                    {
                        "family": family,
                        "path": str(source),
                        "sha256": _sha256(source),
                        "converted_params_sha256": converted_hashes,
                        "source_objective_full_scale": payload.get(
                            "objective_full_scale"
                        ),
                    }
                )
            params_by_run[run] = candidates
            sources_by_run[run] = source_rows
        path = logs_root / f"seed_params_{variant.name}_{host}.json"
        _atomic_json(
            path,
            {
                "metadata": {
                    "created_at_utc": _utc_now(),
                    "campaign_id": CAMPAIGN_ID,
                    "variant": variant.name,
                    "phase_parameterization": variant.phase_separation,
                    "sources": sources_by_run,
                },
                "params_by_run": params_by_run,
            },
        )
        output[variant.name] = path
    return output


def validate_plan() -> dict[str, Any]:
    assigned = [run for runs in RUN_ASSIGNMENTS.values() for run in runs]
    if len(assigned) != 6 or len(set(assigned)) != 6:
        raise RuntimeError(f"Expected six unique runs, got {assigned}")
    if any(len(runs) != 3 for runs in RUN_ASSIGNMENTS.values()):
        raise RuntimeError(f"GPU assignments are not balanced: {RUN_ASSIGNMENTS}")
    if len({variant.name for variant in VARIANTS}) != len(VARIANTS):
        raise RuntimeError("Variant names must be unique")
    return {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "variants": [asdict(variant) for variant in VARIANTS],
        "runs": list(RUNS),
        "run_assignments": {
            role: list(runs) for role, runs in RUN_ASSIGNMENTS.items()
        },
        "calibrations": len(RUNS) * len(VARIANTS),
        "optuna_trials": len(RUNS) * len(VARIANTS) * MAX_ITERS,
        "random_warmups": len(RUNS) * len(VARIANTS) * WARMUP_ITERS,
        "max_iters_per_calibration": MAX_ITERS,
        "random_warmups_per_calibration": WARMUP_ITERS,
        "optuna_seed": OPTUNA_SEED,
        "max_concurrent_gpu_contexts_per_host": 3,
        "max_active_runs_per_context": 1,
        "max_in_flight_per_run": 1,
        "fixed_settings": {
            "phase_transition": "T=max_value_aq=min_value_g",
            "gas_upper_coordinate": "T+W",
            "color_paths": "run-native",
            "color_path_anchor": "off",
            "template_registration": "ac14_template",
            "template_registration_mode": "partial_affine",
            "objective": "point-wise L1",
            "calibration_frames": 16,
            "frame_policy": "exact",
            "static_light_correction": "off",
            # Keep the legacy key for consumers of earlier campaign plans,
            # while naming both hash domains explicitly. DarSIA resizes the
            # cached 500 x 1000 field to the 2979 x 5472 runtime geometry.
            "depth_sha256": EXPECTED_RUNTIME_DEPTH_SHA256,
            "runtime_depth_sha256": EXPECTED_RUNTIME_DEPTH_SHA256,
            "cached_depth_array_sha256": EXPECTED_DEPTH_ARRAY_SHA256,
            "depth_hash_domains": {
                "cached": "depth_map.npz array at 500x1000",
                "runtime": "DarSIA geometry depth at 2979x5472",
            },
            "color_path_source_root": str(COLOR_PATH_SOURCE_ROOT),
        },
    }


def _campaign_env(variant: Variant) -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "FFAC_TITRATION_RECIPE",
        "FFAC_MASTER_LIGHT_CONTEXT",
        "FFAC_STATIC_LIGHT_REFERENCE",
        "FFAC_STATIC_LIGHT_SPATIAL_SIGMA",
        "FFAC_PHOTOMETRIC_ANCHOR",
        "FFAC_PHOTOMETRIC_ANCHOR_MODE",
        "FFAC_SIGNAL_PARAMETERIZATION",
        "FFAC_PHASE_SEPARATION",
        "FFAC_AQUEOUS_NOISE_GATE",
        "FFAC_AQUEOUS_NOISE_QUANTILE_LOW",
        "FFAC_AQUEOUS_NOISE_QUANTILE_HIGH",
        "FFAC_AQUEOUS_NOISE_GATE_STRICT",
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
        "FFAC_DEPTH_EDGE_PRIOR_SIGMA_M",
        "FFAC_DEPTH_BULGE_PRIOR_SIGMA_M",
        "FFAC_HYBRID_OPTICAL_ONSET",
        "FFAC_HYBRID_OPTICAL_WIDTH",
        "FFAC_HYBRID_OPTICAL_SMOOTHING_SIGMA_M",
        "FFAC_HYBRID_MIN_COMPONENT_AREA_M2",
        "FFAC_HYBRID_HISTORY_RADIUS_M",
        "FFAC_HYBRID_REENTRY_RADIUS_M",
        "FFAC_HARD_SEAL_GAS",
        "FFAC_HARD_SEAL_FACIES_IDS",
        "FFAC_CAUSAL_AQ_TRACE_CONFIDENCE",
        "FFAC_CAUSAL_AQ_CORE_CONFIDENCE",
        "FFAC_CAUSAL_AQ_OBSERVATION_CONFIDENCE",
        "FFAC_CAUSAL_AQ_MIN_OBSERVED_FRACTION",
        "FFAC_CAUSAL_AQ_HISTORY_RADIUS_M",
        "FFAC_CAUSAL_AQ_REENTRY_RADIUS_M",
        "FFAC_CAUSAL_AQ_REENTRY_OBSERVED_FRACTION",
        "FFAC_CAUSAL_AQ_REENTRY_CORE_FRACTION",
        "FFAC_CAUSAL_AQ_STRUCTURE_SCALE",
        "FFAC_STATIC_LIGHT_STRIDE",
        "FFAC_STATIC_LIGHT_MIN_SAMPLES",
        "FFAC_STATIC_LIGHT_MAX_SAMPLES",
        "FFAC_STATIC_LIGHT_GAIN_CLIP",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FFAC_REQUIRE_VARYING_DEPTH": "on",
            "FFAC_EXPECTED_DEPTH_SHA256": EXPECTED_RUNTIME_DEPTH_SHA256,
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
    if variant.use_facies:
        env.update(
            {
                "FFAC_COLOR_PATH_SOURCE_ROOT": str(COLOR_PATH_SOURCE_ROOT),
                "FFAC_COLOR_PATH_SOURCE_STRICT": "on",
            }
        )
    else:
        env.pop("FFAC_COLOR_PATH_SOURCE_ROOT", None)
        env.pop("FFAC_COLOR_PATH_SOURCE_STRICT", None)
    if variant.aqueous_noise_gate != "off":
        env.update(
            {
                "FFAC_AQUEOUS_NOISE_QUANTILE_LOW": "0.99",
                "FFAC_AQUEOUS_NOISE_QUANTILE_HIGH": "0.999",
                "FFAC_AQUEOUS_NOISE_GATE_STRICT": "on",
            }
        )
    return env


def _master_command(
    *,
    variant: Variant,
    run: str,
    backend: str,
    queue: str,
    logs_root: Path,
    seed_path: Path,
) -> tuple[str, ...]:
    python = Path(sys.executable)
    logs_dir = logs_root / variant.name / run
    local_optuna = (
        Path(os.environ.get("LOCALAPPDATA", str(project_path('logs'))))
        / "ff_ac"
        / "optuna"
        / CAMPAIGN_ID
        / variant.name
        / run
    )
    return (
        str(python),
        "scripts/distributed_auto_calibration_queue.py",
        "master",
        "--queue",
        queue,
        "--no-clear-queue",
        "--runs",
        run,
        "--config-dir",
        "config_seg6/run_ac",
        "--logs-dir",
        str(logs_dir),
        "--exact-logs-dir",
        "--use-facies",
        "true" if variant.use_facies else "false",
        "--per-label",
        "true",
        "--objective-integral",
        "off",
        "--bounds-file",
        variant.bounds_file,
        "--no-save-calibration",
        "--seed-params-file",
        str(seed_path),
        "--optuna-seed",
        str(OPTUNA_SEED),
        "--optuna-storage-dir",
        str(local_optuna),
        "--max-iters",
        str(MAX_ITERS),
        "--warmup-iters",
        str(WARMUP_ITERS),
        "--run-mode",
        "parallel",
        "--max-active-runs",
        "1",
        "--max-in-flight-per-run",
        "1",
        "--control-dir",
        f"{queue}\\control",
        "--sanity-every",
        "0",
        "--sanity-scale",
        "1.00",
        "--quality-dtype",
        "float32",
        "--local-eval-backend",
        backend,
        "--local-run",
        run,
        "--local-owner-id",
        f"{socket.gethostname()}_{variant.name}_{run}",
        "--poll-seconds",
        "0.05",
        "--max-retries",
        "4",
        "--master-mem-log-seconds",
        "120",
    )


def _validate_backend(role: str) -> dict[str, Any]:
    if role == "moderskipet":
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            raise RuntimeError("No CUDA device is available")
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {"backend": "cuda", "device_count": count, "device": str(name)}

    import pyopencl as cl

    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices()
        if device.type & cl.device_type.GPU
    ]
    if not devices:
        raise RuntimeError("No OpenCL GPU is available")
    return {
        "backend": "opencl",
        "device_count": len(devices),
        "device": str(devices[0].name),
    }


def run_preflight(logs_root: Path, role: str) -> tuple[dict[str, Any], dict[str, Path]]:
    plan = validate_plan()
    python = Path(sys.executable)
    if not python.is_file():
        raise FileNotFoundError(f"Python unavailable: {python}")
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"Results root unavailable: {RESULTS_ROOT}")
    for variant in VARIANTS:
        bounds = REPO_ROOT / variant.bounds_file
        if not bounds.is_file():
            raise FileNotFoundError(f"Bounds unavailable: {bounds}")
    color_paths: dict[str, dict[str, Any]] = {}
    color_correction: dict[str, dict[str, Any]] = {}
    candidate_runs = _candidate_runs(role)
    for run in candidate_runs:
        state_path = (
            project_path('config_seg6/run_ac/.color_state')
            / f"{run}.txt"
        )
        state = (
            state_path.read_text(encoding="utf-8-sig").strip().lower()
            if state_path.is_file()
            else "off"
        )
        if state not in {"on", "off"}:
            raise ValueError(f"Invalid color-correction state for {run}: {state!r}")
        if EXPECTED_COLOR_STATE is not None and state != EXPECTED_COLOR_STATE:
            raise RuntimeError(
                f"Color-correction state for {run} is {state!r}; "
                f"expected {EXPECTED_COLOR_STATE!r}"
            )
        color_correction[run] = {
            "state": state,
            "path": str(state_path),
            "sha256": _sha256(state_path) if state_path.is_file() else None,
        }
        record: dict[str, Any] = {}
        if any(variant.use_facies for variant in VARIANTS):
            files = sorted(
                _source_folder(run).glob("color_path_interpretation_*.json")
            )
            if not files:
                raise FileNotFoundError(
                    f"Run-native facies color paths unavailable for {run}"
                )
            record["facies"] = [str(path) for path in files]
        if any(not variant.use_facies for variant in VARIANTS):
            labels_root = (
                RESULTS_ROOT
                / run
                / "calibration"
                / "color"
                / "relative_colorpath_labels"
            )
            required = (
                labels_root / "color_paths" / "from_labels" / "metadata.json",
                labels_root / "color_to_mass" / "from_labels" / "metadata.json",
            )
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Run-native detailed-layer calibration unavailable for {run}: "
                    + ", ".join(str(path) for path in missing)
                )
            record["labels"] = [str(path) for path in required]
        color_paths[run] = record
    seed_paths = _build_seed_bundles(logs_root)
    depth = verify_depth_maps(
        list(candidate_runs),
        results_root=RESULTS_ROOT,
        measurements_path=MEASUREMENTS_PATH,
    )
    source_paths = [
        SCRIPT_DIR / "auto_calibrate_color_to_mass.py",
        SCRIPT_DIR / "distributed_auto_calibration_queue.py",
        SCRIPT_DIR / "ffac_titration_flash.py",
        SCRIPT_DIR / "opencl_integrated_evaluator.py",
        SCRIPT_DIR / "spatial_mass_evaluator.py",
        SCRIPT_DIR / "aqueous_noise_gate.py",
        SCRIPT_DIR / "ac_prepare_label_basis.py",
        project_path('config_seg6/labels_basis.toml'),
        SCRIPT_DIR / "ac_clearwater_phase_pilot.py",
        SCRIPT_DIR / "clearwater_phase_reference.py",
        Path(__file__).resolve(),
        *EXTRA_SOURCE_PATHS,
        *(REPO_ROOT / variant.bounds_file for variant in VARIANTS),
    ]
    payload = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "checked_at_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "role": role,
        "backend": _validate_backend(role),
        "ok": True,
        "ff_ac_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "ff_ac_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
        "darsia_commit": _git_value(
            project_path('external/darsia'), "rev-parse", "HEAD"
        ),
        "depth": depth,
        "color_correction": color_correction,
        "seed_files": {name: str(path) for name, path in seed_paths.items()},
        "color_path_files": color_paths,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths
        },
        "plan": plan,
    }
    _atomic_json(logs_root / "campaign_plan.json", plan)
    _atomic_json(
        logs_root / f"preflight_{socket.gethostname().upper()}.json",
        payload,
    )
    return payload, seed_paths


def build_specs(
    *,
    role: str,
    logs_root: Path,
    queue_root: str,
    seed_paths: dict[str, Path],
) -> tuple[ProcessSpec, ...]:
    backend = "cuda" if role == "moderskipet" else "opencl"
    local_logs = project_path('logs') / CAMPAIGN_ID / role
    specs: list[ProcessSpec] = []
    for variant in VARIANTS:
        env = _campaign_env(variant)
        for run in _candidate_runs(role):
            job_id = f"{variant.name}_{run}"
            queue = f"{queue_root}_{job_id}"
            specs.append(
                ProcessSpec(
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


def _queue_complete(queue: str) -> bool:
    try:
        return (Path(queue) / "master_complete.json").is_file()
    except OSError:
        return False


def _queue_status(queue: str) -> dict[str, Any]:
    root = Path(queue)

    def count(directory: str) -> int:
        try:
            return sum(path.is_file() for path in (root / directory).iterdir())
        except OSError:
            return 0

    return {
        "queue": queue,
        "exists": root.exists(),
        "pending": count("pending"),
        "in_progress": count("in_progress"),
        "done": count("done"),
        "failed": count("failed"),
        "master_complete": _queue_complete(queue),
    }


def _run_pool_root(queue_root: str) -> Path:
    return Path(f"{queue_root}_run_pool")


def _claim_path(pool_root: Path, job_id: str) -> Path:
    return pool_root / "claims" / job_id


def _read_claim(claim: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((claim / "lease.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_claim(
    claim: Path,
    *,
    spec: ProcessSpec,
    role: str,
    state: str,
    process_pid: int | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(existing or {})
    now = _utc_now()
    payload.update(
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "job_id": spec.job_id,
            "variant": spec.variant,
            "run": spec.run,
            "host": socket.gethostname().upper(),
            "role": role,
            "supervisor_pid": os.getpid(),
            "process_pid": process_pid,
            "state": state,
            "updated_at_utc": now,
        }
    )
    payload.setdefault("claimed_at_utc", now)
    _atomic_json(claim / "lease.json", payload)
    return payload


def _try_claim_job(
    pool_root: Path,
    spec: ProcessSpec,
    *,
    role: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Atomically claim an unstarted run, or observe another host's claim."""

    if _queue_complete(spec.queue):
        return False, "complete", _read_claim(_claim_path(pool_root, spec.job_id))
    claims = pool_root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    claim = _claim_path(pool_root, spec.job_id)
    try:
        claim.mkdir()
    except FileExistsError:
        existing = _read_claim(claim)
        same_supervisor_lane = bool(
            existing
            and str(existing.get("host", "")).upper()
            == socket.gethostname().upper()
            and str(existing.get("role", "")).lower() == role.lower()
        )
        if not same_supervisor_lane:
            return False, "claimed-remote", existing
        payload = _write_claim(
            claim,
            spec=spec,
            role=role,
            state="claimed",
            existing=existing,
        )
        return True, "reclaimed-local", payload
    payload = _write_claim(claim, spec=spec, role=role, state="claimed")
    return True, "claimed", payload


def _release_failed_claim(
    pool_root: Path,
    spec: ProcessSpec,
    *,
    role: str,
    exit_code: int,
) -> None:
    """Archive an exhausted local claim so the other host may retry the run."""

    claim = _claim_path(pool_root, spec.job_id)
    existing = _read_claim(claim)
    if existing is None:
        return
    _write_claim(
        claim,
        spec=spec,
        role=role,
        state="failed-released",
        existing={**existing, "exit_code": int(exit_code)},
    )
    archive_root = pool_root / "failed_claims"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = archive_root / f"{spec.job_id}.{socket.gethostname().upper()}.{stamp}"
    try:
        os.replace(claim, archive)
    except OSError:
        # Keeping the failed claim is safer than allowing duplicate execution.
        return


def _active_processes() -> list[dict[str, Any]]:
    if psutil is None:
        return []
    markers = {
        CAMPAIGN_ID.lower(),
        *(str(marker).lower() for marker in PROCESS_MARKERS),
    }
    excluded = {os.getpid()}
    parent = psutil.Process(os.getpid()).parent()
    while parent is not None:
        excluded.add(parent.pid)
        parent = parent.parent()
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        folded = command.lower()
        name = str(process.info.get("name") or "").lower()
        if (
            process.pid not in excluded
            and name in {"python.exe", "pythonw.exe", "python"}
            and any(marker in folded for marker in markers)
        ):
            rows.append(
                {"pid": process.pid, "name": process.info.get("name"), "command": command}
            )
    return rows


def _spawn(spec: ProcessSpec) -> subprocess.Popen:
    output = Path(spec.stdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("ab")
    try:
        return subprocess.Popen(
            list(spec.command),
            cwd=process_directory(),
            env=spec.env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        stream.close()


def _spec_payload(spec: ProcessSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["env"] = {
        key: value for key, value in spec.env.items() if key.startswith("FFAC_")
    }
    return payload


def _launch_dynamic_pool(
    args: argparse.Namespace,
    *,
    specs: tuple[ProcessSpec, ...],
    logs_root: Path,
    role: str,
) -> int:
    """Run host-preferred jobs while sharing all unstarted jobs across hosts."""

    pending = {spec.job_id: spec for spec in specs}
    running: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    pool_root = _run_pool_root(args.queue_root)
    pool_root.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname().upper()
    manifest_path = logs_root / f"launch_manifest_{hostname}.json"
    status_path = logs_root / f"supervisor_{hostname}.json"
    _atomic_json(
        pool_root / "hosts" / f"{hostname}.json",
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "host": hostname,
            "role": role,
            "supervisor_pid": os.getpid(),
            "candidate_jobs": [spec.job_id for spec in specs],
            "registered_at_utc": _utc_now(),
        },
    )
    _atomic_json(
        manifest_path,
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "started_at_utc": _utc_now(),
            "host": hostname,
            "role": role,
            "dynamic_run_pool": True,
            "run_pool": str(pool_root),
            "processes": [_spec_payload(spec) for spec in specs],
        },
    )

    def update_claim(
        spec: ProcessSpec,
        *,
        state: str,
        process_pid: int | None = None,
    ) -> None:
        claim = _claim_path(pool_root, spec.job_id)
        existing = _read_claim(claim)
        if existing is None:
            return
        try:
            _write_claim(
                claim,
                spec=spec,
                role=role,
                state=state,
                process_pid=process_pid,
                existing=existing,
            )
        except OSError as exc:
            print(f"CLAIM-WRITE {spec.job_id}: {exc}", flush=True)

    def start(spec: ProcessSpec, restarts: int = 0) -> None:
        process = _spawn(spec)
        running[spec.job_id] = {
            "spec": spec,
            "process": process,
            "restarts": restarts,
        }
        update_claim(spec, state="running", process_pid=process.pid)
        print(f"START {spec.job_id} pid={process.pid}", flush=True)

    def observe_complete(
        spec: ProcessSpec,
        *,
        exit_code: int = 0,
        restarts: int | None = None,
    ) -> None:
        claim = _read_claim(_claim_path(pool_root, spec.job_id)) or {}
        execution_host = str(claim.get("host") or hostname).upper()
        finished[spec.job_id] = {
            "job_id": spec.job_id,
            "variant": spec.variant,
            "run": spec.run,
            "exit_code": int(exit_code),
            "queue_complete": True,
            "restarts": restarts,
            "execution_host": execution_host,
            "observed_remote": execution_host != hostname,
        }

    while running or pending:
        for job_id, item in list(running.items()):
            process: subprocess.Popen = item["process"]
            spec: ProcessSpec = item["spec"]
            exit_code = process.poll()
            if exit_code is None:
                update_claim(spec, state="running", process_pid=process.pid)
                continue
            complete = _queue_complete(spec.queue)
            if not complete and item["restarts"] < args.max_restarts:
                restarts = int(item["restarts"]) + 1
                del running[job_id]
                print(
                    f"RESTART {job_id} exit={exit_code} count={restarts}",
                    flush=True,
                )
                time.sleep(10)
                start(spec, restarts=restarts)
                continue
            del running[job_id]
            if complete:
                update_claim(spec, state="complete", process_pid=None)
                observe_complete(
                    spec,
                    exit_code=exit_code,
                    restarts=int(item["restarts"]),
                )
                continue
            finished[job_id] = {
                "job_id": job_id,
                "variant": spec.variant,
                "run": spec.run,
                "exit_code": int(exit_code),
                "queue_complete": False,
                "restarts": int(item["restarts"]),
                "execution_host": hostname,
                "observed_remote": False,
            }
            _release_failed_claim(
                pool_root,
                spec,
                role=role,
                exit_code=int(exit_code),
            )

        for job_id, spec in list(pending.items()):
            if _queue_complete(spec.queue):
                observe_complete(spec)
                del pending[job_id]

        if len(running) < args.max_parallel:
            for job_id, spec in list(pending.items()):
                if len(running) >= args.max_parallel:
                    break
                try:
                    claimed, reason, _claim = _try_claim_job(
                        pool_root,
                        spec,
                        role=role,
                    )
                except OSError as exc:
                    print(f"CLAIM-RETRY {job_id}: {exc}", flush=True)
                    break
                if reason == "complete":
                    observe_complete(spec)
                    del pending[job_id]
                    continue
                if not claimed:
                    continue
                del pending[job_id]
                print(f"CLAIM {job_id} mode={reason}", flush=True)
                start(spec)
                time.sleep(args.gpu_stagger_seconds)

        remote_claims: list[dict[str, Any]] = []
        for spec in pending.values():
            claim = _read_claim(_claim_path(pool_root, spec.job_id))
            if claim is not None:
                remote_claims.append(
                    {
                        "job_id": spec.job_id,
                        "host": claim.get("host"),
                        "state": claim.get("state"),
                        "updated_at_utc": claim.get("updated_at_utc"),
                    }
                )
        _atomic_json(
            status_path,
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "updated_at_utc": _utc_now(),
                "host": hostname,
                "dynamic_run_pool": True,
                "run_pool": str(pool_root),
                "pending": list(pending),
                "claimed_remote": remote_claims,
                "running": [
                    {
                        "job_id": job_id,
                        "pid": item["process"].pid,
                        "restarts": item["restarts"],
                        "queue": item["spec"].queue,
                    }
                    for job_id, item in running.items()
                ],
                "finished": list(finished.values()),
            },
        )
        if running or pending:
            time.sleep(args.supervisor_poll_seconds)

    failed = [row for row in finished.values() if not row["queue_complete"]]
    return 2 if failed else 0


def launch(args: argparse.Namespace) -> int:
    role = args.role.lower()
    logs_root = Path(args.logs_root)
    preflight, seed_paths = run_preflight(logs_root, role)
    specs = build_specs(
        role=role,
        logs_root=logs_root,
        queue_root=args.queue_root,
        seed_paths=seed_paths,
    )
    active = _active_processes()
    if active:
        raise RuntimeError(
            "Phase-curve campaign processes already active: "
            + ", ".join(str(row["pid"]) for row in active)
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "dynamic_run_pool": DYNAMIC_RUN_POOL,
                    "processes": [_spec_payload(spec) for spec in specs],
                },
                indent=2,
            )
        )
        return 0

    if DYNAMIC_RUN_POOL:
        return _launch_dynamic_pool(
            args,
            specs=specs,
            logs_root=logs_root,
            role=role,
        )

    pending = list(specs)
    running: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    manifest_path = logs_root / f"launch_manifest_{socket.gethostname().upper()}.json"
    status_path = logs_root / f"supervisor_{socket.gethostname().upper()}.json"

    def start(spec: ProcessSpec, restarts: int = 0) -> None:
        process = _spawn(spec)
        running[spec.job_id] = {
            "spec": spec,
            "process": process,
            "restarts": restarts,
        }
        print(f"START {spec.job_id} pid={process.pid}", flush=True)

    while pending and len(running) < args.max_parallel:
        start(pending.pop(0))
        time.sleep(args.gpu_stagger_seconds)
    _atomic_json(
        manifest_path,
        {
            "schema": 1,
            "campaign_id": CAMPAIGN_ID,
            "started_at_utc": _utc_now(),
            "host": socket.gethostname().upper(),
            "role": role,
            "processes": [_spec_payload(spec) for spec in specs],
        },
    )

    while running or pending:
        for job_id, item in list(running.items()):
            process: subprocess.Popen = item["process"]
            exit_code = process.poll()
            if exit_code is None:
                continue
            spec: ProcessSpec = item["spec"]
            complete = _queue_complete(spec.queue)
            if not complete and item["restarts"] < args.max_restarts:
                restarts = int(item["restarts"]) + 1
                del running[job_id]
                print(
                    f"RESTART {job_id} exit={exit_code} count={restarts}",
                    flush=True,
                )
                time.sleep(10)
                start(spec, restarts=restarts)
                continue
            finished[job_id] = {
                "job_id": job_id,
                "variant": spec.variant,
                "run": spec.run,
                "exit_code": exit_code,
                "queue_complete": complete,
                "restarts": item["restarts"],
            }
            del running[job_id]
        while pending and len(running) < args.max_parallel:
            start(pending.pop(0))
            time.sleep(args.gpu_stagger_seconds)
        _atomic_json(
            status_path,
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "updated_at_utc": _utc_now(),
                "host": socket.gethostname().upper(),
                "pending": [spec.job_id for spec in pending],
                "running": [
                    {
                        "job_id": job_id,
                        "pid": item["process"].pid,
                        "restarts": item["restarts"],
                        "queue": item["spec"].queue,
                    }
                    for job_id, item in running.items()
                ],
                "finished": list(finished.values()),
            },
        )
        if running or pending:
            time.sleep(args.supervisor_poll_seconds)
    failed = [row for row in finished.values() if not row["queue_complete"]]
    return 2 if failed else 0


def campaign_status(args: argparse.Namespace) -> int:
    queues = []
    for variant in VARIANTS:
        for run in RUNS:
            job_id = f"{variant.name}_{run}"
            status = _queue_status(f"{args.queue_root}_{job_id}")
            status.update({"variant": variant.name, "run": run})
            queues.append(status)
    print(
        json.dumps(
            {
                "schema": 1,
                "campaign_id": CAMPAIGN_ID,
                "checked_at_utc": _utc_now(),
                "plan": validate_plan(),
                "active_processes": _active_processes(),
                "queues": queues,
            },
            indent=2,
        )
    )
    return 0


def stop_campaign(args: argparse.Namespace) -> int:
    if psutil is None:
        raise RuntimeError("psutil is unavailable; stop by process ID")
    active = _active_processes()
    targets: dict[int, Any] = {}
    for row in active:
        try:
            process = psutil.Process(int(row["pid"]))
            targets[process.pid] = process
            for child in process.children(recursive=True):
                targets[child.pid] = child
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    processes = sorted(
        targets.values(),
        key=lambda process: len(process.parents()),
        reverse=True,
    )
    for process in processes:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    _, alive = psutil.wait_procs(processes, timeout=args.stop_timeout)
    for process in alive:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    remaining = _active_processes()
    print(
        json.dumps(
            {"stopped": len(targets) - len(remaining), "remaining": remaining},
            indent=2,
        )
    )
    return 1 if remaining else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "stop", "plan"))
    parser.add_argument(
        "--role",
        choices=("moderskipet", "olav"),
        default=socket.gethostname().lower(),
    )
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT))
    parser.add_argument("--queue-root", default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--gpu-stagger-seconds", type=float, default=20.0)
    parser.add_argument("--supervisor-poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--stop-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "launch":
        raise SystemExit(launch(args))
    if args.action == "status":
        raise SystemExit(campaign_status(args))
    if args.action == "stop":
        raise SystemExit(stop_campaign(args))
    print(json.dumps(validate_plan(), indent=2))


if __name__ == "__main__":
    main()
