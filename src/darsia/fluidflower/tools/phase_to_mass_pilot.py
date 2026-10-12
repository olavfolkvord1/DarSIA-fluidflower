#!/usr/bin/env python3
"""Pilot the reviewed 2D phase model as input to DarSIA mass analysis.

The learned model supplies continuous visible-slice evidence for CO2 and free
gas. It never supplies concentration, saturation, depth, or mass. This pilot
keeps those quantities in a separate, traceable physical layer:

* aqueous concentration is inferred from a radiometrically corrected BTB
  acidification coordinate;
* gas evidence is mapped monotonically to gas saturation;
* DarSIA supplies pressure/temperature dependent gas density and solubility;
* the run-native DarSIA geometry supplies porosity, pixel area, and depth;
* Optuna is fitted only against injected mass and, for FF-AC, measured overflow
  gas upper bounds.

Simple Flash, Titration, and a monotone power-law diagnostic are fitted against
the same phase evidence and evaluated on identical temporal holdout frames.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash  # noqa: E402
from darsia.fluidflower.tools.nas_publish import publish_directory  # noqa: E402
from darsia.fluidflower.tools.observation_support import btb_acidification_coordinate  # noqa: E402
from darsia.fluidflower.tools import schema5_crossseries_reviewed_physics_blind_campaign as schema5  # noqa: E402


CAMPAIGN_ID = "phase_to_mass_pilot_20260831"
PHASE_CANDIDATE = "reviewed_v5_physics"
RUN_SPECS = (("ffac", "ac60"), ("ffum", "run18"))
MODELS = (
    "simple_flash",
    "simple_flash_diffusion",
    "titration",
    "titration_diffusion",
    "monotone_power",
)
CACHE_SCHEMA = 2
OBJECTIVE_CONTRACT_SCHEMA = 1
ACID_BINS = 128
GAS_BINS = 48
PREVIEW_MAX_WIDTH = 1280
DEFAULT_TRIALS = 5000
LOCAL_ROOT = local_cache_root() / CAMPAIGN_ID
OUTPUTS = {
    "ffac": configured_tool_path('ffac', 'calibration_logs') / CAMPAIGN_ID,
    "ffum": configured_tool_path('ffum', 'analysis_v2') / CAMPAIGN_ID,
}
OVERFLOW_PATH = (
    configured_tool_path('ffac', 'calibration_logs')
    / "dual_signal_exact_ab_darsia91f5bc9_20260829_5000"
    / "overflow_targets"
    / "ac60.json"
)
PH_BOUNDS = {
    # AC60 port measurements span 8.18--9.96, with an explicitly uncalibrated
    # electrode. The broad interval is a sensitivity range, not a claim that a
    # single uniform pH describes the cell.
    "ffac": (8.0, 10.1),
    "ffum": (7.6, 9.8),
}
_ATOMIC_FILE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreparedRun:
    dataset: str
    run: str
    times_h: np.ndarray
    injected_mass_kg: np.ndarray
    gas_weight_hist: np.ndarray
    aqueous_weight_hist: np.ndarray
    pore_volume_hist: np.ndarray
    acid_edges: np.ndarray
    gas_edges: np.ndarray
    overflow_gas_upper_kg: np.ndarray
    overflow_sigma_kg: np.ndarray
    frame_paths: tuple[str, ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    acid_quantiles: Mapping[str, float]
    cache_fingerprint: str

    @property
    def acid_centers(self) -> np.ndarray:
        return 0.5 * (self.acid_edges[:-1] + self.acid_edges[1:])

    @property
    def gas_centers(self) -> np.ndarray:
        return 0.5 * (self.gas_edges[:-1] + self.gas_edges[1:])


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{socket.gethostname()}.{os.getpid()}."
        f"{threading.get_ident()}.tmp"
    )
    try:
        with _ATOMIC_FILE_LOCK:
            temporary.write_text(
                json.dumps(payload, indent=2, allow_nan=False, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    )
    try:
        with _ATOMIC_FILE_LOCK:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid smoothstep interval [{low}, {high}]")
    scaled = np.clip((np.asarray(values) - low) / (high - low), 0.0, 1.0)
    return np.asarray(scaled * scaled * (3.0 - 2.0 * scaled), dtype=np.float64)


def _cache_fingerprint(dataset: str, run: str) -> str:
    # Keep the expensive full-resolution preparation cache independent of
    # Optuna, plotting, and reporting changes in this file. CACHE_SCHEMA is the
    # explicit invalidation switch when the preparation contract itself changes.
    paths = [
        project_path('scripts/schema2_phase_transfer_campaign.py'),
        project_path('scripts/schema5_crossseries_reviewed_physics_blind_campaign.py'),
        project_path('scripts/dual_phase_signal.py'),
        project_path('scripts/observation_support.py'),
        schema5.GAS_MODEL_PATHS[PHASE_CANDIDATE],
        schema5.CO2_MODEL_PATHS[PHASE_CANDIDATE],
    ]
    if dataset == "ffac" and run == "ac60" and OVERFLOW_PATH.is_file():
        paths.append(OVERFLOW_PATH)
    payload = {
        "schema": CACHE_SCHEMA,
        "dataset": dataset,
        "run": run,
        "phase_candidate": PHASE_CANDIDATE,
        "files": {str(path): _sha256(path) for path in paths},
        "acid_bins": ACID_BINS,
        "gas_bins": GAS_BINS,
        "phase_to_mass_contract": "fixed-depth-darsia-geometry-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _configure_phase_runtime() -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    schema5._install_overrides()
    base = schema5.base
    transfer = base.transfer
    transfer.optical_control.install_runtime_hooks()
    gas_models, co2_models = schema5._load_model_pairs()
    gas = {PHASE_CANDIDATE: gas_models[PHASE_CANDIDATE]}
    co2 = {PHASE_CANDIDATE: co2_models[PHASE_CANDIDATE]}
    transfer.MODEL_ORDER = (PHASE_CANDIDATE,)
    transfer.CAUSAL_GAS_SETTINGS_BY_CANDIDATE = {
        PHASE_CANDIDATE: schema5.PHYSICS_SETTINGS
    }
    return transfer, gas, co2


def _transient_read(source: Any, path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    error: Exception | None = None
    for attempt in range(8):
        try:
            return source.read(path)
        except OSError as exc:
            error = exc
            time.sleep(min(8.0, 0.25 * (2**attempt)))
    assert error is not None
    raise error


def _injected_mass(experiment: Any, path: Path) -> tuple[Any, float]:
    acquired = experiment.get_datetime(path)
    protocol = experiment.injection_protocol
    try:
        mass = protocol.injected_mass(date=acquired)
    except TypeError:
        mass = protocol.injected_mass(acquired)
    values = np.asarray(mass, dtype=np.float64)
    return acquired, float(np.sum(values))


def _co2_mass_analysis(source: Any) -> darsia.CO2MassAnalysis:
    start = source.experiment.experiment_start
    pressure_temperature = source.experiment.pressure_temperature_protocol
    state = pressure_temperature.get_state(start)
    gradient = pressure_temperature.get_gradient(start)
    return darsia.CO2MassAnalysis(
        baseline=source.rig.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )


def _update_mass_state(analysis: darsia.CO2MassAnalysis, experiment: Any, date: Any) -> None:
    protocol = experiment.pressure_temperature_protocol
    state = protocol.get_state(date)
    gradient = protocol.get_gradient(date)
    analysis.update_state(
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )


def _geometry_volume(source: Any) -> np.ndarray:
    geometry = getattr(source.rig, "geometry", None)
    if geometry is None:
        raise RuntimeError(f"{source.run} has no DarSIA geometry")
    volume = np.asarray(geometry.voxel_volume, dtype=np.float64)
    if volume.ndim == 0:
        volume = np.full(source.baseline.shape[:2], float(volume), dtype=np.float64)
    if volume.shape != source.baseline.shape[:2]:
        raise RuntimeError(
            f"{source.run} geometry shape {volume.shape} differs from "
            f"image shape {source.baseline.shape[:2]}"
        )
    return np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)


def _overflow_arrays(run: str, times_h: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    upper = np.full(times_h.shape, np.nan, dtype=np.float64)
    sigma = np.full(times_h.shape, np.nan, dtype=np.float64)
    if run != "ac60" or not OVERFLOW_PATH.is_file():
        return upper, sigma, None
    payload = json.loads(OVERFLOW_PATH.read_text(encoding="utf-8-sig"))
    tolerance = float(payload.get("match_tolerance_h", 0.12))
    targets = list(payload.get("targets") or [])
    for index, elapsed_h in enumerate(times_h):
        if not targets:
            break
        target = min(targets, key=lambda row: abs(float(row["time_h"]) - elapsed_h))
        if abs(float(target["time_h"]) - elapsed_h) > tolerance:
            continue
        uncertainty = max(float(target["phase_mass_sigma_kg"]), 1.0e-9)
        upper[index] = float(target["gas_mass_kg"]) + uncertainty
        sigma[index] = uncertainty
    return upper, sigma, {
        "path": str(OVERFLOW_PATH),
        "sha256": _sha256(OVERFLOW_PATH),
        "constraint_mode": payload.get("constraint_mode"),
        "matched_frame_count": int(np.count_nonzero(np.isfinite(upper))),
    }


def _temporal_split(frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(frame_count, dtype=np.int32)
    validation = indices[2::4]
    if frame_count >= 8:
        # Keep the two latest observations fully unseen. This turns the late
        # comparison into extrapolation rather than interpolation and is
        # essential when testing post-injection diffusion/dilution.
        validation = np.unique(
            np.concatenate((validation, indices[-2:]))
        ).astype(np.int32)
    if validation.size < 2 and frame_count >= 5:
        validation = indices[[1, frame_count - 2]]
    train = np.setdiff1d(indices, validation, assume_unique=True)
    return train, validation


def _preview_shape(shape: Sequence[int]) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    scale = min(1.0, PREVIEW_MAX_WIDTH / max(1, width))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def prepare_run(dataset: str, run: str, *, force: bool = False) -> PreparedRun:
    cache_dir = LOCAL_ROOT / "prepared" / dataset / run
    cache_path = cache_dir / "histograms.npz"
    manifest_path = cache_dir / "manifest.json"
    fingerprint = _cache_fingerprint(dataset, run)
    if cache_path.is_file() and manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("cache_fingerprint") == fingerprint:
            return _load_prepared(cache_path, manifest)

    transfer, models, co2_models = _configure_phase_runtime()
    requires_clearwater = True
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = cache_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    gas_histograms: list[np.ndarray] = []
    aqueous_histograms: list[np.ndarray] = []
    volume_histograms: list[np.ndarray] = []
    times: list[float] = []
    injected: list[float] = []
    frame_paths: list[str] = []
    acid_samples: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    acid_edges = np.linspace(-2.0, 2.0, ACID_BINS + 1, dtype=np.float64)
    gas_edges = np.linspace(0.0, 1.0, GAS_BINS + 1, dtype=np.float64)

    with transfer._open_source(
        dataset, run, require_clearwater=requires_clearwater
    ) as source:
        preflight = transfer._negative_control_preflight(
            models,
            source,
            co2_models,
            hierarchical_phase=True,
        )
        if not bool(preflight.get("passed")):
            report = cache_dir / "negative_controls.json"
            _atomic_json(report, preflight)
            raise RuntimeError(f"{run} failed phase negative controls; see {report}")
        aqueous_models = transfer._aqueous_models_by_candidate(models, co2_models)
        aqueous_digests = {
            name: transfer._model_digest(model)
            for name, model in aqueous_models.items()
            if model is not None
        }
        aqueous_calibration = dict(
            preflight["aqueous_negative_calibration_by_candidate"]
        )
        gas_calibration = dict(
            preflight["conditional_gas_negative_calibration_by_candidate"]
        )
        evaluation_mask = transfer._phase_evaluation_mask(
            source, hierarchical_phase=True
        )
        causal_context = transfer._causal_topology_context(
            source, (PHASE_CANDIDATE,)
        )
        voxel_volume = _geometry_volume(source)
        baseline_acid = btb_acidification_coordinate(source.baseline)
        mass_analysis = _co2_mass_analysis(source)
        previous: np.ndarray | None = None
        preview_width, preview_height = _preview_shape(source.baseline.shape)
        started = time.perf_counter()

        for index, frame_info in enumerate(source.frames):
            frame_started = time.perf_counter()
            image, correction_rows = _transient_read(source, frame_info.path)
            prepared = transfer._PreparedReviewFrame(
                index=index + 1,
                frame_info=frame_info,
                image=image,
                correction_rows=correction_rows,
                read_seconds=0.0,
            )
            inferred = transfer._infer_review_frame(
                prepared,
                previous,
                source.baseline,
                source.clearwater,
                models,
                aqueous_models,
                aqueous_digests,
                aqueous_calibration,
                evaluation_mask,
                dataset=dataset,
                hierarchical_phase=True,
            )
            states, _causal_seconds = transfer._resolve_review_phase_states(
                inferred,
                gas_calibration,
                evaluation_mask,
                causal_context,
                hierarchical_phase=True,
            )
            state = states[PHASE_CANDIDATE]
            gas_probability = np.clip(
                np.asarray(state.probability, dtype=np.float32), 0.0, 1.0
            )
            aqueous_probability = np.clip(
                np.asarray(state.aqueous_probability, dtype=np.float32), 0.0, 1.0
            )
            co2_probability = np.clip(
                gas_probability + aqueous_probability, 0.0, 1.0
            )
            acidification = np.asarray(
                btb_acidification_coordinate(image) - baseline_acid,
                dtype=np.float32,
            )
            acquired, injected_mass = _injected_mass(source.experiment, frame_info.path)
            _update_mass_state(mass_analysis, source.experiment, acquired)
            density = np.asarray(mass_analysis.density_gaseous_co2, dtype=np.float64)
            solubility = np.asarray(mass_analysis.solubility_co2, dtype=np.float64)
            valid = (
                np.asarray(evaluation_mask, dtype=bool)
                & np.isfinite(acidification)
                & np.isfinite(gas_probability)
                & np.isfinite(co2_probability)
                & (voxel_volume > 0.0)
            )
            co2_gate = _smoothstep(co2_probability, 0.15, 0.50)
            acid_values = np.clip(
                acidification[valid], acid_edges[0] + 1.0e-8, acid_edges[-1] - 1.0e-8
            )
            gas_values = np.clip(
                gas_probability[valid], gas_edges[0] + 1.0e-8, gas_edges[-1] - 1.0e-8
            )
            gas_weight = voxel_volume[valid] * density[valid]
            aqueous_weight = (
                voxel_volume[valid] * solubility[valid] * co2_gate[valid]
            )
            gas_hist, _, _ = np.histogram2d(
                acid_values,
                gas_values,
                bins=(acid_edges, gas_edges),
                weights=gas_weight,
            )
            aqueous_hist, _, _ = np.histogram2d(
                acid_values,
                gas_values,
                bins=(acid_edges, gas_edges),
                weights=aqueous_weight,
            )
            pore_hist, _, _ = np.histogram2d(
                acid_values,
                gas_values,
                bins=(acid_edges, gas_edges),
                weights=voxel_volume[valid],
            )
            gas_histograms.append(gas_hist)
            aqueous_histograms.append(aqueous_hist)
            volume_histograms.append(pore_hist)
            times.append(float(frame_info.elapsed_h))
            injected.append(injected_mass)
            frame_paths.append(str(frame_info.path))

            sampled = valid & (co2_probability >= 0.20)
            sample_values = acidification[::8, ::8][sampled[::8, ::8]]
            if sample_values.size:
                acid_samples.append(np.asarray(sample_values, dtype=np.float32))

            rgb = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            rgb_small = cv2.resize(
                rgb,
                (preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )
            preview_path = preview_dir / f"frame_{index:03d}.jpg"
            cv2.imwrite(
                str(preview_path),
                cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 94],
            )
            compact = {
                "acidification": cv2.resize(
                    acidification,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float16),
                "gas_probability": cv2.resize(
                    gas_probability,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float16),
                "co2_probability": cv2.resize(
                    co2_probability,
                    (preview_width, preview_height),
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float16),
            }
            _atomic_npz(preview_dir / f"frame_{index:03d}.npz", **compact)
            diagnostics.append(
                {
                    "frame_index": index,
                    "elapsed_h": float(frame_info.elapsed_h),
                    "source_path": str(frame_info.path),
                    "injected_mass_kg": injected_mass,
                    "mean_co2_probability": float(np.mean(co2_probability[valid])),
                    "mean_gas_probability": float(np.mean(gas_probability[valid])),
                    "active_pore_volume_m3": float(np.sum(voxel_volume[valid])),
                    "seconds": time.perf_counter() - frame_started,
                }
            )
            previous = image
            print(
                f"[{dataset}/{run}] prepared {index + 1}/{len(source.frames)} "
                f"t={frame_info.elapsed_h:.3f} h",
                flush=True,
            )

        source_contract = {
            "phase_evaluation_domain": source.phase_evaluation_domain,
            "phase_quiz_domain": source.phase_quiz_domain,
            "correction_contract": dict(source.correction_contract),
            "geometry_class": type(source.rig.geometry).__name__,
            "geometry_depth_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.asarray(getattr(source.rig.geometry.depth, "img", source.rig.geometry.depth), dtype=np.float32)
                ).tobytes()
            ).hexdigest(),
            "preparation_seconds": time.perf_counter() - started,
        }

    all_samples = (
        np.concatenate(acid_samples).astype(np.float64, copy=False)
        if acid_samples
        else np.array([0.0, 1.0], dtype=np.float64)
    )
    quantile_values = np.quantile(
        all_samples[np.isfinite(all_samples)],
        (0.01, 0.05, 0.20, 0.40, 0.50, 0.75, 0.90, 0.99, 0.995),
    )
    acid_quantiles = {
        name: float(value)
        for name, value in zip(
            ("q01", "q05", "q20", "q40", "q50", "q75", "q90", "q99", "q995"),
            quantile_values,
        )
    }
    times_array = np.asarray(times, dtype=np.float64)
    train, validation = _temporal_split(len(times_array))
    overflow_upper, overflow_sigma, overflow_contract = _overflow_arrays(
        run, times_array
    )
    _atomic_npz(
        cache_path,
        times_h=times_array,
        injected_mass_kg=np.asarray(injected, dtype=np.float64),
        gas_weight_hist=np.stack(gas_histograms),
        aqueous_weight_hist=np.stack(aqueous_histograms),
        pore_volume_hist=np.stack(volume_histograms),
        acid_edges=acid_edges,
        gas_edges=gas_edges,
        overflow_gas_upper_kg=overflow_upper,
        overflow_sigma_kg=overflow_sigma,
        train_indices=train,
        validation_indices=validation,
    )
    manifest = {
        "schema": CACHE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "dataset": dataset,
        "run": run,
        "created_utc": _utc_now(),
        "cache_fingerprint": fingerprint,
        "phase_candidate": PHASE_CANDIDATE,
        "frame_paths": frame_paths,
        "acid_quantiles": acid_quantiles,
        "source_contract": source_contract,
        "overflow_contract": overflow_contract,
        "frame_diagnostics": diagnostics,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "mass_contract": {
            "gas": "rho_g(P,T,y) * Sg",
            "aqueous": "solubility(P,T,y) * Caq * (1-Sg)",
            "integration": "DarSIA run-native pixel_area * depth * porosity",
            "depth_optimized": False,
            "hidden_3d_path_inferred": False,
            "visible_slice_assumption": (
                "the corrected camera slice is locally representative across the "
                "measured depth map; out-of-plane topology remains structural uncertainty"
            ),
        },
    }
    _atomic_json(manifest_path, manifest)
    return _load_prepared(cache_path, manifest)


def _load_prepared(cache_path: Path, manifest: Mapping[str, Any]) -> PreparedRun:
    with np.load(cache_path, allow_pickle=False) as data:
        values = {name: np.asarray(data[name]) for name in data.files}
    train_indices, validation_indices = _temporal_split(len(values["times_h"]))
    return PreparedRun(
        dataset=str(manifest["dataset"]),
        run=str(manifest["run"]),
        times_h=values["times_h"],
        injected_mass_kg=values["injected_mass_kg"],
        gas_weight_hist=values["gas_weight_hist"],
        aqueous_weight_hist=values["aqueous_weight_hist"],
        pore_volume_hist=values["pore_volume_hist"],
        acid_edges=values["acid_edges"],
        gas_edges=values["gas_edges"],
        overflow_gas_upper_kg=values["overflow_gas_upper_kg"],
        overflow_sigma_kg=values["overflow_sigma_kg"],
        frame_paths=tuple(str(value) for value in manifest["frame_paths"]),
        train_indices=train_indices,
        validation_indices=validation_indices,
        acid_quantiles={
            str(key): float(value)
            for key, value in dict(manifest["acid_quantiles"]).items()
        },
        cache_fingerprint=str(manifest["cache_fingerprint"]),
    )


def _titration_table(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pH_values = np.linspace(*PH_BOUNDS[dataset], 121, dtype=np.float64)
    yellow_values = np.linspace(0.0, 1.0, 1025, dtype=np.float64)
    table = np.empty((len(pH_values), len(yellow_values)), dtype=np.float64)
    for index, initial_pH in enumerate(pH_values):
        flash = TitrationFlash(
            min_value_aq=0.0,
            max_value_aq=1.0,
            min_value_g=0.0,
            max_value_g=1.0,
            initial_pH=float(initial_pH),
            n_lut=384,
            aqueous_blend=0.0,
        )
        table[index] = np.interp(
            yellow_values,
            flash._lut_y,
            flash._lut_caq,
        )
    return pH_values, yellow_values, table


def _interpolate_titration_curve(
    initial_pH: float,
    pH_values: np.ndarray,
    yellow_values: np.ndarray,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.interp(initial_pH, pH_values, np.arange(len(pH_values)))
    lower = int(np.clip(math.floor(position), 0, len(pH_values) - 1))
    upper = min(lower + 1, len(pH_values) - 1)
    fraction = float(position - lower)
    curve = (1.0 - fraction) * table[lower] + fraction * table[upper]
    return yellow_values, curve


def _acid_bounds(prepared: PreparedRun) -> tuple[float, float, float, float]:
    q = prepared.acid_quantiles
    low_min = float(min(q["q01"], q["q20"]))
    low_max = float(max(q["q40"], low_min + 1.0e-3))
    terminal = float(max(q["q995"], q["q90"], low_max + 2.0e-3))
    minimum_width = max(1.0e-3, 0.05 * (terminal - low_min))
    return low_min, low_max, terminal, minimum_width


def suggest_params(
    trial: optuna.Trial,
    prepared: PreparedRun,
    model: str,
) -> dict[str, float]:
    low_min, low_max, terminal, minimum_width = _acid_bounds(prepared)
    acid_low_fraction = trial.suggest_float("acid_low_fraction", 0.0, 1.0)
    acid_low = low_min + acid_low_fraction * (low_max - low_min)
    available = max(minimum_width, terminal - acid_low)
    acid_width_fraction = trial.suggest_float("acid_width_fraction", 0.0, 1.0)
    acid_width = minimum_width + acid_width_fraction * (
        max(minimum_width * 1.001, available) - minimum_width
    )
    gas_onset_fraction = trial.suggest_float("gas_onset_fraction", 0.0, 1.0)
    gas_onset = 0.02 + gas_onset_fraction * (0.70 - 0.02)
    gas_width_fraction = trial.suggest_float("gas_width_fraction", 0.0, 1.0)
    gas_width = 0.03 + gas_width_fraction * (1.0 - gas_onset - 0.03)
    params = {
        "acid_low": acid_low,
        "acid_high": acid_low + acid_width,
        "gas_onset": gas_onset,
        "gas_high": gas_onset + gas_width,
        "max_gas_saturation": trial.suggest_float(
            "max_gas_saturation", 0.05, 0.85
        ),
    }
    if model.startswith("titration"):
        params["initial_pH"] = trial.suggest_float(
            "initial_pH", *PH_BOUNDS[prepared.dataset]
        )
    elif model == "monotone_power":
        params["aqueous_gamma"] = trial.suggest_float(
            "aqueous_gamma", 0.35, 3.0, log=True
        )
    if model.endswith("_diffusion"):
        params["diffusion_beta"] = trial.suggest_float(
            "diffusion_beta", 0.0, 2.0
        )
    return params


def _diffusion_dilution_factor(
    prepared: PreparedRun,
    beta: float,
) -> np.ndarray:
    """Return a post-injection dilution factor from detected plume growth.

    The factor is one during injection. Afterwards, a larger solubility-weighted
    affected pore volume lowers the mean aqueous concentration. This is a
    diagnostic physical hypothesis, not a replacement for the optical signal.
    """

    factor = np.ones_like(prepared.times_h, dtype=np.float64)
    if beta <= 0.0 or not len(factor):
        return factor
    final_injected = float(np.max(prepared.injected_mass_kg))
    if final_injected <= 0.0:
        return factor
    completed = np.flatnonzero(prepared.injected_mass_kg >= 0.9999 * final_injected)
    if not completed.size:
        return factor
    injection_end = int(completed[0])
    active_capacity = np.sum(
        prepared.aqueous_weight_hist,
        axis=(1, 2),
        dtype=np.float64,
    )
    reference = max(float(active_capacity[injection_end]), 1.0e-15)
    post = np.arange(len(factor)) > injection_end
    growth = np.maximum(active_capacity[post] / reference, 1.0)
    factor[post] = np.clip(np.power(growth, -float(beta)), 0.02, 1.0)
    return factor


def evaluate_mass(
    prepared: PreparedRun,
    model: str,
    params: Mapping[str, float],
    *,
    titration_lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    base_model = model.removesuffix("_diffusion")
    acid_fraction = _smoothstep(
        prepared.acid_centers,
        float(params["acid_low"]),
        float(params["acid_high"]),
    )
    if base_model == "simple_flash":
        concentration = acid_fraction
    elif base_model == "monotone_power":
        concentration = np.power(
            acid_fraction,
            float(params["aqueous_gamma"]),
        )
    elif base_model == "titration":
        if titration_lookup is None:
            raise ValueError("Titration evaluation requires a lookup table")
        pH_values, yellow_values, table = titration_lookup
        yellow_grid, curve = _interpolate_titration_curve(
            float(params["initial_pH"]),
            pH_values,
            yellow_values,
            table,
        )
        concentration = np.interp(acid_fraction, yellow_grid, curve)
    else:
        raise ValueError(f"Unknown model {model!r}")
    saturation = float(params["max_gas_saturation"]) * _smoothstep(
        prepared.gas_centers,
        float(params["gas_onset"]),
        float(params["gas_high"]),
    )
    gas_mass = np.sum(
        prepared.gas_weight_hist * saturation[None, None, :],
        axis=(1, 2),
        dtype=np.float64,
    )
    aqueous_mass = np.sum(
        prepared.aqueous_weight_hist
        * concentration[None, :, None]
        * (1.0 - saturation[None, None, :]),
        axis=(1, 2),
        dtype=np.float64,
    )
    dilution = _diffusion_dilution_factor(
        prepared,
        float(params.get("diffusion_beta", 0.0)),
    )
    aqueous_mass *= dilution
    return {
        "gas_mass_kg": gas_mass,
        "aqueous_mass_kg": aqueous_mass,
        "total_mass_kg": gas_mass + aqueous_mass,
        "acid_concentration_curve": concentration,
        "gas_saturation_curve": saturation,
        "diffusion_dilution_factor": dilution,
    }


def _mass_loss(
    prepared: PreparedRun,
    result: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float]:
    expected = prepared.injected_mass_kg[indices]
    detected = np.asarray(result["total_mass_kg"])[indices]
    final_mass = max(float(np.max(prepared.injected_mass_kg)), 1.0e-9)
    scale = np.maximum(expected, 0.05 * final_mass)
    relative = (detected - expected) / scale
    mae = float(np.mean(np.abs(relative)))
    rmse = float(np.sqrt(np.mean(np.square(relative))))

    overflow_mask = (
        np.isfinite(prepared.overflow_gas_upper_kg[indices])
        & np.isfinite(prepared.overflow_sigma_kg[indices])
    )
    overflow_loss = 0.0
    overflow_violations = 0
    if np.any(overflow_mask):
        selected = indices[overflow_mask]
        violation = np.maximum(
            np.asarray(result["gas_mass_kg"])[selected]
            - prepared.overflow_gas_upper_kg[selected],
            0.0,
        ) / prepared.overflow_sigma_kg[selected]
        overflow_loss = float(np.mean(np.square(np.log1p(violation))))
        overflow_violations = int(np.count_nonzero(violation > 0.0))
    objective = mae + 0.25 * rmse + 1.5 * overflow_loss
    return {
        "objective": objective,
        "normalized_mae": mae,
        "normalized_rmse": rmse,
        "overflow_loss": overflow_loss,
        "overflow_violations": overflow_violations,
    }


def _completed_trials_within_budget(
    study: optuna.Study,
    budget: int,
) -> tuple[list[optuna.trial.FrozenTrial], int]:
    completed_all = sorted(
        (
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(float(trial.value))
        ),
        key=lambda trial: trial.number,
    )
    return completed_all[: max(0, int(budget))], len(completed_all)


def _objective_contract(prepared: PreparedRun, model: str) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema": OBJECTIVE_CONTRACT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "dataset": prepared.dataset,
        "run": prepared.run,
        "model": model,
        "prepared_cache_fingerprint": prepared.cache_fingerprint,
        "train_indices": prepared.train_indices.tolist(),
        "validation_indices": prepared.validation_indices.tolist(),
        "loss": {
            "normalized_mae_weight": 1.0,
            "normalized_rmse_weight": 0.25,
            "overflow_upper_bound_weight": 1.5,
            "mass_scale_floor_final_fraction": 0.05,
        },
        "search": {
            "acid_quantiles": dict(prepared.acid_quantiles),
            "gas_onset": [0.02, 0.70],
            "max_gas_saturation": [0.05, 0.85],
            "diffusion_beta": [0.0, 2.0] if model.endswith("_diffusion") else None,
            "initial_pH": (
                list(PH_BOUNDS[prepared.dataset])
                if model.startswith("titration")
                else None
            ),
        },
        "physics": {
            "exclusive_pore_fractions": True,
            "fixed_run_native_depth": True,
            "diffusion_reference": "first frame at 99.99% final injected mass",
        },
    }
    return contract


def _ensure_study_contract(
    study: optuna.Study,
    expected: Mapping[str, Any],
) -> None:
    existing = study.user_attrs.get("objective_contract")
    if study.trials and existing != expected:
        raise RuntimeError(
            "Existing Optuna study has an incompatible or missing objective "
            "contract. Start a fresh study with --force-study instead of mixing "
            "trials from different objectives."
        )
    if existing != expected:
        study.set_user_attr("objective_contract", dict(expected))


def optimize_model(
    prepared: PreparedRun,
    model: str,
    *,
    trials: int,
    jobs: int,
    force: bool = False,
) -> dict[str, Any]:
    output = LOCAL_ROOT / "studies" / prepared.dataset / prepared.run / model
    output.mkdir(parents=True, exist_ok=True)
    storage_path = output / "study.sqlite3"
    if force:
        storage_path.unlink(missing_ok=True)
    storage_url = f"sqlite:///{storage_path.resolve().as_posix()}"
    lookup = (
        _titration_table(prepared.dataset)
        if model.startswith("titration")
        else None
    )
    sampler = optuna.samplers.TPESampler(
        seed=20260831,
        n_startup_trials=min(256, max(32, trials // 10)),
        multivariate=True,
    )
    study = optuna.create_study(
        study_name=f"{CAMPAIGN_ID}_{prepared.dataset}_{prepared.run}_{model}",
        storage=storage_url,
        load_if_exists=not force,
        direction="minimize",
        sampler=sampler,
    )
    objective_contract = _objective_contract(prepared, model)
    _ensure_study_contract(study, objective_contract)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, prepared, model)
        result = evaluate_mass(
            prepared,
            model,
            params,
            titration_lookup=lookup,
        )
        metrics = _mass_loss(prepared, result, prepared.train_indices)
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("resolved_params", params)
        return float(metrics["objective"])

    remaining = max(0, int(trials) - len(study.trials))
    if remaining:
        optimization_started = time.perf_counter()
        initial_count = len(study.trials)
        progress_lock = threading.Lock()
        last_reported = [initial_count]
        _atomic_json(
            output / "progress.json",
            {
                "status": "running",
                "dataset": prepared.dataset,
                "run": prepared.run,
                "model": model,
                "completed_trials": initial_count,
                "target_trials": int(trials),
                "updated_utc": _utc_now(),
            },
        )

        def progress_callback(
            current_study: optuna.Study,
            current_trial: optuna.trial.FrozenTrial,
        ) -> None:
            with progress_lock:
                completed_count = len(current_study.trials)
                if (
                    completed_count < int(trials)
                    and completed_count - last_reported[0] < 100
                ):
                    return
                last_reported[0] = completed_count
                elapsed = max(time.perf_counter() - optimization_started, 1.0e-9)
                _atomic_json(
                    output / "progress.json",
                    {
                        "status": "running",
                        "dataset": prepared.dataset,
                        "run": prepared.run,
                        "model": model,
                        "completed_trials": completed_count,
                        "target_trials": int(trials),
                        "trials_per_hour_since_resume": 3600.0
                        * max(0, completed_count - initial_count)
                        / elapsed,
                        "best_objective": float(current_study.best_value),
                        "updated_utc": _utc_now(),
                    },
                )

        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=max(1, int(jobs)),
            gc_after_trial=False,
            show_progress_bar=False,
            callbacks=(progress_callback,),
        )
    completed, stored_completed_count = _completed_trials_within_budget(
        study,
        trials,
    )
    if not completed:
        raise RuntimeError(f"{model} produced no completed finite trials")
    # Concurrent workers can finish a few trials beyond the requested budget.
    # Restrict model selection to the same number of completed trials so a
    # faster objective cannot gain an accidental search-budget advantage.
    best_trial = min(completed, key=lambda trial: float(trial.value))
    best_value = float(best_trial.value)
    best_params = dict(best_trial.user_attrs["resolved_params"])
    best_result = evaluate_mass(
        prepared,
        model,
        best_params,
        titration_lookup=lookup,
    )
    training = _mass_loss(prepared, best_result, prepared.train_indices)
    validation = _mass_loss(prepared, best_result, prepared.validation_indices)
    threshold = best_value * 1.05 + 1.0e-12
    near_optimal = [trial for trial in completed if float(trial.value) <= threshold]
    parameter_ranges: dict[str, dict[str, float]] = {}
    for name in best_params:
        values = [
            float(trial.user_attrs["resolved_params"][name])
            for trial in near_optimal
            if name in trial.user_attrs.get("resolved_params", {})
        ]
        if values:
            parameter_ranges[name] = {
                "min": min(values),
                "median": float(np.median(values)),
                "max": max(values),
            }
    gas_fraction = np.divide(
        best_result["gas_mass_kg"],
        best_result["total_mass_kg"],
        out=np.zeros_like(best_result["gas_mass_kg"]),
        where=best_result["total_mass_kg"] > 0.0,
    )
    summary = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "dataset": prepared.dataset,
        "run": prepared.run,
        "model": model,
        "trial_count": len(completed),
        "stored_completed_trial_count": stored_completed_count,
        "selection_trial_budget": int(trials),
        "objective_contract": objective_contract,
        "best_trial_number": int(best_trial.number),
        "best_objective": best_value,
        "best_params": best_params,
        "training_metrics": training,
        "validation_metrics": validation,
        "near_optimal_trial_count_within_5_percent": len(near_optimal),
        "near_optimal_parameter_ranges": parameter_ranges,
        "final_detected_mass_kg": float(best_result["total_mass_kg"][-1]),
        "final_injected_mass_kg": float(prepared.injected_mass_kg[-1]),
        "peak_gas_mass_fraction": float(np.max(gas_fraction)),
        "final_gas_mass_fraction": float(gas_fraction[-1]),
        "final_diffusion_dilution_factor": float(
            best_result["diffusion_dilution_factor"][-1]
        ),
        "phase_identifiability_note": (
            "FF-UM has no overflow phase observation; total mass alone does not "
            "uniquely identify gas saturation versus aqueous concentration."
            if prepared.dataset == "ffum"
            else "AC60 gas mass is additionally constrained by measured overflow upper bounds."
        ),
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "progress.json",
        {
            "status": "complete",
            "dataset": prepared.dataset,
            "run": prepared.run,
            "model": model,
            "completed_trials": len(completed),
            "stored_completed_trials": stored_completed_count,
            "target_trials": int(trials),
            "best_objective": best_value,
            "updated_utc": _utc_now(),
        },
    )
    rows: list[dict[str, Any]] = []
    for index, elapsed_h in enumerate(prepared.times_h):
        rows.append(
            {
                "frame_index": index,
                "split": (
                    "validation"
                    if index in set(prepared.validation_indices.tolist())
                    else "training"
                ),
                "elapsed_h": float(elapsed_h),
                "injected_mass_kg": float(prepared.injected_mass_kg[index]),
                "detected_total_mass_kg": float(best_result["total_mass_kg"][index]),
                "detected_gas_mass_kg": float(best_result["gas_mass_kg"][index]),
                "detected_aqueous_mass_kg": float(best_result["aqueous_mass_kg"][index]),
                "diffusion_dilution_factor": float(
                    best_result["diffusion_dilution_factor"][index]
                ),
                "overflow_gas_upper_kg": (
                    ""
                    if not np.isfinite(prepared.overflow_gas_upper_kg[index])
                    else float(prepared.overflow_gas_upper_kg[index])
                ),
                "source_path": prepared.frame_paths[index],
            }
        )
    with (output / "best_mass_per_frame.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        output / "trials.csv", index=False
    )
    return {"summary": summary, "result": best_result, "output": output}


def _plot_run(
    prepared: PreparedRun,
    results: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(MODELS),
        figsize=(24, 9),
        sharex="col",
        sharey="row",
    )
    validation = set(prepared.validation_indices.tolist())
    for column, model in enumerate(MODELS):
        result = results[model]["result"]
        summary = results[model]["summary"]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(
            prepared.times_h,
            1000.0 * prepared.injected_mass_kg,
            color="black",
            linewidth=2.0,
            label="Injected",
        )
        top.plot(
            prepared.times_h,
            1000.0 * result["total_mass_kg"],
            color="#cc3d3d",
            linewidth=1.8,
            label="Detected total",
        )
        top.scatter(
            prepared.times_h[list(validation)],
            1000.0 * result["total_mass_kg"][list(validation)],
            color="#1d70b8",
            marker="D",
            s=34,
            label="Holdout",
            zorder=4,
        )
        top.set_title(
            f"{model.replace('_', ' ').title()}\n"
            f"holdout NMAE={summary['validation_metrics']['normalized_mae']:.3f}"
        )
        top.set_ylabel("CO2 mass [g]")
        top.grid(alpha=0.25)
        if column == 0:
            top.legend(fontsize=8)

        bottom.stackplot(
            prepared.times_h,
            1000.0 * result["aqueous_mass_kg"],
            1000.0 * result["gas_mass_kg"],
            labels=("Aqueous", "Free gas"),
            colors=("#31d9a0", "#f24c9d"),
            alpha=0.8,
        )
        finite = np.isfinite(prepared.overflow_gas_upper_kg)
        if np.any(finite):
            bottom.plot(
                prepared.times_h[finite],
                1000.0 * prepared.overflow_gas_upper_kg[finite],
                color="#54278f",
                linestyle="--",
                linewidth=1.4,
                label="Overflow gas upper bound",
            )
        bottom.set_xlabel("Time [h]")
        bottom.set_ylabel("Detected phase mass [g]")
        bottom.grid(alpha=0.25)
        if column == 0:
            bottom.legend(fontsize=8)
    figure.suptitle(
        f"{prepared.run.upper()} phase-to-mass pilot | fixed DarSIA depth/porosity",
        fontsize=15,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_transfer_functions(
    prepared: PreparedRun,
    results: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    colors = {
        "simple_flash": "#d95f02",
        "simple_flash_diffusion": "#e6ab02",
        "titration": "#1b9e77",
        "titration_diffusion": "#2ca25f",
        "monotone_power": "#7570b3",
    }
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for model in MODELS:
        result = results[model]["result"]
        label = model.replace("_", " ").title()
        axes[0].plot(
            prepared.acid_centers,
            result["acid_concentration_curve"],
            color=colors[model],
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            prepared.gas_centers,
            result["gas_saturation_curve"],
            color=colors[model],
            linewidth=2.0,
            label=label,
        )
        axes[2].plot(
            prepared.times_h,
            result["diffusion_dilution_factor"],
            color=colors[model],
            linewidth=2.0,
            label=label,
        )
    axes[0].set_title("Optical acidification to aqueous concentration")
    axes[0].set_xlabel("Corrected BTB acidification coordinate")
    axes[0].set_ylabel("Normalized aqueous concentration")
    axes[1].set_title("Free-gas evidence to gas saturation")
    axes[1].set_xlabel("Hierarchical gas evidence")
    axes[1].set_ylabel("Gas saturation")
    axes[2].set_title("Post-injection diffusion/dilution")
    axes[2].set_xlabel("Time [h]")
    axes[2].set_ylabel("Aqueous concentration multiplier")
    for axis in axes:
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.suptitle(
        f"{prepared.run.upper()} optimized physical transfer functions",
        fontsize=15,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _phase_rgb(
    acidification: np.ndarray,
    gas_probability: np.ndarray,
    co2_probability: np.ndarray,
    model: str,
    params: Mapping[str, float],
    titration_lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    dilution_factor: float = 1.0,
) -> np.ndarray:
    base_model = model.removesuffix("_diffusion")
    acid = _smoothstep(
        acidification,
        float(params["acid_low"]),
        float(params["acid_high"]),
    )
    if base_model == "simple_flash":
        concentration = acid
    elif base_model == "monotone_power":
        concentration = np.power(acid, float(params["aqueous_gamma"]))
    else:
        assert titration_lookup is not None
        yellow_grid, curve = _interpolate_titration_curve(
            float(params["initial_pH"]), *titration_lookup
        )
        concentration = np.interp(acid, yellow_grid, curve)
    concentration *= (
        _smoothstep(co2_probability, 0.15, 0.50) * float(dilution_factor)
    )
    saturation = float(params["max_gas_saturation"]) * _smoothstep(
        gas_probability,
        float(params["gas_onset"]),
        float(params["gas_high"]),
    )
    aqueous_color = np.array((49, 217, 160), dtype=np.float32)
    gas_color = np.array((242, 76, 157), dtype=np.float32)
    total = np.clip(concentration + saturation, 0.0, 1.0)
    blend = np.divide(
        saturation,
        concentration + saturation,
        out=np.zeros_like(saturation),
        where=(concentration + saturation) > 1.0e-6,
    )
    color = (
        (1.0 - blend)[..., None] * aqueous_color
        + blend[..., None] * gas_color
    )
    return np.clip(color * total[..., None], 0.0, 255.0).astype(np.uint8)


def _write_review_movie(
    prepared: PreparedRun,
    results: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    preview_dir = LOCAL_ROOT / "prepared" / prepared.dataset / prepared.run / "preview"
    lookup = _titration_table(prepared.dataset)
    frames: list[np.ndarray] = []
    for index, elapsed_h in enumerate(prepared.times_h):
        source = cv2.imread(str(preview_dir / f"frame_{index:03d}.jpg"))
        if source is None:
            continue
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        with np.load(preview_dir / f"frame_{index:03d}.npz", allow_pickle=False) as values:
            acidification = np.asarray(values["acidification"], dtype=np.float32)
            gas_probability = np.asarray(values["gas_probability"], dtype=np.float32)
            co2_probability = np.asarray(values["co2_probability"], dtype=np.float32)
        panels = [source]
        titles = ["Corrected image"]
        for model in MODELS:
            params = results[model]["summary"]["best_params"]
            dilution = results[model]["result"]["diffusion_dilution_factor"]
            panels.append(
                _phase_rgb(
                    acidification,
                    gas_probability,
                    co2_probability,
                    model,
                    params,
                    lookup if model.startswith("titration") else None,
                    float(dilution[index]),
                )
            )
            titles.append(model.replace("_", " ").title())
        panel_width = 560
        resized: list[np.ndarray] = []
        for panel, title in zip(panels, titles):
            height = max(1, int(round(panel.shape[0] * panel_width / panel.shape[1])))
            rendered = cv2.resize(panel, (panel_width, height), interpolation=cv2.INTER_AREA)
            rendered = cv2.copyMakeBorder(
                rendered, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18)
            )
            cv2.putText(
                rendered,
                title,
                (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            resized.append(rendered)
        canvas = np.concatenate(resized, axis=1)
        cv2.putText(
            canvas,
            f"{prepared.run.upper()} | t={elapsed_h:.3f} h | injected={1000.0 * prepared.injected_mass_kg[index]:.4f} g",
            (15, canvas.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(canvas)
    if not frames:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.5,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _copy_results(dataset: str, run: str, local_output: Path) -> Path:
    destination = OUTPUTS[dataset] / dataset / run
    return publish_directory(local_output, destination)


def run_pilot(
    dataset: str,
    run: str,
    *,
    trials: int,
    jobs: int,
    model_parallelism: int = 3,
    force_prepare: bool = False,
    force_study: bool = False,
) -> dict[str, Any]:
    prepared = prepare_run(dataset, run, force=force_prepare)
    model_results: dict[str, Mapping[str, Any]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(model_parallelism), len(MODELS)))
    ) as executor:
        futures = {
            executor.submit(
                optimize_model,
                prepared,
                model,
                trials=trials,
                jobs=jobs,
                force=force_study,
            ): model
            for model in MODELS
        }
        for future in as_completed(futures):
            model = futures[future]
            model_results[model] = future.result()
    local_output = LOCAL_ROOT / "results" / dataset / run
    local_output.mkdir(parents=True, exist_ok=True)
    _plot_run(prepared, model_results, local_output / "mass_comparison.png")
    _plot_transfer_functions(
        prepared,
        model_results,
        local_output / "optimized_transfer_functions.png",
    )
    _write_review_movie(
        prepared,
        model_results,
        local_output / "phase_to_mass_review.mp4",
    )
    ranked = sorted(
        MODELS,
        key=lambda model: model_results[model]["summary"]["validation_metrics"][
            "objective"
        ],
    )
    summary = {
        "schema": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "dataset": dataset,
        "run": run,
        "phase_candidate": PHASE_CANDIDATE,
        "trial_budget_per_model": trials,
        "model_ranking_by_temporal_holdout": ranked,
        "models": {
            model: model_results[model]["summary"] for model in MODELS
        },
        "scientific_contract": {
            "detector_outputs": "continuous visible-slice CO2 and free-gas evidence",
            "detector_does_not_output": [
                "aqueous concentration",
                "gas saturation",
                "depth",
                "mass",
            ],
            "darsia_outputs": [
                "pressure-temperature gas density",
                "aqueous solubility",
                "porosity-depth-volume integration",
                "gas and aqueous mass",
            ],
            "objective_reference": "run-native injected CO2 mass",
            "overflow_role": (
                "measured gas upper bound"
                if dataset == "ffac"
                else "unavailable; phase split remains underdetermined"
            ),
            "depth_map": "fixed run-native map; not optimized",
            "2d_to_3d": (
                "local extrusion through measured depth and porosity only; no hidden "
                "3D streamline reconstruction from one camera slice"
            ),
        },
    }
    _atomic_json(local_output / "summary.json", summary)
    for model in MODELS:
        model_dir = local_output / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "best_mass_per_frame.csv", "trials.csv"):
            shutil.copy2(model_results[model]["output"] / name, model_dir / name)
    destination = _copy_results(dataset, run, local_output)
    summary["published_to"] = str(destination)
    _atomic_json(local_output / "summary.json", summary)
    _atomic_json(destination / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"), nargs="?", default="run")
    parser.add_argument("--dataset", choices=("ffac", "ffum", "all"), default="all")
    parser.add_argument("--run")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--model-parallelism", type=int, default=3)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-study", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selected = [
        (dataset, run)
        for dataset, run in RUN_SPECS
        if args.dataset == "all" or args.dataset == dataset
    ]
    if args.run:
        selected = [(args.dataset, args.run)]
        if args.dataset == "all":
            raise SystemExit("--run requires a concrete --dataset")
    results: list[dict[str, Any]] = []
    for dataset, run in selected:
        if args.action == "prepare":
            prepared = prepare_run(dataset, run, force=args.force_prepare)
            results.append(
                {
                    "dataset": dataset,
                    "run": run,
                    "frame_count": len(prepared.times_h),
                    "acid_quantiles": prepared.acid_quantiles,
                    "cache_fingerprint": prepared.cache_fingerprint,
                }
            )
        else:
            results.append(
                run_pilot(
                    dataset,
                    run,
                    trials=max(1, int(args.trials)),
                    jobs=max(1, int(args.jobs)),
                    model_parallelism=max(1, int(args.model_parallelism)),
                    force_prepare=args.force_prepare,
                    force_study=args.force_study,
                )
            )
    print(json.dumps(results, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
