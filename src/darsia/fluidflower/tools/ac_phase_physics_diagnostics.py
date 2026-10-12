"""Diagnose whether inferred free CO2 follows source and buoyancy constraints.

The diagnostic is intentionally read-only. It applies completed calibration
candidates in memory, re-evaluates the calibration frames, and quantifies:

* free-gas mass below the relevant injector heights;
* vertical gas centroid and lower gas front;
* gas near an injector before that injector is activated;
* newly appearing gas below the latest activated injector that is not connected
  to a source or earlier support within the camera-observed 2D slice.

No spatial constraint is applied here. The outputs are evidence for deciding
whether a time-dependent physical constraint is warranted. Connectivity is a
2D image diagnostic only: gas that appears disconnected in the observed slice
may be part of a continuous phase outside that slice in the 3D volume.
"""

from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import gc
import json
import math
import os
import socket
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darsia.presets.workflows.rig import Rig


REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.ac_locked_titration_flash import (  # noqa: E402
    BOUNDS_FILE,
    DEFAULT_LOGS_ROOT,
    LOCKED_FLASH,
    RUNS,
    VARIANT,
    _campaign_env as _titration_campaign_env,
)
from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (  # noqa: E402
    _mass_result_for_evaluation,
    apply_params,
    build_context,
    load_bounds_map,
)
from darsia.fluidflower.tools.spatial_mass_evaluator import (  # noqa: E402
    create_spatial_mass_evaluator,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_LOGS_ROOT / "phase_physics_diagnostics"
CORE_LEVEL = 0.05
SUPPORT_LEVEL = 0.01
AQ_TRACE_LEVEL = 0.01
AQ_ENVELOPE_LEVEL = 0.05
VERTICAL_TOLERANCE_M = 0.03
SOURCE_RADIUS_X_M = 0.04
SOURCE_RADIUS_Y_M = 0.03
HISTORY_DILATION_M = 0.01
TOPOLOGY_SCALE = 0.25


@dataclass(frozen=True)
class InjectionStage:
    index: int
    x_m: float
    y_m: float
    start_h: float
    end_h: float


@dataclass
class TopologyState:
    support: np.ndarray
    core: np.ndarray
    connected: np.ndarray
    suspicious_new_below: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray


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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    import hashlib

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


@contextmanager
def campaign_environment(flash_model: str = "titration") -> Iterator[None]:
    model = str(flash_model).strip().lower()
    if model not in {"titration", "simple"}:
        raise ValueError("flash_model must be 'titration' or 'simple'")
    requested = _titration_campaign_env()
    requested["FFAC_TITRATION_FLASH"] = (
        "on" if model == "titration" else "off"
    )
    if model == "simple":
        requested.pop("FFAC_TITRATION_RECIPE", None)
    relevant = {
        name
        for name in set(os.environ) | set(requested)
        if name.startswith("FFAC_")
        or name
        in {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENCV_NUM_THREADS",
        }
    }
    previous = {name: os.environ.get(name) for name in relevant}
    try:
        for name in relevant:
            if name in requested:
                os.environ[name] = requested[name]
            else:
                os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_config(run: str, config_dir: Path) -> Path:
    path = config_dir / f"{run}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Run config is unavailable: {path}")
    return path


def load_injection_stages(
    run_config: Path,
    experiment_start: datetime,
) -> tuple[InjectionStage, ...]:
    """Load unique injection locations in chronological activation order."""

    with run_config.open("rb") as stream:
        config = tomllib.load(stream)
    protocol_path = Path(config["protocols"]["injection"])
    if not protocol_path.is_absolute():
        protocol_path = REPO_ROOT / protocol_path
    protocol = pd.read_csv(protocol_path)
    protocol["start"] = pd.to_datetime(protocol["start"])
    protocol["end"] = pd.to_datetime(protocol["end"])
    active = protocol.loc[protocol["rate_sccm"].astype(float) > 0.0].copy()
    if active.empty:
        return ()

    origin = pd.Timestamp(experiment_start)
    if origin.tzinfo is not None:
        origin = origin.tz_localize(None)
    locations = list(
        dict.fromkeys(
            zip(
                active["location_x"].astype(float),
                active["location_y"].astype(float),
                strict=True,
            )
        )
    )
    stages: list[InjectionStage] = []
    for index, (x_m, y_m) in enumerate(locations, start=1):
        rows = active.loc[
            np.isclose(active["location_x"].astype(float), x_m)
            & np.isclose(active["location_y"].astype(float), y_m)
        ]
        start = pd.Timestamp(rows["start"].min())
        end = pd.Timestamp(rows["end"].max())
        stages.append(
            InjectionStage(
                index=index,
                x_m=float(x_m),
                y_m=float(y_m),
                start_h=float((start - origin).total_seconds() / 3600.0),
                end_h=float((end - origin).total_seconds() / 3600.0),
            )
        )
    return tuple(sorted(stages, key=lambda stage: stage.start_h))


def stage_state(
    stages: Sequence[InjectionStage],
    elapsed_h: float,
) -> tuple[InjectionStage | None, tuple[InjectionStage, ...]]:
    activated = tuple(stage for stage in stages if stage.start_h <= elapsed_h)
    active = next(
        (
            stage
            for stage in stages
            if stage.start_h <= elapsed_h <= stage.end_h
        ),
        None,
    )
    return active, activated


def _coordinate_axes(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    height, width = np.asarray(reference.img).shape[:2]
    rows = np.column_stack(
        (np.arange(height, dtype=float) + 0.5, np.zeros(height, dtype=float))
    )
    columns = np.column_stack(
        (np.zeros(width, dtype=float), np.arange(width, dtype=float) + 0.5)
    )
    y_m = np.asarray(reference.coordinatesystem.coordinate(rows))[:, 1]
    x_m = np.asarray(reference.coordinatesystem.coordinate(columns))[:, 0]
    return x_m, y_m


def _downsample_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return np.asarray(mask, dtype=bool)
    height, width = mask.shape
    target = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        target,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _resample_axis(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) == size:
        return np.asarray(values, dtype=float)
    positions = np.linspace(0.0, len(values) - 1.0, size)
    return np.interp(positions, np.arange(len(values), dtype=float), values)


def _ellipse_mask(
    x_m: np.ndarray,
    y_m: np.ndarray,
    stage: InjectionStage,
    *,
    radius_x_m: float,
    radius_y_m: float,
) -> np.ndarray:
    xx = (x_m[None, :] - stage.x_m) / max(radius_x_m, 1e-12)
    yy = (y_m[:, None] - stage.y_m) / max(radius_y_m, 1e-12)
    return xx * xx + yy * yy <= 1.0


def _dilate_physical(
    mask: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    if not np.any(mask) or radius_m <= 0.0:
        return np.asarray(mask, dtype=bool)
    dx = float(np.nanmedian(np.abs(np.diff(x_m)))) if len(x_m) > 1 else radius_m
    dy = float(np.nanmedian(np.abs(np.diff(y_m)))) if len(y_m) > 1 else radius_m
    radius_x = max(1, int(math.ceil(radius_m / max(dx, 1e-12))))
    radius_y = max(1, int(math.ceil(radius_m / max(dy, 1e-12))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_x + 1, 2 * radius_y + 1),
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _weighted_row_quantile(
    row_weights: np.ndarray,
    y_m: np.ndarray,
    quantile: float,
) -> float:
    valid = np.isfinite(row_weights) & np.isfinite(y_m) & (row_weights > 0.0)
    if not np.any(valid):
        return float("nan")
    weights = row_weights[valid]
    coordinates = y_m[valid]
    order = np.argsort(coordinates)
    weights = weights[order]
    coordinates = coordinates[order]
    cumulative = np.cumsum(weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(weights) - 1)
    return float(coordinates[index])


def phase_physics_metrics(
    *,
    saturation_g: np.ndarray,
    mass_g: np.ndarray,
    voxel_volume: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    stages: Sequence[InjectionStage],
    elapsed_h: float,
    previous_support: np.ndarray | None,
    core_level: float = CORE_LEVEL,
    support_level: float = SUPPORT_LEVEL,
    vertical_tolerance_m: float = VERTICAL_TOLERANCE_M,
    source_radius_x_m: float = SOURCE_RADIUS_X_M,
    source_radius_y_m: float = SOURCE_RADIUS_Y_M,
    history_dilation_m: float = HISTORY_DILATION_M,
    topology_scale: float = TOPOLOGY_SCALE,
) -> tuple[dict[str, Any], TopologyState]:
    """Compute mass-weighted and topology-aware free-gas diagnostics."""

    saturation = np.asarray(saturation_g, dtype=float)
    gas_mass_density = np.asarray(mass_g, dtype=float)
    volume = np.broadcast_to(np.asarray(voxel_volume, dtype=float), saturation.shape)
    if gas_mass_density.shape != saturation.shape:
        raise ValueError(
            f"Gas mass shape {gas_mass_density.shape} != saturation shape {saturation.shape}."
        )
    if len(x_m) != saturation.shape[1] or len(y_m) != saturation.shape[0]:
        raise ValueError("Coordinate axes do not match the phase field shape.")

    finite = np.isfinite(saturation) & np.isfinite(gas_mass_density) & np.isfinite(volume)
    mass = np.where(finite & (gas_mass_density > 0.0), gas_mass_density * volume, 0.0)
    total_mass = float(np.sum(mass, dtype=np.float64))
    row_mass = np.sum(mass, axis=1, dtype=np.float64)
    column_mass = np.sum(mass, axis=0, dtype=np.float64)
    centroid_y = _safe_fraction(float(np.dot(row_mass, y_m)), total_mass)
    centroid_x = _safe_fraction(float(np.dot(column_mass, x_m)), total_mass)
    lower_front_y = _weighted_row_quantile(row_mass, y_m, 0.05)

    active, activated = stage_state(stages, elapsed_h)
    reference_stage = active or (activated[-1] if activated else None)
    lowest_stage = min(activated, key=lambda stage: stage.y_m) if activated else None

    def mass_below(stage: InjectionStage | None) -> float:
        if stage is None:
            return 0.0
        rows = y_m < stage.y_m - vertical_tolerance_m
        return float(np.sum(mass[rows, :], dtype=np.float64))

    below_reference = mass_below(reference_stage)
    below_lowest = mass_below(lowest_stage)

    support_full = finite & (saturation >= support_level)
    core_full = finite & (saturation >= core_level)
    support = _downsample_mask(support_full, topology_scale)
    core = _downsample_mask(core_full, topology_scale)
    x_topology = _resample_axis(x_m, support.shape[1])
    y_topology = _resample_axis(y_m, support.shape[0])
    if previous_support is not None and previous_support.shape != support.shape:
        raise ValueError(
            f"Previous support shape {previous_support.shape} != {support.shape}."
        )
    history = (
        _dilate_physical(
            previous_support,
            x_topology,
            y_topology,
            history_dilation_m,
        )
        if previous_support is not None
        else np.zeros_like(support)
    )

    source_masks = [
        _ellipse_mask(
            x_topology,
            y_topology,
            stage,
            radius_x_m=source_radius_x_m,
            radius_y_m=source_radius_y_m,
        )
        for stage in activated
    ]
    source_union = (
        np.logical_or.reduce(source_masks)
        if source_masks
        else np.zeros_like(support)
    )
    count, labels = cv2.connectedComponents(
        support.astype(np.uint8),
        connectivity=8,
    )
    accepted_labels: set[int] = set()
    source_labels = set(np.unique(labels[source_union & support]).tolist())
    history_labels = set(np.unique(labels[history & support]).tolist())
    accepted_labels.update(label for label in source_labels | history_labels if label > 0)
    connected = np.isin(labels, list(accepted_labels)) if accepted_labels else np.zeros_like(support)
    unconnected_in_observed_slice_core = core & ~connected
    new_core = core & ~history

    if reference_stage is not None:
        below_reference_topology = (
            y_topology[:, None]
            < reference_stage.y_m - vertical_tolerance_m
        )
        reference_source = _ellipse_mask(
            x_topology,
            y_topology,
            reference_stage,
            radius_x_m=source_radius_x_m,
            radius_y_m=source_radius_y_m,
        )
        suspicious = (
            new_core
            & unconnected_in_observed_slice_core
            & below_reference_topology
            & ~reference_source
        )
    else:
        suspicious = np.zeros_like(core)

    core_count = int(np.count_nonzero(core))
    support_count = int(np.count_nonzero(support))
    future_stages = [stage for stage in stages if stage.start_h > elapsed_h]
    future_mass = 0.0
    if future_stages and total_mass > 0.0:
        future_mask = np.logical_or.reduce(
            [
                _ellipse_mask(
                    x_m,
                    y_m,
                    stage,
                    radius_x_m=source_radius_x_m,
                    radius_y_m=source_radius_y_m,
                )
                for stage in future_stages
            ]
        )
        future_mass = float(np.sum(mass[future_mask], dtype=np.float64))

    metrics = {
        "active_stage": active.index if active is not None else 0,
        "latest_activated_stage": reference_stage.index if reference_stage is not None else 0,
        "reference_injector_x_m": (
            reference_stage.x_m if reference_stage is not None else float("nan")
        ),
        "reference_injector_y_m": (
            reference_stage.y_m if reference_stage is not None else float("nan")
        ),
        "gas_mass_kg": total_mass,
        "gas_centroid_x_m": centroid_x if total_mass > 0.0 else float("nan"),
        "gas_centroid_y_m": centroid_y if total_mass > 0.0 else float("nan"),
        "gas_lower_front_q05_y_m": lower_front_y,
        "gas_mass_below_reference_kg": below_reference,
        "gas_mass_below_reference_fraction": _safe_fraction(below_reference, total_mass),
        "gas_mass_below_lowest_activated_source_kg": below_lowest,
        "gas_mass_below_lowest_activated_source_fraction": _safe_fraction(
            below_lowest, total_mass
        ),
        "gas_mass_near_future_source_kg": future_mass,
        "gas_mass_near_future_source_fraction": _safe_fraction(future_mass, total_mass),
        "gas_support_pixels": support_count,
        "gas_core_pixels": core_count,
        "gas_connected_core_fraction": _safe_fraction(
            int(np.count_nonzero(core & connected)), core_count
        ),
        "gas_observed_slice_unconnected_core_fraction": _safe_fraction(
            int(np.count_nonzero(unconnected_in_observed_slice_core)), core_count
        ),
        "gas_new_observed_slice_unconnected_below_pixels": int(
            np.count_nonzero(suspicious)
        ),
        "gas_new_observed_slice_unconnected_below_fraction": _safe_fraction(
            int(np.count_nonzero(suspicious)), core_count
        ),
        # Compatibility aliases for diagnostic CSV files generated before the
        # 3D interpretation caveat was made explicit.
        "gas_disconnected_core_fraction": _safe_fraction(
            int(np.count_nonzero(unconnected_in_observed_slice_core)), core_count
        ),
        "gas_new_core_fraction": _safe_fraction(
            int(np.count_nonzero(new_core)), core_count
        ),
        "gas_new_disconnected_below_pixels": int(np.count_nonzero(suspicious)),
        "gas_new_disconnected_below_fraction": _safe_fraction(
            int(np.count_nonzero(suspicious)), core_count
        ),
        "gas_support_component_count": max(0, count - 1),
        "gas_source_connected_component_count": len(source_labels - {0}),
        "gas_history_connected_component_count": len(history_labels - {0}),
    }
    return metrics, TopologyState(
        support=support,
        core=core,
        connected=connected,
        suspicious_new_below=suspicious,
        x_m=x_topology,
        y_m=y_topology,
    )


def _voxel_volume(context: Any, shape: tuple[int, int]) -> np.ndarray:
    context.geometry._prepare_cached_voxel_volume(list(shape))
    return np.broadcast_to(
        np.asarray(context.geometry.cached_voxel_volume, dtype=float),
        shape,
    )


def _candidate_path(campaign_root: Path, run: str, variant: str = VARIANT) -> Path:
    path = campaign_root / variant / run / f"final_full_scale_{run}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Locked calibration candidate is unavailable: {path}")
    return path


def _candidate_metric(
    candidate: dict[str, Any],
    elapsed_h: float,
) -> tuple[float, dict[str, float]]:
    available: list[tuple[float, dict[str, float]]] = []
    for key, raw in (candidate.get("metrics") or {}).items():
        try:
            time_h = float(str(key).removesuffix("h"))
        except (TypeError, ValueError):
            continue
        if not isinstance(raw, Mapping):
            continue
        values: dict[str, float] = {}
        for name, value in raw.items():
            if value is None:
                continue
            try:
                values[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        required = {"total_full", "gaseous_full", "aqueous_full"}
        if not required.issubset(values):
            continue
        available.append((time_h, values))
    if not available:
        raise RuntimeError("Calibration candidate has no time-resolved metrics.")
    matched_time, values = min(
        available,
        key=lambda item: abs(item[0] - elapsed_h),
    )
    if abs(matched_time - elapsed_h) > 0.02:
        raise RuntimeError(
            f"No candidate metric matches {elapsed_h:.6f} h; nearest is "
            f"{matched_time:.6f} h."
        )
    return matched_time, values


def _relative_error(actual: float, expected: float) -> float:
    return float(abs(actual - expected) / max(abs(expected), 1e-12))


def _draw_level_outline(
    rgb: np.ndarray,
    values: np.ndarray,
    level: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    mask = (np.isfinite(values) & (values >= level)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(
            rgb,
            contours,
            -1,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _render_diagnostic_frame(
    image: np.ndarray,
    concentration_aq: np.ndarray,
    state: TopologyState,
    *,
    run: str,
    elapsed_h: float,
    stages: Sequence[InjectionStage],
    width: int = 720,
) -> np.ndarray:
    source = np.asarray(image)
    if source.dtype.kind == "f":
        finite = source[np.isfinite(source)]
        scale = 255.0 if finite.size and float(np.nanmax(finite)) <= 1.5 else 1.0
        source = np.clip(source * scale, 0.0, 255.0)
    rgb = np.asarray(source[..., :3], dtype=np.uint8)
    height = max(1, int(round(width * rgb.shape[0] / rgb.shape[1])))
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    aqueous_values = cv2.resize(
        np.asarray(concentration_aq, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    aqueous = np.isfinite(aqueous_values) & (
        aqueous_values >= AQ_ENVELOPE_LEVEL
    )
    core = cv2.resize(
        state.core.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    suspicious = cv2.resize(
        state.suspicious_new_below.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    for mask, color, alpha in (
        (aqueous, np.array([20, 220, 110]), 0.12),
        (core, np.array([245, 65, 55]), 0.28),
        (suspicious, np.array([230, 30, 230]), 0.75),
    ):
        if np.any(mask):
            rgb[mask] = np.rint(
                (1.0 - alpha) * rgb[mask].astype(float) + alpha * color
            ).astype(np.uint8)

    # The production contact sheets use 0.05 as their visible aqueous envelope.
    # A separate low-level trace shows whether apparently omitted yellow areas
    # carry weak, non-zero concentration rather than being entirely undetected.
    _draw_level_outline(rgb, aqueous_values, AQ_TRACE_LEVEL, (30, 205, 235), 1)
    _draw_level_outline(rgb, aqueous_values, AQ_ENVELOPE_LEVEL, (20, 235, 105), 2)

    x_min, x_max = float(np.min(state.x_m)), float(np.max(state.x_m))
    y_min, y_max = float(np.min(state.y_m)), float(np.max(state.y_m))
    active, activated = stage_state(stages, elapsed_h)
    for stage in stages:
        x_px = int(round((stage.x_m - x_min) / max(x_max - x_min, 1e-12) * (width - 1)))
        y_px = int(round((y_max - stage.y_m) / max(y_max - y_min, 1e-12) * (height - 1)))
        color = (255, 220, 20) if stage in activated else (150, 150, 150)
        thickness = 3 if active == stage else 2
        cv2.circle(rgb, (x_px, y_px), 8, color, thickness, lineType=cv2.LINE_AA)
    header = np.full((48, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        header,
        f"{run.upper()}  t={elapsed_h:.3f} h  active={active.index if active else 0}",
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Aq 0.01/0.05 | Gas 0.05",
        (430, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 230, 235),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((header, rgb))


def _contact_sheet(frames: list[np.ndarray], path: Path, columns: int = 4) -> None:
    if not frames:
        return
    width = max(frame.shape[1] for frame in frames)
    height = max(frame.shape[0] for frame in frames)
    rows = math.ceil(len(frames) / columns)
    sheet = np.full((rows * height, columns * width, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[row * height : row * height + frame.shape[0], column * width : column * width + frame.shape[1]] = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        if not cv2.imwrite(str(temporary), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write contact sheet: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summarise_run(frame: pd.DataFrame, run: str) -> dict[str, Any]:
    front = frame["gas_lower_front_q05_y_m"].to_numpy(dtype=float)
    finite_front = front[np.isfinite(front)]
    downward_steps = np.diff(finite_front) if len(finite_front) > 1 else np.asarray([])
    summary = {
        "run": run,
        "frames": len(frame),
        "maximum_gas_mass_below_reference_fraction": float(
            frame["gas_mass_below_reference_fraction"].max()
        ),
        "maximum_gas_mass_below_lowest_source_fraction": float(
            frame["gas_mass_below_lowest_activated_source_fraction"].max()
        ),
        "maximum_gas_mass_near_future_source_fraction": float(
            frame["gas_mass_near_future_source_fraction"].max()
        ),
        "maximum_observed_slice_unconnected_core_fraction": float(
            frame["gas_observed_slice_unconnected_core_fraction"].max()
        ),
        "maximum_new_observed_slice_unconnected_below_fraction": float(
            frame["gas_new_observed_slice_unconnected_below_fraction"].max()
        ),
        "median_aqueous_mass_below_envelope_fraction": float(
            frame["aqueous_mass_below_envelope_fraction"].median()
        ),
        "maximum_aqueous_mass_below_envelope_fraction": float(
            frame["aqueous_mass_below_envelope_fraction"].max()
        ),
        "median_aqueous_mass_below_trace_fraction": float(
            frame["aqueous_mass_below_trace_fraction"].median()
        ),
        "maximum_aqueous_trace_only_pixel_fraction": float(
            frame["aqueous_trace_only_pixel_fraction"].max()
        ),
        "largest_downward_lower_front_step_m": (
            float(min(0.0, np.min(downward_steps))) if downward_steps.size else 0.0
        ),
    }
    warnings: list[str] = []
    if summary["maximum_gas_mass_below_lowest_source_fraction"] > 0.005:
        warnings.append("gas-mass-below-lowest-source")
    if summary["maximum_gas_mass_near_future_source_fraction"] > 0.01:
        warnings.append("gas-near-future-source")
    if summary["largest_downward_lower_front_step_m"] < -0.015:
        warnings.append("downward-lower-front-step")
    reviews: list[str] = []
    if summary["maximum_new_observed_slice_unconnected_below_fraction"] > 0.01:
        reviews.append("new-gas-below-source-not-connected-in-observed-slice")
    summary["warning_count"] = len(warnings)
    summary["warnings"] = ";".join(warnings)
    summary["review_count"] = len(reviews)
    summary["reviews"] = ";".join(reviews)
    return summary


def _plot_grid(
    per_frame: pd.DataFrame,
    stages_by_run: dict[str, tuple[InjectionStage, ...]],
    runs: Sequence[str],
    output: Path,
    flash_model: str = "titration",
) -> None:
    fig, axes = plt.subplots(
        len(runs),
        3,
        figsize=(17, max(4.0, 3.15 * len(runs))),
        squeeze=False,
    )
    for row, run in enumerate(runs):
        frame = per_frame.loc[per_frame["run"] == run].sort_values("elapsed_h")
        time_h = frame["elapsed_h"].to_numpy(dtype=float)
        stages = stages_by_run[run]
        axis = axes[row, 0]
        axis.plot(
            time_h,
            100.0 * frame["gas_mass_below_reference_fraction"],
            color="#c23b32",
            linewidth=1.8,
            label="Below latest source",
        )
        axis.plot(
            time_h,
            100.0 * frame["gas_mass_near_future_source_fraction"],
            color="#7147a8",
            linewidth=1.6,
            label="Near future source",
        )
        axis.set_ylabel(f"{run.upper()}\nGas mass (%)")
        axis.legend(frameon=False, fontsize=7, loc="upper right")

        axis = axes[row, 1]
        axis.plot(
            time_h,
            frame["gas_centroid_y_m"],
            color="#202124",
            linewidth=1.8,
            label="Centroid",
        )
        axis.plot(
            time_h,
            frame["gas_lower_front_q05_y_m"],
            color="#d38b18",
            linewidth=1.5,
            label="Lower 5% front",
        )
        for stage in stages:
            axis.axhline(stage.y_m, color="#7f8c8d", linewidth=0.8, linestyle="--")
        axis.set_ylabel("Height (m)")
        axis.legend(frameon=False, fontsize=7, loc="upper right")

        axis = axes[row, 2]
        axis.plot(
            time_h,
            100.0 * frame["gas_observed_slice_unconnected_core_fraction"],
            color="#31688e",
            linewidth=1.7,
            label="Not connected in observed slice",
        )
        axis.plot(
            time_h,
            100.0 * frame["gas_new_observed_slice_unconnected_below_fraction"],
            color="#d627b0",
            linewidth=1.8,
            label="New + not 2D-connected + below",
        )
        axis.set_ylabel("Gas-core pixels (%)")
        axis.legend(frameon=False, fontsize=7, loc="upper right")

        for axis in axes[row]:
            for stage in stages[1:]:
                axis.axvline(stage.start_h, color="#95a5a6", linewidth=0.8, linestyle=":")
            axis.axvline(stages[-1].end_h, color="#95a5a6", linewidth=0.8, linestyle=":")
            axis.grid(axis="y", color="#e2e5e8", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_xlim(left=0.0)
    for axis in axes[-1]:
        axis.set_xlabel("Elapsed time (h, symlog after 1 h)")
    axes[0, 0].set_title("Vertical and pre-activation gas checks")
    axes[0, 1].set_title("Vertical gas position")
    axes[0, 2].set_title("Observed-slice topology (diagnostic only)")
    flash_label = "TitrationFlash" if flash_model == "titration" else "SimpleFlash"
    fig.suptitle(
        f"Locked {flash_label}: spatial free-CO2 diagnostics",
        fontsize=15,
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, dpi=180, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)


def analyse_run(
    *,
    run: str,
    campaign_root: Path,
    output_root: Path,
    config_dir: Path,
    bounds_map: dict[str, Any],
    backend: str,
    topology_scale: float,
    frame_limit: int | None,
    candidate_variant: str = VARIANT,
) -> tuple[pd.DataFrame, tuple[InjectionStage, ...], dict[str, Any]]:
    candidate_path = _candidate_path(campaign_root, run, candidate_variant)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    params = candidate.get("params")
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Candidate has no params mapping: {candidate_path}")
    actual_flash = {
        name: float(params[name])
        for name in LOCKED_FLASH
        if name in params
    }
    if actual_flash != LOCKED_FLASH:
        raise RuntimeError(
            f"{run} does not use the locked Flash values: {actual_flash}"
        )

    started = time.perf_counter()
    context = build_context(
        run=run,
        config_dir=config_dir,
        rig_cls=Rig,
        ref_config_path=None,
        use_facies=True,
        bounds_map=bounds_map,
        enforce_lower=False,
        per_label_params=True,
        use_label_weights=False,
        label_weights=None,
        quality_scale=1.0,
        quality_dtype="float32",
        objective_integral="off",
        static_light_correction="off",
        signal_parameterization="per-label",
        phase_separation="shared-signal",
        color_path_anchor="off",
        color_path_anchor_weight=0.75,
        color_path_anchor_strict=True,
    )
    apply_params(
        context.calibration,
        params,
        labels=context.signal_labels,
        np_module=np,
    )
    stages = load_injection_stages(
        _run_config(run, config_dir),
        context.experiment.experiment_start,
    )
    if len(stages) != 2:
        raise RuntimeError(f"Expected two injection stages for {run}, got {stages}")
    frames = context._loaded[:frame_limit] if frame_limit else context._loaded
    if not frames:
        raise RuntimeError(f"No calibration frames resolved for {run}")

    evaluator = None
    previous_support: np.ndarray | None = None
    records: list[dict[str, Any]] = []
    contact_frames: list[np.ndarray] = []
    try:
        if backend != "legacy":
            evaluator = create_spatial_mass_evaluator(
                context.calibration,
                backend=backend,
                output_dtype="float32",
                cuda_color_interpretation="on",
            )
        for index, (image, injected, elapsed_h) in enumerate(frames):
            frame_started = time.perf_counter()
            if evaluator is None:
                result = _mass_result_for_evaluation(context, image, index, params)
            else:
                result = evaluator.evaluate(image)
            saturation = np.asarray(result.saturation_g.img)
            gas_mass = np.asarray(result.mass_g.img)
            concentration_aq = np.asarray(result.concentration_aq.img)
            aqueous_mass_density = np.asarray(result.mass_aq.img)
            x_m, y_m = _coordinate_axes(result.saturation_g)
            volume = _voxel_volume(context, saturation.shape)
            detected_mass = float(
                np.sum(np.asarray(result.mass.img, dtype=float) * volume, dtype=np.float64)
            )
            aqueous_mass = float(
                np.sum(
                    np.asarray(aqueous_mass_density, dtype=float) * volume,
                    dtype=np.float64,
                )
            )
            finite_aqueous = np.isfinite(concentration_aq) & np.isfinite(
                aqueous_mass_density
            )
            aqueous_trace = finite_aqueous & (concentration_aq >= AQ_TRACE_LEVEL)
            aqueous_envelope = finite_aqueous & (
                concentration_aq >= AQ_ENVELOPE_LEVEL
            )
            aqueous_trace_count = int(np.count_nonzero(aqueous_trace))
            aqueous_trace_only_count = int(
                np.count_nonzero(aqueous_trace & ~aqueous_envelope)
            )
            aqueous_below_envelope = float(
                np.sum(
                    np.where(
                        finite_aqueous & ~aqueous_envelope,
                        aqueous_mass_density,
                        0.0,
                    )
                    * volume,
                    dtype=np.float64,
                )
            )
            aqueous_below_trace = float(
                np.sum(
                    np.where(
                        finite_aqueous & ~aqueous_trace,
                        aqueous_mass_density,
                        0.0,
                    )
                    * volume,
                    dtype=np.float64,
                )
            )
            metrics, state = phase_physics_metrics(
                saturation_g=saturation,
                mass_g=gas_mass,
                voxel_volume=volume,
                x_m=x_m,
                y_m=y_m,
                stages=stages,
                elapsed_h=float(elapsed_h),
                previous_support=previous_support,
                topology_scale=topology_scale,
            )
            candidate_time_h, candidate_values = _candidate_metric(
                candidate,
                float(elapsed_h),
            )
            reference_total = candidate_values["total_full"]
            reference_gas = candidate_values["gaseous_full"]
            reference_aqueous = candidate_values["aqueous_full"]
            parity = {
                "candidate_metric_time_h": candidate_time_h,
                "candidate_total_mass_kg": reference_total,
                "candidate_gas_mass_kg": reference_gas,
                "candidate_aqueous_mass_kg": reference_aqueous,
                "detected_mass_kg": detected_mass,
                "detected_aqueous_mass_kg": aqueous_mass,
                "candidate_total_relative_error": _relative_error(
                    detected_mass, reference_total
                ),
                "candidate_gas_relative_error": _relative_error(
                    metrics["gas_mass_kg"], reference_gas
                ),
                "candidate_aqueous_relative_error": _relative_error(
                    aqueous_mass, reference_aqueous
                ),
            }
            previous_support = state.support.copy()
            records.append(
                {
                    "run": run,
                    "frame_index": index,
                    "image_name": Path(context.calibration_images[index]).name,
                    "elapsed_h": float(elapsed_h),
                    "injected_mass_kg": float(injected),
                    "evaluation_seconds": time.perf_counter() - frame_started,
                    "aqueous_trace_pixels": aqueous_trace_count,
                    "aqueous_envelope_pixels": int(
                        np.count_nonzero(aqueous_envelope)
                    ),
                    "aqueous_trace_only_pixels": aqueous_trace_only_count,
                    "aqueous_trace_only_pixel_fraction": _safe_fraction(
                        aqueous_trace_only_count,
                        aqueous_trace_count,
                    ),
                    "aqueous_mass_below_envelope_kg": aqueous_below_envelope,
                    "aqueous_mass_below_envelope_fraction": _safe_fraction(
                        aqueous_below_envelope,
                        aqueous_mass,
                    ),
                    "aqueous_mass_below_trace_kg": aqueous_below_trace,
                    "aqueous_mass_below_trace_fraction": _safe_fraction(
                        aqueous_below_trace,
                        aqueous_mass,
                    ),
                    **parity,
                    **metrics,
                }
            )
            contact_frames.append(
                _render_diagnostic_frame(
                    image.img,
                    concentration_aq,
                    state,
                    run=run,
                    elapsed_h=float(elapsed_h),
                    stages=stages,
                )
            )
    finally:
        if evaluator is not None:
            evaluator.close()

    frame = pd.DataFrame.from_records(records).sort_values("elapsed_h")
    maximum_parity_error = float(
        frame[
            [
                "candidate_total_relative_error",
                "candidate_gas_relative_error",
                "candidate_aqueous_relative_error",
            ]
        ].to_numpy(dtype=float).max()
    )
    if maximum_parity_error > 1e-5:
        raise RuntimeError(
            f"{run} spatial evaluation does not reproduce its locked candidate: "
            f"maximum relative phase-mass error={maximum_parity_error:.3e}."
        )
    run_output = output_root / run
    _atomic_csv(run_output / "phase_physics_per_frame.csv", frame)
    _contact_sheet(contact_frames, run_output / "phase_physics_contact_sheet.png")
    summary = _summarise_run(frame, run)
    summary.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "candidate": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "candidate_objective_full_scale": float(candidate["objective_full_scale"]),
            "maximum_candidate_phase_mass_relative_error": maximum_parity_error,
            "injection_stages": [asdict(stage) for stage in stages],
        }
    )
    _atomic_json(run_output / "summary.json", summary)
    del context, evaluator, contact_frames
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    return frame, stages, summary


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    campaign_root = Path(args.campaign_root)
    output_root = Path(args.output_root)
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = REPO_ROOT / config_dir
    bounds_path = Path(args.bounds_file)
    if not bounds_path.is_absolute():
        bounds_path = REPO_ROOT / bounds_path
    bounds_map = load_bounds_map(bounds_path)
    runs = tuple(run.lower() for run in (args.runs or RUNS))
    unknown = sorted(set(runs).difference(RUNS))
    if unknown:
        raise ValueError(f"Runs are not part of the locked campaign: {unknown}")

    all_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    stages_by_run: dict[str, tuple[InjectionStage, ...]] = {}
    with campaign_environment(args.flash_model):
        for index, run in enumerate(runs, start=1):
            print(f"[{index}/{len(runs)}] analysing {run}", flush=True)
            frame, stages, summary = analyse_run(
                run=run,
                campaign_root=campaign_root,
                output_root=output_root,
                config_dir=config_dir,
                bounds_map=bounds_map,
                backend=args.backend,
                topology_scale=args.topology_scale,
                frame_limit=args.frame_limit,
                candidate_variant=args.variant,
            )
            all_frames.append(frame)
            stages_by_run[run] = stages
            summaries.append(summary)
            _atomic_json(
                output_root / "progress.json",
                {
                    "updated_utc": _utc_now(),
                    "completed": [item["run"] for item in summaries],
                    "remaining": list(runs[index:]),
                },
            )

    per_frame = pd.concat(all_frames, ignore_index=True)
    per_run = pd.DataFrame.from_records(summaries)
    _atomic_csv(output_root / "phase_physics_per_frame.csv", per_frame)
    _atomic_csv(output_root / "phase_physics_per_run.csv", per_run)
    _plot_grid(
        per_frame,
        stages_by_run,
        runs,
        output_root / "phase_physics_diagnostics_grid.png",
        flash_model=args.flash_model,
    )
    manifest = {
        "schema": 2,
        "complete": True,
        "created_utc": _utc_now(),
        "host": socket.gethostname().upper(),
        "campaign_root": str(campaign_root),
        "candidate_variant": args.variant,
        "flash_model": args.flash_model,
        "bounds_file": str(bounds_path),
        "output_root": str(output_root),
        "runs": list(runs),
        "backend": args.backend,
        "frame_limit": args.frame_limit,
        "thresholds": {
            "gas_core_saturation": CORE_LEVEL,
            "gas_support_saturation": SUPPORT_LEVEL,
            "aqueous_trace_concentration": AQ_TRACE_LEVEL,
            "aqueous_envelope_concentration": AQ_ENVELOPE_LEVEL,
            "vertical_tolerance_m": VERTICAL_TOLERANCE_M,
            "source_radius_x_m": SOURCE_RADIUS_X_M,
            "source_radius_y_m": SOURCE_RADIUS_Y_M,
            "history_dilation_m": HISTORY_DILATION_M,
            "topology_scale": args.topology_scale,
        },
        "locked_flash": LOCKED_FLASH,
        "interpretation": {
            "read_only": True,
            "warning_not_exclusion": True,
            "dimensionality": "2D camera-observed slice of a 3D volume",
            "connectivity_scope": "observed 2D slice only",
            "physical_3d_connectivity_unobservable": True,
            "connectivity_is_not_an_exclusion_criterion": True,
            "connectivity_caveat": (
                "Gas may move into or out of the observed slice and may be "
                "continuous through the unobserved 3D volume."
            ),
            "gas_below_latest_source_is_descriptive_after_port_2": (
                "Residual gas from port 1 legitimately remains below port 2."
            ),
            "primary_physical_metrics": [
                "gas_mass_below_lowest_activated_source_fraction",
                "gas_mass_near_future_source_fraction",
                "gas_lower_front_q05_y_m",
            ],
            "topology_review_metric": (
                "gas_new_observed_slice_unconnected_below_fraction"
            ),
        },
        "software": {
            "ff_ac_commit": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "ff_ac_dirty": bool(_git_value(REPO_ROOT, "status", "--porcelain")),
            "darsia_commit": _git_value(
                project_path('external/darsia'), "rev-parse", "HEAD"
            ),
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "run_summaries": summaries,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    (output_root / "PHASE_PHYSICS_COMPLETE").write_text(
        _utc_now() + "\n",
        encoding="ascii",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--campaign-root", default=str(DEFAULT_LOGS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config-dir", default="config_seg6/run_ac")
    parser.add_argument("--variant", default=VARIANT)
    parser.add_argument("--bounds-file", default=BOUNDS_FILE)
    parser.add_argument(
        "--flash-model",
        choices=("titration", "simple"),
        default="titration",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "opencl", "cpu", "legacy"),
        default="cuda",
    )
    parser.add_argument("--topology-scale", type=float, default=TOPOLOGY_SCALE)
    parser.add_argument("--frame-limit", type=int, default=None)
    args = parser.parse_args(argv)
    if not 0.05 <= args.topology_scale <= 1.0:
        parser.error("--topology-scale must be between 0.05 and 1.0")
    if args.frame_limit is not None and args.frame_limit < 1:
        parser.error("--frame-limit must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run_diagnostics(parse_args(argv))


if __name__ == "__main__":
    main()
