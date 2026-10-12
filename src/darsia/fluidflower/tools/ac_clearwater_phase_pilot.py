"""Validate inferred free CO2 against run-specific clear-water references.

The calibration objective contains only total injected mass. This read-only
postflight therefore uses corrected wash images as an independent optical phase
cue. It reports agreement between clear-water affinity and inferred gas, checks
spatial-backend parity against the stored calibration metrics, and evaluates a
small alkalinity sensitivity without re-optimizing the signal parameters.
"""

from __future__ import annotations

from darsia.fluidflower.storage_paths import configured_tool_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import darsia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darsia.presets.workflows.rig import Rig

REPO_ROOT = series_root("ffac")

from darsia.fluidflower.tools.ac_phase_curve_campaign import (  # noqa: E402
    CAMPAIGN_ID,
    DEFAULT_LOGS_ROOT,
    RUNS,
    VARIANTS,
    Variant,
    _campaign_env,
)
from darsia.fluidflower.tools.ac_phase_physics_diagnostics import (  # noqa: E402
    _coordinate_axes,
    _candidate_metric,
    _relative_error,
    _run_config,
    _voxel_volume,
    load_injection_stages,
    phase_physics_metrics,
)
from darsia.fluidflower.tools.auto_calibrate_color_to_mass import (  # noqa: E402
    _mass_result_for_evaluation,
    _setup_template_registration,
    _warp_registration_image,
    apply_params,
    build_context,
    load_bounds_map,
    prepare_evaluation_context,
    voxel_volume_with_depth_model,
)
from darsia.fluidflower.tools.calibration_review_movie import (  # noqa: E402
    render_contour_frame,
    render_mass_plot,
    write_calibration_movie,
    write_movie_provenance,
)
from darsia.fluidflower.tools.clearwater_phase_reference import (  # noqa: E402
    CLEAR_REFERENCE_COUNT,
    CLEAR_REFERENCE_SPECS,
    clearwater_affinity,
    select_clear_reference_paths,
)
from darsia.fluidflower.tools.ffac_titration_flash import TitrationFlash  # noqa: E402
from darsia.fluidflower.tools.spatial_mass_evaluator import (  # noqa: E402
    create_spatial_mass_evaluator,
)


RAW_ROOT = configured_tool_path('ffac', 'raw_inputs')
DEFAULT_OUTPUT_ROOT = DEFAULT_LOGS_ROOT / "clearwater_phase_pilot"
CHEMISTRY_ALKALINITY_MM = (1.00, 1.25, 1.50)
STRUCTURE_SCALE = 0.20
SMALL_COMPONENT_MAX_PIXELS = 16
SPATIAL_BACKEND_PARITY_TOLERANCE = 5e-5


def _run_independent_diagnostics(
    jobs: Mapping[str, tuple[Any, dict[str, Any]]],
    executor: ThreadPoolExecutor | None,
) -> dict[str, Any]:
    """Run independent, read-only frame diagnostics with bounded concurrency."""

    if executor is None:
        return {
            name: function(**arguments)
            for name, (function, arguments) in jobs.items()
        }
    futures = {
        name: executor.submit(function, **arguments)
        for name, (function, arguments) in jobs.items()
    }
    return {name: future.result() for name, future in futures.items()}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def campaign_environment(
    variant: Variant,
    requested: Mapping[str, str] | None = None,
) -> Iterator[None]:
    requested = dict(_campaign_env(variant) if requested is None else requested)
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


def clearwater_phase_metrics(
    *,
    baseline_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    observed_rgb: np.ndarray,
    saturation_g: np.ndarray,
    concentration_aq: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    affinity, evidence, clear_like = clearwater_affinity(
        baseline_rgb, clear_rgb, observed_rgb, valid_mask
    )
    saturation = np.asarray(saturation_g, dtype=float)
    concentration = np.asarray(concentration_aq, dtype=float)
    gas = np.asarray(valid_mask, dtype=bool) & (saturation >= 0.05)
    aqueous = (
        np.asarray(valid_mask, dtype=bool)
        & (concentration >= 0.05)
        & (saturation < 0.05)
    )
    overlap = gas & clear_like

    def fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
        count = int(np.count_nonzero(denominator))
        return (
            float(np.count_nonzero(numerator) / count)
            if count
            else float("nan")
        )

    def median(mask: np.ndarray) -> float:
        values = affinity[mask & evidence]
        return float(np.median(values)) if values.size else float("nan")

    correlation_mask = evidence & np.isfinite(saturation)
    correlation = float("nan")
    if np.count_nonzero(correlation_mask) >= 3:
        x = affinity[correlation_mask]
        y = saturation[correlation_mask]
        if float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    metrics = {
        "clear_evidence_fraction": fraction(evidence, np.asarray(valid_mask, dtype=bool)),
        "model_gas_supported_by_clear_fraction": fraction(overlap, gas),
        "clear_like_captured_by_model_gas_fraction": fraction(overlap, clear_like),
        "median_clear_affinity_model_gas": median(gas),
        "median_clear_affinity_model_aqueous": median(aqueous),
        "clear_affinity_gas_saturation_correlation": correlation,
        "model_gas_pixel_fraction": fraction(gas, np.asarray(valid_mask, dtype=bool)),
        "clear_like_pixel_fraction": fraction(clear_like, np.asarray(valid_mask, dtype=bool)),
    }
    return metrics, affinity, clear_like


def gas_boundary_geometry_metrics(
    *,
    saturation_g: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    level: float = 0.05,
    tolerance_pixels: int = 3,
) -> dict[str, float]:
    """Quantify whether a gas contour follows facies or analysis boundaries."""

    saturation = np.asarray(saturation_g, dtype=np.float32)
    label_image = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if saturation.shape != valid.shape or label_image.shape != valid.shape:
        raise ValueError("Gas, label, and valid-mask shapes must match")

    gas = valid & np.isfinite(saturation) & (saturation >= float(level))
    kernel = np.ones((3, 3), dtype=np.uint8)
    gas_u8 = gas.astype(np.uint8)
    eroded_gas = cv2.erode(gas_u8, kernel, iterations=1).astype(bool)
    gas_boundary = gas & ~eroded_gas

    facies_boundary = np.zeros(valid.shape, dtype=bool)
    horizontal = (
        valid[:, 1:]
        & valid[:, :-1]
        & (label_image[:, 1:] != label_image[:, :-1])
    )
    facies_boundary[:, 1:] |= horizontal
    facies_boundary[:, :-1] |= horizontal
    vertical = (
        valid[1:, :]
        & valid[:-1, :]
        & (label_image[1:, :] != label_image[:-1, :])
    )
    facies_boundary[1:, :] |= vertical
    facies_boundary[:-1, :] |= vertical

    eroded_valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    analysis_boundary = valid & ~eroded_valid
    radius = max(0, int(tolerance_pixels))
    if radius:
        near_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        near_facies = cv2.dilate(
            facies_boundary.astype(np.uint8), near_kernel, iterations=1
        ).astype(bool)
        near_analysis = cv2.dilate(
            analysis_boundary.astype(np.uint8), near_kernel, iterations=1
        ).astype(bool)
    else:
        near_facies = facies_boundary
        near_analysis = analysis_boundary

    boundary_count = int(np.count_nonzero(gas_boundary))
    gas_count = int(np.count_nonzero(gas))

    def boundary_fraction(mask: np.ndarray) -> float:
        if not boundary_count:
            return float("nan")
        return float(np.count_nonzero(gas_boundary & mask) / boundary_count)

    component_count = 0
    largest_component_fraction = float("nan")
    if gas_count:
        labels_count, component_labels, stats, _centroids = (
            cv2.connectedComponentsWithStats(gas_u8, connectivity=8)
        )
        component_count = max(0, int(labels_count) - 1)
        if component_count:
            largest_component_fraction = float(
                np.max(stats[1:, cv2.CC_STAT_AREA]) / gas_count
            )

    return {
        "gas_boundary_pixel_count": float(boundary_count),
        "gas_component_count": float(component_count),
        "gas_largest_component_fraction": largest_component_fraction,
        "gas_boundary_near_facies_fraction": boundary_fraction(near_facies),
        "gas_boundary_near_analysis_fraction": boundary_fraction(near_analysis),
        "gas_boundary_near_facies_or_analysis_fraction": boundary_fraction(
            near_facies | near_analysis
        ),
    }


def field_structure_metrics(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    levels: dict[str, float],
    prefix: str,
    scale: float = STRUCTURE_SCALE,
    small_component_max_pixels: int = SMALL_COMPONENT_MAX_PIXELS,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Measure fragmentation without treating displayed contours as truth."""

    field = np.asarray(values, dtype=np.float32)
    label_image = np.asarray(labels)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(field)
    if field.shape != valid.shape or label_image.shape != valid.shape:
        raise ValueError("Field, label, and valid-mask shapes must match")
    if not 0.0 < float(scale) <= 1.0:
        raise ValueError("scale must be in (0, 1]")

    height, width = field.shape
    target = (
        max(1, int(round(width * float(scale)))),
        max(1, int(round(height * float(scale)))),
    )
    finite_field = np.where(valid, field, 0.0)
    if target == (width, height):
        coarse = finite_field.copy()
        coarse_valid = valid.copy()
        coarse_labels = label_image.copy()
    else:
        coarse = cv2.resize(finite_field, target, interpolation=cv2.INTER_AREA)
        valid_fraction = cv2.resize(
            valid.astype(np.float32), target, interpolation=cv2.INTER_AREA
        )
        coarse_valid = valid_fraction >= 0.75
        coarse_labels = cv2.resize(
            label_image.astype(np.int32), target, interpolation=cv2.INTER_NEAREST
        )
    coarse = np.asarray(coarse, dtype=np.float32)
    coarse[~coarse_valid] = 0.0

    horizontal_valid = coarse_valid[:, 1:] & coarse_valid[:, :-1]
    vertical_valid = coarse_valid[1:, :] & coarse_valid[:-1, :]
    horizontal_gradient = np.abs(coarse[:, 1:] - coarse[:, :-1])
    vertical_gradient = np.abs(coarse[1:, :] - coarse[:-1, :])
    horizontal_cross = horizontal_valid & (
        coarse_labels[:, 1:] != coarse_labels[:, :-1]
    )
    vertical_cross = vertical_valid & (
        coarse_labels[1:, :] != coarse_labels[:-1, :]
    )
    horizontal_within = horizontal_valid & ~horizontal_cross
    vertical_within = vertical_valid & ~vertical_cross

    def edge_mean(
        horizontal_mask: np.ndarray,
        vertical_mask: np.ndarray,
    ) -> float:
        count = int(np.count_nonzero(horizontal_mask)) + int(
            np.count_nonzero(vertical_mask)
        )
        if not count:
            return float("nan")
        total = float(
            np.sum(horizontal_gradient[horizontal_mask], dtype=np.float64)
            + np.sum(vertical_gradient[vertical_mask], dtype=np.float64)
        )
        return total / count

    total_variation = float(
        np.sum(horizontal_gradient[horizontal_valid], dtype=np.float64)
        + np.sum(vertical_gradient[vertical_valid], dtype=np.float64)
    )
    field_l1 = float(np.sum(np.abs(coarse[coarse_valid]), dtype=np.float64))
    cross_variation = float(
        np.sum(horizontal_gradient[horizontal_cross], dtype=np.float64)
        + np.sum(vertical_gradient[vertical_cross], dtype=np.float64)
    )
    cross_mean = edge_mean(horizontal_cross, vertical_cross)
    within_mean = edge_mean(horizontal_within, vertical_within)
    metrics: dict[str, float] = {
        f"{prefix}_normalized_total_variation": (
            total_variation / field_l1 if field_l1 > 0.0 else float("nan")
        ),
        f"{prefix}_label_boundary_gradient_mean": cross_mean,
        f"{prefix}_within_label_gradient_mean": within_mean,
        f"{prefix}_label_boundary_gradient_contrast": (
            cross_mean / within_mean
            if np.isfinite(cross_mean)
            and np.isfinite(within_mean)
            and within_mean > 0.0
            else float("nan")
        ),
        f"{prefix}_label_boundary_total_variation_fraction": (
            cross_variation / total_variation
            if total_variation > 0.0
            else float("nan")
        ),
    }

    valid_count = int(np.count_nonzero(coarse_valid))
    kernel = np.ones((3, 3), dtype=np.uint8)
    for level_name, level in levels.items():
        support = coarse_valid & (coarse >= float(level))
        support_u8 = support.astype(np.uint8)
        support_count = int(np.count_nonzero(support))
        component_count = 0
        largest_fraction = float("nan")
        small_fraction = float("nan")
        if support_count:
            count, _component_labels, stats, _centroids = (
                cv2.connectedComponentsWithStats(support_u8, connectivity=8)
            )
            component_count = max(0, int(count) - 1)
            areas = np.asarray(stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
            if areas.size:
                largest_fraction = float(np.max(areas) / support_count)
                small_fraction = float(
                    np.sum(
                        areas[areas <= max(1, int(small_component_max_pixels))],
                        dtype=np.int64,
                    )
                    / support_count
                )
        eroded = cv2.erode(support_u8, kernel, iterations=1).astype(bool)
        boundary_count = int(np.count_nonzero(support & ~eroded))
        stem = f"{prefix}_{level_name}"
        metrics.update(
            {
                f"{stem}_support_fraction": (
                    support_count / valid_count if valid_count else float("nan")
                ),
                f"{stem}_component_count": float(component_count),
                f"{stem}_largest_component_fraction": largest_fraction,
                f"{stem}_small_component_fraction": small_fraction,
                f"{stem}_boundary_to_area_ratio": (
                    boundary_count / support_count
                    if support_count
                    else float("nan")
                ),
            }
        )
    return metrics, coarse, coarse_valid


def temporal_field_metrics(
    *,
    current: np.ndarray,
    previous: np.ndarray | None,
    current_valid: np.ndarray,
    previous_valid: np.ndarray | None,
    delta_h: float | None,
    support_level: float,
    prefix: str,
) -> dict[str, float]:
    """Measure frame-to-frame change rates on already coarsened fields."""

    metrics = {
        f"{prefix}_temporal_normalized_l1_per_h": float("nan"),
        f"{prefix}_temporal_support_xor_per_h": float("nan"),
    }
    if previous is None or previous_valid is None or delta_h is None or delta_h <= 0.0:
        return metrics
    valid = np.asarray(current_valid, dtype=bool) & np.asarray(
        previous_valid, dtype=bool
    )
    if not np.any(valid):
        return metrics
    current_values = np.asarray(current, dtype=np.float32)
    previous_values = np.asarray(previous, dtype=np.float32)
    absolute_change = float(
        np.mean(np.abs(current_values[valid] - previous_values[valid]))
    )
    amplitude = max(
        float(np.mean(np.abs(current_values[valid]))),
        float(np.mean(np.abs(previous_values[valid]))),
        1e-6,
    )
    current_support = valid & (current_values >= float(support_level))
    previous_support = valid & (previous_values >= float(support_level))
    union = current_support | previous_support
    metrics[f"{prefix}_temporal_normalized_l1_per_h"] = (
        absolute_change / amplitude / float(delta_h)
    )
    if np.any(union):
        metrics[f"{prefix}_temporal_support_xor_per_h"] = float(
            np.count_nonzero(current_support ^ previous_support)
            / np.count_nonzero(union)
            / float(delta_h)
        )
    return metrics


def _registration_for_new_frames(context: Any) -> tuple[np.ndarray | None, str]:
    stub = SimpleNamespace(color_analysis=SimpleNamespace(base=None))
    return _setup_template_registration(
        run=context.run,
        config=context.config,
        fluidflower=context.fluidflower,
        calibration=stub,
    )


def _average_clear_reference(
    context: Any,
    paths: Sequence[Path],
    transform: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    mean: np.ndarray | None = None
    for index, path in enumerate(paths, start=1):
        try:
            image = context.fluidflower.read_image(path)
        except ValueError as exc:
            if "No imaging protocol configured" not in str(exc):
                raise
            # Wash/flush acquisitions are not part of the experiment timeline.
            # Keep the rig's complete correction chain and use the JPEG timestamp.
            image = darsia.imread(
                path,
                transformations=context.fluidflower.corrections,
                reference_date=context.fluidflower.reference_date,
                name=path.name,
            )
        if transform is not None:
            image = _warp_registration_image(image, transform, mode=mode)
        values = np.asarray(image.img, dtype=np.float32)[..., :3]
        if mean is None:
            mean = values.copy()
        else:
            mean += (values - mean) / float(index)
        del image, values
    if mean is None:
        raise RuntimeError("No clear-water images were read")
    return mean


def _integrate_field(field: Any, voxel_volume: np.ndarray) -> float:
    values = np.asarray(getattr(field, "img", field), dtype=float)
    return float(np.sum(values * voxel_volume, dtype=np.float64))


def _preview(
    image: np.ndarray,
    affinity: np.ndarray,
    gas: np.ndarray,
    clear_like: np.ndarray,
    *,
    scale: float = 0.20,
) -> np.ndarray:
    rgb = np.clip(np.asarray(image)[..., :3], 0.0, 1.0)
    raw = (rgb * 255.0).astype(np.uint8)
    target = (
        max(1, int(round(raw.shape[1] * scale))),
        max(1, int(round(raw.shape[0] * scale))),
    )
    raw = cv2.resize(raw, target, interpolation=cv2.INTER_AREA)
    gas_small = cv2.resize(
        gas.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST
    )
    clear_small = cv2.resize(
        clear_like.astype(np.uint8), target, interpolation=cv2.INTER_NEAREST
    )
    for mask, color in ((gas_small, (255, 60, 45)), (clear_small, (0, 220, 220))):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(raw, contours, -1, color, 1, lineType=cv2.LINE_AA)
    affinity_small = cv2.resize(
        np.asarray(affinity, dtype=np.float32),
        target,
        interpolation=cv2.INTER_AREA,
    )
    heat = cv2.applyColorMap(
        np.clip(255.0 * affinity_small, 0.0, 255.0).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.concatenate((raw, heat), axis=1)


def _save_contact_sheet(frames: Sequence[np.ndarray], output: Path) -> None:
    if not frames:
        return
    width = max(frame.shape[1] for frame in frames)
    padded = [
        cv2.copyMakeBorder(
            frame,
            0,
            0,
            0,
            width - frame.shape[1],
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(padded), cv2.COLOR_RGB2BGR))


def _chemistry_rows(
    *,
    run: str,
    elapsed_h: float,
    injected_mass: float,
    result: Any,
    flash: TitrationFlash,
    voxel_volume: np.ndarray,
    solubility: np.ndarray,
) -> list[dict[str, float | str]]:
    concentration = np.asarray(result.concentration_aq.img, dtype=float)
    saturation = np.asarray(result.saturation_g.img, dtype=float)
    gas_mass = _integrate_field(result.mass_g, voxel_volume)
    blend = float(flash.aqueous_blend)
    effective = (1.0 - blend) * flash._lut_caq + blend * flash._lut_y
    effective = np.maximum.accumulate(np.asarray(effective, dtype=float))
    y_estimated = np.interp(
        np.clip(concentration, 0.0, 1.0),
        effective,
        flash._lut_y,
    )
    rows: list[dict[str, float | str]] = []
    for alkalinity_mm in CHEMISTRY_ALKALINITY_MM:
        alternative = TitrationFlash(
            min_value_aq=flash.min_value_aq,
            max_value_aq=flash.max_value_aq,
            min_value_g=flash.min_value_g,
            max_value_g=flash.max_value_g,
            alkalinity_M=alkalinity_mm * 1e-3,
            btb_M=float(flash.titration_params["btb_M"]),
            pKa1=float(flash.titration_params["pKa1"]),
            pKa2=float(flash.titration_params["pKa2"]),
            pKaB=float(flash.titration_params["pKaB"]),
            co2_sat_M=float(flash.titration_params["co2_sat_M"]),
            n_lut=int(flash.n_lut),
            aqueous_blend=blend,
        )
        alt_physical = np.interp(
            y_estimated,
            alternative._lut_y,
            alternative._lut_caq,
        )
        alt_concentration = (
            (1.0 - blend) * alt_physical + blend * y_estimated
        )
        alt_aqueous_density = (
            solubility * alt_concentration * np.maximum(0.0, 1.0 - saturation)
        )
        aqueous_mass = float(
            np.sum(alt_aqueous_density * voxel_volume, dtype=np.float64)
        )
        rows.append(
            {
                "run": run,
                "elapsed_h": float(elapsed_h),
                "alkalinity_mM": float(alkalinity_mm),
                "aqueous_blend": blend,
                "injected_mass_kg": float(injected_mass),
                "detected_gas_mass_kg": gas_mass,
                "detected_aqueous_mass_kg": aqueous_mass,
                "detected_total_mass_kg": gas_mass + aqueous_mass,
                "absolute_mass_error_kg": abs(
                    gas_mass + aqueous_mass - float(injected_mass)
                ),
            }
        )
    return rows


def analyse_candidate(
    *,
    run: str,
    variant: Variant,
    campaign_root: Path,
    output_root: Path,
    raw_root: Path,
    backend: str,
    environment: Mapping[str, str] | None = None,
    diagnostic_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_path = (
        campaign_root
        / variant.name
        / run
        / f"final_full_scale_{run}.json"
    )
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    params = candidate.get("params")
    if not isinstance(params, dict) or not params:
        raise RuntimeError(f"Candidate has no params: {candidate_path}")
    run_output = output_root / variant.name / run
    run_output.mkdir(parents=True, exist_ok=True)

    with campaign_environment(variant, environment):
        context = build_context(
            run=run,
            config_dir=project_path('config_seg6/run_ac'),
            rig_cls=Rig,
            ref_config_path=None,
            use_facies=bool(getattr(variant, "use_facies", True)),
            bounds_map=load_bounds_map(REPO_ROOT / variant.bounds_file),
            enforce_lower=False,
            per_label_params=True,
            use_label_weights=False,
            label_weights=None,
            quality_scale=1.0,
            quality_dtype="float32",
            objective_integral="off",
            static_light_correction="off",
            signal_parameterization=str(
                getattr(variant, "signal_parameterization", "per-label")
            ),
            phase_separation=variant.phase_separation,
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
        transform, registration_mode = _registration_for_new_frames(context)
        clear_paths = select_clear_reference_paths(raw_root, run)
        clear = _average_clear_reference(
            context,
            clear_paths,
            transform,
            registration_mode,
        )
        baseline = np.asarray(
            context.calibration.color_analysis.base.img,
            dtype=np.float32,
        )[..., :3]
        labels = np.asarray(context.calibration.labels.img)
        facies_labels = np.asarray(context.fluidflower.facies.img)
        layer_labels = np.asarray(context.fluidflower.labels.img)
        valid = np.isin(labels, np.asarray(context.signal_labels, dtype=int))
        shape = tuple(valid.shape)
        voxel_volume = voxel_volume_with_depth_model(context, params, shape)
        x_m, y_m = _coordinate_axes(context._loaded[0][0])
        stages = load_injection_stages(
            _run_config(run, project_path('config_seg6/run_ac')),
            context.experiment.experiment_start,
        )
        if len(stages) != 2:
            raise RuntimeError(
                f"Expected two injection stages for {run}, got {stages}"
            )
        solubility = np.broadcast_to(
            np.asarray(
                context.calibration.co2_mass_analysis.solubility_co2,
                dtype=float,
            ),
            shape,
        )
        evaluator = None
        if variant.phase_separation == "shared-signal":
            evaluator = create_spatial_mass_evaluator(
                context.calibration,
                backend=backend,
                output_dtype="float32",
                cuda_color_interpretation=(
                    "on" if backend in {"auto", "cuda"} else "off"
                ),
            )
        else:
            # The generic spatial evaluator only knows the Flash fields. Use
            # the calibration evaluator so review products reproduce the exact
            # clear-water direct/gated algebra optimized by the campaign.
            prepare_evaluation_context(
                context,
                backend="prepared",
                release_images=False,
            )
        records: list[dict[str, Any]] = []
        chemistry: list[dict[str, Any]] = []
        previews: list[np.ndarray] = []
        contour_frames: list[np.ndarray] = []
        previous_aqueous: np.ndarray | None = None
        previous_gas: np.ndarray | None = None
        previous_aqueous_valid: np.ndarray | None = None
        previous_gas_valid: np.ndarray | None = None
        previous_phase_support: np.ndarray | None = None
        previous_elapsed_h: float | None = None
        diagnostic_worker_count = max(1, int(diagnostic_workers))
        diagnostic_executor = (
            ThreadPoolExecutor(
                max_workers=diagnostic_worker_count,
                thread_name_prefix="ffac-postflight",
            )
            if diagnostic_worker_count > 1
            else None
        )
        preview_indices = {
            0,
            max(0, len(context._loaded) // 3),
            max(0, 2 * len(context._loaded) // 3),
            max(0, len(context._loaded) - 1),
        }
        try:
            for index, (image, injected, elapsed_h) in enumerate(context._loaded):
                result = (
                    evaluator.evaluate(image)
                    if evaluator is not None
                    else _mass_result_for_evaluation(context, image, index, params)
                )
                total = _integrate_field(result.mass, voxel_volume)
                gas = _integrate_field(result.mass_g, voxel_volume)
                aqueous = _integrate_field(result.mass_aq, voxel_volume)
                saturation_g = np.asarray(result.saturation_g.img)
                concentration_aq = np.asarray(result.concentration_aq.img)
                aqueous_pore_volume = voxel_volume * np.clip(
                    1.0 - saturation_g, 0.0, None
                )
                aqueous_concentration_max = float(np.max(concentration_aq))
                aqueous_c_volume = float(
                    np.sum(
                        aqueous_pore_volume * concentration_aq,
                        dtype=np.float64,
                    )
                )
                aqueous_concentration_peak_equivalent = (
                    float(
                        np.sum(
                            aqueous_pore_volume
                            * concentration_aq
                            * concentration_aq,
                            dtype=np.float64,
                        )
                        / aqueous_c_volume
                    )
                    if aqueous_c_volume > 0.0
                    else 0.0
                )
                aqueous_c2_volume = float(
                    np.sum(
                        aqueous_pore_volume
                        * concentration_aq
                        * concentration_aq,
                        dtype=np.float64,
                    )
                )
                aqueous_effective_pore_volume_m3 = (
                    float(aqueous_c_volume * aqueous_c_volume / aqueous_c2_volume)
                    if aqueous_c2_volume > 0.0
                    else 0.0
                )
                diffusion_support_level = float(
                    getattr(
                        context,
                        "aqueous_diffusion_support_level",
                        0.01,
                    )
                )
                aqueous_affected_pore_volume_m3 = float(
                    np.sum(
                        aqueous_pore_volume
                        * (concentration_aq >= diffusion_support_level),
                        dtype=np.float64,
                    )
                )
                jobs: dict[str, tuple[Any, dict[str, Any]]] = {
                    "phase_physics": (
                        phase_physics_metrics,
                        {
                            "saturation_g": saturation_g,
                            "mass_g": np.asarray(result.mass_g.img),
                            "voxel_volume": voxel_volume,
                            "x_m": x_m,
                            "y_m": y_m,
                            "stages": stages,
                            "elapsed_h": float(elapsed_h),
                            "previous_support": previous_phase_support,
                        },
                    ),
                    "clearwater": (
                        clearwater_phase_metrics,
                        {
                            "baseline_rgb": baseline,
                            "clear_rgb": clear,
                            "observed_rgb": np.asarray(image.img),
                            "saturation_g": saturation_g,
                            "concentration_aq": concentration_aq,
                            "valid_mask": valid,
                        },
                    ),
                    "boundary_facies": (
                        gas_boundary_geometry_metrics,
                        {
                            "saturation_g": saturation_g,
                            "labels": facies_labels,
                            "valid_mask": valid,
                        },
                    ),
                    "boundary_layers": (
                        gas_boundary_geometry_metrics,
                        {
                            "saturation_g": saturation_g,
                            "labels": layer_labels,
                            "valid_mask": valid,
                        },
                    ),
                    "aqueous": (
                        field_structure_metrics,
                        {
                            "values": concentration_aq,
                            "labels": labels,
                            "valid_mask": valid,
                            "levels": {"trace": 0.01, "core": 0.05},
                            "prefix": "aqueous",
                        },
                    ),
                    "aqueous_facies": (
                        field_structure_metrics,
                        {
                            "values": concentration_aq,
                            "labels": facies_labels,
                            "valid_mask": valid,
                            "levels": {"trace": 0.01, "core": 0.05},
                            "prefix": "aqueous_facies",
                        },
                    ),
                    "aqueous_layers": (
                        field_structure_metrics,
                        {
                            "values": concentration_aq,
                            "labels": layer_labels,
                            "valid_mask": valid,
                            "levels": {"trace": 0.01, "core": 0.05},
                            "prefix": "aqueous_layers",
                        },
                    ),
                    "gas": (
                        field_structure_metrics,
                        {
                            "values": saturation_g,
                            "labels": labels,
                            "valid_mask": valid,
                            "levels": {"core": 0.05},
                            "prefix": "gas",
                        },
                    ),
                    "contour": (
                        render_contour_frame,
                        {
                            "image": np.asarray(image.img),
                            "concentration_aq": concentration_aq,
                            "saturation_g": saturation_g,
                            "run": run,
                            "variant": variant.name,
                            "frame_index": index,
                            "frame_count": len(context._loaded),
                            "elapsed_h": float(elapsed_h),
                            "injected_mass_kg": float(injected),
                            "detected_total_mass_kg": total,
                            "detected_gas_mass_kg": gas,
                            "detected_aqueous_mass_kg": aqueous,
                        },
                    ),
                }
                if variant.titration:
                    jobs["chemistry"] = (
                        _chemistry_rows,
                        {
                            "run": run,
                            "elapsed_h": float(elapsed_h),
                            "injected_mass": float(injected),
                            "result": result,
                            "flash": context.calibration.flash,
                            "voxel_volume": voxel_volume,
                            "solubility": solubility,
                        },
                    )
                diagnostics = _run_independent_diagnostics(
                    jobs, diagnostic_executor
                )
                phase_physics, phase_state = diagnostics["phase_physics"]
                previous_phase_support = phase_state.support.copy()
                _, reference = _candidate_metric(candidate, float(elapsed_h))
                phase_metrics, affinity, clear_like = diagnostics["clearwater"]
                boundary_metrics = diagnostics["boundary_facies"]
                layer_boundary_metrics_raw = diagnostics["boundary_layers"]
                layer_boundary_metrics = {
                    "gas_boundary_near_layer_fraction": layer_boundary_metrics_raw[
                        "gas_boundary_near_facies_fraction"
                    ],
                    "gas_boundary_near_layer_or_analysis_fraction": (
                        layer_boundary_metrics_raw[
                            "gas_boundary_near_facies_or_analysis_fraction"
                        ]
                    ),
                }
                aqueous_structure, aqueous_coarse, aqueous_valid = diagnostics[
                    "aqueous"
                ]
                aqueous_facies_structure, _unused_facies, _unused_facies_valid = (
                    diagnostics["aqueous_facies"]
                )
                aqueous_layer_structure, _unused_layers, _unused_layers_valid = (
                    diagnostics["aqueous_layers"]
                )
                gas_structure, gas_coarse, gas_valid = diagnostics["gas"]
                delta_h = (
                    float(elapsed_h) - previous_elapsed_h
                    if previous_elapsed_h is not None
                    else None
                )
                temporal_metrics = {
                    **temporal_field_metrics(
                        current=aqueous_coarse,
                        previous=previous_aqueous,
                        current_valid=aqueous_valid,
                        previous_valid=previous_aqueous_valid,
                        delta_h=delta_h,
                        support_level=0.01,
                        prefix="aqueous",
                    ),
                    **temporal_field_metrics(
                        current=gas_coarse,
                        previous=previous_gas,
                        current_valid=gas_valid,
                        previous_valid=previous_gas_valid,
                        delta_h=delta_h,
                        support_level=0.05,
                        prefix="gas",
                    ),
                }
                record = {
                    "run": run,
                    "variant": variant.name,
                    "frame_index": index,
                    "elapsed_h": float(elapsed_h),
                    "injected_mass_kg": float(injected),
                    "detected_total_mass_kg": total,
                    "detected_gas_mass_kg": gas,
                    "detected_aqueous_mass_kg": aqueous,
                    "aqueous_concentration_max": aqueous_concentration_max,
                    "aqueous_concentration_peak_equivalent": (
                        aqueous_concentration_peak_equivalent
                    ),
                    "aqueous_affected_pore_volume_m3": (
                        aqueous_affected_pore_volume_m3
                    ),
                    "aqueous_effective_pore_volume_m3": (
                        aqueous_effective_pore_volume_m3
                    ),
                    "candidate_total_relative_error": _relative_error(
                        total, reference["total_full"]
                    ),
                    "candidate_gas_relative_error": _relative_error(
                        gas, reference["gaseous_full"]
                    ),
                    "candidate_aqueous_relative_error": _relative_error(
                        aqueous, reference["aqueous_full"]
                    ),
                    **phase_metrics,
                    **boundary_metrics,
                    **layer_boundary_metrics,
                    **aqueous_structure,
                    **aqueous_facies_structure,
                    **aqueous_layer_structure,
                    **gas_structure,
                    **temporal_metrics,
                    **phase_physics,
                }
                records.append(record)
                previous_aqueous = aqueous_coarse
                previous_gas = gas_coarse
                previous_aqueous_valid = aqueous_valid
                previous_gas_valid = gas_valid
                previous_elapsed_h = float(elapsed_h)
                contour_frames.append(diagnostics["contour"])
                if variant.titration:
                    chemistry.extend(diagnostics["chemistry"])
                if index in preview_indices:
                    previews.append(
                        _preview(
                            np.asarray(image.img),
                            affinity,
                            saturation_g >= 0.05,
                            clear_like,
                        )
                    )
        finally:
            if diagnostic_executor is not None:
                diagnostic_executor.shutdown(wait=True, cancel_futures=True)
            if evaluator is not None:
                evaluator.close()

    frame = pd.DataFrame.from_records(records)
    chemistry_frame = pd.DataFrame.from_records(chemistry)
    max_parity = float(
        frame[
            [
                "candidate_total_relative_error",
                "candidate_gas_relative_error",
                "candidate_aqueous_relative_error",
            ]
        ].max().max()
    )
    if (
        not np.isfinite(max_parity)
        or max_parity > SPATIAL_BACKEND_PARITY_TOLERANCE
    ):
        raise RuntimeError(
            f"{variant.name}/{run}: spatial backend does not reproduce "
            "calibration phase masses; "
            f"max relative error={max_parity:.3e}, "
            f"tolerance={SPATIAL_BACKEND_PARITY_TOLERANCE:.3e}"
        )
    frame.to_csv(run_output / "clearwater_phase_per_frame.csv", index=False)
    if not chemistry_frame.empty:
        chemistry_frame.to_csv(
            run_output / "chemistry_sensitivity_per_frame.csv", index=False
        )
    _save_contact_sheet(previews, run_output / "clearwater_phase_contact_sheet.jpg")
    mass_plot = render_mass_plot(
        records,
        run=run,
        variant=variant.name,
        objective_full_scale=float(candidate["objective_full_scale"]),
        mass_objective_full_scale=(
            float(candidate["objective_terms"]["mass_objective"])
            if isinstance(candidate.get("objective_terms"), dict)
            and candidate["objective_terms"].get("mass_objective") is not None
            else None
        ),
    )
    mass_plot_path = run_output / "calibration_mass_fit.png"
    cv2.imwrite(str(mass_plot_path), cv2.cvtColor(mass_plot, cv2.COLOR_RGB2BGR))
    movie_path = run_output / "calibration_review.mp4"
    movie = write_calibration_movie(
        output=movie_path,
        plot_frame=mass_plot,
        contour_frames=contour_frames,
    )
    movie.update(
        {
            "run": run,
            "variant": variant.name,
            "candidate": str(candidate_path),
            "candidate_sha256": _sha256(candidate_path),
            "mass_plot": str(mass_plot_path),
            "maximum_spatial_backend_parity_relative_error": max_parity,
            "spatial_backend_parity_relative_tolerance": (
                SPATIAL_BACKEND_PARITY_TOLERANCE
            ),
        }
    )
    write_movie_provenance(run_output / "calibration_review.json", movie)
    injected_sum = float(frame["injected_mass_kg"].sum())
    maximum_injected = float(frame["injected_mass_kg"].max())
    mass_l1 = float(
        np.abs(
            frame["detected_total_mass_kg"] - frame["injected_mass_kg"]
        ).sum()
    )
    early_row = frame.iloc[
        int(np.argmin(np.abs(frame["elapsed_h"].to_numpy(dtype=float) - 4.1)))
    ]
    late_row = frame.iloc[int(np.argmax(frame["elapsed_h"].to_numpy(dtype=float)))]
    late_frame = frame.loc[frame["elapsed_h"] >= 8.0]

    def median_late(column: str) -> float:
        values = late_frame[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else float("nan")

    def phase_fraction(row: pd.Series, column: str) -> float:
        total_mass = float(row["detected_total_mass_kg"])
        return (
            float(row[column]) / total_mass
            if total_mass > 0.0
            else float("nan")
        )

    summary = {
        "run": run,
        "variant": variant.name,
        "candidate": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "objective_full_scale": float(candidate["objective_full_scale"]),
        "transition_T": float(params["flash.max_value_aq"]),
        "gas_width_W": (
            float(params["flash.gas_width"])
            if "flash.gas_width" in params
            else None
        ),
        "phase_separation": variant.phase_separation,
        "aqueous_blend": float(params.get("flash.aqueous_blend", 0.0)),
        "clear_reference": {
            "selection": asdict(CLEAR_REFERENCE_SPECS[run]),
            "paths": [str(path) for path in clear_paths],
            "sha256": [_sha256(path) for path in clear_paths],
        },
        "maximum_spatial_backend_parity_relative_error": max_parity,
        "spatial_backend_parity_relative_tolerance": (
            SPATIAL_BACKEND_PARITY_TOLERANCE
        ),
        "normalized_mass_l1": (
            mass_l1 / injected_sum if injected_sum > 0.0 else float("nan")
        ),
        "normalized_mae": (
            mass_l1 / len(frame) / maximum_injected
            if len(frame) and maximum_injected > 0.0
            else float("nan")
        ),
        "ratio_near_4p1h": float(early_row["detected_total_mass_kg"])
        / max(float(early_row["injected_mass_kg"]), 1e-12),
        "gas_fraction_near_4p1h": phase_fraction(
            early_row, "detected_gas_mass_kg"
        ),
        "late_mass_ratio": float(late_row["detected_total_mass_kg"])
        / max(float(late_row["injected_mass_kg"]), 1e-12),
        "late_gas_fraction": phase_fraction(late_row, "detected_gas_mass_kg"),
        "median_model_gas_supported_by_clear_fraction": float(
            frame["model_gas_supported_by_clear_fraction"].median()
        ),
        "median_clear_like_captured_by_model_gas_fraction": float(
            frame["clear_like_captured_by_model_gas_fraction"].median()
        ),
        "median_clear_affinity_model_gas": float(
            frame["median_clear_affinity_model_gas"].median()
        ),
        "median_clear_affinity_model_aqueous": float(
            frame["median_clear_affinity_model_aqueous"].median()
        ),
        "median_gas_boundary_near_facies_fraction": float(
            frame["gas_boundary_near_facies_fraction"].median()
        ),
        "median_gas_boundary_near_analysis_fraction": float(
            frame["gas_boundary_near_analysis_fraction"].median()
        ),
        "median_gas_boundary_near_facies_or_analysis_fraction": float(
            frame["gas_boundary_near_facies_or_analysis_fraction"].median()
        ),
        "median_gas_boundary_near_layer_fraction": float(
            frame["gas_boundary_near_layer_fraction"].median()
        ),
        "median_gas_boundary_near_layer_or_analysis_fraction": float(
            frame["gas_boundary_near_layer_or_analysis_fraction"].median()
        ),
        "median_late_aqueous_trace_component_count": median_late(
            "aqueous_trace_component_count"
        ),
        "median_late_aqueous_trace_largest_component_fraction": median_late(
            "aqueous_trace_largest_component_fraction"
        ),
        "median_late_aqueous_trace_small_component_fraction": median_late(
            "aqueous_trace_small_component_fraction"
        ),
        "median_late_aqueous_trace_boundary_to_area_ratio": median_late(
            "aqueous_trace_boundary_to_area_ratio"
        ),
        "median_late_aqueous_label_boundary_gradient_contrast": median_late(
            "aqueous_label_boundary_gradient_contrast"
        ),
        "median_late_aqueous_facies_boundary_gradient_contrast": median_late(
            "aqueous_facies_label_boundary_gradient_contrast"
        ),
        "median_late_aqueous_layer_boundary_gradient_contrast": median_late(
            "aqueous_layers_label_boundary_gradient_contrast"
        ),
        "median_late_aqueous_facies_boundary_total_variation_fraction": median_late(
            "aqueous_facies_label_boundary_total_variation_fraction"
        ),
        "median_late_aqueous_layer_boundary_total_variation_fraction": median_late(
            "aqueous_layers_label_boundary_total_variation_fraction"
        ),
        "median_late_aqueous_normalized_total_variation": median_late(
            "aqueous_normalized_total_variation"
        ),
        "median_late_aqueous_temporal_normalized_l1_per_h": median_late(
            "aqueous_temporal_normalized_l1_per_h"
        ),
        "median_late_gas_core_component_count": median_late(
            "gas_core_component_count"
        ),
        "median_late_gas_core_small_component_fraction": median_late(
            "gas_core_small_component_fraction"
        ),
        "median_late_gas_temporal_normalized_l1_per_h": median_late(
            "gas_temporal_normalized_l1_per_h"
        ),
        "maximum_gas_mass_below_lowest_activated_source_fraction": float(
            frame["gas_mass_below_lowest_activated_source_fraction"].max()
        ),
        "maximum_gas_mass_near_future_source_fraction": float(
            frame["gas_mass_near_future_source_fraction"].max()
        ),
        "maximum_gas_new_observed_slice_unconnected_below_fraction": float(
            frame["gas_new_observed_slice_unconnected_below_fraction"].max()
        ),
        "median_gas_connected_core_fraction": float(
            frame["gas_connected_core_fraction"].median()
        ),
        "minimum_gas_lower_front_q05_y_m": float(
            frame["gas_lower_front_q05_y_m"].min()
        ),
        "calibration_review_movie": movie,
    }
    _atomic_json(run_output / "summary.json", summary)
    del context, clear, baseline
    gc.collect()
    return frame, chemistry_frame, summary


def _plot_summary(frame: pd.DataFrame, output: Path) -> None:
    variants = list(dict.fromkeys(frame["variant"].tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for variant in variants:
        subset = frame.loc[frame["variant"] == variant]
        grouped = subset.groupby("run", sort=True)
        support = grouped["model_gas_supported_by_clear_fraction"].median()
        affinity_g = grouped["median_clear_affinity_model_gas"].median()
        affinity_aq = grouped["median_clear_affinity_model_aqueous"].median()
        x = np.arange(len(support))
        offset = -0.18 if variant == variants[0] else 0.18
        axes[0, 0].bar(x + offset, support.values, width=0.34, label=variant)
        axes[0, 0].set_xticks(x, [name.upper() for name in support.index])
        axes[0, 1].plot(
            x,
            affinity_g.reindex(support.index),
            marker="o",
            linewidth=1.5,
            label=f"{variant}: gas",
        )
        axes[0, 1].plot(
            x,
            affinity_aq.reindex(support.index),
            marker="s",
            linewidth=1.2,
            linestyle="--",
            label=f"{variant}: aqueous",
        )
        axes[0, 1].set_xticks(x, [name.upper() for name in support.index])
    axes[0, 0].set_title("Model gas supported by clear-water cue")
    axes[0, 0].set_ylabel("Median fraction")
    axes[0, 1].set_title("Clear-water affinity by inferred phase")
    axes[0, 1].set_ylabel("Median affinity")
    for axis in axes[0]:
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", color="#d9dde1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_phase_plausibility_scorecard(
    summaries: Sequence[dict[str, Any]], output_root: Path
) -> None:
    """Write a multi-criterion table without collapsing physics to one score."""

    frame = pd.DataFrame.from_records(summaries)
    best_mass = frame.groupby("run")["normalized_mae"].transform("min")
    frame["mass_delta_from_run_best_percentage_points"] = 100.0 * (
        frame["normalized_mae"] - best_mass
    )
    frame["within_two_percentage_points_of_mass_best"] = (
        frame["mass_delta_from_run_best_percentage_points"] <= 2.0 + 1e-12
    )
    columns = [
        "run",
        "variant",
        "normalized_mae",
        "normalized_mass_l1",
        "mass_delta_from_run_best_percentage_points",
        "within_two_percentage_points_of_mass_best",
        "ratio_near_4p1h",
        "late_mass_ratio",
        "gas_fraction_near_4p1h",
        "late_gas_fraction",
        "median_model_gas_supported_by_clear_fraction",
        "median_gas_boundary_near_facies_fraction",
        "median_gas_boundary_near_analysis_fraction",
        "median_gas_boundary_near_layer_fraction",
        "median_late_aqueous_trace_component_count",
        "median_late_aqueous_trace_largest_component_fraction",
        "median_late_aqueous_trace_small_component_fraction",
        "median_late_aqueous_trace_boundary_to_area_ratio",
        "median_late_aqueous_label_boundary_gradient_contrast",
        "median_late_aqueous_facies_boundary_gradient_contrast",
        "median_late_aqueous_layer_boundary_gradient_contrast",
        "median_late_aqueous_facies_boundary_total_variation_fraction",
        "median_late_aqueous_layer_boundary_total_variation_fraction",
        "median_late_aqueous_normalized_total_variation",
        "median_late_aqueous_temporal_normalized_l1_per_h",
        "median_late_gas_core_component_count",
        "median_late_gas_core_small_component_fraction",
        "median_late_gas_temporal_normalized_l1_per_h",
        "maximum_gas_mass_below_lowest_activated_source_fraction",
        "maximum_gas_mass_near_future_source_fraction",
        "maximum_gas_new_observed_slice_unconnected_below_fraction",
        "median_gas_connected_core_fraction",
        "minimum_gas_lower_front_q05_y_m",
    ]
    scorecard = frame.loc[:, columns].sort_values(["run", "normalized_mae"])
    scorecard.to_csv(output_root / "phase_plausibility_scorecard.csv", index=False)
    _atomic_json(
        output_root / "phase_plausibility_scorecard.json",
        {
            "schema": 1,
            "created_at_utc": _utc_now(),
            "selection_policy": (
                "Mass is an eligibility criterion. Spatial metrics are compared "
                "only among candidates on the two-percentage-point mass front. "
                "No scalar plausibility score is imposed."
            ),
            "phase_physics_interpretation": {
                "gas_below_lowest_activated_source": "primary warning metric",
                "gas_near_future_source": "primary warning metric",
                "observed_slice_connectivity": (
                    "diagnostic only; 3D connectivity is not observable in the "
                    "camera-facing slice"
                ),
            },
            "structure_scale": STRUCTURE_SCALE,
            "small_component_max_pixels_at_structure_scale": (
                SMALL_COMPONENT_MAX_PIXELS
            ),
            "rows": scorecard.to_dict(orient="records"),
        },
    )


def run_all(args: argparse.Namespace) -> int:
    campaign_root = Path(args.campaign_root)
    output_root = Path(args.output_root)
    all_frames: list[pd.DataFrame] = []
    all_chemistry: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for run in RUNS:
            frame, chemistry, summary = analyse_candidate(
                run=run,
                variant=variant,
                campaign_root=campaign_root,
                output_root=output_root,
                raw_root=Path(args.raw_root),
                backend=args.backend,
            )
            all_frames.append(frame)
            if not chemistry.empty:
                chemistry["variant"] = variant.name
                all_chemistry.append(chemistry)
            summaries.append(summary)
    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(output_root / "clearwater_phase_all_frames.csv", index=False)
    pd.DataFrame.from_records(summaries).to_csv(
        output_root / "candidate_summary.csv", index=False
    )
    _write_phase_plausibility_scorecard(summaries, output_root)
    if all_chemistry:
        chemistry = pd.concat(all_chemistry, ignore_index=True)
        chemistry.to_csv(output_root / "chemistry_sensitivity_all_frames.csv", index=False)
        chemistry_summary = (
            chemistry.groupby(["run", "variant", "alkalinity_mM"], as_index=False)
            ["absolute_mass_error_kg"]
            .sum()
        )
        chemistry_summary.to_csv(
            output_root / "chemistry_sensitivity_objective.csv", index=False
        )
    _plot_summary(combined, output_root / "clearwater_phase_summary.png")
    _atomic_json(
        output_root / "postflight_complete.json",
        {
            "schema": 1,
            "completed_at_utc": _utc_now(),
            "host": socket.gethostname().upper(),
            "campaign_id": CAMPAIGN_ID,
            "backend": args.backend,
            "runs": list(RUNS),
            "variants": [variant.name for variant in VARIANTS],
            "candidate_count": len(summaries),
        },
    )
    return 0


def _movie_candidate_task(
    payload: tuple[
        str,
        Variant,
        str,
        str,
        str,
        str,
        dict[str, str],
    ]
) -> dict[str, Any]:
    run, variant, campaign_root, output_root, raw_root, backend, environment = payload
    _frame, _chemistry, summary = analyse_candidate(
        run=run,
        variant=variant,
        campaign_root=Path(campaign_root),
        output_root=Path(output_root),
        raw_root=Path(raw_root),
        backend=backend,
        environment=environment,
    )
    return {
        "run": run,
        "variant": variant.name,
        "movie": summary["calibration_review_movie"]["movie"],
    }


def generate_movies(args: argparse.Namespace) -> int:
    """Generate missing review movies with isolated per-candidate outputs."""

    campaign_root = Path(args.campaign_root)
    output_root = Path(args.output_root)
    tasks: list[
        tuple[
            str,
            Variant,
            str,
            str,
            str,
            str,
            dict[str, str],
        ]
    ] = []
    completed: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for run in RUNS:
            movie = output_root / variant.name / run / "calibration_review.mp4"
            provenance = output_root / variant.name / run / "calibration_review.json"
            if not args.overwrite_movies and movie.is_file() and provenance.is_file():
                completed.append(
                    {"run": run, "variant": variant.name, "movie": str(movie)}
                )
                continue
            tasks.append(
                (
                    run,
                    variant,
                    str(campaign_root),
                    str(output_root),
                    str(Path(args.raw_root)),
                    args.backend,
                    {
                        str(name): str(value)
                        for name, value in _campaign_env(variant).items()
                    },
                )
            )

    workers = max(1, min(int(args.movie_workers), len(tasks) or 1))
    if workers == 1:
        for task in tasks:
            completed.append(_movie_candidate_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_movie_candidate_task, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"movie complete: {result['variant']}/{result['run']}",
                    flush=True,
                )

    _atomic_json(
        output_root / "calibration_movies_complete.json",
        {
            "schema": 1,
            "completed_at_utc": _utc_now(),
            "host": socket.gethostname().upper(),
            "campaign_id": CAMPAIGN_ID,
            "backend": args.backend,
            "workers": workers,
            "movies": sorted(completed, key=lambda row: (row["variant"], row["run"])),
        },
    )
    return 0


def _candidate_paths(campaign_root: Path) -> list[Path]:
    return [
        campaign_root
        / variant.name
        / run
        / f"final_full_scale_{run}.json"
        for variant in VARIANTS
        for run in RUNS
    ]


def wait_and_run(args: argparse.Namespace) -> int:
    campaign_root = Path(args.campaign_root)
    output_root = Path(args.output_root)
    deadline = time.monotonic() + float(args.wait_timeout_hours) * 3600.0
    while True:
        paths = _candidate_paths(campaign_root)
        missing = [str(path) for path in paths if not path.is_file()]
        _atomic_json(
            output_root / "postflight_wait_state.json",
            {
                "schema": 1,
                "updated_at_utc": _utc_now(),
                "available": len(paths) - len(missing),
                "expected": len(paths),
                "missing": missing,
            },
        )
        if not missing:
            return run_all(args)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {len(missing)} calibration candidates"
            )
        time.sleep(float(args.poll_seconds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "wait", "movies", "references"))
    parser.add_argument("--campaign-root", default=str(DEFAULT_LOGS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-root", default=str(RAW_ROOT))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "opencl"), default="cuda")
    parser.add_argument("--wait-timeout-hours", type=float, default=12.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--movie-workers", type=int, default=1)
    parser.add_argument("--overwrite-movies", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "references":
        print(
            json.dumps(
                {
                    run: [
                        {"path": str(path), "sha256": _sha256(path)}
                        for path in select_clear_reference_paths(
                            Path(args.raw_root), run
                        )
                    ]
                    for run in RUNS
                },
                indent=2,
            )
        )
        return
    if args.action == "wait":
        raise SystemExit(wait_and_run(args))
    if args.action == "movies":
        raise SystemExit(generate_movies(args))
    raise SystemExit(run_all(args))


if __name__ == "__main__":
    main()
