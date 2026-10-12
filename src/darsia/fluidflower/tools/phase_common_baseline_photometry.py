"""Opt-in, fixed FF-AC photometry from a pre-injection baseline.

This module has no file access or phase-model dependency. A caller supplies the
already stabilized within-run baseline and observed ColorChecker fields, plus a
frozen common target. The fitted transform is fixed for every later frame.
Only pre-injection sand cells and ColorChecker fields participate in fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
from typing import Any, Mapping
from warnings import warn

import numpy as np
from scipy.ndimage import binary_erosion


FIT_FORMAT = "ffac-fixed-common-baseline-photometry-v1"


class PhotometryRejected(ValueError):
    """The inputs or proposed fixed correction failed a photometric guard."""


@dataclass(frozen=True)
class FitPolicy:
    """Conservative bounds for a run-level, blue-sand preserving correction."""

    gain_min: float = 0.90
    gain_max: float = 1.10
    gain_steps: int = 21
    chart_good_mean: float = 0.002
    chart_min_improvement: float = 0.0001
    chart_max_worst_increase: float = 0.003
    min_usable_chart_patches: int = 12
    shadow_cutoff_fraction_of_black: float = 0.80
    shadow_alpha_min: float = -0.50
    shadow_alpha_max: float = 2.50
    min_cell_pixels: int = 20
    min_cells: int = 6
    min_blue_cell_fraction: float = 0.80
    max_blue_chromaticity_difference: float = 0.15
    sand_good_log_error: float = 0.05
    sand_min_relative_improvement: float = 0.05
    sand_max_log_error_after: float = 0.40
    require_facies_map: bool = True
    input_max_overshoot: float = 0.001
    input_max_overshoot_fraction: float = 0.000001


def _rgb_array(value: Any, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise PhotometryRejected(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise PhotometryRejected(f"{name} must contain finite RGB values in [0, 1]")
    return array


def _mask(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise PhotometryRejected(f"{name} must be a boolean array of shape {shape}")
    return array


def _linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _srgb(linear: np.ndarray) -> np.ndarray:
    positive = np.maximum(linear, 0.0)
    return np.where(
        positive <= 0.0031308,
        12.92 * positive,
        1.055 * np.power(positive, 1.0 / 2.4) - 0.055,
    )


def _chart_metrics(observed: np.ndarray, target: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    per_patch = np.mean(np.abs(observed.reshape(24, 3) - target.reshape(24, 3)), axis=1)
    return {
        "mean_all": float(np.mean(per_patch)),
        "mean_usable": float(np.mean(per_patch[usable])),
        "max_all": float(np.max(per_patch)),
    }


def _target_digest(target_spec: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(target_spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PhotometryRejected("target_spec must be a JSON-serializable frozen target") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_target(target_spec: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    if not isinstance(target_spec, Mapping):
        raise PhotometryRejected("target_spec must be a mapping")
    if tuple(target_spec.get("shape_hw", ())) != shape:
        raise PhotometryRejected("baseline shape differs from target geometry shape_hw")
    chart = _rgb_array(target_spec.get("chart_target_rgb"), (4, 6, 3), "chart_target_rgb")
    cells_raw = target_spec.get("cells")
    if not isinstance(cells_raw, list) or not cells_raw:
        raise PhotometryRejected("target_spec.cells must be a nonempty list")
    geometry = target_spec.get("geometry")
    if not isinstance(geometry, Mapping):
        raise PhotometryRejected("target_spec.geometry must be a mapping")
    step = geometry.get("sampling_step")
    erosion = geometry.get("facies_erosion_kernel_sampled_px")
    if type(step) is not int or step < 1 or type(erosion) is not int or erosion < 1 or erosion % 2 == 0:
        raise PhotometryRejected("sampling_step and odd erosion kernel must be positive integers")
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(cells_raw):
        if not isinstance(row, Mapping):
            raise PhotometryRejected(f"cells[{index}] must be a mapping")
        name = row.get("id")
        if not isinstance(name, str) or not name or name in seen:
            raise PhotometryRejected("cell ids must be unique nonempty strings")
        seen.add(name)
        rect = np.asarray(row.get("rect_rc"))
        if rect.shape != (2, 2) or not np.issubdtype(rect.dtype, np.integer):
            raise PhotometryRejected(f"cell {name}: rect_rc must be 2x2 integer coordinates")
        (r0, c0), (r1, c1) = rect.tolist()
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            raise PhotometryRejected(f"cell {name}: rectangle outside target geometry")
        facies = row.get("facies")
        if type(facies) is not int:
            raise PhotometryRejected(f"cell {name}: facies must be an integer label")
        rgb = _rgb_array(row.get("rgb_median"), (3,), f"cell {name} rgb_median")
        cells.append({"id": name, "facies": facies, "rect_rc": ((r0, c0), (r1, c1)), "target_rgb": rgb})
    return chart, cells, step, erosion


def _sample_cells(
    baseline: np.ndarray,
    cells: list[dict[str, Any]],
    step: int,
    erosion: int,
    facies_map: np.ndarray | None,
    domain_mask: np.ndarray | None,
    interior_mask: np.ndarray | None,
    policy: FitPolicy,
) -> list[dict[str, Any]]:
    shape = baseline.shape[:2]
    if facies_map is None and policy.require_facies_map:
        raise PhotometryRejected("facies_map is required to reproduce target cell sampling")
    facies_grid = None
    if facies_map is not None:
        facies_array = np.asarray(facies_map)
        if facies_array.shape != shape or not np.issubdtype(facies_array.dtype, np.integer):
            raise PhotometryRejected("facies_map must be a full-resolution integer label array")
        facies_grid = facies_array[::step, ::step]
    valid_grid = np.ones(baseline[::step, ::step].shape[:2], dtype=bool)
    if domain_mask is not None:
        valid_grid &= _mask(domain_mask, shape, "domain_mask")[::step, ::step]
    if interior_mask is not None:
        valid_grid &= _mask(interior_mask, shape, "interior_mask")[::step, ::step]
    eroded: dict[int, np.ndarray] = {}
    result: list[dict[str, Any]] = []
    for cell in cells:
        facies = cell["facies"]
        if facies_grid is not None and facies not in eroded:
            membership = facies_grid == facies
            eroded[facies] = (
                binary_erosion(membership, structure=np.ones((erosion, erosion), dtype=bool))
                if erosion > 1 else membership
            )
        (r0, c0), (r1, c1) = cell["rect_rc"]
        rows = slice((r0 + step - 1) // step, (r1 + step - 1) // step)
        cols = slice((c0 + step - 1) // step, (c1 + step - 1) // step)
        selected = valid_grid[rows, cols].copy()
        if facies_grid is not None:
            selected &= eroded[facies][rows, cols]
        pixels = baseline[::step, ::step][rows, cols][selected]
        if len(pixels) < policy.min_cell_pixels:
            raise PhotometryRejected(f"cell {cell['id']}: only {len(pixels)} valid sampled pixels")
        result.append({**cell, "pixels": pixels, "n_pixels": int(len(pixels))})
    if len(result) < policy.min_cells:
        raise PhotometryRejected(f"only {len(result)} cells; need {policy.min_cells}")
    return result


def _choose_chart_gains(
    observed: np.ndarray, target: np.ndarray, usable: np.ndarray, policy: FitPolicy,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    before = _chart_metrics(observed, target, usable)
    values = np.linspace(policy.gain_min, policy.gain_max, policy.gain_steps)
    if not np.any(np.isclose(values, 1.0, atol=1e-12)):
        raise PhotometryRejected("gain grid must include identity")
    source_linear = _linear(observed)
    best: tuple[tuple[float, float, float, float, float], np.ndarray, dict[str, float]] | None = None
    for gains_tuple in product(values, repeat=3):
        gains = np.asarray(gains_tuple)
        candidate = np.clip(_srgb(source_linear * gains), 0.0, 1.0)
        after = _chart_metrics(candidate, target, usable)
        for key in ("mean_all", "mean_usable"):
            limit = (
                before[key] - policy.chart_min_improvement
                if before[key] > policy.chart_good_mean
                else before[key] + 1e-8
            )
            if after[key] > limit + 1e-12:
                break
        else:
            if after["max_all"] > before["max_all"] + policy.chart_max_worst_increase:
                continue
            rank = (
                float(np.sum(np.log(gains) ** 2)),
                after["mean_all"] + after["mean_usable"],
                *map(float, gains),
            )
            if best is None or rank < best[0]:
                best = rank, gains, after
    if best is None:
        raise PhotometryRejected("no bounded diagonal gain improves both chart summaries")
    return best[1], before, best[2]


def _apply_pixels(rgb: np.ndarray, gains: np.ndarray, alpha: float, cutoff: float) -> np.ndarray:
    adjusted = _srgb(_linear(rgb) * gains)
    maximum = np.max(adjusted, axis=-1, keepdims=True)
    shadow = np.maximum(1.0 - maximum / cutoff, 0.0)
    return adjusted * (1.0 + alpha * shadow * shadow)


def _blue_chromaticity(rgb: np.ndarray) -> float:
    return float(rgb[2] / max(float(np.sum(rgb)), 1e-9))


def fit_fixed_run_transform(
    baseline_rgb: np.ndarray,
    target_spec: Mapping[str, Any],
    *,
    observed_chart_rgb: np.ndarray,
    chart_usable_patches: np.ndarray,
    facies_map: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
    interior_mask: np.ndarray | None = None,
    policy: FitPolicy | None = None,
) -> dict[str, Any]:
    """Fit one JSON-safe transform from a stabilized pre-injection baseline.

    ``target_spec`` must contain ``shape_hw``, ``chart_target_rgb`` (4x6x3),
    and ``cells`` with ``id``, integer ``facies``, half-open ``rect_rc``, and
    ``rgb_median``. Source receipts and other target metadata are preserved in
    the target digest but never used as phase labels. ``chart_usable_patches``
    marks fields without source saturation in the original raw photograph.
    """

    chosen = policy if policy is not None else FitPolicy()
    if not isinstance(chosen, FitPolicy):
        raise PhotometryRejected("policy must be a FitPolicy")
    if not (0 < chosen.gain_min <= 1 <= chosen.gain_max and chosen.gain_steps >= 3):
        raise PhotometryRejected("invalid gain policy")
    if not (-1 < chosen.shadow_alpha_min <= 0 <= chosen.shadow_alpha_max < 3):
        raise PhotometryRejected("shadow alpha bounds must preserve monotonic intensity")
    baseline = _rgb_array(baseline_rgb, None, "baseline_rgb")
    if baseline.ndim != 3 or baseline.shape[-1] != 3:
        raise PhotometryRejected("baseline_rgb must have shape (height, width, 3)")
    shape = baseline.shape[:2]
    chart_target, target_cells, step, erosion = _validate_target(target_spec, shape)
    chart_observed = _rgb_array(observed_chart_rgb, (4, 6, 3), "observed_chart_rgb")
    usable = np.asarray(chart_usable_patches)
    if usable.shape == (4, 6):
        usable = usable.reshape(24)
    if usable.shape != (24,) or usable.dtype != np.bool_:
        raise PhotometryRejected("chart_usable_patches must be 24 booleans")
    if int(np.count_nonzero(usable)) < chosen.min_usable_chart_patches:
        raise PhotometryRejected("too few usable unsaturated ColorChecker patches")
    sampled = _sample_cells(
        baseline, target_cells, step, erosion,
        facies_map, domain_mask, interior_mask, chosen,
    )
    gains, chart_before, chart_after = _choose_chart_gains(chart_observed, chart_target, usable, chosen)
    chart_after_gains = np.clip(_srgb(_linear(chart_observed) * gains), 0, 1)
    black = min(float(np.max(chart_target[-1, -1])), float(np.max(chart_after_gains[-1, -1])))
    cutoff = chosen.shadow_cutoff_fraction_of_black * black
    if cutoff <= 0 or not np.isfinite(cutoff):
        raise PhotometryRejected("invalid shadow cutoff from ColorChecker black field")
    if np.any(np.max(chart_after_gains.reshape(24, 3), axis=1) <= cutoff):
        raise PhotometryRejected("shadow correction would modify a measured ColorChecker field")

    source_cells: list[np.ndarray] = []
    target_rgbs: list[np.ndarray] = []
    factors: list[float] = []
    hue_differences: list[float] = []
    blue_source_count = blue_target_count = 0
    for cell in sampled:
        source = np.median(_apply_pixels(cell["pixels"], gains, 0.0, cutoff), axis=0)
        target = cell["target_rgb"]
        source_cells.append(source)
        target_rgbs.append(target)
        blue_source_count += int(source[2] > max(source[0], source[1]))
        blue_target_count += int(target[2] > max(target[0], target[1]))
        hue_differences.append(abs(_blue_chromaticity(source) - _blue_chromaticity(target)))
        blue = float(source[2])
        if 0 < blue < cutoff:
            leverage = (1.0 - blue / cutoff) ** 2
            factors.append(float((target[2] / blue - 1.0) / leverage))
    count = len(sampled)
    if min(blue_source_count, blue_target_count) / count < chosen.min_blue_cell_fraction:
        raise PhotometryRejected("source or target sand cells are not consistently blue")
    if float(np.median(hue_differences)) > chosen.max_blue_chromaticity_difference:
        raise PhotometryRejected("source sand hue differs too far from the frozen blue-sand target")
    source_array = np.asarray(source_cells)
    target_array = np.asarray(target_rgbs)
    before_errors = np.abs(np.log(np.maximum(source_array[:, 2], 1e-6) / np.maximum(target_array[:, 2], 1e-6)))
    before_error = float(np.median(before_errors))
    if before_error <= chosen.sand_good_log_error:
        alpha = 0.0
    else:
        if not factors:
            raise PhotometryRejected("blue sand is above chart-protected shadow range")
        alpha = float(np.median(factors))
        if not chosen.shadow_alpha_min <= alpha <= chosen.shadow_alpha_max:
            raise PhotometryRejected("sand match requires a shadow factor beyond policy bounds")
    corrected_cells = np.asarray([
        np.median(_apply_pixels(cell["pixels"], gains, alpha, cutoff), axis=0)
        for cell in sampled
    ])
    after_errors = np.abs(np.log(np.maximum(corrected_cells[:, 2], 1e-6) / np.maximum(target_array[:, 2], 1e-6)))
    after_error = float(np.median(after_errors))
    if before_error > chosen.sand_good_log_error and after_error > before_error * (1 - chosen.sand_min_relative_improvement):
        raise PhotometryRejected("fixed shadow factor does not improve the blue-sand cells enough")
    if after_error > chosen.sand_max_log_error_after:
        raise PhotometryRejected("blue-sand cells remain too far from the frozen target")
    if np.any(corrected_cells[:, 2] <= np.maximum(corrected_cells[:, 0], corrected_cells[:, 1])):
        raise PhotometryRejected("blue sand hue inverted after correction")

    result: dict[str, Any] = {
        "format": FIT_FORMAT,
        "status": "accepted",
        "shape_hw": list(shape),
        "target_sha256": _target_digest(target_spec),
        "parameters": {
            "gains_linear_rgb": gains.tolist(),
            "shadow_alpha": alpha,
            "shadow_cutoff_srgb": cutoff,
        },
        "policy": asdict(chosen),
        "qa": {
            "chart_before": chart_before,
            "chart_after": chart_after,
            "usable_chart_patches": int(np.count_nonzero(usable)),
            "sand_cells": count,
            "sand_sample_pixels": int(sum(cell["n_pixels"] for cell in sampled)),
            "sand_blue_log_error_before": before_error,
            "sand_blue_log_error_after": after_error,
            "median_blue_chromaticity_difference_before": float(np.median(hue_differences)),
            "facies_mask_used": facies_map is not None,
            "domain_mask_used": domain_mask is not None,
            "interior_mask_used": interior_mask is not None,
        },
    }
    # Reuse the same strict apply path here, so a fit cannot be accepted if it
    # would introduce clipping elsewhere in the baseline image.
    apply_fixed_run_transform(baseline, result)
    return result


def apply_fixed_run_transform(frame_rgb: np.ndarray, fit_result: Mapping[str, Any]) -> np.ndarray:
    """Apply fixed baseline parameters to one frame; never re-fit on that frame."""

    qa = inspect_fixed_frame_input(frame_rgb, fit_result)
    if not qa["within_tolerance"]:
        raise PhotometryRejected(
            "frame input exceeds recorded overshoot tolerance: "
            f"nonfinite={qa['nonfinite_channels']}, "
            f"out_of_range={qa['out_of_range_channels']}, "
            f"max_overshoot={qa['max_overshoot']}"
        )
    frame = np.asarray(frame_rgb, dtype=np.float64)
    if qa["out_of_range_channels"]:
        warn(
            "Fixed FF-AC photometry clipped a bounded source-read overshoot "
            f"before correction: channels={qa['out_of_range_channels']}, "
            f"max_overshoot={qa['max_overshoot']:.8f}. "
            "Persist inspect_fixed_frame_input QA with the frame receipt.",
            RuntimeWarning,
        )
    params = fit_result.get("parameters")
    policy = fit_result.get("policy")
    if not isinstance(params, Mapping) or not isinstance(policy, Mapping):
        raise PhotometryRejected("fit_result parameters or policy missing")
    gains = np.asarray(params.get("gains_linear_rgb"), dtype=np.float64)
    alpha = float(params.get("shadow_alpha"))
    cutoff = float(params.get("shadow_cutoff_srgb"))
    if gains.shape != (3,) or not np.all(np.isfinite(gains)) or not np.isfinite(alpha) or not np.isfinite(cutoff):
        raise PhotometryRejected("nonfinite or malformed fixed parameters")
    if not (float(policy["gain_min"]) <= np.min(gains) <= np.max(gains) <= float(policy["gain_max"])):
        raise PhotometryRejected("fixed gains exceed recorded policy")
    if not (float(policy["shadow_alpha_min"]) <= alpha <= float(policy["shadow_alpha_max"]) and cutoff > 0):
        raise PhotometryRejected("fixed shadow parameters exceed recorded policy")
    output = np.empty(frame.shape, dtype=np.float32)
    for row in range(0, frame.shape[0], 256):
        source = np.clip(frame[row:row + 256], 0.0, 1.0)
        transformed = _apply_pixels(source, gains, alpha, cutoff)
        if not np.all(np.isfinite(transformed)) or np.any(transformed < -1e-9):
            raise PhotometryRejected("nonfinite or negative corrected frame values")
        newly_clipped = (source < 1.0 - 1e-7) & (transformed >= 1.0)
        if np.any(newly_clipped):
            raise PhotometryRejected("fixed transform would newly clip a frame channel")
        output[row:row + 256] = np.clip(transformed, 0.0, 1.0).astype(np.float32)
    return output


def inspect_fixed_frame_input(
    frame_rgb: np.ndarray, fit_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Report source-read range excursions before any bounded clipping.

    A tiny interpolation overshoot can be accepted only when both its maximum
    magnitude and channel fraction satisfy the policy recorded at baseline fit.
    The inspector never changes pixels and also reports rejected excursions.
    """

    if (not isinstance(fit_result, Mapping) or fit_result.get("format") != FIT_FORMAT
            or fit_result.get("status") != "accepted"):
        raise PhotometryRejected("fit_result is not an accepted fixed FF-AC photometry fit")
    frame = np.asarray(frame_rgb, dtype=np.float64)
    if frame.ndim != 3 or frame.shape[-1] != 3 or list(frame.shape[:2]) != fit_result.get("shape_hw"):
        raise PhotometryRejected("frame geometry differs from fitted baseline")
    policy = fit_result.get("policy")
    if not isinstance(policy, Mapping):
        raise PhotometryRejected("fit_result policy missing")
    try:
        magnitude_limit = float(policy["input_max_overshoot"])
        fraction_limit = float(policy["input_max_overshoot_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PhotometryRejected("fit_result input range policy missing or invalid") from exc
    if not (np.isfinite(magnitude_limit) and 0 <= magnitude_limit <= 0.01
            and np.isfinite(fraction_limit) and 0 <= fraction_limit <= 0.001):
        raise PhotometryRejected("fit_result input range policy exceeds hard limits")
    finite = np.isfinite(frame)
    low = finite & (frame < 0)
    high = finite & (frame > 1)
    out_of_range = low | high
    out_count = int(np.count_nonzero(out_of_range))
    nonfinite_count = int(frame.size - np.count_nonzero(finite))
    lower_max = float(np.max(-frame[low])) if np.any(low) else 0.0
    upper_max = float(np.max(frame[high] - 1)) if np.any(high) else 0.0
    max_overshoot = max(lower_max, upper_max)
    first = np.argwhere(out_of_range | ~finite)
    finite_values = frame[finite]
    return {
        "format": "ffac-fixed-frame-input-range-qa-v1",
        "shape_hw": list(frame.shape[:2]),
        "channel_count": int(frame.size),
        "nonfinite_channels": nonfinite_count,
        "below_zero_channels": int(np.count_nonzero(low)),
        "above_one_channels": int(np.count_nonzero(high)),
        "out_of_range_channels": out_count,
        "out_of_range_fraction": float(out_count / frame.size),
        "max_lower_overshoot": lower_max,
        "max_upper_overshoot": upper_max,
        "max_overshoot": max_overshoot,
        "input_min_finite": float(np.min(finite_values)) if finite_values.size else None,
        "input_max_finite": float(np.max(finite_values)) if finite_values.size else None,
        "first_excursion_rc_channel": first[0].tolist() if len(first) else None,
        "policy_max_overshoot": magnitude_limit,
        "policy_max_overshoot_fraction": fraction_limit,
        "within_tolerance": bool(
            nonfinite_count == 0 and max_overshoot <= magnitude_limit
            and out_count / frame.size <= fraction_limit
        ),
    }
