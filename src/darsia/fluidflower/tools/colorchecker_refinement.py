"""Run-local, robust ColorChecker normalization for FF-AC.

The implementation is installed on an already loaded DarSIA ColorCorrection
instance. Keeping it in FF-AC makes the campaign reproducible across machines
without requiring a private commit in the DarSIA submodule.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import Any, Mapping
from warnings import warn

import cv2
import numpy as np
import skimage

import darsia


ALGORITHM_ID = "ffac-neutral-tone-affine-v2"
ALGORITHM_REVISIONS = {
    "legacy-v2": ALGORITHM_ID,
    "minimum-improvement-v3": "ffac-neutral-tone-affine-v3",
    "neutral-tone-only-v4": "ffac-neutral-tone-v4",
}

# Physical row-major order of the 24 fields on a ColorChecker Classic. Keeping
# the names here makes diagnostics traceable to an actual field instead of only
# reporting one aggregate residual.
SWATCH_NAMES = (
    "dark_skin",
    "light_skin",
    "blue_sky",
    "foliage",
    "blue_flower",
    "bluish_green",
    "orange",
    "purplish_blue",
    "moderate_red",
    "purple",
    "yellow_green",
    "orange_yellow",
    "blue",
    "green",
    "red",
    "yellow",
    "magenta",
    "cyan",
    "white",
    "neutral_8",
    "neutral_6p5",
    "neutral_5",
    "neutral_3p5",
    "black",
)

# These fields cover the blue/cyan/green/yellow response where the BTB signal
# moves. Neutral fields still constrain exposure and white balance through the
# tone curve, while the remaining chromatic fields constrain the affine map.
BTB_SENSITIVE_SWATCH_INDICES = (2, 3, 4, 5, 7, 10, 11, 12, 13, 15, 17)

_MAX_CACHED_ORIENTATION_SCORE = 0.10
_MIN_NEUTRAL_DYNAMIC_RANGE = 0.50
_MAX_DETECTED_SWATCH_MISMATCH = 0.03
_MIN_TONE_SLOPE_SPAN = 1e-3


def build_swatch_weights(btb_swatch_weight: float = 1.0) -> np.ndarray:
    """Return normalized per-field weights for the ColorChecker fit."""

    weight = float(btb_swatch_weight)
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("BTB ColorChecker swatch weight must be finite and positive")
    weights = np.ones(len(SWATCH_NAMES), dtype=np.float64)
    weights[np.asarray(BTB_SENSITIVE_SWATCH_INDICES, dtype=int)] = weight
    return weights / np.mean(weights)


def residual_summary(
    swatches: np.ndarray,
    reference: np.ndarray,
    *,
    swatch_weights: np.ndarray,
    max_swatch_loss_weight: float,
) -> dict[str, object]:
    """Summarize field-wise sRGB residuals without hiding one bad field."""

    residuals = np.mean(
        np.abs(
            np.asarray(swatches, dtype=np.float64)
            - np.asarray(reference, dtype=np.float64)
        ),
        axis=-1,
    ).reshape(-1)
    weights = np.asarray(swatch_weights, dtype=np.float64).reshape(-1)
    if residuals.size != len(SWATCH_NAMES) or weights.shape != residuals.shape:
        raise ValueError("Expected one residual and weight for each of 24 swatches")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("ColorChecker swatch weights must be finite and positive")
    max_weight = float(max_swatch_loss_weight)
    if not np.isfinite(max_weight) or max_weight < 0.0:
        raise ValueError("Maximum-swatch loss weight must be finite and non-negative")
    worst_index = int(np.argmax(residuals))
    weighted_mean = float(np.average(residuals, weights=weights))
    maximum = float(residuals[worst_index])
    return {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "weighted_mean": weighted_mean,
        "p95": float(np.percentile(residuals, 95.0)),
        "max": maximum,
        "score": weighted_mean + max_weight * maximum,
        "worst_swatch_index": worst_index,
        "worst_swatch_name": SWATCH_NAMES[worst_index],
    }


def srgb_to_linear(
    values: np.ndarray,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Decode sRGB values to linear light."""

    array = np.asarray(values, dtype=dtype)
    return np.asarray(np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    ), dtype=dtype)


def linear_to_srgb(
    values: np.ndarray,
    *,
    dtype: Any = np.float64,
) -> np.ndarray:
    """Encode linear-light RGB values as sRGB."""

    array = np.clip(np.asarray(values, dtype=dtype), 0.0, None)
    return np.asarray(np.where(
        array <= 0.0031308,
        array * 12.92,
        1.055 * (array ** (1.0 / 2.4)) - 0.055,
    ), dtype=dtype)


def extract_swatches(checker_image: np.ndarray) -> np.ndarray:
    """Extract deterministic central-median RGB values from all 24 swatches."""

    image = np.asarray(checker_image)
    if image.dtype in (np.uint8, np.uint16):
        image = skimage.img_as_float(image)
    image = darsia.extract_quadrilateral_ROI(
        image,
        pts_src=None,
        width=27.3,
        height=17.8,
    )
    height, width = image.shape[:2]
    fixed_width = 500
    fixed_height = int(height / width * fixed_width)
    resized = cv2.resize(image, (fixed_width, fixed_height))
    row_positions, column_positions = np.meshgrid(
        [12, 93, 175, 255],
        [12, 95, 177, 260, 344, 427],
        indexing="ij",
    )
    swatches = np.zeros((4, 6, 3), dtype=np.float32)
    for row in range(4):
        for column in range(6):
            row_start = int(row_positions[row, column])
            column_start = int(column_positions[row, column])
            patch = resized[
                row_start : row_start + 50,
                column_start : column_start + 50,
            ]
            patch_height, patch_width = patch.shape[:2]
            row_low = int(patch_height * 0.2)
            row_high = max(int(np.ceil(patch_height * 0.8)), row_low + 1)
            column_low = int(patch_width * 0.2)
            column_high = max(
                int(np.ceil(patch_width * 0.8)), column_low + 1
            )
            central = patch[
                row_low:row_high,
                column_low:column_high,
            ].reshape(-1, 3)
            if central.size == 0:
                central = patch.reshape(-1, 3)
            swatches[row, column] = np.median(
                np.asarray(central, dtype=np.float32), axis=0
            )
    return swatches


def fit_regularized_residual_affine(
    source_swatches: np.ndarray,
    destination_swatches: np.ndarray,
    *,
    ridge: float,
    huber_delta: float,
    max_iterations: int = 8,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a robust near-identity affine residual map in linear RGB."""

    source = np.asarray(source_swatches, dtype=np.float64).reshape(-1, 3)
    destination = np.asarray(destination_swatches, dtype=np.float64).reshape(-1, 3)
    if source.shape != destination.shape or source.shape[0] < 4:
        raise ValueError("Need at least four paired RGB swatches")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("Swatch colours must be finite")

    design = np.column_stack((source, np.ones(source.shape[0], dtype=np.float64)))
    prior = np.zeros((4, 3), dtype=np.float64)
    prior[:3] = np.eye(3, dtype=np.float64)
    penalty = np.diag([ridge, ridge, ridge, 4.0 * ridge])
    if sample_weights is None:
        base_weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        base_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if base_weights.shape != (source.shape[0],):
            raise ValueError("Need one sample weight for each paired RGB swatch")
        if not np.all(np.isfinite(base_weights)) or np.any(base_weights <= 0.0):
            raise ValueError("Sample weights must be finite and positive")
        base_weights = base_weights / np.mean(base_weights)
    robust_weights = np.ones(source.shape[0], dtype=np.float64)
    weights = base_weights.copy()
    coefficients = prior.copy()
    for _ in range(max(1, int(max_iterations))):
        weights = base_weights * robust_weights
        weighted_design = design * weights[:, None]
        coefficients = np.linalg.solve(
            design.T @ weighted_design + penalty,
            design.T @ (destination * weights[:, None]) + penalty @ prior,
        )
        errors = np.linalg.norm(design @ coefficients - destination, axis=1)
        if huber_delta <= 0.0:
            break
        updated = np.minimum(1.0, huber_delta / np.maximum(errors, 1e-12))
        if np.max(np.abs(updated - robust_weights)) < 1e-5:
            robust_weights = updated
            weights = base_weights * robust_weights
            break
        robust_weights = updated

    scaling = coefficients[:3]
    translation = coefficients[3]
    return scaling, translation, {
        "condition": float(np.linalg.cond(scaling)),
        "diagonal_min": float(np.min(np.diag(scaling))),
        "diagonal_max": float(np.max(np.diag(scaling))),
        "off_diagonal_max_abs": float(
            np.max(np.abs(scaling - np.diag(np.diag(scaling))))
        ),
        "translation_max_abs": float(np.max(np.abs(translation))),
        "minimum_base_weight": float(np.min(base_weights)),
        "maximum_base_weight": float(np.max(base_weights)),
        "minimum_weight": float(np.min(weights)),
        "maximum_weight": float(np.max(weights)),
    }


def neutral_tone_curve_correction(
    image: np.ndarray,
    source_swatches: np.ndarray,
    destination_swatches: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Normalize exposure, contrast and white balance from six neutral patches."""

    source = srgb_to_linear(source_swatches)[-1]
    destination = srgb_to_linear(destination_swatches)[-1]
    image_linear = srgb_to_linear(image)
    corrected_linear = np.empty_like(image_linear)
    diagnostics: list[dict[str, object]] = []
    for channel in range(3):
        order = np.argsort(source[:, channel])
        source_sorted = source[order, channel]
        destination_sorted = destination[order, channel]
        monotonic_violation = float(
            np.sum(np.maximum(-np.diff(destination_sorted), 0.0))
        )
        if monotonic_violation > 0.025:
            raise ValueError(
                "Neutral ColorChecker response is not monotone enough for "
                f"channel {channel}: violation={monotonic_violation:.6f}"
            )
        anchors_x = source_sorted.copy()
        anchors_y = destination_sorted.copy()
        if anchors_x[0] > 1e-6:
            anchors_x = np.concatenate(([0.0], anchors_x))
            anchors_y = np.concatenate(([0.0], anchors_y))
        if anchors_x[-1] < 1.0 - 1e-6:
            anchors_x = np.concatenate((anchors_x, [1.0]))
            anchors_y = np.concatenate((anchors_y, [1.0]))

        unique_x: list[float] = []
        unique_y: list[float] = []
        for x_value, y_value in zip(anchors_x, anchors_y, strict=True):
            x_float = float(np.clip(x_value, 0.0, 1.0))
            y_float = float(np.clip(y_value, 0.0, 1.0))
            if unique_x and abs(x_float - unique_x[-1]) <= 1e-6:
                unique_y[-1] = y_float
            else:
                unique_x.append(x_float)
                unique_y.append(y_float)
        x_array = np.asarray(unique_x, dtype=np.float64)
        y_array = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float64))
        intervals = np.diff(x_array)
        slopes = np.diff(y_array) / np.maximum(intervals, 1e-12)
        # Two measured patches can be indistinguishable at the white clipping
        # endpoint of an 8-bit photograph. Interpolation during ROI extraction
        # gives fractional DN values; their tiny source span may exceed the
        # linear-light cutoff even though both source and reference spans are
        # below one sRGB DN. Their ratio is not a measurable tone slope. Only
        # exempt the final measured interval when both upper values are exactly
        # white; an artificial (1, 1) endpoint must not qualify. Leave all tone
        # anchors and the interpolation itself unchanged.
        white_quantization_floor = float(srgb_to_linear(np.asarray(254.0 / 255.0)))
        clipped_white_intervals = (
            (x_array[:-1] >= white_quantization_floor)
            & (y_array[:-1] >= white_quantization_floor)
            & (x_array[1:] == 1.0)
            & (y_array[1:] == 1.0)
            & (source_sorted[-1] == 1.0)
            & (destination_sorted[-1] == 1.0)
        )
        # Adjacent neutral patches can both clip at the white endpoint.  A
        # near-zero slope over a few quantisation levels is then harmless and
        # must not reject an otherwise well-conditioned frame.  Keep the
        # physical slope bounds over every interval with measurable span.
        checked_slopes = slopes[
            (intervals >= _MIN_TONE_SLOPE_SPAN) & ~clipped_white_intervals
        ]
        if not checked_slopes.size:
            raise ValueError(
                "Neutral ColorChecker tone curve has no measurable intervals "
                f"for channel {channel}"
            )
        if np.min(checked_slopes) < 0.05 or np.max(checked_slopes) > 8.0:
            raise ValueError(
                "Neutral ColorChecker tone curve exceeds slope bounds for "
                f"channel {channel}: "
                f"[{np.min(checked_slopes):.4f}, {np.max(checked_slopes):.4f}]"
            )
        corrected_linear[..., channel] = np.interp(
            np.clip(image_linear[..., channel], 0.0, 1.0),
            x_array,
            y_array,
        )
        diagnostics.append(
            {
                "channel": channel,
                "anchors_x": [float(value) for value in x_array],
                "anchors_y": [float(value) for value in y_array],
                "slope_min": float(np.min(slopes)),
                "slope_max": float(np.max(slopes)),
                "checked_slope_min": float(np.min(checked_slopes)),
                "checked_slope_max": float(np.max(checked_slopes)),
                "ignored_short_interval_count": int(
                    np.count_nonzero(intervals < _MIN_TONE_SLOPE_SPAN)
                ),
                "ignored_clipped_white_interval_count": int(
                    np.count_nonzero(clipped_white_intervals)
                ),
                "monotonic_violation": monotonic_violation,
            }
        )
    return linear_to_srgb(corrected_linear), {
        "mode": "neutral-tone",
        "channels": diagnostics,
    }


def _apply_tone_diagnostics(
    image: np.ndarray,
    diagnostics: Mapping[str, object],
    *,
    dtype: Any,
) -> np.ndarray:
    source = np.asarray(image)
    if source.dtype in (np.uint8, np.uint16):
        source = (
            skimage.img_as_float32(source)
            if np.dtype(dtype) == np.dtype(np.float32)
            else skimage.img_as_float64(source)
        )
    image_linear = srgb_to_linear(source, dtype=dtype)
    corrected_linear = np.empty_like(image_linear, dtype=dtype)
    channels = diagnostics.get("channels")
    if not isinstance(channels, list) or len(channels) != 3:
        raise ValueError("Neutral-tone diagnostics require three channel models")
    for channel, model in enumerate(channels):
        if not isinstance(model, Mapping):
            raise ValueError("Invalid neutral-tone channel model")
        anchors_x = np.asarray(model["anchors_x"], dtype=dtype)
        anchors_y = np.asarray(model["anchors_y"], dtype=dtype)
        corrected_linear[..., channel] = np.interp(
            np.clip(image_linear[..., channel], 0.0, 1.0),
            anchors_x,
            anchors_y,
        )
    return linear_to_srgb(corrected_linear, dtype=dtype)


def _apply_affine(
    image: np.ndarray,
    scaling: np.ndarray,
    translation: np.ndarray,
    *,
    dtype: Any,
) -> np.ndarray:
    linear = srgb_to_linear(image, dtype=dtype)
    transformed = linear @ np.asarray(scaling, dtype=dtype)
    transformed += np.asarray(translation, dtype=dtype)
    return linear_to_srgb(transformed, dtype=dtype)


def _apply_pipeline_tiled(
    image: np.ndarray,
    tone_diagnostics: Mapping[str, object],
    affine_steps: list[tuple[np.ndarray, np.ndarray]],
    *,
    dtype: Any,
    tile_rows: int = 256,
) -> np.ndarray:
    """Apply the accepted transform once, with bounded full-frame memory."""

    source = np.asarray(image)
    output = np.empty(source.shape, dtype=dtype)
    rows = max(1, int(tile_rows))
    for row_start in range(0, source.shape[0], rows):
        row_stop = min(source.shape[0], row_start + rows)
        tile_source = source[row_start:row_stop]
        if tile_source.dtype in (np.uint8, np.uint16):
            tile = (
                skimage.img_as_float32(tile_source)
                if np.dtype(dtype) == np.dtype(np.float32)
                else skimage.img_as_float64(tile_source)
            )
        else:
            tile = np.asarray(tile_source, dtype=dtype)
        tile = _apply_tone_diagnostics(
            tile,
            tone_diagnostics,
            dtype=dtype,
        )
        for scaling, translation in affine_steps:
            tile = _apply_affine(
                tile,
                scaling,
                translation,
                dtype=dtype,
            )
        output[row_start:row_stop] = tile
    return output


def _measure_checker_residual(
    correction: Any,
    checker_image: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    swatches = extract_swatches(checker_image)
    summary = residual_summary(
        swatches,
        reference,
        swatch_weights=correction.swatch_weights,
        max_swatch_loss_weight=correction.max_swatch_loss_weight,
    )
    return swatches, summary


def _orientation_candidates(
    correction: Any,
    image: np.ndarray,
    roi: np.ndarray,
) -> list[dict[str, object]]:
    """Score the four rotations of a cached ColorChecker ROI."""

    luminance_weights = np.array([0.2126, 0.7152, 0.0722])
    candidates: list[dict[str, object]] = []
    for shift in range(4):
        candidate_roi = np.roll(roi, -shift, axis=0)
        correction.roi = darsia.make_voxel(candidate_roi)
        swatches = extract_swatches(correction._restrict_to_roi(image))
        neutral = np.asarray(swatches[-1], dtype=np.float64)
        luminance = neutral @ luminance_weights
        chroma = float(np.mean(np.std(neutral, axis=1)))
        descending_violation = float(np.sum(np.maximum(np.diff(luminance), 0.0)))
        dynamic_range = float(np.max(luminance) - np.min(luminance))
        score = chroma + 2.0 * descending_violation + max(0.0, 0.25 - dynamic_range)
        candidates.append(
            {
                "shift": shift,
                "roi": candidate_roi,
                "swatches": swatches,
                "neutral_chroma": chroma,
                "neutral_descending_violation": descending_violation,
                "neutral_dynamic_range": dynamic_range,
                "score": float(score),
            }
        )
    return candidates


def _cached_orientation_is_valid(candidate: Mapping[str, object]) -> bool:
    """Require a low-chroma neutral row with the expected tonal span."""

    return bool(
        float(candidate["score"]) <= _MAX_CACHED_ORIENTATION_SCORE
        and float(candidate["neutral_dynamic_range"])
        >= _MIN_NEUTRAL_DYNAMIC_RANGE
    )


def _polygon_area_xy(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(
        0.5
        * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def _detect_classic_colorchecker_roi(
    correction: Any,
    image: np.ndarray,
    cached_roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Locate the 24-patch Classic chart near an invalid cached ROI."""

    try:
        from colour_checker_detection import detect_colour_checkers_segmentation
    except ImportError as exc:  # pragma: no cover - DarSIA declares this dependency.
        raise RuntimeError(
            "ColorChecker Classic localization requires colour-checker-detection"
        ) from exc

    row_min = int(np.floor(np.min(cached_roi[:, 0])))
    row_max = int(np.ceil(np.max(cached_roi[:, 0])))
    col_min = int(np.floor(np.min(cached_roi[:, 1])))
    col_max = int(np.ceil(np.max(cached_roi[:, 1])))
    roi_height = max(1, row_max - row_min)
    roi_width = max(1, col_max - col_min)
    row_margin = max(100, int(np.ceil(0.5 * roi_height)))
    col_margin = max(200, int(np.ceil(2.0 * roi_width)))
    crop_row_min = max(0, row_min - row_margin)
    crop_row_max = min(image.shape[0], row_max + row_margin)
    crop_col_min = max(0, col_min - col_margin)
    crop_col_max = min(image.shape[1], col_max + col_margin)
    crop = image[
        crop_row_min:crop_row_max,
        crop_col_min:crop_col_max,
    ]
    if min(crop.shape[:2]) < 64:
        raise RuntimeError(
            "Expanded ColorChecker search crop is too small: "
            f"{crop.shape[:2]}"
        )

    working_width = 1440
    detections = detect_colour_checkers_segmentation(
        crop,
        additional_data=True,
        working_width=working_width,
    )
    if not detections:
        raise RuntimeError(
            "Cached ColorChecker ROI failed validation and no Classic chart "
            "was detected in the expanded search area"
        )

    coordinate_scale = float(crop.shape[1]) / float(working_width)
    cached_xy = np.fliplr(np.asarray(cached_roi, dtype=np.float64))
    cached_center = np.mean(cached_xy, axis=0)
    cached_diagonal = max(
        float(np.linalg.norm(np.ptp(cached_xy, axis=0))),
        1.0,
    )
    cached_area = max(_polygon_area_xy(cached_xy), 1.0)
    candidate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for detection_index, detection in enumerate(detections):
        quadrilateral_xy = (
            np.asarray(detection.quadrilateral, dtype=np.float64)
            * coordinate_scale
        )
        quadrilateral_xy[:, 0] += crop_col_min
        quadrilateral_xy[:, 1] += crop_row_min
        quadrilateral_matrix = np.fliplr(quadrilateral_xy)
        detector_swatches = np.asarray(
            detection.swatch_colours,
            dtype=np.float64,
        ).reshape(4, 6, 3)
        center_distance = float(
            np.linalg.norm(np.mean(quadrilateral_xy, axis=0) - cached_center)
            / cached_diagonal
        )
        area_ratio = _polygon_area_xy(quadrilateral_xy) / cached_area
        for reversed_order in (False, True):
            ordered = (
                quadrilateral_matrix[::-1]
                if reversed_order
                else quadrilateral_matrix
            )
            for shift in range(4):
                candidate_roi = np.rint(
                    np.roll(ordered, -shift, axis=0)
                ).astype(int)
                try:
                    correction.roi = darsia.make_voxel(candidate_roi)
                    extracted = extract_swatches(
                        correction._restrict_to_roi(image)
                    )
                    mismatch = float(
                        np.mean(
                            np.abs(
                                np.asarray(extracted, dtype=np.float64)
                                - detector_swatches
                            )
                        )
                    )
                    error = None
                except Exception as exc:  # pragma: no cover - diagnostic path.
                    mismatch = float("inf")
                    error = f"{type(exc).__name__}: {exc}"
                    extracted = None
                row: dict[str, object] = {
                    "detection_index": int(detection_index),
                    "reversed_order": bool(reversed_order),
                    "shift": int(shift),
                    "swatch_mismatch": mismatch,
                    "center_distance_normalized": center_distance,
                    "area_ratio_to_cached": float(area_ratio),
                    "roi": candidate_roi,
                    "swatches": extracted,
                }
                if error is not None:
                    row["error"] = error
                candidate_rows.append(row)
                if (
                    np.isfinite(mismatch)
                    and mismatch <= _MAX_DETECTED_SWATCH_MISMATCH
                    and 0.25 <= area_ratio <= 4.0
                    and center_distance <= 2.5
                ):
                    selected_rows.append(row)

    if not selected_rows:
        correction.roi = darsia.make_voxel(cached_roi)
        best_mismatch = min(
            (
                float(row["swatch_mismatch"])
                for row in candidate_rows
            ),
            default=float("inf"),
        )
        raise RuntimeError(
            "Detected ColorChecker Classic could not be validated against its "
            f"24 swatches; best mismatch={best_mismatch:.4f}"
        )

    winner = min(
        selected_rows,
        key=lambda row: (
            float(row["swatch_mismatch"]),
            float(row["center_distance_normalized"]),
            int(row["detection_index"]),
            bool(row["reversed_order"]),
            int(row["shift"]),
        ),
    )
    winner_roi = np.asarray(winner["roi"], dtype=int)
    winner_swatches = np.asarray(winner["swatches"], dtype=np.float32)
    correction.roi = darsia.make_voxel(winner_roi)
    diagnostics = {
        "search_crop_rc": [
            crop_row_min,
            crop_row_max,
            crop_col_min,
            crop_col_max,
        ],
        "working_width": working_width,
        "detection_count": len(detections),
        "selected_detection_index": int(winner["detection_index"]),
        "selected_reversed_order": bool(winner["reversed_order"]),
        "selected_shift": int(winner["shift"]),
        "selected_swatch_mismatch": float(winner["swatch_mismatch"]),
        "selected_center_distance_normalized": float(
            winner["center_distance_normalized"]
        ),
        "selected_area_ratio_to_cached": float(
            winner["area_ratio_to_cached"]
        ),
        "selected_roi": winner_roi.tolist(),
        "candidate_count": len(candidate_rows),
    }
    return winner_roi, winner_swatches, diagnostics


def auto_orient(correction: Any, baseline: Any) -> dict[str, object]:
    """Orient a cached checker and repair it if it is not the Classic chart."""

    image = np.asarray(baseline.img if isinstance(baseline, darsia.Image) else baseline)
    original_roi = np.asarray(correction.roi).copy()
    candidates = _orientation_candidates(correction, image, original_roi)
    winner = min(candidates, key=lambda row: float(row["score"]))
    localization: dict[str, object] = {
        "mode": "cached-roi",
        "triggered": False,
    }
    if not _cached_orientation_is_valid(winner):
        (
            detected_roi,
            detected_swatches,
            detection_diagnostics,
        ) = _detect_classic_colorchecker_roi(
            correction,
            image,
            original_roi,
        )
        winner = {
            "shift": int(detection_diagnostics["selected_shift"]),
            "roi": detected_roi,
            "swatches": detected_swatches,
            "neutral_chroma": float(
                np.mean(np.std(detected_swatches[-1], axis=1))
            ),
            "neutral_descending_violation": float(
                np.sum(
                    np.maximum(
                        np.diff(
                            detected_swatches[-1]
                            @ np.array([0.2126, 0.7152, 0.0722])
                        ),
                        0.0,
                    )
                )
            ),
            "neutral_dynamic_range": float(
                np.ptp(
                    detected_swatches[-1]
                    @ np.array([0.2126, 0.7152, 0.0722])
                )
            ),
            "score": float(detection_diagnostics["selected_swatch_mismatch"]),
        }
        localization = {
            "mode": "detected-classic-fallback",
            "triggered": True,
            **detection_diagnostics,
        }
    correction.roi = darsia.make_voxel(np.asarray(winner["roi"]))
    correction.colorchecker = darsia.CustomColorChecker(
        reference_colors=np.asarray(winner["swatches"], dtype=np.float32)
    )
    diagnostics = {
        "selected_shift": int(winner["shift"]),
        "selected_first_corner": [int(value) for value in correction.roi[0]],
        "selected_score": float(winner["score"]),
        "selected_neutral_chroma": float(winner["neutral_chroma"]),
        "selected_neutral_descending_violation": float(
            winner["neutral_descending_violation"]
        ),
        "selected_neutral_dynamic_range": float(winner["neutral_dynamic_range"]),
        "cached_orientation_valid": _cached_orientation_is_valid(
            min(candidates, key=lambda row: float(row["score"]))
        ),
        "localization": localization,
        "candidates": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"roi", "swatches"}
            }
            for candidate in candidates
        ],
    }
    correction.last_orientation_diagnostics = diagnostics
    return diagnostics


def _transform_is_bounded(correction: Any, diagnostics: Mapping[str, float]) -> bool:
    return bool(
        diagnostics["condition"] <= correction.residual_refinement_max_condition
        and diagnostics["diagonal_min"]
        >= correction.residual_refinement_diagonal_min
        and diagnostics["diagonal_max"]
        <= correction.residual_refinement_diagonal_max
        and diagnostics["off_diagonal_max_abs"]
        <= correction.residual_refinement_off_diagonal_max_abs
        and diagnostics["translation_max_abs"]
        <= correction.residual_refinement_translation_max_abs
    )


def _measure_residual(
    correction: Any,
    image: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    swatches = extract_swatches(correction._restrict_to_roi(image))
    summary = residual_summary(
        swatches,
        reference,
        swatch_weights=correction.swatch_weights,
        max_swatch_loss_weight=correction.max_swatch_loss_weight,
    )
    return swatches, summary


def _correct_array(correction: Any, image: np.ndarray) -> np.ndarray:
    if not getattr(correction, "active", False):
        return skimage.img_as_float(image).astype(np.float32)
    image = np.asarray(image)
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(
            "Provide image in np.uint8, np.uint16, np.float32, or np.float64 format."
        )

    reference = np.asarray(correction.colorchecker.swatches_rgb)
    checker_source = correction._restrict_to_roi(image)
    source = extract_swatches(checker_source)
    source_summary = residual_summary(
        source,
        reference,
        swatch_weights=correction.swatch_weights,
        max_swatch_loss_weight=correction.max_swatch_loss_weight,
    )
    correction.last_residual_before_correction = float(source_summary["mean"])
    _probe, tone_diagnostics = neutral_tone_curve_correction(
        np.zeros((1, 1, 3), dtype=np.float64), source, reference
    )
    corrected_checker = _apply_tone_diagnostics(
        checker_source,
        tone_diagnostics,
        dtype=np.float64,
    )
    tone_swatches, tone_summary = _measure_checker_residual(
        correction,
        corrected_checker,
        reference,
    )
    initial_tone_summary = tone_summary
    accepted_affines: list[tuple[np.ndarray, np.ndarray]] = []
    revision = getattr(correction, "algorithm_revision", "legacy-v2")
    residual_affine_enabled = revision != "neutral-tone-only-v4"
    if residual_affine_enabled:
        scaling, translation, affine_diagnostics = fit_regularized_residual_affine(
            srgb_to_linear(tone_swatches),
            srgb_to_linear(reference),
            ridge=max(0.0, correction.residual_refinement_ridge),
            huber_delta=max(0.0, correction.residual_refinement_huber_delta),
            sample_weights=correction.swatch_weights,
        )
        affine_bounds_ok = _transform_is_bounded(correction, affine_diagnostics)
    else:
        # The neutral chart constrains exposure and white balance. A residual
        # chromatic affine fit can extrapolate its additive/cross-channel terms
        # into the dark plume, below the chart's measured channel values. The
        # opt-in movie revision retains the exact neutral curves and reference
        # target, with no residual fit or data-dependent acceptance branch.
        affine_diagnostics = {"enabled": False, "reason": "neutral-tone-only-policy"}
        affine_bounds_ok = False
    affine_summary: dict[str, object] | None = None
    affine_accepted = False
    minimum_affine_improvement = (
        correction.residual_refinement_min_improvement
        if revision == "minimum-improvement-v3"
        else 0.0
    )
    affine_relative_improvement = None
    affine_rejection_reason = (
        "neutral-tone-only-policy" if not residual_affine_enabled
        else "transform-bounds" if not affine_bounds_ok else None
    )
    if affine_bounds_ok:
        affine_candidate = _apply_affine(
            corrected_checker,
            scaling,
            translation,
            dtype=np.float64,
        )
        affine_swatches, affine_summary = _measure_checker_residual(
            correction,
            affine_candidate,
            reference,
        )
        affine_relative_improvement = (
            float(tone_summary["score"]) - float(affine_summary["score"])
        ) / max(float(tone_summary["score"]), 1e-12)
        if (
            np.isfinite(float(affine_summary["score"]))
            and float(affine_summary["score"]) < float(tone_summary["score"])
            and affine_relative_improvement >= minimum_affine_improvement
        ):
            corrected_checker = affine_candidate
            tone_swatches = affine_swatches
            tone_summary = affine_summary
            affine_accepted = True
            accepted_affines.append((scaling.copy(), translation.copy()))
        else:
            affine_rejection_reason = (
                "nonfinite-score" if not np.isfinite(float(affine_summary["score"]))
                else "insufficient-improvement"
            )
    correction.last_strategy_diagnostics = {
        "mode": "neutral-tone-affine" if residual_affine_enabled else "neutral-tone-only",
        "residual_affine_enabled": residual_affine_enabled,
        "residual_model": "robust-affine-linear-rgb" if residual_affine_enabled else "none",
        "tone": tone_diagnostics,
        "tone_residual": float(initial_tone_summary["mean"]),
        "tone_weighted_residual": float(initial_tone_summary["weighted_mean"]),
        "tone_residual_score": float(initial_tone_summary["score"]),
        "affine": affine_diagnostics,
        "affine_bounds_ok": affine_bounds_ok,
        "affine_residual": (
            float(affine_summary["mean"])
            if affine_summary is not None
            else float("nan")
        ),
        "affine_weighted_residual": (
            float(affine_summary["weighted_mean"])
            if affine_summary is not None
            else float("nan")
        ),
        "affine_residual_score": (
            float(affine_summary["score"])
            if affine_summary is not None
            else float("nan")
        ),
        "affine_accepted": affine_accepted,
        "algorithm_id": getattr(correction, "algorithm_id", ALGORITHM_ID),
        "algorithm_revision": getattr(correction, "algorithm_revision", "legacy-v2"),
        "affine_relative_improvement": affine_relative_improvement,
        "affine_minimum_improvement": float(minimum_affine_improvement),
        "affine_rejection_reason": affine_rejection_reason,
    }

    current_swatches = tone_swatches
    current_summary = tone_summary
    correction.last_residual_before_refinement = float(current_summary["mean"])
    correction.last_refinement_steps_applied = 0
    correction.last_refinement_diagnostics = []
    refinement_steps = max(0, int(correction.residual_refinement_steps)) if residual_affine_enabled else 0
    for step in range(refinement_steps):
        scaling, translation, diagnostics = fit_regularized_residual_affine(
            srgb_to_linear(current_swatches),
            srgb_to_linear(reference),
            ridge=max(0.0, correction.residual_refinement_ridge),
            huber_delta=max(0.0, correction.residual_refinement_huber_delta),
            sample_weights=correction.swatch_weights,
        )
        before = float(current_summary["score"])
        bounded = _transform_is_bounded(correction, diagnostics)
        record: dict[str, object] = {
            "step": step + 1,
            "residual_before": float(current_summary["mean"]),
            "weighted_residual_before": float(current_summary["weighted_mean"]),
            "residual_score_before": before,
            **diagnostics,
            "bounds_ok": bounded,
        }
        if not bounded:
            record.update(accepted=False, reason="transform-bounds")
            correction.last_refinement_diagnostics.append(record)
            break
        candidate = _apply_affine(
            corrected_checker,
            scaling,
            translation,
            dtype=np.float64,
        )
        candidate_swatches, candidate_summary = _measure_checker_residual(
            correction,
            candidate,
            reference,
        )
        after = float(candidate_summary["score"])
        improvement = (before - after) / max(before, 1e-12)
        record.update(
            residual_after=float(candidate_summary["mean"]),
            weighted_residual_after=float(candidate_summary["weighted_mean"]),
            residual_score_after=after,
            relative_improvement=float(improvement),
            minimum_improvement=float(correction.residual_refinement_min_improvement),
        )
        if (
            not np.isfinite(after)
            or improvement < correction.residual_refinement_min_improvement
        ):
            record.update(accepted=False, reason="insufficient-improvement")
            correction.last_refinement_diagnostics.append(record)
            break
        record["accepted"] = True
        correction.last_refinement_diagnostics.append(record)
        corrected_checker = candidate
        current_swatches = candidate_swatches
        current_summary = candidate_summary
        accepted_affines.append((scaling.copy(), translation.copy()))
        correction.last_refinement_steps_applied += 1

    final_summary = residual_summary(
        current_swatches,
        reference,
        swatch_weights=correction.swatch_weights,
        max_swatch_loss_weight=correction.max_swatch_loss_weight,
    )
    correction.last_swatch_residuals = np.asarray(
        final_summary["residuals"], dtype=np.float64
    )
    correction.last_residual = float(final_summary["mean"])
    correction.last_weighted_residual = float(final_summary["weighted_mean"])
    correction.last_p95_swatch_residual = float(final_summary["p95"])
    correction.last_max_swatch_residual = float(final_summary["max"])
    correction.last_residual_score = float(final_summary["score"])
    correction.last_worst_swatch_index = int(final_summary["worst_swatch_index"])
    correction.last_worst_swatch_name = str(final_summary["worst_swatch_name"])
    flag_reasons: list[str] = []
    if correction.last_residual > correction.residual_warn_threshold:
        flag_reasons.append("mean-residual")
    if correction.last_weighted_residual > correction.residual_warn_threshold:
        flag_reasons.append("weighted-residual")
    if (
        correction.last_max_swatch_residual
        > correction.max_swatch_warn_threshold
    ):
        flag_reasons.append("max-swatch-residual")
    correction.last_flag_reasons = flag_reasons
    correction.last_flagged = bool(flag_reasons)
    if correction.last_flagged:
        warn(
            "ColorChecker refinement left a high residual: "
            f"mean={correction.last_residual:.4f}, "
            f"weighted={correction.last_weighted_residual:.4f}, "
            f"max={correction.last_max_swatch_residual:.4f} "
            f"({correction.last_worst_swatch_name}); "
            f"reasons={','.join(flag_reasons)}."
        )
    work_dtype_name = os.environ.get(
        "FFAC_COLORCHECKER_WORK_DTYPE", "float64"
    ).strip().lower()
    work_dtype = np.float32 if work_dtype_name == "float32" else np.float64
    corrected = _apply_pipeline_tiled(
        image,
        tone_diagnostics,
        accepted_affines,
        dtype=work_dtype,
    )
    if getattr(correction, "clip", False):
        corrected = np.clip(corrected, 0.0, 1.0)
    return np.asarray(corrected, dtype=np.float32)


def install(
    correction: Any,
    *,
    baseline: Any,
    auto_orientation: bool,
    settings: Mapping[str, float | int],
    algorithm_revision: str = "legacy-v2",
) -> dict[str, object]:
    """Install the FF-AC correction on one DarSIA correction instance."""

    if algorithm_revision not in ALGORITHM_REVISIONS:
        raise ValueError(f"Unsupported ColorChecker algorithm revision: {algorithm_revision!r}")
    correction.algorithm_revision = algorithm_revision
    correction.algorithm_id = ALGORITHM_REVISIONS[algorithm_revision]
    orientation = auto_orient(correction, baseline) if auto_orientation else None
    for name, value in settings.items():
        setattr(correction, name, value)
    correction.btb_swatch_weight = float(
        settings.get("btb_swatch_weight", 1.0)
    )
    correction.max_swatch_loss_weight = float(
        settings.get("max_swatch_loss_weight", 0.0)
    )
    correction.max_swatch_warn_threshold = float(
        settings.get("max_swatch_warn_threshold", 1.0)
    )
    correction.swatch_weights = build_swatch_weights(
        correction.btb_swatch_weight
    )
    correction.correction_strategy = (
        "neutral-tone-only" if algorithm_revision == "neutral-tone-only-v4"
        else "neutral-tone-affine"
    )
    correction.last_residual = float("nan")
    correction.last_weighted_residual = float("nan")
    correction.last_residual_score = float("nan")
    correction.last_residual_before_correction = float("nan")
    correction.last_residual_before_refinement = float("nan")
    correction.last_p95_swatch_residual = float("nan")
    correction.last_max_swatch_residual = float("nan")
    correction.last_worst_swatch_index = -1
    correction.last_worst_swatch_name = ""
    correction.last_swatch_residuals = np.empty(0, dtype=np.float64)
    correction.last_refinement_steps_applied = 0
    correction.last_refinement_diagnostics = []
    correction.last_strategy_diagnostics = {"mode": correction.correction_strategy}
    correction.last_flag_reasons = []
    correction.last_flagged = False
    if not hasattr(correction, "_ffac_original_correct_array"):
        correction._ffac_original_correct_array = correction.correct_array
    correction.correct_array = MethodType(_correct_array, correction)
    return {
        "type": type(correction).__name__,
        "implementation": "scripts.colorchecker_refinement",
        "algorithm_id": correction.algorithm_id,
        "algorithm_revision": algorithm_revision,
        "auto_orient": bool(auto_orientation),
        "strategy": correction.correction_strategy,
        "residual_affine_enabled": algorithm_revision != "neutral-tone-only-v4",
        "residual_model": "none" if algorithm_revision == "neutral-tone-only-v4" else "robust-affine-linear-rgb",
        "swatch_names": list(SWATCH_NAMES),
        "btb_sensitive_swatch_indices": list(BTB_SENSITIVE_SWATCH_INDICES),
        "swatch_weights": [float(value) for value in correction.swatch_weights],
        "orientation": orientation,
        **dict(settings),
    }
