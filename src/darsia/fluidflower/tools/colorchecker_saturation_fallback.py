"""Saturated neutral patches: a flagged fallback for the ColorChecker tone curve.

The refinement in scripts/colorchecker_refinement.py maps every photograph onto
the chart as read in the run's own baseline through a curve anchored on the six
neutral patches, and refuses a photograph whose curve has a neighbour slope
outside [0.05, 8]. When the brightest patches are clipped in the raw photograph
(FF-UM RUN14: two clipped in the baseline, three in the clear-water reference
and the late frames) the curve has no information at the top, the interval from
the last measured grey to the clipped ones becomes a steep jump, and the whole
run is refused although the darker patches still measure the drift. The mirror
image happens at the dark end (two indistinguishable or black darkest patches: the
flattened step gives a zero slope). A single patch at full scale is a clipping
signature too: a white at exactly 1.0 in the reference while the photograph is darker
(FF-AC ac50's clear-water reference and ac51's late frames: the baseline's blue white
is clipped alone) makes the (1, 1) end point a zero-rise interval, and a photograph
white just below full scale against a darker reference (FF-AC ac53 at 48 h) makes
the interval to (1, 1) a steep jump.

This module installs a fallback from OUTSIDE that file, which stays byte for
byte the same (its source hash is a deployment fingerprint of the dual-signal
models). Nothing changes for a photograph the frozen function accepts: the
wrapper calls it first and acts only when it raises. It then removes the
neutral patches that are indistinguishable at the top or at the bottom of the
grey ramp, or at full scale (white or black level), in the photograph or in the
reference - the signature of clipping - rebuilds the curve
from the remaining patches with the frozen rules (at least three measured
patches, (0, 0) and (1, 1) end points, the same span, white-clipping and slope
guards), flags the frame and records what was excluded. If the rebuilt curve
still fails, the original error is raised unchanged.

Kill switch: PHASE_COLORCHECKER_SATURATION_FALLBACK=0. The wrapper is only
installed when the frozen module has the verified source hash.
"""
from __future__ import annotations

from collections import deque
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping
import warnings

import numpy as np

from darsia.fluidflower.tools import colorchecker_refinement as refinement

VERSION = "saturated-neutral-patch-fallback-v3"   # v2: dark end too; v3: a lone patch at full scale
ENV = "PHASE_COLORCHECKER_SATURATION_FALLBACK"
FLAG_REASON = "saturated-neutral-patches-excluded"
PLATEAU_TOLERANCE_SRGB = 3.0 / 255.0           # patches closer than this to the brightest/darkest one are one plateau
MINIMUM_PLATEAU_PATCHES = 2                    # a lone maximum is just the white patch (alone at full scale: clipped)
MINIMUM_MEASURED_PATCHES = 3
VERIFIED_SOURCE_SHA256 = "9bf9ba49cf906d3b8bb116e158af28844b126d14bab9d6f4a7b4dc256b39eadd"

_frozen_tone_curve = refinement.neutral_tone_curve_correction
_frozen_correct_array = refinement._correct_array
_state = threading.local()
_lock = threading.Lock()
_installed: dict[str, Any] | None = None
events: deque = deque(maxlen=10000)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def source_sha256() -> str:
    return hashlib.sha256(Path(refinement.__file__).read_bytes()).hexdigest()


def top_plateau(values: np.ndarray, tolerance: float = PLATEAU_TOLERANCE_SRGB) -> list[int]:
    """Indices of the patches indistinguishable from the brightest one (at least two), or at the white level."""
    values = np.asarray(values, dtype=np.float64)
    plateau = [int(index) for index in np.flatnonzero(np.max(values) - values <= tolerance)]
    if len(plateau) < MINIMUM_PLATEAU_PATCHES:
        plateau = []
    white = [int(index) for index in np.flatnonzero(values >= 1.0 - tolerance)]
    return sorted(set(plateau) | set(white))


def bottom_plateau(values: np.ndarray, tolerance: float = PLATEAU_TOLERANCE_SRGB) -> list[int]:
    """Indices of the patches indistinguishable from the darkest one (at least two), or at the black level."""
    values = np.asarray(values, dtype=np.float64)
    plateau = [int(index) for index in np.flatnonzero(values - np.min(values) <= tolerance)]
    if len(plateau) < MINIMUM_PLATEAU_PATCHES:
        plateau = []
    black = [int(index) for index in np.flatnonzero(values <= tolerance)]
    return sorted(set(plateau) | set(black))


def excluded_patches(source_row: np.ndarray, destination_row: np.ndarray, channel: int) -> dict[str, list[int]]:
    source_values = np.asarray(source_row)[:, channel]
    destination_values = np.asarray(destination_row)[:, channel]
    source_top, source_bottom = top_plateau(source_values), bottom_plateau(source_values)
    destination_top, destination_bottom = top_plateau(destination_values), bottom_plateau(destination_values)
    source = sorted(set(source_top) | set(source_bottom))
    destination = sorted(set(destination_top) | set(destination_bottom))
    return {
        "source": source,
        "destination": destination,
        "bright_end": sorted(set(source_top) | set(destination_top)),
        "dark_end": sorted(set(source_bottom) | set(destination_bottom)),
        "excluded": sorted(set(source) | set(destination)),
    }


def _channel_curve(source_values: np.ndarray, destination_values: np.ndarray, channel: int) -> dict[str, Any]:
    """The frozen per-channel construction, applied to the retained patches (linear light in, anchors out)."""
    order = np.argsort(source_values)
    source_sorted = source_values[order]
    destination_sorted = destination_values[order]
    monotonic_violation = float(np.sum(np.maximum(-np.diff(destination_sorted), 0.0)))
    if monotonic_violation > 0.025:
        raise ValueError("Neutral ColorChecker response is not monotone enough for "
                         f"channel {channel}: violation={monotonic_violation:.6f}")
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
    white_quantization_floor = float(refinement.srgb_to_linear(np.asarray(254.0 / 255.0)))
    clipped_white_intervals = (
        (x_array[:-1] >= white_quantization_floor)
        & (y_array[:-1] >= white_quantization_floor)
        & (x_array[1:] == 1.0)
        & (y_array[1:] == 1.0)
        & (source_sorted[-1] == 1.0)
        & (destination_sorted[-1] == 1.0)
    )
    checked_slopes = slopes[(intervals >= refinement._MIN_TONE_SLOPE_SPAN) & ~clipped_white_intervals]
    if not checked_slopes.size:
        raise ValueError(f"Neutral ColorChecker tone curve has no measurable intervals for channel {channel}")
    if np.min(checked_slopes) < 0.05 or np.max(checked_slopes) > 8.0:
        raise ValueError("Neutral ColorChecker tone curve exceeds slope bounds for "
                         f"channel {channel}: [{np.min(checked_slopes):.4f}, {np.max(checked_slopes):.4f}]")
    return {
        "channel": channel,
        "anchors_x": [float(value) for value in x_array],
        "anchors_y": [float(value) for value in y_array],
        "slope_min": float(np.min(slopes)),
        "slope_max": float(np.max(slopes)),
        "checked_slope_min": float(np.min(checked_slopes)),
        "checked_slope_max": float(np.max(checked_slopes)),
        "ignored_short_interval_count": int(np.count_nonzero(intervals < refinement._MIN_TONE_SLOPE_SPAN)),
        "ignored_clipped_white_interval_count": int(np.count_nonzero(clipped_white_intervals)),
        "monotonic_violation": monotonic_violation,
    }


def rebuild_without_saturated_patches(image: np.ndarray, source_swatches: np.ndarray,
                                      destination_swatches: np.ndarray) -> tuple[np.ndarray, dict[str, Any]] | None:
    """The frozen curve on the retained patches, or None when no channel shows a saturated plateau."""
    source_row = np.asarray(source_swatches, dtype=np.float64)[-1]
    destination_row = np.asarray(destination_swatches, dtype=np.float64)[-1]
    exclusions = [excluded_patches(source_row, destination_row, channel) for channel in range(3)]
    if not any(item["excluded"] for item in exclusions):
        return None
    source = refinement.srgb_to_linear(source_swatches)[-1]
    destination = refinement.srgb_to_linear(destination_swatches)[-1]
    image_linear = refinement.srgb_to_linear(image)
    corrected_linear = np.empty_like(image_linear)
    channels: list[dict[str, Any]] = []
    for channel, exclusion in enumerate(exclusions):
        keep = [index for index in range(source.shape[0]) if index not in exclusion["excluded"]]
        if len(keep) < MINIMUM_MEASURED_PATCHES:
            raise ValueError("Neutral ColorChecker tone curve keeps fewer than "
                             f"{MINIMUM_MEASURED_PATCHES} unsaturated patches for channel {channel}")
        model = _channel_curve(source[keep, channel], destination[keep, channel], channel)
        model["excluded_patch_indices"] = list(exclusion["excluded"])
        model["excluded_in_source"] = list(exclusion["source"])
        model["excluded_in_destination"] = list(exclusion["destination"])
        model["excluded_bright_end"] = list(exclusion["bright_end"])
        model["excluded_dark_end"] = list(exclusion["dark_end"])
        corrected_linear[..., channel] = np.interp(np.clip(image_linear[..., channel], 0.0, 1.0),
                                                  np.asarray(model["anchors_x"]), np.asarray(model["anchors_y"]))
        channels.append(model)
    return refinement.linear_to_srgb(corrected_linear), {"mode": "neutral-tone", "channels": channels}


def tone_curve_with_fallback(image: np.ndarray, source_swatches: np.ndarray,
                             destination_swatches: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The frozen function first; the flagged rebuild only when it raises and a plateau explains it."""
    try:
        return _frozen_tone_curve(image, source_swatches, destination_swatches)
    except ValueError as error:
        if not enabled():
            raise
        try:
            rebuilt = rebuild_without_saturated_patches(image, source_swatches, destination_swatches)
        except ValueError:
            rebuilt = None
        if rebuilt is None:
            raise
        corrected, diagnostics = rebuilt
        record = {
            "version": VERSION,
            "original_error": str(error),
            "plateau_tolerance_srgb": PLATEAU_TOLERANCE_SRGB,
            "channels": [{"channel": model["channel"], "excluded_patch_indices": model["excluded_patch_indices"],
                          "excluded_in_source": model["excluded_in_source"],
                          "excluded_in_destination": model["excluded_in_destination"],
                          "excluded_bright_end": model["excluded_bright_end"],
                          "excluded_dark_end": model["excluded_dark_end"],
                          "checked_slope_min": model["checked_slope_min"],
                          "checked_slope_max": model["checked_slope_max"]} for model in diagnostics["channels"]],
            "neutral_row_source_srgb": [[float(v) for v in patch] for patch in np.asarray(source_swatches)[-1]],
            "neutral_row_destination_srgb": [[float(v) for v in patch] for patch in np.asarray(destination_swatches)[-1]],
            "unix_time": time.time(),
        }
        diagnostics["saturation_fallback"] = record
        with _lock:
            events.append(record)
        _state.record = record                        # attributed on this thread, see correct_array_with_flag
        return corrected, diagnostics


def correct_array_with_flag(correction: Any, image: np.ndarray) -> np.ndarray:
    """The frozen _correct_array, then the frame is flagged when the fallback was used for it."""
    _state.record = None
    corrected = _frozen_correct_array(correction, image)
    record = getattr(_state, "record", None)
    _state.record = None
    _state.last_record = record                       # per thread: the preparation attributes it to its frame
    # The correction object is shared between the preparation threads; its attributes are informational.
    correction.last_saturation_fallback = record
    if record is not None:
        reasons = list(getattr(correction, "last_flag_reasons", []) or [])
        if FLAG_REASON not in reasons:
            reasons.append(FLAG_REASON)
        correction.last_flag_reasons = reasons
        correction.last_flagged = True
        excluded = {model["channel"]: model["excluded_patch_indices"] for model in record["channels"]}
        warnings.warn("ColorChecker tone curve rebuilt without saturated neutral patches "
                      f"({excluded}); the frame is flagged. Original error: {record['original_error']}")
    return corrected


def take_last_record() -> dict[str, Any] | None:
    """The fallback record of the last correct_array call on THIS thread, cleared on reading."""
    record = getattr(_state, "last_record", None)
    _state.last_record = None
    return record


def install(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Wrap the frozen functions once; report what was done. Safe to call repeatedly."""
    global _installed
    with _lock:
        if _installed is not None:
            return dict(_installed)
        actual = source_sha256()
        if not enabled(environ):
            _installed = {"version": VERSION, "installed": False, "reason": f"{ENV}=0", "source_sha256": actual}
        elif actual != VERIFIED_SOURCE_SHA256:
            _installed = {"version": VERSION, "installed": False, "source_sha256": actual,
                          "reason": "scripts/colorchecker_refinement.py differs from the verified source"}
        else:
            refinement.neutral_tone_curve_correction = tone_curve_with_fallback
            refinement._correct_array = correct_array_with_flag
            _installed = {"version": VERSION, "installed": True, "source_sha256": actual,
                          "plateau_tolerance_srgb": PLATEAU_TOLERANCE_SRGB,
                          "minimum_measured_patches": MINIMUM_MEASURED_PATCHES, "flag_reason": FLAG_REASON}
        return dict(_installed)


def uninstall() -> None:
    global _installed
    with _lock:
        refinement.neutral_tone_curve_correction = _frozen_tone_curve
        refinement._correct_array = _frozen_correct_array
        _installed = None


def report(limit: int = 200) -> dict[str, Any]:
    """For preparation and inference reports: the installation and the frames that needed the fallback."""
    with _lock:
        recorded = list(events)
    return {**(dict(_installed) if _installed is not None else {"version": VERSION, "installed": False,
                                                                 "reason": "install() was not called"}),
            "events": len(recorded), "recent_events": recorded[-limit:]}
