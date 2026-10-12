"""Shared validation and metadata for DarSIA signal-stage restoration.

DarSIA commit 53ddb288 moves REV/volume averaging from the heterogeneous
color-path coordinate to the common signal domain.  FF-AC's accelerator
backends bypass ``ConcentrationAnalysis.__call__`` and therefore need an
explicit, validated description of that restoration step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


DARSIA_SIGNAL_RESTORATION_COMMIT = (
    "53ddb288f88c4957b3e71174205459bd7c375af4"
)


@dataclass(frozen=True)
class SignalRestorationSpec:
    """Backend-neutral description of homogeneous DarSIA volume averaging."""

    rev_size: int
    mask: np.ndarray
    mean_pore_volume: np.ndarray
    zero_mask: np.ndarray


def signal_restoration_spec(
    calibration: Any,
    shape: Sequence[int],
) -> SignalRestorationSpec | None:
    """Validate and describe the restoration used after signal conversion.

    A restoration object on ``color_analysis`` identifies the pre-53ddb288
    pipeline and is rejected.  Silently accepting it would make CPU and GPU
    evaluate different physical models.
    """

    target = (int(shape[0]), int(shape[1]))
    color_analysis = getattr(calibration, "color_analysis", None)
    color_restoration = getattr(color_analysis, "restoration", None)
    if color_restoration is not None:
        raise RuntimeError(
            "DarSIA restoration is attached to color_analysis. Apply DarSIA "
            f"fix {DARSIA_SIGNAL_RESTORATION_COMMIT} before calibration."
        )

    signal_analysis = getattr(calibration, "signal_model", None)
    restoration = getattr(signal_analysis, "restoration", None)
    if restoration is None:
        return None
    required = ("rev_size", "mask", "mean_pore_volume", "zero_indices")
    missing = [name for name in required if not hasattr(restoration, name)]
    if missing:
        raise RuntimeError(
            "Unsupported signal restoration; missing " + ", ".join(missing)
        )
    if getattr(restoration, "labels", None) is not None:
        raise RuntimeError(
            "Accelerated signal restoration currently supports homogeneous "
            "DarSIA VolumeAveraging only."
        )

    rev_size = int(restoration.rev_size)
    if rev_size < 1:
        raise RuntimeError(f"Invalid signal restoration REV size {rev_size}.")
    mask = np.asarray(restoration.mask.img, dtype=np.float64)
    mean_pore_volume = np.asarray(
        restoration.mean_pore_volume,
        dtype=np.float64,
    )
    if mask.shape != target or mean_pore_volume.shape != target:
        raise RuntimeError(
            "Signal restoration shape mismatch: "
            f"mask={mask.shape}, mean={mean_pore_volume.shape}, frame={target}."
        )
    zero_mask = np.zeros(target, dtype=bool)
    zero_mask[restoration.zero_indices] = True
    return SignalRestorationSpec(
        rev_size=rev_size,
        mask=np.ascontiguousarray(mask),
        mean_pore_volume=np.ascontiguousarray(mean_pore_volume),
        zero_mask=np.ascontiguousarray(zero_mask),
    )


def restore_signal_numpy(
    values: np.ndarray,
    spec: SignalRestorationSpec | None,
) -> np.ndarray:
    """Reference implementation matching DarSIA ``VolumeAveraging``."""

    array = np.asarray(values, dtype=np.float64)
    if spec is None:
        return array
    if array.shape != spec.mask.shape:
        raise RuntimeError(
            f"Signal shape {array.shape} differs from restoration {spec.mask.shape}."
        )
    import scipy.ndimage

    numerator = scipy.ndimage.uniform_filter(
        array * spec.mask,
        size=spec.rev_size,
    )
    restored = np.divide(
        numerator,
        spec.mean_pore_volume,
        out=np.zeros_like(numerator),
        where=~spec.zero_mask,
    )
    restored[spec.zero_mask] = 0.0
    return restored
