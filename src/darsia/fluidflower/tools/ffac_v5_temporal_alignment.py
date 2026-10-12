"""Frame-exact reference alignment for FF_AC V5 optical evidence.

Sand can move within an experiment.  V5 photometry corrects colour but does
not align the baseline and previous photographs used by optical differences.
Only those reference arrays move here: current pixels and user annotations stay
in their original native review coordinates.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _h(value: Any) -> np.ndarray:
    matrix = np.asarray(value, np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(float(np.linalg.det(matrix))) < 1e-9
            or abs(float(matrix[2, 2])) < 1e-9):
        raise ValueError("V5 temporal alignment requires an invertible 3x3 H")
    return matrix / matrix[2, 2]


def _rgb(value: Any, name: str) -> np.ndarray:
    rgb = np.asarray(value)
    if (rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3
            or not np.isfinite(rgb).all() or np.any(rgb < 0)
            or np.any(rgb > 1)):
        raise ValueError(f"{name} must be finite HWC float32 unit RGB")
    return rgb


def warp_reference_to_current(reference: np.ndarray, h_reference_to_current: Any,
                              *, required_valid_mask: np.ndarray | None = None,
                              target_shape_hw: tuple[int, int] | None = None,
                              source_origin_xy: tuple[int, int] = (0, 0),
                              source_valid_mask: np.ndarray | None = None,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return a reference on the current native grid and its exact support.

    ``source_origin_xy`` locates the reference array's (0, 0) in native review
    coordinates.  An extended source-backed RAW reference can therefore carry
    real sensor pixels beyond the old camera-JPEG crop.  It fails closed if
    any required pixel would still need extrapolation.
    """
    reference = _rgb(reference, "Reference")
    if source_valid_mask is None:
        source_valid = np.ones(reference.shape[:2], np.float32)
    else:
        source_valid = np.asarray(source_valid_mask)
        if source_valid.dtype != np.bool_ or source_valid.shape != reference.shape[:2]:
            raise ValueError("V5 reference valid mask must be source HW bool")
        source_valid = source_valid.astype(np.float32)
    if target_shape_hw is None:
        target_shape_hw = reference.shape[:2]
    height, width = (int(target_shape_hw[0]), int(target_shape_hw[1]))
    if height <= 0 or width <= 0 or len(source_origin_xy) != 2:
        raise ValueError("V5 reference warp needs a valid target shape/origin")
    origin = (int(source_origin_xy[0]), int(source_origin_xy[1]))
    if tuple(source_origin_xy) != origin:
        raise ValueError("V5 source origin must be integer native coordinates")
    h = _h(h_reference_to_current)
    source_to_native = np.array([[1., 0., origin[0]], [0., 1., origin[1]],
                                 [0., 0., 1.]], np.float64)
    h_source_to_current = _h(h @ source_to_native)
    aligned = cv2.warpPerspective(reference, h_source_to_current, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0.0, 0.0, 0.0))
    # For bilinear interpolation the entire source footprint must exist.  A
    # fractional support weight is not enough for weak CO2 at the edge.
    footprint = cv2.warpPerspective(source_valid,
                                    h_source_to_current,
                                    (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0.0)
    valid = footprint >= np.float32(1 - 1e-6)
    if required_valid_mask is not None:
        required = np.asarray(required_valid_mask)
        if required.dtype != np.bool_ or required.shape != (height, width):
            raise ValueError("Required optical support mask must be native HW bool")
        if np.any(required & ~valid):
            raise ValueError("V5 reference warp lacks source pixels in required region")
    return np.asarray(aligned, np.float32), valid


def align_temporal_inputs(
    *, baseline: np.ndarray, current: np.ndarray,
    previous: np.ndarray | None,
    h_baseline_to_current: Any,
    h_baseline_to_previous: Any | None = None,
    required_valid_mask: np.ndarray | None = None,
    baseline_origin_xy: tuple[int, int] = (0, 0),
    previous_origin_xy: tuple[int, int] = (0, 0),
    baseline_source_valid_mask: np.ndarray | None = None,
    previous_source_valid_mask: np.ndarray | None = None,
    return_support_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Align baseline and previous into current native coordinates.

    The two transforms must be certified by source-bound per-frame receipts.
    ``previous`` cannot be used with an omitted previous-frame transform.
    """
    baseline = _rgb(baseline, "Baseline")
    current = _rgb(current, "Current")
    if baseline.shape != current.shape and baseline_origin_xy == (0, 0):
        raise ValueError("Extended baseline needs an explicit native source origin")
    h_current = _h(h_baseline_to_current)
    aligned_baseline, baseline_valid = warp_reference_to_current(
        baseline, h_current, required_valid_mask=required_valid_mask,
        target_shape_hw=current.shape[:2], source_origin_xy=baseline_origin_xy,
        source_valid_mask=baseline_source_valid_mask)
    aligned_previous = None
    previous_valid = None
    if previous is not None:
        previous = _rgb(previous, "Previous")
        if (h_baseline_to_previous is None or
                (previous.shape != current.shape and previous_origin_xy == (0, 0))):
            raise ValueError("Previous input needs its exact same-grid H")
        h_previous_to_current = _h(h_current @ np.linalg.inv(_h(h_baseline_to_previous)))
        aligned_previous, previous_valid = warp_reference_to_current(
            previous, h_previous_to_current, required_valid_mask=required_valid_mask,
            target_shape_hw=current.shape[:2], source_origin_xy=previous_origin_xy,
            source_valid_mask=previous_source_valid_mask)
    report = dict(
        baseline_valid_fraction=float(np.mean(baseline_valid)),
        previous_valid_fraction=(None if previous_valid is None
                                 else float(np.mean(previous_valid))),
        reference_motion_policy="warp-baseline-and-previous-to-current-native-v1",
        baseline_source_origin_xy=list(baseline_origin_xy),
        previous_source_origin_xy=(None if previous is None else list(previous_origin_xy)),
        current_pixels_unchanged=True,
    )
    if return_support_mask:
        # Only the caller's in-memory sampler consumes this mask.  Remove it
        # before writing JSON timings or a training receipt.
        report["joint_valid_mask"] = (
            baseline_valid if previous_valid is None else baseline_valid & previous_valid)
    return aligned_baseline, aligned_previous, report


__all__ = ["warp_reference_to_current", "align_temporal_inputs"]
