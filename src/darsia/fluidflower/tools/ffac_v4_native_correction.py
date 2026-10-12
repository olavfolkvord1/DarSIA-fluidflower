"""Validation-only V4 FF_AC native photometric correction core.

The caller must first verify the exact RAW/JPEG/native frame triple, the
revision-keyed V4 baseline fit, and this frame's sand/ColorChecker geometry.
This function applies only the source-bound light correction. It cannot
approve phase labels, chemistry alignment, or scientific mass analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from darsia.fluidflower.tools.ffac_v5_refgeom import (
    apply_refgeom_v5, estimate_frame_light_refgeom, validate_homography,
)
from darsia.fluidflower.tools.ffac_v5_source_processor import (
    _ea_native_light_grid, _repair_isolated_porous_holes,
)


NATIVE = (2979, 5472)


def correct_v4_native_signal(
    *, linear_camera_rgb: np.ndarray, fit_ref: Any, ac28_ref_fit: Any,
    h_ac28_to_frame_native: np.ndarray,
    optical_safe_interior: np.ndarray, porous_domain: np.ndarray,
    exact_frozen_baseline: bool,
    frame_chart_xy8: np.ndarray | None = None,
    frame_chart_usable: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct one full-native EA RAW frame using its certified frame H.

    ``optical_safe_interior`` is binary 0/1 and *not* the uncertain physical
    facies map. The full projected porous domain is retained for label support
    and sparse-hole checks. V4 leaves any optional chemistry mapping out.
    """
    linear = np.asarray(linear_camera_rgb)
    safe = np.asarray(optical_safe_interior)
    domain = np.asarray(porous_domain)
    if linear.shape != (3, *NATIVE) or linear.dtype != np.float32:
        raise ValueError("V4 requires full-native EA camera-linear float32 RGB")
    if (safe.shape != NATIVE or domain.shape != NATIVE
            or safe.dtype != np.bool_ or domain.dtype != np.bool_
            or not np.any(safe) or np.any(safe & ~domain)):
        raise ValueError("V4 optical safe interior must be a nonempty porous subset")
    h = validate_homography(h_ac28_to_frame_native)
    if exact_frozen_baseline:
        if frame_chart_xy8 is not None or frame_chart_usable is not None:
            raise ValueError("Exact V4 baseline must not refit a frame-light field")
        light_field = None
        light_info = dict(mode="exact-revision-bound-B", zero_field=True)
    else:
        if frame_chart_xy8 is None or frame_chart_usable is None:
            raise ValueError("V4 frame needs exact same-frame chart positions")
        grid = _ea_native_light_grid(linear)
        light_field, light_info = estimate_frame_light_refgeom(
            fit_ref, grid, h, frame_chart_xy_run8=frame_chart_xy8,
            frame_chart_usable=frame_chart_usable)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_linear = np.where(linear > 2, np.log(linear), np.nan).astype(np.float32)
    corrected = apply_refgeom_v5(
        log_linear, fit_ref, ac28_ref_fit, h,
        facies_run_native=safe.astype(np.uint8),
        light_field_ref8=light_field)
    corrected = np.asarray(corrected, np.float32)
    corrected, repair = _repair_isolated_porous_holes(
        corrected, domain.astype(np.uint8))
    return corrected, dict(
        contract="ffac-v4-native-light-only-correction-v1",
        frame_light=light_info,
        sparse_porous_hole_repair=repair,
        optical_support="source-bound-binary-safe-interior",
        full_porous_domain_preserved=True,
        chemistry_alignment="none-light-only",
        validation_only=True, scientific_mass_eligible=False,
    )


__all__ = ["correct_v4_native_signal"]
