"""Source-geometry-aware optical inference for validation-only V5 frames.

The current photograph and its annotations stay in their native review grid.
Only the corrected pre-injection and preceding photographs are registered to
that grid before computing baseline-relative features.  The caller supplies
homographies from separately verified, SHA-bound frame receipts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ffac_v5_temporal_alignment import align_temporal_inputs
from .phase_inference_benchmark_backends import (
    CpuFeaturePredictor, CudaFeaturePredictor, partition,
)


def predict_registered_v5_frame(
    model: Any,
    *,
    baseline: np.ndarray,
    current: np.ndarray,
    previous: np.ndarray | None,
    h_baseline_to_current: Any,
    h_baseline_to_previous: Any | None = None,
    required_valid_mask: np.ndarray,
    baseline_origin_xy: tuple[int, int] = (0, 0),
    previous_origin_xy: tuple[int, int] = (0, 0),
    baseline_source_valid_mask: np.ndarray | None = None,
    previous_source_valid_mask: np.ndarray | None = None,
    backend: str = "cpu",
    predictor: CpuFeaturePredictor | CudaFeaturePredictor | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], Any]:
    """Return two-head optical maps and the *same* reference for regional physics.

    A reusable predictor retains model weights, but its baseline and previous
    feature caches are reset on every frame.  No rule is inferred from the
    homographies: their quality and source binding are checked by the caller.
    The required mask makes missing source pixels a hard error, never a zero
    signal at a weak plume edge.
    """
    if backend not in {"cpu", "cuda"}:
        raise ValueError("V5 registered inference backend must be cpu or cuda")
    if required_valid_mask is None:
        raise ValueError("V5 registered inference requires the current native domain")
    aligned_baseline, aligned_previous, alignment = align_temporal_inputs(
        baseline=baseline,
        current=current,
        previous=previous,
        h_baseline_to_current=h_baseline_to_current,
        h_baseline_to_previous=h_baseline_to_previous,
        required_valid_mask=required_valid_mask,
        baseline_origin_xy=baseline_origin_xy,
        previous_origin_xy=previous_origin_xy,
        baseline_source_valid_mask=baseline_source_valid_mask,
        previous_source_valid_mask=previous_source_valid_mask,
    )
    if predictor is None:
        cls = CpuFeaturePredictor if backend == "cpu" else CudaFeaturePredictor
        predictor = cls(model, aligned_baseline, input_encoding="v5-unit-float32")
    else:
        expected = CpuFeaturePredictor if backend == "cpu" else CudaFeaturePredictor
        # CpuFeaturePredictor subclasses CudaFeaturePredictor for the shared
        # feature formula; isinstance alone would accept CPU for a CUDA task.
        if type(predictor) is not expected:
            raise TypeError("Reusable V5 predictor has the wrong backend")
        predictor.set_baseline(aligned_baseline)
    heads = predictor.predict(current, aligned_previous)
    fields = partition(np.stack((heads["co2_presence_probability"],
                                 heads["gas_probability_conditional_on_co2"]), axis=-1))
    return fields, aligned_baseline, alignment, predictor


__all__ = ["predict_registered_v5_frame"]
