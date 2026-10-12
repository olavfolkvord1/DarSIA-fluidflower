"""Explicit common ColorChecker target for an FF_AC source-correction arm.

This operates on the registered rig *before* native review photos are read.
It must not be applied to an already corrected review JPEG.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from darsia.fluidflower.image_correction import _checker_orientation_score


FORMAT = "ffac-common-colorchecker-target-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_target(value: Mapping[str, Any]) -> np.ndarray:
    if value.get("format") != FORMAT or value.get("dataset") != "ffac":
        raise ValueError("Expected a frozen FF_AC common ColorChecker target")
    swatches = np.asarray(value.get("swatches_rgb"), dtype=np.float32)
    if swatches.shape != (4, 6, 3):
        raise ValueError("Common target must hold 24 RGB patches in a 4x6x3 array")
    if not np.all(np.isfinite(swatches)) or np.any((swatches < 0) | (swatches > 1)):
        raise ValueError("Common target patches must be finite sRGB values in [0, 1]")
    if _checker_orientation_score(swatches) > 0.20:
        raise ValueError("Common target ColorChecker orientation or neutrals are invalid")
    source = value.get("source")
    if not isinstance(source, Mapping) or not source.get("kind") or not source.get("sha256"):
        raise ValueError("Common target must identify its chart-only source by SHA-256")
    return swatches


def load_target(path: Path, *, expected_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an immutable target only when its job-bound file digest matches."""

    path = Path(path).resolve()
    actual_sha256 = _sha(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Common ColorChecker target file differs from frozen job")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    swatches = validate_target(value)
    return swatches, {
        "format": FORMAT,
        "path": str(path),
        "sha256": actual_sha256,
        "source": dict(value["source"]),
    }


def install_common_target(
    rig: Any,
    swatches_rgb: np.ndarray,
    *,
    target_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only the per-run chart *target*, then rebuild its own baseline.

    Registration, spatial light correction, ROI, algorithm settings and all raw
    image paths stay on the same rig. Native baseline and subsequent images now
    pass through the same globally targeted correction.
    """

    import darsia

    swatches = np.asarray(swatches_rgb, dtype=np.float32)
    if swatches.shape != (4, 6, 3) or not np.all(np.isfinite(swatches)):
        raise ValueError("Expected 24 finite common-target RGB patches")
    corrections = getattr(rig, "color_corrections", None) or []
    compatible = [
        correction for correction in corrections
        if all(hasattr(correction, name) for name in ("colorchecker", "roi", "correct_array"))
    ]
    if len(compatible) != 1 or len(corrections) != 1:
        raise ValueError("Expected one configured FF_AC ColorChecker correction")
    baseline = getattr(rig, "shape_corrected_baseline", None)
    if baseline is None:
        raise ValueError("FF_AC rig has no shape-corrected raw baseline")
    correction = compatible[0]
    if not getattr(correction, "active", False):
        raise ValueError("FF_AC ColorChecker correction is inactive")
    correction.colorchecker = darsia.CustomColorChecker(reference_colors=swatches)
    corrected_baseline = baseline.copy()
    for item in corrections:
        corrected_baseline = item(corrected_baseline)
    rig.baseline = corrected_baseline
    provenance = dict(target_receipt)
    provenance.update(
        arm="global-common-colorchecker-raw-rig",
        reference_target_applied_before_native_frame_read=True,
        baseline_rebuilt_with_common_target=True,
        chart_roi_unchanged=True,
        geometric_and_spatial_light_corrections_unchanged=True,
    )
    rig._phase_common_colorchecker_target = provenance
    return provenance
