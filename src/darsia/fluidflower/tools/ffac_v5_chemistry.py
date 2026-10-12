"""Optional, source-bound FF_AC chemistry colour alignment experiment.

This is separate from light normalisation. It changes chroma only, on porous
sand, and preserves the camera-linear G channel exactly. The transform is
fitted on 2/6/24 h source pixels and spatial training tiles, never phase
labels. It is not approved for mass or concentration inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTRACT = "ffac-v5-chemistry-log-chroma-source-only-v1"
KNOTS = np.array([0., .25, .5, .75, 1.], dtype=np.float64)
FIT_TIMES = ("2", "6", "24")
HOLDOUT_TIMES = ("0.5", "4", "8", "12")


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return sha256(values.tobytes()).hexdigest()


def spatial_partition(run: str, shape_hw: tuple[int, int],
                      tile_px: int = 256, guard_px: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Frozen 1-in-5 spatial block holdout with a training guard."""
    if tile_px != 256 or guard_px != 32 or not run.startswith("ac"):
        raise ValueError("Unknown chemistry spatial partition")
    h, w = shape_hw
    holdout = np.zeros((h, w), bool)
    train = np.ones((h, w), bool)
    for iy, y0 in enumerate(range(0, h, tile_px)):
        y1 = min(y0 + tile_px, h)
        for ix, x0 in enumerate(range(0, w, tile_px)):
            x1 = min(x0 + tile_px, w)
            key = f"{run}:{ix}:{iy}".encode("ascii")
            if int.from_bytes(sha256(key).digest()[:8], "big") % 5 == 0:
                holdout[y0:y1, x0:x1] = True
                train[max(0, y0-guard_px):min(h, y1+guard_px),
                      max(0, x0-guard_px):min(w, x1+guard_px)] = False
    if not holdout.any() or not train.any():
        raise ValueError("Chemistry spatial partition has no fit or holdout pixels")
    return train, holdout


def _chroma(image: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return np.column_stack((image[0, yy, xx] - image[1, yy, xx],
                            image[2, yy, xx] - image[1, yy, xx])).astype(np.float64)


def _sample_coordinates(mask: np.ndarray, run: str) -> tuple[np.ndarray, np.ndarray]:
    offset = sha256(run.encode("ascii")).digest()
    py, px = offset[0] % 8, offset[1] % 8
    yy, xx = np.mgrid[py:mask.shape[0]:8, px:mask.shape[1]:8]
    valid = mask[yy, xx]
    return yy[valid], xx[valid]


def _require_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    result = np.asarray(image)
    if result.shape != (3, *shape_hw) or result.dtype != np.float32:
        raise ValueError("Chemistry input must be full-native float32 3×H×W log camera-linear RAW")
    if not np.all(np.isfinite(result)):
        raise ValueError("Chemistry input contains nonfinite RAW signal")
    return result


@dataclass(frozen=True)
class ChemistryCurve:
    run: str
    source_contract: dict[str, Any]
    residual_mask_sha256: str
    facies_sha256: str
    fit_mask_sha256: str
    blue_by_facies: np.ndarray  # 256×2, NaN for absent facies
    endpoint: np.ndarray       # source blue-to-yellow displacement, 2
    delta_knots: np.ndarray    # 5×2, chroma displacement from blue
    sample_counts: np.ndarray  # 5, after spatial/frame selection

    def __post_init__(self) -> None:
        if (self.blue_by_facies.shape != (256, 2)
                or self.endpoint.shape != (2,) or self.delta_knots.shape != (5, 2)
                or self.sample_counts.shape != (5,)):
            raise ValueError("Chemistry curve has an unknown shape")

    def identity(self) -> dict[str, Any]:
        return dict(contract=CONTRACT, run=self.run,
                    source_contract=self.source_contract,
                    residual_mask_sha256=self.residual_mask_sha256,
                    facies_sha256=self.facies_sha256,
                    fit_mask_sha256=self.fit_mask_sha256,
                    blue_by_facies_sha256=array_sha256(self.blue_by_facies),
                    endpoint_sha256=array_sha256(self.endpoint),
                    delta_knots_sha256=array_sha256(self.delta_knots),
                    sample_counts=self.sample_counts.tolist())


def fit_curve(*, run: str, baseline: np.ndarray,
              frames: Mapping[str, np.ndarray], facies: np.ndarray,
              baseline_residual_mask: np.ndarray,
              source_contract: Mapping[str, Any]) -> ChemistryCurve:
    """Fit a source-only 5-knot dye curve on frozen fit frames/tiles.

    The caller must verify all source hashes in ``source_contract`` against
    immutable manifests. This function rejects absent manifest/B/frame SHAs.
    """
    if set(frames) != set(FIT_TIMES):
        raise ValueError("Chemistry fit must use exactly 2/6/24 h")
    contract = dict(source_contract)
    if (contract.get("run") != run or not contract.get("manifest_sha256")
            or not contract.get("baseline_arw_sha256")
            or set(contract.get("fit_frame_arw_sha256", {})) != set(FIT_TIMES)
            or not contract.get("photometry_processor_sha256")):
        raise ValueError("Chemistry fit lacks source-bound manifest/B/frame/photometry SHA")
    f = np.asarray(facies)
    if f.ndim != 2 or f.dtype.kind not in "ui" or f.max() > 255:
        raise ValueError("Native source facies must be unsigned integer IDs <=255")
    shape = f.shape
    baseline = _require_image(baseline, shape)
    frames = {key: _require_image(value, shape) for key, value in frames.items()}
    residual = np.asarray(baseline_residual_mask, bool)
    if residual.shape != shape:
        raise ValueError("Baseline residual dye exclusion has another geometry")
    train, _ = spatial_partition(run, shape)
    train &= (f > 0) & ~residual
    train[:64, :] = train[-64:, :] = False
    train[:, :64] = train[:, -64:] = False
    yy, xx = _sample_coordinates(train, run)
    if len(yy) < 10_000:
        raise ValueError("Too few source-only native sand samples for chemistry fit")
    blue_samples = _chroma(baseline, yy, xx)
    facies_samples = f[yy, xx]
    blue = np.full((256, 2), np.nan, np.float64)
    for fid in np.unique(facies_samples):
        group = facies_samples == fid
        if group.sum() < 200:
            raise ValueError(f"Too few baseline samples for facies {fid}")
        blue[int(fid)] = np.median(blue_samples[group], axis=0)
    delta24 = _chroma(frames["24"], yy, xx) - blue_samples
    if not np.all(np.isfinite(delta24)):
        raise ValueError("Nonfinite 24-hour chemistry source samples")
    yellow_limit = np.quantile(delta24[:, 1], .05)
    endpoint = np.median(delta24[delta24[:, 1] <= yellow_limit], axis=0)
    if endpoint[1] >= -.03 or np.linalg.norm(endpoint) <= .03:
        raise ValueError("24-hour fit blocks do not establish a blue-to-yellow endpoint")
    norm2 = float(endpoint @ endpoint)
    all_delta = np.concatenate([_chroma(frames[key], yy, xx) - blue_samples
                                for key in FIT_TIMES], axis=0)
    progress = np.clip((all_delta @ endpoint) / norm2, 0, 1)
    knots = np.empty((5, 2), np.float64)
    counts = np.zeros(5, np.int64)
    knots[0] = 0
    knots[-1] = endpoint
    for i, k in enumerate(KNOTS):
        region = np.abs(progress - k) <= .125
        counts[i] = int(region.sum())
        if counts[i] < 200:
            raise ValueError(f"No source-only support for chemistry progress knot {k}")
        if 0 < i < 4:
            knots[i] = np.median(all_delta[region], axis=0)
    projected = knots @ endpoint / norm2
    if np.any(np.diff(projected) <= 0):
        raise ValueError("Source dye curve is not monotone in blue-to-yellow progress")
    return ChemistryCurve(run=run, source_contract=contract,
        residual_mask_sha256=array_sha256(residual),
        facies_sha256=array_sha256(f), fit_mask_sha256=array_sha256(train),
        blue_by_facies=blue, endpoint=endpoint, delta_knots=knots,
        sample_counts=counts)


def _check_pair(source: ChemistryCurve, reference: ChemistryCurve,
                facies: np.ndarray, residual_mask: np.ndarray) -> np.ndarray:
    if reference.run != "ac28" or source.run == reference.run:
        raise ValueError("Chemistry transform needs a deviating run and AC28 reference")
    if array_sha256(facies) != source.facies_sha256:
        raise ValueError("Chemistry transform facies differ from fitted source")
    if array_sha256(np.asarray(residual_mask, bool)) != source.residual_mask_sha256:
        raise ValueError("Chemistry residual exclusion differs from fitted source")
    if source.facies_sha256 != reference.facies_sha256:
        raise ValueError("Chemistry source and AC28 use a different native facies map")
    source_segments = np.linalg.norm(np.diff(source.delta_knots, axis=0), axis=1)
    ref_segments = np.linalg.norm(np.diff(reference.delta_knots, axis=0), axis=1)
    ratio = ref_segments / source_segments
    if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0) or np.any(ratio > 2):
        raise ValueError("Chemistry curve would amplify weak colour/noise by >2×")
    for fid in np.unique(facies):
        if fid > 0 and (not np.all(np.isfinite(source.blue_by_facies[fid]))
                        or not np.all(np.isfinite(reference.blue_by_facies[fid]))):
            raise ValueError(f"No AC28/source blue anchor for facies {fid}")
    return ratio


def apply_curve(*, source: ChemistryCurve, reference: ChemistryCurve,
                baseline: np.ndarray, frame: np.ndarray,
                facies: np.ndarray, baseline_residual_mask: np.ndarray,
                source_contract: Mapping[str, Any],
                stripe_rows: int = 128) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only chroma alignment, preserving original G and residual colour."""
    facies = np.asarray(facies)
    residual = np.asarray(baseline_residual_mask, bool)
    shape = facies.shape
    baseline, frame = _require_image(baseline, shape), _require_image(frame, shape)
    if residual.shape != shape or stripe_rows < 1:
        raise ValueError("Chemistry mask/stripe geometry invalid")
    if dict(source_contract) != source.source_contract:
        raise ValueError("Chemistry transform is bound to another source revision/B/photometry")
    slope = _check_pair(source, reference, facies, residual)
    result = frame.copy()
    denominator = float(source.endpoint @ source.endpoint)
    for y0 in range(0, shape[0], stripe_rows):
        y1 = min(y0 + stripe_rows, shape[0])
        eligible = (facies[y0:y1] > 0) & ~residual[y0:y1]
        if not eligible.any():
            continue
        yy, xx = np.nonzero(eligible)
        yy += y0
        fid = facies[yy, xx].astype(np.intp)
        current = _chroma(frame, yy, xx)
        blue_pixel = _chroma(baseline, yy, xx)
        delta = current - blue_pixel
        p = np.clip((delta @ source.endpoint) / denominator, 0, 1)
        src_curve = np.column_stack([np.interp(p, KNOTS, source.delta_knots[:, i]) for i in (0, 1)])
        ref_curve = np.column_stack([np.interp(p, KNOTS, reference.delta_knots[:, i]) for i in (0, 1)])
        mapped = current + (reference.blue_by_facies[fid] - source.blue_by_facies[fid]) + ref_curve - src_curve
        result[0, yy, xx] = frame[1, yy, xx] + mapped[:, 0]
        result[2, yy, xx] = frame[1, yy, xx] + mapped[:, 1]
    if not np.array_equal(result[1], frame[1]):
        raise AssertionError("Chemistry map changed camera-linear G")
    return result, dict(contract=CONTRACT, source_curve=source.identity(),
        reference_curve=reference.identity(), max_piecewise_noise_amplification=float(np.max(slope)),
        camera_linear_G_bit_identical=True,
        phase_or_mass_acceptance=False)


def heldout_source_metrics(*, source: ChemistryCurve, reference: ChemistryCurve,
                           baseline: np.ndarray, frame: np.ndarray,
                           facies: np.ndarray, baseline_residual_mask: np.ndarray,
                           source_contract: Mapping[str, Any],
                           time_h: float, frame_arw_sha256: str) -> dict[str, Any]:
    """Evaluate predeclared untouched blocks/frames; labels require a later gate."""
    if time_h in (2., 6., 24.) or not frame_arw_sha256:
        raise ValueError("Chemistry evaluation must use a source-bound held-out time")
    if time_h not in (.5, 4., 8., 12.) and time_h < 47.:
        raise ValueError("Chemistry time is not in the frozen holdout set")
    before = _require_image(frame, facies.shape)
    after, cert = apply_curve(source=source, reference=reference,
        baseline=baseline, frame=before, facies=facies,
        baseline_residual_mask=baseline_residual_mask,
        source_contract=source_contract)
    _, holdout = spatial_partition(source.run, facies.shape)
    holdout &= (facies > 0) & ~baseline_residual_mask
    yy, xx = _sample_coordinates(holdout, source.run)
    if len(yy) < 1000:
        raise ValueError("Too few independent chemistry holdout pixels")
    old_chroma, new_chroma = _chroma(before, yy, xx), _chroma(after, yy, xx)
    old_blue = _chroma(baseline, yy, xx)
    p = np.clip(((old_chroma - old_blue) @ source.endpoint) /
                float(source.endpoint @ source.endpoint), 0, 1)
    fid = facies[yy, xx].astype(np.intp)
    local_residual = old_blue - source.blue_by_facies[fid]
    expected_rg = (reference.blue_by_facies[fid, 0] + local_residual[:, 0]
                   + np.interp(p, KNOTS, reference.delta_knots[:, 0]))
    before_offset = float(np.median(np.abs(old_chroma[:, 0] - expected_rg)))
    after_offset = float(np.median(np.abs(new_chroma[:, 0] - expected_rg)))
    # This offset is descriptive, not a validation gate: the transform itself
    # adds the fitted AC28 curve, so a small distance to that same curve is
    # partially built in. Independent held-out AC28 colour-state evidence and
    # native phase-positive/negative QA must decide whether it helps.
    return dict(contract=CONTRACT, run=source.run, time_h=time_h,
        frame_arw_sha256=frame_arw_sha256, holdout_pixels=int(len(yy)),
        before_rg_median_offset=before_offset, after_rg_median_offset=after_offset,
        curve_offset_diagnostic_only=True,
        same_curve_was_used_in_transform=True,
        camera_linear_G_bit_identical=cert["camera_linear_G_bit_identical"],
        empirical_AC28_heldout_state_gate="required-not-evaluated",
        weak_aqueous_positive_gate="required-not-evaluated",
        free_gas_positive_gate="required-not-evaluated",
        negative_region_gate="required-not-evaluated",
        training_or_mass_accepted=False)
