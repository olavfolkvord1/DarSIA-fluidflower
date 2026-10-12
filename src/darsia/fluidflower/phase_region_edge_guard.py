"""Opt-in, physical-scale smoothing of the displayed aqueous CO2 edge.

The categorical CO2 mask is Gaussian-smoothed on the native registered grid.
Gas pixels are immutable, and an area-change budget makes the operation abstain
if smoothing would materially alter a frame. This is a display decision, not a
calibrated concentration or a new optical observation.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .phase_rendering import two_phase_masks
from .project import sha256_file


VERSION = "physical-aqueous-region-edge-v1"
PROFILE_KEY = "region_edge_guard"


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    smoothing_sigma_mm: float = 0.5
    maximum_changed_area_fraction: float = 0.005

    def __post_init__(self):
        if self.format != VERSION:
            raise ValueError("Invalid region edge guard format")
        if (not np.isfinite(self.smoothing_sigma_mm)
                or not 0 < self.smoothing_sigma_mm <= 2):
            raise ValueError("Region edge smoothing sigma must be within (0, 2] mm")
        if (not np.isfinite(self.maximum_changed_area_fraction)
                or not 0 < self.maximum_changed_area_fraction <= 0.02):
            raise ValueError("Region edge area fraction must be within (0, 0.02]")


def runtime_identity(inherited, specification):
    """Version the setting and implementation without changing inherited runs."""
    settings = Settings(**specification)
    payload = dict(inherited=inherited, settings=asdict(settings),
                   code_sha256=sha256_file(Path(__file__)), opencv=cv2.__version__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def smooth_labels(labels, domain, spacing_m, settings=Settings()):
    """Return smoothed labels and an area audit, preserving gas exactly.

    ``spacing_m`` is native row/column pixel spacing in metres. The Gaussian
    weights are normalized by the registered domain so its outer boundary does
    not artificially erode valid CO2. Both gains and losses count against the
    cap; if it is exceeded, the whole frame is left unchanged.
    """
    if not isinstance(settings, Settings):
        raise TypeError("Region edge guard requires validated Settings")
    labels = np.asarray(labels)
    domain = np.asarray(domain)
    spacing = np.asarray(spacing_m, dtype=float)
    if (labels.ndim != 2 or domain.shape != labels.shape
            or domain.dtype != np.bool_ or not np.isin(labels, (0, 1, 2)).all()):
        raise ValueError("Region edge guard requires same-grid categorical labels and boolean domain")
    if (spacing.shape != (2,) or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)):
        raise ValueError("Region edge guard requires positive registered y/x spacing")

    support = (labels > 0) & domain
    support_pixels = int(np.count_nonzero(support))
    report = dict(format=VERSION, smoothing_sigma_mm=settings.smoothing_sigma_mm,
                  maximum_changed_area_fraction=settings.maximum_changed_area_fraction,
                  pixel_area_mm2=float(np.prod(spacing) * 1e6),
                  source_co2_pixels=support_pixels, applied=False,
                  cap_exceeded=False, candidate_added_pixels=0,
                  candidate_removed_pixels=0, candidate_changed_area_fraction=0.0,
                  added_aqueous_pixels=0, removed_aqueous_pixels=0,
                  changed_area_fraction=0.0, gas_labels_unchanged=True,
                  domain_labels_unchanged=True, calibrated_probability=False)
    if support_pixels == 0:
        return labels.astype(np.uint8, copy=True), report

    sigma_y, sigma_x = (settings.smoothing_sigma_mm * 1e-3 / spacing).tolist()
    # OpenCV's zero-padded Gaussian is normalized against the domain, making
    # the threshold independent of whether a plume touches the registered edge.
    numerator = cv2.GaussianBlur(support.astype(np.float32), (0, 0),
                                 sigmaX=sigma_x, sigmaY=sigma_y,
                                 borderType=cv2.BORDER_CONSTANT)
    weight = cv2.GaussianBlur(domain.astype(np.float32), (0, 0),
                              sigmaX=sigma_x, sigmaY=sigma_y,
                              borderType=cv2.BORDER_CONSTANT)
    smoothed = domain & (numerator >= 0.5 * weight)
    candidate = labels.astype(np.uint8, copy=True)
    editable = domain & (labels != 2)
    candidate[editable] = smoothed[editable].astype(np.uint8)
    added = (labels == 0) & (candidate == 1)
    removed = (labels == 1) & (candidate == 0)
    added_count, removed_count = int(added.sum()), int(removed.sum())
    fraction = (added_count + removed_count) / support_pixels
    report.update(candidate_added_pixels=added_count,
                  candidate_removed_pixels=removed_count,
                  candidate_changed_area_fraction=fraction)
    if fraction > settings.maximum_changed_area_fraction:
        report["cap_exceeded"] = True
        return labels.astype(np.uint8, copy=True), report
    report.update(applied=True, added_aqueous_pixels=added_count,
                  removed_aqueous_pixels=removed_count,
                  changed_area_fraction=fraction)
    return candidate, report


def _spacing(context, shape):
    domain = np.asarray(context.domain)
    x = np.asarray(context.x_m, dtype=float)
    y = np.asarray(context.y_m, dtype=float)
    if (domain.shape != shape or domain.dtype != np.bool_ or min(shape) < 2
            or x.shape != (shape[1],) or y.shape != (shape[0],)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Region edge guard requires a registered native grid")
    dx, dy = np.diff(x), np.diff(y)
    if (not (np.all(dx > 0) or np.all(dx < 0))
            or not (np.all(dy > 0) or np.all(dy < 0))
            or not np.allclose(dx, dx[0], rtol=1e-4, atol=1e-9)
            or not np.allclose(dy, dy[0], rtol=1e-4, atol=1e-9)):
        raise ValueError("Region edge guard requires uniform registered x/y coordinates")
    return abs(float(dy[0])), abs(float(dx[0]))


class RegionEdgeSequence:
    """Postprocess an inherited phase sequence, without mutating its history."""

    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        self.inherited.bind_reference(baseline)

    def apply_with_reference(self, predictions, elapsed_h, current):
        result, report = self.inherited.apply_with_reference(predictions, elapsed_h, current)
        masks = two_phase_masks(result)
        labels = np.where(masks["gas"], 2, masks["aqueous"]).astype(np.uint8)
        smoothed, audit = smooth_labels(labels, self.context.domain,
                                        _spacing(self.context, labels.shape), self.settings)
        added = (labels == 0) & (smoothed == 1)
        removed = (labels == 1) & (smoothed == 0)
        changed = added | removed
        updated = dict(result)
        if changed.any():
            co2 = np.asarray(result["co2_presence_probability"], dtype=np.float32).copy()
            gas = np.asarray(result["gas_probability_conditional_on_co2"], dtype=np.float32).copy()
            # The threshold itself records a new categorical aqueous decision.
            # Removed edge pixels are set to background for a definite label.
            co2[added], co2[removed], gas[changed] = np.float32(.5), np.float32(0), np.float32(0)
            updated["co2_presence_probability"] = co2
            updated["gas_probability_conditional_on_co2"] = gas
            for key, values in (("aqueous_co2_probability", co2 * (1 - gas)),
                                ("free_co2_gas_probability", co2 * gas),
                                ("untouched_water_probability", 1 - co2)):
                if key in result:
                    field = np.asarray(result[key], dtype=np.float32).copy()
                    field[changed] = values[changed]
                    updated[key] = field
        self.last_diagnostics = {**getattr(self.inherited, "last_diagnostics", {}),
                                 "region_edge_added_aqueous": added,
                                 "region_edge_removed_aqueous": removed}
        return updated, {**report, PROFILE_KEY: audit}
