"""Causal, opt-in test of temporal support for small free-gas components.

This is a conservative classification hypothesis, not a transport model. It
only changes free gas to aqueous CO2. Previous gas is a *recent observation*,
not a permanent union of every place that was once gas. Long sampling gaps and
the injection period are left unchanged because temporal support is ambiguous.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .project import sha256_file
from .phase_rendering import two_phase_masks

VERSION = "recent-small-gas-support-pilot-v1"
STRUCTURE_WIDTH = 2048


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    maximum_component_area_mm2: float = 25.0
    history_radius_mm: float = 8.0
    maximum_gap_h: float = 2.0
    post_injection_grace_h: float = 0.25

    def __post_init__(self):
        if self.format != VERSION or any(
            not np.isfinite(value) or value <= 0
            for name, value in asdict(self).items() if name != "format"
        ):
            raise ValueError("Invalid temporal gas support settings")


@dataclass(frozen=True)
class BuoyancySettings:
    """Physical distances shared across rigs; missing end snapshots disable the rule."""

    end_snapshot_max_delay_h: float = 0.5
    post_injection_grace_h: float = 0.25
    below_source_tolerance_mm: float = 5.0
    end_footprint_tolerance_mm: float = 8.0
    facies_boundary_exemption_mm: float = 5.0
    maximum_gap_h: float = 4.0

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Invalid buoyancy settings")


def prepare_buoyancy_floor(x_m, y_m, facies, stages, end_snapshots):
    """Cache geometry and injection-end distances for a chronological run."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    facies = np.asarray(facies)
    if facies.shape != (len(y), len(x)) or len(end_snapshots) != len(stages):
        raise ValueError("Buoyancy geometry and stage snapshots differ")
    spacing_mm = (abs(y[1]-y[0])*1000, abs(x[1]-x[0])*1000)
    edge = np.zeros(facies.shape, bool)
    edge[1:, :] |= facies[1:, :] != facies[:-1, :]
    edge[:, 1:] |= facies[:, 1:] != facies[:, :-1]
    distances = []
    for snapshot in end_snapshots:
        if snapshot is None:
            distances.append(None)
            continue
        snapshot = np.asarray(snapshot, bool)
        if snapshot.shape != facies.shape:
            raise ValueError("Injection-end snapshot grid differs")
        supported = snapshot
        distances.append(distance_transform_edt(~supported, sampling=spacing_mm)
                         if supported.any() else None)
    return dict(shape=facies.shape,
                edge_distance=distance_transform_edt(~edge, sampling=spacing_mm),
                end_distances=distances)


def apply_buoyancy_floor(labels, *, elapsed_h, x_m, y_m, facies, stages,
                         end_snapshots, settings=BuoyancySettings(), prepared=None,
                         previous_h=None):
    """Reject only new below-source gas far from end gas and facies boundaries.

    End snapshots are observed labels at the first sufficiently close photo
    after each injection ends. An unobserved stage cannot impose a floor.
    """
    labels, facies = np.asarray(labels), np.asarray(facies)
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    if (labels.ndim != 2 or facies.shape != labels.shape or
        labels.shape != (len(y), len(x)) or not np.isin(labels, (0, 1, 2)).all() or
        len(end_snapshots) != len(stages)):
        raise ValueError("Buoyancy floor requires one registered grid and snapshot per stage")
    result = labels.astype(np.uint8, copy=True)
    audit = dict(reclassified_gas_pixels=0, stages_tested=0,
                 co2_support_unchanged=True)
    if previous_h is not None and elapsed_h - previous_h > settings.maximum_gap_h:
        return result, dict(audit, skipped="unresolved_time_gap")
    if elapsed_h < max(float(s["end_h"]) for s in stages) + settings.post_injection_grace_h:
        return result, dict(audit, skipped="active_injection_or_grace")
    support = prepared if prepared is not None else prepare_buoyancy_floor(
        x, y, facies, stages, end_snapshots)
    if support["shape"] != labels.shape or len(support["end_distances"]) != len(stages):
        raise ValueError("Cached buoyancy geometry differs")
    edge_distance = support["edge_distance"]
    # A pixel is implausible only if *every* injection fails to support it.
    # There is deliberately no Voronoi split or straight boundary between
    # injectors; ambiguous source attribution remains visible for review.
    rejected = (labels == 2) & (edge_distance > settings.facies_boundary_exemption_mm)
    for i, (stage, snapshot) in enumerate(zip(stages, end_snapshots)):
        end_distance = support["end_distances"][i]
        if snapshot is None or end_distance is None:
            return result, dict(audit, skipped="missing_injection_end_evidence")
        rejected &= (y[:, None] * 1000 < float(stage["y_m"])*1000
                     + settings.below_source_tolerance_mm)
        rejected &= end_distance > settings.end_footprint_tolerance_mm
        audit["stages_tested"] += 1
    result[rejected] = 1
    audit["reclassified_gas_pixels"] = int(rejected.sum())
    return result, dict(audit, skipped=None)


def runtime_identity(inherited, specification):
    return hashlib.sha256(json.dumps(dict(
        inherited=inherited, settings=asdict(Settings(**specification)),
        code_sha256=sha256_file(Path(__file__)), opencv=cv2.__version__),
        sort_keys=True).encode()).hexdigest()


def apply_labels(labels, previous, elapsed_h, previous_h, injection_end_h,
                 spacing_mm, settings=Settings()):
    """Return candidate labels and an audit for one chronologically ordered frame."""
    labels = np.asarray(labels)
    previous = np.asarray(previous)
    spacing = np.asarray(spacing_mm, float)
    if (labels.ndim != 2 or previous.shape != labels.shape or
        not np.isin(labels, (0, 1, 2)).all() or
        not np.isin(previous, (0, 1, 2)).all() or
        spacing.shape != (2,) or not np.isfinite(spacing).all() or
        (spacing <= 0).any()):
        raise ValueError("Temporal support requires aligned categorical labels and spacing")
    if not all(np.isfinite(v) for v in (elapsed_h, previous_h, injection_end_h)) or elapsed_h <= previous_h:
        raise ValueError("Temporal support requires strictly increasing finite times")
    gap_h = elapsed_h - previous_h
    audit = dict(format=VERSION, elapsed_h=float(elapsed_h), gap_h=float(gap_h),
                 removed_components=0, reclassified_gas_pixels=0,
                 co2_support_unchanged=True)
    if elapsed_h < injection_end_h + settings.post_injection_grace_h:
        return labels.copy(), dict(audit, skipped="injection_or_grace")
    if gap_h > settings.maximum_gap_h:
        return labels.copy(), dict(audit, skipped="unresolved_time_gap")
    prior_gas = previous == 2
    if not prior_gas.any():
        return labels.copy(), dict(audit, skipped="no_reliable_prior_gas")
    radius_yx = np.ceil(settings.history_radius_mm / spacing).astype(int)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        (2 * int(radius_yx[1]) + 1, 2 * int(radius_yx[0]) + 1))
    near_recent = cv2.dilate(prior_gas.astype(np.uint8), kernel).astype(bool)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (labels == 2).astype(np.uint8), connectivity=8)
    area_mm2 = stats[:, cv2.CC_STAT_AREA] * float(np.prod(spacing))
    touches_recent = np.bincount(components[near_recent], minlength=count) > 0
    remove_ids = ((area_mm2 <= settings.maximum_component_area_mm2)
                  & ~touches_recent)
    remove_ids[0] = False
    removed = remove_ids[components]
    result = labels.astype(np.uint8, copy=True)
    result[removed] = 1
    return result, dict(audit, removed_components=int(remove_ids.sum()),
                        reclassified_gas_pixels=int(removed.sum()),
                        skipped=None)


class TemporalGasSequence:
    """Opt-in final wrapper; old optical models retain their original runtime."""

    def __init__(self, inherited, context, specification):
        self.inherited = inherited
        self.context = context
        self.settings = Settings(**specification)
        self.end_h = max(float(stage["end_h"]) for stage in context.injection_stages)
        self.previous = None
        self.previous_h = None
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        self.inherited.bind_reference(baseline)

    def apply_with_reference(self, predictions, elapsed_h, current):
        result, report = self.inherited.apply_with_reference(predictions, elapsed_h, current)
        masks = two_phase_masks(result)
        labels = np.where(masks["gas"], 2, masks["aqueous"]).astype(np.uint8)
        if self.previous_h is not None and elapsed_h <= self.previous_h:
            raise ValueError("Temporal gas support requires ordered frames")
        width = min(STRUCTURE_WIDTH, labels.shape[1])
        shape = (max(1, round(labels.shape[0] * width / labels.shape[1])), width)
        small = cv2.resize(labels, shape[::-1], interpolation=cv2.INTER_NEAREST)
        if self.previous is None:
            candidate = small.copy()
            audit = dict(format=VERSION, skipped="first_frame", removed_components=0,
                         reclassified_gas_pixels=0, co2_support_unchanged=True)
        else:
            spacing_mm = (
                abs(float(self.context.y_m[1] - self.context.y_m[0])) * 1000 * labels.shape[0] / shape[0],
                abs(float(self.context.x_m[1] - self.context.x_m[0])) * 1000 * labels.shape[1] / shape[1],
            )
            candidate, audit = apply_labels(small, self.previous, float(elapsed_h),
                                            self.previous_h, self.end_h, spacing_mm, self.settings)
        removed = (small == 2) & (candidate == 1)
        changed = (labels == 2) & cv2.resize(removed.astype(np.uint8),
                                              labels.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        self.previous, self.previous_h = candidate, float(elapsed_h)
        gas = np.where(changed, 0., result["gas_probability_conditional_on_co2"]).astype(np.float32)
        updated = dict(result, gas_probability_conditional_on_co2=gas)
        co2 = result["co2_presence_probability"]
        for key, values in (("aqueous_co2_probability", co2 * (1-gas)),
                            ("free_co2_gas_probability", co2 * gas)):
            if key in result:
                updated[key] = np.where(changed, values, result[key]).astype(np.float32)
        self.last_diagnostics = {**self.inherited.last_diagnostics, "recent_gas_reclassified": changed}
        return updated, {**report, "temporal_gas_support": dict(audit,
            structure_width_px=width, native_reclassified_gas_pixels=int(changed.sum()))}
