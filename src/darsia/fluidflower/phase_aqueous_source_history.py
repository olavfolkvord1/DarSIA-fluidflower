"""Opt-in source-anchored temporal evidence for aqueous phase-map proposals.

Only registered injections and independent *raw optical* evidence create the
history. A new raw signal can extend that history within a generous elapsed-
time envelope; displayed phase maps never seed it. Unsupported weak aqueous
scores may be attenuated slightly, with a whole-frame area safety cap. There
is no hard connectedness, height, or first-arrival classification rule.

This is a conservative diagnostic hypothesis, not a transport simulation,
calibrated concentration model, or substitute for source-aligned reviews.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import phase_aqueous_temporal_support as prior_version
from .phase_rendering import two_phase_masks
from .project import sha256_file


VERSION = "aqueous-source-anchored-temporal-v2"
PROFILE_KEY = "aqueous_temporal_support"


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    working_spacing_m: float = .001
    independent_seed_probability: float = .65
    independent_seed_change: float = .018
    source_anchor_radius_mm: float = 90.
    growth_slack_mm: float = 30.
    growth_mm_per_sqrt_h: float = 90.
    history_retention_h: float = 36.
    maximum_observation_gap_h: float = 48.
    history_distance_mm: float = 25.
    source_distance_mm: float = 80.
    current_optical_protection_probability: float = .80
    current_chromatic_protection: float = .060
    optical_weakness_weight: float = .65
    minimum_suspicion: float = .52
    maximum_score_penalty: float = .10
    maximum_withdrawn_fraction: float = .01

    def __post_init__(self):
        if self.format != VERSION:
            raise ValueError("Unknown source-anchored aqueous history format")
        numeric = [float(value) for key, value in asdict(self).items() if key != "format"]
        if any(not np.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("Source-anchored history settings must be finite and positive")
        if not (
            self.working_spacing_m <= .005
            and .5 <= self.independent_seed_probability < self.current_optical_protection_probability < 1.
            and 0 < self.independent_seed_change < self.current_chromatic_protection < .2
            and self.source_anchor_radius_mm <= 200.
            and self.growth_slack_mm <= 100.
            and self.growth_mm_per_sqrt_h <= 300.
            and self.history_retention_h <= 72.
            and self.maximum_observation_gap_h <= 168.
            and self.history_distance_mm <= 100.
            and self.source_distance_mm <= 200.
            and 0 < self.optical_weakness_weight < 1.
            and 0 < self.minimum_suspicion < 1.
            and self.maximum_score_penalty <= .2
            and self.maximum_withdrawn_fraction <= .05
        ):
            raise ValueError("Source-anchored history settings exceed conservative bounds")


def runtime_identity(inherited: str, specification: dict) -> str:
    settings = Settings(**specification)
    payload = dict(
        inherited=inherited, settings=asdict(settings),
        code_sha256=sha256_file(Path(__file__)),
        shared_geometry_code_sha256=sha256_file(Path(prior_version.__file__)),
        opencv=cv2.__version__, numpy=np.__version__,
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class AqueousSourceHistorySequence:
    """Per-run soft aqueous decision, never trained from its own final masks."""

    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        (self.domain, self.reduced, self.spacing, self.x, self.y) = prior_version._registered_grid(
            context, np.asarray(context.domain).shape, self.settings)
        self.stages = tuple(context.injection_stages)
        self.first_injection_h = min((float(stage["start_h"]) for stage in self.stages), default=np.inf)
        xs = np.linspace(float(self.x[0]), float(self.x[-1]), self.reduced[1], dtype=np.float32)
        ys = np.linspace(float(self.y[0]), float(self.y[-1]), self.reduced[0], dtype=np.float32)
        self.stage_distances = []
        for stage in self.stages:
            position = np.asarray((stage["x_m"], stage["y_m"]), float)
            start, end = float(stage["start_h"]), float(stage["end_h"])
            if not np.isfinite(position).all() or not np.isfinite((start, end)).all() or end <= start:
                raise ValueError("Source-anchored history requires valid registered injection stages")
            self.stage_distances.append(np.hypot(xs[None, :] - position[0],
                                                  ys[:, None] - position[1]).astype(np.float32))
        self.last_trusted_raw_observation_h = np.full(self.reduced, -np.inf, np.float32)
        self.previous_time = None
        self.baseline = None
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        if self.baseline is not None:
            raise ValueError("Source-anchored history reference is immutable")
        value = np.asarray(baseline)
        if value.shape != (*self.domain.shape, 3) or value.dtype != np.uint8:
            raise ValueError("Source-anchored history requires registered native uint8 RGB")
        self.inherited.bind_reference(baseline)
        self.baseline = np.array(value, copy=True)
        self.baseline.setflags(write=False)

    def _source_distance(self, time: float) -> np.ndarray | None:
        active = [distance for stage, distance in zip(self.stages, self.stage_distances)
                  if time >= float(stage["start_h"])]
        if not active:
            return None
        result = active[0].copy()
        for distance in active[1:]:
            np.minimum(result, distance, out=result)
        return result

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.baseline is None:
            raise ValueError("Bind a native pre-injection reference before source history")
        time = float(elapsed_h)
        if not np.isfinite(time) or (self.previous_time is not None and time <= self.previous_time):
            raise ValueError("Source-anchored history requires strictly increasing times")
        raw = np.asarray(predictions["co2_presence_probability"], np.float32)
        current = np.asarray(current)
        if (current.dtype != np.uint8 or current.shape != self.baseline.shape
                or raw.shape != self.domain.shape or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()):
            raise ValueError("Source-anchored history requires registered RGB and raw optical scores")
        gap = None if self.previous_time is None else time - self.previous_time
        result, report = self.inherited.apply_with_reference(predictions, time, current)
        masks = two_phase_masks(result)
        if masks["aqueous"].shape != self.domain.shape:
            raise ValueError("Inherited source-history phase grid differs")
        co2 = np.asarray(result["co2_presence_probability"], np.float32)
        gas = np.asarray(result["gas_probability_conditional_on_co2"], np.float32)
        if (not np.isfinite(co2).all() or not np.isfinite(gas).all()
                or (co2 < 0).any() or (co2 > 1).any()
                or (gas < 0).any() or (gas > 1).any()):
            raise ValueError("Inherited source-history scores must be finite unit-interval values")

        if gap is not None and gap > self.settings.maximum_observation_gap_h:
            self.last_trusted_raw_observation_h[:] = -np.inf
        active_history = self.last_trusted_raw_observation_h >= time - self.settings.history_retention_h
        prior_trusted = int(active_history.sum())
        source_distance = self._source_distance(time)
        change = prior_version._chroma_change(current, self.baseline, self.reduced)
        size = self.reduced[::-1]
        raw_small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        domain_small = cv2.resize(self.domain.astype(np.float32), size,
                                  interpolation=cv2.INTER_AREA) >= .999
        independent = domain_small & (raw_small >= self.settings.independent_seed_probability)
        independent &= change >= self.settings.independent_seed_change
        if time < self.first_injection_h or report.get("pre_injection_negative_control"):
            independent[:] = False

        trusted_current = np.zeros(self.reduced, bool)
        prior_distance = None
        if source_distance is not None:
            trusted_current |= independent & (source_distance <= self.settings.source_anchor_radius_mm * .001)
            if prior_trusted:
                prior_distance = distance_transform_edt(~active_history, sampling=self.spacing).astype(np.float32)
                allowance = (self.settings.growth_slack_mm
                             + self.settings.growth_mm_per_sqrt_h * np.sqrt(max(float(gap or 0.), 0.))) * .001
                trusted_current |= independent & (prior_distance <= allowance)

        abstention = "none"
        if gap is not None and gap > self.settings.maximum_observation_gap_h:
            abstention = "unobserved_history_gap"
        elif source_distance is None:
            abstention = "no_active_registered_source"
        elif not prior_trusted:
            abstention = "no_prior_source_anchored_raw_support"

        candidate = np.zeros(self.domain.shape, bool)
        would_withdraw = np.zeros_like(candidate)
        withdrawn = np.zeros_like(candidate)
        maximum_penalty = 0.
        updated = result
        if abstention == "none":
            history_far = -np.expm1(-prior_distance / (self.settings.history_distance_mm * .001))
            source_far = -np.expm1(-source_distance / (self.settings.source_distance_mm * .001))
            weak_p = np.clip(1. - raw_small / self.settings.current_optical_protection_probability, 0., 1.)
            weak_change = np.clip(1. - change / self.settings.current_chromatic_protection, 0., 1.)
            weak_current = (self.settings.optical_weakness_weight * weak_p
                            + (1. - self.settings.optical_weakness_weight) * weak_change)
            suspicion = history_far * source_far * weak_current
            suspicion = cv2.GaussianBlur(suspicion.astype(np.float32), (0, 0),
                sigmaX=max(.5, .001 / self.spacing[1]),
                sigmaY=max(.5, .001 / self.spacing[0]))
            penalty = self.settings.maximum_score_penalty * np.clip(
                (suspicion - self.settings.minimum_suspicion) /
                (1. - self.settings.minimum_suspicion), 0., 1.)
            penalty = cv2.resize(penalty, self.domain.shape[::-1], interpolation=cv2.INTER_LINEAR)
            protected = (raw_small >= self.settings.current_optical_protection_probability)
            protected |= change >= self.settings.current_chromatic_protection
            # Score against the previous trusted observation. A detached raw
            # patch cannot certify itself merely by entering today's history.
            protected = cv2.resize(protected.astype(np.uint8), self.domain.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            eligible = masks["aqueous"] & self.domain
            penalty[~eligible | protected] = 0.
            candidate = penalty > 0
            would_withdraw = eligible & (co2 >= .5) & (co2 - penalty < .5)
            fraction = float(would_withdraw.sum()) / max(1, int(eligible.sum()))
            if fraction > self.settings.maximum_withdrawn_fraction:
                abstention = "withdrawn_area_cap"
            elif candidate.any():
                new_co2 = np.maximum(co2 - penalty, 0.)
                updated = dict(result)
                updated["co2_presence_probability"] = new_co2
                for key, field in (("aqueous_co2_probability", new_co2 * (1 - gas)),
                                   ("free_co2_gas_probability", new_co2 * gas),
                                   ("untouched_water_probability", 1 - new_co2)):
                    if key in result:
                        updated[key] = field.astype(np.float32)
                withdrawn = would_withdraw
                maximum_penalty = float(penalty.max())

        # An area-cap event is a warning about uncertain evidence, so do not
        # propagate raw support from that frame beyond direct source anchors.
        if abstention == "withdrawn_area_cap" and source_distance is not None:
            trusted_current &= source_distance <= self.settings.source_anchor_radius_mm * .001
        self.last_trusted_raw_observation_h[trusted_current] = time
        self.previous_time = time
        self.last_diagnostics = {**getattr(self.inherited, "last_diagnostics", {}),
                                 "aqueous_source_history_withdrawn": withdrawn}
        audit = dict(format=VERSION, elapsed_h=time, previous_observation_gap_h=gap,
                     prior_trusted_raw_cells=prior_trusted,
                     current_independent_raw_cells=int(independent.sum()),
                     current_source_anchored_raw_cells=int(trusted_current.sum()),
                     candidate_penalty_pixels=int(candidate.sum()),
                     would_withdraw_aqueous_pixels=int(would_withdraw.sum()),
                     withdrawn_aqueous_pixels=int(withdrawn.sum()),
                     maximum_score_penalty_applied=maximum_penalty,
                     gas_labels_unchanged=True, final_map_never_seeds_history=True,
                     hard_connectivity_or_height_veto=False,
                     mass_or_concentration_constraints=False,
                     abstention=abstention)
        return updated, {**report, PROFILE_KEY: audit}
