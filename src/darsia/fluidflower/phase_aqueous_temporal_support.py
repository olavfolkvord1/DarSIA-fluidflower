"""Opt-in, bounded temporal evidence for regional aqueous CO2 decisions.

This is a cautious phase-map proposal, not a transport or concentration model.
Only independent *raw optical* evidence can enter the previous-frame memory;
accepted final maps never teach this sequence their own predictions. Weak,
baseline-like aqueous support may lose a little score when it appears far from
both that evidence and registered injection sources. Strong current evidence,
free gas, and uncertain frame sequences are retained. No reachability or hard
plume-connectivity condition is imposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .phase_rendering import two_phase_masks
from .project import sha256_file


VERSION = "aqueous-temporal-source-support-v1"
PROFILE_KEY = "aqueous_temporal_support"


@dataclass(frozen=True)
class Settings:
    format: str = VERSION
    working_spacing_m: float = .001
    weak_optical_probability: float = .75
    independent_seed_probability: float = .65
    weak_chromatic_change: float = .006
    strong_chromatic_change: float = .030
    independent_seed_change: float = .018
    history_distance_mm: float = 12.
    source_distance_mm: float = 35.
    maximum_history_gap_h: float = 48.
    minimum_suspicion: float = .45
    maximum_score_penalty: float = .12
    maximum_withdrawn_fraction: float = .01

    def __post_init__(self):
        if self.format != VERSION:
            raise ValueError("Unknown aqueous temporal-support format")
        numeric = {key: float(value) for key, value in asdict(self).items() if key != "format"}
        if any(not np.isfinite(value) or value <= 0 for value in numeric.values()):
            raise ValueError("Temporal-support settings must be finite and positive")
        if not (0 < self.working_spacing_m <= .005
                and 0 < self.independent_seed_probability < self.weak_optical_probability < 1
                and 0 < self.weak_chromatic_change < self.independent_seed_change < self.strong_chromatic_change < 1
                and 0 < self.minimum_suspicion < 1
                and 0 < self.maximum_score_penalty <= .2
                and 0 < self.maximum_withdrawn_fraction <= .05
                and self.history_distance_mm <= 100
                and self.source_distance_mm <= 150
                and self.maximum_history_gap_h <= 168):
            raise ValueError("Aqueous temporal-support settings exceed conservative bounds")


def runtime_identity(inherited: str, specification: dict) -> str:
    settings = Settings(**specification)
    payload = dict(inherited=inherited, settings=asdict(settings),
                   code_sha256=sha256_file(Path(__file__)), opencv=cv2.__version__)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _registered_grid(context, shape, settings):
    domain = np.asarray(context.domain)
    x, y = np.asarray(context.x_m, float), np.asarray(context.y_m, float)
    if (domain.shape != shape or domain.dtype != np.bool_ or min(shape) < 2
            or x.shape != (shape[1],) or y.shape != (shape[0],)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Temporal support requires registered native geometry")
    dx, dy = np.diff(x), np.diff(y)
    if (not (np.all(dx > 0) or np.all(dx < 0))
            or not (np.all(dy > 0) or np.all(dy < 0))
            or not np.allclose(dx, dx[0], rtol=1e-4, atol=1e-9)
            or not np.allclose(dy, dy[0], rtol=1e-4, atol=1e-9)):
        raise ValueError("Temporal support requires uniform x/y coordinates")
    spacing = (abs(float(dy[0])), abs(float(dx[0])))
    reduced = tuple(max(2, round(n * step / settings.working_spacing_m))
                    for n, step in zip(shape, spacing))
    reduced_spacing = tuple(n * step / count for n, step, count in zip(shape, spacing, reduced))
    return domain, reduced, reduced_spacing, x, y


def _chroma_change(current, baseline, reduced):
    size = reduced[::-1]
    cur = cv2.resize(np.asarray(current, np.float32), size, interpolation=cv2.INTER_AREA)
    ref = cv2.resize(np.asarray(baseline, np.float32), size, interpolation=cv2.INTER_AREA)
    cur /= np.maximum(cur.sum(axis=2, keepdims=True), 1.)
    ref /= np.maximum(ref.sum(axis=2, keepdims=True), 1.)
    return np.linalg.norm(cur - ref, axis=2).astype(np.float32)


def _source_distance(context, reduced, x, y):
    # The same physical coordinates are used for every run; only the registered
    # injector metadata vary. There is no privileged image row or hard height.
    xs = np.linspace(float(x[0]), float(x[-1]), reduced[1], dtype=np.float32)
    ys = np.linspace(float(y[0]), float(y[-1]), reduced[0], dtype=np.float32)
    source_distance = np.full(reduced, np.inf, np.float32)
    stages = tuple(context.injection_stages)
    for stage in stages:
        position = np.asarray((stage["x_m"], stage["y_m"]), float)
        times = np.asarray((stage["start_h"], stage["end_h"]), float)
        if not np.isfinite(position).all() or not np.isfinite(times).all() or times[1] <= times[0]:
            raise ValueError("Temporal support requires valid registered injection stages")
        source_distance = np.minimum(source_distance,
            np.hypot(xs[None, :] - position[0], ys[:, None] - position[1]))
    # Without reliable source metadata we abstain, rather than call every
    # disconnected weak signal implausible.
    return source_distance


class AqueousTemporalSequence:
    """Bound a soft score penalty with independent optical and temporal cues."""

    def __init__(self, inherited, context, specification):
        self.inherited, self.context = inherited, context
        self.settings = Settings(**specification)
        self.domain, self.reduced, self.spacing, self.x, self.y = _registered_grid(
            context, np.asarray(context.domain).shape, self.settings)
        self.source_distance = _source_distance(context, self.reduced, self.x, self.y)
        stages = tuple(context.injection_stages)
        self.first_injection_h = min((float(stage["start_h"]) for stage in stages), default=np.inf)
        self.independent_history = np.zeros(self.reduced, bool)
        self.previous_time = None
        self.baseline = None
        self.last_diagnostics = {}

    def bind_reference(self, baseline):
        if self.baseline is not None:
            raise ValueError("Temporal-support reference is immutable")
        value = np.asarray(baseline)
        if value.shape != (*self.domain.shape, 3) or value.dtype != np.uint8:
            raise ValueError("Temporal support requires registered native uint8 RGB")
        self.inherited.bind_reference(baseline)
        self.baseline = np.array(value, copy=True)
        self.baseline.setflags(write=False)

    def apply_with_reference(self, predictions, elapsed_h, current):
        if self.baseline is None:
            raise ValueError("Bind the pre-injection reference before temporal support")
        time = float(elapsed_h)
        if not np.isfinite(time) or (self.previous_time is not None and time <= self.previous_time):
            raise ValueError("Temporal support requires strictly increasing observation times")
        current = np.asarray(current)
        raw = np.asarray(predictions["co2_presence_probability"], np.float32)
        if (current.shape != self.baseline.shape or current.dtype != np.uint8
                or raw.shape != self.domain.shape or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()):
            raise ValueError("Temporal support requires registered current RGB and optical scores")
        # Validate before changing the inherited sequence's chronological state.
        gap = None if self.previous_time is None else time - self.previous_time
        result, report = self.inherited.apply_with_reference(predictions, time, current)
        masks = two_phase_masks(result)
        if masks["aqueous"].shape != self.domain.shape:
            raise ValueError("Temporal support and inherited phase grid differ")
        for key in ("co2_presence_probability", "gas_probability_conditional_on_co2"):
            values = np.asarray(result[key])
            if (values.shape != self.domain.shape or not np.isfinite(values).all()
                    or (values < 0).any() or (values > 1).any()):
                raise ValueError("Inherited temporal-support fields must be native unit-interval scores")
        if gap is not None and gap > self.settings.maximum_history_gap_h:
            # A long unobserved interval cannot supply reliable first-arrival
            # information to the following frame either.
            self.independent_history[:] = False
        size = self.reduced[::-1]
        change = _chroma_change(current, self.baseline, self.reduced)
        raw_small = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        independent = (cv2.resize(self.domain.astype(np.float32), size,
                                  interpolation=cv2.INTER_AREA) >= .999)
        independent &= (raw_small >= self.settings.independent_seed_probability)
        independent &= (change >= self.settings.independent_seed_change)
        if time < self.first_injection_h or report.get("pre_injection_negative_control"):
            independent[:] = False
        history_available = bool(self.independent_history.any())
        prior_independent_cells = int(self.independent_history.sum())
        gap_usable = gap is not None and gap <= self.settings.maximum_history_gap_h
        changed = np.zeros(self.domain.shape, bool)
        candidate = np.zeros_like(changed)
        maximum_penalty = 0.
        updated = result
        abstention = "none"
        if not gap_usable and gap is not None:
            abstention = "first_frame_or_history_gap"
        elif not history_available:
            abstention = "no_independent_previous_observation"
        elif not np.isfinite(self.source_distance).all():
            abstention = "no_registered_source"
        else:
            distance = distance_transform_edt(~self.independent_history,
                                               sampling=self.spacing).astype(np.float32)
            optical_weak = np.clip((self.settings.weak_optical_probability - raw_small) /
                                   (self.settings.weak_optical_probability - .4), 0., 1.)
            change_weak = np.clip((self.settings.strong_chromatic_change - change) /
                                  (self.settings.strong_chromatic_change -
                                   self.settings.weak_chromatic_change), 0., 1.)
            history_weak = -np.expm1(-distance / (self.settings.history_distance_mm * .001))
            source_weak = -np.expm1(-self.source_distance /
                                    (self.settings.source_distance_mm * .001))
            suspicion = optical_weak * change_weak * history_weak * source_weak
            suspicion = cv2.GaussianBlur(suspicion.astype(np.float32), (0, 0),
                sigmaX=max(.5, .001 / self.spacing[1]),
                sigmaY=max(.5, .001 / self.spacing[0]))
            penalty = self.settings.maximum_score_penalty * np.clip(
                (suspicion - self.settings.minimum_suspicion) /
                (1. - self.settings.minimum_suspicion), 0., 1.)
            penalty = cv2.resize(penalty, self.domain.shape[::-1], interpolation=cv2.INTER_LINEAR)
            eligible = masks["aqueous"] & self.domain
            # Current independent optical evidence remains protected even if
            # the previous observation had no evidence at this position.
            protected = cv2.resize(independent.astype(np.uint8), self.domain.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            penalty[~eligible | protected] = 0.
            candidate = penalty > 0
            maximum_penalty = float(penalty.max())
            co2 = np.asarray(result["co2_presence_probability"], np.float32)
            new_co2 = np.maximum(co2 - penalty, 0.)
            changed = eligible & (co2 >= .5) & (new_co2 < .5)
            fraction = float(changed.sum()) / max(1, int(eligible.sum()))
            if fraction > self.settings.maximum_withdrawn_fraction:
                abstention = "withdrawn_area_cap"
                changed[:] = False
                maximum_penalty = 0.
            elif candidate.any():
                updated = dict(result)
                updated["co2_presence_probability"] = new_co2
                gas = np.asarray(result["gas_probability_conditional_on_co2"], np.float32)
                for key, field in (("aqueous_co2_probability", new_co2 * (1 - gas)),
                                   ("free_co2_gas_probability", new_co2 * gas),
                                   ("untouched_water_probability", 1 - new_co2)):
                    if key in result:
                        updated[key] = field.astype(np.float32)
        self.independent_history |= independent
        self.previous_time = time
        self.last_diagnostics = {**getattr(self.inherited, "last_diagnostics", {}),
                                 "aqueous_temporal_withdrawn": changed}
        audit = dict(format=VERSION, elapsed_h=time, previous_observation_gap_h=gap,
                     prior_independent_support_cells=prior_independent_cells,
                     current_independent_support_cells=int(independent.sum()),
                     candidate_score_penalty_pixels=int(candidate.sum()),
                     maximum_score_penalty_applied=maximum_penalty,
                     withdrawn_aqueous_pixels=int(changed.sum()),
                     gas_labels_unchanged=True, strong_current_evidence_protected=True,
                     final_map_never_seeds_history=True,
                     hard_connectivity_or_height_veto=False,
                     mass_or_concentration_constraints=False,
                     abstention=abstention)
        return updated, {**report, PROFILE_KEY: audit}
