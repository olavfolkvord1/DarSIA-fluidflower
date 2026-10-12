"""Causal, per-run regional features for a separately trained CO2 head.

This module computes *features only*. It makes no phase decision and never
uses a previously predicted final phase map as historical evidence. A caller
must consume ``peek`` before ``commit`` updates the state with that frame's
frozen raw optical score. Missing pre-injection context causes abstention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


VERSION = "causal-temporal-regional-features-v1"
FEATURE_NAMES = (
    "preinjection_raw_presence",
    "prior_raw_presence_ema_1h",
    "prior_raw_presence_ema_8h",
    "prior_chromatic_change_ema_8h",
    "prior_raw_presence_neighborhood_5mm",
    "current_minus_prior_raw_presence",
    "nearest_started_source_distance_m",
    "nearest_started_source_vertical_offset_m",
    "log1p_hours_since_nearest_source_start",
    "log1p_hours_since_previous_observation",
)


@dataclass(frozen=True)
class Settings:
    fast_ema_h: float = 1.0
    slow_ema_h: float = 8.0
    neighborhood_sigma_m: float = .005

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in asdict(self).values())
                or self.fast_ema_h <= 0 or self.slow_ema_h <= self.fast_ema_h
                or not 0 < self.neighborhood_sigma_m <= .02):
            raise ValueError("Invalid causal temporal feature scales")


class CausalTemporalFeatureState:
    """Build same-source, chronological evidence before a fitted head scores it.

    The explicit manifest SHA identifies the source revision whose entire
    native prefix must be replayed. It is an identity assertion, not a claim
    that this class itself reads or hashes the images; the caller verifies the
    actual manifest and each native frame before calling ``peek``.
    """

    def __init__(self, context, *, run: str, native_manifest_sha256: str,
                 settings: Settings = Settings()):
        if not run or not isinstance(run, str):
            raise ValueError("A unique run identifier is required")
        if (not isinstance(native_manifest_sha256, str)
                or len(native_manifest_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in native_manifest_sha256)):
            raise ValueError("Exact lowercase native-manifest SHA-256 required")
        self.context, self.run, self.native_manifest_sha256 = context, run, native_manifest_sha256
        self.settings = settings
        self.native_domain = np.asarray(context.domain, bool)
        if self.native_domain.ndim != 2 or not self.native_domain.any():
            raise ValueError("Nonempty native porous domain required")
        nh, nw = self.native_domain.shape
        self.native_x = np.asarray(context.x_m, float)
        self.native_y = np.asarray(context.y_m, float)
        if (self.native_x.shape != (nw,) or self.native_y.shape != (nh,)
                or not np.isfinite(self.native_x).all() or not np.isfinite(self.native_y).all()
                or not (np.diff(self.native_x) > 0).all()
                or not (np.diff(self.native_y) < 0).all()):
            raise ValueError("Registered native x-right/y-up axes required")
        self.stages = tuple(context.injection_stages)
        if not self.stages:
            raise ValueError("Registered injection stages required")
        for stage in self.stages:
            if (not all(k in stage for k in ("x_m", "y_m", "start_h", "end_h"))
                    or not all(np.isfinite(float(stage[k])) for k in
                               ("x_m", "y_m", "start_h", "end_h"))
                    or float(stage["end_h"]) <= float(stage["start_h"])):
                raise ValueError("Invalid registered injection stage")
        self.first_injection_h = min(float(s["start_h"]) for s in self.stages)
        self.shape = None
        self.working_domain = None
        self.coverage = None
        self.spacing_m = None
        self.grid_x = None
        self.grid_y = None
        self.geometry_cache = {}
        self.last_frame_index = -1
        self.last_time_h = None
        self.preinjection_raw_presence = None
        self.presence_fast = None
        self.presence_slow = None
        self.change_slow = None
        self.pending = None

    def _bind_working_grid(self, prepared):
        dom = np.asarray(prepared["domain"])
        spacing = np.asarray(prepared["spacing_m"], float)
        if (dom.ndim != 2 or dom.dtype != bool or min(dom.shape) < 2
                or spacing.shape != (2,) or not np.isfinite(spacing).all()
                or (spacing <= 0).any()):
            raise ValueError("Invalid registered regional grid")
        shape = dom.shape
        if self.shape is None:
            self.shape = shape
            self.spacing_m = tuple(float(v) for v in spacing)
            self.coverage = cv2.resize(self.native_domain.astype(np.float32),
                                       shape[::-1], interpolation=cv2.INTER_AREA)
            self.working_domain = self.coverage > .999
            if not np.array_equal(dom, self.working_domain):
                raise ValueError("Prepared regional domain differs from native domain")
            nh, nw = self.native_domain.shape
            h, w = shape
            xx = (np.arange(w, dtype=np.float64) + .5) * nw / w - .5
            yy = (np.arange(h, dtype=np.float64) + .5) * nh / h - .5
            self.grid_x = (self.native_x[0] + xx * (self.native_x[-1] - self.native_x[0])
                           / (nw - 1)).astype(np.float32)
            self.grid_y = (self.native_y[0] + yy * (self.native_y[-1] - self.native_y[0])
                           / (nh - 1)).astype(np.float32)
        elif (shape != self.shape or not np.array_equal(dom, self.working_domain)
              or not np.allclose(spacing, self.spacing_m, rtol=1e-6, atol=0)):
            raise ValueError("The same run must retain one registered working grid")

    def _pool_raw_presence(self, raw_presence):
        raw = np.asarray(raw_presence)
        if (raw.shape != self.native_domain.shape or not np.isfinite(raw).all()
                or (raw < 0).any() or (raw > 1).any()):
            raise ValueError("Frozen raw optical presence must be a native unit-interval grid")
        weighted = cv2.resize(np.asarray(raw, np.float32) * self.native_domain,
                              self.shape[::-1], interpolation=cv2.INTER_AREA)
        pooled = np.divide(weighted, self.coverage,
                           out=np.zeros(self.shape, np.float32), where=self.coverage > 1e-6)
        pooled[~self.working_domain] = 0
        return pooled

    def _started_source_geometry(self, time_h):
        started = tuple(i for i, stage in enumerate(self.stages)
                        if time_h >= float(stage["start_h"]))
        if not started:
            return None
        if started not in self.geometry_cache:
            # Geometry is static for each started-stage set. Stream over
            # sources once instead of stacking 3 full grids per source on
            # every frame of a long native campaign.
            x, y = self.grid_x[None, :], self.grid_y[:, None]
            best = np.full(self.shape, np.inf, np.float32)
            offset = np.zeros(self.shape, np.float32)
            start = np.zeros(self.shape, np.float32)
            for index in started:
                stage = self.stages[index]
                delta_y = y - float(stage["y_m"])
                candidate = np.hypot(x - float(stage["x_m"]), delta_y).astype(np.float32)
                nearer = candidate < best
                best[nearer] = candidate[nearer]
                offset[nearer] = np.broadcast_to(delta_y, self.shape)[nearer]
                start[nearer] = float(stage["start_h"])
            for array in (best, offset, start):
                array.setflags(write=False)
            self.geometry_cache[started] = (best, offset, start)
        distance, vertical, start = self.geometry_cache[started]
        since_start = np.maximum(0., time_h - start).astype(np.float32)
        return distance, vertical, since_start

    def peek(self, *, frame_index: int, elapsed_h: float, prepared: dict,
             raw_presence: np.ndarray):
        """Return 10 extra features from prior evidence, without advancing history.

        Returns ``(None, audit)`` on pre-injection frames or if the exact
        source prefix lacks a pre-injection observation. These frames must use
        an explicitly recorded static fallback; missing context is never
        interpreted as absence of CO2.
        """
        if self.pending is not None:
            raise ValueError("Commit the preceding frame before another peek")
        if type(frame_index) is not int or frame_index != self.last_frame_index + 1:
            raise ValueError("Exact contiguous native frame order required")
        time_h = float(elapsed_h)
        if not np.isfinite(time_h) or (self.last_time_h is not None and time_h <= self.last_time_h):
            raise ValueError("Native observation times must strictly increase")
        self._bind_working_grid(prepared)
        change = np.asarray(prepared["change"], np.float32)
        if (change.shape != self.shape or not np.isfinite(change).all()
                or (change < 0).any()):
            raise ValueError("Invalid registered chromatic change")
        current_p = self._pool_raw_presence(raw_presence)
        self.pending = (frame_index, time_h, current_p.copy(), change.copy())
        base_audit = {"format": VERSION, "run": self.run,
                      "native_manifest_sha256": self.native_manifest_sha256,
                      "frame_index": frame_index, "elapsed_h": time_h,
                      "raw_optical_history_only": True, "final_mask_feedback": False,
                      "hard_connectivity_or_height_veto": False}
        if time_h < self.first_injection_h:
            return None, dict(base_audit, abstention="pre_injection_negative_control")
        if self.preinjection_raw_presence is None or self.last_time_h is None:
            return None, dict(base_audit, abstention="missing_pre_injection_history")
        geometry = self._started_source_geometry(time_h)
        if geometry is None:
            return None, dict(base_audit, abstention="no_started_registered_source")
        distance, vertical, since_start = geometry
        sy, sx = (self.settings.neighborhood_sigma_m / s for s in self.spacing_m)
        weight = cv2.GaussianBlur(self.working_domain.astype(np.float32), (0, 0),
                                  sigmaX=sx, sigmaY=sy)
        numerator = cv2.GaussianBlur(self.presence_fast * self.working_domain,
                                      (0, 0), sigmaX=sx, sigmaY=sy)
        neighbor = np.divide(numerator, weight, out=np.zeros(self.shape, np.float32),
                             where=weight > 1e-6)
        dt = time_h - self.last_time_h
        extra = np.stack((
            self.preinjection_raw_presence, self.presence_fast, self.presence_slow,
            self.change_slow, neighbor, current_p - self.presence_fast,
            distance, vertical, np.log1p(since_start),
            np.full(self.shape, np.log1p(dt), np.float32),
        ), axis=-1).astype(np.float32)
        extra[~self.working_domain] = 0
        if not np.isfinite(extra).all():
            raise ValueError("Causal temporal features must be finite")
        return extra, dict(base_audit, abstention="none", feature_names=FEATURE_NAMES,
                           previous_observation_gap_h=dt,
                           preinjection_observed=True)

    def commit(self, *, frame_index: int):
        """Advance raw evidence only after the caller has scored this frame."""
        if self.pending is None or frame_index != self.pending[0]:
            raise ValueError("Commit requires the exact frame returned by peek")
        _, time_h, current_p, change = self.pending
        if time_h < self.first_injection_h:
            # The latest exact pre-injection frame defines run-start residual.
            self.preinjection_raw_presence = current_p.copy()
            self.presence_fast = current_p.copy()
            self.presence_slow = current_p.copy()
            self.change_slow = change.copy()
        elif self.preinjection_raw_presence is not None:
            dt = time_h - self.last_time_h
            fast = -np.expm1(-dt / self.settings.fast_ema_h)
            slow = -np.expm1(-dt / self.settings.slow_ema_h)
            self.presence_fast += fast * (current_p - self.presence_fast)
            self.presence_slow += slow * (current_p - self.presence_slow)
            self.change_slow += slow * (change - self.change_slow)
        self.last_frame_index = frame_index
        self.last_time_h = time_h
        self.pending = None
