"""Optional, ordered check for isolated regional aqueous-support explosions.

The optical phase map is the primary evidence. This module may withdraw only
aqueous pixels that a regional model added to optical water in one photograph.
It does not inspect acquisition time or vertical position, and it leaves gas
decisions and every optically supported pixel unchanged.

The guard needs the next independently predicted frame. Callers must therefore
delay final map/report materialization by one frame and bind this policy and its
settings into a *new* pipeline identity. Existing model maps are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

import numpy as np


VERSION = "isolated-regional-aqueous-support-guard-v1"


@dataclass(frozen=True)
class Settings:
    """Conservative trigger and physical buffer, all independent of run time."""

    minimum_excess_fraction: float = 0.10
    minimum_excess_area_mm2: float = 100.0
    minimum_regional_share_of_excess: float = 0.80
    neighbor_buffer_mm: float = 10.0

    def __post_init__(self) -> None:
        if not (0 < self.minimum_excess_fraction < 1):
            raise ValueError("minimum_excess_fraction must be between 0 and 1")
        if not (isfinite(self.minimum_excess_area_mm2) and self.minimum_excess_area_mm2 > 0):
            raise ValueError("minimum_excess_area_mm2 must be positive and finite")
        if not (0 < self.minimum_regional_share_of_excess <= 1):
            raise ValueError("minimum_regional_share_of_excess must be in (0, 1]")
        if not (isfinite(self.neighbor_buffer_mm) and self.neighbor_buffer_mm >= 0):
            raise ValueError("neighbor_buffer_mm must be nonnegative and finite")


def _spacing(spacing_mm: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(spacing_mm, (int, float)):
        values = (float(spacing_mm), float(spacing_mm))
    else:
        values = tuple(float(x) for x in spacing_mm)
    if len(values) != 2 or not all(isfinite(x) and x > 0 for x in values):
        raise ValueError("spacing_mm must contain positive row/column pixel sizes")
    return values


def guard_triplet(
    previous_proposed: np.ndarray,
    current_raw_optical: np.ndarray,
    current_proposed: np.ndarray,
    following_proposed: np.ndarray,
    *,
    spacing_mm: float | tuple[float, float],
    settings: Settings = Settings(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return guarded current labels and a diagnostic receipt.

    Labels use 0 for water, 1 for aqueous CO2 and other positive values for
    existing nonaqueous phases. All four maps must share registered geometry.
    ``previous_proposed`` and ``following_proposed`` are independent, *unguarded*
    frame predictions. Only label-1 pixels absent from ``current_raw_optical``
    are eligible for withdrawal. Source arrays are never modified.
    """

    row_mm, column_mm = _spacing(spacing_mm)
    arrays = tuple(np.asarray(a) for a in (
        previous_proposed, current_raw_optical, current_proposed, following_proposed
    ))
    if any(a.ndim != 2 or a.shape != arrays[0].shape or not np.issubdtype(a.dtype, np.integer)
           or np.any(a < 0) for a in arrays):
        raise ValueError("phase labels must be nonnegative integer arrays with identical 2D geometry")
    previous, raw, proposed, following = arrays
    previous_area = int(np.count_nonzero(previous))
    following_area = int(np.count_nonzero(following))
    current_area = int(np.count_nonzero(proposed))
    neighbor_max = max(previous_area, following_area)
    excess = max(0, current_area - neighbor_max)
    regional_additions = (proposed == 1) & (raw == 0)
    added_count = int(np.count_nonzero(regional_additions))
    minimum_pixels = ceil(settings.minimum_excess_area_mm2 / (row_mm * column_mm))
    triggered = (
        excess > minimum_pixels
        and excess > settings.minimum_excess_fraction * max(neighbor_max, 1)
        and added_count >= settings.minimum_regional_share_of_excess * excess
    )
    diagnostic: dict[str, Any] = {
        "format": VERSION,
        "triggered": triggered,
        "previous_support_pixels": previous_area,
        "current_support_pixels": current_area,
        "following_support_pixels": following_area,
        "excess_over_neighbor_pixels": excess,
        "regional_added_pixels": added_count,
        "withdrawn_aqueous_pixels": 0,
        "neighbor_buffer_mm": settings.neighbor_buffer_mm,
        "optical_support_preserved": True,
        "nonaqueous_labels_preserved": True,
    }
    if not triggered or added_count == 0:
        return proposed, diagnostic

    # Requiring agreement with nearby frames applies only to a transient
    # regional addition. Newly appearing optical evidence is always retained.
    from scipy.ndimage import distance_transform_edt

    neighbor_support = (previous != 0) | (following != 0)
    distance_mm = distance_transform_edt(~neighbor_support, sampling=(row_mm, column_mm))
    withdraw = regional_additions & (distance_mm > settings.neighbor_buffer_mm)
    removed_count = int(np.count_nonzero(withdraw))
    if removed_count:
        guarded = proposed.copy()
        guarded[withdraw] = 0
    else:
        guarded = proposed
    diagnostic["withdrawn_aqueous_pixels"] = removed_count
    diagnostic["retained_regional_added_pixels"] = added_count - removed_count
    diagnostic["guarded_support_pixels"] = current_area - removed_count
    return guarded, diagnostic


@dataclass(frozen=True)
class GuardedFrame:
    index: int
    labels: np.ndarray
    diagnostics: dict[str, Any]
    payload: Any = None


@dataclass
class _PendingFrame:
    index: int
    raw: np.ndarray
    proposed: np.ndarray
    payload: Any


class OrderedFrameGuard:
    """One-frame-delay adapter for ordered predictions and final materialization.

    ``push`` yields no frame at first, then the first boundary frame unchanged,
    then each middle frame after its successor arrives. ``finish`` yields the
    final boundary frame unchanged. The caller owns input arrays until emitted.
    """

    def __init__(self, *, spacing_mm: float | tuple[float, float], settings: Settings = Settings()):
        self.spacing_mm = _spacing(spacing_mm)
        self.settings = settings
        self._previous: _PendingFrame | None = None
        self._pending: _PendingFrame | None = None
        self._last_index: int | None = None
        self._finished = False

    @staticmethod
    def _boundary(frame: _PendingFrame) -> GuardedFrame:
        return GuardedFrame(frame.index, frame.proposed, {
            "format": VERSION, "triggered": False, "boundary_frame": True,
            "withdrawn_aqueous_pixels": 0,
        }, frame.payload)

    def push(self, index: int, raw_optical: np.ndarray, proposed: np.ndarray,
             payload: Any = None) -> list[GuardedFrame]:
        if self._finished:
            raise RuntimeError("cannot push after finish")
        if self._last_index is not None and index != self._last_index + 1:
            raise ValueError("frame indices must be consecutive and increasing")
        current = _PendingFrame(index, np.asarray(raw_optical), np.asarray(proposed), payload)
        if current.raw.ndim != 2 or current.raw.shape != current.proposed.shape:
            raise ValueError("raw and proposed labels must share 2D geometry")
        self._last_index = index
        if self._pending is None:
            self._pending = current
            return []
        if self._previous is None:
            emitted = [self._boundary(self._pending)]
        else:
            labels, diagnostics = guard_triplet(
                self._previous.proposed, self._pending.raw, self._pending.proposed,
                current.proposed, spacing_mm=self.spacing_mm, settings=self.settings,
            )
            emitted = [GuardedFrame(self._pending.index, labels, diagnostics, self._pending.payload)]
        self._previous, self._pending = self._pending, current
        return emitted

    def finish(self) -> list[GuardedFrame]:
        if self._finished:
            raise RuntimeError("finish already called")
        self._finished = True
        return [] if self._pending is None else [self._boundary(self._pending)]
