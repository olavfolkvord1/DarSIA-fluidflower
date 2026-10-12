"""Causal, bounded hypotheses for gas hidden behind the photographed face.

This is an *optional scenario*, not a 3D reconstruction or a change to the
observed 2D phase map. A bridge may explain how two visible gas components
could be connected behind the face; it does not prove that they are connected
at the observation time. A previously connected residual ganglion is valid
without a present path to an injector.

Gas in a candidate bridge displaces *affected aqueous pore volume*. It is
never added on top of a full-depth visible-gas column. The reported lower and
upper bounds are geometric scenario bounds, not statistical confidence limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

VERSION = "hidden-gas-depth-corridor-and-persistent-halo-v3"


@dataclass(frozen=True)
class HiddenGasDepthSettings:
    """Declared assumptions, shared across runs when comparing scenarios.

    `max_gap_m` limits the nearest visible-component separation. Corridors
    traverse only affected aqueous cells above `minimum_aqueous_fraction` and
    never cross a supplied barrier mask. `hidden_depth_fraction` is a scenario
    occupying that fraction of the *remaining affected aqueous volume* in a
    candidate corridor; it is not fitted or inferred here.

    The opt-in persistent halo is a separate sensitivity hypothesis around
    currently visible, repeatedly observed gas. Its score combines current
    local visible-gas density, causal uninterrupted age, and bounded physical
    travel distance through affected aqueous cells. Setting
    `halo_require_same_facies` prevents crossing an integer facies boundary;
    `barrier_mask` remains an independent hard obstruction. An exponent below
    one strengthens score-to-occupancy sensitivity without expanding reach.
    """

    minimum_component_area_m2: float = 2.5e-6
    max_gap_m: float = 0.020
    max_path_to_gap_ratio: float = 2.0
    corridor_radius_m: float = 0.001
    minimum_aqueous_fraction: float = 0.05
    hidden_depth_fraction: float = 0.25
    weak_aqueous_cost: float = 2.0
    previous_gas_cost_factor: float = 0.75
    persistence_scale_h: float = 2.0
    max_components: int = 24
    max_candidates: int = 12
    max_search_nodes: int = 100_000
    causal_visible_saturation_enabled: bool = False
    # Distinct opt-in rule: only repeatedly observed gas in the same physical
    # grid cell can occupy the full depth. A long gap or a missing observation
    # starts a new streak; annotation labels alone do not establish a streak.
    causal_visible_full_depth_enabled: bool = False
    visible_saturation_min: float = 0.6
    visible_saturation_max: float = 1.0
    visible_persistence_tau_h: float = 2.0
    full_depth_required_observations: int = 3
    full_depth_min_persistence_h: float = 0.5
    full_depth_max_observation_gap_h: float = 1.0
    full_depth_min_visible_gas_fraction: float = 0.5
    corridor_enabled: bool = True
    persistent_halo_enabled: bool = False
    halo_max_distance_m: float = 0.005
    halo_decay_m: float = 0.005
    halo_density_sigma_m: float = 0.003
    halo_persistence_tau_h: float = 2.0
    halo_max_depth_fraction: float = 0.25
    halo_support_exponent: float = 1.0
    halo_require_same_facies: bool = False
    halo_max_steps: int = 64

    def __post_init__(self) -> None:
        nonnegative = (
            "minimum_component_area_m2", "corridor_radius_m", "weak_aqueous_cost",
        )
        positive = ("max_gap_m", "max_path_to_gap_ratio", "persistence_scale_h",
                    "visible_persistence_tau_h", "halo_max_distance_m",
                    "halo_decay_m", "halo_density_sigma_m", "halo_persistence_tau_h",
                    "full_depth_min_persistence_h", "full_depth_max_observation_gap_h")
        for name in (*nonnegative, *positive, "minimum_aqueous_fraction",
                     "hidden_depth_fraction", "previous_gas_cost_factor",
                     "halo_max_depth_fraction", "halo_support_exponent",
                     "full_depth_min_visible_gas_fraction"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_path_to_gap_ratio < 1:
            raise ValueError("max_path_to_gap_ratio must be at least one")
        if not 0 < self.minimum_aqueous_fraction <= 1:
            raise ValueError("minimum_aqueous_fraction must be in (0, 1]")
        if not 0 <= self.hidden_depth_fraction <= 1:
            raise ValueError("hidden_depth_fraction must be in [0, 1]")
        if not 0 <= self.halo_max_depth_fraction <= 1:
            raise ValueError("halo_max_depth_fraction must be in [0, 1]")
        if not 0 < self.halo_support_exponent <= 1:
            raise ValueError("halo_support_exponent must be in (0, 1]")
        if not 0 < self.previous_gas_cost_factor <= 1:
            raise ValueError("previous_gas_cost_factor must be in (0, 1]")
        if not 0 < self.full_depth_min_visible_gas_fraction <= 1:
            raise ValueError("full_depth_min_visible_gas_fraction must be in (0, 1]")
        for name in ("causal_visible_saturation_enabled", "corridor_enabled",
                     "causal_visible_full_depth_enabled", "persistent_halo_enabled",
                     "halo_require_same_facies"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if not 0 <= self.visible_saturation_min <= self.visible_saturation_max <= 1:
            raise ValueError("Visible saturation bounds must satisfy 0 <= min <= max <= 1")
        if self.causal_visible_full_depth_enabled:
            if self.causal_visible_saturation_enabled:
                raise ValueError("Select only one causal visible-saturation rule")
            if self.visible_saturation_max != 1:
                raise ValueError("Exact full-depth saturation requires visible_saturation_max=1")
        for name in ("max_components", "max_candidates", "max_search_nodes",
                     "halo_max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.full_depth_required_observations, bool)
                or not isinstance(self.full_depth_required_observations, (int, np.integer))
                or self.full_depth_required_observations < 2):
            raise ValueError("full_depth_required_observations must be an integer >= 2")


@dataclass(frozen=True)
class HiddenGasBridge:
    """One proposed hidden route between two visible components."""

    source_component: int
    target_component: int
    nearest_gap_m: float
    route_length_m: float
    route_cost_m: float
    source_persistence_h: float
    target_persistence_h: float
    corridor_pore_volume_m3: float


@dataclass(frozen=True)
class HiddenGasDepthFrame:
    """Independent scenario fields; never overwrite an observed phase map."""

    visible_gas_mask: NDArray[np.bool_]
    unsupported_visible_gas_mask: NDArray[np.bool_]
    candidate_mask: NDArray[np.bool_]
    halo_candidate_mask: NDArray[np.bool_]
    halo_raw_support_score: NDArray[np.float32]
    halo_support_score: NDArray[np.float32]
    observed_gas_occupancy: NDArray[np.float32]
    visible_gas_saturation: NDArray[np.float32]
    hidden_occupancy_scenario: NDArray[np.float32]
    corridor_hidden_occupancy_scenario: NDArray[np.float32]
    halo_hidden_occupancy_scenario: NDArray[np.float32]
    total_gas_occupancy_scenario: NDArray[np.float32]
    residual_affected_aqueous_occupancy: NDArray[np.float32]
    persistence_h: NDArray[np.float32]
    visible_gas_consecutive_observations: NDArray[np.int32]
    visible_gas_volume_m3: float
    hidden_gas_volume_scenario_m3: float
    total_gas_volume_scenario_m3: float
    maximum_hidden_capacity_m3: float
    visible_gas_mass_g: float | None
    hidden_gas_mass_scenario_g: float | None
    visible_mass_over_injected_g: float | None
    bridges: tuple[HiddenGasBridge, ...]
    eligible_component_count: int
    ignored_component_count: int
    elapsed_h: float
    injected_mass_g: float
    method_version: str


def _fraction(name: str, value: ArrayLike, shape: tuple[int, int] | None = None) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must be a nonempty 2D field of matching shape")
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return array


def _candidate_pairs(
    labels: NDArray[np.int32],
    components: list[int],
    persistence_h: NDArray[np.float64],
    spacing: tuple[float, float],
    max_gap_m: float,
    persistence_scale_h: float,
) -> list[tuple[float, int, int, tuple[int, int], tuple[int, int], float, float]]:
    if len(components) < 2:
        return []
    # A single boundary image avoids repeated full-grid erosions. A diagonal
    # contact is already one component because components use 8-connectivity.
    interior = ndi.binary_erosion(labels > 0, structure=np.ones((3, 3)), border_value=0)
    boundary = (labels > 0) & ~interior
    # The erosion above also marks boundaries between separate components;
    # component-specific coordinates are used for nearest-gap search.
    points = {index: np.argwhere(boundary & (labels == index)) for index in components}
    trees = {index: cKDTree(points[index] * spacing) for index in components}
    pairs = []
    for at, source in enumerate(components):
        coords = points[source]
        for target in components[at + 1:]:
            distances, nearest = trees[target].query(coords * spacing, k=1)
            source_index = int(np.argmin(distances))
            gap = float(distances[source_index])
            if gap > max_gap_m:
                continue
            start = tuple(int(v) for v in coords[source_index])
            end = tuple(int(v) for v in points[target][nearest[source_index]])
            source_age = float(np.max(persistence_h[labels == source]))
            target_age = float(np.max(persistence_h[labels == target]))
            pairs.append((gap, source, target, start, end, source_age, target_age))
    # Prefer equally plausible gaps whose two endpoints have persisted, but
    # never relax the physical maximum gap or turn persistence into mass.
    return sorted(pairs, key=lambda item: (
        item[0] / (1 + min(1.0, min(item[5], item[6]) / persistence_scale_h)),
        item[0],
    ))


def _route(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    traversable: NDArray[np.bool_],
    cell_cost: NDArray[np.float64],
    spacing: tuple[float, float],
    gap_m: float,
    settings: HiddenGasDepthSettings,
) -> tuple[list[tuple[int, int]], float, float] | None:
    """Bounded four-neighbour A* in a local physical-distance window."""
    radius = settings.max_gap_m
    y_radius = int(math.ceil(radius / spacing[0]))
    x_radius = int(math.ceil(radius / spacing[1]))
    y0 = max(0, min(start[0], end[0]) - y_radius)
    y1 = min(traversable.shape[0], max(start[0], end[0]) + y_radius + 1)
    x0 = max(0, min(start[1], end[1]) - x_radius)
    x1 = min(traversable.shape[1], max(start[1], end[1]) + x_radius + 1)
    allowed = traversable[y0:y1, x0:x1]
    cost = cell_cost[y0:y1, x0:x1]
    source = (start[0] - y0, start[1] - x0)
    target = (end[0] - y0, end[1] - x0)
    if not allowed[source] or not allowed[target]:
        return None
    distances = np.full(allowed.shape, np.inf)
    distances[source] = 0.0
    predecessor: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(gap_m, 0.0, source)]
    searched = 0
    while queue and searched < settings.max_search_nodes:
        _estimate, travelled, node = heapq.heappop(queue)
        if travelled > distances[node]:
            continue
        if node == target:
            break
        searched += 1
        y, x = node
        for dy, dx, step in ((-1, 0, spacing[0]), (1, 0, spacing[0]),
                             (0, -1, spacing[1]), (0, 1, spacing[1])):
            nxt = (y + dy, x + dx)
            if not (0 <= nxt[0] < allowed.shape[0] and 0 <= nxt[1] < allowed.shape[1]):
                continue
            if not allowed[nxt]:
                continue
            candidate = travelled + step * 0.5 * (cost[node] + cost[nxt])
            if candidate >= distances[nxt]:
                continue
            distances[nxt] = candidate
            predecessor[nxt] = node
            remaining = math.hypot((target[0] - nxt[0]) * spacing[0],
                                   (target[1] - nxt[1]) * spacing[1])
            heapq.heappush(queue, (
                candidate + remaining * settings.previous_gas_cost_factor,
                candidate, nxt,
            ))
    if not np.isfinite(distances[target]):
        return None
    local_path = [target]
    while local_path[-1] != source:
        local_path.append(predecessor[local_path[-1]])
    local_path.reverse()
    path = [(y + y0, x + x0) for y, x in local_path]
    length = sum(spacing[0] if a[0] != b[0] else spacing[1]
                 for a, b in zip(path, path[1:]))
    if length > settings.max_path_to_gap_ratio * max(gap_m, min(spacing)):
        return None
    return path, float(length), float(distances[target])


def _geodesic_corridor(
    path: list[tuple[int, int]],
    traversable: NDArray[np.bool_],
    spacing: tuple[float, float],
    radius_m: float,
) -> NDArray[np.bool_]:
    """Expand a route by physical distance without jumping a thin barrier."""
    corridor = np.zeros(traversable.shape, dtype=bool)
    if radius_m == 0:
        row, col = zip(*path)
        corridor[row, col] = True
        return corridor
    ry = int(math.ceil(radius_m / spacing[0]))
    rx = int(math.ceil(radius_m / spacing[1]))
    y0 = max(0, min(y for y, _ in path) - ry)
    y1 = min(traversable.shape[0], max(y for y, _ in path) + ry + 1)
    x0 = max(0, min(x for _, x in path) - rx)
    x1 = min(traversable.shape[1], max(x for _, x in path) + rx + 1)
    allowed = traversable[y0:y1, x0:x1]
    distance = np.full(allowed.shape, np.inf)
    queue: list[tuple[float, int, int]] = []
    for y, x in path:
        local = (y - y0, x - x0)
        distance[local] = 0.0
        queue.append((0.0, *local))
    heapq.heapify(queue)
    while queue:
        traveled, y, x = heapq.heappop(queue)
        if traveled > radius_m:
            break
        if traveled > distance[y, x]:
            continue
        for dy, dx, step in ((-1, 0, spacing[0]), (1, 0, spacing[0]),
                             (0, -1, spacing[1]), (0, 1, spacing[1])):
            yy, xx = y + dy, x + dx
            next_distance = traveled + step
            if (0 <= yy < allowed.shape[0] and 0 <= xx < allowed.shape[1]
                    and allowed[yy, xx] and next_distance <= radius_m
                    and next_distance < distance[yy, xx]):
                distance[yy, xx] = next_distance
                heapq.heappush(queue, (next_distance, yy, xx))
    corridor[y0:y1, x0:x1] = distance <= radius_m
    return corridor


def _persistent_halo(
    *,
    gas: NDArray[np.float64],
    aqueous: NDArray[np.float64],
    pore: NDArray[np.float64],
    eligible_visible: NDArray[np.bool_],
    persistence_h: NDArray[np.float64],
    barrier: NDArray[np.bool_],
    facies: NDArray[np.int64] | None,
    spacing: tuple[float, float],
    settings: HiddenGasDepthSettings,
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Bounded four-neighbour influence from persistent, dense visible gas.

    Each propagated score carries its own traveled distance, so a high-score
    source cannot borrow a shorter route from a different source to pass a
    barrier or the maximum distance. One best state per cell makes this a
    conservative sensitivity screen, not an exact multipath posterior.
    """
    allowed = (((aqueous >= settings.minimum_aqueous_fraction)
                | eligible_visible) & (pore > 0) & ~barrier)
    sigma = tuple(settings.halo_density_sigma_m / side for side in spacing)
    density = ndi.gaussian_filter(gas, sigma=sigma, mode="constant")
    with np.errstate(over="ignore", under="ignore"):
        age_weight = -np.expm1(-persistence_h / settings.halo_persistence_tau_h)
    source_score = np.where(eligible_visible & allowed,
                            np.clip(density * age_weight * gas, 0, 1), 0.0)
    if not np.any(source_score > 0):
        return np.zeros(gas.shape, dtype=bool), np.zeros(gas.shape, dtype=float)
    max_steps = int(math.ceil(settings.halo_max_distance_m / min(spacing)))
    if max_steps > settings.halo_max_steps:
        raise ValueError(
            "Persistent halo needs a coarser grid or smaller halo_max_distance_m"
        )
    score = source_score.copy()
    distance = np.where(source_score > 0, 0.0, np.inf)
    directions = (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), spacing[0]),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), spacing[0]),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), spacing[1]),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), spacing[1]),
    )
    for _ in range(max_steps):
        next_score = score.copy()
        next_distance = distance.copy()
        changed = False
        for target, source, step_m in directions:
            valid = allowed[target] & allowed[source]
            if facies is not None and settings.halo_require_same_facies:
                valid &= facies[target] == facies[source]
            candidate_distance = distance[source] + step_m
            candidate_score = score[source] * math.exp(-step_m / settings.halo_decay_m)
            improve = (valid & (candidate_distance <= settings.halo_max_distance_m)
                       & (candidate_score > next_score[target] + 1e-15))
            if np.any(improve):
                np.copyto(next_score[target], candidate_score, where=improve)
                np.copyto(next_distance[target], candidate_distance, where=improve)
                changed = True
        score, distance = next_score, next_distance
        if not changed:
            break
    candidate = (score > 0) & (aqueous >= settings.minimum_aqueous_fraction)
    candidate &= ~(gas > 0) & ~barrier & (pore > 0)
    return candidate, np.where(candidate, score, 0.0)


class VisibleFullDepthSaturationTracker:
    """Memory-light version of the visible-only causal full-depth hypothesis.

    This tracker returns saturation for *currently visible* gas cells only.
    It does not label components, infer hidden corridors or halos, or modify
    phase masks. It is useful for dense concentration studies on full grids.
    The same-cell rule matches ``HiddenGasDepthTracker`` with its other
    hypotheses disabled.
    """

    def __init__(self, settings: HiddenGasDepthSettings) -> None:
        if (not settings.causal_visible_full_depth_enabled
                or settings.corridor_enabled or settings.persistent_halo_enabled):
            raise ValueError("Visible-only full-depth settings must disable corridors and halos")
        self.settings = settings
        self._elapsed_h: float | None = None
        self._observations: NDArray[np.int32] | None = None
        self._persistence_h: NDArray[np.float32] | None = None

    def update(self, visible_gas_pore_fraction: ArrayLike, *, elapsed_h: float,
               injected_mass_g: float) -> NDArray[np.float32]:
        gas = _fraction("visible_gas_pore_fraction", visible_gas_pore_fraction)
        elapsed_h = float(elapsed_h)
        injected_mass_g = float(injected_mass_g)
        if not np.isfinite(elapsed_h) or not np.isfinite(injected_mass_g) or injected_mass_g < 0:
            raise ValueError("Elapsed time and injected mass must be finite; mass nonnegative")
        if self._elapsed_h is not None and elapsed_h <= self._elapsed_h:
            raise ValueError("elapsed_h must increase strictly")
        if self._observations is not None and self._observations.shape != gas.shape:
            raise ValueError("Grid shape cannot change within an experiment")
        visible = gas > 0
        core = (visible & (gas >= self.settings.full_depth_min_visible_gas_fraction)
                & (injected_mass_g > 0) & (elapsed_h > 0))
        if self._observations is None:
            continued = np.zeros(gas.shape, dtype=bool)
            previous_count = np.zeros(gas.shape, dtype=np.int32)
            previous_age = np.zeros(gas.shape, dtype=np.float32)
            delta = 0.0
        else:
            delta = elapsed_h - self._elapsed_h
            previous_count = self._observations
            previous_age = self._persistence_h
            continued = (core & (previous_count > 0)
                         & (delta <= self.settings.full_depth_max_observation_gap_h))
        observations = np.where(continued,
            np.minimum(previous_count + 1, self.settings.full_depth_required_observations),
            core.astype(np.int32)).astype(np.int32)
        persistence = np.where(continued, previous_age + delta, 0).astype(np.float32)
        full_depth = (core
                      & (observations >= self.settings.full_depth_required_observations)
                      & (persistence >= self.settings.full_depth_min_persistence_h))
        saturation = np.where(visible,
            np.where(full_depth, 1.0, self.settings.visible_saturation_min), 0).astype(np.float32)
        self._observations = observations
        self._persistence_h = persistence
        self._elapsed_h = elapsed_h
        return saturation


class HiddenGasDepthTracker:
    """Generate optional causal hidden-depth scenarios per observation.

    Inputs are fractions of the *same physical pore column*. They can be
    derived from saved `gas_volume_m3 / pore_volume_m3` and
    `aqueous_volume_m3 / pore_volume_m3`. Observed gas saturation may be a
    scalar or a causal field such as `GasHistoryTracker` output. Calls must be
    in strictly increasing time order and belong to one experiment.
    """

    def __init__(
        self,
        cell_size_m: tuple[float, float],
        settings: HiddenGasDepthSettings | None = None,
    ) -> None:
        spacing = np.asarray(cell_size_m, dtype=float)
        if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("cell_size_m must be positive finite (dy, dx)")
        self.cell_size_m = (float(spacing[0]), float(spacing[1]))
        self.settings = settings or HiddenGasDepthSettings()
        self._elapsed_h: float | None = None
        self._previous_gas: NDArray[np.bool_] | None = None
        self._persistence_h: NDArray[np.float64] | None = None
        self._visible_gas_consecutive_observations: NDArray[np.int32] | None = None

    def update(
        self,
        *,
        visible_gas_pore_fraction: ArrayLike,
        aqueous_pore_fraction: ArrayLike,
        pore_volume_m3: ArrayLike,
        elapsed_h: float,
        injected_mass_g: float,
        visible_gas_saturation: ArrayLike | float | None = None,
        gas_density_g_l: ArrayLike | float | None = None,
        barrier_mask: ArrayLike | None = None,
        facies_labels: ArrayLike | None = None,
    ) -> HiddenGasDepthFrame:
        gas = _fraction("visible_gas_pore_fraction", visible_gas_pore_fraction)
        aqueous = _fraction("aqueous_pore_fraction", aqueous_pore_fraction, gas.shape)
        pore = np.asarray(pore_volume_m3, dtype=np.float64)
        if pore.shape != gas.shape or not np.all(np.isfinite(pore)) or np.any(pore < 0):
            raise ValueError("pore_volume_m3 must be finite, nonnegative, and match the grid")
        if np.any(gas + aqueous > 1 + 1e-6):
            raise ValueError("Gas and aqueous fractions exceed pore capacity")
        if barrier_mask is None:
            barrier = np.zeros(gas.shape, dtype=bool)
        else:
            barrier = np.asarray(barrier_mask)
            if barrier.shape != gas.shape or barrier.dtype != np.bool_:
                raise ValueError("barrier_mask must be a matching boolean field")
            barrier = barrier.copy()
        facies = None
        if facies_labels is not None:
            facies = np.asarray(facies_labels)
            if facies.shape != gas.shape or not np.issubdtype(facies.dtype, np.integer):
                raise ValueError("facies_labels must be a matching integer field")
            facies = facies.astype(np.int64, copy=False)
        if (self.settings.persistent_halo_enabled
                and self.settings.halo_require_same_facies and facies is None):
            raise ValueError("Same-facies halo requires facies_labels")
        elapsed_h = float(elapsed_h)
        injected_mass_g = float(injected_mass_g)
        if not np.isfinite(elapsed_h) or not np.isfinite(injected_mass_g) or injected_mass_g < 0:
            raise ValueError("Elapsed time and injected mass must be finite; mass nonnegative")
        if self._elapsed_h is not None and elapsed_h <= self._elapsed_h:
            raise ValueError("elapsed_h must increase strictly")
        if self._previous_gas is not None and self._previous_gas.shape != gas.shape:
            raise ValueError("Grid shape cannot change within an experiment")

        visible = (gas > 0) & (pore > 0)
        trackable_visible = (visible if injected_mass_g > 0 and elapsed_h > 0
                             else np.zeros(gas.shape, dtype=bool))
        if self.settings.causal_visible_full_depth_enabled:
            # Area-pooled edge slivers do not establish a full-depth gas core.
            core = trackable_visible & (
                gas >= self.settings.full_depth_min_visible_gas_fraction)
            if self._visible_gas_consecutive_observations is None:
                continued = np.zeros(gas.shape, dtype=bool)
                previous_count = np.zeros(gas.shape, dtype=np.int32)
                previous_age = np.zeros(gas.shape, dtype=np.float64)
                delta = 0.0
            else:
                delta = elapsed_h - self._elapsed_h
                previous_count = self._visible_gas_consecutive_observations
                previous_age = self._persistence_h
                continued = (core & (previous_count > 0)
                             & (delta <= self.settings.full_depth_max_observation_gap_h))
            observations = np.where(
                continued,
                np.minimum(previous_count + 1,
                           self.settings.full_depth_required_observations),
                core.astype(np.int32),
            ).astype(np.int32)
            persistence = np.where(continued, previous_age + delta, 0.0)
        elif self._previous_gas is None:
            persistence = np.zeros(gas.shape, dtype=np.float64)
            observations = np.zeros(gas.shape, dtype=np.int32)
        else:
            delta = elapsed_h - self._elapsed_h
            persistence = np.where(trackable_visible & self._previous_gas,
                                   self._persistence_h + delta, 0.0)
            observations = np.zeros(gas.shape, dtype=np.int32)
        if visible_gas_saturation is None:
            if self.settings.causal_visible_full_depth_enabled:
                full_depth = (core
                              & (observations >= self.settings.full_depth_required_observations)
                              & (persistence >= self.settings.full_depth_min_persistence_h))
                saturation = np.where(full_depth, 1.0,
                                      self.settings.visible_saturation_min)
            elif self.settings.causal_visible_saturation_enabled:
                saturation = (self.settings.visible_saturation_min
                              + (self.settings.visible_saturation_max
                                 - self.settings.visible_saturation_min)
                              * -np.expm1(-persistence / self.settings.visible_persistence_tau_h))
            else:
                saturation = np.ones(gas.shape, dtype=np.float64)
        else:
            saturation = np.asarray(visible_gas_saturation, dtype=np.float64)
            if saturation.ndim == 0:
                saturation = np.full(gas.shape, float(saturation))
        if (saturation.shape != gas.shape or not np.all(np.isfinite(saturation))
                or np.any((saturation < 0) | (saturation > 1))):
            raise ValueError("visible_gas_saturation must be finite and in [0, 1]")
        observed_occupancy = gas * saturation
        # If visible gas is not fully gas-saturated, the rest of that labelled
        # pore volume remains aqueous. The corridor excludes visible cells.
        initial_aqueous = aqueous + gas * (1 - saturation)
        initial_aqueous = np.minimum(initial_aqueous, 1 - observed_occupancy)
        labels, number = ndi.label(visible, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(labels.ravel(), minlength=number + 1)
        eligible = [int(i) for i in range(1, number + 1)
                    if sizes[i] * self.cell_size_m[0] * self.cell_size_m[1]
                    >= self.settings.minimum_component_area_m2]
        eligible.sort(key=lambda index: (-sizes[index], index))
        ignored = number - min(len(eligible), self.settings.max_components)
        eligible = eligible[:self.settings.max_components]
        unsupported = visible & ~np.isin(labels, eligible)
        if injected_mass_g == 0 or elapsed_h <= 0:
            unsupported = visible.copy()
            eligible = []
            ignored = number
        if self.settings.persistent_halo_enabled and eligible and gas_density_g_l is None:
            raise ValueError("Persistent halo requires gas_density_g_l for the injected-mass cap")

        candidate = np.zeros(gas.shape, dtype=bool)
        bridges: list[HiddenGasBridge] = []
        if self.settings.corridor_enabled and len(eligible) > 1:
            pairs = _candidate_pairs(
                labels, eligible, persistence, self.cell_size_m,
                self.settings.max_gap_m, self.settings.persistence_scale_h,
            )
            traversable = ((aqueous >= self.settings.minimum_aqueous_fraction)
                           & (pore > 0) & ~barrier) | (visible & ~barrier)
            cost = np.where(visible, 1.0,
                            1.0 + self.settings.weak_aqueous_cost * (1 - aqueous))
            if self._previous_gas is not None:
                cost = np.where(self._previous_gas & ~visible,
                                cost * self.settings.previous_gas_cost_factor, cost)
            for gap, source, target, start, end, source_age, target_age in pairs:
                if len(bridges) >= self.settings.max_candidates:
                    break
                route = _route(start, end, traversable=traversable,
                               cell_cost=cost, spacing=self.cell_size_m,
                               gap_m=gap, settings=self.settings)
                if route is None:
                    continue
                path, length, route_cost = route
                path_mask = _geodesic_corridor(
                    path, traversable, self.cell_size_m,
                    self.settings.corridor_radius_m,
                )
                path_mask &= ~visible
                # Shared corridor cells are counted once in the final scenario.
                candidate |= path_mask
                bridges.append(HiddenGasBridge(
                    source_component=source, target_component=target,
                    nearest_gap_m=gap, route_length_m=length,
                    route_cost_m=route_cost,
                    source_persistence_h=source_age,
                    target_persistence_h=target_age,
                    corridor_pore_volume_m3=float(np.sum(path_mask * aqueous * pore,
                                                         dtype=np.float64)),
                ))

        corridor_hidden = candidate * aqueous * self.settings.hidden_depth_fraction
        halo_candidate = np.zeros(gas.shape, dtype=bool)
        halo_raw_score = np.zeros(gas.shape, dtype=float)
        if self.settings.persistent_halo_enabled and eligible:
            eligible_visible = visible & np.isin(labels, eligible)
            halo_candidate, halo_raw_score = _persistent_halo(
                gas=gas, aqueous=aqueous, pore=pore,
                eligible_visible=eligible_visible, persistence_h=persistence,
                barrier=barrier, facies=facies,
                spacing=self.cell_size_m, settings=self.settings,
            )
        # Exponent below one is an explicitly declared stronger sensitivity
        # hypothesis. It changes occupancy weight, never candidate reach.
        halo_score = np.power(halo_raw_score, self.settings.halo_support_exponent)
        halo_hidden = (halo_candidate * aqueous * self.settings.halo_max_depth_fraction
                       * halo_score)
        # The sensitivity hypotheses compete for the same remaining aqueous
        # pore space; their occupancies are not additive.
        hidden = np.maximum(corridor_hidden, halo_hidden)
        total = observed_occupancy + hidden
        if np.any(total > 1 + 1e-6):
            raise AssertionError("Hidden-gas scenario exceeded physical pore capacity")
        visible_mass = hidden_mass = mass_conflict = None
        if gas_density_g_l is not None:
            density = np.asarray(gas_density_g_l, dtype=np.float64)
            if density.ndim == 0:
                density = np.full(gas.shape, float(density))
            if (density.shape != gas.shape or not np.all(np.isfinite(density))
                    or np.any(density < 0)
                    or np.any((total > 0) & (density <= 0))):
                raise ValueError(
                    "gas_density_g_l must be finite, nonnegative and positive on gas occupancy"
                )
            visible_mass = float(np.sum(observed_occupancy * pore * density,
                                        dtype=np.float64) * 1000)
            hidden_mass = float(np.sum(hidden * pore * density,
                                       dtype=np.float64) * 1000)
            mass_conflict = max(0.0, visible_mass - injected_mass_g)
            # Free gas cannot exceed the cumulative injected CO2 dose. This
            # only scales the *hypothesis*, never the observed 2D gas field.
            if hidden_mass > 0:
                scale = min(1.0, max(0.0, injected_mass_g - visible_mass) / hidden_mass)
                hidden *= scale
                corridor_hidden *= scale
                halo_hidden *= scale
                hidden_mass *= scale
                total = observed_occupancy + hidden
        visible_volume = float(np.sum(observed_occupancy * pore, dtype=np.float64))
        hidden_volume = float(np.sum(hidden * pore, dtype=np.float64))
        capacity = float(np.sum((candidate | halo_candidate) * aqueous * pore,
                                dtype=np.float64))
        result = HiddenGasDepthFrame(
            visible_gas_mask=visible.copy(),
            unsupported_visible_gas_mask=unsupported.copy(),
            candidate_mask=candidate.copy(),
            halo_candidate_mask=halo_candidate.copy(),
            halo_raw_support_score=halo_raw_score.astype(np.float32),
            halo_support_score=halo_score.astype(np.float32),
            observed_gas_occupancy=observed_occupancy.astype(np.float32),
            visible_gas_saturation=np.where(visible, saturation, 0).astype(np.float32),
            hidden_occupancy_scenario=hidden.astype(np.float32),
            corridor_hidden_occupancy_scenario=corridor_hidden.astype(np.float32),
            halo_hidden_occupancy_scenario=halo_hidden.astype(np.float32),
            total_gas_occupancy_scenario=total.astype(np.float32),
            residual_affected_aqueous_occupancy=(initial_aqueous - hidden).astype(np.float32),
            persistence_h=persistence.astype(np.float32),
            visible_gas_consecutive_observations=observations.copy(),
            visible_gas_volume_m3=visible_volume,
            hidden_gas_volume_scenario_m3=hidden_volume,
            total_gas_volume_scenario_m3=visible_volume + hidden_volume,
            maximum_hidden_capacity_m3=capacity,
            visible_gas_mass_g=visible_mass,
            hidden_gas_mass_scenario_g=hidden_mass,
            visible_mass_over_injected_g=mass_conflict,
            bridges=tuple(bridges), eligible_component_count=len(eligible),
            ignored_component_count=ignored, elapsed_h=elapsed_h,
            injected_mass_g=injected_mass_g, method_version=VERSION,
        )
        self._previous_gas = trackable_visible.copy()
        self._persistence_h = persistence
        self._visible_gas_consecutive_observations = observations.copy()
        self._elapsed_h = elapsed_h
        return result
