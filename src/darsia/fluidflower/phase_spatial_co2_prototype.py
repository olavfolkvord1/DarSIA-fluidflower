"""Unpublished spatial CO2 prototype; never selected by runtime dispatch.

The convex objective combines squared probability fidelity with isotropic
total variation weighted by multiscale *baseline-relative change*. Its variable
is the whole CO2 field, without a band around an inherited classification.
Outputs are experimental classification fields, not calibrated concentrations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class SpatialCo2Settings:
    regularization_length_m: float = .001
    working_spacing_m: float = .001
    guide_fine_scale_m: float = .0005
    guide_coarse_scale_m: float = .0015
    guide_contrast: float = .025
    minimum_edge_weight: float = .05
    coherent_evidence_scale_m: float = .0015
    coherent_positive_probability: float = .9
    quiet_negative_probability: float = .1
    quiet_change_magnitude: float = .02
    prior_weight: float = .15
    causal_weight: float = .1
    causal_minimum_current_probability: float = .35
    max_iterations: int = 100
    tolerance: float = .0001
    cpu_threads: int = 4

    def __post_init__(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError('Spatial settings must be finite')
        if not (0 < self.regularization_length_m <= .01 and self.working_spacing_m > 0
                and 0 < self.guide_fine_scale_m <= self.guide_coarse_scale_m
                and self.guide_contrast > 0 and 0 < self.minimum_edge_weight <= 1
                and self.coherent_evidence_scale_m > 0):
            raise ValueError('Spatial scales and guide weights must be positive and bounded')
        if not (.5 < self.coherent_positive_probability < 1
                and 0 <= self.quiet_negative_probability < .5
                and self.quiet_change_magnitude > 0
                and 0 <= self.prior_weight <= .5 and 0 <= self.causal_weight <= .25
                and 0 < self.causal_minimum_current_probability < .5):
            raise ValueError('Evidence and soft-prior settings are invalid')
        if (type(self.max_iterations) is not int or not 1 <= self.max_iterations <= 500
                or type(self.cpu_threads) is not int or self.cpu_threads < 1
                or self.tolerance <= 0):
            raise ValueError('Iteration and CPU limits must be explicit positive values')


def _probability(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f'{name} must be a finite probability field on the native grid')
    return value


def _resize_average(value, domain, coverage, shape):
    field = np.asarray(value, np.float32)
    if not domain.all():
        field = field * domain
    reduced = cv2.resize(field, shape[::-1], interpolation=cv2.INTER_AREA)
    return np.divide(reduced, coverage, out=np.zeros_like(reduced), where=coverage > 1e-7)


def _blur(field, valid, sx, sy):
    denominator = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sx, sigmaY=sy)
    numerator = cv2.GaussianBlur(field * valid, (0, 0), sx, sigmaY=sy)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-7)


def _rgb_scale(value, shape, name):
    value = np.asarray(value)
    if value.shape != (*shape, 3):
        raise ValueError(f'{name} must contain three registered RGB channels')
    if value.dtype == np.uint8:
        return value, np.float32(1/255)
    if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f'{name} requires uint8 or finite RGB in [0,1]')
    return value, np.float32(1)


@njit(parallel=True, cache=True)
def _dual_step(ubar, horizontal, vertical, px, py, sigma, radius, row_changes):
    height, width = ubar.shape
    for row in prange(height):
        largest = 0.
        for col in range(width):
            x = px[row, col]
            y = py[row, col]
            if col+1 < width:
                x += sigma*horizontal[row, col]*(ubar[row, col+1]-ubar[row, col])
            if row+1 < height:
                y += sigma*vertical[row, col]*(ubar[row+1, col]-ubar[row, col])
            norm = max(1., np.sqrt(x*x+y*y)/radius)
            new_x, new_y = x/norm, y/norm
            largest = max(largest, abs(new_x-px[row, col]), abs(new_y-py[row, col]))
            px[row, col], py[row, col] = new_x, new_y
        row_changes[row] = largest


@njit(parallel=True, cache=True)
def _primal_step(u, ubar, horizontal, vertical, px, py, target, fidelity,
                 lower, upper, tau, row_changes):
    height, width = u.shape
    for row in prange(height):
        largest = 0.
        for col in range(width):
            adjoint = -horizontal[row, col]*px[row, col]-vertical[row, col]*py[row, col]
            if col:
                adjoint += horizontal[row, col-1]*px[row, col-1]
            if row:
                adjoint += vertical[row-1, col]*py[row-1, col]
            old = u[row, col]
            new = (old-tau*adjoint+tau*fidelity[row, col]*target[row, col])/(1+tau*fidelity[row, col])
            new = max(lower[row, col], min(upper[row, col], new))
            u[row, col] = new
            ubar[row, col] = 2*new-old
            largest = max(largest, abs(new-old))
        row_changes[row] = largest


@njit(parallel=True, cache=True)
def _energy_rows(u, horizontal, vertical, px, py, target, fidelity,
                 lower, upper, radius, rows):
    """Evaluate primal energy and a lower bound from a feasible dual field."""
    height, width = u.shape
    for row in prange(height):
        primal, dual = 0., 0.
        for col in range(width):
            value = float(u[row, col])
            horizontal_delta = 0.
            vertical_delta = 0.
            if col+1 < width:
                horizontal_delta = horizontal[row, col]*(float(u[row, col+1])-value)
            if row+1 < height:
                vertical_delta = vertical[row, col]*(float(u[row+1, col])-value)
            weight = float(fidelity[row, col])
            center = float(target[row, col])
            primal += .5*weight*(value-center)**2 + radius*np.sqrt(horizontal_delta**2+vertical_delta**2)
            adjoint = -horizontal[row, col]*px[row, col]-vertical[row, col]*py[row, col]
            if col:
                adjoint += horizontal[row, col-1]*px[row, col-1]
            if row:
                adjoint += vertical[row-1, col]*py[row-1, col]
            optimum = max(float(lower[row, col]), min(float(upper[row, col]), center-adjoint/weight))
            dual += .5*weight*(optimum-center)**2+adjoint*optimum
        rows[row, 0], rows[row, 1] = primal, dual


def _certificate(u, horizontal, vertical, px, py, target, fidelity, lower, upper, radius, valid):
    # Float32 projection can exceed the dual ball by rounding. Project a copy
    # inward in float64 before claiming a lower bound on the convex objective.
    safe_x, safe_y = px.astype(np.float64), py.astype(np.float64)
    norm = np.hypot(safe_x, safe_y)
    projection = np.maximum(1., norm/(float(radius)*(1-1e-12)))
    safe_x /= projection
    safe_y /= projection
    rows = np.zeros((u.shape[0], 2), np.float64)
    _energy_rows(u, horizontal, vertical, safe_x, safe_y, target, fidelity,
                 lower, upper, float(radius), rows)
    primal, dual = rows.sum(axis=0)
    gap = float(primal-dual)
    # A tiny negative gap can arise only from summation roundoff.
    if gap < -1e-8*max(1., abs(primal), abs(dual)):
        raise RuntimeError('Spatial primal-dual certificate is inconsistent')
    gap = max(0., gap)
    cells = max(1, int(valid.sum()))
    return dict(primal_energy=float(primal), dual_lower_bound=float(dual),
                primal_dual_gap=gap, gap_per_valid_working_cell=gap/cells,
                relative_gap=gap/max(abs(float(primal)), abs(float(dual)), 1e-12),
                probability_rms_error_upper_bound=float(np.sqrt(2*gap/cells)),
                units='working-cell objective; certificate precedes native interpolation and frozen-gas restoration')


def infer_spatial_co2(co2_probability, current_rgb=None, baseline_rgb=None, domain=None,
                     *, spacing_m, gas_mask, guide=None, prior_probability=None,
                     causal_support=None, settings=None):
    """Infer a whole-domain continuous field on an explicit physical grid.

    ``guide`` may replace RGB inputs with HxWxC baseline-relative changes in
    normalized RGB-equivalent units; this function performs multiscale filtering.
    It must never contain raw image intensity or material/layer boundaries.
    ``co2_probability`` may be a documented morphology observation instead of
    calibrated optical probability; the solver never assigns calibration.
    ``prior_probability`` is an optional soft secondary observation, not truth.
    Causal support cannot add evidence without current probability and change.
    Gas labels are restored exactly at native resolution after interpolation.
    """
    started = time.perf_counter()
    settings = settings or SpatialCo2Settings()
    source = np.asarray(co2_probability)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError('Spatial inference requires a two-dimensional native field')
    shape = source.shape
    source = _probability(source, shape, 'CO2')
    domain = np.asarray(domain)
    gas_mask = np.asarray(gas_mask)
    if domain.shape != shape or domain.dtype != bool or gas_mask.shape != shape or gas_mask.dtype != bool:
        raise ValueError('Domain and frozen gas require native boolean masks')
    if (gas_mask & ~domain).any():
        raise ValueError('Frozen gas must lie inside the valid domain')
    dy, dx = (float(value) for value in spacing_m)
    if not np.isfinite([dy, dx]).all() or min(dy, dx) <= 0:
        raise ValueError('Positive physical pixel spacings are required in (dy,dx) order')
    work_shape = tuple(max(2, min(size, int(round(size*spacing/settings.working_spacing_m))))
                       for size, spacing in zip(shape, (dy, dx)))
    work_dy, work_dx = shape[0]*dy/work_shape[0], shape[1]*dx/work_shape[1]
    minimum_spacing = min(work_dy, work_dx)
    previous_cv_threads, previous_numba_threads = cv2.getNumThreads(), get_num_threads()
    cv2.setNumThreads(settings.cpu_threads)
    set_num_threads(min(settings.cpu_threads, previous_numba_threads))
    try:
        with threadpool_limits(limits=settings.cpu_threads):
            coverage = cv2.resize(domain.astype(np.float32), work_shape[::-1], interpolation=cv2.INTER_AREA)
            valid = coverage > 1e-7
            # Partial boundary cells retain local fidelity, but cannot couple
            # evidence through an unresolved excluded strip at the working scale.
            connected = coverage >= 1-1e-6
            q = _resize_average(source, domain, coverage, work_shape)
            if guide is None:
                current, current_scale = _rgb_scale(current_rgb, shape, 'Current RGB')
                baseline, baseline_scale = _rgb_scale(baseline_rgb, shape, 'Baseline RGB')
                changes = []
                for channel in range(3):
                    delta = current[..., channel].astype(np.float32)*current_scale
                    delta -= baseline[..., channel].astype(np.float32)*baseline_scale
                    changes.append(_resize_average(delta, domain, coverage, work_shape))
                del delta
            else:
                guide = np.asarray(guide)
                if guide.ndim == 2:
                    guide = guide[..., None]
                if (guide.ndim != 3 or guide.shape[:2] != shape or guide.shape[2] == 0
                        or not np.isfinite(guide).all()):
                    raise ValueError('Guide must be a finite registered baseline-relative field')
                changes = [_resize_average(guide[..., channel], domain, coverage, work_shape)
                           for channel in range(guide.shape[2])]
            magnitude = np.sqrt(sum(change*change for change in changes))
            horizontal_distance = np.zeros(work_shape, np.float32)
            vertical_distance = np.zeros(work_shape, np.float32)
            for scale, weight in ((settings.guide_fine_scale_m, 1.), (settings.guide_coarse_scale_m, .5)):
                for change in changes:
                    smooth = _blur(change, valid, scale/work_dx, scale/work_dy)
                    horizontal_distance[:, :-1] += np.float32(weight/len(changes))*np.diff(smooth, axis=1)**2
                    vertical_distance[:-1, :] += np.float32(weight/len(changes))*np.diff(smooth, axis=0)**2
            def edge_weight(distance):
                return (settings.minimum_edge_weight+(1-settings.minimum_edge_weight)
                    *np.exp(-distance/(2*settings.guide_contrast**2))).astype(np.float32)
            horizontal = edge_weight(horizontal_distance)*np.float32(minimum_spacing/work_dx)
            vertical = edge_weight(vertical_distance)*np.float32(minimum_spacing/work_dy)
            horizontal[:, :-1] *= connected[:, :-1] & connected[:, 1:]
            vertical[:-1, :] *= connected[:-1, :] & connected[1:, :]
            horizontal[:, -1] = 0
            vertical[-1, :] = 0
            target = q.copy()
            fidelity = np.ones(work_shape, np.float32)
            if prior_probability is not None:
                prior = _probability(prior_probability, shape, 'Soft prior')
                target += np.float32(settings.prior_weight)*_resize_average(prior, domain, coverage, work_shape)
                fidelity += np.float32(settings.prior_weight)
            causal_used = np.zeros(work_shape, bool)
            if causal_support is not None:
                support = _probability(causal_support, shape, 'Causal support')
                pooled_support = _resize_average(support, domain, coverage, work_shape)
                causal_used = valid & (pooled_support >= .5) & (q >= settings.causal_minimum_current_probability)
                causal_used &= magnitude > settings.quiet_change_magnitude
                target[causal_used] += np.float32(settings.causal_weight)
                fidelity[causal_used] += np.float32(settings.causal_weight)
            target /= fidelity
            coherent = _blur(q, valid, settings.coherent_evidence_scale_m/work_dx,
                settings.coherent_evidence_scale_m/work_dy)
            positive = valid & (coherent >= settings.coherent_positive_probability) & (q >= settings.coherent_positive_probability)
            negative = valid & (q <= settings.quiet_negative_probability) & (magnitude <= settings.quiet_change_magnitude)
            lower = np.zeros(work_shape, np.float32)
            upper = np.ones(work_shape, np.float32)
            lower[positive] = np.float32(settings.coherent_positive_probability)
            upper[negative] = np.float32(settings.quiet_negative_probability)
            upper[~valid] = 0
            u = np.clip(target, lower, upper).astype(np.float32)
            ubar = u.copy()
            px, py = np.zeros(work_shape, np.float32), np.zeros(work_shape, np.float32)
            row_changes = np.zeros(work_shape[0], np.float32)
            dual_row_changes = np.zeros(work_shape[0], np.float32)
            # ||K||^2 <= 8 because guide weights and spacing ratios are <= 1.
            tau = sigma = np.float32(.3)
            radius = np.float32(settings.regularization_length_m/minimum_spacing)
            for iteration in range(1, settings.max_iterations+1):
                _dual_step(ubar, horizontal, vertical, px, py, sigma, radius, dual_row_changes)
                _primal_step(u, ubar, horizontal, vertical, px, py, target, fidelity,
                    lower, upper, tau, row_changes)
                maximum_change = float(row_changes.max())
                maximum_dual_change = float(dual_row_changes.max())
                if iteration >= 20 and max(maximum_change, maximum_dual_change) <= settings.tolerance:
                    break
            certificate = _certificate(u, horizontal, vertical, px, py, target, fidelity,
                                       lower, upper, radius, valid)
            converged = (max(maximum_change, maximum_dual_change) <= settings.tolerance
                         and certificate['gap_per_valid_working_cell'] <= settings.tolerance)
            probability = cv2.resize(u, shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32)
            np.clip(probability, 0, 1, out=probability)
            probability[~domain] = 0
            probability[gas_mask] = 1
            mask = domain & (probability >= .5)
            return dict(co2_probability=probability, co2_mask=mask, gas_mask=gas_mask.copy(),
                diagnostics=dict(method='experimental-whole-domain-guide-weighted-convex-tv',
                    settings=asdict(settings), native_shape=list(shape), working_shape=list(work_shape),
                    native_spacing_m=[dy,dx], working_spacing_m=[work_dy,work_dx],
                    regularization_length_working_pixels=float(radius), iterations=iteration,
                    converged=converged, maximum_iteration_change=maximum_change,
                    maximum_dual_iteration_change=maximum_dual_change, convergence_certificate=certificate,
                    convergence_criterion='both iterate changes and primal-dual gap per valid cell <= tolerance',
                    prior_used=prior_probability is not None, causal_working_cells=int(causal_used.sum()),
                    coherent_positive_working_cells=int(positive.sum()), quiet_negative_working_cells=int(negative.sum()),
                    guide_basis='multiscale current-minus-baseline change; no raw intensity/material edges',
                    native_probability_restoration='bilinear, then threshold; gas restored exactly',
                    gas_labels_exact=True, history_updated=False, seconds=time.perf_counter()-started))
    finally:
        cv2.setNumThreads(previous_cv_threads)
        set_num_threads(previous_numba_threads)
