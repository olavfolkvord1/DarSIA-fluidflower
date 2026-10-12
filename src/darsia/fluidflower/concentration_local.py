"""Isolated local-versus-global dilution experiment on fixed phase support.

Local exposure L is prepared causally outside this module, without mass targets.
The optical law is A * clip(signal/endpoint, 0, 1)**gamma * exp(-beta*L).
Local attenuation replaces the existing global volume factor. Gas and saturated
residual-water masses are fixed prepared inputs and are never attenuated here.

A sparse signal-by-exposure histogram retains the original signal marginal.
Capacity is deposited linearly on adjacent exposure nodes, giving second-order
exponential interpolation error instead of first-order exposure rounding error.
The reported numerical bounds cover both histogram axes, not physical uncertainty.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Mapping

import numpy as np
from scipy.sparse import csr_matrix

from darsia.fluidflower import concentration_history as history
from darsia.fluidflower import concentration_joint as joint


VERSION = "joint-chromatic-local-dilution-experiment-v1"
HISTOGRAM_VERSION = "signal-exposure-linear-histogram-v1"


def _mode(mode):
    if mode not in ("global", "local", "none"):
        raise ValueError("dilution_mode must be global, local or none")
    return mode


def _parameters(parameters):
    values = asdict(parameters) if is_dataclass(parameters) else dict(parameters)
    if values.get("gas_saturation") is not None:
        raise ValueError("Prepared-gas scenarios must report gas_saturation=None")
    return joint.JointConcentrationParameters(**dict(values, gas_saturation=0.))


def add_local_histogram(summary, signal, av, solubility, exposure, *,
                        exposure_bins=64, max_exposure=6.):
    """Copy a joint summary and append a JSON-ready sparse 2D response histogram.

    The original 8192-bin signal grid and aqueous-capacity marginal are retained
    verbatim. The supplied fields must reproduce that marginal; mismatched phase
    support or units fail instead of being rescaled. Exposure is dimensionless,
    finite and bounded by the caller's declared maximum; it is never clipped.
    """
    nodes, expected = joint._validate_summary(summary)
    if (isinstance(exposure_bins, (bool, np.bool_))
            or not isinstance(exposure_bins, (int, np.integer)) or exposure_bins < 2
            or not np.isfinite(max_exposure) or max_exposure <= 0):
        raise ValueError("Invalid local exposure histogram grid")
    signal, av, solubility = [np.asarray(value, np.float32) for value in (signal, av, solubility)]
    exposure = np.asarray(exposure, float)
    if (signal.ndim != 2 or not np.isfinite(signal).all()
            or any(value.shape != signal.shape or not np.isfinite(value).all()
                   or np.any(value < 0) for value in (av, solubility, exposure))
            or np.any(exposure > max_exposure)):
        raise ValueError("Local physical fields must be finite on one grid with 0 <= exposure <= max_exposure")
    half = float(summary.get("signal_quantization_half_width", 0.))
    signal_edges = np.linspace(0., 2 * half * (nodes.size - 1), nodes.size)
    expected_nodes = np.r_[0., .5 * (signal_edges[:-1] + signal_edges[1:])]
    if half <= 0 or not np.allclose(nodes, expected_nodes, rtol=1e-13, atol=1e-14):
        raise ValueError("Local histogram requires the original uniform joint signal grid")
    signal_index = np.where(signal > 0,
        np.minimum(np.searchsorted(signal_edges, signal, side="right"), nodes.size - 1), 0)
    capacity = av.astype(float) * solubility * 1000
    marginal = np.bincount(signal_index.ravel(), weights=capacity.ravel(), minlength=nodes.size)
    if not np.allclose(marginal, expected, rtol=2e-10, atol=1e-11 * max(1., expected.sum())):
        raise ValueError("Local histogram fields differ from the original aqueous-capacity marginal")
    if float(signal.max(initial=0)) > float(summary.get("signal_maximum", 0.)) + 1e-7:
        raise ValueError("Local histogram signal exceeds the recorded signal maximum")
    levels = np.linspace(0., float(max_exposure), int(exposure_bins) + 1)
    step = float(max_exposure / exposure_bins)
    left = np.minimum(np.floor(exposure / step).astype(np.int64), exposure_bins - 1)
    fraction = np.clip((exposure - levels[left]) / step, 0., 1.)
    key = signal_index * levels.size + left
    size = nodes.size * levels.size
    weights = np.bincount(key.ravel(), weights=(capacity * (1 - fraction)).ravel(), minlength=size)
    weights += np.bincount((key + 1).ravel(), weights=(capacity * fraction).ravel(), minlength=size)
    active = np.flatnonzero(weights)
    return dict(summary, local_histogram_format=HISTOGRAM_VERSION,
        local_exposure_values=levels.tolist(),
        local_signal_indices=(active // levels.size).tolist(),
        local_exposure_indices=(active % levels.size).tolist(),
        local_capacity_g=weights[active].tolist(),
        local_exposure_bin_width=step,
        local_exposure_interpolation_weight_g=float(np.sum(capacity * fraction * (1 - fraction))),
        local_exposure_maximum=float(exposure.max(initial=0)),
        local_capacity_at_exposure_limit_g=float(capacity[exposure >= max_exposure].sum()))


def _local_histogram(summary):
    nodes, marginal = joint._validate_summary(summary)
    try:
        levels = np.asarray(summary["local_exposure_values"], float)
        xi = np.asarray(summary["local_signal_indices"])
        li = np.asarray(summary["local_exposure_indices"])
        capacity = np.asarray(summary["local_capacity_g"], float)
        step = float(summary["local_exposure_bin_width"])
        error_weight = float(summary["local_exposure_interpolation_weight_g"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Local dilution requires a prepared local response histogram") from error
    # JSON's empty lists have float dtype; zero-capacity frames are legitimate.
    if xi.size == 0 and li.size == 0:
        xi, li = xi.astype(np.int64), li.astype(np.int64)
    if (summary.get("local_histogram_format") != HISTOGRAM_VERSION
            or levels.ndim != 1 or levels.size < 3 or levels[0] != 0
            or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0)
            or not np.isfinite([step, error_weight]).all() or step <= 0 or error_weight < 0
            or not np.allclose(np.diff(levels), step, rtol=1e-12, atol=1e-14)
            or xi.ndim != 1 or li.shape != xi.shape or capacity.shape != xi.shape
            or xi.dtype.kind not in "iu" or li.dtype.kind not in "iu"
            or np.any(xi < 0) or np.any(xi >= nodes.size)
            or np.any(li < 0) or np.any(li >= levels.size)
            or not np.isfinite(capacity).all() or np.any(capacity < 0)
            or error_weight > .25 * marginal.sum() + 1e-10):
        raise ValueError("Invalid local response histogram")
    keys = xi * levels.size + li
    if (np.any(np.diff(keys) <= 0)
            or not np.allclose(np.bincount(xi, weights=capacity, minlength=nodes.size),
                marginal, rtol=2e-10, atol=1e-11 * max(1., marginal.sum()))):
        raise ValueError("Local response histogram must preserve the original signal marginal")
    return nodes, levels, xi, li, capacity, step, error_weight


def dense_concentration(signal, solubility, parameters, exposure=None, *,
                        dilution_mode="local", global_factor=None):
    """Materialize the optical aqueous concentration in g/L, as float32.

    Global mode requires its explicitly computed runtime factor. Local and none
    reject a global factor, preventing accidental double attenuation. This helper
    neither adds the separate saturated residual-water term nor infers gas mass.
    """
    mode, p = _mode(dilution_mode), _parameters(parameters)
    if mode == "global":
        if global_factor is None:
            raise ValueError("Global dilution requires an explicit global_factor")
        return joint.concentration_field(signal, p, global_factor, solubility)
    if global_factor is not None:
        raise ValueError("Local/none dilution replaces the global factor; do not supply both")
    base = joint.concentration_field(signal, p, 1., solubility)
    if mode == "none":
        if p.beta != 0:
            raise ValueError("No-dilution parameters must have beta=0")
        return base
    values = np.asarray(exposure, float)
    if values.shape != base.shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Local exposure must be finite, nonnegative and share the optical grid")
    return (base * np.exp(-p.beta * values)).astype(np.float32)


def evaluate_summary(summary, parameters, anchor, *, dilution_mode="global"):
    mode, p = _mode(dilution_mode), _parameters(parameters)
    if mode != "local":
        if mode == "none" and p.beta != 0:
            raise ValueError("No-dilution parameters must have beta=0")
        return history.evaluate_summary(summary, parameters, anchor, gas_mode="history")
    nodes, levels, xi, li, capacity, step, error_weight = _local_histogram(summary)
    gas, residual = history._history_terms(summary)
    factor = np.exp(-p.beta * levels[li])
    response = p.amplitude * np.clip(nodes / p.endpoint, 0, 1) ** p.gamma
    optical = float(capacity @ (response[xi] * factor))
    undiluted = float(np.asarray(summary["aqueous_capacity_g"], float) @ response)
    half = float(summary.get("signal_quantization_half_width", 0.))
    lo, hi = np.maximum(nodes - half, 0), nodes + half
    lo[0] = hi[0] = 0
    hi[-1] = max(hi[-1], float(summary.get("signal_maximum", hi[-1])))
    lower_optical = float(capacity @ (p.amplitude * np.clip(lo[xi] / p.endpoint, 0, 1) ** p.gamma * factor))
    upper_optical = float(capacity @ (p.amplitude * np.clip(hi[xi] / p.endpoint, 0, 1) ** p.gamma * factor))
    # exp(-beta*L) is convex. Linear deposition gives its chord (an upper
    # approximation); interpolation error <= beta²*h²*t*(1-t)/2 because
    # max |f''| <= beta² for nonnegative L. A bounds the optical fraction.
    interpolation_error = float(.5 * p.amplitude * p.beta ** 2 * step ** 2 * error_weight)
    roundoff = float(8 * np.finfo(np.float32).eps * p.amplitude * capacity.sum())
    lower = max(0., lower_optical - interpolation_error - roundoff) + residual + gas
    upper = upper_optical + roundoff + residual + gas
    aqueous, total = optical + residual, optical + residual + gas
    return dict(run=summary["run"], elapsed_h=summary["elapsed_h"],
        nominal_injected_mass_g=summary["nominal_injected_mass_g"],
        aqueous_mass_g=aqueous, gas_mass_g=gas, total_mass_g=total,
        total_mass_lower_g=total, total_mass_upper_g=total,
        numerical_lower_g=lower, numerical_upper_g=upper,
        summary_mass_error_bound_g=max(total - lower, upper - total),
        exposure_interpolation_error_bound_g=interpolation_error,
        float32_roundoff_allowance_g=roundoff,
        residual_liquid_in_gas_g=residual, optical_aqueous_mass_g=optical,
        dilution_factor=optical / undiluted if undiluted > 0 else 1.,
        dilution_factor_kind="optical-mass-weighted local attenuation",
        quantification_complete=True, unknown_mass_upper_g=0.)


def predict_summaries(summaries_by_run, parameters_by_run, anchors_by_run, *, dilution_mode="global"):
    """Predict with source-fit parameters; no target endpoint selection or fit."""
    _mode(dilution_mode)
    result = {}
    for run, rows in summaries_by_run.items():
        if any(row["run"] != run for row in rows):
            raise ValueError("Summary identity differs from experiment")
        result[run] = [evaluate_summary(row, parameters_by_run[run], anchors_by_run[run],
            dilution_mode=dilution_mode) for row in rows]
    return result


def _compile_local(summaries_by_run, selected, endpoints):
    """Pool training grids into sparse capacity matrices with equal run losses."""
    grouped = defaultdict(list)
    original_columns = 0
    for run, rows in summaries_by_run.items():
        validated = [_local_histogram(row) for row in rows]
        nodes, levels = validated[0][:2]
        if any(not np.array_equal(item[1], levels) for item in validated):
            raise ValueError("One experiment must use one exposure histogram grid")
        normalized, node_codes = np.unique(np.clip(nodes / endpoints[run], 0, 1), return_inverse=True)
        selected_rows = [rows[i] for i in selected[run]]
        records = [validated[i] for i in selected[run]]
        keys = [node_codes[item[2]] * levels.size + item[3] for item in records]
        capacities = [item[4] for item in records]
        target = np.array([row["nominal_injected_mass_g"] for row in selected_rows])
        fixed_mass = np.array([sum(history._history_terms(row)) for row in selected_rows])
        denominator = np.maximum(target, .05 * max(row["nominal_injected_mass_g"] for row in rows))
        key = (normalized.tobytes(), levels.tobytes())
        grouped[key].append((normalized, levels, keys, capacities, fixed_mass, target, denominator))
        original_columns += nodes.size * levels.size
    groups, retained_columns, nonzeros = [], 0, 0
    for entries in grouped.values():
        normalized, levels = entries[0][:2]
        keys = [key for entry in entries for key in entry[2]]
        capacities = [capacity for entry in entries for capacity in entry[3]]
        all_keys = np.concatenate(keys)
        active = np.unique(all_keys)
        lengths = [len(entry[2]) for entry in entries]
        offsets = np.r_[0, np.cumsum(lengths)]
        row_indices = np.repeat(np.arange(sum(lengths)), [len(key) for key in keys])
        matrix = csr_matrix((np.concatenate(capacities), (row_indices, np.searchsorted(active, all_keys))),
            shape=(sum(lengths), active.size))
        groups.append((normalized, levels, active // levels.size, active % levels.size, matrix,
            np.concatenate([entry[4] for entry in entries]),
            np.concatenate([entry[5] for entry in entries]),
            np.concatenate([entry[6] for entry in entries]), offsets))
        retained_columns += active.size
        nonzeros += matrix.nnz

    def objective(x):
        amplitude, gamma, _unused_sg, beta = x
        losses = []
        for nodes, exposure, xi, li, capacity, fixed_mass, target, denominator, offsets in groups:
            optical = amplitude * nodes ** gamma
            attenuation = np.exp(-beta * exposure)
            mass = capacity @ (optical[xi] * attenuation[li]) + fixed_mass
            errors = (mass - target) / denominator
            for begin, end in zip(offsets[:-1], offsets[1:]):
                local = errors[begin:end]
                losses.append(np.mean(abs(local)) + .25 * np.sqrt(np.mean(local ** 2)))
        return float(np.mean(losses))

    return objective, dict(training_runs=len(selected), response_grid_groups=len(groups),
        original_histogram_columns=original_columns, retained_response_columns=retained_columns,
        sparse_nonzero_capacities=nonzeros, signal_marginals_rebuilt=False,
        response_above_endpoint_pooled=True, local_exposure_interpolation="linear capacity deposition")


def fit_summaries(summaries_by_run, *, scope="shared", dilution_mode="global",
                  training_indices_by_run=None, optimization_settings=None):
    """Fit A, gamma and beta with fixed prepared gas/residual masses.

    Defaults retain twelve deterministic bounded Powell starts and the existing
    budgets. Global/none delegate to the established history fitter. Local uses
    the same training-only endpoint, runtime anchors and equal-run loss; its
    exposure never receives the global volume factor. None removes beta.
    """
    mode = _mode(dilution_mode)
    if mode != "local":
        fit = history.fit_summaries(summaries_by_run, scope=scope, gas_mode="history",
            beta_fixed=0. if mode == "none" else None,
            training_indices_by_run=training_indices_by_run, optimization_settings=optimization_settings)
        fit["format"], fit["dilution_mode"] = VERSION, mode
        fit["diagnostics"]["dilution_mode"] = mode
        fit["diagnostics"]["global_factor_applied"] = mode == "global"
        return fit
    if optimization_settings is None:
        settings = joint.JointOptimizationSettings()
    elif isinstance(optimization_settings, joint.JointOptimizationSettings):
        settings = optimization_settings
    elif isinstance(optimization_settings, Mapping):
        try:
            settings = joint.JointOptimizationSettings(**optimization_settings)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid optimization_settings: " + str(error)) from error
    else:
        raise ValueError("optimization_settings must be a field mapping or JointOptimizationSettings")
    selected, endpoints, anchors, _global_objective, _efficiency = history._prepare(
        summaries_by_run, scope, training_indices_by_run, "history")
    objective, efficiency = _compile_local(summaries_by_run, selected, endpoints)
    shared, best, fits, search, search_warnings = history._search(objective, {"gas_saturation": .5}, settings)
    shared["gas_saturation"] = None
    search["histogram_efficiency"] = efficiency
    search["fixed_parameters"] = {"gas_saturation": None}
    parameters = {run: dict(endpoint=endpoints[run], **shared) for run in summaries_by_run}
    predictions = predict_summaries(summaries_by_run, parameters, anchors, dilution_mode="local")
    active = [name for name, bounds in search["bounds"].items() if bounds["active"]]
    warnings = ["Mass consistency does not independently validate local concentrations or the gas/water split",
        "Local exposure and gas history are imposed scenarios, not measured fluid-parcel ages or saturations",
        *search_warnings]
    if active:
        warnings.append("Parameters close to bounds: " + ", ".join(active))
    if any(row.get("signal_tail_capacity_g", 0) > 0 for rows in summaries_by_run.values() for row in rows):
        warnings.append("Positive optical histogram tail is clipped; inspect endpoint and numerical error bounds")
    return dict(format=VERSION, scope=scope, dilution_mode=mode, gas_mode="history", beta_fixed=None,
        shared_parameters=shared, parameters_by_run=parameters, endpoints_by_run=endpoints,
        anchors_by_run=anchors, predictions=predictions, training_indices_by_run=selected,
        metrics=joint.joint_mass_metrics(predictions, selected), warnings=warnings,
        diagnostics=dict(objective=float(best.fun), optimizer="deterministic bounded multistart Powell",
            optimizer_success=bool(best.success), optimizer_message=str(best.message),
            evaluations=sum(int(fit.nfev) for fit in fits), active_bounds=active,
            parameter_bounds={name: list(history._BOUNDS[history._NAMES.index(name)])
                              for name in search["fitted_parameter_names"]},
            fixed_parameters={"gas_saturation": None}, gas_bounds=None,
            endpoint_policy=("positive training aqueous-capacity weighted 99th percentile, pooled with equal experiment weight"
                             if scope == "shared" else "positive training aqueous-capacity weighted 99th percentile"),
            anchor_policy="runtime anchor retained for comparability; not used by local attenuation",
            loss="equal-run mean of NMAE + 0.25 NRMSE; normalization=max(current injected mass, 5% final dose)",
            dilution_mode=mode, global_factor_applied=False, local_mass_normalization=False,
            mass_projection=False, contact_or_temporal_penalty=False, optimization=search))
