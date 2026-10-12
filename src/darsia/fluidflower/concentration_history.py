"""Bounded gas-history ablation of the existing joint concentration model.

This experimental module does not change the production concentration route.
Prepared history masses use ``1000 * sum(pore_volume_m3 * density_g_l * Sg)``
for gas and ``1000 * sum(pore_volume_m3 * solubility_g_l * (1-Sg))`` for
residual liquid, on the original fixed gas support. No gas history is inferred
from injection mass here. Optical histograms and the existing volume dilution
are used unchanged; fixing beta to zero removes dilution from the fit.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from darsia.fluidflower import concentration_joint as joint


VERSION = "joint-chromatic-od-history-experiment-v1"
_NAMES = ("amplitude", "gamma", "gas_saturation", "beta")
_BOUNDS = ((.05, 1.), (.4, 3.), joint.DEFAULT_GAS_BOUNDS, (0., 2.))
_STARTS = ((.7, 1., .75, .7), (.9, 2., .75, 1.), (.4, .6, .5, .3))


def _history_terms(summary):
    """Validate the fixed phase integrals without altering their support/mass."""
    try:
        gas = float(summary["gas_history_mass_g"])
        residual = float(summary["residual_history_mass_g"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("History mode requires prepared gas and residual history masses") from error
    capacities = (summary["gas_capacity_g"], summary["gas_aqueous_capacity_g"])
    if (not np.isfinite([gas, residual]).all() or min(gas, residual) < 0
            or any(mass > capacity * (1 + 1e-7)
                   for mass, capacity in zip((gas, residual), capacities))):
        raise ValueError("History masses must be finite, nonnegative and within original phase capacities")
    return gas, residual


def evaluate_summary(summary, parameters, anchor, *, gas_mode="free"):
    """Evaluate fixed parameters, preserving joint histogram numerical bounds.

    History masses are already in grams and remain outside the optical power
    response and dilution factor, just as scalar gas/residual masses do.
    """
    if gas_mode not in ("free", "fixed", "history"):
        raise ValueError("gas_mode must be free, fixed or history")
    if gas_mode != "history":
        return joint.evaluate_joint_summary(summary, parameters, anchor)
    joint._validate_summary(summary)
    gas, residual = _history_terms(summary)
    p = asdict(parameters) if is_dataclass(parameters) else dict(parameters)
    if p.get("gas_saturation") is not None:
        raise ValueError("History parameters must report gas_saturation=None")
    # Evaluate precisely the same optical response and numerical interval.
    record = joint.evaluate_joint_summary(
        dict(summary, gas_aqueous_capacity_g=residual, gas_capacity_g=0.),
        dict(p, gas_saturation=0.), anchor)
    record["gas_mass_g"] = gas
    for key in ("total_mass_g", "total_mass_lower_g", "total_mass_upper_g",
                "numerical_lower_g", "numerical_upper_g"):
        record[key] += gas
    return record


def predict_summaries(summaries_by_run, parameters_by_run, anchors_by_run, *, gas_mode="free"):
    """Predict a whole held-out run using supplied parameters and runtime anchor.

    This performs no calibration, endpoint selection or target-driven update.
    A target run can use the source run's shared endpoint and optical parameters
    with its own injection-end anchor and prepared gas-history integrals.
    """
    predictions = {}
    for run, rows in summaries_by_run.items():
        if any(row["run"] != run for row in rows):
            raise ValueError("Summary identity differs from experiment")
        predictions[run] = [evaluate_summary(row, parameters_by_run[run],
            anchors_by_run[run], gas_mode=gas_mode) for row in rows]
    return predictions


def _prepare(summaries_by_run, scope, training_indices_by_run, gas_mode):
    """Reuse the joint training and runtime-anchor conventions verbatim."""
    if (scope not in ("individual", "shared") or not summaries_by_run
            or (scope == "individual" and len(summaries_by_run) != 1)
            or (scope == "shared" and len(summaries_by_run) < 2)):
        raise ValueError("Fit scope must match the number of experiments")
    selected, endpoints, anchors, compiled = {}, {}, {}, {}
    for run, rows in summaries_by_run.items():
        for row in rows:
            joint._validate_summary(row)
            if row["run"] != run:
                raise ValueError("Summary identity differs from experiment")
            if gas_mode == "history":
                _history_terms(row)
        chosen = list(range(len(rows))) if training_indices_by_run is None else list(training_indices_by_run.get(run, ()))
        if (len(set(chosen)) != len(chosen)
                or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                       or i < 0 or i >= len(rows) for i in chosen)):
            raise ValueError("Training indices must be unique and in range")
        chosen = [i for i in chosen if rows[i]["nominal_injected_mass_g"] > 0]
        if len(chosen) < 4:
            raise ValueError(f"{run} needs at least four positive training observations")
        selected[run] = chosen
        endpoints[run] = joint.endpoint_from_summaries(rows, chosen)
        anchors[run] = joint.anchor_from_summaries(rows)
        nodes = np.asarray(rows[0]["signal_values"], float)
        if any(not np.array_equal(row["signal_values"], nodes) for row in rows):
            raise ValueError("One experiment must use one histogram grid")
        observations = [rows[i] for i in chosen]
        times = np.array([row["elapsed_h"] for row in observations])
        ratio = np.maximum(np.array([row["affected_pore_volume_m3"] for row in observations])
                           / anchors[run]["affected_pore_volume_m3"], 1.)
        log_ratio = np.where(times > anchors[run]["elapsed_h"], np.log(ratio), 0.)
        target = np.array([row["nominal_injected_mass_g"] for row in observations])
        residual = np.array([row["gas_aqueous_capacity_g"] for row in observations])
        gas = np.array([row["gas_capacity_g"] for row in observations])
        if gas_mode == "history":
            # In the shared objective, equal terms cancel the unused scalar Sg:
            # (1-Sg)*history_total + Sg*history_total = history_total.
            residual = gas = np.array([sum(_history_terms(row)) for row in observations])
        compiled[run] = (nodes / endpoints[run],
            np.array([row["aqueous_capacity_g"] for row in observations]),
            residual, gas, log_ratio, target,
            np.maximum(target, .05 * max(row["nominal_injected_mass_g"] for row in rows)))
    if scope == "shared":
        pooled_nodes, pooled_weights = [], []
        for run, rows in summaries_by_run.items():
            nodes = np.asarray(rows[0]["signal_values"], float)[1:]
            weights = np.sum([np.asarray(rows[i]["aqueous_capacity_g"], float)[1:]
                              for i in selected[run]], axis=0)
            pooled_nodes.append(nodes)
            pooled_weights.append(weights / weights.sum())
        nodes, weights = np.concatenate(pooled_nodes), np.concatenate(pooled_weights)
        order = np.argsort(nodes, kind="stable")
        index = min(len(nodes) - 1, int(np.searchsorted(np.cumsum(weights[order]), .99 * weights.sum())))
        endpoint = max(.02, float(nodes[order[index]]))
        endpoints = {run: endpoint for run in summaries_by_run}
    # This helper uses the original grids and the final endpoint mapping, pools
    # matching grids and removes only exactly zero-capacity response columns.
    objective, efficiency = joint._compile_broader_objective(compiled, summaries_by_run, endpoints)
    return selected, endpoints, anchors, objective, efficiency


def _search(objective, fixed, settings):
    """Optimize only genuinely free coordinates; record every bounded start."""
    names = [name for name in _NAMES if name not in fixed]
    positions = [_NAMES.index(name) for name in names]
    bounds = [_BOUNDS[index] for index in positions]

    def expand(x):
        parameters = dict(fixed, **dict(zip(names, map(float, x))))
        return [parameters[name] for name in _NAMES]

    def reduced_objective(x):
        return objective(expand(x))

    starts = [tuple(start[index] for index in positions)
              for start in joint._broader_starts(_STARTS, _BOUNDS, settings.start_count)]
    options = {name: getattr(settings, name) for name in ("xtol", "ftol", "maxiter", "maxfev")}
    records, fits = [], []
    for index, start in enumerate(starts):
        initial = reduced_objective(start)
        fit = minimize(reduced_objective, start, method="Powell", bounds=bounds, options=options)
        x = np.asarray(fit.x, float)
        feasible = (x.shape == (len(names),) and np.isfinite(x).all()
                    and all(lo <= value <= hi for value, (lo, hi) in zip(x, bounds)))
        value = reduced_objective(x) if feasible else float("nan")
        valid = bool(feasible and np.isfinite(value))
        records.append(dict(start_index=index, initial_parameters=dict(zip(names, map(float, start))),
            initial_objective=float(initial), parameters=dict(zip(names, map(float, x))) if feasible else None,
            objective=float(value) if valid else None,
            reported_objective=float(fit.fun) if np.isfinite(fit.fun) else None,
            success=bool(fit.success) and valid, feasible_finite=valid,
            status=int(fit.status), message=str(fit.message), evaluations=int(fit.nfev), iterations=int(fit.nit)))
        fits.append(fit)
    eligible = [record for record in records if record["feasible_finite"]]
    if not eligible:
        raise RuntimeError("No optimizer start produced a finite solution within the physical bounds")
    selected = min(eligible, key=lambda record: record["objective"])
    best = fits[selected["start_index"]]
    best.fun = selected["objective"]
    tolerance = max(1e-6, abs(best.fun) * 1e-3)
    near = [record for record in eligible if record["objective"] - best.fun <= tolerance]
    converged = [record for record in eligible if record["success"]]
    spans = {name: float((max(record["parameters"][name] for record in near)
                         - min(record["parameters"][name] for record in near)) / (hi - lo))
             for name, (lo, hi) in zip(names, bounds)}
    bound_diagnostics = {name: dict(value=float(value), lower=lo, upper=hi,
        lower_distance_fraction=float((value - lo) / (hi - lo)),
        upper_distance_fraction=float((hi - value) / (hi - lo)),
        active=bool(min(value - lo, hi - value) <= 1e-3 * (hi - lo)))
        for name, value, (lo, hi) in zip(names, best.x, bounds)}
    diagnostics = dict(settings=asdict(settings),
        start_design="three legacy starts plus unscrambled Halton bases 2,3,5,7; fixed coordinates removed",
        fitted_parameter_names=names, fitted_dimension_count=len(names),
        starts=records, selected_start_index=selected["start_index"],
        feasible_finite_starts=len(eligible), converged_starts=len(converged),
        selected_converged=bool(selected["success"]),
        best_converged_objective=min(record["objective"] for record in converged) if converged else None,
        objective_spread=float(max(record["objective"] for record in eligible) - best.fun),
        near_best_objective_tolerance=tolerance, near_best_starts=len(near),
        near_best_converged_starts=sum(record["success"] for record in near),
        near_best_parameter_span_fraction=spans, bounds=bound_diagnostics,
        solver_evaluation_budget=settings.start_count * settings.maxfev,
        diagnostic_evaluations=settings.start_count + len(eligible),
        convergence_is_global_optimality_proof=False)
    warnings = []
    if not selected["success"]:
        warnings.append("Best optimizer solution did not converge; treat this calibration as provisional")
    if sum(record["success"] for record in near) < 2:
        warnings.append("Best objective lacks agreement from two converged starts; global search agreement is unresolved")
    uncertain = [name for name, span in spans.items() if span > .1]
    if uncertain:
        warnings.append("Near-equal objective values have different parameter estimates: " + ", ".join(uncertain))
    return dict(zip(_NAMES, expand(best.x))), best, fits, diagnostics, warnings


def fit_summaries(summaries_by_run, *, scope="shared", gas_mode="free",
                  gas_saturation=None, beta_fixed=None, training_indices_by_run=None,
                  optimization_settings=None):
    """Fit the same optical law with free scalar, fixed scalar, or history gas.

    The default is the deterministic 12-start broader joint search with bounds
    A=[.05,1], gamma=[.4,3], free Sg=[.5,1], beta=[0,2]. Fixed Sg may be any
    physical value in [0,1]. ``beta_fixed=0`` disables the existing volume
    dilution. Fixed coordinates never enter the optimizer vector. Individual
    scope accepts one run; shared scope accepts at least two, weighted equally.
    """
    if gas_mode not in ("free", "fixed", "history"):
        raise ValueError("gas_mode must be free, fixed or history")
    if gas_mode == "fixed":
        if (gas_saturation is None or isinstance(gas_saturation, (bool, np.bool_))
                or not np.isscalar(gas_saturation) or not np.isfinite(gas_saturation)
                or not 0 <= gas_saturation <= 1):
            raise ValueError("Fixed gas_saturation must be a finite scalar in [0,1]")
    elif gas_saturation is not None:
        raise ValueError("gas_saturation is supplied only for fixed gas mode")
    if beta_fixed is not None and (isinstance(beta_fixed, (bool, np.bool_))
            or not np.isscalar(beta_fixed) or not np.isfinite(beta_fixed) or not 0 <= beta_fixed <= 2):
        raise ValueError("beta_fixed must be a finite scalar in [0,2]")
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
    selected, endpoints, anchors, objective, efficiency = _prepare(
        summaries_by_run, scope, training_indices_by_run, gas_mode)
    fixed = {}
    if gas_mode != "free":
        fixed["gas_saturation"] = float(gas_saturation) if gas_mode == "fixed" else .5
    if beta_fixed is not None:
        fixed["beta"] = float(beta_fixed)
    shared, best, fits, search, search_warnings = _search(objective, fixed, settings)
    if gas_mode == "history":
        shared["gas_saturation"] = None
    search["histogram_efficiency"] = efficiency
    search["fixed_parameters"] = {name: shared[name] for name in fixed}
    parameters = {run: dict(endpoint=endpoints[run], **shared) for run in summaries_by_run}
    predictions = predict_summaries(summaries_by_run, parameters, anchors, gas_mode=gas_mode)
    active = [name for name, bounds in search["bounds"].items() if bounds["active"]]
    warnings = ["Mass consistency does not independently validate local concentrations or the free-gas split",
                *search_warnings]
    if gas_mode == "history":
        warnings.append("Prepared spatial gas history is an imposed scenario, not an independently validated saturation measurement")
    if active:
        warnings.append("Parameters close to bounds: " + ", ".join(active))
    if any(row.get("signal_tail_capacity_g", 0) > 0 for rows in summaries_by_run.values() for row in rows):
        warnings.append("Positive optical histogram tail is clipped; inspect endpoint and numerical error bounds")
    return dict(format=VERSION, scope=scope, gas_mode=gas_mode, beta_fixed=beta_fixed,
        shared_parameters=shared, parameters_by_run=parameters,
        endpoints_by_run=endpoints, anchors_by_run=anchors, predictions=predictions,
        training_indices_by_run=selected, metrics=joint.joint_mass_metrics(predictions, selected), warnings=warnings,
        diagnostics=dict(objective=float(best.fun), optimizer="deterministic bounded multistart Powell",
            optimizer_success=bool(best.success), optimizer_message=str(best.message),
            evaluations=sum(int(fit.nfev) for fit in fits), active_bounds=active,
            parameter_bounds={name: list(_BOUNDS[_NAMES.index(name)]) for name in search["fitted_parameter_names"]},
            gas_bounds=list(joint.DEFAULT_GAS_BOUNDS) if gas_mode == "free" else None,
            fixed_parameters=search["fixed_parameters"],
            endpoint_policy=("positive training aqueous-capacity weighted 99th percentile, pooled with equal experiment weight"
                             if scope == "shared" else "positive training aqueous-capacity weighted 99th percentile"),
            anchor_policy="observed injection-end anchor from each run's runtime injection schedule",
            shared_scope="endpoint and fitted response parameters shared; observed reference and anchor per experiment",
            loss="equal-run mean of NMAE + 0.25 NRMSE; normalization=max(current injected mass, 5% final dose)",
            mass_projection=False, contact_or_temporal_penalty=False, optimization=search))
