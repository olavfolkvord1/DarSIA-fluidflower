"""Time-independent native signal calibration on immutable phase support.

The scalar input is the saved facies color model's signal BEFORE SimpleFlash
clips it. A monotone native PWTransformation can therefore retain distinctions
between colors that the imported calibration mapped to aqueous saturation.
Unknown calibration labels contribute an interval, never fabricated mass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

import darsia


SIGNAL_VERSION = "native-darsia-preflash-signal-calibration-v1"
VERSION = SIGNAL_VERSION
# Fixed, versioned knots; maps and fitting use precisely the same native curve.
SIGNAL_KNOTS = np.unique(np.r_[np.linspace(0., 1., 129), np.geomspace(1., 8., 49)])
PARAMETER_BOUNDS = dict(aqueous_scale=(.03, 4.), aqueous_exponent=(.35, 3.),
                        gas_scale=(.05, 4.))


@dataclass(frozen=True)
class SignalCalibrationParameters:
    aqueous_scale: float = 1.
    aqueous_exponent: float = 1.
    gas_scale: float = 1.

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Signal calibration parameters must be finite and positive")


def _flash(config):
    names = ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g")
    if config.get("model") not in (None, "simple"):
        raise ValueError("Preflash signal calibration currently requires native SimpleFlash")
    values = {name: float(config[name]) for name in names}
    if (not np.isfinite(list(values.values())).all()
            or values["max_value_aq"] <= values["min_value_aq"]
            or values["max_value_g"] <= values["min_value_g"]):
        raise ValueError("Finite, strictly ordered native SimpleFlash bounds required")
    return darsia.SimpleFlash(**values)


def _curve(parameters):
    return darsia.PWTransformation(supports=SIGNAL_KNOTS,
        values=parameters.aqueous_scale * SIGNAL_KNOTS ** parameters.aqueous_exponent)


def signal_response(signal, parameters=SignalCalibrationParameters(), flashconfig=None):
    """Return aqueous response and gas saturation without changing phase labels.

    The aqueous branch is normalized by the SAVED flash bounds and remapped by
    native PWTransformation. The gas branch uses the ORIGINAL native signal and
    saved SimpleFlash bounds. Thus aqueous tuning cannot silently alter gas.
    """
    if flashconfig is None:
        raise ValueError("Saved native SimpleFlash configuration is required")
    signal = np.asarray(signal, np.float32)
    if not np.isfinite(signal).all():
        raise ValueError("Preflash signal must be finite")
    if not signal.size:
        return signal.copy(), signal.copy()
    flash = _flash(flashconfig)
    normalized = np.maximum((signal - flash.min_value_aq)
                            / (flash.max_value_aq - flash.min_value_aq), 0)
    if np.max(normalized) > SIGNAL_KNOTS[-1] + 1e-6:
        raise ValueError("Native signal exceeds the registered calibration knot domain")
    mapped = _curve(parameters)(np.minimum(normalized, SIGNAL_KNOTS[-1]))
    # ScalarImage accepts arbitrary spatial dimension; flattening avoids making
    # geometric metadata part of a pointwise optical transformation.
    original_image = darsia.ScalarImage(signal.reshape(1, -1), dimensions=[1., 1.])
    mapped_image = darsia.ScalarImage(np.asarray(mapped, np.float32).reshape(1, -1),
                                    dimensions=[1., 1.])
    aqueous, _ = darsia.SimpleFlash(0., 1., 1., 2.)(mapped_image)
    _, gas = flash(original_image)
    return (np.asarray(aqueous.img, np.float32).reshape(signal.shape),
            np.clip(np.asarray(gas.img, np.float32).reshape(signal.shape)
                    * parameters.gas_scale, 0, 1))


def _validate_summary(summary):
    names = ("signal_values", "aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g")
    arrays = [np.asarray(summary[name], dtype=float) for name in names]
    if (summary.get("format") != SIGNAL_VERSION or any(
            value.ndim != 1 or value.shape != arrays[0].shape or not np.isfinite(value).all()
            for value in arrays) or any(np.any(value < 0) for value in arrays[1:])):
        raise ValueError("Invalid native preflash summary")
    unknown = summary.get("unknown_mass_upper_g", 0.)
    half_width = summary.get("signal_quantization_half_width", 0.)
    nominal = summary.get("nominal_injected_mass_g")
    if (not np.isfinite([unknown, half_width, summary["elapsed_h"]]).all()
            or min(unknown, half_width) < 0
            or (nominal is not None and (not np.isfinite(nominal) or nominal < 0))):
        raise ValueError("Invalid mass interval, time, or injected-mass observation")
    if summary.get("unquantified_phase_pixels", 0) and unknown <= 0:
        raise ValueError("Unquantified phase requires a positive physical mass upper bound")
    _flash(summary["flash"])
    return arrays


def evaluate_signal_summary(summary, parameters=SignalCalibrationParameters()):
    signal, aw, gaw, gw = _validate_summary(summary)
    aq, gas = signal_response(signal, parameters, summary["flash"])
    mass_aq = float(np.sum(aw * aq + gaw * aq * (1 - gas)))
    mass_gas = float(np.sum(gw * gas))
    flash = _flash(summary["flash"])
    curve = _curve(parameters)
    aq_lipschitz = float(np.max(np.diff(curve.values) / np.diff(curve.supports))) / (
        flash.max_value_aq - flash.min_value_aq)
    gas_lipschitz = parameters.gas_scale / (flash.max_value_g - flash.min_value_g)
    error = float(summary.get("signal_quantization_half_width", 0.) * (
        aq_lipschitz * np.sum(aw + gaw) + gas_lipschitz * np.sum(gaw + gw)))
    known = mass_aq + mass_gas
    upper = float(summary.get("unknown_mass_upper_g", 0.))
    return dict(run=summary["run"], elapsed_h=summary["elapsed_h"],
        nominal_injected_mass_g=summary.get("nominal_injected_mass_g"),
        aqueous_mass_g=mass_aq, gas_mass_g=mass_gas, total_mass_g=known,
        total_mass_lower_g=known, total_mass_upper_g=known + upper,
        numerical_lower_g=max(known-error, 0.), numerical_upper_g=known+upper+error,
        unknown_mass_upper_g=upper,
        unquantified_phase_pixels=int(summary.get("unquantified_phase_pixels", 0)),
        quantification_complete=not bool(summary.get("unquantified_phase_pixels", 0)),
        summary_mass_error_bound_g=error)


def _interval_distance(target, lower, upper):
    return lower - target if target < lower else upper - target if target > upper else 0.


class _CompiledSummaries:
    """Integrate all training histograms with one native response per node.

    Repeated signal histogram nodes across frames are evaluated once. Sparse
    geometry-capacity matrices integrate the response without image decoding,
    full-resolution fields, or one native image allocation per frame per step.
    """
    def __init__(self, records):
        self.count = len(records)
        self.keys = [(run, index) for run, index, _ in records]
        self.unknown = np.array([record[2].get("unknown_mass_upper_g", 0.) for record in records])
        self.half_width = np.array([record[2].get("signal_quantization_half_width", 0.) for record in records])
        groups = {}
        for row, (_, _, summary) in enumerate(records):
            flash = {key:float(summary["flash"][key]) for key in (
                "min_value_aq", "max_value_aq", "min_value_g", "max_value_g")}
            key = tuple(flash.values())
            groups.setdefault(key, (flash, []))[1].append((row, summary))
        self.groups = []
        for flash, items in groups.values():
            lengths = [len(summary["signal_values"]) for _, summary in items]
            raw = np.concatenate([np.asarray(summary["signal_values"], np.float32) for _, summary in items])
            nodes, columns = np.unique(raw, return_inverse=True)
            rows = np.repeat([row for row, _ in items], lengths)
            matrices = []
            for key in ("aqueous_capacity_g", "gas_aqueous_capacity_g", "gas_capacity_g"):
                weights = np.concatenate([np.asarray(summary[key], float) for _, summary in items])
                matrices.append(csr_matrix((weights, (rows, columns)), shape=(self.count, len(nodes))))
            aw, gaw, gw = matrices
            aq_capacity = np.asarray((aw+gaw).sum(axis=1)).ravel()
            gas_capacity = np.asarray((gaw+gw).sum(axis=1)).ravel()
            self.groups.append((flash, nodes, matrices, aq_capacity, gas_capacity))

    def evaluate(self, parameters):
        aqueous_mass = np.zeros(self.count)
        gas_mass = np.zeros(self.count)
        error = np.zeros(self.count)
        curve = _curve(parameters)
        slope = float(np.max(np.diff(curve.values) / np.diff(curve.supports)))
        for flash, nodes, (aw, gaw, gw), aq_capacity, gas_capacity in self.groups:
            aq, gas = signal_response(nodes, parameters, flash)
            aqueous_mass += aw @ aq + gaw @ (aq * (1 - gas))
            gas_mass += gw @ gas
            aq_slope = slope / (flash["max_value_aq"] - flash["min_value_aq"])
            gas_slope = parameters.gas_scale / (flash["max_value_g"] - flash["min_value_g"])
            error += self.half_width * (aq_slope * aq_capacity + gas_slope * gas_capacity)
        known = aqueous_mass + gas_mass
        return dict(aqueous_mass_g=aqueous_mass, gas_mass_g=gas_mass, total_mass_g=known,
            total_mass_lower_g=known, total_mass_upper_g=known + self.unknown,
            summary_mass_error_bound_g=error)


def summarize_prediction_errors(predictions, indices=None):
    """Separate known-support residuals from physically bounded interval error."""
    rows = []
    for run, frames in predictions.items():
        for index in (range(len(frames)) if indices is None else indices.get(run, ())):
            row = frames[index]
            target = row.get("nominal_injected_mass_g")
            if target is None or target <= 0:
                continue
            rows.append(dict(run=run, frame_index=index, target_g=target,
                known_mass_error_g=row["total_mass_g"] - target,
                interval_error_g=_interval_distance(target, row["total_mass_lower_g"], row["total_mass_upper_g"]),
                interval_width_g=row["total_mass_upper_g"] - row["total_mass_lower_g"],
                complete=row["quantification_complete"]))
    if not rows:
        return dict(observations=0, rows=[])
    return dict(observations=len(rows), complete_observations=sum(row["complete"] for row in rows),
        interval_rmse_g=float(np.sqrt(np.mean([row["interval_error_g"] ** 2 for row in rows]))),
        known_support_rmse_g=float(np.sqrt(np.mean([row["known_mass_error_g"] ** 2 for row in rows]))),
        mean_interval_width_g=float(np.mean([row["interval_width_g"] for row in rows])), rows=rows)


def fit_signal_summaries(summaries_by_run, native_provenance_by_run, *, scope="individual",
                         training_indices=None, unknown_policy="bounded_interval",
                         fit_gas=False, references=()):
    """Fit one time-independent curve; unknown areas cannot force known mass up.

    Gas fitting is opt-in and requires independent, sourced gas observations.
    Dose-normalized per-run residuals give experiments equal total weight. The
    resulting errors concern nominal-mass consistency, not local validation.
    """
    runs = summaries_by_run
    if (scope not in ("individual", "shared") or not runs
            or (scope == "individual" and len(runs) != 1)
            or (scope == "shared" and len(runs) < 2)):
        raise ValueError("Choose one run for individual fitting or two or more for shared fitting")
    if unknown_policy not in ("complete_frames", "bounded_interval"):
        raise ValueError("Unknown missing-calibration policy")
    references = list(references)
    for ref in references:
        if (ref.get("run") not in runs or not ref.get("source")
                or not 0 <= int(ref.get("frame_index", -1)) < len(runs[ref["run"]])
                or ref.get("quantity") != "gas_mass_g"
                or not np.isfinite([ref.get("mass_g", np.nan), ref.get("sigma_g", np.nan)]).all()
                or ref["mass_g"] < 0 or ref["sigma_g"] <= 0):
            raise ValueError("Gas references must be sourced gas mass observations with positive uncertainty")
    if fit_gas and not references:
        raise ValueError("Gas-scale fitting requires independent sourced gas observations")
    if fit_gas and any(runs[ref["run"]][int(ref["frame_index"])].get("unquantified_phase_pixels", 0)
                       for ref in references):
        raise ValueError("Gas fitting needs references with complete optical quantification support")
    selected = {}
    excluded = {}
    for run, frames in runs.items():
        if run not in native_provenance_by_run:
            raise ValueError("Native calibration provenance is required for every run")
        for frame in frames:
            _validate_summary(frame)
            if frame["run"] != run:
                raise ValueError("Summary run identity differs from its collection")
        candidates = list(range(len(frames))) if training_indices is None else list(training_indices.get(run, ()))
        if (len(candidates) != len(set(candidates))
                or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(frames) for i in candidates)):
            raise ValueError("Training frame indices must be unique and within the selected timeline")
        selected[run] = [i for i in candidates if frames[i].get("nominal_injected_mass_g") is not None
            and frames[i]["nominal_injected_mass_g"] > 0
            and (unknown_policy != "complete_frames" or not frames[i].get("unquantified_phase_pixels", 0))]
        excluded[run] = [i for i in candidates if i not in selected[run]]
        if len(selected[run]) < 3:
            raise ValueError(f"{run} needs three usable positive-mass training frames under {unknown_policy}")
    active_names = ["aqueous_scale", "aqueous_exponent"] + (["gas_scale"] if fit_gas else [])
    records = [(run, index, frames[index]) for run, frames in runs.items() for index in selected[run]]
    compiled = _CompiledSummaries(records)
    targets = np.array([record[2]["nominal_injected_mass_g"] for record in records])
    weights = np.array([1. / max(runs[run][i]["nominal_injected_mass_g"] for i in selected[run])
                        / np.sqrt(len(selected[run])) for run, _, _ in records])
    reference_rows = [(compiled.keys.index((ref["run"], int(ref["frame_index"]))), ref)
                      for ref in references if (ref["run"], int(ref["frame_index"])) in compiled.keys]
    def parameters(x):
        return SignalCalibrationParameters(**dict(zip(active_names, map(float, x))))
    def predict(x):
        p = parameters(x)
        return {run: [evaluate_signal_summary(frame, p) for frame in frames] for run, frames in runs.items()}
    def residuals(x):
        values = compiled.evaluate(parameters(x))
        result = (np.maximum(values["total_mass_lower_g"] - targets, 0.)
                  + np.minimum(values["total_mass_upper_g"] - targets, 0.)) * weights
        if fit_gas:
            result = np.r_[result, [(values["gas_mass_g"][row] - ref["mass_g"]) / ref["sigma_g"]
                                   for row, ref in reference_rows]]
        return result
    if fit_gas and not any(int(r["frame_index"]) in selected[r["run"]] for r in references):
        raise ValueError("No gas references belong to the selected training frames")
    bounds = ([PARAMETER_BOUNDS[name][0] for name in active_names],
              [PARAMETER_BOUNDS[name][1] for name in active_names])
    starts = [(1., 1.), (.5, 1.), (.5, 2.), (1., 2.), (2., .5)]
    fits = [least_squares(residuals, [*start, *([1.] if fit_gas else [])],
        bounds=bounds, max_nfev=200, diff_step=1e-3, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        for start in starts]
    fit = min(fits, key=lambda item: (float(np.sum(item.fun ** 2)),
                                    float(np.sum(np.log(item.x) ** 2))))
    p = parameters(fit.x)
    predictions = predict(fit.x)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(float(singular.max(initial=0)) * 1e-6, 1e-10)))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 1e-12 else None
    warnings = []
    if rank < len(active_names):
        warnings.append("Mass constraints do not identify every fitted curve parameter")
    if condition is None or condition > 1e4:
        warnings.append("Curve parameters are weakly identifiable; compare held-out runs and parameter stability")
    if any(frame.get("unquantified_phase_pixels", 0) for frames in runs.values() for frame in frames):
        warnings.append("Mass intervals retain uncalibrated phase support; no concentration was invented there")
    boundary = [name for name in active_names if min(getattr(p, name)-PARAMETER_BOUNDS[name][0],
                PARAMETER_BOUNDS[name][1]-getattr(p, name)) < 1e-3]
    if boundary:
        warnings.append("Fitted parameter reached a configured bound: " + ", ".join(boundary))
    return dict(format=SIGNAL_VERSION, scope=scope, fit_performed=True, parameters=asdict(p),
        training_indices=selected, excluded_training_indices=excluded, predictions=predictions,
        training_metrics=summarize_prediction_errors(predictions, selected),
        unknown_policy=unknown_policy, gas_scale_fitted=bool(fit_gas),
        native_calibrations=native_provenance_by_run, optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message), objective=float(np.sum(fit.fun ** 2)),
        jacobian_rank=rank, jacobian_condition_number=condition, parameters_at_bounds=boundary,
        warnings=warnings, model_knots=SIGNAL_KNOTS.tolist(),
        curve_implementation="native PWTransformation followed by native SimpleFlash",
        parameter_meaning="Shared or individual monotone correction of each run's native facies signal before aqueous clipping; original native gas response is retained unless explicitly fitted",
        phase_support_unchanged=True, time_independent_optical_mapping=True,
        identifiable_local_concentration=False,
        calibration_reference="nominal injected mass; no inferred mass-loss subtraction",
        reference_warnings=["Cumulative injection is a nominal reference, not independently measured retained mass; unknown losses must not be interpreted as optical calibration error"],
        validation_scope="nominal mass consistency; independent local concentration validation is unavailable")


def chronological_holdout_indices(summaries_by_run, *, fraction=.3):
    """Contiguous late holdout; never randomized neighboring frames."""
    if not 0 < fraction < 1:
        raise ValueError("Held-out fraction must lie between zero and one")
    training, heldout = {}, {}
    for run, frames in summaries_by_run.items():
        indices = [i for i, frame in enumerate(frames) if frame.get("nominal_injected_mass_g") is not None
                   and frame["nominal_injected_mass_g"] > 0]
        indices.sort(key=lambda i: frames[i]["elapsed_h"])
        split = max(3, int(np.floor(len(indices) * (1-fraction))))
        if split >= len(indices):
            raise ValueError(f"{run} needs at least four positive-mass frames for a holdout")
        training[run], heldout[run] = indices[:split], indices[split:]
    return training, heldout


def cross_validate_signal_summaries(summaries_by_run: Mapping, native_provenance_by_run: Mapping,
                                    *, fraction=.3, unknown_policy="bounded_interval", fit_gas=False):
    """Temporal and leave-one-run-out comparisons from cached scalar histograms."""
    if fit_gas:
        raise ValueError("Cross-validation currently keeps native gas fixed; use explicitly split gas references")
    training, heldout = chronological_holdout_indices(summaries_by_run, fraction=fraction)
    individual, shared, leave_one_out = {}, None, {}
    for run, frames in summaries_by_run.items():
        fit = fit_signal_summaries({run:frames}, {run:native_provenance_by_run[run]},
            training_indices={run:training[run]}, unknown_policy=unknown_policy)
        individual[run] = dict(parameters=fit["parameters"], warnings=fit["warnings"],
            training_indices=fit["training_indices"], heldout_indices={run:heldout[run]},
            training_metrics=fit["training_metrics"],
            heldout_metrics=summarize_prediction_errors(fit["predictions"], {run:heldout[run]}))
    if len(summaries_by_run) >= 2:
        fit = fit_signal_summaries(summaries_by_run, native_provenance_by_run, scope="shared",
            training_indices=training, unknown_policy=unknown_policy)
        shared = dict(parameters=fit["parameters"], warnings=fit["warnings"],
            training_indices=training, heldout_indices=heldout, training_metrics=fit["training_metrics"],
            heldout_metrics=summarize_prediction_errors(fit["predictions"], heldout))
        for omitted, frames in summaries_by_run.items():
            kept = {run:values for run, values in summaries_by_run.items() if run != omitted}
            fit = fit_signal_summaries(kept, {run:native_provenance_by_run[run] for run in kept},
                scope="shared" if len(kept) >= 2 else "individual", unknown_policy=unknown_policy)
            parameters = SignalCalibrationParameters(**fit["parameters"])
            predictions = {omitted:[evaluate_signal_summary(frame, parameters) for frame in frames]}
            leave_one_out[omitted] = dict(parameters=fit["parameters"], training_runs=list(kept),
                warnings=fit["warnings"], metrics=summarize_prediction_errors(predictions))
    return dict(format=SIGNAL_VERSION, heldout_fraction=fraction,
        split_policy="Contiguous late frames and entirely omitted runs; no held-out target used in fitting",
        unknown_policy=unknown_policy, individual=individual, shared=shared,
        leave_one_run_out=leave_one_out,
        interpretation="Generalization of nominal-mass consistency; not independent concentration validation")
