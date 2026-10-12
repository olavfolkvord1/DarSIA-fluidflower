"""Compare reference-normalized concentration curves on fixed Studio phase maps.

Each run is prepared once. Individual/shared fits and validation use cached
histograms; no phase inference, movies, proposals or disagreements are started.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io as streams
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from darsia.fluidflower import concentration_configuration as configuration
from darsia.fluidflower import concentration_inputs as inputs
from darsia.fluidflower import concentration_joint as joint
from darsia.fluidflower.tools import phase_studio_native_mass as io
from darsia.fluidflower.tools import phase_studio_calibration_compare as common

VERSION = "joint-optical-concentration-comparison-v2"
SETTINGS = dict(optical="joint_chromatic_od", chemistry="empirical_power",
                restoration="reference_gaussian", response="joint_reference_dilution",
                sigma_pixels=2., analysis_width=1536)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numerical_source_binding():
    from darsia.fluidflower.tools import phase_studio_joint_inputs as preparation
    from darsia.fluidflower.tools import phase_mass_joint_optics as optics
    from darsia.fluidflower import mass_materialization as physics
    from darsia.fluidflower.tools import overflow_phase_constraint as overflow
    from darsia.fluidflower.tools import phase_studio_series_mass as series
    return {str(Path(path).resolve()): io.sha(path)
            for path in (__file__, joint.__file__, preparation.__file__, optics.__file__,
                         configuration.__file__, physics.__file__, overflow.__file__, series.__file__)}


def load_overflow_references(request, inspected):
    """Bind reviewed target bytes and match gas measurements to selected photos."""
    from darsia.fluidflower.tools.phase_studio_series_mass import references_for
    mode = request.get("overflow_mode", "off")
    if mode not in ("off", "compare", "constrain", "hard"):
        raise ValueError("Unknown overflow mode")
    paths = request.get("overflow", {})
    if not isinstance(paths, dict) or any(run not in inspected for run in paths):
        raise ValueError("Overflow target paths must name selected experiments")
    if mode == "off":
        if paths:
            raise ValueError("Overflow target paths require compare, constrain or hard mode")
        return [], {}
    if not paths:
        raise ValueError("Compare, constrain or hard mode requires time-linked overflow targets")
    references, bindings = [], {}
    for run, path in paths.items():
        path = str(Path(path).resolve())
        selected = inspected[run]["selected_frame_indices"]
        times = [inspected[run]["frames"][index]["elapsed_h"] for index in selected]
        before_sha256 = io.sha(path)
        records, provenance = references_for(run, times, path)
        after_sha256 = io.sha(path)
        if before_sha256 != after_sha256 or provenance.get("sha256") != before_sha256:
            raise ValueError("Overflow target changed while matching measurements to photographs")
        bindings[path] = dict(sha256=before_sha256, provenance=provenance)
        for position, record in enumerate(records):
            if record is None:
                continue
            if mode in ("constrain", "hard") and record["source"] != "measured-run-specific":
                raise ValueError("An imputed overflow target cannot constrain a joint optical fit")
            references.append(dict(run=run, frame_index=position, time_h=record["time_h"],
                mass_g=record["mass_g"], sigma_g=record["sigma_g"], mode=record["mode"],
                source=record["source"], match_tolerance_h=provenance["match_tolerance_h"],
                target_path=path, target_sha256=bindings[path]["sha256"]))
    if not references:
        raise ValueError("No overflow target matches a selected photograph")
    return references, bindings


def read_prepared_arrays(path, expected=None):
    """Read each prepared archive once and verify those same bytes."""
    data = Path(path).read_bytes()
    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Prepared concentration input changed after preparation.")
    with np.load(streams.BytesIO(data), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _reference_path_identity(path):
    return os.path.normcase(str(Path(path).resolve()))


def validate_clearwater_assumptions(request, available, selected):
    """Require explicit, reproducible borrowed-reference declarations."""
    assumptions = request.get("clearwater_reference_assumptions", {})
    if not isinstance(assumptions, dict):
        raise ValueError("clearwater_reference_assumptions must be a run mapping.")
    required = {"kind", "donor_run", "reason", "correction_policy", "sources"}
    for name, assumption in assumptions.items():
        if name not in available or name not in selected:
            raise ValueError("Clearwater assumption targets an unknown or non-selected run: " + str(name))
        if not isinstance(assumption, dict) or set(assumption) != required:
            raise ValueError("Malformed clearwater reference assumption for " + name)
        if (assumption["kind"] != "borrowed_raw"
                or assumption["correction_policy"] != "target_registered_rig"
                or not isinstance(assumption["donor_run"], str)
                or assumption["donor_run"] not in available or assumption["donor_run"] == name
                or not isinstance(assumption["reason"], str) or not assumption["reason"].strip()):
            raise ValueError("Invalid borrowed clearwater reference policy or donor for " + name)
        sources = assumption["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("Borrowed clearwater sources must be a nonempty list for " + name)
        seen = set()
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise ValueError("Malformed borrowed clearwater source for " + name)
            path, checksum = source["path"], source["sha256"]
            if (not isinstance(path, str) or not path or not Path(path).is_absolute()
                    or not isinstance(checksum, str) or len(checksum) != 64
                    or any(character not in "0123456789abcdefABCDEF" for character in checksum)):
                raise ValueError("Borrowed clearwater source needs an absolute path and SHA-256 for " + name)
            identity = _reference_path_identity(path)
            if identity in seen:
                raise ValueError("Duplicate borrowed clearwater source for " + name)
            seen.add(identity)
    return assumptions


def validate_ffac_colorchecker(request, jobs, inspected):
    """Keep FF_AC concentration fits tied to its registered ColorChecker correction."""
    if not any(job.get("dataset") == "ffac" for job in jobs):
        return
    variant = request.get("photometry_variant")
    if variant is not None and (not isinstance(variant, dict)
                                or variant.get("colorchecker_enabled") is not True):
        raise ValueError("FF_AC concentration calibration requires ColorChecker-on photometry.")
    missing = []
    for job in jobs:
        contract = inspected[job["run"]].get("correction_contract") or {}
        environment = contract.get("environment") or {}
        if environment.get("FFAC_COLORCHECKER_REFINEMENT") != "on":
            missing.append(job["run"])
    if missing:
        raise ValueError("FF_AC concentration calibration requires registered ColorChecker correction: "
                         + ", ".join(missing))


def _validate_clearwater_sources(assumption, sources):
    """Compare ordered source identities; reject a different cached reference."""
    def identities(rows):
        return [(_reference_path_identity(row["path"]), row["sha256"].lower()) for row in rows]
    if identities(sources) != identities(assumption["sources"]):
        raise ValueError("Borrowed clearwater reference paths or SHA-256 differ from the declared sources.")


def _scoped_clearwater_assumptions(request, name, scope):
    assumptions = request.get("clearwater_reference_assumptions", {})
    return assumptions if scope == "shared" else ({name: assumptions[name]} if name in assumptions else {})


def _prepare_worker(job, metadata, output, width, prepared_from=None):
    from darsia.fluidflower.tools.phase_studio_joint_inputs import prepare_run, _clearwater_paths
    folder = Path(output) / "preparation" / job["run"]
    update = common._child_update(Path(output), "prepare", job["run"])
    update("Loading registered concentration inputs", run=job["run"])
    assumption = job.get("clearwater_reference_assumption")
    if assumption:
        paths = _clearwater_paths(job)
        if ([_reference_path_identity(path) for path in paths]
                != [_reference_path_identity(row["path"]) for row in assumption["sources"]]):
            raise ValueError("Borrowed clearwater reference paths differ from the declared sources.")
        _validate_clearwater_sources(assumption, [dict(path=str(path), sha256=io.sha(path)) for path in paths])
    config = configuration.load_run_configuration(job, purpose="joint_reference")
    result = prepare_run(job, metadata, config, Path(prepared_from) if prepared_from else folder,
                         metadata["selected_frame_indices"], analysis_width=width, update=update)
    result = dict(result)
    if assumption:
        _validate_clearwater_sources(assumption, result.get("reference_inputs", []))
        result["clearwater_reference_assumption"] = assumption
        result["optical_binding"] = dict(result.get("optical_binding", {}),
                                         clearwater_reference_assumption=assumption)
    if prepared_from:
        result["reused_preparation"] = str(Path(prepared_from).resolve())
    final_dose = float(result["expected_final_injected_mass_g"])
    if not np.isfinite(final_dose) or final_dose <= 0:
        raise ValueError("A known full protocol dose is required before fitting dilution.")
    summaries = []
    for row in result["frames"]:
        arrays = read_prepared_arrays(row["arrays"], row.get("arrays_sha256"))
        summary = joint.summarize_frame(arrays["signal"], arrays["av"], arrays["gv"],
            arrays["solubility"], arrays["density"], run=job["run"],
            elapsed_h=row["elapsed_h"], injected_g=row["injected_mass_g"],
            frame_index=row["frame_index"], expected_final_injected_mass_g=final_dose,
            solubility_max_g_l=float(arrays["solubility"].max()))
        summaries.append(summary)
        del arrays
    result["summaries"] = summaries
    result["color_limit_g_l"] = max(row["solubility_max_g_l"] for row in summaries)
    io.write(folder / "prepared.json", result)
    update("complete", run=job["run"], frames=len(summaries))
    return str(folder / "prepared.json")


def validation_indices(frames, mode):
    """Predeclare time folds; no choice based on a fit's measured error."""
    positive = [i for i, row in enumerate(frames) if row["nominal_injected_mass_g"] > 0]
    if len(positive) < 6:
        raise ValueError("At least six positive-mass times are needed for comparison.")
    if mode == "chronological":
        held = positive[max(3, int(.7 * len(positive))):]
    elif mode == "interleaved":
        held = sorted(set(positive[3::4] + positive[-2:]))
    else:
        raise ValueError("Unknown time validation fold.")
    training = [i for i in positive if i not in held]
    if len(training) < 3 or not held:
        raise ValueError("Insufficient separate training and held-out observations.")
    return training, held


def _metrics(rows):
    values = [dict(row, nominal_injected_mass_g=row.get("nominal_injected_mass_g", row.get("injected_mass_g")))
              for row in rows]
    return common.mass_metrics(values)


def _gas_reference_metrics(predictions, references, indices_by_run=None):
    pairs = []
    for ref in references:
        if indices_by_run is not None and ref["frame_index"] not in indices_by_run.get(ref["run"], ()):
            continue
        gas = predictions[ref["run"]][ref["frame_index"]]["gas_mass_g"]
        target, sigma = ref["mass_g"], ref["sigma_g"]
        lower = 0. if ref["mode"] == "gas-upper-bound" else target - sigma
        outside = max(lower - gas, gas - target - sigma, 0.)
        pairs.append(dict(run=ref["run"], frame_index=ref["frame_index"],
                          measured_g=target, predicted_g=gas, sigma_g=sigma,
                          outside_uncertainty_g=outside, source=ref["source"]))
    return dict(observations=len(pairs), mean_outside_uncertainty_g=(
        float(np.mean([row["outside_uncertainty_g"] for row in pairs])) if pairs else None),
        independent_validation=False,
        interpretation=("Image-fit target values are held out as indicated, but overflow-derived "
                        "gas references may share feed assumptions across experiments."),
        rows=pairs)


def fit_and_validate(summaries, output, update, gas_bounds, optimization_settings=None,
                     overflow_mode="off", overflow_references=(), overflow_weight=.25,
                     overflow_target_bindings=None):
    references = list(overflow_references)
    def fit_joint(data, **kwargs):
        if optimization_settings is not None:
            kwargs['optimization_settings'] = optimization_settings
        kwargs['overflow_mode'] = overflow_mode
        kwargs['overflow_references'] = [row for row in references if row['run'] in data]
        kwargs['overflow_weight'] = overflow_weight
        # A target-free single-run fit is still valid while fitting the shared
        # cohort, and is reported as a mass-only individual comparison.
        if not kwargs['overflow_references']:
            kwargs['overflow_mode'] = 'off'
        return joint.fit_joint_summaries(data, **kwargs)
    individual = {}
    for name, rows in summaries.items():
        update("Fitting individual joint optical response", run=name)
        individual[name] = fit_joint({name: rows}, scope="individual", gas_bounds=gas_bounds)
    update("Fitting one shared joint optical response")
    shared = fit_joint(summaries, scope="shared", gas_bounds=gas_bounds) if len(summaries) > 1 else None
    fits = dict(individual=individual, shared=shared)
    io.write(output / "fits.json", fits)
    validation = dict(format=VERSION, time_folds={}, whole_run_holdout={},
        interpretation=("Development checks against nominal injected mass and overflow-derived gas "
                        "references; neither independently validates local concentration or the phase split."
                        if overflow_mode != "off" else
                        "Development checks against nominal injected mass; not independent local concentration validation."),
        shared_endpoint="One pooled source-training endpoint, never fitted on a held-out experiment.",
        temporal_context="Observed injection-end phase volume is a causal input, not a fitted mass target.",
        overflow_mode=overflow_mode,
        overflow_policy="Only matched training-time target values constrain a fold; held-out values are evaluation only.",
        overflow_source_dependency=(("Frozen feed corrections and donor-derived effective feed rates "
            "can couple overflow targets across experiments. Whole-run overflow figures are source-dependent, "
            "provisional checks, not independent gas validation.") if overflow_mode != "off" else None),
        overflow_target_bindings=overflow_target_bindings or {})
    for mode in ("interleaved", "chronological"):
        update("Testing held-out times", fold=mode)
        folds = {name: validation_indices(rows, mode) for name, rows in summaries.items()}
        training = {name: pair[0] for name, pair in folds.items()}
        block = dict(individual={}, shared={})
        for name, rows in summaries.items():
            fit = fit_joint({name: rows}, scope="individual", gas_bounds=gas_bounds,
                                           training_indices_by_run={name: training[name]})
            held = [fit["predictions"][name][i] for i in folds[name][1]]
            block["individual"][name] = dict(training_indices=training[name], held_out_indices=folds[name][1],
                parameters=fit["parameters_by_run"][name], rows=held, metrics=_metrics(held),
                gas_reference_metrics=_gas_reference_metrics(fit["predictions"], references,
                    {name: folds[name][1]}),
                optimization=fit.get("diagnostics", {}).get("optimization"), warnings=fit.get("warnings", []))
        if len(summaries) > 1:
            fit = fit_joint(summaries, scope="shared", gas_bounds=gas_bounds,
                                           training_indices_by_run=training)
            for name in summaries:
                held = [fit["predictions"][name][i] for i in folds[name][1]]
                block["shared"][name] = dict(training_indices=training[name], held_out_indices=folds[name][1],
                    parameters=fit["parameters_by_run"][name], rows=held, metrics=_metrics(held),
                    gas_reference_metrics=_gas_reference_metrics(fit["predictions"], references,
                        {name: folds[name][1]}),
                    optimization=fit.get("diagnostics", {}).get("optimization"), warnings=fit.get("warnings", []))
        validation["time_folds"][mode] = block
        io.write(output / "cross_validation.json", validation)
    if len(summaries) > 2:
        for target, rows in summaries.items():
            update("Testing transfer to a held-out experiment", run=target)
            fit = fit_joint({name: data for name, data in summaries.items() if name != target},
                                           scope="shared", gas_bounds=gas_bounds)
            parameters = next(iter(fit["parameters_by_run"].values()))
            anchor = individual[target]["anchors_by_run"][target]
            predicted = joint.predict_joint_summaries({target: rows}, {target: parameters}, {target: anchor})[target]
            validation["whole_run_holdout"][target] = dict(parameters=parameters, rows=predicted,
                metrics=_metrics(predicted), target_response_fitted=False, target_endpoint_fitted=False,
                target_overflow_fitted=False,
                gas_reference_metrics=_gas_reference_metrics({target: predicted}, references,
                    {target: range(len(predicted))}),
                optimization=fit.get("diagnostics", {}).get("optimization"), warnings=fit.get("warnings", []))
            io.write(output / "cross_validation.json", validation)
    return fits, validation


def _render_worker(job, metadata, output, catalog, request):
    from darsia_extensions.studio_mass import totals, total_mass_interval, MassPolicy
    from darsia.fluidflower.tools.phase_studio_mass import render_frame
    from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
    from darsia.fluidflower.tools.phase_studio_series_mass import export_csv
    from darsia.fluidflower.tools.phase_studio_series_mass import render_overflow
    output = Path(output)
    name = job["run"]
    request_binding_path = output / "request_binding.json"
    numerical_sources = (io.read(request_binding_path).get("numerical_sources", {})
                         if request_binding_path.exists() else {})
    if any(io.sha(path) != checksum for path, checksum in numerical_sources.items()):
        raise ValueError("Numerical code changed after this comparison was prepared.")
    prepared = io.read(output / "preparation" / name / "prepared.json")
    overflow_path = output / "overflow_binding.json"
    overflow_manifest = (io.read(overflow_path) if overflow_path.exists() else
                         dict(references=[], target_bindings={}, mode="off"))
    def verify_overflow_targets():
        for path, bound_target in overflow_manifest["target_bindings"].items():
            if io.sha(path) != bound_target["sha256"]:
                raise ValueError("Overflow target changed after fitting")
    verify_overflow_targets()
    gas_by_position = {ref["frame_index"]: ref for ref in overflow_manifest["references"]
                       if ref["run"] == name}
    photometry = request.get("photometry_variant")
    if photometry is not None and prepared.get("photometry_variant") != photometry:
        raise ValueError("Prepared photometry differs from the requested variant.")
    prepared_sha = io.sha(output / "preparation" / name / "prepared.json")
    if prepared.get("clearwater_reference_assumption") != request.get("clearwater_reference_assumptions", {}).get(name):
        raise ValueError("Prepared clearwater reference assumption differs from the request.")
    bound = prepared.get("binding", {})
    if (bound.get("native_manifest_sha256") is not None
            and bound["native_manifest_sha256"] != metadata.get("native_manifest_sha256")):
        raise ValueError("Prepared concentration inputs belong to another photograph manifest.")
    fits = io.read(output / "fits.json")
    scopes = dict(individual=fits["individual"][name])
    if fits["shared"] is not None:
        scopes["shared"] = fits["shared"]
    update = common._child_update(output, "render", name)
    delivered = []
    active_scopes = {}
    for scope, fit in scopes.items():
        existing = output / scope / name / "result.json"
        if existing.exists():
            saved = io.read(existing)
            if (not saved.get("complete") or saved.get("selected_inputs", {}).get("candidate") != job["candidate"]
                    or saved.get("calibration_identity") != digest(fit)
                    or (photometry is not None and saved.get("prepared_manifest_sha256") != prepared_sha)
                    or saved.get("reference_assumption_identity", digest({}))
                    != digest(_scoped_clearwater_assumptions(request, name, scope))):
                raise ValueError("An existing result has another identity; select a new output directory.")
            verify_overflow_targets()
            register_variant(existing, catalog["results_root"])
            delivered.append(dict(run=name, scope=scope, result=str(existing), metrics=_metrics(saved["frames"])))
        else:
            active_scopes[scope] = fit
    if not active_scopes:
        update("complete", run=name, completed_results=len(delivered), reused=True)
        return delivered
    rows_by_scope = {scope: [] for scope in active_scopes}
    policy = MassPolicy(gas_saturation_min=request["gas_bounds"][0], gas_saturation_max=request["gas_bounds"][1])
    reference_path = prepared["references"]
    refs = read_prepared_arrays(reference_path, prepared.get("references_sha256"))
    pv = refs["pv"] if "pv" in refs else refs["pore_volume"]
    color_limit = prepared.get("color_limit_g_l", 2.)
    source_bindings = dict(prepared.get("source_bindings", {}))
    for reference in prepared.get("reference_inputs", []):
        source_bindings[reference["path"]] = reference["sha256"]
    optical_provenance = {key: prepared[key] for key in (
        "binding", "binding_sha256", "optical_binding", "reference_diagnostics",
        "reference_inputs", "current_correction_contract", "photometric_policy",
        "references", "references_sha256") if key in prepared}
    for position, record in enumerate(prepared["frames"]):
        index = record["frame_index"]
        update("Rendering joint concentration maps", run=name, frame=index + 1, frames=len(prepared["frames"]))
        source = record.get("source_reference", {})
        current = metadata["frames"][index]
        for key in ("photo", "phase"):
            if source.get(key):
                if current.get(key) and Path(current[key]).resolve() != Path(source[key]).resolve():
                    raise ValueError("Prepared frame refers to another selected source.")
                if io.sha(source[key]) != source[key + "_sha256"]:
                    raise ValueError("Selected photograph or phase map changed after preparation.")
        arrays = read_prepared_arrays(record["arrays"], record.get("arrays_sha256"))
        signal, av, gv, sol, density, rgb = [arrays[key] for key in ("signal", "av", "gv", "solubility", "density", "rgb")]
        for scope, fit in active_scopes.items():
            folder = output / scope / name
            folder.mkdir(parents=True, exist_ok=True)
            parameters = joint.JointConcentrationParameters(**fit["parameters_by_run"][name])
            anchor = fit["anchors_by_run"][name]
            volume = float(np.sum(av, dtype=np.float64) + np.sum(gv, dtype=np.float64))
            factor = joint.dilution_factor(record["elapsed_h"], volume, anchor, parameters.beta)
            concentration = joint.concentration_field(signal, parameters, factor, sol)
            masses = totals(concentration, av, gv, sol, density, parameters.gas_saturation)
            aq, gas, total = [masses[key] for key in ("aqueous_mass_g", "gas_mass_g", "total_mass_g")]
            interval = total_mass_interval(concentration, av, gv, sol, density, policy)
            aq_volume = float(av.sum(dtype=np.float64))
            mean = float(np.sum(concentration.astype(float) * av) / aq_volume) if aq_volume else None
            fields = folder / f"fields_{index:04d}.npz"
            # Save the reusable numerical product, not a movie or suggested labels.
            np.savez_compressed(fields, concentration_g_l=concentration.astype(np.float32),
                aqueous_volume_m3=av, gas_volume_m3=gv, pore_volume_m3=pv,
                signal=signal, solubility_g_l=sol, gas_density_g_l=density,
                photograph_rgb=rgb,
                aqueous_fraction=np.divide(av, pv, out=np.zeros_like(av), where=pv > 0),
                gas_fraction=np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0))
            preview = folder / f"frame_{index:04d}.png"
            row = dict(record, fields=str(fields), preview=str(preview), aqueous_mass_g=float(aq),
                gas_mass_g=float(gas), total_mass_g=float(total), nominal_injected_mass_g=record["injected_mass_g"],
                dilution_factor=float(factor), mean_aqueous_concentration_g_l=mean,
                aqueous_support_area_m2=record.get("aqueous_support_area_m2"),
                color_limit_g_l=color_limit,
                quantification_complete=True, total_mass_lower_g=float(interval["total_mass_min_g"]),
                total_mass_upper_g=float(interval["total_mass_max_g"]),
                calibration_role="individual joint optical fit" if scope == "individual" else "shared series joint optical fit",
                gas_reference=gas_by_position.get(position))
            row.pop("arrays", None)
            render_frame(preview, rgb, concentration, av, gv, pv, row,
                name.upper() + " · " + scope + " reference + dilution", color_limit)
            rows_by_scope[scope].append(row)
            predicted = fit["predictions"][name][position]
            bound = predicted["summary_mass_error_bound_g"]
            if abs(float(total) - predicted["total_mass_g"]) > bound + max(1e-5, abs(total) * 2e-5):
                raise ValueError("Dense evaluation disagrees with the calibrated summary; do not publish.")
        source = record.get("source_reference", {})
        for key in ("photo", "phase"):
            if source.get(key) and source.get(key + "_sha256"):
                source_bindings[source[key]] = source[key + "_sha256"]
    for scope, rows in rows_by_scope.items():
        folder = output / scope / name
        fit = scopes[scope]
        local_fit = dict(fit, parameters=fit["parameters_by_run"][name], scope=scope,
            mass_policy=asdict(policy), validation_scope="Same-run time holdouts and whole-run holdout; see cross_validation.json",
            fitting_scope_description=("One response and endpoint shared by every included experiment." if scope == "shared"
                                       else "One response fitted for this experiment."))
        assumption = prepared.get("physical_binding", {}).get("pressure_temperature_assumption")
        additional_warnings = []
        if assumption:
            additional_warnings.append("Pressure/temperature include an explicit assumption: "
                                       + assumption["reason"])
        if prepared.get("zero_dose_phase_support"):
            additional_warnings.append(
                "Some zero-dose photographs have fixed CO2 phase support. These photographs are "
                "excluded from fitting; their signal may be carryover or a phase-map false positive.")
        if scope == "shared" and request.get("physical_assumptions_note"):
            additional_warnings.append("Shared calibration includes assumed physical inputs. "
                                       + request["physical_assumptions_note"])
        reference_assumptions = _scoped_clearwater_assumptions(request, name, scope)
        for target, reference_assumption in reference_assumptions.items():
            additional_warnings.append(
                ("Shared calibration includes a borrowed clearwater reference: " if scope == "shared"
                 else "Borrowed clearwater reference: ")
                + target.upper() + " uses raw " + reference_assumption["donor_run"].upper()
                + " reference images corrected by the target's registered rig. " + reference_assumption["reason"])
        local_optical_provenance = dict(optical_provenance,
                                       clearwater_reference_assumptions=reference_assumptions)
        local_source_bindings = dict(source_bindings)
        relevant_targets = {ref["target_path"] for ref in overflow_manifest["references"]
                            if scope == "shared" or ref["run"] == name}
        for path in relevant_targets:
            local_source_bindings[path] = overflow_manifest["target_bindings"][path]["sha256"]
        for reference_assumption in reference_assumptions.values():
            for source in reference_assumption["sources"]:
                local_source_bindings[source["path"]] = source["sha256"]
        local_fit["warnings"] = list(dict.fromkeys([*fit.get("warnings", []), *additional_warnings]))
        io.write(folder / "calibration.json", local_fit)
        mass_curve, dilution_curve = folder / "mass_curve.png", folder / "dilution_curve.png"
        io.render_curve(mass_curve, rows, name.upper() + " · " + scope + " joint optical", True)
        common.render_dilution_curve(dilution_curve, rows, name.upper() + " · " + scope + " joint optical")
        overflow_curve = None
        if gas_by_position:
            overflow_curve = folder / "overflow_comparison.png"
            render_overflow(overflow_curve, rows,
                            name.upper() + " · " + scope + " joint optical")
        result = dict(format=VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
            calibration_identity=digest(fit),
            reference_assumption_identity=digest(reference_assumptions),
            selected_inputs=dict(candidate=job["candidate"], review_root=job["review_root"]),
            frames=rows, calibration=local_fit, method=dict(settings=dict(SETTINGS, analysis_width=request["analysis_width"], **(photometry or {})),
                optical_provenance=local_optical_provenance), source_bindings=local_source_bindings,
            numerical_source_sha256=numerical_sources,
            prepared_manifest_sha256=io.sha(output / "preparation" / name / "prepared.json"),
            color_limit_g_l=color_limit,
            physical_provenance=prepared.get("physical_binding", {}), phase_inference_repeated=False,
            phase_labels_modified=False, mass_curve=str(mass_curve), dilution_curve=str(dilution_curve),
            zero_dose_phase_support=prepared.get("zero_dose_phase_support", []),
            overflow_curve=str(overflow_curve) if overflow_curve else None,
            overflow_mode=fit.get("overflow_mode", "off"),
            overflow_target_binding={path: overflow_manifest["target_bindings"][path]
                                     for path in relevant_targets},
            series_calibration=str(folder / "calibration.json"),
            cross_validation=str(output / "cross_validation.json"), requested_frame_indices=metadata["selected_frame_indices"],
            available_review_frames=len(metadata["frames"]),
            excluded_frame_reason=request.get("frame_exclusion_reasons", {}).get(name),
            limitations=[*additional_warnings,
                "Inverse fit to nominal injected mass is not independent local concentration validation.",
                *(["Overflow-derived gas references depend on outflow and feed assumptions; "
                    "shared feed corrections can couple experiments. Holdout gas figures are provisional."]
                   if request.get("overflow_mode", "off") != "off" else []),
                "Phase support is unchanged. Current registered review photometry differs from the September 14 movie revision.",
                "Dilution is an empirical affected-volume correction, not a transport simulation.",
                "Gas saturation and dissolved concentration are not uniquely determined by total mass.",
                "Inspect abrupt changes at sand-layer boundaries and changes between consecutive photographs."])
        io.write(folder / "result.json", result)
        export_csv(result, folder / "mass_per_frame.csv")
        if any(io.sha(path) != checksum for path, checksum in numerical_sources.items()):
            raise ValueError("Numerical code changed before concentration publication.")
        verify_overflow_targets()
        register_variant(folder / "result.json", catalog["results_root"])
        delivered.append(dict(run=name, scope=scope, result=str(folder / "result.json"), metrics=_metrics(rows)))
    update("complete", run=name, completed_results=len(delivered))
    return delivered


def run(request_path):
    start = time.perf_counter()
    common._limit_worker_threads(1)
    request_path = Path(request_path)
    request = io.read(request_path)
    def update(stage, **extra):
        io.write(request_path.with_name("status.json"), dict(stage=stage,
            updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start, **extra))
    try:
        output = Path(request["output"]).resolve()
        physical_profile = request.get("concentration_profile_path")
        expected_profile_sha = request.get("physical_profile_file_sha256")
        if expected_profile_sha and (not physical_profile
                                     or io.sha(physical_profile) != expected_profile_sha):
            raise ValueError("Physical concentration profile changed after selection")
        receipt_path = request.get("physical_profile_receipt_path")
        receipt_sha = request.get("physical_profile_receipt_sha256")
        if receipt_sha and (not receipt_path or io.sha(receipt_path) != receipt_sha):
            raise ValueError("Documented physical-profile receipt changed after selection")
        catalog = inputs.discover_model_inputs(request["model_root"])
        if catalog["model_id"] != request["model_id"]:
            raise ValueError("Requested phase model differs from its registered binding.")
        if not output.is_relative_to(Path(catalog["results_root"]).resolve()):
            raise ValueError("Result files must belong to this series on NAS.")
        request.setdefault("gas_bounds", [.5, 1.])
        request.setdefault("analysis_width", 1536)
        request.setdefault("workers", 1)
        available = {job["run"]: dict(job) for job in catalog["jobs"]}
        if request.get("concentration_profile_path"):
            for job in available.values():
                job["concentration_profile_path"] = request["concentration_profile_path"]
        names = request["runs"]
        if len(set(names)) != len(names) or any(name not in available for name in names):
            raise ValueError("Select unique registered experiments.")
        reference_assumptions = validate_clearwater_assumptions(request, available, names)
        for name, assumption in reference_assumptions.items():
            available[name]["clearwater_reference_assumption"] = assumption
        jobs, blocked = [], []
        for name in names:
            description = configuration.describe_run(available[name], purpose="joint_reference")
            if name in request.get("exclude_runs", {}) or not description["ready"]:
                blocked.append(dict(run=name, reasons=[request["exclude_runs"][name]] if name in request.get("exclude_runs", {})
                                    else description["reasons"]))
            else:
                jobs.append(available[name])
        if blocked and request.get("require_all_runs", False):
            raise ValueError("All requested experiments are required; unavailable: "
                             + "; ".join(item["run"] + ": " + ", ".join(item["reasons"]) for item in blocked))
        if not jobs:
            raise ValueError("No selected experiment has registered physical inputs.")
        if any(name not in {job["run"] for job in jobs} for name in reference_assumptions):
            raise ValueError("Clearwater reference assumption targets an excluded or unavailable run.")
        groups = {configuration.describe_run(job, purpose="joint_reference")["chemistry_group"] for job in jobs}
        if (len(jobs) > 1 and (len(groups) != 1 or groups & {"uncertainty", "provisional", "unconfirmed"}
                or any(not configuration.describe_run(job, purpose="joint_reference").get("shared_fit_eligible") for job in jobs))):
            raise ValueError("One confirmed chemistry group is required for shared fitting.")
        inspected = {job["run"]: inputs.inspect_run(job) for job in jobs}
        validate_ffac_colorchecker(request, jobs, inspected)
        excluded = common.select_frames(request, inspected)
        overflow_references, overflow_bindings = load_overflow_references(request, inspected)
        expected_counts = request.get("expected_frame_counts", {})
        if expected_counts:
            actual_counts = {name: len(metadata["selected_frame_indices"])
                             for name, metadata in inspected.items()}
            if actual_counts != expected_counts:
                raise ValueError(f"Selected frame coverage differs from request: {actual_counts} versus {expected_counts}")
        output.mkdir(parents=True, exist_ok=True)
        binding = {key: value for key, value in request.items() if key not in {"stage", "resume", "workers"}}
        binding["numerical_sources"] = numerical_source_binding()
        binding["overflow_target_sha256"] = {path: item["sha256"]
                                              for path, item in overflow_bindings.items()}
        binding_path = output / "request_binding.json"
        if binding_path.exists() and io.read(binding_path) != binding:
            raise ValueError("This output directory belongs to a different scientific request.")
        io.write(binding_path, binding)
        io.write(output / "overflow_binding.json", dict(references=overflow_references,
                 target_bindings=overflow_bindings, mode=request.get("overflow_mode", "off")))
        io.write(output / "selection.json", dict(runs=[job["run"] for job in jobs], blocked=blocked,
                 excluded_frames=excluded,
                 **({"image_selection_unused_runs": request["image_selection_unused_runs"]}
                    if "image_selection_unused_runs" in request else {})))
        io.write(output / "input_catalog.json", dict(catalog=catalog, inspected=inspected))
        workers = min(int(request["workers"]), len(jobs), os.cpu_count() or 1)
        if workers < 1:
            raise ValueError("At least one worker is required.")
        timings = {}
        stage = request.get("stage", "all")
        if stage not in {"all", "prepare", "fit", "render"}:
            raise ValueError("Unknown comparison stage.")
        if stage in {"all", "prepare"}:
            tick = time.perf_counter()
            manifests = common._parallel_jobs(_prepare_worker,
                [(job, inspected[job["run"]], str(output), request["analysis_width"],
                  request.get("prepared_inputs_by_run", {}).get(job["run"])) for job in jobs],
                workers=workers, threads=1, output=output, stage="prepare", update=update)
            summaries = {io.read(path)["run"]: io.read(path)["summaries"] for path in manifests}
            summaries = {job["run"]: summaries[job["run"]] for job in jobs}
            io.write(output / "summaries.json", summaries)
            timings["preparation_seconds"] = time.perf_counter() - tick
        if stage in {"all", "fit"}:
            tick = time.perf_counter()
            fit_and_validate(io.read(output / "summaries.json"), output, update,
                tuple(request["gas_bounds"]), request.get("optimization_settings"),
                request.get("overflow_mode", "off"), overflow_references,
                request.get("overflow_weight", .25), overflow_bindings)
            timings["fit_and_validation_seconds"] = time.perf_counter() - tick
        if stage in {"all", "render"}:
            tick = time.perf_counter()
            deliveries = common._parallel_jobs(_render_worker,
                [(job, inspected[job["run"]], str(output), catalog, request) for job in jobs],
                workers=workers, threads=1, output=output, stage="render", update=update)
            timings["render_seconds"] = time.perf_counter() - tick
            delivered = [item for group in deliveries for item in group]
            io.write(output / "delivery.json", dict(format=VERSION, complete=True, experiments=delivered,
                blocked=blocked, excluded_frames=excluded, model_id=catalog["model_id"], timings=timings))
        timings["elapsed_seconds"] = time.perf_counter() - start
        if any(io.sha(path) != item["sha256"] for path, item in overflow_bindings.items()):
            raise ValueError("Overflow target changed during calibration")
        if numerical_source_binding() != binding["numerical_sources"]:
            raise ValueError("Numerical code changed during this calibration job.")
        if expected_profile_sha and io.sha(physical_profile) != expected_profile_sha:
            raise ValueError("Physical concentration profile changed during calibration")
        if receipt_sha and io.sha(receipt_path) != receipt_sha:
            raise ValueError("Documented physical-profile receipt changed during calibration")
        io.write(request_path.with_name("timing.json"), timings)
        update("complete", completed_stage=stage, output=str(output), timing=timings)
    except Exception:
        update("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run(parser.parse_args().request)
