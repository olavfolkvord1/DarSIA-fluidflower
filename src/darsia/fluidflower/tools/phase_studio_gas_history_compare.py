"""Controlled gas-occupancy sensitivity study on immutable Studio phase maps.

Photometry, physical volumes and photographs are reused by checksum. Gas
history is an exploratory, causal assumption, never an inferred 3-D measurement.
Only explicitly selected shared fits are published to the ordinary Studio.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io as streams
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from darsia.fluidflower import concentration_joint as joint
from darsia.fluidflower.tools import phase_studio_native_mass as io
from darsia.fluidflower.tools import phase_studio_joint_calibration_compare as comparison

VERSION = "gas-history-sensitivity-v1"
FAMILY = "gas_history_sensitivity_v1"
RUNS = ("run6", "run7", "run9", "run10", "run11", "run12", "run14",
        "run15", "run16", "run17", "run18", "run19", "run20")


def variants():
    """Predeclared choices: sensitivity scenarios, not measured saturations."""
    result = [dict(id="free_sg", label="Fitted constant gas saturation", gas_mode="free", publish=True)]
    for sg in (.5, .6, .7, .75, .8, .9, 1.):
        result.append(dict(id=f"fixed_sg_{round(100 * sg):03d}",
                           label=f"Fixed gas saturation {100 * sg:g}%", gas_mode="fixed",
                           gas_saturation=sg, publish=sg in (.6, .75, .9, 1.)))
    for tau in (.5, 2., 6.):
        result.append(dict(id="history_tau" + str(tau).replace(".", "p"),
                           label=f"Gas history 60–90%, time scale {tau:g} h",
                           gas_mode="history", tau_h=tau, low=.6, high=.9,
                           core_mm=10., publish=tau == 2.))
    result.append(dict(id="free_sg_no_dilution", label="Fitted gas; dilution disabled",
                       gas_mode="free", beta_fixed=0., publish=True))
    return result


def _limits():
    import cv2
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1)
    # Keep the controller alive for the process lifetime.
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)


def _verified(path, expected):
    data = Path(path).read_bytes()
    if not expected or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Immutable input changed: " + str(path))
    return data


def _arrays(path, expected, cache):
    """Read verified archive bytes; dense re-use is local, never inside Git."""
    target = Path(cache) / (expected + ".npz")
    if target.exists():
        data = _verified(target, expected)
    else:
        data = _verified(path, expected)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        try:
            temporary.replace(target)
        except (PermissionError, FileExistsError):
            # Another worker can publish the same content while Windows holds
            # a read handle on it. Reuse only the checksum-verified winner.
            if not target.is_file():
                raise
            data = _verified(target, expected)
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(streams.BytesIO(data), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _snapshot(path, expected):
    return json.loads(_verified(path, expected).decode("utf-8-sig"))


def _injection_intervals(stages):
    intervals = []
    for stage in stages:
        for segment in stage.get("rate_segments") or [stage]:
            if float(segment.get("rate_kg_s", 1.)) <= 0:
                continue
            begin, end = float(segment.get("start_h", 0.)), float(segment["end_h"])
            if not np.isfinite([begin, end]).all() or not 0 <= begin < end:
                raise ValueError("Invalid positive-rate injection interval")
            intervals.append((begin, end))
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((begin, end))
    if not merged:
        raise ValueError("Positive-rate injection schedule required")
    return merged


def validate_template(prepared, template):
    for settings in (prepared.get("photometry_variant", {}), template.get("method", {}).get("settings", {})):
        if (settings.get("photometry_variant_id") != "none_cc_off" or settings.get("illumination") != "none"
                or settings.get("colorchecker_enabled") is not False):
            raise ValueError("This study requires the exact no-illumination/no-ColorChecker branch")
    candidate = template["selected_inputs"]["candidate"]
    identity = prepared["binding"]["identity"]
    for key in ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256"):
        if candidate.get(key) != identity.get(key) or not identity.get(key):
            raise ValueError("Prepared and Studio phase identities differ: " + key)
    if len(prepared["frames"]) != len(template["frames"]):
        raise ValueError("Prepared and Studio frame counts differ")
    for a, b in zip(prepared["frames"], template["frames"]):
        if (a["frame_index"] != b["frame_index"] or a["source_name"] != b["source_name"]
                or abs(a["elapsed_h"]-b["elapsed_h"]) > 1e-8
                or abs(a["injected_mass_g"]-b["injected_mass_g"]) > 1e-8):
            raise ValueError("Prepared and Studio frame identities differ")


def _history_capacity(sg, gv, density, solubility):
    gas = float(np.sum(gv.astype(float) * density * sg) * 1000)
    residual = float(np.sum(gv.astype(float) * solubility * (1 - sg)) * 1000)
    return dict(gas_history_mass_g=gas, residual_history_mass_g=residual)


def _prepare_run(request, name):
    _limits()
    from darsia.fluidflower.gas_history import GasHistoryTracker
    root = Path(request["output"])
    binding = request["inputs"][name]
    prepared = _snapshot(binding["prepared"], binding["prepared_sha256"])
    refs = _arrays(prepared["references"], prepared["references_sha256"], request["cache"])
    pv = refs["pv"]
    dimensions = prepared["dimensions_m"]
    if isinstance(dimensions, dict):
        dimensions = [dimensions["height"], dimensions["width"]]
    spacing = tuple(float(d) / n for d, n in zip(dimensions, pv.shape))
    intervals = _injection_intervals(prepared["binding"]["injection_stages"])
    injection_end = max(end for _, end in intervals)
    history_variants = [v for v in request["variants"] if v["gas_mode"] == "history"]
    trackers = {v["id"]: GasHistoryTracker(spacing, low=v["low"], high=v["high"],
                tau_h=v["tau_h"], core_mm=v["core_mm"], injection_intervals_h=intervals) for v in history_variants}
    records = []
    folder = root / "history" / name
    folder.mkdir(parents=True, exist_ok=True)
    for position, frame in enumerate(prepared["frames"]):
        io.write(folder / "status.json", dict(stage="preparing", frame=position+1,
            frames=len(prepared["frames"]), run=name))
        arrays = _arrays(frame["arrays"], frame["arrays_sha256"], request["cache"])
        gv = arrays["gv"]
        fraction = np.clip(np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0), 0, 1)
        fields, scenarios = {}, {}
        for variant in history_variants:
            sg, diagnostics = trackers[variant["id"]].update(fraction, frame["elapsed_h"], injection_end)
            del diagnostics
            values = _history_capacity(sg, gv, arrays["density"], arrays["solubility"])
            volume = float(gv.sum(dtype=np.float64))
            values["volume_weighted_gas_saturation"] = (
                float(np.sum(gv.astype(float) * sg) / volume) if volume else None)
            scenarios[variant["id"]] = values
            if variant["publish"]:
                fields[variant["id"]] = sg
        path = folder / f"fields_{frame['frame_index']:04d}.npz"
        np.savez_compressed(path, **fields)
        records.append(dict(frame_index=frame["frame_index"], elapsed_h=frame["elapsed_h"],
            fields=str(path), fields_sha256=io.sha(path), scenarios=scenarios))
    receipt = dict(run=name, frames=records, injection_end_h=injection_end, injection_intervals_h=intervals,
        cell_size_m=spacing, source_prepared_sha256=binding["prepared_sha256"],
        temporal_policy="Only observed continuous gas during active injection can increase occupancy; "
                        "no unobserved intermediate frames or future masks are assumed known.")
    io.write(folder / "receipt.json", receipt)
    io.write(folder / "status.json", dict(stage="complete", run=name, frames=len(records)))
    return name, receipt


def _scenario_summaries(request, variant):
    data = {}
    for name, binding in request["inputs"].items():
        prepared = _snapshot(binding["prepared"], binding["prepared_sha256"])
        rows = [dict(row) for row in prepared["summaries"]]
        if variant["gas_mode"] == "history":
            history = io.read(Path(request["output"]) / "history" / name / "receipt.json")
            if len(history["frames"]) != len(rows):
                raise ValueError("History frame coverage changed")
            for row, frame in zip(rows, history["frames"]):
                if (row["frame_index"] != frame["frame_index"]
                        or abs(row["elapsed_h"] - frame["elapsed_h"]) > 1e-8):
                    raise ValueError("History time/phase identity mismatch")
                row.update(frame["scenarios"][variant["id"]])
        data[name] = rows
    return data


def _fit_variant(request, variant):
    _limits()
    from darsia.fluidflower import concentration_history as model
    output = Path(request["output"]) / "results" / variant["id"]
    output.mkdir(parents=True, exist_ok=True)
    summaries = _scenario_summaries(request, variant)
    kwargs = dict(gas_mode=variant["gas_mode"], gas_saturation=variant.get("gas_saturation"),
                  beta_fixed=variant.get("beta_fixed"), optimization_settings=request["optimization_settings"])
    def update(stage, **values):
        io.write(output / "status.json", dict(stage=stage, variant=variant["id"], **values))
    update("fitting shared model")
    shared = model.fit_summaries(summaries, scope="shared", **kwargs)
    individual = {}
    # Individual profiles reveal whether a lower bound is an optical/gas tradeoff.
    if variant["gas_mode"] != "history" and variant.get("beta_fixed") is None:
        for name, rows in summaries.items():
            update("fitting individual profile", run=name)
            individual[name] = model.fit_summaries({name: rows}, scope="individual", **kwargs)
    fits = dict(shared=shared, individual=individual, variant=variant)
    io.write(output / "fits.json", fits)
    validation = dict(time_folds={}, whole_run_holdout={},
        interpretation="Concentration calibration holdouts only; fixed phase model may have seen these images.",
        scope="shared", history_inputs="causal observed phase maps; no held-out mass targets")
    for mode in ("interleaved", "chronological"):
        update("validating time holdout", mode=mode)
        folds = {name: comparison.validation_indices(rows, mode) for name, rows in summaries.items()}
        train = {name: pair[0] for name, pair in folds.items()}
        fit = model.fit_summaries(summaries, scope="shared", training_indices_by_run=train, **kwargs)
        validation["time_folds"][mode] = dict(
            rows={name: [fit["predictions"][name][i] for i in pair[1]] for name, pair in folds.items()},
            training_indices=train, held_out_indices={name: pair[1] for name, pair in folds.items()},
            shared_parameters=fit["shared_parameters"], diagnostics=fit["diagnostics"])
    for target, rows in summaries.items():
        update("validating held-out experiment", run=target)
        fit = model.fit_summaries({name: r for name, r in summaries.items() if name != target}, scope="shared", **kwargs)
        parameters = next(iter(fit["parameters_by_run"].values()))
        predictions = model.predict_summaries({target: rows}, {target: parameters},
            {target: joint.anchor_from_summaries(rows)}, gas_mode=variant["gas_mode"])[target]
        validation["whole_run_holdout"][target] = dict(rows=predictions, parameters=parameters,
            target_response_fitted=False, target_endpoint_fitted=False, diagnostics=fit["diagnostics"])
    io.write(output / "cross_validation.json", validation)
    update("complete")
    return variant["id"], fits, validation


def full_dose_metrics(predictions, doses):
    """Equal-run errors normalized by fixed final dose, identical across folds."""
    by_run = {}
    for name, rows in predictions.items():
        rows = [row for row in rows if row["nominal_injected_mass_g"] > 0]
        errors = np.array([row["total_mass_g"] - row["nominal_injected_mass_g"] for row in rows])
        if not rows or doses[name] <= 0:
            raise ValueError("Positive held-out observations and final dose required")
        by_run[name] = dict(observations=len(rows), nmae=float(np.mean(abs(errors)) / doses[name]),
                           rmse_g=float(np.sqrt(np.mean(errors**2))))
    return dict(mean_run_nmae=float(np.mean([r["nmae"] for r in by_run.values()])), by_run=by_run)


def _report(request, results):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output = Path(request["output"])
    folder = output / "comparison"
    folder.mkdir(exist_ok=True)
    doses = {name: binding["final_dose_g"] for name, binding in request["inputs"].items()}
    records = []
    for variant in request["variants"]:
        fits, validation = results[variant["id"]]
        record = dict(variant, parameters=fits["shared"]["shared_parameters"],
            training=full_dose_metrics(fits["shared"]["predictions"], doses),
            held_out_run=full_dose_metrics({name: v["rows"] for name, v in validation["whole_run_holdout"].items()}, doses))
        for mode, values in validation["time_folds"].items():
            record[mode] = full_dose_metrics(values["rows"], doses)
        records.append(record)
    io.write(folder / "comparison.json", dict(format=VERSION, variants=records,
        primary_metric="shared interleaved equal-run NMAE divided by full injected dose",
        cohorts=dict(runs=len(doses), frames=sum(v["frames"] for v in request["inputs"].values())),
        no_independent_3d_saturation_measurement=True))
    lines = ["# FF_UM: gassmetning, historikk og utvanning", "",
        "Alle varianter bruker samme 13 forsøk, 261 bilder, fasekart, geometri og optiske referanser uten lys-/ColorChecker-korreksjon.", "",
        "Historikkvarianten er en forhåndsdefinert hypotese, ikke en måling av gassmetning i dybden. "
        "Den bruker 60–90 %, 10 mm indre avstand og tre tidsskalaer (0,5, 2, 6 timer). "
        "Oppbygging tillates bare ved vedvarende, ikke-avtakende synlig gass under injeksjon. "
        "Etter injeksjonsstopp kan samme piksel ikke automatisk få økende metning. "
        "Mellom de utvalgte bildene er tilstanden ukjent.", "",
        "Optisk amplitude, gamma og utvanningsparameter tilpasses på nytt for hvert scenario. "
        "Kontrollen uten utvanning setter beta=0. Ingen ekstra tidlig konsentrasjonsøkning er lagt til.", "",
        "| Variant | Tilpasning | Utelatte tidspunkt | Sene tidspunkt | Utelatt forsøk |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        lines.append("| " + r["label"] + " | " + " | ".join(
            f"{100*r[k]['mean_run_nmae']:.3f} %" for k in ("training", "interleaved", "chronological", "held_out_run")) + " |")
    lines += ["", "Feilene er normalisert med hele injisert dose og vektet likt per forsøk. "
        "Kalibreringsfunksjonen bruker fortsatt den tidligere tidsavhengige normaliseringen. "
        "Utelatte bilder/forsøk gjelder konsentrasjonskalibreringen; faseklassifiseringen er ikke en ny blindtest.", "",
        "God totalmasse dokumenterer ikke lokal konsentrasjon eller fasesplitt. "
        "Ingen variant erstatter automatisk den tidligere modellen. "
        "RUN9 beholder godkjent lånt RUN12-klarvannsreferanse og antatte P/T; RUN11 beholder dokumentert videreføring av siste P/T-måling.", "",
        "Kurver og konsentrasjonskart publiseres i vanlig Studio for de forhåndsvalgte forsøksvariantene. "
        "Tidsskala 0,5 og 6 timer samt enkelte mellomliggende konstante metninger rapporteres kun som sensitivitet."]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    fig = Figure(figsize=(10, 13), layout="constrained"); FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1)
    labels = [r["label"] for r in records]
    for ax, key, title in zip(axes, ("interleaved", "held_out_run", "chronological"),
            ("Held-out times", "Held-out experiments", "Late times fitted from early times")):
        numbers = [100*r[key]["mean_run_nmae"] for r in records]
        ax.barh(labels, numbers, color=["#078b78" if r["gas_mode"] == "history" else "#3572ab" for r in records])
        ax.invert_yaxis(); ax.set_xlabel("Mass error / full injected dose (%)"); ax.set_title(title)
        ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlim(0, max(numbers)*1.17); ax.grid(axis="x", alpha=.2)
    fig.suptitle("FF_UM · gas occupancy and dilution sensitivity\n13 experiments · 261 fixed phase images", fontsize=15)
    fig.savefig(folder / "comparison_mobile.png", dpi=140)
    # A mass profile keeps the optical parameters refitted at each fixed Sg.
    fixed = [r for r in records if r["gas_mode"] == "fixed"]
    fig = Figure(figsize=(8, 8), layout="constrained"); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    for ax, scope in zip(axes, ("shared", "individual")):
        if scope == "shared":
            ax.plot([r["gas_saturation"] for r in fixed],
                [results[r["id"]][0]["shared"]["diagnostics"]["objective"] for r in fixed], "o-", label="Shared fit")
        else:
            for name in request["inputs"]:
                ax.plot([r["gas_saturation"] for r in fixed],
                    [results[r["id"]][0]["individual"][name]["diagnostics"]["objective"] for r in fixed], "o-", label=name.upper())
        ax.set(xlabel="Fixed effective gas saturation", ylabel="Calibration objective", title=scope.capitalize())
        ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    fig.suptitle("Saturation sensitivity · optical response refitted at each point")
    fig.savefig(folder / "gas_profile_mobile.png", dpi=140)
    return records


def _render_run(request, name):
    _limits()
    from darsia.fluidflower.tools.phase_studio_mass import render_frame
    from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
    from darsia.fluidflower.tools.phase_studio_series_mass import export_csv
    root = Path(request["output"])
    binding = request["inputs"][name]
    prepared = _snapshot(binding["prepared"], binding["prepared_sha256"])
    template = _snapshot(binding["previous_result"], binding["previous_result_sha256"])
    validate_template(prepared, template)
    refs = _arrays(prepared["references"], prepared["references_sha256"], request["cache"])
    pv = refs["pv"]
    history = io.read(root / "history" / name / "receipt.json")
    publish = [v for v in request["variants"] if v["publish"]]
    fits = {v["id"]: io.read(root / "results" / v["id"] / "fits.json")["shared"] for v in publish}
    rows = {v["id"]: [] for v in publish}
    for variant in publish:
        (root / "results" / variant["id"] / "shared" / name).mkdir(parents=True, exist_ok=True)
    for position, frame in enumerate(prepared["frames"]):
        io.write(root / "history" / name / "render_status.json", dict(stage="rendering", run=name,
            frame=position+1, frames=len(prepared["frames"])))
        a = _arrays(frame["arrays"], frame["arrays_sha256"], request["cache"])
        h = history["frames"][position]
        if h["frame_index"] != frame["frame_index"] or abs(h["elapsed_h"]-frame["elapsed_h"]) > 1e-8:
            raise ValueError("Rendered history frame identity differs")
        hs = _arrays(h["fields"], h["fields_sha256"], request["cache"])
        for variant in publish:
            key = variant["id"]; fit = fits[key]; p = fit["parameters_by_run"][name]
            factor = joint.dilution_factor(frame["elapsed_h"],
                float(a["av"].sum(dtype=np.float64) + a["gv"].sum(dtype=np.float64)), fit["anchors_by_run"][name], p["beta"])
            # Occupancy does not enter the optical concentration function.
            concentration = joint.concentration_field(a["signal"], dict(p, gas_saturation=.5), factor, a["solubility"])
            sg = hs[key] if variant["gas_mode"] == "history" else float(p["gas_saturation"])
            masses = _history_capacity(sg, a["gv"], a["density"], a["solubility"])
            dissolved = float(np.sum(concentration.astype(float) * a["av"]) * 1000) + masses["residual_history_mass_g"]
            gas = masses["gas_history_mass_g"]; total = dissolved + gas
            predicted = fit["predictions"][name][position]
            if abs(total-predicted["total_mass_g"]) > predicted["summary_mass_error_bound_g"] + max(1e-5, abs(total)*2e-5):
                raise ValueError("Dense mass differs from calibrated histogram: " + key + "/" + name)
            folder = root / "results" / key / "shared" / name
            path = folder / f"fields_{frame['frame_index']:04d}.npz"
            # Reuse source photographs by reference, retaining standalone physical fields.
            np.savez_compressed(path, concentration_g_l=concentration, aqueous_volume_m3=a["av"],
                gas_volume_m3=a["gv"], pore_volume_m3=pv, signal=a["signal"], solubility_g_l=a["solubility"],
                gas_density_g_l=a["density"], gas_saturation=np.asarray(sg, dtype=np.float32),
                aqueous_fraction=np.divide(a["av"], pv, out=np.zeros_like(pv), where=pv>0),
                gas_fraction=np.divide(a["gv"], pv, out=np.zeros_like(pv), where=pv>0))
            preview = folder / f"frame_{frame['frame_index']:04d}.png"
            aq_volume = float(a["av"].sum(dtype=np.float64)); gas_volume = float(a["gv"].sum(dtype=np.float64))
            mean = float(np.sum(concentration.astype(float)*a["av"])/aq_volume) if aq_volume else None
            sg_mean = float(np.sum(a["gv"].astype(float)*sg)/gas_volume) if gas_volume else None
            row = dict(template["frames"][position], fields=str(path), preview=str(preview),
                photograph_arrays=frame["arrays"], photograph_arrays_sha256=frame["arrays_sha256"],
                aqueous_mass_g=dissolved, gas_mass_g=gas, total_mass_g=total,
                residual_liquid_in_gas_g=masses["residual_history_mass_g"],
                gas_saturation_mean=sg_mean, gas_saturation=sg_mean,
                gas_saturation_semantics="gas-support pore-volume-weighted mean; spatial field retained",
                gas_mass_min_g=gas, gas_mass_max_g=gas,
                dilution_factor=factor, mean_aqueous_concentration_g_l=mean,
                total_mass_lower_g=total, total_mass_upper_g=total,
                calibration_role="exploratory shared series fit", quantification_complete=True,
                gas_reference=None, experiment_variant_id=key)
            render_frame(preview, a["rgb"], concentration, a["av"], a["gv"], pv, row,
                name.upper()+" · EXPERIMENT: "+variant["label"], prepared["color_limit_g_l"])
            rows[key].append(row)
    delivered = []
    for variant in publish:
        key=variant["id"]; fit=fits[key]; folder=root/"results"/key/"shared"/name
        local_fit = dict(fit, parameters=fit["parameters_by_run"][name], scope="shared",
            validation_scope="Shared held-out times and leave-one-experiment-out; see cross_validation.json",
            fitting_scope_description="Exploratory occupancy scenario, one shared optical response; no measured 3-D gas saturation.")
        io.write(folder/"calibration.json", local_fit)
        curve=folder/"mass_curve.png"; dilution=folder/"dilution_curve.png"
        io.render_curve(curve, rows[key], name.upper()+" · "+variant["label"], True)
        comparison.common.render_dilution_curve(dilution, rows[key], name.upper()+" · "+variant["label"])
        settings=dict(template["method"]["settings"], experiment_family=FAMILY,
            experiment_variant_id=key, experiment_label=variant["label"],
            gas_mode=variant["gas_mode"], beta_fixed=variant.get("beta_fixed"))
        result=dict(template, format=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
            frames=rows[key], calibration=local_fit, calibration_identity=comparison.digest(fit),
            method=dict(template["method"], settings=settings), mass_curve=str(curve), dilution_curve=str(dilution),
            series_calibration=str(folder/"calibration.json"), cross_validation=str(root/"results"/key/"cross_validation.json"),
            numerical_source_sha256=request["numerical_sources"],
            experiment_variant=variant, source_prepared_sha256=binding["prepared_sha256"],
            field_photograph_policy="RGB reused from photograph_arrays by checksum; quantitative fields stored per result",
            mass_interval_policy="No confidence interval inferred; occupancy alternatives are explicit separate scenarios",
            limitations=[*template.get("limitations", []),
                "Experimental saturation scenarios are assumptions, not independent depth measurements.",
                "Gas history between selected snapshots is unknown; persistence is observed only at sampled times.",
                "Saturated residual water within gas-labelled regions remains an inherited assumption."])
        io.write(folder/"result.json", result); export_csv(result, folder/"mass_per_frame.csv")
        _check_code(request)
        register_variant(folder/"result.json", request["results_root"])
        delivered.append(str(folder/"result.json"))
    io.write(root/"history"/name/"render_status.json", dict(stage="complete", run=name, results=len(delivered)))
    return delivered


def _check_code(request):
    if not request.get("numerical_sources") or any(io.sha(path) != digest for path, digest in request["numerical_sources"].items()):
        raise ValueError("Study numerical sources changed; do not continue or publish")


def create_request(source, output, cache, *, workers=4):
    """Freeze an explicit new study; outputs live beside the series on its NAS."""
    from darsia.fluidflower import concentration_history, gas_history
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new study directory")
    inputs = {}
    for name in RUNS:
        path = source / "preparation" / name / "prepared.json"
        result = source / "shared" / name / "result.json"
        prepared = io.read(path); template = io.read(result)
        validate_template(prepared, template)
        inputs[name] = dict(prepared=str(path), prepared_sha256=io.sha(path),
            previous_result=str(result), previous_result_sha256=io.sha(result),
            frames=len(prepared["frames"]), final_dose_g=prepared["expected_final_injected_mass_g"])
    sources = {str(Path(path).resolve()): io.sha(path) for path in
        (__file__, joint.__file__, concentration_history.__file__, gas_history.__file__, comparison.__file__)}
    request = dict(format=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), output=str(output), cache=str(Path(cache).resolve()),
        results_root=str(source.parents[3]), inputs=inputs, variants=variants(),
        numerical_sources=sources, workers=workers, threads_per_worker=1,
        optimization_settings=dict(start_count=12, maxiter=360, maxfev=20000, xtol=1e-6, ftol=1e-8),
        study_design=dict(primary_metric="shared interleaved held-out full-dose equal-run NMAE",
            other_checks=["chronological holdout", "whole-experiment holdout", "individual fixed-Sg profiles"],
            phase_maps_changed=False, automatic_promotion=False,
            history_parameters="predeclared unmeasured sensitivity choices",
            dilution="Existing volume dilution fitted, plus beta=0 control; no extra temporal multiplier"))
    if Path(request["results_root"]).name != "concentration_mass":
        raise ValueError("Expected a series concentration_mass/jobs/<job>/results/<branch> source")
    output.mkdir(parents=True)
    io.write(output/"request.json", request)
    return output/"request.json"


def run(request_path):
    request_path=Path(request_path); request=io.read(request_path); output=Path(request["output"])
    start=time.perf_counter()
    def update(stage, **values):
        io.write(output/"status.json", dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter()-start, **values))
    try:
        _check_code(request)
        for binding in request["inputs"].values():
            validate_template(_snapshot(binding["prepared"], binding["prepared_sha256"]),
                              _snapshot(binding["previous_result"], binding["previous_result_sha256"]))
        if set(request["inputs"]) != set(RUNS):
            raise ValueError("This predeclared FF_UM comparison requires all thirteen runs")
        if sum(v["frames"] for v in request["inputs"].values()) != 261:
            raise ValueError("Fixed 261-frame cohort required")
        workers=max(1, min(int(request.get("workers", 4)), os.cpu_count() or 1))
        _limits(); update("preparing causal history", workers=workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_prepare_run, request, name): name for name in request["inputs"]}
            for count, future in enumerate(as_completed(futures), 1):
                name, receipt=future.result(); update("preparing causal history", completed_runs=count, run=name, workers=workers)
        preparation=time.perf_counter()-start; fit_start=time.perf_counter(); results={}
        update("fitting and validating", completed_variants=0, variants=len(request["variants"]))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_fit_variant, request, v): v["id"] for v in request["variants"]}
            for count, future in enumerate(as_completed(futures), 1):
                key, fits, validation=future.result(); results[key]=(fits, validation)
                update("fitting and validating", completed_variants=count, variants=len(futures), variant=key)
        fit_seconds=time.perf_counter()-fit_start
        records=_report(request, results); render_start=time.perf_counter()
        update("publishing Studio comparisons", completed_runs=0)
        delivered=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_render_run, request, name): name for name in request["inputs"]}
            for count, future in enumerate(as_completed(futures), 1):
                delivered.extend(future.result()); update("publishing Studio comparisons", completed_runs=count,
                    run=futures[future], completed_results=len(delivered))
        expected=len(RUNS)*sum(v["publish"] for v in request["variants"])
        if len(delivered)!=expected or any(not Path(p).is_file() for p in delivered):
            raise ValueError("Studio delivery coverage incomplete")
        _check_code(request)
        io.write(output/"delivery.json", dict(complete=True, results=delivered, runs=list(RUNS), frames=261,
            published_variants=[v for v in request["variants"] if v["publish"]], comparison=str(output/"comparison"/"report.md")))
        timing=dict(preparation_seconds=preparation, fitting_validation_seconds=fit_seconds,
                    studio_render_seconds=time.perf_counter()-render_start, elapsed_seconds=time.perf_counter()-start)
        io.write(output/"timing.json", timing)
        update("complete", results=len(delivered), timing=timing)
    except BaseException as error:
        update("failed", error=str(error), traceback=traceback.format_exc())
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    run(parser.parse_args().request)
