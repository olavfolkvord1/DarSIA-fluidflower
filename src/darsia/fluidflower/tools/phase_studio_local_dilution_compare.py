"""Controlled local-dilution hypotheses on immutable FF_UM optical inputs.

Local exposure uses observed optical/phase history, never injected mass. Mass
targets calibrate one shared response on explicit training folds. The study is
experimental and never promotes a model or changes phase classifications.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.ndimage import gaussian_filter

from darsia.fluidflower import concentration_joint as joint
from darsia.fluidflower.tools import phase_studio_gas_history_compare as previous
from darsia.fluidflower.tools import phase_studio_native_mass as io
from darsia.fluidflower.tools import phase_studio_joint_calibration_compare as comparison

VERSION = "local-optical-dilution-study-v1"
FAMILY = "local_optical_dilution_v1"
RUNS = previous.RUNS


def full_depth_settings():
    """An explicit visible-only, causal depth-saturation sensitivity hypothesis.

    Three observations at the same pixel establish full-depth occupancy only
    after at least 0.5 h, with a gap no longer than 24 h. The 75% baseline is
    the existing fixed-gas comparison. No extra 3-D cells are inferred.
    """
    from darsia.fluidflower.hidden_gas_depth import HiddenGasDepthSettings
    return asdict(HiddenGasDepthSettings(
        causal_visible_full_depth_enabled=True,
        corridor_enabled=False,
        persistent_halo_enabled=False,
        visible_saturation_min=.75,
        full_depth_required_observations=3,
        full_depth_min_persistence_h=.5,
        full_depth_max_observation_gap_h=24.,
        full_depth_min_visible_gas_fraction=.5,
    ))


def variants(*, include_persistent_full_depth=False):
    gas_labels = {"fixed75": "constant gas 75%", "existing_history": "injection-gated gas history",
                  "persistent_core": "persistent local gas core"}
    result = [dict(id=f"{mode}_{gas}", label=f"{mode.title()} dilution + {label}",
                   dilution_mode=mode, gas_mode=gas, publish=True)
              for gas, label in gas_labels.items() for mode in ("global", "local")]
    result.append(dict(id="none_fixed75", label="Optical response only + constant gas 75%",
                       dilution_mode="none", gas_mode="fixed75", publish=True))
    if include_persistent_full_depth:
        result.extend(dict(id=f"{mode}_persistent_full_depth",
                           label=f"{mode.title()} dilution + persistent visible gas at full depth",
                           dilution_mode=mode, gas_mode="persistent_full_depth", publish=True)
                      for mode in ("global", "local"))
    return result


def _write_gzip(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as out:
        json.dump(value, out, allow_nan=False, separators=(",", ":"))


def _read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _weighted(values, weights):
    volume = float(np.sum(weights, dtype=np.float64))
    return float(np.sum(np.asarray(values, float) * weights) / volume) if volume > 0 else None


def _spacing(prepared, shape):
    dimensions = prepared["dimensions_m"]
    return tuple(float(dimensions[key])/n for key, n in zip(("height", "width"), shape))


def _prepare_run(request, name):
    previous._limits()
    from darsia.fluidflower.concentration_local_history import LocalDilutionTracker, PersistentGasCoreTracker
    from darsia.fluidflower import concentration_local as model
    from darsia.fluidflower.gas_history import GasHistoryTracker
    from darsia.fluidflower.hidden_gas_depth import (
        HiddenGasDepthSettings, VisibleFullDepthSaturationTracker)
    root = Path(request["output"]); binding = request["inputs"][name]
    prepared = previous._snapshot(binding["prepared"], binding["prepared_sha256"])
    refs = previous._arrays(prepared["references"], prepared["references_sha256"], request["cache"])
    pv = refs["pv"]; spacing = _spacing(prepared, pv.shape)
    settings = request["local_settings"]; gas_settings = request["core_settings"]
    local = LocalDilutionTracker(spacing, **settings)
    core = PersistentGasCoreTracker(spacing, **gas_settings)
    intervals = previous._injection_intervals(prepared["binding"]["injection_stages"])
    old = GasHistoryTracker(spacing, low=.6, high=.9, tau_h=2., core_mm=10., injection_intervals_h=intervals)
    include_full_depth = any(v["gas_mode"] == "persistent_full_depth" for v in request["variants"])
    full_depth = (VisibleFullDepthSaturationTracker(HiddenGasDepthSettings(
        **request["full_depth_settings"])) if include_full_depth else None)
    sensitivity = {
        "radius_10mm": LocalDilutionTracker(spacing, **dict(settings, radius_m=.01)),
        "radius_40mm": LocalDilutionTracker(spacing, **dict(settings, radius_m=.04)),
        "signal_scale_2": LocalDilutionTracker(spacing, **dict(settings, signal_scale=2.)),
        "signal_scale_4": LocalDilutionTracker(spacing, **dict(settings, signal_scale=4.)),
        "gap_24h": LocalDilutionTracker(spacing, **dict(settings, max_gap_h=24.)),
    }
    core_sensitivity = {f"core_tau_{tau}": PersistentGasCoreTracker(spacing,
        **dict(gas_settings, tau_h=tau)) for tau in (.5, 6.)}
    thin_local = LocalDilutionTracker(spacing, **settings)
    thin_core = PersistentGasCoreTracker(spacing, **gas_settings)
    sums = {key: np.zeros(pv.shape, np.float64) for key in ("gas_contact", "weak_signal", "strong_signal", "gas_retreat")}
    previous_gas = np.zeros(pv.shape, bool)
    folder = root / "history" / name; folder.mkdir(parents=True, exist_ok=True)
    rows, frames, sensitivity_rows = [], [], []
    for position, frame in enumerate(prepared["frames"]):
        io.write(folder / "status.json", dict(stage="local history", frame=position+1, frames=len(prepared["frames"])))
        a = previous._arrays(frame["arrays"], frame["arrays_sha256"], request["cache"])
        exposure, diag = local.update(a["signal"], a["av"], a["gv"], pv, frame["elapsed_h"])
        fraction = np.clip(np.divide(a["gv"], pv, out=np.zeros_like(pv), where=pv>0), 0, 1)
        sg_core, core_diag = core.update(fraction, frame["elapsed_h"])
        sg_old, _ = old.update(fraction, frame["elapsed_h"], max(end for _, end in intervals))
        gas_scenarios = {"fixed75": .75, "existing_history": sg_old, "persistent_core": sg_core}
        if full_depth is not None:
            saturation = full_depth.update(
                fraction,
                elapsed_h=frame["elapsed_h"],
                injected_mass_g=frame["injected_mass_g"],
            )
            gas_scenarios["persistent_full_depth"] = saturation
        terms = {key: previous._history_capacity(sg, a["gv"], a["density"], a["solubility"])
                 for key, sg in gas_scenarios.items()}
        row = model.add_local_histogram(prepared["summaries"][position], a["signal"], a["av"],
            a["solubility"], exposure, exposure_bins=64, max_exposure=settings["max_exposure"])
        row["gas_scenarios"] = terms
        rows.append(row)
        target = folder / f"fields_{frame['frame_index']:04d}.npz"
        np.savez_compressed(target, exposure=exposure, **{k: v for k, v in gas_scenarios.items()
            if k != "fixed75"})
        frames.append(dict(frame_index=frame["frame_index"], elapsed_h=frame["elapsed_h"],
            source_name=frame["source_name"], fields=str(target), fields_sha256=io.sha(target),
            history_reset=diag["history_reset"],exposure_clipped_fraction=diag["exposure_clipped_fraction"],
            mean_local_exposure=_weighted(exposure,a["av"])))
        aq = a["av"] > 0
        # ROI selection is diagnostic only, based on inputs before any fit.
        near = gaussian_filter(fraction.astype(np.float32), sigma=tuple(.01/s for s in spacing))
        sums["gas_contact"] += aq * near
        sums["weak_signal"] += aq * ((a["signal"] > .05) & (a["signal"] < .75))
        sums["strong_signal"] += aq * (a["signal"] >= 1.5)
        sums["gas_retreat"] += aq * previous_gas
        previous_gas = fraction >= .5
        checks = {}
        for key, tracker in sensitivity.items():
            other, _ = tracker.update(a["signal"], a["av"], a["gv"], pv, frame["elapsed_h"])
            checks[key] = dict(exposure_abs_difference=_weighted(abs(other-exposure), a["av"]),
                               factor_beta1_abs_difference=_weighted(abs(np.exp(-other)-np.exp(-exposure)), a["av"]))
        for key, tracker in core_sensitivity.items():
            other, _ = tracker.update(fraction, frame["elapsed_h"])
            checks[key] = dict(saturation_abs_difference=_weighted(abs(other-sg_core), a["gv"]))
        if position % 2 == 0 or position == len(prepared["frames"])-1:
            other, _ = thin_local.update(a["signal"], a["av"], a["gv"], pv, frame["elapsed_h"])
            other_core, _ = thin_core.update(fraction, frame["elapsed_h"])
            checks["every_second_frame"] = dict(exposure_abs_difference=_weighted(abs(other-exposure), a["av"]),
                factor_beta1_abs_difference=_weighted(abs(np.exp(-other)-np.exp(-exposure)), a["av"]),
                saturation_abs_difference=_weighted(abs(other_core-sg_core), a["gv"]))
        sensitivity_rows.append(dict(frame_index=frame["frame_index"], elapsed_h=frame["elapsed_h"], checks=checks))
    rois = []
    for key, score in sums.items():
        if score.max() <= 0:
            continue
        smooth = gaussian_filter(score, sigma=tuple(.01/s for s in spacing))
        smooth[pv <= 0] = -1
        y, x = map(int, np.unravel_index(np.argmax(smooth), pv.shape))
        dy, dx = (max(1, int(round(.015/s))) for s in spacing)
        rois.append(dict(id=key, bounds=[max(0,y-dy), min(pv.shape[0],y+dy+1),
                                       max(0,x-dx), min(pv.shape[1],x+dx+1)], width_m=.03))
    _write_gzip(folder / "summaries.json.gz", rows)
    receipt = dict(format=VERSION, run=name, frames=frames, rois=rois, spacing_m=spacing,
        roi_selection="Fixed 30 mm diagnostic windows from input optical/phase history; no mass targets; not training labels.",
        summaries_sha256=io.sha(folder/"summaries.json.gz"), source_prepared_sha256=binding["prepared_sha256"])
    io.write(folder / "receipt.json", receipt)
    io.write(folder / "sensitivity.json", dict(parameters=request["local_settings"], frames=sensitivity_rows,
        interpretation="Fixed-input cue sensitivity before calibration; not independent measurement or a confidence interval."))
    io.write(folder / "status.json", dict(stage="complete", frames=len(frames)))
    return name


def _scenario_summaries(request, variant):
    data = {}
    for name in request["inputs"]:
        folder = Path(request["output"])/"history"/name
        receipt = io.read(folder/"receipt.json")
        if io.sha(folder/"summaries.json.gz") != receipt["summaries_sha256"]:
            raise ValueError("Prepared local histogram changed")
        rows = _read_gzip(folder/"summaries.json.gz")
        for row in rows:
            row.update(row.pop("gas_scenarios")[variant["gas_mode"]])
        data[name] = rows
    return data


def _fit_variant(request, variant):
    previous._limits()
    from darsia.fluidflower import concentration_local as model
    folder = Path(request["output"])/"results"/variant["id"]; folder.mkdir(parents=True, exist_ok=True)
    summaries = _scenario_summaries(request, variant)
    kwargs = dict(dilution_mode=variant["dilution_mode"], optimization_settings=request["optimization_settings"])
    def status(stage, **more):
        io.write(folder/"status.json", dict(stage=stage, variant=variant["id"], **more))
    status("shared calibration")
    shared = model.fit_summaries(summaries, **kwargs)
    control = previous._snapshot(request["fixed_parameter_control"], request["fixed_parameter_control_sha256"])["shared"]
    frozen_parameters = {name: dict(parameters, gas_saturation=None,
        beta=0. if variant["dilution_mode"] == "none" else parameters["beta"])
        for name, parameters in control["parameters_by_run"].items()}
    frozen = model.predict_summaries(summaries, frozen_parameters, control["anchors_by_run"],
                                     dilution_mode=variant["dilution_mode"])
    fits = dict(shared=shared, fixed_optical_predictions=frozen, variant=variant)
    io.write(folder/"fits.json", fits)
    validation = dict(time_folds={}, whole_run_holdout={},
        interpretation="Concentration-target holdouts only. Fixed phase maps may have been trained on these images; observed history remains an input.")
    for mode in ("interleaved", "chronological"):
        status("time holdout", mode=mode)
        folds = {name: comparison.validation_indices(rows, mode) for name, rows in summaries.items()}
        training = {name: value[0] for name, value in folds.items()}
        fit = model.fit_summaries(summaries, training_indices_by_run=training, **kwargs)
        validation["time_folds"][mode] = dict(rows={name:[fit["predictions"][name][i] for i in value[1]]
            for name, value in folds.items()}, training_indices=training,
            held_out_indices={name:value[1] for name,value in folds.items()},
            parameters=fit["shared_parameters"], diagnostics=fit["diagnostics"])
    for target, rows in summaries.items():
        status("held-out experiment", run=target)
        fit = model.fit_summaries({name:value for name,value in summaries.items() if name != target}, **kwargs)
        p = next(iter(fit["parameters_by_run"].values()))
        predicted = model.predict_summaries({target:rows}, {target:p}, {target:joint.anchor_from_summaries(rows)},
                                            dilution_mode=variant["dilution_mode"])[target]
        validation["whole_run_holdout"][target] = dict(rows=predicted, parameters=p,
            target_response_fitted=False, target_endpoint_fitted=False, diagnostics=fit["diagnostics"])
    io.write(folder/"cross_validation.json", validation); status("complete")
    return variant["id"], fits, validation


def numeric_preview(path, values, mask, title, low, high, rois=()):
    """Pre-render fields off the UI thread with a common scale across frames."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import colormaps
    values = np.broadcast_to(values, mask.shape)
    mapped = colormaps["viridis"](np.clip((values-low)/(high-low),0,1), bytes=True)[...,:3]
    mapped[~mask] = [232,237,242]
    panel = Image.fromarray(mapped)
    canvas = Image.new("RGB", (panel.width, panel.height+115), "white"); canvas.paste(panel,(0,55))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23) if os.name=="nt" else ImageFont.load_default()
    draw.text((15,10),title,fill="black",font=font)
    for roi in rois:
        y0,y1,x0,x1=roi["bounds"]
        draw.rectangle((x0,y0+55,x1,y1+55),outline="white",width=2)
        draw.text((x0,y0+55),roi["id"],fill="black",font=font,stroke_width=1,stroke_fill="white")
    ramp=np.tile(np.linspace(0,1,max(1,panel.width-200)),(15,1))
    bar=Image.fromarray(colormaps["viridis"](ramp,bytes=True)[...,:3]); canvas.paste(bar,(100,panel.height+66))
    draw.text((100,panel.height+82),f"{low:g}",fill="black",font=font)
    draw.text((panel.width-150,panel.height+82),f"{high:g}",fill="black",font=font)
    canvas.save(path)


def _frame_diagnostics(a, refs, c, factor, sg, exposure, rois, spacing):
    av,gv=a["av"],a["gv"]
    values={"mean_factor":_weighted(factor,av), "mean_exposure":_weighted(exposure,av),
            "mean_sg":_weighted(sg,gv), "rois":{}}
    values["factor_spatial_std"] = (float(np.sqrt(_weighted(
        (np.asarray(factor)-values["mean_factor"])**2,av)))
        if values["mean_factor"] is not None else None)
    for roi in rois:
        y0,y1,x0,x1=roi["bounds"]; sl=np.s_[y0:y1,x0:x1]
        values["rois"][roi["id"]]={"concentration_g_l":_weighted(c[sl],av[sl]),
            "signal":_weighted(a["signal"][sl],av[sl]), "factor":_weighted(np.broadcast_to(factor,av.shape)[sl],av[sl]),
            "sg":_weighted(np.broadcast_to(sg,av.shape)[sl],gv[sl]),
            "aqueous_volume_m3":float(av[sl].sum(dtype=np.float64))}
    # Different-facies jumps are diagnostics, not automatically errors: real
    # transport barriers may follow the layers too.
    gradients={"within_facies":[],"across_facies":[]}
    for axis,step in enumerate(spacing):
        left=[slice(None),slice(None)];right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        x,y=tuple(left),tuple(right); valid=(av[x]>0)&(av[y]>0)
        different=refs["facies"][x]!=refs["facies"][y]
        diff=abs(c[x]-c[y])/step
        for key,choose in (("within_facies",valid&~different),("across_facies",valid&different)):
            if np.any(choose): gradients[key].extend([float(diff[choose].sum()),int(choose.sum())])
    values["mean_gradient_g_l_per_m"]={key:sum(v[0::2])/sum(v[1::2]) if v else None for key,v in gradients.items()}
    return values


def _diagnostic_curve(path, rows, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(9,12),layout="constrained");FigureCanvasAgg(fig);axes=fig.subplots(4,1)
    times=[r["elapsed_h"] for r in rows]
    ids=sorted({key for r in rows for key in r["local_diagnostics"]["rois"]})
    for ax,key,label in zip(axes,("concentration_g_l","signal","factor","sg"),
        ("Dissolved concentration (g/L)","Reference-normalized optical signal","Local attenuation (1 = none)","Effective gas saturation")):
        for roi in ids:
            ax.plot(times,[r["local_diagnostics"]["rois"].get(roi,{}).get(key,np.nan) for r in rows],".-",label=roi)
        ax.set(ylabel=label,xlabel="Hours");ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    axes[2].set_ylim(0,1.02);axes[3].set_ylim(.45,1.02)
    fig.suptitle(title+"\nFixed diagnostic windows; no independent concentration targets",fontsize=12)
    fig.savefig(path,dpi=140)


def _render_run(request,name):
    previous._limits()
    from darsia.fluidflower import concentration_local as model
    from darsia.fluidflower.tools.phase_studio_mass import render_frame
    from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
    from darsia.fluidflower.tools.phase_studio_series_mass import export_csv
    root=Path(request["output"]);binding=request["inputs"][name]
    prepared=previous._snapshot(binding["prepared"],binding["prepared_sha256"])
    template=previous._snapshot(binding["previous_result"],binding["previous_result_sha256"])
    previous.validate_template(prepared,template)
    refs=previous._arrays(prepared["references"],prepared["references_sha256"],request["cache"]);pv=refs["pv"]
    history=io.read(root/"history"/name/"receipt.json")
    fits={v["id"]:io.read(root/"results"/v["id"]/"fits.json")["shared"] for v in request["variants"]}
    rows={v["id"]:[] for v in request["variants"]}
    for position,frame in enumerate(prepared["frames"]):
        io.write(root/"history"/name/"render_status.json",dict(stage="rendering",frame=position+1,frames=len(prepared["frames"])))
        a=previous._arrays(frame["arrays"],frame["arrays_sha256"],request["cache"])
        h=history["frames"][position]
        if h["frame_index"]!=frame["frame_index"] or h["source_name"]!=frame["source_name"]:
            raise ValueError("Local history frame identity changed")
        fields=previous._arrays(h["fields"],h["fields_sha256"],request["cache"])
        for variant in request["variants"]:
            key=variant["id"];fit=fits[key];p=fit["parameters_by_run"][name]
            exposure=fields["exposure"];mode=variant["dilution_mode"]
            global_factor=joint.dilution_factor(frame["elapsed_h"],float(a["av"].sum(dtype=np.float64)+a["gv"].sum(dtype=np.float64)),
                fit["anchors_by_run"][name],p["beta"]) if mode=="global" else None
            factor=np.exp(-p["beta"]*exposure) if mode=="local" else (global_factor if mode=="global" else 1.)
            c=model.dense_concentration(a["signal"],a["solubility"],p,exposure,dilution_mode=mode,global_factor=global_factor)
            sg=.75 if variant["gas_mode"]=="fixed75" else fields[variant["gas_mode"]]
            masses=previous._history_capacity(sg,a["gv"],a["density"],a["solubility"])
            gas=masses["gas_history_mass_g"];dissolved=float(np.sum(c.astype(float)*a["av"])*1000)+masses["residual_history_mass_g"]
            total=gas+dissolved;prediction=fit["predictions"][name][position]
            if abs(total-prediction["total_mass_g"])>prediction["summary_mass_error_bound_g"]+max(1e-5,abs(total)*2e-5):
                raise ValueError("Dense/histogram mass mismatch: "+name+"/"+key)
            if not np.isfinite(c).all() or np.any(c < 0) or np.any(c > a["solubility"]+1e-5):
                raise ValueError("Concentration outside physical solubility bound")
            folder=root/"results"/key/"shared"/name;folder.mkdir(parents=True,exist_ok=True)
            stem=f"{frame['frame_index']:04d}";path=folder/f"fields_{stem}.npz"
            np.savez_compressed(path,concentration_g_l=c,aqueous_volume_m3=a["av"],gas_volume_m3=a["gv"],
                pore_volume_m3=pv,signal=a["signal"],solubility_g_l=a["solubility"],gas_density_g_l=a["density"],
                gas_saturation=np.asarray(sg,np.float32),dilution_factor=np.asarray(factor,np.float32),local_exposure=exposure,
                aqueous_fraction=np.divide(a["av"],pv,out=np.zeros_like(pv),where=pv>0),
                gas_fraction=np.divide(a["gv"],pv,out=np.zeros_like(pv),where=pv>0))
            preview=folder/f"frame_{stem}.png";gas_preview=folder/f"gas_saturation_{stem}.png";dilution_preview=folder/f"dilution_{stem}.png"
            diag=_frame_diagnostics(a,refs,c,factor,sg,exposure,history["rois"],history["spacing_m"])
            row=dict(template["frames"][position],fields=str(path),preview=str(preview),
                gas_saturation_preview=str(gas_preview),local_dilution_preview=str(dilution_preview),
                photograph_arrays=frame["arrays"],photograph_arrays_sha256=frame["arrays_sha256"],
                aqueous_mass_g=dissolved,gas_mass_g=gas,total_mass_g=total,
                residual_liquid_in_gas_g=masses["residual_history_mass_g"],gas_saturation_mean=diag["mean_sg"],gas_saturation=diag["mean_sg"],
                gas_mass_min_g=gas,gas_mass_max_g=gas,dilution_factor=diag["mean_factor"],
                mean_aqueous_concentration_g_l=_weighted(c,a["av"]),total_mass_lower_g=total,total_mass_upper_g=total,
                calibration_role="experimental shared series fit",quantification_complete=True,gas_reference=None,
                experiment_variant_id=key,local_diagnostics=diag)
            render_frame(preview,a["rgb"],c,a["av"],a["gv"],pv,row,name.upper()+" · "+variant["label"],prepared["color_limit_g_l"])
            numeric_preview(gas_preview,sg,a["gv"]>0,name.upper()+f" · {frame['elapsed_h']:.2f} h · Assumed gas saturation",0,1)
            numeric_preview(dilution_preview,factor,a["av"]>0,name.upper()+f" · {frame['elapsed_h']:.2f} h · Attenuation: 1 = none",0,1,history["rois"])
            rows[key].append(row)
    delivered=[]
    for variant in request["variants"]:
        key=variant["id"];fit=fits[key];folder=root/"results"/key/"shared"/name
        local_fit=dict(fit,parameters=fit["parameters_by_run"][name],scope="shared")
        io.write(folder/"calibration.json",local_fit)
        curve=folder/"mass_curve.png";dilution=folder/"dilution_curve.png";diagnostics=folder/"local_diagnostics.png"
        io.render_curve(curve,rows[key],name.upper()+" · "+variant["label"],True)
        comparison.common.render_dilution_curve(dilution,rows[key],name.upper()+" · "+variant["label"])
        _diagnostic_curve(diagnostics,rows[key],name.upper()+" · "+variant["label"])
        settings=dict(template["method"]["settings"],experiment_family=FAMILY,experiment_variant_id=key,
            experiment_label=variant["label"],dilution_mode=variant["dilution_mode"],gas_mode=variant["gas_mode"])
        if variant["gas_mode"] == "persistent_full_depth":
            settings["visible_full_depth_rule"] = request["full_depth_settings"]
        result=dict(template,format=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),frames=rows[key],
            calibration=local_fit,calibration_identity=comparison.digest(fit),method=dict(template["method"],settings=settings),
            mass_curve=str(curve),dilution_curve=str(dilution),local_diagnostics_curve=str(diagnostics),
            series_calibration=str(folder/"calibration.json"),cross_validation=str(root/"results"/key/"cross_validation.json"),
            numerical_source_sha256=request["numerical_sources"],experiment_variant=variant,
            source_prepared_sha256=binding["prepared_sha256"],diagnostic_regions=history["rois"],
            field_photograph_policy="Reference to checksum-bound source RGB; fields are standalone numerical arrays",
            mass_interval_policy="Point estimates only; scenarios are not confidence bounds",
            limitations=[*template.get("limitations",[]),"Local dilution and gas persistence are hypotheses, not measured 3-D state or a transport solution.",
                "Sparse image continuity is not fluid residence time. No future frames enter the history update.",
                "Local attenuation replaces global dilution; no frame is forced to the injected mass.",
                "Residual liquid in gas-labelled pore volume remains assumed CO2-saturated."])
        io.write(folder/"result.json",result);export_csv(result,folder/"mass_per_frame.csv")
        previous._check_code(request);register_variant(folder/"result.json",request["results_root"])
        delivered.append(str(folder/"result.json"))
    io.write(root/"history"/name/"render_status.json",dict(stage="complete",results=len(delivered)))
    return delivered


def _report(request, results):
    folder=Path(request["output"])/"comparison";folder.mkdir(exist_ok=True)
    doses={name:b["final_dose_g"] for name,b in request["inputs"].items()};records=[]
    for variant in request["variants"]:
        fits,validation=results[variant["id"]]
        row=dict(variant,parameters=fits["shared"]["shared_parameters"],
            training=previous.full_dose_metrics(fits["shared"]["predictions"],doses),
            fixed_optics=previous.full_dose_metrics(fits["fixed_optical_predictions"],doses),
            held_out_run=previous.full_dose_metrics({name:v["rows"] for name,v in validation["whole_run_holdout"].items()},doses))
        for mode,values in validation["time_folds"].items():row[mode]=previous.full_dose_metrics(values["rows"],doses)
        records.append(row)
    io.write(folder/"comparison.json",dict(format=VERSION,variants=records,runs=13,frames=261,
        metric="Equal-run absolute mass error divided by full injected dose; only positive-dose frames"))
    lines=["# FF_UM: lokal optisk utvanning og vedvarende gasskjerne","",
        "Samme 13 forsøk og 261 faste fasekart som referansen. Ingen lys-/ColorChecker-korreksjon. ",
        "Lokal demping erstatter global volumfaktor og bruker ingen injisert masse som direkte korreksjon. ",
        "Lokal eksponering er en kausal hypotese basert på vekst i nærområdets påvirkede poreandel, svakere fargestøtte og fravær av nær gasskontakt. ",
        "Gasskjernen testes både med tidligere injeksjonsport og med lokal vedvarende kjerne uten slik port. ",
        "Parametre og romlige skalaer er antakelser. Bildehistorikk er ikke vannets oppholdstid eller en måling av metning i dybden.","",
        "| Variant | Faste optiske parametre | Ny tilpasning | Utelatte tidspunkt | Sene tidspunkt | Utelatt forsøk |",
        "|---|---:|---:|---:|---:|---:|"]
    if any(v["gas_mode"] == "persistent_full_depth" for v in request["variants"]):
        lines[0] = "# FF_UM: lokal optisk utvanning og vedvarende synlig gass i full dybde"
        lines.insert(7, "Full dybde er en eksplisitt følsomhetstest: bare synlige gassceller med minst "
            "50 % gassandel i tre påfølgende bilder og minst 0,5 timer mellom første og tredje observasjon "
            "får metning 1,0 i dybden. Før dette brukes 0,75. Et gap over 24 timer nullstiller rekken. "
            "Ingen gass legges til utenfor synlige gassceller; ingen 3D-forbindelser eller halo antas.")
        lines.insert(8, "")
    for r in records:lines.append("| "+r["label"]+" | "+" | ".join(f"{100*r[k]['mean_run_nmae']:.3f} %" for k in
        ("fixed_optics","training","interleaved","chronological","held_out_run"))+" |")
    lines += ["","Totalmasse alene validerer ikke konsentrasjonsgradient eller fasesplitt. Eksperiment-holdout gjelder konsentrasjonsmål, ikke en ny blind faseklassifisering.",
        "Faste 30 mm diagnosevinduer ble valgt fra optikk/faseutvikling før tilpasning; de er ikke treningsbokser. ",
        "Lokale kurver, gassmetningskart og dempingskart finnes i vanlig Studio. Sensitiviteter ligger under history/<run>/sensitivity.json.",
        "RUN9 og RUN11 beholder samme dokumenterte referanse-/P/T-antakelser som kontrollen. Ingen variant aktiveres automatisk som standard."]
    (folder/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(9,12),layout="constrained");FigureCanvasAgg(fig);axes=fig.subplots(3,1)
    for ax,key,title in zip(axes,("interleaved","held_out_run","chronological"),("Held-out times","Held-out experiments","Late times fitted from early times")):
        ax.barh([r["label"] for r in records],[100*r[key]["mean_run_nmae"] for r in records],
            color=["#078b78" if r["dilution_mode"]=="local" else "#3572ab" for r in records])
        ax.set(title=title,xlabel="Absolute error / full injected dose (%)");ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    fig.savefig(folder/"comparison_mobile.png",dpi=140)


def _local_report(request):
    """Compare identical diagnostic windows and retain separate phase masses."""
    root=Path(request["output"]); folder=root/"comparison"
    paired=[]; sensitivity={}; split={}
    variant_ids = {v["id"] for v in request["variants"]}
    paired_modes = sorted({v["gas_mode"] for v in request["variants"] if
                           f"global_{v['gas_mode']}" in variant_ids
                           and f"local_{v['gas_mode']}" in variant_ids})
    for name in RUNS:
        data={v["id"]:io.read(root/"results"/v["id"]/"shared"/name/"result.json")
              for v in request["variants"]}
        split[name]={key: {field:value["frames"][-1][field] for field in
            ("gas_mass_g","aqueous_mass_g","total_mass_g","gas_saturation_mean")}
            for key,value in data.items()}
        for gas in paired_modes:
            global_rows=data[f"global_{gas}"]["frames"]; local_rows=data[f"local_{gas}"]["frames"]
            for g,l in zip(global_rows,local_rows):
                if g["frame_index"]!=l["frame_index"] or g["elapsed_h"]!=l["elapsed_h"]:
                    raise ValueError("Diagnostic comparison frame mismatch")
                gd,ld=g["local_diagnostics"],l["local_diagnostics"]
                paired.append(dict(run=name,gas_mode=gas,frame_index=g["frame_index"],elapsed_h=g["elapsed_h"],
                    global_diagnostics=gd,local_diagnostics=ld,
                    gas_mass_g=l["gas_mass_g"],global_aqueous_mass_g=g["aqueous_mass_g"],
                    local_aqueous_mass_g=l["aqueous_mass_g"]))
        values={}
        for frame in io.read(root/"history"/name/"sensitivity.json")["frames"]:
            for key,fields in frame["checks"].items():
                for field,value in fields.items():
                    if value is not None:values.setdefault(key+"/"+field,[]).append(value)
        sensitivity[name]={key:dict(mean=float(np.mean(v)),maximum=max(v),frames=len(v)) for key,v in values.items()}
    aggregate={key:dict(mean_of_run_means=float(np.mean([v[key]["mean"] for v in sensitivity.values() if key in v])),
        maximum_frame_difference=max(v[key]["maximum"] for v in sensitivity.values() if key in v))
        for key in sorted({key for v in sensitivity.values() for key in v})}
    io.write(folder/"local_diagnostics_comparison.json",dict(paired_frames=paired,final_phase_split=split,
        sensitivity_by_run=sensitivity,sensitivity_aggregate=aggregate,
        interpretation="Optical/phase-derived diagnostics, not independent measurements of concentration or 3-D saturation. Sensitivity attenuation uses beta=1, without refitting."))
    lines=["", "## Lokal utvikling og fasesplitt", "",
        "Alle varianter har de samme fasedelte porevolumene. Gass- og væskemasse lagres separat for hvert bilde; forskjellene mellom gasshypotesene er ikke observert fasit.", "",
        "| Gasshypotese | Middel romlig standardavvik i lokal demping | Gassmasse i siste bilde, snitt | Innløst masse i siste bilde, snitt |",
        "|---|---:|---:|---:|"]
    for gas in paired_modes:
        std=[p["local_diagnostics"]["factor_spatial_std"] for p in paired if p["gas_mode"]==gas
             and p["local_diagnostics"]["factor_spatial_std"] is not None]
        masses=[s[f"local_{gas}"] for s in split.values()]
        lines.append(f"| {gas} | {np.mean(std):.4f} | {np.mean([m['gas_mass_g'] for m in masses]):.3f} g | {np.mean([m['aqueous_mass_g'] for m in masses]):.3f} g |")
    lines += ["", "Den lokale dempingen, konsentrasjonen, fargesignal og gassmetning er plottet i de samme faste vinduene over tid i Studio. Nabogradienter innenfor og mellom facies er lagret per bilde. En gradient ved laggrensen kan også skyldes reell transport; dette er et varselmål, ikke et automatisk kvalitetsstempel.","",
        "Følsomhet nedenfor: endring i dempingsfaktor ved beta=1, med samme optiske input og uten ny tilpasning. Dette er ikke et usikkerhetsintervall.","",
        "| Endring | Middel absolutt forskjell | Største bildemiddel |", "|---|---:|---:|"]
    for key,item in aggregate.items():
        if "factor_beta1" in key:
            lines.append(f"| {key.split('/')[0]} | {item['mean_of_run_means']:.4f} | {item['maximum_frame_difference']:.4f} |")
    lines += ["", "Signalnormaliseringen 3 er valgt før tilpasningen fra det observerte normaliserte optiske spennet (typisk ca. 0,5–2,5). Den er ingen måling av metning; skala 2 og 4 inngår derfor som følsomhet. Historikk over mer enn 48 timer uten bilde nullstilles. Ingen faciesbarriere legges automatisk til."]
    with (folder/"report.md").open("a",encoding="utf-8") as out:out.write("\n".join(lines)+"\n")


def create_request(previous_request, output, *, workers=4,
                   include_persistent_full_depth=False, variant_ids=None):
    from darsia.fluidflower import concentration_local, concentration_local_history, concentration_history, gas_history
    old=io.read(previous_request);output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new study directory")
    if old.get("format") not in (previous.VERSION, VERSION):
        raise ValueError("A frozen FF_UM gas-history or local-dilution request is required")
    control=Path(old["fixed_parameter_control"] if old.get("fixed_parameter_control") else
                 Path(previous_request).parent/"results"/"fixed_sg_075"/"fits.json")
    if not control.is_file():
        raise FileNotFoundError("Frozen 75% gas optical control is missing: " + str(control))
    from darsia.fluidflower import hidden_gas_depth
    numerical=(__file__,previous.__file__,joint.__file__,concentration_local.__file__,
        concentration_local_history.__file__,concentration_history.__file__,gas_history.__file__,comparison.__file__)
    if include_persistent_full_depth:
        numerical=(*numerical,hidden_gas_depth.__file__)
    sources={str(Path(p).resolve()):io.sha(p) for p in numerical}
    selected_variants=variants(include_persistent_full_depth=include_persistent_full_depth)
    if variant_ids is not None:
        ids=tuple(variant_ids)
        if not ids or len(ids)!=len(set(ids)):
            raise ValueError("Select one or more distinct comparison variants")
        unknown=set(ids)-{v["id"] for v in selected_variants}
        if unknown:
            raise ValueError("Unknown comparison variants: " + ", ".join(sorted(unknown)))
        selected_variants=[v for v in selected_variants if v["id"] in ids]
    request=dict(format=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),output=str(output),
        source=old["source"],cache=old["cache"],results_root=old["results_root"],inputs=old["inputs"],
        variants=selected_variants,
        numerical_sources=sources,workers=workers,threads_per_worker=1,
        optimization_settings=old["optimization_settings"],fixed_parameter_control=str(control),
        fixed_parameter_control_sha256=io.sha(control),
        local_settings=old.get("local_settings",dict(radius_m=.02,signal_scale=3.,
            weakness_power=2.,max_gap_h=48.,max_exposure=6.)),
        core_settings=old.get("core_settings",dict(low=.6,high=.9,tau_h=2.,core_mm=10.,max_gap_h=48.)),
        full_depth_settings=full_depth_settings() if include_persistent_full_depth else None,
        prior_request_path=str(Path(previous_request).resolve()),
        prior_request_sha256=io.sha(previous_request),
        design=dict(phase_maps_changed=False,automatic_promotion=False,
            no_global_factor_in_local_mode=True,no_mass_projection=True,
            visible_full_depth_scenario=(
                "Causal, same-cell visible gas only; 75% until 3 observations and 0.5 h, "
                "then 100% if consecutive gap <=24 h; no corridor or halo"
                if include_persistent_full_depth else None),
            primary_metric="Held-out concentration mass plus local gradient/time/phase-split diagnostics",
            sensitivity="radius10/40mm, signal scale2/4, maxgap24h, gas tau0.5/6h, every-second-frame"))
    output.mkdir(parents=True);io.write(output/"request.json",request)
    return output/"request.json"


def run(request_path):
    request=io.read(request_path);root=Path(request["output"]);start=time.perf_counter()
    def status(stage,**values):io.write(root/"status.json",dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,**values))
    try:
        previous._check_code(request)
        if set(request["inputs"])!=set(RUNS) or sum(b["frames"] for b in request["inputs"].values())!=261:
            raise ValueError("Fixed thirteen-run, 261-frame cohort required")
        for binding in request["inputs"].values():previous.validate_template(
            previous._snapshot(binding["prepared"],binding["prepared_sha256"]),
            previous._snapshot(binding["previous_result"],binding["previous_result_sha256"]))
        workers=max(1,min(int(request["workers"]),os.cpu_count() or 1));previous._limits()
        status("preparing local history and sensitivities",workers=workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_prepare_run,request,name):name for name in RUNS}
            for count,future in enumerate(as_completed(futures),1):status("preparing local history and sensitivities",completed_runs=count,run=future.result())
        preparation=time.perf_counter()-start;fit_start=time.perf_counter();results={}
        status("fitting and validating",completed_variants=0)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_fit_variant,request,v):v["id"] for v in request["variants"]}
            for count,future in enumerate(as_completed(futures),1):
                key,fits,validation=future.result();results[key]=(fits,validation)
                status("fitting and validating",completed_variants=count,variant=key)
        fit_seconds=time.perf_counter()-fit_start;_report(request,results);render_start=time.perf_counter();delivered=[]
        status("preparing Studio fields and curves",completed_runs=0)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_render_run,request,name):name for name in RUNS}
            for count,future in enumerate(as_completed(futures),1):
                delivered.extend(future.result());status("preparing Studio fields and curves",completed_runs=count,run=futures[future])
        if len(delivered)!=len(RUNS)*len(request["variants"]):raise ValueError("Missing Studio results")
        previous._check_code(request)
        _local_report(request)
        io.write(root/"delivery.json",dict(complete=True,results=delivered,runs=list(RUNS),frames=261,variants=request["variants"]))
        timing=dict(preparation_seconds=preparation,fitting_validation_seconds=fit_seconds,
            studio_render_seconds=time.perf_counter()-render_start,elapsed_seconds=time.perf_counter()-start)
        io.write(root/"timing.json",timing);status("complete",results=len(delivered),timing=timing)
    except BaseException as error:
        status("failed",error=str(error),traceback=traceback.format_exc());raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--request",required=True,type=Path)
    run(parser.parse_args().request)
