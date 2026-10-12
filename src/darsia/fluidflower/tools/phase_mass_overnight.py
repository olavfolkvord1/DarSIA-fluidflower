"""Compare optical/chemical/dilution hypotheses against time-resolved injection.

All fits are inverse calibrations, not independently measured concentrations.
No frame-by-frame mass rescaling, geometry fitting or production publication.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
import csv
import gc
import html
import itertools
import json
from pathlib import Path
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

REPO = series_root("ffac")

from darsia.fluidflower.tools.phase_mass_overnight_data import METHODS,RUNS,prepare_run,read,write,sha,progress,small

optuna.logging.set_verbosity(optuna.logging.WARNING)
CHEMISTRY=("linear","titration")
DILUTION=("none","area","pore_volume")
PH_GRID=(7.8,8.2,8.6,9.0,9.4,9.8,10.1)
METHOD_LABELS={"baseline_btb":"Baseline BTB","clearwater_gain":"Local clear-water correction",
               "layer_gain":"Clear-water correction by sand layer","indicator_od":"Normalized optical-density change"}


def temporal_split(frames):
    positive=np.array([i for i,r in enumerate(frames) if r["injected_mass_g"]>0 and r["elapsed_h"]>=0],int)
    if len(positive)<8: raise ValueError("Too few time steps for temporal holdout")
    held=np.unique(np.r_[positive[3::4],positive[-2:]])
    train=np.setdiff1d(positive,held)
    assert not np.intersect1d(train,held).size
    return train,held


def dilution_factor(frames,kind,beta):
    result=np.ones(len(frames),float)
    if kind=="none" or beta==0: return result
    mass=np.array([r["injected_mass_g"] for r in frames])
    if mass.max()<=0: return result
    # First observed frame after completion: never infer an exact unobserved
    # footprint at injection end, and never use future maximum footprint.
    completed=np.flatnonzero(mass>=.9999*mass.max())
    if not completed.size: return result
    anchor=int(completed[0])
    key="affected_area_m2" if kind=="area" else "affected_pore_volume_m3"
    support=np.array([r[key] for r in frames],float)
    if support[anchor]<=0: raise ValueError("No affected support at dilution anchor")
    result[anchor+1:]=np.minimum(1,np.maximum(support[anchor+1:]/support[anchor],1)**(-beta))
    return result


@lru_cache(maxsize=16)
def titration_lut(ph):
    import darsia
    flash=darsia.TitrationFlash(0.,1.,0.,1.,initial_pH=float(ph),aqueous_blend=0.,n_lut=2048)
    return flash._lut_y.copy(),flash._lut_caq.copy()


def concentration(signal,params,chemistry):
    z=np.clip((np.asarray(signal)-params["low"])/params["span"],0,1)
    if chemistry=="titration":
        x,y=titration_lut(params["initial_pH"])
        z=np.interp(z,x,y)
    return z


def load_arrays(path):
    path=Path(path)
    return {key:np.load(path/f"{key}.npy",mmap_mode="r") for key in ("signal","rgb","aq_volume","gas_volume")}


def histogram_inputs(path,bins=2048):
    path=Path(path)
    target=path/f"histograms_{bins}.npz"
    if target.exists():
        with np.load(target) as z: return {k:z[k] for k in z.files}
    metadata=read(path/"prepared.json")
    arrays=load_arrays(path)
    n=len(metadata["frames"])
    out={"gas_capacity_g":np.zeros(n),"gas_liquid_capacity_g":np.zeros(n)}
    for method in METHODS:
        maximum=12. if method=="indicator_od" else 2.
        edges=np.linspace(0,maximum,bins+1)
        out[f"centers_{method}"]=(edges[:-1]+edges[1:])/2
        out[f"hist_{method}"]=np.zeros((n,bins))
    for i in range(n):
        with np.load(path/f"properties_{i:03d}.npz") as p:
            solubility,density=p["solubility"],p["density"]
            weight=arrays["aq_volume"][i].astype(float)*solubility*1000
            out["gas_capacity_g"][i]=np.sum(arrays["gas_volume"][i].astype(float)*density)*1000
            out["gas_liquid_capacity_g"][i]=np.sum(arrays["gas_volume"][i].astype(float)*solubility)*1000
        for j,method in enumerate(METHODS):
            maximum=12. if method=="indicator_od" else 2.
            edges=np.linspace(0,maximum,bins+1)
            signal=np.clip(arrays["signal"][i,j],0,np.nextafter(maximum,0))
            out[f"hist_{method}"][i]=np.histogram(signal,bins=edges,weights=weight)[0]
    np.savez_compressed(target,**out)
    return out


def predict_mass(hist,frames,method,chemistry,dilution,params,gas_liquid_fraction=1.):
    c=concentration(hist[f"centers_{method}"],params,chemistry)
    factor=dilution_factor(frames,dilution,params.get("beta",0.))
    aqueous=hist[f"hist_{method}"]@c*factor
    sg=params["gas_saturation"]
    # Visible gas is not a gas saturation measurement. A single fitted occupancy
    # applies to all its pixels/times. Residual liquid is locally equilibrated in
    # the primary hypothesis; the 0/0.5/1 sensitivity is reported separately.
    aqueous+=gas_liquid_fraction*(1-sg)*hist["gas_liquid_capacity_g"]
    gas=sg*hist["gas_capacity_g"]
    return aqueous,gas,factor


def mass_metrics(reference,detected,indices):
    reference,detected=np.asarray(reference),np.asarray(detected)
    den=np.maximum(reference,.05*reference.max())
    residual=(detected[indices]-reference[indices])/den[indices]
    return {"nmae":float(np.mean(np.abs(residual))),"nrmse":float(np.sqrt(np.mean(residual**2))),
            "final_injected_g":float(reference[-1]),"final_detected_g":float(detected[-1]),
            "final_ratio":float(detected[-1]/reference[-1]),"n_frames":int(len(indices))}


def fit_one(path,method,chemistry,dilution,trials=600,seeds=(20260912,20260913)):
    path=Path(path)
    dest=path/"fits"/f"{method}__{chemistry}__{dilution}"
    result_path=dest/"result.json"
    if result_path.exists(): return read(result_path)
    dest.mkdir(parents=True,exist_ok=True)
    meta=read(path/"prepared.json")
    frames=meta["frames"]
    train,held=temporal_split(frames)
    hist=histogram_inputs(path)
    reference=np.array([f["injected_mass_g"] for f in frames])
    started=time.perf_counter()
    seed_results=[]
    for seed in seeds:
        seed_file=dest/f"seed_{seed}.json"
        if seed_file.exists():
            seed_results.append(read(seed_file));continue
        maximum=12. if method=="indicator_od" else 2.
        def objective(trial):
            params={"low":trial.suggest_float("low",0,maximum*.8),
                    "span":trial.suggest_float("span",maximum*.005,maximum,log=True),
                    "gas_saturation":trial.suggest_float("gas_saturation",.01,.99)}
            if dilution!="none": params["beta"]=trial.suggest_float("beta",0,2)
            if chemistry=="titration": params["initial_pH"]=trial.suggest_categorical("initial_pH",PH_GRID)
            aq,gas,_=predict_mass(hist,frames,method,chemistry,dilution,params)
            metrics=mass_metrics(reference,aq+gas,train)
            return metrics["nmae"]+.25*metrics["nrmse"]
        study=optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=60))
        study.optimize(objective,n_trials=trials,show_progress_bar=False)
        record={"seed":seed,"params":study.best_params,"objective":study.best_value,
                "trial_count":len(study.trials),"near_best_parameters":[{"params":t.params,"objective":t.value}
                    for t in sorted(study.trials,key=lambda t:t.value)[:10]]}
        write(seed_file,record)
        with (dest/f"trials_{seed}.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["number","objective",*study.best_params]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for trial in study.trials:writer.writerow({"number":trial.number,"objective":trial.value,**trial.params})
        seed_results.append(record)
    best=min(seed_results,key=lambda r:r["objective"])
    aq,gas,factor=predict_mass(hist,frames,method,chemistry,dilution,best["params"])
    result={"method":method,"chemistry":chemistry,"dilution":dilution,"parameters":best["params"],
            "train_indices":train.tolist(),"holdout_indices":held.tolist(),"seeds":seed_results,
            "train":mass_metrics(reference,aq+gas,train),"holdout":mass_metrics(reference,aq+gas,held),
            "seconds":time.perf_counter()-started,"reference_policy":"cumulative injected mass per time; no per-frame rescaling",
            "selection_policy":"best training objective across fixed seeds; holdout never in objective",
            "rows":[{**frame,"aqueous_g":float(aq[i]),"gas_g":float(gas[i]),"total_g":float(aq[i]+gas[i]),
                     "dilution_factor":float(factor[i]),"split":"train" if i in train else "holdout" if i in held else "baseline"}
                    for i,frame in enumerate(frames)]}
    write(result_path,result)
    write_csv(dest/"mass.csv",result["rows"])
    return result


def write_csv(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def full_fields(path,index,result,gas_liquid_fraction=1.):
    path=Path(path)
    arrays=load_arrays(path)
    meta=read(path/"prepared.json")
    p=result["parameters"]
    factor=dilution_factor(meta["frames"],result["dilution"],p.get("beta",0))[index]
    concentration_map=concentration(arrays["signal"][index,METHODS.index(result["method"])],p,result["chemistry"])*factor
    av=arrays["aq_volume"][index].astype(float)
    gv=arrays["gas_volume"][index].astype(float)
    with np.load(path/f"properties_{index:03d}.npz") as props:
        solub,density=props["solubility"],props["density"]
    mass_aq=(concentration_map*av+gas_liquid_fraction*(1-p["gas_saturation"])*gv)*solub*1000
    mass_g=p["gas_saturation"]*gv*density*1000
    return concentration_map, mass_aq, mass_g, solub


def darsia_parity(path,index,result):
    import darsia
    from darsia.fluidflower.mass_materialization import _mass_analysis,_image
    path=Path(path);meta=read(path/"prepared.json");arrays=load_arrays(path)
    c,ma,mg,_=full_fields(path,index,result)
    with np.load(path/"references.npz") as z: volume=z["pore_volume"].astype(float)
    av,gv=arrays["aq_volume"][index],arrays["gas_volume"][index]
    sg=result["parameters"]["gas_saturation"]
    s_eff=np.divide(sg*gv,volume,out=np.zeros_like(volume),where=volume>0)
    liquid=volume*(1-s_eff)
    c_eff=np.divide(c*av+(1-sg)*gv,liquid,out=np.zeros_like(liquid),where=liquid>0)
    grid=meta["grid"];row=meta["frames"][index]
    analysis=_mass_analysis(c.shape,grid["width_m"],grid["height_m"],row["pressure_bar"],row["temperature_c"],row["pressure_gradient_bar_per_m"],row["temperature_gradient_c_per_m"])
    res=analysis.mass_analysis(_image(c_eff,grid["width_m"],grid["height_m"]),_image(s_eff,grid["width_m"],grid["height_m"]))
    darsia_aq=np.asarray(res.mass_aq.img)*volume*1000
    darsia_g=np.asarray(res.mass_g.img)*volume*1000
    np.testing.assert_allclose(darsia_aq,ma,rtol=3e-6,atol=1e-9)
    np.testing.assert_allclose(darsia_g,mg,rtol=3e-6,atol=1e-9)
    return {"frame":index,"passed":True,"aqueous_g":float(ma.sum()),"gas_g":float(mg.sum()),
            "max_pixel_abs_error_g":float(max(np.max(abs(ma-darsia_aq)),np.max(abs(mg-darsia_g))))}


def render_run(path,results,root):
    path=Path(path);meta=read(path/"prepared.json");arrays=load_arrays(path)
    frames=meta["frames"];run=meta["run"]
    gallery=path/"gallery";gallery.mkdir(exist_ok=True)
    # One representative per optical method chosen on temporal holdout for
    # development display, plus the best no-dilution control and titration.
    selected=[]
    for method in METHODS:
        selected.append(min((r for r in results if r["method"]==method),key=lambda r:r["holdout"]["nmae"]))
    selected.append(min((r for r in results if r["dilution"]=="none"),key=lambda r:r["holdout"]["nmae"]))
    selected.append(min((r for r in results if r["chemistry"]=="titration"),key=lambda r:r["holdout"]["nmae"]))
    selected=list({(r["method"],r["chemistry"],r["dilution"]):r for r in selected}.values())
    with np.load(path/"references.npz") as z:
        baseline,clear,facies,volume=z["baseline"],z["clearwater"],z["facies"],z["pore_volume"]
    fig,axes=plt.subplots(1,3,figsize=(17,4),layout="constrained")
    for ax,img,title in zip(axes,[baseline,clear,facies],["Pre-injection baseline","Clear water: sand optical reference","Sand layers"]):
        ax.imshow(img);ax.set_title(title);ax.axis("off")
    fig.suptitle(run.upper()+" | registered references");fig.savefig(gallery/"references.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
    for method,ax in zip(METHODS,axes.flat):
        for r in results:
            if r["method"]==method:
                ax.plot([f["elapsed_h"] for f in frames],[v["total_g"] for v in r["rows"]],label=r["chemistry"]+" / "+r["dilution"],alpha=.75)
        ax.plot([f["elapsed_h"] for f in frames],[f["injected_mass_g"] for f in frames],"k--",lw=2,label="Injected reference")
        ax.set(xlabel="Time [h]",ylabel="Detected CO2 [g]",title=METHOD_LABELS[method]);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(run.upper()+" | all 24 mass hypotheses, no per-frame rescaling");fig.savefig(gallery/"mass_comparison.png",dpi=150);plt.close(fig)
    summaries=[]
    for r in selected:
        key=f"{r['method']}__{r['chemistry']}__{r['dilution']}"
        out=gallery/key;out.mkdir(exist_ok=True)
        parity=[];exact_rows=[]
        for i,frame in enumerate(frames):
            c,ma,mg,solub=full_fields(path,i,r)
            # kg/m3 == g/L. Clear-water-only and free-gas pixels are masked from
            # the aqueous display; gas-associated liquid remains in mass algebra.
            visible=arrays["aq_volume"][i]>np.maximum(volume*1e-3,1e-18)
            c_gl=np.where(visible,c*solub,np.nan)
            cell_area=meta["grid"]["width_m"]*meta["grid"]["height_m"]/(c.shape[0]*c.shape[1])
            mass_area=np.where(volume>0,(ma+mg)/cell_area,np.nan)
            image_file=out/f"frame_{i:03d}.png"
            if not image_file.exists():
                fig,axes=plt.subplots(1,3,figsize=(18,4.7),layout="constrained")
                axes[0].imshow(arrays["rgb"][i]);axes[0].set_title("Corrected source")
                shown=axes[1].imshow(c_gl,cmap="viridis",norm=LogNorm(vmin=.0001,vmax=2.5))
                axes[1].set_title("Estimated dissolved CO2 [g/L], log scale")
                fig.colorbar(shown,ax=axes[1],fraction=.03,pad=.02)
                shown=axes[2].imshow(mass_area,cmap="magma",norm=LogNorm(vmin=.001,vmax=1000))
                axes[2].set_title("Estimated total CO2 [g/m²], log scale")
                fig.colorbar(shown,ax=axes[2],fraction=.03,pad=.02)
                for ax in axes:ax.axis("off")
                fig.suptitle(f"{run.upper()}  frame {i}  t={frame['elapsed_h']:.3f} h | {METHOD_LABELS[r['method']]} / {r['chemistry']} / dilution: {r['dilution']}\n"
                             f"Injected {frame['injected_mass_g']:.3f} g | detected aq {ma.sum():.3f} g + gas {mg.sum():.3f} g | INVERSE-CALIBRATED ESTIMATE")
                fig.savefig(image_file,dpi=135);plt.close(fig)
            # Standalone linear and logarithmic scientific maps for every frame.
            linear=out/f"concentration_linear_{i:03d}.png"
            if not linear.exists():
                fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
                im=ax.imshow(c_gl,cmap="viridis",vmin=0,vmax=2.5)
                fig.colorbar(im,ax=ax,label="Estimated dissolved CO2 [g/L]");ax.axis("off")
                ax.set_title(f"{run.upper()} | t={frame['elapsed_h']:.3f} h | common linear scale 0–2.5 g/L\nInverse calibration; local concentration not independently validated")
                fig.savefig(linear,dpi=150);plt.close(fig)
            if i in (1,len(frames)//2,len(frames)-1):
                parity.append(darsia_parity(path,i,r))
                np.savez_compressed(out/f"numerical_fields_{i:03d}.npz",aqueous_concentration_g_l=c_gl,
                                    aqueous_mass_g=ma,gas_mass_g=mg,total_mass_g=ma+mg)
            exact_rows.append({"frame":i,"time_h":frame["elapsed_h"],"aqueous_g":float(ma.sum()),"gas_g":float(mg.sum()),
                               "total_g":float((ma+mg).sum()),"injected_g":frame["injected_mass_g"],
                               "histogram_mass_g":r["rows"][i]["total_g"]})
            progress(root,"render",run=run,method=key,frame=i+1,frames=len(frames))
        errors=[abs(x["total_g"]-x["histogram_mass_g"])/max(frames[-1]["injected_mass_g"],1e-10) for x in exact_rows]
        write(out/"verification.json",{"darsia_parity":parity,"max_histogram_error_fraction_final_injected":max(errors),"exact_rows":exact_rows})
        summaries.append({"key":key,"holdout_nmae":r["holdout"]["nmae"],"max_histogram_error":max(errors)})
    write(path/"gallery.json",summaries)
    return summaries


def sensitivities(path,results):
    path=Path(path);meta=read(path/"prepared.json");hist=histogram_inputs(path);frames=meta["frames"]
    reference=np.array([f["injected_mass_g"] for f in frames]);train,held=temporal_split(frames)
    best=min(results,key=lambda r:r["holdout"]["nmae"])
    rows=[]
    for depth_scale,liquid,sg_scale in itertools.product((.9,1.,1.1),(0.,.5,1.),(.8,1.,1.2)):
        p={**best["parameters"],"gas_saturation":min(.99,best["parameters"]["gas_saturation"]*sg_scale)}
        aq,gas,_=predict_mass(hist,frames,best["method"],best["chemistry"],best["dilution"],p,liquid)
        rows.append({"depth_scale":depth_scale,"gas_residual_liquid_concentration_fraction":liquid,"gas_saturation_scale":sg_scale,
                     **mass_metrics(reference,(aq+gas)*depth_scale,held)})
    write_csv(path/"sensitivity.csv",rows)
    write(path/"sensitivity.json",{"candidate":[best[k] for k in ("method","chemistry","dilution")],"refitted":False,"rows":rows})


def transfer_tests(root):
    rows=[]
    for dataset,runs in RUNS.items():
        for source_run,target_run in (runs,runs[::-1]):
            source=Path(root)/dataset/source_run;target=Path(root)/dataset/target_run
            if not (source/"results.json").exists() or not (target/"prepared.json").exists():continue
            hist=histogram_inputs(target);meta=read(target/"prepared.json")
            ref=np.array([r["injected_mass_g"] for r in meta["frames"]]);idx=np.flatnonzero(ref>0)
            for r in read(source/"results.json"):
                aq,gas,_=predict_mass(hist,meta["frames"],r["method"],r["chemistry"],r["dilution"],r["parameters"])
                rows.append({"dataset":dataset,"source_run":source_run,"target_run":target_run,
                             **{k:r[k] for k in ("method","chemistry","dilution")},**mass_metrics(ref,aq+gas,idx)})
    if rows:write_csv(Path(root)/"cross_run_transfer.csv",rows)
    return rows


def report(root):
    root=Path(root);runs=[]
    for dataset,names in RUNS.items():
        for run in names:
            p=root/dataset/run
            if not (p/"results.json").exists():continue
            results=read(p/"results.json");meta=read(p/"prepared.json")
            runs.append({"dataset":dataset,"run":run,"results":results,"frames":meta["frames"],
                         "gallery":read(p/"gallery.json") if (p/"gallery.json").exists() else []})
    payload=json.dumps([{k:v for k,v in r.items() if k!="results"} for r in runs]).replace("</","<\\/")
    tables=[]
    for run in runs:
        rows=[]
        for r in sorted(run["results"],key=lambda x:x["holdout"]["nmae"]):
            rows.append(f"<tr><td>{r['method']}</td><td>{r['chemistry']}</td><td>{r['dilution']}</td><td>{r['train']['nmae']:.1%}</td><td>{r['holdout']['nmae']:.1%}</td><td>{r['holdout']['final_detected_g']:.3f}</td><td>{r['holdout']['final_injected_g']:.3f}</td></tr>")
        prefix=f"{run['dataset']}/{run['run']}"
        tables.append(f"<h2>{run['run'].upper()}</h2><img src='{prefix}/gallery/references.png'><img src='{prefix}/gallery/mass_comparison.png'><table><tr><th>Optical method</th><th>Chemistry</th><th>Dilution</th><th>Train NMAE</th><th>Temporal holdout NMAE</th><th>Final detected [g]</th><th>Injected [g]</th></tr>{''.join(rows)}</table>")
    document='''<!doctype html><meta charset="utf-8"><title>Phase to DarSIA mass — overnight experiment</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1450px;padding:0 20px;background:#f7f8fa;color:#14202c}img{width:100%;background:white}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ccd}th{background:#dfe8f0;text-align:left}select,input{margin:10px;padding:7px}section{background:#fff2cc;padding:15px}a{color:#1859a7}.controls{position:sticky;top:0;background:#eaf0f6;padding:8px;z-index:2}</style>
<h1>Fasekart → konsentrasjon og masse</h1><section><b>Utviklingstest — invers kalibrering, ikke uavhengig målt konsentrasjon.</b>
<p>Referanse: kumulativ injisert CO₂-masse ved hvert tidssteg. Ingen automatisk skalering av hvert kart til referansen. Dybde og porøsitet er låst. Fri-vannslaget er utenfor analyseområdet, mens injisert masse gjelder hele forsøket.</p>
<p>Klarvann beskriver sandens optiske bakgrunn, ikke en kjent høy CO₂-konsentrasjon. Fasekartene bestemmer hvor CO₂ finnes; faseklassifiserings­sannsynlighet brukes ikke som konsentrasjon. Gassmetning og konsentrasjon i restvannet i gassområder er antakelser med følsomhetstester.</p>
<p>Hvert fjerde tidspunkt og de to siste holdes utenfor parameter­tilpasningen. Modeller velges og sammenlignes her på disse utviklingsdataene; dette er ikke en endelig vitenskapelig validering. Optisk tetthet fra kameraets sRGB er en diagnostisk tilnærming. Titrering bruker nominell BTB-kjemi og effektiv pH.</p></section>
<h2>Sammenlign bilder</h2><div class="controls"><select id="run"></select><select id="model"></select><input id="frame" type="range" min="0" value="0"><span id="time"></span><label><input id="linear" type="checkbox">Lineær konsentrasjonsskala</label></div><img id="map">
<p>Felles skala for alle forsøk og metoder: konsentrasjon 0,0001–2,5 g/L (logaritmisk), eller 0–2,5 g/L (lineær). Massekart 0,001–1000 g/m². Hvitt i konsentrasjonskartet betyr utenfor innløst-fasen. Åpne PNG-filen for full oppløsning.</p>
<p><a href="cross_run_transfer.csv">Overføring av parametere mellom forsøk</a> · <a href="protocol.json">Forsøksprotokoll</a> · <a href="assessment.md">Vurdering</a></p>
'''+''.join(tables)+'''
<script>const runs='''+payload+'''; const r=document.querySelector('#run'),m=document.querySelector('#model'),f=document.querySelector('#frame');
function show(){const x=runs[r.value];if(!x||!x.gallery.length)return;const j=+f.value;document.querySelector('#time').textContent='Bilde '+j+' / '+x.frames[j].elapsed_h.toFixed(3)+' h';document.querySelector('#map').src=x.dataset+'/'+x.run+'/gallery/'+m.value+'/'+(document.querySelector('#linear').checked?'concentration_linear_':'frame_')+String(j).padStart(3,'0')+'.png';}
function selectRun(){const x=runs[r.value];m.innerHTML=x.gallery.map(y=>'<option value="'+y.key+'">'+y.key+'</option>').join('');f.max=x.frames.length-1;f.value=Math.min(+f.value,+f.max);show();}
r.innerHTML=runs.map((x,i)=>'<option value="'+i+'">'+x.run.toUpperCase()+'</option>').join('');r.onchange=selectRun;m.onchange=show;f.oninput=show;document.querySelector('#linear').onchange=show;selectRun();</script>'''
    (root/"report.html").write_text(document,encoding="utf-8")
    return runs


def assessment(root,runs,transfer):
    lines=["# Nattest: fra Studio-fasekart til DarSIA-masse", "",
           "Alle konsentrasjoner er estimater kalibrert mot tidsavhengig injisert masse. Totalmasselikhet identifiserer ikke en unik romlig konsentrasjon eller gassmetning.", "",
           "Fasekartene fra 11. september beholdes. Numeriske klasselabels dekodes fra de eksakte native RGBA-palettverdiene; blandede skjermbilder eller alfaverdi brukes aldri som konsentrasjon. DarSIA CO2MassAnalysis er kontrollert mot den eksplisitte massealgebraen.","",
           "| Forsøk | Beste utviklingskandidat | Holdout NMAE | Slutt detektert / injisert |", "|---|---|---:|---:|"]
    for run in runs:
        best=min(run["results"],key=lambda x:x["holdout"]["nmae"])
        lines.append(f"| {run['run'].upper()} | {best['method']} / {best['chemistry']} / {best['dilution']} | {best['holdout']['nmae']:.1%} | {best['holdout']['final_detected_g']:.3f} / {best['holdout']['final_injected_g']:.3f} g |")
    lines += ["", "Kandidatene er sortert på et tidsmessig utviklingsholdout, ikke en urørt ekstern validering. Se cross_run_transfer.csv for uendret parameteroverføring til det andre forsøket i samme serie.",
              "", "Utvanning er D(t)=min(1,(støtte ved første observerte tidspunkt etter injeksjonsslutt / støtte ved t)^beta), med areal eller porevolum som støtte. Ingen fremtidig maksimal utbredelse brukes. En slik faktor er en modellhypotese, ikke en måling av lokal diffusjon. Den følger synlig utbredelse og kan ikke alene bevise lokal massekonservering.",
              "", "Hver kandidat er kjørt med to faste Optuna-seed. Parametere velges på treningsfeil. Effektiv pH er ikke en uavhengig måling. Fri gass i et 2D-fasekart er ikke målt metning gjennom riggdybden; se sensitivity.csv for konsekvensen av dybde, gassmetning og restvæske.",
              "", "Klarvannsreferansene korrigerer optisk bakgrunn, men gir ikke en kjent konsentrasjonskalibrering. Ingen konklusjon om absolutte lokale konsentrasjoner kan baseres på injisert totalmasse alene.",
              "", "Faglig bakgrunn: [DarSIA / UiB om multikromatisk kalibrering](https://www4.uib.no/en/research/research-groups/porous-media-group/quantitative-analysis-for-fluidflower-experiments-using-multichromatic-tracer-data-2023). Der krever kvantitativ fargetolkning sammenhørende farger og kjente konsentrasjoner; nattesten har i stedet en massebasert invers kalibrering."]
    (Path(root)/"assessment.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--trials",type=int,default=600);p.add_argument("--width",type=int,default=1536)
    p.add_argument("--run");p.add_argument("--skip-render",action="store_true")
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4)
    protocol={"methods":METHODS,"chemistry":CHEMISTRY,"dilution":DILUTION,"trials_per_seed":args.trials,
              "seeds":[20260912,20260913],"runs":RUNS,"width":args.width,"nominal_chemistry":"DarSIA TitrationFlash default BTB, varying effective pH",
              "gas_residual_liquid_concentration_fraction":1.,"geometry_fitted":False,"per_frame_mass_rescaling":False,
              "holdout":"every fourth positive-mass time step plus final two, excluded from all trial objectives",
              "code":{name:sha(project_path('scripts')/name) for name in ("phase_mass_overnight.py","phase_mass_overnight_data.py")}}
    if (root/"protocol.json").exists() and read(root/"protocol.json")!=protocol:
        raise ValueError("Frozen campaign protocol changed. Use a separate output directory.")
    write(root/"protocol.json",protocol)
    import shutil
    (root/"source_snapshot").mkdir(exist_ok=True)
    for name in protocol["code"]:shutil.copy2(project_path('scripts')/name,root/"source_snapshot"/name)
    for dataset,runs in RUNS.items():
        for run in runs:
            if args.run and args.run!=run:continue
            prepare_run(root,dataset,run,args.width)
            path=root/dataset/run
            results=[]
            for method,chem,dilution in itertools.product(METHODS,CHEMISTRY,DILUTION):
                progress(root,"fit",run=run,method=method,chemistry=chem,dilution=dilution,completed=len(results),total=24)
                results.append(fit_one(path,method,chem,dilution,args.trials))
            write(path/"results.json",results)
            sensitivities(path,results)
            if not args.skip_render:render_run(path,results,root)
            report(root)
            gc.collect()
    transfer=transfer_tests(root);runs=report(root);assessment(root,runs,transfer)
    progress(root,"completed",runs=len(runs),fits=sum(len(r["results"]) for r in runs),report=str(root/"report.html"))


if __name__=="__main__": main()
