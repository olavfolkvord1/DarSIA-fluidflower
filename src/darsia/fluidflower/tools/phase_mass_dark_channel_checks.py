"""Dark-channel optical sensitivity under frozen phase maps and mass curves.

Perturbations are explicit sensitivity scenarios, not a calibrated camera-noise
distribution. The candidate downweights dark readings with a fixed soft-log
response; its brightness-dependent bias must be assessed separately.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import process_directory, series_root, series_path, package_root, local_cache_root

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = series_root("ffac")

from darsia.fluidflower.tools.phase_mass_joint_optics import build_joint_reference,joint_optical_signals,srgb_to_linear,_regularized_span
from darsia.fluidflower.tools.phase_mass_overnight import dilution_factor

PARENT=local_cache_root() / 'phase_diagnostics/phase_mass_overnight_20260912'
PHYSICS=local_cache_root() / 'phase_diagnostics/phase_mass_physics_20260912'
OUTPUT=local_cache_root() / 'phase_diagnostics/phase_mass_optics_checks_20260912/dark_channels'
Q=np.array([.5,.5,-1],np.float32)
DARK_THRESHOLD=8/255
SCENARIOS=("nominal","current_minus1","current_plus1","current_minus2","current_plus2",
           "baseline_minus1","baseline_plus1","baseline_minus2","baseline_plus2",
           "current_uniform2_seeded","baseline_uniform2_seeded")


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class DarkReference:
    baseline_response:np.ndarray
    denominator:np.ndarray
    reference_reliability:np.ndarray
    epsilon_linear:float
    sigma:float
    endpoint_alignment_gain:float
    diagnostics:dict


def soft_response(linear,epsilon):
    """asinh approaches log at high intensity, with bounded dark derivative."""
    return np.arcsinh(np.maximum(linear,0)/epsilon)


def measurement_reliability(rgb,epsilon):
    intensity=srgb_to_linear(np.clip(rgb,0,1))
    return intensity/np.sqrt(intensity**2+epsilon**2)


def _candidate_unscaled(current,ref):
    response=soft_response(srgb_to_linear(np.clip(current,0,1)),ref.epsilon_linear)
    delta=(response-ref.baseline_response)@Q
    if ref.sigma>0:delta=cv2.GaussianBlur(delta,(0,0),ref.sigma)
    return (delta/ref.denominator).astype(np.float32)


def build_dark_reference(baseline,clear,facies,domain,*,epsilon_dn=2.,sigma=2.):
    """Fit references only; epsilon is an explicit preselected hypothesis.

    The soft response also reduces sensitivity to dark current measurements.
    That is measurement reliability, not a learned reference-only weight.
    The single endpoint gain aligns clear-water optical response, not mass.
    """
    if epsilon_dn<=0 or sigma<0:raise ValueError("Invalid soft-log parameters")
    baseline=np.clip(np.asarray(baseline,np.float32),0,1);clear=np.clip(np.asarray(clear,np.float32),0,1)
    facies=np.asarray(facies);domain=np.asarray(domain,bool)
    if baseline.shape!=clear.shape or baseline.ndim!=3 or baseline.shape[-1]!=3 or facies.shape!=baseline.shape[:2] or domain.shape!=facies.shape:
        raise ValueError("Reference grids must coincide")
    if not np.isfinite(baseline).all() or not np.isfinite(clear).all() or not domain.any():raise ValueError("Invalid references")
    eps=float(srgb_to_linear(np.array(epsilon_dn/255,np.float32)))
    b,c=srgb_to_linear(baseline),srgb_to_linear(clear)
    br=soft_response(b,eps);cr=soft_response(c,eps)
    raw=(cr-br)@Q
    valid=domain&(baseline.max(axis=-1)<.995)&(clear.max(axis=-1)<.995)
    # Dark reference channels are permitted here: their soft-log derivative is
    # bounded. Reference reliability is reported separately from this response.
    denom,rel,diag=_regularized_span(raw,valid,facies,sigma,.15)
    temporary=DarkReference(br,denom,rel,eps,sigma,1.,{})
    candidate_clear=_candidate_unscaled(clear,temporary)
    joint=build_joint_reference(baseline,clear,facies,domain_mask=domain,sigma=sigma)
    target=joint_optical_signals(clear,joint)["joint_chromatic_od"]
    # Area weighting is fixed by references, independent of plume and dose.
    support=domain&(candidate_clear>0)&np.isfinite(target)
    if not support.any():raise ValueError("No positive soft-log reference span")
    gain=float(np.mean(target[support])/np.mean(candidate_clear[support]))
    if not np.isfinite(gain) or gain<=0:raise ValueError("Invalid reference gain")
    diagnostics={"method":"asinh_linear_light_reference_normalized","epsilon_dn_srgb":epsilon_dn,
        "epsilon_linear":eps,"dark_perturbation_threshold_dn_srgb":8.,
        "reference_fit_inputs":["baseline","clearwater","facies","domain"],
        "clearwater_is_known_CO2_endpoint":False,"endpoint_alignment_gain":gain,
        "endpoint_alignment":"One positive-domain mean clear-water response gain; no injection image or mass used",
        "current_weighting":"Measurement response derivative I/sqrt(I^2+epsilon^2), adaptive to current brightness; not reference fitting",
        "bias_risk":"Soft response breaks exact multiplicative-background invariance near black and can change layer response and concentration scale",
        "reference_span":diag,"baseline_dark_channel_fraction_domain":float(np.mean(baseline[domain]<DARK_THRESHOLD))}
    return DarkReference(br,denom,rel,eps,sigma,gain,diagnostics)


def compute_candidate(current,reference):
    """Signed dimensionless signal, clear-reference aligned; not concentration."""
    return (_candidate_unscaled(current,reference)*reference.endpoint_alignment_gain).astype(np.float32)


def perturb(rgb,scenario,seed):
    rgb=np.asarray(rgb,np.float32);dark=rgb<DARK_THRESHOLD
    if "uniform2" in scenario:
        change=np.random.default_rng(seed).uniform(-2,2,size=rgb.shape).astype(np.float32)/255
    else:
        sign=-1 if "minus" in scenario else 1
        amount=2 if scenario.endswith("2") else 1
        change=sign*amount/255
    return np.clip(rgb+np.where(dark,change,0),0,1).astype(np.float32)


def fixed_concentration(signal,p,d,solubility):
    return (p["amplitude"]*np.clip(signal/p["endpoint"],0,1)**p["gamma"]*d*solubility).astype(np.float32)


def weighted_quantiles(values,weights,quantiles=(.1,.5,.9)):
    order=np.argsort(values);v=values[order];w=weights[order]
    return np.interp(np.asarray(quantiles)*w.sum(),np.cumsum(w),v).tolist()


def geometry_pairs(facies,dataset):
    h,w=facies.shape;above,below=[],[]
    labels=(2,1) if dataset=="ffac" else (5,6)
    for x in range(int(.2*w),int(.82*w)):
        for y in np.where((facies[:-1,x]==labels[0])&(facies[1:,x]==labels[1]))[0]+1:
            if .55*h<y<h-6 and facies[y-6,x]==labels[0] and facies[y+6,x]==labels[1]:
                above.append((y-6,x));below.append((y+6,x))
    return np.asarray(above,int),np.asarray(below,int),labels


def pair_contrast(c,above,below,mask):
    if len(above)==0:return {"count":0,"median_lower_minus_upper_g_l":None,"median_lower_over_upper":None}
    a=tuple(above.T);b=tuple(below.T);ok=mask[a]&mask[b]
    if not ok.any():return {"count":0,"median_lower_minus_upper_g_l":None,"median_lower_over_upper":None}
    av,bv=c[a][ok],c[b][ok]
    positive=av>1e-5
    return {"count":int(ok.sum()),"median_lower_minus_upper_g_l":float(np.median(bv-av)),
            "median_lower_over_upper":float(np.median(bv[positive]/av[positive])) if positive.any() else None}


def render_selected(folder,run,index,rgb,facies,av,volume,nominal,plus2,above,below):
    mask=av>0
    differences={"candidate_minus_joint":nominal["candidate"]-nominal["joint"],
                 "joint_current_plus2_delta":plus2["joint"]-nominal["joint"],
                 "candidate_current_plus2_delta":plus2["candidate"]-nominal["candidate"]}
    field_stats={}
    for name,value in {**nominal,**differences}.items():
        field_stats[name]={"minimum_g_l":float(value[mask].min()),"maximum_g_l":float(value[mask].max()),
                           "aqueous_volume_fraction_abs_above_0.2":float(av[mask&(abs(value)>.2)].sum()/av.sum()),
                           "aqueous_volume_fraction_above_0.6":float(av[mask&(value>.6)].sum()/av.sum()),
                           "aqueous_volume_fraction_above_1.6":float(av[mask&(value>1.6)].sum()/av.sum())}
    for mode,limit,suffix in [("overview",1.6,""),("detail",.6,"_detail")]:
        fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        cm=plt.get_cmap("viridis").copy();cm.set_bad("#efefef");cm.set_over("#ff3030")
        images=[(rgb,"Corrected source RGB",None,None),
                (nominal["joint"],"Frozen joint curve/control",0,limit),
                (nominal["candidate"],"Soft-log candidate, same frozen curve",0,limit),
                (differences["candidate_minus_joint"],"Candidate minus control [g/L]",-.2,.2),
                (differences["joint_current_plus2_delta"],"Control: +2 DN dark current sensitivity",-.2,.2),
                (differences["candidate_current_plus2_delta"],"Candidate: +2 DN sensitivity",-.2,.2)]
        for ax,(value,title,lo,hi) in zip(axes.flat,images):
            if lo is None:ax.imshow(np.clip(value,0,1))
            else:
                cmap=cm if lo==0 else plt.get_cmap("coolwarm").copy();cmap.set_bad("#efefef")
                if lo<0:cmap.set_under("#001f5b");cmap.set_over("#8b0000")
                im=ax.imshow(np.ma.masked_where(~mask,value),cmap=cmap,vmin=lo,vmax=hi)
                fig.colorbar(im,ax=ax,fraction=.025,label="g/L (frozen-curve estimate)",extend="max" if lo==0 else "both")
                title+=f"\nrange {value[mask].min():.3g} to {value[mask].max():.3g} g/L"
            ax.set(title=title);ax.set_axis_off()
        fig.suptitle(f"{run.upper()} frame {index}: {mode}; red concentration pixels exceed{limit:g}g/L\nCandidate is not recalibrated; difference bars extend beyond +/-0.2g/L",fontsize=12)
        fig.savefig(folder/f"comparison_{index:03d}{suffix}.png",dpi=145);plt.close(fig)
    if index>=1:
        pairs_ok=mask[tuple(above.T)]&mask[tuple(below.T)] if len(above) else np.array([],bool)
        if pairs_ok.any():
            x=int(np.median(above[pairs_ok,1]));y0=int(np.median((above[pairs_ok,0]+below[pairs_ok,0])/2))
            ys=slice(max(0,y0-55),min(mask.shape[0],y0+55));xx=slice(max(0,x-2),x+3)
            fig,axes=plt.subplots(2,1,figsize=(9,5.5),sharex=True,constrained_layout=True)
            y=np.arange(mask.shape[0])[ys]
            for method in ("joint","candidate"):
                nom=np.mean(nominal[method][ys,xx],axis=1);pert=np.mean(plus2[method][ys,xx],axis=1)
                visible=np.all(mask[ys,xx],axis=1)
                axes[0].plot(y,np.where(visible,nom,np.nan),label=method)
                axes[0].plot(y,np.where(visible,pert,np.nan),linestyle="--",label=method+" current +2 DN")
            axes[0].set(ylabel="Frozen-curve C [g/L]",title=f"{run.upper()} frame{index}, x={x}; fixed geometry profile")
            axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
            axes[1].step(y,facies[ys,x],where="mid");axes[1].set(xlabel="Vertical pixel coordinate",ylabel="Facies label")
            fig.savefig(folder/f"profile_{index:03d}.png",dpi=145);plt.close(fig)
    return field_stats


def run_one(dataset,run):
    source=PARENT/dataset/run;frozen=PHYSICS/dataset/run;folder=OUTPUT/dataset/run;folder.mkdir(parents=True,exist_ok=True)
    refs=np.load(source/"references.npz");baseline,clear,facies,volume=(refs[k] for k in ("baseline","clearwater","facies","pore_volume"))
    domain=volume>0;rgb=np.load(source/"rgb.npy",mmap_mode="r");aq=np.load(source/"aq_volume.npy",mmap_mode="r");gas=np.load(source/"gas_volume.npy",mmap_mode="r")
    meta=read(source/"prepared.json");fit_path=frozen/"fits/joint_chromatic_od__mass_only/result.json";fit=read(fit_path);params=fit["parameters"]
    ds=dilution_factor(meta["frames"],"pore_volume",params["beta"])
    nom_joint=build_joint_reference(baseline,clear,facies,domain_mask=domain)
    nom_candidate=build_dark_reference(baseline,clear,facies,domain)
    contexts={"nominal":(nom_joint,nom_candidate)}
    for scenario in SCENARIOS:
        if scenario.startswith("baseline"):
            b=perturb(baseline,scenario,20260912)
            contexts[scenario]=(build_joint_reference(b,clear,facies,domain_mask=domain),build_dark_reference(b,clear,facies,domain))
    write(folder/"reference.json",nom_candidate.diagnostics)
    np.savez_compressed(folder/"reference_support.npz",denominator=nom_candidate.denominator,reference_reliability=nom_candidate.reference_reliability,
                        baseline_measurement_reliability=measurement_reliability(baseline,nom_candidate.epsilon_linear))
    n,h,w=aq.shape;signal_store=np.lib.format.open_memmap(folder/"signal.npy",mode="w+",dtype=np.float32,shape=(n,h,w))
    selected=sorted(set([1,n//2,n-1]));above,below,labels=geometry_pairs(facies,dataset)
    rows=[]
    for i in range(n):
        with np.load(source/f"properties_{i:03d}.npz") as z:solub,density=z["solubility"],z["density"]
        av,gv=np.asarray(aq[i],float),np.asarray(gas[i],float);mask=av>0;pure=domain&(av>=.99*volume)
        constant_aq=float(np.sum((1-params["gas_saturation"])*gv*solub)*1000)
        constant_gas=float(np.sum(params["gas_saturation"]*gv*density)*1000)
        nominal_fields={};plus2_fields={}
        for scenario in SCENARIOS:
            jref,cref=contexts.get(scenario,contexts["nominal"])
            current=perturb(rgb[i],scenario,20260912+i) if scenario.startswith("current") else rgb[i]
            signals={"joint":joint_optical_signals(current,jref)["joint_chromatic_od"],"candidate":compute_candidate(current,cref)}
            if scenario=="nominal":
                signal_store[i]=signals["candidate"]
                if i==0:
                    for value in signals.values():np.testing.assert_allclose(value,0,atol=2e-6)
            for method,signal in signals.items():
                c=fixed_concentration(signal,params,ds[i],solub)
                if scenario=="nominal":nominal_fields[method]=c
                if scenario=="current_plus2":plus2_fields[method]=c
                aqueous=float(np.sum(c*av)*1000)+constant_aq
                row={"frame":i,"elapsed_h":meta["frames"][i]["elapsed_h"],"scenario":scenario,"method":method,
                     "aqueous_g":aqueous,"gas_g":constant_gas,"total_g":aqueous+constant_gas,
                     "injected_g":meta["frames"][i]["injected_mass_g"],
                     "mean_aqueous_concentration_g_l":float(np.sum(c*av)/av.sum()) if av.sum() else 0.,
                     "boundary":pair_contrast(c,above,below,pure)}
                if i in selected and mask.any():row["aqueous_concentration_q10_q50_q90_g_l"]=weighted_quantiles(c[mask],av[mask])
                rows.append(row)
        if i in selected:
            np.savez_compressed(folder/f"fields_{i:03d}.npz",joint_concentration_g_l=nominal_fields["joint"],candidate_concentration_g_l=nominal_fields["candidate"],
                                current_measurement_reliability=measurement_reliability(rgb[i],nom_candidate.epsilon_linear))
            render_selected(folder,run,i,rgb[i],facies,av,volume,nominal_fields,plus2_fields,above,below)
        print(json.dumps({"run":run,"frame":i+1,"total":n}),flush=True)
    signal_store.flush();write(folder/"rows.json",rows)
    lookup={(r["frame"],r["scenario"],r["method"]):r for r in rows};summary={}
    for scenario in SCENARIOS[1:]:
        summary[scenario]={}
        for method in ("joint","candidate"):
            changes=np.array([lookup[(i,scenario,method)]["total_g"]-lookup[(i,"nominal",method)]["total_g"] for i in range(1,n)])
            final=lookup[(n-1,scenario,method)];nominal=lookup[(n-1,"nominal",method)]
            contrast_delta=None
            if final["boundary"]["median_lower_minus_upper_g_l"] is not None:
                contrast_delta=final["boundary"]["median_lower_minus_upper_g_l"]-nominal["boundary"]["median_lower_minus_upper_g_l"]
            summary[scenario][method]={"mean_absolute_mass_change_g":float(np.mean(abs(changes))),"maximum_absolute_mass_change_g":float(np.max(abs(changes))),
                "maximum_change_percent_final_injected":float(100*np.max(abs(changes))/meta["frames"][-1]["injected_mass_g"]),
                "final_mass_change_g":final["total_g"]-nominal["total_g"],"final_boundary_contrast_change_g_l":contrast_delta}
    # Honest candidate shift at unchanged mass parameters is separate from its
    # sensitivity to further perturbation; reduced noise is not improved truth.
    nominal_final={m:lookup[(n-1,"nominal",m)] for m in ("joint","candidate")}
    final_mass_shift=nominal_final["candidate"]["total_g"]-nominal_final["joint"]["total_g"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for method in ("joint","candidate"):
        times=[meta["frames"][i]["elapsed_h"] for i in range(n)]
        vals=np.array([[lookup[(i,s,method)]["total_g"] for i in range(n)] for s in SCENARIOS])
        axes[0].plot(times,vals[0],label=method);axes[0].fill_between(times,vals[1:].min(axis=0),vals[1:].max(axis=0),alpha=.16)
    axes[0].plot(times,[f["injected_mass_g"] for f in meta["frames"]],color="black",ls="--",label="Injected reference")
    axes[0].set(xlabel="Elapsed hours",ylabel="Detected mass [g]",title="Frozen-curve nominal and sensitivity envelope");axes[0].legend()
    names=list(SCENARIOS[1:]);xx=np.arange(len(names));
    for off,method in [(-.18,"joint"),(.18,"candidate")]:
        axes[1].bar(xx+off,[summary[s][method]["maximum_change_percent_final_injected"] for s in names],width=.36,label=method)
    axes[1].set_xticks(xx,[s.replace("current_","cur ").replace("baseline_","base ").replace("uniform2_seeded","random2") for s in names],rotation=65,ha="right",fontsize=8)
    axes[1].set(ylabel="Max |mass change| / final injection [%]",title="Perturbation sensitivity, all nonbaseline frames");axes[1].legend()
    fig.suptitle(run.upper()+": scenarios are not calibrated camera-noise probabilities")
    fig.savefig(folder/"mass_sensitivity.png",dpi=145);plt.close(fig)
    result={"dataset":dataset,"run":run,"source":str(source),"frozen_fit":str(fit_path),"frozen_fit_sha256":hashlib.file_digest(fit_path.open("rb"),"sha256").hexdigest(),
            "parameters_unchanged":params,"new_candidate_recalibrated":False,"reference":nom_candidate.diagnostics,"frames":n,
            "signal_array":str(folder/"signal.npy"),"rows_json":str(folder/"rows.json"),"selected_frames":selected,
            "comparison_plots":[str(folder/f"comparison_{i:03d}.png") for i in selected],"profile_plots":[str(p) for p in folder.glob("profile_*.png")],
            "mass_plot":str(folder/"mass_sensitivity.png"),"fixed_boundary_labels_upper_lower":labels,
            "nominal_final":nominal_final,"candidate_final_mass_shift_g":final_mass_shift,"sensitivity":summary}
    write(folder/"summary.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only",action="store_true")
    args=parser.parse_args()
    cv2.setNumThreads(2)
    if args.render_only:
        rerender_existing()
        return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(OUTPUT/"status.json",{"status":"running","started_epoch":time.time()})
    results=[]
    for dataset,run in [("ffac","ac32"),("ffac","ac61"),("ffum","run18"),("ffum","run20")]:
        results.append(run_one(dataset,run))
        write(OUTPUT/"manifest.json",{"completed":False,"runs":results})
    manifest={"completed":True,"protocol":"Dark current or baseline RGB channels (<8/255 nominal sRGB) perturbed by +/-1/255, +/-2/255 or deterministic independent uniform +/-2/255; NOT a calibrated camera-noise distribution",
              "baseline_scenario":"Perturbed baseline also rebuilds reference denominators and endpoint alignment; clear water, phase masks, dose and fitted curve remain frozen",
              "candidate":"asinh in linear light, epsilon from2/255 sRGB; reference-derived span and one reference endpoint gain; adaptive current derivative is measurement reliability",
              "curve":"Original joint_chromatic_od mass_only curve, gas occupancy and dilution unchanged; new candidate has no mass refit",
              "api":"build_dark_reference(baseline,clear,facies,domain) -> reference; compute_candidate(current,reference) -> aligned dimensionless signal",
              "runs":results}
    write(OUTPUT/"manifest.json",manifest);write(OUTPUT/"status.json",{"status":"completed","finished_epoch":time.time()})
    print(json.dumps({"completed":True,"manifest":str(OUTPUT/"manifest.json")}))


def rerender_existing():
    """Refresh display ranges from completed numeric data, with no refitting."""
    manifest=read(OUTPUT/"manifest.json")
    if not manifest["completed"]:raise ValueError("Numeric campaign must finish first")
    manifest["numeric_completed"]=True;manifest["render_review_complete"]=False
    write(OUTPUT/"manifest.json",manifest)
    for record in manifest["runs"]:
        dataset,run=record["dataset"],record["run"]
        folder=OUTPUT/dataset/run;source=PARENT/dataset/run
        refs=np.load(source/"references.npz");baseline,clear,facies,volume=(refs[k] for k in ("baseline","clearwater","facies","pore_volume"))
        domain=volume>0;meta=read(source/"prepared.json");p=record["parameters_unchanged"]
        dilution=dilution_factor(meta["frames"],"pore_volume",p["beta"])
        joint=build_joint_reference(baseline,clear,facies,domain_mask=domain)
        candidate=build_dark_reference(baseline,clear,facies,domain)
        rgb=np.load(source/"rgb.npy",mmap_mode="r");aq=np.load(source/"aq_volume.npy",mmap_mode="r")
        above,below,_=geometry_pairs(facies,dataset);stats={}
        for i in record["selected_frames"]:
            saved=np.load(folder/f"fields_{i:03d}.npz")
            nominal={m:saved[m+"_concentration_g_l"] for m in ("joint","candidate")}
            pert=perturb(rgb[i],"current_plus2",20260912+i)
            with np.load(source/f"properties_{i:03d}.npz") as z:solub=z["solubility"]
            signals={"joint":joint_optical_signals(pert,joint)["joint_chromatic_od"],"candidate":compute_candidate(pert,candidate)}
            plus2={m:fixed_concentration(value,p,dilution[i],solub) for m,value in signals.items()}
            stats[str(i)]=render_selected(folder,run,i,rgb[i],facies,aq[i],volume,nominal,plus2,above,below)
        record["render_statistics"]=stats
        record["candidate_signal_complete"]=True
        record["comparison_plots"]=[str(folder/f"comparison_{i:03d}{suffix}.png") for i in record["selected_frames"] for suffix in ("","_detail")]
        record["highlight_plot"]=str(folder/f"comparison_{record['frames']-1:03d}_detail.png")
        write(folder/"summary.json",record)
        print(json.dumps({"rerendered":run}),flush=True)
    manifest["render_review_complete"]=True
    manifest["source_module_sha256"]=hashlib.file_digest(Path(__file__).open("rb"),"sha256").hexdigest()
    write(OUTPUT/"manifest.json",manifest)


if __name__=="__main__":main()
