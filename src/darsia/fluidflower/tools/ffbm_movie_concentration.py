"""BM-specific optical hypothesis and inverse mass calibration for a movie.

The dual-indicator signal tests red-versus-green acidification, with response
normalization from the registered wash and same-format baseline. It is not a
pH calibration, and wash water is not a known dissolved-CO2 endpoint.
This module does not invoke the BTB TitrationFlash or reuse an AC/UM fitted curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from darsia.fluidflower.tools.phase_mass_joint_optics import srgb_to_linear, _regularized_span


@dataclass
class BMReference:
    baseline_response: np.ndarray
    denominator: np.ndarray
    direction: np.ndarray
    domain: np.ndarray
    sigma: float
    epsilon: float
    diagnostics: dict


def _response(rgb, epsilon):
    image=np.asarray(rgb,np.float32)
    if image.ndim!=3 or image.shape[-1]!=3 or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError("BM optical input must be finite normalized RGB")
    response=np.arcsinh(srgb_to_linear(image)/epsilon)
    return response-response.mean(axis=-1,keepdims=True)


def build_reference(baseline, clearwater, facies, domain, *, spacing_mm, sigma_mm=3., epsilon_dn=4.):
    if np.shape(baseline)!=np.shape(clearwater) or np.shape(baseline)[:2]!=np.shape(facies) or np.shape(facies)!=np.shape(domain):
        raise ValueError("BM baseline, wash, facies, and domain grids differ")
    if min(spacing_mm)<=0 or sigma_mm<=0 or epsilon_dn<=0:
        raise ValueError("Invalid BM optical scale")
    epsilon=float(srgb_to_linear(np.array(epsilon_dn/255,np.float32)))
    b,c=_response(baseline,epsilon),_response(clearwater,epsilon)
    vector=c-b
    domain=np.asarray(domain,bool)
    valid=domain&(np.max(baseline,axis=-1)<.995)&(np.max(clearwater,axis=-1)<.995)
    # Washing removes both dyes; its full RGB direction is not the same as
    # acidification of the BTB + methyl-red mixture. The red-green projection
    # is an explicit optical hypothesis for BM's observed green-to-red change.
    # It remains subject to mass holdouts; no pH or known concentration follows.
    direction=np.array([1.,-1.,0.],np.float32)/np.sqrt(2.)
    sigma=float(sigma_mm/np.mean(spacing_mm))
    raw=vector@direction
    denominator,reliability,diagnostics=_regularized_span(raw,valid,facies,sigma,.15)
    if np.mean((raw>.15)[domain])<.25:
        raise ValueError("BM wash-baseline red-green reference span has insufficient support")
    diagnostics.update(method="bm_red_green_soft_log_reference_normalized_v1",direction_rgb=direction.tolist(),
        direction_policy="Fixed red-versus-green acidification hypothesis; washing direction is not used as an acidification axis.",
        reference_inputs=["same_format_baseline","registered_clearwater","facies","porous_domain"],
        smoothing_sigma_mm=sigma_mm,epsilon_dn=epsilon_dn,reference_reliability_mean=float(reliability[domain].mean()),
        chemical_calibration=False,clearwater_is_known_co2_endpoint=False,
        limitations="One optical coordinate for a dual indicator; scalar monotonicity must be checked, especially with precipitation and indicator separation.")
    return BMReference(b,denominator,direction,domain,sigma,epsilon,diagnostics)


def optical_signal(rgb, reference):
    response=_response(rgb,reference.epsilon)
    if response.shape!=reference.baseline_response.shape:
        raise ValueError("BM current photograph grid differs from its optical reference")
    delta=(response-reference.baseline_response)@reference.direction
    # Normalize smoothing by physical domain support to avoid free-water border bleed.
    weights=cv2.GaussianBlur(reference.domain.astype(np.float32),(0,0),reference.sigma)
    signal=cv2.GaussianBlur(np.where(reference.domain,delta,0).astype(np.float32),(0,0),reference.sigma)/np.maximum(weights,.001)
    signal=np.where(reference.domain,signal/reference.denominator,0)
    return signal.astype(np.float32)


def gas_density(pressure_bar, shape):
    # Exact original BM formula; all concentrations/densities here are kg/m3 == g/L.
    depth_from_top=np.linspace(0,1.5,shape[0])[:,None]
    density=1.805726990977443*(float(pressure_bar)+.101325*depth_from_top)-.009218969932330845
    return np.broadcast_to(density,shape).astype(np.float32)


def concentration(signal, parameters, dilution=1.):
    p=parameters
    if p["endpoint"]<=0 or not 0<p["amplitude"]<=1 or p["gamma"]<=0 or not 0<dilution<=1:
        raise ValueError("Invalid BM concentration curve")
    shifted=np.asarray(signal)-float(p.get("signal_offset",0.))
    return (1.8*p["amplitude"]*np.clip(shifted/p["endpoint"],0,1)**p["gamma"]*dilution).astype(np.float32)


def mass_budget(c, aqueous_volume_m3, gas_volume_m3, gas_density_g_l, gas_saturation):
    c,av,gv,density=[np.asarray(a) for a in (c,aqueous_volume_m3,gas_volume_m3,gas_density_g_l)]
    if any(a.shape!=c.shape or not np.isfinite(a).all() for a in (c,av,gv,density)):
        raise ValueError("BM physical fields must be finite on the same grid")
    if not 0<=gas_saturation<=1 or (av<0).any() or (gv<0).any():
        raise ValueError("Invalid BM phase volume or gas saturation")
    gas=float(np.sum(gv*density,dtype=np.float64)*1000*gas_saturation)
    residual=float(np.sum(gv,dtype=np.float64)*1000*1.8*(1-gas_saturation))
    aqueous=float(np.sum(av*c,dtype=np.float64)*1000)+residual
    return dict(aqueous_mass_g=aqueous,gas_mass_g=gas,total_mass_g=aqueous+gas,residual_liquid_in_gas_g=residual)


def probe_references(root):
    from pathlib import Path
    import json
    root=Path(root)
    z=dict(np.load(root/"physical_references.npz"))
    q=np.load(root/"optical_controls/reduced_controls.npz")
    result,params={},{}
    output=root/"optical_signal_probe.json"
    earlier=root/"optical_signal_probe_rejected_wash_axis.json"
    if output.exists() and not earlier.exists():earlier.write_bytes(output.read_bytes())
    for fmt in ("tif","jpg"):
        b=q["pair_0_"+fmt].astype(np.float32)/255
        r=build_reference(b,z["clearwater"],z["facies"],z["pore_volume"]>0,spacing_mm=(1500/823,2800/1536))
        assert np.max(abs(optical_signal(b,r)))==0
        result[fmt]={k:np.percentile(optical_signal(q[k].astype(np.float32)/255,r)[z["pore_volume"]>0],[1,50,90,99]).tolist()
            for k in q.files if k.endswith(fmt)}
        params[fmt]=r.diagnostics
    z["baseline"]=q["pair_0_jpg"].astype(np.float32)/255
    np.savez_compressed(root/"physical_references_jpg.npz",**z)
    output.write_text(json.dumps(dict(reference=params,domain_signal_percentiles_1_50_90_99=result),indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default="C:/FFAC_cache/phase_diagnostics/concentration_movies_ffbm_c1_20260914")
    probe_references(p.parse_args().root)
