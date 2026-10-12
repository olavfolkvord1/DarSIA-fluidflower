"""Frozen Studio phase labels and registered optics for a diagnostic mass campaign.

The native RGBA files encode exact, categorical labels. Only verified palette
values are decoded, never a screenshot/blended colour or a probability from alpha.
Optical concentration is computed separately from the original corrected RGB.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import gc
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image

REPO = series_root("ffac")

from darsia.fluidflower.phase_validation import _source_jobs, PHASE_FILL_COLORS
from darsia.fluidflower.tools.observation_support import btb_acidification_coordinate as btb
from darsia.fluidflower.tools.phase_geometry_contract import require_matching_geometry

CAMPAIGN = local_cache_root() / 'phase_diagnostics/all_series_gas_absence_20260911T145838Z'
RUNS = {"ffac": ("ac32", "ac61"), "ffum": ("run18", "run20")}
METHODS = ("baseline_btb", "clearwater_gain", "layer_gain", "indicator_od")


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with resolve_review_path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def progress(root, stage, **kw):
    record = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **kw}
    write(Path(root)/"progress.json", record)
    print(json.dumps(record), flush=True)


def small(array, width, interpolation=cv2.INTER_AREA):
    h, w = array.shape[:2]
    return cv2.resize(array, (width, round(h*width/w)), interpolation=interpolation)


def crop_rgb(path, box):
    x,y,w,h = box
    with Image.open(path) as im:
        if im.size == (w,h):
            result = np.asarray(im.convert("RGB"), np.float32)/255
        elif im.width >= x+w and im.height >= y+h:
            result = np.asarray(im.crop((x,y,x+w,y+h)).convert("RGB"), np.float32)/255
        else:
            raise ValueError(f"Unexpected RGB grid: {path}: {im.size}")
    return result


def decode_labels(path, box):
    x,y,w,h = box
    with Image.open(path) as im:
        if im.mode != "RGBA" or im.width < x+w or im.height < y+h:
            raise ValueError(f"Not a native categorical RGBA asset: {path}")
        rgba = np.asarray(im.crop((x,y,x+w,y+h)))
    result = np.zeros((h,w), np.uint8)
    known = rgba[...,3] == 0
    for name, label in (("aqueous",1),("gas",2)):
        match = np.all(rgba == np.asarray(PHASE_FILL_COLORS[name], np.uint8), axis=-1)
        result[match] = label
        known |= match
    if not known.all():
        colors = np.unique(rgba[~known],axis=0)
        raise ValueError(f"Unknown/noncategorical phase colours: {colors[:10].tolist()}")
    return result


def srgb_linear(rgb):
    x = np.clip(rgb, 0, 1)
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)


def reference_context(baseline, clear, facies):
    # References alone define all optical normalization. No injected mass or
    # late-time optical endpoint is used to manufacture a concentration signal.
    local = cv2.GaussianBlur(clear, (0,0), 1.5)
    local = np.maximum(local, .025)
    layers = np.empty_like(local)
    layer_records = []
    for label in np.unique(facies):
        mask = facies == label
        median = np.median(clear[mask], axis=0)
        layers[mask] = np.maximum(median,.025)
        layer_records.append({"facies": int(label), "clear_rgb":median.tolist(),
                              "baseline_rgb":np.median(baseline[mask],axis=0).tolist()})
    b_lin, c_lin = srgb_linear(baseline), srgb_linear(local)
    od0 = -np.log((b_lin+.001)/(c_lin+.001))
    chromatic = .5*(od0[...,0]+od0[...,1])-od0[...,2]
    # Positive blue-indicator absorption span. Uninformative spans are flagged,
    # and a bounded floor prevents bright grains/dark pixels dividing by zero.
    span = np.maximum(cv2.GaussianBlur(chromatic,(0,0),1.5), .15)
    return local, layers, span, layer_records


def optical_signals(current, baseline, local, layers, span):
    delta = btb(current)-btb(baseline)
    gain = btb(current/local)-btb(baseline/local)
    layer = btb(current/layers)-btb(baseline/layers)
    change = np.log((srgb_linear(current)+.001)/(srgb_linear(baseline)+.001))
    od = (.5*(change[...,0]+change[...,1])-change[...,2])/span
    return np.stack([delta,gain,layer,od]).astype(np.float32)


def frame_state(source, item):
    from darsia.fluidflower.tools.phase_to_mass_pilot import _injected_mass
    if item["selection_status"] == "pre_injection_negative_control":
        date = source.experiment.experiment_start
        injected = 0.0
    else:
        date, injected = _injected_mass(source.experiment, Path(item["source_path"]))
    p = source.experiment.pressure_temperature_protocol
    state, gradient = p.get_state(date), p.get_gradient(date)
    return {"injected_mass_g":1000*injected, "pressure_bar":float(state.pressure),
            "temperature_c":float(state.temperature), "pressure_gradient_bar_per_m":float(gradient.pressure),
            "temperature_gradient_c_per_m":float(gradient.temperature), "acquired":str(date)}


@contextmanager
def open_reference_source(transfer, root, dataset, run):
    """Resolve the verified UM NAS migration in a private config copy only."""
    if dataset != "ffum":
        with transfer._open_source(dataset,run,require_clearwater=True) as source:
            yield source
        return
    import tomllib
    original_root=transfer.FFUM_WORKFLOW_ROOT
    original=original_root/"config"/f"{run}.toml"
    content=original.read_text(encoding="utf-8-sig")
    config=tomllib.loads(content)
    if Path(config["data"]["folder"]).is_dir():
        with transfer._open_source(dataset,run,require_clearwater=True) as source:
            yield source
        return
    relocated={"run18":r"\\UNAS-Pro-8\UiB_FluidFlower\FF_UM\Experiments\Run18_230412\230420_CO2_injection",
               "run20":r"\\UNAS-Pro-8\UiB_FluidFlower\FF_UM\Experiments\Run20_230504\230512_CO2_injection"}[run]
    if not (Path(relocated)/config["data"]["baseline"]).is_file():
        raise FileNotFoundError(f"Verified relocated UM baseline is missing: {relocated}")
    mirror=Path(root)/"resolved_um_config"
    (mirror/"config").mkdir(parents=True,exist_ok=True)
    old=config["data"]["folder"]
    content=content.replace("folder = '"+old+"'", "folder = '"+relocated+"'")
    content=content.replace('"./data/', '"'+original_root.as_posix()+'/data/')
    (mirror/"config"/f"{run}.toml").write_text(content,encoding="utf-8")
    write(mirror/f"{run}_provenance.json",{"source":str(original),"source_sha256":sha(original),
          "old_folder":old,"resolved_folder":relocated,"original_config_modified":False})
    transfer.FFUM_WORKFLOW_ROOT=mirror
    try:
        with transfer._open_source(dataset,run,require_clearwater=True) as source:
            yield source
    finally:
        transfer.FFUM_WORKFLOW_ROOT=original_root


def prepare_run(root, dataset, run, width=1536):
    import darsia
    from darsia.fluidflower.tools import schema2_phase_transfer_campaign as transfer
    from darsia.fluidflower.tools.materialize_phase_mass_inputs import _registered_grid
    from darsia.fluidflower.mass_materialization import _mass_analysis
    root = Path(root)
    dest = root/dataset/run
    if (dest/"prepared.json").exists():
        meta = read(dest/"prepared.json")
        if meta["analysis_width"] != width:
            raise ValueError("Existing dataset has a different analysis width")
        return meta
    dest.mkdir(parents=True,exist_ok=True)
    prepared = read(CAMPAIGN/dataset/"prepared.json")
    model_root = Path(prepared["output_root"])
    request = read(model_root/"training_request.json")
    job = next(j for j in _source_jobs(model_root/"validation") if j["run"]==run)
    manifest_path = Path(job["review_output_dir"])/"native_frames"/run/"manifest.json"
    manifest = read(manifest_path)
    if dataset == "ffac" and any(row.get("rig_domain_revision") for row in manifest.get("frame_sources", ())):
        raise ValueError("Historical fixed-grid phase-to-mass preparation cannot quantify the new FF_AC rig-domain revision; use source-bound area weights and verified physical coverage.")
    manifest_hash = sha(manifest_path)
    assets = []
    for directory in (Path(request["cache_root"])/"assets").glob(f"{dataset}_{run}_*"):
        staged = directory/"staged_assets.json"
        if staged.exists() and read(staged).get("source_manifest_sha256")==manifest_hash:
            assets.append(directory)
    if len(assets)!=1:
        raise ValueError(f"No unique frozen Studio assets for {run}")
    asset = assets[0]
    box = manifest["content_box_xywh"]
    baseline = small(crop_rgb(asset/"pre_injection.jpg",box),width)
    progress(root,"load_rig_and_clearwater",dataset=dataset,run=run)
    with open_reference_source(transfer,root,dataset,run) as source:
        geometry_binding = require_matching_geometry(
            manifest, source.correction_contract, context=f"{dataset}/{run} frozen phase-to-mass inputs")
        if source.baseline.shape[:2] != tuple(box[2:][::-1]):
            raise ValueError("Rig and Studio grids do not coincide")
        grid = _registered_grid(transfer,source)
        facies_native = np.asarray(source.rig.facies.img)
        facies = small(facies_native.astype(np.float32),width,cv2.INTER_NEAREST).astype(np.int16)
        domain = ~np.isin(facies_native, [0] if dataset=="ffac" else [1])
        depth = np.asarray(getattr(source.rig.geometry.depth,"img",source.rig.geometry.depth),np.float32)
        porosity = np.asarray(getattr(source.rig.porosity,"img",source.rig.porosity),np.float32)
        depth = np.broadcast_to(depth,domain.shape)
        porosity = np.broadcast_to(porosity,domain.shape)
        if not np.isfinite(depth[domain]).all() or not np.isfinite(porosity[domain]).all():
            raise ValueError("Nonfinite geometry inside porous domain")
        if np.any(depth[domain]<0) or np.any((porosity[domain]<0)|(porosity[domain]>1)):
            raise ValueError("Invalid depth/porosity units")
        pore_depth = np.where(domain,depth*porosity,0).astype(np.float32)
        h,w = baseline.shape[:2]
        pixel_area = grid["width_m"]*grid["height_m"]/(h*w)
        pore_volume = small(pore_depth,width)*pixel_area
        clear = small(np.asarray(source.clearwater,np.float32),width)
        rebuilt_baseline = small(source.baseline,width)
        baseline_error = np.abs(rebuilt_baseline-baseline)
        # JPEG/reduction difference is expected; a changed baseline/correction is not.
        if float(np.quantile(baseline_error,.95))>.08:
            raise ValueError(f"Studio and reconstructed reference radiometry differ for {run}")
        local,layers,span,layer_records = reference_context(baseline,clear,facies)
        ref_record = {"baseline_reconstruction_mae":float(baseline_error.mean()),
                      "baseline_reconstruction_q95":float(np.quantile(baseline_error,.95)),
                      "clearwater_contract":source.clearwater_contract,
                      "layers":layer_records,
                      "weak_od_span_fraction":float(np.mean(span<=.15)),
                      "clearwater_is_saturated_CO2_endpoint":False,
                      "reference_alignment":"matching frozen and live registration geometry; see geometry_binding",
                      "geometry_binding":geometry_binding}
        np.savez_compressed(dest/"references.npz",baseline=baseline,clearwater=clear,
                            facies=facies,pore_volume=pore_volume,od_span=span)
        arrays = {"rgb":(len(manifest["frame_sources"]),h,w,3),
                  "signal":(len(manifest["frame_sources"]),4,h,w),
                  "aq_volume":(len(manifest["frame_sources"]),h,w),
                  "gas_volume":(len(manifest["frame_sources"]),h,w)}
        mmap = {k:np.lib.format.open_memmap(dest/f"{k}.npy",mode="w+",dtype=np.float32,shape=s)
                for k,s in arrays.items()}
        rows, sources = [], []
        map_root = Path(job["phase_map_output_dir"])/"native_phase_maps"/run/f"A_{job['params_sha256'][:16]}"
        for i,item in enumerate(manifest["frame_sources"]):
            rgb_path = asset/f"frame_{i:03d}.jpg"
            if not rgb_path.exists():
                rgb_path = manifest_path.parent/f"frame_{i:03d}.jpg"
            map_path = map_root/f"frame_{i:03d}.png"
            labels = decode_labels(map_path,box)
            current = small(crop_rgb(rgb_path,box),width)
            if item["selection_status"]=="pre_injection_negative_control":
                # Use the exact same retained pre-injection RGB as the reference;
                # verify the displayed negative control contains no phase labels.
                if np.any(labels):
                    raise ValueError(f"Baseline phase negative control is nonzero: {run}")
                current = baseline.copy()
            aq_volume = small(pore_depth*(labels==1),width)*pixel_area
            gas_volume = small(pore_depth*(labels==2),width)*pixel_area
            aq_area = np.count_nonzero((labels>0)&domain)*grid["width_m"]*grid["height_m"]/domain.size
            mmap["rgb"][i]=current
            mmap["signal"][i]=optical_signals(current,baseline,local,layers,span)
            mmap["aq_volume"][i]=aq_volume
            mmap["gas_volume"][i]=gas_volume
            state = frame_state(source,item)
            analyzer = _mass_analysis((h,w),grid["width_m"],grid["height_m"],
                     state["pressure_bar"],state["temperature_c"],
                     state["pressure_gradient_bar_per_m"],state["temperature_gradient_c_per_m"])
            density = np.broadcast_to(np.asarray(analyzer.density_gaseous_co2), (h,w))
            solubility = np.broadcast_to(np.asarray(analyzer.solubility_co2), (h,w))
            np.savez_compressed(dest/f"properties_{i:03d}.npz",density=density.astype(np.float32),solubility=solubility.astype(np.float32))
            native_volume = float(np.sum(pore_depth*(labels>0),dtype=np.float64))*grid["width_m"]*grid["height_m"]/domain.size
            reduced_volume = float(aq_volume.sum(dtype=np.float64)+gas_volume.sum(dtype=np.float64))
            if not np.isclose(native_volume,reduced_volume,rtol=2e-5,atol=1e-12):
                raise ValueError("Pore-volume conservation failed during area reduction")
            rows.append({"frame_index":i,"elapsed_h":item["elapsed_h"],"source_name":item["source_name"],
                         "affected_area_m2":float(aq_area),"affected_pore_volume_m3":reduced_volume,
                         "gas_pore_volume_m3":float(gas_volume.sum(dtype=np.float64)),**state})
            sources.append({"rgb":str(rgb_path),"rgb_sha256":sha(rgb_path),"map":str(map_path),"map_sha256":sha(map_path)})
            progress(root,"prepare_frame",dataset=dataset,run=run,frame=i+1,frames=len(manifest["frame_sources"]))
        for array in mmap.values(): array.flush()
        del mmap
        metadata = {"schema":1,"dataset":dataset,"run":run,"analysis_width":width,"grid":grid,
                    "correction_contract":dict(source.correction_contract),"geometry_binding":geometry_binding,
                    "model_id":model_root.name,"model_root":str(model_root),"manifest":str(manifest_path),
                    "manifest_sha256":manifest_hash,"source_assets":sources,"frames":rows,
                    "reference":ref_record,"methods":METHODS,"porous_volume_l":1000*float(pore_volume.sum(dtype=np.float64)),
                    "pore_volume_conservation_checked":True,"phase_labels":"exact native categorical palette; not blended screenshots or probabilities",
                    "domain":"porous facies only; free-water layer excluded", "reference_mass":"cumulative injected CO2; no loss correction",
                    "concentration_status":"inverse-calibrated estimates, no independent local concentration standards",
                    "clearwater_reference_role":"sand/illumination reference, never a known high-CO2 concentration"}
        write(dest/"prepared.json",metadata)
    gc.collect()
    return metadata


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--width",type=int,default=1536)
    p.add_argument("--run",choices=[r for rows in RUNS.values() for r in rows])
    args=p.parse_args()
    cv2.setNumThreads(4)
    for dataset,runs in RUNS.items():
        for run in runs:
            if args.run is None or args.run==run:
                prepare_run(args.output,dataset,run,args.width)
