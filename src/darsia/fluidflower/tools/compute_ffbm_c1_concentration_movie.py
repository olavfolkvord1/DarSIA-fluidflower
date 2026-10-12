"""C1 movie computation using an explicitly bound, completed Studio model.

Photographs are read from one dense JPG acquisition series. New concentration
parameters are fitted for BM; no AC/UM curve or BTB-only chemistry is imported.
"""
from __future__ import annotations
from darsia.fluidflower.review_storage import resolve_review_path
from darsia.fluidflower.series_paths import local_cache_root, series_path

import argparse
import os
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from darsia.fluidflower.tools.ffbm_movie_concentration import build_reference,optical_signal,gas_density,concentration,mass_budget
from darsia.fluidflower.tools.prepare_ffbm_c1_concentration_movie import ROOT,sha,write,rgb
from darsia.fluidflower.tools.concentration_movie_compute import save_npz


def read(path):
    return json.loads(resolve_review_path(path).read_text(encoding="utf-8-sig"))


class BMPhysics:
    def __init__(self,root,*,resolve_path=None):
        resolve=resolve_path or Path
        self.root=Path(root)
        self.meta=read(resolve(self.root/"physical_references.json"))
        self.refs=dict(np.load(resolve(self.root/"physical_references_jpg.npz")))
        self.pv=self.refs["pore_volume"]
        self.shape=self.pv.shape
        self.size=self.shape[::-1]
        self.ref=build_reference(self.refs["baseline"],self.refs["clearwater"],self.refs["facies"],self.pv>0,
            spacing_mm=self.meta["analysis_spacing_mm"])
        preflight=read(resolve(self.root/"preflight.json"))
        self.native_depth=np.load(resolve(preflight["physical_geometry"]["depth_source"]))
        self.native_domain=np.load(resolve(preflight["physical_geometry"]["labels_source"]))!=0
        self.pixel_area=2.8*1.5/np.prod(self.shape)
        self.native_volume_density=.44*self.native_depth*self.native_domain

    def prepare(self,current,labels,row):
        if labels.shape!=self.native_depth.shape or current.shape!=(*labels.shape,3):
            raise ValueError("Native BM photograph, labels and measured depth must have identical geometry")
        if not np.isin(labels,[0,1,2]).all() or np.any(labels[~self.native_domain]):
            raise ValueError("BM labels contain unknown phases or extend outside physical geometry")
        photo=cv2.resize(current,self.size,interpolation=cv2.INTER_AREA)
        signal=optical_signal(photo.astype(np.float32)/255,self.ref)
        av=cv2.resize(self.native_volume_density*(labels==1),self.size,interpolation=cv2.INTER_AREA)*self.pixel_area
        gv=cv2.resize(self.native_volume_density*(labels==2),self.size,interpolation=cv2.INTER_AREA)*self.pixel_area
        aqf=np.divide(av,self.pv,out=np.zeros(self.shape),where=self.pv>0)
        gf=np.divide(gv,self.pv,out=np.zeros(self.shape),where=self.pv>0)
        if np.max(aqf+gf)>1.0001:
            raise ValueError("BM phase fractions exceed measured pore volume")
        pressure=float(np.interp(row["elapsed_h"],self.refs["pressure_elapsed_h"],self.refs["pressure_bar"]))
        density=gas_density(pressure,self.shape)
        return dict(photograph_rgb=photo,signal=signal,aqueous_volume_m3=av.astype(np.float32),
            gas_volume_m3=gv.astype(np.float32),aqueous_fraction=aqf.astype(np.float32),gas_fraction=gf.astype(np.float32),
            gas_density_g_l=density),{**row,"pressure_bar":pressure,"affected_pore_volume_m3":float(av.sum()+gv.sum())}


class BMPhaseSequence:
    def __init__(self,model_root,baseline,*,backend="cpu",require_completion=True,native_manifest_path=None):
        if backend=="cpu":
            os.environ["CUDA_VISIBLE_DEVICES"]=""
            if require_completion:
                import torch
                torch.set_num_threads(max(1,cv2.getNumThreads()))
        from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
        from darsia.fluidflower.phase_runtime import load_physics_profile,load_run_context,phase_physics_sequence
        from darsia.fluidflower.tools.phase_inference_benchmark_backends import CpuFeaturePredictor,CudaFeaturePredictor,partition
        self.model_root=Path(model_root)
        if require_completion and not (self.model_root/"phase_aqueous_context_completion.json").is_file():
            raise ValueError("Movie requires the newly completed BM model with variant-B attachment")
        self.model=HierarchicalPhaseModel.load(self.model_root,adapter="generic")
        self.profile=load_physics_profile(self.model_root)
        if not require_completion and "aqueous_context_completion" in self.profile:
            from darsia.fluidflower.phase_aqueous_context_completion import split_profile
            self.profile,_=split_profile(self.profile)
        source_manifest=Path(native_manifest_path) if native_manifest_path is not None else resolve_review_path(local_cache_root() / 'ffbm_studio_pilot_20260908/V7-FF_BM-C1-optical-transfer-20260908/validation/review_assets/native_frames/c1/manifest.json')
        context,receipt=load_run_context(self.model_root,self.profile,"ffbm","c1",native_manifest_sha256=sha(source_manifest))
        if context.domain.shape!=baseline.shape[:2]:raise ValueError("BM model context differs from the movie correction grid")
        self.sequence=phase_physics_sequence(context,self.profile)
        self.sequence.bind_reference(baseline)
        self.predictor=(CpuFeaturePredictor if backend=="cpu" else CudaFeaturePredictor)(self.model,baseline)
        self.partition=partition
        self.previous=None
        self.last_time=-np.inf
        self.context_receipt=receipt
        self.index=0

    def apply(self,current,row):
        from darsia.fluidflower.phase_rendering import two_phase_masks
        elapsed=float(row["elapsed_h"])
        if elapsed<=self.last_time:
            raise ValueError("BM movie inference must be strictly chronological")
        started=time.monotonic()
        raw=self.predictor.predict(current,self.previous,current_key=self.index,previous_key=self.index-1)
        raw=self.partition(np.stack([raw["co2_presence_probability"],raw["gas_probability_conditional_on_co2"]],axis=-1))
        optical_seconds=time.monotonic()-started;physics_started=time.monotonic()
        output,report=self.sequence.apply_with_reference(raw,elapsed,current)
        masks=two_phase_masks(output)
        labels=np.zeros(current.shape[:2],np.uint8)
        labels[masks["aqueous"]]=1;labels[masks["gas"]]=2
        report={**report,"movie_stage_timing":dict(raw_inference_seconds=optical_seconds,physics_seconds=time.monotonic()-physics_started)}
        self.previous=current;self.last_time=elapsed;self.index+=1
        return labels,report

    def close(self):
        if hasattr(self.predictor,"close"):self.predictor.close()


def transfer_probe(root,model_root,backend="cpu",profile_only=False,require_completion=False):
    """Paired format experiment with the new base model; not final movie fields."""
    import gc
    root=Path(root);model_root=Path(model_root)
    source=root/"optical_controls"
    outputs=root/("stage_profile_"+backend if profile_only else "format_transfer_probe_B" if require_completion else "format_transfer_probe");outputs.mkdir(exist_ok=True)
    hashes={name:sha(model_root/name) for name in ("models/generic/co2_presence_signal.json","models/generic/gas_within_co2_signal.json","phase_reviewed_reference.json")}
    if require_completion:
        hashes.update({name:sha(model_root/name) for name in ("phase_aqueous_context_completion.json","aqueous_completion.pt")})
    result=dict(status="running",model_root=str(model_root),model_hashes=hashes,
        completion_attached_at_start=(model_root/"phase_aqueous_context_completion.json").exists(),
        temporal_policy="Three identical timestamps and same-format baseline for each encoding, previous image from same format. Diagnostic sparse sequence.",
        frames={})
    times=[-200/3600,3.] if profile_only else [-200/3600,3.,72.]
    for fmt in (("jpg",) if profile_only else ("tif","jpg")):
        baseline=rgb(source/f"pair_0_{fmt}.jpg")
        sequence=BMPhaseSequence(model_root,baseline,backend=backend,require_completion=require_completion)
        result["frames"][fmt]=[]
        try:
            for i,t in enumerate(times):
                started=time.monotonic();current=rgb(source/f"pair_{i}_{fmt}.jpg")
                labels,report=sequence.apply(current,dict(elapsed_h=t))
                target=outputs/f"pair_{i}_{fmt}_labels.png"
                cv2.imwrite(str(target),labels,[cv2.IMWRITE_PNG_COMPRESSION,1])
                row=dict(elapsed_h=t,labels_path=str(target),labels_sha256=sha(target),seconds=time.monotonic()-started,
                    counts={str(c):int(np.sum(labels==c)) for c in (0,1,2)},**report["movie_stage_timing"])
                result["frames"][fmt].append(row);write(outputs/f"pair_{i}_{fmt}.json",{**row,"physics_report":report})
                write(outputs/"report.json",result)
                print(f"BM format probe {fmt} {t:.3f}h: {row['seconds']:.1f}s, raw {row['raw_inference_seconds']:.1f}s, physics {row['physics_seconds']:.1f}s",flush=True)
        finally:sequence.close()
        del sequence,baseline,current,labels;gc.collect()
    result["comparisons"]=[]
    for i,t in ([] if profile_only else enumerate(times)):
        a=cv2.imread(str(outputs/f"pair_{i}_tif_labels.png"),cv2.IMREAD_UNCHANGED)
        b=cv2.imread(str(outputs/f"pair_{i}_jpg_labels.png"),cv2.IMREAD_UNCHANGED)
        scores={}
        for c,name in ((1,"aqueous"),(2,"gas")):
            aa,bb=a==c,b==c;union=int(np.count_nonzero(aa|bb))
            scores[name]=dict(iou=float(np.count_nonzero(aa&bb)/union) if union else 1.,
                tif_pixels=int(aa.sum()),jpg_pixels=int(bb.sum()),
                lost_vs_tif=int(np.count_nonzero(aa&~bb)),gained_vs_tif=int(np.count_nonzero(bb&~aa)))
        result["comparisons"].append(dict(elapsed_h=t,classes=scores))
    if hashes!={name:sha(model_root/name) for name in hashes}:raise ValueError("BM base weights changed during transfer probe")
    result["status"]="complete_B_model" if require_completion else "complete_exploratory_base_model"
    write(outputs/"report.json",result)
    return result


def transfer_figures(root,require_completion=False):
    from PIL import ImageDraw
    root=Path(root);out=root/("format_transfer_probe_B" if require_completion else "format_transfer_probe")
    for i in (1,2):
        photo=rgb(root/"optical_controls"/f"pair_{i}_jpg.jpg")
        tif=cv2.imread(str(out/f"pair_{i}_tif_labels.png"),cv2.IMREAD_UNCHANGED)
        jpg=cv2.imread(str(out/f"pair_{i}_jpg_labels.png"),cv2.IMREAD_UNCHANGED)
        def overlay(labels):
            color=np.empty_like(photo);color[:]=[0,0,0];color[labels==1]=[41,211,153];color[labels==2]=[246,82,167]
            return np.where((labels>0)[...,None],np.rint(.5*photo+.5*color).astype(np.uint8),photo)
        diff=(photo.astype(float)*.25).astype(np.uint8)
        lost=(tif==1)&(jpg!=1);gained=(jpg==1)&(tif!=1)
        diff[lost]=[0,180,255];diff[gained]=[255,190,0]
        arrays=[photo,overlay(tif),overlay(jpg),diff]
        headings=["Dense JPG photograph","TIFF-derived phase, 50% opacity","JPG-derived phase, 50% opacity","Aqueous: JPG removed cyan / added gold"]
        canvas=Image.new("RGB",(3200,459),(245,246,248));draw=ImageDraw.Draw(canvas)
        for n,(heading,a) in enumerate(zip(headings,arrays)):
            thumb=Image.fromarray(a);thumb.thumbnail((800,429));canvas.paste(thumb,(n*800,28));draw.text((n*800+7,6),heading,fill=(0,0,0))
        canvas.save(out/f"pair_{i}_comparison.jpg",quality=94)
    return [str(out/f"pair_{i}_comparison.jpg") for i in (1,2)]


def _benchmark_worker(job):
    import os
    root,model_root,worker=job;root=Path(root)
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        started=time.monotonic();baseline=rgb(root/"optical_controls/pair_0_jpg.jpg")
        sequence=BMPhaseSequence(model_root,baseline,backend="cpu",require_completion=False)
        times=[]
        try:
            for index,elapsed in enumerate((-200/3600,3.)):
                current=rgb(root/"optical_controls"/f"pair_{index}_jpg.jpg")
                start=time.monotonic();labels,report=sequence.apply(current,dict(elapsed_h=elapsed))
                times.append(dict(elapsed_h=elapsed,seconds=time.monotonic()-start,**report["movie_stage_timing"]))
        finally:sequence.close()
    return dict(worker=worker,pid=os.getpid(),seconds=time.monotonic()-started,frames=times)


def benchmark_parallel(root,model_root,workers=8):
    from concurrent.futures import ProcessPoolExecutor
    root=Path(root);started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(_benchmark_worker,[(str(root),str(model_root),i) for i in range(workers)]))
    duration=time.monotonic()-started
    positive=np.array([r["frames"][1]["seconds"] for r in results])
    # Both elapsed wall-clock batch throughput and steady-state positive-frame
    # throughput are reported. Neither excludes memory-bandwidth contention.
    result=dict(format="native-bm-eight-worker-throughput-v1",workers=workers,threads_per_worker=2,
        wall_seconds=duration,total_native_inferences=workers*2,results=results,
        positive_frame_median_seconds=float(np.median(positive)),positive_frame_max_seconds=float(positive.max()),
        estimated_full_movie_hours_with_six_second_correction=float((positive.max()+6)*3556/workers/3600),
        estimate_caveat="Base-model timing; final B, canonical-anchor replay, cache verification and video encoding add work.",
        phase_resolution="native 4260x7952, unchanged")
    write(root/f"parallel_throughput_{workers}.json",result)
    write(root/"parallel_throughput.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="results"},indent=2),flush=True)
    return result


def calibration_fields(root,model_root,backend="cpu"):
    root=Path(root);model_root=Path(model_root)
    data=read(root/"calibration_inputs/manifest.json")
    frames=sorted([r for r in data["frames"] if r["kind"]=="calibration_candidate"],key=lambda r:r["elapsed_h"])
    model_binding={str(p.relative_to(model_root)):sha(p) for p in
        [model_root/"phase_aqueous_context_completion.json",model_root/"aqueous_completion.pt",
        model_root/"models/generic/co2_presence_signal.json",model_root/"models/generic/gas_within_co2_signal.json"]}
    binding=hashlib.sha256(json.dumps(model_binding,sort_keys=True).encode()).hexdigest()
    out=root/"calibration_fields"/binding[:16];out.mkdir(parents=True,exist_ok=True)
    baseline=rgb(frames[0]["corrected_path"])
    phase=BMPhaseSequence(model_root,baseline,backend=backend)
    physics=BMPhysics(root)
    result=dict(format="bm-movie-calibration-fields-v1",complete=False,model_root=str(model_root),model_hashes=model_binding,
        model_binding_sha256=binding,context_receipt=phase.context_receipt,
        temporal_policy="strict chronological calibration sequence on same-format JPG inputs; full-movie sequence validated separately",
        optical_reference_sha256=sha(root/"physical_references_jpg.npz"),optical_method=physics.ref.diagnostics,frames=[])
    try:
        for i,row in enumerate(frames):
            started=time.monotonic();current=rgb(row["corrected_path"])
            if sha(row["corrected_path"])!=row["corrected_sha256"]:raise ValueError("BM calibration photo changed")
            labels,report=phase.apply(current,row)
            arrays,record=physics.prepare(current,labels,row)
            target=out/(row["id"]+".npz");save_npz(target,**arrays)
            label_path=out/(row["id"]+"_labels.png");cv2.imwrite(str(label_path),labels,[cv2.IMWRITE_PNG_COMPRESSION,1])
            record.update(fields_path=str(target),fields_sha256=sha(target),labels_path=str(label_path),labels_sha256=sha(label_path),
                model_binding_sha256=binding,phase_report=report,total_seconds=time.monotonic()-started)
            write(out/(row["id"]+".json"),record)
            result["frames"].append({k:v for k,v in record.items() if k!="phase_report"})
            write(out/"manifest.json",result)
            print(f"BM calibration field {i+1}/{len(frames)} {row['elapsed_h']:.3f}h {record['total_seconds']:.1f}s",flush=True)
        result["complete"]=True;write(out/"manifest.json",result)
    finally:phase.close()
    return out/"manifest.json"


def provisional_base_fields(root,model_root):
    """Reuse exact new base TIFF maps only for a named provisional optics fit."""
    from darsia.fluidflower.tools.phase_mass_overnight_data import decode_labels
    root=Path(root);model_root=Path(model_root)
    rows=read(root/"calibration_inputs/manifest.json")["frames"]
    rows=sorted([r for r in rows if r["kind"]=="calibration_candidate"],key=lambda r:r["elapsed_h"])
    source=model_root/"base_validation/postflight_manifest.json"
    base=read(source);job=next(j for j in base["jobs"] if j["dataset"]=="ffbm" and j["run"]=="c1")
    maps=model_root/"base_validation/review_assets/native_phase_maps/c1"/('A_'+job["params_sha256"][:16])
    out=root/"calibration_fields/provisional_base_tif_labels";out.mkdir(parents=True,exist_ok=True)
    physics=BMPhysics(root)
    manifest=dict(format="bm-provisional-optics-calibration-v1",complete=False,production_eligible=False,
        reason="Exploratory dense-JPG optics with corresponding fresh base TIFF masks, before final variant-B JPG re-inference.",
        model_root=str(model_root),base_validation_manifest_sha256=sha(source),model_sha256=base["model_sha256"],
        optical_reference_sha256=sha(root/"physical_references_jpg.npz"),optical_method=physics.ref.diagnostics,frames=[])
    h,w=physics.native_depth.shape
    for i,row in enumerate(rows):
        target=out/(row["id"]+".npz");record_path=target.with_suffix(".json")
        labels_path=maps/f"frame_{row['matching_studio_frame_index']:03d}.png"
        if target.exists() and record_path.exists():
            record=read(record_path)
            if record["source_labels_sha256"]!=sha(labels_path) or record["corrected_sha256"]!=sha(row["corrected_path"]):
                raise ValueError("Provisional calibration input changed")
        else:
            started=time.monotonic();labels=decode_labels(labels_path,(0,0,w,h));current=rgb(row["corrected_path"])
            arrays,record=physics.prepare(current,labels,row);save_npz(target,**arrays)
            record.update(fields_path=str(target),fields_sha256=sha(target),source_labels_path=str(labels_path),
                source_labels_sha256=sha(labels_path),total_seconds=time.monotonic()-started)
            write(record_path,record)
        manifest["frames"].append(record);write(out/"manifest.json",manifest)
        print(f"Provisional BM field {i+1}/{len(rows)}: {row['elapsed_h']:.3f}h",flush=True)
    manifest["complete"]=True;write(out/"manifest.json",manifest)
    return out/"manifest.json"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--backend",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--transfer-probe",action="store_true")
    p.add_argument("--profile-only",action="store_true")
    p.add_argument("--transfer-figures",action="store_true")
    p.add_argument("--provisional-base-fields",action="store_true")
    p.add_argument("--benchmark-parallel",action="store_true")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--require-completion",action="store_true")
    p.add_argument("--threads",type=int,default=4)
    args=p.parse_args();cv2.setNumThreads(args.threads)
    with threadpool_limits(limits=args.threads):
        if args.benchmark_parallel:benchmark_parallel(args.root,args.model_root,args.workers)
        elif args.provisional_base_fields:provisional_base_fields(args.root,args.model_root)
        elif args.transfer_figures:transfer_figures(args.root,args.require_completion)
        elif args.transfer_probe or args.profile_only:transfer_probe(args.root,args.model_root,args.backend,args.profile_only,args.require_completion)
        else:calibration_fields(args.root,args.model_root,args.backend)
