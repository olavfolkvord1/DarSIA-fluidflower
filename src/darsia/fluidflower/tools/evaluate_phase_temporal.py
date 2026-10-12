"""Paired temporal-support evaluation with frozen optical weights and evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
from darsia.fluidflower.phase_physics import load_physics_profile, load_run_context
from darsia.fluidflower.phase_temporal import AqueousFrontSettings, FRONT_VERSION, phase_physics_sequence
from darsia.fluidflower import phase_temporal
from darsia.fluidflower.phase_training import trainer
from darsia.fluidflower.phase_training.review_regions import annotation_mask
from darsia.fluidflower.phase_validation import _source_jobs, _atomic_json, _model_identity, predict_image_tiled
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.phase_inference_benchmark_backends import CudaFeaturePredictor, partition


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def review_regions(parent, jobs):
    request = read(parent / "training_request.json")
    sources, regions, _ = trainer._load_sources_and_regions([
        trainer.ReviewBundle(Path(b["decisions"]["snapshot_path"]), Path(b["combined_manifest"]["snapshot_path"]))
        for b in request["bundles"]])
    old = {s.run: read(s.asset_manifest_path) for s in sources}
    new = {j["run"]: read(Path(j["analysis_source_review_root"]) / "native_frames" / j["run"] / "manifest.json") for j in jobs}
    selected = defaultdict(list)
    for region in regions:
        if region.run in new:
            aligned = trainer._align_region_timeline(region, old[region.run], new[region.run])
            selected[(region.run, aligned.frame_index)].append(aligned)
    return selected


def figure(path, canvas, box, old, new, run, elapsed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    x,y,w,h=box
    photo=Image.fromarray(canvas[y:y+h,x:x+w]).resize((1100,600))
    panels=[]
    for prediction in (old,new):
        co2=prediction["co2_presence_probability"]>=.5
        gas=co2&(prediction["gas_probability_conditional_on_co2"]>=.6)
        rgba=np.zeros((*co2.shape,4),np.uint8)
        rgba[co2]=[49,217,160,200]
        rgba[gas]=[242,76,157,230]
        overlay=Image.fromarray(rgba).resize(photo.size,resample=Image.Resampling.NEAREST)
        panels.append(Image.alpha_composite(photo.convert("RGBA"),overlay))
    fig,axes=plt.subplots(1,2,figsize=(16,5.5))
    for ax,image,title in zip(axes,panels,("V8 support", "Same weights + aqueous-front memory")):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{run.upper()}, {elapsed:g} h | green: aqueous CO2; pink: gas | not mass")
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def evaluate(parent, out, selected_runs=None):
    import cupy as cp
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/"code_snapshot"
    snapshot.mkdir()
    code_files=[Path(__file__),Path(phase_temporal.__file__)]
    code_hashes={path.name:sha256_file(path) for path in code_files}
    for path in code_files:
        shutil.copy2(path,snapshot/path.name)
    source=Path(read(parent / "validation/validation_binding.json")["combined_root"])
    jobs=_source_jobs(source)
    if selected_runs:
        jobs=[job for job in jobs if job["run"] in selected_runs]
        if {job["run"] for job in jobs}!=set(selected_runs):
            raise ValueError("Unknown requested review run")
    regions=review_regions(parent,jobs)
    old_profile=load_physics_profile(parent)
    new_profile={**old_profile,"aqueous_front_memory":{"format":FRONT_VERSION,**asdict(AqueousFrontSettings())}}
    model=HierarchicalPhaseModel.load(parent)
    protected={str(p):sha256_file(p) for p in [parent/"summary.json",parent/"phase_physics.json",*sorted((parent/"models").rglob("*.json"))]}
    report={"parent":str(parent),"parent_identity":_model_identity(parent)[1],"created_utc":utc_now(),
        "new_front_settings":new_profile["aqueous_front_memory"],"scope":"paired training-evidence check; not independent validation",
        "runs":{},"regions":[],"protected_sources":protected,"weights_retrained":False}
    report["selected_runs"]=[job["run"] for job in jobs]
    report["code_sha256"]=code_hashes
    for job in jobs:
        run,dataset=job["run"],job["dataset"]
        root=Path(job["analysis_source_review_root"])
        native=read(root/"native_frames"/run/"manifest.json")
        x,y,w,h=native["content_box_xywh"]
        context,_=load_run_context(parent,old_profile,dataset,run,
            native_manifest_sha256=old_profile["runs"][f"{dataset}/{run}"]["native_manifest_sha256"])
        independent = list(job.get("independent_reference_frames") or [])
        expected_sha=old_profile["runs"][f"{dataset}/{run}"]["native_manifest_sha256"]
        if sha256_file(root/"native_frames"/run/"manifest.json")!=expected_sha:
            revision=native["timeline_revision"]
            assert revision["source_manifest_sha256"]==expected_sha
            assert revision["kind"]=="prepend-evaluated-baseline-v1"
            independent=[0]
        report.setdefault("independent_reference_frames",{})[run]=independent
        assert context.domain.shape==(h,w)
        baseline=np.asarray(Image.open(root/"native_references"/run/"pre_injection.jpg").convert("RGB"))[y:y+h,x:x+w]
        gpu=CudaFeaturePredictor(model,baseline)
        old_seq=phase_physics_sequence(context,old_profile)
        new_seq=phase_physics_sequence(context,new_profile)
        previous=None
        rows=[]
        report["runs"][run]=rows
        for index,item in enumerate(native["frame_sources"]):
            start=time.perf_counter()
            canvas=np.asarray(Image.open(root/"native_frames"/run/f"frame_{index:03d}.jpg").convert("RGB"))
            current=canvas[y:y+h,x:x+w]
            heads=gpu.predict(current,previous,current_key=index,previous_key=index-1)
            if index==10:
                reference=predict_image_tiled(model,current,baseline,previous_rgb=previous)
                parity={}
                for name,values in heads.items():
                    error=np.abs(reference[name]-values)
                    threshold=.6 if name.startswith("gas_") else .5
                    disagreement=float(np.mean((values>=threshold)!=(reference[name]>=threshold)))
                    parity[name]={"q99_abs_error":float(np.quantile(error,.99)),"disagreement":disagreement}
                    assert parity[name]["q99_abs_error"]<=.001 and disagreement<=.0001, parity
                report.setdefault("cpu_cuda_parity",{})[run]=parity
                del reference
            raw=partition(np.stack([heads["co2_presence_probability"],heads["gas_probability_conditional_on_co2"]],axis=-1))
            elapsed=float(item["elapsed_h"])
            old,old_info=old_seq.apply(raw,elapsed)
            new,new_info=new_seq.apply(raw,elapsed)
            assert np.array_equal(old["co2_presence_probability"],new["co2_presence_probability"])
            c=old["co2_presence_probability"]>=.5
            old_gas=c&(old["gas_probability_conditional_on_co2"]>=.6)
            new_gas=c&(new["gas_probability_conditional_on_co2"]>=.6)
            old_path=Path(job["phase_map_output_dir"])/"native_phase_maps"/run/f"A_{job['params_sha256'][:16]}"/f"frame_{index:03d}.png"
            rgba=np.asarray(Image.open(old_path)) [y:y+h,x:x+w]
            published=(rgba[...,3]>0)&(rgba[...,0]==242)
            old_disagreement=float(np.mean(old_gas!=published))
            assert old_disagreement<=.0001,(run,index,old_disagreement)
            row={"frame":index,"elapsed_h":elapsed,"old_gas_pixels":int(old_gas.sum()),"new_gas_pixels":int(new_gas.sum()),
                "gas_added_pixels":int((new_gas&~old_gas).sum()),"old_published_disagreement":old_disagreement,"physics":new_info}
            assert row["gas_added_pixels"]==0
            rows.append(row)
            for region in regions[(run,index)]:
                mask=annotation_mask(region.payload,canvas_wh=native["canvas_wh"],content_box=native["content_box_xywh"])
                if not mask.any():
                    continue
                report["regions"].append({"region_id":region.region_id,"run":run,"frame":index,"elapsed_h":elapsed,
                    "class":region.phase_class,"old_gas_fraction":float(old_gas[mask].mean()),
                    "new_gas_fraction":float(new_gas[mask].mean()),"co2_fraction":float(c[mask].mean())})
            if index in (10,len(native["frame_sources"])-1):
                figure(out/f"{run}_frame_{index:03d}.png",canvas,(x,y,w,h),old,new,run,elapsed)
            if index not in independent:
                previous=current
            row["seconds"]=time.perf_counter()-start
            print(f"{run} {index+1}/{native['frame_count']} t={elapsed:g}h gas={row['old_gas_pixels']} -> {row['new_gas_pixels']} ({row['seconds']:.1f}s)",flush=True)
            _atomic_json(out/"report.partial.json",report)
            del raw,old,new,heads,rgba,old_gas,new_gas,published
        del gpu,old_seq,new_seq,context,baseline,previous
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    groups=defaultdict(list)
    for row in report["regions"]:
        groups[(row["run"],row["class"])].append(row)
    report["region_summary"]={f"{run}/{label}":{"regions":len(rows),
        "old_gas_fraction":float(np.mean([r["old_gas_fraction"] for r in rows])),
        "new_gas_fraction":float(np.mean([r["new_gas_fraction"] for r in rows])),
        "regions_losing_all_detected_gas":sum(r["old_gas_fraction"]>0 and r["new_gas_fraction"]==0 for r in rows)}
        for (run,label),rows in groups.items()}
    for path,digest in protected.items():
        assert sha256_file(Path(path))==digest,path
    for path in code_files:
        assert sha256_file(path)==code_hashes[path.name],path
    report["completed_utc"]=utc_now()
    _atomic_json(out/"report.json",report)
    print(json.dumps(report["region_summary"],indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runs",nargs="*")
    args=parser.parse_args()
    cv2.setNumThreads(4)
    with threadpool_limits(limits=4):
        evaluate(args.parent.resolve(),args.output.resolve(),args.runs)
