"""Create P1-final from saved categories; no training or model inference."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import importlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from darsia.fluidflower import phase_ffac_p1_final as rules
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import atomic_json,canonical_sha256,copy_file_verified,read_json,sha256_file,utc_now,JobLease
from .run_ffac_rig_camera_candidate import artifact

CONTRACT=rules.CONTRACT
SPEC_SHA256=rules.SPEC_SHA256
PLAN="p1_final_plan.json"
SEAL="rig_camera_candidate_receipt.json"
QA="production_qa/p1_final_qa.json"
RUN_DIRECTORY="p1_final_runs"
WARNING=("Studio review candidate: G3-T4c P1 final. Frozen P1 labels with floor fill removed "
         "and fixed 60 mm top-surface rule; no training or new inference. All 43 runs visible. "
         "Bottom strip remains blocked for mass. Regression and physical interpretation pending review.")


def code_binding():
    names=("darsia.fluidflower.tools.build_ffac_p1_final_candidate","darsia.fluidflower.phase_ffac_p1_final")
    return {name:sha256_file(importlib.import_module(name).__file__) for name in names}


def _write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data: raise ValueError("P1-final would overwrite a differing artifact")
    else:
        with path.open("xb") as stream: stream.write(data)
    return artifact(path)


def _packed(root,directory,run,index,mask,source):
    stream=io.BytesIO(); np.save(stream,np.packbits(mask),allow_pickle=False)
    return dict(_write(root/directory/run/f"frame_{index:03d}.npy",stream.getvalue()),
        frame_index=index,source_sha256=source["source_sha256"],shape_hw=list(mask.shape),
        storage="packbits-npy",contract=CONTRACT)


def _previous(root,run,index,value,source):
    stream=io.BytesIO(); np.save(stream,value,allow_pickle=False)
    return dict(_write(root/"top_surface_previous_labels"/run/f"frame_{index:03d}.npy",stream.getvalue()),
        frame_index=index,source_sha256=source["source_sha256"],shape_hw=list(value.shape),
        storage="npy",contract=CONTRACT)


def _overlay(root,run,index,pipeline,mask,color,name):
    rgba=np.zeros((*mask.shape,4),np.uint8); rgba[mask]=color
    stream=io.BytesIO(); Image.fromarray(rgba).save(stream,format="PNG",compress_level=1)
    return _write(root/"validation/run_packages"/run/"review_assets/native_phase_maps"/run/
        ("A_"+pipeline[:16])/name/f"frame_{index:03d}.png",stream.getvalue())


def _png(asset):
    with Image.open(bound_path(asset)) as image: return np.array(image)


def _mask(asset,source,shape):
    if (asset.get("storage")!="packbits-npy" or asset.get("source_sha256")!=source["source_sha256"]
        or asset.get("frame_index")!=source["frame_index"] or asset.get("shape_hw")!=list(shape)):
        raise ValueError("Floor flag lost its exact frame/source/canvas binding")
    packed=np.load(bound_path(asset),allow_pickle=False)
    if packed.dtype!=np.uint8 or packed.ndim!=1 or packed.size!=(np.prod(shape)+7)//8:
        raise ValueError("Floor flag packed representation is invalid")
    return np.unpackbits(packed,count=int(np.prod(shape))).reshape(shape).astype(bool)


def prepare_candidate(parent,output,spec,*,backend="auto",resume=False):
    from darsia.fluidflower.phase_ffac_t4c_review import t4c_saved_review
    from darsia.fluidflower.phase_validation import _model_identity
    from .register_track1_studio_candidate import registration_path
    parent,root,spec=map(lambda p:Path(p).resolve(),(parent,output,spec))
    if root.is_relative_to(parent) or parent.is_relative_to(root): raise ValueError("Use a detached new candidate")
    if sha256_file(spec)!=SPEC_SHA256: raise ValueError("P1-final specification changed")
    admitted=t4c_saved_review(parent)
    if admitted["placement"]!="P1": raise ValueError("P1-final must use the chosen exact P1")
    parent_seal=read_json(parent/SEAL); base_seal=read_json(bound_path(parent_seal["parent_seal"]))
    if (parent_seal.get("complete") is not True or base_seal.get("complete") is not True
        or (parent_seal["run_count"],parent_seal["frame_count"])!=(43,864)
        or (base_seal["run_count"],base_seal["frame_count"])!=(43,864)):
        raise ValueError("P1-final needs complete sealed P1 and BASE")
    base=Path(parent_seal["parent_seal"]["path"]).parent
    parents={f"{row['run']}/frame_{row['frame_index']:03d}":row["receipt"] for row in parent_seal["frames"]}
    bases={f"{row['run']}/frame_{row['frame_index']:03d}":row["receipt"] for row in base_seal["frames"]}
    if set(parents)!=set(bases) or len(parents)!=864: raise ValueError("Parent and BASE source sequences differ")
    actual=rules.select_backend(backend)
    inputs=dict(spec=artifact(spec),parent_seal=artifact(parent/SEAL),parent_registration=artifact(registration_path(parent.name)),
        base_seal=parent_seal["parent_seal"],code=code_binding(),rules=rules.RULE,
        training=False,inference=False,annotations_read=False,mass_eligible=False,bottom_mass_allowed=False,
        backend=actual)
    binding=dict(parent=str(parent),base=str(base),inputs=inputs,heads=parent_seal["heads"])
    if (root/PLAN).exists():
        plan=read_json(root/PLAN)
        if not resume or plan["binding"]!=binding: raise ValueError("Resume changes immutable P1-final binding")
        return plan
    if root.exists() or resume: raise ValueError("Use a new candidate folder or exact resume")
    root.mkdir(parents=True,exist_ok=False)
    # Copies of small model metadata preserve ordinary identity readers. No fit.
    for path in (parent/"models").rglob("*"):
        if path.is_file(): copy_file_verified(path,root/path.relative_to(parent))
    for name in ("regional_model.json","training_request.json","expected_review_runs.json",
                 "physical_scale_measurement.json","vertical_scale_check.json","t4c_floor_plan.json"):
        if (parent/name).is_file(): copy_file_verified(parent/name,root/name)
    if (parent/"regional_models").is_dir():
        for path in (parent/"regional_models").rglob("*"):
            if path.is_file(): copy_file_verified(path,root/path.relative_to(parent))
    summary=read_json(parent/"summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=utc_now(),
        display_name="FF_AC · G3-T4c P1 final · New lamps calibrated to overflow · Optical · No floor fill · Top-surface rule · 08.10.2026",
        inherited_base_id=parent.name,training_performed_in_this_revision=False,optical_training_performed=False,
        regional_training_performed=False,co2_training_performed=False,mass_eligible=False,scientific_acceptance=False,
        automatic_activation=False,bottom_mass_allowed=False,blind_hold_runs=[],floor_fill_removed=True,
        p1_final=True,top_surface_rule=rules.RULE)
    atomic_json(root/"summary.json",summary)
    sources=[]
    epochs={}
    # Avoid embedding the parent's 1.2 GB inference plan in this post-process.
    def inventory_run(run):
        native=read_json(parent/"source_alias/native_frames"/run/"manifest.json")
        values=[]
        for source in native["frame_sources"]:
            key=f"{run}/frame_{source['frame_index']:03d}"
            values.append(dict(run=run,frame_index=source["frame_index"],
                source_sha256=source["source_sha256"],elapsed_h=source["elapsed_h"],
                parent_receipt=parents[key],base_receipt=bases[key]))
        first=read_json(bound_path(values[0]["parent_receipt"]))
        return run,first["light_epoch_id"],values
    with ThreadPoolExecutor(max_workers=8) as executor:
        for run,epoch,values in executor.map(inventory_run,sorted({key.split('/')[0] for key in parents})):
            sources.extend(values);epochs[run]=epoch
    if {(s["run"],s["frame_index"]) for s in sources}!={(key.split('/')[0],int(key[-3:])) for key in parents}:
        raise ValueError("P1 source manifests differ from the sealed frame inventory")
    identity=list(_model_identity(root))
    plan=dict(contract=CONTRACT,created_utc=summary["created_utc"],binding=binding,source_inventory=sources,
        run_to_epoch=epochs,parent_frame_receipts=parents,base_frame_receipts=bases,
        model_identity=identity,job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding,model_identity=identity)),
        output_root=str(root),placement="P1-final",delta=parent_seal["delta"],
        inherited_t4c_plan=artifact(root/"t4c_floor_plan.json"),backend=actual)
    atomic_json(root/PLAN,plan)
    profile=read_json(parent/"phase_rig_camera.json")
    profile.pop("g3_t4c_floor_revision",None)
    profile["parent_model_identity"]=list(_model_identity(parent))
    profile["p1_final_revision"]=dict(contract=CONTRACT,plan=artifact(root/PLAN))
    atomic_json(root/"profile_template.json",profile)
    copy_file_verified(spec,root/"approved_p1_final_spec.md")
    return plan


def _load_frame(source):
    from .run_ffac_rig_camera_inference import load_array
    p1=read_json(bound_path(source["parent_receipt"])); base=read_json(bound_path(source["base_receipt"]))
    if p1["binding"]["source"]!=base["binding"]["source"]:
        raise ValueError("P1 / BASE scientific source or geometry binding differs")
    full=p1["binding"]["source"]
    if any(full[key]!=source[key] for key in ("run","frame_index","elapsed_h","source_sha256")):
        raise ValueError("Minimal source inventory differs from authoritative P1 source")
    labels,base_labels=_png(p1["labels_map"]),_png(base["labels_map"])
    return source,p1,base,labels,base_labels,_mask(p1["floor_contact_fill"],full,labels.shape),load_array(p1["rig_domain_revision"]["area_map"])


def _save_frame(root,plan,loaded,result):
    from .run_ffac_rig_camera_inference import write_maps
    source,p1,base,labels,base_labels,floor,area=loaded
    run,index=source["run"],source["frame_index"]; full=p1["binding"]["source"]
    maps,label_asset=write_maps(root,run,index,plan["pipeline_sha256"],result.labels,phase_shape_hw=result.labels.shape)
    if not np.array_equal(_png(label_asset),result.labels): raise ValueError("Saved P1-final labels changed")
    top=_packed(root,"top_surface_rule_masks",run,index,result.removed_top,full)
    previous=_previous(root,run,index,result.previous_top_labels,full)
    removed_top=_overlay(root,run,index,plan["pipeline_sha256"],result.removed_top,(255,157,66,255),"removed_top")
    removed_floor=_overlay(root,run,index,plan["pipeline_sha256"],floor,(84,156,242,255),"removed_floor")
    qa=dict(result.qa,run=run,frame_index=index,elapsed_h=full["elapsed_h"],source_sha256=full["source_sha256"])
    receipt=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),
        binding=dict(pipeline_sha256=plan["pipeline_sha256"],source=full,parent_receipt=source["parent_receipt"],base_receipt=source["base_receipt"]),
        run=run,frame_index=index,source_sha256=full["source_sha256"],elapsed_h=full["elapsed_h"],
        placement="P1-final",delta=plan["delta"],light_epoch_id=p1["light_epoch_id"],
        before_labels=p1["labels_map"],base_labels=base["labels_map"],labels_map=label_asset,maps=maps,
        floor_contact_fill=p1["floor_contact_fill"],floor_fill_removed=True,floor_rule_applied=False,
        top_surface_rule_masks=top,top_surface_previous_labels=previous,
        removed_display_layers=dict(removed_top=removed_top,removed_floor=removed_floor),
        rig_domain_revision=p1["rig_domain_revision"],full_extended_scientific=p1["full_extended_scientific"],
        baseline_extended_scientific=p1["baseline_extended_scientific"],qa=qa,
        inherited_gas_inference=p1["gas_inference"],backend=result.qa["backend"],
        labels_changed=False,annotations_read=False,training=False,training_performed=False,
        inference_performed=False,co2_inference_performed=False,new_gas_inference_performed=False,
        mass_eligible=False,bottom_mass_allowed=False)
    path=root/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    atomic_json(path,receipt)
    return dict(run=run,frame_index=index,receipt=artifact(path),qa=qa)


def assemble_run(root,plan,run,frames):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    parent=Path(plan["binding"]["parent"])
    native=read_json(parent/"source_alias/native_frames"/run/"manifest.json")
    inputs=read_json(bound_path(native[meta.SOURCE_KEY]["input_manifest"]))
    by_index={row["frame_index"]:row for row in frames}
    if len(by_index)!=len(native["frame_sources"]): raise ValueError("Incomplete P1-final source run")
    for record,row in zip(native["frame_sources"],inputs["frames"]):
        saved=read_json(bound_path(by_index[record["frame_index"]]["receipt"]))
        if saved["source_sha256"]!=record["source_sha256"] or row["source_sha256"]!=record["source_sha256"]:
            raise ValueError("P1-final source order changed")
        for value in (record,row):
            value["floor_fill_removed"]=True
            value["removed_floor_contact_fill"]=value.pop("floor_contact_fill",saved["floor_contact_fill"])
            value["top_surface_rule_masks"]=saved["top_surface_rule_masks"]
    input_path=root/"source_alias/source_bindings"/(run+".json"); atomic_json(input_path,inputs)
    native=meta.build_native_metadata(read_json(bound_path(inputs["parent_native_manifest"])),
        read_json(bound_path(inputs["camera_timeline_run"])),native["frame_sources"],inputs["frames"],inputs["reference"],
        input_manifest=artifact(input_path))
    native_path=root/"source_alias/native_frames"/run/"manifest.json"; atomic_json(native_path,native)
    result=dict(run=run,complete=True,native_manifest=artifact(native_path),input_manifest=artifact(input_path),
        frames=inputs["frames"],reference=inputs["reference"],frame_count=len(frames),
        receipts=[{k:v for k,v in row.items() if k!="qa"} for row in frames],pipeline_sha256=plan["pipeline_sha256"])
    atomic_json(root/RUN_DIRECTORY/(run+".json"),result)
    return result


def bind_p1_final_packages(root,plan,source_runs):
    """Create new review packages with final flags at their first write."""
    from darsia.fluidflower.phase_runtime import load_physics_profile,physics_runtime_identity
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from .phase_reference_readiness import defer_validation_references
    parent=Path(plan["binding"]["parent"])
    old_source=read_json(parent/"review_source/postflight_manifest.json")
    sources={row["run"]:row for row in source_runs}
    updated=[]
    for old in old_source["jobs"]:
        run=old["run"];job=deepcopy(old)
        job.update(review_output_dir=str(root/"source_alias"),analysis_source_review_root=str(root/"source_alias"),
            native_manifest_sha256=sources[run]["native_manifest"]["sha256"],frame_count=sources[run]["frame_count"])
        original=read_json(parent/"review_source"/run/"postflight_manifest.json")
        atomic_json(root/"review_source"/run/"postflight_manifest.json",dict(original,jobs=[job]))
        updated.append(job)
    atomic_json(root/"review_source/postflight_manifest.json",dict(old_source,jobs=updated))
    loaded=load_physics_profile(root);runtime=physics_runtime_identity(loaded);profile=meta.public_profile(loaded)
    packages=[]
    for old in read_json(parent/"validation/postflight_manifest.json")["jobs"]:
        run=old["run"];package=root/"validation/run_packages"/run;job=deepcopy(old)
        job.update(candidate_id=root.name+"__ffac__"+run+"__rigcamera_"+plan["pipeline_sha256"][:12],
            candidate_folder=str(root),review_output_dir=str(root/"source_alias"),
            analysis_source_review_root=str(root/"source_alias"),phase_map_output_dir=str(package/"review_assets"),
            params_sha256=plan["pipeline_sha256"],optical_model_sha256=plan["model_identity"][1],
            physics_runtime=runtime,output_mode="review",phase_model_only=True,raw_phase_map_output_dir=None,
            numeric_phase_output_dir=None,presence_diagnostics_dir=None,
            native_manifest_sha256=sources[run]["native_manifest"]["sha256"],presentation_status="ready",proposals_status="deferred")
        job["quality_flags"]=dict(job.get("quality_flags") or {},approved_g3_rig_camera=True,
            approved_g3_downstream=True,human_review_required=True,mass_analysis_allowed=False,
            scientific_acceptance=False,p1_final_revision=True,floor_contact_rule=False)
        for key in ("summary","mass_plot","contour_movie","per_frame_csv"):job[key]=None
        common=dict(schema=1,format="darsia-phase-validation-campaign-v1",campaign_id="phase-validation-"+root.name,
            model_id=plan["model_identity"][0],model_sha256=plan["model_identity"][1],model_root=str(root),
            source_manifest_sha256=sha256_file(root/"review_source"/run/"postflight_manifest.json"),
            pipeline_sha256=plan["pipeline_sha256"],physics_runtime=runtime,physics_profile=profile,
            output_mode="review",numeric_outputs=False,phase_display_policy=meta.REVISION_CONTRACT)
        atomic_json(package/"postflight_manifest.json",dict(common,jobs=[job],source_campaign_root=str(root/"review_source"/run)))
        atomic_json(package/"validation_binding.json",dict(common,combined_root=str(package),ffac_root=str(package),ffum_root=str(package),
            selected_candidates={run:job["candidate_id"]},run_count=1,completed_utc=plan["created_utc"]))
        defer_validation_references(package);packages.append(package)
    return packages


def seal_candidate(root,plan,frames):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from .build_ffac_g3_approved_candidate import assemble_saved_candidate
    if len(frames)!=864 or len({(f["run"],f["frame_index"]) for f in frames})!=864:
        raise ValueError("P1-final must have exactly43/864")
    source_runs=[assemble_run(root,plan,run,[row for row in frames if row["run"]==run])
                 for run in sorted(plan["run_to_epoch"])]
    meta.validate_campaign_bindings(source_runs)
    profile=read_json(root/"profile_template.json")
    runtime=read_json(bound_path(profile["rig_domain_revision"]["manifest"]))
    runtime["sources"]=[{k:v for k,v in row.items() if k!="receipts"} for row in source_runs]
    path=root/"rig_camera_revision/runtime_manifest.json"; atomic_json(path,runtime)
    profile["rig_domain_revision"]["manifest"]=artifact(path); atomic_json(root/"phase_rig_camera.json",profile)
    packages=bind_p1_final_packages(root,plan,source_runs)
    assemble_saved_candidate(root,packages,plan)
    rows=[row["qa"] for row in frames]
    if not all(row["all_pass"] for row in rows): raise ValueError("P1-final QA failed")
    qa=dict(contract=CONTRACT,technical_all_pass=True,run_count=43,frame_count=864,
        rules=rules.RULE,backend=plan["backend"],rows=rows,
        cost=dict(cpu_seconds=sum(r["cpu_seconds"] for r in rows),gpu_seconds=sum(r["gpu_seconds"] for r in rows)),
        source_time_geometry_area_heads_delta_unchanged=True,annotations_read=False,training=False,inference=False)
    atomic_json(root/QA,qa)
    table=root/"production_qa/p1_final_removals.csv"
    table.parent.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader();writer.writerows(rows)
    _write(table,stream.getvalue().encode("utf-8"))
    seal=dict(contract=CONTRACT,complete=True,model_identity=plan["model_identity"],pipeline_sha256=plan["pipeline_sha256"],
        source_plan_sha256=sha256_file(root/PLAN),run_count=43,frame_count=864,
        frames=[{k:v for k,v in row.items() if k!="qa"} for row in frames],parent_seal=plan["binding"]["inputs"]["parent_seal"],
        heads=plan["binding"]["heads"],delta=plan["delta"],placement="P1-final",qa=artifact(root/QA),
        physical_scale_approved=True,source_time_geometry_area_heads_delta_unchanged=True,
        labels_changed=False,mass_eligible=False,bottom_mass_allowed=False,blind_hold_runs=[],
        training_performed=False,inference_performed=False,automatic_activation=False,scientific_acceptance=False)
    atomic_json(root/SEAL,seal)
    from .register_ffac_p1_final_studio_candidate import register
    registration=register(root)
    return dict(seal=artifact(root/SEAL),registration=artifact(registration),qa=artifact(root/QA))


def build(root,plan,*,workers=8,resume=False,indices=None):
    wall,cpu=time.perf_counter(),time.process_time(); root=Path(root)
    completed=[]; outstanding=[];last_reported=0
    sources=[s for s in plan["source_inventory"] if indices is None or (s["run"],s["frame_index"]) in indices]
    with ThreadPoolExecutor(max_workers=workers) as readers,ThreadPoolExecutor(max_workers=workers) as writers:
        pending=[]; iterator=iter(sources)
        for _ in range(workers):
            source=next(iterator,None)
            if source is not None: pending.append((source,readers.submit(_load_frame,source)))
        while pending:
            source,future=pending.pop(0); loaded=future.result()
            path=root/"rig_camera_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
            if path.exists():
                saved=read_json(path)
                if (not resume or saved.get("contract")!=CONTRACT or saved["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]
                    or saved["binding"]["source"]!=loaded[1]["binding"]["source"] or not saved["qa"]["all_pass"]):
                    raise ValueError("Resume receipt binding differs")
                for asset in (saved["labels_map"],saved["top_surface_rule_masks"],*saved["maps"].values()): bound_path(asset)
                completed.append(dict(run=source["run"],frame_index=source["frame_index"],receipt=artifact(path),qa=saved["qa"]))
            else:
                if loaded[1]["light_epoch_id"]!=plan["run_to_epoch"][source["run"]]:
                    raise ValueError("P1 light epoch changed within source inventory")
                result=rules.apply_p1_final(*loaded[3:],backend=plan["backend"])
                if indices is not None:
                    cpu_result=rules.apply_p1_final(*loaded[3:],backend="cpu")
                    if not all(np.array_equal(getattr(result,key),getattr(cpu_result,key)) for key in
                        ("labels","after_floor_removal","removed_top","previous_top_labels")):
                        raise ValueError("CUDA/CPU probe outputs differ")
                    result.qa["probe_cuda_cpu_identical"]=True
                outstanding.append(writers.submit(_save_frame,root,plan,loaded,result))
                if len(outstanding)>=workers:
                    completed.append(outstanding.pop(0).result())
            source=next(iterator,None)
            if source is not None: pending.append((source,readers.submit(_load_frame,source)))
            if len(completed)>=last_reported+10:
                last_reported=len(completed)
                atomic_json(root/"progress"/f"snapshot_{time.time_ns()}.json",dict(stage="fixed_postprocessing",done=len(completed),total=len(sources),backend=plan["backend"],
                    elapsed_seconds=time.perf_counter()-wall))
                print(json.dumps(dict(stage="fixed_postprocessing",done=len(completed),total=len(sources))),flush=True)
        completed.extend(f.result() for f in outstanding)
    completed.sort(key=lambda row:(row["run"],row["frame_index"]))
    if indices is not None: return completed
    result=seal_candidate(root,plan,completed)
    atomic_json(root/"completion.json",dict(contract=CONTRACT,complete=True,**result,
        wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,backend=plan["backend"]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","output","spec"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8);parser.add_argument("--backend",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    if not 1<=args.workers<=32: raise ValueError("Workers must be1–32")
    with JobLease(args.output.parent/(args.output.name+"-operation"),SPEC_SHA256):
        plan=prepare_candidate(args.parent,args.output,args.spec,backend=args.backend,resume=args.resume)
        indices={(run,index) for run,index in (("ac17",5),("ac17",6),("ac17",20),("ac19",13),("ac28",13),("ac40",13),("ac51",19))} if args.probe else None
        result=build(args.output,plan,workers=args.workers,resume=args.resume,indices=indices)
        print(json.dumps(dict(complete=not args.probe,result=result),default=str),flush=True)


if __name__=="__main__": main()
