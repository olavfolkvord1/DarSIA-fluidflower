"""Build a separate Rule B Studio candidate from saved top-surface G3 maps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
import io
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_g3_production import CO2_SHA256, G3_SHA256, REFIT_SHA256
from darsia.fluidflower.phase_gas_barrier_runtime import (
    VERSION, SPEC_SHA256, OPERATION, bound_path, read_components, read_parent_decision,
)
from darsia.fluidflower.phase_validation import _model_identity
from .build_ffac_g3_approved_candidate import CONFIGURATION, _write_bytes, assemble_saved_candidate, mask_sha256, write_phase_layers
from .build_ffac_g3_top_surface_candidate import _copy_scientific_lineage
from .refit_ffac_g3_downstream import (
    JobLease, ProgressLog, atomic_json, below_normal_priority, canonical_sha256,
    copy_file_verified, preflight_space, read_json, sha256_file, utc_now, worker_initializer,
)

CONTRACT = "ffac-gas-barrier-rule-b-candidate-v1"
QA_CONTRACT = "ffac-gas-barrier-rule-b-qa-v1"
GEOMETRY_CONTRACT = "ffac-gas-barrier-rule-b-geometry-v1"


def _artifact(path):
    return dict(path=str(path), sha256=sha256_file(path))


def _stage_row(root, parent, job, geometry, top):
    run, index = geometry["run"], int(geometry["frame_index"])
    native_path = Path(job["review_output_dir"]) / "native_frames" / run / "manifest.json"
    native = read_json(native_path)
    native_sha = sha256_file(native_path)
    if (Path(geometry["native_manifest_path"]).resolve() != native_path.resolve()
            or geometry["native_manifest_sha256"] != native_sha
            or float(geometry["elapsed_h"]) != float(native["frame_sources"][index]["elapsed_h"])
            or list(geometry["content_box_xywh"]) != list(native["content_box_xywh"])):
        raise ValueError("Rule B geometry differs from its exact inherited source frame")
    component_path = root / "gas_barrier_rule_b/components" / run / f"frame_{index:03d}.npz"
    registration_path = root / "gas_barrier_rule_b/registrations" / run / f"frame_{index:03d}.json"
    copy_file_verified(bound_path(geometry["components"]), component_path)
    copy_file_verified(bound_path(geometry["registration_receipt"]), registration_path)
    parent_frame_path = parent / "top_surface_frames" / run / f"frame_{index:03d}.json"
    parent_frame = read_json(parent_frame_path)
    qa = parent_frame["qa"]
    source_ids = geometry["source_component_ids"]
    if (not source_ids or any(type(value) is not int or value <= 0 for value in source_ids)
            or source_ids != sorted(set(source_ids))):
        raise ValueError("Rule B geometry has no exact source-component IDs")
    return dict(geometry, components=_artifact(component_path), registration_receipt=_artifact(registration_path),
        parent_labels_map=_artifact(Path(qa["map_path"])), parent_phase_map=_artifact(Path(qa["studio_map_path"])),
        parent_frame_receipt=_artifact(parent_frame_path), pixel_area_mm2=float(qa["pixel_area_mm2"]),
        top_surface_mask=top["mask"], revised_photo_mask=top["revised_photo_mask"],
        canvas_wh=native["canvas_wh"], content_box_xywh=native["content_box_xywh"])


def prepare_candidate(parent, output, geometry_manifest, *, resume=False, workers=8):
    from darsia.fluidflower.tools.register_track1_studio_candidate import registration_path, registration_warning
    from darsia.fluidflower.phase_runtime import load_physics_profile
    parent, root, geometry_path = (Path(value).resolve() for value in (parent,output,geometry_manifest))
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Rule B requires a new detached candidate")
    if not registration_warning(parent).startswith("Studio review candidate: approved G3"):
        raise ValueError("Rule B requires an intact approved top-surface parent")
    parent_identity = list(_model_identity(parent))
    geometry = read_json(geometry_path)
    if geometry.get("contract") != GEOMETRY_CONTRACT or geometry.get("spec_sha256") != SPEC_SHA256:
        raise ValueError("Rule B geometry binds another specification")
    original = read_json(parent / "validation/postflight_manifest.json")
    jobs = {job["run"]:job for job in original["jobs"]}
    rows = {(row["run"],int(row["frame_index"])):row for row in geometry["frames"]}
    tops = {(row["run"],row["frame_index"]):row for row in read_json(parent / "photo_sand_top_revision/manifest.json")["frames"]}
    if (len(jobs) != 43 or len(rows) != 864 or len(geometry["frames"]) != 864
            or set(rows) != set(tops) or {run for run,_ in rows} != set(jobs)):
        raise ValueError("Rule B geometry must cover the exact43/864 inherited frames")
    registration = registration_path(parent_identity[0])
    binding = dict(contract=CONTRACT,parent=str(parent),candidate=str(root),parent_model_identity=parent_identity,
        parent_registration=_artifact(registration),geometry_manifest=_artifact(geometry_path),
        spec_sha256=SPEC_SHA256,configuration=CONFIGURATION,g3_sha256=G3_SHA256,
        regional_refit_sha256=REFIT_SHA256,co2_sha256=CO2_SHA256,code_sha256=sha256_file(__file__),
        cpu_only=True,training=False,model_inference=False,component_filtering=False,
        photo_pixels_changed=False,mass_eligible=False,scientific_acceptance=False)
    plan_path = root / "barrier_rule_b_candidate_plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if not resume or plan["binding"] != binding or plan["model_identity"] != list(_model_identity(root)):
            raise ValueError("Rule B resume differs from its immutable plan")
        return plan
    preparation_path = root / "barrier_preparation_binding.json"
    if root.exists():
        if not resume or not preparation_path.is_file() or read_json(preparation_path)["binding"] != binding:
            raise ValueError("Use a new absent candidate, or its exact preparation resume")
        preparation = read_json(preparation_path)
    else:
        if resume:
            raise ValueError("Rule B cannot resume an absent preparation")
        preflight_space(root, 10 * 1024**3)
        root.mkdir(parents=True,exist_ok=False)
        preparation = dict(binding=binding,created_utc=utc_now())
        atomic_json(preparation_path,preparation)
    for name,digest in (("models/g3/gas_head.json",G3_SHA256),("regional_model.json",REFIT_SHA256),
                        ("models/generic/co2_presence_signal.json",CO2_SHA256)):
        if sha256_file(parent/name) != digest:
            raise ValueError("Rule B cannot change approved model heads")
    _copy_scientific_lineage(parent,root)
    # Only small immutable metadata is copied. Native photographs and907 photo
    # masks continue to point to the permanent top-surface parent's source alias.
    for path in [parent/"review_source/postflight_manifest.json",
                 *(parent/"review_source"/run/"postflight_manifest.json" for run in jobs),
                 parent/"photo_sand_top_revision/manifest.json"]:
        copy_file_verified(path,root/path.relative_to(parent))
    copied_geometry = root/"gas_barrier_rule_b/source_geometry_manifest.json"
    derivation_path = root/"gas_barrier_rule_b/barrier_derivation.json"
    reference_path = root/"gas_barrier_rule_b/reference_components.npz"
    copy_file_verified(geometry_path,copied_geometry)
    copy_file_verified(bound_path(geometry["barrier_derivation"]),derivation_path)
    copy_file_verified(bound_path(geometry["reference_components"]),reference_path)
    log=ProgressLog(root)
    staged=[]
    started=time.monotonic()
    log.event("barrier_source_staging_started",total_frames=864,workers=min(8,workers))
    with ThreadPoolExecutor(max_workers=min(8,workers)) as pool:
        futures=[pool.submit(_stage_row,root,parent,jobs[row["run"]],row,tops[key]) for key,row in sorted(rows.items())]
        for future in as_completed(futures):
            row=future.result()
            row["barrier_derivation_sha256"]=geometry["barrier_derivation"]["sha256"]
            staged.append(row)
            if len(staged)%32==0 or len(staged)==864:
                elapsed=time.monotonic()-started
                log.event("barrier_source_staging",completed_frames=len(staged),total_frames=864,
                          elapsed_seconds=elapsed,eta_seconds=elapsed*(864-len(staged))/len(staged))
    staged.sort(key=lambda row:(row["run"],row["frame_index"]))
    runtime_manifest=root/"gas_barrier_rule_b/manifest.json"
    atomic_json(runtime_manifest,dict(contract=VERSION,spec_sha256=SPEC_SHA256,parent_model_identity=parent_identity,
        barrier_derivation_sha256=geometry["barrier_derivation"]["sha256"],source_geometry_manifest_sha256=sha256_file(copied_geometry),
        frames=staged,operation=OPERATION))
    loaded=load_physics_profile(parent)
    profile={key:deepcopy(value) for key,value in loaded.items() if not key.startswith("_")}
    if "photo_sand_top_revision" not in profile or "gas_barrier_rule_b" in profile:
        raise ValueError("Rule B requires exactly one inherited top-surface decision")
    profile["gas_barrier_rule_b"]=dict(format=VERSION,spec_sha256=SPEC_SHA256,operation=OPERATION,
        manifest=dict(path="gas_barrier_rule_b/manifest.json",sha256=sha256_file(runtime_manifest)),
        barrier_derivation=dict(path="gas_barrier_rule_b/barrier_derivation.json",sha256=sha256_file(derivation_path)))
    atomic_json(root/"phase_optical_regions.json",profile)
    summary=read_json(parent/"summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=preparation["created_utc"],
        display_name="FF_AC · G3 gas · Top surface corrected · Barrier rule B",
        inherited_training_candidate_id=parent.name,training_performed_in_this_revision=False,
        raw_inference_performed=False,scientific_acceptance=False,mass_eligible=False)
    summary["approved_g3_production"]["barrier_rule"]=dict(format=VERSION,spec_sha256=SPEC_SHA256,
        operation=OPERATION,parent_model_identity=parent_identity,barrier_derivation_sha256=sha256_file(derivation_path))
    atomic_json(root/"summary.json",summary)
    identity=list(_model_identity(root))
    pipeline=canonical_sha256(dict(binding=binding,model_identity=identity,profile_sha256=sha256_file(root/"phase_optical_regions.json")))
    plan=dict(binding=binding,created_utc=preparation["created_utc"],model_identity=identity,pipeline_sha256=pipeline,
        job_sha256=canonical_sha256(binding),source_inventory=staged,component_manifest_sha256=sha256_file(runtime_manifest),
        barrier_derivation_sha256=sha256_file(derivation_path))
    atomic_json(plan_path,plan)
    return plan


def apply_frame(candidate, row, pipeline):
    from darsia.fluidflower.phase_gas_barrier_rule import apply_barrier_rule_b
    root=Path(candidate)
    run,index=row["run"],row["frame_index"]
    receipt_path=root/"barrier_rule_b_frames"/run/f"frame_{index:03d}.json"
    binding=dict(run=run,frame_index=index,pipeline_sha256=pipeline,components=row["components"],
        parent_labels_map=row["parent_labels_map"],parent_phase_map=row["parent_phase_map"],
        native_manifest_sha256=row["native_manifest_sha256"],barrier_derivation_sha256=row["barrier_derivation_sha256"])
    if receipt_path.exists():
        receipt=read_json(receipt_path)
        if receipt.get("complete") is not True or receipt["binding"]!=binding:
            raise ValueError("Rule B frame resume differs from its completed binding")
        for asset in [*receipt["maps"].values(),receipt["labels_map"],receipt["labels_archive"],
                      row["components"],row["parent_labels_map"],row["parent_phase_map"],row["parent_frame_receipt"]]:
            bound_path(asset)
        return dict(run=run,frame_index=index,receipt_path=str(receipt_path),receipt_sha256=sha256_file(receipt_path))
    bound_path(row["parent_frame_receipt"])
    before,rgba=read_parent_decision(row["parent_labels_map"],row["parent_phase_map"],row["content_box_xywh"])
    arrays=read_components(row["components"],before.shape)
    after,audit=apply_barrier_rule_b(before,arrays["open_components"],row["source_component_ids"],
        pixel_area_mm2=row["pixel_area_mm2"],barrier_mask=arrays["barrier_mask"])
    maps=write_phase_layers(root/"validation/run_packages"/run/"review_assets",run,index,pipeline,rgba,after,row["content_box_xywh"])
    encoded=io.BytesIO()
    Image.fromarray(after,"L").save(encoded,format="PNG",compress_level=1)
    proof=_write_bytes(root/"production_labels"/run/f"frame_{index:03d}.png",encoded.getvalue())
    encoded=io.BytesIO()
    np.savez_compressed(encoded,labels=after)
    archive=_write_bytes(root/"production_labels"/run/f"frame_{index:03d}.npz",encoded.getvalue())
    saved,child_rgba=read_parent_decision(proof,maps["combined"],row["content_box_xywh"])
    union=np.isin(before,(1,2))
    after_union=np.isin(saved,(1,2))
    changed=before!=saved
    if (not np.array_equal(saved,after) or not np.array_equal(union,after_union)
            or np.any(changed & ~((before==2)&(saved==1)))):
        raise ValueError("Rule B saved decision changed CO2 union or a non-gas label")
    x,y,width,height=map(int,row["content_box_xywh"])
    old_local=rgba[y:y+height,x:x+width]
    new_local=child_rgba[y:y+height,x:x+width]
    if not np.array_equal(old_local[~changed],new_local[~changed]):
        raise ValueError("Rule B saved Studio map changed unchanged phase pixels")
    before_gas,after_gas=int(np.count_nonzero(before==2)),int(np.count_nonzero(saved==2))
    area=float(row["pixel_area_mm2"])
    qa=dict(run=run,frame_index=index,elapsed_h=float(row["elapsed_h"]),pixel_area_mm2=area,
        before_gas_pixels=before_gas,after_gas_pixels=after_gas,removed_gas_pixels=before_gas-after_gas,
        before_gas_cm2=before_gas*area/100,after_gas_cm2=after_gas*area/100,removed_gas_cm2=(before_gas-after_gas)*area/100,
        before_co2_pixels=int(union.sum()),after_co2_pixels=int(after_union.sum()),
        co2_union_exact=True,only_gas_to_aqueous=True,unchanged_rgba_exact=True,
        co2_sha256_before=mask_sha256(union),co2_sha256_after=mask_sha256(after_union),
        breached_components=audit["breached_components"],bridging_co2_components=audit["bridging_co2_components"],
        source_component_ids=row["source_component_ids"],components_path=row["components"]["path"],components_sha256=row["components"]["sha256"],
        barrier_derivation_sha256=row["barrier_derivation_sha256"],native_manifest_path=row["native_manifest_path"],native_manifest_sha256=row["native_manifest_sha256"],
        parent_labels_map_path=row["parent_labels_map"]["path"],parent_labels_map_sha256=row["parent_labels_map"]["sha256"],
        map_path=proof["path"],map_sha256=proof["sha256"],studio_map_path=maps["combined"]["path"],studio_map_sha256=maps["combined"]["sha256"],
        parent_map_path=row["parent_phase_map"]["path"],parent_map_sha256=row["parent_phase_map"]["sha256"],
        content_box_xywh=row["content_box_xywh"],canvas_wh=row["canvas_wh"])
    receipt=dict(contract=CONTRACT,complete=True,binding=binding,maps=maps,labels_map=proof,labels_archive=archive,
        qa=qa,rule_audit=audit,training=False,model_inference=False,component_filtering=False,photo_pixels_changed=False)
    atomic_json(receipt_path,receipt)
    return dict(run=run,frame_index=index,receipt_path=str(receipt_path),receipt_sha256=sha256_file(receipt_path))


def bind_packages(root,plan):
    from darsia.fluidflower.phase_runtime import load_physics_profile,physics_runtime_identity
    from darsia.fluidflower.tools.phase_reference_readiness import defer_validation_references
    loaded=load_physics_profile(root)
    runtime=physics_runtime_identity(loaded)
    public={key:deepcopy(value) for key,value in loaded.items() if not key.startswith("_")}
    parent=Path(plan["binding"]["parent"])
    packages=[]
    for old in read_json(parent/"validation/postflight_manifest.json")["jobs"]:
        run=old["run"]
        package=root/"validation/run_packages"/run
        job=deepcopy(old)
        job.update(candidate_id=root.name+"__ffac__"+run+"__barrier_b_"+plan["pipeline_sha256"][:12],candidate_folder=str(root),
            phase_map_output_dir=str(package/"review_assets"),params_sha256=plan["pipeline_sha256"],
            optical_model_sha256=plan["model_identity"][1],physics_runtime=runtime,phase_model_only=True,output_mode="review")
        job["quality_flags"]=dict(job.get("quality_flags") or {},approved_g3_barrier_rule_b=True,
                                 approved_g3_downstream=True,mass_analysis_allowed=False,scientific_acceptance=False)
        common=dict(schema=1,format="darsia-phase-validation-campaign-v1",campaign_id="phase-validation-"+root.name,
            model_id=plan["model_identity"][0],model_sha256=plan["model_identity"][1],model_root=str(root),
            source_manifest_sha256=sha256_file(root/"review_source"/run/"postflight_manifest.json"),
            pipeline_sha256=plan["pipeline_sha256"],physics_runtime=runtime,physics_profile=public,
            output_mode="review",numeric_outputs=False,phase_display_policy="approved-g3-barrier-rule-b-v1")
        atomic_json(package/"postflight_manifest.json",dict(common,jobs=[job],source_campaign_root=str(root/"review_source"/run)))
        atomic_json(package/"validation_binding.json",dict(common,combined_root=str(package),ffac_root=str(package),ffum_root=str(package),
            selected_candidates={run:job["candidate_id"]},run_count=1,completed_utc=plan["created_utc"]))
        defer_validation_references(package)
        packages.append(package)
    return packages


def seal_candidate(root,plan,completed):
    expected={(row["run"],row["frame_index"]) for row in plan["source_inventory"]}
    if len(completed)!=864 or len(expected)!=864 or {(row["run"],row["frame_index"]) for row in completed}!=expected:
        raise ValueError("Rule B cannot seal partial or overlapping frame coverage")
    receipts=[]
    for item in sorted(completed,key=lambda row:(row["run"],row["frame_index"])):
        if sha256_file(item["receipt_path"])!=item["receipt_sha256"]:
            raise ValueError("Rule B frame receipt changed")
        receipt=read_json(item["receipt_path"])
        if receipt.get("complete") is not True or receipt["binding"]["pipeline_sha256"]!=plan["pipeline_sha256"]:
            raise ValueError("Rule B frame lacks an exact completed pipeline binding")
        for asset in [*receipt["maps"].values(),receipt["labels_map"],receipt["labels_archive"],
                      receipt["binding"]["components"],receipt["binding"]["parent_labels_map"],receipt["binding"]["parent_phase_map"]]:
            bound_path(asset)
        receipts.append(receipt)
    header=dict(configuration=CONFIGURATION,spec_sha256=SPEC_SHA256,g3_sha256=G3_SHA256,
        regional_refit_sha256=REFIT_SHA256,co2_sha256=CO2_SHA256,model_identity=plan["model_identity"],
        parent_model_identity=plan["binding"]["parent_model_identity"],component_manifest_sha256=plan["component_manifest_sha256"],
        barrier_derivation_sha256=plan["barrier_derivation_sha256"],run_count=43,frame_count=864)
    qa_path=root/"production_qa/barrier_rule_b_qa.json"
    atomic_json(qa_path,dict(contract=QA_CONTRACT,**header,all_pass=True,rows=[receipt["qa"] for receipt in receipts]))
    assemble_saved_candidate(root,bind_packages(root,plan),plan)
    sealed=dict(contract=CONTRACT,**header,complete=True,qa=_artifact(qa_path),
        parent_registration=plan["binding"]["parent_registration"],source_plan_sha256=sha256_file(root/"barrier_rule_b_candidate_plan.json"),
        phase_profile_sha256=sha256_file(root/"phase_optical_regions.json"),studio_preparation_sha256=sha256_file(root/"studio_preparation.json"),
        frames=sorted(completed,key=lambda row:(row["run"],row["frame_index"])),mass_eligible=False,
        scientific_acceptance=False,automatic_activation=False,training=False,model_inference=False,component_filtering=False)
    atomic_json(root/"barrier_rule_b_candidate_receipt.json",sealed)
    return sealed


def build_candidate(parent,output,geometry_manifest,*,workers=24,threads=1,resume=False,pilot=False):
    if not 1<=workers<=24 or threads!=1 or workers>max(1,(os.cpu_count() or 1)-4):
        raise ValueError("Rule B CPU allocation must leave cores free")
    below_normal_priority()
    root=Path(output).resolve()
    plan=prepare_candidate(parent,root,geometry_manifest,resume=resume,workers=min(8,workers))
    rows=plan["source_inventory"]
    if pilot:
        rows=[row for row in rows if row["run"]=="ac28" and row["frame_index"] in (8,16)]
        if len(rows)!=2:
            raise ValueError("Rule B pilot requires AC28 frames8 and16")
    with JobLease(root,plan["job_sha256"]):
        log=ProgressLog(root)
        completed=[]
        started=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initializer,initargs=(threads,)) as pool:
            pending={pool.submit(apply_frame,str(root),row,plan["pipeline_sha256"]):row for row in rows}
            for future in as_completed(pending):
                item=future.result()
                assigned=pending[future]
                if (item["run"],item["frame_index"])!=(assigned["run"],assigned["frame_index"]):
                    raise ValueError("Rule B worker returned another frame")
                completed.append(item)
                elapsed=time.monotonic()-started
                log.event("barrier_rule_b_frame",run=item["run"],frame_index=item["frame_index"],completed_frames=len(completed),
                          total_frames=len(rows),elapsed_seconds=elapsed,eta_seconds=elapsed*(len(rows)-len(completed))/len(completed))
        if pilot:
            receipt=dict(contract=CONTRACT,pilot=True,complete=True,model_identity=plan["model_identity"],
                         frames=sorted(completed,key=lambda row:(row["run"],row["frame_index"])))
            atomic_json(root/"barrier_rule_b_pilot_receipt.json",receipt)
            return receipt
        return seal_candidate(root,plan,completed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("parent","output","geometry-manifest"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=24)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    build_candidate(args.parent,args.output,args.geometry_manifest,workers=args.workers,threads=args.threads,
                    pilot=args.pilot,resume=args.resume)


if __name__=="__main__":
    main()
