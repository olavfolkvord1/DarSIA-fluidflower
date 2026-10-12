"""Source-bound recovery of completed frames, with explicit categorical-fill lineage.

The failed original job and all its artifacts remain immutable. Completed
outputs are either copied byte-for-byte or derived solely by the declared
prior-fill policy. Uncommitted outputs are never imported as completed frames.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import io
from pathlib import Path

import numpy as np
from PIL import Image

from . import run_ffac_rig_camera_inference as producer
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, copy_file_verified, read_json, sha256_file, utc_now
from .run_ffac_rig_camera_candidate import artifact
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path

CONTRACT = "ffac-rig-camera-source-bound-categorical-fill-recovery-v1"


def _key(source):
    return f"{source['run']}/frame_{source['frame_index']:03d}"


def frame_binding(inference, source):
    return dict(pipeline_sha256=inference["pipeline_sha256"],source=source,
                upstream_preparation_sha256=inference["binding"]["scientific_preparation"]["sha256"])


def validate_original_frame(receipt, original, source):
    expected=frame_binding(original,source)
    if (receipt.get("contract")!=producer.FRAME_CONTRACT or receipt.get("complete") is not True
            or receipt.get("binding")!=expected or receipt.get("qa",{}).get("all_pass") is not True):
        raise ValueError("Recovery requires an original completed, source-identical passing frame")
    if receipt.get("inherited_categorical_fill") is not None:
        raise ValueError("The original producer already had a different prior-fill policy")
    return receipt


def create_recovery_plan(original_plan, output_root, *, resume=False):
    original_asset=artifact(original_plan)
    original=read_json(bound_path(original_asset))
    science=Path(original["upstream_plan"]["binding"]["candidate"])
    root=Path(output_root).resolve()
    if root==science.resolve() or root.exists() and not resume:
        raise ValueError("Recovery requires a new candidate namespace")
    if original.get("job_sha256")!=canonical_sha256(original["binding"]):
        raise ValueError("Original inference binding is corrupt")
    state=producer.load_scientific_preparation(original["binding"]["upstream_plan"]["path"])
    if state["plan"]!=original["upstream_plan"] or state["preparation_asset"]!=original["binding"]["scientific_preparation"]:
        raise ValueError("Recovery cannot change the original scientific preparation")
    current=producer._code_binding()
    changed={key for key in set(current)|set(original["binding"]["code"])
             if current.get(key)!=original["binding"]["code"].get(key)}
    allowed={"driver","darsia.fluidflower.phase_retained_source_inference"}
    if not changed<=allowed:
        raise ValueError("Recovery changed an undeclared numerical module: "+repr(sorted(changed-allowed)))
    inventory={}
    for source in original["source_inventory"]:
        path=science/"rig_camera_frames"/source["run"]/f"frame_{source['frame_index']:03d}.json"
        if path.exists():
            receipt=read_json(path)
            validate_original_frame(receipt,original,source)
            inventory[_key(source)]=artifact(path)
    if len(inventory)!=445:
        raise ValueError("Recovery inventory differs from the audited445 completed frames")
    extended={}
    for path in (science/"extended_scientific").glob("*/receipt.json"):
        row=read_json(path)
        if row.get("complete") is not True or row.get("binding",{}).get("inference_job_sha256")!=original["job_sha256"]:
            raise ValueError("Original RAW ring receipt is incomplete or differently bound")
        extended[path.parent.name]=artifact(path)
    recovery=dict(contract=CONTRACT,created_utc=utc_now(),original_inference_plan=original_asset,
        original_source_plan=state["plan_asset"],scientific_source_root=str(science),output_root=str(root),
        original_inference_job_sha256=original["job_sha256"],original_pipeline_sha256=original["pipeline_sha256"],
        current_code=current,recovery_driver_sha256=sha256_file(__file__),completed_frames=inventory,
        extended_scientific=extended,completed_import_count=len(inventory),fresh_frame_count=864-len(inventory),
        partial_outputs_imported=False,strict_interior_guard_unchanged=True,
        full_prior_fill_equivalence_required=True,cpu_only=True,training=False,labels_changed=False,
        mass_eligible=False,automatic_activation=False)
    recovery_path=root/"rig_camera_recovery_plan.json"
    if recovery_path.exists():
        old=read_json(recovery_path)
        expected=dict(recovery,created_utc=old["created_utc"])
        if not resume or old!=expected:raise ValueError("Recovery resume changed its immutable inputs/code")
        inference=read_json(root/producer.PLAN_FILE)
        if inference["binding"]["recovery_plan"]!=artifact(recovery_path):raise ValueError("Recovery inference plan changed")
        return inference
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(recovery_path,recovery)
    copy_file_verified(bound_path(state["plan_asset"]),root/"rig_camera_candidate_plan.json")
    parent=Path(original["upstream_plan"]["binding"]["parent"])
    producer._copy_heads_and_request(parent,root)
    summary=read_json(science/"summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=recovery["created_utc"],
        display_name="FF_AC · G3 gas · Rig domain · Camera clock · Source-bound fill recovery",
        inherited_failed_candidate_id=science.name,retraining_performed=False,
        scientific_acceptance=False,mass_eligible=False)
    atomic_json(root/"summary.json",summary)
    from darsia.fluidflower.phase_validation import _model_identity
    identity=list(_model_identity(root))
    binding=deepcopy(original["binding"])
    binding.update(code=current,recovery_plan=artifact(recovery_path),output_root=str(root),
        categorical_fill_policy="source-bound inherited categorical fill only on final non-modelled filled pixels")
    result=deepcopy(original)
    result.update(binding=binding,job_sha256=canonical_sha256(binding),
        pipeline_sha256=canonical_sha256(dict(binding=binding,model_identity=identity)),
        model_identity=identity,created_utc=recovery["created_utc"],
        scientific_source_root=str(science),output_root=str(root),recovery=recovery)
    atomic_json(root/producer.PLAN_FILE,result)
    from darsia.fluidflower.phase_rig_bottom_review_policy import write_pending_policy
    write_pending_policy(root,source_plan=state["plan_asset"])
    return result


def _copy_asset(asset,destination):
    copy_file_verified(bound_path(asset),destination)
    return dict(asset,path=str(destination),sha256=sha256_file(destination))


def _derived_qa(root,inference,source,receipt,current_receipt,context,labels,maps,proof,descriptor):
    from darsia.fluidflower.phase_rig_domain_inference import interior_difference
    from darsia.fluidflower.phase_ffac_rig_qa import frame_area_report
    from darsia.fluidflower.phase_ffac_rig_edge_qa import added_strip_report
    run,index=source["run"],source["frame_index"]
    old_index=source["parent_source_frame_index"] if not source["source_changed"] else index
    parent=Path(inference["upstream_plan"]["binding"]["parent"])
    old=read_json(parent/"barrier_rule_b_frames"/run/f"frame_{old_index:03d}.json")
    with Image.open(bound_path(old["labels_map"])) as image:before=np.array(image)
    native=read_json(current_receipt["task"]["parent_native_manifest"]["path"])
    row=native["frame_sources"][old_index]
    photo=np.load(bound_path(dict(path=row["v5_photo_sand_mask_path"],sha256=row["v5_photo_sand_mask_sha256"])),allow_pickle=False)
    before[~photo]=255
    inside=interior_difference(before,producer._crop(labels),photo,(.00016448,.00016448),source_changed=source["source_changed"])
    if not source["source_changed"] and not inside["interior_exact"]:
        raise ValueError("Derived completed-frame interior differs more than30mm from the old map edge")
    with np.load(bound_path(descriptor["classes"]),allow_pickle=False) as archive:
        classes={key:archive[key] for key in ("modelled","filled_observed","filled_unobserved")}
    hidden=np.zeros(labels.shape,bool)
    for name in producer.HIDDEN_CHART_NAMES:hidden|=context.excluded_by_name[name]
    area=frame_area_report(producer._canvas(before,phase_shape_hw=labels.shape),labels,context.pixel_area_mm2,
        dict(rig_domain=context.rig_domain,**classes),hardware_masks=context.excluded_by_name,hidden_chart_mask=hidden)
    strips=added_strip_report(producer._canvas(before,phase_shape_hw=labels.shape),labels,context.pixel_area_mm2,
        dict(rig_domain=context.rig_domain,**classes),inner_polygon_native=source["rig_geometry"]["frame_inner_polygon_native"],
        photo_top_keep=producer.load_array(source["rig_geometry"]["frame_photo_top_keep"]),
        native_origin_xy=producer.metadata.ORIGIN_XY,parent_photo_domain=producer._canvas(photo,False,phase_shape_hw=labels.shape))
    qa=deepcopy(receipt["qa"])
    qa.update(interior_difference=inside,interior_changed_gt30mm_pixels=None if source["source_changed"] else inside["changed_pixels"]["over_30_mm"],
        map_path=proof["path"],map_sha256=proof["sha256"],studio_map_path=maps["combined"]["path"],
        studio_map_sha256=maps["combined"]["sha256"],area_map_sha256=descriptor["area_map"]["sha256"],
        classes_sha256=descriptor["classes"]["sha256"],area_report=area,
        added_strip_by_side=dict(strips,source_changed=source["source_changed"]),
        all_added_co2_zero=strips["added_co2_cm2"]==0,
        outer_photo_extension_co2_zero=sum(s["outer_photo_extension_co2_cm2"] for s in strips["sides"].values())==0,all_pass=True)
    return qa


def import_completed_frame(root,inference,source):
    """Check the whole helper population before declaring copied/derived reuse."""
    from darsia.fluidflower.phase_retained_source_inference import prepare_retained_source_cache,preserve_prior_categorical_fill
    root=Path(root);recovery=inference["recovery"]
    item=recovery["completed_frames"].get(_key(source))
    if item is None:return None
    old=read_json(bound_path(item))
    expected=dict(pipeline_sha256=recovery["original_pipeline_sha256"],source=source,
        upstream_preparation_sha256=inference["binding"]["scientific_preparation"]["sha256"])
    if old.get("binding")!=expected or old.get("complete") is not True or old.get("qa",{}).get("all_pass") is not True:
        raise ValueError("Imported completed frame lost its exact original binding/QA")
    for asset in [*old["maps"].values(),old["labels_map"],old["rig_domain_revision"]["classes"],old["rig_domain_revision"]["area_map"]]:bound_path(asset)
    current_receipt=producer._prepared(root,inference,source["scientific_task_id"])
    current,valid,audit=producer.load_scientific(current_receipt)
    with np.load(bound_path(old["rig_domain_revision"]["classes"]),allow_pickle=False) as archive:
        classes={name:archive[name] for name in ("modelled","filled_observed","filled_unobserved")}
        domain=archive["rig_domain"]
    cache=prepare_retained_source_cache(source["retained_source_cache"],run=source["run"],
        frame_index=source["parent_source_frame_index"] if not source["source_changed"] else source["frame_index"],
        current_source_sha256=source["source_sha256"],current_log=current,allowed_domain=producer._crop(domain),producer_audit=audit)
    with Image.open(bound_path(old["labels_map"])) as image:labels=np.array(image)
    corrected,populations,fill_audit=preserve_prior_categorical_fill(cache,producer._crop(labels),
        **{name:producer._crop(mask) for name,mask in classes.items()})
    changed=int(np.count_nonzero(corrected!=producer._crop(labels)))
    x,y,w,h=producer.metadata.SOURCE_BOX;labels[y:y+h,x:x+w]=corrected
    run,index=source["run"],source["frame_index"]
    encoded=io.BytesIO()
    np.savez_compressed(encoded,shape_hw=np.asarray(producer.metadata.SOURCE_SHAPE,np.int64),
        **{name:np.packbits(mask,bitorder="little") for name,mask in populations.items()})
    fill_asset=producer._write_bytes(root/"rig_camera_revision/inherited_fill"/run/f"frame_{index:03d}.packbits.npz",encoded.getvalue())
    descriptor=deepcopy(old["rig_domain_revision"])
    descriptor["classes"]=_copy_asset(descriptor["classes"],root/"rig_camera_revision/context"/run/f"frame_{index:03d}.classes.npz")
    descriptor["area_map"]=_copy_asset(descriptor["area_map"],root/"rig_camera_revision/context"/run/f"frame_{index:03d}.area.npz")
    if changed:
        maps,proof=producer.write_maps(root,run,index,inference["pipeline_sha256"],labels,phase_shape_hw=labels.shape)
        context,_=producer.geometry_context(source["rig_geometry"],valid,audit)
        with np.load(bound_path(descriptor["area_map"]),allow_pickle=False) as archive:
            if not np.array_equal(context.pixel_area_mm2,archive["pixel_area_mm2"]):raise ValueError("Recovery changed physical area context")
        qa=_derived_qa(root,inference,source,old,current_receipt,context,labels,maps,proof,descriptor)
    else:
        folder=root/"validation/run_packages"/run/"review_assets/native_phase_maps"/run/("A_"+inference["pipeline_sha256"][:16])
        maps={name:_copy_asset(value,folder/(f"frame_{index:03d}.png" if name=="combined" else f"{name}/frame_{index:03d}.png")) for name,value in old["maps"].items()}
        proof=_copy_asset(old["labels_map"],root/"production_labels"/run/f"frame_{index:03d}.png")
        qa=deepcopy(old["qa"])
        qa.update(map_path=proof["path"],map_sha256=proof["sha256"],studio_map_path=maps["combined"]["path"],
            studio_map_sha256=maps["combined"]["sha256"],area_map_sha256=descriptor["area_map"]["sha256"],classes_sha256=descriptor["classes"]["sha256"])
    result=deepcopy(old)
    result.update(created_utc=utc_now(),binding=frame_binding(inference,source),qa=qa,maps=maps,labels_map=proof,
        rig_domain_revision=descriptor,inherited_categorical_fill=dict(audit=fill_audit,populations=fill_asset,
            provenance="source-bound parent categorical fill; never modelled or training support"),
        import_lineage=dict(contract=CONTRACT,original_completed_frame=item,original_pipeline_sha256=recovery["original_pipeline_sha256"],
            recovery_plan=inference["binding"]["recovery_plan"],fresh_inference_repeated=False,
            mode="derived-categorical-fill" if changed else "byte-identical-completed-import",
            full_helper_changed_pixels=changed,class_population_bit_identical=True,
            original_fresh_inference_audit_inherited=True,strict_interior_guard_unchanged=True))
    path=root/"rig_camera_frames"/run/f"frame_{index:03d}.json"
    atomic_json(path,result)
    return dict(run=run,frame_index=index,receipt=artifact(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-inference-plan",required=True)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--workers",type=int,default=24)
    parser.add_argument("--resume",action="store_true")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--probe",action="store_true")
    args=parser.parse_args(argv)
    inference=create_recovery_plan(args.original_inference_plan,args.output_root,resume=args.resume)
    if args.dry_run:
        print("Source-bound recovery planned:445 completed imports;419 fresh frames; no inference submitted.",flush=True)
        return 0
    if args.probe:
        selected=[row for row in inference["source_inventory"] if row["run"]=="ac28" and row["frame_index"] in (0,6)]
        results=[producer.infer_frame(args.output_root,inference,row) for row in selected]
        atomic_json(Path(args.output_root)/"rig_camera_recovery_probe.json",dict(complete=True,frames=results,
            recovery_plan=inference["binding"]["recovery_plan"],registration_performed=False))
        print("Recovery probe complete: importedAC28f0 plus formerly rejectedAC28f6; strict QA retained.",flush=True)
        return 0
    producer.run_inference(inference,workers=args.workers)
    print("Source-bound recovery candidate complete; not activated.",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
