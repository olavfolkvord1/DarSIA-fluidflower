"""Immutable P1-final Aâ€“D postprocessing, with no model execution."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower import phase_ffac_p1_final2 as rules
from darsia.fluidflower.phase_ffac_p1_final_bottom_report import transport_curve
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from .refit_ffac_g3_downstream import atomic_json, canonical_sha256, copy_file_verified, read_json, sha256_file, utc_now, JobLease
from .run_ffac_rig_camera_candidate import artifact
from .build_ffac_p1_final_candidate import _write, _png, _overlay

CONTRACT = "ffac-g3-t4c-p1-final2-candidate-v1"
SPEC_SHA256 = "e843e61751b795984550cfe853548ea7ca4e50364a5cb44681621d485abd2c0b"
ZONE_ARRAY_SHA256 = "76db153124af785ff826ea67e8da9c4e42bf3e2d3d16f4ad4958a01a21a68b8d"
PLAN = "p1_final2_plan.json"
SEAL = "rig_camera_candidate_receipt.json"
QA = "production_qa/p1_final2_qa.json"
RUN_DIRECTORY = "p1_final2_runs"
MASK_DIRECTORIES = dict(A="below_sand_floor_masks", B="floor_connectivity_masks", C="floor_contact_fill_v2", D="gas_envelope_masks")
OVERLAYS = dict(A="below_sand_floor", B="removed_floor_connectivity", C="floor_contact_fill_v2", D="removed_gas_envelope")
COLORS = dict(A=(125,125,125,255), B=(255,157,66,255), C=(84,156,242,255), D=(181,102,219,255))


def code_binding():
    names = ("darsia.fluidflower.tools.build_ffac_p1_final2_candidate", "darsia.fluidflower.phase_ffac_p1_final2",
        "darsia.fluidflower.phase_ffac_p1_final2_carryover", "darsia.fluidflower.phase_ffac_p1_final_bottom_report",
        "darsia.fluidflower.phase_ffac_bottom_band_diagnostics", "darsia.fluidflower.tools.build_ffac_p1_final_candidate",
        "darsia.fluidflower.tools.run_ffac_rig_camera_inference", "darsia.fluidflower.phase_ffac_rig_camera_metadata")
    return {name: sha256_file(importlib.import_module(name).__file__) for name in names}


def check_gate(gate):
    g = gate["gate"]
    if (gate.get("complete") is not True or gate.get('contract')!='ffac-final-round-floor-measurement-gate-v1'
            or g.get("passes") is not True or g.get('required_passing_runs')!=39 or g.get('maximum_median_absolute_residual_mm')!=1.0
            or any(g.get(k,0)<39 for k in ('reference_passing_runs','last_passing_runs'))):
        raise ValueError("Step A photographic gate failed; Bâ€“D must not run")
    return g


def prepare_candidate(parent, output, spec, gate_path, zones_path, *, backend="auto", resume=False):
    from darsia.fluidflower.phase_ffac_p1_final_review import p1_final_saved_review
    from darsia.fluidflower.phase_validation import _model_identity
    from .register_track1_studio_candidate import registration_path
    parent, root, spec, gate_path, zones_path = map(lambda p: Path(p).resolve(), (parent,output,spec,gate_path,zones_path))
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError("Use a detached new candidate")
    if sha256_file(spec) != SPEC_SHA256:
        raise ValueError("Approved Aâ€“D specification changed")
    if p1_final_saved_review(parent)["placement"] != "P1-final":
        raise ValueError("Only exact sealed P1-final is allowed")
    gate = read_json(gate_path); check_gate(gate)
    if gate["candidate"] != artifact(parent/SEAL):
        raise ValueError("Floor measurements bind another parent")
    curve = gate["canonical_curve"]["asset"]
    bound_path(curve)
    zones_asset = artifact(zones_path)
    with np.load(zones_path, allow_pickle=False) as archive:
        zones = archive["zones"]
    if zones.shape != (2979,5472) or zones.dtype != np.uint8 or hashlib.sha256(zones.tobytes()).hexdigest() != ZONE_ARRAY_SHA256:
        raise ValueError("The locked L1 zone array changed")
    zones_asset["array_sha256"] = ZONE_ARRAY_SHA256
    seal = read_json(parent/SEAL)
    if seal.get("complete") is not True or (seal["run_count"],seal["frame_count"]) != (43,864):
        raise ValueError("P1-final is not complete43/864")
    actual = rules.select_backend(backend)
    inputs = dict(parent_seal=artifact(parent/SEAL), parent_registration=artifact(registration_path(parent.name)), spec=artifact(spec),
        a_gate_report=artifact(gate_path), pooled_floor_curve=curve, zones=zones_asset, code=code_binding(), rules=rules.RULE,
        training=False,inference=False,annotations_read=False,mass_eligible=False,bottom_mass_allowed=False,backend=actual)
    binding = dict(parent=str(parent),heads=seal["heads"],inputs=inputs)
    if (root/PLAN).exists():
        plan = read_json(root/PLAN)
        if not resume or plan["binding"] != binding:
            raise ValueError("Resume changes immutable Aâ€“D binding")
        return plan
    if root.exists() or resume:
        raise ValueError("Use a new folder or an exact existing resume")
    root.mkdir(parents=True,exist_ok=False)
    for folder in ("models","regional_models"):
        for path in (parent/folder).rglob("*"):
            if path.is_file(): copy_file_verified(path,root/path.relative_to(parent))
    for name in ("regional_model.json","training_request.json","expected_review_runs.json","physical_scale_measurement.json","vertical_scale_check.json"):
        if (parent/name).is_file(): copy_file_verified(parent/name,root/name)
    summary = read_json(parent/"summary.json")
    summary.update(model_id=root.name,campaign_id=root.name,created_utc=utc_now(),
        display_name="FF_AC \u00b7 G3-T4c P1 final 2 \u00b7 Sand floor \u00b7 Floor connectivity \u00b7 Gas injection envelope \u00b7 08.10.2026",
        inherited_base_id=parent.name,training_performed_in_this_revision=False,optical_training_performed=False,regional_training_performed=False,
        co2_training_performed=False,mass_eligible=False,scientific_acceptance=False,automatic_activation=False,bottom_mass_allowed=False,
        blind_hold_runs=[],p1_final2=True,p1_final2_rules=rules.RULE)
    atomic_json(root/"summary.json",summary)
    parents = {f"{row['run']}/frame_{row['frame_index']:03d}":row["receipt"] for row in seal["frames"]}
    sources=[];epochs={}
    for run in sorted({key.split('/')[0] for key in parents}):
        native=read_json(parent/"source_alias/native_frames"/run/"manifest.json")
        for source in native["frame_sources"]:
            key=f"{run}/frame_{source['frame_index']:03d}"
            sources.append(dict(run=run,frame_index=source["frame_index"],elapsed_h=source["elapsed_h"],source_sha256=source["source_sha256"],parent_receipt=parents[key]))
        epochs[run]=read_json(bound_path(parents[f"{run}/frame_000"]))["light_epoch_id"]
    if len(sources)!=864 or {(s['run'],s['frame_index']) for s in sources} != {(f['run'],f['frame_index']) for f in seal['frames']}:
        raise ValueError("P1-final source inventory differs")
    identity=list(_model_identity(root))
    plan=dict(contract=CONTRACT,created_utc=summary["created_utc"],binding=binding,source_inventory=sources,run_to_epoch=epochs,
        model_identity=identity,job_sha256=canonical_sha256(binding),pipeline_sha256=canonical_sha256(dict(binding=binding,model_identity=identity)),
        output_root=str(root),placement="P1-final2",delta=seal["delta"],backend=actual)
    atomic_json(root/PLAN,plan)
    profile=read_json(parent/"phase_rig_camera.json")
    for key in list(profile):
        if key.endswith("_revision") and key != "rig_domain_revision": profile.pop(key)
    profile["parent_model_identity"]=list(_model_identity(parent))
    profile["p1_final2_revision"]=dict(contract=CONTRACT,plan=artifact(root/PLAN))
    atomic_json(root/"profile_template.json",profile)
    copy_file_verified(spec,root/"approved_p1_final2_spec.md")
    for name in inputs["code"]:
        path=Path(importlib.import_module(name).__file__)
        copy_file_verified(path,root/"executed_code"/(name+".py"))
    return plan


def transported_floor(curve, row):
    shape=row["rig_domain_revision"]["phase_shape_hw"]
    origin=row["rig_domain_revision"]["native_origin_xy"]
    # Canonical AC28 pre-injection H is identity, verified by the gate receipt.
    result=transport_curve(curve,H_source_to_target=np.asarray(row["H_ac28_to_frame_native"],float),
        source_origin_xy=(-416,-496),target_origin_xy=origin,target_width=shape[1])
    good=np.flatnonzero(np.isfinite(result))
    if not len(good): raise ValueError("Transported sand floor has no valid columns")
    # The locked pooled line holds its value beyond the ends. Only occupied
    # domain columns are used; this is not extension of photographic evidence.
    result[:good[0]]=result[good[0]];result[good[-1]+1:]=result[good[-1]]
    if not np.isfinite(result).all(): raise ValueError("Transported pooled floor contains an interior gap")
    return result


def revised_context(root, run, stem, row, curve):
    from .run_ffac_rig_camera_inference import load_array
    descriptor=deepcopy(row["rig_domain_revision"])
    with np.load(bound_path(descriptor["classes"]),allow_pickle=False) as z:
        classes={key:z[key].copy() for key in z.files}
    old=classes["rig_domain"]
    floor=transported_floor(curve,row)
    below=old & (np.arange(old.shape[0])[:,None]>floor[None,:])
    classes["rig_domain"] &= ~below
    for key in ("modelled","filled_observed","filled_unobserved"):
        classes[key] &= ~below
    classes["below_sand_floor"]=below
    area=load_array(descriptor["area_map"])
    stream=io.BytesIO();np.savez_compressed(stream,**classes)
    asset=_write(root/"rig_camera_revision/context"/run/(stem+".classes.npz"),stream.getvalue())
    descriptor["classes"]=asset
    parameters=descriptor["parameters"]
    parameters["domain_area_cm2"]=float(area[classes["rig_domain"]].sum(dtype=np.float64))/100
    parameters["area_cm2_by_class"]={key:float(area[classes[key]].sum(dtype=np.float64))/100 for key in ("modelled","filled_observed","filled_unobserved")}
    parameters["excluded_below_sand_floor_area_cm2"]=float(area[below].sum(dtype=np.float64))/100
    parameters["domain_revision"]="pooled_sand_floor_v1"
    descriptor["sand_floor_revision"]=dict(contract=CONTRACT,curve="AC28 pooled photo floor transported by existing frame H",below_sand_floor_class=True,
        parent_classes=row["rig_domain_revision"]["classes"],area_values_unchanged=True,bottom_mass_allowed=False)
    return descriptor,classes,below,area


def _warp(mask, H, shape):
    return cv2.warpPerspective(mask.astype(np.uint8),H,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)


def native_l1(zones,row):
    H=np.asarray(row["H_ac28_to_frame_native"],float)
    native=_warp(zones==4,H,(2979,5472))
    full=np.zeros(row["rig_domain_revision"]["phase_shape_hw"],bool)
    full[496:496+2979,416:416+5472]=native
    full[496+2979:,416:416+5472]=native[-1]
    return full


def transported_carryover(mask, provenance, row):
    origin=np.asarray(row["rig_domain_revision"]["native_origin_xy"],float)
    source_origin=np.asarray(provenance["native_origin_xy"],float)
    A=np.eye(3);A[:2,2]=source_origin
    B=np.eye(3);B[:2,2]=-origin
    H=np.asarray(row["H_ac28_to_frame_native"],float) @ np.linalg.inv(np.asarray(provenance["H_ac28_to_frame_native"],float))
    return _warp(mask,B@H@A,tuple(row["rig_domain_revision"]["phase_shape_hw"]))


def prepare_run_context(root,plan,run):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    from darsia.fluidflower.phase_ffac_p1_final2_carryover import generate_carryover
    parent=Path(plan["binding"]["parent"])
    native=read_json(parent/"source_alias/native_frames"/run/"manifest.json")
    inputs=read_json(bound_path(native[meta.SOURCE_KEY]["input_manifest"]))
    curve=np.load(bound_path(plan["binding"]["inputs"]["pooled_floor_curve"]),allow_pickle=False)
    zones=np.load(bound_path(plan["binding"]["inputs"]["zones"]),allow_pickle=False)["zones"]
    proxy=root/"carryover_proxy"/run
    if (proxy/'carryover_receipt.json').exists():
        provenance=read_json(proxy/'carryover_receipt.json')
        if provenance['native_manifest']!=artifact(parent/'source_alias/native_frames'/run/'manifest.json'):
            raise ValueError('Resumed carryover source changed')
        for key in ('source_frame_receipt','classes','measured_area','input_manifest','helper_code','diagnostic_code','unchanged_reporter_code'):
            if provenance.get(key):bound_path(provenance[key])
        asset=provenance['mask'];shape=tuple(asset['shape_hw'])
        carryover=np.unpackbits(np.load(bound_path(asset),allow_pickle=False),count=int(np.prod(shape)),bitorder=asset['bitorder']).reshape(shape).astype(bool)
        known=provenance['known']
    else:
        carryover,known,provenance=generate_carryover(parent,run,proxy)
    times=[r['elapsed_h'] for r in inputs['frames']]
    u1=next((i for i,t in enumerate(times) if t>=.935),None)
    u2=next((i for i,t in enumerate(times) if t>=2.389),None)
    if u1 is None or u2 is None: raise ValueError("Source run lacks injection-end envelope frame")
    sources={s['frame_index']:s for s in plan['source_inventory'] if s['run']==run}
    envelope_binding=dict(pipeline_sha256=plan['pipeline_sha256'],parent_frames=[sources[i]['parent_receipt'] for i in range(u2+1)],
        floor_curve=plan['binding']['inputs']['pooled_floor_curve'],u1_frame_index=u1,u2_frame_index=u2,u1_elapsed_h=times[u1],u2_elapsed_h=times[u2])
    envelope_path=root/'envelopes'/run/'receipt.json'
    shape=tuple(inputs['frames'][0]['rig_domain_revision']['phase_shape_hw'])
    if envelope_path.exists():
        saved=read_json(envelope_path)
        if saved['contract']!=CONTRACT or saved['binding']!=envelope_binding:raise ValueError('Resumed gas envelope binding changed')
        env1,env2=[np.unpackbits(np.load(bound_path(saved[key]),allow_pickle=False),count=int(np.prod(shape))).reshape(shape).astype(bool) for key in ('envelope1','envelope2')]
        return dict(native=native,inputs=inputs,curve=curve,zones=zones,carryover=carryover,carryover_known=known,carryover_provenance=provenance,
            env1=env1,env2=env2,u1=u1,u2=u2)
    union1=np.zeros(shape,bool);union2=union1.copy()
    for row in inputs['frames'][:u2+1]:
        labels=_png(read_json(bound_path(sources[row['frame_index']]['parent_receipt']))['labels_map'])
        with np.load(bound_path(row['rig_domain_revision']['classes']),allow_pickle=False) as z: old=z['rig_domain']
        below=old & (np.arange(old.shape[0])[:,None]>transported_floor(curve,row)[None,:])
        gas=(labels==2)&old&~below
        union2|=gas
        if row['frame_index']<=u1:union1|=gas
    env1,env2,audit=rules.build_envelopes(union1,union2,backend=plan['backend'])
    envelope_assets={}
    for key,value in (('envelope1',env1),('envelope2',env2)):
        stream=io.BytesIO();np.save(stream,np.packbits(value),allow_pickle=False)
        envelope_assets[key]=dict(_write(root/'envelopes'/run/(key+'.npy'),stream.getvalue()),shape_hw=list(shape),storage='packbits-npy')
    atomic_json(envelope_path,dict(contract=CONTRACT,run=run,binding=envelope_binding,audit=audit,carryover_known=known,carryover_provenance=provenance,**envelope_assets))
    return dict(native=native,inputs=inputs,curve=curve,zones=zones,carryover=carryover,carryover_known=known,carryover_provenance=provenance,
        env1=env1,env2=env2,u1=u1,u2=u2)


def _packed(root,directory,run,index,mask,source):
    stream=io.BytesIO();np.save(stream,np.packbits(mask),allow_pickle=False)
    return dict(_write(root/directory/run/f'frame_{index:03d}.npy',stream.getvalue()),frame_index=index,
        source_sha256=source['source_sha256'],shape_hw=list(mask.shape),storage='packbits-npy',contract=CONTRACT)


def load_frame(root,plan,source,context):
    p1=read_json(bound_path(source['parent_receipt']));full=p1['binding']['source']
    if any(source[k]!=full[k] for k in ('run','frame_index','source_sha256','elapsed_h')):raise ValueError('Source inventory changed')
    row=context['inputs']['frames'][source['frame_index']]
    if row['source_sha256']!=source['source_sha256']:raise ValueError('Source frame binding changed')
    labels=_png(p1['labels_map'])
    descriptor,classes,below,area=revised_context(root,source['run'],f"frame_{source['frame_index']:03d}",row,context['curve'])
    support=(classes['modelled']|classes['filled_observed']|classes['filled_unobserved']) & classes['rig_domain']
    carry=transported_carryover(context['carryover'],context['carryover_provenance'],row)&classes['rig_domain'] if context['carryover_known'] else None
    return dict(source=source,p1=p1,labels=labels,descriptor=descriptor,classes=classes,below=below,area=area,support=support,
        carryover=carry,l1=native_l1(context['zones'],row),row=row)


def compute(loaded,context,backend):
    return rules.apply_final_round(loaded['labels'],loaded['below'],loaded['support'],loaded['carryover'],loaded['l1'],
        context['env1'],context['env2'],elapsed_h=loaded['source']['elapsed_h'],frame_index=loaded['source']['frame_index'],
        u1_index=context['u1'],u2_index=context['u2'],pixel_area_mm2=loaded['area'],backend=backend,corrected_domain_mask=loaded['classes']['rig_domain'])


def save_frame(root,plan,loaded,result):
    from .run_ffac_rig_camera_inference import write_maps
    source,p1=loaded['source'],loaded['p1'];run,index=source['run'],source['frame_index'];full=p1['binding']['source']
    if not result.qa['all_pass']:raise ValueError(f"Aâ€“D QA failed: {run}/{index}: {result.qa}")
    maps,labels=write_maps(root,run,index,plan['pipeline_sha256'],result.labels,phase_shape_hw=result.labels.shape)
    masks={key:_packed(root,directory,run,index,result.step_masks[key],full) for key,directory in MASK_DIRECTORIES.items()}
    previous={}
    for key,value in result.previous_labels.items():
        stream=io.BytesIO();np.savez_compressed(stream,previous_labels=value)
        previous[key]=dict(_write(root/'previous_labels'/key/run/f'frame_{index:03d}.npz',stream.getvalue()),storage='npz-array',array_key='previous_labels',
            frame_index=index,source_sha256=source['source_sha256'],shape_hw=list(result.labels.shape),contract=CONTRACT)
    overlays={OVERLAYS[key]:_overlay(root,run,index,plan['pipeline_sha256'],result.step_masks[key],COLORS[key],OVERLAYS[key]) for key in OVERLAYS}
    qa=dict(result.qa,run=run,frame_index=index,source_sha256=source['source_sha256'],elapsed_h=source['elapsed_h'],
        carryover_known=loaded['carryover'] is not None,light_epoch_id=p1['light_epoch_id'],
        domain_removed_cm2=float(loaded['area'][loaded['below']].sum(dtype=np.float64))/100)
    receipt=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),binding=dict(pipeline_sha256=plan['pipeline_sha256'],source=full,parent_receipt=source['parent_receipt']),
        run=run,frame_index=index,source_sha256=source['source_sha256'],elapsed_h=source['elapsed_h'],placement='P1-final2',delta=plan['delta'],light_epoch_id=p1['light_epoch_id'],
        before_labels=p1['labels_map'],labels_map=labels,maps=maps,previous_labels=previous,removed_display_layers=overlays,
        rig_domain_revision=loaded['descriptor'],full_extended_scientific=p1['full_extended_scientific'],baseline_extended_scientific=p1['baseline_extended_scientific'],
        fill_reason_masks={name:_packed(root,'floor_contact_fill_reasons/'+name,run,index,value,full)
            for name,value in (('a',result.fill_reason_a),('b',result.fill_reason_b))},
        qa=qa,backend=result.qa['backend'],labels_changed=False,production_labels_modified=True,annotations_read=False,training=False,training_performed=False,
        inference_performed=False,co2_inference_performed=False,new_gas_inference_performed=False,mass_eligible=False,bottom_mass_allowed=False,
        **{MASK_DIRECTORIES[key]:value for key,value in masks.items()})
    path=root/'rig_camera_frames'/run/f'frame_{index:03d}.json';atomic_json(path,receipt)
    return dict(run=run,frame_index=index,receipt=artifact(path),qa=qa)


def assemble_run(root,plan,run,frames):
    from darsia.fluidflower import phase_ffac_rig_camera_metadata as meta
    parent=Path(plan['binding']['parent'])
    native=read_json(parent/'source_alias/native_frames'/run/'manifest.json');inputs=read_json(bound_path(native[meta.SOURCE_KEY]['input_manifest']))
    by_index={f['frame_index']:f for f in frames}
    if len(by_index)!=len(native['frame_sources']):raise ValueError('Incomplete final2 run')
    for record,row in zip(native['frame_sources'],inputs['frames']):
        saved=read_json(bound_path(by_index[record['frame_index']]['receipt']))
        if saved['source_sha256']!=record['source_sha256'] or row['source_sha256']!=record['source_sha256']:raise ValueError('Source order changed')
        for value in (record,row):
            value['rig_domain_revision']=saved['rig_domain_revision']
            for name in MASK_DIRECTORIES.values():value[name]=saved[name]
    curve=np.load(bound_path(plan['binding']['inputs']['pooled_floor_curve']),allow_pickle=False)
    ref_descriptor,_,_,_=revised_context(root,run,'reference',inputs['reference'],curve)
    inputs['reference']['rig_domain_revision']=ref_descriptor
    input_path=root/'source_alias/source_bindings'/(run+'.json');atomic_json(input_path,inputs)
    native=meta.build_native_metadata(read_json(bound_path(inputs['parent_native_manifest'])),read_json(bound_path(inputs['camera_timeline_run'])),
        native['frame_sources'],inputs['frames'],inputs['reference'],input_manifest=artifact(input_path))
    native_path=root/'source_alias/native_frames'/run/'manifest.json';atomic_json(native_path,native)
    result=dict(run=run,complete=True,native_manifest=artifact(native_path),input_manifest=artifact(input_path),frames=inputs['frames'],reference=inputs['reference'],
        frame_count=len(frames),receipts=[{k:v for k,v in f.items() if k!='qa'} for f in frames],pipeline_sha256=plan['pipeline_sha256'])
    atomic_json(root/RUN_DIRECTORY/(run+'.json'),result)
    return result


def bind_final2_packages(root,plan,source_runs):
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
            scientific_acceptance=False,p1_final2_revision=True,floor_contact_rule=False, bottom_mass_allowed=False)
        for key in ('p1_final_revision','p1_final_removed_top','p1_final_removed_floor'):
            job['quality_flags'].pop(key,None)
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
    if len(frames)!=864 or len({(f['run'],f['frame_index']) for f in frames})!=864:raise ValueError('Need exact43/864')
    source_runs=[assemble_run(root,plan,run,[f for f in frames if f['run']==run]) for run in sorted(plan['run_to_epoch'])]
    meta.validate_campaign_bindings(source_runs)
    profile=read_json(root/'profile_template.json');runtime=read_json(bound_path(profile['rig_domain_revision']['manifest']))
    runtime['sources']=[{k:v for k,v in row.items() if k!='receipts'} for row in source_runs]
    path=root/'rig_camera_revision/runtime_manifest.json';atomic_json(path,runtime)
    profile['rig_domain_revision']['manifest']=artifact(path);atomic_json(root/'phase_rig_camera.json',profile)
    packages=bind_final2_packages(root,plan,source_runs);assemble_saved_candidate(root,packages,plan)
    rows=[f['qa'] for f in frames]
    if not all(r['all_pass'] for r in rows):raise ValueError('Aâ€“D identities failed')
    gate=read_json(bound_path(plan['binding']['inputs']['a_gate_report']))['gate']
    a_gate=dict(all_pass=True,reference_passing_runs=gate['reference_passing_runs'],last_passing_runs=gate['last_passing_runs'],
        required_passing_runs=39,maximum_median_absolute_residual_mm=1.0,report=plan['binding']['inputs']['a_gate_report'])
    qa=dict(contract=CONTRACT,technical_all_pass=True,run_count=43,frame_count=864,rules=rules.RULE,backend=plan['backend'],rows=rows,a_gate=a_gate,
        source_time_raw_geometry_area_heads_delta_unchanged=True,domain_revision='pooled_sand_floor_v1',annotations_read=False,training=False,inference=False,
        cost=dict(cpu_seconds=sum(r['cpu_seconds'] for r in rows),gpu_seconds=sum(r['gpu_seconds'] for r in rows)))
    atomic_json(root/QA,qa)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
    _write(root/'production_qa/p1_final2_steps.csv',stream.getvalue().encode('utf-8'))
    seal=dict(contract=CONTRACT,complete=True,model_identity=plan['model_identity'],pipeline_sha256=plan['pipeline_sha256'],source_plan_sha256=sha256_file(root/PLAN),
        run_count=43,frame_count=864,frames=[{k:v for k,v in f.items() if k!='qa'} for f in frames],parent_seal=plan['binding']['inputs']['parent_seal'],
        heads=plan['binding']['heads'],delta=plan['delta'],placement='P1-final2',qa=artifact(root/QA),physical_scale_approved=True,
        source_time_raw_geometry_area_heads_delta_unchanged=True,domain_revision='pooled_sand_floor_v1',labels_changed=False,mass_eligible=False,bottom_mass_allowed=False,
        blind_hold_runs=[],training_performed=False,inference_performed=False,automatic_activation=False,scientific_acceptance=False)
    atomic_json(root/SEAL,seal)
    from .register_ffac_p1_final2_studio_candidate import register
    registration=register(root)
    from darsia.fluidflower.phase_ffac_p1_final2_display import import_parent_displays
    display=import_parent_displays(root,workers=12,resume=True,publish=True)
    return dict(seal=artifact(root/SEAL),registration=artifact(registration),qa=artifact(root/QA),display=str(display))


def build(root,plan,*,workers=8,resume=False,indices=None):
    wall,cpu=time.perf_counter(),time.process_time();root=Path(root);completed=[];contexts={}
    cv2.setNumThreads(1)
    sources=[s for s in plan['source_inventory'] if indices is None or (s['run'],s['frame_index']) in indices]
    runs=sorted({s['run'] for s in sources})
    for run in runs:
        context=prepare_run_context(root,plan,run);contexts[run]=context
        selected=[s for s in sources if s['run']==run]
        with ThreadPoolExecutor(max_workers=workers) as readers,ThreadPoolExecutor(max_workers=workers) as writers:
            pending=[];outstanding=[];iterator=iter(selected)
            for _ in range(workers):
                source=next(iterator,None)
                if source:pending.append((source,readers.submit(load_frame,root,plan,source,context)))
            while pending:
                source,future=pending.pop(0);path=root/'rig_camera_frames'/run/f"frame_{source['frame_index']:03d}.json"
                loaded=future.result()
                if path.exists():
                    saved=read_json(path)
                    if not resume or saved.get('contract')!=CONTRACT or saved['binding']['pipeline_sha256']!=plan['pipeline_sha256'] or saved['binding']['source']!=loaded['p1']['binding']['source'] or not saved['qa']['all_pass']:
                        raise ValueError('Resume receipt changed')
                    for asset in (saved['labels_map'],*[saved[k] for k in MASK_DIRECTORIES.values()],*saved['maps'].values(),
                            *saved['previous_labels'].values(),*saved['removed_display_layers'].values(),*saved['fill_reason_masks'].values(),
                            saved['rig_domain_revision']['classes'],saved['rig_domain_revision']['area_map']):bound_path(asset)
                    completed.append(dict(run=run,frame_index=source['frame_index'],receipt=artifact(path),qa=saved['qa']))
                else:
                    result=compute(loaded,context,plan['backend'])
                    if indices is not None:
                        cpu_result=compute(loaded,context,'cpu')
                        if not np.array_equal(result.labels,cpu_result.labels) or not all(np.array_equal(result.step_masks[k],cpu_result.step_masks[k]) for k in 'ABCD'):
                            raise ValueError('CUDA/CPU outputs differ')
                        result.qa['probe_cuda_cpu_identical']=True
                    outstanding.append(writers.submit(save_frame,root,plan,loaded,result))
                    if len(outstanding)>=workers:completed.append(outstanding.pop(0).result())
                source=next(iterator,None)
                if source:pending.append((source,readers.submit(load_frame,root,plan,source,context)))
            completed.extend(f.result() for f in outstanding)
        contexts.pop(run);del context
        progress=dict(stage='fixed_postprocessing',done=len(completed),total=len(sources),backend=plan['backend'],elapsed_seconds=time.perf_counter()-wall,last_run=run)
        atomic_json(root/'progress'/f'snapshot_{time.time_ns()}.json',progress);print(json.dumps(progress),flush=True)
    completed.sort(key=lambda f:(f['run'],f['frame_index']))
    if indices is not None:return completed
    result=seal_candidate(root,plan,completed)
    atomic_json(root/'completion.json',dict(contract=CONTRACT,complete=True,**result,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,backend=plan['backend']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output','spec','gate','zones'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--backend',choices=('auto','cuda','cpu'),default='auto')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--probe',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=24:raise ValueError('Workers must be1â€“24')
    with JobLease(args.output.parent/(args.output.name+'-operation'),SPEC_SHA256):
        plan=prepare_candidate(args.parent,args.output,args.spec,args.gate,args.zones,backend=args.backend,resume=args.resume)
        indices={('ac28',13),('ac52',6),('ac17',17)} if args.probe else None
        result=build(args.output,plan,workers=args.workers,resume=args.resume,indices=indices)
        print(json.dumps(dict(complete=not args.probe,result=result),default=str),flush=True)


if __name__=='__main__':main()
