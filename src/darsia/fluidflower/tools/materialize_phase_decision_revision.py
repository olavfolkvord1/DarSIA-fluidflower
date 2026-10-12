"""Recompute final maps from immutable causal observations and completion evidence.

Every observation is checked against the original native map. Numerical evidence
is reused, not refitted or recast as fresh optical inference. The shared decision
kernel is identical to live inference. Completion/final maps never update the
inherited provider's history. Publication and scientific approval are separate.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import copy
import hashlib
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from darsia.fluidflower import phase_decision as core,phase_runtime
from darsia.fluidflower.phase_validation import _model_identity,_write_phase_maps
from darsia.fluidflower.project import utc_now
from darsia.fluidflower.tools.phase_mass_overnight_data import read,write,sha,decode_labels
from darsia.fluidflower.tools.prepare_phase_decision_revision import verify
from darsia.fluidflower.tools.train_phase_model_further import native_source


def rgb(path,box):
    x,y,w,h=box
    with Image.open(path) as im:return np.asarray(im.convert('RGB'))[y:y+h,x:x+w].copy()


def decision_implementation(settings):
    from darsia.fluidflower import phase_decision_age
    return phase_decision_age if isinstance(settings,phase_decision_age.Settings) else core


def exposure_history(context,settings):
    implementation=decision_implementation(settings)
    if implementation is core:return None
    spacing=(abs(float(context.y_m[1]-context.y_m[0])),abs(float(context.x_m[1]-context.x_m[0])))
    return implementation.ExposureHistory(context.domain.shape,spacing,settings)


def cached_decision(current,baseline,context,time_h,diagnostics,original_map,box,settings,provenance,*,history=None):
    with np.load(diagnostics) as z:
        base=np.where(z['completion_before_co2'],np.where(z['completion_before_gas'],2,1),0).astype(np.uint8)
        added=z['completion_added_aqueous']
        independent=z['spatial_only_presence'] if 'spatial_only_presence' in z else None
    original=decode_labels(original_map,box)
    if not np.array_equal(original>0,(base>0)|added) or not np.array_equal(original==2,base==2):
        raise ValueError('Frozen completion evidence does not reconstruct its original phase map')
    spacing=(abs(float(context.y_m[1]-context.y_m[0])),abs(float(context.x_m[1]-context.x_m[0])))
    implementation=decision_implementation(settings);extra={}
    if implementation is not core:
        if history is None:raise ValueError('Local-age replay requires its chronological observation history')
        extra['first_observed_h']=history.observe(base,context.domain,time_h,pre_injection=time_h<0)
    result=implementation.decide(implementation.Evidence(current,baseline,base,added,(base>0)&context.domain,
        context.domain,context.facies,spacing,float(time_h),float(time_h)<0,provenance,
        independent_spatial_observation=independent,**extra),settings)
    if np.any(result.labels[~context.domain]):raise ValueError('Phase escaped the exact registered domain')
    return result


def fields(labels):
    return dict(co2_presence_probability=(labels>0).astype(np.float32),
        gas_probability_conditional_on_co2=(labels==2).astype(np.float32))


def one_run(revision,run):
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):return _one_run(Path(revision),run)


def _one_run(revision,run):
    record=read(revision/'revision_provenance.json');parent=Path(record['parent'])
    old=read(parent/'validation/postflight_manifest.json');job=next(j for j in old['jobs'] if j['run']==run)
    source,native_path,native=native_source(job);box=native['content_box_xywh']
    profile=phase_runtime.load_physics_profile(revision);method=phase_runtime.physics_runtime_identity(profile)
    if method!=record['physics_runtime']:raise ValueError('Prepared runtime changed')
    from darsia.fluidflower.phase_decision_versions import implementation
    settings=implementation(profile['unified_phase_decision']).core.Settings(**profile['unified_phase_decision']['settings'])
    identity=_model_identity(revision);fingerprint=hashlib.sha256(f'{identity[1]}:{method}'.encode('ascii')).hexdigest()
    output=revision/'validation';receipt_path=output/'decision_replay'/f'{run}.json'
    if receipt_path.exists():
        saved=read(receipt_path)
        if saved['method']!=method:raise ValueError('Existing replay is from another method')
        for row in saved['frames']:
            for path,digest in row['assets'].items():
                if sha(path)!=digest:raise ValueError('Completed replay asset changed')
        return saved
    context,_=phase_runtime.load_run_context(revision,profile,job['dataset'],run,native_manifest_sha256=sha(native_path))
    history=exposure_history(context,settings)
    baseline_path=source/'native_references'/run/'pre_injection.jpg';baseline=rgb(baseline_path,box)
    rows=[];previous=-float('inf')
    original_folder=Path(job['phase_map_output_dir'])/'native_phase_maps'/run/f"A_{job['params_sha256'][:16]}"
    output_folder=output/'review_assets/native_phase_maps'/run/f'A_{fingerprint[:16]}'
    for index,frame in enumerate(native['frame_sources']):
        elapsed=float(frame['elapsed_h'])
        if elapsed<=previous:raise ValueError('Causal observations must have increasing timestamps')
        previous=elapsed;start=time.monotonic()
        photo=source/'native_frames'/run/f'frame_{index:03d}.jpg'
        diagnostics=parent/'validation/presence_diagnostics'/run/f'frame_{index:03d}.npz'
        original=original_folder/f'frame_{index:03d}.png'
        hashes={str(p):sha(p) for p in (photo,baseline_path,diagnostics,original,native_path)}
        result=cached_decision(rgb(photo,box),baseline,context,elapsed,diagnostics,original,box,settings,
            dict(adapter='verified-frozen-causal-observations-v1',parent_model=str(parent),source_hashes=hashes,
                 completion_and_final_maps_never_update_provider_history=True),history=history)
        predictions=fields(result.labels)
        _write_phase_maps(output/'review_assets',run,fingerprint,index,native['canvas_wh'][::-1],box,predictions)
        target=output_folder/f'frame_{index:03d}.png'
        if not np.array_equal(decode_labels(target,box),result.labels):raise ValueError('Written phase map changes numerical labels')
        # Preserve the raw optical inspection layer byte-for-byte under its new bundle path.
        old_raw=Path(job['raw_phase_map_output_dir'])/'native_phase_maps'/run/f"A_{job['optical_model_sha256'][:16]}"/f'frame_{index:03d}.png'
        new_raw=output/'raw_optical_assets/native_phase_maps'/run/f'A_{identity[1][:16]}'/old_raw.name
        new_raw.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old_raw,new_raw)
        hashes[str(new_raw)]=sha(old_raw)
        if sha(new_raw)!=hashes[str(new_raw)]:raise ValueError('Copied raw optical map changed')
        for layer in ('','gas','aqueous','mixed'):
            path=output_folder/layer/target.name;hashes[str(path)]=sha(path)
        row=dict(frame_index=index,time_h=elapsed,assets=hashes,seconds=time.monotonic()-start,report=result.report)
        rows.append(row);write(output/'decision_replay'/f'{run}_progress.json',dict(run=run,completed=len(rows),total=len(native['frame_sources'])))
        print(f'{run} {index+1}/{len(native["frame_sources"])}: {row["seconds"]:.1f}s',flush=True)
    candidate=f'{identity[0]}__{job["dataset"]}__{run}__physics_{method[:12]}'
    new_job={**job,'candidate_id':candidate,'params_sha256':fingerprint,'optical_model_sha256':identity[1],
        'physics_runtime':method,'phase_map_output_dir':str(output/'review_assets'),
        'raw_phase_map_output_dir':str(output/'raw_optical_assets'),'presence_diagnostics_dir':None}
    result=dict(run=run,method=method,native_manifest_sha256=sha(native_path),frames=rows,job=new_job,
        completed_utc=utc_now(),frozen_provider_history=True,final_decision_recomputed_every_frame=True)
    write(receipt_path,result);return result


def materialize(revision,workers=2):
    revision=Path(revision);checked=verify(revision);parent=Path(checked['parent']);output=revision/'validation'
    old=read(parent/'validation/postflight_manifest.json');old_binding=read(parent/'validation/validation_binding.json')
    if old_binding.get('radiometry_profile') is not None:raise ValueError('A separate model radiometry profile needs an explicit evidence adapter')
    if not (output/'code_snapshot').exists():shutil.copytree(parent/'validation/code_snapshot',output/'code_snapshot')
    wrapper=old_binding['inference_provenance'].get('execution_wrapper')
    if wrapper:
        source=parent/'validation'/wrapper['wrapper_snapshot'];target=output/wrapper['wrapper_snapshot']
        if not target.resolve().is_relative_to(output.resolve()):raise ValueError('Reference wrapper escaped validation bundle')
        if sha(source)!=wrapper['wrapper_sha256']:raise ValueError('Inherited reference wrapper changed')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    snapshot=output/'decision_materializer_snapshot'/Path(__file__).name
    snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,snapshot)
    jobs=old['jobs'];results={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(one_run,str(revision),j['run']):j['run'] for j in jobs}
        for future in as_completed(pending):results[pending[future]]=future.result()
    new_jobs=[results[j['run']]['job'] for j in jobs];fingerprint=new_jobs[0]['params_sha256']
    provenance=copy.deepcopy(old_binding['inference_provenance'])
    provenance['phase_decision_evidence_replay']=dict(format='verified-frozen-causal-observations-v1',
        parent_binding=dict(path=str(parent/'validation/validation_binding.json'),sha256=sha(parent/'validation/validation_binding.json')),
        optical_inference_performed=False,decision_recomputed=True,provider_history_unchanged=True,
        materializer_sha256=sha(__file__),replay_receipts={run:sha(output/'decision_replay'/f'{run}.json') for run in results})
    common=dict(model_id=checked['model_id'],model_root=str(revision),model_sha256=checked['model_sha256'],
        pipeline_sha256=fingerprint,physics_runtime=checked['physics_runtime'],physics_profile=phase_runtime.load_physics_profile(revision),
        inference_provenance=provenance,campaign_id=f'phase-validation-{checked["model_id"]}')
    manifest={**old,**common,'jobs':new_jobs,'created_utc':utc_now()}
    for key,path in manifest.get('models',{}).items():
        for name,value in path.items():path[name]=str(revision/Path(value).relative_to(parent))
    binding={**old_binding,**common,'ffac_root':str(output),'ffum_root':str(output),'combined_root':str(output),
        'selected_candidates':{j['run']:j['candidate_id'] for j in new_jobs},'completed_utc':utc_now()}
    write(output/'postflight_manifest.json',manifest);write(output/'validation_binding.json',binding)
    write(output/'decision_materialization.json',dict(complete=True,frames=sum(len(r['frames']) for r in results.values()),
        runs=list(results),method=checked['physics_runtime'],independent_validation=False,publication_performed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();materialize(a.revision,a.workers)
