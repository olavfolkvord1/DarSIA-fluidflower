"""Train portable variant-B completion on freshly fitted registered Studio runs.

Original annotations remain immutable. Only spatially incompatible dense targets
are masked from this stage's loss, and their exact pixels are recorded for review.
Training uses the same global patch locations and Hann weights as inference.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
REPO = series_root("ffac")
None
import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower import phase_runtime
from darsia.fluidflower import phase_aqueous_context_completion as completion
from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
from darsia.fluidflower.phase_training import trainer as base_trainer
from darsia.fluidflower.phase_training.logistic import regional_presence_target
from darsia.fluidflower.project import sha256_file
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window
from darsia.fluidflower.tools.phase_training_review_comparison import decode_phase, phase_path
from darsia.fluidflower.tools.train_phase_model_further import native_source
from darsia.fluidflower.tools.train_phase_weak_completion import aqueous_loss
from darsia.fluidflower.tools.phase_spatial_context_pilot import _mil, _weighted_mean

CAMPAIGN = local_cache_root() / 'phase_diagnostics/phase_retrain_b_20260914'
SEED = 20260914


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def crop(path, box):
    x, y, w, h = map(int, box)
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').crop((x,y,x+w,y+h)))


def phase_target(record):
    # This head predicts aqueous phase alone. Excluding all CO2, or declaring
    # pure gas, also excludes aqueous; excluding gas alone does not.
    if record.phase_class in ('gas', 'co2_absent_region'):
        return 0
    return regional_presence_target(record.phase_class, 'aqueous')


def target_loss(q, weights, name):
    # Extra production classes that the small pilot did not contain.
    if name in ('aqueous_present_gas_absent_region', 'mixed_gas_aqueous_water'):
        return .35*_mil(q, weights)
    if name == 'current_experiment_co2_artifact':
        return _weighted_mean(-(1-q).clamp_min(1e-7).log(), weights)
    return aqueous_loss(q, weights, name)


def archive_base_validation(model, *, cache_root=None):
    model = Path(model).resolve()
    allowed = (local_cache_root() if cache_root is None else Path(cache_root)).resolve() / 'studio-continuations'
    if not model.is_relative_to(allowed) or model == allowed:
        raise ValueError('Only newly created local continuation bundles may be staged')
    source, archive = model/'validation', model/'base_validation'
    if archive.is_dir():
        return archive
    if not source.is_dir() or not (source/'validation_binding.json').is_file():
        raise ValueError('Fresh base validation is not ready')
    source.rename(archive)
    write(model/'base_validation_archive.json', dict(original=str(source), archive=str(archive),
        files_unchanged=True, reason='Preserve original base maps while materializing the final completion runtime'))
    return archive


def relocate(value, old, new):
    if isinstance(value, dict):
        return {k: relocate(v,old,new) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v,old,new) for v in value]
    if isinstance(value,str) and (value == str(old) or value.startswith(str(old)+'\\') or value.startswith(str(old)+'/')):
        return str(new)+value[len(str(old)):]
    return value


def load_regions(model):
    request=read(model/'training_request.json')
    pairs=[]
    for bundle in request['bundles']:
        pair=[]
        for role in ('decisions','combined_manifest'):
            record=bundle[role];path=Path(record['snapshot_path'])
            if sha256_file(path)!=record['sha256']:
                raise ValueError('Frozen training evidence changed')
            pair.append(path)
        pairs.append(base_trainer.ReviewBundle(*pair))
    return base_trainer._load_sources_and_regions(pairs)[1]


def conflicting_pixels(records, native, domain):
    positive=np.zeros_like(domain);negative=np.zeros_like(domain)
    for region in records:
        if region.phase_class != 'aqueous' and phase_target(region) != 0:
            continue
        bounds,mask=annotation_window(region.payload,canvas_wh=native['canvas_wh'],content_box=native['content_box_xywh'])
        x0,y0,x1,y1=bounds
        target=positive if region.phase_class=='aqueous' else negative
        target[y0:y1,x0:x1] |= mask & domain[y0:y1,x0:x1]
    return positive & negative


def prepare(series, model, output, *, cache_root=None, predictor_factory=None):
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import CpuFeaturePredictor
    model, output = Path(model).resolve(), Path(output).resolve()
    archive=archive_base_validation(model, cache_root=cache_root)
    manifest=read(archive/'postflight_manifest.json')
    jobs=relocate(manifest['jobs'],model/'validation',archive)
    regions=load_regions(model)
    registered={j['run'] for j in jobs}
    excluded=[r for r in regions if r.run not in registered]
    profile=phase_runtime.load_physics_profile(model)
    if completion.PROFILE_KEY in profile:
        raise ValueError('Prepare completion from the unwrapped fresh base only')
    optical=HierarchicalPhaseModel.load(model,adapter='generic')
    entries=[]
    output.mkdir(parents=True,exist_ok=True)
    if predictor_factory is not None and hasattr(predictor_factory,'freeze_sources'):
        predictor_factory.freeze_sources(output)
    for job in jobs:
        run=job['run'];source,native_path,native=native_source(job)
        records=[r for r in regions if r.run==run]
        selected=sorted({r.frame_index for r in records if phase_target(r) is not None})
        context,_=phase_runtime.load_run_context(model,profile,series,run,native_manifest_sha256=sha256_file(native_path))
        spacing=(abs(float(context.y_m[1]-context.y_m[0])),abs(float(context.x_m[1]-context.x_m[0])))
        box=native['content_box_xywh']
        baseline_path=source/'native_references'/run/'pre_injection.jpg'
        baseline=crop(baseline_path,box)
        # Base materialization may still own most of the GPU for another series.
        # This is the reference feature path, with identical model/input semantics.
        predictor=(CpuFeaturePredictor(optical,baseline) if predictor_factory is None
                   else predictor_factory(optical,baseline))
        timeline=sorted(native['frame_sources'],key=lambda r:int(r['frame_index']))
        excluded_reference=set(job.get('independent_reference_frames') or [])
        for index in selected:
            frame_records=[r for r in records if r.frame_index==index and phase_target(r) is not None]
            key=f'{series}_{run}_{index:03d}';folder=output/'prepared'/key
            receipt=folder/'receipt.json'
            if receipt.is_file():
                entry=read(receipt)
                if sha256_file(Path(entry['array_path']))!=entry['array_sha256']:
                    raise ValueError('Prepared completion data changed')
                if (entry.get('regions') != [asdict(r) for r in frame_records]
                        or entry.get('preparation_format') != 'registered-aqueous-completion-inputs-v2'):
                    raise ValueError('Prepared completion evidence or geometry contract changed; use a new output directory')
                entries.append(entry);continue
            position=next(i for i,r in enumerate(timeline) if int(r['frame_index'])==index)
            earlier=[r for r in timeline[:position] if int(r['frame_index']) not in excluded_reference]
            previous_index=int(earlier[-1]['frame_index']) if earlier else None
            current_path=source/'native_frames'/run/f'frame_{index:03d}.jpg'
            previous_path=source/'native_frames'/run/f'frame_{previous_index:03d}.jpg' if previous_index is not None else None
            current=crop(current_path,box)
            previous=crop(previous_path,box) if previous_path else None
            raw=predictor.predict(current,previous,current_key=index,previous_key=previous_index)['co2_presence_probability']
            labels_path=phase_path(job,index)
            labels=decode_phase(labels_path,box,{})
            x,domain=completion.prepare_inputs(current,baseline,raw,labels,context.domain,spacing)
            conflict=conflicting_pixels(frame_records,native,context.domain)
            folder.mkdir(parents=True,exist_ok=True)
            array=folder/'inputs.npz'
            # All stored floats retain the shared runtime representation exactly.
            np.savez_compressed(array,inputs=x,domain=domain)
            np.savez_compressed(folder/'conflicts.npz',mask=conflict,domain=context.domain)
            entry=dict(frame_id=key,series=series,run=run,frame_index=index,elapsed_h=float(timeline[position]['elapsed_h']),
                preparation_format='registered-aqueous-completion-inputs-v2',
                canvas_wh=native['canvas_wh'],content_box_xywh=box,spacing_m=spacing,native_shape=list(context.domain.shape),
                regions=[asdict(r) for r in frame_records],array_path=str(array),array_sha256=sha256_file(array),
                conflict_path=str(folder/'conflicts.npz'),conflicting_native_pixels=int(conflict.sum()),
                current_path=str(current_path),baseline_path=str(baseline_path),phase_path=str(labels_path),
                source_hashes={str(p):sha256_file(p) for p in [current_path,baseline_path,labels_path,native_path]},
                prediction_backend=(dict(backend='cpu-reference-features') if predictor_factory is None
                                    else predictor_factory.provenance),
                prediction_checks=(predictor.provenance_record() if hasattr(predictor,'provenance_record') else None),
                model=str(model),base_runtime=phase_runtime.physics_runtime_identity(profile))
            write(receipt,entry);entries.append(entry)
            write(output/'preparation.json',dict(complete=False,frames=entries))
            print(json.dumps(dict(series=series,prepared=key,regions=len(frame_records),conflict_pixels=int(conflict.sum()))),flush=True)
            del x,current,previous,raw,labels,conflict
        if hasattr(predictor,'close'):predictor.close()
        del predictor,baseline,context
        gc.collect()
    report=dict(complete=True,series=series,frames=entries,base_training_regions=len(regions),
        completion_supervised_regions=sum(len(e['regions']) for e in entries),
        excluded_no_registered_physics=[dict(region_id=r.region_id,run=r.run,phase_class=r.phase_class) for r in excluded],
        exclusion_reason='Historical optical training evidence has no matching current registered full-frame physical context. It remains in the optical fit.',
        original_annotations_modified=False,base_validation_archive=str(archive))
    write(output/'preparation.json',report)
    return report


def _axis_area_weights(start, stop, native_size, working_size, first, last):
    """Sparse overlap of native pixel edges with the fixed working-grid edges."""
    from scipy.sparse import csr_matrix
    scale = native_size / working_size
    rows, columns, weights = [], [], []
    for j in range(first, last):
        left, right = j * scale, (j+1) * scale
        for i in range(max(start, int(np.floor(left))), min(stop, int(np.ceil(right)))):
            overlap = min(i+1, right) - max(i, left)
            if overlap > 0:
                rows.append(j-first); columns.append(i-start); weights.append(overlap/scale)
    return csr_matrix((np.asarray(weights, np.float32), (rows, columns)),
                      shape=(last-first, stop-start), dtype=np.float32)


def pool_annotation(native_mask, bounds, native_shape, working_shape):
    """Exact global area pooling without allocating a full native float image."""
    x0,y0,x1,y1 = bounds
    nh,nw = native_shape; h,w = working_shape
    ax=max(0,int(np.floor(x0*w/nw)));bx=min(w,max(ax+1,int(np.ceil(x1*w/nw))))
    ay=max(0,int(np.floor(y0*h/nh)));by=min(h,max(ay+1,int(np.ceil(y1*h/nh))))
    wx = _axis_area_weights(x0,x1,nw,w,ax,bx)
    wy = _axis_area_weights(y0,y1,nh,h,ay,by)
    mask = wx.dot(wy.dot(np.asarray(native_mask,np.float32)).T).T
    return np.asarray(mask,np.float32), (ax,ay,bx,by)


def make_global_bag(entry,record,shape,domain,conflict,native_domain=None):
    bounds,native_mask=annotation_window(record['payload'],canvas_wh=entry['canvas_wh'],content_box=entry['content_box_xywh'])
    x0,y0,x1,y1=bounds
    native_mask &= ~conflict[y0:y1,x0:x1]
    if native_domain is not None:
        native_mask &= native_domain[y0:y1,x0:x1]
    h,w=shape
    mask,(ax,ay,bx,by)=pool_annotation(native_mask,bounds,entry['native_shape'],shape)
    if native_domain is None:
        mask*=domain[ay:by,ax:bx]
    locations=[(y,x) for y,x in completion.tile_locations(h,w)
               if y<by and y+completion.PATCH>ay and x<bx and x+completion.PATCH>ax]
    window=completion.tile_window();mass=np.zeros_like(mask)
    for y,x in locations:
        iy0,iy1=max(ay,y),min(by,y+completion.PATCH);ix0,ix1=max(ax,x),min(bx,x+completion.PATCH)
        mass[iy0-ay:iy1-ay,ix0-ax:ix1-ax]+=window[iy0-y:iy1-y,ix0-x:ix1-x]
    tiles=[]
    for y,x in locations:
        iy0,iy1=max(ay,y),min(by,y+completion.PATCH);ix0,ix1=max(ax,x),min(bx,x+completion.PATCH)
        target=np.zeros((completion.PATCH,completion.PATCH),np.float32)
        region=np.s_[iy0-ay:iy1-ay,ix0-ax:ix1-ax];part=np.s_[iy0-y:iy1-y,ix0-x:ix1-x]
        target[part]=mask[region]*window[part]/np.maximum(mass[region],1e-8)
        if target.sum()>0:tiles.append((y,x,target))
    if not np.isclose(sum(t[2].sum() for t in tiles),mask.sum(),rtol=2e-5,atol=1e-5):
        raise ValueError('Global tile weights do not preserve complete annotation mass')
    return dict(frame_id=entry['frame_id'],record=record,tiles=tiles,mask=mask,
                mass=float(mask.sum()),bounds=[ax,ay,bx,by])


def blend_bag_probability(probabilities, bag):
    """Reconstruct final Hann-averaged q BEFORE applying nonlinear supervision."""
    import torch
    ax,ay,bx,by=bag['bounds']
    numerator=probabilities.new_zeros((by-ay,bx-ax))
    for k,(y,x,weighted_mask) in enumerate(bag['tiles']):
        iy0,iy1=max(ay,y),min(by,y+completion.PATCH)
        ix0,ix1=max(ax,x),min(bx,x+completion.PATCH)
        region=np.s_[iy0-ay:iy1-ay,ix0-ax:ix1-ax]
        part=np.s_[iy0-y:iy1-y,ix0-x:ix1-x]
        weight=torch.as_tensor(weighted_mask[part],dtype=probabilities.dtype,device=probabilities.device)
        numerator[region]=numerator[region]+probabilities[k][part]*weight
    mask=torch.as_tensor(bag['mask'],dtype=probabilities.dtype,device=probabilities.device)
    return numerator/mask.clamp_min(1e-8),mask


def blended_target_loss(probabilities,bag):
    q,mask=blend_bag_probability(probabilities,bag)
    return target_loss(q,mask,bag['record']['phase_class'])


def training_device(torch,backend):
    if backend not in ('auto','cpu','cuda'):
        raise ValueError('Completion backend must be auto, cpu, or cuda')
    if backend=='cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if backend=='cuda':
        raise RuntimeError('CUDA completion training requested but unavailable')
    return torch.device('cpu')


def fit(series,model_root,output,epochs=16,*,backend='auto'):
    import torch
    model_root,output=Path(model_root).resolve(),Path(output).resolve()
    if epochs<1:raise ValueError('At least one complete training epoch is required')
    device=training_device(torch,backend)
    torch.set_num_threads(4);torch.manual_seed(SEED)
    if device.type=='cuda':torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    prepared=read(output/'preparation.json')
    if not prepared['complete']:raise ValueError('Preparation incomplete')
    if (output/'aqueous_completion.pt').exists():raise FileExistsError('Preserve completed fit')
    plan=dict(series=series,epochs=epochs,seed=SEED,learning_rate=.001,prior_weight=.02,backend=str(device),
        inputs='VariantB shared runtime, 1mm grid, registered current/baseline+3mm difference',
        spatial_shrinkage_loss=False,patch_context='Same global128/96 locations and normalized Hann weights as full-frame inference; reconstruct blended probability before dense/MIL losses',
        annotation_pooling='Exact native-pixel overlap on the global working grid, with native domain and conflicting pixels excluded before pooling',
        incompatible_dense_targets='Exact native pure-aqueous and exclusion intersections are masked from BOTH objectives; original annotation files unchanged',
        supervision='Whole-region MIL presence; dense targets only for explicit pure/absence labels',
        fit_scope='All available registered reviewed runs of this series, no independent holdout claim',
        old_map_role='Input/residual baseline/weak tether, never a replacement human label',
        base_model=str(model_root),preparation_sha256=sha256_file(output/'preparation.json'),
        source_sha256=sha256_file(Path(__file__)),runtime_source_sha256=sha256_file(Path(completion.__file__)))
    write(output/'protocol.json',plan)
    arrays={};bags=[];masked_empty=[]
    for entry in prepared['frames']:
        if sha256_file(Path(entry['array_path']))!=entry['array_sha256']:
            raise ValueError('Prepared completion input hash changed before fit')
        with np.load(entry['array_path']) as a:
            arrays[entry['frame_id']]=(torch.from_numpy(a['inputs'].copy()),torch.from_numpy(a['domain'].copy()))
            domain=a['domain'];shape=domain.shape
        with np.load(entry['conflict_path']) as a:conflict,native_domain=a['mask'],a['domain']
        for record in entry['regions']:
            bag=make_global_bag(entry,record,shape,domain,conflict,native_domain)
            if not bag['tiles']:masked_empty.append(record['region_id'])
            else:bags.append(bag)
    if not bags:raise ValueError('No usable aqueous completion supervision remains')
    strata=Counter((b['record']['run'],b['record']['phase_class']) for b in bags)
    classes=Counter(k[0] for k in strata)
    for b in bags:
        key=(b['record']['run'],b['record']['phase_class'])
        b['weight']=len(bags)/(len(classes)*classes[key[0]]*strata[key])
    model=completion.build_network(torch).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(SEED);start=time.perf_counter();history=[];steps=0
    for epoch in range(epochs):
        losses=[];model.train()
        for idx in rng.permutation(len(bags)):
            b=bags[int(idx)];x_cpu,domain_cpu=arrays[b['frame_id']]
            xx=[];dd=[];mm=[]
            for y,x,mask in b['tiles']:
                piece=x_cpu[:,y:y+128,x:x+128];dom=domain_cpu[y:y+128,x:x+128]
                ph,pw=128-piece.shape[-2],128-piece.shape[-1]
                xx.append(torch.nn.functional.pad(piece,(0,pw,0,ph),mode='replicate'))
                dd.append(torch.nn.functional.pad(dom,(0,pw,0,ph)))
                mm.append(torch.from_numpy(mask))
            tx=torch.stack(xx);td=torch.stack(dd);tm=torch.stack(mm)
            optimizer.zero_grad(set_to_none=True)
            if len(xx)<=8:
                tx,td,tm=tx.to(device),td.to(device),tm.to(device)
                q=completion.predict_probability(model,tx)
                supervised=blended_target_loss(q,b)*b['weight']
                tether=_weighted_mean((q-completion.base_probability(tx)).square(),td)
                loss=supervised+.02*tether
                loss.backward()
            else:
                # Exact whole-bag supervision with bounded activation memory.
                # GroupNorm has no batch running state/dropout; the second pass
                # recomputes the same q and applies the exact dL/dq before one step.
                with torch.no_grad():
                    values=[completion.predict_probability(model,tx[i:i+8].to(device)) for i in range(0,len(xx),8)]
                q=torch.cat(values).requires_grad_(True)
                td,tm=td.to(device),tm.to(device)
                base=completion.base_probability(tx).to(device)
                supervised=blended_target_loss(q,b)*b['weight']
                tether=_weighted_mean((q-base).square(),td)
                loss=supervised+.02*tether
                gradient=torch.autograd.grad(loss,q)[0].detach()
                for i in range(0,len(xx),8):
                    prediction=completion.predict_probability(model,tx[i:i+8].to(device))
                    (prediction*gradient[i:i+8]).sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()));steps+=1
        row=dict(epoch=epoch+1,steps=steps,objective=float(np.mean(losses)),seconds=time.perf_counter()-start)
        history.append(row);write(output/'fit_progress.json',dict(series=series,epochs_completed=epoch+1,epochs=epochs,**row))
        print(json.dumps(dict(series=series,**row)),flush=True)
    checkpoint=output/'aqueous_completion.pt'
    torch.save(dict(state_dict=model.state_dict(),contextual=True,series=series,epochs=epochs,seed=SEED),checkpoint)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in [Path(__file__),Path(completion.__file__)]:shutil.copyfile(source,snapshot/source.name)
    preparation_snapshot=output/'preparation_code_snapshot'
    if preparation_snapshot.is_dir():
        for source in preparation_snapshot.iterdir():shutil.copyfile(source,snapshot/source.name)
    report=dict(complete=True,series=series,epochs=epochs,steps=steps,backend=str(device),training_seconds=time.perf_counter()-start,
        supervised_regions=len(bags),fully_masked_conflicting_regions=masked_empty,
        conflicting_native_pixels=sum(e['conflicting_native_pixels'] for e in prepared['frames']),
        history=history,checkpoint_sha256=sha256_file(checkpoint),independent_generalization=False,
        original_annotations_modified=False,optimization_convergence_certified=False)
    write(output/'fit_report.json',report)
    if sha256_file(Path(completion.__file__))!=plan['runtime_source_sha256']:
        raise ValueError('Runtime changed during fit; inspect recorded provenance before attachment')
    return checkpoint


def pending_decision(model):
    """Validate the integrated method to restore after fresh B weights exist."""
    from darsia.fluidflower import phase_decision as core, phase_decision_runtime as runtime
    model=Path(model).resolve();receipt_path=model/'completion_retraining.json'
    if not receipt_path.is_file():return None
    record=read(receipt_path).get('restore_unified_decision')
    if record is None:return None
    source=(model/'inherited_completion_before_retraining'/runtime.PROFILE_FILE).resolve()
    if not source.is_relative_to(model) or sha256_file(source)!=record['profile_sha256']:
        raise ValueError('Archived unified decision profile changed')
    specification=read(source)
    from darsia.fluidflower.phase_decision_versions import implementation
    selected=implementation(specification);core=selected.core
    if specification!=record['specification']:
        raise ValueError('Archived unified decision settings changed')
    if core.identity(core.Settings(**specification['settings']))!=record['decision_identity']:
        raise ValueError('Unified decision source changed during retraining')
    if (model/runtime.PROFILE_FILE).exists():
        raise FileExistsError('Do not stack or overwrite an attached unified decision')
    return dict(source=source,record=record)


def attach(model,output):
    model,output=Path(model).resolve(),Path(output).resolve()
    checkpoint=output/'aqueous_completion.pt'
    report,protocol=read(output/'fit_report.json'),read(output/'protocol.json')
    if not report['complete']:raise ValueError('Incomplete completion fit')
    if sha256_file(checkpoint)!=report['checkpoint_sha256']:raise ValueError('Completion checkpoint changed after fitting')
    if sha256_file(Path(completion.__file__))!=protocol['runtime_source_sha256']:
        raise ValueError('Completion inference source differs from the fitted contract')
    pending=pending_decision(model)
    target=model/'aqueous_completion.pt'
    if target.exists():raise FileExistsError('Do not overwrite attached completion weights')
    shutil.copyfile(checkpoint,target)
    write(model/completion.PROFILE_FILE,dict(format=completion.VERSION,checkpoint=target.name,checkpoint_sha256=sha256_file(target)))
    snapshot=model/'completion_code_snapshot';snapshot.mkdir()
    for path in (output/'source_snapshot').iterdir():shutil.copyfile(path,snapshot/path.name)
    shutil.copyfile(output/'fit_report.json',snapshot/'fit_report.json')
    shutil.copyfile(output/'protocol.json',snapshot/'training_protocol.json')
    if pending is not None:
        from darsia.fluidflower import phase_decision_runtime as runtime
        shutil.copy2(pending['source'],model/runtime.PROFILE_FILE)
        old_snapshot=model/'inherited_completion_before_retraining/decision_code_snapshot'
        if old_snapshot.is_dir():shutil.copytree(old_snapshot,model/'decision_code_snapshot')
    inheritance=read(model/'physics_inheritance.json')
    inheritance['base_runtime_before_completion']=inheritance['physics_runtime']
    inheritance['physics_runtime']=phase_runtime.physics_runtime_identity(phase_runtime.load_physics_profile(model))
    inheritance['completion']=dict(format=completion.VERSION,checkpoint_sha256=sha256_file(target),
        training_report=str(output/'fit_report.json'),original_physics_history_unchanged=True)
    if pending is not None:
        inheritance['unified_decision_restored_after_completion']=pending['record']
    write(model/'physics_inheritance.json',inheritance)
    write(output/'attached.json',dict(model=str(model),checkpoint_sha256=sha256_file(target),physics_runtime=inheritance['physics_runtime']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--series',required=True,choices=['ffac','ffum','ffbm'])
    p.add_argument('--action',choices=['prepare','fit','attach','all'],default='all');p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--backend',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--cache-root',type=Path,default=local_cache_root())
    args=p.parse_args();cv2.setNumThreads(4)
    state=read(CAMPAIGN/'base_training'/args.series/'state.json')
    if state['stage']!='ready':raise ValueError('Base stage is not ready: '+state['stage'])
    model=Path(state['output_root']);output=CAMPAIGN/'completion_training'/args.series
    if args.action in ['prepare','all'] and (not (output/'preparation.json').is_file() or not read(output/'preparation.json').get('complete')):prepare(args.series,model,output,cache_root=args.cache_root)
    if args.action in ['fit','all']:fit(args.series,model,output,args.epochs,backend=args.backend)
    if args.action in ['attach','all']:attach(model,output)


if __name__=='__main__':main()
