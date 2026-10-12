"""Frozen AC/UM development cases and read-only spatial-pilot evaluation.

No training, production map changes, or physical/history replay. Optional raw
optical materialization writes only beneath the isolated pilot campaign.
"""
from __future__ import annotations

from darsia.fluidflower.tools._paths import project_path, source_path, process_directory, series_root, series_path, tool_path, local_cache_root, package_root

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = series_root("ffac")
None
from darsia.fluidflower import phase_runtime
from darsia.fluidflower.phase_training.logistic import ReviewedRegion, regional_presence_target
from darsia.fluidflower.tools.compare_ac_um_review_update_20260914 import constraints
from darsia.fluidflower.tools.phase_coupled_constraint_comparison import annotation_window
from darsia.fluidflower.tools.phase_training_geometry_revision import _native, read
from darsia.fluidflower.tools.phase_training_review_comparison import _regions, decode_phase, fractions, phase_path
from darsia.fluidflower.tools.diagnose_ffum_continuity_20260914 import crop_uint8

CAMPAIGN = local_cache_root() / 'phase_diagnostics/phase_spatial_pilot_20260914'
RETRAIN = local_cache_root() / 'phase_diagnostics/ac_um_review_update_20260914'
AC44 = RETRAIN/'ac44_review/evaluations/phase-transfer-20260914T160906Z-cee1248e'
_VERIFIED = {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def verify(path, expected):
    path = Path(path)
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, expected)
    if _VERIFIED.get(str(path)) != signature:
        if sha(path) != expected:
            raise ValueError(f'Frozen pilot source changed: {path}')
        _VERIFIED[str(path)] = signature


def document(manifest):
    return read(manifest) if isinstance(manifest, (str, Path)) else manifest


def _record(region):
    return dict(region_id=region.region_id, run=region.run, dataset=region.dataset,
        frame_index=region.frame_index, elapsed_h=region.elapsed_h,
        phase_class=region.phase_class, confidence=region.confidence,
        source=region.source, reviewer=region.reviewer, payload=dict(region.payload))


def build_manifest(output=CAMPAIGN/'evidence_manifest.json'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Retain the existing frozen selection: {output}')
    selection = [('ffum', 'run18', i, 'late weak fronts and negative controls') for i in (18,19,20)]
    selection += [('ffum', 'run20', i, 'late weak fronts and positive controls') for i in (19,20,21)]
    selection += [('ffac','ac32',21,'late broad plume control'),
                  ('ffac','ac60',25,'late gas-absence tradeoff'),
                  ('ffac','ac61',21,'late grain and aqueous boundary controls'),
                  ('ffum','run18',6,'early reviewed aqueous-to-gas counterexample'),
                  ('ffac','ac61',8,'early gas-anchor retention counterexample'),
                  ('ffac','ac44',21,'unlabelled transfer visual check')]
    states = {s:read(RETRAIN/s/'state.json') for s in ('ffum','ffac')}
    evidence = {s:_regions(Path(states[s]['output_root']))[0] for s in states}
    frames = []
    for series, run, index, purpose in selection:
        model = AC44 if run == 'ac44' else Path(states[series]['output_root'])
        manifest_path = model/'validation/postflight_manifest.json'
        post = read(manifest_path)
        job = next(j for j in post['jobs'] if j['run'] == run)
        source, native_path = _native(job)
        native = read(native_path)
        timeline = sorted(native['frame_sources'], key=lambda x:int(x['frame_index']))
        position = next(i for i,f in enumerate(timeline) if int(f['frame_index']) == index)
        frame = timeline[position]
        excluded = set(job.get('independent_reference_frames') or [])
        prior = [f for f in timeline[:position] if int(f['frame_index']) not in excluded]
        previous_index = int(prior[-1]['frame_index']) if prior else None
        if prior and float(prior[-1]['elapsed_h']) >= float(frame['elapsed_h']):
            raise ValueError('Previous optical image must precede the selected frame')
        context, entry = phase_runtime.load_run_context(model, post['physics_profile'], series, run,
            native_manifest_sha256=sha(native_path))
        paths = dict(current_rgb=str(source/'native_frames'/run/f'frame_{index:03d}.jpg'),
            baseline_rgb=str(source/'native_references'/run/'pre_injection.jpg'),
            native_manifest=str(native_path), context=str(model/entry['path']),
            saved_labels=str(phase_path(job,index)), validation_manifest=str(manifest_path),
            validation_binding=str(model/'validation/validation_binding.json'),
            presence_diagnostics=str(Path(job['presence_diagnostics_dir'])/run/f'frame_{index:03d}.npz'))
        if previous_index is not None:
            paths['previous_rgb'] = str(source/'native_frames'/run/f'frame_{previous_index:03d}.jpg')
        for head in ('co2_presence','gas_within_co2'):
            paths[head+'_model'] = str(model/'models/generic'/f'{head}_signal.json')
        digest = {p:sha(p) for p in paths.values()}
        frame_id = f'{series}_{run}_{index:03d}'
        raw = dict(path=str(output.parent/'raw_optics'/frame_id/'raw_co2.npy'),
            receipt_path=str(output.parent/'raw_optics'/frame_id/'receipt.json'), status='missing',
            model_id=post['model_id'], adapter='generic')
        if series == 'ffum' and run in ('run18','run20') and index >= 18:
            audit = local_cache_root() / 'phase_diagnostics/fresh_um_trim_only_20260914'/(run+'_audit')
            receipt = read(audit/'report.json')
            if Path(receipt['model']).resolve() != model.resolve():
                raise ValueError('UM raw-optical cache names another model')
            item = next(f for f in receipt['frames'] if f['run']==run and int(f['frame'])==index)
            if item['previous_index'] != previous_index:
                raise ValueError('Cached raw optics used another preceding image')
            path = audit/run/f'frame_{index:03d}'/'raw_co2.npy'
            verify(path,item['raw_co2_sha256'])
            for role in ('current_rgb','baseline_rgb','native_manifest','previous_rgb'):
                if paths.get(role) and item['input_sha256'].get(paths[role]) != digest[paths[role]]:
                    raise ValueError(f'Cached raw optics used another {role}')
            for role in ('co2_presence_model','gas_within_co2_model'):
                if receipt['protected_source_hashes'].get(paths[role]) != digest[paths[role]]:
                    raise ValueError('Cached raw optics used another optical weight file')
            raw.update(path=str(path),receipt_path=str(audit/'report.json'),status='ready',
                sha256=item['raw_co2_sha256'],receipt_sha256=sha(audit/'report.json'),
                backend='CpuFeaturePredictor float32; cached without physical replay')
        members = [r for r in evidence[series] if r.run==run and r.frame_index==index]
        if run=='ac44' and members:
            raise ValueError('AC44 unexpectedly appears in current optical training evidence')
        frames.append(dict(frame_id=frame_id,series=series,run=run,frame_index=index,
            elapsed_h=float(frame['elapsed_h']),source_name=frame.get('source_name'),purpose=purpose,
            model_path=str(model),model_id=post['model_id'],model_sha256=job['optical_model_sha256'],
            physics_runtime=job['physics_runtime'],candidate_id=job['candidate_id'],
            whole_run_group=f'{series}/{run}',training_seen=bool(members) if run=='ac44' else True,
            evaluation_role='unlabelled_transfer_visual_only' if run=='ac44' else 'development_training_controls',
            independent_labeled_accuracy=False,previous_index=previous_index,
            canvas_wh=native['canvas_wh'],content_box_xywh=native['content_box_xywh'],
            spacing_m=[abs(float(context.y_m[1]-context.y_m[0])),abs(float(context.x_m[1]-context.x_m[0]))],
            paths=paths,source_hashes=digest,raw_co2=raw,regions=[_record(r) for r in members],
            causal_support_scope='Saved trusted_history diagnostic from the original chronological materialization; pilot history is not replayed.'))
        print(json.dumps(dict(frame_id=frame_id,regions=len(members),raw_status=raw['status'])),flush=True)
        del context
    result = dict(format='phase-spatial-pilot-frozen-evidence-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        selection_frozen=True,scope='Bounded development comparison on current models; no model promotion or independent labeled accuracy claim.',
        limitations=['Region presence is not a target area fraction; any-pixel repair and coverage are reported separately.',
            'AC44 is absent from these optical training bundles but has historical exposure; it is an unlabelled visual check, not globally unseen independent validation.',
            'The six current AC/early-control raw caches are initially missing. Existing old-model AC raw caches are deliberately excluded.',
            'Changes must be grouped by whole run; these selected frames cannot establish independent generalization.'],
        evidence_policy='All effective frozen current annotations on each selected frame, including audited revisions; source labels and coordinates unchanged.',
        frames=frames)
    write(output,result)
    return result


@lru_cache(maxsize=2)
def _shared(model_path,validation_manifest,native_manifest,baseline_path):
    post = read(validation_manifest)
    native = read(native_manifest)
    run = next(j['run'] for j in post['jobs'] if str(_native(j)[1])==native_manifest)
    job = next(j for j in post['jobs'] if j['run']==run)
    context,_ = phase_runtime.load_run_context(Path(model_path),post['physics_profile'],job['dataset'],run,
        native_manifest_sha256=sha(native_manifest))
    return context,crop_uint8(baseline_path,native['content_box_xywh'])


def load_frame(manifest,frame_id,*,include_raw=True):
    entry = next(f for f in document(manifest)['frames'] if f['frame_id']==frame_id)
    for path,digest in entry['source_hashes'].items():
        verify(path,digest)
    p=entry['paths'];box=entry['content_box_xywh']
    context,baseline=_shared(entry['model_path'],p['validation_manifest'],p['native_manifest'],p['baseline_rgb'])
    saved=decode_phase(p['saved_labels'],box,{})
    with np.load(p['presence_diagnostics'],allow_pickle=False) as archive:
        before=archive['boundary_before_co2'].astype(np.uint8)
        before[archive['boundary_before_gas']]=2
        causal=archive['trusted_history'].astype(np.float32) if 'trusted_history' in archive.files else None
    raw=None
    if include_raw and entry['raw_co2']['status']=='ready':
        verify(entry['raw_co2']['path'],entry['raw_co2']['sha256'])
        raw=np.load(entry['raw_co2']['path'],allow_pickle=False,mmap_mode='r')
        if raw.shape != context.domain.shape or not np.isfinite(raw).all():
            raise ValueError('Raw optical cache has invalid native geometry or values')
    return dict(series=entry['series'],run=entry['run'],frame_index=entry['frame_index'],elapsed_h=entry['elapsed_h'],
        frame_id=frame_id,current_rgb=crop_uint8(p['current_rgb'],box),baseline_rgb=baseline,
        previous_rgb=crop_uint8(p['previous_rgb'],box) if p.get('previous_rgb') else None,
        domain=context.domain,material=context.facies,context=context,spacing_m=tuple(entry['spacing_m']),
        saved_labels=saved,before_wrapper_labels=before,raw_co2=raw,causal_support=causal,
        model_path=entry['model_path'],regions=entry['regions'],entry=entry)


def materialize_raw(manifest=CAMPAIGN/'evidence_manifest.json',ids=None,threads=4):
    """Only raw generic optics on frozen images; no physical/history sequence."""
    from threadpoolctl import threadpool_limits
    from darsia.fluidflower.tools.phase_inference_benchmark_backends import CpuFeaturePredictor
    from darsia.fluidflower.phase_learning import HierarchicalPhaseModel
    path=Path(manifest);data=read(path);chosen=set(ids or [f['frame_id'] for f in data['frames'] if f['raw_co2']['status']=='missing'])
    cv2.setNumThreads(threads)
    predictor=None;key=None
    with threadpool_limits(limits=threads):
        for entry in data['frames']:
            if entry['frame_id'] not in chosen:continue
            if entry['raw_co2']['status']=='ready':
                verify(entry['raw_co2']['path'],entry['raw_co2']['sha256']);continue
            frame=load_frame(data,entry['frame_id'],include_raw=False)
            newkey=(entry['model_path'],entry['run'])
            if newkey!=key:
                predictor=CpuFeaturePredictor(HierarchicalPhaseModel.load(Path(entry['model_path']),adapter='generic'),frame['baseline_rgb']);key=newkey
            prediction=predictor.predict(frame['current_rgb'],frame['previous_rgb'],current_key=entry['frame_index'],previous_key=entry['previous_index'])
            destination=Path(entry['raw_co2']['path']).resolve()
            if not destination.is_relative_to(path.parent.resolve()):raise ValueError('New raw output escaped the pilot campaign')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():raise FileExistsError(destination)
            np.save(destination,prediction['co2_presence_probability'],allow_pickle=False)
            receipt=dict(model_id=entry['model_id'],model_path=entry['model_path'],frame_id=entry['frame_id'],
                adapter='generic',backend='CpuFeaturePredictor float32',physical_history_replayed=False,
                source_hashes=entry['source_hashes'],raw_co2_sha256=sha(destination))
            write(entry['raw_co2']['receipt_path'],receipt)
            entry['raw_co2'].update(status='ready',sha256=receipt['raw_co2_sha256'],receipt_sha256=sha(entry['raw_co2']['receipt_path']))
            write(path,data)
            print(json.dumps(dict(frame_id=entry['frame_id'],raw_ready=True)),flush=True)
            del frame,prediction
    return data


def evaluate_frame(frame,candidate_labels,candidate_probability=None):
    """Regional categorical controls plus magnitude; any-pixel repairs stay separate."""
    candidate=np.asarray(candidate_labels);domain=frame['domain'];before=frame['saved_labels']
    if candidate.shape!=domain.shape or not np.isin(candidate,[0,1,2]).all():raise ValueError('Candidate must be native categorical labels 0/1/2')
    if np.any(candidate[~domain]):raise ValueError('Candidate labels lie outside the physical domain')
    if candidate_probability is not None:
        probability=np.asarray(candidate_probability)
        if probability.shape!=domain.shape:raise ValueError('Probability geometry differs')
        if not np.isfinite(probability).all() or ((probability<0)|(probability>1)).any():
            raise ValueError('Candidate probability must be finite within [0,1]')
    rows=[];skipped=[];raw=frame.get('raw_co2');pixel_m2=float(np.prod(frame['spacing_m']))
    for saved in frame['regions']:
        region=ReviewedRegion(**saved)
        bounds,mask=annotation_window(region.payload,canvas_wh=frame['entry']['canvas_wh'],content_box=frame['entry']['content_box_xywh'])
        x0,y0,x1,y1=bounds;window=np.s_[y0:y1,x0:x1];mask &= domain[window]
        if not mask.any():skipped.append(region.region_id);continue
        a,b=fractions(before[window],mask),fractions(candidate[window],mask)
        row=dict(region_id=region.region_id,phase_class=region.phase_class,box_xyxy=list(bounds),
            before=a,after=b,before_constraint=constraints(a,region),after_constraint=constraints(b,region),phases={})
        for phase,key in (('co2_presence','co2'),('aqueous','aqueous_evidence'),('gas','gas_evidence')):
            target=regional_presence_target(region.phase_class,phase)
            old,new=a[key+'_pixels'],b[key+'_pixels'];fraction=b[key+'_fraction']
            row['phases'][phase]=dict(target=target,change_pixels=new-old,change_area_m2=(new-old)*pixel_m2,
                change_percentage_points=100*(fraction-a[key+'_fraction']),positive_newly_empty=target==1 and old>0 and new==0,
                positive_coverage_reduced=target==1 and new<old,
                negative_coverage_increased=target==0 and new>old,
                gained_any_support=target==1 and old==0 and new>0,
                gained_only_below_10pct_reference=target==1 and old==0 and new>0 and fraction<.1,
                gained_single_pixel_only=target==1 and old==0 and new==1)
        if raw is not None:
            values=raw[window][mask]
            row['raw_co2_support']=dict(mean_probability=float(values.mean()),p90_probability=float(np.quantile(values,.9)),
                fraction_ge_05=float((values>=.5).mean()),fraction_ge_095=float((values>=.95).mean()))
        if candidate_probability is not None:
            values=np.asarray(candidate_probability)[window][mask]
            k=max(1,round(len(values)*.1))
            row['candidate_probability']=dict(mean=float(values.mean()),best_10pct_mean=float(np.partition(values,len(values)-k)[-k:].mean()))
        rows.append(row)
    counters=Counter()
    for row in rows:
        for name in ('positive_newly_empty','positive_coverage_reduced','negative_coverage_increased','gained_any_support','gained_only_below_10pct_reference','gained_single_pixel_only'):
            counters[name]+=any(p[name] for p in row['phases'].values())
    strong=(raw>=.95)&domain if raw is not None else None
    return dict(frame_id=frame['frame_id'],evaluation_role=frame['entry']['evaluation_role'],independent_labeled_accuracy=False,
        gas_labels_exact=bool(np.array_equal(candidate==2,before==2)),evaluated_regions=len(rows),skipped_regions=skipped,
        counters=dict(counters),before_constraints_satisfied=sum(r['before_constraint']['satisfied'] for r in rows),
        after_constraints_satisfied=sum(r['after_constraint']['satisfied'] for r in rows),
        old_aqueous_pixels=int(((before==1)&domain).sum()),new_aqueous_pixels=int(((candidate==1)&domain).sum()),
        strong_raw_co2_pixels=int(strong.sum()) if strong is not None else None,
        strong_raw_co2_absent_in_saved_pixels=int((strong&(before==0)).sum()) if strong is not None else None,
        strong_raw_co2_removed_pixels=int((strong&(candidate==0)).sum()) if strong is not None else None,
        strong_raw_co2_newly_removed_pixels=int((strong&(before>0)&(candidate==0)).sum()) if strong is not None else None,
        strong_raw_co2_recovered_pixels=int((strong&(before==0)&(candidate>0)).sum()) if strong is not None else None,
        strong_raw_support_scope='Raw >=0.95 is descriptive optical evidence, not independently verified truth or a required single-pixel protection.',
        scope='10% coverage is descriptive, not annotation truth; a one-pixel or below-reference gain is not reported as complete box repair.',regions=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','raw'])
    parser.add_argument('--manifest',type=Path,default=CAMPAIGN/'evidence_manifest.json')
    parser.add_argument('--ids',nargs='*');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.action=='freeze':build_manifest(args.manifest)
    else:materialize_raw(args.manifest,args.ids,args.threads)
