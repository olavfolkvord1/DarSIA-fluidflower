"""Render committed photographs on a disjoint display-only host shard.

The manifest names exact sources. Waiting for a phase receipt never invokes a
phase model; all six transfers and reference roles are the fixed renderer's.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'


def asset(path):
    path=Path(path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(item):
    raw=Path(item['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:
        raise ValueError('Display metadata SHA differs: '+item['path'])
    return json.loads(raw)


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def admit(manifest_path):
    descriptor=asset(manifest_path);manifest=bound(descriptor);plan=bound(manifest['phase_plan'])
    if (manifest.get('contract')!='ffac-selected-display-host-shard-v1'
            or manifest.get('display_only') is not True
            or any(manifest.get(k) is not False for k in
                ('phase_inference','scientific_inputs_changed','maps_changed','labels_changed','scorer_executed','round3_allowed'))
            or manifest.get('pipeline_sha256')!=plan['pipeline_sha256']
            or manifest.get('shared_stop_path')!=plan['phase_campaign_binding']['shared_stop_path']):
        raise ValueError('Display shard lost its immutable display-only plan binding')
    if socket.gethostname().casefold()!=manifest['approved_host'].casefold():
        raise ValueError('Display shard belongs to another host')
    declared=set()
    for run,rows in manifest['runs'].items():
        inventory=bound(plan['runs'][run]['source_inventory'])
        sources={r['source_sha256']:r for r in inventory['sources'] if r.get('baseline') or r.get('selected')}
        for row in rows:
            key=(run,row['source_sha256'])
            if key in declared or sources.get(row['source_sha256'])!=row:
                raise ValueError('Display sources must be unique exact selected/baseline objects')
            declared.add(key)
    if Path(manifest['shared_stop_path']).exists():
        raise ValueError('Stopped campaign cannot dispatch display work')
    return descriptor,manifest,plan


def render_run(task):
    manifest_path,run,poll_seconds,wait_hours=task
    descriptor,manifest,plan=admit(manifest_path)
    from darsia.fluidflower.phase_ffac_selected_display_baseline_roles import BaselineRoleSelectedDisplayRenderer
    import cv2
    cv2.setNumThreads(1)
    rows=sorted(manifest['runs'][run],key=lambda r:(r['elapsed_h'],r['source_sha256']))
    root=Path(manifest['candidate']);renderer=None;results=[];started=time.perf_counter()
    deadline=started+wait_hours*3600
    for source in rows:
        phase=root/'selected_phase_frames'/run/(source['source_sha256']+'.json')
        while not phase.is_file():
            if Path(manifest['shared_stop_path']).exists() or Path(manifest['display_stop_path']).exists():
                raise ValueError('Shared stop before next display source '+run+'/'+source['source_sha256'])
            if time.perf_counter()>deadline:
                raise TimeoutError('No committed phase receipt for display source '+run+'/'+source['source_sha256'])
            time.sleep(poll_seconds)
        if Path(manifest['shared_stop_path']).exists() or Path(manifest['display_stop_path']).exists():
            raise ValueError('Shared stop before photographic rendering')
        if renderer is None:
            renderer=BaselineRoleSelectedDisplayRenderer(root,plan,run,backend='cpu')
        receipt=renderer.render(source['source_sha256'])
        results.append(receipt)
        saved=bound(receipt)
        print(json.dumps({'event':'display_frame_complete','run':run,'source_sha256':source['source_sha256'],
            'elapsed_h':source['elapsed_h'],'seconds':saved['seconds'],'host':socket.gethostname(),
            'phase_inference':False}),flush=True)
    return {'run':run,'complete':True,'frames':len(results),'receipts':results,
        'seconds':time.perf_counter()-started,'display_only':True,'host':socket.gethostname()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--destination',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--poll-seconds',type=float,default=15)
    parser.add_argument('--wait-hours',type=float,default=48)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=8: raise ValueError('Display workers must be bounded to 1..8')
    destination=Path(args.destination)
    if destination.exists(): raise FileExistsError('Use a new display shard stage receipt')
    descriptor,manifest,plan=admit(args.manifest)
    started=datetime.now(timezone.utc).isoformat();results=[];errors=[]
    deadline=time.perf_counter()+args.wait_hours*3600
    with ProcessPoolExecutor(max_workers=min(args.workers,len(manifest['runs'])),mp_context=multiprocessing.get_context('spawn')) as pool:
        waiting=list(manifest['runs']);active={}
        while waiting or active:
            for future,run in list(active.items()):
                if not future.done(): continue
                del active[future]
                try: results.append(future.result())
                except Exception as error:
                    errors.append({'run':run,'error':repr(error)})
                    try: write_new(manifest['display_stop_path'],{'display_only':True,'reason':repr(error),'run':run})
                    except FileExistsError: pass
            stopped=Path(manifest['shared_stop_path']).exists() or Path(manifest['display_stop_path']).exists()
            if stopped or time.perf_counter()>deadline:
                if waiting and not errors: errors.append({'error':'Shared stop or display wait deadline','not_dispatched':waiting})
                waiting=[]
                for future in active: future.cancel()
            if not errors:
                for run in list(waiting):
                    if len(active)>=args.workers: break
                    first=min(manifest['runs'][run],key=lambda r:(r['elapsed_h'],r['source_sha256']))
                    phase=Path(manifest['candidate'])/'selected_phase_frames'/run/(first['source_sha256']+'.json')
                    if phase.is_file():
                        active[pool.submit(render_run,(args.manifest,run,args.poll_seconds,args.wait_hours))]=run
                        waiting.remove(run)
                        print(json.dumps({'event':'display_ready_run_dispatched','run':run,'active_runs':len(active)}),flush=True)
            if waiting or active: time.sleep(args.poll_seconds)
    expected=sum(len(rows) for rows in manifest['runs'].values())
    complete=not errors and len(results)==len(manifest['runs']) and sum(r['frames'] for r in results)==expected
    record={'contract':'ffac-selected-display-host-shard-stage-v1','complete':complete,
        'manifest':descriptor,'phase_plan':manifest['phase_plan'],'pipeline_sha256':plan['pipeline_sha256'],
        'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'host':socket.gethostname(),
        'pid':os.getpid(),'workers':args.workers,'internal_threads':1,'display_only':True,
        'expected_frames':expected,'results':results,'errors':errors,'code':asset(__file__),
        'phase_inference':False,'scientific_inputs_changed':False,'maps_changed':False,
        'labels_changed':False,'sessions_changed':False,'scorer_executed':False,'round3_allowed':False}
    write_new(destination,record);print(json.dumps(asset(destination)),flush=True)
    return 0 if complete else 2


if __name__=='__main__': raise SystemExit(main())
