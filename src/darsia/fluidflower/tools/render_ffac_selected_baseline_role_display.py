"""Render file-backed selected Studio photographs after complete Policy A QA.

One pre-injection reference is retained per worker/run. This companion writes
only photographs and display receipts; it never invokes phase inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import platform
import socket
import time

from darsia.fluidflower.phase_ffac_selected_display_baseline_roles import BaselineRoleSelectedDisplayRenderer
from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json


def admit(plan_path, completion_path):
    plan_descriptor, completion_descriptor = asset(plan_path), asset(completion_path)
    plan, completion = bound_json(plan_descriptor), bound_json(completion_descriptor)
    if (completion.get('complete') is not True or completion.get('all_39_phase_complete') is not True
            or completion.get('coarse_area_gate_passed') is not True
            or completion.get('plan') != plan_descriptor
            or completion.get('identity_gate') != plan['identity_gate']
            or completion.get('pipeline_sha256') != plan['pipeline_sha256']
            or set(completion.get('runs',{})) != set(plan['runs'])
            or completion.get('counts') != plan['phase_campaign_binding']['expected_counts']
            or completion.get('round3_allowed') is not False):
        raise ValueError('Photographs require the actual complete, unmodified Policy A phase package')
    host=bound_json(plan['phase_campaign_binding']['approved_main_host'])
    if socket.gethostname()!=host['hostname'] or platform.machine()!=host['machine']:
        raise ValueError('This rendering dispatch is bound to the approved Main host')
    if Path(plan['phase_campaign_binding']['shared_stop_path']).exists():
        raise ValueError('Stopped Policy A package cannot be rendered for registration')
    return plan_descriptor, plan, completion_descriptor


def render_run(task):
    root,plan_path,run=task
    plan=bound_json(asset(plan_path));started=time.perf_counter()
    renderer=BaselineRoleSelectedDisplayRenderer(root,plan,run,backend='cpu')
    rows=sorted((row for row in renderer.inventory['sources'] if row.get('baseline') or row.get('selected')),
                key=lambda row:(row['elapsed_h'],row['source_sha256']))
    receipts=[]
    for source in rows:
        if Path(plan['phase_campaign_binding']['shared_stop_path']).exists():
            raise ValueError('Shared stop reached before photographic dispatch')
        receipts.append(renderer.render(source['source_sha256']))
    return dict(run=run,complete=True,frames=len(receipts),receipts=receipts,
                seconds=time.perf_counter()-started,display_only=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','plan','completion','destination'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args(argv)
    if args.workers<1 or args.workers>12:
        raise ValueError('Bounded display workers required')
    destination=Path(args.destination)
    if destination.exists():
        raise FileExistsError('Use a new display stage receipt')
    descriptor,plan,completion=admit(args.plan,args.completion)
    results=[];errors=[];started=datetime.now(timezone.utc).isoformat()
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(args.workers,len(plan['runs'])),mp_context=context) as pool:
        tasks={pool.submit(render_run,(args.root,args.plan,run)):run for run in sorted(plan['runs'])}
        for future in as_completed(tasks):
            run=tasks[future]
            try:
                row=future.result();results.append(row)
                print(json.dumps(dict(event='display_run_complete',run=run,frames=row['frames'],seconds=row['seconds'])),flush=True)
            except Exception as error:
                errors.append(dict(run=run,error=repr(error)))
                for pending in tasks: pending.cancel()
    expected=plan['phase_campaign_binding']['expected_counts']
    expected_frames=expected['baseline_frames']+expected['selected_frames']
    complete=not errors and len(results)==len(plan['runs']) and sum(r['frames'] for r in results)==expected_frames
    record=dict(contract='ffac-selected-baseline-role-display-stage-v1',complete=complete,
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),plan=descriptor,
        phase_completion=completion,code=asset(__file__),renderer_code=asset(__import__(BaselineRoleSelectedDisplayRenderer.__module__,fromlist=['']).__file__),
        host=socket.gethostname(),pid=os.getpid(),cpu_only=True,workers=args.workers,
        baseline_role='separately prepared historical pre-injection reference',
        results=results,errors=errors,expected_frames=expected_frames,
        scientific_inputs_changed=False,maps_changed=False,labels_changed=False,
        annotations_read=False,scorer_executed=False,round3_allowed=False)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(asset(destination)),flush=True)
    return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
