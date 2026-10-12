"""Disjoint whole-run T2 inference on shared storage; one final coordinator seal.

Run this orchestration wrapper against the unchanged scientific code snapshot.
Workers resume the same candidate's committed frame receipts. They never fit,
register, seal, expose a held-out run, or modify an existing scientific input.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import re
import socket
import uuid

from darsia.fluidflower.tools import run_ffac_t2_floor_candidate as runner
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, read_json, sha256_file, utc_now

CONTRACT = 'ffac-t2-distributed-inference-v1'
ASSIGNMENT = 'distributed_inference/assignment.json'
_NAME = re.compile(r'^[A-Za-z0-9_-]{1,48}$')


def partition_runs(sources, owners):
    """Deterministic frame-balanced partition, with one owner for every run."""
    if not owners or len(set(owners)) != len(owners) or any(not _NAME.fullmatch(o) for o in owners):
        raise ValueError('Distinct portable worker-owner names are required')
    counts = {}
    keys = set()
    for row in sources:
        run, index = row['run'], row['frame_index']
        if not re.fullmatch(r'ac\d+', run) or type(index) is not int or index < 0 or (run,index) in keys:
            raise ValueError('Duplicate or invalid source-owned native frame')
        keys.add((run,index)); counts[run] = counts.get(run,0) + 1
    assignments = {owner: [] for owner in owners}; load = dict.fromkeys(owners,0)
    for run in sorted(counts,key=lambda name:(-counts[name],name)):
        owner = min(owners,key=lambda name:(load[name],owners.index(name)))
        assignments[owner].append(run); load[owner] += counts[run]
    if any(not runs for runs in assignments.values()):
        raise ValueError('Every assigned computer needs at least one whole run')
    return {owner:dict(runs=sorted(assignments[owner]),frame_count=load[owner]) for owner in owners}


def candidate_state(candidate):
    root = Path(candidate).resolve()
    plan = read_json(root/runner.PLAN)
    if plan['binding']['inputs']['code'] != runner.code_binding():
        raise ValueError('Worker must load the immutable scientific code snapshot')
    template = read_json(root/'profile_template.json')
    runner.validate_runtime_heads(root,template['g3_t2_floor_revision'],plan['binding']['heads'])
    parent_plan = read_json(runner.bound_path(plan['binding']['inputs']['parent_plan']))
    sources = parent_plan['source_inventory']
    if len(sources) != 864 or len({row['run'] for row in sources}) != 43:
        raise ValueError('Distributed work requires the exact sealed 43/864 inventory')
    return root, plan, sources


def prepare_assignment(candidate, owners, *, workers_per_owner=12):
    root,plan,sources = candidate_state(candidate)
    if type(workers_per_owner) is not int or not 1 <= workers_per_owner <= 12:
        raise ValueError('Each computer accepts at most twelve one-thread workers')
    if (root/'job.pidlock').exists():
        raise RuntimeError('An existing full-run inference lease must finish before assigning workers')
    partitions = partition_runs(sources,owners)
    binding = dict(candidate=str(root),candidate_plan=runner.artifact(root/runner.PLAN),
        pipeline_sha256=plan['pipeline_sha256'],heads=plan['binding']['heads'],
        scientific_code=plan['binding']['inputs']['code'],orchestrator_sha256=sha256_file(__file__),
        source_inventory_sha256=canonical_sha256(sources),assignments=partitions,
        workers_per_owner=workers_per_owner,threads_per_worker=1,cpu_only=True,
        bottom_mass_allowed=False,mass_eligible=False,blind_hold_runs=sorted(runner.HIDDEN))
    path=root/ASSIGNMENT
    if path.exists():
        previous=read_json(path)
        if previous.get('contract') != CONTRACT or previous.get('binding') != binding:
            raise ValueError('Refusing to change an existing distributed assignment')
        return path
    atomic_json(path,dict(contract=CONTRACT,binding=binding,assignment_sha256=canonical_sha256(binding),created_utc=utc_now()))
    return path


def load_assignment(path, owner=None):
    path=Path(path); assignment=read_json(path); binding=assignment['binding']
    if assignment.get('contract') != CONTRACT or assignment.get('assignment_sha256') != canonical_sha256(binding):
        raise ValueError('Distributed assignment identity changed')
    if binding['orchestrator_sha256'] != sha256_file(__file__):
        raise ValueError('Distributed wrapper differs from the assigned copy')
    root,plan,sources=candidate_state(binding['candidate'])
    if runner.bound_path(binding['candidate_plan']) != (root/runner.PLAN).resolve():
        raise ValueError('Assignment belongs to another candidate plan')
    if (plan['pipeline_sha256'] != binding['pipeline_sha256'] or plan['binding']['heads'] != binding['heads']
            or canonical_sha256(sources) != binding['source_inventory_sha256']):
        raise ValueError('Distributed source/model binding changed')
    assigned=[run for row in binding['assignments'].values() for run in row['runs']]
    if len(assigned) != 43 or len(set(assigned)) != 43 or set(assigned) != {row['run'] for row in sources}:
        raise ValueError('Runs must be assigned to exactly one computer')
    if owner is not None and owner not in binding['assignments']:
        raise ValueError('Unassigned worker owner')
    return root,plan,sources,assignment


@contextmanager
def exclusive_claim(root, name, assignment_sha256, owner):
    """NAS no-clobber claim. No foreign PID guessing or automatic stale recovery."""
    if not _NAME.fullmatch(name) or not _NAME.fullmatch(owner):
        raise ValueError('Invalid claim name')
    path=Path(root)/'distributed_inference/claims'/f'{name}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(owner=owner,hostname=socket.gethostname(),pid=os.getpid(),token=uuid.uuid4().hex,
        assignment_sha256=assignment_sha256,acquired_utc=utc_now())
    try:
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f'Run/owner already claimed; inspect the existing claim: {path}') from error
    try:
        yield
    finally:
        if path.exists() and read_json(path).get('token') == value['token']:
            path.unlink()


def execute_run(assignment_path, owner, run):
    root,plan,sources,assignment=load_assignment(assignment_path,owner)
    if run not in assignment['binding']['assignments'][owner]['runs']:
        raise ValueError('Worker attempted another computer\'s run')
    if (root/'job.pidlock').exists():
        raise RuntimeError('A full-run candidate lease conflicts with distributed work')
    if (root/'distributed_inference/claims/seal.json').exists():
        raise RuntimeError('Coordinator seal is active')
    with exclusive_claim(root,run,assignment['assignment_sha256'],owner):
        result=runner.run_one_run(root,run)
        expected=[row for row in sources if row['run'] == run]
        validate_run_result(root,plan,expected,result)
        return result


def validate_run_result(root, plan, sources, result):
    """Verify every committed frame before reuse and before the sole seal."""
    expected={(row['run'],row['frame_index']):row for row in sources}
    items=result['receipts']
    keys=[(row['run'],row['frame_index']) for row in items]
    if (result.get('complete') is not True or result.get('pipeline_sha256') != plan['pipeline_sha256']
            or result['frame_count'] != len(expected) or len(items) != len(expected)
            or len(set(keys)) != len(keys) or set(keys) != set(expected)):
        raise ValueError('Run result omits, duplicates or substitutes a source frame')
    runner.bound_path(result['native_manifest']);runner.bound_path(result['input_manifest'])
    for item in items:
        frame=read_json(runner.bound_path(item['receipt']));source=expected[(item['run'],item['frame_index'])]
        binding=frame['binding'];parent_receipt=runner.artifact(Path(plan['binding']['parent'])/'rig_camera_frames'/source['run']/f"frame_{source['frame_index']:03d}.json")
        if (frame.get('complete') is not True or frame.get('contract') != runner.CONTRACT
                or binding != dict(pipeline_sha256=plan['pipeline_sha256'],source=source,parent_receipt=parent_receipt)
                or frame['source_sha256'] != source['source_sha256'] or frame['elapsed_h'] != source['elapsed_h']
                or frame.get('mass_eligible') is not False or frame.get('bottom_mass_allowed') is not False
                or frame['qa'].get('all_pass') is not True or frame['qa']['union_changes_outside_fill']
                or frame['qa']['preinject_fill_pixels']):
            raise ValueError('Committed frame changed source/model/floor/mass policy')
        for asset in [frame['labels_map'],frame['floor_contact_fill'],*frame['maps'].values()]:
            runner.bound_path(asset)


def work(assignment_path, owner, *, workers=None):
    root,_plan,_sources,assignment=load_assignment(assignment_path,owner)
    maximum=assignment['binding']['workers_per_owner']
    requested=maximum if workers is None else workers
    if type(requested) is not int or not 1 <= requested <= maximum:
        raise ValueError('Worker allocation exceeds the assigned CPU budget')
    runs=assignment['binding']['assignments'][owner]['runs']
    admitted=min(runner.admitted_workers(requested),len(runs))
    results=[]
    with exclusive_claim(root,'owner_'+owner,assignment['assignment_sha256'],owner):
        if (root/'job.pidlock').exists() or (root/'distributed_inference/claims/seal.json').exists():
            raise RuntimeError('Inference owner conflicts with a full run or coordinator seal')
        with ProcessPoolExecutor(max_workers=admitted,mp_context=multiprocessing.get_context('spawn'),
                initializer=runner.worker_initializer) as pool:
            futures={pool.submit(execute_run,str(assignment_path),owner,run):run for run in runs}
            for future in as_completed(futures):
                result=future.result();results.append(result)
                print(json.dumps(dict(event='distributed_run_complete',owner=owner,run=result['run'],frames=result['frame_count'])),flush=True)
    output=root/'distributed_inference/owners'/f'{owner}.json'
    atomic_json(output,dict(contract=CONTRACT,complete=True,assignment_sha256=assignment['assignment_sha256'],
        owner=owner,runs=sorted(row['run'] for row in results),
        results=[runner.artifact(root/'t2_floor_runs'/f"{row['run']}.json") for row in sorted(results,key=lambda row:row['run'])]))
    return output


def seal(assignment_path):
    root,plan,sources,assignment=load_assignment(assignment_path)
    if any((root/'distributed_inference/claims').glob('*.json')):
        raise RuntimeError('Cannot seal while any owner/run claim is active')
    with exclusive_claim(root,'seal',assignment['assignment_sha256'],'coordinator'):
        if any(path.name != 'seal.json' for path in (root/'distributed_inference/claims').glob('*.json')):
            raise RuntimeError('A worker claimed work while the coordinator was acquiring its seal')
        rows=[]
        for run in sorted({row['run'] for row in sources}):
            row=read_json(root/'t2_floor_runs'/f'{run}.json')
            validate_run_result(root,plan,[source for source in sources if source['run']==run],row)
            rows.append(row)
        if (root/runner.SEAL).exists():
            # A second call verifies the committed seal instead of rewriting it.
            saved=read_json(root/runner.SEAL)
            if (saved.get('complete') is not True or saved.get('pipeline_sha256') != plan['pipeline_sha256']
                    or saved.get('run_count') != 43 or saved.get('frame_count') != 864):
                raise ValueError('Existing candidate seal differs from the assignment')
            return saved
        return runner.seal_candidate(root,plan,rows)


def install_asset_aliases(manifest, destination):
    """Explicit byte-equivalent file aliases; never rewrite a frozen receipt."""
    value=read_json(manifest)
    if value.get('contract') != 'ffac-sha-bound-asset-aliases-v1':
        raise ValueError('Unsupported worker asset aliases')
    from darsia.fluidflower.review_storage import review_aliases
    rows=list(review_aliases())
    for row in value['assets']:
        source,target=Path(row['source']),Path(row['destination'])
        if not source.is_absolute() or not target.is_absolute() or source.resolve()==target.resolve():
            raise ValueError('Worker alias requires distinct absolute source/destination')
        if sha256_file(target) != row['sha256']:
            raise ValueError('Worker alias destination is not byte-equivalent')
        rows.append(dict(source=str(source),destination=str(target),original_save_root=str(source),save_root=str(target)))
    path=Path(destination)
    atomic_json(path,dict(format='fluidflower-review-storage-aliases-v1',aliases=rows))
    existing=os.environ.get('DARSIA_REVIEW_ALIAS_FILES','')
    os.environ['DARSIA_REVIEW_ALIAS_FILES']=os.pathsep.join([str(path),*([existing] if existing else [])])
    review_aliases(refresh=True)
    return path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-aliases',type=Path)
    p.add_argument('--alias-registry',type=Path)
    commands=p.add_subparsers(dest='command',required=True)
    create=commands.add_parser('assign');create.add_argument('--candidate',required=True,type=Path)
    create.add_argument('--owners',required=True,help='Comma-separated distinct computer names')
    create.add_argument('--workers-per-owner',default=12,type=int)
    worker=commands.add_parser('worker');worker.add_argument('--assignment',required=True,type=Path)
    worker.add_argument('--owner',required=True);worker.add_argument('--workers',type=int)
    final=commands.add_parser('seal');final.add_argument('--assignment',required=True,type=Path)
    a=p.parse_args(argv)
    if a.asset_aliases:
        if not a.alias_registry:p.error('--asset-aliases requires --alias-registry')
        install_asset_aliases(a.asset_aliases,a.alias_registry)
    if a.command=='assign':print(prepare_assignment(a.candidate,a.owners.split(','),workers_per_owner=a.workers_per_owner))
    elif a.command=='worker':print(work(a.assignment,a.owner,workers=a.workers))
    else:seal(a.assignment)


if __name__=='__main__':
    main()
