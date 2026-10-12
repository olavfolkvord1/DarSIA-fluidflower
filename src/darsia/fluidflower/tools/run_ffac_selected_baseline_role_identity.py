"""Main-CPU baseline-role identity with a source-bound constructor error boundary.

The unchanged former row executor is used with the new reference-role adapter.
Old rows may only enter through a separately verified full used-array audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import importlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import socket
import traceback
from unittest.mock import patch

import numpy as np

from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json
from darsia.fluidflower.phase_ffac_selected_required_identity_baseline import RequiredIdentityEvaluator
from darsia.fluidflower.tools import run_ffac_selected_required_identity as old
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now, worker_initializer

SPEC_SHA256 = '62bf2970077c9a270f85fe3f046e48811af70d5e1c1761cd729d6289782bef32'
CONTRACT = 'ffac-selected-baseline-role-required-cpu-identity-v1'
MODULES = old.MODULES + ('darsia.fluidflower.phase_ffac_selected_baseline_roles',
    'darsia.fluidflower.phase_ffac_selected_required_identity_baseline',
    'darsia.fluidflower.tools.run_ffac_selected_baseline_role_identity')
CONTROLS = (('ac19', 1), ('ac54', 2))
MISSING_RUNS = ('ac28', 'ac31', 'ac42', 'ac53', 'ac55', 'ac56', 'ac57', 'ac58', 'ac59', 'ac60', 'ac61')


def host_record():
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
        if hasattr(np, 'show_runtime'): np.show_runtime()
    return dict(hostname=socket.gethostname(), processor=platform.processor(), machine=platform.machine(),
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        cpu_blas_runtime=config.getvalue(), pid=os.getpid(), cpu_only=True, internal_threads=1)


def bind(args):
    root=Path(args.root)
    with patch.object(old, 'SPEC_SHA256', SPEC_SHA256):
        descriptor=old.bind(args.source_plan,args.identity_inventory,root/'former_binding',
            args.frozen_gas_audit,args.historical_validation_source)
    plan=bound_json(descriptor)
    host=host_record();host_path=root/'main_cpu_host.json';atomic_json(host_path,host)
    binding=dict(plan['phase_campaign_binding'],contract=CONTRACT,spec_sha256=SPEC_SHA256,
        baseline_role='independently prepared historical pre-injection reference; current frame0 remains current frame0',
        previous_row_executor=descriptor, approved_main_host=asset(host_path),
        source_producer_host_is_separate_from_phase_host=True,
        constructor_errors_source_bound_before_creation=True,
        imported_identity_requires_full_used_input_array_sha_equivalence=True,
        fresh_control_runs=['ac19','ac54'],
        code={name:{'sha256':asset(importlib.import_module(name).__file__)['sha256']} for name in MODULES})
    plan.update(phase_campaign_binding=binding,pipeline_sha256=canonical_sha256(binding),spec_sha256=SPEC_SHA256)
    path=root/'identity_plan.json'
    if path.exists():raise FileExistsError('New immutable binding required')
    atomic_json(path,plan)
    return asset(path)


def construction_identity(plan,run):
    state=plan['runs'][run];inventory=bound_json(state['source_inventory'])
    source=next(row for row in inventory['sources'] if row['source_sha256']==inventory['baseline_source_sha256'])
    wrapper=asset(Path(plan['scientific_source_root'])/'selected_sources'/run/source['source_sha256']/'receipt.json')
    receipt=bound_json(wrapper)
    native=bound_json(state['parent_native_manifest'])
    return dict(run=run, source_sha256=source['source_sha256'], original_studio_frame_index=0,
        source_work_index=source['work_index'], phase='baseline_constructor',
        scientific_wrapper=wrapper,current_frame0_native_scientific=receipt['native_scientific'],
        historical_reference=receipt['baseline_extended_scientific'],
        original_native_reference=native['reference'],parent_native_manifest=state['parent_native_manifest'])


def run_sequence(plan_descriptor,root,run,through_frame):
    root=Path(root);worker_initializer(1)
    if (root/'identity_stop.json').exists():return dict(run=run,complete=False,shared_stop=True,receipts=[])
    active=dict(run=run,phase='metadata_admission',source_sha256=None,original_studio_frame_index=None)
    receipts=[]
    try:
        plan=bound_json(plan_descriptor)
        if plan['spec_sha256']!=SPEC_SHA256:raise ValueError('Wrong authorized specification')
        old.protected.verify_code(plan)
        host=host_record();approved=bound_json(plan['phase_campaign_binding']['approved_main_host'])
        if host['hostname']!=approved['hostname'] or host['machine']!=approved['machine']:
            raise ValueError('This host has not passed the required Main CPU gate')
        active.update(construction_identity(plan,run))
        journal=root/'execution_journal'/run/'constructor_start.json'
        if not journal.exists():atomic_json(journal,dict(active,created_utc=utc_now(),identity_plan=plan_descriptor,host=host))
        actual_constructor=RequiredIdentityEvaluator

        class TrackedEvaluator(actual_constructor):
            def __init__(self,*positional,**named):
                super().__init__(*positional,**named)
                active['baseline_role_proof']=self.baseline_role_proof
            def evaluate(self,digest):
                row=self.sources[digest]
                active.update(source_sha256=digest,original_studio_frame_index=row['studio_frame_index'],
                    source_work_index=row['work_index'],phase='inference',previous_source_sha256=row['previous_source_sha256'])
                entry=root/'execution_journal'/run/f"frame_{row['studio_frame_index']:03d}_start.json"
                if not entry.exists():atomic_json(entry,dict(active,created_utc=utc_now(),identity_plan=plan_descriptor,host=host))
                return super().evaluate(digest)
            def compare_saved_probabilities(self,row):
                active['phase']='required_comparison'
                return super().compare_saved_probabilities(row)

        original_write=old.atomic_json
        def row_write(path,record):
            if 'stored_probabilities' in record and 'original_studio_frame_index' in record:
                record=dict(record,origin='fresh_new_baseline_role_identity',phase_execution_host=host,
                    baseline_role_proof=active.get('baseline_role_proof'),producer_previous_executor_sha256=asset(old.__file__)['sha256'])
            return original_write(path,record)
        with patch.object(old,'RequiredIdentityEvaluator',TrackedEvaluator),patch.object(old,'SPEC_SHA256',SPEC_SHA256), \
                patch.object(old,'CONTRACT',CONTRACT),patch.object(old,'atomic_json',row_write):
            result=old.run_sequence(plan_descriptor,root,run,through_frame)
        receipts=result.get('receipts',[])
        return result
    except Exception as error:
        cause=getattr(error,'evidence',None)
        failure=dict(contract=CONTRACT,created_utc=utc_now(),stop=True,stop_new_work=True,
            production_allowed=False,reason='Required identity exception',run=run,
            input_identity=dict(active),original_studio_frame_index=active.get('original_studio_frame_index'),
            source_sha256=active.get('source_sha256'),phase=active.get('phase'),
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            adapter_evidence=cause,identity_plan=plan_descriptor,receipts=receipts)
        failure_path=root/'execution_failures'/f'{run}.json'
        if not failure_path.exists():atomic_json(failure_path,failure)
        old._publish_stop(root,dict(failure,failed_input=asset(failure_path)))
        print(json.dumps({'event':'source_bound_identity_exception','run':run,'source_sha256':failure['source_sha256'],
            'phase':failure['phase'],'error':str(error),'receipt':asset(failure_path)}),flush=True)
        return dict(run=run,complete=False,stopped=True,error=asset(failure_path),receipts=receipts)


def run_stage(plan_descriptor,root,stage,workers,used_input_audit=None):
    root=Path(root)
    if (root/'identity_stop.json').exists():raise RuntimeError('Shared identity stop preserved; no retry')
    if stage=='controls':tasks=list(CONTROLS)
    elif stage=='remaining':
        controls=bound_json(asset(root/'controls_stage.json'))
        if controls.get('complete') is not True:raise ValueError('Fresh AC19/54 controls must pass first')
        if used_input_audit is None:raise ValueError('Complete used-input audit required before deciding reused runs')
        audit=bound_json(used_input_audit)
        if audit.get('complete') is not True:raise ValueError('Used-input audit is incomplete')
        plan=bound_json(plan_descriptor)
        identity=bound_json(plan['phase_campaign_binding']['identity_inventory'])
        rerun=set(MISSING_RUNS)|set(audit['rerun_runs'])
        tasks=[(run,max(row['studio_frame_index'] for row in identity['runs'][run]['frames']))
            for run in sorted(rerun)]
    else:raise ValueError('Unknown stage')
    output=root/f'{stage}_stage.json'
    if output.exists():raise FileExistsError('Stage output is immutable')
    results=[]
    if workers==1:
        for run,last in tasks:
            result=run_sequence(plan_descriptor,root,run,last);results.append(result)
            if not result.get('complete'):break
    else:
        pool=ProcessPoolExecutor(max_workers=min(workers,len(tasks)),mp_context=multiprocessing.get_context('spawn'))
        pending={pool.submit(run_sequence,plan_descriptor,str(root),run,last):(run,last) for run,last in tasks}
        try:
            for future in as_completed(pending):
                run,last=pending[future]
                try:result=future.result()
                except Exception as error:
                    record=dict(contract=CONTRACT,created_utc=utc_now(),stop=True,stop_new_work=True,
                        production_allowed=False,run=run,through_frame=last,phase='worker_process',
                        error_type=type(error).__name__,error=str(error),identity_plan=plan_descriptor)
                    old._publish_stop(root,record);result=dict(run=run,complete=False,error=record)
                results.append(result)
                if not result.get('complete'):
                    for item in pending:item.cancel()
                    break
        finally:pool.shutdown(wait=True,cancel_futures=True)
    complete=len(results)==len(tasks) and all(row.get('complete') is True for row in results)
    record=dict(contract=CONTRACT,created_utc=utc_now(),stage=stage,identity_plan=plan_descriptor,
        tasks=[dict(run=run,through_frame=last) for run,last in tasks],results=results,complete=complete,
        cpu_only=True,internal_threads=1,used_input_audit=used_input_audit,
        all_39_complete=False,policy_a_production_allowed=False)
    atomic_json(output,record)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('bind','controls','remaining'),required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--plan')
    parser.add_argument('--source-plan');parser.add_argument('--identity-inventory')
    parser.add_argument('--frozen-gas-audit');parser.add_argument('--historical-validation-source')
    parser.add_argument('--used-input-audit')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv)
    if args.stage=='bind':print(json.dumps(bind(args)));return 0
    if not 1<=args.workers<=12:raise ValueError('Bounded Main CPU worker count required')
    audit=asset(args.used_input_audit) if args.used_input_audit else None
    record=run_stage(asset(args.plan),args.root,args.stage,args.workers,audit)
    print(json.dumps(record),flush=True)
    return 0 if record['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
