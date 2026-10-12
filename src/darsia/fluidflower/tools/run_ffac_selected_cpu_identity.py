"""Bind and replay CPU P1-final3's own Studio history, stopping on any bit difference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import importlib
import io
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_selected_phase import asset,bound_json
from darsia.fluidflower.phase_ffac_selected_cpu_identity import IdentityEvaluator,array_identity,CONTRACT
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json,canonical_sha256,utc_now,worker_initializer
from darsia.fluidflower.tools import run_ffac_selected_phase_corrected as old_driver
from darsia.fluidflower.tools import run_ffac_rig_camera_inference as rig

SPEC_SHA256='a3e4a3b139b8b810d4d650cff5bab29a25f662385f12593bf96d529a54ca10c2'
MODULES=old_driver.MODULES+('darsia.fluidflower.phase_ffac_selected_cpu_identity',
    'darsia.fluidflower.tools.run_ffac_selected_cpu_identity',
    'darsia.fluidflower.phase_g3_probability_layers','darsia.fluidflower.phase_training.v5_linear_input',
    'darsia.fluidflower.tools.ffac_v5_track1_photo_context')


def bind(source_plan,identity_inventory,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    descriptor=asset(source_plan);source=bound_json(descriptor)
    inventory_descriptor=asset(identity_inventory);inventory=bound_json(inventory_descriptor)
    inventories={};runs={}
    for run,state in sorted(source['runs'].items()):
        frames=inventory['runs'][run]['frames']
        rows=[]
        source_rows={row['work_index']:row for row in bound_json(state['source_inventory'])['sources']}
        for frame in frames:
            record=frame['source_record']
            if record['inventory']!=state['source_inventory']:
                raise ValueError('Own-history row references a different source inventory')
            row=dict(source_rows[record['work_index']])
            if row['source_sha256']!=frame['source_sha256']:
                raise ValueError('Own-history acquisition source differs from inventory')
            row['previous_source_sha256']=frame['original_previous_source_sha256']
            row['temporal_input_policy']='P1-final3-own-Studio-history-code-identity-only'
            row['identity_only']=True
            # Keep original work/source index and current aliases unchanged.
            rows.append(row)
        baseline=next(row['source_sha256'] for row in rows if row.get('baseline'))
        own=dict(contract='ffac-selected-own-studio-history-inventory-v1',run=run,
            sources=rows,baseline_source_sha256=baseline,
            history_policy='P1-final3 own original Studio sequence; no selected-grid predecessor',
            source_inventory=state['source_inventory'],identity_inventory=inventory_descriptor)
        path=root/'own_history_inventories'/f'{run}.json'
        if path.exists():
            if bound_json(asset(path))!=own:raise FileExistsError('Identity inventory already differs')
        else:atomic_json(path,own)
        inventories[run]=asset(path)
        runs[run]=dict(state,source_inventory=inventories[run])
    code={name:dict(sha256=asset(importlib.import_module(name).__file__)['sha256']) for name in MODULES}
    binding=dict(contract=CONTRACT,spec_sha256=SPEC_SHA256,source_preparation=descriptor,
        identity_inventory=inventory_descriptor,own_history_inventories=inventories,code=code,
        history_policy='P1-final3 own Studio sequence only',backend='cpu',internal_threads=1,
        tolerance_allowed=False,labels_must_be_bit_identical=True,stored_probabilities_must_be_bit_identical=True,
        unchanged_corrected_operator=True,spacing_or_previous_valid_conventions_prefixed=False,
        training=False,mass_computed=False,policy_a_production_allowed=False)
    plan=dict(source,runs=runs,phase_campaign_binding=binding,pipeline_sha256=canonical_sha256(binding),
        scientific_source_root=str(Path(source_plan).parent.parent),spec_sha256=SPEC_SHA256)
    path=root/'identity_plan.json'
    if path.exists():
        if bound_json(asset(path))!=plan:raise FileExistsError('Existing identity plan is immutable')
    else:atomic_json(path,plan)
    return asset(path)


def verify(plan):
    if plan['phase_campaign_binding']['spec_sha256']!=SPEC_SHA256:raise ValueError('Identity specification changed')
    old_driver.verify_code(plan)


def _save_arrays(folder,capture):
    output={};folder=Path(folder)
    names={'g3_pixel':'g3_pixel.npy','regional_frozen':'regional_frozen.npy',
        'regional_refit':'regional_refit.npy','co2':'co2.packbits.npy','extra_masks':'extra_masks.packbits.npz'}
    for key,name in names.items():
        payload=io.BytesIO()
        if key=='extra_masks':np.savez_compressed(payload,masks=capture[key])
        else:np.save(payload,capture[key],allow_pickle=False)
        output[name]=rig._write_bytes(folder/name,payload.getvalue())
    return output


def run_sequence(plan_descriptor,root,run,through_frame):
    worker_initializer(1)
    plan=bound_json(plan_descriptor);verify(plan)
    root=Path(root)
    evaluator=IdentityEvaluator(root,plan,run)
    rows=sorted(evaluator.inventory['sources'],key=lambda row:row['studio_frame_index'])
    rows=[row for row in rows if row['studio_frame_index']<=through_frame]
    if [row['studio_frame_index'] for row in rows]!=list(range(through_frame+1)):
        raise ValueError('Identity own-history sequence has a missing original Studio frame')
    receipts=[]
    for row in rows:
        if (root/'identity_stop.json').exists():
            return dict(run=run,complete=False,reason='shared identity stop',receipts=receipts)
        digest=row['source_sha256'];index=row['studio_frame_index']
        receipt_path=root/'identity_frames'/run/f'frame_{index:03d}.json'
        if receipt_path.exists():
            existing=bound_json(asset(receipt_path))
            if (existing['all_pass'] is not True or existing['identity_plan']!=plan_descriptor
                    or existing['source_sha256']!=digest or existing['previous_source_sha256']!=row['previous_source_sha256']
                    or existing['original_studio_frame_index']!=index):
                raise FileExistsError('Failed or differently bound identity receipt cannot be resumed')
            for item in existing['generated_probability_files'].values():rig.bound_path(item)
            rig.bound_path(existing['phase_receipt'])
            receipts.append(asset(receipt_path))
            print(json.dumps(dict(event='verified_identity_receipt_reused',run=run,studio_frame_index=index,
                source_sha256=digest,producer=asset(receipt_path))),flush=True)
            continue
        started=time.perf_counter()
        result=evaluator.evaluate(digest)
        parent=bound_json(row['parent_receipt'])
        with Image.open(rig.bound_path(parent['labels_map'])) as image:expected=np.asarray(image).copy()
        label_check=array_identity(result['labels'],expected)
        probability_check=evaluator.compare_saved_probabilities(row)
        capture=evaluator.centre.capture
        expected_probability=bound_json(probability_check['receipt'])
        original_meta=expected_probability['metadata']
        current_source_inputs=capture['source_inputs']
        known_input_differences=[]
        if current_source_inputs['spacing_m']!=original_meta['spacing_m']:
            known_input_differences.append(dict(step='native regional preparation spacing',
                original=original_meta['spacing_m'],actual=current_source_inputs['spacing_m']))
        actual_alignment=result['native_inference']['temporal_alignment']
        if actual_alignment['previous_valid_fraction']!=original_meta['alignment']['previous_valid_fraction']:
            known_input_differences.append(dict(step='previous temporal validity footprint',
                original=original_meta['alignment']['previous_valid_fraction'],actual=actual_alignment['previous_valid_fraction'],
                colour_warp_arithmetic_affected_by_mask=False,
                note='RGB is warped directly; validity affects footprint and required-support check, not mask-normalized colour interpolation.'))
        saved_phase=evaluator.save(result)
        stored=_save_arrays(root/'identity_probability_layers'/run/f'frame_{index:03d}',capture['stored'])
        checks=probability_check['comparisons']
        first_stored=next((name for name in checks if checks[name]['bit_identical'] is not True),None)
        all_pass=label_check['bit_identical'] and probability_check['all_bit_identical']
        record=dict(contract=CONTRACT,complete=True,created_utc=utc_now(),run=run,
            original_studio_frame_index=index,source_work_index=row['work_index'],elapsed_h=row['elapsed_h'],
            source_sha256=digest,previous_source_sha256=row['previous_source_sha256'],identity_plan=plan_descriptor,
            own_history_inventory=plan['runs'][run]['source_inventory'],phase_receipt=saved_phase,
            labels=label_check,stored_probabilities=probability_check,generated_probability_files=stored,
            source_inputs=current_source_inputs,known_input_differences=known_input_differences,
            trace_arrays=capture['trace_arrays'],first_differing_stored_array=first_stored,
            earliest_known_input_difference=None if not known_input_differences else known_input_differences[0],
            causal_attribution='Only recorded input conventions and compared saved arrays are proven. Historical intermediate arrays were not all saved; the isolated cause cannot be inferred from final labels alone.',
            cpu_only=True,internal_threads=1,all_pass=bool(all_pass),training=False,annotations_read=False,
            mass_computed=False,policy_a_production_allowed=False,seconds=time.perf_counter()-started)
        atomic_json(receipt_path,record);descriptor=asset(receipt_path);receipts.append(descriptor)
        print(json.dumps(dict(event='cpu_identity_frame',run=run,studio_frame_index=index,
            source_work_index=row['work_index'],all_pass=all_pass,labels_changed=label_check['changed_elements'],
            first_probability_difference=first_stored,seconds=record['seconds'],receipt=descriptor)),flush=True)
        if not all_pass:
            atomic_json(root/'identity_stop.json',dict(contract=CONTRACT,created_utc=utc_now(),stop=True,
                stop_new_work=True,production_allowed=False,reason='CPU own-history bit identity failed',
                failed_frame=descriptor,run=run,original_studio_frame_index=index,
                first_differing_stored_array=first_stored,earliest_known_input_difference=record['earliest_known_input_difference']))
            return dict(run=run,complete=False,stopped=True,receipts=receipts,failed_frame=descriptor)
    return dict(run=run,complete=True,receipts=receipts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('bind','pilot','all39'),required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--source-plan');parser.add_argument('--identity-inventory')
    parser.add_argument('--plan')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args(argv);root=Path(args.root)
    if args.stage=='bind':
        print(json.dumps(bind(args.source_plan,args.identity_inventory,root)));return 0
    plan_descriptor=asset(args.plan)
    if (root/'identity_stop.json').exists():raise RuntimeError('Identity stop preserved; no retry allowed')
    output=root/(args.stage+'_stage.json')
    if output.exists():raise FileExistsError('Identity stage already exists')
    runs=[('ac19',1),('ac18',5)] if args.stage=='pilot' else [(run,1) for run in sorted(bound_json(plan_descriptor)['runs'])]
    if args.stage=='all39':
        previous=bound_json(asset(root/'pilot_stage.json'))
        if previous['complete'] is not True:raise ValueError('Both pilot sequences must pass before all39')
    if not 1<=args.workers<=24:raise ValueError('CPU workers must be bounded1..24')
    results=[]
    if args.stage=='pilot' or args.workers==1:
        for run,last in runs:
            result=run_sequence(plan_descriptor,root,run,last);results.append(result)
            if result['complete'] is not True:break
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(run_sequence,plan_descriptor,str(root),run,last):run for run,last in runs}
            for future in as_completed(futures):
                result=future.result();results.append(result)
                if result['complete'] is not True:
                    for pending in futures:pending.cancel()
                    break
    complete=len(results)==len(runs) and all(row['complete'] for row in results)
    atomic_json(output,dict(contract=CONTRACT,created_utc=utc_now(),identity_plan=plan_descriptor,
        stage=args.stage,complete=complete,all_39_complete=bool(complete and args.stage=='all39'),
        cpu_only=True,labels_bit_identical=complete,stored_probabilities_bit_identical=complete,
        results=results,training=False,mass_computed=False,policy_a_production_allowed=False))
    return 0 if complete else 2


if __name__=='__main__':raise SystemExit(main())
