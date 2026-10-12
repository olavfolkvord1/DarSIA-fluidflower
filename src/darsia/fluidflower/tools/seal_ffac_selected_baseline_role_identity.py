"""Seal fresh and explicitly equivalent prior CPU identity evidence; no inference."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

SPEC_SHA='62bf2970077c9a270f85fe3f046e48811af70d5e1c1761cd729d6289782bef32'
CONTRACT='ffac-selected-baseline-role-required-identity-seal-v1'
DIRECT_CONTROLS={('ac19',0),('ac19',1),('ac54',0),('ac54',1),('ac54',2)}


def descriptor(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(item):
    raw=Path(item['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Changed metadata: '+item['path'])
    return json.loads(raw)


def write_new(path,record):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as out:json.dump(record,out,indent=2,allow_nan=False);out.write('\n')
    return descriptor(p)


def require_products(row):
    if row.get('all_pass') is not True or row.get('cpu_only') is not True or row.get('internal_threads')!=1:
        raise ValueError('Identity row is not a passed CPU computation')
    if row['labels'].get('bit_identical') is not True:raise ValueError('Final labels differ')
    p=row['stored_probabilities']
    for name in ('g3_pixel.npy','regional_frozen.npy','regional_refit.npy','co2.packbits.npy'):
        if p['comparisons'][name].get('bit_identical') is not True:raise ValueError('Required probability differs: '+name)
    masks=p['extra_masks']
    if masks.get('frozen_gas_required') is not False:raise ValueError('Frozen gas is not audited diagnostic-only')
    for name in ('photo_sand_domain','feature_support','regional_native_domain','flatfield_valid',
                 'feature_eligible','optical_presence_ge_half','frozen_unresolved'):
        if masks['comparisons'][name].get('required') is not True or masks['comparisons'][name].get('bit_identical') is not True:
            raise ValueError('Required mask differs: '+name)
    inherited=p['inherited_map']
    if inherited.get('computed_from_scientific_images') is not True or inherited.get('archived_labels_used_as_inference_input') is not False:
        raise ValueError('Inherited map was not freshly computed')
    if any(inherited['comparisons'][name].get('bit_identical') is not True for name in ('co2_union','water','unresolved')):
        raise ValueError('Required inherited category differs')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--plan',required=True)
    parser.add_argument('--used-input-audit',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args(argv);root=Path(args.root);dest=Path(args.destination)
    if (root/'identity_stop.json').exists():raise RuntimeError('A required identity stop prevents admission')
    if dest.exists():raise FileExistsError('Composite seal output must be new')
    plan_asset=descriptor(args.plan);plan=read(plan_asset)
    if plan['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong specification')
    audit_asset=descriptor(args.used_input_audit);audit=read(audit_asset)
    if audit.get('complete') is not True or audit.get('full_closure_checked') is not True:
        raise ValueError('Incomplete full used-input audit')
    inventory_asset=plan['phase_campaign_binding']['identity_inventory'];inventory=read(inventory_asset)
    host_asset=plan['phase_campaign_binding']['approved_main_host'];host=read(host_asset)
    controls_asset=descriptor(root/'controls_stage.json');controls=read(controls_asset)
    remaining_asset=descriptor(root/'remaining_stage.json');remaining=read(remaining_asset)
    if not controls.get('complete') or not remaining.get('complete'):raise ValueError('Required fresh stages incomplete')
    if remaining['used_input_audit']!=audit_asset:raise ValueError('Remaining stage used another audit')
    audit_rows={}
    for position,row in enumerate(audit['rows']):
        old=read(row['old_identity_receipt'])
        key=(row['run'],old['original_studio_frame_index'])
        if key in audit_rows:raise ValueError('Duplicate input audit row')
        audit_rows[key]=(position,row,old)
    required={(run,frame['studio_frame_index']):frame for run,state in inventory['runs'].items() for frame in state['frames']}
    if len(required)!=153 or len(inventory['runs'])!=39:raise ValueError('Actual 153-frame/39-run scope required')
    prepared=[]
    for key,frame in sorted(required.items()):
        run,index=key;fresh=root/'identity_frames'/run/f'frame_{index:03d}.json'
        equivalence=None
        if fresh.exists():
            row_asset=descriptor(fresh);row=read(row_asset);origin='fresh'
            if row['identity_plan']!=plan_asset:raise ValueError('Fresh row belongs to another immutable plan')
            if row['phase_execution_host']['hostname']!=host['hostname'] or row['phase_execution_host']['machine']!=host['machine']:
                raise ValueError('Fresh identity ran on an unadmitted host')
        else:
            if key in DIRECT_CONTROLS:raise ValueError('Direct controls cannot be imported')
            if key not in audit_rows:raise ValueError('No unchanged-input evidence for required row')
            position,item,row=audit_rows[key]
            if (run in audit['rerun_runs'] or item.get('all_used_input_arrays_sha256_unchanged') is not True
                    or item.get('full_closure_checked') is not True):
                raise ValueError('A changed-input run must be replayed in full')
            row_asset=item['old_identity_receipt'];origin='imported'
            equivalence=dict(contract='ffac-required-identity-unchanged-input-consumer-equivalence-v1',
                original_receipt=row_asset,original_identity_plan=row['identity_plan'],
                source_sha256=row['source_sha256'],previous_source_sha256=row['previous_source_sha256'],
                new_consumer_identity_plan=plan_asset,used_input_audit=audit_asset,used_input_audit_row_index=position,
                all_used_input_arrays_sha256_unchanged=True,no_new_inference=True,
                original_runtime_and_numerical_producer_preserved=True)
        require_products(row)
        if row['run']!=run or row['original_studio_frame_index']!=index or row['source_sha256']!=frame['source_sha256']:
            raise ValueError('Receipt camera/Studio identity differs from required original frame')
        if row['previous_source_sha256']!=frame['original_previous_source_sha256']:
            raise ValueError('Original own-history predecessor changed')
        prepared.append((key,row_asset,row,origin,equivalence))
    dest.mkdir(parents=True,exist_ok=False)
    rows=[];diagnostics=[]
    for (run,index),row_asset,row,origin,equivalence in prepared:
        equiv_asset=write_new(dest/'imports'/run/f'frame_{index:03d}.json',equivalence) if equivalence else None
        rows.append(dict(run=run,original_studio_frame_index=index,source_sha256=row['source_sha256'],
            previous_source_sha256=row['previous_source_sha256'],origin=origin,receipt=row_asset,import_equivalence=equiv_asset))
        gas=row['frozen_gas']
        diagnostics.append(dict(run=run,original_studio_frame_index=index,source_work_index=row['source_work_index'],
            elapsed_h=row['elapsed_h'],source_sha256=row['source_sha256'],origin=origin,required_identity_passed=True,
            frozen_gas_changed_pixels=gas['changed_pixels'],frozen_gas_added_pixels=gas['added_pixels'],
            frozen_gas_removed_pixels=gas['removed_pixels'],frozen_gas_first_yx=gas['first_yx'],receipt_sha256=row_asset['sha256']))
    with (dest/'identity_frames.csv').open('x',encoding='utf8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    counts={origin:sum(row['origin']==origin for row in rows) for origin in ('fresh','imported')}
    seal=dict(contract=CONTRACT,spec_sha256=SPEC_SHA,created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,all_39_complete=True,required_identity_passed=True,required_unique_frame_count=len(rows),
        runs=sorted(inventory['runs']),identity_plan=plan_asset,identity_inventory=inventory_asset,
        used_input_audit=audit_asset,approved_main_host=host_asset,rows=rows,counts=counts,
        stages={'controls':controls_asset,'remaining':remaining_asset},
        frozen_gas_diagnostic_only=True,frozen_gas_changed_frames=sum(bool(row['frozen_gas_changed_pixels']) for row in diagnostics),
        frozen_gas_summed_changed_pixels=sum(row['frozen_gas_changed_pixels'] for row in diagnostics),
        cpu_only=True,internal_threads=1,old_results_unchanged=True,
        imported_rows_are_not_new_inference=True,policy_a_started=False,mass_computed=False,
        diagnostic_table=descriptor(dest/'identity_frames.csv'),sealer_code=descriptor(__file__))
    seal_asset=write_new(dest/'accepted_identity.json',seal)
    print(json.dumps(seal_asset));return 0


if __name__=='__main__':raise SystemExit(main())
