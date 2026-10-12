"""Read-only bottom-strip timing and source-bound old-map L1 contact diagnostics.

No labels, models, maps, radiometry or mass inputs are written. The transported
protocol L1 point is a diagnostic proxy, not a photographed injector certificate.
"""
from __future__ import annotations
import argparse,csv,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from darsia.fluidflower.project import sha256_file

CONTRACT='ffac-rig-bottom-strip-camera-timing-v1'
BLIND_RUNS=('ac30','ac49','ac54')
TARGETS=(.25,.75,1.,2.25,4.,6.,8.,12.,24.,48.,72.,96.)

def asset(path):
    path=Path(path);return dict(path=str(path),sha256=sha256_file(path))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def read_bound(value,base=None):
    path=Path(value['path'])
    if not path.is_absolute() and base is not None:path=Path(base)/path
    if sha256_file(path)!=value['sha256']:raise ValueError('Report source changed: '+str(path))
    return read(path)

def group_for_run(run):
    number=int(run.removeprefix('ac'))
    if number not in range(17,62):raise ValueError('Expected FF_AC current run')
    return 'AC17-38' if number<=38 else 'AC39-61'

def protocol_l1_lower_edge_contact(labels,point_reference_xy,homography):
    """Use the exact seed, 8-connectivity, and mask-defined column terminal rows.

    No seed search, radius, predicted ROI or absolute-position correction is
    introduced. If the seed has no mapped CO2, component contact is unknown.
    Late components may contain both injection plumes after they merge.
    """
    labels=np.asarray(labels);point=np.asarray(point_reference_xy,float);h=np.asarray(homography,float)
    if (labels.dtype!=np.uint8 or labels.ndim!=2 or not np.isin(labels,(0,1,2,255)).all()
            or point.shape!=(2,) or h.shape!=(3,3) or not np.isfinite(point).all()
            or not np.isfinite(h).all()):raise ValueError('Source-bound categorical grid/point/H required')
    q=h@np.r_[point,1.]
    if abs(q[2])<1e-12:raise ValueError('Protocol seed maps to infinity')
    xy=q[:2]/q[2];x,y=np.rint(xy).astype(int);height,width=labels.shape
    result=dict(protocol_L1_native_xy=xy.tolist(),protocol_L1_nearest_pixel_xy=[int(x),int(y)],
        protocol_L1_component_connectivity=8,protocol_L1_lower_edge_contact=None,
        protocol_L1_component_pixels=None,protocol_L1_lower_edge_contact_pixels=None,
        protocol_L1_min_column_clearance_px=None,
        photo_verified_L1_lower_edge_contact=None,
        photo_verified_L1_unavailable_reason='Protocol/provisional reference point has no independent photographed injector-position certificate.',
        lower_edge_definition='Bottommost resolved parent-map pixel in each occupied column, after its source-bound photo mask; no predicted ROI.')
    if not(0<=x<width and 0<=y<height):
        return dict(result,protocol_L1_component_status='protocol_point_outside_old_grid')
    if labels[y,x]==255:return dict(result,protocol_L1_component_status='protocol_point_unmapped_in_old_map')
    if labels[y,x] not in (1,2):return dict(result,protocol_L1_component_status='no_CO2_at_exact_protocol_point; component unidentified')
    co2=np.isin(labels,(1,2)).astype(np.uint8)
    _,components=cv2.connectedComponents(co2,connectivity=8,ltype=cv2.CV_32S)
    component=components[y,x];selected=components==component
    domain=labels!=255;occupied=domain.any(axis=0);columns=np.flatnonzero(occupied)
    bottom=height-1-domain[::-1].argmax(axis=0)
    touches=int(np.count_nonzero(components[bottom[columns],columns]==component))
    yy,xx=np.nonzero(selected)
    return dict(result,protocol_L1_component_status='identified_exact_protocol_point',
        protocol_L1_lower_edge_contact=touches>0,protocol_L1_component_pixels=len(xx),
        protocol_L1_lower_edge_contact_pixels=touches,
        protocol_L1_min_column_clearance_px=int(np.min(bottom[xx]-yy)))

def timing_summary(rows):
    """Keep actual camera times and a strict >0 numerical onset distinct."""
    summary=[]
    for run in sorted({r['run'] for r in rows}):
        values=sorted((r for r in rows if r['run']==run),key=lambda r:(r['elapsed_h'],r['frame_index']))
        onset=next((r for r in values if r['bottom_CO2_cm2']>0),None)
        contact=next(iter(sorted((r for r in values if r['protocol_L1_lower_edge_contact'] is True),
                            key=lambda r:(r['old_actual_camera_h'],r['old_frame_index']))),None)
        negatives=[r for r in values if r['elapsed_h']<0]
        summary.append(dict(run=run,camera_group=group_for_run(run),frame_count=len(values),
            first_nonzero_rule='Recorded bottom added-strip CO2 cm2 > 0; no tolerance/floor or zero forcing',
            first_bottom_CO2_frame_index=None if onset is None else onset['frame_index'],
            first_bottom_CO2_camera_h=None if onset is None else onset['elapsed_h'],
            first_bottom_CO2_source_sha256=None if onset is None else onset['source_sha256'],
            first_bottom_CO2_cm2=None if onset is None else onset['bottom_CO2_cm2'],
            first_protocol_L1_old_edge_frame_index=None if contact is None else contact['old_frame_index'],
            first_protocol_L1_old_edge_camera_h=None if contact is None else contact['old_actual_camera_h'],
            first_protocol_L1_old_edge_source_sha256=None if contact is None else contact['old_source_sha256'],
            bottom_onset_minus_protocol_L1_contact_h=(None if onset is None or contact is None
                else onset['elapsed_h']-contact['old_actual_camera_h']),
            first_photo_verified_L1_old_edge_camera_h=None,
            photo_verified_L1_unavailable_reason='No independent photographed injector-position certificate; protocol proxy is reported separately.',
            true_pre_injection_frames=len(negatives),
            true_pre_injection_bottom_CO2_all_zero=None if not negatives else all(r['bottom_CO2_cm2']==0 for r in negatives),
            blind_review_seal_required=run in BLIND_RUNS,
            interpretation='Temporal agreement is diagnostic, not proof of real CO2; the extrapolated lighting correction can also change after injection.'))
    return summary

def matched_group_rows(rows,targets=TARGETS):
    tables=[]
    groups={'AC17-38':sorted({r['run'] for r in rows if group_for_run(r['run'])=='AC17-38'}),
            'AC39-61':sorted({r['run'] for r in rows if group_for_run(r['run'])=='AC39-61'})}
    for group,runs in groups.items():
        for target in targets:
            chosen=[min((r for r in rows if r['run']==run),key=lambda r:(abs(r['elapsed_h']-target),r['frame_index'])) for run in runs]
            if not chosen:continue
            for row in chosen:
                tables.append(dict(row,target_h=float(target),camera_offset_h=row['elapsed_h']-target,
                    camera_group=group,original_photo_or_light_fit_validated_by_group=False))
    return tables

def _csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        fields=list(rows[0]) if rows else ['run']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def group_summary_rows(matched):
    """Aggregate only the supplied rows; public callers must first redact."""
    result=[]
    for group in ('AC17-38','AC39-61'):
        for target in TARGETS:
            values=[r for r in matched if r['camera_group']==group and r['target_h']==target]
            if not values:continue
            result.append(dict(camera_group=group,target_h=target,runs=len(values),
                matched_actual_camera_h=[r['elapsed_h'] for r in values],
                bottom_CO2_mean_cm2=float(np.mean([r['bottom_CO2_cm2'] for r in values])),
                bottom_CO2_median_cm2=float(np.median([r['bottom_CO2_cm2'] for r in values])),
                original_photo_or_radiometry_validated_by_group=False))
    return result


def public_rows(rows):
    """Withhold each blind run, including its contribution to group aggregates."""
    return [r for r in rows if r['run'] not in BLIND_RUNS]

def report_bottom_strip(candidate,output,*,workers=8,require_complete=True):
    """Write public redacted tables and restricted complete source evidence.

    The return value is the public receipt. Complete per-run records for the
    blind runs occur only under restricted_internal and are not returned.
    """
    root,out=Path(candidate),Path(output)
    if out.exists():raise FileExistsError('Use a new report folder')
    if not 1<=workers<=24:raise ValueError('Bounded report workers required')
    seal_path=root/'rig_camera_candidate_receipt.json'
    plan_path=root/'rig_camera_inference_plan.json'
    if not plan_path.exists():plan_path=root/'rig_camera_probe_plan.json'
    plan=read(plan_path)
    seal=None
    if require_complete:
        seal=read(seal_path)
        qa=read_bound(seal['qa'])
        if not seal.get('complete') or not qa.get('all_pass') or len(qa['rows'])!=864:
            raise ValueError('Bottom report needs completed source-bound 43/864 candidate')
    parent=Path(plan['upstream_plan']['binding']['parent'])
    profile_path=parent/'phase_optical_regions.json';profile=read(profile_path);rule=profile['gas_barrier_rule_b']
    barrier=read_bound(rule['manifest'],parent);derivation=read_bound(rule['barrier_derivation'],parent)
    if rule['barrier_derivation']['sha256']!=barrier['barrier_derivation_sha256']:
        raise ValueError('Protocol point and frozen source geometry differ')
    points=[p for p in derivation['reference_topology']['injectors'] if p['name']=='L1']
    if len(points)!=1:raise ValueError('Unique frozen protocol L1 point required')
    l1=points[0]['xy_px'];old_geometry={(r['run'],r['frame_index']):r for r in barrier['frames']}
    paths=sorted((root/'rig_camera_frames').glob('*/frame_*.json'))
    if require_complete and (len(paths)!=864 or len({p.parent.name for p in paths})!=43):
        raise ValueError('Exact 43/864 committed frame receipts required')
    native_by_run={};camera_by_run={};native_assets={};camera_assets={}
    for run in sorted({p.parent.name for p in paths}):
        first=read(next(p for p in paths if p.parent.name==run));source=first['binding']['source']
        old_row=old_geometry[(run,0)]
        native_assets[run]=dict(path=old_row['native_manifest_path'],sha256=old_row['native_manifest_sha256'])
        native_by_run[run]=read_bound(native_assets[run]);camera_assets[run]=source['camera_run']
        camera_by_run[run]=read_bound(camera_assets[run])
    cv2.setNumThreads(1)
    def measure(path):
        receipt=read(path)
        if not receipt.get('complete'):raise ValueError('Incomplete frame receipt')
        source=receipt['binding']['source'];run=source['run'];index=source['frame_index'];qa=receipt['qa']
        if (qa['source_sha256']!=source['source_sha256'] or qa['elapsed_h']!=source['elapsed_h']
                or qa['frame_index']!=index):raise ValueError('Bottom evidence belongs to another source/time')
        sides=qa['added_strip_by_side']
        if sides['contract']!='ffac-rig-added-strip-side-qa-v1':raise ValueError('Unknown side measurement')
        b=sides['sides']['bottom']
        old_index=index if source['source_changed'] else source['parent_source_frame_index']
        old_geo=old_geometry[(run,old_index)];native=native_by_run[run]['frame_sources'][old_index]
        audit=[a for a in camera_by_run[run]['old_timeline_camera_audit'] if a['frame_index']==old_index]
        if len(audit)!=1 or audit[0]['source_sha256']!=native['source_sha256']:
            raise ValueError('Old map has no exact source-bound camera-clock audit')
        old_path=parent/'barrier_rule_b_frames'/run/f'frame_{old_index:03d}.json';old=read(old_path)
        if old['binding']['components']['sha256']!=old_geo['components']['sha256']:
            raise ValueError('Old map source geometry differs from frozen Rule B row')
        old_asset=old['labels_map']
        if sha256_file(old_asset['path'])!=old_asset['sha256']:raise ValueError('Old categorical map changed')
        with Image.open(old_asset['path']) as im:labels=np.asarray(im).copy()
        photo_asset=dict(path=native['v5_photo_sand_mask_path'],sha256=native['v5_photo_sand_mask_sha256'])
        if sha256_file(photo_asset['path'])!=photo_asset['sha256']:raise ValueError('Old photo support changed')
        photo=np.load(photo_asset['path'],allow_pickle=False)
        if photo.dtype!=bool or photo.shape!=labels.shape:raise ValueError('Old photo support grid mismatch')
        labels[~photo]=255
        diagnostic=protocol_l1_lower_edge_contact(labels,l1,old_geo['H_ac28_to_frame_native'])
        return dict(run=run,frame_index=index,elapsed_h=source['elapsed_h'],source_sha256=source['source_sha256'],
            source_changed=source['source_changed'],camera_group=group_for_run(run),
            bottom_added_domain_cm2=b['added_domain_cm2'],bottom_CO2_cm2=b['co2_cm2'],
            bottom_aqueous_cm2=b['aqueous_cm2'],bottom_gas_cm2=b['gas_cm2'],
            bottom_outer_photo_extension_CO2_cm2=b['outer_photo_extension_co2_cm2'],
            bottom_internal_support_recovery_CO2_cm2=b['internal_support_recovery_co2_cm2'],
            **{'bottom_CO2_'+k+'_cm2':b['co2_cm2_by_class'][k] for k in ('modelled','filled_observed','filled_unobserved')},
            **{'bottom_domain_'+k+'_cm2':b['added_cm2_by_class'][k] for k in ('modelled','filled_observed','filled_unobserved')},
            original_light_fit_coverage_verified=False,independent_edge_radiometric_validation=False,
            group_is_independent_radiometric_certificate=False,
            old_frame_index=old_index,old_source_sha256=native['source_sha256'],old_actual_camera_h=audit[0]['camera_h'],
            source_pair_same_photograph=source['source_sha256']==native['source_sha256'],
            changed_source_qualification='Old L1 proxy is measured on its own OLD source/time/H; not paired-pixel correspondence to new acquisition' if source['source_changed'] else 'Same retained acquisition',
            old_labels_asset=old_asset,old_photo_support_asset=photo_asset,
            old_frame_receipt=asset(old_path),new_frame_receipt=asset(path),
            **diagnostic,blind_review_seal_required=run in BLIND_RUNS)
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(measure,paths))
    summaries=timing_summary(rows);matched=matched_group_rows(rows)
    out.mkdir(parents=True);restricted=out/'restricted_internal';restricted.mkdir()
    names=('bottom_strip_all_frames.csv','bottom_onset_and_protocol_L1_contact.csv',
           'bottom_strip_matched_camera_times.csv','bottom_camera_group_summary.csv')
    group_summary=group_summary_rows(matched)
    for name,values in zip(names,(rows,summaries,matched,group_summary)):_csv(restricted/name,values)
    result=dict(contract=CONTRACT,complete_candidate_required=require_complete,candidate_seal=None if seal is None else asset(seal_path),
        inference_plan=asset(plan_path),frame_count=len(rows),run_count=len(summaries),
        protocol_L1_reference=points[0],protocol_point_photo_verified=False,
        frozen_barrier_derivation=rule['barrier_derivation'],frozen_barrier_manifest=rule['manifest'],
        parent_profile=asset(profile_path),parent_native_manifests=native_assets,camera_run_evidence=camera_assets,
        onset_rule='Numerical bottom added-strip CO2 cm2 > 0, before and after injection retained',
        protocol_proxy_rule='Exact rounded transported protocol L1 point; CO2 8-connected component; any mask-defined column bottom terminal pixel; no invented search disk/ROI',
        photo_verified_L1_arrival_available=False,
        qualifications=[
            'Bottom timing alone does not establish real CO2: the correction forecast may vary after injection.',
            'The protocol L1 seed is source-bound but lacks independent photographed injector-position approval; actual L1 arrival remains null.',
            'Late connected CO2 may join both injection plumes.',
            'AC17-38/AC39-61 are camera/run-number comparison groups, not independent radiometric calibration certificates.',
            'Modelled records finite RAW/feature support, not validated light-field extrapolation.',
            'Changed acquisitions retain separate OLD camera time, SHA and geometry for the old-map contact proxy.',
            'Areas use provisional relative AC28 scale; actual rig width remains unmeasured.'],
        per_run_timing=summaries,camera_group_summaries=group_summary,
        blind_runs_not_for_display_before_label_seal=list(BLIND_RUNS),automatic_stdout_excludes_all_frame_values=True,
        restricted_results_not_for_display_before_label_seal=True,
        data_tables={name:asset(restricted/name) for name in names},
        code=asset(__file__),maps_models_labels_sessions_modified=False,inference=False,mass_analysis=False,scientific_acceptance=False)
    full_receipt=restricted/'bottom_strip_timing_receipt.json'
    full_receipt.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    visible=public_rows(rows);visible_summaries=public_rows(summaries);visible_matched=public_rows(matched)
    visible_groups=group_summary_rows(visible_matched)
    public_names=tuple(name.replace('.csv','_public.csv') for name in names)
    for name,values in zip(public_names,(visible,visible_summaries,visible_matched,visible_groups)):_csv(out/name,values)
    public=dict(result,per_run_timing=visible_summaries,camera_group_summaries=visible_groups,
        parent_native_manifests={k:v for k,v in native_assets.items() if k not in BLIND_RUNS},
        camera_run_evidence={k:v for k,v in camera_assets.items() if k not in BLIND_RUNS},
        redacted_blind_frame_count=len(rows)-len(visible),redacted_blind_run_count=len(summaries)-len(visible_summaries),
        public_group_summary_excludes_blind_runs=True,
        restricted_receipt=asset(full_receipt),
        data_tables={name:asset(out/name) for name in public_names})
    public_receipt=out/'bottom_strip_timing_receipt.json'
    public_receipt.write_text(json.dumps(public,indent=2,allow_nan=False),encoding='utf-8')
    return dict(public,receipt=asset(public_receipt))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--allow-partial',action='store_true',help='Diagnostic partial receipts only; never a completed 43/864 report')
    args=parser.parse_args();r=report_bottom_strip(args.candidate,args.output,workers=args.workers,require_complete=not args.allow_partial)
    print(json.dumps(dict(output=args.output,frames=r['frame_count'],runs=r['run_count'],private_blind_results_not_displayed=True)))
if __name__=='__main__':main()
