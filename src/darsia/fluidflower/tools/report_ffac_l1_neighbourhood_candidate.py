"""Read-only technical, area, floor-support and CPU QA for the stored-map L1 rule.

No annotations or scorers are read. Hidden AC26/55 maps are verified privately
and return pass/fail only; public totals exclude their contributions. The
separate examples report supplies authorized in-sample B-T2 scoring and figures.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
from pathlib import Path
import time
import cv2
import numpy as np

from . import run_ffac_l1_neighbourhood_candidate as runner
from . import report_ffac_t2_floor_candidate as oldqa
from . import report_ffac_t2_l1zone_candidate as supportqa
from darsia.fluidflower.phase_pixel_area import load_rig_classes
from darsia.fluidflower.phase_ffac_floor_contact import prepare_bound_floor_zone, apply_floor_contact
from darsia.fluidflower.phase_l1_neighbourhood_gas import frozen_rule_b_forbidden_mask,select_frozen_base_rule_b

CONTRACT="ffac-l1-neighbourhood-floor-technical-qa-v1"
HIDDEN=runner.HIDDEN


def public(rows):
    return [row for row in rows if row['run'].casefold() not in HIDDEN]


def mask(asset, shape, source_sha, index, contract):
    if (asset.get('contract')!=contract or asset.get('source_sha256')!=source_sha
            or asset.get('frame_index')!=index or tuple(asset.get('shape_hw',()))!=tuple(shape)
            or asset.get('storage')!='packbits-npy'):
        raise ValueError('L1 QA flag lost its exact source/grid/contract binding')
    packed=np.load(runner.bound_path(asset),allow_pickle=False)
    if packed.dtype!=np.uint8 or packed.ndim!=1 or packed.size!=(np.prod(shape)+7)//8:
        raise ValueError('L1 QA flag has invalid packbits storage')
    return np.unpackbits(packed,count=np.prod(shape)).reshape(shape).astype(bool)


def literal_rule(before, H, elapsed, barrier, forbidden, *, gas_fraction_threshold=.3):
    """Independent literal section1 reconstruction from the unmodified base."""
    if type(gas_fraction_threshold) not in (int, float) or gas_fraction_threshold not in (.3, .5):
        raise ValueError('QA requires an approved .3 or .5 neighbourhood threshold')
    H=np.asarray(H,np.float64)
    if H.shape!=(3,3) or not np.isfinite(H).all() or abs(np.linalg.det(H))<1e-15:
        raise ValueError('QA source homography is invalid')
    q=H@np.asarray([2786.,2897.,1.]);point=q[:2]/q[2]+[416.,496.]
    yy,xx=np.ogrid[:before.shape[0],:before.shape[1]]
    disc=(xx-point[0])**2+(yy-point[1])**2<=937.5**2
    active=elapsed>=.935
    flag=np.zeros(before.shape,bool)
    if active:
        fraction=cv2.blur((before==2).astype(np.float32),(51,51))
        flag=(before==1)&disc&(fraction>=gas_fraction_threshold)&~barrier&~forbidden
    revised=before.copy();revised[flag]=2
    return revised,flag,disc,point,active


def identities(before, after, gasflag, fill, expectedflag, disc, barrier, forbidden, active):
    expected=before.copy();expected[gasflag]=2;expected[fill]=1
    return dict(
        literal_neighbourhood_flag_differing_pixels=int(np.count_nonzero(gasflag!=expectedflag)),
        final_labels_differing_from_flagged_changes=int(np.count_nonzero(after!=expected)),
        inactive_neighbourhood_pixels=int(gasflag.sum()) if not active else 0,
        neighbourhood_not_aqueous_to_gas_pixels=int(np.count_nonzero(gasflag&((before!=1)|(after!=2)))),
        neighbourhood_outside_zone_pixels=int(np.count_nonzero(gasflag&~disc)),
        neighbourhood_in_barrier_pixels=int(np.count_nonzero(gasflag&barrier)),
        neighbourhood_in_rule_b_forbidden_pixels=int(np.count_nonzero(gasflag&forbidden)),
        final_co2_union_differing_pixels=int(np.count_nonzero(np.isin(after,(1,2))!=(np.isin(before,(1,2))|fill))),
        final_outside_partition_differing_pixels=int(np.count_nonzero((before==255)!=(after==255))),
        no_co2_changed_without_floor_flag_pixels=int(np.count_nonzero((before==0)&(after!=0)&~fill)),
        flag_overlap_pixels=int(np.count_nonzero(gasflag&fill)),
    )


def floor_statistics(before, after, fill, area, categories, disc, floor_zone, elapsed):
    q=oldqa.floor_qa(before,after,fill,area,elapsed)
    row,col=floor_zone.base_canvas_y,np.arange(after.shape[1])
    in_canvas=floor_zone.floor_in_canvas&(row>=floor_zone.y0)&(row<floor_zone.y1)
    raw=categories['observed_native_photo']|categories['recovered_raw_strips']
    observable=np.zeros(after.shape[1],bool);cols=col[in_canvas]
    observable[cols]=floor_zone.sand[row[cols]-floor_zone.y0,cols]&raw[row[cols],cols]
    denominator=int(observable.sum())
    def detail(m):
        n=int(np.count_nonzero(m.any(0)&observable))
        return dict(pixels=int(m.sum()),area_cm2=float(area[m].sum(dtype=np.float64)/100),
            filled_observable_floor_columns=n,share_of_all_observable_floor_columns=n/denominator if denominator else None)
    q.update(observable_floor_columns=denominator,filled_observable_floor_columns=int(np.count_nonzero(fill.any(0)&observable)),
        by_support={key:detail(fill&m) for key,m in categories.items()},
        by_L1_distance={'within150mm':detail(fill&disc),'beyond150mm':detail(fill&~disc)},
        uncategorised_fill_pixels=int(np.count_nonzero(fill&~np.logical_or.reduce(list(categories.values())))))
    return q


def checks_pass(counters, floor):
    return all(value==0 for value in counters.values()) and all(floor[key]==0 for key in (
        'filled_pixels_not_aqueous','filled_pixels_already_co2_in_base','preinjection_fill_pixels',
        'filled_under_gas_pixels','invalid_aqueous_trigger_columns','discontinuous_fill_columns',
        'co2_union_changes_outside_floor_contact_fill','co2_union_removed_pixels',
        'co2_union_added_without_floor_flag_pixels','uncategorised_fill_pixels'))


def frame_metrics(item, expected_source, *, rule_settings=None):
    start=time.perf_counter()
    row=runner.read_json(runner.bound_path(item['receipt']))
    threshold=.3 if rule_settings is None else rule_settings['gas_fraction_threshold']
    if (row.get('gas_fraction_threshold', .3)!=threshold
            or (rule_settings is not None and row.get('spec_sha256',runner.SPEC_SHA256)!=rule_settings['spec_sha256'])):
        raise ValueError('QA frame threshold/spec differs from its immutable candidate plan')
    if (row.get('contract')!=runner.CONTRACT or row.get('complete') is not True
            or row['binding']['source']!=expected_source
            or row['source_sha256']!=expected_source['source_sha256']
            or row['elapsed_h']!=expected_source['elapsed_h']):
        raise ValueError('QA frame lost its immutable source/time binding')
    parent=runner.read_json(runner.bound_path(row['binding']['parent_receipt']))
    if (row['before_labels']!=parent['labels_map'] or parent['binding']['source']!=expected_source
            or row['rig_domain_revision']!=parent['rig_domain_revision']
            or row['full_extended_scientific']!=parent['full_extended_scientific']
            or row['baseline_extended_scientific']!=parent['baseline_extended_scientific']):
        raise ValueError('QA geometry/area/scientific source differs from the exact base')
    before,after=runner._png(row['before_labels']),runner._png(row['labels_map'])
    if before.dtype!=np.uint8 or after.dtype!=np.uint8 or before.shape!=after.shape or not np.isin(before,(0,1,2,255)).all():
        raise ValueError('QA stored categorical canvas changed')
    shape=before.shape;source_sha=row['source_sha256'];index=row['frame_index']
    gasflag=mask(row['l1_neighbourhood_gas'],shape,source_sha,index,'ffac-l1-neighbourhood-gas-v1')
    fill=mask(row['floor_contact_fill'],shape,source_sha,index,'ffac-t2-floor-contact-v1')
    area=oldqa.area_map(row['rig_domain_revision']['area_map'])
    original=runner.read_json(runner.bound_path(parent['source_producer_receipt']))
    decision,selection_audit=select_frozen_base_rule_b(parent,original)
    if (row['rule_b_geometry']!=original['rule_b_geometry'] or row['rule_b']!=decision
            or row.get('frozen_rule_b_source_selection')!=selection_audit):
        raise ValueError('QA changed frozen RuleB decision/geometry')
    barriers=runner._classes(row['rule_b_geometry']['asset'])
    forbidden,_=frozen_rule_b_forbidden_mask(barriers['open_components'],barriers['barrier_mask'],row['rule_b'])
    pre_floor,expectedflag,disc,point,active=literal_rule(before,expected_source['rig_geometry']['H_ac28_to_frame_native'],
        row['elapsed_h'],barriers['barrier_mask'],forbidden,gas_fraction_threshold=threshold)
    if not np.allclose(point,row['l1_point_xy'],rtol=0,atol=1e-8) or row['l1_rule_active'] is not active:
        raise ValueError('QA L1 point or activation changed from per-frame camera geometry')
    counters=identities(before,after,gasflag,fill,expectedflag,disc,barriers['barrier_mask'],forbidden,active)
    classes,_=load_rig_classes(row['rig_domain_revision']);classes=runner.floor_contact_classes(row,classes)
    centre=supportqa.native_photo_centre(row);categories,raw=supportqa.support_categories(classes,centre)
    observed=raw&classes['rig_domain']&~classes['hardware_excluded']
    filled_support=(classes['filled_observed']|classes['filled_unobserved'])&classes['rig_domain']&~classes['hardware_excluded']
    floor_zone,_=prepare_bound_floor_zone(row,classes,area,observed)
    reconstructed=apply_floor_contact(pre_floor,elapsed_hours=row['elapsed_h'],zone=floor_zone,
        observed_support=observed,filled_support=filled_support,unresolved=~np.isin(pre_floor,(0,1,2)),pixel_area_mm2=area)
    counters['unchanged_floor_rule_flag_differing_pixels']=int(np.count_nonzero(reconstructed.floor_contact_fill!=fill))
    counters['unchanged_floor_rule_final_label_differing_pixels']=int(np.count_nonzero(reconstructed.labels!=after))
    trigger_asset=row['floor_trigger_canvas_y']
    if (trigger_asset['source_sha256']!=source_sha or trigger_asset['frame_index']!=index
            or trigger_asset['shape_hw']!=[shape[1]] or trigger_asset['storage']!='npy'):
        raise ValueError('QA floor trigger lost its exact source/grid binding')
    trigger=np.load(runner.bound_path(trigger_asset),allow_pickle=False)
    if trigger.dtype!=np.int32 or trigger.shape!=(shape[1],):
        raise ValueError('QA floor trigger has invalid representation')
    counters['floor_trigger_differing_columns']=int(np.count_nonzero(trigger!=reconstructed.trigger_canvas_y))
    from . import run_ffac_rig_camera_inference as producer
    top=producer.load_array(expected_source['rig_geometry']['frame_photo_top_keep'])
    counters['top_clip_would_change_pixels']=int(np.count_nonzero((pre_floor==2)&~top))
    floor=floor_statistics(before,after,fill,area,categories,disc,floor_zone,row['elapsed_h'])
    passed=checks_pass(counters,floor)
    if row['run'].casefold() in HIDDEN:
        return dict(run=row['run'].upper(),hidden=True,all_pass=passed)
    return dict(run=row['run'].upper(),frame_index=index,source_sha256=source_sha,elapsed_h=row['elapsed_h'],
        all_pass=passed,identities=counters,floor=floor,l1_point_xy=point.tolist(),rule_active=active,
        gas_fraction_threshold=threshold,
        gas_base_cm2=float(area[before==2].sum(dtype=np.float64)/100),gas_new_cm2=float(area[after==2].sum(dtype=np.float64)/100),
        l1_neighbourhood_gas_cm2=float(area[gasflag].sum(dtype=np.float64)/100),
        l1_neighbourhood_gas_pixels=int(gasflag.sum()),changed_pixels=int(np.count_nonzero(before!=after)),
        co2_base_cm2=float(area[np.isin(before,(1,2))].sum(dtype=np.float64)/100),
        co2_new_cm2=float(area[np.isin(after,(1,2))].sum(dtype=np.float64)/100),
        cost=dict(production_cpu_s=row['timing']['cpu_s'],production_wall_s=row['timing']['wall_s'],qa_wall_s=time.perf_counter()-start),
        provenance=dict(candidate_frame_receipt=item['receipt'],parent_frame_receipt=row['binding']['parent_receipt']))


def run_group(items, sources, rule_settings):
    runner.worker_initializer()
    source_index={(s['run'],s['frame_index']):s for s in sources}
    rows=[];hidden_pass=True;run=None
    for item in items:
        run=item['run'];row=frame_metrics(item,source_index[(run,item['frame_index'])],rule_settings=rule_settings)
        if row.get('hidden'):hidden_pass &= row['all_pass']
        else:rows.append(row)
    return dict(run=run.upper(),all_pass=all(r['all_pass'] for r in rows) and hidden_pass,
        **(dict(hidden=True) if run.casefold() in HIDDEN else dict(rows=rows)))


def target_times(rows):
    groups=defaultdict(list)
    for row in public(rows):groups[row['run']].append(row)
    result=[]
    for run,frames in sorted(groups.items()):
        for target in (1.25,2.25,4.,6.):
            row=min(frames,key=lambda r:(abs(r['elapsed_h']-target),r['frame_index']))
            result.append(dict(run=run,frame_index=row['frame_index'],target_elapsed_h=target,
                actual_elapsed_h=row['elapsed_h'],time_distance_seconds=abs(row['elapsed_h']-target)*3600,
                gas_base_cm2=row['gas_base_cm2'],gas_new_cm2=row['gas_new_cm2'],l1_neighbourhood_gas_cm2=row['l1_neighbourhood_gas_cm2']))
    return result


def flatten(rows):
    result=[]
    for row in public(rows):
        item={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        item.update({'identity_'+k:v for k,v in row['identities'].items()})
        item.update({'floor_'+k:v for k,v in row['floor'].items() if not isinstance(v,dict)})
        for group in ('by_support','by_L1_distance'):
            for name,values in row['floor'][group].items():
                item.update({'floor_'+group+'_'+name+'_'+k:v for k,v in values.items()})
        item.update(row['cost']);result.append(item)
    return result


def write_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def run(args):
    root,plan,sources=runner.candidate_state(args.candidate)
    rule_settings=runner.plan_rule_settings(plan)
    available=oldqa.frame_receipts(root,'rig_camera_frames')
    if not args.allow_partial and (len(available)!=864 or len({key[0] for key in available})!=43):
        raise ValueError('Technical QA needs complete43/864, or explicit --allow-partial')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a new immutable technical QA directory')
    output.mkdir(parents=True,exist_ok=False)
    runner.below_normal_priority()
    source_code={name:runner.artifact(path) for name,path in
        (('report',__file__),('floor_qa',oldqa.__file__),('source_support',supportqa.__file__))}
    snapshots={}
    for name,descriptor in source_code.items():
        destination=output/'code_snapshot'/(name+'.py')
        runner.copy_file_verified(descriptor['path'],destination)
        snapshots[name]=runner.artifact(destination)
    report_code=snapshots['report']
    report_dependencies={name:descriptor for name,descriptor in snapshots.items() if name!='report'}
    import os,psutil
    cpus=len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(),'cpu_affinity') else (os.cpu_count() or 1)
    groups=defaultdict(list)
    for (run,index),entry in available.items():
        groups[run].append(dict(run=run,frame_index=index,receipt=runner.artifact(entry[0])))
    tasks={run:sorted(items,key=lambda r:r['frame_index']) for run,items in groups.items()}
    admission=runner.resource_admission(args.workers,max(1,len(tasks)),2048,2048,
        logical_cpus=cpus,available_bytes=psutil.virtual_memory().available)
    results=[]
    with ProcessPoolExecutor(max_workers=admission['admitted_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=runner.worker_initializer) as pool:
        futures={pool.submit(run_group,items,[s for s in sources if s['run']==run],rule_settings):run for run,items in tasks.items()}
        for future in as_completed(futures):
            value=future.result();results.append(value)
            runner.atomic_json(output/'runs'/(value['run']+'.json'),value)
    rows=sorted([row for group in results for row in group.get('rows',[])],key=lambda r:(r['run'],r['frame_index']))
    if any(runner.artifact(descriptor['path'])!=descriptor for descriptor in source_code.values()):
        raise ValueError('Report code changed while technical QA was running')
    passed=bool(results) and all(group['all_pass'] for group in results)
    report=dict(contract=CONTRACT,complete=not args.allow_partial,technical_all_pass=passed,
        run_count=len(results),frame_count=len(available),candidate_plan=runner.artifact(root/runner.PLAN),
        rule_settings=rule_settings,
        report_code=report_code,report_code_dependencies=report_dependencies,report_source_code_binding=source_code,resource_admission=admission,public_rows=rows,
        hidden_runs=[dict(run=group['run'],all_pass=group['all_pass']) for group in results if group.get('hidden')],
        cost_public_only=dict(frames=len(rows),production_cpu_s=sum(row['cost']['production_cpu_s'] for row in rows),
            summed_frame_wall_s=sum(row['cost']['production_wall_s'] for row in rows),
            excludes_hidden_contributions=True,wall_is_sum_of_frame_times_not_batch_elapsed=True),
        mass_eligible=False,bottom_mass_allowed=False,scientific_acceptance=False,
        annotations_read=False,sealed_scorers_invoked=False,regression_acceptance='Pending Claude sealed scorer',
        sections_3b_and_3e=dict(receipt=runner.artifact(args.examples_receipt) if getattr(args,'examples_receipt',None) else None,
            description='Separate authorized in-sample B-T2 report and public AC30/49 figures; no sealed scorer invoked'))
    runner.atomic_json(output/'technical_report.json',report)
    write_csv(output/'per_frame_public.csv',flatten(rows));write_csv(output/'target_times_public.csv',target_times(rows))
    floor=sum(row['floor']['filled_area_cm2'] for row in rows);gas=sum(row['l1_neighbourhood_gas_cm2'] for row in rows)
    text=f"""# L1 neighbourhood + floor contact: technical QA

Technical checks: {'PASS' if passed else 'FAIL'} on {len(results)} runs/{len(available)} frames.
Public rows exclude AC26 and AC55 and all their contributions; these runs return pass/fail only.
{len(rows)} public frames: added gas summed over frames {gas:.6f} cm²; floor fill summed over frames {floor:.6f} cm².
These are sums over images, not a single physical area.
Public production CPU time: {report['cost_public_only']['production_cpu_s']:.3f} s.

The labels are compared to an independent literal 51×51 one-pass reconstruction with gas fraction >= {rule_settings['gas_fraction_threshold']:g}, taken from the immutable candidate plan, and an unchanged floor-rule reapplication. Source/camera time/geometry/approved area assets equal the stored base exactly. Top clipping is checked and not reapplied. All area values use the approved spatial pixel-area map.

Section 3b in-sample box scoring and section 3e figures are separate from this report. Claude alone performs the sealed regression and B-T3 scoring. This candidate is for Studio review, is not the default, and is not approved for mass analysis. The bottom strip stays blocked.
"""
    (output/'README.md').write_text(text,encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--examples-receipt',type=Path,help='Link the separately source-bound sections3b/e report receipt')
    a=p.parse_args(argv)
    if a.workers<1:p.error('workers must be positive')
    run(a)


if __name__=='__main__':main()
