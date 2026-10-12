"""Measured section 5 tables from completed, immutable rig-candidate receipts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from darsia.fluidflower.phase_ffac_rig_qa import nearest_effect_rows,cohort_statistics
from darsia.fluidflower.project import sha256_file


def artifact(path):
    path=Path(path)
    return dict(path=str(path),sha256=sha256_file(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_asset(asset):
    if sha256_file(asset['path'])!=asset['sha256']:
        raise ValueError('Report source changed: '+asset['path'])
    return read_json(asset['path'])


def physical_area_factors(row):
    """Analytical Jacobian in the actual frame grid, including perspective."""
    params=row['area_parameters']
    forward=np.asarray(params['H_ac28_to_frame_native'],float)
    inverse=np.asarray(params['H_ac28_to_metric'],float) @ np.linalg.inv(forward)
    q=forward @ np.array([2736.,1490.,1.])
    centre=q[:2]/q[2]
    ys=np.full(5472,centre[1])
    xs=np.arange(5472,dtype=float)
    def factor(x,y):
        denominator=inverse[2,0]*x+inverse[2,1]*y+inverse[2,2]
        if np.any(np.abs(denominator)<1e-9): raise ValueError('Area maps to infinity')
        return np.abs(np.linalg.det(inverse)/(denominator**3))
    values=factor(xs,ys)
    return dict(inverse_jacobian_at_reference_centre=float(factor(*centre)),
                centre_native_xy=centre.tolist(),reference_centre_native_xy=[2736.,1490.],
                left_right_native_interval_px=[0,5471],range_y_native=float(centre[1]),
                left_right_min_factor=float(values.min()),left_right_max_factor=float(values.max()),
                left_right_fractional_range=float(values.max()/values.min()-1),
                pixel_area_mm2_fixed_ac28=params['fixed_area_mm2'],
                physical_scale_approved=params['physical_scale_approved'])


def frame_report_tables(rows):
    """Promote already-measured QA fields; changed acquisitions are not interiors."""
    support,interior=[],[]
    for row in sorted(rows,key=lambda item:(item['run'],item['frame_index'])):
        identity={key:row[key] for key in ('run','frame_index','elapsed_h','source_sha256','source_changed')}
        a=row['area_report'];classes=a['support_cm2_by_class'];added=a['added_cm2_by_class']
        hidden=a.get('hidden_chart') or {}
        support.append(dict(identity,rig_domain_cm2=classes['rig_domain'],
            old_fixed_resolved_cm2=a['old_fixed_cm2']['resolved'],
            old_reweighted_resolved_cm2=a['old_reweighted_cm2']['resolved'],
            old_reweighted_geometry_only_eligible=not row['source_changed'],
            old_reweighted_weight_geometry=('new-source geometry applied to old-source map; not geometry-only'
                if row['source_changed'] else 'same retained acquisition geometry'),
            added_domain_cm2=a['added_domain_cm2'],
            **{name+'_cm2':classes[name] for name in ('modelled','filled_observed','filled_unobserved')},
            **{'added_'+name+'_cm2':added[name] for name in ('modelled','filled_observed','filled_unobserved')},
            excluded_hardware_cm2=a['excluded_hardware_cm2'],hidden_chart_cm2=hidden.get('area_cm2'),
            co2_touches_hidden_chart_boundary=hidden.get('co2_touches_boundary'),
            hidden_chart_touching_boundary_cm2=hidden.get('touching_boundary_cm2')))
        difference=row['interior_difference']
        excluded=bool(row['source_changed'])
        counts=difference.get('changed_pixels') or {}
        interior.append(dict(identity,excluded_source_changed=excluded,
            **{name+'_changed_pixels':None if excluded else counts[name]
               for name in ('0_10_mm','10_30_mm','over_30_mm')},
            interior_gt30mm_identical=None if excluded else bool(difference['interior_exact'])))
    return support,interior


def added_strip_time_rows(rows):
    """Keep each measured side/time visible, including nonzero early responses."""
    table=[]
    for row in sorted(rows,key=lambda item:(item['run'],item['frame_index'])):
        evidence=row.get('added_strip_by_side')
        available=evidence is not None
        if available and evidence.get('contract')!='ffac-rig-added-strip-side-qa-v1':
            raise ValueError('Unsupported recorded added-strip QA contract')
        for side in ('top','bottom','left','right'):
            measured=evidence['sides'][side] if available else {}
            result={key:row[key] for key in ('run','frame_index','elapsed_h','source_sha256','source_changed')}
            elapsed=float(row['elapsed_h'])
            if not np.isfinite(elapsed):
                raise ValueError('Added-strip controls require a finite camera-clock time')
            result.update(side=side,measured_side_qa_available=available,
                camera_time_role='pre-injection' if elapsed<0 else 'post-injection',
                true_pre_injection_control=elapsed<0,
                frame0_or_frame1_early_diagnostic=row['frame_index'] in (0,1),
                partition=None if evidence is None else evidence['partition'],
                source_changed_qualification=('New acquisition compared with the frozen old-source footprint; not a same-photograph geometry-only comparison'
                    if row['source_changed'] else 'same retained acquisition'),
                **{key:measured.get(key) for key in ('added_pixels','added_domain_cm2','co2_cm2','aqueous_cm2','gas_cm2',
                    'outer_photo_extension_co2_cm2','internal_support_recovery_co2_cm2')})
            for kind in ('co2','added'):
                values=measured.get(kind+'_cm2_by_class') or {}
                result.update({kind+'_'+name+'_cm2':values.get(name)
                    for name in ('modelled','filled_observed','filled_unobserved')})
            table.append(result)
    summary=[]
    for run in sorted({row['run'] for row in table}):
        for side in ('top','bottom','left','right'):
            values=[row for row in table if row['run']==run and row['side']==side]
            early={row['frame_index']:row for row in values if row['frame_index'] in (0,1)}
            usable=[row for row in values if row['co2_cm2'] is not None]
            early_usable=[row for row in usable if row['frame_index'] in (0,1)]
            negatives=[row for row in usable if row['true_pre_injection_control']]
            negatives_all=[row for row in values if row['true_pre_injection_control']]
            maximum=max(usable,key=lambda row:row['co2_cm2']) if usable else None
            maximum_early=max(early_usable,key=lambda row:row['co2_cm2']) if early_usable else None
            maximum_negative=max(negatives,key=lambda row:row['co2_cm2']) if negatives else None
            summary.append(dict(run=run,side=side,
                frame_count=len(values),measured_frame_count=len(usable),
                **{f'frame{i}_{key}':early.get(i,{}).get(key) for i in (0,1)
                    for key in ('elapsed_h','camera_time_role','source_sha256','co2_cm2','aqueous_cm2','gas_cm2')},
                true_pre_injection_frame_count=len(negatives_all),
                true_pre_injection_measured_frame_count=len(negatives),
                true_pre_injection_all_added_co2_zero=(None if len(negatives)!=len(negatives_all) or not negatives
                    else all(row['co2_cm2']==0 for row in negatives)),
                true_pre_injection_max_added_co2_cm2=None if maximum_negative is None else maximum_negative['co2_cm2'],
                true_pre_injection_max_frame_index=None if maximum_negative is None else maximum_negative['frame_index'],
                true_pre_injection_max_elapsed_h=None if maximum_negative is None else maximum_negative['elapsed_h'],
                early_max_added_co2_cm2=None if maximum_early is None else maximum_early['co2_cm2'],
                early_max_frame_index=None if maximum_early is None else maximum_early['frame_index'],
                all_time_max_added_co2_cm2=None if maximum is None else maximum['co2_cm2'],
                all_time_max_frame_index=None if maximum is None else maximum['frame_index'],
                all_time_max_elapsed_h=None if maximum is None else maximum['elapsed_h'],
                all_time_max_source_sha256=None if maximum is None else maximum['source_sha256'],
                interpretation='Only camera elapsed_h < 0 is a negative control; frame index does not define injection timing. Frame0 can be the same acquisition as B and is then a plumbing check, not an independent negative. Post-injection frame1 is reported without a zero requirement.'))
    return table,summary


def source_wall_residual_rows(certificate,geometry_rows):
    """Bound own-reference photograph controls, with transport uncertainty intact."""
    result=[]
    measurements=certificate['per_run_sides']
    for run,row in sorted(geometry_rows.items()):
        geometry=row['rig_geometry']
        observed=bool(geometry.get('edge_gate_verified_on_own_photograph'))
        for side in ('left','right','bottom'):
            measured=measurements[run][side] if observed else {}
            result.append(dict(run=run,side=side,source_photo_verified=observed,
                median_abs_px=measured.get('median_abs_px'),p95_abs_px=measured.get('p95_abs_px'),
                max_abs_px=measured.get('max_abs_px'),photographic_gate_px=certificate['photograph_gate_px'],
                photographed_source_gate_passed=measured.get('passed'),
                fit_count=measured.get('fit_count'),heldout_count=measured.get('heldout_count'),
                measurement_scope='own baseline photograph' if observed else 'not photographed; transported AC56 geometry',
                boundary_evidence=geometry['boundary_evidence'],
                later_frames='frozen within-run transport; no independent per-frame photographic refit',
                physical_scale_approved=False))
    return result


def _csv_table(path,rows,fields):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(row.get(key),sort_keys=True) if isinstance(row.get(key),dict)
                             else row.get(key) for key in fields})


def report_candidate(root,lighting_csv,output,*,consumer_inventory=None):
    root,output=Path(root),Path(output)
    if output.exists(): raise FileExistsError('Reports must use a new output directory')
    qa_path=root/'production_qa/rig_camera_qa.json'
    seal_path=root/'rig_camera_candidate_receipt.json'
    qa,seal=read_json(qa_path),read_json(seal_path)
    if not seal.get('complete') or qa.get('all_pass') is not True or len(qa.get('rows',[]))!=864:
        raise ValueError('Reports require the completed 43/864 source-bound candidate')
    if seal['qa']['sha256']!=sha256_file(qa_path): raise ValueError('Final QA changed')
    rows=qa['rows']
    by_run={name:[] for name in sorted({row['run'] for row in rows})}
    if len(by_run)!=43: raise ValueError('Candidate does not contain all43 runs')
    for row in rows: by_run[row['run']].append(row)
    effect={run:nearest_effect_rows(values) for run,values in by_run.items()}
    for values in effect.values():
        for row in values:
            row['old_reweighted_geometry_only_eligible']=not row['source_changed']
            row['old_reweighted_weight_geometry']=('new-source geometry applied to old-source map; not geometry-only'
                if row['source_changed'] else 'same retained acquisition geometry')
    groups={}
    with Path(lighting_csv).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['run'] in groups and groups[row['run']]!=row['light']:
                raise ValueError('Source lighting group is ambiguous')
            groups[row['run']]=row['light']
    if len(groups)!=37 or not set(groups)<=set(by_run) or set(groups.values())!={'rig','new_lamps','flicker'}:
        raise ValueError('Lighting evidence must declare the original37-run comparison cohort')
    cohorts={}
    for name,cohort in [('shared37',set(groups)),('shared36_without_ac31',set(groups)-{'ac31'})]:
        tables=[]
        for index,target in enumerate((1.,2.25,4.,12.,24.,48.)):
            for representation in ('old_fixed_cm2','old_reweighted_cm2','new_cm2'):
                for phase in ('co2','gas','aqueous'):
                    values={run:effect[run][index]['area_report'][representation][phase] for run in cohort}
                    stat=cohort_statistics(values,cohort_runs=cohort,
                        rig_light_runs={run for run in cohort if groups[run]=='rig'},
                        new_light_runs={run for run in cohort if groups[run]=='new_lamps'},
                        other_light_runs={run for run in cohort if groups[run]=='flicker'},cv_ddof=1)
                    changed=sorted(run for run in cohort if effect[run][index]['source_changed'])
                    tables.append(dict(target_h=target,representation=representation,phase=phase,
                        camera_changed_runs=changed,
                        reweighted_geometry_only_eligible=representation!='old_reweighted_cm2' or not changed,
                        reweighted_qualification=('Recorded new-source weights applied to old-source maps for '+','.join(changed)
                            if representation=='old_reweighted_cm2' and changed else None),**stat))
        cohorts[name]=tables
    ac31=[]
    rig_cohort={run for run,light in groups.items() if light=='rig' and run!='ac31'}
    for target in (1.75,2.,2.25,2.75,3.,4.,5.):
        selected=nearest_effect_rows(by_run['ac31'],[target])[0]
        cohort_selected={run:nearest_effect_rows(by_run[run],[target])[0] for run in rig_cohort}
        ac31.append(dict(target_h=target,selected=selected,
            rig_light_mean_cm2={phase:float(np.mean([row['area_report']['new_cm2'][phase]
                for row in cohort_selected.values()])) for phase in ('co2','gas','aqueous')},
            rig_light_runs=sorted(rig_cohort),comparison='AC31 excluded from its own comparison mean'))
    geometry_path=Path(qa['geometry_acceptance']['path'])
    if sha256_file(geometry_path)!=qa['geometry_acceptance']['sha256']:
        raise ValueError('Accepted geometry changed after final QA')
    geometry=read_json(geometry_path)
    measurement_asset=qa['geometry_acceptance']
    if 'widths_and_fit_curve_straightness' not in geometry:
        certificate=geometry['relative_geometry_certificate']
        certificate_path=Path(certificate['path'])
        if sha256_file(certificate_path)!=certificate['sha256']:
            raise ValueError('Rig measurement changed after geometry acceptance')
        geometry=read_json(certificate_path)
        measurement_asset=certificate
    inference=read_json(root/'rig_camera_inference_plan.json')
    clock_asset=inference['upstream_plan']['binding']['camera_plan']
    clock=read_asset(clock_asset)
    label_counts=read_asset(clock['label_audit'])
    consumers=None if consumer_inventory is None else artifact(consumer_inventory)
    # Geometry parameters are evaluated independently of saved area summaries.
    geometry_rows={row['run']:row for row in reversed(inference['source_inventory']) if row['frame_index']==0}
    support_table,interior_table=frame_report_tables(rows)
    added_side_table,early_side_summary=added_strip_time_rows(rows)
    wall_table=source_wall_residual_rows(geometry,geometry_rows)
    per_run=[]
    for run,values in by_run.items():
        first=min(values,key=lambda row:row['frame_index'])
        areas=np.asarray([row['area_report']['support_cm2_by_class']['rig_domain'] for row in values],float)
        per_run.append(dict(run=run,**physical_area_factors(geometry_rows[run]['rig_geometry']),
            frame0_domain_cm2=float(first['area_report']['support_cm2_by_class']['rig_domain']),
            minimum_domain_cm2=float(areas.min()),maximum_domain_cm2=float(areas.max()),
            excluded_hardware_cm2_frame0=first['area_report']['excluded_hardware_cm2']))
    domain=np.asarray([row['frame0_domain_cm2'] for row in per_run],float)
    median=float(np.median(domain))
    for row in per_run: row['domain_fractional_deviation_from_cohort_median']=row['frame0_domain_cm2']/median-1
    result=dict(contract='ffac-rig-camera-section5-measured-report-v1',candidate=artifact(seal_path),
        qa=artifact(qa_path),lighting_group_evidence=artifact(lighting_csv),cv_ddof=1,
        warning='Relative AC28 area at assigned scale; actual rig width pending. Validation only, not mass analysis.',
        domain_area_median_cm2=median,domain_outliers_over_one_percent=[row['run'] for row in per_run
            if abs(row['domain_fractional_deviation_from_cohort_median'])>.01],
        per_run_area=per_run,per_run_effect=effect,cohorts=cohorts,ac31_camera_times=ac31,
        geometry_acceptance=qa['geometry_acceptance'],width_and_angles=geometry['widths_and_fit_curve_straightness'],
        source_wall_measurement_certificate=measurement_asset,source_wall_residuals=wall_table,
        source_wall_measurement_scope='Own-reference photographic residuals for AC39–61; later frames inherit frozen within-run transport. AC17–38 walls are unobserved. This is not an independent per-frame wall-image refit.',
        per_frame_support_and_hardware=support_table,per_frame_interior_changed_pixels=interior_table,
        added_strip_all_frames_by_side=added_side_table,added_strip_early_and_max_by_run_side=early_side_summary,
        added_strip_negative_control_rule='Actual camera elapsed_h < 0; do not call frame1 pre-injection solely by its selected index.',
        added_strip_radiometric_qualification='Modelled records finite RAW and feature support. It does not independently validate the frozen light-field extrapolation beyond its original fitting support. Actual photographed grain is retained, rather than cropped by its predicted CO2 class.',
        actual_rig_inner_width_measured=False,
        photographic_top_revision=geometry.get('photographic_top_revision'),
        photographic_top_reference_evidence=geometry.get('photo_top_reference_evidence'),
        added_strip_side_measurement_missing_frames=sum(row.get('added_strip_by_side') is None for row in rows),
        interior_gt30mm_changed_pixels=sum(row['interior_changed_gt30mm_pixels'] or 0 for row in rows),
        interior_comparison_excluded_changed_source_frames=sum(row['source_changed'] for row in rows),
        changed_source_frames=sum(row['source_changed'] for row in rows),
        old_reweighted_interpretation='Optional recorded values. For changed acquisitions the new-source area map weights the old-source map; these rows and their affected cohort summaries are not geometry-only comparisons. Requested old-fixed versus new results are retained.',
        camera_clock={
            'evidence':clock_asset,'maximum_timeline_minus_camera_h':clock['max_timeline_minus_camera_h'],
            'changed_frames':clock['changed_frames'],
            'other_sources_sha_identical':clock['unchanged_source_shas_confirmed'],
            'ac31_round1_source_owned_label_counts':label_counts},
        pixel_area_consumer_inventory=None if consumers is None else dict(evidence=consumers,report=read_asset(consumers)),
        previous_inward_gap_cause={
            'photo_domain':'Conservative 382x710 AC28 B sand core transported to each frame, then an 8px erosion (17x17 kernel).',
            'feature_support':'Intersection of current/reference/previous finite support, full bilinear neighbours, and the recorded 8px feature guard.',
            'gas_support':'Further intersection with flatfield_valid and resolved frozen categories.',
            'interpretation':'These masks overlap; their losses cannot be added as independent widths or attributed to physical sand motion.'},
        new_boundary_filters={
            'fixed_domain_erosion_px':0,'g3_normalized_mean_kernel_hw':[5,5],
            'optical_texture_sigma_px':2.,'optical_context_shape_hw':[2979,5472],
            'policy':'Filters exclude hardware; unsupported current/reference/previous features remain explicitly filled, never called modelled.'},
        all_labels_remain_source_owned=True,scientific_acceptance=False,mass_eligible=False)
    output.mkdir(parents=True)
    fields=['run','target_h','elapsed_h','frame_index','source_sha256','camera_offset_h','representation',
            'co2_cm2','aqueous_cm2','gas_cm2','added_domain_cm2','added_co2_cm2','added_strip_co2_share',
            'source_changed','old_reweighted_geometry_only_eligible','old_reweighted_weight_geometry']
    with (output/'effects_all43.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for run,values in effect.items():
            for row in values:
                for rep in ('old_fixed_cm2','old_reweighted_cm2','new_cm2'):
                    a=row['area_report']
                    writer.writerow(dict(run=run,**{key:row[key] for key in fields[1:5]},
                        camera_offset_h=row['camera_offset_h'],representation=rep,
                        **{phase+'_cm2':a[rep][phase] for phase in ('co2','aqueous','gas')},
                        **{key:a[key] for key in ('added_domain_cm2','added_co2_cm2','added_strip_co2_share')},
                        **{key:row[key] for key in ('source_changed','old_reweighted_geometry_only_eligible','old_reweighted_weight_geometry')}))
    _csv_table(output/'support_hardware_all864.csv',support_table,list(support_table[0]))
    _csv_table(output/'interior_distance_bands_all864.csv',interior_table,list(interior_table[0]))
    _csv_table(output/'source_wall_residuals_all43.csv',wall_table,list(wall_table[0]))
    _csv_table(output/'added_strip_all864_by_side.csv',added_side_table,list(added_side_table[0]))
    _csv_table(output/'added_strip_f0_f1_and_max_by_run_side.csv',early_side_summary,list(early_side_summary[0]))
    result['data_tables']={name:artifact(output/name) for name in (
        'effects_all43.csv','support_hardware_all864.csv','interior_distance_bands_all864.csv','source_wall_residuals_all43.csv',
        'added_strip_all864_by_side.csv','added_strip_f0_f1_and_max_by_run_side.csv')}
    (output/'section5_tables.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--lighting-csv',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--consumer-inventory')
    args=parser.parse_args()
    result=report_candidate(args.candidate,args.lighting_csv,args.output,
                            consumer_inventory=args.consumer_inventory)
    print(json.dumps(dict(output=args.output,runs=len(result['per_run_area']),
                         domain_outliers=result['domain_outliers_over_one_percent'])))


if __name__=='__main__':main()
