"""Seal new round-2 source inputs and all-run MFC dose QA, metadata only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

from darsia.fluidflower import phase_ffac_common_mass_inputs as r1
from darsia.fluidflower import phase_ffac_common_mass_round2_inputs as r2


def discover_logs(raw_root, run):
    """Search the named controller folder and other run-local locations.

    Filter names before requesting file metadata: image folders can hold many
    thousands of NAS assets. No RAW/JPEG/phase array is opened by discovery.
    """
    folder = Path(raw_root)/run.upper()
    files, searched = [], []
    for directory, subdirs, names in os.walk(folder):
        # Directory enumeration is local to this run. Skip actual image
        # acquisition trees while retaining log/config/protocol directories.
        subdirs[:] = [name for name in subdirs if not re.match(r'^\d{6}_\d{6}', name)
                      and name.casefold() not in ('originals', 'images', 'raw', 'jpg', 'jpeg')]
        searched.append(directory)
        for name in names:
            lower = name.casefold()
            if lower.endswith(('.txt', '.csv', '.log')) and any(token in lower for token in ('flow', 'floplot', 'mfc', 'drain')):
                path = Path(directory)/name
                files.append(dict(path=str(path), size_bytes=path.stat().st_size,
                                  warmup=bool(re.search(r'warm[_ -]?up', lower))))
    return dict(run=run, files=sorted(files, key=lambda row: row['path'].casefold()),
                searched_directories=searched, search_scope='run-local metadata/log directories, including outside Flowplot')


def _filename_date_qa(path, origin):
    match = re.search(r'(20\d{12})(?=\.[^.]+$)', Path(path).name)
    if match is None:
        return None
    stamp = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
    if stamp.date() != datetime.fromisoformat(origin).date():
        raise ValueError('Timestamped controller log belongs to another acquisition date')
    return stamp.isoformat()


def unique_measured_sources(passed):
    """Collapse byte-identical copied logs while retaining every source alias."""
    unique = {}
    for value in sorted(passed, key=lambda row: (len(Path(row['source']['path']).parts), row['source']['path'].casefold())):
        key = value['source']['sha256']
        if key not in unique:
            unique[key] = dict(value, source_aliases=[value['source']])
        else:
            unique[key]['source_aliases'].append(value['source'])
    return list(unique.values())


def prepare_run(run, *, raw_root, atmosphere, recipe, frames):
    discovery = discover_logs(raw_root, run)
    record = dict(run=run, injection_stages=atmosphere['injection_stages'],
                  origin_datetime=atmosphere['origin_datetime'],
                  injection_source=atmosphere['injection'],
                  role='fit' if run in r2.FIT_RUNS else 'final_check' if run in r2.HELD_OUT else 'descriptive',
                  protocol_caveats=recipe.get('caveat', ''), discovery=discovery)
    clock, clock_error = None, None
    if recipe.get('protocol'):
        try:
            if r1.sha256(recipe['protocol']['path']) != recipe['protocol']['sha256']:
                raise ValueError('Original protocol SHA changed before clock read')
            clock = r2.script_clock(recipe['protocol'], atmosphere['origin_datetime'])
        except Exception as error:
            clock_error = f'{type(error).__name__}: {error}'
    else:
        clock_error = 'Original protocol workbook/script clock missing; reconstructed timing is not independent clock QA'
    attempted, passed = [], []
    for discovered in discovery['files']:
        path = Path(discovered['path'])
        if discovered['warmup']:
            attempted.append(dict(path=str(path), admitted=False, reason='Warm-up acquisition; not injection log'))
            continue
        try:
            stamp = _filename_date_qa(path, atmosphere['origin_datetime'])
            data = r2.parse_mfc(path)
            if clock is None:
                raise ValueError(clock_error)
            measured = r2.integrate_mfc(data, atmosphere, clock['script_start_datetime'])
            measured.update(source=r1.asset(path), clock_source=clock,
                filename_timestamp=stamp, source_receipt_note='Measured stream retained; no arbitrary clipping or fitted clock shift')
            passed.append(measured)
            attempted.append(dict(path=str(path), admitted=True, source=measured['source'],
                units_qa=measured['units_qa'], clock_qa=measured['clock_qa'],
                measured_final_mass_g=measured['measured_final_mass_g']))
        except Exception as error:
            attempted.append(dict(path=str(path), admitted=False,
                                  reason=f'{type(error).__name__}: {error}'))
    passed = unique_measured_sources(passed)
    nominal = r1.nominal_mass_at(atmosphere, max(row['end_h'] for row in atmosphere['injection_stages']))
    record.update(nominal_final_mass_g=nominal, clock_source=clock, attempts=attempted)
    if len(passed) == 1:
        measured = passed[0]
        record.update(dose_kind='measured_mfc', measured_mfc_dose=measured,
            measured_source=measured['source'], selected_final_mass_g=measured['measured_final_mass_g'],
            measured_over_nominal=measured['measured_over_nominal'], fallback_reasons=[],
            qa=dict(passed=True, units=measured['units_qa'], clock=measured['clock_qa']))
    else:
        reasons = (['No injection acquisition found after run-local alternate-location search'] if not discovery['files'] else
                   ['No injection log passes the same units/clock QA'] if not passed else
                   ['Multiple independent injection logs pass; ambiguous source, no arbitrary selection'])
        if clock_error:
            reasons.append(clock_error)
        record.update(dose_kind='nominal_setpoint', selected_final_mass_g=nominal,
            measured_over_nominal=None, fallback_reasons=reasons,
            qa=dict(passed=False, independent_clock_available=clock is not None))
    if run == 'ac38':
        # This report-only stream is not admitted by clock QA. Its clock origin
        # is explicitly reconstructed and never replaces the nominal main dose.
        candidates = [row for row in discovery['files'] if not row['warmup']]
        if len(candidates) == 1:
            try:
                origin = datetime.fromisoformat(atmosphere['origin_datetime'])
                provisional_clock = (origin-timedelta(seconds=310)).isoformat()
                provisional = r2.integrate_mfc(r2.parse_mfc(candidates[0]['path']), atmosphere, provisional_clock)
                provisional.update(source=r1.asset(candidates[0]['path']),
                    clock_qa=dict(passed=False, reconstructed=True, independent_clock_verified=False),
                    main_target_used=False, timing_uncertainty_seconds=30.,
                    assumption='Report-only reconstruction: injection begins 310 s after controller program start')
                sensitivities = []
                for shift in (-30., 0., 30.):
                    shifted = (datetime.fromisoformat(provisional_clock)+timedelta(seconds=shift)).isoformat()
                    try:
                        sensitivity = r2.integrate_mfc(r2.parse_mfc(candidates[0]['path']), atmosphere,
                            shifted, clock_gate_s=32.)
                        sensitivities.append(dict(script_start_shift_s=shift,
                            measured_final_mass_g=sensitivity['measured_final_mass_g'],
                            measured_over_nominal=sensitivity['measured_over_nominal'],
                            last_observed_h=sensitivity['last_observed_h'],
                            acquired_through_protocol_end=sensitivity['last_observed_h'] >= sensitivity['protocol_end_h'],
                            unobserved_tail_seconds=max(0., (sensitivity['protocol_end_h']-sensitivity['last_observed_h'])*3600),
                            reported_mass_scope='Observed measured integral only; any unobserved tail is not extrapolated',
                            main_target_used=False, clock_qa_passed=False,
                            camera_frame_doses=[dict(frame_index=row['frame_index'],
                                elapsed_h=row['elapsed_h'], mass_g=0. if row['elapsed_h'] <= 0 else
                                float(np.interp(row['elapsed_h'], sensitivity['elapsed_h'], sensitivity['cumulative_mass_g'])))
                                for row in frames if row['run'] == run]))
                    except Exception as error:
                        sensitivities.append(dict(script_start_shift_s=shift, available=False,
                            main_target_used=False, clock_qa_passed=False, reason=repr(error)))
                provisional['timing_sensitivity'] = sensitivities
                provisional['timing_sensitivity_method'] = ('Reconstructed controller script start shifted by -30, 0, +30 s; '
                    'camera and nominal injection clocks unchanged. Report only, no admitted clock QA, no flow extrapolation.')
                record['provisional_measured_sensitivity'] = provisional
            except Exception as error:
                record['provisional_measured_sensitivity_error'] = repr(error)
    record['camera_frame_doses'] = [dict(frame_index=row['frame_index'], elapsed_h=row['elapsed_h'],
        camera_source_sha256=row['source_sha256'], measured_or_nominal_mass_g=r2.dose_at(record,row['elapsed_h']),
        nominal_setpoint_mass_g=r2.dose_at(record,row['elapsed_h'],nominal=True)) for row in frames if row['run']==run]
    return record


def prepare(args):
    job = Path(args.job)
    bootstrap = json.loads((job/'job.json').read_text(encoding='utf-8-sig'))
    spec = Path(args.spec)
    if r1.sha256(spec) != args.spec_sha256:
        raise ValueError('Active round-2 specification SHA differs')
    base_descriptor = r1.asset(bootstrap['input_plan_source'])
    base = r1.read_bound_json(base_descriptor)
    reader = r1.VerifiedAssetReader()
    freeze_descriptor = dict(path=bootstrap['round1_freeze'],sha256=bootstrap['round1_freeze_sha256'])
    freeze = reader.json(freeze_descriptor)
    if freeze.get('frozen') is not True or not freeze.get('calibration_files'):
        raise ValueError('Source round-1 calibration is not a sealed calibration')
    for path, value in freeze['calibration_files'].items():
        reader.path(dict(path=path,sha256=value) if isinstance(value,str) else value)
    source = reader.json(base['source_metadata'])
    recipes = reader.json(base['recipes'])
    atmosphere = reader.json(base['atmosphere'])
    split = dict(production_runs=list(r2.PRODUCTION_RUNS), fit_runs=list(r2.FIT_RUNS),
        held_out_runs=list(r2.HELD_OUT), descriptive_runs=list(r2.DESCRIPTIVE), excluded_runs=list(r2.EXCLUDED))
    r2.validate_split(split)
    refs = Path(bootstrap['clearwater_results_source'])
    transports = {}
    for run in r2.PRODUCTION_RUNS:
        result = json.loads((refs/f'{run}.json').read_text(encoding='utf-8-sig'))
        if result.get('complete') is not True or result['run'] != run:
            raise ValueError(f'Frozen R1 measured clearwater reference missing for {run}')
        transport = reader.json(result['transport'])
        if transport['run'] != run:
            raise ValueError('Measured clearwater/source run binding differs')
        reader.path(dict(path=transport['clearwater_receipt'],sha256=transport['clearwater_receipt_sha256']))
        transports[run] = result['transport']
    root = job/'inputs'
    root.mkdir(parents=True,exist_ok=True)
    if Path(args.output_name).name != args.output_name:
        raise ValueError('Preparation output name must be one new folder within the job inputs')
    destination = root/args.output_name
    destination.mkdir(exist_ok=False)
    physical_sources = {Path(row['path']).name:row for row in base['physical_reference_sources']}
    copied = {}
    for name in ('depth_measurements.csv','facies.csv'):
        source_path = reader.path(physical_sources[name])
        target = destination/name
        shutil.copyfile(source_path,target)
        if r1.sha256(target)!=physical_sources[name]['sha256']:
            raise ValueError('Portable measured CSV copy differs')
        copied[name]=target
    limits = r2.physical_limits(copied['depth_measurements.csv'],copied['facies.csv'])
    epoch_path = Path(args.light_epochs)
    epoch = json.loads(epoch_path.read_text(encoding='utf-8-sig'))
    epochs = {row['run']:row['epoch']for row in epoch['runs']}
    counts = {name:sum(epochs.get(run)==name for run in r2.FIT_RUNS)for name in ('rig_light','new_lamps')}
    if counts != dict(rig_light=18,new_lamps=8):
        raise ValueError('Predeclared 18/8 fit light-epoch cohort differs')
    epoch_target=destination/'light_epochs.json';shutil.copyfile(epoch_path,epoch_target)
    a={row['run']:row for row in atmosphere['runs']};r={row['run']:row for row in recipes['runs']}
    def one(run):
        value=prepare_run(run,raw_root=args.raw_root,atmosphere=a[run],recipe=r[run],frames=source['frames'])
        r1.write_new_json(destination/f'{run}.dose_qa.json',value)
        print(json.dumps(dict(run=run,dose_kind=value['dose_kind'],measured_over_nominal=value['measured_over_nominal'],
                              fallback_reasons=value['fallback_reasons'])),flush=True)
        return value
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        doses=list(pool.map(one,r2.PRODUCTION_RUNS))
    inventory=dict(contract='ffac-round2-camera-time-dose-inventory-v1',runs=doses,
        created_utc=datetime.now(timezone.utc).isoformat(),complete=True,
        source_atmosphere=base['atmosphere'],source_recipes=base['recipes'],source_camera_times=base['source_metadata'],
        scientific_pixels_read=False,heldout_phase_or_mass_results_read=False,
        measured_count=sum(row['dose_kind']=='measured_mfc'for row in doses),
        nominal_fallback_runs=[row['run']for row in doses if row['dose_kind']!='measured_mfc'])
    dose_path=destination/'dose_inventory.json';r1.write_new_json(dose_path,inventory)
    plan=dict(contract=r2.CONTRACT,complete=True,spec=r1.asset(spec),base_input_plan=base_descriptor,
        source_metadata=base['source_metadata'],dose_inventory=r1.asset(dose_path),
        light_epochs=r1.asset(epoch_target),physical_limits=limits,swi_by_facies=limits['swi_by_facies'],
        clearwater_results_folder=str(refs),clearwater_transports=transports,split=split,
        round1_freeze=freeze_descriptor,
        scientific_adapter=r1.asset(r1.__file__),round2_adapter=r1.asset(r2.__file__),
        reused_raw_photometry_unchanged=True,reused_phase_maps_unchanged=True,reused_colour_paths_unchanged=True,
        paths_policy='26 fit runs use pooled R1 paths originally constructed on27, including AC38; no final-check run entered paths',
        missing_geometry_used_for_capacity_bounds_only=True,unknown_main_geometry_imputed=False,
        heldout_pixels_before_round2_freeze=False,ac38_role='descriptive_out_of_fit')
    plan_path=destination/'input_plan.json';r1.write_new_json(plan_path,plan)
    receipt=dict(complete=True,input_plan=r1.asset(plan_path),dose_inventory=r1.asset(dose_path),
        measured_count=inventory['measured_count'],nominal_fallback_runs=inventory['nominal_fallback_runs'],
        physical_limits=limits,scientific_pixels_read=False,old_files_modified=False)
    r1.write_new_json(destination/'preparation_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,type=Path)
    parser.add_argument('--spec',required=True,type=Path)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--raw-root',required=True,type=Path)
    parser.add_argument('--light-epochs',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--output-name',default='round2_preparation_v1')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    prepare(args)


if __name__=='__main__':main()
