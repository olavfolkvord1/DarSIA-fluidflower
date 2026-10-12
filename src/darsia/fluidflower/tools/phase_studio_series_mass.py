"""Studio host adapter for shared-series fits, measured gas and portable exports."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from urllib.parse import unquote, urlparse

import numpy as np

from darsia_extensions import studio_mass as core, series_mass
from darsia.fluidflower.tools.rerender_studio_mass_v3_shared import render_frame, render_mass_curve, encode, sha, read, write
from darsia.fluidflower.tools.phase_studio_photometry_policy import variant_status


def local_path(value):
    if str(value).startswith('file:///'):
        return Path(unquote(urlparse(str(value)).path).lstrip('/'))
    return Path(value)


def result_paths(path):
    payload = read(path)
    if payload.get('complete') and 'selected_inputs' in payload and 'frames' in payload:
        return [Path(path).resolve()]
    if isinstance(payload.get('experiments'), list):
        return [local_path(r['result']).resolve() for r in payload['experiments'] if r.get('result')]
    if payload.get('stage') == 'complete' and payload.get('result'):
        return [local_path(payload['result']).resolve()]
    raise ValueError('Select a completed result.json, status.json or delivery.json')


def inspect_result(path, result=None):
    result = read_result(path) if result is None else result
    if not result.get('complete') or not result.get('frames'):
        raise ValueError('A complete numerical result is required')
    candidate = result['selected_inputs']['candidate']
    return dict(path=str(Path(path).resolve()), dataset=candidate['dataset'], run=candidate['run'],
                model=candidate['candidate_id'], frames=len(result['frames']),
                created_utc=result.get('created_utc', ''), method=result['method']['settings'])


def read_result(path):
    result=read(path);folder=Path(path).resolve().parent
    for row in result.get('frames',[]):
        for key in ('preview','fields'):
            if row.get(key) and not Path(row[key]).is_absolute():row[key]=str(folder/row[key])
    for key in ('mass_curve','overflow_curve','series_calibration','result_source_manifest'):
        if result.get(key) and not Path(result[key]).is_absolute():result[key]=str(folder/result[key])
    return result


def summarize_bound_result(result):
    """Reuse recorded optical fields without repeating image/phase processing."""
    bins=np.linspace(0.,4.,1025);weights=[];volumes=[];gas=[];residual=[]
    for row in result['frames']:
        with np.load(row['fields'],allow_pickle=False) as frame:
            signal=frame['signal'];av=frame['aqueous_volume_m3'];gv=frame['gas_volume_m3']
            sol=frame['solubility_g_l'];density=frame['gas_density_g_l'];positive=signal>0
            weights.append(np.histogram(np.clip(signal[positive],0,4),bins,weights=(av*sol*1000)[positive])[0])
            volumes.append(float(np.sum(av+gv,dtype=np.float64)))
            gas.append(float(np.sum(gv*density,dtype=np.float64)*1000))
            residual.append(float(np.sum(gv*sol,dtype=np.float64)*1000))
    times=np.asarray([r['elapsed_h'] for r in result['frames']],float)
    injected=np.asarray([r['injected_mass_g'] for r in result['frames']],float)
    volumes=np.asarray(volumes);end=np.flatnonzero(injected>=.9999*injected.max())
    selected=np.flatnonzero((injected>0)&(volumes>0))
    if len(selected)<3 or not len(end):raise ValueError('At least three positive-mass observations are required')
    return dict(run=result['selected_inputs']['candidate']['run'],times=times,injected_g=injected,
        volumes_m3=volumes,positive_indices=selected,anchor_index=int(end[0]),
        histogram_centers=(bins[:-1]+bins[1:])*.5,aqueous_histogram_weights=np.asarray(weights),
        gas_full_mass_g=np.asarray(gas),residual_saturated_mass_g=np.asarray(residual))


def references_for(run, times, path):
    from darsia.fluidflower.tools.overflow_phase_constraint import load_overflow_phase_constraint
    constraint = load_overflow_phase_constraint(path, run=run)
    records = []
    for time in times:
        target = constraint.target_at(float(time))
        records.append(None if target is None else dict(mass_g=target.gas_mass_kg*1000,
            sigma_g=target.phase_mass_sigma_kg*1000, time_h=target.time_h,
            mode=constraint.constraint_mode, source=target.target_source))
    return records, constraint.to_dict()


def run(request_path):
    request_path = Path(request_path)
    request = read(request_path)
    root = Path(request['output']).resolve()
    def update(stage, **extra):
        write(request_path.with_name('status.json'), dict(stage=stage, **extra))
    try:
        if root.exists(): raise FileExistsError('Choose a new analysis output folder')
        source_manifest={str(Path(module.__file__).resolve()): sha(module.__file__) for module in (core, series_mass)}
        source_manifest[str(Path(__file__).resolve())]=sha(__file__)
        sources = [Path(p).resolve() for p in request['results']]
        metadata = [inspect_result(p) for p in sources]
        if len({m['dataset'] for m in metadata}) != 1:
            raise ValueError('Analyze one experiment series at a time')
        if len({m['run'].casefold() for m in metadata}) != len(metadata):
            raise ValueError('Choose one phase-model result per experiment')
        if any(variant_status(m['dataset'], m['method']) == 'excluded' for m in metadata):
            raise ValueError('Excluded FF_UM light-correction results are historical only and cannot be fitted again')
        settings = metadata[0]['method']
        if any(m['method'] != settings for m in metadata):
            raise ValueError('Shared calibration requires the same optical and chemistry method in every source')
        method = core.Method(**{k: settings[k] for k in asdict(core.Method())})
        values = [read_result(p) for p in sources]
        policies = [value.get('mass_policy', asdict(core.MassPolicy())) for value in values]
        if any(p != policies[0] for p in policies):
            raise ValueError('Gas phase policies differ; select consistently calculated inputs')
        policy = core.MassPolicy(**policies[0])
        scope = request.get('scope', 'series')
        overflow_mode = request.get('overflow_mode', 'compare')
        if overflow_mode not in ('compare', 'constrain'):
            raise ValueError('Choose comparison or measured gas constraints')
        bindings = {str(p): sha(p) for p in sources}
        for source in values:
            for frame in source['frames']:
                bindings[str(Path(frame['fields']).resolve())] = sha(frame['fields'])
        summaries, reference_records, provenance = [], {}, {}
        for p, m, source in zip(sources, metadata, values):
            update('Preparing shared optical statistics: '+m['run'])
            s = summarize_bound_result(source)
            ref_path = request.get('overflow', {}).get(m['run'])
            refs = [None]*len(s['times'])
            if ref_path:
                refs, provenance[m['run']] = references_for(m['run'], s['times'], ref_path)
                bindings[str(Path(ref_path).resolve())] = sha(ref_path)
            reference_records[m['run']] = refs
            if overflow_mode == 'constrain': s['gas_references'] = refs
            summaries.append(s)
        if request.get('calibration_file'):
            if scope != 'series':raise ValueError('A frozen series model cannot be applied as an individual fit')
            calibration_path=Path(request['calibration_file']).resolve()
            frozen=read(calibration_path)
            if (frozen.get('format')!=series_mass.VERSION or frozen.get('scope')!='series'
                    or frozen.get('dataset')!=metadata[0]['dataset']):
                raise ValueError('Load a shared calibration for this experiment series')
            if frozen.get('mass_policy')!=asdict(policy) or frozen.get('method',{}).get('settings')!=core.provenance(method)['settings']:
                raise ValueError('Frozen calibration uses another optical method or gas policy')
            if frozen.get('overflow_mode')!=overflow_mode:
                raise ValueError('Frozen calibration and selected overflow policy differ')
            bindings[str(calibration_path)]=sha(calibration_path)
            update('Applying frozen shared parameters without fitting')
            fit=json.loads(json.dumps(frozen));fit['by_run']={}
            for summary in summaries:
                total,saturation,records=series_mass.predict(summary,fit['parameters'],policy)
                fit['by_run'][summary['run']]=dict(total_mass_g=total.tolist(),gas_saturation=saturation.tolist(),gas_reference_records=records)
            fit.update(applied_runs=[s['run'] for s in summaries],fit_performed=False,
                       frozen_calibration=str(calibration_path),frozen_calibration_sha256=sha(calibration_path))
        else:
            update('Fitting '+('one shared series response' if scope == 'series' else 'individual comparison response'))
            fit = series_mass.fit(summaries, method, policy, scope=scope, cross_validate=True)
            fit['fit_performed']=True
        root.mkdir(parents=True)
        fit.update(dataset=metadata[0]['dataset'], source_bindings=bindings,
                   overflow_provenance=provenance, overflow_mode=overflow_mode,
                   method=core.provenance(method))
        write(root/'calibration.json', fit)
        delivered = []
        for source, summary, m in zip(values, summaries, metadata):
            folder = root/m['run']; folder.mkdir()
            p = fit['parameters']
            saturation = fit['by_run'][m['run']]['gas_saturation']
            gas_records = fit['by_run'][m['run']]['gas_reference_records']
            dilution = core.dilution_factors(summary['volumes_m3'], summary['times'], summary['anchor_index'], p['beta'])
            rows = []
            limit = float(source['color_limit_g_l'])
            title = m['dataset'].upper()+' · '+m['run'].upper()
            for index, old in enumerate(source['frames']):
                update(f"Rendering {m['run']} {index+1}/{len(source['frames'])}")
                with np.load(old['fields'], allow_pickle=False) as data:
                    fields = {k: np.asarray(data[k]) for k in data.files}
                av, gv = fields['aqueous_volume_m3'], fields['gas_volume_m3']
                solubility, density = fields['solubility_g_l'], fields['gas_density_g_l']
                c = core.concentration(fields['signal'], solubility, p, dilution[index], method)
                totals = core.totals(c, av, gv, solubility, density, saturation[index], policy.residual_liquid_concentration_fraction)
                interval_policy=policy
                if gas_records[index]['status']=='measured overflow constraint':
                    low,high=gas_records[index]['saturation_interval']
                    interval_policy=replace(policy,gas_saturation_min=low,gas_saturation_max=high)
                interval = core.total_mass_interval(c, av, gv, solubility, density, interval_policy)
                archive, preview = folder/f'frame_{index:03d}.npz', folder/f'frame_{index:03d}.png'
                row = {**old, **totals, **interval, 'preview': str(preview), 'fields': str(archive),
                    'calibration_role': ('frozen series model' if not fit['fit_performed'] else
                                         'shared series fit' if scope == 'series' else 'individual comparison fit'),
                    'dilution_factor': float(dilution[index]), 'gas_saturation': saturation[index],
                    'gas_reference': reference_records[m['run']][index], 'gas_constraint': gas_records[index]}
                fields['concentration_g_l'] = np.where(av > 0, c, 0)
                fields['gas_saturation'] = np.asarray(saturation[index])
                fields['free_co2_mass_g'] = gv.astype(np.float64)*density*saturation[index]*1000
                fields['dissolved_co2_mass_g'] = (av.astype(np.float64)*c*1000 +
                    gv.astype(np.float64)*solubility*(1-saturation[index])*policy.residual_liquid_concentration_fraction*1000)
                fields['total_co2_mass_g'] = fields['free_co2_mass_g']+fields['dissolved_co2_mass_g']
                np.savez_compressed(archive, **fields)
                render_frame(preview, fields['photograph_rgb'], c, av, gv, fields['pore_volume_m3'], row, title, limit)
                rows.append(row)
            curve = folder/'mass_curve.png'
            render_mass_curve(curve, rows, title)
            overflow_curve = folder/'overflow_comparison.png'
            render_overflow(overflow_curve, rows, title)
            result = {**source, 'format': series_mass.VERSION, 'frames': rows, 'calibration': fit,
                'mass_curve': str(curve), 'overflow_curve': str(overflow_curve), 'complete': True,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'source_bindings': bindings,
                'series_calibration': str(root/'calibration.json'), 'overflow_mode': overflow_mode,
                'identity': core.fingerprint(dict(calibration=fit, run=m['run'])),
                'result_source_manifest': str(root/'result_sources.json')}
            write(folder/'result.json', result)
            export_csv(result, folder/'mass_per_frame.csv')
            delivered.append(dict(dataset=m['dataset'], run=m['run'], result=str(folder/'result.json')))
        if any(sha(p) != digest for p, digest in bindings.items()):
            raise ValueError('Source results or overflow targets changed during calculation')
        if any(sha(p) != digest for p, digest in source_manifest.items()):
            raise ValueError('Calculation code changed during processing; do not publish these results')
        write(root/'result_sources.json', source_manifest)
        write(root/'delivery.json', dict(complete=True, experiments=delivered, calibration=str(root/'calibration.json')))
        update('complete', delivery=str(root/'delivery.json'))
    except Exception:
        update('failed', error=traceback.format_exc()); raise


def render_overflow(path, rows, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout='constrained'); FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([r['elapsed_h'] for r in rows], [r['gas_mass_g'] for r in rows], label='Image-based free gas')
    matched = [r for r in rows if r.get('gas_reference')]
    if matched:
        ax.errorbar([r['gas_reference']['time_h'] for r in matched],
            [r['gas_reference']['mass_g'] for r in matched],
            yerr=[r['gas_reference']['sigma_g'] for r in matched], fmt='o', label='Overflow-derived gas reference')
    ax.set(xlabel='Time since injection start · h', ylabel='Free CO2 mass · g', title=title+' · measured gas reference')
    ax.legend(); ax.grid(alpha=.2); fig.savefig(path, dpi=120)


def export_csv(result, path):
    columns = ['elapsed_h','injected_mass_g','aqueous_mass_g','gas_mass_g','total_mass_g',
               'gas_mass_min_g','gas_mass_max_g','dilution_factor','gas_saturation','calibration_role']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in result['frames'])


def export(result_path, destination, kind):
    path, destination = Path(result_path), Path(destination)
    if destination.exists(): raise FileExistsError('Choose a new export path')
    result = read_result(path)
    if kind == 'movie': return encode(path, destination)
    if kind == 'csv': return export_csv(result, destination)
    if kind == 'bundle':
        destination.mkdir(parents=True)
        for row in result['frames']:
            for key in ('preview','fields'): shutil.copy2(row[key], destination/Path(row[key]).name)
        for key in ('mass_curve','overflow_curve'):
            if result.get(key): shutil.copy2(result[key], destination/Path(result[key]).name)
        portable = json.loads(json.dumps(result))
        for row in portable['frames']:
            for key in ('preview','fields'): row[key] = Path(row[key]).name
        for key in ('mass_curve','overflow_curve'):
            if portable.get(key): portable[key] = Path(portable[key]).name
        if result.get('result_source_manifest'):
            shutil.copy2(result['result_source_manifest'],destination/'result_sources.json')
            portable['result_source_manifest']='result_sources.json'
        portable['series_calibration']='calibration.json'
        write(destination/'result.json', portable)
        export_csv(result, destination/'mass_per_frame.csv')
        write(destination/'calibration.json', result['calibration'])
        return
    raise ValueError('Unknown export kind')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--result', type=Path); parser.add_argument('--destination', type=Path)
    parser.add_argument('--export', choices=['movie','csv','bundle'])
    args = parser.parse_args()
    if args.export:
        if not args.result or not args.destination: parser.error('Export needs result and destination')
        export(args.result, args.destination, args.export)
    elif args.request: run(args.request)
    else: parser.error('Specify request or export')
