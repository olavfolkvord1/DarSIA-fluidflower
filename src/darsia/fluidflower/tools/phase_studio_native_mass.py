"""On-demand native DarSIA concentration jobs for the ordinary Studio tab.

One experiment and one full-resolution frame are processed at a time. Phase
inference, training proposals, disagreements and movies are never started here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from darsia.fluidflower import concentration_inputs as inputs
from darsia.fluidflower import concentration_configuration as configuration
from darsia.fluidflower import concentration_native as native
from darsia.fluidflower.series_paths import local_cache_root


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    # On Windows/SMB a reader may briefly hold a handle that denies replacement.
    # Keep the old complete document visible and retry the atomic rename; never
    # truncate a live status/result file to work around a sharing violation.
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.01 * (attempt + 1), .1))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def create_analysis(config):
    dimensions = config['dimensions_m']
    if isinstance(dimensions, dict):dimensions = (dimensions['height'], dimensions['width'])
    return native.NativeConcentrationAnalysis(config['calibration_folder'],
        baseline_rgb=config['baseline_rgb'], labels=config['labels'], domain=config['domain'],
        depth_m=config['depth_m'], porosity=config['porosity'], dimensions_m=dimensions,
        basis=config['basis'], allow_legacy_basis=config.get('allow_legacy_basis', False),
        calibration_status='provisional imported native calibration' if config.get('provisional', True)
        else 'registered native calibration')


def physical_binding(config):
    provenance = config.get('provenance', {})
    artifacts = provenance.get('artifacts')
    return json.loads(json.dumps(dict(basis=config['basis'], dimensions_m=config['dimensions_m'],
        scientific_profile_sha256=provenance.get('scientific_profile_sha256',provenance.get('run_profile_sha256')),
        domain_policy=provenance.get('domain_policy'),
        selected_baseline=provenance.get('selected_phase_baseline'),
        artifacts={item['path']:item['sha256'] for item in artifacts} if artifacts else provenance)))


def prepare_frame(analysis, config, frame, name):
    # Empty fixed support has exactly zero mass, independent of atmosphere.
    # In particular, a pre-injection photo needs no fabricated PT extrapolation.
    if not np.any(frame.phase_labels):return None
    state = configuration.pressure_temperature_at(config, frame.time_hours)
    return analysis.prepare(frame.corrected_rgb, frame.phase_labels, run_id=name,
        time_hours=frame.time_hours, **state,
        nominal_injected_mass_g=frame.nominal_injected_mass_g,
        **({"pixel_area_m2": frame.pixel_area_m2} if getattr(frame, "pixel_area_m2", None) is not None else {}))


def empty_arguments(frame, name):
    return dict(run_id=name, time_hours=frame.time_hours,
        nominal_injected_mass_g=frame.nominal_injected_mass_g)


def selected_jobs(request, catalog):
    if catalog['model_id'] != request['model_id']:
        raise ValueError('Phase model changed; reload the concentration workspace.')
    requested = request.get('runs', [])
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('Select unique registered experiments.')
    by_run = {job['run']:job for job in catalog['jobs']}
    if any(run not in by_run for run in requested):
        raise ValueError('A selected experiment is absent from this phase model.')
    jobs = [dict(by_run[run]) for run in requested]
    if request.get('concentration_profile_path'):
        for job in jobs:
            job['concentration_profile_path'] = request['concentration_profile_path']
    mode = request.get('mode')
    if mode not in ('apply', 'frozen', 'series', 'individual'):
        raise ValueError('Choose a supported native concentration method.')
    for job in jobs:
        job['concentration'] = configuration.describe_run(job)
        if not job['concentration']['ready']:
            raise ValueError(job['run']+': '+'; '.join(job['concentration']['reasons']))
    if mode == 'series':
        groups = {j['concentration']['chemistry_group'] for j in jobs}
        if (len(jobs) < 2 or len(groups) != 1 or groups & {'uncertainty', 'provisional', 'unconfirmed'}
                or any(not j['concentration'].get('shared_fit_eligible', False) for j in jobs)):
            raise ValueError('Shared fitting needs at least two experiments with one confirmed chemistry group.')
    if mode == 'individual' and len(jobs) != 1:
        raise ValueError('Individual fitting requires exactly one experiment.')
    return jobs


def overflow_references(request, inspected):
    mode = request.get('overflow_mode', 'off')
    if mode not in ('off', 'compare', 'constrain'):
        raise ValueError('Unknown overflow mode.')
    if mode == 'off':return [], {}
    if mode == 'constrain' and request['mode'] not in ('series', 'individual'):
        raise ValueError('Measured constraints require an explicit fit.')
    from darsia.fluidflower.tools.phase_studio_series_mass import references_for
    references, bindings = [], {}
    for run, path in request.get('overflow', {}).items():
        if run not in inspected:raise ValueError('Overflow targets refer to an unselected experiment.')
        times = [frame['elapsed_h'] for frame in inspected[run]['frames']]
        records, binding = references_for(run, times, path)
        bindings[str(path)] = sha(path)
        for index, record in enumerate(records):
            if record is not None:
                references.append(dict(run=run, frame_index=index, quantity='gas_mass_g',
                    mass_g=record['mass_g'], sigma_g=record['sigma_g'],
                    time_h=times[index], source=dict(path=str(path), sha256=bindings[str(path)], binding=binding)))
    if not references:raise ValueError('No measured overflow targets cover the selected image times.')
    return references, bindings


def render_frame(path, rgb, fields, row, title, limit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(12, 5), layout='constrained'); FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    # Display decimation only; numerical concentration and integration stay native.
    step = max(1, int(np.ceil(rgb.shape[1]/1200)))
    axes[0].imshow(rgb[::step, ::step]); axes[0].set_title('Corrected review photograph')
    phase = fields['phase_labels'][::step, ::step]
    unknown = fields.get('unquantified_phase_mask',np.zeros_like(fields['phase_labels'],dtype=bool))[::step, ::step]
    concentration = np.ma.masked_where((phase == 0) | unknown, fields['concentration_g_l'][::step, ::step])
    axes[1].imshow(rgb[::step, ::step], alpha=.28)
    mapped = axes[1].imshow(concentration, cmap='viridis', vmin=0, vmax=max(limit or 0, 1e-6))
    if unknown.any():
        overlay = np.zeros((*unknown.shape,4),np.float32); overlay[unknown]=[1.,.3,.05,.9]
        axes[1].imshow(overlay)
    axes[1].set_title('Dissolved CO₂ · fixed phase support' +
        ('\nOrange: phase present, native calibration missing' if unknown.any() else ''))
    if limit is not None:fig.colorbar(mapped, ax=axes[1], shrink=.8, label='Concentration · g/L')
    else:axes[1].set_title('No CO₂ phase selected · zero mass')
    for axis in axes:axis.set_axis_off()
    qualifier = ' · partial quantified mass' if not row.get('quantification_complete',True) else ''
    fig.suptitle(f"{title}{qualifier} · {row['elapsed_h']:.3f} h\n"
                 f"Dissolved {row['aqueous_mass_g']:.4f} g · gas {row['gas_mass_g']:.4f} g · total {row['total_mass_g']:.4f} g")
    fig.savefig(path, dpi=120)


def render_curve(path, rows, title, fitted):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout='constrained'); FigureCanvasAgg(fig)
    axis = fig.subplots(); times = [row['elapsed_h'] for row in rows]
    for key, label in (('total_mass_g','Image-derived total'), ('aqueous_mass_g','Dissolved'), ('gas_mass_g','Free gas')):
        axis.plot(times, [row[key] for row in rows], label=label)
    if any(row.get('unknown_mass_upper_g',0)>0 for row in rows):
        axis.fill_between(times, [row.get('total_mass_lower_g',row['total_mass_g']) for row in rows],
            [row.get('total_mass_upper_g',row['total_mass_g']) for row in rows], alpha=.25,
            label='Uncalibrated support: physical capacity bound')
    selected = [row for row in rows if row.get('injected_mass_g') is not None]
    if selected:axis.plot([r['elapsed_h'] for r in selected], [r['injected_mass_g'] for r in selected], '--', label='Nominal injected mass')
    axis.set(xlabel='Time since injection start · h', ylabel='CO₂ mass · g',
        title=title + (' · fitted to injection; not independent validation' if fitted else ' · provisional native calibration'))
    axis.legend(); axis.grid(alpha=.2); fig.savefig(path, dpi=120)


def run(request_path, *, publish=True):
    """Evaluate a native request; distributed attempts defer catalog publication."""
    start = time.perf_counter()
    request_path = Path(request_path)
    request = read(request_path)
    status_path = request_path.with_name('status.json')
    def update(stage, **extra):
        write(status_path, dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra))
    try:
        update('Checking the selected phase model and native concentration configuration')
        catalog = inputs.discover_model_inputs(request['model_root'])
        jobs = selected_jobs(request, catalog)
        output = Path(request['output']).resolve()
        if not output.is_relative_to(Path(catalog['results_root']).resolve()):
            raise ValueError('Concentration results must be stored in this series NAS result directory.')
        if output.exists():raise FileExistsError('Use a new concentration job directory; saved results are immutable.')
        inspected = {j['run']:inputs.inspect_run(j) for j in jobs}
        frame_indices = {}
        for name,metadata in inspected.items():
            indices = request.get('frame_indices',list(range(len(metadata['frames']))))
            if (not indices or any(type(i) is not int or i<0 or i>=len(metadata['frames']) for i in indices)
                    or indices != sorted(set(indices))):
                raise ValueError('Requested image indices must be unique, chronological and inside the selected review timeline.')
            if request['mode'] in ('series','individual') and len(indices)!=len(metadata['frames']):
                raise ValueError('Explicit native mass fitting uses the full registered review timeline.')
            frame_indices[name] = indices
        references, target_bindings = overflow_references(request, inspected)
        output.mkdir(parents=True)
        mode = request['mode']
        fitted = mode in ('series', 'individual')
        unknown_policy = request.get('unknown_policy', 'complete_frames')
        if unknown_policy not in ('complete_frames', 'bounded_interval'):
            raise ValueError('Unknown missing-calibration policy.')
        provenance, summaries, prepared_roots = {}, {}, {}
        physical_bindings, prepared_sources = {}, {}
        cache = local_cache_root()/'native-concentration'/uuid.uuid4().hex
        preparation_seconds = 0.
        if fitted:
            for job in jobs:
                name = job['run']; metadata = inspected[name]
                config = configuration.load_run_configuration(job)
                analysis = create_analysis(config)
                provenance[name] = analysis.provenance()
                physical_bindings[name] = physical_binding(config)
                summaries[name], prepared_roots[name] = [], []
                prepared_sources[name] = []
                tick = time.perf_counter()
                for index in range(len(metadata['frames'])):
                    update(f"Native optical response: {name.upper()} · image {index+1}/{len(metadata['frames'])}")
                    frame = inputs.load_frame(metadata, index)
                    prepared = prepare_frame(analysis,config,frame,name)
                    folder = cache/name/f'{index:04d}' if prepared is not None else None
                    if prepared is not None:native.save_prepared_frame(folder, prepared)
                    prepared_roots[name].append(folder)
                    prepared_sources[name].append(dict(frame.source_reference))
                    summary = (analysis.summarize_frame(prepared) if prepared is not None else
                               analysis.empty_summary(frame.phase_labels, **empty_arguments(frame,name)))
                    if summary.get('unquantified_phase_pixels',0) and unknown_policy == 'complete_frames':
                        raise ValueError(f'{name.upper()} image {index+1}: selected phase areas lack native concentration calibration. '
                                         'Apply the provisional calibration to inspect these areas before mass fitting.')
                    summaries[name].append(summary)
                    del frame, prepared
                preparation_seconds += time.perf_counter()-tick
                del analysis, config
            update('Fitting native response parameters to nominal injected mass')
            fit = native.fit_summaries(summaries, provenance,
                scope='shared' if mode=='series' else 'individual',
                overflow_mode=request.get('overflow_mode','off'), references=references,
                unknown_policy=unknown_policy)
            fit['physical_bindings'] = physical_bindings
        elif mode == 'frozen':
            fit = read(request['calibration_file'])
            if fit.get('format') != native.VERSION or fit.get('dataset') != catalog['dataset']:
                raise ValueError('The saved native fit belongs to another method or series.')
            if any(j['run'] not in fit.get('native_calibrations', {}) for j in jobs):
                raise ValueError('This frozen fit does not register the selected experiment optical calibration.')
            fit = dict(fit, fit_performed=False, applied_calibration=str(request['calibration_file']))
        else:
            fit = dict(format=native.VERSION, scope='registered', fit_performed=False,
                parameters=asdict(native.CalibrationParameters()), native_calibrations={},
                validation_scope='provisional imported native calibration; not concentration validated')
        fit.update(dataset=catalog['dataset'], overflow_mode=request.get('overflow_mode','off'),
            chemistry_groups={j['run']:j['concentration']['chemistry_group'] for j in jobs})
        parameters = native.CalibrationParameters(**fit['parameters'])
        fitting_finished = time.perf_counter()
        delivered = []; exact_predictions = {}
        for job in jobs:
            name = job['run']; metadata = inspected[name]
            config = configuration.load_run_configuration(job)
            analysis = create_analysis(config)
            current_provenance = analysis.provenance()
            if (fitted or mode == 'frozen') and current_provenance['calibration_files'] != fit['native_calibrations'][name]['calibration_files']:
                raise ValueError('Native optical calibration changed since this frozen fit.')
            if ((fitted or mode == 'frozen') and current_provenance['native_implementations'] !=
                    fit['native_calibrations'][name]['native_implementations']):
                raise ValueError('Native scientific implementation changed since this calibration.')
            if fitted or mode == 'frozen':
                if physical_binding(config) != fit.get('physical_bindings', {}).get(name):
                    raise ValueError('Physical inputs, baseline or measurement protocols changed since this calibration.')
            fit.setdefault('physical_bindings', {})[name] = physical_binding(config)
            fit.setdefault('native_calibrations', {})[name] = current_provenance
            folder = output/name; folder.mkdir()
            rows = []; source_bindings = dict(target_bindings)
            for index in frame_indices[name]:
                update(f"Concentration and mass: {name.upper()} · image {index+1}/{len(metadata['frames'])}")
                frame = inputs.load_frame(metadata, index)
                if fitted:
                    if dict(frame.source_reference) != prepared_sources[name][index]:
                        raise ValueError('A photograph or phase map changed after preparing the native fit.')
                    stored = prepared_roots[name][index]
                    prepared = native.load_prepared_frame(stored) if stored is not None else None
                else:
                    prepared = prepare_frame(analysis,config,frame,name)
                evaluation = (analysis.evaluate(prepared, parameters) if prepared is not None else
                              analysis.empty_evaluation(frame.phase_labels, **empty_arguments(frame,name)))
                coverage = frame.source_reference.get('rig_domain_coverage')
                if coverage and coverage['outside_native_grid_area_cm2'] > 0:
                    evaluation.update(quantification_complete=False,
                        mass_estimate_status='partial_observed_native_grid_only',
                        total_mass_upper_g=None, rig_domain_coverage=coverage,
                        unquantified_rig_strip=True)
                fields = evaluation.pop('fields')
                fields_path = folder/f'fields_{index:04d}.npz'
                np.savez_compressed(fields_path, **fields)
                reference = next((ref for ref in references if ref['run']==name and ref['frame_index']==index), None)
                row = dict(evaluation, frame_index=index, source_name=frame.source_reference.get('source_name'),
                    source_reference=dict(frame.source_reference), injected_mass_g=frame.nominal_injected_mass_g,
                    fields=str(fields_path), preview=str(folder/f'frame_{index:04d}.png'),
                    calibration_role='fit to nominal injected mass' if fitted else 'frozen native calibration',
                    gas_reference=reference)
                title = name.upper()+' · native DarSIA · provisional'
                render_frame(row['preview'], frame.corrected_rgb, fields, row, title,
                    evaluation['solubility_max_g_l'])
                rows.append(row)
                for path_key, digest_key in (('photo','photo_sha256'), ('phase','phase_sha256')):
                    source_bindings[frame.source_reference[path_key]] = frame.source_reference[digest_key]
                del frame, prepared, fields
            analysis.verify_sources()
            exact_predictions[name] = [{key:row[key] for key in
                ('elapsed_h','aqueous_mass_g','gas_mass_g','total_mass_g')} for row in rows]
            curve = folder/'mass_curve.png'; render_curve(curve, rows, title, fitted)
            overflow_curve = None
            if references:
                from darsia.fluidflower.tools.phase_studio_series_mass import render_overflow
                overflow_curve = folder/'overflow_comparison.png'; render_overflow(overflow_curve, rows, title)
            result = dict(format=native.VERSION, complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                selected_inputs=dict(candidate=job['candidate'], review_root=job['review_root']),
                frames=rows, calibration=fit, method=dict(settings=dict(optical='native_color_path',
                    chemistry=('darsia_titration_flash' if current_provenance['flash_model']=='TitrationFlash'
                               else 'darsia_simple_flash'), restoration='none'), native_provenance=current_provenance),
                physical_provenance=config.get('provenance', {}), source_bindings=source_bindings,
                mass_curve=str(curve), overflow_curve=str(overflow_curve) if overflow_curve else None,
                series_calibration=str(output/'calibration.json'), overflow_mode=request.get('overflow_mode','off'),
                phase_inference_repeated=False, phase_labels_modified=False,
                requested_frame_indices=frame_indices[name], available_review_frames=len(metadata['frames']),
                limitations=metadata['limitations']+config.get('provenance',{}).get('warnings',[])+
                    ['Historical native calibration is provisional.']+
                    (['Some phase support has no native concentration calibration; displayed masses are partial.']
                     if any(not row.get('quantification_complete',True) for row in rows) else []))
            result_path = folder/'result.json'; write(result_path, result)
            from darsia.fluidflower.tools.phase_studio_series_mass import export_csv
            export_csv(result, folder/'mass_per_frame.csv')
            delivered.append(dict(dataset=job['dataset'], run=name, result=str(result_path)))
            del analysis, config
        if fitted:
            fit['summary_predictions'] = fit['predictions']
            fit['predictions'] = exact_predictions
            fit['fitting_reduction']['final_exact_native_evaluation_required'] = False
            fit['fitting_reduction']['max_observed_total_mass_error_g'] = max(
                abs(row['total_mass_g']-fit['summary_predictions'][name][index]['total_mass_g'])
                for name,rows in exact_predictions.items() for index,row in enumerate(rows))
            for comparison in fit.get('overflow_comparisons',[]):
                comparison['predicted_mass_g'] = exact_predictions[comparison['run']][comparison['frame_index']][comparison['quantity']]
        # Complete the shared provenance before any result is published.
        for item in delivered:
            result = read(item['result']); result['calibration'] = fit; write(item['result'],result)
        write(output/'calibration.json', fit)
        write(output/'timing.json', dict(total_seconds=time.perf_counter()-start,
            native_preparation_seconds=preparation_seconds,
            discovery_preparation_and_fit_seconds=fitting_finished-start,
            concentration_output_seconds=time.perf_counter()-fitting_finished,
            experiments=len(jobs), images=sum(len(value) for value in frame_indices.values())))
        if any(sha(path)!=digest for path,digest in target_bindings.items()):
            raise ValueError('Overflow targets changed during calculation.')
        from darsia.fluidflower.tools.phase_studio_mass_inputs import register_variant
        if publish:
            for item in delivered:register_variant(item['result'], catalog['results_root'])
        write(output/'delivery.json', dict(complete=True, experiments=delivered, calibration=str(output/'calibration.json')))
        update('complete', delivery=str(output/'delivery.json'))
        return delivered
    except Exception:
        update('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    run(parser.parse_args().request)
