"""Six frozen-photometry concentration comparisons on unchanged Studio phases.

Geometry is decoded once per raw source and shared by six optical branches.
Recorded DarSIA pore volumes and thermodynamics are reused byte-verified; old
corrected RGB and optical signals are never reused as a different photometry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np

from darsia.fluidflower import concentration_inputs as inputs
from darsia.fluidflower import concentration_joint as joint
from darsia.fluidflower.tools import phase_studio_joint_calibration_compare as compare
from darsia.fluidflower.tools import phase_studio_calibration_compare as common
from darsia.fluidflower.tools import phase_studio_native_mass as io
from darsia.fluidflower.tools import phase_mass_photometric_variants as photometry
from darsia.fluidflower.tools import phase_studio_photometry_policy as photometry_policy
from darsia.fluidflower.tools.phase_studio_photometry_policy import (
    FFUM_ALLOWED_VARIANTS, verify_policy_snapshot,
)
from darsia.fluidflower.tools.phase_mass_joint_optics import build_joint_reference, joint_optical_signals
from darsia.fluidflower.tools.phase_studio_joint_inputs import _archive, _resize, _retain_recorded_optical_rig

VERSION = "six-photometry-concentration-comparison-v1"
VARIANTS = tuple(dict(photometry_variant_id=mode + '_cc_' + ('on' if enabled else 'off'),
                      illumination=mode, colorchecker_enabled=enabled)
                 for mode in ('none', 'historical', 'facies') for enabled in (False, True))


def _selected_variants(request, dataset):
    """Freeze an explicit optical comparison within series eligibility limits."""
    dataset = str(dataset).casefold()
    by_id = {variant['photometry_variant_id']: variant for variant in VARIANTS}
    requested = request.get('requested_variants')
    if requested is None:
        requested = (['none_cc_on', 'facies_cc_on'] if dataset == 'ffac'
                     else list(FFUM_ALLOWED_VARIANTS) if dataset == 'ffum'
                     else list(by_id))
    if (not isinstance(requested, list) or not requested
            or any(not isinstance(key, str) or key not in by_id for key in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('requested_variants must be a nonempty list of unique registered variant IDs')
    selected = tuple(by_id[key] for key in requested)
    if dataset == 'ffum':
        policy = request.get('photometry_policy')
        allowed = (policy.get('allowed_variants') if isinstance(policy, dict)
                   else list(FFUM_ALLOWED_VARIANTS))
        if (not isinstance(allowed, list) or not allowed
                or any(not isinstance(key, str) for key in allowed)
                or not set(allowed) <= set(FFUM_ALLOWED_VARIANTS)):
            raise ValueError('Invalid FF_UM photometry policy allowed variants')
        if any(variant['photometry_variant_id'] not in allowed for variant in selected):
            raise ValueError('FF_UM historical and facies illumination variants are excluded')
    if dataset == 'ffac' and any(not variant['colorchecker_enabled'] for variant in selected):
        raise ValueError('FF_AC concentration comparison requires ColorChecker-on variants')
    return selected


def numerical_sources():
    from darsia.fluidflower.tools import colorchecker_refinement, colorchecker_saturation_fallback
    from darsia.fluidflower.tools import phase_studio_photometry_report
    from darsia.corrections.color import illuminationcorrection
    sources = compare.numerical_source_binding()
    for module in (photometry, photometry_policy, colorchecker_refinement, colorchecker_saturation_fallback,
                   illuminationcorrection, phase_studio_photometry_report):
        sources[str(Path(module.__file__).resolve())] = io.sha(module.__file__)
    sources[str(Path(__file__).resolve())] = io.sha(__file__)
    return sources


def validate_physical_reuse(old, inspected, request, name):
    """Only physical arrays may cross an optical-method boundary."""
    bound = old['binding']
    for key in ('native_manifest_sha256', 'source_manifest_sha256', 'identity'):
        if bound[key] != inspected[key]:
            raise ValueError('Physical cache belongs to different fixed inputs: ' + key)
    if old['model_id'] != request['model_id'] or old['run'] != name:
        raise ValueError('Physical cache belongs to another run/model')
    if old['analysis_width'] != request['analysis_width']:
        raise ValueError('Physical cache uses a different analysis grid')
    profile_job = dict(dataset=old['dataset'], run=name,
                       concentration_profile_path=request['concentration_profile_path'])
    configuration = compare.configuration
    description = configuration.describe_run(profile_job, purpose='joint_reference')
    if not description['ready'] or not description['shared_fit_eligible']:
        raise ValueError('Physical profile is unavailable or not eligible for shared fitting')
    _, _, series_root, profile_path, _, profile = configuration._profile(profile_job)
    if io.sha(profile_path) != request['physical_profile_file_sha256']:
        raise ValueError('Physical profile file differs from the declared request snapshot')
    scientific = {key: profile[key] for key in configuration.SCIENTIFIC_PROFILE_FIELDS if key in profile}
    scientific_sha = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    if bound['physical_profile_sha256'] != scientific_sha:
        raise ValueError('Scientific physical profile changed since cached DarSIA materialization')
    record_path = configuration._path(profile['artifact_record'], series_root)
    if bound['physical_artifact_record_sha256'] != io.sha(record_path):
        raise ValueError('Physical artifact receipt changed since cached DarSIA materialization')
    if (old.get('complete') is not True
            or old.get('physical_binding', {}).get('purpose') != 'joint_reference'):
        raise ValueError('Physical reuse requires a complete joint-reference preparation')
    if (bound.get('injection_stages') != inspected.get('injection_stages')
            or bound.get('pressure_temperature_assumption', {}) != profile.get('pressure_temperature_assumption', {})):
        raise ValueError('Physical injection schedule or pressure/temperature assumption changed')
    if [row['frame_index'] for row in old['frames']] != inspected['selected_frame_indices']:
        raise ValueError('Physical cache frame coverage differs from requested selection')


def _require_fresh_output(root):
    """This driver has no resume mode: never replace published preparation."""
    root = Path(root)
    reserved = ('request_binding.json', 'preparation', *(v['photometry_variant_id'] for v in VARIANTS))
    if any((root / name).exists() for name in reserved):
        raise FileExistsError('Use a fresh photometry comparison output; existing results are immutable')


def _branch_failure(stage, error, **extra):
    return dict(stage=stage, reason=str(error),
                diagnostics=getattr(error, 'diagnostics', {}), **extra)


def _facies_gain_settings(request):
    values = request.get('facies_gain_settings', {})
    if not isinstance(values, dict):
        raise ValueError('Facies gain settings must be a mapping')
    try:
        settings = photometry.FaciesGainSettings(**values)
        photometry._validate_settings(settings)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid facies gain settings: ' + str(error)) from error
    return settings


def _historical_parity(rgb, raw, illumination, color, source_record):
    """Verify the retained algorithm at the active precision, before clipping.

    Saved native arrays can predate this comparison's float32 work policy.
    Their identity remains visible, but is not a same-precision parity test.
    """
    illuminated = raw if illumination is None else illumination.correct_array(raw)
    independent = np.asarray(color.correct_array(illuminated), np.float32)
    actual = np.asarray(rgb, np.float32)
    checksum = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    actual_sha, independent_sha = checksum(actual), checksum(independent)
    if actual.shape != independent.shape or actual_sha != independent_sha:
        raise ValueError('Historical branch differs from the independent retained rig at the active work precision')
    saved_sha = source_record.get('corrected_array_sha256')
    return dict(actual_sha256=actual_sha, independent_retained_rig_sha256=independent_sha,
        matched=True, original_saved_array_sha256=saved_sha,
        matches_original_saved_array=(actual_sha == saved_sha) if saved_sha else None,
        colorchecker_work_dtype=os.environ.get('FFAC_COLORCHECKER_WORK_DTYPE', 'float64'),
        parity_scope=('Same geometry, retained ColorChecker' if illumination is None
            else 'Same geometry, retained illumination and ColorChecker')
            + ', active work precision; full float32 array before the bounded adapter',
        source_work_precision_change='Fresh comparison work precision is explicit; original saved arrays can use a different work precision. Saved bitwise equivalence is not required or asserted.')


def _concentration_correction_contract(inspected, branch, request):
    normalization = {
        'none': 'Identity gain; no spatial normalization',
        'historical': 'Retained saved illumination field and its original absolute scale; not renormalized to median gain one',
        'facies': 'Baseline-only smooth achromatic field; median porous gain one; declared global log-gain amplitude policy, not pointwise gain clipping',
    }
    return dict(registered_phase_input_contract=inspected['correction_contract'],
        concentration_photometry=branch['variant'], geometry_policy='retained',
        transform_diagnostics=branch['diagnostics']['transform'],
        illumination_normalization=normalization[branch['variant']['illumination']],
        comparison_interpretation='Complete photometric policies with their explicit normalizations, not an isolated spatial-pattern-only comparison',
        colorchecker_work_dtype=request.get('colorchecker_work_dtype',
            os.environ.get('FFAC_COLORCHECKER_WORK_DTYPE', 'float64')),
        output_dtype='float32',
        bounded_adapter='Each complete branch is clipped to [0,1] before identical analysis-grid pooling; clipping diagnostics are retained')


def _bounded_rgb(rgb, shape):
    array = np.asarray(rgb, np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite corrected RGB')
    outside = (array < 0) | (array > 1)
    diagnostic = dict(final_adapter_clipped_channel_fraction=float(outside.mean()),
                      final_adapter_clipped_pixel_fraction=float(outside.any(axis=-1).mean()))
    return _resize(np.clip(array, 0, 1), shape).astype(np.float32), diagnostic


def _verify_physical_profile(request):
    """Bind an optional NAS assumption receipt to the selected model and run set."""
    profile = Path(request['concentration_profile_path'])
    if io.sha(profile) != request['physical_profile_file_sha256']:
        raise ValueError('Physical profile changed since this comparison request')
    receipt_path = request.get('physical_profile_receipt_path')
    receipt_sha = request.get('physical_profile_receipt_sha256')
    if bool(receipt_path) != bool(receipt_sha):
        raise ValueError('Physical profile receipt path and SHA-256 must be supplied together')
    if not receipt_path:
        return
    if io.sha(receipt_path) != receipt_sha:
        raise ValueError('Physical profile receipt changed since this comparison request')
    receipt = io.read(receipt_path)
    if (receipt.get('profile_sha256') != request['physical_profile_file_sha256']
            or receipt.get('selected_phase_model_id') != request['model_id']):
        raise ValueError('Physical profile receipt belongs to another profile or phase model')
    documented = {name: value for name, value in receipt.get('clearwater_reference_assumptions', {}).items()
                  if name in request['runs']}
    if documented != request.get('clearwater_reference_assumptions', {}):
        raise ValueError('Borrowed clearwater assumption differs from the physical profile receipt')


def _prepare_fresh_physical(job, inspected, request, output):
    """Materialize verified geometry and phase volumes without an optical fit.

    This avoids requiring an older joint-calibration job or a successful
    registered-photometry reference merely to compare new photometric branches.
    """
    from darsia.fluidflower.tools import phase_studio_joint_inputs as physical

    name = job['run']
    physical_job = dict(job, concentration_profile_path=request['concentration_profile_path'])
    assumption = request.get('clearwater_reference_assumptions', {}).get(name)
    if assumption is not None:
        physical_job['clearwater_reference_assumption'] = assumption
    config = compare.configuration.load_run_configuration(physical_job, purpose='joint_reference')
    physical._validate_config(physical_job, inspected, config)
    indices = inspected['selected_frame_indices']
    width = request['analysis_width']
    native_shape = inspected['shape_rc']
    if not 1 <= width <= native_shape[1] or not indices:
        raise ValueError('Invalid physical preparation width or empty fixed-frame selection')
    shape = (round(native_shape[0] * width / native_shape[1]), width)
    if shape[0] < 1:
        raise ValueError('Physical preparation has zero analysis height')
    final_dose = physical._final_dose(inspected)
    paths = physical._clearwater_paths(physical_job)
    reference_inputs = [dict(path=str(path), sha256=io.sha(path)) for path in paths]
    volume = physical._PoreVolumes(config, shape)
    labels = _resize(np.asarray(config['labels'], np.float32), shape, cv2.INTER_NEAREST).astype(np.int16)
    output.mkdir(parents=True, exist_ok=False)
    reference_path = output / 'references.npz'
    reference_sha = _archive(reference_path, facies=labels, pv=volume.pv)
    (output / 'frames').mkdir()
    rows = []
    update = common._child_update(Path(request['output']), 'physical', name)
    for position, index in enumerate(indices):
        update('Materializing verified fixed-phase volumes', frame=position + 1, frames=len(indices))
        frame = inputs.load_frame(inspected, index)
        dose = frame.nominal_injected_mass_g
        if dose is None or not np.isfinite(dose) or dose < 0:
            raise ValueError('Verified nominal injected mass is required')
        av, gv = volume.phases(frame.phase_labels)
        sol, density, state, no_state = physical._physics(config, frame.time_hours, frame.phase_labels, shape)
        arrays_path = output / 'frames' / f'frame_{index:04d}.npz'
        arrays_sha = _archive(arrays_path, av=av, gv=gv, solubility=sol, density=density)
        rows.append(dict(frame_index=index, elapsed_h=frame.time_hours,
            source_name=frame.source_reference.get('source_name'), source_reference=dict(frame.source_reference),
            injected_mass_g=float(dose), nominal_injected_mass_g=float(dose),
            expected_final_injected_mass_g=final_dose, state=state, physical_not_required=no_state,
            physical_state_provenance=(dict(assumed=False, mode='not_required_empty_phase') if no_state
                else compare.configuration.pressure_temperature_provenance_at(config, frame.time_hours)),
            arrays=str(arrays_path), arrays_sha256=arrays_sha,
            zero_dose_phase_support_pixels=(int(np.count_nonzero(frame.phase_labels)) if dose == 0 else 0),
            aqueous_support_area_m2=float(np.count_nonzero(frame.phase_labels == 1)
                * np.prod(physical._dimensions(config)) / frame.phase_labels.size),
            aqueous_pore_volume_m3=float(av.sum(dtype=np.float64)),
            gas_pore_volume_m3=float(gv.sum(dtype=np.float64)),
            affected_pore_volume_m3=float(av.sum(dtype=np.float64) + gv.sum(dtype=np.float64))))
    if any(io.sha(row['path']) != row['sha256'] for row in reference_inputs):
        raise ValueError('Clearwater reference changed during physical preparation')
    binding = dict(format='photometry-physical-inputs-v1', run=name, dataset=job['dataset'],
        model_id=job['model_id'], identity=inspected['identity'],
        native_manifest_sha256=inspected['native_manifest_sha256'],
        source_manifest_sha256=inspected['source_manifest_sha256'],
        physical_profile_sha256=config['provenance']['scientific_profile_sha256'],
        physical_artifact_record_sha256=config['provenance']['artifact_record_sha256'],
        pressure_temperature_assumption=config.get('pressure_temperature_assumption') or {},
        injection_stages=inspected['injection_stages'], frame_indices=indices,
        analysis_width=width, reference_inputs=reference_inputs,
        physical_code_sha256=io.sha(physical.__file__))
    path = output / 'prepared.json'
    manifest = dict(format='photometry-physical-inputs-v1', complete=True,
        run=name, dataset=job['dataset'], model_id=job['model_id'], analysis_width=width,
        shape_hw=list(shape), binding=binding, binding_sha256=compare.digest(binding),
        physical_binding=config['provenance'], prepared_path=str(path),
        references=str(reference_path), references_sha256=reference_sha,
        reference_inputs=reference_inputs, frames=rows,
        expected_final_injected_mass_g=final_dose,
        source_bindings={row['path']: row['sha256'] for row in reference_inputs},
        zero_dose_phase_support=[dict(frame_index=row['frame_index'], elapsed_h=row['elapsed_h'],
            pixels=row['zero_dose_phase_support_pixels'])
            for row in rows if row['zero_dose_phase_support_pixels']],
        fixed_phase_support=True, phase_inference_repeated=False,
        photometric_policy='Physical inputs only; each requested branch builds its own optical reference')
    io.write(path, manifest)
    return manifest


def _prepare_worker(job, inspected, request):
    import darsia
    from darsia.fluidflower.review_storage import resolve_review_path
    from darsia.fluidflower.tools.prepare_phase_injection_references import matching_source
    name, root = job['run'], Path(request['output'])
    output = root / 'preparation' / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing run preparation must not be overwritten: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    update = common._child_update(root, 'prepare', name)
    _verify_physical_profile(request)
    if request.get('physical_prepared_root'):
        old_path = Path(request['physical_prepared_root']) / name / 'prepared.json'
        old = io.read(old_path)
    else:
        old_path = output / 'physical' / 'prepared.json'
        old = _prepare_fresh_physical(job, inspected, request, old_path.parent)
    validate_physical_reuse(old, inspected, request, name)
    physical_refs = compare.read_prepared_arrays(old['references'], old['references_sha256'])
    labels, pv = physical_refs['facies'], physical_refs['pv']
    shape, domain = labels.shape, pv > 0
    native = io.read(inspected['native_manifest'])
    assumption = request.get('clearwater_reference_assumptions', {}).get(name)
    ref_inputs = assumption['sources'] if assumption else old['reference_inputs']
    ref_inputs = [dict(path=str(resolve_review_path(Path(r['path']))), sha256=r['sha256']) for r in ref_inputs]
    for record in ref_inputs:
        if io.sha(record['path']) != record['sha256']:
            raise ValueError('Clearwater reference bytes changed')
    branches, reports, failed = {}, {}, {}
    optical = {}
    with matching_source(job['dataset'], name, native, output / 'rig_work') as (source, geometry):
        selected_rig = _retain_recorded_optical_rig(source, native)
        rig = source.rig
        illumination = next((c for c in rig.color_corrections if hasattr(c, 'local_scaling')), None)
        color = next((c for c in rig.color_corrections if hasattr(c, 'colorchecker')), None)

        def decode(record):
            path = resolve_review_path(Path(record.get('source_path', record.get('path'))))
            expected = record.get('source_sha256', record.get('sha256'))
            if not expected or io.sha(path) != expected:
                raise ValueError('Raw source identity changed: ' + str(path))
            rgb = np.asarray(darsia.imread(path, transformations=rig.shape_corrections,
                             reference_date=rig.reference_date, dtype=np.float32).img, np.float32)[..., :3]
            if tuple(rgb.shape[:2]) != tuple(inspected['shape_rc']):
                raise ValueError('Raw geometric image differs from fixed phase grid')
            return rgb

        update('Decoding pre-injection image and fitting frozen facies illumination')
        baseline_raw = decode(native['reference'])
        native_labels = _resize(labels.astype(np.float32), baseline_raw.shape[:2], cv2.INTER_NEAREST).astype(np.int16)
        native_domain = _resize(domain.astype(np.uint8), baseline_raw.shape[:2], cv2.INTER_NEAREST).astype(bool)
        facies_settings = _facies_gain_settings(request)
        try:
            facies_fit = photometry.fit_facies_illumination(baseline_raw, native_labels, native_domain,
                                                         settings=facies_settings)
            gain_path = output / 'facies_gain.npz'
            gain_sha = _archive(gain_path, gain=facies_fit.gain)
            optical['facies_gain'] = dict(path=str(gain_path), sha256=gain_sha, diagnostics=facies_fit.diagnostics,
                settings=asdict(facies_settings),
                label_source=str(old['references']), label_source_sha256=old['references_sha256'],
                label_resolution=list(shape), fitting_data='pre-injection geometry-only RGB, frozen facies/domain',
                label_resampling='Fixed analysis-grid labels expanded by nearest neighbor to raw-corrected geometry; no new label detail is inferred',
                interpretation='Patch-level experimental illumination estimate; not a native-resolution facies reconstruction')
        except ValueError as error:
            facies_fit = None
            optical['facies_gain'] = dict(rejected=True, settings=asdict(facies_settings),
                                         **_branch_failure('facies_gain', error))
        del native_labels, native_domain
        optical.update(geometry_binding=geometry, recorded_rig=selected_rig,
                       reference_inputs=ref_inputs, physical_prepared_path=str(old_path),
                       physical_prepared_sha256=io.sha(old_path), raw_baseline=native['reference'],
                       raw_geometry_only=True, normalized_input_policy='clip each complete branch before common analysis-grid pooling')
        for variant in request['selected_variants']:
            key = variant['photometry_variant_id']
            folder = root / key / 'preparation' / name
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError('Existing optical preparation must not be overwritten: ' + str(folder))
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if variant['illumination'] == 'facies' and facies_fit is None:
                    raise ValueError(optical['facies_gain']['reason'])
                transform = photometry.build_photometric_transformer(baseline_raw,
                    illumination=variant['illumination'], colorchecker_enabled=variant['colorchecker_enabled'],
                    historical_correction=illumination, colorchecker_correction=color, facies_fit=facies_fit)
                result = transform.apply(baseline_raw)
                baseline, diag = _bounded_rgb(result.rgb, shape)
                branches[key] = dict(transform=transform, variant=variant, baseline=baseline, folder=folder,
                    clear=np.zeros_like(baseline), diagnostics=dict(transform=transform.diagnostics,
                    baseline=dict(result.diagnostics, **diag)), rows=[], summaries=[])
            except ValueError as error:
                failed[key] = _branch_failure('baseline', error)
        del baseline_raw
        for record in ref_inputs:
            update('Correcting clearwater consistently in all six branches')
            raw = decode(record)
            for key, branch in list(branches.items()):
                try:
                    result = branch['transform'].apply(raw)
                    rgb, diag = _bounded_rgb(result.rgb, shape)
                    branch['clear'] += rgb / len(ref_inputs)
                    branch['diagnostics'].setdefault('clearwater', []).append(dict(result.diagnostics, **diag))
                except ValueError as error:
                    failed[key] = _branch_failure('clearwater', error)
                    del branches[key]
            del raw
        for key, branch in list(branches.items()):
            try:
                reference = build_joint_reference(branch['baseline'], branch['clear'], labels, domain_mask=domain)
                if not reference.usable:
                    raise ValueError('No usable optical span in baseline/clearwater pair')
                branch['reference'] = reference
                path = branch['folder'] / 'references.npz'
                branch['reference_sha'] = _archive(path, baseline=branch['baseline'], clearwater=branch['clear'],
                    facies=labels, pv=pv, pore_volume=pv, denominator=reference.denominators['joint_chromatic_od'],
                    reliability=reference.reliability['joint_chromatic_od'])
                (branch['folder'] / 'frames').mkdir(exist_ok=True)
            except ValueError as error:
                failed[key] = _branch_failure('reference', error)
                del branches[key]
        for ordinal, record in enumerate(old['frames']):
            index = record['frame_index']
            update('Preparing selected optical variants', frame=ordinal+1, frames=len(old['frames']), active_variants=list(branches))
            arrays = compare.read_prepared_arrays(record['arrays'], record['arrays_sha256'])
            for label in ('photo', 'phase'):
                source_ref = record['source_reference']
                if source_ref.get(label) and io.sha(source_ref[label]) != source_ref[label + '_sha256']:
                    raise ValueError('Fixed review source changed during photometry comparison')
            source_record = native['frame_sources'][index]
            raw = decode(source_record)
            for key, branch in list(branches.items()):
                try:
                    result = branch['transform'].apply(raw)
                    parity_key = 'historical_cc_on' if illumination is not None else 'none_cc_on'
                    if ordinal == 0 and key == parity_key:
                        branch['diagnostics']['registered_colorchecker_parity'] = _historical_parity(
                            result.rgb, raw, illumination, color, source_record)
                    rgb, diag = _bounded_rgb(result.rgb, shape)
                    signal = joint_optical_signals(rgb, branch['reference'])['joint_chromatic_od']
                    path = branch['folder'] / 'frames' / f'frame_{index:04d}.npz'
                    checksum = _archive(path, rgb=rgb, signal=signal,
                        **{k: arrays[k] for k in ('av','gv','solubility','density')})
                    row = dict(record, arrays=str(path), arrays_sha256=checksum,
                        raw_photo_source=source_record, photometric_diagnostics=dict(result.diagnostics, **diag))
                    branch['rows'].append(row)
                    branch['summaries'].append(joint.summarize_frame(signal, arrays['av'], arrays['gv'],
                        arrays['solubility'], arrays['density'], run=name, elapsed_h=row['elapsed_h'],
                        injected_g=row['injected_mass_g'], frame_index=index,
                        expected_final_injected_mass_g=old['expected_final_injected_mass_g'],
                        solubility_max_g_l=float(arrays['solubility'].max())))
                except ValueError as error:
                    failed[key] = _branch_failure('frame', error, frame=index)
                    del branches[key]
            del raw, arrays
        for key, branch in branches.items():
            if [row['frame_index'] for row in branch['rows']] != inspected['selected_frame_indices']:
                raise ValueError('Complete variant frame coverage is required before fitting')
            raw_bindings = {str(resolve_review_path(Path(row['source_path']))): row['source_sha256']
                            for row in [native['reference'], *native['frame_sources']]}
            raw_bindings.update({r['path']:r['sha256'] for r in ref_inputs})
            binding = dict(old['binding'], format=VERSION, photometry_variant=branch['variant'],
                numerical_sources=request['numerical_sources'], reference_inputs=ref_inputs,
                physical_preparation_sha256=io.sha(old_path), optical_binding=optical)
            manifest = dict(old, format=VERSION, complete=True, binding=binding, binding_sha256=compare.digest(binding),
                prepared_path=str(branch['folder']/'prepared.json'), references=str(branch['folder']/'references.npz'),
                references_sha256=branch['reference_sha'], reference_inputs=ref_inputs,
                photometry_variant=branch['variant'], frames=branch['rows'], summaries=branch['summaries'],
                source_phase_correction_contract=inspected['correction_contract'],
                current_correction_contract=_concentration_correction_contract(inspected, branch, request),
                optical_binding=dict(optical, diagnostics=branch['diagnostics']),
                reference_diagnostics=branch['reference'].diagnostics,
                photometric_policy='New raw-derived complete branch, consistent baseline/clearwater/current; fixed recorded phase maps',
                source_bindings=raw_bindings,
                color_limit_g_l=max(row['solubility_max_g_l'] for row in branch['summaries']))
            if assumption:
                manifest['clearwater_reference_assumption'] = assumption
            else:
                manifest.pop('clearwater_reference_assumption', None)
            io.write(branch['folder']/'prepared.json', manifest)
            reports[key] = str(branch['folder']/'prepared.json')
    for key, error in failed.items():
        io.write(root/key/'preparation'/name/'rejected.json', error)
    _verify_physical_profile(request)
    receipt = dict(run=name, complete=True, accepted=reports, rejected=failed)
    io.write(output/'receipt.json', receipt)
    update('complete', accepted=list(reports), rejected=failed)
    return receipt


def _fit_worker(task, request, names):
    # The common process dispatcher tracks tasks by args[0]['run']; for fitting
    # that identity is the variant, independently of the run being optimized.
    variant = task['variant']
    root = Path(request['output']) / variant['photometry_variant_id']
    summaries = {}
    for name in names:
        prepared = io.read(root/'preparation'/name/'prepared.json')
        if (prepared.get('complete') is not True or prepared.get('photometry_variant') != variant
                or len(prepared.get('frames', [])) != request['expected_frame_counts'][name]
                or len(prepared.get('summaries', [])) != request['expected_frame_counts'][name]):
            raise ValueError('Incomplete or mismatched variant cohort; partial-series fits are forbidden')
        summaries[name] = prepared['summaries']
    io.write(root/'summaries.json', summaries)
    update = common._child_update(Path(request['output']), 'fit', variant['photometry_variant_id'])
    tick = time.perf_counter()
    fits, validation = compare.fit_and_validate(summaries, root, update, tuple(request['gas_bounds']),
                                               request['optimization_settings'])
    result = dict(variant=variant, complete=True, elapsed_seconds=time.perf_counter()-tick,
        shared_training=fits['shared']['metrics'] if 'metrics' in fits['shared'] else compare._metrics(
            [row for rows in fits['shared']['predictions'].values() for row in rows]),
        whole_run_holdout=compare._metrics([row for fold in validation['whole_run_holdout'].values() for row in fold['rows']]))
    io.write(root/'fit_receipt.json', result)
    update('complete', **result)
    return result


def run(request_path):
    start = time.perf_counter()
    common._limit_worker_threads(1)
    request_path = Path(request_path).resolve()
    request = io.read(request_path)
    root = Path(request['output']).resolve()
    def update(stage, **extra):
        io.write(request_path.with_name('status.json'), dict(stage=stage,
            updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start, **extra))
    try:
        catalog = inputs.discover_model_inputs(request['model_root'])
        if catalog['model_id'] != request['model_id'] or not root.is_relative_to(Path(catalog['results_root']).resolve()):
            raise ValueError('Wrong phase model or series output root')
        if request.get('overflow_mode','off') != 'off':
            raise ValueError('No measured overflow constraint declared')
        dataset = catalog.get('dataset') or request.get('dataset', 'ffum')
        if request.get('dataset') and catalog.get('dataset') and request['dataset'] != dataset:
            raise ValueError('Requested series differs from the registered phase model')
        request['dataset'] = dataset
        verify_policy_snapshot(request.get('photometry_policy'), dataset, catalog['results_root'])
        request['selected_variants'] = list(_selected_variants(request, dataset))
        names = request['runs']
        available = {job['run']:dict(job) for job in catalog['jobs']}
        if (not names or len(set(names)) != len(names)
                or any(name not in available for name in names)):
            raise ValueError('Select unique registered experiments')
        if (type(request['workers']) is not int or request['workers'] < 1
                or type(request['analysis_width']) is not int or request['analysis_width'] < 1):
            raise ValueError('Worker count and analysis width must be positive integers')
        compare.validate_clearwater_assumptions(request, available, names)
        jobs = [available[name] for name in names]
        descriptions = [compare.configuration.describe_run(
            dict(job, concentration_profile_path=request['concentration_profile_path']), purpose='joint_reference')
            for job in jobs]
        groups = {description['chemistry_group'] for description in descriptions}
        if (len(groups) != 1 or groups & {'uncertainty', 'provisional', 'unconfirmed'}
                or any(not d['ready'] or not d['shared_fit_eligible'] for d in descriptions)):
            raise ValueError('One available, shared-fit-eligible chemistry group is required')
        inspected = {job['run']:inputs.inspect_run(job) for job in jobs}
        excluded_frames = common.select_frames(request, inspected)
        actual = {name:len(meta['selected_frame_indices']) for name,meta in inspected.items()}
        if 'expected_frame_counts' not in request:
            # Freeze the discovered coverage before preparing any branch.
            # An explicit image-plan receipt may select fewer than all frames.
            request['expected_frame_counts'] = actual
            request['frame_coverage_policy'] = (
                'frozen_image_selection_receipt'
                if request.get('image_selection_receipt_path') else 'all_registered_review_images')
        if set(request['expected_frame_counts']) != set(names):
            raise ValueError('Expected frame counts must cover exactly the requested experiments')
        if actual != request['expected_frame_counts']:
            raise ValueError('Expected experiment/frame selection changed')
        request['facies_gain_settings'] = asdict(_facies_gain_settings(request))
        request['colorchecker_work_dtype'] = os.environ.get('FFAC_COLORCHECKER_WORK_DTYPE', 'float64')
        if not request.get('physical_prepared_root') and 'physical_profile_file_sha256' not in request:
            request['physical_profile_file_sha256'] = io.sha(request['concentration_profile_path'])
        _verify_physical_profile(request)
        _require_fresh_output(root)
        source_binding = numerical_sources()
        request['numerical_sources'] = source_binding
        root.mkdir(parents=True,exist_ok=True)
        binding_path = root/'request_binding.json'
        if binding_path.exists() and io.read(binding_path) != request:
            raise ValueError('Immutable comparison request changed')
        io.write(binding_path,request)
        if request.get('image_selection_receipt_path'):
            io.write(root/'selection.json',dict(
                requested_runs=names, selected_frame_counts=actual,
                excluded_frames=excluded_frames,
                image_selection_unused_runs=request['image_selection_unused_runs'],
                image_selection_receipt_path=request['image_selection_receipt_path'],
                image_selection_receipt_sha256=request['image_selection_receipt_sha256']))
        io.write(root/'input_catalog.json',dict(catalog=catalog,inspected=inspected))
        for variant in request['selected_variants']:
            folder=root/variant['photometry_variant_id']
            io.write(folder/'request_binding.json',dict(request, photometry_variant=variant))
        timings={}
        tick=time.perf_counter()
        receipts=common._parallel_jobs(_prepare_worker,[(job,inspected[job['run']],request) for job in jobs],
            workers=request['workers'],threads=1,output=root,stage='prepare',update=update)
        timings['preparation_seconds']=time.perf_counter()-tick
        _verify_physical_profile(request)
        accepted=[variant for variant in request['selected_variants']
            if all(variant['photometry_variant_id'] in r['accepted'] for r in receipts)]
        rejected={variant['photometry_variant_id']:{r['run']:r['rejected'][variant['photometry_variant_id']]
            for r in receipts if variant['photometry_variant_id'] in r['rejected']}
            for variant in request['selected_variants'] if variant not in accepted}
        io.write(root/'coverage.json',dict(receipts=receipts,accepted=accepted,rejected=rejected,
            expected_frames=sum(actual.values()),runs=names,partial_series_fits_allowed=False))
        if not accepted:
            raise ValueError('Every optical branch was rejected; see coverage.json')
        tick=time.perf_counter()
        fits=common._parallel_jobs(_fit_worker,[(dict(run=v['photometry_variant_id'], variant=v),request,names) for v in accepted],workers=min(6,request['workers']),
            threads=1,output=root,stage='fit',update=update)
        timings['fit_validation_wall_seconds']=time.perf_counter()-tick
        io.write(root/'fit_comparison.json',dict(variants=fits,rejected=rejected))
        _verify_physical_profile(request)
        update('Writing phone-readable comparison figures')
        from darsia.fluidflower.tools.phase_studio_photometry_report import build_report
        tick=time.perf_counter()
        build_report(root)
        timings['comparison_report_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        deliveries=[]
        for variant in accepted:
            key=variant['photometry_variant_id']
            update('Publishing concentration comparison',variant=key)
            deliveries.extend(common._parallel_jobs(compare._render_worker,
                [(job,inspected[job['run']],str(root/key),catalog,dict(request,photometry_variant=variant)) for job in jobs],
                workers=request['workers'],threads=1,output=root/key,stage='render',update=update))
            _verify_physical_profile(request)
        timings['render_wall_seconds']=time.perf_counter()-tick
        if numerical_sources()!=source_binding:
            raise ValueError('Numerical code changed during comparison')
        timings['elapsed_seconds']=time.perf_counter()-start
        io.write(root/'delivery.json',dict(complete=True,accepted=accepted,rejected=rejected,deliveries=deliveries,timings=timings))
        io.write(request_path.with_name('timing.json'),timings)
        update('complete',accepted_variants=len(accepted),rejected=rejected,timing=timings)
    except Exception:
        update('failed',error=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    run(parser.parse_args().request)
