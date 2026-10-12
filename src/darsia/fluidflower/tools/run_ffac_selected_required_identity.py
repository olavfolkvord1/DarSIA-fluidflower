"""Required identity from own historical inputs; no Policy A execution."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

from darsia.fluidflower.phase_ffac_selected_phase import asset, bound_json
from darsia.fluidflower.phase_ffac_rig_camera_runtime import bound_path
from darsia.fluidflower.phase_ffac_selected_cpu_identity import array_identity
from darsia.fluidflower.phase_ffac_selected_required_identity import RequiredIdentityEvaluator, CONTRACT, require_frozen_gas_audit
from darsia.fluidflower.tools.refit_ffac_g3_downstream import atomic_json, canonical_sha256, utc_now, worker_initializer
from darsia.fluidflower.tools import run_ffac_selected_cpu_identity as original
from darsia.fluidflower.tools import run_ffac_selected_phase_corrected as protected
from darsia.fluidflower.tools import run_ffac_rig_camera_inference as rig

SPEC_SHA256 = '46060237a9b158d3fcddcd96dc1c4ff12b0b3b57611c0ff9523117d22030204b'
MODULES = original.MODULES + ('darsia.fluidflower.phase_ffac_selected_required_identity',
    'darsia.fluidflower.tools.run_ffac_selected_required_identity')


def inherited_stage_convention(candidate, run, historical_validation_source):
    candidate = Path(candidate)
    manifest_descriptor = asset(candidate/'validation/run_packages'/run/'postflight_manifest.json')
    manifest = bound_json(manifest_descriptor)
    source_hash = manifest['inference_provenance']['registered_v5_sources']['phase_validation.py']
    validation_descriptor = asset(historical_validation_source)
    if validation_descriptor['sha256'] != source_hash:
        raise ValueError('Original inherited-stage source differs from its producer receipt')
    entry = manifest['physics_profile']['runs']['ffac/'+run]
    context = {'path': str(candidate/entry['path']), 'sha256': entry['sha256']}
    with np.load(bound_path(context), allow_pickle=False) as archive:
        x, y = np.asarray(archive['x_m'], np.float64), np.asarray(archive['y_m'], np.float64)
    spacing = [abs(float(y[1]-y[0])), abs(float(x[1]-x[0]))]
    return {'spacing_m': spacing, 'context': context, 'postflight_manifest': manifest_descriptor,
        'historical_validation_source': validation_descriptor,
        'spacing_source': 'original OpticalRegionSequence context.y_m/x_m first differences',
        'native_baseline_mask': None, 'native_previous_mask': None,
        'extended_baseline_previous_masks': 'retained original source-bound masks',
        'no_g3_spacing_substitution': True}


def bind(source_plan, identity_inventory, root, frozen_gas_audit, historical_validation_source):
    root = Path(root)
    source = bound_json(asset(source_plan))
    audit_descriptor = asset(frozen_gas_audit)
    audit = bound_json(audit_descriptor)
    require_frozen_gas_audit(audit)
    with patch.object(original, 'SPEC_SHA256', SPEC_SHA256):
        base = original.bind(source_plan, identity_inventory, root/'base_binding')
    plan = bound_json(base)
    conventions = {run: inherited_stage_convention(source['optical_candidate'], run, historical_validation_source)
                   for run in sorted(source['runs'])}
    binding = dict(plan['phase_campaign_binding'])
    binding.update(contract=CONTRACT, spec_sha256=SPEC_SHA256, frozen_gas_audit=audit_descriptor,
        inherited_stage_conventions=conventions, parent_identity_binding=base,
        stage_conventions_independently_bound=True, fresh_inherited_map_required=True,
        archived_inherited_labels_used_as_inference_input=False,
        required_identity_products=['final_labels', 'g3', 'regional_frozen', 'regional_refit', 'co2',
            'photo_sand_domain', 'feature_support', 'regional_native_domain', 'flatfield_valid',
            'feature_eligible', 'optical_presence_ge_half', 'frozen_unresolved',
            'inherited_co2_union', 'inherited_water', 'inherited_unresolved'],
        diagnostic_only_products=['frozen_gas'],
        code={name: {'sha256': asset(importlib.import_module(name).__file__)['sha256']} for name in MODULES})
    plan.update(phase_campaign_binding=binding, pipeline_sha256=canonical_sha256(binding), spec_sha256=SPEC_SHA256)
    path = root/'identity_plan.json'
    if path.exists():
        raise FileExistsError('Required identity binding already exists')
    atomic_json(path, plan)
    return asset(path)


def _publish_stop(root, record):
    try:
        with (Path(root)/'identity_stop.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        pass


def run_sequence(plan_descriptor, root, run, through_frame):
    worker_initializer(1)
    plan = bound_json(plan_descriptor)
    if plan['phase_campaign_binding']['spec_sha256'] != SPEC_SHA256:
        raise ValueError('Required-identity specification changed')
    protected.verify_code(plan)
    root = Path(root)
    evaluator = RequiredIdentityEvaluator(root, plan, run)
    rows = sorted(evaluator.inventory['sources'], key=lambda row: row['studio_frame_index'])
    rows = [row for row in rows if row['studio_frame_index'] <= through_frame]
    if [row['studio_frame_index'] for row in rows] != list(range(through_frame+1)):
        raise ValueError('Original Studio own-history sequence has a missing frame')
    receipts = []
    for row in rows:
        if (root/'identity_stop.json').exists():
            return {'run': run, 'complete': False, 'reason': 'shared identity stop', 'receipts': receipts}
        index, digest = row['studio_frame_index'], row['source_sha256']
        receipt_path = root/'identity_frames'/run/f'frame_{index:03d}.json'
        if receipt_path.exists():
            previous = bound_json(asset(receipt_path))
            if (previous['all_pass'] is not True or previous['identity_plan'] != plan_descriptor
                    or previous['source_sha256'] != digest
                    or previous['previous_source_sha256'] != row['previous_source_sha256']):
                raise FileExistsError('Failed or differently bound identity receipt cannot be reused')
            bound_path(previous['phase_receipt'])
            for descriptor in previous['generated_probability_files'].values():
                bound_path(descriptor)
            bound_path(previous['generated_inherited_map'])
            receipts.append(asset(receipt_path))
            print(json.dumps({'event': 'verified_required_identity_reused', 'run': run,
                'studio_frame_index': index, 'receipt': receipts[-1]}), flush=True)
            continue
        started = time.perf_counter()
        result = evaluator.evaluate(digest)
        parent = bound_json(row['parent_receipt'])
        with Image.open(bound_path(parent['labels_map'])) as image:
            expected = np.asarray(image).copy()
        labels = array_identity(result['labels'], expected)
        probabilities = evaluator.compare_saved_probabilities(row)
        capture = evaluator.centre.capture
        phase = evaluator.save(result)
        generated = original._save_arrays(root/'identity_probability_layers'/run/f'frame_{index:03d}', capture['stored'])
        buffer = io.BytesIO(); np.save(buffer, capture['inherited_recomputed'], allow_pickle=False)
        inherited_output = rig._write_bytes(root/'identity_inherited_maps'/run/f'frame_{index:03d}.npy', buffer.getvalue())
        first = next((name for name, check in probabilities['comparisons'].items()
                      if name != 'extra_masks.packbits.npz' and not check['bit_identical']), None)
        if first is None:
            first = next((name for name, check in probabilities['extra_masks']['comparisons'].items()
                          if check['required'] and not check['bit_identical']), None)
        if first is None:
            first = next(('inherited_'+name for name, check in probabilities['inherited_map']['comparisons'].items()
                          if not check['bit_identical']), None)
        all_pass = labels['bit_identical'] and probabilities['all_bit_identical']
        record = {'contract': CONTRACT, 'complete': True, 'created_utc': utc_now(), 'run': run,
            'original_studio_frame_index': index, 'source_work_index': row['work_index'], 'elapsed_h': row['elapsed_h'],
            'source_sha256': digest, 'previous_source_sha256': row['previous_source_sha256'],
            'identity_plan': plan_descriptor, 'phase_receipt': phase, 'labels': labels,
            'stored_probabilities': probabilities, 'generated_probability_files': generated,
            'generated_inherited_map': inherited_output, 'source_inputs': capture['source_inputs'],
            'stage_conventions': result['native_inference']['historical_stage_conventions'],
            'trace_arrays': capture['trace_arrays'], 'first_differing_required_product': first,
            'frozen_gas': probabilities['extra_masks']['comparisons']['frozen_gas'],
            'cpu_only': True, 'internal_threads': 1, 'all_pass': bool(all_pass),
            'training': False, 'annotations_read': False, 'mass_computed': False,
            'policy_a_production_allowed': False, 'seconds': time.perf_counter()-started}
        atomic_json(receipt_path, record); descriptor = asset(receipt_path); receipts.append(descriptor)
        print(json.dumps({'event': 'required_cpu_identity_frame', 'run': run, 'studio_frame_index': index,
            'source_work_index': row['work_index'], 'all_pass': all_pass, 'labels_changed': labels['changed_elements'],
            'first_required_difference': first, 'frozen_gas_changed_pixels': record['frozen_gas']['changed_pixels'],
            'seconds': record['seconds'], 'receipt': descriptor}), flush=True)
        if not all_pass:
            _publish_stop(root, {'contract': CONTRACT, 'created_utc': utc_now(), 'stop': True,
                'stop_new_work': True, 'production_allowed': False, 'reason': 'Required CPU own-history identity failed',
                'failed_frame': descriptor, 'run': run, 'original_studio_frame_index': index,
                'first_differing_required_product': first, 'labels_changed': labels['changed_elements']})
            return {'run': run, 'complete': False, 'stopped': True, 'receipts': receipts, 'failed_frame': descriptor}
    return {'run': run, 'complete': True, 'receipts': receipts}


def run_stage(plan, root, stage, workers):
    root = Path(root)
    if (root/'identity_stop.json').exists():
        raise RuntimeError('Required identity stop is immutable; no retry')
    if stage == 'ac19':
        tasks = [('ac19', 1)]
    elif stage == 'full-runs':
        if bound_json(asset(root/'ac19_stage.json'))['complete'] is not True:
            raise ValueError('AC19 own-history identity must pass first')
        inventory = bound_json(bound_json(plan)['phase_campaign_binding']['identity_inventory'])
        tasks = [(run, max(row['studio_frame_index'] for row in inventory['runs'][run]['frames']))
                 for run in ('ac18', 'ac52')]
    else:
        if bound_json(asset(root/'full-runs_stage.json'))['complete'] is not True:
            raise ValueError('Both complete AC18 and AC52 histories must pass before all39')
        tasks = [(run, 2) for run in sorted(bound_json(plan)['runs'])]
    output = root/(stage+'_stage.json')
    if output.exists():
        raise FileExistsError('Required identity stage already exists')
    results = []
    if workers == 1 or len(tasks) == 1:
        for run, last in tasks:
            result = run_sequence(plan, root, run, last); results.append(result)
            if not result['complete']: break
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = [pool.submit(run_sequence, plan, str(root), run, last) for run, last in tasks]
            for future in as_completed(pending):
                result = future.result(); results.append(result)
                if not result['complete']:
                    for task in pending: task.cancel()
                    break
    complete = len(results) == len(tasks) and all(row['complete'] for row in results)
    record = {'contract': CONTRACT, 'stage': stage, 'created_utc': utc_now(), 'identity_plan': plan,
        'complete': complete, 'all_39_complete': complete and stage == 'all39',
        'required_identity_passed': complete, 'cpu_only': True, 'internal_threads': 1,
        'computed_inherited_identity_required': True, 'frozen_gas_diagnostic_only': True,
        'results': results, 'training': False, 'mass_computed': False, 'policy_a_production_allowed': False}
    atomic_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('bind', 'ac19', 'full-runs', 'all39'), required=True)
    parser.add_argument('--root', required=True); parser.add_argument('--plan')
    parser.add_argument('--source-plan'); parser.add_argument('--identity-inventory')
    parser.add_argument('--frozen-gas-audit'); parser.add_argument('--historical-validation-source')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.stage == 'bind':
        print(json.dumps(bind(args.source_plan, args.identity_inventory, args.root,
            args.frozen_gas_audit, args.historical_validation_source))); return 0
    if not 1 <= args.workers <= 12:
        raise ValueError('Main-machine CPU workers must be bounded 1..12')
    record = run_stage(asset(args.plan), args.root, args.stage, args.workers)
    print(json.dumps(record), flush=True)
    return 0 if record['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
