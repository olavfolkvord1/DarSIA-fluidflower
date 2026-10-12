"""Read saved CPU-identity evidence only; never execute inference or retry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_json(descriptor):
    path = Path(descriptor['path'])
    if digest(path) != descriptor['sha256']:
        raise ValueError(f'Changed saved evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def descriptor(path):
    return {'path': str(path), 'sha256': digest(path)}


def report(root, destination, local_copy):
    root, destination, local_copy = map(Path, (root, destination, local_copy))
    if destination.exists() or local_copy.exists():
        raise FileExistsError('Report and local copy must both be new')
    stage = json.loads((root/'pilot_stage.json').read_text(encoding='utf-8-sig'))
    stop = json.loads((root/'identity_stop.json').read_text(encoding='utf-8-sig'))
    failed = bound_json(stop['failed_frame'])
    plan = bound_json(failed['identity_plan'])
    inventory = bound_json(plan['phase_campaign_binding']['identity_inventory'])
    original = next(row for row in inventory['runs'][failed['run']]['frames']
                    if row['studio_frame_index'] == failed['original_studio_frame_index'])
    evidence_rows = [bound_json(item) for result in stage['results'] for item in result['receipts']]
    probabilities = []
    for name, check in failed['stored_probabilities']['comparisons'].items():
        row = {'array': name, 'bit_identical': check['bit_identical'],
               'changed_elements': check['changed_elements'], 'shape': check['actual_shape'],
               'dtype': check['actual_dtype'], 'first_different_index': check['first_different_index']}
        if name.startswith('regional_'):
            actual_descriptor = failed['generated_probability_files'][name]
            for item in (actual_descriptor, check['source']):
                if digest(item['path']) != item['sha256']:
                    raise ValueError(f'Changed saved probability array: {item["path"]}')
            actual = np.load(actual_descriptor['path'], allow_pickle=False)
            expected = np.load(check['source']['path'], allow_pickle=False)
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            row['changed_share_percent'] = 100 * check['changed_elements']/actual.size
            row['maximum_absolute_probability_difference'] = float(np.max(np.abs(delta)))
            where = tuple(check['first_different_index'])
            row['first_actual_value'] = float(actual[where])
            row['first_expected_value'] = float(expected[where])
            row['first_actual_float16_bits_hex'] = f'{int(actual.view(np.uint16)[where]):04x}'
            row['first_expected_float16_bits_hex'] = f'{int(expected.view(np.uint16)[where]):04x}'
        probabilities.append(row)
    record = {
        'contract': 'ffac-selected-cpu-own-history-stop-diagnosis-v1',
        'saved_outputs_only': True, 'inference_executed_by_report': False,
        'identity_plan': failed['identity_plan'], 'pilot_stage': descriptor(root/'pilot_stage.json'),
        'stop': descriptor(root/'identity_stop.json'), 'failed_frame': stop['failed_frame'],
        'run': failed['run'], 'original_studio_frame_index': failed['original_studio_frame_index'],
        'source_work_index': failed['source_work_index'], 'elapsed_h': failed['elapsed_h'],
        'current_source_sha256': failed['source_sha256'],
        'original_previous_source_sha256': original['original_previous_source_sha256'],
        'actual_previous_source_sha256': failed['previous_source_sha256'],
        'own_history_predecessor_matches': original['original_previous_source_sha256'] == failed['previous_source_sha256'],
        'compared_frame_count': len(evidence_rows),
        'all_computed_final_labels_bit_identical': all(row['labels']['bit_identical'] for row in evidence_rows),
        'passed_complete_frames': sum(row['all_pass'] for row in evidence_rows),
        'failed_complete_frames': sum(not row['all_pass'] for row in evidence_rows),
        'ac18_executed': any(row['run'] == 'ac18' for row in evidence_rows),
        'all39_executed': False, 'policy_a_dispatched': False,
        'probability_comparisons': probabilities,
        'first_compared_stored_output_difference': failed['first_differing_stored_array'],
        'known_input_conventions_in_operator_order': sorted(failed['known_input_differences'],
            key=lambda row: 0 if row['step'] == 'previous temporal validity footprint' else 1),
        'ordering_note': 'Feature-support and temporal-alignment validity are consumed before regional preparation. The immutable frame receipt lists spacing first; this report distinguishes recorded-list order from operator order.',
        'causal_limit': 'Only the input-convention differences and stored output differences are verified. The historical full-float intermediate arrays were not all archived; no isolated causal inference or counterfactual execution was performed.',
        'current_native_binding': original['current_native'],
        'baseline_native_binding': original['baseline_native'],
        'new_trace_arrays': failed['trace_arrays'],
        'training': False, 'mass_computed': False, 'retry_executed': False,
        'production_allowed': False,
    }
    destination.mkdir(parents=True)
    (destination/'diagnosis.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    with (destination/'probability_comparisons.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['array','bit_identical','changed_elements','shape','dtype',
            'first_different_index','changed_share_percent','maximum_absolute_probability_difference',
            'first_actual_value','first_expected_value','first_actual_float16_bits_hex','first_expected_float16_bits_hex'])
        writer.writeheader(); writer.writerows(probabilities)
    lines = [
        'CPU own-history identity: STOP',
        'AC19 Studio frame 0 passed final labels and all five encoded probability arrays.',
        'AC19 Studio frame 1 (source work index 2; actual camera time 0.2472175 h) has identical final labels, compact G3, CO2 packbits and all eight extra masks.',
        'The first compared stored output difference is regional_frozen.npy; regional_refit.npy also differs.',
    ]
    for row in probabilities:
        if row['array'].startswith('regional_'):
            lines.append(f"{row['array']}: {row['changed_elements']:,}/441,000 differing float16 cells ({row['changed_share_percent']:.6f}%); first cell {row['first_different_index']}, new={row['first_actual_value']}, original={row['first_expected_value']}; maximum absolute probability difference={row['maximum_absolute_probability_difference']}.")
    lines += [
        'Earliest known operator-convention difference: previous temporal validity footprint (original approximately 0.999477, replay 0.842254). The RGB warp is direct: this validity mask does not normalize the colour interpolation, and packed support masks are identical.',
        'Another verified convention difference: exact historical spacing [0.00016448472641827916, 0.00016447368421052634] versus replay [0.00016448, 0.00016448]. Regional preparation consumes spacing; an isolated cause has not been established.',
        'Original predecessor is the baseline in both runs. No selected 0.125 h predecessor was used.',
        'Two frames were compared: one full pass and one probability-identity failure. Both computed final labels are bit-identical. The stage-level false labels flag denotes incomplete pilot acceptance; it does not denote a categorical mismatch in either compared frame.',
        'AC18, all39 identity, Policy A, training and mass computation were not started. No convention was changed or retried after the stop.',
    ]
    (destination/'README.txt').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    sources = {'identity_stop.json': root/'identity_stop.json', 'pilot_stage.json': root/'pilot_stage.json',
        'ac19_frame_000.json': root/'identity_frames/ac19/frame_000.json',
        'ac19_frame_001.json': root/'identity_frames/ac19/frame_001.json'}
    for name, path in sources.items():
        shutil.copyfile(path, destination/name)
        if digest(path) != digest(destination/name): raise ValueError('Evidence copy mismatch')
    local_copy.mkdir(parents=True)
    files = {}
    for path in sorted(destination.iterdir()):
        shutil.copyfile(path, local_copy/path.name)
        if digest(path) != digest(local_copy/path.name): raise ValueError('Local copy mismatch')
        files[path.name] = {'source': descriptor(path), 'local': descriptor(local_copy/path.name)}
    (local_copy/'copy_receipt.json').write_text(json.dumps({'files': files, 'overwrite': False,
        'all_copy_sha256_equal': True}, indent=2)+'\n', encoding='utf-8')
    return {'report': descriptor(destination/'diagnosis.json'), 'local_readme': str(local_copy/'README.txt'),
            'local_diagnosis': descriptor(local_copy/'diagnosis.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--destination', required=True)
    parser.add_argument('--local-copy', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.root, args.destination, args.local_copy)))


if __name__ == '__main__': main()
