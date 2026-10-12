"""Publish one ordinary Studio candidate covering an explicit complete run set.

Keep the trained weights/evidence frozen; additional runs supply only their own
registered images, baselines and physical geometry. They are not training data.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

from darsia.fluidflower import phase_optical_regions_runtime as runtime
from darsia.fluidflower.phase_plume import freeze_context
from darsia.fluidflower.project import sha256_file, utc_now
from darsia.fluidflower.tools.prepare_optical_region_studio import read, write
from darsia.fluidflower.tools.phase_transfer_context import prepared_target_physics
from darsia.fluidflower.tools.train_phase_model_further import native_source


def require_run_set(jobs, expected):
    actual = [j['run'] for j in jobs]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f'Incomplete review scope: expected {sorted(expected)}, found {sorted(actual)}')


def extend(parent, targets, output, expected):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or output == parent or output.is_relative_to(parent):
        raise ValueError('Use a new, separate model directory')
    profile = runtime.load_profile(parent)
    if profile is None:
        raise ValueError('A fitted optical-region parent is required')
    profile = {k: v for k, v in profile.items() if not k.startswith('_')}
    jobs = read(parent / 'review_source/postflight_manifest.json')['jobs']
    datasets = {j['dataset'] for j in jobs}
    if len(datasets) != 1:
        raise ValueError('Select one experimental series')
    dataset = next(iter(datasets))
    prepared = []
    for target_root in map(Path, targets):
        manifest = target_root / 'postflight_manifest.json'
        rows = read(manifest)['jobs']
        if len(rows) != 1 or rows[0]['dataset'] != dataset:
            raise ValueError('Each prepared target must identify one run of the same series')
        job = rows[0]
        _, native_path, _ = native_source(job)
        target = dict(dataset=dataset, run=job['run'], job=job, source_manifest=str(manifest),
                      native_manifest=str(native_path), native_manifest_sha256=sha256_file(native_path))
        checked = prepared_target_physics(parent, target)
        prepared.append((target, checked))
        jobs.append(copy.deepcopy(job))
    require_run_set(jobs, expected)
    # All descriptors and the full intended scope validate before any output.
    output.mkdir(parents=True)
    shutil.copytree(parent / 'models', output / 'models')
    for name in ('regional_model.json', 'evidence.json', 'extraction.json',
                 'regional_training_protocol.json', 'review_retractions.json', 'region_inventory.csv'):
        if (parent / name).is_file():
            shutil.copy2(parent / name, output / name)
    protected = {p.relative_to(parent).as_posix(): sha256_file(p)
                 for p in (parent / 'models').rglob('*') if p.is_file()}
    protected['regional_model.json'] = sha256_file(parent / 'regional_model.json')
    for key, entry in profile['runs'].items():
        source = parent / entry['path']
        if sha256_file(source) != entry['sha256']:
            raise ValueError('Parent geometry changed')
        destination = output / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for target, checked in prepared:
        for path, digest in checked['source_sha256'].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError('Prepared target changed before context registration')
        profile['runs'][checked['key']] = freeze_context(output, dataset, target['run'], checked['context'],
            native_manifest=Path(target['native_manifest']), provenance=checked['provenance'])
    write(output / runtime.PROFILE_FILE, profile)
    evidence = read(parent / 'evidence.json')
    training_runs = sorted({r['run'] for r in evidence['regions']})
    for job in jobs:
        job['phase_model_only'] = True
        job['phase_support_format'] = runtime.VERSION
        job['quality_flags'] = dict(dataset=dataset, training_seen=job['run'] in training_runs,
            human_review_required=True, independent_validation=False, independent_holdout=False,
            evidence_role='trained_optical_region_candidate', review_mode='visible')
        write(output / 'review_source' / job['run'] / 'original_job.json', job)
        write(output / 'review_source' / job['run'] / 'postflight_manifest.json', dict(jobs=[job]))
    jobs.sort(key=lambda j: j['run'])
    write(output / 'review_source/postflight_manifest.json', dict(jobs=jobs))
    write(output / 'expected_review_runs.json', dict(dataset=dataset, runs=sorted(expected)))
    request = read(parent / 'training_request.json')
    request.update(model_id=output.name, output_root=str(output), created_utc=utc_now())
    request['settings'].update(parent_model=str(parent),
        iteration=f'{dataset.upper()} · Full aqueous corrections · All {len(expected)} runs',
        expected_review_runs=sorted(expected))
    if request.get('review_retractions') and (output / 'review_retractions.json').is_file():
        request['review_retractions']['path'] = str(output / 'review_retractions.json')
    write(output / 'training_request.json', request)
    summary = read(parent / 'summary.json')
    summary.update(model_id=output.name, campaign_id=output.name, created_utc=utc_now(),
        runs=sorted(expected), review_runs=sorted(expected), training_runs=training_runs,
        inherited_optical_model_id=parent.name, human_validation=None, deployment=None,
        coverage_extension=dict(parent=str(parent), trained_weights_unchanged=True,
            training_runs=training_runs, review_runs=sorted(expected), new_training_performed=False))
    summary['inventory'].update(runs=sorted(expected), training_runs=training_runs,
                                review_runs=sorted(expected))
    write(output / 'summary.json', summary)
    for relative, digest in protected.items():
        if sha256_file(parent / relative) != digest or sha256_file(output / relative) != digest:
            raise ValueError('Extending the run set changed trained weights')
    write(output / 'scope_receipt.json', dict(expected_runs=sorted(expected), training_runs=training_runs,
        added_runs=[t['run'] for t, _ in prepared], protected_weights=protected, created_utc=utc_now()))
    from darsia.fluidflower.phase_contributions import publish_regional_inventory
    publish_regional_inventory(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--targets', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-runs', nargs='+', required=True)
    args = parser.parse_args()
    extend(args.parent, args.targets, args.output, args.expected_runs)
